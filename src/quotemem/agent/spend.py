"""Daily spend and call caps for the agent profile's provider calls.

A real agent's memory runs unattended for months, so its owner sets what one day
may cost. Configuration comes from the environment:

- ``QUOTEMEM_DAILY_BUDGET_USD``: the most one UTC day may spend. It needs the
  prices of the configured models, in US dollars per million tokens:
  ``QUOTEMEM_CHAT_PRICE_INPUT_PER_MTOK``, ``QUOTEMEM_CHAT_PRICE_OUTPUT_PER_MTOK``
  and ``QUOTEMEM_EMBED_PRICE_PER_MTOK``;
- ``QUOTEMEM_DAILY_CHAT_CALLS`` and ``QUOTEMEM_DAILY_EMBED_CALLS``: the most calls
  one role may make in one UTC day.

Spend is the existing per-call accounting: the token usage each reply reports
(``ProviderRoutes`` estimates a missing embedding usage) times the configured
prices. The day's totals are kept in ``<data-dir>/spend.json`` (mode 0600), so a
restart does not reset them.

When the budget or a call cap is reached the day is *exhausted*:

- chat calls are refused locally, without a network request, as a 429 the core
  treats like any provider pause: an Add in flight waits durably and resumes
  (with only the embedding call cap reached, embeddings are refused instead);
- the gateway answers a new Add 429 ``spend_cap_reached`` with ``Retry-After`` at
  the next UTC midnight, so a plugin keeps it queued (``status: deferred``);
- Search keeps working on stored data: its optional model judges are skipped
  (they fail soft) and only the query embedding is sent. The embedding call cap,
  when set, also stops that embedding; Search then answers 429 until the reset.
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, cast

from quotemem.core.errors import QuoteMemError
from quotemem.core.platform_files import open_private_file

type Role = Literal["chat", "embedding"]

DAY_MS = 86_400_000
SPEND_FILE = "spend.json"
REFUSAL_CODE = "spend_cap_reached"


def _number(environment: Mapping[str, str], name: str) -> float | None:
    raw = environment.get(name, "").strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        raise QuoteMemError("schema_invalid", f"{name} must be a number") from None
    if not math.isfinite(value) or value < 0:
        raise QuoteMemError("schema_invalid", f"{name} must be a nonnegative number")
    return value


def _count(environment: Mapping[str, str], name: str) -> int | None:
    raw = environment.get(name, "").strip()
    if not raw:
        return None
    if not raw.isdigit():
        raise QuoteMemError("schema_invalid", f"{name} must be a nonnegative integer")
    return int(raw)


@dataclass(frozen=True)
class SpendLimits:
    """The configured caps and prices; every field None when uncapped.

    Attributes:
        budget_usd: The daily budget in US dollars.
        chat_input_per_mtok: The chat input price per million tokens.
        chat_output_per_mtok: The chat output price per million tokens.
        embed_per_mtok: The embedding price per million tokens.
        chat_calls: The daily chat call cap.
        embed_calls: The daily embedding call cap.
    """

    budget_usd: float | None = None
    chat_input_per_mtok: float = 0.0
    chat_output_per_mtok: float = 0.0
    embed_per_mtok: float = 0.0
    chat_calls: int | None = None
    embed_calls: int | None = None

    @property
    def capped(self) -> bool:
        """Whether any cap is set."""
        return any(value is not None for value in (
            self.budget_usd, self.chat_calls, self.embed_calls,
        ))

    @classmethod
    def from_environment(cls, environment: Mapping[str, str]) -> SpendLimits:
        """Read the caps and prices; a budget requires every price.

        Raises:
            QuoteMemError: ``schema_invalid`` naming a malformed or missing variable.
        """
        budget = _number(environment, "QUOTEMEM_DAILY_BUDGET_USD")
        prices = {
            name: _number(environment, name) for name in (
                "QUOTEMEM_CHAT_PRICE_INPUT_PER_MTOK", "QUOTEMEM_CHAT_PRICE_OUTPUT_PER_MTOK",
                "QUOTEMEM_EMBED_PRICE_PER_MTOK",
            )
        }
        if budget is not None:
            missing = [name for name, value in prices.items() if value is None]
            if missing:
                raise QuoteMemError(
                    "schema_invalid", f"QUOTEMEM_DAILY_BUDGET_USD needs {', '.join(missing)}",
                )
        return cls(
            budget_usd=budget,
            chat_input_per_mtok=prices["QUOTEMEM_CHAT_PRICE_INPUT_PER_MTOK"] or 0.0,
            chat_output_per_mtok=prices["QUOTEMEM_CHAT_PRICE_OUTPUT_PER_MTOK"] or 0.0,
            embed_per_mtok=prices["QUOTEMEM_EMBED_PRICE_PER_MTOK"] or 0.0,
            chat_calls=_count(environment, "QUOTEMEM_DAILY_CHAT_CALLS"),
            embed_calls=_count(environment, "QUOTEMEM_DAILY_EMBED_CALLS"),
        )

    def describe(self) -> dict[str, object]:
        """A JSON-ready view; nothing secret."""
        return {
            "budget_usd": self.budget_usd, "chat_calls": self.chat_calls,
            "embed_calls": self.embed_calls,
            "prices_per_mtok": {
                "chat_input": self.chat_input_per_mtok,
                "chat_output": self.chat_output_per_mtok, "embedding": self.embed_per_mtok,
            },
        }


@dataclass
class _Day:
    day: int
    spent_usd: float = 0.0
    calls: dict[str, int] = field(default_factory=lambda: {"chat": 0, "embedding": 0})
    tokens: dict[str, int] = field(default_factory=lambda: {
        "chat_input": 0, "chat_output": 0, "embedding": 0,
    })
    refused: dict[str, int] = field(default_factory=lambda: {"chat": 0, "embedding": 0})
    deferred_adds: int = 0


def _token_count(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else 0


class SpendGuard:
    """One store's daily spend ledger and the caps it enforces; thread-safe.

    Args:
        limits: The configured caps.
        path: The ledger file, or None to keep the day in memory only (tests).
        clock_ms: The wall clock in epoch milliseconds.
    """

    def __init__(
        self, limits: SpendLimits, path: Path | None, *,
        clock_ms: Callable[[], int] = lambda: time.time_ns() // 1_000_000,
    ) -> None:
        self.limits = limits
        self._path = path
        self._clock_ms = clock_ms
        self._lock = threading.Lock()
        self._day = self._load()

    def _today(self) -> int:
        return self._clock_ms() // DAY_MS

    def _load(self) -> _Day:
        today = self._today()
        path = self._path
        if path is None or not (path.exists() or path.is_symlink()):
            return _Day(today)
        try:
            descriptor = open_private_file(path, os.O_RDONLY)
            with os.fdopen(descriptor, "rb") as stream:
                raw = cast(dict[str, object], json.loads(stream.read()))
        except (OSError, ValueError, QuoteMemError):
            # A damaged ledger starts the day again rather than stopping memory.
            return _Day(today)
        if raw.get("day") != today:
            return _Day(today)
        day = _Day(today)
        spent = raw.get("spent_usd")
        if isinstance(spent, int | float) and math.isfinite(spent) and spent >= 0:
            day.spent_usd = float(spent)
        for name, target in (("calls", day.calls), ("tokens", day.tokens),
                             ("refused", day.refused)):
            stored = raw.get(name)
            if isinstance(stored, dict):
                for key in target:
                    target[key] = _token_count(cast(dict[str, object], stored).get(key))
        day.deferred_adds = _token_count(raw.get("deferred_adds"))
        return day

    def _save(self) -> None:
        path = self._path
        if path is None:
            return
        day = self._day
        document = {
            "format_version": 1, "day": day.day, "spent_usd": day.spent_usd,
            "calls": day.calls, "tokens": day.tokens, "refused": day.refused,
            "deferred_adds": day.deferred_adds,
        }
        staged = path.with_name(path.name + ".tmp")
        try:
            if staged.exists():
                staged.unlink()
            descriptor = open_private_file(staged, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(json.dumps(document, sort_keys=True) + "\n")
            os.replace(staged, path)
        except (OSError, QuoteMemError):
            # The in-memory ledger still enforces the caps for this lifetime.
            pass

    def _roll(self) -> _Day:
        today = self._today()
        if self._day.day != today:
            self._day = _Day(today)
        return self._day

    def _exhausted(self, day: _Day) -> bool:
        limits = self.limits
        return (
            (limits.budget_usd is not None and day.spent_usd >= limits.budget_usd)
            or (limits.chat_calls is not None and day.calls["chat"] >= limits.chat_calls)
        )

    def _embeddings_spent(self, day: _Day) -> bool:
        cap = self.limits.embed_calls
        return cap is not None and day.calls["embedding"] >= cap

    def exhausted(self) -> bool:
        """Whether an Add could not finish today: the budget or either call cap is reached."""
        with self._lock:
            day = self._roll()
            return self._exhausted(day) or self._embeddings_spent(day)

    def retry_after_seconds(self) -> int:
        """Seconds until the next UTC midnight, at least one."""
        now = self._clock_ms()
        return max(1, math.ceil(((now // DAY_MS + 1) * DAY_MS - now) / 1000))

    def admit(self, role: Role) -> bool:
        """Whether one call of ``role`` may be sent now; a refusal is counted.

        Chat stops at the budget or its call cap; embeddings stop only at their
        own call cap, so Search can still embed its query on stored data.
        """
        with self._lock:
            day = self._roll()
            allowed = not (self._exhausted(day) if role == "chat" else self._embeddings_spent(day))
            if not allowed:
                day.refused[role] += 1
                self._save()
            return allowed

    def charge(self, role: Role, usage: object) -> None:
        """Count one sent call and price the usage its reply reported."""
        values = cast(dict[str, object], usage) if isinstance(usage, dict) else {}
        limits = self.limits
        with self._lock:
            day = self._roll()
            day.calls[role] += 1
            if role == "chat":
                prompt = _token_count(values.get("prompt_tokens"))
                completion = _token_count(values.get("completion_tokens"))
                day.tokens["chat_input"] += prompt
                day.tokens["chat_output"] += completion
                day.spent_usd += (prompt * limits.chat_input_per_mtok
                                  + completion * limits.chat_output_per_mtok) / 1_000_000
            else:
                tokens = _token_count(values.get("prompt_tokens")) or _token_count(
                    values.get("total_tokens"))
                day.tokens["embedding"] += tokens
                day.spent_usd += tokens * limits.embed_per_mtok / 1_000_000
            self._save()

    def deferred(self) -> None:
        """Count one Add the gateway deferred."""
        with self._lock:
            self._roll().deferred_adds += 1
            self._save()

    def snapshot(self) -> dict[str, object]:
        """Today's ledger and the caps, for ``/agent/status``."""
        with self._lock:
            day = self._roll()
            return {
                "day_utc": time.strftime("%Y-%m-%d", time.gmtime(day.day * DAY_MS / 1000)),
                "limits": self.limits.describe(),
                "spent_usd": round(day.spent_usd, 6),
                "calls": dict(day.calls), "tokens": dict(day.tokens),
                "refused_calls": dict(day.refused), "deferred_adds": day.deferred_adds,
                "exhausted": self._exhausted(day) or self._embeddings_spent(day),
                "resets_at_ms": (day.day + 1) * DAY_MS,
            }
