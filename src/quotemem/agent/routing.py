"""Route the core's provider calls to generic OpenAI-compatible endpoints.

An agent workspace names two fixed endpoints (``CANONICAL_CHAT_BASE_URL`` and
``CANONICAL_EMBEDDING_BASE_URL``) and the core's frozen model names. The routing
clients here are ordinary ``httpx`` clients handed to the bench composition; each
request to a canonical endpoint is rewritten for the configured endpoint before it
leaves the process:

- the URL and bearer key become the configured ones, and the core's
  ``x-aml-deadline-ms`` gateway header is dropped;
- the ``model`` field becomes the configured model;
- chat: ``json_object`` mode replaces strict ``json_schema`` when configured (the
  schema is then stated in a system message, and the core still validates every
  reply against it), and reasoning parameters are sent only as configured. The
  default, ``off``, asks for no extended thinking on every call: a model that thinks
  by default (deepseek-v4.1-flash on the Volcano gateway) otherwise reasons at its own
  effort, 6,000 to 28,000 reasoning tokens a call, and a memory judge reply then runs
  out of its output bound (reasoning counts toward ``max_tokens``) and loses its
  member (seen in an end-to-end test); the same calls answer in seconds without it. A
  provider that answers 400 naming ``thinking`` gets the call again without it, and
  every later call goes without it (``thinking_refused``);
- chat under ``high``: every call asks for thinking (``thinking: {"type": "enabled"}``)
  and requests ``reasoning_effort: "high"``. DashScope's compatible mode accepts both for
  DeepSeek v4.1 flash and validates the effort's value, but no effect of the effort was
  measurable there (``none``, ``low`` and ``high`` reasoned alike); the
  thinking itself comes from ``thinking: enabled`` and the model's default. Reasoning
  counts toward ``max_tokens``, so the call's output bound gets the endpoint's
  ``reasoning_room`` on top of the core's own (``REASONING_ANSWER_TOKENS`` stands in
  for a call the core leaves unbounded), and the core's bound keeps bounding the answer.
  A reasoning attempt waits at most ``reasoning_wait`` and never more than the time the
  core gave the call (its read wait and its ``x-aml-deadline-ms``, read before the header
  is dropped) less what the resend needs (``REASONING_RESEND_SECONDS`` or half that time,
  whichever is more); when less than ``REASONING_MIN_SECONDS`` would be left, the call
  goes without thinking at once (``reasoning_skips``). A reasoning reply cut off by its
  bound (``finish_reason`` ``length``) or slower than its wait is never handed to the
  core: the same call goes again without thinking, under the core's own bound and the
  core's time left, exactly as ``off`` sends it (``reasoning_cutoffs``,
  ``reasoning_timeouts``); so does a reasoning resend after a ``thinking`` refusal.
  So a reasoning reply that its bound cuts off, or that its wait ends, is never used; a
  call the core asked to go without thinking (``thinking: disabled``) goes without it;
- embeddings: ``dimensions`` is omitted when configured.

A successful reply is checked against the configured model (the configured name, or
it with a date suffix, or a configured pattern) and then reported under the core's
model identity, so the core's identity checks hold. A reply from any other model
keeps its reported name, which the core refuses. Embedding vectors shorter than the
store's dimension are zero-padded, which keeps every cosine similarity exactly;
missing ``index`` fields are filled by position. A reply without usage (embeddings or
chat) gets an estimate of four characters per token, and the reply is counted in
``estimated_replies``. Each role sums the prompt and completion tokens of its
successful replies, reported or estimated. Every call, success and failure is
recorded per role for ``/agent/status``. An asynchronous call cancelled after its
request went out and before its reply was read (whatever cancelled it: a judge
confirmation vote no decision needs any more, a deadline, a client that went away, a
shutdown) is counted as a call and in ``cancelled``, and the spend guard counts it as a
sent call without usage: no usage came back, and the provider may still bill the work it
did. A call cancelled before its request went out (waiting for a connection) counts
nothing, and one whose reply was already read is accounted as that reply was.
Keys never leave this module except in the outgoing ``Authorization`` header.

Two optional controls act here, on the wire (``quotemem.agent.spend``, ``quotemem.agent.pii``):
a spend guard refuses a call locally, as a 429 without any network request, once its
daily cap is reached and prices each reply's usage; and PII scrubbing replaces email
addresses and phone numbers in every outgoing text and restores them in chat replies.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import math
import re
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final, Literal, cast

import httpx

from quotemem.agent.pii import Scrubbing
from quotemem.agent.settings import (
    CANONICAL_CHAT_BASE_URL,
    CANONICAL_EMBEDDING_BASE_URL,
    ChatEndpoint,
    EmbeddingEndpoint,
)
from quotemem.core.model_identity import current_inference_model

if TYPE_CHECKING:
    from quotemem.agent.spend import SpendGuard

type Role = Literal["chat", "embedding"]

SCHEMA_INSTRUCTION = (
    "Return only one JSON object, with no surrounding text, that conforms to this "
    "JSON Schema:\n"
)
"""The system message that carries the output schema in ``json_object`` mode."""
_DATE_SUFFIX = re.compile(r"-(?:[0-9]{4}-[0-9]{2}-[0-9]{2}|[0-9]{4,8})")
_DROPPED_REQUEST_HEADERS = frozenset({
    "authorization", "x-aml-deadline-ms", "content-length", "host", "x-api-key",
})
_DROPPED_RESPONSE_HEADERS = frozenset({"content-length", "content-encoding", "transfer-encoding"})
REASONING_RESEND_SECONDS: Final = 30.0
"""The least time a reasoning attempt leaves of its call for the resend without thinking.

A reasoning attempt waits at most the call's time less this or half that time, whichever is
more, so a resend after a cut-off or late reasoning reply still answers within the core's
deadline (non-thinking replies took 5 to 16 s on DashScope)."""
REASONING_MIN_SECONDS: Final = 10.0
"""A call that leaves a reasoning attempt less than this goes without thinking at once."""
REASONING_ANSWER_TOKENS: Final = 8_192
"""The answer room ``high`` gives a chat call whose core request sets no output bound.

An unbounded reasoning reply would wait as long as the model thinks; with this room plus
the endpoint's reasoning room the reasoning attempt is bounded, and a reply that needs more
goes again without thinking and without a bound, as under ``off``."""


def admits_model(configured: str, pattern: re.Pattern[str] | None, reported: object) -> bool:
    """Whether a reply's ``model`` is the configured model.

    Args:
        configured: The configured wire model name.
        pattern: A configured full-match pattern, which replaces the default rule.
        reported: The reply's ``model`` value.

    Returns:
        With a pattern, whether it fully matches. Otherwise whether the reply names
        the configured model exactly, or followed by one date-like suffix
        (``-2024-07-18``, ``-0613``, ``-261017``), as OpenAI-compatible services
        report dated snapshots.
    """
    if not isinstance(reported, str) or not reported:
        return False
    if pattern is not None:
        return pattern.fullmatch(reported) is not None
    if reported == configured:
        return True
    return reported.startswith(configured) and (
        _DATE_SUFFIX.fullmatch(reported[len(configured):]) is not None
    )


@dataclass
class RoleHealth:
    """One role's call record, read by ``/agent/status``."""

    calls: int = 0
    failures: int = 0
    consecutive_failures: int = 0
    last_success_ms: int | None = None
    last_failure_ms: int | None = None
    last_failure: str | None = None
    scrubbed: int = 0
    spend_refusals: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    estimated_replies: int = 0
    unreported_model: int = 0
    model_mismatch: int = 0
    padded_vectors: int = 0
    thinking_refused: int = 0
    reasoning_tokens: int = 0
    reasoning_cutoffs: int = 0
    reasoning_timeouts: int = 0
    reasoning_skips: int = 0
    cancelled: int = 0


@dataclass
class ProviderHealth:
    """Thread-safe call records of both roles."""

    clock_ms: Callable[[], int] = field(default=lambda: time.time_ns() // 1_000_000)
    _roles: dict[Role, RoleHealth] = field(
        default_factory=lambda: {"chat": RoleHealth(), "embedding": RoleHealth()},
    )
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def record(self, role: Role, *, failure: str | None = None, **counts: int) -> None:
        """Record one finished call; ``counts`` add to the role's named counters."""
        with self._lock:
            health = self._roles[role]
            health.calls += 1
            for name, value in counts.items():
                setattr(health, name, getattr(health, name) + value)
            if failure is None:
                health.consecutive_failures = 0
                health.last_success_ms = self.clock_ms()
            else:
                health.failures += 1
                health.consecutive_failures += 1
                health.last_failure_ms = self.clock_ms()
                health.last_failure = failure

    def add(self, role: Role, **counts: int) -> None:
        """Add to the role's named counters without recording a call."""
        with self._lock:
            health = self._roles[role]
            for name, value in counts.items():
                setattr(health, name, getattr(health, name) + value)

    def record_cancel(self, role: Role) -> None:
        """Record one call its caller closed before the reply: a call without usage."""
        with self._lock:
            health = self._roles[role]
            health.calls += 1
            health.cancelled += 1

    def record_refusal(self, role: Role) -> None:
        """Record that the role's provider refused the ``thinking`` parameter."""
        with self._lock:
            self._roles[role].thinking_refused += 1

    def record_scrub(self, role: Role, replaced: int) -> None:
        """Count the texts scrubbing replaced in one outgoing request."""
        with self._lock:
            self._roles[role].scrubbed += replaced

    def count(self, role: Role, name: str) -> int:
        """One counter of one role."""
        with self._lock:
            return cast(int, getattr(self._roles[role], name))

    def snapshot(self) -> dict[Role, dict[str, object]]:
        """A copy of both roles' records."""
        with self._lock:
            return {role: dict(vars(health)) for role, health in self._roles.items()}


@dataclass(frozen=True)
class ProviderRoutes:
    """The configured endpoints, the store's dimension and the shared health record.

    Attributes:
        spend: The daily spend guard, or None when uncapped.
        scrub: Replace email addresses and phone numbers in outgoing texts.
    """

    chat: ChatEndpoint
    embedding: EmbeddingEndpoint
    dimensions: int
    health: ProviderHealth = field(default_factory=ProviderHealth)
    spend: SpendGuard | None = None
    scrub: bool = False
    _thinking_refused: threading.Event = field(default_factory=threading.Event, repr=False)
    """Set once the chat provider refused ``thinking`` (``off``); later calls omit it."""

    def role(self, request: httpx.Request) -> Role:
        """The role of a request to a canonical endpoint.

        Raises:
            httpx.UnsupportedProtocol: For any other destination, which an agent
                composition never sends.
        """
        url = str(request.url)
        if url.startswith(CANONICAL_CHAT_BASE_URL + "/"):
            return "chat"
        if url.startswith(CANONICAL_EMBEDDING_BASE_URL + "/"):
            return "embedding"
        raise httpx.UnsupportedProtocol("Agent routing has no endpoint for this request")

    def route(self, request: httpx.Request) -> tuple[Role, str, httpx.Request]:
        """Rewrite one canonical request for its configured endpoint.

        Returns:
            (role, the model name the core requested, the outgoing request).
        """
        role, logical, outgoing, _ = self.route_scrubbed(request)
        return role, logical, outgoing

    def route_scrubbed(
        self, request: httpx.Request, *, thinking: bool = True, spent: float | None = None,
    ) -> tuple[Role, str, httpx.Request, Scrubbing | None]:
        """``route``, also returning the placeholders scrubbing issued (None when off).

        ``thinking=False`` sends a chat call under ``high`` as ``off`` sends it: the
        resend of a reasoning reply that was cut off or too slow. ``spent`` is the time in
        seconds earlier attempts of this call took, None for its first attempt; under
        ``high`` it comes off the time the core gave the call. A resend is told by
        ``spent`` being set, never by its value: Windows' monotonic clock ticks every
        15.6 ms, so an attempt shorter than one tick has spent 0.0.
        """
        role = self.role(request)
        payload = cast(dict[str, Any], json.loads(request.content))
        logical = cast(str, payload.get("model", ""))
        scrubbing = Scrubbing() if self.scrub else None
        high = role == "chat" and self.chat.reasoning == "high"
        left = _time_left(request, spent or 0.0) if high else None
        wait: float | None = None
        if high and thinking and not _thinking_off(payload):
            wait = _attempt_wait(self.chat.reasoning_wait, left)
            if wait is None:
                thinking = False
                self.health.add(role, reasoning_skips=1)
        if role == "chat":
            endpoint: ChatEndpoint | EmbeddingEndpoint = self.chat
            base, canonical = self.chat.base_url, CANONICAL_CHAT_BASE_URL
            if scrubbing is not None:
                _scrub_messages(payload, scrubbing)
            payload = self._chat_payload(payload, thinking=thinking)
        else:
            endpoint = self.embedding
            base, canonical = self.embedding.base_url, CANONICAL_EMBEDDING_BASE_URL
            payload["model"] = self.embedding.model
            if not self.embedding.send_dimensions:
                payload.pop("dimensions", None)
            if scrubbing is not None:
                _scrub_inputs(payload, scrubbing)
        if scrubbing is not None and scrubbing.replaced:
            self.health.record_scrub(role, scrubbing.replaced)
        headers = [
            (name, value) for name, value in request.headers.multi_items()
            if name.lower() not in _DROPPED_REQUEST_HEADERS
        ]
        headers.append(("Authorization", f"Bearer {endpoint.api_key}"))
        extensions: Mapping[str, Any] = request.extensions
        if wait is not None and _reasons(payload):
            extensions = _read_wait(extensions, wait, shortened=True)
        elif high and spent is not None and left is not None:
            extensions = _read_wait(extensions, max(1.0, left), shortened=False)
        outgoing = httpx.Request(
            request.method, base + str(request.url)[len(canonical):], headers=headers,
            content=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            extensions=extensions,
        )
        return role, logical, outgoing, scrubbing

    def refusal(self, role: Role, request: httpx.Request) -> httpx.Response | None:
        """A local 429 when the spend guard refuses this call, else None."""
        if self.spend is None or self.spend.admit(role):
            return None
        self.health.record(role, failure="spend_cap_reached", spend_refusals=1)
        return httpx.Response(429, json={"error": {
            "code": "spend_cap_reached", "message": "The daily provider spend cap is reached",
        }}, headers={"retry-after": str(self.spend.retry_after_seconds())}, request=request)

    def _chat_payload(self, payload: dict[str, Any], *, thinking: bool = True) -> dict[str, Any]:
        payload["model"] = self.chat.model
        response_format = payload.get("response_format")
        if self.chat.structured_output == "json_object" and isinstance(response_format, dict):
            described = cast(dict[str, Any], response_format)
            schema = cast(dict[str, Any], described.get("json_schema") or {}).get("schema")
            payload["response_format"] = {"type": "json_object"}
            if schema is not None:
                messages = cast(list[Any], payload.get("messages") or [])
                payload["messages"] = [{
                    "role": "system",
                    "content": SCHEMA_INSTRUCTION + json.dumps(
                        schema, ensure_ascii=False, separators=(",", ":"), sort_keys=True,
                    ),
                }, *messages]
        reasoning = self.chat.reasoning
        if reasoning == "high" and (not thinking or _thinking_off(payload)):
            reasoning = "off"
        if reasoning in ("off", "none"):
            payload.pop("thinking", None)
            payload.pop("reasoning_effort", None)
            if reasoning == "off" and not self._thinking_refused.is_set():
                payload["thinking"] = {"type": "disabled"}
        elif reasoning == "high":
            payload.pop("thinking", None)
            if not self._thinking_refused.is_set():
                payload["thinking"] = {"type": "enabled"}
            payload["reasoning_effort"] = "high"
            bound = payload.get("max_tokens")
            answer = bound if type(bound) is int and bound > 0 else REASONING_ANSWER_TOKENS
            payload["max_tokens"] = answer + self.chat.reasoning_room
        elif reasoning == "effort":
            payload.pop("thinking", None)
        return payload

    def reasoning_cut(self, role: Role, outgoing: httpx.Request,
                      response: httpx.Response) -> bool:
        """Whether this is a reasoning reply its bound cut off; records and charges it.

        The caller sends the call again without thinking, so the cut reply never reaches
        the core; its usage is still counted and charged, as the provider bills it.
        """
        if (role != "chat" or self.chat.reasoning != "high" or not response.is_success
                or not _reasons_on_wire(outgoing)):
            return False
        try:
            values = cast(dict[str, Any], json.loads(response.content))
            choice: object = cast(list[Any], values["choices"])[0]
        except (ValueError, KeyError, IndexError, TypeError):
            return False
        if not isinstance(choice, dict) or (
            cast(dict[str, Any], choice).get("finish_reason") != "length"
        ):
            return False
        usage = values.get("usage")
        reported = cast(dict[str, Any], usage) if isinstance(usage, dict) else {}
        if self.spend is not None:
            self.spend.charge(role, reported or None)
        self.health.add(
            role, reasoning_cutoffs=1,
            prompt_tokens=_token_count(reported.get("prompt_tokens")),
            completion_tokens=_token_count(reported.get("completion_tokens")),
            reasoning_tokens=_reasoning_tokens(reported),
        )
        return True

    def reasoning_late(self, role: Role, outgoing: httpx.Request, error: Exception) -> bool:
        """Whether ``error`` is the reasoning wait running out; records it.

        Only a read timeout of a reasoning attempt, whose wait ``high`` set, counts: the
        core's own wait running out on a call without thinking stays the core's timeout.
        """
        if (role != "chat" or self.chat.reasoning != "high"
                or not isinstance(error, httpx.ReadTimeout) or not _reasons_on_wire(outgoing)
                or not outgoing.extensions.get(_SHORTENED_WAIT)):
            return False
        if self.spend is not None:
            # The provider may bill the unanswered call; no usage came back to price.
            self.spend.charge(role, None)
        self.health.add(role, reasoning_timeouts=1)
        return True

    def cancelled(self, role: Role) -> None:
        """Count a call its caller closed before the reply (no usage came back)."""
        if self.spend is not None:
            # The provider may bill the unanswered call; no usage came back to price.
            self.spend.charge(role, None)
        self.health.record_cancel(role)

    def refuses_thinking(self, role: Role, outgoing: httpx.Request,
                         response: httpx.Response) -> bool:
        """Whether this reply refused the ``thinking`` parameter routing added; learns it.

        A 400 whose body names ``thinking``, to a chat call that sent it under ``off`` or
        ``high`` (``high`` keeps sending ``reasoning_effort``),
        means the provider does not take the parameter: every later call omits it, and
        the caller sends this one again without it.
        """
        if (role != "chat" or self.chat.reasoning not in ("off", "high")
                or response.status_code != 400
                or b"thinking" not in response.content.lower()
                or b'"thinking"' not in outgoing.content):
            return False
        if not self._thinking_refused.is_set():
            self._thinking_refused.set()
            self.health.record_refusal(role)
        return True

    def adapt(
        self, role: Role, logical: str, request: httpx.Request, response: httpx.Response,
        scrubbing: Scrubbing | None = None,
    ) -> httpx.Response:
        """Map one reply back to the core's identities and record the call.

        A chat reply has every placeholder ``scrubbing`` issued restored first.
        """
        if not response.is_success:
            if self.spend is not None:
                self.spend.charge(role, None)
            self.health.record(role, failure=f"http_{response.status_code}")
            return response
        try:
            raw: object = json.loads(response.content)
        except ValueError:
            self.health.record(role, failure="reply_not_json")
            return response
        if not isinstance(raw, dict):
            self.health.record(role, failure="reply_not_object")
            return response
        values = cast(dict[str, Any], raw)
        counts: dict[str, int] = {}
        endpoint: ChatEndpoint | EmbeddingEndpoint = self.chat if role == "chat" else self.embedding
        reported = values.get("model")
        if reported is None or reported == "":
            # Some local servers omit it; the request went only to this endpoint.
            reported = endpoint.model
            counts["unreported_model"] = 1
        failure: str | None = None
        if admits_model(endpoint.model, endpoint.model_pattern, reported):
            values["model"] = (
                current_inference_model("openai", logical) if role == "chat" else logical
            )
        else:
            values["model"] = reported
            counts["model_mismatch"] = 1
            failure = "model_mismatch"
        if role == "embedding":
            counts.update(self._embedding_reply(values))
        elif scrubbing is not None:
            _restore_choices(values, scrubbing)
        counts.update(_usage(role, values, request))
        if self.spend is not None:
            self.spend.charge(role, values.get("usage"))
        self.health.record(role, failure=failure, **counts)
        headers = [
            (name, value) for name, value in response.headers.multi_items()
            if name.lower() not in _DROPPED_RESPONSE_HEADERS
        ]
        return httpx.Response(
            response.status_code, headers=headers,
            content=json.dumps(values, ensure_ascii=False).encode("utf-8"), request=request,
        )

    def _embedding_reply(self, values: dict[str, Any]) -> dict[str, int]:
        counts: dict[str, int] = {}
        entries = values.get("data")
        if isinstance(entries, list):
            for position, entry in enumerate(cast(list[Any], entries)):
                if not isinstance(entry, dict):
                    continue
                item = cast(dict[str, Any], entry)
                item.setdefault("index", position)
                vector = item.get("embedding")
                if isinstance(vector, list) and len(cast(list[Any], vector)) < self.dimensions:
                    # Zero coordinates change neither norms nor dot products.
                    item["embedding"] = [
                        *cast(list[Any], vector),
                        *([0.0] * (self.dimensions - len(cast(list[Any], vector)))),
                    ]
                    counts["padded_vectors"] = counts.get("padded_vectors", 0) + 1
        return counts


def _scrub_messages(payload: dict[str, Any], scrubbing: Scrubbing) -> None:
    """Scrub every chat message's text, reserving placeholders the texts already hold."""
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return
    entries = [cast(dict[str, Any], entry) for entry in cast(list[Any], messages)
               if isinstance(entry, dict)]
    parts: list[dict[str, Any]] = []
    for entry in entries:
        content = entry.get("content")
        if isinstance(content, str):
            scrubbing.reserve(content)
        elif isinstance(content, list):
            for part in cast(list[Any], content):
                values = cast(dict[str, Any], part) if isinstance(part, dict) else {}
                if isinstance(values.get("text"), str):
                    parts.append(values)
                    scrubbing.reserve(cast(str, values["text"]))
    for entry in entries:
        if isinstance(entry.get("content"), str):
            entry["content"] = scrubbing.scrub(cast(str, entry["content"]))
    for part in parts:
        part["text"] = scrubbing.scrub(cast(str, part["text"]))


def _scrub_inputs(payload: dict[str, Any], scrubbing: Scrubbing) -> None:
    """Scrub embedding inputs; vectors of scrubbed text are never restored."""
    texts = payload.get("input")
    if isinstance(texts, str):
        payload["input"] = scrubbing.scrub(texts)
    elif isinstance(texts, list):
        payload["input"] = [scrubbing.scrub(text) if isinstance(text, str) else text
                            for text in cast(list[Any], texts)]


def _restore_choices(values: dict[str, Any], scrubbing: Scrubbing) -> None:
    """Restore the issued placeholders in every chat choice's content."""
    choices = values.get("choices")
    if not isinstance(choices, list) or not scrubbing.originals:
        return
    for choice in cast(list[Any], choices):
        message = cast(dict[str, Any], choice).get("message") if isinstance(choice, dict) else None
        if isinstance(message, dict):
            content = cast(dict[str, Any], message).get("content")
            if isinstance(content, str):
                cast(dict[str, Any], message)["content"] = scrubbing.restore(content)


def _tokens(characters: int) -> int:
    return max(1, math.ceil(characters / 4))


def _token_count(value: object) -> int:
    return value if type(value) is int and value >= 0 else 0


def _reasoning_tokens(usage: dict[str, Any]) -> int:
    """A reply's reported reasoning tokens (``completion_tokens_details``), else 0."""
    details = usage.get("completion_tokens_details")
    if not isinstance(details, dict):
        return 0
    return _token_count(cast(dict[str, Any], details).get("reasoning_tokens"))


_SHORTENED_WAIT: Final = "aml_agent_reasoning_wait"
"""The outgoing extension marking a reasoning attempt, whose read wait ``high`` set."""


def _thinking_off(payload: dict[str, Any]) -> bool:
    """Whether the core asked this call to go without thinking."""
    thinking = payload.get("thinking")
    return thinking == "disabled" or (
        isinstance(thinking, dict) and cast(dict[str, Any], thinking).get("type") == "disabled"
    )


def _reasons(payload: dict[str, Any]) -> bool:
    """Whether a routed chat payload asks for reasoning at high effort."""
    return payload.get("reasoning_effort") == "high" and not _thinking_off(payload)


def _reasons_on_wire(outgoing: httpx.Request) -> bool:
    try:
        payload = json.loads(outgoing.content)
    except ValueError:
        return False
    return isinstance(payload, dict) and _reasons(cast(dict[str, Any], payload))


def _time_left(request: httpx.Request, spent: float) -> float | None:
    """The seconds the core gave this call, less ``spent``; None when it set no limit.

    The core bounds a call by its read wait and, for a call under a request budget, by
    the whole send (``post_with_budget``); the adapter states the smaller of the two in
    ``x-aml-deadline-ms``.
    """
    limits: list[float] = []
    timeout = request.extensions.get("timeout")
    if isinstance(timeout, dict):
        read = cast(dict[str, Any], timeout).get("read")
        if isinstance(read, int | float):
            limits.append(float(read))
    header = request.headers.get("x-aml-deadline-ms")
    if header is not None:
        try:
            limits.append(int(header) / 1000)
        except ValueError:
            pass
    return None if not limits else min(limits) - spent


def _attempt_wait(reasoning_wait: float, left: float | None) -> float | None:
    """The wait of a reasoning attempt, or None when the call has no time for one."""
    if left is None:
        return reasoning_wait
    wait = min(reasoning_wait, left - max(REASONING_RESEND_SECONDS, left / 2))
    return wait if wait >= REASONING_MIN_SECONDS else None


def _read_wait(extensions: Mapping[str, Any], seconds: float, *,
               shortened: bool) -> Mapping[str, Any]:
    """``extensions`` with the read wait at most ``seconds``.

    ``shortened`` marks a reasoning attempt, whose timeout is then ``high``'s own.
    """
    timeout = extensions.get("timeout")
    waits = dict(cast(dict[str, Any], timeout)) if isinstance(timeout, dict) else {}
    read = waits.get("read")
    if read is None or read > seconds:
        waits["read"] = seconds
    marked = {_SHORTENED_WAIT: True} if shortened else {}
    return {**extensions, "timeout": waits, **marked}


def _usage(role: Role, values: dict[str, Any], request: httpx.Request) -> dict[str, int]:
    """Estimate missing usage in place and return the reply's token counts."""
    usage = values.get("usage")
    if isinstance(usage, dict):
        reported = cast(dict[str, Any], usage)
        prompt = _token_count(reported.get("prompt_tokens", reported.get("total_tokens")))
        completion = _token_count(reported.get("completion_tokens")) if role == "chat" else 0
        counts = {"prompt_tokens": prompt, "completion_tokens": completion}
        if role == "chat" and (reasoning := _reasoning_tokens(reported)):
            counts["reasoning_tokens"] = reasoning
        return counts
    sent = cast(dict[str, Any], json.loads(request.content))
    if role == "embedding":
        prompt = _tokens(sum(len(text) for text in cast(list[Any], sent.get("input") or [])
                             if isinstance(text, str)))
        values["usage"] = {"prompt_tokens": prompt, "total_tokens": prompt}
        return {"prompt_tokens": prompt, "completion_tokens": 0, "estimated_replies": 1}
    prompt = _tokens(sum(
        len(content) for message in cast(list[Any], sent.get("messages") or [])
        if isinstance(message, dict)
        and isinstance(content := cast(dict[str, Any], message).get("content"), str)
    ))
    completion = _tokens(sum(
        len(content) for choice in cast(list[Any], values.get("choices") or [])
        if isinstance(choice, dict)
        and isinstance(message := cast(dict[str, Any], choice).get("message"), dict)
        and isinstance(content := cast(dict[str, Any], message).get("content"), str)
    ))
    values["usage"] = {"prompt_tokens": prompt, "completion_tokens": completion,
                       "total_tokens": prompt + completion}
    return {"prompt_tokens": prompt, "completion_tokens": completion, "estimated_replies": 1}


class RoutedClient(httpx.Client):
    """A synchronous client that sends canonical provider requests through ``routes``."""

    def __init__(self, routes: ProviderRoutes, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.routes = routes

    def send(self, request: httpx.Request, **kwargs: Any) -> httpx.Response:
        """Route, send and adapt one request; a spend refusal sends nothing."""
        routes = self.routes
        started = time.monotonic()
        role, logical, outgoing, scrubbing = routes.route_scrubbed(request)
        refused = routes.refusal(role, request)
        if refused is not None:
            return refused
        try:
            response = self._attempt(role, outgoing, **kwargs)
            if response is not None and routes.refuses_thinking(role, outgoing, response):
                response.close()
                role, logical, outgoing, scrubbing = routes.route_scrubbed(
                    request, spent=time.monotonic() - started)
                response = self._attempt(role, outgoing, **kwargs)
            if response is None or routes.reasoning_cut(role, outgoing, response):
                if response is not None:
                    response.close()
                role, logical, outgoing, scrubbing = routes.route_scrubbed(
                    request, thinking=False, spent=time.monotonic() - started)
                response = super().send(outgoing, **kwargs)
                response.read()
        except Exception as error:
            routes.health.record(role, failure=type(error).__name__)
            raise
        return routes.adapt(role, logical, request, response, scrubbing)

    def _attempt(self, role: Role, outgoing: httpx.Request,
                 **kwargs: Any) -> httpx.Response | None:
        """Send one attempt; None when it was a reasoning attempt that ran out of its wait."""
        try:
            response = super().send(outgoing, **kwargs)
            response.read()
        except httpx.ReadTimeout as error:
            if not self.routes.reasoning_late(role, outgoing, error):
                raise
            return None
        return response


class RoutedAsyncClient(httpx.AsyncClient):
    """The asynchronous counterpart, used for the core's bounded calls."""

    def __init__(self, routes: ProviderRoutes, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.routes = routes

    async def send(self, request: httpx.Request, **kwargs: Any) -> httpx.Response:
        """Route, send and adapt one request; a spend refusal sends nothing."""
        routes = self.routes
        started = time.monotonic()
        role, logical, outgoing, scrubbing = routes.route_scrubbed(request)
        refused = routes.refusal(role, request)
        if refused is not None:
            return refused
        unread = _Unread()
        try:
            response = await self._attempt(role, outgoing, unread, **kwargs)
            if response is not None and routes.refuses_thinking(role, outgoing, response):
                await response.aclose()
                role, logical, outgoing, scrubbing = routes.route_scrubbed(
                    request, spent=time.monotonic() - started)
                response = await self._attempt(role, outgoing, unread, **kwargs)
            if response is None or routes.reasoning_cut(role, outgoing, response):
                if response is not None:
                    await response.aclose()
                role, logical, outgoing, scrubbing = routes.route_scrubbed(
                    request, thinking=False, spent=time.monotonic() - started)
                response = await self._read(outgoing, unread, **kwargs)
        except asyncio.CancelledError:
            if unread.sent:
                routes.cancelled(role)
            raise
        except Exception as error:
            routes.health.record(role, failure=type(error).__name__)
            raise
        return routes.adapt(role, logical, request, response, scrubbing)

    async def _attempt(self, role: Role, outgoing: httpx.Request, unread: _Unread,
                       **kwargs: Any) -> httpx.Response | None:
        """Send one attempt; None when it was a reasoning attempt that ran out of its wait."""
        try:
            return await self._read(outgoing, unread, **kwargs)
        except httpx.ReadTimeout as error:
            if not self.routes.reasoning_late(role, outgoing, error):
                raise
            unread.sent = False  # charged as an unanswered call
            return None

    async def _read(self, outgoing: httpx.Request, unread: _Unread,
                    **kwargs: Any) -> httpx.Response:
        """Send one physical call and read its reply, keeping ``unread`` current.

        ``unread.sent`` turns true once the call's request headers start going out
        (httpcore's ``trace`` extension) and false once its reply is read; whoever reads
        the reply accounts for it. Behind an HTTPS proxy httpcore passes the request's
        extensions, ``trace`` included, to its CONNECT request, so a cancel during the
        CONNECT or the TLS handshake is charged although the provider received nothing:
        this errs toward counting a call that was not billed, never the reverse.
        """
        previous = outgoing.extensions.get("trace")

        async def trace(name: str, info: dict[str, Any]) -> None:
            if name.endswith(".send_request_headers.started"):
                unread.sent = True
            if previous is not None:
                result = previous(name, info)
                if inspect.isawaitable(result):
                    await result

        outgoing.extensions = {**outgoing.extensions, "trace": trace}
        response = await super().send(outgoing, **kwargs)
        await response.aread()
        unread.sent = False
        return response


@dataclass
class _Unread:
    """Whether a call's latest request went out and its reply is not read yet."""

    sent: bool = False
