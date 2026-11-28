"""The agent profile's ASGI front: input normalization, readiness and status.

``AgentGateway`` wraps the bench application and serves the product's public memory
routes over it (`PUBLIC_ROUTES`): ``POST /memory/add``, ``/memory/search``,
``/memory/history``, ``/memory/provenance`` and ``/memory/add-results`` run the bench
routes ``/add``, ``/search``, ``/memory/history``, ``/memory/why`` and
``/memory/add-items`` with the same request and response bodies. Forget is one pair
(`forget_route`): ``POST /memory/forget/prepare`` names its target with exactly one of
``statement`` (the user's words: forget by statement) or ``memory_id`` (one memory by
id), and ``POST /memory/forget/confirm`` takes the prepare's ``confirmation`` exactly as
returned (a by-id one carries ``memory_id``, a statement plan ``plan_binding``). A body
naming both or neither is answered 422 ``invalid_request``. The bench routes' own names
(``/add``, ``/search``, ``/memory/why``, ``/memory/add-items``, ``/memory/forget``,
``/memory/forget/statement`` and ``/memory/forget/statement/prepare``) are not served
here (422 ``invalid_request``, as any unknown route); the bench server (``quotemem bench serve``)
keeps them.

Before the bench application exists (providers unreachable, store opening) it answers
the memory routes and ``/health`` with 503 and ``Retry-After``, so a plugin treats the
turn as having no memory and carries on. It always answers:

- ``GET /agent/ready``: unauthenticated and content-free, 200 once serving;
- ``GET /agent/listener`` with ``x-quotemem-challenge: <32 to 128 hex digits>``:
  unauthenticated, and only on a loopback address to a loopback client (else 404);
  ``{"proof": "<hex>"}``: the hex HMAC-SHA256 under the service token of
  ``quotemem-listener-proof``, NUL, the address and port the service listens on, NUL
  and the challenge (`listener_proof`). A client on the same computer checks it before
  it sends the token: a program of another user that took the port cannot answer it;
- ``GET /agent/status``: authenticated with the service token; the phase, the
  store binding and details (users, quarantined users, active Add owners, Adds in
  flight and deferred, the embedding canary state), the day's spend, provider call
  records and request counters, never content or keys.

Once serving it also answers two operator routes, authenticated with the token:

- ``POST /agent/admin/delete-user`` ``{"user_id": X, "confirm": X}``: erase one
  user's memories and sources (``AgentStore.delete_user``); ``confirm`` must
  repeat the user id;
- ``POST /agent/admin/repair`` ``{"user_id": X, "action": "inspect"|"resume"|"skip"}``:
  the existing explicit recovery of a stopped or quarantined Add owner.

and, authenticated with the same token as ``/memory/search``, ``POST /memory/list``
``{"user_id": X, "status": "current"|"past"|"possibly_outdated"|"all", "limit": 1..200,
"cursor": C}``: one page of the user's memories with their status and quotes, newest
change first (``quotemem.api.memory_list``); a user the store never saw gets an empty list.

While the day's spend cap is reached a new Add, and any Add body too long to
normalize, is answered 429 ``spend_cap_reached`` (``status: deferred``,
``Retry-After`` at the next UTC midnight) and Search keeps serving stored
memories. An authenticated exact resend of an Add whose result is already
stored (``AgentStore.completed_adds``), such as the resend that reads back an Add
sent with ``Prefer: respond-async``, spends nothing and is still answered with
that result (200 and its ``agent`` object). So is a resend of an Add this gateway
acknowledged 202 and still counts in ``adds.background``: it answers 202 while the
Add completes and its result once stored, never 429. While a user is
being deleted their Adds are answered 409 ``user_deleting``.

``adds.in_flight`` in ``/agent/status`` counts every Add whose extraction has not
finished: the synchronous ones being answered and those acknowledged 202 with
``Prefer: respond-async`` (also reported alone as ``adds.background``), until a
resend reads their result or the status read finds it stored; a resend in progress
is not counted twice. An acknowledged Add
of a user later deleted stops counting; one acknowledged before a restart is
resumed by the store and shows in ``store.active_add_owners`` instead.

A successful Add says ``success: true`` even when some of its facts were not
stored: the bench receipt counts them as ``rejected`` or ``unresolved`` and stops there.
The gateway adds an ``agent`` object to every successful Add (`add_report`): a
``status`` of ``stored`` or ``partial``, the two counts, the reason code of each fact
that did not become a memory and of every item, read from the request's add-results
record, so a plugin can log why a capture was only partly stored, the
memories it created that a read still shows, and what each in-chat forget request of the
Add did (``forgets``): an in-chat forget is never silent. The bench
fields are returned unchanged.

It also adapts three responses of the bench application for agent clients (the
bench profile itself is unchanged):

- Search hit content comes back as the original text. The bench renderer escapes
  ``&``, ``<`` and ``>`` inside each ``<memory untrusted="true">`` block, so a
  quoted ``>`` reached the plugin as ``&gt;``; the gateway reverses exactly that
  escaping and keeps only a ``<`` that would open or close a ``memory`` tag
  escaped, so stored text cannot end the untrusted block early.
- ``/memory/history`` and ``/memory/provenance`` for an id that names no memory of the user
  (an Evidence hit's id, a forgotten memory, an unknown id) answer 404
  ``memory_not_found`` with a message saying so, instead of a bare 403
  ``scope_denied``.
- Every request except ``/agent/ready`` writes one JSON line to stdout (`RequestLog`):
  method, route, a keyed hash of the user, status, duration and counts. Never content,
  ids or credentials.

Every Add body is normalized first (``quotemem.agent.ingest.normalize_add``). In a store
whose Add mode is ``agent`` (the default; self-identity rule ``user-role-v1``) a user
message there is the authenticated user's own words: its ``speaker`` and
``source_participant`` are dropped and counted (``dropped_user_speakers`` in the request
line's counts and in ``/agent/status`` ``ingest``), so "I" in it is the user. A ``full``
or ``lean-local`` store receives the speaker metadata of every message as sent.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import ipaddress
import json
import re
import sys
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any, Literal, cast

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from quotemem.adapters.http.auth import HTTPCredentials
from quotemem.adapters.http.errors import public_error
from quotemem.agent.ingest import IngestNotes, normalize_add, normalize_search
from quotemem.agent.routing import ProviderHealth
from quotemem.core.errors import QuoteMemError
from quotemem.core.extraction_policy import SelfIdentityRule

if TYPE_CHECKING:
    from quotemem.agent.admin import AgentStore
    from quotemem.agent.spend import SpendGuard

type Phase = Literal["probing", "opening", "serving", "stopping"]

MAX_NORMALIZED_BODY = 4 * 1_048_576
"""Bodies above this are forwarded untouched; the strict contract refuses them anyway."""
MAX_ADMIN_BODY = 65_536
_REPAIR_ACTIONS = frozenset({"inspect", "resume", "skip"})
MAX_ADAPTED_RESPONSE = 16 * 1_048_576
"""Responses above this are passed through unadapted."""
PUBLIC_ROUTES: Mapping[str, str] = {
    "/memory/add": "/add",
    "/memory/search": "/search",
    "/memory/history": "/memory/history",
    "/memory/provenance": "/memory/why",
    "/memory/add-results": "/memory/add-items",
}
"""Each public memory route and the bench application route that serves it."""
FORGET_PREPARE = "/memory/forget/prepare"
FORGET_CONFIRM = "/memory/forget/confirm"
_FORGET_TARGETS: Mapping[str, tuple[tuple[str, str], tuple[str, str]]] = {
    # (the field naming a statement forget, its route), (the by-id field, its route)
    FORGET_PREPARE: (("statement", "/memory/forget/statement/prepare"),
                     ("memory_id", "/memory/forget/prepare")),
    FORGET_CONFIRM: (("plan_binding", "/memory/forget/statement"),
                     ("memory_id", "/memory/forget")),
}
BENCH_ROUTES = frozenset({
    "/add", "/search", "/memory/why", "/memory/add-items", "/memory/forget",
    "/memory/forget/statement/prepare", "/memory/forget/statement",
})
"""Bench application routes the product server does not serve under their own names."""
LOGGED_ROUTES = frozenset({
    *PUBLIC_ROUTES, FORGET_PREPARE, FORGET_CONFIRM, "/health", "/agent/status",
    "/memory/list",
})
"""Routes logged by name; any other path is logged as ``other``."""
_USER_ROUTES = LOGGED_ROUTES - {"/health", "/agent/status", "/memory/list"}
LISTENER_ROUTE = "/agent/listener"
_CHALLENGE = re.compile(r"[0-9a-f]{32,128}")
MAX_LIST_BODY = 65_536
_BLOCK_OPEN = '<memory untrusted="true">\n'
_BLOCK_CLOSE = "\n</memory>"
_ESCAPED = re.compile(r"&(amp|lt|gt);")
_UNESCAPED = {"amp": "&", "lt": "<", "gt": ">"}
_TAG = re.compile(r"<(?=/?memory\b)", re.IGNORECASE)
NOT_A_MEMORY = (
    "No memory with this id belongs to this user. History and Provenance take the id of a "
    "memory: a Search hit that states something about the user, with its quoted lines. "
    "An Evidence hit (one source message, shown as its role, date and text) has no "
    "versions, and a forgotten memory has no history either."
)
"""The message of a ``memory_not_found`` answer."""


def original_text(content: str) -> str:
    """Undo the bench renderer's escaping inside one untrusted memory block.

    The renderer wraps each hit as ``<memory untrusted="true">``, a fixed notice and
    the hit's text with ``&``, ``<`` and ``>`` escaped (``html.escape``, no quotes).
    This reverses exactly those three entities in one pass, so any other entity text
    a user wrote stays as written, then escapes again only a ``<`` that starts a
    ``memory`` tag. Content not in that shape is returned unchanged.
    """
    if not (content.startswith(_BLOCK_OPEN) and content.endswith(_BLOCK_CLOSE)):
        return content
    inner = content[len(_BLOCK_OPEN):-len(_BLOCK_CLOSE)]
    restored = _ESCAPED.sub(lambda match: _UNESCAPED[match.group(1)], inner)
    return _BLOCK_OPEN + _TAG.sub("&lt;", restored) + _BLOCK_CLOSE


def write_line(document: dict[str, object]) -> None:
    """Write one JSON line to stdout; a broken stream never fails a request."""
    try:
        print(json.dumps(document, sort_keys=True), file=sys.stdout, flush=True)
    except (OSError, ValueError):
        pass


@dataclass
class RequestLog:
    """What one request's log line reports; counts only."""

    method: str
    route: str
    user: str | None = None
    status: int | None = None
    counts: dict[str, int] = field(default_factory=lambda: {})


def _version() -> str:
    try:
        return version("quotemem")
    except PackageNotFoundError:
        return "unknown"


@dataclass
class ProbeRecord:
    """The latest startup probe of the providers."""

    attempts: int = 0
    ok: bool | None = None
    role: str | None = None
    code: str | None = None
    hint: str | None = None
    at_ms: int | None = None


@dataclass
class RequestCounters:
    """Responses of one route by class, and the latest success.

    ``partial`` counts successful Adds whose ``agent.status`` was ``partial``.
    """

    ok: int = 0
    client_error: int = 0
    server_error: int = 0
    partial: int = 0
    last_ok_ms: int | None = None
    last_error_status: int | None = None


@dataclass
class AgentState:
    """Everything ``/agent/status`` reports; updated by the service and the gateway."""

    store: dict[str, object]
    health: ProviderHealth
    endpoints: dict[str, dict[str, object]]
    clock_ms: Callable[[], int] = field(default=lambda: time.time_ns() // 1_000_000)
    phase: Phase = "probing"
    phase_since_ms: int = 0
    started_at_ms: int = 0
    probe: ProbeRecord = field(default_factory=ProbeRecord)
    requests: dict[str, RequestCounters] = field(default_factory=lambda: {
        "add": RequestCounters(), "search": RequestCounters(),
    })
    ingest: dict[str, int] = field(default_factory=lambda: {
        "redacted_spans": 0, "dropped_messages": 0, "dropped_user_speakers": 0,
        "scaled_timestamps": 0, "parsed_timestamps": 0, "derived_request_ids": 0,
        "defaulted_fields": 0,
    })
    adds_in_flight: int = 0
    adds_background: dict[tuple[str, str, str], None] = field(
        default_factory=dict[tuple[str, str, str], None])
    """Adds acknowledged 202 whose result is not yet known stored, by request."""
    adds_deferred: int = 0
    adds_refused_deleting: int = 0
    add_forgets: Callable[[bytes], list[dict[str, object]]] = field(default=lambda _body: [])
    """The in-chat forget report of one completed Add, from its normalized body (the
    ``forgets`` report; ``service/runtime.BenchRuntime.add_forgets``, set by ``serve``)."""
    followups_running: Callable[[], int] = field(default=lambda: 0)
    """Background completions of acknowledged Adds still running, their forget work
    included (``service/add_followup.py``, set by ``serve``)."""
    controls: dict[str, object] = field(default_factory=dict[str, object])
    """Nonsecret data-control settings: PII scrubbing, the auto-repair policy."""
    followups_overtaken: Callable[[], dict[str, int]] = field(
        default=lambda: {"readmitted": 0, "unstored": 0})
    """Acknowledged Adds a forget of their user overtook: admitted again, or told refused
    once re-admissions ran out (``AddFollowups.readmissions``, set by ``serve``)."""
    followups_refused: Callable[[], list[tuple[str, str, str]]] = field(
        default=lambda: [])
    """Acknowledged Adds refused for good, no longer in flight (``AddFollowups.refused``,
    set by ``serve``)."""
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self) -> None:
        now = self.clock_ms()
        self.started_at_ms = self.phase_since_ms = now

    def enter(self, phase: Phase) -> None:
        """Move to ``phase``."""
        with self._lock:
            self.phase, self.phase_since_ms = phase, self.clock_ms()

    def probed(self, *, ok: bool, role: str | None, code: str | None, hint: str | None) -> None:
        """Record one provider probe."""
        with self._lock:
            self.probe.attempts += 1
            self.probe.ok, self.probe.role, self.probe.code, self.probe.hint = ok, role, code, hint
            self.probe.at_ms = self.clock_ms()

    def normalized(self, notes: IngestNotes) -> None:
        """Count what normalization changed."""
        with self._lock:
            self.ingest["redacted_spans"] += notes.redacted_spans
            self.ingest["dropped_messages"] += notes.dropped_messages
            self.ingest["dropped_user_speakers"] += notes.dropped_user_speakers
            self.ingest["scaled_timestamps"] += notes.scaled_timestamps
            self.ingest["parsed_timestamps"] += notes.parsed_timestamps
            self.ingest["derived_request_ids"] += int(notes.derived_request_id)
            self.ingest["defaulted_fields"] += len(notes.defaults)

    def responded(self, route: str, status: int) -> None:
        """Count one response of ``route``."""
        with self._lock:
            counters = self.requests[route]
            if status < 400:
                counters.ok += 1
                counters.last_ok_ms = self.clock_ms()
            else:
                if status < 500:
                    counters.client_error += 1
                else:
                    counters.server_error += 1
                counters.last_error_status = status

    def add_started(self) -> None:
        """Count one Add handed to the store."""
        with self._lock:
            self.adds_in_flight += 1

    def add_finished(self) -> None:
        """Count one Add the store answered."""
        with self._lock:
            self.adds_in_flight -= 1

    def add_accepted(self, request: tuple[str, str, str]) -> None:
        """Count one Add acknowledged before its extraction finished (``respond-async``)."""
        with self._lock:
            self.adds_background[request] = None

    def adds_settled(self, requests: Iterable[tuple[str, str, str]]) -> None:
        """Stop counting acknowledged Adds whose result is stored."""
        with self._lock:
            for request in requests:
                self.adds_background.pop(request, None)

    def adds_dropped(self, user_id: str) -> None:
        """Stop counting the acknowledged Adds of a deleted user."""
        with self._lock:
            for request in [key for key in self.adds_background if key[0] == user_id]:
                del self.adds_background[request]

    def in_background(self, request: tuple[str, str, str] | None) -> bool:
        """Whether ``request`` is an Add acknowledged 202 and still counted in flight."""
        with self._lock:
            return request is not None and request in self.adds_background

    def background_adds(self) -> list[tuple[str, str, str]]:
        """The acknowledged Adds still counted in flight."""
        with self._lock:
            return list(self.adds_background)

    def add_refused(self, reason: Literal["deferred", "deleting"]) -> None:
        """Count one Add the gateway refused before the store."""
        with self._lock:
            if reason == "deferred":
                self.adds_deferred += 1
            else:
                self.adds_refused_deleting += 1

    def partial_add(self) -> None:
        """Count one successful Add that stored only part of its facts."""
        with self._lock:
            self.requests["add"].partial += 1

    def snapshot(self) -> dict[str, object]:
        """The status document."""
        providers = self.health.snapshot()
        with self._lock:
            return {
                "adds": {"in_flight": self.adds_in_flight + len(self.adds_background),
                         "background": len(self.adds_background),
                         "deferred": self.adds_deferred,
                         "refused_deleting": self.adds_refused_deleting},
                "controls": dict(self.controls),
                "service": "quotemem", "profile": "agent", "version": _version(),
                "phase": self.phase, "phase_since_ms": self.phase_since_ms,
                "started_at_ms": self.started_at_ms, "now_ms": self.clock_ms(),
                "store": dict(self.store),
                "providers": {
                    role: {**self.endpoints.get(role, {}), **providers[role]}
                    for role in ("chat", "embedding")
                },
                "probe": asdict(self.probe),
                "requests": {route: asdict(counters) for route, counters in self.requests.items()},
                "ingest": dict(self.ingest),
            }


MAX_ADD_RESPONSE = 1_048_576
"""Larger ``/add`` bodies than this are passed on untouched; a receipt is a few hundred bytes."""


def _object(value: object) -> dict[str, object] | None:
    """A decoded JSON object with string keys, else None."""
    return cast(dict[str, object], value) if isinstance(value, dict) else None


def add_request(body: bytes) -> tuple[str, str, str] | None:
    """The ``(user_id, session_id, request_id)`` of a normalized ``/add`` body, else None."""
    try:
        fields = _object(json.loads(body))
    except (ValueError, RecursionError):
        return None
    if fields is None:
        return None
    ids = tuple(fields.get(name) for name in ("user_id", "session_id", "request_id"))
    if not all(isinstance(value, str) and value for value in ids):
        return None
    return cast(tuple[str, str, str], ids)


def forget_route(path: str, body: bytes) -> str | None:
    """The bench route that serves one public forget request, else None.

    Args:
        path: ``/memory/forget/prepare`` or ``/memory/forget/confirm``.
        body: The complete request body.

    Returns:
        For a prepare, the statement forget route when the body names ``statement``
        and the by-id route when it names ``memory_id``; for a confirm, the statement
        route when the echoed confirmation carries ``plan_binding`` and the by-id route
        when it carries ``memory_id``. None for a body that is not a JSON object or
        names both fields or neither.
    """
    try:
        fields = _object(json.loads(body))
    except (ValueError, RecursionError):
        return None
    if fields is None:
        return None
    named = [route for name, route in _FORGET_TARGETS[path] if name in fields]
    return named[0] if len(named) == 1 else None


async def _refuse(send: Send, code: Literal["invalid_request", "authentication_failed"]) -> None:
    """Answer the bench application's error body for ``code`` (``PublicErrorV1``)."""
    mapped = public_error(QuoteMemError(code, code))
    body = mapped.body.model_dump_json().encode("utf-8")
    await send({"type": "http.response.start", "status": mapped.status_code, "headers": [
        (b"content-type", b"application/json"), (b"content-length", str(len(body)).encode()),
    ]})
    await send({"type": "http.response.body", "body": body})


def _count(value: object) -> int:
    return value if type(value) is int else 0


def _reasons(items: Sequence[Mapping[str, object]]) -> dict[str, int]:
    """The reason codes of the rejected and unresolved items, with their counts."""
    reasons: dict[str, int] = {}
    for item in items:
        code = item.get("reason_code")
        if item.get("category") in ("rejected", "unresolved") and isinstance(code, str):
            reasons[code] = reasons.get(code, 0) + 1
    return dict(sorted(reasons.items()))


def add_report(
    receipt: Mapping[str, object], items: Sequence[Mapping[str, object]] | None,
    forgets: Sequence[Mapping[str, object]] = (),
) -> dict[str, object]:
    """The ``agent`` object of one successful Add receipt.

    Args:
        receipt: The bench ``/add`` response document.
        items: The request's add-items (`add_items`); None when they could not be read.
        forgets: The Add's in-chat forget requests and what each did (``forgets``).

    Returns:
        - ``status``: ``stored`` when no fact was rejected or left unresolved, else ``partial``;
        - ``unresolved`` and ``rejected``: the receipt's counts;
        - ``reasons``: the reason code of each fact counted ``rejected`` or ``unresolved``, with
          its count (an empty object for a stored Add, null when the items are unknown);
        - ``items``: every item's ``category`` and ``reason_code`` in item order, so a stored
          Add says what it did too (null when unknown);
        - ``added``: the memories this Add created that a read still shows (`_created`),
          a new memory an update or correction wrote included. The bench ``added`` also
          counts a memory the same Add's complete forget removed, such as the record of a
          forget request; null when the items are unknown;
        - ``forgets``: one entry per in-chat forget request the Add carried out: ``request`` (the
          user's words as extracted), ``status`` (``forgotten``, ``nothing_matched`` or
          ``pending``), ``forgotten`` (memories), ``withheld_spans`` (sentences), ``bundled``, and
          ``earlier``; empty when none. No forgotten memory's text is ever included.
    """
    counts = _object(receipt.get("outcome")) or {}
    unresolved, rejected = _count(counts.get("unresolved")), _count(counts.get("rejected"))
    partial = unresolved > 0 or rejected > 0
    reasons = None if items is None else _reasons(items)
    added = None if items is None else len({
        memory for item in items for memory in _created(item)})
    return {
        "status": "partial" if partial else "stored", "unresolved": unresolved,
        "rejected": rejected, "reasons": reasons if partial else {},
        "items": None if items is None else [
            {"category": item.get("category"), "reason_code": item.get("reason_code")}
            for item in items],
        "added": added, "forgets": [dict(entry) for entry in forgets],
    }


def _created(item: Mapping[str, object]) -> list[object]:
    """The memories one add-item created, of those a read still shows.

    - An ``added`` item and a ``history_corrected`` one (a new historical memory beside the
      corrected version) created every memory they name.
    - Any other ``corrected`` item (``updated``, ``corrected``, ``retracted``) that ends a
      memory created each memory it names other than the ones it ends: a new current
      memory beside the old one. One naming no ended memory revised its own
      memory in place (a correction of a memory with the same primary) and created none.

    A ``corrected`` item that merged into or supplemented another memory while ending one,
    or that revised its memory in place while ending or contradicting another, reads as
    created too: add-items does not tell those writes apart from a new memory.
    """
    memories = cast(list[object], item.get("memory_ids") or [])
    if item.get("category") == "added" or item.get("reason_code") == "history_corrected":
        return memories
    targets = cast(list[object], item.get("target_memory_ids") or [])
    if item.get("category") != "corrected" or not targets:
        return []
    return [memory for memory in memories if memory not in targets]


def add_items(document: object) -> list[dict[str, object]] | None:
    """The items of one completed add-items page.

    Args:
        document: The decoded ``/memory/add-items`` response.

    Returns:
        Each item's ``category``, ``reason_code``, ``memory_ids`` and ``target_memory_ids``;
        None when the document is not a completed page.
    """
    data = _object((_object(document) or {}).get("data"))
    if data is None or data.get("state") != "completed":
        return None
    raw = data.get("items")
    if not isinstance(raw, list):
        return None
    found: list[dict[str, object]] = []
    for entry in cast(list[object], raw):
        item = _object(entry)
        if item is None:
            return None
        ids, targets = item.get("memory_ids"), item.get("target_memory_ids")
        found.append({"category": item.get("category"), "reason_code": item.get("reason_code"),
                      "memory_ids": list(cast(list[object], ids)) if isinstance(ids, list)
                      else [],
                      "target_memory_ids": list(cast(list[object], targets))
                      if isinstance(targets, list) else []})
    return found


def item_reasons(document: object) -> dict[str, int] | None:
    """Count the reason codes of the rejected and unresolved items of one add-items page.

    Args:
        document: The decoded ``/memory/add-items`` response.

    Returns:
        The counts by reason code; None when the document is not a completed page.
    """
    items = add_items(document)
    return None if items is None else _reasons(items)


def plain_address(address: str) -> str:
    """An address as the proof names it: lower case, no brackets, IPv4-mapped as IPv4.

    Both forms of an IPv4-mapped address count: ``::ffff:127.0.0.1`` and ``::ffff:7f00:1``
    (as a URL parser writes it) are ``127.0.0.1``.
    """
    text = address.strip("[]").lower()
    if ":" in text:
        try:
            mapped = ipaddress.IPv6Address(text).ipv4_mapped
        except ValueError:
            mapped = None
        if mapped is not None:
            return str(mapped)
    return text


def is_loopback(address: str) -> bool:
    """Whether ``address`` is an IP address of this computer's loopback interface."""
    try:
        return ipaddress.ip_address(plain_address(address)).is_loopback
    except ValueError:
        return False


def listener_proof(secret: str, address: str, port: int, challenge: str) -> str:
    """The answer of ``GET /agent/listener``: what only a holder of the token can give.

    Args:
        secret: The service token.
        address: The address the service listens on (`plain_address`), as the client
            connected to it.
        port: The port the service listens on, as the client connected to it.
        challenge: The client's challenge (hex digits).

    Returns:
        The hex HMAC-SHA256 of ``quotemem-listener-proof``, NUL, the address, NUL, the
        port, NUL and the challenge, keyed by the token.
    """
    message = f"quotemem-listener-proof\0{address}\0{port}\0{challenge}".encode("ascii")
    return hmac.new(secret.encode("ascii"), message, hashlib.sha256).hexdigest()


async def _respond(
    send: Send, status: int, document: dict[str, Any],
    headers: list[tuple[bytes, bytes]] | None = None,
) -> None:
    body = json.dumps(document, sort_keys=True).encode("utf-8")
    await send({"type": "http.response.start", "status": status, "headers": [
        (b"content-type", b"application/json"), (b"content-length", str(len(body)).encode()),
        *(headers or []),
    ]})
    await send({"type": "http.response.body", "body": body})


async def _read_body(receive: Receive, limit: int) -> tuple[bytes, list[Message], bool]:
    """Read up to ``limit`` bytes; returns the body, the messages read and completeness."""
    chunks: list[bytes] = []
    messages: list[Message] = []
    size = 0
    while True:
        message = await receive()
        messages.append(message)
        if message["type"] != "http.request":
            return b"".join(chunks), messages, False
        chunk = cast(bytes, message.get("body", b""))
        chunks.append(chunk)
        size += len(chunk)
        if not message.get("more_body", False):
            return b"".join(chunks), messages, True
        if size > limit:
            return b"".join(chunks), messages, False


class AgentGateway:
    """The ASGI front of one agent service.

    Args:
        credentials: The service token; ``/agent/status`` requires it.
        state: The shared status record.
        default_user_id: The user of requests that name none.
        default_top_k: The Search result count when a request names none.
        self_identity: The store's self-identity rule (``ProviderConfig.self_identity``),
            which every Add's normalization applies; None keeps a user message's speaker.
        log: Where request lines go (stdout by default).
    """

    def __init__(
        self, credentials: HTTPCredentials, state: AgentState, *, default_user_id: str,
        default_top_k: int, store: AgentStore | None = None, spend: SpendGuard | None = None,
        self_identity: SelfIdentityRule | None = None,
        log: Callable[[dict[str, object]], None] = write_line,
    ) -> None:
        self.credentials = credentials
        self.state = state
        self.default_user_id = default_user_id
        self.default_top_k = default_top_k
        self.self_identity: SelfIdentityRule | None = self_identity
        self.store = store
        self.spend = spend
        self.log = log
        self.inner: ASGIApp | None = None
        # Keyed by the service token: stable across restarts, not reversible by
        # hashing guessed user ids without the token.
        self._user_key = hashlib.sha256(
            b"quotemem-user-log\0" + credentials.secret.encode("ascii"),
        ).digest()

    def _authenticated(self, scope: Scope) -> bool:
        try:
            self.credentials.verify(cast(list[tuple[bytes, bytes]], scope["headers"]))
        except QuoteMemError:
            return False
        return True

    async def _completed(self, requests: list[tuple[str, str, str]]) -> frozenset[
            tuple[str, str, str]]:
        """Which of ``requests`` have a stored result; none while the store is closed."""
        store = self.store
        if not requests or store is None or self.inner is None or not store.open:
            return frozenset()
        try:
            return await asyncio.to_thread(store.completed_adds, requests)
        except QuoteMemError:
            return frozenset()

    async def _status(self) -> dict[str, object]:
        self.state.adds_settled(await self._completed(self.state.background_adds()))
        self.state.adds_settled(self.state.followups_refused())
        document = self.state.snapshot()
        # Background completions running, on their own and never within in_flight.
        cast(dict[str, object], document["adds"])["completing"] = self.state.followups_running()
        cast(dict[str, object], document["adds"])["overtaken"] = self.state.followups_overtaken()
        if self.spend is not None:
            document["spend"] = self.spend.snapshot()
        store = self.store
        if store is not None and self.inner is not None and store.open:
            try:
                details = await asyncio.to_thread(store.details)
            except QuoteMemError as error:
                details = {"unavailable": error.code}
            document["store"] = {**cast(dict[str, object], document["store"]), **details}
        return document

    async def _admin(self, path: str, receive: Receive, send: Send) -> None:
        store = self.store
        if store is None or self.inner is None or not store.open:
            await _respond(send, 503, {"code": "service_unhealthy",
                                       "message": "The store is not open"},
                           [(b"retry-after", b"5")])
            return
        body, _, complete = await _read_body(receive, MAX_ADMIN_BODY)
        try:
            request = cast(object, json.loads(body)) if complete else None
        except ValueError:
            request = None
        user_id = cast(dict[str, object], request).get("user_id") if isinstance(
            request, dict) else None
        if not isinstance(request, dict) or not isinstance(user_id, str) or not user_id:
            await _respond(send, 400, {"code": "invalid_request",
                                       "message": "A JSON body with user_id is required"})
            return
        fields = cast(dict[str, object], request)
        try:
            if path == "/agent/admin/delete-user":
                if fields.get("confirm") != user_id:
                    await _respond(send, 400, {
                        "code": "confirmation_required",
                        "message": "Repeat the user id in confirm to erase that user",
                    })
                    return
                result = await asyncio.to_thread(store.delete_user, user_id)
                self.state.adds_dropped(user_id)
            else:
                action = fields.get("action", "inspect")
                if action not in _REPAIR_ACTIONS:
                    await _respond(send, 400, {"code": "invalid_request",
                                               "message": "action is inspect, resume or skip"})
                    return
                result = await asyncio.to_thread(store.repair, user_id, cast(Any, action))
        except QuoteMemError as error:
            status = 409 if error.code == "operation_conflict" else (
                503 if error.code == "service_unhealthy" else 400 if error.code in {
                    "invalid_request", "scope_denied"} else 500)
            await _respond(send, status, {"code": error.code, "message": str(error)})
            return
        await _respond(send, 200, result)

    async def _list(self, receive: Receive, send: Send, record: RequestLog) -> None:
        """``POST /memory/list``: one page of a user's memories (authenticated by the caller)."""
        from quotemem.api.memory_list import parse_list_request

        store = self.store
        if store is None or self.inner is None or not store.open:
            await _respond(send, 503, {"code": "service_unhealthy",
                                       "message": "The store is not open"},
                           [(b"retry-after", b"5")])
            return
        body, _, complete = await _read_body(receive, MAX_LIST_BODY)
        try:
            if not complete:
                raise QuoteMemError("invalid_request", "Memory list request is invalid")
            request = parse_list_request(body)
            record.user = self.user_hash(request.user_id)
            encoded = await asyncio.to_thread(store.list_memories, request)
        except Exception as error:
            mapped = public_error(error)
            await _respond(send, mapped.status_code, mapped.body.model_dump(mode="json"), [
                (name.lower().encode("ascii"), value.encode("ascii"))
                for name, value in mapped.headers.items()])
            return
        await send({"type": "http.response.start", "status": 200, "headers": [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(encoded)).encode())]})
        await send({"type": "http.response.body", "body": encoded})

    async def _defer(self, spend: SpendGuard, send: Send) -> None:
        """Answer an Add 429 ``spend_cap_reached`` while the day's cap is reached."""
        spend.deferred()
        self.state.add_refused("deferred")
        retry = spend.retry_after_seconds()
        await _respond(send, 429, {
            "code": "spend_cap_reached", "status": "deferred", "retry_after_seconds": retry,
            "message": "The daily provider spend cap is reached; retry after the reset",
        }, [(b"retry-after", str(retry).encode())])

    async def _refuse_add(self, scope: Scope, body: bytes, send: Send) -> bool:
        """Answer an Add the gateway defers or refuses; False forwards it.

        While the spend cap is reached, an authenticated Add whose result is already
        stored, or that this gateway acknowledged 202 and still counts in the background,
        is forwarded: the application answers its exact resend from the stored result,
        or 202 while it completes, without starting a new extraction (and refuses a
        changed body as before). Only a new Add is deferred.
        """
        if self.spend is not None and self.spend.exhausted() and not (
            self._authenticated(scope)
            and (request := add_request(body)) is not None
            and (self.state.in_background(request)
                 or request in await self._completed([request]))
        ):
            await self._defer(self.spend, send)
            return True
        if self.store is not None and self.store.deleting_any():
            try:
                user_id = cast(dict[str, object], json.loads(body)).get("user_id")
            except (ValueError, AttributeError):
                user_id = None
            if isinstance(user_id, str) and self.store.deleting(user_id):
                self.state.add_refused("deleting")
                await _respond(send, 409, {
                    "code": "user_deleting", "message": "This user's memory is being erased",
                })
                return True
        return False

    def serving(self, app: ASGIApp) -> ASGIApp:
        """Put ``app`` behind this gateway and report the serving phase."""
        self.inner = app
        self.state.enter("serving")
        return self

    async def _listener(self, scope: Scope, send: Send) -> None:
        """Answer a client's challenge with ``listener_proof`` for this service's port."""
        challenges = [value for name, value in cast(list[tuple[bytes, bytes]], scope["headers"])
                      if name.lower() == b"x-quotemem-challenge"]
        challenge = challenges[0].decode("latin-1") if len(challenges) == 1 else ""
        server = cast(tuple[str, int] | None, scope.get("server"))
        client = cast(tuple[str, int] | None, scope.get("client"))
        # Only a client on this computer checks a listener; a service others can reach
        # gives no proof, so no one elsewhere can test guesses of its token against one.
        if server is None or client is None or not is_loopback(server[0]) \
                or not is_loopback(client[0]):
            await _respond(send, 404, {"code": "not_found", "message": "Not found"})
            return
        if not _CHALLENGE.fullmatch(challenge):
            await _respond(send, 400, {"code": "invalid_request",
                                       "message": "x-quotemem-challenge must be 32 to 128 "
                                                  "lowercase hex digits"})
            return
        await _respond(send, 200, {"proof": listener_proof(
            self.credentials.secret, plain_address(server[0]), server[1], challenge)})

    def user_hash(self, user_id: str) -> str:
        """The log identity of one user: 16 hex digits of a keyed hash."""
        return hmac.new(self._user_key, user_id.encode("utf-8", errors="surrogatepass"),
                        hashlib.sha256).hexdigest()[:16]

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            inner = self.inner
            if inner is not None:
                await inner(scope, receive, send)
            return
        path, method = cast(str, scope["path"]), cast(str, scope["method"])
        if path == "/agent/ready" and method == "GET":
            ready = self.inner is not None and self.state.phase == "serving"
            await _respond(send, 200 if ready else 503, {"ready": ready,
                                                       "phase": self.state.phase})
            return
        if path == LISTENER_ROUTE and method == "GET":
            await self._listener(scope, send)
            return
        record = RequestLog(method, path if path in LOGGED_ROUTES else "other")
        started = time.monotonic()

        async def observed(message: Message) -> None:
            if message["type"] == "http.response.start" and record.status is None:
                record.status = cast(int, message["status"])
            await send(message)

        try:
            await self._dispatch(scope, receive, observed, record)
        finally:
            try:
                self.log({
                    "event": "request", "method": record.method, "route": record.route,
                    "user": record.user, "status": record.status,
                    "duration_ms": round((time.monotonic() - started) * 1000, 1),
                    "phase": self.state.phase, "counts": record.counts,
                })
            except Exception:
                pass

    async def _dispatch(
        self, scope: Scope, receive: Receive, send: Send, record: RequestLog,
    ) -> None:
        inner = self.inner
        path, method = cast(str, scope["path"]), cast(str, scope["method"])
        if path in {"/agent/status", "/agent/admin/delete-user", "/agent/admin/repair"}:
            if (method != "GET") if path == "/agent/status" else (method != "POST"):
                await _respond(send, 405, {"code": "method_not_allowed",
                                           "message": "Method not allowed"})
                return
            if not self._authenticated(scope):
                await _respond(send, 401, {"code": "authentication_failed",
                                           "message": "Authentication failed"})
                return
            if path == "/agent/status":
                await _respond(send, 200, await self._status())
            else:
                await self._admin(path, receive, send)
            return
        if path == "/memory/list":
            if method != "POST":
                await _respond(send, 405, {"code": "method_not_allowed",
                                           "message": "Method not allowed"})
                return
            if not self._authenticated(scope):
                mapped = public_error(QuoteMemError("authentication_failed",
                                                    "authentication_failed"))
                await _respond(send, mapped.status_code, mapped.body.model_dump(mode="json"))
                return
            await self._list(receive, send, record)
            return
        if path in BENCH_ROUTES:
            await _refuse(send, "invalid_request")
            return
        if inner is None:
            await _respond(send, 503, {
                "code": "service_unhealthy", "phase": self.state.phase,
                "message": "Memory is starting or its providers are unavailable",
            }, [(b"retry-after", b"5")])
            return
        if method != "POST" or path not in _USER_ROUTES:
            await inner(scope, receive, send)
            return
        body, messages, complete = await _read_body(receive, MAX_NORMALIZED_BODY)
        if path in _FORGET_TARGETS:
            routed = forget_route(path, body) if complete else None
            if routed is None:
                await _refuse(send, "invalid_request" if self._authenticated(scope)
                              else "authentication_failed")
                return
        else:
            routed = PUBLIC_ROUTES[path]
        # From here on the request is the bench route's: the public name is only logged.
        path = routed
        scope = {**scope, "path": path, "raw_path": path.encode("ascii")}
        if complete:
            body = self._normalized(path, body, record)
            if path == "/add" and await self._refuse_add(scope, body, send):
                return
            replay: list[Message] = [{"type": "http.request", "body": body, "more_body": False}]
            scope = {**scope, "headers": [
                (name, value) for name, value in cast(list[tuple[bytes, bytes]], scope["headers"])
                if name.lower() != b"content-length"
            ] + [(b"content-length", str(len(body)).encode())]}
        else:
            if (path == "/add" and messages[-1]["type"] == "http.request"
                    and self.spend is not None and self.spend.exhausted()):
                # Too long to normalize and read: it can only be a new Add.
                await self._defer(self.spend, send)
                return
            replay = messages

        async def replayed() -> Message:
            if replay:
                return replay.pop(0)
            return await receive()

        counted = {"/add": "add", "/search": "search"}.get(path)
        status: list[int] = []

        async def observed(message: Message) -> None:
            if message["type"] == "http.response.start":
                status.append(cast(int, message["status"]))
            await send(message)

        adapted = path in {"/search", "/add", "/memory/history", "/memory/why"}
        request = add_request(body) if counted == "add" and complete else None
        # A resend of an Add acknowledged 202 is already counted in flight.
        started = counted == "add" and not self.state.in_background(request)
        if started:
            self.state.add_started()
        try:
            if adapted:
                await self._adapted(scope, replayed, observed, path, record,
                                    body if complete else None)
            else:
                await inner(scope, replayed, observed)
        finally:
            if started:
                self.state.add_finished()
            if request is not None and status and status[0] == 202:
                self.state.add_accepted(request)
            elif request is not None and status and status[0] == 200:
                self.state.adds_settled((request,))
            if status and counted is not None:
                self.state.responded(counted, status[0])

    def _normalized(self, path: str, body: bytes, record: RequestLog) -> bytes:
        """Normalize an Add or Search body and note its user and counts."""
        if path == "/add":
            body, notes = normalize_add(body, default_user_id=self.default_user_id,
                                        self_identity=self.self_identity)
            self.state.normalized(notes)
            record.counts.update({
                "dropped_messages": notes.dropped_messages,
                "dropped_user_speakers": notes.dropped_user_speakers,
                "redacted_spans": notes.redacted_spans,
            })
        elif path == "/search":
            body = normalize_search(body, default_user_id=self.default_user_id,
                                    default_top_k=self.default_top_k)
        try:
            raw: object = json.loads(body)
        except (ValueError, RecursionError):
            return body
        if isinstance(raw, dict):
            request = cast(dict[str, Any], raw)
            user = request.get("user_id")
            if isinstance(user, str):
                record.user = self.user_hash(user)
            if path == "/add" and isinstance(request.get("messages"), list):
                record.counts["messages"] = len(cast(list[Any], request["messages"]))
            top_k = request.get("top_k")
            if path == "/search" and type(top_k) is int:
                record.counts["top_k"] = top_k
        return body

    async def _adapted(
        self, scope: Scope, receive: Receive, send: Send, path: str, record: RequestLog,
        request: bytes | None,
    ) -> None:
        """Run the inner application, then adapt and count its complete response.

        A successful ``/add`` (200) whose request was read whole also gets its ``agent``
        report (`add_report`); its response is held until the report is ready.
        """
        inner = self.inner
        assert inner is not None
        start: Message | None = None
        chunks: list[bytes] = []
        size = 0
        passing = False

        async def captured(message: Message) -> None:
            nonlocal start, size, passing
            if passing:
                await send(message)
            elif message["type"] == "http.response.start":
                start = message
            elif message["type"] == "http.response.body":
                chunk = cast(bytes, message.get("body", b""))
                chunks.append(chunk)
                size += len(chunk)
                if size > MAX_ADAPTED_RESPONSE and message.get("more_body", False):
                    # Too large to adapt: send what is held and stream the rest.
                    passing = True
                    assert start is not None
                    await send(start)
                    await send({"type": "http.response.body", "body": b"".join(chunks),
                                "more_body": True})
                elif not message.get("more_body", False):
                    assert start is not None
                    status, headers, body = _adapt(
                        path, cast(int, start["status"]),
                        cast(list[tuple[bytes, bytes]], start.get("headers", [])),
                        b"".join(chunks), record,
                    )
                    if path == "/add" and status == 200 and request is not None:
                        headers, body = await self._reported(scope, request, headers, body)
                    await send({**start, "status": status, "headers": headers})
                    await send({"type": "http.response.body", "body": body,
                                "more_body": False})
            else:
                await send(message)

        await inner(scope, receive, captured)


    async def _reported(
        self, scope: Scope, request: bytes, headers: list[tuple[bytes, bytes]], body: bytes,
    ) -> tuple[list[tuple[bytes, bytes]], bytes]:
        """Add the ``agent`` report to one successful ``/add`` response body.

        A body that is not one successful JSON receipt is returned as it was.
        """
        receipt: dict[str, object] | None = None
        if len(body) <= MAX_ADD_RESPONSE:
            try:
                receipt = _object(json.loads(body))
            except ValueError:
                receipt = None
        if receipt is None or receipt.get("success") is not True:
            return headers, body
        items = await self._items(scope, request)
        forgets = await self._forgets(request)
        report = add_report(receipt, items, forgets)
        if report["status"] == "partial":
            self.state.partial_add()
        reported = json.dumps({**receipt, "agent": report}, ensure_ascii=False,
                              separators=(",", ":")).encode("utf-8")
        return [
            (name, value) for name, value in headers if name.lower() != b"content-length"
        ] + [(b"content-length", str(len(reported)).encode())], reported

    async def _forgets(self, request: bytes) -> list[dict[str, object]]:
        """The Add's in-chat forget report (``AgentState.add_forgets``); empty on failure."""
        try:
            return await asyncio.to_thread(self.state.add_forgets, request)
        except Exception as error:  # noqa: BLE001 - the report is best effort; the Add succeeded
            write_line({"event": "add_forgets_unread", "error": type(error).__name__})
            return []

    async def _items(self, scope: Scope, request: bytes) -> list[dict[str, object]] | None:
        """Read one completed Add's items from ``/memory/add-items``; None on failure.

        The read is a request to the inner application of its own. Its ``receive`` hands
        over the query, then waits until the response is complete before it reports a
        disconnect: the memory reads watch for a disconnect while they run and drop the
        response on one, so a disconnect sent right after the query left every partial
        Add's ``reasons`` null.
        """
        inner = self.inner
        assert inner is not None
        try:
            fields = _object(json.loads(request))
        except ValueError:
            return None
        if fields is None:
            return None
        ids = {key: fields.get(key) for key in ("user_id", "session_id", "request_id")}
        if not all(isinstance(value, str) for value in ids.values()):
            return None
        query = json.dumps({"schema_version": 1, **ids}).encode("utf-8")
        headers = [
            (name, value) for name, value in cast(list[tuple[bytes, bytes]], scope["headers"])
            if name.lower() in (b"authorization", b"host")
        ] + [(b"content-type", b"application/json"),
             (b"content-length", str(len(query)).encode())]
        sub: Scope = {**scope, "path": "/memory/add-items", "raw_path": b"/memory/add-items",
                      "query_string": b"", "headers": headers}
        pending: list[Message] = [{"type": "http.request", "body": query, "more_body": False}]
        answer: list[Message] = []
        done = asyncio.Event()

        async def receive() -> Message:
            if pending:
                return pending.pop(0)
            await done.wait()
            return {"type": "http.disconnect"}

        async def collect(message: Message) -> None:
            answer.append(message)
            if message["type"] == "http.response.body" and not message.get("more_body", False):
                done.set()

        try:
            await inner(sub, receive, collect)
        except Exception:  # noqa: BLE001 - the report is best effort; the Add already succeeded
            return None
        finally:
            done.set()
        starts = [message for message in answer if message["type"] == "http.response.start"]
        if not starts or starts[0]["status"] != 200:
            return None
        data = b"".join(cast(bytes, message.get("body", b"")) for message in answer
                        if message["type"] == "http.response.body")
        try:
            return add_items(json.loads(data))
        except ValueError:
            return None


def _adapt(
    path: str, status: int, headers: list[tuple[bytes, bytes]], body: bytes,
    record: RequestLog,
) -> tuple[int, list[tuple[bytes, bytes]], bytes]:
    """Adapt one complete response for agent clients and count it."""
    try:
        raw: object = json.loads(body)
    except (ValueError, RecursionError):
        return status, headers, body
    if not isinstance(raw, dict):
        return status, headers, body
    document = cast(dict[str, Any], raw)
    changed = False
    if path == "/search" and status == 200 and isinstance(document.get("data"), list):
        hits = cast(list[Any], document["data"])
        record.counts["hits"] = len(hits)
        for hit in hits:
            if not isinstance(hit, dict):
                continue
            fields = cast(dict[str, Any], hit)
            content = fields.get("content")
            if isinstance(content, str) and (restored := original_text(content)) != content:
                fields["content"] = restored
                changed = True
    elif path == "/add" and status < 300 and isinstance(document.get("outcome"), dict):
        for name, value in cast(dict[str, Any], document["outcome"]).items():
            if type(value) is int:
                record.counts[name] = value
    elif path in {"/memory/history", "/memory/why"} and status == 403 \
            and document.get("code") == "scope_denied":
        status, changed = 404, True
        document = {"code": "memory_not_found", "retryable": False,
                    "detail": "memory_not_found", "message": NOT_A_MEMORY}
    if not changed:
        return status, headers, body
    adapted = json.dumps(document, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return status, [
        (name, value) for name, value in headers if name.lower() != b"content-length"
    ] + [(b"content-length", str(len(adapted)).encode())], adapted
