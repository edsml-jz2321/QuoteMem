"""List one user's memories for the agent profile: ``POST /memory/list`` and ``memories``.

A plugin user has no other place to see everything the service remembers about them.
`MemoryListing` reads the scope through the same registered Search pin as an
interactive Search (`StoreReadModel.begin_search` / ``pin_search``), lists every memory
a plain read admits (`quotemem.core.recall.agent_listing`), and gives each the status Search's
``Status:`` line gives it, read from the whole store (`quotemem.core.recall_agent.standing`):
``current`` (still holds, including a past event that is still true), ``past`` (replaced,
retracted or ended) or ``possibly_outdated`` (a newer memory contradicts or restates it,
or a newer user message may correct it, and no memory settled it).

Nothing forgotten is listed: a forgotten memory is gone from the read model, and what an
agent Search never shows (a memory quoting a marked sentence or a forget request, or one
the forget shadow matches) is dropped the same way Search drops it; withheld quote bytes
are never readable (``p_span``), so a quote that is entirely withheld is not shown, and the
counts count only what is listed. The response is encoded inside the guarded callback, so
a privacy commit that revokes the pin fails the read instead of releasing withdrawn bytes.

Order: ``updated_at_ms`` descending, then ``memory_id`` ascending. The cursor is the last
listed memory's key, so a page boundary stays stable while memories are added. A user
the store has never seen gets an empty list, never an error, and is not registered.
"""

from __future__ import annotations

import base64
import binascii
import json
from collections.abc import Callable
from typing import Annotated, Final, Literal

from pydantic import ConfigDict, Field, ValidationError, field_validator

from quotemem.api.bench_search import agent_context, quote_speakers, scope_withdrawn
from quotemem.core.delivery import delivery_projections
from quotemem.core.errors import QuoteMemError
from quotemem.core.identity import scope_id
from quotemem.core.query import ReturnedMemoryV1, ScopeV1
from quotemem.core.recall import ListedMemory, agent_listing
from quotemem.core.recall_agent import SHADOW_PURPOSE, Status
from quotemem.core.recall_snapshot import RecallSnapshot
from quotemem.core.technical_types import ExternalId
from quotemem.providers.auth import AuthProvider, AuthRequest
from quotemem.store.metadata import StrictModel
from quotemem.store.read_model import StoreReadModel
from quotemem.store.writer import Writer

MAX_LIST_LIMIT: Final = 200
DEFAULT_LIST_LIMIT: Final = 50
MAX_SHADOW_DIGESTS: Final = 200_000
STATUSES: Final[tuple[Status, ...]] = ("current", "past", "possibly_outdated")

type StatusFilter = Literal["current", "past", "possibly_outdated", "all"]


class MemoryListRequestV1(StrictModel):
    """``POST /memory/list``: one page of a user's memories."""

    model_config = ConfigDict(frozen=True, hide_input_in_errors=True)
    user_id: Annotated[ExternalId, Field(repr=False)]
    status: StatusFilter = "all"
    limit: Annotated[int, Field(ge=1, le=MAX_LIST_LIMIT)] = DEFAULT_LIST_LIMIT
    cursor: Annotated[str | None, Field(max_length=512)] = None

    @field_validator("user_id")
    @classmethod
    def valid_user(cls, value: str) -> str:
        """Keep the exact user id after checking it is valid Unicode."""
        try:
            value.encode("utf-8")
        except UnicodeEncodeError:
            raise ValueError("The user id must contain valid Unicode") from None
        return value

    @property
    def scope(self) -> ScopeV1:
        """The caller's exact user scope."""
        return ScopeV1(scope_kind="user", scope_id=scope_id("user", self.user_id))


class ListedQuoteV1(StrictModel):
    """One quoted line: who said it, when (null when the source had no time) and what."""

    model_config = ConfigDict(frozen=True)
    role: str | None
    speaker: str | None = None
    said_at_ms: int | None
    text: str


class ListedMemoryV1(StrictModel):
    """One listed memory."""

    model_config = ConfigDict(frozen=True)
    memory_id: str
    statement: str
    status: Status
    kind: str
    created_at_ms: int
    updated_at_ms: int
    quotes: list[ListedQuoteV1]
    supersedes: list[str]
    superseded_by: str | None


class StatusCountsV1(StrictModel):
    """How many listed memories have each status (every status, whatever the filter)."""

    model_config = ConfigDict(frozen=True)
    current: int = 0
    past: int = 0
    possibly_outdated: int = 0


class MemoryListV1(StrictModel):
    """``POST /memory/list`` response."""

    model_config = ConfigDict(frozen=True)
    schema_version: Literal[1] = 1
    memories: list[ListedMemoryV1]
    next_cursor: str | None
    counts: StatusCountsV1


def parse_list_request(body: bytes) -> MemoryListRequestV1:
    """Parse one ``/memory/list`` body; ``invalid_request`` for any body it does not accept."""
    try:
        return MemoryListRequestV1.model_validate_json(body)
    except ValidationError:
        raise QuoteMemError("invalid_request", "Memory list request is invalid") from None


def encode_cursor(updated_at_ms: int, memory_id: str) -> str:
    """The opaque cursor after one listed memory."""
    raw = json.dumps([updated_at_ms, memory_id], separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def decode_cursor(cursor: str) -> tuple[int, str]:
    """The (updated_at_ms, memory_id) key a cursor names; ``invalid_request`` otherwise."""
    try:
        raw = base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4))
        value = json.loads(raw.decode("utf-8"))
    except (binascii.Error, ValueError, UnicodeDecodeError):
        raise QuoteMemError("invalid_request", "The cursor is invalid") from None
    if not (isinstance(value, list) and len(value) == 2 and type(value[0]) is int  # pyright: ignore[reportUnknownArgumentType]
            and isinstance(value[1], str)):
        raise QuoteMemError("invalid_request", "The cursor is invalid")
    return value[0], value[1]


def _order(memory: ListedMemory) -> tuple[int, str]:
    return -memory.document.row.updated_at, memory.document.memory_id


def listing_page(
    root: RecallSnapshot,
    listed: tuple[ListedMemory, ...],
    request: MemoryListRequestV1,
    *,
    now: int,
    speakers: Callable[[tuple[ListedMemory, ...]], dict[tuple[str, int, int], str | None]],
) -> MemoryListV1:
    """One page of a listing, its quotes re-derived by Search's delivery projection.

    Args:
        root: The pin's snapshot the listing read.
        listed: Every listed memory with its status.
        request: The page request.
        now: The read's trusted clock.
        speakers: Resolves the page's quoted spans' speakers.

    Returns:
        The page, its cursor and the counts of every listed memory.

    Raises:
        QuoteMemError: ``invalid_request`` for a bad cursor; the projection's refusals.
    """
    ordered = sorted(listed, key=_order)
    counts = {status: 0 for status in STATUSES}
    for memory in ordered:
        counts[memory.status] += 1
    selected = [memory for memory in ordered
                if request.status == "all" or memory.status == request.status]
    if request.cursor is not None:
        updated, memory_id = decode_cursor(request.cursor)
        selected = [memory for memory in selected if _order(memory) > (-updated, memory_id)]
    page = tuple(selected[:request.limit])
    more = len(selected) > request.limit
    shown = {memory.document.memory_id for memory in ordered}
    successors: dict[str, ListedMemory] = {}
    for memory in ordered:  # Newest first: the newest successor wins.
        for link in memory.document.row.links:
            if link.link_type == "supersedes" and link.target_key in shown:
                successors.setdefault(link.target_key, memory)
    scope = request.scope
    projections = delivery_projections(
        root, scope=scope, returned=tuple(ReturnedMemoryV1(
            kind="memory", object_id=memory.document.memory_id,
            revision_id=memory.document.row.current_revision_id,
            content_hash=memory.document.row.content_hash, rank=rank, score=0.0,
        ) for rank, memory in enumerate(page, 1)),
        mode="plain", at_time=now, speakers=speakers(page),
    ) if page else ()
    memories: list[ListedMemoryV1] = []
    for memory, projection in zip(page, projections, strict=True):
        presented = projection.memory
        if presented is None:
            raise QuoteMemError("service_unhealthy", "A listed memory has no presentation")
        quotes: list[ListedQuoteV1] = []
        for line in presented.quotes.lines:
            source = root.get_source(line.evidence_id)
            if source is None or not line.quote:
                continue
            quotes.append(ListedQuoteV1(role=line.role, speaker=line.speaker,
                                        said_at_ms=source.source_observed_at, text=line.quote))
        row = memory.document.row
        successor = successors.get(memory.document.memory_id)
        memories.append(ListedMemoryV1(
            memory_id=memory.document.memory_id, statement=presented.statement,
            status=memory.status, kind=presented.kind, created_at_ms=row.created_at,
            updated_at_ms=row.updated_at, quotes=quotes,
            supersedes=sorted(link.target_key for link in row.links
                              if link.link_type == "supersedes" and link.target_key in shown),
            superseded_by=None if successor is None else successor.document.memory_id,
        ))
    last = page[-1].document if page else None
    return MemoryListV1(
        memories=memories,
        next_cursor=encode_cursor(last.row.updated_at, last.memory_id)
        if more and last is not None else None,
        counts=StatusCountsV1(**counts),
    )


class MemoryListing:
    """List one user's memories through the registered Search pin (see the module docstring).

    Args:
        writer: The sole Writer, read only.
        publisher: The read model registered for ``writer``.
        auth: Resolves the caller's exact scope.
        clock: Trusted wall clock in epoch milliseconds.
    """

    def __init__(
        self, writer: Writer, publisher: StoreReadModel, auth: AuthProvider, *,
        clock: Callable[[], int],
    ) -> None:
        if StoreReadModel.get(writer) is not publisher:
            raise QuoteMemError("binding_mismatch",
                                "Listing requires the Writer's registered publisher")
        self._writer, self._publisher, self._auth, self._clock = writer, publisher, auth, clock
        self._shadow_keys: dict[str, str] = {}

    def _shadow_key(self, term: str) -> str:
        key = self._shadow_keys.get(term)
        if key is None:
            if len(self._shadow_keys) >= MAX_SHADOW_DIGESTS:
                self._shadow_keys = {}
            key = self._shadow_keys[term] = self._writer.bind_content(SHADOW_PURPOSE, term)
        return key

    def consume(self, request: MemoryListRequestV1, consumer: Callable[[MemoryListV1], None],
                ) -> None:
        """Deliver one page to ``consumer`` while the scope's pin is held.

        Raises:
            QuoteMemError: ``invalid_request`` for a bad cursor; the authorization, pin and
                projection refusals of a Search (``index_rebuilding`` while a forget or
                erasure is unfinished).
        """
        if request.cursor is not None:
            decode_cursor(request.cursor)
        scope = request.scope
        if self._writer.get_control(f"scope:{scope.scope_id}") is None:
            # Never registered: nothing to list, and a listing registers nothing.
            consumer(MemoryListV1(memories=[], next_cursor=None, counts=StatusCountsV1()))
            return
        grant = self._auth.resolve(AuthRequest(requested_scope=scope, now_ms=self._clock()))
        context = self._publisher.begin_search(grant, query_time=self._clock())
        now = context.query_time
        with self._publisher.pin_search(context) as guard:

            def operation(root: RecallSnapshot) -> None:
                listed = agent_listing(
                    root, scope_kind=scope.scope_kind, scope_id=scope.scope_id, now=now,
                    context=agent_context(self._writer, scope, self._shadow_key,
                                          self._writer.bind_content),
                    cache=self._publisher.recall_corpora(root),
                    withdrawn=scope_withdrawn(self._writer, scope),
                )
                consumer(listing_page(
                    root, listed, request, now=now,
                    speakers=lambda page: quote_speakers(
                        self._writer, root, scope,
                        [memory.document.snapshot for memory in page], now=now),
                ))

            guard.consume_search(context, operation)

    def encoded(self, request: MemoryListRequestV1) -> bytes:
        """One page as the JSON bytes ``/memory/list`` answers, encoded under the pin."""
        body: list[bytes] = []
        self.consume(request, lambda page: body.append(
            page.model_dump_json().encode("utf-8")))
        if len(body) != 1:
            raise QuoteMemError("service_unhealthy", "The listing did not deliver one page")
        return body[0]
