"""History v3 (E.2.2): one memory's recorded past, and the guarded reads it shares.

A memory's past spans two authorities. A ``merge``, ``revise``, ``requote`` or
``erase`` appends an immutable revision to one Item, while an update,
correction or retraction that creates a new Item ends the old one behind a
``supersedes`` link on the new Item's revision, so the earlier wording lives
under a different ``memory_id``. History joins both into one timeline, newest
first by ``recorded_at`` and then revision id, and says how each version was
reached. A ``historical`` Item (C.9.5a) is listed under the version it
corrects, and its own History is its version list with its correction targets
as ids.

Every Item the walk reaches is a separate memory, so its exact scope is proved
again at each hop instead of being inherited from the Item the caller named
(INV O22). Forget deletes an Item with every link that targets it, so a
forgotten older Item is simply absent from the walk and leaves no gap marker;
a named Item that is forgotten, never existed or belongs to another scope is
refused with the same ``scope_denied``.

Content passes the D.2 predicates of `aml.core.readability` at the read's
clock. A version that fails P-VER, or that is at or before a ``privacy_erase``
version of its Item, is redacted: its ids, links and recording time stay and
its content goes (``statement``, ``kind``, validity and quotes). Quotes are the
version's P-QUOTE spans with their usable group index.

Reads go through `MemoryRows`, inside one pinned guard's scoped callback
(`MemoryReads`): every storage read first rechecks the guard, everything read
is recorded and rechecked before delivery, so a timeline is never assembled
from two different states of the store, and the result is encoded inside the
callback. Why (E.2.3) reads through the same entry.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated, Any, Final, Literal, NamedTuple, Protocol, Self, cast

from pydantic import (
    BeforeValidator,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    model_validator,
)

from aml.core.entities import (
    REVISION_REASONS,
    MemoryKind,
    MemorySnapshotV2,
    QuoteSpanV1,
    RevisionOp,
    RevisionReason,
    SourceRole,
    SupersedeRelation,
)
from aml.core.errors import AMLError
from aml.core.identity import ULID
from aml.core.query import ScopeV1
from aml.core.readability import Readability
from aml.core.semantic import HistoricalTimestamp, SystemTimestamp, TimeGranularity
from aml.ingestion.authorization import check_add_grant
from aml.providers.auth import AuthProvider, AuthRequest, PrincipalGrant
from aml.store.domain import (
    CorrectsHistoryLinkRowV2,
    EvidenceRowV1,
    ItemRowV2,
    LinkRowV2,
    RevisionRowV2,
    SupersedesLinkRowV2,
)
from aml.store.indexed import MAX_PAGE_SIZE, IndexedQuery, LinkTargetRows, RevisionLinks
from aml.store.metadata import StrictModel
from aml.store.read_model import SnapshotReadGuard, StoreReadModel
from aml.store.writer import Writer

MAX_HISTORY_LIMIT: Final = 128
"""The most entries one History read returns (E.2.2 ``limit``)."""

MAX_WALK_VERSIONS: Final = 512
"""The most revisions one walk reads; a longer past is reported ``truncated``."""

MAX_ROW_LINKS: Final = 512
"""The most rows one index lookup of a memory read may return before it fails closed."""

HistoryChange = Literal[
    "create", "update", "correct", "retract", "correct_history", "supplement",
    "add_support", "privacy_requote", "privacy_erase",
]
"""What a version changed (E.2.2 #2)."""

ReachedVia = Literal["revision", "supersedes"]
"""How the walk reached a version: inside its Item, or as the head of an Item it ended."""

_CREATE_CHANGES: Final[Mapping[str, HistoryChange]] = {
    "independent": "create", "contradiction": "create", "external_correction": "create",
    "update": "update", "correction": "correct", "retraction": "retract",
    "historical_correction": "correct_history",
}
_REVISE_CHANGES: Final[Mapping[str, HistoryChange]] = {
    "supplement": "supplement", "correction": "correct", "retraction": "retract",
    "rewording": "correct",
}
_OP_CHANGES: Final[Mapping[str, HistoryChange]] = {
    "merge": "add_support", "requote": "privacy_requote", "erase": "privacy_erase",
}
_ULID: Final[TypeAdapter[str]] = TypeAdapter(ULID)
_RECORDED_TIME: Final[TypeAdapter[int]] = TypeAdapter(SystemTimestamp)
_ITEM: Final[TypeAdapter[ItemRowV2]] = TypeAdapter(ItemRowV2)
_REVISION: Final[TypeAdapter[RevisionRowV2]] = TypeAdapter(RevisionRowV2)
_LINK: Final[TypeAdapter[LinkRowV2]] = TypeAdapter(LinkRowV2)
_EVIDENCE: Final[TypeAdapter[EvidenceRowV1]] = TypeAdapter(EvidenceRowV1)


def history_change(op: RevisionOp, reason: RevisionReason) -> HistoryChange:
    """Name what one version changed (E.2.2 #2).

    Args:
        op (RevisionOp): The revision's op.
        reason (RevisionReason): Its reason code.

    Returns:
        HistoryChange: ``create``, ``update``, ``correct``, ``retract``,
            ``correct_history``, ``supplement``, ``add_support``,
            ``privacy_requote`` or ``privacy_erase``.

    Raises:
        AMLError: ``parity_mismatch`` for a pair A.3.2 does not allow.
    """
    if reason not in REVISION_REASONS.get(op, frozenset[str]()):
        raise AMLError("parity_mismatch", "A revision carries a reason its op does not allow")
    if op == "create":
        return _CREATE_CHANGES[reason]
    return _REVISE_CHANGES[reason] if op == "revise" else _OP_CHANGES[op]


def json_tuple(value: object) -> object:
    """Read a JSON array as the tuple a frozen wire value holds, members unchanged."""
    return tuple(cast(list[object], value)) if isinstance(value, list) else value


class WireValue(StrictModel):
    model_config = ConfigDict(frozen=True, hide_input_in_errors=True)


class MemoryQuoteV1(WireValue):
    """One readable quote of a version (E.2 ``Quote``).

    ``byte_start`` and ``byte_end`` are original-message UTF-8 offsets of the
    quoted span, and ``quote`` is exactly those bytes. ``group`` is the index
    of the lowest-numbered usable support group (P-GROUP) that contains the
    span, or None when no usable group does.
    """

    evidence_id: ULID
    byte_start: Annotated[int, Field(ge=0)]
    byte_end: Annotated[int, Field(ge=1)]
    quote: Annotated[str, Field(min_length=1)]
    role: SourceRole | None
    speaker: str | None
    source_time: HistoricalTimestamp | None
    time_granularity: TimeGranularity
    group: Annotated[int, Field(ge=0)] | None

    @model_validator(mode="after")
    def check_bytes(self) -> Self:
        """Require ``byte_end - byte_start`` to be the quote's UTF-8 length (E.2)."""
        if self.byte_end - self.byte_start != len(self.quote.encode("utf-8")):
            raise ValueError("A quote spans exactly its UTF-8 bytes")
        return self


Quotes = Annotated[tuple[MemoryQuoteV1, ...], BeforeValidator(json_tuple)]
"""A version's quotes, in span order."""


class HistoryChainLinkV1(WireValue):
    """One ``supersedes`` relation between two Items (E.2.2 ``ChainLink``).

    ``memory_id`` is the other Item: the target of a version's own link, or the
    successor whose head carries a link to the read Item. ``recorded_at`` is
    when that link row was recorded; ``effective_at`` is set only on updates.
    """

    memory_id: ULID
    relation: SupersedeRelation
    effective_at: HistoricalTimestamp | None
    recorded_at: SystemTimestamp


_ChainLinks = Annotated[tuple[HistoryChainLinkV1, ...], BeforeValidator(json_tuple)]


class CorrectedVersionV1(WireValue):
    """A version a ``historical`` Item corrects: its Item and revision ids (read-Q20)."""

    memory_id: ULID
    revision_id: ULID


class HistoricalCorrectionV1(WireValue):
    """A readable ``historical`` Item that corrects the version it is listed under (C.9.5a).

    ``recorded_at`` is when the correction was recorded: its version 1.
    """

    memory_id: ULID
    statement: str | None
    quotes: Quotes
    recorded_at: SystemTimestamp


class HistoryVersionV3(WireValue):
    """One version in a History timeline (E.2.2 ``Version``).

    A redacted version keeps its ids, links and recording time and has no
    content: ``statement``, ``kind``, validity and granularity are None and
    ``quotes`` is empty.
    """

    memory_id: ULID
    revision_id: ULID
    version: Annotated[int, Field(ge=1)]
    recorded_at: SystemTimestamp
    change: HistoryChange
    reached_via: ReachedVia
    supersedes: _ChainLinks
    contradicts: Annotated[tuple[ULID, ...], BeforeValidator(json_tuple)]
    corrects_history: Annotated[tuple[CorrectedVersionV1, ...], BeforeValidator(json_tuple)]
    historical_corrections: Annotated[
        tuple[HistoricalCorrectionV1, ...], BeforeValidator(json_tuple),
    ]
    redacted: bool
    statement: str | None
    kind: MemoryKind | None
    valid_from: HistoricalTimestamp | None
    valid_to: HistoricalTimestamp | None
    time_granularity: TimeGranularity | None
    quotes: Quotes

    @model_validator(mode="after")
    def check_redaction(self) -> Self:
        """Keep content exactly on versions that are not redacted (E.2.2 #3)."""
        content = (self.statement, self.kind, self.time_granularity)
        if self.redacted and (
            content != (None, None, None) or self.valid_from is not None
            or self.valid_to is not None or self.quotes
        ):
            raise ValueError("A redacted version carries no content")
        if not self.redacted and None in content:
            raise ValueError("A readable version carries its statement, kind and granularity")
        return self


class MemoryHistoryV3(WireValue):
    """One memory's History (E.2.2 ``data``).

    Without ``recorded_as_of``, ``entries`` is the newest-first timeline and
    ``status`` is None. With it, ``entries`` holds at most the one version
    recorded last at or before that instant, and ``status`` says whether it was
    ``found``, provably ``not_recorded``, or left undecided by a ``truncated``
    timeline. ``successors`` are the readable Items whose head ends this one,
    one hop. ``completeness`` is ``complete`` only when the timeline holds the
    whole recorded past the walk reaches.
    """

    memory_id: ULID
    observed_at: SystemTimestamp
    recorded_as_of: SystemTimestamp | None
    status: Literal["found", "not_recorded", "truncated"] | None
    successors: _ChainLinks
    entries: Annotated[tuple[HistoryVersionV3, ...], BeforeValidator(json_tuple)]
    completeness: Literal["complete", "truncated"]

    @model_validator(mode="after")
    def check_selection(self) -> Self:
        """Pair a recorded-time selection with its status and at most one entry."""
        if (self.recorded_as_of is None) != (self.status is None):
            raise ValueError("A status is reported exactly for a recorded-time selection")
        if self.status is not None and len(self.entries) != (1 if self.status == "found" else 0):
            raise ValueError("A recorded-time selection returns exactly its found version")
        return self


class QuoteSpeakers(Protocol):
    """Who speaks a quoted span, from the frozen source labels (B.3a, B.4.1; E.2 ``Quote``).

    Search, History and Why name a quote's speaker the same way: the runtime
    supplies the one resolution that Search delivery uses.
    """

    def speaker(self, rows: MemoryRows, span: QuoteSpanV1) -> str | None:
        """Return the speaker of one quoted span, or None.

        Args:
            rows (MemoryRows): The read's guarded rows; the span's Evidence is
                already recorded in them.
            span (QuoteSpanV1): The quoted span.

        Returns:
            str | None: The attributed speaker; None when the source has none,
                an ``ambiguous`` label governs it, labels and metadata conflict
                or the span crosses a source boundary.
        """
        ...


class _Lookup(NamedTuple):
    """One recorded index lookup: its query, cursor, page size and row bound."""

    query: IndexedQuery
    after: str | None
    limit: int | None
    cap: int


class MemoryRows:
    """Guarded storage reads of one memory read inside a pinned scope's callback.

    Every storage read first rechecks the guard's scoped borrow for this
    thread. Each row and each index lookup is read once, recorded and then
    served from the record, and `finish` reads all of them again, so the
    delivered result is exactly what the store held at that moment: anything
    changed fails the read with ``reception_stale`` instead of mixing two
    states. Rows of another scope or tenant are refused, never returned;
    Evidence of another scope simply fails P-SRC.

    Args:
        writer (Writer): The sole Writer whose read model issued ``guard``.
        scope (ScopeV1): The exact read scope.
        grant (PrincipalGrant): Authority resolved outside AML locks; it is
            revalidated at every hop and before delivery.
        guard (SnapshotReadGuard): The pinned guard whose scoped callback runs
            on this thread.
        clock (Callable[[], int]): Trusted wall clock in epoch milliseconds.
    """

    def __init__(
        self, writer: Writer, scope: ScopeV1, grant: PrincipalGrant,
        guard: SnapshotReadGuard, clock: Callable[[], int],
    ) -> None:
        self.writer, self.scope, self.grant, self.guard = writer, scope, grant, guard
        self.clock = clock
        self._indexed = writer.indexed_reader()
        self._rows: dict[tuple[str, str], tuple[str | None, dict[str, Any] | None]] = {}
        self._pages: dict[str, tuple[_Lookup, list[dict[str, Any]]]] = {}
        self._evidence: dict[str, EvidenceRowV1 | None] = {}

    def authorize(self) -> None:
        """Recheck the guard's scoped borrow and the grant at the trusted clock."""
        self.guard.check_operation(self.writer, self.scope)
        check_add_grant(self.grant, self.scope, self.clock())

    def _read_row(self, kind: str, object_id: str) -> tuple[str | None, dict[str, Any] | None]:
        self.guard.check_operation(self.writer, self.scope)
        raw = self.writer.get_row(kind, object_id)
        return (None if raw is None else self.writer.bind_content("row-state", raw)), raw

    def row(self, kind: str, object_id: str) -> dict[str, Any] | None:
        """Read one stored row once and record it for the delivery recheck.

        Args:
            kind (str): The table.
            object_id (str): The row's id.

        Returns:
            dict[str, Any] | None: A copy of the row, or None when it does not exist.

        Raises:
            AMLError: ``scope_denied`` for a row of another exact scope or
                tenant (Evidence excepted); ``binding_mismatch`` for a row
                stored under another id; or the guard's refusal.
        """
        key = kind, object_id
        if key not in self._rows:
            self._rows[key] = self._read_row(kind, object_id)
        raw = self._rows[key][1]
        if raw is None:
            return None
        if raw.get("object_id") != object_id:
            raise AMLError("binding_mismatch", "A memory read returned another stored identity")
        if "tenant_id" in raw and raw["tenant_id"] != self.writer.tenant_id:
            raise AMLError("scope_denied", "A memory read reached another tenant")
        if kind != "evidence" and "scope_kind" in raw and (
            raw["scope_kind"], raw["scope_id"],
        ) != (self.scope.scope_kind, self.scope.scope_id):
            raise AMLError("scope_denied", "A memory read reached another exact scope")
        return dict(raw)

    def _read_page(self, lookup: _Lookup) -> list[dict[str, Any]]:
        if lookup.limit is not None:
            self.guard.check_operation(self.writer, self.scope)
            return self._indexed.page(lookup.query, after=lookup.after, limit=lookup.limit)
        rows: list[dict[str, Any]] = []
        after = lookup.after
        while True:
            self.guard.check_operation(self.writer, self.scope)
            found = self._indexed.page(lookup.query, after=after, limit=MAX_PAGE_SIZE)
            rows.extend(found)
            if len(rows) > lookup.cap:
                raise AMLError("parity_mismatch", "A memory read exceeds its bounded row set")
            if len(found) < MAX_PAGE_SIZE:
                return rows
            after = found[-1]["object_id"]

    def page(
        self, query: IndexedQuery, *, cap: int = MAX_ROW_LINKS, after: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Read one bounded index lookup once and record it.

        Args:
            query (IndexedQuery): The index lookup.
            cap (int): Without ``limit``, the most rows the whole lookup may return.
            after (str | None): Read only rows after this ``object_id``.
            limit (int | None): Read the one page of at most this many rows;
                None reads every page.

        Returns:
            list[dict[str, Any]]: Copies of the rows in ``object_id`` order.

        Raises:
            AMLError: ``parity_mismatch`` beyond ``cap`` rows; the guard's refusal.
        """
        lookup = _Lookup(query, after, limit, cap)
        key = f"{type(query).__name__}:{query.model_dump_json()}:{after}:{limit}:{cap}"
        if key not in self._pages:
            self._pages[key] = lookup, self._read_page(lookup)
        return [dict(raw) for raw in self._pages[key][1]]

    def item(self, memory_id: str) -> ItemRowV2 | None:
        """Read one Item of the exact scope, or None when it does not exist."""
        raw = self.row("item", memory_id)
        return None if raw is None else _parse(_ITEM, raw, "Item")

    def revision(self, revision_id: str) -> RevisionRowV2 | None:
        """Read one revision, or None when it does not exist."""
        raw = self.row("revision", revision_id)
        return None if raw is None else _parse(_REVISION, raw, "revision")

    def head(self, item: ItemRowV2) -> RevisionRowV2:
        """Read an Item's current head and prove that the Item agrees with it (DM-02).

        Raises:
            AMLError: ``parity_mismatch`` when the head is missing, belongs to
                another Item or differs from the Item's keys.
        """
        head = self.revision(item.current_revision_id)
        if head is None or head.memory_id != item.object_id or (
            item.entity_id, item.statement_digest,
        ) != (head.snapshot.entity_id, head.snapshot.statement_digest):
            raise AMLError("parity_mismatch", "An Item differs from its current head")
        return head

    def previous(self, revision: RevisionRowV2) -> RevisionRowV2 | None:
        """Read the version before ``revision`` in its Item, or None for version 1.

        Raises:
            AMLError: ``parity_mismatch`` when that version is missing, is not
                the next lower version of the same Item, or was not recorded
                and committed before ``revision``.
        """
        if revision.previous_revision_id is None:
            return None
        older = self.revision(revision.previous_revision_id)
        if older is None or older.memory_id != revision.memory_id or (
            older.version != revision.version - 1
        ):
            raise AMLError("parity_mismatch", "A version chain skips or revisits a version")
        if older.recorded_at > revision.recorded_at or older.commit_seq >= revision.commit_seq:
            raise AMLError("parity_mismatch", "A version chain is recorded out of order")
        return older

    def links(self, revision_id: str) -> tuple[LinkRowV2, ...]:
        """Read every stored link one revision carries (A.3.3).

        Raises:
            AMLError: ``scope_denied`` for a link of another exact scope.
        """
        links = tuple(
            _parse(_LINK, raw, "link")
            for raw in self.page(RevisionLinks(source_revision_id=revision_id))
        )
        if any(
            (link.scope_kind, link.scope_id) != (self.scope.scope_kind, self.scope.scope_id)
            for link in links
        ):
            raise AMLError("scope_denied", "A memory read reached another exact scope")
        return links

    def links_to(
        self, link_type: Literal["supersedes", "corrects_history"], target: str,
    ) -> tuple[LinkRowV2, ...]:
        """Read every link of one type in the exact scope that targets one key (A.3.3)."""
        query = LinkTargetRows(
            scope_kind=self.scope.scope_kind, scope_id=self.scope.scope_id,
            link_type=link_type, target_key=target,
        )
        return tuple(_parse(_LINK, raw, "link") for raw in self.page(query))

    def evidence(self, evidence_id: str) -> EvidenceRowV1 | None:
        """Read one Evidence part in any state and scope, or None when it does not exist."""
        if evidence_id not in self._evidence:
            raw = self.row("evidence", evidence_id)
            self._evidence[evidence_id] = None if raw is None else _parse(
                _EVIDENCE, raw, "Evidence",
            )
        return self._evidence[evidence_id]

    def readability(self, now: int) -> Readability:
        """Bind the D.2 predicates to this read's scope, a clock and its recorded Evidence.

        The predicates see only Evidence this read recorded, so `finish` has
        rechecked every header they read.

        Args:
            now (int): The predicates' clock in epoch milliseconds.

        Returns:
            Readability: The read's predicates.
        """
        def lookup(evidence_id: str) -> EvidenceRowV1 | None:
            if evidence_id not in self._evidence:
                raise AMLError("binding_mismatch", "A predicate read unrecorded Evidence")
            return self._evidence[evidence_id]

        return Readability(self.scope.scope_kind, self.scope.scope_id, now, lookup)

    def finish(self) -> None:
        """Recheck every recorded row and lookup, and the authority, before delivery.

        Raises:
            AMLError: ``reception_stale`` when anything read has changed since;
                the guard's or the grant's refusal.
        """
        self.authorize()
        for (kind, object_id), (digest, _) in self._rows.items():
            if self._read_row(kind, object_id)[0] != digest:
                raise AMLError("reception_stale", "A memory read changed before delivery")
        for lookup, rows in self._pages.values():
            again = self._read_page(lookup._replace(cap=lookup.cap + len(rows)))
            if again != rows:
                raise AMLError("reception_stale", "A memory read changed before delivery")
        self.authorize()


def _parse[T](adapter: TypeAdapter[T], raw: Mapping[str, Any], name: str) -> T:
    try:
        return adapter.validate_python(raw)
    except ValidationError:
        raise AMLError("schema_invalid", f"A stored {name} row is malformed") from None


def cite(rows: MemoryRows, snapshot: MemorySnapshotV2) -> None:
    """Record every Evidence part a version cites, so the read's predicates may read it."""
    for evidence_id in sorted({span.evidence_id for span in snapshot.quote_spans}):
        rows.evidence(evidence_id)


def version_quotes(
    snapshot: MemorySnapshotV2, read: Readability, rows: MemoryRows, speakers: QuoteSpeakers,
) -> tuple[MemoryQuoteV1, ...]:
    """Render a version's P-QUOTE spans as quotes, in span order (E.2.2 #3, E.2.3 #3).

    Args:
        snapshot (MemorySnapshotV2): The version; its Evidence is recorded (`cite`).
        read (Readability): The read's predicates.
        rows (MemoryRows): The read's guarded rows.
        speakers (QuoteSpeakers): The speaker resolution.

    Returns:
        tuple[MemoryQuoteV1, ...]: One quote per readable span, with the lowest
            usable group containing it; empty for an erased or unreadable version.

    Raises:
        AMLError: ``service_unhealthy`` when a span lies outside its Evidence
            part or splits a character.
    """
    readable = read.p_quote(snapshot)
    return tuple(
        span_quote(
            rows, snapshot.quote_spans[index], speakers,
            group=next(
                (group for group in readable.groups if index in snapshot.support_groups[group]),
                None,
            ),
        )
        for index in readable.spans
    )


def span_quote(
    rows: MemoryRows, span: QuoteSpanV1, speakers: QuoteSpeakers, *, group: int | None,
) -> MemoryQuoteV1:
    """Render one readable span of recorded Evidence as a quote (E.2 ``Quote``).

    Args:
        rows (MemoryRows): The read's guarded rows; the span's Evidence is recorded.
        span (QuoteSpanV1): A span that passes P-SPAN.
        speakers (QuoteSpeakers): The speaker resolution.
        group (int | None): The usable group containing the span, or None.

    Returns:
        MemoryQuoteV1: The quote with its exact bytes and source metadata.

    Raises:
        AMLError: ``binding_mismatch`` when the Evidence is gone;
            ``service_unhealthy`` when the span lies outside its part or splits
            a character.
    """
    evidence = rows.evidence(span.evidence_id)
    if evidence is None:
        raise AMLError("binding_mismatch", "A readable span lost its Evidence")
    data = evidence.text.encode("utf-8")
    if len(data) != evidence.byte_end - evidence.byte_start or not (
        evidence.byte_start <= span.byte_start < span.byte_end <= evidence.byte_end
    ):
        raise AMLError("service_unhealthy", "A quote has inconsistent Evidence coordinates")
    try:
        text = data[span.byte_start - evidence.byte_start:span.byte_end - evidence.byte_start]
        quote = text.decode()
    except UnicodeDecodeError:
        raise AMLError("service_unhealthy", "A quote splits a UTF-8 character") from None
    return MemoryQuoteV1(
        evidence_id=span.evidence_id, byte_start=span.byte_start, byte_end=span.byte_end,
        quote=quote, role=evidence.role, speaker=speakers.speaker(rows, span),
        source_time=evidence.source_observed_at, time_granularity=evidence.time_granularity,
        group=group,
    )


def erased_through(chain: Sequence[RevisionRowV2]) -> int:
    """Return the highest ``privacy_erase`` version of one Item's chain, or 0 (E.2.2 #3).

    Args:
        chain (Sequence[RevisionRowV2]): Versions of one Item, including every
            version newer than any version a caller redacts with the result.

    Returns:
        int: The version at or before which every version is redacted.
    """
    return max((revision.version for revision in chain if revision.op == "erase"), default=0)


@dataclass(frozen=True, slots=True)
class _Reached:
    """One version the walk read, with the links its revision carries."""

    revision: RevisionRowV2
    links: tuple[LinkRowV2, ...]
    reached_via: ReachedVia
    erased: bool
    """Whether the version is at or before a ``privacy_erase`` version of its Item."""


@dataclass(frozen=True, slots=True)
class _Walk:
    """The versions one walk read, and whether it stopped before reading all of them."""

    versions: tuple[_Reached, ...]
    truncated: bool


def _walk(rows: MemoryRows, item: ItemRowV2) -> _Walk:
    """Read an Item's versions and those of every older Item its revisions end (E.2.2 #1).

    Items are visited breadth first from the named Item, each from its head
    back to version 1. A ``supersedes`` link on any revision of a visited Item
    reaches the Item it ended. An Item is ended by revisions of one Item only,
    so an Item reached from two Items, or the named Item reached again, is a
    cycle and fails. The walk stops after `MAX_WALK_VERSIONS` versions.

    Raises:
        AMLError: ``scope_denied`` when a hop reaches another exact scope;
            ``parity_mismatch`` for a missing older Item or head, a broken
            version chain or a cycle.
    """
    reached: list[_Reached] = []
    queue: list[tuple[ItemRowV2, ReachedVia]] = [(item, "revision")]
    ended_by: dict[str, str] = {item.object_id: ""}
    while queue:
        current, via = queue.pop(0)
        rows.authorize()
        chain: list[tuple[RevisionRowV2, tuple[LinkRowV2, ...]]] = []
        revision: RevisionRowV2 | None = rows.head(current)
        truncated = False
        while revision is not None:
            if len(reached) + len(chain) >= MAX_WALK_VERSIONS:
                truncated = True
                break
            links = rows.links(revision.object_id)
            chain.append((revision, links))
            for link in links:
                if isinstance(link, SupersedesLinkRowV2):
                    queue.extend(_older(rows, current, link.target_key, ended_by))
            revision = rows.previous(revision)
        erased = erased_through([revision for revision, _ in chain])
        reached.extend(
            _Reached(
                revision, links, via if position == 0 else "revision",
                erased=revision.version <= erased,
            )
            for position, (revision, links) in enumerate(chain)
        )
        if truncated:
            return _Walk(tuple(reached), truncated=True)
    return _Walk(tuple(reached), truncated=False)


def _older(
    rows: MemoryRows, current: ItemRowV2, target: str, ended_by: dict[str, str],
) -> list[tuple[ItemRowV2, ReachedVia]]:
    """Reach the Item a ``supersedes`` link of ``current`` ended, once."""
    known = ended_by.get(target)
    if known is not None:
        if known != current.object_id:
            raise AMLError("parity_mismatch", "History revisits an Item")
        return []
    rows.authorize()
    older = rows.item(target)
    if older is None:
        raise AMLError("parity_mismatch", "A supersedes link targets a missing Item")
    ended_by[target] = current.object_id
    return [(older, "supersedes")]


def _newest_first(versions: Sequence[_Reached]) -> list[_Reached]:
    """Order versions newest first by ``recorded_at``, then revision id (E.2.2 #1).

    Versions recorded in the same millisecond keep their recording order: the
    transaction that wrote them (``commit_seq``) decides before the id, so two
    versions of one Item always follow their chain.
    """
    return sorted(
        versions,
        key=lambda reached: (
            reached.revision.recorded_at, reached.revision.commit_seq,
            reached.revision.object_id,
        ),
        reverse=True,
    )


@dataclass(frozen=True, slots=True)
class _Correction:
    """A ``historical`` Item correcting a version: its Item, head and recording time."""

    item: ItemRowV2
    head: RevisionRowV2
    recorded_at: int


class _Builder:
    """Reads what the shown versions depend on, then builds their DTOs under the predicates."""

    def __init__(self, rows: MemoryRows, speakers: QuoteSpeakers) -> None:
        self.rows, self.speakers = rows, speakers
        self._corrections: dict[str, tuple[_Correction, ...]] = {}
        self._targets: dict[str, CorrectedVersionV1] = {}
        self._successors: list[tuple[SupersedesLinkRowV2, RevisionRowV2]] = []

    def prepare(self, memory_id: str, versions: Sequence[_Reached]) -> None:
        """Read every row the DTOs read, before `MemoryRows.finish` rechecks them."""
        for reached in versions:
            cite(self.rows, reached.revision.snapshot)
            self._corrections[reached.revision.object_id] = self._correcting(
                reached.revision.object_id,
            )
            for link in reached.links:
                if isinstance(link, CorrectsHistoryLinkRowV2):
                    self._targets[link.target_key] = self._corrected(link.target_key)
        self._successors = self._ending(memory_id)

    def _corrected(self, revision_id: str) -> CorrectedVersionV1:
        target = self.rows.revision(revision_id)
        if target is None or self.rows.item(target.memory_id) is None:
            raise AMLError("parity_mismatch", "A correction targets a missing version")
        return CorrectedVersionV1(memory_id=target.memory_id, revision_id=revision_id)

    def _correcting(self, revision_id: str) -> tuple[_Correction, ...]:
        """Read the ``historical`` Items whose ``corrects_history`` rows target a revision."""
        found: dict[str, _Correction] = {}
        for link in self.rows.links_to("corrects_history", revision_id):
            source = self.rows.revision(link.source_revision_id)
            if source is None:
                raise AMLError("parity_mismatch", "A correction link lost its revision")
            if source.memory_id in found:
                continue
            item = self.rows.item(source.memory_id)
            if item is None or item.lifecycle_state != "historical":
                raise AMLError("parity_mismatch", "A correction lies outside a historical Item")
            head = first = self.rows.head(item)
            while (older := self.rows.previous(first)) is not None:
                first = older
            cite(self.rows, head.snapshot)
            found[item.object_id] = _Correction(item, head, first.recorded_at)
        return tuple(found[key] for key in sorted(found))

    def _ending(self, memory_id: str) -> list[tuple[SupersedesLinkRowV2, RevisionRowV2]]:
        """Read the active ``supersedes`` links that end an Item, with each source's head.

        A link is active while its revision is the head of an Item that is not
        erased (A.3.3); a link on an older revision of its Item is not.
        """
        found: list[tuple[SupersedesLinkRowV2, RevisionRowV2]] = []
        for link in self.rows.links_to("supersedes", memory_id):
            if not isinstance(link, SupersedesLinkRowV2):
                raise AMLError("parity_mismatch", "A supersedes lookup returned another link")
            source = self.rows.revision(link.source_revision_id)
            item = None if source is None else self.rows.item(source.memory_id)
            if source is None or item is None:
                raise AMLError("parity_mismatch", "A supersedes link lost its source")
            if item.current_revision_id != source.object_id or item.lifecycle_state == "erased":
                continue
            head = self.rows.head(item)
            cite(self.rows, head.snapshot)
            found.append((link, head))
        return found

    def successors(self, read: Readability, at: int | None) -> tuple[HistoryChainLinkV1, ...]:
        """The readable successors, one hop; recorded at or before ``at`` when it is set."""
        return tuple(sorted(
            (
                HistoryChainLinkV1(
                    memory_id=head.memory_id, relation=link.relation,
                    effective_at=link.effective_at, recorded_at=link.created_at,
                )
                for link, head in self._successors
                if read.p_ver(head.snapshot) and (at is None or link.created_at <= at)
            ),
            key=lambda link: (link.memory_id, link.recorded_at),
        ))

    def version(self, reached: _Reached, read: Readability, at: int | None) -> HistoryVersionV3:
        """Build one version under the read's predicates (E.2.2 #2, #3)."""
        revision, snapshot = reached.revision, reached.revision.snapshot
        redacted = reached.erased or not read.p_ver(snapshot)
        corrections = [
            HistoricalCorrectionV1(
                memory_id=correction.item.object_id,
                statement=correction.head.snapshot.statement,
                quotes=version_quotes(correction.head.snapshot, read, self.rows, self.speakers),
                recorded_at=correction.recorded_at,
            )
            for correction in self._corrections[revision.object_id]
            if read.p_ver(correction.head.snapshot)
            and (at is None or correction.recorded_at <= at)
        ]
        return HistoryVersionV3(
            memory_id=revision.memory_id, revision_id=revision.object_id,
            version=revision.version, recorded_at=revision.recorded_at,
            change=history_change(revision.op, revision.reason_code),
            reached_via=reached.reached_via,
            supersedes=tuple(sorted(
                (
                    HistoryChainLinkV1(
                        memory_id=link.target_key, relation=link.relation,
                        effective_at=link.effective_at, recorded_at=link.created_at,
                    )
                    for link in reached.links if isinstance(link, SupersedesLinkRowV2)
                ),
                key=lambda link: (link.memory_id, link.recorded_at),
            )),
            contradicts=tuple(sorted({
                link.target_key for link in reached.links if link.link_type == "contradicts"
            })),
            corrects_history=tuple(sorted(
                (
                    self._targets[link.target_key] for link in reached.links
                    if isinstance(link, CorrectsHistoryLinkRowV2)
                ),
                key=lambda target: (target.memory_id, target.revision_id),
            )),
            historical_corrections=tuple(sorted(
                corrections, key=lambda correction: (correction.recorded_at, correction.memory_id),
            )),
            redacted=redacted,
            statement=None if redacted else snapshot.statement,
            kind=None if redacted else snapshot.kind,
            valid_from=None if redacted else snapshot.valid_from,
            valid_to=None if redacted else snapshot.valid_to,
            time_granularity=None if redacted else snapshot.time_granularity,
            quotes=() if redacted else version_quotes(snapshot, read, self.rows, self.speakers),
        )


def read_memory_history(
    memory_id: str,
    *,
    rows: MemoryRows,
    speakers: QuoteSpeakers,
    limit: int,
    recorded_as_of: int | None = None,
) -> MemoryHistoryV3:
    """Read one memory's History inside a pinned scope's callback (E.2.2).

    The timeline is the newest `limit` versions the walk reaches. A
    recorded-time selection picks, among them, the version recorded last at
    or before ``recorded_as_of``; world validity is never synthesized.

    Args:
        memory_id (str): The exact Item the caller names.
        rows (MemoryRows): The read's guarded rows.
        speakers (QuoteSpeakers): The speaker resolution of quotes.
        limit (int): The most entries, 1 through `MAX_HISTORY_LIMIT`.
        recorded_as_of (int | None): The recording-time instant to select at.

    Returns:
        MemoryHistoryV3: The History, evaluated at the clock read after every
            read was rechecked.

    Raises:
        AMLError: ``scope_denied`` when the Item is absent from the exact scope
            (forgotten, never created, another scope's); ``reception_stale``
            when the store changed during the read; ``parity_mismatch`` for an
            inconsistent store.
    """
    rows.authorize()
    item = rows.item(memory_id)
    if item is None:
        raise AMLError("scope_denied", "History target is unavailable")
    walk = _walk(rows, item)
    ordered = _newest_first(walk.versions)
    complete = not walk.truncated and len(ordered) <= limit
    shown = ordered[:limit] if recorded_as_of is None else [
        reached for reached in ordered[:limit] if reached.revision.recorded_at <= recorded_as_of
    ][:1]
    builder = _Builder(rows, speakers)
    builder.prepare(memory_id, shown)
    rows.finish()
    now = rows.clock()
    read = rows.readability(now)
    return MemoryHistoryV3(
        memory_id=memory_id,
        observed_at=now,
        recorded_as_of=recorded_as_of,
        status=None if recorded_as_of is None else (
            "found" if shown else "not_recorded" if complete else "truncated"
        ),
        successors=builder.successors(read, recorded_as_of),
        entries=tuple(builder.version(reached, read, recorded_as_of) for reached in shown),
        completeness="complete" if complete else "truncated",
    )


class MemoryReads:
    """The shared entry of AML's memory reads: authority, the pinned guard and its rows.

    History, Why and add-items each read inside one exact-scope pin of the
    registered read model, under a grant resolved outside AML locks and
    revalidated at every hop and after delivery. An unfinished forget or
    erasure closes the scope's pin with ``index_rebuilding`` (D.6).

    Args:
        writer (Writer): The sole persistence owner.
        auth (AuthProvider): Resolves the caller's exact scope.
        publisher (StoreReadModel): The read model registered for ``writer``.
        clock (Callable[[], int]): Trusted wall clock in epoch milliseconds.
    """

    def __init__(
        self, writer: Writer, auth: AuthProvider, publisher: StoreReadModel,
        *, clock: Callable[[], int],
    ) -> None:
        self.writer, self.auth, self.publisher, self.clock = writer, auth, publisher, clock

    def consume[T](
        self, scope: ScopeV1, read: Callable[[MemoryRows], T], consumer: Callable[[T], None],
        *, completions: bool = False,
    ) -> None:
        """Run one read inside a pinned scope and deliver its result inside the callback.

        The consumer encodes the result while the guard still holds the scope,
        so a privacy commit that revokes the pin fails the delivery instead of
        releasing bytes it withdrew (INV T02, T04).

        Args:
            scope (ScopeV1): The exact read scope.
            read (Callable[[MemoryRows], T]): Builds the result from the rows.
            consumer (Callable[[T], None]): Called once with the result.
            completions (bool): A completion read (add-items), which an Add owner
                holding the scope does not fence (``StoreReadModel.pin``).

        Raises:
            AMLError: ``authorization_unavailable`` for an unregistered read
                model; the provider's refusal; ``scope_denied`` or
                ``index_rebuilding`` from the pin; or the read's refusal.
        """
        exact = ScopeV1.model_validate(scope.model_dump(mode="json"))
        if StoreReadModel.for_writer(self.writer) is not self.publisher:
            raise AMLError(
                "authorization_unavailable", "Memory reads require the registered read model",
            )
        grant = self.auth.resolve(AuthRequest(requested_scope=exact, now_ms=self.clock()))
        check_add_grant(grant, exact, self.clock())
        with self.publisher.pin([exact], completions=completions) as guard:

            def deliver() -> None:
                consumer(read(MemoryRows(self.writer, exact, grant, guard, self.clock)))
                check_add_grant(grant, exact, self.clock())

            guard.consume_operation(exact, deliver)


class MemoryHistoryReader:
    """Answer what one memory said before, across its versions and the Items it ended (E.2.2).

    Args:
        reads (MemoryReads): The shared guarded read entry.
        speakers (QuoteSpeakers): The speaker resolution Search delivery uses.
    """

    def __init__(self, reads: MemoryReads, speakers: QuoteSpeakers) -> None:
        self._reads, self._speakers = reads, speakers

    def consume(
        self,
        memory_id: str,
        *,
        scope: ScopeV1,
        consumer: Callable[[MemoryHistoryV3], None],
        limit: int = 32,
        recorded_as_of: int | None = None,
    ) -> None:
        """Deliver one History while the scope's pin is held.

        Args:
            memory_id (str): The exact Item; History never searches for one.
            scope (ScopeV1): The caller's exact scope.
            consumer (Callable[[MemoryHistoryV3], None]): Called once, inside the
                pinned callback, with the immutable History.
            limit (int): The most entries, 1 through 128.
            recorded_as_of (int | None): An instant of recording time, never
                world time, no later than the read's clock.

        Raises:
            AMLError: ``invalid_request`` for an invalid id, limit or instant;
                ``scope_denied`` for a forgotten, absent or other-scope Item,
                with identical public bytes; ``index_rebuilding`` during an
                unfinished forget or erasure; ``reception_stale`` when the
                store changed during the read; the shared entry's refusals.
        """
        try:
            selected = _ULID.validate_python(memory_id, strict=True)
            at = None if recorded_as_of is None else _RECORDED_TIME.validate_python(
                recorded_as_of, strict=True,
            )
        except ValidationError:
            raise AMLError("invalid_request", "History needs an Item id and a valid time") from None
        if type(limit) is not int or not 1 <= limit <= MAX_HISTORY_LIMIT:
            raise AMLError("invalid_request", "History limit is outside 1 to 128")
        if at is not None and at > self._reads.clock():
            raise AMLError("invalid_request", "Recorded time cannot be in the future")
        self._reads.consume(
            scope,
            lambda rows: read_memory_history(
                selected, rows=rows, speakers=self._speakers, limit=limit, recorded_as_of=at,
            ),
            consumer,
        )
