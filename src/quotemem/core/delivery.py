"""Pure Search eligibility and quote lines shared by ranking and delivery.

Readability owns the read predicates and world visibility. The aliases below are its actual method
objects, not alternate implementations. Folding, rendering and the delivery recheck
import ``rendered_quotes`` from here; none may select or format quotes itself.
The caller owns the read pin, knowledge reconstruction and frozen speaker labels.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import Literal, Protocol, assert_never, cast

from pydantic import TypeAdapter, ValidationError

from quotemem.core.entities import ActivationMode, MemoryKind, MemoryLifecycle, SensitivityClass
from quotemem.core.errors import QuoteMemError
from quotemem.core.forget_remainder import current_remainder, masked
from quotemem.core.identity import canonical_json
from quotemem.core.query import ReturnedEvidenceV1, ReturnedItemV1, ReturnedMemoryV1, ScopeV1
from quotemem.core.readability import (
    EvidencePart,
    Readability,
    VersionContent,
    WorldIndex,
    WorldMemory,
)
from quotemem.core.recall_corpora import MemoryDocument, memory_corpus
from quotemem.core.recall_snapshot import RecallSnapshot, SourceRowV2
from quotemem.core.semantic import SecurityLevel, TimeGranularity, TrustTier

_RETURNED: TypeAdapter[ReturnedItemV1] = TypeAdapter(ReturnedItemV1)

MemoryMode = Literal["current", "plain", "auto", "core_hook", "world", "knowledge"]
QuoteKey = tuple[str, int, int]
"""Evidence id and half-open original-message byte range of one quote."""

p_mem = Readability.p_mem
p_ver = Readability.p_ver
p_quote = Readability.p_quote
p_past = Readability.p_past
world_visible = Readability.world_visible
world_intervals = Readability.world_intervals


@dataclass(frozen=True, slots=True)
class AutomaticRead:
    """Memory activation and shown-head security fields for auto/core_hook reads.

    Callers copy these fields from the same pinned memory and head passed to
    ``memory_eligible``; none has a permissive default.
    """

    activation_mode: ActivationMode
    security_level: SecurityLevel
    trust_tier: TrustTier
    sensitivity_classes: tuple[SensitivityClass, ...]


@dataclass(frozen=True, slots=True)
class KnowledgeSelection:
    """The knowledge reader's newest revision recorded at or before t.

    ``state`` is the memory's state at t, reconstructed from ``retired_at``.
    The reader proves revision identity and chronology, and excludes forgotten
    memories. Eligibility checks this version's readability now, not its head's
    readability and not world validity.
    """

    version: VersionContent
    state: MemoryLifecycle


def memory_eligible(
    memory: WorldMemory,
    *,
    mode: MemoryMode,
    read: Readability,
    index: WorldIndex,
    at: int | None = None,
    automatic: AutomaticRead | None = None,
    knowledge: KnowledgeSelection | None = None,
) -> bool:
    """Check one memory for an already resolved Search mode.

    Args:
        memory: Existing memory and current head at the read pin. The caller
            excludes forgotten memories before calling, including in knowledge mode.
        mode: Product uses current; bench without as_of uses plain. Automatic
            reads always use auto/core_hook, never plain or an as-of mode.
        read: The pin's current scope, clock and Evidence lookup.
        index: The same pin's world index, used only for plain and world reads.
        at: World time, required for world. Knowledge chronology is proved by
            the reader that supplies knowledge.
        automatic: memory/head metadata, required for auto and core_hook.
        knowledge: Selected version and state at t, required for knowledge.

    Returns:
        Whether the memory can participate in ranking or delivery. Plain
        callers obtain its end with the exported p_past function; world callers
        obtain intervals with world_intervals, both the actual functions.

    Raises:
        ValueError: If the mode's required inputs are absent.
    """
    if mode == "current":
        return p_mem(read, memory.lifecycle_state, memory.head)
    if mode == "plain":
        return p_mem(read, memory.lifecycle_state, memory.head) or (
            p_past(read, memory, index) is not None
        )
    if mode == "auto" or mode == "core_hook":
        if automatic is None:
            raise ValueError("Automatic reads require pinned Item and head metadata")
        return (
            p_mem(read, memory.lifecycle_state, memory.head)
            and automatic.activation_mode != "on_demand"
            and automatic.security_level != "restricted"
            and not automatic.sensitivity_classes
            and automatic.trust_tier not in {"inferred", "untrusted"}
        )
    if mode == "world":
        if at is None:
            raise ValueError("World reads require an as-of time")
        return world_visible(read, memory, at, index)
    if mode == "knowledge":
        if knowledge is None:
            raise ValueError("Knowledge reads require a selected version and state at t")
        return (
            memory.lifecycle_state != "historical"
            and knowledge.state == "active"
            and p_ver(read, knowledge.version)
        )
    assert_never(mode)


class QuoteSource(EvidencePart, Protocol):
    """A pinned Evidence row with the display metadata; no label inference."""

    @property
    def role(self) -> str | None: ...

    @property
    def source_observed_at(self) -> int | None: ...

    @property
    def ingested_at(self) -> int: ...

    @property
    def time_granularity(self) -> TimeGranularity: ...


@dataclass(frozen=True, slots=True)
class RenderedQuote:
    """One complete, unescaped delivery quote line and its exact coverage.

    ``text`` includes the numbered label and JSON string literals. The V1
    renderer HTML-escapes its whole body once, after joining these lines.
    Offsets stay program-side and are relative to the original message,
    including when the Evidence is a later part of that message. ``role`` is
    the quoted Evidence's role and ``speaker`` the span's resolved speaker.
    """

    text: str
    evidence_id: str
    byte_start: int
    byte_end: int
    role: str | None = None
    speaker: str | None = None
    quote: str = field(default="", compare=False)
    """The quoted text alone, for renderings other than delivery's; not part of equality."""


@dataclass(frozen=True, slots=True)
class RenderedQuotes:
    """At most four lines, plus the count for the omitted-quote notice."""

    lines: tuple[RenderedQuote, ...]
    omitted: int


_MONTHS = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)  # fmt: skip
_WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")


def spell_day(day: date) -> str:
    """A calendar day as a reader writes it, with its weekday: Monday, 8 May 2023."""
    return f"{_WEEKDAYS[day.weekday()]}, {day.day} {_MONTHS[day.month - 1]} {day.year:04d}"


def spell_month(year: int, month: int) -> str:
    """A calendar month as a reader writes it: May 2023."""
    return f"{_MONTHS[month - 1]} {year:04d}"


def instant(value: int) -> datetime | None:
    """The UTC datetime of epoch milliseconds, or None outside the calendar."""
    try:
        return datetime(1970, 1, 1, tzinfo=UTC) + timedelta(milliseconds=value)
    except OverflowError:
        return None


def format_date(value: int | None, granularity: TimeGranularity) -> str:
    """The date spelling, at day precision or coarser for a reader.

    YEAR and MONTH keep their precision; DAY and HOUR name the calendar day.
    Stored instants and the memory's own statement remain unchanged.
    """
    if value is None:
        return "not provided"
    moment = instant(value)
    if moment is None or granularity == "unanchored":
        return f"epoch milliseconds {value}"
    if granularity == "YEAR":
        return f"{moment.year:04d}"
    if granularity == "MONTH":
        return spell_month(moment.year, moment.month)
    return spell_day(moment.date())


def said_on(sources: Sequence[QuoteSource]) -> date | None:
    """The one UTC day all of these sources were said on, or None.

    Relative words in text taken from them ("yesterday", "last week") are read
    against that day. Sources without a day-precise source time, or spread over
    several days, leave the reading ambiguous.
    """
    days: set[date] = set()
    for source in sources:
        observed = source.source_observed_at
        moment = None if observed is None else instant(observed)
        if moment is None or source.time_granularity not in {"DAY", "HOUR"}:
            return None
        days.add(moment.date())
    return next(iter(days)) if len(days) == 1 else None


def format_last_unit(end: int, granularity: TimeGranularity) -> str | None:
    """Spell the last whole unit inside a half-open period that ends at ``end``.

    An event's own ``valid_to`` is exclusive: "last week" spoken on a Sunday
    ends at the next Monday 00:00. Spelling that instant reads as one unit too
    many, so the last unit the period covers is named instead.

    Args:
        end: The exclusive end in epoch milliseconds.
        granularity: The unit the period was anchored at.

    Returns:
        format_date's spelling of the unit holding the period's last instant;
        None when the end is unanchored or not a calendar date.
    """
    if granularity == "unanchored":
        return None
    rendered = format_date(end - 1, granularity)
    return None if rendered.startswith("epoch milliseconds ") else rendered


def rendered_quotes(
    version: VersionContent,
    *,
    sources: Mapping[str, QuoteSource],
    speakers: Mapping[QuoteKey, str | None],
    scope_kind: str,
    scope_id: str,
    now: int,
) -> RenderedQuotes:
    """Select and render the single shared delivery quote sequence.

    Args:
        version: A head or selected revision with its spans and support groups.
        sources: Evidence rows at the caller's read pin, keyed by evidence id.
            The same rows supply both readability headers and rendered bytes.
        speakers: Per-span speakers already resolved from frozen source labels.
            A missing key or None renders JSON null; source-level speaker
            metadata is never substituted for a span's resolved speaker.
        scope_kind: Kind of the caller's exact authorized scope.
        scope_id: Id of that scope.
        now: Trusted current clock for readability, including as-of reads.

    Returns:
        At most four quote lines and the number not shown. The complete usable
        group with the greatest maximum source time comes first. Its spans,
        and then the other ``p_quote`` spans, sort by ascending source time,
        evidence id and byte_start (byte_end breaks an otherwise equal key).
        Source time is source_observed_at, falling back to ingested_at. Equal
        newest groups sort by their ordered span keys, independent of input
        group order. Shared spans occur once. No trimming or normalization
        changes the quoted bytes or the coverage returned for folding.

    Raises:
        QuoteMemError: delivery_stale when no complete usable support remains;
            service_unhealthy when a selected span has inconsistent Evidence
            identity, part bounds or UTF-8 boundaries.
    """
    read = Readability(scope_kind, scope_id, now, sources.get)
    if not p_ver(read, version):
        raise QuoteMemError("delivery_stale", "The memory has no complete usable support")
    readable = p_quote(read, version)

    def order(position: int) -> tuple[int, str, int, int]:
        span = version.quote_spans[position]
        source = sources[span.evidence_id]
        time = source.source_observed_at
        return (
            source.ingested_at if time is None else time,
            span.evidence_id,
            span.byte_start,
            span.byte_end,
        )

    def group_order(position: int) -> tuple[int, tuple[tuple[int, str, int, int], ...]]:
        keys = tuple(sorted(order(member) for member in version.support_groups[position]))
        return -max(key[0] for key in keys), keys

    newest = version.support_groups[min(readable.groups, key=group_order)]
    first = set(newest)
    ordered = [
        *sorted(first, key=order),
        *sorted((position for position in readable.spans if position not in first), key=order),
    ]
    unique: dict[QuoteKey, int] = {}
    for position in ordered:
        span = version.quote_spans[position]
        unique.setdefault((span.evidence_id, span.byte_start, span.byte_end), position)
    positions = list(unique.values())
    lines: list[RenderedQuote] = []
    for number, position in enumerate(positions[:4], 1):
        span = version.quote_spans[position]
        source = sources[span.evidence_id]
        data = source.text.encode("utf-8")
        if (
            source.object_id != span.evidence_id
            or (len(data) != source.byte_end - source.byte_start)
            or not source.byte_start <= span.byte_start < span.byte_end <= source.byte_end
        ):
            raise QuoteMemError("service_unhealthy",
                                "A quote has inconsistent Evidence coordinates")
        try:
            quote = data[
                span.byte_start - source.byte_start : span.byte_end - source.byte_start
            ].decode("utf-8")
        except UnicodeDecodeError as error:
            raise QuoteMemError("service_unhealthy", "A quote splits a UTF-8 character") from error
        key = (span.evidence_id, span.byte_start, span.byte_end)
        text = (
            f"Quote {number} ({source.role or 'not provided'}; "
            f"speaker: {canonical_json(speakers.get(key))}; "
            f"source date: {format_date(source.source_observed_at, source.time_granularity)}): "
            f"{canonical_json(quote)}"
        )
        lines.append(RenderedQuote(text, *key, role=source.role, speaker=speakers.get(key),
                                   quote=quote))
    return RenderedQuotes(tuple(lines), max(0, len(positions) - len(lines)))


@dataclass(frozen=True, slots=True)
class MemoryPresentation:
    """The bytes-producing memory fields proved at a read pin.

    The end stores its instant and rendered clause, so changing only its basis
    or granularity invalidates delivery. Quote coverage participates in equality.
    ``valid_through`` marks a clause naming the last unit inside an exclusive
    event end; otherwise the clause spells the end instant itself. ``as_of``
    spells the newest complete support's source day (or its coarser unit) and
    ``ingest_seq`` is the support's newest commit sequence; ``said`` is the
    single day all that support was said on, against which the statement's
    relative words are read. ``kind`` is the version's state or event kind.
    """

    statement: str
    valid_from: str
    valid_to: str
    end_at: int | None
    quotes: RenderedQuotes
    created_at: int
    start_at: int | None
    as_of: str | None
    ingest_seq: int
    said: date | None
    valid_through: bool = False
    kind: MemoryKind = "state"


@dataclass(frozen=True, slots=True)
class DeliveryProjection:
    """One ordered locator and its fully re-derived presentation at a pin."""

    target: ReturnedItemV1
    memory: MemoryPresentation | None = None
    source: SourceRowV2 | None = None


def delivery_projections(
    root: RecallSnapshot,
    *,
    scope: ScopeV1,
    returned: tuple[ReturnedItemV1, ...],
    mode: MemoryMode,
    at_time: int,
    speakers: Mapping[QuoteKey, str | None],
    as_of: int | None = None,
    remainders: bool = False,
) -> tuple[DeliveryProjection, ...]:
    """Validate exact locators, eligibility, ends and quote lines at one pin.

    Knowledge callers supply the reader's reconstructed root. All other modes
    use the current root and the world functions through the shared selector.
    Neither ranking scores nor caller-supplied content authorize a target.

    Args:
        root: Guarded immutable root (selected revisions for knowledge).
        scope: Exact authorized scope.
        returned: Strict, unique locators in consecutive rank order.
        mode: Explicit resolved mode.
        at_time: Trusted current expiry clock.
        speakers: Frozen labels resolved for the same pin's spans.
        as_of: Explicit world instant; required in world mode.
        remainders: Whether an Evidence part a complete forget withheld bytes of may be
            delivered by its current remainder (the interactive Search of an ``agent``
            workspace, ``core.forget_remainder``): its source then shows its readable bytes
            alone.

    Returns:
        Ordered, guard-owned presentations for rendering and equality rechecks.

    Raises:
        QuoteMemError: delivery_stale for changed/unreadable targets; scope_denied
            for foreign identities; schema_invalid for malformed inputs.
    """
    if type(at_time) is not int or not 0 <= at_time <= 2**63 - 1:
        raise QuoteMemError("schema_invalid", "Delivery requires a current timestamp")
    fence = root.scope_fence(scope.scope_id)
    if fence.scope_kind != scope.scope_kind or fence.tomb:
        raise QuoteMemError("scope_denied", "Delivery crossed its exact scope")
    if (
        fence.visibility_barrier
        or fence.active_erasure
        or fence.maintenance
        or (fence.finalize_barrier is not None)
    ):
        raise QuoteMemError("index_rebuilding", "Delivery scope is fenced")
    read = Readability(scope.scope_kind, scope.scope_id, at_time, root.get_source)
    if any(
        not isinstance(cast(object, target), ReturnedMemoryV1 | ReturnedEvidenceV1)
        for target in returned
    ):
        raise QuoteMemError("schema_invalid", "Delivery requires strict locators")
    targets: list[ReturnedItemV1] = []
    seen: set[str] = set()
    for rank, supplied in enumerate(returned, 1):
        if not isinstance(cast(object, supplied), ReturnedMemoryV1 | ReturnedEvidenceV1):
            raise QuoteMemError("schema_invalid", "Delivery requires a strict locator")
        try:
            target = _RETURNED.validate_python(supplied.model_dump(mode="python"), strict=True)
        except ValidationError as error:
            raise QuoteMemError("delivery_stale", "Delivery locator is malformed") from error
        if target.rank != rank or target.object_id in seen:
            raise QuoteMemError("delivery_stale",
                                "Delivery ids and ranks must be unique and ordered")
        seen.add(target.object_id)
        targets.append(target)
    corpus = (
        memory_corpus(root, scope.scope_id)
        if any(target.kind == "memory" for target in returned)
        else None
    )
    projections: list[DeliveryProjection] = []
    for target in targets:
        if target.kind == "evidence":
            source = root.get_source(target.object_id)
            if source is not None and (source.scope_kind, source.scope_id) != (
                scope.scope_kind,
                scope.scope_id,
            ):
                raise QuoteMemError("scope_denied", "Delivery Evidence crossed its scope")
            held = target.object_id in root.held_evidence(scope.scope_id)
            readable = source is not None and read.p_evr(source, held=held)
            if (
                remainders and source is not None and not readable and not held
                and read.p_src(source) and current_remainder(source) is not None
            ):
                # Found by its remainder: it shows its readable bytes alone.
                source, readable = masked(source), True
            if (
                mode == "knowledge"
                or source is None
                or source.object_id != target.object_id
                or (
                    source.content_hash != target.content_hash
                    or not readable
                    or mode in {"auto", "core_hook"}
                    and (
                        source.security_level == "restricted"
                        or source.trust_tier in {"inferred", "untrusted"}
                    )
                )
            ):
                raise QuoteMemError("delivery_stale", "Delivery Evidence is no longer readable")
            projections.append(DeliveryProjection(target, source=source))
            continue
        row = root.get_memory(target.object_id)
        if row is not None and (row.scope_kind, row.scope_id) != (scope.scope_kind, scope.scope_id):
            raise QuoteMemError("scope_denied", "Delivery memory crossed its scope")
        if (
            row is None
            or row.object_id != target.object_id
            or (
                row.current_revision_id != target.revision_id
                or row.content_hash != target.content_hash
            )
        ):
            raise QuoteMemError("delivery_stale", "Delivery memory changed identity or content")
        document = MemoryDocument.of(row)
        version = document.snapshot
        if corpus is None:
            raise QuoteMemError("service_unhealthy", "Delivery memory corpus is missing")
        sources: dict[str, SourceRowV2] = {}
        for span in version.quote_spans:
            source = root.get_source(span.evidence_id)
            if source is not None:
                if source.object_id != span.evidence_id:
                    raise QuoteMemError("delivery_stale", "Delivery support changed identity")
                if (source.scope_kind, source.scope_id) != (scope.scope_kind, scope.scope_id):
                    raise QuoteMemError("scope_denied", "Delivery support crossed its scope")
                sources[span.evidence_id] = source
        if not memory_eligible(
            document.world,
            mode=mode,
            read=read,
            index=corpus,
            at=as_of,
            automatic=AutomaticRead(
                row.activation_mode,
                version.security_level,
                version.trust_tier,
                tuple(version.sensitivity_classes),
            ),
            knowledge=KnowledgeSelection(version, row.lifecycle_state),
        ):
            raise QuoteMemError("delivery_stale", "Delivery memory is no longer eligible")
        quotes = rendered_quotes(
            version,
            sources=sources,
            speakers=speakers,
            scope_kind=scope.scope_kind,
            scope_id=scope.scope_id,
            now=at_time,
        )
        end = None
        if mode == "plain" and not p_mem(read, row.lifecycle_state, version):
            past = p_past(read, document.world, corpus)
            if past is None or past.end is None:
                raise QuoteMemError("delivery_stale", "A past memory lost its witnessed end")
            end = past.end
        elif mode == "world":
            intervals = world_intervals(read, document.world, corpus)
            visible = [
                interval for interval in intervals if as_of is not None and interval.contains(as_of)
            ]
            if not visible:
                raise QuoteMemError("delivery_stale", "World memory lost its interval")
            # World visibility returns the canonical interval order, including corrected targets.
            end = visible[0].end
        valid_to = format_date(version.valid_to, version.time_granularity)
        end_at = version.valid_to
        last = None
        if end is None:
            if version.valid_to is not None:
                last = format_last_unit(version.valid_to, version.time_granularity)
        else:
            end_at = end.at
            valid_to = (
                "no later than " if end.basis == "source_observed_at" else ""
            ) + format_date(end.at, end.granularity)
            # A witness ends a state at another fact's time; only an event's own end is a unit.
            if end.source != "witness":
                last = format_last_unit(end.at, end.granularity)
        if last is not None:
            valid_to = last
        usable = read.p_quote(version)
        group_positions = {
            position for group in usable.groups for position in version.support_groups[group]
        }
        support = [
            sources[version.quote_spans[position].evidence_id] for position in group_positions
        ]
        observed = [source for source in support if source.source_observed_at is not None]
        timestamp = (
            min(cast(int, source.source_observed_at) for source in observed)
            if observed
            else min(source.ingested_at for source in support)
        )
        newest = max(
            observed, key=lambda source: cast(int, source.source_observed_at), default=None
        )
        projections.append(
            DeliveryProjection(
                target,
                memory=MemoryPresentation(
                    version.statement,
                    format_date(version.valid_from, version.time_granularity),
                    valid_to,
                    end_at,
                    quotes,
                    timestamp,
                    version.valid_from,
                    None
                    if newest is None
                    else format_date(newest.source_observed_at, newest.time_granularity),
                    max(source.ingested_commit_seq for source in support),
                    said_on(support),
                    last is not None,
                    version.kind,
                ),
            )
        )
    return tuple(projections)


def recheck_delivery(
    previous: tuple[DeliveryProjection, ...],
    root: RecallSnapshot,
    *,
    scope: ScopeV1,
    mode: MemoryMode,
    at_time: int,
    speakers: Mapping[QuoteKey, str | None],
    as_of: int | None = None,
) -> None:
    """Re-derive all shown bytes and ends; a difference requests one recompute.

    Args:
        previous: Presentations actually used to render/fold the response.
        root: Fresh delivery pin.
        scope: Exact authorized scope.
        mode: Original resolved Search mode.
        at_time: Trusted delivery clock.
        speakers: Freshly resolved frozen labels.
        as_of: Original explicit world instant.

    Raises:
        QuoteMemError: delivery_stale when any locator, end or rendered bytes changed.
    """
    current = delivery_projections(
        root,
        scope=scope,
        returned=tuple(value.target for value in previous),
        mode=mode,
        at_time=at_time,
        speakers=speakers,
        as_of=as_of,
        # A part delivered by its remainder showed its readable bytes alone: the same part
        # must still show the same, no more.
        remainders=any(value.source is not None and value.source.withheld_spans
                       for value in previous),
    )
    if current != previous:
        raise QuoteMemError("delivery_stale", "Delivery ends or rendered quote lines changed")
