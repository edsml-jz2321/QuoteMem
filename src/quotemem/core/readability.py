"""The readability predicates, world visibility and the provider renderer.

This module is the single implementation of the predicates, ``p_past`` included, and of the
world-visibility functions. Search, History, Why, add-items,
the Entity SDK, the catalog, context selection, judge input builders, privacy
compensation and the delivery recheck call it; no other module re-implements a
predicate, the withheld-range intersection, a world end, corrected history or a
world interval.

A `Readability` binds one read: its exact scope, its trusted clock and an
Evidence lookup. Its predicates take plain structural inputs, so Evidence rows,
RecallSnapshot projections and the plain records of an `AddReadView` all work:

- ``p_src``, ``p_span``, ``p_group``,
  ``p_ver``, ``p_mem``, ``p_name``, ``p_evr`` and ``p_quote``;
- ``world_intervals``, ``world_visible`` (world(t)) and
  ``p_past`` read one scope's memory corpus through a `WorldIndex` of
  plain `WorldMemory`, `IncomingLink` and `CorrectionRef` records and
  return `WorldInterval` values whose `WorldEnd` names the source delivery renders;
- ``render_masked`` renders one message's parts with every unreadable byte
  replaced by `WITHHELD_MARKER` and returns the segment map that relates the
  rendered text to message bytes.

Predicates read Evidence headers and content-free link metadata only. They
never consult tombstones and never read a correction's ``verdict_ref``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from itertools import pairwise
from typing import Final, Literal, NamedTuple, Protocol

from quotemem.core.entities import (
    EffectiveBasis,
    MemoryKind,
    MemoryLifecycle,
    QuoteSpanV1,
    RevisionReason,
    SupersedeRelation,
)
from quotemem.core.semantic import TimeGranularity

WITHHELD_MARKER = "[withheld]"
"""The literal that replaces every maximal run of unreadable message bytes."""

EndSource = Literal["valid_to", "corrected_valid_to", "witness"]
"""Where a world-visible interval ends: the shown memory's own event
``valid_to``, the event ``valid_to`` of the version a Corrected-history stand-in
replaces, or a witness's ``effective_at``."""

_GRANULARITY_ORDER: Final[Mapping[TimeGranularity, int]] = {
    "HOUR": 0, "DAY": 1, "MONTH": 2, "YEAR": 3, "unanchored": 4,
}
"""Finest first: the tie order among witnesses of one end."""


class ByteRange(Protocol):
    """A half-open range of original-message bytes, such as a withheld span."""

    @property
    def byte_start(self) -> int: ...

    @property
    def byte_end(self) -> int: ...


class SpanRef(ByteRange, Protocol):
    """A quote span: a byte range inside one Evidence part (``QuoteSpanV1``)."""

    @property
    def evidence_id(self) -> str: ...


class EvidenceHeader(Protocol):
    """The Evidence header fields that readability reads."""

    @property
    def object_id(self) -> str: ...

    @property
    def scope_kind(self) -> str: ...

    @property
    def scope_id(self) -> str: ...

    @property
    def state(self) -> str: ...

    @property
    def route_state(self) -> str: ...

    @property
    def expires_at(self) -> int | None: ...

    @property
    def withheld_spans(self) -> Sequence[ByteRange]: ...


class EvidencePart(EvidenceHeader, Protocol):
    """An Evidence part with its message coordinates and text, for rendering."""

    @property
    def session_id(self) -> str: ...

    @property
    def request_id(self) -> str: ...

    @property
    def ordinal(self) -> int: ...

    @property
    def part_no(self) -> int: ...

    @property
    def byte_start(self) -> int: ...

    @property
    def byte_end(self) -> int: ...

    @property
    def text(self) -> str: ...


class VersionContent(Protocol):
    """The parts of a version snapshot that readability reads.

    A version is erased exactly when its statement is empty.
    """

    @property
    def statement(self) -> str: ...

    @property
    def quote_spans(self) -> Sequence[SpanRef]: ...

    @property
    def support_groups(self) -> Sequence[Sequence[int]]: ...


class NamingMemory(Protocol):
    """The memory fields ``p_name`` reads: its lifecycle and its current head."""

    @property
    def lifecycle_state(self) -> str: ...

    @property
    def current_revision_id(self) -> str: ...


class NameRow(Protocol):
    """The fields of an ``entity_name`` row that ``p_name`` reads."""

    @property
    def naming_memory_id(self) -> str: ...

    @property
    def naming_revision_id(self) -> str: ...


class TimedVersion(VersionContent, Protocol):
    """A version snapshot with the time fields world visibility reads."""

    @property
    def kind(self) -> MemoryKind: ...

    @property
    def valid_from(self) -> int | None: ...

    @property
    def valid_to(self) -> int | None: ...

    @property
    def time_granularity(self) -> TimeGranularity: ...


class LinkRef(Protocol):
    """The fields of an active link that world visibility reads."""

    @property
    def link_type(self) -> str: ...

    @property
    def relation(self) -> str | None: ...

    @property
    def effective_at(self) -> int | None: ...


class RenderSegment(NamedTuple):
    """One run of readable message bytes and the rendered characters it became."""

    evidence_id: str
    msg_byte_start: int
    msg_byte_end: int
    rendered_char_start: int
    rendered_char_end: int


@dataclass(frozen=True, slots=True)
class ReadableQuotes:
    """The ``p_quote`` result for one version: what a reader may render and list.

    Attributes:
        spans (tuple[int, ...]): Indices into ``quote_spans`` of the spans that
            pass ``p_span``, ascending.
        groups (tuple[int, ...]): Indices into ``support_groups`` of the groups
            that pass ``p_group``, ascending.
    """

    spans: tuple[int, ...]
    groups: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class MaskedText:
    """A rendered message with its program-side segment map.

    Every character of ``text`` belongs either to exactly one segment, whose
    characters are the original bytes of that part, or to a `WITHHELD_MARKER`.
    Segments are in rendered order and never overlap.

    Attributes:
        text (str): The rendered text.
        segments (tuple[RenderSegment, ...]): The segment map.
    """

    text: str
    segments: tuple[RenderSegment, ...]

    def spans(self, char_start: int, char_end: int) -> tuple[QuoteSpanV1, ...] | None:
        """Map a range of rendered characters back to message bytes.

        A range that touches a marker covers withheld or missing bytes and maps
        to nothing. Any other range maps to one span per Evidence part it covers,
        on character boundaries.

        Args:
            char_start (int): First rendered character, inclusive.
            char_end (int): Last rendered character, exclusive.

        Returns:
            tuple[QuoteSpanV1, ...] | None: The spans in rendered order, or None
                when the range overlaps a marker.

        Raises:
            ValueError: If the range is empty or outside the rendered text.
        """
        if not 0 <= char_start < char_end <= len(self.text):
            raise ValueError(f"Invalid rendered range [{char_start}, {char_end})")
        result: list[QuoteSpanV1] = []
        covered = char_start
        for segment in self.segments:
            if segment.rendered_char_end <= covered:
                continue
            if segment.rendered_char_start > covered:
                return None
            end = min(char_end, segment.rendered_char_end)
            prefix = self.text[segment.rendered_char_start:covered]
            body = self.text[covered:end]
            byte_start = segment.msg_byte_start + len(prefix.encode("utf-8"))
            result.append(QuoteSpanV1(
                evidence_id=segment.evidence_id,
                byte_start=byte_start,
                byte_end=byte_start + len(body.encode("utf-8")),
            ))
            covered = end
            if covered == char_end:
                return tuple(result)
        return None


@dataclass(frozen=True, slots=True)
class WorldMemory:
    """One memory of a scope's memory corpus as world visibility reads it (``MemoryRowV2``).

    Attributes:
        memory_id (str): The memory.
        lifecycle_state (MemoryLifecycle): Its lifecycle.
        head_revision_id (str): Its current revision.
        head_reason (RevisionReason): The reason code of that revision.
        head (TimedVersion): The snapshot of that revision.
        links (tuple[LinkRef, ...]): The active links its head carries;
            its ``supersedes(update)`` links date the start of its own interval.
        corrected_targets (tuple[str, ...]): For a ``historical`` memory, the target
            revision ids of the ``corrects_history`` rows of every one of its
            revisions, which an erase of the memory keeps; empty for every other memory.

    Raises:
        ValueError: If a memory that is not ``historical`` names correction targets.
    """

    memory_id: str
    lifecycle_state: MemoryLifecycle
    head_revision_id: str
    head_reason: RevisionReason
    head: TimedVersion
    links: tuple[LinkRef, ...] = ()
    corrected_targets: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Keep correction targets on ``historical`` memories only."""
        if self.corrected_targets and self.lifecycle_state != "historical":
            raise ValueError("Only a historical Item corrects history")


@dataclass(frozen=True, slots=True)
class IncomingLink:
    """An active ``supersedes`` link that targets a memory, from the incoming-update index.

    Attributes:
        link_id (str): The link's ``object_id``.
        successor_id (str): The memory whose head carries the link.
        relation (SupersedeRelation): ``update``, ``correct`` or ``retract``; only
            an update can witness an end.
        effective_at (int | None): When the update took effect in the world, in
            epoch milliseconds; None when undated and for the other relations.
        effective_basis (EffectiveBasis | None): What ``effective_at`` came from;
            set exactly when ``effective_at`` is.
        observed_granularity (TimeGranularity | None): For basis
            ``source_observed_at``, the granularity delivery renders this end at:
            the finest ``time_granularity`` among the Evidence parts cited by
            the successor's head whose ``source_observed_at`` equals
            ``effective_at``, or ``DAY`` when no cited part still carries that
            time, which keeps "no later than" a true upper bound. None for
            every other link; basis ``valid_from`` renders at the successor
            head's granularity.

    Raises:
        ValueError: If the time fields disagree with the relation or the basis.
    """

    link_id: str
    successor_id: str
    relation: SupersedeRelation
    effective_at: int | None = None
    effective_basis: EffectiveBasis | None = None
    observed_granularity: TimeGranularity | None = None

    def __post_init__(self) -> None:
        """Apply the link time rule: only an update has a time, and a time always has a basis."""
        if (self.effective_at is None) != (self.effective_basis is None):
            raise ValueError("effective_at and effective_basis are set together")
        if self.effective_at is not None and self.relation != "update":
            raise ValueError("Only an update link has an effective time")
        if (self.effective_basis == "source_observed_at") != (
            self.observed_granularity is not None
        ):
            raise ValueError("observed_granularity is set exactly for basis source_observed_at")


class CorrectionRef(NamedTuple):
    """One ``historical`` memory that corrects a revision, in the corrected-version map.

    Attributes:
        memory_id (str): The ``historical`` memory; its head may be erased.
        created_commit_seq (int): The ``commit_seq`` of its version 1, which
            orders corrections.
    """

    memory_id: str
    created_commit_seq: int


class WorldIndex(Protocol):
    """One scope's memory corpus at a read's pin, as the functions read it.

    The memory corpus serves it from the RecallSnapshot projections. It
    holds the scope's memories in every lifecycle and nothing of a forgotten memory.
    Its links are active links, and its corrected-version map counts the
    ``corrects_history`` rows of every revision, erased correcting heads
    included. An incoming link keeps the ``effective_at`` and basis its
    row records; for basis ``source_observed_at`` the index also resolves the
    Evidence granularity from the successor head's cited parts, falling back to
    ``DAY`` when none still carries that time (`IncomingLink`).
    """

    def memory(self, memory_id: str) -> WorldMemory | None:
        """Return a memory of the scope, or None when it does not exist."""
        ...

    def head_memory(self, revision_id: str) -> WorldMemory | None:
        """Return the memory whose current head is this revision, or None."""
        ...

    def incoming(self, memory_id: str) -> Sequence[IncomingLink]:
        """Return the active ``supersedes`` links that target a memory."""
        ...

    def corrections(self, revision_id: str) -> Sequence[CorrectionRef]:
        """Return the ``historical`` memories whose ``corrects_history`` rows target a revision."""
        ...


@dataclass(frozen=True, slots=True)
class WorldEnd:
    """Where a world-visible interval ends and the source delivery renders it from.

    Attributes:
        at (int): The end in epoch milliseconds, exclusive.
        source (EndSource): ``valid_to``, ``corrected_valid_to`` or ``witness``.
        granularity (TimeGranularity): The granularity the end renders at: the
            shown memory's own for ``valid_to``, the replaced version's for
            ``corrected_valid_to``, the witness's for ``witness``.
        basis (EffectiveBasis | None): The witness's ``effective_basis``; None for
            the other sources.
        link_id (str | None): The witness link; None for the other sources.
    """

    at: int
    source: EndSource
    granularity: TimeGranularity
    basis: EffectiveBasis | None = None
    link_id: str | None = None


@dataclass(frozen=True, slots=True)
class WorldInterval:
    """One non-empty interval over which a memory is world-visible: ``start <= t < end``.

    Attributes:
        start (int | None): The first visible instant in epoch milliseconds;
            None when open.
        end (WorldEnd | None): Where visibility ends; None when open.
        corrects (str | None): For a Corrected-history stand-in, the revision it
            replaces; None for a memory's own interval.
    """

    start: int | None
    end: WorldEnd | None
    corrects: str | None = None

    def contains(self, at: int) -> bool:
        """Whether an instant lies in the interval.

        Args:
            at (int): The instant in epoch milliseconds.

        Returns:
            bool: True when ``start <= at < end``, with open bounds unbounded.
        """
        return (self.start is None or self.start <= at) and (
            self.end is None or at < self.end.at
        )


def _intersects(start: int, end: int, ranges: Sequence[ByteRange]) -> bool:
    return any(item.byte_start < end and start < item.byte_end for item in ranges)


def _end_order(end: WorldEnd) -> tuple[int, int, int, str]:
    """Sort key of one end: earliest, then the source order, granularity and link."""
    if end.source == "witness":
        rank = 2 if end.basis == "valid_from" else 3
    else:
        rank = 0 if end.source == "valid_to" else 1
    return end.at, rank, _GRANULARITY_ORDER[end.granularity], end.link_id or ""


def _past_order(interval: WorldInterval) -> tuple[int, int, int, str, str]:
    """Sort key of a past interval: latest end first, then its end's source order."""
    if interval.end is None:
        raise ValueError("A past interval has an end")
    at, rank, granularity, link = _end_order(interval.end)
    return -at, rank, granularity, link, interval.corrects or ""


def _interval(
    start: int | None, ends: Iterable[WorldEnd], corrects: str | None = None,
) -> WorldInterval | None:
    """Return the interval that ends at the first of ``ends``, or None when it is empty."""
    end = min(ends, key=_end_order, default=None)
    if start is not None and end is not None and start >= end.at:
        return None
    return WorldInterval(start, end, corrects)


def _own_end(version: TimedVersion) -> list[WorldEnd]:
    """An event's own ``valid_to`` as an end; nothing for a state."""
    if version.kind != "event" or version.valid_to is None:
        return []
    return [WorldEnd(version.valid_to, "valid_to", version.time_granularity)]


@dataclass(frozen=True, slots=True)
class Readability:
    """The predicates and world functions of one read: its scope, clock and Evidence.

    Attributes:
        scope_kind (str): Kind of the read's exact scope.
        scope_id (str): Id of the read's exact scope; other scopes never pass.
        now (int): The read's trusted clock in epoch milliseconds; Evidence with
            ``expires_at <= now`` is expired.
        evidence (Callable[[str], EvidenceHeader | None]): Evidence lookup by id;
            None for a row that does not exist.

    Examples:
        ```python
        read = Readability("user", scope_id, now, view.evidence)
        heads = [head for head in candidates if read.p_mem(head.lifecycle_state, head.snapshot)]
        past = [row for row in corpus_rows if read.p_past(row, corpus) is not None]
        then = [row for row in corpus_rows if read.world_visible(row, as_of, corpus)]
        ```
    """

    scope_kind: str
    scope_id: str
    now: int
    evidence: Callable[[str], EvidenceHeader | None]

    def p_src(self, evidence: EvidenceHeader | None) -> bool:
        """``p_src``: whether an Evidence row may be read at all, from its header alone.

        Args:
            evidence (EvidenceHeader | None): The row, or None when it does not
                exist.

        Returns:
            bool: True when the row is active, indexed, unexpired and in the
                exact read scope.
        """
        return (
            evidence is not None
            and evidence.state == "active"
            and evidence.route_state == "indexed"
            and (evidence.expires_at is None or evidence.expires_at > self.now)
            and evidence.scope_kind == self.scope_kind
            and evidence.scope_id == self.scope_id
        )

    def p_span(self, span: SpanRef) -> bool:
        """``p_span``: whether a quote span is readable.

        Args:
            span (SpanRef): The span; its Evidence comes from the lookup.

        Returns:
            bool: True when its Evidence passes ``p_src`` and the span intersects no
                withheld range (``w_start < s_end and s_start < w_end``).
        """
        evidence = self.evidence(span.evidence_id)
        return (
            evidence is not None
            and self.p_src(evidence)
            and not _intersects(span.byte_start, span.byte_end, evidence.withheld_spans)
        )

    def p_group(self, spans: Sequence[SpanRef], group: Sequence[int]) -> bool:
        """``p_group``: whether a support group is usable.

        Args:
            spans (Sequence[SpanRef]): The version's quote spans.
            group (Sequence[int]): Indices into ``spans``.

        Returns:
            bool: True when every span of the group passes ``p_span``; an empty
                group supports nothing and fails.
        """
        return bool(group) and all(self.p_span(spans[index]) for index in group)

    def p_ver(self, version: VersionContent) -> bool:
        """``p_ver``: whether a version's content may be read.

        Args:
            version (VersionContent): The version snapshot.

        Returns:
            bool: True when the version is not erased and some support group
                passes ``p_group`` (every group supports the whole statement).
        """
        if version.statement == "":
            return False
        cache: dict[int, bool] = {}

        def readable(index: int) -> bool:
            if index not in cache:
                cache[index] = self.p_span(version.quote_spans[index])
            return cache[index]

        return any(
            bool(group) and all(readable(index) for index in group)
            for group in version.support_groups
        )

    def p_mem(self, lifecycle_state: str, head: VersionContent) -> bool:
        """``p_mem``: whether a memory may be read through its current head.

        World reads admit superseded heads and Corrected-history stand-ins
        through `world_visible` instead.

        Args:
            lifecycle_state (str): The memory's lifecycle.
            head (VersionContent): The memory's current head snapshot.

        Returns:
            bool: True when the lifecycle is ``active`` and the head passes ``p_ver``.
        """
        return lifecycle_state == "active" and self.p_ver(head)

    def p_name(self, row: NameRow, item: NamingMemory | None, head: VersionContent | None) -> bool:
        """``p_name``: whether an ``entity_name`` row counts at read time.

        Args:
            row (NameRow): The name row.
            item (NamingMemory | None): The memory ``row.naming_memory_id`` names, or
                None when it does not exist.
            head (VersionContent | None): The snapshot of that memory's current
                head, or None when it cannot be read.

        Returns:
            bool: True when the naming memory exists and is active, the row's
                ``naming_revision_id`` is its head, and the head passes ``p_ver``.
        """
        return (
            item is not None
            and head is not None
            and item.lifecycle_state == "active"
            and item.current_revision_id == row.naming_revision_id
            and self.p_ver(head)
        )

    def p_evr(self, evidence: EvidenceHeader, *, held: bool) -> bool:
        """``p_evr``: whether an Evidence part may be ranked, rendered as a hit or re-embedded.

        The rule holds per part: its stored vector and postings encode
        every byte, so one withheld range disqualifies the whole part while the
        message's other parts stay eligible.

        Args:
            evidence (EvidenceHeader): The Evidence part.
            held (bool): Whether a pending candidate with ``hold_evidence`` cites
                it.

        Returns:
            bool: True when the part passes ``p_src``, has no withheld range and is
                not held.
        """
        return self.p_src(evidence) and not evidence.withheld_spans and not held

    def p_quote(self, version: VersionContent) -> ReadableQuotes:
        """``p_quote``: the spans a reader may render and the groups it may list.

        Args:
            version (VersionContent): The version snapshot.

        Returns:
            ReadableQuotes: The ``p_span`` span indices and the ``p_group`` group
                indices; both empty for an erased version, which has no spans.
        """
        spans = tuple(
            index for index, span in enumerate(version.quote_spans) if self.p_span(span)
        )
        readable = set(spans)
        groups = tuple(
            index for index, group in enumerate(version.support_groups)
            if group and all(member in readable for member in group)
        )
        return ReadableQuotes(spans, groups)

    def world_intervals(
        self, memory: WorldMemory, index: WorldIndex,
    ) -> tuple[WorldInterval, ...]:
        """The intervals over which a memory is world-visible.

        Only a head that passes ``p_ver`` is world-visible, and never a revision that
        a ``corrects_history`` row targets, whatever the state of the correcting
        memory.

        - An ``active`` memory is visible over its own interval.
        - A ``superseded`` memory is visible over its own interval up to its world
          end, the earliest ``effective_at`` of its witnesses, and never without
          a witness. A witness is a dated ``update`` link whose successor has a
          head that passes ``p_ver`` now, a lifecycle other than ``retracted`` and a
          head reason other than ``retraction``.
        - A ``historical`` memory stands in for each version V it corrects when it
          is V's newest correction (by ``created_commit_seq``, then id): over
          the interval where V would be world-visible, intersected with its own
          validity. A V that fails ``p_ver`` or would not be visible gets no
          stand-in, and an older correction never stands in for a newer one.
        - Every other lifecycle is never visible.

        An own interval starts at ``valid_from``, else at the earliest
        ``effective_at`` of the memory's own ``supersedes(update)`` links, else is
        open, and ends at the earliest of an event's ``valid_to`` and the world
        end. When several sources give the same end, the end comes from the
        first of: the memory's own ``valid_to``, the replaced version's own
        ``valid_to``, a ``valid_from`` witness, a ``source_observed_at`` witness;
        then the finest granularity and the lowest link id.

        Args:
            memory (WorldMemory): The memory, read through its current head.
            index (WorldIndex): The scope's memory corpus at the read's pin.

        Returns:
            tuple[WorldInterval, ...]: The non-empty intervals: at most one own
                interval, or one per version a ``historical`` memory stands in
                for, ordered by that version's revision id. Empty when the
                memory is never world-visible.
        """
        if not self.p_ver(memory.head) or index.corrections(memory.head_revision_id):
            return ()
        if memory.lifecycle_state == "historical":
            found = (
                self._stand_in_interval(memory, target, index)
                for target in sorted(set(memory.corrected_targets))
            )
            return tuple(interval for interval in found if interval is not None)
        own = self._own_interval(memory, index)
        return () if own is None else (own,)

    def world_visible(self, memory: WorldMemory, at: int, index: WorldIndex) -> bool:
        """world_visible(M, t): whether world(t) reads show a memory.

        It is the whole world(t) eligibility of the memory: ``p_ver`` of its head,
        its lifecycle, its witnesses and corrected history, as `world_intervals`
        states them.

        Args:
            memory (WorldMemory): The memory, read through its current head.
            at (int): The world time t in epoch milliseconds.
            index (WorldIndex): The scope's memory corpus at the read's pin.

        Returns:
            bool: True when one of its world intervals contains ``at``.
        """
        return any(interval.contains(at) for interval in self.world_intervals(memory, index))

    def p_past(self, memory: WorldMemory, index: WorldIndex) -> WorldInterval | None:
        """``p_past``: the past state a bench plain Search may return for a memory.

        ``p_past``(m) is ¬``p_mem``(m) ∧ ``p_ver``(head) ∧ m is world-visible over a
        non-empty interval whose end is non-null: a superseded head with a
        witnessed end, or a Corrected-history stand-in in place of the version
        it corrects. It fails closed exactly as `world_intervals` does and
        leaves no trace of what it removes.

        Args:
            memory (WorldMemory): The memory, read through its current head.
            index (WorldIndex): The scope's memory corpus at the read's pin.

        Returns:
            WorldInterval | None: The past interval whose end delivery renders and
                the delivery recheck compares, or None when the memory fails
                ``p_past``. A stand-in for several versions returns the interval
                with the latest end.
        """
        if self.p_mem(memory.lifecycle_state, memory.head):
            return None
        past = [
            interval for interval in self.world_intervals(memory, index)
            if interval.end is not None
        ]
        return min(past, key=_past_order, default=None)

    def p_replaced(self, memory: WorldMemory, index: WorldIndex) -> bool:
        """Whether a memory was replaced by an update that witnesses its end, dated or not.

        It is `p_past` of a superseded head without the date: the head passes ``p_ver``, no
        correction targets it, and a successor that would be a witness has an ``update`` link
        to it, whether or not that link has an ``effective_at``. An undated update leaves the
        replaced memory no end a plain read can deliver, so only a listing of the scope's
        memories admits it this way; every Search keeps `p_past`.

        Args:
            memory (WorldMemory): The memory, read through its current head.
            index (WorldIndex): The scope's memory corpus at the read's pin.

        Returns:
            bool: True for such a superseded memory.
        """
        return (
            memory.lifecycle_state == "superseded"
            and self.p_ver(memory.head)
            and not index.corrections(memory.head_revision_id)
            and any(True for _ in self._witnesses(memory, index))
        )

    def _witnesses(
        self, memory: WorldMemory, index: WorldIndex,
    ) -> Iterator[tuple[IncomingLink, WorldMemory]]:
        """The ``update`` links into a memory, with their successors, that witness its end
        when dated: a successor whose head passes ``p_ver``, not retracted."""
        for link in index.incoming(memory.memory_id):
            successor = index.memory(link.successor_id)
            if (
                link.relation != "update" or successor is None
                or successor.lifecycle_state == "retracted"
                or successor.head_reason == "retraction"
                or not self.p_ver(successor.head)
            ):
                continue
            yield link, successor

    def _world_end(self, memory: WorldMemory, index: WorldIndex) -> WorldEnd | None:
        """The world end of a memory: the earliest end among its witnesses, or None."""
        ends: list[WorldEnd] = []
        for link, successor in self._witnesses(memory, index):
            if link.effective_at is None:
                continue
            granularity = (
                successor.head.time_granularity if link.observed_granularity is None
                else link.observed_granularity
            )
            ends.append(WorldEnd(
                link.effective_at, "witness", granularity, link.effective_basis, link.link_id,
            ))
        return min(ends, key=_end_order, default=None)

    def _own_interval(self, memory: WorldMemory, index: WorldIndex) -> WorldInterval | None:
        """The interval of an active or superseded head, before ``p_ver`` and corrected history."""
        ends = _own_end(memory.head)
        if memory.lifecycle_state == "superseded":
            witnessed = self._world_end(memory, index)
            if witnessed is None:
                return None
            ends.append(witnessed)
        elif memory.lifecycle_state != "active":
            return None
        start = memory.head.valid_from
        if start is None:
            start = min((
                link.effective_at for link in memory.links
                if link.link_type == "supersedes" and link.relation == "update"
                and link.effective_at is not None
            ), default=None)
        return _interval(start, ends)

    def _stand_in_interval(
        self, memory: WorldMemory, revision_id: str, index: WorldIndex,
    ) -> WorldInterval | None:
        """Where a ``historical`` memory stands in for one revision it corrects, or None."""
        newest = max(
            index.corrections(revision_id),
            key=lambda ref: (ref.created_commit_seq, ref.memory_id), default=None,
        )
        replaced = index.head_memory(revision_id)
        if (
            newest is None or newest.memory_id != memory.memory_id
            or replaced is None or not self.p_ver(replaced.head)
        ):
            return None
        own = self._own_interval(replaced, index)
        if own is None:
            return None
        ends = _own_end(memory.head)
        if own.end is not None:
            ends.append(
                replace(own.end, source="corrected_valid_to") if own.end.source == "valid_to"
                else own.end
            )
        starts = [value for value in (own.start, memory.head.valid_from) if value is not None]
        return _interval(max(starts, default=None), ends, revision_id)

    def render_masked(
        self, parts: Sequence[EvidencePart], *, start: int | None = None, end: int | None = None,
        placeholders: Mapping[tuple[str, int, int], str] | None = None,
    ) -> MaskedText:
        """Render one message with every unreadable byte replaced by the marker.

        Bytes of a part that passes ``p_src`` are rendered as written, except bytes
        inside one of its withheld ranges. Every other byte of the rendered
        range is unreadable: a part that fails ``p_src`` (erased, expired, not
        indexed or out of scope) and any byte no supplied part covers, such as
        an erased part whose placeholder lost its offsets. Each maximal run of
        unreadable bytes becomes one `WITHHELD_MARKER`, so readable text is never
        spliced across removed text. Without an explicit edge, an erased part
        numbered before (after) every other part opens (closes) the rendering
        with a marker too.

        Args:
            parts (Sequence[EvidencePart]): Parts of one message (one session,
                request and ordinal) in any order and any state.
            start (int | None): First message byte to render; defaults to the
                lowest ``byte_start`` among parts that are not erased.
            end (int | None): Message byte after the last one to render; defaults
                to the highest ``byte_end`` among parts that are not erased. A
                range edge inside readable text must fall on a character
                boundary.
            placeholders (Mapping[tuple[str, int, int], str] | None): Text to show
                instead of the marker for an unreadable run that is exactly one of
                these ``(evidence_id, byte_start, byte_end)`` ranges of one part (a
                forgotten name's occurrence the owner's settlement withheld, shown to
                its judge as whom it names, ``judge_input``); every other run is the
                marker.

        Returns:
            MaskedText: The rendered text and its segment map. A message whose
                parts are all erased renders as one marker; an empty range
                renders as empty text.

        Raises:
            ValueError: If ``parts`` is empty, mixes messages, repeats a part,
                has overlapping or inconsistent offsets, or if a range edge or a
                withheld range splits a character.
        """
        if not parts:
            raise ValueError("render_masked needs at least one part")
        if len({(part.session_id, part.request_id, part.ordinal) for part in parts}) != 1:
            raise ValueError("render_masked renders the parts of one message")
        if len({part.part_no for part in parts}) != len(parts):
            raise ValueError("render_masked received a part twice")
        placed = sorted((part for part in parts if part.state != "erased"),
                        key=lambda part: part.byte_start)
        if not placed:
            return MaskedText(WITHHELD_MARKER, ())
        for before, after in pairwise(placed):
            if before.byte_end > after.byte_start:
                raise ValueError("Parts of one message cannot overlap")
        low = placed[0].byte_start if start is None else start
        high = placed[-1].byte_end if end is None else end
        if not 0 <= low <= high:
            raise ValueError(f"Invalid message range [{low}, {high})")
        if low == high:
            return MaskedText("", ())
        numbers = [part.part_no for part in placed]
        pending = start is None and any(
            part.state == "erased" and part.part_no < min(numbers) for part in parts
        )
        trailing = end is None and any(
            part.state == "erased" and part.part_no > max(numbers) for part in parts
        )
        pieces: list[str] = []
        segments: list[RenderSegment] = []
        cursor, length = low, 0
        labels = placeholders or {}
        for part in placed:
            if not self.p_src(part):
                continue
            data = part.text.encode("utf-8")
            if len(data) != part.byte_end - part.byte_start:
                raise ValueError("Evidence text does not match its byte offsets")
            withheld = sorted(
                (max(item.byte_start, part.byte_start), min(item.byte_end, part.byte_end))
                for item in part.withheld_spans
                if item.byte_start < part.byte_end and part.byte_start < item.byte_end
            )
            readable: list[tuple[int, int]] = []
            position = part.byte_start
            for withheld_start, withheld_end in withheld:
                if withheld_start > position:
                    readable.append((position, withheld_start))
                position = max(position, withheld_end)
            if position < part.byte_end:
                readable.append((position, part.byte_end))
            for run_start, run_end in readable:
                run_start, run_end = max(run_start, low), min(run_end, high)
                if run_start >= run_end:
                    continue
                if pending or run_start > cursor:
                    marker = WITHHELD_MARKER if pending else labels.get(
                        (part.object_id, cursor, run_start), WITHHELD_MARKER)
                    pieces.append(marker)
                    length += len(marker)
                    pending = False
                try:
                    text = data[run_start - part.byte_start:run_end - part.byte_start].decode()
                except UnicodeDecodeError:
                    raise ValueError(
                        f"Byte range [{run_start}, {run_end}) of {part.object_id} "
                        "splits a character"
                    ) from None
                pieces.append(text)
                segments.append(RenderSegment(
                    part.object_id, run_start, run_end, length, length + len(text),
                ))
                length += len(text)
                cursor = run_end
        if trailing or cursor < high or not segments:
            pieces.append(WITHHELD_MARKER if trailing or not segments else labels.get(
                (placed[-1].object_id, cursor, high), WITHHELD_MARKER))
        return MaskedText("".join(pieces), tuple(segments))
