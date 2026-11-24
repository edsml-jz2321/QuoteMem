"""Extraction units: bounded runs of whole Evidence parts of one request.

A unit is a maximal run of whole Evidence parts of one request in message
order, packed greedily within its bounds: at most ``max_words`` words, at most
``max_parts`` parts, at most 1,048,576 bytes, and at most 256 fresh sources.
A unit never spans requests, and a part is never split. Units are a
pure function of the owner's frozen prefix and its turn labels, so recovery
recomputes them exactly.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from quotemem.core.add_read_view import EvidenceRecord, MessageKey
from quotemem.core.errors import QuoteMemError
from quotemem.core.extraction_policy import MAX_UNIT_PARTS, MAX_UNIT_WORDS, ExtractionPolicyV2
from quotemem.core.identity import IntegrityDigest
from quotemem.core.statement import MAX_FRESH_SOURCES, MAX_MESSAGE_BYTES
from quotemem.ingestion.sources import (
    LabelView,
    TurnLabel,
    governing_label,
    message_key,
    opens_source,
    word_count,
)
from quotemem.store.writer import MAX_TX_ROWS


@dataclass(frozen=True, slots=True)
class Unit:
    """One extraction unit of an owner.

    Attributes:
        unit_no (int): The unit's number within its owner, from 0.
        parts (tuple[EvidenceRecord, ...]): Whole parts of one request in
            message order.
    """

    unit_no: int
    parts: tuple[EvidenceRecord, ...]

    @property
    def session_id(self) -> str:
        """The session of the unit's request."""
        return self.parts[0].session_id

    @property
    def request_id(self) -> str:
        """The request every part of the unit belongs to."""
        return self.parts[0].request_id


@dataclass(frozen=True, slots=True)
class _Open:
    """The source in progress at the end of the pending unit.

    Attributes:
        message (MessageKey): Its message.
        end (int): The message byte after the last readable part it covers.
        label (TurnLabel | None): The label that governs it.
    """

    message: MessageKey
    end: int
    label: TurnLabel | None


@dataclass(slots=True)
class _Pending:
    """The unit being filled, its running totals and its source in progress."""

    parts: list[EvidenceRecord]
    words: int = 0
    size: int = 0
    sources: int = 0
    open: _Open | None = None


def _size(part: EvidenceRecord) -> int:
    return part.byte_end - part.byte_start


def _join(pending: _Pending, part: EvidenceRecord, labels: LabelView) -> tuple[int, _Open | None]:
    """Count the sources a part opens when it joins the pending unit.

    A part that is not erased opens a source unless a source of its message is
    in progress, and one more for every later turn label up to its end that
    opens a source (`opens_source`), exactly as `source_runs` cuts the unit.

    Returns:
        tuple[int, _Open | None]: The sources it opens and the source then in
            progress.
    """
    key = message_key(part)
    current = pending.open if pending.open is not None and pending.open.message == key else None
    if part.state == "erased":
        return 0, current
    turns = labels.turns(key)
    if current is None:
        opened, label, low = 1, governing_label(turns, part.byte_start), part.byte_start + 1
    else:
        opened, label, low = 0, current.label, current.end
    for turn in turns:
        if low <= turn.byte_start < part.byte_end and opens_source(part, label, turn):
            opened, label = opened + 1, turn
    return opened, _Open(key, part.byte_end, label)


def split_units(
    parts: Sequence[EvidenceRecord],
    labels: LabelView,
    *,
    max_words: int,
    max_parts: int,
    first_unit_no: int = 0,
) -> tuple[Unit, ...]:
    """Pack an owner's prefix into extraction units.

    Args:
        parts (Sequence[EvidenceRecord]): Every part of the owner's frozen
            prefix in message order: request by request, and within a message
            by ``part_no``.
        labels (LabelView): The frozen turn labels, which split sources.
        max_words (int): The policy's ``segment_words`` (16,000).
        max_parts (int): The policy's ``max_parts_per_unit`` (2,000).
        first_unit_no (int): The number of the first unit.

    Returns:
        tuple[Unit, ...]: The units in order, numbered consecutively.

    Raises:
        QuoteMemError: ``budget_exceeded`` when one part alone exceeds the word or
            byte bound, before any provider call.
    """
    if (not 1 <= max_words <= MAX_UNIT_WORDS or not 1 <= max_parts <= MAX_UNIT_PARTS
        or first_unit_no < 0):
        raise QuoteMemError("invalid_request", "Extraction unit limits are outside their bounds")
    units: list[Unit] = []
    pending = _Pending([])

    def close() -> None:
        nonlocal pending
        if pending.parts:
            units.append(Unit(first_unit_no + len(units), tuple(pending.parts)))
        pending = _Pending([])

    for part in parts:
        words, size = word_count(part.text), _size(part)
        if words > max_words or size > MAX_MESSAGE_BYTES:
            raise QuoteMemError("budget_exceeded", "One Evidence part exceeds the unit bounds")
        if pending.parts and (pending.parts[-1].session_id, pending.parts[-1].request_id) != (
            part.session_id, part.request_id,
        ):
            close()
        sources, opened = _join(pending, part, labels)
        if pending.parts and (
            len(pending.parts) + 1 > max_parts
            or pending.words + words > max_words
            or pending.size + size > MAX_MESSAGE_BYTES
            or pending.sources + sources > MAX_FRESH_SOURCES
        ):
            close()
            sources, opened = _join(pending, part, labels)
        if sources > MAX_FRESH_SOURCES:
            raise QuoteMemError("budget_exceeded",
                                "One whole Evidence part exceeds the source bound")
        pending.parts.append(part)
        pending.words += words
        pending.size += size
        pending.sources += sources
        pending.open = opened
    close()
    return tuple(units)


@dataclass(frozen=True, slots=True)
class StorageBudget:
    """Exact independent transaction maxima measured by the Writer's StoragePlan.

    Attributes:
        artifact_rowops (int): Largest pre-call Artifact transaction.
        business_rowops (int): Largest admission transaction for 128 facts;
            formal components are separately bounded at the barrier.
        storage_plan_digest (IntegrityDigest): Identity of the measured plan.
    """

    artifact_rowops: int
    business_rowops: int
    storage_plan_digest: IntegrityDigest

    def __post_init__(self) -> None:
        if any(type(value) is not int or value < 0 for value in (
            self.artifact_rowops, self.business_rowops,
        )):
            raise QuoteMemError("schema_invalid", "RowOp budgets must be nonnegative integers")

    @property
    def fits(self) -> bool:
        """Whether both independently materialized transactions fit the Writer ceiling."""
        return max(self.artifact_rowops, self.business_rowops) <= MAX_TX_ROWS


def plan_units(
    parts: Sequence[EvidenceRecord], labels: LabelView, *, policy: ExtractionPolicyV2,
    measure: Callable[[Unit, int], StorageBudget],
) -> tuple[Unit, ...]:
    """Pack whole parts and prove physical capacity before returning any unit.

    Args:
        parts (Sequence[EvidenceRecord]): The owner's complete ordered prefix.
        labels (LabelView): Frozen source-label decisions.
        policy (ExtractionPolicyV2): The owner's frozen production policy.
        measure (Callable[[Unit, int], StorageBudget]): Materialize the worst legal
            artifacts and admission rows through the real Writer StoragePlan,
            for exactly this unit and the policy's fact maximum (128). Counts
            must be monotone as parts are appended; no estimate grants capacity.

    Returns:
        tuple[Unit, ...]: Consecutively numbered units, each measured exactly.

    Raises:
        QuoteMemError: ``budget_exceeded`` when one whole part cannot fit. Other
            measurement errors propagate. No provider is called here.
    """
    sizes = [(word_count(part.text), _size(part)) for part in parts]
    result: list[Unit] = []

    def fits(members: tuple[EvidenceRecord, ...]) -> bool:
        try:
            return measure(Unit(len(result), members), policy.max_facts_per_unit).fits
        except QuoteMemError as error:
            if error.code != "budget_exceeded" or len(members) == 1:
                raise
            return False

    start = 0
    while start < len(parts):
        pending = _Pending([])
        for part, (words, size) in zip(parts[start:], sizes[start:], strict=True):
            if pending.parts and (part.session_id, part.request_id) != (
                pending.parts[0].session_id, pending.parts[0].request_id,
            ):
                break
            sources, opened = _join(pending, part, labels)
            if (pending.words + words > policy.segment_words
                or pending.size + size > MAX_MESSAGE_BYTES
                or len(pending.parts) + 1 > policy.max_parts_per_unit
                or pending.sources + sources > MAX_FRESH_SOURCES):
                break
            pending.parts.append(part)
            pending.words += words
            pending.size += size
            pending.sources += sources
            pending.open = opened
        remaining = tuple(pending.parts)
        whole = len(remaining)
        # Counts are monotone as parts are appended, so a whole candidate that
        # fits is its own longest fitting prefix: one measurement proves it.
        if whole > 1 and fits(remaining):
            result.append(Unit(len(result), remaining))
            start += whole
            continue
        if not remaining or not fits(remaining[:1]):
            raise QuoteMemError("budget_exceeded", "One whole part exceeds unit capacity")
        proven = 1
        while proven < whole:
            trial = min(2 * proven, whole)
            if trial < whole and fits(remaining[:trial]):
                proven = trial
                continue
            low, high = proven, trial
            while high - low > 1:
                middle = (low + high) // 2
                if fits(remaining[:middle]):
                    low = middle
                else:
                    high = middle
            proven = low
            break
        result.append(Unit(len(result), remaining[:proven]))
        start += proven
    return tuple(result)
