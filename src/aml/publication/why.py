"""Why v2 (E.2.3): why one memory version, or one Add item, is what it is.

A memory target explains one version of an Item: the change it made, the
outcome of the position that wrote it and the judge verdict that admitted it.
An item target explains one item of a completed Add request (E.2.1). Access is
History's: a forgotten, absent or other-scope target is refused with the same
``scope_denied``, and an unfinished forget closes the scope (D.6).

Why reads only what DM-06 permits: Item, revision, link, candidate (ids and
codes) and Evidence-header rows through the D.2 predicates, and, for a
verdict, the judge result index, the accepted member output (verdict enums and
the reason) and the call's content-free summary, through `JudgeVerdicts`. It
never loads a tombstone, an extraction reply, a proposal or a neighbour
snapshot, and builds no private reconstruction of a judge call's inputs.

The reason rule (E.2.3 #2) is evaluated at read time over the whole call: the
reason is shown only when the member's reason passed the copy check, no
member of the call was issued a tombstone, every Evidence span the call saw
still passes P-SPAN and every version it saw still passes P-VER and is not
forgotten. Otherwise ``reason`` is null and ``reason_withheld`` is true.
Tombstone verdicts are never exposed, not even as a count.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Annotated, Final, Literal, Protocol, Self

from pydantic import BeforeValidator, Field, model_validator

from aml.core.entities import USER, QuoteSpanV1, ReasonCode
from aml.core.errors import AMLError
from aml.core.identity import ULID, ArtifactId
from aml.core.judge_output import Relation, SupportVerdict
from aml.core.outcome_table import OutcomeCategory, outcome_rule
from aml.core.query import ScopeV1
from aml.core.readability import Readability, SpanRef
from aml.core.semantic import SystemTimestamp
from aml.core.technical_types import ExternalId
from aml.publication.add_items import AddCompletions, ItemRef, frozen_item
from aml.publication.history import (
    HistoryChange,
    MemoryQuoteV1,
    MemoryReads,
    MemoryRows,
    Quotes,
    QuoteSpeakers,
    WireValue,
    cite,
    erased_through,
    history_change,
    json_tuple,
    span_quote,
    version_quotes,
)
from aml.store.domain import CandidateRowV2, ItemRowV2, RevisionRowV2

MECHANICAL_CODES: Final = frozenset({
    "restatement_satisfied", "duplicate_in_add", "secret_rejected", "extraction_item_invalid",
    "unsupported_sensitive_inference", "extraction_capacity_exceeded",
})
"""Codes whose position no judge decided (B.7, B.8); their Why has no verdict."""

UNAVAILABLE_CODES: Final = frozenset({"review_unavailable", "review_unavailable_repeated"})
"""Codes whose judge reply was refused or whose call exhausted its timeouts (C.8 #4/#6).
The position's valid ``unavailable`` result has no verdict."""

EntityVerdict = Literal["confirmed", "repicked", "new", "ambiguous", "mechanical"]
"""How the explained version's primary anchor (slot 0) was settled (E.2.3 #1)."""

SlotVerdict = Literal["confirm", "repick", "ambiguous"]
"""A judge's entity verdict for one slot (C.7)."""

_SLOT_VERDICTS: Final[dict[SlotVerdict, EntityVerdict]] = {
    "confirm": "confirmed", "repick": "repicked", "ambiguous": "ambiguous",
}


class WhyMemoryTargetV2(WireValue):
    """Explain one version of an Item: ``revision_id``, or its current head."""

    kind: Literal["memory"]
    memory_id: ULID
    revision_id: ULID | None = None


class WhyItemTargetV2(WireValue):
    """Explain one item of a completed Add request (E.2.1)."""

    kind: Literal["item"]
    session_id: ExternalId
    request_id: ExternalId
    item_ref: ItemRef


WhyTargetV2 = Annotated[WhyMemoryTargetV2 | WhyItemTargetV2, Field(discriminator="kind")]
"""A Why target (E.2.3); candidate and run targets and cursors are deleted."""


class WhyOutcomeV2(WireValue):
    """The disposition of the position behind the explained version or item (C.10)."""

    category: OutcomeCategory
    reason_code: ReasonCode

    @model_validator(mode="after")
    def check_category(self) -> Self:
        """Keep the category the code's (C.10)."""
        if outcome_rule(self.reason_code).category != self.category:
            raise ValueError("An outcome's category is its code's")
        return self


class WhyRelationV2(WireValue):
    """The judge's relation to one readable issued target version."""

    memory_id: ULID
    revision_id: ULID
    relation: Relation


class WhyVerdictV2(WireValue):
    """The judge verdict that admitted the explained content (E.2.3 ``verdict``)."""

    verdict_ref: ArtifactId
    judged_at: SystemTimestamp
    replayed: bool
    support: SupportVerdict
    entity: EntityVerdict
    relations: Annotated[tuple[WhyRelationV2, ...], BeforeValidator(json_tuple)]
    reason: Annotated[str, Field(min_length=1)] | None
    reason_withheld: bool

    @model_validator(mode="after")
    def check_reason(self) -> Self:
        """Withhold the reason exactly when it is absent."""
        if (self.reason is None) != self.reason_withheld:
            raise ValueError("A reason is either shown or withheld")
        return self


class MemoryWhyV2(WireValue):
    """One explanation (E.2.3 ``data``).

    ``memory_id`` to ``change`` describe the explained version: the target
    version for a memory target, the version a formal item wrote for an item
    target, else null. ``outcome`` is null for a privacy version; ``verdict``
    is null for a privacy version, a mechanical outcome and an unavailable
    one (C.8 #4/#6). ``quotes`` are the
    version's P-QUOTE spans (none when redacted) or a non-formal item's cited
    spans that pass P-SPAN.
    """

    target: WhyTargetV2
    observed_at: SystemTimestamp
    memory_id: ULID | None
    revision_id: ULID | None
    version: Annotated[int, Field(ge=1)] | None
    change: HistoryChange | None
    outcome: WhyOutcomeV2 | None
    verdict: WhyVerdictV2 | None
    quotes: Quotes
    redacted: bool

    @model_validator(mode="after")
    def check_version(self) -> Self:
        """Name the explained version completely or not at all; redact no quotes."""
        named = (self.memory_id, self.revision_id, self.version, self.change)
        if None in named and named != (None, None, None, None):
            raise ValueError("An explained version is named completely")
        if self.redacted and self.quotes:
            raise ValueError("A redacted explanation shows no quotes")
        return self


@dataclass(frozen=True, slots=True)
class JudgeCallSummary:
    """The content-free summary of one judge call (C.13 ``JudgeCallManifestV1``; judge-Q22).

    Attributes:
        tombstone_issued (bool): Any member of the call was issued a tombstone.
        evidence_spans (tuple[SpanRef, ...]): Every quote span and shown
            message part of every member, whole parts, sorted and unique; an
            erased part's ``(0, 0)`` placeholder always withholds the reason.
        revision_ids (tuple[str, ...]): Every version the call showed: targets,
            history, entity facts and names, sorted and unique.
    """

    tombstone_issued: bool
    evidence_spans: tuple[SpanRef, ...]
    revision_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class JudgeVerdictRecord:
    """What Why may read of one accepted judge verdict (DM-06, E.2.3).

    Attributes:
        verdict_ref (str): The ``result_index:judge``.
        judged_at (int): When that index was recorded.
        replayed (bool): The commit reused a recorded verdict.
        support (SupportVerdict): The member's support verdict.
        slot0 (SlotVerdict | None): The judge's verdict for slot 0, or None when
            slot 0 was bound mechanically (self, participant or SDK).
        relations (tuple[tuple[str, str, Relation], ...]): ``(memory_id,
            revision_id, relation)`` per issued target the verdict named; never
            a tombstone verdict.
        reason (str): The member's reason.
        reason_readable (bool): The reason passed the member's copy check.
        call (JudgeCallSummary): The whole call's content-free summary.
    """

    verdict_ref: str
    judged_at: int
    replayed: bool
    support: SupportVerdict
    slot0: SlotVerdict | None
    relations: tuple[tuple[str, str, Relation], ...]
    reason: str
    reason_readable: bool
    call: JudgeCallSummary


class JudgeVerdicts(Protocol):
    """The accepted judge verdicts Why reads (C.13 judge records; the judge records' reader)."""

    def verdict(self, rows: MemoryRows, verdict_ref: str) -> JudgeVerdictRecord | None:
        """Read one accepted verdict through the read's guarded rows.

        Args:
            rows (MemoryRows): The read's guarded rows.
            verdict_ref (str): The ``result_index:judge`` that admitted a version,
                or that a frozen Add item names.

        Returns:
            JudgeVerdictRecord | None: The verdict's readable parts, or None when
                the reference names a valid ``unavailable`` result (C.8 #4/#6)
                rather than a verdict, or privacy erasure removed the result.

        Raises:
            AMLError: When the records are missing or inconsistent.
        """
        ...


@dataclass(frozen=True, slots=True)
class _Explained:
    """The version an explanation names, and whether it is at or before an erase."""

    item: ItemRowV2
    revision: RevisionRowV2
    erased: bool


@dataclass(frozen=True, slots=True)
class _Plan:
    """Everything one explanation reads before `MemoryRows.finish`."""

    explained: _Explained | None
    code: ReasonCode | None
    verdict: JudgeVerdictRecord | None
    minted: bool
    spans: tuple[QuoteSpanV1, ...]
    cited: bool


def _explained(rows: MemoryRows, memory_id: str, revision_id: str | None) -> _Explained:
    """Read the named version and its Item's newer versions, refused like History."""
    item = rows.item(memory_id)
    if item is None:
        raise AMLError("scope_denied", "Why target is unavailable")
    chain = [rows.head(item)]
    while revision_id is not None and chain[-1].object_id != revision_id:
        older = rows.previous(chain[-1])
        if older is None:
            raise AMLError("scope_denied", "Why target is unavailable")
        chain.append(older)
    return _Explained(item, chain[-1], chain[-1].version <= erased_through(chain))


def _minted(rows: MemoryRows, revision: RevisionRowV2) -> bool:
    """Whether the transaction that wrote ``revision`` created its primary anchor."""
    anchor = revision.snapshot.entity_id
    if anchor in {"", USER}:
        return False
    entity = rows.row("entity", anchor)
    return entity is not None and (
        entity["created_by_run_id"], entity["commit_seq"],
    ) == (revision.origin_run_id, revision.commit_seq)


def _candidate(rows: MemoryRows, candidate_id: str) -> CandidateRowV2:
    raw = rows.row("candidate", candidate_id)
    if raw is None:
        raise AMLError("parity_mismatch", "A position lost its candidate row")
    return CandidateRowV2.model_validate(raw)


def _verdict_inputs(
    rows: MemoryRows, verdicts: JudgeVerdicts, verdict_ref: str, code: ReasonCode,
) -> JudgeVerdictRecord | None:
    """Read a verdict and record every row its reason rule and relations read.

    An ``unavailable`` result has no verdict; its outcome must carry an
    unavailable code, and such a code cannot name an accepted verdict.
    A privacy erase deletes a result whose call depended on erased Evidence
    (C.13, INV P06); the version or item still naming it shows no verdict.
    """
    record = verdicts.verdict(rows, verdict_ref)
    if record is None:
        if code not in UNAVAILABLE_CODES and rows.row("artifact", verdict_ref) is not None:
            raise AMLError("binding_mismatch", "A judged position names no accepted verdict")
        return None
    if code in UNAVAILABLE_CODES:
        raise AMLError("binding_mismatch", "An unavailable position names an accepted verdict")
    for evidence_id in sorted({span.evidence_id for span in record.call.evidence_spans}):
        rows.evidence(evidence_id)
    for revision_id in (*record.call.revision_ids, *(target for _, target, _ in record.relations)):
        _record_version(rows, revision_id)
    return record


def _record_version(rows: MemoryRows, revision_id: str) -> None:
    """Record a version, its Item and its Evidence so its readability can be checked."""
    revision = rows.revision(revision_id)
    if revision is not None and rows.item(revision.memory_id) is not None:
        cite(rows, revision.snapshot)


def _readable_version(rows: MemoryRows, read: Readability, revision_id: str) -> bool:
    """Whether a version still exists in an Item that exists, and passes P-VER."""
    revision = rows.revision(revision_id)
    return (
        revision is not None and rows.item(revision.memory_id) is not None
        and read.p_ver(revision.snapshot)
    )


def _verdict(
    rows: MemoryRows, read: Readability, record: JudgeVerdictRecord, minted: bool,
) -> WhyVerdictV2:
    """Build the verdict under the read's predicates and the whole-call reason rule."""
    shown = record.reason_readable and not record.call.tombstone_issued and all(
        span.byte_start < span.byte_end and read.p_span(span)
        for span in record.call.evidence_spans
    ) and all(_readable_version(rows, read, revision) for revision in record.call.revision_ids)
    entity: EntityVerdict = (
        "mechanical" if record.slot0 is None else "new" if minted
        else _SLOT_VERDICTS[record.slot0]
    )
    return WhyVerdictV2(
        verdict_ref=ArtifactId(record.verdict_ref), judged_at=record.judged_at,
        replayed=record.replayed, support=record.support, entity=entity,
        relations=tuple(sorted(
            (
                WhyRelationV2(memory_id=memory_id, revision_id=revision_id, relation=relation)
                for memory_id, revision_id, relation in record.relations
                if _readable_version(rows, read, revision_id)
            ),
            key=lambda relation: (relation.memory_id, relation.revision_id),
        )),
        reason=record.reason if shown else None,
        reason_withheld=not shown,
    )


class WhyReader:
    """Explain one memory version or one Add item (E.2.3).

    Args:
        reads (MemoryReads): The shared guarded read entry.
        speakers (QuoteSpeakers): The speaker resolution Search delivery uses.
        verdicts (JudgeVerdicts): The accepted judge verdicts (C.13).
        completions (AddCompletions): The Add completion records (I14).
    """

    def __init__(
        self, reads: MemoryReads, speakers: QuoteSpeakers, verdicts: JudgeVerdicts,
        completions: AddCompletions,
    ) -> None:
        self._reads, self._speakers = reads, speakers
        self._verdicts, self._completions = verdicts, completions

    def consume(
        self,
        target: WhyMemoryTargetV2 | WhyItemTargetV2,
        *,
        scope: ScopeV1,
        consumer: Callable[[MemoryWhyV2], None],
    ) -> None:
        """Deliver one explanation while the scope's pin is held.

        Args:
            target (WhyMemoryTargetV2 | WhyItemTargetV2): What to explain.
            scope (ScopeV1): The caller's exact scope.
            consumer (Callable[[MemoryWhyV2], None]): Called once, inside the
                pinned callback, with the explanation.

        Raises:
            AMLError: ``scope_denied`` for a forgotten, absent or other-scope
                target, with identical public bytes; ``index_rebuilding``
                during an unfinished forget or erasure; ``reception_stale`` when
                the store changed during the read; the shared entry's refusals.
        """
        self._reads.consume(scope, lambda rows: self.read(rows, target), consumer)

    def read(self, rows: MemoryRows, target: WhyMemoryTargetV2 | WhyItemTargetV2) -> MemoryWhyV2:
        """Build one explanation inside a pinned scope's callback.

        Args:
            rows (MemoryRows): The read's guarded rows.
            target (WhyMemoryTargetV2 | WhyItemTargetV2): What to explain.

        Returns:
            MemoryWhyV2: The explanation, evaluated at the clock read after
                every read was rechecked.
        """
        rows.authorize()
        plan = (
            self._memory_plan(rows, target) if isinstance(target, WhyMemoryTargetV2)
            else self._item_plan(rows, target)
        )
        rows.finish()
        now = rows.clock()
        read = rows.readability(now)
        explained = plan.explained
        redacted = explained is not None and (
            explained.erased or not read.p_ver(explained.revision.snapshot)
        )
        quotes: tuple[MemoryQuoteV1, ...] = ()
        if explained is not None and not redacted:
            quotes = version_quotes(explained.revision.snapshot, read, rows, self._speakers)
        elif explained is None and plan.cited:
            quotes = self._span_quotes(rows, read, plan.spans)
        revision = None if explained is None else explained.revision
        return MemoryWhyV2(
            target=target, observed_at=now,
            memory_id=None if revision is None else revision.memory_id,
            revision_id=None if revision is None else revision.object_id,
            version=None if revision is None else revision.version,
            change=None if revision is None else history_change(revision.op, revision.reason_code),
            outcome=None if plan.code is None else WhyOutcomeV2(
                category=outcome_rule(plan.code).category, reason_code=plan.code,
            ),
            verdict=None if plan.verdict is None else _verdict(
                rows, read, plan.verdict, plan.minted,
            ),
            quotes=quotes, redacted=redacted,
        )

    def _memory_plan(self, rows: MemoryRows, target: WhyMemoryTargetV2) -> _Plan:
        """Read a memory target's version, its position's code and its verdict."""
        explained = _explained(rows, target.memory_id, target.revision_id)
        revision = explained.revision
        cite(rows, revision.snapshot)
        if revision.source_candidate_id is None:
            return _Plan(explained, None, None, False, (), False)
        candidate = _candidate(rows, revision.source_candidate_id)
        if candidate.revision_id != revision.object_id:
            raise AMLError("parity_mismatch", "A version's candidate wrote another revision")
        return self._judged(rows, explained, candidate.reason_code, candidate.judge_result_ref)

    def _judged(
        self, rows: MemoryRows, explained: _Explained | None, code: ReasonCode,
        verdict_ref: str | None, spans: tuple[QuoteSpanV1, ...] = (), cited: bool = False,
    ) -> _Plan:
        """Read the verdict of a judged position; a mechanical code has none."""
        if code in MECHANICAL_CODES or verdict_ref is None:
            return _Plan(explained, code, None, False, spans, cited)
        record = _verdict_inputs(rows, self._verdicts, verdict_ref, code)
        minted = record is not None and explained is not None and _minted(
            rows, explained.revision,
        )
        return _Plan(explained, code, record, minted, spans, cited)

    def _item_plan(self, rows: MemoryRows, target: WhyItemTargetV2) -> _Plan:
        """Read an item target's frozen item, its written version and its verdict."""
        record = self._completions.completion(rows, target.session_id, target.request_id)
        item = None if record is None else frozen_item(record.items, target.item_ref)
        if item is None:
            raise AMLError("scope_denied", "Why target is unavailable")
        explained = None
        if item.revision_id is not None:
            revision = rows.revision(item.revision_id)
            if revision is None or rows.item(revision.memory_id) is None:
                raise AMLError("scope_denied", "Why target is unavailable")
            explained = _explained(rows, revision.memory_id, revision.object_id)
            cite(rows, explained.revision.snapshot)
        cited = explained is None and item.reason_code != "denied_by_forget"
        for evidence_id in sorted({span.evidence_id for span in item.quote_spans}):
            rows.evidence(evidence_id)
        # The frozen item's own verdict locator: the candidate row's current one
        # may belong to a later re-entry of the same position (E.2.3).
        return self._judged(
            rows, explained, item.reason_code, item.judge_result_ref, item.quote_spans, cited,
        )

    def _span_quotes(
        self, rows: MemoryRows, read: Readability, spans: Sequence[QuoteSpanV1],
    ) -> tuple[MemoryQuoteV1, ...]:
        """Render a non-formal item's cited spans that pass P-SPAN, in their order."""
        quotes: list[MemoryQuoteV1] = []
        for span in spans:
            if read.p_span(span):
                quotes.append(span_quote(rows, span, self._speakers, group=None))
        return tuple(quotes)

