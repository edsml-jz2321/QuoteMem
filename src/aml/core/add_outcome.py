"""Content-free Add results: per-request counts, the completed receipt and the owner census."""

from __future__ import annotations

from collections import Counter
from collections.abc import Collection, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Literal, Self, cast, get_args

from pydantic import BeforeValidator, Field, model_validator

from aml.core.entities import CandidateStatus, EntityRef, PositionKey, ReasonCode, Rejudge
from aml.core.errors import AMLError
from aml.core.identity import ULID, ArtifactId, ContentBindingDigest
from aml.core.outcome_table import (
    OUTCOME_CATEGORIES,
    OUTCOME_TABLE,
    OutcomeCategory,
    OutcomeItem,
    outcome_rule,
    tally,
)
from aml.core.request_budget import MAX_OWNER_RESUMES
from aml.core.statement import MAX_MESSAGE_BYTES
from aml.core.technical_types import ContentDigest, NonNegativeInt
from aml.store.metadata import OpaqueIdSpec, RefSpec, StrictModel

if TYPE_CHECKING:
    from aml.core.resolution import Round2EntryV1


class AddOutcomeV1(StrictModel):
    """Count disposed semantic items, not rows, targets, or model calls.

    Each original extraction output position contributes once. A correction
    expanding to multiple targets remains one item. Attribution uses the newest
    request among the assertion's own source spans; context and merged target
    support do not independently create a result. Counts freeze at the first
    successful completion, so later resolution never rewrites an old receipt.
    A formal effect (added, corrected or supplemented) of a position whose own
    spans all belong to already completed requests, such as a joint Add's
    re-extraction of the previous Add or a re-entered pending position, counts
    toward the newest request of the owner that published it, the Add whose
    processing caused it: every formal change is counted exactly once. Such a
    position's other dispositions stay with the completed receipt.
    """

    added: NonNegativeInt = 0
    corrected: NonNegativeInt = 0
    supplemented: NonNegativeInt = 0
    unchanged: NonNegativeInt = 0
    rejected: NonNegativeInt = 0
    unresolved: NonNegativeInt = 0


class CompletedAddOutcomeV1(StrictModel):
    """One request's immutable result, visible only after its owner succeeds.

    ``recovery_count`` is the completing owner's counted automatic/explicit
    resumes: 0 for a normal completion, otherwise the durable total it needed.
    ``skipped`` is an operator's skip of a quarantined cycle
    (``store.quarantine_skip``): the request completed without the rest of its
    owner's work, its counts are what that owner committed before its fault,
    and it is visible once that owner failed ``input_abandoned``.
    """

    state: Literal["completed", "skipped"] = "completed"
    request_key: ContentDigest
    completed_by_run_id: Annotated[
        ULID, RefSpec("run", ref_mode="historical_locator", fulfillment="none"),
    ]
    recovery_count: Annotated[int, Field(ge=0, le=MAX_OWNER_RESUMES)]
    outcome: AddOutcomeV1


# The owner census (T-S2b-1; ingest-Q14; E.4). Every field is nullable: None means
# the input it counts was unavailable (INV H09), 0 is a measured zero. The census
# holds counts, seconds and position keys only, never source or member text.

CensusAction = Literal["unreviewed", "review", "borrow", "quiet"]
"""C.3's re-entry actions as the census counts them (``ingestion/reentry.ReentryAction``)."""

CensusCause = Literal[
    "repick", "inheritance_gap", "reentry_woken",
    "fence_1", "fence_2", "fence_3", "fence_4", "fence_5", "fence_6",
]
"""I10's round-2 causes as the census counts them (``core/resolution.Round2Cause``)."""

NonNegativeSeconds = Annotated[float, Field(ge=0, allow_inf_nan=False)]


def _json_tuple(value: object) -> object:
    """Read a stored JSON array as the tuple it encodes, without coercing its members."""
    return tuple(cast(list[object], value)) if isinstance(value, list) else value


PositionKeys = Annotated[tuple[PositionKey, ...], BeforeValidator(_json_tuple)]
"""Position keys as a tuple; a stored JSON array reads back as the same tuple."""


class ReentryCensusV1(StrictModel):
    """One owner's re-entry selection (C.3; I9) by action.

    ``selected`` counts the pending positions the owner re-entered; each took
    exactly one action. ``digest_changed`` counts selected members whose full
    ``member_input_digest`` changed since their latest review (eval-Q16).
    """

    selected: NonNegativeInt | None
    quiet: NonNegativeInt | None
    borrow: NonNegativeInt | None
    review: NonNegativeInt | None
    unreviewed: NonNegativeInt | None
    digest_changed: NonNegativeInt | None

    @model_validator(mode="after")
    def one_action_each(self) -> Self:
        """Every selected member takes one action, and only selected members change digests."""
        actions = (self.quiet, self.borrow, self.review, self.unreviewed)
        if self.selected is not None and None not in actions and (
            sum(value or 0 for value in actions) != self.selected
        ):
            raise ValueError("Re-entry actions must partition the selected members")
        if None not in (self.selected, self.digest_changed) and (
            (self.digest_changed or 0) > (self.selected or 0)
        ):
            raise ValueError("Only selected members can change their input digest")
        return self

    @classmethod
    def from_actions(
        cls, actions: Iterable[CensusAction] | None, *, digest_changed: int | None,
    ) -> ReentryCensusV1:
        """Count the actions S4a's selection gave the owner's re-entered members.

        Args:
            actions: One action per selected member; None when the selection's
                actions are unavailable.
            digest_changed: Selected members whose full input digest changed, or
                None when unavailable.

        Returns:
            The census, every count None when ``actions`` is None.
        """
        if actions is None:
            return cls(
                selected=None, quiet=None, borrow=None, review=None, unreviewed=None,
                digest_changed=digest_changed,
            )
        counted = Counter(actions)
        unknown = set(counted) - set(get_args(CensusAction))
        if unknown:
            raise ValueError("Re-entry census received an unknown action")
        return cls(
            selected=sum(counted.values()), quiet=counted["quiet"], borrow=counted["borrow"],
            review=counted["review"], unreviewed=counted["unreviewed"],
            digest_changed=digest_changed,
        )


class Round2CausesV1(StrictModel):
    """Round-2 members by the one cause I10 gave each (store-Q2, store-Q3)."""

    repick: NonNegativeInt
    inheritance_gap: NonNegativeInt
    reentry_woken: NonNegativeInt
    fence_1: NonNegativeInt
    fence_2: NonNegativeInt
    fence_3: NonNegativeInt
    fence_4: NonNegativeInt
    fence_5: NonNegativeInt
    fence_6: NonNegativeInt

    @property
    def total(self) -> int:
        """Members counted over every cause."""
        return sum(getattr(self, cause) for cause in get_args(CensusCause))


class Round2CensusV1(StrictModel):
    """One owner's round 2 (C.2 #7): its members, their causes and how they ended.

    ``by_cause`` partitions ``members``: each member has exactly one cause.
    ``target_changed`` counts round-2 members that ended ``target_changed``;
    ``judge_requests`` counts round-2 physical judge requests.
    """

    members: NonNegativeInt | None
    by_cause: Round2CausesV1 | None
    target_changed: NonNegativeInt | None
    judge_requests: NonNegativeInt | None

    @model_validator(mode="after")
    def causes_partition(self) -> Self:
        """The cause bins sum to the members; only members can end ``target_changed``."""
        if self.members is not None and self.by_cause is not None and (
            self.by_cause.total != self.members
        ):
            raise ValueError("Round-2 causes must partition the round-2 members")
        if None not in (self.members, self.target_changed) and (
            (self.target_changed or 0) > (self.members or 0)
        ):
            raise ValueError("Only round-2 members can end target_changed")
        return self

    @classmethod
    def from_entries(
        cls, entries: Iterable[Round2EntryV1] | None, *,
        target_changed: int | None, judge_requests: int | None,
    ) -> Round2CensusV1:
        """Count I10's round-2 entries, one cause per member (store-Q3).

        Raises:
            ValueError: If a position appears twice, which would give it two causes.
        """
        if entries is None:
            return cls(
                members=None, by_cause=None, target_changed=target_changed,
                judge_requests=judge_requests,
            )
        selected = list(entries)
        if len({entry.position_key for entry in selected}) != len(selected):
            raise ValueError("A round-2 member has exactly one cause")
        counted = Counter(entry.cause for entry in selected)
        causes = Round2CausesV1.model_validate(
            {cause: counted[cause] for cause in get_args(CensusCause)},
        )
        if causes.total != len(selected):
            raise ValueError("Round-2 census received an unknown cause")
        return cls(
            members=len(selected), by_cause=causes, target_changed=target_changed,
            judge_requests=judge_requests,
        )


class OwnerPhaseSecondsV1(StrictModel):
    """Monotonic seconds the owner spent in each C.2 phase (E.4; eval-Q11, ingest-Q13).

    ``labels`` is C.2 step 0's label stage; ``extraction`` I8 from the first unit
    to the barrier (steps 1-2); ``embedding`` I8's B.9 batches and ``ProposalV2``
    writes; ``judge_round_1`` re-entry selection, nomination and round-1 review
    (steps 4-5); ``judge_round_2`` step 7's rebuild, nomination and review;
    ``commit`` the settlement transaction, resolution and every rejection and
    component transaction of both rounds (steps 3, 6, 7). Admission, freeze,
    completion and release are outside the six.
    """

    labels: NonNegativeSeconds | None
    extraction: NonNegativeSeconds | None
    embedding: NonNegativeSeconds | None
    judge_round_1: NonNegativeSeconds | None
    judge_round_2: NonNegativeSeconds | None
    commit: NonNegativeSeconds | None


class ReviewedPositionsV1(StrictModel):
    """The positions the owner reviewed in each round, so reviews per position can be
    counted across owners (E.4 ``reentry_reviews_per_position``). Sorted and unique."""

    round_1: PositionKeys | None
    round_2: PositionKeys | None

    @model_validator(mode="after")
    def canonical(self) -> Self:
        """Keep each round's keys sorted and unique, so equal reviews have equal bytes."""
        for keys in (self.round_1, self.round_2):
            if keys is not None and list(keys) != sorted(set(keys)):
                raise ValueError("Reviewed positions are sorted and unique")
        return self

    @classmethod
    def from_rounds(
        cls, round_1: Iterable[PositionKey] | None, round_2: Iterable[PositionKey] | None,
    ) -> ReviewedPositionsV1:
        """Canonicalize the reviewed keys of each round; None stays unavailable."""
        return cls(
            round_1=None if round_1 is None else tuple(sorted(set(round_1))),
            round_2=None if round_2 is None else tuple(sorted(set(round_2))),
        )


class OwnerCensusV1(StrictModel):
    """One owner's content-free census, stored in its completion record (I14; ingest-Q14).

    S9a derives the report-only E.4 rows from it: the pending block (CX6-1),
    round-2 members by cause, owner time per phase and reviews per position.
    ``round1_members`` counts every member of round 1, quiet, borrowed, reused,
    unreviewed and over-capacity ones included: R8-4's denominator for the
    ``fence_3`` share (store-Q3), which ``reviewed.round_1`` is not.
    ``judge_refused_members`` counts the members of both rounds that settled
    ``review_unavailable`` because their solo resend was still not accepted (C.8 #4).
    No field adds an E.8 threshold.
    """

    schema_version: Literal[1] = 1
    reentry: ReentryCensusV1
    pending_selectable: NonNegativeInt | None
    round1_members: NonNegativeInt | None
    round2: Round2CensusV1
    owner_phase_seconds: OwnerPhaseSecondsV1
    reviewed: ReviewedPositionsV1
    judge_refused_members: NonNegativeInt | None


ResultKind = Literal["verdict", "unavailable"]
"""The raw kind of a position's latest judge result (``JudgeResultIndexV8.kind``)."""


class CandidateStatusRowV1(StrictModel):
    """One candidate of the formal export's content-free status census (E.4; contract-Q15).

    ``latest_result_kind`` is the raw kind of the position's latest judge result:
    ``unavailable`` is never relabelled a verdict, and None means no result.
    """

    position_key: PositionKey
    status: CandidateStatus
    reason_code: ReasonCode
    rejudge: Rejudge
    reentry_count: NonNegativeInt
    latest_result_kind: ResultKind | None

    @model_validator(mode="after")
    def follows_the_outcome_table(self) -> Self:
        """A row holds a status its code allows and the code's own rejudge value (C.10)."""
        rule = OUTCOME_TABLE[self.reason_code]
        if self.status not in rule.statuses or self.rejudge != rule.rejudge:
            raise ValueError("Candidate status row contradicts the outcome table")
        return self


# The completion record (A.3.12, E.2.1; ingest-Q12): per request the frozen outcome and
# its items, and the owner census. It is written once, before release, and never
# rewritten. Its ids are historical locators: they never retain their targets, a
# forget or erasure leaves them in place (add-items counts survive), and readers drop
# forgotten, absent and unreadable objects at read time (INV:P05).

ItemLocators = Annotated[list[ULID], RefSpec(
    "item", cardinality="many", ref_mode="historical_locator", fulfillment="none",
    scope_rule="same_scope", gc_root=False,
)]
"""Frozen Item ids, sorted and unique."""

_Offset = Annotated[int, Field(ge=0, le=MAX_MESSAGE_BYTES)]


class ItemSpanV1(StrictModel):
    """One admitted quote span of an item's position, as a content-free locator (cross-Q3).

    The coordinates are ``QuoteSpanV1``'s (A0.6). Why (S8b) reads these for an
    item target and filters them by P-SPAN at read time; add-items never returns
    them.
    """

    evidence_id: Annotated[ULID, RefSpec(
        "evidence", ref_mode="historical_locator", fulfillment="none",
        scope_rule="same_scope", gc_root=False,
    )]
    byte_start: _Offset
    byte_end: _Offset

    @model_validator(mode="after")
    def check_range(self) -> Self:
        """Require a nonempty half-open range."""
        if self.byte_start >= self.byte_end:
            raise ValueError("An item span is a nonempty half-open byte range")
        return self


def _span_key(span: ItemSpanV1) -> tuple[str, int, int]:
    return span.evidence_id, span.byte_start, span.byte_end


class CompletedItemV1(StrictModel):
    """One attributed position as frozen at its request's first completion (E.2.1).

    ``memory_ids`` and ``target_memory_ids`` are sorted and unique; ``entity_ids``
    are the position's anchors, primary first; ``quote_spans`` are sorted and
    unique. The category and rejudge value are the reason code's own (C.10).
    ``judge_result_ref`` is the position's ``result_index:judge`` when the item
    froze (its verdict, or its exhausted-timeout result, C.8 #6), None when no
    judge call settled it; Why explains the item from it, never from the
    candidate row's current one, which a later re-entry replaces (E.2.3).
    """

    position_key: PositionKey
    category: OutcomeCategory
    reason_code: ReasonCode
    rejudge: Rejudge
    memory_ids: ItemLocators
    target_memory_ids: ItemLocators
    # Includes the virtual user, which has no Entity row.
    entity_ids: Annotated[list[EntityRef], Field(max_length=8), OpaqueIdSpec("local")]
    quote_spans: list[ItemSpanV1]
    judge_result_ref: Annotated[ArtifactId, RefSpec(
        "artifact", ref_mode="historical_locator", fulfillment="none", scope_rule="same_scope",
        artifact_kinds=("result_index",),
    )] | None = None

    @model_validator(mode="after")
    def canonical(self) -> Self:
        """Follow the outcome table and keep every list in its canonical order."""
        rule = OUTCOME_TABLE[self.reason_code]
        if (self.category, self.rejudge) != (rule.category, rule.rejudge):
            raise ValueError("A completed item carries its reason code's category and rejudge")
        for ids in (self.memory_ids, self.target_memory_ids):
            if ids != sorted(set(ids)):
                raise ValueError("Completed item memory ids are sorted and unique")
        if len(set(self.entity_ids)) != len(self.entity_ids):
            raise ValueError("Completed item anchors are unique")
        spans = [_span_key(span) for span in self.quote_spans]
        if spans != sorted(set(spans)):
            raise ValueError("Completed item spans are sorted and unique")
        return self

    @property
    def item_ref(self) -> str:
        """The add-items reference ``{owner_run_id}.{unit}.{position}`` (E.2.1)."""
        owner, unit, position = self.position_key
        return f"{owner}.{unit}.{position}"


class CompletedRequestV1(StrictModel):
    """One completed request's frozen outcome and the items whose categories sum to it."""

    request_key: ContentDigest
    outcome: AddOutcomeV1
    items: list[CompletedItemV1]

    @model_validator(mode="after")
    def items_sum_to_outcome(self) -> Self:
        """Items are sorted by position, each once, and their categories are the counts."""
        keys = [item.position_key for item in self.items]
        if keys != sorted(set(keys)):
            raise ValueError("Completed items are sorted by position, each once")
        counted = Counter(item.category for item in self.items)
        if self.outcome.model_dump() != {category: counted[category] for category in (
            OUTCOME_CATEGORIES
        )}:
            raise ValueError("Completed item categories must sum to the request's outcome")
        return self


class UnitRefusalV1(StrictModel):
    """One unit of the owner that a provider refusal ended (W3-RT-20).

    Its extraction call's sends were spent with no accepted reply, so the unit
    yields no fact; its messages stay as Evidence. Content-free.
    """

    unit_no: Annotated[int, Field(ge=0)]
    request_key: ContentDigest
    failure_code: Literal["provider_error"] = "provider_error"


class AddCompletionRecordV1(StrictModel):
    """One owner's completion record, referenced by its ``add_completion`` control.

    ``completed_requests`` lists, sorted by request key, every request the owner
    completed, each with its frozen outcome and items; a position counts toward
    exactly one request. ``recovery_count`` is the owner's counted-resume total.
    ``refused_units`` lists, sorted by unit, the units a provider refusal ended,
    each with the completed request its messages belong to.
    """

    completed_requests: list[CompletedRequestV1]
    owner_run_id: Annotated[ULID, RefSpec("run", ref_mode="historical_locator", fulfillment="none")]
    recovery_count: Annotated[int, Field(ge=0, le=MAX_OWNER_RESUMES)]
    census: OwnerCensusV1
    refused_units: list[UnitRefusalV1] = Field(
        default_factory=list[UnitRefusalV1], exclude_if=lambda value: not value,
    )

    @model_validator(mode="after")
    def one_request_per_position(self) -> Self:
        """Requests are sorted and unique, and no position counts toward two of them."""
        keys = [request.request_key for request in self.completed_requests]
        if keys != sorted(set(keys)):
            raise ValueError("Completed requests are sorted by request key, each once")
        positions = [
            item.position_key for request in self.completed_requests for item in request.items
        ]
        if len(set(positions)) != len(positions):
            raise ValueError("A position counts toward exactly one completed request")
        units = [refusal.unit_no for refusal in self.refused_units]
        if units != sorted(set(units)) or any(
            refusal.request_key not in keys for refusal in self.refused_units
        ):
            raise ValueError("Refused units are sorted, each once, in a completed request")
        return self


@dataclass(frozen=True, slots=True)
class PositionOutcome:
    """One position's final disposition from its owner's last round (C.10).

    Attributes:
        position_key: The position.
        reason_code: Its only final disposition.
        status: Its candidate row's status, or None when admission refused it
            without a row (B.7.3).
        revision_id: The revision a formal code wrote, otherwise None.
        request_key: The newest request among the position's own spans (INV:D20).
        memory_ids: The written memory, or the one already stating the fact (E.2.1).
        target_memory_ids: Ended or contradicted targets, or an unresolved
            position's judged targets.
        entity_ids: Its anchors, primary first.
        quote_spans: Its admitted quote spans as ``(evidence_id, byte_start, byte_end)``.
        judge_result_ref: Its candidate row's ``judge_result_ref`` after this
            round, or None (no row, or no judge call settled it).
    """

    position_key: PositionKey
    reason_code: ReasonCode
    status: CandidateStatus | None
    revision_id: str | None
    request_key: ContentBindingDigest
    memory_ids: tuple[str, ...] = ()
    target_memory_ids: tuple[str, ...] = ()
    entity_ids: tuple[str, ...] = ()
    quote_spans: tuple[tuple[str, int, int], ...] = ()
    judge_result_ref: ArtifactId | None = None


def attribute_outcomes(
    positions: Iterable[PositionOutcome], *, pending: Collection[ContentBindingDigest],
    finished: Collection[ContentBindingDigest], cause: ContentBindingDigest | None,
) -> list[CompletedRequestV1]:
    """Count each position once toward the request it belongs to (C.10; INV:D19, D20).

    A position counts toward the newest request among its own spans when this
    owner completes that request. A position whose own spans all belong to
    requests an earlier owner already completed, such as a re-entered pending
    position or a joint Add's re-extraction of the previous Add, counts toward
    ``cause``, the newest request this owner completes, but only with a formal
    effect; its other dispositions stay with the frozen receipt (receipts freeze
    at first completion). The C.10 table's ``tally`` checks and counts each
    request's positions.

    Args:
        positions: Every position the owner settled, admission refusals included,
            and every re-entered position it settled again.
        pending: The requests this owner completes.
        finished: The requests of its prefix that an earlier owner completed.
        cause: The newest of ``pending`` in reception order; None exactly when
            ``pending`` is empty.

    Returns:
        One record per pending request, zero output included, sorted by request key.

    Raises:
        AMLError: ``binding_mismatch`` when ``cause`` is not a pending request, a
            position's request lies outside the prefix, a position repeats, or a
            disposition contradicts the C.10 table.
    """
    if (cause is None) != (not pending) or cause is not None and cause not in pending:
        raise AMLError("binding_mismatch", "Add result cause is not a completed request")
    if set(pending) & set(finished):
        raise AMLError("binding_mismatch", "A request cannot be both pending and completed")
    attributed: dict[ContentBindingDigest, list[tuple[OutcomeItem, CompletedItemV1]]] = {
        key: [] for key in pending
    }
    seen: set[PositionKey] = set()
    for position in positions:
        if position.position_key in seen:
            raise AMLError("binding_mismatch", "A position has two final dispositions")
        seen.add(position.position_key)
        rule = outcome_rule(position.reason_code)
        request = position.request_key
        if request in finished:
            if not rule.formal or cause is None:
                continue
            request = cause
        elif request not in attributed:
            raise AMLError("binding_mismatch", "A position's source lies outside the owner prefix")
        attributed[request].append((
            OutcomeItem(
                position.position_key, position.reason_code, position.status,
                position.revision_id,
            ),
            CompletedItemV1(
                position_key=position.position_key, category=rule.category,
                reason_code=position.reason_code, rejudge=rule.rejudge,
                memory_ids=sorted(set(position.memory_ids)),
                target_memory_ids=sorted(set(position.target_memory_ids)),
                entity_ids=list(dict.fromkeys(position.entity_ids)),
                quote_spans=[
                    ItemSpanV1(evidence_id=evidence, byte_start=start, byte_end=end)
                    for evidence, start, end in sorted(set(position.quote_spans))
                ],
                judge_result_ref=position.judge_result_ref,
            ),
        ))
    completed: list[CompletedRequestV1] = []
    for key in sorted(attributed):
        counts = tally(outcome for outcome, _ in attributed[key])
        completed.append(CompletedRequestV1(
            request_key=key, outcome=AddOutcomeV1.model_validate(counts),
            items=sorted((item for _, item in attributed[key]), key=_item_order),
        ))
    return completed


def _item_order(item: CompletedItemV1) -> PositionKey:
    return item.position_key
