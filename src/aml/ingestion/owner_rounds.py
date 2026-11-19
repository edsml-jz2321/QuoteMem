"""The owner's judge rounds and their commits (C.1-C.12; I9, I10): RoundStage and RoundCommit.

``OwnerRounds`` is both the ``RoundStage`` and the ``RoundCommit`` the factory
binds, over S4a's one-round module (``ingestion/review_round.py``), S4b's
``JudgeStage`` with ``WriterJudgeRecords`` and S5b's ``commit_round``:

- a cohort's fresh review is grouped by original source request; each request's
  two rounds finish before the next request reads the published heads. Re-entry
  is selected only with the first group, under its existing owner-wide cap;
- round 1 selects the pending positions of earlier owners it re-enters (S4a's
  ``ReentryStage``, at most ``reentry_limit`` in rotation order, C.3), rebuilds
  them with every reviewed position of its own from their stored
  ``ProposalV2`` at the owner's input pin (``load_members``), prepares the round
  (``prepare_round``) and reviews it. A re-entered member's cited messages lie
  before the owner's prefix and previous batch, so their turn labels are
  rebuilt from their durable decisions (``source_label_view``); its judge input
  and wake key read that one attribution (judge-Q3, W3-RT-3);
- each commit turns the verdicts into ``commit_round``'s inputs
  (``round_commit_inputs``), round 1 with ``reentry_woken=()``. Every selected
  row is touched in the round's rejection transaction, which commits before any
  component: also a re-entered member that a component writes or that goes to
  round 2 (C.3, C.12);
- round 2 runs at a fresh publication pin read after the round-1 commit: the
  woken comparison (``woken``) merges with I10's round-2 entries
  (``round2_entries``), the round-2 members are rebuilt (``round2_members``),
  nominated with their picks and the targets resolution asked round 2 to issue
  (C.9.1 G3, C.9.3; L-Q3), reviewed and committed at that pin (W3-RT-6). A
  re-entered member carries its round-1 touch, rewritten after its round-2
  review (``round2_touch``);
- a component the fence finds changed in round 2 goes on to a settle round
  (``core/review_rounds``): its members are rebuilt at a fresh pin after every
  earlier commit, nominated with the targets and picks round 2 gave them,
  reviewed under the settle round's own judge children and committed there, as
  round 2 does. Settle rounds go on while the fence sends members on, up to
  ``MAX_ROUND``, whose changed components alone end ``target_changed``
  (C.11.2 #2). Each round's first component meets no delta of its own round,
  so in a scope no other writer changes, every round commits;
- a member's B.8 #3 duplicates (``ExtractionOutcome.duplicates``) are written in
  the transaction that writes the member's final disposition, mirroring it, in
  its rejection entry or its component (B.8 #3, C.12); a member that goes to
  round 2 takes them along.

An owner a privacy erase moved to the scope's current epochs
(``store/finalize_privacy``) applies C.11.1 #1 to its own members it reviewed
before the erase (``privacy_resume``): round 1 reviews only members no earlier
review covers, a member reviewed in round 1 goes on to round 2 (cause
``fence_1``), and every stale later review sends it to the next settle round.
Only the last round settles ``target_changed`` with its duplicates (C.11.2 #2);
a member an earlier invocation committed stays settled (C.13).

A resumed owner (one that did not run C.2 from its start) runs its rounds again
over what its stopped invocations left: what they committed stays settled, and
an earlier review of a member is used only for the input it was made with,
since one judge child exists per owner, position and round (C.13: rebuild,
verify the digest, resolve and commit behind the fence). A member whose rebuilt
input differs from its earlier review of the round, because the owner's own
commits or a cold adoption moved its pin, is not reviewed in that round again:
from round 1 it goes on to round 2 (``repick`` with the slots its recorded
verdict re-picked, else ``fence_1``), from round 2 or a settle round it goes
on to the next settle round, and in the last round it settles
``target_changed`` (C.11.2 #2). The other members are prepared again without it.

The completion reads every final disposition from the owner's durable records
(``store/add_outcome.owner_positions``), never from a round held in memory, so
a stopped and resumed owner completes with what it committed before its stop.
The rounds' census (``census``) counts what one invocation's rounds selected,
reviewed and sent to round 2; the owner keeps it only for an invocation that ran
C.2 from its start.

An optional ``commits`` observer (FENCE-1's recording) sees each round's whole
membership as its commit starts (``RoundEventV1``), each component's fence
(``ComponentFenceEventV1``) and how the commit ended; it changes nothing.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal, Protocol, cast

from aml.core.add_outcome import ReentryCensusV1, ReviewedPositionsV1, Round2CensusV1
from aml.core.add_read_view import AddReadView, MessageKey, MessageRecord
from aml.core.commit_plan import DuplicatePositionV1
from aml.core.controls import ControlRowV1
from aml.core.embedding import embedding_state_token
from aml.core.entities import PositionKey, ReasonCode
from aml.core.errors import AMLError
from aml.core.extraction_context import ExtractContextV2
from aml.core.extraction_policy import ExtractionPolicyConfigV2
from aml.core.inference import InferenceBindingV1
from aml.core.proposal import ProposalV2
from aml.core.query import ScopeV1
from aml.core.readability import Readability
from aml.core.resolution import IssueV1, PickV1, Round2Cause, Round2EntryV1
from aml.core.review_rounds import MAX_ROUND, RoundNo, next_round
from aml.ingestion.descriptor_repair import prepare_repairs
from aml.ingestion.extraction_stage import ExtractionOutcome, Phase
from aml.ingestion.judge import JudgedMemberV1, JudgeStage, ReviewMemberV1
from aml.ingestion.judge_gate import TargetedJudgeStage, gated
from aml.ingestion.judge_input import (
    BATCHED_PACKING,
    FULL_PACKING,
    MemberInputObserver,
    PackingRuleV1,
)
from aml.ingestion.lean_judge import LeanLocalJudge
from aml.ingestion.nomination import DeclarationV1, NominationIndex, NominationPolicyV1
from aml.ingestion.owner_pipeline import OwnerSession, Round, RoundsCensus
from aml.ingestion.reentry import (
    PendingCandidateV1,
    ReentryAction,
    ReentrySelectionV1,
    ReentryStage,
    ReentryTouchV1,
    pending_candidate,
)
from aml.ingestion.review_round import (
    LoadedMemberV1,
    PreparedMemberV1,
    Round2PlanV1,
    RoundCommitInputsV1,
    RoundPreparationV1,
    load_members,
    prepare_round,
    rejection_entry,
    round2_entries,
    round2_members,
    round2_touch,
    round_commit_inputs,
    siblings,
    usable,
    woken,
)
from aml.ingestion.sources import LabelView, MessageLabels, TurnLabel, message_key
from aml.ingestion.statement_embedding import OwnerEmbedding
from aml.providers.adapters import ProviderError
from aml.providers.auth import PrincipalGrant
from aml.providers.bindings import judge_binding_for_policy, lean_judge_binding
from aml.providers.contracts import EmbeddingProvider, StructuredInferenceProvider
from aml.store.component_commit import ComponentFenceEventV1, RoundCommitV1, commit_round
from aml.store.domain import CandidateRowV2
from aml.store.finalize_privacy import PrivacyResumeV1, committed_positions, privacy_resume
from aml.store.judge_records import EarlierReviewV1, WriterJudgeRecords
from aml.store.memory_commit import (
    DispositionV1,
    FollowerV1,
    RejectionEntryV1,
    commit_rejection,
    journaled_positions,
    proposal_groups,
)
from aml.store.source_labels import source_label_view
from aml.store.writer import CommittedRows, Writer, WriteTicket


@dataclass(frozen=True, slots=True)
class CarriedV1:
    """A member a resumed owner takes to the next round without reviewing it (C.13).

    Its earlier review was admitted under another erase epoch (C.11.1 #1,
    ``privacy_resume``), or with an input its rebuild no longer reproduces; one
    judge child exists per owner, position and round, so the position is
    reviewed again only in a later round (C.11.2 #2, C.17 C12).

    Attributes:
        proposal_ref: Its ``proposal_snapshot``.
        cause: ``repick`` when its recorded round-1 verdict re-picked a slot,
            else ``fence_1``.
        picks: Those re-picks, which round 2 nominates (C.9.1 G3).
        pending: Its stored row, for a member re-entered from an earlier owner.
        candidate_id: That row's id.
        touched: Whether the owner's round-1 touch of that row is durable.
        touch: The touch the previous round gave a re-entered member a settle round
            carries on; it is rewritten after the member's next review (C.3).
    """

    proposal_ref: str
    cause: Round2Cause = "fence_1"
    picks: tuple[PickV1, ...] = ()
    pending: PendingCandidateV1 | None = None
    candidate_id: str | None = None
    touched: bool = False
    touch: ReentryTouchV1 | None = None


@dataclass(frozen=True, slots=True)
class ChangedV1:
    """A member a resumed owner settles ``target_changed`` without a review (C.11.2 #2).

    Attributes:
        proposal_ref: Its ``proposal_snapshot``.
        result_ref: The ``result_index:judge`` of its earlier last-round review, if any.
        candidate_id: Its stored row, for a member re-entered from an earlier owner.
    """

    proposal_ref: str
    result_ref: str | None
    candidate_id: str | None = None


@dataclass
class OwnerRound:
    """One round's result (``R`` of the owner pipeline): prepared, judged, then committed.

    Attributes:
        round: The round.
        prepared: Its members after nomination; None when it has none.
        judged: Each reviewed member's judge outcome.
        inputs: ``commit_round``'s inputs, built after the verdicts.
        committed: I10's result.
        labels: The turn labels its members' cited turns are attributed over.
        candidates: The stored candidate row of each re-entered member (C.3).
        followers: The B.8 #3 duplicates of each member, which its final
            disposition's transaction writes.
        resumed: C.11.1 #1 routing of an owner a privacy erase moved, carried
            through every round (``privacy_resume``); None for other owners.
        pending_selectable: Round 1: the re-entry selection's pre-cap population.
        entries: Round 2 or later: its members' entries, one cause each (store-Q3).
        requests: Each physical judge request it sent, as its members' positions.
        carried: The members it takes to the next round unreviewed (C.13).
        changed: The members it settles ``target_changed`` unreviewed (C.11.2 #2).
        context: Earlier source requests' pending positions actually issued to
            this round, present only as non-writing resolution siblings,
            outside this round's reviews, writes and census.
        context_inventory: All usable earlier pending members, rebuilt in the
            current canonical sibling inventory; only an actually issued one
            needs its own non-writing resolution input.
        issued: Round 2 and later: the targets round 1's resolution asked each
            member to be issued (C.9.3), which a settle round issues again.
        picks: Round 2 and later: each member's re-picked slots (C.9.1 G3).
        declared: Round 2 and later: the new-entity declarations by ``mint_key``
            of the rounds before it (`Round2PlanV1.declarations`).
    """

    round: RoundNo
    prepared: RoundPreparationV1 | None = None
    judged: dict[PositionKey, JudgedMemberV1] = field(
        default_factory=dict[PositionKey, JudgedMemberV1])
    inputs: RoundCommitInputsV1 | None = None
    committed: RoundCommitV1 | None = None
    labels: LabelView = field(default_factory=MessageLabels)
    candidates: dict[PositionKey, str] = field(default_factory=dict[PositionKey, str])
    followers: dict[PositionKey, tuple[FollowerV1, ...]] = field(
        default_factory=dict[PositionKey, tuple[FollowerV1, ...]])
    resumed: PrivacyResumeV1 | None = None
    pending_selectable: int | None = None
    entries: tuple[Round2EntryV1, ...] = ()
    requests: tuple[tuple[PositionKey, ...], ...] = ()
    carried: dict[PositionKey, CarriedV1] = field(default_factory=dict[PositionKey, CarriedV1])
    changed: dict[PositionKey, ChangedV1] = field(default_factory=dict[PositionKey, ChangedV1])
    context: tuple[PreparedMemberV1, ...] = ()
    context_inventory: dict[PositionKey, tuple[LoadedMemberV1, PendingCandidateV1]] = field(
        default_factory=dict[PositionKey, tuple[LoadedMemberV1, PendingCandidateV1]])
    issued: dict[PositionKey, tuple[IssueV1, ...]] = field(
        default_factory=dict[PositionKey, tuple[IssueV1, ...]])
    picks: dict[PositionKey, tuple[PickV1, ...]] = field(
        default_factory=dict[PositionKey, tuple[PickV1, ...]])
    declared: dict[str, DeclarationV1] = field(default_factory=dict[str, DeclarationV1])

    @property
    def reviewed(self) -> list[PositionKey]:
        """The positions it sent in a physical judge request."""
        return [key for request in self.requests for key in request]


class RoundSession(OwnerSession, Protocol):
    """What the rounds need of the owner session (``owner_session._WriterSession``).

    It is also S4b's ``JudgeOwner`` (``writer``, ``run_id``, ``attempt``,
    ``grant``, ``check``, ``clock``).
    """

    @property
    def writer(self) -> Writer: ...

    @property
    def embedder(self) -> EmbeddingProvider: ...

    def mark_stage(self, stage: Phase) -> None: ...

    @property
    def run_id(self) -> str: ...

    @property
    def view(self) -> AddReadView: ...

    @property
    def scope(self) -> ScopeV1: ...

    @property
    def attempt(self) -> int: ...

    @property
    def write_ticket(self) -> WriteTicket: ...

    @property
    def policy(self) -> ExtractionPolicyConfigV2: ...

    @property
    def policy_release_ref(self) -> str: ...

    @property
    def label_view(self) -> LabelView | None: ...

    def grant(self) -> PrincipalGrant: ...

    def check(self) -> None: ...

    def clock(self) -> int: ...

    def publish(self, committed: CommittedRows) -> None: ...

    def read_view(self, consumer: Callable[[AddReadView], None]) -> None: ...


def judge_binding(policy: ExtractionPolicyConfigV2) -> InferenceBindingV1:
    """The owner's frozen judge binding from its frozen extraction policy (C.5)."""
    return judge_binding_for_policy(policy)


def judge_packing(policy: ExtractionPolicyConfigV2) -> PackingRuleV1:
    """The owner's packing rule: the batched packing of ``targeted-v1``'s model stage, or
    the full mode's."""
    batched = policy.parameters.judge_mode == "targeted-v1"
    return BATCHED_PACKING if batched else FULL_PACKING


def judge_provider(
    policy: ExtractionPolicyConfigV2, model: StructuredInferenceProvider,
) -> StructuredInferenceProvider:
    """The provider that answers the owner's judge requests: the model's, or the
    lean mode's local judge (``judge_mode`` ``local-lean-v1``)."""
    return LeanLocalJudge() if policy.parameters.judge_mode == "local-lean-v1" else model


def targeted_gate(policy: ExtractionPolicyConfigV2) -> Callable[[ReviewMemberV1], bool] | None:
    """The targeted mode's member gate (``judge_gate.gated``), once per member input; None
    in every other mode."""
    gate = policy.parameters.judge_gate
    if gate is None:
        return None
    found: dict[str, bool] = {}

    def selects(member: ReviewMemberV1) -> bool:
        digest = member.input.digest
        if digest not in found:
            found[digest] = gated(member, gate)
        return found[digest]

    return selects


def _weights(policy: ExtractionPolicyConfigV2) -> dict[str, float]:
    raw: Mapping[str, Any] = policy.parameters.importance_weights.model_dump()
    return {name: float(value) for name, value in raw.items()}


def _followers(
    duplicates: Iterable[DuplicatePositionV1],
) -> dict[PositionKey, tuple[FollowerV1, ...]]:
    """Each representative's B.8 #3 duplicates, in canonical order, as I10 writes them."""
    found: dict[PositionKey, list[FollowerV1]] = {}
    for duplicate in duplicates:
        found.setdefault(duplicate.representative, []).append(FollowerV1.of(duplicate))
    return {key: tuple(items) for key, items in found.items()}


@dataclass(frozen=True, slots=True)
class _ReentryLabels:
    """The owner's frozen turn labels, and those of the re-entered members' cited messages."""

    owner: LabelView
    earlier: LabelView
    keys: frozenset[MessageKey]

    def turns(self, message: MessageKey) -> Sequence[TurnLabel]:
        return (self.earlier if message in self.keys else self.owner).turns(message)


def _proposal(reader: Writer, reference: str) -> ProposalV2:
    """A stored ``ProposalV2`` by its ``proposal_snapshot``."""
    proposal = reader.read_logical(reference, "proposal_snapshot")
    if not isinstance(proposal, ProposalV2):
        raise AMLError("binding_mismatch", "A member's proposal is not a ProposalV2")
    return proposal


def _labels(owner: RoundSession, view: AddReadView,
            reentered: Iterable[ProposalV2]) -> LabelView:
    """The labels a round attributes its members' cited turns over (B.3a, judge-Q3).

    The owner's own view labels its frozen prefix and previous batch. The cited
    messages of a re-entered member are older; their labels are rebuilt from
    their durable decisions at the round's pin (``source_label_view``, no
    provider call), so every owner attributes them alike (W3-RT-3). A
    re-entered member a rebased owner carries to round 2 is labelled here too.
    """
    messages: dict[MessageKey, MessageRecord] = {}
    for proposal in reentered:
        for span in proposal.quote_spans:
            part = view.evidence(span.evidence_id)
            message = None if part is None else view.message(message_key(part))
            if message is not None:
                messages.setdefault(message.key, message)
    labels = owner.label_view or MessageLabels()
    if not messages:
        return labels
    read = Readability(view.scope_kind, view.scope_id, view.now, view.evidence)
    return _ReentryLabels(labels, source_label_view(owner.writer, read, tuple(messages.values())),
                          frozenset(messages))


def _touched(prepared: RoundPreparationV1, inputs: RoundCommitInputsV1,
             candidates: Mapping[PositionKey, str],
             followers: Mapping[PositionKey, tuple[FollowerV1, ...]]) -> list[RejectionEntryV1]:
    """The round's rejection entries with every selected member's touch (C.3, C.12).

    A re-entered member that a component writes, or that goes to round 2, is
    touched here too: the rejection transaction commits before any component,
    so a member whose component the fence sends to round 2 is touched as well.
    A member whose disposition the entry writes carries its duplicates (B.8 #3).
    """
    entries = [replace(entry, followers=followers.get(entry.position.position_key, ()))
               if entry.disposition is not None else entry for entry in inputs.rejection]
    written = {entry.position.position_key for entry in entries}
    entries.extend(rejection_entry(member, candidate_id=candidates[member.position_key])
                   for member in prepared.members
                   if member.touch is not None and member.position_key not in written)
    return sorted(entries, key=lambda entry: entry.order)


def _carried(carried: Mapping[PositionKey, CarriedV1]) -> list[Round2EntryV1]:
    """The round-2 entries of the members a resumed owner carries past round 1 (C.11.2 #2)."""
    return [Round2EntryV1(position_key=key, cause=item.cause) for key, item in carried.items()]


def _round2_members(
    round1: RoundPreparationV1, entries: Sequence[Round2EntryV1],
    carried: Mapping[PositionKey, CarriedV1], view: AddReadView, *, reader: Writer,
    round: RoundNo = 2,
) -> Round2PlanV1:
    """`round2_members`, rebuilding with them the carried members the previous round did
    not rebuild.

    Every carried member is an entry. All members load together, so each is
    every later one's sibling (C.1). A carried re-entered member takes its
    stored row; the owner's durable round-1 touch of it, or the touch the
    previous round gave it, is rewritten after its next review like any other (C.3).
    """
    regular = [entry for entry in entries if entry.position_key not in carried]
    if len(regular) + len(carried) != len(entries):
        raise AMLError("binding_mismatch", "A carried member enters the next round")
    plan = round2_members(round1, regular, view, reader=reader, round=round)
    if not carried:
        return plan
    members = load_members([*(member.proposal_ref for member in plan.members),
                            *(item.proposal_ref for item in carried.values())],
                           view, reader=reader, round=round)
    pending = dict(plan.pending)
    touches = dict(plan.round1_touch)
    for key, item in carried.items():
        if item.pending is None:
            continue
        pending[key] = item.pending
        if item.touch is not None:
            touches[key] = item.touch
        elif item.touched and item.pending.last_wake_digest is not None:
            touches[key] = ReentryTouchV1(key, item.pending.reentry_count,
                                          item.pending.last_wake_digest, None)
    return replace(plan, members=members, pending=pending, round1_touch=touches)


def _round2_touch(plan: Round2PlanV1, member: PreparedMemberV1,
                  carried: Mapping[PositionKey, CarriedV1]) -> ReentryTouchV1 | None:
    """A round-2 re-entered member's touch: its round-1 touch rewritten after a review
    (`round2_touch`); a carried row the owner never touched takes its own first touch."""
    if member.position_key in carried and member.position_key not in plan.round1_touch:
        return member.touch
    return round2_touch(plan, member)


@dataclass(frozen=True, slots=True)
class RoundMemberV1:
    """One member of an owner round as its commit starts (FENCE-1; R8-4's denominator).

    Attributes:
        position_key: The member.
        reviewed: It was sent in one of the round's physical judge requests.
        action: A re-entered member's action (C.3); None for the owner's own.
        expire: A re-entered member whose row expires at this rebuild.
    """

    position_key: PositionKey
    reviewed: bool
    action: ReentryAction | None
    expire: bool


@dataclass(frozen=True, slots=True)
class RoundEventV1:
    """One owner round's commit as it starts and ends (FENCE-1's recording).

    ``started`` lists every member of the round, quiet, borrowed, reused,
    unreviewed and over-capacity ones included, and in round 2 the entries the
    round reviews (C.2 step 7); ``ended`` lists the entries each round below
    ``MAX_ROUND`` sends on, one cause each (store-Q3); ``interrupted`` marks a commit that
    raised. A started round that never ended was stopped with its process.

    Attributes:
        owner_run_id: The Add owner.
        cycle: The invocation's lease generation.
        attempt: The owner Run's attempt.
        round: The round.
        stage: ``started``, ``ended`` or ``interrupted``.
        members: ``started``: the round's members in canonical order.
        entries: ``started`` in a later round: its entries; ``ended`` below
            ``MAX_ROUND``: the entries it sends on.
    """

    owner_run_id: str
    cycle: int
    attempt: int
    round: RoundNo
    stage: Literal["started", "ended", "interrupted"]
    members: tuple[RoundMemberV1, ...] = ()
    entries: tuple[Round2EntryV1, ...] = ()


type CommitObserver = Callable[[RoundEventV1 | ComponentFenceEventV1], None]
"""Sees each round's ``RoundEventV1`` and each component's ``ComponentFenceEventV1``."""


@dataclass
class OwnerRounds:
    """The owner's ``RoundStage`` and ``RoundCommit`` over the real Writer (C.3-C.12).

    ``observe`` sees each member input a round reviews, with its pin, before the judge
    stage sends it (`MemberInputObserver`; the recorded-replay harness, E.5).
    ``commits`` sees each round's membership and each component's fence as the
    commit runs (`CommitObserver`; FENCE-1's recording, E.9 #10).
    """

    provider: StructuredInferenceProvider
    policy: NominationPolicyV1 = field(default_factory=NominationPolicyV1)
    observe: MemberInputObserver | None = None
    commits: CommitObserver | None = None

    # RoundStage -----------------------------------------------------------------------------

    def _load(
        self, owner: RoundSession, result: OwnerRound, refs: Mapping[PositionKey, str],
        outcome: ExtractionOutcome, view: AddReadView,
    ) -> tuple[LoadedMemberV1, ...]:
        """Keep earlier pending requests visible as siblings without reviewing them again."""
        first_unit = min((key[1] for key in outcome.review), default=0)
        prior = {
            key: row for key, row in ReentryStage(
                owner.writer.indexed_reader(), owner.scope, check=owner.check,
            ).rows().items()
            if key[0] == owner.run_id and key[1] < first_unit
            and key in outcome.proposals and key not in refs
            and owner.writer.get_row("artifact", row.proposal_ref) is not None
            and _proposal(owner.writer, row.proposal_ref).statement_embedding is not None
        } if first_unit else {}
        loaded = load_members(
            [*refs.values(), *(row.proposal_ref for row in prior.values())],
            view, reader=owner.writer, round=result.round,
        )
        result.context_inventory = {
            item.position_key: (item, pending_candidate(prior[item.position_key]))
            for item in loaded if item.position_key in prior and usable(item.member, view)
        }
        result.context = ()
        return tuple(item for item in loaded if item.position_key in refs)

    def batches(
        self, session: OwnerSession, outcome: ExtractionOutcome,
    ) -> tuple[ExtractionOutcome, ...]:
        """Keep each source request's two rounds ahead of the next request's pin.

        Context unit coordinates are immutable metadata; joined duplicate quote
        spans cannot identify the representative's original source request. A
        privacy erase may have removed a needed context. That recovery retains
        its existing single review batch instead of inventing a boundary or
        opening an erased body.
        """
        if not outcome.review:
            return (outcome,)
        owner = cast(RoundSession, session)
        contexts = {
            body.unit.unit_no: body.unit
            for _, body in owner.writer.find_logical_candidates(owner.run_id, "decision_set")
            if isinstance(body, ExtractContextV2)
        }
        units = {key[1] for key in outcome.review}
        if not units <= contexts.keys():
            return (outcome,)
        groups: dict[tuple[str, str], set[PositionKey]] = {}
        for key in sorted(outcome.review):
            unit = contexts[key[1]]
            if key[0] != owner.run_id or unit.owner_run_id != owner.run_id:
                raise AMLError("binding_mismatch", "Fresh review crossed its extraction owner")
            groups.setdefault((unit.session_id, unit.request_id), set()).add(key)
        if len(groups) == 1:
            return (outcome,)
        return tuple(replace(
            outcome, review=tuple(key for key in outcome.review if key in keys),
            duplicates=tuple(item for item in outcome.duplicates if item.representative in keys),
        ) for keys in groups.values())

    def review(
        self, session: OwnerSession, outcome: ExtractionOutcome, round: Round,
        previous: OwnerRound | None, *, reentry: bool = True,
    ) -> OwnerRound:
        """Select, prepare and judge one round's members (C.1, C.3-C.8)."""
        owner = cast(RoundSession, session)
        if round == 1:
            result = OwnerRound(1, followers=_followers(outcome.duplicates))
            writer, view = owner.writer, owner.view
            stage = ReentryStage(writer.indexed_reader(), owner.scope, check=owner.check)
            pending = stage.rows() if reentry else {}
            if not owner.whole_run:
                result.resumed = privacy_resume(writer, owner.run_id, outcome.review, pending)
            routed: Mapping[PositionKey, CandidateRowV2] = (
                {} if result.resumed is None else result.resumed.rows)
            selection = stage.select(owner.run_id, rows=pending, exclude=routed.keys()) if reentry \
                else ReentrySelectionV1((), 0)
            result.pending_selectable = selection.pending_selectable
            rows = {(row.owner_run_id, row.unit_no, row.position): row
                    for row in selection.candidates}
            if not outcome.review and not rows and not routed:
                return result
            touched: set[PositionKey] = set() if owner.whole_run or not (rows or routed) else (
                journaled_positions(writer, owner.run_id, owner_run_id=owner.run_id, round_no=1,
                                    txn="rejection")
            )
            members = outcome.review
            if result.resumed is None and not owner.whole_run:
                # What a stopped invocation committed stays settled (C.13): its
                # earlier review is never checked against a rebuild that the
                # owner's own commit moved, so the member is neither carried to
                # round 2 nor settled ``target_changed`` over its stored row.
                settled = committed_positions(writer, owner.run_id)
                members = tuple(key for key in members if key not in settled)
            if result.resumed is not None:
                members = result.resumed.round1
                for key in result.resumed.round2:
                    row = routed.get(key)
                    result.carried[key] = CarriedV1(outcome.proposals[key]) if row is None else (
                        CarriedV1(row.proposal_ref, pending=pending_candidate(row),
                                  candidate_id=row.object_id, touched=key in touched))
            refs = {**{key: outcome.proposals[key] for key in members},
                    **{key: row.proposal_ref for key, row in rows.items()}}
            loaded = self._load(owner, result, refs, outcome, view)
            result.candidates = {key: row.object_id for key, row in rows.items()}
            result.labels = _labels(owner, view, [
                *(member.proposal for member in loaded if member.position_key in rows),
                *(_proposal(writer, row.proposal_ref) for row in routed.values()),
            ])

            def carry(key: PositionKey, earlier: EarlierReviewV1) -> None:
                row = rows.get(key)
                result.carried[key] = CarriedV1(
                    refs[key], "repick" if earlier.picks else "fence_1", earlier.picks,
                    None if row is None else pending_candidate(row),
                    None if row is None else row.object_id, key in touched,
                )

            self._prepare_and_judge(
                owner, result, loaded, view,
                {key: pending_candidate(row) for key, row in rows.items()},
                reload=lambda excluded: self._load(
                    owner, result, {key: ref for key, ref in refs.items() if key not in excluded},
                    outcome, view,
                ),
                stale=carry,
            )
            return result
        if previous is None or previous.prepared is None or previous.committed is None:
            raise AMLError("binding_mismatch", "A later round requires the previous round's commit")
        result = OwnerRound(round, labels=previous.labels, candidates=dict(previous.candidates),
                            followers=previous.followers, resumed=previous.resumed)
        before, committed = previous.prepared, previous.committed
        # Round 1's resolver requests name what round 2 issues and re-picks; a settle
        # round rebuilds a member with what the previous round gave it (C.9.1 G3, C.9.3).
        requests = () if previous.inputs is None else previous.inputs.plan.round2
        carried = previous.carried
        result.issued = {**previous.issued,
                         **{request.position_key: request.issue for request in requests}}
        result.picks = {**previous.picks,
                        **{request.position_key: request.picks for request in requests},
                        **{key: item.picks for key, item in carried.items() if item.picks}}
        result.candidates.update({key: item.candidate_id for key, item in carried.items()
                                  if item.candidate_id is not None})
        last = next_round(round) is None
        stopped: list[ProviderError] = []

        def at_pin(view: AddReadView) -> None:
            writer = owner.writer
            if round == 2:
                codes: dict[PositionKey, ReasonCode] = dict(committed.codes)
                found = woken(
                    before, codes, view, reader=writer, policy=self.policy,
                    bind=writer.bind_content, owner_run_id=owner.run_id, labels=result.labels,
                    owner_committed=any(item.committed is not None
                                        for item in committed.components),
                )
                entries = round2_entries(committed.round2, found)
            else:
                # A settle round takes exactly what the previous round's fence sent on.
                entries = tuple(sorted(committed.round2, key=lambda entry: entry.position_key))
            result.entries = entries
            if not entries:
                return
            plan = _round2_members(before, entries, carried, view, reader=writer, round=round)
            result.declared = {**previous.declared, **plan.declarations}
            refs = {member.position_key: member.proposal_ref for member in plan.members}
            selected = self._load(owner, result, refs, outcome, view)

            def change(key: PositionKey, earlier: EarlierReviewV1) -> None:
                result.changed[key] = ChangedV1(refs[key], earlier.result_ref,
                                                result.candidates.get(key))

            def carry(key: PositionKey, _earlier: EarlierReviewV1) -> None:
                result.carried[key] = CarriedV1(
                    refs[key], pending=plan.pending.get(key),
                    candidate_id=result.candidates.get(key), touch=plan.round1_touch.get(key),
                )

            try:
                self._prepare_and_judge(
                    owner, result, selected, view, plan.pending,
                    issued=result.issued, picks=result.picks,
                    reload=lambda excluded: self._load(
                        owner, result,
                        {key: ref for key, ref in refs.items() if key not in excluded},
                        outcome, view,
                    ),
                    stale=change if last else carry,
                    declared=result.declared,
                )
            except ProviderError as error:
                # The read guard replaces a callback's error with a content-free
                # failure; a provider failure leaves the callback whole, without
                # its frames, so the owner pauses (C.16).
                error.__cause__ = error.__context__ = None
                stopped.append(error.with_traceback(None))
                return
            prepared = result.prepared
            if prepared is None:
                return
            # Each re-entered member carries its earlier touch, rewritten when it is
            # reviewed again (C.3).
            result.prepared = replace(prepared, members=tuple(
                replace(member, touch=_round2_touch(plan, member, carried))
                if member.pending is not None and not member.expire else member
                for member in prepared.members))
            result.inputs = round_commit_inputs(
                result.prepared, result.judged.values(), view, bind=writer.bind_content,
                weights=_weights(owner.policy), candidates=result.candidates,
                context=result.context,
            )
        owner.read_view(at_pin)
        if stopped:
            raise stopped[0]
        return result

    def _prepare_and_judge(
        self, owner: RoundSession, result: OwnerRound, loaded: Sequence[LoadedMemberV1],
        view: AddReadView, pending: Mapping[PositionKey, PendingCandidateV1], *,
        issued: Mapping[PositionKey, Sequence[IssueV1]] | None = None,
        picks: Mapping[PositionKey, Sequence[PickV1]] | None = None,
        declared: Mapping[str, DeclarationV1] | None = None,
        reload: Callable[[frozenset[PositionKey]], Sequence[LoadedMemberV1]] | None = None,
        stale: Callable[[PositionKey, EarlierReviewV1], None] | None = None,
    ) -> None:
        """Prepare the round and judge its reviewed members (C.3-C.8).

        A resumed owner first verifies each reviewed member against its earlier
        review of the round (C.13): one whose rebuilt input differs is handed to
        ``stale`` and the round is prepared again without it, until every
        remaining review is new or reproduces its recorded input.
        """
        writer = owner.writer
        binding = judge_binding(owner.policy)
        records = WriterJudgeRecords(cast(Any, owner), binding=binding)
        # The targeted mode reviews a member its gate leaves out under the lean binding.
        gate = targeted_gate(owner.policy)
        lean = None if gate is None else lean_judge_binding(owner.policy)
        lean_records = None if lean is None else WriterJudgeRecords(cast(Any, owner), binding=lean)

        def records_of(member: ReviewMemberV1) -> WriterJudgeRecords:
            if gate is None or lean_records is None or gate(member):
                return records
            return lean_records

        excluded: set[PositionKey] = set()
        while True:
            prepared = prepare_round(
                loaded, view, round=result.round, policy=self.policy, bind=writer.bind_content,
                owner_run_id=owner.run_id, labels=result.labels, pending=pending,
                reusable=lambda member: records_of(member).reusable(member), issued=issued,
                picks=picks, declared=declared,
            )
            if owner.whole_run or reload is None or stale is None:
                break
            found = {
                member.position_key: earlier for member in prepared.review
                if (earlier := records.earlier(member.position_key, result.round)) is not None
                and earlier.member_input_digest != member.input.digest
            }
            if result.resumed is not None:
                for member in prepared.review:
                    key = (member.position_key, result.round)
                    if key in result.resumed.invalidated:
                        found[member.position_key] = EarlierReviewV1(
                            member.input.digest, result.resumed.invalidated[key], ())
            if not found:
                break
            for key, earlier in found.items():
                stale(key, earlier)
            excluded |= found.keys()
            loaded = reload(frozenset(excluded))
        result.prepared = prepared
        required = {key for member in prepared.members for key in siblings(member.nomination)}
        selected_context = [item for key, item in result.context_inventory.items()
                            if key in required]
        if selected_context:
            index = NominationIndex(view)
            result.context = tuple(PreparedMemberV1(
                loaded, index.nominate(loaded.member, self.policy), (),
                pending=pending, action="quiet",
            ) for loaded, pending in selected_context)
        # load_members kept every declaration in each current member, even when
        # no issued sibling needs that earlier position's resolution input.
        # A quiet member makes no call and reuses no verdict, but its snapshot is
        # persisted like a reviewed member's, before any dispatch (C.3, C.13).
        quiet = [(owner.write_ticket, "decision_set", member.snapshot, "internal")
                 for member in prepared.members
                 if member.action == "quiet" and member.snapshot is not None]
        if quiet:
            writer.commit_artifacts(owner.write_ticket, quiet, created_at=owner.clock())
        if self.observe is not None:
            for member in prepared.review:
                self.observe(member.input, view)
        if prepared.review:
            stage: JudgeStage | TargetedJudgeStage = JudgeStage(
                records, judge_provider(owner.policy, self.provider), binding=binding,
                bind=writer.bind_content, packing=judge_packing(owner.policy),
                owner_run_id=owner.run_id)
            if gate is not None and lean is not None and lean_records is not None:
                stage = TargetedJudgeStage(stage, JudgeStage(
                    lean_records, LeanLocalJudge(), binding=lean, bind=writer.bind_content,
                    owner_run_id=owner.run_id), gate)
            result.judged = {item.position_key: item
                             for item in stage.review(list(prepared.review), result.round)}
            result.requests = stage.requests

    def next_round(self, previous: OwnerRound) -> RoundNo | None:
        """The round the previous round's commit may leave members for, or None (C.13).

        After round 1, I10's round-2 entries are known; woken re-entered members are
        found at the round-2 pin, so a round with re-entered members also goes on.
        After round 2 or a settle round, the members its fence sent on (and those a
        resumed owner carries on) take the next settle round, up to ``MAX_ROUND``
        (``core/review_rounds``).
        """
        committed = previous.committed
        after = next_round(previous.round)
        if committed is None or after is None:
            return None
        if previous.round == 1:
            if previous.prepared is None:
                return None
            return after if committed.round2 or any(
                member.pending is not None for member in previous.prepared.members) else None
        return after if committed.round2 else None

    def census(self, round_1: OwnerRound, later: Sequence[OwnerRound]) -> RoundsCensus:
        """Count what one invocation's rounds selected, reviewed and sent to round 2 (I14).

        The re-entry counts are round 1's selection and its rebuilt members'
        actions (C.3); ``digest_changed`` counts those whose member input differs
        from the input of their latest review, the row's ``last_input_digest``
        (eval-Q16). ``round1_members`` counts every member of round 1, R8-4's
        denominator. A round's reviewed positions are the members of its physical
        judge requests: quiet, unreviewed, borrowed, reused and over-capacity
        members send none. Round 2 counts its entries, how many of them ended
        ``target_changed`` and its physical requests; the settle rounds after it
        (``later[1:]``) count with it, as they only settle round 2's members. A
        round 2 that did not run has none of each. ``judge_refused_members``
        counts the members of every round whose solo resend was still not
        accepted (C.8 #4).

        The stored reviewed-position sets deduplicate positions across later
        rounds. Harness exports derive exact per-round reviews from judge
        children without changing this stored census.

        Raises:
            AMLError: ``binding_mismatch`` when round 1 ran no re-entry selection.
        """
        if round_1.pending_selectable is None:
            raise AMLError("binding_mismatch", "Round 1 ran no re-entry selection")
        members = () if round_1.prepared is None else round_1.prepared.members
        actions: list[ReentryAction] = []
        changed = 0
        for member in members:
            if member.action is None or member.pending is None:
                continue
            actions.append(member.action)
            digest = None if member.input is None else member.input.digest
            changed += digest != member.pending.last_input_digest
        codes: dict[PositionKey, ReasonCode] = {}
        for judged in later:
            if judged.committed is not None:
                codes.update(judged.committed.codes)
        entries = later[0].entries if later else ()
        return RoundsCensus(
            reentry=ReentryCensusV1.from_actions(actions, digest_changed=changed),
            pending_selectable=round_1.pending_selectable,
            round1_members=len(members),
            round2=Round2CensusV1.from_entries(
                entries,
                target_changed=sum(codes.get(entry.position_key) == "target_changed"
                                   for entry in entries),
                judge_requests=sum(len(judged.requests) for judged in later),
            ),
            reviewed=ReviewedPositionsV1.from_rounds(
                round_1.reviewed, [key for judged in later for key in judged.reviewed],
            ),
            judge_refused_members=sum(
                item.judgment is not None and item.judgment.refused
                for judged in (round_1, *later)
                for item in judged.judged.values()
            ),
        )

    # RoundCommit ----------------------------------------------------------------------------

    def commit(self, session: OwnerSession, outcome: ExtractionOutcome,
               judged: OwnerRound) -> None:
        """Commit one judged round through I10; a rerun replays nothing twice (C.12).

        Each member's duplicates are written in the transaction that writes the
        member's final disposition (B.8 #3, C.12). Round 1 of an owner a privacy
        erase moved sends its previously reviewed members through round 2 and
        the settle rounds; only the last round writes ``target_changed``
        (C.11.1 #1, C.11.2 #2). ``commits`` sees the round start, each
        component's fence and the round end, or an interruption.
        """
        owner = cast(RoundSession, session)
        record = self.commits
        if record is not None:
            record(self._event(owner, judged, "started"))
        try:
            self._commit(owner, outcome, judged)
        except BaseException:
            if record is not None:
                record(self._event(owner, judged, "interrupted"))
            raise
        if record is not None:
            record(self._event(owner, judged, "ended"))

    @staticmethod
    def _event(owner: RoundSession, judged: OwnerRound,
               stage: Literal["started", "ended", "interrupted"]) -> RoundEventV1:
        """The round's ``RoundEventV1`` at one stage of its commit."""
        members: tuple[RoundMemberV1, ...] = ()
        entries: tuple[Round2EntryV1, ...] = ()
        if stage == "started":
            reviewed = set(judged.reviewed)
            prepared = () if judged.prepared is None else judged.prepared.members
            members = tuple(RoundMemberV1(member.position_key, member.position_key in reviewed,
                                          member.action, member.expire) for member in prepared)
            entries = judged.entries
        elif stage == "ended" and judged.round < MAX_ROUND and judged.committed is not None:
            entries = tuple(judged.committed.round2)
        return RoundEventV1(owner.run_id, owner.cycle, owner.attempt, judged.round, stage,
                            members, entries)

    def _commit(self, owner: RoundSession, outcome: ExtractionOutcome,
                judged: OwnerRound) -> None:
        self._settle_changed(owner, judged.changed, judged.followers, round_no=judged.round)
        carried = _carried(judged.carried)
        prepared = judged.prepared
        if prepared is None or not prepared.members:
            if carried:
                judged.committed = RoundCommitV1(round2=carried)
            return
        writer = owner.writer
        inputs = judged.inputs
        if inputs is None:
            inputs = round_commit_inputs(
                prepared, judged.judged.values(), owner.view, bind=writer.bind_content,
                weights=_weights(owner.policy), candidates=judged.candidates,
                context=judged.context,
            )
            judged.inputs = inputs
        followers = judged.followers
        control = ControlRowV1.model_validate(writer.get_control("policy:embedding_space"))
        # The rewordings' vectors are B.9 calls: a failure pauses as the embedding stage.
        owner.mark_stage("embedding")
        members = prepare_repairs(
            inputs.plan, inputs.members, owner.view,
            owner=OwnerEmbedding(owner_run_id=owner.run_id, ticket=owner.write_ticket,
                                 space=owner.view.embedding_space,
                                 token=embedding_state_token(control), clock=owner.clock,
                                 authorize=owner.check),
            provider=owner.embedder, store=writer, reader=writer,
        )
        owner.mark_stage("commit")
        judged.committed = commit_round(
            writer, owner.write_ticket, owner_run_id=owner.run_id, plan=inputs.plan,
            rejection=_touched(prepared, inputs, judged.candidates, followers),
            members={key: replace(member, followers=followers.get(key, ()))
                     for key, member in members.items()},
            mints=inputs.mints, read=owner.read_view, updated_at=owner.clock(),
            publish=owner.publish, groups_of=proposal_groups(writer),
            policy_release_ref=owner.policy_release_ref, reentry_woken=(), clock=owner.clock,
            observe=self.commits,
        )
        if carried:
            judged.committed.round2 = sorted([*judged.committed.round2, *carried],
                                             key=lambda entry: entry.position_key)

    def _settle_changed(
        self, owner: RoundSession, changed: Mapping[PositionKey, ChangedV1],
        followers: Mapping[PositionKey, tuple[FollowerV1, ...]], *, round_no: RoundNo,
    ) -> None:
        """Settle the members a resumed owner cannot review again (C.11.2 #2).

        A member reviewed in the last round before a privacy erase, or whose input in
        the last round no longer reproduces its earlier review of that round,
        gets its pending ``target_changed`` row in a rejection transaction of
        ``round_no`` (always ``MAX_ROUND``), with its
        primary when that is an existing entity, the ``result_index:judge`` of
        that review when it holds one, and its B.8 #3 duplicates mirroring it
        (C.12). A re-entered member keeps the touch it already has.
        """
        if not changed:
            return
        writer = owner.writer
        loaded = load_members([changed[key].proposal_ref for key in sorted(changed)], owner.view,
                              reader=writer, round=round_no)
        commit_rejection(
            writer, owner.write_ticket, owner_run_id=owner.run_id, round_no=round_no,
            entries=[RejectionEntryV1(
                (member.member.sigma, member.position_key),
                member.position(changed[member.position_key].candidate_id),
                disposition=DispositionV1(
                    "target_changed", member.member.slots[0].entity_id,
                    judge_result_ref=changed[member.position_key].result_ref,
                ),
                followers=followers.get(member.position_key, ()),
            ) for member in loaded],
            updated_at=owner.clock(), publish=owner.publish, clock=owner.clock,
        )


__all__ = ["CommitObserver", "OwnerRound", "OwnerRounds", "RoundEventV1", "RoundMemberV1",
           "RoundSession", "judge_binding"]
