"""One owner round's members, from ``ProposalV2`` to the judge and to the commit stage.

The owner loop calls this module once per round, around
`JudgeStage.review` and `commit_round`:

1. `load_members` rebuilds the round's members from their stored ``ProposalV2``,
   statement embeddings and extraction contexts at the round's pin, dropping
   every support group that is no longer usable, puts them
   in canonical order and gives each its siblings and the round's new-entity declarations.
2. `prepare_round` nominates every member with one `NominationIndex` of the
   round's pin, derives its cited-turn attribution once for both its judge input
   and its wake key, records its ``NeighborSnapshotV1``, and
   decides each re-entered member's action from the reuse lookup.
   A re-entered member whose rebuild leaves no usable group expires instead: no
   wake, no touch. ``RoundPreparationV1.review`` is the list for ``JudgeStage.review``.
3. After the verdicts, `round_commit_inputs` resolves every member that does
   not expire, the ones the judge did not see as non-writing siblings, and
   `component_member` and `rejection_entry` build the inputs of each member
   (``ComponentMemberV1``, ``RejectionEntryV1``).
4. After the round-1 commit, `woken` compares each rebuilt re-entered member's
   wake key at the round-2 pin with its round-1 key; `round2_entries` merges the
   ``reentry_woken`` entries with the commit stage's, one cause per position,
   and `round2_members` rebuilds the round-2 members, which
   `prepare_round` nominates with their picks (`with_picks`) and the
   targets resolution asked round 2 to issue.

Nothing here writes; every durable write is the commit stage's.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Final, Protocol, get_args

from quotemem.core.add_read_view import AddReadView, Sigma, span_sigma
from quotemem.core.entities import (
    EntityChoiceV1,
    MemoryKind,
    MemorySnapshotV2,
    NewEntityChoiceV1,
    PositionKey,
    QuoteSpanV1,
    ReasonCode,
)
from quotemem.core.errors import QuoteMemError
from quotemem.core.extraction_context import ExtractContextV2
from quotemem.core.judge_output import JudgeV8
from quotemem.core.judge_structure import explicit_correction, one_replacement
from quotemem.core.neighbor_snapshot import NeighborSnapshotV1, RevisesRouteV1
from quotemem.core.privacy_embeddings import PrivacyEmbeddingV1
from quotemem.core.proposal import EmbeddingUseV1, ProposalV2
from quotemem.core.readability import Readability
from quotemem.core.recall_agent import request_sentences
from quotemem.core.resolution import (
    CandidateV1,
    HeadTargetV1,
    IssueV1,
    MemberResolutionV1,
    PickV1,
    QuoteV1,
    ResolutionMemberV1,
    ResolutionPlanV1,
    RetiredTargetV1,
    Round2Cause,
    Round2EntryV1,
    Round2Reason,
    SiblingTargetV1,
    SlotV1,
    TargetV1,
    VersionV1,
    importance,
    resolve,
)
from quotemem.core.review_rounds import RoundNo
from quotemem.core.semantic import SourceTimeExpressionV1
from quotemem.core.semantic_embedding import StatementEmbeddingEntryV1, StatementEmbeddingV1
from quotemem.core.sensitivity import SensitivityClass
from quotemem.core.snapshot_memo import snapshot_rows
from quotemem.core.vector_score import UnitRows, unit_vector
from quotemem.ingestion.extraction_admission import (
    AdmittedFact,
    AdmittedNaming,
    AdmittedTime,
    RevisedMemory,
    span_key,
)
from quotemem.ingestion.forget_requests import is_request
from quotemem.ingestion.judge import JudgedMemberV1, ReviewMemberV1
from quotemem.ingestion.judge_input import MemberInputV1, build_member_input
from quotemem.ingestion.nomination import (
    ContentBinder,
    CueV1,
    DeclarationV1,
    FactTimeV1,
    HandleKey,
    MemberNamingV1,
    MemberSlotV1,
    MemberV1,
    NominatedTargetV1,
    NominationIndex,
    NominationPolicyV1,
    NominationV1,
    RevisesRefV1,
    SlotNominationV1,
    assign_handles,
    make_handle_key,
    neighbor_snapshot,
    resolve_slots,
    round_members,
    slot_key,
)
from quotemem.ingestion.reentry import (
    PendingCandidateV1,
    ReentryAction,
    ReentryTouchV1,
    cited_attributions,
    needs_round2_rebuild,
    reentry_action,
    round2_review,
    touch,
    wake_digest,
)
from quotemem.ingestion.sources import Attribution, LabelView
from quotemem.store.commit_fence import CAUSE_ORDER, FenceMemberV1, FinalV1
from quotemem.store.component_commit import ComponentMemberV1, MintV1
from quotemem.store.memory_commit import DispositionV1, PositionV1, RejectionEntryV1
from quotemem.store.metadata import StrictModel

Round = RoundNo


class LogicalReader(Protocol):
    """The artifact read the rebuild needs; `Writer.read_logical` satisfies it."""

    def read_logical(self, reference: str, expected_kind: str) -> StrictModel:
        """Load one logical artifact of a kind, validated."""
        ...


# Members from ProposalV2 ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LoadedMemberV1:
    """One member rebuilt from its stored records, with what the commit stage later needs.

    Attributes:
        member: The member at the round's pin, with siblings and declarations.
        proposal_ref: Its ``proposal_snapshot`` artifact.
        proposal: Its ``ProposalV2``.
        statement: Its statement embedding entry (vector and input digest).
    """

    member: MemberV1
    proposal_ref: str
    proposal: ProposalV2
    statement: StatementEmbeddingEntryV1

    @property
    def position_key(self) -> PositionKey:
        """The member's position."""
        return self.member.position_key

    def position(self, candidate_id: str | None = None) -> PositionV1:
        """Its candidate identity; ``candidate_id`` names a stored row (re-entry)."""
        proposal = self.proposal
        return PositionV1(
            proposal.position_key, proposal.extraction_ref, self.proposal_ref,
            proposal.statement_digest,
            tuple(sorted({span.evidence_id for span in proposal.quote_spans})), candidate_id,
        )


def _entry(reader: LogicalReader, use: EmbeddingUseV1, key: PositionKey,
           role: str) -> StatementEmbeddingEntryV1:
    batch = reader.read_logical(use.ref, "semantic_embedding")
    if isinstance(batch, PrivacyEmbeddingV1):
        if batch.original_ref != use.ref or batch.owner_run_id != key[0] or (
            use.index not in batch.indices
        ):
            raise QuoteMemError("binding_mismatch", "A proposal names an erased embedding entry")
        entry = batch.entries[batch.indices.index(use.index)]
    elif isinstance(batch, StatementEmbeddingV1) and use.index < len(batch.entries):
        entry = batch.entries[use.index]
    else:
        raise QuoteMemError("binding_mismatch", "A proposal names an embedding its batch lacks")
    if entry.role != role or key not in entry.uses:
        raise QuoteMemError("binding_mismatch", "A proposal's embedding is not its own")
    return entry


def _earlier(reader: LogicalReader, proposal: ProposalV2) -> frozenset[tuple[str, str]]:
    """The ``(memory_id, revision_id)`` the extraction context showed as earlier versions."""
    if not proposal.revises:
        return frozenset()
    context = reader.read_logical(proposal.context_ref, "decision_set")
    if not isinstance(context, ExtractContextV2):
        raise QuoteMemError("binding_mismatch", "A proposal's context is not an extraction context")
    return frozenset((item.memory_id, item.revision_id) for item in context.memories
                     if item.origin == "earlier")


def _sigma(view: AddReadView, spans: Sequence[QuoteSpanV1]) -> Sigma:
    found: list[Sigma] = []
    for span in spans:
        evidence = view.evidence(span.evidence_id)
        if evidence is None:
            raise QuoteMemError("binding_mismatch", "A member cites Evidence outside its scope")
        found.append(span_sigma(evidence, span.byte_start))
    return max(found)


def usable_support(
    view: AddReadView, spans: Sequence[QuoteSpanV1], groups: Sequence[Sequence[int]],
) -> tuple[tuple[QuoteSpanV1, ...], tuple[tuple[int, ...], ...]]:
    """The quotes and support groups a rebuild keeps at the pin.

    A group with a span that is no longer usable (``p_group``) is dropped with the
    quotes only it cites, and the kept groups are renumbered over the kept
    quotes. When no group is usable, all are kept: a re-entered member then
    expires, and a member of this owner settles ``evidence_invalidated``
    unjudged (`prepare_member`).

    Args:
        view: The review pin.
        spans: Original quote order.
        groups: Canonical support groups over those spans.

    Returns:
        Surviving quotes in original order and groups renumbered over them.
    """
    read = Readability(view.scope_kind, view.scope_id, view.now, view.evidence)
    kept = [group for group in groups if read.p_group(spans, group)] or list(groups)
    cited = sorted({index for group in kept for index in group})
    number = {index: new for new, index in enumerate(cited)}
    return (tuple(spans[index] for index in cited),
            tuple(sorted(tuple(sorted(number[index] for index in group)) for group in kept)))


def member_from_proposal(
    proposal: ProposalV2, view: AddReadView, *, reader: LogicalReader, round: Round,
) -> tuple[MemberV1, StatementEmbeddingEntryV1]:
    """Rebuild one member from its ``ProposalV2`` at a pin.

    Slots resolve at the pin (`resolve_slots`); quotes and groups are the
    settled ones the proposal stores, less every group a span of which is no
    longer usable at the pin, so that nomination, attribution, the judge
    input, the wake key and resolution all read the same rebuilt support;
    σ is the latest kept span's. A ``revises`` reference
    keeps its revision only when the extraction context showed it as an
    earlier version. Siblings and declarations are the round's (`load_members`).

    Args:
        proposal (ProposalV2): The position's stored proposal.
        view (AddReadView): The round's pin.
        reader (LogicalReader): Reads its embedding batch and context.
        round (Round): The review round.

    Returns:
        tuple[MemberV1, StatementEmbeddingEntryV1]: The member and its statement
            embedding entry.

    Raises:
        QuoteMemError: ``binding_mismatch`` if the proposal has no statement
            embedding or names records that are not its own.
    """
    key = proposal.position_key
    if proposal.statement_embedding is None:
        raise QuoteMemError("binding_mismatch", "A reviewed position has a statement embedding")
    statement = _entry(reader, proposal.statement_embedding, key, "statement")
    cue = None
    if proposal.corrects_text is not None:
        if proposal.cue_embedding is None:
            raise QuoteMemError("binding_mismatch", "A correction cue has its embedding")
        cue = CueV1(proposal.corrects_text, tuple(
            _entry(reader, proposal.cue_embedding, key, "correction_cue").vector))
    earlier = _earlier(reader, proposal)
    groups = tuple(tuple(proposal.quote_spans[index] for index in group)
                   for group in proposal.support_groups)
    fact = AdmittedFact(
        position=key[2], statement=proposal.statement, statement_digest=proposal.statement_digest,
        kind=proposal.kind, anchors=tuple(proposal.anchors),
        naming=tuple(AdmittedNaming(item.anchor_index, item.name, item.name_kind)
                     for item in proposal.naming), groups=groups,
        time=AdmittedTime(proposal.time_expr, proposal.valid_from, proposal.valid_to,
                          proposal.time_granularity, proposal.anchored_from),
        keywords=tuple(proposal.keywords),
        revises=tuple(RevisedMemory(item.memory_id, item.revision_id,
                                    (item.memory_id, item.revision_id) in earlier)
                      for item in proposal.revises), corrects_text=proposal.corrects_text,
        security_level=proposal.security_level,
        sensitivity_classes=tuple(proposal.sensitivity_classes),
        sensitivity_basis=tuple(proposal.sensitivity_basis), origin=proposal.origin,
        trust_tier=proposal.trust_tier, concerns=frozenset(proposal.concerns), duplicate_of=None,
    )
    return member_from_fact(
        fact, view, position_key=key, round=round, vector=statement.vector,
        embedding_space=proposal.embedding_space, spans=proposal.quote_spans,
        groups=groups, sensitivity=proposal.sensitivity_classes, cue=cue), statement


def member_from_fact(
    fact: AdmittedFact, view: AddReadView, *, position_key: PositionKey, round: Round,
    vector: Sequence[float], embedding_space: str, spans: Sequence[QuoteSpanV1],
    groups: Sequence[Sequence[QuoteSpanV1]], sensitivity: Sequence[SensitivityClass],
    cue: CueV1 | None = None,
) -> MemberV1:
    """Build a review member from admitted content, without artifact access.

    Args:
        fact: The admitted content, with revises classified at extraction's pin.
        view: The review pin.
        position_key: The position's original identity.
        round: The review round.
        vector: Its recorded statement vector.
        embedding_space: The vector's space.
        spans: Quote spans in their original order.
        groups: Settled support groups, including joined duplicates.
        sensitivity: Settled sensitivity classes.
        cue: Its recorded correction cue and vector, if any.

    Returns:
        The rebuilt member, before round siblings and declarations are attached.
    """
    indices = {span_key(span): index for index, span in enumerate(spans)}
    quotes, numbered = usable_support(
        view, spans, tuple(tuple(indices[span_key(span)] for span in group) for group in groups))
    expression = fact.time.expression
    return MemberV1(
        position_key=position_key, round=round, statement=fact.statement, kind=fact.kind,
        keywords=fact.keywords, vector=tuple(vector), embedding_space=embedding_space,
        slots=resolve_slots(view, position_key, fact.anchors), quotes=quotes, groups=numbered,
        sigma=_sigma(view, quotes),
        time=FactTimeV1(expression.text if isinstance(expression, SourceTimeExpressionV1)
                        else None, fact.time.valid_from, fact.time.valid_to, fact.time.granularity),
        sensitivity_classes=tuple(sensitivity),
        naming=tuple(MemberNamingV1(item.anchor_index, item.name, item.name_kind)
                     for item in fact.naming),
        revises=tuple(RevisesRefV1(item.memory_id, item.revision_id if item.earlier else None)
                      for item in fact.revises), cue=cue)


def member_declarations(
    members: Iterable[tuple[PositionKey, Sigma, Sequence[EntityChoiceV1]]],
) -> tuple[DeclarationV1, ...]:
    """Order a round's new-entity declarations by their first anchored position.

    Args:
        members: Usable members' positions, rebuilt sigma and original anchor choices.

    Returns:
        One declaration per mint key, in canonical order.
    """
    found: dict[tuple[str, int, str], DeclarationV1] = {}
    for position, sigma, anchors in members:
        owner_run_id, unit_no, _ = position
        order = (sigma, position)
        for choice in anchors:
            if not isinstance(choice, NewEntityChoiceV1):
                continue
            key = (owner_run_id, unit_no, choice.local_key)
            known = found.get(key)
            if known is None or order < known.order:
                found[key] = DeclarationV1(owner_run_id, unit_no, choice.local_key, choice.name,
                                           choice.descriptor, choice.entity_type, order)
    return tuple(sorted(found.values(), key=lambda item: (item.order, item.mint_key)))


def load_members(
    proposal_refs: Iterable[str], view: AddReadView, *, reader: LogicalReader, round: Round,
) -> tuple[LoadedMemberV1, ...]:
    """Rebuild one round's members from their stored proposals.

    Args:
        proposal_refs (Iterable[str]): The ``proposal_snapshot`` of every member
            of the round: the owner's own positions and its re-entered ones.
        view (AddReadView): The round's pin.
        reader (LogicalReader): The artifact reader.
        round (Round): The review round.

    Returns:
        tuple[LoadedMemberV1, ...]: The members in canonical order ``(σ,
            position_key)``, each with the round's declarations and every
            earlier member as a sibling. A member without a usable group at
            the pin, which expires or is refused (`prepare_member`), takes no
            part: it is no one's sibling, declares nothing and has no siblings.
    """
    loaded: dict[PositionKey, tuple[str, ProposalV2, MemberV1, StatementEmbeddingEntryV1]] = {}
    for reference in proposal_refs:
        proposal = reader.read_logical(reference, "proposal_snapshot")
        if not isinstance(proposal, ProposalV2):
            raise QuoteMemError("binding_mismatch", "A member's proposal is not a ProposalV2")
        if proposal.position_key in loaded:
            raise QuoteMemError("binding_mismatch", "A position is one member of its round")
        member, statement = member_from_proposal(proposal, view, reader=reader, round=round)
        loaded[proposal.position_key] = (reference, proposal, member, statement)
    taking = {key for key, item in loaded.items() if usable(item[2], view)}
    declarations = member_declarations((item[2].position_key, item[2].sigma, item[1].anchors)
                                       for key, item in loaded.items() if key in taking)
    ordered = sorted((
        *round_members(replace(item[2], declarations=declarations)
                       for key, item in loaded.items() if key in taking),
        *(item[2] for key, item in loaded.items() if key not in taking),
    ), key=lambda member: (member.sigma, member.position_key))
    return tuple(LoadedMemberV1(member, loaded[member.position_key][0],
                                loaded[member.position_key][1], loaded[member.position_key][3])
                 for member in ordered)


# One round's nominations, inputs and re-entry actions ---------------------------


@dataclass(frozen=True, slots=True)
class PreparedMemberV1:
    """One member of a round after nomination, before the judge.

    Attributes:
        loaded: The rebuilt member, with its round-2 picks (`with_picks`).
        nomination: Its nomination at the round's pin.
        attributions: Its cited-turn attribution, derived once.
        input: Its member input; None when it is not reviewable.
        snapshot: Its ``NeighborSnapshotV1``; None when it is not reviewable.
        pending: Its selected row when it re-entered this owner.
        wake: Its wake key at this rebuild; None for a fresh member and an expiry.
        action: Its re-entry action; None for a fresh member and an expiry.
        touch: Its re-entry touch at this owner; None for a fresh member and an
            expiry.
        expire: Its rebuild left no usable group: the row expires.
        invalidated: The owner's own member's rebuild left no usable group, after
            a privacy erase or expiry since its admission: it is never
            judged and settles ``evidence_invalidated``.
    """

    loaded: LoadedMemberV1
    nomination: NominationV1
    attributions: tuple[Attribution, ...]
    input: MemberInputV1 | None = None
    snapshot: NeighborSnapshotV1 | None = None
    pending: PendingCandidateV1 | None = None
    wake: str | None = None
    action: ReentryAction | None = None
    touch: ReentryTouchV1 | None = None
    expire: bool = False
    invalidated: bool = False

    @property
    def position_key(self) -> PositionKey:
        """The member's position."""
        return self.loaded.position_key

    @property
    def review(self) -> ReviewMemberV1 | None:
        """Its ``JudgeStage.review`` entry: reviewable, not expired, not quiet or unreviewed."""
        if self.input is None or self.snapshot is None or self.expire:
            return None
        if self.action in ("quiet", "unreviewed"):
            return None
        return ReviewMemberV1(self.nomination, self.input, self.snapshot)


@dataclass(frozen=True, slots=True)
class RoundPreparationV1:
    """One owner round, ready for the judge.

    Attributes:
        round: The review round.
        members: Every member in canonical order.
    """

    round: Round
    members: tuple[PreparedMemberV1, ...]

    @property
    def review(self) -> tuple[ReviewMemberV1, ...]:
        """The members for ``JudgeStage.review(members, round)``, in canonical order."""
        return tuple(item for member in self.members if (item := member.review) is not None)

    @property
    def actions(self) -> dict[PositionKey, ReentryAction]:
        """Each re-entered, not expired member's action, for `ReentryCensusV1.of`."""
        return {member.position_key: member.action for member in self.members
                if member.action is not None}

    @property
    def expired(self) -> tuple[PositionKey, ...]:
        """The re-entered members whose rows expire at this rebuild."""
        return tuple(member.position_key for member in self.members if member.expire)

    def member(self, key: PositionKey) -> PreparedMemberV1:
        """Return one member by position."""
        for member in self.members:
            if member.position_key == key:
                return member
        raise KeyError(key)


def usable(member: MemberV1, view: AddReadView) -> bool:
    """Tell whether any support group of the member is usable at the pin (``p_group``)."""
    read = Readability(view.scope_kind, view.scope_id, view.now, view.evidence)
    return any(read.p_group(member.quotes, group) for group in member.groups)


Reusable = Callable[[ReviewMemberV1], object | None]
"""The reuse lookup, ``JudgeRecords.reusable``: the accepted output for a member's input."""


def with_picks(member: MemberV1, picks: Sequence[PickV1], view: AddReadView,
               declared: Mapping[str, DeclarationV1] | None = None) -> MemberV1:
    """Return a round-2 member with each re-picked slot nominated to its pick.

    The pick is the judge's round-1 choice: an active entity, by its id or by
    the ``mint_key`` a commit minted it under, or a round-1 declaration that no
    commit minted yet (get-or-create at commit), also when the member
    that declared it is not in round 2. The slot keeps its mention. A pick that
    is none of these at the round-2 pin leaves the slot open, as does a
    declaration whose minted entity was erased (``AddReadView.spent``).

    Args:
        member (MemberV1): The member rebuilt at the round-2 pin.
        picks (Sequence[PickV1]): Its ``Round2RequestV1.picks``.
        view (AddReadView): The round-2 pin.
        declared (Mapping[str, DeclarationV1] | None): Round 1's declarations by
            ``mint_key`` (`Round2PlanV1.declarations`), beside the member's own.

    Returns:
        MemberV1: The member whose picked slots are ``nominated``.
    """
    chosen = {pick.slot: pick.entity for pick in picks}
    declarations = {**(declared or {}), **{item.mint_key: item for item in member.declarations}}
    slots: list[MemberSlotV1] = []
    for slot in member.slots:
        key = chosen.get(slot.slot)
        record = None if key is None else view.entity(key) or view.minted(key)
        declaration = None if key is None or view.spent(key) else declarations.get(key)
        if key is None:
            slots.append(slot)
        elif record is not None:
            slots.append(replace(slot, binding="nominated", entity_id=record.entity_id,
                                 mint_key=None, participant_key=None))
        elif declaration is not None:
            slots.append(replace(slot, binding="nominated", entity_id=None, mint_key=key,
                                 participant_key=None, name=declaration.name,
                                 descriptor=declaration.descriptor,
                                 entity_type=declaration.entity_type))
        else:
            slots.append(MemberSlotV1(slot.slot, "open", mention=slot.mention))
    return replace(member, slots=tuple(slots))


def prepare_member(
    loaded: LoadedMemberV1, index: NominationIndex, *, policy: NominationPolicyV1,
    bind: ContentBinder, owner_run_id: str, labels: LabelView,
    pending: PendingCandidateV1 | None = None, reusable: Reusable = lambda member: None,
    issued: Sequence[IssueV1] = (), picks: Sequence[PickV1] = (),
    declared: Mapping[str, DeclarationV1] | None = None, reason_code: ReasonCode | None = None,
    self_identity: bool = False,
    placeholders: Mapping[tuple[str, int, int], str] | None = None,
) -> PreparedMemberV1:
    """Nominate one member and decide its re-entry action.

    A round-2 member with targets to issue or slots re-picked in round 1 is
    nominated with them for its judge input, snapshot and resolution, while its
    wake key is computed over its plain rebuild: W never covers a round-2 pick
    or targets round 2 was asked to issue, so the snapshot of such a nomination
    records no wake key, and an unchanged wake key never leaves it quiet: it
    gets the review round 2 was asked for.

    Args:
        loaded (LoadedMemberV1): The rebuilt member.
        index (NominationIndex): The round's pin, prepared once for all its members.
        policy (NominationPolicyV1): The frozen nomination policy.
        bind (ContentBinder): The workspace's keyed binder.
        owner_run_id (str): The owner reviewing the round.
        labels (LabelView): The owner's frozen turn labels.
        pending (PendingCandidateV1 | None): The selected row of a re-entered member.
        reusable (Reusable): the reuse lookup (``JudgeRecords.reusable``).
        issued (Sequence[IssueV1]): Round 2: the targets the member must be issued.
        picks (Sequence[PickV1]): Round 2: the slots the judge re-picked in round 1.
        declared (Mapping[str, DeclarationV1] | None): Round 2: round 1's
            declarations by ``mint_key``, which a pick may name (`with_picks`).
        reason_code (ReasonCode | None): The code its wake key reads; defaults
            to the row's code read at selection.
        self_identity (bool): The owner's policy has the ``user-role-v1`` self-identity
            rule: a re-entered ``identity_unresolved`` member selected
            ``IDENTITY_SETTLE_AFTER`` times expires (``PendingCandidateV1.identity_expires``).

    Returns:
        PreparedMemberV1: The member, with its picks, its nomination, input,
            snapshot and, when re-entered, its wake key, action and touch, or
            its expiry.
    """
    view, plain = index.view, loaded.member
    member = with_picks(plain, picks, view, declared) if picks else plain
    nomination = index.nominate(member, policy, issued=issued)
    if not usable(plain, view):
        # An erase or expiry left no usable group, so the member is never
        # judged and its cited turns, an erased one perhaps among them, are not
        # attributed: a re-entered row expires, the owner's own member is refused.
        return PreparedMemberV1(loaded, nomination, (), pending=pending,
                                expire=pending is not None, invalidated=pending is None)
    attributions = cited_attributions(plain, view, labels)
    if self_identity and pending is not None and pending.identity_expires:
        # The user-role-v1 rule ends the wait of an identity nobody supplied.
        return PreparedMemberV1(loaded, nomination, attributions, pending=pending, expire=True)
    code = reason_code if reason_code is not None else (
        pending.reason_code if pending is not None else None)
    rebuilt = bool(issued) or bool(picks)
    wake = None if pending is None or code is None else wake_digest(
        index.nominate(plain, policy) if rebuilt else nomination, view, bind=bind,
        owner_run_id=owner_run_id, reason_code=code, attributions=attributions,
    )
    if picks:
        loaded = replace(loaded, member=member)
    item = snapshot = None
    if nomination.reviewable:
        item = build_member_input(nomination, view, bind=bind, attributions=attributions,
                                  placeholders=placeholders)
        snapshot = neighbor_snapshot(
            nomination, bind=bind, owner_run_id=owner_run_id, member_input_digest=item.digest,
            wake_digest=None if rebuilt else wake,
            reentry_reason_code=code if pending is not None else None,
        )
    if pending is None or wake is None:
        return PreparedMemberV1(loaded, nomination, attributions, item, snapshot)
    accepted = False
    if item is not None and snapshot is not None:
        accepted = reusable(ReviewMemberV1(nomination, item, snapshot)) is not None
    action = reentry_action(pending, reviewable=item is not None, wake_digest=wake,
                            member_input_digest=None if item is None else item.digest,
                            accepted_output=accepted)
    if rebuilt and action == "quiet":
        action = "review"
    reviewed = item.digest if action == "review" and item is not None else None
    return PreparedMemberV1(
        loaded, nomination, attributions, item, snapshot, pending, wake, action,
        touch(pending, action, wake_digest=wake, reviewed_input=reviewed),
    )


def prepare_round(
    members: Sequence[LoadedMemberV1], view: AddReadView, *, round: Round,
    policy: NominationPolicyV1, bind: ContentBinder, owner_run_id: str, labels: LabelView,
    pending: Mapping[PositionKey, PendingCandidateV1] | None = None,
    reusable: Reusable = lambda member: None,
    issued: Mapping[PositionKey, Sequence[IssueV1]] | None = None,
    picks: Mapping[PositionKey, Sequence[PickV1]] | None = None,
    declared: Mapping[str, DeclarationV1] | None = None, self_identity: bool = False,
    relearn: bool = False, placeholders: Mapping[tuple[str, int, int], str] | None = None,
) -> RoundPreparationV1:
    """Prepare one owner round for ``JudgeStage.review`` (the RoundStage).

    Args:
        members (Sequence[LoadedMemberV1]): The round's members (`load_members`).
        view (AddReadView): The round's pin.
        round (Round): The review round.
        policy (NominationPolicyV1): The frozen nomination policy.
        bind (ContentBinder): The workspace's keyed binder.
        owner_run_id (str): The owner reviewing the round.
        labels (LabelView): The owner's frozen turn labels.
        pending (Mapping[PositionKey, PendingCandidateV1] | None): The selected
            rows of the re-entered members (`ReentrySelectionV1.selected`).
        reusable (Reusable): the reuse lookup (``JudgeRecords.reusable``).
        issued (Mapping[PositionKey, Sequence[IssueV1]] | None): Round 2: each
            member's must-issue targets (``Round2RequestV1.issue``, ``?``).
        picks (Mapping[PositionKey, Sequence[PickV1]] | None): Round 2: each
            member's re-picked slots (``Round2RequestV1.picks``).
        declared (Mapping[str, DeclarationV1] | None): Round 2: round 1's
            declarations by ``mint_key`` (`Round2PlanV1.declarations`).
        self_identity (bool): The owner's ``user-role-v1`` rule (`prepare_member`).
        relearn (bool): The owner's policy learns again what the user states after a
            forget (`NominationIndex`).
        placeholders (Mapping[tuple[str, int, int], str] | None): The forgotten name
            occurrences the owner's settlement withheld and how the judge sees each
            (``ExtractionOutcome.placeholders``, `build_member_input`).

    Returns:
        RoundPreparationV1: Every member prepared; ``review`` is the judge's list.
    """
    rows = dict(pending or {})
    targets = dict(issued or {})
    chosen = dict(picks or {})
    if any(member.member.round != round for member in members):
        raise QuoteMemError("binding_mismatch", "A round prepares its own members")
    if not members:
        return RoundPreparationV1(round, ())
    index = NominationIndex(view, relearn=relearn)
    index.prime([member.member for member in members])
    return RoundPreparationV1(round, tuple(
        prepare_member(member, index, policy=policy, bind=bind, owner_run_id=owner_run_id,
                       labels=labels, pending=rows.get(member.position_key),
                       reusable=reusable, issued=targets.get(member.position_key, ()),
                       picks=chosen.get(member.position_key, ()), declared=declared,
                       self_identity=self_identity, placeholders=placeholders)
        for member in members
    ))


# Commit inputs ------------------------------------------------------------------------


def relations(prepared: PreparedMemberV1, judged: JudgedMemberV1, *,
              bind: ContentBinder) -> dict[str, str]:
    """The member's judged relation to each issued head, by memory id."""
    judgment = judged.judgment
    verdict = None if judgment is None else judgment.verdict
    if verdict is None:
        return {}
    handles = assign_handles(prepared.nomination, bind)
    by_handle = {handles[target.handle_key]: target.object_id
                 for target in prepared.nomination.targets if target.kind == "head"}
    return {by_handle[item.target]: item.relation for item in verdict.relations
            if item.target in by_handle}


def siblings(nomination: NominationV1) -> frozenset[PositionKey]:
    """The positions of the member's issued siblings."""
    return frozenset(target.sibling.position_key for target in nomination.targets
                     if target.kind == "sibling" and target.sibling is not None)


class NoveltyRows:
    """One pin's ``p_mem`` heads by primary and space, stacked once for a round's novelty.

    A round scores each of its members against every head of its primary, and a long
    user's heads were stacked again for each member: 20% of an Add at 8k heads.

    Args:
        view: The round's pin.
    """

    def __init__(self, view: AddReadView) -> None:
        self._view = view
        self._heads: dict[tuple[str, str], list[MemorySnapshotV2]] | None = None
        self._rows: dict[tuple[str, str], UnitRows] = {}

    def cosine(self, member: MemberV1) -> float | None:
        """The highest cosine to a ``p_mem`` head with the member's primary at the pin."""
        primary = member.slots[0].entity_id if member.slots else None
        if primary is None:
            return None
        if self._heads is None:
            self._heads = {}
            for record in self._view.heads():
                snapshot = record.head.snapshot
                self._heads.setdefault((snapshot.entity_id, snapshot.embedding_space), []).append(
                    snapshot,
                )
        key = primary, member.embedding_space
        rows = self._rows.get(key)
        if rows is None:
            rows = self._rows[key] = snapshot_rows(self._heads.get(key, []))
        found = rows.cosines(unit_vector(member.vector))
        return max(found) if found else None


def content(loaded: LoadedMemberV1, novelty: NoveltyRows, *,
            weights: Mapping[str, float]) -> dict[str, Any]:
    """The snapshot fields of the member's statement that the commit does not set."""
    proposal = loaded.proposal
    inputs = proposal.importance_inputs
    return {
        "schema_version": 2, "statement": proposal.statement,
        "statement_digest": proposal.statement_digest, "kind": proposal.kind,
        "time_expr": proposal.time_expr.model_dump(mode="json"),
        "valid_from": proposal.valid_from, "valid_to": proposal.valid_to,
        "time_granularity": proposal.time_granularity,
        "anchored_from": proposal.anchored_from.model_dump(mode="json"),
        "keywords": list(proposal.keywords),
        "importance": importance(novelty.cosine(loaded.member),
                                 durability=inputs.durability,
                                 actionability=inputs.actionability, weights=weights),
        "origin": proposal.origin, "trust_tier": proposal.trust_tier,
        "security_level": proposal.security_level,
        "sensitivity_classes": list(proposal.sensitivity_classes),
        "sensitivity_basis": list(proposal.sensitivity_basis),
        "embedding": list(loaded.statement.vector),
        "embedding_space": proposal.embedding_space,
        "embedding_input_digest": loaded.statement.input_digest,
        "pipeline_version": proposal.pipeline_version,
    }


def _slot(item: SlotNominationV1, handles: Mapping[HandleKey, str]) -> SlotV1:
    slot = item.slot
    key = slot_key(slot)
    entity = slot.entity_id or slot.mint_key or slot.participant_key
    if slot.binding not in ("nominated", "open"):
        return SlotV1(slot=slot.slot, binding=slot.binding, entity=entity, nominated=None,
                      candidates=())
    return SlotV1(
        slot=slot.slot, binding=slot.binding,
        entity=entity if slot.binding == "nominated" else None,
        nominated=handles[key] if slot.binding == "nominated" and key is not None else None,
        candidates=tuple(CandidateV1(handle=handles[candidate.handle_key], entity=candidate.key)
                         for candidate in item.candidates),
    )


def _version(memory_id: str, revision_id: str, handle: str | None,
             snapshot: MemorySnapshotV2) -> VersionV1:
    spans = tuple(dict.fromkeys(snapshot.quote_spans[index]
                                for group in snapshot.support_groups for index in group))
    return VersionV1(memory_id=memory_id, revision_id=revision_id, handle=handle, spans=spans)


def _alias_only[K](annotations: Iterable[tuple[K, str]]) -> tuple[K, ...]:
    """The keys whose every naming annotation is an ``alias``, in first order."""
    kinds: dict[K, set[str]] = {}
    for key, kind in annotations:
        kinds.setdefault(key, set()).add(kind)
    return tuple(key for key, found in kinds.items() if found == {"alias"})


def _dated(view: AddReadView, spans: Iterable[QuoteSpanV1], sigma: Sigma) -> bool:
    """Whether a version's σ takes its time from a stated source time, not a receipt."""
    for span in spans:
        evidence = view.evidence(span.evidence_id)
        if evidence is not None and span_sigma(evidence, span.byte_start) == sigma:
            return evidence.source_observed_at is not None
    return True


def _target(target: NominatedTargetV1, handles: Mapping[HandleKey, str],
            history: Mapping[str, str], view: AddReadView) -> TargetV1:
    handle = handles[target.handle_key]
    if target.sibling is not None:
        return SiblingTargetV1(kind="sibling", handle=handle,
                               position_key=target.sibling.position_key)
    version = target.version
    if version is None:
        raise QuoteMemError("binding_mismatch", "A stored target keeps its version")
    snapshot = version.snapshot
    if target.kind == "head":
        return HeadTargetV1(
            kind="head", handle=handle, memory_id=target.object_id,
            revision_id=version.revision_id, sigma=target.sigma, primary=snapshot.entity_id,
            naming=tuple(dict.fromkeys(row.entity_id for row in snapshot.naming)),
            groups=tuple(tuple(snapshot.quote_spans[index] for index in group)
                         for group in snapshot.support_groups),
            retired=tuple(_version(entry.version.memory_id, entry.version.revision_id,
                                   history.get(entry.version.revision_id),
                                   entry.version.snapshot)
                          for entry in view.history(target.object_id)),
            aliases=_alias_only((row.entity_id, row.name_kind) for row in snapshot.naming),
            period=(snapshot.valid_from, snapshot.valid_to),
            dated=_dated(view, snapshot.quote_spans, target.sigma),
        )
    older = [entry.version for entry in view.retired()
             if entry.version.memory_id == version.memory_id
             and entry.version.recorded_at <= version.recorded_at]
    return RetiredTargetV1(
        kind="retired", handle=handle, memory_id=version.memory_id,
        revision_id=version.revision_id, sigma=target.sigma,
        retired=tuple(_version(row.memory_id, row.revision_id,
                               handle if row.revision_id == version.revision_id else None,
                               row.snapshot) for row in older),
        dated=_dated(view, snapshot.quote_spans, target.sigma),
    )


def resolution_member(
    prepared: PreparedMemberV1, judged: JudgedMemberV1 | None, view: AddReadView, *,
    bind: ContentBinder,
) -> ResolutionMemberV1:
    """Build one member's resolution input from its nomination and judge outcome.

    A member the judge did not see settles without a verdict, as a non-writing
    member that a later member's relation to it treats as ``independent``:
    a quiet re-entered member with the code it keeps, an
    unreviewable or ``unreviewed`` one as ``identity_unresolved``; one the
    stage settled (``review_unavailable``, ``review_capacity_exceeded``) with
    the stage's code. An expired member is not resolved: its rejection entry
    carries its expiry alone (`rejection_entry`).

    Args:
        prepared (PreparedMemberV1): The member as `prepare_round` left it.
        judged (JudgedMemberV1 | None): Its outcome in ``JudgeStage.review``;
            None when the judge did not see it.
        view (AddReadView): The round's pin.
        bind (ContentBinder): The workspace's keyed binder (handles).

    Returns:
        ResolutionMemberV1: The member as ``resolve`` reads it.

    Raises:
        QuoteMemError: ``binding_mismatch`` for an expired member, or an outcome of
            another position.
    """
    if prepared.expire:
        raise QuoteMemError("binding_mismatch", "An expired member is not resolved")
    if judged is not None and judged.position_key != prepared.position_key:
        raise QuoteMemError("binding_mismatch", "A judge outcome is its member's")
    nomination = prepared.nomination
    settled: ReasonCode | None = None
    verdict = None
    if prepared.invalidated:
        settled = "evidence_invalidated"
    elif prepared.action == "quiet" and prepared.pending is not None:
        settled = prepared.pending.reason_code
    elif prepared.review is None or judged is None:
        settled = "identity_unresolved"
    elif judged.code is not None:
        settled = judged.code
    elif judged.judgment is not None:
        verdict = judged.judgment.verdict
        if verdict is None:
            settled = "review_unavailable"
    # Only a correction's wait ends. A member that is not one carries the code
    # only as the duplicate of an unresolved correction and resolves on its verdict.
    settle = (verdict is not None and prepared.pending is not None and prepared.pending.settles
              and nomination.flags.correction_requested)
    return resolution_input(nomination, view, bind=bind, verdict=verdict, settled=settled,
                            settle=settle)


def resolution_input(
    nomination: NominationV1, view: AddReadView, *, bind: ContentBinder,
    verdict: JudgeV8 | None = None, settled: ReasonCode | None = None, settle: bool = False,
) -> ResolutionMemberV1:
    """Build the input from a nomination and its accepted verdict or settlement.

    This record-independent seam is shared by the owner and the prompt lab.
    Handles, support, slots and version targets all come from the same pin.

    Args:
        nomination: The member's frozen nomination.
        view: The round's pin.
        bind: The workspace's content binder.
        verdict: The accepted judge verdict, if reviewed.
        settled: The code decided without a verdict, if any.
        settle: A re-entered correction whose wait is over (``SETTLE_AFTER``); never
            set for a member that is not a correction.

    Returns:
        The complete input to the pure resolver.
    """
    member = nomination.member
    handles = assign_handles(nomination, bind)
    quotes: list[QuoteV1] = []
    for span in member.quotes:
        evidence = view.evidence(span.evidence_id)
        if evidence is None:
            raise QuoteMemError("binding_mismatch", "A member cites Evidence outside its scope")
        quotes.append(QuoteV1(
            handle=handles[make_handle_key("quote", span.evidence_id,
                                           f"{span.byte_start}-{span.byte_end}")],
            span=span, sigma=span_sigma(evidence, span.byte_start),
            user_message=evidence.kind == "user_message",
        ))
    history = {entry.entry.version.revision_id: handles[entry.handle_key]
               for entry in nomination.history}
    flags = nomination.flags
    targets = tuple(_target(target, handles, history, view) for target in nomination.targets)
    heads = {target.memory_id for target in targets if isinstance(target, HeadTargetV1)}
    revised = tuple(sorted({reference.issued_memory_id for route in nomination.routes
                            if isinstance(route, RevisesRouteV1) for reference in route.references
                            if reference.revision_id is None
                            and reference.issued_memory_id in heads}))
    return ResolutionMemberV1(
        position_key=member.position_key,
        slots=tuple(_slot(slot, handles) for slot in nomination.slots),
        naming=tuple(sorted({naming.slot for naming in member.naming})),
        aliases=tuple(sorted(_alias_only((naming.slot, naming.name_kind)
                                         for naming in member.naming))),
        quotes=tuple(quotes),
        groups=tuple(tuple(quotes[index].handle for index in group) for group in member.groups),
        correction_requested=flags.correction_requested,
        history_correction_requested=flags.history_correction_requested,
        external_only_allowed=flags.external_only_allowed,
        targets=targets, verdict=verdict, settled=settled,
        period=(member.time.valid_from, member.time.valid_to),
        revised=revised, settle=settle,
    )


def _kinds(members: Iterable[PreparedMemberV1]) -> dict[PositionKey | str, MemoryKind]:
    """The kind of each member, by position, and of each head issued to one, by memory id."""
    found: dict[PositionKey | str, MemoryKind] = {}
    for item in members:
        found[item.position_key] = item.loaded.member.kind
        for target in item.nomination.targets:
            if target.kind == "head" and target.version is not None:
                found[target.object_id] = target.version.snapshot.kind
    return found


def _replaced(members: Iterable[PreparedMemberV1]) -> dict[PositionKey, frozenset[str]]:
    """By member, the issued heads its statement replaces with one new value
    (``judge_structure.one_replacement``). A sensitive member is never read so: its other
    words may state a special-category fact the judge counted."""
    found: dict[PositionKey, frozenset[str]] = {}
    for item in members:
        proposal = item.loaded.proposal
        if proposal.sensitivity_classes:
            continue
        named = frozenset(
            target.object_id for target in item.nomination.targets
            if target.kind == "head" and target.version is not None
            and one_replacement(proposal.statement, target.version.snapshot.statement))
        if named:
            found[item.position_key] = named
    return found


def _told_with_request(members: Iterable[ResolutionMemberV1],
                       view: AddReadView) -> frozenset[PositionKey]:
    """The members a quote of which meets a user's forget-request sentence: told with the
    request, they change no stored memory (``resolve``'s ``requested``)."""
    asking: dict[str, list[tuple[int, int]]] = {}
    found: set[PositionKey] = set()
    for member in members:
        for quote in member.quotes:
            span = quote.span
            if span.evidence_id not in asking:
                part = view.evidence(span.evidence_id)
                asking[span.evidence_id] = [] if part is None or part.role != "user" \
                    else _request_ranges(part.text, part.byte_start)
            if any(low < span.byte_end and span.byte_start < high
                   for low, high in asking[span.evidence_id]):
                found.add(member.position_key)
                break
    return frozenset(found)


def _request_ranges(text: str, offset: int) -> list[tuple[int, int]]:
    """The byte ranges, in message coordinates from ``offset``, of a user part's forget-request
    sentences (`forget_requests.is_request`, the ``agent`` mode's forms)."""
    if not is_request(text, complete=True):
        return []
    return [(offset + len(text[:start].encode("utf-8")), offset + len(text[:end].encode("utf-8")))
            for start, end in request_sentences(text)
            if is_request(text[start:end], complete=True)]


def mints(members: Iterable[PreparedMemberV1]) -> dict[str, MintV1]:
    """The round's get-or-create keys: declarations and participant slots.

    A resolution ``EntityKey`` that is not an entity id is a declaration's
    ``mint_key`` or a participant slot's ``participant_key``; the commit stage resolves it
    through this map.
    """
    found: dict[str, MintV1] = {}
    for prepared in members:
        member = prepared.loaded.member
        for declaration in member.declarations:
            found[declaration.mint_key] = MintV1(mint_key=declaration.mint_key,
                                                 entity_type=declaration.entity_type)
        for slot in member.slots:
            if slot.mint_key is not None:
                found[slot.mint_key] = MintV1(mint_key=slot.mint_key,
                                              entity_type=slot.entity_type)
            elif slot.participant_key is not None and slot.entity_id is None:
                found[slot.participant_key] = MintV1(participant_key=slot.participant_key)
    return found


def _final(member: MemberV1, resolution: MemberResolutionV1,
           keys: Mapping[str, MintV1]) -> FinalV1 | None:
    anchor = resolution.anchors[0] if resolution.anchors else None
    if anchor is None:
        return None
    spec = keys.get(anchor)
    if spec is None:
        return FinalV1(member.statement_digest, entity_id=anchor)
    return FinalV1(member.statement_digest, mint_key=spec.mint_key,
                   participant_key=spec.participant_key)


def component_member(
    prepared: PreparedMemberV1, resolution: MemberResolutionV1, judged: JudgedMemberV1,
    novelty: NoveltyRows, *, bind: ContentBinder, weights: Mapping[str, float],
    keys: Mapping[str, MintV1], candidate_id: str | None = None,
) -> ComponentMemberV1:
    """Build one component member's commit input after its verdict.

    Args:
        prepared (PreparedMemberV1): The member as `prepare_round` left it.
        resolution (MemberResolutionV1): Its final disposition in this round.
        judged (JudgedMemberV1): Its judge outcome (``JudgeStage.review``).
        novelty (NoveltyRows): The round's pin's heads.
        bind (ContentBinder): The workspace's keyed binder (handles).
        weights (Mapping[str, float]): The extraction policy's importance weights.
        keys (Mapping[str, MintV1]): The round's `mints`, the same map the commit stage gets.
        candidate_id (str | None): Its stored row, for a re-entered member.

    Returns:
        ComponentMemberV1: Its candidate identity, resolution, fence inputs,
            statement content, naming, verdict ref, tombstone hold and touch.
    """
    key = prepared.position_key
    if (resolution.position_key, judged.position_key) != (key, key):
        raise QuoteMemError("binding_mismatch", "A component member is one position")
    member = prepared.loaded.member
    write = resolution.write
    judgment = judged.judgment
    fence = FenceMemberV1(
        position_key=key, nomination=prepared.nomination,
        siblings=siblings(prepared.nomination),
        relations=relations(prepared, judged, bind=bind),
        write_heads=write.fence if write is not None else (),
        corrects=write.corrects if write is not None else (),
        groups=write.groups if write is not None else (),
        final=_final(member, resolution, keys) if write is not None else None,
    )
    return ComponentMemberV1(
        position=prepared.loaded.position(candidate_id), resolution=resolution, fence=fence,
        content=content(prepared.loaded, novelty, weights=weights) if write is not None else None,
        naming=tuple((item.slot, item.name, item.name_kind) for item in member.naming),
        judge_result_ref=None if judgment is None else judgment.result_ref,
        hold_evidence=bool(prepared.nomination.tombstones), touch=prepared.touch,
        correction_requested=prepared.loaded.proposal.corrects_text is not None or bool(
            prepared.loaded.proposal.revises,
        ),
        explicit_correction=explicit_correction(
            prepared.loaded.proposal.statement, prepared.loaded.proposal.corrects_text,
        ),
        spans=member.quotes,
    )


def rejection_entry(
    prepared: PreparedMemberV1, *, disposition: DispositionV1 | None = None,
    candidate_id: str | None = None, withheld: tuple[QuoteSpanV1, ...] = (),
) -> RejectionEntryV1:
    """Build one member's entry in the round's rejection transaction.

    An expired member writes only its expiry; a quiet or unreviewed
    re-entered member only its touch; any other member its disposition, with
    its touch when re-entered.

    Args:
        prepared (PreparedMemberV1): The member as `prepare_round` left it.
        disposition (DispositionV1 | None): Its disposition outside every component.
        candidate_id (str | None): Its stored row, for a re-entered member.
        withheld (tuple[QuoteSpanV1, ...]): Spans a ``denied_by_forget`` withholds.

    Returns:
        RejectionEntryV1: The entry, ordered by its canonical key.
    """
    member = prepared.loaded.member
    order = (member.sigma, member.position_key)
    position = prepared.loaded.position(candidate_id)
    if prepared.expire:
        return RejectionEntryV1(order, position, expire=True)
    return RejectionEntryV1(order, position, disposition=disposition, withheld=withheld,
                            touch=prepared.touch)


def rejected_disposition(
    resolution: MemberResolutionV1, judged: JudgedMemberV1 | None, *,
    keys: Mapping[str, MintV1], hold_evidence: bool = False,
) -> DispositionV1:
    """A reviewed member's disposition outside every component (rejection transaction).

    Nothing is minted in the rejection transaction, so a primary that only a
    declaration or participant key names is recorded as no entity.

    Args:
        resolution (MemberResolutionV1): Its final disposition in the round.
        judged (JudgedMemberV1 | None): Its judge outcome; None when it was not reviewed.
        keys (Mapping[str, MintV1]): The round's get-or-create keys (`mints`).
        hold_evidence (bool): Its review issued a tombstone.

    Returns:
        DispositionV1: Its code, existing primary, memory, targets and verdict.
    """
    anchor = resolution.anchors[0] if resolution.anchors else None
    memory = resolution.memory
    judgment = None if judged is None else judged.judgment
    return DispositionV1(
        resolution.code, None if anchor is None or anchor in keys else anchor,
        memory_id=None if memory is None else memory.memory_id, targets=resolution.targets,
        judge_result_ref=None if judgment is None else judgment.result_ref,
        hold_evidence=hold_evidence,
    )


@dataclass(frozen=True, slots=True)
class RoundCommitInputsV1:
    """What `commit_round` takes for one owner round, built after the verdicts.

    Attributes:
        plan: The resolver's plan over the round's members that do not expire.
        rejection: The rejection transaction: rejected reviewed members, the touches
            of quiet and unreviewed re-entered members, and expiries.
        members: Every component member's commit input, by position.
        mints: The round's get-or-create keys.
    """

    plan: ResolutionPlanV1
    rejection: tuple[RejectionEntryV1, ...]
    members: dict[PositionKey, ComponentMemberV1]
    mints: dict[str, MintV1]


def round_commit_inputs(
    prepared: RoundPreparationV1, judged: Iterable[JudgedMemberV1], view: AddReadView, *,
    bind: ContentBinder, weights: Mapping[str, float],
    candidates: Mapping[PositionKey, str] | None = None,
    context: Sequence[PreparedMemberV1] = (),
    kind_bound: bool = False,
) -> RoundCommitInputsV1:
    """Turn one prepared, judged round into `commit_round`'s inputs.

    Every member that does not expire is resolved together, so that each
    sibling a member was issued is present: the reviewed members
    (``prepared.review``) with their verdicts, the others as non-writing
    settled members (`resolution_member`). Component members become
    `ComponentMemberV1`, the rest rejection entries with their disposition.
    Re-entered members that were not reviewed (quiet, unreviewed) write only
    their touch and keep their disposition, expired ones only their expiry.

    Args:
        prepared (RoundPreparationV1): `prepare_round`'s result, at ``view``.
        judged (Iterable[JudgedMemberV1]): `JudgeStage.review(prepared.review, round)`.
        view (AddReadView): The round's pin.
        bind (ContentBinder): The workspace's content binder.
        weights (Mapping[str, float]): the importance weights of the owner's policy.
        candidates (Mapping[PositionKey, str] | None): The stored candidate row of
            each re-entered member, by position.
        context: Earlier pending source requests, as non-writing sibling context
            only. They receive no review, disposition, touch or component here.
        kind_bound (bool): The owner's policy is the ``agent`` Add mode's: a state and an
            event sibling are never ``same`` or a ``supplement`` of each other
            (``resolve``'s ``kinds``), and a member that replaces a head's value with one new
            value is not refused as compound for that (``resolve``'s ``replaced``).

    Returns:
        RoundCommitInputsV1: The plan, rejection entries, component members and mints.

    Raises:
        QuoteMemError: ``binding_mismatch`` when a reviewed member has no outcome, an
            outcome names no reviewed member, or a re-entered member has no row.
    """
    rows = dict(candidates or {})
    context_keys = {item.position_key for item in context}
    if len(context_keys) != len(context) or context_keys & {
        item.position_key for item in prepared.members
    } or any(item.review is not None or item.pending is None or item.action != "quiet"
             or item.touch is not None or item.expire for item in context):
        raise QuoteMemError("binding_mismatch", "Sibling context is disjoint and non-writing")
    outcomes = {item.position_key: item for item in judged}
    if set(outcomes) != {item.position_key for item in prepared.members
                         if item.review is not None}:
        raise QuoteMemError("binding_mismatch", "Every reviewed member has one judge outcome")
    for item in prepared.members:
        if item.pending is not None and item.position_key not in rows:
            raise QuoteMemError("binding_mismatch", "A re-entered member names its candidate row")
    # Context has no review or snapshot of its own: only current members were
    # actually issued its S handle. Its settled input binds that position and
    # source order, without hypothetical targets that would require more quiet
    # context. Current members retain every actual issued target unchanged.
    context_members = [resolution_member(item, None, view, bind=bind).model_copy(
        update={"targets": ()},
    ) for item in context]
    members = [
        *context_members,
        *(resolution_member(item, outcomes.get(item.position_key), view, bind=bind)
          for item in prepared.members if not item.expire),
    ]
    plan = resolve(members, prepared.round,
                   kinds=_kinds((*context, *prepared.members)) if kind_bound else None,
                   replaced=_replaced(prepared.members) if kind_bound else None,
                   requested=_told_with_request(members, view) if kind_bound else frozenset())
    if any(context_keys & set(component) for component in plan.components) or any(
        request.position_key in context_keys for request in plan.round2
    ):
        raise QuoteMemError("binding_mismatch", "A pending sibling context cannot acquire a write")
    plan = plan.model_copy(update={
        "members": tuple(item for item in plan.members if item.position_key not in context_keys),
    })
    keys = mints((*context, *prepared.members))
    resolutions = {item.position_key: item for item in plan.members}
    in_component = {key for component in plan.components for key in component}
    novelty = NoveltyRows(view)
    members = {
        key: component_member(prepared.member(key), resolutions[key], outcomes[key], novelty,
                              bind=bind, weights=weights, keys=keys, candidate_id=rows.get(key))
        for key in sorted(in_component)
    }
    rejection: list[RejectionEntryV1] = []
    for item in prepared.members:
        key = item.position_key
        if key in in_component:
            continue
        if item.expire or (item.pending is not None and item.review is None):
            rejection.append(rejection_entry(item, candidate_id=rows.get(key)))
        elif key in resolutions:
            outcome = outcomes.get(key)
            disposition = rejected_disposition(
                resolutions[key], outcome, keys=keys,
                hold_evidence=outcome is not None and bool(item.nomination.tombstones))
            # A restatement of a forgotten statement withholds its spans.
            rejection.append(rejection_entry(item, disposition=disposition,
                                             candidate_id=rows.get(key),
                                             withheld=resolutions[key].withheld))
    return RoundCommitInputsV1(plan, tuple(sorted(rejection, key=lambda entry: entry.order)),
                               members, keys)


# Round 2 ----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class WokenV1:
    """The round-2 wake comparison of one rebuilt re-entered member.

    Attributes:
        position_key: The member.
        round1_wake: Its wake key at the round-1 rebuild.
        round2_wake: Its wake key at the round-2 pin, with the code read at selection.
    """

    position_key: PositionKey
    round1_wake: str
    round2_wake: str

    @property
    def woken(self) -> bool:
        """Whether it gets its one round-2 review."""
        return round2_review(round1_wake=self.round1_wake, round2_wake=self.round2_wake)


def woken(
    round1: RoundPreparationV1, codes: Mapping[PositionKey, ReasonCode], view: AddReadView, *,
    reader: LogicalReader, policy: NominationPolicyV1, bind: ContentBinder, owner_run_id: str,
    labels: LabelView, owner_committed: bool, self_identity: bool = False,
    relearn: bool = False,
) -> tuple[WokenV1, ...]:
    """Compare every rebuilt re-entered member's wake key at the round-2 pin.

    A re-entered member that stays pending after round 1 is rebuilt when
    `needs_round2_rebuild` says so, and its wake key is recomputed with the code
    read at selection. Quiet members are rebuilt like reviewed ones. Expired
    members never are.

    Args:
        round1 (RoundPreparationV1): The round-1 preparation.
        codes (Mapping[PositionKey, ReasonCode]): Each member's code after round
            1: the committed codes, else its pending code.
        view (AddReadView): The round-2 pin, after the round-1 commit.
        reader (LogicalReader): The artifact reader.
        policy (NominationPolicyV1): The frozen nomination policy.
        bind (ContentBinder): The workspace's keyed binder.
        owner_run_id (str): The owner.
        labels (LabelView): The owner's frozen turn labels.
        owner_committed (bool): Whether the owner committed a component in round 1.
        self_identity (bool): The owner's ``user-role-v1`` rule, as round 1 applied it.
        relearn (bool): Whether the owner's policy learns again what the user restates, as round 1
            applied it.

    Returns:
        tuple[WokenV1, ...]: One comparison per rebuilt member, in canonical order.
    """
    found: list[WokenV1] = []
    index: NominationIndex | None = None
    for prepared in round1.members:
        pending = prepared.pending
        if pending is None or prepared.expire or prepared.wake is None:
            continue
        code = codes.get(prepared.position_key, pending.reason_code)
        if not needs_round2_rebuild(prepared.loaded.member, code, owner_committed=owner_committed):
            continue
        rebuilt, = load_members([prepared.loaded.proposal_ref], view, reader=reader, round=2)
        # Its round-1 siblings that still precede it: a dropped group can lower σ.
        order = (rebuilt.member.sigma, rebuilt.member.position_key)
        rebuilt = replace(rebuilt, member=replace(
            rebuilt.member, siblings=tuple(
                sibling for sibling in prepared.loaded.member.siblings
                if (sibling.sigma, sibling.position_key) < order),
            declarations=prepared.loaded.member.declarations))
        if index is None:
            index = NominationIndex(view, relearn=relearn)
        again = prepare_member(rebuilt, index, policy=policy, bind=bind,
                               owner_run_id=owner_run_id, labels=labels, pending=pending,
                               self_identity=self_identity)
        if again.expire or again.wake is None:
            continue
        found.append(WokenV1(prepared.position_key, prepared.wake, again.wake))
    return tuple(found)


_CAUSE_PRIORITY: Final[tuple[Round2Cause, ...]] = (
    *get_args(Round2Reason), *CAUSE_ORDER, "reentry_woken",
)
"""Of several round-2 causes, a member keeps the first in this order."""


def round2_entries(committed: Sequence[Round2EntryV1],
                   comparisons: Iterable[WokenV1]) -> tuple[Round2EntryV1, ...]:
    """Merge the round-2 entries with the woken members, one entry per position.

    A position with more than one cause, such as a fence cause and a changed
    wake key, enters round 2 once with one cause: ``repick``,
    ``inheritance_gap``, the fence causes in their order, then
    ``reentry_woken``.
    """
    found: dict[PositionKey, Round2Cause] = {}
    for entry in (*committed, *(Round2EntryV1(position_key=item.position_key,
                                              cause="reentry_woken")
                                for item in comparisons if item.woken)):
        known = found.get(entry.position_key)
        if known is None or _CAUSE_PRIORITY.index(entry.cause) < _CAUSE_PRIORITY.index(known):
            found[entry.position_key] = entry.cause
    return tuple(Round2EntryV1(position_key=key, cause=cause)
                 for key, cause in sorted(found.items()))


@dataclass(frozen=True, slots=True)
class Round2PlanV1:
    """The round-2 members and their round-1 context.

    Attributes:
        members: The round-2 members, rebuilt at the round-2 pin.
        pending: The selected rows of the re-entered round-2 members.
        reason_codes: A woken member's wake code: the code read at selection.
        round1_touch: Each re-entered member's round-1 touch, rewritten after
            its round-2 review (`ReentryTouchV1.rewrite`).
        declarations: Round 1's new-entity declarations by ``mint_key``: a
            round-1 re-pick may name one that no commit minted and whose member
            is not in round 2 (`with_picks`).
    """

    members: tuple[LoadedMemberV1, ...]
    pending: dict[PositionKey, PendingCandidateV1] = field(
        default_factory=dict[PositionKey, PendingCandidateV1])
    round1_touch: dict[PositionKey, ReentryTouchV1] = field(
        default_factory=dict[PositionKey, ReentryTouchV1])
    declarations: dict[str, DeclarationV1] = field(default_factory=dict[str, DeclarationV1])


def round2_members(
    round1: RoundPreparationV1, entries: Sequence[Round2EntryV1], view: AddReadView, *,
    reader: LogicalReader, round: Round = 2,
) -> Round2PlanV1:
    """Rebuild a later round's members at its pin.

    Every entry's member is rebuilt for ``round``: repicks, inheritance gaps,
    fence causes and woken re-entered members alike. A same-owner
    ``review_unavailable`` member keeps its round-2 eligibility; the
    judge stage applies the limit. A settle round (``core/review_rounds``)
    rebuilds the members the previous round's fence sent on from that round's
    preparation, whose touches already carry the previous review.

    Args:
        round1 (RoundPreparationV1): The previous round's preparation.
        entries (Sequence[Round2EntryV1]): `round2_entries`' merged list, or the
            entries the previous round's fence sent on.
        view (AddReadView): The round's pin.
        reader (LogicalReader): The artifact reader.
        round (Round): The round rebuilt: 2 or a settle round.

    Returns:
        Round2PlanV1: The round-2 members, with the re-entered ones' rows and
            touches and round 1's declarations.
    """
    keys = {entry.position_key for entry in entries}
    chosen = [member for member in round1.members if member.position_key in keys]
    if len(chosen) != len(keys):
        raise QuoteMemError("binding_mismatch", "A round-2 entry names a round-1 member")
    members = load_members([member.loaded.proposal_ref for member in chosen], view,
                           reader=reader, round=round)
    pending = {member.position_key: member.pending for member in chosen
               if member.pending is not None}
    touches = {member.position_key: member.touch for member in chosen
               if member.touch is not None}
    declarations = {item.mint_key: item for member in round1.members
                    for item in member.loaded.member.declarations}
    return Round2PlanV1(members, pending, touches, declarations)


def round2_touch(plan: Round2PlanV1, prepared: PreparedMemberV1) -> ReentryTouchV1 | None:
    """The round-2 rejection touch of a reviewed re-entered member: both digests rewritten."""
    first = plan.round1_touch.get(prepared.position_key)
    if first is None or prepared.input is None or prepared.review is None:
        return first
    wake = prepared.wake if prepared.wake is not None else first.last_wake_digest
    return first.rewrite(wake_digest=wake, reviewed_input=prepared.input.digest)
