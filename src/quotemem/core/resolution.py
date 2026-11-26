"""Pure resolution of judged members into writes and dispositions.

The judge chooses one relation per issued target; resolution executes those
choices and never redecides meaning. For the members of one owner and
round, taken in canonical order ``(σ, position_key)``, ``resolve`` applies:

1. the verdict gates on each member's own verdict;
2. the same-Add sibling dispositions: duplicates, supersession through the
   inheritance table, and contradictions;
3. for each member still writing: direction, correction of history,
   targetless corrections and the write forms;
4. one transaction per component: a ``contradicts`` link to the memory
   the member's own write writes or ends is void and joins no component, and
   a write that would give a memory a second new head in its component, or a
   new head beside an end, is unresolved ``relation_unclear``. Writes of one
   memory in different components commit in canonical order, each behind the
   commit fence, which follows the earlier write's version.

The plan holds one final disposition per position, the round-2 requests, and
the components the commit writes atomically. It is a pure function of its
inputs, with no clock, no randomness and only ordered iteration, so replay
recomputes exactly the plan the original commit followed. Inputs carry
handles, ids, spans and σ, never statement or quote text, and the verdict's
free-text ``reason`` is never read.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence, Set
from dataclasses import dataclass, field
from decimal import ROUND_HALF_EVEN, Decimal
from typing import Annotated, Final, Literal, Self

from pydantic import Field, model_validator

from quotemem.core import entities
from quotemem.core.add_read_view import Sigma
from quotemem.core.entities import MemoryKind, QuoteSpanV1, ReasonCode
from quotemem.core.errors import QuoteMemError
from quotemem.core.judge_output import JudgeV8, Relation
from quotemem.core.outcome_table import OUTCOME_TABLE, PositionKey
from quotemem.core.review_rounds import RoundNo
from quotemem.store.metadata import StrictModel

EntityKey = str
"""An anchor's identity, compared for equality only: an entity id (``"user"`` or a ULID),
or for an entity the commit will get-or-create a caller-chosen key that names
it, such as its ``mint_key``."""

ChangeRelation = Literal["update", "correct", "retract"]
WriteForm = Literal["create", "merge", "revise", "annotate"]
WriteReason = Literal[
    "independent", "contradiction", "external_correction", "update", "correction",
    "retraction", "same_support", "supplement", "historical_correction",
]
Round2Reason = Literal["repick", "inheritance_gap"]
"""Why a member needs its one re-review: a re-picked entity, or a
sibling's target that the inheritance table cannot compose and the member was
not issued (``?``)."""
Round2Cause = Literal[
    Round2Reason, "fence_1", "fence_2", "fence_3", "fence_4", "fence_5", "fence_6",
    "reentry_woken",
]
"""The store's full set of round-2 admission causes."""

MAX_GROUPS = 8
"""Support groups per memory (``support_limit``)."""
MAX_SPANS = 256
"""Distinct located spans per memory."""

_SUPERSEDE: frozenset[str] = frozenset({"update", "correct", "retract", "supplement"})
_KIND_BOUND: frozenset[str] = frozenset({"same", "supplement", "update", "correct", "retract"})
"""The relations ``resolve``'s ``kinds`` may turn into ``independent`` between a state and an
event: ``same`` and ``supplement`` always, a change only between two quoting one message."""
_HISTORY: frozenset[str] = frozenset({"correct", "retract"})
_ENDS: dict[str, ChangeRelation] = {"update": "update", "correct": "correct", "retract": "retract"}
_REASONS: dict[str, WriteReason] = {
    "update": "update", "correct": "correction", "retract": "retraction",
}
_CODES: dict[str, ReasonCode] = {
    "update": "updated", "correct": "corrected", "retract": "retracted",
}
_GAP = "?"
_COLUMNS = ("same", "supplement", "update", "correct", "retract", "contradicts")
_INHERIT: dict[str, tuple[str, ...]] = {
    # L→S \ S→T:  same          supplement    update    correct    retract contradicts
    "update": ("update", "update", "update", "correct", _GAP, _GAP),
    "correct": ("correct", "correct", "update", "correct", _GAP, _GAP),
    "supplement": ("supplement", "supplement", "update", "correct", _GAP, _GAP),
    "retract": ("retract", "retract", _GAP, _GAP, _GAP, _GAP),
}
"""The inheritance table; an S→T ``independent`` inherits nothing."""

type _Span = tuple[str, int, int]
type _Group = frozenset[_Span]
type _Key = tuple[str, str]


class QuoteV1(StrictModel):
    """One located quote of the member (handle C).

    Attributes:
        handle: The quote's C handle as issued to the judge.
        span: Its location.
        sigma: Its σ, which the change test compares when it is the change span.
        user_message: Whether the cited Evidence is a ``user_message``.
    """

    handle: str
    span: QuoteSpanV1
    sigma: Sigma
    user_message: bool


class CandidateV1(StrictModel):
    """One entity candidate of a judged slot: its E or N handle and the entity it names."""

    handle: str
    entity: EntityKey


class SlotV1(StrictModel):
    """One anchor slot at the round's pin (``MemberV1.anchors``, ``entities[]``).

    Attributes:
        slot: Slot number; slot 0 is the primary anchor.
        binding: ``self``, ``participant`` and ``sdk`` are bound without review;
            ``nominated`` and ``open`` are judged.
        entity: The bound or nominated entity; None for an ``open`` slot.
        nominated: The nominated entity's handle (E or N) for a ``nominated`` slot.
        candidates: Every candidate issued for a judged slot, the nominated one
            included; empty for a bound slot.
    """

    slot: Annotated[int, Field(ge=0, le=7)]
    binding: Literal["self", "participant", "sdk", "nominated", "open"]
    entity: EntityKey | None
    nominated: str | None
    candidates: tuple[CandidateV1, ...]

    @model_validator(mode="after")
    def bound(self) -> Self:
        """Check that the binding, the entity and the candidates agree."""
        handles = [candidate.handle for candidate in self.candidates]
        if len(set(handles)) != len(handles):
            raise ValueError("A slot lists each candidate once")
        if self.binding == "nominated":
            chosen = [c.entity for c in self.candidates if c.handle == self.nominated]
            if self.entity is None or chosen != [self.entity]:
                raise ValueError("A nominated slot's entity is its nominated candidate")
        elif self.binding == "open":
            if self.entity is not None or self.nominated is not None:
                raise ValueError("An open slot has no entity yet")
        elif self.entity is None or self.nominated is not None or self.candidates:
            raise ValueError("A bound slot names its entity and has no candidates")
        return self


class VersionV1(StrictModel):
    """A retired version whose support the restatement test reads.

    Attributes:
        memory_id: The memory the version belongs to.
        revision_id: The version.
        handle: Its H handle when it was issued to the member as a history entry,
            else None.
        spans: Every span of every support group of the version.
    """

    memory_id: str
    revision_id: str
    handle: str | None
    spans: tuple[QuoteSpanV1, ...]


class HeadTargetV1(StrictModel):
    """A current head issued to the member (handle M).

    Attributes:
        kind: Always ``head``.
        handle: Its M handle.
        memory_id: The head's memory.
        revision_id: The head revision at the pin.
        sigma: σ of the head (σT).
        dated: Whether σT's time is a stated source time; False when it is the receipt
            of an undated message, which bounds only from above when it was said
            (`_older`).
        primary: The head's primary anchor (``item.entity_id``).
        naming: The entities its naming annotations name; non-empty makes it a
            naming head.
        groups: Its support groups, which a merge extends.
        retired: Its retired versions at the pin, each issued history entry
            carrying its handle.
        aliases: The entities of ``naming`` it names only as ``alias``; the
            others it gives a name, which an alias never changes.
        period: Its ``valid_from`` and ``valid_to``; both None when untimed.
    """

    kind: Literal["head"]
    handle: str
    memory_id: str
    revision_id: str
    sigma: Sigma
    primary: EntityKey
    naming: tuple[EntityKey, ...]
    groups: tuple[tuple[QuoteSpanV1, ...], ...]
    retired: tuple[VersionV1, ...]
    aliases: tuple[EntityKey, ...] = ()
    period: tuple[int | None, int | None] = (None, None)
    dated: bool = True

    @model_validator(mode="after")
    def aliases_named(self) -> Self:
        """An alias-only entity is one the head names."""
        if not set(self.aliases) <= set(self.naming):
            raise ValueError("A head's alias-only entities are among the entities it names")
        return self


class RetiredTargetV1(StrictModel):
    """A retired version issued as a target to a history correction (handle H).

    Attributes:
        kind: Always ``retired``.
        handle: Its H handle.
        memory_id: The memory the version belongs to.
        revision_id: The version a ``corrects_history`` link would name.
        sigma: σ of the version (σT in the change test).
        retired: The versions the restatement test reads: this version
            itself and the retired versions before it.
        dated: As `HeadTargetV1.dated`.
    """

    kind: Literal["retired"]
    handle: str
    memory_id: str
    revision_id: str
    sigma: Sigma
    retired: tuple[VersionV1, ...]
    dated: bool = True


class SiblingTargetV1(StrictModel):
    """An earlier member of the same owner and round issued as a target (handle S)."""

    kind: Literal["sibling"]
    handle: str
    position_key: PositionKey


type _Committed = HeadTargetV1 | RetiredTargetV1


UNDATED_SLACK_MS: Final = 86_400_000
"""How long before an undated target's receipt a change may be stated and still be ordered
by receipt (`_older`): a day covers a client that stamps the day's midnight in any time
zone, or cuts a time to the minute."""


def _older(sigma: Sigma, target: _Committed) -> bool:
    """Whether a change at σ is older than a target (the direction check).

    By σ when the target's time was stated. An undated target's time is its receipt, which
    says only that it was said by then: a change stated a little earlier than that receipt
    may still be the newer one (a client that stamps the day's midnight, or a time cut to
    the minute), so within `UNDATED_SLACK_MS` of it only the receipt order counts, as
    between two undated messages. A change stated well before it (an imported old message)
    keeps σ.
    """
    if target.dated or sigma[0] < target.sigma[0] - UNDATED_SLACK_MS:
        return sigma < target.sigma
    return sigma[1:] < target.sigma[1:]
TargetV1 = Annotated[
    HeadTargetV1 | RetiredTargetV1 | SiblingTargetV1, Field(discriminator="kind"),
]


def _key(target: HeadTargetV1 | RetiredTargetV1 | SiblingTargetV1) -> _Key:
    """A target's object identity: a head's memory, a retired version, a sibling's position."""
    if isinstance(target, SiblingTargetV1):
        return ("sibling", ".".join(map(str, target.position_key)))
    return ("head", target.memory_id) if isinstance(target, HeadTargetV1) else (
        "retired", target.revision_id
    )


class ResolutionMemberV1(StrictModel):
    """One member of a round with its accepted verdict, as resolution reads it.

    Built from the member's ``ProposalV2``, its ``NeighborSnapshotV1``, its
    accepted ``JudgeV8`` and immutable rows read at the round's pin, so replay
    rebuilds it exactly.

    Attributes:
        position_key: The position.
        slots: Its anchors at the pin, slot 0 first.
        naming: The slots its naming annotations name.
        quotes: Every located quote (handle C); the member's σ is their maximum.
        groups: Its admitted support groups as C handles, in member order; the
            verdict's ``group_support`` judges them by index.
        correction_requested: ``revises`` or ``corrects_text`` is set.
        history_correction_requested: The member may be issued retired versions.
        external_only_allowed: ``correction_requested`` and a complete primary
            partition.
        targets: Every issued target in snapshot order: heads, retired versions
            and siblings.
        verdict: The accepted verdict, fresh or borrowed.
        settled: A disposition decided outside resolution, which resolution
            keeps: ``review_unavailable``, ``review_capacity_exceeded``,
            ``identity_unresolved`` for an open slot without candidates,
            and the commit's refusals when it re-resolves a component
            without some members. A settled member never writes.
        aliases: The slots of ``naming`` it names only as ``alias``: it adds
            a name to them and changes no name a head gives them.
        period: Its resolved ``valid_from`` and ``valid_to``; both None when untimed.
        revised: The current heads its ``revises`` names that were issued to it,
            by memory id: the memories its correction says it changes.
        settle: A re-entered ``correction_target_unresolved`` member whose wait is
            over (``SETTLE_AFTER``): no target it could change arrived while
            later owners re-entered it.
    """

    position_key: PositionKey
    slots: tuple[SlotV1, ...]
    naming: tuple[int, ...]
    quotes: tuple[QuoteV1, ...]
    groups: tuple[tuple[str, ...], ...]
    correction_requested: bool
    history_correction_requested: bool
    external_only_allowed: bool
    targets: tuple[TargetV1, ...]
    verdict: JudgeV8 | None
    settled: ReasonCode | None
    aliases: tuple[int, ...] = ()
    period: tuple[int | None, int | None] = (None, None)
    revised: tuple[str, ...] = ()
    settle: bool = False

    @model_validator(mode="after")
    def consistent(self) -> Self:
        """Check slots, quotes, groups, targets and the source of the disposition."""
        numbers = [slot.slot for slot in self.slots]
        if not numbers or numbers != sorted(set(numbers)) or numbers[0] != 0:
            raise ValueError("Slots are ascending and distinct, with the primary at slot 0")
        if len(set(self.naming)) != len(self.naming) or not set(self.naming) <= set(numbers):
            raise ValueError("Naming annotations name distinct anchor slots")
        if len(set(self.aliases)) != len(self.aliases) or not set(self.aliases) <= set(
                self.naming):
            raise ValueError("Alias-only slots are distinct slots the member names")
        quotes = [quote.handle for quote in self.quotes]
        if len(set(quotes)) != len(quotes):
            raise ValueError("Quote handles are distinct")
        if not 1 <= len(self.groups) <= MAX_GROUPS or any(
            not group or len(set(group)) != len(group) or not set(group) <= set(quotes)
            for group in self.groups
        ):
            raise ValueError("Groups list distinct handles of the member's own quotes")
        handles = [target.handle for target in self.targets]
        keys = [_key(target) for target in self.targets]
        if len(set(handles)) != len(handles) or len(set(keys)) != len(keys):
            raise ValueError("Each target is issued once")
        if not self.history_correction_requested and any(
            isinstance(target, RetiredTargetV1) or isinstance(target, HeadTargetV1) and any(
                version.handle is not None for version in target.retired
            ) for target in self.targets
        ):
            raise ValueError(
                "Only a history correction is issued retired versions or history entries",
            )
        if (self.external_only_allowed or self.history_correction_requested or self.revised
                or self.settle) and not self.correction_requested:
            raise ValueError(
                "External-only, history, revising and settled corrections are requested "
                "corrections",
            )
        if not set(self.revised) <= {
            target.memory_id for target in self.targets if isinstance(target, HeadTargetV1)
        }:
            raise ValueError("A revised head is an issued head")
        if self.verdict is None and self.settled is None:
            raise ValueError("A member has an accepted verdict or a settled disposition")
        if self.settled is not None and OUTCOME_TABLE[self.settled].category not in (
            "rejected", "unresolved",
        ):
            raise ValueError("A settled disposition is a rejection or unresolved")
        return self


class MemoryRefV1(StrictModel):
    """A memory: an existing memory, or the memory a position's write creates.

    Exactly one field is set; the commit binds ``created_by`` to the minted id.
    """

    memory_id: str | None
    created_by: PositionKey | None

    @model_validator(mode="after")
    def one(self) -> Self:
        """Require exactly one reference."""
        if (self.memory_id is None) == (self.created_by is None):
            raise ValueError("A memory reference names an Item or a creating position")
        return self


class EndedTargetV1(StrictModel):
    """A head the write ends, with the ``supersedes`` relation that ends it."""

    memory_id: str
    relation: ChangeRelation


class WriteV1(StrictModel):
    """One member's formal write.

    Attributes:
        form: ``create`` a new memory; ``merge`` support into ``target``; ``revise``
            ``target`` with the member's statement; ``annotate`` a retired version
            with a ``historical`` memory.
        reason: The revision reason (``revision_reason``).
        target: The head merged into or revised; None for create and annotate.
        groups: The groups written: every group of the new revision for create,
            revise and annotate, and only the groups a merge adds.
        ended: Heads the write ends, by memory id.
        contradicts: ``contradicts`` links, to heads or to sibling writes.
        corrects: Revision ids of the corrected versions (annotate only).
        fence: Memory ids of every head in C ∪ D ∪ S ∪ X, fenced as one unit.
    """

    form: WriteForm
    reason: WriteReason
    target: str | None
    groups: tuple[tuple[QuoteSpanV1, ...], ...]
    ended: tuple[EndedTargetV1, ...]
    contradicts: tuple[MemoryRefV1, ...]
    corrects: tuple[str, ...]
    fence: tuple[str, ...]


class MemberResolutionV1(StrictModel):
    """One position's final disposition in this round.

    Attributes:
        position_key: The position.
        code: Its single reason code.
        anchors: Per slot, in slot order, the resolved entity, or None where the
            slot is unresolved.
        write: The formal write; set exactly for a formal code.
        memory: The memory add-items reports: the written one or the one already
            stating the fact; None when there is none.
        targets: Add-items target ids: ended and contradicted heads of a formal
            write, or the judged targets of an unresolved member.
        withheld: Spans the rejection transaction withholds as
            ``restated_forgotten`` (``denied_by_forget``): the located spans of
            the groups the judge found supported.
    """

    position_key: PositionKey
    code: ReasonCode
    anchors: tuple[EntityKey | None, ...]
    write: WriteV1 | None
    memory: MemoryRefV1 | None
    targets: tuple[str, ...]
    withheld: tuple[QuoteSpanV1, ...]


class PickV1(StrictModel):
    """A slot the judge re-picked; round 2 nominates the pick."""

    slot: Annotated[int, Field(ge=0, le=7)]
    handle: str
    entity: EntityKey


class IssueV1(StrictModel):
    """A committed target round 2 must issue to the member (``?``)."""

    kind: Literal["head", "retired"]
    memory_id: str
    revision_id: str


class Round2RequestV1(StrictModel):
    """A member's request for its single re-review at a fresh pin."""

    position_key: PositionKey
    reason: Round2Reason
    picks: tuple[PickV1, ...]
    issue: tuple[IssueV1, ...]


class Round2EntryV1(StrictModel):
    """Content-free store entry for a member admitted to round 2.

    Attributes:
        position_key: The admitted member's position.
        cause: The resolver, commit-fence or reentry cause of admission.
    """

    position_key: entities.PositionKey
    cause: Round2Cause


class ResolutionPlanV1(StrictModel):
    """The resolution of one owner's round.

    Every input position appears exactly once, in ``members`` or in ``round2``.

    Attributes:
        round: The round resolved.
        members: Final dispositions in canonical order.
        round2: Re-review requests in canonical order; always empty in round 2.
        components: Positions written together, each in canonical order, ordered
            by their first position's canonical key, the order they commit in. A
            component holds at least one formal write and the members whose
            disposition depends on it (``duplicate_in_add``, ``superseded_in_add``),
            and gives a memory at most one new head, never beside an end. Every
            member outside a component belongs to the round's rejection transaction.
    """

    round: RoundNo
    members: tuple[MemberResolutionV1, ...]
    round2: tuple[Round2RequestV1, ...]
    components: tuple[tuple[PositionKey, ...], ...]


def _span(span: QuoteSpanV1) -> _Span:
    return (span.evidence_id, span.byte_start, span.byte_end)


def _spans_out(spans: Iterable[_Span]) -> tuple[QuoteSpanV1, ...]:
    return tuple(QuoteSpanV1(evidence_id=evidence, byte_start=start, byte_end=end)
                 for evidence, start, end in sorted(spans))


def _fits(groups: Sequence[_Group]) -> bool:
    return len(groups) <= MAX_GROUPS and len(frozenset[_Span]().union(*groups)) <= MAX_SPANS


def _effective(relation: str, target: _Committed) -> bool:
    """A relation with an effect: not ``independent``, and on a retired version only
    ``correct`` or ``retract``."""
    return relation != "independent" and (
        isinstance(target, HeadTargetV1) or relation in _HISTORY
    )


def _containing(groups: Sequence[_Group], versions: Sequence[VersionV1]) -> VersionV1 | None:
    """Return the version holding the first group when every group lies in a version.

    Each span of a group must lie inside one span of a single version;
    different groups may lie in different versions.
    """
    found: VersionV1 | None = None
    for group in groups:
        holder = next((version for version in versions if all(
            any(
                evidence == outer.evidence_id and outer.byte_start <= start
                and end <= outer.byte_end
                for outer in version.spans
            )
            for evidence, start, end in group
        )), None)
        if holder is None:
            return None
        found = found or holder
    return found


def _pick(targets: Iterable[HeadTargetV1], primary: EntityKey | None) -> HeadTargetV1:
    """``pick``: the head sharing the member's primary anchor first, then the earliest id."""
    return min(targets, key=lambda target: (target.primary != primary, target.memory_id))


def _ref_key(ref: MemoryRefV1) -> tuple[int, str, PositionKey]:
    if ref.memory_id is not None:
        return (0, ref.memory_id, ("", 0, 0))
    return (1, "", ref.created_by or ("", 0, 0))


@dataclass(eq=False)
class _Member:
    """The working state of one member while one round resolves."""

    m: ResolutionMemberV1
    rank: int
    sigma: Sigma
    groups: tuple[_Group, ...]
    anchors: dict[int, EntityKey | None]
    relations: dict[str, Relation] = field(default_factory=dict[str, Relation])
    supported: tuple[_Group, ...] = ()
    code: ReasonCode | None = None
    round2: Round2RequestV1 | None = None
    writing: bool = False
    representative: _Member | None = None
    duplicates: list[_Member] = field(default_factory=lambda: list[_Member]())
    superseder: _Member | None = None
    joined: bool = True
    sibling_change: bool = False
    alias_added: bool = False
    """The naming gate turned a change of a naming head into ``independent``: the alias is a new
    fact."""
    other_occurrence: bool = False
    """The occurrence rule turned a supplement into ``independent``: another occurrence is a new
    fact."""
    contradicted: list[_Member] = field(default_factory=lambda: list[_Member]())
    effective: dict[_Key, tuple[str, _Committed]] = field(
        default_factory=dict[_Key, tuple[str, _Committed]],
    )
    write: WriteV1 | None = None
    memory: MemoryRefV1 | None = None
    withheld: tuple[_Span, ...] = ()

    @property
    def verdict(self) -> JudgeV8:
        if self.m.verdict is None:
            raise QuoteMemError("binding_mismatch", "A judged member lost its verdict")
        return self.m.verdict

    @property
    def primary(self) -> EntityKey | None:
        return self.anchors[0]

    def done(self, code: ReasonCode, memory: str | None = None) -> None:
        self.code, self.writing = code, False
        if memory is not None:
            self.memory = MemoryRefV1(memory_id=memory, created_by=None)


def _load(member: ResolutionMemberV1, rank: int) -> _Member:
    """Index one member and check its verdict against what it was issued."""
    quotes = {quote.handle: quote for quote in member.quotes}
    state = _Member(
        member, rank, max(quote.sigma for quote in member.quotes),
        tuple(frozenset(_span(quotes[handle].span) for handle in group)
              for group in member.groups),
        {slot.slot: slot.entity if slot.binding in ("self", "participant", "sdk") else None
         for slot in member.slots},
    )
    verdict = member.verdict
    if member.settled is not None or verdict is None:
        return state
    slots = {slot.slot: slot for slot in member.slots}
    history = {version.handle for target in member.targets if isinstance(target, HeadTargetV1)
               for version in target.retired if version.handle is not None}
    relations: dict[str, Relation] = {
        relation.target: relation.relation for relation in verdict.relations
    }
    entries = {entry.slot: entry for entry in verdict.entities}
    judged = {number for number, slot in slots.items() if slot.binding in ("nominated", "open")}
    if (
        len(relations) != len(verdict.relations)
        or set(relations) != {target.handle for target in member.targets}
        or len(verdict.group_support) != len(member.groups)
        or len(entries) != len(verdict.entities) or set(entries) != judged
        or any(
            (entry.choice is not None) != (entry.verdict == "repick")
            or entry.verdict == "confirm" and slots[slot].binding != "nominated"
            or entry.verdict == "repick" and (entry.choice == slots[slot].nominated or (
                entry.choice not in {c.handle for c in slots[slot].candidates}
            ))
            for slot, entry in entries.items()
        )
        or verdict.change_span is not None and verdict.change_span not in quotes
        or not set(verdict.restates_history) <= history
        or verdict.external_only is True and not member.external_only_allowed
    ):
        raise QuoteMemError("binding_mismatch", "A verdict names what its member was not issued")
    for slot, entry in entries.items():
        if entry.verdict == "confirm":
            state.anchors[slot] = slots[slot].entity
    state.relations = relations
    state.supported = tuple(group for group, support in zip(
        state.groups, verdict.group_support, strict=True,
    ) if support == "supported")
    return state


def _new_entity_bindings(state: _Member) -> list[tuple[EntityKey, EntityKey]]:
    """Each new entity (N handle) the member's verdict binds, with what it binds it to."""
    pairs: list[tuple[EntityKey, EntityKey]] = []
    slots = {slot.slot: slot for slot in state.m.slots}
    for entry in state.verdict.entities:
        slot = slots[entry.slot]
        if entry.verdict == "ambiguous":
            continue
        chosen = slot.entity if entry.verdict == "confirm" else next(
            candidate.entity for candidate in slot.candidates if candidate.handle == entry.choice
        )
        if chosen is None:
            continue
        if slot.nominated is not None and slot.nominated.startswith("N") and slot.entity:
            pairs.append((slot.entity, chosen))
        if entry.verdict == "repick" and (entry.choice or "").startswith("N"):
            pairs.append((chosen, chosen))
    return pairs


def _named(state: _Member) -> set[EntityKey]:
    """The entities the member's naming annotations name."""
    return {anchor for slot in state.m.naming if (anchor := state.anchors[slot]) is not None}


def _aliased(state: _Member) -> set[EntityKey]:
    """The entities the member names only as ``alias``."""
    return {anchor for slot in state.m.aliases if (anchor := state.anchors[slot]) is not None}


def _adds_alias(state: _Member, target: HeadTargetV1) -> bool:
    """The member names only as ``alias`` an entity the head gives a name, and gives
    none of those entities a name or a former name, so it adds a name beside the head's
    ("The user's friend Alicia goes by Ali now." beside "The user's friend is named
    Alicia.") and never changes it."""
    names = set(target.naming) - set(target.aliases)
    aliased = _aliased(state)
    return bool(names & aliased) and not names & (_named(state) - aliased)


def _other_occurrence(state: _Member, target: HeadTargetV1) -> bool:
    """A member's period shows it states another occurrence than the head's fact.

    A supplement keeps all the head says, its period included. Periods both
    known and disjoint are two occurrences ("Avery went to a jazz festival last year."
    against one bounded in June 2023). A period only one side knows is the same
    occurrence's only when the member quotes Evidence the head quotes: an untimed head
    given another source's year ("Avery enjoyed the jazz festival." from one session,
    about the June festival, against an earlier festival named in another) or a bounded
    head restated without its period by a later retelling (a lake trip of one week) is
    another fact.
    """
    mine, theirs = state.m.period, target.period
    if mine == theirs:
        return False
    if mine != (None, None) and theirs != (None, None):
        start = max(-float("inf") if bound is None else bound for bound in (mine[0], theirs[0]))
        end = min(float("inf") if bound is None else bound for bound in (mine[1], theirs[1]))
        return start >= end
    quoted = {quote.span.evidence_id for quote in state.m.quotes}
    return not quoted & {span.evidence_id for group in target.groups for span in group}


def _violates_naming(state: _Member, relation: str, target: HeadTargetV1) -> bool:
    """A naming head ends or changes only through a naming fact of its entity."""
    if not target.naming:
        return False
    named = set(target.naming)
    if relation == "retract":
        return state.primary not in named
    if relation in ("update", "correct", "supplement"):
        return not named & _named(state)
    return False


class _Round:
    """One owner's round under resolution: the members in canonical order and their state."""

    def __init__(self, members: Sequence[ResolutionMemberV1], round_no: RoundNo,
                 kinds: Mapping[PositionKey | str, MemoryKind] | None = None,
                 replaced: Mapping[PositionKey, frozenset[str]] | None = None,
                 requested: Set[PositionKey] = frozenset()) -> None:
        self.kinds = kinds
        self.replaced = replaced or {}
        self.requested = requested
        keys = [member.position_key for member in members]
        if len(set(keys)) != len(keys):
            raise QuoteMemError("binding_mismatch", "A position appears twice in one round")
        ordered = sorted(members, key=lambda member: (
            max(quote.sigma for quote in member.quotes), member.position_key,
        ))
        self.round_no = round_no
        self.order = [_load(member, rank) for rank, member in enumerate(ordered)]
        self.at = {state.m.position_key: state for state in self.order}
        # The components of the writes admitted so far: each writer's parent rank, and
        # by root the memories its writes touch (True for a new head, False for an end).
        self.parent: dict[int, int] = {}
        self.touched: dict[int, dict[str, bool]] = {}
        for state in self.order:
            for target in state.m.targets:
                if isinstance(target, SiblingTargetV1) and (
                    target.position_key not in self.at
                    or self.at[target.position_key].rank >= state.rank
                ):
                    raise QuoteMemError(
                        "binding_mismatch", "A sibling is an earlier member of the round",
                    )

    # Gates ----------------------------------------------------------

    def gate(self, state: _Member, disputed: set[EntityKey]) -> None:
        """Apply the verdict gates in order; a member that passes them all is writing."""
        verdict = state.verdict
        if any(entry.verdict == "restates" for entry in verdict.tombstones):
            if verdict.support == "contradicted":
                return state.done("unsupported_by_source")
            # Only a group the judge found to state the member carries the forgotten
            # statement; a quote that does not state it stays readable for its siblings.
            state.withheld = tuple(sorted(frozenset[_Span]().union(*state.supported)))
            return state.done("denied_by_forget")
        # A compound verdict of one replacement counts the replaced value (``replaced``);
        # whether it still is one is checked again after the relation rewrites below.
        replacing = verdict.atomicity == "compound" and self.one_replacement(state)
        checks: tuple[tuple[bool, ReasonCode], ...] = (
            (verdict.support == "contradicted", "unsupported_by_source"),
            (verdict.atomicity == "compound" and not replacing, "not_atomic"),
            (verdict.fidelity == "incomplete", "source_fidelity_incomplete"),
            (verdict.support == "insufficient" or verdict.atomicity == "uncertain",
             "semantic_evidence_insufficient"),
            (verdict.fidelity == "uncertain", "source_fidelity_uncertain"),
            (verdict.sensitive_attribution == "inferred" or (
                verdict.sensitive_attribution == "direct" and not self.user_supported(state)
            ), "unsupported_sensitive_inference"),
            (any(entry.verdict == "ambiguous" for entry in verdict.entities) or bool(
                disputed & {key for key, _referent in _new_entity_bindings(state)}
            ), "identity_unresolved"),
        )
        for failed, code in checks:
            if failed:
                return state.done(code)
        repicks = sorted((entry for entry in verdict.entities if entry.verdict == "repick"),
                         key=lambda entry: entry.slot)
        if repicks and self.round_no >= 2:
            return state.done("identity_unresolved")
        if repicks:
            slots = {slot.slot: slot for slot in state.m.slots}
            state.round2 = Round2RequestV1(
                position_key=state.m.position_key, reason="repick", issue=(),
                picks=tuple(PickV1(slot=entry.slot, handle=candidate.handle,
                                   entity=candidate.entity)
                            for entry in repicks for candidate in slots[entry.slot].candidates
                            if candidate.handle == entry.choice),
            )
            return None
        heads = {t.handle: t for t in state.m.targets if isinstance(t, HeadTargetV1)}
        siblings = {t.handle: self.at[t.position_key] for t in state.m.targets
                    if isinstance(t, SiblingTargetV1)}
        relations = state.relations
        for handle in [handle for handle, relation in relations.items()
                       if relation == "supplement"]:
            named = (set(heads[handle].naming) if handle in heads
                     else _named(siblings[handle]) if handle in siblings else set[EntityKey]())
            if named and not named & _named(state):
                # A fact folds into a naming fact only by naming its entity, so a
                # supplement that does not is another fact (as the other-anchor one).
                relations[handle] = "independent"
        for handle in [handle for handle, relation in relations.items()
                       if relation == "supplement" and handle in heads]:
            if _other_occurrence(state, heads[handle]):
                # A supplement of another occurrence is another fact.
                relations[handle] = "independent"
                state.other_occurrence = True
        for handle in [handle for handle, relation in relations.items()
                       if relation in ("update", "correct") and handle in heads]:
            if _adds_alias(state, heads[handle]):
                # An alias adds a name and never changes or ends the name a
                # naming head gives, so the member is another fact beside the head.
                relations[handle] = "independent"
                state.alias_added = True
        if replacing and not self.one_replacement(state):
            # The alias rule above ended the update the compound verdict was read through: the
            # statement would stand beside the value it names as replaced, so it is refused.
            return state.done("not_atomic")
        if state.m.position_key in self.requested:
            # Told with a forget request (``resolve``'s ``requested``): a new fact beside the
            # stored ones, never a change of one, so the request turn corrects nothing the
            # forget left.
            for handle in [handle for handle, relation in relations.items()
                           if relation in _ENDS and handle in heads]:
                relations[handle] = "independent"
        if "unclear" in relations.values() or sum(
            relation == "supplement" for handle, relation in relations.items()
            if handle not in siblings  # sibling supplements count once voided
        ) >= 2 or any(
            _violates_naming(state, relation, heads[handle])
            for handle, relation in relations.items() if handle in heads
        ):
            return state.done("relation_unclear")
        state.writing = True
        return None

    def one_replacement(self, state: _Member) -> bool:
        """Whether a compound verdict only counts the value the member replaces (``replaced``).

        The member's one change is an ``update`` or ``correct`` of a head its statement
        replaces with one new value and that quotes none of the member's Evidence (so the
        ``kinds`` rule never ends that change), and it relates to no other target: that
        change records the replaced value, so the statement states one fact.
        """
        named = self.replaced.get(state.m.position_key, frozenset())
        heads = {target.handle: target for target in state.m.targets
                 if isinstance(target, HeadTargetV1) and target.memory_id in named}
        changed = [(handle, relation) for handle, relation in state.relations.items()
                   if relation != "independent"]
        if len(changed) != 1 or changed[0][1] not in ("update", "correct"):
            return False
        head = heads.get(changed[0][0])
        quoted = {quote.span.evidence_id for quote in state.m.quotes}
        return head is not None and not quoted & {
            span.evidence_id for group in head.groups for span in group}

    @staticmethod
    def user_supported(state: _Member) -> bool:
        """Some supported group consists only of ``user_message`` Evidence."""
        quotes = {quote.handle: quote for quote in state.m.quotes}
        return any(
            support == "supported" and all(quotes[handle].user_message for handle in group)
            for group, support in zip(state.m.groups, state.verdict.group_support, strict=True)
        )

    # Pass 1: same-Add siblings -----------------------------------------------

    def kind_bound(self, state: _Member, other: PositionKey | str, relation: str,
                   evidence: Set[str]) -> bool:
        """Whether ``resolve``'s ``kinds`` make the member's relation to ``other`` (a sibling
        position or a head's memory id quoting ``evidence``) count as ``independent``."""
        kinds = self.kinds
        if kinds is None or relation not in _KIND_BOUND:
            return False
        mine, theirs = kinds.get(state.m.position_key), kinds.get(other)
        if mine is None or theirs is None or mine == theirs:
            return False
        return relation in ("same", "supplement") or bool(
            evidence & {quote.span.evidence_id for quote in state.m.quotes})

    def siblings(self, state: _Member) -> None:
        """Settle the member's sibling relations, then compose its committed relations."""
        same: list[_Member] = []
        superseded: list[tuple[str, _Member]] = []
        for target in state.m.targets:
            if not isinstance(target, SiblingTargetV1):
                continue
            sibling, relation = self.at[target.position_key], state.relations[target.handle]
            if not sibling.writing or sibling.superseder is not None:
                continue  # a relation to a sibling that is not writing is independent
            if self.kind_bound(state, sibling.m.position_key, relation,
                               {quote.span.evidence_id for quote in sibling.m.quotes}):
                continue  # a state and an event are two memories (``resolve``'s ``kinds``)
            if relation == "same":
                same.append(sibling)
            elif relation in _SUPERSEDE:
                superseded.append((relation, sibling))
            elif relation == "contradicts":
                state.contradicted.append(sibling)
        own = {_key(target): (
            "independent" if isinstance(target, HeadTargetV1) and self.kind_bound(
                state, target.memory_id, state.relations[target.handle],
                {span.evidence_id for group in target.groups for span in group})
            else state.relations[target.handle], target)
            for target in state.m.targets if not isinstance(target, SiblingTargetV1)}
        folded = sum(relation == "supplement" for relation, _sibling in superseded)
        if folded and folded + sum(
            relation == "supplement" for relation, _target in own.values()
        ) >= 2:
            return state.done("relation_unclear")  # again over the relations left standing
        if same:
            representative = min(same, key=lambda sibling: sibling.rank)
            if superseded or any(
                _effective(relation, target) and key not in representative.effective
                for key, (relation, target) in own.items()
            ):
                return state.done("relation_unclear")
            state.representative, state.writing = representative, False
            representative.duplicates.append(state)
            return None
        inherited: dict[_Key, list[str]] = {}
        reached: dict[_Key, _Committed] = {}
        for relation, sibling in sorted(superseded, key=lambda pair: pair[1].rank):
            sibling.superseder = state
            state.sibling_change = state.sibling_change or relation in _ENDS
            for key, (through, target) in sibling.effective.items():
                if through == "supplement" and isinstance(target, HeadTargetV1) and (
                    target.primary != sibling.primary
                ):
                    continue  # counts as independent for the sibling
                inherited.setdefault(key, []).append(_INHERIT[relation][_COLUMNS.index(through)])
                reached.setdefault(key, target)
        gaps: list[_Committed] = []
        for key, (mine, target) in own.items():
            cells = inherited.pop(key, [])
            composed = cells[0] if len(set(cells)) == 1 else (_GAP if cells else None)
            chosen = mine if mine != "independent" or composed in (None, _GAP) else composed
            if _effective(chosen, target):
                state.effective[key] = (chosen, target)
        for key, cells in inherited.items():
            composed = cells[0] if len(set(cells)) == 1 else _GAP
            if composed == _GAP:
                gaps.append(reached[key])
            elif _effective(composed, reached[key]):
                state.effective[key] = (composed, reached[key])
        if gaps:
            return self.defer(state, gaps)
        return None

    def defer(self, state: _Member, gaps: list[_Committed]) -> None:
        """An inheritance ``?`` on an unissued target: round 2 with it issued, else incomplete.

        A retired version is issuable only to a history correction.
        """
        if self.round_no >= 2 or any(
            isinstance(target, RetiredTargetV1) and not state.m.history_correction_requested
            for target in gaps
        ):
            return state.done("operation_incomplete")
        state.writing = False
        state.round2 = Round2RequestV1(
            position_key=state.m.position_key, reason="inheritance_gap", picks=(),
            issue=tuple(sorted((IssueV1(
                kind="head" if isinstance(target, HeadTargetV1) else "retired",
                memory_id=target.memory_id, revision_id=target.revision_id,
            ) for target in gaps), key=lambda issue: (issue.kind, issue.revision_id))),
        )
        return None

    # Pass 2: direction, history, targetless corrections, write forms ------------------

    def settle(self, state: _Member) -> None:
        """Decide one writing member's disposition and write."""
        heads = [(r, t) for r, t in state.effective.values() if isinstance(t, HeadTargetV1)]
        if sum(relation == "supplement" for relation, _target in heads) >= 2 or any(
            _violates_naming(state, relation, target) for relation, target in heads
        ):
            return state.done("relation_unclear")  # both gates also bind inherited relations
        changes = [(r, t) for r, t in heads if r in _ENDS]
        history = [t for _r, t in state.effective.values() if isinstance(t, RetiredTargetV1)]
        if changes and history:
            return state.done("relation_unclear")
        verdict = state.verdict
        quotes = {quote.handle: quote for quote in state.m.quotes}
        sigma_c = quotes[verdict.change_span].sigma if verdict.change_span else state.sigma
        late = [t for t in [*(t for _r, t in changes), *history] if _older(sigma_c, t)]
        for target in late:
            holder = next((version for version in target.retired if version.handle is not None
                           and version.handle in verdict.restates_history), None)
            holder = holder or _containing(state.groups, target.retired)
            if holder is not None:
                return state.done("restates_retired_version", holder.memory_id)
        if late:
            return state.done("older_than_target")
        for relation, target in heads:
            if relation == "contradicts" and _older(sigma_c, target):
                holder = _containing(state.groups, target.retired)
                if holder is not None:
                    return state.done("restates_retired_version", holder.memory_id)
        if history:
            return self.annotate(state, history)
        same = [t for r, t in heads if r == "same"]
        supplements = [t for r, t in heads if r == "supplement" and t.primary == state.primary]
        new_fact = state.alias_added or state.other_occurrence
        if state.m.correction_requested and not new_fact and not (
            changes or state.sibling_change or same or supplements
        ):
            return self.targetless(state)
        return self.form(state, changes, same, supplements)

    @staticmethod
    def join(state: _Member, base: Sequence[_Group]) -> list[_Group] | None:
        """Add the member's and its duplicates' new supported groups within bounds.

        Returns:
            The groups the write adds, or None when the member's own new groups do
            not fit (``support_budget_exceeded``). A duplicate whose new groups
            do not fit stays out of the write.
        """
        written: list[_Group] = []
        for contributor in (state, *state.duplicates):
            new = [group for group in dict.fromkeys(contributor.supported)
                   if group not in base and group not in written]
            if not new or _fits([*base, *written, *new]):
                written += new
            elif contributor is state:
                return None
            else:
                contributor.joined = False
        return written

    def annotate(self, state: _Member, corrected: list[RetiredTargetV1]) -> None:
        """A ``historical`` memory bound to each corrected version; no head changes."""
        written = self.join(state, ())
        if written is None:
            return state.done("support_budget_exceeded")
        state.write = WriteV1(
            form="annotate", reason="historical_correction", target=None,
            groups=tuple(_spans_out(group) for group in written), ended=(), contradicts=(),
            corrects=tuple(sorted(target.revision_id for target in corrected)), fence=(),
        )
        state.memory = MemoryRefV1(memory_id=None, created_by=state.m.position_key)
        return state.done("history_corrected")

    def targetless(self, state: _Member) -> None:
        """A correction with no change relation to a head, sibling or retired version.

        (1) A tombstone ``revises`` creates and links nothing. (2) ``external_only``
        creates an external correction. (2b) Otherwise a member that contradicts a head or
        a sibling write is bound to it and creates like any contradiction. (3) A
        member whose ``revises`` named an issued head, or whose wait is over (``settle``),
        creates like any independent member: the judge found that it changes none of the
        memories it was issued, the one it names among them, and no target it waits for
        arrived. (4) Else it stays unresolved until a target arrives or its wait ends.
        (2) and (2b) write the member's ``contradicts`` links and fence their heads:
        a head whose own fact stands while the member makes its description of
        an entity untrue keeps its fact, current and linked.
        """
        verdict = state.verdict
        reason: WriteReason
        code: ReasonCode
        links: list[MemoryRefV1] = []
        if any(entry.verdict == "revises" for entry in verdict.tombstones):
            reason, code = "independent", "created"
        elif verdict.external_only is True:
            reason, code = "external_correction", "external_correction"
            links = self.links(state)
        elif links := self.links(state):
            reason, code = "contradiction", "created_with_contradiction"
        elif state.m.revised or state.m.settle:
            reason, code = "independent", "created"
        else:
            return state.done("correction_target_unresolved")
        written = self.join(state, ())
        if written is None:
            return state.done("support_budget_exceeded")
        state.write = WriteV1(
            form="create", reason=reason, target=None,
            groups=tuple(_spans_out(group) for group in written), ended=(),
            contradicts=tuple(links), corrects=(),
            fence=tuple(sorted({link.memory_id for link in links if link.memory_id is not None})),
        )
        state.memory = MemoryRefV1(memory_id=None, created_by=state.m.position_key)
        return state.done(code)

    @staticmethod
    def links(state: _Member) -> list[MemoryRefV1]:
        """X: ``contradicts`` links to heads and to sibling writes, deduplicated."""
        found = [MemoryRefV1(memory_id=t.memory_id, created_by=None)
                 for r, t in state.effective.values()
                 if r == "contradicts" and isinstance(t, HeadTargetV1)]
        found += [sibling.memory for sibling in state.contradicted if sibling.memory and (
            sibling.code == "redundant" or sibling.write is not None
            and sibling.write.form != "annotate"
        )]
        return sorted({_ref_key(link): link for link in found}.values(), key=_ref_key)

    def form(
        self, state: _Member, changes: list[tuple[str, HeadTargetV1]],
        same: list[HeadTargetV1], supplements: list[HeadTargetV1],
    ) -> None:
        """Choose create, merge or revise and the heads the write ends."""
        primary = state.primary
        ended: dict[str, ChangeRelation] = {
            target.memory_id: _ENDS[relation] for relation, target in changes
        }
        links = self.links(state)
        fence = sorted({*ended, *(t.memory_id for t in same), *(t.memory_id for t in supplements),
                        *(link.memory_id for link in links if link.memory_id is not None)})
        picked = _pick((target for _r, target in changes), primary) if changes else None
        form: WriteForm
        reason: WriteReason
        code: ReasonCode
        target: str | None = None
        if same:
            merged = _pick(same, primary)
            written = self.join(state, tuple(frozenset(map(_span, group))
                                             for group in merged.groups))
            if written is None:
                return state.done("support_budget_exceeded")
            if not written and picked is None:
                return state.done("redundant", merged.memory_id)
            form, reason, target = "merge", "same_support", merged.memory_id
            code = _CODES[ended[picked.memory_id]] if picked else "merged_support"
        else:
            written = self.join(state, ())
            if written is None:
                return state.done("support_budget_exceeded")
            if supplements:
                form, reason, target = "revise", "supplement", supplements[0].memory_id
                code = _CODES[ended[picked.memory_id]] if picked else "supplemented"
            elif picked is not None:
                relation = ended[picked.memory_id]
                code, reason = _CODES[relation], _REASONS[relation]
                if relation in _HISTORY and picked.primary == primary:
                    form, target = "revise", picked.memory_id
                    del ended[picked.memory_id]
                else:
                    form = "create"
            elif links:
                form, reason, code = "create", "contradiction", "created_with_contradiction"
            else:
                form, reason, code = "create", "independent", "created"
        # A link to the memory this write writes or ends is void: a link never names its own
        # Memory, and an ended memory no longer stands beside the write.
        own = {*ended, *(() if target is None else (target,))}
        links = [link for link in links if link.memory_id not in own]
        state.write = WriteV1(
            form=form, reason=reason, target=target,
            groups=tuple(_spans_out(group) for group in written),
            ended=tuple(EndedTargetV1(memory_id=memory, relation=relation)
                        for memory, relation in sorted(ended.items())),
            contradicts=tuple(links), corrects=(), fence=tuple(fence),
        )
        state.memory = MemoryRefV1(
            memory_id=target, created_by=None if target else state.m.position_key,
        )
        return state.done(code)

    # One transaction per component ------------------------------------------------

    @staticmethod
    def joins(state: _Member, sibling: _Member) -> bool:
        """Whether a contradicted sibling's write commits in the member's transaction.

        It does when the member's write links to the memory that write states; a
        void link (`form`) joins nothing.
        """
        write = state.write
        return write is not None and sibling.write is not None and (
            sibling.write.form != "annotate" and sibling.memory in write.contradicts
        )

    def _root(self, rank: int) -> int:
        parent = self.parent
        while parent[rank] != rank:
            parent[rank] = parent[parent[rank]]
            rank = parent[rank]
        return rank

    def admit(self, state: _Member) -> None:
        """Keep the member's component one transaction the Writer accepts.

        A transaction gives a memory at most one new head (merge or revise) and never
        ends a memory it gives a new head; several ends of one memory compose. A write
        that would break this in the component it joins is unresolved
        ``relation_unclear``, as the supplement gate leaves a write that cannot follow two chains,
        and later members see it as not writing; it is re-entered against the committed
        result. Canonical order makes the earlier write the one that stands.
        """
        write = state.write
        if write is None:
            return None
        touched = {ended.memory_id: False for ended in write.ended}
        if write.form in ("merge", "revise") and write.target is not None:
            touched[write.target] = True
        roots = sorted({self._root(sibling.rank) for sibling in state.contradicted
                        if self.joins(state, sibling)})
        for root in roots:
            for memory_id, head in self.touched[root].items():
                if memory_id in touched and (head or touched[memory_id]):
                    state.write = state.memory = None
                    return state.done("relation_unclear")
                touched[memory_id] = head or touched.get(memory_id, False)
        self.parent[state.rank] = state.rank
        for root in roots:
            self.parent[root] = state.rank
            del self.touched[root]
        self.touched[state.rank] = touched
        return None

    # Attached members, components and output --------------------------------------------

    @staticmethod
    def writer(state: _Member) -> _Member | None:
        """Follow a duplicate to its representative and supersession to the member that writes."""
        current: _Member | None = state.representative or state
        while current is not None and current.write is None:
            current = current.superseder
        return current

    def attach(self) -> None:
        """Give superseded members and duplicates their dispositions."""
        for state in reversed(self.order):
            if state.superseder is not None:
                state.code = "superseded_in_add"
                state.memory = state.superseder.memory
        for state in self.order:
            representative = state.representative
            if representative is None:
                continue
            if representative.code is None:
                raise QuoteMemError("binding_mismatch",
                                    "A representative ended without a disposition")
            if not state.joined:
                state.done("support_budget_exceeded")
            elif OUTCOME_TABLE[representative.code].category in (
                "added", "corrected", "supplemented", "unchanged",
            ):
                state.code, state.memory = "duplicate_in_add", representative.memory
            else:
                state.code = representative.code

    def components(self) -> tuple[tuple[PositionKey, ...], ...]:
        """Group formal writes with the sibling writes and dispositions that depend on them."""
        root = {state.rank: state.rank for state in self.order}

        def find(rank: int) -> int:
            while root[rank] != rank:
                root[rank] = root[root[rank]]
                rank = root[rank]
            return rank

        def union(first: _Member, second: _Member) -> None:
            low, high = sorted((find(first.rank), find(second.rank)))
            root[high] = low

        grouped: set[int] = set()
        for state in self.order:
            if state.write is not None:
                grouped.add(state.rank)
                for sibling in state.contradicted:
                    if self.joins(state, sibling):
                        union(state, sibling)
            elif state.code in ("superseded_in_add", "duplicate_in_add"):
                writer = self.writer(state)
                if writer is not None:
                    grouped.add(state.rank)
                    union(state, writer)
        found: dict[int, list[PositionKey]] = {}
        for state in self.order:
            if state.rank in grouped:
                found.setdefault(find(state.rank), []).append(state.m.position_key)
        return tuple(tuple(keys) for _rank, keys in sorted(found.items()))

    @staticmethod
    def output(state: _Member) -> MemberResolutionV1:
        """Render one final disposition."""
        if state.code is None:
            raise QuoteMemError("binding_mismatch", "A position ended without a disposition")
        targets: set[str] = set()
        if state.write is not None:
            targets = {ended.memory_id for ended in state.write.ended} | {
                link.memory_id for link in state.write.contradicts if link.memory_id is not None
            }
        elif OUTCOME_TABLE[state.code].category == "unresolved" and state.m.settled is None:
            targets = {target.memory_id for target in state.m.targets
                       if not isinstance(target, SiblingTargetV1)}
        return MemberResolutionV1(
            position_key=state.m.position_key, code=state.code,
            anchors=tuple(state.anchors[slot.slot] for slot in state.m.slots),
            write=state.write, memory=state.memory, targets=tuple(sorted(targets)),
            withheld=_spans_out(state.withheld),
        )


def resolve(
    members: Sequence[ResolutionMemberV1], round_no: RoundNo, *,
    kinds: Mapping[PositionKey | str, MemoryKind] | None = None,
    replaced: Mapping[PositionKey, frozenset[str]] | None = None,
    requested: Set[PositionKey] = frozenset(),
) -> ResolutionPlanV1:
    """Resolve one owner's round into dispositions, writes and re-review requests.

    The members are put in canonical order ``(σ, position_key)`` first, so the
    plan does not depend on the order they are given in. Gates run on every
    member's own verdict; sibling dispositions run over all members before any
    write form is chosen; then direction, correction of history, targetless
    corrections and the write forms run in canonical order.

    Args:
        members (Sequence[ResolutionMemberV1]): Every member of the owner and
            round, including members whose disposition is already ``settled``,
            so that each issued sibling is present.
        round_no (RoundNo): The round resolved.
            Supported values:
            - `1`: a re-picked entity or an inheritance gap (``?``)
              asks for round 2.
            - `2`: the same needs end the position ``identity_unresolved`` or
              ``operation_incomplete``; there is no round 3.
        kinds (Mapping[PositionKey | str, MemoryKind] | None): The kind of each member (by
            position) and of each issued head (by memory id), given only under the ``agent``
            Add mode. Between a state and an event, sibling or head, a ``same`` or
            ``supplement`` relation then counts as ``independent``, so neither is folded into
            the other, and so does an ``update``, ``correct`` or ``retract`` when both quote
            one message: one sentence cannot replace what it states itself. "The user lives
            in Xiamen now." beside "The user moved from Chengdu to Xiamen last month." of the
            same message was judged a supplement, or an update, of the other and superseded
            or ended in the Add, leaving no current residence. An event of
            another message still ends a state ("moved to Berlin" ends "lives in Paris").
            None (the benchmark modes) keeps the judged relations unchanged.
        replaced (Mapping[PositionKey, frozenset[str]] | None): By position, the issued
            heads (memory ids) whose value the member's statement names as the one it
            replaces with one new value (``judge_structure.one_replacement``), given only
            under the ``agent`` Add mode. A ``compound`` verdict does not reject such a
            member ``not_atomic`` when its only change is an ``update`` or ``correct`` of
            one of them, also, and it relates to no other target: the change
            records the replaced value. "Since the move, the
            user goes climbing at a gym near West Lake instead of playing badminton." was
            judged a compound update of "The user plays badminton every Saturday morning at
            a club in Xuhui." in three smoke runs of eight and rejected, which lost the new
            sport and left the old one current. None keeps the atomicity gate unchanged.
        requested (Set[PositionKey]): The members told with a forget request (a quote meets a
            user's forget-request sentence), given only under the ``agent`` Add mode: their
            ``update``, ``correct`` and ``retract`` of a stored head count as ``independent``,
            so a request turn records a new fact and never a correction of what the forget it
            carries asked about ("Forget my old address since I moved to 27 Birch Lane."
            stores the move and leaves the address to the forget); a ``same`` or a
            ``supplement`` of a head restating the new value told before still folds into it.

    Returns:
        ResolutionPlanV1: One final disposition per position not deferred to
            round 2, the round-2 requests, and the components to commit.

    Raises:
        QuoteMemError: ``binding_mismatch`` when a position appears twice, a sibling
            target is not an earlier member of the round, or a verdict names a
            target, slot candidate, quote or history entry its member was not
            issued.
    """
    resolution = _Round(members, round_no, kinds, replaced, requested)
    referents: dict[EntityKey, set[EntityKey]] = {}
    for state in resolution.order:
        if state.m.settled is None:
            for key, referent in _new_entity_bindings(state):
                referents.setdefault(key, set()).add(referent)
    disputed = {key for key, found in referents.items() if len(found) > 1}
    for state in resolution.order:
        if state.m.settled is not None:
            state.done(state.m.settled)
        else:
            resolution.gate(state, disputed)
    for state in resolution.order:
        if state.writing:
            resolution.siblings(state)
    for state in resolution.order:
        if state.writing and state.superseder is None:
            resolution.settle(state)
            resolution.admit(state)
    resolution.attach()
    return ResolutionPlanV1(
        round=round_no,
        members=tuple(resolution.output(state) for state in resolution.order
                      if state.round2 is None),
        round2=tuple(state.round2 for state in resolution.order if state.round2 is not None),
        components=resolution.components(),
    )


def importance(
    max_cosine: float | None, *, durability: float, actionability: float,
    weights: Mapping[str, float],
) -> float:
    """Compute a new revision's importance at resolution.

    A merge keeps its head's value instead; create, revise and annotate take
    this one.

    Args:
        max_cosine (float | None): The highest cosine to a ``p_mem`` head with the
            same primary at the round's nomination pin, or None when there is no
            such head (novelty 1).
        durability (float): ``ProposalV2.importance_inputs.durability``, in [0, 1].
        actionability (float): ``ProposalV2.importance_inputs.actionability``, in
            [0, 1].
        weights (Mapping[str, float]): The extraction policy's importance weights,
            keyed ``novelty``, ``durability`` and ``actionability``.

    Returns:
        float: ``w_n·novelty + w_d·durability + w_a·actionability`` with
            ``novelty = clip(1 - max_cosine, 0, 1)``, rounded half-even to 6
            decimals.
    """
    novelty = 1.0 if max_cosine is None else min(1.0, max(0.0, 1.0 - max_cosine))
    value = (weights["novelty"] * novelty + weights["durability"] * durability
             + weights["actionability"] * actionability)
    return float(Decimal(str(value)).quantize(Decimal("0.000001"), rounding=ROUND_HALF_EVEN))
