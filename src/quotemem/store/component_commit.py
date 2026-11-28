"""An owner round's commit, and one formal component's fence and transaction.

`commit_round` commits a round's rejection transaction, then its components in
order, and returns the round's codes and content-free ``Round2EntryV1`` list:
the resolver's ``repick``/``inheritance_gap``, ``reentry_woken`` and the
fence cause of every component sent back. A component the
fence finds changed goes to the next round; only in the last round (``MAX_ROUND``,
``core/review_rounds``) does a fence finding write its members as ``target_changed``.

A component holds at least one formal write and the members whose disposition
depends on it (``ResolutionPlanV1.components``). `commit_component` runs, per
component in the order of its minimum canonical key:

Every path checks the actual finalize owner's budget with the trusted clock
(`require_owner_budget`): before the pre-lock work, and under
the Writer mutex right before each ``commit_rows`` (the component, its round-2
refusal and every rejection chunk). A refused budget raises ``delivery_busy``
and leaves the uncommitted positions without a row.

1. **Pre-lock** at the fence pin s1: `ComponentFence.read` inside the owner's
   read callback, which also reads each get-or-create entity and the expected
   head of every write target; `ComponentFence.settle` runs the would-be test.
   Any target-change finding returns the component's single cause;
   the caller sends the component to the next round.
2. **In-lock**: `recheck` over the rows after s1; then the rows are built,
   measured and committed with a CAS on every expected head and a create-once
   ``write_set:commit_journal``. A CAS mismatch is a target change (``fence_2``).

Per-item refusals inside the transaction: the tombstone guard gives
``denied_by_forget`` and withholds the member's spans; a write whose
every stored group lost a span gives ``evidence_invalidated``, and a partly
unusable write drops those groups (not a target change); a merge whose added
groups no longer fit the head it is committed against (a head re-bound after
the resolution pinned it) gives ``support_budget_exceeded``. A merge drops the
groups quoting a name a correction gave onto a memory that still states the
name it replaced (`_unrenamed`), and one left with none gives
``evidence_invalidated``. A refused creator's dependants mirror it and links to it are
dropped. A component whose rows exceed ``MAX_TX_ROWS``, by the Writer's count
or by its capacity refusal while sizing, writes every member as
``support_budget_exceeded`` instead (never split). So does a component
whose transaction a storage constraint refused (``storage_conflict``): the
rolled-back transaction wrote nothing, a replay would be refused the same way,
and the parked members stay pending for explicit re-judgment only, so one
unwritable component neither stops its owner nor quarantines its scope. A member's
duplicates are written in the same transaction, mirroring its final
disposition (``mirrored``), and journaled with it.

Rows per write form: ``create`` a new memory and version 1;
``merge`` the head's content with the added groups as vN+1, carrying
``supersedes`` and ``contradicts``; ``revise`` the member's statement as vN+1,
carrying ``supersedes``; ``annotate`` a ``historical`` memory with one
``corrects_history`` link per corrected version. Every written head
re-projects ``about_entity``; each ended target gets its lifecycle,
``retired_at`` and a ``supersedes{relation}`` link (``effective_at`` per the link time rule
for update); ``contradicts`` links are new. Names are re-derived for every
memory whose head or lifecycle changed.

An optional observer sees each component's fence as it ran, passes included
(``ComponentFenceEventV1``, for the recorded replay): the would-be test's operands,
every pre-lock finding, the in-lock recheck, the head CAS and the outcome. It
is called once per component after its last decision; it changes no decision
and no transaction.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final, Literal, cast

from pydantic import ValidationError

from quotemem.core.add_read_view import AddReadView, ReadPin
from quotemem.core.commit_plan import CommitJournalV1
from quotemem.core.descriptor_repair import HeadRepairV1
from quotemem.core.entities import (
    USER,
    MemorySnapshotV2,
    NameKind,
    PositionKey,
    QuoteSpanV1,
    ReasonCode,
)
from quotemem.core.errors import QuoteMemError
from quotemem.core.forget_match import named_ties
from quotemem.core.forgotten_names import (
    ForgottenName,
    forgotten_names,
    kind_of,
    occurrences,
    touching,
)
from quotemem.core.identity import IntegrityDigest, integrity_digest, new_ulid
from quotemem.core.replaced_names import supports, supports_cut, tie_span, unnamed
from quotemem.core.resolution import (
    MemberResolutionV1,
    MemoryRefV1,
    ResolutionPlanV1,
    Round2EntryV1,
    WriteV1,
)
from quotemem.core.review_rounds import MAX_ROUND, RoundNo
from quotemem.core.semantic import ScopeKind
from quotemem.core.statement import (
    MAX_GROUP_SPANS,
    MAX_MEMORY_SPANS,
    MAX_SUPPORT_GROUPS,
    name_norm,
    statement_digest,
)
from quotemem.ingestion.reentry import ReentryTouchV1
from quotemem.store.commit_fence import (
    ComponentFence,
    FenceCause,
    FenceMemberV1,
    FencePinV1,
    MemberFindingsV1,
    MemberWouldBeV1,
    component_cause,
    recheck,
)
from quotemem.store.entity_ops import (
    ENDED_LIFECYCLE,
    TxnRows,
    carry_links,
    confirmed_at,
    get_or_create_participant,
    get_or_mint,
    head_memory,
    link_row,
    project_about,
    rederive_entity_names,
    requote_on,
    revision_row,
    support_union,
)
from quotemem.store.forget_admission import forgotten_keywords
from quotemem.store.indexed import EntityNameRows, TombstoneRows
from quotemem.store.memory_commit import (
    Clock,
    DispositionV1,
    FollowerV1,
    GroupsOf,
    PositionV1,
    Publish,
    RejectionEntryV1,
    _journal,  # pyright: ignore[reportPrivateUsage]
    commit_rejection,
    expire_unusable,
    journaled_positions,
    mirrored,
    owner_budget,
    readability,
    withhold_spans,
    write_disposition,
)

if TYPE_CHECKING:
    from quotemem.store.writer import CommittedRows, ExpectedHead, Writer, WriteTicket

Row = dict[str, Any]
ReadAt = Callable[[Callable[[AddReadView], None]], None]
"""The owner's read: runs a callback on its Add read view at a fresh pin."""

_OP: Final[Mapping[str, str]] = {
    "create": "create", "merge": "merge", "revise": "revise", "annotate": "create",
}
_CARRY: Final[Mapping[str, frozenset[str]]] = {
    "create": frozenset(), "annotate": frozenset(),
    "merge": frozenset({"supersedes", "contradicts"}), "revise": frozenset({"supersedes"}),
}
_REPAIR_CARRY: Final = frozenset({"supersedes", "contradicts"})
"""Links a descriptor rewording carries: the head keeps its fact, so its
contradictions with other memories still hold, unlike a judged revise."""
_FIXED: Final = frozenset({
    "entity_id", "about_entity_ids", "naming", "quote_spans", "support_groups",
    "last_confirmed_at", "verdict_ref",
})
"""Snapshot fields the commit sets; a member's ``content`` supplies every other one."""


@dataclass(frozen=True, slots=True)
class MintV1:
    """An entity the commit gets or creates for an ``EntityKey``.

    Exactly one of a declaration's ``mint_key`` (with its type) or a
    ``participant_key`` is set.
    """

    mint_key: str | None = None
    entity_type: str | None = None
    participant_key: str | None = None

    def __post_init__(self) -> None:
        """Require exactly one key, and a type for a declaration."""
        if (self.mint_key is None) == (self.participant_key is None) or (
            self.mint_key is not None and not self.entity_type
        ):
            raise QuoteMemError("binding_mismatch", "A get-or-create names one typed key")


@dataclass(frozen=True, slots=True)
class ComponentMemberV1:
    """One member of a component, as its fence and transaction read it.

    Attributes:
        position: Its candidate identity.
        resolution: Its final disposition in this round.
        fence: Its fence inputs (``write_heads`` = the write's ``fence``).
        content: The snapshot fields of a write's statement other than
            anchors, naming, support, ``last_confirmed_at`` and ``verdict_ref``,
            which the commit sets; required for create, revise and annotate.
        naming: Its naming annotations as ``(slot, name, kind)``.
        judge_result_ref: Its latest ``result_index:judge``; the written
            snapshot's ``verdict_ref`` and every judged link's.
        hold_evidence: Its latest review issued a tombstone.
        touch: Its re-entry touch at this owner, if selected.
        spans: Its located quote spans, withheld when the tombstone guard
            refuses it.
        followers: Its duplicates, whose rows mirror its final
            disposition in the same transaction.
        correction_requested: The original proposal explicitly corrects or revises a fact.
        explicit_correction: The original proposal states an explicit correction
            (``judge_structure.explicit_correction``); its ``supersedes`` links record it.
        repairs: The descriptor rewordings its write commits with it: heads
            its correction outdated, whose replaced value the commit rewords.
    """

    position: PositionV1
    resolution: MemberResolutionV1
    fence: FenceMemberV1
    content: Mapping[str, Any] | None = None
    naming: tuple[tuple[int, str, NameKind], ...] = ()
    judge_result_ref: str | None = None
    hold_evidence: bool = False
    correction_requested: bool = False
    explicit_correction: bool = False
    touch: ReentryTouchV1 | None = None
    spans: tuple[QuoteSpanV1, ...] = ()
    followers: tuple[FollowerV1, ...] = ()
    repairs: tuple[HeadRepairV1, ...] = ()

    def __post_init__(self) -> None:
        """Bind the member's parts to one position."""
        key = self.position.position_key
        if (self.resolution.position_key, self.fence.position_key) != (key, key):
            raise QuoteMemError("binding_mismatch", "A component member is one position")


@dataclass
class ComponentResultV1:
    """What one component's commit did.

    Attributes:
        cause: The fence's single target-change cause; the component goes to the next round and
            nothing was written.
        committed: The committed transaction; None for a target change or a replay.
        replayed: An existing journal already covers the component.
        codes: Every member's final code.
        memories: Each written member's memory id.
        entities: The entity each get-or-create key resolved to.
        produced: The memory, entity and ``entity_name`` ids the commit wrote
            or deleted; later components exempt them.
    """

    cause: FenceCause | None = None
    committed: CommittedRows | None = None
    replayed: bool = False
    codes: dict[PositionKey, ReasonCode] = field(default_factory=dict[PositionKey, ReasonCode])
    memories: dict[PositionKey, str] = field(default_factory=dict[PositionKey, str])
    entities: dict[str, str] = field(default_factory=dict[str, str])
    produced: frozenset[str] = frozenset()


_PRODUCED: Final = frozenset({"item", "entity", "entity_name"})

ComponentOutcome = Literal["replayed", "committed", "changed", "raised"]
"""How a component's commit ended: an existing journal covered it, it committed, a
target-change cause sent it on (the last round writes its members ``target_changed``), or it
raised."""


@dataclass(frozen=True, slots=True)
class ComponentFenceEventV1:
    """One component's fence and commit as `commit_component` ran them.

    Content-free: ids, positions, frozen scores and outcomes. A replay ran no
    fence, so it has no pins, findings or tests.

    Attributes:
        owner_run_id: The Add owner.
        round: The round.
        members: The component's members in canonical order.
        pin: The round's pin, at which the members were nominated.
        s1: The pre-lock fence pin.
        findings: Each member's pre-lock findings.
        would_be: Each member's would-be test.
        producers: By delta key of any member, the positions of this round whose
            commits produced its memory; a key without producers is never exempt.
        discarded: A writing member's own target lost its relation (``fence_2``).
        pre_lock: The pre-lock cause; None when it passed.
        rechecked: The in-lock recheck ran.
        recheck: Its cause; None when it passed or did not run.
        head_cas: The head CAS held (True) or failed (False); None when not tried.
        capacity: The component exceeded ``MAX_TX_ROWS`` and its members were
            written ``support_budget_exceeded``.
        conflict: A storage constraint refused the component's transaction, which
            wrote nothing, and its members were written ``support_budget_exceeded``.
        outcome: How the commit ended.
        cause: The target-change cause of a ``changed`` component.
        commit_seq: The ``commit_seq`` of its transaction, when one committed.
        error: The refusal code of a ``raised`` component.
    """

    owner_run_id: str
    round: RoundNo
    members: tuple[PositionKey, ...]
    pin: ReadPin | None = None
    s1: FencePinV1 | None = None
    findings: Mapping[PositionKey, MemberFindingsV1] = field(
        default_factory=dict[PositionKey, MemberFindingsV1])
    would_be: Mapping[PositionKey, MemberWouldBeV1] = field(
        default_factory=dict[PositionKey, MemberWouldBeV1])
    producers: Mapping[str, frozenset[PositionKey]] = field(
        default_factory=dict[str, frozenset[PositionKey]])
    discarded: bool = False
    pre_lock: FenceCause | None = None
    rechecked: bool = False
    recheck: FenceCause | None = None
    head_cas: bool | None = None
    capacity: bool = False
    conflict: bool = False
    outcome: ComponentOutcome = "committed"
    cause: FenceCause | None = None
    commit_seq: int | None = None
    error: str | None = None


type ComponentObserver = Callable[[ComponentFenceEventV1], None]
"""Sees each component's ``ComponentFenceEventV1`` once its commit has ended."""


@dataclass
class _Trace:
    """What one `commit_component` decided, kept for its observer."""

    fence: ComponentFence | None = None
    findings: Mapping[PositionKey, MemberFindingsV1] = field(
        default_factory=dict[PositionKey, MemberFindingsV1])
    discarded: bool = False
    pre_lock: FenceCause | None = None
    rechecked: bool = False
    recheck: FenceCause | None = None
    head_cas: bool | None = None
    capacity: bool = False
    conflict: bool = False

    def event(
        self, owner_run_id: str, round_no: RoundNo, positions: Sequence[PositionKey],
        result: ComponentResultV1 | None, error: str | None = None,
    ) -> ComponentFenceEventV1:
        """The observer's event for an ended commit; ``result`` is None when it raised."""
        fence = self.fence
        outcome: ComponentOutcome = (
            "raised" if result is None else "replayed" if result.replayed
            else "changed" if result.cause is not None else "committed")
        committed = None if result is None else result.committed
        producers: dict[str, frozenset[PositionKey]] = {}
        if fence is not None:
            for test in fence.would_be.values():
                for key in (*test.fence.heads, *test.fence.retired):
                    found = fence.produced.get(test.items.get(key, key))
                    if found:
                        producers[key] = found
        return ComponentFenceEventV1(
            owner_run_id=owner_run_id, round=round_no, members=tuple(positions),
            pin=None if fence is None else fence.pin, s1=None if fence is None else fence.s1,
            findings=self.findings, would_be={} if fence is None else dict(fence.would_be),
            producers=producers, discarded=self.discarded, pre_lock=self.pre_lock,
            rechecked=self.rechecked, recheck=self.recheck, head_cas=self.head_cas,
            capacity=self.capacity, conflict=self.conflict, outcome=outcome,
            cause=None if result is None else result.cause,
            commit_seq=None if committed is None else committed.commit_seq, error=error,
        )


@dataclass
class _Pre:
    """Content-free results of the pre-lock read at s1."""

    heads: dict[str, ExpectedHead] = field(default_factory=lambda: dict[str, "ExpectedHead"]())
    existing: dict[str, str | None] = field(default_factory=dict[str, str | None])


def _pre_read(
    view: AddReadView, members: Sequence[ComponentMemberV1], mints: Mapping[str, MintV1],
    pre: _Pre,
) -> None:
    from quotemem.store.writer import ExpectedHead

    for member in members:
        for memory_id in member.fence.write_heads:
            record = view.memory(memory_id)
            if record is not None:
                pre.heads[memory_id] = ExpectedHead(memory_id, record.current_revision_id,
                                                    record.lifecycle_state)
    for key, mint in mints.items():
        record = (view.minted(mint.mint_key) if mint.mint_key is not None
                  else view.participant(cast(str, mint.participant_key)))
        pre.existing[key] = None if record is None else record.entity_id


def _discarded(write: WriteV1, findings: MemberFindingsV1) -> bool:
    """A discarded relation on the written head itself needs re-resolution."""
    return write.target is not None and write.target in findings.discard


@dataclass
class _Build:
    """The state of one component transaction while its rows are built."""

    txn: TxnRows
    members: Sequence[ComponentMemberV1]
    findings: Mapping[PositionKey, MemberFindingsV1]
    mints: Mapping[str, MintV1]
    existing: Mapping[str, str | None]
    owner_run_id: str
    policy_release_ref: str
    created: dict[PositionKey, str] = field(default_factory=dict[PositionKey, str])
    refused: dict[PositionKey, DispositionV1] = field(
        default_factory=dict[PositionKey, DispositionV1],
    )
    final: dict[PositionKey, DispositionV1] = field(
        default_factory=dict[PositionKey, DispositionV1],
    )
    entities: dict[str, str] = field(default_factory=dict[str, str])
    withheld: set[str] = field(default_factory=set[str])
    changed: dict[str, Row | None] = field(default_factory=dict[str, Row | None])
    candidates: dict[PositionKey, str] = field(default_factory=dict[PositionKey, str])
    repaired: set[str] = field(default_factory=set[str])
    forgotten: tuple[str, ...] | None = None
    names: tuple[ForgottenName, ...] | None = None
    cut: bool = False
    """The ``agent`` mode's settlement cut positions around forgotten names
    (``commit_round``'s ``forgotten_names``); only then does `cut_names` withhold any."""

    def entity(self, key: str | None, *, mint: bool = True) -> str | None:
        """Resolve an ``EntityKey``: an entity id, or the get-or-create of its key."""
        if key is None or key not in self.mints:
            return key
        if key in self.entities:
            return self.entities[key]
        existing = self.existing.get(key)
        if not mint:
            return existing
        spec = self.mints[key]
        if spec.mint_key is not None:
            entity_id = get_or_mint(self.txn, mint_key=spec.mint_key,
                                    entity_type=cast(str, spec.entity_type), existing=existing)
        else:
            entity_id = get_or_create_participant(
                self.txn, participant_key=cast(str, spec.participant_key), existing=existing,
            )
        self.entities[key] = entity_id
        return entity_id

    def memory(self, ref: MemoryRefV1 | None) -> str | None:
        """Bind a memory reference; a refused creator's reference binds to nothing."""
        if ref is None:
            return None
        if ref.memory_id is not None:
            return ref.memory_id
        return self.created.get(cast(PositionKey, ref.created_by))

    def touch_memory(self, memory_id: str) -> None:
        """Record a memory whose head or lifecycle the transaction changes, with its pre-image."""
        if memory_id not in self.changed:
            self.changed[memory_id] = self.txn.writer.get_row("item", memory_id)

    def screened(self, statement: str, keywords: Sequence[str],
                 spans: Sequence[QuoteSpanV1]) -> list[str]:
        """A new or revised head's keywords less those a tombstone of the scope forbids:
        each keyword carrying a word of what a forgotten statement says about
        the head's subject that neither its statement nor a readable quote shows
        (``forgotten_keywords``). The keyword analogue of the tombstone
        screen: a forgotten word does not come back as a keyword."""
        if self.forgotten is None:
            txn = self.txn
            self.forgotten = tuple(
                row["statement"] for row in txn.merged("tombstone", txn.pages(TombstoneRows(
                    scope_kind=txn.scope_kind, scope_id=txn.scope_id,
                    after_tombstone_seq=0)), lambda _row: True)
                if row["statement"])
        if not self.forgotten:
            return list(keywords)
        read = readability(self.txn)
        quotes = [text for span in spans if read.p_span(span)
                  and (text := _quoted(self.txn, span)) is not None]
        dropped = forgotten_keywords(statement, keywords, self.forgotten, quotes)
        return [keyword for keyword in keywords if keyword not in dropped]


    def cut_names(self, groups: Sequence[Sequence[QuoteSpanV1]]) -> list[QuoteSpanV1]:
        """The bytes of a forgotten name a written head's quote was cut around (settlement's
        ``_without_forgotten_names``): each occurrence, in a part the head quotes, of a name a
        forget took from exactly one entity, between two of its quotes or next to one in a
        sentence naming the bearer's sort (`forgotten_names.touching`). The commit withholds
        them, so the new sentence keeps no readable name. The owner's withholding transaction
        withheld the occurrences its own settlement cut before review (``ProposalV2.cut``),
        whatever the review decides; this check covers a head written all the same."""
        txn = self.txn
        if not self.cut:
            return []
        if self.names is None:
            rows = txn.merged("tombstone", txn.pages(TombstoneRows(
                scope_kind=txn.scope_kind, scope_id=txn.scope_id, after_tombstone_seq=0)),
                lambda _row: True)

            def bearers(name: str) -> list[str]:
                norm = name_norm(name)
                return [row["entity_id"] for row in txn.merged(
                    "entity_name", txn.pages(EntityNameRows(
                        scope_kind=txn.scope_kind, scope_id=txn.scope_id, name_norms=(norm,))),
                    lambda row: row["name_norm"] == norm)] if norm else []

            self.names = forgotten_names(sorted(
                (str(row["object_id"]), str(row["entity_id"]),
                 tuple(str(entity) for entity in row["about_entity_ids"]), str(row["statement"]))
                for row in rows if row["statement"]), bearers)
        if not self.names:
            return []
        spans = [span for group in groups for span in group]
        found: list[QuoteSpanV1] = []
        for evidence_id in sorted({span.evidence_id for span in spans}):
            row = txn.get("evidence", evidence_id)
            if row is None:
                continue
            text, start = str(row["text"]), int(row["byte_start"])
            for name in self.names:
                bearer = txn.get("entity", name.entity_id)
                kind = kind_of(name.tie, str(bearer["entity_type"]) if bearer is not None
                               and bearer["state"] == "active" else "")
                found += touching(spans, text, start, evidence_id,
                                  occurrences(name.name, text), kind)
        return found


def _quoted(txn: TxnRows, span: QuoteSpanV1) -> str | None:
    """The text a span quotes, or None when its Evidence cannot show it."""
    row = txn.get("evidence", span.evidence_id)
    if row is None:
        return None
    data = str(row["text"]).encode("utf-8")
    start, end = span.byte_start - row["byte_start"], span.byte_end - row["byte_start"]
    if not 0 <= start < end <= len(data):
        return None
    try:
        return data[start:end].decode("utf-8")
    except UnicodeDecodeError:
        return None


def _support_fits(head: Row, added: Iterable[Sequence[QuoteSpanV1]]) -> bool:
    """Whether a merge's added groups fit the head it is committed against.

    The head is the one the transaction reads, which may be a digest-preserving
    re-binding of the head the resolution pinned: its support can
    have grown since, so the resolver's own bound check does not cover it.
    """
    base = MemorySnapshotV2.model_validate(head["snapshot"])
    spans, indexed = support_union(base.quote_spans, base.support_groups, added)
    return len(indexed) <= MAX_SUPPORT_GROUPS and len(spans) <= MAX_MEMORY_SPANS


def _effective(txn: TxnRows, snapshot: MemorySnapshotV2) -> tuple[int | None, str | None]:
    """``effective_at`` of an update: ``valid_from``, else the latest source time."""
    if snapshot.valid_from is not None:
        return snapshot.valid_from, "valid_from"
    observed: list[int] = []
    for evidence_id in sorted({span.evidence_id for span in snapshot.quote_spans}):
        row = txn.get("evidence", evidence_id)
        if row is not None and row.get("source_observed_at") is not None:
            observed.append(row["source_observed_at"])
    return (max(observed), "source_observed_at") if observed else (None, None)


_PERIOD_FIELDS = ("time_expr", "valid_from", "valid_to", "time_granularity", "anchored_from")
"""The fields that hold a version's time."""


def _kept_period(base: MemorySnapshotV2, body: Row) -> Row:
    """A supplement keeps the period of the head it revises when the member has none.

    A supplement keeps all the head says, so a member without a time leaves the
    head's known bounds in place instead of erasing them (a dated trip would otherwise lose its
    week). A member with its own time, or of another kind
    (only an event has an end), keeps its own.
    """
    untimed = body.get("valid_from") is None and body.get("valid_to") is None
    if (not untimed or body.get("kind") != base.kind
            or (base.valid_from is None and base.valid_to is None)):
        return {}
    kept = base.model_dump(mode="json")
    return {field: kept[field] for field in _PERIOD_FIELDS}


def _snapshot(build: _Build, member: ComponentMemberV1, write: WriteV1,
              groups: Sequence[Sequence[QuoteSpanV1]], head: Row | None) -> MemorySnapshotV2:
    txn = build.txn
    if write.form == "merge":
        if head is None:
            raise QuoteMemError("reference_missing", "A merge target lost its head revision")
        base = MemorySnapshotV2.model_validate(head["snapshot"])
        spans, indexed = support_union(base.quote_spans, base.support_groups, groups)
        body: Row = {**base.model_dump(mode="json")}
    else:
        if member.content is None or _FIXED & member.content.keys():
            raise QuoteMemError("binding_mismatch",
                                "A write's content omits the fields commit sets")
        resolved = [build.entity(key) for key in member.resolution.anchors]
        anchors = [anchor for anchor in resolved if anchor is not None]
        if not anchors or len(anchors) != len(resolved):
            raise QuoteMemError("binding_mismatch", "A formal write has every anchor resolved")
        spans, indexed = support_union((), (), groups)
        # The primary first, then about_entity_ids sorted and unique without it,
        # whatever the slot order; two slots may name one entity, named once.
        naming: list[Row] = []
        named: set[tuple[str, str, NameKind]] = set()
        for slot, name, kind in member.naming:
            if not 0 <= slot < len(anchors):
                raise QuoteMemError("binding_mismatch", "A naming annotation names a member slot")
            key = (anchors[slot], name_norm(name), kind)
            if key not in named:
                named.add(key)
                naming.append({"entity_id": anchors[slot], "name": name, "name_kind": kind})
        body = {**member.content, "entity_id": anchors[0],
                "about_entity_ids": sorted(set(anchors[1:]) - {anchors[0]}), "naming": naming}
        if write.reason == "supplement" and head is not None:
            body.update(_kept_period(MemorySnapshotV2.model_validate(head["snapshot"]), body))
    body.update(
        quote_spans=[span.model_dump(mode="json") for span in spans], support_groups=indexed,
        last_confirmed_at=confirmed_at(txn, (span.evidence_id for span in spans)),
        verdict_ref=member.judge_result_ref,
        keywords=build.screened(str(body["statement"]), list(body["keywords"]), spans),
    )
    return MemorySnapshotV2.model_validate(body)


def _refuse(build: _Build, member: ComponentMemberV1) -> None:
    """Decide a write's per-item refusal before any row is built."""
    txn = build.txn
    resolution = member.resolution
    write = cast(WriteV1, resolution.write)
    key = resolution.position_key
    found = build.findings[key]
    primary = build.entity(resolution.anchors[0] if resolution.anchors else None, mint=False)
    if found.tombstoned:
        spans = member.spans or tuple(span for group in write.groups for span in group)
        build.withheld |= withhold_spans(txn, spans, reason="restated_forgotten",
                                         run_id=build.owner_run_id)
        build.refused[key] = DispositionV1("denied_by_forget", primary,
                                           judge_result_ref=member.judge_result_ref,
                                           hold_evidence=member.hold_evidence)
        return
    groups = [group for index, group in enumerate(write.groups) if index not in found.unusable]
    if write.groups and not groups:
        build.refused[key] = DispositionV1("evidence_invalidated", primary,
                                           judge_result_ref=member.judge_result_ref)


def _apply_write(build: _Build, member: ComponentMemberV1) -> None:
    """Write one unrefused formal member's rows."""
    txn = build.txn
    resolution = member.resolution
    write = cast(WriteV1, resolution.write)
    key = resolution.position_key
    found = build.findings[key]
    if key in build.refused:
        return
    groups = [group for index, group in enumerate(write.groups) if index not in found.unusable]
    build.withheld |= withhold_spans(txn, build.cut_names(groups), reason="restated_forgotten",
                                     run_id=build.owner_run_id)
    if write.form in ("create", "annotate"):
        memory_id, item, head = build.created[key], None, None
    else:
        memory_id = cast(str, write.target)
        item = txn.get("item", memory_id)
        if item is None:
            raise QuoteMemError("reference_missing", "A write target lost its Item")
        head = txn.get("revision", item["current_revision_id"])
        if write.form == "merge" and head is not None:
            kept = _unrenamed(txn, memory_id, head, groups)
            if groups and not kept:
                build.refused[key] = DispositionV1(
                    "evidence_invalidated",
                    build.entity(resolution.anchors[0] if resolution.anchors else None,
                                 mint=False),
                    judge_result_ref=member.judge_result_ref,
                )
                return
            groups = kept
        if write.form == "merge" and head is not None and not _support_fits(head, groups):
            build.refused[key] = DispositionV1(
                "support_budget_exceeded",
                build.entity(resolution.anchors[0] if resolution.anchors else None, mint=False),
                judge_result_ref=member.judge_result_ref, hold_evidence=member.hold_evidence,
            )
            return
    snapshot = _snapshot(build, member, write, groups, head)
    build.touch_memory(memory_id)
    revision = txn.put("revision", revision_row(
        txn, memory_id=memory_id, previous=head, op=_OP[write.form], reason=write.reason,
        snapshot=snapshot, source_candidate_id=build.candidates[key],
        policy_release_ref=build.policy_release_ref,
    ))
    txn.put("item", head_memory(txn, item, revision, memory_id=memory_id))
    carried: set[tuple[str, str]] = set()
    if head is not None and _CARRY[write.form]:
        carried = {(row["link_type"], row["target_key"])
                   for row in carry_links(txn, head, revision, _CARRY[write.form])}
    project_about(txn, revision)
    verdict = member.judge_result_ref
    for ended in write.ended:
        if ended.memory_id in found.discard:
            continue
        target = txn.get("item", ended.memory_id)
        if target is None:
            raise QuoteMemError("reference_missing", "An ended target lost its Item")
        build.touch_memory(ended.memory_id)
        txn.put("item", {**target, "lifecycle_state": ENDED_LIFECYCLE[ended.relation],
                         "retired_at": txn.now, "commit_seq": txn.commit_seq,
                         "updated_at": txn.now})
        effective_at, basis = (_effective(txn, snapshot) if ended.relation == "update"
                               else (None, None))
        txn.put("link", link_row(txn, revision, "supersedes", ended.memory_id,
                                 relation=ended.relation, effective_at=effective_at,
                                 effective_basis=basis, verdict_ref=verdict,
                                 correction_requested=member.explicit_correction))
    for ref in write.contradicts:
        target_id = build.memory(ref)
        # A merged head may already contradict the target: it keeps its one link.
        if target_id is None or target_id in found.discard or (
            ("contradicts", target_id) in carried
        ):
            continue
        txn.put("link", link_row(
            txn, revision, "contradicts", target_id, verdict_ref=verdict,
            correction_requested=member.correction_requested,
        ))
    for corrected in write.corrects:
        txn.put("link", link_row(txn, revision, "corrects_history", corrected,
                                 verdict_ref=verdict))
    build.final[key] = DispositionV1(
        resolution.code, snapshot.entity_id or None, memory_id=memory_id,
        revision_id=revision["object_id"], targets=resolution.targets,
        judge_result_ref=verdict, hold_evidence=member.hold_evidence,
    )
    _repair_heads(build, member, snapshot)


def _repair_heads(build: _Build, member: ComponentMemberV1,
                  correction: MemorySnapshotV2) -> None:
    """Write the member's descriptor rewordings in its own transaction.

    A planned head is reworded only while it is still the active head that states the
    planned wording (a support-only move keeps it) and its relation was not
    discarded; otherwise the head keeps its links alone, as before. The reworded version
    is op ``revise``, reason ``rewording``, with no candidate: the head's fact is
    unchanged, so the version claims no correction of it (the grader projects no change
    link for a rewording, and its old wording is neither current nor retired there).
    Every field is the head's except the statement, its digest, keywords and vector, and
    the verdict and pipeline of the correction, which record the cause: a forget reads
    the verdict to find the heads a forgotten correction reworded. The support stays
    the head's own: the new wording states neither value and only less than the old one,
    so the head's quotes still support every group, and forgetting or erasing the
    correction leaves the head readable. A rewording whose version would break an
    revision rule is not written: the head keeps its links alone, and a rewording never
    fails its correction's commit. A contradicted head keeps the correction's
    ``contradicts`` link as the judge recorded it.
    """
    if not member.repairs or member.judge_result_ref is None:
        return
    txn = build.txn
    found = build.findings[member.position.position_key]
    for repair in member.repairs:
        if repair.memory_id in found.discard or repair.memory_id in build.changed:
            continue
        item = txn.get("item", repair.memory_id)
        if item is None or item["lifecycle_state"] != "active":
            continue
        head = txn.get("revision", item["current_revision_id"])
        if head is None:
            continue
        base = MemorySnapshotV2.model_validate(head["snapshot"])
        if base.erased or base.statement_digest != repair.statement_digest:
            continue
        try:
            snapshot = MemorySnapshotV2.model_validate({
                **base.model_dump(mode="json"),
                "statement": repair.statement,
                "statement_digest": statement_digest(repair.statement),
                "keywords": build.screened(repair.statement, repair.keywords, base.quote_spans),
                "embedding": list(repair.embedding), "embedding_space": repair.embedding_space,
                "embedding_input_digest": repair.embedding_input_digest,
                "pipeline_version": correction.pipeline_version,
                "verdict_ref": member.judge_result_ref,
            })
        except ValidationError:
            # Nothing is written yet: the head keeps its link alone (the docstring).
            continue
        build.touch_memory(repair.memory_id)
        revised = txn.put("revision", revision_row(
            txn, memory_id=repair.memory_id, previous=head, op="revise", reason="rewording",
            snapshot=snapshot, source_candidate_id=None,
            policy_release_ref=build.policy_release_ref,
        ))
        txn.put("item", head_memory(txn, item, revised, memory_id=repair.memory_id))
        carry_links(txn, head, revised, _REPAIR_CARRY)
        project_about(txn, revised)
        build.repaired.add(repair.memory_id)


def _about(txn: TxnRows, entity_id: str) -> list[str]:
    """The memories whose head anchors or is about an entity, in id order."""
    found = {str(row["object_id"]) for row in txn.memories_of(entity_id)}
    for link in txn.links_to("about_entity", entity_id):
        revision = txn.get("revision", str(link["source_revision_id"]))
        if revision is not None:
            found.add(str(revision["memory_id"]))
    return sorted(found)


def _unname_siblings(build: _Build, member: ComponentMemberV1) -> None:
    """Requote the memories about an entity a correction renamed that state no name but
    quote the old one (``core/replaced_names``, the ``agent`` mode).

    After "my cat is called Max, not Bob" corrects the cat's name, "The user has a cat."
    quoting "My cat is called Bob" quotes "My cat": a requote (op ``requote``, the same
    statement) on the same Evidence, its keywords less those showing the old name, when what
    is left still holds every word it claims. The old name stays in the corrected memory's
    history: a correction is no forget, and nothing is withheld. The replaced names are
    those the corrected memory marks ``former`` and those its previous version gave the same
    entity that it no longer gives while it gives that entity another (`_replaced`), so a
    judge leaving out the ``former`` mark changes nothing. A memory this component itself
    wrote, other than the corrected one, is requoted too when its new statement no longer
    shows the old name ("The user's dog Rex loves the park." corrected to "The user's dog
    loves the park." still quoting "My dog Rex loves the park."): its requote is a version
    after the one this component wrote. A memory stating the old name, one not readable, and
    an old name another entity bears are left as they are. Like a head repair, the requote
    of a memory the component did not write changes no fact, so it is no target change for
    the round's later components.

    A memory told in another message than the old name's tie ("Rex and I went hiking.")
    takes the tie from the correction's own quote ("my dog" of "my dog is called Rocky, not
    Rex"). A readable memory still showing the old name after this (its statement states
    it, or no requote holds what it claims) gets a ``contradicts`` link from the corrected
    head, without ``correction_requested``: Search reads a state memory as possibly
    outdated. A support group such a memory has that quotes the new name is dropped first
    (`_without_given`), and a later merge of one into it is refused (`_unrenamed`), so no
    memory quotes both names."""
    write = member.resolution.write
    key = member.position.position_key
    final = build.final.get(key)
    if write is None or write.reason != "correction" or key in build.refused or (
            final is None or final.memory_id is None):
        return
    txn = build.txn
    item = txn.get("item", final.memory_id)
    head = None if item is None else txn.get("revision", str(item["current_revision_id"]))
    if head is None:
        return
    read = readability(txn)
    corrected = MemorySnapshotV2.model_validate(head["snapshot"])
    told = [(span, text) for span in corrected.quote_spans
            if (text := _quoted(txn, span)) is not None]
    linked = {str(link["target_key"]) for link in txn.links_of(str(head["object_id"]))
              if link["link_type"] == "contradicts"}
    for entity_id, name in _replaced(txn, head):
        if _bearers(txn, name_norm(name)) - {entity_id}:
            continue
        given = [entry.name for entry in corrected.naming
                 if entry.entity_id == entity_id and entry.name_kind != "former"]
        for memory_id in _about(txn, entity_id):
            if memory_id == final.memory_id:
                continue
            sibling = txn.get("item", memory_id)
            if sibling is None or sibling["lifecycle_state"] != "active":
                continue
            version = txn.get("revision", str(sibling["current_revision_id"]))
            if version is None:
                continue
            snapshot = MemorySnapshotV2.model_validate(version["snapshot"])
            if snapshot.erased or not read.p_ver(snapshot):
                continue
            groups = None if occurrences(name, snapshot.statement) else _unnamed_groups(
                txn, snapshot, name, told, given)
            if groups is None:
                if memory_id not in linked and _shows(txn, snapshot, name):
                    # No requote hides the old name: the correction questions the memory,
                    # which reads as possibly outdated, as a contradiction does.
                    _without_given(build, memory_id, version, snapshot, given)
                    txn.put("link", link_row(txn, head, "contradicts", memory_id,
                                             verdict_ref=member.judge_result_ref))
                    linked.add(memory_id)
                continue
            keywords = [keyword for keyword in snapshot.keywords
                        if not occurrences(name, keyword)]
            if ("revision", version["object_id"]) in txn.written:
                # This transaction wrote the head: it is written again on the new quotes, as
                # a memory takes one new version per transaction.
                _requote_written(txn, version, snapshot, groups, keywords)
                continue
            if memory_id not in build.changed:
                build.repaired.add(memory_id)
            requote_on(txn, memory_id, version=version, support=groups,
                       policy_release_ref=build.policy_release_ref, keywords=keywords)


def _requote_written(txn: TxnRows, version: Row, snapshot: MemorySnapshotV2,
                     groups: Sequence[Sequence[QuoteSpanV1]], keywords: Sequence[str]) -> None:
    """A revision this transaction wrote, written again on ``groups`` and ``keywords``, as
    ``entity_ops.requote_on`` builds a requote's snapshot: the same id, version and links."""

    def key(span: QuoteSpanV1) -> tuple[str, int, int]:
        return span.evidence_id, span.byte_start, span.byte_end

    spans = sorted({key(span): span for group in groups for span in group}.values(), key=key)
    position = {key(span): index for index, span in enumerate(spans)}
    requoted = MemorySnapshotV2.model_validate({
        **snapshot.model_dump(mode="json"),
        "quote_spans": [span.model_dump(mode="json") for span in spans],
        "support_groups": [list(group) for group in sorted({
            tuple(sorted(position[key(span)] for span in group)) for group in groups})],
        "keywords": list(keywords),
        "last_confirmed_at": confirmed_at(txn, (span.evidence_id for span in spans)),
    })
    image = requoted.model_dump(mode="json")
    txn.put("revision", {
        **version, "snapshot": image, "content_hash": str(integrity_digest(requoted)),
        "evidence_ids": sorted({span["evidence_id"] for span in image["quote_spans"]})})


def _replaced(txn: TxnRows, head: Row) -> list[tuple[str, str]]:
    """The names a correction's new head replaced, as ``(entity_id, name)``: each name it
    marks ``former``, and each its previous version gave an entity (as a name or an alias)
    that it no longer gives while it gives that entity another. A name the new head still
    gives, or one inside a name it gives ("Bob" of "Bob Smith"), and the user's own names,
    are none."""
    corrected = MemorySnapshotV2.model_validate(head["snapshot"])
    current = [entry for entry in corrected.naming if entry.name_kind != "former"]
    found = {(entry.entity_id, entry.name) for entry in corrected.naming
             if entry.name_kind == "former"}
    previous = head.get("previous_revision_id")
    row = None if previous is None else txn.get("revision", str(previous))
    if row is not None:
        before = MemorySnapshotV2.model_validate(row["snapshot"])
        found |= {(entry.entity_id, entry.name) for entry in before.naming
                  if entry.name_kind != "former" and any(
                      own.entity_id == entry.entity_id for own in current)}
    return sorted((entity_id, name) for entity_id, name in found
                  if entity_id != USER and name_norm(name) and not any(
                      occurrences(name, own.name) for own in current))


def _bearers(txn: TxnRows, norm: str) -> set[str]:
    """The entities a name (its ``name_norm``) is an ``entity_name`` row of."""
    return {str(row["entity_id"]) for row in txn.merged("entity_name", txn.pages(
        EntityNameRows(scope_kind=txn.scope_kind, scope_id=txn.scope_id, name_norms=(norm,))),
        lambda row: row["name_norm"] == norm)}


def _unnamed_groups(txn: TxnRows, snapshot: MemorySnapshotV2, name: str,
                    told: Sequence[tuple[QuoteSpanV1, str]] = (),
                    given: Sequence[str] = ()) -> list[list[QuoteSpanV1]] | None:
    """A memory's support groups quoting no form of an old name (`replaced_names.unnamed`),
    or None when none of its quotes shows the name, a group would be left empty, the
    spans exceed their bound, or the pieces no longer hold what the statement claims
    (`supports`, `supports_cut`). A group whose pieces no longer show whom the statement
    is about ("Kiwi is five years old." cut to "is five years old.") also quotes the
    phrase of the same message showing it (`tie_span`: "My parrot"), or else the one of the
    correction's quotes ``told`` that shows no name ``given``."""
    groups: list[list[QuoteSpanV1]] = []
    changed = False
    ties = named_ties(snapshot.statement)
    for group in snapshot.support_groups:
        pieces: list[QuoteSpanV1] = []
        row: Row | None = None
        for index in group:
            span = snapshot.quote_spans[index]
            row = txn.get("evidence", span.evidence_id)
            if row is None:
                return None
            found = unnamed(span, str(row["text"]), int(row["byte_start"]), name)
            if found is None:
                return None
            changed = changed or found != [span]
            pieces.extend(found)
        shown = " ".join(text for span in pieces if (text := _quoted(txn, span)) is not None)
        if ties and row is not None and not ties <= named_ties(shown):
            text, first = str(row["text"]), int(row["byte_start"])
            phrase = tie_span(snapshot.statement, text, name)
            if phrase is not None:
                pieces.insert(0, QuoteSpanV1(
                    evidence_id=snapshot.quote_spans[group[-1]].evidence_id,
                    byte_start=first + len(text[:phrase[0]].encode("utf-8")),
                    byte_end=first + len(text[:phrase[1]].encode("utf-8"))))
            else:
                pieces.extend(_told_tie(snapshot.statement, told, name, given))
        groups.append(sorted({(span.evidence_id, span.byte_start, span.byte_end): span
                              for span in pieces}.values(),
                             key=lambda span: (span.evidence_id, span.byte_start)))
    spans = {(span.evidence_id, span.byte_start, span.byte_end)
             for group in groups for span in group}
    if not changed or len(spans) > MAX_MEMORY_SPANS or any(
            len(group) > MAX_GROUP_SPANS for group in groups):
        return None
    quoted = [text for group in groups for span in group
              if (text := _quoted(txn, span)) is not None]
    before = [text for span in snapshot.quote_spans if (text := _quoted(txn, span)) is not None]
    return groups if supports(snapshot.statement, quoted) or supports_cut(
        snapshot.statement, before, quoted) else None


def _told_tie(statement: str, told: Sequence[tuple[QuoteSpanV1, str]], name: str,
              given: Sequence[str]) -> list[QuoteSpanV1]:
    """The first phrase of the correction's quotes ``told`` showing whom ``statement`` is
    about and no name, old or ``given`` ("my dog" of "my dog is called Rocky, not Rex"), as
    one span; none when no quote holds one."""
    for span, text in told:
        phrase = tie_span(statement, text, name)
        if phrase is None or any(occurrences(other, text[phrase[0]:phrase[1]])
                                 for other in given):
            continue
        return [QuoteSpanV1(
            evidence_id=span.evidence_id,
            byte_start=span.byte_start + len(text[:phrase[0]].encode("utf-8")),
            byte_end=span.byte_start + len(text[:phrase[1]].encode("utf-8")))]
    return []


def _shows(txn: TxnRows, snapshot: MemorySnapshotV2, name: str) -> bool:
    """Whether a memory's statement or one of its quotes shows a name."""
    return bool(occurrences(name, snapshot.statement)) or any(
        occurrences(name, text) for span in snapshot.quote_spans
        if (text := _quoted(txn, span)) is not None)


def _without_given(build: _Build, memory_id: str, version: Row, snapshot: MemorySnapshotV2,
                   given: Sequence[str]) -> None:
    """A memory still showing the old name, requoted without the support groups that quote
    a new name ``given`` (the judge merged "my cat is called Nova" into "Luna is the user's
    cat."); unchanged when no group or every group quotes one. A head this transaction
    wrote is written again; an earlier one is requoted (`entity_ops.requote_on`)."""
    txn = build.txn
    if not given:
        return

    def quotes_given(group: Sequence[int]) -> bool:
        return any(occurrences(other, text) for index in group
                   if (text := _quoted(txn, snapshot.quote_spans[index])) is not None
                   for other in given)

    kept = [group for group in snapshot.support_groups if not quotes_given(group)]
    if not kept or len(kept) == len(snapshot.support_groups):
        return
    support = [[snapshot.quote_spans[index] for index in group] for group in kept]
    if ("revision", version["object_id"]) in txn.written:
        _requote_written(txn, version, snapshot, support, snapshot.keywords)
        return
    if memory_id not in build.changed:
        build.repaired.add(memory_id)
    requote_on(txn, memory_id, version=version, support=support,
               policy_release_ref=build.policy_release_ref, keywords=snapshot.keywords)


def _unrenamed(txn: TxnRows, memory_id: str, head: Row,
               groups: Sequence[Sequence[QuoteSpanV1]]) -> list[Sequence[QuoteSpanV1]]:
    """A merge's groups less those quoting a name a live correction gave an entity while the
    target still shows the name the correction replaced, in its statement or a quote
    (`_unname_siblings` linked the target ``contradicts``): the judge merged "my cat is
    called Nova" into "Luna is the user's cat.", which would then quote both names."""
    target = MemorySnapshotV2.model_validate(head["snapshot"])
    given: set[str] = set()
    for link in txn.links_to("contradicts", memory_id):
        source = txn.get("revision", str(link["source_revision_id"]))
        owner = None if source is None else txn.get("item", str(source["memory_id"]))
        if source is None or owner is None or owner["lifecycle_state"] != "active" or (
                owner["current_revision_id"] != source["object_id"]):
            continue
        corrected = MemorySnapshotV2.model_validate(source["snapshot"])
        for entity_id, name in _replaced(txn, source):
            if _shows(txn, target, name):
                given.update(entry.name for entry in corrected.naming
                             if entry.entity_id == entity_id and entry.name_kind != "former")
    if not given:
        return list(groups)
    return [group for group in groups if not any(
        occurrences(other, text) for span in group
        if (text := _quoted(txn, span)) is not None for other in given)]


def _refused_merges(build: _Build) -> dict[str, DispositionV1]:
    """By target, the refusal of the merges into a head no unrefused member writes."""
    written: set[str] = set()
    refused: dict[str, DispositionV1] = {}
    for member in build.members:
        write = member.resolution.write
        if write is None or write.form != "merge" or write.target is None:
            continue
        key = member.position.position_key
        if key in build.refused:
            refused.setdefault(write.target, build.refused[key])
        else:
            written.add(write.target)
    return {target: found for target, found in refused.items() if target not in written}


def _dependant(build: _Build, member: ComponentMemberV1,
               merges: Mapping[str, DispositionV1]) -> DispositionV1:
    """A non-writing member's disposition; a refused writer's dependant mirrors it.

    A duplicate or superseded member of a refused create names the creator; one
    of a refused merge names the merge's target, which no other write of the
    component then changes.
    """
    resolution = member.resolution
    ref = resolution.memory
    if ref is not None and ref.created_by is not None and ref.created_by in build.refused:
        return mirrored(build.refused[ref.created_by])
    if (ref is not None and ref.memory_id is not None and ref.memory_id in merges
            and resolution.code in ("duplicate_in_add", "superseded_in_add")):
        return mirrored(merges[ref.memory_id])
    return DispositionV1(
        resolution.code,
        build.entity(resolution.anchors[0] if resolution.anchors else None, mint=False),
        memory_id=build.memory(ref), targets=resolution.targets,
        judge_result_ref=member.judge_result_ref, hold_evidence=member.hold_evidence,
    )


def _build(build: _Build) -> None:
    for member in build.members:
        position = member.position
        build.candidates[position.position_key] = position.candidate_id or new_ulid()
    for member in build.members:
        write = member.resolution.write
        if write is not None and write.form in ("create", "annotate"):
            build.created[member.position.position_key] = new_ulid()
    for member in build.members:
        if member.resolution.write is not None:
            _refuse(build, member)
    for key in build.refused:
        build.created.pop(key, None)
    for member in build.members:
        if member.resolution.write is not None:
            _apply_write(build, member)
    if build.cut:
        for member in build.members:
            _unname_siblings(build, member)
    merges = _refused_merges(build)
    for member in build.members:
        key = member.position.position_key
        if key in build.refused:
            build.final[key] = build.refused[key]
        elif member.resolution.write is None:
            build.final[key] = _dependant(build, member, merges)
    for member in build.members:
        key = member.position.position_key
        write_disposition(build.txn, member.position, build.final[key], touch=member.touch,
                          new_id=build.candidates[key])
    _write_followers(build)
    for memory_id, before in build.changed.items():
        rederive_entity_names(build.txn, memory_id, before=before)


def _write_followers(build: _Build) -> None:
    """Write each member's duplicates mirroring its final disposition."""
    for member in build.members:
        mirror = mirrored(build.final[member.position.position_key])
        for follower in member.followers:
            build.final[follower.position.position_key] = mirror
            write_disposition(build.txn, follower.position, mirror)


def _uniform(build: _Build, code: Literal["support_budget_exceeded", "target_changed"],
             ) -> None:
    """Every member with one code and no write or mint (budget, last round)."""
    for member in build.members:
        resolution = member.resolution
        disposition = DispositionV1(
            code,
            build.entity(resolution.anchors[0] if resolution.anchors else None, mint=False),
            judge_result_ref=member.judge_result_ref, hold_evidence=member.hold_evidence,
        )
        build.final[member.position.position_key] = disposition
        write_disposition(build.txn, member.position, disposition, touch=member.touch)
    _write_followers(build)


def _digest(member: ComponentMemberV1) -> IntegrityDigest:
    return integrity_digest(member.resolution)


def commit_component(
    writer: Writer, ticket: WriteTicket, *, owner_run_id: str, round_no: RoundNo,
    members: Sequence[ComponentMemberV1], mints: Mapping[str, MintV1], read: ReadAt,
    produced: Mapping[str, frozenset[PositionKey]], updated_at: int, publish: Publish,
    groups_of: GroupsOf, policy_release_ref: str, clock: Clock,
    observe: ComponentObserver | None = None,
) -> ComponentResultV1:
    """Fence and commit one formal component.

    Args:
        writer: The single Writer.
        ticket: The owner's scoped ticket.
        owner_run_id: The Add owner.
        round_no: The round.
        members: The component's members in canonical order, every one sharing
            its round's pin.
        mints: The get-or-create entities, by the ``EntityKey`` the resolution
            uses for them.
        read: The owner's read at a fresh pin (s1).
        produced: By memory, entity and ``entity_name`` id, the positions whose
            commits this round changed the row (exemptions).
        updated_at: The transaction's time.
        publish: The read model's publication of the committed transaction.
        groups_of: Reads a pending candidate's support groups, for the expiries
            a tombstone-guard withholding causes.
        policy_release_ref: The policy release the transaction is written under.
        clock: The owner's trusted clock; the actual finalize owner's budget
            is checked with it before the pre-lock work and under the mutex
            right before the write.
        observe: Sees the component's fence and outcome once its commit has
            ended, a refusal included, for the recorded replay; None for none.

    Returns:
        The component's result: a target change with its cause, a replay, or the commit.

    Raises:
        QuoteMemError: ``delivery_busy`` once the owner's budget ends (nothing of
            the component is written); ``binding_mismatch`` for inconsistent
            inputs or a ``tombstone_seq`` changed under a valid lease; or any
            refusal of `Writer.commit_rows` other than its head CAS, and of
            its sizing other than the capacity refusal.
    """
    return _observed_component(
        writer, ticket, owner_run_id=owner_run_id, round_no=round_no, members=members,
        mints=mints, read=read, produced=produced, updated_at=updated_at, publish=publish,
        groups_of=groups_of, policy_release_ref=policy_release_ref, clock=clock,
        observe=observe,
    )


def _observed_component(
    writer: Writer, ticket: WriteTicket, *, owner_run_id: str, round_no: RoundNo,
    members: Sequence[ComponentMemberV1], mints: Mapping[str, MintV1], read: ReadAt,
    produced: Mapping[str, frozenset[PositionKey]], updated_at: int, publish: Publish,
    groups_of: GroupsOf, policy_release_ref: str, clock: Clock,
    observe: ComponentObserver | None,
    journaled: Callable[[], set[PositionKey]] | None = None, forgotten_names: bool = False,
) -> ComponentResultV1:
    """Keep one component's observation around its independent or round-local journal read."""
    trace = _Trace()
    positions = sorted(member.position.position_key for member in members)
    try:
        result = _commit_component(
            writer, ticket, owner_run_id=owner_run_id, round_no=round_no, members=members,
            mints=mints, read=read, produced=produced, updated_at=updated_at, publish=publish,
            groups_of=groups_of, policy_release_ref=policy_release_ref, clock=clock, trace=trace,
            journaled=journaled, forgotten_names=forgotten_names,
        )
    except QuoteMemError as error:
        if observe is not None:
            observe(trace.event(owner_run_id, round_no, positions, None, error.code))
        raise
    if observe is not None:
        observe(trace.event(owner_run_id, round_no, positions, result))
    return result


def _commit_component(
    writer: Writer, ticket: WriteTicket, *, owner_run_id: str, round_no: RoundNo,
    members: Sequence[ComponentMemberV1], mints: Mapping[str, MintV1], read: ReadAt,
    produced: Mapping[str, frozenset[PositionKey]], updated_at: int, publish: Publish,
    groups_of: GroupsOf, policy_release_ref: str, clock: Clock, trace: _Trace,
    journaled: Callable[[], set[PositionKey]] | None, forgotten_names: bool = False,
) -> ComponentResultV1:
    """`commit_component`'s body; each decision is also kept in ``trace``."""
    from quotemem.store.writer import MAX_TX_ROWS

    if not members or not any(member.resolution.write is not None for member in members):
        raise QuoteMemError("binding_mismatch", "A component holds at least one formal write")
    if ticket.scope_kind is None or ticket.scope_id is None or ticket.run_id is None:
        raise QuoteMemError("scope_denied", "A component's commit needs a scoped owner Run")
    scope_kind, scope_id, run_id = cast(ScopeKind, ticket.scope_kind), ticket.scope_id, \
        ticket.run_id
    positions = sorted(member.position.position_key for member in members)
    if len(set(positions)) != len(positions):
        raise QuoteMemError("binding_mismatch", "A component names each position once")
    owner_budget(writer, owner_run_id, clock)
    done = journaled() if journaled is not None else journaled_positions(
        writer, run_id, owner_run_id=owner_run_id, round_no=round_no, txn="component",
    )
    if set(positions) & done:
        if not set(positions) <= done:
            raise QuoteMemError("binding_mismatch", "A component's journal covers all its members")
        return ComponentResultV1(replayed=True)
    fence = ComponentFence([member.fence for member in members], produced=produced)
    trace.fence = fence
    pre = _Pre()

    def read_fence_inputs(view: AddReadView) -> None:
        fence.read(view)
        _pre_read(view, members, mints, pre)

    read(read_fence_inputs)
    findings = {found.position_key: found
                for found in fence.settle(writer, scope_kind, scope_id)}
    trace.findings = findings
    causes: set[FenceCause] = {cause for found in findings.values() for cause in found.causes}
    trace.discarded = any(
        member.resolution.write is not None
        and _discarded(member.resolution.write, findings[member.position.position_key])
        for member in members)
    if trace.discarded:
        causes.add("fence_2")
    def changed(cause: FenceCause) -> ComponentResultV1:
        if round_no < MAX_ROUND:
            return ComponentResultV1(cause=cause)
        with writer.mutex:
            build = _new_build(writer, scope_kind, scope_id, run_id, updated_at, members,
                               findings, mints, pre, owner_run_id, policy_release_ref)
            _uniform(build, "target_changed")
            journal = _component_journal(writer, build, owner_run_id, round_no, positions,
                                         members)
            owner_budget(writer, owner_run_id, clock)
            committed = writer.commit_rows(
                ticket, build.txn.row_writes(), deletes=build.txn.row_deletes(),
                journal=journal, updated_at=updated_at, publish=publish,
            )
        return ComponentResultV1(cause=cause, committed=committed, codes={
            key: disposition.code for key, disposition in build.final.items()})

    cause = trace.pre_lock = component_cause(causes)
    if cause is not None:
        return changed(cause)
    s1 = fence.s1
    assert s1 is not None
    committed: CommittedRows | None = None
    with writer.mutex:
        cause = recheck(writer, scope_kind, scope_id, s1, [member.fence for member in members])
        trace.rechecked, trace.recheck = True, cause
        if cause is not None:
            return changed(cause)
        build = _new_build(writer, scope_kind, scope_id, run_id, updated_at, members, findings,
                           mints, pre, owner_run_id, policy_release_ref)
        build.cut = forgotten_names
        _build(build)
        expire_unusable(build.txn, build.withheld, groups_of=groups_of)
        journal = _component_journal(writer, build, owner_run_id, round_no, positions, members)
        heads = sorted({head.memory_id: head for head in pre.heads.values()}.values(),
                       key=lambda head: head.memory_id)
        # One preparation measures the component and writes it when it fits; one that
        # cannot fit (the Writer's capacity refusal included) commits uniformly, and so
        # does one whose transaction a storage constraint refused: that refusal wrote
        # nothing and would refuse every replay of the same rows, so parking the
        # members lets the owner, its scope and every other scope go on.
        limit: int | None = MAX_TX_ROWS
        while True:
            refused = False
            try:
                written = writer.commit_rows_within(
                    ticket, build.txn.row_writes(), deletes=build.txn.row_deletes(),
                    heads=heads, journal=journal, updated_at=updated_at, publish=publish,
                    before_write=lambda: owner_budget(writer, owner_run_id, clock),
                    limit=limit, capacity_refusal=limit is not None,
                )
            except QuoteMemError as error:
                if error.code == "storage_conflict" and limit is not None:
                    trace.conflict = refused = True
                    written = None
                elif error.code != "target_changed":
                    raise
                else:
                    trace.head_cas, cause = False, "fence_2"
                    break
            if refused or isinstance(written, int):
                trace.capacity = not refused
                build = _new_build(writer, scope_kind, scope_id, run_id, updated_at, members,
                                   findings, mints, pre, owner_run_id, policy_release_ref)
                _uniform(build, "support_budget_exceeded")
                journal = _component_journal(writer, build, owner_run_id, round_no,
                                             positions, members)
                limit = None
                continue
            committed = written
            trace.head_cas = True
            break
    if cause is not None:
        return changed(cause)
    return ComponentResultV1(
        committed=committed,
        codes={key: disposition.code for key, disposition in build.final.items()},
        memories={key: cast(str, disposition.memory_id)
                  for key, disposition in build.final.items()
                  if disposition.revision_id is not None},
        entities=dict(build.entities),
        produced=frozenset(
            {identity for kind, identity in build.txn.written if kind in _PRODUCED}
            | {delete.object_id for delete in build.txn.row_deletes()
               if delete.kind in _PRODUCED}) - build.repaired,
    )


def _new_build(
    writer: Writer, scope_kind: ScopeKind, scope_id: str, run_id: str, now: int,
    members: Sequence[ComponentMemberV1], findings: Mapping[PositionKey, MemberFindingsV1],
    mints: Mapping[str, MintV1], pre: _Pre, owner_run_id: str, policy_release_ref: str,
) -> _Build:
    txn = TxnRows(writer, scope_kind, scope_id, writer.get_counter("commit_seq") + 1, now, run_id)
    return _Build(txn, members, findings, mints, pre.existing, owner_run_id, policy_release_ref)


def _component_journal(
    writer: Writer, build: _Build, owner_run_id: str, round_no: RoundNo,
    positions: Sequence[PositionKey], members: Sequence[ComponentMemberV1],
) -> CommitJournalV1:
    """The component's journal: its members, each followed by its duplicates."""
    build.txn.stamp()
    digests = {member.position.position_key: _digest(member) for member in members}
    followers = {member.position.position_key: member.followers for member in members}
    entries: list[tuple[PositionKey, IntegrityDigest]] = []
    for position in positions:
        entries.append((position, digests[position]))
        for follower in followers[position]:
            mirror = build.final[follower.position.position_key]
            entries.append((follower.position.position_key, integrity_digest({
                "code": mirror.code, "entity_id": mirror.entity_id,
                "memory_id": mirror.memory_id, "judge_result_ref": mirror.judge_result_ref,
                "hold_evidence": mirror.hold_evidence,
            })))
    return _journal(writer, build.txn, owner_run_id=owner_run_id, round_no=round_no,
                    kind="component", entries=entries)


@dataclass
class RoundCommitV1:
    """The result for one owner round.

    Attributes:
        codes: Every committed position's final code.
        memories: Each written position's memory id.
        round2: The content-free entries of the next round in canonical order:
            round 1's resolver requests with their reason, ``reentry_woken``
            and every member of a component the fence sent on.
        rejection: The rejection chunks committed now.
        components: Each component's result, in commit order.
    """

    codes: dict[PositionKey, ReasonCode] = field(default_factory=dict[PositionKey, ReasonCode])
    memories: dict[PositionKey, str] = field(default_factory=dict[PositionKey, str])
    round2: list[Round2EntryV1] = field(default_factory=list[Round2EntryV1])
    rejection: list[CommittedRows] = field(default_factory=list["CommittedRows"])
    components: list[ComponentResultV1] = field(default_factory=list[ComponentResultV1])


def commit_round(
    writer: Writer, ticket: WriteTicket, *, owner_run_id: str, plan: ResolutionPlanV1,
    rejection: Sequence[RejectionEntryV1], members: Mapping[PositionKey, ComponentMemberV1],
    mints: Mapping[str, MintV1], read: ReadAt, updated_at: int, publish: Publish,
    groups_of: GroupsOf, policy_release_ref: str, clock: Clock,
    reentry_woken: Iterable[PositionKey] = (), observe: ComponentObserver | None = None,
    forgotten_names: bool = False,
) -> RoundCommitV1:
    """Commit one owner round's resolution.

    The rejection transaction commits first, chunked; then each formal component
    in the order of its minimum canonical key. A component's commits exempt
    later components' findings on the rows it produced. A
    component the fence finds changed goes to the next round whole, each member
    with the component's single cause; in the last round (``MAX_ROUND``) its
    members become ``target_changed``.

    Args:
        writer: The single Writer.
        ticket: The owner's scoped ticket.
        owner_run_id: The Add owner.
        plan: The resolver's plan for this round.
        rejection: The round's positions outside every component, as their
            rejection entries (re-entry touches and quiet members included); a
            re-entered member that a component writes, or that goes to round 2,
            is here only with its touch.
        members: Every component member, by position.
        mints: The get-or-create entities, by ``EntityKey``.
        read: The owner's read at a fresh pin (s1), once per component.
        updated_at: The transactions' time.
        publish: The read model's publication of each transaction.
        groups_of: Reads a pending candidate's support groups.
        clock: The owner's trusted clock for every budget check.
        policy_release_ref: The policy release the transactions are written under.
        reentry_woken: Positions the round-2 wake comparison admits.
        observe: Sees each component's fence and outcome (`commit_component`).
        forgotten_names: The owner's settlement cut positions around forgotten names (the
            ``agent`` mode): a written head withholds the name bytes it was cut around
            (`_Build.cut_names`).

    Returns:
        The round's codes, memories, round-2 entries and transactions.

    Raises:
        QuoteMemError: ``binding_mismatch`` when the inputs do not partition the
            plan; or as `commit_rejection` and `commit_component`.
    """
    round_no = plan.round
    component_keys = [key for component in plan.components for key in component]
    rejected = [entry.position.position_key for entry in rejection]
    settled = {entry.position.position_key for entry in rejection
               if entry.disposition is not None or entry.expire}
    if (set(members) != set(component_keys) or len(component_keys) != len(set(component_keys))
            or settled & set(component_keys) or len(rejected) != len(set(rejected))):
        raise QuoteMemError("binding_mismatch", "Rejection and components partition the round")
    woken = sorted(set(reentry_woken))
    if round_no >= 2 and (plan.round2 or woken):
        raise QuoteMemError("binding_mismatch", "Only round 1 asks for another round")
    result = RoundCommitV1()
    result.rejection = commit_rejection(
        writer, ticket, owner_run_id=owner_run_id, round_no=round_no, entries=rejection,
        updated_at=updated_at, publish=publish, groups_of=groups_of, clock=clock,
    )
    result.codes.update({entry.position.position_key: entry.disposition.code
                         for entry in rejection if entry.disposition is not None})
    entries = [Round2EntryV1(position_key=request.position_key, cause=request.reason)
               for request in plan.round2]
    entries.extend(Round2EntryV1(position_key=key, cause="reentry_woken") for key in woken)
    produced: dict[str, frozenset[PositionKey]] = {}
    journaled: set[PositionKey] | None = None

    def component_positions() -> set[PositionKey]:
        nonlocal journaled
        if journaled is None:
            journaled = journaled_positions(
                writer, cast(str, ticket.run_id), owner_run_id=owner_run_id,
                round_no=round_no, txn="component",
            )
        return journaled

    # The plan lists components and their members in canonical order: a later member's
    # component follows the earlier writes its fence exempts as issued siblings'.
    # One owner runs them serially. Load durable replay coverage once, after the first
    # component's budget check, then extend it only after a committed component returns.
    # An interrupted invocation discards this set and reloads the journals on recovery.
    for component in plan.components:
        done = _observed_component(
            writer, ticket, owner_run_id=owner_run_id, round_no=round_no,
            members=[members[key] for key in component], mints=mints, read=read,
            produced=produced, updated_at=updated_at, publish=publish, groups_of=groups_of,
            policy_release_ref=policy_release_ref, clock=clock, observe=observe,
            journaled=component_positions, forgotten_names=forgotten_names,
        )
        result.components.append(done)
        if done.committed is not None:
            component_positions().update(done.codes)
        if done.cause is not None and round_no < MAX_ROUND:
            entries.extend(Round2EntryV1(position_key=key, cause=done.cause)
                           for key in sorted(component))
            continue
        result.codes.update(done.codes)
        result.memories.update(done.memories)
        for identity in done.produced:
            produced[identity] = produced.get(identity, frozenset()) | frozenset(component)
    keys = [entry.position_key for entry in entries]
    if len(keys) != len(set(keys)):
        raise QuoteMemError("binding_mismatch", "A position enters the next round once")
    result.round2 = sorted(entries, key=lambda entry: entry.position_key)
    return result
