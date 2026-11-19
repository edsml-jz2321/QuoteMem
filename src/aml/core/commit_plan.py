"""Records of an Add owner's transactions: journals, publish plans and inputs (C.12, C.13; I10).

Every C.12 transaction of an Add owner writes one create-once
`CommitJournalV1` (``write_set:commit_journal``): each settlement chunk and
each rejection chunk of a round, and each formal component (store-Q5). Its
`journal_key` binds the owner, the round, the transaction kind and its
positions, so a replay finds the journal before it reads any row and a crash
resumes at the first position no journal covers. The journal names the rows
the transaction wrote and deleted and each member's resolution digest; the
zero-effect proof of an interrupted publish reads it (C.13).

A formal component is committed by one ``publish`` child of the owner (A.3.10).
Its ``run_input:publish:commit`` (``aml.core.run_inputs.PublishCommitRunInputV1``)
names the component; its `CommitPlanV1` (``publish_plan``) is written before
the transaction and fixes what the transaction writes: the members' resolution
digests and the heads it compares. Settlement (B.8, S3c) hands its positions to
the chunked settlement transaction as `SettledPositionV1` records; a B.8 #3
duplicate of a representative that goes to review travels with that
representative as a `DuplicatePositionV1`.

Every record is content-free: positions, codes, ids and digests, never a
statement, quote or name.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Annotated, Final, Literal, Self

from pydantic import Field, model_validator

from aml.core.entities import EntityRef, MemoryLifecycle, PositionKey, QuoteSpanV1, ReasonCode
from aml.core.identity import ULID, ArtifactId, ContentBindingDigest, IntegrityDigest
from aml.core.review_rounds import RoundNo
from aml.core.statement import HexDigest
from aml.store.metadata import OpaqueIdSpec, StrictModel

CommitTxn = Literal["settlement", "rejection", "component"]
"""The C.12 transaction kinds of an Add owner: settlement and rejection chunks, components."""
JOURNAL_PURPOSE: Final = "commit-journal-v1"
"""The binding purpose of `journal_key` (store-Q5)."""
JournalKind = Literal["item", "revision", "link", "candidate", "evidence", "entity", "entity_name"]
"""The row kinds an Add owner's transaction writes or deletes."""
ContentBinder = Callable[[str, object], str]
"""The workspace's keyed content binder, ``Writer.bind_content``."""

_Count = Annotated[int, Field(ge=0)]
_Local = OpaqueIdSpec("local")
_Digest = OpaqueIdSpec("digest")


def key_positions(txn: CommitTxn, positions: Sequence[PositionKey]) -> tuple[PositionKey, ...]:
    """Return the positions a journal key binds (store-Q5).

    Args:
        txn: The transaction kind.
        positions: The transaction's positions; a chunk's in canonical order
            ``(σ, position_key)``.

    Returns:
        A chunk's first and last positions in canonical order (C.12's form), or
        a component's members sorted by position key (C.13's form).

    Raises:
        ValueError: If there are no positions, or a component lists one twice.
    """
    if not positions:
        raise ValueError("A transaction journal covers at least one position")
    if txn == "component":
        if len(set(positions)) != len(positions):
            raise ValueError("A component lists each member once")
        return tuple(sorted(positions))
    return (positions[0], positions[-1])


def journal_key(
    bind: ContentBinder, *, owner_run_id: str, round_no: RoundNo, txn: CommitTxn,
    positions: Sequence[PositionKey],
) -> ContentBindingDigest:
    """Return ``bind_content("commit-journal-v1", [owner_run_id, round, txn, positions])``.

    Args:
        bind: The workspace's keyed content binder.
        owner_run_id: The Add owner.
        round_no: The round; settlement chunks are round 1.
        txn: The transaction kind.
        positions: As `key_positions` takes them.

    Returns:
        The journal's key, also its registry marker key (store-Q19).

    Raises:
        ValueError: For a settlement chunk outside round 1, or invalid positions.
    """
    if txn == "settlement" and round_no != 1:
        raise ValueError("Settlement chunks are journaled in round 1")
    bound = [list(position) for position in key_positions(txn, positions)]
    return ContentBindingDigest(bind(JOURNAL_PURPOSE, [owner_run_id, round_no, txn, bound]))


class JournalRowV1(StrictModel):
    """One row a journaled transaction wrote or deleted, as a content-free locator."""

    kind: JournalKind
    object_id: Annotated[ULID, _Local]


class JournalMemberV1(StrictModel):
    """One position a journaled transaction settled, with its resolution's digest.

    Attributes:
        position_key: The position.
        resolution_digest: The integrity digest of what the transaction wrote for
            it: its member resolution, settlement record or touch.
    """

    position_key: PositionKey
    resolution_digest: Annotated[IntegrityDigest, _Digest]


def _sorted_rows(rows: Sequence[JournalRowV1]) -> bool:
    keys = [(row.kind, row.object_id) for row in rows]
    return keys == sorted(set(keys))


class CommitJournalV1(StrictModel):
    """The create-once journal of one C.12 transaction (``write_set:commit_journal``; C.13).

    Attributes:
        journal_key: `journal_key` of the transaction; the variant's marker key.
        owner_run_id: The Add owner.
        round: The round; 1 for settlement.
        txn: The transaction kind.
        commit_seq: The transaction's ``commit_seq``.
        members: Every position the transaction wrote, in canonical order.
        written: Every row it wrote, sorted by kind and id.
        deleted: Every row it deleted, sorted by kind and id.
    """

    journal_key: Annotated[ContentBindingDigest, _Digest]
    owner_run_id: Annotated[ULID, _Local]
    round: RoundNo
    txn: CommitTxn
    commit_seq: _Count
    members: Annotated[list[JournalMemberV1], Field(min_length=1)]
    written: list[JournalRowV1]
    deleted: list[JournalRowV1]

    @model_validator(mode="after")
    def complete(self) -> Self:
        """Keep each position once and the row lists sorted and unique."""
        keys = [member.position_key for member in self.members]
        if len(set(keys)) != len(keys):
            raise ValueError("A journal names each position once")
        if not _sorted_rows(self.written) or not _sorted_rows(self.deleted):
            raise ValueError("Journal rows are sorted by kind and id and unique")
        if self.txn == "settlement" and self.round != 1:
            raise ValueError("Settlement chunks are journaled in round 1")
        return self


class HeadExpectationV1(StrictModel):
    """An Item's head and lifecycle as a producer read them at its pin (C.11.2 #3, B.8 #2)."""

    memory_id: Annotated[ULID, _Local]
    current_revision_id: Annotated[ULID, _Local]
    lifecycle_state: MemoryLifecycle


class PlannedMemberV1(StrictModel):
    """One member of a component as its plan fixes it: its code and resolution digest."""

    position_key: PositionKey
    code: ReasonCode
    resolution_digest: Annotated[IntegrityDigest, _Digest]


class CommitPlanV1(StrictModel):
    """What one component commit writes, fixed before its transaction (``publish_plan``).

    The ``publish`` child writes it create-once after its pre-lock fence (C.11.2
    #1) and before the transaction; recovery that finds a plan without its
    journal proves the child wrote nothing and resolves the members again
    behind the fence (C.13).

    Attributes:
        journal_key: The component's journal key.
        owner_run_id: The Add owner.
        round: The round.
        fence_commit_seq: The pin s1 the pre-lock fence read.
        members: The component's members in canonical order.
        heads: Every head the transaction compares (C ∪ D ∪ S ∪ X and the
            members' moved heads), sorted by memory id.
    """

    journal_key: Annotated[ContentBindingDigest, _Digest]
    owner_run_id: Annotated[ULID, _Local]
    round: RoundNo
    fence_commit_seq: _Count
    members: Annotated[list[PlannedMemberV1], Field(min_length=1)]
    heads: list[HeadExpectationV1]

    @model_validator(mode="after")
    def complete(self) -> Self:
        """Keep each member once and the heads sorted and unique."""
        keys = [member.position_key for member in self.members]
        if len(set(keys)) != len(keys):
            raise ValueError("A plan names each member once")
        heads = [head.memory_id for head in self.heads]
        if heads != sorted(set(heads)):
            raise ValueError("Planned heads are sorted by memory id and unique")
        return self


SettledCode = Literal[
    "denied_by_forget", "restatement_satisfied", "merged_support", "duplicate_in_add",
    "support_budget_exceeded", "identity_unresolved",
]
"""The codes a settlement transaction writes: B.8 #1–#3, and B.7.3's unresolved primaries."""


class SettledPositionV1(StrictModel):
    """One candidate the settlement transaction writes (B.8, B.7.3; C.12; W3-RT-9).

    S3c builds it after writing the position's ``ProposalV2``; the settlement
    transaction writes the position's candidate row, its withheld spans and
    the expiries they cause.

    Attributes:
        position_key: The position.
        extraction_ref: Its unit's ``llm_output``.
        proposal_ref: Its ``ProposalV2``.
        statement_digest: Its statement's digest.
        entity_id: Its primary's entity at the settlement pin; None when there is
            none: always for ``identity_unresolved``, and for a #3 duplicate that
            does not fit (``support_budget_exceeded``) whose primary is
            unresolved, an unminted declaration or a participant without a row;
            every other code has one.
        evidence_ids: The Evidence its quotes cite, sorted and unique.
        code: ``denied_by_forget`` (#1, or a duplicate mirroring a #1
            representative), ``restatement_satisfied`` (#2), ``merged_support``
            (#2b: the transaction merges ``groups`` into an unreadable head),
            ``duplicate_in_add`` (#3 of a settled #2 or #2b representative),
            ``support_budget_exceeded`` (#3, groups that do not fit) or
            ``identity_unresolved`` (B.7.3, pending).
        memory_id: The memory that already states the fact: the #2 or #2b head,
            or the memory a duplicate's representative holds.
        head: The #2 or #2b head as settlement read it; the transaction
            compares it and a moved head stops the owner with ``target_changed``.
        withheld: The located spans of its admitted groups that the transaction
            withholds as ``restated_forgotten`` (``denied_by_forget`` only).
        groups: The admitted groups a #2b merge adds (``merged_support`` only).
    """

    position_key: PositionKey
    extraction_ref: ArtifactId
    proposal_ref: ArtifactId
    statement_digest: HexDigest
    entity_id: EntityRef | None
    evidence_ids: Annotated[list[ULID], Field(min_length=1)]
    code: SettledCode
    memory_id: ULID | None = None
    head: HeadExpectationV1 | None = None
    withheld: list[QuoteSpanV1] = Field(default_factory=list[QuoteSpanV1])
    groups: list[list[QuoteSpanV1]] = Field(default_factory=list[list[QuoteSpanV1]])

    @model_validator(mode="after")
    def complete(self) -> Self:
        """Require what each code needs, and only that."""
        if self.evidence_ids != sorted(set(self.evidence_ids)):
            raise ValueError("A position's Evidence is sorted and unique")
        holds = self.code in ("restatement_satisfied", "merged_support", "duplicate_in_add")
        if holds != (self.memory_id is not None):
            raise ValueError("Exactly a restatement or a duplicate names the memory holding it")
        if (self.code in ("restatement_satisfied", "merged_support")) != (
            self.head is not None
        ) or (self.head is not None and self.head.memory_id != self.memory_id):
            raise ValueError("Exactly a restatement compares the head that holds it")
        if (self.code == "merged_support") != bool(self.groups) or any(
            span.evidence_id not in self.evidence_ids for group in self.groups for span in group
        ):
            raise ValueError("Exactly a restoring merge adds its own groups")
        if bool(self.withheld) and self.code != "denied_by_forget":
            raise ValueError("Only a forgotten restatement withholds its spans")
        if self.code == "identity_unresolved" and self.entity_id is not None:
            raise ValueError("An unresolved primary has no resolved entity")
        if self.entity_id is None and self.code not in (
            "identity_unresolved", "support_budget_exceeded",
        ):
            raise ValueError("Only an unresolved primary or an unfit duplicate "
                             "lacks a resolved entity")
        if any(span.evidence_id not in self.evidence_ids for span in self.withheld):
            raise ValueError("A withheld span lies in the position's own Evidence")
        return self


class DuplicatePositionV1(StrictModel):
    """A B.8 #3 duplicate of a representative that goes to review (B.8 #3; C.12).

    Its groups joined the representative's before review. Its candidate row is
    written in the representative's transaction: ``duplicate_in_add`` with the
    representative's memory when that position's outcome is formal or
    unchanged, else the representative's code.

    Attributes:
        position_key: The duplicate.
        representative: The position it joined.
        extraction_ref: Its unit's ``llm_output``.
        proposal_ref: Its ``ProposalV2``.
        statement_digest: Its statement's digest.
        entity_id: Its primary at the settlement pin.
        evidence_ids: The Evidence its quotes cite, sorted and unique.
    """

    position_key: PositionKey
    representative: PositionKey
    extraction_ref: ArtifactId
    proposal_ref: ArtifactId
    statement_digest: HexDigest
    entity_id: EntityRef | None
    evidence_ids: Annotated[list[ULID], Field(min_length=1)]

    @model_validator(mode="after")
    def complete(self) -> Self:
        """Keep the Evidence sorted and unique, and the representative another position."""
        if self.evidence_ids != sorted(set(self.evidence_ids)):
            raise ValueError("A position's Evidence is sorted and unique")
        if self.representative == self.position_key:
            raise ValueError("A duplicate's representative is another position")
        return self
