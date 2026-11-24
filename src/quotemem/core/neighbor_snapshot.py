"""Content-free nomination records for persistence, replay and the commit fence.

Builders live in ingestion.nomination; this leaf never imports ingestion.
Dependency lists are the authoritative erasure edges. Repeated identities in
route and display records are local locators, not additional retention edges.
"""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import ConfigDict, Field, model_validator

from quotemem.core.add_read_view import EndedBy
from quotemem.core.entities import PositionKey, ReasonCode
from quotemem.core.identity import ULID
from quotemem.core.review_rounds import RoundNo
from quotemem.core.statement import (
    MAX_JUDGE_HISTORY,
    MAX_JUDGE_TARGETS,
    MAX_JUDGE_TOMBSTONES,
    HexDigest,
)
from quotemem.store.metadata import OpaqueIdSpec, RefSpec, StrictModel

FENCE_LIST = 16
"""Length of each recorded dense and BM25 list."""
Binding = Literal["self", "participant", "sdk", "nominated", "open"]
PartitionMode = Literal["complete", "truncated", "empty_complete", "empty_incomplete"]
TargetKind = Literal["head", "retired", "sibling"]
TargetRoute = Literal["entity", "secondary", "scope_dense", "revises", "corrects", "siblings"]
TombstoneRoute = Literal["exact_digest", "same_entity", "scope"]
CandidateSource = Literal[
    "nominated", "exact_name", "speaker_name", "declaration", "name_bm25", "third_party",
]
_Count = Annotated[int, Field(ge=0)]
_Score = Annotated[float, Field(allow_inf_nan=False)]
_Floor = Annotated[float, Field(ge=-1.0, le=1.0)]
_Sigma = Annotated[list[int], Field(min_length=5, max_length=5)]

class RankedV1(StrictModel):
    """One row of a top-16 list or selection: an object id and its rounded score."""

    model_config = ConfigDict(frozen=True)
    object_id: Annotated[str, OpaqueIdSpec("local")]
    score: _Score


class SelectedRankV1(StrictModel):
    """A selected row's dense and BM25 ranks in its route pool; a zero BM25 has no rank."""

    model_config = ConfigDict(frozen=True)
    object_id: Annotated[str, OpaqueIdSpec("local")]
    dense_rank: Annotated[int, Field(ge=1)]
    bm25_rank: Annotated[int, Field(ge=1)] | None


class RankRecordV1(StrictModel):
    """What the commit fence needs of one RRF selection.

    Attributes:
        selected: Selected object ids in selection order.
        top_dense: The pool's best rows by cosine, at most 16.
        top_bm25: The pool's best rows by positive BM25, at most 16.
        selected_ranks: Each selected row's ranks.
        s_k: The lowest selected RRF, rounded to 6 decimals; None when nothing
            was selected.
    """

    model_config = ConfigDict(frozen=True)
    selected: list[str]
    top_dense: Annotated[list[RankedV1], Field(max_length=FENCE_LIST)]
    top_bm25: Annotated[list[RankedV1], Field(max_length=FENCE_LIST)]
    selected_ranks: list[SelectedRankV1]
    s_k: _Score | None


class EntityRouteV1(StrictModel):
    """Route 1 for the primary slot: its partition P(E) and how much of it was issued.

    Attributes:
        route: Always ``entity``.
        slot: Always 0.
        entity_id: E; None for a new declaration, an unminted participant or an
            open slot.
        mode: ``complete`` (all issued), ``truncated`` (top K), ``empty_complete``
            (no entity yet, so nothing can be missing) or ``empty_incomplete`` (open).
        partition_size: Heads anchored to E at the pin.
        ranks: The selection.
    """

    model_config = ConfigDict(frozen=True)
    route: Literal["entity"]
    slot: Literal[0]
    entity_id: Annotated[str | None, OpaqueIdSpec("local")]
    mode: PartitionMode
    partition_size: _Count
    ranks: RankRecordV1


class SecondaryRouteV1(StrictModel):
    """Route 1 for the secondary slots: the top heads over the union of their partitions.

    The pool excludes heads the primary partition issued; ``mode`` is
    ``complete`` when the whole pool was issued.
    """

    model_config = ConfigDict(frozen=True)
    route: Literal["secondary"]
    slots: list[Annotated[int, Field(ge=1, le=7)]]
    entity_ids: Annotated[list[str], OpaqueIdSpec("local")]
    mode: Literal["complete", "truncated"]
    pool_size: _Count
    ranks: RankRecordV1


class ScopeDenseRouteV1(StrictModel):
    """Route 2: the selected heads with their cosines, and the recorded floor.

    The floor is ``max(0.80, the last selected cosine)`` when the route issued
    its full count, else 0.80.
    """

    model_config = ConfigDict(frozen=True)
    route: Literal["scope_dense"]
    selected: list[RankedV1]
    floor: _Floor


class RevisesReferenceV1(StrictModel):
    """One ``revises`` reference and what it issued.

    Attributes:
        memory_id: The referenced memory.
        revision_id: The retired version an ``earlier`` reference names; None
            for a reference to a current memory.
        issued_memory_id: The memory of the issued head or retired version; None
            when nothing was issued.
        issued_revision_id: The issued head revision or retired version.
    """

    model_config = ConfigDict(frozen=True)
    memory_id: Annotated[str, OpaqueIdSpec("local")]
    revision_id: Annotated[str | None, OpaqueIdSpec("local")]
    issued_memory_id: Annotated[str | None, OpaqueIdSpec("local")]
    issued_revision_id: Annotated[str | None, OpaqueIdSpec("local")]


class Round2IssueV1(StrictModel):
    """One target resolution asked round 2 to issue (``?``), and whether it was.

    Attributes:
        kind: ``head`` for a current head, ``retired`` for a retired version.
        memory_id: Its memory.
        revision_id: The requested revision.
        issued: False when the revision is no longer that kind at the pin (a head
            that changed, or a version that is not readable), when a retired
            version was asked for a member that does not correct history,
            or when the member already holds the wire's 64 targets.
    """

    model_config = ConfigDict(frozen=True)
    kind: Literal["head", "retired"]
    memory_id: Annotated[str, OpaqueIdSpec("local")]
    revision_id: Annotated[str, OpaqueIdSpec("local")]
    issued: bool


class RevisesRouteV1(StrictModel):
    """Route 3: every ``revises`` reference in member order, then every target round 2
    was asked to issue in request order."""

    model_config = ConfigDict(frozen=True)
    route: Literal["revises"]
    references: list[RevisesReferenceV1]
    round2_issue: list[Round2IssueV1]


class CorrectsRouteV1(StrictModel):
    """Route 4: the cue's RRF selection over other heads and the retired versions.

    Object ids are memory ids for heads and revision ids for retired versions;
    ``retired_selected`` lists the retired versions issued as targets.
    """

    model_config = ConfigDict(frozen=True)
    route: Literal["corrects"]
    pool_size: _Count
    retired_selected: list[str]
    ranks: RankRecordV1


class PositionV1(StrictModel):
    """A position key ``(owner_run_id, unit_no, position)`` as a stored record."""

    model_config = ConfigDict(frozen=True)
    owner_run_id: Annotated[ULID, OpaqueIdSpec("local")]
    unit_no: Annotated[int, Field(ge=0, le=65535)]
    position: Annotated[int, Field(ge=0, le=511)]

    @classmethod
    def of(cls, key: PositionKey) -> PositionV1:
        """Record a position key."""
        owner_run_id, unit_no, position = key
        return cls(owner_run_id=owner_run_id, unit_no=unit_no, position=position)

    @property
    def key(self) -> PositionKey:
        """The position key it records."""
        return (self.owner_run_id, self.unit_no, self.position)


class SiblingSelectionV1(StrictModel):
    """One issued sibling and whether it shares an anchor with the member."""

    model_config = ConfigDict(frozen=True)
    position: PositionV1
    same_anchor: bool


class SiblingsRouteV1(StrictModel):
    """Route 6: canonical issued siblings and the full same-anchor pool's RRF order.

    ``same_anchor_rrf`` is content-free nomination-time evidence for cap diagnostics.
    None means an existing recording did not retain the ranks; an empty list records
    an empty pool. It never changes judge input or the nomination selection.
    """

    model_config = ConfigDict(frozen=True)
    route: Literal["siblings"]
    same_anchor_count: _Count
    selected: list[SiblingSelectionV1]
    same_anchor_rrf: Annotated[list[str] | None, OpaqueIdSpec("local")] = Field(
        default=None, exclude_if=lambda value: value is None)

    @model_validator(mode="after")
    def complete_ranking(self) -> Self:
        """A retained ranking covers its full pool once and holds every selected same-anchor
        sibling. The selection is the cap's RRF prefix plus any earlier member the head
        routes would select once committed, so it need not be a prefix."""
        if self.same_anchor_rrf is not None:
            ranked = self.same_anchor_rrf
            selected = {":".join(map(str, row.position.key))
                        for row in self.selected if row.same_anchor}
            if (len(ranked) != self.same_anchor_count or len(set(ranked)) != len(ranked)
                    or not selected <= set(ranked)):
                raise ValueError("sibling_rrf_pool_mismatch")
        return self


class TombstoneSelectionV1(StrictModel):
    """One issued tombstone, its route and its cosine to the member."""

    model_config = ConfigDict(frozen=True)
    tombstone_id: Annotated[str, OpaqueIdSpec("local")]
    route: TombstoneRoute
    cosine: _Floor


class TombstoneFloorsV1(StrictModel):
    """The cosine floors the near-match tombstone routes applied.

    A tombstone whose statement writes letters of other scripts than the member's
    was held to the ``cross_script_*`` floors instead.
    """

    model_config = ConfigDict(frozen=True)
    same_entity: _Floor
    scope: _Floor
    cross_script_same_entity: _Floor
    cross_script_scope: _Floor


class TombstonesRouteV1(StrictModel):
    """Route 7: every issued tombstone, in route order."""

    model_config = ConfigDict(frozen=True)
    route: Literal["tombstones"]
    selected: list[TombstoneSelectionV1]
    floors: TombstoneFloorsV1


RouteRecordV1 = Annotated[
    EntityRouteV1 | SecondaryRouteV1 | ScopeDenseRouteV1 | RevisesRouteV1 | CorrectsRouteV1
    | SiblingsRouteV1 | TombstonesRouteV1,
    Field(discriminator="route"),
]
"""One route record of a nomination, discriminated by ``route``."""


class NominationFlagsV1(StrictModel):
    """The member-level flags.

    Attributes:
        correction_requested: ``revises`` or ``corrects_text`` is set.
        history_correction_requested: An ``earlier`` reference, or the corrects
            route selected a retired version.
        entity_partition_complete: The primary partition is ``complete`` or
            ``empty_complete``.
        external_only_allowed: ``correction_requested`` and a complete primary
            partition.
        revises_unavailable: Some reference reached no readable head or version.
    """

    model_config = ConfigDict(frozen=True)
    correction_requested: bool
    history_correction_requested: bool
    entity_partition_complete: bool
    external_only_allowed: bool
    revises_unavailable: bool


class NominationDependenciesV1(StrictModel):
    """Every row the member input reads, sorted and unique, for erasure and replay."""

    model_config = ConfigDict(frozen=True)
    evidence_ids: Annotated[list[str], RefSpec("evidence", cardinality="many",
        ref_mode="snapshot_locator", scope_rule="same_scope", gc_root=False)]
    revision_ids: Annotated[list[str], RefSpec("revision", cardinality="many",
        ref_mode="snapshot_locator", scope_rule="same_scope", gc_root=False)]
    tombstone_ids: Annotated[list[str], RefSpec("tombstone", cardinality="many",
        ref_mode="snapshot_locator", scope_rule="same_scope", gc_root=False)]
    # Includes the virtual user, which has no Entity row.
    entity_ids: Annotated[list[str], OpaqueIdSpec("local")]
    naming_revision_ids: Annotated[list[str], RefSpec("revision", cardinality="many",
        ref_mode="snapshot_locator", scope_rule="same_scope", gc_root=False)]


class SnapshotPinV1(StrictModel):
    """The pin the member was nominated at."""

    model_config = ConfigDict(frozen=True)
    commit_seq: _Count
    erase_epoch: _Count
    consent_epoch: _Count
    tombstone_seq: _Count
    recall_snapshot_version: _Count


class LexicalRecordV1(StrictModel):
    """The pin's BM25 statistics; ``df`` holds the member's query terms by keyed digest."""

    model_config = ConfigDict(frozen=True)
    doc_count: _Count
    total_length: _Count
    df: dict[str, _Count]


class SnapshotTargetV1(StrictModel):
    """One issued target (``targets[]``)."""

    model_config = ConfigDict(frozen=True)
    handle: str
    kind: TargetKind
    object_id: Annotated[str, OpaqueIdSpec("local")]
    revision_id: Annotated[str | None, OpaqueIdSpec("local")]
    statement_digest: Annotated[HexDigest, OpaqueIdSpec("digest")]
    sigma: _Sigma
    primary_entity_id: Annotated[str | None, OpaqueIdSpec("local")]
    naming: list[str]
    route: TargetRoute


class SnapshotHistoryV1(StrictModel):
    """One context history entry.

    ``head_handle`` is the issued head it is shown beside; ``of_handle`` is that
    head when the entry is an earlier revision of the same memory, and null for
    the head of a memory the chain ended.
    """

    model_config = ConfigDict(frozen=True)
    handle: str
    head_handle: str
    of_handle: str | None
    revision_id: Annotated[str, OpaqueIdSpec("local")]
    ended_by: EndedBy


class SnapshotTombstoneV1(StrictModel):
    """One issued tombstone (``tombstones[]``)."""

    model_config = ConfigDict(frozen=True)
    handle: str
    tombstone_id: Annotated[str, OpaqueIdSpec("local")]
    route: TombstoneRoute


class SnapshotCandidateV1(StrictModel):
    """One entity candidate: an existing entity or a new declaration."""

    model_config = ConfigDict(frozen=True)
    handle: str
    entity_id: Annotated[str | None, OpaqueIdSpec("local")]
    mint_key: str | None
    commit_seq: _Count | None
    source: CandidateSource


class SnapshotSlotV1(StrictModel):
    """One slot: its entity, and for a judged slot its candidates."""

    model_config = ConfigDict(frozen=True)
    slot: Annotated[int, Field(ge=0, le=7)]
    binding: Binding
    entity_handle: str | None
    nominated_handle: str | None
    candidates: list[SnapshotCandidateV1]
    catalog_complete: bool | None


class SnapshotEntityFactsV1(StrictModel):
    """One existing entity the member input names, with the head revisions its entry shows
    as facts, newest first."""

    model_config = ConfigDict(frozen=True)
    handle: str
    entity_id: Annotated[str, OpaqueIdSpec("local")]
    facts: list[str]


class SnapshotQuoteV1(StrictModel):
    """One located quote span of the member (handle C)."""

    model_config = ConfigDict(frozen=True)
    handle: str
    evidence_id: Annotated[str, OpaqueIdSpec("local")]
    byte_start: _Count
    byte_end: _Count


class SnapshotSourceV1(StrictModel):
    """One shown message (handle Q): its parts in part order, and whether it was cited."""

    model_config = ConfigDict(frozen=True)
    handle: str
    evidence_ids: Annotated[list[str], OpaqueIdSpec("local")]
    cited: bool


class NeighborSnapshotV1(StrictModel):
    """The content-free record of one member's nomination in one round.

    It is the input of the member's judge child run: with the immutable rows it
    names it rebuilds the member input, and its route records let the commit
    fence decide whether a later row would have been nominated. It
    holds ids, handles, ranks, scores, σ, unkeyed statement digests and keyed
    term digests only, never a statement, quote, name, mention or term.

    ``reentry_reason_code`` is the candidate's code read when the owner selected
    the member for re-entry, null for a member that did not re-enter; it selects
    W (f), and replay cannot read it back from the row once the owner settles.
    ``wake_digest`` is the wake key of the member's rebuild in this
    round, null for a member that did not re-enter and for a round-2
    member built with a pick or issued targets, which W never covers.
    """

    model_config = ConfigDict(frozen=True)
    schema_version: Literal[1]
    policy_digest: Annotated[HexDigest, OpaqueIdSpec("digest")]
    owner_run_id: Annotated[ULID, OpaqueIdSpec("local")]
    round: RoundNo
    position: PositionV1
    member_input_digest: Annotated[str, OpaqueIdSpec("digest")]
    wake_digest: Annotated[str | None, OpaqueIdSpec("digest")]
    reentry_reason_code: ReasonCode | None
    pin: SnapshotPinV1
    embedding_space: str
    lexical: LexicalRecordV1
    routes: list[RouteRecordV1]
    targets: Annotated[list[SnapshotTargetV1], Field(max_length=MAX_JUDGE_TARGETS)]
    history: Annotated[list[SnapshotHistoryV1], Field(max_length=MAX_JUDGE_HISTORY)]
    tombstones: Annotated[list[SnapshotTombstoneV1], Field(max_length=MAX_JUDGE_TOMBSTONES)]
    entities: list[SnapshotSlotV1]
    entity_facts: list[SnapshotEntityFactsV1]
    quotes: list[SnapshotQuoteV1]
    sources: list[SnapshotSourceV1]
    flags: NominationFlagsV1
    dependencies: NominationDependenciesV1

    @model_validator(mode="after")
    def wake_has_its_code(self) -> Self:
        """Refuse a wake key without the code that selected its W (f)."""
        if self.wake_digest is not None and self.reentry_reason_code is None:
            raise ValueError("A wake digest needs the reason code it was computed with")
        return self


