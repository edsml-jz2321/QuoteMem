"""Bounded work snapshots and frozen finalize provenance from plan.md §3 and §9."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from aml.core.identity import ULID, ArtifactId, ScopeId
from aml.core.semantic import ScopeKind, TrustTier
from aml.core.source_participants import SourceParticipantV1
from aml.core.technical_types import (
    ContentDigest,
    ExternalId,
    GoverningExtractionV1,
    HistoricalTimestamp,
    Integrity,
    MessageRangeKeyV1,
    MessageRangeKeyV2,
    NonNegativeInt,
    PositiveInt,
    RangeKeyV1,
    Timestamp,
)
from aml.store.metadata import OpaqueIdSpec, RefSpec, StrictModel

ConditionalRunRef = Annotated[
    ULID, RefSpec("run", ref_mode="snapshot_locator", gc_root="while_owner_nonterminal")
]
ConditionalEvidenceRef = Annotated[
    ULID, RefSpec("evidence", ref_mode="snapshot_locator", gc_root="while_owner_nonterminal")
]


class PlanMessageRangeKeyV1(MessageRangeKeyV1):
    """Retain the Evidence coordinate while its finalize plan owner is nonterminal."""

    evidence_id: ConditionalEvidenceRef


class PlanMessageRangeKeyV2(MessageRangeKeyV2):
    """Retain a receipt-aware coordinate under the same frozen plan owner."""

    evidence_id: ConditionalEvidenceRef


class CompactWorkSnapshotV1(StrictModel):
    """Store a bounded digest of one stable ordered input stream."""

    work_generation: NonNegativeInt
    ordered_input_count: NonNegativeInt
    ordered_input_digest: ContentDigest
    storage_plan_digest: Integrity
    first_key: RangeKeyV1 | None = None
    last_key: RangeKeyV1 | None = None
    supersedes_root_work_digest: ContentDigest | None = None
    committed_prefix_digest: ContentDigest | None = None

    @model_validator(mode="after")
    def validate_snapshot(self) -> Self:
        """Require complete boundaries and successor proof without input arrays."""
        boundaries = self.first_key is not None and self.last_key is not None
        if (self.ordered_input_count > 0) != boundaries:
            raise ValueError("Nonempty snapshots require both boundary keys")
        if self.ordered_input_count == 0 and (self.first_key or self.last_key):
            raise ValueError("Empty snapshots cannot contain boundary keys")
        successor = self.supersedes_root_work_digest is not None
        prefix = self.committed_prefix_digest is not None
        if successor != prefix or successor != (self.work_generation > 0):
            raise ValueError("Only successor generations require both predecessor digests")
        return self


class SessionWorkSnapshotV1(CompactWorkSnapshotV1, GoverningExtractionV1):
    """Freeze the leased session checkpoint and governing extraction release."""

    work_kind: Literal["session"]
    checkpoint_time: Timestamp
    input_high_watermark: NonNegativeInt
    cursor_start_digest: ContentDigest


FinalizeMetadataRef = Annotated[
    ArtifactId,
    RefSpec(
        "artifact", ref_mode="snapshot_locator", scope_rule="same_scope",
        gc_root="while_owner_nonterminal", artifact_kinds=("decision_set",),
    ),
]


class FinalizeSourceHeaderV1(StrictModel):
    """Fixed source identity and eligibility; never source text or embedding bytes."""

    object_id: ConditionalEvidenceRef
    tenant_id: Annotated[str, Field(min_length=1, max_length=63), OpaqueIdSpec("local")]
    scope_kind: ScopeKind
    scope_id: Annotated[ScopeId, OpaqueIdSpec("local")]
    session_id: ExternalId
    request_id: ExternalId
    source_participant: SourceParticipantV1 | None = Field(
        default=None, exclude_if=lambda value: value is None,
    )
    ordinal: NonNegativeInt
    part_no: NonNegativeInt
    byte_start: NonNegativeInt
    byte_end: NonNegativeInt
    source_observed_at: HistoricalTimestamp | None
    ingested_at: Timestamp
    ingested_commit_seq: NonNegativeInt
    trust_tier: TrustTier
    state: Literal["active", "invalidated", "expired", "erased"]
    expires_at: Timestamp | None
    content_hash: Integrity

    @model_validator(mode="after")
    def validate_window(self) -> Self:
        """Metadata preserves the original physical interval without reading its body."""
        if self.byte_end < self.byte_start:
            raise ValueError("Source header has a reversed byte interval")
        return self


class FinalizeSourceInventoryMemberV1(StrictModel):
    """One original full-row HMAC beside independently frozen exact metadata."""

    index: NonNegativeInt
    key: Annotated[
        PlanMessageRangeKeyV1 | PlanMessageRangeKeyV2, Field(discriminator="key_kind"),
    ]
    row_binding_digest: ContentDigest
    header: FinalizeSourceHeaderV1

    @model_validator(mode="after")
    def validate_coordinate(self) -> Self:
        """Bind receipt coordinates to the same eligible physical source part."""
        if self.header.state != "active" or (
            self.key.evidence_id, self.key.session_id, self.key.request_id,
            self.key.ordinal, self.key.part_no,
        ) != (
            self.header.object_id, self.header.session_id, self.header.request_id,
            self.header.ordinal, self.header.part_no,
        ):
            raise ValueError("Frozen source coordinate differs from its eligible header")
        return self


class FinalizeSourceInventoryBlockV1(StrictModel):
    """A bounded original source inventory; only owner input selection grants authority."""

    finalize_source_inventory: Literal[1]
    owner_run_id: ConditionalRunRef
    finalize_cycle: PositiveInt
    freeze_time: Timestamp
    receipt_high_watermark: NonNegativeInt
    root_input_heads_digest: ContentDigest
    previous_ref: FinalizeMetadataRef | None
    block_no: NonNegativeInt
    records: Annotated[list[FinalizeSourceInventoryMemberV1], Field(max_length=128)]
    record_count: NonNegativeInt
    total_record_count: PositiveInt
    cumulative_digest: ContentDigest
    is_head: bool
    base_ref: FinalizeMetadataRef | None = Field(
        default=None, exclude_if=lambda value: value is None,
    )
    base_count: NonNegativeInt = Field(default=0, exclude_if=lambda value: value == 0)
    removed_ids: list[ConditionalEvidenceRef] = Field(
        default_factory=list, exclude_if=lambda value: not value,
    )

    @model_validator(mode="after")
    def validate_block(self) -> Self:
        """Keep local counts, source eligibility and original ordinals exact."""
        if (self.block_no == 0) != (self.previous_ref is None) or (
            self.record_count != len(self.records)
            or self.total_record_count < self.record_count + (
                self.base_count - len(self.removed_ids) if self.is_head else 0
            )
        ):
            raise ValueError("Source inventory has inconsistent predecessor or counts")
        retained = self.base_count - len(self.removed_ids)
        if (self.base_ref is None) != (self.base_count == 0) or retained < 0 or (
            len(set(self.removed_ids)) != len(self.removed_ids)
        ) or (
            not self.records and not (self.is_head and retained)
        ):
            raise ValueError("Source inventory requires exact base and removal coordinates")
        own_count = self.total_record_count - (retained if self.is_head else 0)
        if self.block_no == 0 and own_count != self.record_count:
            raise ValueError("First source inventory block must start at zero")
        first = own_count - self.record_count
        if [value.index for value in self.records] != list(range(first, own_count)):
            raise ValueError("Source inventory indices must be contiguous")
        if any(value.header.expires_at is not None and value.header.expires_at <= self.freeze_time
               for value in self.records):
            raise ValueError("Frozen inventory contains an already expired source")
        if any(value.header.ingested_commit_seq > self.receipt_high_watermark
               for value in self.records):
            raise ValueError("Frozen inventory contains a source beyond its receipt boundary")
        return self


class FinalizeExpiredSourceV1(StrictModel):
    """One exact frozen deadline observed through the unchanged source header."""

    evidence_id: ConditionalEvidenceRef
    original_binding_digest: ContentDigest
    header_digest: Integrity
    expires_at: Timestamp


class FinalizeSourceTTLDispositionV1(StrictModel):
    """Account for one affected original unit without replacing its ordinal or history."""

    ordinal: NonNegativeInt
    unit_record_digest: Integrity
    original_member_count: PositiveInt
    original_member_digest: ContentDigest
    retained_member_ids: Annotated[list[ConditionalEvidenceRef], Field(max_length=2000)]
    retained_member_digest: ContentDigest
    expired_sources: Annotated[list[FinalizeExpiredSourceV1], Field(max_length=2000)]
    omitted_member_ids: Annotated[list[ConditionalEvidenceRef], Field(max_length=2000)]
    disposition: Literal["projected", "no_output"]

    @model_validator(mode="after")
    def validate_partition(self) -> Self:
        """Require a complete disjoint partition; a consumer proves original order and peers."""
        expired = [value.evidence_id for value in self.expired_sources]
        combined = [*self.retained_member_ids, *expired, *self.omitted_member_ids]
        if not expired and not self.omitted_member_ids:
            raise ValueError("TTL disposition requires actual expiry or a proved message peer")
        if len(combined) != self.original_member_count or len(set(combined)) != len(combined):
            raise ValueError("TTL disposition must partition every original member exactly once")
        if (self.disposition == "no_output") != (not self.retained_member_ids):
            raise ValueError("Only complete source loss records explicit no output")
        return self


class FinalizeSourceTTLSelectionV1(StrictModel):
    """Selected complete TTL accounting under the same original owner and plan."""

    owner_run_id: ConditionalRunRef
    input_ref: Annotated[
        ArtifactId, RefSpec("artifact", ref_mode="snapshot_locator", artifact_kinds=("run_input",)),
    ]
    plan_ref: Annotated[
        ArtifactId,
        RefSpec("artifact", ref_mode="snapshot_locator", artifact_kinds=("finalize_plan",)),
    ]
    source_inventory_ref: FinalizeMetadataRef
    unit_inventory_ref: FinalizeMetadataRef
    previous_result_ref: Annotated[
        ArtifactId,
        RefSpec("artifact", ref_mode="snapshot_locator", artifact_kinds=("result_index",)),
    ]
    previous_projection_ref: FinalizeMetadataRef | None
    head_ref: FinalizeMetadataRef
    count: PositiveInt
    digest: ContentDigest
    observed_at: Timestamp
    original_unit_count: PositiveInt
    retained_member_count: NonNegativeInt
    expired_member_count: PositiveInt
    omitted_member_count: NonNegativeInt
    no_output_unit_count: NonNegativeInt

    @model_validator(mode="after")
    def validate_counts(self) -> Self:
        """The affected subset and explicit empty units cannot exceed the original plan."""
        if not self.no_output_unit_count <= self.count <= self.original_unit_count:
            raise ValueError("TTL selection has inconsistent original and affected unit counts")
        return self


class FinalizeSourceTTLBlockV1(StrictModel):
    """Bounded source-loss decisions tied to the exact previous owner selection."""

    finalize_source_ttl: Literal[1]
    owner_run_id: ConditionalRunRef
    input_ref: Annotated[
        ArtifactId, RefSpec("artifact", ref_mode="snapshot_locator", artifact_kinds=("run_input",)),
    ]
    plan_ref: Annotated[
        ArtifactId,
        RefSpec("artifact", ref_mode="snapshot_locator", artifact_kinds=("finalize_plan",)),
    ]
    source_inventory_ref: FinalizeMetadataRef
    unit_inventory_ref: FinalizeMetadataRef
    previous_result_ref: Annotated[
        ArtifactId,
        RefSpec("artifact", ref_mode="snapshot_locator", artifact_kinds=("result_index",)),
    ]
    previous_projection_ref: FinalizeMetadataRef | None
    observed_at: Timestamp
    previous_ref: FinalizeMetadataRef | None
    block_no: NonNegativeInt
    records: Annotated[list[FinalizeSourceTTLDispositionV1], Field(min_length=1, max_length=32)]
    record_count: PositiveInt
    total_record_count: PositiveInt
    cumulative_digest: ContentDigest
    is_head: bool

    @model_validator(mode="after")
    def validate_block(self) -> Self:
        """Local shape proves neither missing sources nor permission to replace a unit."""
        if (self.block_no == 0) != (self.previous_ref is None) or (
            self.record_count != len(self.records) or self.total_record_count < self.record_count
        ):
            raise ValueError("TTL block has inconsistent predecessor or counts")
        if self.block_no == 0 and self.total_record_count != self.record_count:
            raise ValueError("First TTL block must start at zero")
        ordinals = [value.ordinal for value in self.records]
        if ordinals != sorted(set(ordinals)):
            raise ValueError("TTL block must preserve distinct original unit order")
        if any(source.expires_at > self.observed_at
               for record in self.records for source in record.expired_sources):
            raise ValueError("TTL block cannot dispose a source before its deadline")
        return self
