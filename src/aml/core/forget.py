"""Content-free memory-forget receipts (D.3 and E.2.4).

These values describe a verified operation; parsing them confers no write or
completion authority. The Writer's final scan must establish their bindings.
All content-derived receipt digests are tenant-keyed, including plan and scan.
"""

from __future__ import annotations

from typing import Annotated, Literal, Self, cast

from pydantic import BeforeValidator, ConfigDict, Field, model_validator

from aml.core.identity import ArtifactId
from aml.core.technical_types import ContentDigest, NonNegativeInt
from aml.store.metadata import RefSpec, StrictModel

MAX_FORGET_TOMBSTONES = 256
MAX_FORGET_SOURCES = 2_000


def _array(value: object) -> object:
    """Accept JSON arrays without relaxing strict element validation."""
    return tuple(cast(list[object], value)) if isinstance(value, list) else value


class ForgetCountsV1(StrictModel):
    """Physical deletions and separate NL compensation/withholding effects."""

    model_config = ConfigDict(frozen=True)
    item: NonNegativeInt = 0
    revision: NonNegativeInt = 0
    link: NonNegativeInt = 0
    candidate: NonNegativeInt = 0
    run: NonNegativeInt = 0
    recall: NonNegativeInt = 0
    artifact: NonNegativeInt = 0
    control: NonNegativeInt = 0
    entity_name: NonNegativeInt = 0
    entity_erased: NonNegativeInt = 0
    dependent_requoted: NonNegativeInt = 0
    candidate_expired: NonNegativeInt = 0
    evidence_rows_withheld: Annotated[int, Field(ge=0, le=MAX_FORGET_SOURCES)] = 0
    evidence_spans_withheld: NonNegativeInt = 0
    tombstones_written: Annotated[int, Field(ge=0, le=MAX_FORGET_TOMBSTONES)] = 0


class ForgetWithheldSourceV1(StrictModel):
    """Keyed source identity and span-set binding, without source coordinates."""

    model_config = ConfigDict(frozen=True)
    source_token: ContentDigest
    spans_digest: ContentDigest


class ForgetInternalRetentionV1(StrictModel):
    """Disclose exactly the internal no-relearn retention and its end condition."""

    model_config = ConfigDict(frozen=True)
    kind: Literal["forget_tombstone"] = "forget_tombstone"
    purpose: Literal["block_relearning"] = "block_relearning"
    readable: Literal[False] = False
    tombstones: Annotated[int, Field(ge=0, le=MAX_FORGET_TOMBSTONES)]
    tombstones_created: Annotated[int, Field(ge=0, le=MAX_FORGET_TOMBSTONES)]
    tombstone_set_digest: ContentDigest
    content: Literal["statement_text_vector_anchors"] = "statement_text_vector_anchors"
    judge_provider_exposure: Literal["near_match_only"] = "near_match_only"
    provider_records: Literal["retained_unreadable"] = "retained_unreadable"
    retention_end: Literal["scope_purge_or_all_sources_privacy_erased"] = (
        "scope_purge_or_all_sources_privacy_erased"
    )

    @model_validator(mode="after")
    def bounded_creation(self) -> Self:
        if self.tombstones_created > self.tombstones:
            raise ValueError("Created tombstones exceed the selected retained set")
        return self


class _ForgetReceiptBinding(StrictModel):
    """The immutable reservation shared by the prepare and final receipts."""

    model_config = ConfigDict(frozen=True)
    schema_version: Literal[3] = 3
    target_kind: Literal["memory"] = "memory"
    target_digest: ContentDigest
    worker_token: ContentDigest
    maintenance_handoff_digest: ContentDigest | None
    plan_digest: ContentDigest
    erase_epoch: NonNegativeInt
    tombstone_seq: NonNegativeInt
    intended_generation: NonNegativeInt
    intended_scope_seq: NonNegativeInt
    withheld_sources: Annotated[
        tuple[ForgetWithheldSourceV1, ...], Field(max_length=MAX_FORGET_SOURCES),
        BeforeValidator(_array),
    ]

    @model_validator(mode="after")
    def ordered_sources(self) -> Self:
        tokens = tuple(source.source_token for source in self.withheld_sources)
        if tokens != tuple(sorted(set(tokens))):
            raise ValueError("Withheld sources must be unique and token-ordered")
        return self


class ForgetPrepareReceiptV3(_ForgetReceiptBinding):
    """A bound reservation, never evidence of completed deletion."""

    receipt_kind: Literal["erasure_prepare"] = "erasure_prepare"
    status: Literal["prepared"] = "prepared"


class ForgetReceiptV3(_ForgetReceiptBinding):
    """D.3 final verification with E.2.4's explicit residual and copy disclosures."""

    receipt_kind: Literal["erasure_final"] = "erasure_final"
    status: Literal["verified"] = "verified"
    prepare_ref: Annotated[
        ArtifactId, RefSpec("artifact", scope_rule="global", artifact_kinds=("receipt",)),
    ]
    counts: ForgetCountsV1
    final_scan_digest: ContentDigest
    readable_residual_count: Literal[0] = 0
    internal_retention: ForgetInternalRetentionV1
    not_covered: Annotated[
        tuple[
            Literal["backups"], Literal["existing_exports"], Literal["host_session_logs"],
            Literal["provider_copies"], Literal["replicas"], Literal["wal"],
        ], BeforeValidator(_array),
    ] = ("backups", "existing_exports", "host_session_logs", "provider_copies", "replicas", "wal")

    @model_validator(mode="after")
    def consistent_counts(self) -> Self:
        if self.counts.tombstones_written != self.internal_retention.tombstones_created:
            raise ValueError("Written tombstones differ from the retention disclosure")
        if self.counts.evidence_rows_withheld > len(self.withheld_sources):
            raise ValueError("Withheld row count exceeds the bound source set")
        if (self.counts.evidence_rows_withheld == 0) != (
            self.counts.evidence_spans_withheld == 0
        ) or self.counts.evidence_spans_withheld < self.counts.evidence_rows_withheld:
            raise ValueError("Withheld row and span counts disagree")
        return self
