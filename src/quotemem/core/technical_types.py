"""Shared bounded coordinates and binding leaves for technical schemas."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, TypeAdapter

from quotemem.core.identity import ULID, ArtifactId, ContentBindingDigest, IntegrityDigest
from quotemem.core.semantic import HistoricalTimestamp as HistoricalTimestamp
from quotemem.core.semantic import SystemTimestamp
from quotemem.store.metadata import OpaqueIdSpec, RefSpec, StrictModel

NonNegativeInt = Annotated[int, Field(ge=0)]
PositiveInt = Annotated[int, Field(gt=0)]
Timestamp = SystemTimestamp
ContentDigest = Annotated[ContentBindingDigest, OpaqueIdSpec("digest")]
Integrity = Annotated[IntegrityDigest, OpaqueIdSpec("digest")]
SessionHash = Annotated[str, Field(pattern=r"^[A-Za-z0-9_-]{22}$"), OpaqueIdSpec("digest")]
LocalId = Annotated[str, Field(min_length=1, max_length=1024), OpaqueIdSpec("local")]
ExternalId = Annotated[str, Field(min_length=1, max_length=1024), OpaqueIdSpec("external")]
PolicyReleaseRef = Annotated[
    ArtifactId, RefSpec("artifact", scope_rule="global", artifact_kinds=("policy_release",))
]
EvidenceSnapshotRef = Annotated[
    ULID, RefSpec("evidence", ref_mode="snapshot_locator", target_forget_action="delete_owner")
]
RunSnapshotRef = Annotated[
    ULID, RefSpec("run", ref_mode="snapshot_locator", target_forget_action="retain_locator")
]


class MessageRangeKeyV1(StrictModel):
    """Freeze a message coordinate in the source-time order."""

    key_kind: Literal["message"]
    session_event_time: HistoricalTimestamp | None
    session_id: ExternalId
    request_id: ExternalId
    chunk_index: NonNegativeInt | None
    chunk_event_time: HistoricalTimestamp | None
    ordinal: NonNegativeInt
    part_no: NonNegativeInt
    evidence_id: EvidenceSnapshotRef


class MessageRangeKeyV2(StrictModel):
    """Freeze source coordinates and the original Add receipt separately.

    Every current pipeline preserves declared source order in contiguous coordinated blocks and
    positions unclocked requests by their receipt boundaries. Receipt order is
    never a source timestamp, and V1 payloads retain their original meaning.
    """

    key_kind: Literal["message_v2"]
    session_event_time: HistoricalTimestamp | None
    session_id: ExternalId
    request_id: ExternalId
    chunk_index: NonNegativeInt | None
    chunk_event_time: HistoricalTimestamp | None
    ordinal: NonNegativeInt
    part_no: NonNegativeInt
    evidence_id: EvidenceSnapshotRef
    received_commit_seq: NonNegativeInt
    source_time_present: bool
    receipt_partition_seq: NonNegativeInt


MessageRangeKey = MessageRangeKeyV1 | MessageRangeKeyV2
_MESSAGE_KEY: TypeAdapter[MessageRangeKey] = TypeAdapter(
    Annotated[MessageRangeKey, Field(discriminator="key_kind")],
)


def parse_message_key(raw: object) -> MessageRangeKey:
    """Read an exact frozen coordinate variant without upgrading V1 bytes."""
    return (
        _MESSAGE_KEY.validate_json(raw)
        if isinstance(raw, str)
        else _MESSAGE_KEY.validate_python(raw)
    )


class SupportRangeKeyV1(StrictModel):
    """Freeze one Evidence coordinate in the support order."""

    key_kind: Literal["support"]
    trust_tier: Literal["verified", "reported", "inferred", "untrusted"]
    observed_at: HistoricalTimestamp
    evidence_id: EvidenceSnapshotRef


RangeKeyV1 = Annotated[
    MessageRangeKeyV1 | MessageRangeKeyV2 | SupportRangeKeyV1, Field(discriminator="key_kind"),
]


class GoverningExtractionV1(StrictModel):
    """Carry the immutable extraction release and both normalized bindings."""

    extraction_policy_release_ref: PolicyReleaseRef
    extraction_config_digest: Integrity
    extraction_policy_snapshot_digest: Integrity


class ConfirmedRequestV1(StrictModel):
    """Preserve consumed-intent provenance without granting live authority."""

    family_digest: ContentDigest
    consumed_intent_key: LocalId


class RequestProgressV1(StrictModel):
    """Bind one accepted request identity to its immutable admission digest."""

    admission_digest: ContentDigest
    confirmed_request: ConfirmedRequestV1 | None = None
