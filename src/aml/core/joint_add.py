"""Frozen full-Add citation locators, distinct from checkpoint cursor members."""

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from aml.core.identity import ULID, ArtifactId
from aml.core.ingestion_limits import MAX_ADD_SOURCE_PARTS
from aml.core.semantic import ShortText
from aml.core.technical_types import ContentDigest, Integrity, NonNegativeInt, PositiveInt
from aml.store.metadata import OpaqueIdSpec, RefSpec, StrictModel


class JointAddSourceV1(StrictModel):
    """One complete original Evidence part; mutable routing is not source identity."""

    evidence_id: Annotated[ULID, RefSpec("evidence", ref_mode="snapshot_locator")]
    content_hash: Integrity
    source_digest: Integrity
    byte_start: NonNegativeInt
    byte_end: PositiveInt

    @model_validator(mode="after")
    def validate_interval(self) -> Self:
        """Reject empty or inverted original intervals."""
        if self.byte_start >= self.byte_end:
            raise ValueError("Joint Add source requires a nonempty interval")
        return self


class ErasedJointAddSourceV1(StrictModel):
    """A proved tombstone in the full receipt inventory, never citation content."""

    evidence_id: Annotated[ULID, RefSpec("evidence", ref_mode="snapshot_locator")]
    source_digest: Integrity


class ExpiredJointAddSourceV1(StrictModel):
    """A receipt member whose frozen TTL elapsed; it grants no citation slot."""

    evidence_id: Annotated[ULID, RefSpec("evidence", ref_mode="snapshot_locator")]
    source_digest: Integrity
    expires_at: NonNegativeInt


class FinalizeTTLJointAddOmissionV1(StrictModel):
    """An original full-row HMAC located by selected Finalize metadata, never a body digest."""

    evidence_id: Annotated[ULID, RefSpec("evidence", ref_mode="snapshot_locator")]
    row_binding_digest: ContentDigest


class PrefreezeExpiredFinalizeJointAddSourceV1(StrictModel):
    """An actual pre-freeze TTL part, bound to its header rather than unseen content."""

    evidence_id: Annotated[ULID, RefSpec("evidence", ref_mode="snapshot_locator")]
    expires_at: NonNegativeInt
    header_digest: Integrity


class PrefreezeFinalizeMessagePeerV1(StrictModel):
    """An unusable whole-message peer, identified without claiming a full-row binding."""

    evidence_id: Annotated[ULID, RefSpec("evidence", ref_mode="snapshot_locator")]
    header_digest: Integrity
    blocker_evidence_id: Annotated[ULID, RefSpec("evidence", ref_mode="snapshot_locator")]


class JointAddBatchV1(StrictModel):
    """Complete ordered part inventory of one received Add."""

    session_id: Annotated[ShortText, OpaqueIdSpec("external")]
    request_id: Annotated[ShortText, OpaqueIdSpec("external")]
    sources: Annotated[
        list[JointAddSourceV1], Field(max_length=MAX_ADD_SOURCE_PARTS),
    ]
    erased_sources: list[ErasedJointAddSourceV1] = Field(
        default_factory=list[ErasedJointAddSourceV1], max_length=MAX_ADD_SOURCE_PARTS,
        exclude_if=lambda value: not value,
    )
    expired_sources: list[ExpiredJointAddSourceV1] = Field(
        default_factory=list[ExpiredJointAddSourceV1], max_length=MAX_ADD_SOURCE_PARTS,
        exclude_if=lambda value: not value,
    )
    source_ttl_omissions: list[FinalizeTTLJointAddOmissionV1] = Field(
        default_factory=list[FinalizeTTLJointAddOmissionV1], max_length=MAX_ADD_SOURCE_PARTS,
        exclude_if=lambda value: not value,
    )
    prefreeze_expired_sources: list[PrefreezeExpiredFinalizeJointAddSourceV1] = Field(
        default_factory=list[PrefreezeExpiredFinalizeJointAddSourceV1],
        max_length=MAX_ADD_SOURCE_PARTS, exclude_if=lambda value: not value,
    )
    prefreeze_message_peers: list[PrefreezeFinalizeMessagePeerV1] = Field(
        default_factory=list[PrefreezeFinalizeMessagePeerV1],
        max_length=MAX_ADD_SOURCE_PARTS, exclude_if=lambda value: not value,
    )

    @model_validator(mode="after")
    def validate_membership(self) -> Self:
        """Reject duplicate part identities before resolving any source text."""
        members = [
            *self.sources, *self.erased_sources, *self.expired_sources, *self.source_ttl_omissions,
            *self.prefreeze_expired_sources, *self.prefreeze_message_peers,
        ]
        if not 1 <= len(members) <= MAX_ADD_SOURCE_PARTS:
            raise ValueError("Joint Add requires its bounded complete inventory")
        if len({source.evidence_id for source in members}) != len(members):
            raise ValueError("Joint Add source inventory repeats an Evidence part")
        expired_ids = {source.evidence_id for source in self.prefreeze_expired_sources}
        if any(peer.blocker_evidence_id not in expired_ids
               for peer in self.prefreeze_message_peers):
            raise ValueError("A pre-freeze message peer requires its actual expired blocker")
        return self


class JointAddContextV1(StrictModel):
    """An explicit full predecessor/current pair, never a cumulative tail."""

    version: Literal[1] = 1
    current: JointAddBatchV1
    previous: JointAddBatchV1 | None = None
    source_ttl_ref: Annotated[
        ArtifactId,
        RefSpec("artifact", ref_mode="snapshot_locator", artifact_kinds=("decision_set",)),
    ] | None = Field(default=None, exclude_if=lambda value: value is None)

    @property
    def batches(self) -> tuple[JointAddBatchV1, ...]:
        """Return the frozen citation order, without changing serialized bytes."""
        return (self.current,) if self.previous is None else (self.previous, self.current)

    @model_validator(mode="after")
    def validate_pair(self) -> Self:
        """Require disjoint received requests and source inventories."""
        if (self.current.prefreeze_expired_sources or self.current.prefreeze_message_peers) and (
            not self.current.sources
        ):
            raise ValueError("A pre-freeze TTL gap cannot authorize an unavailable current Add")
        if self.source_ttl_ref is None and any(
            batch.source_ttl_omissions for batch in self.batches
        ):
            raise ValueError("Finalize TTL omissions require the selected disposition reference")
        if self.previous is not None and (
            (self.current.session_id, self.current.request_id)
            == (self.previous.session_id, self.previous.request_id)
            or {source.evidence_id for source in (
                *self.current.sources, *self.current.erased_sources,
                *self.current.expired_sources,
                *self.current.source_ttl_omissions,
                *self.current.prefreeze_expired_sources,
                *self.current.prefreeze_message_peers,
            )} & {source.evidence_id for source in (
                *self.previous.sources, *self.previous.erased_sources,
                *self.previous.expired_sources,
                *self.previous.source_ttl_omissions,
                *self.previous.prefreeze_expired_sources,
                *self.previous.prefreeze_message_peers,
            )}
        ):
            raise ValueError("Joint Add context must contain distinct received batches")
        return self


class PlanJointAddSourceV1(JointAddSourceV1):
    """Keep selected sources only while their finalize plan owner is nonterminal."""

    evidence_id: Annotated[
        ULID, RefSpec("evidence", ref_mode="snapshot_locator", gc_root="while_owner_nonterminal"),
    ]


class PlanJointAddBatchV1(StrictModel):
    """Finalize-plan projection of the same complete original batch."""

    session_id: Annotated[ShortText, OpaqueIdSpec("external")]
    request_id: Annotated[ShortText, OpaqueIdSpec("external")]
    sources: Annotated[
        list[PlanJointAddSourceV1], Field(max_length=MAX_ADD_SOURCE_PARTS),
    ]
    erased_sources: list[ErasedJointAddSourceV1] = Field(
        default_factory=list[ErasedJointAddSourceV1], max_length=MAX_ADD_SOURCE_PARTS,
        exclude_if=lambda value: not value,
    )
    expired_sources: list[ExpiredJointAddSourceV1] = Field(
        default_factory=list[ExpiredJointAddSourceV1], max_length=MAX_ADD_SOURCE_PARTS,
        exclude_if=lambda value: not value,
    )
    prefreeze_expired_sources: list[PrefreezeExpiredFinalizeJointAddSourceV1] = Field(
        default_factory=list[PrefreezeExpiredFinalizeJointAddSourceV1],
        max_length=MAX_ADD_SOURCE_PARTS, exclude_if=lambda value: not value,
    )
    prefreeze_message_peers: list[PrefreezeFinalizeMessagePeerV1] = Field(
        default_factory=list[PrefreezeFinalizeMessagePeerV1],
        max_length=MAX_ADD_SOURCE_PARTS, exclude_if=lambda value: not value,
    )


class PlanJointAddContextV1(StrictModel):
    """Use the same wire shape with the finalize plan's conditional references."""

    version: Literal[1] = 1
    current: PlanJointAddBatchV1
    previous: PlanJointAddBatchV1 | None = None

    @model_validator(mode="after")
    def validate_pair(self) -> Self:
        """Apply ordinary complete-pair invariants without changing reference policy."""
        JointAddContextV1.model_validate(self.model_dump(mode="json"))
        return self
