"""Ephemeral receipt-proven message boundaries.

These code-owned values are produced only after the complete reception inventory
has been verified. They are not provider output, persisted schemas, or an
authorization substitute. A surviving source suffix cannot establish its own
message boundary.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from quotemem.core.errors import QuoteMemError
from quotemem.core.identity import ArtifactId, ContentBindingDigest


@dataclass(frozen=True)
class MessageBoundaryProof:
    """One shared, content-free message membership witness from an exact receipt.

    All per-part capabilities share this object, so the ordered identity tuple
    occupies memory once per message. Byte bounds are intentionally absent:
    privacy tombstones zero them while preserving this immutable membership.
    """

    tenant_id: str
    scope_kind: str
    scope_id: str
    session_id: str
    request_id: str
    ordinal: int
    ingested_commit_seq: int
    ordered_part_ids: tuple[str, ...]
    receipt_ref: ArtifactId
    receipt_admission_digest: ContentBindingDigest
    inventory_digest: ContentBindingDigest

    def __post_init__(self) -> None:
        if (
            type(self.ordinal) is not int
            or self.ordinal < 0
            or type(self.ingested_commit_seq) is not int
            or self.ingested_commit_seq < 0
            or type(self.ordered_part_ids) is not tuple
            or not self.ordered_part_ids
            or len(set(self.ordered_part_ids)) != len(self.ordered_part_ids)
        ):
            raise QuoteMemError("binding_mismatch", "Message boundary has invalid membership")

    @property
    def final_part_no(self) -> int:
        """Return the last part of the independently proven complete message."""
        return len(self.ordered_part_ids) - 1


@dataclass(frozen=True)
class EvidenceBoundaryProof:
    """One exact physical source's position within its received logical message."""

    message: MessageBoundaryProof
    evidence_id: str
    part_no: int

    def __post_init__(self) -> None:
        if (
            type(self.part_no) is not int
            or not 0 <= self.part_no <= self.message.final_part_no
            or self.message.ordered_part_ids[self.part_no] != self.evidence_id
        ):
            raise QuoteMemError("binding_mismatch", "Source boundary has another part identity")

    @property
    def is_final_part(self) -> bool:
        """Finality concerns the original message, not the selected source window."""
        return self.part_no == self.message.final_part_no

    @property
    def descriptor(self) -> dict[str, object]:
        """Copy the compact binding; never duplicate the complete identity tuple."""
        return {
            "evidence_id": self.evidence_id,
            "part_no": self.part_no,
            "final_part_no": self.message.final_part_no,
            "receipt_ref": self.message.receipt_ref,
            "receipt_admission_digest": self.message.receipt_admission_digest,
            "inventory_digest": self.message.inventory_digest,
        }

    def validate(self, row: Mapping[str, object]) -> None:
        """Check the complete immutable source identity before using finality.

        Source text, offsets, role, liveness and action-time authorization remain
        independently checked by grounding and its scoped reader. An erased last
        part still proves an earlier part was not the original message end.
        """
        expected: dict[str, object] = {
            "object_id": self.evidence_id,
            "tenant_id": self.message.tenant_id,
            "scope_kind": self.message.scope_kind,
            "scope_id": self.message.scope_id,
            "session_id": self.message.session_id,
            "request_id": self.message.request_id,
            "ordinal": self.message.ordinal,
            "part_no": self.part_no,
            "ingested_commit_seq": self.message.ingested_commit_seq,
        }
        if any(
            type(row.get(key)) is not type(value) or row.get(key) != value
            for key, value in expected.items()
        ):
            raise QuoteMemError("binding_mismatch",
                                "Source differs from its received message boundary")
