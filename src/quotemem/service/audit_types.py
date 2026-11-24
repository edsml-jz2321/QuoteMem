"""Closed, content-free purge audit shapes.

These models validate shape only. The future purge owner must prove job/path
binding, counts and completed deletion before the HB audit adapter accepts one.
There is no receipt-success factory or persistence operation in this module.
"""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, TypeAdapter, model_validator

from quotemem.core.technical_types import ExternalId, Integrity, NonNegativeInt, Timestamp
from quotemem.service.completion import completion_deadline
from quotemem.store.metadata import StrictModel


class WorkspaceRowCountsV2(StrictModel):
    """Actual pre-deletion counts of all twelve QuoteMem tables, with no implicit zeros.

    HB's separate registration/control database and filesystem companions are
    covered by whole-workspace deletion, not represented as invented QuoteMem rows.
    """

    evidence: NonNegativeInt
    link: NonNegativeInt
    run: NonNegativeInt
    recall: NonNegativeInt
    artifact: NonNegativeInt
    control: NonNegativeInt
    item: NonNegativeInt
    revision: NonNegativeInt
    candidate: NonNegativeInt
    entity: NonNegativeInt
    entity_name: NonNegativeInt
    tombstone: NonNegativeInt


WorkspaceRowCounts = WorkspaceRowCountsV2
_COUNTS = TypeAdapter[WorkspaceRowCounts](WorkspaceRowCounts)


def parse_workspace_row_counts(value: object) -> WorkspaceRowCounts:
    """Validate one complete admitted table inventory without filling missing counts.

    Args:
        value: Actual pre-deletion table counts from the owning workspace reader.

    Returns:
        The exact twelve-table count record.

    Raises:
        ValidationError: Missing, unexpected, negative or non-integer counts.
    """
    return _COUNTS.validate_python(value)


PurgeNotCovered = Literal[
    "backups", "existing_exports", "host_session_logs", "provider_copies", "replicas", "wal",
]
PURGE_NOT_COVERED: tuple[PurgeNotCovered, ...] = (
    "backups", "existing_exports", "host_session_logs", "provider_copies", "replicas", "wal",
)
"""Copies a purge receipt does not attest as deleted: the forget receipt's list."""


class CompletedPurgeReceiptV2(StrictModel):
    """Versioned completed deletion evidence, without business content.

    The job ID is the unchanged opaque external binding. Its bound leaves room
    for the shared Control key's ``job:`` prefix. The path hash must identify the
    canonical workspace path; this model neither resolves nor hashes a path.
    ``deleted_at`` is the actual completed local deletion time, never a predicted
    deadline or a job-completion timestamp. Shape validity proves none of these
    historical facts; the purge owner remains their authority. ``not_covered``
    discloses the copies the receipt does not attest as deleted, exactly as
    the forget receipt does.
    """

    schema_version: Literal[2]
    job_id: Annotated[ExternalId, Field(max_length=1020)]
    workspace_path_hash: Integrity
    deleted_at: Timestamp
    row_counts: WorkspaceRowCountsV2
    not_covered: list[PurgeNotCovered]

    @model_validator(mode="after")
    def canonical_coverage(self) -> Self:
        """Require exactly the forget receipt's disclosure list, in its order."""
        if tuple(self.not_covered) != PURGE_NOT_COVERED:
            raise ValueError("Purge coverage must be exactly the forget receipt's list")
        return self


CompletedPurgeReceipt = CompletedPurgeReceiptV2
_RECEIPT = TypeAdapter[CompletedPurgeReceipt](CompletedPurgeReceipt)


def parse_completed_purge_receipt(value: object) -> CompletedPurgeReceipt:
    """Read one receipt of the current version without upgrading its payload.

    Args:
        value: Receipt fields or persisted receipt JSON text.

    Returns:
        The validated receipt with its original counts.

    Raises:
        ValidationError: A malformed receipt, unsupported version or incomplete inventory.
    """
    if isinstance(value, str):
        return _RECEIPT.validate_json(value)
    return _RECEIPT.validate_python(value)


class ConsoleCompletionV1(StrictModel):
    """Explicit console completion with its original trusted confirmation time.

    This validates neither console truth nor exclusive workspace/job ownership.
    The audit owner separately enforces immutable replay and cross-record clocks.
    """

    event_kind: Literal["console_completion"]
    job_id: Annotated[ExternalId, Field(max_length=1020)]
    workspace_path_hash: Integrity
    completed_at: Timestamp
    recorded_at: Timestamp

    @model_validator(mode="after")
    def valid_confirmation_time(self) -> Self:
        """Reject a future confirmation or an unrepresentable base deadline."""
        completion_deadline(self.completed_at)
        if self.completed_at > self.recorded_at:
            raise ValueError("Completion cannot follow its trusted recording time")
        return self


class RetainedDataViolationV1(StrictModel):
    """A trusted overdue observation, requiring external retained-data proof.

    The exclusive job owner proves actual retained data at ``observed_at``. This
    shape alone cannot prove that fact or its binding to a stored confirmation.
    """

    event_kind: Literal["retention_violation"]
    basis: Literal["retained_data"]
    job_id: Annotated[ExternalId, Field(max_length=1020)]
    workspace_path_hash: Integrity
    completed_at: Timestamp
    observed_at: Timestamp

    @model_validator(mode="after")
    def actual_overdue_time(self) -> Self:
        """Require an observation strictly after the exact 30-day deadline."""
        if self.observed_at <= completion_deadline(self.completed_at):
            raise ValueError("A retained-data observation must follow the completion deadline")
        return self


class CompletedPurgeViolationV1(StrictModel):
    """Late purge completion whose timing proof is the actual completed receipt.

    The audit owner must jointly validate the exact receipt identity and late
    ``deleted_at``. No new observation time, deletion result, or proof is invented
    here. This shape is not an independently valid completed-purge acknowledgement.
    """

    event_kind: Literal["retention_violation"]
    basis: Literal["completed_purge"]
    job_id: Annotated[ExternalId, Field(max_length=1020)]
    workspace_path_hash: Integrity
    completed_at: Timestamp

    @model_validator(mode="after")
    def representable_deadline(self) -> Self:
        """Reject completion inputs whose base deadline cannot be represented."""
        completion_deadline(self.completed_at)
        return self


RetentionViolationV1 = Annotated[
    RetainedDataViolationV1 | CompletedPurgeViolationV1, Field(discriminator="basis")
]
