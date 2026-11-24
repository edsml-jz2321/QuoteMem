"""Manual existing-job inspection and purge through the established owners.

Neither command initializes or recovers a job, adopts a Writer epoch, loads
providers, or stops another process. A serving process must first acknowledge
its ordinary shutdown; its retained exclusive lock prevents offline inspection.
"""

from __future__ import annotations

import hashlib
import os
import stat
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from quotemem.core.controls import JobValueV1
from quotemem.core.errors import QuoteMemError
from quotemem.core.existing_job import ExistingJobResources, open_existing_job
from quotemem.core.identity import IntegrityDigest
from quotemem.service.audit_types import (
    PURGE_NOT_COVERED,
    CompletedPurgeReceipt,
    CompletedPurgeReceiptV2,
    ConsoleCompletionV1,
    RetainedDataViolationV1,
    parse_workspace_row_counts,
)
from quotemem.service.completion import (
    completion_deadline,
    parse_completed_at,
    require_forward_clock,
)
from quotemem.service.ownership import WorkspaceOwner
from quotemem.service.status import JobTrafficStatus, job_traffic_status
from quotemem.store.bench_audit import AuditRecords, BenchAudit


def system_time_ms() -> int:
    """Sample the trusted system clock without converting through float seconds."""
    return time.time_ns() // 1_000_000


@dataclass(frozen=True)
class JobStatus:
    """Content-free observed status; an unknown completion has no inferred deadline."""

    job_id: str
    workspace_path_hash: IntegrityDigest
    state: Literal["retained", "purged"]
    records: AuditRecords
    traffic: JobTrafficStatus | None
    last_platform_traffic_at: int | None
    overdue: bool | None

    @property
    def exit_code(self) -> int:
        """Warn for due traffic reminders or an actually recorded retention violation."""
        return int(
            self.records.violation is not None or self.overdue is True
            or (self.traffic is not None and self.traffic.exit_code != 0)
        )

    def payload(self) -> dict[str, object]:
        """Serialize only exact job/path identity, timing, and external audit facts."""
        completion = self.records.completion
        return {
            "status": self.state,
            "job_id": self.job_id,
            "workspace_path_hash": self.workspace_path_hash,
            "last_platform_traffic_at": self.last_platform_traffic_at,
            "traffic": None if self.traffic is None else {
                "state": self.traffic.state, "elapsed_ms": self.traffic.elapsed_ms,
            },
            "deadline": None if completion is None else (
                completion_deadline(completion.completed_at)
            ),
            "overdue": self.overdue,
            "completion": None if completion is None else completion.model_dump(mode="json"),
            "violation": None if self.records.violation is None else (
                self.records.violation.model_dump(mode="json")
            ),
            "receipt": None if self.records.receipt is None else (
                self.records.receipt.model_dump(mode="json")
            ),
        }


def _paths(root: Path, audit_path: Path) -> tuple[Path, Path, IntegrityDigest]:
    # Keep the final existing root entry visible to the no-redirect check. Parent
    # resolution establishes the same canonical namespace used by the owner.
    named = root.expanduser().absolute()
    try:
        metadata = named.lstat()
    except FileNotFoundError:
        metadata = None
    if metadata is not None and (
        stat.S_ISLNK(metadata.st_mode) or getattr(metadata, "st_file_attributes", 0) & 0x400
    ):
        raise QuoteMemError("workspace_binding_mismatch",
                            "A job workspace cannot be a redirected entry")
    selected = named.resolve(strict=False)
    audit = audit_path.expanduser().absolute()
    resolved_audit = audit.resolve(strict=False)
    if resolved_audit == selected or selected in resolved_audit.parents:
        raise QuoteMemError("audit_ownership", "The audit must be outside the job workspace")
    digest = IntegrityDigest(hashlib.sha256(os.fsencode(selected)).hexdigest())
    return selected, audit, digest


def _absent_job(audit_path: Path, job_id: str, path_hash: IntegrityDigest) -> AuditRecords:
    # The workspace is absent. Without any audit record of this job and path, nothing
    # was ever purged here: the path or the audit names no job (``workspace_not_found``).
    # A recorded completion without its receipt stays an unconfirmed purge.
    records = AuditRecords(None, None, None)
    if os.path.lexists(audit_path):
        with BenchAudit(audit_path) as audit:
            records = audit.read(job_id, path_hash)
    if records.completion is None and records.violation is None and records.receipt is None:
        raise QuoteMemError("workspace_not_found",
                            "No job workspace is at this path and the audit has no record of it")
    if records.receipt is None:
        raise QuoteMemError("purge_unconfirmed",
                            "Missing job data does not prove completed deletion")
    return records


def _history(audit_path: Path, job_id: str, path_hash: IntegrityDigest) -> AuditRecords:
    # An absent workspace or audit never authorizes creating a missing job or
    # reconstructing a receipt. Pure historical reads do not sample a new clock.
    if not os.path.lexists(audit_path):
        raise QuoteMemError("purge_unconfirmed", "No completed deletion receipt is available")
    with BenchAudit(audit_path) as audit:
        records = audit.read(job_id, path_hash)
    if records.receipt is None:
        raise QuoteMemError("purge_unconfirmed",
                            "Missing job data does not prove completed deletion")
    return records


def _validate_retained_time(
    resources: ExistingJobResources, records: AuditRecords, now: int, *previous: int,
) -> None:
    resources.validate_owner()
    records.validate_clock(now, *previous)
    job = resources.metadata.job_control.value
    assert isinstance(job, JobValueV1)
    require_forward_clock(now, job.last_platform_traffic_at)


def _retained(
    resources: ExistingJobResources, audit: BenchAudit, records: AuditRecords,
    *, path_hash: IntegrityDigest, now: int, previous: tuple[int, ...] = (),
) -> AuditRecords:
    _validate_retained_time(resources, records, now, *previous)
    if records.receipt is not None:
        raise QuoteMemError(
            "audit_completion_conflict", "Completed audit identity has retained job data",
        )
    completion = records.completion
    if completion is not None and now > completion_deadline(completion.completed_at):
        audit.record_retained(RetainedDataViolationV1(
            event_kind="retention_violation", basis="retained_data",
            job_id=resources.metadata.job_id, workspace_path_hash=path_hash,
            completed_at=completion.completed_at, observed_at=now,
        ))
        records = audit.read(completion.job_id, path_hash)
    resources.validate_owner()
    return records


def status_job(
    root: Path, job_id: str, *, audit_path: Path,
    clock: Callable[[], int] = system_time_ms,
) -> JobStatus:
    """Inspect an existing job or return its original completed audit history.

    A retained overdue job records the first actual observation under the same
    exclusive owner. Missing or interrupted job files remain an error; they do
    not establish deletion, completion or a retained-data observation: an absent
    workspace the audit never recorded is ``workspace_not_found``, one whose recorded
    completion has no receipt ``purge_unconfirmed``.
    """
    selected, audit_path, path_hash = _paths(root, audit_path)
    if not os.path.lexists(selected):
        records = _absent_job(audit_path, job_id, path_hash)
        receipt, completion = records.receipt, records.completion
        assert receipt is not None
        return JobStatus(job_id, path_hash, "purged", records, None, None,
                         None if completion is None else (
                             receipt.deleted_at > completion_deadline(completion.completed_at)
                         ))
    owner = WorkspaceOwner(selected, profile="bench")
    with open_existing_job(selected, job_id, owner=owner) as resources:
        metadata = resources.metadata
        job = metadata.job_control.value
        assert isinstance(job, JobValueV1)
        records = AuditRecords(None, None, None)
        if os.path.lexists(audit_path):
            with BenchAudit(audit_path) as audit:
                records = audit.read(job_id, path_hash)
                now = clock()
                records = _retained(resources, audit, records, path_hash=path_hash, now=now)
        else:
            now = clock()
            _validate_retained_time(resources, records, now)
        traffic = job_traffic_status(job.last_platform_traffic_at, now=now)
        if traffic.state == "clock_rollback":
            raise QuoteMemError("clock_rollback", "The trusted clock precedes platform traffic")
        resources.validate_owner()
        completion = records.completion
        return JobStatus(job_id, path_hash, "retained", records, traffic,
                         job.last_platform_traffic_at or None,
                         None if completion is None else (
                             now > completion_deadline(completion.completed_at)
                         ))


@dataclass
class _PurgeAttempt:
    # Assigned only after the one filesystem owner acknowledges complete deletion.
    # Keeping this exact payload permits lost-ack readback without replacing time
    # or counts. A subsequent process can use only actual persisted history.
    receipt: CompletedPurgeReceipt | None = None


def _delete_job(
    selected: Path, job_id: str, completed_at: str, audit_path: Path,
    path_hash: IntegrityDigest, clock: Callable[[], int], attempt: _PurgeAttempt,
) -> CompletedPurgeReceipt:
    owner = WorkspaceOwner(selected, profile="bench")
    with open_existing_job(selected, job_id, owner=owner) as resources:
        counts = parse_workspace_row_counts(dict(resources.metadata.row_counts))
        with BenchAudit(audit_path) as audit:
            records = audit.read(job_id, path_hash)
            recorded_at = clock()
            _validate_retained_time(resources, records, recorded_at)
            if records.receipt is not None:
                raise QuoteMemError(
                    "audit_completion_conflict", "Completed audit identity has retained job data",
                )
            audit.record_completion(ConsoleCompletionV1(
                event_kind="console_completion", job_id=job_id, workspace_path_hash=path_hash,
                completed_at=parse_completed_at(completed_at, recorded_at=recorded_at),
                recorded_at=recorded_at,
            ))
            records = audit.read(job_id, path_hash)
            records = _retained(resources, audit, records, path_hash=path_hash, now=recorded_at)
            # Existing-only opening already excludes a serving daemon. Dispose
            # its read pools before the final retained-data/clock check, while
            # the same process lock and pinned storage identities remain held.
            resources.close_storage()
            resources.require_storage_closed()
            before_delete = clock()
            records = _retained(resources, audit, records, path_hash=path_hash,
                                now=before_delete, previous=(recorded_at,))
            owner.purge_existing(resources)
            deleted_at = clock()
            records.validate_clock(deleted_at, recorded_at, before_delete)
            # This command proves only whole local workspace disposal. The fixed
            # provider adapters expose no job-copy deletion contract.
            receipt: CompletedPurgeReceipt = CompletedPurgeReceiptV2(
                schema_version=2, job_id=job_id, workspace_path_hash=path_hash,
                deleted_at=deleted_at, row_counts=counts, not_covered=list(PURGE_NOT_COVERED),
            )
            attempt.receipt = receipt
            audit.append(receipt)
            actual = audit.read(job_id, path_hash).receipt
            if actual != receipt:
                raise QuoteMemError("purge_unconfirmed",
                                    "The exact completed receipt was not returned")
            return receipt


def purge_job(
    root: Path, job_id: str, *, completed_at: str, audit_path: Path,
    clock: Callable[[], int] = system_time_ms,
) -> CompletedPurgeReceipt:
    """Manually purge one actual closed bench job and persist its exact receipt.

    Completion is explicitly supplied from the console. Audit failure before
    deletion leaves all workspace files intact. Failure after deletion returns
    success only after a fresh audit owner reads the exact original receipt and
    required violation. No retry fabricates counts, time or completion history.
    """
    selected, audit_path, path_hash = _paths(root, audit_path)
    if not os.path.lexists(selected):
        # A legacy completed receipt remains unchanged with unknown completion;
        # the new CLI argument cannot retroactively confirm a disappeared job.
        records = _absent_job(audit_path, job_id, path_hash)
        assert records.receipt is not None
        if records.completion is not None and parse_completed_at(
            completed_at, recorded_at=records.completion.recorded_at,
        ) != records.completion.completed_at:
            raise QuoteMemError("audit_completion_conflict", "Completion time is already bound")
        return records.receipt
    attempt = _PurgeAttempt()
    try:
        return _delete_job(selected, job_id, completed_at, audit_path, path_hash, clock, attempt)
    except Exception:
        if attempt.receipt is None:
            raise
    # Leave the failed audit context and its exception traceback before trying
    # a fresh lock. The local exact receipt is the only permitted replay value.
    records = _history(audit_path, job_id, path_hash)
    if records.receipt != attempt.receipt:
        raise QuoteMemError("purge_unconfirmed",
                            "Completed deletion lacks its exact original receipt")
    assert records.receipt is not None
    return records.receipt
