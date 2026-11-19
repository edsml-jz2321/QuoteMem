"""Minimal HB-owned completion and purge audit outside every job workspace (§9).

The purge caller owns completion evidence. This adapter validates and persists
closed records; it has no console, retained-data, deletion or recovery authority.
"""

from __future__ import annotations

import hashlib
import os
import stat
import threading
from collections.abc import Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Self, cast

from pydantic import TypeAdapter, ValidationError
from sqlalchemy import inspect

from aml.core.errors import AMLError
from aml.core.identity import canonical_json
from aml.core.platform_files import (
    ensure_private_directory,
    lock_exclusive,
    open_directory,
    open_file_at,
    stat_at,
    sync_descriptor,
    validate_private_descriptor,
)
from aml.service.audit_types import (
    CompletedPurgeReceipt,
    CompletedPurgeReceiptV2,
    CompletedPurgeViolationV1,
    ConsoleCompletionV1,
    RetainedDataViolationV1,
    RetentionViolationV1,
    parse_completed_purge_receipt,
)
from aml.service.completion import completion_deadline, require_forward_clock
from aml.store.hb_runtime import BootstrapSpec, ConfigManager, Context, hb, verify_privacy

_ENTITY = "aml-bench-purge-receipt"
_TABLE = "aml_bench_purge_receipt"
_APPLICATION_ID = 0x414D4C41
if os.name != "nt":
    _FLAGS = os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
else:
    _FLAGS = 0  # Native open_file_at implements no-follow and noninheritance.
_JOB = TypeAdapter[str](CompletedPurgeReceiptV2.model_fields["job_id"].rebuild_annotation())
_PATH = TypeAdapter[str](
    CompletedPurgeReceiptV2.model_fields["workspace_path_hash"].rebuild_annotation()
)
_VIOLATION = TypeAdapter[RetainedDataViolationV1 | CompletedPurgeViolationV1](RetentionViolationV1)

# Import through AML's explicit-authority boundary before consuming HB schemas.
from heavenbase.backends.sql.sqlite import SQLiteBackend  # noqa: E402
from heavenbase.execution.contracts import RowOp  # noqa: E402
from heavenbase.placement.binding import StorageBinding  # noqa: E402
from heavenbase.schema.model import EntitySchema, FieldSchema  # noqa: E402


def _key(job_id: str, path_hash: str) -> str:
    try:
        job_id = _JOB.validate_python(job_id, strict=True)
        path_hash = _PATH.validate_python(path_hash, strict=True)
    except ValidationError:
        raise AMLError("audit_receipt_invalid", "Audit receipt identity is invalid") from None
    return hashlib.sha256(canonical_json({
        "job_id": job_id, "workspace_path_hash": path_hash,
    }).encode("utf-8")).hexdigest()


def _event_key(event_kind: str, job_id: str, path_hash: str) -> str:
    # The caller selects one of the two closed kinds; identity validation is
    # shared with legacy receipts, whose original technical key is unchanged.
    _key(job_id, path_hash)
    return hashlib.sha256(canonical_json({
        "event_kind": event_kind, "job_id": job_id, "workspace_path_hash": path_hash,
    }).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class AuditRecords:
    """Detached jointly validated historical facts for one exact job/path.

    An absent completion leaves the deadline unknown even when a legacy receipt
    exists. Reading these facts samples no clock and proves no current job state.
    """

    completion: ConsoleCompletionV1 | None
    violation: RetentionViolationV1 | None
    receipt: CompletedPurgeReceipt | None

    def validate_clock(self, now: int, *previous: int) -> None:
        """Check a new sample against this history and prior operation samples.

        Args:
            now: New trusted system timestamp; this method does not sample time.
            *previous: Earlier trusted times from the same calling operation.

        Raises:
            AMLError: An invalid time or clock rollback against any applicable fact.
        """
        times: list[int] = []
        if self.completion is not None:
            times.append(self.completion.recorded_at)
        if isinstance(self.violation, RetainedDataViolationV1):
            times.append(self.violation.observed_at)
        if self.receipt is not None:
            times.append(self.receipt.deleted_at)
        require_forward_clock(now, *times, *previous)


def _validate_records(records: AuditRecords) -> None:
    completion, violation, receipt = records.completion, records.violation, records.receipt
    if completion is None:
        if violation is not None:
            raise AMLError("audit_inconsistent", "Audit violation has no completion confirmation")
        return
    deadline = completion_deadline(completion.completed_at)
    if violation is not None:
        if violation.completed_at != completion.completed_at:
            raise AMLError("audit_inconsistent", "Audit violation has another completion binding")
        if isinstance(violation, RetainedDataViolationV1):
            require_forward_clock(violation.observed_at, completion.recorded_at)
        elif receipt is None or receipt.deleted_at <= deadline:
            raise AMLError("audit_inconsistent", "Late-purge violation lacks its completed receipt")
    if receipt is not None:
        require_forward_clock(receipt.deleted_at, completion.recorded_at)
        if isinstance(violation, RetainedDataViolationV1):
            require_forward_clock(receipt.deleted_at, violation.observed_at)
        if receipt.deleted_at > deadline and violation is None:
            raise AMLError("audit_inconsistent", "Late-purge receipt lacks its required violation")


@dataclass(frozen=True)
class _File:
    name: str
    fd: int
    identity: tuple[int, int, int, int, int]


def _identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return value.st_dev, value.st_ino, value.st_uid, value.st_mode, value.st_nlink


class _AuditFiles:
    """Pin just one audit directory, file, lock and SQLite companions."""

    def __init__(self, path: Path) -> None:
        path = path.expanduser().absolute()
        self.root = path.parent.parent.resolve(strict=True) / path.parent.name
        self.path = self.root / path.name
        self.pid = os.getpid()
        self.files: dict[str, _File] = {}
        self.root_fd: int | None = None
        self.root_identity: tuple[int, int, int, int, int] | None = None
        self.locked = False

    def acquire(self) -> None:
        ensure_private_directory(self.root)
        fd = open_directory(self.root)
        self.root_fd = fd
        value = os.fstat(fd)
        if os.name != "nt" and (
            stat.S_IMODE(value.st_mode) != 0o700 or value.st_uid != os.geteuid()
        ):
            raise AMLError("audit_ownership", "Audit directory must be private and user-owned")
        self.root_identity = _identity(value)
        lock = self._pin(self.path.name + ".lock", create=True)
        try:
            lock_exclusive(lock.fd)
        except BlockingIOError:
            raise AMLError("audit_busy", "Another audit operation owns this file") from None
        self.locked = True
        self._pin(self.path.name, create=True)
        self.validate()
        sync_descriptor(fd)

    def _pin(self, name: str, *, create: bool = False) -> _File:
        assert self.root_fd is not None
        try:
            fd = open_file_at(
                self.root_fd, name, os.O_RDWR | os.O_CREAT | os.O_EXCL | _FLAGS,
            ) if create else open_file_at(
                self.root_fd, name, os.O_RDONLY | _FLAGS,
            )
        except FileExistsError:
            fd = open_file_at(self.root_fd, name, os.O_RDWR | _FLAGS)
        try:
            value = os.fstat(fd)
            try:
                validate_private_descriptor(fd)
            except AMLError:
                raise AMLError(
                    "audit_ownership", "Audit files must be private and singly linked",
                ) from None
            pinned = _File(name, fd, _identity(value))
            self.files[name] = pinned
            return pinned
        except BaseException as error:
            # Until registration this frame alone owns the descriptor, including
            # native metadata failures and asynchronous interruption.
            try:
                os.close(fd)
            except OSError as cleanup:
                error.add_note(f"Audit pin cleanup also failed: errno {cleanup.errno}")
            raise

    def validate(self, *, closing: bool = False) -> None:
        if self.pid != os.getpid() or self.root_fd is None:
            raise AMLError("audit_closed", "Audit ownership is unavailable in this process")
        validate_private_descriptor(self.root_fd, directory=True)
        held = _identity(os.fstat(self.root_fd))
        current = _identity(os.stat(self.root, follow_symlinks=False))
        if self.root_identity is None or held[:4] != self.root_identity[:4] or held != current:
            raise AMLError("audit_ownership", "Audit directory identity changed")
        for name, pinned in tuple(self.files.items()):
            if os.name == "nt":
                validate_private_descriptor(pinned.fd, single_link=False)
            value = _identity(os.fstat(pinned.fd))
            try:
                named = _identity(stat_at(self.root_fd, name))
            except FileNotFoundError:
                if closing and name in self._companions() and value[:4] == pinned.identity[:4] \
                        and value[4] == 0:
                    continue
                raise
            if value != pinned.identity or named != pinned.identity:
                raise AMLError("audit_ownership", "Audit file identity or permissions changed")
        for name in self._companions():
            if name not in self.files:
                try:
                    self._pin(name)
                except FileNotFoundError:
                    continue

    def _companions(self) -> tuple[str, ...]:
        return tuple(self.path.name + suffix for suffix in ("-wal", "-shm", "-journal"))

    def close(self) -> None:
        error: Exception | None = None
        try:
            if self.locked:
                self.validate(closing=True)
        except Exception as caught:
            error = caught
        finally:
            while self.files:
                _, pinned = self.files.popitem()
                try:
                    os.close(pinned.fd)
                except Exception as caught:
                    if error is None:
                        error = caught
            if self.root_fd is not None:
                fd, self.root_fd = self.root_fd, None
                try:
                    os.close(fd)
                except Exception as caught:
                    if error is None:
                        error = caught
            self.locked = False
        if error is not None:
            raise AMLError("audit_ownership", "Audit ownership changed during close") from None


def _schema() -> EntitySchema:
    return EntitySchema(_ENTITY, "CompletedPurgeReceipt", fields={
        "object_id": FieldSchema("object_id", hb.Identifier(max_chars=64), pk=True, required=True),
        "receipt_json": FieldSchema(
            "receipt_json", hb.LongText(max_chars=65535), required=True,
        ),
    })


def _layout(backend: SQLiteBackend, *, empty: bool = False) -> None:
    """Inspect the physical contract before allowing HB's additive ensure operation."""
    db = backend.db
    if db is None:
        raise AMLError("audit_unavailable", "Audit backend is closed")
    objects = db.execute(
        "SELECT type, name FROM sqlite_schema WHERE name NOT LIKE 'sqlite_%' ORDER BY name",
        readonly=True,
    ).to_dicts()
    app_id = db.execute("PRAGMA application_id", readonly=True).scalar()
    version = db.execute("PRAGMA user_version", readonly=True).scalar()
    if empty and objects == [] and app_id == 0 and version == 0:
        return
    columns = db.execute(f"PRAGMA table_xinfo({_TABLE})", readonly=True).to_dicts()
    expected = [
        {"cid": 0, "name": "object_id", "type": "VARCHAR(64)", "notnull": 1,
         "dflt_value": None, "pk": 1, "hidden": 0},
        {"cid": 1, "name": "receipt_json", "type": "VARCHAR(65535)", "notnull": 1,
         "dflt_value": None, "pk": 0, "hidden": 0},
    ]
    indexes = db.execute(f"PRAGMA index_list({_TABLE})", readonly=True).to_dicts()
    inspector = inspect(db.engine)
    if inspector is None:
        raise AMLError("audit_schema_mismatch", "Audit schema cannot be inspected")
    checks = cast(list[dict[str, Any]], inspector.get_check_constraints(_TABLE))
    actual_checks = sorted("".join(str(row["sqltext"]).split()) for row in checks)
    expected_checks = sorted(
        f'typeof("{name}")IN(\'text\',\'null\')' for name in ("object_id", "receipt_json")
    )
    if (
        app_id != _APPLICATION_ID or version != 1
        or objects != [{"type": "table", "name": _TABLE}] or columns != expected
        or len(indexes) != 1 or indexes[0]["origin"] != "pk"
        or indexes[0]["unique"] != 1 or indexes[0]["partial"] != 0
        or actual_checks != expected_checks
        or db.execute(f"PRAGMA foreign_key_list({_TABLE})", readonly=True).to_dicts()
        or db.execute("PRAGMA ignore_check_constraints", readonly=True).scalar() != 0
    ):
        raise AMLError("audit_schema_mismatch", "Existing audit storage has another layout")


class BenchAudit:
    """Own a short audit-only HB session with immutable closed records.

    Args:
        path (Path): Caller-selected audit database outside all job workspaces.
            Its immediate parent must be private; a missing parent is created.

    Notes:
        Use as a context manager. One nonblocking file lock covers this audit
        session only. No job status, Writer, workspace, providers or purge work
        is created. The caller owns console input, positive retained-data proof
        and completed deletion; shape validation cannot establish those facts.
        This separate SQLite database uses WAL with synchronous FULL. Normal
        job storage policy is unchanged; this is not a power-loss test claim.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._stack: ExitStack | None = None
        self._files: _AuditFiles | None = None
        self._backend: SQLiteBackend | None = None
        self._pid: int | None = None
        self._failed = False
        self._mutex = threading.RLock()

    def __enter__(self) -> Self:
        if self._stack is not None or self._pid is not None:
            raise AMLError("audit_closed", "An audit session cannot be reopened")
        self._pid = os.getpid()
        stack = self._stack = ExitStack()
        try:
            files = self._files = _AuditFiles(self.path)
            stack.callback(files.close)
            files.acquire()
            fresh = os.fstat(files.files[files.path.name].fd).st_size == 0
            temporary = stack.enter_context(TemporaryDirectory(prefix="aml-audit-control-"))
            config = ConfigManager(root=temporary, setup=True)
            config.default["heavenbase"]["llm"]["cache"]["enabled"] = False
            config.default["heavenbase"]["dashboard"]["assistant"]["session_log"] = False
            config.default["heavenbase"]["extensions"]["default"] = []
            pragmas = config.default["heavenbase"]["db"]["providers"]["sqlite"]["pragmas"]
            config.default["heavenbase"]["db"]["providers"]["sqlite"]["pragmas"] = [
                *pragmas, "PRAGMA synchronous = FULL;",
            ]
            context = Context(BootstrapSpec(temporary, "aml-audit", {
                "type": "inmem", "module": "heavenbase.backends.inmem.backend",
                "name": "audit-control",
            }, {"hash": "sha256", "max_items": 65536, "max_bytes": 4194304}), config=config)
            stack.callback(context.close)
            resolver = context.modules()
            resolver.setup()
            verify_privacy(config)
            files.validate()
            backend = self._backend = SQLiteBackend(
                name="audit", database=files.path.as_uri(), resolver=resolver, config=config,
            )
            db = backend.db
            if db is None:
                raise AMLError("audit_unavailable", "Audit backend has no database")
            stack.callback(db.dispose)
            engine = db.engine
            if engine is None:
                raise AMLError("audit_unavailable", "Audit database has no engine")
            keeper = engine.connect()
            stack.callback(keeper.close)
            files.validate()
            if db.execute("PRAGMA journal_mode", readonly=True).scalar() != "wal" \
                    or db.execute("PRAGMA synchronous", readonly=True).scalar() != 2:
                raise AMLError("audit_storage_invalid", "Audit storage requires SQLite WAL/FULL")
            _layout(backend, empty=fresh)
            schema = _schema()
            bindings = [StorageBinding(
                _ENTITY, name, backend.name, "inline", "audit", resolver=resolver,
            ) for name in schema.fields]
            backend.ensure(schema, bindings)
            if fresh:
                with db.tx():
                    db.execute(f"PRAGMA application_id = {_APPLICATION_ID}", readonly=False)
                    db.execute("PRAGMA user_version = 1", readonly=False)
            _layout(backend)
            files.validate()
            return self
        except BaseException as error:
            self._failed = True
            try:
                stack.close()
            except Exception:
                pass
            self._stack = None
            if isinstance(error, AMLError | KeyboardInterrupt | SystemExit):
                raise
            raise AMLError("audit_unavailable", "Audit storage could not be opened") from None

    def _live(self) -> tuple[_AuditFiles, SQLiteBackend]:
        if self._failed or self._pid != os.getpid() or self._stack is None \
                or self._files is None or self._backend is None:
            raise AMLError("audit_closed", "Audit session is closed or unavailable")
        self._files.validate()
        return self._files, self._backend

    def get(self, job_id: str, workspace_path_hash: str) -> CompletedPurgeReceipt | None:
        """Return a detached validated original receipt for this exact job/path identity.

        Args:
            job_id (str): Exact external job ID, never a normalized replacement.
            workspace_path_hash (str): Previously established canonical workspace path hash.

        Returns:
            CompletedPurgeReceipt | None: Original versioned payload, or no receipt.

        Raises:
            AMLError: Invalid identity, unavailable ownership or corrupt stored receipt.
        """
        return self.read(job_id, workspace_path_hash).receipt

    def read(self, job_id: str, workspace_path_hash: str) -> AuditRecords:
        """Read all original audit facts with strict joint identity and timing checks.

        Args:
            job_id: Exact opaque external job identity.
            workspace_path_hash: Established canonical workspace path hash.

        Returns:
            Detached records, including legacy receipts with unknown completion.

        Raises:
            AMLError: Invalid identity, corrupt/inconsistent history or unavailable
                ownership. A read never repairs a missing late-purge violation.
        """
        keys = (
            _key(job_id, workspace_path_hash),
            _event_key("console_completion", job_id, workspace_path_hash),
            _event_key("retention_violation", job_id, workspace_path_hash),
        )
        with self._mutex:
            return self._read(job_id, workspace_path_hash, keys)

    def _read(self, job_id: str, path_hash: str, keys: tuple[str, str, str]) -> AuditRecords:
        try:
            files, backend = self._live()
            rows = backend.get_many(_ENTITY, list(keys))
            files.validate()
            payloads: list[str | None] = []
            for row, key in zip(rows, keys, strict=True):
                if row is None:
                    payloads.append(None)
                elif row["object_id"] == key and type(row["receipt_json"]) is str:
                    payloads.append(row["receipt_json"])
                else:
                    raise AMLError("audit_inconsistent", "Stored audit record key is invalid")
            receipt_raw, completion_raw, violation_raw = payloads
            receipt = None if receipt_raw is None else parse_completed_purge_receipt(
                receipt_raw
            )
            completion = (
                None if completion_raw is None
                else ConsoleCompletionV1.model_validate_json(completion_raw)
            )
            violation = None if violation_raw is None else _VIOLATION.validate_json(violation_raw)
            for record, payload in zip((receipt, completion, violation), payloads, strict=True):
                if record is not None and (
                    record.job_id != job_id or record.workspace_path_hash != path_hash
                    or canonical_json(record.model_dump(mode="json")) != payload
                ):
                    raise AMLError("audit_inconsistent", "Stored audit record binding is invalid")
            records = AuditRecords(completion, violation, receipt)
            _validate_records(records)
            return records
        except BaseException as error:
            self._failed = True
            if isinstance(error, KeyboardInterrupt | SystemExit):
                raise
            raise AMLError("audit_unavailable", "Audit records could not be read safely") from None

    def record_completion(self, completion: ConsoleCompletionV1) -> ConsoleCompletionV1:
        """Persist the first supplied console confirmation for a verified existing job.

        Args:
            completion: Explicit operator completion and trusted recording time.
                The caller must first verify the actual job and canonical root
                with its same exclusive existing-job owner.

        Returns:
            The original confirmation; exact completion replay retains its first
            recording time. A replay is a new observation and rejects clock rollback.

        Raises:
            AMLError: Invalid/conflicting input, clock rollback, unavailable audit,
                or attempted retrospective confirmation based on a completed receipt.
        """
        try:
            checked = ConsoleCompletionV1.model_validate(
                completion.model_dump(mode="json", warnings=False)
            )
        except Exception:
            raise AMLError("audit_completion_invalid", "Console confirmation is invalid") from None
        with self._mutex:
            records = self.read(checked.job_id, checked.workspace_path_hash)
            if records.completion is not None:
                if checked.completed_at != records.completion.completed_at:
                    raise AMLError("audit_completion_conflict", "Completion time is already bound")
                records.validate_clock(checked.recorded_at)
                return records.completion
            if records.receipt is not None:
                raise AMLError("audit_completion_closed", "A receipt cannot confirm completion")
            self._write([(_event_key(
                "console_completion", checked.job_id, checked.workspace_path_hash
            ), checked)])
            return checked

    def record_retained(self, violation: RetainedDataViolationV1) -> RetainedDataViolationV1:
        """Persist the first positive overdue retained-data observation before deletion.

        Args:
            violation: Exact stored completion identity and actual trusted observation
                time. The caller must positively verify retained job data under the
                same exclusive existing-job owner. Missing files are not that proof.

        Returns:
            The first observation, unchanged by subsequent valid observations.

        Raises:
            AMLError: Invalid input, missing/conflicting completion, completed deletion,
                clock rollback or unavailable audit. Failure before purge prevents deletion.
        """
        try:
            checked = RetainedDataViolationV1.model_validate(
                violation.model_dump(mode="json", warnings=False)
            )
        except Exception:
            raise AMLError(
                "audit_violation_invalid", "Retained-data observation is invalid",
            ) from None
        with self._mutex:
            records = self.read(checked.job_id, checked.workspace_path_hash)
            if records.completion is None or (
                checked.completed_at != records.completion.completed_at
            ):
                raise AMLError(
                    "audit_completion_conflict", "Observation has no matching completion",
                )
            records.validate_clock(checked.observed_at)
            if records.receipt is not None:
                raise AMLError("audit_completion_closed", "Deletion is already recorded")
            if records.violation is not None:
                # Joint read validation excludes a completed-purge variant here.
                assert isinstance(records.violation, RetainedDataViolationV1)
                return records.violation
            self._write([(_event_key(
                "retention_violation", checked.job_id, checked.workspace_path_hash
            ), checked)])
            return checked

    def append(self, receipt: CompletedPurgeReceipt) -> bool:
        """Persist a validated completed receipt once, without asserting deletion success.

        Args:
            receipt: Completion evidence supplied by the purge owner, with its exact version.

        Returns:
            bool: True for a new committed receipt; False for an exact existing duplicate.

        Raises:
            AMLError: Invalid receipt, conflicting job/path content or unavailable storage.
                A failure after commit leaves an unknown acknowledgement; retry this same
                original receipt rather than replacing its deletion time or counts.
        """
        try:
            checked = parse_completed_purge_receipt(
                receipt.model_dump(mode="json", warnings=False)
            )
            payload = canonical_json(checked.model_dump(mode="json"))
        except Exception:
            raise AMLError("audit_receipt_invalid", "Completed audit receipt is invalid") from None
        with self._mutex:
            return self._append(checked, payload)

    def _append(self, checked: CompletedPurgeReceipt, payload: str) -> bool:
        records = self.read(checked.job_id, checked.workspace_path_hash)
        existing = records.receipt
        if existing is not None:
            if canonical_json(existing.model_dump(mode="json")) != payload:
                raise AMLError("audit_receipt_conflict", "Audit identity has another receipt")
            return False
        records.validate_clock(checked.deleted_at)
        pending: list[tuple[str, CompletedPurgeReceipt | CompletedPurgeViolationV1]] = [
            (_key(checked.job_id, checked.workspace_path_hash), checked),
        ]
        if records.completion is not None and records.violation is None and (
            checked.deleted_at > completion_deadline(records.completion.completed_at)
        ):
            pending.append((_event_key(
                "retention_violation", checked.job_id, checked.workspace_path_hash,
            ), CompletedPurgeViolationV1(
                event_kind="retention_violation", basis="completed_purge",
                job_id=checked.job_id, workspace_path_hash=checked.workspace_path_hash,
                completed_at=records.completion.completed_at,
            )))
        self._write(pending)
        return True

    def _write(self, records: Sequence[tuple[
        str, CompletedPurgeReceipt | ConsoleCompletionV1
        | RetainedDataViolationV1 | CompletedPurgeViolationV1,
    ]]) -> None:
        try:
            files, backend = self._live()
            # HB owns this single transaction, including receipt + late violation.
            backend.upsert([RowOp(_ENTITY, key, {
                "receipt_json": canonical_json(record.model_dump(mode="json")),
            }) for key, record in records])
            files.validate()
        except BaseException as error:
            self._failed = True
            if isinstance(error, KeyboardInterrupt | SystemExit):
                raise
            raise AMLError("audit_unavailable", "Audit commit was not confirmed") from None

    def close(self) -> None:
        """Dispose HB before releasing ownership; never delete audit data or lock identity."""
        with self._mutex:
            self._close()

    def _close(self) -> None:
        if self._pid is not None and self._pid != os.getpid():
            raise AMLError("audit_closed", "An inherited audit session cannot be closed here")
        stack, self._stack = self._stack, None
        if stack is not None:
            try:
                stack.close()
            except Exception:
                raise AMLError("audit_unavailable", "Audit close failed") from None

    def __exit__(self, *_: object) -> None:
        self.close()
