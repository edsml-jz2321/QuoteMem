"""Offline existing bench metadata beneath one retained filesystem owner.

This path has no provider construction, credentials, Writer, Context, epoch
adoption, Catalog repair, workflow recovery, or new workspace initialization.
The service owns status formatting, deletion and receipts. A successful storage
close is required before deletion while the same outer owner remains held.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Literal, Protocol, Self

from pydantic import ConfigDict, ValidationError, model_validator

from quotemem.core.bootstrap import (
    WorkspaceConfiguration,
    _configuration,  # pyright: ignore[reportPrivateUsage]
)
from quotemem.core.controls import ControlRowV1, JobValueV1, PolicyPointerV1, WorkspaceBindingV1
from quotemem.core.errors import QuoteMemError
from quotemem.core.extraction_policy import ExtractionPolicyConfigV2
from quotemem.core.identity import canonical_json, control_id
from quotemem.core.platform_files import open_private_file, validate_private_path
from quotemem.core.retained_policy import read_retained_bootstrap_policy
from quotemem.core.technical_types import NonNegativeInt
from quotemem.providers.auth import SingleUserAuth
from quotemem.providers.bindings import job_provider_bindings
from quotemem.providers.config import ProviderConfig
from quotemem.providers.keyring import LocalKeyring
from quotemem.store.hb_runtime import OWNER_LOCK_NAME, StorageOwner
from quotemem.store.metadata import StrictModel
from quotemem.store.offline import ExistingStore
from quotemem.store.schemas import schema_fingerprint


class ExistingStorageOwner(StorageOwner, Protocol):
    """Retain the ordinary owner with an explicit no-create database operation."""

    def prepare_databases(self, *, existing_only: bool = False) -> None:
        """Pin existing files without creating missing files when requested."""
        ...


class ExistingJobMetadata(StrictModel):
    """Detached nonsecret metadata; no source, storage handle or execution grant."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    workspace_id: str
    db_uuid: str
    tenant_digest: str
    embedding_dimension: Literal[1024, 2048]
    job_id: str
    # Canonical JSON keeps nested binding models immutable across the public seam.
    job_control_json: str
    row_counts: tuple[tuple[str, NonNegativeInt], ...]

    @model_validator(mode="after")
    def complete_counts(self) -> Self:
        """Require the complete current entity inventory in stable order."""
        from quotemem.store.entities import SCHEMA_SPECS

        if tuple(kind for kind, _ in self.row_counts) != tuple(sorted(SCHEMA_SPECS)):
            raise ValueError("Offline counts require one complete sorted entity inventory")
        return self

    @property
    def job_control(self) -> ControlRowV1:
        """Return a strict detached copy, never a mutable stored authority."""
        return ControlRowV1.model_validate_json(self.job_control_json)


def _existing_files(root: Path) -> None:
    """Reject missing identities before an owner can create a lock or database."""
    validate_private_path(root, directory=True)
    for name in (OWNER_LOCK_NAME, "workspace.json", "tenant.key", "memory.db", "hb-system.db"):
        descriptor = open_private_file(root / name, os.O_RDONLY)
        try:
            if name.endswith(".db") and os.fstat(descriptor).st_size < 100:
                raise QuoteMemError("workspace_binding_mismatch", "Existing database is incomplete")
        finally:
            os.close(descriptor)


def _control(store: ExistingStore, key: str) -> ControlRowV1:
    raw = store.get_control(key)
    if raw is None:
        raise QuoteMemError("reference_missing", "Existing job metadata is incomplete")
    row = ControlRowV1.model_validate(raw)
    if row.object_id != control_id(key) or row.key != key:
        raise QuoteMemError("binding_mismatch", "Existing Control has another identity")
    return row


def _metadata(
    store: ExistingStore, stored: WorkspaceConfiguration, keyring: LocalKeyring, job_id: str,
) -> ExistingJobMetadata:
    binding = _control(store, "binding:workspace")
    if not isinstance(binding.value, WorkspaceBindingV1) or (
        binding.control_refs or binding.updated_by_run_id is not None
    ):
        raise QuoteMemError("binding_mismatch", "Workspace binding has unexpected provenance")
    value = binding.value
    expected = (
        stored.workspace_id, stored.db_uuid,
        hashlib.sha256(stored.tenant_id.encode()).hexdigest(),
        stored.providers.embedding_dimensions,
        schema_fingerprint(
            stored.providers.embedding_dimensions,
        ),
    )
    if (value.workspace_id, value.db_uuid, value.tenant_digest,
        value.embedding_dimension, value.schema_fingerprint) != expected:
        raise QuoteMemError("workspace_binding_mismatch", "Stored workspace identity differs")
    jobs = list(store.job_rows())
    if not jobs:
        raise QuoteMemError("reference_missing", "Existing workspace has no job binding")
    if len(jobs) != 1:
        raise QuoteMemError("workspace_binding_mismatch", "Existing workspace has multiple jobs")
    job = ControlRowV1.model_validate(jobs[0])
    if job.key != f"job:{job_id}" or job.object_id != control_id(job.key):
        raise QuoteMemError("workspace_binding_mismatch",
                            "Existing workspace belongs to another job")
    if not isinstance(job.value, JobValueV1) or job.control_refs or (
        job.updated_by_run_id is not None
    ):
        raise QuoteMemError("binding_mismatch", "Existing job has unexpected provenance")
    pointer = _control(store, "policy:extraction")
    if not isinstance(pointer.value, PolicyPointerV1):
        raise QuoteMemError("binding_mismatch", "Existing extraction pointer has another kind")
    extraction, _ = read_retained_bootstrap_policy(
        store, keyring, SingleUserAuth(keyring, []).principal_id,
        "extraction", pointer.value.current_release_ref,
    )
    if not isinstance(extraction, ExtractionPolicyConfigV2) or (
        job.value.provider_bindings != job_provider_bindings(stored.providers, extraction)
    ):
        raise QuoteMemError("binding_mismatch",
                            "Existing job differs from the retained configuration")
    return ExistingJobMetadata(
        workspace_id=stored.workspace_id, db_uuid=stored.db_uuid,
        tenant_digest=value.tenant_digest, embedding_dimension=value.embedding_dimension,
        job_id=job_id, job_control_json=canonical_json(job),
        row_counts=store.row_counts(),
    )


class ExistingJobResources:
    """Close metadata storage while retaining the supplied owner's outer lifetime."""

    def __init__(
        self, metadata: ExistingJobMetadata, store: ExistingStore, owner: StorageOwner,
        *, root: Path,
    ) -> None:
        if not root.is_absolute() or owner.root != root or store.root != root or (
            store.owner is not owner
        ):
            raise QuoteMemError("workspace_binding_mismatch",
                                "Offline resources changed their owner")
        self._root = root
        self._metadata = metadata.model_copy(deep=True)
        self._store = store
        self._owner = owner
        self._closed = False
        self._failed = False

    @property
    def root(self) -> Path:
        """Return the immutable path selected when this existing store was opened."""
        return self._root

    @property
    def owner(self) -> StorageOwner:
        """Return the exact borrowed owner; this resource never reacquires it."""
        return self._owner

    @property
    def store(self) -> ExistingStore:
        """The read-only store, valid until ``close_storage``."""
        return self._store

    @property
    def metadata(self) -> ExistingJobMetadata:
        """Return immutable content-free identity while ownership remains valid."""
        self.validate_owner()
        return ExistingJobMetadata.model_validate(self._metadata.model_dump())

    def validate_owner(self) -> None:
        """Fail closed after a resource failure or loss of the retained owner."""
        if self._failed:
            raise QuoteMemError("service_unhealthy", "Offline resources previously failed")
        try:
            if self._owner.root != self._root:
                raise QuoteMemError("workspace_binding_mismatch", "Offline owner changed its root")
            self._owner.validate()
            if not self._closed:
                self._store.validate()
        except BaseException:
            self._failed = True
            raise

    def close_storage(self) -> None:
        """Dispose all storage before deletion; failure never permits a retry purge."""
        if self._closed:
            if self._failed:
                raise QuoteMemError("service_unhealthy", "Offline resources previously failed")
            return
        try:
            self._store.close()
            if self._owner.root != self._root:
                raise QuoteMemError("workspace_binding_mismatch", "Offline owner changed its root")
            self._owner.validate()
            if self._failed:
                raise QuoteMemError("service_unhealthy", "Offline resources previously failed")
        except BaseException:
            self._failed = True
            raise
        finally:
            self._closed = True

    def require_storage_closed(self) -> None:
        """Prove acknowledged disposal while this exact filesystem owner remains held."""
        self.validate_owner()
        if not self._closed:
            raise QuoteMemError("service_unhealthy", "Offline storage has not acknowledged closure")


@contextmanager
def open_existing_job(
    root: Path, job_id: str, *, owner: ExistingStorageOwner,
    config: ProviderConfig | None = None,
) -> Iterator[ExistingJobResources]:
    """Validate one existing job without credentials, initialization or recovery.

    The caller supplies the same strict process owner used by serving. Missing
    files, incomplete metadata or interrupted WAL state fail closed. This opens
    only a fully closed workspace; crash recovery remains the normal initializer's
    separate responsibility. The owner outlives explicit ``close_storage`` so the
    service can delete verified files without releasing and reacquiring its lock.
    """
    if type(job_id) is not str or not 1 <= len(job_id) <= 1020:
        raise QuoteMemError("schema_invalid", "Job identity must contain 1 to 1020 characters")
    try:
        _existing_files(root)
        selected_root = root.resolve(strict=True)
        if owner.root.resolve(strict=True) != selected_root:
            raise QuoteMemError("workspace_binding_mismatch", "Offline owner has another root")
        with owner as held:
            if held is not owner or owner.root != selected_root:
                raise QuoteMemError("workspace_binding_mismatch", "Offline owner changed identity")
            _existing_files(selected_root)
            stored, keyring = _configuration(selected_root, config, existing_only=True)
            if stored.providers.profile != "bench":
                raise QuoteMemError("unsupported_variant", "Offline job metadata requires bench")
            owner.probe_storage()
            _existing_files(selected_root)
            owner.prepare_databases(existing_only=True)
            store = ExistingStore(
                selected_root, stored.workspace_id, stored.tenant_id,
                stored.providers.embedding_dimensions, owner,
            )
            try:
                metadata = _metadata(store, stored, keyring, job_id)
                # Drop the avoidable key owner before yielding or purge. Python
                # immutable bytes do not provide a physical zeroization promise.
                del keyring
                result = ExistingJobResources(metadata, store, owner, root=selected_root)
            except BaseException:
                store.close()
                raise
            try:
                yield result
            finally:
                result.close_storage()
    except ValidationError:
        raise QuoteMemError("schema_invalid", "Existing job metadata is malformed") from None
    except OSError:
        raise QuoteMemError(
            "workspace_binding_mismatch", "Existing workspace files are unavailable",
        ) from None
