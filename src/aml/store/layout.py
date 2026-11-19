"""Current physical layout selected by the immutable workspace binding."""

from __future__ import annotations

import hashlib
from enum import Enum

from aml.core.controls import ControlRowV1, WorkspaceBindingV1
from aml.core.errors import AMLError
from aml.store.schemas import schema_fingerprint


class StorageLayout(Enum):
    """The current physical declaration, never a mutable configuration."""

    NORMALIZED = "normalized"

    @property
    def normalized_storage(self) -> bool:
        """Return the existing Writer's explicit physical encoding selection."""
        return self is StorageLayout.NORMALIZED


def layout_for_binding(
    binding: ControlRowV1, *, workspace_id: str, db_uuid: str | None,
    tenant_id: str, dimension: int,
) -> StorageLayout:
    """Verify the actual binding and resolve only code-owned fingerprints.

    This pure check grants no storage access and performs no migration. The
    caller must also verify the physical declaration before registering schemas.
    ``db_uuid=None`` leaves that identity comparison to the offline metadata
    consumer, which already compares it to the stored workspace configuration.
    """
    row = ControlRowV1.model_validate(binding.model_dump(mode="python"))
    value = row.value
    if row.key != "binding:workspace" or not isinstance(value, WorkspaceBindingV1) or (
        row.control_refs or row.updated_by_run_id is not None
    ):
        raise AMLError("workspace_binding_mismatch", "Workspace binding provenance differs")
    if (value.workspace_id != workspace_id
        or (db_uuid is not None and value.db_uuid != db_uuid)
        or value.tenant_digest != hashlib.sha256(tenant_id.encode()).hexdigest()
        or value.embedding_dimension != dimension):
        raise AMLError("workspace_binding_mismatch", "Stored workspace identity differs")
    if value.schema_fingerprint == schema_fingerprint(dimension):
        return StorageLayout.NORMALIZED
    raise AMLError("schema_mismatch", "Workspace binding has an unknown physical fingerprint")
