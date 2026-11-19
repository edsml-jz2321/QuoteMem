"""Restore historical V1 wire images from the one normalized physical row.

These images serve existing immutable-row and integrity consumers. They are not
a second live Item interface, a storage resolver, or admission authority. The
caller supplies the verified workspace tenant and actual selected dependencies;
this codec never substitutes a newer head or invents missing creation support.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from pydantic import TypeAdapter

from aml.core.controls import ControlRowV1
from aml.core.errors import AMLError
from aml.core.identity import canonical_json
from aml.store.domain import (
    DOMAIN_ADAPTERS,
    EvidenceInteractionRecallRowV1,
    MemoryInteractionRecallRowV1,
    MinimalRecallRowV1,
    ReplayableRecallRowV1,
    TenantId,
    validate_interaction_parent,
)
from aml.store.metadata import StrictModel
from aml.store.normalization import (
    NORMALIZED_SCHEMA_SPECS,
    RECALL_PARENT_FIELDS,
    project_artifact_row,
    project_checked_normalized_row,
)

_TENANT: TypeAdapter[str] = TypeAdapter(TenantId)
_EVIDENCE_ROLES = {
    "user_message": "user",
    "assistant_message": "assistant",
    "system_message": "system",
    "tool_result": "tool",
}


def restore_v1_row_image(
    kind: str,
    physical: Mapping[str, Any],
    *,
    tenant_id: str,
    parent_recall: StrictModel | None = None,
) -> dict[str, Any]:
    """Recover one exact historical logical image without opening its references.

    Args:
        kind: Existing entity kind in the normalized schema contract.
        physical: Complete HB-decoded physical fields, including explicit nulls.
        tenant_id: Mandatory trusted tenant from the verified workspace binding.
        parent_recall: Interaction's actual retained delivery receipt. An
            interaction cannot act as a parent, so dependency cycles are refused.

    Returns:
        Detached JSON-compatible fields of the existing V1 historical row. Its
        canonical bytes reproduce the original V1 row's integrity input; Artifact
        body strings and all retained physical values remain exact.

    Raises:
        AMLError: On missing/extra fields, malformed rows, wrong or extraneous
            dependencies, tenant mismatch, or failed exact projection parity.

    The caller still owns provenance, existence, admission, privacy and fences.
    This function grants none of them and performs no source or database reads.
    """
    spec = NORMALIZED_SCHEMA_SPECS.get(kind)
    if spec is None:
        raise AMLError("unsupported_variant", "Unknown historical row entity")
    if set(physical) != {"object_id", *spec.fields}:
        raise AMLError("schema_mismatch", "Historical row requires exact physical fields")
    raw = dict(physical)
    interaction = kind == "recall" and raw["record_class"] == "interaction"
    if parent_recall is not None and not interaction:
        raise AMLError("authority_unavailable", "Historical row received an unrelated authority")
    try:
        _TENANT.validate_python(tenant_id, strict=True)
        parent: MinimalRecallRowV1 | ReplayableRecallRowV1 | None = None
        checked: StrictModel
        projected: dict[str, Any] | None = None
        if kind in {"evidence", "artifact", "recall"}:
            raw["tenant_id"] = tenant_id
        if kind == "evidence":
            raw["role"] = None if raw["state"] == "erased" else _EVIDENCE_ROLES.get(raw["kind"])
            raw["byte_end"] = raw["byte_start"] + len(raw["text"].encode("utf-8"))
        elif kind == "artifact":
            raw["size_bytes"] = len(raw["body"].encode("utf-8"))
        elif interaction:
            if parent_recall is None:
                raise AMLError("authority_unavailable", "Interaction requires its delivery")
            supplied = DOMAIN_ADAPTERS["recall"].validate_python(
                parent_recall.model_dump(mode="json"),
            )
            if not isinstance(supplied, MinimalRecallRowV1 | ReplayableRecallRowV1):
                raise AMLError("authority_unavailable", "Interaction parent is not a delivery")
            parent = supplied
            raw.update({name: getattr(parent, name) for name in RECALL_PARENT_FIELDS})
        if kind == "artifact":
            checked, projected = project_artifact_row(raw, tenant_id=tenant_id)
        elif kind == "control":
            checked = ControlRowV1.model_validate(raw)
        else:
            checked = cast(StrictModel, DOMAIN_ADAPTERS[kind].validate_python(raw))
        if interaction:
            if not isinstance(
                checked, MemoryInteractionRecallRowV1 | EvidenceInteractionRecallRowV1,
            ):
                raise AMLError("schema_invalid", "Recall interaction has no supported target")
            assert parent is not None
            validate_interaction_parent(checked, parent)
        if projected is None:
            projected = project_checked_normalized_row(
                kind, checked, tenant_id=tenant_id, parent_recall=parent,
            )
        if canonical_json(projected) != canonical_json(physical):
            raise AMLError("parity_mismatch", "Historical image changed retained physical values")
        return checked.model_dump(mode="json")
    except AMLError:
        raise
    except (TypeError, ValueError, AttributeError) as error:
        raise AMLError(
            "schema_invalid", "Normalized historical row or authority is malformed",
        ) from error
