"""Authority-checked physical projections of the twelve QuoteMem tables.

These pure functions neither admit a migration nor open a workspace. A row of
the memory tables (item, revision, link, candidate), the entity tables and
the tombstone table is stored exactly as its strict model dumps it. Evidence,
Recall and Artifact rows drop the columns the verified workspace already
determines, the tenant above all (the tenant is never a physical column).
The sole Writer still owns references, lifecycle and cross-row parity.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from pydantic import TypeAdapter

from quotemem.core.artifacts import ArtifactRowV1, parse_artifact_body
from quotemem.core.controls import ControlRowV1
from quotemem.core.errors import QuoteMemError
from quotemem.store.artifact_read import verify_artifact_refs
from quotemem.store.domain import DOMAIN_ADAPTERS, TenantId
from quotemem.store.entities import SCHEMA_SPECS, PhysicalSchemaSpec
from quotemem.store.metadata import StrictModel

RECALL_PARENT_FIELDS = ("scope_kind", "scope_id", "environment", "principal_id")
_TENANT_ID: TypeAdapter[str] = TypeAdapter(TenantId)
REMOVED_FIELDS: dict[str, frozenset[str]] = {
    "evidence": frozenset(("tenant_id", "role", "byte_end")),
    "recall": frozenset(("tenant_id",)),
    "artifact": frozenset(("tenant_id", "size_bytes")),
}
LOGICAL_IMAGE_KINDS = frozenset(REMOVED_FIELDS)
"""Kinds whose logical row adds columns the verified workspace determines."""


def normalized_schema_specs() -> dict[str, PhysicalSchemaSpec]:
    """Derive physical types from the existing schema owner, with exact removals.

    This is a new schema contract. Registering it over a legacy workspace is not
    an in-place migration; the caller must retain the old workspace untouched.
    """
    result: dict[str, PhysicalSchemaSpec] = {}
    for kind, original in SCHEMA_SPECS.items():
        removed = REMOVED_FIELDS.get(kind, frozenset())
        if not removed <= original.fields.keys():
            raise QuoteMemError("schema_mismatch", "Normalization source fields changed")
        fields = {name: dtype for name, dtype in original.fields.items() if name not in removed}
        optional = original.optional - removed
        if kind == "recall":
            optional |= frozenset(RECALL_PARENT_FIELDS)
        elif kind == "evidence":
            fields["source_participant"] = "json"
            optional |= frozenset({"source_participant"})
        result[kind] = PhysicalSchemaSpec(
            original.identifier, fields, optional, original.unique, original.indexes,
        )
    return result


NORMALIZED_SCHEMA_SPECS = normalized_schema_specs()


def _checked_artifact(row: StrictModel | Mapping[str, Any]) -> ArtifactRowV1:
    return _checked_artifact_body(row)[0]


def _checked_artifact_body(
    row: StrictModel | Mapping[str, Any],
) -> tuple[ArtifactRowV1, StrictModel | None]:
    raw = row.model_dump(mode="json") if isinstance(row, StrictModel) else dict(row)
    # Validated without a derived address, the row proves its address and byte
    # count (``verify_artifact_identity``'s checks) and parses its body once; that
    # strict body is the one whose reference projections are proved here.
    parsed: list[StrictModel] = []
    checked = ArtifactRowV1.model_validate(raw, context={"parsed_body": parsed})
    # Manifest projections belong to the resolved logical body, not its header.
    # Physical normalization retains them byte-for-byte and does not resolve parts.
    body = None
    if checked.artifact_kind not in {"part", "manifest"}:
        body = (parsed[0] if len(parsed) == 1
                else parse_artifact_body(checked.artifact_kind, checked.body))
        verify_artifact_refs(checked, body, address_verified=True)
    return checked, body


def _checked_row(kind: str, row: StrictModel) -> StrictModel:
    if kind == "artifact":
        return _checked_artifact(row)
    if kind == "control":
        return ControlRowV1.model_validate(row.model_dump(mode="json"))
    adapter = DOMAIN_ADAPTERS.get(kind)
    if adapter is None:
        raise QuoteMemError("unsupported_variant", "Unknown normalization entity")
    return cast(StrictModel, adapter.validate_python(row.model_dump(mode="json")))


def project_artifact_row(
    row: Mapping[str, Any], *, tenant_id: str,
) -> tuple[ArtifactRowV1, dict[str, Any]]:
    """Validate one detached Artifact image and derive its physical projection.

    Args:
        row: Complete historical V1 fields, including the actual canonical body,
            address, byte count and reference projections. All fields are
            validated afresh; a previously constructed model grants no trust.
        tenant_id: Trusted tenant from the caller's verified workspace binding.

    Returns:
        The validated row and its normalized physical fields. Both are local
        values, with no retained cache, database read or borrowed authority.

    Raises:
        ValidationError: If fields, body shape, placement, address or size fail.
        QuoteMemError: If tenant or complete derived reference projections differ.

    The restore codec uses both results to avoid validating the same unchanged
    body again solely to prove round-trip physical parity. The Writer still
    owns current existence, reference targets, privacy and lifecycle checks.
    """
    _TENANT_ID.validate_python(tenant_id)
    checked = _checked_artifact(row)
    if checked.tenant_id != tenant_id:
        raise QuoteMemError("tenant_mismatch", "Normalization row belongs to another workspace")
    return checked, _physical_payload("artifact", checked.model_dump(mode="json"))


def project_checked_artifact_row(
    checked: ArtifactRowV1, body: StrictModel | None, *, tenant_id: str,
) -> dict[str, Any]:
    """Derive the physical projection of an Artifact row its caller has just validated.

    ``encode_normalized_batch`` validates every row of its batch afresh, an Artifact
    row without a derived address, and keeps the body that validation parsed. This
    projection proves that row's tenant and its reference projections from that body
    instead of validating the same row a second time; ``project_artifact_row``
    remains the projection of any other image.

    Args:
        checked: The row, as this call's validation returned it.
        body: The strict body that validation parsed, or None to parse it here.
        tenant_id: Trusted tenant from the caller's verified workspace binding.

    Returns:
        The row's normalized physical fields.

    Raises:
        QuoteMemError: If the tenant or the complete derived reference projections differ.
    """
    _TENANT_ID.validate_python(tenant_id)
    if checked.artifact_kind not in {"part", "manifest"}:
        if body is None:
            body = parse_artifact_body(checked.artifact_kind, checked.body)
        verify_artifact_refs(checked, body, address_verified=True)
    if checked.tenant_id != tenant_id:
        raise QuoteMemError("tenant_mismatch", "Normalization row belongs to another workspace")
    return _physical_payload("artifact", checked.model_dump(mode="json"))


def project_normalized_row(
    kind: str,
    row: StrictModel,
    *,
    tenant_id: str,
    parent_recall: StrictModel | None = None,
) -> dict[str, Any]:
    """Remove physical duplication only after proving exact existing-row parity.

    tenant_id comes from the already verified workspace, not from row input.
    Mismatches fail without changing bytes. This function is not a mutation or
    migration capability; the normal Writer still owns references, epochs and
    transactions.
    """
    if kind == "artifact":
        return project_artifact_row(row.model_dump(mode="json"), tenant_id=tenant_id)[1]
    _TENANT_ID.validate_python(tenant_id)
    checked = _checked_row(kind, row)
    if getattr(checked, "tenant_id", tenant_id) != tenant_id:
        raise QuoteMemError("tenant_mismatch", "Normalization row belongs to another workspace")
    parent = parent_recall
    if kind == "recall" and getattr(checked, "record_class", None) == "interaction":
        parent = None if parent_recall is None else _checked_row("recall", parent_recall)
    return project_checked_normalized_row(
        kind, checked, tenant_id=tenant_id, parent_recall=parent,
    )


def project_checked_normalized_row(
    kind: str,
    checked: StrictModel,
    *,
    tenant_id: str,
    parent_recall: StrictModel | None = None,
) -> dict[str, Any]:
    """Project non-Artifact strict clones this call's batch validation already proved.

    Args:
        kind: The existing non-Artifact entity kind validated with ``checked``.
        checked: The detached strict clone returned by the caller's validation.
        tenant_id: Trusted tenant from the verified workspace binding.
        parent_recall: The validated actual parent delivery, when required.

    Returns:
        Detached physical fields with the same tenant and parent parity checks
        as ``project_normalized_row``. No admission or reference authority is
        granted by the projection.

    Raises:
        QuoteMemError: If the tenant, parent projection or physical shape differs.
    """
    _TENANT_ID.validate_python(tenant_id)
    raw = checked.model_dump(mode="json")
    if raw.get("tenant_id", tenant_id) != tenant_id:
        raise QuoteMemError("tenant_mismatch", "Normalization row belongs to another workspace")
    if kind == "recall" and raw["record_class"] == "interaction":
        if parent_recall is None:
            raise QuoteMemError(
                "authority_unavailable", "Recall interaction requires its parent delivery"
            )
        parent = parent_recall.model_dump(mode="json")
        if (
            parent["object_id"] != raw["parent_recall_id"]
            or parent["tenant_id"] != tenant_id
            or parent["record_class"] not in {"minimal_ref_delivery", "replayable_delivery"}
            or any(raw[name] != parent[name] for name in RECALL_PARENT_FIELDS)
        ):
            raise QuoteMemError("parity_mismatch",
                                "Recall interaction differs from its parent delivery")
    return _physical_payload(kind, raw)


def project_normalized_capacity_row(
    kind: str, row: StrictModel, *, tenant_id: str,
) -> dict[str, Any]:
    """Shape a strict prospective V1 row for the Writer's non-writing HB planner.

    Args:
        kind: Existing QuoteMem entity kind.
        row: Complete sizing post-image, including values whose future authority
            is not persisted yet.
        tenant_id: Verified workspace tenant.

    Returns:
        The same physical shape as a validated write. This does not establish
        reference existence, projection parity or permission to persist the row.
        Only the capacity planner may use this authority-free projection.

    Raises:
        QuoteMemError: For another tenant, unsupported kind or changed physical shape.
        ValueError: For a malformed strict row or generated physical value.
    """
    _TENANT_ID.validate_python(tenant_id)
    checked = _checked_row(kind, row)
    raw = checked.model_dump(mode="json")
    if raw.get("tenant_id", tenant_id) != tenant_id:
        raise QuoteMemError("tenant_mismatch", "Capacity row belongs to another workspace")
    return _physical_payload(kind, raw)


def _physical_payload(kind: str, raw: dict[str, Any]) -> dict[str, Any]:
    """Apply the sole physical shape after the caller's appropriate validation."""
    if kind == "evidence":
        raw.setdefault("source_participant", None)
    elif kind == "recall" and raw["record_class"] == "interaction":
        raw.update(dict.fromkeys(RECALL_PARENT_FIELDS))
    spec = NORMALIZED_SCHEMA_SPECS[kind]
    result = {
        name: value for name, value in raw.items() if name not in REMOVED_FIELDS.get(kind, ())
    }
    if set(result) != {"object_id", *spec.fields}:
        raise QuoteMemError("schema_mismatch",
                            "Normalized row differs from the exact physical schema")
    return result

