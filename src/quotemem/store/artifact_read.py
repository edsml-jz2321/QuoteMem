"""Shared immutable Artifact reads; callers retain storage and content authority.

This module grants no write, execution, retention or hydration lifetime. Both the
sole Writer and the existing-job metadata reader supply their actual row reader.
"""

import base64
import hashlib
import json
from collections.abc import Callable, Sequence
from typing import Any

from pydantic import TypeAdapter

from quotemem.core.artifacts import ArtifactRowV1, parse_artifact_body
from quotemem.core.errors import QuoteMemError
from quotemem.core.identity import artifact_id, canonical_json
from quotemem.core.references import ReferenceProjectionV1
from quotemem.store.metadata import StrictModel
from quotemem.store.reference_context import contextual_references

ReadRow = Callable[[str, str], dict[str, Any] | None]
_REF_ADAPTER: TypeAdapter[ReferenceProjectionV1] = TypeAdapter(ReferenceProjectionV1)


def read_logical_artifact(
    read_row: ReadRow,
    reference: str,
    expected_kind: str,
    visiting: frozenset[str] = frozenset(),
) -> StrictModel:
    """Read existing strict logical bytes; the caller owns storage and content lifetime."""
    if reference in visiting:
        raise QuoteMemError("reference_cycle", "Artifact manifest contains a cycle")
    row = read_row("artifact", reference)
    if row is None:
        raise QuoteMemError("reference_missing", "Logical Artifact does not exist")
    # Validated without a derived address, the row proves its address and byte
    # count (``verify_artifact_identity``'s checks) and parses its body once.
    parsed: list[StrictModel] = []
    typed = ArtifactRowV1.model_validate(row, context={"parsed_body": parsed})
    stored = parsed[0] if len(parsed) == 1 else None
    if typed.artifact_kind == expected_kind:
        body = stored if stored is not None else parse_artifact_body(
            expected_kind, json.loads(typed.body), typed.schema_version)
        if expected_kind != "part":
            verify_artifact_refs(typed, body, address_verified=True)
        return body
    if typed.artifact_kind != "manifest":
        raise QuoteMemError("reference_kind_mismatch", "Artifact has a different logical kind")
    if stored is None:
        stored = parse_artifact_body("manifest", json.loads(typed.body))
    manifest = stored.model_dump(mode="json")
    if manifest["logical_kind"] != expected_kind:
        raise QuoteMemError("reference_kind_mismatch", "Manifest has a different logical kind")
    visiting = visiting | {reference}
    payload = bytearray()
    for part_ref in manifest["parts"]:
        if part_ref in visiting:
            raise QuoteMemError("reference_cycle", "Manifest includes its own physical identity")
        part_row = read_row("artifact", part_ref)
        if part_row is None:
            raise QuoteMemError("reference_missing", "Manifest part is missing")
        part = ArtifactRowV1.model_validate(part_row)
        verify_artifact_identity(part)
        if part.artifact_kind != "part":
            raise QuoteMemError("reference_kind_mismatch", "Manifest child is not a physical part")
        for field in (
            "tenant_id",
            "scope_kind",
            "scope_id",
            "security_level",
            "created_by_run_id",
            "body_refs",
        ):
            if getattr(part, field) != getattr(typed, field):
                raise QuoteMemError(
                    "parity_mismatch", "Manifest part envelope differs from its owner"
                )
        if part.artifact_refs:
            raise QuoteMemError("parity_mismatch", "A physical part cannot own artifact references")
        payload.extend(base64.b64decode(json.loads(part.body)["data"], validate=True))
    if (
        len(payload) != manifest["total_bytes"]
        or hashlib.sha256(payload).hexdigest() != (manifest["body_digest"])
    ):
        raise QuoteMemError("parity_mismatch", "Logical Artifact byte length or digest differs")
    try:
        logical = parse_artifact_body(
            expected_kind, json.loads(payload), manifest["schema_version"]
        )
    except (ValueError, UnicodeDecodeError) as error:
        raise QuoteMemError("schema_invalid", "Reassembled logical payload is invalid") from error
    if canonical_json(logical).encode("utf-8") != bytes(payload):
        raise QuoteMemError("parity_mismatch", "Logical payload is not canonical JSON")
    verify_artifact_refs(typed, logical, manifest["parts"], address_verified=True)
    return logical


def artifact_envelope(row: ArtifactRowV1) -> dict[str, Any]:
    """Return the canonical identity envelope, excluding the outer creation clock."""
    fields = {
        "artifact_kind",
        "schema_version",
        "tenant_id",
        "scope_kind",
        "scope_id",
        "security_level",
        "body_refs",
        "created_by_run_id",
        "body",
    }
    return row.model_dump(mode="json", include=fields)


def verify_artifact_identity(row: ArtifactRowV1) -> None:
    """Verify the actual address and physical byte count without reading targets."""
    if row.object_id != artifact_id(artifact_envelope(row)):
        raise QuoteMemError(
            "artifact_digest_collision", "Artifact envelope does not match its address"
        )
    if row.size_bytes != len(row.body.encode("utf-8")):
        raise QuoteMemError("parity_mismatch", "Artifact byte count differs from its body")


_PROVED_PROJECTIONS: dict[tuple[str, tuple[str, ...], tuple[str, ...]], None] = {}
"""Projections already proved for a verified address, its Artifact references and parts."""

_PROVED_LIMIT = 16384
"""Proved projections remembered before the memo starts over."""


def verify_artifact_refs(
    row: ArtifactRowV1,
    body: StrictModel,
    parts: Sequence[str] = (),
    *,
    address_verified: bool = False,
) -> None:
    """Require exact derived projections from the complete strict logical body.

    With ``address_verified`` the caller has proved ``row``'s address against its
    envelope, so the address fixes the kind, the body bytes and ``body_refs``; the
    only other inputs are ``artifact_refs`` and ``parts``. A proof of that same
    address, references and parts is then remembered for the process and not
    repeated: the projection is a pure function of those inputs.
    """
    key = (row.object_id, tuple(row.artifact_refs), tuple(parts)) if address_verified else None
    if key is not None and key in _PROVED_PROJECTIONS:
        return
    references = contextual_references(body)
    body_refs = [
        _REF_ADAPTER.validate_python({"kind": kind, "object_id": identifier})
        for kind, identifier in sorted({
            (ref.spec.target_kind, ref.target_id)
            for ref in references if ref.spec.target_kind != "artifact"
        })
    ]
    artifact_refs = sorted(
        {ref.target_id for ref in references if ref.spec.target_kind == "artifact"} | set(parts)
    )
    if row.body_refs != body_refs or list(row.artifact_refs) != artifact_refs:
        raise QuoteMemError(
            "parity_mismatch", "Artifact reference projection differs from strict content"
        )
    if key is not None:
        if len(_PROVED_PROJECTIONS) >= _PROVED_LIMIT:
            _PROVED_PROJECTIONS.clear()
        _PROVED_PROJECTIONS[key] = None
