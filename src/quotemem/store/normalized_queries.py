"""Closed bounded SQL projections for the normalized physical schema.

The existing IndexedReader owns execution, locks, routing and HB decoding. These
functions neither open a connection nor grant read or migration authority. SQL
labels below are private transport coordinates, never persisted fields. Selected
rows hydrate the existing logical DTOs without changing historical Artifact bytes.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from pydantic import TypeAdapter, ValidationError, model_validator
from sqlalchemy import LargeBinary, Select, Table, cast, func, select

from quotemem.core.artifacts import ArtifactKind
from quotemem.core.errors import QuoteMemError
from quotemem.core.identity import ULID, ArtifactId, ScopeId
from quotemem.core.semantic import ScopeKind, SecurityLevel, SystemTimestamp
from quotemem.store.domain import TenantId
from quotemem.store.metadata import StrictModel
from quotemem.store.normalization import (
    NORMALIZED_SCHEMA_SPECS,
)

_ARTIFACT_ID: TypeAdapter[str] = TypeAdapter(ArtifactId)


def _fields(kind: str) -> set[str]:
    return {"object_id", *NORMALIZED_SCHEMA_SPECS[kind].fields}


def _table(table: Table, kind: str) -> None:
    if set(table.c.keys()) != _fields(kind):
        raise QuoteMemError("schema_mismatch",
                            "Normalized query requires the exact physical columns")


class ArtifactHeader(StrictModel):
    """Content-free physical ownership and measured UTF-8 body size; no address proof."""

    object_id: ArtifactId
    artifact_kind: ArtifactKind
    schema_version: Literal["1"]
    tenant_id: TenantId
    scope_kind: ScopeKind | None
    scope_id: ScopeId | None
    security_level: SecurityLevel
    created_by_run_id: ULID
    created_at: SystemTimestamp
    body_bytes: int

    @model_validator(mode="after")
    def check_header(self) -> ArtifactHeader:
        """Reject incomplete scope and invalid measured sizes."""
        if (self.scope_kind is None) != (self.scope_id is None) or self.body_bytes < 0:
            raise ValueError("Artifact header scope or byte count is invalid")
        return self


def artifact_header_statement(artifact: Table, object_id: str) -> Select[Any]:
    """Select one Artifact header and its actual stored byte length, excluding body.

    Args:
        artifact: Registered normalized Artifact table on the SQLite main owner.
        object_id: Exact canonical Artifact identity.

    Returns:
        A LIMIT 1 statement measuring length(CAST(body AS BLOB)) and DB encoding.

    Raises:
        QuoteMemError: The table shape or identity is invalid.
    """
    _table(artifact, "artifact")
    try:
        identifier = _ARTIFACT_ID.validate_python(object_id, strict=True)
    except ValidationError as error:
        raise QuoteMemError("invalid_request",
                            "Artifact header requires an exact identity") from error
    encoding = func.pragma_encoding().table_valued("encoding")
    columns = set(ArtifactHeader.model_fields) - {"tenant_id", "body_bytes"}
    return (
        select(
            *(artifact.c[name] for name in sorted(columns)),
            func.length(cast(artifact.c.body, LargeBinary)).label("body_bytes"),
            select(encoding.c.encoding).scalar_subquery().label("_encoding"),
        )
        .where(artifact.c.object_id == identifier)
        .limit(1)
    )


def hydrate_artifact_header(
    selected: Mapping[str, Any],
    *,
    tenant_id: str,
    max_body_bytes: int,
) -> ArtifactHeader:
    """Reject an oversized or non-UTF-8 stored body before any full-body hydration.

    Args:
        selected: One fixed Artifact header returned by artifact_header_statement.
        tenant_id: Already verified workspace tenant identity.
        max_body_bytes: Caller-owned inclusive UTF-8 content budget.

    Returns:
        Validated content-free ownership metadata; this grants no body authority.

    Raises:
        QuoteMemError: Encoding, shape, scope or caller byte budget does not match.
    """
    if type(max_body_bytes) is not int or max_body_bytes < 0:
        raise QuoteMemError("invalid_request", "Artifact byte budget must be a nonnegative integer")
    if selected.get("_encoding") != "UTF-8":
        raise QuoteMemError("schema_mismatch",
                            "Artifact byte measurement requires actual UTF-8 storage")
    try:
        header = ArtifactHeader.model_validate(
            {
                **{key: value for key, value in selected.items() if key != "_encoding"},
                "tenant_id": tenant_id,
            }
        )
    except ValidationError as error:
        raise QuoteMemError("schema_invalid", "Normalized Artifact header is malformed") from error
    if header.body_bytes > max_body_bytes:
        raise QuoteMemError("budget_exceeded",
                            "Stored Artifact body exceeds the caller byte budget")
    return header
