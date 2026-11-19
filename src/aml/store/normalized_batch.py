"""Pure normalized payload preparation for one existing Writer-owned batch.

The Writer supplies already selected post-images, trusted workspace identity,
and exact lookup results. This module performs no reads, writes, admission, or
transaction planning; its dependency closure covers physical projections only.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, cast

from pydantic import TypeAdapter

from aml.core.artifacts import ArtifactRowV1
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
    project_checked_artifact_row,
    project_checked_normalized_row,
)

RowKey = tuple[str, str]
_TENANT: TypeAdapter[str] = TypeAdapter(TenantId)


class RowWriteLike(Protocol):
    """Structural input for existing RowWrite, without importing its Writer owner."""

    @property
    def kind(self) -> str:
        """Return the existing entity kind."""
        ...

    @property
    def row(self) -> StrictModel:
        """Return the complete strict V1 post-image."""
        ...


@dataclass(frozen=True)
class _CheckedRow:
    key: RowKey
    row: StrictModel
    dependency: RowKey | None
    body: StrictModel | None = None
    """An Artifact row's strict body, parsed by the row's own validation."""


def _checked_row(kind: str, row: object, parsed: list[StrictModel] | None = None) -> StrictModel:
    if not isinstance(row, StrictModel):
        raise AMLError("schema_invalid", "Batch rows must be complete strict models")
    # JSON field serializers can turn a forged bool into an integer before
    # validation. Preserve Python field types until the strict clone succeeds.
    raw = row.model_dump(mode="python", warnings=False)
    if kind == "artifact":
        # Validated without a derived address: the clone proves its address and
        # byte count, and the body it parses is kept for its projection.
        checked = ArtifactRowV1.model_validate(
            raw, context=None if parsed is None else {"parsed_body": parsed})
    elif kind == "control":
        checked = ControlRowV1.model_validate(raw)
    else:
        adapter = DOMAIN_ADAPTERS.get(kind)
        if adapter is None:
            raise AMLError("unsupported_variant", "Unknown normalized batch entity")
        checked = cast(StrictModel, adapter.validate_python(raw))
    if canonical_json(checked) != canonical_json(raw):
        raise AMLError("parity_mismatch", "Batch validation changed the supplied V1 image")
    return checked


def _checked_batch(rows: Sequence[RowWriteLike], max_rows: int) -> tuple[_CheckedRow, ...]:
    if type(max_rows) is not int or max_rows < 1:
        raise AMLError("invalid_request", "Normalized batch requires a positive row bound")
    if not rows:
        raise AMLError("invalid_request", "Normalized batch cannot be empty")
    if len(rows) > max_rows:
        raise AMLError("budget_exceeded", "Normalized input batch exceeds its row bound")
    checked: list[_CheckedRow] = []
    identities: set[RowKey] = set()
    for entry in rows:
        if type(entry.kind) is not str:
            raise AMLError("schema_invalid", "Normalized entity kind must be an exact string")
        parsed: list[StrictModel] = []
        row = _checked_row(entry.kind, entry.row, parsed)
        key = entry.kind, cast(str, row.model_dump()["object_id"])
        if key in identities:
            raise AMLError("invalid_request", "Normalized batch contains a duplicate row identity")
        identities.add(key)
        dependency: RowKey | None = None
        if isinstance(row, MemoryInteractionRecallRowV1 | EvidenceInteractionRecallRowV1):
            dependency = "recall", row.parent_recall_id
        checked.append(_CheckedRow(key, row, dependency, parsed[0] if len(parsed) == 1 else None))
    return tuple(checked)


def _external_keys(rows: tuple[_CheckedRow, ...]) -> frozenset[RowKey]:
    batch_keys = {entry.key for entry in rows}
    return frozenset(entry.dependency for entry in rows
                     if entry.dependency is not None and entry.dependency not in batch_keys)


@dataclass(frozen=True)
class CheckedBatch:
    """One batch's strict clones, validated once for its lookup keys and its encoding.

    Attributes:
        source: The very sequence that was validated; ``encode_normalized_batch``
            reuses the clones only for this same sequence object.
        rows: Each row's strict clone, identity and interaction dependency.
        max_rows: The bound the batch was validated against.
    """

    source: Sequence[RowWriteLike]
    rows: tuple[_CheckedRow, ...]
    max_rows: int

    @property
    def authority_keys(self) -> frozenset[RowKey]:
        """The external dependencies (``normalized_batch_authority_keys``)."""
        return _external_keys(self.rows)


def check_normalized_batch(rows: Sequence[RowWriteLike], *, max_rows: int) -> CheckedBatch:
    """Validate one batch once, for ``authority_keys`` and then ``encode_normalized_batch``.

    Args:
        rows: Complete strict V1 post-images in the intended write order; the
            caller must not change them before encoding.
        max_rows: Existing caller-owned maximum input rows for this batch.

    Returns:
        The validated batch.

    Raises:
        AMLError: For malformed, duplicate, empty or oversized input, exactly as
            ``normalized_batch_authority_keys``.
    """
    try:
        return CheckedBatch(rows, _checked_batch(rows, max_rows), max_rows)
    except AMLError:
        raise
    except (TypeError, ValueError, AttributeError) as error:
        raise AMLError("schema_invalid", "Normalized batch input is malformed") from error


def normalized_batch_authority_keys(
    rows: Sequence[RowWriteLike], *, max_rows: int,
) -> frozenset[RowKey]:
    """Name exactly the projection dependencies the Writer must supply from lookup.

    Args:
        rows: Complete strict V1 post-images in the intended write order.
        max_rows: Existing caller-owned maximum input rows for this batch.

    Returns:
        External keys absent from the complete batch. Same-batch delivery rows
        remain the selected projection authority.

    Raises:
        AMLError: For malformed, duplicate, empty or oversized input. This is
            neither a read nor closure validation for ordinary foreign references.
    """
    return check_normalized_batch(rows, max_rows=max_rows).authority_keys


def encode_normalized_batch(
    rows: Sequence[RowWriteLike],
    *,
    tenant_id: str,
    authorities: Mapping[RowKey, StrictModel | None],
    max_rows: int,
    checked: CheckedBatch | None = None,
) -> tuple[dict[str, Any], ...]:
    """Encode a complete bounded batch for the existing one-main-transaction owner.

    Args:
        rows: Strict V1 post-images whose canonical input values remain unchanged.
        tenant_id: Trusted tenant from the verified workspace binding.
        authorities: Actual lookup results for exactly normalized_batch_authority_keys.
            Keys bind both entity kind and object identity. None means missing
            authority and fails closed. Extra or shadowing keys are rejected.
        max_rows: Caller-owned input-row bound, normally the Writer's MAX_TX_ROWS.
        checked: This same ``rows`` sequence's validation by
            ``check_normalized_batch`` under the same bound, reused instead of
            validating every row again; any other value validates afresh.

    Returns:
        Detached physical dictionaries in original input order. No partial result
        escapes if a dependency or projection fails. The caller still prepares
        the HB plan, checks its expanded RowOp budget and commits one transaction.

    Raises:
        AMLError: For invalid closure, scope/tenant/projection mismatch, wrong
            dependency identity, or malformed/duplicate/oversized input.

    No current Evidence is queried.
    Selecting a dependency in this pure batch does not satisfy a RefSpec's
    pre-state existence, authorize an Artifact prewrite, or admit any mutation.
    """
    try:
        _TENANT.validate_python(tenant_id, strict=True)
        batch = (checked.rows if checked is not None and checked.source is rows
                 and checked.max_rows == max_rows else _checked_batch(rows, max_rows))
        required = _external_keys(batch)
        if frozenset(authorities) != required:
            raise AMLError("binding_mismatch", "Batch lookup keys differ from exact dependencies")
        resolved = {entry.key: entry.row for entry in batch}
        for key in sorted(required):
            supplied = authorities[key]
            if supplied is None:
                raise AMLError("reference_missing", "Normalized batch dependency does not exist")
            authority = _checked_row(key[0], supplied)
            if authority.model_dump()["object_id"] != key[1]:
                raise AMLError("binding_mismatch", "Batch lookup returned a different identity")
            resolved[key] = authority
        result: list[dict[str, Any]] = []
        for entry in batch:
            parent: MinimalRecallRowV1 | ReplayableRecallRowV1 | None = None
            if entry.dependency is not None:
                authority = resolved[entry.dependency]
                if not isinstance(authority, MinimalRecallRowV1 | ReplayableRecallRowV1):
                    raise AMLError(
                        "authority_unavailable", "Interaction dependency is not a delivery",
                    )
                assert isinstance(
                    entry.row, MemoryInteractionRecallRowV1 | EvidenceInteractionRecallRowV1,
                )
                parent = authority
                validate_interaction_parent(entry.row, parent)
            if isinstance(entry.row, ArtifactRowV1):
                # This call's own validation of the row is its fresh proof; the
                # projection proves the references of the body it parsed.
                result.append(project_checked_artifact_row(
                    entry.row, entry.body, tenant_id=tenant_id,
                ))
                continue
            result.append(project_checked_normalized_row(
                entry.key[0], entry.row, tenant_id=tenant_id, parent_recall=parent,
            ))
        return tuple(result)
    except AMLError:
        raise
    except (TypeError, ValueError, AttributeError) as error:
        raise AMLError(
            "schema_invalid", "Normalized batch or lookup authority is malformed",
        ) from error
