"""Catalog row builders for workspace writes."""

from __future__ import annotations

__all__ = [
    "CATALOG_AUTO_FIELDS",
    "CATALOG_DEFAULTS",
    "CATALOG_LIFECYCLE_FIELDS",
    "catalog_audit_timestamps",
    "catalog_materialize_timing",
    "catalog_row_id",
    "catalog_write_timestamps",
    "build_catalog_row",
]

from heavenbase.utils import Any, Dict, Mapping, copy_data
from heavenbase.utils.ids import hash_id

from ...modules import ModuleResolver
from ...types.duration import Duration
from ...types.timestamp import TIMESTAMP_DEFAULT_MAX_MS, Timestamp
from .entity import Catalog


def _catalog_policy_dtypes(*, resolver: ModuleResolver | None = None) -> tuple[Timestamp, Duration, Duration]:
    """Return Catalog policy types from the selected module authority."""
    schema = Catalog.schema()
    return (
        schema.get("expires_at").dtype,
        schema.get("expires_after").dtype,
        schema.get("stale_after").dtype,
    )


def _catalog_policy_value(
    field_name: str,
    row: Mapping[str, Any],
    *,
    default: Any,
    resolver: ModuleResolver | None = None,
) -> int:
    expires_at_dtype, expires_after_dtype, stale_after_dtype = _catalog_policy_dtypes(resolver=resolver)
    dtype = {"expires_at": expires_at_dtype, "expires_after": expires_after_dtype, "stale_after": stale_after_dtype}[field_name]
    raw = row.get(field_name, default)
    if raw is None:
        raw = default
    return dtype.validate(raw)


CATALOG_AUTO_FIELDS = {
    "object_id",
    "target_id",
    "target_entity",
    "ws",
    "created_at",
    "updated_at",
    "expires_compute",
    "stale_compute",
}
# The stored fields a Catalog rewrite keeps in every mode; ``updated_at`` is always the write time.
CATALOG_LIFECYCLE_FIELDS = ("created_at", "active", "visible")
CATALOG_DEFAULTS = {
    # Identity (object_id, target_id, target_entity are auto-generated)
    "name": "",
    "version": "",
    "type": "",
    # Payload (long content assembled into LLM prompts)
    "summary": "",
    "instructions": "",
    "examples": [],
    "artifacts": [],
    "confidence": 1.0,
    # Searchable (retrieval index; need not enter the payload)
    "desc": "",
    "keywords": [],
    "tags": [],
    "priority": 0.0,
    # Governance (ws, created_at, updated_at, expires_compute, stale_compute are auto-generated)
    "active": True,
    "visible": True,
    "expires_at": TIMESTAMP_DEFAULT_MAX_MS,
    "expires_after": 0,
    "stale_after": 0,
    "creator": "",
    "owner": "",
    "auth": {},
    "metadata": {},
    "note": "",
    # Lineage
    "source": "",
    "parent_id": "",
    "history": [],
    "transforms": [],
    # Usage
    "statistics": {},
    "feedback": [],
    "annotations": [],
    # Reference
    "related": [],
    "automations": [],
}


def catalog_row_id(
    ws_id: str,
    entity_type: str,
    entity_id: str,
    max_chars: int,
    *,
    hash_length: int | None = None,
) -> str:
    """Return a stable primary key for a Catalog row.

    Args:
        ws_id (str): Owning workspace identifier.
        entity_type (str): Catalog entity type token.
        entity_id (str): Target entity ``object_id``.
        max_chars (int): Maximum allowed primary-key length.
        hash_length (int | None): Optional hash length when truncation is
            required. Defaults to ``None``.

    Returns:
        str: Stable Catalog row primary key, hashed when necessary.
    """
    raw_id = f"catalog:{entity_type}:{entity_id}"
    if len(raw_id) <= max_chars:
        return raw_id
    prefix = "catalog:"
    hashed = hash_id("catalog", ws_id, entity_type, entity_id, length=hash_length)
    if len(prefix) + len(hashed) <= max_chars:
        return f"{prefix}{hashed}"
    return hashed


def _catalog_bool(existing: Dict[str, Any], key: str, *, fallback: bool) -> bool:
    if key not in existing or existing[key] is None:
        return fallback
    return bool(existing[key])


def catalog_write_timestamps(existing: Dict[str, Any] | None, now: int) -> Dict[str, Any]:
    """Return built-in Catalog lifecycle fields for an upsert/write path.

    ``CATALOG_LIFECYCLE_FIELDS`` carry over from ``existing`` when present;
    ``updated_at`` is ``now``.

    Args:
        existing (Dict[str, Any] | None): Existing Catalog row fields, if any.
        now (int): Current UTC epoch milliseconds used for ``updated_at``.

    Returns:
        Dict[str, Any]: Built-in lifecycle fields for a write path.
    """
    existing = existing or {}
    created_at = existing.get("created_at")
    return {
        "created_at": now if created_at is None else created_at,
        "updated_at": now,
        "active": _catalog_bool(existing, "active", fallback=True),
        "visible": _catalog_bool(existing, "visible", fallback=True),
    }


def catalog_audit_timestamps(existing: Dict[str, Any] | None, now: int) -> Dict[str, Any]:
    """Return built-in Catalog lifecycle fields for an audit-only expected row.

    Args:
        existing (Dict[str, Any] | None): Existing Catalog row fields, if any.
        now (int): Current UTC epoch milliseconds used when timestamps are absent.

    Returns:
        Dict[str, Any]: Built-in lifecycle fields for audit comparison.
    """
    timestamps = catalog_write_timestamps(existing, now)
    updated_at = (existing or {}).get("updated_at")
    timestamps["updated_at"] = timestamps["created_at"] if updated_at is None else updated_at
    return timestamps


def catalog_materialize_timing(
    row: Mapping[str, Any],
    *,
    resolver: ModuleResolver | None = None,
) -> Dict[str, int]:
    """Return materialized ``expires_compute`` and ``stale_compute`` from policy fields.

    Absolute expiry policy ``expires_at`` uses :attr:`~heavenbase.types.timestamp.Timestamp.default_max`
    as the disabled sentinel. Duration policies ``expires_after`` and
    ``stale_after`` use ``0`` as disabled. Materialized fields use
    ``Timestamp.default_max`` when the corresponding policy is off.

    Hard expiry (``expires_compute``) is the minimum of applicable constraints:

    - ``expires_at``: absolute ceiling when below ``Timestamp.default_max``.
    - ``expires_after + created_at``: creation TTL when ``expires_after`` is positive.

    Freshness (``stale_compute``) is ``stale_after + updated_at`` when
    ``stale_after`` is positive.

    Args:
        row: Catalog row mapping containing policy and lifecycle timestamp fields.
        resolver: Context-owned module resolver. The default Context is selected
            only for an unbound standalone call.

    Returns:
        Mapping with materialized ``expires_compute`` and ``stale_compute`` values.
    """
    open_deadline = Timestamp.default_max
    expires_at = _catalog_policy_value("expires_at", row, default=open_deadline, resolver=resolver)
    expires_after = _catalog_policy_value("expires_after", row, default=0, resolver=resolver)
    stale_after = _catalog_policy_value("stale_after", row, default=0, resolver=resolver)
    created_at = row.get("created_at")
    updated_at = row.get("updated_at")

    expire_candidates: list[int] = []
    if expires_at < open_deadline:
        expire_candidates.append(expires_at)
    if expires_after > 0 and created_at is not None:
        expire_candidates.append(expires_after + int(created_at))
    expires_compute = min(expire_candidates) if expire_candidates else open_deadline

    stale_compute = stale_after + int(updated_at) if stale_after > 0 and updated_at is not None else open_deadline
    return {"expires_compute": expires_compute, "stale_compute": stale_compute}


def build_catalog_row(
    ws_id: str,
    schema: Any,
    entity_id: str,
    max_chars: int,
    *,
    hash_length: int | None = None,
) -> dict:
    """Build an empty-default Catalog row for one concrete entity instance.

    Args:
        ws_id (str): Owning workspace identifier stored on the row.
        schema (Any): Entity schema providing ``entity_id``.
        entity_id (str): Target entity ``object_id``.
        max_chars (int): Maximum allowed Catalog primary-key length.
        hash_length (int | None): Optional hash length when truncation is
            required. Defaults to ``None``.

    Returns:
        dict: Empty-default Catalog row with identity and governance defaults.
            Container defaults are fresh objects; the source row is never read.
    """
    row = {
        "object_id": catalog_row_id(ws_id, schema.entity_id, entity_id, max_chars, hash_length=hash_length),
        "target_id": entity_id,
        "target_entity": schema.entity_id,
        "ws": ws_id,
    }
    row.update(copy_data(CATALOG_DEFAULTS))
    return row
