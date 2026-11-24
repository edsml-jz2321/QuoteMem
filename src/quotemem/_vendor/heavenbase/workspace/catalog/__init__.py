"""Workspace Catalog row policy and lifecycle services."""

from __future__ import annotations

from heavenbase._lazy import _attach_lazy_exports

__all__ = [
    "Catalog",
    "CATALOG_AUTO_FIELDS",
    "CATALOG_DEFAULTS",
    "CATALOG_LIFECYCLE_FIELDS",
    "build_catalog_row",
    "catalog_audit_timestamps",
    "catalog_materialize_timing",
    "catalog_row_id",
    "catalog_write_timestamps",
]

_LAZY_EXPORTS = {
    "Catalog": (".entity", "Catalog"),
    "CATALOG_AUTO_FIELDS": (".rows", "CATALOG_AUTO_FIELDS"),
    "CATALOG_DEFAULTS": (".rows", "CATALOG_DEFAULTS"),
    "CATALOG_LIFECYCLE_FIELDS": (".rows", "CATALOG_LIFECYCLE_FIELDS"),
    "build_catalog_row": (".rows", "build_catalog_row"),
    "catalog_audit_timestamps": (".rows", "catalog_audit_timestamps"),
    "catalog_materialize_timing": (".rows", "catalog_materialize_timing"),
    "catalog_row_id": (".rows", "catalog_row_id"),
    "catalog_write_timestamps": (".rows", "catalog_write_timestamps"),
}


__getattr__, __dir__ = _attach_lazy_exports(__name__, globals(), _LAZY_EXPORTS, __all__)
