"""SQL templates owned by the database domain."""

from __future__ import annotations

from heavenbase._lazy import _attach_lazy_exports

__all__ = [
    "SqlTemplate",
    "DB_ADD_COLUMN",
    "DB_DDL_AFTER_CREATE",
    "DB_DDL_BEFORE_DROP_FORCE",
    "DB_DDL_CREATE",
    "DB_DDL_DROP",
    "DB_DDL_EXISTS",
    "DB_CURRENT_DATABASE",
    "DB_LIST_DATABASES",
    "DB_REFLECT_COLUMNS",
    "DB_REFLECT_PRIMARY_KEYS",
    "DB_RENAME_TABLE",
    "SR_CREATE_DATABASE",
    "SR_CREATE_OLAP_TABLE",
    "SR_DROP_TABLE",
]

_LAZY_EXPORTS = {
    "SqlTemplate": (".base", "SqlTemplate"),
    "DB_ADD_COLUMN": (".db", "DB_ADD_COLUMN"),
    "DB_CURRENT_DATABASE": (".db", "DB_CURRENT_DATABASE"),
    "DB_DDL_AFTER_CREATE": (".db", "DB_DDL_AFTER_CREATE"),
    "DB_DDL_BEFORE_DROP_FORCE": (".db", "DB_DDL_BEFORE_DROP_FORCE"),
    "DB_DDL_CREATE": (".db", "DB_DDL_CREATE"),
    "DB_DDL_DROP": (".db", "DB_DDL_DROP"),
    "DB_DDL_EXISTS": (".db", "DB_DDL_EXISTS"),
    "DB_LIST_DATABASES": (".db", "DB_LIST_DATABASES"),
    "DB_REFLECT_COLUMNS": (".db", "DB_REFLECT_COLUMNS"),
    "DB_REFLECT_PRIMARY_KEYS": (".db", "DB_REFLECT_PRIMARY_KEYS"),
    "DB_RENAME_TABLE": (".db", "DB_RENAME_TABLE"),
    "SR_CREATE_DATABASE": (".starrocks", "SR_CREATE_DATABASE"),
    "SR_CREATE_OLAP_TABLE": (".starrocks", "SR_CREATE_OLAP_TABLE"),
    "SR_DROP_TABLE": (".starrocks", "SR_DROP_TABLE"),
}


__getattr__, __dir__ = _attach_lazy_exports(__name__, globals(), _LAZY_EXPORTS, __all__)
