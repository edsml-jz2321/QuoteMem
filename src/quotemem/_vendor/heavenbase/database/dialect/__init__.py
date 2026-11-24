"""Database-owned physical dialect contracts and registry."""

from __future__ import annotations

from heavenbase._lazy import _attach_lazy_exports

__all__ = [
    "BootstrapDialectCatalog",
    "DatabaseDialectSpec",
    "DatabaseDialectCatalog",
    "MYSQL_TEXT_MIN_LEN",
    "ORACLE_VARCHAR_MAX",
    "ResolverDialectCatalog",
    "SQL_COMMON_RESERVED_WORDS",
    "SQL_TYPE_CLOB",
    "SQL_TYPE_FLOAT64_BLOB",
    "SQL_TYPE_FLOAT_ARRAY",
    "SQL_TYPE_JSON",
    "SQL_TYPE_JSON_TEXT",
    "SQL_TYPE_MYSQL_TEXT",
    "SQL_TYPE_POSTGRESQL_JSONB",
    "SQL_TYPE_STRING",
    "SQL_TYPE_TEXT",
    "database_dialects",
]

_LAZY_EXPORTS = {
    "BootstrapDialectCatalog": (".registry", "BootstrapDialectCatalog"),
    "DatabaseDialectSpec": (".spec", "DatabaseDialectSpec"),
    "DatabaseDialectCatalog": (".registry", "DatabaseDialectCatalog"),
    "MYSQL_TEXT_MIN_LEN": (".spec", "MYSQL_TEXT_MIN_LEN"),
    "ORACLE_VARCHAR_MAX": (".spec", "ORACLE_VARCHAR_MAX"),
    "ResolverDialectCatalog": (".registry", "ResolverDialectCatalog"),
    "SQL_COMMON_RESERVED_WORDS": (".spec", "SQL_COMMON_RESERVED_WORDS"),
    "SQL_TYPE_CLOB": (".spec", "SQL_TYPE_CLOB"),
    "SQL_TYPE_FLOAT64_BLOB": (".spec", "SQL_TYPE_FLOAT64_BLOB"),
    "SQL_TYPE_FLOAT_ARRAY": (".spec", "SQL_TYPE_FLOAT_ARRAY"),
    "SQL_TYPE_JSON": (".spec", "SQL_TYPE_JSON"),
    "SQL_TYPE_JSON_TEXT": (".spec", "SQL_TYPE_JSON_TEXT"),
    "SQL_TYPE_MYSQL_TEXT": (".spec", "SQL_TYPE_MYSQL_TEXT"),
    "SQL_TYPE_POSTGRESQL_JSONB": (".spec", "SQL_TYPE_POSTGRESQL_JSONB"),
    "SQL_TYPE_STRING": (".spec", "SQL_TYPE_STRING"),
    "SQL_TYPE_TEXT": (".spec", "SQL_TYPE_TEXT"),
    "database_dialects": (".registry", "database_dialects"),
}

__getattr__, __dir__ = _attach_lazy_exports(__name__, globals(), _LAZY_EXPORTS, __all__)
