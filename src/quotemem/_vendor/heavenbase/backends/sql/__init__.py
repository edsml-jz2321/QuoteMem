"""SQL-family backend providers."""

from __future__ import annotations

from heavenbase._lazy import _attach_lazy_exports

_PROVIDER_EXPORTS = {
    "ClickHouseBackend": (".clickhouse", "ClickHouseBackend"),
    "DoltBackend": (".dolt", "DoltBackend"),
    "DuckDBBackend": (".duckdb", "DuckDBBackend"),
    "MsSQLBackend": (".mssql", "MsSQLBackend"),
    "MySQLBackend": (".mysql", "MySQLBackend"),
    "OceanBaseBackend": (".oceanbase", "OceanBaseBackend"),
    "OracleBackend": (".oracle", "OracleBackend"),
    "PostgresBackend": (".postgres", "PostgresBackend"),
    "SeekDBBackend": (".seekdb", "SeekDBBackend"),
    "SQLBackend": (".base", "SQLBackend"),
    "SQLiteBackend": (".sqlite", "SQLiteBackend"),
    "StarRocksBackend": (".starrocks", "StarRocksBackend"),
    "SupabaseBackend": (".supabase", "SupabaseBackend"),
    "TrinoBackend": (".trino", "TrinoBackend"),
}
_DIALECT_EXPORTS = {
    "SQL_JSON_AGG_STYLE_ARRAYAGG": (".dialect", "SQL_JSON_AGG_STYLE_ARRAYAGG"),
    "SQL_JSON_AGG_STYLE_GROUP_ARRAY": (".dialect", "SQL_JSON_AGG_STYLE_GROUP_ARRAY"),
    "SQL_JSON_AGG_STYLE_POSTGRES_ORDERED": (".dialect", "SQL_JSON_AGG_STYLE_POSTGRES_ORDERED"),
    "SQL_JSON_OBJECT_STYLE_BUILD_OBJECT": (".dialect", "SQL_JSON_OBJECT_STYLE_BUILD_OBJECT"),
    "SQL_JSON_OBJECT_STYLE_OBJECT": (".dialect", "SQL_JSON_OBJECT_STYLE_OBJECT"),
    "SQL_REGEX_STYLE_DUCKDB_TILDE_SEARCH": (".dialect", "SQL_REGEX_STYLE_DUCKDB_TILDE_SEARCH"),
    "SQL_REGEX_STYLE_POSTGRES_TILDE": (".dialect", "SQL_REGEX_STYLE_POSTGRES_TILDE"),
    "SQL_REGEX_STYLE_REGEXP_LIKE": (".dialect", "SQL_REGEX_STYLE_REGEXP_LIKE"),
    "SQL_REGEX_STYLE_REGEXP_LIKE_CASE_SENSITIVE": (".dialect", "SQL_REGEX_STYLE_REGEXP_LIKE_CASE_SENSITIVE"),
    "SQL_WILDCARD_STYLE_GLOB": (".dialect", "SQL_WILDCARD_STYLE_GLOB"),
    "SQL_WILDCARD_STYLE_LIKE": (".dialect", "SQL_WILDCARD_STYLE_LIKE"),
    "SQLDialectSpec": (".dialect", "SQLDialectSpec"),
    "register_sql_dialect": (".capabilities", "register_sql_dialect"),
    "registered_sql_dialects": (".capabilities", "registered_sql_dialects"),
    "sql_dialect_spec": (".capabilities", "sql_dialect_spec"),
}
_LAZY_EXPORTS = {**_PROVIDER_EXPORTS, **_DIALECT_EXPORTS}

__all__ = [*_LAZY_EXPORTS]


__getattr__, __dir__ = _attach_lazy_exports(__name__, globals(), _LAZY_EXPORTS, __all__)
