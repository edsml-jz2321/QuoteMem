"""HeavenBase database facade, SQL policy, and physical dialect contracts."""

from __future__ import annotations

from heavenbase._lazy import _attach_lazy_exports

__all__ = [
    "Database",
    "DatabaseEngineRegistry",
    "DatabaseError",
    "DBEngine",
    "DBSpec",
    "DB_ENGINE",
    "SA_TO_SQLGLOT",
    "SQLGLOT_TO_SA",
    "SQLGlotProcessor",
    "SQLProcessor",
    "SQLResponse",
    "SchemaIndex",
    "table_display",
    "compare_sqls",
    "create_database",
    "create_database_engine",
    "create_sql_processor",
    "drop_database",
    "escape_sql_binds",
    "is_sql_readonly",
    "load_builtin_sql",
    "prettify_sql",
    "sa_dialect_to_sqlglot",
    "split_sqls",
    "sqlglot_dialect_to_sa",
    "strip_sql_comments",
    "transpile_sql",
    "validate_sql",
]

_LAZY_EXPORTS = {
    "Database": (".database", "Database"),
    "DatabaseEngineRegistry": (".registry", "DatabaseEngineRegistry"),
    "DatabaseError": (".database", "DatabaseError"),
    "DBEngine": (".config", "DBEngine"),
    "DBSpec": (".config", "DBSpec"),
    "DB_ENGINE": (".config", "DB_ENGINE"),
    "SA_TO_SQLGLOT": (".sql_text", "SA_TO_SQLGLOT"),
    "SQLGLOT_TO_SA": (".sql_text", "SQLGLOT_TO_SA"),
    "SQLGlotProcessor": (".processor", "SQLGlotProcessor"),
    "SQLProcessor": (".processor", "SQLProcessor"),
    "SQLResponse": (".response", "SQLResponse"),
    "SchemaIndex": (".schema_index", "SchemaIndex"),
    "table_display": (".display", "table_display"),
    "compare_sqls": (".sql_text", "compare_sqls"),
    "create_database": (".ddl", "create_database"),
    "create_database_engine": (".engine", "create_database_engine"),
    "create_sql_processor": (".processor", "create_sql_processor"),
    "drop_database": (".ddl", "drop_database"),
    "escape_sql_binds": (".sql_text", "escape_sql_binds"),
    "is_sql_readonly": (".readonly", "is_sql_readonly"),
    "load_builtin_sql": (".resources", "load_builtin_sql"),
    "prettify_sql": (".sql_text", "prettify_sql"),
    "sa_dialect_to_sqlglot": (".sql_text", "sa_dialect_to_sqlglot"),
    "split_sqls": (".sql_text", "split_sqls"),
    "sqlglot_dialect_to_sa": (".sql_text", "sqlglot_dialect_to_sa"),
    "strip_sql_comments": (".sql_text", "strip_sql_comments"),
    "transpile_sql": (".sql_text", "transpile_sql"),
    "validate_sql": (".sql_text", "validate_sql"),
}

__getattr__, __dir__ = _attach_lazy_exports(__name__, globals(), _LAZY_EXPORTS, __all__)
