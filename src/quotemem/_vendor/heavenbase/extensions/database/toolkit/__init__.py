"""Database extension MCP toolkit family."""

from __future__ import annotations

from heavenbase._lazy import _attach_lazy_exports

__all__ = [
    "database_allow_write",
    "database_result_limit",
    "database_tool_functions",
    "ensure_database_extension",
    "ingest_database",
    "list_sources",
    "run_sql",
    "run_snippet",
    "save_snippet",
    "suggest_sql",
]

_LAZY_EXPORTS = {
    "database_allow_write": (".policy", "database_allow_write"),
    "database_result_limit": (".policy", "database_result_limit"),
    "database_tool_functions": (".factory", "database_tool_functions"),
    "ensure_database_extension": (".enable", "ensure_database_extension"),
    "ingest_database": (".tools.ingest_database", "ingest_database"),
    "list_sources": (".tools.list_sources", "list_sources"),
    "run_sql": (".tools.run_sql", "run_sql"),
    "run_snippet": (".tools.run_snippet", "run_snippet"),
    "save_snippet": (".tools.save_snippet", "save_snippet"),
    "suggest_sql": (".tools.suggest_sql", "suggest_sql"),
}

__getattr__, __dir__ = _attach_lazy_exports(__name__, globals(), _LAZY_EXPORTS, __all__)
