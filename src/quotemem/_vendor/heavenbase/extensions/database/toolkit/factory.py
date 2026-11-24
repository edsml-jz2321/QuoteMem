"""Factory for database-owned workspace MCP tools."""

from __future__ import annotations

__all__ = ["database_tool_functions"]

from heavenbase.utils import Any, Dict

from .tools.ingest_database import build_tool as build_ingest_database_tool
from .tools.list_sources import build_tool as build_list_sources_tool
from .tools.run_sql import build_tool as build_run_sql_tool
from .tools.run_snippet import build_tool as build_run_snippet_tool
from .tools.save_snippet import build_tool as build_save_snippet_tool
from .tools.suggest_sql import build_tool as build_suggest_sql_tool


def database_tool_functions(workspace: Any) -> Dict[str, Any]:
    """Return MCP-callable database tool functions bound to one workspace.

    Args:
        workspace (Any): Live workspace bound to database MCP tools.

    Returns:
        Dict[str, Any]: Tool name to callable mapping for database MCP tools.
    """
    return {
        "list_sources": build_list_sources_tool(workspace),
        "ingest_database": build_ingest_database_tool(workspace),
        "run_sql": build_run_sql_tool(workspace),
        "run_snippet": build_run_snippet_tool(workspace),
        "save_snippet": build_save_snippet_tool(workspace),
        "suggest_sql": build_suggest_sql_tool(workspace),
    }
