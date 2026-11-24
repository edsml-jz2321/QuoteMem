"""Factory for memory-owned workspace MCP tools."""

from __future__ import annotations

__all__ = ["memory_tool_functions"]

from heavenbase.utils import Any, Dict

from .tools.list_memory import build_tool as build_list_memory_tool
from .tools.memstate_delete import build_tool as build_memstate_delete_tool
from .tools.memstate_get import build_tool as build_memstate_get_tool
from .tools.memstate_history import build_tool as build_memstate_history_tool
from .tools.memstate_remember import build_tool as build_memstate_remember_tool
from .tools.memstate_search import build_tool as build_memstate_search_tool
from .tools.memstate_set import build_tool as build_memstate_set_tool
from .tools.recall import build_tool as build_recall_tool
from .tools.remember import build_tool as build_remember_tool
from .tools.search_memory import build_tool as build_search_memory_tool
from .tools.set_memory import build_tool as build_set_memory_tool


def memory_tool_functions(workspace: Any) -> Dict[str, Any]:
    """Return MCP-callable memory tool functions bound to one workspace.

    Args:
        workspace (Any): Live workspace bound to memory MCP tools.

    Returns:
        Dict[str, Any]: Tool name to callable mapping for memory MCP tools.
    """
    return {
        "remember": build_remember_tool(workspace),
        "recall": build_recall_tool(workspace),
        "search_memory": build_search_memory_tool(workspace),
        "list_memory": build_list_memory_tool(workspace),
        "set_memory": build_set_memory_tool(workspace),
        "memstate_remember": build_memstate_remember_tool(workspace),
        "memstate_set": build_memstate_set_tool(workspace),
        "memstate_get": build_memstate_get_tool(workspace),
        "memstate_search": build_memstate_search_tool(workspace),
        "memstate_history": build_memstate_history_tool(workspace),
        "memstate_delete": build_memstate_delete_tool(workspace),
    }
