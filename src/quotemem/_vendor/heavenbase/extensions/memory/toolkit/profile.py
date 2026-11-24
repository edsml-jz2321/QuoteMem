"""Memory MCP toolkit vocabulary."""

from __future__ import annotations

__all__ = ["MEMORY_TOOLS", "MEMSTATE_TOOLS"]

MEMORY_TOOLS = ("remember", "recall", "search_memory", "list_memory", "set_memory")
MEMSTATE_TOOLS = (
    "memstate_remember",
    "memstate_set",
    "memstate_get",
    "memstate_search",
    "memstate_history",
    "memstate_delete",
)
