"""Memory extension MCP toolkit family."""

from __future__ import annotations

from heavenbase._lazy import _attach_lazy_exports

__all__ = [
    "MEMORY_TOOLS",
    "MEMSTATE_TOOLS",
    "memory_tool_functions",
]

_LAZY_EXPORTS = {
    "MEMORY_TOOLS": (".profile", "MEMORY_TOOLS"),
    "MEMSTATE_TOOLS": (".profile", "MEMSTATE_TOOLS"),
    "memory_tool_functions": (".factory", "memory_tool_functions"),
}

__getattr__, __dir__ = _attach_lazy_exports(__name__, globals(), _LAZY_EXPORTS, __all__)
