"""MCP and Toolkit CLI command group facade."""

from __future__ import annotations

from heavenbase._lazy import _attach_lazy_exports

__all__ = ["mcp_group"]

_LAZY_EXPORTS = {
    "mcp_group": (".group", "mcp_group"),
}


__getattr__, __dir__ = _attach_lazy_exports(__name__, globals(), _LAZY_EXPORTS, __all__)
