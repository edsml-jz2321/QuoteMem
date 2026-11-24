"""Workspace MCP composition service."""

from __future__ import annotations

from heavenbase._lazy import _attach_lazy_exports

__all__ = ["WorkspaceSkill"]

_LAZY_EXPORTS = {
    "WorkspaceSkill": (".skill", "WorkspaceSkill"),
    # Private, kept importable from this package as in 0.1.2.2.
    "_MCPService": (".service", "_MCPService"),
}


__getattr__, __dir__ = _attach_lazy_exports(__name__, globals(), _LAZY_EXPORTS, __all__)
