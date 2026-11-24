"""Workspace MCP toolkit family public surface."""

from __future__ import annotations

from heavenbase._lazy import _attach_lazy_exports

__all__ = [
    "WorkspaceToolkitFamilySpec",
    "register_workspace_toolkit_family",
    "registered_workspace_toolkit_families",
    "workspace_toolkit_family_specs",
    "workspace_toolkit_tool_names",
]

_LAZY_EXPORTS = {
    "WorkspaceToolkitFamilySpec": (".family", "WorkspaceToolkitFamilySpec"),
    "register_workspace_toolkit_family": (".registry", "register_workspace_toolkit_family"),
    "registered_workspace_toolkit_families": (".registry", "registered_workspace_toolkit_families"),
    "workspace_toolkit_family_specs": (".registry", "workspace_toolkit_family_specs"),
    "workspace_toolkit_tool_names": (".registry", "workspace_toolkit_tool_names"),
}


__getattr__, __dir__ = _attach_lazy_exports(__name__, globals(), _LAZY_EXPORTS, __all__)
