"""Core workspace MCP toolkit family."""

from __future__ import annotations

from heavenbase._lazy import _attach_lazy_exports

__all__ = [
    "WORKSPACE_CORE_TOOL_NAMES",
    "WORKSPACE_QUERY_TOOL_NAMES",
    "WORKSPACE_READ_TOOL_NAMES",
    "WORKSPACE_SCHEMA_TOOL_NAMES",
    "WORKSPACE_WRITE_TOOL_NAMES",
    "WORKSPACE_GUIDE_SKILL",
    "WorkspaceToolContext",
    "entity_summaries",
    "render_workspace_guide",
    "workspace_core_tool_functions",
    "workspace_schema_tool_functions",
]

_LAZY_EXPORTS = {
    "WORKSPACE_CORE_TOOL_NAMES": (".profile", "WORKSPACE_CORE_TOOL_NAMES"),
    "WORKSPACE_QUERY_TOOL_NAMES": (".profile", "WORKSPACE_QUERY_TOOL_NAMES"),
    "WORKSPACE_READ_TOOL_NAMES": (".profile", "WORKSPACE_READ_TOOL_NAMES"),
    "WORKSPACE_SCHEMA_TOOL_NAMES": (".profile", "WORKSPACE_SCHEMA_TOOL_NAMES"),
    "WORKSPACE_WRITE_TOOL_NAMES": (".profile", "WORKSPACE_WRITE_TOOL_NAMES"),
    "WORKSPACE_GUIDE_SKILL": (".guide", "WORKSPACE_GUIDE_SKILL"),
    "WorkspaceToolContext": (".context", "WorkspaceToolContext"),
    "entity_summaries": (".guide", "entity_summaries"),
    "render_workspace_guide": (".guide", "render_workspace_guide"),
    "workspace_core_tool_functions": (".factory", "workspace_core_tool_functions"),
    "workspace_schema_tool_functions": (".factory", "workspace_schema_tool_functions"),
}

__getattr__, __dir__ = _attach_lazy_exports(__name__, globals(), _LAZY_EXPORTS, __all__)
