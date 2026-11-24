"""Capsule-backed executable Toolkits."""

from __future__ import annotations

from heavenbase._lazy import _attach_registry_exports

__all__ = [
    "Tool",
    "AsyncToolRequired",
    "ToolExportError",
    "Toolkit",
    "ToolkitError",
    "ToolkitRegistry",
    "WorkspaceToolContext",
    "arun_for_anthropic",
    "backend_toolkit",
    "callables_from_fastmcp",
    "describe_toolkit_ref",
    "load_toolkit_ref",
    "parse_toolkit_ref",
    "raise_tool_error",
    "run_for_anthropic",
    "sync_mcp_registry_metadata",
    "to_anthropic_tools",
    "toolkit_to_fastmcp",
    "toolkit_from_fastmcp",
    "toolkit_registry",
    "toolkit_to_mcp_json",
    "workspace_toolkit",
]

_REGISTRY_ENTITY_EXPORTS = {
    "Toolkit": "sys-toolkit",
}

_REGISTRY_SIBLING_EXPORTS = {
    "AsyncToolRequired": ("sys-toolkit", "errors", "AsyncToolRequired"),
    "Tool": ("sys-toolkit", "tool", "Tool"),
    "ToolExportError": ("sys-toolkit", "errors", "ToolExportError"),
    "ToolkitError": ("sys-toolkit", "errors", "ToolkitError"),
    "ToolkitRegistry": ("sys-toolkit", "registry", "ToolkitRegistry"),
    "WorkspaceToolContext": ("sys-toolkit", "families.workspace.context", "WorkspaceToolContext"),
    "arun_for_anthropic": ("sys-toolkit", "anthropic", "arun_for_anthropic"),
    "backend_toolkit": ("sys-toolkit", "backend_tools", "backend_toolkit"),
    "callables_from_fastmcp": ("sys-toolkit", "mcp", "callables_from_fastmcp"),
    "describe_toolkit_ref": ("sys-toolkit", "refs", "describe_toolkit_ref"),
    "load_toolkit_ref": ("sys-toolkit", "refs", "load_toolkit_ref"),
    "parse_toolkit_ref": ("sys-toolkit", "refs", "parse_toolkit_ref"),
    "raise_tool_error": ("sys-toolkit", "errors", "raise_tool_error"),
    "run_for_anthropic": ("sys-toolkit", "anthropic", "run_for_anthropic"),
    "sync_mcp_registry_metadata": ("sys-toolkit", "registry_metadata", "sync_mcp_registry_metadata"),
    "to_anthropic_tools": ("sys-toolkit", "anthropic", "to_anthropic_tools"),
    "toolkit_from_fastmcp": ("sys-toolkit", "mcp", "toolkit_from_fastmcp"),
    "toolkit_registry": ("sys-toolkit", "registry", "toolkit_registry"),
    "toolkit_to_fastmcp": ("sys-toolkit", "mcp", "toolkit_to_fastmcp"),
    "toolkit_to_mcp_json": ("sys-toolkit", "mcp", "toolkit_to_mcp_json"),
    "workspace_toolkit": ("sys-toolkit", "workspace_tools", "workspace_toolkit"),
}

__getattr__, __dir__ = _attach_registry_exports(
    globals(),
    __all__,
    entities=_REGISTRY_ENTITY_EXPORTS,
    siblings=_REGISTRY_SIBLING_EXPORTS,
)
