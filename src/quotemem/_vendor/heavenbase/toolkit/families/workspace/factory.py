"""Builders for core workspace MCP tool groups."""

from __future__ import annotations

__all__ = ["workspace_core_tool_functions", "workspace_explain_tool", "workspace_query_tool", "workspace_schema_tool_functions"]

from heavenbase.utils import Any, Dict

from ...profiles import resolve_profile
from .context import WorkspaceToolContext
from .profile import WORKSPACE_SCHEMA_TOOL_NAMES
from .tools.count import build_tool as build_count_tool
from .tools.delete import build_tool as build_delete_tool
from .tools.delete_many import build_tool as build_delete_many_tool
from .tools.define_entity import build_tool as build_define_entity_tool
from .tools.describe_entity import build_tool as build_describe_entity_tool
from .tools.exists import build_tool as build_exists_tool
from .tools.exists_many import build_tool as build_exists_many_tool
from .tools.explain import build_tool as build_explain_tool
from .tools.get import build_tool as build_get_tool
from .tools.get_many import build_tool as build_get_many_tool
from .tools.list_entities import build_tool as build_list_entities_tool
from .tools.query import build_tool as build_query_tool
from .tools.set import build_tool as build_set_tool
from .tools.set_many import build_tool as build_set_many_tool
from .tools.upsert import build_tool as build_upsert_tool
from .tools.upsert_many import build_tool as build_upsert_many_tool


def workspace_schema_tool_functions(context: WorkspaceToolContext) -> Dict[str, Any]:
    """Build core schema/discovery MCP tools for one workspace context.

    Args:
        context (WorkspaceToolContext): Shared workspace and profile state for the tool factory.

    Returns:
        Dict[str, Any]: ``define_entity``, ``list_entities``, and ``describe_entity`` callables keyed by tool name.
    """
    return {
        WORKSPACE_SCHEMA_TOOL_NAMES[0]: build_define_entity_tool(context),
        WORKSPACE_SCHEMA_TOOL_NAMES[1]: build_list_entities_tool(context),
        WORKSPACE_SCHEMA_TOOL_NAMES[2]: build_describe_entity_tool(context),
    }


def workspace_core_tool_functions(context: WorkspaceToolContext) -> Dict[str, Any]:
    """Build all core workspace MCP tools for one workspace context.

    Args:
        context (WorkspaceToolContext): Shared workspace and profile state for the tool factory.

    Returns:
        Dict[str, Any]: Full core workspace tool surface keyed by tool name.
    """
    funcs = workspace_schema_tool_functions(context)
    funcs.update(
        {
            "upsert": build_upsert_tool(context),
            "upsert_many": build_upsert_many_tool(context),
            "get": build_get_tool(context),
            "get_many": build_get_many_tool(context),
            "set": build_set_tool(context),
            "set_many": build_set_many_tool(context),
            "delete": build_delete_tool(context),
            "delete_many": build_delete_many_tool(context),
            "exists": build_exists_tool(context),
            "exists_many": build_exists_many_tool(context),
            "count": build_count_tool(context),
            "query": build_query_tool(context),
            "explain": build_explain_tool(context),
        }
    )
    return funcs


def workspace_query_tool(workspace: Any, *, profile: str = "agent") -> Any:
    """Build the canonical query callable for one eligible workspace.

    Args:
        workspace (Any): HeavenBase workspace whose rows should be queried.
        profile (str): MCP profile name used to resolve entity scope. Defaults to ``"agent"``.

    Returns:
        Any: MCP-callable ``query`` function bound to the resolved profile context.
    """
    return build_query_tool(_workspace_tool_context(workspace, profile, "query"))


def workspace_explain_tool(workspace: Any, *, profile: str = "agent") -> Any:
    """Build the canonical explain callable for one eligible workspace.

    Explain validates a spec and returns the router's plan without executing
    it, so it reads no rows and needs none of the result bounds a query needs.

    Args:
        workspace (Any): HeavenBase workspace whose routing should be explained.
        profile (str): MCP profile name used to resolve entity scope. Defaults to ``"agent"``.

    Returns:
        Any: MCP-callable ``explain`` function bound to the resolved profile context.
    """
    return build_explain_tool(_workspace_tool_context(workspace, profile, "explain"))


def _workspace_tool_context(workspace: Any, profile: str, tool_name: str) -> WorkspaceToolContext:
    """Resolve one profile-scoped context for a single-tool builder."""
    resolved = resolve_profile(
        profile,
        resolver=workspace.module_resolver,
        config=workspace.context.config,
        workspace=workspace,
    )
    return WorkspaceToolContext(workspace=workspace, profile=resolved, tool_names=(tool_name,))
