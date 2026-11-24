"""Factory for agent-owned workspace MCP tools."""

from __future__ import annotations

__all__ = ["agent_tool_functions"]

from heavenbase.utils import Any, Dict

from .tools.list_skills import build_tool as build_list_skills_tool
from .tools.read_skill import build_tool as build_read_skill_tool


def agent_tool_functions(context: Any) -> Dict[str, Any]:
    """Return MCP-callable agent tool functions bound to one workspace context.

    Args:
        context (Any): Workspace MCP tool context supplying workspace and skill
            scope.

    Returns:
        Dict[str, Any]: Tool name to callable mapping for agent MCP tools.
    """
    return {
        "list_skills": build_list_skills_tool(context),
        "read_skill": build_read_skill_tool(context),
    }
