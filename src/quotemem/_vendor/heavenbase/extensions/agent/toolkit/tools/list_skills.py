"""MCP tool for listing available workspace skills."""

from __future__ import annotations

__all__ = ["build_tool"]

from heavenbase.utils import Any, Dict

from ..errors import mcp_raise
from ..skills import list_skill_entries


def build_tool(context: Any) -> Any:
    """Build the ``list_skills`` tool for one workspace context.

    Args:
        context (Any): Workspace MCP tool context supplying workspace and skill
            scope.

    Returns:
        Any: Callable MCP tool function for listing skills.
    """

    def list_skills() -> Dict[str, Any]:
        """List skills available under the active MCP profile scope.

        Returns:
            Workspace id and sorted skill entries with names and descriptions.

        Raises:
            RuntimeError: If skill listing fails; the message is redacted for MCP use.
        """
        try:
            return {"workspace": context.workspace.id, "skills": list_skill_entries(context.workspace, context.skill_scope)}
        except Exception as exc:
            mcp_raise(exc)

    return list_skills
