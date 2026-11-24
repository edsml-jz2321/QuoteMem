"""MCP tool for counting entity rows."""

from __future__ import annotations

__all__ = ["build_tool"]

from heavenbase.utils import Any, Dict

from ..context import WorkspaceToolContext
from ..errors import mcp_raise


def build_tool(context: WorkspaceToolContext) -> Any:
    """Build the ``count`` tool for one workspace context.

    Args:
        context (WorkspaceToolContext): Shared workspace and profile state for the tool factory.

    Returns:
        Any: MCP-callable ``count`` function bound to ``context``.
    """

    def count(entity: str) -> Dict[str, Any]:
        """Count rows for one entity.

        Args:
            entity: Entity id, class name, or display name to resolve.

        Returns:
            Resolved entity id and row count.

        Raises:
            RuntimeError: If counting fails; the message is redacted for MCP use.
        """
        try:
            resolved = context.require_entity(entity)
            return {"entity": resolved, "n": context.workspace.count(resolved)}
        except Exception as exc:
            mcp_raise(exc)

    return count
