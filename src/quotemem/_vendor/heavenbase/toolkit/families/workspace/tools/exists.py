"""MCP tool for checking one object id."""

from __future__ import annotations

__all__ = ["build_tool"]

from heavenbase.utils import Any, Dict, Optional

from ..context import WorkspaceToolContext
from ..errors import mcp_raise


def build_tool(context: WorkspaceToolContext) -> Any:
    """Build the ``exists`` tool for one workspace context.

    Args:
        context (WorkspaceToolContext): Shared workspace and profile state for the tool factory.

    Returns:
        Any: MCP-callable ``exists`` function bound to ``context``.
    """

    def exists(object_id: str, entity: Optional[str] = None) -> Dict[str, Any]:
        """Check whether one object id exists.

        Args:
            object_id: User-facing object id to check.
            entity: Optional entity id, class name, or display name for disambiguation.

        Returns:
            Object id and existence flag.

        Raises:
            RuntimeError: If the existence check fails; the message is redacted for MCP use.
        """
        try:
            resolved = context.entity_for_read(object_id, entity)
            return {
                "object_id": object_id,
                "exists": False if resolved is None and context.entity_scope != "*" else context.workspace.exists(object_id, entity=resolved),
            }
        except Exception as exc:
            mcp_raise(exc)

    return exists
