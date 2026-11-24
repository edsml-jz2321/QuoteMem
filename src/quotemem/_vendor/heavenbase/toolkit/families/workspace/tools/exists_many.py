"""MCP tool for checking multiple object ids."""

from __future__ import annotations

__all__ = ["build_tool"]

from heavenbase.utils import Any, Dict, List, Optional

from ..context import WorkspaceToolContext
from ..errors import mcp_raise


def build_tool(context: WorkspaceToolContext) -> Any:
    """Build the ``exists_many`` tool for one workspace context.

    Args:
        context (WorkspaceToolContext): Shared workspace and profile state for the tool factory.

    Returns:
        Any: MCP-callable ``exists_many`` function bound to ``context``.
    """

    def exists_many(object_ids: List[str], entity: Optional[str] = None) -> Dict[str, Any]:
        """Check whether multiple object ids exist.

        Args:
            object_ids: User-facing object ids to check.
            entity: Optional entity id, class name, or display name for disambiguation.

        Returns:
            Object ids, per-object existence flags, and flag count.

        Raises:
            RuntimeError: If the bulk existence check fails; the message is redacted for MCP use.
        """
        try:
            resolved = context.require_entity(entity)
            if resolved is not None or context.entity_scope == "*":
                exists_flags = context.workspace.exists_many(object_ids, entity=resolved)
            else:
                exists_flags = [context.scoped_existing_entity(object_id) is not None for object_id in object_ids]
            return {"object_ids": object_ids, "exists": exists_flags, "n": len(exists_flags)}
        except Exception as exc:
            mcp_raise(exc)

    return exists_many
