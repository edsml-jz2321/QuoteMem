"""MCP tool for reading multiple rows."""

from __future__ import annotations

__all__ = ["build_tool"]

from heavenbase.utils import Any, Dict, List, Optional

from ..context import WorkspaceToolContext
from ..errors import mcp_raise


def build_tool(context: WorkspaceToolContext) -> Any:
    """Build the ``get_many`` tool for one workspace context.

    Args:
        context (WorkspaceToolContext): Shared workspace and profile state for the tool factory.

    Returns:
        Any: MCP-callable ``get_many`` function bound to ``context``.
    """

    def get_many(object_ids: List[str], entity: Optional[str] = None) -> Dict[str, Any]:
        """Read multiple rows by object id.

        Args:
            object_ids: User-facing object ids to read.
            entity: Optional entity id, class name, or display name for disambiguation.

        Returns:
            Requested object ids, matching rows, and row count.

        Raises:
            RuntimeError: If any read fails; the message is redacted for MCP use.
        """
        try:
            resolved = context.require_entity(entity)
            if resolved is not None or context.entity_scope == "*":
                rows = context.workspace.get_many(object_ids, entity=resolved)
            else:
                rows = [
                    context.workspace.get(object_id, entity=scoped_entity) if (scoped_entity := context.scoped_existing_entity(object_id)) is not None else None
                    for object_id in object_ids
                ]
            return {"object_ids": object_ids, "rows": rows, "n": len(rows)}
        except Exception as exc:
            mcp_raise(exc)

    return get_many
