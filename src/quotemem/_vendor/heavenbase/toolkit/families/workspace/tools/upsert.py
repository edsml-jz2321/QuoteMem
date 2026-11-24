"""MCP tool for upserting one row."""

from __future__ import annotations

__all__ = ["build_tool"]

from heavenbase.utils import Any, Dict

from ..context import WorkspaceToolContext
from ..errors import mcp_raise


def build_tool(context: WorkspaceToolContext) -> Any:
    """Build the ``upsert`` tool for one workspace context.

    Args:
        context (WorkspaceToolContext): Shared workspace and profile state for the tool factory.

    Returns:
        Any: MCP-callable ``upsert`` function bound to ``context``.
    """

    def upsert(entity: str, row: Dict[str, Any], overwrite: bool = True) -> Dict[str, Any]:
        """Insert or update one row for an explicit entity.

        Args:
            entity: Entity id, class name, or display name to resolve.
            row: Row payload to write.
            overwrite: Whether values in ``row`` may replace existing values.

        Returns:
            Written object id and the stored row.

        Raises:
            RuntimeError: If the write fails; the message is redacted for MCP use.
        """
        try:
            resolved = context.require_entity(entity)
            with context.workspace._lock:
                object_id = context.workspace._crud.upsert(
                    resolved,
                    row,
                    overwrite=overwrite,
                    allowed_entities=context.allowed_entities,
                )
            return {"object_id": object_id, "row": context.workspace.get(object_id, entity=resolved)}
        except Exception as exc:
            mcp_raise(exc)

    return upsert
