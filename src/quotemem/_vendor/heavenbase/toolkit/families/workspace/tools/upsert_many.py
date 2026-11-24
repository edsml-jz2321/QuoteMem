"""MCP tool for upserting multiple rows."""

from __future__ import annotations

__all__ = ["build_tool"]

from heavenbase.utils import Any, Dict, List

from ..context import WorkspaceToolContext
from ..errors import mcp_raise


def build_tool(context: WorkspaceToolContext) -> Any:
    """Build the ``upsert_many`` tool for one workspace context.

    Args:
        context (WorkspaceToolContext): Shared workspace and profile state for the tool factory.

    Returns:
        Any: MCP-callable ``upsert_many`` function bound to ``context``.
    """

    def upsert_many(entity: str, rows: List[Dict[str, Any]], overwrite: bool = True) -> Dict[str, Any]:
        """Insert or update multiple rows for an explicit entity.

        Args:
            entity: Entity id, class name, or display name to resolve.
            rows: Row payloads to write.
            overwrite: Whether values in ``rows`` may replace existing values.

        Returns:
            Written object ids and row count.

        Raises:
            RuntimeError: If the bulk write fails; the message is redacted for MCP use.
        """
        try:
            resolved = context.require_entity(entity)
            with context.workspace._lock:
                object_ids = context.workspace._crud.upsert_many(
                    resolved,
                    rows,
                    overwrite=overwrite,
                    allowed_entities=context.allowed_entities,
                )
            return {"object_ids": object_ids, "n": len(object_ids)}
        except Exception as exc:
            mcp_raise(exc)

    return upsert_many
