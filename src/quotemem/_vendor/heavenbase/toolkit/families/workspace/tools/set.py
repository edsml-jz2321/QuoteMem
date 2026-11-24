"""MCP tool for setting one row."""

from __future__ import annotations

__all__ = ["build_tool"]

from heavenbase.utils import Any, Dict, Optional

from ..context import WorkspaceToolContext
from ..errors import mcp_raise


def build_tool(context: WorkspaceToolContext) -> Any:
    """Build the ``set`` tool for one workspace context.

    Args:
        context (WorkspaceToolContext): Shared workspace and profile state for the tool factory.

    Returns:
        Any: MCP-callable ``set`` function bound to ``context``.
    """

    def set(row: Dict[str, Any], entity: Optional[str] = None, overwrite: bool = True) -> Dict[str, Any]:
        """Set one row using its embedded object id.

        Args:
            row: Row payload containing ``object_id`` or enough data to compute it.
            entity: Optional entity id, class name, or display name for disambiguation.
            overwrite: Whether values in ``row`` may replace existing values.

        Returns:
            Written object id and the stored row.

        Raises:
            RuntimeError: If the write fails; the message is redacted for MCP use.
        """
        try:
            with context.workspace._lock:
                resolved = context.entity_for_patch(row, entity)
                object_id = context.workspace._crud.set(
                    row,
                    entity=resolved,
                    overwrite=overwrite,
                    allowed_entities=context.allowed_entities,
                )
                stored = context.workspace.get(object_id, entity=resolved)
            return {"object_id": object_id, "row": stored}
        except Exception as exc:
            mcp_raise(exc)

    return set
