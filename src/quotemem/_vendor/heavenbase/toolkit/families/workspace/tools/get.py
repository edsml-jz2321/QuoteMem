"""MCP tool for reading one row."""

from __future__ import annotations

__all__ = ["build_tool"]

from heavenbase.utils import Any, Dict, Optional

from ..context import WorkspaceToolContext
from ..errors import mcp_raise


def build_tool(context: WorkspaceToolContext) -> Any:
    """Build the ``get`` tool for one workspace context.

    Args:
        context (WorkspaceToolContext): Shared workspace and profile state for the tool factory.

    Returns:
        Any: MCP-callable ``get`` function bound to ``context``.
    """

    def get(object_id: str, entity: Optional[str] = None) -> Dict[str, Any]:
        """Read one row by object id.

        Args:
            object_id: User-facing object id to read.
            entity: Optional entity id, class name, or display name for disambiguation.

        Returns:
            Row payload for ``object_id``.

        Raises:
            RuntimeError: If the row cannot be read; the message is redacted for MCP use.
        """
        try:
            resolved = context.entity_for_read(object_id, entity)
            return {"row": None if resolved is None and context.entity_scope != "*" else context.workspace.get(object_id, entity=resolved)}
        except Exception as exc:
            mcp_raise(exc)

    return get
