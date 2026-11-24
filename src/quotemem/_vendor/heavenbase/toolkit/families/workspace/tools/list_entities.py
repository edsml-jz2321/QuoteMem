"""MCP tool for listing registered entities."""

from __future__ import annotations

__all__ = ["build_tool"]

from heavenbase.utils import Any, Dict

from ..context import WorkspaceToolContext
from ..errors import mcp_raise


def build_tool(context: WorkspaceToolContext) -> Any:
    """Build the ``list_entities`` tool for one workspace context.

    Args:
        context (WorkspaceToolContext): Shared workspace and profile state for the tool factory.

    Returns:
        Any: MCP-callable ``list_entities`` function bound to ``context``.
    """

    def list_entities(include_system: bool = False) -> Dict[str, Any]:
        """List entity ids registered in the workspace.

        Args:
            include_system: Whether to include system/catalog entities.

        Returns:
            Workspace id and sorted entity ids.

        Raises:
            RuntimeError: If entity listing fails; the message is redacted for MCP use.
        """
        try:
            entity_ids = context.filter_entities(sorted(context.workspace.entities) if include_system else context.workspace.list_entities())
            return {"workspace": context.workspace.id, "entities": entity_ids}
        except Exception as exc:
            mcp_raise(exc)

    return list_entities
