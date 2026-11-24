"""MCP tool for explaining workspace queries."""

from __future__ import annotations

__all__ = ["build_tool"]

from heavenbase.utils import Any, Dict, Optional

from ..context import WorkspaceToolContext
from ..errors import mcp_raise


def build_tool(context: WorkspaceToolContext) -> Any:
    """Build the ``explain`` tool for one workspace context.

    Args:
        context (WorkspaceToolContext): Shared workspace and profile state for the tool factory.

    Returns:
        Any: MCP-callable ``explain`` function bound to ``context``.
    """

    def explain(entity: str, spec: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Explain a Mongo-style JSON query for one entity.

        Args:
            entity: Entity id, class name, or display name to resolve.
            spec: Optional Mongo-style query specification.

        Returns:
            Query explanation produced by the workspace router.

        Raises:
            RuntimeError: If routing fails; the message is redacted for MCP use.
        """
        try:
            resolved = context.require_entity(entity)
            return context.workspace.query_json(resolved, spec or {}).explain()
        except Exception as exc:
            mcp_raise(exc)

    return explain
