"""MCP tool for describing one registered entity."""

from __future__ import annotations

__all__ = ["build_tool"]

from heavenbase.utils import Any, Dict

from ..context import WorkspaceToolContext
from ..errors import mcp_raise


def build_tool(context: WorkspaceToolContext) -> Any:
    """Build the ``describe_entity`` tool for one workspace context.

    Args:
        context (WorkspaceToolContext): Shared workspace and profile state for the tool factory.

    Returns:
        Any: MCP-callable ``describe_entity`` function bound to ``context``.
    """

    def describe_entity(entity: str) -> Dict[str, Any]:
        """Describe one registered workspace entity.

        Args:
            entity: Entity id, class name, or display name to resolve.

        Returns:
            Entity schema, storage plan, derives, and query capabilities.

        Raises:
            RuntimeError: If the entity cannot be resolved or described; the message is redacted for MCP use.
        """
        try:
            return context.entity_dict(context.registered_entity(context.require_entity(entity)))
        except Exception as exc:
            mcp_raise(exc)

    return describe_entity
