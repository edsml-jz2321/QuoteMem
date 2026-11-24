"""MCP tool for deleting multiple rows."""

from __future__ import annotations

__all__ = ["build_tool"]

from heavenbase.utils import Any, Dict, List, Literal

from ..context import WorkspaceToolContext
from ..errors import mcp_raise


def build_tool(context: WorkspaceToolContext) -> Any:
    """Build the ``delete_many`` tool for one workspace context.

    Args:
        context (WorkspaceToolContext): Shared workspace and profile state for the tool factory.

    Returns:
        Any: MCP-callable ``delete_many`` function bound to ``context``.
    """

    def delete_many(targets: List[Any], edges: Literal["schema", "restrict", "detach"] = "schema") -> Dict[str, Any]:
        """Delete multiple row or entity-qualified row targets.

        Args:
            targets: Delete targets accepted by ``HeavenBase.delete_many``.
            edges: Incoming GraphEdge policy: `schema`, `restrict`, or `detach`.

        Returns:
            Original targets, per-target delete flags, target count, and deleted count.

        Raises:
            RuntimeError: If the bulk delete fails; the message is redacted for MCP use.
        """
        try:
            deleted = context.delete_targets(targets, edges=edges)
            return {"targets": targets, "deleted": deleted, "n": len(deleted), "deleted_n": sum(1 for item in deleted if item)}
        except Exception as exc:
            mcp_raise(exc)

    return delete_many
