"""MCP tool for deleting one row."""

from __future__ import annotations

__all__ = ["build_tool"]

from heavenbase.utils import Any, Dict, Literal

from ..context import WorkspaceToolContext
from ..errors import mcp_raise


def build_tool(context: WorkspaceToolContext) -> Any:
    """Build the ``delete`` tool for one workspace context.

    Args:
        context (WorkspaceToolContext): Shared workspace and profile state for the tool factory.

    Returns:
        Any: MCP-callable ``delete`` function bound to ``context``.
    """

    def delete(target: Any, edges: Literal["schema", "restrict", "detach"] = "schema") -> Dict[str, Any]:
        """Delete one row or entity-qualified row target.

        Args:
            target: Delete target accepted by ``HeavenBase.delete``.
            edges: Incoming GraphEdge policy: `schema`, `restrict`, or `detach`.

        Returns:
            Original target and delete result flag.

        Raises:
            RuntimeError: If the delete fails; the message is redacted for MCP use.
        """
        try:
            return {"target": target, "deleted": context.delete_targets([target], edges=edges)[0]}
        except Exception as exc:
            mcp_raise(exc)

    return delete
