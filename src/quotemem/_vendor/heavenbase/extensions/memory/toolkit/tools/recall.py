"""MCP tool for recalling one memory note."""

from __future__ import annotations

__all__ = ["build_tool"]

from heavenbase.utils import Any, Dict

from ...extension import ensure_memory_entity
from ..errors import mcp_raise


def build_tool(workspace: Any) -> Any:
    """Build the ``recall`` tool for one workspace.

    Args:
        workspace (Any): Live workspace bound to memory MCP tools.

    Returns:
        Any: Callable MCP tool function for ``recall``.
    """

    def recall(key: str) -> Dict[str, Any]:
        """Read one memory note by key."""
        try:
            entity = ensure_memory_entity(workspace)
            return {"object_id": key, "row": workspace.get(key, entity=entity)}
        except Exception as exc:
            mcp_raise(exc)

    return recall
