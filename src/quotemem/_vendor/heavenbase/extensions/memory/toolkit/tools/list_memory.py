"""MCP tool for listing a bounded page of memory notes."""

from __future__ import annotations

__all__ = ["build_tool"]

from heavenbase.utils import Any, Dict, Optional

from ...extension import ensure_memory_entity
from .._utils import MEMORY_SELECT, list_limit
from ..errors import mcp_raise
from ..result import frame_dict


def build_tool(workspace: Any) -> Any:
    """Build the ``list_memory`` tool for one workspace.

    Args:
        workspace (Any): Live workspace bound to memory MCP tools.

    Returns:
        Any: Callable MCP tool function for ``list_memory``.
    """

    def list_memory(limit: Optional[int] = None) -> Dict[str, Any]:
        """List memory notes in storage order, bounded by a limit.

        Args:
            limit (Optional[int]): Maximum notes to return. Defaults to
                ``heavenbase.mcp.memory.list_limit`` (20), so an unset limit
                still bounds the answer.

        Returns:
            Dict[str, Any]: Payload with ``columns``, ``rows``, ``ids``, and
            ``n`` returned notes. Rows are the first notes the entity yields,
            not the newest ones — this tool applies no recency order. When the
            limit withheld notes the payload also carries ``truncated`` and
            ``total``, the exact number of stored notes; report ``total`` rather
            than ``n`` as the count, and raise ``limit`` to see the rest.

        Raises:
            RuntimeError: If listing fails, with the cause redacted.
        """
        try:
            entity = ensure_memory_entity(workspace)
            spec = {"select": MEMORY_SELECT, "limit": int(limit or list_limit(config=workspace.context.config))}
            return frame_dict(workspace.query_json(entity, spec).execute())
        except Exception as exc:
            mcp_raise(exc)

    return list_memory
