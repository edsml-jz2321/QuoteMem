"""MCP tool for searching memory notes."""

from __future__ import annotations

__all__ = ["build_tool"]

from heavenbase.utils import Any, Dict, Optional

from ...extension import ensure_memory_entity
from .._utils import MEMORY_SELECT, search_limit
from ..errors import mcp_raise
from ..result import frame_dict


def build_tool(workspace: Any) -> Any:
    """Build the ``search_memory`` tool for one workspace.

    Args:
        workspace (Any): Live workspace bound to memory MCP tools.

    Returns:
        Any: Callable MCP tool function for ``search_memory``.
    """

    def search_memory(text: str, limit: Optional[int] = None) -> Dict[str, Any]:
        """Search memory notes by text, bounded by a limit.

        Args:
            text (str): Text matched against the note body. An empty value
                selects every note.
            limit (Optional[int]): Maximum notes to return. Defaults to
                ``heavenbase.mcp.memory.search_limit`` (10), so an unset limit
                still bounds the answer.

        Returns:
            Dict[str, Any]: Payload with ``columns``, ``rows``, ``ids``, and
            ``n`` returned notes. Rows are the first matches the entity yields,
            not the best ones — there is no relevance ranking. When the limit
            withheld matches the payload also carries ``truncated`` and
            ``total``, the exact number that matched; report ``total`` rather
            than ``n`` as the count, and raise ``limit`` to see the rest.

        Raises:
            RuntimeError: If the search fails, with the cause redacted.
        """
        try:
            entity = ensure_memory_entity(workspace)
            spec: Dict[str, Any] = {"select": MEMORY_SELECT, "limit": int(limit or search_limit(config=workspace.context.config))}
            if text:
                spec["filter"] = {"text": {"$match": text}}
            return frame_dict(workspace.query_json(entity, spec).execute())
        except Exception as exc:
            mcp_raise(exc)

    return search_memory
