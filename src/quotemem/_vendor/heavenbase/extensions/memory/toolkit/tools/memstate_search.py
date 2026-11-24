"""MCP tool for searching current memstate rows."""

from __future__ import annotations

__all__ = ["build_tool"]

from heavenbase.utils import Any, Dict, Optional

from ...extension import ensure_memory_entity
from ...versions import MemoryVersions
from .._utils import search_limit
from ..errors import mcp_raise


def build_tool(workspace: Any) -> Any:
    """Build the ``memstate_search`` tool for one workspace.

    Args:
        workspace (Any): Live workspace bound to memory MCP tools.

    Returns:
        Any: Callable MCP tool function for ``memstate_search``.
    """

    def memstate_search(query: str, project_id: str, limit: Optional[int] = None) -> Dict[str, Any]:
        """Search current memstate rows for a project, bounded by a limit.

        Args:
            query (str): Case-insensitive terms matched across keypath, title,
                text, and tags. An empty value selects every current row.
            project_id (str): Project namespace. Empty values resolve to `default`.
            limit (Optional[int]): Maximum rows to return. Defaults to
                ``heavenbase.mcp.memory.search_limit`` (10), so an unset limit
                still bounds the answer.

        Returns:
            Dict[str, Any]: Payload with ``project_id``, ``query``, ``rows``,
            ``ids``, ``n``, the applied ``limit``, ``total``, and ``truncated``.
            Rows are ordered by keypath rather than relevance, so a cut list is
            not the best matches. ``n`` counts returned rows and ``total`` counts
            every row that matched; report ``total`` as the match count and
            raise ``limit`` to see the rest.

        Raises:
            RuntimeError: If the search fails, with the cause redacted.
        """
        try:
            entity = ensure_memory_entity(workspace)
            versions = MemoryVersions(workspace, entity, project_id)
            matches = versions.search(query)
            cap = int(limit or search_limit(config=workspace.context.config))
            rows = matches[:cap]
            return {
                "project_id": versions.project_id,
                "query": query,
                "rows": rows,
                "ids": [row["object_id"] for row in rows],
                "n": len(rows),
                "limit": cap,
                "total": len(matches),
                "truncated": len(matches) > len(rows),
            }
        except Exception as exc:
            mcp_raise(exc)

    return memstate_search
