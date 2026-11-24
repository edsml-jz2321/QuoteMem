"""MCP tool for reading current memstate rows."""

from __future__ import annotations

__all__ = ["build_tool"]

from heavenbase.utils import Any, Dict, Optional

from ...extension import ensure_memory_entity
from ...versions import MemoryVersions
from .._utils import list_limit
from ..errors import mcp_raise


def build_tool(workspace: Any) -> Any:
    """Build the ``memstate_get`` tool for one workspace.

    Args:
        workspace (Any): Live workspace bound to memory MCP tools.

    Returns:
        Any: Callable MCP tool function for ``memstate_get``.
    """

    def memstate_get(project_id: str, keypath: str = "", include_deleted: bool = False, limit: Optional[int] = None) -> Dict[str, Any]:
        """Read current memstate rows below a project keypath, bounded by a limit.

        Args:
            project_id (str): Project namespace. Empty values resolve to `default`.
            keypath (str): Relative or full keypath prefix. Empty selects the project.
            include_deleted (bool): Whether to retain latest soft-delete tombstones.
            limit (Optional[int]): Maximum rows to return. Defaults to
                ``heavenbase.mcp.memory.list_limit`` (20), so an unset limit
                still bounds the answer.

        Returns:
            Dict[str, Any]: Payload with ``project_id``, ``keypath``, ``rows``,
            ``ids``, ``n``, the applied ``limit``, ``total``, and ``truncated``.
            ``n`` counts returned rows and ``total`` counts every current row
            below the keypath, so a caller never mistakes a page for the whole
            subtree; report ``total`` as the inventory size.

        Raises:
            RuntimeError: If the read fails, with the cause redacted.
        """
        try:
            entity = ensure_memory_entity(workspace)
            versions = MemoryVersions(workspace, entity, project_id)
            prefix = versions.keypath(keypath)
            current = versions.current(keypath, include_deleted=include_deleted)
            cap = int(limit or list_limit(config=workspace.context.config))
            rows = current[:cap]
            return {
                "project_id": versions.project_id,
                "keypath": prefix,
                "rows": rows,
                "ids": [row["object_id"] for row in rows],
                "n": len(rows),
                "limit": cap,
                "total": len(current),
                "truncated": len(current) > len(rows),
            }
        except Exception as exc:
            mcp_raise(exc)

    return memstate_get
