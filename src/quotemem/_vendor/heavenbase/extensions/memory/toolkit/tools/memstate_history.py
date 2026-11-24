"""MCP tool for reading memstate version history."""

from __future__ import annotations

__all__ = ["build_tool"]

from heavenbase.utils import Any, Dict

from ...extension import ensure_memory_entity
from ...versions import MemoryVersions
from ..errors import mcp_raise


def build_tool(workspace: Any) -> Any:
    """Build the ``memstate_history`` tool for one workspace.

    Args:
        workspace (Any): Live workspace bound to memory MCP tools.

    Returns:
        Any: Callable MCP tool function for ``memstate_history``.
    """

    def memstate_history(project_id: str, keypath: str) -> Dict[str, Any]:
        """Read the full version history for one memstate keypath.

        Args:
            project_id (str): Project namespace. Empty values resolve to `default`.
            keypath (str): Relative or full memory keypath.

        Returns:
            Dict[str, Any]: Payload with ``project_id``, ``keypath``, ``rows``,
            ``ids``, and ``n``. The chain is read by a scoped query with no
            limit, so ``n`` is the complete version count for this keypath and
            no ``total`` is needed.

        Raises:
            RuntimeError: If the read fails, with the cause redacted.
        """
        try:
            entity = ensure_memory_entity(workspace)
            versions = MemoryVersions(workspace, entity, project_id)
            full_keypath = versions.keypath(keypath)
            rows = versions.history(full_keypath)
            return {
                "project_id": versions.project_id,
                "keypath": full_keypath,
                "rows": rows,
                "ids": [row["object_id"] for row in rows],
                "n": len(rows),
            }
        except Exception as exc:
            mcp_raise(exc)

    return memstate_history
