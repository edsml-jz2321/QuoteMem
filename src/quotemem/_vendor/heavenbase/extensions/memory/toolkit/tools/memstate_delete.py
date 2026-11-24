"""MCP tool for soft-deleting a memstate keypath."""

from __future__ import annotations

__all__ = ["build_tool"]

from heavenbase.utils import Any, Dict, Optional

from ...extension import ensure_memory_entity
from ...versions import MemoryVersions
from ..errors import mcp_raise


def build_tool(workspace: Any) -> Any:
    """Build the ``memstate_delete`` tool for one workspace.

    Args:
        workspace (Any): Live workspace bound to memory MCP tools.

    Returns:
        Any: Callable MCP tool function for ``memstate_delete``.
    """

    def memstate_delete(project_id: str, keypath: str, source: Optional[str] = None) -> Dict[str, Any]:
        """Write a soft-delete version for one memstate keypath."""
        try:
            entity = ensure_memory_entity(workspace)
            versions = MemoryVersions(workspace, entity, project_id)
            full_keypath = versions.keypath(keypath)
            return versions.append(
                full_keypath,
                text="",
                title=f"Deleted {versions.title(full_keypath)}",
                source=source,
                deleted=True,
            )
        except Exception as exc:
            mcp_raise(exc)

    return memstate_delete
