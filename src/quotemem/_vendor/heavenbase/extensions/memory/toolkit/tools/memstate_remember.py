"""MCP tool for writing a versioned memstate markdown entry."""

from __future__ import annotations

__all__ = ["build_tool"]

from heavenbase.utils import Any, Dict, List, Optional

from ...extension import ensure_memory_entity
from ...versions import MemoryVersions
from ..errors import mcp_raise


def build_tool(workspace: Any) -> Any:
    """Build the ``memstate_remember`` tool for one workspace.

    Args:
        workspace (Any): Live workspace bound to memory MCP tools.

    Returns:
        Any: Callable MCP tool function for ``memstate_remember``.
    """

    def memstate_remember(
        project_id: str,
        content: str,
        keypath: Optional[str] = None,
        title: str = "",
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        source: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Store a versioned memstate markdown entry."""
        try:
            entity = ensure_memory_entity(workspace)
            versions = MemoryVersions(workspace, entity, project_id)
            full_keypath = versions.keypath(keypath or versions.derive_keypath(content, title))
            return versions.append(
                full_keypath,
                text=content,
                title=title or versions.title(full_keypath),
                tags=tags,
                metadata=metadata,
                source=source,
            )
        except Exception as exc:
            mcp_raise(exc)

    return memstate_remember
