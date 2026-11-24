"""MCP tool for writing a short versioned memstate value."""

from __future__ import annotations

__all__ = ["build_tool"]

from heavenbase.utils import Any, Dict, Mapping, Optional, dumps_jstr

from ...extension import ensure_memory_entity
from ...versions import MemoryVersions
from ..errors import mcp_raise


def build_tool(workspace: Any) -> Any:
    """Build the ``memstate_set`` tool for one workspace.

    Args:
        workspace (Any): Live workspace bound to memory MCP tools.

    Returns:
        Any: Callable MCP tool function for ``memstate_set``.
    """

    def memstate_set(
        project_id: str,
        keypath: str,
        value: Any,
        source: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Store a short versioned memstate value."""
        try:
            entity = ensure_memory_entity(workspace)
            versions = MemoryVersions(workspace, entity, project_id)
            return versions.append(
                keypath,
                text=dumps_jstr(value, compact=True),
                title=versions.title(keypath),
                metadata=metadata,
                source=source,
            )
        except Exception as exc:
            mcp_raise(exc)

    return memstate_set
