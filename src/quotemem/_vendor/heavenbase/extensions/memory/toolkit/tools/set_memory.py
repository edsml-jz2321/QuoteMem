"""MCP tool for patching one memory note."""

from __future__ import annotations

__all__ = ["build_tool"]

from heavenbase.utils import Any, Dict, List, Optional

from ...extension import ensure_memory_entity
from .._utils import now
from ..errors import mcp_raise


def build_tool(workspace: Any) -> Any:
    """Build the ``set_memory`` tool for one workspace.

    Args:
        workspace (Any): Live workspace bound to memory MCP tools.

    Returns:
        Any: Callable MCP tool function for ``set_memory``.
    """

    def set_memory(
        key: str,
        text: Optional[str] = None,
        title: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        source: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Patch one memory note by key."""
        try:
            entity = ensure_memory_entity(workspace)
            patch: Dict[str, Any] = {"object_id": key, "updated_at": now()}
            if text is not None:
                patch["text"] = text
            if title is not None:
                patch["title"] = title
            if tags is not None:
                patch["tags"] = list(tags)
            if metadata is not None:
                patch["metadata"] = dict(metadata)
            if source is not None:
                patch["source"] = source
            object_id = workspace.set(patch, entity=entity)
            return {"object_id": object_id, "row": workspace.get(object_id, entity=entity)}
        except Exception as exc:
            mcp_raise(exc)

    return set_memory
