"""MCP tool for storing one memory note."""

from __future__ import annotations

__all__ = ["build_tool"]

from heavenbase.utils import Any, Dict, List, Optional, hash_id

from ...extension import ensure_memory_entity
from .._utils import default_source, now
from ..errors import mcp_raise


def build_tool(workspace: Any) -> Any:
    """Build the ``remember`` tool for one workspace.

    Args:
        workspace (Any): Live workspace bound to memory MCP tools.

    Returns:
        Any: Callable MCP tool function for ``remember``.
    """

    def remember(
        text: str,
        key: Optional[str] = None,
        title: str = "",
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        source: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Store one memory note in the workspace memory entity."""
        try:
            entity = ensure_memory_entity(workspace)
            timestamp = now()
            if key:
                object_id = str(key)
            else:
                hash_length = int(workspace.context.config.get("heavenbase.entity.id.hash_length", default=42))
                object_id = hash_id(workspace.id, entity, text, timestamp, length=hash_length)
            row = {
                "object_id": object_id,
                "text": text,
                "title": title,
                "tags": list(tags or []),
                "metadata": dict(metadata or {}),
                "source": source or default_source(config=workspace.context.config),
                "created_at": timestamp,
                "updated_at": timestamp,
            }
            workspace.upsert(entity, row)
            return {"object_id": object_id, "row": workspace.get(object_id, entity=entity)}
        except Exception as exc:
            mcp_raise(exc)

    return remember
