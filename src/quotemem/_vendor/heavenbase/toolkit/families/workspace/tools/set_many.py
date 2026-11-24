"""MCP tool for setting multiple rows."""

from __future__ import annotations

__all__ = ["build_tool"]

from heavenbase.utils import Any, Dict, List, Optional

from ..context import WorkspaceToolContext
from ..errors import mcp_raise


def build_tool(context: WorkspaceToolContext) -> Any:
    """Build the ``set_many`` tool for one workspace context.

    Args:
        context (WorkspaceToolContext): Shared workspace and profile state for the tool factory.

    Returns:
        Any: MCP-callable ``set_many`` function bound to ``context``.
    """

    def set_many(rows: List[Dict[str, Any]], entity: Optional[str] = None, overwrite: bool = True) -> Dict[str, Any]:
        """Set multiple rows using their embedded object ids.

        Args:
            rows: Row payloads containing ``object_id`` or enough data to compute it.
            entity: Optional entity id, class name, or display name for disambiguation.
            overwrite: Whether values in ``rows`` may replace existing values.

        Returns:
            Written object ids, stored rows, and row count.

        Raises:
            RuntimeError: If the bulk write fails; the message is redacted for MCP use.
        """
        try:
            with context.workspace._lock:
                resolved = context.require_entity(entity)
                if resolved is not None or context.entity_scope == "*":
                    object_ids = context.workspace._crud.set_many(
                        rows,
                        entity=resolved,
                        overwrite=overwrite,
                        allowed_entities=context.allowed_entities,
                    )
                    stored = [context.workspace.get(object_id, entity=resolved) for object_id in object_ids]
                else:
                    resolved_rows = []
                    target_entities = []
                    for row in rows:
                        target_entity = context.entity_for_many_patch(row, None)
                        if target_entity is None:
                            raise ValueError(
                                f"set_many with omitted entity could not resolve a target inside MCP profile {context.profile.name!r} entity scope"
                            )
                        scoped_row = dict(context.strip_row_entity(row))
                        scoped_row["entity"] = target_entity
                        resolved_rows.append(scoped_row)
                        target_entities.append(target_entity)
                    object_ids = context.workspace._crud.set_many(
                        resolved_rows,
                        overwrite=overwrite,
                        allowed_entities=context.allowed_entities,
                    )
                    stored = [context.workspace.get(object_id, entity=target_entity) for object_id, target_entity in zip(object_ids, target_entities)]
            return {"object_ids": object_ids, "rows": stored, "n": len(object_ids)}
        except Exception as exc:
            mcp_raise(exc)

    return set_many
