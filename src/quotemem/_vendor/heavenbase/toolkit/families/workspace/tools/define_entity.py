"""MCP tool for registering workspace entity schemas."""

from __future__ import annotations

__all__ = ["build_tool"]

from heavenbase.utils import Any, Dict, Mapping

from ..context import WorkspaceToolContext
from ..errors import mcp_raise
from ..schema import normalize_entity_spec


def build_tool(context: WorkspaceToolContext) -> Any:
    """Build the ``define_entity`` tool for one workspace context.

    Args:
        context (WorkspaceToolContext): Shared workspace and profile state for the tool factory.

    Returns:
        Any: MCP-callable ``define_entity`` function bound to ``context``.
    """

    def define_entity(entity: Dict[str, Any], replace: bool = False) -> Dict[str, Any]:
        """Register an entity schema in the workspace.

        The normalized definition itself is passed to ``HeavenBase.register``, so its ``catalog`` and ``gram`` options
        apply, and an enabled extension's exported definition is accepted as the no-op ``register`` makes it.

        Args:
            entity: Entity schema object accepted by ``HeavenBase.register``.
            replace: Whether to replace an existing entity with the same id.

        Returns:
            Entity schema, storage plan, derives, and query capabilities.

        Raises:
            RuntimeError: If registration fails; the message is redacted for MCP use.
        """
        from heavenbase.workspace.schema.service import _definition_entity_id

        try:
            normalized = normalize_entity_spec(entity)
            context.require_entity(_definition_entity_id(normalized) if isinstance(normalized, Mapping) else normalized)
            entity_cls = context.workspace.register(normalized, allow_replace=replace)
            return context.entity_dict(entity_cls)
        except Exception as exc:
            mcp_raise(exc)

    return define_entity
