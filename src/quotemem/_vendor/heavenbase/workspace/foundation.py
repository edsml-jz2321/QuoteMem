"""Canonical workspace foundation Entity contracts."""

from __future__ import annotations

from heavenbase.utils import Any

from .catalog.entity import Catalog
from .metaschema.entity import MetaSchema

__all__ = [
    "FIXED_ENTITY_IDS",
    "FIXED_ENTITY_TYPES",
    "fixed_entity_type",
]

FIXED_ENTITY_TYPES: tuple[type, ...] = (Catalog, MetaSchema)
FIXED_ENTITY_IDS = frozenset(entity.schema().entity_id for entity in FIXED_ENTITY_TYPES)
_FIXED_BY_ID = {entity.schema().entity_id: entity for entity in FIXED_ENTITY_TYPES}


def fixed_entity_type(entity_id: str) -> type:
    """Return the canonical fixed Entity class for ``entity_id``.

    Args:
        entity_id: Fixed workspace Entity identifier.

    Returns:
        Canonical Catalog or MetaSchema class.

    Raises:
        KeyError: If ``entity_id`` is not a fixed workspace Entity.
    """
    return _FIXED_BY_ID[str(entity_id)]


def bootstrap_fixed_entities(workspace: Any) -> None:
    """Register the exact workspace-kernel Entity classes in dependency order.

    Args:
        workspace (Any): Workspace receiving fixed Catalog and MetaSchema
            entities.

    Returns:
        None: This function does not return a value.
    """
    for entity in FIXED_ENTITY_TYPES:
        workspace._schema._register_fixed(entity)
    for entity in FIXED_ENTITY_TYPES:
        workspace._metaschema.publish_schema_rows(entity)
