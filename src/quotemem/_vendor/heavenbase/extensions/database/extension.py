"""Built-in optional database extension definition."""

from __future__ import annotations

__all__ = [
    "DATABASE_EXTENSION_ID",
    "database_entities",
    "database_entity_ids",
]

from heavenbase.extensions.ids import DATABASE_EXTENSION_ID
from heavenbase.modules import ModuleResolver
from heavenbase.utils import List


def database_entities(*, resolver: ModuleResolver | None = None) -> tuple[type, ...]:
    """Return persistent entity classes owned by the database extension.

    Args:
        resolver: Context-owned module resolver. The default Context is
            selected only for an unbound standalone call.

    Returns:
        tuple[type, ...]: Database entity classes in registration order.
    """
    from heavenbase.entity.registry import load_entity_type

    return tuple(load_entity_type(identifier, resolver=resolver) for identifier in database_entity_ids(resolver=resolver))


def database_entity_ids(*, resolver: ModuleResolver | None = None) -> List[str]:
    """Return entity identifiers declared by the database extension.

    Args:
        resolver: Context-owned module resolver. Accepted for the same explicit
            authority contract as ``database_entities``; identifiers are inert
            module metadata and require no materialization.

    Returns:
        List[str]: Database entity identifiers.
    """
    from .entities import DATABASE_ENTITY_IDS

    return list(DATABASE_ENTITY_IDS)
