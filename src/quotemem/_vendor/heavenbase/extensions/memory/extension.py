"""Built-in optional memory extension definition."""

from __future__ import annotations

__all__ = [
    "MEMORY_EXTENSION_ID",
    "ensure_memory_entity",
    "memory_entities",
    "memory_entity_ids",
]

from heavenbase.extensions.ids import MEMORY_EXTENSION_ID
from heavenbase.modules import ModuleResolver
from heavenbase.utils import Any, Iterable, List, Optional


def memory_entities(*, resolver: ModuleResolver | None = None) -> tuple[type, ...]:
    """Return persistent entity classes owned by the memory extension.

    Args:
        resolver: Context-owned module resolver. The default Context is
            selected only for an unbound standalone call.

    Returns:
        tuple[type, ...]: Memory entity classes in registration order.
    """
    from heavenbase.entity.registry import load_entity_type

    return (load_entity_type("memory-note", resolver=resolver),)


def memory_entity_ids(*, resolver: ModuleResolver | None = None) -> List[str]:
    """Return entity identifiers declared by the memory extension.

    Args:
        resolver: Context-owned module resolver. Accepted for the same explicit
            authority contract as ``memory_entities``; identifiers are inert
            module metadata and require no materialization.

    Returns:
        List[str]: Stable module-declared entity identifiers.
    """
    from .note import memory_entity_id

    return [memory_entity_id()]


def _register_memory_entities(workspace: Any) -> Optional[Iterable[Any]]:
    """Register the memory module's declared entity in a workspace.

    Args:
        workspace: Workspace receiving the optional extension entity.

    Returns:
        Optional[Iterable[Any]]: Entity classes still requiring registration.
    """
    resolver = workspace.module_resolver
    if memory_entity_ids(resolver=resolver)[0] in workspace.entities:
        return ()
    return memory_entities(resolver=resolver)


def ensure_memory_entity(workspace: Any) -> str:
    """Enable memory when needed and return its registered entity identifier.

    Args:
        workspace (Any): Workspace that should own the memory entity.

    Returns:
        str: Stable module-declared memory entity identifier.
    """
    from .note import memory_entity_id

    if MEMORY_EXTENSION_ID not in workspace.extensions():
        workspace.enable_extension(MEMORY_EXTENSION_ID)
    return memory_entity_id()
