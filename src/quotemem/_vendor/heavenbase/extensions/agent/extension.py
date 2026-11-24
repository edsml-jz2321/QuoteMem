"""Built-in optional agent extension definition."""

from __future__ import annotations

__all__ = ["AGENT_EXTENSION_ID", "agent_entities", "agent_entity_ids"]

from heavenbase.extensions.ids import AGENT_EXTENSION_ID
from heavenbase.modules import ModuleResolver
from heavenbase.utils import List


def agent_entities(*, resolver: ModuleResolver | None = None) -> tuple[type, ...]:
    """Return persistent entity classes owned by the agent extension.

    Args:
        resolver: Context-owned module resolver. The default Context is
            selected only for an unbound standalone call.

    Returns:
        tuple[type, ...]: Agent entity classes in registration order.
    """
    from heavenbase.entity.registry import load_entity_type

    return tuple(load_entity_type(identifier, resolver=resolver) for identifier in ("agent-message", "agent-session", "agent-skill", "agent-agent"))


def agent_entity_ids(*, resolver: ModuleResolver | None = None) -> List[str]:
    """Return entity identifiers declared by the agent extension.

    Args:
        resolver: Context-owned module resolver. The default Context is
            selected only for an unbound standalone call.

    Returns:
        List[str]: Agent entity identifiers.
    """
    return [entity.schema().entity_id for entity in agent_entities(resolver=resolver)]
