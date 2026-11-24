"""Built-in optional prompt extension definition."""

from __future__ import annotations

__all__ = [
    "PROMPT_EXTENSION_ID",
    "prompt_entities",
    "prompt_entity_ids",
]

from heavenbase.extensions.ids import PROMPT_EXTENSION_ID
from heavenbase.modules import ModuleResolver
from heavenbase.utils import List


def prompt_entities(*, resolver: ModuleResolver | None = None) -> tuple[type, ...]:
    """Return persistent entity classes owned by the prompt extension.

    Args:
        resolver: Context-owned module resolver. The default Context is
            selected only for an unbound standalone call.

    Returns:
        tuple[type, ...]: Prompt entity classes in registration order.
    """
    from heavenbase.entity.registry import load_entity_type

    return (
        load_entity_type("sys-prompt", resolver=resolver),
        load_entity_type("sys-translation", resolver=resolver),
    )


def prompt_entity_ids(*, resolver: ModuleResolver | None = None) -> List[str]:
    """Return entity identifiers declared by the prompt extension.

    Args:
        resolver: Context-owned module resolver. The default Context is
            selected only for an unbound standalone call.

    Returns:
        List[str]: Prompt entity identifiers.
    """
    return [entity.schema().entity_id for entity in prompt_entities(resolver=resolver)]
