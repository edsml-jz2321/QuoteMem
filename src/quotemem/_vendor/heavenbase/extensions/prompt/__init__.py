"""Default-loaded prompt extension package."""

from __future__ import annotations

from heavenbase._lazy import _attach_registry_exports
from heavenbase.extensions.ids import PROMPT_EXTENSION_ID

__all__ = [
    "PROMPT_EXTENSION_ID",
    "Prompt",
    "Translation",
    "check_prompt_name",
    "ensure_prompt_entities",
    "prompt_entities",
    "prompt_entity_ids",
    "prompt_row_id",
    "translation_row_id",
]

_REGISTRY_ENTITY_EXPORTS = {
    "Prompt": "sys-prompt",
    "Translation": "sys-translation",
}

_REGISTRY_SIBLING_EXPORTS = {
    "check_prompt_name": ("sys-prompt", "name", "check_prompt_name"),
    "ensure_prompt_entities": ("sys-prompt", "prompt", "ensure_prompt_entities"),
    "prompt_entities": ("sys-prompt", "extension", "prompt_entities"),
    "prompt_entity_ids": ("sys-prompt", "extension", "prompt_entity_ids"),
    "prompt_row_id": ("sys-prompt", "name", "prompt_row_id"),
    "translation_row_id": ("sys-prompt", "name", "translation_row_id"),
}

__getattr__, __dir__ = _attach_registry_exports(
    globals(),
    __all__,
    entities=_REGISTRY_ENTITY_EXPORTS,
    siblings=_REGISTRY_SIBLING_EXPORTS,
)
