"""Optional persistent memory extension."""

from __future__ import annotations

from heavenbase._lazy import _attach_registry_exports
from heavenbase.extensions.ids import MEMORY_EXTENSION_ID

__all__ = [
    "MEMORY_EXTENSION_ID",
    "MemoryAPI",
    "MemoryNote",
    "MemoryVersions",
    "ensure_memory_entity",
    "memory_entities",
    "memory_entity_ids",
]

_REGISTRY_ENTITY_EXPORTS = {
    "MemoryNote": "memory-note",
}

_REGISTRY_SIBLING_EXPORTS = {
    "MemoryAPI": ("memory-note", "api", "MemoryAPI"),
    "MemoryVersions": ("memory-note", "versions", "MemoryVersions"),
    "ensure_memory_entity": ("memory-note", "extension", "ensure_memory_entity"),
    "memory_entities": ("memory-note", "extension", "memory_entities"),
    "memory_entity_ids": ("memory-note", "extension", "memory_entity_ids"),
}

__getattr__, __dir__ = _attach_registry_exports(
    globals(),
    __all__,
    entities=_REGISTRY_ENTITY_EXPORTS,
    siblings=_REGISTRY_SIBLING_EXPORTS,
)
