"""Persistent memory note entity."""

from __future__ import annotations

__all__ = ["MEMORY_ENTITY_ID", "MemoryNote", "memory_entity_id"]

from heavenbase.entity import Entity, field
from heavenbase.types import Array, Boolean, Identifier, Integer, Json, LongText, ShortText, Timestamp

MEMORY_ENTITY_ID = "memory-note"


def memory_entity_id() -> str:
    """Return the registered memory entity identity.

    Args:
        None.

    Returns:
        str: Stable module-declared entity identifier.
    """
    return MEMORY_ENTITY_ID


class MemoryNote(Entity):
    """Small agent memory note used by the memory MCP profile."""

    identifier = MEMORY_ENTITY_ID

    text = field(LongText).desc("Memory body")
    title = field(ShortText).default("").desc("Optional short title")
    tags = field(Array[ShortText]).default([]).desc("Optional memory tags")
    metadata = field(Json).default({}).desc("Optional structured metadata")
    source = field(ShortText).default("agent").desc("Producer or agent that wrote the memory")
    project_id = field(ShortText).default("default").desc("Memstate-style project id")
    keypath = field(ShortText).default("").desc("Hierarchical project keypath")
    version = field(Integer).default(1).desc("Monotonic version for this keypath")
    previous_id = field(ShortText).default("").desc("Previous version object id")
    deleted = field(Boolean).default(False).desc("Soft-delete tombstone flag")
    created_at = field(Timestamp).desc("Creation time as UTC epoch seconds")
    updated_at = field(Timestamp).desc("Last update time as UTC epoch seconds")
