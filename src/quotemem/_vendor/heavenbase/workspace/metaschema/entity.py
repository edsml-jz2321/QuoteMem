"""Engine-facing system catalog entity."""

from __future__ import annotations

__all__ = [
    "MetaSchema",
]

from heavenbase.entity.base import Entity
from heavenbase.entity.fields import field
from heavenbase.types import Boolean, Identifier, Json, MediumText, ShortText

_f = field


class MetaSchema(Entity):
    """Queryable metadata catalog for workspace structure.

    MetaSchema rows describe workspace internals: registered entity types,
    fields, storage placements, backends, backend capabilities, and other engine-facing
    records. Agents should use MetaSchema to understand structure, then use
    Catalog to discover concrete objects.

    Attributes:
        object_id: Primary key for the metadata row.
        kind: Metadata category, such as `entity`, `field`, `storage`, or `backend`.
        subject_id: Logical object described by this row.
        name: Human-readable row name.
        dtype: Logical type identifier for field rows.
        desc: Searchable description.
        tags: Searchable tags.
        backend: Physical backend instance name.
        type: Backend provider type or class identifier.
        strategy: Storage or handler strategy.
        source: Storage binding source or mode.
        target: Optional link to another object.
        meta: Structured metadata payload.
        active: Whether this metadata row is currently active.
        ws: Owning workspace ID.
        field: Field name for field and storage rows.
    """

    identifier = "sys-metaschema"
    __tags__ = ["system"]
    object_id = _f(Identifier).pk().describe("Meta row object_id")
    kind = _f(ShortText).describe("entity, field, structure, workspace, backend, storage, capability, or extension")
    subject_id = _f(ShortText).describe("Logical object this row describes")
    name = _f(ShortText).default("").describe("Human readable name")
    dtype = _f(ShortText).default("").describe("Logical type identifier for field rows")
    desc = _f(MediumText).default("").describe("Searchable description")
    tags = _f(Json).default([]).describe("Searchable tags")
    backend = _f(ShortText).default("").describe("Physical backend instance name")
    type = _f(ShortText).default("").describe("Backend provider type or class identifier")
    strategy = _f(ShortText).default("").describe("Storage strategy")
    source = _f(ShortText).default("").describe("Storage binding source or mode")
    target = _f(ShortText).default("").describe("Optional ID link to another object")
    meta = _f(Json).default({}).describe("Structured metadata payload")
    active = _f(Boolean).default(True).describe("Whether this metadata row is active")
    ws = _f(ShortText).default("").describe("Owning workspace")
    field = _f(ShortText).default("").describe("Field name for field and storage rows")
