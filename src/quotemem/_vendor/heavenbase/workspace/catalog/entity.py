"""User-facing instance catalog entity."""

from __future__ import annotations

__all__ = [
    "Catalog",
]

from heavenbase.entity.base import Entity
from heavenbase.entity.fields import field
from heavenbase.types import Array, Boolean, Duration, Float, Identifier, Json, LongText, MediumText, ShortText, Timestamp
from heavenbase.types.timestamp import TIMESTAMP_DEFAULT_MAX_MS

_f = field

#: Longest target ``object_id`` a Catalog row holds. Entity ids may be declared longer than the default
#: ``Identifier`` limit of 63, so ``target_id`` takes the 255 characters of the indexed ``ShortText``
#: Catalog columns; an entity with longer ids stores them with ``catalog="none"``.
_TARGET_ID_MAX_CHARS = 255


class Catalog(Entity):
    """Queryable directory for concrete entity instances.

    Catalog rows are built automatically when non-system entity rows are
    upserted. They let users and agents discover objects through universal
    AI-facing fields before hydrating the typed row.

    Field groups follow the UKF layout: Identity, Payload, Searchable,
    Governance, Lineage, Usage, and Reference. ``desc`` is the short
    searchable blurb; ``summary`` is the long payload text.

    Expiration naming:

    - Policy (projectable): ``expires_at`` uses :attr:`~heavenbase.types.timestamp.Timestamp.default_max`
      as disabled; ``expires_after`` and ``stale_after`` use ``0`` as disabled.
    - Materialized (auto on write, search fields): ``expires_compute``,
      ``stale_compute``. When a policy is off, the materialized field is set
      to :attr:`~heavenbase.types.timestamp.Timestamp.default_max`.

    Notes:
        ``Catalog.target_id`` is not the Catalog row primary key. Use it with
        ``Catalog.target_entity`` to hydrate the target object through
        ``workspace.get(target_id, entity=target_entity)``. It holds target
        ids of up to 255 characters, so an entity whose ``object_id`` domain
        allows longer ids stores those with ``catalog="none"``; the Catalog
        row primary key is hashed when its readable form is longer.

        For discovery queries, prefer ``Query.available()`` in Python or
        the JSON ``available`` / ``$available`` query options instead of hand-rolling
        governance predicates.
    """

    identifier = "sys-catalog"
    __tags__ = ["system", "catalog"]

    # Identity
    object_id = _f(Identifier).pk().describe("Catalog row object_id")
    target_id = _f(Identifier(max_chars=_TARGET_ID_MAX_CHARS)).describe("Target object_id value")
    target_entity = _f(ShortText).describe("Target entity identifier")
    name = _f(ShortText).default("").describe("Projected human readable name")
    version = _f(ShortText).default("").describe("Projected knowledge version")
    type = _f(ShortText).default("").describe("Projected knowledge type")

    # Payload
    summary = _f(LongText).default("").describe("Projected long payload description")
    instructions = _f(LongText).default("").describe("Projected LLM instructions")
    examples = _f(Json).default([]).describe("Projected examples")
    artifacts = _f(Json).default([]).describe("Projected artifacts")
    confidence = _f(Float).default(1.0).describe("Projected confidence score")

    # Searchable
    desc = _f(MediumText).default("").describe("Projected short searchable description")
    keywords = _f(Array[ShortText]).default([]).describe("Projected searchable keywords")
    tags = _f(Json).default([]).describe("Projected searchable tags")
    priority = _f(Float).default(0.0).describe("Projected retrieval priority")

    # Governance
    ws = _f(ShortText).default("").describe("Owning workspace")
    active = _f(Boolean).default(True).describe("Whether this catalog row participates in directory operations")
    visible = _f(Boolean).default(True).describe("Whether this catalog row is visible in discovery surfaces")
    created_at = _f(Timestamp).default(0).describe("Catalog row creation timestamp")
    updated_at = _f(Timestamp).default(0).describe("Catalog row last update timestamp")
    expires_at = _f(Timestamp).default(TIMESTAMP_DEFAULT_MAX_MS).describe("Absolute expiry policy in ms; default_max means no absolute ceiling")
    expires_after = _f(Duration).default(0).describe("Creation TTL policy in ms; 0 means no creation TTL")
    stale_after = _f(Duration).default(0).describe("Freshness TTL policy in ms after updated_at; 0 means never stale")
    expires_compute = _f(Timestamp).default(TIMESTAMP_DEFAULT_MAX_MS).describe("Materialized hard-expiry deadline for search")
    stale_compute = _f(Timestamp).default(TIMESTAMP_DEFAULT_MAX_MS).describe("Materialized freshness deadline for search")
    creator = _f(ShortText).default("").describe("Projected creator")
    owner = _f(ShortText).default("").describe("Projected owner")
    auth = _f(Json).default({}).describe("Projected authentication metadata")
    metadata = _f(Json).default({}).describe("Projected governance metadata")
    note = _f(MediumText).default("").describe("Projected governance note or status comment")

    # Lineage
    source = _f(LongText).default("").describe("Projected source")
    parent_id = _f(ShortText).default("").describe("Projected parent object_id")
    history = _f(Json).default([]).describe("Projected lineage history")
    transforms = _f(Json).default([]).describe("Projected transforms")

    # Usage
    statistics = _f(Json).default({}).describe("Projected usage statistics")
    feedback = _f(Json).default([]).describe("Projected feedback")
    annotations = _f(Json).default([]).describe("Projected annotations")

    # Reference
    related = _f(Json).default([]).describe("Projected related knowledge")
    automations = _f(Json).default([]).describe("Projected related automations")
