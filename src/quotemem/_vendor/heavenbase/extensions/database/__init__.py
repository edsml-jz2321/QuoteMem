"""Optional database extension package."""

from __future__ import annotations

from heavenbase._lazy import _attach_registry_exports
from heavenbase.extensions.ids import DATABASE_EXTENSION_ID

__all__ = [
    "DATABASE_EXTENSION_ID",
    "DatabaseAPI",
    "DbColumn",
    "DbDatabase",
    "DbEnum",
    "DbPredicate",
    "DbRule",
    "DbSnippet",
    "DbTable",
    "database_entities",
    "database_entity_ids",
    "ingest_database",
]

_REGISTRY_ENTITY_EXPORTS = {
    "DbColumn": "db-column",
    "DbDatabase": "db-database",
    "DbEnum": "db-enum",
    "DbPredicate": "db-predicate",
    "DbRule": "db-rule",
    "DbSnippet": "db-snippet",
    "DbTable": "db-table",
}

_REGISTRY_SIBLING_EXPORTS = {
    "DatabaseAPI": ("db-database", "api", "DatabaseAPI"),
    "database_entities": ("db-database", "extension", "database_entities"),
    "database_entity_ids": ("db-database", "extension", "database_entity_ids"),
    "ingest_database": ("db-database", "ingest", "ingest_database"),
}

__getattr__, __dir__ = _attach_registry_exports(
    globals(),
    __all__,
    entities=_REGISTRY_ENTITY_EXPORTS,
    siblings=_REGISTRY_SIBLING_EXPORTS,
)
