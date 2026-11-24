"""Database extension schema entities."""

from __future__ import annotations

__all__ = [
    "DATABASE_ENTITY_IDS",
    "DbColumn",
    "DbDatabase",
    "DbEnum",
    "DbPredicate",
    "DbRule",
    "DbSnippet",
    "DbTable",
]

from heavenbase.entity import Entity, field
from heavenbase.types import Array, Boolean, Identifier, Integer, Json, LongText, MediumText, ShortText, Timestamp
from heavenbase.utils import Any, Mapping

DATABASE_ENTITY_IDS = (
    "db-database",
    "db-table",
    "db-column",
    "db-enum",
    "db-snippet",
    "db-predicate",
    "db-rule",
)

_SEMANTIC_FIELDS = (
    "name",
    "title",
    "database_name",
    "qualified_name",
    "table_name",
    "column_name",
    "source",
    "expression",
    "dtype",
    "semantic_type",
    "relation_kind",
    "desc",
    "comment",
    "profile",
    "card_text",
    "scope",
    "tables",
    "synonyms",
    "tags",
    "triggers",
    "example",
    "values",
)


def _text_parts(value: Any) -> list[str]:
    """Return non-empty semantic text parts from one JSON-like value."""
    if value is None:
        return []
    if isinstance(value, Mapping):
        return [part for key, item in value.items() for part in (*_text_parts(key), *_text_parts(item))]
    if isinstance(value, (list, tuple, set)):
        return [part for item in value for part in _text_parts(item)]
    text = " ".join(str(value).split())
    return [text] if text else []


def _semantic_text(row: Mapping[str, Any]) -> str:
    """Build one editable-row semantic projection without choosing an embedding provider."""
    parts = [part for name in _SEMANTIC_FIELDS for part in _text_parts(row.get(name))]
    return "\n".join(dict.fromkeys(parts))[: LongText.default_max_chars]


def _catalog_name(row: Mapping[str, Any]) -> str:
    """Return the first domain name suitable for a Catalog card."""
    for name in ("name", "title", "qualified_name", "column_name", "source", "expression"):
        value = " ".join(str(row.get(name) or "").split())
        if value:
            return value[: ShortText.default_max_chars]
    return ""


def _catalog_desc(row: Mapping[str, Any]) -> str:
    """Return the best short description suitable for a Catalog card."""
    for name in ("desc", "card_text", "comment", "profile", "dsn_summary"):
        value = " ".join(str(row.get(name) or "").split())
        if value:
            return value[: MediumText.default_max_chars]
    return ""


def _catalog_keywords(row: Mapping[str, Any]) -> list[str]:
    """Return normalized row synonyms for the universal Catalog search surface."""
    return list(dict.fromkeys(str(item).strip() for item in (row.get("synonyms") or []) if str(item).strip()))


def _catalog_statistics(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return detached imported statistics for the universal Catalog projection."""
    return dict(row.get("statistics") or {})


def _database_type() -> str:
    return "database"


def _table_type() -> str:
    return "table"


def _column_type() -> str:
    return "column"


def _enum_type() -> str:
    return "enum"


def _snippet_type() -> str:
    return "snippet"


def _predicate_type() -> str:
    return "predicate"


def _rule_type() -> str:
    return "rule"


def _rule_visible(enabled: bool) -> bool:
    """Return whether an enabled rule should appear in Catalog search."""
    return bool(enabled)


def _catalog_derives(type_fn: Any) -> list[Any]:
    """Return common Catalog derives for one database entity kind."""
    return [
        Entity.derive("catalog.name").compute(_catalog_name),
        Entity.derive("catalog.desc").compute(_catalog_desc),
        Entity.derive("catalog.summary").compute(_semantic_text),
        Entity.derive("catalog.keywords").compute(_catalog_keywords),
        Entity.derive("catalog.statistics").compute(_catalog_statistics),
        Entity.derive("catalog.type").compute(type_fn),
    ]


class DbDatabase(Entity):
    """Ingested external database summary with redacted connection metadata."""

    identifier = "db-database"
    __derive__ = _catalog_derives(_database_type)

    name = field(ShortText).desc("Human-readable database label")
    dsn_summary = field(MediumText).desc("Redacted connection summary safe for storage and MCP output")
    dialect = field(ShortText).default("").desc("SQL dialect reported by SQLAlchemy")
    backend_ref = field(ShortText).default("").desc("Workspace SQL backend name when ingested from a backend")
    source = field(Json).default({}).desc("Redacted provider config used to reconnect without secrets")
    desc = field(LongText).desc("Editable description of the database and its analytical purpose")
    synonyms = field(Array[ShortText]).desc("Editable alternate names and business terms for this database")
    schemas = field(Array[ShortText]).desc("Imported physical schema names")
    selected_tables = field(Array[ShortText]).desc("Qualified table and view names selected for import")
    statistics = field(Json).desc("Imported relation and column counts plus profiling coverage")
    semantic_text = field(LongText).compute(_semantic_text).desc("Derived embedding source built from names, descriptions, and synonyms")


class DbTable(Entity):
    """Ingested table metadata for one database."""

    identifier = "db-table"
    __derive__ = _catalog_derives(_table_type)

    database_id = field(Identifier).desc("Owning db-database object_id")
    table_name = field(ShortText).desc("Physical table name")
    row_estimate = field(Integer).default(0).desc("Approximate row count when available")
    comment = field(MediumText).default("").desc("Table comment or description")
    database_name = field(ShortText).desc("Standalone human-readable database name")
    catalog_name = field(ShortText).desc("Physical catalog name when the provider distinguishes catalogs")
    schema_name = field(ShortText).desc("Physical schema name")
    relation_kind = field(ShortText).desc("Physical relation kind such as table or view")
    qualified_name = field(MediumText).desc("Catalog- and schema-qualified relation name")
    desc = field(LongText).desc("Editable description of what the relation stores")
    synonyms = field(Array[ShortText]).desc("Editable alternate names and business terms for this relation")
    primary_key = field(Array[ShortText]).desc("Reflected primary-key column names")
    foreign_keys = field(Json).desc("Reflected foreign-key descriptors")
    indexes = field(Json).desc("Reflected index and uniqueness descriptors")
    statistics = field(Json).desc("Imported row and column statistics")
    semantic_text = field(LongText).compute(_semantic_text).desc("Derived embedding source built from relation context")


class DbColumn(Entity):
    """Ingested column metadata for one table."""

    identifier = "db-column"
    __derive__ = _catalog_derives(_column_type)

    table_id = field(Identifier).desc("Owning db-table object_id")
    column_name = field(ShortText).desc("Physical column name")
    dtype = field(ShortText).default("").desc("Reported SQL type")
    nullable = field(Boolean).default(True).desc("Whether the column allows null")
    default = field(ShortText).default("").desc("Serialized default expression when present")
    comment = field(MediumText).default("").desc("Column comment when present")
    database_id = field(Identifier).desc("Owning db-database object_id copied for standalone filtering")
    database_name = field(ShortText).desc("Standalone human-readable database name")
    catalog_name = field(ShortText).desc("Physical catalog name")
    schema_name = field(ShortText).desc("Physical schema name")
    table_name = field(ShortText).desc("Physical table or view name")
    qualified_name = field(MediumText).desc("Fully qualified column name")
    ordinal = field(Integer).desc("Zero-based physical column position")
    semantic_type = field(ShortText).desc("Editable analytical type such as categorical, measure, identifier, or time")
    is_primary_key = field(Boolean).desc("Whether the column participates in the primary key")
    foreign_key = field(Json).desc("Reflected foreign-key target or an empty mapping")
    desc = field(LongText).desc("Editable business description of the column")
    synonyms = field(Array[ShortText]).desc("Editable alternate names and business terms for this column")
    profile = field(LongText).desc("Human-readable imported data profile")
    statistics = field(Json).desc("Imported null, distinct, range, length, and common-value statistics")
    semantic_text = field(LongText).compute(_semantic_text).desc("Derived embedding source built from column context and profile")


class DbEnum(Entity):
    """Ingested enum or constrained value set."""

    identifier = "db-enum"
    __derive__ = _catalog_derives(_enum_type)

    source = field(ShortText).desc("Enum source label such as table.column or dialect enum name")
    values = field(Array[ShortText]).default([]).desc("Known enum or distinct values")
    database_id = field(Identifier).desc("Owning db-database object_id")
    database_name = field(ShortText).desc("Standalone human-readable database name")
    catalog_name = field(ShortText).desc("Physical catalog name")
    schema_name = field(ShortText).desc("Physical schema name")
    table_name = field(ShortText).desc("Physical table or view name when column-derived")
    column_name = field(ShortText).desc("Physical column name when column-derived")
    desc = field(LongText).desc("Editable meaning of the value set")
    synonyms = field(Array[ShortText]).desc("Editable alternate terms for the value set")
    statistics = field(Json).desc("Imported cardinality and value-frequency context")
    semantic_text = field(LongText).compute(_semantic_text).desc("Derived embedding source built from value-set context")


class DbSnippet(Entity):
    """Registered SQL snippet executed through the database MCP profile."""

    identifier = "db-snippet"
    __derive__ = _catalog_derives(_snippet_type)

    database_id = field(Identifier).desc("Target db-database object_id")
    title = field(ShortText).desc("Snippet title")
    sql = field(LongText).desc("SQL text")
    params = field(Json).default({}).desc("Default bind parameters")
    readonly = field(Boolean).default(True).desc("Whether the snippet is read-only by default")
    card_text = field(MediumText).default("").desc("Searchable snippet description")
    database_name = field(ShortText).desc("Standalone human-readable database name")
    dialect = field(ShortText).desc("SQL dialect used to verify this snippet")
    tables = field(Array[ShortText]).desc("Qualified relations referenced by the snippet")
    synonyms = field(Array[ShortText]).desc("Editable alternate question phrases for this snippet")
    verified = field(Boolean).desc("Whether the stored SQL executed successfully before persistence")
    last_succeeded_at = field(Timestamp).desc("UTC epoch milliseconds of the latest successful verification")
    semantic_text = field(LongText).compute(_semantic_text).desc("Derived embedding source built from question and SQL context")


class DbPredicate(Entity):
    """Reusable filter expression linked to a column."""

    identifier = "db-predicate"
    __derive__ = _catalog_derives(_predicate_type)

    column_id = field(Identifier).default("").desc("Optional db-column object_id")
    expression = field(MediumText).desc("SQL predicate or expression text")
    desc = field(MediumText).default("").desc("Human-readable predicate description")
    database_id = field(Identifier).desc("Owning db-database object_id")
    database_name = field(ShortText).desc("Standalone human-readable database name")
    table_name = field(ShortText).desc("Physical table or view name")
    column_name = field(ShortText).desc("Physical column name")
    title = field(ShortText).desc("Editable short predicate name")
    synonyms = field(Array[ShortText]).desc("Editable alternate business phrases for the predicate")
    semantic_text = field(LongText).compute(_semantic_text).desc("Derived embedding source built from predicate context")


class DbRule(Entity):
    """Database-scoped natural-language rule for agents that author or review SQL."""

    identifier = "db-rule"
    __derive__ = [*_catalog_derives(_rule_type), Entity.derive("catalog.visible").compute(_rule_visible, inputs=["enabled"])]

    database_id = field(Identifier).desc("Target db-database object_id")
    database_name = field(ShortText).default("").desc("Standalone human-readable database name")
    title = field(ShortText).desc("Stable short rule title")
    desc = field(LongText).desc("Natural-language SQL or business rule")
    scope = field(ShortText).default("database").desc("Rule scope such as database, table, column, or global")
    tables = field(Array[ShortText]).default([]).desc("Qualified relations governed by the rule")
    tags = field(Array[ShortText]).default([]).desc("Rule categories used for filtering")
    triggers = field(Array[ShortText]).default([]).desc("Literal phrases that make the rule relevant")
    synonyms = field(Array[ShortText]).default([]).desc("Alternate terms described or governed by the rule")
    example = field(LongText).default("").desc("Optional worked natural-language or SQL example")
    sql = field(LongText).default("").desc("Optional reusable SQL fragment or statement")
    enabled = field(Boolean).default(True).desc("Whether agents should consider this rule")
    source = field(ShortText).default("agent").desc("Producer or authority that authored the rule")
    semantic_text = field(LongText).compute(_semantic_text).desc("Derived embedding source built from rule context")
