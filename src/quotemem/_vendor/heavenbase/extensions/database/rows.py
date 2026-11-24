"""Build deterministic metadata rows for database ingest."""

from __future__ import annotations

__all__ = [
    "collect_enum_rows",
    "column_id",
    "column_row",
    "database_id",
    "database_row",
    "enum_id",
    "enum_row",
    "serialize_default",
    "snippet_id",
    "table_id",
    "table_row",
]

from heavenbase.utils import Any, Dict, List, Mapping, Optional, hash_id
from heavenbase.utils.redact import redact_secrets

_ID_LEN = 32


def database_id(name: str, summary: str) -> str:
    """Return the stable object id for one ingested database.

    Args:
        name: Human-readable database label.
        summary: Redacted source summary.

    Returns:
        Deterministic ``db-database`` object id.
    """
    return f"db-db-{hash_id(name, summary, length=_ID_LEN)}"


def table_id(database_id: str, table_name: str, schema: Optional[str]) -> str:
    """Return the stable object id for one ingested table.

    Args:
        database_id: Owning ``db-database`` object id.
        table_name: Physical table or view name.
        schema: Optional physical schema name.

    Returns:
        Deterministic ``db-table`` object id.
    """
    return f"db-tbl-{hash_id(database_id, schema or '', table_name, length=_ID_LEN)}"


def column_id(table_id: str, column_name: str) -> str:
    """Return the stable object id for one ingested column.

    Args:
        table_id: Owning ``db-table`` object id.
        column_name: Physical column name.

    Returns:
        Deterministic ``db-column`` object id.
    """
    return f"db-col-{hash_id(table_id, column_name, length=_ID_LEN)}"


def enum_id(database_id: str, source: str) -> str:
    """Return the stable object id for one ingested enum.

    Args:
        database_id: Owning ``db-database`` object id.
        source: Dialect enum source label.

    Returns:
        Deterministic ``db-enum`` object id.
    """
    return f"db-enum-{hash_id(database_id, source, length=_ID_LEN)}"


def snippet_id(database_id: str, title: str) -> str:
    """Return the stable object id for one verified SQL snippet.

    Args:
        database_id: Owning ``db-database`` object id.
        title: Human-readable snippet title.

    Returns:
        Deterministic ``db-snippet`` object id.
    """
    return f"db-sql-{hash_id(database_id, title, length=_ID_LEN)}"


def serialize_default(value: Any) -> str:
    """Return a redacted string representation for one column default.

    Args:
        value: SQLAlchemy default value or expression.

    Returns:
        Empty string for missing defaults, otherwise a redacted string.
    """
    if value is None:
        return ""
    return redact_secrets(str(value))


def database_row(
    *,
    object_id: str,
    name: str,
    dsn_summary: str,
    dialect: str,
    backend_ref: str,
    source: Mapping[str, Any],
    schemas: List[str],
    selected_tables: List[str],
    statistics: Mapping[str, Any],
) -> Dict[str, Any]:
    """Build one ``db-database`` row dictionary.

    Args:
        object_id: Stable database object id.
        name: Human-readable database label.
        dsn_summary: Redacted source summary.
        dialect: SQL dialect label.
        backend_ref: Workspace backend name when ingest used a named backend.
        source: Redacted reconnectable source config.
        schemas: Imported physical schema names.
        selected_tables: Imported qualified relation names.
        statistics: Import coverage counts.

    Returns:
        Row dictionary ready for workspace upsert.
    """
    return {
        "object_id": object_id,
        "name": name,
        "dsn_summary": dsn_summary,
        "dialect": dialect,
        "backend_ref": backend_ref,
        "source": dict(source),
        "desc": "",
        "synonyms": [],
        "schemas": list(schemas),
        "selected_tables": list(selected_tables),
        "statistics": dict(statistics),
    }


def table_row(
    *,
    object_id: str,
    database_id: str,
    database_name: str,
    catalog_name: str,
    schema_name: str,
    table_name: str,
    relation_kind: str,
    qualified_name: str,
    row_estimate: int = 0,
    comment: str = "",
    primary_key: Optional[List[str]] = None,
    foreign_keys: Optional[List[Mapping[str, Any]]] = None,
    indexes: Optional[List[Mapping[str, Any]]] = None,
    statistics: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build one ``db-table`` row dictionary.

    Args:
        object_id: Stable table object id.
        database_id: Owning ``db-database`` object id.
        database_name: Standalone human-readable database name.
        catalog_name: Physical catalog name when available.
        schema_name: Physical schema name when available.
        table_name: Physical table or view name.
        relation_kind: Relation kind, normally ``table`` or ``view``.
        qualified_name: Provider-qualified relation name.
        row_estimate: Approximate row count when available.
        comment: Optional table comment.
        primary_key: Reflected primary-key column names.
        foreign_keys: Reflected foreign-key descriptors.
        indexes: Reflected index descriptors.
        statistics: Imported relation statistics.

    Returns:
        Row dictionary ready for workspace upsert.
    """
    return {
        "object_id": object_id,
        "database_id": database_id,
        "database_name": database_name,
        "catalog_name": catalog_name,
        "schema_name": schema_name,
        "table_name": table_name,
        "relation_kind": relation_kind,
        "qualified_name": qualified_name,
        "row_estimate": row_estimate,
        "comment": comment,
        "desc": "",
        "synonyms": [],
        "primary_key": list(primary_key or []),
        "foreign_keys": [dict(item) for item in (foreign_keys or [])],
        "indexes": [dict(item) for item in (indexes or [])],
        "statistics": dict(statistics or {}),
    }


def column_row(
    *,
    database_id: str,
    database_name: str,
    catalog_name: str,
    schema_name: str,
    table_id: str,
    table_name: str,
    qualified_name: str,
    ordinal: int,
    column: Mapping[str, Any],
    semantic_type: str,
    is_primary_key: bool,
    foreign_key: Optional[Mapping[str, Any]] = None,
    profile: str = "",
    statistics: Optional[Mapping[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Build one ``db-column`` row dictionary from reflected column metadata.

    Args:
        database_id: Owning ``db-database`` object id.
        database_name: Standalone human-readable database name.
        catalog_name: Physical catalog name when available.
        schema_name: Physical schema name when available.
        table_id: Owning ``db-table`` object id.
        table_name: Physical relation name.
        qualified_name: Fully qualified column name.
        ordinal: Zero-based column position.
        column: SQLAlchemy column metadata mapping.
        semantic_type: Initial analytical column type.
        is_primary_key: Whether the column participates in the primary key.
        foreign_key: Reflected foreign-key target descriptor.
        profile: Human-readable imported profile.
        statistics: Machine-readable imported statistics.

    Returns:
        Row dictionary ready for workspace upsert, or ``None`` when the column has no name.
    """
    column_name = str(column.get("name") or "")
    if not column_name:
        return None
    return {
        "object_id": column_id(table_id, column_name),
        "database_id": database_id,
        "database_name": database_name,
        "catalog_name": catalog_name,
        "schema_name": schema_name,
        "table_id": table_id,
        "table_name": table_name,
        "column_name": column_name,
        "qualified_name": qualified_name,
        "ordinal": ordinal,
        "dtype": str(column.get("type") or ""),
        "nullable": bool(column.get("nullable", True)),
        "default": serialize_default(column.get("default")),
        "comment": str(column.get("comment") or ""),
        "semantic_type": semantic_type,
        "is_primary_key": is_primary_key,
        "foreign_key": dict(foreign_key or {}),
        "desc": "",
        "synonyms": [],
        "profile": profile,
        "statistics": dict(statistics or {}),
    }


def enum_row(
    *,
    database_id: str,
    database_name: str,
    source: str,
    values: List[str],
    catalog_name: str = "",
    schema_name: str = "",
    table_name: str = "",
    column_name: str = "",
    statistics: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build one ``db-enum`` row dictionary.

    Args:
        database_id: Owning ``db-database`` object id.
        database_name: Standalone human-readable database name.
        source: Dialect enum source label.
        values: Enum labels or constrained values.
        catalog_name: Physical catalog name when available.
        schema_name: Physical schema name when available.
        table_name: Physical relation name for column-derived enums.
        column_name: Physical column name for column-derived enums.
        statistics: Cardinality and frequency context.

    Returns:
        Row dictionary ready for workspace upsert.
    """
    return {
        "object_id": enum_id(database_id, source),
        "database_id": database_id,
        "database_name": database_name,
        "source": source,
        "values": list(values),
        "catalog_name": catalog_name,
        "schema_name": schema_name,
        "table_name": table_name,
        "column_name": column_name,
        "desc": "",
        "synonyms": [],
        "statistics": dict(statistics or {}),
    }


def collect_enum_rows(
    inspector: Any,
    *,
    database_id: str,
    database_name: str,
    catalog_name: str,
    schema: Optional[str],
) -> List[Dict[str, Any]]:
    """Return all enum rows visible to one SQLAlchemy inspector.

    Args:
        inspector: SQLAlchemy inspector object.
        database_id: Owning ``db-database`` object id.
        database_name: Standalone human-readable database name.
        catalog_name: Physical catalog name when available.
        schema: Optional physical schema name.

    Returns:
        List of ``db-enum`` rows. Dialects without enum inspection return an empty list.
    """
    rows: List[Dict[str, Any]] = []
    get_enums = getattr(inspector, "get_enums", None)
    if get_enums is None:
        return rows
    try:
        enums = get_enums(schema=schema)
    except Exception:
        return rows
    for item in enums or []:
        if not isinstance(item, Mapping):
            continue
        name = str(item.get("name") or "")
        values = [str(value) for value in (item.get("labels") or item.get("values") or [])]
        if not name and not values:
            continue
        source = f"{schema + '.' if schema else ''}{name}".strip(".")
        rows.append(
            enum_row(
                database_id=database_id,
                database_name=database_name,
                source=source,
                values=values,
                catalog_name=catalog_name,
                schema_name=schema or "",
                statistics={"cardinality": len(values), "origin": "dialect"},
            )
        )
    return rows
