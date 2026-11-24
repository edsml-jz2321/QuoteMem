"""Introspect external databases into rich, editable db-* workspace rows."""

from __future__ import annotations

__all__ = ["ingest_database"]

from collections.abc import Iterable

import sqlalchemy as sa

from heavenbase.utils import Any, Dict, List, Mapping, Optional, coerce_posint, safe_error
from heavenbase.utils.redact import redact_secrets

from .entities import DATABASE_ENTITY_IDS
from .profile import ColumnProfile, profile_column
from .rows import collect_enum_rows, column_row, database_id, database_row, enum_row, table_id, table_row
from .source import redact_dsn_summary, redacted_source, resolve_database

_EDITABLE_FIELDS = {
    "db-database": ("desc", "synonyms"),
    "db-table": ("desc", "synonyms"),
    "db-column": ("desc", "synonyms", "semantic_type"),
    "db-enum": ("desc", "synonyms"),
}


def _names(value: str | Iterable[str] | None, *, label: str) -> List[str]:
    """Return normalized, de-duplicated selector names."""
    if value is None:
        return []
    values = [value] if isinstance(value, str) else list(value)
    names = list(dict.fromkeys(str(item).strip() for item in values if str(item).strip()))
    if not names:
        raise ValueError(f"{label} must select at least one name")
    return names


def _qualified(catalog: str, schema: str | None, table: str, column: str = "") -> str:
    """Build a provider-qualified relation or column name."""
    parts = [part for part in (catalog, schema or "", table, column) if part]
    return ".".join(parts)


def _aliases(catalog: str, schema: str | None, table: str) -> set[str]:
    """Return accepted relation selectors from least to most qualified."""
    aliases = {table}
    if schema:
        aliases.add(f"{schema}.{table}")
    if catalog:
        aliases.add(_qualified(catalog, schema, table))
    return aliases


def _json_value(value: Any) -> Any:
    """Detach reflected provider metadata into JSON-safe values."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_value(item) for item in value]
    return str(value)


def _reflect(fn: Any, default: Any) -> Any:
    """Return reflected metadata or a detached fallback for unsupported providers."""
    try:
        return _json_value(fn())
    except Exception:
        return default


def _foreign_key_for(column_name: str, foreign_keys: List[Mapping[str, Any]]) -> Dict[str, Any]:
    """Return the reflected foreign-key target governing one column."""
    for foreign_key in foreign_keys:
        constrained = list(foreign_key.get("constrained_columns") or [])
        if column_name not in constrained:
            continue
        index = constrained.index(column_name)
        referred = list(foreign_key.get("referred_columns") or [])
        return {
            "schema": str(foreign_key.get("referred_schema") or ""),
            "table": str(foreign_key.get("referred_table") or ""),
            "column": str(referred[index] if index < len(referred) else ""),
        }
    return {}


def _preserve(workspace: Any, entity_id: str, row: Dict[str, Any]) -> Dict[str, Any]:
    """Preserve agent-authored fields when refreshed metadata has the same id."""
    existing = workspace.get(row["object_id"], entity=entity_id)
    if existing is None:
        return row
    for name in _EDITABLE_FIELDS.get(entity_id, ()):
        if name in existing:
            row[name] = existing[name]
    return row


def _profile_limits(workspace: Any) -> tuple[int, int]:
    """Return bounded profiling limits from the workspace config authority."""
    config = workspace.context.config
    max_enum_values = coerce_posint(config.get("heavenbase.database.profile.max_enum_values", default=50), "max_enum_values")
    top_values = coerce_posint(config.get("heavenbase.database.profile.top_values", default=10), "top_values")
    return max_enum_values, top_values


def _empty_profile(column: Mapping[str, Any], *, is_primary_key: bool, foreign_key: Mapping[str, Any]) -> ColumnProfile:
    """Return type-only context when data profiling is disabled."""
    dtype = column.get("type")
    if is_primary_key:
        semantic_type = "identifier"
    elif foreign_key:
        semantic_type = "reference"
    elif isinstance(dtype, sa.types.Boolean):
        semantic_type = "boolean"
    elif isinstance(dtype, (sa.types.Date, sa.types.DateTime, sa.types.Time, sa.types.Interval)):
        semantic_type = "time"
    elif isinstance(dtype, sa.types.Numeric):
        semantic_type = "measure"
    elif isinstance(dtype, sa.types.Enum):
        semantic_type = "categorical"
    else:
        semantic_type = "text"
    text = f"SQL type {dtype or 'unknown'}; analytical type {semantic_type}; data profiling disabled"
    return ColumnProfile(semantic_type=semantic_type, text=text, statistics={"profiled": False}, enum_values=[])


def ingest_database(
    workspace: Any,
    db: Any,
    *,
    name: Optional[str] = None,
    tables: str | Iterable[str] | None = None,
    columns: Mapping[str, str | Iterable[str]] | None = None,
    profile: bool = True,
) -> Dict[str, List[str]]:
    """Introspect a selected database surface into rich, editable rows.

    Args:
        workspace: Target HeavenBase workspace with the database extension enabled.
        db: ``Database`` instance, workspace SQL backend name, DSN URL, or local SQLite path.
        name: Optional display label. Defaults to the current database name or source path.
        tables: Optional relation selectors. Names may be unqualified, schema-qualified, or catalog-qualified.
        columns: Optional relation-to-column selectors. When ``tables`` is omitted, its keys also select the relations.
        profile: Collect exact null, cardinality, range, length, and common-value statistics when true.

    Returns:
        Created or updated object ids grouped by entity kind.

    Raises:
        ValueError: If the database cannot be resolved or a relation or column selector is unknown.
    """
    if "database" not in getattr(workspace, "extensions", lambda: [])():
        workspace.enable_extension("database")
    if columns is not None and not isinstance(columns, Mapping):
        raise TypeError("columns must map relation selectors to column names")

    resolved = resolve_database(workspace, db)
    live_db = resolved.db
    spec = live_db.spec
    summary = redact_dsn_summary(spec)
    current_database = live_db.current_database()
    label = str(name or current_database or spec.database or resolved.backend_ref or "database")
    catalog_name = str(spec.catalog or current_database or spec.database or "")
    database_object_id = database_id(label, summary)
    inspector = sa.inspect(live_db.engine)
    schemas: List[str | None] = [None] if live_db.dialect == "sqlite" else list(live_db.schemas() or [None])

    relations: List[Dict[str, Any]] = []
    for schema in schemas:
        table_names = set(live_db.tables(schema=schema))
        view_names = set(live_db.views(schema=schema))
        for table_name in sorted(table_names | view_names):
            relations.append(
                {
                    "catalog": catalog_name,
                    "schema": schema,
                    "table": table_name,
                    "kind": "view" if table_name in view_names and table_name not in table_names else "table",
                    "aliases": _aliases(catalog_name, schema, table_name),
                }
            )

    table_selectors = _names(tables, label="tables") if tables is not None else []
    column_selectors = {str(key).strip(): _names(value, label=f"columns[{key!r}]") for key, value in (columns or {}).items()}
    for selector in [*table_selectors, *column_selectors]:
        matches = [relation for relation in relations if selector in relation["aliases"]]
        if not matches:
            raise ValueError(f"Unknown table or view selector {selector!r}")
        if "." not in selector and len(matches) > 1:
            choices = ", ".join(_qualified(catalog_name, relation["schema"], relation["table"]) for relation in matches)
            raise ValueError(f"Ambiguous table or view selector {selector!r}; choose one of: {choices}")

    relation_selectors = table_selectors or (list(column_selectors) if columns is not None else [])

    selected = [relation for relation in relations if not relation_selectors or any(selector in relation["aliases"] for selector in relation_selectors)]
    if table_selectors:
        outside = [selector for selector in column_selectors if not any(selector in relation["aliases"] for relation in selected)]
        if outside:
            raise ValueError(f"Column selectors must belong to selected tables: {', '.join(outside)}")
    selected_columns: Dict[str, List[Mapping[str, Any]]] = {}
    column_ordinals: Dict[str, Dict[str, int]] = {}
    for relation in selected:
        full_columns = [item for item in live_db.columns(relation["table"], schema=relation["schema"], full=True) if isinstance(item, Mapping)]
        requested = [names for selector, names in column_selectors.items() if selector in relation["aliases"]]
        names = list(dict.fromkeys(name for group in requested for name in group))
        available = {str(item.get("name") or "") for item in full_columns}
        missing = [column_name for column_name in names if column_name not in available]
        if missing:
            qualified = _qualified(catalog_name, relation["schema"], relation["table"])
            raise ValueError(f"Unknown columns for {qualified!r}: {', '.join(missing)}")
        relation_key = _qualified(catalog_name, relation["schema"], relation["table"])
        selected_columns[relation_key] = [item for item in full_columns if not names or str(item.get("name") or "") in names]
        column_ordinals[relation_key] = {str(item.get("name") or ""): ordinal for ordinal, item in enumerate(full_columns)}

    max_enum_values, top_values = _profile_limits(workspace)
    table_rows: List[Dict[str, Any]] = []
    column_rows: List[Dict[str, Any]] = []
    enum_rows: List[Dict[str, Any]] = []
    selected_schemas = list(dict.fromkeys(str(relation["schema"] or "") for relation in selected))
    selected_names = [_qualified(catalog_name, relation["schema"], relation["table"]) for relation in selected]

    for schema in dict.fromkeys(relation["schema"] for relation in selected):
        enum_rows.extend(
            collect_enum_rows(
                inspector,
                database_id=database_object_id,
                database_name=label,
                catalog_name=catalog_name,
                schema=schema,
            )
        )

    profile_errors: List[str] = []
    for relation in selected:
        schema = relation["schema"]
        table_name = relation["table"]
        qualified_name = _qualified(catalog_name, schema, table_name)
        object_id = table_id(database_object_id, table_name, schema)
        primary_key = list(_reflect(lambda: live_db.pks(table_name, schema=schema), []))
        foreign_keys = list(_reflect(lambda: live_db.fks(table_name, schema=schema), []))
        # SQLAlchemy reports no indexes for some dialects (DuckDB); the Database reads those from the catalog.
        indexes = list(_reflect(lambda: live_db._reflected_indexes(table_name, schema), []))
        comment_data = _reflect(lambda: inspector.get_table_comment(table_name, schema=schema), {})
        comment = str(comment_data.get("text") or "") if isinstance(comment_data, Mapping) else ""
        try:
            n_rows = int(live_db.n_rows(table_name, schema=schema)) if profile else 0
        except Exception as exc:
            n_rows = 0
            profile_errors.append(f"{qualified_name}: {redact_secrets(safe_error(exc))}")
        relation_columns = selected_columns[qualified_name]
        table_rows.append(
            table_row(
                object_id=object_id,
                database_id=database_object_id,
                database_name=label,
                catalog_name=catalog_name,
                schema_name=schema or "",
                table_name=table_name,
                relation_kind=relation["kind"],
                qualified_name=qualified_name,
                row_estimate=n_rows,
                comment=comment,
                primary_key=primary_key,
                foreign_keys=foreign_keys,
                indexes=indexes,
                statistics={"row_count": n_rows, "selected_column_count": len(relation_columns), "profiled": bool(profile)},
            )
        )

        for column in relation_columns:
            column_name = str(column.get("name") or "")
            ordinal = column_ordinals[qualified_name][column_name]
            foreign_key = _foreign_key_for(column_name, foreign_keys)
            is_primary_key = column_name in primary_key
            column_profile = (
                profile_column(
                    live_db,
                    table_name=table_name,
                    schema=schema,
                    column=column,
                    n_rows=n_rows,
                    is_primary_key=is_primary_key,
                    foreign_key=foreign_key,
                    max_enum_values=max_enum_values,
                    top_values=top_values,
                )
                if profile
                else _empty_profile(column, is_primary_key=is_primary_key, foreign_key=foreign_key)
            )
            row = column_row(
                database_id=database_object_id,
                database_name=label,
                catalog_name=catalog_name,
                schema_name=schema or "",
                table_id=object_id,
                table_name=table_name,
                qualified_name=f"{qualified_name}.{column_name}",
                ordinal=ordinal,
                column=column,
                semantic_type=column_profile.semantic_type,
                is_primary_key=is_primary_key,
                foreign_key=foreign_key,
                profile=column_profile.text,
                statistics=column_profile.statistics,
            )
            if row is not None:
                column_rows.append(row)
            if column_profile.enum_values:
                enum_rows.append(
                    enum_row(
                        database_id=database_object_id,
                        database_name=label,
                        source=f"{qualified_name}.{column_name}",
                        values=column_profile.enum_values,
                        catalog_name=catalog_name,
                        schema_name=schema or "",
                        table_name=table_name,
                        column_name=column_name,
                        statistics={
                            "cardinality": column_profile.statistics.get("distinct_count"),
                            "common_values": column_profile.statistics.get("common_values", []),
                            "origin": "profile",
                        },
                    )
                )

    database = database_row(
        object_id=database_object_id,
        name=label,
        dsn_summary=summary,
        dialect=str(live_db.dialect or spec.dialect or ""),
        backend_ref=resolved.backend_ref,
        source=redacted_source(spec),
        schemas=[schema for schema in selected_schemas if schema],
        selected_tables=selected_names,
        statistics={
            "relation_count": len(table_rows),
            "column_count": len(column_rows),
            "enum_count": len(enum_rows),
            "profiled": bool(profile),
            "errors": profile_errors,
        },
    )
    rows_by_entity = {
        "db-database": [_preserve(workspace, "db-database", database)],
        "db-table": [_preserve(workspace, "db-table", row) for row in table_rows],
        "db-column": [_preserve(workspace, "db-column", row) for row in column_rows],
        "db-enum": [_preserve(workspace, "db-enum", row) for row in enum_rows],
    }
    for entity_id, rows in rows_by_entity.items():
        if rows:
            workspace.upsert_many(entity_id, rows)
    bind = getattr(getattr(workspace, "database", None), "_bind_source", None)
    if bind is not None:
        bind(database_object_id, live_db)

    return {
        "database": [database_object_id],
        "tables": [row["object_id"] for row in table_rows],
        "columns": [row["object_id"] for row in column_rows],
        "enums": [row["object_id"] for row in enum_rows],
        "entities": list(DATABASE_ENTITY_IDS),
    }
