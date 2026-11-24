"""Schema-index normalization for database metadata caches."""

from __future__ import annotations

__all__ = ["SchemaIndex", "normalize_schema_index"]

from ..utils.typing import Any, Mapping, Sequence

SchemaIndex = dict[str, list[str]]


def normalize_schema_index(schema_index: Mapping[str, Sequence[str]] | dict[str, Any] | None) -> SchemaIndex:
    """Normalize schema-index payloads to `{table: [columns...]}`.

    Args:
        schema_index (Mapping[str, Sequence[str]] | dict[str, Any] | None): Raw schema index. Supported shapes are a direct
            table-to-columns mapping or `{"tables": {...}}`. Column values may be sequences, dictionaries, strings, or
            `None`.

    Returns:
        SchemaIndex: Normalized table-to-column mapping with blank names removed and duplicate columns collapsed.

    Raises:
        ValueError: If `schema_index` is not a supported mapping shape.
    """
    if not schema_index:
        return {}
    table_map: Any = schema_index.get("tables") if isinstance(schema_index, dict) and isinstance(schema_index.get("tables"), dict) else schema_index
    if not isinstance(table_map, dict):
        raise ValueError("schema_index must be a mapping of table -> columns, or {'tables': {...}}")
    out: SchemaIndex = {}
    for raw_table, raw_columns in table_map.items():
        table = str(raw_table).strip()
        if not table:
            continue
        if raw_columns is None:
            out[table] = []
            continue
        candidates = list(raw_columns.keys()) if isinstance(raw_columns, dict) else [raw_columns] if isinstance(raw_columns, str) else list(raw_columns)
        cols = []
        seen = set()
        for item in candidates:
            col = str(item).strip()
            if col and col not in seen:
                seen.add(col)
                cols.append(col)
        out[table] = cols
    return out
