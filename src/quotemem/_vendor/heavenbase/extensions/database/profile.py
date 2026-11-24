"""Portable, bounded column profiling for database ingest."""

from __future__ import annotations

__all__ = ["ColumnProfile", "profile_column"]

import math
from datetime import date, datetime, time
from decimal import Decimal

import sqlalchemy as sa

from heavenbase.utils import Any, Dict, List, Mapping, dataclass, safe_error
from heavenbase.utils.redact import redact_secrets


@dataclass(frozen=True)
class ColumnProfile:
    """Detached analytical profile for one reflected database column."""

    semantic_type: str
    text: str
    statistics: Dict[str, Any]
    enum_values: List[str]


def _json_value(value: Any) -> Any:
    """Convert one provider scalar into a JSON-safe detached value."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Decimal):
        number = float(value)
        return number if math.isfinite(number) else str(value)
    if isinstance(value, (date, datetime, time)):
        return value.isoformat()
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_value(item) for item in value]
    return str(value)


def _base_semantic_type(dtype: Any, *, is_primary_key: bool, has_foreign_key: bool) -> str:
    """Infer the initial analytical role from portable SQLAlchemy types."""
    if is_primary_key:
        return "identifier"
    if has_foreign_key:
        return "reference"
    if isinstance(dtype, sa.types.Boolean):
        return "boolean"
    if isinstance(dtype, (sa.types.Date, sa.types.DateTime, sa.types.Time, sa.types.Interval)):
        return "time"
    if isinstance(dtype, sa.types.Numeric):
        return "measure"
    if isinstance(dtype, (sa.types.LargeBinary, sa.types.PickleType)):
        return "binary"
    if isinstance(dtype, sa.types.Enum):
        return "categorical"
    return "text"


def _profile_text(dtype: str, semantic_type: str, statistics: Mapping[str, Any]) -> str:
    """Render concise human context from one machine-readable profile."""
    parts = [f"SQL type {dtype or 'unknown'}", f"analytical type {semantic_type}"]
    for key, label in (("non_null", "non-null"), ("null_count", "null"), ("distinct_count", "distinct")):
        if statistics.get(key) is not None:
            parts.append(f"{label} {statistics[key]}")
    if statistics.get("min") is not None or statistics.get("max") is not None:
        parts.append(f"range {statistics.get('min')} to {statistics.get('max')}")
    if statistics.get("avg") is not None:
        parts.append(f"average {statistics['avg']}")
    lengths = statistics.get("lengths") or {}
    if lengths:
        parts.append(f"length {lengths.get('min_len')} to {lengths.get('max_len')} (average {lengths.get('avg_len')})")
    common = statistics.get("common_values") or []
    if common:
        parts.append("common values " + ", ".join(f"{item.get('value')} ({item.get('n')})" for item in common))
    return "; ".join(parts)


def profile_column(
    live_db: Any,
    *,
    table_name: str,
    schema: str | None,
    column: Mapping[str, Any],
    n_rows: int,
    is_primary_key: bool,
    foreign_key: Mapping[str, Any] | None,
    max_enum_values: int,
    top_values: int,
) -> ColumnProfile:
    """Profile one reflected column through portable aggregate operations.

    Profiling failure is isolated to the affected statistic. The import still
    creates an editable schema row and records redacted error context.
    """
    column_name = str(column.get("name") or "")
    dtype = column.get("type")
    semantic_type = _base_semantic_type(dtype, is_primary_key=is_primary_key, has_foreign_key=bool(foreign_key))
    statistics: Dict[str, Any] = {"row_count": n_rows}
    errors: List[str] = []

    def collect(name: str, fn: Any) -> Any:
        try:
            value = _json_value(fn())
            statistics[name] = value
            return value
        except Exception as exc:
            errors.append(f"{name}: {redact_secrets(safe_error(exc))}")
            return None

    non_null = collect("non_null", lambda: live_db.col_agg(table_name, column_name, "count", schema=schema))
    distinct = collect("distinct_count", lambda: live_db.col_agg(table_name, column_name, "count", distinct=True, schema=schema))
    statistics["null_count"] = max(0, n_rows - int(non_null)) if non_null is not None else None

    if semantic_type == "measure":
        collect("min", lambda: live_db.col_agg(table_name, column_name, "min", schema=schema))
        collect("max", lambda: live_db.col_agg(table_name, column_name, "max", schema=schema))
        collect("avg", lambda: live_db.col_agg(table_name, column_name, "avg", schema=schema))
    elif semantic_type == "time":
        collect("min", lambda: live_db.col_agg(table_name, column_name, "min", schema=schema))
        collect("max", lambda: live_db.col_agg(table_name, column_name, "max", schema=schema))
    elif semantic_type == "text":
        collect("lengths", lambda: live_db.col_lengths(table_name, column_name, schema=schema))

    enum_values: List[str] = []
    if distinct is not None and 0 < int(distinct) <= max_enum_values and semantic_type in {"text", "categorical", "boolean"}:
        semantic_type = "categorical"
        frequencies = collect("common_values", lambda: live_db.col_freqs(table_name, column_name, topk=max_enum_values, schema=schema)) or []
        statistics["common_values"] = frequencies[:top_values]
        enum_values = [str(item.get("value")) for item in frequencies if item.get("value") is not None]
    if errors:
        statistics["errors"] = errors
    text = _profile_text(str(dtype or ""), semantic_type, statistics)
    return ColumnProfile(semantic_type=semantic_type, text=text, statistics=statistics, enum_values=enum_values)
