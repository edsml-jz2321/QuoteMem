"""External row normalization helpers for interop import paths."""

from __future__ import annotations

__all__ = [
    "clean_row",
    "clean_value",
    "is_missing",
    "row_from_any",
    "rows_from_naive_table",
    "source_row",
]

from heavenbase.utils import Any, Dict, Iterable, List, Mapping, Optional, Sequence


def is_missing(value: Any) -> bool:
    """Return whether a value is a common external missing sentinel.

    Args:
        value (Any): Candidate source value from an external table, row object,
            or scalar wrapper.

    Returns:
        bool: True when the value is `None`, a known pandas missing singleton,
        or a NaN-like scalar.
    """
    if value is None:
        return True
    if value.__class__.__name__ in ("NAType", "NaTType"):
        return True
    try:
        return bool(value != value)
    except Exception:
        return False


def clean_value(value: Any) -> Any:
    """Return a Python-native value for common external scalar wrappers.

    Args:
        value (Any): Source scalar or array-like value.

    Returns:
        Any: `None` for missing values, a Python scalar for NumPy scalar values,
        a list for list-like array wrappers, or the original value when no
        normalization applies.
    """
    if is_missing(value):
        return None
    if hasattr(value, "item") and value.__class__.__module__.startswith("numpy"):
        try:
            return value.item()
        except Exception:
            return value
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes, dict)):
        try:
            return value.tolist()
        except Exception:
            return value
    return value


def source_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a row with source columns preserved and values normalized.

    Args:
        row (Mapping[str, Any]): Source row mapping.

    Returns:
        Dict[str, Any]: Row keyed by stringified source column names. Missing
        values are kept as `None` so schema inference can still see null-only
        columns.
    """
    return {str(key): clean_value(value) for key, value in row.items()}


def clean_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a row dict with missing values omitted.

    Args:
        row (Mapping[str, Any]): Source row mapping.

    Returns:
        Dict[str, Any]: Row keyed by source column names, excluding missing
        values after normalization.
    """
    return {key: value for key, value in source_row(row).items() if not is_missing(value)}


def row_from_any(row: Any) -> Dict[str, Any]:
    """Coerce one record-shaped external object to a HeavenBase row mapping.

    Args:
        row (Any): Entity instance, mapping, SQLAlchemy row-like object,
        Pydantic model-like object, or pandas Series-like object.

    Returns:
        Dict[str, Any]: Cleaned HeavenBase row mapping.

    Raises:
        TypeError: If the value is not a supported record-shaped object.
    """
    from ..entity import Entity

    if isinstance(row, Entity):
        return row.to_dict()
    if isinstance(row, Mapping):
        return clean_row(row)
    if hasattr(row, "_mapping"):
        return clean_row(dict(row._mapping))
    if hasattr(row, "model_dump"):
        return clean_row(row.model_dump())
    if hasattr(row, "dict") and (hasattr(row, "__fields__") or row.__class__.__module__.startswith("pydantic")):
        return clean_row(row.dict())
    if row.__class__.__module__.startswith("pandas") and row.__class__.__name__ == "Series" and hasattr(row, "to_dict"):
        return clean_row(row.to_dict())
    raise TypeError(f"Cannot coerce {type(row).__name__} to a HeavenBase row mapping")


def rows_from_naive_table(data: Iterable[Any], columns: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    """Return normalized rows from mapping rows or table-shaped sequences.

    Args:
        data (Iterable[Any]): Iterable of mapping rows or row-shaped sequences.
        columns (Optional[Sequence[str]]): Column names required for sequence
            rows. Mapping rows use their own keys.

    Returns:
        List[Dict[str, Any]]: Source rows with normalized values.

    Raises:
        ValueError: If sequence rows are provided without column names, column
        names are duplicated, or a row length differs from the column count.
    """
    items = list(data)
    if not items:
        return []
    if all(isinstance(item, Mapping) for item in items):
        return [source_row(item) for item in items]
    if columns is None:
        raise ValueError("List/List tuple table input needs explicit columns")
    names = [str(name) for name in columns]
    if len(set(names)) != len(names):
        raise ValueError("Table columns must be unique")
    rows = []
    for item in items:
        values = list(item)
        if len(values) != len(names):
            raise ValueError(f"Table row has {len(values)} values but columns has {len(names)} names")
        rows.append(source_row(dict(zip(names, values))))
    return rows
