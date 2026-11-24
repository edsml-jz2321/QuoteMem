"""ResultFrame export helpers with late optional imports."""

from __future__ import annotations

__all__ = [
    "rows",
    "to_daft",
    "to_numpy",
    "to_numpy_dict",
    "to_pandas",
    "to_pydantic",
    "to_pyarrow",
]

from heavenbase.utils import Any, Optional, Sequence

from .base import need


def rows(frame: Any, columns: Optional[Sequence[str]] = None) -> list[dict[str, Any]]:
    """Return frame rows restricted to optional columns.

    Args:
        frame (Any): ResultFrame-compatible value.
        columns (Optional[Sequence[str]]): Optional columns to export.

    Returns:
        list[dict[str, Any]]: Selected row mappings in frame order.

    Raises:
        ValueError: If a requested column is unknown.
    """
    cols = list(columns) if columns is not None else frame.col_names()
    for col in cols:
        frame.column(col)
    return [{col: row.get(col) for col in cols} for row in frame.rows()]


def to_pandas(frame: Any, *, columns: Optional[Sequence[str]] = None) -> Any:
    """Export a result frame to pandas.

    Args:
        frame (Any): ResultFrame-compatible value.
        columns (Optional[Sequence[str]]): Optional columns to export.

    Returns:
        Any: ``pandas.DataFrame`` with selected columns.

    Raises:
        ImportError: If pandas is unavailable.
        ValueError: If a requested column is unknown.
    """
    pd = need("pandas", "ResultFrame.to_pandas()")

    return pd.DataFrame(rows(frame, columns))


def to_pyarrow(frame: Any, *, columns: Optional[Sequence[str]] = None, schema: Any = None) -> Any:
    """Export a result frame to PyArrow.

    Args:
        frame (Any): ResultFrame-compatible value.
        columns (Optional[Sequence[str]]): Optional columns to export.
        schema (Any): Optional PyArrow schema.

    Returns:
        Any: ``pyarrow.Table`` with selected columns.

    Raises:
        ImportError: If PyArrow is unavailable.
        ValueError: If a requested column is unknown.
    """
    pa = need("pyarrow", "ResultFrame.to_pyarrow()")

    return pa.Table.from_pylist(rows(frame, columns), schema=schema)


def to_daft(frame: Any, *, columns: Optional[Sequence[str]] = None) -> Any:
    """Export a result frame to Daft through PyArrow.

    Args:
        frame (Any): ResultFrame-compatible value.
        columns (Optional[Sequence[str]]): Optional columns to export.

    Returns:
        Any: ``daft.DataFrame`` with selected columns.

    Raises:
        ImportError: If Daft or PyArrow is unavailable.
        ValueError: If a requested column is unknown.
    """
    try:
        import daft
    except ImportError as exc:
        raise ImportError("daft is required for ResultFrame.to_daft(); install heavenbase[daft].") from exc

    table = to_pyarrow(frame, columns=columns)
    return daft.from_arrow(table)


def to_numpy(frame: Any, *columns: str, dtype: Any = None) -> Any:
    """Export a result frame to NumPy.

    Args:
        frame (Any): ResultFrame-compatible value.
        *columns (str): Optional columns to export.
        dtype (Any): Optional NumPy dtype.

    Returns:
        Any: ``numpy.ndarray`` with selected columns.

    Raises:
        ImportError: If NumPy is unavailable.
        ValueError: If a requested column is unknown.
    """
    np = need("numpy", "ResultFrame.to_numpy()")

    cols = list(columns) if columns else frame.col_names()
    return np.asarray([[row[col] for col in cols] for row in rows(frame, cols)], dtype=dtype)


def to_numpy_dict(frame: Any, *, columns: Optional[Sequence[str]] = None) -> dict[str, Any]:
    """Export a result frame to named NumPy arrays.

    Args:
        frame (Any): ResultFrame-compatible value.
        columns (Optional[Sequence[str]]): Optional columns to export.

    Returns:
        dict[str, Any]: Column names mapped to ``numpy.ndarray`` values.

    Raises:
        ImportError: If NumPy is unavailable.
        ValueError: If a requested column is unknown.
    """
    np = need("numpy", "ResultFrame.to_numpy_dict()")

    cols = list(columns) if columns is not None else frame.col_names()
    return {col: np.asarray(frame.column(col)) for col in cols}


def to_pydantic(frame: Any, model: Any, *, rename: Optional[dict] = None, strict: bool = True) -> list[Any]:
    """Export frame rows to Pydantic model instances.

    Args:
        frame (Any): ResultFrame-compatible value.
        model (Any): Pydantic model class used to validate each row.
        rename (Optional[dict]): Optional source-to-model field mapping.
        strict (bool): Enable strict validation on Pydantic v2 models.

    Returns:
        list[Any]: Model instances in frame order.

    Raises:
        TypeError: If ``model`` cannot construct row objects.
        ValueError: If row validation fails.
    """
    rename = dict(rename or {})
    out = []
    for row in frame.rows():
        data = {rename.get(key, key): value for key, value in row.items()}
        if hasattr(model, "model_validate"):
            out.append(model.model_validate(data, strict=strict))
        elif hasattr(model, "parse_obj"):
            out.append(model.parse_obj(data))
        else:
            out.append(model(**data))
    return out
