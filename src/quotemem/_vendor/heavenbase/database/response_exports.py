"""Optional tabular export adapters for SQL responses."""

from __future__ import annotations

from ..utils.typing import Any, List, Optional, Union

ColumnRef = Union[str, int]


def _to_pandas(response: Any, *, columns: Optional[List[ColumnRef]] = None, check: bool = True) -> Any:
    """Return response rows as a pandas DataFrame."""
    import pandas as pd

    names = response._column_names(columns, check=check)
    return pd.DataFrame(response.to_dicts(columns=columns, check=check), columns=names)


def _to_pyarrow(response: Any, *, columns: Optional[List[ColumnRef]] = None, schema: Any = None, check: bool = True) -> Any:
    """Return response rows as a pyarrow Table."""
    import pyarrow as pa

    return pa.Table.from_pylist(response.to_dicts(columns=columns, check=check), schema=schema)


def _to_numpy(response: Any, *columns: str, dtype: Any = None, check: bool = True) -> Any:
    """Return response rows as a two-dimensional NumPy array."""
    import numpy as np

    cols = list(columns) if columns else list(response.columns)
    data = response.to_lists(columns=cols, check=check)
    array_dtype = dtype if dtype is not None else object
    return np.asarray(data, dtype=array_dtype) if data else np.empty((0, len(cols)), dtype=array_dtype)


def _to_numpy_dict(response: Any, *, columns: Optional[List[str]] = None, check: bool = True) -> dict[str, Any]:
    """Return response rows as a dictionary of NumPy arrays."""
    import numpy as np

    cols = response._column_names(columns, check=check)
    rows = response.to_dicts(columns=cols, check=check)
    return {col: np.asarray([row.get(col) for row in rows]) for col in cols}
