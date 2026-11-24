"""Dependency-light public interop loader functions."""

from __future__ import annotations

__all__ = [
    "from_daft",
    "from_numpy",
    "from_pandas",
    "from_pydantic",
    "from_pyarrow",
    "from_sql_result",
]

from heavenbase.utils import Any


def from_pydantic(*args: Any, **kwargs: Any) -> Any:
    """Build an InteropSet from a Pydantic model class and optional rows."""
    from .pydantic import from_pydantic as _from_pydantic

    return _from_pydantic(*args, **kwargs)


def from_pandas(*args: Any, **kwargs: Any) -> Any:
    """Build an InteropSet from a Pandas DataFrame."""
    from .pandas import from_pandas as _from_pandas

    return _from_pandas(*args, **kwargs)


def from_pyarrow(*args: Any, **kwargs: Any) -> Any:
    """Build an InteropSet from a PyArrow Table."""
    from .pyarrow import from_pyarrow as _from_pyarrow

    return _from_pyarrow(*args, **kwargs)


def from_numpy(*args: Any, **kwargs: Any) -> Any:
    """Build an InteropSet from a NumPy array."""
    from .numpy import from_numpy as _from_numpy

    return _from_numpy(*args, **kwargs)


def from_sql_result(*args: Any, **kwargs: Any) -> Any:
    """Build an InteropSet from SQL-like rows or results."""
    from .sql import from_sql_result as _from_sql_result

    return _from_sql_result(*args, **kwargs)


def from_daft(*args: Any, **kwargs: Any) -> Any:
    """Build an InteropSet from a Daft DataFrame."""
    from .daft import from_daft as _from_daft

    return _from_daft(*args, **kwargs)
