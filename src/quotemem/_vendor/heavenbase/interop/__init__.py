"""Dependency-light public interop facade."""

from __future__ import annotations

from heavenbase._lazy import _attach_lazy_exports

__all__ = [
    "InteropIssue",
    "InteropReport",
    "InteropSet",
    "InteropTypeInferer",
    "from_daft",
    "from_numpy",
    "from_pandas",
    "from_pydantic",
    "from_pyarrow",
    "from_sql_result",
    "registered_interop_type_inferers",
    "register_interop_type_inferer",
    "row_from_any",
]

_LAZY_EXPORTS = {
    "from_daft": ("._loaders", "from_daft"),
    "from_numpy": ("._loaders", "from_numpy"),
    "from_pandas": ("._loaders", "from_pandas"),
    "from_pydantic": ("._loaders", "from_pydantic"),
    "from_pyarrow": ("._loaders", "from_pyarrow"),
    "from_sql_result": ("._loaders", "from_sql_result"),
    "InteropIssue": (".base", "InteropIssue"),
    "InteropReport": (".base", "InteropReport"),
    "InteropSet": (".base", "InteropSet"),
    "InteropTypeInferer": (".mapping", "InteropTypeInferer"),
    "registered_interop_type_inferers": (".mapping", "registered_interop_type_inferers"),
    "register_interop_type_inferer": (".mapping", "register_interop_type_inferer"),
    "row_from_any": (".rows", "row_from_any"),
}


__getattr__, __dir__ = _attach_lazy_exports(__name__, globals(), _LAZY_EXPORTS, __all__)
