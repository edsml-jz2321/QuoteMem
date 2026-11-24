"""NumPy table import helpers."""

from __future__ import annotations

__all__ = [
    "from_numpy",
]

from heavenbase.utils import Any, Mapping, Optional, Sequence

from ..modules import ModuleResolver
from .base import InteropSet, need
from .mapping import build_set
from .rows import rows_from_naive_table


def from_numpy(
    data: Any,
    *,
    columns: Optional[Sequence[str]] = None,
    entity_id: str,
    id: Optional[str] = None,
    overrides: Optional[Mapping[str, Any]] = None,
    rename: Optional[Mapping[str, str]] = None,
    desc: Optional[str] = None,
    tags: Optional[Sequence[str]] = None,
    field_meta: Optional[Mapping[str, Mapping[str, Any]]] = None,
    strict: bool = False,
    resolver: ModuleResolver | None = None,
) -> InteropSet:
    """Build an InteropSet from a NumPy structured or table-shaped array.

    Args:
        data (Any): Structured array or 1D/2D table-shaped NumPy array.
        columns (Sequence[str] | None): Column names for non-structured
            table-shaped arrays. Required when ``data`` has no field names.
        entity_id (str): Target entity identifier for the generated class.
        id (str | None): Source column mapped to ``object_id``. Defaults to
            ``"id"`` when present.
        overrides (Mapping[str, Any] | None): Per-field dtype or ``FieldSpec``
            overrides keyed by source or target name.
        rename (Mapping[str, str] | None): Source-to-target field renames.
        desc (str | None): Entity description for the generated class.
        tags (Sequence[str] | None): Entity tags for the generated class.
        field_meta (Mapping[str, Mapping[str, Any]] | None): Per-field
            description, examples, and inference hints.
        strict (bool): Promote warnings to errors. Defaults to ``False``.
        resolver (ModuleResolver | None): Context-owned module resolver. When
            omitted, the default Context is selected.

    Returns:
        InteropSet: Staged entity, normalized rows, and conversion report.

    Raises:
        TypeError: If NumPy is not installed.
        ValueError: If ``data`` is an unsupported shape or ``columns`` is
            missing for a non-structured array.
    """
    np = need("numpy", "hb.from_numpy()")

    arr = np.asarray(data)
    if arr.dtype.names:
        names = list(arr.dtype.names)
        rows = [{name: row[name] for name in names} for row in arr]
        hints = {name: arr.dtype[name] for name in names}
    else:
        if arr.ndim > 2:
            raise ValueError("from_numpy supports structured arrays or 1D/2D table-shaped arrays")
        if arr.ndim == 1:
            arr = arr.reshape((-1, 1))
        if columns is None:
            raise ValueError("from_numpy needs columns for non-structured arrays")
        rows = rows_from_naive_table(arr.tolist(), columns)
        hints = {str(name): arr.dtype for name in columns}
    return build_set(
        "numpy",
        rows,
        entity_id=entity_id,
        id=id,
        hints=hints,
        overrides=overrides,
        rename=rename,
        desc=desc,
        tags=tags,
        field_meta=field_meta,
        strict=strict,
        resolver=resolver,
    )
