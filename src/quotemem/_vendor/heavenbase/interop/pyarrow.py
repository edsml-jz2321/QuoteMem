"""PyArrow Table import helpers."""

from __future__ import annotations

__all__ = [
    "from_pyarrow",
]

from heavenbase.utils import Any, Mapping, Optional, Sequence

from ..modules import ModuleResolver
from .base import InteropIssue, InteropSet, need
from .mapping import build_set


def from_pyarrow(
    table: Any,
    *,
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
    """Build an InteropSet from a PyArrow Table.

    Args:
        table (Any): ``pyarrow.Table`` source table.
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
        TypeError: If ``table`` is not a ``pyarrow.Table``.
        ImportError: If PyArrow is not installed.
    """
    pa = need("pyarrow", "hb.from_pyarrow()")

    if not isinstance(table, pa.Table):
        raise TypeError("from_pyarrow expects a pyarrow.Table")
    names = list(table.schema.names)
    dupes = sorted({name for name in names if names.count(name) > 1})
    issues = [InteropIssue("error", "duplicate_source_column", "columns", f"PyArrow Table has duplicate columns: {dupes}")] if dupes else []
    hints = {field.name: field.type for field in table.schema}
    return build_set(
        "pyarrow",
        table.to_pylist(),
        entity_id=entity_id,
        id=id,
        hints=hints,
        overrides=overrides,
        rename=rename,
        desc=desc,
        tags=tags,
        field_meta=field_meta,
        issues=issues,
        strict=strict,
        resolver=resolver,
    )
