"""Daft DataFrame import helpers."""

from __future__ import annotations

__all__ = [
    "from_daft",
]

from heavenbase.utils import Any, Mapping, Optional, Sequence

from ..modules import ModuleResolver
from .base import InteropIssue, InteropSet
from .mapping import build_set


def _need_daft(feature: str) -> Any:
    try:
        import daft
    except ImportError as exc:
        raise ImportError(f"daft is required for {feature}; install heavenbase[daft].") from exc
    return daft


def from_daft(
    df: Any,
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
    """Build an InteropSet from a Daft DataFrame via PyArrow.

    Args:
        df (Any): ``daft.DataFrame`` source table.
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
        TypeError: If ``df`` is not a ``daft.DataFrame`` or lacks
            ``to_arrow()``.
        ImportError: If Daft or PyArrow is not installed.
    """
    daft = _need_daft("hb.from_daft()")
    pa = __import__("pyarrow")

    data_cls = getattr(daft, "DataFrame", None)
    if data_cls is None or not isinstance(df, data_cls):
        raise TypeError("from_daft expects a daft.DataFrame")

    if hasattr(df, "to_arrow"):
        table = df.to_arrow()
    else:
        raise TypeError("Daft DataFrame does not support to_arrow()")

    if not isinstance(table, pa.Table):
        raise TypeError("from_daft expected a pyarrow.Table from Daft")

    names = list(table.schema.names)
    dupes = sorted({name for name in names if names.count(name) > 1})
    issues = [InteropIssue("error", "duplicate_source_column", "columns", f"Daft DataFrame has duplicate columns: {dupes}")] if dupes else []
    hints = {field.name: field.type for field in table.schema}
    iset = build_set(
        "daft",
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
    iset.report.source = "daft"
    return iset
