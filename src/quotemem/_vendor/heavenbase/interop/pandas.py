"""Pandas DataFrame import helpers."""

from __future__ import annotations

__all__ = [
    "from_pandas",
]

from heavenbase.utils import Any, Mapping, Optional, Sequence

from ..modules import ModuleResolver
from .base import InteropIssue, InteropSet, need
from .mapping import build_set


def from_pandas(
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
    """Build an InteropSet from a Pandas DataFrame.

    Args:
        df (Any): ``pandas.DataFrame`` source table.
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
        TypeError: If ``df`` is not a ``pandas.DataFrame``.
        ImportError: If Pandas is not installed.
    """
    pd = need("pandas", "hb.from_pandas()")

    if not isinstance(df, pd.DataFrame):
        raise TypeError("from_pandas expects a pandas.DataFrame")
    names = [str(col) for col in df.columns]
    dupes = sorted({name for name in names if names.count(name) > 1})
    issues = [InteropIssue("error", "duplicate_source_column", "columns", f"Pandas DataFrame has duplicate columns: {dupes}")] if dupes else []
    rows = df.to_dict("records")
    hints = {}
    for pos, name in enumerate(names):
        hints.setdefault(name, df.iloc[:, pos].dtype)
    return build_set(
        "pandas",
        rows,
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
