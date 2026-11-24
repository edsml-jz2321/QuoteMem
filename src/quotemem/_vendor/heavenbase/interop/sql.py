"""SQL-like result import helpers."""

from __future__ import annotations

__all__ = [
    "from_sql_result",
]

from heavenbase.utils import Any, Mapping, Optional, Sequence

from ..modules import ModuleResolver
from .base import InteropSet
from .mapping import build_set
from .rows import rows_from_naive_table, source_row


def from_sql_result(
    result: Any,
    *,
    entity_id: str,
    id: Optional[str] = None,
    columns: Optional[Sequence[str]] = None,
    overrides: Optional[Mapping[str, Any]] = None,
    rename: Optional[Mapping[str, str]] = None,
    desc: Optional[str] = None,
    tags: Optional[Sequence[str]] = None,
    field_meta: Optional[Mapping[str, Mapping[str, Any]]] = None,
    strict: bool = False,
    resolver: ModuleResolver | None = None,
) -> InteropSet:
    """Build an InteropSet from SQLAlchemy, SQLResponse, or naive table rows.

    Args:
        result (Any): SQLAlchemy result, ``SQLResponse``, mapping, or
            table-shaped row sequence.
        entity_id (str): Target entity identifier for the generated class.
        id (str | None): Source column mapped to ``object_id``. Defaults to
            ``"id"`` when present.
        columns (Sequence[str] | None): Column names for naive table-shaped
            row sequences.
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
        ValueError: If a ``SQLResponse`` payload is not ok.
    """
    rows = _rows(result, columns=columns)
    hints = {name: None for name in (columns or (rows[0].keys() if rows else []))}
    return build_set(
        "sql",
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


def _rows(result: Any, *, columns: Optional[Sequence[str]] = None):
    if hasattr(result, "ok") and hasattr(result, "to_list"):
        if not result.ok:
            raise ValueError(getattr(result, "error_message", "SQLResponse is not ok"))
        return [source_row(row) for row in result.to_list(row_fmt="dict")]
    if hasattr(result, "mappings"):
        return [source_row(dict(row)) for row in result.mappings()]
    if hasattr(result, "_mapping"):
        return [source_row(dict(result._mapping))]
    if isinstance(result, Mapping):
        return [source_row(result)]
    return rows_from_naive_table(list(result), columns=columns)
