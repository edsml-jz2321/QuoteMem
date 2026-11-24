"""Pydantic import/export helpers."""

from __future__ import annotations

__all__ = [
    "from_pydantic",
]

from heavenbase.utils import Any, Dict, Iterable, Mapping, Optional, Sequence

from ..modules import ModuleResolver
from .base import InteropSet
from .mapping import build_set
from .rows import row_from_any


def from_pydantic(
    model: Any,
    *,
    rows: Optional[Iterable[Any]] = None,
    entity_id: Optional[str] = None,
    id: Optional[str] = None,
    overrides: Optional[Mapping[str, Any]] = None,
    rename: Optional[Mapping[str, str]] = None,
    desc: Optional[str] = None,
    tags: Optional[Sequence[str]] = None,
    field_meta: Optional[Mapping[str, Mapping[str, Any]]] = None,
    strict: bool = False,
    resolver: ModuleResolver | None = None,
) -> InteropSet:
    """Build an InteropSet from a Pydantic model class and optional model rows.

    Args:
        model (Any): Pydantic model class defining source fields.
        rows (Iterable[Any] | None): Optional model instances or row mappings.
        entity_id (str | None): Target entity identifier. Defaults to a
            kebab-case form of the model class name.
        id (str | None): Source field mapped to ``object_id``.
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
        TypeError: If ``model`` is not a Pydantic model class.
    """
    model_fields = _model_fields(model)
    hints = {name: _field_hint(field) for name, field in model_fields.items()}
    meta = {**{name: _field_meta(field) for name, field in model_fields.items()}, **dict(field_meta or {})}
    required = {name: _field_required(field) for name, field in model_fields.items()}
    defaults = {name: _field_default(field) for name, field in model_fields.items() if _has_default(field)}
    data = [row_from_any(row) for row in rows or []]
    return build_set(
        "pydantic",
        data,
        entity_id=entity_id or _default_entity_id(model),
        id=id,
        hints=hints,
        overrides=overrides,
        rename=rename,
        desc=desc if desc is not None else getattr(model, "__doc__", "") or "",
        tags=tags,
        field_meta=meta,
        required=required,
        defaults=defaults,
        strict=strict,
        resolver=resolver,
    )


def _model_fields(model: Any) -> Mapping[str, Any]:
    fields = getattr(model, "model_fields", None) or getattr(model, "__fields__", None)
    if not fields:
        raise TypeError("from_pydantic expects a Pydantic model class")
    return fields


def _default_entity_id(model: Any) -> str:
    from ..utils.values import kebab

    return kebab(getattr(model, "__name__", "pydantic-model"))


def _field_hint(field: Any) -> Any:
    return getattr(field, "annotation", None) or getattr(field, "outer_type_", None) or getattr(field, "type_", None)


def _field_meta(field: Any) -> Dict[str, Any]:
    info = getattr(field, "field_info", field)
    desc = getattr(info, "description", None) or ""
    examples = getattr(info, "examples", None) or []
    meta = {"desc": desc, "examples": examples}
    max_length = _max_length(info)
    if max_length is not None:
        meta["max_length"] = max_length
    return meta


def _max_length(info: Any) -> Optional[int]:
    value = getattr(info, "max_length", None)
    if value is not None:
        return int(value)
    for item in getattr(info, "metadata", []) or []:
        value = getattr(item, "max_length", None)
        if value is not None:
            return int(value)
    return None


def _field_required(field: Any) -> bool:
    required = getattr(field, "is_required", None)
    if callable(required):
        return bool(required())
    if required is not None:
        return bool(required)
    return bool(getattr(field, "required", False))


def _has_default(field: Any) -> bool:
    if _field_required(field):
        return False
    default = getattr(field, "default", None)
    return default is not None and default.__class__.__name__ not in ("PydanticUndefinedType", "UndefinedType")


def _field_default(field: Any) -> Any:
    return getattr(field, "default", None)
