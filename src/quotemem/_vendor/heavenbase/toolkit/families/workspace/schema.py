"""Schema-normalization helpers for workspace MCP tools."""

from __future__ import annotations

__all__ = ["normalize_entity_spec"]

from heavenbase.utils import Any, Dict, Mapping

_JSON_SCHEMA_TYPE_ALIASES = {"string": "short-text", "number": "float", "object": "json"}


def normalize_entity_spec(entity: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize common JSON-schema-style entity specs for MCP callers.

    Args:
        entity (Dict[str, Any]): Entity definition mapping, often using ``properties``, ``title``, or JSON Schema type aliases.

    Returns:
        Dict[str, Any]: HeavenBase entity spec with normalized ``fields``, ``name``, and logical dtype aliases.

    Raises:
        ValueError: If ``entity`` looks like a bare field map instead of an entity object.
    """
    if not isinstance(entity, Mapping):
        return entity
    spec = dict(entity)
    if "fields" not in spec and isinstance(spec.get("properties"), Mapping):
        spec["fields"] = spec["properties"]
    if not spec.get("name") and spec.get("title"):
        spec["name"] = spec["title"]
    if "fields" not in spec and _looks_like_field_map(spec):
        raise ValueError("define_entity expects an entity object such as {'name': 'Product', 'fields': {...}}, not a bare field map")
    required = set(spec.get("required") or []) if isinstance(spec.get("required"), list) else set()
    fields = spec.get("fields")
    if isinstance(fields, Mapping):
        spec["fields"] = {str(name): _normalize_field_spec(str(name), raw, required) for name, raw in fields.items()}
    elif isinstance(fields, list):
        spec["fields"] = [_normalize_list_field_spec(raw, required) for raw in fields]
    return spec


def _looks_like_field_map(spec: Mapping[str, Any]) -> bool:
    return bool(spec) and "fields" not in spec and "properties" not in spec and all(_looks_like_field_spec(raw) for raw in spec.values())


def _looks_like_field_spec(raw: Any) -> bool:
    return isinstance(raw, str) or (isinstance(raw, Mapping) and any(key in raw for key in ("type", "dtype", "description", "desc", "required", "pk")))


def _normalize_list_field_spec(raw: Any, required: set[str]) -> Any:
    if not isinstance(raw, Mapping):
        return raw
    name = str(raw.get("name") or "")
    out = dict(_normalize_field_spec(name, raw, required))
    if name:
        out["name"] = name
    return out


def _normalize_field_spec(field_name: str, raw: Any, required: set[str]) -> Any:
    if isinstance(raw, str):
        dtype = _normalize_type_alias(raw)
        return "identifier" if field_name == "object_id" and dtype == "short-text" else dtype
    if not isinstance(raw, Mapping):
        return raw
    out = dict(raw)
    for key in ("type", "dtype"):
        if key in out:
            out[key] = _normalize_type_alias(out[key])
            if field_name == "object_id" and out[key] == "short-text":
                out[key] = "identifier"
    if field_name in required and "required" not in out:
        out["required"] = True
    return out


def _normalize_type_alias(raw: Any) -> Any:
    if isinstance(raw, str):
        return _JSON_SCHEMA_TYPE_ALIASES.get(raw.strip().lower(), raw)
    if isinstance(raw, Mapping):
        out = dict(raw)
        for key in ("type", "dtype", "items", "item_type"):
            if key in out:
                out[key] = _normalize_type_alias(out[key])
        return out
    return raw
