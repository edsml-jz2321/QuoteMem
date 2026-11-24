"""JSON/dict entity definition compiler."""

from __future__ import annotations

__all__ = []

import datetime as dt
import re

from heavenbase.utils import Any, Callable, List, Mapping, Optional, Tuple

from ..modules import ModuleResolver
from ..strategies import InlineColumn, StorageStrategy
from ..types import LogicalType, logical_type_from_any
from ..utils.values import class_name, coerce_list, kebab
from ..utils.ids import check_entity_id, check_field_name, check_py_name
from ..filters.ops import normalize_op
from .base import Entity
from .computes import COMPUTE_KIND, QUERY_COMPUTE_KIND, _resolve as _resolve_compute, registered_computes, registered_query_computes
from .fields import FieldSpec, field as make_field
from ..placement import IndexRule, UniqueRule

ComputeRegistry = Optional[Mapping[str, Callable[..., Any]]]
QueryComputeRegistry = Optional[Mapping[str, Callable[[Any], Any]]]


def _entity_from_schema(
    spec: Mapping[str, Any],
    *,
    computes: ComputeRegistry = None,
    query_computes: QueryComputeRegistry = None,
    resolver: ModuleResolver,
) -> type:
    """Build an Entity subclass from a JSON-compatible mapping."""
    if not isinstance(spec, Mapping):
        raise TypeError("Entity JSON definition must be a mapping")
    if "id" in spec:
        raise ValueError("Entity JSON definition uses entity_id, not id")
    entity_id = str(spec.get("entity_id") or "").strip()
    raw_name = str(spec.get("name") or class_name(entity_id)).strip()
    if not raw_name:
        raise ValueError("Entity JSON definition needs name or entity_id")
    name = check_py_name(raw_name, "entity name")
    entity_id = check_entity_id(entity_id or kebab(name))
    card = spec.get("card") if isinstance(spec.get("card"), Mapping) else {}
    desc = str(spec.get("desc") or spec.get("description") or card.get("short_desc") or card.get("desc") or card.get("long_desc") or "")
    tags = coerce_list(spec.get("tags", card.get("tags", [])), "entity tags")
    fields = spec.get("fields")
    if not fields:
        raise ValueError(f"Entity {entity_id!r} needs at least one field")

    namespace = {"identifier": entity_id, "__tags__": tags, "__doc__": desc, "__module__": "heavenbase.dynamic"}
    seen_fields = set()
    for field_name, field_spec in _field_items(fields):
        field_name = check_field_name(field_name)
        if field_name in seen_fields:
            raise ValueError(f"Duplicate field name {field_name!r} in entity {entity_id!r}")
        seen_fields.add(field_name)
        namespace[field_name] = _field_from_dict(field_name, field_spec, computes, query_computes, resolver)
    derives = [_derive_from_any(item, computes, resolver) for item in _derive_items(spec)]
    if derives:
        namespace["__derive__"] = derives
    indexes = [_index_from_any(item) for item in coerce_list(spec.get("indexes", []), "indexes")]
    if indexes:
        namespace["__indexes__"] = indexes
    constraints = [_unique_from_any(item) for item in coerce_list(spec.get("constraints", []), "constraints")]
    if constraints:
        namespace["__constraints__"] = constraints
    entity_cls = type(name, (Entity,), namespace)
    entity_cls._hb_json_spec = _json_spec(spec, resolver)
    return entity_cls


def _json_spec(spec: Mapping[str, Any], resolver: ModuleResolver) -> dict:
    """Return a JSON-serializable copy of an entity definition spec."""
    out = dict(spec)
    fields = spec.get("fields")
    if isinstance(fields, Mapping):
        out["fields"] = {name: _json_field_spec(raw, resolver) for name, raw in fields.items()}
    elif isinstance(fields, list):
        out["fields"] = [{**item, **({"type": _json_type_spec(item["type"])} if "type" in item else {})} for item in fields]
    return out


def _json_field_spec(raw: Any, resolver: ModuleResolver) -> Any:
    if isinstance(raw, str):
        return raw
    if not isinstance(raw, Mapping):
        return raw
    spec = dict(raw)
    if "type" in spec:
        spec["type"] = _json_type_spec(spec["type"])
    if "dtype" in spec:
        spec["dtype"] = _json_type_spec(spec["dtype"])
    if "storage" in spec:
        spec["storage"] = _json_storage_spec(spec["storage"], resolver)
    if "stores" in spec and isinstance(spec["stores"], list):
        spec["stores"] = [_json_storage_spec(item, resolver) for item in spec["stores"]]
    return spec


def _json_storage_spec(raw: Any, resolver: ModuleResolver) -> Any:
    if not isinstance(raw, Mapping):
        return raw
    spec = dict(raw)
    if "strategy" in spec:
        spec["strategy"] = StorageStrategy.spec_of(spec["strategy"], resolver=resolver)
    return spec


def _json_type_spec(raw: Any) -> Any:
    if isinstance(raw, LogicalType):
        return raw.to_dict()
    if isinstance(raw, type) and issubclass(raw, LogicalType):
        return raw().to_dict()
    return raw


# A definition is JSON, so ``to_manifest()`` writes a date or datetime default or example as its ``isoformat()`` text, with
# the type named beside it under "default_type" or "example_types". Only that text is restored, checked before it is
# parsed, because ``fromisoformat`` accepts more forms on Python 3.11 and later than on 3.10.
_DATED_TYPES: dict[str, Tuple[type, Any]] = {
    "datetime.date": (dt.date, re.compile(r"\d{4}-\d{2}-\d{2}", re.ASCII)),
    "datetime.datetime": (
        dt.datetime,
        re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{6})?(?:[+-]\d{2}:\d{2}(?::\d{2}(?:\.\d{6})?)?)?", re.ASCII),
    ),
}


def _encoded_value(value: Any) -> Tuple[Any, Optional[str]]:
    """Return ``value`` as a JSON definition holds it, and its type name when that is needed to restore it."""
    for name, (kind, _) in _DATED_TYPES.items():
        if type(value) is kind:
            return value.isoformat(), name
    return value, None


def _decoded_value(value: Any, name: Any) -> Any:
    """Return the value ``_encoded_value`` wrote as ``value``; any other value is returned as it is."""
    kind, pattern = _DATED_TYPES.get(name, (None, None)) if isinstance(name, str) else (None, None)
    if kind is None or not isinstance(value, str) or not pattern.fullmatch(value):
        return value
    try:
        return kind.fromisoformat(value)
    except ValueError:
        return value


def _encoded_field(raw: Any) -> Any:
    """Return one field specification with its date and datetime default and examples as a JSON definition holds them."""
    if not isinstance(raw, Mapping):
        return raw
    out = dict(raw)
    default, default_type = _encoded_value(raw.get("default"))
    if default_type is not None:
        out.update(default=default, default_type=default_type)
    examples = raw.get("examples")
    if isinstance(examples, (list, tuple)):
        encoded = [_encoded_value(example) for example in examples]
        if any(name is not None for _, name in encoded):
            known = raw.get("example_types")
            known = known if isinstance(known, list) and len(known) == len(encoded) else [None] * len(encoded)
            out["examples"] = [value for value, _ in encoded]
            out["example_types"] = [name or before for (_, name), before in zip(encoded, known)]
    return out


def _decoded_field(raw: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return one field specification with the default and examples ``_encoded_field`` wrote restored."""
    out = dict(raw)
    if "default" in raw:
        out["default"] = _decoded_value(raw["default"], raw.get("default_type"))
    examples, names = raw.get("examples"), raw.get("example_types")
    if isinstance(examples, list) and isinstance(names, list) and len(examples) == len(names):
        out["examples"] = [_decoded_value(example, name) for example, name in zip(examples, names)]
    return out


def _field_from_dict(
    field_name: str,
    raw: Any,
    computes: ComputeRegistry,
    query_computes: QueryComputeRegistry,
    resolver: ModuleResolver,
) -> FieldSpec:
    if isinstance(raw, str):
        raw = {"type": raw}
    if not isinstance(raw, Mapping):
        raise TypeError(f"Field {field_name!r} definition must be a mapping or type string")
    raw = _decoded_field(raw)
    dtype = logical_type_from_any(raw.get("type", raw.get("dtype")), resolver=resolver)
    card = raw.get("card") if isinstance(raw.get("card"), Mapping) else {}
    desc = str(raw.get("desc") or raw.get("description") or card.get("short_desc") or card.get("desc") or card.get("long_desc") or "")
    spec = make_field(
        dtype,
        desc=desc,
        tags=coerce_list(raw.get("tags", card.get("tags", [])), f"{field_name}.tags"),
        examples=coerce_list(raw.get("examples", card.get("examples", [])), f"{field_name}.examples"),
    )
    if raw.get("pk"):
        spec.pk()
    if "required" in raw:
        spec.required() if raw.get("required") else spec.optional()
    if raw.get("has_default") or ("default" in raw and raw.get("default") is not None):
        spec.default(raw.get("default"))
    for rule in _storage_items(raw):
        spec.store(
            to=rule.get("to", rule.get("backend")),
            strategy=rule.get("strategy", InlineColumn.identifier),
            optional=bool(rule.get("optional", False)),
            reason=rule.get("reason", "json"),
            params=rule.get("params"),  # restore a parametrised strategy's params (e.g. GraphEdge target) the serialized strategy id alone cannot carry
            resolver=resolver,
        )
    for rule in _field_index_items(raw):
        spec.index(bool(rule.get("enabled", True)), to=rule.get("to", rule.get("backend")))
    for rule in _field_unique_items(raw):
        spec.unique(to=rule.get("to", rule.get("backend")))
    for compute_spec in _compute_items(raw):
        fn, inputs = _compute_from_any(compute_spec, computes, field_name, resolver)
        spec.compute(fn, inputs=inputs)
    for query_compute_spec in _query_compute_items(raw):
        fn, ops = _query_compute_from_any(query_compute_spec, query_computes, field_name, resolver)
        spec.query_compute(fn, ops=ops, resolver=resolver)
    return spec


def _field_items(fields: Any) -> list[tuple[str, Any]]:
    if isinstance(fields, Mapping):
        return [(str(name), value) for name, value in fields.items()]
    if isinstance(fields, list):
        out = []
        for item in fields:
            if not isinstance(item, Mapping) or not item.get("name"):
                raise ValueError("List-style fields must be mappings with a name")
            out.append((str(item["name"]), item))
        return out
    raise TypeError("Entity fields must be a mapping or a list")


def _storage_items(raw: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    items = []
    if "storage" in raw:
        items.append(raw["storage"])
    items.extend(coerce_list(raw.get("stores", []), "stores"))
    out = []
    for item in items:
        if isinstance(item, str):
            item = {"to": item}
        if not isinstance(item, Mapping):
            raise TypeError("Each storage placement must be a mapping or backend name string")
        out.append(item)
    return out


def _field_index_items(raw: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Return normalized field-level scalar index declarations."""
    items: list[Any] = []
    if "index" in raw:
        items.append(raw["index"])
    items.extend(coerce_list(raw.get("indexes", []), "indexes"))
    out = []
    for item in items:
        if isinstance(item, bool):
            item = {"enabled": item}
        if not isinstance(item, Mapping):
            raise TypeError("Field index must be a boolean or mapping; mode strings are not supported")
        out.append(item)
    return out


def _field_unique_items(raw: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Return normalized field-level uniqueness declarations."""
    if "unique" not in raw:
        return []
    item = raw["unique"]
    if item is False or item is None:
        return []
    if item is True:
        return [{}]
    if not isinstance(item, Mapping):
        raise TypeError("Field unique must be a boolean or mapping")
    return [item]


def _index_from_any(raw: Any) -> IndexRule:
    """Build one entity-level scalar index rule from JSON data."""
    if not isinstance(raw, Mapping):
        raise TypeError("Each entity index must be a mapping with fields")
    fields = coerce_list(raw.get("fields", []), "index fields")
    return IndexRule(tuple(fields), raw.get("to", raw.get("backend")), bool(raw.get("enabled", True)))


def _unique_from_any(raw: Any) -> UniqueRule:
    """Build one entity-level uniqueness rule from JSON data."""
    if not isinstance(raw, Mapping):
        raise TypeError("Each entity constraint must be a mapping with fields")
    fields = coerce_list(raw.get("fields", []), "constraint fields")
    return UniqueRule(tuple(fields), raw.get("to", raw.get("backend")))


def _compute_items(raw: Mapping[str, Any]) -> list[Any]:
    items = []
    if "compute" in raw:
        items.append(raw["compute"])
    items.extend(coerce_list(raw.get("computes", []), "computes"))
    return items


def _query_compute_items(raw: Mapping[str, Any]) -> list[Any]:
    items = []
    if "query_compute" in raw:
        items.append(raw["query_compute"])
    items.extend(coerce_list(raw.get("query_computes", []), "query_computes"))
    return items


def _derive_items(raw: Mapping[str, Any]) -> list[Any]:
    items = []
    if "derive" in raw:
        item = raw["derive"]
        items.extend(item if isinstance(item, list) else [item])
    items.extend(coerce_list(raw.get("derives", []), "derives"))
    return items


def _derive_from_any(raw: Any, computes: ComputeRegistry, resolver: ModuleResolver):
    if not isinstance(raw, Mapping):
        raise TypeError("Each derive item must be a mapping")
    target = str(raw.get("target") or "").strip()
    if not target:
        raise ValueError("Each derive item needs target")
    if "compute" not in raw and "fn" not in raw and "name" not in raw:
        raise ValueError(f"Derive target {target!r} needs compute")
    compute_spec = raw.get("compute") if "compute" in raw else raw
    if isinstance(compute_spec, Mapping):
        compute_spec = dict(compute_spec)
        if "inputs" not in compute_spec and "fields" not in compute_spec:
            compute_spec["inputs"] = raw.get("inputs", raw.get("fields", []))
    else:
        compute_spec = {"fn": compute_spec, "inputs": raw.get("inputs", raw.get("fields", []))}
    fn, inputs = _compute_from_any(compute_spec, computes, target, resolver)
    return Entity.derive(target).compute(fn, inputs=inputs)


def _compute_from_any(raw: Any, computes: ComputeRegistry, field_name: str, resolver: ModuleResolver) -> tuple[Callable[..., Any], list[str]]:
    inputs: list[str] = []
    fn_ref = raw
    if isinstance(raw, Mapping):
        fn_ref = raw.get("fn", raw.get("name"))
        raw_inputs = raw.get("inputs", raw.get("fields", []))
        inputs = [check_field_name(input_name) for input_name in coerce_list(raw_inputs, f"{field_name}.compute.inputs")]
    if callable(fn_ref):
        return fn_ref, inputs
    if isinstance(fn_ref, str):
        if computes and fn_ref in computes:
            return computes[fn_ref], inputs
        fn = _resolve_compute(COMPUTE_KIND, fn_ref, resolver=resolver)
        if fn is not None:
            return fn, inputs
        raise KeyError(f"Compute function {fn_ref!r} for field {field_name!r} is not registered; known={list(registered_computes(resolver=resolver))}")
    raise TypeError(f"Compute for field {field_name!r} needs a callable or registered function name")


def _query_compute_from_any(
    raw: Any,
    query_computes: QueryComputeRegistry,
    field_name: str,
    resolver: ModuleResolver,
) -> tuple[Callable[[Any], Any], list[str]]:
    ops = ["near"]
    fn_ref = raw
    if isinstance(raw, Mapping):
        fn_ref = raw.get("fn", raw.get("name"))
        raw_ops = raw.get("ops", raw.get("op", ["near"]))
        ops = [
            normalize_op(op, resolver=resolver) for op in ([raw_ops] if isinstance(raw_ops, str) else coerce_list(raw_ops, f"{field_name}.query_compute.ops"))
        ]
    if callable(fn_ref):
        return fn_ref, ops
    if isinstance(fn_ref, str):
        if query_computes and fn_ref in query_computes:
            return query_computes[fn_ref], ops
        fn = _resolve_compute(QUERY_COMPUTE_KIND, fn_ref, resolver=resolver)
        if fn is not None:
            return fn, ops
        known = list(registered_query_computes(resolver=resolver))
        raise KeyError(f"Query compute function {fn_ref!r} for field {field_name!r} is not registered; known={known}")
    raise TypeError(f"Query compute for field {field_name!r} needs a callable or registered function name")
