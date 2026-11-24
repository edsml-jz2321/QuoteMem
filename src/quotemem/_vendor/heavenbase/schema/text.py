"""Entity schema text-format registry and built-in renderers."""

from __future__ import annotations

__all__ = [
    "SchemaTextFn",
    "llm_field_lines",
    "llm_schema_text",
    "register_schema_text",
    "registered_schema_texts",
    "resolve_schema_text",
]

from heavenbase.utils import Any, Callable, Dict, List, Mapping

from ..modules import ModuleResolver, ModuleService, module_resolver
from ..modules.base import ResolverCache
from ..types import LogicalType
from ..utils.values import raise_mismatch
from heavenbase.utils.registry import named_key
from heavenbase.utils.registry import python_target
from ..utils.serialize import dumps_json
from .model import EntitySchema, FieldSchema

SchemaTextFn = Callable[[EntitySchema], str]


_SCHEMA_TEXT_CACHE = ResolverCache[SchemaTextFn]()


def _module_service(resolver: ModuleResolver) -> ModuleService:
    """Require lifecycle authority for one schema-text mutation."""
    if not isinstance(resolver, ModuleService):
        raise TypeError("Schema text registration requires a mutable ModuleService resolver")
    return resolver


def register_schema_text(
    name: str,
    fn: SchemaTextFn,
    *,
    replace: bool = False,
    resolver: ModuleResolver | None = None,
) -> None:
    """Register one entity schema text format.

    Args:
        name: Format identifier accepted by ``EntitySchema.to_str(format=...)``.
        fn: Callable that receives an ``EntitySchema`` and returns text.
        replace: Whether an existing format with the same name may be replaced.
        resolver: Context-owned module resolver. The default Context is
            selected when omitted. Registration requires a mutable
            `ModuleService` implementation.

    Raises:
        ValueError: If the name is blank or already registered and ``replace`` is false.
        TypeError: If ``fn`` is not callable.

    Returns:
        None: This function does not return a value."""
    key = named_key(name, "schema text format name")
    if not callable(fn):
        raise TypeError(f"Schema text format {key!r} must be callable")
    service = _module_service(module_resolver(resolver))
    found = service.inspect("schema_text", key)
    if found:
        existing = _schema_text_from_row(found[0], service)
        if existing is fn and replace:
            return
        if not replace:
            raise ValueError(f"Schema text format {key!r} is already registered")
        coordinate = found[0]["record"]["meta"]["installation"]["coordinate"]
        if coordinate != f"programmatic/schema_text/{key}":
            raise ValueError(f"Schema text format {key!r} belongs to {coordinate!r} and cannot be replaced")
    service.register(
        "schema_text",
        key,
        {
            "name": key,
            "desc": _doc_summary(fn),
            "source": "python-import",
            "target": python_target(fn, label=f"schema text format {key!r}"),
            "active": True,
            "meta": {"schema_version": 1, "tags": {"schema": True}},
        },
    )


def registered_schema_texts(*, resolver: ModuleResolver | None = None) -> Dict[str, SchemaTextFn]:
    """Return registered schema text formats keyed by name.

    Args:
        resolver: Context-owned module resolver. The default Context is
            selected when omitted.

    Returns:
        Dict[str, SchemaTextFn]: Registered renderers keyed by canonical name.
    """
    selected = module_resolver(resolver)
    return {str(row["identifier"]): _schema_text_from_row(row, selected) for row in selected.inspect("schema_text")}


def resolve_schema_text(name: str = "llm", *, resolver: ModuleResolver | None = None) -> SchemaTextFn:
    """Return one registered schema text format by name.

    Args:
        name: Format identifier accepted by ``EntitySchema.to_str(format=...)``.
        resolver: Context-owned module resolver. The default Context is
            selected when omitted.

    Returns:
        Callable that converts an ``EntitySchema`` to text.

    Raises:
        ValueError: If ``name`` is unknown.
    """
    key = str(name).strip()
    known = registered_schema_texts(resolver=resolver)
    if key not in known:
        raise_mismatch(sorted(known), key, name="schema text format", comment="Use Entity.schema().to_str(format=...).")
    return known[key]


def _schema_text_from_row(row: Mapping[str, Any], resolver: ModuleResolver) -> SchemaTextFn:
    """Materialize and cache one schema text renderer inspection row."""

    def build(current_row: Mapping[str, Any]) -> SchemaTextFn:
        fn = resolver.resolve("schema_text", str(current_row["identifier"]))
        if not callable(fn):
            raise TypeError(f"Schema text format {current_row['identifier']!r} did not resolve to a callable")
        return fn

    return _SCHEMA_TEXT_CACHE.get(resolver, row, build)


def _doc_summary(fn: Any) -> str:
    """Return the first documentation line for one renderer callable."""
    lines = str(getattr(fn, "__doc__", "") or "").strip().splitlines()
    return lines[0].strip() if lines else ""


def llm_schema_text(schema: EntitySchema) -> str:
    """Return the default LLM-friendly text for an entity schema.

    Args:
        schema: Entity schema to describe.

    Returns:
        Plain text that names the entity and each exact output field.
    """
    lines = [f"Entity schema: {schema.name} ({schema.entity_id})"]
    if schema.desc:
        lines.append(f"Description: {schema.desc}")
    if schema.tags:
        lines.append(f"Tags: {', '.join(schema.tags)}")
    lines.extend(["Use field names exactly.", "", "Fields:"])
    for field_schema in schema.fields.values():
        lines.extend(llm_field_lines(field_schema))
    return "\n".join(lines).strip()


def llm_field_lines(field_schema: FieldSchema) -> List[str]:
    """Return LLM-friendly text lines for one field schema.

    Args:
        field_schema (FieldSchema): Compiled field schema to describe.

    Returns:
        List[str]: Bullet and indented detail lines for one field.
    """
    attrs = []
    if field_schema.pk:
        attrs.append("primary key")
    attrs.append("required" if field_schema.required or field_schema.pk else "optional")
    if field_schema.has_default:
        attrs.append(f"default={_value_text(field_schema.default)}")
    lines = [f"- {field_schema.name}: {_dtype_text(field_schema.dtype)} ({'; '.join(attrs)})"]
    if field_schema.desc:
        lines.append(f"  Description: {field_schema.desc}")
    if field_schema.tags:
        lines.append(f"  Tags: {', '.join(field_schema.tags)}")
    if field_schema.examples:
        lines.append(f"  Examples: {_value_text(field_schema.examples)}")
    return lines


def _dtype_text(dtype: LogicalType) -> str:
    spec = dtype.to_dict()
    dtype_name = str(spec.get("type") or dtype.to_str())
    params = [f"{key}={_type_param_text(value)}" for key, value in spec.items() if key != "type" and value is not None]
    return f"{dtype_name}({', '.join(params)})" if params else dtype_name


def _type_param_text(value: Any) -> str:
    if isinstance(value, Mapping):
        return _dtype_mapping_text(value)
    return _value_text(value)


def _dtype_mapping_text(spec: Mapping[str, Any]) -> str:
    dtype_name = str(spec.get("type") or spec.get("identifier") or spec.get("dtype") or "type")
    params = [f"{key}={_type_param_text(value)}" for key, value in spec.items() if key not in {"type", "identifier", "dtype"} and value is not None]
    return f"{dtype_name}({', '.join(params)})" if params else dtype_name


def _value_text(value: Any) -> str:
    return dumps_json(value, compact=True)
