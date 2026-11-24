"""Logical type registration and discovery."""

from __future__ import annotations

__all__ = [
    "get_logical_type",
    "logical_type_aliases",
    "register_logical_type",
]

from heavenbase.utils import Any, Mapping, Sequence

from ..modules import ModuleResolver, ModuleService, module_resolver
from ..modules.base import ResolverCache
from ..utils.ids import check_id
from heavenbase.utils.registry import python_target
from heavenbase.utils.registry import class_identifier
from .base import LogicalType

_TYPE_CACHE = ResolverCache[type[LogicalType]]()


def _row(dtype: Any, resolver: ModuleResolver) -> dict[str, Any]:
    """Return the canonical inspection row for one type selector."""
    if isinstance(dtype, LogicalType):
        dtype = type(dtype)
    if isinstance(dtype, type):
        if not issubclass(dtype, LogicalType):
            raise TypeError(f"Expected logical type class, instance, identifier, or alias; got {dtype!r}")
        key = class_identifier(dtype, label="logical type identifier")
    elif isinstance(dtype, str):
        key = dtype.strip()
    else:
        raise TypeError(f"Expected logical type class, instance, identifier, or alias; got {dtype!r}")
    rows = resolver.inspect("logical_type", key)
    if not rows:
        check_id(key, "logical type identifier")
        raise KeyError(f"Unknown logical type {key!r}")
    return rows[0]


def _from_row(row: dict[str, Any], resolver: ModuleResolver) -> type[LogicalType]:
    """Materialize and validate one logical-type inspection row."""

    def build(current_row: Mapping[str, Any]) -> type[LogicalType]:
        identifier = str(current_row["identifier"])
        restored = resolver.resolve("logical_type", identifier)
        if not isinstance(restored, type) or not issubclass(restored, LogicalType):
            raise TypeError(f"Logical type {identifier!r} did not resolve to a LogicalType subclass")
        actual = class_identifier(restored, label="logical type identifier")
        if actual != identifier:
            raise ValueError(f"Logical type record {identifier!r} resolved class identifier {actual!r}")
        return restored

    return _TYPE_CACHE.get(resolver, row, build)


def _module_service(resolver: ModuleResolver) -> ModuleService:
    """Require lifecycle authority for one registration mutation."""
    if not isinstance(resolver, ModuleService):
        raise TypeError("Logical type registration requires a mutable ModuleService resolver")
    return resolver


def register_logical_type(
    dtype: type[LogicalType],
    tags: Mapping[str, Any] | None = None,
    aliases: Sequence[str] = (),
    *,
    resolver: ModuleResolver | None = None,
) -> type[LogicalType]:
    """Register one logical type class.

    Args:
        dtype (type[LogicalType]): Logical type subclass to register.
        tags (Mapping[str, Any] | None): Raw JSON-compatible Tag claims copied
            onto the published logical-type record. Definitions are optional
            and registration never creates them.
        aliases (Sequence[str]): Additional accepted descriptor names.
        resolver (ModuleResolver | None): Context-owned module resolver. The
            default Context is selected when omitted. Registration requires a
            mutable `ModuleService` implementation.

    Returns:
        type[LogicalType]: Registered logical type class.

    Raises:
        TypeError: If `dtype` is not a `LogicalType` subclass.
        ValueError: If its identifier is invalid.
    """
    if not isinstance(dtype, type) or not issubclass(dtype, LogicalType):
        raise TypeError("register_logical_type expects a LogicalType subclass")
    identifier = class_identifier(dtype, label="logical type identifier")
    dtype.identifier = identifier
    service = _module_service(module_resolver(resolver))
    found = service.inspect("logical_type", identifier)
    existing_aliases = tuple(found[0]["aliases"]) if found else ()
    if tags is not None and not isinstance(tags, Mapping):
        raise TypeError("Logical type tags must be a string-keyed mapping")
    authored_tags = dict(tags or {})
    if not all(isinstance(tag_identifier, str) for tag_identifier in authored_tags):
        raise TypeError("Logical type tags must be a string-keyed mapping")
    existing_meta = found[0]["record"].get("meta") if found else {}
    existing_tags = existing_meta.get("tags", {}) if isinstance(existing_meta, Mapping) else {}
    if not isinstance(existing_tags, Mapping):
        raise TypeError(f"Logical type {identifier!r} record meta.tags must be a mapping")
    alias_values = tuple(dict.fromkeys((*existing_aliases, *aliases, dtype.__name__)))
    tag_values = {**existing_tags, **authored_tags}
    if found:
        restored = _from_row(found[0], service)
        if restored is dtype and alias_values == existing_aliases and tag_values == existing_tags:
            return dtype
        coordinate = found[0]["record"]["meta"]["installation"]["coordinate"]
        if coordinate != f"programmatic/logical_type/{identifier}":
            raise ValueError(f"Logical type {identifier!r} belongs to {coordinate!r} and cannot be changed through register_logical_type")
    service.register(
        "logical_type",
        identifier,
        {
            "name": dtype.__name__,
            "desc": (dtype.__doc__ or "Registered logical type.").strip().splitlines()[0],
            "source": "python-import",
            "target": python_target(dtype, label=f"Logical type {identifier!r}"),
            "active": True,
            "meta": {"schema_version": 1, **({"tags": tag_values} if tag_values else {})},
        },
        aliases=alias_values,
        coordinate=f"programmatic/logical_type/{identifier}",
    )
    return _from_row(service.inspect("logical_type", identifier)[0], service)


def get_logical_type(dtype: Any, *, resolver: ModuleResolver | None = None) -> type[LogicalType]:
    """Load a registered logical type class.

    Args:
        dtype (Any): Logical type class, instance, identifier, or alias.
        resolver (ModuleResolver | None): Context-owned module resolver. The
            default Context is selected when omitted.

    Returns:
        type[LogicalType]: Registered logical type class.

    Raises:
        KeyError: If the logical type is not registered.
        TypeError: If `dtype` cannot identify a logical type.
    """
    selected = module_resolver(resolver)
    return _from_row(_row(dtype, selected), selected)


def logical_type_aliases(dtype: Any, *, resolver: ModuleResolver | None = None) -> tuple[str, ...]:
    """Return accepted aliases for a registered logical type.

    Args:
        dtype (Any): Logical type class, instance, identifier, or alias.
        resolver (ModuleResolver | None): Context-owned module resolver. The
            default Context is selected when omitted.

    Returns:
        tuple[str, ...]: Registered aliases excluding the canonical id.
    """
    selected = module_resolver(resolver)
    return tuple(str(alias) for alias in _row(dtype, selected)["aliases"])
