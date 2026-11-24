"""Storage strategy registration and discovery."""

from __future__ import annotations

__all__ = [
    "register_strategy",
    "strategy_aliases",
]

from heavenbase.utils import Any, Mapping, Sequence

from ..modules import ModuleResolver, ModuleService, module_resolver
from ..modules.base import ResolverCache
from ..utils.ids import check_id
from heavenbase.utils.registry import python_target
from heavenbase.utils.registry import class_identifier
from .base import StorageStrategy

_STRATEGY_CACHE = ResolverCache[type[StorageStrategy]]()


def _row(strategy: Any, resolver: ModuleResolver) -> dict[str, Any]:
    """Return the canonical inspection row for one strategy selector."""
    if isinstance(strategy, Mapping):
        strategy = StorageStrategy._mapping_identifier(strategy)
    if isinstance(strategy, StorageStrategy):
        strategy = type(strategy)
    if isinstance(strategy, type):
        if not issubclass(strategy, StorageStrategy):
            raise TypeError(f"Expected storage strategy class, value, identifier, alias, or spec; got {strategy!r}")
        key = class_identifier(strategy, label="strategy identifier")
    elif isinstance(strategy, str):
        key = strategy.strip()
    else:
        raise TypeError(f"Invalid storage strategy identifier {strategy!r}; expected class, value, alias, identifier, or spec")
    rows = resolver.inspect("strategy", key)
    if not rows:
        check_id(key, "strategy identifier")
        raise KeyError(f"Unknown storage strategy {key!r}")
    return rows[0]


def _from_row(row: dict[str, Any], resolver: ModuleResolver) -> type[StorageStrategy]:
    """Materialize and validate one strategy inspection row."""

    def build(current_row: Mapping[str, Any]) -> type[StorageStrategy]:
        identifier = str(current_row["identifier"])
        restored = resolver.resolve("strategy", identifier)
        if not isinstance(restored, type) or not issubclass(restored, StorageStrategy):
            raise TypeError(f"Storage strategy {identifier!r} did not resolve to a StorageStrategy subclass")
        actual = class_identifier(restored, label="strategy identifier")
        if actual != identifier:
            raise ValueError(f"Storage strategy record {identifier!r} resolved class identifier {actual!r}")
        return restored

    return _STRATEGY_CACHE.get(resolver, row, build)


def _module_service(resolver: ModuleResolver) -> ModuleService:
    """Require lifecycle authority for one registration mutation."""
    if not isinstance(resolver, ModuleService):
        raise TypeError("Storage strategy registration requires a mutable ModuleService resolver")
    return resolver


def register_strategy(
    strategy: type[StorageStrategy],
    tags: Mapping[str, Any] | None = None,
    aliases: Sequence[str] = (),
    *,
    resolver: ModuleResolver | None = None,
) -> type[StorageStrategy]:
    """Register one storage strategy class.

    Args:
        strategy (type[StorageStrategy]): Strategy subclass to register.
        tags (Mapping[str, Any] | None): Raw JSON-compatible Tag claims copied
            onto the published strategy record. Definitions are optional and
            registration never creates them.
        aliases (Sequence[str]): Additional accepted lookup names.
        resolver (ModuleResolver | None): Context-owned module resolver. The
            default Context is selected when omitted. Registration requires a
            mutable `ModuleService` implementation.

    Returns:
        type[StorageStrategy]: Registered strategy class.

    Raises:
        TypeError: If `strategy` is not a `StorageStrategy` subclass.
        ValueError: If its identifier is invalid.
    """
    if not isinstance(strategy, type) or not issubclass(strategy, StorageStrategy):
        raise TypeError("register_strategy expects a StorageStrategy subclass")
    identifier = class_identifier(strategy, label="strategy identifier")
    strategy.identifier = identifier
    service = _module_service(module_resolver(resolver))
    found = service.inspect("strategy", identifier)
    existing_aliases = tuple(found[0]["aliases"]) if found else ()
    if tags is not None and not isinstance(tags, Mapping):
        raise TypeError("Storage strategy tags must be a string-keyed mapping")
    authored_tags = dict(tags or {})
    if not all(isinstance(tag_identifier, str) for tag_identifier in authored_tags):
        raise TypeError("Storage strategy tags must be a string-keyed mapping")
    existing_meta = found[0]["record"].get("meta") if found else {}
    existing_tags = existing_meta.get("tags", {}) if isinstance(existing_meta, Mapping) else {}
    if not isinstance(existing_tags, Mapping):
        raise TypeError(f"Storage strategy {identifier!r} record meta.tags must be a mapping")
    alias_values = tuple(dict.fromkeys((*existing_aliases, *aliases, strategy.__name__)))
    tag_values = {**existing_tags, **authored_tags}
    if found:
        restored = _from_row(found[0], service)
        if restored is strategy and alias_values == existing_aliases and tag_values == existing_tags:
            return strategy
        coordinate = found[0]["record"]["meta"]["installation"]["coordinate"]
        if coordinate != f"programmatic/strategy/{identifier}":
            raise ValueError(f"Storage strategy {identifier!r} belongs to {coordinate!r} and cannot be changed through register_strategy")
    service.register(
        "strategy",
        identifier,
        {
            "name": strategy.__name__,
            "desc": (strategy.__doc__ or "Registered storage strategy.").strip().splitlines()[0],
            "source": "python-import",
            "target": python_target(strategy, label=f"Storage strategy {identifier!r}"),
            "active": True,
            "meta": {"schema_version": 1, **({"tags": tag_values} if tag_values else {})},
        },
        aliases=alias_values,
        coordinate=f"programmatic/strategy/{identifier}",
    )
    return _from_row(service.inspect("strategy", identifier)[0], service)


def _load_strategy(key: Any, resolver: ModuleResolver) -> type[StorageStrategy]:
    """Load one strategy through an explicitly selected resolver."""
    return _from_row(_row(key, resolver), resolver)


def strategy_aliases(strategy: Any, *, resolver: ModuleResolver | None = None) -> tuple[str, ...]:
    """Return accepted aliases for a registered strategy.

    Args:
        strategy (Any): Strategy class, value, identifier, alias, or spec.
        resolver (ModuleResolver | None): Context-owned module resolver. The
            default Context is selected when omitted.

    Returns:
        tuple[str, ...]: Registered aliases excluding the canonical id.
    """
    selected = module_resolver(resolver)
    return tuple(str(alias) for alias in _row(strategy, selected)["aliases"])
