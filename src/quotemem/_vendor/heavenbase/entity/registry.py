"""Registry-backed resolution for durable Entity class definitions."""

from __future__ import annotations

from collections.abc import Mapping
from importlib import import_module
import re
from typing import Any

from ..modules import ModuleResolver, module_resolver
from ..modules.base import ResolverCache
from .base import Entity

_ENTITY_CACHE = ResolverCache[type[Entity]]()
_ARTIFACT_NAMESPACE = re.compile(r"_heavenbase_artifact_[0-9a-f]{64}\Z")


def load_entity_type(identifier: str, *, resolver: ModuleResolver | None = None) -> type[Entity]:
    """Load one registered Entity class by identifier or alias.

    Args:
        identifier: Entity registration identifier or alias.
        resolver: Context-owned module resolver. The default Context is
            selected when omitted.

    Returns:
        type[Entity]: Entity class restored from the authoritative record.

    Raises:
        KeyError: If the Entity registration is absent.
        TypeError: If the record does not restore an Entity class or schema mapping.
        ValueError: If the restored schema identifier differs from the record identifier.
    """
    selected = module_resolver(resolver)
    rows = selected.inspect("entity", identifier)
    if not rows:
        raise KeyError(f"Unknown entity registration {identifier!r}")
    row = rows[0]

    def build(current_row: Mapping[str, Any]) -> type[Entity]:
        restored = selected.resolve("entity", str(current_row["identifier"]))
        if isinstance(restored, Mapping):
            restored = Entity.from_schema(dict(restored), resolver=selected)
        if not isinstance(restored, type) or not issubclass(restored, Entity):
            raise TypeError(f"Entity registration {identifier!r} did not resolve to an Entity class")
        actual = restored.schema().entity_id
        expected = str(current_row["identifier"])
        if actual != expected:
            raise ValueError(f"Entity registration {expected!r} resolved schema identifier {actual!r}")
        return restored

    return _ENTITY_CACHE.get(selected, row, build)


def registered_entity_identifiers(*, resolver: ModuleResolver | None = None) -> tuple[str, ...]:
    """Return durable Entity registration identifiers in Registry order.

    Args:
        resolver: Context-owned module resolver. The default Context is
            selected when omitted.

    Returns:
        tuple[str, ...]: Canonical Entity identifiers.
    """
    return tuple(str(row["identifier"]) for row in module_resolver(resolver).inspect("entity"))


def load_entity_artifact_export(
    identifier: str,
    module: str,
    attribute: str,
    *,
    resolver: ModuleResolver | None = None,
) -> Any:
    """Load one export beside a Registry-resolved Entity artifact.

    This is the cross-record adapter for cohesive source folders. The Entity
    record selects the verified artifact generation; ``module`` and
    ``attribute`` then select another implementation from that same private
    package instead of importing an ambient optional bundle.

    Args:
        identifier (str): Registered entity identifier.
        module (str): Relative dotted module name beside the entity artifact.
        attribute (str): Export name to load from that module.
        resolver (ModuleResolver | None): Context-owned module resolver. When
            omitted, the default Context is selected.

    Returns:
        Any: Loaded export from the verified entity artifact package.

    Raises:
        ValueError: If ``module``, ``attribute``, or the entity registration is
            invalid.
        AttributeError: If the sibling module has no matching export.
    """
    parts = str(module or "").split(".")
    if not parts or any(not part.isidentifier() for part in parts):
        raise ValueError(f"Entity artifact module must be a relative dotted name, got {module!r}")
    name = str(attribute or "")
    if not name.isidentifier():
        raise ValueError(f"Entity artifact attribute must be an identifier, got {attribute!r}")
    entity = load_entity_type(identifier, resolver=resolver)
    namespace = entity.__module__.partition(".")[0]
    if _ARTIFACT_NAMESPACE.fullmatch(namespace) is None:
        raise ValueError(f"Entity registration {identifier!r} does not belong to a verified module artifact")
    sibling = import_module(f"{namespace}.{'.'.join(parts)}")
    try:
        return getattr(sibling, name)
    except AttributeError:
        raise AttributeError(f"Entity artifact {identifier!r} module {module!r} has no export {name!r}") from None
