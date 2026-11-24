"""Registry-backed workspace Extension definitions."""

from __future__ import annotations

__all__: list[str] = []

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from ..entity.base import Entity
from ..entity.registry import load_entity_type
from ..modules import ModuleResolver, ModuleService, python_executable_ref
from ..modules.base import ResolverCache
from heavenbase.utils.registry import python_target
from heavenbase.version import __version__
from .base import Extension, _identifier

_EXTENSION_CACHE = ResolverCache[Extension]()


def _extension_identifiers(resolver: ModuleResolver) -> tuple[str, ...]:
    """Return registered Extension identifiers in Registry order.

    Returns:
        tuple[str, ...]: Canonical Extension identifiers.
    """
    return tuple(str(row["identifier"]) for row in resolver.inspect("extension"))


def _load_extension(identifier: str, resolver: ModuleResolver) -> Extension:
    """Load one Extension descriptor from its authoritative record.

    Args:
        identifier: Extension identifier.

    Returns:
        Extension: Materialized extension descriptor.

    Raises:
        KeyError: If the Extension registration is absent.
        TypeError: If its inline definition has an invalid shape.
        ValueError: If its record identity differs from the definition.
    """
    key = _identifier(identifier)
    rows = resolver.inspect("extension", key)
    if not rows:
        raise KeyError(f"No extension registered for {key!r}")
    row = rows[0]

    def build(current_row: Mapping[str, Any]) -> Extension:
        raw = resolver.resolve("extension", str(current_row["identifier"]))
        if not isinstance(raw, Mapping):
            raise TypeError(f"Extension registration {key!r} must resolve to an inline definition")
        return _extension_from_definition(
            str(current_row["identifier"]),
            raw,
            resolver,
            owner=current_row,
        )

    return _EXTENSION_CACHE.get(resolver, row, build)


def _register_extension(extension: Extension, resolver: ModuleResolver) -> Extension:
    """Publish one programmatic Extension and its static Entity records atomically.

    Args:
        extension: Extension descriptor containing durable static declarations.

    Returns:
        Extension: Registered descriptor, preserving identity for the initial
        materialization cache.

    Raises:
        TypeError: If `extension` is not an Extension or an Entity declaration
            has an unsupported shape.
        ValueError: If an existing registration differs or a locator is not durable.
    """
    if not isinstance(extension, Extension):
        raise TypeError(f"Expected Extension, got {type(extension).__name__}")
    if extension.required:
        raise ValueError("Programmatic extensions cannot declare required=True; universal activation is installer-owned")
    if not isinstance(resolver, ModuleService):
        raise TypeError("Extension registration requires a writable ModuleService")
    service = resolver
    found = service.inspect("extension", extension.identifier)
    if found:
        existing = _load_extension(extension.identifier, service)
        if not existing._matches(extension):
            raise ValueError(f"Extension {extension.identifier!r} is already registered")
        return existing

    items: list[dict[str, Any]] = []
    dependencies: list[str] = []
    for entity in extension.entities:
        item, identifier = _entity_item(entity)
        dependencies.append(f"entity:{identifier}")
        existing_rows = service.inspect("entity", identifier)
        if existing_rows:
            existing = load_entity_type(identifier, resolver=service)
            if isinstance(entity, type):
                if existing_rows[0]["record"].get("source") == "python-import" and existing is not entity:
                    raise ValueError(f"Entity registration {identifier!r} resolves to a different class")
                if existing.schema().to_dict() != entity.schema().to_dict():
                    raise ValueError(f"Entity registration {identifier!r} resolves to a different schema")
            if isinstance(entity, Mapping) and existing.schema().to_dict() != Entity.from_schema(dict(entity)).schema().to_dict():
                raise ValueError(f"Entity registration {identifier!r} resolves to a different schema")
            continue
        items.append(item)
    definition = _extension_definition(extension)
    items.append(
        {
            "kind": "extension",
            "identifier": extension.identifier,
            "groups": {"scope": ["system"] if extension.required else []},
            "name": extension.name,
            "desc": extension.desc,
            "source": "inline",
            "target": "definition",
            "active": True,
            "meta": {
                "schema_version": 1,
                "dependencies": dependencies,
                "tags": deepcopy(dict(extension.tags)),
                "definition": definition,
            },
        }
    )
    service.install(
        {
            "manifest_version": 2,
            "coordinate": f"programmatic/extension/{extension.identifier}",
            "version": __version__,
            "compatibility": {},
            "bundle": f"extension-{extension.identifier}",
            "items": items,
        },
        publisher="programmatic",
    )
    row = service.inspect("extension", extension.identifier)[0]
    return _EXTENSION_CACHE.get(service, row, lambda _current_row: extension)


def _entity_item(entity: Any) -> tuple[dict[str, Any], str]:
    """Return one programmatic Entity record item and identifier."""
    if isinstance(entity, type) and issubclass(entity, Entity):
        schema = entity.schema()
        identifier = schema.entity_id
        try:
            target = python_target(entity, label=f"Entity {identifier!r}")
        except ValueError:
            definition = schema.to_dict()
            source = "inline"
            target = "definition"
            meta = {"schema_version": 1, "definition": definition}
        else:
            source = "python-import"
            meta = {"schema_version": 1}
        return (
            {
                "kind": "entity",
                "identifier": identifier,
                "aliases": [entity.__name__],
                "name": schema.name,
                "desc": schema.desc,
                "source": source,
                "target": target,
                "active": True,
                "meta": meta,
            },
            identifier,
        )
    if isinstance(entity, Mapping):
        definition = deepcopy(dict(entity))
        identifier = str(definition.get("entity_id") or definition.get("identifier") or "")
        if not identifier:
            raise ValueError("Extension Entity mapping requires entity_id")
        compiled = Entity.from_schema(definition)
        identifier = compiled.schema().entity_id
        return (
            {
                "kind": "entity",
                "identifier": identifier,
                "name": compiled.schema().name,
                "desc": compiled.schema().desc,
                "source": "inline",
                "target": "definition",
                "active": True,
                "meta": {"schema_version": 1, "definition": definition},
            },
            identifier,
        )
    raise TypeError(f"Extension Entity declaration {entity!r} must be an Entity class or schema mapping")


def _extension_definition(extension: Extension) -> dict[str, Any]:
    """Return the JSON-safe inline definition for one Extension."""
    return {
        "identifier": extension.identifier,
        "name": extension.name,
        "version": extension.version,
        "desc": extension.desc,
        "required": extension.required,
        "requires": list(extension.requires),
        "entities": extension.entity_ids(),
        "meta": deepcopy(extension.meta),
        "setup": None if extension.setup is None else python_executable_ref(extension.setup, label=f"Extension {extension.identifier!r} setup"),
        "api": None if extension.api is None else python_executable_ref(extension.api, label=f"Extension {extension.identifier!r} API"),
        "api_name": extension.api_name,
    }


def _extension_from_definition(
    identifier: str,
    raw: Mapping[str, Any],
    resolver: ModuleResolver,
    *,
    owner: Mapping[str, Any],
) -> Extension:
    """Materialize one validated Extension from an inline record definition."""
    expected = _identifier(identifier)
    actual = _identifier(str(raw.get("identifier") or expected))
    if actual != expected:
        raise ValueError(f"Extension record {expected!r} contains identifier {actual!r}")
    entities_raw = raw.get("entities", [])
    if not isinstance(entities_raw, list):
        raise TypeError(f"Extension {expected!r} entities must be a list")
    requires_raw = raw.get("requires", [])
    if not isinstance(requires_raw, list):
        raise TypeError(f"Extension {expected!r} requires must be a list")
    record = owner.get("record")
    record_meta = record.get("meta") if isinstance(record, Mapping) else None
    tags_raw = record_meta.get("tags", {}) if isinstance(record_meta, Mapping) else {}
    if not isinstance(tags_raw, Mapping):
        raise TypeError(f"Extension {expected!r} record meta.tags must be a mapping")

    def restored(field: str) -> Any:
        reference = raw.get(field)
        if reference is None:
            return None
        if not isinstance(reference, Mapping):
            raise TypeError(f"Extension {expected!r} {field} must be an executable reference")
        return resolver.resolve_executable(reference, owner=owner)

    return Extension(
        expected,
        name=str(raw.get("name") or expected),
        version=str(raw.get("version") or ""),
        desc=str(raw.get("desc") or ""),
        entities=tuple(load_entity_type(str(item), resolver=resolver) for item in entities_raw),
        required=bool(raw.get("required", False)),
        requires=tuple(str(item) for item in requires_raw),
        tags=dict(tags_raw),
        meta=dict(raw.get("meta") or {}),
        setup=restored("setup"),
        api=restored("api"),
        api_name=str(raw.get("api_name")) if raw.get("api_name") is not None else None,
    )
