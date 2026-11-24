"""Workspace extension entity-bundle contract."""

from __future__ import annotations

__all__ = ["Extension"]

from inspect import Parameter, signature
from types import MappingProxyType

from heavenbase.utils import Any, Callable, Iterable, Mapping, coerce_bool, deepcopy

from ..modules import ModuleResolver, module_resolver
from ..utils.ids import check_id
from ..utils.serialize import validate_json


def _identifier(value: str) -> str:
    return check_id(str(value).strip(), "extension identifier")


def _api_name(identifier: str, value: str | None) -> str:
    name = str(value or identifier).strip().replace("-", "_")
    if not name.isidentifier() or name.startswith("_"):
        raise ValueError(f"Extension API name {name!r} must be a public Python identifier")
    return name


def _api_slot_conflict(extension: "Extension", claimant: str | None = None) -> AttributeError:
    """Return a record-qualified AttributeError for a taken workspace API slot."""
    if claimant is not None:
        return AttributeError(f"Workspace API name {extension.api_name!r} for extension " f"{extension.identifier!r} conflicts with extension {claimant!r}")
    return AttributeError(
        f"Workspace API name {extension.api_name!r} for extension " f"{extension.identifier!r} conflicts with an existing workspace attribute"
    )


class Extension:
    """Describe and enable one workspace entity bundle.

    An extension owns the entities and optional workspace behavior enabled together.
    Register it once with `register()` and enable it in a workspace by identifier.

    Args:
        identifier (str): Stable registry and workspace identifier.
        name (str): Human-readable name. Defaults to `identifier`.
        version (str): Extension version shown in metadata.
        desc (str): Human-readable purpose of the extension.
        entities (Iterable[Any]): Entity classes or schema mappings enabled together.
        required (bool): Whether every workspace must enable this extension.
        requires (Iterable[str]): Exact extension identifiers that must be
            active before this extension.
        tags (Mapping[str, Any] | None): Raw semantic tag claims copied into
            the published extension record.
        meta (Mapping[str, Any] | None): Additional metadata copied for inspection.
        setup (Callable[[Any], Iterable[Any] | None] | None): Optional workspace hook
            that returns additional entities to register. Use it only when the entity
            set depends on workspace configuration.
        api (Callable[[Any], Any] | None): Optional factory for a workspace-bound API
            object.
        api_name (str | None): Public workspace attribute for the API object. Defaults
            to the normalized extension identifier when `api` is provided.

    Raises:
        TypeError: If entities, tags, required, setup, or api has an invalid shape.
        ValueError: If the identifier or API name is invalid, or `api_name` is supplied
            without an `api` factory.
    """

    def __init__(
        self,
        identifier: str,
        *,
        name: str = "",
        version: str = "",
        desc: str = "",
        entities: Iterable[Any] = (),
        required: bool = False,
        requires: Iterable[str] = (),
        tags: Mapping[str, Any] | None = None,
        meta: Mapping[str, Any] | None = None,
        setup: Callable[[Any], Iterable[Any] | None] | None = None,
        api: Callable[[Any], Any] | None = None,
        api_name: str | None = None,
    ) -> None:
        """Construct one workspace extension descriptor.

        Returns:
            None: This initializer does not return a value.
        """
        if isinstance(entities, (str, bytes)):
            raise TypeError("Extension entities must be an iterable of entity declarations")
        if tags is not None and not isinstance(tags, Mapping):
            raise TypeError("Extension tags must be a mapping of tag identifiers to JSON values")
        if isinstance(requires, (str, bytes)):
            raise TypeError("Extension requires must be an iterable of extension identifiers")
        required = coerce_bool(required, "Extension required")
        self.identifier = _identifier(identifier)
        self.name = str(name or self.identifier)
        self.version = str(version)
        self.desc = str(desc)
        self.entities = tuple(entities or ())
        self.required = required
        self.requires = tuple(dict.fromkeys(_identifier(value) for value in (requires or ())))
        if self.identifier in self.requires:
            raise ValueError(f"Extension {self.identifier!r} cannot require itself")
        tag_claims: dict[str, Any] = {}
        for tag, value in (tags or {}).items():
            if not isinstance(tag, str):
                raise TypeError("Extension tags must use string identifiers")
            try:
                validate_json(value)
            except (TypeError, ValueError) as exc:
                raise TypeError(f"Extension tag {tag!r} must have a JSON-compatible value") from exc
            tag_claims[tag] = deepcopy(value)
        self.tags = MappingProxyType(tag_claims)
        self.meta = dict(meta or {})
        if setup is not None and not callable(setup):
            raise TypeError("Extension setup must be callable")
        if api is not None and not callable(api):
            raise TypeError("Extension api must be callable")
        if api is None and api_name is not None:
            raise ValueError("Extension api_name requires an api factory")
        self.setup = setup
        self.api = api
        self.api_name = _api_name(self.identifier, api_name) if api is not None else None

    @classmethod
    def load(cls, key: str, *, resolver: ModuleResolver | None = None) -> "Extension":
        """Load a registered extension by identifier.

        Args:
            key (str): Registered extension identifier.
            resolver (ModuleResolver | None): Context-owned module resolver.
                When omitted, the default Context is selected.

        Returns:
            Extension: Registered extension descriptor.

        Raises:
            KeyError: If `key` is not registered.
        """
        from .registry import _load_extension

        return _load_extension(key, module_resolver(resolver))

    @classmethod
    def identifiers(cls, *, resolver: ModuleResolver | None = None) -> tuple[str, ...]:
        """Return registered extension identifiers.

        Args:
            resolver (ModuleResolver | None): Context-owned module resolver.
                When omitted, the default Context is selected.

        Returns:
            tuple[str, ...]: Canonical extension identifiers in Registry order.
        """
        from .registry import _extension_identifiers

        return _extension_identifiers(module_resolver(resolver))

    def register(self, *, resolver: ModuleResolver | None = None) -> "Extension":
        """Publish this extension through the central module Registry.

        Static Entity declarations, setup hooks, and API factories must be
        durable top-level objects. Registration publishes absent Entity records
        and the Extension definition atomically.

        Args:
            resolver (ModuleResolver | None): Writable Context-owned module
                resolver. When omitted, the default Context is selected.

        Returns:
            Extension: Registered extension descriptor.

        Raises:
            ValueError: If the identifier is already owned by a different descriptor
                or a callable cannot be represented by a durable import locator.
        """
        from .registry import _register_extension

        return _register_extension(self, module_resolver(resolver))

    def _apply(self, workspace: Any) -> list[type]:
        """Apply this descriptor after workspace-owned activation preflight.

        Args:
            workspace (Any): Workspace receiving the extension entity bundle.

        Returns:
            list[type]: Entity classes registered while enabling the extension.

        Raises:
            AttributeError: If the extension API name collides with a workspace
                attribute.
            TypeError: If `setup` returns a string instead of entity declarations.
        """
        from ..workspace.foundation import FIXED_ENTITY_IDS

        fixed = sorted(FIXED_ENTITY_IDS.intersection(self.entity_ids()))
        if fixed:
            raise ValueError(f"Extension {self.identifier!r} cannot own fixed workspace Entities {fixed!r}")
        self._check_api_slot(workspace)
        registered = [workspace.register(entity) for entity in self.entities]
        if self.setup is not None:
            extra = self.setup(workspace) or ()
            if isinstance(extra, (str, bytes)):
                raise TypeError("Extension setup must return an iterable of entity declarations")
            registered.extend(workspace.register(entity) for entity in extra)
        publish = getattr(getattr(workspace, "_catalog", None), "publish_schema_rows", None)
        if callable(publish):
            for entity in registered:
                publish(entity)
        self._attach_api(workspace)
        return registered

    def entity_ids(self) -> list[str]:
        """Return identifiers for the statically declared entities.

        Args:
            None.

        Returns:
            list[str]: Entity identifiers in declaration order.
        """
        out = []
        for entity in self.entities:
            schema = getattr(entity, "schema", None)
            if callable(schema):
                out.append(str(schema().entity_id))
                continue
            if isinstance(entity, Mapping):
                entity_id = entity.get("identifier") or entity.get("entity_id")
                if entity_id:
                    out.append(str(entity_id))
                continue
            out.append(str(entity))
        return out

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible extension metadata.

        Args:
            None.

        Returns:
            dict[str, Any]: Extension metadata for catalog and workspace inspection.
        """
        data = {
            "identifier": self.identifier,
            "name": self.name,
            "version": self.version,
            "desc": self.desc,
            "required": self.required,
            "requires": list(self.requires),
            "tags": deepcopy(dict(self.tags)),
            "entities": self.entity_ids(),
            "meta": dict(self.meta),
        }
        if self.api_name is not None:
            data["api_name"] = self.api_name
        return data

    def _attach_api(self, workspace: Any) -> Any | None:
        if self.api is None or self.api_name is None:
            return None
        self._check_api_slot(workspace)
        if self.api_name in workspace._extension_apis:
            return workspace._extension_apis[self.api_name]
        try:
            parameters = signature(self.api).parameters.values()
        except (TypeError, ValueError):
            parameters = ()
        accepts_resolver = any(parameter.name == "resolver" or parameter.kind is Parameter.VAR_KEYWORD for parameter in parameters)
        value = self.api(workspace, resolver=workspace.context.modules()) if accepts_resolver else self.api(workspace)
        setattr(workspace, self.api_name, value)
        workspace._extension_apis[self.api_name] = value
        workspace._extension_api_owners[self.api_name] = self.identifier
        return value

    def _check_api_slot(self, workspace: Any, owners: Mapping[str, str] | None = None) -> None:
        if self.api_name is None:
            return
        claimant = (workspace._extension_api_owners if owners is None else owners).get(self.api_name)
        if claimant is not None:
            if claimant != self.identifier:
                raise _api_slot_conflict(self, claimant)
            return
        if hasattr(workspace, self.api_name):
            raise _api_slot_conflict(self)

    def _matches(self, other: "Extension") -> bool:
        return self.to_dict() == other.to_dict() and self.setup is other.setup and self.api is other.api
