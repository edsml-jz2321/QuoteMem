"""Workspace-owned backend registry and config-driven backend construction."""

from __future__ import annotations

__all__ = [
    "BackendType",
    "BackendRegistry",
    "backend_aliases",
]

from typing import TYPE_CHECKING

from heavenbase.utils import Any, Mapping, Sequence
from heavenbase.utils.registry import Registry, class_identifier, python_target

from ..modules import ModuleResolver, module_resolver, python_executable_ref
from ..modules.base import ResolverCache
from ..utils.ids import check_id
from ._construction import BackendFactory, BackendRequest, adapt_backend_factory, construct_backend, normalize_backend_leaf
from .base import Backend

if TYPE_CHECKING:
    from ..modules import ModuleService


def _backend_name(value: str) -> str:
    return check_id(str(value).strip(), "backend instance name")


def _module_service(resolver: ModuleResolver) -> "ModuleService":
    """Require lifecycle authority for a backend registration mutation."""
    from ..modules import ModuleService

    if not isinstance(resolver, ModuleService):
        raise TypeError("Backend registration requires a ModuleService resolver")
    return resolver


def _backend_key(value: Any) -> str:
    """Return the normalized lookup key for a Backend selector."""
    if isinstance(value, Backend):
        value = value.type
    if isinstance(value, type):
        if not issubclass(value, Backend):
            raise TypeError(f"Invalid backend {value!r}; expected Backend class, instance, or string")
        value = class_identifier(value, label="backend identifier")
    if not isinstance(value, str):
        raise TypeError(f"Invalid backend {value!r}; expected Backend class, instance, or string")
    return check_id(value.strip(), "backend identifier")


def _canonical(value: Any, resolver: ModuleResolver) -> str:
    """Resolve one Backend selector through the central record indexes."""
    key = _backend_key(value)
    rows = resolver.inspect("backend_type", key)
    if not rows:
        raise KeyError(f"Unknown backend identifier {key!r}")
    return str(rows[0]["identifier"])


_BACKEND_CACHE = ResolverCache[tuple[type[Backend], BackendFactory]]()


def _backend_from_row(row: Mapping[str, Any], resolver: ModuleResolver) -> tuple[type[Backend], BackendFactory]:
    """Materialize one Backend class and its optional custom factory."""

    def build(current_row: Mapping[str, Any]) -> tuple[type[Backend], BackendFactory]:
        identifier = str(current_row["identifier"])
        backend_cls = resolver.resolve("backend_type", identifier)
        if not isinstance(backend_cls, type) or not issubclass(backend_cls, Backend):
            raise TypeError(f"Backend type {identifier!r} did not resolve to a Backend subclass")
        actual = class_identifier(backend_cls, label="backend identifier")
        if actual != identifier:
            raise ValueError(f"Backend type record {identifier!r} resolved class identifier {actual!r}")
        factory_ref = current_row["record"]["meta"].get("factory")
        if factory_ref is not None and not isinstance(factory_ref, Mapping):
            raise TypeError(f"Backend type {identifier!r} factory must be an executable reference or null")
        factory = None if factory_ref is None else resolver.resolve_executable(factory_ref, owner=current_row)
        return backend_cls, adapt_backend_factory(backend_cls, factory)

    return _BACKEND_CACHE.get(resolver, row, build)


def _backend_row(identifier: Any, resolver: ModuleResolver) -> Mapping[str, Any]:
    """Return the inspection row for one registered Backend selector."""
    key = _backend_key(identifier)
    rows = resolver.inspect("backend_type", key)
    if not rows:
        raise KeyError(f"Unknown backend identifier {key!r}")
    return rows[0]


class BackendType:
    """Typed facade over authoritative `backend_type` module records."""

    @classmethod
    def _register(
        cls,
        backend_cls: type[Backend],
        tags: Mapping[str, Any] | None = None,
        aliases: Sequence[str] = (),
        *,
        identifier: str | None = None,
        factory: BackendFactory | None = None,
        resolver: ModuleResolver,
    ) -> type[Backend]:
        """Register a backend implementation class and its static Tag claims.

        Args:
            backend_cls: Backend implementation class.
            tags: Raw JSON-compatible Tag claims copied to the published record.
            aliases: Additional backend identifiers accepted for lookup.
            identifier: Optional canonical backend identifier. Defaults to class-owned metadata.
            factory: Optional importable custom Backend factory. The restored
                class's `from_config` method is used when omitted.
            resolver: Context-owned module lifecycle authority.

        Returns:
            type[Backend]: Registered backend class.
        """
        if not isinstance(backend_cls, type) or not issubclass(backend_cls, Backend):
            raise TypeError("BackendType registration expects a Backend subclass")
        ident = check_id(identifier or class_identifier(backend_cls), "backend identifier")
        old_identifier = backend_cls.__dict__.get("identifier")
        backend_cls.identifier = ident
        extra = tuple(aliases)
        for old in (old_identifier,):
            if old not in (None, ident, "backend"):
                extra += (str(old),)
        service = _module_service(resolver)
        found = service.inspect("backend_type", ident)
        existing_aliases = tuple(found[0]["aliases"]) if found else ()
        alias_values = tuple(dict.fromkeys((*existing_aliases, *extra, backend_cls.__name__)))
        if tags is not None and not isinstance(tags, Mapping):
            raise TypeError("Backend tags must be a string-keyed mapping")
        authored_tags = dict(tags or {})
        if not all(isinstance(tag_identifier, str) for tag_identifier in authored_tags):
            raise TypeError("Backend tags must be a string-keyed mapping")
        existing_meta = found[0]["record"].get("meta") if found else {}
        existing_tags = existing_meta.get("tags", {}) if isinstance(existing_meta, Mapping) else {}
        if not isinstance(existing_tags, Mapping):
            raise TypeError(f"Backend type {ident!r} record meta.tags must be a mapping")
        tag_values = {**existing_tags, **authored_tags}
        factory_ref = None if factory is None else python_executable_ref(factory, label=f"Backend {ident!r} factory")
        coordinate = f"programmatic/backend_type/{ident}"
        if found:
            existing_class, _ = _backend_from_row(found[0], resolver)
            existing_factory = found[0]["record"]["meta"].get("factory")
            if existing_class is backend_cls and existing_factory == factory_ref and alias_values == existing_aliases and tag_values == existing_tags:
                return backend_cls
            existing_coordinate = found[0]["record"]["meta"]["installation"]["coordinate"]
            if existing_coordinate != coordinate:
                raise ValueError(f"Backend type {ident!r} belongs to {existing_coordinate!r} and cannot be changed through register")
        service.register(
            "backend_type",
            ident,
            {
                "name": backend_cls.__name__,
                "desc": (backend_cls.__doc__ or "Registered Backend type.").strip().splitlines()[0],
                "source": "python-import",
                "target": python_target(backend_cls, label=f"Backend type {ident!r}"),
                "active": True,
                "meta": {
                    "schema_version": 1,
                    "factory": factory_ref,
                    **({"tags": tag_values} if tag_values else {}),
                },
            },
            aliases=alias_values,
            coordinate=coordinate,
        )
        _BACKEND_CACHE.clear(resolver)
        return BackendType.load(ident, resolver=resolver)

    @classmethod
    def load(cls, key: Any, *, resolver: ModuleResolver | None = None) -> type[Backend]:
        """Look up a backend class by backend identifier or alias.

        Args:
            key: Backend identifier or alias.
            resolver: Context-owned module resolver. The default Context is
                selected when omitted.

        Returns:
            type[Backend]: Registered backend class.

        Raises:
            KeyError: If the backend key is unknown.
            TypeError: If the key resolves to a non-backend object.
        """
        selected = module_resolver(resolver)
        return _backend_from_row(_backend_row(key, selected), selected)[0]

    @classmethod
    def factory(cls, key: Any, *, resolver: ModuleResolver | None = None) -> BackendFactory:
        """Return the one factory adapted from a `backend_type` record.

        Args:
            key: Backend identifier or alias.
            resolver: Context-owned module resolver. The default Context is
                selected when omitted.

        Returns:
            BackendFactory: Factory restored or adapted from the authoritative record.
        """
        selected = module_resolver(resolver)
        return _backend_from_row(_backend_row(key, selected), selected)[1]

    @classmethod
    def build(
        cls,
        alias: str,
        spec: Mapping[str, Any],
        ws_id: str | None = None,
        *,
        resolver: ModuleResolver | None = None,
        config: Any | None = None,
    ) -> Backend:
        """Build one Backend through its authoritative type record.

        Args:
            alias: Config key used as the instance-name fallback.
            spec: Flat Backend leaf mapping with a required `type`.
            ws_id: Optional workspace namespace to bind.
            resolver: Context-owned module resolver. The default Context is
                selected when omitted and is retained by the Backend's
                effective Tag view.
            config: Optional Context-owned config authority retained for
                backend policy reads.

        Returns:
            Backend: Configured Backend produced by the record's one factory.

        Raises:
            ValueError: If only one of ``resolver`` and ``config`` is supplied.
        """
        if (resolver is None) != (config is None):
            raise ValueError("Backend construction requires resolver and config together, or neither for process defaults")
        selected, manager = _authority_pair(resolver, config)
        request = normalize_backend_leaf(alias, spec, ws_id=ws_id)
        identifier = cls.identifier_of(request.identifier, resolver=selected)
        normalized = BackendRequest(identifier, request.name, request.provider_config, request.module)
        return construct_backend(
            normalized,
            cls.factory(identifier, resolver=selected),
            resolver=selected,
            config=manager,
        )

    @classmethod
    def identifier_of(cls, backend: Any, *, resolver: ModuleResolver | None = None) -> str:
        """Return the canonical backend identifier for a class, instance, or alias.

        Args:
            backend: Backend class, backend instance, backend identifier, alias, or ``"*"``.
            resolver: Context-owned module resolver. The default Context is
                selected when omitted.

        Returns:
            str: Canonical backend identifier.

        Raises:
            KeyError: If a string identifier is unknown.
            TypeError: If ``backend`` is not a supported backend reference.
        """
        selected = module_resolver(resolver)
        if isinstance(backend, Backend):
            backend = backend.type
        if isinstance(backend, type) and issubclass(backend, Backend):
            row = _backend_row(backend, selected)
            identifier = str(row["identifier"])
            if _backend_from_row(row, selected)[0] is not backend:
                raise ValueError(f"Backend class {backend!r} differs from registered type {identifier!r}")
            return identifier
        if isinstance(backend, str):
            if backend == "*":
                return backend
            return _canonical(backend, selected)
        raise TypeError(f"Invalid backend {backend!r}")

    @classmethod
    def types(cls, *, resolver: ModuleResolver | None = None) -> tuple[type[Backend], ...]:
        """Return all registered backend classes.

        Args:
            resolver: Context-owned module resolver. The default Context is
                selected when omitted.

        Returns:
            tuple[type[Backend], ...]: Registered backend classes in registry order.
        """
        selected = module_resolver(resolver)
        return tuple(cls.load(identifier, resolver=selected) for identifier in cls.identifiers(resolver=selected))

    @classmethod
    def identifiers(cls, *, resolver: ModuleResolver | None = None) -> tuple[str, ...]:
        """Return all registered backend identifiers.

        Args:
            resolver: Context-owned module resolver. The default Context is
                selected when omitted.

        Returns:
            tuple[str, ...]: Canonical backend identifiers in registry order.
        """
        selected = module_resolver(resolver)
        return tuple(str(row["identifier"]) for row in selected.inspect("backend_type"))

    @classmethod
    def aliases(cls, backend: Any, *, resolver: ModuleResolver | None = None) -> tuple[str, ...]:
        """Return accepted aliases for a registered backend type.

        Args:
            backend: Backend class, instance, identifier, or alias.
            resolver: Context-owned module resolver. The default Context is
                selected when omitted.

        Returns:
            tuple[str, ...]: Accepted aliases for the backend type.
        """
        return tuple(str(alias) for alias in _backend_row(backend, module_resolver(resolver))["aliases"])


def _authority_pair(resolver: ModuleResolver | None, config: Any | None) -> tuple[ModuleResolver, Any]:
    """Return explicit authorities, or the process default Context's paired modules and config."""
    if resolver is None and config is None:
        from ..context.authority import resolve_runtime_authority

        authority = resolve_runtime_authority(label="Backend")
        return authority.resolver, authority.config
    return module_resolver(resolver), config


def _load_configured_backends(
    raw: Mapping[str, Any] | None,
    ws_id: str | None,
    *,
    resolver: ModuleResolver,
    config: Any | None = None,
) -> dict[str, Backend]:
    """Build all backends declared by the explicit or bound configuration.

    Args:
        raw: Optional explicit backend config mapping. When omitted, reads
            ``heavenbase.backends`` from the bound config authority.
        ws_id: Optional workspace namespace to bind on construction.
        resolver: Context-owned module resolver used for every Backend type and
            Tag lookup.
        config: Optional Context-owned config authority used for defaults and
            retained by every constructed Backend.

    Returns:
        dict[str, Backend]: Backend instances keyed by backend instance name.

    Raises:
        TypeError: If the selected backend configuration is not a mapping.
        ValueError: If no enabled backend is explicitly or configurationally
            declared.
    """
    if config is None:
        from ..context.authority import omitted_config

        config = omitted_config()
    manager = config
    source = manager.get("heavenbase.backends", default=None) if raw is None else raw
    if not isinstance(source, Mapping):
        raise TypeError("Backend configuration must be a mapping at 'heavenbase.backends'")
    src = dict(source)
    out: dict[str, Backend] = {}
    for alias, spec in src.items():
        if alias == "aux":
            for aux_alias, aux_spec in (spec or {}).items():
                if aux_spec:
                    backend = BackendType.build(aux_alias, aux_spec, ws_id=ws_id, resolver=resolver, config=manager)
                    out[backend.name] = backend
            continue
        if not spec:
            continue
        backend = BackendType.build(alias, spec, ws_id=ws_id, resolver=resolver, config=manager)
        out[backend.name] = backend
    if out:
        return out
    raise ValueError("Backend configuration does not define any enabled backends")


class BackendRegistry:
    """Workspace-owned physical backend registry."""

    def __init__(self, *, resolver: ModuleResolver | None = None, config: Any | None = None) -> None:
        """Create an empty backend registry.

        Args:
            resolver: Context-owned module resolver. The default Context is
                selected when omitted.
            config: Optional Context-owned config authority to bind to live
                Backend instances.

        Returns:
            None: This function does not return a value.

        Raises:
            ValueError: If only one construction authority is supplied.
        """
        if (resolver is None) != (config is None):
            raise ValueError("BackendRegistry requires resolver and config together, or neither for process defaults")
        if config is not None and not callable(getattr(config, "get", None)):
            raise TypeError("BackendRegistry config authority must provide get(path, default)")
        self._module_resolver, self._config = _authority_pair(resolver, config)
        self._backends = Registry[str, Backend]("backend-instances", normalize=_backend_name, value_policy="process_identity")

    @property
    def backends(self) -> dict[str, Backend]:
        """Return registered backend instances.

        Returns:
            dict[str, Backend]: Snapshot of backend names mapped to backend instances.
        """
        return dict(self._backends.snapshot())

    @classmethod
    def from_config(
        cls,
        raw: Mapping[str, Any] | None = None,
        ws_id: str | None = None,
        *,
        resolver: ModuleResolver | None = None,
        config: Any | None = None,
    ) -> "BackendRegistry":
        """Build a registry from config or an explicit mapping.

        Args:
            raw: Optional backend config mapping. When omitted, uses configured defaults.
            ws_id: Optional workspace namespace to bind to each backend.
            resolver: Context-owned module resolver. The default Context is
                selected when omitted.
            config: Optional Context-owned config authority used for defaults
                and bound to every Backend instance.

        Returns:
            BackendRegistry: Registry populated with configured backend instances.
        """
        if (resolver is None) != (config is None):
            raise ValueError("BackendRegistry.from_config requires resolver and config together, or neither for process defaults")
        reg = cls(resolver=resolver, config=config)
        selected = reg._module_resolver
        for backend in _load_configured_backends(raw, ws_id, resolver=selected, config=reg._config).values():
            reg.register(backend)
        return reg

    def register(self, backend: Backend) -> Backend:
        """Register a backend by its name.

        Args:
            backend: Backend instance constructed for this registry's exact
                resolver and config authorities. A fully unbound standalone
                Backend may be adopted only by the process-default registry.

        Returns:
            Backend: Registered backend instance.

        Raises:
            TypeError: If ``backend`` is not a Backend instance.
            ValueError: If the live Backend was constructed for different or
                only partially specified authorities.
        """
        if not isinstance(backend, Backend):
            raise TypeError("BackendRegistry.register expects a live Backend instance")
        bound_resolver = backend._module_resolver
        bound_config = backend._config
        if bound_resolver is None and bound_config is None:
            default_resolver, default_config = _authority_pair(None, None)
            if self._module_resolver is not default_resolver or self._config is not default_config:
                raise ValueError(
                    "An unbound live Backend can enter only the process-default registry; " "construct it with this registry's resolver and config authorities"
                )
            backend._adopt_process_defaults()
        elif bound_resolver is None or bound_config is None:
            raise ValueError("A live Backend has a partial construction authority; provide both resolver and config when constructing it")
        elif bound_resolver is not self._module_resolver or bound_config is not self._config:
            raise ValueError("A live Backend cannot be registered under different resolver or config authorities")
        return self._backends.update(backend.name, backend)

    def get(self, name: str) -> Backend:
        """Return a backend by name.

        Args:
            name: Backend instance name.

        Returns:
            Backend: Registered backend instance.

        Raises:
            KeyError: If ``name`` is not registered.
        """
        name = _backend_name(name)
        backend = self._backends.get(name)
        if backend is not None:
            return backend
        known = sorted(self._backends.list())
        raise KeyError(f"Unknown backend {name!r}; known={known}") from None

    def names(self) -> list[str]:
        """Return backend names in registry order.

        Args:
            None.

        Returns:
            list[str]: Backend instance names in registry order.
        """
        return list(self._backends.list())

    def items(self) -> list[tuple[str, Backend]]:
        """Return a snapshot of backend items for iteration.

        Args:
            None.

        Returns:
            list[tuple[str, Backend]]: ``(name, backend)`` pairs in registry order.
        """
        return list(self._backends.items())

    def first_by_type(self, backend_identifiers: Sequence[str]) -> Backend | None:
        """Return the first backend whose type is in ``backend_identifiers``.

        Args:
            backend_identifiers: Backend identifiers or aliases in priority order.

        Returns:
            Backend | None: First matching backend, or ``None`` when no match exists.
        """
        backends = list(self._backends.values())
        for backend_identifier in backend_identifiers:
            for backend in backends:
                if BackendType.identifier_of(backend, resolver=self._module_resolver) == BackendType.identifier_of(
                    backend_identifier,
                    resolver=self._module_resolver,
                ):
                    return backend
        return None

    def has_type(self, backend_identifiers: Sequence[str]) -> bool:
        """Return whether any backend matches one of the requested types.

        Args:
            backend_identifiers: Backend identifiers or aliases.

        Returns:
            bool: True when at least one registered backend matches.
        """
        return self.first_by_type(backend_identifiers) is not None


def backend_aliases(backend: Any, *, resolver: ModuleResolver | None = None) -> tuple[str, ...]:
    """Return accepted aliases for one Backend selector.

    Args:
        backend: Backend identifier, class, or live instance.
        resolver: Context-owned module resolver. The default Context is selected
            when omitted.

    Returns:
        tuple[str, ...]: Registered Backend aliases.
    """
    return BackendType.aliases(backend, resolver=resolver)
