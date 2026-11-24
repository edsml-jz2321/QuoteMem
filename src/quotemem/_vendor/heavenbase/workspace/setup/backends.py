"""Workspace backend construction and replay policy."""

from __future__ import annotations

__all__ = [
    "WorkspaceBackendSetup",
    "WorkspaceConstruction",
    "backend_identifiers_from_config",
    "build_backend_registry",
    "resolve_workspace_backend_setup",
]

from heavenbase.utils import Any, Dict, Literal, Mapping, Optional, dataclass, deepcopy, field

from ...backends._construction import normalize_backend_leaf
from ...backends.base import Backend
from ...backends.registry import BackendRegistry, BackendType, _load_configured_backends
from ...modules import ModuleResolver
from ..presets import resolve_workspace_backends, resolve_workspace_preset


@dataclass(frozen=True)
class WorkspaceConstruction:
    """Canonical construction contract for one workspace identity.

    Args:
        kind: Closed construction mode: ``ambient``, ``preset``, ``backends``,
            or ``runtime``.
        preset: Canonical preset identifier for ``preset`` construction.
        backends: Replayable Backend configuration for ``backends``
            construction, or Backend-name to canonical-identifier metadata for
            ``runtime`` construction. Serialization keeps those two shapes
            separate as ``config`` and ``identifiers``.
    """

    kind: Literal["ambient", "preset", "backends", "runtime"] = "ambient"
    preset: Optional[str] = None
    backends: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the closed construction shape and isolate mutable values.

        Args:
            None.

        Returns:
            None: This initializer normalizes the selected construction value.

        Raises:
            ValueError: If fields do not match the selected construction kind.
        """
        kind = str(self.kind)
        if kind not in {"ambient", "preset", "backends", "runtime"}:
            raise ValueError(f"Unsupported workspace construction kind {kind!r}")
        preset = None if self.preset is None else str(self.preset)
        backends = deepcopy(dict(self.backends))
        if kind == "ambient" and (preset is not None or backends):
            raise ValueError("Ambient workspace construction cannot declare a preset or backends")
        if kind == "preset" and (not preset or backends):
            raise ValueError("Preset workspace construction requires only a preset identifier")
        if kind in {"backends", "runtime"} and preset is not None:
            raise ValueError(f"{kind.title()} workspace construction cannot declare a preset")
        if kind == "runtime":
            backends = {str(name): str(identifier) for name, identifier in backends.items()}
            if not backends:
                raise ValueError("Runtime workspace construction requires at least one backend identifier")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "preset", preset)
        object.__setattr__(self, "backends", backends)

    @classmethod
    def from_inputs(
        cls,
        backends: Optional[Mapping[str, Any]],
        preset: Optional[str],
        *,
        resolver: ModuleResolver | None = None,
    ) -> "WorkspaceConstruction":
        """Normalize public workspace backend inputs into replay policy.

        Args:
            backends (Mapping[str, Any] | None): Explicit backend config or live
                backend instances. Explicit backends take precedence over
                `preset`.
            preset (str | None): Optional configured workspace preset name.
            resolver: Module resolver used to canonicalize live Backend
                identifiers. Defaults to the process resolver.

        Returns:
            WorkspaceConstruction: Canonical construction contract.

        Raises:
            TypeError: If `backends` mixes live backend objects with config
                mappings.
        """
        if backends is not None:
            values = tuple(backends.values())
            runtime_only = bool(values) and all(isinstance(backend, Backend) for backend in values)
            if any(isinstance(backend, Backend) for backend in values) and not runtime_only:
                raise TypeError("backends must contain either config mappings or live Backend instances, not both")
            if runtime_only:
                return cls.from_runtime_backends(backends, resolver=resolver)
            return cls("backends", backends=dict(backends))
        return cls("preset", preset=str(preset)) if preset is not None else cls()

    @classmethod
    def from_runtime_backends(
        cls,
        backends: Any,
        *,
        resolver: ModuleResolver | None = None,
    ) -> "WorkspaceConstruction":
        """Build the runtime signature for live Backend instances.

        Args:
            backends: Mapping or BackendRegistry-compatible object exposing
                live Backend values.
            resolver: Module resolver used to canonicalize Backend identifiers.

        Returns:
            WorkspaceConstruction: Runtime name-to-identifier signature.

        Raises:
            TypeError: If any value is not a live Backend instance.
            ValueError: If multiple values expose the same Backend name.
        """
        values = tuple(backends.values()) if callable(getattr(backends, "values", None)) else tuple(backend for _, backend in backends.items())
        if not values or not all(isinstance(backend, Backend) for backend in values):
            raise TypeError("runtime workspace construction requires live Backend instances")
        identifiers = {backend.name: BackendType.identifier_of(backend, resolver=resolver) for backend in values}
        if len(identifiers) != len(values):
            raise ValueError("live Backend instances must have unique backend names")
        return cls("runtime", backends=identifiers)

    @classmethod
    def from_preset(cls, preset: Optional[str]) -> "WorkspaceConstruction":
        """Record the canonical preset selected during preset resolution.

        Args:
            preset (str | None): Resolved preset identifier, or `None` when no
                preset was selected.

        Returns:
            WorkspaceConstruction: Replayable canonical preset construction.
        """
        return cls("preset", preset=str(preset)) if preset is not None else cls()

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None) -> "WorkspaceConstruction":
        """Build a replayable construction contract from constructor kwargs.

        Args:
            config: Persisted constructor kwargs containing at most ``preset``
                or ``backends``.

        Returns:
            WorkspaceConstruction: Parsed replayable construction contract.

        Raises:
            ValueError: If the mapping is not a supported constructor shape.
        """
        data = deepcopy(dict(config or {}))
        if not data:
            return cls()
        if set(data) == {"preset"}:
            return cls.from_preset(data["preset"])
        if set(data) == {"backends"} and isinstance(data["backends"], Mapping):
            return cls("backends", backends=dict(data["backends"]))
        raise ValueError("Workspace construction config must contain only 'preset' or 'backends'")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "WorkspaceConstruction":
        """Parse a persisted construction contract.

        Args:
            data: Mapping produced by :meth:`to_dict`.

        Returns:
            WorkspaceConstruction: Validated construction contract.

        Raises:
            ValueError: If the persisted shape is unsupported.
        """
        record = deepcopy(dict(data))
        kind = str(record.pop("kind", ""))
        if kind == "ambient" and not record:
            return cls()
        if kind == "preset" and set(record) == {"preset"}:
            return cls(kind, preset=record["preset"])
        if kind == "backends":
            key = "config" if "config" in record else "backends"
            if set(record) == {key} and isinstance(record[key], Mapping):
                return cls(kind, backends=dict(record[key]))
        if kind == "runtime":
            key = "identifiers" if "identifiers" in record else "backends"
            if set(record) == {key} and isinstance(record[key], Mapping):
                return cls(kind, backends=dict(record[key]))
        raise ValueError(f"Invalid persisted workspace construction contract for kind {kind!r}")

    @property
    def runtime_only(self) -> bool:
        """Return whether reconstruction requires caller-supplied Backend instances."""
        return self.kind == "runtime"

    @property
    def kwargs(self) -> Dict[str, Any]:
        """Return replayable ``HeavenBase`` keyword arguments.

        Returns:
            Dict[str, Any]: Detached constructor kwargs.

        Raises:
            RuntimeError: If construction used live Backend instances.
        """
        if self.kind == "runtime":
            raise RuntimeError("Runtime-only workspace construction requires explicit live Backend instances")
        if self.kind == "preset":
            return {"preset": self.preset}
        if self.kind == "backends":
            return {"backends": deepcopy(self.backends)}
        return {}

    @property
    def backend_config(self) -> Dict[str, Any]:
        """Return isolated replayable Backend config, when present."""
        return deepcopy(self.backends) if self.kind == "backends" else {}

    @property
    def backend_identifiers(self) -> Dict[str, str]:
        """Return the runtime Backend name-to-identifier signature."""
        return deepcopy(self.backends) if self.kind == "runtime" else {}

    def to_dict(self) -> Dict[str, Any]:
        """Return the canonical persisted construction contract.

        Args:
            None.

        Returns:
            Dict[str, Any]: JSON-compatible construction contract for one
            workspace.
        """
        if self.kind == "preset":
            return {"kind": self.kind, "preset": self.preset}
        if self.kind == "backends":
            return {"kind": self.kind, "config": deepcopy(self.backends)}
        if self.kind == "runtime":
            return {"kind": self.kind, "identifiers": deepcopy(self.backends)}
        return {"kind": self.kind}


@dataclass(frozen=True)
class WorkspaceBackendSetup:
    """Bundle a bound backend registry with its construction provenance.

    Args:
        backends (BackendRegistry): Backend registry bound to the workspace.
        construction (WorkspaceConstruction): Construction provenance used by
            manifest export.
        preset_spec (Any): Resolved preset `ConfigSpec`, or `None` when explicit
            backends or default backend config were used.
    """

    backends: BackendRegistry
    construction: WorkspaceConstruction
    preset_spec: Any = None

    @property
    def preset(self) -> Optional[str]:
        """Return the canonical selected preset identifier, when present."""
        return self.construction.preset


def backend_identifiers_from_config(
    backends: Mapping[str, Any],
    *,
    resolver: ModuleResolver,
) -> Dict[str, str]:
    """Resolve a configured Backend topology without constructing providers.

    Args:
        backends: Complete configured Backend mapping.
        resolver: Module authority used to canonicalize Backend identifiers.

    Returns:
        Dict[str, str]: Effective Backend names mapped to canonical identifiers.

    Raises:
        TypeError: If the configuration or ``aux`` group is malformed.
        ValueError: If no enabled Backends remain or effective names collide.
        KeyError: If a Backend identifier is not installed.
    """
    if not isinstance(backends, Mapping):
        raise TypeError("Backend configuration must be a mapping")
    entries: list[tuple[Any, Any]] = []
    for alias, spec in backends.items():
        if alias == "aux":
            if spec is None:
                continue
            if not isinstance(spec, Mapping):
                raise TypeError("Backend configuration 'aux' must be a mapping")
            entries.extend(spec.items())
        else:
            entries.append((alias, spec))
    identifiers: Dict[str, str] = {}
    for alias, spec in entries:
        if not spec:
            continue
        request = normalize_backend_leaf(str(alias), spec)
        name = str(request.name)
        if name in identifiers:
            raise ValueError(f"Backend configuration resolves duplicate effective name {name!r}")
        identifiers[name] = BackendType.identifier_of(
            request.identifier,
            resolver=resolver,
        )
    if not identifiers:
        raise ValueError("Backend configuration does not define any enabled backends")
    return identifiers


def resolve_workspace_backend_setup(
    ws_id: str,
    backends: Optional[Mapping[str, Any]],
    preset: Optional[str],
    *,
    manager: Any,
    resolver: ModuleResolver,
) -> WorkspaceBackendSetup:
    """Resolve workspace backend inputs into a registry and replay policy.

    Args:
        ws_id (str): Workspace identifier used to bind backend namespaces and
            apply preset `{ws_id}` placeholders.
        backends (Mapping[str, Any] | None): Explicit backend config or live
            backend instances. Explicit backends skip preset resolution.
        preset (str | None): Optional workspace preset under
            `heavenbase.workspace.presets`.

    Returns:
        WorkspaceBackendSetup: Bound backend registry, construction provenance,
        and optional resolved preset spec.
    """
    construction = WorkspaceConstruction.from_inputs(backends, preset, resolver=resolver)
    backend_config = backends
    preset_spec = None
    if backend_config is None and preset is not None:
        preset_spec = resolve_workspace_preset(preset, manager=manager)
        construction = WorkspaceConstruction.from_preset(preset_spec.get("preset"))
        backend_config = resolve_workspace_backends(preset_spec, ws_id=ws_id, manager=manager)
    registry = (
        BackendRegistry.from_config(ws_id=ws_id, resolver=resolver, config=manager)
        if backend_config is None
        else build_backend_registry(ws_id, backend_config, resolver=resolver, config=manager)
    )
    return WorkspaceBackendSetup(registry, construction, preset_spec)


def build_backend_registry(
    ws_id: str,
    backends: Mapping[str, Any],
    *,
    resolver: ModuleResolver,
    config: Any,
) -> BackendRegistry:
    """Build a backend registry from config or live backend instances.

    Args:
        ws_id (str): Workspace identifier used to bind backend namespaces.
        backends (Mapping[str, Any]): Backend config mappings or live `Backend`
            instances keyed by workspace backend name.

    Returns:
        BackendRegistry: Registry populated with bound backend instances.
    """
    registry = BackendRegistry(resolver=resolver, config=config)
    if backends and all(isinstance(backend, Backend) for backend in backends.values()):
        source = backends
        rebound = [(backend.name, backend.ws_id) for backend in source.values() if backend.ws_id and backend.ws_id != ws_id]
        if rebound:
            names = ", ".join(f"{name!r} ({owner!r})" for name, owner in rebound)
            raise ValueError(f"Live Backend instances already bound to another workspace cannot enter {ws_id!r}: {names}")
        for backend in source.values():
            registry.register(backend)
            backend.bind_workspace(ws_id)
    else:
        source = _load_configured_backends(backends, ws_id=ws_id, resolver=resolver, config=config)
        for backend in source.values():
            registry.register(backend)
    return registry
