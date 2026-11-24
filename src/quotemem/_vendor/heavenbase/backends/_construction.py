"""Shared Backend leaf normalization and factory execution."""

from __future__ import annotations

__all__ = ["BackendFactory", "BackendRequest", "adapt_backend_factory", "construct_backend", "normalize_backend_leaf"]

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ..modules import ModuleResolver
    from .base import Backend

from ..utils.ids import check_id, check_workspace_id


@runtime_checkable
class BackendFactory(Protocol):
    """Construct one Backend from explicit runtime authorities."""

    def __call__(
        self,
        name: str,
        provider_config: Mapping[str, Any],
        *,
        resolver: "ModuleResolver | None",
        config: Any,
    ) -> "Backend":
        """Return one configured Backend instance.

        Args:
            name: Requested Backend instance name.
            provider_config: Provider-owned construction arguments.
            resolver: Context-owned module authority. ``None`` is reserved for
                direct control-plane bootstrap before the module catalog exists.
            config: Config authority used by provider construction and runtime policy.

        Returns:
            Backend: Configured Backend instance.
        """


@dataclass(frozen=True)
class BackendRequest:
    """Normalized construction request shared by bootstrap and workspaces.

    Args:
        identifier: Canonical or aliased Backend type selector.
        name: Requested Backend instance name.
        provider_config: Provider-owned arguments after control fields are removed.
        module: Explicit direct-bootstrap module, when supplied.
    """

    identifier: str
    name: str
    provider_config: Mapping[str, Any]
    module: str | None = None


def normalize_backend_leaf(
    alias: str,
    spec: Mapping[str, Any],
    *,
    ws_id: str | None = None,
    root: str | None = None,
    require_module: bool = False,
) -> BackendRequest:
    """Normalize one flat Backend mapping without selecting a provider.

    Args:
        alias: Mapping key used as the Backend instance-name fallback.
        spec: Flat Backend mapping with required `type` and optional `name` and
            direct-bootstrap `module` fields.
        ws_id: Optional workspace namespace injected into provider config.
        root: Optional root used to normalize a direct file-URI database.
        require_module: Whether the direct-bootstrap module must be present.

    Returns:
        BackendRequest: Provider-neutral normalized construction request.

    Raises:
        TypeError: If the leaf is not a mapping or control fields have invalid types.
        ValueError: If `type`, `name`, or a required `module` is missing or invalid.
    """
    if not isinstance(spec, Mapping):
        raise TypeError(f"Backend {alias!r} config must be a mapping")
    raw = dict(spec)
    selected = raw.pop("type", None)
    if not isinstance(selected, str) or not selected.strip():
        raise ValueError(f"Backend {alias!r} config requires a non-empty 'type'")
    identifier = check_id(selected.strip().lower(), "backend identifier")
    name = check_id(str(raw.pop("name", alias) or alias), "backend instance name")
    module_value = raw.pop("module", None)
    if module_value is not None and (not isinstance(module_value, str) or not module_value.strip()):
        raise TypeError(f"Backend {alias!r} module must be a non-empty string")
    module = module_value.strip() if isinstance(module_value, str) else None
    if require_module and module is None:
        raise ValueError(f"Backend {alias!r} direct bootstrap requires 'module'")
    database = raw.get("database")
    if root is not None and isinstance(database, str) and database.startswith("file://"):
        from ..utils.files import normalize_file_uri

        raw["database"] = normalize_file_uri(database, home=root)
    if ws_id is not None:
        raw["ws_id"] = check_workspace_id(ws_id)
    return BackendRequest(identifier, name, raw, module)


def adapt_backend_factory(target: Any, factory: Any = None) -> BackendFactory:
    """Adapt a restored Backend class or explicit factory to one contract.

    Args:
        target: Restored Backend subclass.
        factory: Optional callable with the `BackendFactory` signature.

    Returns:
        BackendFactory: Validated callable accepting provider arguments plus
            explicit ``resolver`` and ``config`` keyword authorities.

    Raises:
        TypeError: If the target is not a Backend subclass or the factory is not callable.
    """
    from .base import Backend

    if not isinstance(target, type) or not issubclass(target, Backend):
        raise TypeError(f"Backend record resolved {target!r}, expected a Backend subclass")
    selected = target.from_config if factory is None else factory
    if not callable(selected):
        raise TypeError("Backend factory must be callable")
    return selected


def construct_backend(
    request: BackendRequest,
    factory: BackendFactory,
    *,
    resolver: "ModuleResolver | None",
    config: Any,
) -> "Backend":
    """Execute one Backend factory and validate the requested identity.

    Args:
        request: Normalized Backend construction request.
        factory: Adapted factory to execute.
        resolver: Context-owned module authority. ``None`` is allowed only for
            direct control-plane bootstrap.
        config: Config authority retained by the constructed Backend.

    Returns:
        Backend: Constructed and identity-validated Backend instance.

    Raises:
        TypeError: If the factory does not return a Backend.
        ValueError: If the returned name, type, or workspace binding differs
            from the normalized request.
    """
    from .base import Backend

    if not callable(getattr(config, "get", None)):
        raise TypeError("Backend config authority must provide get(path, default)")
    backend = factory(request.name, dict(request.provider_config), resolver=resolver, config=config)
    if not isinstance(backend, Backend):
        raise TypeError(f"Backend factory returned {backend!r}, expected Backend")
    if backend.name != request.name:
        raise ValueError(f"Backend factory returned name {backend.name!r}, expected {request.name!r}")
    if backend.type != request.identifier:
        raise ValueError(f"Backend factory returned type {backend.type!r}, expected {request.identifier!r}")
    requested_workspace = request.provider_config.get("ws_id")
    if requested_workspace is not None and backend.ws_id != requested_workspace:
        raise ValueError(f"Backend factory returned workspace {backend.ws_id!r}, expected {requested_workspace!r}")
    if backend._module_resolver is not resolver:
        raise ValueError("Backend factory did not retain the requested module resolver")
    if backend._config is not config:
        raise ValueError("Backend factory did not retain the requested config authority")
    return backend
