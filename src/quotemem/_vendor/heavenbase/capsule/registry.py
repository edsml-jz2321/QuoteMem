"""Entity-backed Capsule registry over an injected row-store port."""

from __future__ import annotations

__all__ = ["CapsuleRegistry", "capsule_registry", "reset_capsule_registry", "setup_capsule_registry"]

import threading
from time import time
from weakref import WeakValueDictionary

from heavenbase.utils import Any, Dict, Mapping, Optional, dmerge, resolve_interpolations
from heavenbase.utils.config import ConfigManager

from ..modules import ModuleResolver
from .authority import CapsuleAuthority
from .errors import CapsuleNotFoundError
from .manifest import CapsuleManifest
from .runtime import Capsule
from .store import CapsuleRowStore
from .versions import select_versioned_row

_DEFAULT_REGISTRY: "CapsuleRegistry | None" = None
_AUTHORITY_REGISTRIES: "WeakValueDictionary[tuple[int, int], CapsuleRegistry]" = WeakValueDictionary()
# Never held while a Context lock is taken: a Context-lock holder may ask for a registry.
_REGISTRY_LOCK = threading.RLock()
# Counts reset_capsule_registry() calls, under the registry lock, so a facade built outside the lock while a reset
# ran is not cached after it.
_RESET_GENERATION = 0


def _registry_root(config: ConfigManager) -> Dict[str, Any]:
    """Return the effective executable-registry workspace configuration."""
    default = ((config.load_default().get("heavenbase", {}) or {}).get("capsule", {}) or {}).get("registry", {}) or {}
    runtime = config.get("heavenbase.capsule.registry", default={}) or {}
    if not isinstance(default, Mapping) or not isinstance(runtime, Mapping):
        raise TypeError("heavenbase.capsule.registry config must be a mapping")
    cfg = dmerge([default, runtime])
    if "backend" in runtime:
        cfg["backend"] = runtime["backend"]
    return resolve_interpolations(cfg)


def _effective_registry_config(authority: CapsuleAuthority, override: Mapping[str, Any] | None) -> Dict[str, Any]:
    cfg = dmerge([_registry_root(authority.config), dict(override or {})])
    if override and "backend" in override:
        cfg["backend"] = override["backend"]
    return cfg


def _workspace_store(authority: CapsuleAuthority, config: Mapping[str, Any]) -> Any:
    """Resolve the Context-owned executable store without importing workspace."""
    context = getattr(authority.config, "_context", None)
    if context is None or context.modules() is not authority.resolver:
        raise RuntimeError("CapsuleRegistry needs a resolver and ConfigManager bound to the same Context")
    return context._hidden_store(config)


class CapsuleRegistry:
    """Persist Capsules through an injected ``sys-capsule`` row store.

    The registry owns Capsule identity, conflict, verification, and manifest
    semantics. It does not construct a HeavenBase workspace and has no Toolkit
    operations.
    """

    def __init__(
        self,
        store: CapsuleRowStore,
        *,
        registry_config: Mapping[str, Any] | None = None,
        resolver: ModuleResolver | None = None,
        config: ConfigManager | None = None,
    ) -> None:
        """Bind a Capsule registry to one injected row store.

        Args:
            store (CapsuleRowStore): Persistence port for ``sys-capsule`` rows.
            registry_config (Mapping[str, Any] | None): Optional registry
                workspace overrides merged with config defaults.
            resolver (ModuleResolver | None): Context-owned module resolver.
                The default Context is selected when omitted.
            config (ConfigManager | None): Config authority paired with
                `resolver`. The standalone process config is selected when omitted.

        Returns:
            None: This constructor does not return a value.
        """
        if not isinstance(store, CapsuleRowStore):
            raise TypeError("CapsuleRegistry store must implement the CapsuleRowStore protocol")
        self.authority = CapsuleAuthority.resolve(resolver=resolver, config=config)
        self.config = dict(registry_config or {})
        self.store = store

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any] | None = None,
        *,
        store: CapsuleRowStore,
        resolver: ModuleResolver | None = None,
        config: ConfigManager | None = None,
        **values: Any,
    ) -> "CapsuleRegistry":
        """Build a registry over an explicitly supplied row store.

        Args:
            data (Mapping[str, Any] | None): Optional registry configuration
                mapping merged with keyword overrides.
            store (CapsuleRowStore): Persistence port for ``sys-capsule`` rows.
            resolver (ModuleResolver | None): Context-owned module resolver.
                The default Context is selected when omitted.
            config (ConfigManager | None): Config authority paired with
                `resolver`. The standalone process config is selected when omitted.
            **values (Any): Additional registry configuration overrides.

        Returns:
            CapsuleRegistry: Registry bound to ``store``.
        """
        cfg = dict(data or {})
        cfg.update(values)
        return cls(store, registry_config=cfg, resolver=resolver, config=config)

    def register(
        self,
        capsule: Capsule | CapsuleManifest | Mapping[str, Any],
        *,
        overwrite: bool = False,
        reason: str = "",
    ) -> str:
        """Register one Capsule manifest.

        Args:
            capsule (Capsule | CapsuleManifest | Mapping[str, Any]): Capsule
                object, manifest, or manifest mapping to persist.
            overwrite (bool): Whether an existing row with different content may
                be replaced. Defaults to ``False``.
            reason (str): Registry write reason stored with the row. Defaults to
                ``""``.

        Returns:
            str: Stable Capsule id written to the row store.
        """
        manifest = (
            capsule.manifest
            if isinstance(capsule, Capsule)
            else CapsuleManifest.from_dict(capsule.to_dict() if isinstance(capsule, CapsuleManifest) else capsule)
        )
        stored_capsule = (
            capsule
            if isinstance(capsule, Capsule)
            else Capsule.from_manifest(
                manifest,
                resolver=self.authority.resolver,
                config=self.authority.config,
            )
        )
        existing = self.store.get(manifest.id)
        if existing and not overwrite and existing.get("fingerprints") != manifest.fingerprints:
            raise ValueError(f"Capsule {manifest.id!r} already exists with different content; use a new version or overwrite=True")
        self.store.upsert(stored_capsule.to_row(existing, reason=reason))
        return manifest.id

    def load(
        self,
        capsule_id: Optional[str] = None,
        *,
        name: Optional[str] = None,
        namespace: str = "default",
        version: Optional[str] = None,
    ) -> Capsule:
        """Load and verify one Capsule by id or versioned name.

        Args:
            capsule_id (str | None): Stable Capsule id. Defaults to ``None``.
            name (str | None): Capsule name when loading by identity. Defaults
                to ``None``.
            namespace (str): Capsule namespace when loading by name. Defaults
                to ``"default"``.
            version (str | None): Capsule version when loading by name. Defaults
                to latest matching row.

        Returns:
            Capsule: Verified lazy Capsule restored from the row store.

        Raises:
            CapsuleNotFoundError: If no matching row exists.
        """
        row = self._find(capsule_id, name=name, namespace=namespace, version=version)
        if row is None:
            raise CapsuleNotFoundError(capsule_id or f"{namespace}/{name}:{version or '*'}")
        capsule = Capsule.from_row(
            row,
            resolver=self.authority.resolver,
            config=self.authority.config,
        )
        capsule.verify()
        return capsule

    def _find(
        self,
        capsule_id: Optional[str] = None,
        *,
        name: Optional[str] = None,
        namespace: str = "default",
        version: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        if capsule_id:
            return self.store.get(capsule_id)
        matches = [row for row in self.store.rows() if row.get("name") == name and row.get("namespace") == namespace]
        return select_versioned_row(matches, version=version)

    def list(self, namespace: Optional[str] = None, status: Optional[str] = "active") -> list[Dict[str, Any]]:
        """List matching Capsule rows.

        Args:
            namespace (str | None): Optional namespace filter. Defaults to
                ``None`` for all namespaces.
            status (str | None): Optional status filter. Defaults to
                ``"active"``. Pass ``None`` to include every status.

        Returns:
            list[Dict[str, Any]]: Matching ``sys-capsule`` rows.
        """
        rows = self.store.rows()
        if namespace is not None:
            rows = [row for row in rows if row.get("namespace") == namespace]
        if status is not None:
            rows = [row for row in rows if row.get("status") == status]
        return rows

    def delete(self, capsule_id: str, *, tombstone: bool = True) -> bool:
        """Delete or tombstone one Capsule row.

        Args:
            capsule_id (str): Stable Capsule id to remove.
            tombstone (bool): Mark deleted instead of physically deleting.
                Defaults to ``True``.

        Returns:
            bool: Whether a row was deleted or tombstoned.
        """
        if not tombstone:
            return self.store.delete(capsule_id)
        row = self.store.get(capsule_id)
        if row is None:
            return False
        row["status"] = "deleted"
        row["updated_at"] = time()
        self.store.set(row)
        return True

    def export(self, capsule_id: str) -> Dict[str, Any]:
        """Return a portable Capsule manifest payload.

        Args:
            capsule_id (str): Stable Capsule id to export.

        Returns:
            Dict[str, Any]: JSON-compatible manifest dictionary.
        """
        return self.load(capsule_id).to_dict()

    def import_manifest(self, payload: Mapping[str, Any], *, overwrite: bool = False) -> str:
        """Import and register a Capsule manifest payload.

        Args:
            payload (Mapping[str, Any]): Portable manifest mapping to register.
            overwrite (bool): Whether an existing row may be replaced. Defaults
                to ``False``.

        Returns:
            str: Stable Capsule id written to the row store.
        """
        return self.register(
            Capsule.from_manifest(
                payload,
                resolver=self.authority.resolver,
                config=self.authority.config,
            ),
            overwrite=overwrite,
            reason="import",
        )


def _new_registry(authority: CapsuleAuthority, override: Mapping[str, Any] | None) -> CapsuleRegistry:
    config = _effective_registry_config(authority, override)
    owner = _workspace_store(authority, config)
    store = owner.rows(Capsule, extension="capsule")
    return CapsuleRegistry(
        store,
        registry_config=config,
        resolver=authority.resolver,
        config=authority.config,
    )


def capsule_registry(
    registry_config: Mapping[str, Any] | None = None,
    *,
    resolver: ModuleResolver | None = None,
    config: ConfigManager | None = None,
) -> CapsuleRegistry:
    """Return the Capsule registry for one exact runtime authority.

    Args:
        registry_config (Mapping[str, Any] | None): Optional isolated registry
            configuration. When provided, a fresh registry is returned instead
            of the cached default.
        resolver (ModuleResolver | None): Context-owned module resolver.
            The default Context is selected when omitted.
        config (ConfigManager | None): Config authority paired with
            `resolver`. The standalone process config is selected when omitted.

    Returns:
        CapsuleRegistry: Registry facade for the resolved authority. A
        facade whose build was running when `reset_capsule_registry()`
        started is returned to this call but not cached, so later calls
        build one from the configuration the reset left.
    """
    authority = CapsuleAuthority.resolve(resolver=resolver, config=config)
    if registry_config is not None:
        return _new_registry(authority, registry_config)
    global _DEFAULT_REGISTRY
    while True:
        with _REGISTRY_LOCK:
            registry = _cached_registry(authority)
            if registry is not None:
                if resolver is None and config is None:
                    _DEFAULT_REGISTRY = registry
                return registry
            generation = _RESET_GENERATION
        # Opening the store takes the Context lock, which the caller may already
        # hold (a compute hook inside a workspace write), so the registry lock is
        # never held while the Context lock is taken.
        built = _new_registry(authority, None)
        if _retired(built):
            continue
        with _REGISTRY_LOCK:
            registry = _cached_registry(authority)
            if registry is None:
                if _RESET_GENERATION != generation:
                    # A reset ran during the build, which read the configuration before it: the facade serves
                    # this call, as if the call had ended before the reset, and is never cached.
                    return built
                registry = _AUTHORITY_REGISTRIES[authority.identity] = built
            if resolver is None and config is None:
                _DEFAULT_REGISTRY = registry
            return registry


def _retired(registry: CapsuleRegistry) -> bool:
    """Return whether the hidden store under ``registry`` has left its Context generation."""
    owner = getattr(registry.store, "owner", None)
    return owner is not None and owner.retired


def _cached_registry(authority: CapsuleAuthority) -> CapsuleRegistry | None:
    """Return the cached facade of ``authority`` unless its store retired; the caller holds the registry lock."""
    registry = _AUTHORITY_REGISTRIES.get(authority.identity)
    if registry is not None and _retired(registry):
        _AUTHORITY_REGISTRIES.pop(authority.identity, None)
        return None
    return registry


def reset_capsule_registry(
    drop: bool = False,
    *,
    resolver: ModuleResolver | None = None,
    config: ConfigManager | None = None,
) -> None:
    """Retire cached Capsule registry facades for one authority or globally.

    Args:
        drop (bool): Whether to drop the Capsule registry's hidden workspace
            storage after retiring cached facades. Defaults to ``False``.
            A drop removes the storage of every open registry store and the
            storage that the ``heavenbase.capsule.registry`` configuration
            names, also when no store on it is open yet, such as in a new
            process or a new Context over the same root. That configured
            storage is opened only when it may exist: storage whose routes are
            all local files that do not exist is skipped, so a Context that
            never created it is not initialized for the drop, while storage on
            a server database is opened. A process-wide drop reaches every
            open root Context of the process, the default Context once it
            exists included, whether or not it used a Capsule registry; no
            default Context is built for it. A Context discarded without
            ``close()`` stays open until it is garbage-collected, so a drop
            still reaches it; close Contexts you no longer use. A closed
            Context owns no storage, so its rows stay in place. A process-wide
            drop treats a Context that a close or reset has marked as closed:
            configured storage that is not open yet is skipped and the reset
            returns normally, and a drop of open storage raises
            ``RuntimeError``; if that close or reset gives up, the Context is
            open again and still serves the rows. Other hidden stores of the Context,
            such as LLM cache rows, are kept.
        resolver (ModuleResolver | None): Context-owned module resolver whose
            cached facade should be retired. When both ``resolver`` and
            ``config`` are omitted, every cached facade is retired.
        config (ConfigManager | None): Config authority paired with
            `resolver`.

    Returns:
        None: This function does not return a value.

    Raises:
        Exception: The drop failure, unchanged, when ``drop`` is true and one
            Capsule registry store fails to open or drop. Every other Capsule
            registry store of the selected Contexts is still dropped and
            retired first.
        RuntimeError: One combined error, chained from the first failure,
            when several Capsule registry stores fail to open or drop.
    """
    global _DEFAULT_REGISTRY, _RESET_GENERATION
    # Each selected Context with the config authority whose registry configuration names its storage.
    contexts: dict[int, tuple[Any, ConfigManager]] = {}
    # Resolving an authority can open the Context's module service under the Context lock, which is never taken
    # while the registry lock is held.
    authority = None if resolver is None and config is None else CapsuleAuthority.resolve(resolver=resolver, config=config)
    with _REGISTRY_LOCK:
        _RESET_GENERATION += 1
        if authority is None:
            _DEFAULT_REGISTRY = None
            _AUTHORITY_REGISTRIES.clear()
            from ..context.runtime import _root_contexts

            for context in _root_contexts():
                contexts[id(context)] = (context, context.config)
        else:
            registry = _AUTHORITY_REGISTRIES.pop(authority.identity, None)
            if _DEFAULT_REGISTRY is registry:
                _DEFAULT_REGISTRY = None
            context = getattr(authority.config, "_context", None)
            if context is not None:
                contexts[id(context)] = (context, authority.config)
    if drop:
        from ..context.runtime import _contexts_with_hidden_stores, _local_storage_absent, _raise_drop_failures

        # The Contexts that hold a Capsule registry store, which the drop retires.
        holders = {id(context) for context in _contexts_with_hidden_stores(None)}
        # The facades above are already retired, so one Context's failed
        # drop must not leave the other Contexts' hidden storage in place.
        errors: list[Exception] = []
        for context, authority_config in contexts.values():
            # The configured storage, when it may hold rows. A closed Context owns no storage, and local
            # storage that does not exist is not opened, so a Context that never created it is not
            # initialized for the drop.
            storage = None
            if _is_open(context):
                try:
                    storage = _registry_root(authority_config)
                except Exception as exc:
                    errors.append(exc)
                if storage is not None and _local_storage_absent(storage, authority_config):
                    storage = None
            root = context
            while root.parent is not None:
                root = root.parent
            if storage is None and id(root) not in holders:
                continue
            with context._lock:
                if storage is not None and _is_open(context):
                    try:
                        # Open the store the configuration names, so the drop also removes rows on disk
                        # that no store of this process has opened yet.
                        context._hidden_store(storage).workspace
                    except Exception as exc:
                        errors.append(exc)
                try:
                    context._reset_hidden_stores(drop=True)
                except Exception as exc:
                    errors.append(exc)
        _raise_drop_failures(errors)


def _is_open(context: Any) -> bool:
    """Return whether a Context can open storage; a closed Context owns none to drop."""
    try:
        context._need_open()
    except RuntimeError:
        return False
    return True


def setup_capsule_registry(
    reset: bool = False,
    *,
    resolver: ModuleResolver | None = None,
    config: ConfigManager | None = None,
) -> Dict[str, Any]:
    """Initialize one authority-owned Capsule registry workspace.

    Args:
        reset (bool): Whether to retire cached facades before initialization.
            Defaults to ``False``.
        resolver (ModuleResolver | None): Context-owned module resolver.
            The default Context is selected when omitted.
        config (ConfigManager | None): Config authority paired with
            `resolver`. The standalone process config is selected when omitted.

    Returns:
        Dict[str, Any]: Effective registry workspace configuration.
    """
    if reset:
        reset_capsule_registry(resolver=resolver, config=config)
    registry = capsule_registry(resolver=resolver, config=config)
    registry.store.rows()
    return dict(registry.config)
