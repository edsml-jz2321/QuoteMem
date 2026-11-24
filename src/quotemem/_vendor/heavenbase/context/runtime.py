"""Context composition root for Backends, Registries, and configuration."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping
from contextlib import AbstractContextManager, nullcontext
from contextvars import ContextVar
from importlib import import_module
import os
from pathlib import Path
import sys
import threading
import time
from types import ModuleType
import weakref
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from heavenbase._lazy import bind_late
from heavenbase.utils.registry import Registry, RegistryStore, identifier_key

from .bootstrap import BootstrapSpec, load_bootstrap
from .locks import _at_exit, _ATTEMPTS, _ContextLock, _Until, _WorkspaceLock, _WorkspaceLockTable, holds_any_lock
from ..utils.config.manager import CM_HVNB, ConfigManager
from ..utils.ids import check_workspace_id

if TYPE_CHECKING:
    from ..modules import ModuleService

_ACTIVE_WORKSPACE_PATH = "heavenbase.registry.active.workspace"
_UNRESOLVED_ACTIVE_WORKSPACE = object()
_BOOTSTRAPPING_CONFIG: ContextVar[ConfigManager | None] = ContextVar("control_backend_config", default=None)
# Seconds that Context.close() and reset_default_local_state() wait, in all, for other threads' work on the Context
# (see `_shutdown_deadline`).
_SHUTDOWN_WAIT = 30.0
# Seconds between the checks of a reset that waits for active workspace reads to end.
_READS_POLL = 0.02
# Seconds between the checks of a close or reset that waits for resolver operations or another thread's reset to end.
_SHUTDOWN_POLL = 0.02


def _shutdown_deadline() -> float | None:
    """Return until when a close or reset called on this thread may wait for other threads' work, or ``None``.

    A thread that holds a Context or workspace lock, or runs a database drop,
    may hold something that work waits for, so waiting could deadlock: it is
    refused at once instead, as before these waits existed. Any other thread
    waits for at most `_SHUTDOWN_WAIT` seconds in all.
    """
    from ..database.registry import DatabaseEngineRegistry

    if holds_any_lock() or not DatabaseEngineRegistry._may_wait(None):
        return None
    return time.monotonic() + _SHUTDOWN_WAIT


class _TrackedRead:
    """Block of one workspace read, which a Context and its ancestors count among their ``_readers``.

    Args:
        chain (tuple[Context, ...]): The Context and its ancestors.
        check (Callable[[], None]): Liveness check, made as the block is entered.
        lock (_WorkspaceLock | None): Lock whose shared side the block holds.
        scope (AbstractContextManager[Any] | None): Block to enter inside, whose
            exit changes nothing when it runs again.
    """

    __slots__ = ("_chain", "_check", "_lock", "_scope")

    def __init__(
        self,
        chain: "tuple[Context, ...]",
        check: Callable[[], None],
        lock: _WorkspaceLock | None,
        scope: AbstractContextManager[Any] | None,
    ) -> None:
        self._chain = chain
        self._check = check
        self._lock = lock
        self._scope = scope

    def __enter__(self) -> None:
        try:
            if self._lock is not None:
                self._lock.acquire_read()
            with self._chain[0]._read_lock:
                self._check()
                for context in self._chain:
                    context._readers.add(self)
            if self._scope is not None:
                self._scope.__enter__()
        except BaseException:
            for _ in _ATTEMPTS:
                try:
                    self._end()
                    break
                except BaseException:
                    pass
            raise

    @property
    def __exit__(self) -> Callable[..., Any]:
        return _at_exit(self._end)

    def _end(self) -> None:
        try:
            if self._scope is not None:
                self._scope.__exit__(None, None, None)
        finally:
            for context in self._chain:
                context._readers.discard(self)
            if self._lock is not None:
                self._lock._leave_read(0)


class Context:
    """Own machine services and memoized persistent Registry namespaces.

    Backend construction remains lazy. A root Context owns its machine
    composition; child contexts are workspace-selection views over that exact
    Backend, ModuleService, and ConfigManager authority.

    Args:
        bootstrap (BootstrapSpec): Validated machine-service configuration.
        config (ConfigManager | None): ConfigManager to bind to this Context.
            HeavenBase constructs one below `bootstrap.root` when omitted.
        backend (Any): Optional already-constructed Backend for a child Context.
        modules (ModuleService | None): Module authority paired with a supplied
            child Backend.
        owns_backend (bool): Whether `close` disposes the supplied Backend.
        parent (Context | None): Parent Context when resources are inherited.
    """

    def __init__(
        self,
        bootstrap: BootstrapSpec,
        *,
        config: ConfigManager | None = None,
        backend: Any = None,
        modules: "ModuleService | None" = None,
        owns_backend: bool = True,
        parent: "Context | None" = None,
    ) -> None:
        """Construct one machine composition root or child workspace view.

        Returns:
            None: This initializer does not return a value.
        """
        self.bootstrap = bootstrap
        self.root = bootstrap.root
        self._bootstrap_workspace_id = check_workspace_id(bootstrap.workspace)
        self.identity_token = parent.identity_token if parent is not None else uuid4().hex
        self._authority_token = parent.authority_token if parent is not None else uuid4().hex
        self._active_workspace_id: str | None | object = self._bootstrap_workspace_id if parent is not None else _UNRESOLVED_ACTIVE_WORKSPACE
        self.parent = parent
        self.config = config or ConfigManager(root=self.root, setup=True)
        self._backend = backend
        self._owns_backend = owns_backend
        self._registries: dict[str, Registry[Any, Any]] = {}
        self._registry_options: dict[str, tuple[Any, ...]] = {}
        self._registry_store: RegistryStore | None = None
        self._modules: ModuleService | None = modules
        self._hidden_stores: dict[str, Any] = parent._hidden_stores if parent is not None else {}
        # Hidden stores owned by one object each, such as an LLM cache built from a config; each is released with its owner.
        self._private_stores: weakref.WeakSet[Any] = parent._private_stores if parent is not None else weakref.WeakSet()
        self._lock: _ContextLock = parent._lock if parent is not None else _ContextLock()
        self._workspace_locks: _WorkspaceLockTable = parent._workspace_locks if parent is not None else _WorkspaceLockTable(self._lock)
        self._read_lock = parent._read_lock if parent is not None else threading.RLock()
        self._readers: set[_TrackedRead] = set()  # workspace reads running (see `_track_read`)
        self._chain: tuple["Context", ...] | None = None
        self._workspace_registry = Registry[str, Any](
            "live-workspaces",
            normalize=check_workspace_id,
            value_policy="process_identity",
        )
        self._live_workspaces = Registry[str, Any](
            "open-workspaces",
            normalize=check_workspace_id,
            value_policy="process_identity",
        )
        self._workspace_lock = threading.RLock()
        # Default selection can hold the config manager's lock, so it must
        # check lifecycle state without taking the Context lock.
        self._lifecycle_state_lock = threading.Lock()
        self._closed = False
        self._closing = False
        self._resetting = False
        self._lifecycle_owner: int | None = None
        if parent is not None and self.config is not parent.config:
            raise ValueError("Child Contexts must share their parent's ConfigManager")
        owns_config = parent is None or self.config is not parent.config
        if owns_config:
            # Claim the manager under its lock: default_context() decides between
            # adopting CM_HVNB's owner and building one under that same lock, so
            # a concurrent Context.load(config=CM_HVNB) cannot slip in between.
            with self.config._lock:
                previous_owner = getattr(self.config, "_context", None)
                if previous_owner is not None and previous_owner is not self:
                    raise ValueError("ConfigManager is already bound to another Context")
                if self.config is CM_HVNB:
                    # An installed default with its own manager may have opened
                    # process-only scopes. The owner always binds its own Registry.
                    self.config._registry = None
                    self.config._setup_pending = True
                if self.config._registry is None or not len(self.config._registry):
                    # Opening the lazy default can bind its factory during a
                    # refresh. The refresh decides whether to adopt the store.
                    opening = threading.get_ident() in self.config._opening_threads
                    self.config._bind_registry_factory(self._config_registry, preserve_fallback=opening)
                self.config._context = self
        if parent is None:
            with _ROOT_CONTEXTS_LOCK:
                _ROOT_CONTEXTS.add(self)

    def _config_registry(self) -> Registry[Any, Any]:
        """Open the Registry that persists this Context's ConfigManager.

        For ``CM_HVNB``, a fault of the bootstrap is raised as a ``ValueError``
        naming the bootstrap (see :func:`_build_bootstrap_backend`). Its
        ConfigManager can then use process-only scopes.
        """
        if self.config is CM_HVNB:
            _build_bootstrap_backend(self)
        return self.registry("system-config")

    @staticmethod
    def _config_store_persists(registry: Registry[Any, Any]) -> bool:
        """Whether this Registry's store keeps config beyond the process lifetime."""
        from .registry import HeavenBaseRegistryStore

        store = registry._store
        if not isinstance(store, HeavenBaseRegistryStore):
            return registry.persistent
        backend = store.backend
        database = getattr(backend, "db", None)
        return getattr(backend, "identifier", None) != "inmem" and not getattr(getattr(database, "spec", None), "in_memory", False)

    @classmethod
    def load(
        cls,
        bootstrap: Mapping[str, Any] | None = None,
        *,
        home_path: str | Path | None = None,
        config: ConfigManager | None = None,
    ) -> "Context":
        """Load a Context without opening its selected Backend.

        Args:
            bootstrap (Mapping[str, Any] | None): Explicit bootstrap mapping.
                When omitted, the home or packaged bootstrap is selected.
            home_path (str | Path | None): Optional bootstrap YAML file.
            config (ConfigManager | None): ConfigManager to bind lazily.

        Returns:
            Context: Lazily constructed machine-service composition root.

        Raises:
            ValueError: If the selected bootstrap configuration is invalid.
        """
        return cls(load_bootstrap(bootstrap, home_path=home_path), config=config)

    def _need_open(self) -> None:
        if self._closed:
            raise RuntimeError("Context is closed")
        if self._closing:
            raise RuntimeError("Context close is in progress")
        if self._resetting:
            raise RuntimeError("Context reset is in progress")
        if self.parent is not None:
            self.parent._need_open()
            if self._authority_token != self.parent.authority_token:
                raise RuntimeError("Context belongs to a retired parent authority generation")

    @property
    def authority_token(self) -> str:
        """Return the current root resource-generation identity."""
        return self._authority_token

    @property
    def active_workspace_id(self) -> str | None:
        """Return this Context's in-memory active-workspace projection.

        Root Contexts resolve the persisted profile selection once and retain
        that value for their lifetime. Child Contexts begin with their explicit
        local selection. Neither form polls for later profile changes.

        Returns:
            str | None: Explicitly active workspace id, or ``None`` when the
            configured default workspace is the effective fallback.
        """
        with self._lock:
            self._need_open()
            if self._active_workspace_id is _UNRESOLVED_ACTIVE_WORKSPACE:
                with self.config.scoped(self.config.base_scope):
                    value = self.config.get(_ACTIVE_WORKSPACE_PATH, default=None)
                self._active_workspace_id = None if value is None else check_workspace_id(str(value))
            return self._active_workspace_id

    @property
    def default_workspace_id(self) -> str:
        """Return the stable configured default-workspace identifier."""
        self._need_open()
        with self.config.scoped(self.config.base_scope):
            return check_workspace_id(self.config.get("heavenbase.registry.default_workspace", default="default"))

    @property
    def effective_workspace_id(self) -> str:
        """Return the active workspace id, falling back to the configured default."""
        return self.active_workspace_id or self.default_workspace_id

    def activate_workspace(self, workspace: str) -> str:
        """Select one registered workspace for this Context.

        The selection is persisted in the Context's configuration profile.
        Other already-running Context projections remain unchanged.

        Args:
            workspace: Registered workspace identifier to activate.

        Returns:
            str: Normalized active workspace identifier.

        Raises:
            KeyError: If the workspace is not durably registered.
        """
        self._need_open()
        clean_id = check_workspace_id(workspace)
        from ..workspace import WorkspaceSpec

        WorkspaceSpec.load(clean_id, context=self)
        with self._lock:
            self._need_open()
            self.config.set(_ACTIVE_WORKSPACE_PATH, clean_id, scope=self.config.base_scope)
            self._active_workspace_id = clean_id
            from ..workspace.setup.lifecycle import _sync_entity_default

            _sync_entity_default(self)
        return clean_id

    def deactivate_workspace(self) -> str | None:
        """Clear this Context's explicit active-workspace selection.

        Args:
            None.

        Returns:
            str | None: Previously active workspace identifier, if any.
        """
        with self._lock:
            self._need_open()
            previous = self.active_workspace_id
            self.config.set(_ACTIVE_WORKSPACE_PATH, None, scope=self.config.base_scope)
            self._active_workspace_id = None
            from ..workspace.setup.lifecycle import _sync_entity_default

            _sync_entity_default(self)
            return previous

    def _shutdown_lock(self, action: str, deadline: float | None = None) -> _Until:
        """Return a block that holds the Context lock for a shutdown.

        Args:
            action (str): ``"close"`` or ``"reset"``, for the error message.
            deadline (float | None): ``time.monotonic()`` value until which to
                wait for another thread's hold (see `_shutdown_deadline`), or
                ``None`` to require the lock at once.

        Returns:
            _Until: Context manager; the caller marks or performs the shutdown
            holding the lock. Entering it raises ``RuntimeError`` if another
            thread holds the lock at once, or still at the deadline.
        """
        return _Until(self._lock, deadline, f"Cannot {action} Context while its Context lock is held by another thread")

    def _await_resolver_idle(self, modules: "ModuleService", deadline: float | None) -> None:
        """Wait, holding no HeavenBase lock, until no resolver operation of ``modules`` runs.

        The Context is marked closing or resetting meanwhile, so no new
        operation starts and the running ones end.

        Args:
            modules (ModuleService): Module authority of this root Context.
            deadline (float | None): ``time.monotonic()`` value until which to
                wait (see `_shutdown_deadline`); ``None`` does not wait.

        Raises:
            RuntimeError: If the calling thread runs one itself, or one still
                runs at the deadline.
        """
        modules._need_retirable()
        while modules._authority_users:
            if deadline is None or time.monotonic() >= deadline:
                raise RuntimeError("Cannot retire Context while resolver operations are active")
            time.sleep(_SHUTDOWN_POLL)

    def _await_reset_by_another_thread(self, deadline: float | None) -> None:
        """Wait until a reset that another thread runs on this Context ends.

        Args:
            deadline (float | None): ``time.monotonic()`` value until which to
                wait (see `_shutdown_deadline`); ``None`` does not wait.

        Raises:
            RuntimeError: If the reset still runs at the deadline.
        """
        while self._resetting:
            if deadline is None or time.monotonic() >= deadline:
                raise RuntimeError("Context reset is in progress")
            time.sleep(_SHUTDOWN_POLL)

    def _mark_shutdown(self, *, reset: bool = False) -> None:
        """Refuse new operations while holding the Context lock and the resolver admission lock.

        The operations that hold a resolver lease go on: the caller waits for
        them next (`_await_resolver_idle`), holding no HeavenBase lock.
        """
        me = threading.get_ident()
        modules = self._modules if self.parent is None else None
        admission = nullcontext() if modules is None else modules._authority_lock
        with self._lifecycle_state_lock, admission:
            if modules is not None:
                modules._need_retirable()
            self._lifecycle_owner = me  # first: an interrupted call takes back its own mark only
            self._closing = not reset
            self._resetting = reset

    def _abandon_shutdown(self) -> None:
        """Take back this thread's mark of a close or reset that does not go through, so that operations are admitted again."""
        with self._lifecycle_state_lock:
            if self._lifecycle_owner == threading.get_ident():
                self._closing = self._resetting = False
                self._lifecycle_owner = None

    def _finish_shutdown(self) -> None:
        """Publish completion without taking the Context lock again."""
        with self._lifecycle_state_lock:
            self._closed = self._closing
            self._closing = self._resetting = False
            self._lifecycle_owner = None

    def _context_chain(self) -> tuple["Context", ...]:
        chain = [self]
        while chain[-1].parent is not None:
            chain.append(chain[-1].parent)
        return tuple(chain)

    def _raise_nested_shutdown(self, action: str) -> None:
        """Reject retirement from code running under a Context or workspace lock."""
        if self._lock.owned():
            raise RuntimeError(f"Cannot {action} Context while holding its active Context lock")
        raise RuntimeError(f"Cannot {action} Context during an active workspace read or while holding another Context lock")

    def _track_read(
        self,
        check: Callable[[], None],
        lock: _WorkspaceLock | None = None,
        scope: AbstractContextManager[Any] | None = None,
    ) -> _TrackedRead:
        """Return a block that counts a read on each ancestor, for the idle-child close check and for resets.

        Args:
            check (Callable[[], None]): Liveness check that raises, as the
                block is entered, when the Context or workspace has retired.
            lock (_WorkspaceLock | None): Lock whose shared side the block
                holds, taken before the check.
            scope (AbstractContextManager[Any] | None): Config scope of the
                block.

        Returns:
            _TrackedRead: Context manager; the caller reads storage while the
            read is counted.
        """
        chain = self._chain
        if chain is None:
            chain = self._chain = self._context_chain()
        return _TrackedRead(chain, check, None if lock is None or lock.held() else lock, scope)

    def backend(self, name: str | None = None) -> Any:
        """Return the fully composed system Backend, constructing it lazily.

        Args:
            name (str | None): Expected configured Backend name. When supplied,
                a mismatch is rejected before construction.

        Returns:
            Any: Context-owned Backend instance.

        Raises:
            KeyError: If `name` does not match the configured Backend.
            RuntimeError: If the Context is closed.
            ValueError: If the configured module does not expose exactly one
                matching Backend implementation.
        """
        backend = self._control_backend(name)
        if backend._module_resolver is not None:
            return backend
        # A reset can replace an unfinished backend before we enter module
        # composition. Resolve and finish the current backend together.
        with self._lock:
            backend = self._control_backend(name)
            if backend._module_resolver is None:
                self.modules()
            return backend

    def _control_backend(self, name: str | None = None) -> Any:
        """Return the bootstrap Backend before module authority is available.

        This private seam exists only to break the composition cycle: the
        Backend persists the Registry that stores the module catalog, while
        the public :meth:`backend` surface must never expose that Backend until
        the catalog has been created and bound as its resolver.

        Args:
            name: Expected configured Backend name, when validation is needed.

        Returns:
            Any: Context-owned control Backend, possibly awaiting its one-shot
            ModuleService finalization.
        """
        self._need_open()
        expected = str(self.bootstrap.backend["name"])
        if name is not None and name != expected:
            raise KeyError(f"Unknown Context backend {name!r}; known={[expected]!r}")
        backend = self._backend
        if backend is None:
            with self._lock:
                self._need_open()
                if self._backend is None:
                    with self.config.scoped(self.config.base_scope):
                        from ..backends.base import Backend
                        from ..backends._construction import adapt_backend_factory, construct_backend, normalize_backend_leaf

                        spec = dict(self.bootstrap.backend)
                        database_config = spec.pop("_override", None)
                        bootstrap_backend_tags = spec.pop("tags", {})
                        if database_config is not None:
                            if not isinstance(database_config, Mapping):
                                raise TypeError("Bootstrap backend _override must be a database config mapping")
                            from ..database.config import DBEngine
                            from ..database.dialect.registry import BootstrapDialectCatalog

                            definitions = database_config.get("dialects", {})
                            spec["config_engine"] = DBEngine(
                                config=database_config,
                                manager=None,
                                dialects=BootstrapDialectCatalog(definitions),
                                bootstrap_backend_tags=bootstrap_backend_tags,
                            )
                        request = normalize_backend_leaf(
                            str(spec.get("name") or "system"),
                            spec,
                            ws_id=self._bootstrap_workspace_id,
                            root=self.root,
                            require_module=True,
                        )
                        module = import_module(request.module or "")
                        # A lazy package facade holds its exports only once read. A listed name
                        # the module does not define is skipped, as 0.1.2.2 scanned only bound
                        # names; an export whose import fails still raises.
                        exported = [getattr(module, name, None) for name in getattr(module, "__all__", ())]
                        candidates = tuple(
                            dict.fromkeys(
                                value
                                for value in [*vars(module).values(), *exported]
                                if isinstance(value, type)
                                and issubclass(value, Backend)
                                and value is not Backend
                                and getattr(value, "identifier", None) == request.identifier
                            )
                        )
                        if len(candidates) != 1:
                            raise ValueError(f"Bootstrap module {request.module!r} must expose exactly one Backend " f"with identifier {request.identifier!r}")
                        candidate = candidates[0]
                        token = _BOOTSTRAPPING_CONFIG.set(self.config)
                        try:
                            self._backend = construct_backend(
                                request,
                                adapt_backend_factory(candidate),
                                resolver=None,
                                config=self.config,
                            )
                        finally:
                            _BOOTSTRAPPING_CONFIG.reset(token)
                backend = self._backend
        return backend

    def registry(
        self,
        identifier: str,
        **options: Any,
    ) -> Registry[Any, Any]:
        """Open a caller-managed Registry namespace.

        The reserved `system-modules` namespace has one validated mutation
        protocol and is available only through :meth:`modules` and its
        read-only catalog.

        Args:
            identifier (str): Stable Registry namespace identifier.
            **options (Any): Optional codec, normalizer, and limit options
                forwarded to the underlying Registry loader.

        Returns:
            Registry[Any, Any]: Persistent Registry cached by namespace.

        Raises:
            ValueError: If ``identifier`` is ``system-modules`` or the
                namespace is already open with different options.
        """
        self._need_open()
        namespace = identifier_key(identifier, "registry identifier")
        if namespace == "system-modules":
            raise ValueError("Registry 'system-modules' is reserved; use Context.modules() and ModuleService.catalog")
        return self._open_registry(namespace, **options)

    def _open_registry(
        self,
        identifier: str,
        *,
        encode_key: Callable[[Any], Any] | None = None,
        decode_key: Callable[[Any], Any] | None = None,
        encode_value: Callable[[Any], Any] | None = None,
        decode_value: Callable[[Any], Any] | None = None,
        normalize: Callable[[Any], Any] | None = None,
        normalize_alias: Callable[[Any], Any] | None = None,
        normalize_group: Callable[[Hashable], Hashable] | None = None,
        max_items: int | None = None,
        max_bytes: int | None = None,
    ) -> Registry[Any, Any]:
        """Return one memoized persistent Registry namespace.

        Args:
            identifier (str): Stable Registry namespace identifier.
            encode_key (Callable[[Any], Any] | None): Optional JSON-safe key encoder.
            decode_key (Callable[[Any], Any] | None): Decoder paired with `encode_key`.
            encode_value (Callable[[Any], Any] | None): Optional JSON-safe value encoder.
            decode_value (Callable[[Any], Any] | None): Decoder paired with `encode_value`.
            normalize (Callable[[Any], Any] | None): Canonical key normalizer.
            normalize_alias (Callable[[Any], Any] | None): Alias normalizer.
            normalize_group (Callable[[Hashable], Hashable] | None): Group normalizer.
            max_items (int | None): Namespace item limit. `None` inherits the
                bootstrap policy, whose explicit `null` value is unbounded.
            max_bytes (int | None): Encoded snapshot limit. `None` inherits the
                bootstrap policy, whose explicit `null` value is unbounded.

        Returns:
            Registry[Any, Any]: Persistent Registry cached by namespace.

        Raises:
            RuntimeError: If the Context is closed.
            ValueError: If the namespace is already open with different codecs,
                normalizers, or limits.
        """
        self._need_open()
        namespace = identifier_key(identifier, "registry identifier")
        policy = self.bootstrap.registry
        effective_max_items = policy.get("max_items") if max_items is None else max_items
        effective_max_bytes = policy.get("max_bytes") if max_bytes is None else max_bytes
        options = (
            encode_key,
            decode_key,
            encode_value,
            decode_value,
            policy.get("hash", "sha256"),
            normalize,
            normalize_alias,
            normalize_group,
            effective_max_items,
            effective_max_bytes,
        )
        # Workspace reads reach here; an open namespace is returned without the
        # Context lock so a read does not queue behind a writer of another workspace.
        registry = self._registries.get(namespace)
        if registry is not None and self._registry_options.get(namespace) == options:
            return registry
        with self._lock:
            self._need_open()
            registry = self._registries.get(namespace)
            if registry is None:
                if self._registry_store is None:
                    from .registry import HeavenBaseRegistryStore

                    self._registry_store = HeavenBaseRegistryStore(
                        self._control_backend(),
                        control=lambda: self.config.scoped(self.config.base_scope),
                    )
                registry = Registry.load(
                    namespace,
                    store=self._registry_store,
                    encode_key=encode_key,
                    decode_key=decode_key,
                    encode_value=encode_value,
                    decode_value=decode_value,
                    hash_fn=policy.get("hash", "sha256"),
                    normalize=normalize,
                    normalize_alias=normalize_alias,
                    normalize_group=normalize_group,
                    max_items=effective_max_items,
                    max_bytes=effective_max_bytes,
                )
                self._registry_options[registry.identifier] = options
                self._registries[registry.identifier] = registry
            elif self._registry_options[namespace] != options:
                raise ValueError(f"Registry namespace {namespace!r} is already open with different codecs, normalizers, or limits")
            return registry

    def modules(self) -> "ModuleService":
        """Return the memoized central module lifecycle service.

        Access opens `system-modules` but performs no package discovery or
        registration. Explicit setup and workspace creation own built-in
        module indexing; runtime consumers subsequently read only Registry
        records.

        Args:
            None.

        Returns:
            ModuleService: Root machine authority over the durable
            `system-modules` namespace, shared unchanged by child views.

        Raises:
            RuntimeError: If the Context is closed.
        """
        self._need_open()
        modules = self._modules
        if modules is not None:
            return modules
        with self._lock:
            self._need_open()
            if self._modules is None:
                from ..modules import ModuleService
                from ..modules.artifacts import ModuleArtifactStore

                def restore_capsule(target: Any, resolver: "ModuleService") -> Any:
                    from ..capsule.errors import CapsuleNotFoundError
                    from ..capsule.registry import capsule_registry
                    from ..modules.artifacts import SourceUnavailable

                    try:
                        return capsule_registry(resolver=resolver, config=self.config).load(str(target))
                    except CapsuleNotFoundError as error:
                        raise SourceUnavailable(f"Capsule {target!r} is unavailable") from error

                service = ModuleService(
                    self._open_registry("system-modules"),
                    artifacts=ModuleArtifactStore(Path(self.root) / "artifacts" / "modules"),
                    authority_token=self.authority_token,
                    authority_config=self.config,
                    source_restorers={"capsule": restore_capsule},
                )
                self._control_backend()._finalize_bootstrap_resolver(service)
                self._modules = service
            return self._modules

    def _hidden_store(self, config: Mapping[str, Any], *, kind: str | None = None, entities: tuple[type, ...] = ()) -> Any:
        """Return the Context's shared hidden domain-workspace adapter for one service and configuration.

        Args:
            config (Mapping[str, Any]): Hidden workspace construction config.
            kind (str | None): Service that owns the store. ``None`` selects
                the stores that `_reset_hidden_stores` retires, such as the
                Capsule registry's.
            entities (tuple[type, ...]): Entities registered in every
                workspace the store opens.

        Returns:
            Any: The live store for ``kind`` and ``config``, created when absent.
        """
        self._need_open()
        owner = self
        while owner.parent is not None:
            owner = owner.parent
        from ..workspace.hidden_store import HiddenWorkspaceStore

        key = HiddenWorkspaceStore.key(config) if kind is None else f"{kind}:{HiddenWorkspaceStore.key(config)}"
        with owner._lock:
            owner._need_open()
            store = owner._hidden_stores.get(key)
            if store is None or store.retired:
                store = HiddenWorkspaceStore(owner, config, kind=kind, entities=entities)
                owner._hidden_stores[key] = store
            return store

    def _new_hidden_store(self, config: Mapping[str, Any], *, entities: tuple[type, ...] = ()) -> Any:
        """Return one hidden domain-workspace adapter owned by the caller.

        The Context holds it only weakly: it is released with the object that
        keeps it, and until then Context reset drops it and Context close retires it.
        """
        self._need_open()
        owner = self
        while owner.parent is not None:
            owner = owner.parent
        from ..workspace.hidden_store import HiddenWorkspaceStore

        with owner._lock:
            owner._need_open()
            store = HiddenWorkspaceStore(owner, config, entities=entities)
            owner._private_stores.add(store)
            return store

    def _retire_hidden_store(self, store: Any, *, drop: bool = False) -> None:
        """Remove and retire one exact hidden domain-workspace adapter."""
        owner = self
        while owner.parent is not None:
            owner = owner.parent
        with owner._lock:
            for key, candidate in tuple(owner._hidden_stores.items()):
                if candidate is store:
                    owner._hidden_stores.pop(key)
            owner._private_stores.discard(store)
        store.drop() if drop else store.close()

    def _reset_hidden_stores(self, *, drop: bool = False) -> None:
        """Retire this Context's shared hidden domain-workspace adapters of no service kind.

        These are the Capsule registry's. The stores of a service kind, such
        as the LLM cache's, and the stores owned by one object are kept. A drop
        holds the Context lock until it ends and detaches the hidden stores of
        every Context that share the dropped storage (see `_drop_hidden_stores`).
        """
        owner = self
        while owner.parent is not None:
            owner = owner.parent
        if not drop:
            with owner._lock:
                stores = owner._pop_hidden_stores(None)
            for store in stores:
                store.close()
            return
        with owner._lock:
            _drop_hidden_stores(owner._pop_hidden_stores(None))

    def _pop_hidden_stores(self, kind: str | None) -> tuple[Any, ...]:
        """Remove the shared stores of one service kind; the caller holds the Context lock."""
        keys = tuple(key for key, store in self._hidden_stores.items() if store.kind == kind)
        return tuple(self._hidden_stores.pop(key) for key in keys)

    def _pop_all_hidden_stores(self) -> tuple[Any, ...]:
        """Remove every hidden store, shared or owned by one object; the caller holds the Context lock."""
        stores = (*self._hidden_stores.values(), *self._private_stores)
        self._hidden_stores.clear()
        self._private_stores.clear()
        return stores

    def _setup_modules(self, force: bool = False) -> bool:
        """Index packaged module folders at an explicit setup boundary.

        Args:
            force: Whether to rediscover the current package version even when
                its durable module-set marker is current.

        Returns:
            bool: Whether package discovery and validation ran.
        """
        self._need_open()
        with self._lock:
            return self.modules().setup(force=force)

    def child(self, workspace: str) -> "Context":
        """Create a workspace-selection view over this machine Context.

        Args:
            workspace (str): Workspace selected by the child Context.

        Returns:
            Context: Child Context sharing this Context's Backend,
            ModuleService, and ConfigManager, with separate ordinary Registry
            facade caches and workspace identity maps.

        Raises:
            RuntimeError: If this Context is closed.
            ValueError: If `workspace` is not a valid workspace identifier.
        """
        self._need_open()
        with self._lock:
            self._need_open()
            child_spec = BootstrapSpec(self.root, check_workspace_id(workspace), dict(self.bootstrap.backend), dict(self.bootstrap.registry))
            modules = self.modules()
            return type(self)(
                child_spec,
                config=self.config,
                backend=self.backend(),
                modules=modules,
                owns_backend=False,
                parent=self,
            )

    def close(self) -> None:
        """Release resources owned by this Context without deleting data.

        Once close marks the Context, new work is refused, as in 0.1.2.2, and
        admitted reads may finish or fail if they need a retired resource. A
        caller that holds no Context or workspace lock first waits for another
        thread's hold of the Context lock (such as a write), which is reserved
        for it meanwhile so that it gets the lock as soon as its current
        holder lets it go, and for a reset that another thread runs. It then
        marks the Context and waits for the resolver operations of a root
        Context that are already running, which go on with the resolver
        operations they nest. It waits for at most 30 seconds in all
        (``_SHUTDOWN_WAIT``). When that work has not ended, it raises, and the
        Context is open and admits operations again.
        Admitted reads are not waited for. A call during an existing close
        returns immediately, without waiting for it: that close may still give
        up, so a close that returned without an error can leave the Context
        open. A reentrant call on the thread performing a reset also returns
        immediately.

        Args:
            None.

        Returns:
            None: This function does not return a value.

        Raises:
            RuntimeError: If the Context lock is still held by another thread,
                a resolver operation still runs, or another thread's reset
                still runs after that wait. A caller that holds a Context or
                workspace lock is refused at once, as is one that runs a
                database drop and would have to wait, because the work it
                would wait for may wait for it; so is a caller that runs a
                resolver operation of this Context, before any wait, and one
                with a database connection in use when another thread resets
                this Context. A caller already holding a
                Context or workspace lock may close an idle child with no
                owned backend to dispose when the child's shared Context lock
                is immediately available.
        """
        if self._closed or self._closing or self._lifecycle_owner == threading.get_ident():
            return
        nested = holds_any_lock()
        if nested and self.parent is None:
            if self._resetting:
                raise RuntimeError("Context reset is in progress")
            self._raise_nested_shutdown("close")
        if not nested and self.parent is None and self._modules is not None:
            # Refused at once, before any wait: this thread's own resolver operation would never end.
            self._modules._need_retirable()
        deadline = None if nested else _shutdown_deadline()
        try:
            while True:
                with self._shutdown_lock("close", deadline):
                    if self._closed or self._closing:
                        return
                    if self._resetting:
                        from ..database.registry import DatabaseEngineRegistry

                        # The reset's drops may wait for a connection this thread has in use.
                        if deadline is None or DatabaseEngineRegistry._holds_any_connection():
                            raise RuntimeError("Context reset is in progress")
                    elif nested:
                        with self._read_lock:
                            if self._readers:
                                raise RuntimeError("Cannot close Context with an active workspace read from a nested call")
                            if self._owns_backend and self._backend is not None:
                                self._raise_nested_shutdown("close")
                            # This needs no disposal and is valid even when the
                            # child's parent generation has already retired.
                            self._closed = True
                        self._clear_service_caches()
                        return
                    else:
                        modules = self._modules if self.parent is None else None
                        self._mark_shutdown()
                        break
                self._await_reset_by_another_thread(deadline)  # another thread resets the Context
            # New operations are refused from here on, as in 0.1.2.2; the running resolver operations end.
            if modules is not None:
                self._await_resolver_idle(modules, deadline)
            with self._shutdown_lock("close", deadline):
                backend = self._backend
                authority = self.authority_token
                hidden_stores = self._pop_all_hidden_stores() if self.parent is None else ()
                self._clear_service_caches()
        except BaseException:
            for _ in _ATTEMPTS:
                try:
                    self._abandon_shutdown()
                    break
                except BaseException:
                    pass
            raise
        try:
            if self.parent is None and modules is not None:
                modules._retire_authority()
                from ..llm.gateways._protocols import dispose_llm_clients

                dispose_llm_clients(modules)
            if self._owns_backend and backend is not None:
                db = getattr(backend, "db", None)
                dispose = getattr(db, "dispose", None)
                if callable(dispose):
                    dispose()
            if self.parent is None:
                from ..database.registry import DatabaseEngineRegistry

                DatabaseEngineRegistry.dispose_authority(authority)
        finally:
            try:
                for store in hidden_stores:
                    store.close()
            finally:
                self._finish_shutdown()

    def _clear_service_caches(self) -> None:
        """Clear this Context's service references under the Context lock."""
        self._registries.clear()
        self._registry_options.clear()
        self._registry_store = None
        self._modules = None
        for workspace_id in tuple(self._workspace_registry):
            self._workspace_registry.delete(workspace_id)
        for workspace_id in tuple(self._live_workspaces):
            self._live_workspaces.delete(workspace_id)
        self._backend = None


def _drop_hidden_stores(stores: tuple[Any, ...], *, retiring_context: bool = False) -> None:
    """Drop every hidden store that left its Context, then report failures.

    Each store is removed from its Context before this runs, so one failed
    drop must not leave a later store unretired with its rows in place. The
    hidden stores of any Context that share a database or a Backend object
    with these are detached first, and no store with the storage
    configuration of a dropped or detached store opens until the drops end
    (see `heavenbase.workspace.hidden_store`).

    Args:
        stores (tuple[Any, ...]): Hidden stores already removed from their Context.
        retiring_context (bool): Whether the Context blocks writes and will
            retire the stores' generation.

    Returns:
        None: This function does not return a value.

    Raises:
        Exception: The drop failure, unchanged, when one store fails to drop.
        RuntimeError: One combined error, chained from the first failure,
            when several stores fail to drop.
    """
    from ..workspace.hidden_store import _dropping_stores

    errors: list[Exception] = []
    try:
        with _dropping_stores(stores):
            for store in stores:
                try:
                    store.drop(retiring_context=True) if retiring_context else store.drop()
                except Exception as exc:
                    errors.append(exc)
    finally:
        for store in stores:
            store.close()
    _raise_drop_failures(errors)


def _contexts_with_hidden_stores(kind: str | None) -> list[Context]:
    """Return every Context that holds a live shared hidden store of one service kind.

    Args:
        kind (str | None): Service kind of the stores, as `Context._hidden_store` keys them.

    Returns:
        list[Context]: Each such root Context once.
    """
    from ..workspace import hidden_store

    contexts: dict[int, Context] = {}
    for store in hidden_store._live_stores():
        if store.kind == kind and not store.retired:
            contexts[id(store.context)] = store.context
    return list(contexts.values())


def _local_storage_absent(storage: Mapping[str, Any], config: ConfigManager) -> bool:
    """Return whether hidden storage is known not to exist yet, without opening anything.

    It is known absent when every route of ``storage`` names a local file
    location (a ``database`` or ``local_root`` file URI) and none of those
    paths exists. Any other route, such as a server database, is treated as
    possibly present.

    Args:
        storage (Mapping[str, Any]): Hidden workspace storage configuration
            (``workspace``, ``backend`` or ``backends``).
        config (ConfigManager): Configuration whose roots resolve the
            ``file://%/`` and ``file://&/`` aliases.

    Returns:
        bool: ``True`` only when no route can hold persisted rows.
    """
    from ..utils import file_uri_to_path

    routes = storage.get("backend") or storage.get("backends")
    if not isinstance(routes, Mapping) or not routes:
        return False
    specs = list(routes.values()) if all(isinstance(spec, Mapping) for spec in routes.values()) else [routes]
    try:
        resource = config.pj("&", abs=True)
        for spec in specs:
            locations = [spec[key] for key in ("database", "local_root") if isinstance(spec.get(key), str) and spec[key].startswith("file://")]
            if not locations:
                return False
            for location in locations:
                if os.path.exists(file_uri_to_path(location, home=config.root, resource=resource)):
                    return False
    except Exception:
        # A location that cannot be resolved here may still exist: opening the store reports its error.
        return False
    return True


def _root_contexts() -> list[Context]:
    """Return every root Context of this process that is not yet closed or collected.

    The process holds its root Contexts only weakly, so it keeps none of them
    alive. Process-wide maintenance, such as a dropping ``LLMCache.reset()``,
    reaches every open Context through this list, including one that never
    used the service.

    Returns:
        list[Context]: Each live root Context once, in no particular order.
    """
    with _ROOT_CONTEXTS_LOCK:
        contexts = list(_ROOT_CONTEXTS)
    return [context for context in contexts if not context._closed]


def _raise_drop_failures(errors: list[Exception]) -> None:
    """Raise the failures collected while dropping hidden stores.

    Args:
        errors (list[Exception]): Drop failures in the order they occurred.

    Returns:
        None: This function returns only when no drop failed.

    Raises:
        Exception: The only failure, unchanged, when exactly one drop failed.
        RuntimeError: One combined error, chained from the first failure,
            when several drops failed.
    """
    if len(errors) == 1:
        raise errors[0]
    if errors:
        from ..utils import safe_error

        raise RuntimeError("; ".join(safe_error(error) for error in errors)) from errors[0]


_DEFAULT_CONTEXT: Context | None = None
_DEFAULT_CONTEXT_LOCK = threading.Lock()
# Every root Context of the process, held weakly (see `_root_contexts`).
_ROOT_CONTEXTS: weakref.WeakSet[Context] = weakref.WeakSet()
_ROOT_CONTEXTS_LOCK = threading.Lock()


def _reset_root_contexts_lock_after_fork() -> None:
    """Give a forked child its own lock; the parent's may have been held by another thread."""
    global _ROOT_CONTEXTS_LOCK

    _ROOT_CONTEXTS_LOCK = threading.Lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_root_contexts_lock_after_fork)


def _publish_default_context(context: Context, *, adopting: bool = False) -> Context:
    """Validate and publish a root atomically with respect to shutdown."""
    global _DEFAULT_CONTEXT
    with context._lifecycle_state_lock:
        if adopting and context._closed:
            raise RuntimeError(
                "CM_HVNB belongs to a closed Context that was never installed as the process default; "
                "call set_default_context() on a Context built over CM_HVNB before closing it, or pass context="
            )
        context._need_open()
        _DEFAULT_CONTEXT = context
    return context


def default_context() -> Context:
    """Return the process default Context, constructing it on first use.

    The first call loads the machine bootstrap (``~/.heavenbase/bootstrap.yaml``
    or the packaged default) over ``CM_HVNB``. Importing HeavenBase never does
    this; only selecting the default Context does. A root Context the caller
    already built over ``CM_HVNB`` (``Context.load(..., config=CM_HVNB)``) is
    ``CM_HVNB``'s owner and already persists it, so the first call adopts that
    Context as the default if it is open.

    Args:
        None.

    Returns:
        Context: The installed, adopted or lazily constructed process default.

    Raises:
        ValueError: If the machine bootstrap is invalid.
        RuntimeError: If explicit Context mode is on, or the Context that owns
            ``CM_HVNB`` is closed, closing, or resetting.
    """
    context = _DEFAULT_CONTEXT
    if context is None:
        # CM_HVNB's lock comes first everywhere: Context.__init__ checks and
        # binds a manager's owner under that manager's lock, so the owner seen
        # here cannot change before this call adopts it or builds its own; and
        # a first CM_HVNB read can reach this function.
        with CM_HVNB._lock, _DEFAULT_CONTEXT_LOCK:
            context = _DEFAULT_CONTEXT
            if context is None:
                from .._initialization import require_default_context_allowed

                require_default_context_allowed()
                owner = getattr(CM_HVNB, "_context", None)
                if owner is not None:
                    return _publish_default_context(owner, adopting=True)
                context = _publish_default_context(Context.load(config=CM_HVNB))
    return context


def set_default_context(context: Context) -> Context:
    """Install an explicitly constructed root Context as the process default.

    Install an open root before anything selects the default Context. Every later
    omission of ``context=`` resolves to ``context``, and HeavenBase never reads
    the personal bootstrap or opens the package-home control database.
    Calling this function again with the same open root succeeds.
    ``CM_HVNB`` persists through the Context that owns it. With no owner, it
    stays a process-only convenience until a Context binds it.

    Args:
        context (Context): Root Context to install.

    Returns:
        Context: ``context``, for fluent use.

    Assigning ``heavenbase.context.runtime.DEFAULT_CONTEXT = context`` before
    first use installs ``context`` the same way. After first use that
    assignment replaces the default in use, as in 0.1.2.2, so
    ``unittest.mock.patch.object`` and pytest's ``monkeypatch`` can swap the
    default and restore it; this function still refuses to.

    Raises:
        TypeError: If ``context`` is not a Context.
        ValueError: If ``context`` is a child view.
        RuntimeError: If the supplied Context is closed, closing, or resetting,
            a different process default is already in use, or explicit Context
            mode is on.
    """
    if not isinstance(context, Context):
        raise TypeError("Expected a heavenbase.context.Context")
    if context.parent is not None:
        raise ValueError("Only a root Context can be the process default")
    with CM_HVNB._lock, _DEFAULT_CONTEXT_LOCK:
        from .._initialization import require_default_context_allowed

        require_default_context_allowed(installing=True)
        current = _DEFAULT_CONTEXT
        if current is not None and current is not context:
            raise RuntimeError("The process default Context is already in use; install one before HeavenBase selects the default")
        return _publish_default_context(context)


def _default_context_instance() -> Context | None:
    """Return the process default Context only when it already exists."""
    return _DEFAULT_CONTEXT


def _default_context_config_of(resolver: Any) -> ConfigManager | None:
    """Return the default Context's config when ``resolver`` is its live ModuleService."""
    context = _DEFAULT_CONTEXT
    if context is None or context._modules is None or resolver is not context._modules:
        return None
    return context.config


def _config_context(config: Any) -> Context | None:
    """Return the Context that owns ``config``, building the default for ``CM_HVNB``.

    ``CM_HVNB`` is the default Context's ConfigManager, but that Context is
    built on first use, so ``CM_HVNB`` has no owner until then. Code that
    reaches ``CM_HVNB`` as a fallback for an omitted config (a standalone
    Backend, ``DB_ENGINE``, an omitted handler config) therefore selects the
    default Context here, as it did when the default was built on import. In
    explicit Context mode that selection raises the explicit-mode error.
    ``CM_HVNB`` stays unowned only under an installed default Context with its
    own ConfigManager.

    Args:
        config (Any): ConfigManager-compatible authority.

    Returns:
        Context | None: The owning Context, or ``None`` when ``config`` has none.

    Raises:
        RuntimeError: If ``config`` is an unowned ``CM_HVNB`` in explicit
            Context mode.
    """
    # Read the installed default before the owner: the builder binds
    # CM_HVNB's owner before it publishes the default, so an owner read after
    # a published default is never stale.
    installed = _DEFAULT_CONTEXT
    owner = getattr(config, "_context", None)
    if config is not CM_HVNB or installed is not None:
        return owner
    if owner is None:
        default_context()
    else:
        # The default binds CM_HVNB before it finishes loading. Its builder
        # holds CM_HVNB's re-entrant lock throughout, so waiting for that lock
        # never returns another thread's half-built default.
        with CM_HVNB._lock:
            pass
    return getattr(config, "_context", None)


def _default_context_config_registry() -> Registry[str, Any]:
    """Persist ``CM_HVNB`` through the default Context when that Context owns it.

    ``CM_HVNB``'s scope factory, bound by the package root, calls this on the
    first read, which selects the default Context. A default installed with
    its own ConfigManager leaves unowned ``CM_HVNB`` with process-only scopes.
    """
    context = default_context()
    if context.config is CM_HVNB:
        return context._config_registry()
    return Registry[str, Any]("heavenbase-config", normalize=CM_HVNB._scope_key)


# DB-API exception classes: a failure to reach the database, which may pass.
_DATABASE_ERRORS = frozenset({"DBAPIError", "DatabaseError", "OperationalError", "InterfaceError"})


def _bootstrap_fault(error: BaseException) -> bool:
    """Whether building the control backend failed because of what the bootstrap says.

    A backend module or database driver that this interpreter cannot import,
    and a setting the backend rejects, are faults of the bootstrap file and
    last until it changes. Building a backend reports them as
    ``ImportError`` (a module that is not installed, or one whose import
    fails, such as a driver missing its shared library), ``ValueError``,
    ``TypeError``, or a plain ``RuntimeError`` or ``LookupError`` (a dialect
    the bootstrap's database catalog lacks). A failure that may pass is not
    one, nor is any error raised while handling it: a failure to reach the
    database (a lock, a refused connection, an I/O error), and an import that
    failed only because another thread was importing the same modules (a
    ``partially initialized module`` or an import-lock deadlock). Any other
    error, such as a ``KeyError`` or ``IndexError``, is a fault of the code,
    not of the bootstrap, and is raised.
    """
    if not (isinstance(error, (ImportError, ValueError, TypeError)) or type(error) in (RuntimeError, LookupError)):
        return False
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, OSError) or _DATABASE_ERRORS.intersection(cls.__name__ for cls in type(current).__mro__):
            return False
        if type(current).__name__ == "_DeadlockError" or (isinstance(current, ImportError) and "partially initialized module" in str(current)):
            return False
        current = current.__cause__ or current.__context__
    return True


def _build_bootstrap_backend(context: Context) -> None:
    """Build the default Context's control backend, naming the bootstrap when it is at fault.

    ``CM_HVNB``'s tolerant readers (log records, ``hash_id``, the RNG) fall
    back to the packaged values on an invalid bootstrap file, but not on a
    failure that may pass. A fault of the bootstrap is therefore raised as a
    ``ValueError`` that names the bootstrap, with the original error as its
    cause; any other failure is raised unchanged.

    Raises:
        ValueError: If the bootstrap names a backend that cannot be built.
    """
    try:
        backend = context._control_backend()
    except Exception as error:
        if not _bootstrap_fault(error) or not _context_open(context):
            raise
        raise ValueError(f"HeavenBase bootstrap backend cannot be built: {type(error).__name__}: {error}") from error
    if context.config._open_failed:
        from ..database.database import Database

        database = getattr(backend, "db", None)
        if isinstance(database, Database) and database.autocreate and database.spec.dialect_spec.supports("database-local-path"):
            # A cached SQL engine does not repeat its initial best-effort
            # autocreate after the permissions on its database path change.
            from ..database.ddl import create_database

            create_database(database.spec)


def _context_open(context: Context) -> bool:
    try:
        context._need_open()
    except RuntimeError:
        return False
    return True


def context_or_default(context: Context | None = None) -> Context:
    """Return an explicit Context or the process default composition root.

    Args:
        context: Optional caller-owned Context.

    Returns:
        Context: The supplied Context or ``default_context()``.

    Raises:
        TypeError: If ``context`` is not a Context.
    """
    selected = default_context() if context is None else context
    if not isinstance(selected, Context):
        raise TypeError("Expected a heavenbase.context.Context")
    selected._need_open()
    return selected


class _Unassigned:
    """Namespace marker for a ``DEFAULT_CONTEXT`` that no assignment has bound."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "<DEFAULT_CONTEXT not assigned>"


_UNASSIGNED = _Unassigned()
# The process default in use (or None) when DEFAULT_CONTEXT was first assigned; undoing the assignments restores it.
_DEFAULT_BEFORE_ASSIGNMENT: Context | None = None
DEFAULT_CONTEXT: Context
# Reads, assignments and deletion go through ``_RuntimeModule``. The namespace entry is only what a patch helper
# saves and restores: ``mock.patch.object`` reads ``vars(module)`` first, so patching never selects (or builds) the
# default just to remember it.
globals()["DEFAULT_CONTEXT"] = _UNASSIGNED


def _assign_default_context(value: Any) -> None:
    """Rebind the process default for an assignment to, or deletion of, ``DEFAULT_CONTEXT``.

    Before anything selects the default, assigning a Context installs it, as
    :func:`set_default_context` does. Once a default is in use, assigning
    another Context replaces it, as rebinding the module global did in 0.1.2.2:
    later omissions of ``context=`` resolve to the new Context, while whatever
    already holds the previous one (``CM_HVNB`` persistence, a cached
    ``hb.DEFAULT_CONTEXT``) keeps it. Assigning the namespace marker that
    ``mock.patch.object`` saved, or deleting the attribute, undoes every
    assignment and restores the default that was in use before the first one,
    or none. Assigning the Context already in use changes nothing, which is how
    pytest's ``monkeypatch`` restores it.

    Args:
        value (Any): Root Context to assign, or ``_UNASSIGNED`` to undo the assignments.

    Returns:
        None: The process default is rebound in place.

    Raises:
        TypeError: If ``value`` is neither a Context nor the marker.
        ValueError: If ``value`` is a child view.
        RuntimeError: If explicit Context mode is on and ``value`` is not the
            Context already in use, or the Context is unavailable for first
            installation because it is closed, closing, or resetting.
    """
    global _DEFAULT_CONTEXT, _DEFAULT_BEFORE_ASSIGNMENT
    if value is not _UNASSIGNED:
        if not isinstance(value, Context):
            raise TypeError("Expected a heavenbase.context.Context")
        if value.parent is not None:
            raise ValueError("Only a root Context can be the process default")
    namespace = globals()
    with CM_HVNB._lock, _DEFAULT_CONTEXT_LOCK:
        assigned = namespace["DEFAULT_CONTEXT"]
        if value is _UNASSIGNED:
            if assigned is not _UNASSIGNED:
                if _DEFAULT_BEFORE_ASSIGNMENT is None and getattr(CM_HVNB, "_context", None) is None:
                    # Discard process-only scopes opened under the temporary
                    # default. Replace the factory as well so a still-opening
                    # temporary Registry cannot publish after this restore.
                    CM_HVNB._registry = None
                    CM_HVNB._bind_registry_factory(lambda: _default_context_config_registry())
                    CM_HVNB._setup_pending = True
                    CM_HVNB._clear_caches()
                _DEFAULT_CONTEXT, _DEFAULT_BEFORE_ASSIGNMENT = _DEFAULT_BEFORE_ASSIGNMENT, None
        else:
            current = _DEFAULT_CONTEXT
            if value is not current:
                from .._initialization import require_default_context_allowed

                require_default_context_allowed(installing=True, installer="assigning heavenbase.context.runtime.DEFAULT_CONTEXT")
            if current is None:
                _publish_default_context(value)
            else:
                _DEFAULT_CONTEXT = value
            if assigned is _UNASSIGNED:
                _DEFAULT_BEFORE_ASSIGNMENT = current
        namespace["DEFAULT_CONTEXT"] = value


def _read_default_context(module: ModuleType) -> Context:
    return default_context()


def _write_default_context(module: ModuleType, value: Any) -> None:
    _assign_default_context(value)


def _delete_default_context(module: ModuleType) -> None:
    _assign_default_context(_UNASSIGNED)


def __dir__() -> list[str]:
    # As before the namespace entry existed: listing the module (``inspect.getmembers``, ``help``) never selects the default.
    return sorted(name for name in globals() if name != "DEFAULT_CONTEXT")


class _RuntimeModule(ModuleType):
    """Route ``DEFAULT_CONTEXT`` reads to :func:`default_context` and writes to ``_assign_default_context``."""

    DEFAULT_CONTEXT = property(_read_default_context, _write_default_context, _delete_default_context)


sys.modules[__name__].__class__ = _RuntimeModule
bind_late("heavenbase.log.default_context", __name__, "_default_context_instance")

__all__ = ["Context", "context_or_default", "default_context", "set_default_context"]
