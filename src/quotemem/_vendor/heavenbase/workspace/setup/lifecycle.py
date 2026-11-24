"""Context-owned workspace lifecycle and process lookup."""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from functools import wraps

from heavenbase.utils import Any, Callable, Iterator, Mapping, Optional

from ...context import Context, context_or_default
from ...context.locks import _WorkspaceLock, holds_any_lock
from ...context.runtime import _default_context_instance
from ...entity.base import _set_default_workspace, _set_workspace_binder
from ...utils.ids import check_workspace_id


def _new_workspace_lock(context: Context, ws_id: str | None) -> _WorkspaceLock:
    """Return the workspace lock shared by every facade of ``ws_id`` in one Context tree.

    Args:
        context (Context): Context that owns the workspace.
        ws_id (str | None): Workspace id, or ``None`` for a detached facade,
            which owns its lock alone.

    Returns:
        _WorkspaceLock: Reader-writer lock bound to the Context lock.
    """
    return context._workspace_locks.get(None if ws_id is None else check_workspace_id(ws_id))


def construction_write_scope(context: Context, ws_id: Any, *, detached: bool) -> AbstractContextManager[Any]:
    """Return a block that takes a shared workspace id's write lock before the Context lock for construction.

    An outer thread (one holding no HeavenBase lock) queues behind that id's
    other writers and drains its readers before it takes the Context lock, as
    any outer write does, so construction does not hold the Context lock while
    readers of the id finish. Detached facades own a private lock that no one
    else can read, and nested threads already hold the Context lock, so both
    take no lock here.

    Args:
        context (Context): Context that owns the workspace.
        ws_id (Any): Requested workspace id; invalid ids take no lock here and
            fail later in construction.
        detached (bool): Whether the facade is detached.

    Returns:
        AbstractContextManager[Any]: Context manager; the caller constructs
        while the lock is held.
    """
    if detached or holds_any_lock():
        return nullcontext()
    try:
        clean_id = check_workspace_id(ws_id)
    except Exception:
        return nullcontext()
    return context._workspace_locks.get(clean_id)


def _default_workspace_name(context: Context | None = None) -> str:
    """Return one Context's effective active-or-default workspace id."""
    selected = context_or_default(context)
    return selected.effective_workspace_id


def _workspace_config_scope(context: Context, ws_id: str) -> Any:
    """Return the exact config scope derived from one workspace id."""
    manager = context.config
    return manager.scoped(f"{manager.base_scope}.{check_workspace_id(ws_id)}")


def _refresh_workspace_state(context: Context) -> None:
    """Refresh durable config and workspace definitions before reconstruction."""
    with context.config.scoped(context.config.base_scope):
        context.config.refresh()
        from ..registry import _registry

        _registry(context).refresh()


def workspace_locked(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Run a mutating workspace method under the exclusive side of its lock.

    A writer that holds no other HeavenBase lock first waits, in arrival
    order behind other such writers of the same workspace, for that
    workspace's readers to drain, without holding the Context lock. Every
    writer then takes the Context lock and the exclusive side, so one writer
    per Context tree proceeds at a time. Context close and reset wait,
    for a bounded time, until no other thread holds the Context lock.
    Nested reads and writes by the writing thread are allowed, including
    writes of other workspaces from a compute hook; readers that wait for
    this writer are not waited for (see ``heavenbase.context.locks``).

    Args:
        fn (Callable[..., Any]): Workspace method to wrap.

    Returns:
        Callable[..., Any]: Wrapped method that holds the Context lock and the
        workspace write lock while calling ``fn``.
    """

    @wraps(fn)
    def _wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        self._need_live()
        with self._lock:
            self._need_live()
            with _workspace_config_scope(self.context, self.id):
                return fn(self, *args, **kwargs)

    return _wrapped


def workspace_read_locked(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Run a read-only workspace method under the shared side of its lock.

    Readers of one workspace run concurrently with each other and with any
    workspace of the same Context, and never take the Context lock. A reader
    may call a mutating method of the same workspace: that write takes the
    Context lock and waits for the workspace's other running readers, as a
    nested write does. An admitted read can finish or fail if its resources retire.
    A reset that would drop hidden storage or delete local database files
    waits, for a bounded time, for active workspace reads to end.
    New reads fail fast once the Context marks itself as closing or resetting.

    Args:
        fn (Callable[..., Any]): Read-only workspace method to wrap.

    Returns:
        Callable[..., Any]: Wrapped method that holds the workspace read lock
        while calling ``fn``.
    """

    @wraps(fn)
    def _wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        with workspace_read_scope(self):
            return fn(self, *args, **kwargs)

    return _wrapped


def workspace_read_scope(workspace: Any) -> AbstractContextManager[None]:
    """Return a block that holds one workspace's shared lock, liveness check, and config scope.

    ``workspace_read_locked`` applies this scope to a whole method call. A
    public iterator that must not keep the lock across ``yield`` applies it
    to each storage page instead, so writers are admitted between pages.

    Args:
        workspace (Any): Live ``HeavenBase`` facade to read under.

    Returns:
        AbstractContextManager[None]: Context manager; the caller reads
        storage while the scope is held.

    Raises:
        RuntimeError: If the workspace's Context authority has retired.
    """
    # A queue head may be waiting for a reader that is waiting for this call.
    workspace._need_live()
    return workspace.context._track_read(workspace._need_live, workspace._lock, _workspace_config_scope(workspace.context, workspace.id))


def workspace_read_pages(workspace: Any, pages: Iterator[list[Any]]) -> Iterator[Any]:
    """Yield the items of each page, fetching every page under the read scope.

    The lock and liveness check are taken to pull one page from ``pages`` and
    released before that page's items are yielded, so a consumer may mutate
    the workspace between pages and the sequence is not a snapshot. Closing
    the returned iterator closes ``pages``.

    Args:
        workspace (Any): Live ``HeavenBase`` facade to read under.
        pages (Iterator[list[Any]]): Page iterator that reads storage on each step.

    Yields:
        Any: Items of each page in page order.

    Raises:
        RuntimeError: If the workspace's Context authority retires before a page.
    """
    try:
        while True:
            with workspace_read_scope(workspace):
                page = next(pages, None)
            if page is None:
                return
            yield from page
    finally:
        close = getattr(pages, "close", None)
        if callable(close):
            close()


def _registered_workspace(context: Context, ws_id: str) -> Optional[Any]:
    """Return a Context-owned registered workspace without creating one."""
    return context._workspace_registry.get(check_workspace_id(ws_id))


def prepare_workspace(
    context: Context,
    ws_id: str,
    *,
    backends: Optional[Mapping[str, Any]],
    preset: Optional[str],
    asserted_construction: Any = None,
) -> tuple[str, Optional[Mapping[str, Any]], Optional[str], Any]:
    """Resolve durable ensure inputs before workspace initialization.

    Args:
        context: Context that owns workspace lifecycle and persistence.
        ws_id: Workspace identifier to ensure.
        backends: Explicit replayable backend configuration or live Backend
            instances. ``None`` makes no backend assertion.
        preset: Explicit preset assertion. ``None`` makes no preset assertion.
        asserted_construction: Optional canonical construction assertion used
            when effective replay inputs intentionally differ from durable
            identity, as with runtime-manifest connection replacements.

    Returns:
        tuple: Validated id, effective backends, effective preset, and an
        existing ``WorkspaceSpec`` or ``None``.

    Raises:
        FileExistsError: If explicit construction input conflicts with the
            registered construction contract.
        RuntimeError: If a runtime-only workspace has no live Backend
            instances available for reconstruction.
    """
    clean_id = check_workspace_id(ws_id)
    with context._lock:
        context._need_open()
        _refresh_workspace_state(context)
        from ..registry import WorkspaceSpec, _registry
        from .backends import WorkspaceConstruction

        registered = _registry(context).get(clean_id)
        if registered is None:
            return clean_id, backends, preset, None
        spec = WorkspaceSpec.load(clean_id, context=context)
        if asserted_construction is not None:
            requested = (
                asserted_construction if isinstance(asserted_construction, WorkspaceConstruction) else WorkspaceConstruction.from_dict(asserted_construction)
            )
            _require_compatible_construction(clean_id, spec.construction, requested)
            if backends is not None or preset is not None:
                return clean_id, backends, preset, spec
            if not spec.runtime_only:
                config = spec.config
                return clean_id, config.get("backends"), config.get("preset"), spec
            live = _registered_workspace(context, clean_id)
            if live is None:
                raise RuntimeError(f"Workspace {clean_id!r} uses runtime-only backends; reconstruct it with " "HeavenBase(id, backends=..., context=...)")
            runtime_backends = {name: backend for name, backend in live.backends.items()}
            actual = WorkspaceConstruction.from_runtime_backends(
                runtime_backends,
                resolver=live.module_resolver,
            )
            if actual != spec.construction:
                raise RuntimeError(f"Live backend identifiers disagree with the registered construction for workspace {clean_id!r}")
            return clean_id, runtime_backends, None, spec
        explicit = backends is not None or preset is not None
        if explicit:
            with _workspace_config_scope(context, clean_id):
                if backends is not None:
                    requested = WorkspaceConstruction.from_inputs(
                        backends,
                        preset,
                        resolver=context.modules(),
                    )
                else:
                    from ..presets import resolve_workspace_preset

                    preset_spec = resolve_workspace_preset(preset, manager=context.config)
                    requested = WorkspaceConstruction.from_preset(preset_spec.get("preset"))
            _require_compatible_construction(clean_id, spec.construction, requested)
            return clean_id, backends, preset, spec
        if not spec.runtime_only:
            config = spec.config
            return clean_id, config.get("backends"), config.get("preset"), spec
        live = _registered_workspace(context, clean_id)
        if live is None:
            raise RuntimeError(f"Workspace {clean_id!r} uses runtime-only backends; reconstruct it with " "HeavenBase(id, backends=..., context=...)")
        runtime_backends = {name: backend for name, backend in live.backends.items()}
        actual = WorkspaceConstruction.from_inputs(
            runtime_backends,
            None,
            resolver=live.module_resolver,
        )
        if actual != spec.construction:
            raise RuntimeError(f"Live backend identifiers disagree with the registered construction for workspace {clean_id!r}")
        return clean_id, runtime_backends, None, spec


def _require_compatible_construction(
    ws_id: str,
    existing: Any,
    requested: Any,
) -> None:
    """Raise when two canonical workspace construction contracts differ."""
    if requested == existing:
        return
    comparison = f"{existing.kind} != {requested.kind}" if existing.kind != requested.kind else f"different {requested.kind} definitions"
    raise FileExistsError(f"Workspace {ws_id!r} already exists with a different construction contract " f"({comparison})")


def load_workspace(workspace_cls: type, ws_id: Optional[str], *, context: Context) -> Any:
    """Load or reuse a workspace through one explicit Context authority.

    Args:
        workspace_cls (type): Workspace facade class to construct, typically
            ``HeavenBase``.
        ws_id (str | None): Workspace id to load. ``None`` uses the active
            registered workspace or configured default id.
        context (Context): Machine composition root that owns durable and live
            workspace lookup.

    Returns:
        Any: Live workspace facade for ``ws_id``.

    Raises:
        KeyError: If no matching persistent workspace registration exists.
        RuntimeError: If runtime-only construction requires a live facade with
            a different construction contract.
    """
    with context._lock:
        context._need_open()
        _refresh_workspace_state(context)
        from heavenbase.workspace import WorkspaceSpec

        spec = WorkspaceSpec.load(check_workspace_id(ws_id) if ws_id is not None else None, context=context)
        workspace = _registered_workspace(context, spec.id)
        if workspace is not None:
            workspace._need_live()
            if workspace._construction == spec.construction:
                return workspace
            if spec.runtime_only:
                raise RuntimeError(
                    f"Workspace {spec.id!r} uses runtime-only backends, but its live facade has a different "
                    "construction contract; reconstruct it with HeavenBase(id, backends=..., context=...)"
                )
        if spec.runtime_only:
            raise RuntimeError(f"Workspace {spec.id!r} uses runtime-only backends; reconstruct it with " "HeavenBase(id, backends=..., context=...)")
        if spec.manifest:
            from ..manifest import open_workspace_manifest

            return open_workspace_manifest(
                workspace_cls,
                spec.manifest,
                context=context,
                register_workspace=False,
            )
        return workspace_cls._construct(
            spec.id,
            context=context,
            register_workspace=spec.extension_roots is None,
            extension_roots=spec.extension_roots,
            **spec.config,
        )


def _bind_entity_to_registered_workspace(entity: type, ws_id: str, context: Context) -> None:
    """Bind an Entity declaration only to its explicitly selected Context."""
    workspace = context._workspace_registry.get(check_workspace_id(ws_id))
    if workspace is not None:
        workspace.register(entity)


_set_workspace_binder(_bind_entity_to_registered_workspace)


def _set_entity_default(context: Context, workspace: Any) -> None:
    """Publish the effective legacy Entity workspace for the default Context."""
    if context is _default_context_instance():
        _set_default_workspace(workspace)


def _sync_entity_default(context: Context) -> None:
    """Project one Context's effective live workspace to the legacy Entity hook."""
    if context is not _default_context_instance():
        return
    with context._workspace_lock:
        workspace = context._workspace_registry.get(context.effective_workspace_id)
    _set_entity_default(context, workspace)


def _track_workspace(workspace: Any) -> tuple[Optional[Any]]:
    """Publish a workspace and return its prior live-cache owner."""
    context = workspace.context
    with context._workspace_lock:
        previous = (context._workspace_registry.get(workspace.id),)
        try:
            context._workspace_registry.update(workspace.id, workspace)
            _sync_entity_default(context)
        except BaseException:
            _rollback_workspace_publication(workspace, previous)
            raise
        return previous


def _rollback_workspace_publication(workspace: Any, previous: tuple[Optional[Any]]) -> None:
    """Restore Context lookup after this facade fails during publication."""
    context = workspace.context
    (previous_owner,) = previous
    with context._workspace_lock:
        if context._workspace_registry.get(workspace.id) is not workspace:
            return
        if previous_owner is None:
            context._workspace_registry.delete(workspace.id)
        else:
            context._workspace_registry.update(workspace.id, previous_owner)
        _sync_entity_default(context)


def _release_workspace(workspace: Any) -> None:
    """Remove a workspace from its owning Context if this object owns it."""
    context = workspace.context
    with context._workspace_lock:
        owned = context._workspace_registry.get(workspace.id) is workspace
        if owned:
            context._workspace_registry.delete(workspace.id)
            _sync_entity_default(context)


def _track_live_workspace(workspace: Any, *, publication: Optional[list[Optional[Any]]] = None) -> None:
    """Track a workspace for explicit live teardown in its owning Context."""
    context = workspace.context

    def replace(previous_owner: Optional[Any]) -> Any:
        if publication is not None:
            publication[:] = [previous_owner]
        return workspace

    with context._workspace_lock:
        context._live_workspaces.update(workspace.id, derive=replace)


def _untrack_live_workspace(workspace: Any) -> None:
    """Stop tracking a workspace for explicit live teardown."""
    context = workspace.context
    with context._workspace_lock:
        if context._live_workspaces.get(workspace.id) is workspace:
            context._live_workspaces.delete(workspace.id)


def _rollback_live_workspace_publication(workspace: Any, previous_owner: Optional[Any]) -> None:
    """Restore live-drop lookup when this facade still owns its failed write."""
    context = workspace.context
    with context._workspace_lock:
        if context._live_workspaces.get(workspace.id) is not workspace:
            return
        if previous_owner is None:
            context._live_workspaces.delete(workspace.id)
        else:
            context._live_workspaces.update(workspace.id, previous_owner)


def _publish_workspace(
    workspace: Any,
    *,
    tracked: bool,
    track_live: bool,
    finalize: Optional[Callable[[], None]] = None,
) -> None:
    """Publish a fully initialized facade and roll back partial publication.

    ``finalize`` runs last inside the rollback envelope. It is the transaction
    boundary for durable metadata whose failure must not leave a new live
    facade published.
    """
    new_bindings = []
    publication = None
    previous_tracked = bool(getattr(workspace, "_tracked", False))
    live_publication: list[Optional[Any]] = []
    workspace._tracked = bool(tracked)
    try:
        if workspace._tracked:
            system_ids = workspace._catalog.system_entity_ids()
            binding = (workspace.context.identity_token, workspace.id)
            for entity_id, entity in workspace.entities.items():
                if entity_id in system_ids:
                    continue
                with getattr(entity, "_hb_workspace_lock", workspace._lock):
                    if binding not in entity._hb_workspace_bindings:
                        entity._hb_workspace_bindings.add(binding)
                        new_bindings.append(entity)
            publication = _track_workspace(workspace)
        if track_live:
            _track_live_workspace(workspace, publication=live_publication)
        if finalize is not None:
            finalize()
    except BaseException:
        if live_publication:
            _rollback_live_workspace_publication(workspace, live_publication[0])
        if publication is not None:
            _rollback_workspace_publication(workspace, publication)
        for entity in new_bindings:
            with getattr(entity, "_hb_workspace_lock", workspace._lock):
                entity._hb_workspace_bindings.discard((workspace.context.identity_token, workspace.id))
        workspace._tracked = previous_tracked
        raise


def _drop_workspace_by_id(ws_id: str, *, context: Context | None = None) -> bool:
    """Drop a live workspace by id from one Context when it exists."""
    selected = context_or_default(context)
    workspace = selected._live_workspaces.get(check_workspace_id(ws_id))
    if workspace is None:
        return False
    workspace._drop_live()
    return True


class _WorkspaceDrop:
    """Expose ``HeavenBase.drop(ws_id)`` and ``ws.drop()`` on one name."""

    def __get__(self, obj: Optional[Any], owner: type) -> Callable[..., Any]:
        if obj is None:

            def drop(ws_id: str, *, context: Context | None = None) -> bool:
                """Drop a live workspace from the selected Context by id."""
                return _drop_workspace_by_id(ws_id, context=context)

            return drop

        def drop() -> None:
            """Drop this workspace instance and its Backend data."""
            obj._drop_live()

        return drop
