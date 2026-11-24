"""Reset package-owned local state coordinated by Context."""

from __future__ import annotations

import time
from uuid import uuid4

__all__ = [
    "reset_default_local_state",
]

from heavenbase.utils import delete_file, dget, file_uri_to_path, normalize_file_uri
from heavenbase.utils.typing import Any, Mapping

from .locks import _ATTEMPTS, holds_any_lock
from .runtime import (
    _READS_POLL,
    Context,
    _default_context_instance,
    _drop_hidden_stores,
    _shutdown_deadline,
    context_or_default,
)


def reset_default_local_state(context: Context | None = None) -> None:
    """Remove package-owned local database files used by default services.

    The Context's hidden stores are dropped. The local database files of the
    LLM cache and Capsule registry of the packaged configuration are dropped
    like any database, so the hidden stores of every Context on them reopen
    on fresh storage at their next use. The Context's own SQLite system
    database file is deleted directly, without the engine registry: another
    Context with an engine on it, such as one over the same root, keeps that
    engine on the unlinked file.

    A caller that holds no Context or workspace lock first waits for another
    thread's hold of the Context lock (such as a write), which is reserved for
    it meanwhile so that it gets the lock as soon as its current holder lets
    it go. It then marks the Context, so that new operations and workspace
    calls fail, as in 0.1.2.2, and waits for the Context's resolver operations
    that are already running, which go on with the resolver operations they
    nest, and, when the reset removes storage, for the workspace reads already
    running. It waits for at most 30 seconds in all. When that work has not
    ended, it raises, and the Context is as it was, so that operations are
    admitted again.

    Args:
        context: Context whose package-owned local resources should be reset.

    Returns:
        None: This function deletes package-owned local files in place.

    Raises:
        RuntimeError: If the Context is closing, closed, or already
            resetting, or after the wait above if the Context lock is still
            held by another thread, a resolver operation still runs, or a
            workspace read still runs. A caller that holds a Context or
            workspace lock is refused at once, as is one that runs a database
            drop and would have to wait, because the work it would wait for
            may wait for it; so is a caller that runs a resolver operation of
            this Context, before any wait.
    """
    from ..database.ddl import drop_database
    from ..database.registry import DatabaseEngineRegistry
    from ..workspace import hidden_store

    selected = context_or_default(context)
    while selected.parent is not None:
        selected = selected.parent
    if holds_any_lock():
        selected._raise_nested_shutdown("reset")
    if selected._modules is not None:
        # Refused at once, before any wait: this thread's own resolver operation would never end.
        selected._modules._need_retirable()
    deadline = _shutdown_deadline()
    reading = False
    try:
        while True:
            with selected._shutdown_lock("reset", deadline):
                selected._need_open()
                previous_authority = selected.authority_token
                if selected._modules is not None:
                    selected._modules._need_retirable()
            # Resolving storage paths can invoke a configured resolver. Do this
            # before retirement and outside the shutdown lock.
            backend, service_files, paths = _default_local_db_paths(selected)
            with selected._shutdown_lock("reset", deadline):
                selected._need_open()
                if previous_authority != selected.authority_token or (backend is not selected._backend and deadline is None):
                    raise RuntimeError("Context authority changed while preparing reset")
                if backend is selected._backend:  # otherwise the backend was built meanwhile: prepare again
                    modules = selected._modules
                    # A destructive reset can unlink a SQLite database or drop a store,
                    # so the workspace reads that may still use them end first.
                    destructive = bool(service_files or paths or selected._hidden_stores or len(selected._private_stores))
                    with selected._read_lock:
                        reading = destructive and bool(selected._readers)
                        if reading and deadline is None:
                            raise RuntimeError("Cannot reset Context storage during active workspace reads")
                        selected._mark_shutdown(reset=True)
                    break
        # New operations are refused from here on; the running resolver operations and, for a destructive reset, the reads
        # admitted before end, without the Context lock, which their nested writes may need (those fail, as the reset is in
        # progress).
        if modules is not None:
            selected._await_resolver_idle(modules, deadline)
        if reading:
            _await_reads(selected, deadline)
        with selected._shutdown_lock("reset", deadline):
            hidden_stores = selected._pop_all_hidden_stores()
            selected._clear_service_caches()
            selected._active_workspace_id = None
    except BaseException:
        for _ in _ATTEMPTS:
            try:
                selected._abandon_shutdown()
                break
            except BaseException:
                pass
        raise
    try:
        try:
            _drop_hidden_stores(hidden_stores, retiring_context=True)
            # The service files are dropped like any database, so the hidden
            # stores of other Contexts on them reopen on fresh storage.
            with hidden_store._dropping_stores(specs=service_files):
                for spec in service_files:
                    with DatabaseEngineRegistry._dropping_database(spec, authority_id=previous_authority, resource_id=None):
                        drop_database(spec)
        finally:
            # Drops use the old database authority. Retire it after the drop
            # phase even when a drop raises.
            selected._authority_token = uuid4().hex
            if modules is not None:
                from ..llm.gateways._protocols import dispose_llm_clients

                modules._retire_authority()
                dispose_llm_clients(modules)
            db = getattr(backend, "db", None)
            if db is not None:
                db.dispose()
            DatabaseEngineRegistry.dispose_authority(previous_authority)
        for path in paths:
            for target in (path, f"{path}-wal", f"{path}-shm"):
                delete_file(target)
    finally:
        try:
            if selected is _default_context_instance():
                from ..entity.base import _set_default_workspace

                _set_default_workspace(None)
            cm = selected.config
            with cm._lock:
                cm._registry = None
                cm._bind_registry_factory(selected._config_registry)
                cm._setup_pending = True
                cm._clear_caches()
        finally:
            selected._finish_shutdown()


def _await_reads(context: Context, deadline: float | None) -> None:
    """Wait until no workspace read of ``context`` runs, for at most until ``deadline``.

    Raises:
        RuntimeError: If a read still runs at the deadline.
    """
    while True:
        with context._read_lock:
            if not context._readers:
                return
        if deadline is None or time.monotonic() >= deadline:
            raise RuntimeError("Cannot reset Context storage during active workspace reads")
        time.sleep(_READS_POLL)


def _default_local_db_paths(context: Context) -> tuple[Any, list[Any], list[str]]:
    """Return the Context's backend, its default services' local databases and its other local files.

    The default services are the Capsule registry and the LLM cache, as the
    packaged configuration places them. Their droppable local databases come
    back as specs, for a drop; every other path is the Context's own system
    database file or a service file that no database drop removes.
    """
    from heavenbase.database import DBEngine
    from heavenbase.database.ddl import _drops_namespace
    from heavenbase.database.dialect.registry import BootstrapDialectCatalog, ResolverDialectCatalog

    cm = context.config
    root = cm.load_default()
    bootstrap_backend = context.bootstrap.backend
    override = bootstrap_backend.get("_override") if isinstance(bootstrap_backend, Mapping) else None
    capsule = dget(root, "heavenbase.capsule.registry.backend") or dget(root, "heavenbase.capsule.registry.backends") or {}
    cache = dget(root, "heavenbase.llm.cache.backend") or dget(root, "heavenbase.llm.cache.backends") or {}
    backends = (*capsule.values(), *cache.values())
    services: dict[str, Any] = {}
    paths: list[str] = []
    if backends:
        if isinstance(override, Mapping):
            database_config = _localize_file_aliases(override, cm)
            dialects = BootstrapDialectCatalog(database_config.get("dialects", {}))
        else:
            database_config = _localize_file_aliases(dget(root, "heavenbase.db", default={}) or {}, cm)
            dialects = ResolverDialectCatalog(context.modules())
        db_engine = DBEngine(config=database_config, manager=None, dialects=dialects)
        for backend in backends:
            cfg = _localize_file_aliases(backend, cm)
            provider = cfg.pop("provider", None) or cfg.pop("type", None)
            if provider is not None:
                cfg["provider"] = provider
            spec = db_engine.resolve(cfg)
            if spec.database and str(spec.database).startswith("file://"):
                path = file_uri_to_path(spec.database, home=cm.root)
                if _drops_namespace(spec):
                    services.setdefault(path, spec)
                else:
                    paths.append(path)
    backend = getattr(context, "_backend", None)
    backend_spec = getattr(getattr(backend, "db", None), "spec", None)
    database = getattr(backend_spec, "database", None)
    if database is None:
        database = context.bootstrap.backend.get("database")
    if database not in (None, "", ":memory:"):
        paths.append(file_uri_to_path(str(database), home=cm.root))
    return backend, [services[path] for path in sorted(services)], sorted(set(paths) - set(services))


def _localize_file_aliases(config: Mapping[str, Any], manager: Any) -> dict[str, Any]:
    """Expand reset-only file aliases against one Context's config roots."""
    localized = dict(config)
    resource = manager.pj("&", abs=True)
    for key in ("database", "local_root"):
        value = localized.get(key)
        if isinstance(value, str) and value.startswith("file://"):
            localized[key] = normalize_file_uri(value, home=manager.root, resource=resource)
    return localized
