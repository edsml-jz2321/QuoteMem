"""Context-owned hidden workspace for private domain state.

A drop of a database removes it for the whole process (on DuckDB, see
``Database.drop_database`` for when a file in use is removed). Every hidden
store in the process, of any Context, whose workspace has a SQL backend on that
database (the same database, or a local file reached through another path)
is then *detached*: it keeps that workspace only as its drop target and opens
a new one at its next use, which recreates its tables on fresh storage. Its
routes on other storage keep their rows, and its own drop still reaches every
route it opened.

Every ``Database.drop_database()`` that removes a database and every
``Database.drop()``, whoever calls them, records the drop and detaches the
stores on that database before the drop and again after it. The engine
registry runs these drops of one database one after another, from any
Context and through any name of a local file while it exists. Once a
``Database.drop_database()`` has detached the stores, the dropping
Context's facades, engine uses and schema indexes on the database wait for
the removal, unless the calling thread holds a Context or workspace lock or
a connection to the database, which then fails at once. A store builds its
workspace under the Context lock, so a build of a store of the dropping
Context meanwhile fails and is tried again after the drop. A store whose
new workspace was built while one of its databases was being dropped does
not attach it: it waits for the drop to end and builds another one, and a
build that fails while a drop runs is tried again once the drop ends. The
next drop of a database that the discarded build used first waits until the
stores it made build again have built, for at most ten seconds in all, so
such a store is postponed again only when its rebuild, behind the other
stores with its storage configuration, takes longer than that. A thread that
has a connection in use, such as an open transaction, on a database that a
running drop removes raises ``RuntimeError`` instead of waiting for that
drop: for its end before a rebuild, for the open lock that a hidden-storage
drop holds, or for a Context lock whose owner runs or waits for the drop.
The drop may wait for that connection: on MySQL, ``DROP DATABASE`` waits for
the metadata lock of the connection's open transaction. A
detach waits for the writes admitted on the detached workspace by other
threads (the LLM cache's hit bookkeeping), except those of a store that
holds the Backend object being dropped, which cannot run before the drop
ends and then fail on the dropped backend.

HeavenBase's own drops of hidden storage (an LLM cache drop, a Capsule
registry drop, a Context reset) also detach, before dropping anything, the
stores that share a database or a Backend object with the stores they drop,
and keep every store with the storage configuration of a dropped or detached
store from opening until they end. A store opens its workspace under the
open lock of its storage configuration, so a call that starts on such a store
meanwhile, in any Context, waits for the drop and then opens fresh storage.
A store also builds its workspace and registers its entities under its
Context lock, which workspace construction takes anyway, so the stores of one
Context open one at a time and a slow open holds up the Context's other
writes until the build ends; stores of other Contexts with other
configurations open independently. A drop that ran during the build is
waited for outside both locks, unless the caller already holds the Context
lock.
"""

from __future__ import annotations

import os
import threading
import weakref
from collections import deque
from collections.abc import Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from time import monotonic

from heavenbase.utils import Any, sha256hash

from ..context import locks as _context_locks
from ..database.registry import DatabaseEngineRegistry, _same_file, _thread_token
from .rows.adapter import WorkspaceEntityRowStore
from .setup.lifecycle import _workspace_config_scope

__all__: list[str] = []

# Every hidden store not yet collected, for the drop listener.
_STORES: "weakref.WeakSet[HiddenWorkspaceStore]" = weakref.WeakSet()
_STORES_LOCK = threading.Lock()


class _NamedLock:
    """Reentrant lock shared under one name for as long as anything refers to it."""

    __slots__ = ("lock", "__weakref__")

    def __init__(self) -> None:
        """Create an unheld lock.

        Returns:
            None: This initializer does not return a value.
        """
        self.lock = threading.RLock()


# One open lock per storage configuration, held by a store of any Context with that configuration while it opens a
# workspace. Take it after any Context lock; drops take several in key order.
_OPEN_LOCKS: "weakref.WeakValueDictionary[str, _NamedLock]" = weakref.WeakValueDictionary()
_LOCKS_LOCK = threading.Lock()

# The latest starts and ends of database drops, numbered in order, and the drops still running, each with an event
# set when it ends, so a store whose workspace was built while one ran waits for it and builds again.
_DROP_LOG = 256
_DROPS: "deque[tuple[int, _DroppedDatabase, threading.Event]]" = deque(maxlen=_DROP_LOG)
_RUNNING_DROPS: "dict[int, tuple[int, _DroppedDatabase, threading.Event]]" = {}
_DROPS_LOCK = threading.Lock()
_drop_count = 0
# Builds of one opening that may fail because a drop ran meanwhile before the failure is raised.
_FAILED_BUILDS = 3
# Stores whose workspace a drop made them build again, by id, with the SQL databases of the discarded build. A drop of
# one of those databases first lets their next build finish (see `_letting_rebuilds_finish`), for at most
# `_REBUILD_GRACE` seconds in all.
_POSTPONED: "dict[int, tuple[HiddenWorkspaceStore, tuple[Any, ...]]]" = {}
_POSTPONED_CHANGED = threading.Condition(threading.Lock())
_REBUILD_GRACE = 10.0
# Open-lock keys that the current thread holds for a drop (see `_holding_open_locks`).
_HELD_OPEN = threading.local()
# Open-lock keys that drops of any thread hold, each with the databases of every drop that holds it (guarded by
# `_POSTPONED_CHANGED`).
_DROPPING_OPEN: "dict[str, list[tuple[_DroppedDatabase, ...]]]" = {}
# Threads that run a drop, hold open locks for one or wait for one to end, each with the databases those drops remove
# (guarded by `_DROPS_LOCK`). A thread with a connection in use on one of them is not made to wait for such a thread.
_BLOCKING: "dict[int, list[tuple[_DroppedDatabase, ...]]]" = {}


def _refuse_if_connected(databases: Sequence[_DroppedDatabase], waited: str) -> None:
    """Raise instead of a wait for ``waited`` when this thread has a connection in use on one of ``databases``.

    A drop of such a database may wait for that connection: on MySQL,
    ``DROP DATABASE`` waits for the metadata lock of the connection's open
    transaction, so the two threads would wait for each other.
    """
    me = _thread_token()  # the engine registry keys checked-out connections by thread token, not by identifier
    if any(DatabaseEngineRegistry._holds_connection(database.key, me) for database in databases):
        raise RuntimeError(
            f"Cannot wait for {waited} while this thread has a connection in use, such as an open transaction, on a "
            "database that a running drop removes: the drop waits for that connection. End the transaction and retry."
        )


@contextmanager
def _blocking_on(databases: Sequence[_DroppedDatabase]) -> Iterator[None]:
    """Record that this thread runs, holds open locks for or waits for drops of ``databases`` while the block runs."""
    me = threading.get_ident()
    entry = tuple(databases)
    with _DROPS_LOCK:
        _BLOCKING.setdefault(me, []).append(entry)
    try:
        yield
    finally:
        with _DROPS_LOCK:
            entries = _BLOCKING.get(me, [])
            if entry in entries:
                entries.remove(entry)
            if not entries:
                _BLOCKING.pop(me, None)


def _guard_polls() -> bool:
    """Return whether a Context-lock wait of this thread needs checking: it has a connection in use."""
    return DatabaseEngineRegistry._holds_any_connection()


def _guard_check(owner: int | None) -> None:
    """Refuse a Context-lock wait when the lock's owner runs, or waits for, a drop that waits for this thread."""
    if owner is None:
        return
    with _DROPS_LOCK:
        groups = list(_BLOCKING.get(owner, ()))
    _refuse_if_connected([database for group in groups for database in group], "the Context lock")


def _named_lock(locks: "weakref.WeakValueDictionary[str, _NamedLock]", name: str) -> _NamedLock:
    """Return the lock that ``locks`` keeps under ``name``, creating it when absent."""
    with _LOCKS_LOCK:
        lock = locks.get(name)
        if lock is None:
            lock = locks[name] = _NamedLock()
        return lock


class HiddenWorkspaceStore:
    """Own one Context-bound hidden workspace for private domain services.

    Application-owned Capsule and Toolkit rows use explicit workspaces.
    Hidden stores remain available for Context-private state such as LLM
    caches whose lifecycle does not belong to an application workspace.

    Args:
        context (Any): Owning machine Context.
        config (Mapping[str, Any]): Hidden workspace construction config with
            optional ``workspace``, ``backend``, and ``backends`` keys.
        kind (str | None): Service that owns a shared store, or ``None``.
        entities (Sequence[type]): Entities registered in every workspace the
            store opens, before any caller can use it.
    """

    def __init__(self, context: Any, config: Mapping[str, Any], *, kind: str | None = None, entities: Sequence[type] = ()) -> None:
        """Construct one Context-bound hidden workspace store.

        Returns:
            None: This initializer does not return a value.
        """
        self.context = context
        self.config = dict(config)
        self.kind = kind
        self._entities = tuple(entities)
        self._workspace: Any = None
        self._detached = False
        self._rows: dict[str, WorkspaceEntityRowStore] = {}
        self._lock = threading.RLock()
        self._retired = False
        # Writes admitted on the open workspace, by thread; a detach waits for them.
        self._admitted: set[tuple[int, threading.Event]] = set()
        self._open_key = self.key(self.config)
        self._open = _named_lock(_OPEN_LOCKS, self._open_key)
        with _STORES_LOCK:
            _STORES.add(self)

    @staticmethod
    def key(config: Mapping[str, Any]) -> str:
        """Return the stable Context cache key for an effective store config.

        Args:
            config (Mapping[str, Any]): Hidden workspace construction config.

        Returns:
            str: Stable content-addressed cache key.
        """
        return sha256hash(dict(config))

    @property
    def retired(self) -> bool:
        """Return whether this owner has left its Context generation."""
        with self._lock:
            return self._retired

    @property
    def workspace(self) -> Any:
        """Return the open hidden workspace, opening a new one when there is none or a drop detached it.

        The store builds its workspace under the Context lock and the open
        lock of its storage configuration. A drop that ran during the build is
        waited for outside both locks, so the Context's other writes go on
        while it runs, unless the caller already holds the Context lock. The
        store then builds again, and the next drop of one of its databases
        first waits, for at most ten seconds, until that build ends. A thread
        with a connection in use on a database that such a drop removes raises
        ``RuntimeError`` instead of waiting for the drop.
        """
        failed = 0
        postponed = False
        try:
            while True:
                with self._lock:
                    if self._retired:
                        raise RuntimeError("Hidden workspace store belongs to a retired Context generation")
                    if self._workspace is not None and not self._detached:
                        return self._workspace
                # Construction takes the Context lock, and a Context-lock holder may
                # reach this property, so the Context lock is taken first.
                with self.context._lock:
                    self.context._need_open()
                    with self._opening():
                        with self._lock:
                            if self._retired:
                                raise RuntimeError("Hidden workspace store belongs to a retired Context generation")
                            if self._workspace is not None and not self._detached:
                                return self._workspace
                        mark = _drop_mark()
                        try:
                            workspace = self._build()
                        except Exception:
                            # A drop that ran meanwhile can make the build fail: build again once it ends.
                            # The detached workspace, when there is one, names the databases a drop must remove.
                            overlapping = _drops_during(mark, self._opened())
                            failed += 1
                            if overlapping is None or failed >= _FAILED_BUILDS:
                                raise
                        else:
                            with self._lock:
                                if self._retired:
                                    raise RuntimeError("Hidden workspace store belongs to a retired Context generation")
                                # A drop of one of its databases while it was built may have removed its tables.
                                overlapping = _drops_during(mark, workspace)
                                if overlapping is None:
                                    # The new workspace opens every route, so it replaces a detached one as the drop target.
                                    self._workspace = workspace
                                    self._detached = False
                                    return workspace
                            if not postponed:
                                # The next drop of its databases lets the next build finish first.
                                postponed = True
                                _postpone(self, workspace)
                running = [database for database, ended in overlapping if not ended.is_set()]
                _refuse_if_connected(running, "a drop of hidden storage")
                with _blocking_on(running):
                    for _, ended in overlapping:
                        ended.wait()
        finally:
            if postponed:
                _end_postponement(self)

    @contextmanager
    def _opening(self) -> Iterator[None]:
        """Hold the open lock of this store's storage configuration.

        A thread that has a connection in use on a database that a drop
        holding the lock removes (see `_dropping_stores`) raises
        ``RuntimeError`` instead of waiting: the drop may wait for that
        connection.
        """
        lock = self._open.lock
        if _guard_polls():
            while not lock.acquire(timeout=0.05):
                with _POSTPONED_CHANGED:
                    groups = list(_DROPPING_OPEN.get(self._open_key, ()))
                _refuse_if_connected([database for group in groups for database in group], "a drop of hidden storage")
        else:
            lock.acquire()
        try:
            yield
        finally:
            lock.release()

    def _build(self) -> Any:
        """Construct a new workspace for this store and register its entities."""
        from .runtime import HeavenBase

        workspace = HeavenBase._construct(
            str(self.config.get("workspace") or "hidden-state"),
            context=self.context,
            backends=self.config.get("backend") or self.config.get("backends"),
            detached=True,
            tracked=False,
            register_workspace=False,
            register_pending=False,
            track_live=False,
            publish=False,
            extension_roots=(),
        )
        for entity in self._entities:
            if entity.schema().entity_id not in workspace.entities:
                workspace.register(entity)
        return workspace

    def _attached(self) -> Any:
        """Return the open workspace, or ``None`` when there is none, a drop detached it, or the store retired."""
        with self._lock:
            return None if self._retired or self._detached else self._workspace

    def _opened(self) -> Any:
        """Return the workspace that holds this store's routes, detached or not, or ``None``."""
        with self._lock:
            return None if self._retired else self._workspace

    @contextmanager
    def _admitted_on(self, workspace: Any) -> Iterator[bool]:
        """Admit one write on ``workspace`` while it is this store's open workspace.

        Args:
            workspace (Any): Workspace the caller read through.

        Yields:
            bool: Whether the write may run; a detach of ``workspace`` waits until it ends.
        """
        entry = (threading.get_ident(), threading.Event())
        with self._lock:
            admitted = not self._retired and not self._detached and self._workspace is workspace
            if admitted:
                self._admitted.add(entry)
        try:
            yield admitted
        finally:
            if admitted:
                with self._lock:
                    self._admitted.discard(entry)
                entry[1].set()

    def _detach(self, workspace: Any, *, wait: bool = True) -> None:
        """Keep ``workspace`` only as the drop target, so the next use opens a new workspace.

        Args:
            workspace (Any): Open workspace to detach, which may be detached
                already; another one is left alone.
            wait (bool): Whether to wait for the writes other threads were
                admitted to run on it, including those admitted before an
                earlier detach that are still running.

        Returns:
            None: This function does not return a value.
        """
        me = threading.get_ident()
        with self._lock:
            if self._workspace is not workspace:
                return
            self._detached = True
            pending = [done for owner, done in self._admitted if owner != me]
        if wait:
            for done in pending:
                done.wait()

    def rows(self, entity: type, *, extension: str) -> WorkspaceEntityRowStore:
        """Return one typed adapter over the shared hidden workspace.

        Args:
            entity (type): Entity class stored through the adapter.
            extension (str): Extension that owns the Entity schema.

        Returns:
            WorkspaceEntityRowStore: Typed row-store adapter bound to the
            shared hidden workspace.

        Raises:
            RuntimeError: If this store belongs to a retired Context
                generation, or the entity schema changed within one store.
        """
        entity_id = entity.schema().entity_id
        with self._lock:
            if self._retired:
                raise RuntimeError("Hidden workspace store belongs to a retired Context generation")
            adapter = self._rows.get(entity_id)
            if adapter is None:
                adapter = WorkspaceEntityRowStore(lambda: self.workspace, entity, extension=extension, owner=self)
                self._rows[entity_id] = adapter
            elif adapter._entity is not entity and adapter._entity.schema().to_dict() != entity.schema().to_dict():
                raise RuntimeError(f"Hidden Entity {entity_id!r} changed schema within one Context store")
            return adapter

    def drop(self, *, retiring_context: bool = False) -> None:
        """Drop the storage of every route this store opened and retire it.

        A workspace detached by a drop of one of its databases is still
        dropped, so the routes on other storage are dropped too.

        Args:
            retiring_context (bool): Whether the Context already blocks writes
                and will retire this workspace's generation.

        Returns:
            None: This function does not return a value.
        """
        with self._lock:
            workspace = self._workspace
            self._workspace = None
            self._detached = False
            self._rows.clear()
            self._retired = True
        if workspace is not None:
            if retiring_context:
                with _workspace_config_scope(workspace.context, workspace.id):
                    workspace._maintenance.drop_backends()
            else:
                workspace.drop()

    def close(self) -> None:
        """Release references without deleting persisted hidden rows.

        Args:
            None.

        Returns:
            None: This function does not return a value.
        """
        with self._lock:
            self._workspace = None
            self._detached = False
            self._rows.clear()
            self._retired = True


def _live_stores() -> tuple[HiddenWorkspaceStore, ...]:
    """Return every hidden store in the process that is not yet collected."""
    with _STORES_LOCK:
        return tuple(_STORES)


class _DroppedDatabase:
    """Recognize the SQL facades on one dropped database, including a local file reached through another path.

    Args:
        spec (Any): Resolved spec of the database being dropped.
        paths (frozenset[str]): Local-file paths already known to name the dropped file.
    """

    def __init__(self, spec: Any, paths: frozenset[str] = frozenset()) -> None:
        """Capture the dropped database's identity.

        Returns:
            None: This initializer does not return a value.
        """
        self.key = spec._database_key
        self.path = DatabaseEngineRegistry._local_path(spec)
        # Local paths compared with the dropped file. Comparisons made while it
        # still exists, before the drop, let a later check recognize an alias.
        self.same: set[str] = set() if self.path is None else {self.path, *paths}
        self.other: set[str] = set()

    def holds(self, db: Any) -> bool:
        """Return whether ``db`` opens the dropped database."""
        spec = db.spec
        if spec.in_memory:
            return False
        if spec._database_key == self.key:
            return True
        path = None if self.path is None else DatabaseEngineRegistry._local_path(spec)
        if path is None or path in self.other:
            return False
        if path not in self.same:
            (self.same if _same_file(path, self.path) else self.other).add(path)
        return path in self.same


def _uses(workspace: Any, databases: Sequence[_DroppedDatabase]) -> bool:
    """Return whether ``workspace`` has a SQL backend on one of ``databases``."""
    from ..database.database import Database

    for _, backend in workspace.backends.items():
        db = getattr(backend, "db", None)
        if isinstance(db, Database) and any(database.holds(db) for database in databases):
            return True
    return False


def _drop_mark() -> int:
    """Return the number of drop starts and ends recorded so far."""
    with _DROPS_LOCK:
        return _drop_count


def _begin_drops(databases: Sequence[_DroppedDatabase]) -> list[tuple[int, _DroppedDatabase, threading.Event]]:
    """Record that ``databases`` are being dropped, for stores that are opening meanwhile.

    Returns:
        list[tuple[int, _DroppedDatabase, threading.Event]]: The running drops, for `_end_drops`.
    """
    global _drop_count
    entries = []
    with _DROPS_LOCK:
        for database in databases:
            _drop_count += 1
            entry = (_drop_count, database, threading.Event())
            _DROPS.append(entry)
            _RUNNING_DROPS[_drop_count] = entry
            entries.append(entry)
    return entries


def _end_drops(entries: Sequence[tuple[int, _DroppedDatabase, threading.Event]]) -> None:
    """Record that the drops `_begin_drops` returned have ended; their events are set by the caller."""
    global _drop_count
    with _DROPS_LOCK:
        for start, database, ended in entries:
            _RUNNING_DROPS.pop(start, None)
            _drop_count += 1
            _DROPS.append((_drop_count, database, ended))


def _drops_during(mark: int, workspace: Any = None) -> list[tuple[_DroppedDatabase, threading.Event]] | None:
    """Return the drops, with their end events, that may have run during a build that started at ``mark``.

    These are the drops that started or ended since ``mark`` and those still
    running, of a database ``workspace`` uses, or of any database when no
    workspace names the store's databases (``workspace`` is ``None``).
    ``None`` means there are none; when drops have left the log since
    ``mark``, the result is never ``None``.
    """
    with _DROPS_LOCK:
        if _drop_count == mark and not _RUNNING_DROPS:
            return None
        complete = _drop_count == mark or (bool(_DROPS) and _DROPS[0][0] <= mark + 1)
        candidates = [(database, ended) for count, database, ended in _DROPS if count > mark]
        candidates += [(database, ended) for _, database, ended in _RUNNING_DROPS.values()]
    found = [(database, ended) for database, ended in candidates if workspace is None or _uses(workspace, (database,))]
    return found if found or not complete else None


def _sharing_stores(
    databases: Sequence[_DroppedDatabase], backends: frozenset[int], exclude: Sequence[HiddenWorkspaceStore] = ()
) -> list[tuple[HiddenWorkspaceStore, Any]]:
    """Return the stores whose workspace holds a Backend object in ``backends`` or uses one of ``databases``.

    A detached workspace counts too: hit writes admitted on it before its
    detach may still be running.
    """
    from ..database.database import Database

    skip = {id(store) for store in exclude}
    found = []
    for store in _live_stores():
        if id(store) in skip:
            continue
        workspace = store._opened()
        if workspace is None:
            continue
        for _, backend in workspace.backends.items():
            db = getattr(backend, "db", None)
            if id(backend) in backends or (isinstance(db, Database) and any(database.holds(db) for database in databases)):
                found.append((store, workspace))
                break
    return found


def _detach_stores_on(databases: Sequence[_DroppedDatabase], dropping: Any = None) -> None:
    """Detach every hidden store whose workspace uses one of ``databases``, after its admitted writes end.

    The writes that other threads were admitted to run on such a workspace
    are waited for, except on a store whose workspace holds the Backend object
    that runs the drop (its ``db`` is the ``Database`` facade ``dropping``):
    the drop holds that Backend's lock, so those writes cannot run before it
    ends, and then they fail on the dropped backend. None of the writes
    waited for waits for the drop in turn: no write takes a drop lock, and
    those of the Context whose ``Database.drop_database()`` runs find its
    scope open, since the drop closes it only after this first pass and
    reopens it before the second.
    """
    for store, workspace in _sharing_stores(databases, frozenset()):
        holds_dropping = dropping is not None and any(getattr(backend, "db", None) is dropping for _, backend in workspace.backends.items())
        store._detach(workspace, wait=not holds_dropping)


def _postpone(store: HiddenWorkspaceStore, workspace: Any) -> None:
    """Record that a drop made ``store`` discard ``workspace`` and build again."""
    from ..database.database import Database

    databases = []
    for _, backend in workspace.backends.items():
        db = getattr(backend, "db", None)
        if isinstance(db, Database) and not db.spec.in_memory:
            databases.append(db)
    with _POSTPONED_CHANGED:
        _POSTPONED[id(store)] = (store, tuple(databases))


def _end_postponement(store: HiddenWorkspaceStore) -> None:
    """Record that ``store`` has stopped building again, opened or not, and wake the drops waiting for it."""
    with _POSTPONED_CHANGED:
        _POSTPONED.pop(id(store), None)
        _POSTPONED_CHANGED.notify_all()


def _letting_rebuilds_finish(database: _DroppedDatabase) -> None:
    """Wait, for at most `_REBUILD_GRACE` seconds, until the stores that a drop postponed on ``database`` have built again.

    A store that the caller's own locks would keep from building is not
    waited for: a store of a Context whose lock the caller holds, or one whose
    open lock it holds for a drop. A store's build waits for no drop lock, so
    the wait ends as soon as the builds do, unless they take longer than the
    grace in all (stores with one storage configuration build one at a time,
    across Contexts) or another thread holds a lock a build needs while it
    waits for this drop. A store whose rebuild has not ended by then can be
    postponed again.
    """
    deadline = monotonic() + _REBUILD_GRACE
    held = getattr(_HELD_OPEN, "keys", ())
    with _POSTPONED_CHANGED:
        while True:
            pending = [
                store
                for store, databases in _POSTPONED.values()
                if not store.context._lock.owned() and store._open_key not in held and any(database.holds(db) for db in databases)
            ]
            remaining = deadline - monotonic()
            if not pending or remaining <= 0:
                return
            _POSTPONED_CHANGED.wait(remaining)


@contextmanager
def _holding_open_locks(stores: Sequence[HiddenWorkspaceStore], databases: Sequence[_DroppedDatabase] = ()) -> Iterator[None]:
    """Hold the open locks of the storage configurations of ``stores``, taken in key order, for a drop of ``databases``."""
    locks = sorted({store._open_key: store._open for store in stores}.items(), key=lambda item: item[0])
    keys = [key for key, _ in locks]
    dropped = tuple(databases)
    before = getattr(_HELD_OPEN, "keys", frozenset())
    with _POSTPONED_CHANGED:
        for key in keys:
            _DROPPING_OPEN.setdefault(key, []).append(dropped)
    try:
        with _blocking_on(dropped), ExitStack() as held:
            for _, lock in locks:
                held.enter_context(lock.lock)
            _HELD_OPEN.keys = before | set(keys)
            try:
                yield
            finally:
                _HELD_OPEN.keys = before
    finally:
        with _POSTPONED_CHANGED:
            for key in keys:
                entries = _DROPPING_OPEN.get(key, [])
                if dropped in entries:
                    entries.remove(dropped)
                if not entries:
                    _DROPPING_OPEN.pop(key, None)


@contextmanager
def _dropping_stores(stores: Sequence[HiddenWorkspaceStore] = (), specs: Sequence[Any] = ()) -> Iterator[None]:
    """Keep the hidden stores that share storage with ``stores`` or sit on ``specs`` off it while the block drops it.

    The stores of any Context whose workspace shares a database or a Backend
    object with ``stores``, or uses a database ``specs`` name, are detached,
    and the writes other threads were admitted to run on them are waited for.
    No store with the storage configuration of one of ``stores`` or of a
    detached store opens a workspace until the block ends: its open lock is
    held, so a call that starts on one meanwhile waits for the drop and then
    opens fresh storage, and one that was opening is waited for first. A store
    with another configuration that opens on the same storage meanwhile is
    handled by the drop listener, as for any drop.

    Args:
        stores (Sequence[HiddenWorkspaceStore]): Stores about to be dropped.
        specs (Sequence[Any]): Specs of databases about to be dropped.

    Yields:
        None: The caller drops the storage.
    """
    from ..database.database import Database

    databases: list[_DroppedDatabase] = [_DroppedDatabase(spec) for spec in specs if not spec.in_memory]
    backends: set[int] = set()
    for store in stores:
        workspace = store._opened()
        if workspace is None:
            continue
        for _, backend in workspace.backends.items():
            backends.add(id(backend))
            db = getattr(backend, "db", None)
            if isinstance(db, Database) and not db.spec.in_memory:
                databases.append(_DroppedDatabase(db.spec))
    peers = [store for store, _ in _sharing_stores(databases, frozenset(backends), exclude=stores)]
    with _holding_open_locks((*stores, *peers), databases):
        # Stores that attached before their open locks were taken are found again.
        for store, workspace in _sharing_stores(databases, frozenset(backends), exclude=stores):
            store._detach(workspace)
        yield


@contextmanager
def _detaching_stores_on(spec: Any, dropping: Any = None, paths: frozenset[str] = frozenset()) -> Iterator[None]:
    """Record a database drop and detach the hidden stores on it, before the drop and again after it.

    A store that is opening meanwhile, which only a drop outside
    `_dropping_stores` allows on its storage configuration, finds the
    recorded drop, waits for it to end and builds its workspace again; the
    second pass detaches a store that attached while the drop ran all the same.
    The engine registry runs the listener inside the drop's locks, so drops
    of one database, from any Context, reach it one after another.

    Args:
        spec (Any): Resolved spec of the database being dropped.
        dropping (Any): ``Database`` facade that drops it, or ``None``.
        paths (frozenset[str]): Local-file paths known to name the dropped file.

    Yields:
        None: The caller drops the database.
    """
    if spec.in_memory:
        # An in-memory database belongs to one facade, so no other store holds it.
        yield
        return
    database = _DroppedDatabase(spec, paths)
    with _blocking_on((database,)):
        _letting_rebuilds_finish(database)
        running = _begin_drops((database,))
        try:
            _detach_stores_on((database,), dropping)
            yield
        finally:
            try:
                _end_drops(running)
                _detach_stores_on((database,), dropping)
            finally:
                for _, _, ended in running:
                    ended.set()


DatabaseEngineRegistry._drop_listeners.append(_detaching_stores_on)
# A thread with a connection in use that waits for a Context lock whose owner runs, or waits for, a drop of that
# connection's database raises instead of waiting (see `_guard_check`).
_context_locks._WAIT_GUARD = (_guard_polls, _guard_check)


def _reset_store_locks_after_fork() -> None:
    """Give the child its own store and coordination locks; the parent's drops do not run in it."""
    global _STORES_LOCK, _LOCKS_LOCK, _DROPS_LOCK, _POSTPONED_CHANGED
    _STORES_LOCK = threading.Lock()
    _LOCKS_LOCK = threading.Lock()
    _DROPS_LOCK = threading.Lock()
    _POSTPONED_CHANGED = threading.Condition(threading.Lock())
    _DROPS.clear()
    _RUNNING_DROPS.clear()
    _POSTPONED.clear()
    _DROPPING_OPEN.clear()
    _BLOCKING.clear()
    for lock in tuple(_OPEN_LOCKS.values()):
        lock.lock = threading.RLock()
    for store in tuple(_STORES):
        store._lock = threading.RLock()
        store._admitted = set()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_store_locks_after_fork)
