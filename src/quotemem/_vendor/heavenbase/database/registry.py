"""Thread-safe SQLAlchemy engine and schema-index cache."""

from __future__ import annotations

__all__ = ["DatabaseEngineRegistry", "DropRefusedError"]

import itertools
import os
import threading
import time
from collections import OrderedDict, deque
from contextlib import AbstractContextManager, ExitStack, contextmanager
from os.path import isabs, normcase, realpath, samefile
from weakref import WeakKeyDictionary, WeakSet, WeakValueDictionary, ref

from sqlalchemy import event

from ..context.locks import holds_any_lock
from ..utils.typing import Any, Callable, Iterator, Mapping, Sequence, deepcopy
from . import engine as db_engine
from .schema_index import SchemaIndex, normalize_schema_index
from .config import DB_ENGINE, DBSpec
from .response import abandoned_connections, settle_abandoned
from ..utils.files import file_uri_to_path
from ..utils.log import get_logger

logger = get_logger(__name__)

# Seconds after which a wait for a notification checks again, as an interrupted call can end without notifying.
_WAIT_STEP = 1.0
# States of a recorded DuckDB pool entry besides the number of the checkout it is lent by (see _track).
_IDLE = 0
_CONNECTING = -1
_CHECKOUTS = itertools.count(1)
_TOKENS = itertools.count(1)
_THREAD = threading.local()


def _thread_token() -> int:
    """Return a number that identifies the calling thread and is never given to another thread, unlike its identifier."""
    token = getattr(_THREAD, "token", None)
    if token is None:
        token = _THREAD.token = next(_TOKENS)
    return token


def _file_identity(path: str) -> str | None:
    """Return the drop-lock name of the file at ``path`` (see DatabaseEngineRegistry._drop_names), or ``None``."""
    try:
        found = os.stat(path)
    except OSError:
        return None
    return f"file:{found.st_dev}:{found.st_ino}"


class _OpenCloseGate:
    """Run the opens and closes of connections to one DuckDB file one at a time, without ever waiting on itself.

    DuckDB can fail an open with "Unique file handle conflict" when another thread closes the last connection of that
    file's database meanwhile (see DatabaseEngineRegistry._track). A garbage collection can run inside an open or a
    close and free a pooled connection whose return closes it, on the thread that holds the gate. Such a close is not
    run there, in the middle of DuckDB's own open or close: it is deferred and runs right after the outermost hold
    ends. The lock is reentrant, so a close that a collection starts after that point, while the thread still holds
    the lock, runs at once instead of waiting on the thread's own hold.
    """

    __slots__ = ("_lock", "_local", "_deferred", "__weakref__")

    def __init__(self) -> None:
        """Create an open gate.

        Returns:
            None: This initializer does not return a value.
        """
        self._reset()
        _GATES.add(self)

    def _reset(self) -> None:
        """Start without a holder, as a forked child must: the parent's other threads do not run in it."""
        self._lock = threading.RLock()
        self._local = threading.local()
        self._deferred: "deque[Callable[[], None]]" = deque()

    @contextmanager
    def held(self) -> Iterator[None]:
        """Hold the gate for an open or a close; the outermost hold then runs the closes deferred meanwhile.

        A deferred close that fails is logged, as SQLAlchemy logs a failing close: it neither stops the other deferred
        closes nor replaces an error of the hold.
        """
        # A fork inside the hold replaces both in the child, so the hold ends on the ones it started with.
        local = self._local
        deferred: "deque[Callable[[], None]]" = deque()
        try:
            # Only the lock's own C code runs between taking it and the block, so an interrupt leaves it free.
            with self._lock:
                depth = getattr(local, "depth", 0)
                try:
                    local.depth = depth + 1
                    yield
                finally:
                    local.depth = depth
                    if not depth:
                        deferred, self._deferred = self._deferred, deferred
        finally:
            self._run(deferred)

    def _run(self, deferred: "deque[Callable[[], None]]") -> None:
        """Run the closes deferred by a hold, also the ones after a close that is interrupted."""
        try:
            while deferred:
                close = deferred.popleft()
                try:
                    self.close(close)
                except Exception:
                    logger.error("Exception closing a DuckDB connection", exc_info=True)
        finally:
            if deferred:
                self._run(deferred)

    def close(self, close: Callable[[], None]) -> None:
        """Run ``close`` under the gate, or defer it while the calling thread is inside an open or a close."""
        if getattr(self._local, "depth", 0):
            self._deferred.append(close)
            return
        with self.held():
            close()


# Every gate in use, also one that only a connection's close still refers to, so a forked child can reset them all.
_GATES: "WeakSet[_OpenCloseGate]" = WeakSet()


def _serialize_close(connection: Any, gate: _OpenCloseGate) -> None:
    """Make ``connection.close()`` pass ``gate``, the gate its opening held (see DatabaseEngineRegistry._track)."""
    close = connection.close

    def close_serially() -> None:
        gate.close(close)

    try:
        connection.close = close_serially
    except (AttributeError, TypeError):
        pass


class _Lent(dict):
    """The connections checked out on one database: id of the pool entry -> (token of the thread, weak pool entry).

    A checkout is one store and a checkin one removal, each of which an interrupt leaves done or not done, so no lock
    guards them. An entry whose checkin an interrupt cut off is not in use any more and does not count (see
    ``DatabaseEngineRegistry._held``).
    """

    __slots__ = ("__weakref__",)


class _FileDrop:
    """One drop that is removing a DuckDB file; it ends with a single store, which an interrupt leaves done or not done."""

    __slots__ = ("over",)

    def __init__(self) -> None:
        """Start a drop.

        Returns:
            None: This initializer does not return a value.
        """
        self.over = False


class DropRefusedError(RuntimeError):
    """A drop refused to remove a database that a connection still uses, and removed nothing of that database.

    A workspace drop first checks every route for a connection of the dropping thread, and waits, without a lock, for
    the connections then in use on its DuckDB files; a refusal of either drops no route. It then drops its routes in
    order and stops at a refusal that a connection lent later causes: the routes after it are kept. The error says
    which routes were dropped and which routes' drops failed before the refusal, or that no route was dropped when
    neither happened.
    """


# Raised by a call of the dropping Context on its database while the drop has the scope closed, when the calling thread
# may hold something the drop waits for (see DatabaseEngineRegistry._may_wait), so waiting could deadlock.
_CANNOT_WAIT = (
    "Database is being dropped by this Context, and this thread cannot wait for the drop because it holds a "
    "HeavenBase lock or a connection to the database (such as an open transaction). Retry after the drop."
)


class _DropLock:
    """Reentrant lock held by one drop of a database, kept while anything refers to it."""

    __slots__ = ("lock", "__weakref__")

    def __init__(self) -> None:
        """Create an unheld lock.

        Returns:
            None: This initializer does not return a value.
        """
        self.lock = threading.RLock()


class DatabaseEngineRegistry:
    """Cache SQLAlchemy engines and schema indexes by resolved database spec."""

    _CACHE_MAX = 2048
    _engines: "OrderedDict[str, Any]" = OrderedDict()
    _schema_indices: "OrderedDict[str, SchemaIndex]" = OrderedDict()
    _schema_builds: dict[str, object] = {}
    _disposed: set[str] = set()
    # Lifecycle keys whose scope a drop of the key's authority has closed while it removes the database.
    _dropping: set[str] = set()
    # Locks a drop holds for its whole run, by name (see _drop_names), so drops of one database, from any
    # Context, run one at a time. Taken before the registry lock and never while it is held.
    _drop_locks: "WeakValueDictionary[str, _DropLock]" = WeakValueDictionary()
    _authorities: "WeakValueDictionary[str, Any]" = WeakValueDictionary()
    _released: "deque[str]" = deque()
    # Lifecycle key of each cached local-file database -> its resolved path, so a
    # drop also reaches the engines that other authorities opened on an alias.
    _paths: dict[str, str] = {}
    # Twice the paths the last prune kept: the map is pruned again once it holds
    # max(_CACHE_MAX, _paths_limit) paths.
    _paths_limit = 0
    # Called with the spec, dropping facade and alias paths of each database a drop removes; each
    # returns a context manager that is held, outside the lock, for the whole drop.
    _drop_listeners: "list[Callable[[DBSpec, Any, frozenset[str]], AbstractContextManager[None]]]" = []
    _lock = threading.Lock()
    # Notified whenever a drop reopens a scope it closed (see _await_scope).
    _reopened = threading.Condition(_lock)
    # Lifecycle key -> thread running its drop, and thread -> how many drops it runs (see _may_wait).
    _drop_threads: dict[str, int] = {}
    _droppers: dict[int, int] = {}
    # Drop-lock file identity last seen for each local path, so a drop names a file it can no longer stat.
    _identities: "OrderedDict[str, str]" = OrderedDict()
    # Connection use of the engines this registry creates (see _track), under its own lock, taken after _lock. The lock
    # itself is entered, which an interrupted call always leaves free; the condition on it is only waited on and notified.
    _uses_lock = threading.RLock()
    _uses = threading.Condition(_uses_lock)
    # Database identity (DBSpec._database_key) -> the connections checked out on it; the engines' listeners own the maps.
    _checked_out: "WeakValueDictionary[str, _Lent]" = WeakValueDictionary()
    # DuckDB file path -> its pool entries that hold an open connection -> the number of the checkout it is lent
    # by, which no other checkout has, _IDLE while it sits in a pool, or _CONNECTING right after it connected. Entries
    # are held weakly.
    _duckdb_entries: "dict[str, WeakKeyDictionary[Any, int]]" = {}
    # (DuckDB file path, opening) of each connection being opened (see _track).
    _connecting: "set[tuple[str, object]]" = set()
    # DuckDB file path -> the gate that its connections' opens and closes pass (see _track and _OpenCloseGate).
    _duckdb_open_locks: "dict[str, _OpenCloseGate]" = {}
    # DuckDB file paths and drop-lock file identities that drops are removing -> those drops.
    _closing: "dict[str, tuple[_FileDrop, ...]]" = {}
    # Seconds a DuckDB drop waits for the connections still open on its file before it refuses.
    _DUCKDB_DROP_WAIT = 30.0
    # Seconds after which such a drop takes a connection's return that has not ended for one that an interrupt cut off.
    _RETURN_WAIT = 5.0

    @staticmethod
    def _authority(authority_id: int | str | None) -> str:
        """Return the normalized authority segment used by cache keys."""
        return "standalone" if authority_id is None else str(authority_id)

    @classmethod
    def bind_authority(cls, authority_id: int | str | None, owner: Any) -> None:
        """Weakly bind one cache namespace to its lifecycle owner.

        Args:
            authority_id (int | str | None): Cache authority segment, or ``None`` for
                standalone facades.
            owner (Any): Lifecycle owner whose retirement should dispose cached engines.

        Returns:
            None: This function does not return a value.

        Raises:
            ValueError: If ``authority_id`` is already bound to a different owner.
        """
        authority = cls._authority(authority_id)
        if authority == "standalone" or owner is None:
            return
        try:
            with cls._lock:
                existing = cls._authorities.get(authority)
                if existing is not None and existing is not owner:
                    existing_token = getattr(existing, "authority_token", None)
                    owner_token = getattr(owner, "authority_token", None)
                    if str(existing_token) != authority or str(owner_token) != authority:
                        raise ValueError(f"Database authority {authority!r} is already bound to another owner")
                cls._authorities[authority] = owner
        except TypeError:
            return

    @classmethod
    def _authority_disposed(cls, authority: str) -> bool:
        """Return whether a weakly bound owner retired this exact generation."""
        owner = cls._authorities.get(authority)
        if owner is None:
            return False
        context = getattr(getattr(owner, "authority_config", None), "_context", None)
        for lifecycle in (owner, context):
            if lifecycle is not None and (bool(getattr(lifecycle, "_closed", False)) or bool(getattr(lifecycle, "_closing", False))):
                return True
        # Reset keeps the old generation available until it drops the hidden
        # stores that use these engines. The token then retires the generation.
        current = getattr(owner, "authority_token", None)
        if current is not None and str(current) != authority:
            return True
        context_token = getattr(context, "authority_token", None)
        if context_token is not None and str(context_token) != authority:
            return True
        return getattr(owner, "_active", True) is False

    @classmethod
    def key(
        cls,
        spec: DBSpec,
        *,
        authority_id: int | str | None = None,
        resource_id: str | None = None,
    ) -> str:
        """Return the cache key for a resolved database spec.

        Args:
            spec (DBSpec): Resolved database configuration.

        Returns:
            str: Stable engine-cache key.
        """
        return cls._keys(spec, authority_id, resource_id)[0]

    @classmethod
    def _keys(cls, spec: DBSpec, authority_id: int | str | None, resource_id: str | None) -> tuple[str, str, str]:
        """Return the engine, schema-index and database lifecycle keys."""
        # Engine setup (pool, engine arguments, credentials, initialization) can change
        # which schema a pool sees, so it separates engines and schema indexes. Writes,
        # DDL and drops reach every setup over the same database, so they use its key.
        database_key = f"{cls._authority(authority_id)}:{spec._database_key}"
        key = f"{database_key}:{DB_ENGINE.apply(spec, profile='cache-key')}"
        engine_key = f"{key}:{resource_id}" if resource_id else key
        if spec.in_memory:
            return engine_key, engine_key, engine_key
        return engine_key, key, database_key

    @classmethod
    def _need_available(cls, authority_id: int | str | None, engine_key: str, database_key: str) -> None:
        """Raise unless the engine key is usable, while holding the lock; a closed scope that ``_await_scope`` did not wait for raises its own error."""
        if database_key in cls._dropping:
            raise RuntimeError(_CANNOT_WAIT)
        if cls._unavailable(authority_id, engine_key, database_key):
            raise RuntimeError("Engine has been disposed. Create a new Database instance.")

    @classmethod
    def _unavailable(cls, authority_id: int | str | None, engine_key: str, database_key: str) -> bool:
        """Check lifecycle state while holding the registry lock."""
        return (
            engine_key in cls._disposed
            or database_key in cls._disposed
            or database_key in cls._dropping
            or cls._authority_disposed(cls._authority(authority_id))
        )

    @classmethod
    def _await_scope(cls, spec: DBSpec, database_key: str) -> None:
        """Wait, while holding the lock, until no drop of this authority has the database's scope closed, if that is safe.

        The wait ends when the drop reopens the scope, which is after it removed the database and before its drop
        listeners end. It is skipped, so the caller fails at once, when the thread may hold something the drop waits
        for (see ``_may_wait``).
        """
        while database_key in cls._dropping and cls._may_wait(spec):
            cls._reopened.wait()

    @classmethod
    def _may_wait(cls, spec: DBSpec | None) -> bool:
        """Return whether the calling thread may wait for a drop without deadlocking with it.

        A thread may wait unless it runs a drop itself, holds a Context or workspace lock (another thread that the
        drop waits for, such as one whose transaction holds the database, may wait for that lock), or has a
        connection of this registry's engines checked out on the database of ``spec`` (such as an open transaction,
        whose locks the drop's own statement or file removal waits for). The connection of a response whose iterator
        was abandoned does not count: the settler thread ends it without this thread (see ``settle_abandoned``).
        """
        me = _thread_token()
        if cls._droppers.get(me) or holds_any_lock():
            return False
        if spec is None:
            return True
        key = spec._database_key
        held = cls._held(key, me)
        return not held or held <= abandoned_connections(me).get(key, 0)

    @classmethod
    def _holds_connection(cls, identity: str, thread: int) -> bool:
        """Return whether the thread with token ``thread`` (see ``_thread_token``) has a connection of this registry's engines checked out on database ``identity``."""
        return bool(cls._held(identity, thread))

    @classmethod
    def _held(cls, identity: str, thread: int) -> int:
        """Return how many connections of this registry's engines the thread with token ``thread`` has checked out on database ``identity``."""
        with cls._uses_lock:
            lent = cls._checked_out.get(identity)
        count = 0
        for token, entry in () if lent is None else tuple(lent.values()):
            record = entry() if token == thread else None
            # An entry that went back to its pool while an interrupt cut off its checkin event is not in use.
            count += record is not None and record.in_use
        return count

    @classmethod
    def _holds_any_connection(cls, thread: int | None = None) -> bool:
        """Return whether a thread (the calling one by default) really has a connection of this registry's engines in use.

        ``thread`` is a token from ``_thread_token``. A connection counts from its checkout until its checkin, so one
        held by an open transaction or by a response not yet consumed or closed counts; a response dropped unconsumed
        releases its connection at once. The connection of a response whose iterator was abandoned (see
        ``settle_abandoned``) does not count: the settler thread ends it without this thread. A thread that waits for
        a drop, such as a hidden store that waits for the drops that overlapped its build, must not hold one on the
        dropped database: the drop's statement or file removal may wait for it.
        """
        thread = _thread_token() if thread is None else thread
        with cls._uses_lock:
            identities = [identity for identity, lent in list(cls._checked_out.items()) if lent]
        held = {identity: count for identity in identities if (count := cls._held(identity, thread))}
        if not held:
            return False
        abandoned = abandoned_connections(thread)
        return any(count > abandoned.get(identity, 0) for identity, count in held.items())

    @staticmethod
    def _drop_names(spec: DBSpec) -> list[str]:
        """Return the names of the drop locks of ``spec``'s database, in the order a drop takes them.

        A local file is named by its resolved path and by its file identity,
        so another name of the file (a hard link, or another letter case on a
        case-insensitive file system) shares a lock. A path that no longer
        names a file keeps the identity this process last saw through it (see
        ``_record_identity``), so a drop through another name that starts
        after the file is gone still waits for a running drop of the file.
        A server database is named by its identity. An in-memory database
        belongs to one facade and has none.
        """
        if spec.in_memory:
            return []
        path = DatabaseEngineRegistry._local_path(spec)
        if path is None:
            return [f"database:{spec._database_key}"]
        names = [f"path:{normcase(path)}"]
        identity = DatabaseEngineRegistry._record_identity(path)
        if identity is not None:
            names.append(identity)
        return sorted(names)

    @classmethod
    def _record_identity(cls, path: str) -> str | None:
        """Return the drop-lock name of the file at ``path``, or the one last seen there; the caller does not hold the lock.

        While ``path`` names a file, its identity is recorded for the path (at most ``_CACHE_MAX`` paths, the least
        recently seen forgotten first). Once it names none, the recorded identity is returned, or ``None``.
        """
        key = normcase(path)
        identity = _file_identity(path)
        if identity is None:
            with cls._lock:
                return cls._identities.get(key)
        with cls._lock:
            cls._identities[key] = identity
            cls._identities.move_to_end(key)
            while len(cls._identities) > cls._CACHE_MAX:
                cls._identities.popitem(last=False)
        return identity

    @classmethod
    @contextmanager
    def _holding_drop_locks(cls, spec: DBSpec) -> Iterator[None]:
        """Hold the drop locks of ``spec``'s database; the caller does not hold the registry lock."""
        # Naming a local file reads the file system, so it runs before the lock is taken.
        names = cls._drop_names(spec)
        with cls._lock:
            locks = []
            for name in names:
                lock = cls._drop_locks.get(name)
                if lock is None:
                    lock = cls._drop_locks[name] = _DropLock()
                locks.append(lock)
        with ExitStack() as held:
            for lock in locks:
                held.enter_context(lock.lock)
            yield

    @classmethod
    def clear_disposed(
        cls,
        spec: DBSpec,
        *,
        authority_id: int | str | None = None,
        resource_id: str | None = None,
    ) -> None:
        """Allow a newly created facade to create a fresh engine.

        Args:
            spec (DBSpec): Resolved database configuration.

        Returns:
            None: This function does not return a value.

        Raises:
            RuntimeError: If the authority was disposed. While the authority's
                drop of the database has the scope closed, the call waits for
                the drop, unless the thread may hold something the drop waits
                for (see ``_may_wait``): then it raises at once.
        """
        engine_key, _, database_key = cls._keys(spec, authority_id, resource_id)
        with cls._lock:
            if cls._authority_disposed(cls._authority(authority_id)):
                raise RuntimeError("Database authority has been disposed")
            cls._await_scope(spec, database_key)
            if database_key in cls._dropping:
                raise RuntimeError(_CANNOT_WAIT)
            if cls._authority_disposed(cls._authority(authority_id)):
                raise RuntimeError("Database authority has been disposed")
            cls._disposed.difference_update((engine_key, database_key))

    @classmethod
    def is_disposed(
        cls,
        spec: DBSpec,
        *,
        authority_id: int | str | None = None,
        resource_id: str | None = None,
    ) -> bool:
        """Return whether this spec key has been disposed.

        Args:
            spec (DBSpec): Resolved database configuration.

        Returns:
            bool: `True` when the engine key is marked disposed.
        """
        engine_key, _, database_key = cls._keys(spec, authority_id, resource_id)
        with cls._lock:
            return cls._unavailable(authority_id, engine_key, database_key)

    @classmethod
    def has_engine(
        cls,
        spec: DBSpec,
        *,
        authority_id: int | str | None = None,
        resource_id: str | None = None,
    ) -> bool:
        """Return whether this spec currently has a cached engine.

        Args:
            spec (DBSpec): Resolved database configuration.

        Returns:
            bool: `True` when an active engine is cached for `spec`.
        """
        engine_key, _, database_key = cls._keys(spec, authority_id, resource_id)
        with cls._lock:
            return not cls._unavailable(authority_id, engine_key, database_key) and engine_key in cls._engines

    @classmethod
    def get(
        cls,
        spec: DBSpec,
        autocreate: bool = True,
        *,
        authority_id: int | str | None = None,
        resource_id: str | None = None,
    ) -> Any:
        """Return a cached engine or create one.

        Args:
            spec (DBSpec): Resolved database configuration.
            autocreate (bool): Whether newly created engines should attempt namespace creation first.

        Returns:
            Any: SQLAlchemy engine for `spec`.

        Raises:
            RuntimeError: If the engine key was disposed before or during
                creation. While the same authority's drop of the database has
                the scope closed, the call waits for the drop, also for an
                engine already cached, and then raises as a retired facade
                does; a thread that may hold something the drop waits for
                (see ``_may_wait``) raises at once instead.
        """
        key, _, database_key = cls._keys(spec, authority_id, resource_id)
        with cls._lock:
            cls._await_scope(spec, database_key)
            cls._need_available(authority_id, key, database_key)
            cls._drop_released()
            if key in cls._engines:
                cls._engines.move_to_end(key)
                return cls._engines[key]
        engine = db_engine.create_database_engine(spec, autocreate=autocreate)
        cls._track(engine, spec)
        # Resolving a local file reads the file system, so it runs before the lock is taken.
        path = cls._local_path(spec)
        if path is not None:
            cls._record_identity(path)
        with cls._lock:
            cls._await_scope(spec, database_key)
            try:
                cls._need_available(authority_id, key, database_key)
            except RuntimeError:
                engine.dispose()
                raise
            if key in cls._engines:
                engine.dispose()
                cls._engines.move_to_end(key)
                return cls._engines[key]
            if len(cls._engines) >= cls._CACHE_MAX:
                old_key, old = cls._engines.popitem(last=False)
                cls._forget_schema_index(old_key)
                old.dispose()
            cls._engines[key] = engine
            if path is not None:
                if database_key not in cls._paths and len(cls._paths) >= max(cls._CACHE_MAX, cls._paths_limit):
                    cls._prune_paths()
                cls._paths[database_key] = path
            return engine

    @classmethod
    def _prune_paths(cls) -> None:
        """Forget the paths of lifecycle keys with nothing cached; the caller holds the lock.

        One pass collects the keys that still have an engine, schema index or schema build. The
        next pass waits until the map has doubled again, so a new database costs amortized
        constant work here however many engines are cached. The map stays within
        max(``_CACHE_MAX``, twice the number of live scopes at the last prune).
        """
        live = {_scope(key) for key in (*cls._engines, *cls._schema_indices, *cls._schema_builds)}
        for scope in [scope for scope in cls._paths if scope not in live]:
            del cls._paths[scope]
        cls._paths_limit = 2 * len(cls._paths)

    @classmethod
    def get_schema_index(
        cls,
        spec: DBSpec,
        *,
        builder: Callable[[], Mapping[str, Sequence[str]] | dict[str, Any]] | None = None,
        refresh: bool = False,
        authority_id: int | str | None = None,
        resource_id: str | None = None,
    ) -> SchemaIndex:
        """Return cached schema-index data for a database key.

        Args:
            spec (DBSpec): Resolved database configuration.
            builder (Callable[[], Mapping[str, Sequence[str]] | dict[str, Any]] | None): Optional callback that
                builds fresh schema-index data when the cache is empty or `refresh=True`.
            refresh (bool): Whether to rebuild the schema index even when cached data exists.

        Returns:
            SchemaIndex: Detached schema index for the database key.

        Raises:
            RuntimeError: If the engine key was disposed. While the same
                authority's drop of the database has the scope closed, the
                call waits for the drop, unless the thread may hold something
                the drop waits for (see ``_may_wait``): then it raises at once.
        """
        engine_key, key, database_key = cls._keys(spec, authority_id, resource_id)
        with cls._lock:
            cls._await_scope(spec, database_key)
            cls._need_available(authority_id, engine_key, database_key)
            if not refresh and key in cls._schema_indices:
                cls._schema_indices.move_to_end(key)
                return deepcopy(cls._schema_indices[key])
            if builder is None:
                return {}
            build = object()
            cls._schema_builds[key] = build

        schema_index: SchemaIndex | None = None
        try:
            schema_index = normalize_schema_index(builder())
        finally:
            with cls._lock:
                cls._drop_released()
                # Reflection runs outside the lock. A pre-invalidation result
                # can still reach its caller but must never refill the cache.
                if cls._schema_builds.get(key) is build:
                    del cls._schema_builds[key]
                    if schema_index is not None and not cls._unavailable(authority_id, engine_key, database_key):
                        cls._schema_indices[key] = deepcopy(schema_index)
                        cls._schema_indices.move_to_end(key)
                        while len(cls._schema_indices) > cls._CACHE_MAX:
                            cls._schema_indices.popitem(last=False)
        return deepcopy(schema_index)

    @classmethod
    def set_schema_index(
        cls,
        spec: DBSpec,
        schema_index: Mapping[str, Sequence[str]] | dict[str, Any] | None,
        *,
        authority_id: int | str | None = None,
        resource_id: str | None = None,
    ) -> None:
        """Set the schema-index cache for a database key.

        Args:
            spec (DBSpec): Resolved database configuration.
            schema_index (Mapping[str, Sequence[str]] | dict[str, Any] | None): Raw schema-index data accepted by
                `normalize_schema_index`.

        Returns:
            None: This function does not return a value.

        Raises:
            RuntimeError: If the engine key was disposed.
            ValueError: If `schema_index` has an unsupported shape.
        """
        engine_key, key, database_key = cls._keys(spec, authority_id, resource_id)
        with cls._lock:
            cls._await_scope(spec, database_key)
            cls._need_available(authority_id, engine_key, database_key)
            cls._drop_released()
            cls._schema_indices[key] = normalize_schema_index(schema_index)
            cls._schema_builds.pop(key, None)
            cls._schema_indices.move_to_end(key)
            while len(cls._schema_indices) > cls._CACHE_MAX:
                cls._schema_indices.popitem(last=False)

    @classmethod
    def invalidate_schema_index(
        cls,
        spec: DBSpec,
        *,
        authority_id: int | str | None = None,
        resource_id: str | None = None,
    ) -> None:
        """Remove cached schema-index data for one spec.

        Args:
            spec (DBSpec): Resolved database configuration.

        Returns:
            None: This function does not return a value.
        """
        _, _, database_key = cls._keys(spec, authority_id, resource_id)
        with cls._lock:
            cls._discard_schema_indexes(database_key)

    @classmethod
    def _forget_schema_index(cls, key: str) -> None:
        """Discard the cached index and pending publication while holding the registry lock."""
        cls._schema_indices.pop(key, None)
        cls._schema_builds.pop(key, None)

    @classmethod
    def _discard_schema_indexes(cls, database_key: str) -> None:
        """Discard reflected schemas for every setup variant while holding the lock."""
        prefix = f"{database_key}:"
        for key in cls._schema_indices.keys() | cls._schema_builds.keys():
            if key == database_key or key.startswith(prefix):
                cls._forget_schema_index(key)

    @classmethod
    def dispose(
        cls,
        spec: DBSpec,
        *,
        authority_id: int | str | None = None,
        resource_id: str | None = None,
    ) -> None:
        """Dispose one cached engine.

        Args:
            spec (DBSpec): Resolved database configuration.

        Returns:
            None: This function does not return a value.
        """
        key, schema_key, _ = cls._keys(spec, authority_id, resource_id)
        with cls._lock:
            engine = cls._engines.pop(key, None)
            cls._forget_schema_index(schema_key)
            cls._disposed.add(key)
        if engine is not None:
            engine.dispose()

    @classmethod
    @contextmanager
    def _dropping_database(cls, spec: DBSpec, *, authority_id: int | str | None, resource_id: str | None, db: Any = None) -> Iterator[None]:
        """Close the database's pools before physical removal and keep the scope closed during it.

        A thread that has a connection of this registry's engines checked out
        on the database, such as an open transaction, cannot drop it: the
        drop's statement or file removal would wait for that connection, so
        the call fails at once. Otherwise the drop holds the database's drop
        locks (see ``_drop_names``) for its whole run, so it waits for any
        other drop of the database, from any Context and through any facade.
        The drop listeners record the drop and detach the hidden stores of
        every Context on the database, outside the registry lock (see
        ``heavenbase.workspace.hidden_store``), before the drop and again
        after it. Once the first pass has detached them, the drop closes this
        authority's scope on the database and takes the cached engines: until
        it reopens the scope, after the removal and before the listeners end,
        a facade, an engine use or a schema index of this authority on the
        database waits for the drop (see ``_await_scope``), also through an
        engine or index cached before the drop. A thread that may hold
        something the drop waits for (see ``_may_wait``) fails at once
        instead, so a thread whose server transaction the drop's statement
        waits for never waits for the drop in turn. ``db`` is the ``Database``
        facade that drops it, if any.
        """
        _, _, database_key = cls._keys(spec, authority_id, resource_id)
        cls._need_no_connection(spec)
        with cls._holding_drop_locks(spec), cls._running_drop(database_key):
            # Comparing files reads the file system, so it runs before the lock is taken.
            paths = cls._same_paths(spec)
            try:
                with cls._notifying_drop(spec, db=db, paths=paths):
                    try:
                        with cls._lock:
                            cls._dropping.add(database_key)
                            engines = cls._take_database(spec, database_key, paths)
                        for engine in engines:
                            engine.dispose()
                        dropped = False
                        late: list[Any] = []
                        try:
                            yield
                            dropped = True
                        finally:
                            with cls._lock:
                                if dropped:
                                    cls._disposed.add(database_key)
                                    # Other authorities may have opened the database during the drop.
                                    late = cls._take_database(spec, database_key, paths)
                                else:
                                    cls._disposed.discard(database_key)
                            for engine in late:
                                engine.dispose()
                    finally:
                        cls._reopen(database_key)
            finally:
                # Also when the listeners fail to start.
                cls._reopen(database_key)

    @classmethod
    @contextmanager
    def _running_drop(cls, database_key: str | None) -> Iterator[None]:
        """Record the calling thread as running a drop (of ``database_key``, if given) while the block runs.

        The thread holds the database's drop locks, which are reentrant, so a
        drop that this thread makes from inside its own drop of the same
        database is refused here.
        """
        me = _thread_token()
        before = cls._droppers.get(me, 0)
        owned = False
        try:
            with cls._lock:
                if database_key is not None:
                    if database_key in cls._drop_threads:
                        # The drop locks make every other drop of the database wait, so only a drop made from inside
                        # this one, on its own thread, gets here.
                        raise RuntimeError("Database is being dropped")
                    cls._drop_threads[database_key] = me
                    owned = True
                cls._droppers[me] = before + 1
            yield
        finally:
            # Only this thread changes either record, each in one operation, so an interrupt leaves none behind.
            try:
                if before:
                    cls._droppers[me] = before
                else:
                    cls._droppers.pop(me, None)
            finally:
                if owned:
                    cls._drop_threads.pop(database_key, None)

    @classmethod
    def _need_no_connection(cls, spec: DBSpec) -> None:
        """Refuse a drop by a thread that has a connection of this registry's engines checked out on the database.

        The responses this thread executed on the database whose iterators were abandoned, and that a garbage
        collection or a read left to the settler thread, are first settled here, on this thread (see
        ``settle_abandoned``), so their connections do not refuse the drop.
        """
        me = _thread_token()
        if spec.in_memory or not cls._holds_connection(spec._database_key, me):
            return
        settle_abandoned(me, spec._database_key)
        if not cls._holds_connection(spec._database_key, me):
            return
        raise DropRefusedError(
            "Cannot drop the database while this thread has a connection to it in use, such as an open transaction: "
            "the drop would wait for that connection. End the transaction or close the result first."
        )

    @classmethod
    def _reopen(cls, database_key: str) -> None:
        """Reopen the scope a drop closed and wake the calls waiting for it; the caller does not hold the lock."""
        with cls._lock:
            cls._dropping.discard(database_key)
            cls._reopened.notify_all()

    @classmethod
    @contextmanager
    def _drop_scope(cls, spec: DBSpec, *, db: Any = None) -> Iterator[frozenset[str]]:
        """Run one drop of ``spec``'s database at a time, from any Context, inside the drop listeners.

        The drop holds the database's drop locks (see ``_drop_names``) from
        before its listeners start until after they end, so a drop of the
        same database through any facade, in this Context or another, waits
        for it. ``Database.drop()`` runs here. Like ``_dropping_database``,
        it refuses a thread that has a connection checked out on the database.

        Args:
            spec (DBSpec): Spec of the database being dropped.
            db (Any): ``Database`` facade that drops it, or ``None``.

        Yields:
            frozenset[str]: The recorded local-file paths that name the dropped file.
        """
        cls._need_no_connection(spec)
        with cls._holding_drop_locks(spec), cls._running_drop(None):
            # Comparing files reads the file system, so it runs before the lock is taken.
            paths = cls._same_paths(spec)
            with cls._notifying_drop(spec, db=db, paths=paths):
                yield paths

    @classmethod
    @contextmanager
    def _notifying_drop(cls, spec: DBSpec, *, db: Any = None, paths: frozenset[str] | None = None) -> Iterator[None]:
        """Hold every drop listener's context, outside the lock, while the database or its tables are dropped.

        Each listener receives the spec, the dropping ``Database`` facade (or
        ``None``) and the recorded local-file paths that name the dropped file.
        """
        if paths is None:
            paths = cls._same_paths(spec)
        with ExitStack() as listeners:
            for listener in tuple(cls._drop_listeners):
                listeners.enter_context(listener(spec, db, paths))
            yield

    @staticmethod
    def _local_path(spec: DBSpec) -> str | None:
        """Return the resolved path of a local-file database, or ``None``."""
        if spec.in_memory or spec.database is None or not spec.dialect_spec.supports("database-local-path"):
            return None
        path = file_uri_to_path(spec.database)
        return realpath(path) if isabs(path) else None

    @classmethod
    def _same_paths(cls, spec: DBSpec) -> frozenset[str]:
        """Return the recorded local-file paths that name the file ``spec`` opens, such as a case alias.

        The caller must not hold the lock: each recorded path other than the
        file's own is compared with it on the file system, while it still exists.
        """
        path = cls._local_path(spec)
        if path is None:
            return frozenset()
        with cls._lock:
            recorded = set(cls._paths.values())
        return frozenset({path, *(other for other in recorded if other != path and _same_file(other, path))})

    @classmethod
    def _take_database(cls, spec: DBSpec, database_key: str, paths: frozenset[str]) -> list[Any]:
        """Remove every cached engine and schema index on one database; the caller holds the lock.

        A drop removes the database for the whole process, so the engines that
        other authorities (Contexts) opened on it, directly or through a path
        in ``paths`` (from ``_same_paths``), are removed too. Their facades
        open a fresh engine at their next use, instead of serving pooled
        connections to the removed database. Disposing an engine closes only
        its idle connections: on DuckDB a connection still in use keeps the
        database open in the process (see ``Database.drop_database``). Only
        this authority's scope is marked disposed. Each cache is scanned a
        fixed number of times, however many authorities hold an engine on the
        database.

        Args:
            spec (DBSpec): Spec of the database being dropped.
            database_key (str): This authority's lifecycle key for the database.
            paths (frozenset[str]): Local-file paths that name the dropped file.

        Returns:
            list[Any]: The removed engines, for the caller to dispose outside the lock.
        """
        scopes = {database_key}
        if not spec.in_memory:
            identity = spec._database_key
            for key in (*cls._engines, *cls._schema_indices, *cls._schema_builds):
                authority, _, rest = key.partition(":")
                if rest == identity or rest.startswith(f"{identity}:"):
                    scopes.add(f"{authority}:{identity}")
            scopes.update(scope for scope, other in cls._paths.items() if other in paths)
        engines = [cls._engines.pop(key) for key in [key for key in cls._engines if key in scopes or _scope(key) in scopes]]
        for key in [key for key in cls._schema_indices.keys() | cls._schema_builds.keys() if key in scopes or _scope(key) in scopes]:
            cls._forget_schema_index(key)
        for scope in scopes:
            cls._paths.pop(scope, None)
        return engines

    @classmethod
    def _track(cls, engine: Any, spec: DBSpec) -> None:
        """Record the connection use of a new engine on a stored database; the caller does not hold the lock.

        Every checkout is counted per database and thread until its checkin,
        so a drop can tell whether a thread really has a connection on the
        database in use (see ``_may_wait`` and ``_need_no_connection``). On a
        DuckDB file, which DuckDB keeps open in the process while any
        connection to it is open, every pool entry that holds an open
        connection is also recorded, in whichever pool it sits, so a drop of
        the file can close them all first (see ``_closing_duckdb_file``).
        While such a drop runs, opening a connection to the file waits for
        the drop before DuckDB opens the file, a checkout closes the
        connection it was handed and connects again after the drop, and a
        connection returned is closed; a thread that may hold something the
        drop waits for (see ``_may_wait``) fails at once instead of waiting.
        """
        if spec.in_memory:
            return
        identity = spec._database_key
        path = cls._local_path(spec) if spec.dialect == "duckdb" else None
        with cls._uses_lock:
            lent = cls._checked_out.get(identity)
            if lent is None:
                lent = cls._checked_out[identity] = _Lent()

        if path is None:

            def checked_out(dbapi_connection: Any, record: Any, proxy: Any) -> None:
                lent[id(record)] = (_thread_token(), ref(record))

            def checked_in(dbapi_connection: Any, record: Any) -> None:
                lent.pop(id(record), None)

            event.listen(engine, "checkout", checked_out)
            event.listen(engine, "checkin", checked_in)
            return

        # Bumped by every dispose of the engine, which replaces its pool. Each pool entry keeps the generation of the
        # pool that created it, so a connection returned to a pool that a dispose replaced is closed, not left open
        # there for nothing to use.
        generation = [0]
        created: "WeakKeyDictionary[Any, int]" = WeakKeyDictionary()

        def refused() -> RuntimeError:
            return RuntimeError(
                f"DuckDB database {path!r} is being dropped, and this thread cannot wait for the drop because it holds a "
                "HeavenBase lock or a connection to the file. Retry after the drop."
            )

        def opening(dialect: Any, record: Any, cargs: Any, cparams: Any) -> Any:
            # Runs before DuckDB opens the file, so no connection opens it between a drop's last check and the removal.
            file = _file_identity(path)
            connection = None
            connecting = (path, object())
            try:
                with cls._uses_lock:
                    created.setdefault(record, generation[0])
                    while cls._is_closing(path, file):
                        if not cls._may_wait(spec):
                            raise refused()
                        cls._uses.wait(_WAIT_STEP)
                    cls._connecting.add(connecting)
                serial = cls._duckdb_open_lock(path)
                # DuckDB can fail a connection with "Unique file handle conflict" when it opens a file while another
                # thread closes the last connection of that file's database, so opens and closes of one file run one
                # at a time.
                with serial.held():
                    connection = dialect.connect(*cargs, **cparams)
                _serialize_close(connection, serial)
                return connection
            finally:
                try:
                    if connection is not None:
                        with cls._uses_lock:
                            # Lent at once: the pool connects an entry only to hand it out.
                            cls._duckdb_entry_map(path)[record] = _CONNECTING
                finally:
                    cls._connecting.discard(connecting)
                    with cls._uses_lock:
                        cls._uses.notify_all()

        def checked_out(dbapi_connection: Any, record: Any, proxy: Any) -> None:
            token = _thread_token()
            while True:
                with cls._uses_lock:
                    closing = cls._is_closing(path)
                    if not closing and not getattr(proxy.dbapi_connection, "closed", False):
                        lent[id(record)] = (token, ref(record))
                        cls._duckdb_entry_map(path)[record] = next(_CHECKOUTS)
                        return
                # A drop closed this connection or is removing the file: close it, so the drop does not wait for it,
                # wait for the drop to end, and connect again, which reaches the file as the drop left it. Opening the
                # connection waits for a drop that starts meanwhile (see opening), and this checks again.
                record.invalidate()
                if closing:
                    if not cls._may_wait(spec):
                        raise refused()
                    with cls._uses_lock:
                        while cls._is_closing(path):
                            cls._uses.wait(_WAIT_STEP)
                proxy.dbapi_connection = record.get_connection()

        def checked_in(dbapi_connection: Any, record: Any) -> None:
            if dbapi_connection is not None:
                with cls._uses_lock:
                    closing = cls._is_closing(path) or created.get(record, generation[0]) != generation[0]
                    if not closing:
                        entries = cls._duckdb_entries.get(path)
                        if entries is not None and record in entries:
                            entries[record] = _IDLE
                if closing:
                    record.invalidate()
            lent.pop(id(record), None)
            # Wake the drops waiting for it.
            with cls._uses_lock:
                cls._uses.notify_all()

        def closed(dbapi_connection: Any, record: Any) -> None:
            with cls._uses_lock:
                entries = cls._duckdb_entries.get(path)
                if entries is not None:
                    entries.pop(record, None)
                cls._uses.notify_all()

        def errored(context: Any) -> None:
            # A pre-ping on a connection that a drop closed reports a closed connection: treat it as a disconnect, so
            # the pool connects again instead of failing the checkout.
            if getattr(context, "is_pre_ping", False) and "closed" in str(context.original_exception).lower():
                context.is_disconnect = True

        def disposed(engine: Any) -> None:
            with cls._uses_lock:
                generation[0] += 1

        event.listen(engine, "engine_disposed", disposed)
        event.listen(engine, "do_connect", opening)
        event.listen(engine, "checkout", checked_out)
        event.listen(engine, "checkin", checked_in)
        event.listen(engine, "close", closed)
        event.listen(engine, "handle_error", errored)

    @classmethod
    def _duckdb_entry_map(cls, path: str) -> "WeakKeyDictionary[Any, int]":
        """Return the recorded pool entries of a DuckDB file, creating the map; the caller holds ``_uses_lock``."""
        entries = cls._duckdb_entries.get(path)
        if entries is None:
            if len(cls._duckdb_entries) >= cls._CACHE_MAX:
                connecting = {known for known, _ in tuple(cls._connecting)}
                for known in [known for known, recorded in cls._duckdb_entries.items() if not recorded and known not in connecting]:
                    del cls._duckdb_entries[known]
            entries = cls._duckdb_entries[path] = WeakKeyDictionary()
        return entries

    @classmethod
    def _duckdb_open_lock(cls, path: str) -> _OpenCloseGate:
        """Return the gate that opens and closes of connections to one DuckDB file pass, creating it."""
        with cls._uses_lock:
            lock = cls._duckdb_open_locks.get(path)
            if lock is None:
                if len(cls._duckdb_open_locks) >= cls._CACHE_MAX:
                    connecting = {known for known, _ in tuple(cls._connecting)}
                    for known in [known for known in cls._duckdb_open_locks if not cls._duckdb_entries.get(known) and known not in connecting]:
                        del cls._duckdb_open_locks[known]
                lock = cls._duckdb_open_locks[path] = _OpenCloseGate()
            return lock

    @classmethod
    def _is_closing(cls, path: str, file: str | None = None) -> bool:
        """Return whether a drop is removing the DuckDB file at ``path`` (or with drop-lock identity ``file``); the caller holds ``_uses_lock``."""
        if not cls._closing:
            return False
        drops = cls._closing.get(path, ()) + (() if file is None else cls._closing.get(file, ()))
        return any(not drop.over for drop in drops)

    @classmethod
    @contextmanager
    def _closing_duckdb_file(cls, spec: DBSpec) -> Iterator[None]:
        """Close every connection of this registry's engines on a DuckDB file before the block removes the file.

        DuckDB keeps a database open in the process while any connection to
        its file is open, and a file removed meanwhile stays readable and
        takes writes that are lost once the database closes. So the block
        first refuses a thread that has a connection to the file in use (see
        ``_need_no_connection``). Then, until it ends, opening a connection to
        the file or to another name of it waits for the drop before DuckDB
        opens the file, a checkout or a connection returned is closed, and
        every idle connection on the file is closed, in whichever pool it
        sits. The block starts once no connection on the file is open or
        being opened. A connection still in use after ``_DUCKDB_DROP_WAIT``
        seconds, such as one in another thread's long transaction, makes the
        drop raise ``DropRefusedError`` before it removes anything. Another
        dialect, or a path the registry cannot resolve, runs the block at once.
        """
        path = None if spec.in_memory or spec.dialect != "duckdb" else cls._local_path(spec)
        if path is None:
            yield
            return
        cls._need_no_connection(spec)
        file = _file_identity(path)
        with cls._uses_lock:
            known = [known for known in cls._duckdb_entries if known != path]
        # Comparing files reads the file system, so it runs before the lock is taken.
        paths = frozenset({path, *(known for known in known if _same_file(known, path))})
        keys = [*paths, *([file] if file is not None else [])]
        drop = _FileDrop()
        try:
            with cls._uses_lock:
                for key in keys:
                    cls._closing[key] = (*cls._closing.get(key, ()), drop)
            cls._close_duckdb_connections(paths)
            yield
        finally:
            # Ends the drop for every waiting call; the rest only forgets it and wakes them.
            drop.over = True
            with cls._uses_lock:
                for key in keys:
                    running = tuple(other for other in cls._closing.get(key, ()) if not other.over)
                    if running:
                        cls._closing[key] = running
                    else:
                        cls._closing.pop(key, None)
                cls._uses.notify_all()

    @classmethod
    def _close_duckdb_connections(cls, paths: frozenset[str]) -> None:
        """Close the idle connections on a closing DuckDB file and wait until none is open, or raise after the bound."""
        deadline = time.monotonic() + cls._DUCKDB_DROP_WAIT
        # Checkout number of each entry that is not in use, though its checkin event has not ended -> when that was first seen.
        unused: dict[int, float] = {}
        while True:
            idle: list[Any] = []
            with cls._uses_lock:
                busy = sum(1 for path, _ in tuple(cls._connecting) if path in paths)
                for path in paths:
                    entries = cls._duckdb_entries.get(path)
                    for record, state in list(entries.items()) if entries is not None else ():
                        connection = record.dbapi_connection
                        if connection is None or getattr(connection, "closed", False):
                            entries.pop(record, None)
                        elif state == _IDLE and not record.in_use:
                            idle.append(record)
                        elif state != _CONNECTING and not record.in_use and time.monotonic() - unused.setdefault(state, time.monotonic()) >= cls._RETURN_WAIT:
                            # An interrupt cut off the return of this entry. A return still running is left to its thread.
                            idle.append(record)
                        else:
                            busy += 1
                if not idle and not busy:
                    return
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise DropRefusedError(
                        f"Cannot drop DuckDB database {min(paths)!r}: {busy} connection(s) to it stayed in use for "
                        f"{cls._DUCKDB_DROP_WAIT:g} seconds, such as one in a long transaction. The file was not removed; retry once they end."
                    )
                if not idle:
                    cls._uses.wait(min(remaining, 0.05))
                    continue
            # Closing an idle entry's connection also reaches a pool that a dispose replaced; the entry connects
            # again at its next checkout, after the drop.
            for record in idle:
                record.invalidate()

    @classmethod
    def _await_duckdb_quiet(cls, spec: DBSpec) -> None:
        """Wait, before taking any lock, until the connections now in use on a DuckDB file were returned, or refuse after the bound.

        A workspace drop calls this before it takes its Context lock, so it does
        not hold that lock while it waits for another Context's transaction on
        the file. Connections lent after this begins are left to the drop itself,
        which closes the file to new ones, so steady short reads do not hold it up.
        """
        path = None if spec.in_memory or spec.dialect != "duckdb" else cls._local_path(spec)
        if path is None:
            return
        cls._need_no_connection(spec)
        deadline = time.monotonic() + cls._DUCKDB_DROP_WAIT

        def in_use() -> dict[int, Any]:
            # The caller holds _uses_lock. Checkout number -> pool entry; a later checkout of the entry has another number.
            entries = cls._duckdb_entries.get(path)
            lent = () if entries is None else list(entries.items())
            return {state: record for record, state in lent if state not in (_IDLE, _CONNECTING) and record.in_use}

        with cls._uses_lock:
            awaited = set(in_use())
            while True:
                lent = in_use()
                connections = [lent[checkout].dbapi_connection for checkout in awaited & lent.keys()]
                busy = sum(1 for connection in connections if connection is not None and not getattr(connection, "closed", False))
                remaining = deadline - time.monotonic()
                if not busy:
                    return
                if remaining <= 0:
                    raise DropRefusedError(
                        f"Cannot drop DuckDB database {path!r}: {busy} connection(s) to it that were in use when the drop began "
                        f"to wait for them stayed in use for {cls._DUCKDB_DROP_WAIT:g} seconds, such as one in a long transaction. "
                        "The file was not removed; retry once they end."
                    )
                cls._uses.wait(min(remaining, 0.05))

    @classmethod
    def release_resource(cls, resource_id: str) -> None:
        """Schedule the engine and schema index cached for one private resource to be forgotten.

        A private resource (the engine of a facade that serves one thread at a time) is unreachable once its owner is
        collected. This runs as that owner's garbage-collection finalizer, which may fire on a thread that already
        holds the registry lock, so it only queues the id; the next cache write drops the keys, disposed markers
        included, since the id is never reused. The engine is not disposed either: an object still holding it keeps
        working, and its connection closes when the last reference goes.

        Args:
            resource_id (str): Resource segment that ends the cache keys to forget.

        Returns:
            None: This function does not return a value.
        """
        cls._released.append(resource_id)

    @classmethod
    def _drop_released(cls) -> None:
        """Drop the cache entries of released private resources; the caller holds the lock."""
        released: set[str] = set()
        while cls._released:
            released.add(cls._released.popleft())
        if not released:
            return
        for cache in (cls._engines, cls._schema_indices, cls._schema_builds):
            for key in tuple(cache):
                if key.rsplit(":", 1)[-1] in released:
                    cache.pop(key, None)
        cls._disposed.difference_update(key for key in tuple(cls._disposed) if key.rsplit(":", 1)[-1] in released)

    @classmethod
    def dispose_authority(cls, authority_id: int | str) -> None:
        """Dispose and permanently retire every engine owned by one authority.

        Args:
            authority_id (int | str): Cache authority segment whose engines should be
                disposed.

        Returns:
            None: This function does not return a value.
        """
        authority = cls._authority(authority_id)
        prefix = f"{authority}:"
        with cls._lock:
            keys = tuple(key for key in cls._engines if key.startswith(prefix))
            engines = [cls._engines.pop(key) for key in keys]
            for key in set(cls._schema_indices) | set(cls._schema_builds):
                if key.startswith(prefix):
                    cls._forget_schema_index(key)
            cls._disposed.difference_update(key for key in tuple(cls._disposed) if key.startswith(prefix))
            for scope in [scope for scope in cls._paths if scope.startswith(prefix)]:
                del cls._paths[scope]
        for engine in engines:
            engine.dispose()

    @classmethod
    def dispose_all(cls) -> None:
        """Dispose every cached engine.

        Args:
            None.

        Returns:
            None: This function does not return a value.
        """
        with cls._lock:
            engines = list(cls._engines.values())
            cls._disposed.update(cls._engines.keys())
            cls._engines.clear()
            cls._schema_indices.clear()
            cls._schema_builds.clear()
            cls._paths.clear()
            cls._paths_limit = 0
        for engine in engines:
            engine.dispose()


def _scope(key: str) -> str:
    """Return the database lifecycle key (authority and database identity) that a cache key starts with."""
    authority, _, rest = key.partition(":")
    return f"{authority}:{rest.partition(':')[0]}"


def _same_file(left: str, right: str) -> bool:
    """Return whether two existing paths name one file, such as through a case alias."""
    try:
        return samefile(left, right)
    except OSError:
        return False


def _reset_drop_state_after_fork() -> None:
    """Give the child its own registry and drop locks; the parent's drops do not run in it."""
    DatabaseEngineRegistry._lock = threading.Lock()
    DatabaseEngineRegistry._reopened = threading.Condition(DatabaseEngineRegistry._lock)
    DatabaseEngineRegistry._uses_lock = threading.RLock()
    DatabaseEngineRegistry._uses = threading.Condition(DatabaseEngineRegistry._uses_lock)
    for lock in tuple(DatabaseEngineRegistry._drop_locks.values()):
        lock.lock = threading.RLock()
    DatabaseEngineRegistry._dropping.clear()
    DatabaseEngineRegistry._drop_threads.clear()
    DatabaseEngineRegistry._droppers.clear()
    # The parent's other threads, and their connections, do not run in the child.
    for lent in list(DatabaseEngineRegistry._checked_out.values()):
        lent.clear()
    DatabaseEngineRegistry._closing.clear()
    DatabaseEngineRegistry._connecting.clear()
    # Reset in place: the close of an inherited connection refers to its gate.
    for gate in tuple(_GATES):
        gate._reset()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_drop_state_after_fork)
