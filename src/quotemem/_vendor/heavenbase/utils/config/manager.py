"""Scoped orchestration for HeavenBase configuration."""

from __future__ import annotations

__all__ = ["CM_HVNB", "ConfigManager"]

import contextlib
import contextvars
import copy
import datetime as dt
import errno
import inspect
import os
import random
import re
import sys
import threading
import time
import warnings
from collections.abc import Mapping
from functools import wraps
from hashlib import sha256
from importlib import resources
from math import copysign
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterator, List, NamedTuple, Optional, Tuple, TypeVar

from heavenbase._lazy import resolve_late

from ..files import pj as _pj
from .data import (
    _OVERWRITE,
    _clear_tombstone_ancestors,
    _list_tail,
    _merge_layers,
    _split_path,
    _tombstone_unset,
    dget,
    dset,
    dunset,
    to_mutable_data,
)
from .interpolation import InterpolationPolicy, _capture_env, _Environment, _identity, _resolve_interpolation
from .snapshot import ConfigSnapshot

if TYPE_CHECKING:
    from heavenbase.utils.registry import Registry

_DEFAULT_CACHE_MAX = 65536
# A write that loses a compare-and-set to another writer refreshes and runs
# again. Retries back off exponentially with full jitter up to this cap, so
# many contending processes thin out instead of colliding at a fixed rate,
# and stop with the last conflict once the deadline has passed.
_COMMIT_BACKOFF_MAX = 1.0
_COMMIT_DEADLINE = 300.0
# Writers of one home take this OS file lock, in the home, around their
# read-modify-write, so they run one after another instead of racing.
_WRITE_LOCK_NAME = ".config-write.lock"
_WRITE_LOCKS_HELD = threading.local()
# A writer waits at most this long for the lock; then it goes on with
# compare-and-set alone, which stays correct and only contends.
_WRITE_LOCK_WAIT = 5.0
_R = TypeVar("_R")
_SCOPE_STACKS: contextvars.ContextVar[Dict[str, List[str]]] = contextvars.ContextVar("_HB_CONFIG_SCOPE_STACKS", default={})
_ENV_REFERENCE = re.compile(r"""\$\{[ \t]*env[ \t]*:[ \t]*(?P<quote>['"]?)(?P<name>[A-Za-z_][A-Za-z_0-9-]*)(?P=quote)[ \t]*(?:,[^{}]*)?\}""")


def _env_names(value: Any) -> frozenset[str | None]:
    """Find captured environment names, marking expressions that need live resolution."""
    if isinstance(value, str):
        names: set[str | None] = set()
        start = value.find("${")
        while start != -1:
            reference = _ENV_REFERENCE.match(value, start)
            if reference:
                names.add(reference["name"])
            else:
                reference = _NODE_REFERENCE.match(value, start)
                if reference is None:
                    # Computed arguments and other resolvers can introduce dependencies.
                    return frozenset({None})
            start = value.find("${", reference.end())
        return frozenset(names)
    if isinstance(value, Mapping):
        return frozenset(name for item in value.values() for name in _env_names(item))
    if isinstance(value, (list, tuple)):
        return frozenset(name for item in value for name in _env_names(item))
    return frozenset()


def _same_data(left: Any, right: Any) -> bool:
    """Compare stored config values by type and preserve the sign of zero."""
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        if left.keys() != right.keys():
            return False
        right_keys = {key: key for key in right}
        return all(_same_data(key, right_keys[key]) and _same_data(value, right[key]) for key, value in left.items())
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(_same_data(a, b) for a, b in zip(left, right))
    if type(left) is float and left == right == 0.0:
        return copysign(1.0, left) == copysign(1.0, right)
    return bool(left == right)


_UNOPENABLE_TEXT = ("unable to open database file", "readonly database", "read-only file system")


def _stamp(generation: int, layer: Mapping[str, Any]) -> str:
    """Digest a scope's generation and latest layer, all a snapshot reads of it."""
    return sha256(repr((generation, _identity(layer))).encode("utf-8")).hexdigest()


class _ScopeStamp(NamedTuple):
    digest: str
    environment_names: frozenset[str | None]


class _CachedGeneration(NamedTuple):
    checked_at: float
    generations: Tuple[int, ...]
    stamps: Tuple[_ScopeStamp, ...]
    registry: Registry[str, Dict[str, Any]]
    publication: int


def _explicit_context_required() -> bool:
    return bool(getattr(sys.modules.get("heavenbase._initialization"), "explicit_context_required", False))


def _cannot_open(error: BaseException) -> bool:
    """Whether a failed first open of the saved config is one that lasts.

    These are the causes the tolerant reads and the log helpers fall back on:
    an unwritable home, an invalid bootstrap file, and explicit Context mode.
    The default Context raises a ``ValueError`` naming the bootstrap both for
    a bootstrap that fails validation and for one whose backend cannot be
    built (a module or driver it names cannot be imported, or the backend
    rejects its settings). Any other error, such as a busy database or a
    Registry conflict, may pass or needs attention, so it is raised and the
    open is tried again later.
    """
    explicit = _explicit_context_required()
    seen: set[int] = set()
    current: Optional[BaseException] = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        text = str(current).lower()
        if isinstance(current, RuntimeError) and explicit and "explicit context mode" in text:
            return True
        if isinstance(current, ValueError) and "bootstrap" in text:
            return True
        if isinstance(current, PermissionError) or (isinstance(current, OSError) and current.errno in (errno.EACCES, errno.EPERM, errno.EROFS)):
            return True
        if any(marker in text for marker in _UNOPENABLE_TEXT):
            return True
        current = current.__cause__ or current.__context__
    return False


def _cache_max(value: Any, path: str) -> int:
    out = int(_DEFAULT_CACHE_MAX if value is None else value)
    if out < 0:
        raise ValueError(f"{path} must be >= 0")
    return out


def _load_yaml_resource(package: str, *parts: str) -> Dict[str, Any]:
    import yaml

    try:
        text = resources.files(package).joinpath("resources", *parts).read_text(encoding="utf-8")
    except Exception:
        return {}
    return yaml.safe_load(text) or {}


def _home(package: str) -> str:
    return _pj("~", f".{package}", abs=True)


class _Contended(Exception):
    """Another holder has the home write lock right now."""


def _try_lock_file(fd: int) -> None:
    """Take the exclusive lock on this open file's first byte without waiting.

    Raises:
        _Contended: If another open file holds the lock.
        OSError: If the filesystem cannot lock the file (``ENOLCK``, ``ENOTSUP``, ...).
    """
    if sys.platform == "win32":
        import msvcrt

        os.lseek(fd, 0, os.SEEK_SET)
        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        except OSError as error:
            if error.errno in (errno.EACCES, errno.EDEADLOCK):
                raise _Contended from None
            raise
    else:
        import fcntl

        # ``flock`` locks belong to the open file, so threads of one process
        # that open the file separately exclude each other as well.
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise _Contended from None


def _unlock_file(fd: int) -> None:
    """Release the lock :func:`_try_lock_file` took."""
    if sys.platform == "win32":
        import msvcrt

        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_UN)


# Lock files this process holds, by descriptor. A child forked while a write
# holds one closes its copy at once, so a lock never outlives its writer
# through a forked child. ``_WRITE_LOCKS_GUARD`` covers taking a lock and
# recording it, and a fork holds it from its first to its last hook, so no
# child inherits an unrecorded lock. It is reentrant: a config write made by
# another library's fork hook runs on the forking thread, which holds it, so
# that write never waits for the fork (nor, in the child, for the guard the
# child inherited from that same thread).
_WRITE_LOCK_FDS: Dict[int, Tuple[int, int]] = {}
_WRITE_LOCKS_GUARD = threading.RLock()


def _before_fork() -> None:
    _WRITE_LOCKS_GUARD.acquire()


def _after_fork_in_parent() -> None:
    _WRITE_LOCKS_GUARD.release()


def _after_fork_in_child() -> None:
    """In a forked child, close the copies of the parent's held lock files and forget them."""
    global _WRITE_LOCKS_GUARD
    for fd in list(_WRITE_LOCK_FDS):
        try:
            os.close(fd)
        except OSError:
            pass
    _WRITE_LOCK_FDS.clear()
    _WRITE_LOCKS_HELD.files = set()
    _WRITE_LOCKS_GUARD = threading.RLock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(before=_before_fork, after_in_parent=_after_fork_in_parent, after_in_child=_after_fork_in_child)


def _acquire_home_write_lock(home: str) -> Optional[Tuple[int, Tuple[int, int]]]:
    """Take the home write lock, returning its descriptor and file identity, or ``None`` to go on without it."""
    try:
        fd = os.open(os.path.join(home, _WRITE_LOCK_NAME), os.O_RDWR | os.O_CREAT, 0o600)
    except OSError:
        return None  # a home that cannot hold the lock file
    try:
        stat = os.fstat(fd)
        key = (stat.st_dev, stat.st_ino)
        held = getattr(_WRITE_LOCKS_HELD, "files", None)
        if held is None:
            held = _WRITE_LOCKS_HELD.files = set()
        if key in held:
            # This thread holds it already, through this or another spelling of the home.
            os.close(fd)
            return None
        deadline = time.monotonic() + _WRITE_LOCK_WAIT
        pause = 0.001
        while True:
            with _WRITE_LOCKS_GUARD:
                try:
                    _try_lock_file(fd)
                except _Contended:
                    pass
                else:
                    _WRITE_LOCK_FDS[fd] = key
                    break
            if time.monotonic() + pause > deadline:
                os.close(fd)
                return None  # a stopped or hung holder: go on with compare-and-set alone
            time.sleep(pause)
            pause = min(pause * 2, 0.05)
        held.add(key)
        return fd, key
    except OSError:
        # A filesystem that cannot lock files (NFS without lockd, some SMB or FUSE mounts).
        os.close(fd)
        return None


def _release_home_write_lock(fd: int, key: Tuple[int, int]) -> None:
    getattr(_WRITE_LOCKS_HELD, "files", set()).discard(key)
    with _WRITE_LOCKS_GUARD:
        if _WRITE_LOCK_FDS.pop(fd, None) is None:
            return  # closed in a forked child
    try:
        _unlock_file(fd)
    finally:
        os.close(fd)


@contextlib.contextmanager
def _home_write_lock(home: Optional[str]) -> Iterator[None]:
    """Serialize config writers of one Context home across processes and threads.

    The lock is an OS file lock on ``.config-write.lock`` in ``home``, which
    the operating system releases when its holder exits; a child forked while
    the lock is held closes its copy. A thread that already holds the lock of
    the same file, whatever spelling of the home it uses, does not wait for
    itself. A writer goes on with compare-and-set alone, as without the lock,
    when ``home`` is ``None`` or not a directory, when the lock file cannot be
    created or locked, or when another holder keeps the lock for
    ``_WRITE_LOCK_WAIT`` seconds, such as a stopped process.
    """
    lock = _acquire_home_write_lock(home) if home and os.path.isdir(home) else None
    if lock is None:
        yield
        return
    try:
        yield
    finally:
        _release_home_write_lock(*lock)


def _resource_root(package: str) -> str:
    try:
        return str(resources.files(package).joinpath("resources"))
    except Exception:
        return ""


def _path_aliases(package: str, root: Optional[str] = None) -> Dict[str, str]:
    return {"%": root or _home(package), "&": _resource_root(package)}


class ConfigManager:
    """Scoped, cached HeavenBase config manager."""

    def __init__(
        self,
        package: str = "heavenbase",
        scope: Optional[str] = None,
        root: Optional[str] = None,
        default: Optional[Dict[str, Any]] = None,
        registry: Optional[Registry[str, Dict[str, Any]]] = None,
        keep_last_k: Optional[int] = None,
        refresh_interval: Optional[float] = None,
        setup: bool = True,
    ) -> None:
        """
        Initialize the utility object with the provided runtime settings.

        Use this helper for nested config dictionaries and scoped runtime configuration layers.

        Args:
            package (str): Package name whose resources or config layers are being managed. Defaults to ``'heavenbase'``.
            scope (Optional[str]): Config scope. ``None`` uses the current active scope. Defaults to ``None``.
            root (Optional[str]): Optional root path used for relative path resolution. Defaults to ``None``.
            default (Optional[Dict[str, Any]]): Fallback value used when a path or config key is missing. Defaults to ``None``.
            registry (Registry[str, Dict[str, Any]] | None): Raw scope-record Registry.
                A process-only Registry is created when omitted.
            keep_last_k (int | None): Optional retained version count override.
            refresh_interval (float | None): Optional generation polling interval.
            setup (bool): Whether to initialize the base config layer during construction. Defaults to ``True``.

        Returns:
            None: This function returns ``None`` after initializing the object.
        """
        self.package = package.strip().lower()
        self.base_scope = (scope or self.package).strip().lower()
        self.root = root or _home(self.package)
        self.default = copy.deepcopy(default) if default is not None else _load_yaml_resource(self.package, "configs", "default.yaml")
        cache_cfg = dget(self.default, "heavenbase.config.cache", {}) or {}
        configured_keep = dget(self.default, "heavenbase.config.versioning.keep_last_k", 10)
        configured_refresh = cache_cfg.get("refresh_interval") or dget(self.default, "heavenbase.config.refresh_interval", 60.0)
        self.keep_last_k = int(configured_keep if keep_last_k is None else keep_last_k)
        self.refresh_interval = float(configured_refresh if refresh_interval is None else refresh_interval)
        self.max_snapshots = _cache_max(cache_cfg.get("max_snapshots"), "heavenbase.config.cache.max_snapshots")
        self.max_generations = _cache_max(cache_cfg.get("max_generations"), "heavenbase.config.cache.max_generations")
        self._registry = registry
        self._registry_factory: Optional[Callable[[], Registry[str, Dict[str, Any]]]] = None
        self._opening_threads: set[int] = set()
        self._open_failed = False
        self._fallback_registry: Optional[Registry[str, Dict[str, Any]]] = None
        self._fallback_initial_publication = 0
        self._fallback_writers = 0
        self._last_write_persisted: contextvars.ContextVar[bool | None] = contextvars.ContextVar("last_config_write_persisted", default=None)
        self._lock = threading.RLock()
        self._snap_cache: Dict[Any, ConfigSnapshot] = {}
        self._gen_cache: Dict[Tuple[str, ...], _CachedGeneration] = {}
        self._cache_epoch = 0
        self._stamp_memo: Dict[str, Tuple[Any, int, _ScopeStamp]] = {}
        self._default_env_names = _env_names(self.default)
        self._setup_pending = bool(setup)

    @staticmethod
    def _scope_key(value: str) -> str:
        key = str(value).strip().lower()
        if not key:
            raise ValueError("Config scope cannot be empty")
        return key

    def _require_scope_authority(self) -> None:
        """Reject access to unowned process defaults in explicit Context mode."""
        if self is CM_HVNB and _explicit_context_required() and getattr(self, "_context", None) is None:
            # Explicit mode is set by the already loaded initialization module.
            sys.modules["heavenbase._initialization"].require_default_context_allowed()

    @property
    def _scope_registry(self) -> Registry[str, Dict[str, Any]]:
        """Return the current scope Registry, opening the store on first use."""
        self._require_scope_authority()
        fallback = self._fallback_registry
        if fallback is not None:
            return fallback
        try:
            return self._open_scope_registry()
        except Exception as error:
            fallback = self._use_process_config_after(error)
            if fallback is None:
                raise
            return fallback

    def _open_scope_registry(self) -> Registry[str, Dict[str, Any]]:
        """Open the configured scope Registry and apply deferred setup."""
        me = threading.get_ident()
        while True:
            with self._lock:
                registry = self._registry
                factory = self._registry_factory
                pending = self._setup_pending
                if registry is None and factory is None:
                    from heavenbase.utils.registry import Registry

                    registry = self._registry = Registry[str, Dict[str, Any]](f"{self.package}-config", normalize=self._scope_key)
            if registry is not None:
                if not pending or me in self._opening_threads:
                    return registry
                # Setup is an idempotent Registry mutation. Use the captured
                # Registry and keep the manager lock free during store I/O.
                self._opening_threads.add(me)
                try:
                    self._init_scope(self.base_scope, self.default, registry=registry)
                except Exception:
                    with self._lock:
                        if self._registry is registry:
                            raise
                    continue
                finally:
                    self._opening_threads.discard(me)
                with self._lock:
                    if self._registry is registry:
                        self._setup_pending = False
                        return registry
                continue
            if factory is None:
                continue
            # The factory opens a Context Registry under the Context lock, and a
            # Context-lock holder may read this manager, so it runs outside
            # ``self._lock``. Context factories memoize, so a racing call
            # returns the same Registry; a result from a replaced factory is dropped.
            self._opening_threads.add(me)
            try:
                registry = factory()
            finally:
                self._opening_threads.discard(me)
            with self._lock:
                if self._registry is None and self._registry_factory is factory:
                    self._registry = registry

    def _bind_registry(self, registry: Registry[str, Dict[str, Any]]) -> None:
        """Bind an unused manager to its Context-owned persistent Registry."""
        with self._lock:
            if self._registry is not None and len(self._registry):
                raise RuntimeError("Cannot bind ConfigManager after config state has been initialized")
            self._registry = registry
            self._fallback_registry = None
            self._open_failed = False
            self._clear_caches()

    def _bind_registry_factory(self, factory: Callable[[], Registry[str, Dict[str, Any]]], *, preserve_fallback: bool = False) -> None:
        """Bind a lazy Context registry factory before config initialization."""
        with self._lock:
            if self._registry is not None and len(self._registry):
                raise RuntimeError("Cannot bind ConfigManager after config state has been initialized")
            self._registry = None
            self._registry_factory = factory
            if not preserve_fallback:
                self._fallback_registry = None
            self._open_failed = self._fallback_registry is not None
            self._clear_caches()

    def scoped(self, name: Optional[str | Callable[[], Optional[str]]] = None) -> Any:
        """
        Return a scope context that also works as a decorator.

        Relative names extend the current scope chain. Names rooted at
        ``base_scope`` select an absolute chain and temporarily replace any
        ambient child scope.

        Args:
            name (Optional[str | Callable[[], Optional[str]]]): Relative or
                base-qualified scope name. ``None`` captures and restores the
                current scope context. Defaults to ``None``.

        Returns:
            Any: Scope context that also works as a decorator.
        """
        package = self.package
        is_snapshot = name is None
        snapshot = {k: v[:] for k, v in _SCOPE_STACKS.get().items()} if is_snapshot else None

        def resolve_name() -> tuple[List[str], bool] | None:
            value = name() if callable(name) else name
            if not value:
                return None
            value = value.strip().lower()
            absolute = value == self.base_scope or value.startswith(f"{self.base_scope}.")
            if absolute:
                value = value[len(self.base_scope) :].lstrip(".")
            return [part for part in value.split(".") if part], absolute

        class Scoped:
            def __init__(self):
                """Initialize one temporary config scope stack.

                Args:
                    None.

                Returns:
                    None: This function does not return a value.
                """
                self.tokens: List[Any] = []

            def __enter__(self):
                if is_snapshot and snapshot is not None:
                    self.tokens.append(_SCOPE_STACKS.set({k: v[:] for k, v in snapshot.items()}))
                    return self
                resolved = resolve_name()
                if resolved is None:
                    self.tokens.append(None)
                    return self
                parts, absolute = resolved
                current = {k: v[:] for k, v in _SCOPE_STACKS.get().items()}
                if absolute:
                    if parts:
                        current[package] = parts
                    else:
                        current.pop(package, None)
                elif parts:
                    current.setdefault(package, []).extend(parts)
                self.tokens.append(_SCOPE_STACKS.set(current))
                return self

            def __exit__(self, exc_type, exc, tb):
                token = self.tokens.pop() if self.tokens else None
                if token is not None:
                    _SCOPE_STACKS.reset(token)

            def __call__(self, obj):
                if inspect.iscoroutinefunction(obj):

                    @wraps(obj)
                    async def async_wrapper(*args, **kwargs):
                        with Scoped():
                            return await obj(*args, **kwargs)

                    return async_wrapper
                if callable(obj):

                    @wraps(obj)
                    def wrapper(*args, **kwargs):
                        with Scoped():
                            return obj(*args, **kwargs)

                    return wrapper
                return obj

        return Scoped()

    @property
    def scope_chain(self) -> List[str]:
        """Use ``scope_chain`` in the HeavenBase utility surface.

        Use this helper for nested config dictionaries and scoped runtime configuration layers.

        Args:
            None.

        Returns:
            List[str]: The result of ``scope_chain``.
        """
        parts = [self.base_scope, *_SCOPE_STACKS.get().get(self.package, [])]
        out = []
        cur = ""
        for part in parts:
            cur = part if not cur else f"{cur}.{part}"
            out.append(cur)
        return out

    @property
    def scope(self) -> str:
        """Use ``scope`` in the HeavenBase utility surface.

        Use this helper for nested config dictionaries and scoped runtime configuration layers.

        Args:
            None.

        Returns:
            str: The result of ``scope``.
        """
        return self.scope_chain[-1]

    @property
    def package_version(self) -> Optional[str]:
        """Return the installed package version when it can be resolved.

        Use this helper for nested config dictionaries and scoped runtime configuration layers.

        Args:
            None.

        Returns:
            Optional[str]: The the installed package version when it can be resolved.
        """
        try:
            from importlib import metadata

            return str(metadata.version(self.package))
        except Exception:
            try:
                import importlib

                value = getattr(importlib.import_module(self.package), "__version__", None)
                return str(value) if value is not None else None
            except Exception:
                return None

    @staticmethod
    def _empty_record(scope: str) -> Dict[str, Any]:
        return {"scope": scope, "generation": 0, "versions": []}

    def _record(self, scope: str) -> Dict[str, Any]:
        key = self._scope_key(scope)
        return copy.deepcopy(self._scope_registry.get(key) or self._empty_record(key))

    @staticmethod
    def _versions(record: Mapping[str, Any]) -> List[Dict[str, Any]]:
        values = record.get("versions") or []
        if not isinstance(values, list):
            raise TypeError("Config scope record versions must be a list")
        return [copy.deepcopy(dict(value)) for value in values]

    def _scope_version(self, scope: str) -> int:
        versions = self._versions(self._record(scope))
        return max((int(item.get("version") or 0) for item in versions), default=0)

    def _chain_state(
        self, chain: Tuple[str, ...], *, reload: bool
    ) -> Tuple[Tuple[int, ...], Tuple[_ScopeStamp, ...], List[Dict[str, Any]], Registry[str, Dict[str, Any]], int]:
        """Return the chain's generations, stamps, latest layers, and source publication.

        A stamp digests a scope's generation and latest layer, which is all a
        snapshot reads. A reset or removal deletes a record, so its generation
        can repeat with other content; snapshot caches key on the stamp
        instead. Every scope is read from the same published state, so a
        snapshot's stamps always describe exactly the layers it was built from.
        """
        registry = self._scope_registry
        if reload:
            registry.refresh()
        with self._lock:
            epoch = self._cache_epoch
        while True:
            publication = registry.publication
            records = [registry.get(self._scope_key(scope)) for scope in chain]
            if registry.publication == publication:
                break
        with self._lock:
            memo = self._memoized_stamps(chain, registry, publication)
        gens, stamps, layers = [], [], []
        for record, cached_stamp in zip(records, memo):
            generation = int((record or {}).get("generation") or 0)
            _, layer = self._latest_of(record)
            gens.append(generation)
            stamps.append(cached_stamp if cached_stamp is not None else _ScopeStamp(_stamp(generation, layer), _env_names(layer)))
            layers.append(layer)
        if self.max_generations:
            with self._lock:
                if epoch == self._cache_epoch and registry is self._registry and publication == registry.publication:
                    for scope, stamp in zip(chain, stamps):
                        self._stamp_memo.pop(scope, None)
                        self._stamp_memo[scope] = (registry, publication, stamp)
                    while len(self._stamp_memo) > self.max_generations:
                        self._stamp_memo.pop(next(iter(self._stamp_memo)))
        return tuple(gens), tuple(stamps), layers, registry, publication

    def _memoized_stamps(self, chain: Tuple[str, ...], registry: Any, publication: int) -> Tuple[Optional[_ScopeStamp], ...]:
        """Read matching scope stamps and update their recency under the manager lock."""
        stamps: List[Optional[_ScopeStamp]] = []
        for scope in chain:
            entry = self._stamp_memo.get(scope) if self.max_generations else None
            if entry is not None and entry[0] is registry and entry[1] == publication:
                self._stamp_memo[scope] = self._stamp_memo.pop(scope)
                stamps.append(entry[2])
            else:
                stamps.append(None)
        return tuple(stamps)

    def _memory_snapshot(self, chain: Tuple[str, ...]) -> "_Known":
        """Return the resolved snapshot of ``chain`` in the records this process holds, with where it was read.

        Reads no store and is cached like :meth:`snapshot`. The stamps of the
        records a Registry instance holds at one publication never change, so
        once a read has computed them the records are not copied again until
        the Registry publishes. A write that reads resolved values takes this
        before it starts, so it resolves nothing under the Registry lock when
        its draft holds these same records.
        """
        policy = InterpolationPolicy()
        registry = self._scope_registry
        publication = registry.publication
        with self._lock:
            epoch = self._cache_epoch
            memo = self._memoized_stamps(chain, registry, publication)
        cached_stamps = tuple(stamp for stamp in memo if stamp is not None)
        if len(cached_stamps) == len(chain):
            key, _ = self._snapshot_context(chain, cached_stamps, None, policy)
            cached = self._cached_snapshot(key, epoch)
            if cached is not None:
                return _Known(registry, publication, tuple(stamp.digest for stamp in cached_stamps), cached)
        while True:
            gen, stamps, layers, registry, publication = self._chain_state(chain, reload=False)
            if registry.publication == publication:
                break
        key, environment = self._snapshot_context(chain, stamps, None, policy)
        digests = tuple(stamp.digest for stamp in stamps)
        cached = self._cached_snapshot(key, epoch)
        if cached is not None:
            return _Known(registry, publication, digests, cached)
        snap = self._resolved(layers, gen, policy, environment=environment)
        with self._lock:
            if epoch == self._cache_epoch:
                self._cache_snapshot(key, snap)
        return _Known(registry, publication, digests, snap)

    def _latest_layer(self, scope: str) -> Tuple[int, Dict[str, Any]]:
        return self._latest_of(self._record(scope))

    @staticmethod
    def _latest_of(record: Optional[Mapping[str, Any]]) -> Tuple[int, Dict[str, Any]]:
        """Return the latest version number and a copy of its data, copying no other version."""
        versions = (record or {}).get("versions") or []
        if not isinstance(versions, list):
            raise TypeError("Config scope record versions must be a list")
        if not versions:
            return 0, {}
        latest = max(versions, key=lambda item: int(item.get("version") or 0))
        return int(latest.get("version") or 0), copy.deepcopy(dict(latest.get("data") or {}))

    def _resolved(
        self,
        layers: List[Dict[str, Any]],
        gen: Tuple[int, ...],
        policy: InterpolationPolicy,
        overrides: Optional[Dict[str, Any]] = None,
        *,
        environment: _Environment | None = None,
    ) -> ConfigSnapshot:
        """Merge the packaged defaults, ``layers`` and ``overrides`` and resolve their interpolations."""
        stack = [copy.deepcopy(self.default), *layers]
        if overrides:
            stack.append(overrides)
        return ConfigSnapshot(_resolve_interpolation(_merge_layers(stack), policy, environment), gen, policy)

    def _draft_view(
        self,
        draft: Registry[str, Dict[str, Any]],
        chain: Tuple[str, ...],
        known: Optional["_Known"] = None,
        stored: Optional[Tuple[str, ...]] = None,
    ) -> "_DraftView":
        """Return the view of ``chain`` in a :meth:`_commit` draft, the revision the write reads.

        ``known`` comes from :meth:`_memory_snapshot`; the view uses its
        snapshot when the draft holds the same records. ``stored`` is the
        chain of the written scope, whose stored values list writes extend;
        it defaults to ``chain``.
        """
        return _DraftView(self, draft, chain, known, stored)

    def _chain_of(self, scope: str) -> Tuple[str, ...]:
        """Return the scope chain whose reads see ``scope`` last: the base scope and each scope down to ``scope``.

        A scope outside the base scope is read by no chain, so its chain is
        the scope alone.
        """
        target = self._scope_key(scope)
        base = self.base_scope
        if not target.startswith(f"{base}."):
            return (base,) if target == base else (target,)
        out = [base]
        for part in target[len(base) + 1 :].split("."):
            out.append(f"{out[-1]}.{part}")
        return tuple(out)

    def _saved_record(
        self,
        scope: str,
        data: Dict[str, Any],
        *,
        package_version: Optional[str],
        keep_last_k: int,
    ) -> Callable[[Optional[Dict[str, Any]]], Dict[str, Any]]:
        def save(current: Optional[Dict[str, Any]]) -> Dict[str, Any]:
            record = copy.deepcopy(current or self._empty_record(scope))
            versions = self._versions(record)
            version = max((int(item.get("version") or 0) for item in versions), default=0) + 1
            generation = int(record.get("generation") or 0) + 1
            versions.append(
                {
                    "version": version,
                    "package_version": package_version,
                    "data": copy.deepcopy(data or {}),
                    "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                }
            )
            record.update(scope=scope, generation=generation, versions=versions[-keep_last_k:])
            return record

        return save

    def _commit(
        self,
        change: Callable[[Registry[str, Dict[str, Any]]], Tuple[_R, bool]],
        *,
        registry: Optional[Registry[str, Dict[str, Any]]] = None,
    ) -> _R:
        """Run one read-modify-write against the latest persisted scope records.

        :meth:`Registry.mutate` applies ``change`` to an isolated draft of the
        Registry's current revision, so ``change`` reads exactly the revision
        it publishes against, and writes of one process run one at a time under
        the Registry's lock. ``change`` returns its result and whether it
        staged a write.

        A staged write is published with one compare-and-set. When the draft's
        revision is no longer the latest, because another manager or process
        published since this one last refreshed, the compare-and-set fails;
        the write then backs off, refreshes and runs again on the newer
        records, so it never overwrites a value published since its read.
        An up-to-date write therefore reads the store only to publish.

        A change that stages nothing publishes nothing, so no compare-and-set
        checks the revision it decided on. The write then refreshes, and runs
        again on the newer records when the store has moved past that
        revision, so a write is never dropped because this process had not yet
        seen another one. A change that raises, such as a list index past the
        end of the list it read, is treated the same way: it is raised only
        when the revision it read is still the latest, so a write never fails
        because this process had not yet seen the records it needs.

        Writes of a Context's ConfigManager (``CM_HVNB`` and ``Context.config``)
        to its persistent Registry hold the Context home's OS file lock
        (:func:`_home_write_lock`) for the whole loop, so writers of one home
        run one after another and a compare-and-set loses only to a writer
        that does not hold the lock, such as another host writing the same
        store or a writer that stopped waiting for it. A write still losing
        after ``_COMMIT_DEADLINE`` seconds raises the last conflict or error;
        the time spent waiting for the lock does not count.

        The Registry is opened before the write starts and no manager lock is
        held, so a first open, which takes the owning Context's lock, never
        waits inside a write while a Context-lock holder waits for that write.
        """
        if registry is None:
            registry = self._scope_registry
        if not registry.persistent:
            return self._commit_loop(change, registry)
        # Only a Context's home holds the lock file; ``root`` of a manager
        # without a Context may be a project or resource directory.
        context = getattr(self, "_context", None)
        with _home_write_lock(getattr(context, "root", None)):
            return self._commit_loop(change, registry)

    def _commit_loop(self, change: Callable[[Registry[str, Dict[str, Any]]], Tuple[_R, bool]], registry: Registry[str, Dict[str, Any]]) -> _R:
        """Run :meth:`_commit`'s compare-and-set loop on an opened Registry."""
        from heavenbase.utils.registry import RegistryConflictError

        deadline = time.monotonic() + _COMMIT_DEADLINE
        attempt = 0
        while True:
            read: List[int] = []
            failed: List[BaseException] = []

            def run(draft: Registry[str, Dict[str, Any]]) -> Tuple[_R, bool]:
                read.append(draft.revision)
                try:
                    return change(draft)
                except Exception as error:
                    failed.append(error)
                    raise

            try:
                result, staged = registry.mutate(run)
            except RegistryConflictError:
                pause = random.uniform(0, min(_COMMIT_BACKOFF_MAX, 0.001 * 2 ** min(attempt, 16)))
                if time.monotonic() + pause > deadline:
                    raise
                attempt += 1
                time.sleep(pause)
                registry.refresh()
                continue
            except Exception as error:
                # Only an error of ``change`` itself may come from a stale read;
                # an error opening or publishing the store is raised as it is.
                if not failed or failed[-1] is not error:
                    raise
                registry.refresh()
                if registry.revision == read[0] or time.monotonic() > deadline:
                    raise
                continue
            if staged:
                return result
            registry.refresh()
            if registry.revision == read[0]:
                return result
            if time.monotonic() > deadline:
                raise RegistryConflictError(f"Config scopes kept changing for {_COMMIT_DEADLINE:g} seconds while a write that changes nothing checked them")

    def _commit_scope(self, scope: str, change: Callable[[Registry[str, Dict[str, Any]]], Tuple[_R, bool]]) -> _R:
        """Publish a scoped write and revalidate cached reads that depend on it."""
        with self._mutation_registry() as registry:
            publication = registry.publication
            result = self._commit(change, registry=registry)
            if registry.publication != publication:
                self._clear_caches(scope)
            else:
                # An older publication can have changes anywhere in the chain.
                # Advance the read epoch even when the cache is still empty.
                with self._lock:
                    self._cache_epoch += 1
                    self._gen_cache = {
                        chain: cached
                        for chain, cached in self._gen_cache.items()
                        if scope not in chain or (cached.registry is registry and cached.publication == publication)
                    }
            return result

    @contextlib.contextmanager
    def _mutation_registry(self) -> Iterator[Registry[str, Dict[str, Any]]]:
        """Keep a process fallback in use for the duration of one write."""
        while True:
            registry = self._scope_registry
            with self._lock:
                fallback = registry is self._fallback_registry
                if fallback:
                    self._fallback_writers += 1
                elif not registry.persistent and registry is not self._registry:
                    continue
            try:
                persists = self._registry_persists(registry)
                yield registry
                self._last_write_persisted.set(persists)
            finally:
                if fallback:
                    with self._lock:
                        self._fallback_writers -= 1
            return

    def _registry_persists(self, registry: Registry[str, Dict[str, Any]]) -> bool:
        context = getattr(self, "_context", None)
        return registry.persistent and (context is None or context._config_store_persists(registry))

    @property
    def last_write_persisted(self) -> bool | None:
        """Report the destination of the last completed mutation in this execution context.

        ``True`` means the operation was applied or already satisfied in the
        configured Registry store. ``False`` means the operation used
        process-only storage. ``None`` means the execution context has no prior status.
        Reads and failures leave this status unchanged. A custom Registry
        store controls its own durability. The write method's return value
        continues to report its individual operation result.
        """
        return self._last_write_persisted.get()

    def _stage_layer(
        self,
        draft: Registry[str, Dict[str, Any]],
        scope: str,
        data: Dict[str, Any],
        *,
        package_version: Optional[str] = None,
        keep_last_k: Optional[int] = None,
    ) -> None:
        """Append ``data`` to the scope record in a :meth:`_commit` draft as its next version."""
        keep = max(1, int(keep_last_k or self.keep_last_k))
        draft.update(scope, derive=self._saved_record(scope, data, package_version=package_version, keep_last_k=keep))

    def _write_layer(self, scope: str, edit: Callable[[Dict[str, Any], _DraftView], Optional[bool]], *, resolves: bool = False) -> bool:
        """Save ``edit`` applied to the scope's latest persisted layer as its next version.

        ``edit`` receives the layer and a view of the revision it was read
        from: the resolved config of the current scope chain, and the stored,
        unresolved values of the chain that reads ``scope``. A write whose
        reads went stale runs again on the newer records. An edit that returns
        ``False`` saves nothing, and so does one that leaves an existing layer
        unchanged: neither publishes, and :meth:`_commit` runs it again when
        the store has moved past the revision it read. An ``edit`` that reads resolved
        values passes ``resolves``, so the write resolves the config this
        process holds before it starts and resolves again only when its draft
        differs. Returns whether ``edit`` asked for the write.
        """
        target = self._scope_key(scope)
        chain = tuple(self.scope_chain)
        stored = self._chain_of(target)
        package_version = self.package_version
        known = self._memory_snapshot(chain) if resolves else None

        def change(draft: Registry[str, Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], bool]:
            view = self._draft_view(draft, chain, known, stored)
            record = draft.get(target)
            _, layer = self._latest_of(record)
            before = to_mutable_data(layer) if (record or {}).get("versions") else None
            if edit(layer, view) is False:
                return None, False
            layer = to_mutable_data(layer)
            if before is not None and _same_data(before, layer):
                return layer, False
            self._stage_layer(draft, target, layer, package_version=package_version, keep_last_k=self._keep_last_k(view))
            return layer, True

        layer = self._commit_scope(target, change)
        return layer is not None

    def _init_scope(self, scope: str, data: Dict[str, Any], *, registry: Optional[Registry[str, Dict[str, Any]]] = None) -> None:
        if registry is None:
            registry = self._scope_registry
        target = self._scope_key(scope)
        if self._versions(registry.get(target) or {}):
            return
        data = to_mutable_data(data)

        def change(draft: Registry[str, Dict[str, Any]]) -> Tuple[None, bool]:
            # Another writer may have seeded the scope since the check above.
            if self._versions(draft.get(target) or {}):
                return None, False
            self._stage_layer(draft, target, data)
            return None, True

        self._commit(change, registry=registry)

    def _history(self, scope: str, limit: int) -> List[Dict[str, Any]]:
        record = self._record(scope)
        generation = int(record.get("generation") or 0)
        versions = sorted(self._versions(record), key=lambda item: int(item.get("version") or 0), reverse=True)
        return [
            {
                "pkg": self.package,
                "scope": scope,
                "ver": int(item.get("version") or 0),
                "pkg_ver": item.get("package_version"),
                "gen": generation,
                "data": copy.deepcopy(item.get("data") or {}),
                "created_at": item.get("created_at"),
            }
            for item in versions[: int(limit)]
        ]

    @property
    def version_chain(self) -> List[int]:
        """Return latest retained versions for the active scope chain.

        Use this helper for nested config dictionaries and scoped runtime configuration layers.

        Args:
            None.

        Returns:
            List[int]: The latest retained versions for the active scope chain.
        """
        return [self._scope_version(scope) for scope in self.scope_chain]

    @property
    def version(self) -> int:
        """Return the latest retained version for the current scope.

        Use this helper for nested config dictionaries and scoped runtime configuration layers.

        Args:
            None.

        Returns:
            int: The the latest retained version for the current scope.
        """
        return self.version_chain[-1]

    def _generation_state(
        self, chain: Tuple[str, ...], force: bool = False
    ) -> Tuple[Tuple[int, ...], Tuple[_ScopeStamp, ...], int, Optional[List[Dict[str, Any]]]]:
        """Return the chain's generations and stamps, cached for ``refresh_interval``.

        The fourth item holds the layers the stamps were computed from when
        this call read them, or ``None`` when the stamps came from the cache.
        """
        while True:
            now = time.monotonic()
            with self._lock:
                epoch = self._cache_epoch
                cached = self._gen_cache.get(chain) if self.max_generations else None
                if cached and not force and now - cached.checked_at < self.refresh_interval:
                    self._gen_cache[chain] = self._gen_cache.pop(chain)
                    return cached.generations, cached.stamps, epoch, None
            try:
                gens, stamps, layers, registry, publication = self._chain_state(chain, reload=True)
            except Exception:
                with self._lock:
                    if epoch == self._cache_epoch:
                        raise
                continue
            with self._lock:
                if epoch != self._cache_epoch:
                    continue
                if self.max_generations:
                    self._gen_cache[chain] = _CachedGeneration(now, gens, stamps, registry, publication)
                    while len(self._gen_cache) > self.max_generations:
                        self._gen_cache.pop(next(iter(self._gen_cache)))
                return gens, stamps, epoch, layers

    def snapshot(
        self,
        overrides: Optional[Dict[str, Any]] = None,
        policy: Optional[InterpolationPolicy] = None,
        refresh: bool = False,
    ) -> ConfigSnapshot:
        """
        Return an immutable resolved snapshot for the current scope chain.

        Use this helper for nested config dictionaries and scoped runtime configuration layers.
        If ``CM_HVNB`` cannot first open its home, a process-only Registry
        supplies packaged defaults and local edits. Pass ``refresh=True`` to
        retry the open and reload saved settings.

        Args:
            overrides (Optional[Dict[str, Any]]): Optional final config layer merged after stored layers. Defaults to ``None``.
            policy (Optional[InterpolationPolicy]): Interpolation policy. ``None`` uses the default policy. Defaults to ``None``.
            refresh (bool): Whether to bypass cached generations and read fresh store state. Defaults to ``False``.

        Returns:
            ConfigSnapshot: The an immutable resolved snapshot for the current scope chain.
        """
        self._require_scope_authority()
        policy = policy or InterpolationPolicy()
        chain = tuple(self.scope_chain)
        if refresh and self._fallback_registry is not None:
            self.refresh()
        while True:
            gen, stamps, epoch, layers = self._generation_state(chain, force=refresh)
            key, environment = self._snapshot_context(chain, stamps, overrides, policy)
            cached = self._cached_snapshot(key, epoch)
            if cached is not None:
                return cached
            if layers is None:
                # The stamps came from the cache and may describe an older
                # revision than the in-memory records; build from records read
                # now and key the snapshot on their own stamps, so a snapshot is
                # never cached under stamps of other content.
                gen, stamps, layers, _, _ = self._chain_state(chain, reload=False)
                key, environment = self._snapshot_context(chain, stamps, overrides, policy)
                cached = self._cached_snapshot(key, epoch)
                if cached is not None:
                    return cached
            snap = self._resolved(layers, gen, policy, overrides, environment=environment)
            with self._lock:
                if epoch != self._cache_epoch:
                    continue
                self._cache_snapshot(key, snap)
                return snap

    def _use_process_config_after(self, error: Exception) -> Optional[Registry[str, Dict[str, Any]]]:
        """Open a process-only Registry after an unavailable initial global open."""
        if self is not CM_HVNB or _explicit_context_required() or not _cannot_open(error):
            return None
        from heavenbase.utils.registry import Registry

        with self._lock:
            if self._fallback_registry is not None:
                return self._fallback_registry
            if self._registry is not None and not self._setup_pending:
                return None
            fallback = Registry[str, Dict[str, Any]](f"{self.package}-config", normalize=self._scope_key)
            self._init_scope(self.base_scope, self.default, registry=fallback)
            self._fallback_registry = fallback
            self._fallback_initial_publication = fallback.publication
            self._open_failed = True
            self._clear_caches()
        warnings.warn(
            "HeavenBase could not open saved configuration. Using packaged defaults and keeping edits in this process only.",
            RuntimeWarning,
            stacklevel=3,
        )
        return fallback

    def _snapshot_context(
        self, chain: Tuple[str, ...], stamps: Tuple[_ScopeStamp, ...], overrides: Optional[Dict[str, Any]], policy: InterpolationPolicy
    ) -> Tuple[Any, _Environment | None]:
        """Key a snapshot on the same environment values its resolvers will use."""
        if not self.max_snapshots:
            return None, None
        environment_names = self._default_env_names.union(_env_names(overrides), *(stamp.environment_names for stamp in stamps))
        environment = _capture_env(environment_names)
        if environment is None:
            return None, None
        key = (chain, tuple(stamp.digest for stamp in stamps), _identity(overrides), policy.digest(), environment.digest)
        return key, environment

    def _cached_snapshot(self, key: Any, epoch: int) -> Optional[ConfigSnapshot]:
        """Return the snapshot cached under ``key``, or ``None`` on a miss or a cleared cache."""
        with self._lock:
            if epoch != self._cache_epoch or key is None or not self.max_snapshots:
                return None
            cached = self._snap_cache.get(key)
            if cached is not None:
                self._snap_cache[key] = self._snap_cache.pop(key)
            return cached

    def _cache_snapshot(self, key: Any, snap: ConfigSnapshot) -> None:
        """Publish a cacheable snapshot under the manager lock."""
        if key is None or not self.max_snapshots:
            return
        self._snap_cache[key] = snap
        while len(self._snap_cache) > self.max_snapshots:
            self._snap_cache.pop(next(iter(self._snap_cache)))

    def get(self, path: Optional[str] = None, default: Any = None) -> Any:
        """
        Return one value from the current resolved config.

        Use this helper for nested config dictionaries and scoped runtime configuration layers.

        Args:
            path (Optional[str]): Local path, config path, or dotted dictionary path used by the helper. Defaults to ``None``.
            default (Any): Fallback value used when a path or config key is missing. Defaults to ``None``.

        Returns:
            Any: The one value from the current resolved config.
        """
        if threading.get_ident() in self._opening_threads:
            self._require_scope_authority()
            return copy.deepcopy(dget(self.default, path, default)) if path else copy.deepcopy(self.default)
        return self.snapshot().get(path, default=default)

    def _get_or_packaged(self, path: str, default: Any = None) -> Any:
        """Return :meth:`get`, or the packaged value when the saved config cannot be opened.

        For utilities that must not need the saved config, such as log records,
        ``hash_id``, ``StableRNG`` and the serialization and parallel helpers.
        HeavenBase's own Context-scoped code calls them too, so in explicit
        Context mode they must not need the process default. Only an open that
        fails for a lasting cause (an unwritable home, an invalid bootstrap file
        or explicit Context mode) falls back; the packaged values are then read
        without retrying until a later call opens the saved config. For
        ``CM_HVNB``, ``refresh()`` retries the open explicitly. Any other
        error is raised and not remembered.
        """
        if self._open_failed and self._registry is None and self._fallback_registry is None:
            return copy.deepcopy(dget(self.default, path, default))
        try:
            return self.get(path, default=default)
        except Exception as error:
            if self._registry is not None or not _cannot_open(error):
                raise
            self._open_failed = True
            return copy.deepcopy(dget(self.default, path, default))

    def _set_if_openable(self, path: str, value: Any) -> bool:
        """Run :meth:`set`, or return ``False`` when the saved config cannot be opened.

        For the global log helpers, which keep a setting for the current
        process when it cannot be saved. Like :meth:`_get_or_packaged`, only a
        lasting cause returns ``False``, and the open is not retried until a
        later call opens the saved config. Every other error is raised.
        """
        if self._fallback_registry is not None or (self._open_failed and self._registry is None):
            return False
        try:
            self.set(path, value)
        except Exception as error:
            if self._registry is not None or not _cannot_open(error):
                raise
            self._open_failed = True
            return False
        return self is not CM_HVNB or self.last_write_persisted is True

    def load(self, refresh: bool = False) -> Dict[str, Any]:
        """
        Return the current merged config as a mutable dictionary.

        Use this helper for nested config dictionaries and scoped runtime configuration layers.

        Args:
            refresh (bool): Whether to bypass cached generations and read fresh store state. Defaults to ``False``.

        Returns:
            Dict[str, Any]: The the current merged config as a mutable dictionary.
        """
        return self.snapshot(refresh=refresh).to_dict()

    def layer(self, scope: Optional[str] = None, version: Optional[int] = None) -> Dict[str, Any]:
        """
        Return one raw layer before parent merge and interpolation.

        Use this helper for nested config dictionaries and scoped runtime configuration layers.

        Args:
            scope (Optional[str]): Config scope. ``None`` uses the current active scope. Defaults to ``None``.
            version (Optional[int]): Optional retained config version. ``None`` uses the latest layer. Defaults to ``None``.

        Returns:
            Dict[str, Any]: The one raw layer before parent merge and interpolation.
        """
        target = (scope or self.scope).strip().lower()
        if version is None:
            _, data = self._latest_layer(target)
            return data
        for item in self._versions(self._record(target)):
            if int(item.get("version") or 0) == int(version):
                return copy.deepcopy(item.get("data") or {})
        return {}

    def set(self, path: Optional[str], value: Any = None, scope: Optional[str] = None) -> bool:
        """
        Set a value in the current or explicit scope.

        A list path (``items[]`` appends, ``items[i]`` replaces and
        ``items[i+]`` inserts) edits the list the written scope's chain
        reads and saves the edited list in that scope. Its items are saved as
        stored, so an interpolation such as ``${env:NAME}`` stays an
        interpolation and its value is not saved. A list path at a value that
        is not a list raises ``TypeError``; a missing or ``None`` value starts
        a new list. A path that goes through a list or through a node held by
        an interpolation, such as ``servers[0].port`` or ``y.port`` under
        ``y: ${src}``, first saves that whole list or node in the scope, its
        plain node interpolations followed to the stored nodes they name, so
        the write keeps the node's other items and keys. Such a write raises
        ``ValueError`` when it goes through a mapping or list that another
        kind of interpolation yields, such as a resolver, since only its
        resolved values could be saved, or when no copy of the node it
        follows can keep the value of every interpolation in it.

        Use this helper for nested config dictionaries and scoped runtime configuration layers.

        Args:
            path (Optional[str]): Local path, config path, or dotted dictionary path used by the helper.
            value (Any): Value to read, set, validate, hash, or format. Defaults to ``None``.
            scope (Optional[str]): Config scope. ``None`` uses the current active scope. Defaults to ``None``.

        Returns:
            bool: The result of ``set``.
        """
        return self._write_layer((scope or self.scope).strip().lower(), self._assign(path, value))

    @staticmethod
    def _assign(path: Optional[str], value: Any) -> Callable[[Dict[str, Any], _DraftView], None]:
        """Return the :meth:`_write_layer` edit that stores ``value`` at ``path``."""
        list_tail = _list_tail(path) if path else None

        def edit(layer: Dict[str, Any], view: _DraftView) -> None:
            if list_tail:
                parent, op = list_tail
                through = view.through(parent, into_last=True)
                _keep(layer, through.held)
                values = _as_list(through.value, parent)
                if op == "":
                    values.append(value)
                elif op.endswith("+"):
                    values.insert(int(op[:-1]), value)
                else:
                    values[int(op)] = value
                _clear_tombstone_ancestors(layer, parent)
                dset(layer, parent, values)
            else:
                if path:
                    if view.crosses(path):
                        _keep(layer, view.through(path, into_last=False).held)
                    _clear_tombstone_ancestors(layer, path)
                dset(layer, path, value)

        return edit

    def unset(self, path: str, scope: Optional[str] = None) -> bool:
        """
        Hide a dict key or remove a list item in the current or explicit scope.

        A list path (``items[i]``) removes the item from the list the written
        scope's chain reads and saves the rest in that scope as stored, like
        a list path of :meth:`set`. A path that goes through a list or through
        a node held by an interpolation saves that node without the key, like
        :meth:`set`, and saves nothing when the key is missing. It raises
        ``ValueError`` where :meth:`set` does.

        Use this helper for nested config dictionaries and scoped runtime configuration layers.

        Args:
            path (str): Local path, config path, or dotted dictionary path used by the helper.
            scope (Optional[str]): Config scope. ``None`` uses the current active scope. Defaults to ``None``.

        Returns:
            bool: The result of ``unset``.
        """
        list_tail = _list_tail(path)
        if list_tail and (list_tail[1] == "" or list_tail[1].endswith("+")):
            raise ValueError(f"unset does not support list op {path!r}")

        def edit(layer: Dict[str, Any], view: _DraftView) -> Optional[bool]:
            if list_tail:
                parent, op = list_tail
                through = view.through(parent, into_last=True)
                _keep(layer, through.held)
                values = _as_list(through.value, parent)
                del values[int(op)]
                dset(layer, parent, values)
                return None
            if view.crosses(path):
                through = view.through(path, into_last=False)
                if through.crossed:
                    if through.value is _MISSING_LIST:
                        return False
                    _keep(layer, through.held)
                    dunset(layer, path)
                    return None
            _tombstone_unset(layer, path)
            return None

        return self._write_layer((scope or self.scope).strip().lower(), edit)

    def setdef(self, path: str, default: Any = None, scope: Optional[str] = None) -> bool:
        """
        Set a value only when the current resolved value is missing or None.

        The value is checked in the latest saved revision, so a value another
        process saves first is kept rather than overwritten, and a value
        another process removes is set again.

        Use this helper for nested config dictionaries and scoped runtime configuration layers.

        Args:
            path (str): Local path, config path, or dotted dictionary path used by the helper.
            default (Any): Fallback value used when a path or config key is missing. Defaults to ``None``.
            scope (Optional[str]): Config scope. ``None`` uses the current active scope. Defaults to ``None``.

        Returns:
            bool: Whether the value was missing and ``default`` was set.
        """
        assign = self._assign(path, default)

        def edit(layer: Dict[str, Any], view: _DraftView) -> Optional[bool]:
            if view.get(path, default=None) is not None:
                return False
            assign(layer, view)
            return None

        return self._write_layer((scope or self.scope).strip().lower(), edit, resolves=True)

    def _keep_last_k(self, view: Optional[_DraftView] = None) -> int:
        """Return the retained-version count that ``view``, or the current config, sets."""
        path = "heavenbase.config.versioning.keep_last_k"
        value = view.setting(path, self.keep_last_k) if view is not None else self.get(path, default=self.keep_last_k)
        return int(value or self.keep_last_k)

    def history(self, scope: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Return retained version history for a scope.

        Use this helper for nested config dictionaries and scoped runtime configuration layers.

        Args:
            scope (Optional[str]): Config scope. ``None`` uses the current active scope. Defaults to ``None``.
            limit (int): Maximum number of items, rows, or rendered entries. Defaults vary by helper. Defaults to ``10``.

        Returns:
            List[Dict[str, Any]]: The retained version history for a scope.
        """
        return self._history(self._scope_key(scope or self.scope), limit)

    def scopes(self) -> List[str]:
        """Return all stored scopes for this package.

        Use this helper for nested config dictionaries and scoped runtime configuration layers.

        Args:
            None.

        Returns:
            List[str]: The all stored scopes for this package.
        """
        return list(self._scope_registry.list())

    def remove(self, scope: Optional[str] = None, version: Optional[int] = None) -> bool:
        """
        Remove a scope or one retained version from the config store.

        Use this helper for nested config dictionaries and scoped runtime configuration layers.

        Args:
            scope (Optional[str]): Config scope. ``None`` uses the current active scope. Defaults to ``None``.
            version (Optional[int]): Optional retained config version. ``None`` uses the latest layer. Defaults to ``None``.

        Returns:
            bool: The result of ``remove``.
        """
        target = (scope or self.scope).strip().lower()

        def change(draft: Registry[str, Dict[str, Any]]) -> Tuple[bool, bool]:
            if version is None:
                removed = draft.delete(target, missing_ok=True) is not None
                return removed, removed
            record = draft.get(target)
            if record is None:
                return False, False
            versions = self._versions(record)
            kept = [item for item in versions if int(item.get("version") or 0) != int(version)]
            if len(kept) == len(versions):
                return False, False
            if kept:
                next_record = copy.deepcopy(record)
                next_record["generation"] = int(next_record.get("generation") or 0) + 1
                next_record["versions"] = kept
                draft.update(target, next_record)
            else:
                draft.delete(target)
            return True, True

        return self._commit_scope(target, change)

    def compact(self, scope: Optional[str] = None, keep_last_k: Optional[int] = None, reset: bool = False) -> int:
        """
        Compact retained version history for a scope.

        Use this helper for nested config dictionaries and scoped runtime configuration layers.

        Args:
            scope (Optional[str]): Config scope. ``None`` uses the current active scope. Defaults to ``None``.
            keep_last_k (Optional[int]): Optional retained-version limit. ``None`` uses the configured default.
            reset (bool): Whether to clear existing state before initializing or compacting. Defaults to ``False``.

        Returns:
            int: The result of ``compact``.
        """
        target = (scope or self.scope).strip().lower()
        keep = max(1, int(keep_last_k or self.keep_last_k))

        def change(draft: Registry[str, Dict[str, Any]]) -> Tuple[int, bool]:
            record = draft.get(target)
            if record is None:
                return 0, False
            versions = sorted(self._versions(record), key=lambda item: int(item.get("version") or 0), reverse=True)
            if reset:
                latest = copy.deepcopy(versions[0])
                latest["version"] = 1
                kept = [latest]
            else:
                kept = list(reversed(versions[:keep]))
            removed = len(versions) - len(kept)
            if removed:
                next_record = copy.deepcopy(record)
                next_record["generation"] = int(next_record.get("generation") or 0) + 1
                next_record["versions"] = kept
                draft.update(target, next_record)
            return removed, bool(removed)

        return self._commit_scope(target, change)

    def _clear_caches(self, scope: Optional[str] = None) -> None:
        """Invalidate all derived snapshots, or only chains containing a scope."""
        with self._lock:
            self._cache_epoch += 1
            if scope is None:
                self._snap_cache.clear()
                self._gen_cache.clear()
                self._stamp_memo.clear()
                return
            target = self._scope_key(scope)
            self._stamp_memo.pop(target, None)
            self._snap_cache = {key: value for key, value in self._snap_cache.items() if target not in key[0]}
            self._gen_cache = {chain: value for chain, value in self._gen_cache.items() if target not in chain}

    def refresh(self, *, discard_process_changes: bool = False) -> bool:
        """Refresh Registry state and clear derived config caches.

        For ``CM_HVNB``, retry opening the home after a process-only fallback.
        A persistent store can replace the temporary state once the caller
        has agreed to discard any local edits.

        Args:
            discard_process_changes: Adopt the saved configuration even when
                the fallback Registry holds process-only edits. Defaults to False.

        Returns:
            bool: Whether a newer persistent Registry snapshot was adopted.
                Process-only managers always return `False`; their derived
                caches are still cleared.
        """
        self._require_scope_authority()
        try:
            registry = self._open_scope_registry()
            changed = registry.refresh()
        except Exception as error:
            if self._use_process_config_after(error) is None:
                raise
            return False
        persists = self._registry_persists(registry)
        with self._lock:
            if registry is not self._registry or self._fallback_writers:
                return False
            fallback = self._fallback_registry
            if fallback is not None:
                if not persists or (not discard_process_changes and fallback.publication != self._fallback_initial_publication):
                    return False
            self._fallback_registry = None
            self._open_failed = False
            self._clear_caches()
        return persists and (changed or fallback is not None)

    async def arefresh(self, *, discard_process_changes: bool = False) -> bool:
        """Refresh Registry state without blocking the event loop.

        Args:
            discard_process_changes: Adopt saved settings even if this
                replaces edits made to the process-only fallback.

        Returns:
            bool: Whether a newer persistent Registry snapshot was published.
        """
        import asyncio

        if discard_process_changes:
            return await asyncio.to_thread(self.refresh, discard_process_changes=True)
        return await asyncio.to_thread(self.refresh)

    def setup(self, reset: bool = False) -> bool:
        """
        Initialize or reset the base config layer.

        Use this helper for nested config dictionaries and scoped runtime configuration layers.

        Args:
            reset (bool): Whether to clear existing state before initializing or compacting. Defaults to ``False``.

        Returns:
            bool: The result of ``setup``.
        """

        def clear(draft: Registry[str, Dict[str, Any]]) -> Tuple[None, bool]:
            scopes = list(draft.list())
            for scope in scopes:
                draft.delete(scope)
            return None, bool(scopes)

        with self._mutation_registry() as registry:
            if reset:
                self._commit(clear, registry=registry)
                self._clear_caches()
            else:
                # Another process may have removed the base scope.
                registry.refresh()
            self._init_scope(self.base_scope, self.default, registry=registry)
        return True

    def init(self, scope: Optional[str] = None, reset: bool = False, data: Optional[Dict[str, Any]] = None) -> bool:
        """
        Initialize a scope with default or explicit data.

        Use this helper for nested config dictionaries and scoped runtime configuration layers.

        Args:
            scope (Optional[str]): Config scope. ``None`` uses the current active scope. Defaults to ``None``.
            reset (bool): Whether to clear existing state before initializing or compacting. Defaults to ``False``.
            data (Optional[Dict[str, Any]]): Dictionary layer or payload to read, write, merge, or render. Defaults to ``None``.

        Returns:
            bool: The result of ``init``.
        """
        target = (scope or self.scope).strip().lower()
        layer = to_mutable_data(copy.deepcopy(data if data is not None else self.default))
        chain = tuple(self.scope_chain)

        def change(draft: Registry[str, Dict[str, Any]]) -> Tuple[bool, bool]:
            if reset:
                draft.delete(target, missing_ok=True)
            elif self._versions(draft.get(target) or {}):
                return False, False
            keep = self._keep_last_k(self._draft_view(draft, chain))
            self._stage_layer(draft, target, layer, package_version=self.package_version, keep_last_k=keep)
            return True, True

        return self._commit_scope(target, change)

    def resource(self, *parts: str) -> Optional[str]:
        """
        Return an absolute path inside the package resources directory when available.

        Use this helper for nested config dictionaries and scoped runtime configuration layers.

        Args:
            *parts (str): Path parts or resource parts joined by the helper.

        Returns:
            Optional[str]: The an absolute path inside the package resources directory when available.
        """
        try:
            target = resources.files(self.package).joinpath("resources", *parts)
            with resources.as_file(target) as path:
                return str(path.resolve())
        except Exception:
            return None

    def load_default(self) -> Dict[str, Any]:
        """Return the default config template as a mutable dictionary.

        Use this helper for nested config dictionaries and scoped runtime configuration layers.

        Args:
            None.

        Returns:
            Dict[str, Any]: The the default config template as a mutable dictionary.
        """
        return copy.deepcopy(self.default)

    def pj(self, *parts: str, abs: bool = False) -> str:
        """
        Join path parts with package aliases.

        Use this helper for nested config dictionaries and scoped runtime configuration layers.

        Args:
            abs (bool): Whether to return an absolute resolved path. Defaults to ``False``.
            *parts (str): Path parts or resource parts joined by the helper.

        Returns:
            str: Normalized path string, absolute when ``abs=True``.
        """
        return _pj(*parts, aliases=_path_aliases(self.package, self.root), abs=abs)

    def to_dict(self) -> Dict[str, Any]:
        """Return current config as a mutable dictionary.

        Use this helper for nested config dictionaries and scoped runtime configuration layers.

        Args:
            None.

        Returns:
            Dict[str, Any]: The current config as a mutable dictionary.
        """
        return self.load()


class _Known(NamedTuple):
    """A resolved snapshot of a scope chain and the Registry state it was read from."""

    registry: Any
    publication: int
    stamps: Tuple[str, ...]
    snapshot: ConfigSnapshot


_MISSING_LIST = object()
_Location = Tuple[Any, ...]
# OmegaConf's grammar for a key in a node interpolation (``INTER_KEY``).
_KEY = r"""[^\\{}()\[\]:. \t'"]+"""
_NODE_REFERENCE = re.compile(rf"\$\{{[ \t]*(\.*)((?:{_KEY}|\[{_KEY}\])(?:\.{_KEY}|\[{_KEY}\])*)[ \t]*\}}")
_REFERENCE_PART = re.compile(rf"\[({_KEY})\]|({_KEY})")
_PLAIN_KEY = re.compile(_KEY)
_RELATIVE = re.compile(r"\$\{[ \t]*(\.+)")
_INTERPOLATION_OPEN = re.compile(r"\$\{[ \t]*")
# Resolvers whose value does not depend on where the interpolation is stored.
_PLACE_FREE_RESOLVERS = re.compile(r"(?:env|oc\.env)[ \t]*:")
_INDEX = re.compile(r"[0-9]+")
_FOLLOW_LIMIT = 64
_OPAQUE = object()


def _keep(layer: Dict[str, Any], held: Optional[Tuple[str, Any]]) -> None:
    """Save in ``layer`` the node a write goes through, from :meth:`_DraftView.through`."""
    if held is not None:
        dset(layer, held[0], held[1])


def _as_list(value: Any, path: str) -> List[Any]:
    """Return the list a list write at ``path`` edits: a copy of ``value``, or a new list when ``path`` holds none."""
    if value is _MISSING_LIST or value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    raise TypeError(f"config path {path!r} holds a {type(value).__name__}, not a list")


class _Through(NamedTuple):
    """What a write at a path goes through: see :meth:`_DraftView.through`."""

    value: Any
    held: Optional[Tuple[str, Any]]
    crossed: bool


def _write_steps(path: str) -> List[Tuple[str, Optional[str]]]:
    """Split ``path`` into the mapping keys and list ops :func:`dset` walks, as ``(key, None)`` or ``(key, op)``.

    A part such as ``items[0]`` is the key ``items`` and then the op ``0``
    on its list; the key is ``""`` for a part that indexes the list itself.
    """
    steps: List[Tuple[str, Optional[str]]] = []
    for part in _split_path(path):
        if part.endswith("]") and "[" in part:
            key, op = part[:-1].rsplit("[", 1)
            if key:
                steps.append((key, None))
            steps.append(("", op))
        else:
            steps.append((part, None))
    return steps


def _child_at(data: Any, location: _Location) -> Any:
    """Return the value at ``location`` in ``data``, or :data:`_MISSING_LIST`."""
    for key in location:
        if isinstance(data, Mapping) and key in data:
            data = data[key]
        elif isinstance(data, (list, tuple)) and isinstance(key, int) and -len(data) <= key < len(data):
            data = data[key]
        else:
            return _MISSING_LIST
    return data


def _reference(value: str, location: _Location) -> Optional[_Location]:
    """Return the location a plain node interpolation at ``location``, such as ``${a.b}`` or ``${..b}``, names.

    Keys follow OmegaConf's grammar, so a key such as ``服务`` or ``db@main``
    is followed; every key is a string, as OmegaConf reads it.
    """
    match = _NODE_REFERENCE.fullmatch(value)
    if match is None:
        return None
    dots = len(match.group(1))
    if dots > len(location):
        return None
    base = location[: len(location) - dots] if dots else ()
    return (*base, *(bracketed or key for bracketed, key in _REFERENCE_PART.findall(match.group(2))))


def _single(text: str) -> bool:
    """Whether ``text`` may be one interpolation and nothing else, the only kind of value that can yield a container.

    Text outside the interpolation, such as ``"pre ${env:NAME}"``, always
    yields a string. Quoted text may hold braces, so text with quotes counts
    as a possible single interpolation.
    """
    if not (text.startswith("${") and text.endswith("}")):
        return False
    if "'" in text or '"' in text:
        return True
    depth = 0
    escaped = False
    for at, char in enumerate(text):
        if escaped:
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return at == len(text) - 1
    return True


class _Found(NamedTuple):
    """What a value stored in the config yields: see :func:`_lookup`."""

    node: Any
    location: _Location


_SCALAR = "scalar"
_UNKNOWN = "unknown"


def _lookup(root: Dict[str, Any], location: _Location, value: str, budget: Optional[List[int]] = None) -> Any:
    """Follow the plain node interpolation ``value`` at ``location`` in the stored config ``root``.

    Returns a :class:`_Found` with the container it names and that
    container's own location, after following each plain node interpolation
    on the way. Returns ``_UNKNOWN`` when it reaches another interpolation
    that may yield a container, such as a resolver, and ``_SCALAR`` when it
    yields no container: a value that is not one interpolation, a node
    interpolation naming a scalar or nothing, or interpolations that form a
    cycle.
    """
    budget = [_FOLLOW_LIMIT] if budget is None else budget
    while budget[0] > 0:
        budget[0] -= 1
        target = _reference(value, location)
        if target is None:
            return _UNKNOWN if _single(value) else _SCALAR
        node: Any = root
        location = ()
        for key in target:
            if isinstance(node, str):
                found = _lookup(root, location, node, budget)
                if not isinstance(found, _Found):
                    return found
                node, location = found
            if isinstance(node, Mapping) and key in node:
                node = node[key]
            elif isinstance(node, (list, tuple)) and _INDEX.fullmatch(key) and int(key) < len(node):
                key = int(key)
                node = node[key]
            else:
                return _SCALAR
            location = (*location, key)
        if isinstance(node, (Mapping, list, tuple)):
            return _Found(node, location)
        if not isinstance(node, str):
            return _SCALAR
        value = node
    return _SCALAR


def _relocate(value: Any, source: _Location, target: _Location) -> Any:
    """Return ``value``, copied from ``source`` to ``target``, with each interpolation in it keeping its value.

    An interpolation that names a node inside the copy, such as ``${.name}``
    beside ``name``, is kept as written. A relative interpolation that names
    a node outside it, such as ``${..name}``, is rewritten to name that node
    from ``target`` (:func:`_name`). A string holding a resolver whose value
    may depend on where it is stored, such as ``${oc.select:..name}``,
    becomes a node interpolation naming the string at ``source``. Raises
    :class:`_NotWritable` when such a node cannot be named.
    """
    if source == target:
        return value

    def walk(node: Any, below: _Location) -> Any:
        if isinstance(node, dict):
            for key in node:
                node[key] = walk(node[key], (*below, key if isinstance(key, str) else _OPAQUE))
        elif isinstance(node, list):
            for index, item in enumerate(node):
                node[index] = walk(item, (*below, index))
        elif isinstance(node, str) and "${" in node:
            if not _place_free(node):
                name = _name((*target, *below), (*source, *below))
                if not name.strip("."):
                    raise _NotWritable
                return "${" + name + "}"
            node = _moved(node, (*source, *below), (*target, *below), len(below))
        return node

    return walk(value, ())


class _NotWritable(Exception):
    """A path that no node interpolation can name."""


def _place_free(text: str) -> bool:
    """Whether every interpolation in ``text`` is a node interpolation or a resolver that :func:`_moved` can move."""
    for match in _INTERPOLATION_OPEN.finditer(text):
        rest = text[match.end() :]
        if not (rest.startswith((".", "[")) or _NODE_REFERENCE.match(text, match.start()) or _PLACE_FREE_RESOLVERS.match(rest)):
            return False
    return True


def _path_text(location: _Location) -> str:
    """Return ``location`` as the key path of a node interpolation, or raise :class:`_NotWritable`."""
    path = ""
    for key in location:
        if type(key) is int:
            path += f"[{key}]"
        elif isinstance(key, str) and _PLAIN_KEY.fullmatch(key):
            path += f".{key}" if path else key
        else:
            raise _NotWritable
    return path


def _name(at: _Location, node: _Location) -> str:
    """Return how an interpolation stored at ``at`` names ``node``, without the closing key text.

    The absolute key path when every key on it can be written, else a
    relative one: a dot for each level from ``at`` up to the deepest node
    the two share, then the keys down to ``node``. Raises
    :class:`_NotWritable` when neither can be written.
    """
    try:
        return _path_text(node)
    except _NotWritable:
        shared = 0
        while shared < min(len(at), len(node)) and at[shared] == node[shared]:
            shared += 1
        return "." * (len(at) - shared) + _path_text(node[shared:])


def _moved(text: str, source: _Location, target: _Location, depth: int) -> str:
    """Rewrite each relative interpolation in ``text``, moved from ``source`` to ``target``, that leaves the copied node.

    ``depth`` is how far ``source`` lies below the copied node, so a
    relative interpolation with at most ``depth`` dots names a node inside
    the copy and is kept as written.
    """

    def moved(match: "re.Match[str]") -> str:
        escapes = len(text[: match.start()]) - len(text[: match.start()].rstrip("\\"))
        dots = len(match.group(1))
        if escapes % 2 or dots <= depth:
            return match.group(0)
        if dots > len(source):
            raise _NotWritable
        name = _name(target, source[: len(source) - dots])
        if not name.strip(".") or text[match.end() : match.end() + 1] in ("[", "}"):
            return "${" + name
        return "${" + name + "."

    return _RELATIVE.sub(moved, text)


class _DraftView:
    """The resolved config a write reads, built from its draft only when an edit needs it.

    Resolving every interpolation of the merged config is most of a write's
    own work, and only :meth:`ConfigManager.setdef` reads resolved values;
    edits read the stored values of the written scope's chain
    (:meth:`through`), resolving only the path to an interpolation other
    than a plain node reference to learn whether it yields a mapping or a
    list (:meth:`_yields_container`), and the retained-version count is
    read unresolved unless it is itself an interpolation. A draft holds the
    records of the Registry publication it was made from, so a ``known``
    snapshot read at that same publication is used without reading the
    draft's layers.
    """

    def __init__(
        self,
        manager: ConfigManager,
        draft: Registry[str, Dict[str, Any]],
        chain: Tuple[str, ...],
        known: Optional[_Known] = None,
        stored: Optional[Tuple[str, ...]] = None,
    ) -> None:
        self._manager = manager
        self._draft = draft
        self._chain = chain
        self._known = known
        self._stored_chain = chain if stored is None else stored
        self._merged: Optional[Dict[str, Any]] = None
        self._read: Dict[Tuple[str, ...], Tuple[Tuple[int, ...], List[Dict[str, Any]]]] = {}
        self._snapshot: Optional[ConfigSnapshot] = None

    def _layers(self, chain: Optional[Tuple[str, ...]] = None) -> Tuple[Tuple[int, ...], List[Dict[str, Any]]]:
        chain = self._chain if chain is None else chain
        if chain not in self._read:
            records = [self._draft.get(scope) for scope in chain]
            gen = tuple(int((record or {}).get("generation") or 0) for record in records)
            self._read[chain] = (gen, [self._manager._latest_of(record)[1] for record in records])
        return self._read[chain]

    def get(self, path: Optional[str] = None, default: Any = None) -> Any:
        if self._snapshot is None:
            known = self._known
            if known is not None and known.registry is self._manager._registry and known.publication == self._draft.publication:
                self._snapshot = known.snapshot
            else:
                gen, layers = self._layers()
                if known is not None and known.stamps == tuple(_stamp(generation, layer) for generation, layer in zip(gen, layers)):
                    self._snapshot = known.snapshot
                else:
                    self._snapshot = self._manager._resolved([copy.deepcopy(layer) for layer in layers], gen, InterpolationPolicy())
        return self._snapshot.get(path, default=default)

    def _stored(self) -> Dict[str, Any]:
        """Return the merged, unresolved config of the written scope's chain."""
        if self._merged is None:
            self._merged = _merge_layers([self._manager.default, *self._layers(self._stored_chain)[1]])
        return self._merged

    def _yields_container(self, location: _Location) -> bool:
        """Whether the value at ``location`` in the written scope's chain resolves to a mapping or a list.

        Only the interpolations on the way to ``location`` and in its value
        are resolved, so an unrelated interpolation that fails does not fail
        the write.
        """
        from omegaconf import DictConfig, ListConfig, OmegaConf

        from .interpolation import _INTERP_POLICY, _setup_resolvers

        _setup_resolvers()
        token = _INTERP_POLICY.set(InterpolationPolicy())
        try:
            node: Any = OmegaConf.create(self._stored())
            for key in location:
                node = node[key]
            return isinstance(node, (DictConfig, ListConfig))
        finally:
            _INTERP_POLICY.reset(token)

    def _follow(self, path: str, value: str, location: _Location) -> Any:
        """Return the container that the interpolation ``value`` at ``location`` holds, as a layer may save it.

        A plain node reference such as ``${a.b}`` or ``${..b}`` is followed to
        the stored node it names, so the node's interpolations, such as
        ``${env:NAME}``, are kept as written (:func:`_relocate`). Returns
        :data:`_MISSING_LIST` when ``value`` holds no container. Raises
        ``ValueError`` when the container can be saved only with resolved
        values, which may hold credentials: another interpolation, such as a
        resolver, yields it, or a copy of it cannot keep the value of each
        interpolation it holds. ``path`` is the written path the error names.
        """
        found = _lookup(self._stored(), location, value)
        where = ".".join(str(key) if isinstance(key, str) else f"[{key}]" for key in location).replace(".[", "[")
        if isinstance(found, _Found):
            try:
                return _relocate(to_mutable_data(found.node), found.location, location)
            except _NotWritable:
                raise ValueError(
                    f"cannot write config path {path!r}: copying the node that {where!r} names into it would change "
                    f"the value of an interpolation the node holds; set {where!r} as a whole instead"
                ) from None
        if found == _UNKNOWN and self._yields_container(location):
            raise ValueError(
                f"cannot write config path {path!r}: {where!r} is an interpolation that is not a plain node reference, "
                f"so the write could keep its other values only by saving their resolved values; set {where!r} as a whole instead"
            )
        return _MISSING_LIST

    def through(self, path: str, *, into_last: bool) -> "_Through":
        """Walk ``path`` in the stored config of the written scope's chain, as a write at ``path`` goes through it.

        A write below a list, or below a node that an interpolation holds,
        cannot be saved as an edit of that node alone: the written layer's
        list replaces the whole list a read merges, and its mapping replaces
        the interpolation. The walk therefore returns the shallowest such
        node as ``held``, the stored node a layer must save at that path
        before the write, with each interpolation along ``path`` followed
        (:meth:`_follow`). ``into_last`` also follows an interpolation at
        ``path`` itself, for a list edit. ``value`` is the stored value at
        ``path``, or :data:`_MISSING_LIST`, and ``crossed`` tells whether the
        walk went through a list or an interpolation.
        """
        steps = _write_steps(path)
        node: Any = self._stored()
        location: _Location = ()
        held: Optional[Tuple[_Location, Any]] = None
        crossed = False
        for at, (key, op) in enumerate(steps):
            last = at == len(steps) - 1
            if op is None:
                if not isinstance(node, dict) or key not in node:
                    return _Through(_MISSING_LIST, self._hold(held), crossed)
                container: Any = node
                index: Any = key
            else:
                if not isinstance(node, (list, tuple)):
                    return _Through(_MISSING_LIST, self._hold(held), crossed)
                crossed = True
                if held is None:
                    node = to_mutable_data(node)
                    held = (location, node)
                if op in ("", "-") or op.endswith("+"):
                    return _Through(_MISSING_LIST, self._hold(held), crossed)
                index = int(op)
                if not -len(node) <= index < len(node):
                    return _Through(_MISSING_LIST, self._hold(held), crossed)
                index %= len(node)
                container = node
            location = (*location, index)
            child = container[index]
            if (into_last or not last) and isinstance(child, str) and "${" in child:
                crossed = True
                followed = self._follow(path, child, location)
                if followed is _MISSING_LIST:
                    return _Through(child if last else _MISSING_LIST, self._hold(held), crossed)
                child = followed
                if held is None:
                    held = (location, child)
                else:
                    container[index] = child
            node = child
        return _Through(node, self._hold(held), crossed)

    def crosses(self, path: str) -> bool:
        """Whether a stored layer of the written scope's chain holds a list or an interpolation above ``path``.

        Only a write that goes through one needs :meth:`through`; this check
        reads the raw layers without merging them.
        """
        steps = _write_steps(path)[:-1]
        for layer in (self._manager.default, *self._layers(self._stored_chain)[1]):
            node: Any = layer
            for key, op in steps:
                if op is not None or not isinstance(node, Mapping) or key not in node:
                    break
                node = node[key]
                if isinstance(node, (list, tuple)) or (isinstance(node, str) and "${" in node):
                    return True
        return False

    def _hold(self, held: Optional[Tuple[_Location, Any]]) -> Optional[Tuple[str, Any]]:
        """Return ``held`` as the path and value a layer saves.

        Its location is a path of mapping keys, since the walk holds the first
        list it meets. A mapping that replaces an interpolation saved in the
        written layer itself replaces, rather than extends, the mapping below.
        """
        if held is None:
            return None
        location, value = held
        if isinstance(value, dict) and isinstance(self._below(location), Mapping):
            value[_OVERWRITE] = True
        return ".".join(str(key).replace(".", "\\.") for key in location), value

    def _below(self, location: _Location) -> Any:
        """Return the merged stored value at ``location`` under the written layer."""
        layers = self._layers(self._stored_chain)[1]
        return _child_at(_merge_layers([self._manager.default, *layers[:-1]]), location)

    def setting(self, path: str, default: Any) -> Any:
        """Return one scalar setting, resolving the whole config only when it is an interpolation."""
        value = dget(_merge_layers([self._manager.default, *self._layers()[1]]), path, default)
        if isinstance(value, str) and "${" in value:
            return self.get(path, default=default)
        return value


def _package_scopes() -> Registry[str, Dict[str, Any]]:
    """Open ``CM_HVNB``'s scopes where the process default Context persists them.

    ``CM_HVNB`` is the default Context's ConfigManager, but ``utils`` may not
    import that Context, so the package root binds this hook to it. Bound when
    ``CM_HVNB`` is created, so the first read sees config saved with
    ``hb config set`` whatever was imported before; selecting the default
    Context happens on that read, never on import.
    """
    return resolve_late("heavenbase.config.package_scopes")()


CM_HVNB = ConfigManager(package="heavenbase", scope="heavenbase", setup=True)
CM_HVNB._bind_registry_factory(_package_scopes)
