"""Content-addressed capture and isolated loading for local module folders."""

from __future__ import annotations

from collections import OrderedDict, deque
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from hashlib import sha256
import io
from importlib import import_module
from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import EXTENSION_SUFFIXES, ModuleSpec
from importlib.util import decode_source

try:
    from importlib.resources.abc import Traversable, TraversableResources
except ModuleNotFoundError:  # Python 3.10 keeps these ABCs on importlib.abc.
    from importlib.abc import Traversable, TraversableResources
import os
from pathlib import Path
from pathlib import PurePosixPath
import re
import sys
import tempfile
import threading
import time
from types import MappingProxyType, ModuleType
from typing import Any
import weakref

from heavenbase.utils.files import delete_dir, enum_dirs, enum_files, exists_dir, exists_file, exists_path, is_link, path_inside, pj
from heavenbase.utils.log import get_logger
from heavenbase.utils.serialize import load_bin

from ._signature import tree_signature

__all__ = ["ModuleArtifact", "ModuleArtifactStore", "SourceUnavailable"]

_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_MODULE_RE = re.compile(r"^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*$")
_QUALNAME_RE = re.compile(r"^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*$")
_NAMESPACE_RE = re.compile(r"^_heavenbase_artifact_[0-9a-f]{64}$")
_BYTECODE_SUFFIXES = frozenset((".pyc", ".pyo"))
_NATIVE_SUFFIXES = tuple(dict.fromkeys((*EXTENSION_SUFFIXES, ".so", ".pyd", ".dll", ".dylib")))
_CAPTURE_ATTEMPTS = 3
_logger = get_logger(__name__)

# Per-process memos keyed by folder fingerprints (see `_signature.tree_signature`),
# each bounded so that many short-lived Contexts or folders cannot grow them:
# package-resource folders read before (path -> (fingerprint, tree)), the bytes
# of each verified capture (digest -> tree), and the store captures whose bytes
# matched their digest (root -> (digest, fingerprint)).
_MEMO_LIMIT = 512
_RESOURCE_TREES: OrderedDict[str, tuple[tuple, "_Tree"]] = OrderedDict()
_CAPTURE_TREES: OrderedDict[str, "_Tree"] = OrderedDict()
_VERIFIED_CAPTURES: OrderedDict[str, tuple[str, tuple]] = OrderedDict()
_MEMO_LOCK = threading.Lock()


def _new_memo_lock_in_child() -> None:
    """Give a forked child an unheld memo lock: the thread that held the parent's does not exist there."""
    global _MEMO_LOCK
    _MEMO_LOCK = threading.Lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_new_memo_lock_in_child)


def _remember(memo: OrderedDict[Any, Any], key: Any, value: Any) -> None:
    """Keep one memo entry as the newest, dropping the oldest beyond the limit; the caller holds the lock."""
    memo[key] = value
    memo.move_to_end(key)
    while len(memo) > _MEMO_LIMIT:
        memo.popitem(last=False)


def forget_readings() -> None:
    """Drop every per-process memo, so the next capture reads and verifies all bytes again."""
    with _MEMO_LOCK:
        _RESOURCE_TREES.clear()
        _CAPTURE_TREES.clear()
        _VERIFIED_CAPTURES.clear()


class SourceUnavailable(ImportError):
    """Report that a captured source tree is unavailable before execution."""


class _SourceChanged(RuntimeError):
    """Report a source-tree race that may succeed on a fresh capture attempt."""


@dataclass(frozen=True, slots=True)
class ModuleArtifact:
    """Identify one immutable module-folder capture.

    Args:
        digest (str): SHA-256 content digest of every retained path, type, and
            file byte in the captured root.
        root (Path): Captured root-package path inside its owning store.
    """

    digest: str
    root: Path


@dataclass(frozen=True, slots=True)
class _TreeEntry:
    path: str
    kind: str
    content: bytes = b""


@dataclass(frozen=True, slots=True)
class _Tree:
    entries: tuple[_TreeEntry, ...]
    digest: str

    @property
    def files(self) -> Mapping[str, bytes]:
        return MappingProxyType({entry.path: entry.content for entry in self.entries if entry.kind == "file"})

    @property
    def directories(self) -> frozenset[str]:
        return frozenset(entry.path for entry in self.entries if entry.kind == "directory")


@dataclass(frozen=True, slots=True)
class _ModuleSource:
    path: str | None
    content: bytes
    package: bool


@dataclass(frozen=True, slots=True)
class _Generation:
    namespace: str
    digest: str
    files: Mapping[str, bytes]
    directories: frozenset[str]

    def source(self, relative: str) -> _ModuleSource | None:
        base = relative.replace(".", "/")
        if not base:
            return _ModuleSource("__init__.py", self.files["__init__.py"], True)
        if self._native(base):
            raise ImportError(f"artifact module {relative!r} requires an unsupported native loader")
        package_path = f"{base}/__init__.py"
        if package_path in self.files:
            return _ModuleSource(package_path, self.files[package_path], True)
        module_path = f"{base}.py"
        if module_path in self.files:
            return _ModuleSource(module_path, self.files[module_path], False)
        if base in self.directories:
            return _ModuleSource(None, b"", True)
        return None

    def _native(self, base: str) -> bool:
        candidates = tuple(f"{base}{suffix}" for suffix in _NATIVE_SUFFIXES) + tuple(f"{base}/__init__{suffix}" for suffix in _NATIVE_SUFFIXES)
        return any(candidate in self.files for candidate in candidates)


@dataclass(slots=True)
class _OwnerLease:
    reference: weakref.ReferenceType[object]
    loading: int = 0
    retained: bool = False


@dataclass(frozen=True, slots=True)
class _ArtifactPath(Traversable):
    """Traverse verified generation bytes without exposing their store path."""

    generation: _Generation
    path: str = ""

    @property
    def name(self) -> str:
        """Return the final virtual path component."""
        return self.path.rsplit("/", 1)[-1] if self.path else self.generation.namespace

    def is_dir(self) -> bool:
        """Return whether this virtual path is a captured directory."""
        return not self.path or self.path in self.generation.directories

    def is_file(self) -> bool:
        """Return whether this virtual path is a captured regular file."""
        return self.path in self.generation.files

    def iterdir(self):
        """Yield immediate captured children in deterministic name order."""
        if not self.is_dir():
            raise NotADirectoryError(self.path)
        prefix = f"{self.path}/" if self.path else ""
        children: set[str] = set()
        candidates = (*self.generation.directories, *self.generation.files)
        for candidate in candidates:
            if candidate == "." or not candidate.startswith(prefix):
                continue
            relative = candidate[len(prefix) :]
            if relative:
                children.add(f"{prefix}{relative.split('/', 1)[0]}")
        return iter(_ArtifactPath(self.generation, child) for child in sorted(children))

    def joinpath(self, *descendants):
        """Return a virtual descendant while rejecting escaping paths."""
        parts = self.path.split("/") if self.path else []
        for descendant in descendants:
            if not isinstance(descendant, (str, os.PathLike)):
                raise TypeError("artifact resource path components must be strings or path-like values")
            candidate = PurePosixPath(os.fspath(descendant))
            if candidate.is_absolute() or any(part == ".." for part in candidate.parts):
                raise ValueError(f"artifact resource path escapes its package: {descendant!r}")
            parts.extend(part for part in candidate.parts if part not in ("", "."))
        return _ArtifactPath(self.generation, "/".join(parts))

    def open(self, mode="r", *args, **kwargs):
        """Open one verified resource as an in-memory binary or text stream."""
        if mode not in ("r", "rb"):
            raise ValueError(f"artifact resources are read-only; unsupported mode {mode!r}")
        if self.is_dir():
            raise IsADirectoryError(self.path)
        content = self.generation.files.get(self.path)
        if content is None:
            raise FileNotFoundError(self.path)
        stream = io.BytesIO(content)
        if mode == "rb":
            if args or kwargs:
                raise ValueError("binary artifact resources do not accept text encoding options")
            return stream
        return io.TextIOWrapper(stream, *args, **kwargs)


@dataclass(frozen=True, slots=True)
class _ArtifactResources(TraversableResources):
    """Serve package resources from one verified in-memory generation."""

    generation: _Generation
    package: str

    def files(self) -> Traversable:
        """Return the virtual package root for `importlib.resources`."""
        return _ArtifactPath(self.generation, self.package)


class _ArtifactLoader(Loader):
    def __init__(self, generation: _Generation, fullname: str, source: _ModuleSource) -> None:
        self.generation = generation
        self.fullname = fullname
        self.source = source

    def create_module(self, spec: ModuleSpec) -> ModuleType | None:
        return None

    def exec_module(self, module: ModuleType) -> None:
        if self.source.path is None:
            return
        filename = self.get_filename(self.fullname)
        code = compile(self.source.content, filename, "exec", dont_inherit=True)
        exec(code, module.__dict__)

    def get_filename(self, fullname: str) -> str:
        """Return the stable virtual source location for this exact module."""
        if fullname != self.fullname or self.source.path is None:
            raise ImportError(f"artifact loader does not own source for {fullname!r}")
        return f"heavenbase-artifact://{self.generation.digest}/{self.source.path}"

    def get_source(self, fullname: str) -> str | None:
        """Return verified source text so inspection can capture nested callbacks."""
        if fullname != self.fullname:
            raise ImportError(f"artifact loader does not own source for {fullname!r}")
        if self.source.path is None:
            return None
        return decode_source(self.source.content)

    def get_resource_reader(self, fullname: str) -> TraversableResources | None:
        """Return a verified resource reader for artifact packages."""
        if not self.source.package:
            return None
        namespace, separator, relative = fullname.partition(".")
        if namespace != self.generation.namespace:
            return None
        package = relative.replace(".", "/") if separator else ""
        return _ArtifactResources(self.generation, package)


class _ArtifactFinder(MetaPathFinder):
    def __init__(self) -> None:
        self._generations: dict[str, _Generation] = {}
        self._owners: dict[str, dict[int, _OwnerLease]] = {}
        self._loading: dict[str, int] = {}
        self._lock = threading.RLock()
        # Leases of collected owners, queued by their weak-reference callbacks.
        self._collected: deque[tuple[str, int, weakref.ReferenceType[object]]] = deque()

    def register(self, generation: _Generation, owner: object) -> _Generation:
        """Begin loading one generation for a weak lifecycle owner."""
        owner_id = id(owner)
        collected = self._collected
        try:
            candidate_reference = weakref.ref(
                owner,
                lambda value, namespace=generation.namespace, identity=owner_id: collected.append((namespace, identity, value)),
            )
        except TypeError as error:
            raise TypeError("artifact lifecycle owner must support weak references") from error
        self._reap()
        with self._lock:
            current = self._generations.get(generation.namespace)
            if current is None:
                self._generations[generation.namespace] = generation
                current = generation
            elif current.digest != generation.digest or current.files != generation.files or current.directories != generation.directories:
                raise ImportError(f"private artifact namespace collision: {generation.namespace}")
            owners = self._owners.setdefault(generation.namespace, {})
            lease = owners.get(owner_id)
            if lease is None or lease.reference() is not owner:
                lease = _OwnerLease(candidate_reference)
                owners[owner_id] = lease
            lease.loading += 1
            self._loading[generation.namespace] = self._loading.get(generation.namespace, 0) + 1
            return current

    def finish_load(self, namespace: str, owner: object, *, succeeded: bool) -> None:
        """End one load, retaining only generations loaded successfully."""
        self._reap()
        with self._lock:
            owners = self._owners.get(namespace)
            lease = owners.get(id(owner)) if owners is not None else None
            if lease is None or lease.reference() is not owner or lease.loading <= 0:
                raise RuntimeError(f"artifact generation {namespace!r} has no active owner load")
            if succeeded:
                lease.retained = True
            lease.loading -= 1
            if lease.loading == 0 and not lease.retained:
                del owners[id(owner)]
            loading = self._loading.get(namespace, 0)
            if loading <= 0:
                raise RuntimeError(f"artifact generation {namespace!r} has no active load")
            if loading == 1:
                self._loading.pop(namespace, None)
            else:
                self._loading[namespace] = loading - 1
            self._collect(namespace)

    def release(self, namespace: str, owner: object) -> bool:
        """Release one owner's generation lease and collect it when unused."""
        self._reap()
        with self._lock:
            lease = self._owners.get(namespace, {}).get(id(owner))
            if lease is None or lease.reference() is not owner:
                return False
            return self._release(namespace, id(owner), lease.reference)

    def release_except(self, owner: object, retained: frozenset[str]) -> tuple[str, ...]:
        """Release every generation owned by `owner` except retained names."""
        self._reap()
        with self._lock:
            candidates: list[tuple[str, weakref.ReferenceType[object]]] = []
            for namespace, owners in tuple(self._owners.items()):
                lease = owners.get(id(owner))
                if namespace not in retained and lease is not None and lease.reference() is owner:
                    candidates.append((namespace, lease.reference))
            released: list[str] = []
            for namespace, reference in candidates:
                if self._release(namespace, id(owner), reference):
                    released.append(namespace)
            return tuple(released)

    def _reap(self) -> None:
        """Release the leases of owners collected since the previous finder call.

        An owner can be collected inside any critical section of any thread,
        this finder's own included, where its weak-reference callback must
        neither wait for a lock nor change the registry under the interrupted
        section. The callback therefore only queues the lease with an atomic
        append; it is released here, at an ordinary call, under the lock.
        """
        if not self._collected:
            return
        with self._lock:
            while self._collected:
                namespace, owner_id, reference = self._collected.popleft()
                self._release(namespace, owner_id, reference)

    def _release(self, namespace: str, owner_id: int, expected: weakref.ReferenceType[object]) -> bool:
        """Release an exact weak owner reference."""
        with self._lock:
            owners = self._owners.get(namespace)
            lease = owners.get(owner_id) if owners is not None else None
            if lease is None or lease.reference is not expected:
                return False
            lease.retained = False
            if lease.loading == 0:
                del owners[owner_id]
            return self._collect(namespace)

    def _collect(self, namespace: str) -> bool:
        """Collect one generation once it has no owners or in-flight loads."""
        owners = self._owners.get(namespace)
        if owners or self._loading.get(namespace, 0):
            return False
        self._owners.pop(namespace, None)
        generation = self._generations.pop(namespace, None)
        if generation is not None:
            self._purge_modules(generation)
            return True
        return False

    @staticmethod
    def _purge_modules(generation: _Generation) -> None:
        """Remove only modules loaded by the exact retired artifact generation."""
        namespace = generation.namespace
        names = sorted(
            (name for name in tuple(sys.modules) if name == namespace or name.startswith(f"{namespace}.")),
            key=lambda name: (name.count("."), name),
            reverse=True,
        )
        for name in names:
            module = sys.modules.get(name)
            loader = getattr(getattr(module, "__spec__", None), "loader", None)
            if isinstance(loader, _ArtifactLoader) and loader.generation is generation:
                sys.modules.pop(name, None)

    def find_spec(self, fullname: str, path: object = None, target: ModuleType | None = None) -> ModuleSpec | None:
        namespace, separator, relative = fullname.partition(".")
        if not _NAMESPACE_RE.fullmatch(namespace):
            return None
        self._reap()
        with self._lock:
            generation = self._generations.get(namespace)
        if generation is None:
            return None
        source = generation.source(relative if separator else "")
        if source is None:
            return None
        loader = _ArtifactLoader(generation, fullname, source)
        spec = ModuleSpec(fullname, loader, origin=f"heavenbase-artifact://{generation.digest}/{source.path or relative}", is_package=source.package)
        spec.has_location = source.path is not None
        if source.package:
            spec.submodule_search_locations = [f"heavenbase-artifact://{generation.digest}/{relative.replace('.', '/')}"]
        return spec


_FINDER = _ArtifactFinder()
_FINDER_LOCK = threading.RLock()


def _install_finder() -> None:
    with _FINDER_LOCK:
        if _FINDER not in sys.meta_path:
            sys.meta_path.insert(0, _FINDER)


def _frame(digest: Any, domain: bytes, value: bytes) -> None:
    digest.update(len(domain).to_bytes(8, "big"))
    digest.update(domain)
    digest.update(len(value).to_bytes(8, "big"))
    digest.update(value)


def _tree_digest(entries: tuple[_TreeEntry, ...]) -> str:
    digest = sha256()
    _frame(digest, b"format", b"heavenbase-module-artifact-v1")
    _frame(digest, b"entry-count", len(entries).to_bytes(8, "big"))
    for entry in entries:
        _frame(digest, b"entry", b"")
        _frame(digest, b"path", entry.path.encode("utf-8"))
        _frame(digest, b"type", entry.kind.encode("ascii"))
        _frame(digest, b"content", entry.content)
    return digest.hexdigest()


def _relative(parts: tuple[str, ...]) -> str:
    value = "/".join(parts)
    if not value or value.startswith("/") or any(part in ("", ".", "..") or "/" in part or "\\" in part for part in parts):
        raise ValueError(f"module artifact contains an escaping path: {value!r}")
    value.encode("utf-8")
    return value


def _ignored_name(name: str) -> bool:
    """Return whether an interpreter cache entry is outside source identity."""
    return name == "__pycache__" or Path(name).suffix.lower() in _BYTECODE_SUFFIXES


def _validate_name(path: str, name: str) -> None:
    """Reject source entries that cannot belong to a clean Python artifact."""
    if _ignored_name(name):
        raise ValueError(f"internal error: ignored cache entry reached validation: {path}")


def _absolute_under(root: Path, relative: str) -> tuple[str, Path]:
    """Return the artifact-relative path and absolute filesystem path for one entry."""
    parts = Path(relative).parts
    path = _relative(parts)
    return path, root.joinpath(*parts)


def _guard_source_entry(root: Path, path: str, absolute: Path) -> None:
    """Reject symlinks and paths that escape the capture root."""
    if is_link(absolute):
        raise ValueError(f"module artifact contains a symlink: {path}")
    if not path_inside(root, absolute):
        raise ValueError(f"module artifact path escapes its root: {path}")


def _long_path(path: Path) -> Path:
    """Return an absolute path in Windows extended-length form; other platforms keep it unchanged.

    The content-addressed layout adds about 90 characters below the store root, so
    captures under an ordinary profile can pass the 260-character ``MAX_PATH`` limit
    that Windows applies when long paths are not enabled. Extended-length paths
    (``\\\\?\\``) are exempt from that limit.
    """
    if os.name != "nt":
        return path
    text = str(path)
    if text.startswith("\\\\?\\"):
        return path
    if text.startswith("\\\\"):
        return Path("\\\\?\\UNC\\" + text[2:])
    return Path("\\\\?\\" + text)


def _raise_walk_error(error: OSError) -> None:
    """Propagate an error that `os.walk` reports for a source directory."""
    raise error


def _source_paths(root: Path) -> Iterator[tuple[str, bool]]:
    """Yield each source path below `root` with whether it is a directory.

    The walk never enters an interpreter cache folder. Caches are not source,
    so a `__pycache__` folder that a cleanup removes during the walk is not a
    source change, while a source directory that cannot be listed still fails
    the scan.
    """
    for current, dir_names, file_names in os.walk(root, onerror=_raise_walk_error):
        dir_names[:] = [name for name in dir_names if not _ignored_name(name)]
        for name in dir_names:
            yield os.path.relpath(os.path.join(current, name), root), True
        for name in file_names:
            if not _ignored_name(name):
                yield os.path.relpath(os.path.join(current, name), root), False


def _scan_source_tree(root: Path) -> list[_TreeEntry]:
    """Read one whole-tree content snapshot, leaving interpreter caches out."""
    entries = [_TreeEntry(".", "directory")]
    try:
        for relative, is_directory in _source_paths(root):
            path, absolute = _absolute_under(root, relative)
            if is_directory:
                _guard_source_entry(root, path, absolute)
                if not exists_dir(absolute):
                    raise ValueError(f"module artifact contains a special file: {path}")
                entries.append(_TreeEntry(path, "directory"))
                continue
            _validate_name(path, Path(relative).name)
            _guard_source_entry(root, path, absolute)
            if not exists_file(absolute):
                raise ValueError(f"module artifact contains a special file: {path}")
            try:
                content = load_bin(str(absolute), strict=True)
            except FileNotFoundError as error:
                raise _SourceChanged(f"module artifact file changed while captured: {path}") from error
            if not isinstance(content, bytes):
                raise TypeError(f"module artifact file returned non-bytes content: {path}")
            entries.append(_TreeEntry(path, "file", content))
    except OSError as error:
        raise _SourceChanged(f"module artifact source changed while captured: {root}") from error
    return entries


def _read_tree_once(root: Path) -> _Tree:
    source = Path(root).expanduser()
    if is_link(source):
        raise ValueError(f"module artifact root is a symlink: {source}")
    if not exists_path(source):
        raise FileNotFoundError(source)
    if not exists_dir(source):
        raise ValueError(f"module artifact root must be a directory: {source}")
    canonical = _long_path(Path(pj(source, abs=True)).resolve(strict=True))
    ordered = tuple(sorted(_scan_source_tree(canonical), key=lambda entry: entry.path))
    kinds = {entry.path: entry.kind for entry in ordered}
    for required in ("__init__.py", "meta.yaml"):
        if kinds.get(required) != "file":
            raise ValueError(f"module artifact root requires a regular {required}")
    return _Tree(ordered, _tree_digest(ordered))


def _read_tree(root: Path) -> _Tree:
    """Read two matching whole-tree content digests with bounded retries."""
    changed: _SourceChanged | None = None
    previous: _Tree | None = None
    for _ in range(_CAPTURE_ATTEMPTS):
        try:
            current = _read_tree_once(root)
        except _SourceChanged as error:
            changed = error
            previous = None
            continue
        if previous is not None and current == previous:
            return current
        if previous is not None:
            changed = _SourceChanged(f"module artifact source changed between complete scans: {root}")
        previous = current
    raise ValueError(f"module artifact source changed repeatedly while captured: {root}") from changed


_WINDOWS = sys.platform == "win32"
# Windows errors a handle that another program briefly holds can cause: access denied, sharing violation.
_TRANSIENT_WINERRORS = frozenset((5, 32))
_RENAME_ATTEMPTS = 8
_RENAME_FIRST_DELAY = 0.05


def _rename_capture(temporary: Path, target: Path) -> None:
    """Move a written capture into place, retrying on Windows while another program holds a handle.

    On Windows a virus scanner or indexer can hold a handle on a just-written
    folder for a moment, and the rename then fails with access denied or a
    sharing violation. Such a failure is retried with backoff, about six
    seconds in all, while the target does not exist. A target that exists
    meanwhile, published by another process, ends the retries with the error,
    so the caller uses that capture.
    """
    delay = _RENAME_FIRST_DELAY
    for attempt in range(_RENAME_ATTEMPTS):
        try:
            os.rename(temporary, target)
            return
        except PermissionError as error:
            transient = _WINDOWS and getattr(error, "winerror", None) in _TRANSIENT_WINERRORS
            if not transient or attempt == _RENAME_ATTEMPTS - 1 or target.exists():
                raise
        time.sleep(delay)
        delay *= 2


# Removing a folder whose files another program holds can also fail with "delete pending" (145).
_TRANSIENT_DELETE_WINERRORS = frozenset((5, 32, 145))


def _discard_capture(temporary: Path) -> None:
    """Remove a capture's temporary folder without ever failing the capture.

    The capture's outcome is already decided when this runs: the verified
    capture is in place, or an error is on its way to the caller. On Windows
    a removal that another program's handle blocks (WinError 5, 32 or 145)
    is retried with backoff, about six seconds in all. A removal that still
    fails only logs a warning and leaves the ``.capture-*`` folder in the
    store root, where nothing reads it.
    """
    delay = _RENAME_FIRST_DELAY
    for attempt in range(_RENAME_ATTEMPTS):
        try:
            _discard(temporary)
            return
        except OSError as error:
            transient = _WINDOWS and getattr(error, "winerror", None) in _TRANSIENT_DELETE_WINERRORS
            if not transient or attempt == _RENAME_ATTEMPTS - 1:
                _logger.warning("Could not remove the temporary module capture folder %s (%s); it is left in place", temporary, error)
                return
        time.sleep(delay)
        delay *= 2


def _every_file(name: str) -> bool:
    return True


def _read_package_tree(root: Path) -> _Tree:
    """Read one filesystem package-resource folder, reusing an unchanged earlier reading.

    Package folders are read for every module setup. An earlier reading is
    returned only while the folder's fingerprint is the same as when it was
    read; any change to the folder, or a folder too new for its timestamps to
    show one, is read in full with every check of `_read_tree`. On Windows the
    fingerprint also holds a digest of every file's bytes, because a stat
    there cannot show an in-place rewrite that keeps the size and puts the
    modification time back.
    """
    key = str(root.absolute())
    signature = tree_signature(root, ignore=_ignored_name, contents=_every_file)
    if signature is not None:
        with _MEMO_LOCK:
            cached = _RESOURCE_TREES.get(key)
        if cached is not None and cached[0] == signature and not is_link(root):
            return cached[1]
    tree = _read_tree(root)
    if signature is not None and tree_signature(root, ignore=_ignored_name, contents=_every_file) == signature:
        with _MEMO_LOCK:
            _remember(_RESOURCE_TREES, key, (signature, tree))
    return tree


def _read_resource_tree_once(root: Traversable) -> _Tree:
    """Read one deterministic snapshot from a trusted package resource."""
    if not root.is_dir():
        raise ValueError(f"module artifact resource root must be a directory: {root}")
    entries = [_TreeEntry(".", "directory")]
    pending: list[tuple[tuple[str, ...], Traversable]] = [((), root)]
    while pending:
        prefix, directory = pending.pop()
        children = tuple(child for child in directory.iterdir() if not _ignored_name(child.name))
        names = tuple(child.name for child in children)
        if not all(isinstance(name, str) for name in names):
            raise TypeError("module artifact resource names must be strings")
        if len(set(names)) != len(names):
            raise ValueError(f"module artifact resource repeats an entry below {'/'.join(prefix) or '<root>'}")
        for child in sorted(children, key=lambda item: item.name, reverse=True):
            path = _relative((*prefix, child.name))
            _validate_name(path, child.name)
            is_dir = child.is_dir()
            is_file = child.is_file()
            if is_dir == is_file:
                raise ValueError(f"module artifact resource is not a regular file or directory: {path}")
            if is_dir:
                entries.append(_TreeEntry(path, "directory"))
                pending.append(((*prefix, child.name), child))
            else:
                content = child.read_bytes()
                if not isinstance(content, bytes):
                    raise TypeError(f"module artifact resource returned non-bytes content: {path}")
                entries.append(_TreeEntry(path, "file", content))
    ordered = tuple(sorted(entries, key=lambda entry: entry.path))
    kinds = {entry.path: entry.kind for entry in ordered}
    for required in ("__init__.py", "meta.yaml"):
        if kinds.get(required) != "file":
            raise ValueError(f"module artifact root requires a regular {required}")
    return _Tree(ordered, _tree_digest(ordered))


def _read_resource_tree(root: Traversable) -> _Tree:
    """Read two matching trusted-resource snapshots with bounded retries."""
    previous: _Tree | None = None
    for _ in range(_CAPTURE_ATTEMPTS):
        current = _read_resource_tree_once(root)
        if previous is not None and current == previous:
            return current
        previous = current
    raise ValueError(f"module artifact resource changed repeatedly while captured: {root}")


def _freeze(path: Path) -> None:
    root = Path(pj(path, abs=True))
    for candidate in enum_files(root, abs=True):
        Path(candidate).chmod(0o444)
    for candidate in sorted(enum_dirs(root, abs=True), key=lambda item: len(Path(item).parts), reverse=True):
        Path(candidate).chmod(0o555)
    root.chmod(0o555)


def _discard(path: Path) -> None:
    if not exists_path(path):
        return
    root = Path(pj(path, abs=True))
    for candidate in enum_files(root, abs=True):
        Path(candidate).chmod(0o644)
    for candidate in enum_dirs(root, abs=True):
        Path(candidate).chmod(0o755)
    root.chmod(0o755)
    delete_dir(root)


def _write_tree(root: Path, tree: _Tree) -> None:
    root.mkdir(parents=True)
    for entry in tree.entries:
        if entry.path == ".":
            continue
        destination = root / entry.path
        if entry.kind == "directory":
            destination.mkdir()
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as stream:
                stream.write(entry.content)


def _clean_module(module: str | None) -> str:
    if module is None or module == "":
        return ""
    if not isinstance(module, str):
        raise TypeError("artifact module must be a string or None")
    if not _MODULE_RE.fullmatch(module):
        raise ValueError(f"artifact module must be a relative dotted module name, got {module!r}")
    return module


def _clean_identity(coordinate: str, generation: str | int) -> tuple[str, bytes, bytes]:
    if not isinstance(coordinate, str):
        raise TypeError("artifact coordinate must be a string")
    clean_coordinate = coordinate.strip()
    if not clean_coordinate:
        raise ValueError("artifact coordinate must be non-empty")
    if isinstance(generation, bool) or not isinstance(generation, (str, int)):
        raise TypeError("artifact generation must be a non-empty string or non-negative integer")
    if isinstance(generation, int):
        if generation < 0:
            raise ValueError("artifact generation must be non-negative")
        return clean_coordinate, b"integer", str(generation).encode("ascii")
    clean_generation = generation.strip()
    if not clean_generation:
        raise ValueError("artifact generation must be non-empty")
    return clean_coordinate, b"text", clean_generation.encode("utf-8")


def _namespace(coordinate: str, generation: str | int, artifact_digest: str) -> str:
    clean_coordinate, generation_kind, generation_value = _clean_identity(coordinate, generation)
    digest = sha256()
    _frame(digest, b"format", b"heavenbase-module-generation-v1")
    _frame(digest, b"coordinate", clean_coordinate.encode("utf-8"))
    _frame(digest, b"generation-kind", generation_kind)
    _frame(digest, b"generation", generation_value)
    _frame(digest, b"artifact-digest", artifact_digest.encode("ascii"))
    return f"_heavenbase_artifact_{digest.hexdigest()}"


def _clean_digest(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("artifact digest must be a string")
    if not _DIGEST_RE.fullmatch(value):
        raise ValueError(f"artifact digest must be a lowercase SHA-256 digest, got {value!r}")
    return value


def _guard_namespace(namespace: str) -> None:
    names = [name for name in tuple(sys.modules) if name == namespace or name.startswith(f"{namespace}.")]
    for name in names:
        module = sys.modules.get(name)
        if module is None:
            continue
        loader = getattr(getattr(module, "__spec__", None), "loader", None)
        if not isinstance(loader, _ArtifactLoader) or loader.generation.namespace != namespace:
            raise ImportError(f"private artifact namespace is already occupied: {name}")


class ModuleArtifactStore:
    """Capture and load immutable ADR 0012 root-package folders.

    The store hashes and copies bytes without parsing `meta.yaml`. Loading
    imports only from an in-memory snapshot whose bytes matched the digest,
    under a private generation namespace. It reads and verifies the complete
    captured tree before any code runs, unless the process verified that
    capture before and its stat fingerprint is unchanged since; then it serves
    those verified bytes without reading the capture again. On Windows the
    fingerprint cannot show an in-place rewrite that keeps a file's size and
    puts its modification time back, so such a damaged capture still loads its
    verified bytes while the process remembers the capture: the memos of
    verified captures and of their bytes each keep the 512 most recently recorded
    entries, and a capture that drops out of either is read and verified again.

    Args:
        root (str | Path): Filesystem root for content-addressed captures.
    """

    def __init__(self, root: str | Path) -> None:
        """Initialize the content-addressed capture root directory."""
        if not isinstance(root, (str, Path)):
            raise TypeError("module artifact store root must be a string or Path")
        self.root = _long_path(Path(root).expanduser().resolve(strict=False))
        self._lock = threading.RLock()

    def capture(self, source: str | Path) -> ModuleArtifact:
        """Capture one root-package folder without parsing or executing it.

        Args:
            source (str | Path): Folder containing regular `meta.yaml` and
                `__init__.py` files.

        Returns:
            ModuleArtifact: Content digest and immutable captured root path.

        Raises:
            FileNotFoundError: If the source folder does not exist.
            OSError: If the source or store cannot be read or written.
            TypeError: If `source` has an invalid type.
            ValueError: If the retained tree contains a symlink, special file,
                escaping path, or lacks the required root files. Interpreter
                cache entries are ignored.
            SourceUnavailable: If an existing object for the same digest is
                corrupt or unavailable.
        """
        if not isinstance(source, (str, Path)):
            raise TypeError("module artifact source must be a string or Path")
        tree = _read_tree(Path(source).expanduser())
        return self._capture_tree(tree)

    def capture_resource(self, source: Traversable) -> ModuleArtifact:
        """Capture one trusted package-resource module folder.

        Filesystem-backed resources retain the symlink, special-file, and
        matching-content-digest checks of `capture`. Other `Traversable`
        providers are accepted only for package resources selected by trusted
        discovery. External and otherwise untrusted folders must use `capture`.

        Args:
            source: Package-resource folder containing `meta.yaml` and
                `__init__.py`.

        Returns:
            ModuleArtifact: Content digest and immutable captured root path.

        Raises:
            OSError: If the resource or store cannot be read or written.
            TypeError: If `source`, its names, or file payloads are invalid.
            ValueError: If the resource changes, contains invalid entries, or
                lacks the required root files.
            SourceUnavailable: If an existing object for the same digest is
                corrupt or unavailable.
        """
        if not isinstance(source, Traversable):
            raise TypeError("module artifact package resource must implement Traversable")
        if isinstance(source, Path):
            return self._capture_tree(_read_package_tree(Path(source).expanduser()))
        return self._capture_tree(_read_resource_tree(source))

    def _capture_tree(self, tree: _Tree) -> ModuleArtifact:
        """Commit one validated tree to the content-addressed store."""
        artifact = ModuleArtifact(tree.digest, self._artifact_root(tree.digest))
        with self._lock:
            self._ensure_root()
            if artifact.root.parent.exists():
                self._snapshot(artifact)
                return artifact
            temporary = Path(tempfile.mkdtemp(prefix=".capture-", dir=self.root))
            try:
                _write_tree(temporary / "root", tree)
                _freeze(temporary / "root")
                artifact.root.parent.parent.mkdir(parents=True, exist_ok=True)
                self._check_store_path(artifact.root.parent.parent)
                try:
                    _rename_capture(temporary, artifact.root.parent)
                except OSError:
                    if not artifact.root.parent.exists():
                        raise
                    self._snapshot(artifact)
                else:
                    artifact.root.parent.chmod(0o555)
                    self._snapshot(artifact)
                return artifact
            finally:
                _discard_capture(temporary)

    def get(self, digest: str) -> ModuleArtifact:
        """Open one verified capture by its content digest.

        Reconstructing captures from the configured store root keeps absolute
        machine paths out of durable module descriptors.

        Args:
            digest (str): Lowercase SHA-256 artifact digest.

        Returns:
            ModuleArtifact: Verified capture identity and local root locator.

        Raises:
            SourceUnavailable: If the capture is missing, corrupt, unreadable,
                or escapes the store.
            TypeError: If `digest` is not a string.
            ValueError: If `digest` is not a lowercase SHA-256 digest.
        """
        if not isinstance(digest, str):
            raise TypeError("artifact digest must be a string")
        artifact = ModuleArtifact(digest, self._artifact_root(digest))
        self._snapshot(artifact)
        return artifact

    def load(
        self,
        artifact: ModuleArtifact,
        module: str | None = None,
        *,
        coordinate: str,
        generation: str | int,
        lifecycle_owner: object | None = None,
    ) -> ModuleType:
        """Load one artifact-relative source module from verified bytes.

        The root package is selected by leaving `module` as `None`; nested
        modules use a relative dotted name such as `handler.runtime`. The
        coordinate, installation generation, and content digest jointly choose
        a deterministic private package namespace.

        Args:
            artifact (ModuleArtifact): Capture returned by this store.
            module (str | None): Artifact-relative dotted module, or `None` for
                the root package.
            coordinate (str): Stable installation coordinate.
            generation (str | int): Installation generation. Integers must be
                non-negative; strings must be non-empty.
            lifecycle_owner (object | None): Weak-referenceable consumer that
                owns this runtime generation. Defaults to this store.

        Returns:
            ModuleType: Loaded private package or nested source module.

        Raises:
            ImportError: If the requested module is absent, unsupported, or
                collides with an occupied private namespace.
            SourceUnavailable: If capture bytes are missing, unreadable, or do
                not match the recorded digest before execution begins. A
                capture this process verified before is read again only after
                its stat fingerprint changed.
            TypeError: If an argument has an invalid type.
            ValueError: If an identity or module name has an invalid value.

        Notes:
            Exceptions raised by module code propagate unchanged. In
            particular, they are never translated to `SourceUnavailable`.
        """
        clean_module = _clean_module(module)
        tree = self._snapshot(artifact)
        namespace = _namespace(coordinate, generation, artifact.digest)
        candidate = _Generation(namespace, artifact.digest, tree.files, tree.directories)
        source = candidate.source(clean_module)
        if source is None:
            raise ImportError(f"artifact has no source module {clean_module or '<root>'!r}")
        owner = self if lifecycle_owner is None else lifecycle_owner
        _install_finder()
        loaded_generation = _FINDER.register(candidate, owner)
        succeeded = False
        try:
            _guard_namespace(namespace)
            fullname = namespace if not clean_module else f"{namespace}.{clean_module}"
            loaded = import_module(fullname)
            loader = getattr(getattr(loaded, "__spec__", None), "loader", None)
            if not isinstance(loader, _ArtifactLoader) or loader.generation is not loaded_generation:
                raise ImportError(f"artifact module {fullname!r} escaped its private loader")
            succeeded = True
            return loaded
        finally:
            _FINDER.finish_load(namespace, owner, succeeded=succeeded)

    def resolve(
        self,
        artifact: ModuleArtifact,
        module: str | None,
        qualname: str,
        *,
        coordinate: str,
        generation: str | int,
        lifecycle_owner: object | None = None,
    ) -> Any:
        """Resolve one qualified object from an artifact-relative module.

        Args:
            artifact (ModuleArtifact): Capture returned by this store.
            module (str | None): Artifact-relative dotted module, or `None` for
                the root package.
            qualname (str): Dotted top-level attribute path inside the module.
            coordinate (str): Stable installation coordinate.
            generation (str | int): Installation generation.
            lifecycle_owner (object | None): Weak-referenceable consumer that
                owns this runtime generation. Defaults to this store.

        Returns:
            Any: Resolved runtime object.

        Raises:
            ImportError: If the source module or qualified object is absent or
                unsupported.
            SourceUnavailable: If capture verification fails before execution.
            TypeError: If an argument has an invalid type.
            ValueError: If an identity, module, or qualified name is invalid.
        """
        if not isinstance(qualname, str):
            raise TypeError("artifact qualname must be a string")
        if not _QUALNAME_RE.fullmatch(qualname) or "<locals>" in qualname:
            raise ValueError(f"artifact qualname must be a dotted top-level attribute path, got {qualname!r}")
        value: Any = self.load(
            artifact,
            module,
            coordinate=coordinate,
            generation=generation,
            lifecycle_owner=lifecycle_owner,
        )
        try:
            for part in qualname.split("."):
                value = getattr(value, part)
        except AttributeError as error:
            raise ImportError(f"artifact import target {qualname!r} does not exist in {module or '<root>'!r}") from error
        return value

    def release_unreferenced(
        self,
        identities: Iterable[tuple[str, str | int, str]],
        *,
        lifecycle_owner: object | None = None,
    ) -> tuple[str, ...]:
        """Release owned runtime generations absent from `identities`.

        Each identity is `(coordinate, generation, artifact_digest)`. Collection
        unregisters verified in-memory bytes and only the matching private
        modules loaded by this artifact runtime. Immutable on-disk captures are
        deliberately retained because other durable catalogs may reference the
        same content-addressed object.

        Args:
            identities: Artifact installation generations still reachable from
                the owner's authoritative catalog.
            lifecycle_owner: Consumer supplied to `load` or `resolve`. Defaults
                to this store.

        Returns:
            tuple[str, ...]: Private namespaces collected after their final
                owner released them.
        """
        if isinstance(identities, (str, bytes)) or not isinstance(identities, Iterable):
            raise TypeError("artifact identities must be an iterable of 3-tuples")
        retained: set[str] = set()
        for identity in identities:
            if not isinstance(identity, tuple) or len(identity) != 3:
                raise TypeError("each artifact identity must be a (coordinate, generation, digest) tuple")
            coordinate, generation, digest = identity
            retained.add(_namespace(coordinate, generation, _clean_digest(digest)))
        owner = self if lifecycle_owner is None else lifecycle_owner
        with self._lock:
            return _FINDER.release_except(owner, frozenset(retained))

    def release_all(self, *, lifecycle_owner: object | None = None) -> tuple[str, ...]:
        """Release every runtime generation retained by one lifecycle owner.

        Args:
            lifecycle_owner (object | None): Weak-referenceable consumer that
                owns loaded artifact generations. Defaults to this store.

        Returns:
            tuple[str, ...]: Private namespaces collected after release.
        """
        owner = self if lifecycle_owner is None else lifecycle_owner
        with self._lock:
            return _FINDER.release_except(owner, frozenset())

    def _artifact_root(self, digest: str) -> Path:
        return self.root / "objects" / "sha256" / digest[:2] / digest / "root"

    def _ensure_root(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        if is_link(self.root) or not exists_dir(self.root):
            raise ValueError(f"module artifact store root must be a regular directory: {self.root}")

    def _check_store_path(self, path: Path) -> None:
        try:
            relative = path.relative_to(self.root)
        except ValueError as error:
            raise SourceUnavailable(f"artifact path escapes its content-addressed store: {path}") from error
        current = self.root
        try:
            candidates = [current]
            for part in relative.parts:
                current /= part
                candidates.append(current)
            for candidate in candidates:
                if not exists_path(candidate):
                    raise SourceUnavailable(f"artifact store path is unavailable: {path}")
                if is_link(candidate) or not exists_dir(candidate):
                    raise SourceUnavailable(f"artifact store path is not a regular directory: {candidate}")
        except OSError as error:
            raise SourceUnavailable(f"artifact store path is unavailable: {path}: {error}") from error

    def _snapshot(self, artifact: ModuleArtifact) -> _Tree:
        if not isinstance(artifact, ModuleArtifact):
            raise TypeError("artifact must be a ModuleArtifact")
        if not _DIGEST_RE.fullmatch(artifact.digest):
            raise ValueError(f"artifact digest must be a lowercase SHA-256 digest, got {artifact.digest!r}")
        if not isinstance(artifact.root, Path):
            raise TypeError("artifact root must be a Path")
        expected = self._artifact_root(artifact.digest)
        if artifact.root.absolute() != expected.absolute():
            raise SourceUnavailable(f"artifact root escapes its content-addressed store: {artifact.root}")
        self._check_store_path(artifact.root)
        # A capture whose fingerprint is unchanged since its bytes last matched
        # the digest is served from those verified bytes; any change the
        # fingerprint shows is read and verified in full (see `tree_signature`
        # for what Windows cannot show). Code always runs from verified bytes.
        key = str(artifact.root)
        signature = tree_signature(artifact.root, ignore=_ignored_name, settled=False)
        if signature is not None:
            with _MEMO_LOCK:
                remembered = _VERIFIED_CAPTURES.get(key) == (artifact.digest, signature)
                verified = _CAPTURE_TREES.get(artifact.digest) if remembered else None
            if verified is not None:
                return verified
        try:
            tree = _read_tree(artifact.root)
        except (OSError, UnicodeError, ValueError) as error:
            raise SourceUnavailable(f"artifact source is unavailable: {artifact.root}: {error}") from error
        if tree.digest != artifact.digest:
            raise SourceUnavailable(f"artifact digest mismatch for {artifact.root}: expected {artifact.digest}, got {tree.digest}")
        if signature is not None and tree_signature(artifact.root, ignore=_ignored_name, settled=False) == signature:
            with _MEMO_LOCK:
                _remember(_CAPTURE_TREES, artifact.digest, tree)
                _remember(_VERIFIED_CAPTURES, key, (artifact.digest, signature))
        return tree
