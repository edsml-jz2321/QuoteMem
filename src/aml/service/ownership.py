"""Exclusive workspace ownership and boot-time storage capability checks.

The caller prepares a private directory before acquisition and closes all
listeners, work and database connections before releasing this owner. Filesystem
checks complement, but do not replace, the real HB runtime/keeper startup gate.
"""

from __future__ import annotations

import ctypes
import json
import os
import re
import sqlite3
import stat
import subprocess
import sys
import time
import uuid
import warnings
from collections.abc import Callable
from dataclasses import dataclass, replace
from importlib import import_module
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Literal, Self, cast

from aml.core.errors import AMLError
from aml.core.native_stat import (
    STANDARD_LIBRARY,
    IdentityReader,
    NativeReader,
    StatIdentity,
)
from aml.core.platform_files import (
    link_at,
    list_directory,
    lock_exclusive,
    open_directory,
    open_file_at,
    stat_at,
    sync_descriptor,
    unlink_at,
    unlink_probe_at,
    unlock_file,
    validate_private_descriptor,
)
from aml.store.hb_runtime import OWNER_LOCK_NAME, HBRuntime

if TYPE_CHECKING:
    from aml.core.existing_job import ExistingJobResources

Profile = Literal["solo", "team", "bench", "enterprise"]
DATABASES = ("memory.db", "hb-system.db")
LOCK_NAME = OWNER_LOCK_NAME
if os.name != "nt":
    import fcntl

    _FLAGS = os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
else:
    _FLAGS = 0  # Native open_file_at implements no-follow and noninheritance.
_NETWORK_FILESYSTEMS = frozenset(
    {
        "nfs",
        "nfs4",
        "smb",
        "smbfs",
        "cifs",
        "afp",
        "afpfs",
        "sshfs",
        "webdav",
        "davfs",
        "davfs2",
        "ceph",
        "ceph-fuse",
        "glusterfs",
        "lustre",
        "9p",
        "afs",
        "gfs",
        "gfs2",
        "ocfs2",
        "s3fs",
        "rclone",
    }
)


class OwnershipError(RuntimeError):
    """A startup or lifecycle ownership check failed; service must stay unhealthy."""


@dataclass(frozen=True)
class StorageCapabilities:
    """Observed boot probe results, not a real HB or milestone acceptance claim.

    ``lock_child_seconds`` holds each independent lock child's wall time from
    spawn to reaped exit, in ``independent_processes`` order.
    """

    filesystem: str
    sqlite_version: str
    file_sync: str
    directory_sync: str
    independent_processes: tuple[int, int]
    lock_child_seconds: tuple[float, float]
    tamper_checks: tuple[str, ...] = ("hardlink", "replacement", "unlink")


_Identity = StatIdentity


@dataclass(frozen=True)
class _Pinned:
    name: str
    fd: int
    identity: _Identity


def _regular_identity(fd: int, *, companion: bool = False) -> _Identity:
    if os.name == "nt":
        try:
            validate_private_descriptor(fd)
        except AMLError as error:
            raise OwnershipError("Storage file must have a private native identity") from error
        return _Identity.read(os.fstat(fd))
    identity = _Identity.read(os.fstat(fd))
    mode_valid = identity.mode & ~0o600 == 0 if companion else identity.mode == 0o600
    if (
        identity.kind != stat.S_IFREG
        or identity.links != 1
        or identity.owner != os.geteuid()
        or not mode_valid
    ):
        raise OwnershipError("Storage file must be private, singly linked and owned by this user")
    return identity


def _pin(parent_fd: int, name: str, *, create: bool = False, companion: bool = False) -> _Pinned:
    flags = os.O_RDONLY | _FLAGS
    if create:
        flags = os.O_CREAT | os.O_EXCL | os.O_RDWR | _FLAGS
    fd = open_file_at(parent_fd, name, flags)
    try:
        result = _Pinned(name, fd, _regular_identity(fd, companion=companion))
        _check(parent_fd, result)
        return result
    except BaseException:
        os.close(fd)
        raise


def _check(
    parent_fd: int, pinned: _Pinned, reader: IdentityReader = STANDARD_LIBRARY,
) -> None:
    if os.name == "nt":
        try:
            validate_private_descriptor(pinned.fd, single_link=False)
        except AMLError as error:
            raise OwnershipError("Storage file permissions changed") from error
    # Plan section 2 rechecks both the retained descriptor and the no-follow
    # path state of every pinned object at each Writer critical section. The
    # owner passes the reader it verified at start; it keeps the interpreter
    # lock across both reads (ADR 0038).
    current = reader.descriptor(pinned.fd)
    path = _Identity.read(stat_at(parent_fd, pinned.name)) if os.name == "nt" else (
        reader.anchored(parent_fd, pinned.name)
    )
    if current != pinned.identity or path != pinned.identity:
        raise OwnershipError("Storage file identity, links or permissions changed")


class _DarwinStatFS(ctypes.Structure):
    # Darwin's 64-bit statfs ABI: 72-byte scalar prefix, 16-byte filesystem name,
    # two MAXPATHLEN(1024) names and eight reserved/extension uint32 fields.
    _fields_ = [
        ("prefix", ctypes.c_byte * 72),
        ("fstypename", ctypes.c_char * 16),
        ("tail", ctypes.c_byte * 2080),
    ]
    fstypename: bytes


def _filesystem_name(parent_fd: int, root: Path) -> str:
    if os.name == "nt":
        file, runtime, api = (import_module(name) for name in ("win32file", "msvcrt", "win32api"))
        located = file.GetFinalPathNameByHandle(runtime.get_osfhandle(parent_fd), 0)
        volume = file.GetVolumePathName(located)
        if file.GetDriveType(volume) == 4:
            return "smb"
        return api.GetVolumeInformation(volume)[4].lower()
    if sys.platform == "darwin":
        library = ctypes.CDLL(None, use_errno=True)
        function = library.fstatfs
        function.argtypes = [ctypes.c_int, ctypes.POINTER(_DarwinStatFS)]
        function.restype = ctypes.c_int
        result = _DarwinStatFS()
        if function(parent_fd, ctypes.byref(result)) != 0:
            raise OSError(ctypes.get_errno(), "Cannot classify workspace filesystem")
        name = result.fstypename.decode("ascii", errors="strict")
        if not name:
            raise OwnershipError("Empty native filesystem identity")
        return name.lower()
    if sys.platform == "linux":
        device = os.fstat(parent_fd).st_dev
        device_name = f"{os.major(device)}:{os.minor(device)}"
        matches: list[tuple[int, str]] = []
        for line in Path("/proc/self/mountinfo").read_text().splitlines():
            fields = line.split()
            if len(fields) < 10 or "-" not in fields:
                raise OwnershipError("Malformed Linux mount metadata")
            separator = fields.index("-")
            mount = Path(re.sub(r"\\([0-7]{3})", lambda m: chr(int(m[1], 8)), fields[4]))
            if fields[2] == device_name and root.is_relative_to(mount):
                matches.append((len(mount.parts), fields[separator + 1].lower()))
        if not matches:
            raise OwnershipError("Workspace filesystem has no matching mount identity")
        return max(matches, key=lambda row: row[0])[1]
    raise OwnershipError("Storage ownership supports Linux and Darwin platform adapters only")


def _file_sync(fd: int) -> str:
    if os.name == "nt":
        sync_descriptor(fd)
        return "FlushFileBuffers"
    os.fsync(fd)
    if sys.platform == "darwin":
        fcntl.fcntl(fd, fcntl.F_FULLFSYNC)
        return "fsync+F_FULLFSYNC"
    return "fsync"


# Allow bounded interpreter/module startup as well as the nonblocking lock call.
_LOCK_CHILD_TIMEOUT_SECONDS = 15.0
# Bound collection after a timeout kill, so a failure never waits on a survivor.
_LOCK_CHILD_DRAIN_SECONDS = 5.0

# The result is flushed at once, so a timeout shows whether the child got that far.
_LOCK_CHILD = """
import json, os, sys
from pathlib import Path
from aml.core.platform_files import open_regular_file, lock_exclusive
try:
    fd = open_regular_file(Path(sys.argv[1]), os.O_RDWR)
    value = os.fstat(fd)
    try:
        lock_exclusive(fd)
        outcome = "acquired"
    except BlockingIOError:
        outcome = "blocked"
    print(json.dumps([outcome, os.getpid(), value.st_dev, value.st_ino]), flush=True)
    os.close(fd)
except OSError:
    sys.exit(2)
"""


def _kill_and_drain(
    process: subprocess.Popen[bytes],
) -> tuple[bytes | None, bytes | None] | None:
    """Kill the direct child, then collect its output and reap it within the drain limit."""
    process.kill()
    try:
        return process.communicate(timeout=_LOCK_CHILD_DRAIN_SECONDS)
    except subprocess.TimeoutExpired:
        return None


def _child_lock(root: Path, pinned: _Pinned, expected: str) -> tuple[int, float]:
    """Return one independent lock child's PID and its spawn-to-reaped-exit seconds."""
    # No owner/probe FD is inherited. The -c program starts no process, but in
    # a Windows venv sys.executable is a launcher that runs the base
    # interpreter as its own child in a kill-on-close job: killing the launcher
    # ends the interpreter only once that job closes, and the pipes reach EOF
    # only after both have exited. A POSIX interpreter is the direct child.
    started = time.monotonic()
    process = subprocess.Popen(
        [sys.executable, "-I", "-B", "-c", _LOCK_CHILD, str(root / pinned.name)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        close_fds=True,
        env={**os.environ, "LC_ALL": "C", "PYTHONUTF8": "1", "PYTHONNOUSERSITE": "1",
             "AML_ACCEPTANCE_REPORT": str(root / (".probe-" + uuid.uuid4().hex + ".json"))},
    )
    spawned = time.monotonic()
    try:
        stdout, stderr = process.communicate(timeout=_LOCK_CHILD_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired as error:
        expired = time.monotonic()
        drained = _kill_and_drain(process)
        finished = time.monotonic()
        result_line = "unknown"
        if drained is not None:
            error.output, error.stderr = drained
            result_line = "yes" if b"\n" in (drained[0] or b"") else "no"
        # Content-free facts only: creation, the deadline wait and the post-kill
        # drain are timed separately and sum to elapsed.
        raise OwnershipError(
            f"Independent lock probe timed out (phase={expected}, "
            f"deadline={_LOCK_CHILD_TIMEOUT_SECONDS:g}s, elapsed={finished - started:.3f}s, "
            f"spawn={spawned - started:.3f}s, waited={expired - spawned:.3f}s, "
            f"drain={finished - expired:.3f}s, drained={'no' if drained is None else 'yes'}, "
            f"result_line={result_line})"
        ) from error
    except BaseException:
        _kill_and_drain(process)
        raise
    elapsed = time.monotonic() - started
    if process.returncode != 0 or stderr or len(stdout) > 1024:
        raise OwnershipError("Independent lock probe did not complete cleanly")
    decoded: object = json.loads(stdout)
    if not isinstance(decoded, list):
        raise OwnershipError("Independent lock probe returned a non-list result")
    payload = cast(list[object], decoded)
    if (
        len(payload) != 4
        or payload[0] != expected
        or type(payload[1]) is not int
        or payload[1] == os.getpid()
        or payload[2:] != [pinned.identity.device, pinned.identity.inode]
    ):
        raise OwnershipError("Independent lock probe returned the wrong identity or result")
    return payload[1], elapsed


def _unlink_owned(parent_fd: int, pinned: _Pinned) -> None:
    try:
        actual = _Identity.read(stat_at(parent_fd, pinned.name))
    except FileNotFoundError:
        return
    if actual != pinned.identity:
        raise OwnershipError("Probe cleanup refused a changed pathname")
    _check(parent_fd, pinned)
    unlink_probe_at(
        parent_fd, pinned.name, (pinned.identity.device, pinned.identity.inode),
        pinned.identity.links,
    )


def _purge_tree(parent_fd: int, device: int, *, remove: bool, keep_lock: bool = False) -> None:
    """Remove owned entries beneath the pinned private root without following links."""
    if os.name == "nt":
        _purge_windows_tree(parent_fd, device, remove=remove, keep_lock=keep_lock)
        return
    for name in os.listdir(parent_fd):
        if keep_lock and name == LOCK_NAME:
            continue
        before = _Identity.read(os.stat(name, dir_fd=parent_fd, follow_symlinks=False))
        if before.device != device or before.owner != os.geteuid():
            raise OwnershipError("Purge refused another owner or filesystem")
        if before.kind == stat.S_IFDIR:
            if before.mode & 0o7022 or before.mode & 0o500 != 0o500:
                raise OwnershipError("Purge directory has unsafe permissions")
            child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | _FLAGS, dir_fd=parent_fd)
            try:
                if _Identity.read(os.fstat(child)) != before:
                    raise OwnershipError("Purge directory identity changed")
                if remove and not before.mode & 0o200:
                    # HB retains immutable module copies beneath the private
                    # workspace. Only terminal deletion, after runtime close,
                    # may make these owned copies writable via their pinned FD.
                    if _Identity.read(
                        os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
                    ) != before:
                        raise OwnershipError("Purge directory was replaced")
                    os.fchmod(child, before.mode | 0o200)
                    before = replace(before, mode=before.mode | 0o200)
                    if _Identity.read(os.fstat(child)) != before:
                        raise OwnershipError("Purge directory permissions changed unexpectedly")
                _purge_tree(child, device, remove=remove)
                current = _Identity.read(os.stat(name, dir_fd=parent_fd, follow_symlinks=False))
                held = _Identity.read(os.fstat(child))
                if (
                    current != held
                    or held != replace(before, links=held.links)
                    or held.links <= 0
                ):
                    raise OwnershipError("Purge directory was replaced")
                if remove:
                    os.fsync(child)
                    os.rmdir(name, dir_fd=parent_fd)
            finally:
                os.close(child)
        elif before.kind == stat.S_IFREG:
            if before.links != 1 or before.mode & 0o7022:
                raise OwnershipError("Purge file has unsafe permissions or links")
            if remove:
                if _Identity.read(
                    os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
                ) != before:
                    raise OwnershipError("Purge file was replaced")
                os.unlink(name, dir_fd=parent_fd)
        else:
            raise OwnershipError("Purge refused a linked or special entry")
    if remove:
        os.fsync(parent_fd)


def _purge_windows_tree(
    parent: int, device: int, *, remove: bool, keep_lock: bool = False,
    relative: tuple[str, ...] = (),
) -> None:
    """Validate each anchored native handle before terminal identity-bound removal."""
    for name in list_directory(parent):
        if keep_lock and name == LOCK_NAME:
            continue
        before = _Identity.read(stat_at(parent, name))
        if before.device != device or before.kind not in {stat.S_IFDIR, stat.S_IFREG}:
            raise OwnershipError("Purge refused another filesystem or a special entry")
        directory = before.kind == stat.S_IFDIR
        parts = (*relative, name)
        # This relative chain is built only by recursively opening each checked
        # ancestor beneath the retained workspace handle, never from an external
        # path or a basename match. Limit CPython700 equivalence to HB captures.
        artifact = (
            len(parts) >= 6 and parts[:4] == ("artifacts", "modules", "objects", "sha256")
            and re.fullmatch(r"[0-9a-f]{64}", parts[5]) is not None
            and parts[4] == parts[5][:2]
        )
        child = open_file_at(parent, name, os.O_RDONLY, directory=directory)
        try:
            validate_private_descriptor(
                child, directory=directory, allow_cpython_private=artifact,
            )
            if _Identity.read(os.fstat(child)) != before:
                raise OwnershipError("Purge entry identity changed")
            if directory:
                _purge_windows_tree(child, device, remove=remove, relative=parts)
                if remove:
                    sync_descriptor(child)
            current = _Identity.read(stat_at(parent, name))
            if current != _Identity.read(os.fstat(child)) or current != before:
                raise OwnershipError("Purge entry identity changed")
            if remove:
                unlink_at(parent, name, (before.device, before.inode), directory=directory,
                          allow_readonly=True, allow_cpython_private=artifact)
        except AMLError as error:
            raise OwnershipError("Purge refused unsafe native ownership") from error
        finally:
            os.close(child)
    if remove:
        sync_descriptor(parent)


class WorkspaceOwner:
    """Hold the exclusive owner lock for one canonical workspace directory.

    Args:
        root: Existing directory owned by the effective user with exact 0700 mode.
        profile: Closed deployment profile; bench always rejects virtiofs.
        allow_virtiofs: Explicit product-only permission, still requiring a probe.

    The constructor does no I/O. Enter/acquire before opening HB, run the storage
    probe and configuration checks, then prepare databases. Call validate before
    healthy and at every sole-Writer business critical section. Close runtime
    pools/keepers and drain work before leaving this context. Startup/shutdown
    must be serialized: the owned process umask is 077 until close.
    """

    def __init__(self, root: Path, *, profile: Profile = "bench", allow_virtiofs: bool = False):
        if profile not in {"solo", "team", "bench", "enterprise"}:
            raise ValueError("Unsupported deployment profile")
        if type(allow_virtiofs) is not bool or (profile == "bench" and allow_virtiofs):
            raise ValueError("bench cannot enable virtiofs")
        self.root = root
        self.profile = profile
        self.allow_virtiofs = allow_virtiofs
        self._parent_fd: int | None = None
        self._parent_identity: _Identity | None = None
        self._reader: IdentityReader = STANDARD_LIBRARY
        self._files: dict[str, _Pinned] = {}
        self._pid: int | None = None
        self._umask: int | None = None
        self._failed = False
        self._closed = False
        self._prepared = False
        self._capabilities: StorageCapabilities | None = None

    def acquire(self) -> Self:
        """Acquire a private sidecar lock without opening either database.

        Returns:
            This owner, with directory and sidecar descriptors retained.

        Raises:
            OwnershipError: If the directory, lock or process state is unsafe.
        """
        if self._pid is not None or self._closed:
            raise OwnershipError("An owner instance can be acquired only once")
        self._pid = os.getpid()
        try:
            self.root = self.root.parent.resolve(strict=True) / self.root.name
            self._parent_fd = open_directory(self.root)
            self._parent_identity = _Identity.read(os.fstat(self._parent_fd))
            if os.name != "nt" and (
                self._parent_identity.kind != stat.S_IFDIR
                or self._parent_identity.owner != os.geteuid()
                or self._parent_identity.mode != 0o700
            ):
                raise OwnershipError("Workspace parent must be owned by this user with mode 0700")
            fd = open_file_at(self._parent_fd, LOCK_NAME, os.O_CREAT | os.O_RDWR)
            try:
                pinned = _Pinned(LOCK_NAME, fd, _regular_identity(fd))
                _check(self._parent_fd, pinned)
            except BaseException:
                os.close(fd)
                raise
            self._files[LOCK_NAME] = pinned
            self.validate()
            lock_exclusive(fd)
            self.validate()
            self._adopt_native_reader()
            self._umask = os.umask(0o077)
            _file_sync(fd)
            sync_descriptor(self._parent_fd)
            return self
        except BaseException as error:
            self._failed = True
            self._close_preserving(error)
            if isinstance(error, OSError | AMLError):
                raise OwnershipError("Workspace owner acquisition failed") from error
            raise

    def _adopt_native_reader(self) -> None:
        """Use the interpreter-lock-keeping reads once they match the standard library.

        Every recheck compares the parent descriptor, the root path and each
        pinned object. A reader whose platform layout is unknown, whose symbols
        are missing or whose answer differs from ``os.fstat``/``os.stat`` for
        any retained object at this point is not adopted; the owner then keeps
        the standard library reads, which are correct but each release the
        interpreter lock (ADR 0038).
        """
        if os.name == "nt" or not NativeReader.supported() or self._parent_fd is None:
            return
        fd = self._parent_fd
        try:
            reader = NativeReader.for_platform()
            matched = (
                reader.descriptor(fd) == StatIdentity.read(os.fstat(fd))
                and reader.path(self.root)
                == StatIdentity.read(os.stat(self.root, follow_symlinks=False))
                and all(
                    reader.descriptor(pinned.fd) == StatIdentity.read(os.fstat(pinned.fd))
                    and reader.anchored(fd, pinned.name)
                    == StatIdentity.read(stat_at(fd, pinned.name))
                    for pinned in self._files.values()
                )
            )
        except OSError:
            return
        if matched:
            self._reader = reader

    @property
    def native_reads(self) -> bool:
        """Whether rechecks keep the interpreter lock (the verified native reader)."""
        return self._reader is not STANDARD_LIBRARY

    def _directory(self) -> int:
        if self._closed or self._failed or self._pid != os.getpid() or self._parent_fd is None:
            raise OwnershipError("Workspace owner is closed, failed or belongs to another process")
        return self._parent_fd

    def validate(self) -> None:
        """Recheck retained identities and pin newly observed SQLite companions.

        Raises:
            OwnershipError: On any missing, linked, replaced or permission-changed
                object. A failed owner cannot later resume service.
        """
        fd = self._directory()
        try:
            if os.name == "nt":
                validate_private_descriptor(fd, directory=True)
            reader = self._reader
            held = reader.descriptor(fd)
            path = reader.path(self.root)
            # APFS increments directory nlink for ordinary entry creation. Retain
            # identity/security fields; compare the complete current FD/path tuple.
            if (
                self._parent_identity is None
                or held.links <= 0
                or (os.name != "nt" and held.owner != os.geteuid())
                or held != replace(self._parent_identity, links=held.links)
                or held != path
            ):
                raise OwnershipError("Workspace parent identity or permissions changed")
            for pinned in self._files.values():
                _check(fd, pinned, reader)
            if self._prepared:
                for database in DATABASES:
                    for suffix in ("-wal", "-shm"):
                        name = database + suffix
                        if name not in self._files:
                            try:
                                pinned = _pin(fd, name, companion=True)
                            except FileNotFoundError:
                                continue
                            self._files[name] = pinned
        except (OSError, OwnershipError, AMLError) as error:
            self._failed = True
            raise OwnershipError("Workspace lifecycle validation failed") from error

    def probe_storage(self) -> StorageCapabilities:
        """Run this boot's actual filesystem and same-stdlib SQLite capability gate.

        Returns:
            Immutable observations from two sequential independent lock children,
            including each child's wall time, and tiny temporary SQLite files.
            Unknown filesystem names still probe.

        Raises:
            OwnershipError: If classification or any capability fails. No real
                database may be prepared after a failed probe. A lock child
                timeout's cause states its content-free phase, spawn, wait and
                drain times and whether its result line arrived.
        """
        fd = self._directory()
        if self._capabilities is not None or self._prepared:
            raise OwnershipError("Storage capability probe may run only once before databases")
        try:
            self.validate()
            filesystem = _filesystem_name(fd, self.root)
            if any(part in _NETWORK_FILESYSTEMS for part in filesystem.split(".")):
                raise OwnershipError("Network or distributed filesystems are unsupported")
            if filesystem.split(".")[-1] == "virtiofs":
                if self.profile == "bench" or not self.allow_virtiofs:
                    raise OwnershipError("virtiofs requires explicit product-only permission")
                warnings.warn(
                    "Product virtiofs requires a successful boot capability probe",
                    RuntimeWarning,
                    stacklevel=2,
                )
            self._capabilities = _capability_probe(fd, self.root, self.validate, filesystem)
            self.validate()
            return self._capabilities
        except BaseException as error:
            self._failed = True
            if isinstance(error, Exception):
                raise OwnershipError("Storage capability probe failed") from error
            raise

    def prepare_databases(self, *, existing_only: bool = False) -> None:
        """After configuration/probe success, create or pin both private databases.

        Retains readonly validation descriptors for existing files; creates only
        missing files using exclusive 0600 creation unless existing_only is true.
        Never repairs files or creates companions. A preparation failure is
        terminal; any pins acquired before failure remain held until owner close.
        This does not initialize HB or verify a job binding.

        Args:
            existing_only: Require both database files to exist. Validated as a
                strict Boolean before any owner validation or filesystem access.

        Raises:
            ValueError: If existing_only is not a Boolean.
            OwnershipError: If the owner/probe or any file/companion is invalid.
        """
        if type(existing_only) is not bool:
            raise ValueError("existing_only must be a Boolean")
        fd = self._directory()
        if self._capabilities is None or self._prepared:
            raise OwnershipError("Database preparation requires one successful boot probe")
        try:
            self.validate()
            for name in DATABASES:
                created = False
                try:
                    pinned = _pin(fd, name)
                except FileNotFoundError:
                    if existing_only:
                        raise
                    pinned = _pin(fd, name, create=True)
                    created = True
                self._files[name] = pinned
                if created:
                    _file_sync(pinned.fd)
            self._prepared = True
            sync_descriptor(fd)
            self.validate()
        except BaseException as error:
            self._failed = True
            if isinstance(error, Exception):
                raise OwnershipError("Database path preparation failed") from error
            raise

    def purge(self, runtime: HBRuntime) -> None:
        """Close the borrowed runtime and terminally delete this bench workspace.

        Args:
            runtime: This owner's actual initialized HB runtime. The composition
                must first verify the exact job, withdraw admission, stop and
                join all workers/listeners, drain callbacks, clear published
                roots/memo content and collect the authoritative row counts.

        Returns:
            None only after runtime/keepers close, the complete anchored root is
            removed and synchronized, and this owner's descriptors are closed.
            This operation does not produce a receipt or attest provider deletion.

        Raises:
            OwnershipError: On a different runtime/profile, unprepared or unsafe
                owner, changed identity, or a nonprivate/linked/foreign entry.
            Exception: Runtime shutdown or filesystem failure. Runtime shutdown
                failure causes no deletion. Later failures may leave a partial
                workspace and never establish success or permit service restart.

        The caller must not release and reacquire the owner before this operation.
        After removing all data and closing its retained content descriptors,
        the sidecar is removed last while its flock remains held. This ends
        ownership of the deleted database namespace: a new owner may then claim
        an empty replacement. A new entry racing the final rmdir causes failure;
        it is never recursively retried or removed. An absent root without this
        successful operation proves neither
        deletion counts nor a completed purge. Ordinary context exit stays safe
        after this terminal operation.
        """
        if (
            self.profile != "bench" or not self._prepared
            or runtime.owner is not self or runtime.root != self.root
        ):
            raise OwnershipError("Purge requires this prepared bench owner's runtime")
        self._purge_owned(runtime.close)

    def purge_existing(self, resources: ExistingJobResources) -> None:
        """Delete an inspected closed job through the same terminal filesystem path.

        The command must bind the job and persist required external audit facts
        first. No runtime, Writer, epoch takeover or recovery is constructed.
        A live service retains the same exclusive lock and prevents this open.
        """
        from aml.core.existing_job import ExistingJobResources

        if type(resources) is not ExistingJobResources or (
            self.profile != "bench" or not self._prepared
            or resources.owner is not self or resources.root != self.root
        ):
            raise OwnershipError("Purge requires this owner's actual existing job resources")
        resources.validate_owner()

        def close() -> None:
            resources.close_storage()
            resources.require_storage_closed()

        self._purge_owned(close)

    def _purge_owned(self, close_storage: Callable[[], None]) -> None:
        """Own the single anchored deletion path after acknowledged storage disposal."""
        fd = self._directory()
        self.validate()
        identity = self._parent_identity
        if identity is None:
            raise OwnershipError("Purge has no retained root identity")
        outer = open_directory(self.root.parent, private=False)

        def check_root() -> None:
            validate_private_descriptor(fd, directory=True)
            held = _Identity.read(os.fstat(fd))
            current = _Identity.read(stat_at(outer, self.root.name))
            if (
                held != current or held.links <= 0
                or held != replace(identity, links=held.links)
            ):
                raise OwnershipError("Purge root identity changed")

        try:
            check_root()
            _purge_tree(fd, identity.device, remove=False, keep_lock=True)
            # Storage shutdown may still validate this owner. All admitted work
            # must already be drained; fence every later path before deletion,
            # including a failed or interrupted storage close.
            try:
                close_storage()
            finally:
                self._failed = True
            check_root()
            for name, pinned in self._files.items():
                try:
                    _check(fd, pinned)
                except FileNotFoundError:
                    if name not in {
                        database + suffix for database in DATABASES for suffix in ("-wal", "-shm")
                    }:
                        raise
            _purge_tree(fd, identity.device, remove=True, keep_lock=True)
            check_root()
            # No old data handle survives the terminal namespace handoff. A new
            # sidecar can appear after unlink; it cannot reopen the deleted DB.
            close_error: OSError | None = None
            for name in tuple(self._files):
                if name != LOCK_NAME:
                    pinned = self._files.pop(name)
                    try:
                        os.close(pinned.fd)
                    except OSError as error:
                        close_error = close_error or error
            if close_error is not None:
                raise close_error
            _check(fd, self._files[LOCK_NAME])
            pinned_lock = self._files[LOCK_NAME]
            unlink_at(fd, LOCK_NAME, (pinned_lock.identity.device, pinned_lock.identity.inode))
            sync_descriptor(fd)
            unlink_at(outer, self.root.name, (identity.device, identity.inode), directory=True)
            sync_descriptor(outer)
            self.close()
        finally:
            os.close(outer)

    def close(self) -> None:
        """Close validation descriptors, release the lock last, and restore umask.

        The caller must first close HB/keepers and drain work. Missing companions
        during that normal shutdown are not revalidated or unlinked here. The
        sidecar remains in place. A fork child never unlocks its parent's flock.
        """
        if self._closed:
            return
        self._closed = True
        first: OSError | None = None
        lock = self._files.pop(LOCK_NAME, None)
        descriptors = [pinned.fd for pinned in self._files.values()]
        self._files.clear()
        if self._parent_fd is not None:
            descriptors.append(self._parent_fd)
            self._parent_fd = None
        if lock is not None:
            descriptors.append(lock.fd)
        for fd in descriptors:
            try:
                if lock is not None and fd == lock.fd and self._pid == os.getpid():
                    unlock_file(fd)
            except OSError as error:
                first = first or error
            finally:
                try:
                    os.close(fd)
                except OSError as error:
                    first = first or error
        if self._umask is not None:
            os.umask(self._umask)
            self._umask = None
        if first is not None:
            raise first

    def _close_preserving(self, error: BaseException | None) -> None:
        try:
            self.close()
        except OSError as cleanup:
            if error is None:
                raise
            error.add_note(f"Owner descriptor cleanup also failed: errno {cleanup.errno}")

    def __enter__(self) -> Self:
        """Acquire this owner for an explicit service lifetime."""
        return self.acquire()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Release descriptors after caller-owned runtime shutdown."""
        self._close_preserving(exc_value)


def _capability_probe(
    fd: int, root: Path, validate: Callable[[], None], filesystem: str
) -> StorageCapabilities:
    prefix = ".aml-probe-" + uuid.uuid4().hex
    owned: list[_Pinned] = []
    error: BaseException | None = None
    connection: sqlite3.Connection | None = None
    try:
        lock = _pin(fd, prefix + "-lock", create=True)
        owned.append(lock)
        probe_bytes = b"aml-storage-probe-v1"
        if os.write(lock.fd, probe_bytes) != len(probe_bytes):
            raise OwnershipError("Storage probe write was incomplete")
        primitive = _file_sync(lock.fd)
        os.lseek(lock.fd, 0, os.SEEK_SET)
        if os.read(lock.fd, len(probe_bytes) + 1) != probe_bytes:
            raise OwnershipError("Storage probe file readback failed")
        sync_descriptor(fd)
        lock_exclusive(lock.fd)
        blocked_pid, blocked_seconds = _child_lock(root, lock, "blocked")
        _check(fd, lock)
        unlock_file(lock.fd)
        acquired_pid, acquired_seconds = _child_lock(root, lock, "acquired")
        _check(fd, lock)
        if blocked_pid == acquired_pid:
            raise OwnershipError("Lock probe did not use two independent processes")

        alias = prefix + "-alias"
        link_at(fd, lock.name, alias)
        try:
            _check(fd, lock)
        except OwnershipError:
            pass
        else:
            raise OwnershipError("Hardlink tampering was not detected")
        alias_identity = replace(lock.identity, links=2)
        _unlink_owned(fd, _Pinned(alias, lock.fd, alias_identity))
        _check(fd, lock)
        # Exclusive hardlink publication cannot overwrite a late destination.
        link_at(fd, lock.name, prefix + "-original")
        _check(fd, _Pinned(prefix + "-original", lock.fd, alias_identity))
        _unlink_owned(fd, _Pinned(lock.name, lock.fd, alias_identity))
        original = _Pinned(prefix + "-original", lock.fd, lock.identity)
        owned[-1] = original
        replacement = _pin(fd, lock.name, create=True)
        owned.append(replacement)
        try:
            _check(fd, lock)
        except OwnershipError:
            pass
        else:
            raise OwnershipError("Replacement tampering was not detected")
        _unlink_owned(fd, original)
        try:
            _check(fd, original)
        except FileNotFoundError:
            pass
        else:
            raise OwnershipError("Unlink tampering was not detected")

        database = _pin(fd, prefix + ".db", create=True)
        owned.append(database)
        validate()
        # SQLite's default VFS accepts canonical names, not portable dir_fd.
        # Our create/stat/unlink operations stay anchored; validate around VFS use.
        uri = (root / database.name).as_uri() + "?mode=rw"
        connection = sqlite3.connect(uri, uri=True, timeout=1)
        _check(fd, database)
        validate()
        if connection.execute("PRAGMA journal_mode=WAL").fetchone() != ("wal",):
            raise OwnershipError("SQLite probe did not enter WAL")
        connection.execute("CREATE TABLE probe (value TEXT NOT NULL)")
        connection.execute("INSERT INTO probe VALUES (?)", (prefix,))
        connection.commit()
        if connection.execute("SELECT value FROM probe").fetchall() != [(prefix,)]:
            raise OwnershipError("SQLite probe readback failed")
        for suffix in ("-wal", "-shm"):
            owned.append(_pin(fd, database.name + suffix, companion=True))
        checkpoint = connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        if checkpoint != (0, 0, 0):
            raise OwnershipError("SQLite probe checkpoint did not finish")
        connection.close()
        connection = None
        _check(fd, database)
        validate()
        connection = sqlite3.connect(uri, uri=True, timeout=1)
        if (
            connection.execute("PRAGMA journal_mode").fetchone() != ("wal",)
            or connection.execute("SELECT value FROM probe").fetchall() != [(prefix,)]
            or connection.execute("PRAGMA integrity_check").fetchall() != [("ok",)]
        ):
            raise OwnershipError("SQLite probe reopen or integrity check failed")
        connection.close()
        connection = None
        _file_sync(database.fd)
        sync_descriptor(fd)
        validate()
        return StorageCapabilities(
            filesystem, sqlite3.sqlite_version, primitive,
            "FlushFileBuffers" if os.name == "nt" else "fsync", (blocked_pid, acquired_pid),
            (blocked_seconds, acquired_seconds),
        )
    except BaseException as caught:
        error = caught
        raise
    finally:
        cleanup_error: Exception | None = None
        if connection is not None:
            try:
                connection.close()
            except Exception as caught:
                cleanup_error = caught
        for pinned in reversed(owned):
            try:
                _unlink_owned(fd, pinned)
            except (OSError, OwnershipError, AMLError) as caught:
                cleanup_error = cleanup_error or caught
            finally:
                try:
                    os.close(pinned.fd)
                except OSError as caught:
                    cleanup_error = cleanup_error or caught
        try:
            sync_descriptor(fd)
        except OSError as caught:
            cleanup_error = cleanup_error or caught
        if cleanup_error is not None:
            if error is None:
                raise cleanup_error
            error.add_note("Storage probe cleanup could not remove every verified temporary inode")
