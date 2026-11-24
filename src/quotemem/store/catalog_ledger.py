"""A running lifetime's ledger of whether its Catalog may lag its rows (boot step 9).

The clean-close proof (``quotemem.store.catalog_proof``) covers a lifetime that closed. A
lifetime that stops without closing (a kill or crash of the process) leaves its
databases exactly as its last completed storage calls left them: every completed write
is in the kernel's page cache, which the next process on the same kernel boot reads.
The only Catalog rows that can then lag their source rows are those of a Writer
mutation that had not finished its Catalog step, or of a failed one.

The ledger records exactly that, cheaply. Its file holds this lifetime's header (the
workspace, its Writer lease, the kernel boot and the database files' device and inode)
and one state record, rewritten in place with a single ``pwrite`` (the file is
opened for each write and never held open): ``dirty`` before the
first storage mutation of a Writer critical section, ``clean`` when that section ends
(``WriterMutex`` outermost release) with every mutation finished, and ``unproven``,
which never clears, when a mutation failed or stopped, or ran outside the Writer mutex.
No state is flushed to disk: a kernel crash or power loss changes the boot id, which
voids the ledger.

The next boot takes the ledger before any storage is opened (``take_catalog_ledger``):
it deletes the file whatever it holds and keeps the header only when the state is
``clean``, the header names this workspace and this kernel boot, the database files are
the same files, and none of them changed after the ledger's last state. The Writer then
skips the audit only when the lease its own epoch CAS replaced is the header's lease
(``Writer.recover_catalog``). Anything else, including a system without a boot id,
keeps the complete audit and repair.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import struct
import threading
import zlib
from pathlib import Path
from typing import Literal

from pydantic import ValidationError

from quotemem.core.errors import QuoteMemError
from quotemem.core.identity import canonical_json
from quotemem.core.platform_files import open_private_file
from quotemem.store.catalog_proof import DATABASE_FILES, DATABASES
from quotemem.store.metadata import StrictModel

LEDGER_NAME = "catalog-ledger.bin"
"""The ledger's file name in the workspace directory."""

BOOT_ID = Path("/proc/sys/kernel/random/boot_id")
"""The kernel's identity of its current boot (Linux); absent elsewhere."""

CLEAN, DIRTY, UNPROVEN = 1, 2, 3
"""Ledger states: no mutation in flight; a mutation in flight; never provable again."""

_MAGIC = b"AMLCLG01"
_HEADER_OFFSET = 4096
_HEADER_LIMIT = 4096
_BODY = struct.Struct("<8sB32s")
_RECORD_SIZE = _BODY.size + 4
_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_CLOEXEC = getattr(os, "O_CLOEXEC", 0)


class CatalogLedgerV1(StrictModel):
    """One lifetime's ledger header.

    Attributes:
        schema_version: 1.
        workspace_id: The workspace the ledger belongs to.
        owner_instance_id: The lifetime's Writer lease instance.
        writer_epoch: Its Writer epoch.
        boot_id: The kernel boot the lifetime ran in.
        databases: Each database file's ``[device, inode]``.
    """

    schema_version: Literal[1]
    workspace_id: str
    owner_instance_id: str
    writer_epoch: int
    boot_id: str
    databases: dict[str, list[int]]


def current_boot_id() -> str | None:
    """Return the kernel's current boot id, or None where the system has none."""
    try:
        value = BOOT_ID.read_text(encoding="ascii").strip()
    except (OSError, UnicodeDecodeError):
        return None
    return value or None


def _record(state: int, digest: bytes) -> bytes:
    body = _BODY.pack(_MAGIC, state, digest)
    return body + struct.pack("<I", zlib.crc32(body))


def _state_of(raw: bytes, digest: bytes) -> int | None:
    if len(raw) != _RECORD_SIZE or struct.unpack("<I", raw[-4:])[0] != zlib.crc32(raw[:-4]):
        return None
    magic, state, named = _BODY.unpack(raw[:-4])
    return state if magic == _MAGIC and named == digest else None


class CatalogLedger:
    """This lifetime's ledger; every state change opens the file and rewrites one record.

    The file is never held open, so closing or deleting the workspace needs nothing
    from the ledger. A private leaf lock orders state changes, so a ``clean`` record
    never overwrites an ``unproven`` one written by a mutation outside the Writer
    mutex; nothing else is acquired while it is held.
    """

    def __init__(self, path: Path, digest: bytes) -> None:
        self._path: Path | None = path
        self._digest = digest
        self._lock = threading.Lock()
        self.state = CLEAN

    def mark(self, state: int) -> None:
        """Record ``state``; ``unproven`` is final.

        Raises:
            QuoteMemError: ``service_unhealthy`` when a ``dirty`` or ``unproven`` state
                cannot be recorded and the ledger cannot be deleted: the caller's
                mutation must not proceed under a record that proves it.
        """
        with self._lock:
            if state == self.state or self.state == UNPROVEN or self._path is None:
                return
            try:
                descriptor = os.open(self._path, os.O_WRONLY | _NOFOLLOW | _CLOEXEC)
                try:
                    os.pwrite(descriptor, _record(state, self._digest), 0)
                finally:
                    os.close(descriptor)
            except FileNotFoundError:
                self.state = UNPROVEN  # no ledger left: the next boot audits
                return
            except OSError as error:
                if state == CLEAN:
                    return  # the recorded state stays dirty, which proves nothing
                try:
                    self._path.unlink(missing_ok=True)
                except OSError:
                    raise QuoteMemError(
                        "service_unhealthy", "The Catalog ledger cannot record a mutation",
                    ) from error
                self.state = UNPROVEN
                return
            self.state = state

    def close(self) -> None:
        """Stop recording; the last state stays for the next boot."""
        with self._lock:
            self._path = None


def _identity(root: Path, name: str) -> os.stat_result | None:
    try:
        return (root / name).lstat()
    except FileNotFoundError:
        return None


def start_catalog_ledger(
    root: Path, workspace_id: str, lease: tuple[str, int],
) -> CatalogLedger | None:
    """Open a proven lifetime's ledger in state ``clean``.

    Args:
        root: The workspace directory, owned by this process.
        workspace_id: The workspace binding's identity.
        lease: This lifetime's Writer lease instance and epoch.

    Returns:
        The open ledger, or None on a system without a boot id or when the file
        cannot be created; the next unclean boot then audits.
    """
    boot = current_boot_id()
    if boot is None:
        return None
    databases: dict[str, list[int]] = {}
    for name in DATABASES:
        found = _identity(root, name)
        if found is not None:
            databases[name] = [found.st_dev, found.st_ino]
    owner_instance_id, writer_epoch = lease
    header = canonical_json(CatalogLedgerV1(
        schema_version=1, workspace_id=workspace_id, owner_instance_id=owner_instance_id,
        writer_epoch=writer_epoch, boot_id=boot, databases=databases,
    )).encode()
    if len(header) > _HEADER_LIMIT:
        return None
    digest = hashlib.sha256(header).digest()
    path = root / LEDGER_NAME
    try:
        with contextlib.suppress(FileNotFoundError):
            path.unlink()
        descriptor = open_private_file(path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
    except (OSError, QuoteMemError):
        return None
    try:
        os.pwrite(descriptor, header, _HEADER_OFFSET)
        os.pwrite(descriptor, _record(CLEAN, digest), 0)
    except OSError:
        with contextlib.suppress(OSError):
            path.unlink()
        return None
    finally:
        os.close(descriptor)
    return CatalogLedger(path, digest)


def take_catalog_ledger(root: Path, workspace_id: str) -> CatalogLedgerV1 | None:
    """Take and delete the previous lifetime's ledger before any storage is opened.

    Args:
        root: The workspace directory, owned by this process.
        workspace_id: The workspace binding's identity.

    Returns:
        The header when the last state is ``clean``, it names this workspace and
        this kernel boot, every database file it names is the same file, no
        database, WAL or journal file was modified, and no database file had its
        status changed, after the ledger's last state; else None.

    Raises:
        QuoteMemError: ``service_unhealthy`` when a ledger exists but cannot be deleted.
    """
    path = root / LEDGER_NAME
    try:
        descriptor = open_private_file(path, os.O_RDONLY)
    except FileNotFoundError:
        return None
    except (OSError, QuoteMemError):
        descriptor = None  # an unsafe or unreadable ledger proves nothing; it is retired
    raw = header = b""
    written: int | None = None
    if descriptor is not None:
        try:
            written = os.fstat(descriptor).st_mtime_ns
            raw = os.pread(descriptor, _RECORD_SIZE, 0)
            header = os.pread(descriptor, _HEADER_LIMIT, _HEADER_OFFSET)
        except OSError:
            raw = b""
        finally:
            os.close(descriptor)
    files = {name: _identity(root, name) for name in DATABASE_FILES}
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except OSError as error:
        raise QuoteMemError("service_unhealthy", "A Catalog ledger could not be retired") from error
    if written is None or _state_of(raw, hashlib.sha256(header).digest()) != CLEAN:
        return None
    try:
        ledger = CatalogLedgerV1.model_validate(json.loads(header))
    except (ValidationError, ValueError):
        return None
    if ledger.workspace_id != workspace_id or ledger.boot_id != current_boot_id():
        return None
    for name, identity in ledger.databases.items():
        found = files.get(name)
        if found is None or [found.st_dev, found.st_ino] != identity:
            return None
    for name, found in files.items():
        # Written after the ledger's last state; a database file replaced or touched
        # since is caught by its status change time, which cannot be set back. SQLite
        # changes a WAL's or journal's status (ownership) when it opens it, so only
        # their content time is compared, and an empty one holds no rows: a read-only
        # connection creates it just by opening the database.
        if found is None or name not in DATABASES and found.st_size == 0:
            continue
        changed = (
            max(found.st_mtime_ns, found.st_ctime_ns) if name in DATABASES
            else found.st_mtime_ns
        )
        if changed > written:
            return None
    return ledger
