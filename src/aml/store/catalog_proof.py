"""A clean close's proof that the derived Catalog matches its rows (C.13 boot step 9).

Boot step 9 audits every Catalog row against every source row, which is linear in the
workspace. A service lifetime proves the Catalog at its own boot, then tracks every
storage mutation: one that fails, or a two-step write (main rows, then Catalog rows)
that does not finish, leaves the Catalog unproven until a complete audit proves it
again (``Writer.recover_catalog``). At a clean close the Writer seals a proven
Catalog against any further mutation (``Writer.seal_catalog_proof``); once the
storage has closed, ``leave_catalog_proof`` records ``catalog-proof.json``: the
lifetime's Writer lease (owner instance and epoch) and the identity of every database
file it closed (device, inode, size and modification time).

The next boot takes the proof before any storage is opened (``take_catalog_proof``):
it reads the file, deletes it durably whatever it says, and keeps it only when every
database file still has the identity it names. The Writer then skips the audit only
when the lease its own epoch CAS replaced is the proof's lease and none of its own boot
writes failed. No proof, an unreadable or foreign one, a file touched while the service
was down, another lifetime's lease or a failed or interrupted mutation all keep the
complete audit and repair. A process that crashes leaves no proof: its boot deleted the
previous one and its close never writes another.
"""

from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
from typing import Literal

from pydantic import ValidationError

from aml.core.errors import AMLError
from aml.core.identity import canonical_json
from aml.core.platform_files import open_private_file, sync_directory
from aml.store.metadata import StrictModel

PROOF_NAME = "catalog-proof.json"
"""The proof's file name in the workspace directory."""

_PENDING_NAME = PROOF_NAME + ".pending"

DATABASES = ("memory.db", "hb-system.db")
"""The database files proper; each also has a WAL and a journal beside it."""

DATABASE_FILES = (
    "memory.db", "memory.db-wal", "memory.db-journal",
    "hb-system.db", "hb-system.db-wal", "hb-system.db-journal",
)
"""Every file whose bytes the Catalog and its source rows live in."""

FileIdentity = tuple[int, int, int, int]


class CatalogProofV1(StrictModel):
    """One clean close's Catalog proof.

    Attributes:
        schema_version: 1.
        workspace_id: The workspace the proof belongs to.
        owner_instance_id: The closing lifetime's Writer lease instance.
        writer_epoch: Its Writer epoch.
        files: Each database file's ``[device, inode, size, mtime_ns]``, or null
            when it does not exist.
    """

    schema_version: Literal[1]
    workspace_id: str
    owner_instance_id: str
    writer_epoch: int
    files: dict[str, list[int] | None]


def storage_identity(root: Path) -> dict[str, list[int] | None]:
    """Name each database file by its device, inode, size and modification time.

    An empty WAL or journal counts as absent: it holds no rows, and a read-only SQLite
    connection creates one just by opening the database. An inspection of a stopped
    workspace therefore keeps its proof and its fast restart.

    Args:
        root: The workspace directory.

    Returns:
        Every ``DATABASE_FILES`` name with its identity, or None when it is absent.
    """
    identity: dict[str, list[int] | None] = {}
    for name in DATABASE_FILES:
        try:
            found = (root / name).lstat()
        except FileNotFoundError:
            identity[name] = None
            continue
        identity[name] = None if found.st_size == 0 and name not in DATABASES else [
            found.st_dev, found.st_ino, found.st_size, found.st_mtime_ns,
        ]
    return identity


def take_catalog_proof(root: Path, workspace_id: str) -> CatalogProofV1 | None:
    """Take and delete the previous lifetime's proof before any storage is opened.

    The file is deleted durably whatever it holds, so a later crash can never find it.

    Args:
        root: The workspace directory, owned by this process.
        workspace_id: The workspace binding's identity.

    Returns:
        The proof when it is well formed, names this workspace and every database
        file still has the identity it names; else None.

    Raises:
        AMLError: ``service_unhealthy`` when a proof exists but cannot be deleted.
    """
    path = root / PROOF_NAME
    with contextlib.suppress(FileNotFoundError):
        (root / _PENDING_NAME).unlink()
    try:
        descriptor = open_private_file(path, os.O_RDONLY)
    except FileNotFoundError:
        return None
    except (OSError, AMLError):
        # An unsafe or unreadable proof proves nothing; it is still retired.
        descriptor = None
    raw = b""
    if descriptor is not None:
        try:
            with os.fdopen(descriptor, "rb") as stream:
                raw = stream.read(65536)
        except OSError:
            raw = b""
    try:
        path.unlink()
        sync_directory(root)
    except FileNotFoundError:
        pass
    except OSError as error:
        raise AMLError("service_unhealthy", "A Catalog proof could not be retired") from error
    try:
        proof = CatalogProofV1.model_validate(json.loads(raw))
    except (ValidationError, ValueError):
        return None
    if proof.workspace_id != workspace_id or proof.files != storage_identity(root):
        return None
    return proof


def leave_catalog_proof(
    root: Path, workspace_id: str, lease: tuple[str, int],
) -> bool:
    """Record a sealed lifetime's proof after its storage has closed.

    The proof is written privately beside the workspace, flushed and renamed into
    place, so a reader finds either the complete proof or none. Any failure leaves no
    proof and the next boot audits.

    Args:
        root: The workspace directory, still owned by this process.
        workspace_id: The workspace binding's identity.
        lease: The sealed lifetime's Writer lease instance and epoch.

    Returns:
        bool: Whether the proof was recorded.
    """
    owner_instance_id, writer_epoch = lease
    body = canonical_json(CatalogProofV1(
        schema_version=1, workspace_id=workspace_id, owner_instance_id=owner_instance_id,
        writer_epoch=writer_epoch, files=storage_identity(root),
    )).encode()
    pending = root / _PENDING_NAME
    try:
        with contextlib.suppress(FileNotFoundError):
            pending.unlink()
        descriptor = open_private_file(pending, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(pending, root / PROOF_NAME)
        sync_directory(root)
    except OSError:
        with contextlib.suppress(OSError):
            pending.unlink()
        return False
    return True
