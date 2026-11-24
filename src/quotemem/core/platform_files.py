"""Private workspace files and process ownership on POSIX and Windows.

Windows privacy is checked against the handle's owner and DACL, not chmod bits.
Reparse points and multiply linked files are rejected. Anchored native operations
use retained NT directory handles; write-capable file and directory handles use
FlushFileBuffers. Unsupported flushes fail closed, never silently succeed.
"""

from __future__ import annotations

import errno
import os
import stat
import struct
from collections.abc import Iterator
from contextlib import contextmanager
from importlib import import_module
from pathlib import Path
from typing import Any

from quotemem.core.errors import QuoteMemError

OWNER_LOCK_NAME = "memory.db.lock"
NEW_FILES_NOT_OWNED = ("New files would not be owned by this Windows user; run QuoteMem "
                       "from a normal, non-elevated terminal")


def _unsafe() -> QuoteMemError:
    return QuoteMemError("workspace_unsafe", "Private path has unsafe identity or permissions")


if os.name == "nt":
    import ctypes
    import msvcrt

    _file = import_module("win32file")
    _security = import_module("win32security")
    _api = import_module("win32api")
    _types = import_module("pywintypes")
    _nt = ctypes.WinDLL("ntdll")
    _kernel = ctypes.WinDLL("kernel32", use_last_error=True)

    class _UnicodeString(ctypes.Structure):
        _fields_ = [("Length", ctypes.c_ushort), ("MaximumLength", ctypes.c_ushort),
                    ("Buffer", ctypes.c_void_p)]

    class _ObjectAttributes(ctypes.Structure):
        _fields_ = [("Length", ctypes.c_ulong), ("RootDirectory", ctypes.c_void_p),
                    ("ObjectName", ctypes.POINTER(_UnicodeString)),
                    ("Attributes", ctypes.c_ulong), ("SecurityDescriptor", ctypes.c_void_p),
                    ("SecurityQualityOfService", ctypes.c_void_p)]

    class _IOStatus(ctypes.Structure):
        _fields_ = [("Status", ctypes.c_void_p), ("Information", ctypes.c_size_t)]

    _nt.NtCreateFile.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_ulong,
        ctypes.POINTER(_ObjectAttributes), ctypes.POINTER(_IOStatus), ctypes.c_void_p,
        ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong,
        ctypes.c_void_p, ctypes.c_ulong]
    _nt.NtCreateFile.restype = ctypes.c_long
    _nt.RtlNtStatusToDosError.argtypes = [ctypes.c_long]
    _nt.RtlNtStatusToDosError.restype = ctypes.c_ulong
    _nt.NtQueryDirectoryFile.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.POINTER(_IOStatus), ctypes.c_void_p, ctypes.c_ulong,
        ctypes.c_ulong, ctypes.c_ubyte, ctypes.c_void_p, ctypes.c_ubyte]
    _nt.NtQueryDirectoryFile.restype = ctypes.c_long
    _nt.NtSetInformationFile.argtypes = [ctypes.c_void_p, ctypes.POINTER(_IOStatus),
        ctypes.c_void_p, ctypes.c_ulong, ctypes.c_ulong]
    _nt.NtSetInformationFile.restype = ctypes.c_long
    _kernel.SetFileInformationByHandle.argtypes = [ctypes.c_void_p, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_ulong]
    _kernel.SetFileInformationByHandle.restype = ctypes.c_int

    def _status(status: int) -> None:
        if status < 0:
            raise ctypes.WinError(_nt.RtlNtStatusToDosError(status))

    def _open_at(
        parent: int, name: str, access: int, disposition: int = 1, *, directory: bool = False,
        sharing: int = 7,
    ) -> int:
        # One basename only: the retained RootDirectory is the namespace anchor.
        if not name or name in {".", ".."} or any(c in name for c in "/\\:\x00"):
            raise ValueError("Anchored native open requires one ordinary basename")
        encoded = name.encode("utf-16-le")
        if len(encoded) > 65532:
            raise ValueError("Native basename is too long")
        buffer = ctypes.create_string_buffer(encoded + b"\0\0")
        unicode = _UnicodeString(len(encoded), len(encoded) + 2, ctypes.addressof(buffer))
        security = ctypes.create_string_buffer(
            bytes(_private_attributes(directory).SECURITY_DESCRIPTOR),
        )
        attributes = _ObjectAttributes(
            ctypes.sizeof(_ObjectAttributes), msvcrt.get_osfhandle(parent),
            ctypes.pointer(unicode), 0x40, ctypes.addressof(security), None,
        )
        result, io = ctypes.c_void_p(), _IOStatus()
        _status(_nt.NtCreateFile(
            ctypes.byref(result), access | 0x00100000, ctypes.byref(attributes), ctypes.byref(io),
            None, 0x80, sharing, disposition,
            0x00200000 | 0x20 | (1 if directory else 0), None, 0,
        ))
        handle = _types.HANDLE(result.value)
        try:
            descriptor = msvcrt.open_osfhandle(
                int(handle), (os.O_RDWR if access & 0x40000000 else os.O_RDONLY)
                | os.O_BINARY | os.O_NOINHERIT,
            )
            handle.Detach()
            return descriptor
        finally:
            handle.Close()

    def _current_sid() -> Any:
        """Return this user's SID after making it the default owner of new objects.

        An elevated administrator token gives new objects the default owner
        BUILTIN\\Administrators, also the files that SQLite and HeavenBase create
        themselves (WAL, shared memory, artifacts). The process default owner is
        set to the user, as in a normal terminal, before this module creates or
        compares any owner (`own_new_files` does it at service start), so every
        new storage file stays owned by the user.
        """
        token = _security.OpenProcessToken(_api.GetCurrentProcess(), 0x0008)  # TOKEN_QUERY
        try:
            sid = _security.GetTokenInformation(token, 1)[0]  # TokenUser
            if _security.GetTokenInformation(token, 4) == sid:  # TokenOwner
                return sid
        finally:
            token.Close()
        try:
            # TOKEN_QUERY | TOKEN_ADJUST_DEFAULT; the user SID is always a valid owner.
            token = _security.OpenProcessToken(_api.GetCurrentProcess(), 0x0008 | 0x0080)
            try:
                _security.SetTokenInformation(token, 4, sid)
            finally:
                token.Close()
        except _types.error as error:
            raise QuoteMemError("workspace_unsafe", NEW_FILES_NOT_OWNED) from error
        return sid

    def _private_attributes(directory: bool = False) -> Any:
        sid = _current_sid()
        acl = _security.ACL()
        # FILE_ALL_ACCESS; directories propagate privacy to newly created children.
        acl.AddAccessAllowedAceEx(2, 0x03 if directory else 0, 0x001F01FF, sid)
        descriptor = _security.SECURITY_DESCRIPTOR()
        descriptor.SetSecurityDescriptorOwner(sid, False)
        descriptor.SetSecurityDescriptorDacl(True, acl, False)
        descriptor.SetSecurityDescriptorControl(0x1000, 0x1000)  # SE_DACL_PROTECTED
        attributes = _types.SECURITY_ATTRIBUTES()
        attributes.SECURITY_DESCRIPTOR = descriptor
        return attributes

    def _win_open(
        path: Path, access: int, disposition: int, *, directory: bool = False,
        private: bool = True,
    ) -> Any:
        try:
            return _file.CreateFile(
                str(path), access, 0x01 | 0x02,
                _private_attributes(directory) if private else None, disposition,
                # OPEN_REPARSE_POINT prevents following links; BACKUP_SEMANTICS opens dirs.
                0x00200000 | (0x02000000 if directory else 0x80000000), None,
            )
        except _types.error as error:
            raise ctypes.WinError(error.winerror) from error

    def _validate_windows_handle(
        handle: Any, *, directory: bool = False, single_link: bool = True,
        allow_cpython_private: bool = False,
    ) -> None:
        information = _file.GetFileInformationByHandle(handle)
        attributes, links = information[0], information[7]
        if (attributes & 0x400 or bool(attributes & 0x10) != directory
                or (not directory and single_link and links != 1)):
            raise _unsafe()
        descriptor = _security.GetSecurityInfo(handle, 1, 0x01 | 0x04)
        sid = _current_sid()
        if descriptor.GetSecurityDescriptorOwner() != sid:
            raise _unsafe()
        acl = descriptor.GetSecurityDescriptorDacl()
        if acl is None or not acl.GetAceCount():
            raise _unsafe()
        if allow_cpython_private and acl.GetAceCount() == 3:
            # CPython v3.12.13 posixmodule.c:5370 uses exactly this protected
            # inheritable 0700 DACL. Only an anchored HB artifact caller opts in;
            # database/root/sidecar admission retains the current-SID predicate.
            aces = [acl.GetAce(index) for index in range(3)]
            flags = {ace[0][1] for ace in aces}
            canonical_flags = (
                flags == {0x13} or (
                    flags == {0x03} and descriptor.GetSecurityDescriptorControl()[0] & 0x1000
                )
            ) if directory else flags == {0x10}
            if canonical_flags and all(ace[0][0] == 0 and ace[1] == 0x001F01FF for ace in aces):
                principals = {_security.ConvertSidToStringSid(ace[2]) for ace in aces}
                if principals == {"S-1-5-18", "S-1-5-32-544", "S-1-3-4"}:
                    return
        allowed = 0
        for index in range(acl.GetAceCount()):
            ace = acl.GetAce(index)
            # Accept only ordinary grants to this exact account. Fail closed on
            # foreign principals and unfamiliar/object/callback ACE semantics.
            if ace[0][0] != 0 or ace[2] != sid:
                raise _unsafe()
            if not ace[0][1] & 0x08:  # INHERIT_ONLY_ACE grants no access to this object.
                allowed |= ace[1]
        if allowed & 0x001F01FF != 0x001F01FF:
            raise _unsafe()
else:
    import fcntl


def own_new_files() -> None:
    """Make this user the owner of every file this process creates from now on.

    Windows only: under an elevated administrator token the files SQLite and
    HeavenBase create would otherwise be owned by BUILTIN\\Administrators, which
    the privacy checks refuse. A POSIX file is always owned by the process's user.

    Raises:
        QuoteMemError: ``workspace_unsafe`` when the token refuses the change; the
            message tells the user to run from a normal, non-elevated terminal.
    """
    if os.name == "nt":
        _current_sid()


def _validate_posix(metadata: os.stat_result, *, directory: bool = False) -> None:
    if os.name == "nt":
        raise RuntimeError("POSIX permission validation is unavailable on Windows")
    expected_type = stat.S_ISDIR if directory else stat.S_ISREG
    if (not expected_type(metadata.st_mode) or metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) != (0o700 if directory else 0o600)
            or (not directory and metadata.st_nlink != 1)):
        raise _unsafe()


def validate_private_path(path: Path, *, directory: bool = False) -> None:
    """Require a private current-user regular file or directory without links."""
    if os.name == "nt":
        handle = _win_open(path, 0x00020080, 3, directory=directory)
        try:
            _validate_windows_handle(handle, directory=directory)
        finally:
            handle.Close()
    else:
        _validate_posix(path.lstat(), directory=directory)


def ensure_private_directory(path: Path) -> None:
    """Create one private directory, or validate an existing one without repair."""
    if os.name == "nt":
        try:
            _file.CreateDirectory(str(path), _private_attributes(directory=True))
        except _types.error as error:
            if error.winerror != 183:  # ERROR_ALREADY_EXISTS
                raise OSError(error.winerror, error.strerror, str(path)) from error
    else:
        path.mkdir(mode=0o700, parents=False, exist_ok=True)
    validate_private_path(path, directory=True)


def open_private_file(path: Path, flags: int) -> int:
    """Open/create a private single-link file and return an owning binary fd."""
    if os.name == "nt":
        access = 0x80000000 | 0x00020000
        if flags & (os.O_WRONLY | os.O_RDWR):
            access |= 0x40000000
        disposition = 1 if flags & os.O_EXCL else (4 if flags & os.O_CREAT else 3)
        handle = _win_open(path, access, disposition)
        try:
            _validate_windows_handle(handle)
            descriptor = msvcrt.open_osfhandle(
                int(handle), (flags & (os.O_WRONLY | os.O_RDWR))
                | os.O_BINARY | os.O_NOINHERIT,
            )
            handle.Detach()
        finally:
            handle.Close()
    else:
        descriptor = os.open(path, flags | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    try:
        validate_private_file(descriptor, path)
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def validate_private_file(descriptor: int, path: Path) -> None:
    """Revalidate a retained private file's ACL/mode, links and no-follow identity.

    The caller owns the descriptor throughout validation. Windows checks the
    handle's current owner/DACL each time; a pathname-only ACL check cannot
    establish the identity of a previously opened file.
    """
    metadata = os.fstat(descriptor)
    if os.name == "nt":
        _validate_windows_handle(msvcrt.get_osfhandle(descriptor))
    else:
        _validate_posix(metadata)
    located = path.lstat()
    if not stat.S_ISREG(located.st_mode) or (
        metadata.st_dev, metadata.st_ino
    ) != (located.st_dev, located.st_ino):
        raise _unsafe()


def open_regular_file(path: Path, flags: int) -> int:
    """Open public data without following links; return an owning binary descriptor.

    Existing ACLs/modes are preserved. Files must be regular, singly linked and
    match the final pathname's identity. Creation is exclusive when requested;
    append uses the native descriptor flag. Truncation is deliberately unsupported
    so admission never destroys bytes before identity validation.
    """
    if flags & os.O_TRUNC:
        raise ValueError("Regular-file admission cannot truncate existing data")
    if os.name == "nt":
        access = 0x80000000
        if flags & (os.O_WRONLY | os.O_RDWR):
            access |= 0x40000000
        disposition = 1 if flags & os.O_CREAT and flags & os.O_EXCL else (
            4 if flags & os.O_CREAT else 3
        )
        handle = _win_open(path, access, disposition, private=False)
        try:
            information = _file.GetFileInformationByHandle(handle)
            if information[0] & (0x400 | 0x10) or information[7] != 1:
                raise OSError(errno.EINVAL, "Expected a singly linked regular file", str(path))
            descriptor = msvcrt.open_osfhandle(
                int(handle), (flags & (os.O_WRONLY | os.O_RDWR | os.O_APPEND))
                | os.O_BINARY | os.O_NOINHERIT,
            )
            handle.Detach()
        finally:
            handle.Close()
    else:
        descriptor = os.open(
            path, flags | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK, 0o600,
        )
    try:
        held, located = os.fstat(descriptor), path.lstat()
        if (
            not stat.S_ISREG(held.st_mode) or held.st_nlink != 1
            or not stat.S_ISREG(located.st_mode)
            or (held.st_dev, held.st_ino) != (located.st_dev, located.st_ino)
        ):
            raise OSError(errno.EINVAL, "Regular file identity changed", str(path))
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def lock_exclusive(descriptor: int) -> None:
    """Acquire a real nonblocking process lock, released when the descriptor closes.

    Raises BlockingIOError on contention; other native errors remain failures.
    Windows locks byte zero even for empty files; POSIX uses an exclusive flock.
    """
    if os.name == "nt":
        try:
            _file.LockFileEx(msvcrt.get_osfhandle(descriptor), 0x03, 1, 0, _types.OVERLAPPED())
        except _types.error as error:
            if error.winerror == 33:  # ERROR_LOCK_VIOLATION
                raise BlockingIOError(errno.EAGAIN, "Another process owns this file") from error
            raise ctypes.WinError(error.winerror) from error
    else:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)


def unlock_file(descriptor: int) -> None:
    """Release exactly the native lock acquired by lock_exclusive."""
    if os.name == "nt":
        try:
            _file.UnlockFileEx(msvcrt.get_osfhandle(descriptor), 1, 0, _types.OVERLAPPED())
        except _types.error as error:
            raise ctypes.WinError(error.winerror) from error
    else:
        fcntl.flock(descriptor, fcntl.LOCK_UN)


def open_directory(path: Path, *, private: bool = True) -> int:
    """Retain a no-follow directory descriptor supporting real metadata flushes."""
    if os.name == "nt":
        try:
            handle = _file.CreateFile(str(path), 0xC0020000, 7, None, 3, 0x02200000, None)
        except _types.error as error:
            raise ctypes.WinError(error.winerror) from error
        try:
            attributes = _file.GetFileInformationByHandle(handle)[0]
            if attributes & 0x400 or not attributes & 0x10:
                raise _unsafe()
            if private:
                _validate_windows_handle(handle, directory=True)
            descriptor = msvcrt.open_osfhandle(
                int(handle), os.O_RDWR | os.O_BINARY | os.O_NOINHERIT,
            )
            handle.Detach()
            return descriptor
        finally:
            handle.Close()
    return os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)


def open_file_at(parent: int, name: str, flags: int, *, directory: bool = False) -> int:
    """Open one entry relative to a retained directory, including on Windows.

    Windows uses NtCreateFile RootDirectory, never a reconstructed pathname.
    The returned descriptor owns the native handle. Callers validate its identity
    and ACL before using it; no reparse target is followed or existing file truncated.
    """
    if os.name == "nt":
        if flags & os.O_TRUNC:
            raise ValueError("Anchored admission cannot truncate existing files")
        access = 0x80020000
        if directory or flags & (os.O_WRONLY | os.O_RDWR):
            access |= 0x40000000
        disposition = 2 if flags & os.O_CREAT and flags & os.O_EXCL else (
            3 if flags & os.O_CREAT else 1
        )
        return _open_at(parent, name, access, disposition, directory=directory)
    return os.open(name, flags | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK
                   | (os.O_DIRECTORY if directory else 0),
                   0o700 if directory else 0o600, dir_fd=parent)


def stat_at(parent: int, name: str) -> os.stat_result:
    """Read one anchored entry's identity without following a redirected target."""
    if os.name != "nt":
        return os.stat(name, dir_fd=parent, follow_symlinks=False)
    descriptor = _open_at(parent, name, 0x00020080)
    try:
        return os.fstat(descriptor)
    finally:
        os.close(descriptor)


def validate_private_descriptor(
    descriptor: int, *, directory: bool = False, single_link: bool = True,
    allow_cpython_private: bool = False,
) -> None:
    """Check retained ownership, ACL/type and optional single-link admission.

    Args:
        descriptor: Borrowed no-follow handle to validate without mutation.
        directory: Require a directory rather than a regular file.
        single_link: Require one link for file admission.
        allow_cpython_private: Also accept the exact CPython 3.12.13 Windows
            0700 DACL or its directory/file inheritance. Restricted to explicitly
            anchored owned artifact trees; never root, database or lock admission.

    Returns:
        None: The current native security descriptor satisfied the predicate.

    Raises:
        QuoteMemError: Owner, ACL, type, reparse or link validation failed.
        OSError: The retained descriptor could not be queried.
    """
    if os.name == "nt":
        _validate_windows_handle(
            msvcrt.get_osfhandle(descriptor), directory=directory, single_link=single_link,
            allow_cpython_private=allow_cpython_private,
        )
    else:
        _validate_posix(os.fstat(descriptor), directory=directory)


def sync_descriptor(descriptor: int) -> None:
    """Flush a retained write-capable file/directory handle; propagate every failure."""
    if os.name == "nt":
        try:
            _file.FlushFileBuffers(msvcrt.get_osfhandle(descriptor))
        except _types.error as error:
            raise ctypes.WinError(error.winerror) from error
    else:
        os.fsync(descriptor)


def list_directory(parent: int) -> list[str]:
    """Enumerate a retained directory itself, independent of its current pathname."""
    if os.name != "nt":
        return os.listdir(parent)
    result: list[str] = []
    restart = True
    while True:
        buffer, io = ctypes.create_string_buffer(65536), _IOStatus()
        status = _nt.NtQueryDirectoryFile(
            msvcrt.get_osfhandle(parent), None, None, None, ctypes.byref(io), buffer,
            len(buffer), 12, False, None, restart,  # FileNamesInformation
        )
        restart = False
        if status & 0xFFFFFFFF == 0x80000006:  # STATUS_NO_MORE_FILES
            return result
        _status(status)
        offset = 0
        while True:
            following, _, length = struct.unpack_from("<III", buffer, offset)
            name = buffer.raw[offset + 12:offset + 12 + length].decode("utf-16-le")
            if name not in {".", ".."}:
                result.append(name)
            if not following:
                break
            offset += following


def _unlink_at(
    parent: int, name: str, expected: tuple[int, int], *, directory: bool = False,
    allow_readonly: bool = False,
    allow_cpython_private: bool = False,
    expected_links: int = 1,
) -> None:
    """Delete the exact opened entry beneath an anchor; never a replacement path.

    Windows POSIX disposition unlinks even while a validation handle is retained.
    Readonly immutable copies may be removed only by a terminal purge caller.
    """
    if os.name != "nt":
        value = os.stat(name, dir_fd=parent, follow_symlinks=False)
        if (value.st_dev, value.st_ino) != expected:
            raise _unsafe()
        if not directory and value.st_nlink != expected_links:
            raise _unsafe()
        os.rmdir(name, dir_fd=parent) if directory else os.unlink(name, dir_fd=parent)
        return
    # Omit FILE_SHARE_DELETE to exclude competing delete/rename handles. NTFS
    # hardlinks bypass sharing; require both pre- and post-unlink link counts.
    descriptor = _open_at(parent, name, 0x00030080, directory=directory, sharing=3)
    try:
        value = os.fstat(descriptor)
        if (value.st_dev, value.st_ino) != expected:
            raise _unsafe()
        if not directory and value.st_nlink != expected_links:
            raise _unsafe()
        _validate_windows_handle(
            msvcrt.get_osfhandle(descriptor), directory=directory,
            single_link=expected_links == 1,
            allow_cpython_private=allow_cpython_private,
        )
        flags = ctypes.c_ulong(0x03 | (0x10 if allow_readonly else 0))
        if not _kernel.SetFileInformationByHandle(
            msvcrt.get_osfhandle(descriptor), 21, ctypes.byref(flags), ctypes.sizeof(flags),
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        if not directory and os.fstat(descriptor).st_nlink != expected_links - 1:
            # A link created after the first check must never produce successful
            # purge evidence. Namespace removal may already be partial.
            raise _unsafe()
    finally:
        os.close(descriptor)


def unlink_at(
    parent: int, name: str, expected: tuple[int, int], *, directory: bool = False,
    allow_readonly: bool = False, allow_cpython_private: bool = False,
) -> None:
    """Remove an identity-checked entry; terminal files must have exactly one link.

    Args:
        parent: Borrowed retained no-follow directory descriptor.
        name: One entry beneath that descriptor.
        expected: Required device and inode of the deletion handle.
        directory: Require a directory rather than a regular file.
        allow_readonly: Permit terminal removal of immutable artifact copies.
        allow_cpython_private: Exact native 0700 equivalence for anchored HB artifacts only.

    Returns:
        None: The namespace operation completed; caller must sync the parent.

    Raises:
        QuoteMemError: Identity, link count, type or ACL changed.
        OSError: Native deletion or exclusive sharing failed.
    """
    _unlink_at(parent, name, expected, directory=directory, allow_readonly=allow_readonly,
               allow_cpython_private=allow_cpython_private)


def unlink_probe_at(parent: int, name: str, expected: tuple[int, int], links: int) -> None:
    """Remove only an explicitly counted one/two-link capability probe entry.

    Args:
        parent: Borrowed probe directory descriptor.
        name: Verified probe basename.
        expected: Exact probe device/inode.
        links: Exact observed link count, restricted to one or two.

    Returns:
        None: Probe entry removed; this function is never a terminal purge API.

    Raises:
        ValueError: The probe supplied another link count.
        QuoteMemError: The deletion handle no longer matches the probe evidence.
        OSError: Native deletion or sharing failed.
    """
    if type(links) is not int or links not in {1, 2}:
        raise ValueError("Probe cleanup requires one or two verified links")
    _unlink_at(parent, name, expected, expected_links=links)


def link_at(parent: int, source: str, target: str) -> None:
    """Create an exclusive hardlink between entries anchored to one directory."""
    if os.name != "nt":
        os.link(source, target, src_dir_fd=parent, dst_dir_fd=parent, follow_symlinks=False)
        return
    if not target or target in {".", ".."} or any(c in target for c in "/\\:\x00"):
        raise ValueError("Anchored link requires one ordinary basename")
    descriptor = _open_at(parent, source, 0x00030080)
    try:
        class LinkInfo(ctypes.Structure):
            _fields_ = [("ReplaceIfExists", ctypes.c_ubyte), ("RootDirectory", ctypes.c_void_p),
                        ("FileNameLength", ctypes.c_ulong), ("FileName", ctypes.c_wchar * 1)]

        encoded = target.encode("utf-16-le")
        buffer = ctypes.create_string_buffer(
            max(ctypes.sizeof(LinkInfo), LinkInfo.FileName.offset + len(encoded)),
        )
        info = LinkInfo.from_buffer(buffer)
        info.ReplaceIfExists = False
        info.RootDirectory = msvcrt.get_osfhandle(parent)
        info.FileNameLength = len(encoded)
        ctypes.memmove(ctypes.addressof(buffer) + LinkInfo.FileName.offset, encoded, len(encoded))
        io = _IOStatus()
        _status(_nt.NtSetInformationFile(
            msvcrt.get_osfhandle(descriptor), ctypes.byref(io), buffer, len(buffer), 11,
        ))
    finally:
        os.close(descriptor)


def sync_directory(path: Path) -> None:
    """Flush a real no-follow directory handle and propagate unsupported operation."""
    if os.name == "nt":
        descriptor = open_directory(path, private=False)
        try:
            sync_descriptor(descriptor)
        finally:
            os.close(descriptor)
        return
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def exclusive_owner(root: Path) -> Iterator[None]:
    """Hold a real nonblocking process lock before either HB database is opened."""
    ensure_private_directory(root)
    directory: Any = None
    descriptor = -1
    locked, failed, pid = False, True, os.getpid()
    try:
        if os.name == "nt":
            directory = _win_open(root, 0x00020080, 3, directory=True)
            _validate_windows_handle(directory, directory=True)
            descriptor = open_private_file(root / OWNER_LOCK_NAME, os.O_CREAT | os.O_RDWR)
            try:
                lock_exclusive(descriptor)
            except BlockingIOError as error:
                raise QuoteMemError("workspace_locked",
                                    "Another process owns this workspace") from error
            locked = True
        else:
            directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
            descriptor = os.open(
                OWNER_LOCK_NAME, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC,
                0o600, dir_fd=directory,
            )
            metadata = os.fstat(descriptor)
            _validate_posix(metadata)
            located = os.stat(OWNER_LOCK_NAME, dir_fd=directory, follow_symlinks=False)
            if (metadata.st_dev, metadata.st_ino) != (located.st_dev, located.st_ino):
                raise _unsafe()
            try:
                lock_exclusive(descriptor)
            except BlockingIOError as error:
                raise QuoteMemError("workspace_locked",
                                    "Another process owns this workspace") from error
            locked = True
            os.fsync(directory)
        yield
        failed = False
    finally:
        try:
            if descriptor >= 0:
                try:
                    # Windows may release a closed handle's lock only some time later.
                    # A fork child never unlocks its parent's lock (as WorkspaceOwner.close).
                    if locked and os.getpid() == pid:
                        unlock_file(descriptor)
                except OSError:
                    if not failed:  # Never in place of the body's own exception.
                        raise
                finally:
                    os.close(descriptor)
        finally:
            if directory is not None:
                directory.Close() if os.name == "nt" else os.close(directory)
