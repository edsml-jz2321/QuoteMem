"""Identity reads that keep the interpreter lock.

Plan section 2 rechecks the workspace parent, sidecar lock, databases and
SQLite companions at every Writer critical section: the retained descriptor
and the no-follow path state of each. Through the standard library every one
of those sixteen to twenty-four stats releases and reacquires the interpreter
lock. Under the normal-load registration one reacquisition cost a mean 76 us
on macOS and 506 us on Windows while the same ``fstat`` through
``ctypes.PyDLL``, which keeps the lock, cost 4 us and 10 us (harness X11 and
ownF, 2026-09-11); the recheck therefore held the Writer mutex for 2.7 ms and
4-8 ms per critical section, 39-64 percent of its hold time.

This module reads exactly the fields the owner compares, through ``PyDLL`` on
the POSIX platforms whose ``struct stat`` layout it knows. The owner adopts a
reader only after it reproduced the standard library's answer for every
retained object, so an unknown layout or platform keeps the standard library.
"""

from __future__ import annotations

import ctypes
import os
import platform
import stat
import sys
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class StatIdentity:
    """The stat fields the owner compares; ``kind`` and ``mode`` split ``st_mode``."""

    device: int
    inode: int
    kind: int
    links: int
    owner: int
    mode: int

    @classmethod
    def read(cls, value: os.stat_result) -> StatIdentity:
        return cls(
            value.st_dev,
            value.st_ino,
            stat.S_IFMT(value.st_mode),
            value.st_nlink,
            value.st_uid,
            stat.S_IMODE(value.st_mode),
        )


class IdentityReader(Protocol):
    """Three identity reads the owner performs at every recheck."""

    def descriptor(self, fd: int) -> StatIdentity:
        """Identity of a retained descriptor."""
        ...

    def anchored(self, parent_fd: int, name: str) -> StatIdentity:
        """Identity of one directory entry without following a symlink."""
        ...

    def path(self, path: os.PathLike[str] | str) -> StatIdentity:
        """Identity of a path without following a final symlink."""
        ...


class StandardLibraryReader:
    """The ``os`` module reads; each releases the interpreter lock."""

    def descriptor(self, fd: int) -> StatIdentity:
        return StatIdentity.read(os.fstat(fd))

    def anchored(self, parent_fd: int, name: str) -> StatIdentity:
        return StatIdentity.read(os.stat(name, dir_fd=parent_fd, follow_symlinks=False))

    def path(self, path: os.PathLike[str] | str) -> StatIdentity:
        return StatIdentity.read(os.stat(path, follow_symlinks=False))


@dataclass(frozen=True)
class _Layout:
    """Byte offsets and widths of the compared fields plus the platform constants."""

    size: int
    dev: tuple[int, int]
    ino: tuple[int, int]
    mode: tuple[int, int]
    nlink: tuple[int, int]
    uid: tuple[int, int]
    at_fdcwd: int
    at_symlink_nofollow: int
    fstat_symbols: tuple[str, ...]
    fstatat_symbols: tuple[str, ...]
    stat_version: int | None = None


_DARWIN = _Layout(
    size=144, dev=(0, 4), mode=(4, 2), nlink=(6, 2), ino=(8, 8), uid=(16, 4),
    at_fdcwd=-2, at_symlink_nofollow=0x20,
    fstat_symbols=("fstat",) if platform.machine() == "arm64" else ("fstat$INODE64",),
    fstatat_symbols=("fstatat",) if platform.machine() == "arm64" else ("fstatat$INODE64",),
)
_LINUX_X86_64 = _Layout(
    size=144, dev=(0, 8), ino=(8, 8), nlink=(16, 8), mode=(24, 4), uid=(28, 4),
    at_fdcwd=-100, at_symlink_nofollow=0x100,
    fstat_symbols=("fstat", "__fxstat"), fstatat_symbols=("fstatat", "__fxstatat"),
    stat_version=1,
)
_LINUX_AARCH64 = _Layout(
    size=128, dev=(0, 8), ino=(8, 8), mode=(16, 4), nlink=(20, 4), uid=(24, 4),
    at_fdcwd=-100, at_symlink_nofollow=0x100,
    fstat_symbols=("fstat", "__fxstat"), fstatat_symbols=("fstatat", "__fxstatat"),
    stat_version=0,
)
_LAYOUTS: dict[tuple[str, str], _Layout] = {
    ("darwin", "arm64"): _DARWIN,
    ("darwin", "x86_64"): _DARWIN,
    ("linux", "x86_64"): _LINUX_X86_64,
    ("linux", "aarch64"): _LINUX_AARCH64,
}


def _field(buffer: bytes, position: tuple[int, int]) -> int:
    offset, width = position
    return int.from_bytes(buffer[offset:offset + width], sys.byteorder, signed=False)


class NativeReader:
    """``fstat``/``fstatat`` through ``ctypes.PyDLL``; the interpreter lock stays held.

    Construction resolves the platform's symbols; ``supported()`` says whether
    the running platform has a known layout. Every read allocates its own
    buffer, so concurrent callers never share one.
    """

    def __init__(self, layout: _Layout) -> None:
        self._layout = layout
        self._buffer_size = max(layout.size, 256)
        library = ctypes.PyDLL(None, use_errno=True)
        self._fstat: Any
        self._fstatat: Any
        self._fstat, self._fstat_versioned = self._resolve(library, layout.fstat_symbols)
        self._fstatat, self._fstatat_versioned = self._resolve(library, layout.fstatat_symbols)
        if self._fstat_versioned:
            self._fstat.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        else:
            self._fstat.argtypes = [ctypes.c_int, ctypes.c_void_p]
        if self._fstatat_versioned:
            self._fstatat.argtypes = [
                ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p, ctypes.c_int,
            ]
        else:
            self._fstatat.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p, ctypes.c_int]
        self._fstat.restype = self._fstatat.restype = ctypes.c_int

    @staticmethod
    def _resolve(library: ctypes.PyDLL, symbols: tuple[str, ...]) -> tuple[Any, bool]:
        for symbol in symbols:
            try:
                return getattr(library, symbol), symbol.startswith("__")
            except AttributeError:
                continue
        raise OSError(f"No stat symbol among {symbols}")

    @classmethod
    def supported(cls) -> bool:
        """Report whether this platform and architecture have a known layout."""
        return (sys.platform, platform.machine()) in _LAYOUTS

    @classmethod
    def for_platform(cls) -> NativeReader:
        """Build the reader for the running platform.

        Raises:
            OSError: The platform has no known layout or lacks the symbols.
        """
        layout = _LAYOUTS.get((sys.platform, platform.machine()))
        if layout is None:
            raise OSError(f"No native stat layout for {sys.platform}/{platform.machine()}")
        return cls(layout)

    def _identity(self, buffer: ctypes.Array[ctypes.c_char]) -> StatIdentity:
        raw = buffer.raw
        layout = self._layout
        mode = _field(raw, layout.mode)
        return StatIdentity(
            _field(raw, layout.dev),
            _field(raw, layout.ino),
            stat.S_IFMT(mode),
            _field(raw, layout.nlink),
            _field(raw, layout.uid),
            stat.S_IMODE(mode),
        )

    @staticmethod
    def _failed(name: object) -> OSError:
        code = ctypes.get_errno()
        return OSError(code, os.strerror(code), name)

    def descriptor(self, fd: int) -> StatIdentity:
        buffer = ctypes.create_string_buffer(self._buffer_size)
        if self._fstat_versioned:
            result = int(self._fstat(self._layout.stat_version, fd, buffer))
        else:
            result = int(self._fstat(fd, buffer))
        if result != 0:
            raise self._failed(fd)
        return self._identity(buffer)

    def _at(self, directory: int, name: bytes) -> StatIdentity:
        buffer = ctypes.create_string_buffer(self._buffer_size)
        flags = self._layout.at_symlink_nofollow
        if self._fstatat_versioned:
            result = int(self._fstatat(self._layout.stat_version, directory, name, buffer, flags))
        else:
            result = int(self._fstatat(directory, name, buffer, flags))
        if result != 0:
            raise self._failed(os.fsdecode(name))
        return self._identity(buffer)

    def anchored(self, parent_fd: int, name: str) -> StatIdentity:
        return self._at(parent_fd, os.fsencode(name))

    def path(self, path: os.PathLike[str] | str) -> StatIdentity:
        return self._at(self._layout.at_fdcwd, os.fsencode(path))


STANDARD_LIBRARY = StandardLibraryReader()
