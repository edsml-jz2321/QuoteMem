"""Stat fingerprints that tell when a folder's earlier reading is still current."""

from __future__ import annotations

from collections.abc import Callable
from hashlib import sha256
import os
from pathlib import Path
import stat
import sys
import time

__all__ = ["interpreter_cache", "tree_signature"]

# A file changed within this window may still change again without a new
# modification time, because file systems keep coarse timestamps (FAT keeps
# two seconds). Such a folder gets no fingerprint, so it is always read again.
_RECENT_NS = 2_000_000_000
_BYTECODE_SUFFIXES = (".pyc", ".pyo")
# Windows keeps no change time (`st_ctime` is the creation time there), so an in-place
# rewrite that keeps the size and puts the modification time back changes no stat field.
_WINDOWS = sys.platform == "win32"


def _entry_stat(entry: os.DirEntry) -> os.stat_result:
    """Stat one entry, following symlinks; on Windows through `os.stat`, which fills the file ID and volume."""
    return os.stat(entry.path) if _WINDOWS else entry.stat()


def _content_digest(path: str) -> str:
    with open(path, "rb") as stream:
        return sha256(stream.read()).hexdigest()


def interpreter_cache(name: str) -> bool:
    """Return whether an entry name is an interpreter cache, which is never source."""
    return name == "__pycache__" or name.lower().endswith(_BYTECODE_SUFFIXES)


def tree_signature(
    root: Path,
    *,
    ignore: Callable[[str], bool] | None = None,
    settled: bool = True,
    contents: Callable[[str], bool] | None = None,
) -> tuple | None:
    """Return a stat fingerprint of `root` and every entry below it.

    The fingerprint holds each entry's relative path, type, symlink flag,
    and, for files, size, modification and change times, inode and device.
    On POSIX any write, replacement, rename, addition or removal changes it:
    a write always sets the change time, which only a clock change can
    forge. Windows keeps no change time (`st_ctime` is the creation time), so
    there an in-place rewrite that keeps a file's size and puts its
    modification time back changes no stat field; for the files that
    `contents` selects, the fingerprint on Windows also holds a digest of
    their bytes, which notices such a rewrite. Directories are followed as
    `Path.iterdir` and `Path.is_dir` follow them.

    Args:
        root: Folder to fingerprint.
        ignore: Optional predicate for entry names left out, such as
            interpreter caches.
        settled: Whether to refuse a fingerprint while a file is too new
            for its timestamps to show a further change. Pass `False` only
            for content whose every later use checks the bytes again.
        contents: Optional predicate for file names whose bytes the
            fingerprint also holds on Windows.

    Returns:
        tuple | None: The fingerprint, or `None` when it cannot vouch for the
        content: the folder cannot be read, a directory repeats through a
        symlink, or a file changed too recently for its timestamps to show a
        further change.
    """
    started = time.time_ns()
    newest = 0
    try:
        top = os.stat(root)
        if not stat.S_ISDIR(top.st_mode):
            return None
        entries: list[tuple] = [(".", os.path.islink(root), top.st_ino, top.st_dev)]
        seen = {(top.st_dev, top.st_ino)}
        pending: list[tuple[str, str]] = [("", os.fspath(root))]
        while pending:
            relative, directory = pending.pop()
            with os.scandir(directory) as listing:
                children = sorted(listing, key=lambda entry: entry.name)
            for child in children:
                if ignore is not None and ignore(child.name):
                    continue
                name = f"{relative}/{child.name}" if relative else child.name
                link = child.is_symlink()
                try:
                    info = _entry_stat(child)
                except FileNotFoundError:
                    entries.append((name, "missing", link))
                    continue
                if stat.S_ISDIR(info.st_mode):
                    if (info.st_dev, info.st_ino) in seen and info.st_ino:
                        return None
                    seen.add((info.st_dev, info.st_ino))
                    entries.append((name, info.st_mode, link, info.st_ino, info.st_dev))
                    pending.append((name, child.path))
                    continue
                newest = max(newest, info.st_mtime_ns, info.st_ctime_ns)
                entry = (name, info.st_mode, link, info.st_size, info.st_mtime_ns, info.st_ctime_ns, info.st_ino, info.st_dev)
                if _WINDOWS and contents is not None and contents(child.name):
                    entry = (*entry, _content_digest(child.path))
                entries.append(entry)
    except OSError:
        return None
    if settled and newest > started - _RECENT_NS:
        return None
    return tuple(entries)
