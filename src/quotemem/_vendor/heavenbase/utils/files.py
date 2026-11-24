"""Path and file helpers for HeavenBase."""

from __future__ import annotations

__all__ = [
    "copy_dir",
    "copy_file",
    "copy_path",
    "delete_dir",
    "delete_file",
    "delete_path",
    "empty_dir",
    "enum_dirs",
    "enum_files",
    "enum_paths",
    "exists_dir",
    "exists_file",
    "exists_path",
    "folder_diagram",
    "file_uri_to_path",
    "get_file_basename",
    "get_file_dir",
    "get_file_ext",
    "get_file_name",
    "has_file_ext",
    "is_link",
    "list_dirs",
    "list_files",
    "list_paths",
    "nonempty_dir",
    "normalize_file_uri",
    "path_inside",
    "path_to_file_uri",
    "pj",
    "replace_file",
    "temp_file",
    "touch_dir",
    "touch_file",
]

import os
import ntpath
import re
import shutil
import tempfile
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit

from .typing import Dict, Generator, Iterable, List, Literal, Mapping, Optional, Sequence, Union

PathLike = Union[str, os.PathLike[str]]
CopyFileMode = Literal["replace", "skip", "strict"]
CopyDirMode = Literal["replace", "skip", "strict", "merge"]

_WINDOWS_ABSOLUTE = re.compile(r"^[A-Za-z]:[\\/]")


def path_to_file_uri(path: PathLike) -> str:
    """Encode one absolute native path as a canonical local-file URI.

    Args:
        path (PathLike): Absolute POSIX or Windows-drive path. Relative paths,
            UNC paths, and network locations are rejected.

    Returns:
        str: Canonical ``file:///`` URI with UTF-8 percent encoding.

    Raises:
        ValueError: If ``path`` is not an absolute local filesystem path.
    """
    value = os.fspath(path)
    if not value or value.startswith(("//", "\\\\")):
        raise ValueError("Local file paths must not use a network or UNC location")
    if "\x00" in value:
        raise ValueError("Local file paths cannot contain NUL bytes")
    if _WINDOWS_ABSOLUTE.match(value):
        normalized = ntpath.normpath(value).replace("\\", "/")
        normalized = f"{normalized[0].upper()}{normalized[1:]}"
        return f"file:///{quote(normalized, safe='/:')}"
    if not value.startswith("/"):
        raise ValueError("Local file paths must be absolute POSIX or Windows-drive paths")
    normalized = os.path.abspath(os.path.normpath(value))
    return f"file://{quote(normalized, safe='/')}"


def file_uri_to_path(uri: str, *, home: PathLike | None = None, resource: PathLike | None = None) -> str:
    """Decode the strict HeavenBase local-file URI grammar.

    Args:
        uri (str): Canonical ``file:///`` URI or the ``file://%/`` package-home
            and ``file://&/`` resource-root aliases.
        home (PathLike | None): Absolute package-home root required for ``%``.
        resource (PathLike | None): Stable absolute resource root required for
            ``&``.

    Returns:
        str: Normalized absolute path, decoded exactly once. Windows drive
        paths keep forward slashes (``C:/...``), which Windows APIs accept.

    Raises:
        TypeError: If ``uri`` is not a string.
        ValueError: If syntax, authority, root, or path is invalid.
    """
    if not isinstance(uri, str):
        raise TypeError(f"Local file URI must be str, got {type(uri).__name__}")
    if not uri.startswith("file://"):
        raise ValueError("Local file locations must use file:///..., file://%/..., or file://&/...")
    parsed = urlsplit(uri)
    if "\\" in uri or re.search(r"%(?![0-9A-Fa-f]{2})", parsed.path):
        raise ValueError("Local file URIs require forward slashes and valid percent encoding")
    if parsed.scheme != "file" or parsed.query or parsed.fragment:
        raise ValueError("Local file URIs cannot contain query strings or fragments")
    if parsed.username is not None or parsed.password is not None or parsed.port is not None:
        raise ValueError("Local file URIs cannot contain credentials or ports")
    authority = parsed.netloc
    decoded = unquote(parsed.path)
    if "\x00" in decoded or "\\" in decoded:
        raise ValueError("Local file URIs cannot contain NUL bytes or backslashes")
    if authority in {"%", "&"}:
        root = home if authority == "%" else resource
        if root is None:
            name = "package home" if authority == "%" else "resource root"
            raise ValueError(f"Local file alias {authority!r} requires a stable {name}")
        if not decoded.startswith("/") or decoded in {"", "/"}:
            raise ValueError(f"Local file alias {authority!r} requires a relative path")
        base = os.fspath(root)
        if not (base.startswith("/") or _WINDOWS_ABSOLUTE.match(base)):
            raise ValueError(f"Local file alias {authority!r} requires an absolute root")
        return _join_file_root(base, decoded[1:])
    if authority:
        raise ValueError(f"Network file authority {authority!r} is not supported")
    if not uri.startswith("file:///"):
        raise ValueError("Legacy file: spellings are not accepted; use canonical file:/// syntax")
    if re.match(r"^/[A-Za-z]:/", decoded):
        normalized = ntpath.normpath(decoded[1:]).replace("\\", "/")
        return f"{normalized[0].upper()}{normalized[1:]}"
    if not decoded.startswith("/") or decoded.startswith("//"):
        raise ValueError("Canonical local file URIs require an absolute local path")
    return os.path.abspath(os.path.normpath(decoded))


def normalize_file_uri(uri: str, *, home: PathLike | None = None, resource: PathLike | None = None) -> str:
    """Expand an accepted local-file URI to canonical absolute URI form.

    Args:
        uri (str): Strict canonical or aliased HeavenBase local-file URI.
        home (PathLike | None): Package-home root for the ``%`` alias.
        resource (PathLike | None): Resource root for the ``&`` alias.

    Returns:
        str: Canonical absolute ``file:///`` URI.
    """
    return path_to_file_uri(file_uri_to_path(uri, home=home, resource=resource))


def _join_file_root(root: str, relative: str) -> str:
    if not relative or relative.startswith(("/", "\\")) or _WINDOWS_ABSOLUTE.match(relative):
        raise ValueError("Local file aliases require a relative child path")
    if _WINDOWS_ABSOLUTE.match(root):
        joined = ntpath.normpath(ntpath.join(root, relative.replace("/", "\\")))
        root_normal = ntpath.normcase(ntpath.normpath(root))
        if ntpath.commonpath((root_normal, ntpath.normcase(joined))) != root_normal:
            raise ValueError("Local file alias path escapes its configured root")
        normalized = joined.replace("\\", "/")
        return f"{normalized[0].upper()}{normalized[1:]}"
    joined = os.path.abspath(os.path.normpath(os.path.join(root, relative)))
    if os.path.commonpath((os.path.abspath(os.path.normpath(root)), joined)) != os.path.abspath(os.path.normpath(root)):
        raise ValueError("Local file alias path escapes its configured root")
    return joined


def _clean(part: PathLike) -> str:
    return os.fspath(part).strip().rstrip("/\\")


def _apply_alias(first: str, aliases: Optional[Mapping[str, PathLike]]) -> List[str]:
    if not aliases:
        return [first]
    for alias, target in sorted(aliases.items(), key=lambda item: len(str(item[0])), reverse=True):
        alias = str(alias).rstrip("/\\")
        if alias and first.startswith(alias):
            rest = first[len(alias) :].lstrip("/\\")
            return [os.fspath(target), rest] if rest else [os.fspath(target)]
    return [first]


def pj(*parts: PathLike, abs: bool = False, aliases: Optional[Mapping[str, PathLike]] = None, root: Optional[PathLike] = None) -> str:
    """
    Join path parts with user expansion, environment expansion, optional aliases, and normalization.

    Use this helper for local artifact paths, temporary folders, and deterministic file traversal.

    Args:
        abs (bool): Whether to return an absolute resolved path. Defaults to ``False``.
        aliases (Optional[Mapping[str, PathLike]]): Optional alias map used only for the first path segment. Defaults to ``None``.
        root (Optional[PathLike]): Optional root path used for relative path resolution. Defaults to ``None``.
        *parts (PathLike): Path parts or resource parts joined by the helper.

    Returns:
        str: Normalized path string, absolute when ``abs=True``.

    Examples:
        >>> from heavenbase.utils.files import pj
        >>> pj("a", "b").replace("\\", "/")
        'a/b'
    """
    cleaned = [_clean(part) for part in parts if part is not None and _clean(part)]
    if not cleaned:
        cleaned = [os.fspath(root)] if root is not None else [os.curdir]
    cleaned = _apply_alias(cleaned[0], aliases) + cleaned[1:]
    if root is not None and cleaned and not Path(os.path.expandvars(os.path.expanduser(cleaned[0]))).is_absolute():
        cleaned = [os.fspath(root), *cleaned]
    path = Path(os.path.expandvars(os.path.expanduser(os.path.join(*cleaned))))
    return str(path.resolve()) if abs or path.is_absolute() else os.path.normpath(str(path))


def get_file_ext(path: PathLike) -> str:
    """
    Return a file extension without the leading dot.

    Use this helper for local artifact paths, temporary folders, and deterministic file traversal.

    Args:
        path (PathLike): Local path, config path, or dotted dictionary path used by the helper.

    Returns:
        str: The a file extension without the leading dot.
    """
    return Path(pj(path)).suffix.lstrip(".")


def get_file_name(path: PathLike, ext: Union[bool, str] = True, abs: bool = False) -> str:
    """
    Return a path with its extension kept, removed, or replaced.

    Use this helper for local artifact paths, temporary folders, and deterministic file traversal.

    Args:
        path (PathLike): Local path, config path, or dotted dictionary path used by the helper.
        ext (Union[bool, str]): Extension policy. ``True`` keeps an extension, ``False`` removes it, and a string replaces or filters it. Defaults to ``True``.
        abs (bool): Whether to return an absolute resolved path. Defaults to ``False``.

    Returns:
        str: The a path with its extension kept, removed, or replaced.
    """
    value = pj(path, abs=abs)
    if ext is True:
        return value
    stem = str(Path(value).with_suffix(""))
    if not ext:
        return stem
    suffix = str(ext)
    return f"{stem}{suffix if suffix.startswith('.') else f'.{suffix}'}"


def get_file_basename(path: PathLike, ext: Union[bool, str] = True) -> str:
    """
    Return the basename from `get_file_name`.

    Use this helper for local artifact paths, temporary folders, and deterministic file traversal.

    Args:
        path (PathLike): Local path, config path, or dotted dictionary path used by the helper.
        ext (Union[bool, str]): Extension policy. ``True`` keeps an extension, ``False`` removes it, and a string replaces or filters it. Defaults to ``True``.

    Returns:
        str: The the basename from `get_file_name`.
    """
    return Path(get_file_name(path, ext=ext)).name


def get_file_dir(path: PathLike, abs: bool = False) -> str:
    """
    Return the directory containing a path.

    Use this helper for local artifact paths, temporary folders, and deterministic file traversal.

    Args:
        path (PathLike): Local path, config path, or dotted dictionary path used by the helper.
        abs (bool): Whether to return an absolute resolved path. Defaults to ``False``.

    Returns:
        str: The the directory containing a path.
    """
    return str(Path(pj(path, abs=abs)).parent)


def has_file_ext(path: PathLike, ext: Union[None, str, Sequence[Optional[str]]] = None) -> bool:
    """
    Return whether a path has one of the requested extensions.

    Use this helper for local artifact paths, temporary folders, and deterministic file traversal.

    Args:
        path (PathLike): Local path, config path, or dotted dictionary path used by the helper.
        ext (Union[None, str, Sequence[Optional[str]]]): Extension policy. ``True`` keeps an extension, ``False`` removes it, and a string replaces or filters it. Defaults to ``None``.

    Returns:
        bool: The whether a path has one of the requested extensions.
    """
    value = get_file_ext(path)
    if ext is None:
        return bool(value)
    items = [ext] if isinstance(ext, str) else list(ext)
    if None in items and not value:
        return True
    wanted = {part.lstrip(".") for item in items if item is not None for part in re.split(r"[;,]", str(item)) if part.strip()}
    return value in wanted


def touch_file(path: PathLike, encoding: str = "utf-8", clear: bool = False, content: Optional[str] = None) -> str:
    """
    Create a file and any missing parent directories, optionally replacing its content.

    Use this helper for local artifact paths, temporary folders, and deterministic file traversal.

    Args:
        path (PathLike): Local path, config path, or dotted dictionary path used by the helper.
        encoding (str): Text encoding. Defaults to ``"utf-8"``.
        clear (bool): Whether to clear an existing file or directory before writing. Defaults to ``False``.
        content (Optional[str]): Optional file content to write. Defaults to ``None``.

    Returns:
        str: Absolute path to the created or existing file.

    Examples:
        >>> from heavenbase.utils.files import exists_file, touch_file
        >>> path = touch_file("tmp/example.txt", content="ready")
        >>> exists_file(path)
        True
    """
    value = Path(pj(path, abs=True))
    value.parent.mkdir(parents=True, exist_ok=True)
    if value.exists() and not value.is_file():
        raise FileExistsError(f"Path {value} exists and is not a file")
    if value.exists() and not clear and content is None:
        return str(value)
    value.write_text("" if content is None else f"{content}\n", encoding=encoding, errors="ignore")
    return str(value)


def touch_dir(path: PathLike, clear: bool = False) -> str:
    """
    Create a directory and optionally clear any existing tree before recreating it.

    Use this helper for local artifact paths, temporary folders, and deterministic file traversal.

    Args:
        path (PathLike): Local path, config path, or dotted dictionary path used by the helper.
        clear (bool): Whether to clear an existing file or directory before writing. Defaults to ``False``.

    Returns:
        str: Absolute path to the created directory.
    """
    value = Path(pj(path, abs=True))
    if clear and value.exists():
        shutil.rmtree(value)
    value.mkdir(parents=True, exist_ok=True)
    return str(value)


def temp_file(path: PathLike, suffix: str = ".tmp") -> str:
    """Create an empty same-directory temporary file for an eventual replace.

    Args:
        path (PathLike): Target path whose parent directory receives the temp file.
        suffix (str): Temporary-file suffix. Defaults to ``".tmp"``.

    Returns:
        str: Absolute path to the created temporary file.
    """
    value = Path(pj(path, abs=True))
    value.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{value.name}.", suffix=suffix, dir=str(value.parent))
    os.close(fd)
    return tmp


def replace_file(src: PathLike, dst: PathLike) -> str:
    """Atomically replace ``dst`` with ``src`` when the filesystem supports it.

    Args:
        src (PathLike): Existing source file path.
        dst (PathLike): Destination path to replace.

    Returns:
        str: Absolute path to the replaced destination file.
    """
    out = Path(pj(dst, abs=True))
    Path(pj(src, abs=True)).replace(out)
    return str(out)


def exists_file(path: PathLike) -> bool:
    """
    Return whether path exists and is a file.

    Use this helper for local artifact paths, temporary folders, and deterministic file traversal.

    Args:
        path (PathLike): Local path, config path, or dotted dictionary path used by the helper.

    Returns:
        bool: The whether path exists and is a file.
    """
    return Path(pj(path, abs=True)).is_file()


def exists_dir(path: PathLike) -> bool:
    """
    Return whether path exists and is a directory.

    Use this helper for local artifact paths, temporary folders, and deterministic file traversal.

    Args:
        path (PathLike): Local path, config path, or dotted dictionary path used by the helper.

    Returns:
        bool: The whether path exists and is a directory.
    """
    return Path(pj(path, abs=True)).is_dir()


def exists_path(path: PathLike) -> bool:
    """
    Return whether path exists.

    Use this helper for local artifact paths, temporary folders, and deterministic file traversal.

    Args:
        path (PathLike): Local path, config path, or dotted dictionary path used by the helper.

    Returns:
        bool: The whether path exists.
    """
    return Path(pj(path, abs=True)).exists()


def is_link(path: PathLike) -> bool:
    """Return whether path is a symbolic link without following it.

    Use this helper when local artifact capture or cleanup must reject or skip
    symlinks before ordinary file or directory classification. Unlike `pj` with
    `abs=True`, this check expands `~` and environment variables but does not
    resolve through the link target.

    Args:
        path (PathLike): Local filesystem path to inspect.

    Returns:
        bool: `True` when the path is a symbolic link; otherwise `False`.
    """
    return Path(os.path.expandvars(os.path.expanduser(os.fspath(path)))).is_symlink()


def path_inside(root: PathLike, path: PathLike) -> bool:
    """
    Return whether a path resolves inside a root directory.

    Use this helper before reading user-supplied file lists, archives, or copied paths where symlinks could escape the declared root.

    Args:
        root (PathLike): Root directory that should contain the path.
        path (PathLike): Candidate child path.

    Returns:
        bool: Whether ``path`` resolves inside ``root``.
    """
    root_path = Path(pj(root, abs=True)).resolve()
    path_value = Path(pj(path, abs=True)).resolve()
    try:
        path_value.relative_to(root_path)
    except ValueError:
        return False
    return True


def _sorted_entries(path: Path, want_dir: bool, ext: Union[None, str, Sequence[str]] = None, abs: bool = False) -> List[str]:
    if not path.is_dir():
        return []
    out = []
    for item in path.iterdir():
        if item.is_dir() == want_dir and (want_dir or ext is None or has_file_ext(item, ext)):
            out.append(str(item.resolve()) if abs else item.name)
    return sorted(out, key=lambda name: name.lower())


def list_files(path: PathLike, ext: Union[None, str, Sequence[str]] = None, abs: bool = False) -> Generator[str, None, None]:
    """Yield direct child files in stable case-insensitive order.

    Use this helper for local artifact paths, temporary folders, and deterministic file traversal.

    Args:
        path (PathLike): Local path, config path, or dotted dictionary path used by the helper.
        ext (Union[None, str, Sequence[str]]): Extension policy. ``True`` keeps an extension, ``False`` removes it, and a string replaces or filters it. Defaults to ``None``.
        abs (bool): Whether to return an absolute resolved path. Defaults to ``False``.

    Yields:
        str: Direct child file names or absolute paths in stable order.

    Examples:
        >>> from heavenbase.utils.files import list_files
        >>> sorted(list_files("tmp", ext="txt"))
        ['example.txt']
    """
    yield from _sorted_entries(Path(pj(path, abs=True)), False, ext=ext, abs=abs)


def list_dirs(path: PathLike, abs: bool = False) -> Generator[str, None, None]:
    """Yield direct child directories in stable case-insensitive order.

    Use this helper for local artifact paths, temporary folders, and deterministic file traversal.

    Args:
        path (PathLike): Local path, config path, or dotted dictionary path used by the helper.
        abs (bool): Whether to return an absolute resolved path. Defaults to ``False``.

    Yields:
        str: Direct child directory names or absolute paths in stable order.
    """
    yield from _sorted_entries(Path(pj(path, abs=True)), True, abs=abs)


def list_paths(path: PathLike, ext: Union[None, str, Sequence[str]] = None, abs: bool = False) -> Generator[str, None, None]:
    """Yield direct child directories first, then files.

    Use this helper for local artifact paths, temporary folders, and deterministic file traversal.

    Args:
        path (PathLike): Local path, config path, or dotted dictionary path used by the helper.
        ext (Union[None, str, Sequence[str]]): Extension policy. ``True`` keeps an extension, ``False`` removes it, and a string replaces or filters it. Defaults to ``None``.
        abs (bool): Whether to return an absolute resolved path. Defaults to ``False``.

    Yields:
        str: Direct child directories followed by files.
    """
    yield from list_dirs(path, abs=abs)
    yield from list_files(path, ext=ext, abs=abs)


def _raise_walk_error(error: OSError) -> None:
    """Propagate an error reported by `os.walk`."""
    raise error


def _walk(path: Path, dirs: bool, files: bool, ext: Union[None, str, Sequence[str]], abs: bool) -> Generator[str, None, None]:
    if not path.is_dir():
        return
    for root, dir_names, file_names in os.walk(path, onerror=_raise_walk_error):
        dir_names.sort(key=str.lower)
        file_names.sort(key=str.lower)
        root_path = Path(root)
        if dirs:
            for name in dir_names:
                item = root_path / name
                yield str(item.resolve()) if abs else os.path.relpath(item, path)
        if files:
            for name in file_names:
                item = root_path / name
                if ext is None or has_file_ext(item, ext):
                    yield str(item.resolve()) if abs else os.path.relpath(item, path)


def enum_files(path: PathLike, ext: Union[None, str, Sequence[str]] = None, abs: bool = False) -> Generator[str, None, None]:
    """Yield files recursively in stable order.

    Use this helper for local artifact paths, temporary folders, and deterministic file traversal.

    Args:
        path (PathLike): Local path, config path, or dotted dictionary path used by the helper.
        ext (Union[None, str, Sequence[str]]): Extension policy. ``True`` keeps an extension, ``False`` removes it, and a string replaces or filters it. Defaults to ``None``.
        abs (bool): Whether to return an absolute resolved path. Defaults to ``False``.

    Yields:
        str: Recursive file paths in stable order.

    Raises:
        OSError: If a directory below `path` cannot be traversed.
    """
    yield from _walk(Path(pj(path, abs=True)), dirs=False, files=True, ext=ext, abs=abs)


def enum_dirs(path: PathLike, abs: bool = False) -> Generator[str, None, None]:
    """Yield directories recursively in stable order.

    Use this helper for local artifact paths, temporary folders, and deterministic file traversal.

    Args:
        path (PathLike): Local path, config path, or dotted dictionary path used by the helper.
        abs (bool): Whether to return an absolute resolved path. Defaults to ``False``.

    Yields:
        str: Recursive directory paths in stable order.

    Raises:
        OSError: If a directory below `path` cannot be traversed.
    """
    yield from _walk(Path(pj(path, abs=True)), dirs=True, files=False, ext=None, abs=abs)


def enum_paths(path: PathLike, ext: Union[None, str, Sequence[str]] = None, abs: bool = False) -> Generator[str, None, None]:
    """Yield directories and files recursively in stable order.

    Use this helper for local artifact paths, temporary folders, and deterministic file traversal.

    Args:
        path (PathLike): Local path, config path, or dotted dictionary path used by the helper.
        ext (Union[None, str, Sequence[str]]): Extension policy. ``True`` keeps an extension, ``False`` removes it, and a string replaces or filters it. Defaults to ``None``.
        abs (bool): Whether to return an absolute resolved path. Defaults to ``False``.

    Yields:
        str: Recursive directory and file paths in stable order.

    Raises:
        OSError: If a directory below `path` cannot be traversed.
    """
    yield from _walk(Path(pj(path, abs=True)), dirs=True, files=True, ext=ext, abs=abs)


def empty_dir(path: PathLike, ext: Union[None, str, Sequence[str]] = None) -> bool:
    """
    Return whether a directory has no matching files below it.

    Use this helper for local artifact paths, temporary folders, and deterministic file traversal.

    Args:
        path (PathLike): Local path, config path, or dotted dictionary path used by the helper.
        ext (Union[None, str, Sequence[str]]): Extension policy. ``True`` keeps an extension, ``False`` removes it, and a string replaces or filters it. Defaults to ``None``.

    Returns:
        bool: ``True`` when the directory has no matching files; otherwise ``False``.
    """
    value = Path(pj(path, abs=True))
    return value.is_dir() and next(enum_files(value, ext=ext), None) is None


def nonempty_dir(path: PathLike, ext: Union[None, str, Sequence[str]] = None) -> bool:
    """
    Return whether a directory has at least one matching file below it.

    Use this helper for local artifact paths, temporary folders, and deterministic file traversal.

    Args:
        path (PathLike): Local path, config path, or dotted dictionary path used by the helper.
        ext (Union[None, str, Sequence[str]]): Extension policy. ``True`` keeps an extension, ``False`` removes it, and a string replaces or filters it. Defaults to ``None``.

    Returns:
        bool: ``True`` when the directory has at least one matching file; otherwise ``False``.
    """
    return next(enum_files(path, ext=ext), None) is not None


def delete_file(path: PathLike) -> bool:
    """Delete a file if it exists.

    Use this helper for local artifact paths, temporary folders, and deterministic file traversal.

    Warnings:
        This helper deletes local data. Resolve paths explicitly before calling it outside temporary folders.

    Args:
        path (PathLike): Local path, config path, or dotted dictionary path used by the helper.

    Returns:
        bool: ``True`` when a file was deleted; ``False`` when it was already absent.
    """
    value = Path(pj(path, abs=True))
    if not value.exists():
        return False
    if not value.is_file():
        raise FileExistsError(f"Path {value} exists and is not a file")
    value.unlink()
    return True


def delete_dir(path: PathLike) -> bool:
    """Delete a directory tree if it exists.

    Use this helper for local artifact paths, temporary folders, and deterministic file traversal.

    Warnings:
        This helper deletes a full directory tree. Keep cleanup targets under known temporary roots.

    Args:
        path (PathLike): Local path, config path, or dotted dictionary path used by the helper.

    Returns:
        bool: ``True`` when a directory was deleted; ``False`` when it was already absent.
    """
    value = Path(pj(path, abs=True))
    if not value.exists():
        return False
    if not value.is_dir():
        raise FileExistsError(f"Path {value} exists and is not a directory")
    shutil.rmtree(value)
    return True


def delete_path(path: PathLike) -> bool:
    """Delete either a file or directory tree.

    Use this helper for local artifact paths, temporary folders, and deterministic file traversal.

    Warnings:
        This helper deletes either a file or a directory tree. Keep cleanup targets under known temporary roots.

    Args:
        path (PathLike): Local path, config path, or dotted dictionary path used by the helper.

    Returns:
        bool: ``True`` when a path was deleted; ``False`` when it was already absent.
    """
    value = Path(pj(path, abs=True))
    if not value.exists():
        return False
    return delete_dir(value) if value.is_dir() else delete_file(value)


def _copy_mode(mode: Optional[Union[str, bool]], overwrite: Optional[bool], default: str, supported: Sequence[str]) -> str:
    if isinstance(mode, bool):
        mode = "replace" if mode else "skip"
    elif mode is None and overwrite is not None:
        mode = "replace" if overwrite else "skip"
    else:
        mode = mode or default
    if mode not in supported:
        raise ValueError(f"Unsupported copy mode {mode!r}. Supported: {list(supported)}")
    return str(mode)


def copy_file(src: PathLike, dst: PathLike, mode: Optional[Union[CopyFileMode, bool]] = None, overwrite: Optional[bool] = None) -> str:
    """
    Copy one file with explicit conflict handling.

    Use this helper for local artifact paths, temporary folders, and deterministic file traversal.

    Args:
        src (PathLike): Source file or directory path.
        dst (PathLike): Destination file or directory path.
        mode (Optional[Union[CopyFileMode, bool]]): Operation mode. Literal choices are described by the helper body and validation errors. Defaults to ``None``.
        overwrite (Optional[bool]): Legacy conflict flag mapped to an explicit copy mode. Defaults to ``None``.

    Returns:
        str: Absolute destination file path.
    """
    mode = _copy_mode(mode, overwrite, "replace", ("replace", "skip", "strict"))
    source = Path(pj(src, abs=True))
    target = Path(pj(dst, abs=True))
    if not source.is_file():
        raise FileNotFoundError(f"Source file {source} does not exist or is not a file.")
    if target.exists():
        if target.is_dir():
            raise FileExistsError(f"Destination {target} exists and is a directory.")
        if mode == "strict":
            raise FileExistsError(f"Destination file {target} already exists.")
        if mode == "skip":
            return str(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return str(target)


def _copy_dir_skip(source: Path, target: Path) -> None:
    for item in sorted(source.rglob("*"), key=lambda path: (not path.is_dir(), str(path).lower())):
        dst_item = target / item.relative_to(source)
        if item.is_dir():
            dst_item.mkdir(parents=True, exist_ok=True)
        elif not dst_item.exists():
            dst_item.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, dst_item)


def _copy_dir_conflicts(source: Path, target: Path) -> List[str]:
    return [str(target / item.relative_to(source)) for item in source.rglob("*") if (target / item.relative_to(source)).exists()]


def copy_dir(src: PathLike, dst: PathLike, mode: Optional[Union[CopyDirMode, bool]] = None, overwrite: Optional[bool] = None) -> str:
    """
    Copy a directory tree with explicit conflict handling.

    Use this helper for local artifact paths, temporary folders, and deterministic file traversal.

    Args:
        src (PathLike): Source file or directory path.
        dst (PathLike): Destination file or directory path.
        mode (Optional[Union[CopyDirMode, bool]]): Operation mode. Literal choices are described by the helper body and validation errors. Defaults to ``None``.
        overwrite (Optional[bool]): Legacy conflict flag mapped to an explicit copy mode. Defaults to ``None``.

    Returns:
        str: Absolute destination directory path.
    """
    mode = _copy_mode(mode, overwrite, "merge", ("replace", "skip", "strict", "merge"))
    source = Path(pj(src, abs=True))
    target = Path(pj(dst, abs=True))
    if not source.is_dir():
        raise FileNotFoundError(f"Source directory {source} does not exist or is not a directory.")
    if target.exists() and not target.is_dir():
        raise FileExistsError(f"Destination {target} exists and is not a directory.")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and mode == "replace":
        shutil.rmtree(target)
    if target.exists() and mode == "skip":
        _copy_dir_skip(source, target)
        return str(target)
    if target.exists() and mode == "strict":
        conflicts = _copy_dir_conflicts(source, target)
        if conflicts:
            raise FileExistsError(f"Destination directory {target} has conflicting paths: {', '.join(conflicts)}")
    shutil.copytree(source, target, dirs_exist_ok=mode in ("merge", "strict"))
    return str(target)


def copy_path(src: PathLike, dst: PathLike, mode: Optional[Union[CopyDirMode, bool]] = None, overwrite: Optional[bool] = None) -> str:
    """
    Copy either a file or directory tree with mode-aware conflict handling.

    Use this helper for local artifact paths, temporary folders, and deterministic file traversal.

    Args:
        src (PathLike): Source file or directory path.
        dst (PathLike): Destination file or directory path.
        mode (Optional[Union[CopyDirMode, bool]]): Operation mode. Literal choices are described by the helper body and validation errors. Defaults to ``None``.
        overwrite (Optional[bool]): Legacy conflict flag mapped to an explicit copy mode. Defaults to ``None``.

    Returns:
        str: Absolute destination path.

    Examples:
        >>> from heavenbase.utils.files import copy_path
        >>> copy_path("tmp/example.txt", "tmp/latest.txt", mode="replace").endswith("latest.txt")
        True
    """
    source = Path(pj(src, abs=True))
    if source.is_dir():
        return copy_dir(src, dst, mode=mode, overwrite=overwrite)
    if source.is_file():
        file_mode = "replace" if mode in (None, "merge") else mode
        return copy_file(src, dst, mode=file_mode, overwrite=overwrite)
    raise FileNotFoundError(f"Source path {source} does not exist or is not a file/directory.")


def folder_diagram(
    path: PathLike,
    annotations: Optional[Mapping[str, str]] = None,
    name: Optional[str] = None,
    limit: int = 16,
    max_depth: Optional[int] = None,
    prefix: str = "",
) -> str:
    """
    Render a compact text tree for a local file or directory.

    Use this helper for local artifact paths, temporary folders, and deterministic file traversal.

    Args:
        path (PathLike): Local path, config path, or dotted dictionary path used by the helper.
        annotations (Optional[Mapping[str, str]]): Optional mapping from relative path to note text shown beside that path. Defaults to ``None``.
        name (Optional[str]): Name, label, scope, package, or logger identifier used by the helper. Defaults to ``None``.
        limit (int): Maximum number of items, rows, or rendered entries. Defaults vary by helper. Defaults to ``16``.
        max_depth (Optional[int]): Maximum directory recursion depth. ``None`` means unlimited depth. Defaults to ``None``.
        prefix (str): Text prefix inserted before rendered lines. Defaults to ``""``.

    Returns:
        str: Text tree diagram for the requested path.
    """
    root = Path(pj(path, abs=True))
    if not root.exists():
        raise ValueError(f"Path {root} is neither a file nor a directory.")
    annotations = annotations or {}

    def note(rel: str, fallback: str) -> str:
        value = annotations.get(rel) or annotations.get(fallback)
        return f"  # {value}" if value else ""

    label = name or root.name
    if root.is_file():
        return f"{label}{note(root.name, root.name)}"

    lines: List[str] = []

    def visible(entries: List[Path]) -> List[Optional[Path]]:
        if limit < 0 or len(entries) <= limit:
            return entries
        bottom = max(limit, 0) // 2
        top = max(limit - bottom, 0)
        return [*entries[:top], None, *entries[-bottom:]]

    def visit(current: Path, depth: int, lead: str) -> None:
        if max_depth is not None and depth > max_depth:
            return
        entries = sorted(current.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower()))
        shown = visible(entries)
        for pos, item in enumerate(shown):
            marker = "`-- " if pos == len(shown) - 1 else "|-- "
            if item is None:
                lines.append(f"{lead}{marker}... ({len(entries) - len(shown) + 1} omitted)")
                continue
            rel = item.relative_to(root).as_posix()
            name = f"{item.name}/" if item.is_dir() else item.name
            lines.append(f"{lead}{marker}{name}{note(rel, item.name)}")
            if item.is_dir():
                visit(item, depth + 1, lead + ("    " if pos == len(shown) - 1 else "|   "))

    lines.append(f"{prefix}{label}/")
    visit(root, 1, prefix)
    return "\n".join(lines)
