"""Deterministic archive helpers for agent Skills."""

from __future__ import annotations

__all__ = ["pack_skill_dir", "read_archive_file"]

from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from heavenbase.utils import Any, Dict, enum_files, exists_dir, exists_file, get_file_ext, load_bin, path_inside, pj, sha256hash

_ZIP_TIME = (1980, 1, 1, 0, 0, 0)
_TEXT_SUFFIXES = {"md", "txt", "json", "yaml", "yml", "py", "toml", "ini", "cfg"}


def _archive_path(path: str) -> str:
    clean = str(path or "").replace("\\", "/").strip("/")
    parts = clean.split("/")
    if not clean or clean.startswith("/") or any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"Invalid skill archive path {path!r}")
    return "/".join(parts)


def pack_skill_dir(path: str) -> tuple[bytes, Dict[str, Any]]:
    """Pack a skill folder into deterministic zip bytes and an index.

    Args:
        path: Skill folder containing ``SKILL.md``.

    Returns:
        tuple[bytes, Dict[str, Any]]: Zip bytes plus metadata keyed by relative path.

    Raises:
        FileNotFoundError: If ``SKILL.md`` is missing.
        ValueError: If ``path`` is not a directory.
    """
    root = pj(path, abs=True)
    if not exists_dir(root):
        raise ValueError(f"Skill path must be a directory: {path}")
    if not exists_file(pj(root, "SKILL.md")):
        raise FileNotFoundError(f"Skill folder needs SKILL.md: {path}")
    files = sorted(enum_files(root, abs=False))
    index: Dict[str, Any] = {}
    buffer = BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as archive:
        for rel in files:
            item = pj(root, rel)
            if not path_inside(root, item):
                raise ValueError(f"Skill file escaped root: {item}")
            arcname = _archive_path(rel)
            data = load_bin(item)
            info = ZipInfo(arcname, _ZIP_TIME)
            info.compress_type = ZIP_DEFLATED
            archive.writestr(info, data)
            index[arcname] = {
                "size": len(data),
                "hash": sha256hash(data),
                "text": get_file_ext(rel).lower() in _TEXT_SUFFIXES,
            }
    return buffer.getvalue(), index


def read_archive_file(archive: bytes, path: str) -> bytes:
    """Read one file from deterministic skill archive bytes.

    Args:
        archive: Zip archive bytes.
        path: Relative path inside the archive.

    Returns:
        bytes: Raw file contents.

    Raises:
        KeyError: If ``path`` is absent.
        ValueError: If ``path`` is unsafe.
    """
    arcname = _archive_path(path)
    with ZipFile(BytesIO(bytes(archive or b"")), "r") as zip_file:
        names = set(zip_file.namelist())
        if arcname not in names:
            raise KeyError(f"No skill file found for {arcname!r}")
        return zip_file.read(arcname)
