"""Prompt extension name and row object_id helpers."""

from __future__ import annotations

__all__ = ["check_prompt_name", "prompt_row_id", "translation_row_id"]

from heavenbase.utils import Optional, hash_id
from heavenbase.utils.ids import check_id


def check_prompt_name(name: str) -> str:
    """Validate and normalize a dotted prompt name.

    Args:
        name (str): Dotted prompt name made of safe id segments.

    Returns:
        str: Normalized dotted prompt name.

    Raises:
        ValueError: If the name is empty, has empty segments, or contains unsafe id characters.

    Examples:
        >>> check_prompt_name("demo.greet")
        'demo.greet'
    """
    value = str(name or "").strip()
    parts = value.split(".")
    if not value or any(not part for part in parts):
        raise ValueError(f"Invalid prompt name {name!r}; use one or more safe id segments joined by dots")
    return ".".join(check_id(part, "prompt name segment") for part in parts)


def _check_version(version: str) -> str:
    value = str(version or "").strip()
    if not value or ":" in value:
        raise ValueError(f"Invalid prompt version {version!r}")
    return value


def prompt_row_id(name: str, version: str = "1") -> str:
    """Return the stable row object_id for a prompt name and version.

    Args:
        name (str): Dotted prompt name.
        version (str): Prompt version. Defaults to ``"1"``.

    Returns:
        str: Stable hashed prompt row object_id.
    """
    return hash_id("prompt", check_prompt_name(name), _check_version(version))


def translation_row_id(dict_name: str, source_key: str, lang: str) -> str:
    """Return the stable row object_id for one translation entry.

    Args:
        dict_name (str): Dotted translation dictionary name.
        source_key (str): Source text or source template.
        lang (str): Target language key.

    Returns:
        str: Stable hashed translation row object_id.

    Raises:
        ValueError: If ``lang`` is empty or ``dict_name`` is invalid.
    """
    lang = str(lang or "").strip()
    if not lang:
        raise ValueError("Translation language is required")
    return hash_id("translation", check_prompt_name(dict_name), str(source_key), lang)


def parse_prompt_ref(ref: str) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Parse a prompt ref into ``(name, version, row_id)``.

    Args:
        ref (str): Prompt reference string. Accepts ``name``, ``name:version``,
            or a raw row ``object_id``.

    Returns:
        tuple[Optional[str], Optional[str], Optional[str]]: Parsed ``name``,
        ``version``, and ``row_id`` when the ref is a raw object id.

    Raises:
        ValueError: If ``ref`` is empty or a version token is invalid.
    """
    raw = str(ref or "").strip()
    if not raw:
        raise ValueError("Prompt ref is required")
    if ":" in raw:
        name, version = raw.rsplit(":", 1)
        return check_prompt_name(name), _check_version(version), None
    try:
        return check_prompt_name(raw), None, None
    except ValueError:
        return None, None, raw
