"""Build identities for registry-backed classes and named values."""

from __future__ import annotations

import re
from typing import Any

__all__ = ["class_identifier", "identifier_key", "named_key"]

_IDENTIFIER_RE = re.compile(r"^[a-z][a-z0-9]*(?:[-_][a-z0-9]+)*$")


def _check_identifier(value: str, label: str) -> str:
    clean = str(value or "").strip()
    if not _IDENTIFIER_RE.fullmatch(clean):
        raise ValueError(f"Invalid {label} {clean!r}; use lower-case letters, digits, '-' or '_', starting with a letter")
    return clean


def _kebab(name: str) -> str:
    out: list[str] = []
    for index, char in enumerate(name):
        if char.isupper() and index > 0:
            out.append("-")
        out.append(char.lower())
    return "".join(out).replace("_", "-")


def class_identifier(
    cls: type,
    *,
    identifier_attr: str = "identifier",
    label: str = "identifier",
) -> str:
    """Return the canonical identifier declared directly by a class.

    Inherited identifiers are ignored so subclasses derive their own identity
    unless they explicitly declare one.

    Args:
        cls (type): Registrable class to identify.
        identifier_attr (str): Direct class attribute that may declare identity.
        label (str): Human-readable identity label for validation errors.

    Returns:
        str: Validated canonical class identifier.

    Raises:
        ValueError: If the declared or derived identifier is invalid.
    """
    raw = cls.__dict__.get(identifier_attr)
    return _check_identifier(str(raw) if raw else _kebab(cls.__name__), label)


def identifier_key(value: Any, label: str = "identifier") -> str:
    """Normalize a value as a strict Registry identifier.

    Args:
        value (Any): Candidate identifier value.
        label (str): Human-readable identity label for validation errors.

    Returns:
        str: Validated canonical identifier.

    Raises:
        ValueError: If the value is not a valid identifier.
    """
    return _check_identifier(str(value).strip(), label)


def named_key(value: Any, label: str = "registry key") -> str:
    """Normalize a free-form registry key to stripped non-empty text.

    Args:
        value (Any): Candidate registry key.
        label (str): Human-readable key label for validation errors.

    Returns:
        str: Stripped non-empty key.

    Raises:
        ValueError: If the value is empty after stripping whitespace.
    """
    key = str(value).strip()
    if not key:
        raise ValueError(f"{label} must be non-empty")
    return key
