"""Deterministic provider names for shared physical namespaces."""

from __future__ import annotations

from heavenbase.utils.hash import sha256hash

from .base import Backend


def provider_scope(
    ws_id: str | None,
    backend_name: str,
    backend_type: str,
    *,
    separator: str = "_",
    prefix: str = "hb",
    max_length: int = 63,
) -> str:
    """Return one bounded owner token containing workspace, name, and type."""
    readable = Backend._physical_name(ws_id or "standalone", backend_name, sep=separator).lower()
    digest = sha256hash([ws_id or "standalone", backend_name, backend_type])[:12]
    fixed = len(prefix) + len(digest) + 2 * len(separator)
    available = max_length - fixed
    if available < 1:
        raise ValueError("Provider identifier limit leaves no room for a scope name")
    return f"{prefix}{separator}{readable[:available]}{separator}{digest}"


def scoped_provider_name(
    scope: str,
    *parts: str,
    separator: str = "_",
    max_length: int = 255,
) -> str:
    """Append logical name parts to a bounded, unambiguous owner scope."""
    logical = Backend._physical_name(*parts, sep=separator).lower()
    available = max_length - len(scope) - len(separator)
    if available < 14:
        raise ValueError("Provider identifier limit leaves no room for a logical name")
    if len(logical) > available:
        digest = sha256hash(logical)[:12]
        logical = f"{logical[: available - len(digest) - len(separator)]}{separator}{digest}"
    return f"{scope}{separator}{logical}"
