"""Logical type storage-affinity helpers."""

from __future__ import annotations

__all__ = ["stores_as_integer", "stores_as_text"]

from heavenbase.utils import Any


def stores_as_integer(dtype: Any) -> bool:
    """Return whether a logical type uses integer-like physical storage.

    Args:
        dtype (Any): Logical type class or instance.

    Returns:
        bool: True for integer, timestamp, duration, and integer-backed
        categoricals.
    """
    if isinstance(dtype, type):
        identifier = getattr(dtype, "identifier", None)
        if identifier in ("integer", "timestamp", "duration"):
            return True
        if identifier == "categorical":
            return False
        try:
            instance = dtype()
        except TypeError:
            return False
    else:
        instance = dtype
    if instance is None:
        return False
    identifier = type(instance).identifier
    if identifier in ("integer", "timestamp", "duration"):
        return True
    return identifier == "categorical" and instance.base == "integer"


def stores_as_text(dtype: Any) -> bool:
    """Return whether a logical type uses text-like physical storage.

    Args:
        dtype (Any): Logical type class or instance.

    Returns:
        bool: True for identifiers, text values, dates, and string-backed
        categoricals.
    """
    if isinstance(dtype, type):
        identifier = getattr(dtype, "identifier", None)
        if identifier in ("date", "identifier", "long-text", "medium-text", "short-text"):
            return True
        if identifier == "categorical":
            return False
        try:
            instance = dtype()
        except TypeError:
            return False
    else:
        instance = dtype
    if instance is None:
        return False
    identifier = type(instance).identifier
    if identifier in ("date", "identifier", "long-text", "medium-text", "short-text"):
        return True
    return identifier == "categorical" and instance.base == "string"
