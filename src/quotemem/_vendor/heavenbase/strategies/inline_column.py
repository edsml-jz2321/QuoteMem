"""Inline column storage strategy."""

__all__ = [
    "InlineColumn",
]

from .base import StorageStrategy


class InlineColumn(StorageStrategy):
    """Marker for storing a field inline with the main backend row."""

    identifier = "inline"
