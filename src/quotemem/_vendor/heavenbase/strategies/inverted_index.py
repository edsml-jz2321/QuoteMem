"""Inverted index storage strategy."""

__all__ = [
    "InvertedIndex",
]

from .base import StorageStrategy


class InvertedIndex(StorageStrategy):
    """Marker for storing text in an inverted full-text index."""

    identifier = "inverted"
