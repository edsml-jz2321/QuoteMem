"""Vector index storage strategy."""

__all__ = [
    "VectorIndex",
]

from .base import StorageStrategy


class VectorIndex(StorageStrategy):
    """Marker for storing vector fields in a vector-search index."""

    identifier = "vector"
