"""Side table storage strategy."""

__all__ = [
    "SideTable",
]

from .base import StorageStrategy


class SideTable(StorageStrategy):
    """Marker for storing repeatable field items in a side table."""

    identifier = "side"
