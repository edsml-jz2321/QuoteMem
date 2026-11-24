"""JSON payload storage strategy."""

__all__ = [
    "JsonField",
]

from .base import StorageStrategy


class JsonField(StorageStrategy):
    """Marker for storing a field as a JSON payload."""

    identifier = "json"
