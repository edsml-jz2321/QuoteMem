"""External reference storage strategy."""

__all__ = [
    "ExternalRef",
]

from .base import StorageStrategy


class ExternalRef(StorageStrategy):
    """Marker for storing only an external reference to the field value."""

    identifier = "external"
