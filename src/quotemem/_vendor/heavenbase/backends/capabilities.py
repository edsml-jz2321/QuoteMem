"""Backend-facing exports for structured Tag values."""

from __future__ import annotations

__all__ = [
    "DEFAULT_INDEXABLE_TYPES",
    "ScalarIndexSpec",
]

from ..placement.scalar import ScalarIndexSpec

DEFAULT_INDEXABLE_TYPES = (
    "boolean",
    "categorical",
    "date",
    "duration",
    "float",
    "identifier",
    "integer",
    "long-text",
    "medium-text",
    "short-text",
    "timestamp",
)
