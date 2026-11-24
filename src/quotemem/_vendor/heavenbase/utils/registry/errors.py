"""Define Registry errors."""

from __future__ import annotations

__all__ = ["RegistryConflictError", "RegistryIntegrityError"]


class RegistryConflictError(RuntimeError):
    """Report a failed persistent Registry compare-and-set operation."""


class RegistryIntegrityError(RuntimeError):
    """Report invalid, corrupt, or rolled-back persistent Registry state."""
