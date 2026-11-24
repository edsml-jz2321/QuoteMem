"""Capsule error types."""

from __future__ import annotations

__all__ = ["CapsuleCapabilityError", "CapsuleCaptureError", "CapsuleError", "CapsuleNotFoundError", "CapsuleRestoreError"]

from heavenbase.utils import Any, List


class CapsuleError(Exception):
    """Base error for Capsule operations."""


class CapsuleCaptureError(CapsuleError):
    """Raised when a callable cannot be captured."""


class CapsuleCapabilityError(CapsuleError):
    """Raised when a Capsule cannot be exported to a requested target."""


class CapsuleNotFoundError(KeyError, CapsuleError):
    """Raised when a Capsule registry has no matching active reference."""


class CapsuleRestoreError(CapsuleError):
    """Raised when all restore layers fail."""

    def __init__(self, capsule_id: str, errors: List[Any]):
        """Record restore failures for one Capsule id.

        Args:
            capsule_id (str): Stable Capsule identifier that could not be restored.
            errors (List[Any]): Layer-level failure messages collected during restore.

        Returns:
            None: This constructor does not return a value.
        """
        self.capsule_id = capsule_id
        self.errors = errors
        super().__init__(f"Failed to restore capsule {capsule_id!r}: {errors}")
