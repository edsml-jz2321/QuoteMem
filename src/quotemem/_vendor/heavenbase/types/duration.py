"""Duration logical type."""

from __future__ import annotations

__all__ = [
    "Duration",
]

from heavenbase.utils import Any

from .integer import Integer


class Duration(Integer):
    """Non-negative elapsed time stored as an integer count of milliseconds.

    This is a semantic alias over :class:`Integer` for TTL and interval policy
    fields such as Catalog ``expires_after`` and ``stale_after``. A value of
    ``0`` means the policy is disabled.

    Examples:
        ``hb.Duration().validate("500")`` returns ``500``.
    """

    identifier = "duration"

    def validate(self, value: Any) -> int:
        """Coerce a value to a non-negative duration in milliseconds.

        Args:
            value: Numeric input coercible to ``int``.

        Returns:
            Non-negative duration in milliseconds.

        Raises:
            ValueError: If ``value`` is negative or cannot be coerced to ``int``.
        """
        if isinstance(value, bool):
            raise ValueError("Duration cannot be bool")
        try:
            out = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Duration expects a numeric value, got {type(value).__name__}") from exc
        if out < 0:
            raise ValueError(f"Duration cannot be negative, got {out}")
        return out
