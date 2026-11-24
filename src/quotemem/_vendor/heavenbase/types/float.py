"""Float logical type."""

from __future__ import annotations

__all__ = [
    "Float",
]

from heavenbase.utils import Any

from .base import LogicalType
from .filtering import normalize_scalar_filter_value


class Float(LogicalType):
    """Floating-point logical type."""

    identifier = "float"

    def validate(self, value: Any) -> float:
        """Coerce a value to float.

        Args:
            value (Any): Numeric or string input accepted by ``float(value)``.

        Returns:
            float: Coerced floating-point value.

        Raises:
            ValueError: If ``value`` cannot be coerced to float.
            TypeError: If ``value`` has an unsupported type for float coercion.
        """
        return float(value)

    def filter_value(self, op: str, value: Any) -> Any:
        """Normalize numeric filter operands through :meth:`validate`.

        Args:
            op (str): Filter operation identifier such as ``eq``, ``ne``, or ``in``.
            value (Any): User-provided query operand.

        ``None`` stays ``None`` for ``eq``, ``ne`` and ``in`` so the filter
        tests for a missing value.

        Returns:
            Any: Canonical float operand, ``None`` for a missing-value test,
            list of operands for ``in``, or the unchanged operand for
            operations without numeric normalization.

        Raises:
            ValueError: If an order filter (``lt``, ``lte``, ``gt``, ``gte``)
                is given ``None``, or the operand cannot be coerced.
            TypeError: If the operand has an unsupported type for coercion.
        """
        return normalize_scalar_filter_value(self, op, value)
