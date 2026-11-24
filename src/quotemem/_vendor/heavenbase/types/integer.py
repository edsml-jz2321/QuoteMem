"""Integer logical type."""

from __future__ import annotations

__all__ = [
    "Integer",
]

from heavenbase.utils import Any

from .base import LogicalType
from .filtering import normalize_scalar_filter_value


class Integer(LogicalType):
    """Integer logical type."""

    identifier = "integer"

    def validate(self, value: Any) -> int:
        """Coerce a value to int.

        Args:
            value (Any): Numeric or string input accepted by ``int(value)``.

        Returns:
            int: Coerced integer value.

        Raises:
            ValueError: If ``value`` cannot be coerced to int.
            TypeError: If ``value`` has an unsupported type for int coercion.
        """
        return int(value)

    def filter_value(self, op: str, value: Any) -> Any:
        """Normalize numeric filter operands through :meth:`validate`.

        Args:
            op (str): Filter operation identifier such as ``eq``, ``ne``, or ``in``.
            value (Any): User-provided query operand.

        ``None`` stays ``None`` for ``eq``, ``ne`` and ``in`` so the filter
        tests for a missing value.

        Returns:
            Any: Canonical int operand, ``None`` for a missing-value test,
            list of operands for ``in``, or the unchanged operand for
            operations without numeric normalization.

        Raises:
            ValueError: If an order filter (``lt``, ``lte``, ``gt``, ``gte``)
                is given ``None``, or the operand cannot be coerced.
            TypeError: If the operand has an unsupported type for coercion.
        """
        return normalize_scalar_filter_value(self, op, value)
