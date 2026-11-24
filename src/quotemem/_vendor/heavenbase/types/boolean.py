"""Boolean logical type with optional three-value (True/False/None) support."""

from __future__ import annotations

__all__ = [
    "Boolean",
]

from heavenbase.utils import Any, Optional

from .base import LogicalType
from .filtering import normalize_scalar_filter_value

_NULL_STRINGS = frozenset({"null", "none", "nil", "nan", "na"})
_FALSE_STRINGS = frozenset({"false", "0", "no", "off", ""})
_TRUE_STRINGS = frozenset({"true", "1", "yes", "on"})


class Boolean(LogicalType):
    """Boolean logical type with optional ``None``.

    Accepts ``True``, ``False``, and ``None``. String aliases:

    - True: ``"true"``, ``"1"``, ``"yes"``, ``"on"``
    - False: ``"false"``, ``"0"``, ``"no"``, ``"off"``, ``""``
    - None: ``"null"``, ``"none"``, ``"nil"``, ``"nan"``, ``"na"``

    Integers and floats: ``1``, ``1.0`` → ``True``; ``0``, ``0.0`` → ``False``.
    Other values fall back to Python ``bool(value)``.

    Examples:
        ``hb.Boolean().validate("false")`` returns ``False``.
        ``hb.Boolean().validate("null")`` returns ``None``.
        ``hb.Boolean().validate(1)`` returns ``True``.
    """

    identifier = "boolean"

    def validate(self, value: Any) -> Optional[bool]:
        """Coerce a value to ``bool``, ``None``, or raise.

        Args:
            value: Boolean scalar, recognised string alias, ``0``/``1`` numeric
                input, or any value handled by ``bool(value)``.

        Returns:
            ``True``, ``False``, or ``None``.

        Raises:
            ValueError: If a string or numeric value cannot be coerced.
        """
        if value is None:
            return None
        if isinstance(value, str):
            v = value.strip().lower()
            if v in _TRUE_STRINGS:
                return True
            if v in _FALSE_STRINGS:
                return False
            if v in _NULL_STRINGS:
                return None
            raise ValueError(f"Cannot coerce {value!r} to Boolean")
        if isinstance(value, (int, float)):
            if value == 0 or value == 0.0:
                return False
            if value == 1 or value == 1.0:
                return True
            raise ValueError(f"Cannot coerce numeric {value!r} to Boolean")
        return bool(value)

    def filter_value(self, op: str, value: Any) -> Any:
        """Canonicalize boolean query operands through :meth:`validate`.

        Without this, ``flag == 1`` keeps the raw ``1`` and a backend that keys its value
        index on ``str(operand)`` would probe ``"1"`` while the stored member is
        ``str(True)`` — a native false negative versus the scan predicate (``True == 1``).
        ``None`` passes through unchanged for ``eq``, ``ne`` and ``in`` so the filter tests
        for a missing value and handlers can route ``eq None`` to scan rather than indexing
        a non-existent NULL value. An order filter given ``None`` raises.

        Args:
            op (str): Filter operation identifier such as ``eq``, ``ne``, or ``in``.
            value (Any): User-provided query operand.

        Returns:
            Any: Canonical boolean operand, list of operands for ``in``, or the unchanged
            operand for operations without boolean normalization.

        Raises:
            ValueError: If an order filter (``lt``, ``lte``, ``gt``, ``gte``) is given
                ``None``, or a string or numeric operand cannot be coerced.
        """
        return normalize_scalar_filter_value(self, op, value)
