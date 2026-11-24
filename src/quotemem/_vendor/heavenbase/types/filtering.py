"""Shared logical-type query operand normalization helpers."""

from __future__ import annotations

from collections.abc import Iterable

from heavenbase.utils import Any

_ORDER_OPS = ("lt", "lte", "gt", "gte")


def require_order_bound(dtype: Any, op: str, value: Any) -> None:
    """Reject ``None`` as the bound of an order filter.

    ``None`` in an ``eq``, ``ne`` or ``in`` filter tests for a missing value.
    An order filter has no bound to compare with ``None``.

    Args:
        dtype: Logical type instance that owns the filtered field.
        op: Portable filter operation identifier.
        value: User-authored query operand.

    Returns:
        None: The operand is a usable order bound, or ``op`` is not an order
        operation.

    Raises:
        ValueError: If ``op`` is ``lt``, ``lte``, ``gt`` or ``gte`` and
            ``value`` is ``None``.
    """
    if value is None and op in _ORDER_OPS:
        raise ValueError(f"{type(dtype).__name__} {op!r} filter needs a value; None only matches a missing value through eq, ne or in")


def require_in_collection(value: Any) -> None:
    """Reject an ``in`` operand that is not a collection of candidates.

    A string would otherwise be read one character at a time, and a scalar
    would fail with Python's bare "object is not iterable" error.

    Args:
        value: User-authored ``in`` operand.

    Returns:
        None: The operand is a non-string iterable.

    Raises:
        TypeError: If ``value`` is a string, bytes or not iterable.
    """
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Iterable):
        raise TypeError(f"in filter expects a collection of values such as a list, tuple, or set, got {type(value).__name__}")


def normalize_scalar_filter_value(dtype: Any, op: str, value: Any) -> Any:
    """Canonicalize one exact, membership or order operand for a scalar type.

    ``None`` stays ``None`` for ``eq``, ``ne`` and each ``in`` candidate, so
    ``field == None``, ``{"field": None}`` and ``{"$in": [None]}`` match a
    missing value instead of failing scalar validation.

    Args:
        dtype: Logical type instance that validates canonical stored values.
        op: Portable filter operation identifier.
        value: User-authored query operand, or an iterable of operands for
            ``in``.

    Returns:
        Any: Canonical scalar, ``None`` for a missing-value test, a list for
        ``in``, or the unchanged operand for other operations.

    Raises:
        ValueError: If an order filter is given ``None``, or scalar
            logical-type validation fails.
        TypeError: If an ``in`` operand is a string or not iterable, or scalar
            logical-type validation rejects the operand type.
    """
    if op == "in":
        require_in_collection(value)
        return [None if item is None else dtype.validate(item) for item in value]
    if op in ("eq", "ne"):
        return None if value is None else dtype.validate(value)
    if op in _ORDER_OPS:
        require_order_bound(dtype, op, value)
        return dtype.validate(value)
    return value


def normalize_text_filter_value(dtype: Any, op: str, value: Any) -> Any:
    """Canonicalize one query operand for a text-valued logical type.

    ``None`` stays ``None`` for ``eq``, ``ne`` and each ``in`` candidate, so
    the filter tests for a missing value. An order filter given ``None``
    raises instead of comparing with the empty string.

    Args:
        dtype: Logical type instance that validates canonical stored values.
        op: Portable filter operation identifier.
        value: User-authored query operand.

    Returns:
        Any: Canonical scalar, list, or pattern string consumed identically by
        native and Python-scan compilers.

    Raises:
        TypeError: If an `in` operand is not a supported collection.
        ValueError: If an order filter is given ``None``, or scalar
            logical-type validation fails.
    """
    if op == "in":
        if not isinstance(value, (list, tuple, set)):
            raise TypeError("in filter expects a list, tuple, or set")
        return [None if item is None else dtype.validate(item) for item in value]
    if op in ("eq", "ne"):
        return None if value is None else dtype.validate(value)
    if op in _ORDER_OPS:
        require_order_bound(dtype, op, value)
        return dtype.validate(value)
    if op in ("contains", "ends_with", "ilike", "like", "match", "regex", "starts_with", "wildcard"):
        return str(value)
    return value
