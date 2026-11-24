"""Categorical logical type."""

from __future__ import annotations

__all__ = [
    "Categorical",
]

from heavenbase.utils import Any, Sequence

from ..modules import ModuleResolver
from .base import LogicalType
from .filtering import normalize_scalar_filter_value, normalize_text_filter_value


class Categorical(LogicalType):
    """Constrained enum-like logical type backed by strings or integers.

    Args:
        values: Non-empty sequence of unique allowed values. String categories
            require strings; integer categories require integers.
        base: Physical scalar base, either ``"string"`` or ``"integer"``.

    Warnings:
        String categories support exact and text filters. Integer categories
        support exact and order filters. Cross-base filter domains raise
        ``ValueError`` instead of silently casting to surprising semantics.

    Examples:
        ``hb.Categorical(["todo", "doing", "done"])``
        ``hb.Categorical([1, 2, 3], base="integer")``
    """

    identifier = "categorical"
    _TEXT_OPS = ("match", "like", "ilike", "contains", "starts_with", "ends_with", "wildcard", "regex")
    _ORDER_OPS = ("lt", "lte", "gt", "gte")

    @classmethod
    def from_dict(cls, spec: dict, *, resolver: ModuleResolver | None = None) -> "Categorical":
        """Build a Categorical logical type from a serialized descriptor.

        Args:
            spec (dict): Serialized descriptor with ``values`` and optional
                ``base`` keys.
            resolver (ModuleResolver | None): Context-owned module resolver.
                The default Context is selected when omitted.

        Returns:
            Categorical: Configured categorical logical type.
        """
        return cls(spec.get("values", ()), spec.get("base", "string"))

    def __class_getitem__(cls, item: Any) -> "Categorical":
        items = item if isinstance(item, tuple) else (item,)
        base = "integer" if items and all(isinstance(value, int) and not isinstance(value, bool) for value in items) else "string"
        return cls(items, base=base)

    def __init__(self, values: Sequence[str] | Sequence[int], base: str = "string"):
        """Initialize categorical membership and scalar base."""
        self.base = str(base).strip().lower()
        if self.base not in ("string", "integer"):
            raise ValueError("Categorical base must be 'string' or 'integer'")
        if isinstance(values, (str, bytes)):
            raise TypeError("Categorical values must be a sequence, not a string")
        self.values = tuple(self._choice(value) for value in values)
        if not self.values:
            raise ValueError("Categorical values cannot be empty")
        if len(set(self.values)) != len(self.values):
            raise ValueError("Categorical values must be unique")

    @property
    def max_chars(self) -> int:
        """Return the longest category length."""
        return max(len(value) for value in self.values) if self.base == "string" else max(len(str(value)) for value in self.values)

    def validate(self, value: Any) -> Any:
        """Coerce a value to the configured base type and enforce membership.

        Args:
            value: Candidate category value. ``None`` is preserved for optional
                fields.

        Returns:
            The normalized string or integer category value.

        Raises:
            ValueError: If ``value`` is not a member, is ``bool`` for integer
                categories, or would require lossy integer coercion.
        """
        if value is None:
            return None
        item = self._value(value)
        if item not in self.values:
            raise ValueError(f"Categorical value {item!r} is not one of {self.values!r}")
        return item

    def filter_value(self, op: str, value: Any) -> Any:
        """Normalize a categorical query value for one filter operation.

        Args:
            op: Filter operation name such as ``eq``, ``in``, ``lt``, or
                ``contains``.
            value: User-provided filter value.

        ``None`` stays ``None`` for ``eq``, ``ne`` and ``in`` so the filter
        tests for a missing value.

        Returns:
            A normalized scalar value, ``None`` for a missing-value test,
            normalized list for ``in``, or the original text pattern for text
            filters.

        Raises:
            ValueError: If the operation does not match the categorical base,
                if exact/order values are outside the allowed set, or if an
                order filter is given ``None``.
        """
        self._check_op(op)
        if self.base == "string":
            return normalize_text_filter_value(self, op, value)
        return normalize_scalar_filter_value(self, op, value)

    def to_dict(self) -> dict:
        """Return a JSON-compatible Categorical descriptor.

        Args:
            None.

        Returns:
            dict: Mapping with ``type``, ``values``, and ``base`` keys.
        """
        return {"type": type(self).identifier, "values": list(self.values), "base": self.base}

    def clone(self) -> "Categorical":
        """Return a new Categorical with the same values and base.

        Args:
            None.

        Returns:
            Categorical: Independent ``Categorical`` instance with copied configuration.
        """
        return Categorical(self.values, self.base)

    def _choice(self, value: Any) -> str | int:
        if self.base == "integer":
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError("Integer Categorical values must be ints")
            return int(value)
        if not isinstance(value, str):
            raise TypeError("String Categorical values must be strings")
        return value

    def _value(self, value: Any) -> str | int:
        if self.base == "integer":
            if isinstance(value, bool):
                raise ValueError("Integer Categorical value cannot be bool")
            try:
                item = int(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"Integer Categorical value {value!r} is not an int") from exc
            if not isinstance(value, str) and item != value:
                raise ValueError(f"Integer Categorical value {value!r} is not an int")
            return item
        return str(value)

    def _check_op(self, op: str) -> None:
        if self.base == "integer" and op in self._TEXT_OPS:
            raise ValueError("Integer Categorical fields do not support text filter operations")
        if self.base == "string" and op in self._ORDER_OPS:
            raise ValueError("String Categorical fields do not support order filter operations")
