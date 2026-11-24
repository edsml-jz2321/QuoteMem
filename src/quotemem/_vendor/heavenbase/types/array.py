"""Array logical type."""

from __future__ import annotations

__all__ = [
    "Array",
]

from heavenbase.utils import Any, Optional

from ..modules import ModuleResolver
from .base import LogicalType
from .descriptor import coerce_type, logical_type_from_any


class Array(LogicalType):
    """List-valued logical type with an optional item type.

    Canonical form is a Python ``list``. ``None`` normalises to ``[]``.
    When ``item_type`` is set, each element is validated through that type.

    Warnings:
        Bare strings are rejected; wrap single values in a one-element list
        when storing scalar-like payloads in an untyped ``Array``.

    Examples:
        ``hb.Array().validate([1, 2])`` returns ``[1, 2]``.
        ``hb.Array[hb.Integer].validate(["1", 2])`` returns ``[1, 2]``.
        ``hb.Array(hb.Integer).validate(["1", 2])`` returns ``[1, 2]``.
        ``hb.Array().validate(None)`` returns ``[]``.
    """

    identifier = "array"

    @classmethod
    def from_dict(cls, spec: dict, *, resolver: ModuleResolver | None = None) -> "Array":
        """Build an Array logical type from a serialized descriptor.

        Args:
            spec (dict): Serialized descriptor with optional ``item_type``,
                ``items``, or ``value`` keys.
            resolver (ModuleResolver | None): Context-owned module resolver.
                The default Context is selected when omitted.

        Returns:
            Array: Configured array logical type.
        """
        item = spec.get("item_type", spec.get("items", spec.get("value")))
        return cls(logical_type_from_any(item, resolver=resolver) if item is not None else None)

    def __init__(self, item_type: Optional[LogicalType] = None):
        """Create an Array logical type with optional item validation.

        Args:
            item_type (LogicalType | None): Logical type applied to each list
                element. When omitted, elements are stored without coercion.

        Returns:
            None: This constructor does not return a value.
        """
        self.item_type = coerce_type(item_type) if item_type is not None else None

    def __class_getitem__(cls, item: Any) -> "Array":
        return cls(coerce_type(item))

    def validate(self, value: Any) -> list:
        """Validate a list and each typed item when configured.

        Args:
            value: List, tuple, array-like object with ``tolist()``, or
                ``None``.

        Returns:
            Normalised ``list`` with optional per-item validation.

        Raises:
            ValueError: If ``value`` is not list-like or an item fails
                validation.
        """
        if value is None:
            return []
        if hasattr(value, "tolist"):
            value = value.tolist()
        if isinstance(value, tuple):
            value = list(value)
        if not isinstance(value, list):
            if isinstance(value, str):
                raise ValueError("Array expects a list")
            try:
                value = list(value)
            except TypeError:
                raise ValueError("Array expects a list")
        return [self.item_type.validate(v) if self.item_type is not None else v for v in value]

    def filter_value(self, op: str, value: Any) -> Any:
        """Normalize exact, membership, and item-containment query operands.

        Args:
            op: Portable filter operation identifier.
            value: User-authored array or item operand.

        Returns:
            Any: Canonical list, list of candidates, contained item, or the
            unchanged operand for operations without array normalization.

        Raises:
            TypeError: If an `in` operand is not a supported collection.
            ValueError: If array or item logical-type validation fails.
        """
        if op in ("eq", "ne"):
            return None if value is None else self.validate(value)
        if op == "in":
            if not isinstance(value, (list, tuple, set)):
                raise TypeError("in filter expects a list, tuple, or set")
            return [None if candidate is None else self.validate(candidate) for candidate in value]
        if op == "array_contains" and self.item_type is not None:
            return self.item_type.validate(value)
        return value

    def to_dict(self) -> dict:
        """Return a JSON-compatible Array descriptor.

        Args:
            None.

        Returns:
            dict: Descriptor containing ``type`` and optional ``item_type``.
        """
        return {"type": type(self).identifier, "item_type": self.item_type.to_dict() if self.item_type else None}

    def clone(self) -> "Array":
        """Return a new Array with a cloned item type.

        Args:
            None.

        Returns:
            Array: Independent array logical type with a cloned item type.
        """
        return Array(self.item_type.clone() if self.item_type else None)
