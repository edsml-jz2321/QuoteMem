"""Descriptor-owned bounded-text semantics."""

from __future__ import annotations

__all__: list[str] = []

from heavenbase.utils import Any, Mapping

from ..modules import ModuleResolver
from .base import LogicalType
from .filtering import normalize_text_filter_value


class _BoundedText(LogicalType):
    """Own one immutable, serializable text-length constraint."""

    default_max_chars = 0

    def __init__(self, max_chars: int | None = None) -> None:
        value = self.default_max_chars if max_chars is None else max_chars
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{type(self).__name__} max_chars must be an integer")
        if value <= 0:
            raise ValueError(f"{type(self).__name__} max_chars must be positive")
        self._max_chars = value

    @classmethod
    def from_dict(
        cls,
        spec: Mapping[str, Any],
        *,
        resolver: ModuleResolver | None = None,
    ) -> "_BoundedText":
        """Build a bounded text type from a durable descriptor.

        Args:
            spec (Mapping[str, Any]): Descriptor containing a type selector and
                optional positive `max_chars` value.
            resolver (ModuleResolver | None): Context-owned module resolver.
                The default Context is selected when omitted.

        Returns:
            _BoundedText: Configured bounded text value.

        Raises:
            TypeError: If `spec` or `max_chars` has an invalid type.
            ValueError: If the selector, descriptor keys, or limit is invalid.
        """
        cls._check_descriptor(spec, resolver=resolver)
        return cls(max_chars=spec.get("max_chars"))

    @classmethod
    def _check_descriptor(
        cls,
        spec: Mapping[str, Any],
        *,
        resolver: ModuleResolver | None,
        extra: tuple[str, ...] = (),
    ) -> None:
        """Validate selector and owned keys for one bounded-text descriptor."""
        if not isinstance(spec, Mapping):
            raise TypeError(f"{cls.__name__}.from_dict expects a mapping")
        key = spec.get("type") or spec.get("identifier") or spec.get("dtype")
        if key is not None:
            resolved = cls.load(key, resolver=resolver)
            if resolved.identifier != cls.identifier:
                raise ValueError(f"{cls.__name__}.from_dict expected {cls.identifier!r}, got {resolved.identifier!r}")
        allowed = {"type", "identifier", "dtype", "max_chars", *extra}
        extras = sorted(set(spec) - allowed)
        if extras:
            raise ValueError(f"Unknown {cls.identifier} descriptor key {extras[0]!r}")

    @property
    def max_chars(self) -> int:
        """Return this descriptor's maximum character count."""
        return self._max_chars

    def validate(self, value: Any) -> str:
        """Normalize text and enforce this descriptor's length.

        Args:
            value (Any): Text input or `None`.

        Returns:
            str: Normalized text within `max_chars`.

        Raises:
            ValueError: If normalized text exceeds `max_chars`.
        """
        text = "" if value is None else str(value)
        if len(text) > self.max_chars:
            raise ValueError(f"{type(self).__name__} exceeds {self.max_chars} characters")
        return text

    def filter_value(self, op: str, value: Any) -> Any:
        """Normalize an exact, ordered, membership, or pattern operand.

        Args:
            op (str): Portable filter operation identifier.
            value (Any): User-authored query operand.

        Returns:
            Any: Canonical text, null, membership list, or pattern string.
        """
        return normalize_text_filter_value(self, op, value)

    def to_dict(self) -> dict[str, Any]:
        """Return the complete durable bounded-text descriptor.

        Args:
            None.

        Returns:
            dict[str, Any]: Type identifier and concrete `max_chars` value.
        """
        return {"type": type(self).identifier, "max_chars": self.max_chars}

    def clone(self) -> "_BoundedText":
        """Return an independent value with the same text constraint.

        Args:
            None.

        Returns:
            _BoundedText: Independent descriptor value.
        """
        return type(self)(max_chars=self.max_chars)
