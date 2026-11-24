"""Neutral typed facade for the structured scalar-index Tag value."""

from __future__ import annotations

__all__ = ["ScalarIndexSpec"]

from collections.abc import Mapping, Sequence
from typing import Any

from heavenbase.utils import coerce_bool, coerce_tuple, dataclass


@dataclass(frozen=True)
class ScalarIndexSpec:
    """Expose one validated ``scalar-index`` Tag value as an immutable facade.

    Args:
        value: Whether explicit scalar lookup indexes are supported.
        unique: Whether single-field uniqueness constraints are supported.
        composite: Whether multi-field scalar lookup indexes are supported.
        composite_unique: Whether multi-field uniqueness constraints are supported.
        opt_out: Whether provider-default single-field indexes can be disabled.
        supported_types: Logical type identifiers accepted by explicit declarations.
        default_types: Logical type identifiers indexed without user declarations.

    Raises:
        TypeError: If Boolean flags or logical-type sequences use the wrong type.
        ValueError: If a default logical type is not supported.
    """

    value: bool = False
    unique: bool = False
    composite: bool = False
    composite_unique: bool = False
    opt_out: bool = False
    supported_types: Sequence[str] = ()
    default_types: Sequence[str] = ()

    def __post_init__(self) -> None:
        """Normalize and validate immutable scalar-index values."""
        for name in ("value", "unique", "composite", "composite_unique", "opt_out"):
            coerce_bool(getattr(self, name), f"ScalarIndexSpec {name}")
        for name in ("supported_types", "default_types"):
            value = coerce_tuple(getattr(self, name), f"ScalarIndexSpec {name}")
            object.__setattr__(self, name, tuple(str(item) for item in value))
        unsupported_defaults = sorted(set(self.default_types) - set(self.supported_types))
        if unsupported_defaults:
            raise ValueError(f"Scalar index default_types must be supported: {unsupported_defaults}")

    @classmethod
    def load(cls, value: Any) -> "ScalarIndexSpec":
        """Load a scalar-index Tag value from a facade or mapping.

        Args:
            value: Existing facade or canonical Tag object value.

        Returns:
            ScalarIndexSpec: Normalized immutable scalar-index metadata.
        """
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            return cls(**dict(value))
        raise TypeError("Backend scalar-index tag must be ScalarIndexSpec or a mapping")

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical JSON-compatible Tag object value.

        Args:
            None.

        Returns:
            dict[str, Any]: Scalar-index capability flags and supported logical types.
        """
        return {
            "value": self.value,
            "unique": self.unique,
            "composite": self.composite,
            "composite_unique": self.composite_unique,
            "opt_out": self.opt_out,
            "supported_types": list(self.supported_types),
            "default_types": list(self.default_types),
        }
