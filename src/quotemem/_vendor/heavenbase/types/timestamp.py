"""Timestamp logical type."""

from __future__ import annotations

__all__ = [
    "Timestamp",
]

from datetime import date, datetime, time, timezone

from heavenbase.utils import Any, Optional

from ..modules import ModuleResolver
from .base import LogicalType
from .filtering import require_in_collection

TIMESTAMP_DEFAULT_MAX_MS = 253_402_300_799_000
TIMESTAMP_DEFAULT_MIN_MS = 0


def _scale_ms(value_ms: int, unit: str) -> int:
    if unit == "ms":
        return value_ms
    if unit == "s":
        return value_ms // 1000
    if unit == "us":
        return value_ms * 1000
    raise ValueError("Timestamp unit must be 's', 'ms', or 'us'")


class Timestamp(LogicalType):
    """Point-in-time logical type stored as UTC+0 epoch integers.

    Class attributes :attr:`default_max` and :attr:`default_min` expose stable
    protocol bounds in milliseconds. Catalog and other policy fields use
    ``default_max`` as the open-deadline sentinel for disabled absolute expiry.

    Args:
        unit: Epoch precision. Supported values are ``"s"``, ``"ms"``, and
            ``"us"``. Milliseconds are the default.
        max: Optional upper bound in ``unit``. When omitted, uses the stable
            protocol bound scaled from milliseconds.
        min: Optional lower bound in ``unit``. When omitted, uses the stable
            protocol bound scaled from milliseconds.

    Warnings:
        Stored values are timezone-free integers and must be interpreted as
        UTC+0. Values are clipped to ``[min, max]`` during validation and
        query filter normalization.

    Examples:
        ``hb.Timestamp().validate(datetime.now(timezone.utc))`` returns epoch
        milliseconds. ``hb.Timestamp(unit="s")`` stores epoch seconds.
    """

    identifier = "timestamp"
    default_max = TIMESTAMP_DEFAULT_MAX_MS
    default_min = TIMESTAMP_DEFAULT_MIN_MS
    _UNITS = {"s": 1, "ms": 1000, "us": 1000000}

    @classmethod
    def from_dict(cls, spec: dict, *, resolver: ModuleResolver | None = None) -> "Timestamp":
        """Build a Timestamp logical type from a serialized descriptor.

        Args:
            spec (dict): Serialized descriptor with optional ``unit``, ``max``,
                and ``min`` keys.
            resolver (ModuleResolver | None): Context-owned module resolver.
                The default Context is selected when omitted.

        Returns:
            Timestamp: Configured timestamp logical type.
        """
        return cls(spec.get("unit", "ms"), max=spec.get("max"), min=spec.get("min"))

    def __class_getitem__(cls, unit: str) -> "Timestamp":
        return cls(unit)

    def __init__(self, unit: str = "ms", *, max: Optional[int] = None, min: Optional[int] = None):
        """Initialize epoch unit and optional clip bounds."""
        self._unit = str(unit).strip().lower()
        if self._unit not in self._UNITS:
            raise ValueError("Timestamp unit must be 's', 'ms', or 'us'")
        self._max = int(max) if max is not None else None
        self._min = int(min) if min is not None else None
        if self._max is not None and self._min is not None and self._min > self._max:
            raise ValueError(f"Timestamp min ({self._min}) cannot exceed max ({self._max})")

    @property
    def unit(self) -> str:
        """Return the immutable epoch precision owned by this descriptor."""
        return self._unit

    @property
    def max_bound(self) -> int:
        """Return the effective upper bound for this type in ``unit``."""
        return self._max if self._max is not None else _scale_ms(self.default_max, self.unit)

    @property
    def min_bound(self) -> int:
        """Return the effective lower bound for this type in ``unit``."""
        return self._min if self._min is not None else _scale_ms(self.default_min, self.unit)

    def validate(self, value: Any) -> int:
        """Normalize a timestamp-like value to an epoch integer.

        Args:
            value: Epoch integer, epoch float, ``datetime``, ``datetime.date``,
                or ISO 8601 timestamp string. A trailing ``Z`` is accepted as
                UTC+0. Plain ``date`` values and offsetless ISO strings are
                interpreted as UTC+0 midnight. ``None`` is rejected: a missing
                timestamp never silently becomes the current time, so callers
                that mean "now" pass ``datetime.now(timezone.utc)`` explicitly
                and optional fields declare ``.optional()``.

        Returns:
            Epoch integer in this type's configured unit, clipped to
            ``[min_bound, max_bound]``.

        Raises:
            ValueError: If ``value`` is ``None``, ``bool``, has an unsupported
                type, or is not a valid ISO timestamp string.
        """
        if value is None:
            raise ValueError("Timestamp requires a value; pass datetime.now(timezone.utc) for the current time or declare the field optional")
        if isinstance(value, bool):
            raise ValueError("Timestamp cannot be bool")
        if isinstance(value, int):
            return self._clip(value)
        if isinstance(value, float):
            return self._clip(int(value))
        if isinstance(value, datetime):
            return self._clip(self._from_datetime(value))
        if isinstance(value, date):
            return self._clip(self._from_date(value))
        if isinstance(value, str):
            text = value.strip()
            try:
                return self._clip(self._from_datetime(datetime.fromisoformat(text[:-1] + "+00:00" if text.endswith("Z") else text)))
            except ValueError as exc:
                raise ValueError(f"Invalid Timestamp string {value!r}") from exc
        raise ValueError(f"Timestamp expects int, float, datetime, date, or ISO string, got {type(value).__name__}")

    def filter_value(self, op: str, value: Any) -> Any:
        """Normalize temporal query values for exact and order filters.

        ``None`` stays ``None`` for ``eq``, ``ne`` and ``in``, as it does for
        text fields, so ``field == None``, ``{"field": None}`` and
        ``{"$in": [None]}`` match a missing timestamp instead of failing
        timestamp validation. An order filter has no bound to compare with
        ``None`` and raises.

        Args:
            op: Filter operation name.
            value: Scalar timestamp-like value, or an iterable of values for
                ``in``.

        Returns:
            Normalized epoch integer, normalized list for ``in``, ``None`` for
            an equality test of a missing value, or the original value for
            unsupported operations.

        Raises:
            ValueError: If an order filter (``lt``, ``lte``, ``gt``, ``gte``)
                is given ``None``, or a value is not a valid timestamp.
            TypeError: If an ``in`` operand is a string or not iterable.
        """
        if op == "in":
            require_in_collection(value)
            return [None if item is None else self.validate(item) for item in value]
        if op in ("eq", "ne"):
            return None if value is None else self.validate(value)
        if op in ("lt", "lte", "gt", "gte"):
            if value is None:
                raise ValueError(f"Timestamp {op!r} filter needs a value; None only matches a missing timestamp through eq, ne or in")
            return self.validate(value)
        return value

    def to_dict(self) -> dict:
        """Return a JSON-compatible Timestamp descriptor.

        Args:
            None.

        Returns:
            dict: Mapping with ``type``, ``unit``, and optional bound keys.
        """
        payload: dict[str, Any] = {"type": type(self).identifier, "unit": self.unit}
        if self._max is not None:
            payload["max"] = self._max
        if self._min is not None:
            payload["min"] = self._min
        return payload

    def clone(self) -> "Timestamp":
        """Return a new Timestamp with the same unit and bounds.

        Args:
            None.

        Returns:
            Timestamp: Independent ``Timestamp`` instance with copied configuration.
        """
        return Timestamp(self.unit, max=self._max, min=self._min)

    def _clip(self, value: int) -> int:
        return max(self.min_bound, min(self.max_bound, value))

    def _from_date(self, value: date) -> int:
        return self._from_datetime(datetime.combine(value, time.min))

    def _from_datetime(self, value: datetime) -> int:
        dt = value.replace(tzinfo=timezone.utc) if value.tzinfo is None or value.utcoffset() is None else value.astimezone(timezone.utc)
        return int(dt.timestamp() * self._UNITS[self.unit])
