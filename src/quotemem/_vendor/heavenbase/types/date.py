"""Date logical type."""

from __future__ import annotations

__all__ = [
    "Date",
]

from datetime import date, datetime, timezone

from heavenbase.utils import Any
from ..modules import ModuleResolver
from .base import LogicalType
from .filtering import normalize_text_filter_value


class Date(LogicalType):
    """Calendar date logical type stored as canonical UTC+0 ISO text.

    Warnings:
        ``datetime`` inputs are normalized to UTC+0 before the date part is
        taken. Naive ``datetime`` values are assumed to already be UTC+0. Plain
        ``date`` and ``YYYY-MM-DD`` strings carry no timezone and are stored as
        provided after validation.

    Examples:
        ``hb.Date().validate(date(2026, 5, 14))`` returns ``"2026-05-14"``.
        ``hb.Date().validate(datetime(2026, 5, 14, 0, 30, tzinfo=timezone(timedelta(hours=2))))``
        stores the corresponding UTC+0 date.
    """

    identifier = "date"
    max_chars = 10

    @classmethod
    def from_dict(cls, spec: dict, *, resolver: ModuleResolver | None = None) -> "Date":
        """Build a Date logical type from a serialized descriptor.

        Args:
            spec (dict): Serialized descriptor. Date accepts only the type
                selector and ignores extra keys.
            resolver (ModuleResolver | None): Context-owned module resolver.
                The default Context is selected when omitted.

        Returns:
            Date: Configured date logical type.
        """
        return cls()

    def validate(self, value: Any) -> str:
        """Normalize a date-like value to canonical ``YYYY-MM-DD`` text.

        Args:
            value: ISO date string, ``datetime.date``, or ``datetime.datetime``.

        Returns:
            Canonical ISO date string.

        Raises:
            ValueError: If ``value`` is numeric, ``bool``, an invalid date
                string, or an unsupported type.
        """
        if isinstance(value, bool) or isinstance(value, (int, float)):
            raise ValueError("Date does not accept epoch numbers")
        if isinstance(value, datetime):
            return self._from_datetime(value)
        if isinstance(value, date):
            return value.isoformat()
        if isinstance(value, str):
            try:
                return date.fromisoformat(value.strip()).isoformat()
            except ValueError as exc:
                raise ValueError(f"Invalid Date string {value!r}") from exc
        raise ValueError(f"Date expects ISO date string, date, or datetime, got {type(value).__name__}")

    def filter_value(self, op: str, value: Any) -> Any:
        """Normalize date query values for exact and order filters.

        Args:
            op: Filter operation name.
            value: Scalar date-like value, or an iterable of values for ``in``.

        Returns:
            Canonical date string, normalized list for ``in``, or the original
            text pattern for text filters.
        """
        return normalize_text_filter_value(self, op, value)

    def clone(self) -> "Date":
        """Return a new Date instance.

        Args:
            None.

        Returns:
            Date: Independent ``Date`` instance.
        """
        return Date()

    def _from_datetime(self, value: datetime) -> str:
        dt = value.replace(tzinfo=timezone.utc) if value.tzinfo is None or value.utcoffset() is None else value.astimezone(timezone.utc)
        return dt.date().isoformat()
