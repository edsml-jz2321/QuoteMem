"""Exact manual console-completion time rules.

These pure helpers neither sample a clock nor infer completion or retained data.
The command owner supplies every trusted time; historical reads need no new one.
"""

from __future__ import annotations

import re
from datetime import date

from pydantic import TypeAdapter, ValidationError

from quotemem.core.errors import QuoteMemError
from quotemem.core.technical_types import Timestamp

_TIMESTAMP = TypeAdapter[int](Timestamp)
_DAY_MS = 86_400_000
_RETENTION_MS = 30 * _DAY_MS
_EPOCH_ORDINAL = date(1970, 1, 1).toordinal()
_MANUAL_TIME = re.compile(
    r"([0-9]{4})-([0-9]{2})-([0-9]{2})T"
    r"([0-9]{2}):([0-9]{2}):([0-9]{2})"
    r"(?:\.([0-9]{1,3}))?(Z|[+-][0-9]{2}:[0-9]{2})"
)


def _timestamp(value: int) -> int:
    try:
        return _TIMESTAMP.validate_python(value, strict=True)
    except ValidationError:
        raise QuoteMemError("invalid_completion_timestamp",
                            "A valid system time is required") from None


def completion_deadline(completed_at: int) -> int:
    """Return the exact 30-day deadline within the shared system-time range.

    Args:
        completed_at: Actual console completion as strict UTC integer milliseconds.

    Raises:
        QuoteMemError: The completion is invalid or its deadline exceeds SystemTimestamp.
    """
    deadline = _timestamp(completed_at) + _RETENTION_MS
    try:
        return _TIMESTAMP.validate_python(deadline, strict=True)
    except ValidationError:
        raise QuoteMemError(
            "completion_deadline_overflow", "Completion deadline is out of range",
        ) from None


def require_forward_clock(now: int, *previous: int) -> None:
    """Reject a new trusted sample earlier than any applicable prior fact.

    Args:
        now: New trusted system timestamp; equality is permitted.
        *previous: Persisted confirmation, observation, applicable receipt, or the
            same operation's prior trusted times. The caller selects these facts.

    Raises:
        QuoteMemError: A time is invalid or the new sample would roll the clock back.

    This stores no watermark. In particular, an old unpersisted status sample
    cannot become a recoverable cross-process clock authority through this API.
    """
    current = _timestamp(now)
    prior = tuple(_timestamp(value) for value in previous)
    if any(current < value for value in prior):
        raise QuoteMemError("clock_rollback", "The trusted clock precedes an applicable prior time")


def parse_completed_at(value: str, *, recorded_at: int) -> int:
    """Parse one new manual completion input without floating-point conversion.

    Accept exactly ``YYYY-MM-DDTHH:MM:SS[.sss](Z|±HH:MM)``: the optional fraction
    has one to three digits. Numeric offsets have hours 00–23 and minutes 00–59;
    both signed zero offsets denote UTC. Seconds 60 and alternate ISO spellings
    are outside this grammar. The returned UTC instant and its deadline must
    both fit SystemTimestamp, even when the local date itself is representable.

    Args:
        value: The operator's explicit console timestamp, preserved until parsing.
        recorded_at: Trusted current time for this new confirmation, not its source.

    Returns:
        The exact completion instant in UTC integer milliseconds.

    Raises:
        QuoteMemError: Syntax, calendar, offset, range, future time, or deadline is invalid.

    Historical records are validated by their strict models without calling this
    new-input parser or sampling a current clock. Errors contain no input text.
    """
    current = _timestamp(recorded_at)
    match = _MANUAL_TIME.fullmatch(value) if type(value) is str else None
    if match is None:
        raise QuoteMemError(
            "invalid_completion_timestamp", "An exact timezone-explicit time is required",
        )
    year, month, day, hour, minute, second, fraction, zone = match.groups()
    try:
        calendar = date(int(year), int(month), int(day))
    except ValueError:
        raise QuoteMemError("invalid_completion_timestamp",
                            "The calendar time is invalid") from None
    hours, minutes, seconds = int(hour), int(minute), int(second)
    if hours > 23 or minutes > 59 or seconds > 59:
        raise QuoteMemError("invalid_completion_timestamp", "The clock time is invalid")
    offset_ms = 0
    if zone != "Z":
        offset_hours, offset_minutes = int(zone[1:3]), int(zone[4:6])
        if offset_hours > 23 or offset_minutes > 59:
            raise QuoteMemError("invalid_completion_timestamp", "The UTC offset is invalid")
        offset_ms = (offset_hours * 60 + offset_minutes) * 60_000
        if zone[0] == "-":
            offset_ms = -offset_ms
    milliseconds = int(fraction.ljust(3, "0")) if fraction else 0
    completed_at = _timestamp(
        (calendar.toordinal() - _EPOCH_ORDINAL) * _DAY_MS
        + ((hours * 60 + minutes) * 60 + seconds) * 1000
        + milliseconds
        - offset_ms
    )
    completion_deadline(completed_at)
    if completed_at > current:
        raise QuoteMemError("future_completion",
                            "Completion cannot follow its trusted recording time")
    return completed_at
