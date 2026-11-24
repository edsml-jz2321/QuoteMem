"""Deterministic source-time interpretation (validity and time rules).

Calendar expressions without an explicit zone use UTC. Calendar windows describe
the source's precision, not an invented state expiry. Unsupported or ambiguous
expressions remain unanchored; neither ingestion time nor the system clock is an
input to this module. An expression in another language gets the window its
English equivalent gets when it is one of the closed phrases or forms of
`temporal_lexicon`; a year of 2400 or more is never read as Gregorian.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Final, Literal

from quotemem.core.extraction_output import MemoryKind
from quotemem.core.semantic import (
    CURRENT_EXTRACTION_PIPELINES,
    AnchorV1,
    TimeExpressionV1,
    TimeGranularity,
)
from quotemem.ingestion.temporal_lexicon import (
    AGO,
    BUDDHIST_ERA_OFFSET,
    CALENDAR_AGO,
    CJK_NUMERALS,
    COUNT_WORDS,
    DATE_FORMS,
    DAYS,
    ENGLISH_WORD_MONTHS,
    GREGORIAN_LIMIT,
    HOUR_UNITS,
    MONTH_YEAR_FORMS,
    MONTHS,
    NAMED_MONTH_FORMS,
    PERIODS,
    SINCE_YEAR_FORMS,
    THAI_MONTHS,
    THAI_YEAR_FORMS,
    WEEK_UNITS,
    WEEKDAYS,
    WEEKEND_UNITS,
    WEEKENDS,
    YEAR_FORMS,
    YEAR_MONTH_FORMS,
    YEAR_UNITS,
    buddhist_era,
    fold_time,
)

DAY_MS = 86_400_000
HOUR_MS = 3_600_000
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
_MONTHS = {
    name: index
    for index, names in enumerate((
        ("january", "jan"), ("february", "feb"), ("march", "mar"), ("april", "apr"),
        ("may",), ("june", "jun"), ("july", "jul"), ("august", "aug"), ("september", "sep"),
        ("october", "oct"), ("november", "nov"), ("december", "dec"),
    ), 1)
    for name in names
}
"""English month names and abbreviations, independent of the process locale."""


@dataclass(frozen=True)
class TemporalResolution:
    """Source precision and provenance, separate from a memory's validity interval."""

    start: int | None
    end: int | None
    granularity: TimeGranularity
    anchored_from: AnchorV1
    reason: Literal["resolved", "not_applicable", "missing_source_time", "unsupported_expression"]

    def validity(self, kind: MemoryKind) -> tuple[int | None, int | None]:
        """Use an event's source window; a state never acquires an invented end."""
        return self.start, self.end if kind == "event" else None


def _milliseconds(value: datetime) -> int:
    difference = value.astimezone(UTC) - _EPOCH
    return difference.days * DAY_MS + difference.seconds * 1000 + difference.microseconds // 1000


def _window(
    year: int, month: int | None = None, day: int | None = None,
) -> tuple[int, int, TimeGranularity]:
    if month == 0 or day == 0:
        raise ValueError("Explicit calendar components must be positive")
    if year >= GREGORIAN_LIMIT:
        raise ValueError("A year of 2400 or more is not read as Gregorian")
    start = datetime(year, month or 1, day or 1, tzinfo=UTC)
    if day is not None:
        end, granularity = start + timedelta(days=1), "DAY"
    elif month is not None:
        end = datetime(year + (month == 12), month % 12 + 1, 1, tzinfo=UTC)
        granularity = "MONTH"
    else:
        end, granularity = datetime(year + 1, 1, 1, tzinfo=UTC), "YEAR"
    return _milliseconds(start), _milliseconds(end), granularity


def _absolute(
    text: str,
) -> tuple[int, int | None, TimeGranularity] | None:
    # ISO calendar forms do not borrow a missing year from any unrelated clock.
    match = re.fullmatch(r"(\d{4})(?:-(\d{2})(?:-(\d{2}))?)?", text)
    if match:
        year, month, day = match.groups()
        return _window(
            int(year), int(month) if month else None, int(day) if day else None,
        )
    match = re.fullmatch("(\\d{4})\u5e74(?:(\\d{1,2})\u6708(?:(\\d{1,2})[\u65e5\u53f7])?)?", text)
    if match:
        year, month, day = match.groups()
        return _window(
            int(year), int(month) if month else None, int(day) if day else None,
        )
    match = re.fullmatch(r"([A-Za-z]+) (?:(\d{1,2}),? )?(\d{4})", text)
    if match and match[1].casefold() in _MONTHS:
        return _window(
            int(match[3]), _MONTHS[match[1].casefold()], int(match[2]) if match[2] else None,
        )
    match = re.fullmatch(r"(\d{1,2}) ([A-Za-z]+) (\d{4})", text)
    if match and match[2].casefold() in _MONTHS:
        return _window(
            int(match[3]), _MONTHS[match[2].casefold()], int(match[1]),
        )
    if re.fullmatch(
        r"\d{4}-\d{2}-\d{2}[Tt ]\d{2}:\d{2}(?::\d{2}(?:\.\d{1,3})?)?(?:[Zz]|[+-]\d{2}:\d{2})?", text
    ):
        value = datetime.fromisoformat(text.upper().replace("Z", "+00:00"))
        value = value.replace(tzinfo=UTC) if value.tzinfo is None else value
        if value.year >= GREGORIAN_LIMIT:
            return None
        # A timestamp identifies an instant, not a one-hour event duration.
        return _milliseconds(value), None, "HOUR"
    return _lexicon_absolute(fold_time(text))


def _month_of(match: re.Match[str]) -> int:
    """The month a lexicon match names: a month name, a month number or a CJK numeral."""
    groups = match.groupdict()
    if groups.get("month"):
        return MONTHS[groups["month"]]
    if groups.get("number"):
        return int(groups["number"])
    return CJK_NUMERALS[groups["cjk"]]


def _year_of(match: re.Match[str], *, thai: bool = False) -> int | None:
    """The Gregorian year a lexicon match names, converting a Thai Buddhist-era year.

    ``พ.ศ.`` marks a Buddhist-era year and ``ค.ศ.`` a Gregorian one; an unmarked year
    of a Thai year form or of a Thai month name is read by `buddhist_era`.
    """
    groups = match.groupdict()
    year = int(groups["year"])
    if groups.get("be"):
        converted = year - BUDDHIST_ERA_OFFSET
        return converted if converted >= 1800 else None
    if groups.get("ce") is None and (thai or groups.get("month") in THAI_MONTHS):
        return buddhist_era(year)
    return year


def _lexicon_absolute(folded: str) -> tuple[int, int | None, TimeGranularity] | None:
    """Anchor a day, a month or a year written in a `temporal_lexicon` form."""
    for patterns, since in ((DATE_FORMS, False), (MONTH_YEAR_FORMS, False),
                            (YEAR_FORMS, False), (SINCE_YEAR_FORMS, True)):
        for pattern in patterns:
            match = pattern.fullmatch(folded)
            if match is None:
                continue
            year = _year_of(match)
            if year is None:
                return None
            groups = match.groupdict()
            month = _month_of(match) if "month" in groups or "number" in groups else None
            day = int(groups["day"]) if groups.get("day") else None
            start, end, granularity = _window(year, month, day)
            return start, None if since else end, granularity
    for pattern in THAI_YEAR_FORMS:
        match = pattern.fullmatch(folded)
        if match is not None:
            year = _year_of(match, thai=True)
            return None if year is None else _window(year)
    return None


_OFFSETS: Final = {"last": -1, "this": 0, "next": 1}


def _day_window(day: int, length: int = 1) -> tuple[int, int, TimeGranularity]:
    """A UTC calendar-day range, rejecting dates outside datetime's calendar."""
    start = _EPOCH + timedelta(days=day)
    end = start + timedelta(days=length)
    return _milliseconds(start), _milliseconds(end), "DAY"


def _weekday_or_weekend(
    folded: str, source_time: int,
) -> tuple[int, int, TimeGranularity] | None:
    """Strict weekday occurrences or an explicitly named ISO-week calendar range."""
    weekday = WEEKDAYS.get(folded)
    weekend = WEEKENDS.get(folded)
    if weekday is None and weekend is None:
        return None
    observed = _EPOCH + timedelta(milliseconds=source_time)
    day = source_time // DAY_MS
    monday = day - observed.weekday()
    if weekend is not None:
        return _day_window(monday + _OFFSETS[weekend] * 7 + 5, 2)
    assert weekday is not None
    reading, target = weekday
    if reading == "previous":
        return _day_window(day - ((observed.weekday() - target) % 7 or 7))
    if reading == "following":
        return _day_window(day + ((target - observed.weekday()) % 7 or 7))
    week_offset = -1 if reading == "previous_week" else 1 if reading == "following_week" else 0
    return _day_window(monday + week_offset * 7 + target)


def _iso_week(source_time: int, offset: int) -> tuple[int, int, TimeGranularity]:
    """The Monday-to-Sunday ISO week ``offset`` weeks from the source's, at DAY precision."""
    observed = _EPOCH + timedelta(milliseconds=source_time)
    start = (source_time // DAY_MS - observed.weekday() + offset * 7) * DAY_MS
    return start, start + 7 * DAY_MS, "DAY"


def _calendar_count(folded: str, source_time: int) -> tuple[int, int, TimeGranularity] | None:
    """The calendar month or year a `CALENDAR_AGO` count names from the source time."""
    for pattern, sign in CALENDAR_AGO:
        if (match := pattern.fullmatch(folded)) is None:
            continue
        count = COUNT_WORDS.get(match[1])
        offset = (int(match[1]) if count is None else count) * sign
        observed = _EPOCH + timedelta(milliseconds=source_time)
        if match["unit"] in YEAR_UNITS:
            return _window(observed.year + offset)
        year, month_index = divmod(observed.year * 12 + observed.month - 1 + offset, 12)
        return _window(year, month_index + 1)
    return None


def _relative(text: str, source_time: int) -> tuple[int, int, TimeGranularity] | None:
    """Anchor a relative expression at the source time under ``calendar-unit-v1``.

    A week count is the whole offset ISO week, as "last week" already is, and "N months
    ago" and "N years ago" (`CALENDAR_AGO`) are the calendar month or year they name.
    """
    folded = fold_time(text)
    if (offset := DAYS.get(folded)) is not None:
        start = (source_time // DAY_MS + offset) * DAY_MS
        return start, start + DAY_MS, "DAY"
    for pattern, sign in AGO:
        if (match := pattern.fullmatch(folded)) is not None:
            count = COUNT_WORDS.get(match[1])
            count = int(match[1]) if count is None else count
            if match["unit"] in WEEKEND_UNITS:
                observed = _EPOCH + timedelta(milliseconds=source_time)
                monday = source_time // DAY_MS - observed.weekday()
                return _day_window(monday + count * sign * 7 + 5, 2)
            if match["unit"] in WEEK_UNITS:
                return _iso_week(source_time, count * sign)
            unit = HOUR_MS if match["unit"] in HOUR_UNITS else DAY_MS
            start = (source_time // unit + count * sign) * unit
            return start, start + unit, "HOUR" if unit == HOUR_MS else "DAY"
    if (weekday := _weekday_or_weekend(folded, source_time)) is not None:
        return weekday
    if (calendar := _calendar_count(folded, source_time)) is not None:
        return calendar
    if (period := PERIODS.get(folded)) is None:
        return _named_month(text, source_time) or _lexicon_month(folded, source_time)
    relation, period_unit = period
    offset = _OFFSETS[relation]
    observed = _EPOCH + timedelta(milliseconds=source_time)
    if period_unit == "week":
        return _iso_week(source_time, offset)
    if period_unit == "month":
        year, month_index = divmod(observed.year * 12 + observed.month - 1 + offset, 12)
        return _window(year, month_index + 1)
    return _window(observed.year + offset)



def _named_month(text: str, source_time: int) -> tuple[int, int, TimeGranularity] | None:
    """Anchor "this|last|next <month name>" only where its common readings agree.

    "this September" is September of the source's year. "last September" is September of
    the year before when the source's month is September or earlier; later in the year it
    could also mean this year's September, so it stays unanchored. "next September" is
    September of the year after when the source's month is September or later, and stays
    unanchored earlier in the year for the same reason.
    """
    match = re.fullmatch(r"(last|this|next)\s+([A-Za-z]+)", text.strip(), re.IGNORECASE)
    month = None if match is None else _MONTHS.get(match[2].casefold())
    if match is None or month is None or (
        match[2].casefold() in ENGLISH_WORD_MONTHS and not match[2][0].isupper()
    ):
        return None
    return _relative_month(match[1].casefold(), month, source_time)


def _relative_month(
    relation: str, month: int, source_time: int,
) -> tuple[int, int, TimeGranularity] | None:
    """The month a relation names where its readings agree (`_named_month`)."""
    observed = _EPOCH + timedelta(milliseconds=source_time)
    if relation == "last" and observed.month > month:
        return None
    if relation == "next" and observed.month < month:
        return None
    return _window(observed.year + _OFFSETS[relation], month)


def _lexicon_month(folded: str, source_time: int) -> tuple[int, int, TimeGranularity] | None:
    """Anchor a `temporal_lexicon` named month under `_named_month`'s rule.

    "this, last or next <month>" in another language agrees as its English form
    does; "<month> of this, last or next year" names its year explicitly.
    """
    for pattern, relation in NAMED_MONTH_FORMS:
        if (match := pattern.fullmatch(folded)) is not None:
            return _relative_month(relation, _month_of(match), source_time)
    observed = _EPOCH + timedelta(milliseconds=source_time)
    for pattern, offset in YEAR_MONTH_FORMS:
        if (match := pattern.fullmatch(folded)) is not None:
            return _window(observed.year + offset, _month_of(match))
    return None


def _introduced_absolute(text: str) -> tuple[int, int | None, TimeGranularity] | None:
    """An absolute window, a leading 'on'/'in' naming the date's own window and 'since'
    opening a boundary with no invented end (as in locate_calendar_adjunct)."""
    introduced = re.fullmatch(r"(since|on|in)\s+(.+)", text, re.IGNORECASE)
    window = _absolute(introduced[2] if introduced is not None else text)
    if window is not None and introduced is not None and introduced[1].casefold() == "since":
        return window[0], None, window[2]
    return window


def resolve_time(
    expression: TimeExpressionV1, *, source_observed_at: int | None, pipeline_version: str,
) -> TemporalResolution:
    """Interpret a source expression with no ingestion/system-clock fallback.

    Args:
        expression: Strict model output; admission locates its text in a cited
            source separately. Supported forms include complete ISO dates/timestamps,
            English month dates, and closed relative calendar expressions.
        source_observed_at: Original signed epoch milliseconds of the source that
            contains the expression. Missing source time leaves relatives unanchored.
            The expression's kind picks the reading tried first; an expression that
            only the other reading anchors is anchored by it, with its provenance.
        pipeline_version: The owner policy's extraction pipeline, one of
            ``CURRENT_EXTRACTION_PIPELINES``. An absolute ``since``
            boundary preserves the raw expression and has no invented end; a
            leading ``on`` or ``in`` names the same window as the bare date. Relative
            expressions read week, month and year counts as calendar units (`_relative`).

    Returns:
        A source-time window and anchoring provenance. Unsupported calendar ranges,
        incomplete dates, and unknown expressions remain explicitly unanchored.

    Raises:
        ValueError: If the pipeline is not current, or the supplied source timestamp
            is not a signed 64-bit integer.
    """
    if pipeline_version not in CURRENT_EXTRACTION_PIPELINES:
        raise ValueError("Time resolution requires a current extraction pipeline")
    if source_observed_at is not None and (
        type(source_observed_at) is not int or not -(2**63) <= source_observed_at < 2**63
    ):
        raise ValueError("Source timestamp must be signed integer epoch milliseconds")
    if expression.kind == "none":
        return TemporalResolution(
            None, None, "unanchored", AnchorV1(kind="not_applicable"), "not_applicable"
        )
    try:
        absolute = _introduced_absolute(expression.text.strip())
    except (ValueError, OverflowError):
        absolute = None
    try:
        relative = None if source_observed_at is None else _relative(
            expression.text, source_observed_at,
        )
    except (ValueError, OverflowError):
        relative = None
    # The model's kind picks the reading tried first. An expression only the other
    # reading anchors keeps that reading's window and provenance: sources in
    # other languages often label "this September" ("Ce septembre") absolute.
    readings = [(absolute, False), (relative, True)]
    if expression.kind == "relative":
        readings.reverse()
    for window, observed in readings:
        if window is None or any(
            value is not None and not -(2**63) <= value < 2**63 for value in window[:2]
        ):
            continue
        anchor = (
            AnchorV1(kind="source_observed_at", source_time=source_observed_at) if observed
            else AnchorV1(kind="absolute_source_text")
        )
        return TemporalResolution(window[0], window[1], window[2], anchor, "resolved")
    return TemporalResolution(
        None, None, "unanchored", AnchorV1(kind="unanchored"),
        "missing_source_time" if expression.kind == "relative" and source_observed_at is None
        else "unsupported_expression",
    )


_CALENDAR_SPAN: Final = 6
"""The most whitespace tokens one written calendar date spans ("el 4 de mayo de 2023")."""
_TOKEN_EDGES: Final = (
    " .,;:!?\"'()[]{}<>\u2026\u3001\u3002\uff01\uff0c\uff1a\uff1b\uff1f\u060c\u061b\u061f"
)
_ORDINAL_DAY: Final = re.compile(r"(\d{1,2})(?:st|nd|rd|th)", re.IGNORECASE)
_EMBEDDED_DATE: Final = re.compile(
    "\\d{4}-\\d{2}(?:-\\d{2})?|\\d{4}\u5e74\\d{1,2}\u6708(?:\\d{1,2}[\u65e5\u53f7])?",
)


def calendar_windows(text: str) -> tuple[tuple[int, int], ...]:
    """Locate the explicit calendar days and months a free text names.

    Search reads these as the dates a query asks about. Only a written day or
    month of a written year counts, in the forms `resolve_time` reads as absolute:
    ISO dates, English month dates and the `temporal_lexicon` forms. A bare year,
    a relative phrase and a timestamp name no window. Calendar windows use UTC,
    as every source time does. The text is scanned left to right and the longest
    date wins, so "25 May, 2022" names its day and not its month.

    Args:
        text (str): Free text such as a Search query.

    Returns:
        tuple[tuple[int, int], ...]: Distinct half-open UTC millisecond windows,
            in text order.

    Examples:
        ```python
        calendar_windows("What did Ezra make on 9 November, 2022?")
        # ((1667952000000, 1668038400000),)
        calendar_windows("Where did she go in 2023?")  # ()
        ```
    """
    tokens = [
        day[1] if (day := _ORDINAL_DAY.fullmatch(stripped)) else stripped
        for stripped in (token.strip(_TOKEN_EDGES) for token in text.split())
    ]
    found: list[tuple[int, int]] = []

    def read(candidate: str) -> tuple[int, int] | None:
        try:
            window = _absolute(candidate)
        except (ValueError, OverflowError):
            return None
        if window is None or window[1] is None or window[2] not in {"DAY", "MONTH"}:
            return None
        return window[0], window[1]

    index = 0
    while index < len(tokens):
        for length in range(min(_CALENDAR_SPAN, len(tokens) - index), 0, -1):
            window = read(" ".join(token for token in tokens[index:index + length] if token))
            if window is not None:
                found.append(window)
                index += length
                break
        else:
            for embedded in _EMBEDDED_DATE.finditer(tokens[index]):
                window = read(embedded[0])
                if window is not None:
                    found.append(window)
            index += 1
    return tuple(dict.fromkeys(found))
