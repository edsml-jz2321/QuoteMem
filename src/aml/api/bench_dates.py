"""Calendar notes for relative time words in rendered Search text; pure, no I/O.

A relative phrase ("yesterday", "last week", "two weeks ago") is relative to the
day its source was said. Search shows that day next to the words, so a reader
need not do the arithmetic. Day, month and year phrases read as calendar dates;
week-based phrases ("last week", "last Friday", "two weeks ago") and durations
("in the last month", "over the past two months") stay relative
to the day they were said ("the Friday before Sunday, 23 July 2023"), as a
reader keeps them, so no computed day range competes with the speaker's own
words. Nothing stored changes: the note is derived from the rendered text and
its single source day at render time.
"""

import re
from datetime import date, timedelta

from aml.core.delivery import spell_day, spell_month

_WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
_NUMBERS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}  # fmt: skip
_DAYS = {"day before yesterday": -2, "yesterday": -1, "last night": -1, "today": 0,
         "tonight": 0, "this morning": 0, "this afternoon": 0, "this evening": 0,
         "tomorrow": 1}  # fmt: skip
_COUNT = "|".join(_NUMBERS) + r"|\d{1,2}"
_PHRASE = re.compile(
    r"\b(?:the\s+)?(?:"
    r"(?:(?:in|over|during|for|within) (?:the )?(?:last|past)|(?:this )?past"
    r"|last(?= (?:" + _COUNT + r") (?:week|month)s?\b)) "
    r"(?:(?P<duration_count>" + _COUNT + r") )?(?P<duration_unit>week|month)s?"
    r"|(?P<day>day before yesterday|yesterday|last night|today|tonight|tomorrow"
    r"|this (?:morning|afternoon|evening))"
    r"|(?P<side>last|this past|past|next) (?P<span>weekend|week)"
    r"|(?P<cal_side>last|this|next) (?P<cal>month|year)"
    r"|(?P<wd_side>last|this past) (?P<weekday>" + "|".join(_WEEKDAYS) + r")"
    r"|(?P<count>" + _COUNT + r") (?P<unit>day|week|month|year)s? ago"
    r")\b",
    re.IGNORECASE,
)
MAX_NOTES = 4


def _shift_month(day: date, months: int) -> tuple[int, int] | None:
    year, month = divmod(day.year * 12 + day.month - 1 + months, 12)
    return (year, month + 1) if 1 <= year <= 9999 else None


def _resolve(match: re.Match[str], said: date) -> str | None:
    """One phrase's calendar reading relative to the day it was said, or None."""
    spoken = spell_day(said)
    try:
        if (unit := match["duration_unit"]) is not None:
            count = match["duration_count"]
            if count is None:
                return f"the {unit.lower()} before {spoken}"
            count = count.lower()
            number = _NUMBERS[count] if count in _NUMBERS else int(count)
            return f"{count} {unit.lower()}{'' if number == 1 else 's'} before {spoken}"
        if (phrase := match["day"]) is not None:
            return spell_day(said + timedelta(days=_DAYS[phrase.lower()]))
        if (span := match["span"]) is not None:
            side = match["side"].lower()
            relation = "after" if side == "next" else "before"
            return f"the {span.lower()} {relation} {spoken}"
        if (unit := match["cal"]) is not None:
            offset = {"last": -1, "this": 0, "next": 1}[match["cal_side"].lower()]
            if unit.lower() == "year":
                year = said.year + offset
                return f"{year:04d}" if 1 <= year <= 9999 else None
            shifted = _shift_month(said, offset)
            return None if shifted is None else spell_month(*shifted)
        if (weekday := match["weekday"]) is not None:
            return f"the {weekday.capitalize()} before {spoken}"
        count = match["count"].lower()
        number = _NUMBERS[count] if count in _NUMBERS else int(count)
        unit = match["unit"].lower()
        if unit == "day":
            return spell_day(said - timedelta(days=number))
        if unit == "week":
            return f"{count} week{'' if number == 1 else 's'} before {spoken}"
        if unit == "month":
            shifted = _shift_month(said, -number)
            return None if shifted is None else f"around {spell_month(*shifted)}"
        year = said.year - number
        return f"around {year:04d}" if 1 <= year <= 9999 else None
    except OverflowError:
        return None


def _relative(match: re.Match[str]) -> bool:
    """Whether a phrase keeps its reading relative to the source day."""
    unit = match["unit"]
    return (
        match["duration_unit"] is not None or match["span"] is not None
        or match["weekday"] is not None or (unit is not None and unit.lower() == "week")
    )


def _readings(text: str, said: date) -> list[tuple[str, str, bool]]:
    """The first ``MAX_NOTES`` distinct phrases with a reading, in text order.

    Each entry is the phrase as written, its reading and whether it stays relative.
    """
    readings: list[tuple[str, str, bool]] = []
    seen: set[str] = set()
    for match in _PHRASE.finditer(text):
        phrase = match.group().lower()
        if phrase in seen:
            continue
        seen.add(phrase)
        reading = _resolve(match, said)
        if reading is not None:
            readings.append((match.group(), reading, _relative(match)))
        if len(readings) == MAX_NOTES:
            break
    return readings


def relative_reading(text: str, said: date | None) -> str | None:
    """The first week or duration reading among ``date_notes(text, said)``, or None.

    Such a reading ("the week before Thursday, 16 March 2023") places the event
    relative to the day it was said; a computed day range for the same phrase
    would contradict that form.
    """
    if said is None:
        return None
    return next((reading for _, reading, relative in _readings(text, said) if relative), None)


def date_notes(text: str, said: date | None) -> tuple[str, ...]:
    """Calendar readings of the relative time phrases in ``text``.

    Args:
        text: Rendered statement or Evidence text.
        said: The single UTC day its source was said, or None when unknown or
            ambiguous (no note is then derived).

    Returns:
        At most ``MAX_NOTES`` notes such as ``"yesterday" = Sunday, 7 May 2023``
        or ``"last Friday" = the Friday before Monday, 8 May 2023``, in text
        order, one per distinct phrase. Ambiguous phrases ("this Friday",
        "next Friday", "a few days ago") get none.
    """
    if said is None:
        return ()
    return tuple(f'"{phrase}" = {reading}' for phrase, reading, _ in _readings(text, said))
