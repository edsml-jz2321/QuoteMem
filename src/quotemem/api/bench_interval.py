"""Day counts for interval reads over explicitly dated user statements; pure, no I/O.

"How many days passed between when I X and when I Y?" is answered from two
delivered statements that each name a calendar day ("I logged 3,600 words by
March 3", "my draft is due April 15"). The reader must find both, read the
year, and count the days, and often miscounts or pairs the wrong statements.
The temporal note (``bench_temporal``) counts only between memories whose own
validity is one day, which most user statements are not.

When an interval read names two clauses and, for each, exactly one calendar
day is stated next to the most of that clause's words (at least
``MIN_SHARED``) in the delivered user statements, the first hit opens with a
note giving both days, the words that name each, and the count of days and
weeks between them. Context stops at sentence boundaries and comma-separated
conjunctions so a later clause cannot name an earlier date's event. A day
without a year takes the year that puts it nearest
the day it was said. Any tie between two days, a missing year with no said
day, or two clauses naming one day gives no note: a wrong pair would be worse
than none. Nothing stored or ranked changes.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

from quotemem.api.bench_temporal import comparison_pair, temporal_note
from quotemem.core.delivery import DeliveryProjection, instant, said_on
from quotemem.core.recall_corpora import keyword_forms, sparse_terms

TOP = 30
"""Only statements among the first ``TOP`` delivered items are read."""
MIN_SHARED = 2
"""Clause words a day's context must repeat."""
CONTEXT = 10
"""Words either side of a stated day that make up its context."""
EXCERPT = 12
"""Words either side of a stated day that the note quotes."""

_MONTHS = (
    "january", "february", "march", "april", "may", "june", "july", "august", "september",
    "october", "november", "december",
)
_NUMBER = (
    {name: number for number, name in enumerate(_MONTHS, 1)}
    | {name[:3]: number for number, name in enumerate(_MONTHS, 1)}
    | {"sept": 9}
)
"""Month names and abbreviations by month number."""
_MONTH = r"(?P<{0}>" + "|".join((*_MONTHS, "sept", *(name[:3] for name in _MONTHS))) + r")\.?"
_DAY = re.compile(
    r"\b(?:" + _MONTH.format("m1") + r"\s+(?P<d1>\d{1,2})(?:st|nd|rd|th)?(?![\d:])"
    r"(?:,?\s+(?P<y1>\d{4}))?"
    r"|(?P<d2>\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?" + _MONTH.format("m2")
    + r"(?:,?\s+(?P<y2>\d{4}))?"
    r"|(?P<y3>\d{4})-(?P<n3>\d{2})-(?P<d3>\d{2}))\b",
    re.IGNORECASE,
)
_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+|,\s+(?:and|but)\s+", re.IGNORECASE)
_STOP = frozenset(
    "a an the i my me we our you your user users narrator s to of in on at for with and or "
    "is was were be been being am are have has had did do does it that this how many long "
    "much days day weeks week months month years year time passed pass between before after "
    "when since which what who first last from then ago today yesterday by until till".split()
)


@dataclass(frozen=True, slots=True)
class Dated:
    """One delivered item's text, whether the user said it, the day it was said, and the
    single day its event happened when the memory records one."""

    text: str
    user: bool
    said: date | None
    event: date | None = None


@dataclass(frozen=True, slots=True)
class _Day:
    day: date
    context: frozenset[str]
    excerpt: str


def _terms(text: str) -> frozenset[str]:
    return frozenset(
        form for term in sparse_terms(text) if term not in _STOP and not term.isdigit()
        and term not in _NUMBER
        for form in keyword_forms(term)
    )


def _calendar(match: re.Match[str], said: date | None) -> date | None:
    """The calendar day a match names; a day without a year takes the year nearest ``said``."""
    try:
        if match["y3"] is not None:
            return date(int(match["y3"]), int(match["n3"]), int(match["d3"]))
        month = _NUMBER[(match["m1"] or match["m2"]).lower()]
        day = int(match["d1"] or match["d2"])
        year = match["y1"] or match["y2"]
        if year is not None:
            return date(int(year), month, day)
        if said is None:
            return None
        candidates = [
            date(year, month, day) for year in (said.year - 1, said.year, said.year + 1)
            if 1 <= year <= 9999
        ]
    except ValueError:
        return None
    return min(candidates, key=lambda candidate: abs((candidate - said).days))


def _window(sentence: str, start: int, end: int, words: int) -> str:
    """The text ``words`` words either side of a span, marked where it was cut."""
    before, rest = sentence[:start].split(), sentence[start:].split()
    keep = len(sentence[start:end].split()) + words
    text = " ".join([*before[-words:], *rest[:keep]])
    return f"{'…' if len(before) > words else ''}{text}{'…' if len(rest) > keep else ''}"


def _days(item: Dated) -> list[_Day]:
    if not item.user:
        return []
    found: list[_Day] = []
    if item.event is not None:
        words = item.text.split()
        excerpt = " ".join(words[: 2 * EXCERPT]) + ("…" if len(words) > 2 * EXCERPT else "")
        found.append(_Day(item.event, _terms(item.text), excerpt))
    for sentence in (" ".join(part.split()) for part in _SENTENCE.split(item.text)):
        for match in _DAY.finditer(sentence):
            day = _calendar(match, item.said)
            if day is not None:
                near = _window(sentence, match.start(), match.end(), CONTEXT)
                found.append(_Day(day, _terms(near), _window(
                    sentence, match.start(), match.end(), EXCERPT,
                )))
    return found


def _named(clause: str, days: Sequence[_Day]) -> _Day | None:
    """The one stated day whose context repeats the most of a clause's words, or None."""
    words = _terms(clause)
    scored = [(len(words & day.context), day) for day in days]
    best = max((score for score, _ in scored), default=0)
    chosen = [day for score, day in scored if score == best]
    if best < MIN_SHARED or len({day.day for day in chosen}) != 1:
        return None
    return chosen[0]


def interval_note(query: str, items: Sequence[Dated]) -> str | None:
    """Count the days between the two stated days an interval read names.

    Args:
        query: The Search text.
        items: Delivered items in response order.

    Returns:
        A note naming each clause's day (`_named`) with the words that state
        it and the days and weeks between them, or None when the read is not
        an interval read or either day is missing, tied or shared.
    """
    pair = comparison_pair(query)
    if pair is None:
        return None
    days = [day for item in items[:TOP] for day in _days(item)]
    first, second = (_named(clause, days) for clause in pair)
    if first is None or second is None or first.day == second.day:
        return None
    first, second = sorted((first, second), key=lambda item: item.day)
    count = (second.day - first.day).days
    weeks, remainder = divmod(count, 7)
    duration = f"{count} day{'s' if count != 1 else ''}"
    if weeks:
        duration += f" ({weeks} week{'s' if weeks != 1 else ''}"
        duration += f" and {remainder} day{'s' if remainder != 1 else ''})" if remainder else ")"
    return (
        f"Note: from {_MONTHS[first.day.month - 1].capitalize()} {first.day.day}, {first.day.year} "
        f"till {_MONTHS[second.day.month - 1].capitalize()} {second.day.day}, {second.day.year} "
        f'is {duration}; stated as "{first.excerpt}" and "{second.excerpt}".'
    )


def _dated(projection: DeliveryProjection) -> Dated:
    """A delivered item as `interval_note` reads it."""
    memory = projection.memory
    if memory is not None:
        event = None
        if (
            memory.kind == "event" and memory.start_at is not None and memory.end_at is not None
            and memory.end_at - memory.start_at == 86_400_000 and memory.valid_through
            and memory.valid_from == memory.valid_to
            and (moment := instant(memory.start_at)) is not None
        ):
            event = moment.date()
        user = any(line.role == "user" for line in memory.quotes.lines)
        return Dated(memory.statement, user, memory.said, event)
    source = projection.source
    if source is None:
        return Dated("", user=False, said=None)
    return Dated(source.text, source.role == "user", said_on((source,)))


def delivered_interval(query: str, projections: Sequence[DeliveryProjection]) -> str | None:
    """`interval_note` over delivered items, unless the temporal note already counts the interval.

    The temporal note (``bench_temporal``) counts between two single-day event
    memories; when it states a calendar interval, this note would repeat it.
    """
    temporal = temporal_note(query, tuple(projections))
    if temporal is not None and "Calendar interval" in temporal:
        return None
    return interval_note(query, [_dated(projection) for projection in projections])
