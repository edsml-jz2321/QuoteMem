"""Source-grounded calendar help for explicit comparisons of delivered events."""

import re
from datetime import UTC, datetime

from aml.api.bench_dates import relative_reading
from aml.core.delivery import DeliveryProjection
from aml.core.recall_corpora import keyword_forms, sparse_terms

_ORDER = re.compile(r"\b(?:order|earliest|latest|chronological|first)\b", re.IGNORECASE)
_INTERVAL = re.compile(
    r"\bhow (?:many (?:days|weeks|months|years)|long)\b", re.IGNORECASE,
)
_BETWEEN = re.compile(r"\bbetween\s+", re.IGNORECASE)
_AND = re.compile(r"\s+and\s+", re.IGNORECASE)
_ANCHOR = re.compile(r"\s+(?:when|before|after)\s+", re.IGNORECASE)
_STOP = frozenset(
    "a an the i my me we our user users narrator s to of in on at for with and or "
    "is was were be been being am are have has had did do does it that this "
    "how many long much days day weeks week months month years year time "
    "passed pass between before after when since which what who event events "
    "first last earliest latest order from then ago today yesterday".split()
)


def _terms(text: str) -> frozenset[str]:
    return frozenset(
        form for term in sparse_terms(text) if term not in _STOP
        for form in keyword_forms(term)
    )



def comparison_pair(query: str) -> tuple[str, str] | None:
    """The two clauses of an interval query, found by linear delimiter scans."""
    if not _INTERVAL.search(query):
        return None
    if (between := _BETWEEN.search(query)) and (join := _AND.search(query, between.end())):
        left, right = query[between.end():join.start()], query[join.end():]
    elif anchor := _ANCHOR.search(query):
        left, right = query[:anchor.start()], query[anchor.end():]
    else:
        return None
    return (left, right) if left.strip() and right.strip() else None


def temporal_note(query: str, projections: tuple[DeliveryProjection, ...]) -> str | None:
    """Explain dated delivered events for an explicit interval or ordering query.

    Only single-day event validity is used; a source day never becomes an event
    day. Ambiguous week-relative readings are excluded. The note is descriptive
    and bounded, and never supplies a current date or asserts a complete history.
    """
    pair = comparison_pair(query)
    if pair is None and not _ORDER.search(query):
        return None
    terms = _terms(query)
    events: list[tuple[str, str, frozenset[str]]] = []
    for projection in projections:
        memory = projection.memory
        if (
            memory is None or memory.kind != "event" or memory.start_at is None
            or memory.end_at is None or memory.end_at - memory.start_at != 86_400_000
            or not memory.valid_through or memory.valid_from != memory.valid_to
            or relative_reading(memory.statement, memory.said) is not None
        ):
            continue
        words = _terms(memory.statement)
        if len(words & terms) < 2:
            continue
        try:
            day = datetime.fromtimestamp(memory.start_at / 1000, UTC).date().isoformat()
        except (OverflowError, OSError, ValueError):
            continue
        value = (day, memory.statement, words)
        if value not in events:
            events.append(value)
    if len({event[0] for event in events}) < 2:
        return None
    selected = sorted(events, key=lambda event: (event[0], -len(event[2] & terms), event[1]))
    shown: list[tuple[str, str, frozenset[str]]] = []
    for event in selected:
        if sum(other[0] == event[0] for other in shown) < 2:
            shown.append(event)
    if len(shown) > 8:
        return None
    descriptions = "; ".join(f"{day}: {text}" for day, text, _ in shown)
    note = f"Note: Matching delivered event dates, earliest first: {descriptions}"
    if len(note) > 3000:
        return None
    if pair is not None:
        chosen: list[str] = []
        for clause in pair:
            words = _terms(clause)
            scores = [(len(words & event[2]), event[0]) for event in events]
            best = max(score for score, _ in scores)
            days = {day for score, day in scores if score == best}
            if best < 2 or len(days) != 1:
                break
            chosen.append(days.pop())
        if len(chosen) == 2 and chosen[0] != chosen[1]:
            days = abs((datetime.fromisoformat(chosen[1]) - datetime.fromisoformat(chosen[0])).days)
            weeks, remainder = divmod(days, 7)
            note += f" Calendar interval between {chosen[0]} and {chosen[1]}: {days} days"
            if weeks:
                duration = f"{weeks} week{'s' if weeks != 1 else ''}"
                if remainder:
                    duration += f" and {remainder} day{'s' if remainder != 1 else ''}"
                note += f" ({duration})"
            note += "."
    return note
