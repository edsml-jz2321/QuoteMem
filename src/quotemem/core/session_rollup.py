"""The session rollup: one dated memory per session recounting its main events in order.

A broad question about a long history ("in what order did I bring up ...", "summarize how my
project went") needs evidence from every part of it. The extraction supplement
(``core/extraction_supplement.py``) states one overview per extraction call, about 600 for a
BEAM-1M user, far more than the 100 hits of one Search hold: on BEAM-1M the gold messages of
event-ordering and summarization questions in the top 100 fell from 0.60 and 0.41 at 100K to
0.22 and 0.24.

An Add that sends the supplement therefore keeps one rollup per session: after the supplement
call answers, one more call without reasoning (``session-rollup-v1``) reads the session's
newest rollup and the supplement's facts about the unit's fresh sources, and returns the
session's updated timeline: at most `MAX_ROLLUP_ENTRIES` dated entries, in order, older ones
merged, within `ROLLUP_BODY_CHARS` (`fit_entries` shortens a longer reply entry by entry). The
program turns it into one fact of the reply (`rollup_fact`), opened by the session's date span
(`rollup_statement`) and grounded in the opening words of the unit's first fresh user source, so
the rollup is admitted, stored and replayed like any fact and carries its own origin,
``session_rollup``. Both calls run beside the extraction call, so the Add waits for neither
unless they outlast it.

A session's rollup is its rollup whose quoted parts are newest (`newest_per_session`): each
Add's rollup folds the one before it, so the newest covers the whole session. Extraction context
and every Search corpus leave rollups out; a summary or ordering read of a scope above
`ROLLUP_MIN_PARTS` readable parts opens with at most `MAX_ROLLUPS_FIRST` sessions' rollups, in
date order (``core/recall.py``).

A rollup is a summary-only memory (``recall.SUMMARY_ONLY_ORIGINS``): an in-chat forget judges it
like an overview and withdraws it when it states the forgotten fact (``forget_withdrawn:{s}``).
A session whose newest rollup is withdrawn has none: no read delivers it, and the next Add folds
nothing older, which may state the fact too, so its rollup starts again. Neither the rollup
call's input nor the stored rollup keeps an entry or note restating a forgotten statement
(`restates_forgotten`).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Annotated, Any, Final

from pydantic import Field

from quotemem.core.extraction_output import AdvertisedRequest, ExtractOutputV12, FactV12
from quotemem.core.sensitivity_rules import PROTECTED_CLASSES, classify
from quotemem.core.statement import MAX_STATEMENT_CHARS, token_seq
from quotemem.store.metadata import StrictModel

ROLLUP_SCHEMA_ID: Final = "session-rollup-v1"
ROLLUP_PROMPT_VERSION: Final = "session-rollup-v1"
ROLLUP_BODY_CHARS: Final = 900
"""The longest rollup body: its entries, joined by "; ". A longer reply is shortened entry by
entry (`fit_entries`), so every entry keeps its date and first words."""
MAX_ROLLUP_ENTRIES: Final = 12
"""The entries the prompt keeps at most; older ones merge."""
ROLLUP_MAX_TOKENS: Final = 1_200
"""The reply's completion bound."""
ROLLUP_TIMEOUT_MS: Final = 60_000
"""The read timeout of the rollup call."""
MAX_ROLLUP_NOTES: Final = 24
"""The supplement facts one rollup call reads, in reply order."""
ROLLUP_QUOTE_CHARS: Final = 160
"""The longest quote a rollup keeps of its user source."""
ROLLUP_MIN_PARTS: Final = 2048
"""Readable Evidence parts above which a summary or ordering read opens with rollups.

The boundary of the summary re-rank: below it the hits already reach most of a history,
and putting rollups first did not help there, so every read of a smaller scope is exactly
as if no rollup were stored."""
MAX_ROLLUPS_FIRST: Final = 5
"""The rollups a read opens with at most, so they never crowd out an ordering read's detail:
with more sessions, those whose rollups lie nearest the query, still in date order.

Chosen by cross-validation in internal replays: 5 keeps an ordering read's supporting
parts near the top as no rollup does, while still helping summary reads."""

ROLLUP_PROMPT: Final = (
    "You keep the dated rollup of one conversation session between a user and an assistant: its "
    "main events and topics, in the order they happened, as a short timeline. You get the "
    "timeline so far (nothing for the session's first part) and notes on the session's newest "
    "part, with its dates. Return the whole updated timeline in \"entries\": every entry so far, "
    "in its order, then the newest part's main events as one or two new entries. Each entry is "
    "its date (\"3 Mar\", or \"3-9 Mar\" for a merged entry), a colon and a telegraphic phrase "
    "of at most 100 characters naming the topics raised and what was decided, planned, done or "
    "recommended, with key names, figures and versions (\"3 Mar: started Flask budget app; "
    "asked about storage, assistant recommended SQLite\"). Write the user as \"the user\" when "
    "you name who acted, and mark an assistant's suggestion as the assistant's. "
    f"Keep at most {MAX_ROLLUP_ENTRIES} entries and {ROLLUP_BODY_CHARS} characters in all: when "
    "the newest entries would pass either limit, merge the oldest adjacent entries into one "
    "entry with a date range that still names each of their topics, rather than dropping a "
    "topic. Do not repeat the session's span, which the program adds. Leave out greetings and "
    "thanks, and anyone's health, religion, political views, sexual orientation or gender "
    "identity."
)
"""The rollup call's whole system message."""

_MONTHS: Final = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)  # fmt: skip
_DAY: Final = r"(\d{1,2}) (" + "|".join(_MONTHS) + r") (\d{4})"
_PREFIX: Final = re.compile(rf"^Session (?:from {_DAY} to {_DAY}|on {_DAY}|of undated messages): ")
"""The span `rollup_statement` opens a rollup with."""


class SessionRollupV1(StrictModel):
    """One ``session-rollup-v1`` reply: the session's whole updated timeline, oldest first."""

    entries: Annotated[list[Annotated[str, Field(max_length=2_000)]], Field(max_length=64)]


def advertise_rollup(request: AdvertisedRequest) -> dict[str, Any]:
    """The strict ``session-rollup-v1`` schema; it does not depend on the request."""
    del request
    return {
        "type": "object", "additionalProperties": False, "required": ["entries"],
        "properties": {"entries": {"type": "array", "items": {"type": "string"}}},
    }


@dataclass(frozen=True, slots=True)
class PriorRollup:
    """A session's newest rollup, as the next rollup call reads it.

    Attributes:
        statement: Its stored statement, span included.
        start: The first day of its span in epoch milliseconds, or None when undated.
    """

    statement: str
    start: int | None


def _day(value: int) -> date | None:
    try:
        return (datetime(1970, 1, 1, tzinfo=UTC) + timedelta(milliseconds=value)).date()
    except OverflowError:
        return None


def _spell(day: date) -> str:
    return f"{day.day} {_MONTHS[day.month - 1]} {day.year:04d}"


def _span(start: int | None, end: int | None) -> str:
    days = [day for value in (start, end) if value is not None and (day := _day(value))]
    if not days:
        return "Session of undated messages: "
    first, last = min(days), max(days)
    if first == last:
        return f"Session on {_spell(first)}: "
    return f"Session from {_spell(first)} to {_spell(last)}: "


def prior_rollup(statement: str) -> PriorRollup:
    """Read a stored rollup and the first day of the span it opens with.

    Args:
        statement: A rollup's statement, as `rollup_statement` wrote it.

    Returns:
        The rollup with its span's first day; None as the day when it is undated.
    """
    match = _PREFIX.match(statement)
    start = None
    if match is not None and (first := match.group(1) or match.group(7)) is not None:
        at = 1 if match.group(1) is not None else 7
        month = _MONTHS.index(match.group(at + 1)) + 1
        moment = datetime(int(match.group(at + 2)), month, int(first), tzinfo=UTC)
        start = int(moment.timestamp() * 1000)
    return PriorRollup(statement, start)


_FUNCTION_WORDS: Final = frozenset(
    "a an the and or but of to in on at for with by from about as into over is are was were be "
    "been being am do does did have has had i me my mine myself you your yours we our it its "
    "that this these those not no any all some user assistant said says told asked mentioned "
    "wants wanted".split())
"""Words that carry no content of a fact, for `restates_forgotten`."""
RESTATED_SHARE: Final = 0.5
"""The share of a forgotten statement's content words a text holding at least two of them
must hold to restate it."""


def _content(text: str) -> frozenset[str]:
    return frozenset(token for token in token_seq(text)
                     if token not in _FUNCTION_WORDS and len(token) > 1)


def restates_forgotten(text: str, forgotten: Sequence[str]) -> bool:
    """Whether a rollup entry or note restates a statement the user had forgotten.

    It does when it holds at least two of a forgotten statement's content words and at least
    `RESTATED_SHARE` of them. A rollup recounts many points, so the test is per entry and per
    note: the one that restates the fact is left out, the rest stay.

    Args:
        text: An entry of a rollup, or a supplement statement the rollup call would read.
        forgotten: The scope's forgotten statements (its tombstones).

    Returns:
        True when the text restates one of them.
    """
    words = _content(text)
    for statement in forgotten:
        fact = _content(statement)
        shared = len(words & fact)
        if fact and shared >= 2 and shared >= RESTATED_SHARE * len(fact):
            return True
    return False


def rollup_message(
    prior: PriorRollup | None, notes: Sequence[str], first: int | None, last: int | None,
    forgotten: Sequence[str] = (),
) -> str:
    """The rollup call's user message, without anything that restates a forgotten fact.

    Args:
        prior: The session's newest rollup, or None for its first part.
        notes: The supplement's statements about the newest part, in reply order.
        first: The newest part's earliest source time, in epoch milliseconds.
        last: Its latest source time.
        forgotten: The scope's forgotten statements; entries and notes that restate one
            are left out (`restates_forgotten`).

    Returns:
        The timeline so far, one entry per line, then the newest part's span and notes.
    """
    notes = [note for note in notes if not restates_forgotten(note, forgotten)]
    if prior is None:
        so_far = "Timeline so far: nothing yet (this is the session's first part)."
    else:
        span = _PREFIX.match(prior.statement)
        named = "" if span is None else f" ({span.group(0).removesuffix(': ')})"
        lines = "\n".join(f"- {entry}" for entry in rollup_entries(prior.statement)
                          if not restates_forgotten(entry, forgotten))
        so_far = f"Timeline so far{named}:\n{lines}"
    part = _span(first, last).removeprefix("Session ").removesuffix(": ")
    noted = "\n".join(f"- {note}" for note in notes[:MAX_ROLLUP_NOTES])
    return f"{so_far}\n\nNewest part, {part}:\n{noted}"


def _body(text: str) -> str:
    """The text without any span lines it opens with (a reply may copy the rollup's own)."""
    text = " ".join(text.split())
    while (match := _PREFIX.match(text)) is not None:
        text = text[match.end():]
    return text


def rollup_entries(statement: str) -> list[str]:
    """A stored rollup's entries, oldest first, without its span."""
    return [entry.strip() for entry in _body(statement).split("; ") if entry.strip()]


def _cut(entry: str, cap: int) -> str:
    if len(entry) <= cap:
        return entry
    cut = entry[: cap - 1].rsplit(" ", 1)[0].rstrip(" ,.:")
    return f"{cut or entry[: cap - 1]}…"


def fit_entries(entries: Sequence[str], budget: int) -> list[str]:
    """Shorten the longest entries, each at a word, until all of them fit the budget joined.

    Every entry keeps its date and first words: the cap is the largest one under which the
    entries, each cut to it, fit; an entry list that fits keeps every word. Entries that do
    not fit even at the shortest cap are left out from the oldest on.
    """
    kept = list(entries)
    while kept:
        separators = 2 * (len(kept) - 1)
        if sum(map(len, kept)) + separators <= budget:
            return kept
        low, high = 24, max(map(len, kept))
        while low < high:
            cap = (low + high + 1) // 2
            if sum(min(len(entry), cap) for entry in kept) + separators <= budget:
                low = cap
            else:
                high = cap - 1
        if sum(min(len(entry), low) for entry in kept) + separators <= budget:
            return [_cut(entry, low) for entry in kept]
        kept.pop(0)
    return kept


def rollup_statement(
    entries: Sequence[str], start: int | None, end: int | None, quoted: Sequence[str],
    forgotten: Sequence[str] = (),
) -> str | None:
    """The rollup's statement: its span, then its sourced entries fitted within a statement.

    An entry naming a protected class (health, religion, ...) that the rollup's quote does not
    state is left out: admission refuses a statement that names one its quotes do not state,
    and one such entry would otherwise cost the whole rollup.

    Args:
        entries: The model's timeline, oldest first.
        start: The session's first source day, in epoch milliseconds.
        end: The newest part's latest source time.
        quoted: The texts of the quotes the rollup fact keeps.
        forgotten: The scope's forgotten statements; an entry that restates one is left
            out, so a rollup made after a forget never carries the fact.

    Returns:
        The statement, or None when no entry is left.
    """
    prefix = _span(start, end)
    allowed = frozenset(classify(quoted) & PROTECTED_CLASSES)
    cleaned = [" ".join(_body(entry).replace(";", ",").split()) for entry in entries]
    sourced = [entry for entry in cleaned
               if entry and classify([entry]) & PROTECTED_CLASSES <= allowed
               and not restates_forgotten(entry, forgotten)]
    kept = fit_entries(sourced, min(ROLLUP_BODY_CHARS, MAX_STATEMENT_CHARS - len(prefix)))
    return f"{prefix}{'; '.join(kept)}" if kept else None


def rollup_quote(text: str) -> str:
    """The rollup's quote of a user source: its opening words, verbatim, at most
    `ROLLUP_QUOTE_CHARS` characters cut at a word."""
    text = text.strip()
    if len(text) <= ROLLUP_QUOTE_CHARS:
        return text
    cut = text[:ROLLUP_QUOTE_CHARS].rsplit(" ", 1)[0].rstrip()
    return cut or text[:ROLLUP_QUOTE_CHARS]


def rollup_fact(
    entries: Sequence[str], source_id: str, quote: str, supplement: ExtractOutputV12,
    start: int | None, end: int | None, forgotten: Sequence[str] = (),
) -> FactV12 | None:
    """Build the rollup's fact of the reply, or None when no entry of it can stand.

    The fact recounts the whole session, so its words come from the model's timeline. It says
    "the user", which admission accepts only with a quote of the user's own words, so its one
    support group quotes the opening of the unit's first fresh user source (`rollup_quote`).
    It anchors the user alone, names nothing and takes its keywords from the supplement's
    first fact about the user.

    Args:
        entries: The model's timeline, oldest first.
        source_id: The handle of the quoted fresh user source.
        quote: The quoted text, verbatim from that source.
        supplement: The supplement call's reply about the same fresh sources.
        start: The session's first source day.
        end: The newest part's latest source time.
        forgotten: The scope's forgotten statements (`rollup_statement`).

    Returns:
        The rollup fact, or None when no entry is kept.
    """
    statement = rollup_statement(entries, start, end, [quote], forgotten)
    if statement is None:
        return None
    keywords = next((list(fact.keywords) for fact in supplement.facts
                     if "user" in fact.anchors and fact.keywords), ["session"])
    return FactV12.model_validate({
        "support_groups": [{"quotes": [{"source_id": source_id, "quote": quote}]}],
        "anchors": ["user"], "statement": statement, "kind": "event",
        "time": {"kind": "none", "text": None}, "naming": [], "revises": [],
        "corrects_text": None, "keywords": keywords, "sensitivity": [],
    })


def newest_per_session[T, S: Hashable](
    rollups: Iterable[T], *, session_of: Callable[[T], S | None],
    recency: Callable[[T], tuple[int, ...]],
) -> dict[S, T]:
    """Each session's rollup: the one whose quoted parts are newest.

    Args:
        rollups: Rollup memories.
        session_of: The session a rollup's quoted parts belong to, or None when unknown.
        recency: The rollup's order key; larger is newer.

    Returns:
        The newest rollup of each session.
    """
    newest: dict[S, T] = {}
    for rollup in rollups:
        session = session_of(rollup)
        if session is not None and (
            session not in newest or recency(rollup) > recency(newest[session])
        ):
            newest[session] = rollup
    return newest


def by_session_start[S](
    sessions: Iterable[S], starts: Mapping[S, tuple[int, int]],
) -> list[S]:
    """Sessions in date order: their first source time, then first ingest order."""
    return sorted(sessions, key=lambda session: starts.get(session, (2**63, 2**63)))
