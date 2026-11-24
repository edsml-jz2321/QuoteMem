"""Event-level delivery of a plain read that adds up or orders several events.

A question that counts, totals or orders things ("how many ...", "in total",
"in what order") is answered from several source messages, usually said in
different sessions. Ranked by relevance alone, its first slots go to the one
session that matches best, and one said event arrives several times: as the
Evidence part that holds the message and again as each memory that quotes it.
The reader then counts an event twice and misses the session that ranks last.

For such a read Search delivers each event once and spreads the first slots
over sessions. A memory whose every quoted part is already delivered as
Evidence is passed over, since that Evidence holds the whole message and its
said dates; the freed slots go to the next candidates. Then, within the first
`SPREAD_WINDOW` slots, a candidate whose sessions already hold `SPREAD_CAP`
delivered items waits until the window is full. Nothing else is reordered, and
other reads rank exactly as before.

A read that asks for an order ("in what order did I bring up ...", "list in
order", "from earliest to latest") is an aggregating read whose events are the
things the user said or did. Before its per-event cut, candidates said only by
the assistant follow all others, so the hit slots go to the user's own
messages and the memories quoting them, and assistant answers fill only the
slots left. A read asking back what the assistant said is not deferred. On the
retained BEAM-100K event-ordering reads the user's messages held every gold
event while assistant Evidence took about a third of the 100 slots.

A plain read about a named participant of a multi-party conversation
(`quotemem.core.recall_preparation.PreparedRecall.participant_read`: "which books
has Blair read", "when did Kai go to Rome") also delivers each event once,
without the session spread. Such a conversation says most events once in the
turn and again in every memory that quotes it, so the repeats take the slots
of the other events a third-person question needs (a list's other members,
the second fact of a multi-hop question). On the retained envelope LoCoMo
conversations the per-event cut put more gold turns in the first ten hits
while the session spread put fewer there and lost temporal answers. A
first-person read about the user's own history is left as it is.

Unlike a counting read, which must not see an event twice, a participant read
passes a repeating memory over only for the slot it frees: when no other
candidate takes it, the passed-over memories fill the slots left, after the
delivered events. A conversation smaller than the read's
``top_k`` would otherwise deliver its Evidence alone and none of the memories
that state its corrected, current facts.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Sequence

AGGREGATE_QUERY = re.compile(
    r"\b(?:how many|how much (?:\w+ ){0,3}(?:in total|altogether|combined|overall|so far)"
    r"|in total|total (?:amount|number|cost|of)|combined|altogether|on average|average"
    r"|in (?:what|which|the) order|the order (?:of|in which)|order of"
    r"|from (?:first|earliest) to (?:last|latest)|from (?:last|latest) to (?:first|earliest))\b",
    re.IGNORECASE,
)
"""Wording that counts, totals, averages or orders several events. Case-insensitive,
whole words, no topic vocabulary."""
ORDER_QUERY = re.compile(
    r"\b(?:in (?:what|which) (?:order|sequence)|the (?:order|sequence) (?:of|in which)|order of"
    r"|in the (?:order|sequence) (?:in which|that|of|they|it|I|we|you|these|those)\b"
    r"|(?:list|name|tell|walk|put|arrange|sort|rank|recount)\w*(?: [\w'’]+){0,4}"
    r" in order(?! (?:to|for)\b)"
    r"|in chronological order|chronologically|from (?:first|earliest) to (?:last|latest)"
    r"|(?:earliest|oldest) to (?:latest|newest|most recent))\b",
    re.IGNORECASE,
)
"""Wording that asks for several events in time order. A bare "in order" ("my documents
are in order", "in order to") asks for none. Case-insensitive, whole words, no topic
vocabulary."""
DURATION_QUERY = re.compile(
    r"\bhow many (?:seconds|minutes|hours|days|weeks|months|years)\b", re.IGNORECASE,
)
"""A span of time ("how many days passed between ..."), which counts no events unless
the query adds them up (`SPENT_QUERY`). On the retained BEAM-100K reads, spreading
such two-event questions lost temporal-reasoning score."""
SPENT_QUERY = re.compile(r"\b(?:spen[dt]|spending)\b", re.IGNORECASE)
"""Time adding up several events ("how many days did I spend ...")."""
SPREAD_WINDOW = 20
"""Leading slots over which an aggregating read spreads its sessions."""
SPREAD_CAP = 3
"""Delivered items one session may hold within `SPREAD_WINDOW` while other sessions wait."""


def aggregate_query(query: str) -> bool:
    """Return whether a read counts, totals or orders several events.

    Aggregating wording is `AGGREGATE_QUERY` or the ordering wording of `order_query`.

    A span of time between events (`DURATION_QUERY`) does not count, unless the
    query says the time was spent (`SPENT_QUERY`).

    Args:
        query: The frozen query text.

    Returns:
        bool: True when the query carries aggregating wording.
    """
    if order_query(query):
        return True
    if SPENT_QUERY.search(query) is None:
        query = DURATION_QUERY.sub(" ", query)
    return AGGREGATE_QUERY.search(query) is not None


def said_by_user_first[T](ranked: Sequence[T], *, assistant_only: Callable[[T], bool]) -> list[T]:
    """Move candidates said only by the assistant after all others, each group in its order.

    Args:
        ranked: Candidates in delivery order.
        assistant_only: Whether every part a candidate shows was said by the assistant.

    Returns:
        list[T]: The same candidates, reordered.
    """
    flags = [assistant_only(member) for member in ranked]
    return [member for member, flag in zip(ranked, flags, strict=True) if not flag] + [
        member for member, flag in zip(ranked, flags, strict=True) if flag
    ]


def order_query(query: str) -> bool:
    """Return whether a read asks for several events in time order (`ORDER_QUERY`).

    Args:
        query: The frozen query text.

    Returns:
        bool: True when the query carries ordering wording.
    """
    return ORDER_QUERY.search(query) is not None


def one_per_event[T](
    ranked: Sequence[T],
    count: int,
    *,
    part_of: Callable[[T], str | None],
    quoted_by: Callable[[T], frozenset[str]],
    refill: bool = False,
) -> list[T]:
    """Keep the first ``count`` candidates, each event once, at its best rank.

    A memory is passed over when every part it quotes is itself among the kept
    candidates. Those parts then move up to the memory's place when it ranked
    above them, so the event keeps its rank in its fuller form. Passing a memory
    over admits the next candidate, which may show more parts, so the cut is
    repeated until it is stable.

    Args:
        ranked: Candidates in delivery order, longer than ``count`` when available.
        count: Slots to fill.
        part_of: The Evidence id a candidate delivers, None for a memory.
        quoted_by: The Evidence ids a memory quotes, empty for Evidence.
        refill: Whether passed-over memories fill, in rank order after the kept
            candidates, the slots no other candidate took (a participant read, which
            passes a repeat over only for the slot it frees).

    Returns:
        list[T]: At most ``count`` candidates.
    """
    passed: set[int] = set()
    while True:
        kept = [index for index in range(len(ranked)) if index not in passed][:count]
        shown = {part: index for index in kept if (part := part_of(ranked[index])) is not None}
        repeated = {
            index for index in kept
            if (quotes := quoted_by(ranked[index])) and quotes <= shown.keys()
        }
        if repeated <= passed:
            break
        passed |= repeated
    place = {index: index for index in kept}
    for index in sorted(passed):
        for part in quoted_by(ranked[index]):
            if part in shown:
                place[shown[part]] = min(place[shown[part]], index)
    delivered = [ranked[index] for index in sorted(kept, key=lambda index: (place[index], index))]
    if refill:
        delivered += [ranked[index] for index in sorted(passed)][:count - len(delivered)]
    return delivered


def spread_sessions[T](
    ranked: Sequence[T],
    *,
    sessions_of: Callable[[T], frozenset[str]],
    window: int | None = None,
    cap: int | None = None,
) -> list[T]:
    """Reorder the first ``window`` slots so no session holds over ``cap`` while others wait.

    A candidate waits when each of its sessions already holds ``cap`` items in
    the window; waiting candidates follow the full window in their original
    order, then the rest. A candidate without a session never waits.

    Args:
        ranked: Candidates in delivery order.
        sessions_of: The source sessions a candidate shows.
        window: Leading slots to spread; `SPREAD_WINDOW` when None.
        cap: Items per session within the window; `SPREAD_CAP` when None.

    Returns:
        list[T]: The same candidates, reordered.
    """
    window = SPREAD_WINDOW if window is None else window
    cap = SPREAD_CAP if cap is None else cap
    head: list[T] = []
    waiting: list[T] = []
    held: Counter[str] = Counter()
    for index, member in enumerate(ranked):
        if len(head) == window:
            return [*head, *waiting, *ranked[index:]]
        sessions = sessions_of(member)
        if sessions and all(held[session] >= cap for session in sessions):
            waiting.append(member)
            continue
        head.append(member)
        held.update(sessions)
    return [*head, *waiting]
