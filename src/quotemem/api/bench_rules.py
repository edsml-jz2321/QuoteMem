"""Standing-instruction note for rendered Search text; pure, no I/O.

A user gives standing instructions for later answers ("Always format dates as
Month Day, Year when I ask about timelines"). Ranking treats such a statement
like any other memory, so it is often delivered far below the facts it
governs, where a reader answering from the top never applies it. When a
delivered user item states an instruction for future answers whose subject
meets the query, the first hit opens with a note quoting it. Nothing stored or
ranked changes, and the note adds no fact: it repeats the user's own words.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass

MAX_RULES = 2
"""Most standing instructions one note quotes."""
MIN_SHARED = 2
"""Query words an instruction without a "when I ask" condition must repeat."""
MAX_QUOTE = 400

_CONDITION = re.compile(r"\bwhen(?:ever)? (?:i|the user) asks? about\b", re.IGNORECASE)
"""The condition of an instruction: "when I ask about ..." names the answers it governs."""
_RULE = re.compile(
    r"^(?:please\s+)?(?:always|never)\b|\b(?:from now on|going forward)\b|"
    + _CONDITION.pattern,
    re.IGNORECASE,
)
"""An instruction for future answers: an imperative always/never, or a `_CONDITION`."""
_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")
_WORD = re.compile(r"[a-z][a-z']*")
_STOP = frozenset(
    """a an the and or but if of to in on at for from by with about as into than then so that
    this these those it its i i'm i've i'd me my mine we our you your he she they them their his
    her is are was were be been being am do does did done have has had having can could would
    should will shall may might must not no any some all each every just also too very really
    up out over more most much many what which who whom whose when where why how there here
    such own same other another again ever still yet even always never please user user's
    users ask asks asked asking answer answers want wants""".split()
)


@dataclass(frozen=True, slots=True)
class Stated:
    """One delivered item's text and whether the user said it."""

    text: str
    user: bool


def _words(text: str) -> frozenset[str]:
    """Content words with a regular English suffix and final e folded off."""
    folded: set[str] = set()
    for word in _WORD.findall(text.lower().replace("-", " ")):
        if word in _STOP or len(word) < 3:
            continue
        for suffix in ("ing", "ed", "es", "s"):
            if len(word) > len(suffix) + 3 and word.endswith(suffix):
                word = word[: -len(suffix)]
                break
        folded.add(word[:-1] if len(word) > 4 and word.endswith("e") else word)
    return frozenset(folded)


def _quoted(sentence: str) -> str:
    return sentence if len(sentence) <= MAX_QUOTE else sentence[: MAX_QUOTE - 1] + "…"


def standing_note(items: Sequence[Stated], query: str) -> str | None:
    """Quote the user's standing instructions that apply to the query, best match first.

    Args:
        items: Delivered items in response order.
        query: The Search text.

    Returns:
        A note quoting at most ``MAX_RULES`` distinct user instructions for
        future answers (`_RULE`) whose condition (`_CONDITION`)
        shares a word with the query or, without a condition, whose words
        share ``MIN_SHARED``; ranked by shared words, then delivery order.
        None when no instruction applies.
    """
    terms = _words(query)
    found: list[tuple[int, int, str, frozenset[str]]] = []
    for rank, item in enumerate(items):
        if not item.user:
            continue
        for text in (" ".join(part.split()) for part in _SENTENCE.split(item.text)):
            if not _RULE.search(text):
                continue
            words = _words(text)
            condition = _CONDITION.search(text)
            topic = words if condition is None else _words(text[condition.end():])
            shared = len(topic & terms)
            if shared >= (1 if condition is not None else MIN_SHARED):
                found.append((-shared, rank, text, words))
    chosen: list[tuple[str, frozenset[str]]] = []
    for _, _, text, words in sorted(found, key=lambda rule: rule[:2]):
        if any(len(words & seen) * 2 > min(len(words), len(seen)) for _, seen in chosen):
            continue
        chosen.append((text, words))
        if len(chosen) == MAX_RULES:
            break
    if not chosen:
        return None
    quoted = "; ".join(f'"{_quoted(text)}"' for text, _ in chosen)
    return f"Standing instruction from the user for answers like this one: {quoted}"
