"""Deterministic line-start speaker labels: the code-settled part of B.3a (PD:P6).

A transcript written into message text names each turn with a line-leading
``Name:`` label. This module finds those labels in the readable text of one
message and settles each one without a model call:

- ``speaker`` when the name equals the message's metadata speaker under
  ``name_norm``, or when it is a generic user form (``User:``) on a message
  without a metadata speaker, which can only designate the authenticated user
  (B.4.2 (1), X-Q3);
- ``ambiguous`` otherwise: the label opens a turn whose speaker nobody may
  infer (B.4.1, X-Q2) until a model pass settles it.

The grammar reads a label at the start of the message or of a line, after
optional spaces: a name, optional spaces, then a colon that ends the line or
is followed by whitespace (a full-width colon may be followed by anything, as
CJK text sets no space after it). It refuses:

- a colon inside a sentence, as only a line start can open a label, and
  a run of words that do not look like a name (a lowercase word that is not a
  name particle, a digit embedded in a word, or a sentence word such as ``Dear`` or
  ``Then``);
- times (``10:30``) and URLs (``https://``), whose colon is followed by text;
- a short list of non-name prefixes such as ``Note:``, ``P.S.:``, ``Q:`` and
  ``Update:``;
- a heading whose last word is a heading noun (``Action Items:``), and a CJK
  phrase holding a pronoun or particle that no personal name contains (LB-5);
- a line start inside a quotation opened on an earlier line and closed later;
- a name of more than four words or eight tokens, or one that is not a name
  under A0.6.

In an AMLB speaker slot, a name may end in one separate number or ``#number``
(``Speaker 2``). Outside an envelope, metadata must confirm a numbered name:
capitalisation alone cannot distinguish it from a numbered heading. A bare
``#number`` is accepted only in an envelope. Bare numerals and calendar dates
are not speaker labels.

An AMLB envelope header (``core/envelope.py``) at a line start is transparent:
the label is read after it, so ``[set][session_1][D1:2][text] Name:`` is found
like ``Name:``. The label's range then begins at the header, so the turn it opens
covers its own framing and a header never forms a source of its own. The
envelope's format places the turn's speaker in exactly that slot, so on a message
without a metadata speaker the label is settled ``speaker`` in code, with no
model pass; a metadata speaker still confirms or contradicts it. An envelope
label that only restates the message's wire role (``User:`` on a user message,
``Assistant:`` on an assistant message) names no one the role does not already
name, and is not a label: such a message reads as before.

A label whose name is the metadata speaker's needs only to be a name under
A0.6: the metadata confirms it, so a lowercase handle such as ``kit_99:`` is a
label of its own message.

A label it locates at worst blocks inheritance (``ambiguous``), while a label
it misses lets the previous speaker run on, so the exclusions stay narrow: an
opening quotation mark that is never closed (the apostrophe of ``'90s``)
quotes nothing.

Labels are read through ``render_masked`` (D.2), so a label is never read in
withheld or unreadable text.

`label_view` is the one deterministic view builder of B.3a: the locator plus
the durable model decisions (``store/source_labels.py`` stores and reads them)
for the labels the locator leaves ``ambiguous``, matched by exact byte range.
It calls no provider, and a located label without a decision stays
``ambiguous``.
"""

from __future__ import annotations

import re
from bisect import bisect_left, bisect_right
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from aml.core.add_read_view import EvidenceRecord, MessageKey
from aml.core.entities import EvidenceKind
from aml.core.envelope import envelope_end
from aml.core.readability import MaskedText, Readability
from aml.core.source_labels import SourceLabelState
from aml.core.statement import (
    COMBINING_MARKS,
    LABEL_MAX_TOKENS,
    bound_length,
    is_name,
    name_norm,
    token_seq,
)
from aml.ingestion.sources import (
    GENERIC_USER_FORMS,
    LabelState,
    MessageLabels,
    TurnLabel,
    message_key,
)

LABEL_MAX_WORDS: Final = 4
"""Words of a label name, so a sentence ending in a colon is not a name."""

NON_NAME_PREFIXES: Final = frozenset({
    # Notes, postscripts and asides.
    "btw", "caution", "edit", "edited", "fyi", "hint", "important", "n b", "nb", "note",
    "notes", "p p s", "p s", "pps", "ps", "reminder", "tip", "tips", "tl dr", "tldr", "todo",
    "update", "warning",
    # Questions, answers and list heads.
    "a", "actually", "agenda", "answer", "conclusion", "cons", "correcting", "correction",
    "example", "goal", "how", "plan", "pros", "q", "question", "result", "step", "steps",
    "summary", "what", "when", "where", "who", "why",
    # Message and form fields.
    "address", "bcc", "cc", "date", "e mail", "email", "error", "fw", "fwd", "input", "link",
    "location", "name", "output", "phone", "price", "re", "source", "status", "subject",
    "tel", "time", "total", "url", "website",
    # Days and times of day.
    "afternoon", "evening", "friday", "monday", "morning", "saturday", "sunday", "thursday",
    "today", "tomorrow", "tonight", "tuesday", "wednesday", "yesterday",
    # Chinese, Japanese and Korean notes, corrections, questions, answers and list heads.
    "\u5907 \u6ce8", "\u9644 \u8a00", "\u66f4 \u65b0", "\u95ee", "\u7b54", "\u95ee \u9898",
    "\u7b54 \u6848", "\u63d0 \u793a", "\u6ce8", "\u6ce8 \u610f", "\u603b \u7ed3",
    "\u30e1 \u30e2", "\u8cea \u554f", "\u7b54 \u3048", "\ucc38 \uace0", "\uc9c8 \ubb38",
    "\ub2f5", "\u66f4 \u6b63", "\u7ea0 \u6b63", "\u8865 \u5145", "\u8bf4 \u660e",
    "\u4f8b \u5982", "\u6bd4 \u5982", "\u539f \u56e0", "\u7ed3 \u679c", "\u8a02 \u6b63",
    "\u8ffd \u8a18", "\uc815 \uc815", "\ucd94 \uc2e0", "\uc694 \uc57d",
})
"""``name_norm`` forms of common line prefixes that are never a speaker's name."""

ROLE_FORMS: Final = {
    "user": GENERIC_USER_FORMS, "assistant": frozenset({"assistant", "the assistant"}),
}
"""``name_norm`` forms that restate a wire role; an envelope label of its own role is no label."""

HEADING_NOUNS: Final = frozenset({
    "agenda", "answers", "checklist", "conclusions", "contents", "decisions", "deliverables",
    "details", "findings", "goals", "guidelines", "highlights", "ideas", "ingredients",
    "instructions", "issues", "itinerary", "items", "learnings", "lessons", "notes",
    "objectives", "options", "overview", "points", "priorities", "questions",
    "recap", "recommendations", "reminders", "requirements", "results", "risks", "schedule",
    "steps", "summary", "takeaways", "tasks", "thoughts", "timeline", "todos", "updates",
})
"""``name_norm`` forms of the last word of a heading (``Action Items:``, ``Key Takeaways:``).

A line-start label whose last word is one of them reads as a heading, never a
name (LB-5): settling it in code saves the model pass a call it would decide
``content``. Each is a common noun of lists and notes, not a surname.
"""

CJK_FUNCTION_CHARACTERS: Final = frozenset("们的着吗呢吧啊呀么我你您她他它")
"""Chinese pronouns and particles that no personal name contains.

A CJK run before a colon that holds one is a phrase, not a name (LB-5).
Characters that also occur in names (``了``, ``过``, ``在``) are left out.
"""

NAME_PARTICLES: Final = frozenset({
    "al", "bin", "binti", "da", "das", "de", "del", "della", "den", "der", "di", "do", "dos",
    "du", "el", "la", "le", "ter", "ten", "van", "von", "y",
})
"""Lowercase words a name may contain after its first word (the ``van`` of a Dutch surname)."""

SENTENCE_WORDS: Final = frozenset({
    "about", "after", "and", "as", "at", "because", "before", "but", "by", "dear", "for",
    "from", "hello", "hey", "hi", "i", "if", "in", "is", "it", "me", "my", "of", "on", "or",
    "our", "that", "the", "their", "then", "these", "they", "this", "those", "to", "was", "we",
    "when", "with", "you", "your",
})
"""Words no name contains, even capitalised at a sentence start (``Dear Odile:``, ``Then Odile:``).

Articles, pronouns, conjunctions, prepositions and greetings; words that are also
common surnames (``An``, ``So``) are left out, as a missed label is the costlier error.
"""

_NUMBERED_NON_NAMES: Final = frozenset({
    "jan", "january", "feb", "february", "mar", "march", "apr", "april", "may",
    "jun", "june", "jul", "july", "aug", "august", "sep", "sept", "september",
    "oct", "october", "nov", "november", "dec", "december", "score", "ratio",
})
"""Calendar dates and numeric fields must not become names through a numeric suffix."""

_LABEL = re.compile(
    r"[ \t\u00a0\u3000]*(?P<name>#[0-9]+|[^\W\d_]"
    rf"(?:[\w{COMBINING_MARKS}.'\u2019\u00b7\u30fb\- ]*[\w{COMBINING_MARKS}.])?(?: +#[0-9]+)?)"
    r"[ \t]*(?P<colon>[:\uff1a])",
)
"""A line-start name and colon. A name keeps its combining marks (ML-1), so a
Devanagari, Bengali, Thai or vocalized Arabic name is a label name."""
_PAIRS: Final = {
    '"': '"', "'": "'", "\u201c": "\u201d", "\u2018": "\u2019", "\u300c": "\u300d",
    "\u300e": "\u300f",
}
_MARKS: Final = frozenset(_PAIRS) | frozenset(_PAIRS.values())
_MARK: Final = re.compile("[" + re.escape("".join(sorted(_MARKS))) + "]")
_APOSTROPHES: Final = frozenset({"'", "\u2019"})


@dataclass(frozen=True, slots=True)
class LabelMatch:
    """One label found in a text, in character offsets of that text.

    Attributes:
        start (int): The first character of the label: its envelope header if
            one opens the line, else its name.
        end (int): The character after the colon.
        name (str): The name as written, without surrounding spaces or header.
        framed (bool): An envelope header opens the label's line.
    """

    start: int
    end: int
    name: str
    framed: bool = False


def _quote_changes(text: str) -> tuple[int, ...]:
    """Offsets of the opening and closing mark of every quotation, in text order.

    The quotation rule: a quotation runs from an opening mark to the next
    matching closing mark, and other marks inside it change nothing; a mark
    after a backslash and an apostrophe between two alphanumerics (``don't``)
    are not marks. Two changes keep a mark that cannot open from hiding the
    rest of the message: an apostrophe after an alphanumeric (``the dogs'``)
    never opens, and an opening mark with no closing mark after it opens
    nothing.
    """
    marks: list[tuple[int, str]] = []
    for mark in _MARK.finditer(text):
        index, char = mark.start(), mark.group()
        if index and text[index - 1] == "\\":
            continue
        if char in _APOSTROPHES and 0 < index < len(text) - 1 and (
            text[index - 1].isalnum() and text[index + 1].isalnum()
        ):
            continue
        marks.append((index, char))
    closers: dict[str, list[int]] = {}
    for index, char in marks:
        closers.setdefault(char, []).append(index)
    changes: list[int] = []
    closed = -1
    for index, char in marks:
        if index <= closed or char not in _PAIRS:
            continue
        if char in _APOSTROPHES and index and text[index - 1].isalnum():
            continue
        candidates = closers.get(_PAIRS[char], [])
        found = bisect_right(candidates, index)
        if found < len(candidates):
            closed = candidates[found]
            changes.extend((index, closed))
    return tuple(changes)


def _word_is_name(word: str, *, first: bool) -> bool:
    """Whether a word can be part of a name: letters, no digit, capitalised or uncased (CJK).

    A lowercase word is a name word only as a particle after the first word, and
    a word of `SENTENCE_WORDS` never is.
    """
    letters = [char for char in word if char.isalpha()]
    if not letters or any(char.isdigit() for char in word) or word.casefold() in SENTENCE_WORDS:
        return False
    head = letters[0]
    if head.isupper() or not head.islower():
        return True
    return not first and word.casefold() in NAME_PARTICLES


def _names_a_speaker(name: str, speaker: str | None, *, framed: bool) -> bool:
    """Whether a line-start ``name:`` is a label: a name, or the confirmed metadata speaker."""
    if not is_name(name):
        return False
    if name.startswith("#"):
        return framed and re.fullmatch(r"#[0-9]+", name) is not None
    norm = name_norm(name)
    if speaker is not None and norm == name_norm(speaker):
        return True
    words = name.split()
    numbered = re.fullmatch(r"#?[0-9]+", words[-1]) is not None
    if numbered and not framed:
        return False
    name_words = words[:-1] if numbered else words
    stem = name_norm(" ".join(name_words))
    if (
        len(words) > LABEL_MAX_WORDS or bound_length(token_seq(name)) > LABEL_MAX_TOKENS
        or stem in NON_NAME_PREFIXES or name_norm(name_words[-1]) in HEADING_NOUNS
        or (numbered and stem in _NUMBERED_NON_NAMES)
        or not CJK_FUNCTION_CHARACTERS.isdisjoint(name)
    ):
        return False
    return norm in GENERIC_USER_FORMS or all(
        _word_is_name(word, first=index == 0) for index, word in enumerate(name_words)
    )


def scan_labels(
    text: str, speaker: str | None = None, role: str | None = None,
) -> tuple[LabelMatch, ...]:
    """Find the line-start speaker labels of one text.

    Args:
        text (str): The text, read from its first character; a line starts at
            offset 0 and after each ``\\n``.
        speaker (str | None): The message's metadata speaker. A label with its
            name is accepted even when it does not look like a name.
        role (str | None): The message's wire role; an envelope label that
            restates it is not a label.

    Returns:
        tuple[LabelMatch, ...]: The labels in text order.
    """
    quotes = _quote_changes(text)
    starts = (0, *(match.end() for match in re.finditer("\n", text)))
    found: list[LabelMatch] = []
    for start in starts:
        if bisect_left(quotes, start) % 2 == 1:
            continue
        head = envelope_end(text, start)
        match = _LABEL.match(text, head)
        if match is None:
            continue
        end = match.end()
        if match["colon"] == ":" and end < len(text) and not text[end].isspace():
            continue
        name = match["name"]
        if head > start and name_norm(name) in ROLE_FORMS.get(role or "", frozenset()):
            continue
        framed = head > start
        if _names_a_speaker(name, speaker, framed=framed):
            found.append(LabelMatch(start if framed else match.start("name"), end, name, framed))
    return tuple(found)


def label_state(name: str, speaker: str | None, *, framed: bool = False) -> LabelState:
    """Settle one label without a model call (B.3a, X-Q3).

    Args:
        name (str): The label's name as written.
        speaker (str | None): The message's metadata speaker.
        framed (bool): An envelope header opens the label's line, which puts
            the turn's speaker in the label's slot.

    Returns:
        LabelState: ``speaker`` when the name is the metadata speaker, or on a
            message without one when it is a generic user form or framed by an
            envelope header; else ``ambiguous``.
    """
    norm = name_norm(name)
    if speaker is not None:
        return "speaker" if norm == name_norm(speaker) else "ambiguous"
    return "speaker" if framed or norm in GENERIC_USER_FORMS else "ambiguous"


def locate_labels(
    text: str, speaker: str | None = None, role: str | None = None,
) -> tuple[TurnLabel, ...]:
    """Locate and settle the labels of a wholly readable message text.

    Args:
        text (str): The whole message text, from its first byte.
        speaker (str | None): The message's metadata speaker.
        role (str | None): The message's wire role.

    Returns:
        tuple[TurnLabel, ...]: The turn labels in message byte coordinates.
    """
    labels: list[TurnLabel] = []
    char = byte = 0
    for match in scan_labels(text, speaker, role):
        byte += len(text[char:match.start].encode("utf-8"))
        char = match.start
        end = byte + len(text[match.start:match.end].encode("utf-8"))
        state = label_state(match.name, speaker, framed=match.framed)
        labels.append(TurnLabel(byte, end, match.name, state, match.framed))
    return tuple(labels)


@dataclass(frozen=True, slots=True)
class LocatedLabel:
    """One label the locator found in a message's rendered text, before any model decision.

    Attributes:
        char_start (int): First rendered character of the label: its envelope
            header, if any, else its name.
        char_end (int): Rendered character after the colon.
        byte_start (int): First message byte of the label (header or name).
        byte_end (int): Message byte after the colon.
        name (str): The name as written.
        framed (bool): An envelope header opens the label's line.
        state (LabelState): The locator's settlement: ``speaker`` when the
            metadata confirms the name, or on a message without metadata when
            it is a generic user form or framed by an envelope header;
            ``ambiguous`` otherwise, which only a model decision (B.3a) can
            settle further.
    """

    char_start: int
    char_end: int
    byte_start: int
    byte_end: int
    name: str
    state: LabelState
    framed: bool = False

    @property
    def span(self) -> tuple[int, int]:
        """The label's message byte range, its identity for a durable decision."""
        return self.byte_start, self.byte_end


@dataclass(frozen=True, slots=True)
class LocatedMessage:
    """One message rendered through ``render_masked`` with the labels located in it.

    Attributes:
        key (MessageKey): The message.
        kind (EvidenceKind): The message's Evidence kind; it never names a speaker.
        speaker (str | None): The message's metadata speaker.
        masked (MaskedText): The rendered text; withheld and unreadable bytes
            are markers.
        labels (tuple[LocatedLabel, ...]): The located labels in text order.
    """

    key: MessageKey
    kind: EvidenceKind
    speaker: str | None
    masked: MaskedText
    labels: tuple[LocatedLabel, ...]

    @property
    def undecided(self) -> tuple[LocatedLabel, ...]:
        """The labels the locator leaves ``ambiguous``: the only ones a model may settle."""
        return tuple(label for label in self.labels if label.state == "ambiguous")


type LabelDecisions = Mapping[MessageKey, Mapping[tuple[int, int], SourceLabelState]]
"""Durable model decisions (B.3a) by message and label byte range."""


def locate_message(parts: Sequence[EvidenceRecord], read: Readability) -> LocatedMessage | None:
    """Render one message through ``render_masked`` and locate its labels (B.3a, D.2).

    Args:
        parts (Sequence[EvidenceRecord]): Every part of the message, in any
            state; erased, expired and withheld bytes are never read.
        read (Readability): The read the labels are located under.

    Returns:
        LocatedMessage | None: The rendered message and its labels in message
            byte coordinates, or None when no part is placed. A label with an
            unreadable byte is not a label, and neither is one whose line start
            is unreadable: the message's first byte, or a readable newline with
            readable text up to the name.
    """
    readable = [part for part in parts if part.state != "erased"]
    if not readable:
        return None
    head = min(readable, key=lambda part: part.part_no)
    speaker = head.speaker_entity
    masked = read.render_masked(parts)
    segments = masked.segments
    opens_message = bool(segments) and segments[0].msg_byte_start == 0 and (
        segments[0].rendered_char_start == 0
    )
    labels: list[LocatedLabel] = []
    for match in scan_labels(masked.text, speaker, head.role):
        line = masked.text.rfind("\n", 0, match.start) + 1
        if line == 0 and not opens_message:
            continue
        indent = masked.spans(line, match.start) if line < match.start else ()
        spans = masked.spans(match.start, match.end)
        if indent is None or spans is None:
            continue
        labels.append(LocatedLabel(
            match.start, match.end, spans[0].byte_start, spans[-1].byte_end, match.name,
            label_state(match.name, speaker, framed=match.framed), match.framed,
        ))
    return LocatedMessage(message_key(head), head.kind, speaker, masked, tuple(labels))


def decided_turns(
    located: LocatedMessage, decisions: Mapping[tuple[int, int], SourceLabelState] | None = None,
) -> tuple[TurnLabel, ...]:
    """Settle a located message's turn labels from the locator and durable decisions (B.3a).

    A locator ``speaker`` label stands. A locator ``ambiguous`` label takes the
    durable model decision of its exact byte range: ``speaker`` opens that
    speaker's turn, ``content`` opens no turn, ``ambiguous`` stays a barrier; a
    label without a decision stays ``ambiguous``. No provider is called.

    Args:
        located (LocatedMessage): The message.
        decisions (Mapping[tuple[int, int], SourceLabelState] | None): Its
            durable decisions by label byte range; None when it has none.

    Returns:
        tuple[TurnLabel, ...]: The turn labels in message byte order.
    """
    turns: list[TurnLabel] = []
    for label in located.labels:
        state: SourceLabelState = label.state
        if state == "ambiguous" and decisions is not None:
            state = decisions.get(label.span, "ambiguous")
        if state == "content":
            continue
        turns.append(TurnLabel(label.byte_start, label.byte_end, label.name, state, label.framed))
    return tuple(turns)


def message_labels(
    parts: Sequence[EvidenceRecord], read: Readability,
    decisions: Mapping[tuple[int, int], SourceLabelState] | None = None,
) -> tuple[TurnLabel, ...]:
    """Locate and settle the turn labels of one message in its readable text only (B.3a, D.2).

    Args:
        parts (Sequence[EvidenceRecord]): Every part of the message, in any
            state; erased, expired and withheld bytes are never read.
        read (Readability): The read the labels are located under.
        decisions (Mapping[tuple[int, int], SourceLabelState] | None): The
            message's durable model decisions by label byte range, if any.

    Returns:
        tuple[TurnLabel, ...]: The turn labels in message byte coordinates.
    """
    located = locate_message(parts, read)
    return () if located is None else decided_turns(located, decisions)


def label_view(
    parts: Iterable[EvidenceRecord], read: Readability, decisions: LabelDecisions | None = None,
) -> MessageLabels:
    """Build the frozen label view of the messages an owner renders (B.3a, B.4.4).

    This is the one deterministic view builder: the locator plus the durable
    model decisions, under the caller's readability; it never calls a provider,
    and a located label without a decision stays ``ambiguous``.

    Args:
        parts (Iterable[EvidenceRecord]): Every part of every message to label,
            such as the prefix and the previous batch, in any order and state.
        read (Readability): The read the labels are located under.
        decisions (LabelDecisions | None): Durable model decisions by message
            and label byte range; None when there are none.

    Returns:
        MessageLabels: The turn labels of every message that has any.
    """
    messages: dict[MessageKey, list[EvidenceRecord]] = {}
    for part in parts:
        messages.setdefault(message_key(part), []).append(part)
    return MessageLabels({
        key: labels for key, members in messages.items()
        if (labels := message_labels(
            members, read, None if decisions is None else decisions.get(key),
        ))
    })
