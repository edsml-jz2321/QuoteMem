"""Explicit in-chat forget requests: the user's own sentences asking to forget a fact.

An Add whose frozen extraction policy names ``chat_forget`` carries out the forget
requests of its user messages once the Add has completed, through forget by statement
(FS-1, ``service/chat_forget.py``). This module only finds the requests and the
statement each one asks to forget; it reads nothing else and calls nothing.

A sentence of a user message is a request when, after an optional opener ("ok",
"also", "one more thing", ...) and an optional polite frame ("please", "kindly",
"can/could/would you", "I'd like you to", "I want/need you to"), it starts with:

- ``forget`` or ``forget about``, followed by what to forget; a bare "Forget about
  ..." with neither a polite frame nor the memory named is the idiom for "stop
  thinking about" ("Forget about my ex, she's not worth it"), never a request;
- a remembering verb that it negates or stops: "don't/do not/never remember",
  "stop remembering", "no longer remember";
- a verb that keeps or drops a record, with the memory it names: "don't keep, store,
  save, record or retain <it> in your memory", "remove, delete, erase, wipe, clear or
  drop <it> from your memory" (or records, notes, history).

What it asks to forget must be about the speaker or what the speaker told: its
first clause says I, me, my, mine or myself, or names something told ("the detail
about ...", "the fact that ...", "any information about ..."), or its sentence names
the memory; and it must be more than one word. A bare
"it", "this" or "that" after ``forget`` is a dismissal ("Forget it, let's move on"),
never a request; after a remembering or record verb, or as "what I said / told you /
mentioned / shared", it refers to what the user said just before: the earlier
sentences of the same message, else the previous user message of the Add. "Forget
that I asked" retracts a question and asks nothing.

Each form must begin its sentence, so a report ("She told me to forget about it"),
a statement about forgetting ("I keep forgetting my keys", "history books too often
forget"), a reminder ("Don't forget that I ...") and a question about forgetting
("Do you ever forget what I tell you?", "Did you forget my birthday?") never
qualify. A sentence ending in a question mark qualifies only through a polite
"can/could/would you" frame. A message may start with its speaker's label ("Mira:
Please forget that I ..."), as transcripts write it; the label is not part of the
sentence. Unless the sentence names the memory, the Add must be a conversation with
the memory: it holds an assistant message, or its user messages come from one speaker
(their ``speaker``, else their label). Between people, two speakers and no assistant
("Forget about my ex, she's not worth it"), the imperative is advice to the other
person, not an instruction to the memory.

Chinese requests follow the same rule with their own words: an optional ``请`` (and
``帮我``) before ``忘记``, ``忘掉`` or ``不要/别 记住/记得/记录``, or ``删除/删掉 ...
记忆/记录``, about the speaker (``我``).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Literal

from aml.core.forget_pick import MAX_FORGET_STATEMENT_CHARS
from aml.core.run_inputs import InputMessageV1
from aml.core.statement import token_seq

MAX_CHAT_FORGETS: Final = 16
"""The requests one Add carries out at most, in message order; later ones are left."""
_STOP: Final = frozenset(
    "a an the and or but of to in on at for with by from about as into over is are was were be "
    "been being am do does did have has had i me my mine myself you your yours we our it its "
    "that this these those please forget remember not no don't dont never any all some".split())

_SENTENCE = re.compile(r"[^.!?;\n。！？；]+[.!?;。！？；]*")
_OPENER = (r"(?:(?:ok(?:ay)?|also|and|so|oh|hey|hi|actually|now|well|lastly|finally|"
           r"one more thing|by the way|btw)\b[\s,:\-—–]*)*")
_POLITE = (r"(?P<polite>(?:please|kindly|(?:can|could|would) you(?: please)?|"
           r"i(?:'d| would) like (?:you )?to|i want you to|i need you to|"
           r"i(?:'m| am) asking you to)[\s,]+)?(?:please[\s,]+)?")
_MEMORY = r"(?:(?:your |the )?(?:memory|memories)|your (?:records?|notes|history))"
_FORGET = re.compile(
    rf"^\W*{_OPENER}{_POLITE}(?P<verb>forget(?: all)?(?: about)?)\b[\s,]*(?P<object>.*)$",
    re.IGNORECASE | re.DOTALL)
_REMEMBER = re.compile(
    rf"^\W*{_OPENER}{_POLITE}(?P<verb>(?:do not|don't|dont|never) remember|"
    rf"stop remembering|no longer remember)\b\s*(?P<object>.*)$",
    re.IGNORECASE | re.DOTALL)
_RECORD = re.compile(
    rf"^\W*{_OPENER}{_POLITE}(?P<verb>(?:(?:do not|don't|dont|never) "
    rf"(?:keep|store|save|record|retain))|stop (?:keeping|storing|saving|recording)|"
    rf"remove|delete|erase|wipe|clear|drop|purge)\s+(?P<object>.+?)\s+"
    rf"(?:in|into|to|from|out of) {_MEMORY}\b\W*(?P<rest>.*)$",
    re.IGNORECASE | re.DOTALL)
_ZH = re.compile(
    r"^\W*(?:请|麻烦|能不能|可以)?(?:你)?(?:帮我)?(?:(?P<verb>忘记|忘掉|(?:不要|别|不用)(?:再)?"
    r"(?:记住|记得|记录|保存))(?P<object>.*)|(?:删除|删掉|清除)(?P<deleted>.*?)(?:的)?"
    r"(?:记忆|记录))\W*$", re.DOTALL)
_LABEL = re.compile(r"^\s*([A-Z][\w'\u2019.-]*(?: [A-Z][\w'\u2019.-]*){0,2}):[ \t]+(?=\S)")
_ZH_EDGES = " \t，,：:。！？.!?；;、"
_SPEAKER = re.compile(r"\b(?:i|i'm|i've|i'd|i'll|me|my|mine|myself)\b", re.IGNORECASE)
_NAMES_MEMORY = re.compile(rf"\b(?:from|in|into|out of) {_MEMORY}\b", re.IGNORECASE)
_TRAILING_MEMORY = re.compile(rf"\s*,?\s*(?:from|out of|in) {_MEMORY}\W*$", re.IGNORECASE)
_TRAILING = re.compile(r"(?:[\s,]+(?:please|for me|thanks|thank you))*\W*$", re.IGNORECASE)
_DISMISSAL = re.compile(r"^(?:about )?(?:it|this|that|them|everything)\b(?:\W*$|\s*[,—–-].*$)",
                        re.IGNORECASE | re.DOTALL)
_RETRACTION = re.compile(r"^(?:that |what )?i (?:just |ever )?asked\b", re.IGNORECASE)
_REFERENCE = re.compile(
    r"^(?:(?:it|this|that|those|these)(?: information| detail| details)?|"
    r"(?:what|everything|all|anything|whatever) (?:that )?i (?:just |have |'ve |had )?"
    r"(?:said|told you|mentioned|shared|wrote|typed)(?: (?:earlier|before|just now))?)\W*$",
    re.IGNORECASE)
_TOLD = re.compile(
    r"^(?:the|that|this|any|all|those|these)\s+(?:detail|details|fact|facts|information|info|"
    r"preference|preferences|memory|memories|thing|things|part|stuff|bit)\s+"
    r"(?:about|that|of|regarding|concerning|on)\b", re.IGNORECASE)
_CLAUSE = re.compile(r"\s*[,;:—–]\s*|\s+-\s+")
_KNOWN = re.compile(r"^(?:that\s+)?you(?:'ve| have)?\s+(?:remember(?:ed)?|know|knew|noted|learned|"
                    r"stored|saved)\s+(?:that\s+|about\s+)?", re.IGNORECASE)
_SUBJECT_THAT = re.compile(r"^that\s+(?=(?:i|i'm|i've|i'd|my|me|we|our|it's|it|there)\b)",
                           re.IGNORECASE)


@dataclass(frozen=True)
class ForgetRequest:
    """One request found in an Add: where it is and the statement it asks to forget.

    Attributes:
        message (int): The index of its user message in the Add.
        statement (str): What to forget, in the user's words (at most
            ``MAX_FORGET_STATEMENT_CHARS`` characters). A named speaker's request is
            written as its transcript writes it ("Leah: I do swimming"), so the judge
            reads whose fact the first person is, as the memories name that speaker.
    """

    message: int
    statement: str


def _fold(text: str) -> str:
    """The text with curly apostrophes straight, character for character."""
    return text.replace("’", "'").replace("ʼ", "'")


def _bounded(text: str) -> str:
    text = " ".join(text.split())
    if len(text) <= MAX_FORGET_STATEMENT_CHARS:
        return text
    cut = text[:MAX_FORGET_STATEMENT_CHARS]
    return cut[:cut.rfind(" ")] if " " in cut else cut


def _object(raw: str) -> str:
    """What follows the verb, without the memory it names and trailing courtesy."""
    value = _TRAILING.sub("", raw.strip())
    value = _KNOWN.sub("", _TRAILING.sub("", _TRAILING_MEMORY.sub("", value)).strip())
    return _SUBJECT_THAT.sub("", value).strip()


def _speaker(message: InputMessageV1) -> str:
    """Who wrote a user message: its ``speaker``, else its leading label, else nobody named."""
    if message.speaker is not None:
        return message.speaker
    label = _LABEL.match(message.content)
    return "" if label is None else label.group(1)


def _english(sentence: str, *, assistant: bool) -> tuple[str, bool] | None:
    """The object of an English request sentence and whether it refers back, else None.

    ``assistant`` says whether the Add is a conversation with the memory.
    """
    text = _fold(sentence).strip()
    names_memory = _NAMES_MEMORY.search(text) is not None
    match = _FORGET.match(text)
    verb: Literal["forget", "remember", "record"]
    if match is not None:
        verb = "forget"
    elif (match := _REMEMBER.match(text)) is not None:
        verb = "remember"
    elif (match := _RECORD.match(text)) is not None:
        verb = "record"
    else:
        return None
    if text.endswith("?") and not (match.group("polite") or "").strip().lower().startswith(
            ("can", "could", "would")):
        return None
    if not (assistant or names_memory):
        return None
    polite = bool((match.group("polite") or "").strip())
    if verb == "forget" and match.group("verb").lower().endswith("about") and not (
            polite or names_memory):
        return None
    target = _object(match.group("object"))
    if _RETRACTION.match(target):
        return None
    if verb == "forget" and not names_memory and _DISMISSAL.match(target):
        return None
    if _REFERENCE.match(target):
        return "", True
    clause = _CLAUSE.split(target, maxsplit=1)[0]
    if len(target.split()) < 2 or not (
            names_memory or _SPEAKER.search(clause) or _TOLD.match(clause)):
        return None
    return target, False


def _chinese(sentence: str, *, assistant: bool) -> str | None:
    """The object of a Chinese request sentence, else None."""
    match = _ZH.match(sentence.strip())
    if match is None or not (assistant or match.group("deleted") is not None):
        return None
    target = (match.group("object") or match.group("deleted") or "").strip(_ZH_EDGES)
    target = re.sub(r"^(?:关于|有关)", "", target).strip(_ZH_EDGES)
    if "我" not in target or len(target) < 3:
        return None
    return target


def forget_requests(messages: Sequence[InputMessageV1]) -> tuple[ForgetRequest, ...]:
    """Find the explicit forget requests of an Add's user messages (module docstring).

    Args:
        messages: The Add's messages in order.

    Returns:
        tuple[ForgetRequest, ...]: At most ``MAX_CHAT_FORGETS`` requests in message
            order, each statement once.

    Examples:
        ```python
        from aml.core.run_inputs import InputMessageV1

        found = forget_requests([
            InputMessageV1(role="user", content="Please forget that I like jazz."),
            InputMessageV1(role="assistant", content="Done."),
        ])
        assert [request.statement for request in found] == ["I like jazz"]
        ```
    """
    users = [message for message in messages if message.role == "user"]
    assistant = any(message.role == "assistant" for message in messages) or len(
        {_speaker(message) for message in users}) <= 1
    found: list[ForgetRequest] = []
    seen: set[str] = set()
    previous = ""
    for index, message in enumerate(messages):
        if message.role != "user":
            continue
        before = ""
        speaker = _speaker(message)
        label = _LABEL.match(message.content)
        content = message.content if label is None else message.content[label.end():]
        for sentence in _SENTENCE.finditer(content):
            text = sentence.group()
            english = _english(text, assistant=assistant)
            statement: str | None
            if english is None:
                statement = _chinese(text, assistant=assistant)
            elif english[1]:
                statement = before.strip() or previous.strip() or None
            else:
                statement = english[0]
            if statement:
                statement = _bounded(f"{speaker}: {statement}" if speaker else statement)
                if statement not in seen and len(found) < MAX_CHAT_FORGETS:
                    seen.add(statement)
                    found.append(ForgetRequest(index, statement))
            before += text
        previous = content
    return tuple(found)


def is_request(text: str) -> bool:
    """Whether a text, read as one user message, makes a forget request: it records a request.

    An in-chat forget never forgets or marks the record of its own request: the memory
    that only quotes it ("The user asked to forget that ...") and the request sentence stay.
    """
    return bool(forget_requests([InputMessageV1(role="user", content=text)]))


def content_words(text: str) -> frozenset[str]:
    """The tokens of a text that carry its content (`token_seq` less function words)."""
    return frozenset(token for token in token_seq(text) if token not in _STOP and len(token) > 1)


def sentence_ranges(text: str) -> list[tuple[int, int]]:
    """The character ranges of a text's sentences, without surrounding white space."""
    ranges: list[tuple[int, int]] = []
    for sentence in _SENTENCE.finditer(text):
        piece = sentence.group()
        start = sentence.start() + len(piece) - len(piece.lstrip())
        end = sentence.end() - (len(piece) - len(piece.rstrip()))
        if start < end:
            ranges.append((start, end))
    return ranges
