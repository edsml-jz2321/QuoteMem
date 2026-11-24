"""A name the user asked to forget, as a later Add meets it again (``agent`` mode).

After "请忘掉我的猫的名字" the user still talks about the cat by the name the forget took:
"我昨天带bob去看兽医了". The extraction writes "The user took the user's cat bob to the vet
yesterday." and often "The user's cat is named bob." beside it. The second restates the forgotten
fact and is refused (settlement's tombstone rule, the judge, the commit fence); its quote is
withheld. The first is a new fact of its own: it must not be refused with it, and it must not
show the name either.

`forgotten_names` reads the names a forget took from the scope's tombstones: a non-user entity's
naming or tie statement ("The user's cat is named bob.", "Priya is the user's manager.") whose
name exactly one known entity bears. A name another entity or the user also bears, a single Han
character, and an alias ("Ben is also known as Benjamin") are none: those memories go on as before.

For one such name, `occurrences` finds it in a text in each of its forms (`name_forms`; a Latin
name as whole words, a capitalized one in Latin text as written or capitalized; a Han name as
written), `reword` writes a statement without it, and `cut` takes its bytes out of a quote span. A
statement that still `shows` the name in a form those rules produce is refused; a transliteration
they cannot produce ("鲍勃" for "bob") is left to the review judge, which reads each withheld
occurrence as whom it names (``statement_settlement.name_placeholders``). Settlement uses them
before review (``ingestion/statement_settlement._without_forgotten_names``); the commit withholds
the name's bytes that a written head's quote was cut around (``store/component_commit``). Nothing
here reads a withheld byte: the name comes from a tombstone, which only Add reads.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Final

from quotemem.core.entities import USER, QuoteSpanV1
from quotemem.core.forget_match import name_forms, tied_name
from quotemem.core.forget_match import ties as tie_words
from quotemem.core.name_forms import name_located
from quotemem.core.privacy import naming_value
from quotemem.core.recall_agent import sentences
from quotemem.core.statement import canonical_statement, own_token, token_seq

_HAN: Final = re.compile(r"[㐀-䶿一-鿿豈-﫿]")
_SUBJECT: Final = (
    re.compile(r"(?P<subject>(?:An?|The) .+?) is (?:also )?(?:named|called) .+"),
    re.compile(r"(?P<subject>(?:An?|The) .+?)['’]s name is .+"),
    re.compile(r"The name of (?P<subject>(?:an?|the) .+?) is .+"),
)
"""The subject of a statement that only names someone or something (`naming_value`): "The
user's cat" of "The user's cat is named bob.", "A person" of "A person is named Max."."""
_ALIAS: Final = re.compile(r"\b(?:also known as|known as|nicknamed|referred to as)\b")
NAME_LINK: Final = re.compile(
    r"\b(?:called|call|calls|named|name|names|known\s+as|goes\s+by)\b|叫|名字|取名|起名",
    re.IGNORECASE)
"""A word giving someone a name: the user's words teach a name again only with one ("我的猫
叫bob"), never by mentioning it ("我昨天带bob去看兽医了")."""
NEUTRAL: Final = "someone"
"""What a statement says in place of the name when nothing ties it to the forgotten one."""
_ANIMALS: Final = tie_words("pet")
"""Every animal tie (`forget_match.TIES`): "pet" and "宠物" cover them all."""
_ANIMAL_WORDS: Final = frozenset({"animal", "animals", "creature", "kitty", "动物"})
"""Words for an animal that no tie word covers."""
_PERSON_WORDS: Final = frozenset({"person", "people", "human", "man", "woman", "guy", "lady",
                                  "人"})
"""Words, and entity types, for a person that no tie word covers."""


@dataclass(frozen=True, slots=True)
class ForgottenName:
    """A name a forget took from exactly one known entity.

    Attributes:
        entity_id: The entity that bore it.
        name: The name as the forgotten statement wrote it.
        tie: How a statement names that entity without the name ("the user's cat",
            "the user's manager", "a person"), in no form or word of it (`bare`).
        tombstone_id: The tombstone that forgot it.
    """

    entity_id: str
    name: str
    tie: str
    tombstone_id: str

    @property
    def han(self) -> bool:
        """Whether the name holds a Han character: it may be an ordinary word too ("可乐")."""
        return _HAN.search(self.name) is not None


def _says(word: str, text: str) -> bool:
    """Whether ``text`` holds ``word``: a Han word anywhere, a Latin one as a whole word."""
    if _HAN.search(word):
        return word in text
    return re.search(rf"(?<![A-Za-z]){re.escape(word)}(?![A-Za-z])", text,
                     re.IGNORECASE) is not None


@dataclass(frozen=True, slots=True)
class Kind:
    """What sort of one bore a forgotten name, as a later text can show it.

    Attributes:
        ties: The tie words' ties (`forget_match.ties`) of its tie and type ({"cat"}).
        head: The last word of its tie ("cat" of "the user's cat", "person" of "a person").
        animal: It is an animal.
        person: It is a person.
    """

    ties: frozenset[str]
    head: str
    animal: bool
    person: bool

    def shown(self, text: str) -> bool:
        """Whether ``text`` names its sort: a word of its ties ("cat", "猫", "pet", "宠物" for
        a cat), a word for any animal when it is one ("animal", "动物"), or its head word."""
        return bool(tie_words(text) & self.ties) or (
            self.animal and any(_says(word, text) for word in _ANIMAL_WORDS)) or bool(
            self.head and _says(self.head, text))


def kind_of(tie: str, entity_type: str = "") -> Kind:
    """The `Kind` of the one a tie or descriptor names, its entity type beside it: an animal
    when its tie words cover only animals or its type or head word is one ("pet", "animal"),
    a person when they cover only people or its type or head word is one ("person"), and
    neither when it cannot tell ("an individual", or both)."""
    words = [] if tie == NEUTRAL else tie.split()
    head = words[-1].casefold().strip(".,") if words else ""
    kind = entity_type.casefold().strip()
    found = tie_words(f"{tie} {kind}")
    animal = bool(found) and found <= _ANIMALS or head in _ANIMAL_WORDS or (
        kind in _ANIMAL_WORDS)
    person = bool(found) and not found & _ANIMALS or head in _PERSON_WORDS or (
        kind in _PERSON_WORDS)
    return Kind(found, head, animal and not person, person and not animal)


def other_kind(bearer: Kind, entity_type: str, descriptor: str) -> bool:
    """Whether an entity is clearly not the one that bore a forgotten name: neither its
    descriptor nor its type names the bearer's sort, and one of them is a person and the
    other an animal ("the user's coworker", a person, beside a cat). When it cannot tell,
    it is the bearer."""
    if bearer.shown(descriptor) or bearer.shown(entity_type):
        return False
    own = kind_of(descriptor, entity_type)
    return (bearer.animal and own.person) or (bearer.person and own.animal)


def tie_of(statement: str) -> tuple[str, str] | None:
    """The name a naming or tie statement gives and how it names its subject without it:
    ("bob", "the user's cat") of "The user's cat is named bob.", ("Priya", "the user's
    manager") of "Priya is the user's manager.", ("Max", "a person") of "A person is named
    Max."; a subject that shows the name too gives its noun alone ("the user's daughter" of
    "The user's daughter Lily is named Lily Chen.", `bare`), or `NEUTRAL` when it has none.
    None for any other statement and for an alias, whose subject already has a name."""
    text = statement.strip()
    if _ALIAS.search(text) is not None:
        return None
    tied = tied_name(text)
    if tied is not None:
        return tied[1], bare(f"the user's {tied[0]}", tied[1]) or NEUTRAL
    name = naming_value(text)
    if name is None:
        return None
    for pattern in _SUBJECT:
        match = pattern.fullmatch(text)
        if match is not None:
            subject = match["subject"]
            return name, bare(subject[:1].lower() + subject[1:], name) or NEUTRAL
    return None


def names_of(statement: str, naming: Iterable[str] = ()) -> frozenset[str]:
    """The names a statement gives someone (folded): its `tie_of` name and its naming
    annotations'."""
    found = {name.casefold() for name in naming}
    tied = tie_of(statement)
    value = tied[0] if tied is not None else naming_value(statement.strip())
    if value is not None:
        found.add(value.casefold())
    return frozenset(found)


def forgotten_names(
    tombstones: Iterable[tuple[str, str, str]], bearers: Callable[[str], Iterable[str]],
) -> tuple[ForgottenName, ...]:
    """The names the scope's forgets took, each from exactly one known entity.

    Args:
        tombstones: Each tombstone's ``(object_id, entity_id, statement)``, in ``object_id``
            order.
        bearers: The active entities a name (its ``name_norm``'s source text) is readable
            on now, the user included.

    Returns:
        One entry per name, by its first tombstone: a non-user entity's naming or tie
        statement (`tie_of`) whose name no other entity's tombstone gives and no other
        entity bears now, and that is more than one Han character.
    """
    found: dict[str, list[ForgottenName]] = {}
    for object_id, entity_id, statement in tombstones:
        tied = tie_of(statement)
        if entity_id == USER or tied is None:
            continue
        name, tie = tied
        tokens = token_seq(name)
        if not tokens or (len(tokens) == 1 and own_token(tokens[0])):
            continue
        found.setdefault(name.casefold(), []).append(
            ForgottenName(entity_id, name, tie, object_id))
    return tuple(
        entries[0] for _key, entries in sorted(found.items())
        if len({entry.entity_id for entry in entries}) == 1
        and set(bearers(entries[0].name)) <= {entries[0].entity_id})


def occurrences(name: str, text: str) -> list[tuple[int, int]]:
    """Where ``text`` shows ``name`` in any of its `name_forms` (character offsets, overlapping
    finds joined): a Han name as written; a Latin name as whole words, in any case in a text
    with Han characters or for a lowercase name, else as written or capitalized ("Pepper",
    never the "pepper" on eggs)."""
    joined: list[tuple[int, int]] = []
    for start, end in sorted(found for form in name_forms(name) for found in _found(form, text)):
        if joined and start < joined[-1][1]:
            joined[-1] = (joined[-1][0], max(joined[-1][1], end))
        else:
            joined.append((start, end))
    return joined


def _found(name: str, text: str) -> list[tuple[int, int]]:
    """`occurrences` of one form of a name."""
    if _HAN.search(name) is not None:
        return [(match.start(), match.end()) for match in re.finditer(re.escape(name), text)]
    body = r"\s+".join(re.escape(word) for word in name.split())
    if not body:
        return []
    if name == name.casefold() or _HAN.search(text) is not None:
        pattern = re.compile(rf"(?<![0-9A-Za-z]){body}(?![0-9A-Za-z])", re.IGNORECASE)
    else:
        spellings = {name, " ".join(word[:1].upper() + word[1:] for word in name.split())}
        alternatives = "|".join(r"\s+".join(re.escape(word) for word in form.split())
                                for form in sorted(spellings))
        pattern = re.compile(rf"(?<![0-9A-Za-z])(?:{alternatives})(?![0-9A-Za-z])")
    return [(match.start(), match.end()) for match in pattern.finditer(text)]


_LEAD: Final = r"(?:the user['\u2019]s|the(?! user\b)|an?|their|his|her|its|my|our|your)"
"""What opens a noun phrase that may name someone: an article or a possessive, never the
subject "the user" ("The user took cat Bob ..." holds no phrase "The user took cat")."""
_PHRASE: Final = rf"{_LEAD}(?: [A-Za-z-]+){{1,3}}"
"""A noun phrase that may name the one bearing a name beside it ("the user's new colleague")."""


def _names(name: ForgottenName) -> str:
    """A pattern matching the name in each of its `name_forms`, longest first, words joined by any
    blanks."""
    return "|".join(r"\s+".join(re.escape(word) for word in form.split())
                    for form in sorted(name_forms(name.name), key=len, reverse=True))


_NO_NOUN: Final = re.compile(rf"{_LEAD}?", re.IGNORECASE)
_PLAIN: Final = frozenset({"a", "an", "and", "of", "the"})
"""Words of a name that tell no one ("the" of "Winnie the Pooh")."""


def bare(phrase: str, name: str) -> str:
    """A noun phrase without the name: each of the name's `name_forms` and each word of one ("the
    user's daughter" of "the user's daughter Lily" and "Lily Chen", "the cat" of "the cat bob"),
    but a plain word or one for a sort ("the user's uncle" of "Uncle Bob" stays), in any case.

    Returns:
        The phrase; "" when no noun is left or a form or word of the name would be.
    """
    given = name_forms(name)
    parts = sorted({*given, *(
        word for form in given for word in re.split(r"[\s()（）·-]+", form)
        if word and word.casefold() not in _PLAIN and not tie_words(word))}, key=len, reverse=True)
    found = phrase
    for part in parts:
        body = r"\s+".join(re.escape(word) for word in part.split())
        found = re.sub(rf"(?:\s+(?:named|called))?\s*(?<![0-9A-Za-z]){body}(?![0-9A-Za-z])", "",
                       found, flags=re.IGNORECASE)
    found = " ".join(found.split()).strip(" ,")
    if _NO_NOUN.fullmatch(found) or any(name_located(part, [found]) for part in parts):
        return ""
    return found


def _naming_noun(phrase: str, name: ForgottenName) -> bool:
    """Whether a phrase's last word names someone or something's sort: a tie word ("cat",
    "colleague"), a word for an animal or a person, or the forgotten one's own tie word."""
    word = phrase.split()[-1].casefold()
    return bool(tie_words(word)) or word in _ANIMAL_WORDS or word in _PERSON_WORDS or (
        word == name.tie.split()[-1].casefold())


def _apposed(statement: str, name: ForgottenName) -> list[tuple[int, int, str]]:
    """Each occurrence of the name beside a noun phrase naming the one bearing it, as the
    character range to replace and what to put there: "the user's cat bob" and "the user's
    new colleague, Bob," keep the phrase, as does "bob, the user's cat," ."""
    names = _names(name)
    found: list[tuple[int, int, str]] = []
    for match in re.finditer(rf"({_PHRASE})\s*,?\s*(?:{names})(?![0-9A-Za-z])(?:\s*,(?=\s))?",
                             statement, re.IGNORECASE):
        if _naming_noun(match[1], name):
            found.append((match.start(), match.end(), match[1]))
    for match in re.finditer(rf"(?<![0-9A-Za-z])(?:{names})\s*,\s*({_PHRASE})\s*,",
                             statement, re.IGNORECASE):
        if _naming_noun(match[1], name) and not any(
                start < match.end() and match.start() < end for start, end, _ in found):
            found.append((match.start(), match.end(), match[1]))
    return sorted(found)


def shows(text: str, name: ForgottenName) -> bool:
    """Whether a text shows the forgotten name in a form the name rules produce: one of its
    `name_forms` ("Lin Xiaoyu" of "林晓雨（Lin Xiaoyu）") as written (`occurrences`) or inflected
    (``name_forms.name_located``: "Роуэна" for "Роуэн"). A transliteration or a translation
    they cannot produce ("鲍勃" for "bob") is none: the review judge reads it unsupported."""
    return bool(occurrences(name.name, text)) or any(
        name_located(form, [text]) for form in name_forms(name.name))


def ties(statement: str, name: ForgottenName) -> bool:
    """Whether a statement names the forgotten one by a noun phrase beside the name ("the
    user's cat bob", "the user's pet, Bob,", "the pet bob", "bob, the user's cat,")."""
    return bool(_apposed(statement, name))


def reword(statement: str, name: ForgottenName, tie: str, *,
           read_as_tie: bool = False) -> str | None:
    """``statement`` with ``name`` in no form: a noun phrase beside the name naming its sort
    keeps the phrase alone ("the user's cat bob" -> "the user's cat", "the user's new
    colleague, Bob," -> "the user's new colleague", "bob, the user's cat," -> "the user's
    cat"), "named Bob" goes ("an animal named Bob" -> "an animal"), and every other
    occurrence says ``tie`` ("The user took bob to the vet." -> "The user took the user's cat
    to the vet.").

    With ``read_as_tie`` the review reads the name as ``tie`` ("[the user's cat]",
    ``statement_settlement.name_placeholders``), so a noun phrase beside the name or named by
    it that lacks the tie's last word says the tie instead ("Bob, a pet," and "a pet named
    Bob" -> "the user's cat"): the statement keeps what the judge reads.

    Returns:
        The canonical statement; None when it would still show the name, would be empty, or
        would only name or tie someone (a naming statement is never a fact of its own).
    """
    if tie_of(statement) is not None or naming_value(statement.strip()) is not None:
        return None
    names = _names(name)
    head = tie.split()[-1].casefold()

    def kept(phrase: str) -> str:
        return tie if read_as_tie and head not in phrase.casefold().split() else phrase

    text = statement
    for start, end, phrase in reversed(_apposed(statement, name)):
        text = text[:start] + kept(phrase) + text[end:]
    if read_as_tie:
        for match in reversed(list(re.finditer(
                rf"(?<![A-Za-z])({_PHRASE})\s+(?:named|called)\s+(?:{names})(?![0-9A-Za-z])",
                text, re.IGNORECASE))):
            text = text[:match.start()] + kept(match[1]) + text[match.end():]
    text = re.sub(rf"\s+(?:named|called)\s+(?:{names})(?![0-9A-Za-z])", "", text,
                  flags=re.IGNORECASE)
    for start, end in reversed(occurrences(name.name, text)):
        text = text[:start] + tie + text[end:]
    if text[:1].islower():
        text = text[:1].upper() + text[1:]
    try:
        text = canonical_statement(text)
    except ValueError:
        return None
    if not text or shows(text, name) or tie_of(text) is not None or (
            naming_value(text) is not None):
        return None
    return text


def untie(statement: str, name: ForgottenName, kind: Kind) -> str | None:
    """``statement`` saying "someone" for the one only the forgotten name told it about: the
    name, a noun phrase beside it, and every noun phrase naming the bearer's sort ("the
    user's cat") give way to `NEUTRAL` ("The user took the user's cat bob to the vet
    yesterday." -> "The user took someone to the vet yesterday."). None as `reword`'s, and
    when a word for the bearer's sort is left."""
    text = statement
    for start, end, _phrase in reversed(_apposed(text, name)):
        text = text[:start] + NEUTRAL + text[end:]
    phrases: list[tuple[int, int]] = []
    for lead in re.finditer(rf"\b{_LEAD}\b", text, re.IGNORECASE):
        if phrases and lead.start() < phrases[-1][1]:
            continue
        following = re.match(r"(?: [A-Za-z-]+){1,3}", text[lead.end():])
        if following is None:
            continue
        end = lead.end()
        for word in following.group().split():
            end = text.index(word, end) + len(word)
            if kind.shown(word):
                phrases.append((lead.start(), end))
                break
    for start, end in reversed(phrases):
        text = text[:start] + NEUTRAL + text[end:]
    found = reword(text, name, NEUTRAL)
    return None if found is None or kind.shown(found) else found


def without_name(text: str, name: ForgottenName) -> str:
    """A description without the name ("the pet named bob" -> "the pet", `bare`), or the
    bearer's tie when nothing else is left."""
    found = text
    for start, end, phrase in reversed(_apposed(found, name)):
        found = found[:start] + phrase + found[end:]
    return bare(found, name.name) or name.tie


def cut(span: QuoteSpanV1, text: str, byte_start: int,
        found: Sequence[tuple[int, int]]) -> tuple[list[QuoteSpanV1], list[QuoteSpanV1]]:
    """A quote span without the name: its pieces around each occurrence it meets, blanks
    at their edges dropped, and the occurrences' own bytes inside it.

    Args:
        span: The quote span.
        text: Its Evidence part's text.
        byte_start: The part's first message byte.
        found: The name's `occurrences` in ``text`` (character offsets).

    Returns:
        ``(pieces, names)`` in message bytes; ``([span], [])`` when the span meets none.
    """
    data = text.encode("utf-8")
    cuts: list[tuple[int, int]] = []
    for start, end in found:
        first = byte_start + len(text[:start].encode("utf-8"))
        last = byte_start + len(text[:end].encode("utf-8"))
        if first < span.byte_end and span.byte_start < last:
            cuts.append((max(first, span.byte_start), min(last, span.byte_end)))
    if not cuts:
        return [span], []
    pieces: list[QuoteSpanV1] = []
    at = span.byte_start
    for first, last in [*cuts, (span.byte_end, span.byte_end)]:
        piece = data[at - byte_start:first - byte_start]
        lead = len(piece) - len(piece.lstrip())
        trail = len(piece.rstrip())
        if trail > lead:
            pieces.append(span.model_copy(update={"byte_start": at + lead,
                                                  "byte_end": at + trail}))
        at = last
    names = [span.model_copy(update={"byte_start": first, "byte_end": last})
             for first, last in cuts]
    return pieces, names


def touching(spans: Sequence[QuoteSpanV1], text: str, byte_start: int, evidence_id: str,
             found: Sequence[tuple[int, int]], kind: Kind) -> list[QuoteSpanV1]:
    """The occurrences of a forgotten name (character offsets in ``text``) that a head's
    quote was cut around: between two of its spans in the same part, or right before or
    after one in a sentence that names the bearer's sort (`Kind.shown`), with only blanks
    between. A quote that merely ends before the name ("My new coworker" of "My new coworker
    Max ...") cut nothing."""
    data = text.encode("utf-8")
    own = [span for span in spans if span.evidence_id == evidence_id]
    bounds = sentences(text)
    found_spans: list[QuoteSpanV1] = []
    for start, end in found:
        first = byte_start + len(text[:start].encode("utf-8"))
        last = byte_start + len(text[:end].encode("utf-8"))
        if any(first < span.byte_end and span.byte_start < last for span in own):
            continue  # The head quotes the name itself: no cut was made here.
        after = any(not data[last - byte_start:span.byte_start - byte_start].strip()
                    and last <= span.byte_start for span in own)
        before = any(not data[span.byte_end - byte_start:first - byte_start].strip()
                     and span.byte_end <= first for span in own)
        sentence = next((text[low:high] for low, high in bounds if low <= start < high), text)
        if (after and before) or ((after or before) and kind.shown(sentence)):
            found_spans.append(QuoteSpanV1(evidence_id=evidence_id, byte_start=first,
                                           byte_end=last))
    return found_spans
