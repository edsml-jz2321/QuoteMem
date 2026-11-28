"""A name a correction replaced, as another memory's quote still shows it (``agent`` mode).

"Sorry, I got that wrong earlier: my cat is called Max, not Bob." corrects "The user's cat is
named Bob." to "The user's cat is named Max, not Bob."; the old version stays in that
memory's history, since a correction is no forget. "The user has a cat." states no name, but
its quote may be "My cat is called Bob". The commit of such a correction requotes each memory
about the renamed one that states no name but quotes the old name (``store/component_commit``):
`unnamed` cuts the quote around the old name and the words giving it ("is called", "named",
"叫"), so the memory quotes "My cat", and only when what is left still holds every word the
statement claims (`supports`), or loses nothing but whom the old name stood for and shows that
again (`supports_cut`): "Kiwi is five years old." quotes "My parrot" (`tie_span`, from the same
message) and "is five years old."; "我的狗叫旺财，它特别喜欢去公园" quotes "我的狗" and "，它特别
喜欢去公园". Otherwise the memory keeps its quote.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Final

from quotemem.core.entities import QuoteSpanV1
from quotemem.core.forget_match import named_ties
from quotemem.core.forgotten_names import cut, occurrences
from quotemem.core.recall_agent import GENERIC, content_terms, word_stem

_LINK: Final = re.compile(
    r"(?:\s*,?\s*(?:(?:is|was|'s|’s)\s+)?(?:called|named)|\s*['’]s\s+name\s+(?:is|was)"
    r"|\s+name\s+(?:is|was)|\s*名字(?:叫|是)|\s*名叫|\s*叫做|\s*叫)\s*$",
    re.IGNORECASE)
"""The words giving a name right before it ("is called", "named", "'s name is", "叫")."""
_WORD: Final = re.compile(r"[^\W_]", re.UNICODE)


def cuts(name: str, text: str) -> list[tuple[int, int]]:
    """Each occurrence of the name in ``text`` (`forgotten_names.occurrences`, character
    offsets) with the words giving it right before it: " is called Bob" of "My cat is
    called Bob"."""
    found: list[tuple[int, int]] = []
    for start, end in occurrences(name, text):
        link = _LINK.search(text[:start])
        found.append((start if link is None else link.start(), end))
    return found


def unnamed(span: QuoteSpanV1, text: str, byte_start: int,
            name: str) -> list[QuoteSpanV1] | None:
    """A quote span without the old name: its pieces around each of its `cuts`, pieces
    without a letter dropped; ``[span]`` when it meets none, None when nothing is left."""
    pieces, names = cut(span, text, byte_start, cuts(name, text))
    if not names:
        return [span]
    data = text.encode("utf-8")
    kept = [piece for piece in pieces if _WORD.search(
        data[piece.byte_start - byte_start:piece.byte_end - byte_start].decode("utf-8"))]
    return kept or None


def _held(quoted: Sequence[str]) -> set[str]:
    text = " ".join(quoted)
    return {word_stem(term) for term in content_terms(text)} | named_ties(text)


def supports(statement: str, quoted: Sequence[str]) -> bool:
    """Whether the quoted texts hold every word the statement claims: its content terms
    less the `GENERIC` ones, each held as a term of theirs or a tie a person or animal word
    of theirs names ("猫" names "cat")."""
    wanted = {word_stem(term) for term in content_terms(statement)} - GENERIC
    return bool(wanted) and wanted <= _held(quoted)


def supports_cut(statement: str, before: Sequence[str], after: Sequence[str]) -> bool:
    """Whether quotes cut around an old name (``after``) still support the statement the
    uncut ones (``before``) supported: the statement names whom it is about by a person or
    animal word ("The user's parrot ..."), ``after`` shows that word ("My parrot", "我的狗"),
    and every word of the statement ``after`` does not hold, ``before`` did not hold either
    (the English words of a statement quoting Chinese) or is that word."""
    ties = named_ties(statement)
    if not ties or not ties <= named_ties(" ".join(after)):
        return False
    wanted = {word_stem(term) for term in content_terms(statement)} - GENERIC
    return bool(wanted) and wanted - _held(after) <= (wanted - _held(before)) | {
        word_stem(tie) for tie in ties}


_TIE_PHRASE: Final = re.compile(
    r"\b(?:[Mm]y|[Oo]ur)(?:\s+[A-Za-z]+){1,2}\b|我们?的[\u4e00-\u9fff]{1,2}")


def tie_span(statement: str, text: str, name: str) -> tuple[int, int] | None:
    """The first phrase of ``text`` (character offsets) showing the person or animal word
    the statement names whom it is about by, and no form of the name: "My parrot" of "My
    parrot is called Kiwi. Kiwi is five years old." for "The user's parrot is five years
    old."; None when there is none."""
    ties = named_ties(statement)
    if not ties:
        return None
    for found in _TIE_PHRASE.finditer(text):
        start = found.start()
        for end in sorted({found.start() + len(found.group(0)[:size]) for size in range(
                1, len(found.group(0)) + 1)}):
            phrase = text[start:end]
            if ties <= named_ties(phrase) and not occurrences(name, phrase) and (
                    end == len(text) or not text[end].isalnum() or _HAN.match(text[end - 1])):
                return start, end
    return None


_HAN: Final = re.compile(r"[\u4e00-\u9fff]")
