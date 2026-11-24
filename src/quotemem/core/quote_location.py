"""Quote location: map an extracted quote to the message bytes it copies.

The extractor never counts bytes. It copies a quote from one rendered source,
and the program finds that quote in the source's rendered text and maps it
through the render's segment map to original-message spans. A quote is found
exactly after NFC normalization of both texts, else after collapsing every
whitespace run to one space on both sides, else, for a quote whose first character
is a cased letter, with that character in the other case ("My dad is Muslim" for
"... and my dad is Muslim."): the span is the source's own text. The leftmost occurrence that
overlaps no ``[withheld]`` marker wins; a quote every occurrence of which
overlaps one is ``quote_overlaps_withheld``, and a quote that occurs nowhere is
``quote_not_found``. The result is one `QuoteSpanV1` per Evidence part the
quote covers, on character boundaries.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal

from quotemem.core.entities import QuoteSpanV1
from quotemem.core.readability import MaskedText

QuoteFault = Literal["quote_not_found", "quote_overlaps_withheld"]


@dataclass(frozen=True, slots=True)
class LocatedQuote:
    """Where one quote lies in its rendered source.

    Attributes:
        char_start (int): First rendered character of the quote, inclusive.
        char_end (int): Rendered character after the quote.
        spans (tuple[QuoteSpanV1, ...]): One span per covered Evidence part, in
            rendered order, in original-message byte coordinates.
    """

    char_start: int
    char_end: int
    spans: tuple[QuoteSpanV1, ...]


def _independent(left: str, right: str) -> bool:
    """Tell whether NFC normalizes ``left + right`` piecewise."""
    return unicodedata.normalize("NFC", left + right) == (
        unicodedata.normalize("NFC", left) + unicodedata.normalize("NFC", right)
    )


@dataclass(frozen=True, slots=True)
class _NormalText:
    """NFC text with the rendered position of every NFC position it can map back.

    The rendered text is cut into the shortest pieces that NFC normalizes
    independently: each piece starts at a character of canonical combining
    class 0 that does not compose with what precedes it. ``origin[i]`` is the
    rendered position of NFC position ``i`` when ``i`` starts a piece (or ends
    the text), else None: a position inside a composed piece has no exact
    rendered counterpart.
    """

    text: str
    origin: tuple[int | None, ...]

    @staticmethod
    def of(rendered: str) -> _NormalText:
        if unicodedata.is_normalized("NFC", rendered):
            return _NormalText(rendered, tuple(range(len(rendered) + 1)))
        cuts = [0]
        for index in range(1, len(rendered)):
            if unicodedata.combining(rendered[index]) == 0 and _independent(
                rendered[cuts[-1]:index], rendered[index],
            ):
                cuts.append(index)
        cuts.append(len(rendered))
        text: list[str] = []
        origin: list[int | None] = []
        for start, end in zip(cuts, cuts[1:], strict=False):
            piece = unicodedata.normalize("NFC", rendered[start:end])
            text.append(piece)
            origin.extend([start, *([None] * (len(piece) - 1))])
        origin.append(len(rendered))
        return _NormalText("".join(text), tuple(origin))


def _collapsed(text: str) -> tuple[str, tuple[int, ...]]:
    """Collapse every whitespace run to one space; map each kept character back."""
    kept: list[str] = []
    origin: list[int] = []
    for index, char in enumerate(text):
        if char.isspace():
            if index and text[index - 1].isspace():
                continue
            char = " "
        kept.append(char)
        origin.append(index)
    return "".join(kept), tuple(origin)


def _occurrences(text: str, needle: str) -> Iterator[int]:
    start = text.find(needle)
    while start >= 0:
        yield start
        start = text.find(needle, start + 1)


def _first_letter_other_case(needle: str) -> str | None:
    """The needle with its first character's case swapped, if it is a cased letter."""
    first = needle[0]
    other = first.lower() if first.isupper() else first.upper() if first.islower() else first
    if other == first or len(other) != 1:
        return None
    return other + needle[1:]


def locate_quote(quote: str, source: MaskedText) -> LocatedQuote | QuoteFault:
    """Locate a quote in one rendered source.

    The quote is NFC-normalized and stripped first. Occurrences are tried
    leftmost first, the exact ones before the whitespace-collapsed ones, and only
    then the same two with the quote's first letter in the other case; an
    occurrence whose edges fall inside a piece that NFC composed is skipped.
    The located range is trimmed of edge whitespace before it is mapped.

    Args:
        quote (str): The quote as the extractor copied it; it must contain a
            character other than whitespace (admission refuses a blank quote as
            ``quote_blank`` before locating it).
        source (MaskedText): The rendered source text and its segment map, as
            ``Readability.render_masked`` produced them.

    Returns:
        LocatedQuote | QuoteFault: The rendered range and its spans, or
            ``quote_overlaps_withheld`` when every occurrence overlaps a
            ``[withheld]`` marker, or ``quote_not_found``.

    Raises:
        ValueError: If the quote is blank.
    """
    needle = unicodedata.normalize("NFC", quote).strip()
    if not needle:
        raise ValueError("A quote to locate must contain a character other than whitespace")
    normal = _NormalText.of(source.text)
    found = False
    collapsed_text, collapsed_origin = _collapsed(normal.text)

    def ranges(text: str) -> Iterator[tuple[int, int]]:
        for start in _occurrences(normal.text, text):
            yield start, start + len(text)
        collapsed_needle, _ = _collapsed(text)
        for start in _occurrences(collapsed_text, collapsed_needle):
            yield (collapsed_origin[start],
                   collapsed_origin[start + len(collapsed_needle) - 1] + 1)

    other = _first_letter_other_case(needle)
    tried = (*ranges(needle), *(ranges(other) if other is not None else ()))
    for normal_start, normal_end in tried:
        start, end = normal.origin[normal_start], normal.origin[normal_end]
        if start is None or end is None:
            continue
        text = source.text
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if start == end:
            continue
        found = True
        spans = source.spans(start, end)
        if spans is not None:
            return LocatedQuote(start, end, spans)
    return "quote_overlaps_withheld" if found else "quote_not_found"
