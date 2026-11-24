"""Locate issued lexical tokens in unchanged original UTF-8 source text.

Token indices only name exact byte intervals. They establish no semantic
sentence boundary, source ownership, scope, fact support or correction authority.
The complete original source remains a separate input to its actual owners.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from quotemem.core.errors import QuoteMemError

_LEXICAL = re.compile(r"\w+|[^\w\s]")


@dataclass(frozen=True, slots=True)
class SourceAssertionToken:
    """One exact raw token and its half-open UTF-8 interval in the whole source."""

    token_id: int
    byte_start: int
    byte_end: int
    text: str


def assertion_tokens(text: object) -> tuple[SourceAssertionToken, ...]:
    """Index Unicode lexical tokens without normalizing, trimming or changing case.

    Words follow Python's Unicode word-character definition; each non-word,
    non-whitespace character is a separate token. Byte coordinates include all
    preceding original whitespace and are relative to the whole input text.
    Invalid Unicode is refused even when it occurs outside a selected token.
    """
    if not isinstance(text, str):
        raise QuoteMemError("schema_invalid", "Assertion token source must be text")
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        raise QuoteMemError("schema_invalid",
                            "Assertion token source must be valid UTF-8") from None
    result: list[SourceAssertionToken] = []
    char_cursor = byte_cursor = 0
    for match in _LEXICAL.finditer(text):
        raw = match.group()
        start = byte_cursor + len(text[char_cursor:match.start()].encode("utf-8"))
        end = start + len(raw.encode("utf-8"))
        result.append(SourceAssertionToken(len(result), start, end, raw))
        char_cursor, byte_cursor = match.end(), end
    return tuple(result)


def locate_assertion_tokens(text: str, start_token: int, end_token: int) -> tuple[int, int]:
    """Resolve an inclusive issued-token range to one exact half-open byte span.

    Whitespace between selected tokens stays inside the resulting interval.
    Boolean, non-integer, negative, reversed and unissued indices are refused.
    A valid interval does not itself assert that its contents are a supported fact.
    """
    tokens = assertion_tokens(text)
    if type(start_token) is not int or type(end_token) is not int or not (
        0 <= start_token <= end_token < len(tokens)
    ):
        raise QuoteMemError("binding_mismatch",
                            "Assertion token range must name issued source tokens")
    return tokens[start_token].byte_start, tokens[end_token].byte_end
