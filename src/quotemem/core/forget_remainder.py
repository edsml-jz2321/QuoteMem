"""The readable rest of an Evidence part a complete forget withheld part of (``agent`` mode).

A forget withholds byte ranges of a part ("from Chengdu" of "Correction: last month I moved
from Chengdu to Xiamen, and I live in Xiamen now."). The part's stored vector and terms encode
every byte, so the readability check ``p_evr`` keeps it from being ranked or rendered again, and the
rest of the sentence, the current city, could no longer be found. A complete forget embeds the
part's readable bytes once, before its fence holds the scope, and keeps the vector on the part's
withheld ranges (``ForgetRemainderV1``, ``ForgetOwner._prefill``); until then the part stays
unfound. The interactive Search of an ``agent`` workspace ranks the part by that vector and by the
terms of its readable bytes only, and delivers those bytes alone (``masked``). Nothing here reads a
withheld byte: the readable text replaces each withheld byte by a space.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from typing import Any, Final

from quotemem.core.entities import ForgetRemainderV1
from quotemem.core.recall_snapshot import SourceRowV2

FORGET_REASONS: Final = frozenset({"forgotten", "restated_forgotten"})
"""Why a forget withholds a range: a part withheld for any other reason keeps no remainder."""


def readable_text(text: str, byte_start: int, ranges: Iterable[tuple[int, int]]) -> str:
    """A part's text with every byte of the withheld ``ranges`` (message coordinates, on
    character boundaries) replaced by a space, so it keeps the part's byte offsets."""
    raw = bytearray(text.encode("utf-8"))
    for start, end in ranges:
        raw[start - byte_start:end - byte_start] = b" " * (end - start)
    return raw.decode("utf-8")


def remainder_text(text: str, byte_start: int, ranges: Iterable[tuple[int, int]]) -> str:
    """What a part's remainder is embedded and found by: its readable text, its runs of
    blanks made one ("Correction: last month I moved to Xiamen, and I live in Xiamen now.")."""
    return " ".join(readable_text(text, byte_start, ranges).split())


def stored_remainder(row: Mapping[str, Any]) -> Sequence[float] | None:
    """``current_remainder`` of a stored Evidence row (its mapping): the remainder's vector
    while it holds; None otherwise, and for a part no forget withheld."""
    spans: list[Mapping[str, Any]] = row["withheld_spans"]
    if row["state"] != "active" or not spans or any(
            span["reason"] not in FORGET_REASONS for span in spans):
        return None
    ranges = [[span["byte_start"], span["byte_end"]] for span in spans]
    for span in spans:
        found = span.get("remainder")
        if found is not None and [list(pair) for pair in found["ranges"]] == ranges and (
                found["embedding_space"] == row["embedding_space"]
                and len(found["embedding"]) == len(row["embedding"])):
            return found["embedding"]
    return None


def current_remainder(row: SourceRowV2) -> ForgetRemainderV1 | None:
    """A part's remainder while it holds: the part is active, every withheld range is a
    forget's (``FORGET_REASONS``), and one range carries a remainder embedded under exactly
    the part's current withheld ranges, in the part's own space and dimension; else None."""
    spans = row.withheld_spans
    if row.state != "active" or not spans or any(
            span.reason not in FORGET_REASONS for span in spans):
        return None
    ranges = [(span.byte_start, span.byte_end) for span in spans]
    for span in spans:
        found = span.remainder
        if found is not None and [tuple(pair) for pair in found.ranges] == ranges and (
                found.embedding_space == row.embedding_space
                and len(found.embedding) == len(row.embedding)):
            return found
    return None


def masked(row: SourceRowV2) -> SourceRowV2:
    """A part as the interactive Search ranks and shows its remainder: its text with every
    withheld byte a space (``readable_text``) and, while its remainder holds
    (``current_remainder``), that remainder's vector in place of the stored one, which
    encodes every byte; every other field, its withheld ranges included, as stored."""
    remainder = current_remainder(row)
    text = readable_text(row.text, row.byte_start, ((span.byte_start, span.byte_end)
                                                    for span in row.withheld_spans))
    if remainder is None:
        return replace(row, text=text)
    return replace(row, text=text, embedding=tuple(remainder.embedding))
