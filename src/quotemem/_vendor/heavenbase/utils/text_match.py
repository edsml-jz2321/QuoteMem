"""Shared normalized text matching helpers."""

from __future__ import annotations

__all__ = [
    "NormalizedText",
    "contained_in_matches",
    "default_normalize_text",
    "keyword_in_query",
    "match_spans",
    "raw_contained_in_matches",
]

import unicodedata

from heavenbase.utils import Any, Callable, dataclass

Origin = tuple[int, int]
NormalizeFn = Callable[[Any], "NormalizedText"]


@dataclass(frozen=True)
class NormalizedText:
    """Normalized text plus optional mapping back to original character spans."""

    text: str
    origins: list[Origin] | None = None
    lang: str = ""
    normalizer: str = ""
    precision: str = "exact"


def default_normalize_text(text: Any) -> NormalizedText:
    """Return dependency-free text normalization for generic scan fallbacks.

    Args:
        text (Any): Source text to normalize; ``None`` becomes an empty string.

    Returns:
        NormalizedText: Case-folded, whitespace-collapsed text with origin spans.
    """
    source = "" if text is None else str(text)
    out = []
    origins: list[Origin] = []
    last_space = False
    for pos, char in enumerate(source):
        piece = unicodedata.normalize("NFC", char).casefold()
        for norm_char in piece:
            if norm_char.isspace() or unicodedata.category(norm_char).startswith("P"):
                if out and not last_space:
                    out.append(" ")
                    origins.append((pos, pos + 1))
                last_space = True
                continue
            if unicodedata.category(norm_char).startswith("C"):
                continue
            out.append(norm_char)
            origins.append((pos, pos + 1))
            last_space = False
    if out and out[-1] == " ":
        out.pop()
        origins.pop()
    return NormalizedText("".join(out), origins, "auto", "default:auto", "exact")


def keyword_in_query(keyword_norm: str, query_norm: str) -> bool:
    """Return whether one normalized keyword appears in one normalized query.

    Args:
        keyword_norm (str): Normalized keyword text.
        query_norm (str): Normalized query text.

    Returns:
        bool: Whether ``keyword_norm`` is a non-empty substring of ``query_norm``.
    """
    return bool(keyword_norm) and keyword_norm in query_norm


def raw_contained_in_matches(field: str, value: Any, query: Any, *, op: str = "contained_in") -> list[dict[str, Any]]:
    """Return raw substring match metadata for scan-compatible fallback paths.

    Args:
        field (str): Field name attached to each match row.
        value (Any): Keyword value or list of keyword values.
        query (Any): Query text searched for raw substrings.
        op (str): Operation token stored on each match row. Defaults to ``"contained_in"``.

    Returns:
        list[dict[str, Any]]: Raw match metadata rows without normalization.
    """
    query_text = "" if query is None else str(query)
    values = list(enumerate(value)) if isinstance(value, list) else [(0, value)]
    matches: list[dict[str, Any]] = []
    for ordinal, keyword in values:
        keyword_text = "" if keyword is None else str(keyword)
        if not keyword_text or keyword_text not in query_text:
            continue
        start = query_text.find(keyword_text)
        matches.append(
            {
                "field": field,
                "op": op,
                "keyword": keyword_text,
                "keyword_norm": keyword_text,
                "query_span": [start, start + len(keyword_text)],
                "query_norm_span": [start, start + len(keyword_text)],
                "query_text": query_text[start : start + len(keyword_text)],
                "ordinal": ordinal,
                "normalizer": "raw",
                "precision": "exact",
            }
        )
    return matches


def contained_in_matches(
    field: str,
    value: Any,
    query: Any,
    *,
    normalize: NormalizeFn | None = None,
    op: str = "contained_in",
) -> list[dict[str, Any]]:
    """Return normalized reverse-containment match metadata.

    Args:
        field (str): Field name attached to each match row.
        value (Any): Keyword value or list of keyword values.
        query (Any): Query text searched after normalization.
        normalize (NormalizeFn | None): Normalizer applied to keywords and query.
            Defaults to :func:`default_normalize_text`.
        op (str): Operation token stored on each match row. Defaults to ``"contained_in"``.

    Returns:
        list[dict[str, Any]]: Normalized match metadata rows for every hit.
    """
    normalize = normalize or default_normalize_text
    query_source = "" if query is None else str(query)
    query_norm = normalize(query_source)
    if not query_norm.text:
        return []
    values = list(enumerate(value)) if isinstance(value, list) else [(0, value)]
    matches: list[dict[str, Any]] = []
    for ordinal, keyword in values:
        keyword_text = "" if keyword is None else str(keyword)
        keyword_norm = normalize(keyword_text)
        if not keyword_in_query(keyword_norm.text, query_norm.text):
            continue
        matches.extend(match_spans(field, op, keyword_text, keyword_norm, query_source, query_norm, ordinal))
    return matches


def match_spans(
    field: str,
    op: str,
    keyword: str,
    keyword_norm: NormalizedText,
    query: str,
    query_norm: NormalizedText,
    ordinal: int,
) -> list[dict[str, Any]]:
    """Return match metadata rows for every normalized keyword occurrence.

    Args:
        field (str): Field name attached to each match row.
        op (str): Operation token stored on each match row.
        keyword (str): Original keyword text.
        keyword_norm (NormalizedText): Normalized keyword payload.
        query (str): Original query text.
        query_norm (NormalizedText): Normalized query payload.
        ordinal (int): Keyword ordinal when ``value`` was a list.

    Returns:
        list[dict[str, Any]]: Match metadata rows for every normalized occurrence.
    """
    if not keyword_norm.text or not query_norm.text:
        return []
    out = []
    start = query_norm.text.find(keyword_norm.text)
    while start >= 0:
        end = start + len(keyword_norm.text)
        query_span, query_text, precision = _origin_span(query, query_norm, start, end)
        out.append(
            {
                "field": field,
                "op": op,
                "keyword": keyword,
                "keyword_norm": keyword_norm.text,
                "query_span": query_span,
                "query_norm_span": [start, end],
                "query_text": query_text,
                "ordinal": ordinal,
                "normalizer": query_norm.normalizer or keyword_norm.normalizer,
                "precision": precision,
            }
        )
        start = query_norm.text.find(keyword_norm.text, start + 1)
    return out


def _origin_span(query: str, normalized: NormalizedText, start: int, end: int) -> tuple[list[int] | None, str, str]:
    if normalized.origins is None:
        return None, "", "normalized_only"
    if start >= len(normalized.origins) or end > len(normalized.origins):
        return None, "", "normalized_only"
    spans = normalized.origins[start:end]
    if not spans:
        return None, "", "normalized_only"
    raw_start = min(span[0] for span in spans)
    raw_end = max(span[1] for span in spans)
    return [raw_start, raw_end], query[raw_start:raw_end], "exact"
