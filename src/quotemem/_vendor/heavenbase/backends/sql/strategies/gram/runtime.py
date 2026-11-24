"""Shared sparse-GRAM indexing and query execution helpers."""

from __future__ import annotations

__all__ = [
    "GramKeywordRow",
    "GramRowWrite",
    "StoredKeywordMatcher",
    "contained_in_matches",
    "contained_in_matches_by_id",
    "contained_in_probe_data",
    "index_keyword_rows",
    "keyword_in_query",
    "match_spans",
    "raw_contained_in_matches",
    "scan_contained_in_matcher",
]

from bisect import bisect_left

from heavenbase.utils import Any, Callable, Mapping, Sequence, dataclass
from heavenbase.utils.hash import md5hash
from heavenbase.utils.text_match import (
    NormalizedText,
    keyword_in_query,
    match_spans,
    raw_contained_in_matches,
    contained_in_matches as _contained_in_matches,
)

from .....modules import ModuleResolver, module_resolver
from .analyzers import resolve_gram_tokenizer
from .config import GramConfig
from .normalizers import normalize_many, normalize_text
from .sparse import keyword_grams, query_grams, unit_count

ConfigFor = Callable[[str, str], GramConfig]
SpanMatcher = Callable[[str, str, str, NormalizedText, int], list[dict[str, Any]]]


def _token_starts(text: str, units: list[str]) -> tuple[list[int], list[int]]:
    """Find the earliest and latest ordered placements of literal tokenizer units."""
    earliest = []
    offset = 0
    for unit in units:
        start = text.find(unit, offset)
        if start < 0:
            return [], []
        earliest.append(start)
        offset = start + len(unit)

    latest = earliest.copy()
    offset = len(text)
    for index in range(len(units) - 1, -1, -1):
        offset = text.rfind(units[index], 0, offset)
        latest[index] = offset
    return earliest, latest


def _span_matcher(
    query: str,
    query_norm: NormalizedText,
    config: GramConfig,
    resolver: ModuleResolver,
) -> SpanMatcher:
    text = query_norm.text
    tokenizer = resolve_gram_tokenizer(config, resolver=resolver) if text and config.gram_unit != "char" else None

    def units(value: str) -> list[str]:
        return [unit for unit in tokenizer(value, config) if unit] if tokenizer is not None else []

    query_units = units(text)
    earliest, latest = _token_starts(text, query_units)
    unambiguous = earliest == latest
    positions_known = bool(earliest)

    def contextual_span(keyword_units: list[str], start: int, end: int) -> bool:
        prefix = text[:start]
        before = units(prefix)
        stop = len(before) + len(keyword_units)
        return (
            query_units[:stop] == before + keyword_units
            and units(text[:end]) == query_units[:stop]
            and units(prefix + text[end:]) == before + query_units[stop:]
        )

    def enclosed(starts: list[int], start: int, end: int) -> tuple[int, int] | None:
        first = bisect_left(starts, start)
        stop = bisect_left(starts, end, lo=first)
        if first == stop or (first and starts[first - 1] + len(query_units[first - 1]) > start):
            return None
        if starts[stop - 1] + len(query_units[stop - 1]) > end:
            return None
        return first, stop

    def aligned(keyword_units: list[str], start: int, end: int) -> bool:
        if start == 0 and end == len(text):
            return keyword_units == query_units
        if not positions_known:
            # Keep the query prefix when verifying the remaining units, since
            # the tokenizer may choose its parsing mode from that prefix.
            return contextual_span(keyword_units, start, end)
        token_range = enclosed(earliest, start, end)
        # Tokenizers emit values without offsets. A partial-query span is exact
        # only when both possible extremes enclose the same whole token units.
        if token_range is None or (not unambiguous and token_range != enclosed(latest, start, end)):
            return False
        first, stop = token_range
        return stop - first == len(keyword_units) and query_units[first:stop] == keyword_units

    def matches(field: str, op: str, keyword: str, keyword_norm: NormalizedText, ordinal: int) -> list[dict[str, Any]]:
        spans = match_spans(field, op, keyword, keyword_norm, query, query_norm, ordinal)
        if tokenizer is None or not spans:
            return spans
        keyword_units = units(keyword_norm.text)
        return [span for span in spans if keyword_units and aligned(keyword_units, *span["query_norm_span"])]

    return matches


@dataclass(frozen=True)
class GramKeywordRow:
    """Stored normalized sparse-GRAM keyword candidate."""

    entity_id: str
    field_name: str
    keyword_id: str
    object_id: str
    ordinal: int
    keyword: str
    keyword_norm: str
    normalizer: str
    full_mask: int
    part_count: int
    unit_count: int = 0


@dataclass(frozen=True)
class GramRowWrite:
    """Complete sparse-GRAM index replacement payload for one object."""

    entity_id: str
    object_id: str
    row: dict[str, Any]
    keywords: Sequence[GramKeywordRow]
    postings: Sequence[dict[str, Any]]


def contained_in_matches(
    field: str,
    value: Any,
    query: Any,
    config: GramConfig | None = None,
    *,
    op: str = "contained_in",
    resolver: ModuleResolver | None = None,
) -> list[dict[str, Any]]:
    """Return reverse-containment match metadata using GRAM normalizer config.

    Args:
        field: Logical field name.
        value: Indexed field value to search inside the query.
        query: Query text that may contain indexed values.
        config: Optional sparse-GRAM runtime config. Defaults are loaded when omitted.
        op: Logical operation name attached to match metadata.
        resolver: Context-owned module resolver.

    Returns:
        list[dict[str, Any]]: Exact match metadata rows.
    """
    selected = module_resolver(resolver)
    config = config or GramConfig.load()
    return _contained_in_matches(field, value, query, normalize=lambda text: normalize_text(text, config, resolver=selected), op=op)


def scan_contained_in_matcher(
    field: str,
    query: Any,
    config: GramConfig,
    *,
    short_widths: set[int] | None,
    resolver: ModuleResolver | None = None,
) -> Callable[[Any], list[dict[str, Any]]]:
    """Compile sparse-GRAM candidate and exact span checks for scanned keyword arrays.

    Args:
        field: Logical field name.
        query: Text searched for stored keywords.
        config: Field-specific sparse-GRAM config.
        short_widths: Indexed short keyword widths, or None to generate all widths.
        resolver: Context-owned module resolver.

    Returns:
        A row value matcher that returns metadata for every matching keyword occurrence.
    """
    selected = module_resolver(resolver)
    source = "" if query is None else str(query)
    query_norm = normalize_text(source, config, resolver=selected)
    hashes = query_grams(query_norm.text, config, short_widths=short_widths, resolver=selected) if query_norm.text else set()
    verified_spans = _span_matcher(source, query_norm, config, selected)

    def match(value: Any) -> list[dict[str, Any]]:
        if not hashes or not isinstance(value, list):
            return []
        keywords = ["" if item is None else str(item) for item in value]
        matches = []
        for ordinal, (keyword, norm) in enumerate(zip(keywords, normalize_many(keywords, config, resolver=selected))):
            if not keyword_in_query(norm.text, query_norm.text):
                continue
            parts = keyword_grams(norm.text, config, resolver=selected)
            if parts and all(part.gram_hash in hashes for part in parts):
                matches.extend(verified_spans(field, "contained_in", keyword, norm, ordinal))
        return matches

    return match


def index_keyword_rows(
    entity_id: str,
    object_id: str,
    values: Mapping[str, Any],
    fields: Mapping[str, Any],
    config_for: ConfigFor,
    *,
    resolver: ModuleResolver | None = None,
) -> tuple[list[GramKeywordRow], list[dict[str, Any]]]:
    """Build normalized keyword rows and sparse posting rows for one object.

    Args:
        entity_id: Logical entity identifier.
        object_id: User-facing row object id.
        values: Backend row values for the object.
        fields: Sparse-GRAM field schemas keyed by field name.
        config_for: Callable returning the field-specific sparse-GRAM config.
        resolver: Context-owned module resolver.

    Returns:
        tuple[list[GramKeywordRow], list[dict[str, Any]]]: Normalized keyword rows and
            posting rows for SQL persistence.
    """
    selected = module_resolver(resolver)
    keywords: list[GramKeywordRow] = []
    postings: list[dict[str, Any]] = []
    for field_name in sorted(fields):
        raw_values = values.get(field_name) or []
        if not isinstance(raw_values, list):
            continue
        source_values = ["" if value is None else str(value) for value in raw_values]
        config = config_for(entity_id, field_name)
        normalized = normalize_many(source_values, config, resolver=selected)
        for ordinal, (source, norm) in enumerate(zip(source_values, normalized)):
            parts = keyword_grams(norm.text, config, resolver=selected)
            if not parts:
                continue
            keyword_id = md5hash([entity_id, field_name, object_id, ordinal, norm.text, config.fingerprint()])
            full_mask = 0
            for part in parts:
                full_mask |= part.mask_bit
                postings.append(
                    {
                        "entity_id": entity_id,
                        "field_name": field_name,
                        "keyword_id": keyword_id,
                        "gram_hash": part.gram_hash,
                        "mask_bit": part.mask_bit,
                    }
                )
            keywords.append(
                GramKeywordRow(
                    entity_id=entity_id,
                    field_name=field_name,
                    keyword_id=keyword_id,
                    object_id=object_id,
                    ordinal=ordinal,
                    keyword=source,
                    keyword_norm=norm.text,
                    normalizer=norm.normalizer or config.normalizer_id(),
                    unit_count=unit_count(norm.text, config, resolver=selected),
                    full_mask=full_mask,
                    part_count=len(parts),
                )
            )
    return keywords, postings


class StoredKeywordMatcher:
    """Decide stored keywords against one query exactly as the index route decides them.

    A keyword matches when its normalized text appears in the normalized
    query and the query holds every gram hash the keyword is indexed under,
    the test :func:`contained_in_matches_by_id` applies to the candidates the
    index returns. With a token or space gram unit the exact span check
    requires the keyword to cover whole query units, so ``cat`` is not found
    in ``concatenate``. A keyword without analyzer units has no index row and
    never matches. Match metadata is built as the index route builds it, so
    for a row the index holds :meth:`matches` returns the metadata the lookup
    returns for it. Each distinct keyword is normalized and decided once per
    matcher.

    Args:
        field_name: Field name attached to each match row.
        query: Query text that may contain stored keywords.
        config: Field-specific sparse-GRAM config.
        op: Logical operation name attached to each match row.
        short_widths: Short keyword widths observed in the index; the width
            of the keyword being decided is added, as the index holds it once
            it holds the keyword.
        resolver: Context-owned module resolver.
    """

    def __init__(
        self,
        field_name: str,
        query: Any,
        config: GramConfig,
        *,
        op: str,
        short_widths: set[int],
        resolver: ModuleResolver | None = None,
    ) -> None:
        self._field_name = field_name
        self._op = op
        self._config = config
        self._short_widths = set(short_widths)
        self._resolver = module_resolver(resolver)
        query_source = "" if query is None else str(query)
        self._query_norm = normalize_text(query_source, config, resolver=self._resolver)
        self._verified_spans = _span_matcher(query_source, self._query_norm, config, self._resolver)
        self._hashes_by_width: dict[int, set[str]] = {}
        # stored keyword text -> its match rows at ordinal 0, or None when it does not match
        self._spans: dict[str, list[dict[str, Any]] | None] = {}

    def holds(self, values: Any) -> bool:
        """Return whether any keyword in a stored field value matches the query.

        Args:
            values: Stored field value; a value that is not a list never matches.

        Returns:
            bool: ``True`` when at least one keyword matches.
        """
        return any(self._spans[source] is not None for source in self._decide(values))

    def matches(self, values: Any) -> list[dict[str, Any]]:
        """Return the match metadata of every matching keyword in a stored field value, in keyword order.

        Args:
            values: Stored field value; a value that is not a list never matches.

        Returns:
            list[dict[str, Any]]: Match metadata rows; empty when no keyword matches.
        """
        out: list[dict[str, Any]] = []
        for ordinal, source in enumerate(self._decide(values)):
            spans = self._spans[source]
            if spans:
                out.extend({**span, "ordinal": ordinal} for span in spans)
        return out

    def _decide(self, values: Any) -> list[str]:
        if not self._query_norm.text or not isinstance(values, list):
            return []
        sources = ["" if value is None else str(value) for value in values]
        new = [source for source in dict.fromkeys(sources) if source not in self._spans]
        for source, norm in zip(new, normalize_many(new, self._config, resolver=self._resolver)):
            self._spans[source] = self._keyword_spans(source, norm)
        return sources

    def _keyword_spans(self, source: str, norm: NormalizedText) -> list[dict[str, Any]] | None:
        config, selected = self._config, self._resolver
        if not keyword_in_query(norm.text, self._query_norm.text):
            return None
        parts = keyword_grams(norm.text, config, resolver=selected)
        if not parts:
            return None
        width = unit_count(norm.text, config, resolver=selected)
        extra_width = width if width <= config.gram_width and width not in self._short_widths else 0
        if extra_width not in self._hashes_by_width:
            widths = self._short_widths | {extra_width} if extra_width else self._short_widths
            self._hashes_by_width[extra_width] = query_grams(self._query_norm.text, config, short_widths=widths, resolver=selected)
        if not all(part.gram_hash in self._hashes_by_width[extra_width] for part in parts):
            return None
        keyword_norm = NormalizedText(norm.text, None, normalizer=norm.normalizer or config.normalizer_id(), precision="normalized_only")
        return self._verified_spans(self._field_name, self._op, source, keyword_norm, 0) or None


def contained_in_matches_by_id(
    entity_id: str,
    field_name: str,
    query: Any,
    *,
    op: str,
    config: GramConfig,
    short_widths: set[int],
    candidates_for: Callable[[set[str]], list[GramKeywordRow]],
    resolver: ModuleResolver | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Return exact-verified reverse-containment match metadata keyed by object id.

    Args:
        entity_id: Logical entity identifier.
        field_name: Sparse-GRAM field name.
        query: Query text that may contain indexed keywords.
        op: Logical operation name attached to match metadata.
        config: Field-specific sparse-GRAM config.
        short_widths: Short keyword widths observed in the index.
        candidates_for: Callable that loads candidate keyword rows for query gram hashes.
        resolver: Context-owned module resolver.

    Returns:
        dict[str, list[dict[str, Any]]]: Match metadata keyed by object id.
    """
    selected = module_resolver(resolver)
    query_source = "" if query is None else str(query)
    query_norm = normalize_text(query_source, config, resolver=selected)
    if not query_norm.text:
        return {}
    candidates = candidates_for(query_grams(query_norm.text, config, short_widths=short_widths, resolver=selected))
    verified_spans = _span_matcher(query_source, query_norm, config, selected)
    matches_by_id: dict[str, list[dict[str, Any]]] = {}
    for candidate in candidates:
        if not keyword_in_query(candidate.keyword_norm, query_norm.text):
            continue
        keyword_norm = NormalizedText(candidate.keyword_norm, None, normalizer=candidate.normalizer, precision="normalized_only")
        matches = verified_spans(field_name, op, candidate.keyword, keyword_norm, candidate.ordinal)
        if matches:
            matches_by_id.setdefault(candidate.object_id, []).extend(matches)
    return matches_by_id


def contained_in_probe_data(
    entity_id: str,
    field_name: str,
    query: Any,
    *,
    config: GramConfig,
    short_widths: set[int],
    candidates_for: Callable[[set[str]], list[GramKeywordRow]],
    posting_hits: Callable[[set[str]], int],
    resolver: ModuleResolver | None = None,
) -> dict[str, Any]:
    """Return exact-recall lookup metrics for one sparse-GRAM field.

    Args:
        entity_id: Logical entity identifier.
        field_name: Sparse-GRAM field name.
        query: Query text to probe.
        config: Field-specific sparse-GRAM config.
        short_widths: Short keyword widths observed in the index.
        candidates_for: Callable that loads candidate keyword rows for query gram hashes.
        posting_hits: Callable that counts posting rows for query gram hashes.
        resolver: Context-owned module resolver.

    Returns:
        dict[str, Any]: Query units, hash counts, candidate counts, and analyzer metadata.
    """
    selected = module_resolver(resolver)
    query_source = "" if query is None else str(query)
    query_norm = normalize_text(query_source, config, resolver=selected)
    hashes = query_grams(query_norm.text, config, short_widths=short_widths, resolver=selected)
    candidates = candidates_for(hashes)
    verified_spans = _span_matcher(query_source, query_norm, config, selected) if candidates and config.gram_unit != "char" else None
    exact_matches = 0
    for candidate in candidates:
        if not keyword_in_query(candidate.keyword_norm, query_norm.text):
            continue
        if verified_spans is not None:
            keyword_norm = NormalizedText(candidate.keyword_norm, None, normalizer=candidate.normalizer, precision="normalized_only")
            if not verified_spans(field_name, "contained_in", candidate.keyword, keyword_norm, candidate.ordinal):
                continue
        exact_matches += 1
    return {
        "query_units": unit_count(query_norm.text, config, resolver=selected),
        "query_hashes": len(hashes),
        "posting_hits": posting_hits(hashes),
        "candidates": len(candidates),
        "exact_matches": exact_matches,
        "normalizer": config.normalizer_id(),
        "gram_unit": config.gram_unit,
        "tokenizer": config.tokenizer,
    }
