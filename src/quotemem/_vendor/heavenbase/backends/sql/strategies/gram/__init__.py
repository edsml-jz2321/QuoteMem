"""Sparse-GRAM runtime for SQL strategy adapters."""

from __future__ import annotations

from heavenbase._lazy import _attach_lazy_exports

_CONFIG_EXPORTS = {
    "GramConfig": (".config", "GramConfig"),
}
_NORMALIZER_EXPORTS = {
    "NormalizedText": (".normalizers", "NormalizedText"),
    "normalize_many": (".normalizers", "normalize_many"),
    "normalize_text": (".normalizers", "normalize_text"),
    "register_gram_normalizer": (".normalizers", "register_gram_normalizer"),
    "resolve_gram_normalizer": (".normalizers", "resolve_gram_normalizer"),
}
_SPARSE_EXPORTS = {
    "keyword_grams": (".sparse", "keyword_grams"),
    "query_grams": (".sparse", "query_grams"),
}
_RUNTIME_EXPORTS = {
    "GramKeywordRow": (".runtime", "GramKeywordRow"),
    "GramRowWrite": (".runtime", "GramRowWrite"),
    "contained_in_matches": (".runtime", "contained_in_matches"),
    "match_spans": (".runtime", "match_spans"),
    "raw_contained_in_matches": (".runtime", "raw_contained_in_matches"),
}
_TOKENIZER_EXPORTS = {
    "register_gram_tokenizer": (".analyzers", "register_gram_tokenizer"),
    "resolve_gram_tokenizer": (".analyzers", "resolve_gram_tokenizer"),
}
_LAZY_EXPORTS = {
    **_CONFIG_EXPORTS,
    **_NORMALIZER_EXPORTS,
    **_SPARSE_EXPORTS,
    **_RUNTIME_EXPORTS,
    **_TOKENIZER_EXPORTS,
}

__all__ = [*_LAZY_EXPORTS]


__getattr__, __dir__ = _attach_lazy_exports(__name__, globals(), _LAZY_EXPORTS, __all__)
