"""Built-in storage strategies."""

from __future__ import annotations

from heavenbase._lazy import _attach_lazy_exports

__all__ = [
    "ExternalRef",
    "GraphEdge",
    "InlineColumn",
    "InvertedIndex",
    "JsonField",
    "register_strategy",
    "SideTable",
    "SparseGramIndex",
    "strategy_aliases",
    "StorageStrategy",
    "VectorIndex",
]

_LAZY_EXPORTS = {
    "StorageStrategy": (".base", "StorageStrategy"),
    "ExternalRef": (".external_ref", "ExternalRef"),
    "GraphEdge": (".graph_edge", "GraphEdge"),
    "InlineColumn": (".inline_column", "InlineColumn"),
    "InvertedIndex": (".inverted_index", "InvertedIndex"),
    "JsonField": (".json_field", "JsonField"),
    "register_strategy": (".registry", "register_strategy"),
    "strategy_aliases": (".registry", "strategy_aliases"),
    "SideTable": (".side_table", "SideTable"),
    "SparseGramIndex": (".sparse_gram", "SparseGramIndex"),
    "VectorIndex": (".vector_index", "VectorIndex"),
}


__getattr__, __dir__ = _attach_lazy_exports(__name__, globals(), _LAZY_EXPORTS, __all__)
