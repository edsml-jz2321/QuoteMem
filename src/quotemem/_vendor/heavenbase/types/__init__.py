"""Canonical logical type extension surface."""

from __future__ import annotations

from heavenbase._lazy import _attach_lazy_exports

__all__ = [
    "Array",
    "Artifact",
    "Boolean",
    "Categorical",
    "Date",
    "Duration",
    "Float",
    "get_logical_type",
    "HyperG",
    "Identifier",
    "Integer",
    "Json",
    "logical_type_aliases",
    "LogicalType",
    "logical_type_from_any",
    "logical_type_from_mapping",
    "logical_type_from_text",
    "LongText",
    "MediumText",
    "register_logical_type",
    "ShortText",
    "Timestamp",
    "Vector",
    "coerce_type",
    "cosine_score",
    "euclidean_score",
    "inner_product_score",
]

_LAZY_EXPORTS = {
    "Array": (".array", "Array"),
    "Artifact": (".artifact", "Artifact"),
    "LogicalType": (".base", "LogicalType"),
    "Boolean": (".boolean", "Boolean"),
    "Categorical": (".categorical", "Categorical"),
    "Date": (".date", "Date"),
    "coerce_type": (".descriptor", "coerce_type"),
    "logical_type_from_any": (".descriptor", "logical_type_from_any"),
    "logical_type_from_mapping": (".descriptor", "logical_type_from_mapping"),
    "logical_type_from_text": (".descriptor", "logical_type_from_text"),
    "Duration": (".duration", "Duration"),
    "Float": (".float", "Float"),
    "HyperG": (".hyperg", "HyperG"),
    "Identifier": (".identifier", "Identifier"),
    "Integer": (".integer", "Integer"),
    "Json": (".json", "Json"),
    "LongText": (".long_text", "LongText"),
    "MediumText": (".medium_text", "MediumText"),
    "get_logical_type": (".registry", "get_logical_type"),
    "logical_type_aliases": (".registry", "logical_type_aliases"),
    "register_logical_type": (".registry", "register_logical_type"),
    "ShortText": (".short_text", "ShortText"),
    "Timestamp": (".timestamp", "Timestamp"),
    "Vector": (".vector", "Vector"),
    "cosine_score": (".vector", "cosine_score"),
    "euclidean_score": (".vector", "euclidean_score"),
    "inner_product_score": (".vector", "inner_product_score"),
}


__getattr__, __dir__ = _attach_lazy_exports(__name__, globals(), _LAZY_EXPORTS, __all__)
