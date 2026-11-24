"""Public logical placement exports."""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any

from heavenbase._lazy import _attach_lazy_exports

__all__ = [
    "ExternalRef",
    "GraphEdge",
    "InlineColumn",
    "InvertedIndex",
    "IndexBinding",
    "IndexRule",
    "JsonField",
    "registered_storage_profiles",
    "SideTable",
    "SparseGramIndex",
    "StorageBinding",
    "StoragePlan",
    "StorageProfile",
    "StorageResolver",
    "StorageRule",
    "UniqueBinding",
    "UniqueRule",
    "VectorIndex",
    "index",
    "storage_profile_for",
    "unique",
]

_LAZY_EXPORTS = {
    "ExternalRef": ("..strategies", "ExternalRef"),
    "GraphEdge": ("..strategies", "GraphEdge"),
    "InlineColumn": ("..strategies", "InlineColumn"),
    "InvertedIndex": ("..strategies", "InvertedIndex"),
    "JsonField": ("..strategies", "JsonField"),
    "SideTable": ("..strategies", "SideTable"),
    "SparseGramIndex": ("..strategies", "SparseGramIndex"),
    "VectorIndex": ("..strategies", "VectorIndex"),
    "StorageBinding": (".binding", "StorageBinding"),
    "StoragePlan": (".binding", "StoragePlan"),
    "StorageProfile": (".profile", "StorageProfile"),
    "registered_storage_profiles": (".profiles", "registered_storage_profiles"),
    "storage_profile_for": (".profiles", "storage_profile_for"),
    "StorageResolver": (".resolver", "StorageResolver"),
    "StorageRule": (".rule", "StorageRule"),
    "IndexBinding": (".index", "IndexBinding"),
    "IndexRule": (".index", "IndexRule"),
    "UniqueBinding": (".index", "UniqueBinding"),
    "UniqueRule": (".index", "UniqueRule"),
    "index": (".index", "index"),
    "unique": (".index", "unique"),
}


class _PlacementModule(ModuleType):
    """Keep ``index`` bound to the public ``index()`` function.

    Python sets a submodule as an attribute of its package when the submodule
    is first imported. ``heavenbase.placement.index`` names both the
    ``index()`` export and the submodule that defines it, so that binding must
    not replace the export.
    """

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "index" and isinstance(value, ModuleType):
            return
        super().__setattr__(name, value)


__getattr__, __dir__ = _attach_lazy_exports(__name__, globals(), _LAZY_EXPORTS, __all__)
sys.modules[__name__].__class__ = _PlacementModule
