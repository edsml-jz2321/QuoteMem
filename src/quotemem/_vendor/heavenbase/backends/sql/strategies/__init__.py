"""SQL-hosted storage strategy adapters."""

from __future__ import annotations

from heavenbase._lazy import _attach_lazy_exports

_ADAPTER_EXPORTS = {
    "SQLSparseGramAdapter": (".sparse_gram", "SQLSparseGramAdapter"),
    "SparseGramIndexConflictError": (".sparse_gram", "SparseGramIndexConflictError"),
}

__all__ = [*_ADAPTER_EXPORTS]


__getattr__, __dir__ = _attach_lazy_exports(__name__, globals(), _ADAPTER_EXPORTS, __all__)
