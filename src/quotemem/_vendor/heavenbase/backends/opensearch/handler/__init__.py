"""OpenSearch Handler role."""

from __future__ import annotations

from heavenbase._lazy import _attach_lazy_exports

__all__ = [
    "opensearch_filter_handler",
    "register_handlers",
    "register_reasons",
]

_LAZY_EXPORTS = {
    "opensearch_filter_handler": (".runtime", "opensearch_filter_handler"),
    "register_handlers": (".runtime", "register_handlers"),
    "register_reasons": (".runtime", "register_reasons"),
}


__getattr__, __dir__ = _attach_lazy_exports(__name__, globals(), _LAZY_EXPORTS, __all__)
