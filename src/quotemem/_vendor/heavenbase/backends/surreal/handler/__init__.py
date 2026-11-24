"""SurrealDB backend handler role."""

from __future__ import annotations

from heavenbase._lazy import _attach_lazy_exports

__all__ = [
    "register_handlers",
    "register_reasons",
    "surreal_filter_handler",
    "surreal_near_handler",
]

_LAZY_EXPORTS = {
    "register_handlers": (".runtime", "register_handlers"),
    "register_reasons": (".runtime", "register_reasons"),
    "surreal_filter_handler": (".runtime", "surreal_filter_handler"),
    "surreal_near_handler": (".runtime", "surreal_near_handler"),
}


__getattr__, __dir__ = _attach_lazy_exports(__name__, globals(), _LAZY_EXPORTS, __all__)
