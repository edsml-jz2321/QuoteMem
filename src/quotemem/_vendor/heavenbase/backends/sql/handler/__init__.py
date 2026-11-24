"""SQL-family backend handler role."""

from __future__ import annotations

from heavenbase._lazy import _attach_lazy_exports

__all__ = [
    "register_handlers",
    "register_reasons",
]

_LAZY_EXPORTS = {
    "register_handlers": (".base", "register_handlers"),
    "register_reasons": (".base", "register_reasons"),
}


__getattr__, __dir__ = _attach_lazy_exports(__name__, globals(), _LAZY_EXPORTS, __all__)
