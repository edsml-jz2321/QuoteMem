"""Redis backend handler role."""

from __future__ import annotations

from heavenbase._lazy import _attach_lazy_exports

__all__ = ["register_handlers"]

_LAZY_EXPORTS = {
    "register_handlers": (".runtime", "register_handlers"),
}


__getattr__, __dir__ = _attach_lazy_exports(__name__, globals(), _LAZY_EXPORTS, __all__)
