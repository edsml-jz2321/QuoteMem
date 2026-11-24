"""TinyDB Backend role."""

from __future__ import annotations

from heavenbase._lazy import _attach_lazy_exports

__all__ = ["TinyDBBackend"]

_LAZY_EXPORTS = {
    "TinyDBBackend": (".runtime", "TinyDBBackend"),
}


__getattr__, __dir__ = _attach_lazy_exports(__name__, globals(), _LAZY_EXPORTS, __all__)
