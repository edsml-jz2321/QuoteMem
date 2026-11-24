"""Neutral result-frame execution values."""

from __future__ import annotations

from heavenbase._lazy import _attach_lazy_exports

__all__ = ["ResultFrame", "difference", "intersect", "union"]

_LAZY_EXPORTS = {
    "difference": (".merge", "difference"),
    "intersect": (".merge", "intersect"),
    "union": (".merge", "union"),
    "ResultFrame": (".result", "ResultFrame"),
}


__getattr__, __dir__ = _attach_lazy_exports(__name__, globals(), _LAZY_EXPORTS, __all__)
