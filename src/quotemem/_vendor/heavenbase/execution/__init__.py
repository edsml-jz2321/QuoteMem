"""Neutral execution contracts, result frames, and handler compilation."""

from __future__ import annotations

from heavenbase._lazy import _attach_lazy_exports

_LAZY_EXPORTS = {
    "HandlerContext": (".contracts", "HandlerContext"),
    "QueryFragment": (".contracts", "QueryFragment"),
    "RowOp": (".contracts", "RowOp"),
    "ResultFrame": (".frame", "ResultFrame"),
    "difference": (".frame", "difference"),
    "intersect": (".frame", "intersect"),
    "union": (".frame", "union"),
}

__all__ = [*_LAZY_EXPORTS]


__getattr__, __dir__ = _attach_lazy_exports(__name__, globals(), _LAZY_EXPORTS, __all__)
