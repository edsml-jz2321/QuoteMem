"""Workspace GraphEdge lifecycle and traversal services."""

from __future__ import annotations

from heavenbase._lazy import _attach_lazy_exports

__all__ = ["GraphDeleteRestrictedError"]

_LAZY_EXPORTS = {
    "GraphDeleteRestrictedError": (".errors", "GraphDeleteRestrictedError"),
}


__getattr__, __dir__ = _attach_lazy_exports(__name__, globals(), _LAZY_EXPORTS, __all__)
