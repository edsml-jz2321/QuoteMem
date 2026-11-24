"""Workspace MetaSchema projection and publication services."""

from __future__ import annotations

from heavenbase._lazy import _attach_lazy_exports

__all__ = ["MetaSchema"]

_LAZY_EXPORTS = {
    "MetaSchema": (".entity", "MetaSchema"),
}


__getattr__, __dir__ = _attach_lazy_exports(__name__, globals(), _LAZY_EXPORTS, __all__)
