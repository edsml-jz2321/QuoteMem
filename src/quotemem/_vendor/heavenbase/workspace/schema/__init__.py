"""Workspace schema registration, metadata, and placement policy."""

from __future__ import annotations

from heavenbase._lazy import _attach_lazy_exports

__all__ = ["CATALOG_MODES", "EntityOptions"]

_LAZY_EXPORTS = {
    "CATALOG_MODES": (".options", "CATALOG_MODES"),
    "EntityOptions": (".options", "EntityOptions"),
}


__getattr__, __dir__ = _attach_lazy_exports(__name__, globals(), _LAZY_EXPORTS, __all__)
