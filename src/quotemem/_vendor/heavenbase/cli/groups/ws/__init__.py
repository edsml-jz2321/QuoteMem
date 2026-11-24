"""Workspace CLI group package API."""

from __future__ import annotations

from heavenbase._lazy import _attach_lazy_exports

__all__ = ["ws_group"]

_LAZY_EXPORTS = {
    "ws_group": (".group", "ws_group"),
}


__getattr__, __dir__ = _attach_lazy_exports(__name__, globals(), _LAZY_EXPORTS, __all__)
