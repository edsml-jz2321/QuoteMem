"""Prompt CLI command group."""

from __future__ import annotations

from heavenbase._lazy import _attach_lazy_exports

__all__ = ["prompt_group"]

_LAZY_EXPORTS = {
    "prompt_group": (".group", "prompt_group"),
}


__getattr__, __dir__ = _attach_lazy_exports(__name__, globals(), _LAZY_EXPORTS, __all__)
