"""LLM CLI command group."""

from __future__ import annotations

from heavenbase._lazy import _attach_lazy_exports

__all__ = ["llm_group"]

_LAZY_EXPORTS = {
    "llm_group": (".group", "llm_group"),
}


__getattr__, __dir__ = _attach_lazy_exports(__name__, globals(), _LAZY_EXPORTS, __all__)
