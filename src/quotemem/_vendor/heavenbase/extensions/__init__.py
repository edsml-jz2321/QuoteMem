"""Workspace extension contracts, registry, and built-in bundles."""

from __future__ import annotations

from heavenbase._lazy import _attach_lazy_exports

__all__ = [
    "Extension",
    "AGENT_EXTENSION_ID",
    "CAPSULE_EXTENSION_ID",
    "DATABASE_EXTENSION_ID",
    "MEMORY_EXTENSION_ID",
    "PROMPT_EXTENSION_ID",
    "TOOLKIT_EXTENSION_ID",
]

_LAZY_EXPORTS = {
    "Extension": (".base", "Extension"),
    "AGENT_EXTENSION_ID": (".ids", "AGENT_EXTENSION_ID"),
    "CAPSULE_EXTENSION_ID": (".ids", "CAPSULE_EXTENSION_ID"),
    "DATABASE_EXTENSION_ID": (".ids", "DATABASE_EXTENSION_ID"),
    "MEMORY_EXTENSION_ID": (".ids", "MEMORY_EXTENSION_ID"),
    "PROMPT_EXTENSION_ID": (".ids", "PROMPT_EXTENSION_ID"),
    "TOOLKIT_EXTENSION_ID": (".ids", "TOOLKIT_EXTENSION_ID"),
}


__getattr__, __dir__ = _attach_lazy_exports(__name__, globals(), _LAZY_EXPORTS, __all__)
