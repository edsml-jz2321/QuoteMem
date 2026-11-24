"""Public schema exports."""

from __future__ import annotations

from heavenbase._lazy import _attach_lazy_exports

__all__ = [
    "EntitySchema",
    "FieldSchema",
    "llm_schema_text",
    "register_schema_text",
    "registered_schema_texts",
    "resolve_schema_text",
]

_LAZY_EXPORTS = {
    "EntitySchema": (".model", "EntitySchema"),
    "FieldSchema": (".model", "FieldSchema"),
    "llm_schema_text": (".text", "llm_schema_text"),
    "register_schema_text": (".text", "register_schema_text"),
    "registered_schema_texts": (".text", "registered_schema_texts"),
    "resolve_schema_text": (".text", "resolve_schema_text"),
}


__getattr__, __dir__ = _attach_lazy_exports(__name__, globals(), _LAZY_EXPORTS, __all__)
