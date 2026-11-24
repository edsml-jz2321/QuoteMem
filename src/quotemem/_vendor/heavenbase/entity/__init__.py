"""Entity declaration and JSON compiler."""

from __future__ import annotations

from heavenbase._lazy import _attach_lazy_exports

__all__ = [
    "ComputeCallable",
    "ComputeHook",
    "DeriveSpec",
    "Entity",
    "EntityMeta",
    "FieldRef",
    "FieldSpec",
    "QueryComputeHook",
    "field",
    "pending_entities",
    "register_compute",
    "register_query_compute",
    "registered_computes",
    "registered_query_computes",
]

_LAZY_EXPORTS = {
    "Entity": (".base", "Entity"),
    "EntityMeta": (".base", "EntityMeta"),
    "pending_entities": (".base", "pending_entities"),
    "ComputeCallable": (".computes", "ComputeCallable"),
    "register_compute": (".computes", "register_compute"),
    "register_query_compute": (".computes", "register_query_compute"),
    "registered_computes": (".computes", "registered_computes"),
    "registered_query_computes": (".computes", "registered_query_computes"),
    "ComputeHook": (".fields", "ComputeHook"),
    "DeriveSpec": (".fields", "DeriveSpec"),
    "FieldSpec": (".fields", "FieldSpec"),
    "QueryComputeHook": (".fields", "QueryComputeHook"),
    "field": (".fields", "field"),
    "FieldRef": (".ref", "FieldRef"),
}


__getattr__, __dir__ = _attach_lazy_exports(__name__, globals(), _LAZY_EXPORTS, __all__)
