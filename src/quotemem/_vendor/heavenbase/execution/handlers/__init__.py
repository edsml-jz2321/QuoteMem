"""Shared operation-handler extension surface."""

from __future__ import annotations

from heavenbase._lazy import _attach_lazy_exports

_REGISTRY_EXPORTS = {
    "AggregateCompileResult": (".aggregate", "AggregateCompileResult"),
    "AggregateCompiler": (".aggregate", "AggregateCompiler"),
    "AggregateCompilerInput": (".aggregate", "AggregateCompilerInput"),
    "AggregateMetricInput": (".aggregate", "AggregateMetricInput"),
    "Handler": (".base", "Handler"),
    "HandlerCompiler": (".base", "HandlerCompiler"),
    "HandlerKey": (".base", "HandlerKey"),
    "HandlerRegistry": (".registry", "HandlerRegistry"),
}
_PLUGIN_EXPORTS = {
    "HandlerPlugin": (".plugins", "HandlerPlugin"),
    "HandlerSeedContext": (".plugins", "HandlerSeedContext"),
    "handler_plugins": (".plugins", "handler_plugins"),
}
_PREDICATE_EXPORTS = {
    "Predicate": (".predicates", "Predicate"),
}
_LAZY_EXPORTS = {**_REGISTRY_EXPORTS, **_PLUGIN_EXPORTS, **_PREDICATE_EXPORTS}

__all__ = [*_LAZY_EXPORTS]


__getattr__, __dir__ = _attach_lazy_exports(__name__, globals(), _LAZY_EXPORTS, __all__)
