"""Public immutable query and request-spec exports."""

from __future__ import annotations

from heavenbase._lazy import _attach_lazy_exports

__all__ = [
    "Aggregate",
    "AggregateMetric",
    "AggregateReducer",
    "Near",
    "OrderKey",
    "Query",
    "QuerySpec",
    "Traverse",
    "aggregate_reducers",
    "register_aggregate_reducer",
]

_LAZY_EXPORTS = {
    "AggregateReducer": (".aggregate", "AggregateReducer"),
    "aggregate_reducers": (".aggregate", "aggregate_reducers"),
    "register_aggregate_reducer": (".aggregate", "register_aggregate_reducer"),
    "Aggregate": (".spec", "Aggregate"),
    "AggregateMetric": (".spec", "AggregateMetric"),
    "Near": (".spec", "Near"),
    "OrderKey": (".spec", "OrderKey"),
    "QuerySpec": (".spec", "QuerySpec"),
    "Traverse": (".spec", "Traverse"),
    "Query": (".value", "Query"),
}


__getattr__, __dir__ = _attach_lazy_exports(__name__, globals(), _LAZY_EXPORTS, __all__)
