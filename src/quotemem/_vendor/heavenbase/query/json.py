"""JSON query input validation and fluent-query lowering."""

from __future__ import annotations

__all__ = ["load"]

from heavenbase.utils import Any, List, Mapping, Optional, Sequence, raise_mismatch

from ..filters.ops import normalize_op
from ..filters.spec import Expr, Filter, expr_from_dict
from ..modules import ModuleResolver, module_resolver
from .value import Query, _AggregateMetricLookupError
from .spec import AggregateMetric

_QUERY_KEYS = ("aggregate", "available", "filter", "group_by", "having", "limit", "near", "offset", "order", "unsafe", "select", "traverse")
_NEAR_KEYS = ("cache_version", "exact", "field", "metric", "query", "top_k")
_TRAVERSE_KEYS = ("direction", "field", "max_hops", "min_hops", "start")


def load(workspace: Any, entity: type, spec: Mapping[str, Any]) -> Query:
    """Load one validated JSON query into an immutable query value.

    This adapter owns external query shape only. It delegates clause semantics
    and value validation to `Query`, then returns the same value used
    by the Python query surface.

    Args:
        workspace (Any): Workspace facade that will execute the query.
        entity (type): Registered entity class being queried.
        spec (Mapping[str, Any]): JSON query object. Supported top-level keys
            are `filter`, `available`, `near`, `traverse`, `select`, `group_by`,
            `aggregate`, `having`, `order`, `offset`, `limit`, and
            `unsafe`. The unsafe flag is frozen query execution policy rather
            than a filter, aggregate, or traversal clause.

    Returns:
        Query: Immutable query populated from the validated query object.

    Raises:
        TypeError: If the query or one of its clauses has the wrong JSON shape.
        ValueError: If a key is unsupported, a required clause key is missing,
            or a clause value violates fluent-query validation.
    """
    query = _mapping(spec, "query spec")
    _keys(query, _QUERY_KEYS, "JSON query key")
    value = Query(workspace, entity)
    value = _filter(value, query)
    value = _available(value, query)
    value = _near(value, query)
    value = _traverse(value, query)
    value = _select(value, query)
    value = _aggregate(value, query)
    value = _having(value, query)
    value = _order(value, query)
    if "unsafe" in query:
        value = value.unsafe(query["unsafe"])
    if "offset" in query:
        value = value.offset(query["offset"])
    if "limit" in query:
        value = value.limit(query["limit"])
    return value


def query_expr(
    spec: Optional[Mapping[str, Any]],
    schema: Any = None,
    *,
    resolver: ModuleResolver | None = None,
) -> Optional[Expr]:
    """Lower a query-aware filter mapping into an expression tree.

    Args:
        spec (Mapping[str, Any] | None): Mongo-style filter mapping. `None` or
            an empty mapping produces no expression.
        schema (Any): Optional entity schema used for field-aware operation
            normalization and Catalog availability validation.
        resolver (ModuleResolver | None): Context-owned module resolver. The
            default Context is selected when omitted.

    Returns:
        Expr | None: Parsed expression tree, or `None` when no filter exists.

    Raises:
        TypeError: If a nested filter expression has an invalid shape.
        ValueError: If `$available` is used outside the Catalog entity.
    """
    selected = module_resolver(resolver)
    return expr_from_dict(
        spec,
        field_op_normalizer=lambda field_name, op: _normalize_field_op(schema, field_name, op, selected),
        special_parser=lambda key, value: _query_special_exprs(key, value, schema, selected),
        resolver=selected,
    )


def _filter(query: Query, spec: Mapping[str, Any]) -> Query:
    if "filter" not in spec or spec["filter"] is None:
        return query
    return query.where(_mapping(spec["filter"], "filter clause"))


def _available(query: Query, spec: Mapping[str, Any]) -> Query:
    if "available" not in spec or spec["available"] is None:
        return query
    from .catalog import CatalogAvailable

    available = CatalogAvailable.load(spec["available"])
    if available is not None:
        return query.where(available.check(query.entity.schema().entity_id).to_expr())
    return query


def _near(query: Query, spec: Mapping[str, Any]) -> Query:
    if "near" not in spec or spec["near"] is None:
        return query
    near = _mapping(spec["near"], "near clause")
    _keys(near, _NEAR_KEYS, "JSON near key")
    _required(near, ("field", "query"), "near")
    return query.near(
        near["field"], near["query"], near.get("top_k"), near.get("metric"), exact=near.get("exact", False), cache_version=near.get("cache_version")
    )


def _traverse(query: Query, spec: Mapping[str, Any]) -> Query:
    if "traverse" not in spec or spec["traverse"] is None:
        return query
    traverse = _mapping(spec["traverse"], "traverse clause")
    _keys(traverse, _TRAVERSE_KEYS, "JSON traverse key")
    _required(traverse, ("field", "start"), "traverse")
    return query.traverse(
        traverse["field"],
        traverse["start"],
        min_hops=traverse.get("min_hops", 1),
        max_hops=traverse.get("max_hops", 1),
        direction=traverse.get("direction", "out"),
    )


def _select(query: Query, spec: Mapping[str, Any]) -> Query:
    if "select" not in spec:
        return query
    fields = _sequence(spec["select"], "select clause")
    return query.select(*fields)


def _aggregate(query: Query, spec: Mapping[str, Any]) -> Query:
    """Lower canonical group and metric objects into the fluent aggregate surface."""
    if "group_by" in spec:
        query = query.group_by(*_sequence(spec["group_by"], "group_by clause"))
    if "aggregate" not in spec:
        return query
    aggregate = _mapping(spec["aggregate"], "aggregate clause")
    metrics = {}
    for output, raw_metric in aggregate.items():
        if not isinstance(output, str):
            raise TypeError("JSON aggregate output names must be strings")
        metric = _mapping(raw_metric, f"aggregate metric {output!r}")
        if len(metric) != 1:
            raise ValueError(f"JSON aggregate metric {output!r} must contain exactly one operation")
        op, source = next(iter(metric.items()))
        if not isinstance(op, str):
            raise TypeError(f"JSON aggregate metric {output!r} operation must be a string")
        parameterized = _mapping(source, f"aggregate metric {output!r} operation {op!r} parameters")
        _required(parameterized, ("field",), f"aggregate metric {output!r}")
        field = parameterized["field"]
        options = {name: value for name, value in parameterized.items() if name != "field"}
        if not isinstance(field, str):
            raise TypeError(f"JSON aggregate metric {output!r} field must be a string")
        metrics[output] = AggregateMetric(field, op, options=options)
    try:
        return query.aggregate(**metrics)
    except _AggregateMetricLookupError as exc:
        raise_mismatch(exc.supported, exc.op, name="JSON aggregate operation")


def _having(query: Query, spec: Mapping[str, Any]) -> Query:
    """Lower a post-aggregate predicate through the fluent query surface."""
    if "having" not in spec or spec["having"] is None:
        return query
    return query.having(_mapping(spec["having"], "having clause"))


def _order(query: Query, spec: Mapping[str, Any]) -> Query:
    if "order" not in spec:
        return query
    for item in _sequence(spec["order"], "order clause"):
        if not isinstance(item, str):
            raise TypeError("JSON order entries must be strings")
        query = query.order_by(item[1:] if item.startswith("-") else item, desc=item.startswith("-"))
    return query


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"JSON {name} must be an object, got {type(value).__name__}")
    return value


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"JSON {name} must be an array, got {type(value).__name__}")
    return value


def _keys(spec: Mapping[str, Any], supported: Sequence[str], name: str) -> None:
    for key in spec:
        if key not in supported:
            raise_mismatch(supported, key, name=name)


def _required(spec: Mapping[str, Any], required: Sequence[str], name: str) -> None:
    missing = [key for key in required if key not in spec]
    if missing:
        raise ValueError(f"JSON {name} clause requires {', '.join(repr(key) for key in missing)}")


def _query_special_exprs(
    key: str,
    value: Any,
    schema: Any,
    resolver: ModuleResolver,
) -> Optional[List[Expr]]:
    if key == "$available":
        from .catalog import CatalogAvailable

        available = CatalogAvailable.load(value)
        if available is None:
            return []
        if schema is not None:
            available.check(schema.entity_id)
        return [available.to_expr()]
    if "." in key and not key.startswith("$"):
        return _json_path_exprs(key, value, schema, resolver)
    return None


def _json_path_exprs(key: str, value: Any, schema: Any, resolver: ModuleResolver) -> Optional[List[Expr]]:
    """Desugar a dotted filter key into `get` filters on its Json base field.

    ``{"meta.venue": "NeurIPS"}`` lowers to the registered ``get`` operation on
    the ``meta`` field with path ``["venue"]``. Condition mappings keep their
    per-operation shape: ``{"meta.score": {"$gte": 8}}``. Dotted keys whose
    base field is unknown or not Json return ``None`` so the ordinary
    unknown-field diagnostics stay loud.
    """
    if schema is None:
        return None
    base, _, rest = key.partition(".")
    try:
        field_schema = schema.get(base)
    except KeyError:
        return None
    if type(field_schema.dtype).identifier != "json" or not rest:
        return None
    conditions = value.items() if isinstance(value, Mapping) else [("eq", value)]
    return [
        Filter(
            base,
            "get",
            {"path": rest, "op": normalize_op(str(op), resolver=resolver), "value": item},
        )
        for op, item in conditions
    ]


def _normalize_field_op(schema: Any, field_name: str, op: str, resolver: ModuleResolver) -> str:
    normalized = normalize_op(op, resolver=resolver)
    if normalized == "contains" and schema is not None:
        try:
            field_schema = schema.get(field_name)
        except KeyError:
            field_schema = None
        if field_schema is not None and type(field_schema.dtype).identifier == "array":
            return "array_contains"
    return normalized
