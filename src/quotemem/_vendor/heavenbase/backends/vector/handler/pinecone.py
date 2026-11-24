"""Pinecone-native operation compilers."""

from __future__ import annotations

__all__ = [
    "pinecone_filter_handler",
    "pinecone_near_handler",
    "register_handlers",
    "register_reasons",
]

from heavenbase.utils import Any

from heavenbase.execution.contracts import HandlerContext
from heavenbase.execution.contracts import QueryFragment
from heavenbase.filters.ops import OP_EQ, OP_NEAR
from heavenbase.strategies import VectorIndex
from heavenbase.types import cosine_score, euclidean_score, inner_product_score
from heavenbase.execution.handlers.filters import filter_handler, normalize_filter_value
from heavenbase.backends.vector._metadata import FLOAT64_EXACT_OPERANDS, exact_operand

BACKEND_IDENTIFIER = "pinecone"
PINECONE_NATIVE_OPS = {"eq": "$eq", "ne": "$ne", "lt": "$lt", "lte": "$lte", "gt": "$gt", "gte": "$gte", "in": "$in"}
PINECONE_TEXT_ARRAY_TYPES = {"identifier", "long-text", "medium-text", "short-text"}
PINECONE_SCAN_TYPES = ("json", "hyper-g", "vector")
# Near metric -> Pinecone index metric.
PINECONE_METRICS = {
    "cosine": "cosine",
    "l2": "euclidean",
    "euclidean": "euclidean",
    "ip": "dotproduct",
    "inner": "dotproduct",
    "inner_product": "dotproduct",
    "dot": "dotproduct",
}
SCAN_SCORERS = {"cosine": cosine_score, "euclidean": euclidean_score, "dotproduct": inner_product_score}


def _supports_array_contains(field_schema: Any, value: Any) -> bool:
    item_type = getattr(field_schema.dtype, "item_type", None)
    item = type(item_type).identifier if item_type is not None else None
    return type(field_schema.dtype).identifier == "array" and item in PINECONE_TEXT_ARRAY_TYPES and isinstance(value, str)


def pinecone_near_handler(field_schema: Any, value: Any, ctx: HandlerContext) -> QueryFragment:
    """Return a Pinecone-native vector near fragment with dimension validation.

    A Pinecone index ranks by the metric it was created with
    (``heavenbase.backend_defaults.pinecone.metric`` at creation; the
    handler reads the metric and data-plane host together) and Pinecone has no row
    scan, so a near must request that metric. The fragment's scan scorer in
    ``notes['row_payload']`` ranks with the same metric and leaves a row without
    a vector out, as native search does.

    Args:
        field_schema (Any): Routed vector field schema.
        value (Any): Query vector validated against the field dtype.
        ctx (HandlerContext): Selected backend and strategy context.

    Returns:
        QueryFragment: Native Pinecone vector near fragment.

    Raises:
        ValueError: If the query dimension or metric is unsupported, or the
            metric is not the one the index ranks by.
    """
    query = field_schema.dtype.validate(value)
    expected_dim = getattr(field_schema.dtype, "dim", None)
    if expected_dim is not None and len(query) != expected_dim:
        raise ValueError(f"Pinecone near query dimension mismatch: expected {expected_dim}, got {len(query)}")
    metric = PINECONE_METRICS.get(str(ctx.metric or "cosine").lower())
    if metric is None:
        raise ValueError(f"Pinecone near: unsupported metric {ctx.metric!r}; use 'cosine', 'l2'/'euclidean', or 'ip'/'inner_product'")
    # An existing index keeps the metric it was created with, whatever the config says now.
    backend = getattr(ctx, "backend_instance", None)
    near_index = getattr(backend, "_near_index", None)
    near_metric = getattr(backend, "near_metric", None)
    entity_id = getattr(ctx, "entity_id", None)
    host, inspected = None, False
    if callable(near_index) and entity_id is not None:
        index_metric, host, inspected = near_index(entity_id)
    elif callable(near_metric) and entity_id is not None:
        index_metric = str(near_metric(entity_id)).lower()
    else:
        index_metric = str(ctx.config.get("heavenbase.backend_defaults.pinecone.metric", default="cosine")).lower()
    if metric != index_metric:
        raise ValueError(
            f"Pinecone near: the index ranks by {index_metric!r}, so metric {ctx.metric!r} would need a row scan, which Pinecone does not provide;"
            " request the index metric or set heavenbase.backend_defaults.pinecone.metric before the index is created"
        )
    scorer = SCAN_SCORERS[metric]

    # Native search never returns a row without a vector, so the scan scorer (a filter that is not
    # pushed down) leaves it out too, under every metric: the filter cannot change the near's rows.
    def _payload(row: dict) -> float | None:
        vec = row.get(field_schema.name)
        return scorer(vec, query) if vec is not None else None

    payload = {"vector": query, "top_k": ctx.top_k, "metric": metric}
    if inspected:
        payload["index_host"] = host
    return QueryFragment(
        ctx.backend,
        "near",
        field_schema.name,
        payload,
        returns=["score"],
        notes={"kind": "pinecone_search", "mode": "native", "top_k": ctx.top_k, "row_payload": _payload},
    )


def pinecone_filter_handler(op: str):
    """Return a Pinecone metadata filter handler for one operation.

    Args:
        op (str): Filter operation identifier.

    Returns:
        Callable[[Any, Any, HandlerContext], QueryFragment]: Compiler that
        produces native or scan-marked Pinecone metadata fragments.
    """

    def _compile(field_schema: Any, value: Any, ctx: HandlerContext) -> QueryFragment:
        value = normalize_filter_value(field_schema.dtype, op, value, resolver=ctx.resolver)
        py_fragment = filter_handler(op)(field_schema, value, ctx)
        if not exact_operand(value, FLOAT64_EXACT_OPERANDS):
            # Pinecone numbers are 64-bit floats, which round an integer beyond +/-2**53, so a
            # stored 2**53 + 1 equals 2**53; the scan predicate compares the exact numbers.
            return py_fragment.mark_scan()
        if op == "array_contains" and _supports_array_contains(field_schema, value):
            payload = {field_schema.name: {"$eq": value}}
            return QueryFragment(
                ctx.backend, op, field_schema.name, payload, notes={"kind": "pinecone_filter", "mode": "native_with_near", "row_payload": py_fragment.payload}
            )
        if op not in PINECONE_NATIVE_OPS or type(field_schema.dtype).identifier in (*PINECONE_SCAN_TYPES, "array"):
            reason = ctx.fallback_reason(op, type(field_schema.dtype).identifier)
            return py_fragment.mark_scan(reason)
        if op == "in" and not isinstance(value, (list, tuple, set)):
            raise TypeError("in filter expects a list, tuple, or set")
        if value is None or (op == "in" and (not value or any(item is None for item in value))):
            # Pinecone metadata filters reject null operands and an empty $in list; the scan predicate tests for a
            # missing value and matches no row for an empty list.
            return py_fragment.mark_scan()
        if op == "in":
            payload = {field_schema.name: {"$in": list(value)}}
        else:
            payload = {field_schema.name: {PINECONE_NATIVE_OPS[op]: value}}
        return QueryFragment(
            ctx.backend, op, field_schema.name, payload, notes={"kind": "pinecone_filter", "mode": "native_with_near", "row_payload": py_fragment.payload}
        )

    return _compile


def register_reasons(fallback: Any, _unsupported: Any, *, resolver: Any = None) -> None:
    """Register Pinecone-specific fallback reasons.

    Args:
        fallback: Reason registry receiving fallback explanations.
        _unsupported: Unsupported-operation registry (unused for Pinecone).
        resolver (Any | None): Optional module resolver for backend discovery.

    Returns:
        None: Fallback reasons are registered in place.
    """
    fallback.register(BACKEND_IDENTIFIER, "*", "json", "*", "Pinecone does not support Json field native filtering; using Python scan fallback.")
    fallback.register(BACKEND_IDENTIFIER, "*", "hyper-g", "*", "Pinecone does not support HyperG field native filtering; using Python scan fallback.")
    fallback.register(
        BACKEND_IDENTIFIER,
        "array_contains",
        "array",
        "*",
        "Pinecone only supports text array_contains natively when combined with near; using Python scan fallback.",
    )


def register_handlers(registry: Any, ctx: Any) -> None:
    """Register Pinecone-native handlers.

    Args:
        registry: Workspace handler registry receiving descriptors.
        ctx: Shared Registry-derived handler vocabulary.

    Returns:
        None: The supplied workspace registry is updated in place.
    """
    for dtype in ctx.filter_types:
        for strategy in ctx.filter_strategies:
            for op in ctx.filter_ops(dtype):
                ctx.handler(dtype, op, BACKEND_IDENTIFIER, strategy, pinecone_filter_handler(op)).register(registry, replace=True)
    for dtype in ctx.vector_types:
        ctx.handler(dtype, OP_NEAR, BACKEND_IDENTIFIER, VectorIndex.identifier, pinecone_near_handler).register(registry, replace=True)
        ctx.handler(dtype, OP_EQ, BACKEND_IDENTIFIER, VectorIndex.identifier, pinecone_filter_handler(OP_EQ)).register(registry, replace=True)
