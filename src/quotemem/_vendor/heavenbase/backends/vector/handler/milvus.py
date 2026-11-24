"""Milvus-native operation compilers."""

from __future__ import annotations

__all__ = [
    "milvus_filter_handler",
    "milvus_near_handler",
    "register_handlers",
    "register_reasons",
]

from heavenbase.utils import Any

from heavenbase.execution.contracts import HandlerContext
from heavenbase.execution.contracts import QueryFragment
from heavenbase.backends.vector.milvus import MILVUS_MAX_TOP_K, milvus_string_literal
from heavenbase.filters.ops import OP_EQ, OP_NEAR
from heavenbase.strategies import VectorIndex
from heavenbase.types import cosine_score, euclidean_score, inner_product_score
from heavenbase.execution.handlers.filters import filter_handler, normalize_filter_value

BACKEND_IDENTIFIERS = ("milvus", "milvus-lite")
MILVUS_NATIVE_OPS = {"eq": "==", "ne": "!=", "lt": "<", "lte": "<=", "gt": ">", "gte": ">=", "in": "in"}
MILVUS_ARRAY_ITEM_TYPES = {"boolean", "float", "identifier", "integer", "long-text", "medium-text", "short-text"}
MILVUS_SCAN_TYPES = ("json", "hyper-g", "vector")
MILVUS_METRICS = {
    "cosine": "COSINE",
    "l2": "L2",
    "euclidean": "L2",
    "ip": "IP",
    "inner": "IP",
    "inner_product": "IP",
    "dot": "IP",
}
SCAN_SCORERS = {
    "cosine": cosine_score,
    "l2": euclidean_score,
    "euclidean": euclidean_score,
    "ip": inner_product_score,
    "inner": inner_product_score,
    "inner_product": inner_product_score,
    "dot": inner_product_score,
}


def _lit(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    return milvus_string_literal(value)


def _array_item_type(field_schema: Any) -> Any:
    return getattr(field_schema.dtype, "item_type", None)


def _supports_array_contains(field_schema: Any) -> bool:
    item_type = _array_item_type(field_schema)
    return type(field_schema.dtype).identifier == "array" and item_type is not None and type(item_type).identifier in MILVUS_ARRAY_ITEM_TYPES


def milvus_near_handler(field_schema: Any, value: Any, ctx: HandlerContext) -> QueryFragment:
    """Return a Milvus-native vector near fragment with dimension validation.

    Args:
        field_schema (Any): Routed vector field schema.
        value (Any): Query vector validated against the field dtype.
        ctx (HandlerContext): Selected backend and strategy context.

    Returns:
        QueryFragment: Native or scan-marked Milvus vector near fragment. On
        a Milvus server, a ``top_k`` above the search maximum of 16384 is
        scan-marked, so the portable scan returns every requested neighbour.

    Raises:
        ValueError: If the query dimension, metric, or backend config is unsupported.
    """
    query = field_schema.dtype.validate(value)
    expected_dim = getattr(field_schema.dtype, "dim", None)
    if expected_dim is not None and len(query) != expected_dim:
        raise ValueError(f"Milvus near query dimension mismatch: expected {expected_dim}, got {len(query)}")

    metric = str(ctx.metric or "cosine").lower()
    scorer = SCAN_SCORERS.get(metric)
    if scorer is None:
        raise ValueError(f"Milvus near: unsupported metric {ctx.metric!r}")
    configured = str(ctx.config.get("heavenbase.backend_defaults.milvus.metric", default="COSINE")).lower()
    if configured not in MILVUS_METRICS:
        raise ValueError(f"Milvus backend config has unsupported metric {configured!r}")

    # Native search never returns a row without a vector, so the scan scorer (a filter that is not
    # pushed down) leaves it out too, under every metric: the filter cannot change the near's rows.
    def _payload(row: dict) -> float | None:
        vec = row.get(field_schema.name)
        return scorer(vec, query) if vec is not None else None

    # A Milvus server rejects a search limit above its maximum; Milvus Lite has none.
    default_max_top_k = MILVUS_MAX_TOP_K if ctx.backend_identifier == "milvus" else None
    max_top_k = getattr(getattr(ctx, "backend_instance", None), "provider_max_top_k", default_max_top_k)
    notes = {"kind": "milvus_search", "mode": "native", "top_k": ctx.top_k, "row_payload": _payload}
    if ctx.backend_identifier == "milvus-lite" and (ctx.exact or metric == "cosine"):
        notes["mode"] = "py_scan"
        notes["fallback_reason"] = "Milvus Lite uses a binary64 scan for exact or cosine searches to avoid float32 scoring errors."
    elif MILVUS_METRICS[metric] != MILVUS_METRICS[configured]:
        notes["mode"] = "py_scan"
        notes["fallback_reason"] = f"Milvus collection indexes use {MILVUS_METRICS[configured]}; {metric!r} ranks through the portable scan scorer."
    elif max_top_k is not None and ctx.top_k > max_top_k:
        notes["mode"] = "py_scan"
        notes["fallback_reason"] = f"Milvus search returns at most {max_top_k} neighbours; top_k={ctx.top_k} ranks through the portable scan scorer."
    if ctx.exact and notes["mode"] == "native" and getattr(ctx.backend_instance, "index_type", None) == "FLAT":
        notes["exact"] = True  # The executor revalidates the live index before searching.
    return QueryFragment(
        ctx.backend,
        "near",
        field_schema.name,
        {"vector": query, "top_k": ctx.top_k, "metric": metric, **({"exact": True} if ctx.exact else {})},
        returns=["score"],
        notes=notes,
    )


def milvus_filter_handler(op: str):
    """Return a Milvus scalar or array filter handler for one operation.

    Args:
        op (str): Filter operation identifier.

    Returns:
        Callable[[Any, Any, HandlerContext], QueryFragment]: Compiler that
        produces native or scan-marked Milvus filter fragments.
    """

    def _compile(field_schema: Any, value: Any, ctx: HandlerContext) -> QueryFragment:
        value = normalize_filter_value(field_schema.dtype, op, value, resolver=ctx.resolver)
        py_fragment = filter_handler(op)(field_schema, value, ctx)
        if not field_schema.required or value is None or (op == "in" and (not value or any(item is None for item in value))):
            return py_fragment.mark_scan(ctx.fallback_reason(op, type(field_schema.dtype).identifier))
        if op == "array_contains" and _supports_array_contains(field_schema):
            item_type = _array_item_type(field_schema)
            payload = f"ARRAY_CONTAINS({field_schema.name}, {_lit(item_type.validate(value))})"
            return QueryFragment(
                ctx.backend, op, field_schema.name, payload, notes={"kind": "milvus_filter", "mode": "native", "row_payload": py_fragment.payload}
            )
        if op not in MILVUS_NATIVE_OPS or type(field_schema.dtype).identifier in (*MILVUS_SCAN_TYPES, "array"):
            reason = ctx.fallback_reason(op, type(field_schema.dtype).identifier)
            return py_fragment.mark_scan(reason)
        if op == "in":
            if not isinstance(value, (list, tuple, set)):
                raise TypeError("in filter expects a list, tuple, or set")
            payload = f"{field_schema.name} in [{', '.join(_lit(item) for item in value)}]"
        else:
            payload = f"{field_schema.name} {MILVUS_NATIVE_OPS[op]} {_lit(value)}"
        return QueryFragment(ctx.backend, op, field_schema.name, payload, notes={"kind": "milvus_filter", "mode": "native", "row_payload": py_fragment.payload})

    return _compile


def register_reasons(fallback: Any, _unsupported: Any, *, resolver: Any = None) -> None:
    """Register Milvus-specific fallback reasons.

    Args:
        fallback: Reason registry receiving fallback explanations.
        _unsupported: Unsupported-operation registry (unused for Milvus).
        resolver (Any | None): Optional module resolver for backend discovery.

    Returns:
        None: Fallback reasons are registered in place.
    """
    for backend_identifier in BACKEND_IDENTIFIERS:
        fallback.register(backend_identifier, "*", "json", "*", "Milvus does not support Json field native filtering; using Python scan fallback.")
        fallback.register(backend_identifier, "*", "hyper-g", "*", "Milvus does not support HyperG field native filtering; using Python scan fallback.")


def register_handlers(registry: Any, ctx: Any) -> None:
    """Register Milvus-native handlers.

    Args:
        registry: Workspace handler registry receiving descriptors.
        ctx: Shared Registry-derived handler vocabulary.

    Returns:
        None: The supplied workspace registry is updated in place.
    """
    for backend_identifier in BACKEND_IDENTIFIERS:
        for dtype in ctx.filter_types:
            for strategy in ctx.filter_strategies:
                for op in ctx.filter_ops(dtype):
                    ctx.handler(dtype, op, backend_identifier, strategy, milvus_filter_handler(op)).register(registry, replace=True)
        for dtype in ctx.vector_types:
            ctx.handler(dtype, OP_NEAR, backend_identifier, VectorIndex.identifier, milvus_near_handler).register(registry, replace=True)
            ctx.handler(dtype, OP_EQ, backend_identifier, VectorIndex.identifier, milvus_filter_handler(OP_EQ)).register(registry, replace=True)
