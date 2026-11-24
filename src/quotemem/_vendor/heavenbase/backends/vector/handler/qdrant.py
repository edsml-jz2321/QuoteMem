"""Qdrant-native vector and payload filter compilers."""

from __future__ import annotations

__all__ = ["qdrant_filter_handler", "qdrant_near_handler", "register_handlers", "register_reasons"]

from heavenbase.execution.contracts import HandlerContext, QueryFragment
from heavenbase.execution.handlers.filters import filter_handler, normalize_filter_value
from heavenbase.filters.ops import OP_EQ, OP_NEAR
from heavenbase.strategies import VectorIndex
from heavenbase.types import cosine_score, euclidean_score, inner_product_score
from heavenbase.utils import Any

BACKEND_IDENTIFIER = "qdrant"
QDRANT_EQ_TYPES = {
    "boolean",
    "categorical",
    "date",
    "duration",
    "float",
    "identifier",
    "integer",
    "long-text",
    "medium-text",
    "short-text",
    "timestamp",
}
QDRANT_RANGE_TYPES = {"duration", "float", "integer", "timestamp"}
SCAN_SCORERS = {
    "cosine": cosine_score,
    "l2": euclidean_score,
    "euclidean": euclidean_score,
    "ip": inner_product_score,
    "inner": inner_product_score,
    "inner_product": inner_product_score,
    "dot": inner_product_score,
}


def _match_value_supported(value: Any) -> bool:
    """Return whether Qdrant's strict MatchValue model accepts the value."""
    return type(value) in (bool, int, str)


def _match_any_supported(value: Any) -> bool:
    """Return whether Qdrant's strict homogeneous MatchAny model accepts the values."""
    if not isinstance(value, (list, tuple, set)):
        return False
    values = list(value)
    return bool(values) and (all(type(item) is int for item in values) or all(type(item) is str for item in values))


def qdrant_near_handler(field_schema: Any, value: Any, ctx: HandlerContext) -> QueryFragment:
    """Compile a cosine query to Qdrant and other metrics to an exact scan.

    Args:
        field_schema (Any): Routed vector field schema.
        value (Any): Query vector validated against the field dtype.
        ctx (HandlerContext): Selected backend and strategy context.

    Returns:
        QueryFragment: Native or scan-marked Qdrant vector near fragment.

    Raises:
        ValueError: If the vector is dimensionless, dimension-mismatched, or uses
            an unsupported metric.
    """
    query = field_schema.dtype.validate(value)
    expected_dim = getattr(field_schema.dtype, "dim", None)
    if expected_dim is None:
        raise ValueError("Qdrant native vector search requires a fixed-dimension Vector")
    if len(query) != expected_dim:
        raise ValueError(f"Qdrant near query dimension mismatch: expected {expected_dim}, got {len(query)}")
    metric = str(ctx.metric or "cosine").lower()
    scorer = SCAN_SCORERS.get(metric)
    if scorer is None:
        raise ValueError(f"Qdrant near: unsupported metric {ctx.metric!r}")

    # Native search never returns a row without a vector, so the scan scorer (a filter that is not
    # pushed down) leaves it out too, under every metric: the filter cannot change the near's rows.
    def _payload(row: dict[str, Any]) -> float | None:
        vector = row.get(field_schema.name)
        return scorer(vector, query) if vector is not None else None

    notes = {"kind": "qdrant_search", "mode": "native", "top_k": ctx.top_k, "row_payload": _payload}
    if metric != "cosine":
        notes["mode"] = "py_scan"
        notes["fallback_reason"] = f"Qdrant collections use cosine distance; {metric!r} ranks through the portable scan scorer."
    return QueryFragment(
        ctx.backend,
        OP_NEAR,
        field_schema.name,
        {"vector": query, "top_k": ctx.top_k, "metric": metric},
        returns=["score"],
        notes=notes,
    )


def _native_filter(field_schema: Any, op: str, value: Any) -> bool:
    dtype = type(field_schema.dtype).identifier
    if value is None:
        return False
    if op == "eq" and dtype in QDRANT_EQ_TYPES:
        return _match_value_supported(value)
    if op == "in" and dtype in QDRANT_EQ_TYPES:
        return _match_any_supported(value)
    if op in ("lt", "lte", "gt", "gte") and dtype in QDRANT_RANGE_TYPES:
        return True
    if op == "array_contains" and dtype == "array":
        item_type = getattr(field_schema.dtype, "item_type", None)
        return item_type is not None and type(item_type).identifier in QDRANT_EQ_TYPES and _match_value_supported(value)
    return False


def qdrant_filter_handler(op: str):
    """Return a parity-gated Qdrant payload filter compiler.

    Args:
        op (str): Filter operation identifier.

    Returns:
        Callable[[Any, Any, HandlerContext], QueryFragment]: Compiler that
        produces native or scan-marked Qdrant payload fragments.
    """

    def _compile(field_schema: Any, value: Any, ctx: HandlerContext) -> QueryFragment:
        normalized = normalize_filter_value(field_schema.dtype, op, value, resolver=ctx.resolver)
        py_fragment = filter_handler(op)(field_schema, normalized, ctx)
        if not _native_filter(field_schema, op, normalized):
            return py_fragment.mark_scan(ctx.fallback_reason(op, type(field_schema.dtype).identifier))
        payload_value = list(normalized) if op == "in" else normalized
        return QueryFragment(
            ctx.backend,
            op,
            field_schema.name,
            {"op": op, "field": field_schema.name, "value": payload_value},
            notes={"kind": "qdrant_filter", "mode": "native", "row_payload": py_fragment.payload},
        )

    return _compile


def register_reasons(fallback: Any, _unsupported: Any, *, resolver: Any = None) -> None:
    """Register explicit Qdrant portable-fallback reasons.

    Args:
        fallback: Reason registry receiving fallback explanations.
        _unsupported: Unsupported-operation registry (unused for Qdrant).
        resolver (Any | None): Optional module resolver for backend discovery.

    Returns:
        None: Fallback reasons are registered in place.
    """
    for dtype in ("json", "hyper-g", "vector"):
        fallback.register(
            BACKEND_IDENTIFIER,
            "*",
            dtype,
            "*",
            f"Qdrant does not provide an exact native {dtype} filter for HeavenBase semantics; using Python scan fallback.",
        )


def register_handlers(registry: Any, ctx: Any) -> None:
    """Register exact Qdrant filter and vector-near handlers.

    Args:
        registry: Workspace handler registry receiving descriptors.
        ctx: Shared Registry-derived handler vocabulary.

    Returns:
        None: The supplied workspace registry is updated in place.
    """
    for dtype in ctx.filter_types:
        for strategy in ctx.filter_strategies:
            for op in ctx.filter_ops(dtype):
                ctx.handler(dtype, op, BACKEND_IDENTIFIER, strategy, qdrant_filter_handler(op)).register(registry, replace=True)
    for dtype in ctx.vector_types:
        ctx.handler(dtype, OP_NEAR, BACKEND_IDENTIFIER, VectorIndex.identifier, qdrant_near_handler).register(registry, replace=True)
        ctx.handler(dtype, OP_EQ, BACKEND_IDENTIFIER, VectorIndex.identifier, qdrant_filter_handler(OP_EQ)).register(registry, replace=True)
