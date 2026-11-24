"""ChromaDB-native operation compilers."""

from __future__ import annotations

__all__ = [
    "chroma_filter_handler",
    "chroma_near_handler",
    "register_handlers",
    "register_reasons",
]

from heavenbase.utils import Any

from heavenbase.execution.contracts import HandlerContext
from heavenbase.execution.contracts import QueryFragment
from heavenbase.filters.ops import OP_EQ, OP_NEAR
from heavenbase.strategies import InlineColumn, VectorIndex
from heavenbase.types import cosine_score, euclidean_score, inner_product_score
from heavenbase.execution.handlers.filters import filter_handler, normalize_filter_value
from heavenbase.backends.vector._metadata import INT64_EXACT_OPERANDS, exact_operand

BACKEND_IDENTIFIER = "chroma"
CHROMA_NATIVE_OPS = {"eq": "$eq", "ne": "$ne", "lt": "$lt", "lte": "$lte", "gt": "$gt", "gte": "$gte", "in": "$in"}
CHROMA_SCALAR_TYPES = (str, int, float, bool)
CHROMA_SCAN_TYPES = ("json", "hyper-g", "vector")
# Near metric -> Chroma ``hnsw:space``.
CHROMA_SPACES = {
    "cosine": "cosine",
    "l2": "l2",
    "euclidean": "l2",
    "ip": "ip",
    "inner": "ip",
    "inner_product": "ip",
    "dot": "ip",
}
SCAN_SCORERS = {"cosine": cosine_score, "l2": euclidean_score, "ip": inner_product_score}

# Array item types whose non-null items share one scalar type, so the Chroma metadata list of such
# an Array holds every item but ``None``. ``$contains`` on another Array would miss a row whose value
# Chroma stores only as encoded JSON, such as a mixed untyped Array.
CHROMA_LIST_ITEM_TYPES = (
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
)


def _is_scalar(value: Any) -> bool:
    return isinstance(value, CHROMA_SCALAR_TYPES)


def _is_native_list_field(field_schema: Any) -> bool:
    item_type = getattr(field_schema.dtype, "item_type", None)
    return type(field_schema.dtype).identifier == "array" and item_type is not None and type(item_type).identifier in CHROMA_LIST_ITEM_TYPES


def chroma_near_handler(field_schema: Any, value: Any, ctx: HandlerContext) -> QueryFragment:
    """Return a Chroma vector near fragment with dimension validation.

    A Chroma collection indexes one embedding field with one distance space.
    A near targeting that field and space searches natively. A near on a
    different field or metric uses the portable scan scorer. A scan uses the
    requested field and metric and leaves rows without the requested vector out.

    Args:
        field_schema (Any): Routed vector field schema.
        value (Any): Query vector validated against the field dtype.
        ctx (HandlerContext): Selected backend and strategy context.

    Returns:
        QueryFragment: Native or scan-marked Chroma vector near fragment.

    Raises:
        ValueError: If the query dimension, metric, or backend config is unsupported.
    """
    query = field_schema.dtype.validate(value)
    expected_dim = getattr(field_schema.dtype, "dim", None)
    if expected_dim is not None and len(query) != expected_dim:
        raise ValueError(f"Chroma near query dimension mismatch: expected {expected_dim}, got {len(query)}")
    space = CHROMA_SPACES.get(str(ctx.metric or "cosine").lower())
    if space is None:
        raise ValueError(f"Chroma near: unsupported metric {ctx.metric!r}; use 'cosine', 'l2'/'euclidean', or 'ip'/'inner_product'")
    configured = str(ctx.config.get("heavenbase.backend_defaults.chroma.metric", default="cosine")).lower()
    if configured not in CHROMA_SPACES.values():
        raise ValueError(f"Chroma backend config has unsupported metric {configured!r}")
    # An existing collection keeps the space it was created with, whatever the config says now.
    backend = getattr(ctx, "backend_instance", None)
    near_collection = getattr(backend, "_near_collection", None)
    near_space = getattr(backend, "near_space", None)
    near_vector_field = getattr(backend, "near_vector_field", None)
    entity_id = getattr(ctx, "entity_id", None)
    collection = None
    if callable(near_collection) and entity_id is not None:
        collection_space, collection = near_collection(entity_id)
    else:
        collection_space = str(near_space(entity_id)).lower() if callable(near_space) and entity_id is not None else configured
    scorer = SCAN_SCORERS[space]

    # Native search never returns a row without a vector, so the scan scorer (a filter that is not
    # pushed down) leaves it out too, under every metric: the filter cannot change the near's rows.
    def _payload(row: dict) -> float | None:
        vec = row.get(field_schema.name)
        return scorer(vec, query) if vec is not None else None

    notes = {"kind": "chroma_search", "mode": "native", "top_k": ctx.top_k, "row_payload": _payload}
    if callable(near_vector_field) and entity_id is not None and near_vector_field(entity_id) != field_schema.name:
        notes["mode"] = "py_scan"
        notes["fallback_reason"] = f"The Chroma collection does not index {field_schema.name!r} as its embedding. This near uses the portable scorer."
    elif space != collection_space:
        notes["mode"] = "py_scan"
        notes["fallback_reason"] = f"The Chroma collection uses the {collection_space!r} space; {ctx.metric!r} ranks through the portable scan scorer."
    payload = {"vector": query, "top_k": ctx.top_k, "metric": space}
    if collection is not None:
        payload.update(collection=collection, collection_space=collection_space)
    return QueryFragment(
        ctx.backend,
        "near",
        field_schema.name,
        payload,
        returns=["score"],
        notes=notes,
    )


def chroma_filter_handler(op: str):
    """Return a Chroma metadata filter handler for one operation.

    Args:
        op (str): Filter operation identifier.

    Returns:
        Callable[[Any, Any, HandlerContext], QueryFragment]: Compiler that
        produces native or scan-marked Chroma metadata fragments.
    """

    def _compile(field_schema: Any, value: Any, ctx: HandlerContext) -> QueryFragment:
        value = normalize_filter_value(field_schema.dtype, op, value, resolver=ctx.resolver)
        py_fragment = filter_handler(op)(field_schema, value, ctx)
        if not exact_operand(value, INT64_EXACT_OPERANDS):
            # Chroma clamps an integer outside the signed 64-bit range and stores one as its
            # nearest float, which it compares with any number within 2**10 of 2**63 as equal;
            # the scan predicate compares the exact numbers.
            return py_fragment.mark_scan()
        if op == "array_contains" and _is_native_list_field(field_schema) and _is_scalar(value):
            payload = {field_schema.name: {"$contains": value}}
            return QueryFragment(
                ctx.backend, op, field_schema.name, payload, notes={"kind": "chroma_where", "mode": "native", "row_payload": py_fragment.payload}
            )
        if op not in CHROMA_NATIVE_OPS or type(field_schema.dtype).identifier in (*CHROMA_SCAN_TYPES, "array"):
            reason = ctx.fallback_reason(op, type(field_schema.dtype).identifier)
            return py_fragment.mark_scan(reason)
        if op == "in" and not isinstance(value, (list, tuple, set)):
            raise TypeError("in filter expects a list, tuple, or set")
        if value is None or (op == "in" and (not value or any(item is None for item in value))):
            # Chroma metadata filters reject null operands and an empty $in list; the scan predicate tests for a
            # missing value and matches no row for an empty list.
            return py_fragment.mark_scan()
        if op == "in":
            payload = {field_schema.name: {"$in": list(value)}}
        else:
            payload = {field_schema.name: {CHROMA_NATIVE_OPS[op]: value}}
        return QueryFragment(ctx.backend, op, field_schema.name, payload, notes={"kind": "chroma_where", "mode": "native", "row_payload": py_fragment.payload})

    return _compile


def register_reasons(fallback: Any, _unsupported: Any, *, resolver: Any = None) -> None:
    """Register ChromaDB-specific fallback reasons.

    Args:
        fallback: Reason registry receiving fallback explanations.
        _unsupported: Unsupported-operation registry (unused for ChromaDB).
        resolver (Any | None): Optional module resolver for backend discovery.

    Returns:
        None: Fallback reasons are registered in place.
    """
    fallback.register(BACKEND_IDENTIFIER, "*", "json", "*", "ChromaDB does not support Json field native filtering; using Python scan fallback.")
    fallback.register(BACKEND_IDENTIFIER, "*", "hyper-g", "*", "ChromaDB does not support HyperG field native filtering; using Python scan fallback.")
    fallback.register(BACKEND_IDENTIFIER, "*", "vector", InlineColumn.identifier, f"ChromaDB requires {VectorIndex.__name__} strategy for vector operations.")


def register_handlers(registry: Any, ctx: Any) -> None:
    """Register ChromaDB-native handlers.

    Args:
        registry: Workspace handler registry receiving descriptors.
        ctx: Shared Registry-derived handler vocabulary.

    Returns:
        None: The supplied workspace registry is updated in place.
    """
    for dtype in ctx.filter_types:
        for strategy in ctx.filter_strategies:
            for op in ctx.filter_ops(dtype):
                ctx.handler(dtype, op, BACKEND_IDENTIFIER, strategy, chroma_filter_handler(op)).register(registry, replace=True)
    for dtype in ctx.vector_types:
        ctx.handler(dtype, OP_NEAR, BACKEND_IDENTIFIER, VectorIndex.identifier, chroma_near_handler).register(registry, replace=True)
        ctx.handler(dtype, OP_EQ, BACKEND_IDENTIFIER, VectorIndex.identifier, chroma_filter_handler(OP_EQ)).register(registry, replace=True)
