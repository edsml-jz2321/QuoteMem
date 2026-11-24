"""LanceDB-native operation compilers."""

from __future__ import annotations

__all__ = [
    "lance_near_handler",
    "lance_filter_handler",
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

BACKEND_IDENTIFIER = "lance"
LANCE_NATIVE_OPS = {"eq": "=", "ne": "!=", "lt": "<", "lte": "<=", "gt": ">", "gte": ">=", "in": "IN"}
LANCE_SCAN_TYPES = ("array", "json", "hyper-g", "vector")
# Requested near metric -> LanceDB distance type. ``dot`` is LanceDB's own name for the inner product.
LANCE_METRICS = {
    "cosine": "cosine",
    "l2": "l2",
    "euclidean": "l2",
    "ip": "dot",
    "inner": "dot",
    "inner_product": "dot",
    "dot": "dot",
}
SCAN_SCORERS = {"cosine": cosine_score, "l2": euclidean_score, "dot": inner_product_score}


def _lit(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, (int, float)):
        return repr(value)
    text = str(value).replace("'", "''")
    return f"'{text}'"


def lance_near_handler(field_schema: Any, value: Any, ctx: HandlerContext) -> QueryFragment:
    """Return a LanceDB-native vector near fragment with dimension validation.

    LanceDB searches natively with the requested metric. A near that runs as a
    scan instead (a filter LanceDB cannot push down) ranks with the same
    metric through ``notes['row_payload']``, which leaves a row without a
    vector out, as native search does.

    Args:
        field_schema (Any): Routed vector field schema.
        value (Any): Query vector validated against the field dtype.
        ctx (HandlerContext): Selected backend and strategy context.

    Returns:
        QueryFragment: Native LanceDB vector near fragment.

    Raises:
        ValueError: If the query dimension or metric is unsupported.
    """
    query = field_schema.dtype.validate(value)
    expected_dim = getattr(field_schema.dtype, "dim", None)
    if expected_dim is not None and len(query) != expected_dim:
        raise ValueError(f"LanceDB near query dimension mismatch: expected {expected_dim}, got {len(query)}")
    distance = LANCE_METRICS.get(str(ctx.metric or "cosine").lower())
    if distance is None:
        raise ValueError(f"LanceDB near: unsupported metric {ctx.metric!r}; use 'cosine', 'l2'/'euclidean', or 'ip'/'inner_product'")
    scorer = SCAN_SCORERS[distance]

    # Native search never returns a row without a vector, so the scan scorer (a filter that is not
    # pushed down) leaves it out too, under every metric: the filter cannot change the near's rows.
    def _payload(row: dict) -> float | None:
        vec = row.get(field_schema.name)
        return scorer(vec, query) if vec is not None else None

    return QueryFragment(
        ctx.backend,
        "near",
        field_schema.name,
        {"vector": query, "top_k": ctx.top_k, "metric": distance},
        returns=["score"],
        notes={"kind": "lance_search", "mode": "native", "top_k": ctx.top_k, "row_payload": _payload},
    )


def _where(name: str, op: str, value: Any) -> str:
    """Compile one portable predicate to a LanceDB where clause.

    SQL compares NULL as unknown, so ``= NULL`` and ``IN (NULL)`` match
    nothing, ``!= value`` drops a row without the field, and ``NOT`` over an
    unknown clause drops that row too. The portable predicate reads a missing
    value as ``None``: ``eq`` and ``in`` with ``None`` match it, ``ne`` keeps
    it, and a comparison with a value is false for it. Every clause compiled
    here is therefore true or false, never unknown, so ``NOT``, ``AND`` and
    ``OR`` combine them like the portable scan does.

    Args:
        name (str): Column name.
        op (str): Portable operation in ``LANCE_NATIVE_OPS``.
        value (Any): Normalized operand, or a list of operands for ``in``.

    Returns:
        str: LanceDB where clause.
    """
    if op == "in":
        values = list(value)
        concrete = [item for item in values if item is not None]
        clauses = [f"{name} IS NULL"] if len(concrete) < len(values) else []
        if concrete:
            clauses.append(f"({name} IS NOT NULL AND {name} IN ({', '.join(_lit(item) for item in concrete)}))")
        if not clauses:
            return "false"
        return f"({' OR '.join(clauses)})" if len(clauses) > 1 else clauses[0]
    if op == "eq" and value is None:
        return f"{name} IS NULL"
    if op == "ne":
        return f"{name} IS NOT NULL" if value is None else f"({name} IS NULL OR {name} != {_lit(value)})"
    return f"({name} IS NOT NULL AND {name} {LANCE_NATIVE_OPS[op]} {_lit(value)})"


def lance_filter_handler(op: str):
    """Return a LanceDB scalar filter handler for one operation.

    Args:
        op (str): Filter operation identifier.

    Returns:
        Callable[[Any, Any, HandlerContext], QueryFragment]: Compiler that
        produces native or scan-marked LanceDB filter fragments.
    """

    def _compile(field_schema: Any, value: Any, ctx: HandlerContext) -> QueryFragment:
        value = normalize_filter_value(field_schema.dtype, op, value, resolver=ctx.resolver)
        py_fragment = filter_handler(op)(field_schema, value, ctx)
        if op not in LANCE_NATIVE_OPS or type(field_schema.dtype).identifier in LANCE_SCAN_TYPES:
            reason = ctx.fallback_reason(op, type(field_schema.dtype).identifier)
            return py_fragment.mark_scan(reason)
        payload = _where(field_schema.name, op, value)
        return QueryFragment(ctx.backend, op, field_schema.name, payload, notes={"kind": "lance_where", "mode": "native", "row_payload": py_fragment.payload})

    return _compile


def register_reasons(fallback: Any, _unsupported: Any, *, resolver: Any = None) -> None:
    """Register LanceDB-specific fallback reasons.

    Args:
        fallback: Reason registry receiving fallback explanations.
        _unsupported: Unsupported-operation registry (unused for LanceDB).
        resolver (Any | None): Optional module resolver for backend discovery.

    Returns:
        None: Fallback reasons are registered in place.
    """
    fallback.register(BACKEND_IDENTIFIER, "*", "array", "*", "LanceDB does not support Array field native filtering; using Python scan fallback.")
    fallback.register(BACKEND_IDENTIFIER, "*", "json", "*", "LanceDB does not support Json field native filtering; using Python scan fallback.")
    fallback.register(BACKEND_IDENTIFIER, "*", "hyper-g", "*", "LanceDB does not support HyperG field native filtering; using Python scan fallback.")


def register_handlers(registry: Any, ctx: Any) -> None:
    """Register LanceDB-native handlers.

    Args:
        registry: Workspace handler registry receiving descriptors.
        ctx: Shared Registry-derived handler vocabulary.

    Returns:
        None: The supplied workspace registry is updated in place.
    """
    for dtype in ctx.filter_types:
        for strategy in ctx.filter_strategies:
            for op in ctx.filter_ops(dtype):
                ctx.handler(dtype, op, BACKEND_IDENTIFIER, strategy, lance_filter_handler(op)).register(registry, replace=True)
    for dtype in ctx.vector_types:
        ctx.handler(dtype, OP_NEAR, BACKEND_IDENTIFIER, VectorIndex.identifier, lance_near_handler).register(registry, replace=True)
        ctx.handler(dtype, OP_EQ, BACKEND_IDENTIFIER, VectorIndex.identifier, lance_filter_handler(OP_EQ)).register(registry, replace=True)
