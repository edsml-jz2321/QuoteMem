"""PgVector-native operation compilers."""

from __future__ import annotations

__all__ = [
    "pgvector_near_handler",
    "register_handlers",
    "register_reasons",
]

from heavenbase.utils import Any

from heavenbase.execution.contracts import HandlerContext
from heavenbase.execution.contracts import QueryFragment
from heavenbase.filters.ops import OP_NEAR
from heavenbase.strategies import VectorIndex
from heavenbase.types import cosine_score, euclidean_score, inner_product_score

BACKEND_IDENTIFIER = "pgvector"
SCAN_SCORERS = {
    "cosine": cosine_score,
    "l2": euclidean_score,
    "euclidean": euclidean_score,
    "ip": inner_product_score,
    "inner": inner_product_score,
    "inner_product": inner_product_score,
    "dot": inner_product_score,
}


def pgvector_near_handler(field_schema: Any, value: Any, ctx: HandlerContext) -> QueryFragment:
    """Return a pgvector-native vector near fragment with dimension validation.

    pgvector orders natively by the requested metric's distance operator. A
    near that runs as a scan instead (a filter that is not SQL) ranks with the
    same metric through ``notes['row_payload']``, which leaves a row without a
    vector out, as native search does.

    Args:
        field_schema (Any): Routed vector field schema.
        value (Any): Query vector validated against the field dtype.
        ctx (HandlerContext): Selected backend and strategy context.

    Returns:
        QueryFragment: Native pgvector vector near fragment.

    Raises:
        ValueError: If the query dimension or metric is unsupported.
    """
    query = field_schema.dtype.validate(value)
    expected_dim = getattr(field_schema.dtype, "dim", None)
    if expected_dim is not None and len(query) != expected_dim:
        raise ValueError(f"PgVector near query dimension mismatch: expected {expected_dim}, got {len(query)}")
    metric = str(ctx.metric or "cosine").lower()
    scorer = SCAN_SCORERS.get(metric)
    if scorer is None:
        raise ValueError(f"PgVector near: unsupported metric {ctx.metric!r}; use 'cosine', 'l2'/'euclidean', or 'ip'/'inner_product'")

    # Native search never returns a row without a vector, so the scan scorer (a filter that is not
    # pushed down) leaves it out too, under every metric: the filter cannot change the near's rows.
    def _payload(row: dict) -> float | None:
        vec = row.get(field_schema.name)
        return scorer(vec, query) if vec is not None else None

    return QueryFragment(
        ctx.backend,
        "near",
        field_schema.name,
        {"vector": query, "top_k": ctx.top_k, "metric": metric},
        returns=["score"],
        notes={"kind": "pgvector_search", "mode": "native", "top_k": ctx.top_k, "row_payload": _payload},
    )


def register_reasons(_fallback: Any, _unsupported: Any, *, resolver: Any = None) -> None:
    """Register pgvector-specific fallback reasons (none currently).

    Args:
        _fallback: Reason registry (unused because pgvector registers no reasons).
        _unsupported: Unsupported-operation registry (unused for pgvector).
        resolver (Any | None): Optional module resolver for backend discovery.

    Returns:
        None: This function does not register any fallback reasons.
    """


def register_handlers(registry: Any, ctx: Any) -> None:
    """Register PgVector-native handlers.

    Args:
        registry: Workspace handler registry receiving descriptors.
        ctx: Shared Registry-derived handler vocabulary.

    Returns:
        None: The supplied workspace registry is updated in place.
    """
    for dtype in ctx.vector_types:
        ctx.handler(dtype, OP_NEAR, BACKEND_IDENTIFIER, VectorIndex.identifier, pgvector_near_handler).register(registry, replace=True)
