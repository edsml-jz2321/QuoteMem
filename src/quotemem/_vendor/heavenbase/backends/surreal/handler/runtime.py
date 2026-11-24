"""Native SurrealQL filter compilers for the SurrealDB entity/row backend.

The SurrealDB backend stores one record per object, so by default every filter runs through
the portable Python scan fallback. This module pushes the parity-exact subset down into a
SurrealQL ``WHERE`` clause (index-backed by the per-field indexes ``ensure`` defines):

- ``eq`` / ``in`` on any indexable scalar field — a ``None`` operand and the inline compound
  types (json / array / hyper-g / vector) stay on scan.
- ``lt`` / ``lte`` / ``gt`` / ``gte`` on a numeric or ISO-date field (lexical order ==
  chronological), where SurrealDB drops rows whose field is NONE/absent, matching the scan
  over valued rows.
- ``contains`` / ``match`` / ``starts_with`` / ``ends_with`` on a text field — pushed to the
  SurrealDB ``string::*`` functions (NONE-guarded, ``match`` case-insensitive), parity-exact with
  HeavenBase's substring predicates (BM25 ``@@`` is a token match, not a substring, so it is not used).

A pure cosine vector ``near`` also pushes down to native HNSW KNN (see ``surreal_near_handler``); a
non-cosine metric or a near co-located with a filter degrades to the metric-matched scan.

``ne``, logical ``not``, ``like`` / ``wildcard`` / ``regex``, and a ``None`` operand degrade to scan.
The portable predicate is always carried in ``notes['row_payload']`` so mixed boolean trees can
compose native and scan leaves; the backend re-uses the exact stored operand form, so the
normalized value binds directly as a SurrealQL parameter.
"""

from __future__ import annotations

__all__ = [
    "surreal_filter_handler",
    "surreal_near_handler",
    "register_handlers",
    "register_reasons",
]

from heavenbase.utils import Any

from heavenbase.execution.contracts import HandlerContext
from heavenbase.execution.contracts import QueryFragment
from heavenbase.strategies import VectorIndex
from heavenbase.types import cosine_score, euclidean_score, inner_product_score
from heavenbase.types.storage import stores_as_integer
from heavenbase.filters.ops import OP_NEAR
from heavenbase.execution.handlers.filters import filter_handler, normalize_filter_value

BACKEND_IDENTIFIER = "surreal"
SURREAL_NATIVE_OPS = ("eq", "in", "lt", "lte", "gt", "gte")  # ne/like/wildcard/regex ops stay on scan
SURREAL_ORDER_OPS = ("lt", "lte", "gt", "gte")
SURREAL_ORDER_TYPES = ("integer", "float", "timestamp", "duration", "date")  # numeric, or ISO date (lexical == chronological)
# Substring/affix text ops push to SurrealQL string functions (string::contains/starts_with/ends_with),
# which match HeavenBase's substring predicates exactly. SurrealDB's BM25 @@ is a token match, not a
# substring, so it would diverge from the scan and is deliberately NOT used (see ADR 0007).
SURREAL_TEXT_OPS = ("contains", "match", "starts_with", "ends_with")
SURREAL_TEXT_TYPES = ("identifier", "short-text", "medium-text", "long-text")
# Requested-metric -> scan-fallback scorer. SurrealDB's native KNN path is cosine-only (see
# SurrealBackend._run_near / the cosine HNSW index), so a non-cosine near degrades to the Python scan;
# the carried scorer must match the REQUESTED metric or the fallback ranks by the wrong distance.
SCAN_SCORERS = {
    "cosine": cosine_score,
    "l2": euclidean_score,
    "euclidean": euclidean_score,
    "ip": inner_product_score,
    "inner": inner_product_score,
    "inner_product": inner_product_score,
    "dot": inner_product_score,
}


def _native_ok(field_schema: Any, op: str, value: Any, ctx: HandlerContext) -> bool:
    """Whether (field, op, value) compiles to a SurrealQL WHERE whose result equals the scan.

    A ``None`` operand and the inline compound types always degrade to scan; order ops push down
    only on an ordered scalar dtype (numeric or ISO-date), where SurrealDB's NONE exclusion
    matches the Python scan for every valued row.
    """
    ident = type(field_schema.dtype).identifier
    if not ctx.indexable(field_schema.dtype) or value is None:
        return False
    if op == "eq":
        return True
    if op == "in":
        return isinstance(value, (list, tuple, set)) and not any(item is None for item in value)
    if op in SURREAL_ORDER_OPS:
        return ident in SURREAL_ORDER_TYPES or stores_as_integer(field_schema.dtype)
    if op in SURREAL_TEXT_OPS:
        # A non-empty string operand on a text field maps to string::* with NONE-guard parity; an empty
        # operand (degenerate substring) and non-text dtypes stay on the scan.
        return ident in SURREAL_TEXT_TYPES and isinstance(value, str) and value != ""
    return False


def surreal_filter_handler(op: str):
    """Return a SurrealDB filter handler that pushes the parity-safe subset to SurrealQL, else scans.

    Args:
        op (str): Filter operation identifier.

    Returns:
        Callable[[Any, Any, HandlerContext], QueryFragment]: Compiler that
        produces native or scan-marked SurrealQL fragments.
    """

    def _compile(field_schema: Any, value: Any, ctx: HandlerContext) -> QueryFragment:
        value = normalize_filter_value(field_schema.dtype, op, value, resolver=ctx.resolver)
        py_fragment = filter_handler(op)(field_schema, value, ctx)  # portable scan predicate (the fallback)
        if not _native_ok(field_schema, op, value, ctx):
            reason = ctx.fallback_reason(op, type(field_schema.dtype).identifier)
            return py_fragment.mark_scan(reason)
        if op == "in":
            native_value = list(value)
        elif op == "match":
            native_value = value.lower()  # case-insensitive substring: lowercase the operand; the leaf lowercases the field
        else:
            native_value = value
        payload = {"op": op, "field": field_schema.name, "value": native_value}
        return QueryFragment(
            ctx.backend,
            op,
            field_schema.name,
            payload,
            notes={"kind": "surreal_where", "mode": "native", "row_payload": py_fragment.payload},
        )

    return _compile


def surreal_near_handler(field_schema: Any, value: Any, ctx: HandlerContext) -> QueryFragment:
    """Return a native SurrealDB vector ``near`` fragment (HNSW/brute-force KNN), with a metric-correct scan scorer.

    The backend runs a pure cosine ``near`` through SurrealDB's KNN operator; a near co-located with a
    filter or any non-cosine metric degrades to the portable scan, which scores with the metric-matched
    scorer carried in ``notes['row_payload']`` (cosine -> ``cosine_score``, l2/euclidean ->
    ``euclidean_score``, ip/inner_product -> ``inner_product_score``).

    Args:
        field_schema (Any): Routed vector field schema.
        value (Any): Query vector validated against the field dtype.
        ctx (HandlerContext): Selected backend and strategy context.

    Returns:
        QueryFragment: Native or scan-marked SurrealDB vector near fragment.

    Raises:
        ValueError: If the query dimension or metric is unsupported.
    """
    query = field_schema.dtype.validate(value)
    expected_dim = getattr(field_schema.dtype, "dim", None)
    if expected_dim is not None and len(query) != expected_dim:
        raise ValueError(f"SurrealDB near query dimension mismatch: expected {expected_dim}, got {len(query)}")
    metric = str(ctx.metric or "cosine").lower()
    scorer = SCAN_SCORERS.get(metric)
    if scorer is None:
        raise ValueError(f"SurrealDB near: unsupported metric {ctx.metric!r}; use 'cosine', 'l2'/'euclidean', or 'ip'/'inner_product'")

    # Native search never returns a row without a vector, so the scan scorer (a filter that is not
    # pushed down) leaves it out too, under every metric: the filter cannot change the near's rows.
    def _payload(row: dict) -> float | None:
        vec = row.get(field_schema.name)
        return scorer(vec, query) if vec is not None else None

    # SurrealDB's native KNN is cosine-only (see SurrealBackend._run_near); a non-cosine metric ranks via
    # the carried scan scorer, so report it honestly as py_scan in explain() rather than a native that
    # silently scans. A cosine near co-located with a filter still degrades at runtime, but that is the
    # near+filter route's call (combined_scan), not knowable at this leaf.
    notes = {"kind": "surreal_vector", "mode": "native", "top_k": ctx.top_k, "row_payload": _payload}
    if metric != "cosine":
        notes["mode"] = "py_scan"
        notes["fallback_reason"] = f"SurrealDB native KNN is cosine-only; {metric!r} ranks via the Python scan scorer."
    return QueryFragment(ctx.backend, "near", field_schema.name, {"vector": query, "top_k": ctx.top_k, "metric": ctx.metric}, returns=["score"], notes=notes)


def register_reasons(fallback: Any, _unsupported: Any, *, resolver: Any = None) -> None:
    """Register SurrealDB-specific fallback reasons for the inline compound types.

    Args:
        fallback: Reason registry receiving fallback explanations.
        _unsupported: Unsupported-operation registry (unused for SurrealDB).
        resolver (Any | None): Optional module resolver for backend discovery.

    Returns:
        None: Fallback reasons are registered in place.
    """
    for dtype in ("json", "array", "hyper-g"):
        fallback.register(BACKEND_IDENTIFIER, "*", dtype, "*", f"SurrealDB stores {dtype} inline; using Python scan fallback.")


def register_handlers(registry: Any, ctx: Any) -> None:
    """Register SurrealDB-native filter handlers (scalar ops) and the vector near handler.

    Args:
        registry: Workspace handler registry receiving descriptors.
        ctx: Shared Registry-derived handler vocabulary.

    Returns:
        None: The supplied workspace registry is updated in place.
    """
    for dtype in ctx.filter_types:
        for strategy in ctx.filter_strategies:
            for op in ctx.filter_ops(dtype):
                if op in SURREAL_NATIVE_OPS or op in SURREAL_TEXT_OPS:
                    ctx.handler(dtype, op, BACKEND_IDENTIFIER, strategy, surreal_filter_handler(op)).register(registry, replace=True)
    for dtype in ctx.vector_types:
        ctx.handler(dtype, OP_NEAR, BACKEND_IDENTIFIER, VectorIndex.identifier, surreal_near_handler).register(registry, replace=True)
