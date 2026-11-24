"""Native Cypher filter compilers for the Neo4j entity/row backend.

The Neo4j backend stores one node per object, so by default every filter runs through the
portable Python scan fallback. This module pushes the parity-exact subset down into a Cypher
``WHERE`` clause (index-backed by the property indexes ``ensure`` creates):

- ``eq`` / ``in`` on any indexable scalar field — NULL is excluded by both the Python predicate
  (``None == v`` -> False) and Cypher (``NULL = v`` -> NULL), so any non-``None`` operand is
  parity-exact.
- ``lt`` / ``lte`` / ``gt`` / ``gte`` on a numeric or ISO-date field (lexical order == chronological)
  — Cypher's 3-valued logic drops NULL/absent rows, so the result equals the Python scan for every
  row that has a value. On a nullable field with missing values native is in fact *more robust* than
  the scalar scan, which raises ``TypeError`` on a ``None`` operand rather than dropping the row.

``contains`` / ``starts_with`` / ``ends_with`` push down on identifier/text fields with a non-empty
string operand; the backend renders a null guard so absent values match the portable predicate.
``ne``, logical ``not``, match/LIKE/wildcard/regex ops, empty text operands, a ``None`` operand, and
the JSON-string / list stored compound types (json / array / hyper-g / vector) all degrade to scan,
because native Cypher's 3-valued NULL logic or pattern language would diverge from the Python
predicate. The portable predicate is always carried in ``notes['row_payload']`` so mixed boolean trees
can compose native and scan leaves.
"""

from __future__ import annotations

__all__ = [
    "neo4j_filter_handler",
    "neo4j_near_handler",
    "register_handlers",
    "register_reasons",
]

from heavenbase.utils import Any

from heavenbase.execution.contracts import HandlerContext
from heavenbase.execution.contracts import QueryFragment
from heavenbase.strategies import GraphEdge, VectorIndex
from heavenbase.types import HyperG, cosine_score, euclidean_score, inner_product_score
from heavenbase.types.storage import stores_as_integer
from heavenbase.filters.ops import OP_NEAR
from heavenbase.execution.handlers.filters import filter_handler, normalize_filter_value

BACKEND_IDENTIFIER = "neo4j"
# Requested-metric -> scan-fallback scorer. Neo4j's native vector index is cosine-only (see
# Neo4jBackend.ensure), so a non-cosine near always degrades to the Python scan; the carried scorer
# must match the REQUESTED metric or the fallback ranks/scores by the wrong distance.
SCAN_SCORERS = {
    "cosine": cosine_score,
    "l2": euclidean_score,
    "euclidean": euclidean_score,
    "ip": inner_product_score,
    "inner": inner_product_score,
    "inner_product": inner_product_score,
    "dot": inner_product_score,
}
CYPHER_NATIVE_OPS = ("eq", "in", "lt", "lte", "gt", "gte")  # scalar pushdown candidates (ne/not stay on scan)
CYPHER_ORDER_OPS = ("lt", "lte", "gt", "gte")
CYPHER_ORDER_TYPES = ("integer", "float", "timestamp", "duration", "date")  # numeric, or ISO date (lexical == chronological)
CYPHER_TEXT_OPS = ("contains", "starts_with", "ends_with")
CYPHER_TEXT_TYPES = ("identifier", "short-text", "medium-text", "long-text")
CYPHER_PUSH_OPS = CYPHER_NATIVE_OPS + CYPHER_TEXT_OPS
# Logical types stored as a list / JSON string -> no native WHERE pushdown. Shared with backend index
# decisions through Backend.indexable so WHERE pushdown and DDL cannot drift.


def _native_ok(field_schema: Any, op: str, value: Any, ctx: HandlerContext) -> bool:
    """Whether (field, op, value) compiles to a Cypher WHERE whose result equals the Python scan.

    See the module docstring for the parity argument. A ``None`` operand and the JSON-string /
    list compound types always degrade to scan; order ops push down only on an ordered scalar
    dtype (numeric or ISO-date), where Cypher's NULL exclusion matches the scan for valued rows.
    Text substring/affix ops push down only for non-empty string operands, avoiding the ``"" in None``
    edge case in the portable ``contains`` predicate.
    """
    ident = type(field_schema.dtype).identifier
    if not ctx.indexable(field_schema.dtype) or value is None:
        return False
    if op == "eq":
        return True
    if op == "in":
        return isinstance(value, (list, tuple, set)) and not any(item is None for item in value)
    if op in CYPHER_ORDER_OPS:
        return ident in CYPHER_ORDER_TYPES or stores_as_integer(field_schema.dtype)
    if op in CYPHER_TEXT_OPS:
        return ident in CYPHER_TEXT_TYPES and isinstance(value, str) and value != ""
    return False  # ne / match / pattern ops -> Python scan


def neo4j_filter_handler(op: str):
    """Return a Neo4j filter handler that pushes the parity-safe subset to Cypher, else scans.

    Args:
        op (str): Filter operation identifier.

    Returns:
        Callable[[Any, Any, HandlerContext], QueryFragment]: Compiler that
        produces native or scan-marked Cypher fragments.
    """

    def _compile(field_schema: Any, value: Any, ctx: HandlerContext) -> QueryFragment:
        value = normalize_filter_value(field_schema.dtype, op, value, resolver=ctx.resolver)
        py_fragment = filter_handler(op)(field_schema, value, ctx)  # portable scan predicate (the fallback)
        if not _native_ok(field_schema, op, value, ctx):
            reason = ctx.fallback_reason(op, type(field_schema.dtype).identifier)
            return py_fragment.mark_scan(reason)
        # The normalized operand is already in the stored property form (validate and _encode_props
        # coerce identically per dtype), so it binds directly as a Cypher parameter.
        payload = {"op": op, "field": field_schema.name, "value": list(value) if op == "in" else value}
        return QueryFragment(
            ctx.backend,
            op,
            field_schema.name,
            payload,
            notes={"kind": "cypher_where", "mode": "native", "row_payload": py_fragment.payload},
        )

    return _compile


def neo4j_graph_exists_handler(field_schema: Any, value: Any, ctx: HandlerContext) -> QueryFragment:
    """Compile a GraphEdge 1-hop ``exists`` to a native Cypher relationship EXISTS, with a Python-scan fallback.

    Scalar-equality predicates push to ``EXISTS { (n)-[r:rel]->(b) WHERE ... }`` (the backend resolves
    endpoint vs edge-property columns from the field's spec at render time); a predicate carrying an
    operator map or non-scalar value degrades to the portable scan over the rebuilt edge list, and the
    scan predicate is always carried in ``notes['row_payload']`` so mixed boolean trees can compose leaves.

    Args:
        field_schema (Any): Routed GraphEdge field schema.
        value (Any): HyperG existential predicate payload.
        ctx (HandlerContext): Selected backend and strategy context.

    Returns:
        QueryFragment: Native relationship-exists or scan-marked GraphEdge fragment.
    """
    normalized = normalize_filter_value(field_schema.dtype, "exists", value, resolver=ctx.resolver)
    py_fragment = filter_handler("exists")(field_schema, normalized, ctx)
    predicates = _native_edge_exists_predicates(normalized)
    if predicates is None:
        return py_fragment.mark_scan()
    return QueryFragment(
        ctx.backend,
        "exists",
        field_schema.name,
        {"field": field_schema.name, "predicates": dict(predicates)},
        notes={"kind": "cypher_where", "edge_exists": True, "mode": "native", "row_payload": py_fragment.payload},
    )


def _native_edge_exists_predicates(value: Any) -> dict[str, Any] | None:
    """Return exact-match predicates that can render as relationship equality, else None."""
    if not isinstance(value, dict):
        return None
    out: dict[str, Any] = {}
    for column, predicates in value.items():
        if not isinstance(predicates, dict) or set(predicates) != {"eq"}:
            return None
        predicate_value = predicates["eq"]
        if not isinstance(predicate_value, (str, int, float, bool)):
            return None
        out[column] = predicate_value
    return out


def neo4j_near_handler(field_schema: Any, value: Any, ctx: HandlerContext) -> QueryFragment:
    """Return a Neo4j-native vector ``near`` fragment (KNN via the vector index), with a metric-correct scan scorer.

    The backend runs a pure cosine ``near`` through Neo4j's cosine vector index; a near co-located
    with a filter, a dimensionless/unindexed vector field, or any non-cosine metric degrades to the
    portable scan, which scores with the metric-matched scorer carried in ``notes['row_payload']``
    (cosine -> ``cosine_score``, l2/euclidean -> ``euclidean_score``, ip/inner_product ->
    ``inner_product_score``).

    Args:
        field_schema (Any): Routed vector field schema.
        value (Any): Query vector validated against the field dtype.
        ctx (HandlerContext): Selected backend and strategy context.

    Returns:
        QueryFragment: Native or scan-marked Neo4j vector near fragment.

    Raises:
        ValueError: If the query dimension or metric is unsupported.
    """
    query = field_schema.dtype.validate(value)
    expected_dim = getattr(field_schema.dtype, "dim", None)
    if expected_dim is not None and len(query) != expected_dim:
        raise ValueError(f"Neo4j near query dimension mismatch: expected {expected_dim}, got {len(query)}")
    metric = str(ctx.metric or "cosine").lower()
    scorer = SCAN_SCORERS.get(metric)
    if scorer is None:
        raise ValueError(f"Neo4j near: unsupported metric {ctx.metric!r}; use 'cosine', 'l2'/'euclidean', or 'ip'/'inner_product'")

    # Native search never returns a row without a vector, so the scan scorer (a filter that is not
    # pushed down) leaves it out too, under every metric: the filter cannot change the near's rows.
    def _payload(row: dict) -> float | None:
        vec = row.get(field_schema.name)
        return scorer(vec, query) if vec is not None else None

    # Neo4j native KNN needs a fixed-dim cosine vector index (see Neo4jBackend._run_near); a non-cosine
    # metric or a dimensionless field ranks via the carried scan scorer, so report it honestly as py_scan
    # in explain() rather than a native that silently scans. A cosine near co-located with a filter still
    # degrades at runtime, but that is the near+filter route's call (combined_scan), not knowable here.
    notes = {"kind": "neo4j_vector_search", "mode": "native", "top_k": ctx.top_k, "row_payload": _payload}
    if metric != "cosine" or expected_dim is None:
        notes["mode"] = "py_scan"
        notes["fallback_reason"] = (
            f"Neo4j native KNN is cosine-only; {metric!r} ranks via the Python scan scorer."
            if metric != "cosine"
            else "Neo4j native KNN needs a fixed-dimension vector index; a dimensionless Vector ranks via the Python scan scorer."
        )
    return QueryFragment(ctx.backend, "near", field_schema.name, {"vector": query, "top_k": ctx.top_k, "metric": ctx.metric}, returns=["score"], notes=notes)


def register_reasons(fallback: Any, _unsupported: Any, *, resolver: Any = None) -> None:
    """Register Neo4j-specific fallback reasons for the JSON/list-stored types.

    Args:
        fallback: Reason registry receiving fallback explanations.
        _unsupported: Unsupported-operation registry (unused for Neo4j).
        resolver (Any | None): Optional module resolver for backend discovery.

    Returns:
        None: Fallback reasons are registered in place.
    """
    for dtype in ("json", "array", "hyper-g"):
        fallback.register(BACKEND_IDENTIFIER, "*", dtype, "*", f"Neo4j stores {dtype} as a JSON/list property; using Python scan fallback.")


def register_handlers(registry: Any, ctx: Any) -> None:
    """Register Neo4j-native filter handlers (scalar ops) and the vector near handler.

    Args:
        registry: Workspace handler registry receiving descriptors.
        ctx: Shared Registry-derived handler vocabulary.

    Returns:
        None: The supplied workspace registry is updated in place.
    """
    for dtype in ctx.filter_types:
        for strategy in ctx.filter_strategies:
            for op in ctx.filter_ops(dtype):
                if op in CYPHER_PUSH_OPS:
                    ctx.handler(dtype, op, BACKEND_IDENTIFIER, strategy, neo4j_filter_handler(op)).register(registry, replace=True)
    for dtype in ctx.vector_types:
        ctx.handler(dtype, OP_NEAR, BACKEND_IDENTIFIER, VectorIndex.identifier, neo4j_near_handler).register(registry, replace=True)
    # GraphEdge `exists` pushes to a native relationship EXISTS subquery, overriding the
    # generic Python-scan handler the seed registers for the graph strategy; it degrades back to that scan
    # for complex predicates, while mixed logical trees compose native and scan leaves. `all` stays on the seed's portable scan.
    ctx.handler(
        HyperG,
        "exists",
        BACKEND_IDENTIFIER,
        GraphEdge.identifier,
        neo4j_graph_exists_handler,
    ).register(registry, replace=True)
