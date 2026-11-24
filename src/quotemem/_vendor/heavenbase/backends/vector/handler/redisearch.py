"""RediSearch-native operation compilers.

Each scalar/text operation compiles into an ``FT.SEARCH`` clause (numeric range,
tag, or text) carried as the native fragment payload, alongside the portable
Python predicate in ``notes['row_payload']`` so the workspace router can fall
back to scan for any combination RediSearch cannot express natively. ``near``
compiles into a vector fragment the backend renders as KNN; when the router
co-locates a native filter with the vector it becomes ``combined_native``.
"""

from __future__ import annotations

__all__ = [
    "redisearch_filter_handler",
    "redisearch_near_handler",
    "register_handlers",
    "register_reasons",
]

from heavenbase.utils import Any

from heavenbase.execution.contracts import HandlerContext
from heavenbase.execution.contracts import QueryFragment
from heavenbase.backends.vector.redisearch import normalize_metric, tag_escape
from heavenbase.strategies import VectorIndex
from heavenbase.types import cosine_score, euclidean_score, inner_product_score
from heavenbase.types.storage import stores_as_integer
from heavenbase.filters.ops import OP_EQ, OP_NEAR
from heavenbase.execution.handlers.filters import filter_handler, normalize_filter_value

BACKEND_IDENTIFIER = "redisearch"
RS_NATIVE_OPS = ("eq", "ne", "lt", "lte", "gt", "gte", "in", "starts_with")
RS_SCAN_TYPES = ("array", "json", "hyper-g", "vector")
# Scan-fallback scorer per FT metric, keyed like normalize_metric. A near whose metric is the
# index metric ranks natively, and its scan predicate ranks the same way, so combined_scan and
# combined_native return the same top-k; any other metric ranks through the scan scorer alone.
_SCAN_SCORERS = {"COSINE": cosine_score, "L2": euclidean_score, "IP": inner_product_score}
# Requested near metric -> FT DISTANCE_METRIC; an unknown name is refused, not read as cosine.
_REQUESTED_METRICS = {"cosine": "COSINE", "l2": "L2", "euclidean": "L2", "ip": "IP", "inner": "IP", "inner_product": "IP", "dot": "IP"}


def _is_numeric(field_schema: Any) -> bool:
    return type(field_schema.dtype).identifier == "float" or stores_as_integer(field_schema.dtype)


def _is_text(field_schema: Any) -> bool:
    return type(field_schema.dtype).identifier in ("long-text", "medium-text")


def _tag_value(field_schema: Any, value: Any) -> str:
    if type(field_schema.dtype).identifier == "boolean":
        return "1" if value else "0"
    return tag_escape(value)


def _native_ok(field_schema: Any, op: str) -> bool:
    if op not in RS_NATIVE_OPS or type(field_schema.dtype).identifier in RS_SCAN_TYPES:
        return False
    if _is_numeric(field_schema):
        return op in ("eq", "ne", "lt", "lte", "gt", "gte", "in")
    if _is_text(field_schema):
        # FT TEXT fields are tokenized (phrase match, NOT exact equality) and the
        # phrase channel cannot be value-escaped safely, so eq/ne scan too — keeping
        # native results byte-identical to the case-exact Python predicate.
        return False
    # TAG fields: exact match, membership, and whole-value prefix (starts_with).
    # contains (substring) / match (case-insensitive substring) are NOT expressible
    # in FT tag/text without changing semantics -> they fall back to scan.
    return op in ("eq", "ne", "in", "starts_with")


def _clause(field_schema: Any, op: str, value: Any) -> str:
    name = field_schema.name
    if _is_numeric(field_schema):
        if op == "eq":
            return f"@{name}:[{value} {value}]"
        if op == "ne":
            return f"-@{name}:[{value} {value}]"
        if op == "lt":
            return f"@{name}:[-inf ({value}]"
        if op == "lte":
            return f"@{name}:[-inf {value}]"
        if op == "gt":
            return f"@{name}:[({value} +inf]"
        if op == "gte":
            return f"@{name}:[{value} +inf]"
        if op == "in":
            return "(" + " | ".join(f"@{name}:[{item} {item}]" for item in value) + ")"
        raise ValueError(f"Unsupported numeric op {op!r}")
    # TAG field (identifier / short-text / categorical / boolean / date)
    if op == "eq":
        return f"@{name}:{{{_tag_value(field_schema, value)}}}"
    if op == "ne":
        return f"-@{name}:{{{_tag_value(field_schema, value)}}}"
    if op == "starts_with":
        return f"@{name}:{{{_tag_value(field_schema, value)}*}}"  # DIALECT 2 whole-value tag prefix
    if op == "in":
        return f"@{name}:{{{'|'.join(_tag_value(field_schema, item) for item in value)}}}"
    raise ValueError(f"Unsupported tag op {op!r}")


def redisearch_near_handler(field_schema: Any, value: Any, ctx: HandlerContext) -> QueryFragment:
    """Return a RediSearch-native KNN vector fragment with dimension validation.

    Args:
        field_schema (Any): Routed vector field schema.
        value (Any): Query vector validated against the field dtype.
        ctx (HandlerContext): Selected backend and strategy context.

    Returns:
        QueryFragment: Native or scan-marked RediSearch vector near fragment.

    Raises:
        ValueError: If the query dimension or metric is unsupported.
    """
    query = field_schema.dtype.validate(value)
    expected_dim = getattr(field_schema.dtype, "dim", None)
    if expected_dim is not None and len(query) != expected_dim:
        raise ValueError(f"RediSearch near query dimension mismatch: expected {expected_dim}, got {len(query)}")

    requested = _REQUESTED_METRICS.get(str(ctx.metric or "cosine").lower())
    if requested is None:
        raise ValueError(f"RediSearch near: unsupported metric {ctx.metric!r}; use 'cosine', 'l2'/'euclidean', or 'ip'/'inner_product'")
    # Native KNN always ranks by the metric the index was built with, which the backend reads from
    # the index (a config change alone does not rebuild it). The scan scorer ranks by the requested
    # metric: for a near that matches the index, a combined_scan fallback then ranks rows
    # identically to combined_native; any other metric ranks through the scan alone.
    backend = getattr(ctx, "backend_instance", None)
    near_index = getattr(backend, "_near_index", None)
    near_metric = getattr(backend, "near_metric", None)
    entity_id = getattr(ctx, "entity_id", None)
    exists = None
    if callable(near_index) and entity_id is not None:
        index_metric, exists = near_index(entity_id, field_schema.name)
    elif callable(near_metric) and entity_id is not None:
        index_metric = normalize_metric(near_metric(entity_id, field_schema.name))
    else:
        index_metric = normalize_metric(str(ctx.config.get("heavenbase.backend_defaults.redisearch.metric", default="cosine")))
    scorer = _SCAN_SCORERS[requested]

    # Native search never returns a row without a vector, so the scan scorer (a filter that is not
    # pushed down) leaves it out too, under every metric: the filter cannot change the near's rows.
    def _payload(row: dict) -> float | None:
        vec = row.get(field_schema.name)
        return scorer(vec, query) if vec is not None else None

    notes = {"kind": "redis_search", "mode": "native", "top_k": ctx.top_k, "row_payload": _payload}
    if requested != index_metric:
        notes["mode"] = "py_scan"
        notes["fallback_reason"] = f"RediSearch indexes use {index_metric}; {ctx.metric!r} ranks through the portable scan scorer."
    payload = {"vector": query, "top_k": ctx.top_k, "metric": ctx.metric, "index_metric": index_metric}
    if exists is not None:
        payload["index_exists"] = exists
    return QueryFragment(
        ctx.backend,
        "near",
        field_schema.name,
        payload,
        returns=["score"],
        notes=notes,
    )


def redisearch_filter_handler(op: str):
    """Return a RediSearch scalar/text filter handler for one operation.

    Args:
        op (str): Filter operation identifier.

    Returns:
        Callable[[Any, Any, HandlerContext], QueryFragment]: Compiler that
        produces native or scan-marked RediSearch filter fragments.
    """

    def _compile(field_schema: Any, value: Any, ctx: HandlerContext) -> QueryFragment:
        value = normalize_filter_value(field_schema.dtype, op, value, resolver=ctx.resolver)
        py_fragment = filter_handler(op)(field_schema, value, ctx)
        if not _native_ok(field_schema, op):
            reason = ctx.fallback_reason(op, type(field_schema.dtype).identifier)
            return py_fragment.mark_scan(reason)
        if op == "in" and not isinstance(value, (list, tuple, set)):
            raise TypeError("in filter expects a list, tuple, or set")
        if op == "in" and not value:
            # empty membership matches nothing; scan returns [] (native @f:{} / () is invalid FT).
            return py_fragment.mark_scan()
        if op == "starts_with" and len(str(value)) < int(ctx.config.get("heavenbase.backend_defaults.redisearch.min_prefix", default=2)):
            # FT prefix queries shorter than the index MINPREFIX (default 2) silently match nothing,
            # diverging from the scan predicate (which matches any prefix) -> scan to stay consistent.
            return py_fragment.mark_scan()
        members = value if op == "in" else [value]
        if any(member is None for member in members):
            # A None operand never pushes down faithfully on ANY field: a numeric field (incl. an
            # integer-backed Categorical, which preserves None through normalization) compiles to an
            # invalid @f:[None None] range, a TAG field matches the literal token "None" (boolean: every
            # False row), and NULL values are never indexed. The scan predicate matches NULL rows
            # correctly -> scan to stay consistent.
            return py_fragment.mark_scan()
        if not _is_numeric(field_schema) and any(str(member) == "" for member in members):
            # An empty-string TAG operand compiles to invalid FT (@f:{} / -@f:{} / a leading empty
            # alternative); the scan predicate matches empty-string rows correctly -> scan.
            return py_fragment.mark_scan()
        return QueryFragment(
            ctx.backend,
            op,
            field_schema.name,
            _clause(field_schema, op, value),
            notes={"kind": "redis_filter", "mode": "native", "row_payload": py_fragment.payload},
        )

    return _compile


def register_reasons(fallback: Any, _unsupported: Any, *, resolver: Any = None) -> None:
    """Register RediSearch-specific fallback reasons.

    Args:
        fallback: Reason registry receiving fallback explanations.
        _unsupported: Unsupported-operation registry (unused for RediSearch).
        resolver (Any | None): Optional module resolver for backend discovery.

    Returns:
        None: Fallback reasons are registered in place.
    """
    fallback.register(BACKEND_IDENTIFIER, "*", "json", "*", "RediSearch does not natively filter Json fields here; using Python scan fallback.")
    fallback.register(BACKEND_IDENTIFIER, "*", "hyper-g", "*", "RediSearch does not natively filter HyperG fields here; using Python scan fallback.")
    fallback.register(BACKEND_IDENTIFIER, "*", "array", "*", "RediSearch does not natively filter Array fields here; using Python scan fallback.")
    fallback.register(BACKEND_IDENTIFIER, "contains", "*", "*", "RediSearch cannot express substring containment; using Python scan fallback.")
    fallback.register(BACKEND_IDENTIFIER, "match", "*", "*", "RediSearch cannot express case-insensitive substring match; using Python scan fallback.")
    fallback.register(BACKEND_IDENTIFIER, "ends_with", "*", "*", "RediSearch suffix search needs WITHSUFFIXTRIE; using Python scan fallback.")
    for dtype in ("long-text", "medium-text"):
        fallback.register(BACKEND_IDENTIFIER, "starts_with", dtype, "*", "RediSearch TEXT fields are tokenized; whole-value prefix uses Python scan fallback.")
        fallback.register(
            BACKEND_IDENTIFIER,
            "eq",
            dtype,
            "*",
            "RediSearch TEXT fields are tokenized (phrase match, not exact equality); using case-exact Python scan fallback.",
        )
        fallback.register(
            BACKEND_IDENTIFIER,
            "ne",
            dtype,
            "*",
            "RediSearch TEXT fields are tokenized (phrase match, not exact equality); using case-exact Python scan fallback.",
        )
    for dtype in ("identifier", "short-text", "categorical", "boolean", "date"):
        fallback.register(BACKEND_IDENTIFIER, "lt", dtype, "*", "RediSearch TAG fields do not support range; using Python scan fallback.")
        fallback.register(BACKEND_IDENTIFIER, "gt", dtype, "*", "RediSearch TAG fields do not support range; using Python scan fallback.")


def register_handlers(registry: Any, ctx: Any) -> None:
    """Register RediSearch-native filter and near handlers.

    Args:
        registry: Workspace handler registry receiving descriptors.
        ctx: Shared Registry-derived handler vocabulary.

    Returns:
        None: The supplied workspace registry is updated in place.
    """
    for dtype in ctx.filter_types:
        for strategy in ctx.filter_strategies:
            for op in ctx.filter_ops(dtype):
                ctx.handler(dtype, op, BACKEND_IDENTIFIER, strategy, redisearch_filter_handler(op)).register(registry, replace=True)
    for dtype in ctx.vector_types:
        ctx.handler(dtype, OP_NEAR, BACKEND_IDENTIFIER, VectorIndex.identifier, redisearch_near_handler).register(registry, replace=True)
        ctx.handler(dtype, OP_EQ, BACKEND_IDENTIFIER, VectorIndex.identifier, redisearch_filter_handler(OP_EQ)).register(registry, replace=True)
