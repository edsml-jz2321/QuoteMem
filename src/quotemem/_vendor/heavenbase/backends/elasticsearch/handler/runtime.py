"""Elasticsearch-native Handler runtime implementation."""

from __future__ import annotations

__all__ = [
    "elasticsearch_filter_handler",
    "register_handlers",
    "register_reasons",
]

from heavenbase.utils import Any

from heavenbase.execution.contracts import HandlerContext
from heavenbase.execution.contracts import QueryFragment
from heavenbase.execution.handlers.filters import filter_handler, normalize_filter_value

BACKEND_IDENTIFIER = "elasticsearch"
ES_NATIVE_OPS = ("eq", "ne", "lt", "lte", "gt", "gte", "match", "like", "ilike", "wildcard", "contains", "starts_with", "ends_with", "in")
ES_SCAN_TYPES = ("array", "json", "hyper-g", "vector")
ES_TEXT_PATTERN_TYPES = ("identifier", "long-text", "medium-text", "short-text")
ES_LITERAL_WILDCARD_META = frozenset("*?\\")


def _wildcard(value: Any, *, sql: bool = False) -> str:
    text = str(value)
    return text.replace("%", "*").replace("_", "?") if sql else text


def _is_text_pattern_field(field_schema: Any) -> bool:
    ident = type(field_schema.dtype).identifier
    if ident == "categorical":
        return getattr(field_schema.dtype, "base", "string") == "string"
    return ident in ES_TEXT_PATTERN_TYPES


def _literal_wildcard_safe(value: Any) -> bool:
    text = str(value)
    return bool(text) and not any(char in ES_LITERAL_WILDCARD_META for char in text)


def _like_wildcard_safe(value: Any) -> bool:
    return _literal_wildcard_safe(value)


def _fnmatch_wildcard_safe(value: Any) -> bool:
    text = str(value)
    return "[" not in text and "]" not in text and "\\" not in text


def _has_none_operand(op: str, value: Any) -> bool:
    return value is None or (op == "in" and isinstance(value, (list, tuple, set)) and any(item is None for item in value))


def _native_ok(field_schema: Any, op: str, value: Any) -> bool:
    if op not in ES_NATIVE_OPS or type(field_schema.dtype).identifier in ES_SCAN_TYPES:
        return False
    if _has_none_operand(op, value):
        # A term/terms/range query cannot carry null; the scan predicate tests for a missing value.
        return False
    if op in ("match", "like", "ilike", "wildcard", "contains", "starts_with", "ends_with"):
        if not _is_text_pattern_field(field_schema):
            return False
        if op in ("match", "contains", "starts_with", "ends_with"):
            return _literal_wildcard_safe(value)
        if op in ("like", "ilike"):
            return _like_wildcard_safe(value)
        return _fnmatch_wildcard_safe(value)
    return True


def _query(field: str, op: str, value: Any) -> dict:
    if op == "eq":
        return {"term": {field: value}}
    if op == "ne":
        return {"bool": {"must_not": [{"term": {field: value}}]}}
    if op in ("lt", "lte", "gt", "gte"):
        return {"range": {field: {op: value}}}
    if op == "in":
        if not isinstance(value, (list, tuple, set)):
            raise TypeError("in filter expects a list, tuple, or set")
        return {"terms": {field: list(value)}}
    if op == "match":
        return {"wildcard": {field: {"value": f"*{value}*", "case_insensitive": True}}}
    if op == "contains":
        return {"wildcard": {field: {"value": f"*{value}*", "case_insensitive": False}}}
    if op == "starts_with":
        return {"wildcard": {field: {"value": f"{value}*", "case_insensitive": False}}}
    if op == "ends_with":
        return {"wildcard": {field: {"value": f"*{value}", "case_insensitive": False}}}
    if op in ("like", "ilike"):
        return {"wildcard": {field: {"value": _wildcard(value, sql=True), "case_insensitive": op == "ilike"}}}
    if op == "wildcard":
        return {"wildcard": {field: {"value": _wildcard(value), "case_insensitive": False}}}
    raise ValueError(f"Unsupported elasticsearch op {op}")


def elasticsearch_filter_handler(op: str):
    """Return an Elasticsearch query handler for one filter operation.

    Args:
        op (str): Filter operation identifier.

    Returns:
        Callable[[Any, Any, HandlerContext], QueryFragment]: Compiler that
        produces native or scan-marked Elasticsearch fragments.
    """

    def _compile(field_schema: Any, value: Any, ctx: HandlerContext) -> QueryFragment:
        value = normalize_filter_value(field_schema.dtype, op, value, resolver=ctx.resolver)
        py_fragment = filter_handler(op)(field_schema, value, ctx)
        if (field_schema.name != "object_id" and not ctx.scalar_indexed) or not _native_ok(field_schema, op, value):
            reason = ctx.fallback_reason(op, type(field_schema.dtype).identifier)
            return py_fragment.mark_scan(reason)
        return QueryFragment(
            ctx.backend,
            op,
            field_schema.name,
            _query(field_schema.name, op, value),
            notes={"kind": "es_query", "mode": "native", "row_payload": py_fragment.payload},
        )

    return _compile


def register_reasons(fallback: Any, _unsupported: Any, *, resolver: Any = None) -> None:
    """Register Elasticsearch-specific fallback reasons.

    Args:
        fallback: Reason registry receiving fallback explanations.
        _unsupported: Unsupported-operation registry (unused for Elasticsearch).
        resolver (Any | None): Optional module resolver for backend discovery.

    Returns:
        None: Fallback reasons are registered in place.
    """
    fallback.register(
        BACKEND_IDENTIFIER, "*", "array", "*", "Elasticsearch does not support Array field native filtering in current mapping; using Python scan fallback."
    )
    fallback.register(
        BACKEND_IDENTIFIER, "*", "json", "*", "Elasticsearch does not support Json field native filtering in current mapping; using Python scan fallback."
    )
    fallback.register(
        BACKEND_IDENTIFIER, "*", "hyper-g", "*", "Elasticsearch does not support HyperG field native filtering in current mapping; using Python scan fallback."
    )
    fallback.register(
        BACKEND_IDENTIFIER, "*", "vector", "*", "Elasticsearch does not support Vector field native filtering in current mapping; using Python scan fallback."
    )
    fallback.register(
        BACKEND_IDENTIFIER,
        "regex",
        "*",
        "*",
        "Elasticsearch regexp syntax is not Python re-compatible; using Python scan fallback for exact HeavenBase regex semantics.",
    )


def register_handlers(registry: Any, ctx: Any) -> None:
    """Register Elasticsearch-native handlers.

    Args:
        registry: Workspace handler registry receiving descriptors.
        ctx: Shared Registry-derived handler vocabulary.

    Returns:
        None: The supplied workspace registry is updated in place.
    """
    for dtype in ctx.filter_types:
        for strategy in ctx.filter_strategies:
            for op in ctx.filter_ops(dtype):
                ctx.handler(dtype, op, BACKEND_IDENTIFIER, strategy, elasticsearch_filter_handler(op)).register(registry, replace=True)
