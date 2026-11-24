"""OpenSearch-native scalar filter handlers."""

from __future__ import annotations

__all__ = ["opensearch_filter_handler", "register_handlers", "register_reasons"]

from heavenbase.utils import Any

from heavenbase.execution.contracts import HandlerContext, QueryFragment
from heavenbase.execution.handlers.filters import filter_handler, normalize_filter_value

BACKEND_IDENTIFIER = "opensearch"
OPENSEARCH_NATIVE_OPS = (
    "eq",
    "ne",
    "lt",
    "lte",
    "gt",
    "gte",
    "match",
    "like",
    "ilike",
    "wildcard",
    "contains",
    "starts_with",
    "ends_with",
    "in",
)
OPENSEARCH_SCAN_TYPES = ("array", "json", "hyper-g", "vector")
OPENSEARCH_TEXT_TYPES = ("identifier", "long-text", "medium-text", "short-text")
OPENSEARCH_LITERAL_META = frozenset("*?\\")
OPENSEARCH_NULL_AS_EMPTY_OPS = frozenset(("match", "like", "ilike", "wildcard", "contains"))
OPENSEARCH_UNICODE_CASEFOLD_OPS = frozenset(("match", "ilike"))


def _is_text_field(field_schema: Any) -> bool:
    """Return whether a field has portable text-pattern semantics."""
    identifier = type(field_schema.dtype).identifier
    if identifier == "categorical":
        return getattr(field_schema.dtype, "base", "string") == "string"
    return identifier in OPENSEARCH_TEXT_TYPES


def _literal_pattern_safe(value: Any) -> bool:
    """Return whether a value can be embedded as literal wildcard text."""
    text = str(value)
    return bool(text) and not any(character in OPENSEARCH_LITERAL_META for character in text)


def _wildcard_pattern_safe(value: Any) -> bool:
    """Return whether provider wildcard syntax matches Python fnmatch syntax."""
    text = str(value)
    return "[" not in text and "]" not in text and "\\" not in text


def _native_ok(field_schema: Any, op: str, value: Any) -> bool:
    """Return whether OpenSearch preserves HeavenBase scalar semantics."""
    if op not in OPENSEARCH_NATIVE_OPS or type(field_schema.dtype).identifier in OPENSEARCH_SCAN_TYPES:
        return False
    if op in OPENSEARCH_UNICODE_CASEFOLD_OPS:
        # OpenSearch's case-insensitive wildcard matching is ASCII-only, while
        # HeavenBase's portable predicates are Unicode-aware.
        return False
    if value is None or (op == "in" and isinstance(value, (list, tuple, set)) and (not value or any(item is None for item in value))):
        return False
    if op in ("match", "contains", "starts_with", "ends_with", "like", "ilike", "wildcard"):
        if not _is_text_field(field_schema):
            return False
        if op == "wildcard":
            return _wildcard_pattern_safe(value)
        return _literal_pattern_safe(value)
    return True


def _portable_pattern_matches_missing(fragment: QueryFragment, field_name: str) -> bool:
    """Return whether a portable text predicate accepts absent or null input."""
    try:
        return bool(fragment.payload({})) or bool(fragment.payload({field_name: None}))
    except Exception:
        return True


def _sql_wildcard(value: Any) -> str:
    """Translate SQL LIKE wildcards to OpenSearch wildcard syntax."""
    return str(value).replace("%", "*").replace("_", "?")


def _query(field: str, op: str, value: Any) -> dict[str, Any]:
    """Compile one scalar operation to OpenSearch Query DSL."""
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
        return {"wildcard": {field: {"value": _sql_wildcard(value), "case_insensitive": op == "ilike"}}}
    if op == "wildcard":
        return {"wildcard": {field: {"value": str(value), "case_insensitive": False}}}
    raise ValueError(f"Unsupported OpenSearch operation {op!r}")


def opensearch_filter_handler(op: str) -> Any:
    """Return an OpenSearch scalar compiler for one filter operation.

    Args:
        op (str): Filter operation identifier.

    Returns:
        Callable[[Any, Any, HandlerContext], QueryFragment]: Compiler that
        produces native or scan-marked OpenSearch fragments.
    """

    def _compile(field_schema: Any, value: Any, ctx: HandlerContext) -> QueryFragment:
        normalized = normalize_filter_value(field_schema.dtype, op, value, resolver=ctx.resolver)
        py_fragment = filter_handler(op)(field_schema, normalized, ctx)
        missing_mismatch = (
            not field_schema.required and op in OPENSEARCH_NULL_AS_EMPTY_OPS and _portable_pattern_matches_missing(py_fragment, field_schema.name)
        )
        if (field_schema.name != "object_id" and not ctx.scalar_indexed) or not _native_ok(field_schema, op, normalized) or missing_mismatch:
            reason = ctx.fallback_reason(op, type(field_schema.dtype).identifier)
            return py_fragment.mark_scan(reason)
        return QueryFragment(
            ctx.backend,
            op,
            field_schema.name,
            _query(field_schema.name, op, normalized),
            notes={
                "kind": "opensearch_query",
                "mode": "native",
                "row_payload": py_fragment.payload,
            },
        )

    return _compile


def register_reasons(fallback: Any, _unsupported: Any, *, resolver: Any = None) -> None:
    """Register OpenSearch-specific scan fallback explanations.

    Args:
        fallback: Reason registry receiving fallback explanations.
        _unsupported: Unsupported-operation registry (unused for OpenSearch).
        resolver (Any | None): Optional module resolver for backend discovery.

    Returns:
        None: Fallback reasons are registered in place.
    """
    del resolver
    for identifier, label in (
        ("array", "Array"),
        ("json", "Json"),
        ("hyper-g", "HyperG"),
        ("vector", "Vector"),
    ):
        fallback.register(
            BACKEND_IDENTIFIER,
            "*",
            identifier,
            "*",
            f"OpenSearch does not preserve {label} native filter semantics in the current mapping; using Python scan fallback.",
        )
    fallback.register(
        BACKEND_IDENTIFIER,
        "regex",
        "*",
        "*",
        "OpenSearch regexp syntax is not Python re-compatible; using Python scan fallback for exact HeavenBase regex semantics.",
    )
    for op in sorted(OPENSEARCH_UNICODE_CASEFOLD_OPS):
        fallback.register(
            BACKEND_IDENTIFIER,
            op,
            "*",
            "*",
            "OpenSearch case-insensitive wildcard matching is ASCII-only; " "using Python scan fallback for Unicode-aware HeavenBase text semantics.",
        )


def register_handlers(registry: Any, ctx: Any) -> None:
    """Register OpenSearch-native scalar handlers.

    Args:
        registry: Workspace handler registry receiving descriptors.
        ctx: Shared Registry-derived handler vocabulary.

    Returns:
        None: The supplied workspace registry is updated in place.
    """
    for dtype in ctx.filter_types:
        for strategy in ctx.filter_strategies:
            for op in ctx.filter_ops(dtype):
                ctx.handler(
                    dtype,
                    op,
                    BACKEND_IDENTIFIER,
                    strategy,
                    opensearch_filter_handler(op),
                ).register(registry, replace=True)
