"""Portable scalar and text filter compilers."""

from __future__ import annotations

__all__ = [
    "filter_handler",
    "normalize_filter_value",
    "register_reasons",
]

from heavenbase.utils import Any, Callable, Dict
from heavenbase.utils.text_match import contained_in_matches

from ...filters.ops import normalize_op
from ...modules import ModuleResolver
from ...strategies import InlineColumn, VectorIndex
from ...types.descriptor import coerce_type, logical_type_from_any
from ..contracts import HandlerContext, QueryFragment
from .predicates import Predicate, _compile_hyperg_predicates, _hyperg_all, _hyperg_exists


def normalize_filter_value(
    dtype: Any,
    op: str,
    value: Any,
    *,
    resolver: ModuleResolver | None = None,
) -> Any:
    """Return a logical-type-aware query value for one filter operation.

    Args:
        dtype (Any): Logical type class, instance, identifier, or alias.
        op (str): Operation identifier.
        value (Any): User-authored query value.
        resolver (ModuleResolver | None): Context-owned module resolver. The
            default Context is selected when `dtype` needs lookup and this is
            omitted.

    Returns:
        Any: Value normalized by the logical type for the requested operation.
    """
    normalized = logical_type_from_any(dtype, resolver=resolver).filter_value(op, value)
    if op == "get" and isinstance(normalized, dict) and "op" in normalized:
        normalized = dict(normalized)
        inner_op = normalize_op(normalized["op"], resolver=resolver)
        if inner_op not in ("eq", "ne", "in", "lt", "lte", "gt", "gte", "contains"):
            raise ValueError(f"Json get predicates do not support {inner_op!r}")
        normalized["op"] = inner_op
    return normalized


def filter_handler(op: str) -> Callable[[Any, Any, HandlerContext], QueryFragment]:
    """Return a Python-scan compiler for one filter operation.

    Args:
        op (str): Operation identifier, such as `eq`, `contains`, `exists`, or
            `contained_in`.

    Returns:
        Callable[[Any, Any, HandlerContext], QueryFragment]: Compiler that
        produces a row-payload `QueryFragment`.

    Raises:
        ValueError: If the operation or pattern value is unsupported.
        TypeError: If an `in` filter receives a non-container query value.
    """

    def _compile(field_schema: Any, value: Any, ctx: HandlerContext) -> QueryFragment:
        dtype = coerce_type(field_schema.dtype)
        value = normalize_filter_value(dtype, op, value, resolver=ctx.resolver)
        if op == "exists":
            predicates = _compile_hyperg_predicates(value, ctx.resolver, ctx.config)

            def _payload(row: dict) -> bool:
                return _hyperg_exists(row.get(field_schema.name), predicates)

            reason = ctx.fallback_reason(op, type(field_schema.dtype).identifier)
            notes: Dict[str, Any] = {"mode": "py_scan", "top_k": ctx.top_k}
            if reason:
                notes["fallback_reason"] = reason
            return QueryFragment(ctx.backend, op, field_schema.name, _payload, notes=notes)
        if op == "all":
            where = _compile_hyperg_predicates(value["where"], ctx.resolver, ctx.config)
            then = _compile_hyperg_predicates(value["then"], ctx.resolver, ctx.config)

            def _payload(row: dict) -> bool:
                return _hyperg_all(row.get(field_schema.name), where, then)

            reason = ctx.fallback_reason(op, type(field_schema.dtype).identifier)
            notes: Dict[str, Any] = {"mode": "py_scan", "top_k": ctx.top_k}
            if reason:
                notes["fallback_reason"] = reason
            return QueryFragment(ctx.backend, op, field_schema.name, _payload, notes=notes)
        predicate = Predicate.load(op, dtype, resolver=ctx.resolver)
        prepared = predicate.compile(value, resolver=ctx.resolver, config=ctx.config)
        if op == "contained_in":

            def _matches(row: dict) -> list[dict[str, Any]]:
                return contained_in_matches(field_schema.name, row.get(field_schema.name), value)

            def _payload(row: dict) -> bool:
                return bool(_matches(row))

            def _enricher(row: dict) -> dict[str, Any]:
                matches = _matches(row)
                return {"match": matches} if matches else {}

            reason = ctx.fallback_reason(op, type(field_schema.dtype).identifier)
            notes: Dict[str, Any] = {"mode": "py_scan", "top_k": ctx.top_k, "row_enricher": _enricher, "metadata_columns": ["match"]}
            if reason:
                notes["fallback_reason"] = reason
            return QueryFragment(ctx.backend, op, field_schema.name, _payload, notes=notes)

        if op == "get":

            def _payload(row: dict) -> bool:
                return predicate.match(row.get(field_schema.name), prepared)

        else:

            def _payload(row: dict) -> bool:
                return predicate.match(dtype.decode(row.get(field_schema.name)), prepared)

        reason = ctx.fallback_reason(op, type(field_schema.dtype).identifier)
        notes: Dict[str, Any] = {"mode": "py_scan", "top_k": ctx.top_k}
        if reason:
            notes["fallback_reason"] = reason
        return QueryFragment(ctx.backend, op, field_schema.name, _payload, notes=notes)

    return _compile


def register_reasons(
    fallback: Any,
    unsupported: Any,
    *,
    resolver: ModuleResolver | None = None,
) -> None:
    """Register generic fallback and unsupported reasons.

    These apply to all backends unless overridden by a more specific handler.

    Args:
        fallback (Any): Fallback reason registry.
        unsupported (Any): Unsupported-operation reason registry.
        resolver (ModuleResolver | None): Context-owned resolver. Generic
            reason registration does not perform typed lookups.

    Returns:
        None: This function updates both registries in place.
    """
    del resolver
    # Generic fallbacks for pattern-matching ops
    fallback.register("*", "regex", "*", "*", "Regex is not supported natively by this backend; using Python scan fallback.")
    fallback.register(
        "*", "wildcard", "*", "*", "Wildcard pattern uses unsupported features or backend lacks case-sensitive LIKE/GLOB; using Python scan fallback."
    )
    fallback.register("*", "*", "json", "*", "Json type fields do not support native scalar filtering on this backend.")
    fallback.register("*", "*", "hyper-g", "*", "HyperG type fields do not support native scalar filtering on this backend.")
    fallback.register("*", "*", "vector", InlineColumn.identifier, f"Vector fields require {VectorIndex.__name__} strategy for native operations.")

    # File / inmem backends always scan
    fallback.register("inmem", "*", "*", "*", "In-memory backend always uses Python scan fallback.")
    fallback.register("file", "*", "*", "*", "File backend always uses Python scan fallback.")
    fallback.register("json", "*", "*", "*", "JSON file backend always uses Python scan fallback.")
    fallback.register("pickle", "*", "*", "*", "Pickle file backend always uses Python scan fallback.")

    # Unsupported type/op combinations
    for dtype in ("float", "integer", "boolean", "vector", "array"):
        unsupported.register("*", "regex", dtype, "*", f"{dtype} fields do not support regex operations.")
        unsupported.register("*", "wildcard", dtype, "*", f"{dtype} fields do not support wildcard operations.")
    for dtype in ("float", "integer", "boolean", "vector"):
        unsupported.register("*", "like", dtype, "*", f"{dtype} fields do not support LIKE pattern matching.")
        unsupported.register("*", "ilike", dtype, "*", f"{dtype} fields do not support ILIKE pattern matching.")
        unsupported.register("*", "match", dtype, "*", f"{dtype} fields do not support text match operations.")
        unsupported.register("*", "contains", dtype, "*", f"{dtype} fields do not support text contains operations.")
        unsupported.register("*", "contained_in", dtype, "*", f"{dtype} fields do not support reverse containment operations.")
        unsupported.register("*", "starts_with", dtype, "*", f"{dtype} fields do not support prefix (starts_with) matching.")
        unsupported.register("*", "ends_with", dtype, "*", f"{dtype} fields do not support suffix (ends_with) matching.")
    for dtype in ("json", "hyper-g"):
        unsupported.register("*", "contained_in", dtype, "*", f"{dtype} fields do not support reverse containment operations.")
    unsupported.register("*", "array_contains", "*", "*", "Array contains is only supported on Array fields.")
