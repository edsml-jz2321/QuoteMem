"""SQL-native filter compilers for SQLAlchemy backends.

Supported native ops are declared by each SQL backend's dialect metadata.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial

__all__ = [
    "sql_filter_handler",
    "sql_array_contains_handler",
    "sql_array_column_exact_handler",
    "sql_array_exact_handler",
    "sql_sparse_gram_contained_in_handler",
    "sql_hyperg_all_handler",
    "sql_hyperg_exists_handler",
    "register_handlers",
    "register_reasons",
]

from heavenbase.utils import Any

from heavenbase.execution.contracts import HandlerContext
from heavenbase.execution.contracts import QueryFragment
from heavenbase.backends.sql.capabilities import registered_sql_dialects, sql_dialect_spec, sql_filter_requires_scan
from heavenbase.modules import ModuleResolver, module_resolver
from heavenbase.strategies import SideTable, SparseGramIndex
from heavenbase.types import Array, HyperG, LogicalType, ShortText, get_logical_type
from heavenbase.filters.ops import OP_ALL, OP_ARRAY_CONTAINS, OP_CONTAINED_IN, OP_EQ, OP_EXISTS, OP_GET, OP_IN, OP_NE
from heavenbase.types.storage import stores_as_text
from heavenbase.execution.handlers.filters import filter_handler, normalize_filter_value
from .json import sql_json_get_handler
from .text import TEXT_SEMANTICS_NOT_PROVEN, sql_text_semantics_proven

SQL_HYPERG_EXISTS_OPS = ("eq", "ne", "lt", "lte", "gt", "gte", "in")
# LIKE escape char. '/' (not '\\'): DuckDB and the MySQL wire (StarRocks) treat a
# backslash escape as a two-char literal and reject it. SQLAlchemy's autoescape uses
# '/' too, so _sql_affix and _sql_wildcard share one convention.
_LIKE_ESCAPE = "/"


def _row_raw_claims(row: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return one inspected record's locally validated static Tag claims."""
    record = row.get("record")
    meta = record.get("meta") if isinstance(record, Mapping) else None
    value = meta.get("tags", {}) if isinstance(meta, Mapping) else {}
    if not isinstance(value, Mapping) or not all(isinstance(identifier, str) for identifier in value):
        return {}
    return value


def _identifier_set(claims: Mapping[str, Any], tag: str) -> frozenset[str]:
    """Return one locally validated identifier-set claim."""
    value = claims.get(tag)
    if not isinstance(value, (list, tuple)) or not all(isinstance(item, str) for item in value):
        return frozenset()
    return frozenset(value)


def _sql_backend_identifiers_from_resolver(resolver: ModuleResolver | None) -> tuple[str, ...]:
    """Return SQL row-storage candidates from one batched Backend inspection."""
    selected = module_resolver(resolver)
    identifiers: list[str] = []
    for row in selected.inspect("backend_type"):
        claims = _row_raw_claims(row)
        if claims.get("sql") is True and "row" in _identifier_set(claims, "storage-roles"):
            identifiers.append(str(row["identifier"]))
    return tuple(identifiers)


def _sql_backend_identifiers(ctx: Any) -> tuple[str, ...]:
    """Return SQL row-storage candidates from the shared seed snapshot."""
    return tuple(
        identifier
        for identifier in ctx.identifiers("backend_type")
        if (claims := ctx.raw_tags("backend_type", identifier)).get("sql") is True and "row" in _identifier_set(claims, "storage-roles")
    )


def _logical_types_with(ctx: Any, tag: str) -> tuple[type[LogicalType], ...]:
    """Materialize logical types carrying one claim in the seed snapshot."""
    return tuple(
        get_logical_type(identifier, resolver=ctx.resolver)
        for identifier in ctx.identifiers("logical_type")
        if ctx.raw_tags("logical_type", identifier).get(tag) is True
    )


def _logical_types_in_domain(ctx: Any, domain: str) -> tuple[type[LogicalType], ...]:
    """Materialize logical types in one captured filter domain."""
    return tuple(
        get_logical_type(identifier, resolver=ctx.resolver)
        for identifier in ctx.identifiers("logical_type")
        if domain in _identifier_set(ctx.raw_tags("logical_type", identifier), "filter-domains")
    )


def _sql_ilike(col: Any, value: Any, ctx: HandlerContext) -> Any:
    from sqlalchemy import String, func

    pattern = str(value).replace(_LIKE_ESCAPE, _LIKE_ESCAPE * 2)
    if sql_dialect_spec(ctx.backend_identifier, resolver=ctx.resolver).case_insensitive_like:
        return func.lower(col.cast(String)).like(pattern.lower(), escape=_LIKE_ESCAPE)
    return col.cast(String).ilike(pattern, escape=_LIKE_ESCAPE)


def _sql_like(col: Any, value: Any, _ctx: HandlerContext) -> Any:
    from sqlalchemy import String

    pattern = str(value).replace(_LIKE_ESCAPE, _LIKE_ESCAPE * 2)
    return col.cast(String).like(pattern, escape=_LIKE_ESCAPE)


def _sql_match(col: Any, value: Any, _ctx: HandlerContext) -> Any:
    from sqlalchemy import String, func

    return func.lower(col.cast(String)).contains(str(value).lower(), autoescape=True)


def _sql_contains(col: Any, value: Any, _ctx: HandlerContext) -> Any:
    from sqlalchemy import String

    return col.cast(String).contains(str(value), autoescape=True)


def _sql_affix(col: Any, value: Any, _ctx: HandlerContext, *, prefix: bool) -> Any:
    from sqlalchemy import String

    # autoescape escapes LIKE metacharacters and uses '/' as the escape char (see _LIKE_ESCAPE).
    text = col.cast(String)
    return text.startswith(value, autoescape=True) if prefix else text.endswith(value, autoescape=True)


def _can_translate_wildcard(value: Any) -> bool:
    """Return True if a fnmatch-style wildcard can be mapped to SQL LIKE/GLOB."""
    return "[" not in str(value) and "]" not in str(value)


def _sql_wildcard(col: Any, value: Any, ctx: HandlerContext) -> Any:
    return sql_dialect_spec(ctx.backend_identifier, resolver=ctx.resolver).wildcard_expression(col, value, escape=_LIKE_ESCAPE)


def _sql_regex(col: Any, value: Any, ctx: HandlerContext) -> Any:
    return sql_dialect_spec(ctx.backend_identifier, resolver=ctx.resolver).regex_expression(col, value)


_SQL_OP_DISPATCH = {
    "eq": lambda col, value, _ctx: col == value,
    "ne": lambda col, value, _ctx: col != value,
    "lt": lambda col, value, _ctx: col < value,
    "lte": lambda col, value, _ctx: col <= value,
    "gt": lambda col, value, _ctx: col > value,
    "gte": lambda col, value, _ctx: col >= value,
    "like": _sql_like,
    "ilike": _sql_ilike,
    "match": _sql_match,
    "contains": _sql_contains,
    "starts_with": lambda col, value, ctx: _sql_affix(col, value, ctx, prefix=True),
    "ends_with": lambda col, value, ctx: _sql_affix(col, value, ctx, prefix=False),
    "in": lambda col, value, _ctx: col.in_(value),
    "wildcard": _sql_wildcard,
    "regex": _sql_regex,
}

_NULL_AS_EMPTY_OPS = {"contains", "ilike", "like", "match", "regex", "wildcard"}
_NULL_FALSE_OPS = {"ends_with", "gt", "gte", "lt", "lte", "starts_with"}


def _sql_predicate(col: Any, op: str, value: Any, ctx: HandlerContext, *, text: bool) -> Any:
    if text:
        expression = sql_dialect_spec(ctx.backend_identifier, resolver=ctx.resolver).text_filter_expression(col, op, value)
        if expression is not None:
            return expression
    return _SQL_OP_DISPATCH[op](col, value, ctx)


def _sql_scalar_expr(col: Any, op: str, value: Any, ctx: HandlerContext, *, text: bool = False) -> Any:
    """Lower one portable two-valued predicate into a total SQL expression."""
    from sqlalchemy import and_, false, func, or_

    if op == "eq":
        return col.is_(None) if value is None else and_(col.is_not(None), _sql_predicate(col, op, value, ctx, text=text))
    if op == "ne":
        return col.is_not(None) if value is None else or_(col.is_(None), _sql_predicate(col, op, value, ctx, text=text))
    if op == "in":
        values = list(value)
        concrete = [item for item in values if item is not None]
        clauses = [and_(col.is_not(None), _sql_predicate(col, op, concrete, ctx, text=text))] if concrete else []
        if any(item is None for item in values):
            clauses.append(col.is_(None))
        return or_(*clauses) if clauses else false()
    source = func.coalesce(col, "") if op in _NULL_AS_EMPTY_OPS else col
    expression = _sql_predicate(source, op, value, ctx, text=text)
    return and_(col.is_not(None), expression) if op in _NULL_FALSE_OPS else expression


def _with_scan_fallback(py_fragment: QueryFragment, field_schema: Any, op: str, ctx: HandlerContext) -> QueryFragment:
    reason = ctx.fallback_reason(op, type(field_schema.dtype).identifier)
    return py_fragment.mark_scan(reason)


def sql_filter_handler(op: str):
    """Return a SQLAlchemy-native filter handler when the backend supports it.

    Args:
        op (str): Filter operation identifier.

    Returns:
        Callable[[Any, Any, HandlerContext], QueryFragment]: Compiler that
        produces native or scan-marked SQL fragments.
    """

    def _compile(field_schema: Any, value: Any, ctx: HandlerContext) -> QueryFragment:
        value = normalize_filter_value(field_schema.dtype, op, value, resolver=ctx.resolver)
        py_fragment = filter_handler(op)(field_schema, value, ctx)
        dialect = sql_dialect_spec(ctx.backend_identifier, resolver=ctx.resolver)
        text = stores_as_text(field_schema.dtype)
        proved = sql_text_semantics_proven(field_schema, op, ctx)

        if text and not (proved or ctx.unsafe):
            return py_fragment.mark_scan(TEXT_SEMANTICS_NOT_PROVEN)
        native_available = dialect.supports_native_filter(op) or (text and dialect.supports_native_text_filter(op))
        if text and ctx.unsafe:
            native_available = native_available or dialect.has_native_filter_compiler(op)
        if (
            (op == "wildcard" and not _can_translate_wildcard(value))
            or not native_available
            or sql_filter_requires_scan(
                field_schema.dtype,
                resolver=ctx.resolver,
            )
        ):
            return _with_scan_fallback(py_fragment, field_schema, op, ctx)

        if op == "in" and not isinstance(value, (list, tuple, set)):
            raise TypeError("in filter expects a list, tuple, or set")

        def _where(table: Any) -> Any:
            col = table.c[field_schema.name]
            return _sql_scalar_expr(col, op, value, ctx, text=text)

        return QueryFragment(
            ctx.backend,
            op,
            field_schema.name,
            _where,
            notes={
                "kind": "sql_where",
                "mode": "native",
                "row_payload": py_fragment.payload,
                "portable_proof": proved,
                "unsafe": ctx.unsafe,
                "unsafe_reasons": (TEXT_SEMANTICS_NOT_PROVEN,) if text and ctx.unsafe and not proved else (),
            },
        )

    return _compile


def register_reasons(fallback: Any, _unsupported: Any, *, resolver: Any = None) -> None:
    """Register SQL Backend fallback reasons.

    Args:
        fallback: Reason registry receiving fallback explanations.
        _unsupported: Unsupported-operation registry (unused for SQL backends).
        resolver (Any | None): Optional module resolver for backend discovery.

    Returns:
        None: Fallback reasons are registered in place.
    """
    backends = _sql_backend_identifiers_from_resolver(resolver)
    dialects = registered_sql_dialects(resolver=resolver)
    for backend in backends:
        dialect = dialects[backend]
        for op, reason in dialect.fallback_reasons.items():
            fallback.register(backend, op, "*", "*", reason)
    for backend in backends:
        dialect = dialects[backend]
        if not dialect.case_insensitive_like:
            continue
        for affix in ("starts_with", "ends_with"):
            fallback.register(backend, affix, "*", "*", f"{backend} LIKE is case-insensitive; using case-sensitive Python scan for {affix}.")


def register_handlers(registry: Any, ctx: Any) -> None:
    """Register SQL-native overrides from one snapshot-coherent claim context.

    Args:
        registry: Workspace handler registry receiving descriptors.
        ctx: Shared Registry-derived handler vocabulary.

    Returns:
        None: The supplied workspace registry is updated in place.
    """
    backend_identifiers = _sql_backend_identifiers(ctx)
    for dtype in ctx.filter_types:
        for strategy in ctx.filter_strategies:
            for op in ctx.filter_ops(dtype):
                for backend_identifier in backend_identifiers:
                    ctx.handler(dtype, op, backend_identifier, strategy, sql_filter_handler(op)).register(registry, replace=True)
    for dtype in _logical_types_with(ctx, "side"):
        for op in ctx.filter_ops(dtype):
            for backend_identifier in backend_identifiers:
                ctx.handler(dtype, op, backend_identifier, SideTable.identifier, sql_filter_handler(op)).register(
                    registry,
                    replace=True,
                )
    for dtype in _logical_types_in_domain(ctx, "json"):
        if OP_GET in ctx.filter_ops(dtype):
            for strategy in ctx.filter_strategies:
                for backend_identifier in backend_identifiers:
                    ctx.handler(dtype, OP_GET, backend_identifier, strategy, sql_json_get_handler).register(registry, replace=True)

    for dtype in _logical_types_in_domain(ctx, "array"):
        operations = frozenset(ctx.filter_ops(dtype))
        for backend_identifier in backend_identifiers:
            for strategy in ctx.filter_strategies:
                if strategy == SideTable.identifier:
                    continue
                for op in (OP_EQ, OP_NE, OP_IN):
                    if op in operations:
                        ctx.handler(
                            dtype,
                            op,
                            backend_identifier,
                            strategy,
                            sql_array_column_exact_handler(op),
                        ).register(registry, replace=True)
            ctx.handler(
                dtype,
                OP_ARRAY_CONTAINS,
                backend_identifier,
                SideTable.identifier,
                sql_array_contains_handler,
            ).register(registry, replace=True)
            for op in (OP_EQ, OP_NE, OP_IN):
                ctx.handler(
                    dtype,
                    op,
                    backend_identifier,
                    SideTable.identifier,
                    sql_array_exact_handler(op),
                ).register(registry, replace=True)
    for backend_identifier in backend_identifiers:
        ctx.handler(
            Array,
            OP_CONTAINED_IN,
            backend_identifier,
            SparseGramIndex.identifier,
            sql_sparse_gram_contained_in_handler,
        ).register(registry, replace=True)
        ctx.handler(
            Array,
            OP_ARRAY_CONTAINS,
            backend_identifier,
            SparseGramIndex.identifier,
            filter_handler(OP_ARRAY_CONTAINS),
        ).register(registry, replace=True)
    for backend_identifier in backend_identifiers:
        ctx.handler(
            HyperG,
            OP_EXISTS,
            backend_identifier,
            SideTable.identifier,
            sql_hyperg_exists_handler,
        ).register(registry, replace=True)
        ctx.handler(HyperG, OP_ALL, backend_identifier, SideTable.identifier, sql_hyperg_all_handler).register(
            registry,
            replace=True,
        )
    from .aggregate import register_aggregate_compilers

    register_aggregate_compilers(registry, resolver=ctx.resolver)


def sql_array_contains_handler(field_schema: Any, value: Any, ctx: HandlerContext) -> QueryFragment:
    """Compile SideTable item containment through the portable scan path.

    Args:
        field_schema (Any): Routed array field schema.
        value (Any): Item value validated against the array item type.
        ctx (HandlerContext): Selected backend and strategy context.

    Returns:
        QueryFragment: Scan-marked SideTable array-containment fragment.
    """
    py_fragment = filter_handler("array_contains")(field_schema, value, ctx)
    # SideTable execution joins through text-stored object_id. Until both owner
    # columns have concrete equality evidence, even numeric item predicates can
    # leak across provider-equal owners.
    return py_fragment.mark_scan(TEXT_SEMANTICS_NOT_PROVEN)


def _array_exact_item_safe(field_schema: Any, ctx: HandlerContext) -> bool:
    item_type = getattr(field_schema.dtype, "item_type", None)
    return item_type is not None and ctx.indexable(item_type)


def _array_exact_value_safe(op: str, value: Any) -> bool:
    if op in (OP_EQ, OP_NE):
        return isinstance(value, list)
    if op == OP_IN:
        return isinstance(value, (list, tuple, set)) and all(isinstance(item, list) for item in value)
    return False


def sql_array_column_exact_handler(op: str):
    """Native whole-array equality/membership for SQL main-table array storage.

    Args:
        op (str): Filter operation identifier, either ``"eq"``, ``"ne"``, or ``"in"``.

    Returns:
        Callable[[Any, Any, HandlerContext], QueryFragment]: Compiler that
        produces native or scan-marked whole-array SQL fragments.
    """

    def _compile(field_schema: Any, value: Any, ctx: HandlerContext) -> QueryFragment:
        py_fragment = filter_handler(op)(field_schema, value, ctx)
        if op == OP_IN and not isinstance(value, (list, tuple, set)):
            raise TypeError("in filter expects a list, tuple, or set")
        item_type = getattr(field_schema.dtype, "item_type", None)
        if stores_as_text(item_type) or not _array_exact_item_safe(field_schema, ctx) or not _array_exact_value_safe(op, value):
            if stores_as_text(item_type):
                return py_fragment.mark_scan(TEXT_SEMANTICS_NOT_PROVEN)
            return _with_scan_fallback(py_fragment, field_schema, op, ctx)

        values = list(value) if op == OP_IN else value

        def _where(table: Any) -> Any:
            col = table.c[field_schema.name]
            return _SQL_OP_DISPATCH[op](col, values, ctx)

        return QueryFragment(ctx.backend, op, field_schema.name, _where, notes={"kind": "sql_where", "mode": "native", "row_payload": py_fragment.payload})

    return _compile


def sql_array_exact_handler(op: str):
    """Compile whole-array equality/membership for SQL SideTable storage.

    Args:
        op (str): Filter operation identifier, either ``"eq"``, ``"ne"``, or ``"in"``.

    Returns:
        Callable[[Any, Any, HandlerContext], QueryFragment]: Compiler that
        produces scan-marked SideTable whole-array fragments.
    """

    def _compile(field_schema: Any, value: Any, ctx: HandlerContext) -> QueryFragment:
        py_fragment = filter_handler(op)(field_schema, value, ctx)
        if op == OP_IN and not isinstance(value, (list, tuple, set)):
            raise TypeError("in filter expects a list, tuple, or set")
        return py_fragment.mark_scan(TEXT_SEMANTICS_NOT_PROVEN)

    return _compile


def sql_sparse_gram_contained_in_handler(field_schema: Any, value: Any, ctx: HandlerContext) -> QueryFragment:
    """Native reverse containment for SQL-hosted sparse-GRAM strategy indexes.

    Args:
        field_schema (Any): Routed sparse-GRAM array field schema.
        value (Any): Query text validated for reverse containment.
        ctx (HandlerContext): Selected backend and strategy context.

    Returns:
        QueryFragment: Native or scan-marked sparse-GRAM contained-in fragment.
    """
    item_type = getattr(field_schema.dtype, "item_type", None)
    if item_type is None or type(item_type).identifier != ShortText.identifier:
        return QueryFragment(
            ctx.backend,
            OP_CONTAINED_IN,
            field_schema.name,
            None,
            notes={
                "unsupported": True,
                "unsupported_reason": "SQL sparse-GRAM contained_in requires Array[ShortText] storage with SparseGramIndex.",
                "strategy": ctx.strategy,
            },
        )
    py_fragment = filter_handler(OP_CONTAINED_IN)(field_schema, value, ctx)
    strategy_adapter = None
    if ctx.backend_instance is not None:
        resolve_adapter = getattr(ctx.backend_instance, "strategy_adapter", None)
        if callable(resolve_adapter):
            strategy_adapter = resolve_adapter(ctx.strategy)
    maintains_index = strategy_adapter is not None and getattr(strategy_adapter, "maintains_index", True) is not False
    scan_matcher = getattr(strategy_adapter, "contained_in_scan_matcher", None) if strategy_adapter is not None else None
    if callable(scan_matcher):
        if maintains_index:
            make_matcher = partial(scan_matcher, ctx.entity_id, field_schema.name, value)
        else:
            make_matcher = partial(scan_matcher, ctx.entity_id, field_schema.name, value, strategy_options=ctx.strategy_options)
    elif strategy_adapter is None and ctx.backend_instance is not None:
        # A backend without the adapter (a shared-database pgvector, Supabase or ClickHouse route) scans with the
        # field's own sparse-GRAM options, as every route that keeps no keyword index does.
        make_matcher = partial(_field_option_scan_matcher, ctx.backend_instance, field_schema.name, value, ctx.strategy_options)
    else:
        make_matcher = None
    if make_matcher is not None:
        match_values = None

        def _matches(row: dict[str, Any]) -> list[dict[str, Any]]:
            nonlocal match_values
            if match_values is None:
                match_values = make_matcher()
            return match_values(row.get(field_schema.name))

        def _enricher(row: dict[str, Any]) -> dict[str, Any]:
            matches = _matches(row)
            return {"match": matches} if matches else {}

        py_fragment.payload = lambda row: bool(_matches(row))
        py_fragment.notes["row_enricher"] = _enricher

    if strategy_adapter is None:
        return py_fragment.mark_scan(f"SQL backend {ctx.backend_identifier!r} has no SparseGramIndex strategy adapter; " "using Python scan fallback.")
    if not maintains_index:
        return py_fragment.mark_scan(
            f"SQL backend {ctx.backend!r} keeps no SparseGramIndex keyword index (its backend type keeps none, or its engine may "
            "commit every statement); using Python scan fallback."
        )
    return QueryFragment(
        ctx.backend,
        OP_CONTAINED_IN,
        field_schema.name,
        {"query": value},
        notes={
            "kind": "sql_sparse_gram_contained_in",
            "mode": "native",
            "row_payload": py_fragment.payload,
            "row_enricher": py_fragment.notes.get("row_enricher"),
            "metadata_columns": ["match"],
            "strategy_adapter": "sql_sparse_gram",
        },
    )


def _field_option_scan_matcher(backend: Any, field_name: str, query: Any, strategy_options: Mapping[str, Any] | None) -> Any:
    """Compile a ``contained_in`` row matcher from the field's sparse-GRAM options and the backend's defaults."""
    from ..strategies.gram import GramConfig
    from ..strategies.gram.runtime import scan_contained_in_matcher

    config = GramConfig.load(manager=backend.config).with_options(**dict(strategy_options or {}))
    return scan_contained_in_matcher(field_name, query, config, short_widths=None, resolver=getattr(backend, "module_resolver", None))


def sql_hyperg_exists_handler(field_schema: Any, value: Any, ctx: HandlerContext) -> QueryFragment:
    """Compile HyperG existential predicates through portable SideTable scan.

    Args:
        field_schema (Any): Routed HyperG field schema.
        value (Any): HyperG existential predicate payload.
        ctx (HandlerContext): Selected backend and strategy context.

    Returns:
        QueryFragment: Scan-marked HyperG exists fragment.
    """
    py_fragment = filter_handler("exists")(field_schema, value, ctx)
    predicates = normalize_filter_value(field_schema.dtype, "exists", value, resolver=ctx.resolver)
    if any(predicate_op not in SQL_HYPERG_EXISTS_OPS for ops in predicates.values() for predicate_op in ops):
        return py_fragment
    return py_fragment.mark_scan(TEXT_SEMANTICS_NOT_PROVEN)


def sql_hyperg_all_handler(field_schema: Any, value: Any, ctx: HandlerContext) -> QueryFragment:
    """Compile HyperG universal predicates through portable SideTable scan.

    Args:
        field_schema (Any): Routed HyperG field schema.
        value (Any): HyperG universal predicate payload.
        ctx (HandlerContext): Selected backend and strategy context.

    Returns:
        QueryFragment: Scan-marked HyperG all fragment.
    """
    py_fragment = filter_handler("all")(field_schema, value, ctx)
    predicates = normalize_filter_value(field_schema.dtype, "all", value, resolver=ctx.resolver)
    if any(predicate_op not in SQL_HYPERG_EXISTS_OPS for group in predicates.values() for ops in group.values() for predicate_op in ops):
        return py_fragment
    return py_fragment.mark_scan(TEXT_SEMANTICS_NOT_PROVEN)
