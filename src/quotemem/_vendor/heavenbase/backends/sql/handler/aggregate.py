"""Parity-safe SQL aggregate compiler producing provider-ready fragments."""

from __future__ import annotations

__all__ = ["register_aggregate_compilers", "sql_aggregate_compiler"]

from heavenbase.utils import Any, Optional

from heavenbase.execution.contracts import HandlerContext
from heavenbase.execution.contracts import QueryFragment
from heavenbase.backends.sql.capabilities import registered_sql_dialects
from heavenbase.modules import ModuleResolver
from heavenbase.types.storage import stores_as_text
from heavenbase.execution.handlers.aggregate import AggregateCompileResult, AggregateCompiler, AggregateCompilerInput
from .text import TEXT_SEMANTICS_NOT_PROVEN, sql_text_key_expression, sql_text_key_ops, sql_text_key_semantics_proven

_EXACT_GROUP_TYPES = {"boolean", "date", "duration", "integer", "timestamp"}
_EXACT_DISTINCT_TYPES = {"boolean", "date", "duration", "integer", "timestamp"}
_EXACT_ORDER_TYPES = {"date", "duration", "integer", "timestamp"}


def _fragment_has_portable_proof(fragment: Optional[QueryFragment]) -> bool:
    """Return whether a native filter tree avoided unproved provider semantics."""
    if fragment is None:
        return True
    if fragment.notes.get("portable_proof") is False:
        return False
    children = fragment.payload if isinstance(fragment.payload, (list, tuple)) else ()
    return all(_fragment_has_portable_proof(child) for child in children if isinstance(child, QueryFragment))


def _text_keys_have_portable_proof(requirements: list[tuple[Any, str]], ctx: HandlerContext) -> bool:
    """Return whether every text-key requirement has concrete provider proof."""
    return all(not stores_as_text(field_schema.dtype) or purpose in sql_text_key_ops(field_schema, ctx) for field_schema, purpose in requirements)


def _compile_sql(
    schema: Any,
    aggregate: AggregateCompilerInput,
    where: Optional[QueryFragment],
    ctx: HandlerContext,
) -> AggregateCompileResult:
    """Compile a validated aggregate into a lazy SQLAlchemy statement builder."""
    if where is not None and where.uses_scan():
        return AggregateCompileResult.decline("filter_not_native")
    group_fields = tuple(aggregate.group_by)
    group_schemas = [schema.get(field_name) for field_name in group_fields]
    group_types = [type(field_schema.dtype).identifier for field_schema in group_schemas]
    if "float" in group_types:
        return AggregateCompileResult.decline("floating_group_parity_not_proven")
    if any(
        stores_as_text(field_schema.dtype) and not (sql_text_key_semantics_proven(field_schema, "group", ctx) or ctx.unsafe) for field_schema in group_schemas
    ):
        return AggregateCompileResult.decline(TEXT_SEMANTICS_NOT_PROVEN)
    if any(not stores_as_text(field_schema.dtype) and dtype not in _EXACT_GROUP_TYPES for field_schema, dtype in zip(group_schemas, group_types)):
        return AggregateCompileResult.decline("text_collation_parity_not_proven")
    if group_fields and not (sql_text_key_semantics_proven(schema.get(schema.pk_name), "order", ctx) or ctx.unsafe):
        return AggregateCompileResult.decline(TEXT_SEMANTICS_NOT_PROVEN)
    order_schemas = [schema.get(metric.field) for metric in aggregate.metrics.values() if metric.op in ("min", "max")]
    order_types = [type(field_schema.dtype).identifier for field_schema in order_schemas]
    if "float" in order_types:
        return AggregateCompileResult.decline("floating_order_parity_not_proven")
    if any(
        stores_as_text(field_schema.dtype) and not (sql_text_key_semantics_proven(field_schema, "order", ctx) or ctx.unsafe) for field_schema in order_schemas
    ):
        return AggregateCompileResult.decline(TEXT_SEMANTICS_NOT_PROVEN)
    if any(not stores_as_text(field_schema.dtype) and dtype not in _EXACT_ORDER_TYPES for field_schema, dtype in zip(order_schemas, order_types)):
        return AggregateCompileResult.decline("text_collation_parity_not_proven")
    distinct_schemas = [schema.get(metric.field) for metric in aggregate.metrics.values() if metric.op == "count" and metric.options.get("distinct", False)]
    distinct_types = [type(field_schema.dtype).identifier for field_schema in distinct_schemas]
    if "float" in distinct_types:
        return AggregateCompileResult.decline("floating_distinct_parity_not_proven")
    if any(
        stores_as_text(field_schema.dtype) and not (sql_text_key_semantics_proven(field_schema, "distinct", ctx) or ctx.unsafe)
        for field_schema in distinct_schemas
    ):
        return AggregateCompileResult.decline(TEXT_SEMANTICS_NOT_PROVEN)
    if any(not stores_as_text(field_schema.dtype) and dtype not in _EXACT_DISTINCT_TYPES for field_schema, dtype in zip(distinct_schemas, distinct_types)):
        return AggregateCompileResult.decline("text_collation_parity_not_proven")
    key_requirements = [
        *[(field_schema, "group") for field_schema in group_schemas],
        *([(schema.get(schema.pk_name), "order")] if group_fields else []),
        *[(field_schema, "order") for field_schema in order_schemas],
        *[(field_schema, "distinct") for field_schema in distinct_schemas],
    ]
    portable_proof = _fragment_has_portable_proof(where) and _text_keys_have_portable_proof(key_requirements, ctx)
    metric_specs = tuple((output, metric.op, metric.field, dict(metric.options)) for output, metric in aggregate.metrics.items())

    def build(table: Any, where_clause: Any = None) -> Any:
        from sqlalchemy import func, select

        group_expressions = [sql_text_key_expression(table.c[field_name], schema.get(field_name), "group", ctx) for field_name in group_fields]
        expressions = [expression.label(field_name) for expression, field_name in zip(group_expressions, group_fields)]
        functions = {
            "count": func.count,
            "min": func.min,
            "max": func.max,
        }
        for output, op, field_name, options in metric_specs:
            field_schema = schema.get(field_name)
            source = table.c[field_name]
            if op == "count" and options.get("distinct", False):
                source = func.distinct(sql_text_key_expression(source, field_schema, "distinct", ctx))
            elif op in ("min", "max"):
                source = sql_text_key_expression(source, field_schema, "order", ctx)
            expressions.append(functions[op](source).label(output))
        statement = select(*expressions)
        if where_clause is not None:
            statement = statement.where(where_clause)
        if group_fields:
            statement = statement.group_by(*group_expressions)
            object_id = sql_text_key_expression(table.c.object_id, schema.get(schema.pk_name), "order", ctx)
            statement = statement.order_by(func.min(object_id))
        return statement

    columns = [*aggregate.group_by, *aggregate.metrics]
    casts = {output: "int" for output, metric in aggregate.metrics.items() if metric.op == "count"}
    empty_rows = []
    if not aggregate.group_by:
        empty_rows = [{output: 0 if metric.op == "count" else None for output, metric in aggregate.metrics.items()}]
    fragment = QueryFragment(
        ctx.backend,
        "aggregate",
        "*",
        {"build": build, "where": where, "columns": columns, "casts": casts, "empty_rows": empty_rows},
        notes={
            "kind": "sql_aggregate",
            "mode": "native",
            "compiler": ctx.backend_identifier,
            "unsafe": ctx.unsafe,
            "portable_proof": portable_proof,
            "unsafe_reasons": (TEXT_SEMANTICS_NOT_PROVEN,) if ctx.unsafe and not portable_proof else (),
        },
    )
    return AggregateCompileResult.native(fragment)


def sql_aggregate_compiler(
    backend_identifier: str,
    operations: tuple[str, ...],
    *,
    resolver: ModuleResolver | None = None,
) -> AggregateCompiler:
    """Return one registered-capability descriptor for safe SQL aggregation.

    Args:
        backend_identifier (str): Canonical SQL backend identifier.
        operations (tuple[str, ...]): Native aggregate operations supported by the backend.
        resolver (ModuleResolver | None): Optional module resolver for backend discovery.

    Returns:
        AggregateCompiler: Registered SQL aggregate compiler descriptor.
    """
    return AggregateCompiler(backend_identifier, operations, _compile_sql, resolver=resolver)


def register_aggregate_compilers(
    registry: Any,
    *,
    resolver: ModuleResolver | None = None,
) -> None:
    """Register SQL aggregate compilers implemented by this handler package.

    Args:
        registry: Workspace handler registry receiving aggregate compilers.
        resolver (ModuleResolver | None): Optional module resolver for backend discovery.

    Returns:
        None: Supported SQL aggregate compilers are registered in place.
    """
    for backend_identifier, dialect in registered_sql_dialects(resolver=resolver).items():
        if not dialect.native_aggregate_ops:
            continue
        try:
            compiler = sql_aggregate_compiler(backend_identifier, dialect.native_aggregate_ops, resolver=resolver)
        except KeyError:
            continue  # SQLAlchemy dialect alias without a registered HeavenBase backend type.
        registry.register_aggregate_compiler(compiler, replace=True)
