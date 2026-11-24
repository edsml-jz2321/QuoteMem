"""Aggregate validation, native compiler routing, fold execution, and explain."""

from __future__ import annotations

from functools import partial

from heavenbase.utils import Any, Optional, dataclass

from ...execution.contracts import HandlerContext
from ...execution.contracts import QueryFragment
from ...execution.frame.aggregate import aggregate_frame, filter_frame
from ...filters.spec import And, Expr, Filter, Not, Or
from ...execution.frame.result import ResultFrame
from ...query.aggregate import having_predicate, reducer_for, validate_aggregate, validate_having
from ...query.json_path import json_path_columns
from ...query.spec import Aggregate, QuerySpec
from ...strategies import InlineColumn


@dataclass(frozen=True)
class _AggregateRoute:
    """One pre-execution aggregate route decision."""

    mode: str
    backend: Optional[str] = None
    compiler: Optional[str] = None
    fragment: Optional[QueryFragment] = None
    reason: Optional[str] = None


class _AggregateRouteService:
    """Own aggregate routing independently of row selection and providers."""

    def __init__(self, router: Any):
        self.router = router
        self.workspace = router.workspace

    def validate(self, aggregate: Aggregate, entity: type) -> None:
        """Validate one normalized clause against its entity schema."""
        validate_aggregate(entity.schema(), aggregate, resolver=self.workspace.module_resolver)

    def plan(self, query: QuerySpec, entity: type) -> _AggregateRoute:
        """Select a registered native compiler or a named portable fold."""
        aggregate = query.aggregate
        if aggregate is None:
            raise ValueError("Aggregate route planning requires QuerySpec.aggregate")
        schema = entity.schema()
        self.validate(aggregate, entity)
        validate_having(schema, aggregate, query.having, resolver=self.workspace.module_resolver, config=self.workspace.context.config)
        if query.near is not None:
            return _AggregateRoute("py_fold", reason="near_clause_present")
        if query.traverse is not None:
            return _AggregateRoute("py_fold", reason="traverse_clause_present")
        if json_path_columns(entity.schema(), aggregate.group_by):
            return _AggregateRoute("py_fold", reason="json_path_group_by_present")

        candidates = self._candidate_backends(aggregate, entity)
        if not candidates:
            return _AggregateRoute("py_fold", reason="aggregate_fields_not_colocated_inline")
        dependencies = set(self.fold_backend_names(query, entity))
        plan = self.workspace.storage_plans[schema.entity_id]
        identity_backends = {binding.backend for binding in plan.field_bindings(schema.pk_name)}
        last_reason = "native_aggregate_compiler_not_registered"
        for backend_name in candidates:
            if dependencies - {backend_name}:
                last_reason = "aggregate_dependencies_not_colocated"
                continue
            if identity_backends != {backend_name}:
                last_reason = "aggregate_identity_not_single_route"
                continue
            backend = self.workspace.backends.get(backend_name)
            compiler = self.workspace.handlers.aggregate_compiler(backend.type)
            if compiler is None:
                continue
            where, reason = self._where_fragment(query, entity, backend_name)
            if reason is not None:
                last_reason = reason
                continue
            result = compiler(
                schema,
                aggregate,
                where,
                HandlerContext(
                    backend_name,
                    backend.type,
                    InlineColumn.identifier,
                    backend_instance=backend,
                    entity_id=schema.entity_id,
                    unsafe=query.unsafe,
                    resolver=self.workspace.module_resolver,
                    config=self.workspace.context.config,
                    handlers=self.workspace.handlers,
                ),
            )
            if result.fragment is not None:
                return _AggregateRoute("native", backend_name, compiler.backend_identifier, result.fragment)
            last_reason = result.reason or last_reason
        return _AggregateRoute("py_fold", reason=last_reason)

    def execute_native(self, route: _AggregateRoute, entity_id: str) -> ResultFrame:
        """Execute one provider-ready native fragment without query-AST delegation."""
        if route.mode != "native" or route.backend is None or route.fragment is None:
            raise ValueError("execute_native requires a native aggregate route")
        route.fragment.bind(entity_id, backend=route.backend)
        return self.workspace.backends.get(route.backend).execute(route.fragment)

    def execute_fold(self, frame: ResultFrame, aggregate: Aggregate, entity: type) -> ResultFrame:
        """Execute the portable reducer oracle over complete materialized rows."""
        self.validate(aggregate, entity)
        resolver = self.workspace.module_resolver
        metrics = {
            output: (
                metric.field,
                partial(
                    reducer_for(metric.op, resolver=resolver).reduce,
                    options=metric.options,
                    config=self.workspace.context.config,
                ),
            )
            for output, metric in aggregate.metrics.items()
        }
        return aggregate_frame(frame, group_by=aggregate.group_by, metrics=metrics)

    def execute_having(self, frame: ResultFrame, query: QuerySpec, entity: type) -> ResultFrame:
        """Apply the universal post-aggregate `HAVING` predicate."""
        if query.aggregate is None:
            raise ValueError("HAVING execution requires QuerySpec.aggregate")
        if query.having is None:
            return frame
        schema = entity.schema()
        resolver = self.workspace.module_resolver
        match = having_predicate(schema, query.aggregate, query.having, resolver=resolver, config=self.workspace.context.config)
        return filter_frame(frame, match)

    def fold_backend_names(self, query: QuerySpec, entity: type) -> list[str]:
        """Return routed backends needed by one exact portable fold.

        Aggregate sources and pre-aggregate predicates are direct query
        dependencies; dotted Json path groups depend on their base field route.
        Required and defaulted fields are included so the normal query-time
        schema-version check remains intact. Optional unrelated routes are
        deliberately excluded; a sparse vector route must not block an
        aggregate that never reads it.
        """
        if query.aggregate is None:
            raise ValueError("Aggregate fold dependency planning requires QuerySpec.aggregate")
        schema = entity.schema()
        json_paths = json_path_columns(schema, query.aggregate.group_by)
        fields = {
            *(json_paths[name][0] if name in json_paths else name for name in query.aggregate.group_by),
            *(metric.field for metric in query.aggregate.metrics.values()),
            *_expr_fields(query.filter),
            *(field.name for field in schema.fields.values() if field.required or field.has_default),
        }
        fields.discard(schema.pk_name)
        fields.discard("object_id")
        plan = self.workspace.storage_plans[schema.entity_id]
        backends = []
        for binding in plan.bindings:
            if binding.field_name in fields and binding.backend not in backends:
                backends.append(binding.backend)
        return backends

    def plan_step(self, query: QuerySpec, entity: type) -> dict[str, Any]:
        """Render the exact route decision shared with execution."""
        aggregate = query.aggregate
        if aggregate is None:
            raise ValueError("Aggregate explain planning requires QuerySpec.aggregate")
        route = self.plan(query, entity)
        step = {
            "node": "aggregate",
            "group_by": list(aggregate.group_by),
            "metrics": {name: metric.to_dict() for name, metric in aggregate.metrics.items()},
            "handler_mode": route.mode,
            "fallback_reason": route.reason,
        }
        if query.having is not None:
            step.update({"having": query.having.to_dict(), "having_mode": "py_post_filter"})
        if route.backend is not None:
            backend = self.workspace.backends.get(route.backend)
            step.update({"backend": route.backend, "type": backend.type, "compiler": route.compiler})
        if route.fragment is not None:
            step.update(
                {
                    "portable_proof": route.fragment.notes.get("portable_proof"),
                    "unsafe": route.fragment.notes.get("unsafe", query.unsafe),
                }
            )
            unsafe_reasons = list(route.fragment.notes.get("unsafe_reasons", ()))
            if unsafe_reasons:
                step["unsafe_reasons"] = unsafe_reasons
        return step

    def _candidate_backends(self, aggregate: Aggregate, entity: type) -> list[str]:
        """Return inline backends storing every aggregate source field."""
        plan = self.workspace.storage_plans[entity.schema().entity_id]
        fields = {*aggregate.group_by, *(metric.field for metric in aggregate.metrics.values())}
        ordered = []
        for binding in plan.bindings:
            if binding.backend not in ordered:
                ordered.append(binding.backend)
        return [
            backend_name for backend_name in ordered if all(plan.has_binding(field_name, backend=backend_name, strategy=InlineColumn) for field_name in fields)
        ]

    def _where_fragment(self, query: QuerySpec, entity: type, backend_name: str) -> tuple[Optional[QueryFragment], Optional[str]]:
        """Compile a colocated filter for one candidate backend without executing it."""
        if query.filter is None:
            return None, None
        try:
            fragment = self.router._filters.compile_expr(query.filter, entity, backend_name, unsafe=query.unsafe)
        except LookupError:
            return None, "filter_fields_not_colocated"
        unsupported = fragment.unsupported_reason()
        if unsupported:
            return None, unsupported
        if fragment.uses_scan():
            reason = fragment.fallback_reason()
            return None, reason if reason == "text_semantics_not_proven" else "filter_not_native"
        return fragment, None


def _expr_fields(expr: Optional[Expr]) -> set[str]:
    """Return every entity field referenced by one filter expression tree."""
    if expr is None:
        return set()
    if isinstance(expr, Filter):
        return {expr.field}
    if isinstance(expr, (And, Or)):
        return {field for clause in expr.clauses for field in _expr_fields(clause)}
    if isinstance(expr, Not):
        return _expr_fields(expr.inner)
    raise TypeError(f"Unsupported expression {type(expr).__name__}")
