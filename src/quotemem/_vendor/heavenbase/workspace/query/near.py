"""Near-query normalization, route selection, and execution."""

from __future__ import annotations

from dataclasses import replace
from math import isfinite

from heavenbase.utils import Any, List, Optional

from ...execution.contracts import HandlerContext
from ...execution.contracts import QueryFragment
from ...execution.handlers.near import _batch_scores, near_handler
from ...backends.scanning import execute_scan
from ...execution.frame.merge import intersect
from ...execution.frame.result import ResultFrame
from ...filters.spec import And, Expr, Filter, Not, Or
from ...query.spec import Near, QuerySpec
from ...placement import StorageBinding, StoragePlan
from .cache import _CandidateCache

_CompiledNear = tuple[StorageBinding, QueryFragment]


def _near_filter_modes(backend: Any) -> frozenset[str]:
    """Return locally usable near/filter hints for one concrete Backend."""
    result = backend.tag("near-filter-modes")
    value = result.value if result is not None else ()
    if not isinstance(value, (list, tuple, set, frozenset)) or any(not isinstance(item, str) for item in value):
        return frozenset()
    return frozenset(value)


def _expr_backends(expr: Expr, storage_plan: StoragePlan) -> List[str]:
    """Return backends that can host the whole expression from storage placement alone."""
    if isinstance(expr, Filter):
        return [binding.backend for binding in storage_plan.field_bindings(expr.field)]
    if isinstance(expr, (And, Or)):
        groups = [_expr_backends(clause, storage_plan) for clause in expr.clauses]
        if not groups:
            return []
        first = groups[0]
        return [backend for backend in first if all(backend in group for group in groups[1:])]
    if isinstance(expr, Not):
        return _expr_backends(expr.inner, storage_plan)
    return []


def _expr_has_backend(expr: Expr, storage_plan: StoragePlan, backend: str) -> bool:
    """Return whether every field in an expression has a copy on one backend."""
    return backend in _expr_backends(expr, storage_plan)


class _NearValueService:
    """Own near-clause query-value normalization before routing."""

    def __init__(self, router: Any):
        self.router = router
        self.workspace = router.workspace

    def normalize_query(self, query: QuerySpec) -> QuerySpec:
        """Normalize near values that need query-time compute hooks."""
        ws = self.workspace
        if query.near is None:
            return query
        if query.entity_id not in ws.entities:
            raise KeyError(f"Workspace {ws.id!r} has no registered entity {query.entity_id!r}")
        near = self.normalize_near(query.near, ws.entities[query.entity_id])
        if near is query.near:
            return query
        return replace(query, near=near)

    def normalize_near(self, near: Near, entity: type) -> Near:
        """Normalize a near clause through a field query-compute hook when needed."""
        ws = self.workspace
        schema = entity.schema()
        field_schema = schema.get(near.field)
        if self.query_value_valid(field_schema, near.value):
            return near
        hook = self.query_compute_hook(entity, near.field, "near")
        if hook is None:
            if isinstance(near.value, str):
                raise ValueError(f"near on field {schema.entity_id}.{near.field} received text, but the field has no query_compute hook for op 'near'")
            return near
        computed = hook.fn(near.value)
        try:
            value = field_schema.dtype.validate(computed)
        except Exception as exc:
            dtype_id = type(field_schema.dtype).identifier
            raise ValueError(f"query_compute for field {schema.entity_id}.{near.field} returned an invalid {dtype_id} value: {exc}") from exc
        notes = dict(near.notes)
        notes.update(
            {
                "query_compute": True,
                "query_compute_fn": ws._schema_metadata.callable_ref(hook.fn),
                "query_input_type": type(near.value).__name__,
            }
        )
        max_top_k = int(ws.context.config.get("heavenbase.query.near.max_top_k", default=1000))
        return Near(near.field, value, near.top_k, near.metric, notes, exact=near.exact, cache_version=near.cache_version, max_top_k=max_top_k)

    def query_value_valid(self, field_schema: Any, value: Any) -> bool:
        """Return whether a raw query value already validates for a field."""
        try:
            field_schema.dtype.validate(value)
        except Exception:
            return False
        return True

    def query_compute_hook(self, entity: type, field_name: str, op: str) -> Optional[Any]:
        """Return the unique query-compute hook for one field/op pair."""
        spec = self.workspace._schema_metadata.field_spec(entity, field_name)
        if spec is None:
            return None
        matches = [hook for hook in spec.query_computes if op in hook.ops]
        if len(matches) > 1:
            raise ValueError(f"Field {entity.schema().entity_id}.{field_name} has multiple query_compute hooks for op {op!r}")
        return matches[0] if matches else None


class _NearRouteService:
    """Own near-fragment compilation, route selection, and execution."""

    def __init__(self, router: Any):
        self.router = router
        self.workspace = router.workspace
        self.cache = _CandidateCache()

    def eval_near(self, compiled: _CompiledNear) -> ResultFrame:
        """Execute the same near fragment that selected the query's route."""
        binding, fragment = compiled
        return self.router.execute_fragment(fragment, binding.backend, binding.entity_id)

    def compile_near(self, near: Near, entity: type, *, scan: bool = False) -> _CompiledNear:
        """Compile a near clause through its selected storage binding."""
        ws = self.workspace
        schema = entity.schema()
        field_schema = schema.get(near.field)
        binding = ws.storage_plans[schema.entity_id].first(near.field)
        if near.exact and not all(isfinite(item) for item in field_schema.dtype.validate(near.value)):
            raise ValueError("Exact near requires finite query coordinates")
        backend = ws.backends.get(binding.backend)
        try:
            handler = near_handler if scan else ws.handlers.get(field_schema.dtype, "near", backend.type, binding.strategy)
        except LookupError:
            if not near.exact:
                raise
            handler = near_handler
        context = HandlerContext(
            binding.backend,
            backend.type,
            binding.strategy,
            dict(binding.strategy_options),
            top_k=near.top_k,
            metric=near.metric,
            exact=near.exact,
            entity_id=schema.entity_id,
            backend_instance=backend,
            resolver=ws.module_resolver,
            config=ws.context.config,
            handlers=ws.handlers,
        )
        fragment = handler(field_schema, near.value, context)
        if near.exact and (fragment.uses_scan() or fragment.notes.get("exact") is not True):
            fragment = near_handler(field_schema, near.value, context)
            fragment.notes["fallback_reason"] = "Exact near requires exhaustive scoring after the complete filter; no proven native route is selected."
        return binding, fragment

    def eval_near_filter(
        self,
        compiled: _CompiledNear | None,
        expr: Optional[Expr],
        entity: type,
        *,
        mode: str,
        exact: bool,
        unsafe: bool = False,
    ) -> Optional[ResultFrame]:
        """Evaluate a near/filter pair through the shared route selector."""
        if compiled is None or expr is None:
            return None
        if mode == "unsupported" and exact:
            raise LookupError("Exact near cannot execute an unsupported candidate filter")
        if mode in ("combined_native", "combined_scan"):
            return self.eval_filtered_near(compiled, expr, entity, unsafe=unsafe)
        if mode == "metadata_prefilter":
            return self.eval_metadata_filtered_near(compiled, expr, entity, unsafe=unsafe)
        if mode == "candidate_id_prefilter":
            return self.eval_candidate_filtered_near(compiled, expr, entity, unsafe=unsafe)
        return None

    def uses_exact_scan(self, near: Near | None, compiled: _CompiledNear | None) -> bool:
        """Return whether the near alone already needs exhaustive portable scoring."""
        return near is not None and near.exact and compiled is not None and compiled[1].uses_scan()

    def exact_scan(self, near: Near, expr: Optional[Expr], entity: type, *, fields: set[str] | None = None) -> ResultFrame:
        """Filter complete streamed logical candidates before exhaustive Top-K."""
        ws = self.workspace
        binding, fragment = self.compile_near(near, entity, scan=True)
        config = ws.context.config
        batch_size = int(config.get("heavenbase.query.near.scan_batch_size", default=256))
        if batch_size <= 0:
            raise ValueError("near scan_batch_size must be positive")
        predicate = self.router._filters.compile_frame_expr(expr, entity) if expr is not None else None
        max_bytes = int(config.get("heavenbase.query.near.cache.max_bytes", default=67108864))
        max_entries = int(config.get("heavenbase.query.near.cache.max_entries", default=16))
        prepared = self.cache.prepare(entity.schema().entity_id, near.cache_version, near.field, max_bytes=max_bytes, max_entries=max_entries)
        if prepared is not None:
            fragment.notes["batch_payload"] = lambda batch: _batch_scores(
                batch, near.field, near.value, str(near.metric).lower(), fragment.payload, prepared=prepared
            )
        rows = self.cache.rows(
            entity.schema().entity_id,
            near.cache_version,
            # The query holds the workspace read lock for its whole run, so routes without a
            # native keyset page are read in one iter_scan pass instead of once per page.
            lambda: ws._row_store.iter_rows(entity.schema().entity_id, batch_size=batch_size, held=True),
            max_bytes=max_bytes,
            max_entries=max_entries,
        )
        try:
            selected = rows if predicate is None else (row for row in rows if predicate(row))
            fragment.notes["batch_size"] = batch_size
            fragment.notes["result_fields"] = fields
            fragment.bind(binding.entity_id, backend=binding.backend)
            return execute_scan(ws.backends.get(binding.backend), fragment, rows=selected)
        finally:
            rows.close()

    def eval_filtered_near(
        self,
        compiled: _CompiledNear,
        expr: Expr,
        entity: type,
        *,
        unsafe: bool = False,
    ) -> Optional[ResultFrame]:
        """Evaluate near plus filter as a combined backend fragment when possible."""
        binding, near_fragment = compiled
        if self.router._filters.expr_backend(expr, entity) != binding.backend:
            return None
        filter_fragment = self.router._filters.compile_expr(expr, entity, binding.backend, unsafe=unsafe)
        reason = filter_fragment.unsupported_reason()
        if reason:
            raise LookupError(reason)
        backend = self.workspace.backends.get(binding.backend)
        supports_native = "combined_native" in _near_filter_modes(backend)
        if not supports_native or self.fragment_uses_scan(near_fragment) or self.fragment_uses_scan(filter_fragment):
            mode = "combined_scan"  # backends without a native combined near+filter path scan even when both fragments are individually native
        else:
            mode = "combined_native"
        if isinstance(near_fragment.payload, dict):
            near_fragment.payload = {**near_fragment.payload, "filter": filter_fragment}
        near_fragment.notes.update({"filter": filter_fragment, "near_filter_mode": mode})
        return self.router.execute_fragment(near_fragment, binding.backend, binding.entity_id)

    def eval_metadata_filtered_near(
        self,
        compiled: _CompiledNear,
        expr: Expr,
        entity: type,
        *,
        unsafe: bool = False,
    ) -> Optional[ResultFrame]:
        """Evaluate near plus filter with backend metadata prefilter support."""
        ws = self.workspace
        binding, near_fragment = compiled
        backend = ws.backends.get(binding.backend)
        if "metadata_prefilter" not in _near_filter_modes(backend):
            return None
        if not self.expr_has_backend(expr, entity, binding.backend):
            return None
        filter_fragment = self.router._filters.compile_expr(expr, entity, binding.backend, unsafe=unsafe)
        if filter_fragment.unsupported_reason() or self.fragment_uses_scan(filter_fragment) or self.fragment_uses_scan(near_fragment):
            return None
        if isinstance(near_fragment.payload, dict):
            near_fragment.payload = {**near_fragment.payload, "filter": filter_fragment}
        near_fragment.notes.update({"filter": filter_fragment, "near_filter_mode": "metadata_prefilter"})
        return self.router.execute_fragment(near_fragment, binding.backend, binding.entity_id)

    def eval_candidate_filtered_near(
        self,
        compiled: _CompiledNear,
        expr: Expr,
        entity: type,
        *,
        unsafe: bool = False,
    ) -> Optional[ResultFrame]:
        """Evaluate near plus filter through candidate object-id prefilter support."""
        ws = self.workspace
        binding, near_fragment = compiled
        backend = ws.backends.get(binding.backend)
        if "candidate_id_prefilter" not in _near_filter_modes(backend):
            return None
        filt = self.router._filters.eval_expr(expr, entity, unsafe=unsafe)
        max_ids = int(ws.context.config.get("heavenbase.query.near.candidate_prefilter.max_ids", default=4096))
        if len(filt) == 0:
            return ResultFrame.empty(filt.col_names(), filt.metadata_columns)
        if len(filt) > max_ids:
            return None
        filter_fragment = backend.candidate_id_fragment(filt.ids)
        if isinstance(near_fragment.payload, dict):
            near_fragment.payload = {**near_fragment.payload, "filter": filter_fragment}
        near_fragment.notes.update(
            {"filter": filter_fragment, "near_filter_mode": "candidate_id_prefilter", "candidate_id_count": len(filt), "candidate_id_limit": max_ids}
        )
        return intersect([self.router.execute_fragment(near_fragment, binding.backend, binding.entity_id), filt])

    def near_filter_route(
        self, near: Optional[Near], expr: Optional[Expr], entity: type, *, unsafe: bool = False, compiled: _CompiledNear | None = None
    ) -> str:
        """Return the selected near/filter execution route for execute() and explain()."""
        if near is None or expr is None:
            return "none"
        ws = self.workspace
        binding, near_fragment = compiled if compiled is not None else self.compile_near(near, entity)
        if near.exact:
            if self.expr_has_unsupported(expr, entity, unsafe=unsafe):
                return "unsupported"  # execute() raises LookupError; explain() reports the filter step's unsupported_reason
            backend = ws.backends.get(binding.backend)
            modes = _near_filter_modes(backend)
            if not self.fragment_uses_scan(near_fragment) and self.expr_has_backend(expr, entity, binding.backend):
                filter_fragment = self.router._filters.compile_expr(expr, entity, binding.backend, unsafe=unsafe)
                if not filter_fragment.unsupported_reason() and not self.fragment_uses_scan(filter_fragment):
                    if self.router._filters.expr_backend(expr, entity) == binding.backend and "combined_native" in modes:
                        return "combined_native"
                    if "metadata_prefilter" in modes:
                        return "metadata_prefilter"
            return "exact_scan"
        if self.router._filters.expr_backend(expr, entity) != binding.backend:
            backend = ws.backends.get(binding.backend)
            if "metadata_prefilter" in _near_filter_modes(backend) and self.expr_has_backend(expr, entity, binding.backend):
                filter_fragment = self.router._filters.compile_expr(expr, entity, binding.backend, unsafe=unsafe)
                if not filter_fragment.unsupported_reason() and not self.fragment_uses_scan(filter_fragment) and not self.fragment_uses_scan(near_fragment):
                    return "metadata_prefilter"
            if self.expr_has_unsupported(expr, entity, unsafe=unsafe):
                return "unsupported"
            return "candidate_id_prefilter" if "candidate_id_prefilter" in _near_filter_modes(backend) else "post_filter"
        if self.expr_has_unsupported(expr, entity, unsafe=unsafe):
            return "unsupported"
        filter_fragment = self.router._filters.compile_expr(expr, entity, binding.backend, unsafe=unsafe)
        backend = ws.backends.get(binding.backend)
        supports_native = "combined_native" in _near_filter_modes(backend)
        if not supports_native or self.fragment_uses_scan(near_fragment) or self.fragment_uses_scan(filter_fragment):
            return "combined_scan"  # backends without a native combined near+filter path scan even when both fragments are individually native
        return "combined_native"

    def fragment_uses_scan(self, fragment: QueryFragment) -> bool:
        """Return whether a query fragment falls back to Python scan semantics."""
        return fragment.uses_scan()

    def expr_has_unsupported(self, expr: Expr, entity: type, *, unsafe: bool = False) -> bool:
        """Return whether an expression contains an unsupported filter fragment."""
        if isinstance(expr, Filter):
            return self.router._filters.compile_filter(expr, entity, unsafe=unsafe).unsupported_reason() is not None
        if isinstance(expr, And):
            return any(self.expr_has_unsupported(clause, entity, unsafe=unsafe) for clause in expr.clauses)
        if isinstance(expr, Or):
            return any(self.expr_has_unsupported(clause, entity, unsafe=unsafe) for clause in expr.clauses)
        if isinstance(expr, Not):
            return self.expr_has_unsupported(expr.inner, entity, unsafe=unsafe)
        return False

    def expr_has_backend(self, expr: Expr, entity: type, backend: str) -> bool:
        """Return whether an expression includes fields stored on a backend."""
        ws = self.workspace
        return _expr_has_backend(expr, ws.storage_plans[entity.schema().entity_id], backend)
