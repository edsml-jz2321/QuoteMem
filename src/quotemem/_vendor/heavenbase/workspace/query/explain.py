"""Workspace query explain rendering helpers."""

from __future__ import annotations

from heavenbase.utils import Any, Dict, List

from ...filters.spec import And, Expr, Filter, Not, Or
from ...query.json_path import json_path_columns
from ...query.spec import Near, QuerySpec
from ...placement import StoragePlan
from .near import _CompiledNear


def _binding_rows(storage_plan: StoragePlan, field: str, selected: str | None = None) -> List[dict]:
    """Return explain rows for all bindings of one field."""
    return [dict(binding.to_dict(), selected=binding.backend == selected) for binding in storage_plan.field_bindings(field)]


class _QueryExplainService:
    """Own query explain assembly while the router owns execution."""

    def __init__(self, router: Any):
        self.router = router
        self.workspace = router.workspace

    def explain(self, query: QuerySpec) -> Dict[str, Any]:
        """Return storage, handler, and fallback diagnostics for a query."""
        ws = self.workspace
        query = self.router._near_values.normalize_query(query)
        entity = ws.entities[query.entity_id]
        steps = []
        if query.traverse is not None:
            steps.append(self.router._traverse.plan_step(query, entity))
        if query.filter is not None:
            steps.extend(self.explain_expr(query.filter, entity, unsafe=query.unsafe))
        if query.near is not None:
            compiled = self.router._near.compile_near(query.near, entity)
            near_filter_mode = self.router._near.near_filter_route(query.near, query.filter, entity, unsafe=query.unsafe, compiled=compiled)
            if near_filter_mode == "metadata_prefilter":
                self.mark_metadata_pruned_steps(steps, query.near, entity)
            if near_filter_mode == "exact_scan":
                compiled = self.router._near.compile_near(query.near, entity, scan=True)
            step = self.explain_near(query.near, entity, compiled)
            step["near_filter_mode"] = near_filter_mode
            steps.append(step)
        if query.aggregate is not None:
            steps.append(self.router._aggregate.plan_step(query, entity))
        else:
            json_paths = json_path_columns(entity.schema(), [*(order.field for order in query.order), *query.select])
            if json_paths:
                # dotted order/select columns derive per row in Python after backend merge; report the py route honestly
                steps.append(
                    {
                        "node": "json_path",
                        "fields": list(json_paths),
                        "logical_type": "json",
                        "handler_mode": "py_scan",
                        "fallback_reason": "Dotted Json path columns derive in Python after backend merge for order and select.",
                    }
                )
        return {"workspace": ws.id, "query": query.to_dict(), "storage_plan": ws.storage_plans[query.entity_id].to_dict(), "steps": steps}

    def explain_expr(self, expr: Expr, entity: type, *, unsafe: bool = False) -> List[dict]:
        """Explain how an expression will execute."""
        ws = self.workspace
        if isinstance(expr, Filter):
            schema = entity.schema()
            field_schema = schema.get(expr.field)
            binding = ws.storage_plans[schema.entity_id].first(expr.field)
            backend = ws.backends.get(binding.backend)
            fragment = self.router._filters.compile_filter(expr, entity, unsafe=unsafe)
            step: Dict[str, Any] = {
                "node": "filter",
                "field": expr.field,
                "op": expr.op,
                "handler_op": expr.op,
                "logical_type": type(field_schema.dtype).identifier,
                "backend": binding.backend,
                "type": backend.type,
                "strategy": binding.strategy,
                **({"strategy_options": dict(binding.strategy_options)} if binding.strategy_options else {}),
                "field_storage": _binding_rows(ws.storage_plans[schema.entity_id], expr.field, binding.backend),
                "handler": f"{type(field_schema.dtype).identifier}.{expr.op}.{backend.type}.{binding.strategy}",
                "handler_kind": fragment.kind("row_payload"),
                "handler_mode": fragment.mode(),
                "strategy_adapter": fragment.notes.get("strategy_adapter"),
                "fallback_reason": fragment.fallback_reason(),
                "unsupported_reason": fragment.unsupported_reason(),
                "portable_proof": fragment.notes.get("portable_proof"),
                "unsafe": unsafe,
            }
            unsafe_reasons = list(fragment.notes.get("unsafe_reasons", ()))
            if unsafe_reasons:
                step["unsafe_reasons"] = unsafe_reasons
            return [step]
        if isinstance(expr, And):
            clauses = self.router._filters.flat_clauses(expr, And)
            return (
                [{"node": "and", "n": len(clauses)}]
                + self.explain_groups(clauses, entity, "and")
                + [step for clause in clauses for step in self.explain_expr(clause, entity, unsafe=unsafe)]
            )
        if isinstance(expr, Or):
            clauses = self.router._filters.flat_clauses(expr, Or)
            return (
                [{"node": "or", "n": len(clauses)}]
                + self.explain_groups(clauses, entity, "or")
                + [step for clause in clauses for step in self.explain_expr(clause, entity, unsafe=unsafe)]
            )
        if isinstance(expr, Not):
            return [{"node": "not"}] + self.explain_expr(expr.inner, entity, unsafe=unsafe)
        return [{"node": type(expr).__name__}]

    def explain_groups(self, clauses: List[Expr], entity: type, op: str) -> List[dict]:
        """Explain grouped backend execution for logical clauses."""
        groups: Dict[str, int] = {}
        for clause in clauses:
            backend = self.router._filters.expr_backend(clause, entity)
            if backend is not None:
                groups[backend] = groups.get(backend, 0) + 1
        return [{"node": "backend_group", "op": op, "backend": backend, "n": n} for backend, n in groups.items() if n > 1]

    def mark_metadata_pruned_steps(self, steps: list[dict], near: Near, entity: type) -> None:
        """Mark scalar filter steps covered by a vector metadata prefilter."""
        ws = self.workspace
        schema = entity.schema()
        binding = ws.storage_plans[schema.entity_id].first(near.field)
        pruned_backends = []
        for step in steps:
            if step.get("node") != "filter" or step.get("backend") == binding.backend:
                continue
            if any(storage.get("backend") == binding.backend for storage in step.get("field_storage", [])):
                step["pruned"] = True
                step["pruned_by"] = binding.backend
                step["prune_reason"] = "covered_by_near_metadata_prefilter"
                if step["backend"] not in pruned_backends:
                    pruned_backends.append(step["backend"])
        steps.extend(
            {
                "node": "pruned_backend_group",
                "backend": backend,
                "pruned_by": binding.backend,
                "reason": "covered_by_near_metadata_prefilter",
            }
            for backend in pruned_backends
        )

    def explain_near(self, near: Near, entity: type, compiled: _CompiledNear) -> Dict[str, Any]:
        """Explain near-clause handler selection and execution mode."""
        schema = entity.schema()
        field_schema = schema.get(near.field)
        binding, fragment = compiled
        backend = self.workspace.backends.get(binding.backend)
        config = self.workspace.context.config
        max_bytes = config.get("heavenbase.query.near.cache.max_bytes", default=67108864)
        max_entries = config.get("heavenbase.query.near.cache.max_entries", default=16)
        reuses = False
        if near.exact and fragment.uses_scan():
            max_bytes, max_entries = int(max_bytes), int(max_entries)
            if max_bytes < 0 or max_entries < 0:
                raise ValueError("near cache bounds must be non-negative")
            reuses = near.cache_version is not None and max_bytes > 0 and max_entries > 0
        return {
            **dict(near.notes),
            "node": "near",
            "field": near.field,
            "op": "near",
            "handler_op": "near",
            "logical_type": type(field_schema.dtype).identifier,
            "backend": binding.backend,
            "type": backend.type,
            "strategy": binding.strategy,
            **({"strategy_options": dict(binding.strategy_options)} if binding.strategy_options else {}),
            "metric": near.metric,
            "top_k": near.top_k,
            "cache": {
                "version": near.cache_version,
                "mode": "versioned_candidates" if reuses else "live",
                "max_bytes": max_bytes,
                "max_entries": max_entries,
                "scope": "workspace_entity",
            },
            **({"exact_required": True, "exact_guaranteed": fragment.notes.get("exact") is True} if near.exact else {}),
            "handler": f"{type(field_schema.dtype).identifier}.near.{backend.type}.{binding.strategy}",
            "handler_kind": fragment.kind("row_payload"),
            "handler_mode": fragment.mode(),
            "strategy_adapter": fragment.notes.get("strategy_adapter"),
            "fallback_reason": fragment.fallback_reason(),
            "unsupported_reason": fragment.unsupported_reason(),
        }
