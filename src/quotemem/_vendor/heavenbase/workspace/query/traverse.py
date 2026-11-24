"""Private routed GraphEdge traversal helpers for workspace queries."""

from __future__ import annotations

from dataclasses import replace

from heavenbase.utils import Any, Dict, List, Optional

from ...filters.spec import And, Expr, Filter
from ...modules import ModuleResolver
from ...query.spec import OrderKey, QuerySpec
from ..graph.edges import resolve_edge
from ..graph.traversal import _native_graph_backend, reached_ids


def _graph_endpoint_pushdown(backend: Any) -> bool:
    """Return one Backend's unverified endpoint-pushdown declaration."""
    result = backend.tag("graph-endpoint-pushdown")
    return result is not None and result.value is True


def pushable_endpoint_preds(
    expr: Optional[Expr],
    schema: Any,
    storage_plan: Any,
    backend: str,
    *,
    resolver: ModuleResolver | None = None,
) -> List[tuple]:
    """Extract scalar eq/range/in leaves that can safely fuse into a traverse endpoint filter.

    Predicate pushdown is placement-aware: a field filter can be fused only when
    that field is stored on the same backend that executes the graph traversal.

    Args:
        expr (Expr | None): Filter expression to inspect.
        schema (Any): Entity schema used to normalize filter operands.
        storage_plan (Any): Storage plan describing field placement.
        backend (str): Backend name executing the graph traversal.
        resolver (ModuleResolver | None): Module resolver used for filter-value
            normalization. Defaults to ``None``.

    Returns:
        List[tuple]: Pushable ``(field, op, value)`` predicate leaves.
    """
    if expr is None:
        return []
    from ...execution.handlers.filters import normalize_filter_value  # lazy: keep handlers off the eager workspace-import path

    pushable_ops = {"eq", "lt", "lte", "gt", "gte", "in"}
    leaves = expr.clauses if isinstance(expr, And) else [expr] if isinstance(expr, Filter) else []
    preds: List[tuple] = []
    for leaf in leaves:
        if not isinstance(leaf, Filter) or leaf.op not in pushable_ops:
            continue
        if not any(b.backend == backend for b in storage_plan.field_bindings(leaf.field)):
            continue
        value = leaf.value
        try:
            value = normalize_filter_value(schema.get(leaf.field).dtype, leaf.op, value, resolver=resolver)
        except (ValueError, TypeError, KeyError, AttributeError):
            continue
        if leaf.op == "in":
            if isinstance(value, (list, tuple, set)) and all(isinstance(item, (str, int, float, bool)) for item in value):
                preds.append((leaf.field, leaf.op, list(value)))
        elif isinstance(value, (str, int, float, bool)):
            preds.append((leaf.field, leaf.op, value))
    return preds


class _TraverseRouteService:
    """Own routed ``QuerySpec.traverse`` validation, filter fusion, and explain metadata."""

    def __init__(self, workspace: Any):
        self.workspace = workspace

    def resolve_edge(self, query: QuerySpec, entity: type) -> tuple:
        """Resolve and validate a routed traverse edge without executing traversal."""
        field_ref = getattr(entity, query.traverse.field)
        _entity_id, _field_name, binding, _endpoint_col, target_entity = resolve_edge(self.workspace, field_ref)
        if target_entity != query.entity_id:
            raise ValueError(
                f"chained .traverse({query.traverse.field!r}) reaches entity {target_entity!r}, not the queried {query.entity_id!r}; "
                f"routed traverse requires a self-referential edge - use ws.traverse(...) for cross-entity traversal"
            )
        return field_ref, binding, target_entity

    def resolve(self, query: QuerySpec, entity: type, endpoint_preds: Optional[List[tuple]] = None) -> tuple:
        """Resolve a routed traverse op: (field_ref, reachable object_ids, mode, target_entity)."""
        field_ref, _binding, target_entity = self.resolve_edge(query, entity)
        traverse = query.traverse
        ids, mode, _target = reached_ids(
            self.workspace,
            field_ref,
            traverse.start,
            min_hops=traverse.min_hops,
            max_hops=traverse.max_hops,
            direction=traverse.direction,
            endpoint_preds=endpoint_preds,
        )
        return field_ref, ids, mode, target_entity

    def apply(self, query: QuerySpec, entity: type) -> tuple:
        """Fold routed traversal into an object-id filter while preserving stable pagination."""
        pk = entity.schema().pk_name
        _edge_ref, binding, _edge_target = self.resolve_edge(query, entity)
        storage_plan = self.workspace.storage_plans[query.entity_id]
        backend = _native_graph_backend(self.workspace, query.entity_id, query.traverse.field, binding)
        can_push = backend is not None and _graph_endpoint_pushdown(backend)
        endpoint_preds = (
            pushable_endpoint_preds(
                query.filter,
                entity.schema(),
                storage_plan,
                binding.backend,
                resolver=self.workspace.module_resolver,
            )
            if can_push
            else []
        )
        _field_ref, ids, mode, _target = self.resolve(query, entity, endpoint_preds)
        edge_filter = Filter(pk, "in", list(ids))
        new_filter = edge_filter if query.filter is None else And([edge_filter, query.filter])
        order = query.order
        if query.aggregate is None and order and order[-1].field != pk:
            order = (*order, OrderKey(pk, False))
        return replace(query, filter=new_filter, traverse=None, order=order), mode

    def plan_step(self, query: QuerySpec, entity: type) -> Dict[str, Any]:
        """Return the honest explain step for a routed traverse without executing traversal."""
        traverse = query.traverse
        _field_ref, binding, target_entity = self.resolve_edge(query, entity)
        backend = _native_graph_backend(self.workspace, query.entity_id, traverse.field, binding)
        native = backend is not None
        storage_plan = self.workspace.storage_plans[query.entity_id]
        fused = (
            native
            and backend is not None
            and _graph_endpoint_pushdown(backend)
            and bool(
                pushable_endpoint_preds(
                    query.filter,
                    entity.schema(),
                    storage_plan,
                    binding.backend,
                    resolver=self.workspace.module_resolver,
                )
            )
        )
        if query.filter is None:
            endpoint_filter = "none"
        else:
            endpoint_filter = "native+postfilter" if fused else "postfilter"
        step = {
            "node": "traverse",
            "field": traverse.field,
            "start": traverse.start,
            "min_hops": traverse.min_hops,
            "max_hops": traverse.max_hops,
            "direction": traverse.direction,
            "mode": "native_traverse" if native else "py_traverse",
            "endpoint_filter": endpoint_filter,
            "target_entity": target_entity,
        }
        if native:
            step["backend"] = binding.backend
        else:
            step["backends"] = sorted({item.backend for item in storage_plan.field_bindings(traverse.field)})
        return step
