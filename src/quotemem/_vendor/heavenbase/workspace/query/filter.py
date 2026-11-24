"""Private scalar-filter route, compile, and execution helpers."""

from __future__ import annotations

from heavenbase.utils import Any, Dict, List, Optional

from ...execution.contracts import HandlerContext
from ...execution.contracts import QueryFragment
from ...execution.frame.result import ResultFrame
from ...execution.frame.merge import difference, intersect, union
from ...filters.ops import OP_CONTAINED_IN
from ...filters.spec import And, Expr, Filter, Not, Or
from ...placement import StorageBinding, StoragePlan
from ...types import ShortText


def _gram_only(op: str, field_schema: Any) -> bool:
    """Return whether an operation exists only through a sparse-GRAM keyword index."""
    item_type = getattr(field_schema.dtype, "item_type", None)
    return op == OP_CONTAINED_IN and item_type is not None and type(item_type).identifier == ShortText.identifier


def option_refusal(entity_id: str, options: Any, field: str, op: str, field_schema: Any) -> Optional[str]:
    """Return why an entity's registration options refuse a filter operation on a field, or ``None``.

    The query compiler refuses such a filter, and ``describe_entity`` leaves the operation out of the field's
    ``filter_ops``, from this one rule.

    Args:
        entity_id (str): Registered entity identifier.
        options (Any): The entity's ``EntityOptions``, or ``None`` when it has no live registration.
        field (str): Filtered field name.
        op (str): Canonical filter operation identifier.
        field_schema (Any): The field's schema.

    Returns:
        Optional[str]: The refusal reason, or ``None`` when the options allow the operation.
    """
    if options is not None and not options.gram and _gram_only(op, field_schema):
        # The entity opted out of keyword maintenance, so the reverse
        # containment index does not exist; refuse instead of scanning.
        return (
            f"Entity {entity_id!r} is registered with gram=False, so {field}.{op} has no sparse-GRAM "
            "keyword index; re-register with gram=True and allow_replace=True to search it."
        )
    return None


def _field_binding(storage_plan: StoragePlan, field: str, backend: Optional[str] = None) -> StorageBinding:
    """Return the default binding, or the binding for a requested backend."""
    if backend is None:
        return storage_plan.first(field)
    for binding in storage_plan.field_bindings(field):
        if binding.backend == backend:
            return binding
    raise LookupError(f"No storage binding for {storage_plan.entity_id}.{field} on backend {backend!r}")


def _expr_backend(expr: Expr, storage_plan: StoragePlan) -> Optional[str]:
    """Return the current conservative backend choice for an expression."""
    if isinstance(expr, Filter):
        return storage_plan.first(expr.field).backend
    if isinstance(expr, (And, Or)):
        backends = [_expr_backend(clause, storage_plan) for clause in expr.clauses]
        return backends[0] if backends and backends[0] is not None and all(backend == backends[0] for backend in backends) else None
    if isinstance(expr, Not):
        return _expr_backend(expr.inner, storage_plan)
    return None


class _FilterRouteService:
    """Own scalar filter backend selection, compilation, and execution."""

    def __init__(self, router: Any):
        self.router = router
        self.workspace = router.workspace

    def eval_expr(self, expr: Optional[Expr], entity: type, *, unsafe: bool = False) -> ResultFrame:
        """Evaluate a filter expression against its best available execution path."""
        ws = self.workspace
        if expr is None:
            return ws._row_store.universe(entity.schema().entity_id)
        backend = self.expr_backend(expr, entity)
        if backend is not None:
            return self.execute_expr_fragment(expr, entity, backend, unsafe=unsafe)
        if isinstance(expr, Filter):
            return self.execute_filter(expr, entity, unsafe=unsafe)
        if isinstance(expr, And):
            return intersect(self.grouped_frames(self.flat_clauses(expr, And), entity, "and", unsafe=unsafe))
        if isinstance(expr, Or):
            return union(self.grouped_frames(self.flat_clauses(expr, Or), entity, "or", unsafe=unsafe))
        if isinstance(expr, Not):
            return difference(ws._row_store.universe(entity.schema().entity_id), self.eval_expr(expr.inner, entity, unsafe=unsafe))
        raise TypeError(f"Unsupported expression {type(expr).__name__}")

    def eval_frame(self, expr: Optional[Expr], entity: type, frame: ResultFrame) -> ResultFrame:
        """Evaluate an expression over one complete logical candidate frame.

        This is the correctness path for exact query-layer transforms. It uses
        the same logical-type normalization and portable predicates as backend
        scan handlers, while avoiding provider page limits and filter-only gaps.
        """
        if expr is None:
            return frame
        match = self.compile_frame_expr(expr, entity)
        rows = [row for row in frame.rows() if match(row)]
        return ResultFrame.from_rows(
            rows,
            cols=frame.col_names(),
            metadata_columns=frame.metadata_columns,
            identity_column=frame.identity_column,
        )

    def compile_frame_expr(self, expr: Expr, entity: type) -> Any:
        """Compile one expression tree into a portable logical-row predicate."""
        if isinstance(expr, Filter):
            fragment = self.compile_filter(expr, entity)
            reason = fragment.unsupported_reason()
            if reason:
                raise LookupError(reason)
            payload = fragment.notes.get("row_payload", fragment.payload)
            if not callable(payload):
                raise TypeError(f"Filter handler for {expr.field}.{expr.op} does not expose a portable row predicate")
            return payload
        if isinstance(expr, And):
            clauses = [self.compile_frame_expr(clause, entity) for clause in expr.clauses]
            return lambda row: all(clause(row) for clause in clauses)
        if isinstance(expr, Or):
            clauses = [self.compile_frame_expr(clause, entity) for clause in expr.clauses]
            return lambda row: any(clause(row) for clause in clauses)
        if isinstance(expr, Not):
            inner = self.compile_frame_expr(expr.inner, entity)
            return lambda row: not inner(row)
        raise TypeError(f"Unsupported expression {type(expr).__name__}")

    def execute_filter(self, expr: Filter, entity: type, *, unsafe: bool = False) -> ResultFrame:
        """Execute one filter expression on its routed backend."""
        schema = entity.schema()
        binding = self.field_binding(schema.entity_id, expr.field)
        return self.execute_expr_fragment(expr, entity, binding.backend, unsafe=unsafe)

    def field_binding(self, entity_id: str, field: str, backend: Optional[str] = None) -> StorageBinding:
        """Return the storage binding for a field and optional backend."""
        ws = self.workspace
        with ws._lock.read():
            storage_plan = ws.storage_plans[entity_id]
        return _field_binding(storage_plan, field, backend)

    def expr_backend(self, expr: Expr, entity: type) -> Optional[str]:
        """Return the single backend that can execute an expression, when one exists."""
        ws = self.workspace
        return _expr_backend(expr, ws.storage_plans[entity.schema().entity_id])

    def grouped_frames(self, clauses: List[Expr], entity: type, op: str, *, unsafe: bool = False) -> List[ResultFrame]:
        """Evaluate logical clauses while preserving backend grouping opportunities."""
        groups: Dict[str, List[Expr]] = {}
        slots = []
        for clause in clauses:
            backend = self.expr_backend(clause, entity)
            if backend is None:
                slots.append(("frame", self.eval_expr(clause, entity, unsafe=unsafe)))
                continue
            if backend not in groups:
                groups[backend] = []
                slots.append(("backend", backend))
            groups[backend].append(clause)
        frames = []
        for kind, value in slots:
            if kind == "frame":
                frames.append(value)
                continue
            group = groups[value]
            expr = group[0] if len(group) == 1 else And(group) if op == "and" else Or(group)
            frames.append(self.execute_expr_fragment(expr, entity, value, unsafe=unsafe))
        return frames

    def execute_expr_fragment(self, expr: Expr, entity: type, backend: str, *, unsafe: bool = False) -> ResultFrame:
        """Compile and execute one expression fragment."""
        schema = entity.schema()
        fragment = self.compile_expr(expr, entity, backend, unsafe=unsafe)
        reason = fragment.unsupported_reason()
        if reason:
            raise LookupError(reason)
        return self.router.execute_fragment(fragment, backend, schema.entity_id)

    def compile_expr(self, expr: Expr, entity: type, backend: str, *, unsafe: bool = False) -> QueryFragment:
        """Compile an expression tree into backend query fragments."""
        if isinstance(expr, Filter):
            return self.compile_filter(expr, entity, backend, unsafe=unsafe)
        if isinstance(expr, And):
            return QueryFragment(
                backend,
                "and",
                "*",
                [self.compile_expr(clause, entity, backend, unsafe=unsafe) for clause in expr.clauses],
                notes={"kind": "logic"},
            )
        if isinstance(expr, Or):
            return QueryFragment(
                backend,
                "or",
                "*",
                [self.compile_expr(clause, entity, backend, unsafe=unsafe) for clause in expr.clauses],
                notes={"kind": "logic"},
            )
        if isinstance(expr, Not):
            return QueryFragment(
                backend,
                "not",
                "*",
                [self.compile_expr(expr.inner, entity, backend, unsafe=unsafe)],
                notes={"kind": "logic"},
            )
        raise TypeError(f"Unsupported expression {type(expr).__name__}")

    def flat_clauses(self, expr: Expr, cls: Any) -> List[Expr]:
        """Flatten adjacent logical clauses of the same expression type."""
        out: List[Expr] = []
        for clause in expr.clauses:
            out.extend(self.flat_clauses(clause, cls) if isinstance(clause, cls) else [clause])
        return out

    def compile_filter(
        self,
        expr: Filter,
        entity: type,
        backend_name: Optional[str] = None,
        *,
        unsafe: bool = False,
    ) -> QueryFragment:
        """Compile one filter expression through the registered handler."""
        ws = self.workspace
        schema = entity.schema()
        field_schema = schema.get(expr.field)
        binding = self.field_binding(schema.entity_id, expr.field, backend_name)
        backend = ws.backends.get(binding.backend)
        refusal = option_refusal(schema.entity_id, ws.entity_options.get(schema.entity_id), expr.field, expr.op, field_schema)
        if refusal is not None:
            return QueryFragment(
                binding.backend,
                expr.op,
                expr.field,
                None,
                notes={"unsupported": True, "unsupported_reason": refusal, "strategy": binding.strategy},
            )
        try:
            handler = ws.handlers.get(field_schema.dtype, expr.op, backend.type, binding.strategy)
        except LookupError:
            reason = ws.handlers.unsupported_reason(backend.type, expr.op, type(field_schema.dtype).identifier, binding.strategy)
            return QueryFragment(
                binding.backend,
                expr.op,
                expr.field,
                None,
                notes={
                    "unsupported": True,
                    "unsupported_reason": reason or "No handler registered for this type, operation, backend, and strategy combination.",
                    "strategy": binding.strategy,
                },
            )
        fragment = handler(
            field_schema,
            expr.value,
            HandlerContext(
                binding.backend,
                backend.type,
                binding.strategy,
                dict(binding.strategy_options),
                backend_instance=backend,
                resolver=ws.module_resolver,
                config=ws.context.config,
                handlers=ws.handlers,
                entity_id=schema.entity_id,
                scalar_indexed=(expr.field == schema.pk_name or ws.storage_plans[entity.schema().entity_id].has_index(expr.field, backend=binding.backend)),
                unsafe=unsafe,
            ),
        )
        fragment.notes.setdefault("strategy", binding.strategy)
        if binding.strategy_options:
            fragment.notes.setdefault("strategy_options", dict(binding.strategy_options))
        return fragment
