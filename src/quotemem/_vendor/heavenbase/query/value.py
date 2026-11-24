"""Immutable, workspace-bound query values."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from heavenbase.utils import Any, Mapping, Optional, dataclass, field

from ..filters.ops import normalize_op
from ..filters.spec import And, Expr, Filter, Not, Or
from ..utils.values import coerce_nonnegint, field_name as _field_name
from .spec import Aggregate, AggregateMetric, Near, OrderKey, QuerySpec, Traverse

if TYPE_CHECKING:
    from ..execution.frame import ResultFrame
    from ..modules import ModuleResolver


class _AggregateMetricLookupError(LookupError):
    """Carry the raw reducer token when workspace binding cannot resolve it."""

    def __init__(self, op: str, supported: tuple[str, ...], cause: LookupError) -> None:
        super().__init__(str(cause))
        self.op = op
        self.supported = supported


def _check_field_owner(value: Any, target: str, operation: str) -> None:
    """Reject a typed field reference owned by another Entity."""
    owner = getattr(value, "entity_id", None)
    if owner is not None and owner != target:
        raise ValueError(f"{operation} field belongs to {owner}, but the query targets {target}")


def _target_field(value: Any, target: str, operation: str) -> str:
    """Return one target-relative field name after validating typed ownership."""
    _check_field_owner(value, target, operation)
    return _field_name(value)


def _bind_expr(
    expr: Expr,
    resolver: "ModuleResolver",
    *,
    target: Optional[str] = None,
    operation: str = "where",
) -> Expr:
    """Return a detached expression normalized by one workspace resolver."""
    if isinstance(expr, Filter):
        if target is not None:
            _check_field_owner(expr, target, operation)
        op = normalize_op(expr.op, resolver=resolver)
        value = expr.value
        if op == "get" and isinstance(value, Mapping) and "op" in value:
            value = {**value, "op": normalize_op(str(value["op"]), resolver=resolver)}
        return Filter(expr.field, op, value)
    if isinstance(expr, And):
        return And([_bind_expr(clause, resolver, target=target, operation=operation) for clause in expr.clauses])
    if isinstance(expr, Or):
        return Or([_bind_expr(clause, resolver, target=target, operation=operation) for clause in expr.clauses])
    if isinstance(expr, Not):
        return Not(_bind_expr(expr.inner, resolver, target=target, operation=operation))
    raise TypeError(f"Unsupported filter expression node {type(expr).__name__}")


def _bind_metric(metric: AggregateMetric, resolver: "ModuleResolver") -> AggregateMetric:
    """Return one reducer metric normalized by a workspace resolver."""
    if not isinstance(metric, AggregateMetric):
        raise TypeError("Aggregate metrics must be built from field reducers")
    from .aggregate import aggregate_reducers, reducer_for
    from .spec import _thaw_json

    try:
        reducer = reducer_for(metric.op, resolver=resolver)
    except LookupError as exc:
        supported = tuple(item.identifier for item in aggregate_reducers(resolver=resolver))
        raise _AggregateMetricLookupError(metric.op, supported, exc) from exc
    args = tuple(_thaw_json(value) for value in metric.args)
    options = reducer.bind(*args, **_thaw_json(metric.options))
    return AggregateMetric(metric.field, reducer.identifier, metric.entity_id, options)


@dataclass(frozen=True, slots=True, init=False)
class Query:
    """Represent one immutable query specification with workspace authority.

    `Query` contains request data only. Each fluent operation returns another
    value and leaves its source unchanged. Execution and explanation may
    produce temporary plans or provider fragments inside the workspace, but
    those artifacts and any returned `ResultFrame` are never stored on the
    query.

    Args:
        workspace: Workspace facade or compatible query runtime.
        entity: Registered entity class targeted by the query.
        spec: Optional existing immutable request. Omit it to start an empty
            query for `entity`.
    """

    _workspace: Any = field(repr=False, compare=False)
    _entity: type = field(repr=False, compare=False)
    _spec: QuerySpec

    def __init__(self, workspace: Any, entity: type, spec: Optional[QuerySpec] = None) -> None:
        """Initialize one workspace-bound query value."""
        schema = entity.schema()
        request = QuerySpec(schema.entity_id) if spec is None else spec
        if not isinstance(request, QuerySpec):
            raise TypeError("Query spec must be a QuerySpec")
        if request.entity_id != schema.entity_id:
            raise ValueError(f"Query spec targets {request.entity_id!r}, but the entity is {schema.entity_id!r}")
        object.__setattr__(self, "_workspace", workspace)
        object.__setattr__(self, "_entity", entity)
        object.__setattr__(self, "_spec", request)

    @property
    def entity(self) -> type:
        """Return the registered entity class targeted by this query."""
        return self._entity

    @property
    def spec(self) -> QuerySpec:
        """Return the immutable portable request carried by this query."""
        return self._spec

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible diagnostic representation.

        Args:
            None.

        Returns:
            dict[str, Any]: Serialized query spec for diagnostics and explain
                output.
        """
        return self.spec.to_dict()

    def where(self, expr: Optional[Expr | Mapping[str, Any]]) -> "Query":
        """Return a query with an additional filter expression.

        Args:
            expr: Python expression, JSON filter mapping, or `None`. `None`
                leaves the query unchanged.

        Returns:
            Query: Derived immutable query.
        """
        self._require_open("where")
        if isinstance(expr, Mapping):
            from .json import query_expr

            expr = query_expr(expr, self._entity.schema(), resolver=self._workspace.module_resolver)
        if expr is None:
            return self
        if not isinstance(expr, Expr):
            raise TypeError(f"Query.where expects Expr or mapping, got {type(expr).__name__}")
        clause = _bind_expr(expr, self._workspace.module_resolver, target=self.spec.entity_id)
        current = self.spec.filter
        if current is None:
            filt = clause
        elif isinstance(current, And):
            filt = And([*current.clauses, clause])
        else:
            filt = And([current, clause])
        return self._derive(filter=filt)

    def available(
        self,
        *,
        stale: bool = False,
        invisible: bool = False,
        now: Any = None,
    ) -> "Query":
        """Return a Catalog query restricted to discoverable rows.

        Args:
            stale (bool): Whether stale Catalog rows remain discoverable.
                Defaults to ``False``.
            invisible (bool): Whether hidden Catalog rows remain discoverable.
                Defaults to ``False``.
            now (Any): Evaluation instant as epoch milliseconds, ``int``,
                ``float``, ``datetime``, or ISO-8601 string. ``None`` snapshots
                the current clock.

        Returns:
            Query: Derived immutable query with an availability filter.

        Raises:
            ValueError: If this query does not target ``sys-catalog``.
        """
        self._require_open("available")
        from .catalog import CatalogAvailable

        available = CatalogAvailable(now=now, stale=stale, invisible=invisible).check(self._entity.schema().entity_id)
        return self.where(available.to_expr())

    def near(
        self, field_ref: Any, value: Any, top_k: Optional[int] = None, metric: Optional[str] = None, *, exact: bool = False, cache_version: str | None = None
    ) -> "Query":
        """Return a query with one vector-similarity clause.

        Args:
            field_ref (Any): Vector field name or field reference on the
                queried entity.
            value (Any): Query vector or provider-native vector payload.
            top_k (int | None): Maximum nearest-neighbor rows to request.
                ``None`` defers the default to the executing workspace.
            metric (str | None): Similarity metric, such as ``cosine``.
                ``None`` defers the default to the executing workspace.
            exact (bool): Require exhaustive metric Top-K rather than ANN.
                Native handlers must establish exact execution; otherwise HB
                scans all eligible vectors. Inspect `explain()` before large
                queries. This does not promise cross-provider score bits or
                identical membership at an equal-score Top-K boundary.
            cache_version (str | None): Explicit stable source-version token for
                bounded candidate reuse by portable exact scans. Requires
                exact=True. Omit for live reads. HB writes and refresh invalidate
                cached candidates; external writes require changing this token
                or calling workspace.refresh(). It is not a historical read or
                database snapshot. Proven native routes do not use this cache.
                Experimental: may change in a later release.

        Returns:
            Query: Derived immutable query.

        Raises:
            ValueError: If the query already has a near clause.
        """
        self._require_open("near")
        if self.spec.near is not None:
            raise ValueError("Query already has a near clause")
        return self._derive(near=Near(_target_field(field_ref, self.spec.entity_id, "near"), value, top_k, metric, exact=exact, cache_version=cache_version))

    def traverse(
        self,
        field_ref: Any,
        start_id: Any,
        *,
        min_hops: int = 1,
        max_hops: int = 1,
        direction: str = "out",
    ) -> "Query":
        """Return a query restricted to rows reachable over a GraphEdge field.

        Args:
            field_ref (Any): GraphEdge field name or field reference on the
                queried entity.
            start_id (Any): Starting ``object_id`` for traversal.
            min_hops (int): Smallest hop depth to include. Defaults to ``1``.
            max_hops (int): Largest hop depth to include. Defaults to ``1``.
            direction (str): Traversal direction. Supported values:
                ``out`` (follow outgoing edges), ``in`` (follow incoming
                edges), and ``any`` (follow both).

        Returns:
            Query: Derived immutable query.

        Raises:
            ValueError: If the query already has a traverse clause.
        """
        self._require_open("traverse")
        if self.spec.traverse is not None:
            raise ValueError("Query already has a traverse clause")
        return self._derive(traverse=Traverse(_target_field(field_ref, self.spec.entity_id, "traverse"), start_id, min_hops, max_hops, direction))

    def select(self, *fields: Any) -> "Query":
        """Return a query projecting the requested result fields.

        Args:
            *fields (Any): Field names or field references to include in the
                result projection.

        Returns:
            Query: Derived immutable query.
        """
        self._require_open("select")
        target = self.spec.entity_id
        return self._derive(select=tuple(_target_field(item, target, "select") for item in fields))

    def group_by(self, *fields: Any) -> "Query":
        """Return a query grouped by the requested fields.

        Grouping is incomplete until ``aggregate()`` supplies named metrics.
        The pending fields remain ordinary query-spec data and no plan is
        created.

        Args:
            *fields (Any): Field names or field references to group by.

        Returns:
            Query: Derived immutable query.

        Raises:
            ValueError: If duplicate group fields are supplied after
                normalization.
        """
        self._require_open("group_by")
        target = self.spec.entity_id
        groups = tuple(_target_field(item, target, "group_by") for item in fields)
        if len(set(groups)) != len(groups):
            raise ValueError(f"Duplicate aggregate group fields: {list(groups)}")
        if self.spec.aggregate is not None:
            return self._derive(aggregate=Aggregate(self.spec.aggregate.metrics, groups))
        return self._derive(group_by=groups)

    def aggregate(self, **metrics: AggregateMetric) -> "Query":
        """Return a query with an identityless aggregate projection.

        Args:
            **metrics (AggregateMetric): Output column names mapped to field
                reducer targets such as ``price.sum()``.

        Returns:
            Query: Derived immutable query with grouping cleared.

        Raises:
            ValueError: If the query already has an aggregate clause or no
                metrics are supplied.
        """
        self._require_open("aggregate")
        if self.spec.aggregate is not None:
            raise ValueError("Query already has an aggregate clause")
        if not metrics:
            raise ValueError("aggregate() needs at least one named metric")
        bound = {name: _bind_metric(metric, self._workspace.module_resolver) for name, metric in metrics.items()}
        return self._derive(aggregate=Aggregate(bound, self.spec.group_by), group_by=())

    def having(self, expr: Optional[Expr | Mapping[str, Any]]) -> "Query":
        """Return an aggregate query with an additional post-reduction filter.

        Args:
            expr (Expr | Mapping[str, Any] | None): Aggregate-result filter
                expression, JSON mapping, or ``None``. ``None`` leaves the
                query unchanged.

        Returns:
            Query: Derived immutable query.

        Raises:
            ValueError: If the query has no aggregate projection.
            TypeError: If ``expr`` is not an expression or mapping.
        """
        self._require_open("having")
        if self.spec.aggregate is None:
            raise ValueError("having() needs aggregate(); HAVING filters aggregate result rows")
        if isinstance(expr, Mapping):
            from ..filters.spec import expr_from_dict

            expr = expr_from_dict(expr, resolver=self._workspace.module_resolver)
        if expr is None:
            return self
        if not isinstance(expr, Expr):
            raise TypeError(f"Query.having expects Expr or mapping, got {type(expr).__name__}")
        clause = _bind_expr(expr, self._workspace.module_resolver)
        current = self.spec.having
        if current is None:
            having = clause
        elif isinstance(current, And):
            having = And([*current.clauses, clause])
        else:
            having = And([current, clause])
        return self._derive(having=having)

    def order_by(self, field_ref: Any, desc: bool = False) -> "Query":
        """Return a query with an additional result ordering key.

        Args:
            field_ref (Any): Field name or field reference to sort by.
            desc (bool): Whether to sort descending. Defaults to ``False``.

        Returns:
            Query: Derived immutable query.
        """
        self._require_open("order_by")
        return self._derive(order=(*self.spec.order, OrderKey(_target_field(field_ref, self.spec.entity_id, "order_by"), desc)))

    def offset(self, n: int) -> "Query":
        """Return a query that skips ``n`` ordered rows.

        Args:
            n (int): Non-negative number of rows to skip after ordering.

        Returns:
            Query: Derived immutable query.
        """
        self._require_open("offset")
        return self._derive(offset=coerce_nonnegint(n, "offset"))

    def limit(self, n: int) -> "Query":
        """Return a query limited to at most ``n`` rows.

        Args:
            n (int): Non-negative maximum number of rows to return.

        Returns:
            Query: Derived immutable query.
        """
        self._require_open("limit")
        return self._derive(limit=coerce_nonnegint(n, "limit"))

    def unsafe(self, enabled: bool = True) -> "Query":
        """Return a query allowing documented result-relaxing optimizations.

        Args:
            enabled (bool): Whether unsafe optimizations are allowed.
                Defaults to ``True``.

        Returns:
            Query: Derived immutable query.

        Raises:
            TypeError: If ``enabled`` is not a boolean.
        """
        self._require_open("unsafe")
        if not isinstance(enabled, bool):
            raise TypeError("unsafe enabled must be a boolean")
        return self._derive(unsafe=enabled)

    def count(self, field_ref: Any = "object_id", *, distinct: bool = True) -> "Query":
        """Return a source-only terminal count query.

        Only `where`, `near`, `traverse`, and `unsafe` may precede this
        terminal projection. The returned query performs no I/O until
        `execute()` and cannot accept further query transformations.

        Args:
            field_ref: Field name or field reference to count. Defaults to
                `object_id`.
            distinct: Whether equal non-null values count once. Defaults to
                `True`.

        Returns:
            Query: Terminal query whose execution returns a one-cell,
            identityless `ResultFrame`. Use `.execute().scalar()` to extract
            the integer.
        """
        self._require_open("count")
        conflicts = self._count_conflicts()
        if conflicts:
            raise ValueError(f"count() is source-only and cannot follow {', '.join(conflicts)}")
        target = self.spec.entity_id
        field = _target_field(field_ref, target, "count")
        metric = _bind_metric(
            AggregateMetric(field, "count", target, {"distinct": distinct}),
            self._workspace.module_resolver,
        )
        return self._derive(aggregate=Aggregate({"count": metric}), terminal="count")

    def execute(self) -> "ResultFrame":
        """Execute this query and return a detached result container.

        Every call delegates again and recomputes the result. Explicit near
        cache versions may reuse candidate data, never previous query results. The returned
        frame is not retained by, or connected back to, this query. User code
        should explicitly export values through methods such as ``rows()``,
        ``to_pandas()``, ``to_pyarrow()``, or ``scalar()``.

        Args:
            None.

        Returns:
            ResultFrame: Detached query result container.
        """
        return self._workspace.execute(self.spec)

    def explain(self) -> Any:
        """Return transient routing and handler diagnostics for this query.

        Args:
            None.

        Returns:
            Any: Workspace-specific explain payload for this query spec.
        """
        return self._workspace.explain(self.spec)

    def _derive(self, **changes: Any) -> "Query":
        """Return one query value with replaced request fields."""
        return Query(self._workspace, self._entity, replace(self.spec, **changes))

    def _require_open(self, operation: str) -> None:
        """Reject transformations after a terminal projection."""
        if self.spec.terminal is not None:
            raise ValueError(f"{operation}() cannot follow terminal count(); only execute() and explain() are valid")

    def _count_conflicts(self) -> list[str]:
        """Return clauses that cannot feed the source-only count terminal."""
        spec = self.spec
        conflicts = []
        if spec.select:
            conflicts.append("select")
        if spec.group_by:
            conflicts.append("group_by")
        if spec.aggregate is not None:
            conflicts.append("aggregate")
        if spec.having is not None:
            conflicts.append("having")
        if spec.order:
            conflicts.append("order_by")
        if spec.offset:
            conflicts.append("offset")
        if spec.limit is not None:
            conflicts.append("limit")
        return conflicts
