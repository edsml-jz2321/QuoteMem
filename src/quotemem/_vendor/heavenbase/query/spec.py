"""Typed query request objects shared by query surfaces and workspace routing."""

from __future__ import annotations

__all__ = [
    "Aggregate",
    "AggregateMetric",
    "Near",
    "OrderKey",
    "QuerySpec",
    "Traverse",
]

from types import MappingProxyType
from dataclasses import InitVar, KW_ONLY, replace

from heavenbase.utils import Any, Dict, Mapping, Optional, Sequence, dataclass, deepcopy, field, raise_mismatch

from ..filters.spec import Expr
from ..utils.ids import check_py_name
from ..utils.values import coerce_dict, coerce_nonnegint, coerce_posint, coerce_tuple
from ..utils.serialize import validate_json

_FIELD_REF_RESERVED_NAMES = frozenset(
    {
        "aggregate",
        "all",
        "any",
        "array_contains",
        "contained_in",
        "contains",
        "ends_with",
        "entity_id",
        "exists",
        "ilike",
        "in_",
        "like",
        "match",
        "name",
        "op",
        "regex",
        "starts_with",
        "wildcard",
    }
)


def _check_aggregate_method_name(name: str) -> str:
    """Validate a reducer identifier against the query field DSL namespace."""
    identifier = check_py_name(name, "aggregate reducer identifier")
    if identifier in _FIELD_REF_RESERVED_NAMES:
        raise ValueError(f"Aggregate reducer identifier {identifier!r} collides with FieldRef.{identifier}")
    return identifier


def _freeze_json(value: Any) -> Any:
    """Recursively isolate and freeze one JSON-compatible request value."""
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    return deepcopy(value)


def _thaw_json(value: Any) -> Any:
    """Return a detached mutable JSON representation of a frozen request value."""
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return deepcopy(value)


@dataclass(frozen=True)
class AggregateMetric:
    """One aggregate reducer intent built by a field reference.

    Args:
        field: Logical source field.
        op: Reducer operation identifier or alias to bind in a workspace.
        entity_id: Optional entity identity carried by Python field references.
        options: Named reducer arguments retained until workspace binding.
        args: Positional reducer arguments retained until workspace binding.
    """

    field: str
    op: str
    entity_id: Optional[str] = None
    options: Mapping[str, Any] = field(default_factory=dict)
    args: Sequence[Any] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        """Validate names and freeze detached reducer arguments."""
        object.__setattr__(self, "field", _name(self.field, "Aggregate field"))
        object.__setattr__(self, "op", _name(self.op, "Aggregate operation"))
        if self.entity_id is not None:
            object.__setattr__(self, "entity_id", _name(self.entity_id, "Aggregate entity_id"))
        args = []
        for value in coerce_tuple(self.args, "Aggregate args"):
            value = _thaw_json(value)
            try:
                validate_json(value)
            except (TypeError, ValueError) as exc:
                raise TypeError("Aggregate reducer parameters must be JSON-compatible") from exc
            args.append(_freeze_json(value))
        raw_options = coerce_dict(self.options, "Aggregate options")
        options: Dict[str, Any] = {}
        for raw_name, value in raw_options.items():
            name = _name(raw_name, "Aggregate option name")
            if name == "field":
                raise ValueError("Aggregate option name 'field' is reserved for the metric source field")
            value = _thaw_json(value)
            try:
                validate_json(value)
            except (TypeError, ValueError) as exc:
                raise TypeError("Aggregate reducer parameters must be JSON-compatible") from exc
            options[name] = _freeze_json(value)
        object.__setattr__(self, "args", tuple(args))
        object.__setattr__(self, "options", MappingProxyType(options))

    def to_dict(self) -> Dict[str, Any]:
        """Return the canonical JSON metric shape.

        Args:
            None.

        Returns:
            Dict[str, Any]: A one-key operation mapping whose value always
            contains the source ``field`` and any reducer parameters.

        Raises:
            ValueError: If positional reducer arguments have not yet been bound
                to their registered parameter names by a `Query`.
        """
        if self.args:
            raise ValueError("Aggregate positional arguments must be bound by Query.aggregate before serialization")
        options = _thaw_json(self.options)
        return {self.op: {"field": self.field, **options}}


@dataclass(frozen=True)
class Aggregate:
    """Frozen aggregate projection with optional grouping.

    Args:
        metrics: Output column names mapped to reducer targets.
        group_by: Ordered logical group fields.
    """

    metrics: Mapping[str, AggregateMetric]
    group_by: Sequence[str] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        """Validate names and freeze aggregate collections."""
        if not isinstance(self.metrics, Mapping):
            raise TypeError("Aggregate metrics must be a mapping")
        metrics: Dict[str, AggregateMetric] = {}
        for raw_name, metric in self.metrics.items():
            name = _name(raw_name, "Aggregate output name")
            if name == "object_id":
                raise ValueError("Aggregate output name 'object_id' is reserved because aggregate rows are not entities")
            if name in metrics:
                raise ValueError(f"Aggregate output name collision after normalization: {name!r}")
            if not isinstance(metric, AggregateMetric):
                raise TypeError(f"Aggregate output {name!r} must be built from a field reducer such as Product.price.sum()")
            metrics[name] = metric
        if not metrics:
            raise ValueError("Aggregate needs at least one named metric")
        groups = tuple(_name(name, "Aggregate group field") for name in coerce_tuple(self.group_by, "Aggregate group_by"))
        if len(set(groups)) != len(groups):
            raise ValueError(f"Duplicate aggregate group fields: {list(groups)}")
        overlap = sorted(set(groups) & set(metrics))
        if overlap:
            raise ValueError(f"Aggregate output names collide with group fields: {overlap}")
        object.__setattr__(self, "metrics", MappingProxyType(metrics))
        object.__setattr__(self, "group_by", groups)

    def to_dict(self) -> Dict[str, Any]:
        """Return the normalized diagnostic aggregate representation.

        This request-object shape separates `group_by` and `metrics`; it is not
        the external `query_json` input shape, where both are top-level clauses.

        Args:
            None.

        Returns:
            Dict[str, Any]: Frozen request state rendered as group fields and
            named metric mappings for diagnostics and explain output.
        """
        return {
            "group_by": list(self.group_by),
            "metrics": {name: metric.to_dict() for name, metric in self.metrics.items()},
        }


@dataclass(frozen=True)
class Near:
    """Vector similarity clause attached to a query.

    Args:
        field (str): Vector field name on the queried entity.
        value (Any): Query vector or provider-native vector payload.
        top_k (int | None): Maximum nearest-neighbor rows to request. ``None``
            defers the default to the workspace that executes the query.
        metric (str | None): Similarity metric, such as `cosine`. ``None``
            defers the default to the executing workspace.
        notes (Mapping[str, Any]): Route and normalization metadata carried by
            workspace services.
        exact (bool): Require exhaustive metric Top-K, never an approximate
            candidate window. Unsupported native routes use an explicit scan
            fallback. Scores use the selected executor's numeric precision;
            equal-score boundary membership may differ between providers.
        cache_version (str | None): Explicit caller-owned immutable source version
            for bounded candidate reuse on portable exact scans. Requires exact.
            Omit for live reads. This does not select historical database data;
            external writes require a new version or workspace refresh. Proven
            native routes bypass this cache and continue reading the provider.
            Experimental: may change in a later release.

    Raises:
        TypeError: If `field` or `metric` is not a string, `top_k` is not
            integer-like, or `notes` is not a mapping.
        ValueError: If `field`, `metric`, or `top_k` is invalid.
    """

    field: str
    value: Any
    top_k: int | None = None
    metric: str | None = None
    notes: Mapping[str, Any] = field(default_factory=dict)
    _: KW_ONLY
    exact: bool = False
    cache_version: str | None = None
    max_top_k: InitVar[int | None] = None

    def __post_init__(self, max_top_k: int | None) -> None:
        name = _name(self.field, "Near field")
        top_k = None if self.top_k is None else coerce_posint(self.top_k, "top_k")
        maximum = None if max_top_k is None else coerce_posint(max_top_k, "max_top_k")
        if top_k is not None and maximum is not None and top_k > maximum:
            raise ValueError(f"top_k exceeds configured maximum {maximum}")
        metric = None if self.metric is None else _name(self.metric, "Near metric")
        if not isinstance(self.notes, Mapping):
            raise TypeError("Near notes must be a mapping")
        if not isinstance(self.exact, bool):
            raise TypeError("Near exact must be a boolean")
        if self.cache_version is not None:
            object.__setattr__(self, "cache_version", _name(self.cache_version, "Near cache_version"))
            if not self.exact:
                raise ValueError("cache_version requires exact=True")
        object.__setattr__(self, "field", name)
        object.__setattr__(self, "top_k", top_k)
        object.__setattr__(self, "metric", metric)
        object.__setattr__(self, "notes", MappingProxyType(dict(self.notes)))

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-compatible near-clause representation.

        Args:
            None.

        Returns:
            Dict[str, Any]: Near clause with `field`, `value`, `top_k`, `metric`, and
            optional `notes`.
        """
        out = {"field": self.field, "value": self.value, "top_k": self.top_k, "metric": self.metric}
        if self.exact:
            out["exact"] = True
        if self.cache_version is not None:
            out["cache_version"] = self.cache_version
        if self.notes:
            out["notes"] = dict(self.notes)
        return out


@dataclass(frozen=True)
class Traverse:
    """Routed graph-traversal clause attached to a query.

    Args:
        field (str): GraphEdge field name on the queried entity.
        start (Any): Starting `object_id` for traversal.
        min_hops (int): Smallest hop depth to include.
        max_hops (int): Largest hop depth to include.
        direction (str): Traversal direction. Supported values:
            - `out`: Follow outgoing edges.
            - `in`: Follow incoming edges.
            - `any`: Follow both incoming and outgoing edges.

    Raises:
        TypeError: If `field` or `direction` is not a string, or hop bounds are
            not integer-like.
        ValueError: If the field, start id, hop bounds, or direction is invalid.
    """

    field: str
    start: Any
    min_hops: int = 1
    max_hops: int = 1
    direction: str = "out"
    _: KW_ONLY
    max_allowed: InitVar[int | None] = None

    def __post_init__(self, max_allowed: int | None) -> None:
        name = _name(self.field, "Traverse field")
        if self.start is None or (isinstance(self.start, str) and not self.start.strip()):
            raise ValueError("Traverse start must not be empty")
        min_hops = coerce_posint(self.min_hops, "min_hops")
        max_hops = coerce_posint(self.max_hops, "max_hops")
        if min_hops > max_hops:
            raise ValueError(f"min_hops {min_hops} cannot exceed max_hops {max_hops}")
        maximum = None if max_allowed is None else coerce_posint(max_allowed, "max_allowed")
        if maximum is not None and max_hops > maximum:
            raise ValueError(f"max_hops {max_hops} exceeds configured maximum {maximum}")
        direction = raise_mismatch(("out", "in", "any"), _name(self.direction, "Traverse direction"), name="traverse direction")
        object.__setattr__(self, "field", name)
        object.__setattr__(self, "min_hops", min_hops)
        object.__setattr__(self, "max_hops", max_hops)
        object.__setattr__(self, "direction", direction)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-compatible traverse-clause representation.

        Args:
            None.

        Returns:
            Dict[str, Any]: Traverse clause with field, start, hop bounds, and direction.
        """
        return {"field": self.field, "start": self.start, "min_hops": self.min_hops, "max_hops": self.max_hops, "direction": self.direction}


@dataclass(frozen=True)
class OrderKey:
    """Ordering clause for one result column.

    Args:
        field (str): Result column or entity field name used for sorting.
        desc (bool): Whether ordering is descending.

    Raises:
        TypeError: If `desc` is not a boolean.
        ValueError: If `field` is empty.
    """

    field: str
    desc: bool = False

    def __post_init__(self) -> None:
        name = _name(self.field, "Order field")
        if not isinstance(self.desc, bool):
            raise TypeError("Order desc must be a boolean")
        object.__setattr__(self, "field", name)


@dataclass(frozen=True)
class QuerySpec:
    """Normalized query request consumed by workspace routing.

    `QuerySpec` is request data, not a materialized execution plan. Fluent and
    JSON query surfaces build it; workspace routing services interpret it and
    lower the request into backend `QueryFragment` objects.

    Args:
        entity_id (str): Queried entity identifier.
        filter (Optional[Expr]): Optional typed filter expression tree.
        near (Optional[Near]): Optional vector similarity clause.
        select (Sequence[str]): Result columns to keep. Empty means all columns.
        order (Sequence[OrderKey]): Ordered result sorting keys.
        offset (int): Row offset after ordering.
        limit (Optional[int]): Maximum row count after offset.
        traverse (Optional[Traverse]): Optional self-referential graph traversal
            clause folded into query routing before execution.
        aggregate (Optional[Aggregate]): Optional identityless aggregate projection.
        having (Optional[Expr]): Optional post-aggregate predicate over group fields
            and named aggregate outputs.
        unsafe (bool): Whether documented result-relaxing query optimizations
            may run. Defaults to `False`.
        group_by (Sequence[str]): Pending grouping fields retained by an
            immutable `Query` until `aggregate()` supplies the result
            projection. A spec with pending groups is intentionally not
            executable.
        terminal (Optional[str]): Optional terminal projection marker. The
            only supported value is `"count"`.

    Raises:
        TypeError: If a clause or collection item has the wrong request type.
        ValueError: If the entity id, pagination, or projection is invalid.
    """

    entity_id: str
    filter: Optional[Expr] = None
    near: Optional[Near] = None
    select: Sequence[str] = field(default_factory=tuple)
    order: Sequence[OrderKey] = field(default_factory=tuple)
    offset: int = 0
    limit: Optional[int] = None
    traverse: Optional[Traverse] = None
    aggregate: Optional[Aggregate] = None
    having: Optional[Expr] = None
    unsafe: bool = False
    group_by: Sequence[str] = field(default_factory=tuple)
    terminal: Optional[str] = None

    def __post_init__(self) -> None:
        entity_id = _name(self.entity_id, "QuerySpec entity_id")
        if self.filter is not None and not isinstance(self.filter, Expr):
            raise TypeError("QuerySpec filter must be an Expr or None")
        if self.near is not None and not isinstance(self.near, Near):
            raise TypeError("QuerySpec near must be a Near or None")
        if self.traverse is not None and not isinstance(self.traverse, Traverse):
            raise TypeError("QuerySpec traverse must be a Traverse or None")
        if self.aggregate is not None and not isinstance(self.aggregate, Aggregate):
            raise TypeError("QuerySpec aggregate must be an Aggregate or None")
        if self.having is not None and not isinstance(self.having, Expr):
            raise TypeError("QuerySpec having must be an Expr or None")
        if self.having is not None and self.aggregate is None:
            raise ValueError("having needs aggregate(); HAVING filters aggregate result rows")
        if not isinstance(self.unsafe, bool):
            raise TypeError("QuerySpec unsafe must be a boolean")
        group_by = tuple(_name(name, "QuerySpec group_by field") for name in coerce_tuple(self.group_by, "QuerySpec group_by"))
        if len(set(group_by)) != len(group_by):
            raise ValueError(f"Duplicate aggregate group fields: {list(group_by)}")
        if group_by and self.aggregate is not None:
            raise ValueError("QuerySpec pending group_by fields must be consumed by aggregate()")
        terminal = None if self.terminal is None else _name(self.terminal, "QuerySpec terminal")
        if terminal not in (None, "count"):
            raise ValueError(f"Unsupported QuerySpec terminal {terminal!r}")
        select = tuple(_name(name, "QuerySpec select field") for name in coerce_tuple(self.select, "QuerySpec select"))
        if self.aggregate is not None and select:
            raise ValueError("Aggregate defines the result projection; do not combine aggregate() with select()")
        order = coerce_tuple(self.order, "QuerySpec order")
        if any(not isinstance(item, OrderKey) for item in order):
            raise TypeError("QuerySpec order entries must be OrderKey instances")
        object.__setattr__(self, "entity_id", entity_id)
        object.__setattr__(self, "select", select)
        object.__setattr__(self, "order", order)
        object.__setattr__(self, "offset", coerce_nonnegint(self.offset, "offset"))
        object.__setattr__(self, "limit", None if self.limit is None else coerce_nonnegint(self.limit, "limit"))
        object.__setattr__(self, "group_by", group_by)
        object.__setattr__(self, "terminal", terminal)
        if terminal == "count":
            self._validate_count_terminal()

    def require_executable(self) -> None:
        """Reject structurally valid intermediate specs that cannot execute.

        Immutable query composition may retain `group_by()` fields before an
        aggregate projection is supplied. Execution and explanation are the
        boundaries where that incomplete request must fail.

        Args:
            None.

        Returns:
            None: The spec is executable.

        Raises:
            ValueError: If grouping has no aggregate projection.
        """
        if self.group_by:
            raise ValueError("group_by() needs aggregate(); grouping alone does not define a result projection")

    def _validate_count_terminal(self) -> None:
        """Validate the source-only terminal-count shape."""
        conflicts = []
        if self.select:
            conflicts.append("select")
        if self.order:
            conflicts.append("order_by")
        if self.offset:
            conflicts.append("offset")
        if self.limit is not None:
            conflicts.append("limit")
        if self.group_by:
            conflicts.append("group_by")
        if self.having is not None:
            conflicts.append("having")
        if conflicts:
            raise ValueError(f"count() is source-only and cannot follow {', '.join(conflicts)}")
        aggregate = self.aggregate
        if aggregate is None:
            raise ValueError("QuerySpec terminal 'count' requires a count aggregate projection")
        metrics = aggregate.metrics
        if aggregate.group_by or tuple(metrics) != ("count",) or metrics["count"].op != "count":
            raise ValueError("QuerySpec terminal 'count' requires one ungrouped count metric named 'count'")

    def to_dict(self) -> Dict[str, Any]:
        """Return the normalized diagnostic query representation.

        The result exposes internal request structure and is not accepted as a
        `query_json` input object. External JSON lowering remains owned by
        `query/json.py`.

        Args:
            None.

        Returns:
            Dict[str, Any]: Diagnostic request state with normalized filter, near,
            selection, aggregation, HAVING, ordering, pagination, and traversal
            fields.
        """
        return {
            "entity_id": self.entity_id,
            "filter": self.filter.to_dict() if self.filter else None,
            "near": self.near.to_dict() if self.near else None,
            "select": list(self.select),
            "order": [(order.field, order.desc) for order in self.order],
            "offset": self.offset,
            "limit": self.limit,
            "traverse": self.traverse.to_dict() if self.traverse else None,
            "aggregate": self.aggregate.to_dict() if self.aggregate else None,
            "having": self.having.to_dict() if self.having else None,
            "unsafe": self.unsafe,
            "group_by": list(self.group_by),
            "terminal": self.terminal,
        }


def _bind_query_policy(query: QuerySpec, config: Any) -> QuerySpec:
    """Return a request validated against its executing workspace policy."""
    if not isinstance(query, QuerySpec):
        raise TypeError("Workspace query execution expects a QuerySpec")
    query.require_executable()
    if not callable(getattr(config, "get", None)):
        raise TypeError("Workspace query policy requires a configuration authority")
    near = query.near
    if near is not None:
        near = Near(
            near.field,
            near.value,
            int(config.get("heavenbase.query.near.default_top_k", default=50)) if near.top_k is None else near.top_k,
            str(config.get("heavenbase.query.near.default_metric", default="cosine")) if near.metric is None else near.metric,
            near.notes,
            exact=near.exact,
            cache_version=near.cache_version,
            max_top_k=int(config.get("heavenbase.query.near.max_top_k", default=1000)),
        )
    traverse = query.traverse
    if traverse is not None:
        traverse = Traverse(
            traverse.field,
            traverse.start,
            traverse.min_hops,
            traverse.max_hops,
            traverse.direction,
            max_allowed=int(config.get("heavenbase.query.graph.max_hops", default=64)),
        )
    return replace(query, near=near, traverse=traverse)


def _name(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string")
    out = value.strip()
    if not out:
        raise ValueError(f"{label} must not be empty")
    return out
