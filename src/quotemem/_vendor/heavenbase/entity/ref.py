"""Entity field references used by the query DSL."""

from __future__ import annotations

__all__ = [
    "FieldRef",
]

from typing import TYPE_CHECKING

from heavenbase.utils import Any, Callable, List

from ..filters.spec import Filter

if TYPE_CHECKING:
    from ..query.spec import AggregateMetric


class FieldRef:
    """Class attribute proxy used by the query DSL.

    Args:
        entity_id (str): Owning entity identifier.
        name (str): Logical field name on that entity.
    """

    def __init__(self, entity_id: str, name: str) -> None:
        """Initialize one entity field reference."""
        self.entity_id = entity_id
        self.name = name

    def op(self, op: str, value: Any = None) -> Filter:
        """Build a filter retaining an explicit operation token.

        The owning query value resolves the token through its workspace's
        module authority. This lets the same authored expression bind
        differently in isolated Contexts without consulting process globals.

        Args:
            op (str): Filter operation token such as ``eq`` or ``contains``.
            value (Any): Operation operand. Defaults to ``None``.

        Returns:
            Filter: Typed filter clause bound to this field.
        """
        return Filter(self.name, op, value, entity_id=self.entity_id)

    def match(self, value: str) -> Filter:
        """Build a text match filter.

        Args:
            value (str): Text match operand.

        Returns:
            Filter: ``match`` filter on this field.
        """
        return self.op("match", value)

    def like(self, value: str) -> Filter:
        """Build a SQL LIKE-style pattern filter.

        Args:
            value (str): SQL ``LIKE`` pattern operand.

        Returns:
            Filter: ``like`` filter on this field.
        """
        return self.op("like", value)

    def ilike(self, value: str) -> Filter:
        """Build a case-insensitive LIKE-style pattern filter.

        Args:
            value (str): Case-insensitive ``LIKE`` pattern operand.

        Returns:
            Filter: ``ilike`` filter on this field.
        """
        return self.op("ilike", value)

    def wildcard(self, value: str) -> Filter:
        """Build a shell-wildcard pattern filter.

        Args:
            value (str): Shell-style wildcard pattern operand.

        Returns:
            Filter: ``wildcard`` filter on this field.
        """
        return self.op("wildcard", value)

    def regex(self, value: str) -> Filter:
        """Build a regex pattern filter.

        Args:
            value (str): Regular-expression pattern operand.

        Returns:
            Filter: ``regex`` filter on this field.
        """
        return self.op("regex", value)

    def contains(self, value: Any) -> Filter:
        """Build a text (substring) containment filter.

        Args:
            value (Any): Substring or element containment operand.

        Returns:
            Filter: ``contains`` filter on this field.
        """
        return self.op("contains", value)

    def contained_in(self, value: Any) -> Filter:
        """Build a reverse-containment filter against a larger query text.

        Args:
            value (Any): Larger query text tested for reverse containment.

        Returns:
            Filter: ``contained_in`` filter on this field.
        """
        return self.op("contained_in", value)

    def starts_with(self, value: Any) -> Filter:
        """Build a prefix (starts-with) filter.

        Args:
            value (Any): Prefix operand.

        Returns:
            Filter: ``starts_with`` filter on this field.
        """
        return self.op("starts_with", value)

    def ends_with(self, value: Any) -> Filter:
        """Build a suffix (ends-with) filter.

        Args:
            value (Any): Suffix operand.

        Returns:
            Filter: ``ends_with`` filter on this field.
        """
        return self.op("ends_with", value)

    def array_contains(self, value: Any) -> Filter:
        """Build an array membership filter.

        Args:
            value (Any): Array element tested for membership.

        Returns:
            Filter: ``array_contains`` filter on this field.
        """
        return self.op("array_contains", value)

    def exists(self, predicates: Any = None, **kw: Any) -> Filter:
        """Build a HyperG mini-row existence filter.

        Accepts a mapping (``exists({"target": "d2"})``) or keyword predicates
        (``exists(target="d2")``); the two forms are merged, keywords last.

        Args:
            predicates (Any): Optional mapping of mini-row predicates.
            **kw (Any): Keyword mini-row predicates merged after
                ``predicates``.

        Returns:
            Filter: HyperG ``exists`` filter on this field.
        """
        return self.op("exists", {**(predicates or {}), **kw})

    def any(self, predicates: Any = None, **kw: Any) -> Filter:
        """Alias for HyperG mini-row existence filters (mapping or keywords).

        Args:
            predicates (Any): Optional mapping of mini-row predicates.
            **kw (Any): Keyword mini-row predicates merged after
                ``predicates``.

        Returns:
            Filter: HyperG ``exists`` filter on this field.
        """
        return self.exists(predicates, **kw)

    def all(self, where: Any = None, **predicates: Any) -> Filter:
        """Build a HyperG universal mini-row filter.

        Args:
            where (Any): Optional predicate mapping restricting which
                mini-rows are considered.
            **predicates (Any): Required mini-row predicates that every
                matching row must satisfy.

        Returns:
            Filter: HyperG ``all`` filter on this field.
        """
        payload = {"then": predicates}
        if where is not None:
            payload["where"] = where
        return self.op("all", payload)

    def in_(self, values: List[Any]) -> Filter:
        """Build a membership-in-values filter.

        Args:
            values (List[Any]): Candidate values for membership testing.

        Returns:
            Filter: ``in`` filter on this field.
        """
        return self.op("in", values)

    def aggregate(self, op: str, *args: Any, **options: Any) -> "AggregateMetric":
        """Build an aggregate reducer intent for this field.

        Reducer lookup and parameter binding happen when the metric enters a
        `Query`, whose workspace owns the relevant module resolver.

        Args:
            op (str): Registered aggregate operation identifier.
            *args (Any): Positional reducer parameters following the group values.
            **options (Any): Named reducer parameters such as `distinct=True` or
                `q=0.95`.

        Returns:
            AggregateMetric: Immutable reducer intent used by
                `Query.aggregate`.

        Raises:
            TypeError: If reducer arguments are not JSON-compatible.

        Notes:
            Reducer lookup, signature checks, and reducer-owned value
            validation occur in `Query.aggregate`.
        """
        from ..query.spec import AggregateMetric

        return AggregateMetric(self.name, op, self.entity_id, options, args)

    def count(self, *, distinct: bool = False) -> "AggregateMetric":
        """Count non-null field values, optionally collapsing equal values.

        Args:
            distinct (bool): Whether Python-equal non-null values count once.

        Returns:
            AggregateMetric: Registered `count` reducer target.

        Raises:
            TypeError: When bound to a query, if `distinct` is not a Boolean.
        """
        from ..query.aggregate import _count_options

        normalized = _count_options({"distinct": distinct})
        return self.aggregate("count", **normalized) if distinct else self.aggregate("count")

    def min(self) -> "AggregateMetric":
        """Build a minimum reducer over non-null field values.

        Args:
            None.

        Returns:
            AggregateMetric: Registered `min` reducer target.
        """
        return self.aggregate("min")

    def max(self) -> "AggregateMetric":
        """Build a maximum reducer over non-null field values.

        Args:
            None.

        Returns:
            AggregateMetric: Registered `max` reducer target.
        """
        return self.aggregate("max")

    def sum(self) -> "AggregateMetric":
        """Build a sum reducer over non-null numeric field values.

        Args:
            None.

        Returns:
            AggregateMetric: Registered `sum` reducer target.
        """
        return self.aggregate("sum")

    def avg(self) -> "AggregateMetric":
        """Build an average reducer over non-null numeric field values.

        Args:
            None.

        Returns:
            AggregateMetric: Registered `avg` reducer target.
        """
        return self.aggregate("avg")

    def median(self) -> "AggregateMetric":
        """Build an exact continuous-median reducer over non-null numeric values.

        Args:
            None.

        Returns:
            AggregateMetric: Registered `median` reducer target returning a
            `float` for non-empty input.
        """
        return self.aggregate("median")

    def percentile(self, q: float) -> "AggregateMetric":
        """Return a continuous percentile of non-null numeric field values.

        Args:
            q (float): Percentile position between `0.0` and `1.0`, inclusive.

        Returns:
            AggregateMetric: Registered `percentile` reducer target.

        Raises:
            TypeError: When bound to a query, if `q` is not numeric or is a Boolean.
            ValueError: When bound to a query, if `q` is not finite or falls
                outside `[0.0, 1.0]`.
        """
        from ..query.aggregate import _percentile_options

        return self.aggregate("percentile", _percentile_options({"q": q})["q"])

    def code_reduce(self, code: str, dtype: Any = None) -> "AggregateMetric":
        """Reduce this field's values with a caller-supplied expression.

        The expression fills the same slot a registered reducer would, for a
        one-off question not worth registering. It executes in the query
        process; that is allowed by default, and a deployment that forbids
        running code there refuses it by setting
        `heavenbase.query.allow_code_reduce` to false.

        `dtype` is accepted as a logical type, an instance, or an identifier
        and travels as its complete JSON descriptor, so the fluent and JSON
        surfaces carry the same metric. It declares what the output column is and the result
        normalizes into it exactly as a declared field normalizes a write;
        omitting it leaves the output untyped.

        Args:
            code (str): Python expression evaluating to a one-argument callable
                over one group's non-null values, such as
                ``"lambda values: max(values) - min(values)"``.
            dtype (Any): Logical type for the result. Defaults to `None`.

        Returns:
            AggregateMetric: Registered `code_reduce` reducer target.

        Raises:
            KeyError: If `dtype` names no registered logical type.
        """
        if dtype is None:
            return self.aggregate("code_reduce", code=code)
        from ..types import logical_type_from_any

        return self.aggregate("code_reduce", code=code, dtype=logical_type_from_any(dtype).to_dict())

    def __getattr__(self, name: str) -> Callable[..., "AggregateMetric"]:
        """Build a deferred reducer intent for a dynamic method name."""
        if name.startswith("_") or not name.isidentifier():
            raise AttributeError(name)

        def build(*args: Any, **options: Any) -> "AggregateMetric":
            return self.aggregate(name, *args, **options)

        build.__name__ = name
        build.__doc__ = f"Build the {name!r} aggregate reducer intent."
        return build

    def __getitem__(self, part: str) -> "_JsonPathRef":
        """Return a nested-path reference for Json field query clauses.

        Args:
            part: Nested object key, such as ``"venue"``.

        Returns:
            _JsonPathRef: Internal path proxy whose comparisons build registered ``get``
            filters (``Doc.meta["venue"] == "NeurIPS"``) and whose ``name`` addresses
            the derived dotted column in ``group_by``, ``order_by``, and ``select``.
        """
        return _JsonPathRef(self.entity_id, self.name, [_check_json_path_part(part)])

    def __eq__(self, value: Any) -> Filter:  # type: ignore[override]
        return self.op("eq", value)

    def __ne__(self, value: Any) -> Filter:  # type: ignore[override]
        return self.op("ne", value)

    def __lt__(self, value: Any) -> Filter:
        return self.op("lt", value)

    def __le__(self, value: Any) -> Filter:
        return self.op("lte", value)

    def __gt__(self, value: Any) -> Filter:
        return self.op("gt", value)

    def __ge__(self, value: Any) -> Filter:
        return self.op("gte", value)


class _JsonPathRef:
    """Nested Json path proxy building portable ``get`` filters and dotted columns.

    Comparisons follow SQL NULL semantics: a missing or ``null`` path matches
    nothing for every operation, and heterogeneous value comparisons match
    nothing instead of raising mid-scan. The same reference names a derived
    dotted-path column in ``group_by``, ``order_by``, and ``select``: rows with
    a missing or null path keep a ``None`` value (grouping under the ``None``
    key and sorting last), and such queries take the portable Python route.

    Examples:
        >>> ws.query(Doc).where(Doc.meta["venue"] == "NeurIPS").execute().rows()
        >>> ws.query(Doc).where(Doc.meta["review"]["score"] >= 8).execute().rows()
        >>> ws.query(Doc).group_by(Doc.meta["venue"]).aggregate(n=Doc.object_id.count()).execute().rows()
    """

    def __init__(self, entity_id: str, field_name: str, path: List[str]) -> None:
        self.entity_id = entity_id
        self.field_name = field_name
        self.path = list(path)

    @property
    def name(self) -> str:
        """Return the dotted column name used by ``group_by``, ``order_by``, and ``select``.

        Returns:
            str: Base field and nested keys joined with dots, such as ``"meta.venue"``.
        """
        return ".".join([self.field_name, *self.path])

    def __getitem__(self, part: str) -> "_JsonPathRef":
        """Return a deeper path reference for one nested object key."""
        return _JsonPathRef(self.entity_id, self.field_name, [*self.path, _check_json_path_part(part)])

    def op(self, op: str, value: Any) -> Filter:
        """Build a ``get`` filter comparing this path with an explicit operation.

        Args:
            op: Comparison operation: ``eq``, ``ne``, ``in``, ``lt``, ``lte``,
                ``gt``, ``gte``, or ``contains``.
            value: Comparison operand.

        Returns:
            Filter: Registered ``get`` filter on the owning Json field.
        """
        return Filter(self.field_name, "get", {"path": list(self.path), "op": op, "value": value}, entity_id=self.entity_id)

    def contains(self, value: Any) -> Filter:
        """Build a substring containment filter on the path's string value."""
        return self.op("contains", value)

    def in_(self, values: List[Any]) -> Filter:
        """Build a membership-in-values filter on the path's value."""
        return self.op("in", values)

    def __eq__(self, value: Any) -> Filter:  # type: ignore[override]
        return self.op("eq", value)

    def __ne__(self, value: Any) -> Filter:  # type: ignore[override]
        return self.op("ne", value)

    def __lt__(self, value: Any) -> Filter:
        return self.op("lt", value)

    def __le__(self, value: Any) -> Filter:
        return self.op("lte", value)

    def __gt__(self, value: Any) -> Filter:
        return self.op("gt", value)

    def __ge__(self, value: Any) -> Filter:
        return self.op("gte", value)


def _check_json_path_part(part: Any) -> str:
    """Validate one bracket-authored Json object-key path segment."""
    if not isinstance(part, str):
        raise TypeError(f"Json path segments must be strings, got {type(part).__name__}")
    if not part:
        raise ValueError("Json path segments must not be empty")
    return part
