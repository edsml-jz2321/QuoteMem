"""Typed filter expression nodes shared by entity refs and query builders."""

from __future__ import annotations

__all__ = [
    "And",
    "Expr",
    "FieldOpNormalizer",
    "Filter",
    "Not",
    "Or",
    "SpecialExprParser",
    "expr_from_dict",
]

from heavenbase.utils import Any, Callable, Dict, List, Mapping, Optional, dataclass, field, raise_mismatch

from ..modules import ModuleResolver, module_resolver
from .ops import normalize_op

FieldOpNormalizer = Callable[[str, str], str]
SpecialExprParser = Callable[[str, Any], Optional[List["Expr"]]]


class Expr:
    """Base node for typed filter expression trees.

    Filter expressions describe caller intent. Query surfaces build them from
    Python field refs or JSON dictionaries; workspace routing compiles them
    through handler registrations.
    """

    def __and__(self, other: "Expr") -> "And":
        """Compose this expression with another expression using logical AND.

        Args:
            other (Expr): Right-hand expression.

        Returns:
            And: Conjunction containing both expressions.
        """
        return And([self, other])

    def __or__(self, other: "Expr") -> "Or":
        """Compose this expression with another expression using logical OR.

        Args:
            other (Expr): Right-hand expression.

        Returns:
            Or: Disjunction containing both expressions.
        """
        return Or([self, other])

    def __invert__(self) -> "Not":
        """Return the logical negation of this expression.

        Args:
            None.

        Returns:
            Not: Negated expression node.
        """
        return Not(self)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-compatible representation of this expression.

        Args:
            None.

        Returns:
            Dict[str, Any]: JSON-compatible expression representation.

        Raises:
            NotImplementedError: If a subclass does not implement serialization.
        """
        raise NotImplementedError

    @classmethod
    def from_dict(
        cls,
        spec: Optional[Mapping[str, Any]],
        *,
        resolver: ModuleResolver | None = None,
    ) -> Optional["Expr"]:
        """Parse a Mongo-style filter mapping into an expression tree.

        Args:
            spec (Mapping[str, Any] | None): Mongo-style filter mapping.
            resolver (ModuleResolver | None): Context-owned module resolver.
                The default Context is selected when omitted.

        Returns:
            Expr | None: Parsed expression tree, or `None` for an empty filter.

        Raises:
            TypeError: If an expression is not a mapping or `$and` / `$or`
                does not contain an array of expression mappings.
            ValueError: If a top-level logical operator is unsupported.
        """
        return expr_from_dict(spec, resolver=resolver)


@dataclass
class Filter(Expr):
    """Leaf predicate on one entity field.

    Args:
        field (str): Entity field name.
        op (str): Logical operation identifier or alias. A workspace-bound
            `Query` resolves authored aliases before carrying a
            normalized `QuerySpec`.
        value (Any): Operation value.
        entity_id (str | None): Optional owner retained by typed Python field
            references until a `Query` validates and binds the expression.
            Raw filters omit it and remain relative to the query target.
    """

    field: str
    op: str
    value: Any = None
    entity_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Return this filter as a JSON-compatible dictionary.

        Args:
            None.

        Returns:
            Dict[str, Any]: Serialized field, operation, and value.
        """
        return {"filter": {"field": self.field, "op": self.op, "value": self.value}}


@dataclass
class And(Expr):
    """Logical conjunction of child expressions.

    Args:
        clauses (List[Expr]): Child expressions that must all match.
    """

    clauses: List[Expr] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Return this conjunction as a JSON-compatible dictionary.

        Args:
            None.

        Returns:
            Dict[str, Any]: Serialized `and` expression.
        """
        return {"and": [clause.to_dict() for clause in self.clauses]}


@dataclass
class Or(Expr):
    """Logical disjunction of child expressions.

    Args:
        clauses (List[Expr]): Child expressions where any match is enough.
    """

    clauses: List[Expr] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Return this disjunction as a JSON-compatible dictionary.

        Args:
            None.

        Returns:
            Dict[str, Any]: Serialized `or` expression.
        """
        return {"or": [clause.to_dict() for clause in self.clauses]}


@dataclass
class Not(Expr):
    """Logical negation of one child expression.

    Args:
        inner (Expr): Expression to negate.
    """

    inner: Expr

    def to_dict(self) -> Dict[str, Any]:
        """Return this negation as a JSON-compatible dictionary.

        Args:
            None.

        Returns:
            Dict[str, Any]: Serialized `not` expression.
        """
        return {"not": self.inner.to_dict()}


def expr_from_dict(
    spec: Optional[Mapping[str, Any]],
    *,
    field_op_normalizer: Optional[FieldOpNormalizer] = None,
    special_parser: Optional[SpecialExprParser] = None,
    resolver: ModuleResolver | None = None,
) -> Optional[Expr]:
    """Parse a Mongo-style filter mapping into expression nodes.

    Args:
        spec (Mapping[str, Any] | None): Filter mapping to parse.
        field_op_normalizer (FieldOpNormalizer | None): Optional hook for
            field-specific operation normalization.
        special_parser (SpecialExprParser | None): Optional hook for query
            surface operators such as Catalog availability.
        resolver (ModuleResolver | None): Context-owned module resolver. The
            default Context is selected when omitted.

    Returns:
        Expr | None: Parsed expression tree, or `None` for an empty filter.

    Raises:
        TypeError: If an expression is not a mapping or `$and` / `$or` does
            not contain an array of expression mappings.
        ValueError: If a top-level logical operator is unsupported.
    """
    return _from_dict(
        spec,
        field_op_normalizer=field_op_normalizer,
        special_parser=special_parser,
        resolver=module_resolver(resolver),
    )


def _from_dict(
    spec: Optional[Mapping[str, Any]],
    *,
    field_op_normalizer: Optional[FieldOpNormalizer],
    special_parser: Optional[SpecialExprParser],
    resolver: ModuleResolver,
) -> Optional[Expr]:
    """Parse one expression subtree through an explicit module resolver."""
    if spec is None:
        return None
    if not isinstance(spec, Mapping):
        raise TypeError(f"Filter expression must be a mapping, got {type(spec).__name__}")
    if not spec:
        return None
    clauses: List[Expr] = []
    for key, value in spec.items():
        if key == "$and":
            inner = [
                _from_dict(
                    item,
                    field_op_normalizer=field_op_normalizer,
                    special_parser=special_parser,
                    resolver=resolver,
                )
                for item in _expr_items(value, key)
            ]
            inner = [item for item in inner if item is not None]
            if inner:
                clauses.append(inner[0] if len(inner) == 1 else And(inner))
            continue
        if key == "$or":
            inner = [
                _from_dict(
                    item,
                    field_op_normalizer=field_op_normalizer,
                    special_parser=special_parser,
                    resolver=resolver,
                )
                for item in _expr_items(value, key)
            ]
            inner = [item for item in inner if item is not None]
            if inner:
                clauses.append(inner[0] if len(inner) == 1 else Or(inner))
            continue
        if key == "$not":
            inner = _from_dict(
                value,
                field_op_normalizer=field_op_normalizer,
                special_parser=special_parser,
                resolver=resolver,
            )
            if inner is not None:
                clauses.append(Not(inner))
            continue
        if special_parser is not None:
            special = special_parser(key, value)
            if special is not None:
                clauses.extend(special)
                continue
        if key.startswith("$"):
            raise_mismatch(("$and", "$not", "$or"), key, name="logical filter operator")
        clauses.extend(_field_exprs(key, value, field_op_normalizer, resolver))
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else And(clauses)


def _expr_items(value: Any, operator: str) -> List[Any]:
    if not isinstance(value, list):
        raise TypeError(f"JSON filter {operator} must be an array")
    return value


def _field_exprs(
    field_name: str,
    cond: Any,
    field_op_normalizer: Optional[FieldOpNormalizer],
    resolver: ModuleResolver,
) -> List[Expr]:
    if isinstance(cond, Mapping):
        return [Filter(field_name, _normalize_field_op(field_name, op, field_op_normalizer, resolver), value) for op, value in cond.items()]
    return [Filter(field_name, "eq", cond)]


def _normalize_field_op(
    field_name: str,
    op: str,
    field_op_normalizer: Optional[FieldOpNormalizer],
    resolver: ModuleResolver,
) -> str:
    if field_op_normalizer is not None:
        return field_op_normalizer(field_name, op)
    return normalize_op(op, resolver=resolver)
