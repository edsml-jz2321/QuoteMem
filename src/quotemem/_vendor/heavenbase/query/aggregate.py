"""Registered aggregate reducers and aggregate-request validation."""

from __future__ import annotations

__all__ = [
    "AggregateReducer",
    "aggregate_reducers",
    "register_aggregate_reducer",
    "validate_aggregate",
    "validate_having",
]

import inspect
import math
import numbers
from functools import lru_cache
from dataclasses import InitVar
from typing import TYPE_CHECKING

from heavenbase.utils import Any, Callable, Dict, List, Mapping, Optional, Sequence, dataclass, deepcopy, raise_mismatch
from heavenbase.utils.registry import python_target

from ..filters.ops import normalize_op
from ..filters.spec import And, Expr, Filter, Not, Or
from ..modules import ModuleResolver, module_resolver, python_executable_ref
from ..modules.base import ResolverCache
from heavenbase.version import __version__
from ..types import (
    LogicalType,
    logical_type_from_any,
)
from ..utils.ids import check_id
from ..utils.serialize import validate_json
from .json_path import split_json_path
from .spec import Aggregate, _check_aggregate_method_name, _thaw_json

if TYPE_CHECKING:
    from ..modules import ModuleService

_OptionNormalizer = Callable[[Mapping[str, Any]], Mapping[str, Any]]


@dataclass(frozen=True)
class AggregateReducer:
    """Describe one portable aggregation operation.

    Args:
        identifier: Stable operation identifier.
        fold: Function receiving non-null values followed by named reducer
            parameters for one group.
        input_modes: Optional aggregate modes accepted by the reducer.
            An empty tuple accepts every logical type.
        normalize_options: Optional reducer-owned parameter normalization hook.
        result_type: Logical type returned by the reducer, `"source"` when it
            preserves the source field type, a resolver callable receiving the
            source type and normalized options, or `None` when unknown.
        resolver: Context-owned resolver used to normalize a fixed result type.
            It is not retained by the descriptor.
    """

    identifier: str
    fold: Callable[..., Any]
    input_modes: tuple[str, ...] = ()
    normalize_options: Optional[_OptionNormalizer] = None
    result_type: Any = None
    resolver: InitVar[ModuleResolver | None] = None

    def __post_init__(self, resolver: ModuleResolver | None) -> None:
        """Normalize the identifier and validate the registered fold signature."""
        identifier = check_id(self.identifier, "aggregate reducer identifier")
        object.__setattr__(self, "identifier", _check_aggregate_method_name(identifier))
        if not callable(self.fold):
            raise TypeError("Aggregate reducer fold must be callable")
        if inspect.iscoroutinefunction(self.fold):
            raise TypeError("Aggregate reducer fold must be synchronous")
        object.__setattr__(self, "input_modes", tuple(check_id(item, "aggregate input mode") for item in self.input_modes))
        if self.normalize_options is not None and not callable(self.normalize_options):
            raise TypeError("Aggregate reducer normalize_options must be callable")
        object.__setattr__(self, "result_type", _normalize_result_type(self.result_type, resolver))
        parameters = tuple(inspect.signature(self.fold).parameters.values())
        if not parameters or parameters[0].kind not in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
            raise TypeError("Aggregate reducer fold must accept non-null group values as its first parameter")
        unsupported = [
            parameter.name
            for parameter in parameters[1:]
            if parameter.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        ]
        if unsupported:
            raise TypeError(f"Aggregate reducer parameters must be named and JSON-addressable, got {unsupported}")
        for parameter in parameters[1:]:
            if parameter.name == "field":
                raise ValueError("Aggregate reducer parameter name 'field' is reserved for the metric source field")
            if parameter.default is inspect.Parameter.empty:
                continue
            try:
                validate_json(parameter.default)
            except (TypeError, ValueError) as exc:
                raise TypeError(f"Aggregate reducer parameter {parameter.name!r} default must be JSON-compatible") from exc

    @property
    def parameters(self) -> tuple[inspect.Parameter, ...]:
        """Return reducer parameters following the group-values input.

        Returns:
            tuple[inspect.Parameter, ...]: Named parameters accepted by Python
            and JSON aggregate metrics.
        """
        return tuple(inspect.signature(self.fold).parameters.values())[1:]

    def bind(self, *args: Any, **options: Any) -> dict[str, Any]:
        """Bind and normalize one reducer invocation.

        Args:
            *args (Any): Positional reducer parameters after group values.
            **options (Any): Named reducer parameters.

        Returns:
            dict[str, Any]: Canonical explicit options. Omitted defaults stay
            omitted and are supplied by the registered fold at execution.

        Raises:
            TypeError: If arguments do not match the registered fold signature
                or the normalizer returns a non-mapping value.
            ValueError: If reducer-owned option normalization rejects a value.
        """
        return self._bind(args, options, include_defaults=False)

    def result_dtype(
        self,
        source_dtype: Any,
        options: Mapping[str, Any],
        *,
        resolver: ModuleResolver | None = None,
    ) -> Any:
        """Resolve the logical result type used by typed `HAVING` predicates.

        Args:
            source_dtype (Any): Logical type of the reducer source field.
            options (Mapping[str, Any]): Explicit reducer parameters.
            resolver (ModuleResolver | None): Context-owned module resolver.
                The default Context is selected when a dynamic result-type
                callback needs logical-type resolution and this is omitted.

        Returns:
            LogicalType | None: Resolved logical result type, or `None` when the
            reducer deliberately leaves its output untyped.

        Raises:
            TypeError: If a resolver returns an invalid logical type.
        """
        if self.result_type is None:
            return None
        if self.result_type == "source":
            return source_dtype
        if isinstance(self.result_type, LogicalType):
            return self.result_type
        resolved = self.result_type(source_dtype, self._bind((), options, include_defaults=True))
        return None if resolved is None else logical_type_from_any(resolved, resolver=resolver)

    def reduce(self, values: List[Any], options: Mapping[str, Any], *, config: Any = None) -> Any:
        """Reduce non-null group values with validated metric options.

        Args:
            values (List[Any]): Non-null logical values for one group.
            options (Mapping[str, Any]): Explicit metric parameters.
            config (Any): Config authority for reducer execution policy.
                Defaults to the process config for direct reducer calls.

        Returns:
            Any: Reducer-defined aggregate result.
        """
        bound = self._bind((), options, include_defaults=True)
        if self.identifier == "code_reduce":
            _require_code_reduce(config)
        return self.fold(values, **bound)

    def to_dict(self) -> dict[str, Any]:
        """Return registry-derived reducer metadata.

        Args:
            None.

        Returns:
            dict[str, Any]: Identifier, accepted aggregate modes, named
            reducer parameters, and result-type capability metadata.
        """
        parameters = []
        for parameter in self.parameters:
            item = {"name": parameter.name, "required": parameter.default is inspect.Parameter.empty}
            if parameter.default is not inspect.Parameter.empty:
                item["default"] = deepcopy(parameter.default)
            parameters.append(item)
        return {
            "identifier": self.identifier,
            "input_modes": list(self.input_modes),
            "parameters": parameters,
            "result_type": _result_type_metadata(self.result_type),
        }

    def _bind(self, args: Sequence[Any], options: Mapping[str, Any], *, include_defaults: bool) -> dict[str, Any]:
        """Bind explicit parameters and normalize against isolated defaults."""
        signature = inspect.signature(self.fold)
        try:
            bound = signature.bind([], *_thaw_json(args), **_thaw_json(options))
        except TypeError as exc:
            raise TypeError(f"Aggregate reducer {self.identifier!r} parameters are invalid: {exc}") from None
        values_name = next(iter(signature.parameters))
        explicit = {name: value for name, value in bound.arguments.items() if name != values_name}
        effective = dict(explicit)
        for parameter in self.parameters:
            if parameter.name not in effective and parameter.default is not inspect.Parameter.empty:
                effective[parameter.name] = deepcopy(parameter.default)
        normalized: Mapping[str, Any] = effective
        if self.normalize_options is not None:
            normalized = self.normalize_options(_thaw_json(effective))
        if not isinstance(normalized, Mapping):
            raise TypeError(f"Aggregate reducer {self.identifier!r} option normalizer must return a mapping")
        canonical = _thaw_json(normalized)
        try:
            signature.bind([], **canonical)
        except TypeError as exc:
            raise TypeError(f"Aggregate reducer {self.identifier!r} normalized parameters are invalid: {exc}") from None
        try:
            validate_json(canonical)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"Aggregate reducer {self.identifier!r} parameters must be JSON-compatible") from exc
        if include_defaults:
            return canonical
        return {name: canonical[name] for name in explicit if name in canonical}


def _normalize_result_type(result_type: Any, resolver: ModuleResolver | None) -> Any:
    """Normalize fixed logical output types while preserving resolvers."""
    if result_type is None or result_type == "source":
        return result_type
    if isinstance(result_type, LogicalType) or (isinstance(result_type, type) and issubclass(result_type, LogicalType)):
        return logical_type_from_any(result_type, resolver=module_resolver(resolver))
    if callable(result_type):
        if inspect.iscoroutinefunction(result_type):
            raise TypeError("Aggregate reducer result_type resolver must be synchronous")
        try:
            inspect.signature(result_type).bind(None, {})
        except TypeError as exc:
            raise TypeError(f"Aggregate reducer result_type resolver must accept source dtype and options: {exc}") from None
        return result_type
    try:
        return logical_type_from_any(result_type, resolver=module_resolver(resolver))
    except (KeyError, TypeError, ValueError) as exc:
        raise TypeError(f"Aggregate reducer result_type must be a logical type, 'source', resolver, or None; got {result_type!r}") from exc


def _result_type_metadata(result_type: Any) -> Optional[str]:
    """Return one JSON-compatible result-type capability label."""
    if result_type is None or result_type == "source":
        return result_type
    if isinstance(result_type, LogicalType):
        return type(result_type).identifier
    return "resolver"


def _tag_claim(row: Mapping[str, Any], identifier: str) -> Any:
    """Return one raw subject claim from a Registry inspection row."""
    record = row.get("record")
    meta = record.get("meta") if isinstance(record, Mapping) else None
    claims = meta.get("tags") if isinstance(meta, Mapping) else None
    return claims.get(identifier) if isinstance(claims, Mapping) else None


def _tag_strings(
    row: Mapping[str, Any],
    identifier: str,
    *,
    default: tuple[str, ...],
) -> tuple[str, ...]:
    """Interpret one aggregate-owned extensible string-set claim."""
    value = _tag_claim(row, identifier)
    if value is None:
        return default
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{row['kind']} {row['identifier']!r} tag {identifier!r} must be a list of identifiers")
    if any(not isinstance(item, str) for item in value):
        raise TypeError(f"{row['kind']} {row['identifier']!r} tag {identifier!r} must contain only identifiers")
    return tuple(dict.fromkeys(check_id(item, f"{identifier} value") for item in value))


def _aggregate_modes(dtype: Any, resolver: ModuleResolver) -> tuple[str, ...]:
    """Return locally validated aggregate modes for one logical type."""
    identifier = getattr(type(dtype), "identifier", None)
    if not isinstance(identifier, str):
        raise TypeError(f"Aggregate source {dtype!r} has no logical-type identifier")
    rows = resolver.inspect("logical_type", identifier)
    if not rows:
        raise LookupError(f"Unknown aggregate logical type {identifier!r}")
    return _tag_strings(rows[0], "aggregate-modes", default=())


_REDUCER_CACHE = ResolverCache[AggregateReducer]()


def _module_service(resolver: ModuleResolver) -> "ModuleService":
    """Require lifecycle authority for an aggregate-reducer mutation."""
    from ..modules import ModuleService

    if not isinstance(resolver, ModuleService):
        raise TypeError("Aggregate reducer mutation requires a ModuleService resolver")
    return resolver


def _result_type_record(result_type: Any, resolver: ModuleResolver) -> Any:
    """Encode one validated result type as JSON registration data."""
    if result_type is None or result_type == "source":
        return result_type
    if isinstance(result_type, LogicalType):
        return type(result_type).identifier
    if isinstance(result_type, type) and issubclass(result_type, LogicalType):
        return result_type.identifier
    if callable(result_type):
        return python_executable_ref(result_type, label="Aggregate reducer result_type")
    return logical_type_from_any(result_type, resolver=resolver).identifier


def _result_type_value(value: Any, resolver: ModuleResolver, owner: Mapping[str, Any]) -> Any:
    """Restore one result-type definition from registration data."""
    if isinstance(value, Mapping):
        return resolver.resolve_executable(value, owner=owner)
    return value


def _reducer_from_row(row: Mapping[str, Any], resolver: ModuleResolver) -> AggregateReducer:
    """Materialize and cache one validated Registry reducer record."""

    def build(current_row: Mapping[str, Any]) -> AggregateReducer:
        definition = current_row["record"]["meta"].get("reducer")
        if not isinstance(definition, Mapping):
            raise TypeError(f"Aggregate reducer {current_row['identifier']!r} meta.reducer must be a mapping")
        input_modes = _tag_strings(current_row, "aggregate-input-modes", default=())
        normalizer_ref = definition.get("normalize_options")
        if normalizer_ref is None:
            normalizer = None
        elif isinstance(normalizer_ref, Mapping):
            normalizer = resolver.resolve_executable(normalizer_ref, owner=current_row)
        else:
            raise TypeError(f"Aggregate reducer {current_row['identifier']!r} normalize_options must be an executable reference or null")
        return AggregateReducer(
            str(current_row["identifier"]),
            resolver.resolve("reducer", str(current_row["identifier"])),
            input_modes,
            normalizer,
            _result_type_value(definition.get("result_type"), resolver, current_row),
            resolver=resolver,
        )

    return _REDUCER_CACHE.get(resolver, row, build)


def register_aggregate_reducer(
    op: str,
    fold: Callable[..., Any],
    *,
    tags: Mapping[str, Any] | None = None,
    normalize_options: Optional[_OptionNormalizer] = None,
    result_type: Any = None,
    replace: bool = False,
    resolver: ModuleResolver | None = None,
) -> AggregateReducer:
    """Register a reducer used by Python, JSON, explain, and capability metadata.

    Args:
        op: Stable aggregate operation identifier.
        fold: Portable correctness fold over non-null values. Additional named
            parameters automatically become Python/JSON metric options.
        tags: Raw JSON-compatible Tag claims copied onto the reducer record.
            `aggregate=true` is always added. The optional
            `aggregate-input-modes` string sequence limits accepted source
            modes; an absent or empty value accepts all.
        normalize_options: Optional hook that validates or canonicalizes bound
            reducer options before request construction and execution.
        result_type: Logical output type, `"source"`, or a resolver callable.
            Supplying it gives custom reducers the same typed `HAVING` operand
            normalization and predicate validation as core reducers.
        replace: Whether to replace an existing custom reducer registration.
            Built-in reducers cannot be replaced because public behavior and
            native compilers depend on their stable semantics.
        resolver: Context-owned module resolver. The default Context is
            selected when omitted. Registration requires a `ModuleService`.

    Returns:
        AggregateReducer: The registered immutable descriptor.

    Raises:
        TypeError: If the fold, parameters, defaults, normalizer, or result type
            violates the registry contract.
        ValueError: If the identifier is invalid, collides with `FieldRef`, is
            already registered, or attempts to replace a core reducer.

    Examples:
        ```python
        def spread(values, scale=1.0):
            return (max(values) - min(values)) * scale if values else None

        hb.ext.register_aggregate_reducer(
            "spread",
            spread,
            tags={"aggregate-input-modes": ["numeric"]},
            result_type=hb.Float,
        )
        generic = Sale.amount.aggregate("spread", scale=2.0)
        dynamic = Sale.amount.spread(scale=2.0)
        ```
    """
    selected = module_resolver(resolver)
    if tags is not None and not isinstance(tags, Mapping):
        raise TypeError("Aggregate reducer tags must be a string-keyed mapping")
    tag_values = dict(tags or {})
    if not all(isinstance(identifier, str) for identifier in tag_values):
        raise TypeError("Aggregate reducer tags must be a string-keyed mapping")
    tag_values["aggregate"] = True
    raw_input_modes = tag_values.get("aggregate-input-modes", [])
    if isinstance(raw_input_modes, (str, bytes)) or not isinstance(raw_input_modes, Sequence):
        raise TypeError("Aggregate reducer aggregate-input-modes Tag must be a sequence of strings")
    if not all(isinstance(mode, str) for mode in raw_input_modes):
        raise TypeError("Aggregate reducer aggregate-input-modes Tag must be a sequence of strings")
    input_modes = tuple(raw_input_modes)
    reducer = AggregateReducer(op, fold, input_modes, normalize_options, result_type, resolver=selected)
    service = _module_service(selected)
    existing = service.inspect("reducer", reducer.identifier)
    if existing:
        publisher = existing[0]["record"]["meta"]["installation"]["publisher"]
        if publisher == "heavenbase":
            raise ValueError(f"Built-in aggregate reducer {reducer.identifier!r} cannot be replaced")
        if not replace:
            raise ValueError(f"Aggregate reducer {reducer.identifier!r} is already registered")
    fold_target = python_target(fold, label="Aggregate reducer fold")
    normalizer_ref = python_executable_ref(normalize_options, label="Aggregate reducer normalize_options") if normalize_options is not None else None
    owner = f"programmatic:reducer:{reducer.identifier}"
    items: list[dict[str, Any]] = []
    operation = service.inspect("operation", reducer.identifier)
    if operation:
        if _tag_claim(operation[0], "aggregate") is not True:
            raise ValueError(f"Operation {reducer.identifier!r} must claim aggregate=true before reducer registration")
    else:
        items.append(
            {
                "kind": "operation",
                "identifier": reducer.identifier,
                "aliases": [f"${reducer.identifier}"],
                "name": reducer.identifier.replace("_", " ").title(),
                "desc": "Registered aggregate operation.",
                "source": "inline",
                "target": "definition",
                "active": True,
                "meta": {"schema_version": 1, "definition": {"kind": "operation"}, "tags": {"aggregate": True}},
            }
        )
    items.append(
        {
            "kind": "reducer",
            "identifier": reducer.identifier,
            "name": reducer.identifier.replace("_", " ").title(),
            "desc": "Portable aggregate reducer.",
            "source": "python-import",
            "target": fold_target,
            "active": True,
            "meta": {
                "schema_version": 1,
                "dependencies": [f"operation:{reducer.identifier}"],
                "tags": tag_values,
                "reducer": {
                    "normalize_options": normalizer_ref,
                    "result_type": _result_type_record(reducer.result_type, selected),
                },
            },
        }
    )
    service.install(
        {
            "manifest_version": 2,
            "coordinate": f"programmatic/reducer/{reducer.identifier}",
            "version": __version__,
            "compatibility": {},
            "bundle": f"reducer-{reducer.identifier}",
            "items": items,
        },
        publisher=owner,
    )
    _REDUCER_CACHE.clear(selected)
    return reducer_for(reducer.identifier, resolver=selected)


def aggregate_reducers(*, resolver: ModuleResolver | None = None) -> tuple[AggregateReducer, ...]:
    """Return registered reducers in deterministic registration order.

    Args:
        resolver: Context-owned module resolver. The default Context is
            selected when omitted.

    Returns:
        tuple[AggregateReducer, ...]: Live reducer descriptors.
    """
    selected = module_resolver(resolver)
    return tuple(_reducer_from_row(row, selected) for row in selected.inspect("reducer"))


def reducer_for(op: str, *, resolver: ModuleResolver | None = None) -> AggregateReducer:
    """Return one reducer or raise a contextual lookup error.

    Args:
        op: Aggregate operation identifier or alias.
        resolver: Context-owned module resolver. The default Context is
            selected when omitted.

    Returns:
        AggregateReducer: Registered immutable reducer descriptor.

    Raises:
        LookupError: If `op` has no active reducer registration.
    """
    selected = module_resolver(resolver)
    name = normalize_op(op, resolver=selected)
    found = selected.inspect("reducer", name)
    if not found:
        known = ", ".join(item.identifier for item in aggregate_reducers(resolver=selected))
        raise LookupError(f"Unknown aggregate operation {op!r}; registered operations: {known}")
    return _reducer_from_row(found[0], selected)


def validate_aggregate(
    schema: Any,
    aggregate: Aggregate,
    *,
    resolver: ModuleResolver | None = None,
) -> None:
    """Validate a normalized aggregate against the queried entity schema.

    Group fields are declared groupable columns or dotted Json paths such as
    ``meta.venue``; dotted paths carry dynamic payload values, so they skip the
    logical type's `aggregate-modes` group check and always fold portably.
    Metric sources must be declared fields.

    Args:
        schema: Queried entity schema.
        aggregate: Frozen aggregate projection.
        resolver: Context-owned module resolver. The default Context is
            selected when omitted.

    Returns:
        None: A valid aggregate has no return value.

    Raises:
        ValueError: If a field is foreign, unknown, ungroupable, sourced from a
            dotted Json path metric, or incompatible with the selected reducer.
        LookupError: If a dotted group or metric name has a declared non-Json
            base field.
    """
    selected = module_resolver(resolver)
    for field_name in aggregate.group_by:
        if split_json_path(schema, field_name) is not None:
            continue  # dotted Json path groups hold dynamic payload values; the portable fold groups them directly
        dtype = _field_dtype(schema, field_name)
        if "group" not in _aggregate_modes(dtype, selected):
            raise ValueError(f"Field {schema.entity_id}.{field_name} ({type(dtype).identifier}) cannot be used as an aggregate group")
    for output, metric in aggregate.metrics.items():
        if metric.entity_id is not None and metric.entity_id != schema.entity_id:
            raise ValueError(f"Aggregate output {output!r} uses {metric.entity_id}.{metric.field}, but the query targets {schema.entity_id}")
        if split_json_path(schema, metric.field) is not None:
            raise ValueError(
                f"Aggregate output {output!r} reads {schema.entity_id}.{metric.field}; metric sources need a declared field — "
                "dotted Json paths apply to group_by, order, and select"
            )
        dtype = _field_dtype(schema, metric.field)
        try:
            reducer = reducer_for(metric.op, resolver=selected)
        except LookupError as exc:
            raise ValueError(str(exc)) from None
        options = reducer.bind(**_thaw_json(metric.options))
        if reducer.input_modes and not set(reducer.input_modes).intersection(_aggregate_modes(dtype, selected)):
            raise ValueError(
                f"Aggregate operation {reducer.identifier!r} does not support {type(dtype).identifier} field " f"{schema.entity_id}.{metric.field}"
            )
        reducer.result_dtype(dtype, options, resolver=selected)


def validate_having(
    schema: Any,
    aggregate: Aggregate,
    having: Optional[Expr],
    *,
    resolver: ModuleResolver | None = None,
    config: Any | None = None,
) -> None:
    """Validate a post-aggregate predicate against result columns and predicates.

    Args:
        schema: Queried entity schema used to normalize typed predicate values.
        aggregate: Frozen aggregate projection defining available result columns.
        having: Optional filter expression over group fields and metric aliases.
        resolver: Context-owned module resolver. The default Context is
            selected when omitted.

    Returns:
        None: A valid `HAVING` clause has no return value.

    Raises:
        TypeError: If the expression tree contains an unsupported node.
        ValueError: If a field alias or portable predicate is unavailable, or a
            predicate query value cannot be prepared.
    """
    if having is None:
        return
    from ..execution.handlers.predicates import Predicate
    from ..execution.handlers.seed import filter_ops

    selected = module_resolver(resolver)
    columns = {*aggregate.group_by, *aggregate.metrics}
    for clause in _having_filters(having):
        if clause.field not in columns:
            raise ValueError(f"Unknown HAVING field {clause.field!r}; available aggregate columns: {sorted(columns)}")
        dtype = _having_dtype(schema, aggregate, clause.field, selected)
        predicate = Predicate.load(clause.op, dtype, resolver=selected)
        if dtype is not None and clause.op not in filter_ops(dtype, resolver=selected):
            dtype_id = getattr(dtype, "identifier", getattr(type(dtype), "identifier", type(dtype).__name__))
            raise LookupError(f"{dtype_id} aggregate fields do not support {clause.op} predicates")
        value = _normalize_having_value(dtype, clause.op, clause.value, selected)
        predicate.compile(value, resolver=selected, config=config)


def having_predicate(
    schema: Any,
    aggregate: Aggregate,
    having: Optional[Expr],
    *,
    resolver: ModuleResolver | None = None,
    config: Any | None = None,
) -> Callable[[Mapping[str, Any]], bool]:
    """Prepare one validated portable predicate over aggregate result rows.

    Args:
        schema: Queried entity schema used to normalize typed predicate values.
        aggregate: Frozen aggregate projection defining result columns.
        having: Filter expression over result columns.
        resolver: Context-owned module resolver. The default Context is
            selected when omitted.
        config: Configuration authority used by registered predicate compilers.

    Returns:
        Callable[[Mapping[str, Any]], bool]: Prepared row predicate.

    Raises:
        ValueError: If `having` is omitted or references an invalid aggregate
            result column.
    """
    if having is None:
        raise ValueError("HAVING predicate preparation requires an expression")
    validate_having(schema, aggregate, having, resolver=resolver, config=config)
    return _compile_having(schema, aggregate, having, module_resolver(resolver), config)


def _having_filters(expr: Expr) -> list[Filter]:
    """Flatten one supported `HAVING` expression into predicate leaves."""
    if isinstance(expr, Filter):
        return [expr]
    if isinstance(expr, (And, Or)):
        return [leaf for clause in expr.clauses for leaf in _having_filters(clause)]
    if isinstance(expr, Not):
        return _having_filters(expr.inner)
    raise TypeError(f"Unsupported HAVING expression {type(expr).__name__}")


def _compile_having(
    schema: Any,
    aggregate: Aggregate,
    expr: Expr,
    resolver: ModuleResolver,
    config: Any | None,
) -> Callable[[Mapping[str, Any]], bool]:
    """Compile a validated `HAVING` expression into one row predicate."""
    from ..execution.handlers.predicates import Predicate

    if isinstance(expr, Filter):
        dtype = _having_dtype(schema, aggregate, expr.field, resolver)
        predicate = Predicate.load(expr.op, dtype, resolver=resolver)
        value = _normalize_having_value(dtype, expr.op, expr.value, resolver)
        prepared = predicate.compile(value, resolver=resolver, config=config)

        def match(row: Mapping[str, Any]) -> bool:
            try:
                return predicate.match(row.get(expr.field), prepared)
            except TypeError:
                if row.get(expr.field) is None:
                    return False
                raise

        return match
    if isinstance(expr, And):
        clauses = [_compile_having(schema, aggregate, clause, resolver, config) for clause in expr.clauses]
        return lambda row: all(clause(row) for clause in clauses)
    if isinstance(expr, Or):
        clauses = [_compile_having(schema, aggregate, clause, resolver, config) for clause in expr.clauses]
        return lambda row: any(clause(row) for clause in clauses)
    if isinstance(expr, Not):
        inner = _compile_having(schema, aggregate, expr.inner, resolver, config)
        return lambda row: not inner(row)
    raise TypeError(f"Unsupported HAVING expression {type(expr).__name__}")


def _having_dtype(
    schema: Any,
    aggregate: Aggregate,
    field_name: str,
    resolver: ModuleResolver,
) -> Any:
    """Return a known logical result type for HAVING value normalization."""
    if field_name in aggregate.group_by:
        if split_json_path(schema, field_name) is not None:
            return None  # dotted Json path groups hold dynamic values, so HAVING operands stay unnormalized
        return schema.get(field_name).dtype
    metric = aggregate.metrics[field_name]
    return reducer_for(metric.op, resolver=resolver).result_dtype(
        schema.get(metric.field).dtype,
        metric.options,
        resolver=resolver,
    )


def _normalize_having_value(
    dtype: Any,
    op: str,
    value: Any,
    resolver: ModuleResolver,
) -> Any:
    """Normalize a typed HAVING operand while preserving aggregate nulls."""
    if dtype is None:
        return value
    if value is None and op in ("eq", "ne"):
        return None
    if op == "in" and isinstance(value, (list, tuple, set)):
        from ..execution.handlers.filters import normalize_filter_value

        return [None if item is None else normalize_filter_value(dtype, "eq", item, resolver=resolver) for item in value]
    from ..execution.handlers.filters import normalize_filter_value

    return normalize_filter_value(dtype, op, value, resolver=resolver)


def _field_dtype(schema: Any, field_name: str) -> Any:
    """Return one field dtype with aggregate-specific diagnostics."""
    try:
        return schema.get(field_name).dtype
    except KeyError:
        raise ValueError(f"Unknown aggregate field {schema.entity_id}.{field_name}") from None


_CODE_REDUCE_KEY = "heavenbase.query.allow_code_reduce"


def _require_code_reduce(config: Any = None) -> None:
    """Require caller-authored reductions in the executing config authority."""
    from heavenbase.context.authority import omitted_config

    manager = omitted_config() if config is None else config
    if manager.get(_CODE_REDUCE_KEY, default=False) is not True:
        raise ValueError(f"code_reduce executes caller-supplied code in this process and is disabled; set {_CODE_REDUCE_KEY} to true to allow it")


@lru_cache(maxsize=256)
def _compiled_reduction(code: str) -> Callable[[List[Any]], Any]:
    """Compile one reduction expression, reusing the result for repeated groups.

    A grouped query calls the fold once per group, so compiling per call would
    scale with group count rather than with distinct expressions.

    Args:
        code (str): Python expression evaluating to a one-argument callable.

    Returns:
        Callable[[List[Any]], Any]: The callable the expression produced.

    Raises:
        ValueError: If the expression does not parse or does not evaluate to a
            callable taking one argument.
    """
    try:
        produced = eval(compile(code, "<code_reduce>", "eval"), {"__builtins__": __builtins__}, {})  # noqa: S307 - caller-authored code is the feature
    except SyntaxError as error:
        raise ValueError(f"code_reduce expression does not parse: {error}") from None
    if not callable(produced):
        raise ValueError(f"code_reduce expression must evaluate to a callable over one group's values, got {type(produced).__name__}")
    return produced


def _code_reduce(values: List[Any], code: str, dtype: Any = None) -> Any:
    """Reduce one group's values with a caller-supplied expression.

    The expression is evaluated once and applied to the list of non-null values
    for each group, so it fills the same slot as a registered reducer without
    needing one. It runs in this process with no restriction. That is allowed
    by default; setting `heavenbase.query.allow_code_reduce` to false refuses it
    engine-wide. Registered reducers such as ``column_summary`` remain available.

    `dtype` declares what the output column is, and the value is normalized
    into it exactly as a declared field normalizes a write — an `Integer`
    column truncates, and a value the type cannot accept raises. Omitting it
    leaves the output untyped, which `HAVING` compares without normalization.

    Args:
        values (List[Any]): Non-null values for one group.
        code (str): Python expression evaluating to a one-argument callable,
            such as ``"lambda values: max(values) - min(values)"``.
        dtype (Any): Logical type for the result, as a type, an instance, or an
            identifier such as ``"float"``. Defaults to `None`, leaving the
            output untyped.

    Returns:
        Any: The expression's result, normalized into `dtype` when one is given.

    Raises:
        ValueError: If code execution is not enabled, the expression does not
            parse or produce a callable, or the result does not fit `dtype`.
    """
    result = _compiled_reduction(code)(values)
    return result if dtype is None else logical_type_from_any(dtype).validate(result)


def _code_reduce_result_type(source_dtype: Any, options: Mapping[str, Any]) -> Any:
    """Return the caller-declared output descriptor for typed HAVING."""
    del source_dtype
    return options.get("dtype")


def _key_counts(values: List[Any]) -> Dict[str, int]:
    """Count, per key, how many payloads in one group carried that key.

    A `Json` column whose keys vary per row cannot be grouped by those keys —
    they are values inside the payload, not a column. This folds the whole
    group into one mapping instead, so a caller learns every key and its exact
    row count from one aggregate rather than by paging rows and counting.

    Args:
        values (List[Any]): Non-null payload values for one group.

    Returns:
        Dict[str, int]: Key mapped to the number of payloads containing it,
        most common first. Payloads that are not objects contribute nothing.
    """
    counts: Dict[str, int] = {}
    objects = 0
    for value in values:
        if isinstance(value, Mapping):
            objects += 1
            for key in value:
                counts[str(key)] = counts.get(str(key), 0) + 1
    _require_object_payloads(values, objects, "key_counts")
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _key_sums(values: List[Any]) -> Dict[str, Any]:
    """Total, per key, the numeric values one group's payloads hold at that key.

    This is the fold for a counter-shaped `Json` column — one whose values are
    per-key tallies. A key missing from a payload contributes nothing rather
    than zero, matching how a declared numeric column treats a null.

    Args:
        values (List[Any]): Non-null payload values for one group.

    Returns:
        Dict[str, Any]: Key mapped to the sum of its numeric values, largest
        first. Non-object payloads and non-numeric values are skipped.

    Raises:
        ValueError: If a key holds a Boolean, which would otherwise fold as
            0/1 and silently corrupt a total.
    """
    totals: Dict[str, Any] = {}
    objects = 0
    for value in values:
        if not isinstance(value, Mapping):
            continue
        objects += 1
        for key, item in value.items():
            if isinstance(item, bool):
                raise_mismatch(
                    ("number",),
                    "boolean",
                    name=f"key_sums value at {str(key)!r}",
                    comment=f"got {item!r}; a Boolean would fold as 0/1 and corrupt the total",
                )
            if isinstance(item, numbers.Number):
                totals[str(key)] = totals.get(str(key), 0) + item
    _require_object_payloads(values, objects, "key_sums")
    if objects and not totals:
        sample = next((f"{key!r}: {item!r}" for value in values if isinstance(value, Mapping) for key, item in value.items()), "")
        raise ValueError(
            f"key_sums found no numeric value across {objects} object payload(s); the first key holds {sample}. "
            "An empty result would be indistinguishable from totals of zero — use key_counts to count rows per key, "
            "or a registered reducer shaped for these values."
        )
    return dict(sorted(totals.items(), key=lambda item: (-item[1], item[0])))


def _value_counts(values: List[Any]) -> Dict[str, int]:
    """Count how often each element appears across one group's list payloads.

    A `Json` column holding tag lists answers "how often does each tag occur"
    only by looking inside the lists, which no declared column exposes. This
    folds the whole group into one mapping. A scalar payload counts as itself,
    so a column mixing lists and single values still totals correctly.

    Args:
        values (List[Any]): Non-null payload values for one group.

    Returns:
        Dict[str, int]: Element mapped to its occurrence count, most common
        first. Mapping payloads are skipped; use `key_counts` for those.

    Raises:
        ValueError: If the group held values but none of them yielded a
            countable element, since an empty result would be
            indistinguishable from a column with nothing in it.
    """
    counts: Dict[str, int] = {}
    seen = 0
    for value in values:
        if isinstance(value, Mapping):
            continue
        items = value if isinstance(value, (list, tuple)) else [value]
        for item in items:
            if isinstance(item, (list, tuple, Mapping)):
                continue
            seen += 1
            counts[str(item)] = counts.get(str(item), 0) + 1
    if values and not seen:
        sample = repr(values[0])[:80]
        raise ValueError(
            f"value_counts found no countable element in {len(values)} value(s); the first is {sample}. "
            "An empty result would be indistinguishable from an empty column — use key_counts or key_sums "
            "for object payloads."
        )
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _require_object_payloads(values: List[Any], objects: int, op: str) -> None:
    """Reject a fold whose whole input held no object to read keys from.

    A sparse column is ordinary: rows without an object contribute nothing, the
    same way a null contributes nothing to a numeric fold. A column where
    *nothing* is an object is different — the empty mapping it would return is
    indistinguishable from "these rows genuinely had no keys", which is how a
    wrong answer gets reported as a real one. A doubly-encoded payload, stored
    as a JSON string whose content is itself JSON, lands here.

    Args:
        values (List[Any]): Non-null payload values for one group.
        objects (int): How many of them were mappings.
        op (str): Reducer identifier used in the error message.

    Returns:
        None: A group with at least one object payload, or none at all, passes.

    Raises:
        ValueError: If the group had values but none of them was an object.
    """
    if values and not objects:
        sample = repr(values[0])[:80]
        raise ValueError(
            f"{op} found no object payload in {len(values)} value(s); the first is {sample}. "
            "An empty result would be indistinguishable from a column with no keys — decode the payload "
            "to objects first if it is stored as encoded text."
        )


def _count(values: List[Any], distinct: bool = False) -> int:
    """Count non-null values, optionally collapsing Python-equal values."""
    if not distinct:
        return len(values)
    unique: list[Any] = []
    for value in values:
        if not any(value == existing for existing in unique):
            unique.append(value)
    return len(unique)


def _count_options(options: Mapping[str, Any]) -> Mapping[str, Any]:
    """Require `count(distinct=...)` to use an explicit Boolean flag."""
    if "distinct" in options and not isinstance(options["distinct"], bool):
        raise TypeError("Aggregate count distinct must be a boolean")
    return options


def _percentile_options(options: Mapping[str, Any]) -> Mapping[str, Any]:
    """Normalize the continuous percentile control to a finite unit float."""
    value = options["q"]
    if isinstance(value, bool):
        raise TypeError("Aggregate percentile q must be a float between 0 and 1")
    try:
        q = float(value)
    except (TypeError, ValueError):
        raise TypeError("Aggregate percentile q must be a float between 0 and 1") from None
    if not math.isfinite(q) or not 0.0 <= q <= 1.0:
        raise ValueError("Aggregate percentile q must be between 0 and 1 inclusive")
    return {**options, "q": q}


def _percentile(values: List[Any], q: float) -> Any:
    """Return the linearly interpolated continuous percentile as a float."""
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return float(ordered[lower])
    return float(ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower))


def _minimum(values: List[Any]) -> Any:
    """Return the minimum non-null value or `None` for empty input."""
    return min(values) if values else None


def _maximum(values: List[Any]) -> Any:
    """Return the maximum non-null value or `None` for empty input."""
    return max(values) if values else None


def _total(values: List[Any]) -> Any:
    """Return the numeric sum or `None` for empty input."""
    return sum(values) if values else None


def _average(values: List[Any]) -> Any:
    """Return the arithmetic mean or `None` for empty input."""
    return float(sum(values) / len(values)) if values else None


def _median(values: List[Any]) -> Any:
    """Return the continuous median or `None` for empty input."""
    return _percentile(values, 0.5)
