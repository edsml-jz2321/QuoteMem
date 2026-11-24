"""Portable Python-scan predicate semantics."""

from __future__ import annotations

__all__ = ["Predicate"]

import fnmatch
import inspect
import re

from heavenbase.utils import Any, Callable, Mapping, dataclass
from heavenbase.utils.config import ConfigManager
from heavenbase.utils.ids import check_id
from heavenbase.utils.text_match import contained_in_matches

from ...filters.ops import normalize_op
from ...modules import ModuleResolver, ModuleService, module_resolver, python_executable_ref
from ...modules.base import ResolverCache
from ...types.json import Json
from ._text_patterns import LikePattern, RegexSearch

_NESTED_QUANTIFIER_RE = re.compile(r"\((?:[^()\\]|\\.)*[+*](?:[^()\\]|\\.)*\)\s*[+*?{]")

_Prepare = Callable[..., Any]
_Matcher = Callable[[Any, Any], bool]
_HyperGPredicates = dict[str, dict[str, tuple["Predicate", Any]]]
_JsonGet = tuple[tuple[str, ...], str, Any]
_Regex = tuple[RegexSearch, int]
_JSON_TYPE = Json()
_PREDICATE_CACHE = ResolverCache["Predicate"]()


def _identity(value: Any, resolver: ModuleResolver, config: Any) -> Any:
    del resolver, config
    return value


@dataclass(frozen=True)
class Predicate:
    """Define portable Python-scan semantics for one filter operation.

    A predicate prepares the normalized query value once, then tests stored
    values without knowing about schemas, backends, handlers, or result rows.

    Args:
        identifier (str): Canonical operation identifier registered through
            `hb.ext.register_op(...)`.
        matcher (Callable[[Any, Any], bool]): Function receiving a stored value
            and the prepared query value.
        preparer (Callable[[Any], Any]): Optional one-time query-value
            preparation function. Defaults to returning the value unchanged.
    """

    identifier: str
    matcher: _Matcher
    preparer: _Prepare = _identity

    def __post_init__(self) -> None:
        """Validate and normalize this predicate definition.

        Args:
            None.

        Returns:
            None: This method normalizes the identifier in place.

        Raises:
            TypeError: If `matcher` or `preparer` is not callable.
        """
        object.__setattr__(self, "identifier", check_id(self.identifier, "filter predicate identifier"))
        if not callable(self.matcher):
            raise TypeError("Predicate matcher must be callable")
        if not callable(self.preparer):
            raise TypeError("Predicate preparer must be callable")

    def register(
        self,
        *,
        dtype: Any = None,
        replace: bool = False,
        resolver: ModuleResolver | None = None,
    ) -> "Predicate":
        """Register this predicate as the portable scan behavior for its operation.

        Args:
            dtype (Any): Optional logical type class, instance, or identifier
                that scopes this predicate ahead of the generic fallback.
            replace (bool): Whether to replace an existing predicate for the
                same operation.
            resolver (ModuleResolver | None): Writable Context-owned module
                resolver. The default Context is selected when omitted.

        Returns:
            Predicate: This registered predicate.

        Raises:
            TypeError: If the resolver is not a mutable `ModuleService`.
            ValueError: If the operation already has a predicate and `replace`
                is `False`, another installation owns the record, or a callable
                cannot be represented by a durable import target.
        """
        selected = module_resolver(resolver)
        if not isinstance(selected, ModuleService):
            raise TypeError("Predicate registration requires a mutable ModuleService resolver")
        dtype_identifier = None if dtype is None else _dtype_identifier(dtype)
        record_identifier = _record_identifier(self.identifier, dtype_identifier)
        coordinate = f"programmatic/predicate/{record_identifier}"
        found = selected.inspect("predicate", record_identifier)
        if found:
            existing = _predicate_from_row(found[0], selected)
            if existing == self and replace:
                return existing
            if not replace:
                raise ValueError(f"Predicate {self.identifier!r} is already registered for {_scope_label(dtype_identifier)}")
            owner = found[0]["record"]["meta"]["installation"]["coordinate"]
            if owner != coordinate:
                raise ValueError(f"Predicate {self.identifier!r} belongs to {owner!r} and cannot be replaced")
        matcher_ref = python_executable_ref(self.matcher, label=f"Predicate {self.identifier!r} matcher")
        preparer_ref = python_executable_ref(self.preparer, label=f"Predicate {self.identifier!r} preparer")
        lines = str(getattr(self.matcher, "__doc__", "") or "").strip().splitlines()
        selected.register(
            "predicate",
            record_identifier,
            {
                "name": self.identifier.replace("_", " ").title(),
                "desc": lines[0].strip() if lines else "Registered portable scan predicate.",
                "source": "inline",
                "target": "definition",
                "active": True,
                **({"dtype": dtype_identifier} if dtype_identifier is not None else {}),
                "meta": {
                    "schema_version": 1,
                    "tags": {"portable": True},
                    "operation": self.identifier,
                    "dependencies": [f"operation:{self.identifier}"],
                    "definition": {"matcher": matcher_ref, "preparer": preparer_ref},
                },
            },
            groups=() if dtype_identifier is None else (("dtype", dtype_identifier),),
            coordinate=coordinate,
        )
        return _predicate_from_row(selected.inspect("predicate", record_identifier)[0], selected)

    @classmethod
    def load(
        cls,
        op: str,
        dtype: Any = None,
        *,
        resolver: ModuleResolver | None = None,
    ) -> "Predicate":
        """Load the portable predicate for an operation.

        Args:
            op (str): Canonical operation identifier or registered alias.
            dtype (Any): Optional logical type class, instance, or identifier.
                A matching type-scoped predicate takes precedence.
            resolver (ModuleResolver | None): Context-owned module resolver.
                The default Context is selected when omitted.

        Returns:
            Predicate: Registered portable predicate.

        Raises:
            LookupError: If the operation is not registered.
            TypeError: If the selected predicate record is malformed.
            ValueError: If the operation has no portable predicate.
        """
        selected = module_resolver(resolver)
        identifier = normalize_op(op, resolver=selected)
        if dtype is not None:
            typed = selected.inspect("predicate", _record_identifier(identifier, _dtype_identifier(dtype)))
            if typed:
                return _predicate_from_row(typed[0], selected)
        rows = selected.inspect("predicate", identifier)
        if rows:
            return _predicate_from_row(rows[0], selected)
        raise ValueError(f"Operation {identifier!r} has no portable predicate")

    @classmethod
    def identifiers(
        cls,
        dtype: Any = None,
        *,
        resolver: ModuleResolver | None = None,
    ) -> tuple[str, ...]:
        """Return operations with registered portable predicates.

        Args:
            dtype (Any): Optional logical type class, instance, or identifier
                used to include matching type-scoped predicates. When omitted,
                only generic predicates are returned.
            resolver (ModuleResolver | None): Context-owned module resolver.
                The default Context is selected when omitted.

        Returns:
            tuple[str, ...]: Operation identifiers in registration order.
        """
        selected = module_resolver(resolver)
        dtype_identifier = None if dtype is None else _dtype_identifier(dtype)
        generic: list[str] = []
        typed: list[str] = []
        for row in selected.inspect("predicate"):
            operation = _row_operation(row)
            scope = _row_dtype(row)
            if scope is None:
                generic.append(operation)
            elif scope == dtype_identifier:
                typed.append(operation)
        return tuple(dict.fromkeys((*generic, *typed)))

    def compile(
        self,
        value: Any,
        *,
        resolver: ModuleResolver | None = None,
        config: Any | None = None,
    ) -> Any:
        """Prepare one normalized query value for repeated row scans.

        Args:
            value (Any): Logical-type-normalized query value.
            resolver (ModuleResolver | None): Context-owned module resolver.
                The default Context is selected when omitted.
            config (Any | None): Config authority supplying predicate safety
                limits. When omitted, the config of the Context that owns
                ``resolver`` is used, or the default Context's config when
                both are omitted.

        Returns:
            Any: Prepared value consumed by `match`.

        Raises:
            TypeError: If the operation requires a different value shape.
            ValueError: If a pattern is malformed or exceeds safety limits.
        """
        selected = module_resolver(resolver)
        manager = config
        if resolver is None and manager is not None:
            from heavenbase.context.runtime import _config_context

            owner = _config_context(manager)
            if owner is None:
                raise TypeError("Predicate config-only authority is unbound; supply its paired resolver")
            selected = owner.modules()
        elif resolver is not None and manager is None:
            manager = getattr(selected, "authority_config", None)
            if manager is None:
                from heavenbase.context.runtime import _default_context_config_of

                manager = _default_context_config_of(selected)
                if manager is None:
                    raise TypeError("Predicate resolver-only authority has no bound config; supply its paired config")
        if manager is None:
            # Neither was given: ``selected`` is the default Context's modules,
            # so read that Context's config, which is not CM_HVNB after
            # set_default_context installs a Context with its own config.
            from heavenbase.context.authority import omitted_config

            manager = omitted_config(selected)
        owner = getattr(manager, "_context", None)
        if owner is not None and owner.modules() is not selected:
            raise ValueError("Predicate resolver and config belong to different Contexts")
        bound_config = getattr(selected, "authority_config", None)
        if isinstance(manager, ConfigManager) and bound_config is not None and bound_config is not manager:
            raise ValueError("Predicate resolver and config belong to different Contexts")
        if not callable(getattr(manager, "get", None)):
            raise TypeError("Predicate config must provide get(key, default=...)")
        try:
            inspect.signature(self.preparer).bind(value, selected, manager)
        except (TypeError, ValueError):
            return self.preparer(value)
        return self.preparer(value, selected, manager)

    def match(self, left: Any, prepared: Any) -> bool:
        """Return whether one stored value satisfies a prepared query value.

        Args:
            left (Any): Stored value read from a row.
            prepared (Any): Query value returned by `compile`.

        Returns:
            bool: `True` when the stored value matches.
        """
        return bool(self.matcher(left, prepared))


def _dtype_identifier(dtype: Any) -> str:
    """Return a logical type identifier without importing the type registry."""
    if isinstance(dtype, str):
        identifier = dtype
    else:
        target = dtype if isinstance(dtype, type) else type(dtype)
        identifier = getattr(target, "identifier", None)
    if identifier is None:
        raise TypeError(f"Expected a logical type class, instance, or identifier; got {dtype!r}")
    return check_id(str(identifier), "logical type identifier")


def _record_identifier(operation: str, dtype: str | None) -> str:
    """Return the collision-free module identifier for one predicate scope."""
    op = check_id(str(operation), "filter predicate identifier")
    return op if dtype is None else f"for-{len(dtype)}-{check_id(dtype, 'logical type identifier')}-{op}"


def _scope_label(dtype: str | None) -> str:
    """Return a concise predicate-scope label for diagnostics."""
    return "the generic scope" if dtype is None else f"logical type {dtype!r}"


def _row_dtype(row: Mapping[str, Any]) -> str | None:
    """Return the optional logical-type scope declared by an inert record."""
    record = row.get("record")
    if not isinstance(record, Mapping):
        raise TypeError(f"Predicate record {row.get('identifier')!r} has no record mapping")
    dtype = record.get("dtype")
    return None if dtype is None else _dtype_identifier(dtype)


def _row_operation(row: Mapping[str, Any]) -> str:
    """Return the operation declared by an inert predicate record."""
    record = row.get("record")
    meta = record.get("meta") if isinstance(record, Mapping) else None
    operation = meta.get("operation") if isinstance(meta, Mapping) else None
    if not isinstance(operation, str):
        raise TypeError(f"Predicate record {row.get('identifier')!r} must declare meta.operation")
    return check_id(operation, "filter predicate identifier")


def _predicate_from_row(row: Mapping[str, Any], resolver: ModuleResolver) -> Predicate:
    """Materialize and cache one resolver-owned predicate record."""

    def build(current_row: Mapping[str, Any]) -> Predicate:
        operation = _row_operation(current_row)
        dtype = _row_dtype(current_row)
        record_identifier = str(current_row["identifier"])
        expected = _record_identifier(operation, dtype)
        if record_identifier != expected:
            raise ValueError(f"Predicate record {record_identifier!r} must use canonical identifier {expected!r}")
        raw = resolver.resolve("predicate", record_identifier)
        if isinstance(raw, Predicate):
            predicate = raw
        elif isinstance(raw, Mapping):
            if set(raw) != {"matcher", "preparer"}:
                raise TypeError(f"Predicate {operation!r} inline definition must contain matcher and preparer locators")
            matcher = raw.get("matcher")
            preparer = raw.get("preparer")
            if not isinstance(matcher, Mapping) or not isinstance(preparer, Mapping):
                raise TypeError(f"Predicate {operation!r} matcher and preparer must be executable references")
            predicate = Predicate(
                operation,
                resolver.resolve_executable(matcher, owner=current_row),
                resolver.resolve_executable(preparer, owner=current_row),
            )
        else:
            raise TypeError(f"Predicate {operation!r} must resolve to a Predicate or inline executable-reference mapping")
        if predicate.identifier != operation:
            raise ValueError(f"Predicate record operation {operation!r} resolved implementation {predicate.identifier!r}")
        return predicate

    return _PREDICATE_CACHE.get(resolver, row, build)


def _max_len(config: Any, key: str, default: int) -> int:
    return int(config.get(f"heavenbase.{key}", default=default))


def _checked_pattern(value: Any, kind: str, config: Any) -> str:
    text = str(value)
    max_len = _max_len(config, f"query.{kind}.max_chars", 512)
    if len(text) > max_len:
        raise ValueError(f"{kind} pattern exceeds {max_len} characters")
    return text


def _like_pattern(value: Any, resolver: ModuleResolver, config: Any, *, ignore_case: bool = False) -> LikePattern:
    del resolver
    # LIKE matches the whole value: `%` and `_` span line breaks, and a trailing newline is not ignored.
    return LikePattern.compile(_checked_pattern(value, "like", config), ignore_case=ignore_case)


def _regex_pattern(value: Any, resolver: ModuleResolver, config: Any) -> _Regex:
    del resolver
    text = _checked_pattern(value, "regex", config)
    guarded = not bool(config.get("heavenbase.query.regex.allow_nested_quantifiers", default=False))
    if guarded and _NESTED_QUANTIFIER_RE.search(text):
        raise ValueError(f"Unsafe regex pattern {text!r}: nested quantifiers are disabled")
    try:
        pattern = re.compile(text)
    except (re.error, RecursionError) as exc:
        raise ValueError(f"Invalid regex pattern {text!r}: {exc}") from exc
    max_text_chars = _max_len(config, "query.regex.max_text_chars", 8192)
    # A pattern whose backtracking could outgrow the text cap runs on a linear-time matcher when it is regular.
    matcher, reason = RegexSearch.plan(pattern, max_text_chars=max_text_chars)
    if guarded and reason is not None:
        raise ValueError(f"Unsafe regex pattern {text!r}: {reason}")
    return matcher, max_text_chars


def _regex_text(value: Any, max_len: int) -> str:
    text = str(value or "")
    if len(text) > max_len:
        raise ValueError(f"regex target text exceeds {max_len} characters")
    return text


def _in_values(value: Any, resolver: ModuleResolver, config: Any) -> Any:
    del resolver, config
    if not isinstance(value, (list, tuple, set)):
        raise TypeError("in filter expects a list, tuple, or set")
    return value


def _json_kind(value: Any) -> str | None:
    """Return the strict JSON value kind used by portable comparisons."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, (list, tuple)):
        return "array"
    if isinstance(value, Mapping):
        return "object"
    return None


def _json_equal(left: Any, right: Any) -> bool:
    """Compare JSON values without Python's Boolean/integer equivalence.

    Tagged storage restores tuples as tuples, while strict query operands
    canonicalize them to JSON arrays. Treat both Python sequence forms as the
    same JSON array kind without admitting other extended storage values.
    """
    kind = _json_kind(left)
    if kind is None or kind != _json_kind(right):
        return False
    if kind == "array":
        return len(left) == len(right) and all(_json_equal(a, b) for a, b in zip(left, right))
    if kind == "object":
        return set(left) == set(right) and all(_json_equal(left[key], right[key]) for key in left)
    return bool(left == right)


def _json_in(left: Any, values: Any) -> bool:
    """Return whether one strict JSON value equals any query candidate."""
    return any(_json_equal(left, value) for value in values)


def _prepare_json_get(value: Any, resolver: ModuleResolver, config: Any) -> _JsonGet:
    """Compile a normalized Json ``get`` payload once per scan."""
    if not isinstance(value, Mapping):
        raise TypeError("Json get predicates expect a normalized mapping")
    path = tuple(value["path"])
    op = normalize_op(str(value["op"]), resolver=resolver)
    if op not in _JSON_GET_OPS:
        raise ValueError(f"Json get predicates do not support {op!r}")
    inner_value = _in_values(value["value"], resolver, config) if op == "in" else value["value"]
    return path, op, inner_value


def _json_leaf_match(left: Any, op: str, value: Any) -> bool:
    """Match one non-null Json path leaf with strict type compatibility."""
    if left is None:
        return False
    if op == "in":
        return _json_in(left, value)
    kind = _json_kind(left)
    if kind is None or kind != _json_kind(value):
        return False
    if op == "eq":
        return _json_equal(left, value)
    if op == "ne":
        return not _json_equal(left, value)
    if op in ("lt", "lte", "gt", "gte"):
        if kind not in ("number", "string"):
            return False
        if op == "lt":
            return bool(left < value)
        if op == "lte":
            return bool(left <= value)
        if op == "gt":
            return bool(left > value)
        return bool(left >= value)
    if op == "contains":
        return kind == "string" and value in left
    return False


def _json_get_match(left: Any, prepared: _JsonGet) -> bool:
    """Walk a Json object path and match its leaf value."""
    path, op, value = prepared
    return _json_leaf_match(_JSON_TYPE.path_value(left, path), op, value)


def _contained_in(left: Any, value: Any) -> bool:
    return bool(contained_in_matches("", left, value))


def _equal(left: Any, value: Any) -> bool:
    """Return Python equality for one stored and prepared value."""
    return bool(left == value)


def _not_equal(left: Any, value: Any) -> bool:
    """Return Python inequality for one stored and prepared value."""
    return bool(left != value)


def _less_than(left: Any, value: Any) -> bool:
    """Return strict ordering while treating null stored values as false."""
    return left is not None and bool(left < value)


def _less_than_or_equal(left: Any, value: Any) -> bool:
    """Return inclusive ordering while treating null stored values as false."""
    return left is not None and bool(left <= value)


def _greater_than(left: Any, value: Any) -> bool:
    """Return strict reverse ordering while treating null values as false."""
    return left is not None and bool(left > value)


def _greater_than_or_equal(left: Any, value: Any) -> bool:
    """Return inclusive reverse ordering while treating null values as false."""
    return left is not None and bool(left >= value)


def _member_of(left: Any, values: Any) -> bool:
    """Return whether one stored value belongs to prepared query values."""
    return bool(left in values)


def _text_match(left: Any, value: Any) -> bool:
    """Return case-insensitive substring matching over text coercions."""
    return str(value).lower() in str(left or "").lower()


def _like_match(left: Any, pattern: LikePattern) -> bool:
    """Return whether a prepared case-sensitive LIKE pattern matches."""
    return pattern.match(str(left or ""))


def _ilike_match(left: Any, pattern: LikePattern) -> bool:
    """Return whether a prepared case-insensitive LIKE pattern matches."""
    return pattern.match(str(left or ""))


def _wildcard_match(left: Any, value: str) -> bool:
    """Return case-sensitive shell-style wildcard matching."""
    return fnmatch.fnmatchcase(str(left or ""), value)


def _contains(left: Any, value: Any) -> bool:
    """Return whether coerced stored text contains coerced query text."""
    return str(value) in str(left or "")


def _starts_with(left: Any, value: Any) -> bool:
    """Return whether non-null stored text starts with the query text."""
    return left is not None and str(left).startswith(str(value))


def _ends_with(left: Any, value: Any) -> bool:
    """Return whether non-null stored text ends with the query text."""
    return left is not None and str(left).endswith(str(value))


def _array_contains(left: Any, value: Any) -> bool:
    """Return whether a stored list contains one query value."""
    return bool(value in left) if isinstance(left, list) else False


def _json_not_equal(left: Any, value: Any) -> bool:
    """Return strict JSON inequality while treating null stored values as false."""
    return left is not None and not _json_equal(left, value)


def _compile_hyperg_predicates(value: Mapping[str, Any], resolver: ModuleResolver, config: Any) -> _HyperGPredicates:
    return {
        field_name: {
            op: (predicate := Predicate.load(op, resolver=resolver), predicate.compile(predicate_value, resolver=resolver, config=config))
            for op, predicate_value in predicates.items()
        }
        for field_name, predicates in value.items()
    }


def _regex_match(left: Any, prepared: _Regex) -> bool:
    matcher, max_len = prepared
    return matcher.search(_regex_text(left, max_len))


def _wildcard_pattern(value: Any, resolver: ModuleResolver, config: Any) -> str:
    del resolver
    return _checked_pattern(value, "wildcard", config)


def _ilike_pattern(value: Any, resolver: ModuleResolver, config: Any) -> LikePattern:
    return _like_pattern(value, resolver, config, ignore_case=True)


def _hyperg_match(left: Any, predicates: dict[str, tuple[Predicate, Any]]) -> bool:
    for predicate, prepared in predicates.values():
        try:
            if not predicate.match(left, prepared):
                return False
        except TypeError:
            return False
    return True


def _hyperg_row_match(item: Mapping[str, Any], predicates: _HyperGPredicates) -> bool:
    return all(_hyperg_match(item.get(field_name), operations) for field_name, operations in predicates.items())


def _hyperg_exists(value: Any, predicates: _HyperGPredicates) -> bool:
    return isinstance(value, list) and any(isinstance(item, Mapping) and _hyperg_row_match(item, predicates) for item in value)


def _hyperg_all(value: Any, where: _HyperGPredicates, then: _HyperGPredicates) -> bool:
    if not isinstance(value, list):
        return False
    for item in value:
        if not isinstance(item, Mapping):
            if not where:
                return False
            continue
        if _hyperg_row_match(item, where) and not _hyperg_row_match(item, then):
            return False
    return True


_JSON_GET_OPS = ("eq", "ne", "in", "lt", "lte", "gt", "gte", "contains")
