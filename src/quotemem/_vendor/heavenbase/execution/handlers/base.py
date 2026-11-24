"""Shared handler capability and compiler objects."""

from __future__ import annotations

__all__ = ["Handler", "HandlerCompiler", "HandlerKey"]

from dataclasses import InitVar

from heavenbase.utils import Any, Callable, Dict, Mapping, dataclass
from heavenbase.utils.registry import class_identifier

from ...filters.ops import normalize_op, op_aliases
from ...modules import ModuleResolver, module_resolver
from ...strategies.base import StorageStrategy
from ...strategies.registry import strategy_aliases
from ...types import LogicalType, get_logical_type, logical_type_aliases, logical_type_from_any
from ...utils.ids import check_id
from ..contracts import HandlerContext, QueryFragment

HandlerCompiler = Callable[[Any, Any, HandlerContext], QueryFragment]


def _backend_lookup_key(value: Any) -> tuple[str, type[Any] | None]:
    """Return a backend record selector without importing backend packages."""
    expected: type[Any] | None = None
    if isinstance(value, str):
        return check_id(value.strip(), "backend identifier"), expected
    if isinstance(value, type):
        expected = value
        return class_identifier(value, label="backend identifier"), expected
    instance_type = getattr(value, "type", None)
    if isinstance(instance_type, str):
        return check_id(instance_type.strip(), "backend identifier"), type(value)
    raise TypeError(f"Invalid backend {value!r}; expected a registered identifier, class, or live backend")


def backend_row(value: Any, *, resolver: ModuleResolver | None = None, wildcard: bool = False) -> Mapping[str, Any]:
    """Resolve one backend selector through neutral module records.

    Args:
        value (Any): Backend identifier, class, live backend instance, or ``"*"``.
        resolver (ModuleResolver | None): Context-owned module resolver.
        wildcard (bool): Whether ``"*"`` is accepted as a wildcard selector.

    Returns:
        Mapping[str, Any]: Detached backend-type record from the module authority.

    Raises:
        KeyError: If the backend identifier is unknown.
        TypeError: If ``value`` is not a supported backend selector.
        ValueError: If a class differs from its registered backend type.
    """
    selected = module_resolver(resolver)
    if value == "*":
        if wildcard:
            return {"identifier": "*", "aliases": ()}
        raise ValueError("Wildcard backend is not valid for this capability")
    key, expected = _backend_lookup_key(value)
    rows = selected.inspect("backend_type", key)
    if not rows:
        raise KeyError(f"Unknown backend identifier {key!r}")
    row = rows[0]
    if expected is not None:
        resolved = selected.resolve("backend_type", str(row["identifier"]))
        if resolved is not expected:
            raise ValueError(f"Backend class {expected!r} differs from registered type {row['identifier']!r}")
    return row


def backend_identifier(value: Any, *, resolver: ModuleResolver | None = None, wildcard: bool = False) -> str:
    """Return the canonical backend identifier from the module authority.

    Args:
        value (Any): Backend identifier, class, live backend instance, or ``"*"``.
        resolver (ModuleResolver | None): Context-owned module resolver.
        wildcard (bool): Whether ``"*"`` is accepted as a wildcard selector.

    Returns:
        str: Canonical backend type identifier.
    """
    return str(backend_row(value, resolver=resolver, wildcard=wildcard)["identifier"])


def backend_aliases(value: Any, *, resolver: ModuleResolver | None = None) -> tuple[str, ...]:
    """Return aliases from the authoritative backend-type record.

    Args:
        value (Any): Backend identifier, class, or live backend instance.
        resolver (ModuleResolver | None): Context-owned module resolver.

    Returns:
        tuple[str, ...]: Declared backend aliases in registry order.
    """
    return tuple(str(alias) for alias in backend_row(value, resolver=resolver)["aliases"])


@dataclass(frozen=True, order=True)
class HandlerKey:
    """Identify one normalized handler capability.

    Args:
        dtype (Any): Logical type class, instance, identifier, or alias.
        op (str): Operation identifier or alias.
        backend_identifier (str): Backend type identifier or alias.
        strategy (Any): Storage strategy class, instance, identifier, or alias.
        resolver (ModuleResolver | None): Resolver used to normalize the four
            dimensions. It is not retained by the canonical key.
    """

    dtype: Any
    op: str
    backend_identifier: str
    strategy: Any
    resolver: InitVar[ModuleResolver | None] = None

    def __post_init__(self, resolver: ModuleResolver | None) -> None:
        """Normalize every capability dimension.

        Args:
            resolver: Context-owned module resolver used for normalization.

        Returns:
            None: This initializer stores canonical identifiers in place.
        """
        selected = module_resolver(resolver)
        dtype = self.dtype
        if isinstance(dtype, str):
            try:
                dtype = get_logical_type(dtype, resolver=selected).identifier
            except KeyError:
                dtype = LogicalType.identifier_of(dtype, resolver=selected)
        else:
            dtype = LogicalType.identifier_of(dtype, resolver=selected)
        object.__setattr__(self, "dtype", dtype)
        object.__setattr__(self, "op", normalize_op(self.op, resolver=selected))
        object.__setattr__(self, "backend_identifier", backend_identifier(self.backend_identifier, resolver=selected))
        object.__setattr__(self, "strategy", StorageStrategy.identifier_of(self.strategy, resolver=selected))

    @classmethod
    def _from_canonical(cls, dtype: str, op: str, backend_identifier: str, strategy: str) -> "HandlerKey":
        """Build a key from one already-validated module snapshot."""
        key = object.__new__(cls)
        object.__setattr__(key, "dtype", dtype)
        object.__setattr__(key, "op", op)
        object.__setattr__(key, "backend_identifier", backend_identifier)
        object.__setattr__(key, "strategy", strategy)
        return key

    @classmethod
    def load(cls, raw: Any, *, resolver: ModuleResolver | None = None) -> "HandlerKey":
        """Load a handler key from an existing key or four-part sequence.

        Args:
            raw (Any): Existing key or sequence ordered as type, operation,
                backend identifier, and strategy.
            resolver (ModuleResolver | None): Context-owned module resolver.
                The default Context is selected when omitted.

        Returns:
            HandlerKey: Normalized handler capability key.

        Raises:
            TypeError: If ``raw`` is not a key or four-part sequence.
        """
        selected = module_resolver(resolver)
        if isinstance(raw, cls):
            return cls(*raw.to_tuple(), resolver=selected)
        try:
            dtype, op, backend_identifier, strategy = raw
        except (TypeError, ValueError):
            raise TypeError("HandlerKey expects a key or four-part sequence") from None
        return cls(dtype, op, backend_identifier, strategy, resolver=selected)

    def aliases(self, *, resolver: ModuleResolver | None = None) -> Dict[str, tuple[str, ...]]:
        """Return aliases grouped by capability dimension.

        Args:
            resolver: Context-owned module resolver. The default Context is
                selected when omitted.

        Returns:
            Dict[str, tuple[str, ...]]: Aliases for the type, operation,
            backend, and strategy identifiers.
        """
        selected = module_resolver(resolver)
        try:
            dtype = get_logical_type(self.dtype, resolver=selected)
        except KeyError:
            dtype = logical_type_from_any(self.dtype, resolver=selected)
        return {
            "dtype": logical_type_aliases(dtype, resolver=selected),
            "op": op_aliases(self.op, resolver=selected),
            "backend": backend_aliases(self.backend_identifier, resolver=selected),
            "strategy": strategy_aliases(self.strategy, resolver=selected),
        }

    def to_dict(self) -> dict[str, str]:
        """Serialize the capability key.

        Args:
            None.

        Returns:
            dict[str, str]: Canonical capability dimensions.
        """
        return {
            "dtype": self.dtype,
            "op": self.op,
            "backend_identifier": self.backend_identifier,
            "strategy": self.strategy,
        }

    def to_tuple(self) -> tuple[str, str, str, str]:
        """Return the canonical composite registry key.

        Args:
            None.

        Returns:
            tuple[str, str, str, str]: Capability dimensions in registry order.
        """
        return self.dtype, self.op, self.backend_identifier, self.strategy


class Handler:
    """Compile one leaf operation for one normalized capability key."""

    def __init__(
        self,
        dtype: Any,
        op: str,
        backend_identifier: str,
        strategy: Any,
        compile: HandlerCompiler,
        *,
        resolver: ModuleResolver | None = None,
    ) -> None:
        """Create a handler descriptor around a pure compiler.

        Args:
            dtype (Any): Logical type class, instance, identifier, or alias.
            op (str): Operation identifier or alias.
            backend_identifier (str): Backend type identifier or alias.
            strategy (Any): Storage strategy class, instance, identifier, or alias.
            compile (HandlerCompiler): Pure compiler returning a ``QueryFragment``.
            resolver (ModuleResolver | None): Context-owned module resolver.
                The default Context is selected when omitted.

        Returns:
            None: This initializer stores the normalized key and compiler.

        Raises:
            TypeError: If ``compile`` is not callable.
        """
        if not callable(compile):
            raise TypeError("Handler compile must be callable")
        self._module_resolver = module_resolver(resolver)
        self.key = HandlerKey(dtype, op, backend_identifier, strategy, resolver=self._module_resolver)
        self._compile = compile

    @classmethod
    def _from_canonical(
        cls,
        dtype: str,
        op: str,
        backend_identifier: str,
        strategy: str,
        compile: HandlerCompiler,
        *,
        resolver: ModuleResolver,
    ) -> "Handler":
        """Build a handler from identities validated by one seed snapshot."""
        if not callable(compile):
            raise TypeError("Handler compile must be callable")
        handler = object.__new__(cls)
        handler._module_resolver = resolver
        handler.key = HandlerKey._from_canonical(dtype, op, backend_identifier, strategy)
        handler._compile = compile
        return handler

    def compile(self, field_schema: Any, value: Any, ctx: HandlerContext) -> QueryFragment:
        """Compile one field operation without performing IO.

        Args:
            field_schema (Any): Routed field schema.
            value (Any): Normalized operation value.
            ctx (HandlerContext): Selected backend and strategy context.

        Returns:
            QueryFragment: Backend-neutral or provider-native query fragment.

        Raises:
            TypeError: If the compiler does not return a ``QueryFragment``.
        """
        fragment = self._compile(field_schema, value, ctx)
        if not isinstance(fragment, QueryFragment):
            raise TypeError(f"Handler compiler must return QueryFragment, got {type(fragment).__name__}")
        return fragment

    def __call__(self, field_schema: Any, value: Any, ctx: HandlerContext) -> QueryFragment:
        """Compile one field operation.

        Args:
            field_schema (Any): Routed field schema.
            value (Any): Normalized operation value.
            ctx (HandlerContext): Selected backend and strategy context.

        Returns:
            QueryFragment: Compiled query fragment.
        """
        return self.compile(field_schema, value, ctx)

    def register(self, registry: Any, *, replace: bool = False) -> "Handler":
        """Register this handler in one workspace registry.

        Args:
            registry (Any): ``HandlerRegistry`` receiving this descriptor.
            replace (bool): Whether to replace an existing capability. Defaults
                to ``False``.

        Returns:
            Handler: This registered descriptor.
        """
        return registry.register(self, replace=replace)
