"""Filter normalization for capability inspection."""

from __future__ import annotations

from heavenbase.utils import Any, Mapping, Optional, dataclass

from ..backends.registry import BackendType
from ..strategies import StorageStrategy
from ..types import get_logical_type
from ..filters.ops import normalize_op
from ..modules import ModuleResolver, module_resolver
from .options import BackendOption


@dataclass(frozen=True)
class BackendFilter:
    """Normalized backend type and workspace-name selectors for discovery queries.

    Args:
        identifiers (tuple[str, ...]): Backend type identifiers accepted by the filter.
        names (tuple[str, ...]): Workspace backend instance names accepted by the filter.
    """

    identifiers: tuple[str, ...]
    names: tuple[str, ...]

    def matches(self, option: BackendOption) -> bool:
        """Return whether a backend option satisfies this filter.

        Args:
            option (BackendOption): Backend type or workspace backend instance option.

        Returns:
            bool: True when the option's backend type or workspace backend name matches.
        """
        return option.type in self.identifiers or (option.name is not None and option.name in self.names)


@dataclass(frozen=True)
class CapabilityFilter:
    """Normalized capability selectors used by `CapabilityIndex`.

    Args:
        dtype_identifiers (tuple[str, ...] | None): Logical type identifiers accepted by the filter.
        strategy_identifiers (tuple[str, ...] | None): Storage strategy identifiers accepted by the filter.
        backend (BackendFilter | None): Optional backend type/name filter.
        ops (tuple[str, ...] | None): Normalized operation tokens accepted by the filter.
    """

    dtype_identifiers: Optional[tuple[str, ...]] = None
    strategy_identifiers: Optional[tuple[str, ...]] = None
    backend: Optional[BackendFilter] = None
    ops: Optional[tuple[str, ...]] = None

    def matches_row(self, row: Mapping[str, Any]) -> bool:
        """Return whether a handler capability row matches scalar filters.

        Args:
            row (Mapping[str, Any]): Handler capability row with `dtype`, `op`, `backend_identifier`,
                and `strategy` keys.

        Returns:
            bool: True when the row matches dtype, strategy, and op filters.
        """
        if self.dtype_identifiers is not None and row["dtype"] not in self.dtype_identifiers:
            return False
        if self.strategy_identifiers is not None and row["strategy"] not in self.strategy_identifiers:
            return False
        if self.ops is not None and row["op"] not in self.ops:
            return False
        return True

    def matches_backend(self, option: BackendOption) -> bool:
        """Return whether a backend option matches the backend filter.

        Args:
            option (BackendOption): Backend type or workspace backend instance option.

        Returns:
            bool: True when no backend filter is active or the backend option matches it.
        """
        return self.backend is None or self.backend.matches(option)


def capability_filter(
    dtype: Any = None,
    strategy: Any = None,
    backend: Any = None,
    op: Any = None,
    *,
    backends_registry: Any = None,
    resolver: ModuleResolver | None = None,
) -> CapabilityFilter:
    """Normalize caller-authored discovery filters.

    Args:
        dtype (Any): Logical type class, instance, identifier, alias, or iterable of those values.
        strategy (Any): Storage strategy class, instance, identifier, alias, or iterable of those values.
        backend (Any): Backend type, backend class, backend instance, workspace backend name, or iterable of those values.
        op (Any): Operation token, alias, or iterable of operation tokens.
        backends_registry (Any): Optional live workspace backend registry. When present, backend-name filters can match
            configured backend instances.

    Returns:
        CapabilityFilter: Normalized immutable filter object.
    """
    selected = module_resolver(resolver)
    return CapabilityFilter(
        dtype_identifiers=_dtype_identifiers(dtype, selected),
        strategy_identifiers=_strategy_identifiers(strategy, selected),
        backend=_backend_filter(backend, backends_registry=backends_registry, resolver=selected),
        ops=_op_tokens(op, selected),
    )


def _dtype_identifiers(dtype: Any, resolver: ModuleResolver) -> Optional[tuple[str, ...]]:
    items = _items(dtype)
    if not items:
        return None
    return tuple(
        item.identifier if isinstance(item, type) else get_logical_type(item, resolver=resolver).identifier if isinstance(item, str) else type(item).identifier
        for item in items
    )


def _strategy_identifiers(strategy: Any, resolver: ModuleResolver) -> Optional[tuple[str, ...]]:
    items = _items(strategy)
    if not items:
        return None
    return tuple(StorageStrategy.identifier_of(item, resolver=resolver) for item in items)


def _op_tokens(op: Any, resolver: ModuleResolver) -> Optional[tuple[str, ...]]:
    items = _items(op)
    if not items:
        return None
    return tuple(normalize_op(item, resolver=resolver) for item in items)


def _backend_filter(
    backend: Any,
    *,
    backends_registry: Any = None,
    resolver: ModuleResolver,
) -> Optional[BackendFilter]:
    items = _items(backend)
    if not items:
        return None
    if backends_registry is None:
        BackendType.identifiers(resolver=resolver)
    identifiers = []
    names = []
    for item in items:
        if hasattr(item, "type"):
            identifiers.append(BackendType.identifier_of(item, resolver=resolver))
            if hasattr(item, "name"):
                names.append(str(item.name))
            continue
        try:
            identifiers.append(BackendType.identifier_of(item, resolver=resolver))
        except KeyError:
            names.append(str(item))
    return BackendFilter(tuple(dict.fromkeys(identifiers)), tuple(dict.fromkeys(names)))


def _items(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple, set)):
        return tuple(value)
    return (value,)
