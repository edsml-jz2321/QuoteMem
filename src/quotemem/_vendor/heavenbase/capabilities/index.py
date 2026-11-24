"""Capability inspection over registered HeavenBase pieces."""

from __future__ import annotations

from heavenbase.utils import Any, Dict, List, Optional

from ..backends.registry import BackendType, backend_aliases
from ..execution.handlers.base import HandlerKey
from ..execution.handlers.registry import HandlerRegistry
from ..strategies import StorageStrategy, strategy_aliases
from ..types import LogicalType, logical_type_aliases
from ..filters.ops import registered_ops
from ..modules import ModuleResolver, module_resolver
from .filters import capability_filter
from .options import BackendOption, CapabilityOption, LogicalTypeOption, StrategyOption


class CapabilityIndex:
    """Discovery index for selectable types, strategies, backends, operations, and capabilities."""

    def __init__(
        self,
        handlers: Optional[HandlerRegistry] = None,
        backends: Any = None,
        *,
        resolver: ModuleResolver | None = None,
    ) -> None:
        """Create a discovery index from process-global or workspace-local registries.

        Args:
            handlers (HandlerRegistry | None): Optional handler registry. When omitted, the index uses the
                seeded process handler registry. Defaults to None.
            backends (Any): Optional live workspace backend registry. When omitted, backend options describe
                registered backend types globally. Defaults to None.

        Returns:
            None: This function initializes the discovery index.
        """
        self.handlers = handlers
        self.backends_registry = backends
        self._module_resolver = resolver

    @property
    def resolver(self) -> ModuleResolver:
        """Return this index's explicit or lazily selected module authority."""
        return module_resolver(self._module_resolver)

    def logical_types(self, *, strategy: Any = None, backend: Any = None, op: Any = None) -> tuple[LogicalTypeOption, ...]:
        """Return logical types, optionally filtered by capability support.

        Args:
            strategy (Any): Optional storage strategy selector accepted by `StorageStrategy.identifier_of`.
                Defaults to None.
            backend (Any): Optional backend type, class, instance, workspace backend name, or iterable of those
                values. Defaults to None.
            op (Any): Optional operation token, alias, or iterable of operation tokens. Defaults to None.

        Returns:
            tuple[LogicalTypeOption, ...]: Matching logical type options in registry order.
        """
        options = self._logical_type_options()
        if strategy is None and backend is None and op is None:
            return options
        identifiers = {row.dtype for row in self.capabilities(strategy=strategy, backend=backend, op=op)}
        return tuple(option for option in options if option.identifier in identifiers)

    def strategies(self, dtype: Any = None, *, backend: Any = None, op: Any = None) -> tuple[StrategyOption, ...]:
        """Return storage strategies, optionally filtered by type, backend, and operation support.

        Args:
            dtype (Any): Optional logical type selector accepted by `get_logical_type`, or an iterable of
                selectors. Defaults to None.
            backend (Any): Optional backend type, class, instance, workspace backend name, or iterable of those
                values. Defaults to None.
            op (Any): Optional operation token, alias, or iterable of operation tokens. Defaults to None.

        Returns:
            tuple[StrategyOption, ...]: Matching storage strategy options in registry order.
        """
        options = self._strategy_options()
        if dtype is None and backend is None and op is None:
            return options
        identifiers = {row.strategy for row in self.capabilities(dtype=dtype, backend=backend, op=op)}
        return tuple(option for option in options if option.identifier in identifiers)

    def backends(self, dtype: Any = None, strategy: Any = None, *, op: Any = None) -> tuple[BackendOption, ...]:
        """Return backend choices, optionally filtered by type, strategy, and operation support.

        Args:
            dtype (Any): Optional logical type selector accepted by `get_logical_type`, or an iterable of
                selectors. Defaults to None.
            strategy (Any): Optional storage strategy selector accepted by `StorageStrategy.identifier_of`, or an
                iterable of selectors. Defaults to None.
            op (Any): Optional operation token, alias, or iterable of operation tokens. Defaults to None.

        Returns:
            tuple[BackendOption, ...]: Matching backend type options for the global index, or workspace backend
            instance options when the index is bound to a workspace.
        """
        options = self._backend_options()
        if dtype is None and strategy is None and op is None:
            return options
        capabilities = self.capabilities(dtype=dtype, strategy=strategy, op=op)
        keys = {(row.backend_identifier, row.backend_name) for row in capabilities}
        return tuple(option for option in options if (option.type, option.name) in keys or (option.type, None) in keys)

    def ops(self, dtype: Any = None, strategy: Any = None, backend: Any = None) -> tuple[str, ...]:
        """Return operation tokens, optionally filtered by capability support.

        Args:
            dtype (Any): Optional logical type selector accepted by `get_logical_type`, or an iterable of
                selectors. Defaults to None.
            strategy (Any): Optional storage strategy selector accepted by `StorageStrategy.identifier_of`, or an
                iterable of selectors. Defaults to None.
            backend (Any): Optional backend type, class, instance, workspace backend name, or iterable of those
                values. Defaults to None.

        Returns:
            tuple[str, ...]: Matching normalized operation tokens, preserving first-seen capability order.
        """
        if dtype is None and strategy is None and backend is None:
            return registered_ops(resolver=self.resolver)
        return tuple(dict.fromkeys(row.op for row in self.capabilities(dtype=dtype, strategy=strategy, backend=backend)))

    def capabilities(self, dtype: Any = None, strategy: Any = None, backend: Any = None, op: Any = None) -> tuple[CapabilityOption, ...]:
        """Return capability rows matching the supplied filters.

        Args:
            dtype (Any): Optional logical type selector accepted by `get_logical_type`, or an iterable of
                selectors. Defaults to None.
            strategy (Any): Optional storage strategy selector accepted by `StorageStrategy.identifier_of`, or an
                iterable of selectors. Defaults to None.
            backend (Any): Optional backend type, class, instance, workspace backend name, or iterable of those
                values. Defaults to None.
            op (Any): Optional operation token, alias, or iterable of operation tokens. Defaults to None.

        Returns:
            tuple[CapabilityOption, ...]: Matching handler capability rows expanded over global backend types or
            workspace backend instances.
        """
        resolver = self.resolver
        filt = capability_filter(
            dtype=dtype,
            strategy=strategy,
            backend=backend,
            op=op,
            backends_registry=self.backends_registry,
            resolver=resolver,
        )
        backend_options = self._backend_options_by_type()
        rows: List[CapabilityOption] = []
        for row in self._handlers().capability_rows():
            if not filt.matches_row(row):
                continue
            for option in backend_options.get(row["backend_identifier"], ()):  # Workspace indexes expand backend instances.
                if not filt.matches_backend(option):
                    continue
                key = HandlerKey(row["dtype"], row["op"], option.type, row["strategy"], resolver=resolver)
                aliases = key.aliases(resolver=resolver)
                rows.append(CapabilityOption(row["dtype"], row["op"], option.type, row["strategy"], option.name, aliases=aliases))
        return tuple(rows)

    def supports(self, dtype: Any, op: Any, backend: Any, strategy: Any) -> bool:
        """Return whether at least one matching capability is registered.

        Args:
            dtype (Any): Logical type selector accepted by `get_logical_type`.
            op (Any): Operation token or alias.
            backend (Any): Backend type, class, instance, or workspace backend name.
            strategy (Any): Storage strategy selector accepted by `StorageStrategy.identifier_of`.

        Returns:
            bool: True when a matching capability row is discoverable.
        """
        return bool(self.capabilities(dtype=dtype, strategy=strategy, backend=backend, op=op))

    def _handlers(self) -> HandlerRegistry:
        from ..execution.handlers.seed import seed_handlers

        if self.handlers is None:
            self.handlers = seed_handlers(resolver=self.resolver)
        return self.handlers

    def _logical_type_options(self) -> tuple[LogicalTypeOption, ...]:
        return tuple(
            LogicalTypeOption(
                dtype.identifier,
                dtype,
                logical_type_aliases(dtype, resolver=self.resolver),
            )
            for dtype in (LogicalType.load(str(row["identifier"]), resolver=self.resolver) for row in self.resolver.inspect("logical_type"))
        )

    def _strategy_options(self) -> tuple[StrategyOption, ...]:
        return tuple(
            StrategyOption(
                strategy.identifier,
                strategy,
                strategy_aliases(strategy, resolver=self.resolver),
            )
            for strategy in (StorageStrategy.load(str(row["identifier"]), resolver=self.resolver) for row in self.resolver.inspect("strategy"))
        )

    def _backend_options(self) -> tuple[BackendOption, ...]:
        options = []
        if self.backends_registry is not None:
            for name, backend in sorted(self.backends_registry.items()):
                backend_identifier = BackendType.identifier_of(backend, resolver=self.resolver)
                option = BackendOption(
                    backend_identifier,
                    backend_aliases(backend_identifier, resolver=self.resolver),
                    name=name,
                    backend=backend,
                )
                options.append(option)
            return tuple(options)
        for backend_identifier in BackendType.identifiers(resolver=self.resolver):
            option = BackendOption(
                backend_identifier,
                backend_aliases(backend_identifier, resolver=self.resolver),
            )
            options.append(option)
        return tuple(options)

    def _backend_options_by_type(self) -> Dict[str, tuple[BackendOption, ...]]:
        out: Dict[str, List[BackendOption]] = {}
        for option in self._backend_options():
            out.setdefault(option.type, []).append(option)
        return {backend_identifier: tuple(options) for backend_identifier, options in out.items()}
