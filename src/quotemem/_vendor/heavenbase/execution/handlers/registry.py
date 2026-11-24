"""Workspace-owned handler and reason registry."""

from __future__ import annotations

__all__ = ["HandlerRegistry"]

import inspect

from heavenbase.utils import Any, Callable, List

from heavenbase.utils.registry import Registry
from ...modules import ModuleResolver, module_resolver
from .aggregate import AggregateCompiler
from .base import Handler, HandlerKey, backend_identifier
from .reasons import DEFAULT_FALLBACK_TEMPLATE, DEFAULT_UNSUPPORTED_TEMPLATE, ReasonRegistry


def _copy_handler(handler: Handler) -> Handler:
    """Return a descriptor with the same attributes; a registry copies tens of thousands of them."""
    copied = object.__new__(type(handler))
    copied.__dict__.update(handler.__dict__)
    return copied


class HandlerRegistry:
    """Compile-operation registry for one workspace.

    The registry maps `(logical type, operation, backend type, strategy)` keys to
    pure handler callables. Handlers compile a routed leaf operation into a
    `QueryFragment`; they do not perform IO or mutate backend state.
    """

    def __init__(self, *, resolver: ModuleResolver | None = None) -> None:
        """Create an empty handler registry.

        Args:
            resolver: Context-owned module resolver. The default Context is
                selected when omitted.

        Returns:
            None: This initializer stores an empty composite key registry.
        """
        self._module_resolver = module_resolver(resolver)
        self._handlers = Registry(
            "handlers",
            normalize=lambda raw: raw if isinstance(raw, HandlerKey) else HandlerKey.load(raw, resolver=self.module_resolver),
            value_policy="process_identity",
        )
        self._aggregate_compilers = Registry(
            "aggregate-compilers",
            normalize=lambda raw: backend_identifier(raw, resolver=self.module_resolver),
            normalize_alias=lambda raw: backend_identifier(raw, resolver=self.module_resolver),
            value_policy="process_identity",
        )
        self._fallback_reasons = ReasonRegistry(DEFAULT_FALLBACK_TEMPLATE, resolver=self.module_resolver)
        self._unsupported_reasons = ReasonRegistry(DEFAULT_UNSUPPORTED_TEMPLATE, resolver=self.module_resolver)

    def _copy(self) -> "HandlerRegistry":
        """Return an independent registry that starts with this one's registrations.

        Each registry has handler descriptors of its own, so neither a later
        registration nor a change to a descriptor reaches the other one.
        """
        copied = HandlerRegistry(resolver=self.module_resolver)
        copied._handlers._start_from(self._handlers, copy_value=_copy_handler)
        copied._aggregate_compilers._start_from(self._aggregate_compilers)
        copied._fallback_reasons = self._fallback_reasons._copy()
        copied._unsupported_reasons = self._unsupported_reasons._copy()
        return copied

    @property
    def module_resolver(self) -> ModuleResolver:
        """Return this registry's immutable module authority binding."""
        return self._module_resolver

    def register(self, handler: Handler, *, replace: bool = False) -> Handler:
        """Register one handler descriptor.

        Args:
            handler (Handler): Capability and pure compiler to register.
            replace (bool): Whether to replace an existing capability. Defaults
                to ``False``.

        Returns:
            Handler: Registered descriptor.

        Raises:
            TypeError: If ``handler`` is not a ``Handler``.
        """
        if not isinstance(handler, Handler):
            raise TypeError("HandlerRegistry.register expects a Handler")
        if handler._module_resolver is not self.module_resolver:
            handler.key = HandlerKey.load(handler.key, resolver=self.module_resolver)
            handler._module_resolver = self.module_resolver
        return self._handlers.update(handler.key, handler) if replace else self._handlers.create(handler.key, handler)

    def _mutate_handlers(self, change: Callable[["HandlerRegistry"], Any]) -> Any:
        """Apply handler registrations through one private Registry draft."""
        handlers = self._handlers

        def apply(draft: Registry[HandlerKey, Handler]) -> Any:
            self._handlers = draft
            try:
                return change(self)
            finally:
                self._handlers = handlers

        return handlers.mutate(apply)

    def get(self, dtype: Any, op: str, backend_identifier: str, strategy: Any) -> Handler:
        """Return a compiler for one handler capability key.

        Args:
            dtype (Any): Logical type class, instance, identifier, or alias.
            op (str): Operation identifier or alias.
            backend_identifier (str): Backend type identifier or alias.
            strategy (Any): Storage strategy class, instance, identifier, or alias.

        Returns:
            Handler: Registered compiler callable.

        Raises:
            LookupError: If no compiler is registered for the normalized key.
        """
        key = HandlerKey(dtype, op, backend_identifier, strategy, resolver=self.module_resolver)
        handler = self._handlers.get(key)
        if handler is None:
            raise LookupError(f"No handler for type={key.dtype} op={key.op} " f"backend_identifier={key.backend_identifier} strategy={key.strategy}")
        return handler

    def capabilities(self) -> tuple[HandlerKey, ...]:
        """Return registered capability keys in deterministic order.

        Args:
            None.

        Returns:
            tuple[HandlerKey, ...]: Sorted normalized capability keys.
        """
        return tuple(sorted(self._handlers.list()))

    def capability_rows(self) -> List[dict]:
        """Return registered handler capabilities as serializable rows.

        Args:
            None.

        Returns:
            List[dict]: Capability rows with `dtype`, `op`, `backend_identifier`,
            and `strategy` fields.
        """
        return [key.to_dict() for key in self.capabilities()]

    def register_aggregate_compiler(self, compiler: AggregateCompiler, *, replace: bool = False) -> AggregateCompiler:
        """Register one backend aggregate compiler descriptor.

        Args:
            compiler: Handler-layer native aggregate compiler.
            replace: Whether to replace the current compiler for the backend type.

        Returns:
            AggregateCompiler: The registered descriptor.
        """
        if not isinstance(compiler, AggregateCompiler):
            raise TypeError("register_aggregate_compiler expects AggregateCompiler")
        compiler = AggregateCompiler(
            compiler.backend_identifier,
            compiler.operations,
            compiler.compile,
            resolver=self.module_resolver,
        )
        method = self._aggregate_compilers.update if replace else self._aggregate_compilers.create
        return method(compiler.backend_identifier, compiler)

    def aggregate_compiler(self, backend_identifier: str) -> AggregateCompiler | None:
        """Return the registered compiler for one backend type, if any.

        Args:
            backend_identifier (str): Backend type identifier or alias.

        Returns:
            AggregateCompiler | None: Registered aggregate compiler, or ``None``.
        """
        return self._aggregate_compilers.get(backend_identifier)

    def aggregate_capability_rows(self) -> List[dict]:
        """Return native aggregate compiler capabilities as metadata rows.

        Args:
            None.

        Returns:
            List[dict]: Serializable aggregate compiler capability rows.
        """
        return [compiler.to_dict() for compiler in self._aggregate_compilers.values()]

    def register_reasons(self, callback: Any) -> None:
        """Apply one plugin callback to this registry's private reason catalogs.

        Args:
            callback (Any): Callable receiving fallback and unsupported reason
                registries, optionally plus ``resolver=...``.

        Returns:
            None: This function does not return a value.

        Raises:
            TypeError: If ``callback`` is not callable.
        """
        if not callable(callback):
            raise TypeError("Handler reason registration expects a callable")
        try:
            inspect.signature(callback).bind(self._fallback_reasons, self._unsupported_reasons, resolver=self.module_resolver)
        except (TypeError, ValueError):
            callback(self._fallback_reasons, self._unsupported_reasons)
            return
        callback(self._fallback_reasons, self._unsupported_reasons, resolver=self.module_resolver)

    def fallback_reason(self, backend_identifier: str, op: str, dtype: Any, strategy: Any) -> str | None:
        """Return the best resolver-local Python fallback explanation.

        Args:
            backend_identifier (str): Backend type identifier.
            op (str): Operation identifier.
            dtype (Any): Logical type class, instance, identifier, or alias.
            strategy (Any): Storage strategy class, instance, identifier, or alias.

        Returns:
            str | None: Best matching fallback explanation, or ``None``.
        """
        return self._fallback_reasons.get(backend_identifier, op, dtype, strategy)

    def unsupported_reason(self, backend_identifier: str, op: str, dtype: Any, strategy: Any) -> str | None:
        """Return the best resolver-local unsupported-operation explanation.

        Args:
            backend_identifier (str): Backend type identifier.
            op (str): Operation identifier.
            dtype (Any): Logical type class, instance, identifier, or alias.
            strategy (Any): Storage strategy class, instance, identifier, or alias.

        Returns:
            str | None: Best matching unsupported explanation, or ``None``.
        """
        return self._unsupported_reasons.get(backend_identifier, op, dtype, strategy)
