"""Resolver-bound fallback and unsupported-operation explanations."""

from __future__ import annotations

__all__ = ["ReasonRegistry"]

from heavenbase.utils import Any, List, Optional, Tuple
from heavenbase.utils.registry import Registry

from ...filters.ops import normalize_op
from ...modules import ModuleResolver, module_resolver
from ...strategies.base import StorageStrategy
from ...types import LogicalType, get_logical_type
from .base import backend_identifier

ReasonKey = Tuple[str, str, str, str]

DEFAULT_FALLBACK_TEMPLATE = "{op} on {dtype} ({strategy}) is not natively supported by {backend}; using Python scan fallback."
DEFAULT_UNSUPPORTED_TEMPLATE = "No handler is registered for {backend}.{op} on {dtype} ({strategy})."


class ReasonRegistry:
    """Store handler explanations under one immutable module authority."""

    def __init__(self, default_template: Optional[str] = None, *, resolver: ModuleResolver | None = None) -> None:
        """Create an empty resolver-bound reason catalog.

        Args:
            default_template (Optional[str]): Optional fallback template used when no
                specific reason is registered.
            resolver (ModuleResolver | None): Context-owned module resolver used to
                normalize capability keys.

        Returns:
            None: This function does not return a value.
        """
        self._module_resolver = module_resolver(resolver)
        self._reasons = Registry("handler-reasons", normalize=self._key)
        self.default_template = default_template

    def _copy(self) -> "ReasonRegistry":
        """Return an independent catalog that starts with this one's reasons."""
        copied = ReasonRegistry(self.default_template, resolver=self._module_resolver)
        copied._reasons._start_from(self._reasons)
        return copied

    def _key(self, raw: tuple[Any, Any, Any, Any]) -> ReasonKey:
        backend, op, dtype, strategy = raw
        return self._backend(backend), self._op(op), self._dtype(dtype), self._strategy(strategy)

    def _backend(self, value: Any) -> str:
        return "*" if value == "*" else backend_identifier(value, resolver=self._module_resolver)

    def _op(self, value: Any) -> str:
        return "*" if value == "*" else normalize_op(str(value), resolver=self._module_resolver)

    def _dtype(self, value: Any) -> str:
        if value == "*":
            return "*"
        if isinstance(value, str):
            try:
                return get_logical_type(value, resolver=self._module_resolver).identifier
            except KeyError:
                return LogicalType.identifier_of(value, resolver=self._module_resolver)
        return LogicalType.identifier_of(value, resolver=self._module_resolver)

    def _strategy(self, value: Any) -> str:
        return "*" if value == "*" else StorageStrategy.identifier_of(value, resolver=self._module_resolver)

    def register(self, backend_identifier: str, op: str, dtype: Any, strategy: Any, reason: str) -> None:
        """Register one normalized explanation.

        Args:
            backend_identifier (str): Backend type identifier or ``"*"`` wildcard.
            op (str): Operation identifier or ``"*"`` wildcard.
            dtype (Any): Logical type identifier or ``"*"`` wildcard.
            strategy (Any): Storage strategy identifier or ``"*"`` wildcard.
            reason (str): Human-readable explanation text.

        Returns:
            None: This function does not return a value.

        Raises:
            ValueError: If ``reason`` is empty after normalization.
        """
        text = str(reason).strip()
        if not text:
            raise ValueError("Handler reason must not be empty")
        self._reasons.update((backend_identifier, op, dtype, strategy), text)

    def get(self, backend_identifier: str, op: str, dtype: Any, strategy: Any) -> Optional[str]:
        """Return the most-specific explanation for one capability key.

        Args:
            backend_identifier (str): Backend type identifier.
            op (str): Operation identifier.
            dtype (Any): Logical type class, instance, identifier, or alias.
            strategy (Any): Storage strategy class, instance, identifier, or alias.

        Returns:
            Optional[str]: Best matching explanation, default-template text, or ``None``.
        """
        backend_identifier, op, dtype, strategy = self._key((backend_identifier, op, dtype, strategy))
        keys: List[ReasonKey] = [
            (backend_identifier, op, dtype, strategy),
            (backend_identifier, op, dtype, "*"),
            (backend_identifier, op, "*", "*"),
            (backend_identifier, "*", dtype, "*"),
            (backend_identifier, "*", "*", "*"),
            ("*", op, dtype, "*"),
            ("*", op, "*", "*"),
            ("*", "*", dtype, "*"),
            ("*", "*", "*", "*"),
        ]
        reasons = self._reasons.snapshot()
        for key in keys:
            reason = reasons.get(key)
            if reason is not None:
                return reason
        if self.default_template is None:
            return None
        return self.default_template.format(backend=backend_identifier, op=op, dtype=dtype, strategy=strategy)
