"""User-facing capability inspection front door."""

from __future__ import annotations

import threading

from heavenbase._lazy import _attach_lazy_exports
from heavenbase.utils import Any

_LAZY_EXPORTS = {
    "BackendOption": (".options", "BackendOption"),
    "CapabilityOption": (".options", "CapabilityOption"),
    "CapabilityIndex": (".index", "CapabilityIndex"),
    "LogicalTypeOption": (".options", "LogicalTypeOption"),
    "StrategyOption": (".options", "StrategyOption"),
}

__all__ = [*_LAZY_EXPORTS, "logical_types", "strategies", "backends", "ops", "capabilities", "supports"]

_DEFAULT_INDEX: Any = None
_DEFAULT_RESOLVER: Any = None
_DEFAULT_LOCK = threading.RLock()


def _default_index() -> Any:
    """Return an index bound to the current default Context's resolver."""
    global _DEFAULT_INDEX, _DEFAULT_RESOLVER
    from ..modules import module_resolver

    resolver = module_resolver()
    with _DEFAULT_LOCK:
        if _DEFAULT_INDEX is None or _DEFAULT_RESOLVER is not resolver:
            from .index import CapabilityIndex

            _DEFAULT_INDEX = CapabilityIndex(resolver=resolver)
            _DEFAULT_RESOLVER = resolver
        return _DEFAULT_INDEX


def logical_types(*, strategy: Any = None, backend: Any = None, op: Any = None) -> tuple[Any, ...]:
    """Delegate logical-type discovery to the default capability index.

    Args:
        strategy (Any): Optional storage strategy filter. Defaults to ``None``.
        backend (Any): Optional backend filter. Defaults to ``None``.
        op (Any): Optional operation filter. Defaults to ``None``.

    Returns:
        tuple[Any, ...]: Matching logical type options in registry order.
    """
    return _default_index().logical_types(strategy=strategy, backend=backend, op=op)


def strategies(dtype: Any = None, *, backend: Any = None, op: Any = None) -> tuple[Any, ...]:
    """Delegate strategy discovery to the default capability index.

    Args:
        dtype (Any): Optional logical type filter. Defaults to ``None``.
        backend (Any): Optional backend filter. Defaults to ``None``.
        op (Any): Optional operation filter. Defaults to ``None``.

    Returns:
        tuple[Any, ...]: Matching storage strategy options in registry order.
    """
    return _default_index().strategies(dtype, backend=backend, op=op)


def backends(dtype: Any = None, strategy: Any = None, *, op: Any = None) -> tuple[Any, ...]:
    """Delegate backend discovery to the default capability index.

    Args:
        dtype (Any): Optional logical type filter. Defaults to ``None``.
        strategy (Any): Optional storage strategy filter. Defaults to ``None``.
        op (Any): Optional operation filter. Defaults to ``None``.

    Returns:
        tuple[Any, ...]: Matching backend options for the active index binding.
    """
    return _default_index().backends(dtype, strategy, op=op)


def ops(dtype: Any = None, strategy: Any = None, backend: Any = None) -> tuple[str, ...]:
    """Delegate operation discovery to the default capability index.

    Args:
        dtype (Any): Optional logical type filter. Defaults to ``None``.
        strategy (Any): Optional storage strategy filter. Defaults to ``None``.
        backend (Any): Optional backend filter. Defaults to ``None``.

    Returns:
        tuple[str, ...]: Matching normalized operation tokens.
    """
    return _default_index().ops(dtype, strategy, backend)


def capabilities(dtype: Any = None, strategy: Any = None, backend: Any = None, op: Any = None) -> tuple[Any, ...]:
    """Delegate capability-row discovery to the default capability index.

    Args:
        dtype (Any): Optional logical type filter. Defaults to ``None``.
        strategy (Any): Optional storage strategy filter. Defaults to ``None``.
        backend (Any): Optional backend filter. Defaults to ``None``.
        op (Any): Optional operation filter. Defaults to ``None``.

    Returns:
        tuple[Any, ...]: Matching handler capability rows.
    """
    return _default_index().capabilities(dtype, strategy, backend, op)


def supports(dtype: Any, op: Any, backend: Any, strategy: Any) -> bool:
    """Return whether the default capability index supports one key.

    Args:
        dtype (Any): Logical type selector.
        op (Any): Operation token.
        backend (Any): Backend selector.
        strategy (Any): Storage strategy selector.

    Returns:
        bool: Whether the default index reports native support.
    """
    return _default_index().supports(dtype, op, backend, strategy)


__getattr__, __dir__ = _attach_lazy_exports(__name__, globals(), _LAZY_EXPORTS, __all__)
