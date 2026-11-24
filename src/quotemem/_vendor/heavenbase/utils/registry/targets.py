"""Durable Python target helpers for Registry-backed values."""

from __future__ import annotations

from importlib import import_module
from typing import Any


def restore_python_target(target: str) -> Any:
    """Restore an object from a validated Python import target.

    Args:
        target: Import target using `module:qualified_name` syntax.

    Returns:
        Any: Object restored from the target.

    Raises:
        ImportError: If the module or qualified attribute does not exist.
        ValueError: If the target does not contain a qualified-name separator.
    """
    module_name, separator, qualname = str(target).partition(":")
    if not separator:
        raise ValueError(f"Python import target {target!r} must use 'module:qualified_name' syntax")
    value = import_module(module_name)
    try:
        for part in qualname.split("."):
            value = getattr(value, part)
    except AttributeError as error:
        raise ImportError(f"Python import target {target!r} does not exist") from error
    return value


def python_target(value: Any, *, label: str) -> str:
    """Return an importable, identity-preserving locator for an object.

    Args:
        value: Class, function, or other module-level object to locate.
        label: Human-readable object label used in validation errors.

    Returns:
        str: Durable `module:qualified_name` locator.

    Raises:
        ValueError: If `value` is local, anonymous, absent from its module, or
            resolves to a different object.
    """
    module_name = str(getattr(value, "__module__", ""))
    qualname = str(getattr(value, "__qualname__", ""))
    if not module_name or not qualname or "<locals>" in qualname or "<lambda>" in qualname:
        raise ValueError(f"{label} must be an importable top-level object")
    target = f"{module_name}:{qualname}"
    try:
        restored = restore_python_target(target)
    except (ImportError, ValueError) as error:
        raise ValueError(f"{label} must be importable as {target!r}") from error
    same_bound_method = (
        getattr(restored, "__func__", None) is not None
        and getattr(restored, "__func__", None) is getattr(value, "__func__", None)
        and getattr(restored, "__self__", None) is getattr(value, "__self__", None)
    )
    if restored is not value and not same_bound_method:
        raise ValueError(f"{label} locator {target!r} resolves to a different object")
    return target
