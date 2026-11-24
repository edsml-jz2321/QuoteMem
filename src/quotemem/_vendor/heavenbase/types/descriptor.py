"""Logical type descriptor parsing and coercion."""

from __future__ import annotations

__all__ = [
    "coerce_type",
    "logical_type_from_any",
    "logical_type_from_mapping",
    "logical_type_from_text",
]

from heavenbase.utils import Any, Mapping

from ..modules import ModuleResolver, module_resolver
from .base import LogicalType


def coerce_type(dtype: Any) -> LogicalType:
    """Return a logical type instance from a class or instance.

    Args:
        dtype (Any): Logical type class or instance.

    Returns:
        LogicalType: Concrete logical type instance.

    Raises:
        TypeError: If `dtype` is not a logical type class or instance.
    """
    if isinstance(dtype, LogicalType):
        return dtype
    if isinstance(dtype, type) and issubclass(dtype, LogicalType):
        return dtype()
    raise TypeError(f"Expected LogicalType, got {dtype!r}")


def logical_type_from_any(raw: Any, *, resolver: ModuleResolver | None = None) -> LogicalType:
    """Build a logical type from any supported descriptor.

    Args:
        raw (Any): Logical type class, instance, identifier, alias, parametric
            shorthand, or JSON-compatible descriptor mapping.
        resolver (ModuleResolver | None): Context-owned module resolver. The
            default Context is selected when omitted.

    Returns:
        LogicalType: Configured logical type instance.

    Raises:
        TypeError: If `raw` has an unsupported descriptor shape.
        ValueError: If the descriptor is unknown or under-specified.
    """
    if isinstance(raw, LogicalType):
        return raw
    if isinstance(raw, type) and issubclass(raw, LogicalType):
        return raw()
    if isinstance(raw, str):
        return _from_text(raw, module_resolver(resolver))
    if isinstance(raw, Mapping):
        return logical_type_from_mapping(raw, resolver=resolver)
    raise TypeError(f"Expected LogicalType class, instance, identifier, alias, or mapping; got {raw!r}")


def logical_type_from_mapping(
    spec: Mapping[str, Any],
    *,
    resolver: ModuleResolver | None = None,
) -> LogicalType:
    """Build a logical type from a descriptor mapping.

    Args:
        spec (Mapping[str, Any]): Mapping with `type`, `identifier`, or `dtype`
            plus type-specific parameters.
        resolver (ModuleResolver | None): Context-owned module resolver. The
            default Context is selected when omitted.

    Returns:
        LogicalType: Configured logical type instance.

    Raises:
        TypeError: If a class selector is not a `LogicalType` subclass.
        ValueError: If the mapping lacks a selector or parameters are invalid.
    """
    key = spec.get("type") or spec.get("identifier") or spec.get("dtype")
    if isinstance(key, LogicalType):
        return key
    if isinstance(key, type):
        if not issubclass(key, LogicalType):
            raise TypeError(f"Invalid logical type class {key!r}")
        return key.from_dict(dict(spec), resolver=resolver)
    return _from_mapping(spec, module_resolver(resolver))


def _from_mapping(spec: Mapping[str, Any], resolver: ModuleResolver) -> LogicalType:
    """Build one logical type mapping through an explicit resolver."""
    from .registry import get_logical_type

    key = spec.get("type") or spec.get("identifier") or spec.get("dtype")
    if key is None:
        raise ValueError("Type mapping needs type, identifier, or dtype")
    if isinstance(key, LogicalType):
        return key
    elif isinstance(key, type):
        if not issubclass(key, LogicalType):
            raise TypeError(f"Invalid logical type class {key!r}")
        cls = key
    else:
        text = str(key).strip()
        if _is_bracket_shorthand(text):
            return _from_text(text, resolver)
        cls = get_logical_type(text, resolver=resolver)
    return cls.from_dict(dict(spec), resolver=resolver)


def logical_type_from_text(text: str, *, resolver: ModuleResolver | None = None) -> LogicalType:
    """Build a logical type from an identifier or shorthand.

    Args:
        text (str): Identifier or bracket shorthand such as `vector[768]` or
            `array[short-text]`.
        resolver (ModuleResolver | None): Context-owned module resolver. The
            default Context is selected when omitted.

    Returns:
        LogicalType: Configured logical type instance.

    Raises:
        KeyError: If the identifier or alias is unknown.
        ValueError: If shorthand parameters are invalid or missing.
    """
    return _from_text(text, module_resolver(resolver))


def _from_text(text: str, resolver: ModuleResolver) -> LogicalType:
    """Build one logical type text descriptor through an explicit resolver."""
    from .registry import get_logical_type

    value = str(text).strip()
    if not value:
        raise ValueError("Logical type text cannot be empty")
    if _is_bracket_shorthand(value):
        name, parameter = value[:-1].split("[", 1)
        cls = get_logical_type(name.strip(), resolver=resolver)
        return cls.from_dict({"type": cls.identifier, "value": parameter}, resolver=resolver)
    cls = get_logical_type(value, resolver=resolver)
    return cls.from_dict({"type": cls.identifier}, resolver=resolver)


def _is_bracket_shorthand(value: str) -> bool:
    return value.endswith("]") and "[" in value
