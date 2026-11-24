"""Immutable configuration specification values."""

from __future__ import annotations

__all__ = ["ConfigSpec", "resolve_interpolations"]

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from .data import _freeze, to_mutable_data
from .interpolation import InterpolationPolicy, _digest, _resolve_interpolation


def resolve_interpolations(data: dict[str, Any], policy: InterpolationPolicy | None = None) -> dict[str, Any]:
    """Resolve supported interpolations in a JSON-like config mapping.

    Args:
        data (dict[str, Any]): Mutable or immutable config mapping to resolve.
        policy (InterpolationPolicy | None): Interpolation permissions. When
            omitted, environment interpolation is enabled and command
            interpolation is disabled.

    Returns:
        dict[str, Any]: Independent resolved mapping.
    """
    return _resolve_interpolation(to_mutable_data(data), policy or InterpolationPolicy())


class _Memoized:
    """Attribute of an immutable value, computed at its first use and kept in the instance's ``__dict__``.

    Before Python 3.12 ``functools.cached_property`` computes under one lock that every instance shares, so one slow
    computation holds up all the others, and a child forked meanwhile waits forever. This one takes no lock: threads
    that use the attribute first at the same time each compute it, to equal values.

    Args:
        compute (Callable[[Any], Any]): Method that returns the value.
    """

    def __init__(self, compute: Callable[[Any], Any]) -> None:
        self._compute = compute
        self.__doc__ = compute.__doc__

    def __set_name__(self, owner: type, name: str) -> None:
        self._name = name

    def __get__(self, instance: Any, owner: type | None = None) -> Any:
        if instance is None:
            return self
        return instance.__dict__.setdefault(self._name, self._compute(instance))


@dataclass(frozen=True, eq=False)
class ConfigSpec:
    """Store one immutable normalized configuration envelope.

    Behavioral identity includes only ``domain``, canonical ``data``, and
    executable ``bindings``. Raw input and metadata remain diagnostic and do
    not affect equality or cache identity. The executable envelope may contain
    secrets; use a domain-owned ``print`` Apply profile for display.

    Args:
        domain (str): Stable configuration-domain identifier.
        data (Mapping[str, Any]): Minimal user-facing canonical mapping.
        bindings (Mapping[str, Any]): Selected immutable executable facts.
        raw (Mapping[str, Any]): Original direct input for diagnostics.
        meta (Mapping[str, Any]): Non-behavioral resolution metadata.
    """

    domain: str
    data: Mapping[str, Any]
    bindings: Mapping[str, Any] = field(default_factory=dict)
    raw: Mapping[str, Any] = field(default_factory=dict)
    meta: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.domain, str) or not self.domain.strip():
            raise ValueError("ConfigSpec.domain must be a non-empty string")
        for name in ("data", "bindings", "raw", "meta"):
            value = getattr(self, name)
            if not isinstance(value, Mapping):
                raise TypeError(f"ConfigSpec.{name} must be a mapping, got {type(value).__name__}")
            object.__setattr__(self, name, _freeze(value))
        object.__setattr__(self, "domain", self.domain.strip())

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ConfigSpec":
        """Build an executable spec from its dictionary representation.

        Args:
            data (Mapping[str, Any]): Envelope returned by ``to_dict``.

        Returns:
            ConfigSpec: Recursively frozen executable specification.

        Raises:
            TypeError: If the envelope or one mapping field has the wrong type.
            ValueError: If ``domain`` or ``data`` is missing.
        """
        if not isinstance(data, Mapping):
            raise TypeError(f"ConfigSpec.from_dict expects a mapping, got {type(data).__name__}")
        for key in ("domain", "data"):
            if key not in data:
                raise ValueError(f"ConfigSpec JSON is missing {key!r}")
        return cls(
            domain=str(data["domain"]),
            data=_mapping_field(data, "data"),
            bindings=_mapping_field(data, "bindings"),
            raw=_mapping_field(data, "raw"),
            meta=_mapping_field(data, "meta"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return an independent executable dictionary envelope.

        Args:
            None.

        Returns:
            dict[str, Any]: Mutable envelope preserving canonical data and
            executable bindings.

        Warnings:
            Bindings may contain credentials or other secrets. This method is
            for persistence and restoration, not logging or display.
        """
        return {
            "domain": self.domain,
            "data": to_mutable_data(self.data),
            "bindings": to_mutable_data(self.bindings),
            "raw": to_mutable_data(self.raw),
            "meta": to_mutable_data(self.meta),
        }

    def get(self, path: str | None = None, default: Any = None) -> Any:
        """Read canonical user-facing data by dotted path.

        Args:
            path (str | None): Dotted canonical-data path. ``None`` returns the
                complete immutable mapping.
            default (Any): Value returned when the path is absent.

        Returns:
            Any: Immutable canonical value or ``default``.
        """
        from .data import dget

        return dget(self.data, path, default=default)

    @_Memoized
    def _cache_identity(self) -> tuple[Any, ...] | None:
        """Return the behavioral identity engine caches key on, computed once per immutable spec.

        Returns:
            tuple[Any, ...] | None: Canonical structure of ``domain``, ``data``, and ``bindings``, or ``None`` when a value
            cannot be fingerprinted.
        """
        from .engine import _Uncacheable, _fingerprint

        try:
            return _fingerprint((self.domain, self.data, self.bindings))
        except _Uncacheable:
            return None

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ConfigSpec):
            return NotImplemented
        return self.domain == other.domain and self.data == other.data and self.bindings == other.bindings

    def __hash__(self) -> int:
        return hash((self.domain, _digest(self.data), _digest(self.bindings)))


def _mapping_field(data: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name) or {}
    if not isinstance(value, Mapping):
        raise TypeError(f"ConfigSpec.{name} must be a mapping, got {type(value).__name__}")
    return to_mutable_data(value)
