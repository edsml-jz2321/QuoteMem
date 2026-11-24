"""Immutable resolved snapshots for HeavenBase configuration."""

from __future__ import annotations

__all__ = ["ConfigSnapshot"]

from collections.abc import Mapping
from typing import Any, Dict, Iterator, Optional, Tuple

from .data import _freeze, _unfreeze
from .interpolation import InterpolationPolicy, _index

_MISSING = object()


class ConfigSnapshot(Mapping):
    """Immutable resolved config snapshot with O(1) path lookup."""

    def __init__(self, data: Dict[str, Any], gen: Tuple[int, ...], policy: InterpolationPolicy) -> None:
        """
        Initialize the utility object with the provided runtime settings.

        Use this helper for nested config dictionaries and scoped runtime configuration layers.

        Args:
            data (Dict[str, Any]): Dictionary layer or payload to read, write, merge, or render.
            gen (Tuple[int, ...]): Generation tuple used to detect when a cached snapshot is stale.
            policy (InterpolationPolicy): Interpolation policy. ``None`` uses the default policy.

        Returns:
            None: This function returns ``None`` after initializing the object.
        """
        self.data = _freeze(data)
        self.gen = gen
        self.policy = policy
        self.flat, self.sub = _index(self.data)

    def get(self, path: Optional[str] = None, default: Any = None) -> Any:
        """
        Return one config value by dot path.

        Use this helper for nested config dictionaries and scoped runtime configuration layers.

        Args:
            path (Optional[str]): Local path, config path, or dotted dictionary path used by the helper. Defaults to ``None``.
            default (Any): Fallback value used when a path or config key is missing. Defaults to ``None``.

        Returns:
            Any: The one config value by dot path.
        """
        if path is None:
            return self.data
        if path in self.flat:
            return self.flat[path]
        if path in self.sub:
            return self.sub[path]
        return default

    def to_dict(self) -> Dict[str, Any]:
        """Return a mutable dictionary copy.

        Use this helper for nested config dictionaries and scoped runtime configuration layers.

        Args:
            None.

        Returns:
            Dict[str, Any]: The a mutable dictionary copy.
        """
        return _unfreeze(self.data)

    def __getitem__(self, key: str) -> Any:
        value = self.get(key, _MISSING)
        if value is _MISSING:
            raise KeyError(key)
        return value

    def __iter__(self) -> Iterator[str]:
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)
