"""Interpolation policy and resolution for HeavenBase configuration."""

from __future__ import annotations

__all__ = ["InterpolationPolicy"]

import contextvars
import inspect
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any, Dict, Optional, Tuple

from ..commands import cmd

_INTERP_POLICY: contextvars.ContextVar["InterpolationPolicy"] = contextvars.ContextVar("_HB_CONFIG_INTERP_POLICY")
_ENV_VALUES: contextvars.ContextVar[Mapping[str, str | None] | None] = contextvars.ContextVar("_HB_CONFIG_ENV_VALUES", default=None)


@dataclass(frozen=True)
class _Environment:
    values: Mapping[str, str | None] = field(repr=False)
    digest: tuple[tuple[str, bytes | None], ...]


def _capture_env(names: frozenset[str | None]) -> _Environment | None:
    """Read each tracked variable once for both interpolation and cache identity."""
    if None in names:
        return None
    values = {name: os.environ.get(name) for name in sorted(name for name in names if name is not None)}
    # A value can itself introduce a new interpolation when OmegaConf resolves it.
    if any(value is not None and "${" in value for value in values.values()):
        return None
    digest = tuple((name, sha256(value.encode("utf-8", errors="surrogatepass")).digest() if value is not None else None) for name, value in values.items())
    return _Environment(values, digest)


@dataclass(frozen=True)
class InterpolationPolicy:
    """Control whether config interpolation may access external process state.

    Args:
        allow_env (bool): Whether `${env:NAME}` and
            `${env:NAME,default}` may read environment variables. Enabled by
            default.
        allow_cmd (bool): Whether `${cmd:COMMAND}` may execute a shell command.
            Disabled by default and intended only for trusted configuration.
    """

    allow_env: bool = True
    allow_cmd: bool = False

    def digest(self) -> tuple[bool, bool]:
        """Return the stable cache identity for these permissions.

        Args:
            None.

        Returns:
            tuple[bool, bool]: Environment and command permission flags.
        """
        return self.allow_env, self.allow_cmd


def _index(data: Any, prefix: str = "", flat: Optional[Dict[str, Any]] = None, sub: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    flat = flat if flat is not None else {}
    sub = sub if sub is not None else {}
    if prefix:
        sub[prefix] = data
    if isinstance(data, Mapping):
        for key, value in data.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            _index(value, next_prefix, flat, sub)
    elif isinstance(data, tuple):
        for idx, value in enumerate(data):
            next_prefix = f"{prefix}[{idx}]"
            _index(value, next_prefix, flat, sub)
    else:
        flat[prefix] = data
    return flat, sub


def _digest(value: Any) -> Any:
    """Return a hashable digest of nested config data for ``hash()``.

    Data that compare equal have equal digests, as ``hash()`` requires of
    equal objects: mapping items form a set, so neither key order nor keys
    that cannot be compared with each other, such as ``404`` beside
    ``"default"``, matter. Unequal data may share a digest, such as ``True``
    and ``1``; :func:`_identity` tells those apart.
    """
    if isinstance(value, Mapping):
        return frozenset((k, _digest(v)) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return tuple(_digest(v) for v in value)
    return value


def _identity(value: Any) -> Any:
    """Return a hashable, order-independent identity of nested config data.

    Two values have the same identity only when they hold the same data:
    each scalar other than a ``str``, an ``int`` or ``None`` carries its
    type, so ``True``, ``1`` and ``1.0`` differ, and a mapping differs from a
    list of its items. A list and a tuple with the same items are the same config
    data. Mapping items are ordered by key, so the identity, and its
    ``repr``, do not depend on insertion order.
    """
    kind = type(value)
    if kind is str or kind is int or value is None:
        return value
    if isinstance(value, Mapping):
        items = [(_identity(k), _identity(v)) for k, v in value.items()]
        if all(type(k) is str for k in value):
            items.sort(key=lambda item: item[0])
        else:
            items.sort(key=lambda item: repr(item[0]))
        return ("m", tuple(items))
    if isinstance(value, (list, tuple)):
        return ("l", tuple(_identity(v) for v in value))
    name = f"{kind.__module__}.{kind.__qualname__}"
    if kind is float:
        return (name, repr(value))
    try:
        hash(value)
    except TypeError:
        return (name, repr(value))
    return (name, value)


def _setup_resolvers() -> None:
    from omegaconf import OmegaConf

    # The grammar passes a literal default such as `${env:PORT,5432}` as an int; it is returned unchanged.
    def env_resolver(var: str, default: Any = "") -> Any:
        policy = _INTERP_POLICY.get(InterpolationPolicy())
        if not policy.allow_env:
            raise ValueError("OmegaConf env resolver is disabled by policy")
        values = _ENV_VALUES.get()
        value = os.environ.get(var) if values is None or not isinstance(var, str) else values[var]
        return default if value is None else value

    def cmd_resolver(command: str) -> str:
        policy = _INTERP_POLICY.get(InterpolationPolicy())
        if not policy.allow_cmd:
            raise ValueError("OmegaConf cmd resolver is disabled by policy")
        result = cmd(command, shell=True, include=["stdout", "stderr", "returncode"])
        if result["returncode"]:
            raise RuntimeError(result["stderr"] or result["stdout"] or f"Command interpolation failed: {command}")
        return (result["stdout"] or "").strip()

    # OmegaConf 2.4 deprecates register_new_resolver for register_resolver; 2.3's register_resolver is an older legacy API.
    if "annotation_validation" in inspect.signature(OmegaConf.register_resolver).parameters:
        register = OmegaConf.register_resolver
    else:
        register = OmegaConf.register_new_resolver
    for name, resolver in (("env", env_resolver), ("cmd", cmd_resolver)):
        register(name, resolver, replace=True)


def _resolve_interpolation(data: Dict[str, Any], policy: InterpolationPolicy, environment: _Environment | None = None) -> Dict[str, Any]:
    from omegaconf import OmegaConf

    _setup_resolvers()
    token = _INTERP_POLICY.set(policy)
    env_token = _ENV_VALUES.set(environment.values if environment is not None else None)
    try:
        cfg = OmegaConf.create(data)
        OmegaConf.resolve(cfg)
        return OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False) or {}
    finally:
        _ENV_VALUES.reset(env_token)
        _INTERP_POLICY.reset(token)
