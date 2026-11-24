"""Capsule restore policy, verification, and built-in restorers."""

from __future__ import annotations

__all__ = ["RestorePolicy", "restore_callable"]

import base64
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
import importlib
import threading

from heavenbase.utils import Any, Callable, Dict, List, Mapping, sha256hash
from heavenbase.utils.config import ConfigManager

from ..modules import ModuleResolver
from .authority import CapsuleAuthority
from .errors import CapsuleRestoreError
from .layers import restore_capsule_layer, verify_capsule_layer
from .manifest import CapsuleManifest


@dataclass(frozen=True, slots=True)
class RestorePolicy:
    """Policy controlling trusted-local Capsule restoration.

    Args:
        layers (Sequence[str] | None): Optional restore layer order/filter. ``None`` restores by manifest priority.
            Defaults to ``None``.
        allow_source (bool): Whether source restore layers may execute. Defaults to ``True``.
        allow_import (bool): Whether import restore layers may import modules. Defaults to ``True``.
        allow_binary (bool): Whether trusted-local ``cloudpickle`` payloads may load. Defaults to ``False``.
        verify_source_hash (bool | None): Whether to verify stored source and binary payload checksums before restore.
            ``None`` uses the manifest/config policy. Defaults to ``None``.
        block_source (bool): Test hook that rejects source restore layers even when ``allow_source`` is true. Defaults to
            ``False``.

    Returns:
        None: Dataclass initialization only.

    Warnings:
        This policy verifies that a loaded Capsule record still matches its stored checksums. It does not prove the
        original function was safe, and it does not protect against a user replacing both executable content and
        checksums in the registry.
    """

    layers: Sequence[str] | None = None
    allow_source: bool = True
    allow_import: bool = True
    allow_binary: bool = False
    verify_source_hash: bool | None = None
    block_source: bool = False
    _authority: CapsuleAuthority | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.layers is not None:
            object.__setattr__(self, "layers", tuple(str(layer) for layer in self.layers))

    def _bind(self, authority: CapsuleAuthority) -> "RestorePolicy":
        if self._authority is authority:
            return self
        bound = replace(self)
        object.__setattr__(bound, "_authority", authority)
        return bound

    @property
    def resolver(self) -> ModuleResolver:
        """Return the exact module authority bound for this restore."""
        return (self._authority or CapsuleAuthority.resolve()).resolver

    @property
    def config(self) -> ConfigManager:
        """Return the exact config authority bound for this restore."""
        return (self._authority or CapsuleAuthority.resolve()).config


_VERIFY_LOCK = threading.RLock()
_VERIFY_CACHE: Dict[tuple[str, str, str], bool] = {}


def _verify_cache_max(policy: RestorePolicy) -> int:
    return int(policy.config.get("heavenbase.capsule.registry.trust.verify_cache_size", default=1024))


def _verify_enabled(manifest: CapsuleManifest, policy: RestorePolicy) -> bool:
    if policy.verify_source_hash is not None:
        return bool(policy.verify_source_hash)
    if "verify_source_hash" in manifest.trust:
        return bool(manifest.trust.get("verify_source_hash"))
    return bool(policy.config.get("heavenbase.capsule.registry.trust.verify_source_hash", default=True))


def _cache_get(key: tuple[str, str, str]) -> bool:
    with _VERIFY_LOCK:
        if key not in _VERIFY_CACHE:
            return False
        _VERIFY_CACHE[key] = _VERIFY_CACHE.pop(key)
        return True


def _cache_set(key: tuple[str, str, str], policy: RestorePolicy) -> None:
    with _VERIFY_LOCK:
        _VERIFY_CACHE[key] = True
        while len(_VERIFY_CACHE) > _verify_cache_max(policy):
            _VERIFY_CACHE.pop(next(iter(_VERIFY_CACHE)))


def _verify_layer_hash(
    manifest: CapsuleManifest,
    layer: Mapping[str, Any],
    policy: RestorePolicy,
    *,
    hash_key: str,
    payload_key: str,
    manifest_hash_key: str | None = None,
) -> None:
    verify_source_hash = _verify_enabled(manifest, policy)
    if not verify_source_hash:
        return
    layer_type = str(layer.get("type") or "")
    expected = str(layer.get(hash_key) or (manifest.fingerprints.get(manifest_hash_key) if manifest_hash_key else "") or "")
    payload = layer.get(payload_key)
    if payload is None:
        payload = ""
    if not expected:
        return
    if not isinstance(payload, str):
        raise TypeError(f"{layer_type} checksum payload {payload_key!r} must be text")
    key = (layer_type, expected, payload)
    if _cache_get(key):
        return
    actual = sha256hash(payload)
    if actual != expected:
        raise ValueError(f"{layer_type} checksum mismatch for capsule {manifest.id!r}")
    _cache_set(key, policy)


def _verify_source_layer(manifest: CapsuleManifest, layer: Dict[str, Any], policy: RestorePolicy) -> None:
    _verify_layer_hash(manifest, layer, policy, hash_key="source_hash", payload_key="source", manifest_hash_key="source_hash")


def _verify_cloudpickle_layer(manifest: CapsuleManifest, layer: Dict[str, Any], policy: RestorePolicy) -> None:
    _verify_layer_hash(manifest, layer, policy, hash_key="payload_hash", payload_key="payload_b64")


def _resolve_qualname(scope: Mapping[str, Any], qualname: str) -> Any:
    parts = [part for part in qualname.split(".") if part and part != "<locals>"]
    if not parts:
        raise KeyError(qualname)
    value = scope.get(parts[0])
    if value is None and parts[-1] in scope:
        value = scope[parts[-1]]
        parts = [parts[-1]]
    if value is None:
        raise KeyError(qualname)
    for part in parts[1:]:
        value = getattr(value, part)
    return value


def _resolve_global_ref(ref: Mapping[str, Any]) -> Any:
    kind = ref.get("kind")
    if kind == "module":
        return importlib.import_module(str(ref["module"]))
    if kind == "attr":
        module = importlib.import_module(str(ref["module"]))
        return _resolve_qualname(module.__dict__, str(ref["qualname"]))
    if kind == "value":
        return ref.get("value")
    raise KeyError(f"Unsupported global ref kind {kind!r}")


def _restore_source(manifest: CapsuleManifest, layer: Dict[str, Any], policy: RestorePolicy) -> Callable[..., Any]:
    if policy.block_source or not policy.allow_source:
        raise PermissionError("source restore is disabled")
    _verify_source_layer(manifest, layer, policy)
    globals_dict: Dict[str, Any] = {"__builtins__": __builtins__, "__name__": f"heavenbase_capsule_{manifest.id}"}
    for name, ref in (layer.get("global_refs") or {}).items():
        globals_dict[name] = _resolve_global_ref(ref)
    locals_dict: Dict[str, Any] = {}
    compiled = compile(layer["source"], layer.get("filename") or f"<capsule:{manifest.id}>", "exec")
    exec(compiled, globals_dict, locals_dict)
    return _resolve_qualname({**globals_dict, **locals_dict}, layer.get("qualname") or manifest.name)


def _restore_import(manifest: CapsuleManifest, layer: Dict[str, Any], policy: RestorePolicy) -> Callable[..., Any]:
    if not policy.allow_import:
        raise PermissionError("import restore is disabled")
    module = importlib.import_module(layer["module"])
    return _resolve_qualname(module.__dict__, layer.get("qualname") or manifest.name)


def _restore_cloudpickle(manifest: CapsuleManifest, layer: Dict[str, Any], policy: RestorePolicy) -> Callable[..., Any]:
    if not policy.allow_binary:
        raise PermissionError("cloudpickle restore requires allow_binary=True; this trusted-local binary fallback is disabled by default")
    _verify_cloudpickle_layer(manifest, layer, policy)
    import cloudpickle

    return cloudpickle.loads(base64.b64decode(layer["payload_b64"].encode("ascii")))


def _selected_restore_layers(manifest: CapsuleManifest, policy: RestorePolicy) -> List[Dict[str, Any]]:
    layers = [dict(layer) for layer in manifest.restore_layers]
    if not policy.layers:
        return sorted(layers, key=lambda item: item.get("priority", 100))
    requested = [str(name) for name in policy.layers]
    order = {name: index for index, name in enumerate(requested)}
    selected = [layer for layer in layers if str(layer.get("type") or "") in order]
    selected.sort(key=lambda layer: order.get(str(layer.get("type") or ""), 10**9))
    if not selected:
        raise CapsuleRestoreError(manifest.id, [{"layer": requested, "error": "no requested layers found in capsule"}])
    return selected


def restore_callable(
    manifest: CapsuleManifest,
    policy: RestorePolicy | None = None,
    *,
    resolver: ModuleResolver | None = None,
    config: ConfigManager | None = None,
) -> Callable[..., Any]:
    """Restore a callable from the first allowed working restore layer.

    Args:
        manifest (CapsuleManifest): Manifest containing ordered restore layers.
        policy (RestorePolicy | None): Restore policy and optional layer filter. Defaults to ``None``.
        resolver (ModuleResolver | None): Context-owned module resolver used
            for Capsule layers. When omitted, the default Context is selected.
        config (ConfigManager | None): Config authority paired with `resolver`.
            When omitted, the standalone process config is selected.

    Returns:
        Callable[..., Any]: First callable restored from an allowed working layer.

    Raises:
        CapsuleRestoreError: If every selected restore layer fails.
    """
    authority = CapsuleAuthority.resolve(resolver=resolver, config=config)
    chosen = authority.resolver
    policy = (policy or RestorePolicy())._bind(authority)
    errors: List[Dict[str, Any]] = []
    for layer in _selected_restore_layers(manifest, policy):
        layer_type = layer.get("type")
        try:
            return restore_capsule_layer(manifest, layer, policy, resolver=chosen)
        except Exception as exc:
            errors.append({"layer": layer_type, "error": str(exc)})
    if "cloudpickle" not in {layer.get("type") for layer in manifest.restore_layers}:
        errors.append(
            {
                "layer": "cloudpickle",
                "error": "no binary fallback is stored; source-only capsules cannot use trusted-local binary loading",
            }
        )
    raise CapsuleRestoreError(manifest.id, errors)


def _verify_manifest(
    manifest: CapsuleManifest,
    resolver: ModuleResolver | None = None,
    policy: RestorePolicy | None = None,
    *,
    config: ConfigManager | None = None,
) -> None:
    """Verify stored source and binary checksums without executing code.

    Args:
        manifest (CapsuleManifest): Manifest loaded from a registry or serialized payload.
        resolver (ModuleResolver | None): Context-owned module resolver used for
            Capsule layers.
        policy (RestorePolicy | None): Optional restore policy controlling checksum verification. Defaults to ``None``.
        config (ConfigManager | None): Config authority paired with `resolver`.
            When omitted, the standalone process config is selected.

    Returns:
        None: Raises when a stored checksum does not match its payload.
    """
    authority = CapsuleAuthority.resolve(resolver=resolver, config=config)
    policy = (policy or RestorePolicy())._bind(authority)
    for layer in _selected_restore_layers(manifest, policy):
        verify_capsule_layer(manifest, layer, policy, resolver=authority.resolver)
