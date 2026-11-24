"""Discover packaged module-folder metadata without importing implementations."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
import copy
from dataclasses import dataclass
from importlib import resources
import os
from pathlib import Path, PurePosixPath
import threading
from typing import Any

from ._signature import interpreter_cache, tree_signature

try:
    from importlib.resources.abc import Traversable
except ModuleNotFoundError:  # Python 3.10 keeps Traversable on importlib.abc.
    from importlib.abc import Traversable

_CONFIG_PATH = ("resources", "modules.yaml")
_CONFIG_SCHEMA = "heaven.modules/v1"
_META_FILE = "meta.yaml"
# The only files whose bytes discovery reads.
_DESCRIPTOR_NAMES = frozenset((_META_FILE, _CONFIG_PATH[-1]))

# Per-process readings of package module folders: package root -> (fingerprint, modules),
# the newest few packages only.
_DISCOVERED_LIMIT = 16
_DISCOVERED: OrderedDict[str, tuple[tuple, tuple["DiscoveredModule", ...]]] = OrderedDict()
_DISCOVERED_LOCK = threading.Lock()


def _new_lock_in_child() -> None:
    """Give a forked child an unheld lock: the thread that held the parent's does not exist there."""
    global _DISCOVERED_LOCK
    _DISCOVERED_LOCK = threading.Lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_new_lock_in_child)


@dataclass(frozen=True, slots=True)
class DiscoveredModule:
    """Carry inert module metadata with its package-resource root.

    Args:
        name: Stable package-relative descriptor path.
        manifest: Plain parsed `meta.yaml` mapping.
        root: Folder containing the descriptor and its co-located resources.
    """

    name: str
    manifest: dict[str, Any]
    root: Traversable


def _mapping(resource: Traversable, label: str) -> dict[str, Any]:
    """Read one YAML mapping without importing package implementation code."""
    import yaml

    try:
        value = yaml.safe_load(resource.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"Invalid {label}: {error}") from error
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} root must be a mapping")
    return dict(value)


def _roots(package_root: Traversable) -> tuple[PurePosixPath, ...]:
    """Return validated relative discovery roots from the package config."""
    config_resource = package_root.joinpath(*_CONFIG_PATH)
    config = _mapping(config_resource, "module discovery config resources/modules.yaml")
    unknown = set(config) - {"schema", "roots"}
    if unknown:
        raise ValueError(f"Module discovery config has unsupported fields {sorted(unknown)!r}")
    if config.get("schema") != _CONFIG_SCHEMA:
        raise ValueError(f"Module discovery config schema must be {_CONFIG_SCHEMA!r}")
    raw_roots = config.get("roots")
    if not isinstance(raw_roots, list) or not raw_roots:
        raise ValueError("Module discovery config roots must be a non-empty list")
    roots: list[PurePosixPath] = []
    for index, value in enumerate(raw_roots):
        if not isinstance(value, str) or not value.strip():
            raise TypeError(f"Module discovery root {index} must be a non-empty string")
        raw = value.strip()
        path = PurePosixPath(raw)
        if path.is_absolute() or ".." in path.parts or "\\" in raw:
            raise ValueError(f"Module discovery root {raw!r} must stay inside the package")
        roots.append(path)
    return tuple(dict.fromkeys(roots))


def _metadata_files(root: Traversable, prefix: str) -> tuple[tuple[str, Traversable, Traversable], ...]:
    """Return recursively discovered exact-name metadata files."""
    pending = [(prefix, root)]
    found: list[tuple[str, Traversable, Traversable]] = []
    while pending:
        relative, directory = pending.pop()
        for child in sorted(directory.iterdir(), key=lambda item: item.name, reverse=True):
            if interpreter_cache(child.name):
                continue
            name = f"{relative}/{child.name}" if relative else child.name
            if child.is_dir():
                pending.append((name, child))
            elif child.is_file() and child.name == _META_FILE:
                found.append((name, child, directory))
    return tuple(found)


def discover_package_modules(package: str = "heavenbase") -> tuple[DiscoveredModule, ...]:
    """Discover co-located module metadata below configured package roots.

    Args:
        package: Installed package whose `resources/modules.yaml` owns scan roots.

    Returns:
        tuple[DiscoveredModule, ...]: Stable descriptor paths, plain metadata,
        and their co-located package-resource roots. No implementation module
        is imported or executed.

    Raises:
        RuntimeError: If no `meta.yaml` module folders are discovered.
        TypeError: If configuration or metadata has an invalid root shape.
        ValueError: If configuration, paths, or YAML content is invalid.
    """
    package_root = resources.files(package)
    key = signature = None
    if isinstance(package_root, Path):
        key = str(package_root)
        signature = tree_signature(package_root, ignore=interpreter_cache, contents=_DESCRIPTOR_NAMES.__contains__)
    if signature is not None:
        with _DISCOVERED_LOCK:
            cached = _DISCOVERED.get(key)
        if cached is not None and cached[0] == signature:
            return _copies(cached[1])
    found = _discover(package, package_root)
    if signature is not None and tree_signature(package_root, ignore=interpreter_cache, contents=_DESCRIPTOR_NAMES.__contains__) == signature:
        kept = _copies(found)
        with _DISCOVERED_LOCK:
            _DISCOVERED[key] = (signature, kept)
            _DISCOVERED.move_to_end(key)
            while len(_DISCOVERED) > _DISCOVERED_LIMIT:
                _DISCOVERED.popitem(last=False)
    return found


def forget_discoveries() -> None:
    """Drop every per-process reading, so the next discovery reads the package again."""
    with _DISCOVERED_LOCK:
        _DISCOVERED.clear()


def _copies(modules: tuple[DiscoveredModule, ...]) -> tuple[DiscoveredModule, ...]:
    """Return modules whose manifests a caller may change without changing the cached reading."""
    return tuple(DiscoveredModule(module.name, copy.deepcopy(module.manifest), module.root) for module in modules)


def _discover(package: str, package_root: Traversable) -> tuple[DiscoveredModule, ...]:
    """Read the configured roots of one package and parse each module descriptor."""
    discovered: dict[str, tuple[Traversable, Traversable]] = {}
    for root in _roots(package_root):
        parts = () if str(root) == "." else root.parts
        directory = package_root.joinpath(*parts)
        if not directory.is_dir():
            raise ValueError(f"Module discovery root {str(root)!r} does not exist in package {package!r}")
        prefix = "" if not parts else root.as_posix()
        for name, resource, module_root in _metadata_files(directory, prefix):
            discovered[name] = resource, module_root
    if not discovered:
        raise RuntimeError(f"No {_META_FILE} module folders were found in package {package!r}")
    return tuple(
        DiscoveredModule(name, _mapping(resource, f"module metadata {name}"), module_root) for name, (resource, module_root) in sorted(discovered.items())
    )
