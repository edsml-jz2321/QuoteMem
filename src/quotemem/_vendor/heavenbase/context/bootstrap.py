"""Load minimal machine bootstrap data without ConfigManager dependencies."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any, Mapping

_REGISTRY_DEFAULTS = {"hash": "sha256", "max_items": 65536, "max_bytes": 4 * 1024 * 1024}


@dataclass(frozen=True)
class BootstrapSpec:
    """Describe validated construction-critical machine services.

    Args:
        root (str): Expanded HeavenBase machine-state root.
        workspace (str): Default workspace identifier.
        backend (dict[str, Any]): Backend construction mapping, including any
            construction-only ``tags`` required before the Registry exists.
        registry (dict[str, Any]): Registry integrity and capacity policy.
    """

    root: str
    workspace: str
    backend: dict[str, Any]
    registry: dict[str, Any]


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as error:
        raise ValueError(f"Invalid HeavenBase bootstrap file {path}: {error}") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"Invalid HeavenBase bootstrap file {path}: root must be a mapping")
    return dict(value)


def _packaged_bootstrap() -> dict[str, Any]:
    path = resources.files("heavenbase").joinpath("resources", "configs", "bootstrap.yaml")
    import yaml

    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, Mapping):
        raise ValueError("Packaged HeavenBase bootstrap root must be a mapping")
    return dict(value)


def load_bootstrap(
    bootstrap: Mapping[str, Any] | None = None,
    *,
    home_path: str | Path | None = None,
) -> BootstrapSpec:
    """Load and validate construction-critical machine configuration.

    Args:
        bootstrap (Mapping[str, Any] | None): Explicit bootstrap mapping. When
            omitted, HeavenBase reads `home_path` and then the packaged default.
        home_path (str | Path | None): Optional user bootstrap file. The normal
            default is `~/.heavenbase/bootstrap.yaml`.

    Returns:
        BootstrapSpec: Validated bootstrap with expanded local paths and a
        complete Registry policy.

    Raises:
        ValueError: If the selected document has an unsupported version,
            malformed required fields, or invalid Registry policy.
    """
    if bootstrap is not None:
        raw = copy.deepcopy(dict(bootstrap))
    else:
        path = Path(home_path).expanduser() if home_path is not None else Path.home() / ".heavenbase" / "bootstrap.yaml"
        raw = _load_yaml(path) if path.exists() else _packaged_bootstrap()
    version = raw.get("version")
    if not isinstance(version, int) or isinstance(version, bool) or version != 1:
        raise ValueError(f"HeavenBase bootstrap version must be 1, got {version!r}")
    root_raw = raw.get("root")
    workspace_raw = raw.get("workspace")
    workspace = workspace_raw.strip().lower() if isinstance(workspace_raw, str) else ""
    backend = raw.get("backend")
    registry = raw.get("registry")
    if not isinstance(root_raw, str) or not root_raw.strip():
        raise ValueError("HeavenBase bootstrap root must be a non-empty path")
    root = str(Path(root_raw.strip()).expanduser())
    if not workspace:
        raise ValueError("HeavenBase bootstrap workspace is required")
    if not isinstance(backend, Mapping):
        raise ValueError("HeavenBase bootstrap backend must be a mapping")
    if registry is None:
        registry = {}
    if not isinstance(registry, Mapping):
        raise ValueError("HeavenBase bootstrap registry must be a mapping")
    backend_data = copy.deepcopy(dict(backend))
    for field in ("type", "module", "name"):
        if not isinstance(backend_data.get(field), str) or not str(backend_data[field]).strip():
            raise ValueError(f"HeavenBase bootstrap backend {field!r} is required")
        backend_data[field] = str(backend_data[field]).strip()
    if "database" in backend_data:
        database = backend_data["database"]
        if not isinstance(database, str) or not database.strip():
            raise ValueError("HeavenBase bootstrap backend 'database' must be a non-empty string")
        database = database.strip()
        if database.startswith("file:") and not database.startswith("file://"):
            raise ValueError("HeavenBase bootstrap local databases must use canonical file:// URI syntax")
        backend_data["database"] = database
    if "tags" in backend_data:
        tags = backend_data["tags"]
        if not isinstance(tags, Mapping):
            raise ValueError("HeavenBase bootstrap backend 'tags' must be a mapping")
        normalized_tags: dict[str, Any] = {}
        for identifier, value in tags.items():
            if not isinstance(identifier, str) or not identifier.strip() or identifier != identifier.strip():
                raise ValueError("HeavenBase bootstrap backend tag identifiers must be non-empty trimmed strings")
            normalized_tags[identifier] = copy.deepcopy(value)
        workspace_namespace = normalized_tags.get("workspace-database-namespace")
        if "workspace-database-namespace" in normalized_tags and not isinstance(workspace_namespace, bool):
            raise ValueError("HeavenBase bootstrap backend tag 'workspace-database-namespace' must be Boolean")
        backend_data["tags"] = normalized_tags
    registry_data = {**_REGISTRY_DEFAULTS, **copy.deepcopy(dict(registry))}
    from ..utils.registry.codec import _resolve_hash

    try:
        _resolve_hash(registry_data["hash"])
    except ValueError as error:
        raise ValueError(f"HeavenBase bootstrap registry.hash is invalid: {error}") from error
    for field in ("max_items", "max_bytes"):
        value = registry_data[field]
        if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 1):
            raise ValueError(f"HeavenBase bootstrap registry {field} must be a positive integer or null")
    return BootstrapSpec(root, workspace, backend_data, registry_data)
