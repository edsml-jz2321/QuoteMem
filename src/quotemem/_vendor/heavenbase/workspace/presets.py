"""Workspace backend preset resolution and Apply profiles."""

from __future__ import annotations

__all__ = [
    "WorkspacePreset",
    "WorkspacePresetEngine",
    "list_workspace_presets",
    "resolve_workspace_backends",
    "resolve_workspace_preset",
]

from collections.abc import Mapping
from typing import Any

from heavenbase.entity import Entity, field
from heavenbase.types import Json, MediumText, ShortText
from heavenbase.utils.config import CM_HVNB, ConfigEngine, ConfigSpec
from heavenbase.utils.config.data import to_mutable_data


def _apply_value(value: Any, ws_id: str | None) -> Any:
    if isinstance(value, str):
        return value.replace("{ws_id}", ws_id) if ws_id else value
    if isinstance(value, Mapping):
        return {key: _apply_value(item, ws_id) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_apply_value(item, ws_id) for item in value]
    return value


class WorkspacePreset(Entity):
    """Describe one named backend configuration accepted by a workspace."""

    identifier = "workspace-preset"

    preset = field(ShortText).optional().desc("Selected workspace preset name")
    desc = field(MediumText).default("").desc("Human-readable preset purpose")
    backends = field(Json).required().desc("Backend config map passed to BackendRegistry")
    args = field(Json).default({}).desc("Extra preset metadata for docs and tooling")


class WorkspacePresetEngine(ConfigEngine):
    """Resolve workspace presets through the universal config pipeline.

    Args:
        manager (Any): Config manager that owns ``heavenbase.workspace``.
        root (Mapping[str, Any] | None): Optional frozen workspace-root overlay.
    """

    domain = "workspace-preset"
    config_path = "heavenbase.workspace"
    schema = WorkspacePreset
    extra = "allow"

    def __init__(self, manager: Any = CM_HVNB, root: Mapping[str, Any] | None = None) -> None:
        """Construct the workspace preset resolution engine.

        Returns:
            None: This initializer does not return a value.
        """
        super().__init__(manager=manager, root=root)
        self.register_profile("backends", self._apply_backends)
        self.register_profile("preset", self._apply_preset)

    def _normalize(self, data: Mapping[str, Any], *, context: Any) -> Mapping[str, Any]:
        return data

    def _validate(self, spec: ConfigSpec, *, context: Any) -> None:
        backends = spec.get("backends")
        if not isinstance(backends, Mapping) or not backends:
            raise ValueError(f"workspace-preset {spec.get('preset')!r} must define a non-empty backends map")

    def _default_profile(self, spec: ConfigSpec) -> str:
        return "backends"

    @staticmethod
    def _apply_preset(spec: ConfigSpec, *, ws_id: str | None = None) -> dict[str, Any]:
        data = to_mutable_data(spec.data)
        data["backends"] = _apply_value(data["backends"], ws_id)
        return data

    @staticmethod
    def _apply_backends(spec: ConfigSpec, *, ws_id: str | None = None) -> dict[str, Any]:
        return _apply_value(spec.get("backends"), ws_id)


def resolve_workspace_preset(config: Any = None, *, manager: Any = CM_HVNB) -> ConfigSpec:
    """Resolve one workspace preset without applying runtime placeholders.

    Args:
        config (Any): Preset identifier, override mapping, compatible spec, or
            ``None`` for the configured default.
        manager (Any): Config manager that owns ``heavenbase.workspace``.

    Returns:
        ConfigSpec: Canonical preset data with raw construction provenance.
    """
    raw = {"preset": config} if isinstance(config, str) else config
    return WorkspacePresetEngine(manager=manager).resolve(raw)


def resolve_workspace_backends(config: Any = None, *, ws_id: str | None = None, manager: Any = CM_HVNB) -> dict[str, Any]:
    """Apply one workspace preset to a backend config mapping.

    Args:
        config (Any): Preset identifier, override mapping, compatible spec, or
            ``None`` for the configured default.
        ws_id (str | None): Workspace identifier used for ``{ws_id}``
            placeholders.
        manager (Any): Config manager that owns ``heavenbase.workspace``.

    Returns:
        dict[str, Any]: Independent backend config mapping.
    """
    raw = {"preset": config} if isinstance(config, str) else config
    return WorkspacePresetEngine(manager=manager).apply(raw, ws_id=ws_id)


def list_workspace_presets(manager: Any = CM_HVNB) -> dict[str, dict[str, Any]]:
    """Return mutable copies of configured workspace preset templates.

    Args:
        manager (Any): Config manager that owns ``heavenbase.workspace``.

    Returns:
        dict[str, dict[str, Any]]: Preset templates keyed by identifier.
    """
    return to_mutable_data(WorkspacePresetEngine(manager=manager).presets())
