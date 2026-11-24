"""Workspace preset command handlers."""

from __future__ import annotations

__all__ = ["browse_presets"]

from heavenbase.utils import Any, Dict, List
from heavenbase.workspace.presets import list_workspace_presets, resolve_workspace_preset


def _preset_rows(presets: Dict[str, Dict[str, Any]]) -> List[List[str]]:
    return [[preset, cfg.get("desc", ""), ", ".join((cfg.get("backends") or {}).keys())] for preset, cfg in presets.items()]


def browse_presets(context: Any, action: str | None = None, name: str | None = None) -> None:
    """List workspace presets or show one preset.

    Args:
        context (Any): CLI context carrying output and config services.
        action (str | None): `list`, `ls`, `show`, or a preset name. `None` lists presets.
        name (str | None): Preset name when `action` is `show`.

    Returns:
        None: This function writes a table or YAML to the CLI output.

    Raises:
        ValueError: If `show` is requested without a preset name.
    """
    action = action or "list"
    if action not in ("list", "ls", "show"):
        name, action = action, "show"
    presets = list_workspace_presets(manager=context.cm)
    if action in ("list", "ls"):
        context.out.table(["preset", "desc", "backends"], _preset_rows(presets), title="Workspace Presets")
        return
    if not name:
        raise ValueError("preset name is required for 'hb ws presets show <name>'")
    context.out.yaml(resolve_workspace_preset(name, manager=context.cm).to_dict())
