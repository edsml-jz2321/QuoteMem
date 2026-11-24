"""Workspace runtime command handlers."""

from __future__ import annotations

__all__ = [
    "create_workspace",
    "export_workspace_skill",
    "health_workspace",
    "open_workspace",
]

from heavenbase.utils import Any, List
from heavenbase.workspace import HeavenBase, WorkspaceSpec
from heavenbase.workspace.presets import resolve_workspace_preset
from heavenbase.workspace.setup.lifecycle import _workspace_config_scope


def _backend_names(workspace: Any) -> List[str]:
    return [name for name, _ in workspace.backends.items()]


def create_workspace(context: Any, ws_id: str, preset: str | None = None) -> None:
    """Create a workspace from a preset and register it.

    Args:
        context (Any): CLI context carrying output and config services.
        ws_id (str): Workspace id to create and register.
        preset (str | None): Optional workspace backend preset. `None` uses the configured default preset.

    Returns:
        None: This function writes a YAML creation summary to the CLI output.
    """
    machine = context.runtime_context()
    selected = preset
    if selected is None:
        with _workspace_config_scope(machine, ws_id):
            selected = resolve_workspace_preset(manager=machine.config).get("preset")
    workspace = HeavenBase(ws_id, context=machine, preset=selected)
    selected = getattr(workspace, "preset", None) or preset
    spec = WorkspaceSpec.load(ws_id, context=machine)
    context.out.yaml(
        {
            "created": workspace.id,
            "preset": selected,
            "registered": spec.to_dict(),
            "backend_names": _backend_names(workspace),
        }
    )


def open_workspace(context: Any, ws_id: str | None = None) -> None:
    """Open a registered workspace and print a load summary.

    Args:
        context (Any): CLI context carrying output and config services.
        ws_id (str | None): Workspace id. `None` resolves the active or default workspace.

    Returns:
        None: This function writes YAML to the CLI output.

    Raises:
        KeyError: If the requested workspace is not registered.
    """
    machine = context.runtime_context()
    spec = WorkspaceSpec.load(ws_id, context=machine)
    workspace = spec.open(context=machine)
    context.out.yaml(
        {
            "opened": workspace.id,
            "registered": spec.to_dict(),
            "entities": sorted(workspace.entities),
            "backend_names": _backend_names(workspace),
        }
    )


def health_workspace(context: Any, ws_id: str | None = None) -> None:
    """Print registry metadata and runtime health for a workspace.

    Args:
        context (Any): CLI context carrying output and config services.
        ws_id (str | None): Workspace id. `None` resolves the active or default workspace.

    Returns:
        None: This function writes YAML to the CLI output.

    Raises:
        KeyError: If the requested workspace is not registered.
    """
    machine = context.runtime_context()
    spec = WorkspaceSpec.load(ws_id, context=machine)
    workspace = spec.open(context=machine)
    context.out.yaml({"registered": spec.to_dict(), "runtime": workspace.health()})


def export_workspace_skill(
    context: Any,
    ws_id: str | None = None,
    profile: str | None = None,
    name: str | None = None,
    url: str | None = None,
    host: str | None = None,
    port: int | None = None,
    output: str | None = None,
    root: str | None = None,
    force: bool = False,
) -> None:
    """Generate and install or export one workspace MCP skill.

    Args:
        context: CLI context carrying output and runtime services.
        ws_id: Registered workspace id. ``None`` selects active/default.
        profile: Explicit MCP profile exposed to the skill.
        name: Optional kebab-case skill name.
        url: Optional credential-free streamable-HTTP MCP URL.
        host: Optional local MCP host used when ``url`` is omitted.
        port: Optional local MCP port used when ``url`` is omitted.
        output: Exact export folder. When omitted, install under the skill root.
        root: Optional Codex skill root used for installation.
        force: Replace an existing exact skill folder.

    Returns:
        None: This function writes a YAML artifact summary.

    Raises:
        ValueError: If no profile is designated.
        KeyError: If the workspace is not registered.
    """
    selected_profile = str(profile or "").strip()
    if not selected_profile:
        raise ValueError("workspace skill generation requires --profile")
    machine = context.runtime_context()
    spec = WorkspaceSpec.load(ws_id, context=machine)
    workspace = spec.open(context=machine)
    skill = workspace.to_skill(profile=selected_profile, name=name, url=url, host=host, port=port)
    path = skill.save(output, replace=force) if output else skill.install(root=root, replace=force)
    context.out.yaml({"installed": output is None, "path": path, **skill.to_dict()})
