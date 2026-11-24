"""Workspace registry command handlers."""

from __future__ import annotations

__all__ = [
    "activate_workspace",
    "deactivate_workspace",
    "get_workspace",
    "list_workspaces",
    "register_workspace",
    "unregister_workspace",
]

from heavenbase.workspace import WorkspaceSpec
from heavenbase.utils import Any, List


def workspace_rows(context: Any) -> List[List[str]]:
    """Return display rows for registered workspaces.

    Args:
        context (Any): CLI context carrying the config manager.

    Returns:
        List[List[str]]: Table rows containing active marker, workspace id, and update time.
    """
    return [["*" if spec.active else "", spec.id, spec.updated_at] for spec in WorkspaceSpec.list(context=context.runtime_context())]


def list_workspaces(context: Any) -> None:
    """Print registered workspace records.

    Args:
        context (Any): CLI context carrying output and config services.

    Returns:
        None: This function writes a table to the CLI output.
    """
    context.out.table(["", "id", "updated_at"], workspace_rows(context), title="Workspaces")


def get_workspace(context: Any, ws_id: str | None = None) -> None:
    """Print one workspace registry record.

    Args:
        context (Any): CLI context carrying output and config services.
        ws_id (str | None): Workspace id. `None` resolves the active or default workspace.

    Returns:
        None: This function writes YAML to the CLI output.

    Raises:
        KeyError: If the requested workspace is not registered.
    """
    context.out.yaml(WorkspaceSpec.load(ws_id, context=context.runtime_context()).to_dict())


def register_workspace(context: Any, ws_id: str) -> None:
    """Register or update one workspace registry record.

    Args:
        context (Any): CLI context carrying output and config services.
        ws_id (str): Workspace id to register.

    Returns:
        None: This function writes a success message to the CLI output.
    """
    spec = WorkspaceSpec.register(ws_id, context=context.runtime_context())
    context.out.success(f"registered workspace {spec.id}")


def activate_workspace(context: Any, ws_id: str) -> None:
    """Mark one registered workspace as active.

    Args:
        context (Any): CLI context carrying output and config services.
        ws_id (str): Workspace id to activate.

    Returns:
        None: This function writes a success message to the CLI output.

    Raises:
        KeyError: If the workspace is not registered.
    """
    machine = context.runtime_context()
    machine.activate_workspace(ws_id)
    spec = WorkspaceSpec.load(ws_id, context=machine)
    context.out.success(f"active workspace is {spec.id}")


def deactivate_workspace(context: Any) -> None:
    """Clear the active workspace marker.

    Args:
        context (Any): CLI context carrying output and config services.

    Returns:
        None: This function writes a success message to the CLI output.
    """
    machine = context.runtime_context()
    previous = machine.active_workspace_id
    machine.deactivate_workspace()
    context.out.success(f"deactivated workspace {previous}" if previous else "no active workspace")


def unregister_workspace(context: Any, ws_id: str) -> None:
    """Remove one workspace registry record.

    Args:
        context (Any): CLI context carrying output and config services.
        ws_id (str): Workspace id to unregister.

    Returns:
        None: This function writes a success message to the CLI output.
    """
    removed = WorkspaceSpec.unregister(ws_id, context=context.runtime_context())
    context.out.success(f"removed workspace {ws_id}" if removed else f"workspace {ws_id} was not registered")
