"""Workspace manifest command handlers."""

from __future__ import annotations

__all__ = [
    "export_manifest",
    "import_manifest",
]

from heavenbase.utils import Any
from heavenbase.workspace import WorkspaceManifest, WorkspaceSpec


def export_manifest(context: Any, ws_id: str | None = None) -> None:
    """Export a workspace construction and schema manifest.

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
    context.out.yaml(spec.open(context=machine).to_manifest())


def import_manifest(context: Any, path: str, allow_replace: bool = False) -> None:
    """Import and register a workspace manifest file.

    Args:
        context (Any): CLI context carrying output and config services.
        path (str): Path to a JSON or YAML workspace manifest.
        allow_replace (bool): Whether an entity in the manifest may replace an
            existing schema with the same identifier, such as one stored with
            other `catalog` or `gram` options. Defaults to `False`.

    Returns:
        None: This function writes a YAML import summary to the CLI output.
    """
    manifest = WorkspaceManifest.load(path)
    machine = context.runtime_context()
    workspace = manifest.open(context=machine, allow_replace=allow_replace)
    spec = WorkspaceSpec.load(workspace.id, context=machine)
    context.out.yaml({"imported": workspace.id, "registered": spec.to_dict(), "entities": workspace.list_entities()})
