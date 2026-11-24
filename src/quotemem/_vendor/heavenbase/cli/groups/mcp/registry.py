"""Workspace access helpers for MCP CLI commands."""

from __future__ import annotations

from heavenbase.utils import Any, Dict, loads_json

__all__ = ["json_args", "load_toolkit", "registry_rows", "setup_registry", "toolkit_row"]


def _workspace(context: Any, workspace_id: str | None = None) -> Any:
    """Return the existing workspace selected for one CLI invocation."""
    runtime = getattr(context, "runtime_context", None)
    if not callable(runtime):
        raise TypeError("MCP CLI commands require a PackageCLIContext runtime authority")
    owner = runtime()
    owner._setup_modules()
    from heavenbase.workspace import HeavenBase

    workspace = HeavenBase.load(workspace_id, context=owner)
    workspace.enable_extension("toolkit")
    return workspace


def _toolkit_export(workspace: Any, module: str, name: str) -> Any:
    from heavenbase.entity.registry import load_entity_artifact_export

    return load_entity_artifact_export("sys-toolkit", module, name, resolver=workspace.module_resolver)


def load_toolkit(context: Any, ref: str, namespace: str | None = None, version: str | None = None, workspace_id: str | None = None) -> Any:
    """Load a registered Toolkit for one CLI command.

    Args:
        context (Any): CLI context with a runtime authority.
        ref (str): Toolkit id, name, or `namespace/name:version` reference.
        namespace (str | None): Optional namespace used for unqualified refs.
        version (str | None): Optional version used for unqualified refs.
        workspace_id (str | None): Existing workspace id. Defaults to the
            active or configured default workspace.

    Returns:
        Toolkit: Loaded runtime Toolkit.
    """
    workspace = _workspace(context, workspace_id)
    return _toolkit_export(workspace, "refs", "load_toolkit_ref")(
        ref,
        namespace=namespace,
        version=version,
        ws=workspace,
    )


def toolkit_row(
    context: Any,
    ref: str,
    namespace: str | None = None,
    version: str | None = None,
    workspace_id: str | None = None,
) -> Dict[str, Any]:
    """Load one Toolkit row for display or mutation.

    Args:
        context (Any): CLI context with a runtime authority.
        ref (str): Toolkit id, name, or `namespace/name:version` reference.
        namespace (str | None): Optional namespace used for unqualified refs.
        version (str | None): Optional version used for unqualified refs.
        workspace_id (str | None): Existing workspace id. Defaults to the
            active or configured default workspace.

    Returns:
        Dict[str, Any]: Toolkit row.
    """
    workspace = _workspace(context, workspace_id)
    return _toolkit_export(workspace, "refs", "describe_toolkit_ref")(
        ref,
        namespace=namespace,
        version=version,
        ws=workspace,
    )


def json_args(text: str | None) -> Dict[str, Any]:
    """Parse a CLI JSON argument object.

    Args:
        text (str | None): JSON object text. Empty input becomes `{}`.

    Returns:
        Dict[str, Any]: Parsed JSON object.

    Raises:
        TypeError: If the parsed value is not a JSON object.
    """
    data = loads_json(text or "{}")
    if not isinstance(data, dict):
        raise TypeError("Tool arguments must be a JSON object")
    return data


def registry_rows(
    context: Any,
    kind: str,
    namespace: str | None = None,
    status: str | None = "active",
    workspace_id: str | None = None,
) -> list[Dict[str, Any]]:
    """List Toolkit or Capsule rows from one workspace.

    Args:
        context (Any): CLI context with a runtime authority.
        kind (str): Row family. Supported values are `toolkit` and `capsule`.
        namespace (str | None): Optional namespace filter.
        status (str | None): Optional status filter.
        workspace_id (str | None): Existing workspace id.

    Returns:
        list[Dict[str, Any]]: Matching workspace rows.

    Raises:
        ValueError: If `kind` is unsupported.
    """
    workspace = _workspace(context, workspace_id)
    from heavenbase.entity.registry import load_entity_type

    if kind == "toolkit":
        entity = load_entity_type("sys-toolkit", resolver=workspace.module_resolver)
    elif kind == "capsule":
        entity = load_entity_type("sys-capsule", resolver=workspace.module_resolver)
    else:
        raise ValueError(f"Unknown MCP registry row kind {kind!r}")
    rows = workspace.rows(entity)
    if namespace is not None:
        rows = [row for row in rows if row.get("namespace") == namespace]
    if status is not None:
        rows = [row for row in rows if row.get("status") == status]
    return rows


def setup_registry(context: Any, workspace_id: str | None = None) -> str:
    """Enable Toolkit persistence in one existing workspace.

    Args:
        context (Any): CLI context with a runtime authority.
        workspace_id (str | None): Existing workspace id.

    Returns:
        str: Workspace identifier.
    """
    return str(_workspace(context, workspace_id).id)


def delete_toolkit(context: Any, toolkit_id: str, workspace_id: str | None = None, hard: bool = False) -> bool:
    """Delete or tombstone one Toolkit row.

    Args:
        context (Any): CLI context with a runtime authority.
        toolkit_id (str): Toolkit object id.
        workspace_id (str | None): Existing workspace id.
        hard (bool): Whether to delete instead of tombstone.

    Returns:
        bool: Whether the row was found and removed.
    """
    workspace = _workspace(context, workspace_id)
    from heavenbase.entity.registry import load_entity_type

    toolkit_type = load_entity_type("sys-toolkit", resolver=workspace.module_resolver)
    return toolkit_type.delete(toolkit_id, ws=workspace, tombstone=not hard)
