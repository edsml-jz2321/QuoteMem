"""Command handlers for workspace-owned MCP Toolkits."""

from __future__ import annotations

from heavenbase.utils import Any

from .registry import delete_toolkit, json_args, load_toolkit, registry_rows, setup_registry, toolkit_row

__all__ = ["call_tool", "capsules", "list_toolkits", "mcp_json", "remove", "serve", "serve_workspace", "setup", "show", "stdio", "tools"]


def list_toolkits(
    context: Any,
    namespace: str | None = None,
    version: str | None = None,
    status: str | None = "active",
    workspace_id: str | None = None,
    as_json: bool = False,
) -> None:
    """List registered Toolkits.

    Args:
        context (Any): CLI context with an output adapter.
        namespace (str | None): Optional Toolkit namespace filter.
        version (str | None): Accepted for shared MCP options; unused here.
        status (str | None): Optional status filter.
        workspace_id (str | None): Existing workspace id.
        as_json (bool): Whether to print JSON instead of a table.

    Returns:
        None: This function writes to the CLI output adapter.
    """
    del version
    rows = registry_rows(context, "toolkit", namespace=namespace, status=status, workspace_id=workspace_id)
    if as_json:
        context.out.json(rows)
        return
    context.out.table(
        ["id", "namespace", "name", "version", "tools"],
        [[row.get("object_id"), row.get("namespace"), row.get("name"), row.get("version"), len(row.get("capsule_refs") or [])] for row in rows],
    )


def capsules(
    context: Any,
    namespace: str | None = None,
    version: str | None = None,
    status: str | None = "active",
    workspace_id: str | None = None,
    as_json: bool = False,
) -> None:
    """List registered Capsules.

    Args:
        context (Any): CLI context with an output adapter.
        namespace (str | None): Optional Capsule namespace filter.
        version (str | None): Accepted for shared MCP options; unused here.
        status (str | None): Optional status filter.
        workspace_id (str | None): Existing workspace id.
        as_json (bool): Whether to print JSON instead of a table.

    Returns:
        None: This function writes to the CLI output adapter.
    """
    del version
    rows = registry_rows(context, "capsule", namespace=namespace, status=status, workspace_id=workspace_id)
    if as_json:
        context.out.json(rows)
        return
    context.out.table(
        ["id", "namespace", "name", "version", "status"],
        [[row.get("object_id"), row.get("namespace"), row.get("name"), row.get("version"), row.get("status")] for row in rows],
    )


def show(
    context: Any,
    ref: str,
    namespace: str | None = None,
    version: str | None = None,
    workspace_id: str | None = None,
    as_json: bool = False,
) -> None:
    """Show one Toolkit row.

    Args:
        context (Any): CLI context with an output adapter.
        ref (str): Toolkit id, name, or qualified reference.
        namespace (str | None): Optional namespace for unqualified refs.
        version (str | None): Optional version for unqualified refs.
        workspace_id (str | None): Existing workspace id.
        as_json (bool): Whether to print JSON instead of YAML.

    Returns:
        None: This function writes to the CLI output adapter.
    """
    row = toolkit_row(context, ref, namespace=namespace, version=version, workspace_id=workspace_id)
    context.out.json(row) if as_json else context.out.yaml(row)


def tools(
    context: Any,
    ref: str,
    namespace: str | None = None,
    version: str | None = None,
    workspace_id: str | None = None,
    as_json: bool = False,
) -> None:
    """List tools in one Toolkit.

    Args:
        context (Any): CLI context with an output adapter.
        ref (str): Toolkit id, name, or qualified reference.
        namespace (str | None): Optional namespace for unqualified refs.
        version (str | None): Optional version for unqualified refs.
        workspace_id (str | None): Existing workspace id.
        as_json (bool): Whether to print JSON instead of a table.

    Returns:
        None: This function writes to the CLI output adapter.
    """
    toolkit = load_toolkit(context, ref, namespace=namespace, version=version, workspace_id=workspace_id)
    rows = [{"name": name, "description": tool.description, "input_schema": tool.input_schema} for name, tool in toolkit.tools.items()]
    if as_json:
        context.out.json(rows)
        return
    context.out.table(["name", "description"], [[row["name"], row["description"]] for row in rows])


def call_tool(
    context: Any,
    ref: str,
    tool: str,
    args: str = "{}",
    namespace: str | None = None,
    version: str | None = None,
    workspace_id: str | None = None,
    as_json: bool = False,
) -> None:
    """Call one Toolkit tool with JSON object arguments.

    Args:
        context (Any): CLI context with an output adapter.
        ref (str): Toolkit id, name, or qualified reference.
        tool (str): Tool name to run.
        args (str): JSON object argument text.
        namespace (str | None): Optional namespace for unqualified refs.
        version (str | None): Optional version for unqualified refs.
        workspace_id (str | None): Existing workspace id.
        as_json (bool): Whether to force JSON output.

    Returns:
        None: This function writes to the CLI output adapter.
    """
    value = load_toolkit(context, ref, namespace=namespace, version=version, workspace_id=workspace_id).run(tool, **json_args(args))
    context.out.json(value) if as_json or isinstance(value, (dict, list)) else context.out.echo(value)


def remove(
    context: Any,
    ref: str,
    namespace: str | None = None,
    version: str | None = None,
    workspace_id: str | None = None,
    hard: bool = False,
) -> None:
    """Remove one Toolkit from a workspace.

    Args:
        context (Any): CLI context with an output adapter.
        ref (str): Toolkit id, name, or qualified reference.
        namespace (str | None): Optional namespace for unqualified refs.
        version (str | None): Optional version for unqualified refs.
        workspace_id (str | None): Existing workspace id.
        hard (bool): Whether to delete instead of tombstone.

    Returns:
        None: This function writes to the CLI output adapter.
    """
    row = toolkit_row(context, ref, namespace=namespace, version=version, workspace_id=workspace_id)
    toolkit_id = row["object_id"]
    ok = delete_toolkit(context, toolkit_id, workspace_id=workspace_id, hard=hard)
    context.out.success(f"removed toolkit {toolkit_id}") if ok else context.out.warning(f"toolkit {toolkit_id} was not found")


def mcp_json(
    context: Any,
    ref: str,
    namespace: str | None = None,
    version: str | None = None,
    workspace_id: str | None = None,
    transport: str = "http",
    host: str | None = None,
    port: int | None = None,
) -> None:
    """Print MCP client configuration JSON for one Toolkit.

    Args:
        context (Any): CLI context with an output adapter.
        ref (str): Toolkit id, name, or qualified reference.
        namespace (str | None): Optional namespace for unqualified refs.
        version (str | None): Optional version for unqualified refs.
        workspace_id (str | None): Existing workspace id.
        transport (str): MCP transport name.
        host (str | None): Optional MCP host.
        port (int | None): Optional MCP port.

    Returns:
        None: This function writes to the CLI output adapter.
    """
    toolkit = load_toolkit(context, ref, namespace=namespace, version=version, workspace_id=workspace_id)
    context.out.echo(toolkit.to_mcp_json(transport=transport, host=host, port=port, workspace=workspace_id))


def serve(
    context: Any,
    ref: str,
    namespace: str | None = None,
    version: str | None = None,
    workspace_id: str | None = None,
    transport: str | None = None,
    host: str | None = None,
    port: int | None = None,
) -> None:
    """Serve a registered Toolkit over MCP.

    Args:
        context (Any): CLI context with an output adapter.
        ref (str): Toolkit id, name, or qualified reference.
        namespace (str | None): Optional namespace for unqualified refs.
        version (str | None): Optional version for unqualified refs.
        workspace_id (str | None): Existing workspace id.
        transport (str | None): Optional MCP transport override.
        host (str | None): Optional MCP host.
        port (int | None): Optional MCP port.

    Returns:
        None: This function blocks while serving the MCP server.
    """
    toolkit = load_toolkit(context, ref, namespace=namespace, version=version, workspace_id=workspace_id)
    toolkit.serve(transport=transport, host=host, port=port)


def serve_workspace(
    context: Any,
    ws_id: str | None = None,
    profile: str | None = None,
    transport: str | None = None,
    host: str | None = None,
    port: int | None = None,
) -> None:
    """Serve one registered workspace/profile directly over MCP.

    Args:
        context: CLI context with the owning runtime Context.
        ws_id: Registered workspace id. ``None`` selects active/default.
        profile: Explicit workspace MCP profile.
        transport: Optional MCP transport override.
        host: Optional MCP host override.
        port: Optional MCP port override.

    Returns:
        None: This function blocks while serving the workspace MCP server.

    Raises:
        ValueError: If no profile is designated.
        KeyError: If the workspace is not registered.
    """
    from heavenbase.workspace import WorkspaceSpec

    selected_profile = str(profile or "").strip()
    if not selected_profile:
        raise ValueError("workspace MCP serving requires --profile")
    machine = context.runtime_context()
    workspace = WorkspaceSpec.load(ws_id, context=machine).open(context=machine)
    workspace.serve(profile=selected_profile, transport=transport, host=host, port=port)


def stdio(
    context: Any,
    ref: str,
    namespace: str | None = None,
    version: str | None = None,
    workspace_id: str | None = None,
) -> None:
    """Serve a registered Toolkit over MCP stdio.

    Args:
        context (Any): CLI context with an output adapter.
        ref (str): Toolkit id, name, or qualified reference.
        namespace (str | None): Optional namespace for unqualified refs.
        version (str | None): Optional version for unqualified refs.
        workspace_id (str | None): Existing workspace id.

    Returns:
        None: This function blocks while serving stdio.
    """
    load_toolkit(context, ref, namespace=namespace, version=version, workspace_id=workspace_id).serve(transport="stdio")


def setup(context: Any, workspace_id: str | None = None) -> None:
    """Enable Toolkit persistence in one existing workspace.

    Args:
        context (Any): CLI context with an output adapter.
        workspace_id (str | None): Existing workspace id.

    Returns:
        None: This function writes to the CLI output adapter.
    """
    context.out.success(f"mcp toolkit storage enabled in workspace {setup_registry(context, workspace_id)}")
