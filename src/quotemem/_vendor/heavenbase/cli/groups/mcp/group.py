"""Command-spec assembly for the MCP CLI group."""

from __future__ import annotations

from heavenbase.cli.spec import ArgumentSpec, CommandGroupSpec, CommandSpec, OptionSpec

from .commands import call_tool, capsules, list_toolkits, mcp_json, remove, serve, serve_workspace, setup, show, stdio, tools

__all__ = ["mcp_group"]


def _common_options() -> list[OptionSpec]:
    return [
        OptionSpec.from_flags("namespace", "--namespace", "-n", help="Toolkit/Capsule namespace."),
        OptionSpec.from_flags("version", "--version", help="Toolkit version."),
        OptionSpec.from_flags("workspace_id", "--workspace", "-w", help="Existing workspace id."),
    ]


def mcp_group() -> CommandGroupSpec:
    """Return MCP command group specs.

    Args:
        None.

    Returns:
        CommandGroupSpec: CLI command group for registered Capsule-backed Toolkits.
    """
    json_opt = OptionSpec.from_flags("as_json", "--json", "-j", help="Print JSON.", default=False, is_flag=True, type=bool)
    status_opt = OptionSpec.from_flags("status", "--status", help="Registry status filter.", default="active")
    workspace_opt = OptionSpec.from_flags("workspace_id", "--workspace", "-w", help="Existing workspace id.")
    transport_opt = OptionSpec.from_flags("transport", "--transport", "-t", help="MCP transport.", default="http")
    host_opt = OptionSpec.from_flags("host", "--host", help="MCP host.")
    port_opt = OptionSpec.from_flags("port", "--port", "-p", help="MCP port.", type=int)
    profile_opt = OptionSpec.from_flags("profile", "--profile", help="Explicit workspace MCP profile.")
    return CommandGroupSpec(
        "mcp",
        help="Serve workspaces and manage Capsule-backed MCP Toolkits.",
        commands=[
            CommandSpec("setup", setup, help="Enable Toolkit storage in a workspace.", options=[workspace_opt]),
            CommandSpec(
                "workspace",
                serve_workspace,
                help="Serve a registered workspace/profile directly over MCP.",
                args=[ArgumentSpec("ws_id", help="Workspace id; omit for active/default.", required=False)],
                options=[profile_opt, transport_opt, host_opt, port_opt],
            ),
            CommandSpec("list", list_toolkits, help="List registered Toolkits.", aliases=["ls"], options=[*_common_options(), status_opt, json_opt]),
            CommandSpec("capsules", capsules, help="List registered Capsules.", options=[*_common_options(), status_opt, json_opt]),
            CommandSpec(
                "show",
                show,
                help="Show one Toolkit record.",
                aliases=["browse", "get"],
                args=[ArgumentSpec("ref", help="Toolkit id, name, or namespace/name:version.")],
                options=[*_common_options(), json_opt],
            ),
            CommandSpec(
                "tools",
                tools,
                help="List tools in a Toolkit.",
                args=[ArgumentSpec("ref", help="Toolkit id, name, or namespace/name:version.")],
                options=[*_common_options(), json_opt],
            ),
            CommandSpec(
                "call",
                call_tool,
                help="Call one Toolkit tool with JSON object args.",
                args=[ArgumentSpec("ref", help="Toolkit id, name, or namespace/name:version."), ArgumentSpec("tool", help="Tool name.")],
                options=[OptionSpec.from_flags("args", "--args", "-a", help="JSON object arguments.", default="{}"), *_common_options(), json_opt],
            ),
            CommandSpec(
                "remove",
                remove,
                help="Remove a Toolkit from the registry.",
                aliases=["rm", "del", "delete"],
                args=[ArgumentSpec("ref", help="Toolkit id, name, or namespace/name:version.")],
                options=[
                    *_common_options(),
                    OptionSpec.from_flags("hard", "--hard", help="Delete instead of tombstone.", default=False, is_flag=True, type=bool),
                ],
            ),
            CommandSpec(
                "mcp-json",
                mcp_json,
                help="Print MCP client config JSON for a Toolkit.",
                aliases=["config"],
                args=[ArgumentSpec("ref", help="Toolkit id, name, or namespace/name:version.")],
                options=[*_common_options(), transport_opt, host_opt, port_opt],
            ),
            CommandSpec(
                "stdio",
                stdio,
                help="Serve a registered Toolkit over MCP stdio.",
                args=[ArgumentSpec("ref", help="Toolkit id, name, or namespace/name:version.")],
                options=[*_common_options()],
            ),
            CommandSpec(
                "serve",
                serve,
                help="Serve a Toolkit as MCP.",
                args=[ArgumentSpec("ref", help="Toolkit id, name, or namespace/name:version.")],
                options=[
                    *_common_options(),
                    transport_opt,
                    host_opt,
                    port_opt,
                ],
            ),
        ],
    )
