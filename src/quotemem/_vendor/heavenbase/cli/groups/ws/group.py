"""Workspace command group specification."""

from __future__ import annotations

__all__ = ["ws_group"]

from ...spec import ArgumentSpec, CommandGroupSpec, CommandSpec, OptionSpec
from .manifest import export_manifest, import_manifest
from .presets import browse_presets
from .registry import (
    activate_workspace,
    deactivate_workspace,
    get_workspace,
    list_workspaces,
    register_workspace,
    unregister_workspace,
)
from .runtime import create_workspace, export_workspace_skill, health_workspace, open_workspace


def ws_group() -> CommandGroupSpec:
    """Return the workspace command group specification.

    Args:
        None.

    Returns:
        CommandGroupSpec: CLI command group for browsing and managing registered workspaces.
    """
    preset_opt = OptionSpec.from_flags("preset", "--preset", "-p", help="Workspace backend preset.")
    return CommandGroupSpec(
        "ws",
        help="Browse and manage globally registered workspaces.",
        commands=[
            CommandSpec("list", list_workspaces, help="List registered workspaces.", aliases=["ls"]),
            CommandSpec("get", get_workspace, help="Show one workspace registry record.", args=[ArgumentSpec("ws_id", help="Workspace id.", required=False)]),
            CommandSpec(
                "create",
                create_workspace,
                help="Create a workspace from a backend preset and register it.",
                args=[ArgumentSpec("ws_id", help="Workspace id.")],
                options=[preset_opt],
            ),
            CommandSpec(
                "set",
                register_workspace,
                help="Register or update a workspace.",
                aliases=["add"],
                args=[ArgumentSpec("ws_id", help="Workspace id.")],
            ),
            CommandSpec(
                "presets",
                browse_presets,
                help="List workspace presets or show one preset.",
                aliases=["preset"],
                args=[
                    ArgumentSpec("action", help="list, show, or preset name.", required=False),
                    ArgumentSpec("name", help="Preset name for show.", required=False),
                ],
            ),
            CommandSpec(
                "activate",
                activate_workspace,
                help="Activate a registered workspace.",
                aliases=["act", "use"],
                args=[ArgumentSpec("ws_id", help="Workspace id.")],
            ),
            CommandSpec("deactivate", deactivate_workspace, help="Deactivate the current workspace.", aliases=["deact"]),
            CommandSpec(
                "unset",
                unregister_workspace,
                help="Unregister a workspace.",
                aliases=["remove", "rm", "del", "delete"],
                args=[ArgumentSpec("ws_id", help="Workspace id.")],
            ),
            CommandSpec(
                "open",
                open_workspace,
                help="Instantiate a registered workspace and show a load summary.",
                args=[ArgumentSpec("ws_id", help="Workspace id.", required=False)],
            ),
            CommandSpec(
                "health",
                health_workspace,
                help="Show registry metadata plus runtime workspace health.",
                args=[ArgumentSpec("ws_id", help="Workspace id.", required=False)],
            ),
            CommandSpec(
                "skill",
                export_workspace_skill,
                help="Generate and install an MCP-backed skill for a workspace profile.",
                args=[ArgumentSpec("ws_id", help="Workspace id; omit for active/default.", required=False)],
                options=[
                    OptionSpec.from_flags("profile", "--profile", help="Explicit workspace MCP profile."),
                    OptionSpec.from_flags("name", "--name", help="Kebab-case skill name."),
                    OptionSpec.from_flags("url", "--url", help="Credential-free streamable-HTTP MCP URL."),
                    OptionSpec.from_flags("host", "--host", help="Local MCP host used when URL is omitted."),
                    OptionSpec.from_flags("port", "--port", help="Local MCP port used when URL is omitted.", type=int),
                    OptionSpec.from_flags("output", "--output", "-o", help="Exact export folder instead of installation."),
                    OptionSpec.from_flags("root", "--root", help="Skill installation root."),
                    OptionSpec.from_flags("force", "--force", help="Replace an existing exact skill folder.", default=False, is_flag=True, type=bool),
                ],
            ),
            CommandSpec(
                "manifest",
                export_manifest,
                help="Export a workspace construction and schema manifest.",
                args=[ArgumentSpec("ws_id", help="Workspace id.", required=False)],
            ),
            CommandSpec(
                "import",
                import_manifest,
                help="Import a workspace construction and schema manifest.",
                args=[ArgumentSpec("path", help="Manifest JSON or YAML path.")],
                options=[
                    OptionSpec.from_flags(
                        "allow_replace",
                        "--allow-replace",
                        help="Let manifest entities replace existing schemas with the same identifier, as allow_replace=True does.",
                        default=False,
                        is_flag=True,
                        type=bool,
                    ),
                ],
            ),
        ],
    )
