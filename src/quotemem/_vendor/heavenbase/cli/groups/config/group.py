"""Config CLI command specifications."""

from __future__ import annotations

__all__ = ["config_group"]

from ...spec import ArgumentSpec, CommandGroupSpec, CommandSpec, OptionSpec
from .maintain import compact_config, edit_config, open_config, reset_config
from .read import get_config, list_config
from .version import diff_config, history_config, list_scopes
from .write import add_config, copy_config, set_config, unset_config


def _scope_option() -> OptionSpec:
    """Return the shared config-scope option."""
    return OptionSpec.from_flags("scope", "--scope", "-s", help="Config scope.")


def _json_option() -> OptionSpec:
    """Return the shared JSON input/output option."""
    return OptionSpec.from_flags("as_json", "--json", "-j", help="Use JSON parsing/output.", default=False, is_flag=True, type=bool)


def config_group() -> CommandGroupSpec:
    """Return the built-in config command group specification.

    Args:
        None.

    Returns:
        CommandGroupSpec: Backend-neutral CLI command metadata for config
            inspection, mutation, versioning, and maintenance commands.
    """
    scope_opt = _scope_option()
    json_opt = _json_option()
    return CommandGroupSpec(
        "config",
        help="Inspect and edit HeavenBase config.",
        aliases=["cfg"],
        commands=[
            CommandSpec(
                "get",
                get_config,
                help="Get merged config or one key.",
                args=[ArgumentSpec("key", help="Config key.", required=False)],
                options=[scope_opt, OptionSpec.from_flags("version", "--version", "-v", help="Config version.", type=int), json_opt],
            ),
            CommandSpec(
                "list",
                list_config,
                help="List flattened config keys.",
                aliases=["ls", "show"],
                args=[ArgumentSpec("prefix", help="Key prefix.", required=False)],
                options=[scope_opt, json_opt],
            ),
            CommandSpec(
                "set",
                set_config,
                help="Set one config key.",
                args=[ArgumentSpec("key", help="Config key."), ArgumentSpec("value", help="Config value.")],
                options=[scope_opt, json_opt],
            ),
            CommandSpec(
                "add",
                add_config,
                help="Append to one config list key.",
                args=[ArgumentSpec("key", help="Config list key."), ArgumentSpec("value", help="Config value.")],
                options=[scope_opt, json_opt],
            ),
            CommandSpec(
                "unset",
                unset_config,
                help="Unset one config key.",
                aliases=["remove", "rm", "del", "delete"],
                args=[ArgumentSpec("key", help="Config key.")],
                options=[scope_opt],
            ),
            CommandSpec(
                "copy",
                copy_config,
                help="Copy config from default/base/scope.",
                aliases=["cp"],
                args=[ArgumentSpec("key", help="Config key.", required=False)],
                options=[
                    OptionSpec.from_flags("source", "--source", help="Source: base or default.", default="base"),
                    OptionSpec.from_flags("from_scope", "--from-scope", help="Source scope."),
                    OptionSpec.from_flags("from_version", "--from-version", help="Source version.", type=int),
                    OptionSpec.from_flags("target_scope", "--target-scope", help="Target scope."),
                ],
            ),
            CommandSpec(
                "history",
                history_config,
                help="Show config history.",
                aliases=["hist"],
                options=[scope_opt, OptionSpec.from_flags("limit", "--limit", "-n", help="Max rows.", default=10, type=int), json_opt],
            ),
            CommandSpec("scope", list_scopes, help="List config scopes."),
            CommandSpec(
                "diff",
                diff_config,
                help="Diff two config snapshots.",
                options=[
                    OptionSpec.from_flags("scope_a", "--scope-a", help="From scope."),
                    OptionSpec.from_flags("version_a", "--version-a", help="From version.", type=int),
                    OptionSpec.from_flags("scope_b", "--scope-b", help="To scope."),
                    OptionSpec.from_flags("version_b", "--version-b", help="To version.", type=int),
                ],
            ),
            CommandSpec(
                "edit",
                edit_config,
                help="Edit one config layer.",
                options=[scope_opt, OptionSpec.from_flags("editor", "--editor", "-e", help="Editor command.")],
            ),
            CommandSpec("open", open_config, help="Open one config layer as output.", options=[scope_opt]),
            CommandSpec("reset", reset_config, help="Reset one config scope.", options=[scope_opt]),
            CommandSpec("compact", compact_config, help="Compact retained config versions.", options=[scope_opt]),
        ],
    )
