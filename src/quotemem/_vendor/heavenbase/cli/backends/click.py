"""Click backend compiler for HeavenBase CLI specs."""

from __future__ import annotations

__all__ = ["build_click", "run_click"]

import click

from heavenbase.utils import Any, Dict, List, Optional

from ..base import CLIContext
from ..registry import CLIRegistry
from ..spec import CommandGroupSpec, CommandSpec


def _click_params(command: CommandSpec) -> List[Any]:
    params: List[Any] = []
    for arg in command.args:
        variadic = arg.is_variadic()
        default = arg.default if not arg.required and not variadic else None
        params.append(click.Argument([arg.name], required=arg.required, nargs=-1 if variadic else 1, type=arg.type, default=default))
    for option in command.options:
        kwargs = {"default": option.default, "help": option.help, "multiple": option.multiple, "hidden": option.hidden}
        if not option.negatable:
            kwargs["is_flag"] = option.is_flag
            if not option.is_flag:
                kwargs["type"] = option.type
        # Supplying the backend-neutral destination explicitly matters when a
        # flag name differs from it (for example --open -> open_window).
        params.append(click.Option([*option.click_flags(), option.name], **kwargs))
    return params


def _command(command: CommandSpec, context: CLIContext) -> click.Command:
    def callback(**values: Any) -> Any:
        return command.invoke(context, values)

    return click.Command(command.name, params=_click_params(command), callback=callback, help=command.help)


class AliasedClickGroup(click.Group):
    """Click group with alias lookup.

    Args:
        aliases: Alias-to-canonical command map consumed by ``get_command``.
        *args: Positional arguments forwarded to ``click.Group``.
        **kwargs: Keyword arguments forwarded to ``click.Group`` except
            ``aliases``, which this class extracts before construction.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the aliased Click group.

        Constructor parameters are documented on ``AliasedClickGroup``.
        """
        self._aliases: Dict[str, str] = kwargs.pop("aliases", {})
        super().__init__(*args, **kwargs)

    def get_command(self, ctx: click.Context, cmd_name: str) -> Any:
        """Resolve a command name through the configured alias map.

        Args:
            ctx (click.Context): Active Click invocation context.
            cmd_name (str): User-supplied command or alias name.

        Returns:
            Any: Resolved Click command object, or ``None`` when unknown.
        """
        return super().get_command(ctx, self._aliases.get(cmd_name, cmd_name))

    def format_commands(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        """Render canonical command names together with their aliases.

        Args:
            ctx (click.Context): Active Click invocation context.
            formatter (click.HelpFormatter): Help formatter writing the
                ``Commands`` section.

        Returns:
            None: This function does not return a value.
        """
        rows = []
        for command_name in self.list_commands(ctx):
            command = super().get_command(ctx, command_name)
            if command is None or command.hidden:
                continue
            aliases = [alias for alias, canonical in self._aliases.items() if canonical == command_name]
            rows.append(("|".join(dict.fromkeys([command_name, *aliases])), command.get_short_help_str()))
        if rows:
            with formatter.section("Commands"):
                formatter.write_dl(rows)


def _group(group: CommandGroupSpec, registry: CLIRegistry, context: CLIContext) -> click.Group:
    alias_map = registry.alias_map((group.name,))
    click_group = AliasedClickGroup(group.name, help=group.help, aliases=alias_map)
    for command in group.commands:
        click_group.add_command(_command(command, context))
    return click_group


def build_click(registry: CLIRegistry, context: CLIContext) -> click.Group:
    """Build a Click command group from a CLI registry.

    Args:
        registry (CLIRegistry): Backend-neutral command registry to compile.
        context (CLIContext): Runtime package metadata and command context.

    Returns:
        click.Group: Root Click group with aliased root commands and nested
            group subcommands.
    """
    root = AliasedClickGroup(context.package, help=f"{context.package} CLI", aliases=registry.alias_map(()))
    if context.version:
        root.params.append(
            click.Option(
                ["--version", "-v"],
                is_flag=True,
                is_eager=True,
                expose_value=False,
                callback=lambda ctx, _param, value: (click.echo(context.version), ctx.exit())[1] if value else None,
            )
        )
    for command in registry.root_commands:
        root.add_command(_command(command, context))
    for group in registry.groups:
        root.add_command(_group(group, registry, context))
    return root


def run_click(app: click.Group, argv: Optional[List[str]] = None) -> Any:
    """Run a Click app with explicit argv.

    Args:
        app (click.Group): Compiled HeavenBase Click application.
        argv (List[str] | None): Arguments to parse instead of ``sys.argv[1:]``.

    Returns:
        Any: Return value from the selected command callback, if any.
    """
    return app.main(args=argv, prog_name=app.name, standalone_mode=True)
