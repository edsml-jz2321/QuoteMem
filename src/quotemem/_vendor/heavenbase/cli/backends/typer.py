"""Typer backend compiler for HeavenBase CLI specs."""

from __future__ import annotations

__all__ = ["build_typer", "run_typer"]

import inspect

import typer
from typer.core import TyperGroup
from typer.main import get_command

from heavenbase.utils import Any, Dict, List, Optional

from ..base import CLIContext
from ..registry import CLIRegistry
from ..spec import ArgumentSpec, CommandGroupSpec, CommandSpec, OptionSpec


class _AliasedTyperGroup(TyperGroup):
    _aliases: Dict[str, str] = {}

    def get_command(self, ctx: Any, cmd_name: str) -> Any:
        return super().get_command(ctx, self._aliases.get(cmd_name, cmd_name))

    def _display_name(self, canonical: str) -> str:
        aliases = [alias for alias, target in self._aliases.items() if target == canonical]
        return "|".join(dict.fromkeys([canonical, *aliases]))

    def format_help(self, ctx: Any, formatter: Any) -> None:
        renamed = []
        for command_name in self.list_commands(ctx):
            command = super().get_command(ctx, command_name)
            if command is None or getattr(command, "hidden", False):
                continue
            display_name = self._display_name(command_name)
            if command.name != display_name:
                renamed.append((command, command.name))
                command.name = display_name
        try:
            super().format_help(ctx, formatter)
        finally:
            for command, name in renamed:
                command.name = name

    def format_commands(self, ctx: Any, formatter: Any) -> None:
        rows = []
        for command_name in self.list_commands(ctx):
            command = super().get_command(ctx, command_name)
            if command is None or getattr(command, "hidden", False):
                continue
            rows.append((self._display_name(command_name), command.get_short_help_str()))
        if rows:
            with formatter.section("Commands"):
                formatter.write_dl(rows)


def _aliased_group(alias_map: Dict[str, str]) -> type[_AliasedTyperGroup]:
    class BoundAliasedTyperGroup(_AliasedTyperGroup):
        _aliases = alias_map

    return BoundAliasedTyperGroup


def _typer(context_settings: Dict[str, Any], alias_map: Dict[str, str], **kwargs: Any) -> typer.Typer:
    return typer.Typer(cls=_aliased_group(alias_map), context_settings=context_settings, **kwargs)


def _annotation(arg: ArgumentSpec | OptionSpec) -> Any:
    value_type = str if arg.type is None else arg.type
    if isinstance(arg, OptionSpec) and (arg.is_flag or arg.type is bool):
        if arg.multiple:
            return Optional[List[bool]]
        return bool
    if isinstance(arg, OptionSpec) and arg.multiple:
        return Optional[List[value_type]]  # type: ignore[valid-type]
    if isinstance(arg, ArgumentSpec) and arg.is_variadic():
        return List[value_type]  # type: ignore[valid-type]
    return value_type


def _argument_default(arg: ArgumentSpec) -> Any:
    default = ... if arg.required and not arg.is_variadic() else arg.default
    if arg.is_variadic():
        default = None
    return typer.Argument(default, help=arg.help)


def _option_default(option: OptionSpec) -> Any:
    if option.is_flag or option.type is bool:
        return typer.Option(option.default, *option.click_flags(), help=option.help, is_flag=True, hidden=option.hidden)
    return typer.Option(option.default, *option.click_flags(), help=option.help, hidden=option.hidden)


def _callback(command: CommandSpec, context: CLIContext) -> Any:
    params = []
    for arg in command.args:
        params.append(inspect.Parameter(arg.name, inspect.Parameter.POSITIONAL_OR_KEYWORD, default=_argument_default(arg), annotation=_annotation(arg)))
    for option in command.options:
        params.append(inspect.Parameter(option.name, inspect.Parameter.KEYWORD_ONLY, default=_option_default(option), annotation=_annotation(option)))
    signature = inspect.Signature(params)

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        bound = signature.bind_partial(*args, **kwargs)
        values: Dict[str, Any] = dict(bound.arguments)
        return command.invoke(context, values)

    wrapped.__name__ = command.name.replace("-", "_")
    wrapped.__doc__ = command.help
    wrapped.__annotations__ = {param.name: param.annotation for param in signature.parameters.values() if param.annotation is not inspect.Signature.empty}
    wrapped.__signature__ = signature  # type: ignore[attr-defined]
    return wrapped


def _add_command(app: typer.Typer, command: CommandSpec, context: CLIContext) -> None:
    app.command(command.name, help=command.help, context_settings={"help_option_names": ["-h", "--help"]})(_callback(command, context))


def _group(group: CommandGroupSpec, registry: CLIRegistry, context: CLIContext) -> typer.Typer:
    app = _typer(
        {"help_option_names": ["-h", "--help"]},
        registry.alias_map((group.name,)),
        help=group.help,
        no_args_is_help=True,
    )
    for command in group.commands:
        _add_command(app, command, context)
    return app


def build_typer(registry: CLIRegistry, context: CLIContext) -> typer.Typer:
    """Build a Typer app from a CLI registry.

    Args:
        registry (CLIRegistry): Backend-neutral command registry to compile.
        context (CLIContext): Runtime package metadata and command context.

    Returns:
        typer.Typer: Root Typer application with aliased root commands and
            nested group sub-apps.
    """
    app = _typer(
        {"help_option_names": ["-h", "--help"]},
        registry.alias_map(()),
        name=context.package,
        help=f"{context.package} CLI",
        no_args_is_help=True,
    )

    @app.callback(invoke_without_command=True)
    def root(version: bool = typer.Option(False, "--version", "-v", help="Show version and exit.")) -> None:
        if version:
            typer.echo(context.version)
            raise typer.Exit()

    for command in registry.root_commands:
        _add_command(app, command, context)
    for group in registry.groups:
        group_app = _group(group, registry, context)
        app.add_typer(group_app, name=group.name)
    return app


def run_typer(app: typer.Typer, argv: Optional[List[str]] = None) -> Any:
    """Run a Typer app without mutating ``sys.argv``.

    Args:
        app (typer.Typer): Compiled HeavenBase Typer application.
        argv (List[str] | None): Arguments to parse instead of ``sys.argv[1:]``.

    Returns:
        Any: Return value from the selected command callback, if any.
    """
    command = get_command(app)
    return command.main(args=argv, prog_name=command.name, standalone_mode=True)
