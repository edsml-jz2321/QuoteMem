"""argparse backend compiler for HeavenBase CLI specs."""

from __future__ import annotations

__all__ = ["build_argparse", "run_argparse"]

import argparse

from heavenbase.utils import Any, List, Optional

from ..base import CLIContext
from ..registry import CLIRegistry
from ..spec import ArgumentSpec, CommandGroupSpec, CommandSpec, OptionSpec


def _argparse_name(name: str) -> str:
    return name.replace("-", "_")


def _add_argument(parser: argparse.ArgumentParser, arg: ArgumentSpec) -> None:
    kwargs: dict[str, Any] = {"help": arg.help}
    if arg.is_variadic():
        kwargs["nargs"] = "*"
    elif not arg.required:
        kwargs["nargs"] = "?"
        kwargs["default"] = arg.default
    if arg.type is not bool:
        kwargs["type"] = arg.type
    parser.add_argument(arg.name, **kwargs)


def _add_option(parser: argparse.ArgumentParser, option: OptionSpec) -> None:
    help_text = argparse.SUPPRESS if option.hidden else option.help
    if option.negatable:
        negative = option.negative_flag()
        parser.add_argument(*option.positive_flags(), dest=option.name, action="store_true", help=help_text)
        if negative:
            parser.add_argument(negative, dest=option.name, action="store_false", help=argparse.SUPPRESS)
        parser.set_defaults(**{option.name: option.default})
        return
    flags = list(option.flag_names())
    kwargs: dict[str, Any] = {"dest": option.name, "default": option.default, "help": help_text}
    if option.is_flag:
        kwargs["action"] = "store_true"
    elif option.multiple:
        kwargs["action"] = "append"
        kwargs["type"] = option.type
    elif option.type is not bool:
        kwargs["type"] = option.type
    parser.add_argument(*flags, **kwargs)


def _set_callback(parser: argparse.ArgumentParser, command: CommandSpec, context: CLIContext) -> None:
    def callback(namespace: argparse.Namespace) -> Any:
        values = {arg.name: getattr(namespace, _argparse_name(arg.name)) for arg in command.args}
        values.update({option.name: getattr(namespace, option.name) for option in command.options})
        return command.invoke(context, values)

    parser.set_defaults(_func=callback)


def _add_command(subparsers: Any, command: CommandSpec, context: CLIContext, aliases: List[str]) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(command.name, help=command.help, aliases=aliases)
    for arg in command.args:
        _add_argument(parser, arg)
    for option in command.options:
        _add_option(parser, option)
    _set_callback(parser, command, context)
    return parser


def _add_group(subparsers: Any, group: CommandGroupSpec, registry: CLIRegistry, context: CLIContext) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(group.name, help=group.help, aliases=registry.aliases_for((), group.name))
    nested = parser.add_subparsers(dest=f"{group.name}_cmd")
    for command in group.commands:
        _add_command(nested, command, context, registry.aliases_for((group.name,), command.name))
    parser.set_defaults(_func=lambda namespace: parser.print_help())
    return parser


def build_argparse(registry: CLIRegistry, context: CLIContext) -> argparse.ArgumentParser:
    """Build an ArgumentParser from a CLI registry.

    Args:
        registry (CLIRegistry): Backend-neutral command registry to compile.
        context (CLIContext): Runtime package metadata and command context.

    Returns:
        argparse.ArgumentParser: Root parser with subcommands for every
            registered root command and group.
    """
    parser = argparse.ArgumentParser(prog=context.package, description=f"{context.package} CLI")
    if context.version:
        parser.add_argument("--version", "-v", action="version", version=context.version)
    subparsers = parser.add_subparsers(dest="cmd")
    for command in registry.root_commands:
        _add_command(subparsers, command, context, registry.aliases_for((), command.name))
    for group in registry.groups:
        _add_group(subparsers, group, registry, context)
    parser.set_defaults(_func=lambda namespace: parser.print_help())
    return parser


def run_argparse(parser: argparse.ArgumentParser, argv: Optional[List[str]] = None) -> Any:
    """Parse argv and invoke the selected argparse command.

    Args:
        parser (argparse.ArgumentParser): Compiled HeavenBase CLI parser.
        argv (List[str] | None): Arguments to parse instead of ``sys.argv[1:]``.

    Returns:
        Any: Return value from the selected command callback, if any.
    """
    namespace = parser.parse_args(argv)
    return getattr(namespace, "_func")(namespace)
