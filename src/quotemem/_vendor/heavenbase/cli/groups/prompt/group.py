"""Prompt CLI command specifications."""

from __future__ import annotations

__all__ = ["prompt_group"]

from ...spec import ArgumentSpec, CommandGroupSpec, CommandSpec, OptionSpec
from .read import list_prompts, render_prompt, show_prompt
from .translate import bind_translation, list_translations, remove_translation, set_translation, unbind_translation
from .write import create_prompt, remove_prompt


def _json_option() -> OptionSpec:
    """Return the shared JSON output option."""
    return OptionSpec.from_flags("as_json", "--json", "-j", help="Print JSON.", default=False, is_flag=True, type=bool)


def _lang_option() -> OptionSpec:
    """Return the shared prompt language option."""
    return OptionSpec.from_flags("lang", "--lang", "-l", help="Prompt language.")


def _dict_option() -> OptionSpec:
    """Return the shared translation dictionary option."""
    return OptionSpec.from_flags("dict_name", "--dict", help="Translation dictionary name.")


def prompt_group() -> CommandGroupSpec:
    """Return the built-in prompt command group specification.

    Args:
        None.

    Returns:
        CommandGroupSpec: Backend-neutral CLI command metadata for prompt
            listing, rendering, lifecycle, and translation commands.
    """
    json_opt = _json_option()
    lang_opt = _lang_option()
    dict_opt = _dict_option()
    return CommandGroupSpec(
        "prompt",
        help="Manage HeavenBase prompts and translations.",
        commands=[
            CommandSpec(
                "list",
                list_prompts,
                help="List prompts.",
                aliases=["ls"],
                options=[
                    OptionSpec.from_flags("prefix", "--prefix", "-p", help="Prompt name prefix."),
                    OptionSpec.from_flags("status", "--status", help="Prompt status filter.", default="active"),
                    json_opt,
                ],
            ),
            CommandSpec(
                "show",
                show_prompt,
                help="Show one prompt row.",
                aliases=["browse", "get"],
                args=[ArgumentSpec("ref", help="Prompt id, name, or name:version.")],
                options=[json_opt],
            ),
            CommandSpec(
                "create",
                create_prompt,
                help="Create a template prompt.",
                args=[ArgumentSpec("name", help="Prompt name.")],
                options=[
                    OptionSpec.from_flags("template", "--template", "-t", help="Format-string template."),
                    OptionSpec.from_flags("tr_keys", "--tr-key", help="Template argument value to translate.", multiple=True),
                ],
            ),
            CommandSpec(
                "render",
                render_prompt,
                help="Render a prompt with JSON object args.",
                args=[ArgumentSpec("ref", help="Prompt id, name, or name:version.")],
                options=[OptionSpec.from_flags("args", "--args", "-a", help="JSON object arguments.", default="{}"), lang_opt],
            ),
            CommandSpec(
                "remove",
                remove_prompt,
                help="Remove a prompt.",
                aliases=["rm", "del", "delete"],
                args=[ArgumentSpec("ref", help="Prompt id, name, or name:version.")],
                options=[OptionSpec.from_flags("hard", "--hard", help="Delete instead of tombstone.", default=False, is_flag=True, type=bool)],
            ),
            CommandSpec(
                "tr-list", list_translations, help="List prompt translations.", args=[ArgumentSpec("ref", help="Prompt ref.")], options=[lang_opt, json_opt]
            ),
            CommandSpec(
                "tr-set",
                set_translation,
                help="Set a prompt translation.",
                args=[
                    ArgumentSpec("ref", help="Prompt ref."),
                    ArgumentSpec("key", help="Source key."),
                    ArgumentSpec("lang", help="Target language."),
                    ArgumentSpec("value", help="Target value."),
                ],
                options=[dict_opt],
            ),
            CommandSpec(
                "tr-remove",
                remove_translation,
                help="Remove a prompt translation.",
                aliases=["tr-rm", "tr-del", "tr-delete"],
                args=[ArgumentSpec("ref", help="Prompt ref."), ArgumentSpec("key", help="Source key."), ArgumentSpec("lang", help="Target language.")],
                options=[dict_opt],
            ),
            CommandSpec(
                "tr-bind",
                bind_translation,
                help="Bind a translation dictionary.",
                args=[ArgumentSpec("ref", help="Prompt ref."), ArgumentSpec("dict_name", help="Dictionary name.")],
            ),
            CommandSpec(
                "tr-unbind",
                unbind_translation,
                help="Unbind a translation dictionary.",
                args=[ArgumentSpec("ref", help="Prompt ref."), ArgumentSpec("dict_name", help="Dictionary name.")],
            ),
        ],
    )
