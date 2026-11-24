"""Prompt read and render CLI command handlers."""

from __future__ import annotations

__all__ = ["list_prompts", "parse_prompt_args", "render_prompt", "show_prompt"]

from heavenbase.utils import Any, loads_json

from .runtime import prompt_type, prompt_workspace


def parse_prompt_args(text: str | None) -> dict[str, Any]:
    """Parse prompt render arguments from JSON object text.

    Args:
        text (str | None): JSON object text from the CLI. `None` and empty
            strings are treated as an empty object.

    Returns:
        dict[str, Any]: Parsed prompt render arguments.

    Raises:
        TypeError: If the JSON value is not an object.
    """
    data = loads_json(text or "{}")
    if not isinstance(data, dict):
        raise TypeError("Prompt args must be a JSON object")
    return data


def list_prompts(context: Any, prefix: str | None = None, status: str | None = "active", as_json: bool = False) -> None:
    """Print prompt rows matching the requested filters.

    Args:
        context (Any): CLI command context carrying `out`.
        prefix (str | None): Optional prompt-name prefix filter.
        status (str | None): Optional prompt status filter.
        as_json (bool): When true, print raw JSON rows; otherwise print a table.

    Returns:
        None: This function prints prompt rows and does not return a value.
    """
    Prompt = prompt_type(context)
    rows = Prompt.list(prefix=prefix, status=status, ws=prompt_workspace(context))
    if as_json:
        context.out.json(rows)
        return
    context.out.table(["id", "name", "version", "status"], [[row.get("object_id"), row.get("name"), row.get("version"), row.get("status")] for row in rows])


def show_prompt(context: Any, ref: str, as_json: bool = False) -> None:
    """Print one prompt row.

    Args:
        context (Any): CLI command context carrying `out`.
        ref (str): Prompt id, name, or `name:version` reference.
        as_json (bool): When true, print JSON; otherwise print YAML.

    Returns:
        None: This function prints one prompt row and does not return a value.
    """
    Prompt = prompt_type(context)
    row = Prompt.describe(ref, ws=prompt_workspace(context))
    context.out.json(row) if as_json else context.out.yaml(row)


def render_prompt(context: Any, ref: str, args: str = "{}", lang: str | None = None) -> None:
    """Render a prompt with JSON object arguments.

    Args:
        context (Any): CLI command context carrying `out`.
        ref (str): Prompt id, name, or `name:version` reference.
        args (str): JSON object text containing template arguments.
        lang (str | None): Optional target language for translation-aware rendering.

    Returns:
        None: This function prints rendered prompt text and does not return a value.
    """
    Prompt = prompt_type(context)
    context.out.echo(Prompt.load(ref, lang=lang, ws=prompt_workspace(context))(**parse_prompt_args(args)))
