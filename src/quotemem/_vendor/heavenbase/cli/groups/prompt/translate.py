"""Prompt translation CLI command handlers."""

from __future__ import annotations

__all__ = ["bind_translation", "list_translations", "remove_translation", "set_translation", "unbind_translation"]

from heavenbase.utils import Any

from .runtime import prompt_type, prompt_workspace


def list_translations(context: Any, ref: str, lang: str | None = None, as_json: bool = False) -> None:
    """Print translation rows for one prompt.

    Args:
        context (Any): CLI command context carrying `out`.
        ref (str): Prompt id, name, or `name:version` reference.
        lang (str | None): Optional target language filter.
        as_json (bool): When true, print raw JSON rows; otherwise print a table.

    Returns:
        None: This function prints translation rows and does not return a value.
    """
    Prompt = prompt_type(context)
    rows = Prompt.load(ref, ws=prompt_workspace(context)).tr.list(lang=lang)
    if as_json:
        context.out.json(rows)
        return
    context.out.table(
        ["dict", "lang", "source", "target"], [[row.get("dict_name"), row.get("lang"), row.get("source_key"), row.get("target_value")] for row in rows]
    )


def set_translation(context: Any, ref: str, key: str, lang: str, value: str, dict_name: str | None = None) -> None:
    """Set one prompt translation.

    Args:
        context (Any): CLI command context carrying `out`.
        ref (str): Prompt id, name, or `name:version` reference.
        key (str): Source text or translation key.
        lang (str): Target language identifier.
        value (str): Target translation value.
        dict_name (str | None): Optional translation dictionary name.

    Returns:
        None: This function prints a success message and does not return a value.
    """
    Prompt = prompt_type(context)
    Prompt.load(ref, ws=prompt_workspace(context)).tr.set(key, lang, value, dict_name=dict_name)
    context.out.success(f"set translation {lang} {key}")


def remove_translation(context: Any, ref: str, key: str, lang: str, dict_name: str | None = None) -> None:
    """Remove one prompt translation.

    Args:
        context (Any): CLI command context carrying `out`.
        ref (str): Prompt id, name, or `name:version` reference.
        key (str): Source text or translation key.
        lang (str): Target language identifier.
        dict_name (str | None): Optional translation dictionary name.

    Returns:
        None: This function prints a success or warning message and does not
            return a value.
    """
    Prompt = prompt_type(context)
    ok = Prompt.load(ref, ws=prompt_workspace(context)).tr.delete(key, lang, dict_name=dict_name)
    context.out.success(f"removed translation {lang} {key}") if ok else context.out.warning(f"translation {lang} {key} was not found")


def bind_translation(context: Any, ref: str, dict_name: str) -> None:
    """Bind a translation dictionary to one prompt.

    Args:
        context (Any): CLI command context carrying `out`.
        ref (str): Prompt id, name, or `name:version` reference.
        dict_name (str): Translation dictionary name.

    Returns:
        None: This function prints a success message and does not return a value.
    """
    Prompt = prompt_type(context)
    Prompt.load(ref, ws=prompt_workspace(context)).tr.bind(dict_name)
    context.out.success(f"bound translation dict {dict_name}")


def unbind_translation(context: Any, ref: str, dict_name: str) -> None:
    """Unbind a translation dictionary from one prompt.

    Args:
        context (Any): CLI command context carrying `out`.
        ref (str): Prompt id, name, or `name:version` reference.
        dict_name (str): Translation dictionary name.

    Returns:
        None: This function prints a success message and does not return a value.
    """
    Prompt = prompt_type(context)
    Prompt.load(ref, ws=prompt_workspace(context)).tr.unbind(dict_name)
    context.out.success(f"unbound translation dict {dict_name}")
