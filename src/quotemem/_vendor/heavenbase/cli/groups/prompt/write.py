"""Prompt lifecycle CLI command handlers."""

from __future__ import annotations

__all__ = ["create_prompt", "remove_prompt"]

from heavenbase.utils import Any

from .runtime import prompt_type, prompt_workspace


def create_prompt(context: Any, name: str, template: str, tr_keys: list[str] | None = None) -> None:
    """Create and register a template prompt.

    Args:
        context (Any): CLI command context carrying `out`.
        name (str): Prompt name.
        template (str): Format-string template text.
        tr_keys (list[str] | None): Template argument names whose values should
            be translated at render time.

    Returns:
        None: This function prints a success message and does not return a value.
    """
    Prompt = prompt_type(context)
    workspace = prompt_workspace(context)
    prompt_id = Prompt(template, name=name, tr_keys=list(tr_keys or []), ws=workspace).register(ws=workspace)
    context.out.success(f"created prompt {prompt_id}")


def remove_prompt(context: Any, ref: str, hard: bool = False) -> None:
    """Remove or tombstone one prompt.

    Args:
        context (Any): CLI command context carrying `out`.
        ref (str): Prompt id, name, or `name:version` reference.
        hard (bool): When true, delete the row instead of tombstoning it.

    Returns:
        None: This function prints a success or warning message and does not
            return a value.
    """
    Prompt = prompt_type(context)
    ok = Prompt.delete(ref, tombstone=not hard, ws=prompt_workspace(context))
    context.out.success(f"removed prompt {ref}") if ok else context.out.warning(f"prompt {ref} was not found")
