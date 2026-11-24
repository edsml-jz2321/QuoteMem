"""Lazy workspace resolution for prompt extension rows."""

from __future__ import annotations

from heavenbase.utils import Any
from heavenbase.utils.config import CM_HVNB


def prompt_config(ws: Any = None) -> Any:
    """Return a prompt workspace's config or the process default when unbound."""
    context = getattr(ws, "context", None)
    return getattr(context, "config", CM_HVNB)


def resolve_prompt_workspace(ws: Any = None) -> Any:
    """Return the explicit, active, or configured default prompt workspace."""
    if ws is not None:
        return ws
    from heavenbase.workspace import HeavenBase

    try:
        return HeavenBase.load()
    except KeyError:
        return HeavenBase.default()
