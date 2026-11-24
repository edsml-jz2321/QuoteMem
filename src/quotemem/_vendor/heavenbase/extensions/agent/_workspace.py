"""Lazy workspace resolution for persistent agent rows."""

from __future__ import annotations

from heavenbase.utils import Any


def resolve_agent_workspace(ws: Any = None) -> Any:
    """Return a workspace with the optional agent extension enabled."""
    if ws is not None:
        ws.enable_extension("agent")
        return ws
    from heavenbase.workspace import HeavenBase

    workspace = HeavenBase.default()
    workspace.enable_extension("agent")
    return workspace
