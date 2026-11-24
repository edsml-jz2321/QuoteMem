"""Existing-only workspace resolution for concrete Toolkit rows."""

from __future__ import annotations

from heavenbase.utils import Any


def resolve_toolkit_workspace(workspace: Any = None) -> Any:
    """Return an explicit or selected workspace with Toolkit enabled.

    Args:
        workspace (Any): Explicit workspace. Defaults to the active or
            configured default workspace resolved by `HeavenBase.load()`.

    Returns:
        Any: Existing workspace that owns Toolkit and Capsule rows.

    Raises:
        KeyError: If no active/default workspace is registered.
    """
    if workspace is None:
        from heavenbase.workspace import HeavenBase

        workspace = HeavenBase.load()
    workspace.enable_extension("toolkit")
    return workspace
