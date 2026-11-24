"""Config maintenance CLI command handlers."""

from __future__ import annotations

__all__ = ["compact_config", "edit_config", "open_config", "reset_config"]

from heavenbase.utils import Any
from heavenbase.utils.config.api import ConfigOps

from .read import get_config
from .write import _require_config_saved


def reset_config(context: Any, scope: str | None = None) -> None:
    """Reset one config scope.

    Args:
        context: CLI command context carrying `cm` and `out`.
        scope: Optional config scope to reset.

    Returns:
        None: This function does not return a value.
    """
    manager = context.cm
    manager.init(scope=scope, reset=True)
    _require_config_saved(manager)
    context.out.success(f"reset {scope or manager.scope}")


def compact_config(context: Any, scope: str | None = None) -> None:
    """Compact retained config versions for one scope.

    Args:
        context: CLI command context carrying `cm` and `out`.
        scope: Optional config scope to compact.

    Returns:
        None: This function does not return a value.
    """
    manager = context.cm
    removed = manager.compact(scope=scope)
    _require_config_saved(manager)
    context.out.success(f"compacted {scope or manager.scope}; removed {removed}")


def edit_config(context: Any, scope: str | None = None, editor: str | None = None) -> None:
    """Open one config layer in an editor.

    Args:
        context: CLI command context carrying `cm` and `out`.
        scope: Optional config scope to edit.
        editor: Optional editor command override.

    Returns:
        None: This function does not return a value.
    """
    ops = ConfigOps(context.cm)
    ops.edit(scope=scope, editor=editor)
    _require_config_saved(ops.manager)
    context.out.success(f"edited {scope or ops.manager.scope}")


def open_config(context: Any, scope: str | None = None) -> None:
    """Print one config layer as structured output.

    Args:
        context: CLI command context carrying `cm` and `out`.
        scope: Optional config scope to inspect.

    Returns:
        None: This function does not return a value.
    """
    get_config(context, scope=scope)
