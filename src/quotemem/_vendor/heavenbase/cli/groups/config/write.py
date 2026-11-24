"""Config mutation CLI command handlers."""

from __future__ import annotations

__all__ = ["add_config", "copy_config", "set_config", "unset_config"]

from heavenbase.utils import Any
from heavenbase.utils.config import ConfigManager
from heavenbase.utils.config.api import ConfigOps


def _parse_mode(as_json: bool) -> str:
    """Return the config API parse mode for CLI value input."""
    return "json" if as_json else "auto"


def _require_config_saved(manager: ConfigManager) -> None:
    """Reject process-only writes before printing a success message."""
    if manager.last_write_persisted is False:
        raise RuntimeError("Configuration change was not saved. The configuration store is unavailable or process-only.")


def set_config(context: Any, key: str, value: str, scope: str | None = None, as_json: bool = False) -> None:
    """Set one config key.

    Args:
        context: CLI command context carrying `cm` and `out`.
        key: Flattened config key to set.
        value: Config value text parsed by the config API.
        scope: Optional config scope to mutate.
        as_json: When true, parse the value strictly as JSON.

    Returns:
        None: This function does not return a value.
    """
    ops = ConfigOps(context.cm)
    ops.set(key, value, scope=scope, parse=_parse_mode(as_json))
    _require_config_saved(ops.manager)
    context.out.success(f"set {key}")


def add_config(context: Any, key: str, value: str, scope: str | None = None, as_json: bool = False) -> None:
    """Append one value to a config list key.

    Args:
        context: CLI command context carrying `cm` and `out`.
        key: Flattened list config key to append.
        value: Config value text parsed by the config API.
        scope: Optional config scope to mutate.
        as_json: When true, parse the value strictly as JSON.

    Returns:
        None: This function does not return a value.
    """
    ops = ConfigOps(context.cm)
    ops.add(key, value, scope=scope, parse=_parse_mode(as_json))
    _require_config_saved(ops.manager)
    context.out.success(f"added {key}")


def unset_config(context: Any, key: str, scope: str | None = None) -> None:
    """Unset one config key.

    Args:
        context: CLI command context carrying `cm` and `out`.
        key: Flattened config key to remove.
        scope: Optional config scope to mutate.

    Returns:
        None: This function does not return a value.
    """
    ops = ConfigOps(context.cm)
    ops.unset(key, scope=scope)
    _require_config_saved(ops.manager)
    context.out.success(f"unset {key}")


def copy_config(
    context: Any,
    key: str | None = None,
    source: str = "base",
    from_scope: str | None = None,
    from_version: int | None = None,
    target_scope: str | None = None,
) -> None:
    """Copy config values into the target scope.

    Args:
        context: CLI command context carrying `cm` and `out`.
        key: Optional flattened config key to copy; copies all keys when omitted.
        source: Source preset understood by the config API.
        from_scope: Optional source config scope.
        from_version: Optional source config version.
        target_scope: Optional target config scope.

    Returns:
        None: This function does not return a value.
    """
    ops = ConfigOps(context.cm)
    ops.copy(key, source=source, from_scope=from_scope, from_version=from_version, target_scope=target_scope)
    _require_config_saved(ops.manager)
    context.out.success(f"copied {key or 'all'}")
