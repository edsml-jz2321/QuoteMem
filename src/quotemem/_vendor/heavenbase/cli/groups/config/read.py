"""Read-only config CLI command handlers."""

from __future__ import annotations

__all__ = ["emit_config", "get_config", "list_config"]

from heavenbase.utils import Any
from heavenbase.utils.config.api import ConfigOps


def emit_config(context: Any, data: Any, as_json: bool = False) -> None:
    """Emit config data through the requested structured output format.

    Args:
        context: CLI command context carrying `out`.
        data: Structured config data to print.
        as_json: When true, emit JSON; otherwise emit YAML.

    Returns:
        None: This function does not return a value.
    """
    context.out.json(data) if as_json else context.out.yaml(data)


def get_config(context: Any, key: str | None = None, scope: str | None = None, version: int | None = None, as_json: bool = False) -> None:
    """Print merged config or a single config key.

    Args:
        context: CLI command context carrying `cm` and `out`.
        key: Optional flattened config key.
        scope: Optional config scope to inspect.
        version: Optional historical config version.
        as_json: When true, emit JSON; otherwise emit YAML.

    Returns:
        None: This function does not return a value.
    """
    emit_config(context, ConfigOps(context.cm).get(key, scope=scope, version=version), as_json=as_json)


def list_config(context: Any, prefix: str | None = None, scope: str | None = None, as_json: bool = False) -> None:
    """Print flattened config key/value rows.

    Args:
        context: CLI command context carrying `cm` and `out`.
        prefix: Optional key prefix filter.
        scope: Optional config scope to inspect.
        as_json: When true, emit raw JSON rows; otherwise emit a table.

    Returns:
        None: This function does not return a value.
    """
    rows = ConfigOps(context.cm).list(prefix, scope=scope)
    if as_json:
        context.out.json(rows)
    else:
        context.out.table(["key", "value"], [[row["key"], row["value"]] for row in rows], title="Config")
