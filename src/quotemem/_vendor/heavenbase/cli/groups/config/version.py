"""Config history and diff CLI command handlers."""

from __future__ import annotations

__all__ = ["diff_config", "history_config", "list_scopes"]

from heavenbase.utils import Any
from heavenbase.utils.config.api import ConfigOps


def history_config(context: Any, scope: str | None = None, limit: int = 10, as_json: bool = False) -> None:
    """Print retained config versions.

    Args:
        context: CLI command context carrying `cm` and `out`.
        scope: Optional config scope to inspect.
        limit: Maximum number of version rows to print.
        as_json: When true, emit raw JSON rows; otherwise emit a table.

    Returns:
        None: This function does not return a value.
    """
    rows = context.cm.history(scope=scope, limit=limit)
    if as_json:
        context.out.json(rows)
    else:
        context.out.table(
            ["scope", "ver", "gen", "created_at"],
            [[row["scope"], row["ver"], row["gen"], row["created_at"]] for row in rows],
            title="Config History",
        )


def list_scopes(context: Any) -> None:
    """Print available config scopes.

    Args:
        context: CLI command context carrying `cm` and `out`.

    Returns:
        None: This function does not return a value.
    """
    context.out.table(["scope"], [[scope] for scope in context.cm.scopes()], title="Config Scopes")


def diff_config(
    context: Any,
    scope_a: str | None = None,
    version_a: int | None = None,
    scope_b: str | None = None,
    version_b: int | None = None,
) -> None:
    """Print a unified diff between two config snapshots.

    Args:
        context: CLI command context carrying `cm` and `out`.
        scope_a: Optional source config scope.
        version_a: Optional source config version.
        scope_b: Optional target config scope.
        version_b: Optional target config version.

    Returns:
        None: This function does not return a value.
    """
    context.out.echo(ConfigOps(context.cm).diff(scope_a=scope_a, version_a=version_a, scope_b=scope_b, version_b=version_b) or "No differences.")
