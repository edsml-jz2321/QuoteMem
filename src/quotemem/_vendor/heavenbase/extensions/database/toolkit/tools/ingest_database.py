"""MCP tool for ingesting one credential-safe named database backend."""

from __future__ import annotations

__all__ = ["build_tool", "ingest_database"]

from heavenbase.utils import Any, Dict, List, Mapping, Optional

from ..enable import ensure_database_extension
from ..errors import mcp_raise


def ingest_database(
    workspace: Any,
    backend_ref: str,
    *,
    name: Optional[str] = None,
    tables: Optional[List[str]] = None,
    columns: Optional[Mapping[str, List[str]]] = None,
    profile: bool = True,
) -> Dict[str, List[str]]:
    """Ingest a selected named workspace SQL backend into ``db-*`` rows.

    This tool deliberately accepts only a backend reference already configured
    on the workspace. Connection URLs, usernames, passwords, and tokens never
    enter the model tool call.

    Args:
        workspace: Target workspace with the database extension enabled.
        backend_ref: Name returned by ``list_sources``.
        name: Optional human-readable database label.
        tables: Optional qualified or unqualified table and view selectors.
        columns: Optional table-to-column selector mapping.
        profile: Whether to collect bounded column statistics and common values.

    Returns:
        Created or refreshed object ids grouped by database entity kind.

    Raises:
        RuntimeError: If the named backend does not expose a SQL database or
            introspection fails; error text is redacted for the MCP boundary.
    """
    ensure_database_extension(workspace)
    try:
        ref = str(backend_ref or "").strip()
        if not ref:
            raise ValueError("backend_ref is empty; call list_sources first")
        backend = workspace.backends.get(ref)
        if getattr(backend, "db", None) is None:
            raise ValueError(f"Workspace backend {ref!r} does not expose a SQL database")
        return workspace.database.ingest(ref, name=name, tables=tables, columns=columns, profile=profile)
    except Exception as exc:
        mcp_raise(exc)


def build_tool(workspace: Any) -> Any:
    """Build the ``ingest_database`` tool for one workspace.

    Args:
        workspace: Live workspace bound to database MCP tools.

    Returns:
        Callable MCP tool function for ``ingest_database``.
    """

    def ingest_database_tool(
        backend_ref: str,
        name: Optional[str] = None,
        tables: Optional[List[str]] = None,
        columns: Optional[Mapping[str, List[str]]] = None,
        profile: bool = True,
    ) -> Dict[str, List[str]]:
        return ingest_database(workspace, backend_ref, name=name, tables=tables, columns=columns, profile=profile)

    return ingest_database_tool
