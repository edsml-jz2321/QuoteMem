"""MCP tool for inspecting credential-safe workspace database sources."""

from __future__ import annotations

__all__ = ["build_tool", "list_sources"]

from heavenbase.backends.registry import BackendType
from heavenbase.utils import Any, Dict, List, safe_error
from heavenbase.utils.redact import redact_secrets

from ...source import redact_dsn_summary
from ..enable import ensure_database_extension
from ..errors import mcp_raise


def list_sources(workspace: Any) -> Dict[str, Any]:
    """List named workspace SQL backends without exposing credentials.

    Args:
        workspace: Target workspace whose named backends are inspected.

    Returns:
        Credential-safe source cards with backend references, database and
        dialect labels, connection status, context-storage use, and any
        ingested ``db-database`` ids.
    """
    ensure_database_extension(workspace)
    try:
        plan = workspace.storage_plans.get("db-database")
        context_backends = {binding.backend for binding in plan.bindings} if plan is not None else set()
        ingested = workspace.query("db-database").execute().rows()
        rows: List[Dict[str, Any]] = []
        for backend_ref, backend in workspace.backends.items():
            db = getattr(backend, "db", None)
            if db is None:
                continue
            row: Dict[str, Any] = {
                "backend_ref": backend_ref,
                "backend_type": BackendType.identifier_of(backend, resolver=workspace.module_resolver),
                "dialect": str(getattr(db, "dialect", "") or ""),
                "database": "",
                "dsn_summary": redact_dsn_summary(db.spec),
                "status": "connected",
                "context_backend": backend_ref in context_backends,
                "database_ids": [item["object_id"] for item in ingested if item.get("backend_ref") == backend_ref],
            }
            try:
                row["database"] = str(db.current_database() or db.spec.database or "")
            except Exception as exc:
                row["status"] = "error"
                row["error"] = redact_secrets(safe_error(exc))
            rows.append(row)
        return {"sources": rows, "n": len(rows)}
    except Exception as exc:
        mcp_raise(exc)


def build_tool(workspace: Any) -> Any:
    """Build the ``list_sources`` tool for one workspace.

    Args:
        workspace: Live workspace bound to database MCP tools.

    Returns:
        Callable MCP tool function for ``list_sources``.
    """

    def list_sources_tool() -> Dict[str, Any]:
        return list_sources(workspace)

    return list_sources_tool
