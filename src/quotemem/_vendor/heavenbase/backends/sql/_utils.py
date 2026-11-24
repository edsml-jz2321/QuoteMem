"""Internal SQL-backend config normalization."""

from __future__ import annotations

from heavenbase.utils import Any

from ...database.dialect.registry import database_dialects
from ..base import Backend


def default_database(ws_id: str | None, raw: dict[str, Any], key: str = "database", suffix: str = "") -> Any:
    """Pop a database value or derive one from the workspace identifier."""
    value = raw.pop(key, None)
    if value is not None:
        return value
    return f"{Backend._physical_name(ws_id)}{suffix}" if ws_id else None


def database_uses_local_path(provider: Any, *, resolver: Any = None) -> bool:
    """Return whether a SQLAlchemy provider uses local database path tokens."""
    provider_name = str(provider or "").strip().lower()
    return bool(provider_name and database_dialects.get(provider_name, resolver=resolver).supports("database-local-path"))
