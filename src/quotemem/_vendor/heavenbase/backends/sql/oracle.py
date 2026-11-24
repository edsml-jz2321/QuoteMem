"""Oracle backend."""

from __future__ import annotations

__all__ = [
    "OracleBackend",
]

from heavenbase.utils import Any

from ...modules import ModuleResolver
from .base import SQLBackend
from ._utils import default_database
from .capabilities import register_sql_dialect
from .dialect import SQLDialectSpec


class OracleBackend(SQLBackend):
    """SQL backend configured for Oracle."""

    identifier = "oracle"
    sql_name_warning_provider = "oracle"

    @classmethod
    def from_config(cls, name: str, data: dict[str, Any], *, resolver: ModuleResolver | None = None, config: Any | None = None) -> "OracleBackend":
        """Build the Oracle backend from a config mapping.

        Args:
            name: Workspace backend instance name.
            data: Backend config mapping. Supports ``database``, optional ``ws_id``, and
                database connection options.
            resolver: Context-owned module authority.
            config: Config authority retained by the backend.

        Returns:
            OracleBackend: Configured Oracle backend instance.
        """
        cfg = dict(data or {})
        ws_id = cfg.pop("ws_id", None)
        return cls(name, default_database(ws_id, cfg), ws_id=ws_id, resolver=resolver, config=config, **cfg)

    def __init__(
        self,
        name: str = "oracle",
        database: str | None = None,
        ws_id: str | None = None,
        *,
        resolver: ModuleResolver | None = None,
        config: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """Create an Oracle SQL backend.

        Args:
            name: Workspace backend instance name.
            database: Oracle database name or configured database route.
            ws_id: Optional workspace namespace used to derive workspace database names.
            resolver: Context-owned module authority.
            config: Config authority retained by the backend.
            **kwargs: Additional database connection options forwarded to ``SQLBackend``.

        Returns:
            None: This function does not return a value.
        """
        super().__init__(name, "oracle", database, identifier="oracle", ws_id=ws_id, resolver=resolver, config=config, **kwargs)
