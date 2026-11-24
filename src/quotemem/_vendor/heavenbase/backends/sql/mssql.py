"""Microsoft SQL Server backend."""

from __future__ import annotations

__all__ = [
    "MsSQLBackend",
]

from heavenbase.utils import Any

from ...modules import ModuleResolver
from .base import SQLBackend
from ._utils import default_database
from .capabilities import register_sql_dialect
from .dialect import SQL_DEFAULT_NATIVE_AGGREGATE_OPS, SQLDialectSpec


class MsSQLBackend(SQLBackend):
    """SQL backend configured for Microsoft SQL Server."""

    identifier = "mssql"
    aliases = ("msql",)
    sql_name_warning_provider = "mssql"

    @classmethod
    def from_config(cls, name: str, data: dict[str, Any], *, resolver: ModuleResolver | None = None, config: Any | None = None) -> "MsSQLBackend":
        """Build the Microsoft SQL Server backend from a config mapping.

        Args:
            name: Workspace backend instance name.
            data: Backend config mapping. Supports ``database``, optional ``ws_id``, and
                database connection options.
            resolver: Context-owned module authority.
            config: Config authority retained by the backend.

        Returns:
            MsSQLBackend: Configured Microsoft SQL Server backend instance.
        """
        cfg = dict(data or {})
        ws_id = cfg.pop("ws_id", None)
        return cls(name, default_database(ws_id, cfg), ws_id=ws_id, resolver=resolver, config=config, **cfg)

    def __init__(
        self,
        name: str = "mssql",
        database: str | None = None,
        ws_id: str | None = None,
        *,
        resolver: ModuleResolver | None = None,
        config: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """Create a Microsoft SQL Server backend.

        Args:
            name: Workspace backend instance name.
            database: SQL Server database name or configured database route.
            ws_id: Optional workspace namespace used to derive workspace database names.
            resolver: Context-owned module authority.
            config: Config authority retained by the backend.
            **kwargs: Additional database connection options forwarded to ``SQLBackend``.

        Returns:
            None: This function does not return a value.
        """
        super().__init__(name, "mssql", database, identifier="mssql", ws_id=ws_id, resolver=resolver, config=config, **kwargs)

    def _database_config(self, values: dict[str, Any]) -> dict[str, Any]:
        """Resolve SQL Server defaults from this backend's config authority."""
        values.setdefault("driver", self._construction_config_get("heavenbase.backend_defaults.mssql.driver", default="pymssql"))
        if values.get("driver") == "pymssql" and "params" not in values:
            values["params"] = {}
        return values
