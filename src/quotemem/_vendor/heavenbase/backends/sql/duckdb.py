"""DuckDB backend."""

from __future__ import annotations

__all__ = [
    "DuckDBBackend",
]

from heavenbase.utils import Any

from ...modules import ModuleResolver
from .base import SQLBackend
from ._utils import default_database
from .capabilities import register_sql_dialect
from .dialect import SQL_DEFAULT_NATIVE_AGGREGATE_OPS, SQLDialectSpec, SQL_REGEX_STYLE_DUCKDB_TILDE_SEARCH


class DuckDBBackend(SQLBackend):
    """SQL backend configured for DuckDB."""

    identifier = "duckdb"
    sql_name_warning_provider = "duckdb"

    @classmethod
    def from_config(cls, name: str, data: dict[str, Any], *, resolver: ModuleResolver | None = None, config: Any | None = None) -> "DuckDBBackend":
        """Build the DuckDB backend from a config mapping.

        Args:
            name: Workspace backend instance name.
            data: Backend config mapping. Supports ``database``, local file path options,
                optional ``ws_id``, and database connection options.
            resolver: Context-owned module authority.
            config: Config authority retained by the backend.

        Returns:
            DuckDBBackend: Configured DuckDB backend instance.
        """
        cfg = dict(data or {})
        ws_id = cfg.pop("ws_id", None)
        return cls(name, default_database(ws_id, cfg), ws_id=ws_id, resolver=resolver, config=config, **cfg)

    def __init__(
        self,
        name: str = "duckdb",
        database: str | None = None,
        ws_id: str | None = None,
        *,
        resolver: ModuleResolver | None = None,
        config: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """Create a DuckDB SQL backend.

        Args:
            name: Workspace backend instance name.
            database: Logical name, canonical ``file://`` URI, or ``:memory:`` value.
            ws_id: Optional workspace namespace used to derive file database names.
            resolver: Context-owned module authority.
            config: Config authority retained by the backend.
            **kwargs: Additional database connection options forwarded to ``SQLBackend``.

        Returns:
            None: This function does not return a value.
        """
        super().__init__(name, "duckdb", database, identifier="duckdb", ws_id=ws_id, resolver=resolver, config=config, **kwargs)
