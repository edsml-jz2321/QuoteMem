"""PostgreSQL backend."""

from __future__ import annotations

__all__ = [
    "PostgresBackend",
]

from heavenbase.utils import Any

from ...modules import ModuleResolver
from .base import SQLBackend
from ._utils import default_database
from .capabilities import register_sql_dialect
from .dialect import (
    SQL_DEFAULT_NATIVE_AGGREGATE_OPS,
    SQL_JSON_AGG_STYLE_POSTGRES_ORDERED,
    SQL_JSON_OBJECT_STYLE_BUILD_OBJECT,
    SQL_REGEX_STYLE_POSTGRES_TILDE,
    SQLDialectSpec,
)


class PostgresBackend(SQLBackend):
    """SQL backend configured for PostgreSQL."""

    identifier = "postgres"
    aliases = ("postgresql", "pg")
    sql_name_warning_provider = "postgres"

    @classmethod
    def from_config(cls, name: str, data: dict[str, Any], *, resolver: ModuleResolver | None = None, config: Any | None = None) -> "PostgresBackend":
        """Build the PostgreSQL backend from a config mapping.

        Args:
            name: Workspace backend instance name.
            data: Backend config mapping. Supports ``database``, optional ``ws_id``, and
                database connection options.
            resolver: Context-owned module authority.
            config: Config authority retained by the backend.

        Returns:
            PostgresBackend: Configured PostgreSQL backend instance.
        """
        cfg = dict(data or {})
        ws_id = cfg.pop("ws_id", None)
        return cls(name, default_database(ws_id, cfg), ws_id=ws_id, resolver=resolver, config=config, **cfg)

    def __init__(
        self,
        name: str = "postgres",
        database: str | None = None,
        ws_id: str | None = None,
        *,
        resolver: ModuleResolver | None = None,
        config: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """Create a PostgreSQL SQL backend.

        Args:
            name: Workspace backend instance name.
            database: PostgreSQL database name or configured database route.
            ws_id: Optional workspace namespace used to derive workspace database names.
            resolver: Context-owned module authority.
            config: Config authority retained by the backend.
            **kwargs: Additional database connection options forwarded to ``SQLBackend``.

        Returns:
            None: This function does not return a value.
        """
        super().__init__(name, "pg", database, identifier=type(self).identifier, ws_id=ws_id, resolver=resolver, config=config, **kwargs)
