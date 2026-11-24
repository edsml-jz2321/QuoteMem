"""MySQL backend."""

from __future__ import annotations

__all__ = [
    "MySQLBackend",
]

from heavenbase.utils import Any

from ...modules import ModuleResolver
from .base import SQLBackend
from ._utils import default_database
from .capabilities import register_sql_dialect
from .dialect import (
    SQL_DEFAULT_NATIVE_AGGREGATE_OPS,
    SQL_JSON_AGG_STYLE_ARRAYAGG,
    SQL_REGEX_STYLE_REGEXP_LIKE_CASE_SENSITIVE,
    SQLDialectSpec,
)
from .parity.mysql import MYSQL_TEXT_OPS, text_evidence as mysql_text_evidence, text_filter as mysql_text_filter


class MySQLBackend(SQLBackend):
    """SQL backend configured for MySQL."""

    identifier = "mysql"
    sql_name_warning_provider = "mysql"
    _mysql_exact_ids = True

    @classmethod
    def from_config(cls, name: str, data: dict[str, Any], *, resolver: ModuleResolver | None = None, config: Any | None = None) -> "MySQLBackend":
        """Build the MySQL backend from a config mapping.

        Args:
            name: Workspace backend instance name.
            data: Backend config mapping. Supports ``database``, optional ``ws_id``, and
                database connection options.
            resolver: Context-owned module authority.
            config: Config authority retained by the backend.

        Returns:
            MySQLBackend: Configured MySQL backend instance.
        """
        cfg = dict(data or {})
        ws_id = cfg.pop("ws_id", None)
        return cls(name, default_database(ws_id, cfg), ws_id=ws_id, resolver=resolver, config=config, **cfg)

    def __init__(
        self,
        name: str = "mysql",
        database: str | None = None,
        ws_id: str | None = None,
        *,
        resolver: ModuleResolver | None = None,
        config: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """Create a MySQL SQL backend.

        Args:
            name: Workspace backend instance name.
            database: MySQL database name or configured database route.
            ws_id: Optional workspace namespace used to derive workspace database names.
            resolver: Context-owned module authority.
            config: Config authority retained by the backend.
            **kwargs: Additional database connection options forwarded to ``SQLBackend``.

        Returns:
            None: This function does not return a value.
        """
        super().__init__(name, "mysql", database, identifier=type(self).identifier, ws_id=ws_id, resolver=resolver, config=config, **kwargs)
