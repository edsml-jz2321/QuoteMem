"""Dolt backend over its MySQL-compatible SQL server."""

from __future__ import annotations

__all__ = [
    "DoltBackend",
]

from heavenbase.utils import Any

from ...modules import ModuleResolver
from ._utils import default_database
from .mysql import MySQLBackend


class DoltBackend(MySQLBackend):
    """MySQL-wire backend with a distinct Dolt routing identity."""

    identifier = "dolt"

    @classmethod
    def from_config(
        cls,
        name: str,
        data: dict[str, Any],
        *,
        resolver: ModuleResolver | None = None,
        config: Any | None = None,
    ) -> "DoltBackend":
        """Build a Dolt backend from a config mapping.

        Args:
            name: Workspace backend instance name.
            data: Backend config mapping. Supports ``database``, optional ``ws_id``, and
                MySQL-compatible connection options.
            resolver: Context-owned module authority.
            config: Config authority retained by the backend.

        Returns:
            DoltBackend: Configured Dolt backend instance.
        """
        cfg = dict(data or {})
        ws_id = cfg.pop("ws_id", None)
        return cls(name, default_database(ws_id, cfg), ws_id=ws_id, resolver=resolver, config=config, **cfg)

    def __init__(
        self,
        name: str = "dolt",
        database: str | None = None,
        ws_id: str | None = None,
        *,
        resolver: ModuleResolver | None = None,
        config: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """Create a Dolt backend through the MySQL wire protocol.

        Args:
            name: Workspace backend instance name.
            database: Dolt database name.
            ws_id: Optional workspace namespace.
            resolver: Context-owned module authority.
            config: Config authority retained by the backend.
            **kwargs: Additional MySQL-compatible connection options.

        Returns:
            None: This function does not return a value.
        """
        super().__init__(name, database, ws_id=ws_id, resolver=resolver, config=config, **kwargs)
