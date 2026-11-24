"""seekdb backend over its MySQL-compatible server protocol."""

from __future__ import annotations

__all__ = [
    "SeekDBBackend",
]

from heavenbase.utils import Any

from ...modules import ModuleResolver
from ._utils import default_database
from .mysql import MySQLBackend


class SeekDBBackend(MySQLBackend):
    """MySQL-wire backend with seekdb connection defaults and routing identity."""

    identifier = "seekdb"

    @classmethod
    def from_config(
        cls,
        name: str,
        data: dict[str, Any],
        *,
        resolver: ModuleResolver | None = None,
        config: Any | None = None,
    ) -> "SeekDBBackend":
        """Build a seekdb backend from a config mapping.

        Args:
            name: Workspace backend instance name.
            data: Backend config mapping. Supports ``database``, optional ``ws_id``, and
                MySQL-compatible connection options.
            resolver: Context-owned module authority.
            config: Config authority retained by the backend.

        Returns:
            SeekDBBackend: Configured seekdb backend instance.
        """
        cfg = dict(data or {})
        ws_id = cfg.pop("ws_id", None)
        return cls(name, default_database(ws_id, cfg), ws_id=ws_id, resolver=resolver, config=config, **cfg)

    def __init__(
        self,
        name: str = "seekdb",
        database: str | None = None,
        ws_id: str | None = None,
        *,
        resolver: ModuleResolver | None = None,
        config: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """Create a seekdb backend through the MySQL wire protocol.

        Args:
            name: Workspace backend instance name.
            database: seekdb database name.
            ws_id: Optional workspace namespace.
            resolver: Context-owned module authority.
            config: Config authority retained by the backend.
            **kwargs: Additional MySQL-compatible connection options.

        Returns:
            None: This function does not return a value.
        """
        super().__init__(name, database, ws_id=ws_id, resolver=resolver, config=config, **kwargs)

    def _database_config(self, values: dict[str, Any]) -> dict[str, Any]:
        """Apply seekdb's documented local server connection defaults, unless a ``url`` names the server."""
        if values.get("url"):
            return values  # its own host, port and user win over the local defaults
        values.setdefault("driver", self._construction_config_get("heavenbase.backend_defaults.seekdb.driver", default="pymysql"))
        values.setdefault("host", self._construction_config_get("heavenbase.backend_defaults.seekdb.host", default="localhost"))
        values.setdefault("port", int(self._construction_config_get("heavenbase.backend_defaults.seekdb.port", default=2881)))
        values.setdefault("username", self._construction_config_get("heavenbase.backend_defaults.seekdb.username", default="root"))
        values.setdefault("password", self._construction_config_get("heavenbase.backend_defaults.seekdb.password", default=""))
        values.setdefault(
            "params",
            self._construction_config_get("heavenbase.backend_defaults.seekdb.params", default={"charset": "utf8mb4"}),
        )
        return values
