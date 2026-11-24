"""Supabase backend over native PostgreSQL connectivity."""

from __future__ import annotations

__all__ = [
    "SupabaseBackend",
]

from heavenbase.utils import Any

from ...modules import ModuleResolver
from ..base import backend_locked
from ._scoped import ScopedSQLTableMixin
from .postgres import PostgresBackend


class SupabaseBackend(ScopedSQLTableMixin, PostgresBackend):
    """PostgreSQL backend with a distinct Supabase routing identity."""

    identifier = "supabase"
    aliases = ()

    @classmethod
    def from_config(
        cls,
        name: str,
        data: dict[str, Any],
        *,
        resolver: ModuleResolver | None = None,
        config: Any | None = None,
    ) -> "SupabaseBackend":
        """Build a Supabase backend from a PostgreSQL connection mapping.

        Args:
            name: Workspace backend instance name.
            data: Backend config mapping. Supports ``database``, optional ``ws_id``, and
                PostgreSQL connection options.
            resolver: Context-owned module authority.
            config: Config authority retained by the backend.

        Returns:
            SupabaseBackend: Configured Supabase backend instance.
        """
        cfg = dict(data or {})
        ws_id = cfg.pop("ws_id", None)
        database = cfg.pop("database", None) or "postgres"
        return cls(name, database, ws_id=ws_id, resolver=resolver, config=config, **cfg)

    def __init__(
        self,
        name: str = "supabase",
        database: str | None = None,
        ws_id: str | None = None,
        *,
        resolver: ModuleResolver | None = None,
        config: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """Create a Supabase backend through the PostgreSQL wire protocol.

        Args:
            name: Workspace backend instance name.
            database: Supabase PostgreSQL database name. Defaults to ``postgres``.
            ws_id: Optional workspace namespace.
            resolver: Context-owned module authority.
            config: Config authority retained by the backend.
            **kwargs: Additional PostgreSQL connection options.

        Returns:
            None: This function does not return a value.
        """
        super().__init__(name, database or "postgres", ws_id=ws_id, resolver=resolver, config=config, **kwargs)

    def _install_strategy_adapters(self) -> None:
        """Keep shared Supabase databases free of global SQL strategy tables."""
        self._strategy_adapters.clear()

    @backend_locked
    def drop(self) -> None:
        """Drop only tables owned by this Supabase backend instance.

        Args:
            None.

        Returns:
            None: This function does not return a value.
        """
        if self.db is None:
            return
        self._drain_reads(self.db)
        try:
            self._drop_owned_tables(self.db)
        finally:
            self.db.close()
            self._reset_storage_state()
            self.db = None
