"""Workspace-safe table naming for shared SQL database namespaces."""

from __future__ import annotations

from heavenbase.utils import Any
from heavenbase.utils.hash import sha256hash

from ..base import Backend


class ScopedSQLTableMixin:
    """Prefix and bound tables owned inside a shared physical database."""

    scoped_table_max_identifier_length = 63

    def _derived_table_scope_prefix(self) -> str:
        """Derive the physical table prefix for the currently bound workspace."""
        ws_id = str(getattr(self, "ws_id", None) or "standalone")
        name = str(getattr(self, "name", None) or "backend")
        readable = Backend._physical_name(ws_id, name)[:16]
        digest = sha256hash([ws_id, name, getattr(self, "identifier", "sql")])[:12]
        return f"hb_{readable}_{digest}"

    def _table_scope_prefix(self) -> str:
        """Return this backend instance's active physical table prefix."""
        prefix = getattr(self, "_active_table_scope_prefix", None)
        if prefix is None:
            prefix = self._derived_table_scope_prefix()
            self._active_table_scope_prefix = prefix
        return str(prefix)

    def _refresh_workspace(self) -> None:
        """Reset table state whenever a shared-database workspace scope changes."""
        next_prefix = self._derived_table_scope_prefix()
        old_prefix = getattr(self, "_active_table_scope_prefix", None)
        self._active_table_scope_prefix = next_prefix
        next_database = self._database_name()
        if next_database != self.database:
            super()._refresh_workspace()
            return
        if old_prefix == next_prefix:
            return
        self._reset_storage_state()
        self._install_strategy_adapters()

    def _scoped_table_name(self, *parts: str) -> str:
        """Return one bounded, workspace-owned physical table name."""
        prefix = self._table_scope_prefix()
        logical = Backend._physical_name(*parts)
        available = int(self.scoped_table_max_identifier_length) - len(prefix) - 1
        if available < 14:
            raise ValueError("Scoped SQL table identifier limit leaves no room for a logical name")
        if len(logical) > available:
            digest = sha256hash(logical)[:12]
            logical = f"{logical[: available - len(digest) - 1]}_{digest}"
        return f"{prefix}_{logical}"

    def _table_name(self, entity_id: str) -> str:
        """Return a workspace-owned main-table name."""
        return self._scoped_table_name(entity_id)

    def _side_table_name(self, entity_id: str, field_name: str) -> str:
        """Return a workspace-owned side-table name."""
        return self._scoped_table_name(entity_id, field_name)

    def _owns_table(self, table_name: str) -> bool:
        """Return whether one reflected table belongs to this backend instance."""
        return str(table_name).startswith(f"{self._table_scope_prefix()}_")

    def _drop_owned_tables(self, db: Any) -> None:
        """Drop only reflected tables owned by this backend instance."""
        names = sorted((name for name in db.tables() if self._owns_table(name)), reverse=True)
        for name in names:
            db.drop_table(name)
