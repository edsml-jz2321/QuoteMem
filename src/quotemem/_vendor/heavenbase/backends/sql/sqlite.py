"""SQLite backend."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager

__all__ = [
    "SQLiteBackend",
]

from heavenbase.utils import Any
from heavenbase.schema.model import EntitySchema

from ...modules import ModuleResolver
from ..base import backend_locked
from heavenbase.execution.contracts import RowOp
from .base import SQLBackend
from ._utils import default_database
from .parity.sqlite import (
    column_constraints as sqlite_text_constraints,
    column_type as sqlite_text_type,
)


class SQLiteBackend(SQLBackend):
    """SQL backend configured for SQLite."""

    identifier = "sqlite"
    sql_name_warning_provider = "sqlite"

    @classmethod
    def from_config(cls, name: str, data: dict[str, Any], *, resolver: ModuleResolver | None = None, config: Any | None = None) -> "SQLiteBackend":
        """Build the SQLite backend from a config mapping.

        Args:
            name: Workspace backend instance name.
            data: Backend config mapping. Supports ``database``, local file path options,
                optional ``ws_id``, and database connection options. A ``pool`` mapping
                sizes the engine's connection pool and therefore how many threads can
                read at once: ``pool_class`` (``queue`` by default; ``static``, ``null``,
                ``singleton``, and ``assertion`` are also accepted), ``pool_size``,
                ``max_overflow``, and ``pool_timeout`` (``queue`` only), ``pool_pre_ping``,
                ``pool_recycle``, and ``connect_args``. Keys left out keep the provider
                defaults (``pool_size: 5``, ``max_overflow: 10`` for SQLite). ``static``,
                ``assertion``, and ``singleton`` cannot serve two threads at once
                (``singleton`` closes other threads' connections once more than five
                threads use it), so such an engine is private to the backend and its
                reads keep the backend lock across their statements and run one at a
                time; so do reads on
                ``:memory:`` whatever its ``pool_class``, because an
                in-memory database lives inside its first connection and a second
                connection is a separate empty database (``static`` is its default,
                ``singleton`` gives each thread a private database, and ``queue`` and
                ``null`` are not meaningful for it).
            resolver: Context-owned module authority.
            config: Config authority retained by the backend.

        Returns:
            SQLiteBackend: Configured SQLite backend instance.
        """
        cfg = dict(data or {})
        ws_id = cfg.pop("ws_id", None)
        return cls(name, default_database(ws_id, cfg), ws_id=ws_id, resolver=resolver, config=config, **cfg)

    def __init__(
        self,
        name: str = "sqlite",
        database: str | None = None,
        ws_id: str | None = None,
        *,
        resolver: ModuleResolver | None = None,
        config: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """Create a SQLite SQL backend.

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
        super().__init__(name, "sqlite", database, identifier="sqlite", ws_id=ws_id, resolver=resolver, config=config, **kwargs)

    def _column_type(self, dtype: Any) -> Any:
        """Return SQLite's provider-owned physical type for a logical field."""
        return sqlite_text_type(dtype) or super()._column_type(dtype)

    def _column_constraints(self, name: str, dtype: Any) -> tuple[Any, ...]:
        """Return SQLite's enforced physical text constraints."""
        return sqlite_text_constraints(name, dtype)

    @backend_locked
    def ensure(self, schema: EntitySchema, bindings: Sequence[Any]) -> None:
        """Ensure SQLite storage, tolerating a concurrent first-table creator.

        SQLite's check-then-create DDL is not atomic across Backend instances.
        The losing instance's failed call forgets the table it cached, so a
        retry finds the table created by the winner.

        Args:
            schema: Entity schema being registered.
            bindings: Storage bindings routed to this backend.

        Returns:
            None: This function does not return a value.
        """
        try:
            super().ensure(schema, bindings)
        except Exception as error:
            from sqlalchemy.exc import OperationalError

            if not isinstance(error, OperationalError) or "already exists" not in str(error).lower():
                raise
            super().ensure(schema, bindings)

    @backend_locked
    def compare_upsert(
        self,
        row: RowOp,
        *,
        revision_field: str,
        expected_revision: int | None,
    ) -> bool:
        """Conditionally replace one SQLite row in a provider transaction.

        Args:
            row (RowOp): Replacement upsert containing the next revision.
            revision_field (str): Row field holding the current integer revision.
            expected_revision (int | None): Required revision, or `None` when
                the row must be absent.

        Returns:
            bool: Whether the conditional replacement succeeded.

        Raises:
            KeyError: If the row entity has not been ensured.
            ValueError: If `row` is a delete mutation or `revision_field` is unknown.
        """
        if not row.is_upsert():
            raise ValueError("compare_upsert requires an upsert RowOp")
        table = self.tables[row.entity_id]
        if revision_field not in table.c:
            raise ValueError(f"Unknown revision field {revision_field!r} on {row.entity_id!r}")
        values = row.to_row(fields=table.c.keys())
        with self.db.tx():
            if expected_revision is None:
                from sqlalchemy.dialects.sqlite import insert

                response = self.db.orm_execute(insert(table).values(values).on_conflict_do_nothing(index_elements=[table.c.object_id]), readonly=False)
            else:
                from sqlalchemy import update

                stmt = update(table).where(table.c.object_id == row.object_id).where(table.c[revision_field] == expected_revision).values(values)
                response = self.db.orm_execute(stmt, readonly=False)
        return response.row_count == 1

    @contextmanager
    def _guard_registration_rows(self, entity_id: str, object_ids: Sequence[str]) -> Iterator[None]:
        """Acquire SQLite's write lock before comparing and restoring MetaSchema."""
        from sqlalchemy import false, update

        with self._lock, self._row_write_tx(maintain_strategies=True):
            table = self.tables[entity_id]
            # Even an UPDATE that matches no rows opens SQLite's write transaction.
            self.db.orm_execute(update(table).where(false()).values(object_id=table.c.object_id), readonly=False)
            yield
