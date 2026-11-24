"""PgVector backend: PostgreSQL storage selected as a vector backend type."""

from __future__ import annotations

__all__ = [
    "PgVectorBackend",
]

from heavenbase.utils import Any, Mapping, Sequence

from ...execution.frame.result import ResultFrame
from ...modules import ModuleResolver
from ...schema.model import EntitySchema
from ...types.vector import _scale_cosine_query
from heavenbase.execution.contracts import QueryFragment
from ..base import backend_locked
from ..sql._scoped import ScopedSQLTableMixin
from ..sql.capabilities import register_sql_dialect
from ..sql.dialect import SQLDialectSpec
from ..sql.base import SQLBackend, _sa_col
from ..sql.postgres import PostgresBackend
from ..sql._utils import default_database


def _pgvector_col(dtype: Any, dialect_spec: Any, *, max_chars: int) -> Any:
    if type(dtype).identifier != "vector":
        return _sa_col(dtype, dialect_spec, max_chars=max_chars)
    import pgvector
    from pgvector.sqlalchemy import Vector as PgVectorType

    return PgVectorType(dtype.dim)


class PgVectorBackend(ScopedSQLTableMixin, PostgresBackend):
    """PostgreSQL backend with pgvector-native near search."""

    identifier = "pgvector"
    aliases = ()
    #: ``SparseGramIndex`` fields are matched by scanning the stored rows: 0.1.2.2 kept no keyword index for them and
    #: still writes them without one (see ``SQLSparseGramAdapter.maintains_index``).
    _keeps_keyword_index = False

    @classmethod
    def from_config(
        cls,
        name: str,
        data: Mapping[str, Any],
        *,
        resolver: ModuleResolver | None = None,
        config: Any | None = None,
    ) -> "PgVectorBackend":
        """Build the PgVector backend from a config mapping.

        Args:
            name (str): Workspace backend instance name.
            data (Mapping[str, Any]): Backend config mapping. Supported keys are inherited
                PostgreSQL constructor parameters, plus ``database``, ``ws_id``, and
                ``shared_database``.
            resolver: Context-owned module authority.
            config: Config authority retained by the backend.

        Returns:
            PgVectorBackend: Configured pgvector backend instance.

        Raises:
            TypeError: If ``shared_database`` is not Boolean.
        """
        cfg = dict(data or {})
        ws_id = cfg.pop("ws_id", None)
        shared_database = cfg.pop("shared_database", False)
        db = cfg.pop("database", None) or default_database(ws_id, cfg) or "postgres"
        return cls(
            name,
            db,
            ws_id=ws_id,
            shared_database=shared_database,
            resolver=resolver,
            config=config,
            **cfg,
        )

    def __init__(
        self,
        name: str = "pgvector",
        database: str | None = None,
        ws_id: str | None = None,
        *,
        shared_database: bool = False,
        resolver: ModuleResolver | None = None,
        config: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """Create a PostgreSQL backend with pgvector-native near search.

        Args:
            name (str): Workspace backend instance name.
            database (str | None): PostgreSQL database name or configured database route.
            ws_id (str | None): Optional workspace namespace for database materialization.
            shared_database (bool): Keep ``database`` fixed and scope physical tables by
                workspace and backend name instead of creating a workspace database.
            resolver: Context-owned module authority.
            config: Config authority retained by the backend.
            **kwargs (Any): Additional PostgreSQL backend constructor options.

        Returns:
            None: This function does not return a value.

        Raises:
            TypeError: If ``shared_database`` is not Boolean.
        """
        if not isinstance(shared_database, bool):
            raise TypeError("PgVector shared_database must be Boolean")
        self.shared_database = shared_database
        super().__init__(name, database, ws_id=ws_id, resolver=resolver, config=config, **kwargs)
        self.type = "pgvector"

    def _construction_tag_claims(self) -> dict[str, Any]:
        """Overlay the physical namespace contract selected at construction."""
        return {
            **super()._construction_tag_claims(),
            "workspace-database-namespace": not self.shared_database,
        }

    def _uses_workspace_database(self) -> bool:
        """Use the configured database when shared table scoping is enabled."""
        return False if self.shared_database else super()._uses_workspace_database()

    def _table_name(self, entity_id: str) -> str:
        """Return a scoped or database-local main-table name."""
        if self.shared_database:
            return ScopedSQLTableMixin._table_name(self, entity_id)
        return SQLBackend._table_name(self, entity_id)

    def _side_table_name(self, entity_id: str, field_name: str) -> str:
        """Return a scoped or database-local side-table name."""
        if self.shared_database:
            return ScopedSQLTableMixin._side_table_name(self, entity_id, field_name)
        return SQLBackend._side_table_name(self, entity_id, field_name)

    def _refresh_workspace(self) -> None:
        """Refresh the selected database or shared table scope."""
        if self.shared_database:
            ScopedSQLTableMixin._refresh_workspace(self)
        else:
            SQLBackend._refresh_workspace(self)

    def _install_strategy_adapters(self) -> None:
        """Keep shared pgvector databases free of global strategy tables."""
        if self.shared_database:
            self._strategy_adapters.clear()
        else:
            SQLBackend._install_strategy_adapters(self)

    @backend_locked
    def drop(self) -> None:
        """Drop only this backend's tables in shared-database mode.

        Args:
            None.

        Returns:
            None: This function does not return a value.
        """
        if not self.shared_database:
            SQLBackend.drop(self)
            return
        if self.db is None:
            return
        self._drain_reads(self.db)
        try:
            self._drop_owned_tables(self.db)
        finally:
            self.db.close()
            self._reset_storage_state()
            self.db = None

    def _column_type(self, dtype: Any) -> Any:
        """Store ``Vector`` fields as pgvector columns and other types as PostgreSQL does."""
        if type(dtype).identifier != "vector":
            return super()._column_type(dtype)
        return _pgvector_col(dtype, self.db.dialect_spec, max_chars=self._sql_long_text_max_chars)

    def _side_table_fields(self, bindings: Sequence[Any]) -> Sequence[Any]:
        """Keep no side tables: ``SideTable`` fields are stored in the main table."""
        return []

    def _main_table_fields(self, bindings: Sequence[Any]) -> Sequence[Any]:
        """Store every field routed here, ``SideTable`` fields included, in the main table."""
        return self._my_bindings(bindings)

    @backend_locked
    def ensure(self, schema: EntitySchema, bindings: Sequence[Any]) -> None:
        """Ensure the pgvector extension and the entity's table, as every SQL backend ensures it.

        A column reaches the cached table only once the database holds it, so a
        failed replacement leaves the entity readable through its previous
        registration; see ``SQLBackend._ensure_storage``.

        Args:
            schema (EntitySchema): Entity schema to store in PostgreSQL.
            bindings (Sequence[Any]): Storage bindings supplied by the workspace.

        Returns:
            None: This function does not return a value.

        Raises:
            RuntimeError: If the backend has been dropped and not reconnected.
        """
        self._need_db()
        self.db.execute("CREATE EXTENSION IF NOT EXISTS vector", readonly=False)
        super().ensure(schema, bindings)

    def _distance(self, column: Any, query: list, metric: str) -> Any:
        metric = (metric or "cosine").lower()
        if metric in ("l2", "euclidean"):
            return column.l2_distance(query)
        if metric in ("ip", "inner", "inner_product", "dot"):
            return column.max_inner_product(query)
        # pgvector stores float32 coordinates; cosine ignores the exact power-of-two scale.
        return column.cosine_distance(_scale_cosine_query(query))

    def execute(self, fragment: QueryFragment) -> ResultFrame:
        """Execute pgvector near fragments or delegate to SQL execution.

        A near statement is planned under the backend lock and, like the other
        SQL reads, runs and decodes outside it.

        Args:
            fragment (QueryFragment): Backend-ready query fragment compiled by the handler layer.

        Returns:
            ResultFrame: Matching rows and object identifiers.

        Raises:
            RuntimeError: If the backend has been dropped and not reconnected.
        """
        if fragment.kind() != "pgvector_search" or fragment.op != "near":
            return super().execute(fragment)
        filt = fragment.filter_fragment()
        if filt is not None and not self._is_sql_fragment(filt):
            return super().execute(fragment)
        from sqlalchemy import select

        entity_id = fragment.entity_id()
        params = fragment.payload
        metric = (params.get("metric") or "cosine").lower()
        with self._lock:
            storage = self._read_storage(entity_id) if entity_id else None
            if storage is None:
                return ResultFrame.empty()
            db, table, side_specs = storage
            column = table.c[fragment.field]
            dist = self._distance(column, params["vector"], metric).label("_distance")
            # A row without a vector has no distance, and native search never returns one (nor does the
            # scan scorer), so it is left out under every metric. ORDER BY stays on the bare distance,
            # so a pgvector ANN index on the column can still serve the near.
            stmt = select(*self._select_cols(entity_id, table), dist).where(column.isnot(None)).order_by(dist).limit(params["top_k"])
            if filt is not None:
                stmt = stmt.where(self._where(filt, table, entity_id))
        with self._read_scope(db):
            rows = self._decode_side_rows(side_specs, db.orm_execute(stmt, readonly=True).to_dicts())
        scored = []
        for row in rows:
            distance = row.pop("_distance", None)
            if distance is not None:
                row["score"] = 1.0 - float(distance) if metric == "cosine" else -float(distance)
                scored.append(row)
        return ResultFrame.from_rows(scored)
