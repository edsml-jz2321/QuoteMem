"""ClickHouse backend with MergeTree lifecycle and non-transactional writes."""

from __future__ import annotations

__all__ = [
    "ClickHouseBackend",
]

from functools import lru_cache

from heavenbase.execution.contracts import RowOp
from heavenbase.utils import Any, Sequence
from heavenbase.utils.serialize import dumps_b64, dumps_json, loads_b64, loads_json

from ...modules import ModuleResolver
from ...schema.model import EntitySchema
from ...types.storage import stores_as_integer
from ..base import backend_locked
from ._scoped import ScopedSQLTableMixin
from .base import SQLBackend


@lru_cache(maxsize=1)
def _clickhouse_text_types() -> tuple[type[Any], type[Any], type[Any]]:
    """Return lazily defined ClickHouse text codecs for structured values."""
    import clickhouse_connect
    from clickhouse_connect.cc_sqlalchemy import types

    class _ClickHouseJSONText(types.String):
        """ClickHouse String that preserves HeavenBase JSON values."""

        def bind_processor(self, _dialect: Any) -> Any:
            return lambda value: None if value is None else dumps_json(value, indent=None)

        def result_processor(self, _dialect: Any, _coltype: Any) -> Any:
            return lambda value: None if value is None else (loads_json(value) if isinstance(value, str) else value)

    class _ClickHouseVectorText(_ClickHouseJSONText):
        """ClickHouse String that preserves vectors as JSON float arrays."""

        def bind_processor(self, _dialect: Any) -> Any:
            return lambda value: None if value is None else dumps_json([float(item) for item in value], indent=None)

        def result_processor(self, _dialect: Any, _coltype: Any) -> Any:
            def decode(value: Any) -> Any:
                if value is None:
                    return None
                items = loads_json(value) if isinstance(value, str) else value
                return [float(item) for item in items]

            return decode

    class _ClickHouseArtifactText(types.String):
        """ClickHouse String that stores artifacts through canonical base64."""

        def bind_processor(self, _dialect: Any) -> Any:
            def encode(value: Any) -> Any:
                if value is None:
                    return None
                if isinstance(value, str):
                    data = loads_b64(value)
                elif isinstance(value, memoryview):
                    data = bytes(value)
                elif isinstance(value, bytearray):
                    data = bytes(value)
                elif isinstance(value, bytes):
                    data = value
                else:
                    raise TypeError(f"Artifact expects bytes, got {type(value).__name__}")
                return dumps_b64(data)

            return encode

        def result_processor(self, _dialect: Any, _coltype: Any) -> Any:
            return lambda value: None if value is None else str(value)

    return _ClickHouseJSONText, _ClickHouseVectorText, _ClickHouseArtifactText


def _clickhouse_col(dtype: Any) -> Any:
    """Return a ClickHouse-native SQLAlchemy type for one logical type."""
    import clickhouse_connect
    from clickhouse_connect.cc_sqlalchemy import types

    json_text, vector_text, artifact_text = _clickhouse_text_types()
    identifier = type(dtype).identifier
    if identifier == "categorical":
        return types.Int64() if dtype.base == "integer" else types.String()
    if stores_as_integer(dtype):
        return types.Int64()
    if identifier in ("date", "identifier", "short-text", "medium-text", "long-text"):
        return types.String()
    if identifier == "float":
        return types.Float64()
    if identifier == "boolean":
        return types.Bool()
    if identifier == "artifact":
        return artifact_text()
    if identifier == "vector":
        return vector_text()
    return json_text()


class ClickHouseBackend(ScopedSQLTableMixin, SQLBackend):
    """ClickHouse row backend over ClickHouse Connect's SQLAlchemy dialect."""

    identifier = "clickhouse"
    scoped_table_max_identifier_length = 127
    sql_name_warning_provider = "clickhousedb"

    @classmethod
    def from_config(
        cls,
        name: str,
        data: dict[str, Any],
        *,
        resolver: ModuleResolver | None = None,
        config: Any | None = None,
    ) -> "ClickHouseBackend":
        """Build a ClickHouse backend from a config mapping.

        Args:
            name: Workspace backend instance name.
            data: Backend config mapping. Supports ``database``, optional ``ws_id``, and
                ClickHouse Connect connection options.
            resolver: Context-owned module authority.
            config: Config authority retained by the backend.

        Returns:
            ClickHouseBackend: Configured ClickHouse backend instance.
        """
        cfg = dict(data or {})
        ws_id = cfg.pop("ws_id", None)
        database = cfg.pop("database", None) or "default"
        return cls(name, database, ws_id=ws_id, resolver=resolver, config=config, **cfg)

    def __init__(
        self,
        name: str = "clickhouse",
        database: str | None = None,
        ws_id: str | None = None,
        *,
        resolver: ModuleResolver | None = None,
        config: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """Create a ClickHouse backend.

        Args:
            name: Workspace backend instance name.
            database: Existing ClickHouse database name. Defaults to ``default``.
            ws_id: Optional workspace namespace used to scope physical table names.
            resolver: Context-owned module authority.
            config: Config authority retained by the backend.
            **kwargs: Additional ClickHouse Connect connection options.

        Returns:
            None: This function does not return a value.
        """
        super().__init__(
            name,
            "clickhouse",
            database or "default",
            identifier=type(self).identifier,
            ws_id=ws_id,
            resolver=resolver,
            config=config,
            **kwargs,
        )

    def _install_strategy_adapters(self) -> None:
        """Avoid row-store strategy tables and transactions on ClickHouse."""
        self._strategy_adapters.clear()

    def _column_type(self, dtype: Any) -> Any:
        """Return a ClickHouse-native physical type for one logical type."""
        return _clickhouse_col(dtype)

    def _table_args(self, *, order_by: Sequence[str]) -> tuple[Any, ...]:
        """Attach a MergeTree engine ordered by the provider access key."""
        import clickhouse_connect
        from clickhouse_connect.cc_sqlalchemy import engines

        key: str | tuple[str, ...] = order_by[0] if len(order_by) == 1 else tuple(order_by)
        return (engines.MergeTree(order_by=key),)

    def _drop_owned_tables(self, db: Any) -> None:
        """Drop scoped tables without relying on clickhouse-connect's legacy inspector."""
        rows = db.execute("SELECT name FROM system.tables WHERE database = currentDatabase()", readonly=True).to_dicts()
        names = sorted((str(row["name"]) for row in rows if row.get("name") is not None and self._owns_table(str(row["name"]))), reverse=True)
        for name in names:
            db.execute(f"DROP TABLE IF EXISTS {db.dialect_spec.quote_identifier(name)}", readonly=False)

    def ensure(self, schema: EntitySchema, bindings: Sequence[Any]) -> None:
        """Ensure scalar ClickHouse storage while rejecting ordered side-table types.

        Args:
            schema: Entity schema being registered.
            bindings: Storage bindings routed to this backend.

        Returns:
            None: This function does not return a value.

        Raises:
            ValueError: If Array or HyperG placement would require ordered SideTable
                reconstruction that ClickHouse cannot guarantee.
        """
        side_bindings = self._side_table_fields(bindings)
        if side_bindings:
            fields = ", ".join(sorted(binding.field_name for binding in side_bindings))
            raise ValueError(f"ClickHouse does not support ordered Array/HyperG SideTable storage: {fields}")
        super().ensure(schema, bindings)

    def _delete_rows(self, table: Any, object_ids: Sequence[Any]) -> None:
        """Synchronously apply one ClickHouse lightweight-delete batch."""
        from sqlalchemy import delete

        statement = delete(table).where(table.c.object_id.in_(object_ids))
        statement = statement.execution_options(settings={"lightweight_deletes_sync": 1})
        self.db.orm_execute(statement, readonly=False)

    @backend_locked
    def upsert(self, rows: Sequence[RowOp]) -> None:
        """Apply ordered ClickHouse deletes and inserts without claiming transactions.

        Args:
            rows: Backend row mutations. Each delete is made visible before its
                replacement insert is attempted.

        Returns:
            None: This function does not return a value.

        Raises:
            RuntimeError: If the backend has been dropped and not reconnected.

        Notes:
            ClickHouse is not an OLTP transaction manager. A later insert failure does
            not roll back an already completed lightweight delete. Reads of this
            backend instance keep its lock while their statements run, so they never
            see a row between its delete and its insert; another process or backend
            instance on the same tables can.
        """
        self._need_db()
        from sqlalchemy import insert

        for row in rows:
            table = self.tables[row.entity_id]
            self._delete_rows(table, [row.object_id])
            if row.is_upsert():
                self.db.orm_execute(insert(table), [row.to_row(fields=table.c.keys())], readonly=False)

    @backend_locked
    def drop(self) -> None:
        """Drop only tables owned by this ClickHouse backend instance.

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
