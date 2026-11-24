"""StarRocks backend over the MySQL wire protocol."""

from __future__ import annotations

__all__ = [
    "StarRocksBackend",
]

import warnings
from contextlib import contextmanager

from heavenbase.utils import Any, Iterator, Sequence, safe_error

from ...modules import ModuleResolver
from ...database.resources.sql.starrocks import SR_ADD_COLUMN, SR_CREATE_DATABASE, SR_CREATE_OLAP_TABLE, SR_DROP_TABLE
from ...schema.model import EntitySchema
from ...types.registry import get_logical_type
from ...types.storage import stores_as_integer
from ...utils.ids import check_entity_id
from ..base import backend_locked
from .base import SQLBackend
from ._utils import default_database
from .capabilities import register_sql_dialect
from .dialect import (
    SQL_JSON_AGG_STYLE_ARRAYAGG,
    SQL_REGEX_STYLE_REGEXP_LIKE_CASE_SENSITIVE,
    SQLDialectSpec,
)

# ``Column.info`` key holding the StarRocks type a column's DDL declares.
_SR_TYPE = "starrocks_type"


def _q(name: str) -> str:
    return f"`{str(name).replace('`', '``')}`"


def _sr_varchar_len(length: Any, config: Any) -> int:
    """Cap StarRocks VARCHAR widths to its configured hard limit."""
    max_len = int(config.get("heavenbase.backend_defaults.starrocks.varchar.max_length", default=65533))
    return min(int(length), max_len)


def _sr_type(dtype: Any, *, config: Any, resolver: ModuleResolver) -> str:
    cls = type(dtype)
    identifier = cls.identifier if "identifier" in cls.__dict__ else get_logical_type(getattr(dtype, "identifier", cls.__name__), resolver=resolver).identifier
    if identifier == "categorical":
        return "BIGINT" if dtype.base == "integer" else f"VARCHAR({_sr_varchar_len(dtype.max_chars, config)})"
    if stores_as_integer(dtype):
        return "BIGINT"
    if identifier == "date":
        return f"VARCHAR({_sr_varchar_len(dtype.max_chars, config)})"
    if identifier in ("identifier", "short-text", "medium-text", "long-text"):
        return f"VARCHAR({_sr_varchar_len(dtype.max_chars, config)})"
    if identifier == "float":
        return "DOUBLE"
    if identifier == "boolean":
        return "BOOLEAN"
    return f"VARCHAR({_sr_varchar_len(65533, config)})"


class StarRocksBackend(SQLBackend):
    """SQL backend for StarRocks using MySQL-compatible connectivity."""

    identifier = "starrocks"
    sql_name_warning_provider = "mysql"
    #: ``SparseGramIndex`` fields are matched by scanning the stored rows: 0.1.2.2 kept no keyword index for them and
    #: still writes them without one (see ``SQLSparseGramAdapter.maintains_index``).
    _keeps_keyword_index = False

    @classmethod
    def from_config(cls, name: str, data: dict[str, Any], *, resolver: ModuleResolver | None = None, config: Any | None = None) -> "StarRocksBackend":
        """Build the StarRocks backend from a config mapping.

        Args:
            name: Workspace backend instance name.
            data: Backend config mapping. Supports ``database``, optional ``ws_id``, and
                database connection options.
            resolver: Context-owned module authority.
            config: Config authority retained by the backend.

        Returns:
            StarRocksBackend: Configured StarRocks backend instance.
        """
        cfg = dict(data or {})
        ws_id = cfg.pop("ws_id", None)
        return cls(name, default_database(ws_id, cfg), ws_id=ws_id, resolver=resolver, config=config, **cfg)

    def __init__(
        self,
        name: str = "starrocks",
        database: str | None = None,
        ws_id: str | None = None,
        *,
        resolver: ModuleResolver | None = None,
        config: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """Create a StarRocks SQL backend.

        Args:
            name: Workspace backend instance name.
            database: StarRocks database name or configured database route.
            ws_id: Optional workspace namespace used to derive workspace database names.
            resolver: Context-owned module authority.
            config: Config authority retained by the backend.
            **kwargs: Additional database connection options forwarded to ``SQLBackend``.

        Returns:
            None: This function does not return a value.
        """
        super().__init__(name, "mysql", database, identifier="starrocks", ws_id=ws_id, resolver=resolver, config=config, **kwargs)

    def _database_config(self, values: dict[str, Any]) -> dict[str, Any]:
        """Resolve StarRocks connection defaults from this backend's config authority."""
        values.setdefault("driver", self._construction_config_get("heavenbase.backend_defaults.starrocks.driver", default="pymysql"))
        values.setdefault("host", self._construction_config_get("heavenbase.backend_defaults.starrocks.host", default="localhost"))
        values.setdefault("port", int(self._construction_config_get("heavenbase.backend_defaults.starrocks.port", default=9030)))
        values.setdefault("username", self._construction_config_get("heavenbase.backend_defaults.starrocks.username", default="root"))
        values.setdefault("password", self._construction_config_get("heavenbase.backend_defaults.starrocks.password", default=""))
        values.setdefault(
            "params",
            self._construction_config_get("heavenbase.backend_defaults.starrocks.params", default={"charset": "utf8mb4"}),
        )
        self._fallback_database = self._construction_config_get("heavenbase.backend_defaults.starrocks.database", default="heavenbase")
        self.client_error = ""
        return values

    def _database_name(self) -> str | None:
        database = self._database_override or (self._physical_name(self.ws_id) if self.ws_id else None) or self._fallback_database
        return check_entity_id(database) if database else None

    def _connect_database(self) -> None:
        database = self._database_name()
        try:
            self._ensure_database(database, self._db_kwargs)
        except Exception as exc:
            self.client_error = safe_error(exc)
        super()._connect_database()

    def _ensure_database(self, database: str | None, kwargs: dict) -> None:
        if not database:
            return
        from heavenbase.database import Database

        root = Database(provider="mysql", database=None, resolver=self._module_resolver, config=self.config, **kwargs)
        root.execute(SR_CREATE_DATABASE.render(database=_q(database)), readonly=False)

    def _column(self, name: str, dtype: Any, *, primary_key: bool = False, nullable: bool = True) -> Any:
        """Build one column and record the StarRocks type its DDL declares."""
        column = super()._column(name, dtype, primary_key=primary_key, nullable=nullable)
        if name == "object_id":
            column.info[_SR_TYPE] = f"VARCHAR({self._object_id_varchar_length(dtype)})"
        else:
            column.info[_SR_TYPE] = _sr_type(dtype, config=self.config, resolver=self.module_resolver)
        return column

    def _side_table_fields(self, bindings: Sequence[Any]) -> Sequence[Any]:
        """Keep no side tables: StarRocks stores every field in its OLAP table."""
        return []

    def _main_table_fields(self, bindings: Sequence[Any]) -> Sequence[Any]:
        """Store every field routed here, ``SideTable`` fields included, in the OLAP table."""
        return self._my_bindings(bindings)

    @staticmethod
    def _column_ddl(column: Any) -> str:
        return f"{_q(column.name)} {column.info[_SR_TYPE]} {'NULL' if column.nullable else 'NOT NULL'}"

    @staticmethod
    @contextmanager
    def _reflecting_olap_columns() -> Iterator[None]:
        """Silence SQLAlchemy's notes on the OLAP clauses of ``SHOW CREATE TABLE`` while the stored columns are reflected.

        SQLAlchemy's MySQL reflection reads the column names correctly but warns
        about each StarRocks clause it does not know (``PRIMARY KEY(...)``,
        ``DISTRIBUTED BY ...``, ``PROPERTIES (...)``). Only those warnings are
        silenced.
        """
        from sqlalchemy.exc import SAWarning

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Unknown schema content", category=SAWarning)
            yield

    def _add_columns(self, table: Any, columns: Sequence[Any]) -> None:
        with self._reflecting_olap_columns():
            super()._add_columns(table, columns)

    def _sync_columns(self, table: Any) -> None:
        with self._reflecting_olap_columns():
            super()._sync_columns(table)

    def _create_main_table(self, table: Any) -> None:
        """Create the entity's OLAP table, keyed and distributed by ``object_id``, unless it exists."""
        buckets = int(self.config.get("heavenbase.backend_defaults.starrocks.buckets", default=1))
        replication = int(self.config.get("heavenbase.backend_defaults.starrocks.replication_num", default=1))
        columns = ", ".join(self._column_ddl(column) for column in table.c)
        ddl = SR_CREATE_OLAP_TABLE.render(table=_q(table.name), columns=columns, object_id=_q("object_id"), buckets=buckets, replication=replication)
        self.db.execute(ddl, readonly=False)
        self.db.invalidate_schema_index()

    def _add_stored_column(self, table: Any, column: Any) -> None:
        """Add one column to the stored OLAP table with its StarRocks type."""
        self.db.execute(SR_ADD_COLUMN.render(table=_q(table.name), column=self._column_ddl(column)), readonly=False)
        self.db.invalidate_schema_index()

    def _object_id_length(self, schema: EntitySchema) -> int:
        """Return the ``object_id`` VARCHAR length a new table declares for ``schema``.

        The column takes the entity's own ``object_id`` domain, capped by the
        StarRocks VARCHAR limit and, when it is set, by
        ``heavenbase.backend_defaults.starrocks.object_id.max_length``.
        """
        from ...types import Identifier

        object_id = schema.fields.get("object_id")
        return self._object_id_varchar_length(object_id.dtype if object_id is not None else Identifier())

    def _object_id_varchar_length(self, dtype: Any) -> int:
        """Return the ``object_id`` VARCHAR length for an ``object_id`` of logical type ``dtype``; see ``_object_id_length``."""
        length = dtype.max_chars
        cap = self.config.get("heavenbase.backend_defaults.starrocks.object_id.max_length", default=None)
        return _sr_varchar_len(length if cap is None else min(int(length), int(cap)), self.config)

    @backend_locked
    def drop(self) -> None:
        """Drop StarRocks tables and dispose the active engine.

        Args:
            None.

        Returns:
            None: This function does not return a value.

        Notes:
            StarRocks external root connections often lack ``DROP DATABASE`` privileges, so
            this provider drops tables individually.
        """
        if self.db is None:
            return
        self._drain_reads(self.db)
        try:
            if self.database:
                for table in list(self.tables.values()):
                    self.db.execute(SR_DROP_TABLE.render(table=_q(table.name)), readonly=False, safe=True)
        except Exception:
            pass
        self.db.dispose()
        self._reset_storage_state()
        self.db = None

    def to_dict(self) -> dict[str, Any]:
        """Return StarRocks backend metadata and connection state.

        Args:
            None.

        Returns:
            dict[str, Any]: Backend name, type, provider, database, and connection error text.
        """
        return {"name": self.name, "type": self.type, "provider": self.provider, "database": self.database, "error": self.client_error}
