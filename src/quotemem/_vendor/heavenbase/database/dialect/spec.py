"""Physical database dialect metadata and tag value contract."""

from __future__ import annotations

__all__ = [
    "DatabaseDialectSpec",
    "MYSQL_TEXT_MIN_LEN",
    "ORACLE_VARCHAR_MAX",
    "SQL_COMMON_RESERVED_WORDS",
    "SQL_TYPE_CLOB",
    "SQL_TYPE_FLOAT64_BLOB",
    "SQL_TYPE_FLOAT_ARRAY",
    "SQL_TYPE_JSON",
    "SQL_TYPE_JSON_TEXT",
    "SQL_TYPE_MYSQL_TEXT",
    "SQL_TYPE_POSTGRESQL_JSONB",
    "SQL_TYPE_STRING",
    "SQL_TYPE_TEXT",
]

from importlib import import_module
from string import Formatter
from types import MappingProxyType

from ..resources.sql import (
    DB_ADD_COLUMN,
    DB_DDL_AFTER_CREATE,
    DB_DDL_BEFORE_DROP_FORCE,
    DB_DDL_CREATE,
    DB_DDL_DROP,
    DB_DDL_EXISTS,
    DB_RENAME_TABLE,
)
from ...utils.ids import check_id
from ...utils.typing import Any, Callable, Dict, Iterable, Mapping, Optional, Tuple, dataclass, field

ORACLE_VARCHAR_MAX = 4000
MYSQL_TEXT_MIN_LEN = 16384

SQL_TYPE_CLOB = "clob"
SQL_TYPE_FLOAT64_BLOB = "float64_blob"
SQL_TYPE_FLOAT_ARRAY = "float_array"
SQL_TYPE_JSON = "json"
SQL_TYPE_JSON_TEXT = "json_text"
SQL_TYPE_MYSQL_TEXT = "mysql_text"
SQL_TYPE_POSTGRESQL_JSONB = "postgresql_jsonb"
SQL_TYPE_STRING = "string"
SQL_TYPE_TEXT = "text"
_SQL_TYPE_KEYS = {
    SQL_TYPE_CLOB,
    SQL_TYPE_FLOAT64_BLOB,
    SQL_TYPE_FLOAT_ARRAY,
    SQL_TYPE_JSON,
    SQL_TYPE_JSON_TEXT,
    SQL_TYPE_MYSQL_TEXT,
    SQL_TYPE_POSTGRESQL_JSONB,
    SQL_TYPE_STRING,
    SQL_TYPE_TEXT,
}
_SQL_TEXT_TYPE_KEYS = {SQL_TYPE_CLOB, SQL_TYPE_MYSQL_TEXT, SQL_TYPE_STRING, SQL_TYPE_TEXT}
_SQL_JSON_NATIVE_TYPE_KEYS = {SQL_TYPE_JSON, SQL_TYPE_POSTGRESQL_JSONB}
_SQL_VECTOR_NATIVE_TYPE_KEYS = {SQL_TYPE_FLOAT_ARRAY}
_SQL_VECTOR_PACKED_TYPE_KEYS = {SQL_TYPE_FLOAT64_BLOB, SQL_TYPE_JSON_TEXT}


def _word_set(text: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(text).split()))


SQL_COMMON_RESERVED_WORDS = _word_set(
    "all alter and any as asc between by case cast check column constraint create cross current current_date "
    "current_time current_timestamp delete desc distinct drop else end except exists false for foreign from full "
    "grant group having in inner insert intersect into is join left like limit natural not null on or order outer "
    "primary references right select set table then true union unique update using values when where with"
)


@dataclass(frozen=True)
class DatabaseDialectSpec:
    """Physical SQLAlchemy dialect metadata for portable database types.

    Args:
        identifier: SQLAlchemy dialect name such as ``"sqlite"`` or ``"postgresql"``.
        tags: Construction-only or Registry-derived physical-dialect tag values.
            Normal Registry-backed specs source these values exclusively from
            the dialect module record; bootstrap specs source them from the
            dedicated bootstrap dialect mapping.
        varchar_max: Optional max VARCHAR length before portable text storage should use a large text type.
        text_min_len: Optional minimum String length that should use an unbounded text type.
        superuser_database: Optional database token used when a superuser connection omits an explicit database.
            Use ``"{database}"`` when the maintenance connection should target the resolved user database.
        sqlglot_dialect: SQLGlot dialect name for parsing, formatting, and transpilation. Defaults to ``identifier``.
        sqlalchemy_aliases: Additional SQLAlchemy dialect names that resolve to this dialect metadata.
        sqlglot_preferred: Whether reverse SQLGlot-to-SQLAlchemy lookup should choose this dialect for ``sqlglot_dialect``.
        current_database_sql: Optional resource-backed SQL query whose first row exposes the active database as ``name``.
        list_databases_sql: Optional resource-backed SQL query whose rows expose visible databases as ``name``.
        static_database_names: Static database names for dialects that do not expose a database-list query.
        ddl_resource_key: Optional SQL resource key for create/drop namespace templates. Defaults to ``identifier``.
        identifier_quote_style: Identifier quoting style used by lifecycle DDL templates: ``"double"``, ``"backtick"``, or ``"bracket"``.
        lifecycle_namespace_source: DBSpec attribute used as the lifecycle namespace. ``"schema_or_username"`` supports Oracle user/schema DDL.
        lifecycle_namespace_upper: Whether lifecycle DDL should render the target namespace uppercase.
        requires_host: Whether URL materialization requires a host when the dialect does not use local database paths.
        url_database_omit_params: Query params that make SQLAlchemy URL materialization omit the database path token.
        scope_style: Provider-owned database scope contract: ``"database"``,
            ``"database_only"``, ``"catalog_schema"``, or ``"service_sid_dsn"``.
        supports_memory: Whether an empty or ``:memory:`` database token selects
            an in-memory database for this dialect.
        file_uri_query: Whether local file URIs use SQLAlchemy's ``file:...``
            database token plus ``uri=true`` query convention.
        memory_pool_class: Optional default pool class used for in-memory specs.
        memory_connect_args: Default DBAPI connect args used for in-memory specs.
        retryable_error_messages: Lowercase exception-message fragments that should trigger transient execution retry.
        after_create_sql_keys: Resource keys in ``DB_DDL_AFTER_CREATE`` to run after namespace creation.
        owner_after_create_sql_keys: Resource keys in ``DB_DDL_AFTER_CREATE`` to run only when the requested user differs from the superuser.
        before_drop_force_sql_key: Resource key in ``DB_DDL_BEFORE_DROP_FORCE`` to run before a forced namespace drop.
        before_drop_force_mode: Render mode for forced-drop SQL: ``"bind_name"`` or ``"quoted_database"``.
        rename_table_sql: Resource-backed SQL template for table rename DDL.
        rename_table_bind_params: Whether ``rename_table_sql`` expects ``old_name`` and ``new_name`` as bind params.
        add_column_sql: Resource-backed DDL template for adding one column. Templates use
            ``{table}`` and ``{column}`` placeholders for already-rendered SQL fragments.
        column_reflection_sql: Optional fallback SQL template for reflected columns. Templates can use ``{table_name}`` and
            ``{table_name_literal}`` placeholders and should return columns named like SQLAlchemy/DuckDB inspection rows.
        primary_key_reflection_sql: Optional fallback SQL template for reflected primary keys, used when the SQLAlchemy
            inspector fails or reports none. Templates can use ``{table_name}``, ``{table_name_literal}``,
            ``{table_literal}`` and ``{schema_literal}`` placeholders and should return rows with ``name`` and ``pk``
            fields, the key columns in declared key order.
        index_reflection_sql: Optional SQL template that reads a table's indexes from the database catalog, for a
            dialect whose SQLAlchemy inspector reports none (DuckDB). Templates can use ``{table_literal}`` and
            ``{schema_literal}`` (the schema as ``Database.schemas()`` names it, such as DuckDB's ``db.main``;
            ``NULL`` when no schema is given) and return rows with ``name``, ``unique`` and
            ``columns`` fields, ``columns`` being a ``[a, b]`` list text or a sequence of column names. An
            optional ``definition`` field holding the ``CREATE INDEX`` statement supplies the columns of a row
            whose ``columns`` is ``NULL``.
        empty_foreign_keys_on_reflection_error: Whether foreign-key reflection errors should degrade to an empty list.
        varchar_overflow_type: SQLAlchemy storage type token used when a String exceeds ``varchar_max``.
        text_threshold_type: SQLAlchemy storage type token used when String length reaches ``text_min_len``.
        blob_text_type: SQLAlchemy storage type token for base64 Artifact fallback when native BLOB is unavailable.
        json_native_type: Optional SQLAlchemy storage type token for native JSON columns.
        json_text_type: SQLAlchemy storage type token for JSON string fallback columns.
        vector_native_type: Optional SQLAlchemy storage type token for native Vector columns.
        vector_packed_type: Optional storage form for Vector columns when no native vector type exists.
            ``float64_blob`` (an opt-in; no packaged dialect record declares it) stores each vector as a
            little-endian IEEE-754 float64 BLOB, so values round-trip bit-exactly, and still reads JSON text
            through the same column. It requires the ``database-dynamic-typing`` tag (SQLite), whose columns
            hold a BLOB whatever their declared type; other dialects without a native vector type refuse it.
            ``json_text``, like ``None``, keeps the JSON text form of ``json_text_type`` and refuses a packed
            row. A HeavenBase release without packed storage cannot read packed rows.
        name_warning_dialect_factory: Optional ``(module, attr)`` pair used to ask SQLAlchemy for identifier length and
            reserved words.
        name_warning_reserved_words: Provider-owned fallback reserved words for schema warning messages.
        name_warning_max_identifier_length: Optional provider-owned fallback identifier length for schema warning messages.
    """

    identifier: str
    tags: Mapping[str, Any] = field(default_factory=dict)
    varchar_max: Optional[int] = None
    text_min_len: Optional[int] = None
    superuser_database: Optional[str] = None
    sqlglot_dialect: Optional[str] = None
    sqlalchemy_aliases: tuple[str, ...] = ()
    sqlglot_preferred: bool = True
    current_database_sql: Optional[str] = None
    list_databases_sql: Optional[str] = None
    static_database_names: tuple[str, ...] = ()
    ddl_resource_key: Optional[str] = None
    identifier_quote_style: str = "double"
    lifecycle_namespace_source: str = "database"
    lifecycle_namespace_upper: bool = False
    requires_host: bool = True
    url_database_omit_params: tuple[str, ...] = ()
    scope_style: str = "database"
    supports_memory: bool = False
    file_uri_query: bool = False
    memory_pool_class: Optional[str] = None
    memory_connect_args: Mapping[str, Any] = field(default_factory=dict)
    retryable_error_messages: tuple[str, ...] = ()
    after_create_sql_keys: tuple[str, ...] = ()
    owner_after_create_sql_keys: tuple[str, ...] = ()
    before_drop_force_sql_key: Optional[str] = None
    before_drop_force_mode: str = "bind_name"
    rename_table_sql: Optional[str] = None
    rename_table_bind_params: bool = False
    add_column_sql: Optional[str] = None
    column_reflection_sql: Optional[str] = None
    primary_key_reflection_sql: Optional[str] = None
    index_reflection_sql: Optional[str] = None
    empty_foreign_keys_on_reflection_error: bool = False
    varchar_overflow_type: str = SQL_TYPE_CLOB
    text_threshold_type: str = SQL_TYPE_MYSQL_TEXT
    blob_text_type: str = SQL_TYPE_TEXT
    json_native_type: Optional[str] = None
    json_text_type: str = SQL_TYPE_STRING
    vector_native_type: Optional[str] = None
    name_warning_dialect_factory: Optional[Tuple[str, str]] = None
    name_warning_reserved_words: tuple[str, ...] = SQL_COMMON_RESERVED_WORDS
    name_warning_max_identifier_length: Optional[int] = None
    vector_packed_type: Optional[str] = None

    def __post_init__(self) -> None:
        identifier = check_id(str(self.identifier).strip().lower(), "database dialect identifier")
        if not isinstance(self.tags, Mapping) or not all(isinstance(tag_identifier, str) for tag_identifier in self.tags):
            raise TypeError("Database dialect tags must be a string-keyed mapping")
        tags = _freeze_data(dict(self.tags), "tags")
        varchar_max = None if self.varchar_max is None else int(self.varchar_max)
        text_min_len = None if self.text_min_len is None else int(self.text_min_len)
        superuser_database = None if self.superuser_database is None else str(self.superuser_database)
        sqlglot_dialect = None if self.sqlglot_dialect is None else check_id(str(self.sqlglot_dialect).strip().lower(), "SQLGlot dialect identifier")
        sqlalchemy_aliases = tuple(check_id(str(alias).strip().lower(), "SQLAlchemy dialect alias") for alias in self.sqlalchemy_aliases)
        current_database_sql = _clean_sql(self.current_database_sql, "current_database_sql")
        list_databases_sql = _clean_sql(self.list_databases_sql, "list_databases_sql")
        static_database_names = tuple(str(name).strip() for name in self.static_database_names)
        ddl_resource_key = None if self.ddl_resource_key is None else check_id(str(self.ddl_resource_key).strip().lower(), "database DDL resource key")
        identifier_quote_style = str(self.identifier_quote_style or "double").strip().lower()
        lifecycle_namespace_source = str(self.lifecycle_namespace_source or "database").strip().lower()
        url_database_omit_params = tuple(str(param).strip() for param in self.url_database_omit_params)
        scope_style = str(self.scope_style or "database").strip().lower()
        memory_pool_class = None if self.memory_pool_class is None else str(self.memory_pool_class).strip().lower()
        memory_connect_args = _freeze_data(dict(self.memory_connect_args or {}), "memory_connect_args")
        retryable_error_messages = tuple(str(message).strip().lower() for message in self.retryable_error_messages)
        after_create_sql_keys = tuple(check_id(str(key).strip().lower(), "after-create DDL key") for key in self.after_create_sql_keys)
        owner_after_create_sql_keys = tuple(check_id(str(key).strip().lower(), "owner after-create DDL key") for key in self.owner_after_create_sql_keys)
        before_drop_force_sql_key = (
            None if self.before_drop_force_sql_key is None else check_id(str(self.before_drop_force_sql_key).strip().lower(), "before-drop-force DDL key")
        )
        before_drop_force_mode = str(self.before_drop_force_mode or "bind_name").strip().lower()
        rename_table_sql = _clean_sql(self.rename_table_sql, "rename_table_sql") or DB_RENAME_TABLE["generic"].text
        add_column_sql = _clean_sql_template(
            DB_ADD_COLUMN["generic"].text if self.add_column_sql is None else self.add_column_sql,
            "add_column_sql",
            fields=("table", "column"),
        )
        column_reflection_sql = _clean_sql(self.column_reflection_sql, "column_reflection_sql")
        primary_key_reflection_sql = _clean_sql(self.primary_key_reflection_sql, "primary_key_reflection_sql")
        index_reflection_sql = _clean_sql(self.index_reflection_sql, "index_reflection_sql")
        varchar_overflow_type = _clean_sql_type(self.varchar_overflow_type, "varchar_overflow_type", allowed=_SQL_TEXT_TYPE_KEYS)
        text_threshold_type = _clean_sql_type(self.text_threshold_type, "text_threshold_type", allowed=_SQL_TEXT_TYPE_KEYS)
        blob_text_type = _clean_sql_type(self.blob_text_type, "blob_text_type", allowed=_SQL_TEXT_TYPE_KEYS)
        json_native_type = _clean_sql_type(self.json_native_type, "json_native_type", optional=True, allowed=_SQL_JSON_NATIVE_TYPE_KEYS)
        json_text_type = _clean_sql_type(self.json_text_type, "json_text_type", allowed=_SQL_TEXT_TYPE_KEYS)
        vector_native_type = _clean_sql_type(self.vector_native_type, "vector_native_type", optional=True, allowed=_SQL_VECTOR_NATIVE_TYPE_KEYS)
        vector_packed_type = _clean_sql_type(self.vector_packed_type, "vector_packed_type", optional=True, allowed=_SQL_VECTOR_PACKED_TYPE_KEYS)
        name_warning_dialect_factory = _clean_name_warning_factory(self.name_warning_dialect_factory)
        name_warning_reserved_words = tuple(dict.fromkeys(str(word).strip().lower() for word in self.name_warning_reserved_words if str(word).strip()))
        name_warning_max_identifier_length = None if self.name_warning_max_identifier_length is None else int(self.name_warning_max_identifier_length)
        if identifier in sqlalchemy_aliases:
            raise ValueError(f"Database dialect {identifier!r} cannot alias itself")
        if vector_packed_type == SQL_TYPE_FLOAT64_BLOB and vector_native_type is None and tags.get("database-dynamic-typing") is not True:
            # A packed vector is bound as bytes whatever the live column is: a Vector column an existing database created
            # as JSON text or TEXT on a statically typed dialect would store it as escaped text and never read back.
            raise ValueError(
                f"Database dialect {identifier!r} cannot use vector_packed_type {SQL_TYPE_FLOAT64_BLOB!r}: packed vectors need "
                "dynamically typed columns (database-dynamic-typing, as on SQLite) so an existing text Vector column keeps "
                f"working; use {SQL_TYPE_JSON_TEXT!r} or leave it unset"
            )
        if varchar_max is not None and varchar_max <= 0:
            raise ValueError("Database dialect varchar_max must be positive")
        if text_min_len is not None and text_min_len <= 0:
            raise ValueError("Database dialect text_min_len must be positive")
        if any(not name for name in static_database_names):
            raise ValueError("Database dialect static_database_names cannot contain empty names")
        if any(not param for param in url_database_omit_params):
            raise ValueError("Database dialect url_database_omit_params cannot contain empty names")
        if scope_style not in {"database", "database_only", "catalog_schema", "service_sid_dsn"}:
            raise ValueError("Database dialect scope_style must be database, database_only, catalog_schema, or service_sid_dsn")
        if memory_pool_class is not None and not memory_pool_class:
            raise ValueError("Database dialect memory_pool_class cannot be empty")
        if any(not message for message in retryable_error_messages):
            raise ValueError("Database dialect retryable_error_messages cannot contain empty messages")
        if identifier_quote_style not in {"double", "backtick", "bracket"}:
            raise ValueError("Database dialect identifier_quote_style must be 'double', 'backtick', or 'bracket'")
        if lifecycle_namespace_source not in {"database", "schema_or_username"}:
            raise ValueError("Database dialect lifecycle_namespace_source must be 'database' or 'schema_or_username'")
        if before_drop_force_mode not in {"bind_name", "quoted_database"}:
            raise ValueError("Database dialect before_drop_force_mode must be 'bind_name' or 'quoted_database'")
        if name_warning_max_identifier_length is not None and name_warning_max_identifier_length <= 0:
            raise ValueError("Database dialect name_warning_max_identifier_length must be positive")
        object.__setattr__(self, "identifier", identifier)
        object.__setattr__(self, "tags", tags)
        object.__setattr__(self, "varchar_max", varchar_max)
        object.__setattr__(self, "text_min_len", text_min_len)
        object.__setattr__(self, "superuser_database", superuser_database)
        object.__setattr__(self, "sqlglot_dialect", sqlglot_dialect)
        object.__setattr__(self, "sqlalchemy_aliases", sqlalchemy_aliases)
        object.__setattr__(self, "current_database_sql", current_database_sql)
        object.__setattr__(self, "list_databases_sql", list_databases_sql)
        object.__setattr__(self, "static_database_names", static_database_names)
        object.__setattr__(self, "ddl_resource_key", ddl_resource_key)
        object.__setattr__(self, "identifier_quote_style", identifier_quote_style)
        object.__setattr__(self, "lifecycle_namespace_source", lifecycle_namespace_source)
        object.__setattr__(self, "lifecycle_namespace_upper", bool(self.lifecycle_namespace_upper))
        object.__setattr__(self, "requires_host", bool(self.requires_host))
        object.__setattr__(self, "url_database_omit_params", url_database_omit_params)
        object.__setattr__(self, "scope_style", scope_style)
        object.__setattr__(self, "supports_memory", bool(self.supports_memory))
        object.__setattr__(self, "file_uri_query", bool(self.file_uri_query))
        object.__setattr__(self, "memory_pool_class", memory_pool_class)
        object.__setattr__(self, "memory_connect_args", memory_connect_args)
        object.__setattr__(self, "retryable_error_messages", retryable_error_messages)
        object.__setattr__(self, "after_create_sql_keys", after_create_sql_keys)
        object.__setattr__(self, "owner_after_create_sql_keys", owner_after_create_sql_keys)
        object.__setattr__(self, "before_drop_force_sql_key", before_drop_force_sql_key)
        object.__setattr__(self, "before_drop_force_mode", before_drop_force_mode)
        object.__setattr__(self, "rename_table_sql", rename_table_sql)
        object.__setattr__(self, "rename_table_bind_params", bool(self.rename_table_bind_params))
        object.__setattr__(self, "add_column_sql", add_column_sql)
        object.__setattr__(self, "column_reflection_sql", column_reflection_sql)
        object.__setattr__(self, "primary_key_reflection_sql", primary_key_reflection_sql)
        object.__setattr__(self, "index_reflection_sql", index_reflection_sql)
        object.__setattr__(self, "empty_foreign_keys_on_reflection_error", bool(self.empty_foreign_keys_on_reflection_error))
        object.__setattr__(self, "varchar_overflow_type", varchar_overflow_type)
        object.__setattr__(self, "text_threshold_type", text_threshold_type)
        object.__setattr__(self, "blob_text_type", blob_text_type)
        object.__setattr__(self, "json_native_type", json_native_type)
        object.__setattr__(self, "json_text_type", json_text_type)
        object.__setattr__(self, "vector_native_type", vector_native_type)
        object.__setattr__(self, "vector_packed_type", vector_packed_type)
        object.__setattr__(self, "name_warning_dialect_factory", name_warning_dialect_factory)
        object.__setattr__(self, "name_warning_reserved_words", name_warning_reserved_words)
        object.__setattr__(self, "name_warning_max_identifier_length", name_warning_max_identifier_length)

    def tag(self, identifier: str, *, default: Any = None) -> Any:
        """Return one construction-only or Registry-derived dialect tag value.

        Args:
            identifier (str): Physical-dialect tag identifier.
            default (Any): Value returned when the tag is absent.

        Returns:
            Any: Tag value stored on this dialect spec, or ``default``.
        """
        return self.tags.get(str(identifier), default)

    def supports(self, identifier: str) -> bool:
        """Return one physical-dialect Boolean tag with strict value typing.

        Args:
            identifier (str): Physical-dialect Boolean tag identifier.

        Returns:
            bool: Tag value when the stored tag is Boolean.

        Raises:
            TypeError: If the stored tag value is not Boolean.
        """
        value = self.tag(identifier, default=False)
        if not isinstance(value, bool):
            raise TypeError(f"Database dialect tag {identifier!r} must be Boolean")
        return value

    def __hash__(self) -> int:
        """Return a stable process hash for SQLAlchemy type-cache identity."""
        return hash(repr(self.to_dict()))

    def sqlglot_name(self) -> str:
        """Return the SQLGlot dialect name for this SQLAlchemy dialect.

        Args:
            None.

        Returns:
            str: Configured SQLGlot dialect, or the SQLAlchemy identifier when omitted.
        """
        return self.sqlglot_dialect or self.identifier

    def requires_host_for_url(self) -> bool:
        """Return whether DBSpec URL materialization must include a host.

        Args:
            None.

        Returns:
            bool: ``True`` when this dialect requires a host and does not use local
            database paths.
        """
        return self.requires_host and not self.supports("database-local-path")

    def url_database(self, data: Mapping[str, Any]) -> Any:
        """Return the database path token for SQLAlchemy URL creation.

        Args:
            data (Mapping[str, Any]): Canonical DBSpec fields containing ``database``
                and ``params``.

        Returns:
            Any: Database path token, or ``None`` when omit-params rules apply.
        """
        params = dict(data.get("params") or {})
        if any(params.get(param) for param in self.url_database_omit_params):
            return None
        return data.get("database")

    def parse_url_database(self, database: Any, params: Mapping[str, Any]) -> tuple[Any, dict[str, Any], bool]:
        """Normalize a parsed SQLAlchemy URL database token.

        Args:
            database (Any): Database token parsed from a SQLAlchemy URL.
            params (Mapping[str, Any]): Parsed URL query parameters.

        Returns:
            tuple[Any, dict[str, Any], bool]: Database token, query params, and
            whether the URL selects this dialect's in-memory mode.
        """
        query = dict(params)
        in_memory = self.supports_memory and database in (None, "", ":memory:")
        if self.file_uri_query and isinstance(database, str) and database.startswith("file:") and str(query.get("uri", "")).lower() == "true":
            from urllib.parse import unquote

            query.pop("uri", None)
            database = unquote(database[len("file:") :])
        return database, query, in_memory

    def normalize_scope(self, values: dict[str, Any], *, parsed: Mapping[str, Any], explicit: set[str]) -> None:
        """Apply this dialect's provider-owned database scope contract.

        Args:
            values (dict[str, Any]): Mutable canonical DBSpec fields being normalized.
            parsed (Mapping[str, Any]): Fields parsed from an explicit URL, when present.
            explicit (set[str]): Top-level DBSpec field names supplied by the caller.

        Returns:
            None: This function mutates ``values`` in place.
        """
        if self.scope_style == "service_sid_dsn":
            params = values["params"]
            service_name = params.pop("service_name", None)
            if service_name is not None and parsed.get("database"):
                raise ValueError(f"{self.identifier} URL cannot combine a database SID path with service_name")
            if service_name is not None:
                if values.get("service") not in (None, service_name):
                    raise ValueError(f"{self.identifier} service conflicts with params.service_name")
                values["service"] = service_name
            if parsed.get("database") and not parsed.get("in_memory"):
                values["sid"] = parsed["database"]
                if "service" not in explicit:
                    values["service"] = None
                if "dsn" not in explicit:
                    values["dsn"] = None
            elif values.get("database") is not None:
                values["service"] = values["database"]
                values["sid"] = None
                values["dsn"] = None
            elif "sid" in explicit and values.get("sid"):
                values["service"] = None
                values["dsn"] = None
            elif "dsn" in explicit and values.get("dsn"):
                values["service"] = None
                values["sid"] = None
            values["database"] = None
            values["schema"] = values.get("schema") or values.get("username")
            return
        if self.scope_style == "catalog_schema":
            packed = values.get("database")
            if packed:
                parts = str(packed).split("/", 1)
                if "catalog" in explicit and values.get("catalog") not in (None, parts[0]):
                    raise ValueError(f"{self.identifier} database shorthand conflicts with catalog")
                values["catalog"] = parts[0]
                if len(parts) == 2:
                    if "schema" in explicit and values.get("schema") not in (None, parts[1]):
                        raise ValueError(f"{self.identifier} database shorthand conflicts with schema")
                    values["schema"] = parts[1]
            values["database"] = None
            return
        if self.scope_style == "database_only":
            schema = values.get("schema")
            if schema is not None and values.get("database") not in (None, schema):
                raise ValueError(f"{self.identifier} database and schema conflict")
            values["database"] = values.get("database") or schema
            values["schema"] = None

    def validate_scope(self, spec: Any) -> None:
        """Validate one DBSpec against this dialect's scope contract.

        Args:
            spec (Any): Resolved ``DBSpec`` or compatible scope-bearing object.

        Returns:
            None: This function does not return a value.

        Raises:
            ValueError: If the scope fields violate this dialect's contract.
        """
        if self.scope_style == "service_sid_dsn":
            routes = [value for value in (spec.service, spec.sid, spec.dsn) if value]
            if len(routes) != 1:
                raise ValueError(f"{self.identifier} config requires exactly one of service, sid, or dsn")
            if spec.database is not None:
                raise ValueError(f"{self.identifier} canonical config uses service, sid, or dsn instead of database")
        if self.scope_style == "database_only" and spec.schema is not None:
            raise ValueError(f"{self.identifier} canonical config uses database instead of schema")

    def normalize_memory_pool(self, pool: Mapping[str, Any], *, explicit: set[str]) -> dict[str, Any]:
        """Apply dialect-owned in-memory pool defaults without overriding user input.

        Args:
            pool (Mapping[str, Any]): Candidate SQLAlchemy pool configuration.
            explicit (set[str]): Pool keys explicitly supplied by the caller.

        Returns:
            dict[str, Any]: Pool configuration with dialect-owned in-memory defaults applied.
        """
        out = dict(pool)
        if self.memory_pool_class is not None and "pool_class" not in explicit:
            out["pool_class"] = self.memory_pool_class
        connect_args = dict(out.get("connect_args") or {})
        for key, value in _thaw_data(self.memory_connect_args).items():
            connect_args.setdefault(key, value)
        if connect_args:
            out["connect_args"] = connect_args
        return out

    def url_scope(self, spec: Any, params: Mapping[str, Any]) -> tuple[Any, Any, dict[str, Any]]:
        """Return database, host, and query params for SQLAlchemy URL creation.

        Args:
            spec (Any): Resolved ``DBSpec`` or compatible scope-bearing object.
            params (Mapping[str, Any]): Canonical URL query parameters.

        Returns:
            tuple[Any, Any, dict[str, Any]]: Database token, host token, and query
            parameters for ``URL.create``.
        """
        query = dict(params)
        database = self.url_database({"database": spec.database, "params": query})
        host = spec.host
        if self.scope_style == "catalog_schema":
            database = "/".join(item for item in (spec.catalog, spec.schema) if item)
        elif self.scope_style == "service_sid_dsn":
            if spec.service:
                query["service_name"] = spec.service
                database = None
            elif spec.sid:
                database = spec.sid
            elif spec.dsn:
                host = spec.dsn
                database = None
        return database, host, query

    def normalize_superuser(self, data: dict[str, Any], superuser: Mapping[str, Any], *, database: Any) -> None:
        """Apply this dialect's maintenance-connection scope contract.

        Args:
            data (dict[str, Any]): Mutable superuser connection fields.
            superuser (Mapping[str, Any]): Explicit superuser overrides from config.
            database (Any): Resolved user database token for ``"{database}"`` templates.

        Returns:
            None: This function mutates ``data`` in place.
        """
        if self.scope_style == "service_sid_dsn":
            if superuser.get("database") is not None:
                data["service"] = superuser["database"]
                data["database"] = None
            return
        if "database" not in superuser:
            data["database"] = database if self.superuser_database == "{database}" else self.superuser_database

    def is_retryable_error(self, exc: Exception) -> bool:
        """Return whether an execution error should trigger transient retry.

        Args:
            exc (Exception): Execution error raised by the database driver or SQLAlchemy.

        Returns:
            bool: ``True`` when the lowered message matches a configured retry fragment.
        """
        message = str(exc).lower()
        return bool(message and any(item in message for item in self.retryable_error_messages))

    def rename_table_statement(self, old_name: str, new_name: str, quote: Callable[[str], str]) -> tuple[str, Optional[Dict[str, Any]]]:
        """Return SQL and optional params for table rename DDL.

        Args:
            old_name (str): Current table name.
            new_name (str): Target table name.
            quote (Callable[[str], str]): Identifier quoter supplied by the live dialect.

        Returns:
            tuple[str, Optional[Dict[str, Any]]]: Rendered SQL and optional bind params.
        """
        if self.rename_table_bind_params:
            return str(self.rename_table_sql), {"old_name": str(old_name), "new_name": str(new_name)}
        return str(self.rename_table_sql).format(old_name=quote(old_name), new_name=quote(new_name)), None

    def add_column_statement(self, table_name: str, column_def: str) -> str:
        """Render this dialect's DDL for adding one column.

        Args:
            table_name: Dialect-quoted table name.
            column_def: Column definition compiled by the live SQLAlchemy dialect.

        Returns:
            str: Executable ``ALTER TABLE`` statement.
        """
        return str(self.add_column_sql).format(table=table_name, column=column_def)

    def quote_identifier(self, name: str) -> str:
        """Return one SQL identifier quoted with this dialect's lifecycle style.

        Args:
            name (str): Raw SQL identifier.

        Returns:
            str: Identifier quoted for lifecycle DDL templates.
        """
        value = str(name)
        if self.identifier_quote_style == "bracket":
            return f"[{value.replace(']', ']]')}]"
        if self.identifier_quote_style == "backtick":
            return f"`{value.replace('`', '``')}`"
        return f'"{value.replace(chr(34), chr(34) * 2)}"'

    def lifecycle_namespace(self, spec: Any) -> Optional[str]:
        """Return the DBSpec namespace that create/drop helpers manage.

        Args:
            spec (Any): Resolved ``DBSpec`` or compatible scope-bearing object.

        Returns:
            Optional[str]: Database or schema/username namespace token, or ``None``
            when lifecycle helpers should no-op.
        """
        if self.lifecycle_namespace_source == "schema_or_username":
            value = getattr(spec, "schema", None) or getattr(spec, "username", None)
        else:
            value = getattr(spec, "database", None)
        return None if value is None else str(value)

    def create_database_statement(self, spec: Any, namespace: str) -> Optional[str]:
        """Return rendered create-namespace SQL for this dialect, when registered.

        Args:
            spec (Any): Resolved ``DBSpec`` used to render owner-specific templates.
            namespace (str): Target database or schema namespace.

        Returns:
            Optional[str]: Rendered create SQL, or ``None`` when this dialect has no
            registered create template.
        """
        template = DB_DDL_CREATE.get(self._ddl_key())
        if not template:
            return None
        return template.text.format(name=self._lifecycle_name(namespace), password=self._ddl_password(spec, namespace))

    def drop_database_statement(self, namespace: str) -> Optional[str]:
        """Return rendered drop-namespace SQL for this dialect, when registered.

        Args:
            namespace (str): Target database or schema namespace.

        Returns:
            Optional[str]: Rendered drop SQL, or ``None`` when this dialect has no
            registered drop template.
        """
        template = DB_DDL_DROP.get(self._ddl_key())
        return None if not template else template.text.format(name=self._lifecycle_name(namespace))

    def database_exists_sql(self) -> Optional[str]:
        """Return existence-check SQL for this dialect's create/drop namespace, when registered.

        Args:
            None.

        Returns:
            Optional[str]: Existence-check SQL template, or ``None`` when unsupported.
        """
        template = DB_DDL_EXISTS.get(self._ddl_key())
        return None if not template else template.text

    def after_create_statements(self, spec: Any, namespace: str) -> list[str]:
        """Return rendered post-create SQL statements for this dialect.

        Args:
            spec (Any): Resolved ``DBSpec`` used to render owner-specific templates.
            namespace (str): Target database or schema namespace.

        Returns:
            list[str]: Rendered post-create SQL statements in execution order.
        """
        out = [sql for key in self.after_create_sql_keys for sql in self._after_create_group(key, namespace)]
        superuser = getattr(spec, "superuser", None) or {}
        username = getattr(spec, "username", None)
        if username and superuser.get("username", username) != username:
            out.extend(sql for key in self.owner_after_create_sql_keys for sql in self._after_create_group(key, namespace, username=str(username)))
        return out

    def before_drop_force_statements(self, namespace: str, *, force: bool = False) -> list[tuple[str, Optional[Dict[str, Any]]]]:
        """Return force-drop cleanup SQL and optional params for this dialect.

        Args:
            namespace (str): Target database or schema namespace.
            force (bool): Whether a forced drop requested active-session cleanup.

        Returns:
            list[tuple[str, Optional[Dict[str, Any]]]]: Cleanup SQL statements and
            optional bind-parameter mappings.
        """
        if not force or not self.before_drop_force_sql_key:
            return []
        template = DB_DDL_BEFORE_DROP_FORCE.get(self.before_drop_force_sql_key)
        if not template:
            return []
        if self.before_drop_force_mode == "quoted_database":
            return [(template.render(database=self._lifecycle_name(namespace)), None)]
        return [(template.text, {"name": namespace})]

    def column_reflection_statement(self, table: str, schema: Optional[str], quote: Callable[[str], str], quote_schema: Callable[[str], str]) -> Optional[str]:
        """Return fallback SQL for reflected column metadata.

        Args:
            table (str): Table name being reflected.
            schema (Optional[str]): Optional schema or owner name.
            quote (Callable[[str], str]): Table identifier quoter from the live dialect.
            quote_schema (Callable[[str], str]): Schema identifier quoter from the live dialect.

        Returns:
            Optional[str]: Rendered reflection SQL, or ``None`` when unsupported.
        """
        return _render_reflection_sql(self.column_reflection_sql, table, schema, quote, quote_schema)

    def primary_key_reflection_statement(
        self, table: str, schema: Optional[str], quote: Callable[[str], str], quote_schema: Callable[[str], str]
    ) -> Optional[str]:
        """Return fallback SQL for reflected primary-key metadata.

        Args:
            table (str): Table name being reflected.
            schema (Optional[str]): Optional schema or owner name.
            quote (Callable[[str], str]): Table identifier quoter from the live dialect.
            quote_schema (Callable[[str], str]): Schema identifier quoter from the live dialect.

        Returns:
            Optional[str]: Rendered primary-key reflection SQL, or ``None`` when unsupported.
        """
        return _render_reflection_sql(self.primary_key_reflection_sql, table, schema, quote, quote_schema)

    def index_reflection_statement(self, table: str, schema: Optional[str], quote: Callable[[str], str], quote_schema: Callable[[str], str]) -> Optional[str]:
        """Return catalog SQL for a table's indexes, for a dialect whose SQLAlchemy inspector reports none.

        Args:
            table (str): Table name being reflected.
            schema (Optional[str]): Optional schema or owner name.
            quote (Callable[[str], str]): Table identifier quoter from the live dialect.
            quote_schema (Callable[[str], str]): Schema identifier quoter from the live dialect.

        Returns:
            Optional[str]: Rendered index reflection SQL, or ``None`` when the dialect's inspector reports indexes.
        """
        return _render_reflection_sql(self.index_reflection_sql, table, schema, quote, quote_schema)

    def name_warning_rules(self, *, dialect: Any = None, extra_reserved: Iterable[str] = ()) -> tuple[int | None, set[str]]:
        """Return SQL identifier warning rules declared by this dialect.

        Args:
            dialect (Any): Optional live SQLAlchemy dialect supplying runtime limits.
            extra_reserved (Iterable[str]): Additional reserved words to include.

        Returns:
            tuple[int | None, set[str]]: Maximum identifier length, if known, and
            reserved-word set for schema warnings.
        """
        max_length, reserved = _live_name_warning_rules(dialect)
        factory_max, factory_reserved = _factory_name_warning_rules(self.name_warning_dialect_factory)
        if max_length is None:
            max_length = factory_max
        if self.name_warning_max_identifier_length is not None:
            max_length = self.name_warning_max_identifier_length
        reserved.update(factory_reserved)
        reserved.update(self.name_warning_reserved_words)
        reserved.update(str(word).strip().lower() for word in extra_reserved if str(word).strip())
        return max_length, reserved

    def to_dict(self) -> Dict[str, object]:
        """Return JSON-compatible database dialect metadata.

        Args:
            None.

        Returns:
            Dict[str, object]: Serializable dialect metadata and tag values.
        """
        return {
            "identifier": self.identifier,
            "tags": _thaw_data(self.tags),
            "varchar_max": self.varchar_max,
            "text_min_len": self.text_min_len,
            "superuser_database": self.superuser_database,
            "sqlglot_dialect": self.sqlglot_name(),
            "sqlalchemy_aliases": list(self.sqlalchemy_aliases),
            "sqlglot_preferred": self.sqlglot_preferred,
            "current_database_sql": self.current_database_sql,
            "list_databases_sql": self.list_databases_sql,
            "static_database_names": list(self.static_database_names),
            "ddl_resource_key": self.ddl_resource_key or self.identifier,
            "identifier_quote_style": self.identifier_quote_style,
            "lifecycle_namespace_source": self.lifecycle_namespace_source,
            "lifecycle_namespace_upper": self.lifecycle_namespace_upper,
            "requires_host": self.requires_host,
            "url_database_omit_params": list(self.url_database_omit_params),
            "scope_style": self.scope_style,
            "supports_memory": self.supports_memory,
            "file_uri_query": self.file_uri_query,
            "memory_pool_class": self.memory_pool_class,
            "memory_connect_args": _thaw_data(self.memory_connect_args),
            "retryable_error_messages": list(self.retryable_error_messages),
            "after_create_sql_keys": list(self.after_create_sql_keys),
            "owner_after_create_sql_keys": list(self.owner_after_create_sql_keys),
            "before_drop_force_sql_key": self.before_drop_force_sql_key,
            "before_drop_force_mode": self.before_drop_force_mode,
            "rename_table_sql": self.rename_table_sql,
            "rename_table_bind_params": self.rename_table_bind_params,
            "add_column_sql": self.add_column_sql,
            "column_reflection_sql": self.column_reflection_sql,
            "primary_key_reflection_sql": self.primary_key_reflection_sql,
            "index_reflection_sql": self.index_reflection_sql,
            "empty_foreign_keys_on_reflection_error": self.empty_foreign_keys_on_reflection_error,
            "varchar_overflow_type": self.varchar_overflow_type,
            "text_threshold_type": self.text_threshold_type,
            "blob_text_type": self.blob_text_type,
            "json_native_type": self.json_native_type,
            "json_text_type": self.json_text_type,
            "vector_native_type": self.vector_native_type,
            "vector_packed_type": self.vector_packed_type,
            "name_warning_dialect_factory": list(self.name_warning_dialect_factory) if self.name_warning_dialect_factory else None,
            "name_warning_reserved_words": list(self.name_warning_reserved_words),
            "name_warning_max_identifier_length": self.name_warning_max_identifier_length,
        }

    def _ddl_key(self) -> str:
        return self.ddl_resource_key or self.identifier

    def _lifecycle_name(self, namespace: str) -> str:
        value = str(namespace).upper() if self.lifecycle_namespace_upper else str(namespace)
        return self.quote_identifier(value)

    def _target_password(self, spec: Any, namespace: str) -> str:
        superuser = getattr(spec, "superuser", None) or {}
        return getattr(spec, "password", None) or superuser.get("password") or f"{namespace}_pw"

    def _ddl_password(self, spec: Any, namespace: str) -> str:
        return str(self._target_password(spec, namespace)).replace('"', '""')

    def _after_create_group(self, key: str, namespace: str, *, username: Optional[str] = None) -> list[str]:
        templates = DB_DDL_AFTER_CREATE.get(key, ())
        values = {"name": self._lifecycle_name(namespace)}
        if username is not None:
            values.update(database=self.quote_identifier(namespace), username=self.quote_identifier(username))
        return [template.render(**values) for template in templates]


def _clean_sql(value: Optional[str], label: str) -> Optional[str]:
    if value is None:
        return None
    sql = str(value).strip()
    if not sql:
        raise ValueError(f"Database dialect {label} cannot be empty")
    return sql


def _freeze_data(value: Any, label: str) -> Any:
    """Return immutable JSON-shaped dialect metadata."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_data(item, label) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_data(item, label) for item in value)
    raise TypeError(f"Database dialect {label} must contain only JSON-compatible values")


def _thaw_data(value: Any) -> Any:
    """Return a mutable JSON-compatible copy of frozen dialect metadata."""
    if isinstance(value, Mapping):
        return {str(key): _thaw_data(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_data(item) for item in value]
    return value


def _clean_sql_template(value: str, label: str, *, fields: tuple[str, ...]) -> str:
    sql = str(value).strip()
    if not sql:
        raise ValueError(f"Database dialect {label} cannot be empty")
    try:
        parts = tuple(Formatter().parse(sql))
    except ValueError as exc:
        raise ValueError(f"Database dialect {label} is not a valid format template: {exc}") from exc
    found = {field_name for _text, field_name, _format, _conversion in parts if field_name is not None}
    expected = set(fields)
    if found != expected or any(format_spec or conversion for _text, _field, format_spec, conversion in parts):
        raise ValueError(f"Database dialect {label} must use only plain placeholders {list(fields)}")
    return sql


def _clean_sql_type(value: Optional[str], label: str, *, optional: bool = False, allowed: set[str] = _SQL_TYPE_KEYS) -> Optional[str]:
    if value is None:
        if optional:
            return None
        raise ValueError(f"Database dialect {label} is required")
    key = str(value).strip().lower().replace("-", "_")
    if key not in allowed:
        raise ValueError(f"Database dialect {label} {key!r} is unsupported; expected one of {sorted(allowed)}")
    return key


def _clean_name_warning_factory(value: Optional[Tuple[str, str]]) -> Optional[Tuple[str, str]]:
    if value is None:
        return None
    if len(value) != 2:
        raise ValueError("Database dialect name_warning_dialect_factory must be a (module, attr) pair")
    module_name, attr = (str(item).strip() for item in value)
    if not module_name or not attr:
        raise ValueError("Database dialect name_warning_dialect_factory entries must be non-empty")
    return module_name, attr


def _live_name_warning_rules(dialect: Any) -> tuple[int | None, set[str]]:
    if dialect is None:
        return None, set()
    preparer = getattr(dialect, "identifier_preparer", None)
    return getattr(dialect, "max_identifier_length", None), set(getattr(preparer, "reserved_words", set()) or set())


def _factory_name_warning_rules(factory: Optional[Tuple[str, str]]) -> tuple[int | None, set[str]]:
    if factory is None:
        return None, set()
    module_name, attr = factory
    try:
        dialect_factory = getattr(import_module(module_name), attr)
        dialect = dialect_factory()
    except Exception:
        return None, set()
    return _live_name_warning_rules(dialect)


def _render_reflection_sql(
    template: Optional[str], table: str, schema: Optional[str], quote: Callable[[str], str], quote_schema: Callable[[str], str]
) -> Optional[str]:
    if not template:
        return None

    def literal(value: str) -> str:
        return "'" + value.replace("'", "''") + "'"

    name = f"{schema}.{table}" if schema else str(table)
    table_name = f"{quote_schema(str(schema))}.{quote(str(table))}" if schema else quote(str(table))
    return template.format(
        table_name=table_name,
        table_name_literal=literal(name),
        table_literal=literal(str(table)),
        schema_literal=literal(str(schema)) if schema else "NULL",
    )
