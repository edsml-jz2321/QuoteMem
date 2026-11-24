"""SQLAlchemy type decorators used by SQL-family backends."""

from __future__ import annotations

__all__ = [
    "DatabaseBigIntegerType",
    "DatabaseBlobType",
    "DatabaseBooleanType",
    "DatabaseIntegerType",
    "DatabaseJsonType",
    "DatabaseTextType",
    "DatabaseVectorType",
]

import json
import sys
from array import array
from codecs import BOM_UTF16_BE, BOM_UTF16_LE, BOM_UTF32_BE, BOM_UTF32_LE
from contextlib import contextmanager
from contextvars import ContextVar
from functools import cached_property
from sqlite3 import Connection as SQLiteConnection
from sqlite3 import Cursor as SQLiteCursor
from threading import Lock

import sqlalchemy as sa
import sqlalchemy.dialects.mysql as sa_mysql
import sqlalchemy.dialects.oracle as sa_oracle
import sqlalchemy.dialects.postgresql as sa_pg
import sqlalchemy.types as sa_types
from sqlalchemy.dialects.sqlite.base import SQLiteExecutionContext
from sqlalchemy.dialects.sqlite.pysqlite import SQLiteDialect_pysqlite

from heavenbase.utils.typing import Any
from heavenbase.utils.serialize import dumps_b64, dumps_json, loads_b64, loads_json

from .dialect.spec import (
    SQL_TYPE_CLOB,
    SQL_TYPE_FLOAT64_BLOB,
    SQL_TYPE_FLOAT_ARRAY,
    SQL_TYPE_JSON,
    SQL_TYPE_MYSQL_TEXT,
    SQL_TYPE_POSTGRESQL_JSONB,
    SQL_TYPE_STRING,
    SQL_TYPE_TEXT,
    DatabaseDialectSpec,
)
from .dialect.registry import database_dialects


def _text_type(kind: str, length: int | None = None, *, max_chars: int = 65535) -> Any:
    if kind == SQL_TYPE_CLOB:
        return sa_oracle.CLOB()
    if kind == SQL_TYPE_MYSQL_TEXT:
        return _mysql_text_type(length)
    if kind == SQL_TYPE_TEXT:
        return sa_types.Text()
    return sa_types.String(length if kind != SQL_TYPE_STRING else length or max_chars)


def _mysql_text_type(length: int | None) -> Any:
    """Choose a MySQL text family that can hold `length` utf8mb4 characters."""
    if length is None:
        return sa_mysql.TEXT()
    max_bytes = int(length) * 4
    if max_bytes <= 255:
        return sa_mysql.TINYTEXT()
    if max_bytes <= 65_535:
        return sa_mysql.TEXT()
    if max_bytes <= 16_777_215:
        return sa_mysql.MEDIUMTEXT()
    return sa_mysql.LONGTEXT()


def _native_json_type(kind: str | None) -> Any | None:
    if kind == SQL_TYPE_POSTGRESQL_JSONB:
        return sa_pg.JSONB()
    if kind == SQL_TYPE_JSON:
        return sa_types.JSON()
    return None


def _native_vector_type(kind: str | None) -> Any | None:
    if kind == SQL_TYPE_FLOAT_ARRAY:
        return sa.ARRAY(sa_types.Float())
    return None


def _pack_vector(values: list[float]) -> bytes:
    """Pack floats as little-endian IEEE-754 float64 so every double round-trips bit-exactly."""
    packed = array("d", values)
    if sys.byteorder != "little":
        packed.byteswap()
    return packed.tobytes()


def _unpack_vector(payload: Any) -> list[float]:
    """Restore a little-endian float64 payload to Python floats.

    Raises:
        ValueError: If the payload length is not a whole number of float64 values.
    """
    raw = bytes(payload)
    if len(raw) % 8:
        raise ValueError(f"Packed vector payload of {len(raw)} bytes is not a whole number of float64 values")
    values = array("d")
    values.frombytes(raw)
    if sys.byteorder != "little":
        values.byteswap()
    return values.tolist()


class _PackedVectorBinary(sa_types.LargeBinary):
    """``LargeBinary`` storage for packed vectors that hands every stored value through unchanged.

    A packed column also holds JSON text written before the opt-in. Before SQLAlchemy
    2.0.11 the pysqlite dialect did not declare native bytes, so ``LargeBinary``'s
    result processor ran ``bytes(value)`` on a ``str`` and raised ``TypeError``.
    ``DatabaseVectorType.process_result_value`` decodes bytes, buffers and text itself.
    """

    cache_ok = True

    def result_processor(self, dialect, coltype):
        """Return no result processor, so text and bytes both reach the vector decoder.

        Args:
            dialect: Live SQLAlchemy dialect reading the column.
            coltype: DBAPI column type code.

        Returns:
            None: Stored values are passed through unchanged.
        """
        return None


def _vector_text(values: list[float]) -> str:
    """Encode floats as compact JSON text.

    A vector holds only floats, so the tagged-value walk of ``dumps_json`` has
    nothing to adapt and is skipped; the text is the same either way.
    """
    return dumps_json(values, compact=True, adapt=False)


def _text_vector(text: str | bytes) -> list[float]:
    """Decode JSON vector text without the tagged-value restore walk, which a list of numbers never needs."""
    return [float(item) for item in json.loads(text)]


def _sqlite_packed_projection_text(value: str | bytes) -> str:
    """Decode the packed projection envelope before inspecting its ASCII marker."""
    if isinstance(value, str):
        return value
    raw = bytes(value)
    if raw.startswith((BOM_UTF32_BE, BOM_UTF32_LE)):
        return raw.decode("utf-32")
    if raw.startswith((BOM_UTF16_BE, BOM_UTF16_LE)):
        return raw.decode("utf-16")
    # Without a BOM, the first character is ASCII (T or a hex digit).
    if raw[:3] == b"\0\0\0":
        return raw.decode("utf-32-be")
    if raw[1:4] == b"\0\0\0":
        return raw.decode("utf-32-le")
    if raw[:1] == b"\0":
        return raw.decode("utf-16-be")
    if raw[1:2] == b"\0":
        return raw.decode("utf-16-le")
    return raw.decode("utf-8-sig")


def _sqlite_vector_blob_error(size: int) -> ValueError:
    return ValueError(f"Vector column returned a SQLite BLOB of {size} bytes. Reading packed vectors requires vector_packed_type: float64_blob")


_SQLITE_VECTOR_PROJECTION = "_heavenbase_vector_projection"


def _encode_sqlite_vector_projection(value, packed):
    """Mark a single SQLite value before a text factory or converter can change its Python type."""
    if value is None:
        return None
    if isinstance(value, str):
        return "T" + value if packed else value
    if isinstance(value, bytes):
        return value.hex().upper() if packed else len(value)
    return -1


def _install_sqlite_vector_projection(connection, info):
    connection.create_function(_SQLITE_VECTOR_PROJECTION, 2, _encode_sqlite_vector_projection)
    info[_SQLITE_VECTOR_PROJECTION] = True


@sa.event.listens_for(sa.Engine, "connect")
def _register_sqlite_vector_projection(connection, record):
    if isinstance(connection, SQLiteConnection):
        _install_sqlite_vector_projection(connection, record.info)


@sa.event.listens_for(sa.Engine, "before_cursor_execute")
def _ensure_sqlite_vector_projection(conn, cursor, statement, parameters, context, executemany):
    # Engines can already have open connections when this module is imported.
    if context.dialect.name == "sqlite" and _SQLITE_VECTOR_PROJECTION in statement and _SQLITE_VECTOR_PROJECTION not in conn.info:
        _install_sqlite_vector_projection(conn.connection.dbapi_connection, conn.info)


class _SQLiteVectorResult(sa_types.TypeDecorator):
    """Decode a projection that preserves SQLite's TEXT/BLOB distinction."""

    impl = sa_types.Text
    cache_ok = True

    def __init__(self, packed: bool) -> None:
        self.packed = packed
        super().__init__()

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        if isinstance(value, int):
            if value >= 0:
                raise _sqlite_vector_blob_error(value)
            raise ValueError("SQLite vector storage must be TEXT, BLOB, or NULL")
        if not self.packed:
            return _text_vector(value if isinstance(value, str) else bytes(value))
        text = _sqlite_packed_projection_text(value)
        if text.startswith("T"):
            return _text_vector(text[1:])
        return _unpack_vector(bytes.fromhex(text))


def _sqlite_vector_projection(column, *, packed: bool):
    # Pass the expression only once. The fixed storage flag consumes no bind slot.
    projection = getattr(sa.func, _SQLITE_VECTOR_PROJECTION)
    return projection(column, sa.true() if packed else sa.false(), type_=_SQLiteVectorResult(packed))


class _SQLiteVectorText(sa_types.TypeDecorator):
    """Keep the configured SQLite text type for DDL and wrap typed reads."""

    impl = sa_types.Text
    cache_ok = True

    def __init__(self, storage) -> None:
        self.storage = storage
        super().__init__()

    def load_dialect_impl(self, dialect):
        return self.storage

    def column_expression(self, column):
        return _sqlite_vector_projection(column, packed=False)


class _SQLitePackedVectorBinary(_PackedVectorBinary):
    cache_ok = True

    def column_expression(self, column):
        return _sqlite_vector_projection(column, packed=True)


def _bound_dialect(dialect: Any, spec: DatabaseDialectSpec | None) -> DatabaseDialectSpec:
    """Return explicit type metadata, falling back only for standalone use."""
    if spec is None:
        return database_dialects.get(dialect.name)
    name = str(dialect.name).strip().lower()
    if name != "default" and name not in {spec.identifier, *spec.sqlalchemy_aliases}:
        raise ValueError(f"Bound database dialect {spec.identifier!r} cannot compile SQLAlchemy dialect {name!r}")
    return spec


class DatabaseTextType(sa_types.TypeDecorator):
    """Portable text type."""

    impl = sa_types.String
    cache_ok = True

    def __init__(
        self,
        *args: Any,
        dialect_spec: DatabaseDialectSpec | None = None,
        max_chars: int = 65535,
        **kwargs: Any,
    ) -> None:
        """Create a portable text type decorator.

        Args:
            *args (Any): Positional arguments forwarded to ``TypeDecorator``.
            dialect_spec (DatabaseDialectSpec | None): Bound dialect metadata used for
                overflow and text-threshold selection.
            max_chars (int): Maximum string length used by text fallback types.
            **kwargs (Any): Keyword arguments forwarded to ``TypeDecorator``.

        Returns:
            None: This function does not return a value.
        """
        self.dialect_spec = dialect_spec
        self.max_chars = int(max_chars)
        super().__init__(*args, **kwargs)

    def load_dialect_impl(self, dialect):
        """Return the concrete SQLAlchemy text type for one SQL dialect.

        Args:
            dialect: Live SQLAlchemy dialect compiling this type.

        Returns:
            Any: Concrete SQLAlchemy text or string implementation.
        """
        spec = _bound_dialect(dialect, self.dialect_spec)
        varchar_max = spec.varchar_max
        if varchar_max is not None and self.length and self.length > varchar_max:
            return dialect.type_descriptor(_text_type(spec.varchar_overflow_type, self.length, max_chars=self.max_chars))
        text_min_len = spec.text_min_len
        if text_min_len is not None and self.length and self.length >= text_min_len:
            return dialect.type_descriptor(_text_type(spec.text_threshold_type, self.length, max_chars=self.max_chars))
        return dialect.type_descriptor(sa_types.String(self.length))

    def process_bind_param(self, value, dialect):
        """Convert Python values to database text parameters.

        Args:
            value: Python value bound to the column.
            dialect: Live SQLAlchemy dialect compiling this type.

        Returns:
            str | None: Database text parameter, or ``None`` for SQL NULL.
        """
        return None if value is None else str(value)

    def process_result_value(self, value, dialect):
        """Convert database text values back to Python strings.

        Args:
            value: Database value read from the column.
            dialect: Live SQLAlchemy dialect compiling this type.

        Returns:
            str | None: Python string, or ``None`` for SQL NULL.
        """
        return None if value is None else str(value)


class DatabaseIntegerType(sa_types.TypeDecorator):
    """Portable integer type."""

    impl = sa_types.Integer
    cache_ok = True

    def process_bind_param(self, value, dialect):
        """Convert Python values to database integer parameters.

        Args:
            value: Python value bound to the column.
            dialect: Live SQLAlchemy dialect compiling this type.

        Returns:
            int | None: Database integer parameter, or ``None`` for SQL NULL.
        """
        return None if value is None else int(value)

    def process_result_value(self, value, dialect):
        """Convert database integer values back to Python integers.

        Args:
            value: Database value read from the column.
            dialect: Live SQLAlchemy dialect compiling this type.

        Returns:
            int | None: Python integer, or ``None`` for SQL NULL.
        """
        return None if value is None else int(value)


class DatabaseBigIntegerType(DatabaseIntegerType):
    """Portable 64-bit integer type."""

    impl = sa_types.BigInteger
    cache_ok = True


class DatabaseBooleanType(sa_types.TypeDecorator):
    """Portable boolean type."""

    impl = sa_types.Boolean
    cache_ok = True

    def process_bind_param(self, value, dialect):
        """Convert Python values to database boolean parameters.

        Args:
            value: Python value bound to the column.
            dialect: Live SQLAlchemy dialect compiling this type.

        Returns:
            bool | None: Database boolean parameter, or ``None`` for SQL NULL.
        """
        return None if value is None else bool(value)

    def process_result_value(self, value, dialect):
        """Convert database boolean values back to Python booleans.

        Args:
            value: Database value read from the column.
            dialect: Live SQLAlchemy dialect compiling this type.

        Returns:
            bool | None: Python boolean, or ``None`` for SQL NULL.
        """
        return None if value is None else bool(value)


class DatabaseBlobType(sa_types.TypeDecorator):
    """Portable Artifact storage with native BLOB or base64 TEXT fallback."""

    impl = sa_types.LargeBinary
    cache_ok = True

    def __init__(
        self,
        *args: Any,
        dialect_spec: DatabaseDialectSpec | None = None,
        max_chars: int = 65535,
        **kwargs: Any,
    ) -> None:
        """Create a portable artifact blob type decorator.

        Args:
            *args (Any): Positional arguments forwarded to ``TypeDecorator``.
            dialect_spec (DatabaseDialectSpec | None): Bound dialect metadata used to
                choose native blob storage or base64 text fallback.
            max_chars (int): Maximum string length used by text fallback storage.
            **kwargs (Any): Keyword arguments forwarded to ``TypeDecorator``.

        Returns:
            None: This function does not return a value.
        """
        self.dialect_spec = dialect_spec
        self.max_chars = int(max_chars)
        super().__init__(*args, **kwargs)

    def _uses_native_blob(self, dialect) -> bool:
        return _bound_dialect(dialect, self.dialect_spec).supports("database-blob-native")

    def load_dialect_impl(self, dialect):
        """Return the concrete blob or text fallback type for one dialect.

        Args:
            dialect: Live SQLAlchemy dialect compiling this type.

        Returns:
            Any: Native ``LargeBinary`` implementation or text fallback type.
        """
        if self._uses_native_blob(dialect):
            return dialect.type_descriptor(sa_types.LargeBinary())
        return dialect.type_descriptor(_text_type(_bound_dialect(dialect, self.dialect_spec).blob_text_type, max_chars=self.max_chars))

    def _to_bind_value(self, value: Any) -> bytes:
        if value is None:
            return None
        if isinstance(value, str):
            return loads_b64(value)
        if isinstance(value, memoryview):
            return bytes(value)
        if isinstance(value, bytearray):
            return bytes(value)
        if isinstance(value, bytes):
            return value
        raise TypeError(f"Artifact expects bytes, got {type(value).__name__}")

    def process_bind_param(self, value, dialect):
        """Convert artifact bytes to native blob or base64 text storage.

        Args:
            value: Artifact bytes, base64 text, or compatible buffer value.
            dialect: Live SQLAlchemy dialect compiling this type.

        Returns:
            bytes | str | None: Native blob value, base64 text fallback, or ``None``.

        Raises:
            TypeError: If ``value`` is not bytes-compatible artifact data.
        """
        if value is None:
            return None
        data = self._to_bind_value(value)
        return data if self._uses_native_blob(dialect) else dumps_b64(data)

    def process_result_value(self, value, dialect):
        """Convert database blob or base64 text values back to artifact data.

        Args:
            value: Database blob or base64 text value.
            dialect: Live SQLAlchemy dialect compiling this type.

        Returns:
            bytes | str | None: Artifact bytes for native storage, base64 text for
            fallback storage, or ``None`` for SQL NULL.
        """
        if value is None:
            return None
        if self._uses_native_blob(dialect):
            return bytes(value) if not isinstance(value, bytes) else value
        return str(value)


class DatabaseJsonType(sa_types.TypeDecorator):
    """Portable JSON type with string fallback."""

    impl = sa_types.String
    cache_ok = True

    def __init__(
        self,
        *args: Any,
        dialect_spec: DatabaseDialectSpec | None = None,
        max_chars: int = 65535,
        **kwargs: Any,
    ) -> None:
        """Create a portable JSON type decorator.

        Args:
            *args (Any): Positional arguments forwarded to ``TypeDecorator``.
            dialect_spec (DatabaseDialectSpec | None): Bound dialect metadata used to
                choose native JSON storage or string fallback.
            max_chars (int): Maximum string length used by JSON text fallback.
            **kwargs (Any): Keyword arguments forwarded to ``TypeDecorator``.

        Returns:
            None: This function does not return a value.
        """
        self.dialect_spec = dialect_spec
        self.max_chars = int(max_chars)
        super().__init__(*args, **kwargs)

    def _native_impl(self, dialect):
        return _native_json_type(_bound_dialect(dialect, self.dialect_spec).json_native_type)

    def _string_impl(self, dialect):
        return _text_type(_bound_dialect(dialect, self.dialect_spec).json_text_type, max_chars=self.max_chars)

    def load_dialect_impl(self, dialect):
        """Return the concrete JSON or string fallback type for one dialect.

        Args:
            dialect: Live SQLAlchemy dialect compiling this type.

        Returns:
            Any: Native JSON implementation or string fallback type.
        """
        native = self._native_impl(dialect)
        if native is not None and _bound_dialect(dialect, self.dialect_spec).supports("database-json-native"):
            return dialect.type_descriptor(native)
        return dialect.type_descriptor(self._string_impl(dialect))

    def process_bind_param(self, value, dialect):
        """Convert Python JSON payloads to database parameters.

        Args:
            value: Python JSON-compatible payload bound to the column.
            dialect: Live SQLAlchemy dialect compiling this type.

        Returns:
            Any | None: Native JSON value, serialized JSON text, or ``None``.
        """
        if value is None:
            return None
        return value if _bound_dialect(dialect, self.dialect_spec).supports("database-json-native") else dumps_json(value, compact=True)

    def process_result_value(self, value, dialect):
        """Convert database JSON values back to Python payloads.

        Args:
            value: Database JSON or text value read from the column.
            dialect: Live SQLAlchemy dialect compiling this type.

        Returns:
            Any | None: Python JSON payload, or ``None`` for SQL NULL.
        """
        if value is None:
            return None
        return value if _bound_dialect(dialect, self.dialect_spec).supports("database-json-native") else loads_json(value)


class DatabaseVectorType(sa_types.TypeDecorator):
    """Portable vector type using native arrays, packed float64 blobs, or JSON text.

    Storage is chosen by the bound dialect: ``vector_native_type`` selects a native
    float array, otherwise ``vector_packed_type`` selects a little-endian float64
    BLOB, and otherwise the JSON text fallback is used. A packed dialect also reads
    JSON text, so a column written as text before the dialect declared packed
    storage keeps reading. SQLite JSON-text readers accept TEXT and reject BLOBs.
    Other JSON-text dialects also accept UTF-8 bytes returned for text columns.
    """

    impl = DatabaseJsonType
    cache_ok = True

    def __init__(
        self,
        *args: Any,
        dialect_spec: DatabaseDialectSpec | None = None,
        max_chars: int = 65535,
        **kwargs: Any,
    ) -> None:
        """Create a portable vector type decorator.

        Args:
            *args (Any): Positional arguments forwarded to ``TypeDecorator``.
            dialect_spec (DatabaseDialectSpec | None): Bound dialect metadata used to
                choose native array storage or JSON fallback.
            max_chars (int): Maximum string length used by JSON fallback storage.
            **kwargs (Any): Keyword arguments forwarded to ``TypeDecorator``.

        Returns:
            None: This function does not return a value.
        """
        # TypeDecorator constructs ``impl`` from the forwarded arguments, but
        # those values are not retained on the outer decorator.  Vector bind
        # and result processing run on that outer instance, so keep the exact
        # dialect authority there as well as on its JSON implementation.
        self.dialect_spec = dialect_spec
        self.max_chars = int(max_chars)
        super().__init__(*args, dialect_spec=dialect_spec, max_chars=max_chars, **kwargs)

    @cached_property
    def _static_cache_key(self):
        # SQLAlchemy derives cache keys from positional constructor names.
        # The keyword-only storage settings also change the compiled projection.
        key = super()._static_cache_key
        if not isinstance(key, tuple):
            return key
        return (*key, ("dialect_spec", self.dialect_spec), ("max_chars", self.max_chars))

    def _packs(self, dialect) -> bool:
        spec = _bound_dialect(dialect, self.dialect_spec)
        return spec.vector_native_type is None and spec.vector_packed_type == SQL_TYPE_FLOAT64_BLOB

    def load_dialect_impl(self, dialect):
        """Return native array storage, packed BLOB storage, or the JSON fallback type.

        Args:
            dialect: Live SQLAlchemy dialect compiling this type.

        Returns:
            Any: Native float array implementation, ``LargeBinary``, or JSON fallback type.
        """
        native = _native_vector_type(_bound_dialect(dialect, self.dialect_spec).vector_native_type)
        if native is not None:
            return dialect.type_descriptor(native)
        if self._packs(dialect):
            storage = _SQLitePackedVectorBinary() if dialect.name == "sqlite" else _PackedVectorBinary()
            return dialect.type_descriptor(storage)
        storage = DatabaseJsonType(dialect_spec=self.dialect_spec, max_chars=self.max_chars).load_dialect_impl(dialect)
        return _SQLiteVectorText(storage) if dialect.name == "sqlite" else storage

    def process_bind_param(self, value, dialect):
        """Convert vector values to database parameters.

        Args:
            value: Iterable of numeric vector components.
            dialect: Live SQLAlchemy dialect compiling this type.

        Returns:
            list[float] | bytes | str | None: Native float list, packed float64 bytes,
            JSON fallback text, or ``None``.
        """
        if value is None:
            return None
        vec = [float(item) for item in value]
        if _bound_dialect(dialect, self.dialect_spec).vector_native_type is not None:
            return vec
        if self._packs(dialect):
            return _pack_vector(vec)
        return vec if _bound_dialect(dialect, self.dialect_spec).supports("database-json-native") else _vector_text(vec)

    def process_result_value(self, value, dialect):
        """Convert a stored vector back to a list of floats.

        SQLite Core projections encode the storage class in SQL. For typed
        textual statements with a byte text factory, the cursor marks TEXT
        as sqlite3 materializes rows. Drivers on other JSON-text dialects may return UTF-8
        bytes for text columns.

        Args:
            value: Native array, packed float64 bytes, JSON text (``str`` or
                ``bytes``), or native JSON list read from the column.
            dialect: Live SQLAlchemy dialect compiling this type.

        Returns:
            list[float] | None: Python float vector, or ``None`` for SQL NULL.

        Raises:
            ValueError: If a packed payload is not a whole number of float64
                values, a SQLite JSON-text column returns a BLOB, or bytes
                returned as JSON text on another dialect do not decode.
        """
        if value is None:
            return None
        if isinstance(value, _SQLiteTextBytes):
            return _text_vector(bytes(value.value))
        if isinstance(value, (bytes, bytearray, memoryview)):
            if self._packs(dialect):
                return _unpack_vector(value)
            raw = bytes(value)
            if dialect.name == "sqlite":
                raise _sqlite_vector_blob_error(len(raw))
            spec = _bound_dialect(dialect, self.dialect_spec)
            try:
                return _text_vector(raw.decode("utf-8"))
            except (ValueError, TypeError) as exc:
                raise ValueError(
                    f"Vector column returned {len(raw)} bytes that are not JSON text; dialect {spec.identifier!r} stores vectors as "
                    f"{spec.vector_packed_type or 'JSON text'}, so a packed float64 row needs vector_packed_type: float64_blob"
                ) from exc
        if isinstance(value, str):
            return _text_vector(value)
        return [float(item) for item in value]


class _SQLiteTextBytes:
    """SQLite TEXT returned as bytes with its storage class intact."""

    def __init__(self, value: bytes | bytearray | memoryview) -> None:
        self.value = value


_sqlite_vector_fetch = ContextVar("sqlite_vector_fetch", default=None)
_sqlite_text_factory_lock = Lock()


class _SQLiteVectorTextFactory:
    """Tag TEXT only while materializing rows for the requesting cursor."""

    def __init__(self, original) -> None:
        self.original = original
        self.users = 0

    def __call__(self, value: bytes) -> Any:
        scope = _sqlite_vector_fetch.get()
        if scope is None or scope[0] is not self:
            return self.original(value)
        _, driver_frame, method_code = scope
        caller = sys._getframe(1)
        # SQLite can invoke Python callbacks while stepping. Their own
        # cursor calls use the application factory without tagging.
        if caller is not driver_frame and (caller.f_back is not driver_frame or caller.f_code is not method_code):
            return self.original(value)
        token = _sqlite_vector_fetch.set(None)
        try:
            converted = self.original(value)
        finally:
            _sqlite_vector_fetch.reset(token)
        if isinstance(converted, (bytes, bytearray, memoryview)):
            return _SQLiteTextBytes(converted)
        return converted


@contextmanager
def _tag_sqlite_vector_text(connection, driver_frame, method):
    """Temporarily preserve the origin of TEXT converted to bytes."""
    with _sqlite_text_factory_lock:
        factory = connection.text_factory
        if factory is not str:
            if not isinstance(factory, _SQLiteVectorTextFactory):
                factory = connection.text_factory = _SQLiteVectorTextFactory(factory)
            factory.users += 1
    if factory is str:
        yield
        return

    token = _sqlite_vector_fetch.set((factory, driver_frame, getattr(method, "__code__", None)))
    try:
        yield
    finally:
        _sqlite_vector_fetch.reset(token)
        with _sqlite_text_factory_lock:
            factory.users -= 1
            if not factory.users and connection.text_factory is factory:
                connection.text_factory = factory.original


@contextmanager
def _sqlite_application_text_factory(connection):
    """Expose the configured factory while calling the application's row factory."""
    with _sqlite_text_factory_lock:
        factory = connection.text_factory
        if isinstance(factory, _SQLiteVectorTextFactory):
            connection.text_factory = factory.original
    try:
        yield
    finally:
        with _sqlite_text_factory_lock:
            if isinstance(factory, _SQLiteVectorTextFactory) and not isinstance(connection.text_factory, _SQLiteVectorTextFactory):
                factory.original = connection.text_factory
                connection.text_factory = factory


class _SQLiteVectorCursor:
    """Preserve textual vector values while SQLite materializes result rows."""

    def __init__(self, cursor: SQLiteCursor, vector_positions: set[int]) -> None:
        self._cursor = cursor
        self._vector_positions = vector_positions

    def __getattr__(self, name: str) -> Any:
        return getattr(self._cursor, name)

    @property
    def arraysize(self):
        return self._cursor.arraysize

    @arraysize.setter
    def arraysize(self, value):
        self._cursor.arraysize = value

    @property
    def row_factory(self):
        return self._cursor.row_factory

    @row_factory.setter
    def row_factory(self, value):
        self._cursor.row_factory = value

    def execute(self, *args, **kwargs):
        execute = self._cursor.execute
        with _tag_sqlite_vector_text(self._cursor.connection, sys._getframe(), execute):
            execute(*args, **kwargs)
        return self

    def __iter__(self):
        return self

    def __next__(self):
        return self._fetch(self._cursor.__next__)

    def _fetch(self, fetch, *args):
        cursor = self._cursor
        row_factory = cursor.row_factory

        def preserve_vector_columns(cursor, row):
            values = tuple(value.value if isinstance(value, _SQLiteTextBytes) else value for value in row)
            if row_factory is not None:
                token = _sqlite_vector_fetch.set(None)
                try:
                    with _sqlite_application_text_factory(cursor.connection):
                        result = row_factory(cursor, values)
                finally:
                    _sqlite_vector_fetch.reset(token)
            else:
                result = values
            tagged = {index: row[index] for index in self._vector_positions if isinstance(row[index], _SQLiteTextBytes)}
            if not tagged or result is None:
                return result
            # Preserve a custom factory's replacements. Unchanged cells keep
            # their SQLite storage class for SQLAlchemy's result processor.
            return tuple(tagged[index] if index in tagged and value is values[index] else value for index, value in enumerate(result))

        with _tag_sqlite_vector_text(cursor.connection, sys._getframe(), fetch):
            try:
                cursor.row_factory = preserve_vector_columns
                return fetch(*args)
            finally:
                cursor.row_factory = row_factory

    def fetchone(self):
        return self._fetch(self._cursor.fetchone)

    def fetchmany(self, *args):
        return self._fetch(self._cursor.fetchmany, *args)

    def fetchall(self):
        return self._fetch(self._cursor.fetchall)


class _SQLiteVectorExecutionContext(SQLiteExecutionContext):
    """Wrap a typed cursor before SQLAlchemy dispatches execution listeners."""

    def pre_exec(self):
        super().pre_exec()
        cursor = self.cursor
        result_columns = self.result_column_struct
        if not isinstance(cursor, SQLiteCursor) or cursor.connection.text_factory is str or not result_columns:
            return
        if any(isinstance(column.type, DatabaseVectorType) for column in result_columns[0]):
            self.cursor = _SQLiteVectorCursor(cursor, set())


if sys.version_info[:2] == (3, 10):
    # Python 3.10 materializes the first row in execute(). Install the cursor
    # wrapper before SQLAlchemy saves the cursor for execution and its events.
    SQLiteDialect_pysqlite.execution_ctx_cls = _SQLiteVectorExecutionContext


@sa.event.listens_for(sa.Engine, "after_cursor_execute")
def _preserve_sqlite_textual_vectors(conn, cursor, statement, parameters, context, executemany):
    prefetched = isinstance(cursor, _SQLiteVectorCursor)
    if not (prefetched or isinstance(cursor, SQLiteCursor)) or cursor.description is None:
        return
    if cursor.connection.text_factory is str and not prefetched:
        return
    # Use the result map SQLAlchemy built for this execution. It is available
    # for precompiled statements and ORM from_statement(), too. Core vector
    # projections already use _SQLiteVectorResult as their compiled type.
    result_columns = context.result_column_struct
    if not result_columns:
        return
    columns, ordered, textual_ordered, ad_hoc_textual, loose_names = result_columns
    if not any(isinstance(column.type, DatabaseVectorType) for column in columns):
        return
    if textual_ordered or ((ordered or ad_hoc_textual) and len(columns) == len(cursor.description)):
        positions = {index for index, column in enumerate(columns[: len(cursor.description)]) if isinstance(column.type, DatabaseVectorType)}
    else:
        types_by_name = {}
        for column in columns:
            types_by_name.setdefault(column.keyname, column.type)
            if loose_names:
                for name in column.objects:
                    if isinstance(name, str):
                        types_by_name.setdefault(name, column.type)
        positions = {
            index
            for index, (name, *_) in enumerate(cursor.description)
            if isinstance(types_by_name.get(context._translate_colname(name)[0]), DatabaseVectorType)
        }
    # A named result map may match no columns. Strip any prefetch tags from
    # the untyped cells even in that case.
    if prefetched:
        cursor._vector_positions = positions
    elif positions:
        context.cursor = _SQLiteVectorCursor(cursor, positions)
