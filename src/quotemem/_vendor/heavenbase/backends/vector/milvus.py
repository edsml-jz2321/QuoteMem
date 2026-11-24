"""Milvus vector backend."""

from __future__ import annotations

__all__ = [
    "MilvusBackend",
    "MilvusLiteBackend",
]

import json
import os
import threading
from bisect import bisect_left
from collections.abc import Collection
from contextlib import closing
from dataclasses import replace
from itertools import chain

from heavenbase.utils import Any, Iterable, Iterator, Mapping, Sequence, safe_error
from heavenbase.utils.files import delete_path, file_uri_to_path, get_file_dir, get_file_ext, get_file_name, touch_dir
from heavenbase.utils.redact import redact_secrets

from ...execution.frame.result import ResultFrame
from ...modules import ModuleResolver
from ...schema.model import EntitySchema
from ...types.storage import stores_as_integer
from ...types.vector import _scale_cosine_query, cosine_score, euclidean_score, inner_product_score
from .._scope import provider_scope
from ..base import Backend, backend_locked
from ..scanning import execute_scan
from heavenbase.execution.contracts import LOGICAL_OPS, QueryFragment, RowOp

_SQL_COMMON_RESERVED_WORDS = frozenset(
    "all alter and any as asc between by case cast check column constraint create cross current current_date "
    "current_time current_timestamp delete desc distinct drop else end except exists false for foreign from full "
    "grant group having in inner insert intersect into is join left like limit natural not null on or order outer "
    "primary references right select set table then true union unique update using values when where with".split()
)
MILVUS_RESERVED_WORDS = _SQL_COMMON_RESERVED_WORDS | frozenset(
    "array_contains array_contains_all array_contains_any bool double float int int64 json varchar vector".split()
)
MILVUS_MAX_NAME_LENGTH = 255
# A Milvus server rejects a vector search whose limit (topK) is above this value by default.
MILVUS_MAX_TOP_K = 16384
# Bounds on the key listings kept for Milvus Lite keyset enumerations still in progress (see _LiteKeyListings).
_MAX_OPEN_KEY_LISTINGS = 16
_MAX_LISTED_KEYS = 1_000_000
MILVUS_JSON_TEXT_KEY = "__heavenbase_json__"
_MILVUS_JSON_VERSION_KEY = "__heavenbase_json_version__"
# The Json encoder tags integers above 2**53. This odd integer is also
# outside exact float64 precision, so neither a legacy int nor float could
# have been written as this literal JSON number.
_MILVUS_JSON_VERSION = (1 << 53) + 1
# A Milvus double-quoted literal takes every character raw except these four.
_MILVUS_STRING_ESCAPES = str.maketrans({"\\": "\\\\", '"': '\\"', "\n": "\\n", "\r": "\\r"})


def milvus_json_value(text: Any) -> Any:
    """Return what a Milvus JSON column stores for one Json field's JSON text.

    A Json field's backend value is its tagged JSON text. Milvus keeps a JSON
    object or array as written, but Milvus Lite refuses a number or Boolean at
    the top level and keeps a top-level string unquoted, and a Milvus server
    returns a top-level string as the bare string, which is no longer JSON
    text. Store scalars in a marked object so reads can recover their JSON text
    and leave objects written by earlier releases intact.

    Args:
        text (Any): The field's backend value; anything but a string is returned unchanged.

    Returns:
        Any: The JSON text to write.
    """
    if not isinstance(text, str):
        return text
    if text.lstrip()[:1] in ("{", "["):
        return text
    return json.dumps({MILVUS_JSON_TEXT_KEY: text, _MILVUS_JSON_VERSION_KEY: _MILVUS_JSON_VERSION}, ensure_ascii=False, separators=(",", ":"))


def milvus_json_text(value: Any) -> Any:
    """Return the JSON text a Milvus JSON column value holds, or the value itself.

    Unwrap only the marked two-field object written by ``milvus_json_value``.
    A user object can contain either key and round-trip through the Json encoder.

    Args:
        value (Any): One value Milvus returned for a row field.

    Returns:
        Any: The JSON text inside an object written by ``milvus_json_value``; any other
        value unchanged.
    """
    if not isinstance(value, dict) or len(value) != 2:
        return value
    version = value.get(_MILVUS_JSON_VERSION_KEY)
    text = value.get(MILVUS_JSON_TEXT_KEY)
    if type(version) is not int or version != _MILVUS_JSON_VERSION or not isinstance(text, str):
        return value
    return text


def milvus_string_literal(value: Any) -> str:
    """Return ``str(value)`` as a Milvus double-quoted string literal.

    Milvus and Milvus Lite read any character raw inside a double-quoted
    literal except the quote, the backslash, a line feed and a carriage
    return. Milvus Lite parses only the ``\\\\``, ``\\"``, ``\\n``, ``\\r`` and
    ``\\t`` escapes and rejects ``\\b``, ``\\f``, ``\\uXXXX`` and the other
    JSON or Go escapes, so exactly those four characters are escaped and
    every other character, control characters included, is kept raw.

    Args:
        value (Any): Value whose text the literal holds.

    Returns:
        str: The double-quoted literal.
    """
    return '"' + str(value).translate(_MILVUS_STRING_ESCAPES) + '"'


class _LiteKeyListings:
    """Sorted key listings of open Milvus Lite keyset enumerations, shared by every backend of the process.

    Milvus Lite lets one process at a time open a database file, and all of
    that process's clients share it. Listings are therefore keyed by the file,
    and a write through any backend on the file ends that file's generation:
    it drops the file's listings and refuses to keep a listing made before
    it. ``Backend.end_scan`` drops the listing of an enumeration that stops
    early, as ``ws.iter_rows`` does when its iterator is closed. A caller that
    pages ``scan_page`` itself and abandons it without ``end_scan`` leaves its
    listing here, so all open listings are bounded: beyond
    ``_MAX_OPEN_KEY_LISTINGS`` listings or ``_MAX_LISTED_KEYS`` retained keys
    the oldest listings are dropped, abandoned or not. An enumeration whose
    listing was dropped continues another open listing that already served its
    cursor, as ``resume`` allows, and otherwise lists its keys again from its
    cursor; that other listing's own enumeration then lists its keys again.
    The bounds never drop the listing a page has just kept, the newest, so a
    single enumeration stays linear at any size.
    """

    def __init__(self) -> None:
        # Reentrant: a generator finalized by the garbage collector inside a locked section may end its scan.
        # Such a finalizer may drop a listing at any allocation, even one inside a C-level copy of the dict on
        # Python 3.10 and 3.11, so loops below walk a copy of the keys alone, which allocates nothing per entry,
        # and look each entry up again, skipping one dropped meanwhile.
        self._lock = threading.RLock()
        self._generations: dict[str, int] = {}
        # (file, collection, cursor) -> (sorted keys, index of the first key after the cursor, file generation)
        self._listings: dict[tuple[str, str, str], tuple[list[str], int, int]] = {}

    def generation(self, path: str) -> int:
        """Return the write generation of one database file; take it before listing the file's keys."""
        with self._lock:
            return self._generations.get(path, 0)

    def take(self, path: str, collection: str, cursor: str) -> tuple[list[str], int, int] | None:
        """Remove and return the listing kept for the page after ``cursor``, if one is open."""
        with self._lock:
            return self._listings.pop((path, collection, cursor), None)

    def resume(self, path: str, collection: str, cursor: str) -> tuple[list[str], int, int] | None:
        """Remove and return a listing that continues from ``cursor``, positioned just after it.

        Besides the listing kept for ``cursor`` itself, this takes one whose
        pages already served ``cursor``: a multi-route merge may use a page
        only partway and continue from a key inside it. Of those, the one kept
        for the nearest later cursor is taken. Every kept listing is of its
        file's current generation, so any of them lists every key after
        ``cursor``.
        """
        with self._lock:
            if (listing := self._listings.pop((path, collection, cursor), None)) is not None:
                return listing
            found: tuple[str, str, str] | None
            while True:
                found = None
                for key in list(self._listings):
                    if key[0] != path or key[1] != collection or (found is not None and key[2] >= found[2]):
                        continue
                    if (entry := self._listings.get(key)) is None:
                        continue
                    keys, index, _ = entry
                    position = bisect_left(keys, cursor, 0, index)
                    if position < index and keys[position] == cursor:
                        found = key
                if found is None:
                    return None
                # An enumeration ended during the search may have dropped the listing found; search the rest again.
                if (listing := self._listings.pop(found, None)) is not None:
                    keys, _, generation = listing
                    return keys, bisect_left(keys, cursor) + 1, generation

    def keep(self, path: str, collection: str, cursor: str, listing: tuple[list[str], int, int]) -> None:
        """Keep a listing for the page that continues from ``cursor`` unless the file was written since it was made."""
        with self._lock:
            if listing[2] != self._generations.get(path, 0):
                return
            # Re-inserted, so a listing replacing one kept for the same cursor is the newest and is never dropped below.
            self._listings.pop((path, collection, cursor), None)
            self._listings[(path, collection, cursor)] = listing
            while len(order := list(self._listings)) > 1 and (
                len(order) > _MAX_OPEN_KEY_LISTINGS or sum(len(entry[0]) for key in order if (entry := self._listings.get(key)) is not None) > _MAX_LISTED_KEYS
            ):
                self._listings.pop(order[0], None)

    def forget(self, path: str) -> None:
        """End the current write generation of one database file and drop its listings."""
        with self._lock:
            self._generations[path] = self._generations.get(path, 0) + 1
            for key in list(self._listings):
                if key[0] == path:
                    self._listings.pop(key, None)


_LITE_KEY_LISTINGS = _LiteKeyListings()


class MilvusBackend(Backend):
    """Milvus provider adapter.

    Uses ``pymilvus.MilvusClient`` for persistence, row lookup, native vector
    search, and metadata filtering through provider storage only.
    """

    identifier = "milvus"
    _index_types = ("AUTOINDEX", "FLAT")
    provider_reserved_words = MILVUS_RESERVED_WORDS
    provider_max_name_length = MILVUS_MAX_NAME_LENGTH
    # The largest native search limit; a larger top_k ranks through the portable scan. None means no limit.
    provider_max_top_k: int | None = MILVUS_MAX_TOP_K
    # pymilvus rejects a query-iterator batch above this size (``pymilvus.orm.constants.MAX_BATCH_SIZE``).
    _max_iterator_batch_size = 16384

    @classmethod
    def from_config(
        cls,
        name: str,
        data: Mapping[str, Any],
        *,
        resolver: ModuleResolver | None = None,
        config: Any | None = None,
    ) -> "MilvusBackend":
        """Build the Milvus backend from a config mapping.

        Args:
            name (str): Workspace backend instance name.
            data (Mapping[str, Any]): Backend config mapping. Supported keys are Milvus
                constructor parameters, including ``uri``, ``token``, and ``ws_id``.
            resolver: Context-owned module authority.
            config: Config authority retained by the backend.

        Returns:
            MilvusBackend: Configured Milvus backend instance.
        """
        cfg = dict(data or {})
        ws_id = cfg.pop("ws_id", None)
        return cls(name, cfg.pop("uri", None), cfg.pop("token", None), ws_id=ws_id, resolver=resolver, config=config, **cfg)

    def _lit(self, value: Any) -> str:
        """Return ``value`` as a Milvus double-quoted string literal (see ``milvus_string_literal``)."""
        return milvus_string_literal(value)

    def _key_filter(self, object_ids: Iterable[Any]) -> str:
        """Return the escaped ``object_id in [...]`` expression for primary keys.

        ``MilvusClient.get`` and ``delete(ids=...)`` quote string keys without
        escaping them, so a key holding a quote or backslash breaks (or
        silently changes) their expression; key lookups use this filter instead.
        """
        return f"object_id in [{', '.join(self._lit(item) for item in object_ids)}]"

    def _rows_by_key(self, collection: str, object_ids: Sequence[str], filter_expr: str = "", *, json_fields: Collection[str]) -> dict[str, dict[str, Any]]:
        """Return the stored rows for primary keys, keyed by ``object_id``, under Strong consistency.

        A non-empty ``filter_expr`` is applied in the same query, so a row returned matches it as read.
        """
        expr = self._key_filter(object_ids)
        if filter_expr:
            expr = f"{expr} and ({filter_expr})"
        rows = self.client.query(collection_name=collection, filter=expr, output_fields=["*"], consistency_level="Strong")
        return {str(row["object_id"]): row for row in (self._row_from_milvus(item, json_fields) for item in rows)}

    def __init__(
        self,
        name: str = "milvus",
        uri: str | None = None,
        token: str | None = None,
        ws_id: str | None = None,
        *,
        index_type: str | None = None,
        resolver: ModuleResolver | None = None,
        config: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """Create a Milvus backend.

        Args:
            name (str): Workspace backend instance name.
            uri (str | None): Milvus endpoint URI. When omitted, uses configured backend
                defaults.
            token (str | None): Milvus authentication token.
            ws_id (str | None): Optional workspace namespace for Milvus database selection.
            index_type (str | None): ``AUTOINDEX`` (default) or exhaustive ``FLAT``.
                Existing collections must already match; no automatic index rebuild occurs.
            resolver: Context-owned module authority.
            config: Config authority retained by the backend.
            **kwargs (Any): Extra keyword arguments passed to ``pymilvus.MilvusClient``.

        Returns:
            None: This function does not return a value.
        """
        super().__init__(name, ws_id=ws_id, resolver=resolver, config=config)
        index_key = self.identifier.replace("-", "_")
        configured_index = (
            index_type
            if index_type is not None
            else self._construction_config_get(f"heavenbase.backend_defaults.{index_key}.index_type", default=self._index_types[0])
        )
        if not isinstance(configured_index, str) or configured_index.upper() not in self._index_types:
            raise ValueError(f"{self.identifier} index_type must be one of {self._index_types!r}; got {configured_index!r}")
        self.index_type = configured_index.upper()
        configured_uri = uri or self._construction_config_get(self._uri_config_path(), default=self._default_uri())
        self.uri = self._normalize_uri(str(configured_uri))
        kwargs.setdefault("timeout", float(self._construction_config_get("heavenbase.backend_defaults.milvus.timeout", default=1.0)))
        self.token = token or self._construction_config_get("heavenbase.backend_defaults.milvus.token", default="")
        self._client_kwargs = dict(kwargs)
        self._schemas: dict[str, EntitySchema] = {}
        self._owned_fields: dict[str, set[str]] = {}
        self._collections: dict[str, str] = {}
        self._dirty_collections: set[str] = set()
        self._existing_collections: set[str] = set()
        self.client = None
        self.database = self._database_name()
        self._connect()

    def _uri_config_path(self) -> str:
        """Return the config path owning this provider's default URI."""
        return "heavenbase.backend_defaults.milvus.uri"

    def _default_uri(self) -> str:
        """Return the provider default used when no config override exists."""
        return "http://localhost:19530"

    def _normalize_uri(self, uri: str) -> str:
        """Normalize the configured provider URI."""
        return uri

    def _database_name(self) -> str:
        return self._owner_scope() if self.ws_id else ""

    def _workspace_collection_prefix(self) -> str:
        """Return a collection prefix when the deployment has no database namespace."""
        database = getattr(self, "database", self._database_name())
        return "" if database else f"{self._owner_scope()}_"

    def _owner_scope(self) -> str:
        """Return one bounded physical owner for this backend instance."""
        return provider_scope(self.ws_id, self.name, self.identifier, max_length=96)

    def _close_client(self) -> None:
        """Close the current provider client before replacing or discarding it."""
        client = self.client
        self.client = None
        close = getattr(client, "close", None)
        if callable(close):
            close()

    def _vector_index_type(self) -> str:
        """Return the vector index type supported by this deployment."""
        return self.index_type

    def _supports_array_index(self) -> bool:
        """Return whether Array fields may receive a provider scalar index."""
        return True

    def _connect(self) -> None:
        try:
            from pymilvus import MilvusClient

            client_kwargs = dict(self._client_kwargs)
            self.database = self._database_name()
            if self.database:
                self._ensure_database(MilvusClient, self.database)
                client_kwargs["db_name"] = self.database
            self.client = MilvusClient(uri=self.uri, token=self.token, **client_kwargs)
            self.client_error = ""
        except Exception as exc:
            self.client = None
            self.client_error = redact_secrets(safe_error(exc))

    def _ensure_database(self, client_cls: Any, db_name: str) -> None:
        admin = None
        try:
            admin = client_cls(uri=self.uri, token=self.token, **self._client_kwargs)
            list_databases = getattr(admin, "list_databases", None)
            create_database = getattr(admin, "create_database", None)
            names = [str(item.get("name") if hasattr(item, "get") else item) for item in (list_databases() if list_databases else [])]
            if create_database and db_name not in names:
                try:
                    create_database(db_name)
                except Exception:
                    refreshed = [str(item.get("name") if hasattr(item, "get") else item) for item in (list_databases() if list_databases else [])]
                    if db_name not in refreshed:
                        raise
        finally:
            close = getattr(admin, "close", None)
            if callable(close):
                close()

    def _refresh_workspace(self) -> None:
        close_error = None
        try:
            self._close_client()
        except Exception as exc:
            close_error = exc
        self._schemas.clear()
        self._owned_fields.clear()
        self._collections.clear()
        self._dirty_collections.clear()
        self._existing_collections.clear()
        self._connect()
        if close_error is not None:
            raise close_error

    def _delete_collection(self, collection: str) -> None:
        if self.client is None:
            return
        drop_collection = getattr(self.client, "drop_collection", None)
        if drop_collection is None:
            return
        try:
            drop_collection(collection_name=collection)
        except TypeError:
            drop_collection(collection)
        self._existing_collections.discard(collection)

    @backend_locked
    def drop(self) -> None:
        """Drop the current Milvus database or known collections.

        Args:
            None.

        Returns:
            None: This function does not return a value.
        """
        admin = None
        try:
            from pymilvus import MilvusClient

            self._need_client()
            collections_to_drop: set[str] = set(self._collections.values())
            prefix = self._workspace_collection_prefix()
            listed = {str(name) for name in self.client.list_collections()}
            collections_to_drop.update(listed if self.database else {name for name in listed if name.startswith(prefix)})
            for collection in list(collections_to_drop):
                self._delete_collection(collection)

            if self.database:
                self._close_client()
                admin = MilvusClient(uri=self.uri, token=self.token, **self._client_kwargs)
                drop_database = getattr(admin, "drop_database", None)
                if not callable(drop_database):
                    raise RuntimeError("Milvus client does not expose database deletion")
                try:
                    drop_database(db_name=self.database)
                except TypeError:
                    drop_database(self.database)
        except Exception as exc:
            self.client_error = redact_secrets(safe_error(exc))
            raise RuntimeError(f"Milvus drop failed for backend {self.name!r}: {self.client_error}") from exc
        finally:
            close = getattr(admin, "close", None)
            if callable(close):
                close()
            self._close_client()
            self._schemas.clear()
            self._owned_fields.clear()
            self._collections.clear()
            self._dirty_collections.clear()
            self._existing_collections.clear()

    def _need_client(self) -> None:
        if self.client is None:
            detail = f": {self.client_error}" if self.client_error else ""
            raise RuntimeError(f"Milvus backend {self.name!r} is unavailable{detail}")

    def _collection(self, entity_id: str) -> str:
        return self._collections.setdefault(entity_id, f"{self._workspace_collection_prefix()}{self._physical_name(entity_id)}")

    def schema_warnings(self, schema: EntitySchema, bindings: Sequence[Any]) -> list[str]:
        """Return Milvus collection and field-name warnings for this schema.

        Args:
            schema (EntitySchema): Entity schema being registered.
            bindings (Sequence[Any]): Storage bindings for the entity.

        Returns:
            list[str]: Provider naming warnings for physical collection and field names.
        """
        warnings = super().schema_warnings(schema, bindings)
        warnings.extend(
            self._physical_name_warnings(
                [(schema.entity_id, self._collection(schema.entity_id))],
                kind="Milvus collection",
                max_length=self.provider_max_name_length,
                reserved=self.provider_reserved_words,
                case_sensitive=False,
            )
        )
        warnings.extend(
            self._physical_name_warnings(
                [(field_name, field_name) for field_name in sorted(schema.fields)],
                kind=f"Milvus field for {schema.entity_id!r}",
                max_length=self.provider_max_name_length,
                reserved=self.provider_reserved_words,
                case_sensitive=False,
            )
        )
        return warnings

    def _vector_fields(self, schema: EntitySchema) -> list[Any]:
        return [field_schema for field_schema in schema.fields.values() if type(field_schema.dtype).identifier == "vector"]

    def _varchar_len(self, dtype: Any) -> int:
        max_len = int(self.config.get("heavenbase.backend_defaults.milvus.varchar.max_length", default=65535))
        return min(int(getattr(dtype, "max_chars", max_len)), max_len)

    def _add_json_field(self, schema_def: Any, field_schema: Any) -> None:
        from pymilvus import DataType

        json_type = getattr(DataType, "JSON", None)
        if json_type is None:
            raise RuntimeError(f"Milvus DataType.JSON is required for {field_schema.name!r}")
        schema_def.add_field(field_schema.name, json_type, nullable=True)

    def _add_field(self, schema_def: Any, index_params: Any, field_schema: Any, metric: str) -> None:
        from pymilvus import DataType

        name = type(field_schema.dtype).identifier
        if field_schema.name == "object_id":
            return
        if name == "vector":
            schema_def.add_field(field_schema.name, DataType.FLOAT_VECTOR, dim=field_schema.dtype.dim)
            index_params.add_index(field_name=field_schema.name, metric_type=metric, index_type=self._vector_index_type())
        elif name in ("identifier", "short-text", "medium-text", "long-text", "artifact", "date") or (
            name == "categorical" and field_schema.dtype.base == "string"
        ):
            schema_def.add_field(field_schema.name, DataType.VARCHAR, max_length=self._varchar_len(field_schema.dtype), nullable=True)
        elif stores_as_integer(field_schema.dtype):
            schema_def.add_field(field_schema.name, DataType.INT64, nullable=True)
        elif name == "float":
            schema_def.add_field(field_schema.name, DataType.DOUBLE, nullable=True)
        elif name == "boolean":
            schema_def.add_field(field_schema.name, DataType.BOOL, nullable=True)
        elif name == "array":
            if self._add_array_field(schema_def, field_schema):
                if self._supports_array_index():
                    index_params.add_index(field_name=field_schema.name, index_type="AUTOINDEX")
            else:
                self._add_json_field(schema_def, field_schema)
        else:
            self._add_json_field(schema_def, field_schema)

    def _array_element_type(self, item_type: Any) -> Any:
        from pymilvus import DataType

        name = type(item_type).identifier if item_type is not None else ""
        if name in ("identifier", "short-text", "medium-text", "long-text", "date") or (name == "categorical" and item_type.base == "string"):
            return DataType.VARCHAR
        if stores_as_integer(item_type):
            return DataType.INT64
        if name == "float":
            return DataType.DOUBLE
        if name == "boolean":
            return DataType.BOOL
        return None

    def _add_array_field(self, schema_def: Any, field_schema: Any) -> bool:
        from pymilvus import DataType

        item_type = getattr(field_schema.dtype, "item_type", None)
        element_type = self._array_element_type(item_type)
        if element_type is None:
            return False
        max_capacity = int(self.config.get("heavenbase.backend_defaults.milvus.array.max_capacity", default=4096))
        kwargs: dict[str, Any] = {"element_type": element_type, "max_capacity": max_capacity, "nullable": True}
        if element_type == DataType.VARCHAR:
            type_length = int(getattr(item_type, "max_chars", 65535))
            kwargs["max_length"] = min(
                type_length,
                int(self.config.get("heavenbase.backend_defaults.milvus.array.max_length", default=65535)),
            )
        schema_def.add_field(field_schema.name, DataType.ARRAY, **kwargs)
        return True

    def _has_collection(self, name: str) -> bool:
        exists = bool(self.client and self.client.has_collection(name))
        if exists:
            self._existing_collections.add(name)
        else:
            self._existing_collections.discard(name)
        return exists

    def _describe_collection(self, collection: str) -> dict[str, Any]:
        describe = getattr(self.client, "describe_collection", None)
        if describe is None:
            return {}
        return dict(describe(collection_name=collection) or {})

    def _collection_field_names(self, description: dict[str, Any]) -> set[str]:
        names = set()
        for field in description.get("fields") or []:
            name = field.get("name") if hasattr(field, "get") else getattr(field, "name", "")
            if name:
                names.add(str(name))
        return names

    def _collection_fields(self, description: dict[str, Any]) -> dict[str, Any]:
        fields = {}
        for field in description.get("fields") or []:
            name = field.get("name") if hasattr(field, "get") else getattr(field, "name", "")
            if name:
                fields[str(name)] = field
        return fields

    def _field_attr(self, field: Any, name: str) -> Any:
        if hasattr(field, "get"):
            return field.get(name)
        return getattr(field, name, None)

    def _field_params(self, field: Any) -> dict[str, Any]:
        params = self._field_attr(field, "params") or self._field_attr(field, "kwargs") or {}
        if hasattr(params, "to_dict"):
            params = params.to_dict()
        return dict(params) if hasattr(params, "items") else {}

    def _same_descriptor_value(self, actual: Any, expected: Any) -> bool:
        if actual in (None, ""):
            return True
        try:
            return int(actual) == int(expected)
        except Exception:
            return str(actual) == str(expected)

    def _expected_field_metadata(self, field_schema: Any) -> dict[str, Any]:
        from pymilvus import DataType

        name = type(field_schema.dtype).identifier
        if name == "vector":
            return {"dtype": DataType.FLOAT_VECTOR, "dim": field_schema.dtype.dim}
        if name in ("identifier", "short-text", "medium-text", "long-text", "artifact", "date") or (
            name == "categorical" and field_schema.dtype.base == "string"
        ):
            return {"dtype": DataType.VARCHAR, "max_length": self._varchar_len(field_schema.dtype)}
        if stores_as_integer(field_schema.dtype):
            return {"dtype": DataType.INT64}
        if name == "float":
            return {"dtype": DataType.DOUBLE}
        if name == "boolean":
            return {"dtype": DataType.BOOL}
        if name == "array":
            item_type = getattr(field_schema.dtype, "item_type", None)
            element_type = self._array_element_type(item_type)
            if element_type is not None:
                metadata = {
                    "dtype": DataType.ARRAY,
                    "element_type": element_type,
                    "max_capacity": int(self.config.get("heavenbase.backend_defaults.milvus.array.max_capacity", default=4096)),
                }
                if element_type == DataType.VARCHAR:
                    metadata["max_length"] = min(
                        int(getattr(item_type, "max_chars", 65535)),
                        int(self.config.get("heavenbase.backend_defaults.milvus.array.max_length", default=65535)),
                    )
                return metadata
        return {"dtype": DataType.JSON}

    def _object_id_max_length(self, schema: EntitySchema) -> int:
        """Return the ``object_id`` VARCHAR length a new collection declares for ``schema``.

        The column takes the entity's own ``object_id`` domain, capped by the
        Milvus VARCHAR limit and, when it is set, by
        ``heavenbase.backend_defaults.milvus.object_id.max_length``.
        """
        from ...types import Identifier

        object_id = schema.fields.get("object_id")
        length = self._varchar_len(object_id.dtype if object_id is not None else Identifier())
        cap = self.config.get("heavenbase.backend_defaults.milvus.object_id.max_length", default=None)
        return length if cap is None else min(length, int(cap))

    def _expected_object_id_metadata(self) -> dict[str, Any]:
        """Return the ``object_id`` metadata an existing collection must carry.

        Its ``max_length`` is not compared: a collection created by an earlier
        release declares 63 characters whatever the entity's ``object_id``
        domain, and it stays usable for the ids that fit.
        """
        from pymilvus import DataType

        return {"dtype": DataType.VARCHAR, "is_primary": True}

    def _collection_metadata_mismatches(
        self,
        schema: EntitySchema,
        fields: dict[str, Any],
        owned_fields: set[str],
    ) -> list[str]:
        mismatches = []
        checks = [("object_id", self._expected_object_id_metadata())]
        checks.extend(
            (field_schema.name, self._expected_field_metadata(field_schema)) for field_schema in schema.fields.values() if field_schema.name in owned_fields
        )
        for field_name, expected in checks:
            field = fields.get(field_name)
            if field is None:
                continue
            actual_dtype = self._field_attr(field, "type") or self._field_attr(field, "dtype") or self._field_attr(field, "data_type")
            if actual_dtype is not None and not self._same_descriptor_value(actual_dtype, expected["dtype"]):
                mismatches.append(f"{field_name}.dtype expected {expected['dtype']!r} got {actual_dtype!r}")
            params = self._field_params(field)
            for key, expected_value in expected.items():
                if key == "dtype":
                    continue
                actual_value = self._field_attr(field, key)
                if actual_value in (None, ""):
                    actual_value = params.get(key)
                if key == "is_primary" and actual_value in (None, ""):
                    mismatches.append(f"{field_name}.{key} expected {expected_value!r} got missing")
                    continue
                if actual_value not in (None, "") and not self._same_descriptor_value(actual_value, expected_value):
                    mismatches.append(f"{field_name}.{key} expected {expected_value!r} got {actual_value!r}")
        return mismatches

    def _validate_existing_collection(
        self,
        schema: EntitySchema,
        collection: str,
        owned_fields: set[str],
    ) -> None:
        description = self._describe_collection(collection)
        fields = self._collection_fields(description)
        names = set(fields)
        if not names:
            raise RuntimeError(f"Milvus collection {collection!r} does not expose physical field metadata")
        required = {"object_id", *owned_fields}
        missing = sorted(required - names)
        extra = sorted(names - required)
        dynamic = bool(description.get("enable_dynamic_field", False))
        mismatched = self._collection_metadata_mismatches(schema, fields, owned_fields)
        if "auto_id" in description and not self._same_descriptor_value(description["auto_id"], False):
            mismatched.append(f"auto_id expected False got {description['auto_id']!r}")
        if not (missing or extra or dynamic or mismatched):
            mismatched.extend(
                self._vector_index_metadata_mismatches(
                    collection,
                    [field_schema for field_schema in self._vector_fields(schema) if field_schema.name in owned_fields],
                )
            )
        if missing or extra or dynamic or mismatched:
            reason = f"dynamic_field={dynamic}, missing_fields={missing}, extra_fields={extra}, mismatched_fields={mismatched}"
            raise RuntimeError(f"Milvus collection {collection!r} does not match HeavenBase physical schema: {reason}")

    def _vector_index_metadata_mismatches(self, collection: str, vectors: Sequence[Any]) -> list[str]:
        """Return existing-index differences from the configured near-search contract."""
        list_indexes = getattr(self.client, "list_indexes", None)
        describe_index = getattr(self.client, "describe_index", None)
        if not callable(list_indexes) or not callable(describe_index):
            raise RuntimeError("Milvus collection validation requires list_indexes and describe_index")
        expected_metric = self._configured_metric()
        expected_index_type = self._vector_index_type().upper()
        mismatches: list[str] = []
        for field_schema in vectors:
            index_names = list_indexes(collection_name=collection, field_name=field_schema.name) or []
            if isinstance(index_names, str):
                index_names = [index_names]
            descriptors = [describe_index(collection_name=collection, index_name=index_name) for index_name in index_names]
            descriptors = [
                descriptor for descriptor in descriptors if str(self._field_attr(descriptor, "field_name") or field_schema.name) == field_schema.name
            ]
            if not descriptors:
                mismatches.append(f"{field_schema.name}.index expected {expected_index_type!r} got missing")
                continue
            compatible = False
            actual_contracts: list[tuple[str, str]] = []
            for descriptor in descriptors:
                actual_metric = str(self._field_attr(descriptor, "metric_type") or "").upper()
                actual_index_type = str(self._field_attr(descriptor, "index_type") or "").upper()
                actual_contracts.append((actual_metric, actual_index_type))
                if actual_metric == expected_metric and actual_index_type == expected_index_type:
                    compatible = True
            if not compatible:
                mismatches.append(f"{field_schema.name}.index expected ({expected_metric!r}, {expected_index_type!r}) " f"got {actual_contracts!r}")
        return mismatches

    def _flush(self, collection: str) -> None:
        """Force a Milvus flush for operations that need sealed segments.

        Kept for ``execute`` / ``search`` ANN index coverage. Removed from
        ``get_many``, ``scan``, and exact ``count(*)`` because Strong scalar
        queries already see growing segments; the flush RPC was adding
        multi-second stalls to point-read workloads.
        """
        if collection not in self._dirty_collections:
            return
        flush = getattr(self.client, "flush", None)
        if flush:
            flush(collection_name=collection)
        self._dirty_collections.discard(collection)

    @backend_locked
    def ensure(self, schema: EntitySchema, bindings: Sequence[Any]) -> None:
        """Ensure a Milvus collection and vector index for an entity.

        Args:
            schema (EntitySchema): Entity schema to store in Milvus.
            bindings (Sequence[Any]): Storage bindings supplied by the workspace.

        Returns:
            None: This function does not return a value.

        Raises:
            ValueError: If fields route to this Milvus binding without a routed Vector field.
            RuntimeError: If an existing collection does not match the expected physical schema.
        """
        owned_fields = {binding.field_name for binding in bindings if binding.backend == self.name and binding.field_name != schema.pk_name}
        if not owned_fields:
            self._schemas[schema.entity_id] = schema
            self._owned_fields[schema.entity_id] = set()
            return
        vectors = [field_schema for field_schema in self._vector_fields(schema) if field_schema.name in owned_fields]
        if not vectors:
            raise ValueError(
                f"Milvus backend {self.name!r} cannot store {schema.entity_id!r} without a routed Vector field; " f"routed fields: {sorted(owned_fields)}"
            )
        optional_vectors = sorted(field_schema.name for field_schema in vectors if not field_schema.required)
        if optional_vectors:
            raise ValueError(
                "Milvus routed Vector fields must be required because provider rows cannot represent a missing "
                f"FLOAT_VECTOR value; optional fields: {optional_vectors}"
            )
        if any(field_schema.dtype.dim is None for field_schema in vectors):
            raise ValueError("Milvus native vector search requires fixed-dimension Vector fields")
        self._schemas[schema.entity_id] = schema
        self._owned_fields[schema.entity_id] = owned_fields
        if self.client is None:
            return
        collection = self._collection(schema.entity_id)
        if self._has_collection(collection):
            self._validate_existing_collection(schema, collection, owned_fields)
            self._load_collection(collection)
            return
        metric = self._configured_metric()
        from pymilvus import DataType

        schema_def = self.client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema_def.add_field("object_id", DataType.VARCHAR, is_primary=True, max_length=self._object_id_max_length(schema))
        index_params = self.client.prepare_index_params()
        for field_schema in schema.fields.values():
            if field_schema.name not in owned_fields:
                continue
            self._add_field(schema_def, index_params, field_schema, metric)
        try:
            self.client.create_collection(
                collection_name=collection,
                schema=schema_def,
                index_params=index_params,
                consistency_level="Strong",
            )
        except Exception:
            if not self._has_collection(collection):
                raise
            self._validate_existing_collection(schema, collection, owned_fields)
            self._load_collection(collection)
        self._existing_collections.add(collection)

    def _load_collection(self, collection: str) -> None:
        """Load an existing collection into memory so reads can run.

        A collection created here with index parameters is loaded by
        ``create_collection``. One opened from an earlier client may be released;
        Milvus Lite releases every collection when its process ends. Loading an
        already loaded collection is a no-op.

        Args:
            collection (str): Physical collection name.

        Returns:
            None: This function does not return a value.
        """
        self.client.load_collection(collection_name=collection)

    @backend_locked
    def upsert(self, rows: Sequence[RowOp]) -> None:
        """Upsert available vectors and scalar fields into Milvus.

        Args:
            rows (Sequence[RowOp]): Row mutations to apply. ``delete`` operations remove
                primary-key rows; other operations upsert scalar and vector values.

        Returns:
            None: This function does not return a value.

        Raises:
            RuntimeError: If the Milvus client is unavailable.
        """
        self._need_client()
        for entity_id, is_delete, ops in self.ordered_row_op_batches(rows):
            collection = self._collection(entity_id)
            if not self._has_collection(collection):
                if not is_delete:
                    raise RuntimeError(f"Milvus collection {collection!r} disappeared after ensure; " f"refusing to discard {len(ops)} upsert operation(s)")
                continue
            if is_delete:
                self.client.delete(collection_name=collection, filter=self._key_filter(str(row.object_id) for row in ops))
            else:
                values = []
                json_fields = self._json_fields(entity_id)
                for row in ops:
                    item = row.to_row(fields=self._owned_fields.get(entity_id))
                    item["object_id"] = str(row.object_id)
                    for name in json_fields.intersection(item):
                        item[name] = milvus_json_value(item[name])
                    values.append(item)
                self.client.upsert(collection_name=collection, data=values)
            self._dirty_collections.add(collection)

    @backend_locked
    def scan(self, entity_id: str) -> list[dict[str, Any]]:
        """Return every row from Milvus provider storage by exhausting an iterator.

        Integrity enumeration forces Strong consistency so an immediately preceding
        GraphEdge write cannot be hidden by Milvus's bounded-staleness default.

        Args:
            entity_id (str): Entity identifier whose collection should be scanned.

        Returns:
            list[dict[str, Any]]: Stored rows, or an empty list when the collection is absent.

        Raises:
            NotImplementedError: If the client lacks exhaustive query-iterator support.
            RuntimeError: If the client is unavailable or the scan query fails.
        """
        self._need_client()
        collection = self._collection(entity_id)
        if not self._has_collection(collection):
            return []
        try:
            return self._query_iterator_rows(collection, filter_expr="", batch_config="scan_limit", json_fields=self._json_fields(entity_id))
        except NotImplementedError:
            raise
        except Exception as exc:
            self.client_error = redact_secrets(safe_error(exc))
            raise RuntimeError(f"Milvus scan failed for {entity_id!r}: {self.client_error}") from exc

    def iter_scan(self, entity_id: str, *, batch_size: int, columns: Sequence[str] | None = None) -> Iterable[dict[str, Any]]:
        """Exhaust a Milvus primary-key iterator with bounded client memory.

        Args:
            entity_id (str): Logical entity identifier.
            batch_size (int): Positive maximum iterator batch size; a Milvus server caps each batch at 16384 rows.
            columns (Sequence[str] | None): Optional field names to return; ``None`` returns the complete fragment.

        Yields:
            dict[str, Any]: Next normalized row fragment.

        Raises:
            ValueError: If the page size is not positive.
            NotImplementedError: If query iterators are unavailable.
            RuntimeError: If provider enumeration fails.

        Notes:
            Uses Strong consistency and always closes the iterator, including
            on cancellation. This does not establish a cross-backend snapshot,
            and a batch that ends at a key holding a quote, a backslash, LF or
            CR continues on a new iterator at a later timestamp (see
            ``_key_ordered_batches``).
        """
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        selection = self._row_columns(columns)
        with self._lock:
            self._need_client()
            collection = self._collection(entity_id)
            if not self._has_collection(collection):
                return
            batches = self._key_ordered_batches(collection, filter_expr="", batch_size=batch_size, json_fields=self._json_fields(entity_id))
            try:
                for batch in batches:
                    for row in batch:
                        yield self._project_row(row, selection)
            finally:
                batches.close()

    @backend_locked
    def scan_page(self, entity_id: str, *, after: Any, batch_size: int, columns: Sequence[str] | None = None) -> list[dict[str, Any]]:
        """Return one primary-key keyset page from a Strong-consistency query iterator.

        The page is the first key-ordered batch of the rows with
        ``object_id > after`` (see ``_key_ordered_batches``). Varchar keys
        compare bytewise, which for UTF-8 is code-point order, the order of
        Python strings. The enumeration is closed before the page is returned,
        so no cursor or lock outlives the call. On a Milvus server a full scan
        costs one pass over the collection instead of the portable default's
        one pass per page.

        Args:
            entity_id (str): Logical entity identifier.
            after (Any): ``object_id`` the page must follow, or ``None`` for the first page.
            batch_size (int): Positive maximum number of rows in the page; a Milvus server
                returns at most 16384 rows per page, so a larger value yields
                shorter pages.
            columns (Sequence[str] | None): Optional field names to return; ``None`` returns the complete fragment.

        Returns:
            list[dict[str, Any]]: Detached row fragments in ascending ``object_id``
            order, at most ``batch_size`` of them; an empty list ends the scan.

        Raises:
            ValueError: If the page size is not positive.
            NotImplementedError: If query iterators are unavailable.
            RuntimeError: If the provider read fails.
        """
        self._check_batch_size(batch_size)
        selection = self._row_columns(columns)
        self._need_client()
        collection = self._collection(entity_id)
        if not self._has_collection(collection):
            return []
        expr = "" if after is None else f"object_id > {self._lit(after)}"
        batches = self._key_ordered_batches(collection, filter_expr=expr, batch_size=batch_size, json_fields=self._json_fields(entity_id))
        try:
            batch = next(batches, [])
        except NotImplementedError:
            raise
        except Exception as exc:
            self.client_error = redact_secrets(safe_error(exc))
            raise RuntimeError(f"Milvus scan_page failed for {entity_id!r}: {self.client_error}") from exc
        finally:
            batches.close()
        rows = sorted(batch, key=lambda row: str(row["object_id"]))
        return [self._project_row(row, selection) for row in rows[:batch_size]]

    def _key_ordered_batches(self, collection: str, *, filter_expr: str, batch_size: int, json_fields: Collection[str] = ()) -> Iterator[list[dict[str, Any]]]:
        """Yield normalized rows accepted by one scalar expression, in ascending ``object_id`` batches.

        A Milvus server's query iterator pages by a primary-key cursor and returns
        each batch in key order, so its batches are keyset pages. pymilvus rejects
        an iterator batch above 16384 rows, so a larger ``batch_size`` is capped
        there; a batch may hold fewer rows than requested. pymilvus writes the
        iterator cursor as ``object_id > "<last key>"`` without escaping, so a
        batch whose last key holds a quote, a backslash, LF or CR would make the
        next batch fail to parse or start from a different key. After such a
        batch the iterator is closed and a new one starts after the escaped key;
        the new iterator reads at its own MVCC timestamp. Closing the generator
        closes the iterator.

        Raises:
            NotImplementedError: If the client has no query iterator.
        """
        self._prepare_iterator_read(collection)
        factory = getattr(self.client, "query_iterator", None)
        if not callable(factory):
            raise NotImplementedError("Milvus bounded enumeration requires client.query_iterator")
        batch_size = min(batch_size, self._max_iterator_batch_size)
        after: str | None = None
        while True:
            # The cursor goes left of AND, as pymilvus places its own, for operators that must stay right-most.
            cursor = None if after is None else f"object_id > {self._lit(after)}"
            expr = filter_expr if cursor is None else (f"{cursor} and ({filter_expr})" if filter_expr else cursor)
            iterator = factory(collection_name=collection, batch_size=batch_size, filter=expr, output_fields=["*"], consistency_level="Strong")
            after = None
            try:
                while batch := iterator.next():
                    rows = [self._row_from_milvus(row, json_fields) for row in batch]
                    yield rows
                    last = str(rows[-1]["object_id"])
                    if self._lit(last) != f'"{last}"':
                        after = last
                        break
            finally:
                close = getattr(iterator, "close", None)
                if callable(close):
                    close()
            if after is None:
                return

    def _query_iterator_rows(
        self,
        collection: str,
        *,
        filter_expr: str,
        batch_config: str,
        json_fields: Collection[str],
    ) -> list[dict[str, Any]]:
        """Return every row accepted by one Milvus scalar expression."""
        batch_size = max(
            1,
            int(self.config.get(f"heavenbase.backend_defaults.milvus.{batch_config}", default=10000)),
        )
        rows: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        batches = self._key_ordered_batches(collection, filter_expr=filter_expr, batch_size=batch_size, json_fields=json_fields)
        try:
            for batch in batches:
                for normalized in batch:
                    object_id = str(normalized.get("object_id"))
                    if object_id in seen_ids:
                        continue
                    seen_ids.add(object_id)
                    rows.append(normalized)
        finally:
            batches.close()
        return rows

    def _prepare_iterator_read(self, collection: str) -> None:
        """Prepare provider state for an exhaustive iterator read."""
        del collection

    @backend_locked
    def get_many(self, entity_id: str, object_ids: Iterable[Any], *, columns: Sequence[str] | None = None) -> list[dict[str, Any] | None]:
        """Return rows through Milvus primary-key lookup.

        Point lookup also forces Strong consistency because endpoint validation relies
        on a current logical-existence answer.

        Args:
            entity_id (str): Entity identifier whose rows should be loaded.
            object_ids (Iterable[Any]): Object identifiers to fetch in caller order.
            columns (Sequence[str] | None): Optional field names to return; ``None`` returns the complete fragment.

        Returns:
            list[dict[str, Any] | None]: Rows aligned with ``object_ids``. Missing rows are
            returned as ``None``.

        Raises:
            TypeError: If ``columns`` contains a non-string name.
            RuntimeError: If the client is unavailable or primary-key lookup fails.
        """
        selection = self._row_columns(columns)
        ordered_ids = [str(object_id) for object_id in object_ids]
        if not ordered_ids:
            return []
        collection = self._collection(entity_id)
        self._need_client()
        if not self._has_collection(collection):
            return [None for _ in ordered_ids]
        try:
            rows_by_id = self._rows_by_key(collection, list(dict.fromkeys(ordered_ids)), json_fields=self._json_fields(entity_id))
            return self._project_rows([rows_by_id.get(row_id) for row_id in ordered_ids], selection)
        except Exception as exc:
            self.client_error = redact_secrets(safe_error(exc))
            raise RuntimeError(f"Milvus get_many failed for {entity_id!r}: {self.client_error}") from exc

    @backend_locked
    def exists_many(self, entity_id: str, object_ids: Iterable[Any]) -> list[bool]:
        """Return existence flags through Milvus primary-key lookup.

        Args:
            entity_id (str): Entity identifier whose rows should be checked.
            object_ids (Iterable[Any]): Object identifiers to check in caller order.

        Returns:
            list[bool]: Existence flags aligned with ``object_ids``.
        """
        return [row is not None for row in self.get_many(entity_id, object_ids)]

    @backend_locked
    def count(self, entity_id: str) -> int:
        """Return Milvus collection row count without a scan query.

        Args:
            entity_id (str): Entity identifier whose collection should be counted.

        Returns:
            int: Number of rows in the collection, or ``0`` when the collection is absent.

        Raises:
            RuntimeError: If the client is unavailable or the exact count query fails.
        """
        collection = self._collection(entity_id)
        self._need_client()
        if not self._has_collection(collection):
            return 0
        try:
            rows = self.client.query(collection_name=collection, filter="", output_fields=["count(*)"], consistency_level="Strong")
            if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence) or len(rows) != 1:
                raise ValueError(f"expected one count row, got {rows!r}")
            row = rows[0]
            if not isinstance(row, Mapping) or row.get("count(*)") is None:
                raise ValueError(f"missing count(*) value in {row!r}")
            return int(row["count(*)"])
        except Exception as exc:
            self.client_error = redact_secrets(safe_error(exc))
            raise RuntimeError(f"Milvus count failed for {entity_id!r}: {self.client_error}") from exc

    def exact_count(self, entity_id: str) -> int:
        """Return the strong-consistency scalar count as an integrity proof.

        Args:
            entity_id: Logical entity identifier whose fragments are counted.

        Returns:
            int: Exact number of visible Milvus fragments.
        """
        return self.count(entity_id)

    def _live_tag_claims(self) -> dict[str, Any]:
        """Return live tag claims derived from Milvus availability.

        Args:
            None.

        Returns:
            dict[str, Any]: Live Backend tag claims.
        """
        return {
            **super()._live_tag_claims(),
            "backend-availability": "ready" if self.client is not None else "unavailable",
        }

    def candidate_id_fragment(self, object_ids: Sequence[Any]) -> QueryFragment:
        """Return a Milvus object-id predicate for candidate prefiltering.

        Args:
            object_ids (Sequence[Any]): Candidate object identifiers.

        Returns:
            QueryFragment: Native ``object_id in [...]`` fragment for Milvus.
        """
        py_fragment = super().candidate_id_fragment(object_ids)
        return QueryFragment(
            self.name,
            "in",
            "object_id",
            f"object_id in [{', '.join(self._lit(item) for item in object_ids)}]",
            notes={**py_fragment.notes, "kind": "milvus_filter", "mode": "native"},
        )

    def _json_fields(self, entity_id: str | None) -> set[str]:
        """Return the names of the entity's Json fields stored on this backend."""
        schema = self._schemas.get(entity_id) if entity_id is not None else None
        if schema is None:
            return set()
        owned = self._owned_fields.get(entity_id)
        return {name for name, field_schema in schema.fields.items() if type(field_schema.dtype).identifier == "json" and (owned is None or name in owned)}

    @staticmethod
    def _decode_json_fields(row: Any, json_fields: Collection[str]) -> dict[str, Any]:
        """Copy one provider row and unwrap only fields written by the Json codec."""
        item = dict(row or {})
        for name in json_fields:
            if name in item:
                item[name] = milvus_json_text(item[name])
        return item

    def _row_from_milvus(self, row: Any, json_fields: Collection[str]) -> dict[str, Any]:
        item = self._decode_json_fields(row, json_fields)
        item["object_id"] = item.get("object_id") or item.get("id")
        return item

    def _metric(self, metric: str) -> str:
        value = (metric or "cosine").lower()
        metrics = {
            "cosine": "COSINE",
            "l2": "L2",
            "euclidean": "L2",
            "ip": "IP",
            "inner": "IP",
            "inner_product": "IP",
            "dot": "IP",
        }
        if value not in metrics:
            raise ValueError(f"Milvus near: unsupported metric {metric!r}")
        return metrics[value]

    def _configured_metric(self) -> str:
        """Return the provider metric fixed into collection vector indexes."""
        configured = self.config.get("heavenbase.backend_defaults.milvus.metric", default="COSINE")
        return self._metric(str(configured))

    def _hit_rows(self, hits: Any, metric: str, json_fields: Collection[str]) -> list[dict[str, Any]]:
        rows = []
        for hit in (hits or [[]])[0]:
            if isinstance(hit, dict):
                entity = hit.get("entity") or hit.get("fields") or {}
                object_id = hit.get("id")
                distance = hit.get("distance", hit.get("score", 0.0)) or 0.0
            else:
                entity = getattr(hit, "entity", None)
                object_id = getattr(hit, "id", None)
                distance = getattr(hit, "distance", getattr(hit, "score", 0.0)) or 0.0
            row = self._decode_json_fields(entity, json_fields)
            row["object_id"] = row.get("object_id") or object_id
            row["score"] = float(distance) if self._metric(metric) in ("COSINE", "IP") else -float(distance)
            rows.append(row)
        return rows

    def _is_milvus_fragment(self, fragment: QueryFragment) -> bool:
        if fragment.kind() == "milvus_filter":
            return True
        if fragment.op in ("and", "or"):
            return all(self._is_milvus_fragment(child) for child in fragment.children())
        if fragment.op == "not":
            return self._is_milvus_fragment(fragment.children()[0])
        return False

    def _expr(self, fragment: QueryFragment) -> str:
        if fragment.kind() == "milvus_filter":
            return fragment.payload
        if fragment.op == "and":
            return " and ".join(f"({self._expr(child)})" for child in fragment.children())
        if fragment.op == "or":
            return " or ".join(f"({self._expr(child)})" for child in fragment.children())
        if fragment.op == "not":
            return f"not ({self._expr(fragment.children()[0])})"
        raise TypeError(f"Cannot compile {fragment.op!r} to Milvus expr")

    @backend_locked
    def execute(self, fragment: QueryFragment) -> ResultFrame:
        """Execute Milvus-native search and filter fragments when possible.

        Args:
            fragment (QueryFragment): Backend-ready query fragment compiled by the handler layer.

        Returns:
            ResultFrame: Matching rows and object identifiers.

        Raises:
            NotImplementedError: If the fragment cannot be executed natively or by scan fallback.
            RuntimeError: If the Milvus client is unavailable.
        """
        entity_id = fragment.entity_id()
        collection = self._collection(entity_id) if entity_id else ""
        self._need_client()
        if fragment.kind() == "milvus_search" and fragment.op == "near" and self._has_collection(collection):
            params = fragment.payload
            metric = self._metric(params.get("metric", "cosine"))
            if self.identifier == "milvus-lite" and (params.get("exact") is True or metric == "COSINE"):
                return self._lite_near_scan(collection, fragment, metric)
            if fragment.uses_scan():
                return super().execute(fragment)
            self._flush(collection)
            if params.get("exact") is True:
                if self._vector_index_type() != "FLAT":
                    raise RuntimeError("Milvus exact near requires a FLAT index")
                differences = self._vector_index_metadata_mismatches(collection, [self._schemas[entity_id].get(fragment.field)])
                if differences:
                    raise RuntimeError(f"Milvus exact near cannot verify the live FLAT index: {differences!r}")
            if metric != self._configured_metric():
                return super().execute(fragment)
            if self.provider_max_top_k is not None and int(params["top_k"]) > self.provider_max_top_k:
                # The provider rejects this limit; the portable scan returns all top_k neighbours instead of fewer.
                return super().execute(fragment)
            filt = fragment.filter_fragment()
            if filt is not None and not self._is_milvus_fragment(filt):
                return super().execute(fragment)
            output_fields = ["*"]
            if self.identifier == "milvus-lite" and entity_id is not None:
                # Lite's native search returns only the primary key for "*".
                output_fields = sorted(self._owned_fields.get(entity_id) or ()) or output_fields
            vector = params["vector"]
            if metric == "COSINE":
                # Milvus searches float32 vectors; cosine ignores the exact power-of-two scale.
                vector = _scale_cosine_query(vector)
            hits = self.client.search(
                collection_name=collection,
                data=[vector],
                anns_field=fragment.field,
                limit=params["top_k"],
                output_fields=output_fields,
                filter=self._expr(filt) if filt is not None else "",
                search_params={"metric_type": metric},
                consistency_level="Strong",
            )
            return ResultFrame.from_rows(self._hit_rows(hits, params.get("metric", "cosine"), self._json_fields(entity_id)))
        if fragment.mode() == "py_scan":
            return super().execute(fragment)
        if (fragment.kind() == "milvus_filter" or fragment.op in LOGICAL_OPS) and self._has_collection(collection):
            if not self._is_milvus_fragment(fragment):
                return super().execute(fragment)
            self._flush(collection)
            try:
                return ResultFrame.from_rows(
                    self._query_iterator_rows(
                        collection,
                        filter_expr=self._expr(fragment),
                        batch_config="filter_limit",
                        json_fields=self._json_fields(entity_id),
                    )
                )
            except Exception as exc:
                self.client_error = redact_secrets(safe_error(exc))
                raise RuntimeError(f"Milvus filter query failed for {entity_id!r}: {self.client_error}") from exc
        if fragment.op in LOGICAL_OPS and not self._is_milvus_fragment(fragment):
            return super().execute(fragment)
        if entity_id and not self._has_collection(collection):
            return ResultFrame.empty()
        raise NotImplementedError(f"Milvus cannot execute fragment {fragment.op!r}")

    def _lite_near_scan(self, collection: str, fragment: QueryFragment, metric: str) -> ResultFrame:
        """Score Milvus Lite's stored float32 coordinates in binary64."""
        params = fragment.payload
        query = params["vector"]
        scorer = {"COSINE": cosine_score, "L2": euclidean_score, "IP": inner_product_score}[metric]

        def score(row):
            vector = row.get(fragment.field)
            return scorer(vector, query) if vector is not None else None

        filt = fragment.filter_fragment()
        provider_filter = self._expr(filt) if filt is not None and self._is_milvus_fragment(filt) else ""
        remaining_filter = None if provider_filter else filt
        scan = replace(
            fragment,
            payload={**params, "filter": remaining_filter},
            notes={**fragment.notes, "filter": remaining_filter, "row_payload": score, "exact": remaining_filter is None or params.get("exact") is True},
        )
        batch_size = int(self.config.get("heavenbase.query.near.scan_batch_size", default=256))
        self._check_batch_size(batch_size)
        batches = self._key_ordered_batches(
            collection,
            filter_expr=provider_filter,
            batch_size=batch_size,
            json_fields=self._json_fields(fragment.entity_id()),
        )
        with closing(batches):
            return execute_scan(self, scan, rows=chain.from_iterable(batches))

    def to_dict(self) -> dict[str, Any]:
        """Return Milvus backend metadata and connection state.

        Args:
            None.

        Returns:
            dict[str, Any]: Backend metadata, URI, database, and current provider connection
            status.
        """
        return {
            "name": self.name,
            "type": self.type,
            "deployment": "local" if self.identifier == "milvus-lite" else "remote",
            "connected": self.client is not None,
        }


class MilvusLiteBackend(MilvusBackend):
    """Embedded Milvus Lite adapter with file-local workspace isolation."""

    identifier = "milvus-lite"
    # Milvus Lite accepts a search limit above the server's 16384.
    provider_max_top_k = None
    _index_types = ("FLAT",)

    @classmethod
    def from_config(
        cls,
        name: str,
        data: Mapping[str, Any],
        *,
        resolver: ModuleResolver | None = None,
        config: Any | None = None,
    ) -> "MilvusLiteBackend":
        """Build an embedded Milvus Lite backend from a config mapping.

        Args:
            name: Workspace backend instance name.
            data: Backend config mapping. Supports a local file ``uri``, optional
                ``ws_id``, and Milvus client keyword arguments.
            resolver: Context-owned module authority.
            config: Config authority retained by the backend.

        Returns:
            MilvusLiteBackend: Configured embedded Milvus backend instance.

        Raises:
            ValueError: If ``uri`` is not a supported local file URI.
        """
        cfg = dict(data or {})
        ws_id = cfg.pop("ws_id", None)
        return cls(name, cfg.pop("uri", None), ws_id=ws_id, resolver=resolver, config=config, **cfg)

    def __init__(
        self,
        name: str = "milvus-lite",
        uri: str | None = None,
        ws_id: str | None = None,
        *,
        resolver: ModuleResolver | None = None,
        config: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """Create a lazily connected Milvus Lite database backed by one local file.

        Args:
            name: Workspace backend instance name.
            uri: Local Milvus Lite file URI. Defaults to backend configuration.
            ws_id: Optional workspace namespace used to derive an isolated file.
            resolver: Context-owned module authority.
            config: Config authority retained by the backend.
            **kwargs: Extra keyword arguments passed to ``pymilvus.MilvusClient``.

        Returns:
            None: This function does not return a value.

        Raises:
            ValueError: If ``uri`` is not a supported local file URI.
        """
        super().__init__(
            name,
            uri,
            token="",
            ws_id=ws_id,
            resolver=resolver,
            config=config,
            **kwargs,
        )

    def _uri_config_path(self) -> str:
        """Return the embedded provider's local-file config path."""
        return "heavenbase.backend_defaults.milvus_lite.uri"

    def _default_uri(self) -> str:
        """Return the strict data-root-relative default URI."""
        return "file://%/milvus-lite.db"

    def _normalize_uri(self, uri: str) -> str:
        """Resolve the strict local-file URI to a native workspace path."""
        self._base_path = file_uri_to_path(uri, home=self.config.root)
        return self._database_path()

    def _database_path(self) -> str:
        """Return one deterministic file per bound workspace and backend name."""
        if not self.ws_id:
            return self._base_path
        stem = get_file_name(self._base_path, ext=False)
        extension = get_file_ext(self._base_path)
        owner = provider_scope(self.ws_id, self.name, self.identifier, max_length=96)
        return f"{stem}_{owner}.{extension}" if extension else f"{stem}_{owner}.db"

    def _connect(self) -> None:
        """Defer importing and opening pymilvus until provider IO is required."""
        self.client = None
        if not hasattr(self, "client_error"):
            self.client_error = ""
        self._lazy_connect_pending = True

    def _need_client(self) -> None:
        if self.client is None and self._lazy_connect_pending:
            self._lazy_connect_pending = False
            touch_dir(get_file_dir(self.uri))
            MilvusBackend._connect(self)
        super()._need_client()

    def _database_name(self) -> str:
        """Milvus Lite has no database namespace."""
        return ""

    def _workspace_collection_prefix(self) -> str:
        """Avoid duplicate namespace suffixes because the file is workspace-owned."""
        return ""

    def _close_client(self) -> None:
        """Close the embedded client before rebinding or deleting its file."""
        client = self.client
        self.client = None
        close = getattr(client, "close", None)
        if callable(close):
            close()

    def _refresh_workspace(self) -> None:
        """Re-arm the lazy client under the newly bound workspace path."""
        try:
            self._close_client()
        finally:
            self._schemas.clear()
            self._owned_fields.clear()
            self._collections.clear()
            self._dirty_collections.clear()
            self._existing_collections.clear()
            self.uri = self._database_path()
            self.database = ""
            self._lazy_connect_pending = True

    def _supports_array_index(self) -> bool:
        """Milvus Lite does not expose the server scalar-index surface."""
        return False

    def _prepare_iterator_read(self, collection: str) -> None:
        """Seal pending Lite writes before an exhaustive iterator read.

        Milvus Lite can otherwise expose multiple primary-key revisions to a
        query iterator immediately after an upsert.  A flush provides the
        stable, exhaustive snapshot required by ``scan`` and Python fallback
        filtering without adding the same latency to remote point reads.
        """
        self._flush(collection)

    def _list_keys(self, collection: str, filter_expr: str) -> list[str]:
        """Return the sorted primary keys accepted by one scalar expression, after sealing pending writes."""
        self._prepare_iterator_read(collection)
        listed = self.client.query(
            collection_name=collection,
            filter=filter_expr or 'object_id != ""',
            output_fields=["object_id"],
            consistency_level="Strong",
        )
        return sorted({str(self._row_from_milvus(row, ())["object_id"]) for row in listed})

    def _listing_file(self) -> str:
        """Return the resolved database file that keys this backend's open key listings."""
        return os.path.realpath(self.uri)

    def _key_ordered_batches(self, collection: str, *, filter_expr: str, batch_size: int, json_fields: Collection[str] = ()) -> Iterator[list[dict[str, Any]]]:
        """Fetch rows by sorted primary key, because Milvus Lite's query iterator is not key-ordered.

        Milvus Lite 3 answers a query in storage order, so the query iterator's
        ``object_id > last`` cursor skips every row inserted out of key order
        once a collection outgrows one batch. The matching keys (not rows) are
        listed once, sorted, and fetched by key one batch at a time; Lite builds
        the full candidate list for any query anyway. Each fetch applies
        ``filter_expr`` again, because another backend on the same file may
        write between the listing and the fetch: a row it changed so that it
        no longer matches is left out rather than returned with its new values.
        """
        keys = self._list_keys(collection, filter_expr)
        for start in range(0, len(keys), batch_size):
            chunk = keys[start : start + batch_size]
            by_key = self._rows_by_key(collection, chunk, filter_expr, json_fields=json_fields)
            if rows := [by_key[key] for key in chunk if key in by_key]:
                yield rows

    @backend_locked
    def upsert(self, rows: Sequence[RowOp]) -> None:
        """Apply row mutations, then drop this database file's open keyset listings so their next pages list again.

        Args:
            rows (Sequence[RowOp]): Row mutations to apply.

        Returns:
            None: This function does not return a value.
        """
        try:
            super().upsert(rows)
        finally:
            _LITE_KEY_LISTINGS.forget(self._listing_file())

    def refresh(self, entity_id: str | None = None) -> None:
        """Drop the open keyset listings of this backend's database file.

        Writes through HeavenBase drop them already. Call this after writing
        the file through a raw ``pymilvus`` client in this process, so an
        enumeration in progress lists its keys again and sees those rows, or to
        free the listings of enumerations abandoned without ``end_scan``.

        Args:
            entity_id (str | None): Accepted for the workspace refresh protocol; the whole file's listings are dropped.

        Returns:
            None: This function does not return a value.
        """
        del entity_id
        _LITE_KEY_LISTINGS.forget(self._listing_file())

    def end_scan(self, entity_id: str, *, after: Any) -> None:
        """Drop the sorted key listing kept for the page after ``after``.

        Args:
            entity_id (str): Logical entity identifier of the enumeration.
            after (Any): Last ``object_id`` of the enumeration's last page.

        Returns:
            None: This function does not return a value.
        """
        _LITE_KEY_LISTINGS.take(self._listing_file(), self._collection(entity_id), str(after))

    @backend_locked
    def scan_page(self, entity_id: str, *, after: Any, batch_size: int, columns: Sequence[str] | None = None) -> list[dict[str, Any]]:
        """Return one primary-key keyset page from the enumeration's sorted key listing.

        Lite cannot page in key order (see ``_key_ordered_batches``), so the
        first page of an enumeration (``after=None``), or a page whose cursor
        no open listing continues, lists the keys after the cursor once; each
        page then reads its rows by key, and the listing is kept for the page
        that continues from its last key. A cursor inside a page already
        served, as a multi-route merge leaves when it uses a page only partway,
        continues that page's listing too. A full scan therefore lists the keys
        once instead of once per page. A write through any backend on the same
        database file drops that file's listings, so the next page lists again
        from its cursor and sees it. ``end_scan`` or ``refresh`` drops the
        listing of an enumeration that stops early.

        Args:
            entity_id (str): Logical entity identifier.
            after (Any): ``object_id`` the page must follow, or ``None`` for the first page.
            batch_size (int): Positive maximum number of rows in the page.
            columns (Sequence[str] | None): Optional field names to return; ``None`` returns the complete fragment.

        Returns:
            list[dict[str, Any]]: Detached row fragments in ascending ``object_id``
            order, at most ``batch_size`` of them; an empty list ends the scan.

        Raises:
            ValueError: If the page size is not positive.
            RuntimeError: If the provider read fails.
        """
        self._check_batch_size(batch_size)
        selection = self._row_columns(columns)
        self._need_client()
        collection = self._collection(entity_id)
        if not self._has_collection(collection):
            return []
        try:
            path = self._listing_file()
            listing = None if after is None else _LITE_KEY_LISTINGS.resume(path, collection, str(after))
            if listing is None:
                generation = _LITE_KEY_LISTINGS.generation(path)
                listing = (self._list_keys(collection, "" if after is None else f"object_id > {self._lit(after)}"), 0, generation)
            keys, index, generation = listing
            json_fields = self._json_fields(entity_id)
            while index < len(keys):
                chunk = keys[index : index + batch_size]
                index += len(chunk)
                by_key = self._rows_by_key(collection, chunk, json_fields=json_fields)
                if page := [by_key[key] for key in chunk if key in by_key]:
                    _LITE_KEY_LISTINGS.keep(path, collection, str(page[-1]["object_id"]), (keys, index, generation))
                    return [self._project_row(row, selection) for row in page]
        except Exception as exc:
            self.client_error = redact_secrets(safe_error(exc))
            raise RuntimeError(f"Milvus scan_page failed for {entity_id!r}: {self.client_error}") from exc
        return []

    @backend_locked
    def ensure(self, schema: EntitySchema, bindings: Sequence[Any]) -> None:
        """Open the embedded database lazily, then ensure the workspace collection.

        Args:
            schema (EntitySchema): Entity schema to store in Milvus Lite.
            bindings (Sequence[Any]): Storage bindings supplied by the workspace.

        Returns:
            None: This function does not return a value.
        """
        self._need_client()
        super().ensure(schema, bindings)

    @backend_locked
    def drop(self) -> None:
        """Drop every collection in this workspace-owned file, then delete it.

        Args:
            None.

        Returns:
            None: This function does not return a value.

        Raises:
            RuntimeError: If Milvus Lite drop fails.
        """
        try:
            self._need_client()
            for collection in list(self.client.list_collections()):
                self._delete_collection(str(collection))
            self._close_client()
            delete_path(self.uri)
        except Exception as exc:
            self.client_error = redact_secrets(safe_error(exc))
            raise RuntimeError(f"Milvus Lite drop failed for {self.uri!r}: {self.client_error}") from exc
        finally:
            _LITE_KEY_LISTINGS.forget(self._listing_file())
            self._schemas.clear()
            self._owned_fields.clear()
            self._collections.clear()
            self._dirty_collections.clear()
            self._existing_collections.clear()
            self.client = None
            self._lazy_connect_pending = False

    def _live_tag_claims(self) -> dict[str, Any]:
        """Report unopened lazy state separately from a failed provider."""
        availability = "ready" if self.client is not None else ("unknown" if self._lazy_connect_pending else "unavailable")
        return {
            **Backend._live_tag_claims(self),
            "backend-availability": availability,
        }
