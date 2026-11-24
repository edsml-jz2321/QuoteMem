"""RediSearch (Redis Query Engine) backend: scalar + text + native vector search.

Wraps a Redis server that has the RediSearch / Redis Query Engine module
(Redis Stack, or Redis 8 where it is built in). One HASH per object stores the
full row; one FT index per entity provides native scalar filters (``eq/ne/lt/gt/in``
on TAG and numeric fields; substring ``contains``/``match`` and tokenised TEXT fields
fall back to scan) and native vector ``near`` -- including a single ``FT.SEARCH`` that
combines a scalar pre-filter with KNN (``near_filter_mode='combined_native'``).

Rows are stored with a binary Redis client so FLOAT32 vector blobs round-trip
losslessly; ``scan`` / row lookup decode every field back through the entity
schema. Filter and near ``execute`` return only object ids + score from
``FT.SEARCH`` and re-hydrate full rows through the same decode path, so result
fidelity never depends on how the search module serialises returned fields.
"""

from __future__ import annotations

__all__ = [
    "RediSearchBackend",
    "tag_escape",
]

import struct
import time

from heavenbase.utils import Any, Iterable, Mapping, Sequence, dumps_json, loads_json, safe_error

from ...execution.frame.result import ResultFrame
from ...modules import ModuleResolver
from ...schema.model import EntitySchema
from ...types.storage import stores_as_integer
from ...types.vector import _scale_cosine_query
from ..base import Backend, backend_locked
from heavenbase.execution.contracts import LOGICAL_OPS, QueryFragment, RowOp

_SCAN_TYPES = ("array", "json", "hyper-g")  # stored as JSON, never FT-indexed -> Python scan
REDISEARCH_RESERVED_WORDS = frozenset(
    "as dialect dim distance_metric ef_construction ef_runtime epsilon filter flat geo geoshape hnsw "
    "inf initial_cap language limit load m nan nocontent nofields nohighlight noindex nostem nosummarize "
    "numeric params phonetic return schema separator slop sortable sortby spherical summarize tag text timeout "
    "type unf vector weight withcount withcursor withscores withsortkeys withsuffixtrie".split()
) | frozenset({"__score"})


def _pack_vector(values: Sequence[float]) -> bytes:
    """Encode a float sequence as a little-endian FLOAT32 blob for RediSearch."""
    floats = [float(v) for v in values]
    return struct.pack(f"<{len(floats)}f", *floats)


def _unpack_vector(blob: bytes | None) -> list[float] | None:
    """Decode a little-endian FLOAT32 blob back to a list of floats."""
    if not blob:
        return None
    count = len(blob) // 4
    return list(struct.unpack(f"<{count}f", blob[: count * 4]))


# RediSearch TAG metacharacters that must be backslash-escaped to match literally.
_TAG_METACHARS = frozenset(",.<>{}[]\\\"':;!@#$%^&*()-+=~| ")

# A HASH TAG field splits its stored value on this separator. Each scalar field holds
# one whole value, so use the ASCII unit separator (never present in real categorical/
# identifier/date data) instead of the default ',', which would split a value like
# "New York, NY" into two tags and make native eq/in diverge from the scan predicate.
_TAG_SEPARATOR = "\x1f"


def tag_escape(value: Any) -> str:
    """Escape RediSearch TAG metacharacters so a value matches as one literal token.

    Escapes in a single pass so the backslash inserted for one metacharacter is
    never re-escaped by a later one (the cause of the earlier double-escape bug).

    Args:
        value (Any): Value to render as one literal RediSearch TAG token.

    Returns:
        str: Escaped RediSearch TAG token.
    """
    return "".join("\\" + char if char in _TAG_METACHARS else char for char in str(value))


def normalize_metric(metric: str | None) -> str:
    """Normalize a configured metric name to an FT DISTANCE_METRIC (``COSINE``/``L2``/``IP``).

    One source of truth for the index DISTANCE_METRIC, the native distance-to-score
    conversion, and the scan-fallback scorer, so all three agree on the metric.

    Args:
        metric (str | None): Metric alias such as ``cosine``, ``l2``, or ``inner_product``.

    Returns:
        str: RediSearch distance metric identifier.
    """
    value = (metric or "cosine").lower()
    if value in ("l2", "euclidean"):
        return "L2"
    if value in ("ip", "inner", "inner_product"):
        return "IP"
    return "COSINE"


class RediSearchBackend(Backend):
    """RediSearch provider adapter (HASH-per-object + one FT index per entity)."""

    identifier = "redisearch"
    aliases = ("redis_query_engine",)
    provider_reserved_words = REDISEARCH_RESERVED_WORDS

    @classmethod
    def from_config(
        cls,
        name: str,
        data: Mapping[str, Any],
        *,
        resolver: ModuleResolver | None = None,
        config: Any | None = None,
    ) -> "RediSearchBackend":
        """Build the RediSearch backend from a config mapping.

        Args:
            name (str): Workspace backend instance name.
            data (Mapping[str, Any]): Backend config mapping. Supported keys are RediSearch
                constructor parameters, including ``url`` and ``ws_id``.
            resolver: Context-owned module authority.
            config: Config authority retained by the backend.

        Returns:
            RediSearchBackend: Configured RediSearch backend instance.
        """
        cfg = dict(data or {})
        ws_id = cfg.pop("ws_id", None)
        return cls(name, cfg.pop("url", None), ws_id=ws_id, resolver=resolver, config=config, **cfg)

    def __init__(
        self,
        name: str = "redisearch",
        url: str | None = None,
        *,
        ws_id: str | None = None,
        resolver: ModuleResolver | None = None,
        config: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """Create a RediSearch backend.

        Args:
            name (str): Workspace backend instance name.
            url (str | None): Redis URL for the Redis Query Engine server.
            ws_id (str | None): Optional workspace namespace for Redis keys and FT indexes.
            resolver: Context-owned module authority.
            config: Config authority retained by the backend.
            **kwargs (Any): Extra keyword arguments passed to ``redis.Redis.from_url``.

        Returns:
            None: This function does not return a value.
        """
        super().__init__(name, ws_id=ws_id, resolver=resolver, config=config)
        self.url = url or self._construction_config_get("heavenbase.backend_defaults.redisearch.url", default="redis://localhost:6379")
        self._client_kwargs = dict(kwargs)
        self._schemas: dict[str, EntitySchema] = {}
        self._indices: dict[str, str] = {}
        self._ensured: set[str] = set()
        self._connect()

    # ------------------------------------------------------------------ wiring
    def _namespace(self) -> str:
        return self.ws_id or "default"

    def _connect(self) -> None:
        try:
            import redis

            # Binary client for HASH row IO (FLOAT32 vector blobs must round-trip).
            self.client = redis.Redis.from_url(self.url, decode_responses=False, **self._client_kwargs)
            # Decoded client for FT.* — redis-py's search result parser returns no
            # docs on a bytes (decode_responses=False) client.
            self.text_client = redis.Redis.from_url(self.url, decode_responses=True, **self._client_kwargs)
            self.client.ping()
            self.client_error = ""
        except Exception as exc:  # provider-unavailable: degrade, raise lazily in _need_client
            self.client = None
            self.text_client = None
            self.client_error = safe_error(exc)

    def _need_client(self) -> None:
        if self.client is None:
            detail = f": {self.client_error}" if self.client_error else ""
            raise RuntimeError(f"RediSearch backend {self.name!r} is unavailable{detail}")

    def _refresh_workspace(self) -> None:
        self._schemas.clear()
        self._indices.clear()
        self._ensured.clear()
        self._connect()

    # ------------------------------------------------------------------ naming
    def _prefix(self, entity_id: str) -> str:
        # Namespace row keys by the backend instance name so a redis + a redisearch backend
        # (or two redisearch instances) sharing one server and workspace never collide.
        return f"{self._namespace()}:{self.name}:{entity_id}:"

    def _key(self, entity_id: str, object_id: Any) -> str:
        return f"{self._prefix(entity_id)}{object_id}"

    def _index(self, entity_id: str) -> str:
        return self._indices.setdefault(entity_id, f"idx:{self._namespace()}:{self.name}:{entity_id}")

    def schema_warnings(self, schema: EntitySchema, bindings: Sequence[Any]) -> list[str]:
        """Return RediSearch field-name warnings without rewriting user schema names.

        Args:
            schema (EntitySchema): Entity schema being registered.
            bindings (Sequence[Any]): Storage bindings for the entity.

        Returns:
            list[str]: Provider naming warnings for RediSearch field names.
        """
        return self._physical_name_warnings(
            [(field.name, field.name) for field in schema.fields.values()],
            kind=f"RediSearch field for {schema.entity_id!r}",
            reserved=self.provider_reserved_words,
            case_sensitive=False,
        )

    def _has_index(self, entity_id: str) -> bool:
        index = self._index(entity_id)
        if index in self._ensured:
            return True
        try:
            self.text_client.ft(index).info()
            self._ensured.add(index)
            return True
        except Exception as exc:
            # Only a genuinely-absent index means "no index"; a connection/timeout
            # error must propagate rather than silently degrade a query to empty.
            message = str(exc).lower()
            if "unknown index" in message or "no such index" in message:
                return False
            self.client_error = safe_error(exc)
            raise RuntimeError(f"RediSearch index probe failed for {entity_id!r}: {self.client_error}") from exc

    # ------------------------------------------------------------ field typing
    def _vector_fields(self, schema: EntitySchema) -> list[Any]:
        return [f for f in schema.fields.values() if type(f.dtype).identifier == "vector"]

    def _index_field_specs(self, index: str) -> dict[str, dict[str, Any]]:
        """Return ``{field_name: {type, dim, metric, separator}}`` for an existing FT index."""
        info = self.text_client.ft(index).info()
        attributes = info.get("attributes", []) if isinstance(info, dict) else []
        specs: dict[str, dict[str, Any]] = {}
        for attr in attributes:
            if not isinstance(attr, (list, tuple)):
                continue
            kv = {attr[i]: attr[i + 1] for i in range(0, len(attr) - 1, 2)}
            name = kv.get("identifier") or kv.get("attribute")
            if name is None:
                continue
            ftype = str(kv.get("type", "")).upper()
            spec: dict = {"type": ftype}
            if ftype == "VECTOR":
                try:
                    spec["dim"] = int(kv.get("dim"))
                    spec["metric"] = str(kv.get("distance_metric", "")).upper()
                except (TypeError, ValueError):
                    continue  # unexpected FT.INFO shape -> skip, never raise a false mismatch
            elif ftype == "TAG":
                spec["separator"] = kv.get("SEPARATOR")
            specs[str(name)] = spec
        return specs

    def _ft_type(self, field_schema: Any) -> str | None:
        """The FT attribute type a field is indexed as, or None when store-only (mirrors ``_ft_field``).

        Returns the ``FT.INFO`` type name (``"TAG"`` / ``"NUMERIC"`` / ``"TEXT"`` / ``"VECTOR"``) so it
        can be compared directly against an existing index spec, or None for array / json / hyper-g
        fields that are stored only and answered by Python scan.
        """
        if field_schema.name == "object_id":
            return "TAG"
        name = type(field_schema.dtype).identifier
        if name == "vector":
            return "VECTOR"
        if name == "float" or stores_as_integer(field_schema.dtype):
            return "NUMERIC"
        if name in ("long-text", "medium-text"):
            return "TEXT"
        if name in ("identifier", "short-text", "categorical", "boolean", "date"):
            return "TAG"
        return None

    def _index_needs_rebuild(self, schema: EntitySchema, index: str) -> bool:
        """Whether an existing FT index has a recoverable drift (type/metric/separator/added field) a rebuild fixes.

        Returns True when the index can be dropped and recreated in place (documents kept) to
        match the schema — a changed FT attribute type, distance metric, or TAG separator, or an
        indexable field added to the schema after the index was built, is re-indexed from the kept
        documents. Raises on an irrecoverable vector-dim drift, where the stored FLOAT32 blobs are a
        different length and the data itself must be rebuilt.
        """
        existing = self._index_field_specs(index)
        if not existing:
            return False
        want_metric = self._index_metric()
        rebuild = False
        for field_schema in schema.fields.values():
            want_type = self._ft_type(field_schema)
            spec = existing.get(field_schema.name)
            if spec is None:
                # an indexable field absent from the existing index is recoverable drift (e.g. a field
                # added to the schema after the index was built): drop+recreate re-indexes the kept
                # documents with the new attribute. store-only fields are absent by design, not drift.
                if want_type is not None:
                    rebuild = True
                continue
            if want_type is None:
                continue  # field is now store-only; a leftover index attribute is harmless (never queried natively)
            if want_type != spec.get("type"):
                rebuild = True  # field changed FT type (e.g. TAG <-> NUMERIC): recreate so the index matches the schema
                continue
            if want_type == "VECTOR" and "dim" in spec:
                if spec["dim"] != int(field_schema.dtype.dim):
                    raise RuntimeError(
                        f"RediSearch backend {self.name!r}: existing index {index!r} for {schema.entity_id!r} has "
                        f"vector field {field_schema.name!r} dim={spec['dim']}, but the schema requires "
                        f"dim={int(field_schema.dtype.dim)}. The stored vectors are a different length — "
                        f"drop the workspace and re-embed to rebuild."
                    )
                if spec.get("metric") != want_metric:
                    rebuild = True
            elif want_type == "TAG" and spec.get("separator") != _TAG_SEPARATOR:
                rebuild = True
        return rebuild

    def _wait_for_reindex(self, index: str, expected_docs: int) -> None:
        """Wait until a rebuilt index has consumed every retained HASH row."""
        if expected_docs < 1:
            return
        timeout = float(self.config.get("heavenbase.backend_defaults.redisearch.index_timeout", default=5.0))
        deadline = time.monotonic() + timeout
        while True:
            info = self.text_client.ft(index).info()
            indexed = int(float(info.get("num_docs", 0)))
            indexing = bool(int(float(info.get("indexing", 0))))
            if indexed >= expected_docs and not indexing:
                return
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"RediSearch backend {self.name!r}: rebuilt index {index!r} indexed "
                    f"{indexed}/{expected_docs} retained rows before the {timeout:g}s timeout"
                )
            time.sleep(0.01)

    def _ft_field(self, field_schema: Any, metric: str) -> Any:
        """Return a redis-py search Field for one logical field, or None to store-only (scan)."""
        import redis
        from redis.commands.search.field import NumericField, TagField, TextField, VectorField

        name = type(field_schema.dtype).identifier
        if field_schema.name == "object_id":
            # case_sensitive: TAG fields case-fold values at index time by default,
            # which would make native tag/prefix matching diverge from the
            # case-sensitive Python scan predicate (eq/ne/in/starts_with).
            return TagField("object_id", separator=_TAG_SEPARATOR, case_sensitive=True)
        if name == "vector":
            return VectorField(
                field_schema.name,
                str(self.config.get("heavenbase.backend_defaults.redisearch.vector_algorithm", default="HNSW")),
                {"TYPE": "FLOAT32", "DIM": int(field_schema.dtype.dim), "DISTANCE_METRIC": metric},
            )
        if name == "float" or stores_as_integer(field_schema.dtype):
            return NumericField(field_schema.name)
        if name in ("long-text", "medium-text"):
            return TextField(field_schema.name)
        if name in ("identifier", "short-text", "categorical", "boolean", "date"):
            return TagField(field_schema.name, separator=_TAG_SEPARATOR, case_sensitive=True)  # byte-exact whole-value tag
        return None  # array / json / hyper-g -> stored only, Python scan fallback

    # --------------------------------------------------------------- row codec
    def _encode_row(self, schema: EntitySchema, object_id: Any, values: dict[str, Any]) -> dict[bytes, bytes]:
        mapping: dict[bytes, bytes] = {b"object_id": str(object_id).encode()}
        for name, value in values.items():
            field_schema = schema.fields.get(name)
            if field_schema is None or value is None:
                continue
            ident = type(field_schema.dtype).identifier
            if ident == "vector":
                mapping[name.encode()] = _pack_vector(value)
            elif ident in _SCAN_TYPES or isinstance(value, (list, dict)):
                mapping[name.encode()] = dumps_json(value).encode()
            elif ident == "boolean":
                mapping[name.encode()] = b"1" if value else b"0"
            elif isinstance(value, (bytes, bytearray)):
                mapping[name.encode()] = bytes(value)  # Artifact/BLOB: Redis HASH values are binary-safe
            else:
                mapping[name.encode()] = str(value).encode()
        return mapping

    def _decode_row(self, schema: EntitySchema, raw: dict[Any, Any]) -> dict[str, Any]:
        row: dict[str, Any] = {}
        for raw_key, raw_value in raw.items():
            key = raw_key.decode() if isinstance(raw_key, (bytes, bytearray)) else str(raw_key)
            if key == "object_id":
                row["object_id"] = raw_value.decode() if isinstance(raw_value, (bytes, bytearray)) else raw_value
                continue
            field_schema = schema.fields.get(key)
            if field_schema is None:
                row[key] = raw_value.decode("utf-8", "replace") if isinstance(raw_value, (bytes, bytearray)) else raw_value
                continue
            ident = type(field_schema.dtype).identifier
            if ident == "vector":
                row[key] = _unpack_vector(raw_value)
            elif ident == "artifact":
                row[key] = bytes(raw_value) if isinstance(raw_value, (bytes, bytearray)) else raw_value  # raw BLOB bytes
            elif ident in _SCAN_TYPES:
                row[key] = loads_json(raw_value.decode()) if raw_value else None
            elif ident == "float":
                row[key] = float(raw_value) if raw_value not in (b"", None) else None
            elif stores_as_integer(field_schema.dtype):
                row[key] = int(raw_value) if raw_value not in (b"", None) else None
            elif ident == "boolean":
                row[key] = raw_value in (b"1", b"true", b"True")
            else:
                row[key] = raw_value.decode() if isinstance(raw_value, (bytes, bytearray)) else raw_value
        return row

    # --------------------------------------------------------------- lifecycle
    @backend_locked
    def ensure(self, schema: EntitySchema, bindings: Sequence[Any]) -> None:
        """Create the FT index for an entity, or rebuild it in place on a recoverable drift.

        Idempotent via FT.INFO. If the existing index has a recoverable mismatch (changed
        distance metric or TAG separator) it is dropped — keeping the documents — and recreated
        so FT re-indexes the existing rows with the new config; a vector-dim change raises instead.

        Args:
            schema (EntitySchema): Entity schema to store and index in RediSearch.
            bindings (Sequence[Any]): Storage bindings supplied by the workspace.

        Returns:
            None: This function does not return a value.

        Raises:
            RuntimeError: If the server lacks RediSearch, index rebuild fails, or a vector
                dimension drift requires data rebuild.
        """
        self._schemas[schema.entity_id] = schema
        if self.client is None:
            return
        index = self._index(schema.entity_id)
        rebuilding = False
        retained_docs = 0
        if self._has_index(schema.entity_id):
            if not self._index_needs_rebuild(schema, index):
                return
            try:
                self.text_client.ft(index).dropindex(delete_documents=False)  # keep rows; recreate re-indexes them
            except Exception as exc:
                # Only a genuinely-absent index is benign here; a real connection/permission error
                # must surface, or the stale index survives and the create below silently no-ops on
                # "Index already exists", leaving the drifted native config active.
                message = str(exc).lower()
                if "unknown index" not in message and "no such index" not in message:
                    self.client_error = safe_error(exc)
                    raise RuntimeError(
                        f"RediSearch backend {self.name!r}: could not drop stale index {index!r} for "
                        f"{schema.entity_id!r} to rebuild it: {self.client_error}"
                    ) from exc
            self._ensured.discard(index)
            rebuilding = True
            retained_docs = sum(1 for _ in self.client.scan_iter(match=f"{self._prefix(schema.entity_id)}*"))
        import redis
        from redis.commands.search.index_definition import IndexDefinition, IndexType

        metric = self._index_metric()
        fields = [spec for spec in (self._ft_field(f, metric) for f in schema.fields.values()) if spec is not None]
        try:
            self.text_client.ft(index).create_index(
                fields,
                definition=IndexDefinition(prefix=[self._prefix(schema.entity_id)], index_type=IndexType.HASH),
            )
            if rebuilding:
                self._wait_for_reindex(index, retained_docs)
            self._ensured.add(index)
        except Exception as exc:
            message = str(exc)
            if "Index already exists" in message and not rebuilding:
                self._ensured.add(index)  # created concurrently between the probe and create -> benign
                return
            self.client_error = safe_error(exc)
            if "Index already exists" in message:
                raise RuntimeError(
                    f"RediSearch backend {self.name!r}: stale index {index!r} for {schema.entity_id!r} still "
                    f"exists after the rebuild drop, so its drifted config remains active: {self.client_error}"
                ) from exc
            if "unknown command" in message.lower() or "FT.CREATE" in message:
                raise RuntimeError(
                    f"RediSearch backend {self.name!r}: server lacks the Redis Query Engine module "
                    f"(need Redis Stack or Redis 8). Detail: {self.client_error}"
                ) from exc
            raise RuntimeError(f"RediSearch ensure failed for {schema.entity_id!r}: {self.client_error}") from exc

    @backend_locked
    def drop(self) -> None:
        """Drop FT indexes (with docs) and remaining keys for this workspace+instance namespace.

        Indexes are discovered from the server (``FT._LIST``) by the ``idx:{ns}:{name}:`` prefix,
        not just from in-memory state, so an index created in an earlier process is not orphaned.
        A missing index is ignored, but any other dropindex error is surfaced after the remaining
        cleanup so a real failure is not silently swallowed.

        Args:
            None.

        Returns:
            None: This function does not return a value.

        Raises:
            RuntimeError: If RediSearch reports a real drop failure.
        """
        if self.client is None:
            return
        first_error = None
        try:
            for index in self._owned_indices():
                try:
                    self.text_client.ft(index).dropindex(delete_documents=True)
                except Exception as exc:
                    message = str(exc).lower()
                    if "unknown index" not in message and "no such index" not in message and first_error is None:
                        first_error = exc
            for key in self.client.scan_iter(match=f"{self._namespace()}:{self.name}:*"):
                self.client.delete(key)
        finally:
            self._schemas.clear()
            self._indices.clear()
            self._ensured.clear()
        if first_error is not None:
            self.client_error = safe_error(first_error)
            raise RuntimeError(f"RediSearch drop failed: {self.client_error}") from first_error

    def _owned_indices(self) -> list[str]:
        """Return this workspace+instance's FT index names, discovered from the server.

        Merges in-memory ``_indices`` with an ``FT._LIST`` scan filtered by the ``idx:{ns}:{name}:``
        prefix, so an index created in an earlier process (absent from memory) is still dropped.
        FT._LIST discovery is best-effort: if it errors, the in-memory set is used and a real
        connection failure resurfaces from the dropindex/scan calls in ``drop``.
        """
        prefix = f"idx:{self._namespace()}:{self.name}:"
        owned = {self._index(entity_id) for entity_id in self._schemas} | set(self._indices.values())
        try:
            listed = self.text_client.execute_command("FT._LIST")
        except Exception:
            return sorted(owned)
        for name in listed or []:
            text = name.decode() if isinstance(name, (bytes, bytearray)) else str(name)
            if text.startswith(prefix):
                owned.add(text)
        return sorted(owned)

    # ------------------------------------------------------------------- write
    @backend_locked
    def upsert(self, rows: Sequence[RowOp]) -> None:
        """Write HASH-per-object rows; FT auto-indexes keys under the entity prefix.

        Each adjacent same-kind row batch uses a separate ``MULTI/EXEC``. Redis
        runtime command errors can partially execute one transaction and continue
        later queued commands, so phase separation ensures a failed source rewrite
        cannot reach a later target delete. Within a phase, per-key ``DELETE + HSET``
        full-row replacement remains isolated from other clients.

        Args:
            rows (Sequence[RowOp]): Row mutations to apply. ``delete`` operations remove
                hashes; other operations replace the stored row hash.

        Returns:
            None: This function does not return a value.

        Raises:
            RuntimeError: If the Redis client is unavailable or the entity was not ensured.
        """
        self._need_client()
        rows = list(rows)
        for row in rows:
            if row.entity_id not in self._schemas:
                raise RuntimeError(f"RediSearch backend {self.name!r}: ensure() not called for {row.entity_id!r}")
        for _entity_id, _is_delete, batch in self.ordered_row_op_batches(rows):
            pipe = self.client.pipeline(transaction=True)
            for row in batch:
                key = self._key(row.entity_id, row.object_id)
                if row.is_delete():
                    pipe.delete(key)
                    continue
                mapping = self._encode_row(self._schemas[row.entity_id], row.object_id, row.values)
                pipe.delete(key)  # full-row replace semantics (drop stale fields)
                if mapping:
                    pipe.hset(key, mapping=mapping)
            pipe.execute()

    # -------------------------------------------------------------------- read
    def _fetch_rows(self, entity_id: str, object_ids: Sequence[Any]) -> list[dict[str, Any] | None]:
        schema = self._schemas.get(entity_id)
        if schema is None:
            return [None for _ in object_ids]
        pipe = self.client.pipeline(transaction=False)
        for object_id in object_ids:
            pipe.hgetall(self._key(entity_id, object_id))
        return [self._decode_row(schema, raw) if raw else None for raw in pipe.execute()]

    @backend_locked
    def scan(self, entity_id: str) -> list[dict[str, Any]]:
        """Return every row for one entity by scanning the key prefix.

        Args:
            entity_id (str): Entity identifier whose Redis hash namespace should be scanned.

        Returns:
            list[dict[str, Any]]: Decoded row dictionaries.

        Raises:
            RuntimeError: If the Redis client is unavailable.
        """
        self._need_client()
        schema = self._schemas.get(entity_id)
        if schema is None:
            return []
        rows: list[dict[str, Any]] = []
        for key in self.client.scan_iter(match=f"{self._prefix(entity_id)}*"):
            raw = self.client.hgetall(key)
            if raw:
                rows.append(self._decode_row(schema, raw))
        return rows

    @backend_locked
    def get_many(self, entity_id: str, object_ids: Iterable[Any], *, columns: Sequence[str] | None = None) -> list[dict[str, Any] | None]:
        """Return rows via direct ``HGETALL`` key lookup.

        Args:
            entity_id (str): Entity identifier whose rows should be loaded.
            object_ids (Iterable[Any]): Object identifiers to fetch in caller order.
            columns (Sequence[str] | None): Optional field names to return; ``None`` returns the complete fragment.

        Returns:
            list[dict[str, Any] | None]: Rows aligned with ``object_ids``. Missing hashes
            are returned as ``None``.

        Raises:
            RuntimeError: If the Redis client is unavailable.
        """
        self._need_client()
        return self._project_rows(self._fetch_rows(entity_id, list(object_ids)), columns)

    @backend_locked
    def count(self, entity_id: str) -> int:
        """Return row count via FT.INFO num_docs, falling back to a key scan.

        Only an unexpected FT.INFO *shape* (missing/garbled num_docs) drops to the key scan; a
        real connection/permission error from FT.INFO propagates loudly instead of being masked
        as a scan, matching ``_has_index`` and the rest of this backend.

        Args:
            entity_id (str): Entity identifier whose indexed rows should be counted.

        Returns:
            int: Number of rows for the entity.

        Raises:
            RuntimeError: If the Redis client is unavailable or index probing fails.
        """
        self._need_client()
        if self._has_index(entity_id):
            info = self.text_client.ft(self._index(entity_id)).info()
            try:
                num = info.get("num_docs") if isinstance(info, dict) else info[info.index(b"num_docs") + 1]
                return int(num)
            except (KeyError, ValueError, TypeError, IndexError):
                pass
        return sum(1 for _ in self.client.scan_iter(match=f"{self._prefix(entity_id)}*"))

    def _live_tag_claims(self) -> dict[str, Any]:
        """Return live tag claims derived from RediSearch availability.

        Args:
            None.

        Returns:
            dict[str, Any]: Live Backend tag claims.
        """
        return {
            **super()._live_tag_claims(),
            "backend-availability": "ready" if self.client is not None else "unavailable",
        }

    # --------------------------------------------------------- native compile
    def candidate_id_fragment(self, object_ids: Sequence[Any]) -> QueryFragment:
        """Return a native FT ``object_id`` TAG filter for candidate prefiltering.

        Args:
            object_ids (Sequence[Any]): Candidate object identifiers.

        Returns:
            QueryFragment: Native RediSearch object-id filter fragment.
        """
        py_fragment = super().candidate_id_fragment(object_ids)
        members = "|".join(tag_escape(item) for item in object_ids) or "__none__"
        return QueryFragment(
            self.name,
            "in",
            "object_id",
            f"@object_id:{{{members}}}",
            notes={**py_fragment.notes, "kind": "redis_filter", "mode": "native"},
        )

    def _is_rs_fragment(self, fragment: QueryFragment) -> bool:
        if fragment.kind() == "redis_filter":
            return True
        if fragment.op in ("and", "or"):
            return all(self._is_rs_fragment(child) for child in fragment.children())
        if fragment.op == "not":
            return self._is_rs_fragment(fragment.children()[0])
        return False

    def _expr(self, fragment: QueryFragment) -> str:
        if fragment.kind() == "redis_filter":
            return fragment.payload
        if fragment.op == "and":
            return " ".join(f"({self._expr(child)})" for child in fragment.children())  # FT default conjunction
        if fragment.op == "or":
            return "(" + " | ".join(f"({self._expr(child)})" for child in fragment.children()) + ")"
        if fragment.op == "not":
            return f"-({self._expr(fragment.children()[0])})"
        raise TypeError(f"Cannot compile {fragment.op!r} to a RediSearch query")

    def _is_negative(self, fragment: QueryFragment) -> bool:
        """Whether the fragment renders to a purely-negative FT query needing a ``*`` scope.

        A RediSearch query that is only negation (a bare ``ne`` -> ``-@f:{v}``, a top-level
        ``not``, or an ``or`` with a negative branch) has no positive set to subtract from and
        returns ZERO documents, diverging from the portable scan predicate (which returns the
        complement). Such expressions must be scoped against the full keyspace so the negation
        resolves to the complement.
        """
        if fragment.kind() == "redis_filter":
            return str(fragment.payload).lstrip().startswith("-")
        if fragment.op == "not":
            return True
        if fragment.op == "and":
            return all(self._is_negative(child) for child in fragment.children())  # a positive child anchors the AND
        if fragment.op == "or":
            return any(self._is_negative(child) for child in fragment.children())  # a negative branch can't resolve alone
        return False

    def _scoped_expr(self, fragment: QueryFragment) -> str:
        """Render a fragment to an FT query, prefixing ``*`` when it is purely negative."""
        expr = self._expr(fragment)
        return f"* {expr}" if self._is_negative(fragment) else expr

    def _metric(self, metric: str) -> str:
        return normalize_metric(metric)

    def _index_metric(self) -> str:
        """Normalized FT DISTANCE_METRIC from config, which a new or rebuilt index is built with.

        RediSearch KNN always scores with the metric the index was built with, so
        the distance-to-score conversion uses ``near_metric``, which reads it from
        an existing index, not this value or the per-query metric.
        """
        return self._metric(str(self.config.get("heavenbase.backend_defaults.redisearch.metric", default="cosine")))

    def near_metric(self, entity_id: str, field: str) -> str:
        """Return the FT DISTANCE_METRIC a native KNN on one entity's vector field ranks by.

        An FT index keeps the metric it was built with until ``ensure()`` rebuilds it, so a later
        change of ``heavenbase.backend_defaults.redisearch.metric`` alone does not change how an
        existing index ranks. Before the index exists (no client, an entity not stored here, or no
        index yet) the configured metric, which a new index gets, is returned.

        Args:
            entity_id (str): Logical entity identifier.
            field (str): Vector field name.

        Returns:
            str: ``"COSINE"``, ``"L2"`` or ``"IP"``.

        Raises:
            RuntimeError: If the provider cannot describe an existing index.
        """
        return self._near_index(entity_id, field)[0]

    @backend_locked
    def _near_index(self, entity_id: str, field: str) -> tuple[str, bool | None]:
        """Read the metric and existence from one FT.INFO request."""
        configured = self._index_metric()
        if self.client is None or entity_id not in self._schemas:
            return configured, None
        index = self._index(entity_id)
        try:
            specs = self._index_field_specs(index)
        except Exception as exc:
            message = str(exc).lower()
            if "unknown index" in message or "no such index" in message:
                self._ensured.discard(index)
                return configured, False
            self.client_error = safe_error(exc)
            raise RuntimeError(f"RediSearch index lookup failed for {entity_id!r}: {self.client_error}") from exc
        self._ensured.add(index)
        metric = specs.get(field, {}).get("metric")
        return normalize_metric(metric) if metric else configured, True

    def _score(self, metric: str, distance: float) -> float:
        """Convert a RediSearch distance to a similarity score (higher == closer).

        COSINE distance is ``1 - cosine_similarity``, so ``1 - distance`` recovers
        the cosine similarity and matches ``cosine_score`` used by the scan
        fallback (native and ``combined_scan`` results stay on one scale). L2/IP
        distances have no normalised similarity; ``-distance`` preserves ordering.
        """
        return 1.0 - distance if metric == "COSINE" else -distance

    def _doc_object_id(self, doc: Any, entity_id: str) -> str:
        key = doc.id.decode() if isinstance(doc.id, (bytes, bytearray)) else str(doc.id)
        prefix = self._prefix(entity_id)
        return key[len(prefix) :] if key.startswith(prefix) else key

    def _doc_distance(self, doc: Any) -> float:
        for attr in ("__score", "score"):
            raw = getattr(doc, attr, None)
            if raw is not None:
                try:
                    return float(raw)
                except (TypeError, ValueError):
                    continue
        return 0.0

    # ----------------------------------------------------------------- execute
    def _run_near(self, entity_id: str, index: str, fragment: QueryFragment) -> ResultFrame:
        import redis
        from redis.commands.search.query import Query

        params = fragment.payload
        filt = fragment.filter_fragment()
        if filt is not None and not self._is_rs_fragment(filt):
            # combined_scan: a non-native filter leaf -> apply filter + score row-by-row.
            return super().execute(fragment)
        top_k = int(params["top_k"])
        base = self._scoped_expr(filt) if filt is not None else "*"
        query_str = f"({base})=>[KNN {top_k} @{fragment.field} $vec AS __score]"
        query = Query(query_str).return_fields("__score").sort_by("__score", asc=True).paging(0, top_k).dialect(2)
        # KNN ranks by the index's own metric, which the handler read when it chose the native route.
        metric = params.get("index_metric") or self.near_metric(entity_id, fragment.field)
        vector = params["vector"]
        if metric == "COSINE":
            # The query is packed as FLOAT32; cosine ignores the exact power-of-two scale.
            vector = _scale_cosine_query(vector)
        try:
            result = self.text_client.ft(index).search(query, query_params={"vec": _pack_vector(vector)})
        except Exception as exc:
            self.client_error = safe_error(exc)
            raise RuntimeError(f"RediSearch near failed for {entity_id!r}: {self.client_error}") from exc
        scored = [(self._doc_object_id(doc, entity_id), self._score(metric, self._doc_distance(doc))) for doc in result.docs]
        by_id = {row["object_id"]: row for row in self._fetch_rows(entity_id, [oid for oid, _ in scored]) if row}
        rows: list[dict[str, Any]] = []
        for object_id, score in scored:
            row = by_id.get(object_id)
            if row is not None:
                row = dict(row)
                row["score"] = score
                rows.append(row)
        return ResultFrame.from_rows(rows)

    def _run_filter(self, entity_id: str, index: str, fragment: QueryFragment) -> ResultFrame:
        import redis
        from redis.commands.search.query import Query

        # filter_limit is the FT.SEARCH PAGE size, not a hard cap: page through every match so a
        # filter matching more than one page is not silently truncated before the public query's
        # own limit/offset runs.
        page = int(self.config.get("heavenbase.backend_defaults.redisearch.filter_limit", default=10000))
        expr = self._scoped_expr(fragment)
        ids: list[str] = []
        offset = 0
        while True:
            query = Query(expr).return_fields("object_id").paging(offset, page).dialect(2)
            try:
                result = self.text_client.ft(index).search(query)
            except Exception as exc:
                self.client_error = safe_error(exc)
                raise RuntimeError(f"RediSearch filter failed for {entity_id!r}: {self.client_error}") from exc
            ids.extend(self._doc_object_id(doc, entity_id) for doc in result.docs)
            offset += page
            if not result.docs or offset >= int(result.total):
                break
        return ResultFrame.from_rows([row for row in self._fetch_rows(entity_id, ids) if row])

    @backend_locked
    def execute(self, fragment: QueryFragment) -> ResultFrame:
        """Run native FT search and filter fragments.

        Args:
            fragment (QueryFragment): Backend-ready query fragment compiled by the handler layer.

        Returns:
            ResultFrame: Matching rows and object identifiers.

        Raises:
            NotImplementedError: If the fragment cannot be executed natively or by scan fallback.
            RuntimeError: If the Redis client is unavailable or RediSearch execution fails.
        """
        entity_id = fragment.entity_id()
        self._need_client()
        index = self._index(entity_id) if entity_id else ""
        kind = fragment.kind()
        if fragment.op == "near" and fragment.mode() == "py_scan":
            # A near whose metric is not the index metric: KNN would rank by the index metric.
            return super().execute(fragment)
        if entity_id and kind == "redis_search" and fragment.op == "near":
            have = fragment.payload.get("index_exists")
            if have is None:
                have = self._has_index(entity_id)
            return self._run_near(entity_id, index, fragment) if have else ResultFrame.empty()
        have = self._has_index(entity_id) if entity_id else False
        if (kind == "redis_filter" or fragment.op in LOGICAL_OPS) and have and self._is_rs_fragment(fragment):
            return self._run_filter(entity_id, index, fragment)
        if fragment.mode() == "py_scan" or (fragment.op in LOGICAL_OPS and not self._is_rs_fragment(fragment)):
            return super().execute(fragment)  # logical fragments mixing native + scan-only children degrade here
        if entity_id and not have:
            return ResultFrame.empty()
        raise NotImplementedError(f"RediSearch cannot execute fragment {fragment.op!r}")

    def to_dict(self) -> dict[str, Any]:
        """Return RediSearch backend metadata and connection state.

        Args:
            None.

        Returns:
            dict[str, Any]: Backend metadata, Redis URL, namespace, and current provider
            connection status.
        """
        return {
            "name": self.name,
            "type": self.type,
            "url": self.url,
            "namespace": self._namespace(),
            "connected": self.client is not None,
            "error": self.client_error,
        }
