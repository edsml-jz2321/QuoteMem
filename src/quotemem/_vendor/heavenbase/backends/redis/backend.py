"""Redis key/value backend: one JSON document per key, with eq/in value indexes.

Rows are stored as one JSON document per namespaced key. Plain Redis has no
query engine, so non-equality operators run through the inherited scan fallback.
For ``eq``/``in`` the backend maintains server-side ``SET`` indexes
(``{ns}:{name}:_idx:{entity}:{field}:{index_hash(value)}`` -> object_ids); those operators
then resolve via ``SMEMBERS``/``SUNION`` (AND -> intersection, OR -> union) instead of
scanning the keyspace.

Index consistency assumes a single writer per ``(workspace, instance name)``: per-instance
writes are serialized by ``@backend_locked``, ``upsert`` reads the previous value before its
``MULTI/EXEC``, and ``ensure`` backfills the index for rows written before a field became
indexable (or before this process started). ``execute`` also re-applies the portable
``row_payload`` predicate to every fetched row, so a stale index member can never yield a
*false positive*. It cannot, however, recover a *false negative*: a true match missing from
the SET index — only reachable under concurrent cross-process writers — is never fetched. Run
one writer per instance, or use the redisearch backend, when writers share a server.
"""

from __future__ import annotations

__all__ = [
    "RedisBackend",
]

from heavenbase.utils import Any, Iterable, Mapping, Sequence, dumps_json, fmt_hash, loads_json, resolve_hash_int, safe_error

from ...execution.frame.result import ResultFrame
from ...modules import ModuleResolver
from ...schema.model import EntitySchema
from ..base import Backend, backend_locked
from heavenbase.execution.contracts import QueryFragment, RowOp

_DEFAULT_INDEX_HASH_FUNCTION = "crc32"
_DEFAULT_INDEX_HASH_LENGTH = 10


class RedisBackend(Backend):
    """Redis provider adapter storing one JSON document per object."""

    identifier = "redis"

    @classmethod
    def from_config(cls, name: str, data: Mapping[str, Any], *, resolver: ModuleResolver | None = None, config: Any | None = None) -> "RedisBackend":
        """Build the Redis key/value backend from a config mapping.

        Args:
            name (str): Workspace backend instance name.
            data (Mapping[str, Any]): Backend config mapping. Supported keys are the Redis
                constructor parameters, including ``url`` and ``ws_id``.
            resolver: Context-owned module authority.
            config: Config authority retained by the backend.

        Returns:
            RedisBackend: Configured Redis backend instance.
        """
        cfg = dict(data or {})
        ws_id = cfg.pop("ws_id", None)
        return cls(name, cfg.pop("url", None), ws_id=ws_id, resolver=resolver, config=config, **cfg)

    def __init__(
        self,
        name: str = "redis",
        url: str | None = None,
        *,
        ws_id: str | None = None,
        index_hash_function: str | None = None,
        index_hash_length: int | None = None,
        resolver: ModuleResolver | None = None,
        config: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """Create a Redis key/value backend.

        Args:
            name: Workspace backend instance name.
            url: Redis connection URL. Defaults to config.
            ws_id: Optional workspace namespace for Redis keys.
            index_hash_function: Integer hash method for value-index keys. Defaults to config.
            index_hash_length: Minimum decimal digest length for value-index keys. Defaults to config.
            resolver: Context-owned module authority.
            config: Config authority retained by the backend.
            **kwargs: Extra keyword arguments passed to ``redis.Redis.from_url``.

        Raises:
            ValueError: If ``index_hash_length`` is not positive.
            ImportError: If the configured hash method requires an unavailable optional package.

        Returns:
            None: This function does not return a value.

        Note:
            Changing the index hash method or length changes Redis SET index keys; drop or rebuild
            stale index keys after changing either setting for an existing namespace.
        """
        super().__init__(name, ws_id=ws_id, resolver=resolver, config=config)
        self.url = url or self._construction_config_get("heavenbase.backend_defaults.redis.url", default="redis://localhost:6379")
        configured_hash = (
            index_hash_function
            if index_hash_function is not None
            else self._construction_config_get("heavenbase.backend_defaults.redis.index_hash_function", default=_DEFAULT_INDEX_HASH_FUNCTION)
        )
        self.index_hash_function = str(configured_hash or _DEFAULT_INDEX_HASH_FUNCTION)
        self._index_hasher = resolve_hash_int(self.index_hash_function)
        configured_length = (
            index_hash_length
            if index_hash_length is not None
            else self._construction_config_get("heavenbase.backend_defaults.redis.index_hash_length", default=_DEFAULT_INDEX_HASH_LENGTH)
        )
        self.index_hash_length = int(_DEFAULT_INDEX_HASH_LENGTH if configured_length is None else configured_length)
        if self.index_hash_length < 1:
            raise ValueError("Redis index_hash_length must be positive")
        self._client_kwargs = dict(kwargs)
        self._schemas: dict[str, EntitySchema] = {}
        self._backfilled: set[tuple[str, tuple[str, ...], tuple[str, ...]]] = set()
        self._planned_index_fields: dict[str, tuple[str, ...]] = {}
        self._connect()

    def _namespace(self) -> str:
        return self.ws_id or "default"

    def _connect(self) -> None:
        try:
            import redis  # top-level redis-py (absolute import; not this subpackage)

            self.client = redis.Redis.from_url(self.url, decode_responses=True, **self._client_kwargs)
            self.client.ping()
            self.client_error = ""
        except Exception as exc:  # provider-unavailable: degrade, raise lazily in _need_client
            self.client = None
            self.client_error = safe_error(exc)

    def _need_client(self) -> None:
        if self.client is None:
            detail = f": {self.client_error}" if self.client_error else ""
            raise RuntimeError(f"Redis backend {self.name!r} is unavailable{detail}")

    def _refresh_workspace(self) -> None:
        self._schemas.clear()
        self._backfilled.clear()
        self._planned_index_fields.clear()
        self._connect()

    def _prefix(self, entity_id: str) -> str:
        # Namespace row/index keys by the backend instance name so a redis + a redisearch
        # (or another redis) backend sharing one server and workspace never collide.
        return f"{self._namespace()}:{self.name}:{entity_id}:"

    def _key(self, entity_id: str, object_id: Any) -> str:
        return f"{self._prefix(entity_id)}{object_id}"

    # ------------------------------------------------------------- value index
    def _indexed_fields(self, entity_id: str) -> list[str]:
        schema = self._schemas.get(entity_id)
        if schema is None:
            return []
        if entity_id in self._planned_index_fields:
            return list(self._planned_index_fields[entity_id])
        return [
            field_schema.name
            for field_schema in schema.fields.values()
            if field_schema.name != "object_id" and self.indexable(field_schema.dtype, resolver=self.module_resolver)
        ]

    def _array_indexed_fields(self, entity_id: str) -> list[str]:
        schema = self._schemas.get(entity_id)
        if schema is None:
            return []
        return [field_schema.name for field_schema in schema.fields.values() if self._array_contains_indexable(field_schema)]

    def _array_contains_indexable(self, field_schema: Any) -> bool:
        """Return whether array item SET indexes preserve scan semantics for this field."""
        if field_schema.name == "object_id" or type(field_schema.dtype).identifier != "array":
            return False
        item_type = getattr(field_schema.dtype, "item_type", None)
        if item_type is None:
            return False
        item_identifier = type(item_type).identifier
        if item_identifier == "categorical":
            return getattr(item_type, "base", "string") == "string"
        return item_identifier in ("date", "identifier", "long-text", "medium-text", "short-text")

    def _array_index_values(self, value: Any) -> set[Any]:
        return set(value or []) if isinstance(value, list) else set()

    def _idx_key(self, entity_id: str, field: str, value: Any) -> str:
        digest = fmt_hash(self._index_hasher(str(value)), length=self.index_hash_length)
        return f"{self._namespace()}:{self.name}:_idx:{entity_id}:{field}:{digest}"

    @backend_locked
    def ensure(self, schema: EntitySchema, bindings: Sequence[Any]) -> None:
        """Record the entity schema and backfill eq/in indexes for any pre-existing rows.

        Recording the schema tells ``upsert`` which fields to index. Backfilling makes rows
        written before this ``ensure`` (a fresh process, or a field that just became indexable)
        discoverable through the SET index, so native eq/in returns the same rows as the scan
        predicate instead of silently dropping them. Idempotent and run once per indexed-field
        set per instance; a no-op when the client is down or the entity has no indexable fields.

        Args:
            schema (EntitySchema): Entity schema to record for row writes and index backfill.
            bindings (Sequence[Any]): Storage bindings supplied by the workspace. Redis stores
                rows as whole documents and does not need per-binding setup.

        Returns:
            None: This function does not return a value.
        """
        self._schemas[schema.entity_id] = schema
        self._backfill_index(schema.entity_id)

    @backend_locked
    def ensure_indexes(self, schema: EntitySchema, storage_plan: Any) -> None:
        """Reconcile Redis SET indexes with the resolved single-field plan.

        Args:
            schema: Entity schema owning the row documents.
            storage_plan: Resolved storage, index, and constraint plan.

        Returns:
            None: Removed fields are purged and active fields are backfilled.
        """
        old_fields = set(self._indexed_fields(schema.entity_id))
        fields = tuple(
            binding.fields[0]
            for binding in [*storage_plan.backend_indexes(self.name), *storage_plan.backend_constraints(self.name)]
            if len(binding.fields) == 1
        )
        self._planned_index_fields[schema.entity_id] = fields
        self._purge_index_fields(schema.entity_id, old_fields - set(fields))
        self._backfilled = {signature for signature in self._backfilled if signature[0] != schema.entity_id}
        self._backfill_index(schema.entity_id)

    def _purge_index_fields(self, entity_id: str, fields: set[str]) -> None:
        """Delete server-side SET keys for fields removed from the plan."""
        if self.client is None:
            return
        for field_name in fields:
            prefix = f"{self._namespace()}:{self.name}:_idx:{entity_id}:{field_name}:"
            for key in self.client.scan_iter(match=prefix + "*"):
                self.client.delete(key)

    def _backfill_index(self, entity_id: str) -> None:
        """Re-assert SET-index membership for every stored row's indexable fields (idempotent)."""
        if self.client is None:
            return
        fields = self._indexed_fields(entity_id)
        array_fields = self._array_indexed_fields(entity_id)
        signature = (entity_id, tuple(fields), tuple(array_fields))
        if (not fields and not array_fields) or signature in self._backfilled:
            return
        for row in self.scan(entity_id):
            object_id = str(row.get("object_id"))
            pipe = self.client.pipeline(transaction=True)
            for field in fields:
                value = row.get(field)
                if value is not None:
                    pipe.sadd(self._idx_key(entity_id, field, value), object_id)
            for field in array_fields:
                for value in self._array_index_values(row.get(field)):
                    pipe.sadd(self._idx_key(entity_id, field, value), object_id)
            pipe.execute()
        self._backfilled.add(signature)

    @backend_locked
    def drop(self) -> None:
        """Delete every row and index key under this workspace namespace.

        Args:
            None.

        Returns:
            None: This function does not return a value.
        """
        if self.client is None:
            return
        for key in self.client.scan_iter(match=f"{self._namespace()}:{self.name}:*"):
            self.client.delete(key)
        self._schemas.clear()
        self._backfilled.clear()
        self._planned_index_fields.clear()

    @backend_locked
    def upsert(self, rows: Sequence[RowOp]) -> None:
        """Write JSON documents and update eq/in indexes in ordered ``MULTI/EXEC`` phases.

        Each adjacent same-kind row batch uses a separate transaction. Redis does not roll
        back earlier commands or stop later queued commands after a runtime command error,
        so phase separation ensures a failed source rewrite cannot continue to a later
        target-delete phase. The previous value is read before each phase. Under the
        single-writer assumption (see the module docstring) the index stays consistent;
        ``execute`` additionally re-checks the row predicate, so a stale member can never
        yield a false positive (it can still drop a true match under concurrent
        cross-process writers).

        Args:
            rows (Sequence[RowOp]): Row mutations to apply. ``delete`` operations remove the
                JSON document and value-index memberships; other operations replace the row.

        Returns:
            None: This function does not return a value.

        Raises:
            RuntimeError: If the Redis client is unavailable or the entity was not ensured.
        """
        self._need_client()
        rows = list(rows)
        if not rows:
            return
        for row in rows:
            if row.entity_id not in self._schemas:
                raise RuntimeError(f"Redis backend {self.name!r}: ensure() not called for {row.entity_id!r}")
        for _entity_id, _is_delete, batch in self.ordered_row_op_batches(rows):
            keys = [self._key(row.entity_id, row.object_id) for row in batch]
            old_raws = self.client.mget(keys)
            shadow = {key: loads_json(old_raw) if old_raw else None for key, old_raw in zip(keys, old_raws)}
            pipe = self.client.pipeline(transaction=True)
            for row, key in zip(batch, keys):
                old = shadow[key]
                fields = self._indexed_fields(row.entity_id)
                array_fields = self._array_indexed_fields(row.entity_id)
                object_id = str(row.object_id)
                if row.is_delete():
                    pipe.delete(key)
                    if old is not None:
                        for field in fields:
                            old_value = old.get(field)
                            if old_value is not None:
                                pipe.srem(self._idx_key(row.entity_id, field, old_value), object_id)
                        for field in array_fields:
                            for old_value in self._array_index_values(old.get(field)):
                                pipe.srem(self._idx_key(row.entity_id, field, old_value), object_id)
                    shadow[key] = None
                    continue
                item = row.to_row()
                item["object_id"] = object_id
                pipe.set(key, dumps_json(item))
                for field in fields:
                    old_value = None if old is None else old.get(field)
                    new_value = item.get(field)
                    if old is not None and old_value == new_value:
                        continue
                    if old_value is not None:
                        pipe.srem(self._idx_key(row.entity_id, field, old_value), object_id)
                    if new_value is not None:
                        pipe.sadd(self._idx_key(row.entity_id, field, new_value), object_id)
                for field in array_fields:
                    old_values = set() if old is None else self._array_index_values(old.get(field))
                    new_values = self._array_index_values(item.get(field))
                    if old_values == new_values:
                        continue
                    for old_value in old_values:
                        pipe.srem(self._idx_key(row.entity_id, field, old_value), object_id)
                    for new_value in new_values:
                        pipe.sadd(self._idx_key(row.entity_id, field, new_value), object_id)
                shadow[key] = item
            pipe.execute()

    @backend_locked
    def scan(self, entity_id: str) -> list[dict[str, Any]]:
        """Return every row for one entity by scanning the key prefix.

        Args:
            entity_id (str): Entity identifier whose row namespace should be scanned.

        Returns:
            list[dict[str, Any]]: Decoded row dictionaries.

        Raises:
            RuntimeError: If the Redis client is unavailable.
        """
        self._need_client()
        rows: list[dict[str, Any]] = []
        for key in self.client.scan_iter(match=f"{self._prefix(entity_id)}*"):
            raw = self.client.get(key)
            if raw:
                rows.append(loads_json(raw))
        return rows

    @backend_locked
    def get_many(self, entity_id: str, object_ids: Iterable[Any], *, columns: Sequence[str] | None = None) -> list[dict[str, Any] | None]:
        """Return rows via native ``MGET`` key lookup.

        Args:
            entity_id (str): Entity identifier whose rows should be loaded.
            object_ids (Iterable[Any]): Object identifiers to fetch in caller order.
            columns (Sequence[str] | None): Optional field names to return; ``None`` returns the complete fragment.

        Returns:
            list[dict[str, Any] | None]: Decoded rows aligned with ``object_ids``. Missing
            keys are returned as ``None``.

        Raises:
            RuntimeError: If the Redis client is unavailable.
        """
        self._need_client()
        ordered = [str(object_id) for object_id in object_ids]
        if not ordered:
            return []
        values = self.client.mget([self._key(entity_id, object_id) for object_id in ordered])
        return self._project_rows([loads_json(value) if value else None for value in values], columns)

    @backend_locked
    def count(self, entity_id: str) -> int:
        """Return row count by scanning the key prefix.

        Args:
            entity_id (str): Entity identifier whose row namespace should be counted.

        Returns:
            int: Number of stored rows for the entity.

        Raises:
            RuntimeError: If the Redis client is unavailable.
        """
        self._need_client()
        return sum(1 for _ in self.client.scan_iter(match=f"{self._prefix(entity_id)}*"))

    def _live_tag_claims(self) -> dict[str, Any]:
        """Return live tag claims derived from Redis availability.

        Args:
            None.

        Returns:
            dict[str, Any]: Live Backend tag claims.
        """
        return {
            **super()._live_tag_claims(),
            "backend-availability": "ready" if self.client is not None else "unavailable",
        }

    # --------------------------------------------------- native eq/in execute
    def _is_index_fragment(self, fragment: QueryFragment) -> bool:
        if fragment.kind() == "index_filter":
            return True
        if fragment.op in ("and", "or"):
            return all(self._is_index_fragment(child) for child in fragment.children())
        return False

    def _resolve_index(self, entity_id: str, fragment: QueryFragment) -> set[str]:
        if fragment.kind() == "index_filter":
            if fragment.field == "object_id":
                # object_id is the key itself, not an indexed column -> resolve to the value(s)
                # directly; get_many() then drops ids that do not exist.
                if fragment.notes.get("op") == "in":
                    return {str(value) for value in fragment.payload["values"]}
                return {str(fragment.payload["value"])}
            if fragment.notes.get("op") == "in":
                keys = [self._idx_key(entity_id, fragment.field, value) for value in fragment.payload["values"]]
                return set(self.client.sunion(keys)) if keys else set()
            return set(self.client.smembers(self._idx_key(entity_id, fragment.field, fragment.payload["value"])))
        if fragment.op == "and":
            sets = [self._resolve_index(entity_id, child) for child in fragment.children()]
            return set.intersection(*sets) if sets else set()
        if fragment.op == "or":
            sets = [self._resolve_index(entity_id, child) for child in fragment.children()]
            return set().union(*sets) if sets else set()
        return set()

    @backend_locked
    def execute(self, fragment: QueryFragment) -> ResultFrame:
        """Resolve native eq/in via SET indexes; otherwise fall back to scan.

        The SET index only narrows the candidate set; the portable ``row_payload``
        predicate is re-applied to every fetched row, so a stale index member (e.g.
        from a cross-process write race) can never yield a false-positive result.

        Args:
            fragment (QueryFragment): Backend-ready query fragment compiled by the handler layer.

        Returns:
            ResultFrame: Matching rows and object identifiers.

        Raises:
            RuntimeError: If the Redis client is unavailable.
        """
        self._need_client()
        if self._is_index_fragment(fragment):
            entity_id = fragment.entity_id()
            if entity_id:
                from ..scanning import execute_scan

                object_ids = sorted(self._resolve_index(entity_id, fragment))
                rows = [row for row in self.get_many(entity_id, object_ids) if row]
                return execute_scan(self, fragment, rows=rows)
        return super().execute(fragment)

    def to_dict(self) -> dict[str, Any]:
        """Return Redis backend metadata and connection state.

        Args:
            None.

        Returns:
            dict[str, Any]: Backend metadata, namespace, index hash settings, and current
            provider connection status.
        """
        return {
            "name": self.name,
            "type": self.type,
            "url": self.url,
            "namespace": self._namespace(),
            "index_hash_function": self.index_hash_function,
            "index_hash_length": self.index_hash_length,
            "connected": self.client is not None,
            "error": self.client_error,
        }
