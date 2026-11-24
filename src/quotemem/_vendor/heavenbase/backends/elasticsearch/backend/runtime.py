"""Elasticsearch Backend runtime implementation."""

from __future__ import annotations

__all__ = [
    "ElasticsearchBackend",
]

from heavenbase.utils import Any, Iterable, Mapping, Sequence, safe_error

from heavenbase.backends.base import Backend, backend_locked
from heavenbase.execution.contracts import LOGICAL_OPS, QueryFragment, RowOp
from heavenbase.execution.frame.result import ResultFrame
from heavenbase.modules import ModuleResolver
from heavenbase.schema.model import EntitySchema
from heavenbase.types.storage import stores_as_integer

ELASTICSEARCH_FIELD_RESERVED_WORDS = frozenset("_field_names _id _ignored _index _primary_term _routing _score _seq_no _source _type".split())
ELASTICSEARCH_INDEX_MAX_LENGTH = 255


class ElasticsearchBackend(Backend):
    """Elasticsearch provider adapter with native index-backed execution."""

    identifier = "elasticsearch"
    aliases = ("elastic", "search")
    provider_reserved_words = ELASTICSEARCH_FIELD_RESERVED_WORDS
    provider_max_name_length = ELASTICSEARCH_INDEX_MAX_LENGTH

    @classmethod
    def from_config(cls, name: str, data: Mapping[str, Any], *, resolver: ModuleResolver | None = None, config: Any | None = None) -> "ElasticsearchBackend":
        """Build the Elasticsearch backend from a config mapping.

        Args:
            name (str): Workspace backend instance name.
            data (Mapping[str, Any]): Backend config mapping. Supported keys are the
                constructor parameters, including ``hosts`` and ``ws_id``.
            resolver: Context-owned module authority.
            config: Config authority retained by the backend.

        Returns:
            ElasticsearchBackend: Configured Elasticsearch backend instance.
        """
        cfg = dict(data or {})
        ws_id = cfg.pop("ws_id", None)
        return cls(name, cfg.pop("hosts", None), ws_id=ws_id, resolver=resolver, config=config, **cfg)

    def __init__(
        self,
        name: str = "elasticsearch",
        hosts: list[str] | None = None,
        ws_id: str | None = None,
        *,
        resolver: ModuleResolver | None = None,
        config: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """Create an Elasticsearch backend.

        Args:
            name (str): Workspace backend instance name.
            hosts (list[str] | None): Elasticsearch host URLs. When omitted, uses
                ``heavenbase.backend_defaults.elasticsearch.hosts``.
            ws_id (str | None): Optional workspace namespace for physical index names.
            resolver: Context-owned module authority.
            config: Config authority retained by the backend.
            **kwargs (Any): Extra keyword arguments passed to ``Elasticsearch(...)``.

        Returns:
            None: This function does not return a value.
        """
        super().__init__(name, ws_id=ws_id, resolver=resolver, config=config)
        resolved_hosts = hosts or self._construction_config_get(
            "heavenbase.backend_defaults.elasticsearch.hosts",
            default=["http://localhost:9200"],
        )
        self.hosts = [resolved_hosts] if isinstance(resolved_hosts, str) else list(resolved_hosts)
        self._indices: dict[str, str] = {}
        self._schemas: dict[str, EntitySchema] = {}
        try:
            from elasticsearch import Elasticsearch

            self.client = Elasticsearch(self.hosts, **kwargs)
            self.client_error = ""
        except Exception as exc:
            self.client = None
            self.client_error = safe_error(exc)

    def _refresh_workspace(self) -> None:
        self._indices.clear()

    @backend_locked
    def drop(self) -> None:
        """Delete all Elasticsearch indexes tracked by this backend.

        Args:
            None.

        Returns:
            None: This function does not return a value.
        """
        if self.client is None:
            return
        for index in sorted(set(self._indices.values())):
            try:
                self.client.indices.delete(index=index, ignore=[404])
            except TypeError:
                self.client.indices.delete(index=index)
        self._indices.clear()
        self._schemas.clear()

    def _need_client(self) -> None:
        if self.client is None:
            detail = f": {self.client_error}" if self.client_error else ""
            raise RuntimeError(f"Elasticsearch backend {self.name!r} is unavailable{detail}")

    def _index(self, entity_id: str) -> str:
        parts = (self.ws_id, entity_id) if self.ws_id else (entity_id,)
        return self._indices.setdefault(entity_id, self._physical_name(*parts, sep="-"))

    def schema_warnings(self, schema: EntitySchema, bindings: Sequence[Any]) -> list[str]:
        """Return Elasticsearch index and field-name warnings for this schema.

        Args:
            schema (EntitySchema): Entity schema being registered.
            bindings (Sequence[Any]): Storage bindings for the entity.

        Returns:
            list[str]: Provider naming warnings for physical index and field names.
        """
        warnings = super().schema_warnings(schema, bindings)
        warnings.extend(
            self._physical_name_warnings(
                [(schema.entity_id, self._index(schema.entity_id))],
                kind="Elasticsearch index",
                max_length=self.provider_max_name_length,
                case_sensitive=False,
            )
        )
        warnings.extend(
            self._physical_name_warnings(
                [(field_name, field_name) for field_name in sorted(schema.fields)],
                kind=f"Elasticsearch field for {schema.entity_id!r}",
                reserved=self.provider_reserved_words,
                case_sensitive=False,
            )
        )
        return warnings

    def _mapping(self, schema: EntitySchema, indexed_fields: set[str] | None = None) -> dict[str, Any]:
        props = {"object_id": {"type": "keyword"}}
        for field_schema in schema.fields.values():
            name = type(field_schema.dtype).identifier
            if field_schema.name == "object_id":
                continue
            indexed = indexed_fields is None or field_schema.name in indexed_fields
            if stores_as_integer(field_schema.dtype):
                props[field_schema.name] = {"type": "long"}
            elif name == "date":
                props[field_schema.name] = {"type": "date", "format": "strict_date"}
            elif name == "float":
                props[field_schema.name] = {"type": "double"}
            elif name == "boolean":
                props[field_schema.name] = {"type": "boolean"}
            elif name == "vector":
                props[field_schema.name] = {"type": "dense_vector", "dims": field_schema.dtype.dim or 1, "index": False}
            elif name == "artifact":
                props[field_schema.name] = {"type": "keyword", "index": False, "doc_values": False}
            elif name in ("array", "json", "hyper-g"):
                props[field_schema.name] = {"type": "object", "enabled": False}
            elif not indexed:
                props[field_schema.name] = {"type": "keyword", "index": False, "doc_values": False}
            else:
                props[field_schema.name] = {"type": "wildcard"}
            if not indexed and props[field_schema.name].get("type") in ("long", "double", "boolean", "date"):
                props[field_schema.name].update({"index": False, "doc_values": False})
        return {"mappings": {"properties": props}}

    @backend_locked
    def ensure(self, schema: EntitySchema, bindings: Sequence[Any]) -> None:
        """Ensure an Elasticsearch index and mapping for the entity.

        Args:
            schema (EntitySchema): Entity schema to store in Elasticsearch.
            bindings (Sequence[Any]): Storage bindings supplied by the workspace.

        Returns:
            None: This function does not return a value.

        Raises:
            Exception: If Elasticsearch rejects index creation for a reason other than an
                already-created index race.
        """
        self._ensure(schema, indexed_fields=None)

    @backend_locked
    def ensure_plan(self, schema: EntitySchema, storage_plan: Any) -> None:
        """Ensure an Elasticsearch mapping from the resolved scalar index plan.

        Args:
            schema: Entity schema to store in Elasticsearch.
            storage_plan: Resolved storage, index, and constraint plan.

        Returns:
            None: The index exists with compatible field mappings.
        """
        indexed_fields = {
            binding.fields[0]
            for binding in [*storage_plan.backend_indexes(self.name), *storage_plan.backend_constraints(self.name)]
            if len(binding.fields) == 1
        }
        self._ensure(schema, indexed_fields=indexed_fields)

    def _ensure(self, schema: EntitySchema, *, indexed_fields: set[str] | None) -> None:
        """Create or additively extend one provider index mapping."""
        if self.client is None:
            self._schemas[schema.entity_id] = schema
            return
        index = self._index(schema.entity_id)
        if self.client.indices.exists(index=index):
            self._ensure_existing_mapping(index, schema, indexed_fields)
            self._schemas[schema.entity_id] = schema
            return
        try:
            self.client.indices.create(index=index, **self._mapping(schema, indexed_fields))
        except Exception as exc:
            if "resource_already_exists_exception" in safe_error(exc):
                self._ensure_existing_mapping(index, schema, indexed_fields)
            else:
                raise
        self._schemas[schema.entity_id] = schema

    def _ensure_existing_mapping(self, index: str, schema: EntitySchema, indexed_fields: set[str] | None) -> None:
        """Add missing fields and reject immutable index-policy changes."""
        desired = self._mapping(schema, indexed_fields)["mappings"]["properties"]
        response = self.client.indices.get_mapping(index=index)
        current: dict[str, Any] = {}
        for payload in dict(response).values():
            current.update((payload.get("mappings") or {}).get("properties") or {})
        for field_name in set(current) & set(desired):
            current_indexed = current[field_name].get("index", True) is not False
            desired_indexed = desired[field_name].get("index", True) is not False
            if current_indexed != desired_indexed:
                raise ValueError(
                    f"Elasticsearch mapping for {schema.entity_id}.{field_name} cannot change index policy in place; create a migrated entity/index"
                )
        missing = {field_name: mapping for field_name, mapping in desired.items() if field_name not in current}
        if missing:
            self.client.indices.put_mapping(index=index, properties=missing)

    @backend_locked
    def upsert(self, rows: Sequence[RowOp]) -> None:
        """Index or delete documents in Elasticsearch.

        Args:
            rows (Sequence[RowOp]): Row mutations to apply. ``delete`` operations remove
                documents; other operations replace the indexed document body.

        Returns:
            None: This function does not return a value.

        Raises:
            RuntimeError: If the Elasticsearch client is unavailable.
        """
        self._need_client()
        touched = set()
        for row in rows:
            index = self._index(row.entity_id)
            touched.add(index)
            if row.is_delete():
                self.client.delete(index=index, id=str(row.object_id), ignore=[404])
                continue
            self.client.index(index=index, id=str(row.object_id), document=row.to_row())
        for index in touched:
            self.client.indices.refresh(index=index)

    @backend_locked
    def scan(self, entity_id: str) -> list[dict[str, Any]]:
        """Return all indexed documents for one entity when reachable.

        Args:
            entity_id (str): Entity identifier whose index should be scanned.

        Returns:
            list[dict[str, Any]]: Indexed document rows.

        Raises:
            RuntimeError: If the client is unavailable or the scan request fails.
        """
        self._need_client()
        index = self._index(entity_id)
        try:
            if not self.client.indices.exists(index=index):
                return []
            import elasticsearch
            from elasticsearch.helpers import scan as scan_hits

            batch_size = max(1, int(self.config.get("heavenbase.backend_defaults.elasticsearch.scan_limit", default=10000)))
            hits = scan_hits(
                self.client,
                index=index,
                query={"query": {"match_all": {}}},
                size=batch_size,
                preserve_order=False,
            )
            return [self._row(hit) for hit in hits]
        except Exception as exc:
            self.client_error = safe_error(exc)
            raise RuntimeError(f"Elasticsearch scan failed for {entity_id!r}: {self.client_error}") from exc

    @backend_locked
    def get_many(self, entity_id: str, object_ids: Iterable[Any], *, columns: Sequence[str] | None = None) -> list[dict[str, Any] | None]:
        """Return rows through Elasticsearch multi-get.

        Args:
            entity_id (str): Entity identifier whose rows should be loaded.
            object_ids (Iterable[Any]): Object identifiers to fetch in caller order.
            columns (Sequence[str] | None): Optional field names to return; ``None`` returns the complete fragment.

        Returns:
            list[dict[str, Any] | None]: Rows aligned with ``object_ids``. Missing documents
            are returned as ``None``.

        Raises:
            TypeError: If ``columns`` contains a non-string name.
            RuntimeError: If the client is unavailable or the multi-get request fails.
        """
        selection = self._row_columns(columns)
        ordered_ids = [str(object_id) for object_id in object_ids]
        if not ordered_ids:
            return []
        self._need_client()
        index = self._index(entity_id)
        try:
            if not self.client.indices.exists(index=index):
                return [None for _ in ordered_ids]
            resp = self.client.mget(index=index, ids=list(dict.fromkeys(ordered_ids)))
            rows_by_id = {str(hit.get("_id")): self._row(hit) for hit in resp.get("docs", []) if hit.get("found", True)}
            return self._project_rows([rows_by_id.get(row_id) for row_id in ordered_ids], selection)
        except Exception as exc:
            self.client_error = safe_error(exc)
            raise RuntimeError(f"Elasticsearch get_many failed for {entity_id!r}: {self.client_error}") from exc

    @backend_locked
    def exists_many(self, entity_id: str, object_ids: Iterable[Any]) -> list[bool]:
        """Return existence flags through Elasticsearch multi-get.

        Args:
            entity_id (str): Entity identifier whose rows should be checked.
            object_ids (Iterable[Any]): Object identifiers to check in caller order.

        Returns:
            list[bool]: Existence flags aligned with ``object_ids``.
        """
        return [row is not None for row in self.get_many(entity_id, object_ids)]

    @backend_locked
    def count(self, entity_id: str) -> int:
        """Return Elasticsearch index count without scanning documents.

        Args:
            entity_id (str): Entity identifier whose index should be counted.

        Returns:
            int: Number of indexed documents.

        Raises:
            RuntimeError: If the client is unavailable or the count request fails.
        """
        self._need_client()
        index = self._index(entity_id)
        try:
            if not self.client.indices.exists(index=index):
                return 0
            return int(self.client.count(index=index).get("count", 0))
        except Exception as exc:
            self.client_error = safe_error(exc)
            raise RuntimeError(f"Elasticsearch count failed for {entity_id!r}: {self.client_error}") from exc

    def _live_tag_claims(self) -> dict[str, Any]:
        """Return live tag claims derived from Elasticsearch availability.

        Args:
            None.

        Returns:
            dict[str, Any]: Live Backend tag claims.
        """
        return {
            **super()._live_tag_claims(),
            "backend-availability": "ready" if self.client is not None else "unavailable",
        }

    def _row(self, hit: dict[str, Any]) -> dict[str, Any]:
        row = dict(hit.get("_source") or {})
        row["object_id"] = row.get("object_id") or hit.get("_id")
        if "_score" in hit:
            row["score"] = hit.get("_score")
        return row

    def _is_es_fragment(self, fragment: QueryFragment) -> bool:
        if fragment.kind() == "es_query":
            return True
        if fragment.op in ("and", "or"):
            return all(self._is_es_fragment(child) for child in fragment.children())
        if fragment.op == "not":
            return self._is_es_fragment(fragment.children()[0])
        return False

    def _query(self, fragment: QueryFragment) -> dict:
        if fragment.kind() == "es_query":
            return fragment.payload
        if fragment.op == "and":
            return {"bool": {"must": [self._query(child) for child in fragment.children()]}}
        if fragment.op == "or":
            return {"bool": {"should": [self._query(child) for child in fragment.children()], "minimum_should_match": 1}}
        if fragment.op == "not":
            return {"bool": {"must_not": [self._query(fragment.children()[0])]}}
        raise TypeError(f"Cannot compile {fragment.op!r} to Elasticsearch query")

    @backend_locked
    def execute(self, fragment: QueryFragment) -> ResultFrame:
        """Execute Elasticsearch-native query fragments when possible.

        Args:
            fragment (QueryFragment): Backend-ready query fragment compiled by the handler layer.

        Returns:
            ResultFrame: Matching rows and object identifiers.

        Raises:
            NotImplementedError: If the fragment cannot be executed natively or by scan fallback.
            RuntimeError: If the Elasticsearch client is unavailable.
        """
        entity_id = fragment.entity_id()
        index = self._index(entity_id) if entity_id else ""
        self._need_client()
        if fragment.mode() == "py_scan" or (fragment.op in LOGICAL_OPS and not self._is_es_fragment(fragment)):
            return super().execute(fragment)
        if fragment.kind() == "es_query" or fragment.op in LOGICAL_OPS:
            if not self._is_es_fragment(fragment):
                raise NotImplementedError(f"Elasticsearch cannot execute non-native filter fragment {fragment.op!r}")
            if not self.client.indices.exists(index=index):
                return ResultFrame.empty()
            size = int(self.config.get("heavenbase.backend_defaults.elasticsearch.filter_limit", default=10000))
            resp = self.client.search(index=index, query=self._query(fragment), size=size)
            return ResultFrame.from_rows([self._row(hit) for hit in resp["hits"]["hits"]])
        raise NotImplementedError(f"Elasticsearch cannot execute fragment {fragment.op!r}")

    def to_dict(self) -> dict[str, Any]:
        """Return Elasticsearch backend metadata and connection state.

        Args:
            None.

        Returns:
            dict[str, Any]: Backend metadata, host list, physical namespace, and current
            provider connection status.
        """
        return {
            "name": self.name,
            "type": self.type,
            "cluster": self._physical_name(self.ws_id) if self.ws_id else "",
            "hosts": list(self.hosts),
            "connected": self.client is not None,
            "error": self.client_error,
        }
