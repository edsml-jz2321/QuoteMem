"""OpenSearch Backend runtime implementation."""

from __future__ import annotations

__all__ = ["OpenSearchBackend"]

from heavenbase.utils import Any, Iterable, Mapping, Sequence, safe_error
from heavenbase.utils.redact import redact_secrets

from heavenbase.backends._scope import provider_scope, scoped_provider_name
from heavenbase.backends.base import Backend, backend_locked
from heavenbase.execution.contracts import LOGICAL_OPS, QueryFragment, RowOp
from heavenbase.execution.frame.result import ResultFrame
from heavenbase.modules import ModuleResolver
from heavenbase.schema.model import EntitySchema
from heavenbase.types.storage import stores_as_integer

OPENSEARCH_FIELD_RESERVED_WORDS = frozenset("_field_names _id _ignored _index _primary_term _routing _score _seq_no _source _type".split())
OPENSEARCH_INDEX_MAX_LENGTH = 255


class OpenSearchBackend(Backend):
    """OpenSearch provider adapter with native index-backed scalar execution."""

    identifier = "opensearch"
    aliases = ("open-search",)
    provider_reserved_words = OPENSEARCH_FIELD_RESERVED_WORDS
    provider_max_name_length = OPENSEARCH_INDEX_MAX_LENGTH

    @classmethod
    def from_config(
        cls,
        name: str,
        data: Mapping[str, Any],
        *,
        resolver: ModuleResolver | None = None,
        config: Any | None = None,
    ) -> "OpenSearchBackend":
        """Build an OpenSearch backend from a config mapping.

        Args:
            name: Workspace backend instance name.
            data: Backend config mapping. Supports ``hosts``, optional
                ``ws_id``, and OpenSearch client keyword arguments.
            resolver: Context-owned module authority.
            config: Config authority retained by the backend.

        Returns:
            OpenSearchBackend: Configured backend instance.
        """
        cfg = dict(data or {})
        ws_id = cfg.pop("ws_id", None)
        return cls(name, cfg.pop("hosts", None), ws_id=ws_id, resolver=resolver, config=config, **cfg)

    def __init__(
        self,
        name: str = "opensearch",
        hosts: list[str] | str | None = None,
        ws_id: str | None = None,
        *,
        resolver: ModuleResolver | None = None,
        config: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """Create an OpenSearch backend.

        Args:
            name: Workspace backend instance name.
            hosts: OpenSearch host URL or URL list.
            ws_id: Optional workspace namespace for physical index names.
            resolver: Context-owned module authority.
            config: Config authority retained by the backend.
            **kwargs: Extra keyword arguments passed to ``OpenSearch``.

        Returns:
            None: This function does not return a value.
        """
        super().__init__(name, ws_id=ws_id, resolver=resolver, config=config)
        resolved_hosts = hosts or self._construction_config_get(
            "heavenbase.backend_defaults.opensearch.hosts",
            default=["http://localhost:9201"],
        )
        self.hosts = [resolved_hosts] if isinstance(resolved_hosts, str) else list(resolved_hosts)
        kwargs.setdefault(
            "timeout",
            float(self._construction_config_get("heavenbase.backend_defaults.opensearch.timeout", default=1.0)),
        )
        self._indices: dict[str, str] = {}
        self._schemas: dict[str, EntitySchema] = {}
        self.client = None
        self.client_error = ""
        try:
            from opensearchpy import OpenSearch

            self.client = OpenSearch(self.hosts, **kwargs)
        except Exception as exc:
            self.client = None
            self.client_error = redact_secrets(safe_error(exc))

    def _refresh_workspace(self) -> None:
        """Clear workspace-derived physical index names."""
        self._indices.clear()
        self._schemas.clear()

    def _need_client(self) -> None:
        """Raise a contextual error when the provider is unavailable."""
        if self.client is None:
            detail = f": {self.client_error}" if self.client_error else ""
            raise RuntimeError(f"OpenSearch backend {self.name!r} is unavailable{detail}")

    def _index(self, entity_id: str) -> str:
        """Return a workspace-safe physical OpenSearch index name."""
        return self._indices.setdefault(
            entity_id,
            scoped_provider_name(
                self._index_scope(),
                entity_id,
                separator="-",
                max_length=self.provider_max_name_length,
            ),
        )

    def _index_scope(self) -> str:
        """Return an unambiguous lowercase index-owner token."""
        return provider_scope(
            self.ws_id,
            self.name,
            self.identifier,
            separator="-",
            max_length=64,
        )

    def schema_warnings(self, schema: EntitySchema, bindings: Sequence[Any]) -> list[str]:
        """Return OpenSearch index and field-name warnings.

        Args:
            schema: Entity schema being registered.
            bindings: Storage bindings routed to this backend.

        Returns:
            list[str]: Provider naming warnings.
        """
        warnings = super().schema_warnings(schema, bindings)
        warnings.extend(
            self._physical_name_warnings(
                [(schema.entity_id, self._index(schema.entity_id))],
                kind="OpenSearch index",
                max_length=self.provider_max_name_length,
                case_sensitive=False,
            )
        )
        warnings.extend(
            self._physical_name_warnings(
                [(field_name, field_name) for field_name in sorted(schema.fields)],
                kind=f"OpenSearch field for {schema.entity_id!r}",
                reserved=self.provider_reserved_words,
                case_sensitive=False,
            )
        )
        return warnings

    def _mapping(self, schema: EntitySchema, indexed_fields: set[str] | None = None) -> dict[str, Any]:
        """Build one OpenSearch mapping from logical fields and index policy."""
        properties: dict[str, Any] = {"object_id": {"type": "keyword"}}
        for field_schema in schema.fields.values():
            if field_schema.name == "object_id":
                continue
            identifier = type(field_schema.dtype).identifier
            indexed = indexed_fields is None or field_schema.name in indexed_fields
            if stores_as_integer(field_schema.dtype):
                mapping: dict[str, Any] = {"type": "long"}
            elif identifier == "date":
                mapping = {"type": "date", "format": "strict_date"}
            elif identifier == "float":
                mapping = {"type": "double"}
            elif identifier == "boolean":
                mapping = {"type": "boolean"}
            elif identifier == "vector":
                mapping = {"type": "float", "index": False, "doc_values": False}
            elif identifier == "artifact":
                mapping = {"type": "keyword", "index": False, "doc_values": False}
            elif identifier in ("array", "json", "hyper-g"):
                mapping = {"type": "object", "enabled": False}
            elif not indexed:
                mapping = {"type": "keyword", "index": False, "doc_values": False}
            else:
                mapping = {"type": "wildcard"}
            if not indexed and mapping.get("type") in ("long", "double", "boolean", "date"):
                mapping.update({"index": False, "doc_values": False})
            properties[field_schema.name] = mapping
        return {"mappings": {"properties": properties}}

    @backend_locked
    def ensure(self, schema: EntitySchema, bindings: Sequence[Any]) -> None:
        """Ensure an OpenSearch index and mapping for an entity.

        Args:
            schema: Entity schema to store.
            bindings: Storage bindings routed to this backend.

        Returns:
            None: Index creation is idempotent.
        """
        del bindings
        self._ensure(schema, indexed_fields=None)

    @backend_locked
    def ensure_plan(self, schema: EntitySchema, storage_plan: Any) -> None:
        """Ensure an OpenSearch mapping from a resolved scalar index plan.

        Args:
            schema: Entity schema to store.
            storage_plan: Resolved storage, index, and constraint plan.

        Returns:
            None: The index has compatible mappings.
        """
        indexed_fields = {
            binding.fields[0]
            for binding in [*storage_plan.backend_indexes(self.name), *storage_plan.backend_constraints(self.name)]
            if len(binding.fields) == 1
        }
        self._ensure(schema, indexed_fields=indexed_fields)

    def _ensure(self, schema: EntitySchema, *, indexed_fields: set[str] | None) -> None:
        """Create or additively extend one provider mapping."""
        if self.client is None:
            self._schemas[schema.entity_id] = schema
            return
        index = self._index(schema.entity_id)
        try:
            if self.client.indices.exists(index=index):
                self._ensure_existing_mapping(index, schema, indexed_fields)
                self._schemas[schema.entity_id] = schema
                return
            try:
                self.client.indices.create(index=index, body=self._mapping(schema, indexed_fields))
            except Exception as exc:
                if "resource_already_exists_exception" in safe_error(exc):
                    self._ensure_existing_mapping(index, schema, indexed_fields)
                else:
                    raise
            self._schemas[schema.entity_id] = schema
        except Exception as exc:
            self.client_error = redact_secrets(safe_error(exc))
            raise RuntimeError(f"OpenSearch ensure failed for {schema.entity_id!r}: {self.client_error}") from exc

    def _ensure_existing_mapping(self, index: str, schema: EntitySchema, indexed_fields: set[str] | None) -> None:
        """Add missing fields and reject immutable index-policy changes."""
        desired = self._mapping(schema, indexed_fields)["mappings"]["properties"]
        response = self.client.indices.get_mapping(index=index)
        current: dict[str, Any] = {}
        for payload in dict(response).values():
            current.update((payload.get("mappings") or {}).get("properties") or {})
        defaults = {"index": True, "doc_values": True, "enabled": True, "format": None}
        immutable = ("type", "index", "doc_values", "enabled", "format")
        for field_name in sorted(set(current) & set(desired)):
            mismatches = []
            for attribute in immutable:
                actual_value = current[field_name].get(attribute, defaults.get(attribute))
                expected_value = desired[field_name].get(attribute, defaults.get(attribute))
                if actual_value != expected_value:
                    mismatches.append(f"{attribute} expected {expected_value!r} got {actual_value!r}")
            if mismatches:
                raise ValueError(f"OpenSearch mapping for {schema.entity_id}.{field_name} is incompatible: {mismatches}; " "create a migrated entity/index")
        missing = {field_name: mapping for field_name, mapping in desired.items() if field_name not in current}
        if missing:
            self.client.indices.put_mapping(index=index, body={"properties": missing})

    @backend_locked
    def drop(self) -> None:
        """Delete every OpenSearch index tracked by this backend.

        Args:
            None.

        Returns:
            None: Deletion is idempotent.
        """
        if self.client is None:
            return
        try:
            names = set(self._indices.values())
            response = self.client.indices.get_alias(index=f"{self._index_scope()}-*", ignore=[404])
            if isinstance(response, Mapping) and "status" not in response:
                names.update(str(name) for name in response if str(name).startswith(f"{self._index_scope()}-"))
            for index in sorted(names):
                self.client.indices.delete(index=index, ignore=[404])
        except Exception as exc:
            self.client_error = redact_secrets(safe_error(exc))
            raise RuntimeError(f"OpenSearch drop failed: {self.client_error}") from exc
        self._indices.clear()
        self._schemas.clear()

    @backend_locked
    def upsert(self, rows: Sequence[RowOp]) -> None:
        """Index replacement documents or delete them idempotently.

        Args:
            rows: Backend-ready row mutations.

        Returns:
            None: Touched indexes are refreshed after ordered mutations.
        """
        self._need_client()
        touched: set[str] = set()
        mutation_error: Exception | None = None
        try:
            for row in rows:
                index = self._index(row.entity_id)
                if row.is_delete():
                    self.client.delete(index=index, id=str(row.object_id), ignore=[404])
                else:
                    self.client.index(index=index, id=str(row.object_id), body=row.to_row())
                touched.add(index)
        except Exception as exc:
            mutation_error = exc

        refresh_errors = []
        for index in sorted(touched):
            try:
                if self.client.indices.exists(index=index):
                    self.client.indices.refresh(index=index)
            except Exception as exc:
                refresh_errors.append(f"{index}: {redact_secrets(safe_error(exc))}")

        if mutation_error is not None:
            self.client_error = redact_secrets(safe_error(mutation_error))
            if refresh_errors:
                self.client_error = f"{self.client_error}; refresh failures: {refresh_errors}"
            raise RuntimeError(f"OpenSearch mutation failed: {self.client_error}") from mutation_error
        if refresh_errors:
            self.client_error = f"refresh failures: {refresh_errors}"
            raise RuntimeError(f"OpenSearch mutation failed: {self.client_error}")

    @staticmethod
    def _row(hit: Mapping[str, Any]) -> dict[str, Any]:
        """Return one OpenSearch hit as a HeavenBase row."""
        row = dict(hit.get("_source") or {})
        row["object_id"] = row.get("object_id") or hit.get("_id")
        return row

    @backend_locked
    def scan(self, entity_id: str) -> list[dict[str, Any]]:
        """Return every indexed document for one entity.

        Args:
            entity_id: Logical entity identifier.

        Returns:
            list[dict[str, Any]]: Exhaustive provider rows.
        """
        self._need_client()
        index = self._index(entity_id)
        try:
            if not self.client.indices.exists(index=index):
                return []
            import opensearchpy
            from opensearchpy.helpers import scan as scan_hits

            batch_size = max(
                1,
                int(self.config.get("heavenbase.backend_defaults.opensearch.scan_limit", default=10000)),
            )
            hits = scan_hits(
                self.client,
                index=index,
                query={"query": {"match_all": {}}},
                size=batch_size,
                preserve_order=False,
            )
            return [self._row(hit) for hit in hits]
        except Exception as exc:
            self.client_error = redact_secrets(safe_error(exc))
            raise RuntimeError(f"OpenSearch scan failed for {entity_id!r}: {self.client_error}") from exc

    @backend_locked
    def get_many(self, entity_id: str, object_ids: Iterable[Any], *, columns: Sequence[str] | None = None) -> list[dict[str, Any] | None]:
        """Return rows through OpenSearch multi-get in caller order.

        Args:
            entity_id (str): Logical entity identifier.
            object_ids (Iterable[Any]): Object ids to fetch in caller order.
            columns (Sequence[str] | None): Optional field names to return; ``None`` returns the complete fragment.

        Returns:
            list[dict[str, Any] | None]: Rows aligned with ``object_ids``. Missing
            documents are returned as ``None``.

        Raises:
            TypeError: If ``columns`` contains a non-string name.
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
            response = self.client.mget(index=index, body={"ids": list(dict.fromkeys(ordered_ids))})
            rows = {str(hit.get("_id")): self._row(hit) for hit in response.get("docs", []) if hit.get("found", True)}
            return self._project_rows([rows.get(object_id) for object_id in ordered_ids], selection)
        except Exception as exc:
            self.client_error = redact_secrets(safe_error(exc))
            raise RuntimeError(f"OpenSearch get_many failed for {entity_id!r}: {self.client_error}") from exc

    @backend_locked
    def exists_many(self, entity_id: str, object_ids: Iterable[Any]) -> list[bool]:
        """Return existence flags through OpenSearch multi-get.

        Args:
            entity_id (str): Logical entity identifier.
            object_ids (Iterable[Any]): Object ids to test in caller order.

        Returns:
            list[bool]: Existence flags aligned with ``object_ids``.
        """
        return [row is not None for row in self.get_many(entity_id, object_ids)]

    @backend_locked
    def count(self, entity_id: str) -> int:
        """Return the exact OpenSearch document count for one entity.

        Args:
            entity_id (str): Logical entity identifier.

        Returns:
            int: Exact number of OpenSearch documents for the entity.
        """
        self._need_client()
        index = self._index(entity_id)
        try:
            if not self.client.indices.exists(index=index):
                return 0
            return int(self.client.count(index=index).get("count", 0))
        except Exception as exc:
            self.client_error = redact_secrets(safe_error(exc))
            raise RuntimeError(f"OpenSearch count failed for {entity_id!r}: {self.client_error}") from exc

    def exact_count(self, entity_id: str) -> int:
        """Return the exact provider document count.

        Args:
            entity_id (str): Logical entity identifier.

        Returns:
            int: Exact OpenSearch document count suitable for integrity proofs.
        """
        return self.count(entity_id)

    def _is_native_fragment(self, fragment: QueryFragment) -> bool:
        """Return whether a fragment tree contains only OpenSearch Query DSL."""
        if fragment.kind() == "opensearch_query":
            return True
        if fragment.op in ("and", "or"):
            return all(self._is_native_fragment(child) for child in fragment.children())
        if fragment.op == "not":
            return self._is_native_fragment(fragment.children()[0])
        return False

    def _query(self, fragment: QueryFragment) -> dict[str, Any]:
        """Compile a native fragment tree to OpenSearch Query DSL."""
        if fragment.kind() == "opensearch_query":
            return dict(fragment.payload)
        if fragment.op == "and":
            return {"bool": {"must": [self._query(child) for child in fragment.children()]}}
        if fragment.op == "or":
            return {
                "bool": {
                    "should": [self._query(child) for child in fragment.children()],
                    "minimum_should_match": 1,
                }
            }
        if fragment.op == "not":
            return {"bool": {"must_not": [self._query(fragment.children()[0])]}}
        raise TypeError(f"Cannot compile {fragment.op!r} to OpenSearch Query DSL")

    @backend_locked
    def execute(self, fragment: QueryFragment) -> ResultFrame:
        """Execute native OpenSearch fragments or the portable scan fallback.

        Args:
            fragment: Backend-ready query fragment.

        Returns:
            ResultFrame: Matching row fragments.
        """
        self._need_client()
        if fragment.mode() == "py_scan" or (fragment.op in LOGICAL_OPS and not self._is_native_fragment(fragment)):
            return super().execute(fragment)
        if fragment.kind() != "opensearch_query" and fragment.op not in LOGICAL_OPS:
            raise NotImplementedError(f"OpenSearch cannot execute fragment {fragment.op!r}")
        if not self._is_native_fragment(fragment):
            raise NotImplementedError(f"OpenSearch cannot execute non-native filter fragment {fragment.op!r}")
        entity_id = fragment.entity_id()
        index = self._index(entity_id) if entity_id else ""
        if not self.client.indices.exists(index=index):
            return ResultFrame.empty()
        try:
            import opensearchpy
            from opensearchpy.helpers import scan as scan_hits

            size = max(1, int(self.config.get("heavenbase.backend_defaults.opensearch.filter_limit", default=10000)))
            hits = scan_hits(
                self.client,
                index=index,
                query={"query": self._query(fragment)},
                size=size,
                preserve_order=False,
            )
            return ResultFrame.from_rows([self._row(hit) for hit in hits])
        except Exception as exc:
            self.client_error = redact_secrets(safe_error(exc))
            raise RuntimeError(f"OpenSearch query failed for {entity_id!r}: {self.client_error}") from exc

    def _live_tag_claims(self) -> dict[str, Any]:
        """Return live availability derived from the optional OpenSearch client."""
        return {
            **super()._live_tag_claims(),
            "backend-availability": "ready" if self.client is not None else "unavailable",
        }

    def to_dict(self) -> dict[str, Any]:
        """Return connection-safe OpenSearch backend metadata.

        Args:
            None.

        Returns:
            dict[str, Any]: Backend metadata without credentials.
        """
        return {
            "name": self.name,
            "type": self.type,
            "deployment": "remote",
            "connected": self.client is not None,
        }
