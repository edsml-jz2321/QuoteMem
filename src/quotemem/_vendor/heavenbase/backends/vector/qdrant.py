"""Qdrant vector backend with exact row lifecycle and native search."""

from __future__ import annotations

__all__ = ["QdrantBackend"]

from uuid import NAMESPACE_URL, uuid5

from heavenbase.execution.contracts import LOGICAL_OPS, QueryFragment, RowOp
from heavenbase.utils import Any, Iterable, Mapping, Sequence, safe_error
from heavenbase.utils.files import file_uri_to_path, pj
from heavenbase.utils.redact import redact_secrets

from ...execution.frame.result import ResultFrame
from ...modules import ModuleResolver
from ...schema.model import EntitySchema
from ...types.vector import _scale_cosine_query
from .._scope import provider_scope, scoped_provider_name
from ..base import Backend, backend_locked


class QdrantBackend(Backend):
    """Qdrant provider adapter using named vectors and JSON payload rows."""

    identifier = "qdrant"

    @classmethod
    def from_config(
        cls,
        name: str,
        data: Mapping[str, Any],
        *,
        resolver: ModuleResolver | None = None,
        config: Any | None = None,
    ) -> "QdrantBackend":
        """Build a Qdrant backend from a config mapping.

        Args:
            name: Workspace backend instance name.
            data: Backend config mapping. Supports ``url`` or ``path``, ``api_key``,
                optional ``ws_id``, and Qdrant client keyword arguments.
            resolver: Context-owned module authority.
            config: Config authority retained by the backend.

        Returns:
            QdrantBackend: Configured Qdrant backend instance.

        Raises:
            ValueError: If both a remote ``url`` and local ``path`` are selected.
        """
        cfg = dict(data or {})
        ws_id = cfg.pop("ws_id", None)
        return cls(
            name,
            cfg.pop("url", None),
            path=cfg.pop("path", None),
            api_key=cfg.pop("api_key", None),
            ws_id=ws_id,
            resolver=resolver,
            config=config,
            **cfg,
        )

    def __init__(
        self,
        name: str = "qdrant",
        url: str | None = None,
        *,
        path: str | None = None,
        api_key: str | None = None,
        ws_id: str | None = None,
        resolver: ModuleResolver | None = None,
        config: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """Create a lazily connected remote or local Qdrant backend.

        Args:
            name: Workspace backend instance name.
            url: Remote Qdrant endpoint. Defaults to backend configuration.
            path: Local Qdrant base storage path. File URIs resolve relative to
                the configured data root; each backend/workspace receives a
                deterministic child directory. Cannot be combined with ``url``.
            api_key: Optional remote authentication key.
            ws_id: Optional workspace namespace for collection ownership.
            resolver: Context-owned module authority.
            config: Config authority retained by the backend.
            **kwargs: Extra keyword arguments passed to ``QdrantClient``.

        Returns:
            None: This function does not return a value.

        Raises:
            ValueError: If both a remote ``url`` and local ``path`` are selected,
                or if the local path URI is invalid.
        """
        super().__init__(name, ws_id=ws_id, resolver=resolver, config=config)
        configured_path = self._construction_config_get("heavenbase.backend_defaults.qdrant.path", default=None)
        selected_path = path if path is not None else (configured_path if url is None else None)
        configured_url = self._construction_config_get(
            "heavenbase.backend_defaults.qdrant.url",
            default="http://localhost:6333",
        )
        selected_url = url if url is not None else (None if selected_path is not None else configured_url)
        if selected_url and selected_path:
            raise ValueError("Qdrant accepts either url or path, not both")
        self.url = str(selected_url) if selected_path is None else None
        self._path_root = file_uri_to_path(str(selected_path), home=self.config.root) if selected_path is not None else None
        self.path = self._local_path()
        self.api_key = api_key if api_key is not None else self._construction_config_get("heavenbase.backend_defaults.qdrant.api_key", default=None)
        kwargs.setdefault(
            "timeout",
            float(self._construction_config_get("heavenbase.backend_defaults.qdrant.timeout", default=5.0)),
        )
        self._client_kwargs = dict(kwargs)
        self._schemas: dict[str, EntitySchema] = {}
        self._owned_fields: dict[str, set[str]] = {}
        self._collections: dict[str, str] = {}
        self.client = None
        self.client_error = ""
        self._models = None
        self._connect_pending = True

    def _connect(self) -> None:
        try:
            from qdrant_client import QdrantClient, models

            kwargs = dict(self._client_kwargs)
            if self.path is not None:
                kwargs["path"] = self.path
            else:
                kwargs["url"] = self.url
                if self.api_key:
                    kwargs["api_key"] = self.api_key
            self.client = QdrantClient(**kwargs)
            self._models = models
            self.client_error = ""
        except Exception as exc:
            self.client = None
            self._models = None
            self.client_error = redact_secrets(safe_error(exc))

    def _need_client(self) -> Any:
        if self.client is None and self._connect_pending:
            self._connect_pending = False
            self._connect()
        if self.client is None:
            detail = f": {self.client_error}" if self.client_error else ""
            raise RuntimeError(f"Qdrant backend {self.name!r} is unavailable{detail}")
        return self.client

    def _refresh_workspace(self) -> None:
        try:
            self._close_client()
        finally:
            self._schemas.clear()
            self._owned_fields.clear()
            self._collections.clear()
            self.path = self._local_path()
            self._connect_pending = True
            self.client_error = ""

    def _close_client(self) -> None:
        client = self.client
        self.client = None
        self._models = None
        close = getattr(client, "close", None)
        if callable(close):
            close()

    def _workspace_prefix(self) -> str:
        return f"{self._scope()}_"

    def _scope(self) -> str:
        """Return this backend instance's bounded physical owner token."""
        return provider_scope(self.ws_id, self.name, self.identifier, max_length=96)

    def _local_path(self) -> str | None:
        """Return an isolated embedded-storage directory for this owner."""
        return pj(self._path_root, self._scope()) if self._path_root is not None else None

    def _collection(self, entity_id: str) -> str:
        return self._collections.setdefault(entity_id, scoped_provider_name(self._scope(), entity_id, max_length=255))

    def _point_id(self, entity_id: str, object_id: Any) -> str:
        return str(uuid5(NAMESPACE_URL, f"heavenbase:qdrant:{self._scope()}:{entity_id}:{object_id}"))

    def _listed_collections(self) -> set[str]:
        response = self._need_client().get_collections()
        items = response.get("collections", []) if isinstance(response, Mapping) else getattr(response, "collections", [])
        return {str(item.get("name") if isinstance(item, Mapping) else getattr(item, "name", item)) for item in (items or [])}

    def _has_collection(self, name: str) -> bool:
        client = self._need_client()
        exists = getattr(client, "collection_exists", None)
        if callable(exists):
            return bool(exists(name))
        return name in self._listed_collections()

    def _vector_fields(self, schema: EntitySchema) -> list[Any]:
        return [field for field in schema.fields.values() if type(field.dtype).identifier == "vector"]

    def _distance(self) -> Any:
        distance = getattr(self._models, "Distance")
        return getattr(distance, "COSINE")

    @staticmethod
    def _descriptor_value(value: Any) -> str:
        """Return a stable scalar representation for provider descriptor values."""
        return str(getattr(value, "value", value)).lower()

    def _validate_existing_collection(self, collection: str, vectors: Sequence[Any]) -> None:
        """Reject a Qdrant collection whose named-vector schema is incompatible."""
        getter = getattr(self._need_client(), "get_collection", None)
        if not callable(getter):
            raise RuntimeError("Qdrant collection validation requires client.get_collection")
        try:
            response = getter(collection_name=collection)
        except TypeError:
            response = getter(collection)
        config = self._attr(response, "config", {})
        params = self._attr(config, "params", {})
        actual = self._attr(params, "vectors", None)
        if not isinstance(actual, Mapping):
            raise RuntimeError(f"Qdrant collection {collection!r} does not expose named-vector metadata")
        expected = {field.name: field for field in vectors}
        mismatches: list[str] = []
        if set(actual) != set(expected):
            mismatches.append(f"vectors expected {sorted(expected)} got {sorted(actual)}")
        expected_distance = self._descriptor_value(self._distance())
        for field_name in sorted(set(actual) & set(expected)):
            descriptor = actual[field_name]
            size = self._attr(descriptor, "size", None)
            distance = self._attr(descriptor, "distance", None)
            if size is None or int(size) != int(expected[field_name].dtype.dim):
                mismatches.append(f"{field_name}.size expected {expected[field_name].dtype.dim!r} got {size!r}")
            if distance is None or self._descriptor_value(distance) != expected_distance:
                mismatches.append(f"{field_name}.distance expected {expected_distance!r} got {distance!r}")
        if mismatches:
            raise RuntimeError(f"Qdrant collection {collection!r} does not match HeavenBase physical schema: {mismatches}")

    @backend_locked
    def ensure(self, schema: EntitySchema, bindings: Sequence[Any]) -> None:
        """Ensure one workspace-scoped Qdrant collection with named vectors.

        Args:
            schema (EntitySchema): Entity schema to store in Qdrant.
            bindings (Sequence[Any]): Storage bindings supplied by the workspace.

        Returns:
            None: This function does not return a value.

        Raises:
            ValueError: If routed fields omit a required Vector field or use
                dimensionless vectors.
        """
        owned_fields = {binding.field_name for binding in bindings if binding.backend == self.name and binding.field_name != schema.pk_name}
        vectors = [field for field in self._vector_fields(schema) if field.name in owned_fields]
        if owned_fields and not vectors:
            raise ValueError(
                f"Qdrant backend {self.name!r} cannot store {schema.entity_id!r} without a routed Vector field; " f"routed fields: {sorted(owned_fields)}"
            )
        if any(field.dtype.dim is None for field in vectors):
            raise ValueError("Qdrant native vector search requires fixed-dimension Vector fields")
        self._schemas[schema.entity_id] = schema
        self._owned_fields[schema.entity_id] = owned_fields
        if not owned_fields:
            return
        collection = self._collection(schema.entity_id)
        if self._has_collection(collection):
            self._validate_existing_collection(collection, vectors)
            return
        vector_params = {field.name: self._models.VectorParams(size=field.dtype.dim, distance=self._distance()) for field in vectors}
        try:
            self._need_client().create_collection(collection_name=collection, vectors_config=vector_params)
        except Exception:
            if not self._has_collection(collection):
                raise
            self._validate_existing_collection(collection, vectors)

    def _payload_and_vectors(self, entity_id: str, row: RowOp) -> tuple[dict[str, Any], dict[str, Any]]:
        item = row.to_row(fields=self._owned_fields.get(entity_id))
        item["object_id"] = str(row.object_id)
        vectors: dict[str, Any] = {}
        schema = self._schemas.get(entity_id)
        if schema is not None:
            for field in self._vector_fields(schema):
                value = item.pop(field.name, None)
                if value is not None:
                    vectors[field.name] = list(value)
        return item, vectors

    @backend_locked
    def upsert(self, rows: Sequence[RowOp]) -> None:
        """Apply ordered replacement upserts and idempotent deletes.

        Args:
            rows (Sequence[RowOp]): Row mutations to apply in caller order.

        Returns:
            None: This function does not return a value.
        """
        client = self._need_client()
        for entity_id, is_delete, ops in self.ordered_row_op_batches(rows):
            collection = self._collection(entity_id)
            if not self._has_collection(collection):
                if not is_delete:
                    raise RuntimeError(f"Qdrant collection {collection!r} disappeared after ensure; " f"refusing to discard {len(ops)} upsert operation(s)")
                continue
            if is_delete:
                selector = self._models.PointIdsList(points=[self._point_id(entity_id, row.object_id) for row in ops])
                client.delete(collection_name=collection, points_selector=selector, wait=True)
                continue
            points = []
            for row in ops:
                payload, vectors = self._payload_and_vectors(entity_id, row)
                points.append(
                    self._models.PointStruct(
                        id=self._point_id(entity_id, row.object_id),
                        vector=vectors,
                        payload=payload,
                    )
                )
            client.upsert(collection_name=collection, points=points, wait=True)

    def _attr(self, value: Any, name: str, default: Any = None) -> Any:
        return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)

    def _row_from_point(self, entity_id: str, point: Any, *, include_score: bool = False) -> dict[str, Any]:
        row = dict(self._attr(point, "payload", {}) or {})
        vector = self._attr(point, "vector", None)
        if isinstance(vector, Mapping):
            row.update({name: list(value) for name, value in vector.items() if value is not None})
        elif vector is not None:
            fields = self._vector_fields(self._schemas[entity_id]) if entity_id in self._schemas else []
            if len(fields) == 1:
                row[fields[0].name] = list(vector)
        row["object_id"] = str(row.get("object_id") or self._attr(point, "id", ""))
        if include_score:
            row["score"] = float(self._attr(point, "score", 0.0) or 0.0)
        return row

    def _scroll(self, entity_id: str, query_filter: Any = None) -> list[dict[str, Any]]:
        collection = self._collection(entity_id)
        if not self._has_collection(collection):
            return []
        client = self._need_client()
        offset = None
        seen_offsets: set[str] = set()
        rows: list[dict[str, Any]] = []
        while True:
            page = client.scroll(
                collection_name=collection,
                scroll_filter=query_filter,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )
            if isinstance(page, tuple):
                points, next_offset = page
            else:
                points = self._attr(page, "points", [])
                next_offset = self._attr(page, "next_page_offset", None)
            rows.extend(self._row_from_point(entity_id, point) for point in (points or []))
            if next_offset is None:
                break
            marker = repr(next_offset)
            if marker in seen_offsets:
                raise RuntimeError(f"Qdrant scan repeated page offset {next_offset!r}")
            seen_offsets.add(marker)
            offset = next_offset
        return rows

    @backend_locked
    def scan(self, entity_id: str) -> list[dict[str, Any]]:
        """Return every Qdrant point by exhausting scroll pagination.

        Args:
            entity_id (str): Entity identifier whose collection should be scanned.

        Returns:
            list[dict[str, Any]]: Stored rows reconstructed from Qdrant points.

        Raises:
            RuntimeError: If the Qdrant scan request fails.
        """
        try:
            return self._scroll(entity_id)
        except Exception as exc:
            self.client_error = redact_secrets(safe_error(exc))
            raise RuntimeError(f"Qdrant scan failed for {entity_id!r}: {self.client_error}") from exc

    @backend_locked
    def get_many(self, entity_id: str, object_ids: Iterable[Any], *, columns: Sequence[str] | None = None) -> list[dict[str, Any] | None]:
        """Return rows aligned with caller-provided object identifiers.

        Args:
            entity_id (str): Entity identifier whose rows should be loaded.
            object_ids (Iterable[Any]): Object identifiers to fetch in caller order.
            columns (Sequence[str] | None): Optional field names to return; ``None`` returns the complete fragment.

        Returns:
            list[dict[str, Any] | None]: Rows aligned with ``object_ids``. Missing
            points are returned as ``None``.
        """
        ordered_ids = [str(value) for value in object_ids]
        if not ordered_ids:
            return []
        collection = self._collection(entity_id)
        if not self._has_collection(collection):
            return [None for _ in ordered_ids]
        points = self._need_client().retrieve(
            collection_name=collection,
            ids=[self._point_id(entity_id, value) for value in dict.fromkeys(ordered_ids)],
            with_payload=True,
            with_vectors=True,
        )
        rows = [self._row_from_point(entity_id, point) for point in (points or [])]
        found = {str(row["object_id"]): row for row in rows}
        return self._project_rows([found.get(value) for value in ordered_ids], columns)

    @backend_locked
    def exists_many(self, entity_id: str, object_ids: Iterable[Any]) -> list[bool]:
        """Return existence flags aligned with caller-provided identifiers.

        Args:
            entity_id (str): Entity identifier whose rows should be checked.
            object_ids (Iterable[Any]): Object identifiers to check in caller order.

        Returns:
            list[bool]: Existence flags aligned with ``object_ids``.
        """
        return [row is not None for row in self.get_many(entity_id, object_ids)]

    @backend_locked
    def count(self, entity_id: str) -> int:
        """Return Qdrant's exact collection count.

        Args:
            entity_id (str): Entity identifier whose collection should be counted.

        Returns:
            int: Exact number of Qdrant points for the entity.
        """
        collection = self._collection(entity_id)
        if not self._has_collection(collection):
            return 0
        response = self._need_client().count(collection_name=collection, exact=True)
        return int(self._attr(response, "count", 0))

    def exact_count(self, entity_id: str) -> int:
        """Return Qdrant's exact count as an integrity proof.

        Args:
            entity_id (str): Logical entity identifier.

        Returns:
            int: Exact Qdrant point count suitable for integrity proofs.
        """
        return self.count(entity_id)

    @backend_locked
    def drop(self) -> None:
        """Drop only collections owned by this workspace.

        Args:
            None.

        Returns:
            None: This function does not return a value.
        """
        try:
            known = set(self._collections.values())
            prefix = self._workspace_prefix()
            if prefix:
                known.update(name for name in self._listed_collections() if name.startswith(prefix))
            client = self._need_client()
            for collection in sorted(known):
                if self._has_collection(collection):
                    client.delete_collection(collection_name=collection)
        finally:
            self._schemas.clear()
            self._owned_fields.clear()
            self._collections.clear()
            self._close_client()
            self._connect_pending = False

    def _is_qdrant_fragment(self, fragment: QueryFragment) -> bool:
        if fragment.kind() == "qdrant_filter":
            return True
        if fragment.op in ("and", "or"):
            return all(self._is_qdrant_fragment(child) for child in fragment.children())
        if fragment.op == "not":
            children = fragment.children()
            return len(children) == 1 and self._is_qdrant_fragment(children[0])
        return False

    def _field_condition(self, payload: Mapping[str, Any]) -> Any:
        field = str(payload["field"])
        op = str(payload["op"])
        value = payload["value"]
        if op in ("eq", "array_contains"):
            return self._models.FieldCondition(key=field, match=self._models.MatchValue(value=value))
        if op == "in":
            return self._models.FieldCondition(key=field, match=self._models.MatchAny(any=list(value)))
        ranges = {
            "lt": {"lt": value},
            "lte": {"lte": value},
            "gt": {"gt": value},
            "gte": {"gte": value},
        }
        if op in ranges:
            return self._models.FieldCondition(key=field, range=self._models.Range(**ranges[op]))
        raise TypeError(f"Cannot compile {op!r} to a Qdrant condition")

    def _filter_condition(self, fragment: QueryFragment) -> Any:
        if fragment.kind() == "qdrant_filter":
            return self._field_condition(fragment.payload)
        children = [self._filter_condition(child) for child in fragment.children()]
        if fragment.op == "and":
            return self._models.Filter(must=children)
        if fragment.op == "or":
            return self._models.Filter(should=children)
        if fragment.op == "not":
            return self._models.Filter(must_not=children)
        raise TypeError(f"Cannot compile {fragment.op!r} to a Qdrant filter")

    def _filter(self, fragment: QueryFragment) -> Any:
        condition = self._filter_condition(fragment)
        return condition if isinstance(condition, self._models.Filter) else self._models.Filter(must=[condition])

    def candidate_id_fragment(self, object_ids: Sequence[Any]) -> QueryFragment:
        """Return a native payload identifier filter for candidate preselection.

        Args:
            object_ids (Sequence[Any]): Candidate object identifiers.

        Returns:
            QueryFragment: Native Qdrant payload identifier filter.
        """
        py_fragment = super().candidate_id_fragment(object_ids)
        return QueryFragment(
            self.name,
            "in",
            "object_id",
            {"op": "in", "field": "object_id", "value": [str(value) for value in object_ids]},
            notes={**py_fragment.notes, "kind": "qdrant_filter", "mode": "native"},
        )

    @backend_locked
    def execute(self, fragment: QueryFragment) -> ResultFrame:
        """Execute exact Qdrant filters and cosine vector search.

        Args:
            fragment (QueryFragment): Backend-ready query fragment compiled by the handler layer.

        Returns:
            ResultFrame: Matching rows and object identifiers.
        """
        if fragment.mode() == "py_scan":
            return super().execute(fragment)
        entity_id = fragment.entity_id()
        collection = self._collection(entity_id) if entity_id else ""
        if entity_id and not self._has_collection(collection):
            return ResultFrame.empty()
        if fragment.kind() == "qdrant_search" and fragment.op == "near":
            filt = fragment.filter_fragment()
            if filt is not None and not self._is_qdrant_fragment(filt):
                return super().execute(fragment)
            params = fragment.payload
            try:
                response = self._need_client().query_points(
                    collection_name=collection,
                    # Qdrant collections use cosine over float32 vectors; cosine ignores the exact power-of-two scale.
                    query=_scale_cosine_query(params["vector"]),
                    using=fragment.field,
                    query_filter=self._filter(filt) if filt is not None else None,
                    limit=params["top_k"],
                    with_payload=True,
                    with_vectors=True,
                )
                points = self._attr(response, "points", response if isinstance(response, Sequence) else [])
                return ResultFrame.from_rows([self._row_from_point(entity_id, point, include_score=True) for point in (points or [])])
            except Exception as exc:
                self.client_error = redact_secrets(safe_error(exc))
                raise RuntimeError(f"Qdrant vector query failed for {entity_id!r}: {self.client_error}") from exc
        if fragment.kind() == "qdrant_filter" or fragment.op in LOGICAL_OPS:
            if not self._is_qdrant_fragment(fragment):
                return super().execute(fragment)
            try:
                return ResultFrame.from_rows(self._scroll(entity_id, self._filter(fragment)))
            except Exception as exc:
                self.client_error = redact_secrets(safe_error(exc))
                raise RuntimeError(f"Qdrant filter query failed for {entity_id!r}: {self.client_error}") from exc
        raise NotImplementedError(f"Qdrant cannot execute fragment {fragment.op!r}")

    def _live_tag_claims(self) -> dict[str, Any]:
        availability = "ready" if self.client is not None else ("unknown" if self._connect_pending else "unavailable")
        return {
            **super()._live_tag_claims(),
            "backend-availability": availability,
        }

    def to_dict(self) -> dict[str, Any]:
        """Return Qdrant connection metadata without credentials.

        Args:
            None.

        Returns:
            dict[str, Any]: Backend metadata and connection state without credentials.
        """
        return {
            "name": self.name,
            "type": self.type,
            "deployment": "local" if self.path is not None else "remote",
            "connected": self.client is not None,
        }
