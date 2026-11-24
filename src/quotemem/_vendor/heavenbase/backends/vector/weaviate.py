"""Weaviate vector backend using the synchronous Python v4 client."""

from __future__ import annotations

__all__ = ["WeaviateBackend"]

from urllib.parse import urlparse
from uuid import NAMESPACE_URL, uuid5

from heavenbase.execution.contracts import LOGICAL_OPS, QueryFragment, RowOp
from heavenbase.utils import Any, Iterable, Mapping, Sequence, safe_error
from heavenbase.utils.redact import redact_secrets

from ...execution.frame.result import ResultFrame
from ...modules import ModuleResolver
from ...schema.model import EntitySchema
from ...types.storage import stores_as_integer
from ...types.vector import _scale_cosine_query
from .._scope import provider_scope, scoped_provider_name
from ..base import Backend, backend_locked


class WeaviateBackend(Backend):
    """Weaviate adapter with self-provided named vectors and exact payload rows."""

    identifier = "weaviate"

    @classmethod
    def from_config(
        cls,
        name: str,
        data: Mapping[str, Any],
        *,
        resolver: ModuleResolver | None = None,
        config: Any | None = None,
    ) -> "WeaviateBackend":
        """Build a Weaviate backend from a config mapping.

        Args:
            name: Workspace backend instance name.
            data: Backend config mapping. Supports ``url``, ``grpc_host``,
                ``grpc_port``, ``api_key``, optional ``ws_id``, and Weaviate client
                keyword arguments.
            resolver: Context-owned module authority.
            config: Config authority retained by the backend.

        Returns:
            WeaviateBackend: Configured Weaviate backend instance.

        Raises:
            TypeError: If ``grpc_port`` cannot be converted to an integer.
            ValueError: If ``grpc_port`` is not a valid integer value.
        """
        cfg = dict(data or {})
        ws_id = cfg.pop("ws_id", None)
        return cls(
            name,
            cfg.pop("url", None),
            grpc_host=cfg.pop("grpc_host", None),
            grpc_port=cfg.pop("grpc_port", None),
            api_key=cfg.pop("api_key", None),
            ws_id=ws_id,
            resolver=resolver,
            config=config,
            **cfg,
        )

    def __init__(
        self,
        name: str = "weaviate",
        url: str | None = None,
        *,
        grpc_host: str | None = None,
        grpc_port: int | None = None,
        api_key: str | None = None,
        ws_id: str | None = None,
        resolver: ModuleResolver | None = None,
        config: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """Create a lazily connected Weaviate backend.

        Args:
            name: Workspace backend instance name.
            url: Weaviate HTTP endpoint. Defaults to backend configuration.
            grpc_host: Optional gRPC host override. The HTTP host is used by default.
            grpc_port: Weaviate gRPC port. Defaults to backend configuration.
            api_key: Optional Weaviate authentication key.
            ws_id: Optional workspace namespace for collection ownership.
            resolver: Context-owned module authority.
            config: Config authority retained by the backend.
            **kwargs: Extra keyword arguments passed to ``connect_to_custom``.

        Returns:
            None: This function does not return a value.

        Raises:
            TypeError: If ``grpc_port`` cannot be converted to an integer.
            ValueError: If ``grpc_port`` is not a valid integer value.
        """
        super().__init__(name, ws_id=ws_id, resolver=resolver, config=config)
        self.url = str(
            url
            if url is not None
            else self._construction_config_get(
                "heavenbase.backend_defaults.weaviate.url",
                default="http://localhost:8080",
            )
        )
        self.grpc_host = grpc_host if grpc_host is not None else self._construction_config_get("heavenbase.backend_defaults.weaviate.grpc_host", default=None)
        configured_grpc_port = (
            grpc_port if grpc_port is not None else self._construction_config_get("heavenbase.backend_defaults.weaviate.grpc_port", default=50051)
        )
        self.grpc_port = int(configured_grpc_port)
        self.api_key = api_key if api_key is not None else self._construction_config_get("heavenbase.backend_defaults.weaviate.api_key", default=None)
        self._client_kwargs = dict(kwargs)
        self._schemas: dict[str, EntitySchema] = {}
        self._owned_fields: dict[str, set[str]] = {}
        self._collections: dict[str, str] = {}
        self.client = None
        self.client_error = ""
        self._wvc = None
        self._connect_pending = True

    def _connect(self) -> None:
        try:
            import weaviate
            import weaviate.classes as wvc
            from weaviate.classes.init import Auth

            parsed = urlparse(self.url if "://" in self.url else f"http://{self.url}")
            host = parsed.hostname or "localhost"
            secure = parsed.scheme == "https"
            kwargs = {
                "http_host": host,
                "http_port": parsed.port or (443 if secure else 80),
                "http_secure": secure,
                "grpc_host": self.grpc_host or host,
                "grpc_port": self.grpc_port,
                "grpc_secure": secure,
                **self._client_kwargs,
            }
            if self.api_key:
                kwargs["auth_credentials"] = Auth.api_key(self.api_key)
            self.client = weaviate.connect_to_custom(**kwargs)
            self._wvc = wvc
            self.client_error = ""
        except Exception as exc:
            self.client = None
            self._wvc = None
            self.client_error = redact_secrets(safe_error(exc))

    def _need_client(self) -> Any:
        if self.client is None and self._connect_pending:
            self._connect_pending = False
            self._connect()
        if self.client is None:
            detail = f": {self.client_error}" if self.client_error else ""
            raise RuntimeError(f"Weaviate backend {self.name!r} is unavailable{detail}")
        return self.client

    def _close_client(self) -> None:
        client = self.client
        self.client = None
        self._wvc = None
        close = getattr(client, "close", None)
        if callable(close):
            close()

    def _refresh_workspace(self) -> None:
        try:
            self._close_client()
        finally:
            self._schemas.clear()
            self._owned_fields.clear()
            self._collections.clear()
            self._connect_pending = True
            self.client_error = ""

    def _workspace_prefix(self) -> str:
        return f"{self._scope()}_"

    def _scope(self) -> str:
        """Return this backend instance's bounded physical owner token."""
        return provider_scope(self.ws_id, self.name, self.identifier, max_length=48)

    def _collection(self, entity_id: str) -> str:
        physical = scoped_provider_name(self._scope(), entity_id, max_length=125)
        return self._collections.setdefault(entity_id, f"Hb_{physical}")

    def _object_uuid(self, entity_id: str, object_id: Any) -> str:
        return str(uuid5(NAMESPACE_URL, f"heavenbase:weaviate:{self._scope()}:{entity_id}:{object_id}"))

    def _collection_handle(self, entity_id: str) -> Any:
        return self._need_client().collections.use(self._collection(entity_id))

    def _has_collection(self, name: str) -> bool:
        return bool(self._need_client().collections.exists(name))

    def _vector_fields(self, schema: EntitySchema) -> list[Any]:
        return [field for field in schema.fields.values() if type(field.dtype).identifier == "vector"]

    def _data_type(self, dtype: Any) -> tuple[Any, Any | None]:
        data_type = self._wvc.config.DataType
        tokenization = None
        identifier = type(dtype).identifier
        if identifier in ("identifier", "short-text", "medium-text", "long-text", "artifact", "date"):
            tokenization = self._wvc.config.Tokenization.FIELD
            return data_type.TEXT, tokenization
        if stores_as_integer(dtype):
            return data_type.INT, tokenization
        if identifier == "float":
            return data_type.NUMBER, tokenization
        if identifier == "boolean":
            return data_type.BOOL, tokenization
        if identifier == "categorical":
            if getattr(dtype, "base", None) == "string":
                tokenization = self._wvc.config.Tokenization.FIELD
                return data_type.TEXT, tokenization
            return data_type.INT, tokenization
        if identifier == "array":
            item_type = getattr(dtype, "item_type", None)
            if item_type is None:
                raise TypeError("Weaviate requires a concrete Array item type")
            item_identifier = type(item_type).identifier
            if item_identifier in ("identifier", "short-text", "medium-text", "long-text", "date"):
                tokenization = self._wvc.config.Tokenization.FIELD
                return data_type.TEXT_ARRAY, tokenization
            if stores_as_integer(item_type):
                return data_type.INT_ARRAY, tokenization
            if item_identifier == "float":
                return data_type.NUMBER_ARRAY, tokenization
            if item_identifier == "boolean":
                return data_type.BOOL_ARRAY, tokenization
        raise TypeError(f"Weaviate does not support routed logical type {identifier!r}")

    def _properties(self, schema: EntitySchema, owned_fields: set[str]) -> list[Any]:
        config = self._wvc.config
        properties = [
            config.Property(
                name="object_id",
                data_type=config.DataType.TEXT,
                tokenization=config.Tokenization.FIELD,
            )
        ]
        for field in schema.fields.values():
            if field.name not in owned_fields or field.name == "object_id" or type(field.dtype).identifier == "vector":
                continue
            data_type, tokenization = self._data_type(field.dtype)
            kwargs = {"name": field.name, "data_type": data_type}
            if tokenization is not None:
                kwargs["tokenization"] = tokenization
            properties.append(config.Property(**kwargs))
        return properties

    @staticmethod
    def _descriptor_value(value: Any) -> str:
        """Return a stable representation for Weaviate descriptor values."""
        if isinstance(value, (list, tuple)) and len(value) == 1:
            value = value[0]
        value = getattr(value, "value", value)
        return str(value).split(".")[-1].lower()

    @staticmethod
    def _config_attr(value: Any, name: str, default: Any = None) -> Any:
        """Read one descriptor attribute from object or mapping responses."""
        return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)

    def _validate_existing_collection(
        self,
        schema: EntitySchema,
        collection_name: str,
        owned_fields: set[str],
        vectors: Sequence[Any],
    ) -> None:
        """Reject an existing Weaviate collection with incompatible physical schema."""
        collection = self._collection_handle(schema.entity_id)
        getter = getattr(getattr(collection, "config", None), "get", None)
        if not callable(getter):
            raise RuntimeError("Weaviate collection validation requires collection.config.get")
        descriptor = getter()
        actual_properties = {str(self._config_attr(prop, "name")): prop for prop in (self._config_attr(descriptor, "properties", []) or [])}
        expected_properties = {str(self._config_attr(prop, "name")): prop for prop in self._properties(schema, owned_fields)}
        actual_vectors = self._config_attr(descriptor, "vector_config", {}) or {}
        if isinstance(actual_vectors, Mapping):
            actual_vector_descriptors = {str(name): value for name, value in actual_vectors.items()}
            actual_vector_names = {str(name) for name in actual_vectors}
        else:
            actual_vector_descriptors = {
                str(self._config_attr(vector, "name")): vector for vector in actual_vectors if self._config_attr(vector, "name", None) is not None
            }
            actual_vector_names = set(actual_vector_descriptors)
        expected_vector_names = {field.name for field in vectors}
        mismatches: list[str] = []
        if set(actual_properties) != set(expected_properties):
            mismatches.append(f"properties expected {sorted(expected_properties)} got {sorted(actual_properties)}")
        if actual_vector_names != expected_vector_names:
            mismatches.append(f"vectors expected {sorted(expected_vector_names)} got {sorted(actual_vector_names)}")
        for field_name in sorted(actual_vector_names & expected_vector_names):
            vector_descriptor = actual_vector_descriptors[field_name]
            vectorizer_config = self._config_attr(vector_descriptor, "vectorizer", None)
            vectorizer = self._config_attr(vectorizer_config, "vectorizer", vectorizer_config)
            if self._descriptor_value(vectorizer) != "none":
                mismatches.append(f"{field_name}.vectorizer expected 'none' got {self._descriptor_value(vectorizer)!r}")
            index_config = self._config_attr(
                vector_descriptor,
                "vector_index_config",
                self._config_attr(vector_descriptor, "vectorIndexConfig", None),
            )
            index_type = self._config_attr(
                vector_descriptor,
                "vector_index_type",
                self._config_attr(vector_descriptor, "vectorIndexType", None),
            )
            if index_type is None and index_config is not None:
                index_type_name = type(index_config).__name__.lower()
                index_type = "hnsw" if "hnsw" in index_type_name else None
            if self._descriptor_value(index_type) != "hnsw":
                mismatches.append(f"{field_name}.index_type expected 'hnsw' got {self._descriptor_value(index_type)!r}")
            distance = self._config_attr(index_config, "distance_metric", self._config_attr(index_config, "distance", None))
            if self._descriptor_value(distance) != "cosine":
                mismatches.append(f"{field_name}.distance expected 'cosine' got {self._descriptor_value(distance)!r}")
        for field_name in sorted(set(actual_properties) & set(expected_properties)):
            actual = actual_properties[field_name]
            expected = expected_properties[field_name]
            for attribute in ("data_type", "tokenization"):
                actual_value = self._config_attr(actual, attribute, None)
                expected_value = self._config_attr(expected, attribute, None)
                if expected_value is None and actual_value is None:
                    continue
                if self._descriptor_value(actual_value) != self._descriptor_value(expected_value):
                    mismatches.append(
                        f"{field_name}.{attribute} expected {self._descriptor_value(expected_value)!r} " f"got {self._descriptor_value(actual_value)!r}"
                    )
        if not mismatches:
            unresolved = {field.name: field for field in vectors}
            for item in collection.iterator(include_vector=True):
                actual_values = self._config_attr(item, "vector", {})
                if not isinstance(actual_values, Mapping):
                    continue
                for field_name in list(unresolved):
                    value = actual_values.get(field_name)
                    if value is None:
                        continue
                    field = unresolved.pop(field_name)
                    if len(value) != int(field.dtype.dim):
                        mismatches.append(f"{field.name}.dim expected {field.dtype.dim!r} got {len(value)!r}")
                if not unresolved:
                    break
        if mismatches:
            raise RuntimeError(f"Weaviate collection {collection_name!r} does not match HeavenBase physical schema: {mismatches}")

    @backend_locked
    def ensure(self, schema: EntitySchema, bindings: Sequence[Any]) -> None:
        """Ensure one workspace-scoped collection with self-provided named vectors.

        Args:
            schema (EntitySchema): Entity schema to store in Weaviate.
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
                f"Weaviate backend {self.name!r} cannot store {schema.entity_id!r} without a routed Vector field; " f"routed fields: {sorted(owned_fields)}"
            )
        if any(field.dtype.dim is None for field in vectors):
            raise ValueError("Weaviate native vector search requires fixed-dimension Vector fields")
        self._schemas[schema.entity_id] = schema
        self._owned_fields[schema.entity_id] = owned_fields
        if not owned_fields:
            return
        name = self._collection(schema.entity_id)
        if self._has_collection(name):
            self._validate_existing_collection(schema, name, owned_fields, vectors)
            return
        vector_config = [
            self._wvc.config.Configure.Vectors.self_provided(
                name=field.name,
                vector_index_config=self._wvc.config.Configure.VectorIndex.hnsw(
                    distance_metric=self._wvc.config.VectorDistances.COSINE,
                ),
            )
            for field in vectors
        ]
        try:
            self._need_client().collections.create(
                name=name,
                vector_config=vector_config,
                properties=self._properties(schema, owned_fields),
            )
        except Exception:
            if not self._has_collection(name):
                raise
            self._validate_existing_collection(schema, name, owned_fields, vectors)

    def _properties_and_vectors(self, entity_id: str, row: RowOp) -> tuple[dict[str, Any], dict[str, Any]]:
        source = row.to_row()
        properties = {"object_id": str(row.object_id)}
        vectors: dict[str, Any] = {}
        owned_fields = self._owned_fields.get(entity_id, set(source))
        schema = self._schemas.get(entity_id)
        if schema is not None:
            for field in self._vector_fields(schema):
                if field.name not in owned_fields:
                    continue
                value = source.get(field.name)
                if value is not None:
                    vectors[field.name] = list(value)
        for field_name in owned_fields:
            if field_name == "object_id" or field_name in vectors or field_name not in source:
                continue
            properties[field_name] = source[field_name]
        return properties, vectors

    @backend_locked
    def upsert(self, rows: Sequence[RowOp]) -> None:
        """Apply ordered full-row replacements and idempotent deletes.

        Args:
            rows (Sequence[RowOp]): Row mutations to apply in caller order.

        Returns:
            None: This function does not return a value.
        """
        self._need_client()
        for entity_id, is_delete, ops in self.ordered_row_op_batches(rows):
            name = self._collection(entity_id)
            if not self._has_collection(name):
                if not is_delete:
                    raise RuntimeError(f"Weaviate collection {name!r} disappeared after ensure; " f"refusing to discard {len(ops)} upsert operation(s)")
                continue
            collection = self._collection_handle(entity_id)
            if is_delete:
                for row in ops:
                    collection.data.delete_by_id(self._object_uuid(entity_id, row.object_id))
                continue
            for row in ops:
                object_uuid = self._object_uuid(entity_id, row.object_id)
                properties, vectors = self._properties_and_vectors(entity_id, row)
                if collection.data.exists(object_uuid):
                    try:
                        collection.data.replace(uuid=object_uuid, properties=properties, vector=vectors)
                    except Exception as exc:
                        if not self._is_missing_error(exc):
                            raise
                        try:
                            collection.data.insert(uuid=object_uuid, properties=properties, vector=vectors)
                        except Exception as retry_exc:
                            if not self._is_conflict_error(retry_exc):
                                raise
                            collection.data.replace(uuid=object_uuid, properties=properties, vector=vectors)
                else:
                    try:
                        collection.data.insert(uuid=object_uuid, properties=properties, vector=vectors)
                    except Exception as exc:
                        if not self._is_conflict_error(exc):
                            raise
                        collection.data.replace(uuid=object_uuid, properties=properties, vector=vectors)

    @staticmethod
    def _error_status(exc: Exception) -> int | None:
        """Return a provider HTTP status when exposed by a client exception."""
        for value in (getattr(exc, "status_code", None), getattr(getattr(exc, "response", None), "status_code", None)):
            if value is not None:
                try:
                    return int(value)
                except (TypeError, ValueError):
                    return None
        return None

    @classmethod
    def _is_missing_error(cls, exc: Exception) -> bool:
        """Return whether one provider write failed because the object disappeared."""
        message = safe_error(exc).lower()
        return cls._error_status(exc) == 404 or "not found" in message

    @classmethod
    def _is_conflict_error(cls, exc: Exception) -> bool:
        """Return whether one provider insert raced with an existing object."""
        message = safe_error(exc).lower()
        return cls._error_status(exc) in (409, 422) or "already exists" in message or "conflict" in message

    def _object_row(self, entity_id: str, item: Any, *, include_score: bool = False) -> dict[str, Any]:
        properties = getattr(item, "properties", None)
        if properties is None and isinstance(item, Mapping):
            properties = item.get("properties", {})
        row = dict(properties or {})
        vector = getattr(item, "vector", None)
        if vector is None and isinstance(item, Mapping):
            vector = item.get("vector")
        if isinstance(vector, Mapping):
            row.update({name: list(value) for name, value in vector.items() if value is not None})
        elif vector is not None:
            fields = self._vector_fields(self._schemas[entity_id]) if entity_id in self._schemas else []
            if len(fields) == 1:
                row[fields[0].name] = list(vector)
        row["object_id"] = str(row.get("object_id") or getattr(item, "uuid", ""))
        if include_score:
            metadata = getattr(item, "metadata", None)
            distance = getattr(metadata, "distance", None) if metadata is not None else None
            row["score"] = 1.0 - float(distance or 0.0)
        return row

    @backend_locked
    def scan(self, entity_id: str) -> list[dict[str, Any]]:
        """Return every object through Weaviate's cursor-backed iterator.

        Args:
            entity_id (str): Entity identifier whose collection should be scanned.

        Returns:
            list[dict[str, Any]]: Stored rows reconstructed from Weaviate objects.
        """
        name = self._collection(entity_id)
        if not self._has_collection(name):
            return []
        try:
            return [self._object_row(entity_id, item) for item in self._collection_handle(entity_id).iterator(include_vector=True)]
        except Exception as exc:
            self.client_error = redact_secrets(safe_error(exc))
            raise RuntimeError(f"Weaviate scan failed for {entity_id!r}: {self.client_error}") from exc

    @backend_locked
    def get_many(self, entity_id: str, object_ids: Iterable[Any], *, columns: Sequence[str] | None = None) -> list[dict[str, Any] | None]:
        """Return rows aligned with caller-provided object identifiers.

        Args:
            entity_id (str): Entity identifier whose rows should be loaded.
            object_ids (Iterable[Any]): Object identifiers to fetch in caller order.
            columns (Sequence[str] | None): Optional field names to return; ``None`` returns the complete fragment.

        Returns:
            list[dict[str, Any] | None]: Rows aligned with ``object_ids``. Missing
            objects are returned as ``None``.
        """
        ordered_ids = [str(value) for value in object_ids]
        if not ordered_ids:
            return []
        name = self._collection(entity_id)
        if not self._has_collection(name):
            return [None for _ in ordered_ids]
        collection = self._collection_handle(entity_id)
        out: list[dict[str, Any] | None] = []
        for object_id in ordered_ids:
            item = collection.query.fetch_object_by_id(
                uuid=self._object_uuid(entity_id, object_id),
                include_vector=True,
            )
            out.append(None if item is None else self._object_row(entity_id, item))
        return self._project_rows(out, columns)

    @backend_locked
    def exists_many(self, entity_id: str, object_ids: Iterable[Any]) -> list[bool]:
        """Return existence flags aligned with caller-provided identifiers.

        Args:
            entity_id (str): Entity identifier whose rows should be checked.
            object_ids (Iterable[Any]): Object identifiers to check in caller order.

        Returns:
            list[bool]: Existence flags aligned with ``object_ids``.
        """
        ordered_ids = [str(value) for value in object_ids]
        if not ordered_ids:
            return []
        name = self._collection(entity_id)
        if not self._has_collection(name):
            return [False for _ in ordered_ids]
        data = self._collection_handle(entity_id).data
        return [bool(data.exists(self._object_uuid(entity_id, value))) for value in ordered_ids]

    @backend_locked
    def count(self, entity_id: str) -> int:
        """Return Weaviate's exact aggregate object count.

        Args:
            entity_id (str): Entity identifier whose collection should be counted.

        Returns:
            int: Exact number of Weaviate objects for the entity.
        """
        name = self._collection(entity_id)
        if not self._has_collection(name):
            return 0
        response = self._collection_handle(entity_id).aggregate.over_all(total_count=True)
        return int(getattr(response, "total_count", 0) or 0)

    def exact_count(self, entity_id: str) -> int:
        """Return the aggregate object count as an integrity proof.

        Args:
            entity_id (str): Logical entity identifier.

        Returns:
            int: Exact Weaviate object count suitable for integrity proofs.
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
            client = self._need_client()
            names = set(self._collections.values())
            prefix = f"Hb_{self._workspace_prefix()}"
            listed = client.collections.list_all(simple=True)
            listed_names = listed.keys() if isinstance(listed, Mapping) else listed
            names.update(str(name) for name in listed_names if str(name).startswith(prefix))
            for name in sorted(names):
                if self._has_collection(name):
                    client.collections.delete(name)
        finally:
            self._schemas.clear()
            self._owned_fields.clear()
            self._collections.clear()
            self._close_client()
            self._connect_pending = False

    def _is_weaviate_fragment(self, fragment: QueryFragment) -> bool:
        if fragment.kind() == "weaviate_filter":
            return True
        if fragment.op in ("and", "or"):
            return all(self._is_weaviate_fragment(child) for child in fragment.children())
        if fragment.op == "not":
            children = fragment.children()
            return len(children) == 1 and self._is_weaviate_fragment(children[0])
        return False

    def _leaf_filter(self, payload: Mapping[str, Any]) -> Any:
        builder = self._wvc.query.Filter.by_property(str(payload["field"]))
        op = str(payload["op"])
        value = payload["value"]
        methods = {
            "eq": "equal",
            "lt": "less_than",
            "lte": "less_or_equal",
            "gt": "greater_than",
            "gte": "greater_or_equal",
        }
        if op in methods:
            return getattr(builder, methods[op])(value)
        if op == "in":
            return builder.contains_any(list(value))
        if op == "array_contains":
            return builder.contains_any([value])
        raise TypeError(f"Cannot compile {op!r} to a Weaviate filter")

    def _filter(self, fragment: QueryFragment) -> Any:
        if fragment.kind() == "weaviate_filter":
            return self._leaf_filter(fragment.payload)
        children = [self._filter(child) for child in fragment.children()]
        if fragment.op == "and":
            return self._wvc.query.Filter.all_of(children)
        if fragment.op == "or":
            return self._wvc.query.Filter.any_of(children)
        if fragment.op == "not":
            return self._wvc.query.Filter.not_(children[0])
        raise TypeError(f"Cannot compile {fragment.op!r} to a Weaviate filter")

    def _filtered_rows(self, entity_id: str, query_filter: Any) -> list[dict[str, Any]]:
        collection = self._collection_handle(entity_id)
        rows: list[dict[str, Any]] = []
        after = None
        seen_cursors: set[str] = set()
        batch_size = 256
        while True:
            response = collection.query.fetch_objects(
                filters=query_filter,
                limit=batch_size,
                after=after,
                include_vector=True,
            )
            objects = list(getattr(response, "objects", []) or [])
            rows.extend(self._object_row(entity_id, item) for item in objects)
            if len(objects) < batch_size:
                return rows
            next_cursor = getattr(objects[-1], "uuid", None)
            if next_cursor is None:
                raise RuntimeError("Weaviate exhaustive filter pagination requires object UUID cursors")
            marker = str(next_cursor)
            if marker in seen_cursors:
                raise RuntimeError(f"Weaviate filter pagination repeated cursor {marker!r}")
            seen_cursors.add(marker)
            after = next_cursor

    def candidate_id_fragment(self, object_ids: Sequence[Any]) -> QueryFragment:
        """Return a native object-id filter for candidate preselection.

        Args:
            object_ids (Sequence[Any]): Candidate object identifiers.

        Returns:
            QueryFragment: Native Weaviate object-id filter.
        """
        py_fragment = super().candidate_id_fragment(object_ids)
        return QueryFragment(
            self.name,
            "in",
            "object_id",
            {"op": "in", "field": "object_id", "value": [str(value) for value in object_ids]},
            notes={**py_fragment.notes, "kind": "weaviate_filter", "mode": "native"},
        )

    @backend_locked
    def execute(self, fragment: QueryFragment) -> ResultFrame:
        """Execute cosine near and exact scalar filters through Weaviate.

        Args:
            fragment (QueryFragment): Backend-ready query fragment compiled by the handler layer.

        Returns:
            ResultFrame: Matching rows and object identifiers.
        """
        if fragment.mode() == "py_scan":
            return super().execute(fragment)
        entity_id = fragment.entity_id()
        collection_name = self._collection(entity_id) if entity_id else ""
        if entity_id and not self._has_collection(collection_name):
            return ResultFrame.empty()
        if fragment.kind() == "weaviate_search" and fragment.op == "near":
            filt = fragment.filter_fragment()
            if filt is not None and not self._is_weaviate_fragment(filt):
                return super().execute(fragment)
            params = fragment.payload
            try:
                metadata = self._wvc.query.MetadataQuery(distance=True)
                response = self._collection_handle(entity_id).query.near_vector(
                    # Weaviate collections use cosine over float32 vectors; cosine ignores the exact power-of-two scale.
                    near_vector=_scale_cosine_query(params["vector"]),
                    target_vector=fragment.field,
                    filters=self._filter(filt) if filt is not None else None,
                    limit=params["top_k"],
                    include_vector=True,
                    return_metadata=metadata,
                )
                objects = getattr(response, "objects", []) or []
                return ResultFrame.from_rows([self._object_row(entity_id, item, include_score=True) for item in objects])
            except Exception as exc:
                self.client_error = redact_secrets(safe_error(exc))
                raise RuntimeError(f"Weaviate vector query failed for {entity_id!r}: {self.client_error}") from exc
        if fragment.kind() == "weaviate_filter" or fragment.op in LOGICAL_OPS:
            if not self._is_weaviate_fragment(fragment):
                return super().execute(fragment)
            try:
                return ResultFrame.from_rows(self._filtered_rows(entity_id, self._filter(fragment)))
            except Exception as exc:
                self.client_error = redact_secrets(safe_error(exc))
                raise RuntimeError(f"Weaviate filter query failed for {entity_id!r}: {self.client_error}") from exc
        raise NotImplementedError(f"Weaviate cannot execute fragment {fragment.op!r}")

    def _live_tag_claims(self) -> dict[str, Any]:
        availability = "ready" if self.client is not None else ("unknown" if self._connect_pending else "unavailable")
        return {
            **super()._live_tag_claims(),
            "backend-availability": availability,
        }

    def to_dict(self) -> dict[str, Any]:
        """Return Weaviate connection metadata without credentials.

        Args:
            None.

        Returns:
            dict[str, Any]: Backend metadata and connection state without credentials.
        """
        return {
            "name": self.name,
            "type": self.type,
            "deployment": "remote",
            "connected": self.client is not None,
        }
