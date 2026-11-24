"""MongoDB document backend runtime."""

from __future__ import annotations

__all__ = ["MongoDBBackend"]

from heavenbase.utils import Any, Iterable, Mapping, Sequence, safe_error
from heavenbase.utils.hash import sha256hash
from heavenbase.utils.redact import redact_secrets

from heavenbase.backends._scope import provider_scope, scoped_provider_name
from heavenbase.backends.base import Backend, backend_locked
from heavenbase.execution.contracts import RowOp
from heavenbase.modules import ModuleResolver
from heavenbase.schema.model import EntitySchema

MONGODB_COLLECTION_MAX_LENGTH = 120
MONGODB_INDEXABLE_BSON_TYPES = ("bool", "date", "decimal", "double", "int", "long", "string")


class MongoDBBackend(Backend):
    """MongoDB provider adapter storing one native document per row fragment."""

    identifier = "mongodb"
    aliases = ("mongo",)
    provider_max_name_length = MONGODB_COLLECTION_MAX_LENGTH

    @classmethod
    def from_config(
        cls,
        name: str,
        data: Mapping[str, Any],
        *,
        resolver: ModuleResolver | None = None,
        config: Any | None = None,
    ) -> "MongoDBBackend":
        """Build a MongoDB backend from a config mapping.

        Args:
            name: Workspace backend instance name.
            data: Backend config mapping. Supports ``uri``, ``database``,
                optional ``ws_id``, and MongoClient keyword arguments.
            resolver: Context-owned module authority.
            config: Config authority retained by the backend.

        Returns:
            MongoDBBackend: Configured MongoDB backend instance.
        """
        cfg = dict(data or {})
        ws_id = cfg.pop("ws_id", None)
        return cls(
            name,
            cfg.pop("uri", None),
            database=cfg.pop("database", None),
            ws_id=ws_id,
            resolver=resolver,
            config=config,
            **cfg,
        )

    def __init__(
        self,
        name: str = "mongodb",
        uri: str | None = None,
        *,
        database: str | None = None,
        ws_id: str | None = None,
        resolver: ModuleResolver | None = None,
        config: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """Create a MongoDB backend.

        Args:
            name: Workspace backend instance name.
            uri: MongoDB connection URI. Defaults to backend configuration.
            database: Database used while unbound. A bound workspace receives
                its own normalized database name.
            ws_id: Optional workspace namespace.
            resolver: Context-owned module authority.
            config: Config authority retained by the backend.
            **kwargs: Extra keyword arguments passed to ``MongoClient``.

        Returns:
            None: This function does not return a value.
        """
        super().__init__(name, ws_id=ws_id, resolver=resolver, config=config)
        self.uri = uri or self._construction_config_get(
            "heavenbase.backend_defaults.mongodb.uri",
            default="mongodb://localhost:27017",
        )
        self._database_default = str(
            database
            or self._construction_config_get(
                "heavenbase.backend_defaults.mongodb.database",
                default="heavenbase",
            )
        )
        self._client_kwargs = dict(kwargs)
        self._collections: dict[str, str] = {}
        self._schemas: dict[str, EntitySchema] = {}
        self._planned_indexes: dict[str, tuple[tuple[tuple[str, ...], bool], ...]] = {}
        self.database = self._database_name()
        self.client = None
        self.db = None
        self.client_error = ""
        self._connect()

    def _database_name(self) -> str:
        """Return the current workspace-safe physical database name."""
        return self._physical_name(self.ws_id) if self.ws_id else self._physical_name(self._database_default)

    def _connect(self) -> None:
        """Create the lazy Mongo client and select the current database."""
        try:
            from pymongo import MongoClient

            self.client = MongoClient(self.uri, **self._client_kwargs)
            self.database = self._database_name()
            self.db = self.client[self.database]
            self.client_error = ""
        except Exception as exc:
            self.client = None
            self.db = None
            self.client_error = redact_secrets(safe_error(exc))

    def _refresh_workspace(self) -> None:
        """Rebind physical names when a workspace is assigned."""
        self._collections.clear()
        self._schemas.clear()
        self._planned_indexes.clear()
        self.database = self._database_name()
        if self.client is not None:
            try:
                self.db = self.client[self.database]
                self.client_error = ""
            except Exception as exc:
                self.db = None
                self.client_error = redact_secrets(safe_error(exc))

    def _need_client(self) -> None:
        """Raise a contextual error when the optional provider is unavailable."""
        if self.client is None or self.db is None:
            detail = f": {self.client_error}" if self.client_error else ""
            raise RuntimeError(f"MongoDB backend {self.name!r} is unavailable{detail}")

    def _collection_name(self, entity_id: str) -> str:
        """Return a backend-instance and entity scoped collection name."""
        return self._collections.setdefault(
            entity_id,
            scoped_provider_name(self._collection_scope(), entity_id, max_length=self.provider_max_name_length),
        )

    def _collection_scope(self) -> str:
        """Return an unambiguous collection-owner token."""
        return provider_scope(self.ws_id, self.name, self.identifier, max_length=48)

    def _collection(self, entity_id: str) -> Any:
        """Return the provider collection for one entity."""
        self._need_client()
        return self.db[self._collection_name(entity_id)]

    def schema_warnings(self, schema: EntitySchema, bindings: Sequence[Any]) -> list[str]:
        """Return provider naming warnings for a MongoDB entity schema.

        Args:
            schema: Entity schema being registered.
            bindings: Storage bindings routed to this backend.

        Returns:
            list[str]: Collection-length and unsupported field-name warnings.
        """
        warnings = super().schema_warnings(schema, bindings)
        warnings.extend(
            self._physical_name_warnings(
                [(schema.entity_id, self._collection_name(schema.entity_id))],
                kind="MongoDB collection",
                max_length=self.provider_max_name_length,
            )
        )
        for field_name in sorted(schema.fields):
            if field_name.startswith("$") or "." in field_name:
                warnings.append(
                    f"Backend {self.name!r} (mongodb) cannot safely map field {schema.entity_id}.{field_name}; "
                    "MongoDB document field names must not start with '$' or contain '.'"
                )
        return warnings

    @backend_locked
    def ensure(self, schema: EntitySchema, bindings: Sequence[Any]) -> None:
        """Ensure the entity collection and fixed identity index.

        Args:
            schema: Entity schema to store.
            bindings: Storage bindings routed to this backend.

        Returns:
            None: The schema is recorded and provider storage is prepared.
        """
        del bindings
        self._schemas[schema.entity_id] = schema
        if self.client is None or self.db is None:
            return
        try:
            collection = self._collection(schema.entity_id)
            collection.create_index([("object_id", 1)], unique=True, name="hb_object_id")
        except Exception as exc:
            self.client_error = redact_secrets(safe_error(exc))
            raise RuntimeError(f"MongoDB ensure failed for {schema.entity_id!r}: {self.client_error}") from exc

    @backend_locked
    def ensure_plan(self, schema: EntitySchema, storage_plan: Any) -> None:
        """Reconcile MongoDB indexes with one resolved storage plan.

        Args:
            schema: Entity schema owning the indexes.
            storage_plan: Resolved storage, index, and constraint plan.

        Returns:
            None: Missing non-unique and unique indexes are created.
        """
        unique_fields = {tuple(binding.fields) for binding in storage_plan.backend_constraints(self.name)}
        indexes = [(tuple(binding.fields), False) for binding in storage_plan.backend_indexes(self.name) if tuple(binding.fields) not in unique_fields]
        indexes.extend((fields, True) for fields in sorted(unique_fields))
        self._planned_indexes[schema.entity_id] = tuple(indexes)
        self.ensure(schema, storage_plan.bindings)
        if self.client is None or self.db is None:
            return
        try:
            collection = self._collection(schema.entity_id)
            self._reconcile_indexes(collection, indexes)
        except Exception as exc:
            self.client_error = redact_secrets(safe_error(exc))
            raise RuntimeError(f"MongoDB index ensure failed for {schema.entity_id!r}: {self.client_error}") from exc

    def _index_spec(self, fields: tuple[str, ...], unique: bool) -> tuple[str, list[tuple[str, int]], dict[str, Any]]:
        """Return one deterministic managed provider index specification."""
        prefix = "hb_unique" if unique else "hb_index"
        readable = self._physical_name(*fields).lower()
        digest = sha256hash({"role": prefix, "fields": list(fields)})[:12]
        available = self.provider_max_name_length - len(prefix) - len(digest) - 2
        name = f"{prefix}_{readable[:available]}_{digest}"
        keys = [(field_name, 1) for field_name in fields]
        options: dict[str, Any] = {"unique": unique, "name": name}
        if unique:
            options["partialFilterExpression"] = {field_name: {"$type": list(MONGODB_INDEXABLE_BSON_TYPES)} for field_name in fields}
        return name, keys, options

    def _reconcile_indexes(self, collection: Any, indexes: Sequence[tuple[tuple[str, ...], bool]]) -> None:
        """Drop stale managed indexes and create the exact resolved plan."""
        desired = {name: (keys, options) for fields, unique in indexes for name, keys, options in [self._index_spec(fields, unique)]}
        information = collection.index_information()
        managed_prefixes = ("hb_index_", "hb_unique_")
        for name, current in information.items():
            if not name.startswith(managed_prefixes):
                continue
            if name not in desired:
                collection.drop_index(name)
                continue
            keys, options = desired[name]
            compatible = (
                tuple(current.get("key") or ()) == tuple(keys)
                and bool(current.get("unique", False)) == bool(options["unique"])
                and current.get("partialFilterExpression") == options.get("partialFilterExpression")
            )
            if not compatible:
                collection.drop_index(name)
        for name, (keys, options) in desired.items():
            collection.create_index(keys, **options)

    @backend_locked
    def drop(self) -> None:
        """Delete this backend instance's collections in the current database.

        Args:
            None.

        Returns:
            None: Matching collections are removed idempotently.
        """
        if self.client is None or self.db is None:
            return
        prefix = self._collection_scope() + "_"
        try:
            names = set(self._collections.values())
            list_names = getattr(self.db, "list_collection_names", None)
            if callable(list_names):
                names.update(name for name in list_names() if str(name).startswith(prefix))
            for collection_name in sorted(names):
                self.db.drop_collection(collection_name)
        except Exception as exc:
            self.client_error = redact_secrets(safe_error(exc))
            raise RuntimeError(f"MongoDB drop failed for database {self.database!r}: {self.client_error}") from exc
        self._collections.clear()
        self._schemas.clear()
        self._planned_indexes.clear()

    @backend_locked
    def upsert(self, rows: Sequence[RowOp]) -> None:
        """Apply ordered replacement upserts and idempotent deletes.

        Args:
            rows: Backend-ready row mutations.

        Returns:
            None: Every adjacent operation phase is applied in caller order.
        """
        self._need_client()
        try:
            for entity_id, is_delete, batch in self.ordered_row_op_batches(rows):
                collection = self._collection(entity_id)
                if is_delete:
                    collection.delete_many({"_id": {"$in": [str(row.object_id) for row in batch]}})
                    continue
                for row in batch:
                    document = row.to_row()
                    document["_id"] = str(row.object_id)
                    collection.replace_one({"_id": str(row.object_id)}, document, upsert=True)
        except Exception as exc:
            self.client_error = redact_secrets(safe_error(exc))
            raise RuntimeError(f"MongoDB mutation failed: {self.client_error}") from exc

    @staticmethod
    def _row(document: Mapping[str, Any]) -> dict[str, Any]:
        """Return a provider document without MongoDB's private identity field."""
        row = dict(document)
        private_id = row.pop("_id", None)
        if row.get("object_id") is None and private_id is not None:
            row["object_id"] = str(private_id)
        return row

    @backend_locked
    def scan(self, entity_id: str) -> list[dict[str, Any]]:
        """Return every document for one entity.

        Args:
            entity_id: Logical entity identifier.

        Returns:
            list[dict[str, Any]]: Exhaustive provider rows.
        """
        try:
            return [self._row(document) for document in self._collection(entity_id).find({})]
        except Exception as exc:
            self.client_error = redact_secrets(safe_error(exc))
            raise RuntimeError(f"MongoDB scan failed for {entity_id!r}: {self.client_error}") from exc

    @backend_locked
    def get_many(self, entity_id: str, object_ids: Iterable[Any], *, columns: Sequence[str] | None = None) -> list[dict[str, Any] | None]:
        """Return documents aligned to caller-supplied object ids.

        Args:
            entity_id: Logical entity identifier.
            object_ids: Object ids to fetch.
            columns: Optional field names to return; ``None`` returns the complete fragment.

        Returns:
            list[dict[str, Any] | None]: Rows in input order, with ``None`` for misses.

        Raises:
            TypeError: If ``columns`` contains a non-string name.
        """
        selection = self._row_columns(columns)
        ordered_ids = [str(object_id) for object_id in object_ids]
        if not ordered_ids:
            return []
        try:
            documents = self._collection(entity_id).find({"_id": {"$in": list(dict.fromkeys(ordered_ids))}})
            rows = {str(document.get("_id")): self._row(document) for document in documents}
            return self._project_rows([rows.get(object_id) for object_id in ordered_ids], selection)
        except Exception as exc:
            self.client_error = redact_secrets(safe_error(exc))
            raise RuntimeError(f"MongoDB get_many failed for {entity_id!r}: {self.client_error}") from exc

    @backend_locked
    def exists_many(self, entity_id: str, object_ids: Iterable[Any]) -> list[bool]:
        """Return existence flags aligned to caller-supplied object ids.

        Args:
            entity_id (str): Logical entity identifier.
            object_ids (Iterable[Any]): Object ids to test in caller order.

        Returns:
            list[bool]: Existence flags aligned with ``object_ids``.
        """
        return [row is not None for row in self.get_many(entity_id, object_ids)]

    @backend_locked
    def count(self, entity_id: str) -> int:
        """Return the exact document count for one entity.

        Args:
            entity_id (str): Logical entity identifier.

        Returns:
            int: Exact number of MongoDB documents for the entity.

        Raises:
            RuntimeError: If the MongoDB count query fails.
        """
        try:
            return int(self._collection(entity_id).count_documents({}))
        except Exception as exc:
            self.client_error = redact_secrets(safe_error(exc))
            raise RuntimeError(f"MongoDB count failed for {entity_id!r}: {self.client_error}") from exc

    def exact_count(self, entity_id: str) -> int:
        """Return the exact provider document count.

        Args:
            entity_id (str): Logical entity identifier.

        Returns:
            int: Exact MongoDB document count suitable for integrity proofs.
        """
        return self.count(entity_id)

    def _live_tag_claims(self) -> dict[str, Any]:
        """Return live availability derived from the optional MongoDB client."""
        return {
            **super()._live_tag_claims(),
            "backend-availability": "ready" if self.client is not None and self.db is not None else "unavailable",
        }

    def to_dict(self) -> dict[str, Any]:
        """Return connection-safe MongoDB backend metadata.

        Args:
            None.

        Returns:
            dict[str, Any]: Backend metadata without credentials.
        """
        return {
            "name": self.name,
            "type": self.type,
            "connected": self.client is not None and self.db is not None,
        }
