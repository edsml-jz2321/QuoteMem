"""ChromaDB vector backend."""

from __future__ import annotations

__all__ = [
    "ChromaBackend",
]

import re
from sys import float_info

from heavenbase.utils import Any, Iterable, Iterator, Mapping, Sequence, safe_error

from heavenbase.utils.files import file_uri_to_path, touch_dir
from ...execution.frame.result import ResultFrame
from ...modules import ModuleResolver
from ...schema.model import EntitySchema
from ...types.vector import _scale_cosine_query, cosine_score
from ..base import Backend, backend_locked
from ..scanning import execute_scan
from ._metadata import ENCODED_PREFIX, INT64_INTS, LEGACY_PLACEHOLDER, SCALAR_TYPES, metadata_entry, metadata_fields
from heavenbase.execution.contracts import LOGICAL_OPS, QueryFragment, RowOp

CHROMA_MAX_NAME_LENGTH = 512
# Oldest chromadb this backend runs on: the first release whose metadata holds lists and whose
# ``where`` takes ``$contains``. It also takes a ``None`` metadata value, which removes the key on
# upsert; chromadb 1.0.0 and earlier reject one.
CHROMA_MIN_VERSION = (1, 5, 0)
# Largest id list one scanning ``get`` names: Chroma binds a SQL variable per id, and SQLite allows 32,766.
CHROMA_MAX_GET_IDS = 4096
# Most SQL variables one native ``where`` may bind, as ``_where_variables`` counts them. SQLite allows
# 32,766 per statement; Chroma binds one per string or bool ``$in`` value and two per number, so the
# largest lists that ran were 32,762 strings and 16,380 numbers (chromadb 1.5.9).
CHROMA_MAX_WHERE_VARIABLES = 32_000
# Variables ``_where_variables`` counts for each field clause and each ``$and``/``$or`` besides its values.
# Chroma binds the field name of a clause, and a second column for a number: chromadb 1.5.9 bound at most
# 2 more per field clause and 1 more per ``$and``/``$or`` than the values; one more of each is spare.
CHROMA_CLAUSE_VARIABLES = 3
CHROMA_LOGICAL_VARIABLES = 2
# Most variables a by-id read with a ``where`` may bind for its ids and ``where`` together, as counted:
# SQLite's 32,766 less the 3 Chroma binds for any statement (chromadb 1.5.9) and 3 spare.
CHROMA_MAX_STATEMENT_VARIABLES = 32_760
# Most operands one ``$and``/``$or`` list holds. SQLite parses ``a OR b OR c ...`` one level deeper per
# operand and allows an expression 1,000 deep, so a flat list of about 990 operands fails (chromadb 1.5.9);
# a longer list is regrouped into a balanced tree of lists this long, about 32 levels deep per 32 times more.
CHROMA_MAX_WHERE_OPERANDS = 32
_EMBEDDING_FIELD_KEY = "heavenbase:embedding_field"
# Width of the embeddings Chroma's default model computes. 0.1.2.2 stored them for rows it sent no embeddings for.
CHROMA_DEFAULT_MODEL_WIDTH = 384


def _version_error(chromadb: Any) -> str | None:
    """Return why the installed chromadb is too old for this backend, or ``None``.

    Args:
        chromadb (Any): The imported ``chromadb`` module.

    Returns:
        str | None: The refusal message, or ``None`` when the version is supported or the
        module reports no parseable version.
    """
    version = getattr(chromadb, "__version__", None)
    match = re.match(r"(\d+)\.(\d+)\.(\d+)", version) if isinstance(version, str) else None
    if match is None or tuple(int(part) for part in match.groups()) >= CHROMA_MIN_VERSION:
        return None
    floor = ".".join(str(part) for part in CHROMA_MIN_VERSION)
    return f"the Chroma backend needs chromadb>={floor}, found chromadb {version}"


def _where_variables(where: Any) -> int:
    """Return an upper bound on the SQL variables Chroma binds for a ``where`` document.

    Args:
        where (Any): Chroma ``where`` document, or one operand of it.

    Returns:
        int: One per string or bool ``$in``/``$nin`` value and two per other ``$in``/``$nin`` value or
        comparison, plus ``CHROMA_CLAUSE_VARIABLES`` per field clause and ``CHROMA_LOGICAL_VARIABLES``
        per ``$and``/``$or``.
    """
    if not isinstance(where, Mapping):
        return 2
    total = 0
    for key, value in where.items():
        if key in ("$and", "$or"):
            total += CHROMA_LOGICAL_VARIABLES + sum(_where_variables(child) for child in value)
        elif key in ("$in", "$nin"):
            total += sum(1 if isinstance(item, (str, bool)) else 2 for item in value)
        elif str(key).startswith("$"):
            total += _where_variables(value)
        else:
            total += CHROMA_CLAUSE_VARIABLES + _where_variables(value)
    return total


def _grouped_where(key: str, operands: list[Any]) -> dict:
    """Return ``{key: operands}``, a list longer than ``CHROMA_MAX_WHERE_OPERANDS`` regrouped into a balanced tree.

    Args:
        key (str): ``"$and"`` or ``"$or"``.
        operands (list[Any]): Operands of that kind, in order.

    Returns:
        dict: Chroma ``where`` of the same meaning whose every ``key`` list holds at most
        ``CHROMA_MAX_WHERE_OPERANDS`` operands, each group holding at least two.
    """
    while len(operands) > CHROMA_MAX_WHERE_OPERANDS:
        parts = -(-len(operands) // CHROMA_MAX_WHERE_OPERANDS)
        size, extra = divmod(len(operands), parts)
        grouped: list[Any] = []
        start = 0
        for part in range(parts):
            end = start + size + (1 if part < extra else 0)
            grouped.append({key: operands[start:end]})
            start = end
        operands = grouped
    return {key: list(operands)}


def _in_chunks(values: Sequence[Any]) -> Iterator[list[Any]]:
    """Yield consecutive runs of ``$in`` values whose field clause binds at most ``CHROMA_MAX_WHERE_VARIABLES`` variables."""
    budget = CHROMA_MAX_WHERE_VARIABLES - CHROMA_CLAUSE_VARIABLES
    chunk: list[Any] = []
    used = 0
    for value in values:
        cost = _where_variables({"$in": [value]})
        if chunk and used + cost > budget:
            yield chunk
            chunk, used = [], 0
        chunk.append(value)
        used += cost
    if chunk:
        yield chunk


def _collection_space(collection: Any) -> str:
    """Read the collection's effective embedding index space."""
    schema = getattr(collection, "schema", None)
    embedding = schema.keys.get("#embedding") if schema is not None else None
    index = embedding.float_list.vector_index if embedding is not None and embedding.float_list is not None else None
    if index is not None and index.config.space is not None:
        return str(index.config.space).lower()

    configuration = getattr(collection, "configuration_json", None) or {}
    settings = configuration.get("hnsw") or configuration.get("spann") or {}
    metadata = collection.metadata or {}
    return str(settings.get("space") or metadata.get("hnsw:space", "l2")).lower()


def _chroma_score(space: str, vector: Sequence[float], query: Sequence[float]) -> float:
    """Return ``1 - distance`` under a Chroma collection's ``hnsw:space``, the score a native query reports.

    Args:
        space (str): Collection distance space: ``"cosine"``, ``"ip"`` or ``"l2"``.
        vector (Sequence[float]): Stored embedding.
        query (Sequence[float]): Query embedding.

    Returns:
        float: Cosine similarity, inner product, or one minus the squared Euclidean distance.
    """
    if space == "cosine":
        return cosine_score(vector, query)
    if space == "ip":
        return sum(float(left) * float(right) for left, right in zip(vector, query))
    # Chroma uses squared distance. An unrepresentable square is infinite.
    deltas = (float(left) - float(right) for left, right in zip(vector, query))
    return 1.0 - sum(delta * delta for delta in deltas)


def _rank_chroma_rows(rows: Iterable[dict[str, Any]], field: str, query: Sequence[float], space: str) -> list[dict[str, Any]]:
    """Score each returned embedding and keep the input order for ties."""
    ranked = [row for row in rows if row.get(field) is not None]
    for row in ranked:
        row["score"] = _chroma_score(space, row[field], query)
    return sorted(ranked, key=lambda row: -row["score"])


class ChromaBackend(Backend):
    """ChromaDB provider adapter with native vector search and metadata filters.

    Uses the Chroma client for persistence, row lookup, native ``near`` search,
    and metadata filtering through provider storage only.
    """

    identifier = "chroma"
    provider_max_name_length = CHROMA_MAX_NAME_LENGTH

    @classmethod
    def from_config(
        cls,
        name: str,
        data: Mapping[str, Any],
        *,
        resolver: ModuleResolver | None = None,
        config: Any | None = None,
    ) -> "ChromaBackend":
        """Build the Chroma backend from a config mapping.

        Args:
            name (str): Workspace backend instance name.
            data (Mapping[str, Any]): Backend config mapping. Supported keys are Chroma
                constructor parameters, including ``path`` and ``ws_id``.
            resolver: Context-owned module authority.
            config: Config authority retained by the backend.

        Returns:
            ChromaBackend: Configured Chroma backend instance.
        """
        cfg = dict(data or {})
        ws_id = cfg.pop("ws_id", None)
        return cls(name, cfg.pop("path", None), ws_id=ws_id, resolver=resolver, config=config, **cfg)

    def __init__(
        self,
        name: str = "chroma",
        path: str | None = None,
        ws_id: str | None = None,
        *,
        resolver: ModuleResolver | None = None,
        config: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """Create a Chroma backend.

        Args:
            name (str): Workspace backend instance name.
            path (str | None): Optional local persistence directory. Passing ``None`` uses
                Chroma's ephemeral client.
            ws_id (str | None): Optional workspace namespace for Chroma database selection.
            resolver: Context-owned module authority.
            config: Config authority retained by the backend.
            **kwargs (Any): Extra keyword arguments passed to the Chroma client. ``database``
                may be supplied to override the workspace-derived database name.

        Returns:
            None: This function does not return a value.

        Notes:
            The backend needs chromadb 1.5.0 or later. With an older chromadb, or none, the
            backend is unavailable and its operations raise ``RuntimeError`` with the reason.
        """
        super().__init__(name, ws_id=ws_id, resolver=resolver, config=config)
        self.path = None if path is None else file_uri_to_path(path, home=self.config.root)
        self._database_override = kwargs.pop("database", None)
        self._client_kwargs = dict(kwargs)
        self.client_error = ""
        self._schemas: dict[str, EntitySchema] = {}
        self._owned_fields: dict[str, set[str] | None] = {}
        self._embedding_fields: dict[str, str | None] = {}
        self._collections: dict[str, str] = {}
        self.database = self._database_name()
        self._connect()

    def _database_name(self) -> str | None:
        return self._database_override or (self._physical_name(self.ws_id) if self.ws_id else None)

    def _connect(self) -> None:
        try:
            import chromadb

            too_old = _version_error(chromadb)
            if too_old is not None:
                raise RuntimeError(too_old)
            kwargs = dict(self._client_kwargs)
            self.database = self._database_name()
            if self.database:
                self._ensure_database(chromadb, self.database)
                kwargs.setdefault("database", self.database)
            self.client = chromadb.EphemeralClient(**kwargs) if self.path is None else chromadb.PersistentClient(path=touch_dir(self.path), **kwargs)
        except Exception as exc:
            self.client = None
            self.client_error = safe_error(exc)

    def _admin_client(self, chromadb: Any) -> Any:
        admin_cls = getattr(chromadb, "AdminClient", None)
        if admin_cls is None:
            return None
        settings = None
        if self.path is not None:
            from chromadb.config import Settings

            settings = Settings(is_persistent=True, persist_directory=touch_dir(self.path))
        return admin_cls(settings) if settings is not None else admin_cls()

    def _ensure_database(self, chromadb: Any, database: str) -> None:
        try:
            admin = self._admin_client(chromadb)
            if admin is None:
                return
            get_database = getattr(admin, "get_database", None)
            create_database = getattr(admin, "create_database", None)
            if create_database is None:
                return
            if get_database is not None:
                try:
                    get_database(database)
                    return
                except Exception:
                    pass
            try:
                create_database(name=database)
            except TypeError:
                create_database(database)
        except Exception:
            pass

    def _refresh_workspace(self) -> None:
        self._collections.clear()
        self._embedding_fields.clear()
        self._connect()

    def _delete_collection(self, name: str) -> None:
        if self.client is None:
            return
        delete_collection = getattr(self.client, "delete_collection", None)
        if delete_collection is None:
            return
        try:
            delete_collection(name=name)
        except TypeError:
            delete_collection(name)

    @backend_locked
    def drop(self) -> None:
        """Drop the current Chroma database or known collections.

        Args:
            None.

        Returns:
            None: This function does not return a value.
        """
        try:
            if self.database:
                import chromadb

                admin = self._admin_client(chromadb)
                delete_database = getattr(admin, "delete_database", None) if admin is not None else None
                if delete_database is not None:
                    try:
                        delete_database(name=self.database)
                    except TypeError:
                        delete_database(self.database)
                    return
            for name in list(self._collections.values()):
                self._delete_collection(name)
        finally:
            self._schemas.clear()
            self._owned_fields.clear()
            self._embedding_fields.clear()
            self._collections.clear()
            self.client = None

    def _need_client(self) -> None:
        if self.client is None:
            detail = f": {self.client_error}" if self.client_error else ""
            raise RuntimeError(f"Chroma backend {self.name!r} is unavailable{detail}")

    def _collection_name(self, entity_id: str) -> str:
        return self._collections.setdefault(entity_id, self._physical_name(entity_id))

    def schema_warnings(self, schema: EntitySchema, bindings: Sequence[Any]) -> list[str]:
        """Return Chroma collection-name warnings for this schema.

        Args:
            schema (EntitySchema): Entity schema being registered.
            bindings (Sequence[Any]): Storage bindings for the entity.

        Returns:
            list[str]: Provider naming warnings for physical collection names.
        """
        warnings = super().schema_warnings(schema, bindings)
        warnings.extend(
            self._physical_name_warnings(
                [(schema.entity_id, self._collection_name(schema.entity_id))],
                kind="Chroma collection",
                max_length=self.provider_max_name_length,
                case_sensitive=False,
            )
        )
        return warnings

    def _collection(self, entity_id: str, *, embedding_field: str | None = None) -> Any:
        name = self._collection_name(entity_id)
        metric = str(self.config.get("heavenbase.backend_defaults.chroma.metric", default="cosine")).lower()
        metadata = {"hnsw:space": metric}
        if embedding_field is not None:
            metadata[_EMBEDDING_FIELD_KEY] = embedding_field
        return self.client.get_or_create_collection(name=name, metadata=metadata)

    def near_space(self, entity_id: str) -> str:
        """Return the distance space a native near on one entity's collection ranks by.

        A collection keeps the space it was created with, so a later change of
        ``heavenbase.backend_defaults.chroma.metric`` does not change how an
        existing collection ranks. Before the collection can exist (no client, or
        an entity not stored here) the configured space, which a new collection
        gets, is returned.

        Args:
            entity_id (str): Logical entity identifier.

        Returns:
            str: Lower-case distance space (``"cosine"``, ``"l2"`` or ``"ip"``).

        Raises:
            RuntimeError: If the provider cannot open the collection.
        """
        return self._near_collection(entity_id)[0]

    @backend_locked
    def _near_collection(self, entity_id: str) -> tuple[str, Any]:
        """Open the current collection once for near planning and execution."""
        if self.client is None or entity_id not in self._schemas:
            return str(self.config.get("heavenbase.backend_defaults.chroma.metric", default="cosine")).lower(), None
        try:
            collection = self._collection(entity_id)
            return _collection_space(collection), collection
        except Exception as exc:
            self.client_error = safe_error(exc)
            raise RuntimeError(f"Chroma collection lookup failed for {entity_id!r}: {self.client_error}") from exc

    @backend_locked
    def near_vector_field(self, entity_id: str) -> str | None:
        """Return the field indexed as the collection embedding.

        Args:
            entity_id (str): Logical entity identifier.

        Returns:
            str | None: Indexed field name, or ``None`` for a vectorless collection.
        """
        return self._vector_field(entity_id)

    def _vector_field(self, entity_id: str) -> str | None:
        """Return the registered embedding field without reacquiring the backend lock."""
        return self._embedding_fields.get(entity_id) if entity_id in self._schemas else None

    def _metadata(self, entity_id: str, values: dict[str, Any]) -> dict[str, Any]:
        """Return the Chroma metadata that stores one row's fields.

        Chroma metadata holds scalars and non-empty lists of one scalar type. Any other
        value, such as an Array of objects, an empty Array or an integer outside the
        signed 64-bit range, is stored as tagged JSON under its encoded key (see
        ``heavenbase.backends.vector._metadata``). Json
        values arrive as their JSON text and Artifact values as base64 text. The vector
        Chroma stores as the embedding is not metadata.

        Chroma merges an upsert's metadata into the stored record and removes a key set
        to ``None``. Every field this backend stores is therefore named, a ``None``,
        cleared or omitted one as ``None``, and so is the key a value is not stored
        under, so an upsert replaces the row's stored fields as on other backends. The
        ``_hb`` placeholder of 0.1.2.2 is named as ``None`` too, which also keeps the
        metadata from being empty, which Chroma rejects. An explicit null in a
        nullable field with a default is kept under the encoded key.
        """
        vector = self._vector_field(entity_id)
        schema = self._schemas.get(entity_id)
        names = dict.fromkeys(values)
        if schema is not None:
            owned = self._owned_fields.get(entity_id)
            names.update(dict.fromkeys(name for name in schema.fields if owned is None or name in owned))
        out: dict[str, Any] = {LEGACY_PLACEHOLDER: None}
        for key in names:
            if key == "object_id" or key == vector:
                continue
            value = values.get(key)
            field_schema = schema.fields.get(key) if schema is not None else None
            preserve_null = key in values and field_schema is not None and field_schema.has_default and not field_schema.required
            native, encoded = metadata_entry(value, self._is_metadata_array, INT64_INTS, preserve_null=preserve_null)
            if native is None and isinstance(value, int):
                # The tagged value keeps the exact integer. A finite value of the same sign
                # preserves its order against every operand the handler can compare natively.
                native = -float_info.max if value < 0 else float_info.max
            out[key] = native
            out[ENCODED_PREFIX + key] = encoded
        return out

    def _is_metadata_array(self, value: Any) -> bool:
        if not isinstance(value, list) or not value:
            return False
        first_type = type(value[0])
        if first_type not in SCALAR_TYPES:
            return False
        return all(type(item) is first_type for item in value)

    def _document(self, entity_id: str, values: dict[str, Any], object_id: Any) -> str:
        """Return a row's Chroma document: its text values, without the base64 text of an Artifact."""
        schema = self._schemas.get(entity_id)
        artifacts = {name for name, field_schema in schema.fields.items() if type(field_schema.dtype).identifier == "artifact"} if schema else set()
        parts = [str(value) for key, value in values.items() if isinstance(value, str) and value and key not in artifacts]
        return "\n".join(parts) or str(object_id)

    @backend_locked
    def ensure(self, schema: EntitySchema, bindings: Sequence[Any]) -> None:
        """Ensure a Chroma collection for the entity.

        Args:
            schema (EntitySchema): Entity schema to store in Chroma.
            bindings (Sequence[Any]): Storage bindings supplied by the workspace.

        Returns:
            None: This function does not return a value.

        Raises:
            ValueError: If the collection records an embedding field that the schema no longer stores here.

        Notes:
            The first Vector field stored here becomes the collection embedding. A collection written by
            0.1.2.2 records no embedding field. When it holds embeddings that Chroma's default model
            computed, because 0.1.2.2 took a Vector field stored on another backend as the embedding,
            and their width differs from the dimension the field declares, no field becomes the embedding:
            every Vector field stored here is kept as metadata.
        """
        entity_id = schema.entity_id
        owned = {binding.field_name for binding in bindings if getattr(binding, "backend", None) == self.name}
        # A direct caller without bindings for this backend stores the whole schema here.
        vectors = [name for name, field in schema.fields.items() if type(field.dtype).identifier == "vector" and (not owned or name in owned)]
        current = self._vector_field(entity_id)
        embedding_field = current if current in vectors else next(iter(vectors), None)
        if self.client is not None:
            collection = self._collection(entity_id, embedding_field=embedding_field)
            stored = (collection.metadata or {}).get(_EMBEDDING_FIELD_KEY)
            if isinstance(stored, str):
                if stored not in vectors:
                    raise ValueError(f"Chroma entity {entity_id!r} must keep its collection embedding field {stored!r}; use a new collection to change it")
                embedding_field = stored
            elif embedding_field is not None and self._holds_generated_embeddings(schema, vectors, embedding_field, collection):
                embedding_field = None
        self._schemas[entity_id] = schema
        self._owned_fields[entity_id] = owned or None
        self._embedding_fields[entity_id] = embedding_field

    def _holds_generated_embeddings(self, schema: EntitySchema, vectors: Sequence[str], field: str, collection: Any) -> bool:
        """Return whether an unlabeled collection holds embeddings 0.1.2.2 let Chroma compute instead of ``field``'s vectors.

        0.1.2.2 took the first Vector field of the whole schema as the embedding. When another backend
        stored that field, it sent no embeddings and Chroma computed them from the documents with its
        default model, so the collection takes only vectors of that width.

        Args:
            schema (EntitySchema): Entity schema being registered.
            vectors (Sequence[str]): Vector fields stored on this backend, in declaration order.
            field (str): Field chosen as the collection embedding.
            collection (Any): Open collection that records no embedding field.

        Returns:
            bool: Whether the first declared Vector is stored elsewhere and the collection has the
            default model's width, which ``field`` declares not to have.
        """
        first = next((name for name, item in schema.fields.items() if type(item.dtype).identifier == "vector"), None)
        # A field without a dimension may hold real vectors of that width: 0.1.2.2 wrote them while the field was the first
        # Vector, or a later release into a collection 0.1.2.2 left empty. A placeholder would overwrite them.
        dim = schema.fields[field].dtype.dim
        if first in vectors or dim is None or dim == CHROMA_DEFAULT_MODEL_WIDTH:
            return False
        return self._collection_width(collection) == CHROMA_DEFAULT_MODEL_WIDTH

    @backend_locked
    def upsert(self, rows: Sequence[RowOp]) -> None:
        """Upsert vectors and metadata into Chroma.

        Args:
            rows (Sequence[RowOp]): Row mutations to apply. ``delete`` operations remove
                documents; other operations upsert document text, metadata, and vectors.

        Returns:
            None: This function does not return a value.

        Raises:
            RuntimeError: If the Chroma client is unavailable.
            ValueError: If a row of a vector entity sets its vector to ``None``, or a new row omits it.

        Notes:
            Every upsert passes explicit embeddings, so Chroma never computes them. A row of a
            vector entity that omits its vector keeps its stored vector, and an entity whose
            Vector field is not stored on this backend stores a constant placeholder embedding.
            So does one whose collection holds embeddings that 0.1.2.2 let Chroma compute (see
            ``ensure``); its Vector values are stored as metadata.
            Every other stored field is replaced. Omitted fields are removed. Chroma encodes
            explicit ``None`` for nullable metadata fields with defaults. For other metadata
            fields, setting the value to ``None`` removes the field.

            Chroma accepts at most the client's ``get_max_batch_size()`` rows per upsert call
            (5,461 on the local clients of chromadb 1.5.9), so a larger upsert batch goes to
            Chroma as consecutive calls of at most that many rows, in order. Each call commits
            on its own: a provider error in a later call leaves the earlier calls applied.
            Missing vectors are checked for the whole batch before the first call. Deletes stay
            one call per entity batch; chromadb 1.5.9 deletes 70,000 ids in one local call.
        """
        self._need_client()
        size = self._write_batch_size()
        for entity_id, is_delete, ops in self.ordered_row_op_batches(rows):
            collection = self._collection(entity_id)
            ids = [str(row.object_id) for row in ops]
            if is_delete:
                collection.delete(ids=ids)
                continue
            metadatas = [self._metadata(entity_id, row.values) for row in ops]
            documents = [self._document(entity_id, row.values, row.object_id) for row in ops]
            embeddings = self._upsert_embeddings(collection, entity_id, ids, ops)
            for start in range(0, len(ids), size):
                stop = start + size
                collection.upsert(ids=ids[start:stop], metadatas=metadatas[start:stop], documents=documents[start:stop], embeddings=embeddings[start:stop])

    def _write_batch_size(self) -> int:
        """Return the most rows one Chroma upsert call sends: the client's maximum batch size.

        Returns:
            int: ``client.get_max_batch_size()``, or ``max_batch_size`` on clients without that
            method; ``CHROMA_MAX_GET_IDS`` when the client reports neither.
        """
        getter = getattr(self.client, "get_max_batch_size", None)
        size = getter() if callable(getter) else getattr(self.client, "max_batch_size", None)
        return size if isinstance(size, int) and not isinstance(size, bool) and size > 0 else CHROMA_MAX_GET_IDS

    def _upsert_embeddings(self, collection: Any, entity_id: str, ids: Sequence[str], ops: Sequence[RowOp]) -> list[list[float]]:
        """Return one explicit embedding per upserted row so Chroma never embeds documents itself.

        Chroma embeds ``documents`` with its default model (a network download) whenever an
        upsert carries no embeddings. HeavenBase searches only with caller vectors, so an
        entity without a Vector field on this backend stores a constant placeholder whose
        width matches the collection, and a vector entity row that omits its vector keeps
        its stored vector. Chroma cannot store a record without an embedding, so a vector
        set explicitly to ``None`` is rejected rather than silently kept.

        Args:
            collection (Any): Chroma collection receiving the upsert.
            entity_id (str): Logical entity identifier.
            ids (Sequence[str]): Provider ids aligned with ``ops``.
            ops (Sequence[RowOp]): Upsert operations for one entity.

        Returns:
            list[list[float]]: Embeddings aligned with ``ids``.

        Raises:
            ValueError: If a row sets its vector to ``None``, or omits it with no stored vector to keep.
        """
        vector = self._vector_field(entity_id)
        if vector is None:
            return [[1.0] * self._placeholder_width(collection) for _ in ids]
        cleared = [object_id for object_id, row in zip(ids, ops) if vector in row.values and row.values[vector] is None]
        if cleared:
            raise ValueError(f"Chroma entity {entity_id!r} cannot clear its {vector!r} vector; rows {cleared} set it to None")
        supplied = [row.values.get(vector) for row in ops]
        missing = [object_id for object_id, value in zip(ids, supplied) if value is None]
        if missing:
            by_id: dict[str, Any] = {}
            for start in range(0, len(missing), CHROMA_MAX_GET_IDS):
                stored = collection.get(ids=missing[start : start + CHROMA_MAX_GET_IDS], include=["embeddings"])
                embeddings = stored.get("embeddings")
                by_id.update(zip(stored.get("ids") or [], embeddings if embeddings is not None else []))
            supplied = [by_id.get(object_id) if value is None else value for object_id, value in zip(ids, supplied)]
            absent = [object_id for object_id, value in zip(ids, supplied) if value is None]
            if absent:
                raise ValueError(f"Chroma entity {entity_id!r} needs a {vector!r} vector for new rows {absent}")
        return [value.tolist() if hasattr(value, "tolist") else [float(item) for item in value] for value in supplied]

    def _collection_width(self, collection: Any) -> int | None:
        """Return the embedding width a collection already uses, or ``None`` when it has none."""
        dimension = getattr(getattr(collection, "_model", None), "dimension", None)
        if isinstance(dimension, int) and dimension > 0:
            return dimension
        stored = collection.get(limit=1, include=["embeddings"]).get("embeddings")
        return len(stored[0]) if stored is not None and len(stored) else None

    def _placeholder_width(self, collection: Any) -> int:
        """Return the embedding width a vectorless collection already uses, or 1 when it has none.

        Collections written before placeholders hold 384-wide default-model embeddings; matching
        their width keeps them writable.
        """
        return self._collection_width(collection) or 1

    def _rows_from_chroma(
        self,
        entity_id: str,
        object_ids: Sequence[Any],
        metadatas: Sequence[Any] | None = None,
        embeddings: Sequence[Any] | None = None,
        distances: Sequence[Any] | None = None,
    ) -> list[dict[str, Any]]:
        rows = []
        metadatas = metadatas or []
        for pos, object_id in enumerate(object_ids):
            embedding = embeddings[pos] if embeddings is not None and pos < len(embeddings) else None
            row = self._row_from_chroma(entity_id, object_id, metadatas[pos] if pos < len(metadatas) else None, embedding)
            if distances is not None:
                distance = distances[pos] or 0.0
                row["score"] = 1.0 - float(distance)
            rows.append(row)
        return rows

    def _row_from_chroma(self, entity_id: str, object_id: Any, metadata: Any = None, embedding: Any = None) -> dict[str, Any]:
        row = metadata_fields(metadata)
        row["object_id"] = str(object_id)
        vector = self._vector_field(entity_id)
        if vector is not None and embedding is not None:
            row[vector] = embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)
        return row

    @backend_locked
    def scan(self, entity_id: str) -> list[dict[str, Any]]:
        """Return rows from Chroma provider storage.

        Args:
            entity_id (str): Entity identifier whose collection should be scanned.

        Returns:
            list[dict[str, Any]]: Rows reconstructed from Chroma metadata and embeddings,
            in Chroma's own order.

        Raises:
            RuntimeError: If the client is unavailable or the scan request fails.

        Notes:
            Rows are read through bounded id lookups (see ``iter_scan``), because one
            Chroma ``get`` of a whole collection binds a SQL variable per row and
            fails above SQLite's limit of 32,766. The backend lock is held for the
            whole scan.
        """
        return [row for page in self._id_pages(entity_id, CHROMA_MAX_GET_IDS, "scan") for row in page]

    def iter_scan(self, entity_id: str, *, batch_size: int, columns: Sequence[str] | None = None) -> Iterable[dict[str, Any]]:
        """Iterate a Chroma collection in one linear pass of bounded provider pages.

        Chroma has no key cursor, and a ``get`` page at ``offset`` costs time that
        grows with the offset, so offset paging makes one pass quadratic. The
        collection's ids (not rows) are listed once, in Chroma's own order, and
        rows are fetched by id at most ``batch_size`` at a time, so one pass costs
        time linear in the collection size.

        Args:
            entity_id (str): Logical entity identifier.
            batch_size (int): Positive maximum page size. A provider request names
                at most ``CHROMA_MAX_GET_IDS`` ids, whatever the page size.
            columns (Sequence[str] | None): Optional field names to return; ``None`` returns the complete fragment.

        Yields:
            dict[str, Any]: Row fragment with metadata and embedding.

        Raises:
            ValueError: If the page size is not positive.
            RuntimeError: If the provider read fails.

        Notes:
            The id list takes memory linear in the collection size; rows are held
            one page at a time. The backend lock is taken per provider request and
            never held across a ``yield``. A row present for the whole scan is
            yielded exactly once; a row deleted after the listing is skipped, and a
            row added after it is not seen.
        """
        self._check_batch_size(batch_size)
        selection = self._row_columns(columns)
        for page in self._id_pages(entity_id, batch_size, "iter_scan"):
            for row in page:
                yield self._project_row(row, selection)

    def _id_pages(self, entity_id: str, page_size: int, operation: str, *, collection: Any = None) -> Iterator[list[dict[str, Any]]]:
        """List a collection's ids once, then yield its rows fetched by id, one bounded page at a time.

        Args:
            entity_id (str): Logical entity identifier.
            page_size (int): Positive maximum page size, capped at ``CHROMA_MAX_GET_IDS``.
            operation (str): Public method named in provider errors.
            collection (Any): Collection already opened for this near query, if available.

        Yields:
            list[dict[str, Any]]: Rows still present, in Chroma's listing order.

        Raises:
            RuntimeError: If the client is unavailable or a provider read fails.
        """
        with self._lock:
            self._need_client()
            try:
                opened = self._collection(entity_id) if collection is None else collection
                listed = [str(object_id) for object_id in opened.get(include=[]).get("ids") or []]
            except Exception as exc:
                self.client_error = safe_error(exc)
                raise RuntimeError(f"Chroma {operation} failed for {entity_id!r}: {self.client_error}") from exc
        step = min(page_size, CHROMA_MAX_GET_IDS)
        for start in range(0, len(listed), step):
            with self._lock:
                self._need_client()
                try:
                    rows = self._rows_by_ids(entity_id, listed[start : start + step], collection=collection)
                except Exception as exc:
                    self.client_error = safe_error(exc)
                    raise RuntimeError(f"Chroma {operation} failed for {entity_id!r}: {self.client_error}") from exc
            yield rows

    def _rows_by_ids(self, entity_id: str, object_ids: Sequence[str], where: Any = None, *, collection: Any = None) -> list[dict[str, Any]]:
        """Return the stored rows among ``object_ids``, read at most ``CHROMA_MAX_GET_IDS`` ids per provider request.

        Args:
            entity_id (str): Logical entity identifier.
            object_ids (Sequence[str]): Distinct provider ids, in the order rows should follow.
            where (Any): Optional Chroma ``where`` document of at most ``CHROMA_MAX_WHERE_VARIABLES``
                variables. Each request then also applies it, so a row returned matches it as read;
                a request names fewer ids when the ``where`` leaves fewer than ``CHROMA_MAX_GET_IDS``
                of ``CHROMA_MAX_STATEMENT_VARIABLES``.
            collection (Any): Collection already opened for this request, if available.

        Returns:
            list[dict[str, Any]]: Rows found (and matching ``where``), in ``object_ids`` order.
        """
        if collection is None:
            collection = self._collection(entity_id)
        step = CHROMA_MAX_GET_IDS if where is None else max(1, min(CHROMA_MAX_GET_IDS, CHROMA_MAX_STATEMENT_VARIABLES - _where_variables(where)))
        rows: list[dict[str, Any]] = []
        for start in range(0, len(object_ids), step):
            chunk = object_ids[start : start + step]
            page = collection.get(ids=list(chunk), include=["metadatas", "embeddings"], **({} if where is None else {"where": where}))
            by_id = {
                row["object_id"]: row for row in self._rows_from_chroma(entity_id, page.get("ids", []), page.get("metadatas") or [], page.get("embeddings"))
            }
            rows.extend(by_id[object_id] for object_id in chunk if object_id in by_id)
        return rows

    @backend_locked
    def get_many(
        self, entity_id: str, object_ids: Iterable[Any], *, columns: Sequence[str] | None = None, _near_fragment: QueryFragment | None = None
    ) -> list[dict[str, Any] | None]:
        """Return rows through Chroma object-id lookup.

        Args:
            entity_id (str): Entity identifier whose rows should be loaded.
            object_ids (Iterable[Any]): Object identifiers to fetch in caller order.
            columns (Sequence[str] | None): Optional field names to return; ``None`` returns the complete fragment.
            _near_fragment (QueryFragment | None): Internal query route for hydrating near hits.

        Returns:
            list[dict[str, Any] | None]: Rows aligned with ``object_ids``. Missing documents
            are returned as ``None``.

        Raises:
            TypeError: If ``columns`` contains a non-string name.
            RuntimeError: If the client is unavailable or the lookup request fails.

        Notes:
            Ids are looked up at most ``CHROMA_MAX_GET_IDS`` per provider request, because
            Chroma binds a SQL variable per id and SQLite allows 32,766.
        """
        selection = self._row_columns(columns)
        ordered_ids = [str(object_id) for object_id in object_ids]
        if not ordered_ids:
            return []
        self._need_client()
        try:
            collection = _near_fragment.payload.get("collection") if _near_fragment is not None and _near_fragment.kind() == "chroma_search" else None
            rows_by_id = {row["object_id"]: row for row in self._rows_by_ids(entity_id, list(dict.fromkeys(ordered_ids)), collection=collection)}
            return self._project_rows([rows_by_id.get(row_id) for row_id in ordered_ids], selection)
        except Exception as exc:
            self.client_error = safe_error(exc)
            raise RuntimeError(f"Chroma get_many failed for {entity_id!r}: {self.client_error}") from exc

    @backend_locked
    def exists_many(self, entity_id: str, object_ids: Iterable[Any]) -> list[bool]:
        """Return existence flags through Chroma object-id lookup.

        Args:
            entity_id (str): Entity identifier whose rows should be checked.
            object_ids (Iterable[Any]): Object identifiers to check in caller order.

        Returns:
            list[bool]: Existence flags aligned with ``object_ids``.

        Raises:
            RuntimeError: If the client is unavailable or the lookup request fails.

        Notes:
            Ids are checked at most ``CHROMA_MAX_GET_IDS`` per provider request (see ``get_many``).
        """
        ordered_ids = [str(object_id) for object_id in object_ids]
        if not ordered_ids:
            return []
        self._need_client()
        try:
            collection = self._collection(entity_id)
            unique = list(dict.fromkeys(ordered_ids))
            found = {
                str(row_id)
                for start in range(0, len(unique), CHROMA_MAX_GET_IDS)
                for row_id in collection.get(ids=unique[start : start + CHROMA_MAX_GET_IDS], include=[]).get("ids", [])
            }
            return [row_id in found for row_id in ordered_ids]
        except Exception as exc:
            self.client_error = safe_error(exc)
            raise RuntimeError(f"Chroma exists_many failed for {entity_id!r}: {self.client_error}") from exc

    @backend_locked
    def count(self, entity_id: str) -> int:
        """Return Chroma collection count without scanning row bodies.

        Args:
            entity_id (str): Entity identifier whose collection should be counted.

        Returns:
            int: Number of stored Chroma documents for the entity.

        Raises:
            RuntimeError: If the client is unavailable or the count request fails.
        """
        self._need_client()
        try:
            return int(self._collection(entity_id).count())
        except Exception as exc:
            self.client_error = safe_error(exc)
            raise RuntimeError(f"Chroma count failed for {entity_id!r}: {self.client_error}") from exc

    def _live_tag_claims(self) -> dict[str, Any]:
        """Return live tag claims derived from Chroma availability.

        Args:
            None.

        Returns:
            dict[str, Any]: Live Backend tag claims.
        """
        return {
            **super()._live_tag_claims(),
            "backend-availability": "ready" if self.client is not None else "unavailable",
        }

    def _is_chroma_fragment(self, fragment: QueryFragment) -> bool:
        if fragment.kind() == "chroma_where":
            return True
        if fragment.op in ("and", "or"):
            return all(self._is_chroma_fragment(child) for child in fragment.children())
        return False

    def _matching_id_set(self, collection: Any, where: Any) -> set[str]:
        """Return the ids matching a Chroma ``where``, split into statements SQLite can bind.

        Args:
            collection (Any): Chroma collection to read.
            where (Any): Chroma ``where`` document.

        Returns:
            set[str]: Matching provider ids. A ``where`` above ``CHROMA_MAX_WHERE_VARIABLES`` is read
            per ``$and``/``$or`` operand and per run of ``$in`` values, and the id sets are combined.
        """
        if _where_variables(where) <= CHROMA_MAX_WHERE_VARIABLES or not isinstance(where, Mapping) or len(where) != 1:
            return {str(object_id) for object_id in collection.get(where=where, include=[]).get("ids") or []}
        ((key, value),) = where.items()
        if key == "$and":
            matched: set[str] | None = None
            for child in value:
                found = self._matching_id_set(collection, child)
                matched = found if matched is None else matched & found
                if not matched:
                    break
            return matched or set()
        if key == "$or":
            return set().union(*(self._matching_id_set(collection, child) for child in value))
        if isinstance(value, Mapping) and set(value) == {"$in"}:
            return {
                str(object_id)
                for chunk in _in_chunks(list(value["$in"]))
                for object_id in collection.get(where={key: {"$in": chunk}}, include=[]).get("ids") or []
            }
        return {str(object_id) for object_id in collection.get(where=where, include=[]).get("ids") or []}

    def _matching_ids(self, collection: Any, where: Any) -> list[str]:
        """Return the ids matching a Chroma ``where`` in Chroma's own order, however many values it names.

        Args:
            collection (Any): Chroma collection to read.
            where (Any): Chroma ``where`` document.

        Returns:
            list[str]: Matching provider ids in collection order.
        """
        if _where_variables(where) <= CHROMA_MAX_WHERE_VARIABLES:
            return [str(object_id) for object_id in collection.get(where=where, include=[]).get("ids") or []]
        matched = self._matching_id_set(collection, where)
        return [str(object_id) for object_id in collection.get(include=[]).get("ids") or [] if str(object_id) in matched]

    def _matched_rows(self, entity_id: str, fragment: QueryFragment, *, collection: Any = None) -> list[dict[str, Any]]:
        """Return the rows that match a native Chroma filter fragment, in Chroma's own order.

        The matches are listed natively and their rows read by id. Another backend instance on the
        same storage may write between the two reads, so a row read by id must match again: a
        ``where`` that fits one statement goes with each by-id read, and a larger one, which no
        single read can apply, is checked again on each row read with the fragment's row predicate.

        Args:
            entity_id (str): Logical entity identifier.
            fragment (QueryFragment): Native Chroma filter fragment, or an ``and``/``or`` of them.
            collection (Any): Collection already opened for this request, if available.

        Returns:
            list[dict[str, Any]]: Rows that matched the filter when they were read.
        """
        where = self._where(fragment)
        opened = self._collection(entity_id) if collection is None else collection
        matched = self._matching_ids(opened, where)
        if _where_variables(where) <= CHROMA_MAX_WHERE_VARIABLES:
            return self._rows_by_ids(entity_id, matched, where, collection=collection)
        return execute_scan(self, fragment, rows=self._rows_by_ids(entity_id, matched, collection=collection)).rows()

    def _rank_matches(self, entity_id: str, field: str, params: Mapping[str, Any], filt: QueryFragment, space: str, collection: Any) -> list[dict[str, Any]]:
        """Return the ``top_k`` nearest rows among those a filter too large for one statement matches.

        Chroma cannot run such a ``where`` inside one query, so its matches are read as
        ``_matched_rows`` reads them and ranked by the collection's own ``hnsw:space``, giving the
        score a native query reports; ties keep the collection's order.

        Args:
            entity_id (str): Logical entity identifier.
            field (str): Vector field holding each row's embedding.
            params (Mapping[str, Any]): Native near payload with ``vector`` and ``top_k``.
            filt (QueryFragment): Native Chroma filter fragment of the near.
            space (str): Distance space of the embedding index.
            collection (Any): Open collection used for the near query.

        Returns:
            list[dict[str, Any]]: Up to ``top_k`` rows with ``score``, best first.
        """
        return _rank_chroma_rows(self._matched_rows(entity_id, filt, collection=collection), field, params["vector"], space)[: params["top_k"]]

    def _where(self, fragment: QueryFragment) -> dict:
        """Compile a native fragment to a Chroma ``where``, with nested ``and``/``or`` of one kind rebalanced.

        Chroma rejects a ``where`` nested more than 63 levels deep (chromadb 1.5.9), and ``a | b | c``
        nests one level per operand, so the operands of an ``or`` of ``or``s (or an ``and`` of
        ``and``s) are collected into one list. SQLite rejects a list of about 990 operands, so that
        list is then regrouped into lists of at most ``CHROMA_MAX_WHERE_OPERANDS``.

        Args:
            fragment (QueryFragment): Native Chroma filter fragment, or an ``and``/``or`` of them.

        Returns:
            dict: Chroma ``where`` document.

        Raises:
            TypeError: If the fragment is not a native Chroma filter.
        """
        if fragment.kind() == "chroma_where":
            return fragment.payload
        if fragment.op not in ("and", "or"):
            raise TypeError(f"Cannot compile {fragment.op!r} to Chroma where")
        operands: list[Any] = []
        pending = list(reversed(fragment.children()))
        while pending:
            child = pending.pop()
            if child.kind() != "chroma_where" and child.op == fragment.op:
                pending.extend(reversed(child.children()))
            else:
                operands.append(self._where(child))
        return _grouped_where(f"${fragment.op}", operands)

    @backend_locked
    def execute(self, fragment: QueryFragment) -> ResultFrame:
        """Execute Chroma-native search and filter fragments when possible.

        Args:
            fragment (QueryFragment): Backend-ready query fragment compiled by the handler layer.

        Returns:
            ResultFrame: Matching rows and object identifiers.

        Raises:
            NotImplementedError: If the fragment cannot be executed natively or by scan fallback.
            RuntimeError: If the client is unavailable or the provider request fails.
        """
        entity_id = fragment.entity_id()
        self._need_client()
        if entity_id and fragment.kind() == "chroma_search" and fragment.op == "near":
            return self._execute_near(entity_id, fragment)
        if fragment.mode() == "py_scan":
            return super().execute(fragment)
        if entity_id and (fragment.kind() == "chroma_where" or fragment.op in LOGICAL_OPS):
            if not self._is_chroma_fragment(fragment):
                return super().execute(fragment)
            try:
                # One read of every match binds a SQL variable per match, so the matching ids are
                # listed first and their rows read by id in bounded requests that match them again.
                return ResultFrame.from_rows(self._matched_rows(entity_id, fragment))
            except Exception as exc:
                self.client_error = safe_error(exc)
                raise RuntimeError(f"Chroma filter failed for {entity_id!r}: {self.client_error}") from exc
        raise NotImplementedError(f"Chroma cannot execute fragment {fragment.op!r}")

    def _execute_near(self, entity_id: str, fragment: QueryFragment) -> ResultFrame:
        params = fragment.payload
        collection = params.get("collection")
        filt = fragment.filter_fragment()
        if fragment.field != self._vector_field(entity_id):
            fragment = fragment.mark_scan()
        if fragment.mode() == "py_scan" or (filt is not None and not self._is_chroma_fragment(filt)):
            return self._scan_near(entity_id, fragment, collection)
        try:
            if collection is None:
                collection = self._collection(entity_id)
                space = _collection_space(collection)
                if params.get("metric", space) != space:
                    return self._scan_near(entity_id, fragment.mark_scan(), collection)
            else:
                space = params["collection_space"]
            where = self._where(filt) if filt is not None else None
            if where is not None and _where_variables(where) > CHROMA_MAX_WHERE_VARIABLES:
                return ResultFrame.from_rows(self._rank_matches(entity_id, fragment.field, params, filt, space, collection))
            query = params["vector"]
            if space == "cosine":
                # Chroma searches in float32; cosine ignores the exact power-of-two scale.
                query = _scale_cosine_query(query)
            kwargs = {"query_embeddings": [query], "n_results": params["top_k"], "include": ["distances", "metadatas", "embeddings"]}
            if where is not None:
                kwargs["where"] = where
            if params["top_k"] > CHROMA_MAX_GET_IDS:
                # Chroma reads the metadata of every hit in one statement, a SQL variable per hit.
                kwargs["include"] = []
                hits = collection.query(**kwargs)
                rows = self._rows_by_ids(entity_id, hits.get("ids", [[]])[0], where, collection=collection)
                # A writer may change an embedding between requests. Rank the embeddings we return.
                return ResultFrame.from_rows(_rank_chroma_rows(rows, fragment.field, query, space))
            rows = collection.query(**kwargs)
            return ResultFrame.from_rows(
                self._rows_from_chroma(
                    entity_id,
                    rows.get("ids", [[]])[0],
                    (rows.get("metadatas") or [[]])[0],
                    (rows.get("embeddings") or [[]])[0] if rows.get("embeddings") is not None else None,
                    rows.get("distances", [[]])[0],
                )
            )
        except Exception as exc:
            self.client_error = safe_error(exc)
            raise RuntimeError(f"Chroma query failed for {entity_id!r}: {self.client_error}") from exc

    def _scan_near(self, entity_id: str, fragment: QueryFragment, collection: Any) -> ResultFrame:
        if collection is None:
            return super().execute(fragment)
        rows = (row for page in self._id_pages(entity_id, CHROMA_MAX_GET_IDS, "scan", collection=collection) for row in page)
        return execute_scan(self, fragment, rows=rows)

    def to_dict(self) -> dict[str, Any]:
        """Return Chroma backend metadata and connection state.

        Args:
            None.

        Returns:
            dict[str, Any]: Backend metadata, storage path, database name, and current
            provider connection status.
        """
        return {
            "name": self.name,
            "type": self.type,
            "path": self.path,
            "database": self.database,
            "connected": self.client is not None,
            "error": self.client_error,
        }
