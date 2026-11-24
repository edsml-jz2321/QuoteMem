"""Pinecone vector backend."""

from __future__ import annotations

__all__ = [
    "PineconeBackend",
]

import time
import weakref
from collections import OrderedDict
from typing import NoReturn

from heavenbase.utils import Any, Iterable, Mapping, Sequence, safe_error
from ...execution.frame.result import ResultFrame
from ...modules import ModuleResolver
from ...schema.model import EntitySchema
from ...types.vector import _scale_cosine_query
from ..base import Backend, backend_locked
from ._metadata import ENCODED_PREFIX, FLOAT64_EXACT_INTS, metadata_entry, metadata_fields
from heavenbase.execution.contracts import QueryFragment, RowOp

PINECONE_INDEX_MAX_LENGTH = 45
PINECONE_FETCH_MAX_IDS = 1000
# Response headers that report how far the index view behind a read has caught up with the write log
# (Pinecone's read-after-write guidance; the second name is the one pinecone>=9 parses).
PINECONE_VIEW_LSN_HEADERS = ("x-pinecone-max-indexed-lsn", "x-pinecone-lsn-reconciled")
# Query matches remembered per backend, with the view LSN of the query that returned them (a wait hint only).
PINECONE_QUERY_SEEN_MAX = 10_000
# Every live Pinecone backend, so that ensure() can tell which other backends of a workspace are Pinecone backends too.
# Keyed by id() with weak values, so a backend subclass need not be hashable (a dataclass subclass is not).
_LIVE_PINECONE_BACKENDS: "weakref.WeakValueDictionary[int, PineconeBackend]" = weakref.WeakValueDictionary()


def _view_lsn(response: Any) -> int | None:
    """Return the index-view LSN a Pinecone read response reports, or ``None`` when it reports none."""
    info = getattr(response, "response_info", None) or getattr(response, "_response_info", None)
    if info is None and isinstance(response, Mapping):
        info = response.get("response_info")
    headers = info.get("raw_headers") if isinstance(info, Mapping) else getattr(info, "raw_headers", None)
    if not isinstance(headers, Mapping):
        return None
    for name in PINECONE_VIEW_LSN_HEADERS:
        value = headers.get(name)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                return None
    return None


def _is_local_host(host: Any) -> bool:
    text = str(host).lower()
    return text.startswith(("http://localhost", "https://localhost", "localhost", "http://127.0.0.1", "https://127.0.0.1", "127.0.0.1"))


class PineconeBackend(Backend):
    """Pinecone provider adapter with native vector search and metadata filters.

    Uses the Pinecone client for persistence, row lookup, native ``near``
    search, and metadata filtering through provider storage only.

    Attributes:
        miss_settle_seconds (float): Upper bound, in seconds, of the wait in
            ``settle_query_misses`` for a lagging fetch to return a record
            that a query returned.
    """

    miss_settle_seconds: float = 5.0

    identifier = "pinecone"
    provider_max_name_length = PINECONE_INDEX_MAX_LENGTH

    @classmethod
    def from_config(
        cls,
        name: str,
        data: Mapping[str, Any],
        *,
        resolver: ModuleResolver | None = None,
        config: Any | None = None,
    ) -> "PineconeBackend":
        """Build the Pinecone backend from a config mapping.

        Args:
            name (str): Workspace backend instance name.
            data (Mapping[str, Any]): Backend config mapping. Supported keys are Pinecone
                constructor parameters, including ``host``, ``api_key``, and ``ws_id``.
            resolver: Context-owned module authority.
            config: Config authority retained by the backend.

        Returns:
            PineconeBackend: Configured Pinecone backend instance.
        """
        cfg = dict(data or {})
        ws_id = cfg.pop("ws_id", None)
        return cls(name, cfg.pop("host", None), cfg.pop("api_key", None), ws_id=ws_id, resolver=resolver, config=config, **cfg)

    def __init__(
        self,
        name: str = "pinecone",
        host: str | None = None,
        api_key: str | None = None,
        ws_id: str | None = None,
        *,
        resolver: ModuleResolver | None = None,
        config: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """Create a Pinecone backend.

        Args:
            name (str): Workspace backend instance name.
            host (str | None): Pinecone control-plane host. When omitted, uses configured
                backend defaults.
            api_key (str | None): Pinecone API key. Local hosts default to ``pclocal`` when
                no key is configured.
            ws_id (str | None): Optional workspace namespace used to prefix index names.
            resolver: Context-owned module authority.
            config: Config authority retained by the backend.
            **kwargs (Any): Extra keyword arguments passed to ``Pinecone(...)``.

        Returns:
            None: This function does not return a value.
        """
        super().__init__(name, ws_id=ws_id, resolver=resolver, config=config)
        self.host = host or self._construction_config_get("heavenbase.backend_defaults.pinecone.host", default="http://localhost:5080")
        local = _is_local_host(self.host)
        configured_api_key = self._construction_config_get("heavenbase.backend_defaults.pinecone.api_key", default="")
        self.api_key = api_key or configured_api_key or ("pclocal" if local else "")
        default_ssl = not (str(self.host).startswith("http://") or local)
        kwargs.setdefault(
            "ssl_verify",
            self._construction_config_get("heavenbase.backend_defaults.pinecone.ssl_verify", default=default_ssl),
        )
        self._client_kwargs = dict(kwargs)
        self._schemas: dict[str, EntitySchema] = {}
        self._owned_fields: dict[str, set[str] | None] = {}
        self._shared_entities: set[str] = set()
        self._fields_requiring_record: dict[str, frozenset[str]] = {}
        self._indexes: dict[str, str] = {}
        # (entity_id, object_id) -> view LSN of the latest query that returned the record, when it reported one.
        self._query_seen: OrderedDict[tuple[str, str], int] = OrderedDict()
        self.project_name = self._project_name()
        self.client_error = ""
        self._connect()
        _LIVE_PINECONE_BACKENDS[id(self)] = self

    def _project_name(self) -> str:
        if "project_name" in self._client_kwargs:
            return str(self._client_kwargs["project_name"])
        return self._ws_prefix()

    def _ws_prefix(self) -> str:
        # Workspace namespace used to prefix and reap this workspace's indexes. Derived from ws_id alone
        # (not the user-supplied client project_name) so client scoping and index ownership stay separate.
        return self._physical_name(self.ws_id, sep="-") if self.ws_id else ""

    def _connect(self) -> None:
        try:
            from pinecone import Pinecone

            kwargs = dict(self._client_kwargs)
            self.project_name = self._project_name()
            if self.project_name and "project_name" not in kwargs:
                kwargs["project_name"] = self.project_name
            self.client = Pinecone(api_key=self.api_key, host=self.host, **kwargs)
            self.client.list_indexes()
        except Exception as exc:
            self.client = None
            self.client_error = safe_error(exc)

    def _refresh_workspace(self) -> None:
        self._indexes.clear()
        self._query_seen.clear()
        self._connect()

    def _listed_index_names(self) -> list[str]:
        if self.client is None:
            return []
        indexes = self.client.list_indexes()
        if hasattr(indexes, "names"):
            return [str(name) for name in indexes.names()]
        if hasattr(indexes, "indexes"):
            indexes = indexes.indexes
        names = []
        for item in indexes or []:
            if isinstance(item, str):
                names.append(item)
            elif hasattr(item, "get") and item.get("name") is not None:
                names.append(str(item.get("name")))
            elif getattr(item, "name", None) is not None:
                names.append(str(item.name))
        return names

    @backend_locked
    def drop(self) -> None:
        """Drop this Pinecone project or workspace-prefixed indexes.

        Args:
            None.

        Returns:
            None: This function does not return a value.
        """
        try:
            if self.client is None:
                return
            delete_project = getattr(self.client, "delete_project", None)
            if delete_project is not None and self.project_name:
                try:
                    delete_project(name=self.project_name)
                except TypeError:
                    delete_project(self.project_name)
                return
            names = set(self._indexes.values())
            prefix = f"{self._ws_prefix()}-" if self.ws_id else ""
            if prefix:
                names.update(name for name in self._listed_index_names() if name.startswith(prefix))
            delete_index = getattr(self.client, "delete_index", None)
            if delete_index is not None:
                for name in sorted(names):
                    delete_index(name)
        finally:
            self._schemas.clear()
            self._owned_fields.clear()
            self._shared_entities.clear()
            self._fields_requiring_record.clear()
            self._indexes.clear()
            self._query_seen.clear()
            self.client = None

    def _need_client(self) -> None:
        if self.client is None:
            detail = f": {self.client_error}" if self.client_error else ""
            raise RuntimeError(f"Pinecone backend {self.name!r} is unavailable{detail}")

    def _index_name(self, entity_id: str) -> str:
        prefix = self._ws_prefix()
        if not prefix:
            return self._indexes.setdefault(entity_id, self._physical_name(entity_id, sep="-"))
        entity_part = self._physical_name(entity_id, sep="-")
        return self._indexes.setdefault(entity_id, f"{prefix}-{entity_part}")

    def schema_warnings(self, schema: EntitySchema, bindings: Sequence[Any]) -> list[str]:
        """Return Pinecone index-name warnings for this schema.

        Args:
            schema (EntitySchema): Entity schema being registered.
            bindings (Sequence[Any]): Storage bindings for the entity.

        Returns:
            list[str]: Provider naming warnings for physical index names.
        """
        warnings = super().schema_warnings(schema, bindings)
        warnings.extend(
            self._physical_name_warnings(
                [(schema.entity_id, self._index_name(schema.entity_id))],
                kind="Pinecone index",
                max_length=self.provider_max_name_length,
                case_sensitive=False,
            )
        )
        return warnings

    def _vector_field(self, entity_id: str) -> Any:
        """Return the entity's Vector field stored on this backend, or ``None`` when none is."""
        schema = self._schemas.get(entity_id)
        if schema is None:
            return None
        owned = self._owned_fields.get(entity_id)
        return next(
            (
                field_schema
                for name, field_schema in schema.fields.items()
                if type(field_schema.dtype).identifier == "vector" and (owned is None or name in owned)
            ),
            None,
        )

    def _dim(self, entity_id: str) -> int:
        vector = self._vector_field(entity_id)
        return vector.dtype.dim if vector is not None else 0

    def _index(self, name: str) -> Any:
        return self._index_at(self.client.describe_index(name).host)

    def _index_at(self, host: str) -> Any:
        if "://" not in host:
            scheme = "http" if str(self.host).startswith("http://") or _is_local_host(self.host) else "https"
            host = f"{scheme}://{host}"
        kwargs = {"host": host}
        if str(host).startswith("http://"):
            kwargs["ssl_verify"] = False
        return self.client.Index(**kwargs)

    def _has_index(self, name: str) -> bool:
        return bool(self.client and self.client.has_index(name))

    def near_metric(self, entity_id: str) -> str:
        """Return the metric a near on one entity's index ranks by.

        An index keeps the metric it was created with, so a later change of
        ``heavenbase.backend_defaults.pinecone.metric`` does not change how an
        existing index ranks. Before the index can exist (no client, an entity
        not stored here, or no index yet) the configured metric, which a new
        index gets, is returned.

        Args:
            entity_id (str): Logical entity identifier.

        Returns:
            str: Lower-case Pinecone metric (``"cosine"``, ``"euclidean"`` or ``"dotproduct"``).

        Raises:
            RuntimeError: If the provider cannot check or describe the index.
        """
        return self._near_index(entity_id)[0]

    @backend_locked
    def _near_index(self, entity_id: str) -> tuple[str, str | None, bool]:
        """Return the live metric and data-plane host for one near compilation.

        The final flag says whether the host result is authoritative. A null
        host with that flag set means that the index is absent.
        """
        configured = str(self.config.get("heavenbase.backend_defaults.pinecone.metric", default="cosine")).lower()
        if self.client is None or entity_id not in self._schemas:
            return configured, None, False
        name = self._index_name(entity_id)
        try:
            if not self._has_index(name):
                return configured, None, True
            description = self.client.describe_index(name)
        except Exception as exc:
            raise RuntimeError(f"Pinecone operation failed: {safe_error(exc)}") from exc
        metric = getattr(description, "metric", None)
        if metric is None and isinstance(description, Mapping):
            metric = description.get("metric")
        host = getattr(description, "host", None)
        if host is None and isinstance(description, Mapping):
            host = description.get("host")
        effective_metric = configured if metric is None else str(getattr(metric, "value", metric)).lower()
        return effective_metric, str(host) if host is not None else None, host is not None

    def _degrade(self, exc: BaseException) -> NoReturn:
        self.client_error = safe_error(exc)
        self.client = None
        raise RuntimeError(f"Pinecone operation failed: {self.client_error}") from exc

    def _metadata(self, entity_id: str, values: dict[str, Any]) -> dict[str, Any]:
        """Return the Pinecone metadata that stores one row's fields.

        Pinecone metadata holds strings, numbers, booleans and lists of strings, and its
        numbers are 64-bit floats. A text Array is stored as a list of strings; any other
        value, such as an Integer Array, an empty Array or an integer beyond +/-2**53, is
        stored as tagged JSON under its encoded key (see
        ``heavenbase.backends.vector._metadata``). Json values arrive as their JSON text
        and Artifact values as base64 text. The vector Pinecone stores as the record's
        values is not metadata. An explicit null in a nullable field with a
        default is kept under the encoded key.
        """
        schema = self._schemas.get(entity_id)
        vector = self._vector_field(entity_id)
        out: dict[str, Any] = {}
        for key, value in values.items():
            if key == "object_id" or (vector is not None and key == vector.name):
                continue
            field_schema = schema.fields.get(key) if schema else None
            native, encoded = metadata_entry(
                value,
                lambda item: field_schema is not None and self._is_metadata_string_array(field_schema, item),
                FLOAT64_EXACT_INTS,
                preserve_null=field_schema is not None and field_schema.has_default and not field_schema.required,
            )
            if native is not None:
                out[key] = native
            if encoded is not None:
                out[ENCODED_PREFIX + key] = encoded
        return out

    def _is_metadata_string_array(self, field_schema: Any, value: Any) -> bool:
        item_type = getattr(field_schema.dtype, "item_type", None)
        return (
            item_type is not None
            and type(item_type).identifier in ("identifier", "long-text", "medium-text", "short-text")
            and isinstance(value, list)
            and bool(value)
            and all(isinstance(item, str) for item in value)
        )

    @backend_locked
    def ensure(self, schema: EntitySchema, bindings: Sequence[Any]) -> None:
        """Ensure a Pinecone index for the entity's vector field.

        A Pinecone record is one dense vector with its metadata, and it cannot exist
        without the vector. The fields routed here must therefore include exactly one
        Vector field, with a fixed dimension; that vector is the record's values and
        every other routed field is metadata. See ``upsert`` for rows without a vector.

        Args:
            schema (EntitySchema): Entity schema to store in Pinecone.
            bindings (Sequence[Any]): Storage bindings supplied by the workspace. A
                direct caller whose bindings route nothing to this backend stores the
                whole schema here.

        Returns:
            None: This function does not return a value.

        Raises:
            ValueError: If the fields routed here hold no Vector field or more than one,
                the Vector field has no fixed dimension, or the entity's existing index
                has another dimension.
        """
        owned = {binding.field_name for binding in bindings if getattr(binding, "backend", None) == self.name and binding.field_name != schema.pk_name}
        routed = owned or {name for name in schema.fields if name != schema.pk_name}
        vectors = [field_schema for name, field_schema in schema.fields.items() if name in routed and type(field_schema.dtype).identifier == "vector"]
        if not vectors:
            raise ValueError(
                f"Pinecone backend {self.name!r} cannot store {schema.entity_id!r} without a routed Vector field, because a Pinecone "
                f"record cannot exist without its dense vector; routed fields: {sorted(routed)}"
            )
        if len(vectors) > 1:
            raise ValueError(
                f"Pinecone backend {self.name!r} stores one dense vector per record, but {schema.entity_id!r} routes "
                f"{len(vectors)} Vector fields to it: {[field_schema.name for field_schema in vectors]}; route the others to another backend"
            )
        vector = vectors[0]
        if vector.dtype.dim is None:
            raise ValueError(f"Pinecone indexes require a fixed-dimension Vector field; {schema.entity_id}.{vector.name} has none")
        index_name = self._index_name(schema.entity_id)
        exists = self._has_index(index_name)
        if exists:
            self._check_index_dimension(index_name, schema.entity_id, vector)
        self._schemas[schema.entity_id] = schema
        # A direct caller without bindings for this backend stores the whole schema here.
        self._owned_fields[schema.entity_id] = owned or None
        holders = {
            name for name in {getattr(binding, "backend", None) for binding in bindings} - {self.name, None} if self._keeps_every_row(schema, bindings, name)
        }
        replicated = {binding.field_name for binding in bindings if getattr(binding, "backend", None) in holders}
        self._fields_requiring_record[schema.entity_id] = frozenset(
            field.name for field in schema.fields.values() if field.name in routed - replicated and (field.required or field.has_default)
        )
        if owned and holders:
            self._shared_entities.add(schema.entity_id)
        else:
            self._shared_entities.discard(schema.entity_id)
        if self.client is None or exists:
            return
        metric = str(self.config.get("heavenbase.backend_defaults.pinecone.metric", default="cosine")).lower()
        from pinecone import ServerlessSpec

        cloud = self.config.get("heavenbase.backend_defaults.pinecone.serverless.cloud", default="aws")
        region = self.config.get("heavenbase.backend_defaults.pinecone.serverless.region", default="us-east-1")
        self.client.create_index(
            name=index_name,
            dimension=vector.dtype.dim,
            metric=metric,
            spec=ServerlessSpec(cloud=cloud, region=region),
        )

    def _keeps_every_row(self, schema: EntitySchema, bindings: Sequence[Any], backend_name: str) -> bool:
        """Return whether another backend of the workspace keeps every row of the entity it is given.

        Every backend keeps every row except a Pinecone backend, which has no record for a row
        without its vector. A Pinecone backend keeps every row only when the Vector field routed
        to it is required, so that no row lacks it.

        Args:
            schema (EntitySchema): Entity schema being registered.
            bindings (Sequence[Any]): Storage bindings of the whole entity.
            backend_name (str): Workspace backend instance name of the other backend.

        Returns:
            bool: Whether that backend holds every row of the entity.
        """
        if not any(other is not self and other.name == backend_name and other.ws_id == self.ws_id for other in list(_LIVE_PINECONE_BACKENDS.values())):
            return True
        routed = {binding.field_name for binding in bindings if getattr(binding, "backend", None) == backend_name}
        return any(field_schema.required for name, field_schema in schema.fields.items() if name in routed and type(field_schema.dtype).identifier == "vector")

    def _check_index_dimension(self, index_name: str, entity_id: str, vector: Any) -> None:
        """Refuse an existing index whose dimension differs from the routed Vector field.

        Args:
            index_name (str): Existing Pinecone index name.
            entity_id (str): Entity identifier the index belongs to.
            vector (Any): Field schema of the Vector field routed to Pinecone.

        Returns:
            None: This function does not return a value.

        Raises:
            ValueError: If the index was created with another dimension. HeavenBase
                0.1.2.2 sized an index from the first declared Vector field, which need
                not be the one routed to Pinecone.
        """
        description = self.client.describe_index(index_name)
        dimension = description.get("dimension") if isinstance(description, Mapping) else getattr(description, "dimension", None)
        if dimension is not None and int(dimension) != vector.dtype.dim:
            raise ValueError(
                f"Pinecone index {index_name!r} for {entity_id!r} has dimension {int(dimension)}, but the Vector field routed to "
                f"Pinecone, {entity_id}.{vector.name}, has dimension {vector.dtype.dim}; its records cannot hold that vector. "
                f"Delete the index {index_name!r} in Pinecone and register the entity again, then upsert its rows again"
            )

    @backend_locked
    def validate_missing_row(self, entity_id: str, object_id: str, *, for_patch: bool = False) -> None:
        """Reject a mutation that would rely on an inconclusive Pinecone fetch.

        A fetch can miss a record written recently, so a miss never shows that
        the record is absent. The workspace asks this only when no route that
        keeps every row of the entity has shown the row is absent.
        """
        if for_patch:
            raise ValueError(
                f"Pinecone backend {self.name!r} returned no record for entity {entity_id!r} object_id {object_id!r}; "
                "retry it once the record can be read, or use a full upsert if the record is absent"
            )
        raise ValueError(
            f"Pinecone backend {self.name!r} returned no record for entity {entity_id!r} object_id {object_id!r}, which does not show "
            "that the row is absent, because a fetch can miss a record written recently; if the row exists, retry it once the record can be read"
        )

    def validate_row_ops(self, rows: Sequence[RowOp]) -> None:
        """Reject fragments Pinecone cannot represent."""
        self._validate_upsert_batches(self.ordered_row_op_batches(rows))

    @backend_locked
    def upsert(self, rows: Sequence[RowOp]) -> None:
        """Upsert vector records into Pinecone.

        Args:
            rows (Sequence[RowOp]): Row mutations to apply. ``delete`` operations remove
                vectors; other operations upsert vector values and metadata.

        Returns:
            None: This function does not return a value.

        Raises:
            ValueError: If an upsert names an entity with no Vector field registered on
                this backend, or a row without its vector that Pinecone cannot leave out
                (see Notes). Nothing is written then.
            RuntimeError: If the Pinecone client is unavailable, the index of an upserted
                entity does not exist, or an operation fails.

        Notes:
            A Pinecone record cannot exist without its dense vector. A row whose vector is
            ``None``, or omitted from a whole replacement, therefore has no Pinecone
            record, and any record stored for its id is deleted, when another backend
            that keeps every row holds the entity, every other field the row routes
            here is ``None``. Any required or defaulted fields routed here must also be
            stored on backends that keep every row. Every backend but Pinecone keeps
            every row. Another Pinecone backend keeps every row only when the Vector
            field routed to it is required.
            The whole call is checked before the first write. Deletes of an entity
            whose index does not exist have nothing to remove.
        """
        self._need_client()
        batches = list(self.ordered_row_op_batches(rows))
        self._validate_upsert_batches(batches)
        indexes = self._upsert_indexes(batches)
        for entity_id, is_delete, ops in batches:
            index_name = indexes[entity_id]
            if index_name is None:
                continue
            vector = self._vector_field(entity_id)
            try:
                index = self._index(index_name)
                if is_delete:
                    index.delete(ids=[str(row.object_id) for row in ops])
                    continue
                vectors = []
                cleared = []
                for row in ops:
                    vec = row.values.get(vector.name)
                    if vec is None:
                        cleared.append(str(row.object_id))
                        continue
                    record = {"id": str(row.object_id), "values": vec}
                    metadata = self._metadata(entity_id, row.values)
                    if metadata:
                        record["metadata"] = metadata
                    vectors.append(record)
                if vectors:
                    index.upsert(vectors=vectors)
                if cleared:
                    index.delete(ids=cleared)
            except Exception as exc:
                self._degrade(exc)

    def _validate_upsert_batches(self, batches: Iterable[tuple[str, bool, list[RowOp]]]) -> None:
        """Check all writes against the record format."""
        for entity_id, is_delete, ops in batches:
            if is_delete:
                continue
            object_ids = [str(row.object_id) for row in ops]
            vector = self._vector_field(entity_id)
            if vector is None:
                raise ValueError(
                    f"Pinecone backend {self.name!r} has no Vector field registered for {entity_id!r}, so it cannot store rows "
                    f"{object_ids[:8]}; register the entity with its Vector field routed to this backend"
                )
            shared = entity_id in self._shared_entities
            requires_record = not shared or bool(self._fields_requiring_record.get(entity_id))
            lost = [
                object_id
                for object_id, row in zip(object_ids, ops)
                if row.values.get(vector.name) is None
                and (requires_record or any(value is not None for key, value in row.values.items() if key not in ("object_id", vector.name)))
            ]
            if lost:
                where = "fields routed to Pinecone" if shared else "these rows, which no other backend keeps,"
                raise ValueError(
                    f"Pinecone cannot store {entity_id!r} rows {lost[:8]} without their {vector.name!r} vector: a Pinecone record "
                    f"cannot exist without its dense vector, so {where} would be lost"
                )

    def _upsert_indexes(self, batches: Sequence[tuple[str, bool, list[RowOp]]]) -> dict[str, str | None]:
        """Resolve index names before writing and reject upserts to missing indexes."""
        indexes: dict[str, str | None] = {}
        for entity_id, is_delete, ops in batches:
            if entity_id not in indexes:
                index_name = self._index_name(entity_id)
                indexes[entity_id] = index_name if self._has_index(index_name) else None
            if not is_delete and indexes[entity_id] is None:
                raise RuntimeError(f"Pinecone index {self._index_name(entity_id)!r} for {entity_id!r} does not exist; refusing to discard {len(ops)} upsert(s)")
        return indexes

    def _hit_rows(self, res: Any, entity_id: str) -> list[dict[str, Any]]:
        rows = []
        matches = res.get("matches") if hasattr(res, "get") else getattr(res, "matches", [])
        for match in matches or []:
            row_id = match.get("id") if hasattr(match, "get") else getattr(match, "id", None)
            metadata = match.get("metadata") if hasattr(match, "get") else getattr(match, "metadata", {})
            row = metadata_fields(metadata)
            row["object_id"] = str(row_id)
            row["score"] = match.get("score", 0.0) if hasattr(match, "get") else getattr(match, "score", 0.0)
            rows.append(row)
        return rows

    @backend_locked
    def scan(self, entity_id: str) -> list[dict[str, Any]]:
        """Reject unbounded scans because Pinecone has no row scan API.

        Args:
            entity_id (str): Entity identifier that would be scanned.

        Returns:
            list[dict[str, Any]]: This method never returns rows.

        Raises:
            RuntimeError: If the Pinecone client is unavailable.
            NotImplementedError: Always raised for reachable clients.
        """
        self._need_client()
        raise NotImplementedError("Pinecone scan is unsupported; use fetch(ids=...) or native near search")

    def _vector_map(self, res: Any) -> dict[str, Any]:
        vectors = res.get("vectors") if hasattr(res, "get") else getattr(res, "vectors", {})
        return dict(vectors or {})

    def _fetch_vector_map(self, index: Any, object_ids: Sequence[str]) -> dict[str, Any]:
        """Fetch unique vector ids in provider-sized batches."""
        unique_ids = list(dict.fromkeys(object_ids))
        vectors = {}
        for offset in range(0, len(unique_ids), PINECONE_FETCH_MAX_IDS):
            batch = unique_ids[offset : offset + PINECONE_FETCH_MAX_IDS]
            vectors.update(self._vector_map(index.fetch(ids=batch)))
        return vectors

    def _vector_item_get(self, item: Any, key: str, default: Any = None) -> Any:
        return item.get(key, default) if hasattr(item, "get") else getattr(item, key, default)

    def _row_from_vector(self, entity_id: str, object_id: Any, item: Any) -> dict[str, Any]:
        row = metadata_fields(self._vector_item_get(item, "metadata", {}))
        row["object_id"] = str(object_id)
        vector = self._vector_field(entity_id)
        values = self._vector_item_get(item, "values")
        if vector is not None and values is not None:
            row[vector.name] = values.tolist() if hasattr(values, "tolist") else list(values)
        return row

    @backend_locked
    def get_many(
        self, entity_id: str, object_ids: Iterable[Any], *, columns: Sequence[str] | None = None, _near_fragment: QueryFragment | None = None
    ) -> list[dict[str, Any] | None]:
        """Return rows through Pinecone fetch.

        Args:
            entity_id (str): Entity identifier whose vectors should be loaded.
            object_ids (Iterable[Any]): Object identifiers to fetch in caller order.
            columns (Sequence[str] | None): Optional field names to return; ``None`` returns the complete fragment.
            _near_fragment (QueryFragment | None): Internal query route for hydrating near hits.

        Returns:
            list[dict[str, Any] | None]: Rows aligned with ``object_ids``. Missing vectors
            are returned as ``None``.

        Raises:
            TypeError: If ``columns`` contains a non-string name.
            RuntimeError: If the Pinecone client is unavailable or fetch fails.
        """
        selection = self._row_columns(columns)
        ordered_ids = [str(object_id) for object_id in object_ids]
        if not ordered_ids:
            return []
        index_name = self._index_name(entity_id)
        self._need_client()
        host = _near_fragment.payload.get("index_host") if _near_fragment is not None and _near_fragment.kind() == "pinecone_search" else None
        if host is None and not self._has_index(index_name):
            return [None for _ in ordered_ids]
        try:
            index = self._index(index_name) if host is None else self._index_at(host)
            vectors = self._fetch_vector_map(index, ordered_ids)
            rows_by_id = {str(row_id): self._row_from_vector(entity_id, row_id, item) for row_id, item in vectors.items()}
            return self._project_rows([rows_by_id.get(row_id) for row_id in ordered_ids], selection)
        except Exception as exc:
            self._degrade(exc)

    @backend_locked
    def exists_many(self, entity_id: str, object_ids: Iterable[Any]) -> list[bool]:
        """Return existence flags through Pinecone fetch.

        Args:
            entity_id (str): Entity identifier whose vectors should be checked.
            object_ids (Iterable[Any]): Object identifiers to check in caller order.

        Returns:
            list[bool]: Existence flags aligned with ``object_ids``.

        Raises:
            RuntimeError: If the Pinecone client is unavailable or fetch fails.
        """
        ordered_ids = [str(object_id) for object_id in object_ids]
        if not ordered_ids:
            return []
        index_name = self._index_name(entity_id)
        self._need_client()
        if not self._has_index(index_name):
            return [False for _ in ordered_ids]
        try:
            found = {str(row_id) for row_id in self._fetch_vector_map(self._index(index_name), ordered_ids)}
            return [row_id in found for row_id in ordered_ids]
        except Exception as exc:
            self._degrade(exc)

    @backend_locked
    def count(self, entity_id: str) -> int:
        """Return Pinecone's provider-reported count without materializing vectors.

        Args:
            entity_id (str): Entity identifier whose index should be counted.

        Returns:
            int: Provider-reported total vector count, or ``0`` when the index is absent.

        Raises:
            RuntimeError: If the Pinecone client is unavailable or stats lookup fails.

        Notes:
            Pinecone is eventually consistent, so this value is not used as an
            exhaustive logical-identity proof by portable query folds.
        """
        index_name = self._index_name(entity_id)
        self._need_client()
        if not self._has_index(index_name):
            return 0
        try:
            stats = self._index(index_name).describe_index_stats()
            return int(stats.get("total_vector_count", 0) if hasattr(stats, "get") else getattr(stats, "total_vector_count", 0) or 0)
        except Exception as exc:
            self._degrade(exc)

    def exact_count(self, entity_id: str) -> int:
        """Reject use of eventually consistent Pinecone stats as an exact proof.

        Args:
            entity_id: Logical entity identifier whose fragments would be counted.

        Returns:
            int: This method never returns a count.

        Raises:
            NotImplementedError: Always, because provider stats are eventually consistent.
        """
        raise NotImplementedError(f"Pinecone cannot prove an exact fragment count for {entity_id!r}")

    def _live_tag_claims(self) -> dict[str, Any]:
        """Return live tag claims derived from Pinecone availability.

        Args:
            None.

        Returns:
            dict[str, Any]: Live Backend tag claims.
        """
        return {
            **super()._live_tag_claims(),
            "backend-availability": "ready" if self.client is not None else "unavailable",
        }

    def _find_near_child(self, fragment: QueryFragment) -> QueryFragment | None:
        if fragment.op == "near" and fragment.kind() == "pinecone_search":
            return fragment
        if fragment.op in ("and", "or"):
            for child in fragment.children():
                found = self._find_near_child(child)
                if found is not None:
                    return found
        return None

    def _is_pinecone_filter_child(self, fragment: QueryFragment) -> bool:
        if fragment.kind() == "pinecone_filter":
            return True
        if fragment.op in ("and", "or"):
            return all(self._is_pinecone_filter_child(child) for child in fragment.children())
        return False

    def _to_pinecone_filter(self, fragment: QueryFragment) -> dict:
        if fragment.kind() == "pinecone_filter":
            return fragment.payload
        if fragment.op == "and":
            return {"$and": [self._to_pinecone_filter(child) for child in fragment.children()]}
        if fragment.op == "or":
            return {"$or": [self._to_pinecone_filter(child) for child in fragment.children()]}
        raise TypeError(f"Cannot compile {fragment.op!r} to Pinecone filter")

    def _build_filter(self, fragments: list[QueryFragment]) -> dict | None:
        if not fragments:
            return None
        if len(fragments) == 1:
            return self._to_pinecone_filter(fragments[0])
        return {"$and": [self._to_pinecone_filter(f) for f in fragments]}

    def _query(self, index_name: str, near_fragment: QueryFragment, filter_dict: dict | None = None, entity_id: str = "") -> ResultFrame:
        params = near_fragment.payload
        host = params.get("index_host")
        index = self._index_at(host) if host is not None else self._index(index_name)
        vector = params["vector"]
        if params.get("metric") == "cosine":
            # Pinecone keeps float32 values; cosine ignores the exact power-of-two scale.
            vector = _scale_cosine_query(vector)
        kwargs = {"vector": vector, "top_k": params["top_k"], "include_metadata": True}
        if filter_dict:
            kwargs["filter"] = filter_dict
        res = index.query(**kwargs)
        rows = self._hit_rows(res, entity_id)
        self._remember_query_matches(entity_id, rows, _view_lsn(res))
        if params.get("metric") == "euclidean":
            # A euclidean index reports a distance, nearest (lowest) first; negate it so a
            # higher score is a nearer row, as on every other near route.
            for row in rows:
                row["score"] = -float(row["score"])
        return ResultFrame.from_rows(rows)

    def _remember_query_matches(self, entity_id: str, rows: Sequence[Mapping[str, Any]], lsn: int | None) -> None:
        """Remember the view LSN of this query for each record it returned; forget an older query's LSN otherwise."""
        if not entity_id:
            return
        for row in rows:
            key = (entity_id, str(row["object_id"]))
            self._query_seen.pop(key, None)
            if lsn is not None:
                self._query_seen[key] = lsn
        while len(self._query_seen) > PINECONE_QUERY_SEEN_MAX:
            self._query_seen.popitem(last=False)

    @backend_locked
    def _fetch_for_settle(self, entity_id: str, object_ids: Sequence[str], host: str | None) -> tuple[dict[str, dict[str, Any]], int | None]:
        """Fetch ``object_ids`` once; return the rows found and the lowest view LSN the responses report."""
        index = self._index(self._index_name(entity_id)) if host is None else self._index_at(host)
        rows: dict[str, dict[str, Any]] = {}
        views: list[int | None] = []
        for offset in range(0, len(object_ids), PINECONE_FETCH_MAX_IDS):
            res = index.fetch(ids=list(object_ids[offset : offset + PINECONE_FETCH_MAX_IDS]))
            views.append(_view_lsn(res))
            for row_id, item in self._vector_map(res).items():
                rows[str(row_id)] = self._row_from_vector(entity_id, row_id, item)
        known = [view for view in views if view is not None]
        return rows, (min(known) if len(known) == len(views) else None)

    def settle_query_misses(self, entity_id: str, object_ids: Iterable[Any], *, _near_fragment: QueryFragment | None = None) -> dict[str, dict[str, Any]]:
        """Fetch again the records that a near query returned and a hydration fetch missed.

        A fetch can miss a record written recently, because Pinecone's reads are
        eventually consistent, so a miss never shows the record is absent. This
        fetches the missed records again with a short backoff, bounded by
        ``miss_settle_seconds`` plus the fetch in progress, and returns those a
        later fetch returns. The index-view LSN that Pinecone may report on a
        response (``x-pinecone-max-indexed-lsn``) only decides how long to wait:
        once a fetch reports a view at or past the view the query reported, the
        backend stops waiting for that record. Without a reported LSN, which
        Pinecone documents for queries but not for fetches, it fetches once more.
        These fetches only improve the hydration: an error on one of them, such
        as a timeout, ends the retries, keeps the client and returns the rows
        found so far, so the match keeps the values the query returned.

        Args:
            entity_id (str): Entity whose records were missed.
            object_ids (Iterable[Any]): Record ids a hydration fetch missed.
            _near_fragment (QueryFragment | None): The near fragment whose index route the query used.

        Returns:
            dict[str, dict[str, Any]]: Rows a later fetch returned, by object id.

        """
        pending = list(dict.fromkeys(str(object_id) for object_id in object_ids))
        if not pending or self.client is None:
            return {}
        host = _near_fragment.payload.get("index_host") if _near_fragment is not None and _near_fragment.kind() == "pinecone_search" else None
        seen = {object_id: self._query_seen.get((entity_id, object_id)) for object_id in pending}
        deadline = time.monotonic() + max(0.0, float(self.miss_settle_seconds))
        delay = 0.05
        found: dict[str, dict[str, Any]] = {}
        try:
            if host is None and not self._has_index(self._index_name(entity_id)):
                return {}
            while pending:
                rows, view = self._fetch_for_settle(entity_id, pending, host)
                found.update(rows)
                # Keep waiting only for a record whose query reported a view this fetch has not reached.
                pending = [
                    object_id for object_id in pending if object_id not in rows and view is not None and seen[object_id] is not None and view < seen[object_id]
                ]
                if not pending or time.monotonic() + delay > deadline:
                    break
                time.sleep(delay)
                delay = min(delay * 2, 0.5)
        except Exception:
            # A failed retry settles nothing; the hydration fetch already succeeded, so the query keeps its values.
            return found
        return found

    def _has_near_index(self, name: str, fragment: QueryFragment) -> bool:
        if "index_host" in fragment.payload:
            return fragment.payload["index_host"] is not None
        return self._has_index(name)

    @backend_locked
    def execute(self, fragment: QueryFragment) -> ResultFrame:
        """Execute Pinecone-native near and filter fragments when possible.

        Args:
            fragment (QueryFragment): Backend-ready query fragment compiled by the handler layer.

        Returns:
            ResultFrame: Matching rows and object identifiers.

        Raises:
            LookupError: If the fragment requires standalone non-near execution.
            NotImplementedError: If a near query carries a non-native filter fragment.
            RuntimeError: If the Pinecone client is unavailable or the provider operation fails.
        """
        entity_id = fragment.entity_id()
        index_name = self._index_name(entity_id) if entity_id else ""
        self._need_client()
        if entity_id and fragment.op == "near" and fragment.kind() == "pinecone_search":
            if not self._has_near_index(index_name, fragment):
                return ResultFrame.empty()
            filt = fragment.filter_fragment()
            if filt is not None and not self._is_pinecone_filter_child(filt):
                raise NotImplementedError(f"Pinecone cannot execute non-native filter fragment {filt.op!r}")
            try:
                return self._query(index_name, fragment, filter_dict=self._to_pinecone_filter(filt) if filt is not None else None, entity_id=entity_id)
            except Exception as exc:
                self._degrade(exc)
        if fragment.op == "and" and entity_id:
            near_child = self._find_near_child(fragment)
            if near_child is not None:
                other_children = [child for child in fragment.children() if child is not near_child]
                if all(self._is_pinecone_filter_child(child) for child in other_children):
                    if not self._has_near_index(index_name, near_child):
                        return ResultFrame.empty()
                    filter_dict = self._build_filter(other_children)
                    try:
                        return self._query(index_name, near_child, filter_dict=filter_dict, entity_id=entity_id)
                    except Exception as exc:
                        self._degrade(exc)
        if entity_id and not self._has_index(index_name):
            return ResultFrame.empty()
        raise LookupError(f"Pinecone does not support standalone {fragment.op!r} queries without a near vector query.")

    def to_dict(self) -> dict[str, Any]:
        """Return Pinecone backend metadata and connection state.

        Args:
            None.

        Returns:
            dict[str, Any]: Backend metadata, host, project, and current provider connection
            status.
        """
        return {
            "name": self.name,
            "type": self.type,
            "host": self.host,
            "project": self.project_name,
            "connected": self.client is not None,
            "error": self.client_error,
        }
