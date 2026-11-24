"""Private MetaSchema publishing service for workspaces."""

from __future__ import annotations

from heavenbase.utils import Any, Dict, Iterable, Iterator, List, Mapping, Optional, coerce_bool, deepcopy, dumps_json, loads_json, sha256hash

from ...backends.base import is_portable_layer
from ...backends.registry import BackendType
from ...execution.contracts import RowOp
from ...extensions.base import Extension
from ...filters.spec import Filter
from ...placement import StorageBinding, StoragePlan
from ...placement.index import IndexBinding, UniqueBinding
from ...schema.model import EntitySchema
from ...utils.ids import hash_id
from ..foundation import FIXED_ENTITY_IDS
from ..schema.options import _ROUTE_GENERATIONS_KEY, EntityOptions
from .entity import MetaSchema
from .projection import _MetaSchemaRegistryRows

_SCHEMA_ROW_KINDS = frozenset({"entity", "field", "storage", "index", "constraint"})


class _MetaSchemaService:
    """Own MetaSchema row ids, publishing, materialization, and writes."""

    def __init__(self, workspace: Any):
        self.workspace = workspace
        self.registry = _MetaSchemaRegistryRows(self)
        self._published_registry_digest: str = ""
        self.revision = 0

    def invalidate_readers(self) -> None:
        """Invalidate MetaSchema readers after an independent row mutation."""
        self.revision += 1

    def establish_entity_classes(
        self,
        rows: Iterable[Mapping[str, Any]],
        *,
        allowed_entities: frozenset[str] | None = None,
    ) -> List[type]:
        """Establish absent classes described by manual Entity-kind rows.

        This is the narrow command-ingress exception to MetaSchema's ordinary
        projection role. A definition-shaped, active Entity row delegates to
        the normal schema registration service, so built-in class, placement,
        and publication invariants stay centralized. The caller-authored
        MetaSchema row is still written independently and may intentionally be
        noncanonical or disagree with an already-present class.

        Args:
            rows: Logical MetaSchema rows before physical write planning.
            allowed_entities: Optional closed mutation scope used by MCP
                profiles. Both MetaSchema and a newly defined Entity must be in
                scope.

        Returns:
            Entity classes newly established by these rows.

        Raises:
            PermissionError: If class establishment exceeds a closed mutation
                scope.
            TypeError: If a definition-shaped ``meta`` payload is invalid.
            ValueError: If the Entity definition cannot be compiled or safely
                registered.
        """
        ws = self.workspace
        metaschema_id = MetaSchema.schema().entity_id
        established: List[type] = []
        for row in rows:
            if row.get("kind") != "entity":
                continue
            if not coerce_bool(row.get("active", True), "MetaSchema active"):
                continue
            meta = row.get("meta")
            if not isinstance(meta, Mapping) or "fields" not in meta:
                continue
            declared_id = str(meta.get("entity_id") or "").strip()
            if declared_id in FIXED_ENTITY_IDS:
                raise ValueError(f"Fixed workspace Entity {declared_id!r} cannot be defined through MetaSchema")
            if declared_id in ws.entities:
                if allowed_entities is not None and (metaschema_id not in allowed_entities or declared_id not in allowed_entities):
                    raise PermissionError("MetaSchema Entity definition exceeds the active entity scope")
                continue
            entity_cls = ws._schema.entity_cls(meta, allow_dict=True)
            entity_id = entity_cls.schema().entity_id
            if allowed_entities is not None and (metaschema_id not in allowed_entities or entity_id not in allowed_entities):
                raise PermissionError("MetaSchema Entity definition exceeds the active entity scope")
            if entity_id in ws.entities:
                continue
            recorded = self.options_of(meta)
            established.append(ws._schema.register(entity_cls, catalog=recorded.get("catalog"), gram=recorded.get("gram")))
        return established

    @staticmethod
    def options_of(meta: Mapping[str, Any]) -> Dict[str, Any]:
        """Return the registration options an entity row's ``meta`` records.

        A published entity row carries them under ``meta["options"]``; a manifest-shaped
        definition states them as top-level ``catalog``/``gram`` keys. A row written before
        registration options existed carries neither and yields an empty mapping.

        Args:
            meta (Mapping[str, Any]): ``meta`` payload of a MetaSchema entity row.

        Returns:
            Dict[str, Any]: The ``catalog`` and ``gram`` values the row states, when it states them.
        """
        options = meta.get("options")
        source = options if isinstance(options, Mapping) else meta
        return {key: source[key] for key in ("catalog", "gram") if key in source}

    def stored_options(self, entity_id: str) -> Optional[EntityOptions]:
        """Return the registration options recorded by the stored MetaSchema entity row.

        The row is read directly from the MetaSchema routes through
        :meth:`stored_rows`, so an entity row published by an earlier process,
        or by a release that knew no registration options, is seen without a
        live registration, whether the route's Backend type looks rows up
        natively or through the portable scan fallback.

        Args:
            entity_id (str): Entity identifier whose stored registration is wanted.

        Returns:
            EntityOptions | None: The stored options; the defaults for a row that
            records none; ``None`` when no entity row is stored.

        Raises:
            ValueError: If the stored row records an unsupported ``catalog`` mode.
            TypeError: If the stored row records a non-boolean ``gram`` value.
        """
        row_id = self.system_row_id("entity", entity_id)
        row = self.stored_rows([row_id]).get(row_id)
        if row is None:
            return None
        meta = row.get("meta")
        if isinstance(meta, str):
            meta = loads_json(meta)
        recorded = self.options_of(meta) if isinstance(meta, Mapping) else {}
        return EntityOptions(**recorded)

    def stored_rows(self, row_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """Read MetaSchema rows by id directly from the routed backends.

        Use native point lookups or native key filters where available. The first
        readable route that holds an id supplies its row.

        Args:
            row_ids (List[str]): MetaSchema row ids to read.

        Returns:
            Dict[str, Dict[str, Any]]: Raw stored rows by id, in route order; ids
            that no readable route holds are absent.
        """
        ws = self.workspace
        metaschema_id = MetaSchema.schema().entity_id
        if metaschema_id not in ws.entities or metaschema_id not in ws.storage_plans:
            return {}
        found: Dict[str, Dict[str, Any]] = {}
        for backend in self._readable_routes():
            pending = [row_id for row_id in dict.fromkeys(row_ids) if row_id not in found]
            if not pending:
                break
            found.update(self._rows_by_id(backend, pending))
        return found

    def _rows_by_id(self, backend: Any, ids: list[str]) -> Dict[str, Dict[str, Any]]:
        """Read fresh row fragments without a full scan when the route can filter keys."""
        if not ids:
            return {}
        if is_portable_layer(backend.get_many, "get_many"):
            rows = self._native_rows(backend, Filter("object_id", "in", ids))
            if rows is not None:
                return {row["object_id"]: row for row in rows}
        return {row_id: row for row_id, row in zip(ids, backend.get_many(MetaSchema.identifier, ids)) if row is not None}

    def _native_rows(self, backend: Any, predicate: Filter) -> Optional[List[Dict[str, Any]]]:
        """Use the registered native filter when it can read this MetaSchema route."""
        ws = self.workspace
        plan = ws.storage_plans[MetaSchema.identifier]
        if backend.name not in {binding.backend for binding in plan.field_bindings(predicate.field)}:
            return None
        fragment = ws._router._filters.compile_filter(predicate, MetaSchema, backend.name)
        if fragment.unsupported_reason() or fragment.uses_scan():
            return None
        return backend.execute(fragment.bind(MetaSchema.identifier, backend=backend.name)).rows()

    def _readable_routes(self) -> Iterator[Any]:
        """Yield available MetaSchema backends in route order."""
        ws = self.workspace
        for name in ws._row_store.routed_backend_names(MetaSchema.identifier):
            backend = ws.backends.get(name)
            availability = backend.tag("backend-availability")
            if availability is None or availability.value != "unavailable":
                yield backend

    def publish_workspace_rows(self) -> None:
        """Publish workspace, backend, and handler capability rows.

        The registry mirror is content-addressed per workspace instance: when
        the generated rows match the last published set exactly, the write is
        skipped, so each entity registration does not re-upsert the thousands
        of unchanged registry rows. Backend or registry changes alter the row
        content and therefore the digest, which forces a fresh publish. The
        projection also keeps the registry rows themselves while the module
        catalog, the handler registrations and the backends are unchanged, so
        an unchanged publish does not rebuild them either.
        """
        rows, digest = self.registry.workspace_rows_and_digest()
        if digest == self._published_registry_digest:
            return
        self.write_meta_rows([deepcopy(row) for row in rows])
        self._published_registry_digest = digest

    def publish_schema_rows(self, entity: type) -> None:
        """Publish entity, field, and storage rows into MetaSchema."""
        ws = self.workspace
        if MetaSchema.schema().entity_id in ws.entities:
            entity_id = entity.schema().entity_id
            self.write_meta_rows(self._schema_rows(entity, ws.storage_plans[entity_id], ws.entity_options.get(entity_id)))

    def delete_schema_rows(self, row_ids: list[str]) -> None:
        """Remove projection rows no longer part of the published schema."""
        if row_ids:
            backend = self.workspace.backends.get(self.system_backend_name(MetaSchema))
            backend.delete(MetaSchema.identifier, row_ids)

    def stored_schema_rows(self, entity_id: str, *, backend: Any = None) -> Dict[str, Dict[str, Any]]:
        """Read the complete persisted projection for an entity, including indexes and constraints."""
        found: Dict[str, Dict[str, Any]] = {}
        routes = self._readable_routes() if backend is None else iter((backend,))
        entity_row_id = self.system_row_id("entity", entity_id)
        for route in routes:
            entity_row = self._rows_by_id(route, [entity_row_id]).get(entity_row_id)
            if entity_row is None:
                continue
            meta = self._row_meta(entity_row)
            projection = meta.get("projection_ids") if isinstance(meta, Mapping) else None
            if isinstance(projection, list) and all(isinstance(row_id, str) for row_id in projection):
                rows = iter((entity_row, *self._rows_by_id(route, projection).values()))
            else:
                native = self._native_rows(route, Filter("subject_id", "eq", entity_id))
                rows = iter(route.iter_scan(MetaSchema.identifier, batch_size=256) if native is None else native)
            try:
                for row in rows:
                    if row.get("subject_id") == entity_id and row.get("kind") in _SCHEMA_ROW_KINDS:
                        found.setdefault(row["object_id"], deepcopy(row))
            finally:
                close = getattr(rows, "close", None)
                if close is not None:
                    close()
        return found

    def stored_storage(self, entity_id: str, stored: Mapping[str, Any]) -> tuple[EntitySchema, StoragePlan]:
        """Load the stored logical schema and exact physical bindings without compiling a class."""
        resolver = self.workspace.module_resolver
        entity_row = stored[self.system_row_id("entity", entity_id)]
        schema = EntitySchema.from_dict(self._row_meta(entity_row), resolver=resolver)
        bindings, indexes, constraints = [], [], []
        for row in stored.values():
            kind = row["kind"]
            if kind == "storage":
                binding = StorageBinding(**self._row_meta(row), resolver=resolver)
                if binding.field_name in schema.fields:
                    bindings.append(binding)
            elif kind in ("index", "constraint"):
                meta = self._row_meta(row)
                if set(meta["fields"]) <= schema.fields.keys():
                    if kind == "index":
                        indexes.append(IndexBinding(**meta))
                    else:
                        constraints.append(UniqueBinding(**meta))
        return schema, StoragePlan(entity_id, bindings, indexes, constraints)

    @staticmethod
    def _row_meta(row: Mapping[str, Any]) -> Mapping[str, Any]:
        meta = row["meta"]
        return loads_json(meta) if isinstance(meta, str) else meta

    def matches_stored_schema(self, entity: type, storage_plan: StoragePlan, options: EntityOptions, stored: Mapping[str, Any]) -> bool:
        """Check that reattachment would reuse the published schema and bindings."""
        expected = self._schema_rows(entity, storage_plan, options)
        expected_ids = {row["object_id"] for row in expected}
        if not expected_ids <= stored.keys():
            return False
        entity_meta = self._row_meta(stored[expected[0]["object_id"]])
        extra = stored.keys() - expected_ids
        if extra and (
            not isinstance(entity_meta, Mapping)
            or "projection_ids" in entity_meta
            or any(not self._is_obsolete_scalar_row(entity.schema().entity_id, stored[row_id]) for row_id in extra)
        ):
            return False
        for row in expected:
            meta = self._row_meta(stored[row["object_id"]])
            if not isinstance(meta, Mapping):
                return False
            proposed = row["meta"]
            if row["kind"] == "entity":
                if EntityOptions(**self.options_of(meta)) != options:
                    return False
                # Registry provenance can vary by Context. Old entity rows may omit default options.
                ignored = {"registration", "options", "catalog", "gram", "projection_ids", _ROUTE_GENERATIONS_KEY}
                meta = {key: value for key, value in meta.items() if key not in ignored}
                proposed = {key: value for key, value in proposed.items() if key not in ignored}
            if dumps_json(meta, compact=True, sort_keys=True) != dumps_json(proposed, compact=True, sort_keys=True):
                return False
        return True

    def _is_obsolete_scalar_row(self, entity_id: str, row: Mapping[str, Any]) -> bool:
        """Accept legacy leftover index rows only when the route proves the index is absent."""
        kind = row.get("kind")
        if kind not in ("index", "constraint"):
            return False
        meta = self._row_meta(row)
        if not isinstance(meta, Mapping) or "backend" not in meta or "fields" not in meta:
            return False
        backend = self.workspace.backends.get(meta["backend"])
        return backend._stored_scalar_index_exists(entity_id, meta["fields"], unique=kind == "constraint") is False

    def capture_schema_rows(self, entity: type, proposed: List[dict], *, stored: Optional[Mapping[str, Any]] = None) -> Dict[str, Dict[str, Any]]:
        """Save the projection and any other rows the proposed schema would overwrite.

        ``stored`` is the projection already read from the MetaSchema system backend, when the caller has it.
        """
        backend = self.workspace.backends.get(self.system_backend_name(MetaSchema))
        saved = deepcopy(dict(stored)) if stored is not None else self.stored_schema_rows(entity.schema().entity_id, backend=backend)
        pending = [row["object_id"] for row in proposed if row["object_id"] not in saved]
        saved.update(deepcopy(self._rows_by_id(backend, pending)))
        return saved

    def prepare_schema_rows(self, entity: type, storage_plan: StoragePlan, options: EntityOptions) -> List[dict]:
        """Build the exact row payload for publication and conditional recovery."""
        rows = self._schema_rows(entity, storage_plan, options)
        for row in rows:
            self.workspace._materializer.materialize(MetaSchema, row)
        return rows

    def restore_schema_rows(self, saved: Mapping[str, Any], attempted: Mapping[str, Any]) -> None:
        """Restore rows still holding exactly what the failed publication wrote."""
        entity_id = MetaSchema.identifier
        backend = self.workspace.backends.get(self.system_backend_name(MetaSchema))
        keys = list(attempted)

        def pending() -> list[RowOp]:
            current = self._rows_by_id(backend, keys)
            ops = []
            for row_id in keys:
                before, after, actual = (dumps_json(rows.get(row_id), compact=True, sort_keys=True) for rows in (saved, attempted, current))
                if before != after and actual == after:
                    previous = saved.get(row_id)
                    ops.append(RowOp(entity_id, row_id, previous) if previous is not None else RowOp(entity_id, row_id, op="delete"))
            return ops

        if pending():
            with backend._guard_registration_rows(entity_id, keys):
                ops = pending()
                if ops:
                    backend.upsert(ops)

    def _schema_rows(self, entity: type, storage_plan: StoragePlan, options: Optional[EntityOptions]) -> List[dict]:
        """Build the schema projection for the supplied registration."""
        ws = self.workspace
        schema = entity.schema()
        metadata = ws._schema_metadata
        entity_meta = schema.to_dict()
        entity_meta["derives"] = metadata.entity_derives(entity)
        if options is not None:
            entity_meta["options"] = options.to_dict()
        if generations := ws._schema.route_generations(schema.entity_id):
            entity_meta[_ROUTE_GENERATIONS_KEY] = generations
        registration = self.registry.registration_meta("entity", schema.entity_id)
        if registration:
            entity_meta["registration"] = registration
        rows = [self.build_system_row("entity", schema.entity_id, name=schema.name, desc=schema.desc, tags=schema.tags, meta=entity_meta)]
        for field_schema in schema.fields.values():
            meta = field_schema.to_dict()
            field_spec = metadata.field_spec(entity, field_schema.name)
            if field_spec.computes:
                meta["computes"] = [metadata.compute_dict(hook) for hook in field_spec.computes]
            if field_spec.query_computes:
                meta["query_computes"] = [metadata.query_compute_dict(hook) for hook in field_spec.query_computes]
            if query_capability := metadata.field_query_capability(entity, field_schema.name):
                meta["query_capabilities"] = query_capability
            rows.append(
                self.build_system_row(
                    "field",
                    f"{schema.entity_id}.{field_schema.name}",
                    subject_id=schema.entity_id,
                    field=field_schema.name,
                    name=field_schema.name,
                    dtype=type(field_schema.dtype).identifier,
                    desc=field_schema.desc,
                    tags=field_schema.tags,
                    meta=meta,
                )
            )
        storage_rows = []
        for binding in storage_plan.bindings:
            rid = f"{schema.entity_id}.{binding.field_name}.{binding.backend}.{binding.strategy}"
            backend_identifier = BackendType.identifier_of(ws.backends.get(binding.backend), resolver=ws.module_resolver)
            storage_rows.append(
                self.build_system_row(
                    "storage",
                    rid,
                    subject_id=schema.entity_id,
                    field=binding.field_name,
                    backend=binding.backend,
                    type=backend_identifier,
                    strategy=binding.strategy,
                    source=binding.source,
                    meta=binding.to_dict(),
                )
            )
        for binding in storage_plan.indexes:
            rid = f"{schema.entity_id}.{'-'.join(binding.fields)}.{binding.backend}.index"
            storage_rows.append(
                self.build_system_row(
                    "index",
                    rid,
                    subject_id=schema.entity_id,
                    field=",".join(binding.fields),
                    backend=binding.backend,
                    type="value",
                    source=binding.source,
                    meta=binding.to_dict(),
                )
            )
        for binding in storage_plan.constraints:
            rid = f"{schema.entity_id}.{'-'.join(binding.fields)}.{binding.backend}.unique"
            storage_rows.append(
                self.build_system_row(
                    "constraint",
                    rid,
                    subject_id=schema.entity_id,
                    field=",".join(binding.fields),
                    backend=binding.backend,
                    type="unique",
                    source=binding.source,
                    meta=binding.to_dict(),
                )
            )
        projection = rows + storage_rows
        entity_meta["projection_ids"] = [row["object_id"] for row in projection[1:]]
        return projection

    def publish_extension_row(self, extension: Extension, entities: List[type]) -> None:
        """Publish one enabled-extension row into MetaSchema."""
        meta = extension.to_dict()
        meta.pop("tags", None)
        meta["entities"] = [entity.schema().entity_id for entity in entities]
        registration = self.registry.registration_meta("extension", extension.identifier)
        if registration:
            meta["registration"] = registration
        self.publish_system_row(
            "extension",
            extension.identifier,
            name=extension.name,
            desc=extension.desc,
            tags=["extension"],
            target=extension.identifier,
            meta=meta,
        )

    def publish_system_row(self, category: str, row_id: str, **values: Any) -> None:
        """Publish one system metadata row into MetaSchema."""
        self.write_meta_rows([self.build_system_row(category, row_id, **values)])

    def build_system_row(self, category: str, row_id: str, **values: Any) -> dict:
        """Build one MetaSchema row, returning an empty row before bootstrap completes."""
        ws = self.workspace
        meta = values.pop("meta", None)
        field_name = values.pop("field", "")
        if MetaSchema.schema().entity_id not in ws.entities:
            return {}
        return {
            "object_id": self.system_row_id(category, row_id),
            "kind": category,
            "subject_id": values.pop("subject_id", row_id),
            "name": values.pop("name", row_id),
            "dtype": values.pop("dtype", ""),
            "desc": values.pop("desc", ""),
            "tags": values.pop("tags", []),
            "backend": values.pop("backend", ""),
            "type": values.pop("type", ""),
            "strategy": values.pop("strategy", ""),
            "source": values.pop("source", ""),
            "target": values.pop("target", ""),
            "meta": meta if meta is not None else dict(values),
            "active": True,
            "ws": ws.id,
            "field": field_name,
        }

    def system_row_id(self, category: str, row_id: str) -> str:
        """Return a stable system row id that fits the Identifier limit."""
        raw_id = f"{category}:{row_id}"
        max_chars = MetaSchema.schema().get("object_id").dtype.max_chars
        if len(raw_id) <= max_chars:
            return raw_id
        prefix = f"{category}:"
        hash_length = int(self.workspace.context.config.get("heavenbase.entity.id.hash_length", default=42))
        hashed = hash_id(category, row_id, length=hash_length)
        if len(prefix) + len(hashed) <= max_chars:
            return f"{prefix}{hashed}"
        return hashed

    def write_meta_rows(self, rows: List[dict], *, invalidate_mcp_registry_cache: bool = True) -> None:
        """Write MetaSchema rows."""
        self.write_system_rows(MetaSchema, rows)
        if rows and invalidate_mcp_registry_cache:
            self.revision += 1

    def write_system_rows(self, entity: type, rows: List[dict]) -> None:
        """Materialize and write system rows to the entity's system backend."""
        ws = self.workspace
        rows = [row for row in rows if row]
        if not rows:
            return
        for row in rows:
            ws._materializer.materialize(entity, row)
        ws.backends.get(self.system_backend_name(entity)).upsert([RowOp(entity.schema().entity_id, row["object_id"], row) for row in rows])

    def system_backend_name(self, entity: type) -> str:
        """Return the backend that owns system rows for an entity."""
        ws = self.workspace
        schema_id = entity.schema().entity_id
        if schema_id in ws.storage_plans and ws.storage_plans[schema_id].bindings:
            return ws.storage_plans[schema_id].bindings[0].backend
        names = ws.backends.names()
        if not names:
            raise RuntimeError("No backend is registered for system rows")
        return names[0]
