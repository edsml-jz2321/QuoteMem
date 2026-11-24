"""Workspace entity write and delete planning."""

from __future__ import annotations

__all__ = [
    "DeletePlan",
    "DeleteRoute",
    "UpsertPlan",
]

from collections import Counter
from dataclasses import replace

from heavenbase.utils import Any, Dict, Iterable, List, Literal, Optional, dataclass

from ...execution.contracts import RowOp
from ..catalog.entity import Catalog
from ..catalog.rows import CATALOG_LIFECYCLE_FIELDS
from .inputs import DeleteTarget, UpsertRow
from .errors import SchemaVersionMismatchError


@dataclass
class UpsertPlan:
    """Describe materialized entity and Catalog writes for one upsert batch.

    Args:
        object_ids (List[str]): Object identifiers in caller input order.
        entity_ops (Dict[str, List[RowOp]]): Entity operations grouped by backend.
        catalog_ops (Dict[str, List[RowOp]]): Catalog operations grouped by backend.
        entity_classes (List[type]): Registered entity classes represented by
            the planned rows.
    """

    object_ids: List[str]
    entity_ops: Dict[str, List[RowOp]]
    catalog_ops: Dict[str, List[RowOp]]
    entity_classes: List[type]


@dataclass(frozen=True)
class DeleteRoute:
    """Identify one backend and entity receiving a delete batch.

    Args:
        backend_name (str): Workspace backend name.
        entity_id (str): Entity identifier on that backend.
    """

    backend_name: str
    entity_id: str


@dataclass
class DeletePlan:
    """Describe caller-order delete results and routed backend work.

    Args:
        deleted (List[bool]): Per-input flags reporting prior row existence.
        object_ids_by_route (Dict[DeleteRoute, List[str]]): Object identifiers
            grouped by backend and entity route.
        entity_classes (List[type]): Entity classes whose rows existed.
    """

    deleted: List[bool]
    object_ids_by_route: Dict[DeleteRoute, List[str]]
    entity_classes: List[type]


class _EntityWriter:
    """Plan and commit entity plus Catalog operations across backend routes."""

    def __init__(self, workspace: Any):
        self.workspace = workspace
        # (Catalog storage plan, the backends it routes to, the Catalog fields their strategy adapters maintain)

    def _require_new_row(self, entity_cls: type, object_id: str) -> None:
        """Refuse a create-only write of a row that exists or that may exist.

        Args:
            entity_cls: Entity class of the row.
            object_id: Normalized object id of the row.

        Raises:
            ValueError: If a route returns the row, or if no route returned it
                but none shows it is absent, as with an entity stored only on
                Pinecone, whose fetch can miss a stored record.
        """
        lookup = self.workspace._row_lookup
        entity_id = entity_cls.schema().entity_id
        if lookup.id_match(object_id, entity=entity_cls) is not None:
            raise ValueError(f"{entity_id} object_id {object_id!r} already exists")
        error = lookup.absent_row_error(entity_id, object_id)
        if error is not None:
            raise ValueError(
                f"{error}; overwrite=False cannot confirm that {entity_id} object_id {object_id!r} is new, "
                "so pass overwrite=True to write it, or route a field of the entity to a backend other than Pinecone"
            ) from error

    def prepare_upsert_rows(
        self,
        rows: Iterable[UpsertRow],
        *,
        overwrite: bool,
        catalog_mode: Literal["merge", "fresh"] = "merge",
    ) -> UpsertPlan:
        """Materialize logical rows and plan entity plus Catalog operations.

        The Catalog work follows each entity's registered Catalog mode:
        ``"full"`` projects a row; ``"identity"`` publishes an identity-only row;
        ``"none"`` publishes nothing. With
        ``catalog_mode="merge"`` the stored Catalog rows of every ``"full"`` and
        ``"identity"`` row in the batch are read by id, one lookup per Catalog
        route, so their lifecycle fields carry over; ``"fresh"`` skips that read.
        An ``"identity"`` row that the lookup finds stored keeps strategy
        maintenance, so the adapter removes whatever it derived from values that
        a direct Catalog write, or a writer still registered as ``"full"``, put
        there, including a write that lands between this lookup and the commit.
        An ``"identity"`` row with no stored Catalog row skips maintenance, and so
        does every ``"identity"`` row under ``"fresh"``, which asserts the rows
        are new.

        ``overwrite=False`` refuses a row that already exists, and a row whose
        absence no route can show; see ``_require_new_row``.
        """
        if catalog_mode not in ("merge", "fresh"):
            raise ValueError("catalog_mode must be 'merge' or 'fresh'")
        ws = self.workspace
        resolved_rows = [(ws._schema.require(item.entity_cls), item) for item in rows]
        object_ids: List[str] = []
        entity_classes: List[type] = []
        seen: set[tuple[str, str]] = set()
        entity_ops: Dict[str, List[RowOp]] = {}
        catalog_ops: Dict[str, List[RowOp]] = {}
        catalog_hash_length: int | None = None
        pending: List[tuple[type, Any, str, Dict[str, Any], str, str]] = []
        row_store = ws._row_store
        row_codec = ws._row_codec
        for entity_cls, item in resolved_rows:
            data = item.data
            entity_classes.append(entity_cls)
            schema = entity_cls.schema()
            ws._materializer.materialize(entity_cls, data)
            object_id = data[schema.pk_name]
            key = (schema.entity_id, object_id)
            if key in seen:
                raise ValueError(f"{schema.entity_id} batch contains duplicate object_id {object_id!r}")
            if not overwrite:
                self._require_new_row(entity_cls, object_id)
            seen.add(key)
            object_ids.append(object_id)
            for backend_name in row_store.routed_backend_names(schema.entity_id):
                values = row_codec.encode_for_backend(schema, backend_name, data)
                entity_ops.setdefault(backend_name, []).append(RowOp(schema.entity_id, object_id, values))
            mode = ws._catalog.mode(schema.entity_id)
            if mode is None or mode == "none":
                continue
            if catalog_hash_length is None:
                catalog_hash_length = ws._catalog.hash_length()
            catalog_id = ws._catalog.row_id(schema, object_id, hash_length=catalog_hash_length)
            pending.append((entity_cls, schema, object_id, data, mode, catalog_id))
        if pending:
            catalog_entity_id = Catalog.schema().entity_id
            existing: Dict[str, Dict[str, Any]] = {}
            stored: set[str] = set()
            if catalog_mode == "merge":
                existing, stored = self._existing_catalog_rows([catalog_id for *_, catalog_id in pending])
            catalog_routes = row_store.routed_backend_names(catalog_entity_id)
            for entity_cls, schema, object_id, data, mode, catalog_id in pending:
                catalog_row = ws._catalog.row(
                    entity_cls, schema, object_id, data, existing=existing.get(catalog_id), hash_length=catalog_hash_length, mode=mode
                )
                maintained = mode == "full" or catalog_id in stored
                for backend_name in catalog_routes:
                    catalog_ops.setdefault(backend_name, []).append(RowOp(catalog_entity_id, catalog_row["object_id"], catalog_row, strategies=maintained))
        return UpsertPlan(object_ids, entity_ops, catalog_ops, entity_classes)

    def _existing_catalog_rows(self, catalog_ids: List[str]) -> tuple[Dict[str, Dict[str, Any]], set[str]]:
        """Read the stored lifecycle fields of many Catalog rows, in one projected lookup per Catalog route.

        Only ``CATALOG_LIFECYCLE_FIELDS`` are fetched and decoded, since a rewrite
        keeps nothing else. A stored row that lacks one of them counts as absent
        from the lifecycle rows, so the write starts that row's lifecycle afresh
        instead of failing; it still counts as stored.

        Returns:
            tuple[Dict[str, Dict[str, Any]], set[str]]: Stored lifecycle rows by
                Catalog id, and the ids that have a stored row on any route.
        """
        ws = self.workspace
        schema = Catalog.schema()
        fragments = ws._row_store.rows_for_ids(schema.entity_id, catalog_ids, columns=CATALOG_LIFECYCLE_FIELDS)
        existing: Dict[str, Dict[str, Any]] = {}
        stored = {catalog_id for catalog_id in catalog_ids if fragments.get(catalog_id)}
        for catalog_id in stored:
            try:
                row = ws._row_store.merge_id_rows(schema, catalog_id, fragments[catalog_id], unique=True, columns=CATALOG_LIFECYCLE_FIELDS)
            except SchemaVersionMismatchError:
                continue
            if row is not None:
                existing[catalog_id] = row
        return existing, stored

    def write_upsert(self, plan: UpsertPlan, *, atomic: bool = False) -> None:
        """Commit entity operations before publishing matching Catalog rows.

        Args:
            plan (UpsertPlan): Prepared entity and Catalog operations.
            atomic (bool): Commit the entity and Catalog operations through one
                write call to a backend that claims the ``atomic-upsert`` tag,
                so they all land or none does. Every routed operation must land
                on that one backend.

        Raises:
            ValueError: If a destination rejects the plan during validation,
                or the requested ``atomic`` write cannot run on one backend
                that claims ``atomic-upsert``. Nothing is written then.
        """
        if not atomic:
            self._validate_ops(plan.entity_ops)
            self._validate_ops(plan.catalog_ops)
            self.write_ops(plan.entity_ops)
            self.write_ops(plan.catalog_ops)
            return
        backend_name = self._atomic_backend(set(plan.entity_ops) | set(plan.catalog_ops))
        if backend_name is None:
            return
        ops = self._phased(backend_name, plan.entity_ops.get(backend_name, ())) + self._phased(backend_name, plan.catalog_ops.get(backend_name, ()))
        backend = self.workspace.backends.get(backend_name)
        backend.validate_row_ops(ops)
        self._invalidate_near({backend_name: ops})
        backend.upsert(ops)

    def require_atomic_upsert(self, entity_classes: Iterable[type]) -> None:
        """Refuse an atomic upsert of these Entity classes before any of its work runs.

        The routes are the ones ``prepare_upsert_rows`` plans: each backend an
        entity routes to, plus the Catalog routes of an entity whose Catalog
        mode publishes rows. Checking them first means a refused call also
        establishes no Entity class from a MetaSchema row.

        Args:
            entity_classes (Iterable[type]): Registered Entity classes of the rows.

        Raises:
            ValueError: If the routes span more than one backend, or their
                backend does not claim ``atomic-upsert``.
        """
        ws = self.workspace
        catalog_entity_id = Catalog.schema().entity_id
        backend_names: set[str] = set()
        for entity_cls in entity_classes:
            entity_id = entity_cls.schema().entity_id
            backend_names.update(ws._row_store.routed_backend_names(entity_id))
            if ws._catalog.mode(entity_id) not in (None, "none"):
                backend_names.update(ws._row_store.routed_backend_names(catalog_entity_id))
        self._atomic_backend(backend_names)

    def _atomic_backend(self, backend_names: Iterable[str]) -> str | None:
        """Return the one backend an atomic upsert writes through, or ``None`` when it writes nothing.

        Raises:
            ValueError: If ``backend_names`` holds more than one backend, or the
                backend does not claim ``atomic-upsert``.
        """
        backends = sorted(set(backend_names))
        if len(backends) > 1:
            raise ValueError(f"atomic upsert requires one backend for entity and Catalog rows; routed backends: {backends}")
        if not backends:
            return None
        backend = self.workspace.backends.get(backends[0])
        claim = backend.tag("atomic-upsert")
        if claim is None or claim.value is not True:
            raise ValueError(
                f"atomic upsert requires a backend whose write call applies all of its rows or none; backend {backends[0]!r} "
                f"of type {backend.type!r} does not claim the 'atomic-upsert' tag"
            )
        return backends[0]

    def write_ops(self, ops_by_backend: Dict[str, List[RowOp]]) -> None:
        """Commit uniqueness-owner writes before replicas for every entity."""
        self._invalidate_near(ops_by_backend)
        ops_by_backend = self._combine_write_fragments(ops_by_backend)
        owner_ops: Dict[str, List[RowOp]] = {}
        replica_ops: Dict[str, List[RowOp]] = {}
        for backend_name, ops in ops_by_backend.items():
            for op in ops:
                owner = self._unique_owner(op.entity_id)
                if owner is not None:
                    owner = self.workspace.backends.get(owner)._row_write_routes(op.entity_id)[0]
                target = owner_ops if owner == backend_name else replica_ops
                target.setdefault(backend_name, []).append(op)
        for phase in (owner_ops, replica_ops):
            for backend_name in sorted(phase):
                self.workspace.backends.get(backend_name)._write_workspace_rows(phase[backend_name])

    def _combine_write_fragments(self, ops_by_backend: Dict[str, List[RowOp]]) -> Dict[str, List[RowOp]]:
        """Assemble one replacement when several routes own columns of the same row."""
        routed: Dict[str, List[RowOp]] = {}
        shared = {}
        for name in sorted(ops_by_backend):
            backend = self.workspace.backends.get(name)
            occurrences: Counter[tuple[str, Any, str]] = Counter()
            for op in ops_by_backend[name]:
                routes = backend._row_write_routes(op.entity_id)
                destination = routed.setdefault(routes[0], [])
                if len(routes) == 1:
                    destination.append(op)
                    continue
                identity = (op.entity_id, op.object_id, op.op)
                key = (routes[0], identity, occurrences[identity])
                occurrences[identity] += 1
                if key not in shared:
                    shared[key] = (len(destination), routes, [])
                    destination.append(op)
                shared[key][2].append(op)
        for (name, _, _), (index, routes, fragments) in shared.items():
            first = fragments[0]
            if first.is_upsert() and len(fragments) != len(routes):
                raise ValueError(f"Writing {first.entity_id!r} requires row fragments from all shared SQL routes: {routes}")
            values = {field: value for fragment in fragments for field, value in fragment.values.items()}
            routed[name][index] = replace(first, values=values, strategies=any(fragment.strategies for fragment in fragments))
        return routed

    def _phased(self, backend_name: str, ops: Iterable[RowOp]) -> List[RowOp]:
        """Return one backend's operations in uniqueness-owner-then-replica order."""
        ordered = list(ops)
        owned = [op for op in ordered if self._unique_owner(op.entity_id) == backend_name]
        return owned + [op for op in ordered if self._unique_owner(op.entity_id) != backend_name]

    def _invalidate_near(self, ops_by_backend: Dict[str, List[RowOp]]) -> None:
        """Drop cached near-search state for every entity about to change."""
        for ops in ops_by_backend.values():
            for entity_id in {op.entity_id for op in ops}:
                self.workspace._router._near.cache.invalidate(entity_id)

    def _validate_ops(self, ops_by_backend: Dict[str, List[RowOp]]) -> None:
        """Check every destination's local constraints before the first write."""
        for backend_name in sorted(ops_by_backend):
            self.workspace.backends.get(backend_name).validate_row_ops(ops_by_backend[backend_name])

    def write_dependency_ordered_ops(self, ops_by_backend: Dict[str, List[RowOp]]) -> None:
        """Commit preflighted mutation closures without reordering backend operations.

        GraphEdge detach planning places every surviving-source rewrite before the
        corresponding target delete. Those internal rewrites retain existing unique
        values, so repartitioning them through the ordinary uniqueness-owner phases
        adds no validation and can move a target delete ahead of its required rewrite.

        Args:
            ops_by_backend (Dict[str, List[RowOp]]): Dependency-ordered operations
                grouped by backend.
        """
        self._validate_ops(ops_by_backend)
        self._invalidate_near(ops_by_backend)
        routed = self._combine_write_fragments(ops_by_backend)
        for backend_name in sorted(routed):
            self.workspace.backends.get(backend_name)._write_workspace_rows(routed[backend_name])

    def _unique_owner(self, entity_id: str) -> Optional[str]:
        """Return the single resolved uniqueness owner for an entity, if any."""
        plans = self.workspace.storage_plans
        owners = {constraint.backend for constraint in (plans[entity_id].constraints if entity_id in plans else ())}
        if len(owners) > 1:
            raise RuntimeError(f"Entity {entity_id!r} has multiple uniqueness owners: {sorted(owners)}")
        return next(iter(owners), None)

    def delete_rows(self, targets: Iterable[Optional[DeleteTarget]]) -> DeletePlan:
        """Plan and commit normalized delete targets."""
        plan = self.prepare_delete_rows(targets)
        self.write_delete(plan)
        return plan

    def prepare_delete_rows(self, targets: Iterable[Optional[DeleteTarget]]) -> DeletePlan:
        """Plan backend row deletes, existence flags, and Catalog cleanup.

        Every route of a target is deleted whether or not its row was found. A
        target that no route returned is reported as not deleted. Its Catalog
        row is still deleted when no route shows the row is absent, as after a
        Pinecone fetch miss, so a Catalog row never outlives the record.
        """
        ws = self.workspace
        target_list = list(targets)
        object_ids_by_route: Dict[DeleteRoute, List[str]] = {}
        object_ids_by_entity: Dict[type, List[str]] = {}
        for target in target_list:
            if target is None:
                continue
            schema = target.entity_cls.schema()
            object_ids_by_entity.setdefault(target.entity_cls, []).append(target.object_id)
            for backend_name in ws._row_store.routed_backend_names(schema.entity_id):
                route = DeleteRoute(backend_name, schema.entity_id)
                object_ids_by_route.setdefault(route, []).append(target.object_id)

        existed: Dict[tuple[type, str], bool] = {}
        for entity_cls, object_ids in object_ids_by_entity.items():
            for object_id, flag in zip(object_ids, ws._row_lookup.id_exists_many(object_ids, entity=entity_cls)):
                existed[(entity_cls, object_id)] = flag

        deleted = [existed.get((target.entity_cls, target.object_id), False) if target is not None else False for target in target_list]
        entity_classes: List[type] = []
        catalog_ids: List[str] = []
        for target, was_present in zip(target_list, deleted):
            if target is None:
                continue
            schema = target.entity_cls.schema()
            if was_present:
                entity_classes.append(target.entity_cls)
            elif ws._row_lookup.absent_row_error(schema.entity_id, target.object_id) is None:
                continue
            mode = ws._catalog.mode(schema.entity_id)
            if mode is None or mode == "none":
                continue
            catalog_ids.append(ws._catalog.row_id(schema, target.object_id))
        if catalog_ids:
            # A Catalog delete keeps full strategy maintenance whatever the mode: deleting index rows that do not exist is
            # idempotent, and a keyword row that outlived its "full" form can then never make a later read fail.
            catalog_schema = Catalog.schema()
            for backend_name in ws._row_store.routed_backend_names(catalog_schema.entity_id):
                route = DeleteRoute(backend_name, catalog_schema.entity_id)
                object_ids_by_route.setdefault(route, []).extend(catalog_ids)
        return DeletePlan(deleted, object_ids_by_route, entity_classes)

    def write_delete(self, plan: DeletePlan) -> None:
        """Commit deduplicated object identifiers for every delete route."""
        self.write_ops(self.delete_ops(plan))

    def delete_ops(self, plan: DeletePlan) -> Dict[str, List[RowOp]]:
        """Return backend-ready delete operations from a prepared delete plan."""
        ops: Dict[str, List[RowOp]] = {}
        for route, object_ids in plan.object_ids_by_route.items():
            ops.setdefault(route.backend_name, []).extend(RowOp(route.entity_id, object_id, op="delete") for object_id in dict.fromkeys(object_ids))
        return ops
