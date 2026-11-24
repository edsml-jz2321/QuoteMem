"""In-process backend implementation."""

from __future__ import annotations

__all__ = [
    "InMemBackend",
]

from bisect import bisect_right
from contextlib import AbstractContextManager

from heavenbase.utils import Any, Dict, Iterable, List, Optional, Sequence

from ...execution.frame.result import ResultFrame
from ...modules import ModuleResolver
from ...schema.model import EntitySchema
from ..base import Backend, backend_locked
from heavenbase.execution.contracts import QueryFragment, RowOp


class InMemBackend(Backend):
    """Zero-dependency backend used for tests, demos, and local workspaces."""

    identifier = "inmem"
    aliases = ("memory",)

    def _guard_registration_rows(self, entity_id: str, object_ids: Sequence[str]) -> AbstractContextManager[Any]:
        """Serialize the read and restore against writes to the in-process bucket."""
        return self._lock

    @classmethod
    def from_config(
        cls,
        name: str,
        data: dict[str, Any],
        *,
        resolver: ModuleResolver | None = None,
        config: Any | None = None,
    ) -> "InMemBackend":
        """Build the in-memory backend from a config mapping.

        Args:
            name: Workspace backend instance name.
            data: Backend config mapping. Supports optional ``ws_id``.
            resolver: Context-owned module authority.
            config: Config authority retained by the backend.

        Returns:
            InMemBackend: Configured in-memory backend instance.
        """
        cfg = dict(data or {})
        return cls(name, ws_id=cfg.pop("ws_id", None), resolver=resolver, config=config)

    def __init__(
        self,
        name: str = "main",
        ws_id: str | None = None,
        *,
        resolver: ModuleResolver | None = None,
        config: Any | None = None,
    ) -> None:
        """Create an in-memory backend.

        Args:
            name: Workspace backend instance name.
            ws_id: Optional workspace namespace. Rows are partitioned by this value.
            resolver: Context-owned module authority.
            config: Config authority retained by the backend.

        Returns:
            None: This function does not return a value.
        """
        super().__init__(name, ws_id=ws_id, resolver=resolver, config=config)
        self.data: Dict[str, Dict[str, Dict[Any, Dict[str, Any]]]] = {}
        self._schemas: Dict[str, EntitySchema] = {}
        self._vindex: Dict[str, Dict[str, Dict[Any, set[Any]]]] = {}
        self._vfields: Dict[str, tuple[str, ...]] = {}
        self._unique_fields: Dict[str, tuple[tuple[str, ...], ...]] = {}
        # Sorted object ids per (namespace, entity), reused by scan_page while the key set is unchanged.
        self._page_keys: Dict[tuple[Any, str], tuple[frozenset[Any], list[Any]]] = {}

    @backend_locked
    def ensure(self, schema: EntitySchema, bindings: Sequence[Any]) -> None:
        """Record a schema for later plan-aware index reconciliation.

        Args:
            schema: Entity schema to record.
            bindings: Legacy storage bindings, unused by the whole-row backend.

        Returns:
            None: The schema snapshot is retained by entity identifier.
        """
        self._schemas[schema.entity_id] = schema

    @backend_locked
    def ensure_indexes(self, schema: EntitySchema, storage_plan: Any) -> None:
        """Rebuild declared value indexes and validate uniqueness constraints.

        Args:
            schema: Entity schema owning the resolved declarations.
            storage_plan: Resolved storage, index, and constraint plan.

        Returns:
            None: Index state is rebuilt from stored rows.

        Raises:
            ValueError: If stored rows violate a uniqueness constraint.
        """
        # A single-field unique constraint is also a scalar lookup path (StoragePlan.has_index),
        # so the eq/in compiler emits native index fragments for it: index its values too.
        lookups = [*storage_plan.backend_indexes(self.name), *storage_plan.backend_constraints(self.name)]
        self._vfields[schema.entity_id] = tuple(dict.fromkeys(binding.fields[0] for binding in lookups if len(binding.fields) == 1))
        self._unique_fields[schema.entity_id] = tuple(binding.fields for binding in storage_plan.backend_constraints(self.name))
        self._rebuild_indexes(schema.entity_id)

    def _rebuild_indexes(self, entity_id: str) -> None:
        """Rebuild one entity's value indexes and validate unique tuples."""
        rows = self.data.get(self.ws_id, {}).get(entity_id, {})
        self._check_unique(entity_id, rows)
        index: Dict[str, Dict[Any, set[Any]]] = {field_name: {} for field_name in self._vfields.get(entity_id, ())}
        for object_id, row in rows.items():
            for field_name in index:
                value = row.get(field_name)
                if value is not None:
                    index[field_name].setdefault(value, set()).add(object_id)
        self._vindex[entity_id] = index

    def _check_unique(self, entity_id: str, rows: Dict[Any, Dict[str, Any]]) -> None:
        """Raise ``ValueError`` when two rows share a value tuple of one unique constraint.

        A tuple holding ``None`` never conflicts, as in SQL.
        """
        for fields in self._unique_fields.get(entity_id, ()):
            values: dict[tuple[Any, ...], Any] = {}
            for object_id, row in rows.items():
                key = tuple(row.get(field_name) for field_name in fields)
                if any(value is None for value in key):
                    continue
                if key in values and values[key] != object_id:
                    raise ValueError(
                        f"Unique constraint on {entity_id}({', '.join(fields)}) rejects value {key!r} for object_ids {values[key]!r} and {object_id!r}"
                    )
                values[key] = object_id

    @backend_locked
    def drop(self) -> None:
        """Remove all in-memory rows for the current workspace namespace.

        Args:
            None.

        Returns:
            None: This function does not return a value.
        """
        self.data.pop(self.ws_id, None)
        self._page_keys.clear()
        self._schemas.clear()
        self._vindex.clear()
        self._vfields.clear()
        self._unique_fields.clear()

    @backend_locked
    def upsert(self, rows: Sequence[RowOp]) -> None:
        """Apply row mutations to the in-process dictionary store.

        Args:
            rows: Backend row mutations. ``"upsert"`` replaces a row fragment and
                ``"delete"`` removes it if present.

        Returns:
            None: This function does not return a value.

        Notes:
            The call is failure-atomic: every replacement row is built and every
            unique constraint is checked before the store changes, so a failed
            call applies none of its mutations.
        """
        ns = self.data.setdefault(self.ws_id, {})
        if not any(self._unique_fields.get(row.entity_id) for row in rows):
            staged = [(row, None if row.is_delete() else row.to_row()) for row in rows]
            rebuild = set()
            for row, item in staged:
                bucket = ns.setdefault(row.entity_id, {})
                if item is None:
                    if bucket.pop(row.object_id, None) is not None:
                        self._ids_changed(row.entity_id)
                else:
                    if row.object_id not in bucket:
                        self._ids_changed(row.entity_id)
                    bucket[row.object_id] = item
                if self._vfields.get(row.entity_id):
                    rebuild.add(row.entity_id)
            for entity_id in rebuild:
                self._rebuild_indexes(entity_id)
            return
        prospective: Dict[str, Dict[Any, Dict[str, Any]]] = {}
        for row in rows:
            bucket = prospective.setdefault(row.entity_id, {object_id: dict(item) for object_id, item in ns.get(row.entity_id, {}).items()})
            if row.is_delete():
                bucket.pop(row.object_id, None)
                continue
            bucket[row.object_id] = row.to_row()
        previous = {entity_id: ns.get(entity_id) for entity_id in prospective}
        for entity_id, bucket in prospective.items():
            if bucket.keys() != (previous[entity_id] or {}).keys():
                self._ids_changed(entity_id)
        try:
            for entity_id, bucket in prospective.items():
                ns[entity_id] = bucket
                self._rebuild_indexes(entity_id)
        except ValueError:
            for entity_id, bucket in previous.items():
                if bucket is None:
                    ns.pop(entity_id, None)
                else:
                    ns[entity_id] = bucket
                self._rebuild_indexes(entity_id)
            raise

    @backend_locked
    def upsert_owned(self, rows: Sequence[RowOp], *, owner_field: str, owner: str) -> list[RowOp]:
        """Apply owner-checked mutations under the backend's mutation lock."""
        return self._upsert_owned_under_lock(rows, owner_field=owner_field, owner=owner)

    @backend_locked
    def compare_upsert(
        self,
        row: RowOp,
        *,
        revision_field: str,
        expected_revision: int | None,
    ) -> bool:
        """Conditionally replace one in-memory row under the Backend lock.

        Args:
            row (RowOp): Replacement upsert containing the next revision.
            revision_field (str): Row field holding the current integer revision.
            expected_revision (int | None): Required revision, or `None` when
                the row must be absent.

        Returns:
            bool: Whether the conditional replacement succeeded.

        Raises:
            ValueError: If `row` is a delete mutation.
        """
        if not row.is_upsert():
            raise ValueError("compare_upsert requires an upsert RowOp")
        bucket = self.data.setdefault(self.ws_id, {}).setdefault(row.entity_id, {})
        current = bucket.get(row.object_id)
        if expected_revision is None:
            if current is not None:
                return False
        elif current is None or current.get(revision_field) != expected_revision:
            return False
        replacement = row.to_row()
        if current is None:
            self._ids_changed(row.entity_id)
        bucket[row.object_id] = replacement
        try:
            self._rebuild_indexes(row.entity_id)
        except ValueError:
            if current is None:
                bucket.pop(row.object_id, None)
            else:
                bucket[row.object_id] = current
            self._rebuild_indexes(row.entity_id)
            raise
        return True

    @backend_locked
    def scan(self, entity_id: str) -> List[Dict[str, Any]]:
        """Return copies of all rows for one entity.

        Args:
            entity_id: Logical entity identifier.

        Returns:
            List[Dict[str, Any]]: Row fragments for the entity in backend iteration order.
        """
        return [self._copy_row(row) for row in self.data.get(self.ws_id, {}).get(entity_id, {}).values()]

    def _copy_row(self, row: Dict[str, Any], selection: tuple[str, ...] | None = None) -> Dict[str, Any]:
        """Copy the row or project the selected columns into a new fragment."""
        return dict(row) if selection is None else self._project_row(row, selection)

    def iter_scan(self, entity_id: str, *, batch_size: int, columns: Sequence[str] | None = None) -> Iterable[dict[str, Any]]:
        """Iterate detached in-memory row mappings without copying the bucket.

        Args:
            entity_id (str): Logical entity identifier.
            batch_size (int): Positive scan batch size.
            columns (Sequence[str] | None): Optional field names to return; ``None`` returns the complete fragment.

        Yields:
            dict[str, Any]: Next row fragment.

        Raises:
            ValueError: If the page size is not positive.

        Notes:
            The generator holds the backend lock across ``yield``; consume it
            without taking another lock between items, or page with ``scan_page``.
        """
        self._check_batch_size(batch_size)
        selection = self._row_columns(columns)
        with self._lock:
            for row in self.data.get(self.ws_id, {}).get(entity_id, {}).values():
                yield self._copy_row(row, selection)

    @backend_locked
    def scan_page(self, entity_id: str, *, after: Any, batch_size: int, columns: Sequence[str] | None = None) -> List[Dict[str, Any]]:
        """Return one ``object_id``-ordered slice of copied rows; the lock is released before the page is returned.

        Args:
            entity_id (str): Logical entity identifier.
            after (Any): ``object_id`` the page must follow, or ``None`` for the first page.
            batch_size (int): Positive maximum number of rows in the page.
            columns (Sequence[str] | None): Optional field names to return; ``None`` returns the complete fragment.

        Returns:
            List[Dict[str, Any]]: Detached row fragments in ascending ``object_id`` order.

        Raises:
            ValueError: If the page size is not positive.
        """
        self._check_batch_size(batch_size)
        selection = self._row_columns(columns)
        bucket = self.data.get(self.ws_id, {}).get(entity_id, {})
        # The first page of an enumeration proves the cached order against the whole id set;
        # a later page trusts it while the id count is unchanged.
        keys = self._sorted_keys(entity_id, bucket, verify=after is None)
        start = 0 if after is None else bisect_right(keys, after)
        page = [bucket.get(key) for key in keys[start : start + batch_size]]
        if any(row is None for row in page):
            # an id was removed by a direct ``data`` write since the order was cached
            keys = self._sorted_keys(entity_id, bucket, verify=True)
            start = 0 if after is None else bisect_right(keys, after)
            page = [bucket[key] for key in keys[start : start + batch_size]]
        return [self._copy_row(row, selection) for row in page]

    def _ids_changed(self, entity_id: str) -> None:
        """Forget the cached ``scan_page`` order of one entity after its id set changed.

        A subclass that adds or removes ids in ``data`` outside ``upsert``,
        ``compare_upsert``, and ``drop`` calls this so a page that continues an
        enumeration sees the change; the first page of an enumeration checks the
        whole id set anyway.
        """
        self._page_keys.pop((self.ws_id, entity_id), None)

    def _sorted_keys(self, entity_id: str, bucket: Dict[Any, Dict[str, Any]], *, verify: bool) -> list[Any]:
        """Return the bucket's ids in order, sorting again only when the id set changed.

        Writes through this backend's mutators forget the cached order
        (``_ids_changed``). Subclasses and callers may also write ``data``
        directly, so with ``verify`` the cached id set is compared with the
        bucket, one hashed pass; without it only the sizes are compared, so a
        page continuing an enumeration costs no pass over the ids and a full
        keyset scan stays linear in the row count.
        """
        slot = (self.ws_id, entity_id)
        cached = self._page_keys.get(slot)
        if cached is not None and len(cached[1]) == len(bucket) and (not verify or bucket.keys() == cached[0]):
            return cached[1]
        keys = sorted(bucket)
        self._page_keys[slot] = (frozenset(keys), keys)
        return keys

    @backend_locked
    def get_many(self, entity_id: str, object_ids: Iterable[Any], *, columns: Sequence[str] | None = None) -> List[Optional[Dict[str, Any]]]:
        """Return rows by object id without scanning the whole entity bucket.

        Args:
            entity_id: Logical entity identifier.
            object_ids: Object ids to read.
            columns: Optional field names to return; ``None`` returns the complete fragment.

        Returns:
            List[Optional[Dict[str, Any]]]: Row fragments aligned to ``object_ids``.
                Missing rows are ``None``.
        """
        bucket = self.data.get(self.ws_id, {}).get(entity_id, {})
        out: List[Optional[Dict[str, Any]]] = []
        for object_id in object_ids:
            row = bucket.get(object_id)
            out.append(dict(row) if row is not None else None)
        return self._project_rows(out, columns)

    @backend_locked
    def count(self, entity_id: str) -> int:
        """Return the number of rows for one entity.

        Args:
            entity_id: Logical entity identifier.

        Returns:
            int: Count of rows stored for the entity in the current workspace namespace.
        """
        return len(self.data.get(self.ws_id, {}).get(entity_id, {}))

    @backend_locked
    def execute(self, fragment: QueryFragment) -> ResultFrame:
        """Resolve indexed equality fragments and scan for every other query.

        Args:
            fragment: Backend-ready query fragment compiled by a handler.

        Returns:
            ResultFrame: Rows matching the fragment.
        """
        if self._is_index_fragment(fragment):
            entity_id = fragment.notes.get("entity_id")
            if entity_id:
                bucket = self.data.get(self.ws_id, {}).get(entity_id, {})
                rows = [dict(bucket[object_id]) for object_id in sorted(self._resolve_index(entity_id, fragment), key=str) if object_id in bucket]
                from ..scanning import execute_scan

                return execute_scan(self, fragment, rows=rows)
        return super().execute(fragment)

    def _is_index_fragment(self, fragment: QueryFragment) -> bool:
        """Return whether every leaf in a fragment can use an in-memory index."""
        if fragment.notes.get("kind") == "index_filter":
            return True
        if fragment.op in ("and", "or"):
            return bool(fragment.payload) and all(self._is_index_fragment(child) for child in fragment.payload)
        return False

    def _resolve_index(self, entity_id: str, fragment: QueryFragment) -> set[Any]:
        """Return candidate object ids for one index-compatible fragment."""
        if fragment.notes.get("kind") == "index_filter":
            if fragment.field == "object_id":
                values = fragment.payload["values"] if fragment.notes.get("op") == "in" else [fragment.payload["value"]]
                bucket = self.data.get(self.ws_id, {}).get(entity_id, {})
                return {value for value in values if value in bucket}
            index = self._vindex.get(entity_id, {}).get(fragment.field, {})
            if fragment.notes.get("op") == "in":
                out: set[Any] = set()
                for value in fragment.payload["values"]:
                    out.update(index.get(value, set()))
                return out
            return set(index.get(fragment.payload["value"], set()))
        children = [self._resolve_index(entity_id, child) for child in fragment.payload]
        if fragment.op == "and":
            out = set(children[0])
            for child in children[1:]:
                out.intersection_update(child)
            return out
        return set().union(*children)
