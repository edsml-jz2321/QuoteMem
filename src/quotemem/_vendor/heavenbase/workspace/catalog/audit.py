"""Private Catalog audit and repair service for HeavenBase workspaces."""

from __future__ import annotations

import json
from contextlib import contextmanager
from itertools import islice
from tempfile import SpooledTemporaryFile

from heavenbase.utils import Any, Dict, Generator, Iterable, Iterator, List

from ...backends.base import accepts_keyword
from ...execution.contracts import RowOp
from ...placement import StoragePlan
from ...schema.model import EntitySchema
from ...strategies import SparseGramIndex
from .entity import Catalog

_RECONCILE_PAGE = 256


class _CatalogAuditService:
    """Own Catalog source reconstruction, integrity audit, and repair row planning."""

    def __init__(self, workspace: Any):
        self.workspace = workspace

    def participating_entity_ids(self) -> List[str]:
        """Return user entity ids whose rows publish a Catalog row."""
        ws = self.workspace
        return [entity_id for entity_id in sorted(ws.entities) if ws._catalog.mode(entity_id) not in (None, "none")]

    def source_rows(self) -> Iterator[tuple[type, Any, str, Dict[str, Any], str]]:
        """Yield materialized user rows used as authoritative Catalog sources.

        Entities registered with ``catalog="none"`` publish nothing, so their
        rows are not sources; this workspace's Catalog rows that name one
        are dangling.

        Yields:
            tuple: Entity class, schema, ``object_id``, materialized row, and
            the entity's Catalog mode.
        """
        ws = self.workspace
        for entity_id in self.participating_entity_ids():
            entity_cls = ws.entities[entity_id]
            schema = entity_cls.schema()
            mode = ws._catalog.mode(entity_id)
            for row in ws._row_store.universe(entity_id).rows():
                data = dict(row)
                ws._materializer.materialize(entity_cls, data)
                object_id = data.get(schema.pk_name)
                if object_id is not None:
                    yield entity_cls, schema, object_id, data, mode

    def _outside_workspace(self, row: Dict[str, Any] | None) -> bool:
        """Check whether an existing row is outside this workspace's ownership."""
        return row is not None and row.get("ws") != self.workspace.id

    def expected_rows(self, actual: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Return expected Catalog rows for keys this workspace can repair."""
        rows: Dict[str, Dict[str, Any]] = {}
        for entity_cls, schema, object_id, data, mode in self.source_rows():
            catalog_id = self.workspace._catalog.row_id(schema, object_id)
            existing = actual.get(catalog_id)
            if self._outside_workspace(existing):
                continue
            catalog_row = self.workspace._catalog.audit_row(entity_cls, schema, object_id, data, existing, mode=mode)
            rows[catalog_row["object_id"]] = catalog_row
        return rows

    def repair_rows(self, object_ids: Iterable[str]) -> List[Dict[str, Any]]:
        """Build fresh Catalog rows for the requested Catalog object_ids."""
        wanted = set(object_ids)
        if not wanted:
            return []
        rows: Dict[str, Dict[str, Any]] = {}
        actual = self.rows_by_id()
        for entity_cls, schema, object_id, data, mode in self.source_rows():
            catalog_id = self.workspace._catalog.row_id(schema, object_id)
            existing = actual.get(catalog_id)
            if catalog_id in wanted and not self._outside_workspace(existing):
                rows[catalog_id] = self.workspace._catalog.row(entity_cls, schema, object_id, data, existing=existing, mode=mode)
        return [rows[object_id] for object_id in object_ids if object_id in rows]

    def rows_by_id(self) -> Dict[str, Dict[str, Any]]:
        """Return current Catalog rows keyed by object_id."""
        rows: Dict[str, Dict[str, Any]] = {}
        for row in self.workspace._row_store.universe(Catalog.schema().entity_id).rows():
            object_id = row.get("object_id")
            if object_id is not None:
                rows[object_id] = row
        return rows

    def write_owned(self, ops: List[RowOp]) -> set[str]:
        """Apply Catalog mutations under each route's physical ownership guard."""
        if not ops:
            return set()
        ws = self.workspace
        entity_id = Catalog.schema().entity_id
        routes = ws._row_store.routed_backend_names(entity_id)
        owner_routes = {binding.backend for binding in ws.storage_plans[entity_id].bindings if binding.field_name == "ws"}
        if any(name not in owner_routes for name in routes):
            raise NotImplementedError("Catalog maintenance requires every Catalog route to store the ws field")
        ws._router._near.cache.invalidate(entity_id)
        applied = set()
        for name in routes:
            applied.update(op.object_id for op in ws.backends.get(name).upsert_owned(ops, owner_field="ws", owner=ws.id))
        return applied

    def stale_keyword_ids(self, catalog_ids: Iterable[str]) -> List[str]:
        """Return the Catalog ids whose sparse-GRAM keyword rows differ from what their stored rows index.

        Each Catalog route whose backend keeps a keyword index of a Catalog
        field compares the keyword and posting rows it stores for those ids
        with the rows its stored Catalog row indexes; an id with no stored row
        must have none. Routes that keep no keyword index are not asked.
        """
        ids = list(dict.fromkeys(catalog_ids))
        if not ids:
            return []
        ws = self.workspace
        entity_id = Catalog.schema().entity_id
        stale: set[str] = set()
        for name in ws._row_store.routed_backend_names(entity_id):
            adapters = getattr(ws.backends.get(name), "_strategy_adapters", {})
            check = getattr(adapters.get(SparseGramIndex.identifier), "stale_keyword_objects", None)
            if callable(check):
                stale.update(check(entity_id, ids))
        return [object_id for object_id in ids if object_id in stale]

    def audit(self) -> Dict[str, Any]:
        """Compare expected Catalog rows and their keyword rows with the rows owned by this workspace.

        Catalog keys occupied by rows owned elsewhere or with no named
        owner are outside the scope of the comparison and repair.
        The sparse-GRAM keyword and posting rows of every Catalog key in scope
        (a stored row this workspace owns, or one its sources expect) are
        compared with the rows its stored Catalog row indexes: a key whose
        keyword rows differ, or that keeps keyword rows without a stored row,
        is listed under ``stale_keywords``.

        Returns:
            Dict[str, Any]: ``ok``, ``missing``, ``dangling``, ``mismatched``,
            ``stale_keywords``, and ``checked`` counts.
        """
        actual = self.rows_by_id()
        expected = self.expected_rows(actual)
        actual = {object_id: row for object_id, row in actual.items() if not self._outside_workspace(row)}
        missing = [{"expected": row} for object_id, row in sorted(expected.items()) if object_id not in actual]
        dangling = [{"catalog": row} for object_id, row in sorted(actual.items()) if object_id not in expected]
        mismatched = []
        for object_id, expected_row in sorted(expected.items()):
            catalog_row = actual.get(object_id)
            if catalog_row is None:
                continue
            fields = [field for field in sorted(expected_row) if catalog_row.get(field) != expected_row.get(field)]
            if fields:
                mismatched.append({"catalog": catalog_row, "expected": expected_row, "fields": fields})
        stale_keywords = [{"object_id": object_id} for object_id in self.stale_keyword_ids(sorted(set(actual) | set(expected)))]
        ok = not missing and not dangling and not mismatched and not stale_keywords
        return {
            "ok": ok,
            "missing": missing,
            "dangling": dangling,
            "mismatched": mismatched,
            "stale_keywords": stale_keywords,
            "checked": {
                "objects": len(expected),
                "catalog_rows": len(actual),
                "entities": len(self.participating_entity_ids()),
            },
        }

    def require_publishable(self, schema: EntitySchema, *, mode: str, storage_plan: StoragePlan | None = None) -> None:
        """Refuse a change into a publishing Catalog mode that a stored row cannot take.

        An entity registered with ``catalog="none"`` may hold ids longer than a
        Catalog row's ``target_id``. Such a row can never get a Catalog row, so
        the change is refused before anything is switched or rewritten. The
        stored IDs are read in bounded pages. The caller holds the workspace
        write lock, so one scan-only route can stream this validation in a
        single pass.

        Args:
            schema: Schema whose stored rows are checked.
            mode (str): Target Catalog mode, ``"full"`` or ``"identity"``.
            storage_plan: Prepared routes of a registration that is still detached
                from the workspace. A live registration uses the ordinary row store.

        Raises:
            ValueError: If a stored ``object_id`` is longer than ``Catalog.target_id`` holds.
        """
        limit = self.workspace._catalog.target_id_limit()
        rows = self._preflight_rows(schema, storage_plan)
        try:
            for row in rows:
                object_id = row.get(schema.pk_name)
                if object_id is not None and len(str(object_id)) > limit:
                    raise ValueError(
                        f"Entity {schema.entity_id!r} cannot change to catalog={mode!r}: its stored object_id starting {str(object_id)[:40]!r} has "
                        f"{len(str(object_id))} characters, but a Catalog row's target_id holds at most {limit}; keep catalog='none' "
                        "or delete the rows with longer ids first"
                    )
        finally:
            rows.close()

    def _preflight_rows(self, schema: EntitySchema, storage_plan: StoragePlan | None) -> Iterator[dict[str, Any]]:
        ws = self.workspace
        columns = [schema.pk_name]
        if storage_plan is None:
            yield from ws._row_store.iter_rows(schema.entity_id, batch_size=_RECONCILE_PAGE, columns=columns, held=True)
            return
        for name in sorted({binding.backend for binding in storage_plan.bindings}):
            backend = ws.backends.get(name)
            projection = {"columns": columns} if accepts_keyword(backend, "iter_scan", "columns") else {}
            yield from backend.iter_scan(schema.entity_id, batch_size=_RECONCILE_PAGE, **projection)

    def _target_pages(
        self, schema: EntitySchema, hash_length: int, stored_source: tuple[EntitySchema, StoragePlan] | None = None
    ) -> Generator[list[tuple[dict, str]], None, None]:
        ws = self.workspace
        if stored_source is not None:
            stored_schema, plan = stored_source
            seen = set() if len({binding.backend for binding in plan.bindings}) > 1 else None
            rows = self._preflight_rows(stored_schema, plan)
            try:
                while page := list(islice(rows, _RECONCILE_PAGE)):
                    targets = []
                    for row in page:
                        object_id = row[stored_schema.pk_name]
                        if seen is not None:
                            if object_id in seen:
                                continue
                            seen.add(object_id)
                        targets.append(({schema.pk_name: object_id}, ws._catalog.row_id(stored_schema, object_id, hash_length=hash_length)))
                    if targets:
                        yield targets
            finally:
                rows.close()
            return
        pages = ws._row_store.iter_row_pages(schema.entity_id, batch_size=_RECONCILE_PAGE, held=True)
        try:
            for page in pages:
                yield [(row, ws._catalog.row_id(schema, row[schema.pk_name], hash_length=hash_length)) for row in page]
        finally:
            pages.close()

    def _existing_targets(self, targets: list[tuple[dict, str]]) -> Dict[str, Dict[str, Any]]:
        return {
            match.object_id: match.row
            for match in self.workspace._row_lookup.entity_matches_many(Catalog, [catalog_id for _, catalog_id in targets])
            if match is not None
        }

    @contextmanager
    def prepare_catalog_rewrite(
        self, entity_cls: type, *, mode: str, stored_source: tuple[EntitySchema, StoragePlan] | None
    ) -> Iterator[Generator[list[tuple[dict, str]], None, None]]:
        """Capture source identities before storage preparation or a GRAM rebuild changes them."""
        ws = self.workspace
        schema = entity_cls.schema()
        pages = self._target_pages(schema, ws._catalog.hash_length(), stored_source)
        # Identity projections only use the primary key. Keep large rewrites bounded without
        # holding the Catalog rows or their lifecycle fields across the registration transaction.
        with SpooledTemporaryFile(max_size=1024 * 1024, mode="w+t", encoding="utf-8") as prepared:
            try:
                for targets in pages:
                    if stored_source is None and mode == "identity":
                        existing = self._existing_targets(targets)
                        targets = [(row, catalog_id) for row, catalog_id in targets if not self._outside_workspace(existing.get(catalog_id))]
                        for row, _ in targets:
                            ws._materializer.materialize(entity_cls, row)
                    prepared.write(json.dumps([({schema.pk_name: row[schema.pk_name]}, catalog_id) for row, catalog_id in targets]) + "\n")
            except Exception as exc:
                raise self._rewrite_error(schema.entity_id, mode, written=0, deleted=0, stored_full=stored_source is not None) from exc
            finally:
                pages.close()
            prepared.seek(0)
            yield (json.loads(page) for page in prepared)

    def _rewrite_error(self, entity_id: str, mode: str, *, written: int, deleted: int, stored_full: bool = False) -> RuntimeError:
        current = self.workspace.entity_options.get(entity_id)
        if stored_full and current is None:
            kept_mode = "full"
            recovery = "register the previous definition before running audit() or repair()"
        else:
            kept_mode = self.workspace._catalog.mode(entity_id)
            recovery = "audit() lists the rows that do not match it, and repair() rewrites them"
        return RuntimeError(
            f"Rewriting the stored Catalog rows of entity {entity_id!r} into catalog={mode!r} form failed after "
            f"{written} written and {deleted} deleted; the entity keeps "
            f"catalog={kept_mode!r}; {recovery}"
        )

    def reconcile_entity(
        self, entity_cls: type, *, mode: str, prepared_rows: Generator[list[tuple[dict, str]], None, None] | None = None, stored_full: bool = False
    ) -> Dict[str, int]:
        """Rewrite or delete one entity's stored Catalog rows into the form of ``mode``.

        Called when a re-registration changes the entity's Catalog mode,
        including a registration whose predecessor was only the stored one
        (re-established first by the schema service); the registration in
        effect may still be the previous one, so the target mode is explicit.
        Rows are read in bounded pages and written through the ordinary
        Catalog routes with full strategy maintenance, so keyword rows that a
        ``"full"`` row left behind are removed together with the projection.
        The caller holds the workspace write lock across the whole rewrite and
        writes only Catalog rows between pages, so the entity is read with
        ``held=True``: a route without its own ``scan_page`` is enumerated in
        one ``iter_scan`` pass rather than once per page, unless another route
        of the entity also lacks its own ``scan_page`` (see
        ``iter_row_pages``). Pages then come in no particular ``object_id``
        order.
        Keys occupied by Catalog rows outside this workspace are skipped.

        Pages committed before a failure stay in the new form. Pages written
        in an open SQL registration transaction roll back with it. ``audit()``
        reports rows that do not match the registration in effect as missing,
        dangling, or mismatched, and ``repair()`` rewrites exactly those rows.

        Args:
            entity_cls (type): Entity class whose rows are reconciled.
            mode (str): Target Catalog mode: ``"full"``, ``"identity"``, or ``"none"``.
            prepared_rows: Source identity pages materialized before strategy staging, when supplied.
            stored_full: The previous full registration is available only from storage.

        Returns:
            Dict[str, int]: ``written`` and ``deleted`` Catalog row counts.

        Raises:
            RuntimeError: If a page cannot be rewritten; the message carries the
                counts completed so far and the original error is chained.
        """
        ws = self.workspace
        schema = entity_cls.schema()
        entity_id = schema.entity_id
        counts = {"written": 0, "deleted": 0}
        catalog_entity_id = Catalog.schema().entity_id
        hash_length = ws._catalog.hash_length()
        # The workspace write lock is held until the pages are closed, and the rewrite writes only Catalog rows between
        # them, so held=True may read a route that has only the portable scan_page in one pass instead of once per page.
        pages = self._target_pages(schema, hash_length) if prepared_rows is None else prepared_rows
        try:
            for ids_by_row in pages:
                existing = self._existing_targets(ids_by_row)
                ids_by_row = [(row, catalog_id) for row, catalog_id in ids_by_row if not self._outside_workspace(existing.get(catalog_id))]
                if mode == "none":
                    ops = [RowOp(catalog_entity_id, catalog_id, op="delete") for _, catalog_id in ids_by_row]
                else:
                    ops = []
                    for row, catalog_id in ids_by_row:
                        data = dict(row)
                        if prepared_rows is None:
                            ws._materializer.materialize(entity_cls, data)
                        catalog_row = ws._catalog.row(
                            entity_cls, schema, data[schema.pk_name], data, existing=existing.get(catalog_id), hash_length=hash_length, mode=mode
                        )
                        ops.append(RowOp(catalog_entity_id, catalog_row["object_id"], catalog_row))
                counts["deleted" if mode == "none" else "written"] += len(self.write_owned(ops))
        except Exception as exc:
            raise self._rewrite_error(entity_id, mode, **counts, stored_full=stored_full) from exc
        finally:
            pages.close()
        return counts
