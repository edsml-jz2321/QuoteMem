"""Workspace maintenance, health, and teardown operations."""

from __future__ import annotations

from heavenbase.utils import Any, Dict, safe_error

from ...backends.base import Backend
from ...entity import pending_entities
from ...execution.contracts import RowOp
from ..catalog.entity import Catalog
from .lifecycle import _release_workspace, _untrack_live_workspace


class _MaintenanceService:
    """Own operational workspace commands that are not row CRUD or query planning."""

    def __init__(self, workspace: Any):
        self.workspace = workspace

    def _database_specs(self) -> list[Any]:
        """Return the specs of the databases that this workspace's routes store in."""
        from ...database.database import Database

        databases = (getattr(backend, "db", None) for _, backend in self.workspace.backends.items())
        return [db.spec for db in databases if isinstance(db, Database)]

    def _need_no_connection(self) -> None:
        """Refuse the drop, before it drops anything, when this thread has a connection in use on a route's database."""
        from ...database.registry import DatabaseEngineRegistry, DropRefusedError

        try:
            for spec in self._database_specs():
                DatabaseEngineRegistry._need_no_connection(spec)
        except DropRefusedError as exc:
            raise DropRefusedError(f"Failed to drop workspace {self.workspace.id!r}: {safe_error(exc)} No route was dropped.") from exc

    def await_quiet_storage(self) -> None:
        """Wait, before the drop takes the workspace lock, for the connections now in use on this workspace's DuckDB files.

        The drop then does not hold the workspace lock, and with it the Context lock, while it waits for another
        Context's transaction on such a file. No lock is held and no file is closed meanwhile, so other calls go on;
        connections lent after the wait for their file began are left to the drop itself. A connection of this thread
        on a route's database, or a connection that stays in use for 30 seconds, raises ``DropRefusedError`` before
        anything is dropped.
        """
        from ...database.registry import DatabaseEngineRegistry, DropRefusedError

        self._need_no_connection()
        try:
            for spec in self._database_specs():
                DatabaseEngineRegistry._await_duckdb_quiet(spec)
        except DropRefusedError as exc:
            raise DropRefusedError(f"Failed to drop workspace {self.workspace.id!r}: {safe_error(exc)} No route was dropped.") from exc

    def drop_live(self) -> None:
        """Drop backend storage and remove this workspace from live/process registries."""
        ws = self.workspace
        with ws._lock:
            ws._router._near.cache.invalidate()
            self.drop_backends()
            if ws._tracked:
                # Classes registered in the dropped workspace must not re-register into a new workspace of the same id.
                # Bindings declared through Entity.register stay, as the declaration would run again in a fresh process.
                # A facade rebuilt from a manifest holds reconstructed classes, so every known class is checked, not only ws.entities.
                binding = (ws.context.identity_token, ws.id)
                for entity_cls in {*pending_entities(), *ws.entities.values()}:
                    with getattr(entity_cls, "_hb_workspace_lock", ws._lock):
                        if binding not in getattr(entity_cls, "_hb_declared_bindings", set()):
                            getattr(entity_cls, "_hb_workspace_bindings", set()).discard(binding)
            ws.entities.clear()
            ws.storage_plans.clear()
            ws.entity_options.clear()
            if ws._tracked:
                _release_workspace(ws)
            _untrack_live_workspace(ws)

    def drop_backends(self) -> None:
        """Drop storage for a workspace with writes excluded by its owner.

        Nothing is dropped while this thread has a connection in use on one of the routes' databases: every route is
        checked first, without waiting. The routes are then dropped in order. Only a DuckDB file's drop can still be
        refused, after its bounded wait for connections in use. A refusal stops the drop: the routes after it are kept,
        and the error says which routes were dropped and which routes' drops failed before it, or that none was dropped.

        Raises:
            DropRefusedError: If a route's drop was refused.
            RuntimeError: If a route's drop failed; the other routes are dropped all the same.
        """
        from ...database.registry import DropRefusedError

        ws = self.workspace
        routes: dict[int, tuple[list[str], Backend]] = {}
        for backend_name, backend in ws.backends.items():
            routes.setdefault(id(backend), ([], backend))[0].append(str(backend_name))
        self._need_no_connection()
        errors, dropped, failed = [], [], []
        for names, backend in routes.values():
            try:
                backend.drop()
            except DropRefusedError as exc:
                errors.append(f"{names[0]}: {safe_error(exc)}")
                kept = [f"Routes already dropped: {', '.join(dropped)}."] if dropped else []
                if failed:
                    kept.append(f"Routes whose drop failed, so that they can be partly dropped: {', '.join(failed)}.")
                raise DropRefusedError(f"Failed to drop workspace {ws.id!r}: {'; '.join(errors)} {' '.join(kept) or 'No route was dropped.'}") from exc
            except Exception as exc:
                errors.append(f"{names[0]}: {safe_error(exc)}")
                failed.extend(names)
            else:
                dropped.extend(names)
        if errors:
            detail = "; ".join(errors)
            raise RuntimeError(f"Failed to drop workspace {ws.id!r}: {detail}")

    def release_live(self) -> None:
        """Remove this facade from live/process lookup without dropping storage."""
        ws = self.workspace
        with ws._lock:
            ws._router._near.cache.invalidate()
            if ws._tracked:
                _release_workspace(ws)
            _untrack_live_workspace(ws)
            ws._tracked = False

    def refresh(self, entity: Any = None, backend: Any = None) -> None:
        """Refresh cache-capable backends in this workspace."""
        ws = self.workspace
        entity_id = None
        if entity is not None:
            entity_cls = ws._schema.require(entity)
            entity_id = entity_cls.schema().entity_id
        if backend is None:
            selected = [item for _, item in ws.backends.items()]
        elif isinstance(backend, Backend):
            selected = [backend]
        else:
            selected = [ws.backends.get(str(backend))]
        ws._router._near.cache.invalidate(entity_id)
        for item in selected:
            refresh = getattr(item, "refresh", None)
            if callable(refresh):
                refresh(entity_id)

    def audit(self) -> Dict[str, Any]:
        """Audit system row consistency without mutating workspace data."""
        return {"catalog": self.workspace._catalog_audit.audit()}

    def repair(self, *, dry_run: bool = False) -> Dict[str, Any]:
        """Explicitly repair system rows from authoritative entity rows."""
        ws = self.workspace
        before = ws._catalog_audit.audit()
        missing_ids = [item["expected"]["object_id"] for item in before["missing"]]
        mismatched_ids = [item["expected"]["object_id"] for item in before["mismatched"]]
        dangling_ids = [item["catalog"].get("object_id") for item in before["dangling"] if item["catalog"].get("object_id") is not None]
        # a stored row whose keyword rows differ is written again, with keyword maintenance, from its source
        settled = set(missing_ids) | set(mismatched_ids) | set(dangling_ids)
        keyword_ids = [item["object_id"] for item in before.get("stale_keywords", ()) if item["object_id"] not in settled]
        rewritten_ids: set[str] = set()
        deleted_ids: set[str] = set()
        if not dry_run:
            catalog_entity_id = Catalog.schema().entity_id
            rewrite_rows = ws._catalog_audit.repair_rows(missing_ids + mismatched_ids + keyword_ids)
            ops = [RowOp(catalog_entity_id, row["object_id"], row) for row in rewrite_rows]
            rewritten_ids = ws._catalog_audit.write_owned(ops)
            deleted_ids = ws._catalog_audit.write_owned([RowOp(catalog_entity_id, object_id, op="delete") for object_id in dangling_ids])
        after = before if dry_run else ws._catalog_audit.audit()
        return {
            "catalog": {
                "dry_run": dry_run,
                "missing_written": len(rewritten_ids.intersection(missing_ids)),
                "mismatched_rewritten": len(rewritten_ids.intersection(mismatched_ids)),
                "stale_keywords_rewritten": len(rewritten_ids.intersection(keyword_ids)),
                "dangling_deleted": len(deleted_ids),
                "before": before,
                "after": after,
                "ok": after["ok"],
            }
        }

    def health(self) -> Dict[str, Any]:
        """Return lightweight workspace health and backend metadata."""
        ws = self.workspace
        return {
            "workspace": ws.id,
            "entities": sorted(ws.entities),
            "backends": [ws._metaschema.registry.backend_payload(backend) for _, backend in ws.backends.items()],
        }
