"""Private schema registration and validation service for workspaces."""

from __future__ import annotations

import json
import warnings
from contextlib import ExitStack

from heavenbase.utils import Any, Dict, List, Mapping, Optional, Sequence

from ...backends.base import _RegistrationPublication
from ...entity import Entity
from ...utils.values import kebab
from ..catalog.entity import Catalog
from ..foundation import FIXED_ENTITY_IDS, fixed_entity_type
from ..manifest import _encoded_definition, _same_manifest_data
from ..metaschema.entity import MetaSchema
from .options import _OPTION_KEYS, _ROUTE_GENERATIONS_KEY, EntityOptions


class _SchemaService:
    """Own entity input normalization, registration, alteration, and schema validation flow."""

    def __init__(self, workspace: Any):
        self.workspace = workspace
        # entity id -> the saved replay definition this facade last wrote or knowingly left behind
        self._known_saved: Dict[str, Dict[str, Any]] = {}
        # compute hooks this facade was given by name; a saved definition naming one replays with it when compared
        self._computes: Dict[str, Any] = {}
        self._query_computes: Dict[str, Any] = {}
        # set while a manifest is replayed into this facade before it is published
        self._replaying = False
        # entity id -> route -> how many route changes after the entity's first routes it joined; absent routes are 0
        self._route_generations: Dict[str, Dict[str, int]] = {}

    def absence_routes(self, entity_id: str) -> List[str]:
        """Return the routes of an entity whose miss can show that a row is absent.

        Every row is written to every route the entity has at the time, so the
        routes that joined first hold every row any route of the entity holds.
        A route that ``register(..., allow_replace=True)`` added later holds only
        the rows written after it joined, so its miss shows nothing while an
        older route is left. Once every older route is gone, the rows it lacks
        are held only by routes the entity no longer reads.

        Args:
            entity_id: The registered entity.

        Returns:
            List[str]: Backend names of the routes that joined first, in route order.
        """
        routes = self.workspace._row_store.routed_backend_names(entity_id)
        generations = self._route_generations.get(entity_id, {})
        first = min((generations.get(name, 0) for name in routes), default=0)
        return [name for name in routes if generations.get(name, 0) == first]

    def route_generations(self, entity_id: str) -> Dict[str, int]:
        """Return the routes of an entity that joined after its first routes, with the route change each joined in."""
        return dict(self._route_generations.get(entity_id, {}))

    @staticmethod
    def _joined_generations(previous_routes: set, previous: Mapping[str, int], routes: set) -> Dict[str, int]:
        """Return the route generations after a registration replaces ``previous_routes`` with ``routes``.

        A route the previous registration also had keeps its generation. A route it lacked joins after every
        route that stays. When no route stays, no route the entity reads holds an earlier row, so every route
        starts afresh. Generations are shifted so the oldest route is 0, and only nonzero ones are kept.
        """
        kept = previous_routes & routes
        if not kept:
            return {}
        joined = {name: previous.get(name, 0) for name in kept}
        latest = max(joined.values())
        joined.update({name: latest + 1 for name in routes - kept})
        first = min(joined.values())
        return {name: generation - first for name, generation in sorted(joined.items()) if generation != first}

    def _stored_generations(self, entity_id: str, stored: Mapping[str, Any]) -> tuple[set, Dict[str, int]]:
        """Return the routes and route generations that a stored MetaSchema projection records for an entity."""
        metaschema = self.workspace._metaschema
        routes = {str(row["backend"]) for row in stored.values() if row.get("kind") == "storage" and row.get("subject_id") == entity_id and row.get("backend")}
        entity_row = stored.get(metaschema.system_row_id("entity", entity_id))
        meta = metaschema._row_meta(entity_row) if entity_row is not None and entity_row.get("meta") is not None else None
        recorded = meta.get(_ROUTE_GENERATIONS_KEY) if isinstance(meta, Mapping) else None
        generations = {}
        if isinstance(recorded, Mapping):
            generations = {str(name): value for name, value in recorded.items() if isinstance(value, int) and not isinstance(value, bool) and value >= 0}
        return routes, generations

    def register(
        self,
        entity: Any,
        *,
        computes: Optional[Mapping[str, Any]] = None,
        query_computes: Optional[Mapping[str, Any]] = None,
        allow_replace: bool = False,
        catalog: Optional[str] = None,
        gram: Optional[bool] = None,
        replay_definition: Optional[Mapping[str, Any]] = None,
    ) -> type:
        """Register an entity while the workspace lifecycle lock is held.

        A change to an entity that the workspace's saved replay manifest
        defines is written to that manifest; see ``follow_saved_definition``.
        """
        self._remember_computes(computes, query_computes)
        return self._register(
            entity,
            computes=computes,
            query_computes=query_computes,
            allow_replace=allow_replace,
            catalog=catalog,
            gram=gram,
            fixed=False,
            replay_definition=replay_definition,
        )

    def _register_fixed(self, entity: type) -> type:
        """Register one exact code-owned workspace foundation Entity."""
        entity_id = entity.schema().entity_id
        if entity_id not in FIXED_ENTITY_IDS or fixed_entity_type(entity_id) is not entity:
            raise ValueError(f"Entity {entity_id!r} is not an exact fixed workspace contract")
        return self._register(entity, allow_replace=False, fixed=True)

    def _register(
        self,
        entity: Any,
        *,
        computes: Optional[Mapping[str, Any]] = None,
        query_computes: Optional[Mapping[str, Any]] = None,
        allow_replace: bool = False,
        catalog: Optional[str] = None,
        gram: Optional[bool] = None,
        fixed: bool,
        reattaching: bool = False,
        verify_stored_schema: Optional[Mapping[str, Any]] = None,
        replay_definition: Optional[Mapping[str, Any]] = None,
    ) -> type:
        """Register an ordinary or explicitly authorized fixed Entity."""
        ws = self.workspace
        declared = entity if isinstance(entity, Mapping) else None
        entity_cls = self._definition_class(entity, computes=computes, query_computes=query_computes, action="registered with a different definition")
        schema = entity_cls.schema()
        if schema.entity_id in FIXED_ENTITY_IDS:
            canonical = fixed_entity_type(schema.entity_id)
            if not fixed or entity_cls is not canonical:
                raise ValueError(f"Fixed workspace Entity {schema.entity_id!r} cannot be registered or replaced")
        is_system_entity = schema.entity_id in FIXED_ENTITY_IDS
        previous_options = ws.entity_options.get(schema.entity_id)
        # Without a live registration the store's MetaSchema entity row says which registration wrote the stored rows:
        # an entity row from a release without options recorded the defaults, and no entity row means nothing was stored.
        stored_options = None if previous_options is not None or is_system_entity else ws._metaschema.stored_options(schema.entity_id)
        options = EntityOptions.resolve(previous_options if previous_options is not None else stored_options, declared, catalog=catalog, gram=gram)
        if is_system_entity and options != EntityOptions():
            raise ValueError(f"Fixed workspace Entity {schema.entity_id!r} keeps the default catalog and gram options")
        if not (fixed or reattaching) and previous_options is not None and schema.entity_id in ws.entities:
            if self._is_extension_definition(ws.entities[schema.entity_id], entity_cls, options, previous_options, "registered with a different definition"):
                # The extension's own definition again: keep the extension's class registered, as a plain re-registration does.
                allow_replace = False
        self.validate_derives(entity_cls)
        follow_manifest = not (fixed or reattaching or self._replaying)
        before = self.saved_definition_state(schema.entity_id) if follow_manifest else None
        if schema.entity_id in ws.entities and not allow_replace:
            existing = ws.entities[schema.entity_id]
            if ws._schema_metadata.entity_structure(existing) != ws._schema_metadata.entity_structure(entity_cls):
                raise ValueError(f"Entity {schema.entity_id!r} is already registered with a different structure; pass allow_replace=True explicitly")
            if previous_options is not None and options != previous_options:
                raise ValueError(
                    f"Entity {schema.entity_id!r} is already registered with catalog={previous_options.catalog!r} and "
                    f"gram={previous_options.gram!r}; pass allow_replace=True explicitly to change them"
                )
            ws._schema_metadata.refresh_metadata(existing, entity_cls)
            ws._metaschema.publish_schema_rows(existing)
            if follow_manifest:
                self.follow_saved_definition(existing, before)
            return existing
        if stored_options is not None and options != stored_options and not allow_replace:
            # A changed Catalog mode rewrites every stored Catalog row and a changed gram option re-places the stored
            # keyword index, so a fresh process asks for the same explicit consent as a live re-registration.
            raise ValueError(
                f"Entity {schema.entity_id!r} is stored with catalog={stored_options.catalog!r} and gram={stored_options.gram!r}; "
                "pass allow_replace=True explicitly to change them"
            )
        stored_fallback: Optional[EntityOptions] = None
        stored_entity_cls = entity_cls
        cold_catalog = None
        if stored_options is not None and options != stored_options:
            if not stored_options.gram and options.gram:
                stored_entity_cls = ws._schema_placement.with_inline_keywords(entity_cls)
            if options.catalog != stored_options.catalog:
                if options.gram != stored_options.gram:
                    # Reject incompatible incoming placement before using the definition to prepare the old mode.
                    try:
                        ws._schema_placement.resolve(entity_cls, gram=options.gram)
                    except Exception:
                        self._reestablish_stored(stored_entity_cls, stored_options, fixed=fixed)
                        raise
                published = ws._metaschema.stored_schema_rows(schema.entity_id)
                old_plan = ws._schema_placement.resolve(stored_entity_cls, gram=stored_options.gram)
                if ws._metaschema.matches_stored_schema(stored_entity_cls, old_plan, stored_options, published):
                    self._register(
                        stored_entity_cls,
                        allow_replace=allow_replace,
                        catalog=stored_options.catalog,
                        gram=stored_options.gram,
                        fixed=fixed,
                        reattaching=True,
                        verify_stored_schema=published,
                    )
                    return self._register(
                        entity_cls, allow_replace=True, catalog=options.catalog, gram=options.gram, fixed=fixed, replay_definition=replay_definition
                    )
                cold_catalog = ws._metaschema.stored_storage(schema.entity_id, published)
                if stored_options.catalog != "none":
                    for name in sorted({binding.backend for binding in old_plan.bindings}):
                        ws.backends.get(name)._validate_registration_reattachment(stored_entity_cls.schema(), old_plan)
                if stored_options.catalog in ("none", "full"):
                    stored_schema, stored_plan = cold_catalog
                    for name in sorted({binding.backend for binding in stored_plan.bindings}):
                        ws.backends.get(name)._prepare_registration_scan(stored_schema, stored_plan)
            # A gram-only change rewrites no Catalog row, so it is applied directly and the stored registration is
            # re-established only if it fails. A gram=False change therefore never first rebuilds the keyword index it
            # is about to retire (an index HeavenBase 0.1.2.2 wrote has no shared fingerprint yet).
            if cold_catalog is None:
                stored_fallback = stored_options
        missing = object()
        previous_entity = ws.entities.get(schema.entity_id, missing)
        previous_plan = ws.storage_plans.get(schema.entity_id, missing)
        previous_generations = self._route_generations.get(schema.entity_id, missing)
        binding = (ws.context.identity_token, ws.id)
        tracked_before = binding in getattr(entity_cls, "_hb_workspace_bindings", set())
        precursor = previous_options or stored_options
        reconcile = precursor is not None and precursor.catalog != options.catalog
        leaving_full = precursor is not None and precursor.catalog == "full" and reconcile
        stored_full = leaving_full and cold_catalog is not None
        attempted = []
        staged = []
        saved_schema_rows = None
        attempted_schema_rows: dict[str, dict | None] | None = None
        published_registry_digest = ws._metaschema._published_registry_digest
        publication = _RegistrationPublication()
        preparation = ExitStack()
        prepared_rows = None
        try:
            ws._router._near.cache.invalidate(schema.entity_id)
            storage_plan = ws._schema_placement.resolve(entity_cls, gram=options.gram)
            if verify_stored_schema is not None and not ws._metaschema.matches_stored_schema(entity_cls, storage_plan, options, verify_stored_schema):
                raise ValueError("The supplied entity definition differs from the stored schema")
            backends = [ws.backends.get(name) for name in sorted({binding.backend for binding in storage_plan.bindings})]
            for backend in backends:
                backend.warn_schema(schema, storage_plan.bindings)
            self.preflight_unique_constraints(schema, storage_plan)
            if reconcile and previous_options is not None and previous_options.catalog == "none":
                # Rows stored without Catalog rows may hold ids no Catalog row can take; refuse the change before switching.
                ws._catalog_audit.require_publishable(previous_entity.schema(), mode=options.catalog)
            if cold_catalog is not None and stored_options.catalog == "none":
                ws._catalog_audit.require_publishable(cold_catalog[0], mode=options.catalog, storage_plan=cold_catalog[1])
            if stored_full:
                prepared_rows = preparation.enter_context(
                    ws._catalog_audit.prepare_catalog_rewrite(entity_cls, mode=options.catalog, stored_source=cold_catalog)
                )
            peers = list(backends)
            projection_backends = []
            if MetaSchema.identifier in ws.storage_plans:
                metadata_backend = ws.backends.get(ws._metaschema.system_backend_name(MetaSchema))
                projection_backends.append(metadata_backend)
            if leaving_full:
                projection_backends.extend(ws.backends.get(name) for name in ws._row_store.routed_backend_names(Catalog.identifier))
            for backend in projection_backends:
                if backend not in peers:
                    peers.append(backend)
            # Prepare storage before opening strategy transactions, so no schema DDL runs in them.
            for backend in backends:
                attempted.append(backend)
                staged.append(backend._stage_registration(schema, storage_plan, peers=peers, publication=publication, reattaching=reattaching))
            # Every route's storage now exists, and no publication transaction holds a connection yet.
            for backend in backends:
                backend._decide_registration_aliases(schema, storage_plan, peers)
            if cold_catalog is not None and stored_options.catalog == "none":
                ws._catalog_audit.require_publishable(schema, mode=options.catalog, storage_plan=storage_plan)
            if leaving_full and not stored_full and options.catalog == "identity":
                prepared_rows = preparation.enter_context(ws._catalog_audit.prepare_catalog_rewrite(entity_cls, mode=options.catalog, stored_source=None))
            stored_projection = None
            routes = {binding.backend for binding in storage_plan.bindings}
            if is_system_entity:
                generations: Dict[str, int] = {}
            elif previous_plan is not missing:
                previous_routes = {binding.backend for binding in previous_plan.bindings}
                generations = self._joined_generations(previous_routes, self._route_generations.get(schema.entity_id, {}), routes)
            elif verify_stored_schema is not None or stored_options is not None:
                # Only the store knows which routes held the rows written before this facade registered the entity.
                if verify_stored_schema is None:
                    metadata_backend = ws.backends.get(ws._metaschema.system_backend_name(MetaSchema))
                    stored_projection = ws._metaschema.stored_schema_rows(schema.entity_id, backend=metadata_backend)
                stored_routes, stored_generations = self._stored_generations(
                    schema.entity_id, verify_stored_schema if verify_stored_schema is not None else stored_projection
                )
                generations = self._joined_generations(stored_routes, stored_generations, routes)
            else:
                generations = {}
            self._route_generations[schema.entity_id] = generations
            rows = None
            obsolete = []
            if not reattaching and not is_system_entity and precursor is not None:
                rows = ws._metaschema.prepare_schema_rows(entity_cls, storage_plan, options)
                saved_schema_rows = ws._metaschema.capture_schema_rows(entity_cls, rows, stored=stored_projection)
                new_ids = {row["object_id"] for row in rows}
                obsolete = [row_id for row_id, row in saved_schema_rows.items() if row_id not in new_ids and row.get("ws") == ws.id]
            with ExitStack() as stages:
                for stage in staged:
                    stages.enter_context(stage)
                if leaving_full and publication.must_commit_before_projecting(projection_backends):
                    publication.commit()
                elif leaving_full:
                    ws._catalog_audit.reconcile_entity(entity_cls, mode=options.catalog, prepared_rows=prepared_rows, stored_full=stored_full)
                    reconcile = False
                ws.entities[schema.entity_id] = entity_cls
                ws.storage_plans[schema.entity_id] = storage_plan
                ws.entity_options[schema.entity_id] = options
                if not is_system_entity and ws._tracked:
                    with getattr(entity_cls, "_hb_workspace_lock", ws._lock):
                        entity_cls._hb_workspace_bindings.add(binding)
                if not reattaching:
                    if schema.entity_id not in ws._catalog.system_entity_ids():
                        ws._metaschema.publish_workspace_rows()
                    if rows is not None:
                        attempted_schema_rows = {row["object_id"]: row for row in rows}
                    ws._metaschema.publish_schema_rows(entity_cls)
                    if attempted_schema_rows is not None and obsolete:
                        attempted_schema_rows.update(dict.fromkeys(obsolete))
                        ws._metaschema.delete_schema_rows(obsolete)
                publication.commit()
        except BaseException:
            preparation.close()
            ws._metaschema._published_registry_digest = published_registry_digest
            restore_error = None
            if attempted_schema_rows is not None and saved_schema_rows is not None:
                try:
                    ws._metaschema.restore_schema_rows(saved_schema_rows, attempted_schema_rows)
                except Exception as exc:
                    restore_error = exc
                ws._metaschema.invalidate_readers()
            if previous_entity is missing:
                ws.entities.pop(schema.entity_id, None)
            else:
                ws.entities[schema.entity_id] = previous_entity
            if previous_plan is missing:
                ws.storage_plans.pop(schema.entity_id, None)
            else:
                ws.storage_plans[schema.entity_id] = previous_plan
            if previous_options is None:
                ws.entity_options.pop(schema.entity_id, None)
            else:
                ws.entity_options[schema.entity_id] = previous_options
            if previous_generations is missing:
                self._route_generations.pop(schema.entity_id, None)
            else:
                self._route_generations[schema.entity_id] = previous_generations
            if not tracked_before:
                with getattr(entity_cls, "_hb_workspace_lock", ws._lock):
                    entity_cls._hb_workspace_bindings.discard(binding)
            if previous_entity is not missing and previous_plan is not missing:
                self._restore_backends(attempted, previous_entity.schema(), previous_plan)
            elif attempted and stored_options is not None:
                try:
                    snapshot = ws._metaschema.stored_schema_rows(schema.entity_id)
                    stored_schema, stored_plan = ws._metaschema.stored_storage(schema.entity_id, snapshot)
                except (KeyError, ValueError, TypeError):
                    self._restore_backends(attempted, schema, None)
                else:
                    self._restore_backends(attempted, stored_schema, stored_plan)
            else:
                self._restore_backends(attempted, schema, None)
            if stored_fallback is not None:
                self._reestablish_stored(stored_entity_cls, stored_fallback, fixed=fixed)
            if restore_error is not None:
                warnings.warn(f"Restoring MetaSchema for entity {schema.entity_id!r} failed: {restore_error}", RuntimeWarning, stacklevel=2)
            raise
        try:
            # Persist the committed registration before any post-publication Catalog rewrite can fail.
            if replay_definition is not None and (previous_options or stored_options) not in (None, options):
                self.follow_saved_definition(entity_cls, None, replay_definition=replay_definition)
            elif follow_manifest:
                self.follow_saved_definition(entity_cls, before)
        finally:
            with preparation:
                try:
                    self._prune_backends(backends, schema, storage_plan, reattaching=reattaching)
                finally:
                    # A warning promoted to an error cannot skip a rewrite for an already committed mode.
                    if reconcile:
                        ws._catalog_audit.reconcile_entity(entity_cls, mode=options.catalog, prepared_rows=prepared_rows, stored_full=stored_full)
        return entity_cls

    def _is_extension_definition(self, existing: type, entity_cls: type, options: EntityOptions, previous: EntityOptions, action: str) -> bool:
        """Return whether ``entity_cls`` is the definition an enabled extension registered as ``existing``.

        Activation registers the extension's definition again each time the workspace opens, and a default manifest leaves
        the schema to activation, so a changed structure, option or description would be lost on reopen or import. The
        extension's code loads in a package whose name differs per Context root, so the definitions are compared as manifest
        replay compares them, with compute and derive references relative to that package: the extension's own class from a
        Context with another root, ``hb.Message`` included, is its definition.

        Returns:
            bool: ``True`` for the extension's own definition, ``False`` when no enabled extension defines the entity.

        Raises:
            ValueError: If an enabled extension defines the entity and ``entity_cls`` or ``options`` differ from it.
        """
        from ..manifest import _definition_data, _entity_manifest

        ws = self.workspace
        entity_id = existing.schema().entity_id
        owner = ws._extension_service.owner(entity_id)
        if owner is None:
            return False
        metadata = ws._schema_metadata
        if (
            options == previous
            and metadata.entity_structure(existing) == metadata.entity_structure(entity_cls)
            and _same_manifest_data(
                _definition_data(_entity_manifest(ws, existing), metadata=True), _definition_data(_entity_manifest(ws, entity_cls), metadata=True)
            )
        ):
            return True
        raise ValueError(self._extension_entity_message(entity_id, owner, action))

    def _definition_class(self, entity: Any, *, computes: Optional[Mapping[str, Any]], query_computes: Optional[Mapping[str, Any]], action: str) -> type:
        """Return the Entity class ``register`` or ``alter_entity`` was given, compiling a definition mapping.

        A mapping for an enabled extension's registered entity is compared with the live definition as data first, as
        manifest replay compares it, so the definition ``to_manifest(include_system=True)`` exports returns the extension's
        class even when it cannot be compiled (a lambda derive, or a compute hook the call does not pass). Its ``catalog``
        and ``gram`` options are compared by the caller.

        Raises:
            ValueError: If a mapping for such an entity differs from the live definition and cannot be compiled: it is
                not the extension's definition, whatever hooks the entity has.
        """
        from ..manifest import _definition_data, _entity_manifest

        ws = self.workspace
        entity_id = _definition_entity_id(entity) if isinstance(entity, Mapping) else None
        live = ws.entities.get(entity_id) if entity_id else None
        owner = ws._extension_service.owner(entity_id) if live is not None else None
        if owner is None:
            return self.entity_cls(entity, computes=computes, query_computes=query_computes, allow_dict=True)
        if _same_manifest_data(_definition_data(_entity_manifest(ws, live), metadata=True), _definition_data(entity, metadata=True)):
            return live
        try:
            return self.entity_cls(entity, computes=computes, query_computes=query_computes, allow_dict=True)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(self._extension_entity_message(entity_id, owner, action)) from error

    @staticmethod
    def _changes_metadata(entity_cls: type, desc: Optional[str], tags: Optional[Sequence[str]], fields: Optional[Mapping[str, Mapping[str, Any]]]) -> bool:
        """Return whether ``describe`` with these arguments would change the entity's metadata."""
        schema = entity_cls.schema()
        if desc is not None and str(desc) != schema.desc:
            return True
        if tags is not None and not _same_manifest_data(list(tags), list(schema.tags)):
            return True
        for name, meta in (fields or {}).items():
            field_schema = schema.get(str(name))
            if "desc" in meta and str(meta.get("desc") or "") != field_schema.desc:
                return True
            for key in ("tags", "examples"):
                if key in meta and not _same_manifest_data(list(meta.get(key) or []), list(getattr(field_schema, key))):
                    return True
        return False

    @staticmethod
    def _extension_entity_message(entity_id: str, owner: str, action: str) -> str:
        return (
            f"Entity {entity_id!r} is defined by extension {owner!r} and cannot be {action}: the extension registers its own "
            "definition each time the workspace opens, so the change would be lost. Keep extra fields in a separate entity."
        )

    def _reestablish_stored(self, entity_cls: type, stored: EntityOptions, *, fixed: bool) -> None:
        """Reattach a matching stored schema after a gram-only change from a fresh process failed.

        The caller raises the failure of the change; a failure here is reported as a ``RuntimeWarning``, and the entity
        is then left unregistered in this workspace until it is registered again.
        """
        try:
            published = self.workspace._metaschema.stored_schema_rows(entity_cls.schema().entity_id)
            self._register(
                entity_cls, allow_replace=True, catalog=stored.catalog, gram=stored.gram, fixed=fixed, reattaching=True, verify_stored_schema=published
            )
        except Exception as exc:  # reported: the caller raises the original failure
            warnings.warn(
                f"Changing the gram option of entity {entity_cls.schema().entity_id!r} failed, and re-establishing its stored "
                f"registration (catalog={stored.catalog!r}, gram={stored.gram!r}) failed as well: {type(exc).__name__}: {exc}. "
                "The entity is not registered in this workspace; register it again.",
                RuntimeWarning,
                stacklevel=3,
            )

    @staticmethod
    def _prune_backends(backends: Sequence[Any], schema: Any, storage_plan: Any, *, reattaching: bool) -> None:
        """Release the strategy state that the published registration no longer binds.

        The registration has taken effect, so a failure here is reported as a ``RuntimeWarning`` instead of raised. A
        strategy adapter deletes a field's stored index before it stops maintaining the field and checks the field
        again the next time a registration binds it, so a failed release never leaves an index that is trusted while
        stale: one this backend maintained stays maintained, which costs writes, and while one it did not maintain
        keeps its fingerprint, this backend's writes of the entity raise. Either lasts until the entity is registered
        again with ``allow_replace=True`` or in a new process.
        """
        failures = []
        for backend in backends:
            try:
                backend._prune_registration(schema, storage_plan.bindings, backends, reattaching=reattaching)
            except Exception as exc:  # reported: the registration stands
                failures.append(f"{backend.name!r}: {type(exc).__name__}: {exc}")
        if failures:
            warnings.warn(
                f"Entity {schema.entity_id!r} is registered, but releasing the strategy indexes it no longer binds failed on "
                f"backend {'; '.join(failures)}. Each stays in place, maintained where this backend maintained it, until the "
                "entity is registered again with allow_replace=True or in a new process.",
                RuntimeWarning,
                stacklevel=2,
            )

    @staticmethod
    def _restore_backends(backends: Sequence[Any], schema: Any, previous_plan: Any) -> None:
        """Re-establish the live registration on the backends that a failed registration ensured.

        The publication context rolls back sparse-GRAM changes. Staging the previous plan re-creates any scalar
        indexes the failed plan dropped. Its publication context is discarded, so this can restore storage without
        rebuilding a shared sparse-GRAM index with the previous local options. Other adapters release what they
        prepared through ``abandon``. The caller raises the original failure and reports any failure here as a
        ``RuntimeWarning``.
        """
        bindings = previous_plan.bindings if previous_plan is not None else ()
        previous_backends = {binding.backend for binding in bindings}
        failures = []
        for backend in backends:
            if backend.name in previous_backends:
                try:
                    backend._stage_registration(schema, previous_plan)
                except Exception as exc:  # reported: the caller raises the original failure
                    failures.append(f"{backend.name!r} (ensure): {type(exc).__name__}: {exc}")
            try:
                backend.abandon(schema, bindings)
            except Exception as exc:  # reported: the caller raises the original failure
                failures.append(f"{backend.name!r} (release): {type(exc).__name__}: {exc}")
        if failures:
            undone = "re-establishing its previous registration" if previous_plan is not None else "undoing what it set up"
            warnings.warn(
                f"Registering entity {schema.entity_id!r} failed, and {undone} failed as well "
                f"on backend {'; '.join(failures)}. Register the entity again with allow_replace=True to settle its indexes.",
                RuntimeWarning,
                stacklevel=2,
            )

    def preflight_unique_constraints(self, schema: Any, storage_plan: Any) -> None:
        """Reject duplicate stored values before attempting uniqueness DDL.

        Args:
            schema: Incoming entity schema.
            storage_plan: Resolved plan containing uniqueness owners.

        Returns:
            None: Valid stored data returns without mutation.

        Raises:
            ValueError: If an enforcement backend already contains duplicate non-null tuples.
        """
        ws = self.workspace
        for constraint in storage_plan.constraints:
            backend = ws.backends.get(constraint.backend)
            try:
                rows = backend.scan(schema.entity_id)
            except NotImplementedError:
                continue
            seen: dict[tuple[Any, ...], Any] = {}
            for row in rows:
                key = tuple(row.get(field_name) for field_name in constraint.fields)
                if any(value is None for value in key):
                    continue
                object_id = row.get("object_id")
                if key in seen and seen[key] != object_id:
                    fields = ", ".join(constraint.fields)
                    raise ValueError(
                        f"Cannot enforce unique constraint on {schema.entity_id}({fields}); value {key!r} is used by object_ids "
                        f"{seen[key]!r} and {object_id!r} on backend {constraint.backend!r}"
                    )
                seen[key] = object_id

    def alter(
        self, entity: Any, *, computes: Optional[Mapping[str, Any]] = None, query_computes: Optional[Mapping[str, Any]] = None, confirm: str = ""
    ) -> type:
        """Alter an existing entity with guarded additive schema changes."""
        ws = self.workspace
        self._remember_computes(computes, query_computes)
        entity_cls = self._definition_class(entity, computes=computes, query_computes=query_computes, action="altered")
        schema = entity_cls.schema()
        entity_id = schema.entity_id
        if entity_id in FIXED_ENTITY_IDS:
            raise ValueError(f"System entity {entity_id!r} cannot be altered")
        if entity_id not in ws.entities:
            raise KeyError(f"Workspace {ws.id!r} has no registered entity {entity_id!r}; use register first")
        if confirm != entity_id:
            raise ValueError(f"alter_entity requires confirm={entity_id!r}")
        options = ws.entity_options[entity_id]
        if isinstance(entity, Mapping) and EntityOptions.resolve(options, entity, catalog=None, gram=None) != options:
            # A changed Catalog mode rewrites stored Catalog rows and a changed gram option re-places the keyword index, so
            # neither is an additive alteration; register(..., allow_replace=True) is the explicit path for both.
            raise ValueError(
                f"alter_entity cannot change the catalog or gram option of {entity_id!r} (registered with catalog={options.catalog!r} and "
                f"gram={options.gram!r}); use register(..., allow_replace=True)"
            )
        if self._is_extension_definition(ws.entities[entity_id], entity_cls, options, options, "altered"):
            return ws.entities[entity_id]
        new_storage_plan = ws._schema_placement.resolve(entity_cls, gram=options.gram)
        self.validate_entity_alter(ws.entities[entity_id], entity_cls)
        old_fields = set(ws.entities[entity_id].schema().fields)
        ws._schema_placement.validate_unchanged(ws.storage_plans[entity_id], new_storage_plan, old_fields)
        return self.register(entity_cls, allow_replace=True)

    def describe(
        self, entity: Any, *, desc: Optional[str] = None, tags: Optional[Sequence[str]] = None, fields: Optional[Mapping[str, Mapping[str, Any]]] = None
    ) -> type:
        """Apply metadata-only descriptions, tags, and examples to a registered entity."""
        from ...interop.base import apply_entity_meta

        ws = self.workspace
        entity_cls = self.require(entity)
        if entity_cls.schema().entity_id in FIXED_ENTITY_IDS:
            raise ValueError(f"Fixed workspace Entity {entity_cls.schema().entity_id!r} cannot be described or replaced")
        owner = ws._extension_service.owner(entity_cls.schema().entity_id)
        if owner is not None:
            if not self._changes_metadata(entity_cls, desc, tags, fields):
                return entity_cls
            raise ValueError(self._extension_entity_message(entity_cls.schema().entity_id, owner, "described"))
        before = self.saved_definition_state(entity_cls.schema().entity_id)
        apply_entity_meta(entity_cls, desc=desc, tags=tags, fields=fields)
        ws._metaschema.publish_schema_rows(entity_cls)
        self.follow_saved_definition(entity_cls, before)
        return entity_cls

    def _remember_computes(self, computes: Optional[Mapping[str, Any]], query_computes: Optional[Mapping[str, Any]]) -> None:
        """Keep the compute hooks a registration names, so a saved definition naming them can be compared with a live one.

        Each hook is kept under its name and under the reference ``to_manifest()`` exports for it, so both the saved
        definition and the live one resolve it.
        """
        metadata = self.workspace._schema_metadata
        for mapping, known in ((computes, self._computes), (query_computes, self._query_computes)):
            for name, fn in (mapping or {}).items():
                known[name] = fn
                if callable(fn):
                    known[metadata.callable_ref(fn)] = fn

    def saved_definition_state(self, entity_id: Optional[str]) -> Optional[Dict[str, Any]]:
        """Return the live definition ``follow_saved_definition`` compares a schema change against.

        Args:
            entity_id: Identifier of the entity about to change, when the input names one.

        Returns:
            Dict[str, Any] | None: The entity's current manifest definition, or ``None`` when it is
            not registered.
        """
        ws = self.workspace
        if entity_id is None or entity_id not in ws.entities:
            return None
        from ..manifest import _entity_manifest

        return _entity_manifest(ws, ws.entities[entity_id])

    def follow_saved_definition(self, entity_cls: type, before: Optional[Dict[str, Any]], *, replay_definition: Optional[Mapping[str, Any]] = None) -> None:
        """Write a changed live definition into the workspace's saved replay manifest.

        Reopening a manifest-backed workspace by id (``HeavenBase(id)``, ``HeavenBase.load(id)``,
        ``WorkspaceSpec.open_detached()``) replays the manifest saved in its ``WorkspaceSpec``. A
        definition left there after ``register(..., allow_replace=True)``, ``alter_entity`` or
        ``describe`` changed the live one would be replayed over the stored registration: an added
        field would vanish and the next write would store NULL in its column. So the saved entry of an
        entity the manifest defines (by ``entity_id``, or by the ``name`` it is derived from) is
        replaced with the live definition, as ``to_manifest()`` exports it. Entities the manifest does
        not define are not added. The workspace registry is refreshed before the read and the
        read-modify-write is repeated when another Context or process published in between, so the
        definitions of other entities it saved are kept.

        The saved entry is left unchanged, with a ``UserWarning``, when the change was made through a
        detached or untracked facade built from the same construction, its first registration of the
        entity included (a first registration is compared with the registry as this Context last read it,
        so a manifest another Context published after that is not seen and no warning is emitted), or
        when another Context or process saved a different definition of this entity after this facade
        read it (the saved entry neither replays to ``before`` nor is the one this facade last
        wrote or left), or when the live definition cannot be replayed (a value that is not JSON,
        other than a date or datetime default or example, or a
        compute hook not registered as a module). In the last case the saved entry takes the live
        ``catalog`` and ``gram`` options if its own options would disagree with the store. No warning is emitted when
        the saved entry already replays to the live definition: a detached registration that matches
        it, or the same change saved first by another Context or process. A tracked manifest replay
        can explicitly replace options in an existing store. It saves that entity's supplied
        definition as soon as the new options commit, before replay continues.

        Args:
            entity_cls: Entity class now registered.
            before: The value ``saved_definition_state`` returned before the change.
            replay_definition: The supplied definition during a tracked manifest replacement.

        Returns:
            None: The saved manifest is updated in place when the definition changed.
        """
        ws = self.workspace
        entity_id = entity_cls.schema().entity_id
        if ws.entities.get(entity_id) is not entity_cls:
            return
        from ..manifest import _entity_manifest

        after = _entity_manifest(ws, entity_cls)
        if _same_manifest_data(after, before):
            return
        following = replay_definition is not None or (bool(getattr(ws, "_tracked", False)) and not getattr(ws, "_detached", False))
        if self._replaying and replay_definition is None:
            return
        from ...utils.registry import RegistryConflictError
        from ..registry import _registry

        context = ws.context
        registry = _registry(context)
        with context._lock:
            if before is None and self._saved_spec(entity_id) is None:
                # A first registration is checked against the cached snapshot, so registering many new entities does not re-read the registry each time.
                return
            registry.refresh()
            spec = self._saved_spec(entity_id)
            if spec is None:
                return
            saved = self._saved_item(spec, entity_id)
            if not following:
                if ws._construction == spec.construction and not self._same_definition(saved, after):
                    warnings.warn(
                        f"Entity {entity_id!r} of workspace {ws.id!r} changed through a detached or untracked facade, so the definition "
                        f"saved for reopening the workspace by id was not updated. {self._reopen_note(saved, after)} Make the change through "
                        f"the tracked workspace (HeavenBase.load({ws.id!r})) to save it.",
                        UserWarning,
                        stacklevel=2,
                    )
                return
            while True:
                if replay_definition is not None and ws._construction != spec.construction:
                    return
                if before is not None and not self._knows_saved(entity_id, saved, before):
                    if self._same_definition(saved, after):
                        # Another Context or process saved this same change first.
                        self._known_saved[entity_id] = saved
                        return
                    warnings.warn(
                        f"The definition saved for reopening workspace {ws.id!r} by id was not updated for entity {entity_id!r}: another "
                        "Context or process saved a different definition of it after this facade read it, and that definition is kept. "
                        f"{self._reopen_note(saved, after)} To save the change, release this facade, reopen the workspace with "
                        f"HeavenBase.load({ws.id!r}), which replays the saved definition, and make the change again.",
                        UserWarning,
                        stacklevel=2,
                    )
                    return
                problem = None
                if replay_definition is not None:
                    definition = _encoded_definition(replay_definition)
                else:
                    problem = self._replay_problem(after, entity_cls)
                    # An unreplayable definition keeps the saved schema and updates any conflicting options.
                    definition = json.loads(json.dumps(after)) if problem is None else self._with_options(saved, after)
                if self._write_saved_definition(spec, entity_id, definition):
                    self._known_saved[entity_id] = definition
                    if problem is not None:
                        warnings.warn(
                            f"The definition saved for reopening workspace {ws.id!r} by id was not updated for entity {entity_id!r}: "
                            f"{problem}. Reopening it by id replays the earlier definition, with the current catalog and gram options, "
                            "without this change.",
                            UserWarning,
                            stacklevel=2,
                        )
                    return
                # Another Context or process published the workspace registry since it was read.
                if not registry.refresh():
                    raise RegistryConflictError(f"Workspace {ws.id!r} registry conflict did not resolve after a refresh")
                spec = self._saved_spec(entity_id)
                if spec is None:
                    return
                saved = self._saved_item(spec, entity_id)

    def _saved_item(self, spec: Any, entity_id: str) -> Dict[str, Any]:
        """Return the saved definition of ``entity_id`` in ``spec``'s manifest."""
        return next(
            dict(item) for item in (spec.manifest or {}).get("entities") or [] if isinstance(item, Mapping) and _definition_entity_id(item) == entity_id
        )

    def _knows_saved(self, entity_id: str, saved: Dict[str, Any], before: Dict[str, Any]) -> bool:
        """Return whether ``saved`` is the definition this facade last knew: the one it wrote or left, or its live one before the change."""
        return _same_manifest_data(saved, self._known_saved.get(entity_id)) or self._same_definition(saved, before)

    def _same_definition(self, saved: Mapping[str, Any], live: Mapping[str, Any]) -> bool:
        """Return whether a saved definition replays to the live one, comparing only the options the saved one states."""
        if _same_manifest_data(saved, live):
            return True
        replayed, current = self._normalized(saved), self._normalized(live)
        if replayed is None or not _same_manifest_data(replayed, current):
            return False
        return self._same_options(saved, live)

    @staticmethod
    def _same_options(saved: Mapping[str, Any], live: Mapping[str, Any]) -> bool:
        """Compare replay options, letting omitted saved keys inherit the current registration."""
        live_options = EntityOptions.resolve(None, live, catalog=None, gram=None)
        return EntityOptions.resolve(live_options, saved, catalog=None, gram=None) == live_options

    def _normalized(self, definition: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
        """Return ``definition`` replayed and exported as ``to_manifest()`` exports it, without options; ``None`` if it cannot replay.

        A compute hook named only through the ``computes``/``query_computes`` mapping this facade was given resolves through
        that mapping here, so such a definition compares equal to the live one it was registered as.
        """
        from ..manifest import _entity_manifest

        payload = {key: value for key, value in definition.items() if key not in _OPTION_KEYS}
        try:
            payload = json.loads(json.dumps(payload, allow_nan=False))
            entity_cls = Entity.from_schema(
                payload,
                computes=self._computes or None,
                query_computes=self._query_computes or None,
                resolver=self.workspace.module_resolver,
            )
            exported = _entity_manifest(self.workspace, entity_cls)
        except (KeyError, TypeError, ValueError):
            return None
        return {key: value for key, value in exported.items() if key not in _OPTION_KEYS}

    def _with_options(self, saved: Mapping[str, Any], after: Mapping[str, Any]) -> Dict[str, Any]:
        """Replace saved options when they would reopen with conflicting values."""
        if self._same_options(saved, after):
            return dict(saved)
        definition = {key: value for key, value in saved.items() if key not in _OPTION_KEYS}
        definition.update({key: after[key] for key in _OPTION_KEYS if key in after})
        return definition

    def _reopen_note(self, saved: Mapping[str, Any], after: Mapping[str, Any]) -> str:
        """Say what reopening the workspace by id does with ``saved`` after the store took ``after``'s options."""
        if not self._same_options(saved, after):
            return (
                "Reopening it by id raises ValueError, because the saved definition states other catalog or gram options than the "
                "store now has, until HeavenBase.from_manifest(..., allow_replace=True) replays a definition with the new options."
            )
        return "Reopening it by id replays the saved definition without this change, so a field it added is not read."

    def _saved_spec(self, entity_id: str) -> Any:
        """Return the workspace's ``WorkspaceSpec`` when its saved manifest defines ``entity_id``."""
        from ..registry import WorkspaceSpec

        ws = self.workspace
        try:
            spec = WorkspaceSpec.load(ws.id, context=ws.context)
        except KeyError:
            return None
        entities = (spec.manifest or {}).get("entities") or []
        if not any(isinstance(item, Mapping) and _definition_entity_id(item) == entity_id for item in entities):
            return None
        return spec

    def _write_saved_definition(self, spec: Any, entity_id: str, definition: Dict[str, Any]) -> bool:
        """Replace one saved definition; return ``False`` on a registry compare-and-set conflict."""
        from ...utils.registry import RegistryConflictError
        from ..manifest import WorkspaceManifest
        from ..registry import WorkspaceSpec

        saved = WorkspaceManifest.from_dict(spec.manifest)
        entities = [definition if _definition_entity_id(item) == entity_id else item for item in saved.entities]
        if _same_manifest_data(entities, saved.entities):
            return True
        manifest = WorkspaceManifest(
            id=saved.id,
            construction=saved.construction,
            entities=entities,
            extensions=saved.extensions,
            backend_summary=saved.backend_summary,
        ).to_dict()
        try:
            WorkspaceSpec._save_manifest(spec, manifest, context=self.workspace.context)
        except RegistryConflictError:
            return False
        return True

    def _replay_problem(self, definition: Dict[str, Any], live: type) -> Optional[str]:
        """Return why a reopen by id could not replay ``definition``, the export of ``live``, or ``None`` when it can."""
        try:
            payload = json.loads(json.dumps(definition, allow_nan=False))
        except (TypeError, ValueError) as error:
            return f"its live definition is not JSON ({error})"
        try:
            replayed = Entity.from_schema(payload, resolver=self.workspace.module_resolver)
        except (KeyError, TypeError, ValueError) as error:
            return f"its live definition cannot be replayed ({error})"
        for name, field_schema in live.schema().fields.items():
            if field_schema.has_default and replayed.schema().get(name).default != field_schema.default:
                return f"its live definition is not JSON (the default {field_schema.default!r} of field {name!r} would replay as another value)"
        return None

    def validate_entity_alter(self, existing: type, incoming: type) -> None:
        """Validate that an entity alteration is metadata-only or additive."""
        metadata = self.workspace._schema_metadata
        old_schema = existing.schema()
        new_schema = incoming.schema()
        old_fields = set(old_schema.fields)
        new_fields = set(new_schema.fields)
        removed = sorted(old_fields - new_fields)
        if removed:
            raise ValueError(f"alter_entity cannot remove fields from {old_schema.entity_id!r}: {removed}")
        for name in sorted(old_fields):
            old_field = old_schema.fields[name]
            new_field = new_schema.fields[name]
            old_sig = (old_field.dtype.to_dict(), old_field.pk, old_field.required)
            new_sig = (new_field.dtype.to_dict(), new_field.pk, new_field.required)
            if old_sig != new_sig:
                raise ValueError(f"alter_entity cannot change dtype, primary key, or required flag for field {name!r}")
            old_computes = [metadata.compute_structure_dict(hook) for hook in metadata.field_spec(existing, name).computes]
            new_computes = [metadata.compute_structure_dict(hook) for hook in metadata.field_spec(incoming, name).computes]
            if old_computes != new_computes:
                raise ValueError(f"alter_entity cannot change compute hooks for existing field {name!r}; register a replacement with a migration/backfill plan")
            old_query_computes = [metadata.query_compute_structure_dict(hook) for hook in metadata.field_spec(existing, name).query_computes]
            new_query_computes = [metadata.query_compute_structure_dict(hook) for hook in metadata.field_spec(incoming, name).query_computes]
            if old_query_computes != new_query_computes:
                raise ValueError(
                    f"alter_entity cannot change query_compute hooks for existing field {name!r}; register a replacement with a migration/backfill plan"
                )
        for name in sorted(new_fields - old_fields):
            field_schema = new_schema.fields[name]
            if field_schema.required and not field_schema.has_default:
                raise ValueError(f"New field {name!r} must be optional or define a default")
        if metadata.entity_structure_derives(existing) != metadata.entity_structure_derives(incoming):
            old_targets = {item["target"] for item in metadata.entity_structure_derives(existing)}
            new_targets = {item["target"] for item in metadata.entity_structure_derives(incoming)}
            changed = sorted(old_targets ^ new_targets)
            if not changed:
                changed = sorted(
                    target for target in old_targets & new_targets if metadata.derive_by_target(existing, target) != metadata.derive_by_target(incoming, target)
                )
            raise ValueError(f"alter_entity cannot change entity derives for {old_schema.entity_id!r}; changed targets={changed}")

    def validate_derives(self, entity: type) -> None:
        """Validate generic and Catalog-specific entity derives."""
        for target, derive in getattr(entity, "_hb_derives", {}).items():
            if not derive.computes:
                raise ValueError(f"Derive target {target!r} needs compute")
        self.workspace._catalog.validate_derives(entity)

    def entity_cls(
        self, entity: Any, *, computes: Optional[Mapping[str, Any]] = None, query_computes: Optional[Mapping[str, Any]] = None, allow_dict: bool = False
    ) -> type:
        """Normalize user entity input into an Entity subclass."""
        ws = self.workspace
        if isinstance(entity, str):
            if entity not in ws.entities:
                raise KeyError(f"Workspace {ws.id!r} has no registered Entity class {entity!r}; establish the class before use")
            return ws.entities[entity]
        if isinstance(entity, Entity):
            return type(entity)
        if isinstance(entity, Mapping):
            if not allow_dict:
                raise TypeError("Entity definitions must be established through register() or alter_entity() before use")
            return Entity.from_schema(
                entity,
                computes=computes,
                query_computes=query_computes,
                resolver=ws.module_resolver,
            )
        if isinstance(entity, type) and issubclass(entity, Entity):
            return entity
        suffix = ", or an entity definition mapping" if allow_dict else ""
        raise TypeError(f"Expected Entity subclass, Entity instance, or registered entity identifier{suffix}")

    def require(self, entity: Any) -> type:
        """Resolve one already-present Entity class without mutating workspace state.

        Args:
            entity: Entity class, instance, or registered identifier.

        Returns:
            type: Canonical entity class already owned by this workspace.

        Raises:
            KeyError: If the Entity class is not present in this workspace.
            ValueError: If a supplied class reuses a registered identifier with
                a different structure.
        """
        ws = self.workspace
        entity_cls = self.entity_cls(entity)
        entity_id = entity_cls.schema().entity_id
        if entity_id not in ws.entities:
            raise KeyError(f"Workspace {ws.id!r} has no registered Entity class {entity_id!r}; establish the class before use")
        registered = ws.entities[entity_id]
        if ws._schema_metadata.entity_structure(registered) != ws._schema_metadata.entity_structure(entity_cls):
            raise ValueError(f"Entity {entity_id!r} is registered with a different structure")
        return registered


def _definition_entity_id(definition: Mapping[str, Any]) -> Optional[str]:
    """Return the entity identifier a JSON definition declares, deriving it from ``name`` as the compiler does."""
    entity_id = str(definition.get("entity_id") or "").strip()
    if entity_id:
        return entity_id
    name = str(definition.get("name") or "").strip()
    return kebab(name) if name else None
