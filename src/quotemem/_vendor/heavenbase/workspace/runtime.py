"""Workspace facade for the merged HeavenBase package."""

from __future__ import annotations

from heavenbase.utils import Any, Dict, Iterable, Iterator, List, Literal, Mapping, Optional, Sequence

from ..context import Context, context_or_default
from ..extensions.base import Extension
from .graph.errors import GraphDeleteRestrictedError
from .manifest import (
    WorkspaceManifest,
    open_workspace_manifest,
    restore_workspace_manifest,
    workspace_manifest,
)
from .setup.bootstrap import initialize_workspace
from .setup.lifecycle import (
    construction_write_scope,
    _WorkspaceDrop,
    _workspace_config_scope,
    load_workspace,
    prepare_workspace,
    workspace_locked,
    workspace_read_locked,
    workspace_read_pages,
    workspace_read_scope,
)

__all__ = ["HeavenBase"]


class HeavenBase:
    """Named workspace for entity definitions, storage plans, backends, and data.

    Args:
        id (str): Stable workspace identifier used for backend namespaces and system rows.
        context (Context | None): Machine composition root that owns this
            workspace's modules, configuration, and resources.
        backends (Mapping[str, Any] | None): Optional backend config or ready backend instances. Explicit backends override
            ``preset``. Defaults to configured HeavenBase backends when no preset is provided.
        preset (str | None): Optional named workspace backend preset under ``heavenbase.workspace.presets``. Defaults to ``None``.
        detached: Whether to create a caller-owned facade outside durable and
            live Context lookup. Defaults to ``False``.

    Examples:
        >>> ws = HeavenBase("shop", preset="debug")
        >>> sorted(ws.extensions())
        ['capsule', 'prompt', 'toolkit']

    Notes:
        Construction ensures a durable workspace by default. Existing
        registrations reopen from their persisted construction contract;
        conflicting explicit preset/backend input raises ``FileExistsError``.
        Detached construction never registers or activates the workspace and
        does not imply temporary or isolated backend data.
        Every workspace registers the required ``capsule`` extension and any
        configured default extensions. User entity writes publish concrete
        object rows to Catalog after entity backend writes succeed.
        Call ``HeavenBase.drop(ws_id)`` to tear down a live workspace by id, or
        ``ws.drop()`` when you already hold the workspace instance. Call
        ``ws.release()`` to remove only the live facade while retaining durable
        backend storage.
    """

    @classmethod
    def load(cls, ws_id: Optional[str] = None, *, context: Context | None = None) -> "HeavenBase":
        """Load an existing workspace through one Context.

        Args:
            ws_id (str | None): Workspace id to load. ``None`` uses the active
                registered workspace or the configured default workspace id.
                Defaults to ``None``.
            context: Machine composition root that owns the durable directory,
                module resolver, and live workspace lookup. Defaults to the
                process Context.

        Returns:
            HeavenBase: Live workspace facade.

        Raises:
            KeyError: If no matching persistent workspace registration exists.
        """
        return load_workspace(cls, ws_id, context=context_or_default(context))

    @classmethod
    def default(cls, *, context: Context | None = None) -> "HeavenBase":
        """Return one Context's configured default workspace.

        Args:
            context: Owning machine Context. Defaults to the process Context.

        Returns:
            HeavenBase: Live default workspace facade.
        """
        selected = context_or_default(context)
        return cls.load(selected.default_workspace_id, context=selected)

    drop = _WorkspaceDrop()

    def __init__(
        self,
        id: str,
        *,
        context: Context | None = None,
        backends: Optional[Mapping[str, Any]] = None,
        preset: Optional[str] = None,
        detached: bool = False,
    ) -> None:
        """Construct or reopen one workspace facade.

        Returns:
            None: This initializer does not return a value.
        """
        selected = context_or_default(context)
        with construction_write_scope(selected, id, detached=detached), selected._lock:
            selected._need_open()
            if detached:
                initialize_workspace(
                    self,
                    id,
                    context=selected,
                    backends=backends,
                    preset=preset,
                    detached=True,
                    tracked=False,
                    register_workspace=False,
                    register_pending=True,
                    track_live=False,
                    publish=False,
                    extension_roots=None,
                )
                return
            clean_id, effective_backends, effective_preset, spec = prepare_workspace(
                selected,
                id,
                backends=backends,
                preset=preset,
            )
            if spec is not None and spec.manifest:
                construction_config = {"backends": effective_backends} if spec.runtime_only else None
                open_workspace_manifest(
                    type(self),
                    spec.manifest,
                    context=selected,
                    tracked=True,
                    register_workspace=False,
                    track_live=True,
                    detached=False,
                    workspace=self,
                    construction_config=construction_config,
                    required_backend_identifiers=spec.construction.backend_identifiers or None,
                )
                return
            initialize_workspace(
                self,
                clean_id,
                context=selected,
                backends=effective_backends,
                preset=effective_preset,
                detached=False,
                tracked=True,
                register_workspace=spec is None or spec.extension_roots is None,
                register_pending=True,
                track_live=True,
                publish=True,
                extension_roots=None if spec is None else spec.extension_roots,
            )

    @classmethod
    def _construct(
        cls,
        id: str,
        *,
        context: Context,
        backends: Optional[Mapping[str, Any]] = None,
        preset: Optional[str] = None,
        detached: bool = False,
        tracked: bool = True,
        register_workspace: bool = True,
        register_pending: bool = True,
        track_live: bool = True,
        publish: bool = True,
        extension_roots: Optional[Iterable[str]] = None,
    ) -> "HeavenBase":
        with construction_write_scope(context, id, detached=detached), context._lock:
            context._need_open()
            workspace = cls.__new__(cls)
            initialize_workspace(
                workspace,
                id,
                context=context,
                backends=backends,
                preset=preset,
                detached=detached,
                tracked=tracked,
                register_workspace=register_workspace,
                register_pending=register_pending,
                track_live=track_live,
                publish=publish,
                extension_roots=extension_roots,
            )
            return workspace

    def _need_live(self) -> None:
        """Reject use after this workspace's Context generation retires."""
        self.context._need_open()
        if self._authority_token != self.context.authority_token:
            raise RuntimeError("Workspace belongs to a retired Context authority generation")

    @property
    def detached(self) -> bool:
        """Return whether this facade is outside durable and live Context lookup."""
        return self._detached

    @workspace_locked
    def activate(self) -> "HeavenBase":
        """Make this workspace active in its owning Context.

        Args:
            None.

        Returns:
            HeavenBase: This workspace, for fluent use.

        Raises:
            RuntimeError: If this workspace is detached from Context lookup.
        """
        if self.detached:
            raise RuntimeError("Detached workspaces cannot be activated")
        self.context.activate_workspace(self.id)
        return self

    @workspace_locked
    def deactivate(self) -> "HeavenBase":
        """Clear this workspace when it is the owning Context's active choice.

        Args:
            None.

        Returns:
            HeavenBase: This workspace, for fluent use.

        Raises:
            RuntimeError: If this workspace is detached from Context lookup.
            ValueError: If a different workspace is currently active.
        """
        if self.detached:
            raise RuntimeError("Detached workspaces cannot participate in active workspace selection")
        active_id = self.context.active_workspace_id
        if active_id is not None and active_id != self.id:
            raise ValueError(f"Cannot deactivate workspace {self.id!r}; active workspace is {active_id!r}")
        if active_id == self.id:
            self.context.deactivate_workspace()
        return self

    @workspace_locked
    def sign(
        self,
        *,
        owner: Optional[str] = None,
        creator: Optional[str] = None,
        auth: Optional[Mapping[str, Any]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> "HeavenBase":
        """Set default Catalog ownership fields for future entity writes.

        Args:
            owner: Default Catalog ``owner`` value. Defaults to ``None``.
            creator: Default Catalog ``creator`` value. Defaults to ``None``.
            auth: Default Catalog authentication/governance metadata. Defaults to ``None``.
            metadata: Default Catalog metadata payload. Defaults to ``None``.

        Returns:
            This workspace, for fluent setup.
        """
        return self._catalog.sign(owner=owner, creator=creator, auth=auth, metadata=metadata)

    @workspace_read_locked
    def catalog_signature(self) -> Dict[str, Any]:
        """Return the default Catalog signature applied to future writes.

        Args:
            None.

        Returns:
            Dict[str, Any]: Default Catalog ownership and metadata fields.
        """
        return self._catalog.signature_dict()

    @workspace_read_locked
    def to_manifest(self, *, include_system: bool = False) -> Dict[str, Any]:
        """Return a workspace construction and schema manifest.

        Args:
            include_system (bool): Whether to include the schemas of the
                entities that enabled extensions register, setup-hook entities
                included. Without them, importing the manifest gets them from
                extension activation, which defines them. Requested extension
                roots are unchanged. Defaults to ``False``.

        Returns:
            Dict[str, Any]: JSON-compatible workspace manifest. Row data is not
            included.
        """
        return workspace_manifest(self, include_system=include_system)

    @classmethod
    def from_manifest(
        cls,
        data: Mapping[str, Any] | WorkspaceManifest,
        *,
        context: Context | None = None,
        backends: Optional[Mapping[str, Any]] = None,
        preset: Optional[str] = None,
        computes: Optional[Mapping[str, Any]] = None,
        query_computes: Optional[Mapping[str, Any]] = None,
        allow_replace: bool = False,
        detached: bool = False,
    ) -> "HeavenBase":
        """Construct a workspace from a construction/schema manifest.

        Args:
            data (Mapping[str, Any] | WorkspaceManifest): Manifest mapping or
                parsed ``WorkspaceManifest``.
            context: Machine composition root that owns the reconstructed
                workspace. Defaults to the process Context.
            backends: Optional complete Backend config or live-instance
                replacement. Runtime replacements must match the recorded
                Backend name-to-identifier signature.
            preset: Optional complete preset replacement. Explicit Backends
                take precedence.
            computes (Mapping[str, Any] | None): Optional compute function
                registry for entity definitions with compute hooks.
            query_computes (Mapping[str, Any] | None): Optional query-compute
                function registry for entity definitions with query hooks.
            allow_replace (bool): Whether entity registration may replace an
                existing schema. Defaults to ``False``.
            detached: Whether the returned workspace is caller-owned and
                absent from durable and live Context lookup. Defaults to
                ``False``.

        Returns:
            HeavenBase: Live workspace reconstructed from the manifest.
        """
        selected = context_or_default(context)
        with selected._lock:
            selected._need_open()
            return restore_workspace_manifest(
                cls,
                data,
                context=selected,
                backends=backends,
                preset=preset,
                computes=computes,
                query_computes=query_computes,
                allow_replace=allow_replace,
                detached=detached,
            )

    @workspace_locked
    def register(
        self,
        entity: Any,
        *,
        computes: Optional[Mapping[str, Any]] = None,
        query_computes: Optional[Mapping[str, Any]] = None,
        allow_replace: bool = False,
        catalog: Optional[Literal["full", "identity", "none"]] = None,
        gram: Optional[bool] = None,
    ) -> type:
        """Register an entity schema in this workspace.

        ``catalog`` and ``gram`` choose how much per-row maintenance the
        entity's writes carry; see ``EntityOptions``. The choice is stored
        with the registration in ``entity_options``, exported with the
        entity definition in ``to_manifest()``, and published in the
        MetaSchema entity row, so a workspace reopened from its manifest
        behaves the same. Re-registering with a different value is rejected
        like a structural change unless ``allow_replace`` is true; a changed
        ``catalog`` then rewrites or deletes the entity's stored Catalog rows
        in bounded pages, and a changed ``gram`` re-places its
        ``Array[ShortText]`` fields (a keyword index is dropped or rebuilt from
        stored rows). Without a live registration (a fresh process on an
        existing store) the registration the stored MetaSchema entity row
        recorded is the one being changed: ``None`` inherits it, and a
        different ``catalog`` or ``gram`` is rejected the same way unless
        ``allow_replace`` is true, in which case the stored rows are
        reconciled from it; a row written by a release without options records
        the defaults, and no row means nothing is stored. The rewrite runs
        under whichever registration carries full Catalog maintenance: leaving
        ``"full"`` rewrites before the registration switches, entering it
        switches first. A page that fails raises ``RuntimeError`` and rolls
        nothing back; the entity keeps that registration, ``audit()`` lists
        the rows that do not match it, and ``repair()`` rewrites them.
        Re-running the same registration completes a change that failed
        before the switch.

        Args:
            entity (Any): ``Entity`` class, registered entity identifier,
                instance whose class should be registered, or JSON-compatible
                entity definition. Supplying an instance uses its class
                definition only and never writes that instance as a row. A
                definition may state ``catalog`` and ``gram`` itself.
            computes (Mapping[str, Any] | None): Optional computed-field hooks keyed by field name. Defaults to ``None``.
            query_computes (Mapping[str, Any] | None): Optional query-time compute hooks keyed by schema reference. Defaults to ``None``.
            allow_replace (bool): Whether an existing schema with the same entity identifier may be replaced.
                A replacement does not move stored rows: a route it adds holds only the rows written after
                it, so its miss does not show that a row is absent while the entity keeps an older route.
                A route added back is read as it was stored. Defaults to ``False``.
            catalog (str | None): Catalog projection mode for the entity's rows.
                Supported values:
                - ``"full"``: project, sign, and derive a Catalog row per write, with
                  keyword index rows for its text. The default for a new registration.
                - ``"identity"``: publish only ``target_id``, ``target_entity``, ``ws``,
                  lifecycle fields, and field defaults; no projection, no Catalog
                  derives, no workspace signature (``sign(...)`` values are not applied),
                  and no keyword index rows. The stored Catalog row is read by id so its
                  ``created_at``, ``active``, and ``visible`` carry over exactly as under
                  ``"full"``; nothing else is merged from it.
                - ``"none"``: publish no Catalog row. Such rows are absent from Catalog
                  discovery (``ws.query(hb.Catalog)``, ``ws.search``, ``Query.available()``)
                  and from Catalog audit; reads by ``object_id`` still work, with or
                  without ``entity=...``, because they read the entity tables directly.
                ``None`` keeps the definition's ``catalog`` key, else the value of the live
                registration or of the registration the store's MetaSchema entity row recorded,
                else ``"full"``. Defaults to ``None``.
            gram (bool | None): Whether the entity's own ``Array[ShortText]`` fields keep a
                sparse-GRAM keyword index. ``False`` stores them inline without index
                maintenance and makes ``contained_in`` on them fail with ``LookupError``
                instead of scanning. ``None`` keeps the definition's ``gram`` key, else the
                value of the live or stored registration, else ``True``. Defaults to ``None``.

        Returns:
            type: Registered ``Entity`` class.

        Raises:
            ValueError: If the entity identifier is already registered with a different structure or different
                ``catalog``/``gram`` options, or the store's MetaSchema entity row records different options, and
                ``allow_replace`` is false; if ``catalog`` is not a supported mode;
                if an explicit argument disagrees with the definition's key; if ``gram=False`` meets an explicit
                ``SparseGramIndex`` placement; or if a change from ``catalog="none"`` meets a stored ``object_id``
                longer than a Catalog row's ``target_id`` holds, before anything is switched. Also if an enabled
                extension defines the entity and the definition differs from the registered one in structure,
                options or metadata (name, descriptions, tags or examples), whatever ``allow_replace`` is: the
                extension registers its own definition each time the workspace opens. The extension's own
                definition is accepted and leaves the extension's class registered.
            TypeError: If ``gram`` is not a boolean.
            RuntimeError: If a changed ``catalog`` cannot rewrite every stored Catalog row; the message names the
                registration the entity keeps and the original error is chained.
        """
        return self._schema.register(entity, computes=computes, query_computes=query_computes, allow_replace=allow_replace, catalog=catalog, gram=gram)

    @workspace_locked
    def alter_entity(
        self, entity: Any, *, computes: Optional[Mapping[str, Any]] = None, query_computes: Optional[Mapping[str, Any]] = None, confirm: str = ""
    ) -> type:
        """Alter an existing entity with guarded additive schema changes.

        Args:
            entity (Any): Updated entity class or JSON-compatible entity definition. The entity keeps its registered
                ``catalog`` and ``gram`` options. Omit either key or pass JSON null to inherit its registered value.
            computes (Mapping[str, Any] | None): Optional computed-field hooks keyed by field name. Defaults to ``None``.
            query_computes (Mapping[str, Any] | None): Optional query-time compute hooks keyed by schema reference. Defaults to ``None``.
            confirm (str): Entity identifier confirmation required for destructive-looking schema changes.

        Returns:
            type: Re-registered entity class.

        Raises:
            KeyError: If the entity is not already registered.
            ValueError: If ``confirm`` does not match, fields are removed, an existing field's type or storage changes,
                the Catalog mode is unsupported, the definition changes a registered option, or an enabled
                extension defines the entity and the definition differs from it (the extension's own definition
                changes nothing and is accepted).
            TypeError: If a JSON definition's ``gram`` value is neither a boolean nor null.
        """
        return self._schema.alter(entity, computes=computes, query_computes=query_computes, confirm=confirm)

    @workspace_locked
    def describe(
        self, entity: Any, *, desc: Optional[str] = None, tags: Optional[Sequence[str]] = None, fields: Optional[Mapping[str, Mapping[str, Any]]] = None
    ) -> type:
        """Apply metadata-only descriptions, tags, and examples to a registered entity.

        Args:
            entity (Any): Present Entity class, registered entity identifier, or instance whose class is present.
            desc (str | None): Entity description. Defaults to ``None``.
            tags (Sequence[str] | None): Entity-level discovery tags. Defaults to ``None``.
            fields (Mapping[str, Mapping[str, Any]] | None): Per-field metadata patches. Defaults to ``None``.

        Returns:
            type: Registered entity class with updated metadata.

        Raises:
            ValueError: If the entity is a fixed workspace Entity, or if an enabled extension defines it and the
                call would change its metadata (a call that changes nothing is accepted).
        """
        return self._schema.describe(entity, desc=desc, tags=tags, fields=fields)

    @workspace_locked
    def upsert(self, entity: Any, row: Optional[Dict[str, Any]] = None, **values: Any) -> str:
        """Insert or replace one row of an Entity class already present in this workspace.

        Args:
            entity: Present Entity class, registered identifier, or instance
                whose class is already present.
            row: Optional row mapping when ``entity`` is not already an instance.
            **values: Extra field values merged into the row. Two keyword names
                are also write flags, each a bool. ``overwrite`` (default
                ``True``): when false, reject an existing row with the same
                ``object_id``. ``atomic`` (default ``False``): commit the row and
                its Catalog row through one write call, so both land or neither
                does; every routed write must then land on one backend that
                claims the ``atomic-upsert`` tag, such as a transactional SQL
                backend or ``inmem``. A definition-shaped MetaSchema row registers
                its Entity class before that call, and a failed write does not undo
                the registration. Without it the entity row commits before the
                Catalog row in separate backend calls. When the row's Entity
                class declares a field named ``overwrite`` or ``atomic``, that
                keyword is the field's value instead, and such an entity takes
                the flag through ``upsert_many``.

        Returns:
            The normalized target ``object_id``.

        Raises:
            TypeError: If ``overwrite`` or ``atomic`` is read as a flag and is not a bool.
            KeyError: If the row's Entity class is not present in the workspace.
            ValueError: If ``overwrite`` is false and the ``object_id`` already exists or the
                entity is stored only on Pinecone and no route returns the ``object_id`` (a fetch
                miss does not show that it is new), a GraphEdge endpoint is missing or duplicated
                within its source field, or
                ``atomic`` is requested while the writes span more than one backend or
                land on a backend that does not claim ``atomic-upsert``.
        """
        return self._crud.upsert(entity, row, values)

    @workspace_locked
    def upsert_many(
        self,
        entity: Any,
        rows: Iterable[Any],
        *,
        overwrite: bool = True,
        catalog_mode: Literal["merge", "fresh"] = "merge",
        atomic: bool = False,
    ) -> List[str]:
        """Insert or replace many rows whose Entity classes are already present.

        Args:
            entity: Default present Entity class or registered identifier.
            rows: Entity instances or row mappings. Entity instances may have
                mixed types when every represented class is present.
            overwrite: When false, reject existing rows with matching ``object_id`` values.
            catalog_mode: ``"merge"`` preserves lifecycle fields from existing
                Catalog rows, read by id in one lookup per Catalog route for the
                whole batch, for entities registered with ``catalog="full"`` or
                ``"identity"``. ``"fresh"`` asserts those Catalog rows are new and
                skips that lookup. Defaults to ``"merge"``.
            atomic: Commit the entity rows and their Catalog rows through one
                write call, so they all land or none does. Every routed write
                must land on one backend that claims the ``atomic-upsert`` tag,
                such as a transactional SQL backend or ``inmem``. Defaults to
                ``False``, which commits entity rows before Catalog rows in
                separate backend calls.

        Returns:
            Normalized ``object_id`` values in input order.

        Raises:
            TypeError: If ``atomic`` is not a bool.
            KeyError: If any row's Entity class is not present in the workspace.
            ValueError: If the batch contains duplicate ``object_id`` values or ``overwrite`` is false
                and any target row already exists, or belongs to an entity stored only on Pinecone and
                no route returns its ``object_id``, or a GraphEdge endpoint is missing or duplicated
                within its source field, or ``catalog_mode`` is unsupported, or ``atomic`` is requested
                while the writes span more than one backend or land on a backend that does not claim
                ``atomic-upsert``.

        Notes:
            Without ``atomic``, entity backend writes complete before Catalog rows are
            published, and if an entity backend fails no Catalog row is written for that
            failed batch. With ``atomic`` both land or neither does, which is why it is
            refused on a backend that does not claim ``atomic-upsert``; a definition-shaped
            MetaSchema row still registers its Entity class before the write, and a failed
            write does not undo that registration. Entities registered with
            ``catalog="none"`` publish no Catalog rows at all.
        """
        return self._crud.upsert_many(entity, rows, overwrite=overwrite, catalog_mode=catalog_mode, atomic=atomic)

    @workspace_read_locked
    def get(self, object_id: Any, entity: Any = None, *, unique: bool = True, columns: Sequence[str] | None = None) -> Optional[Dict[str, Any]]:
        """Return one logical row by ``object_id``.

        With ``entity`` the read goes straight to the backends routed by that
        entity's storage plan, using batched lookups and the workspace row
        codec; Catalog and the other present entities are not consulted.
        Without it every present entity is read and the ``object_id`` must
        belong to exactly one of them; an id that one entity's ``object_id``
        domain rejects, such as one longer than its ``max_chars``, is simply
        absent from that entity.

        Args:
            object_id (Any): ``object_id`` to fetch.
            entity (Any): Optional present Entity class, registered identifier, or instance whose class is present.
                Defaults to ``None``.
            unique (bool): Whether routed fragments of the matched row that disagree on a stored field raise
                (``True``) or merge with the first route's value kept (``False``). It does not relax the
                cross-entity check of an entity-less read. Defaults to ``True``.
            columns (Sequence[str] | None): Optional declared field names to read. The row then carries ``object_id``
                plus those fields only. ``SQLBackend.get_many`` selects the fields in storage. HeavenBase trims fragments
                before decoding when a backend method omits the ``columns`` parameter. Requires ``entity``.
                Defaults to ``None`` (every field).

        Returns:
            Dict[str, Any] | None: Merged logical row, or ``None`` when no row matches.

        Raises:
            KeyError: If ``entity`` is given and not present in this workspace.
            TypeError: If ``columns`` is a single string rather than a sequence of names, or holds a name that is
                not a string.
            ValueError: If ``entity`` is omitted and the ``object_id`` exists under more than one entity, if
                ``unique`` is true and routed fragments of the row disagree, if ``columns`` is given without
                ``entity``, or if ``columns`` names a field the entity does not declare.

        Examples:
            >>> ws.get(doc_id, entity=Doc, columns=["title", "revision"])
        """
        return self._crud.get(object_id, entity=entity, unique=unique, columns=columns)

    @workspace_read_locked
    def get_many(
        self,
        object_ids: Iterable[Any],
        entity: Any = None,
        *,
        unique: bool = True,
        columns: Sequence[str] | None = None,
    ) -> List[Optional[Dict[str, Any]]]:
        """Return rows for ``object_id`` values in input order.

        ``entity`` and ``unique`` mean exactly what they mean for ``get``: a
        named entity is read directly without Catalog, and ``unique`` governs
        only routed-fragment disagreement inside a row.

        Args:
            object_ids (Iterable[Any]): ``object_id`` values to fetch.
            entity (Any): Optional present Entity class, registered identifier, or instance whose class is present.
                Defaults to ``None``.
            unique (bool): Whether disagreeing routed fragments raise instead of merging first-route-wins.
                Defaults to ``True``.
            columns (Sequence[str] | None): Optional declared field names to read, as for ``get``. Requires ``entity``.
                Defaults to ``None`` (every field).

        Returns:
            List[Dict[str, Any] | None]: Rows in input order, with missing ``object_id`` values preserved as ``None``.

        Raises:
            KeyError: If ``entity`` is given and not present in this workspace.
            TypeError: If ``columns`` is a single string rather than a sequence of names, or holds a name that is
                not a string; checked even when ``object_ids`` is empty.
            ValueError: If ``entity`` is omitted and an ``object_id`` exists under more than one entity, if
                ``unique`` is true and routed fragments of a row disagree, or if ``columns`` is given without
                ``entity`` or names an undeclared field; ``columns`` is checked even when ``object_ids`` is empty.
        """
        return self._crud.get_many(object_ids, entity=entity, unique=unique, columns=columns)

    @workspace_read_locked
    def read_rows(self, entity: Any, object_ids: Iterable[Any]) -> List[Optional[Dict[str, Any]]]:
        """Return decoded logical rows of one present entity, aligned to ``object_ids``.

        This is the entity-first form of ``get_many(object_ids, entity=entity)``:
        the entity is named, so the read goes straight to the backends routed by
        its storage plan with batched ``get_many`` lookups and the workspace row
        codec, and Catalog is never consulted. Routed fragments of one row that
        disagree raise, as with ``unique=True``.

        Args:
            entity (Any): Present Entity class, registered identifier, or instance whose class is present.
            object_ids (Iterable[Any]): ``object_id`` values to read.

        Returns:
            List[Dict[str, Any] | None]: Decoded rows in input order; a missing ``object_id`` yields ``None``.

        Raises:
            KeyError: If the entity is not present in this workspace.
            ValueError: If routed fragments of one row disagree on a stored field.
        """
        return self._crud.read_rows(entity, object_ids)

    def iter_rows(self, entity: Any, *, batch_size: int = 128) -> Iterator[Dict[str, Any]]:
        """Iterate decoded logical rows of one present entity in bounded keyset pages.

        Rows come from the routed backends' keyset ``scan_page`` calls. They
        are merged across routes and decoded as in ``get`` without consulting
        Catalog. Rows follow ascending ``object_id`` order when all routes use
        Python's string ordering. Backend collations can change that order.
        The workspace checks liveness and holds its read lock for each page
        fetch. Workspace and backend locks are released before the rows are
        yielded, so the consumer or another thread may write between pages. The
        iteration follows later writes ahead of the cursor on the routes
        selected when the scan starts. Each page uses the schema current at
        that page, including any newly added fields on those routes. Start a
        new scan to include newly added routes. Closing the iterator early
        ends the scan.

        Args:
            entity (Any): Present Entity class, registered identifier, or instance whose class is present.
            batch_size (int): Positive number of rows read per backend page. Defaults to ``128``.

        Returns:
            Iterator[Dict[str, Any]]: Decoded logical rows from the keyset scan.

        Raises:
            KeyError: If the entity is not present in this workspace.
            ValueError: If ``batch_size`` is not a positive integer.
            RuntimeError: If the workspace's Context authority retires before a page is read.
        """
        with workspace_read_scope(self):
            pages = self._row_access.row_pages(entity, batch_size=batch_size)
        return workspace_read_pages(self, pages)

    @workspace_locked
    def set(self, row: Any, values: Optional[Dict[str, Any]] = None, *, entity: Any = None, overwrite: bool = True) -> str:
        """Patch one row by ``object_id`` and return that ``object_id``.

        Args:
            row: Entity instance, row patch, or ``object_id`` when `values` is provided.
            values: Optional patch values for ``object_id``-first calls.
            entity: Entity class or entity identifier used to disambiguate plain mappings or duplicate ``object_id`` values.
            overwrite: When false, reject an existing target row.

        Returns:
            The patched row's normalized ``object_id``.

        Raises:
            KeyError: If the resolved Entity class is not present in the workspace.
            TypeError: If a plain mapping cannot be resolved to an entity.
            ValueError: If the patch omits ``object_id``, tries to move the row identity, or leaves a
                GraphEdge endpoint missing or duplicated within its source field. A missing Pinecone
                source record also raises before writing, and so does a new ``object_id`` of an
                entity stored only on Pinecone. Inference raises when an unread Pinecone row could
                make the entity ambiguous. Pass ``entity=`` to select the target. Use a full upsert
                to create a row of an entity stored only on Pinecone or update a deliberately
                vectorless row.
        """
        return self._crud.set(row, values, entity=entity, overwrite=overwrite)

    @workspace_locked
    def set_many(self, rows: Iterable[Any], *, entity: Any = None, overwrite: bool = True) -> List[str]:
        """Patch many rows and return ``object_id`` values in input order.

        Args:
            rows: Entity instances or row patch mappings.
            entity: Entity class or entity identifier used for every patch when provided.
            overwrite: When false, reject existing target rows.

        Returns:
            Patched row ``object_id`` values in input order.

        Raises:
            KeyError: If any resolved Entity class is not present in the workspace.
            TypeError: If any plain mapping cannot be resolved to an entity.
            ValueError: If any patch omits ``object_id``, duplicates or moves an ``object_id``, or leaves
                a GraphEdge endpoint missing or duplicated within its source field. A missing Pinecone
                source record also rejects the batch before writing. Entity inference
                rejects the batch when an unread Pinecone row could make it ambiguous.

        Notes:
            Like `upsert_many`, entity backend writes complete before Catalog rows are
            published for the batch.
        """
        return self._crud.set_many(rows, entity=entity, overwrite=overwrite)

    @workspace_locked
    def delete(self, target: Any, *, edges: Literal["schema", "restrict", "detach"] = "schema") -> bool:
        """Delete one row target under its GraphEdge referential policy.

        Args:
            target: Entity instance, ``object_id``, ``(entity, object_id)``, or a mapping with
                ``entity`` and ``object_id``.
            edges: Incoming GraphEdge policy.
                Supported values:
                - `schema`: Use each GraphEdge field's `on_delete` policy.
                - `restrict`: Reject deletion when a surviving source points to the target.
                - `detach`: Remove matching edges before deleting the target.

        Returns:
            True when a route returned the target before deletion, otherwise false. For an
            entity stored only on Pinecone a fetch miss returns false, but the record and its
            Catalog row are still deleted, and incoming GraphEdge references to it are
            restricted or detached as for a present target.

        Raises:
            KeyError: If an explicitly targeted Entity class is not present in the workspace.
            GraphDeleteRestrictedError: If an effective `restrict` policy finds
                surviving incoming references.
            ValueError: If `edges` is unsupported, a Pinecone miss leaves entity inference
                uncertain, or a detach source misses a Pinecone record.
        """
        return self._crud.delete(target, edges=edges)

    @workspace_locked
    def delete_many(self, targets: Iterable[Any], *, edges: Literal["schema", "restrict", "detach"] = "schema") -> List[bool]:
        """Delete many row targets as one GraphEdge referential batch.

        Args:
            targets: Entity instances, ``object_id`` values, ``(entity, object_id)`` tuples, or mappings
                with ``entity`` and ``object_id``.
            edges: Incoming GraphEdge policy. Uses the `schema`, `restrict`, and
                `detach` values documented on `delete`.

        Returns:
            Boolean existence flags aligned with the input targets, each as ``delete`` returns it.

        Raises:
            KeyError: If an explicitly targeted Entity class is not present in the workspace.
            GraphDeleteRestrictedError: If an effective `restrict` policy finds
                surviving incoming references. No entity/Catalog row is mutated.
            ValueError: If `edges` is unsupported, a Pinecone miss leaves entity inference
                uncertain, or a detach source misses a Pinecone record.

        Notes:
            Edges whose source is also in this delete batch disappear with that source and
            do not restrict the batch. Each backend receives one coherent mutation batch;
            independently routed backends do not form a distributed transaction.
        """
        return self._crud.delete_many(targets, edges=edges)

    def _drop_live(self) -> None:
        """Drop backend storage and remove this workspace from live/process registries.

        Connections in use on the workspace's DuckDB files are waited for first, before the workspace lock (and with it
        the Context lock) is taken, so the Context's other writes go on meanwhile.
        """
        self._maintenance.await_quiet_storage()
        self._drop_live_locked()

    @workspace_locked
    def _drop_live_locked(self) -> None:
        """Drop backend storage and remove this workspace from live/process registries, under the workspace lock."""
        self._maintenance.drop_live()

    @workspace_locked
    def release(self) -> None:
        """Remove this facade from process lookup without dropping backend data.

        The object remains usable by its current owner. Release does not create
        persistence: a later :meth:`load` can reconstruct the facade only when
        a persistent ``WorkspaceSpec`` exists. Releasing a stale facade
        leaves any newer same-id owner unchanged. Use :meth:`drop` only when
        backend data should be destroyed.

        Args:
            None.

        Returns:
            None: The live facade is no longer registered for process lookup.
        """
        self._maintenance.release_live()

    @workspace_locked
    def refresh(self, entity: Any = None, backend: Any = None) -> None:
        """Invalidate explicit query candidates and refresh cache-capable backends.

        Args:
            entity: Optional present Entity class, instance, or registered
                identifier. When provided, only that Entity class's backend
                candidate cache is invalidated and backend caches are refreshed
                where the backend supports entity-scoped refresh.
            backend: Optional backend name or backend instance. When omitted, every registered backend with a ``refresh``
                method is called.

        Returns:
            None.

        Warnings:
            Explicit near candidate reuse is also invalidated, including prepared
            vector matrices. External writers must call refresh or change the
            near cache version before expecting fresh results.
            ``refresh`` is a cache invalidation hook, not a transaction boundary. Persistent SQL backends do not need it for
            ordinary reads; file backends use it to discard in-memory snapshots after external file changes or manual edits.

        Examples:
            ``ws.refresh(entity=Note)`` clears cache state for one entity. ``ws.refresh(backend="store")`` refreshes one backend.
        """
        self._maintenance.refresh(entity=entity, backend=backend)

    @workspace_read_locked
    def exists(self, object_id: Any, entity: Any = None, *, unique: bool = True) -> bool:
        """Return whether an ``object_id`` exists.

        Args:
            object_id (Any): Row identity to test.
            entity (Any | None): Optional entity class or identifier used to
                disambiguate duplicate ``object_id`` values. Defaults to
                ``None``.
            unique (bool): Whether duplicate matches across entities should
                raise instead of returning the first match. Defaults to
                ``True``.

        Returns:
            bool: ``True`` when a matching row exists, otherwise ``False``.

        Raises:
            ValueError: If ``unique`` is true and the ``object_id`` matches
                more than one entity.
        """
        return self._crud.exists(object_id, entity=entity, unique=unique)

    @workspace_read_locked
    def exists_many(self, object_ids: Iterable[Any], entity: Any = None, *, unique: bool = True) -> List[bool]:
        """Return existence flags for ``object_id`` values in input order.

        Args:
            object_ids (Iterable[Any]): Row identities to test.
            entity (Any | None): Optional entity class or identifier used to
                disambiguate duplicate ``object_id`` values. Defaults to
                ``None``.
            unique (bool): Whether duplicate matches across entities should
                raise. Defaults to ``True``.

        Returns:
            List[bool]: Existence flags aligned with ``object_ids``.

        Raises:
            ValueError: If ``unique`` is true and any ``object_id`` matches
                more than one entity.
        """
        return self._crud.exists_many(object_ids, entity=entity, unique=unique)

    @workspace_read_locked
    def count(self, entity: Any) -> int:
        """Return the number of logical rows for one present Entity class.

        Args:
            entity (Any): Present Entity class, registered identifier, or
                instance whose class is present.

        Returns:
            int: Largest row count reported by the entity's routed backends.
            This is the logical row count whenever every route agrees, which is
            the steady state. A backend whose own count is eventually
            consistent can report a stale value while it catches up, and taking
            the maximum keeps the answer from under-reporting during that
            window rather than hiding the disagreement.

        Raises:
            KeyError: If the entity is not present in the workspace.
        """
        return self._crud.count(entity)

    @workspace_read_locked
    def ids(self, entity: Any = None) -> List[Any]:
        """Return ``object_id`` values for one entity, or the sole user entity in a small workspace.

        Args:
            entity (Any | None): Present Entity class, registered identifier,
                or instance whose class is present. When omitted in a small
                workspace with one user entity, that entity is selected.
                Defaults to ``None``.

        Returns:
            List[Any]: ``object_id`` values in workspace order.

        Raises:
            KeyError: If the resolved entity is not present.
            ValueError: If ``entity`` is omitted and the workspace has zero or
                multiple user entities.
        """
        return self._row_access.ids(entity)

    @workspace_read_locked
    def rows(self, entity: Any = None) -> List[Dict[str, Any]]:
        """Return logical rows for one entity, or the sole user entity in a small workspace.

        Args:
            entity (Any | None): Present Entity class, registered identifier,
                or instance whose class is present. When omitted in a small
                workspace with one user entity, that entity is selected.
                Defaults to ``None``.

        Returns:
            List[Dict[str, Any]]: Logical rows in workspace order.

        Raises:
            KeyError: If the resolved entity is not present.
            ValueError: If ``entity`` is omitted and the workspace has zero or
                multiple user entities.
        """
        return self._row_access.rows(entity)

    @workspace_read_locked
    def audit(self) -> Dict[str, Any]:
        """Audit system row consistency without mutating workspace data.

        The Catalog audit compares rows owned by this workspace with the rows
        its current entity registrations expect. It skips keys occupied by
        rows without this workspace's ownership. A source row without a
        Catalog row is ``missing``, a Catalog row without a source (including
        each owned row of an entity registered with ``catalog="none"``) is
        ``dangling``, and a row in the other mode's form is ``mismatched``.
        On a SQL Catalog route that keeps a sparse-GRAM keyword index, the
        keyword and posting rows of every Catalog key in scope are compared
        with the rows its stored Catalog row indexes, and a key whose rows
        differ, or that keeps keyword rows without a stored row, is listed
        under ``stale_keywords``.

        Args:
            None.

        Returns:
            Dict[str, Any]: Report keyed by system surface. The current report
            contains a ``catalog`` entry with ``ok``, ``missing``,
            ``dangling``, ``mismatched`` and ``stale_keywords`` row lists and
            ``checked`` counts.
        """
        return self._maintenance.audit()

    @workspace_locked
    def repair(self, *, dry_run: bool = False) -> Dict[str, Any]:
        """Explicitly repair system rows from authoritative entity rows.

        Repair applies the current registration to the Catalog rows that
        ``audit()`` reports, and writes a row listed under ``stale_keywords``
        again with keyword maintenance. A key acquired by another workspace before its
        mutation is skipped. Repair also finishes a Catalog mode change
        whose rewrite failed part-way.

        Args:
            dry_run: When true, return the repair plan without writing or
                deleting rows.

        Returns:
            dict: Counts and before/after Catalog audit reports.

        Raises:
            NotImplementedError: If a Catalog route that needs changes cannot
                guard ownership or does not store ``ws``.
        """
        return self._maintenance.repair(dry_run=dry_run)

    @workspace_read_locked
    def items(self, entity: Any = None) -> List[tuple[Any, Dict[str, Any]]]:
        """Return ``(object_id, row)`` pairs for one entity.

        Args:
            entity (Any | None): Present Entity class, registered identifier,
                or instance whose class is present. When omitted in a small
                workspace with one user entity, that entity is selected.
                Defaults to ``None``.

        Returns:
            List[tuple[Any, Dict[str, Any]]]: ``(object_id, row)`` pairs in
            workspace order.

        Raises:
            KeyError: If the resolved entity is not present.
            ValueError: If ``entity`` is omitted and the workspace has zero or
                multiple user entities.
        """
        return self._row_access.items(entity)

    @workspace_read_locked
    def __getitem__(self, key: Any) -> Dict[str, Any]:
        return self._row_access.get_item(key)

    @workspace_locked
    def __setitem__(self, key: Any, value: Any) -> None:
        self._row_access.set_item(key, value)

    @workspace_locked
    def __delitem__(self, key: Any) -> None:
        self._row_access.delete_item(key)

    @workspace_read_locked
    def __contains__(self, key: Any) -> bool:
        return self._row_access.contains_item(key)

    @workspace_read_locked
    def __iter__(self):
        return self._row_access.iter_ids()

    @workspace_read_locked
    def __len__(self) -> int:
        return self._row_access.len_user_entity()

    @workspace_read_locked
    def query(self, entity: Any) -> "Query":
        """Return an empty immutable query for a registered entity.

        Args:
            entity (Any): Registered entity class, identifier, or instance.

        Returns:
            Query: Immutable query value. Every transformation returns another
                query and leaves this value unchanged.

        Examples:
            >>> ws.query(Product).where(Product.price < 100).select("name", "price").execute().rows()
        """
        return self._query.query(entity)

    @workspace_read_locked
    def traverse(self, field_ref: Any, start_id: Any, *, min_hops: int = 1, max_hops: int = 1, direction: str = "out") -> List[Dict[str, Any]]:
        """Return the records reachable from ``start_id`` over a GraphEdge field.

        Args:
            field_ref (Any): A GraphEdge field reference, e.g. ``Doc.cites``.
            start_id (Any): The ``object_id`` to traverse from.
            min_hops (int): Smallest hop depth to include. Defaults to 1.
            max_hops (int): Largest hop depth to include. Defaults to 1 (direct neighbours).
            direction (str): ``"out"`` (follow edges), ``"in"`` (reverse), or ``"any"``.

        Returns:
            List[Dict[str, Any]]: The reached target records, deduplicated and ordered by object_id.

        Examples:
            >>> ws.traverse(Doc.cites, "d1", max_hops=2)
        """
        return self._graph_service.traverse(field_ref, start_id, min_hops=min_hops, max_hops=max_hops, direction=direction)

    @workspace_read_locked
    def shortest_path(self, field_ref: Any, start_id: Any, end_id: Any, *, max_hops: int = 10, direction: str = "out") -> Optional[List[tuple[str, str]]]:
        """Return the shortest path from ``start_id`` to ``end_id`` over a GraphEdge field, or None if unreachable.

        Args:
            field_ref (Any): A GraphEdge field reference, e.g. ``Doc.cites``.
            start_id (Any): Path start ``object_id``.
            end_id (Any): Path end ``object_id``.
            max_hops (int): Maximum path length to search. Defaults to 10.
            direction (str): ``"out"`` (follow edges), ``"in"`` (reverse), or ``"any"``. With ``"any"`` the
                path is undirected and may traverse edges against their stored direction, so it is not
                necessarily a valid directed chain.

        Returns:
            Optional[List[tuple[str, str]]]: The ``(entity_id, object_id)`` sequence from start to end
            (inclusive) — pairs so a cross-entity path stays resolvable — or None.
        """
        return self._graph_service.shortest_path(field_ref, start_id, end_id, max_hops=max_hops, direction=direction)

    @workspace_read_locked
    def check_graph(self, field_ref: Any = None) -> List[Dict[str, str]]:
        """Report stored GraphEdge values whose target object no longer exists.

        Normal workspace writes validate endpoints, and a successful workspace delete either restricts
        or detaches incoming edges according to its policy. This on-demand scan is therefore a recovery
        tool for legacy data, direct backend mutations, or mergeable dangling rows after an apply error.
        It audits one field when ``field_ref`` is provided and every registered GraphEdge field otherwise.
        Every physical owner of a selected field is scanned exhaustively; the audit fails closed when a
        required owner cannot scan or routed replicas cannot be merged. It does not choose between
        conflicting physical replicas or replay a failed mutation intent.

        Args:
            field_ref (Any): Optional GraphEdge field reference to audit. When omitted,
                audits every registered GraphEdge field.

        Returns:
            List[Dict[str, str]]: One ``{entity, field, source, target}`` record per dangling edge.
        """
        return self._graph_service.check_graph(field_ref)

    @workspace_locked
    def scrub_dangling(self, field_ref: Any = None, *, confirm: bool = False) -> Dict[str, Any]:
        """Repair stored GraphEdge values whose target object is missing.

        The operation is a dry run unless ``confirm=True``. Confirmation rewrites affected source rows
        through the normal workspace mutation path after removing dangling entries. When ``field_ref``
        is provided, it selects source rows through that field; repair removes every dangling GraphEdge
        on each selected row so the replacement remains strongly valid. This repairs mergeable logical
        dangling-reference reports; it neither resolves conflicting replicas nor audits arbitrary native
        projection corruption.

        Args:
            field_ref (Any): Optional GraphEdge field reference selecting source rows. When omitted,
                scans every registered GraphEdge field.
            confirm (bool): Whether to apply the repair. ``False`` returns a dry-run report;
                ``True`` rewrites affected source rows.

        Returns:
            Dict[str, Any]: ``{dangling, removed}`` — the report plus how many edges were removed (0 on a dry run).

        Raises:
            ValueError: If a confirmed repair misses a Pinecone source record or cannot
                establish that a Pinecone-routed target is absent.
        """
        return self._graph_service.scrub_dangling(field_ref, confirm=confirm)

    @workspace_read_locked
    def query_json(self, entity: Any, spec: Mapping[str, Any]) -> "Query":
        """Return an immutable query loaded from a JSON query spec.

        Args:
            entity (Any): Registered entity class, identifier, or instance.
            spec (Mapping[str, Any]): Query object with optional `filter`,
                `available`, `near`, `traverse`, `select`, `group_by`,
                `aggregate`, `having`, `order`, `offset`, and `limit` keys.

        Returns:
            Query: Immutable query value populated from the JSON spec.

        Raises:
            TypeError: If the query or a clause has the wrong JSON shape.
            ValueError: If a key is unsupported or a required clause key is missing.

        Examples:
            ```python
            ws.query_json(
                Sale,
                {
                    "group_by": ["region"],
                    "aggregate": {
                        "customers": {"count": {"field": "customer_id", "distinct": True}},
                        "p95": {"percentile": {"field": "amount", "q": 0.95}},
                    },
                    "having": {"p95": {"$gte": 100}},
                },
            )
            ```
        """
        return self._query.query_json(entity, spec)

    @workspace_read_locked
    def search(
        self,
        card_text: str,
        *,
        limit: Optional[int] = None,
        target_entities: Optional[Iterable[str]] = None,
    ) -> "ResultFrame":
        """Search Catalog and MetaSchema cards by free text.

        Args:
            card_text: Free-text query matched against Catalog object cards and MetaSchema structure cards.
            limit: Maximum number of result cards. Defaults to ``heavenbase.search.default_limit``.
            target_entities: Optional Catalog target-entity allowlist. When
                provided, structural MetaSchema cards are excluded and only
                object cards for these entities are ranked.

        Returns:
            ResultFrame: Search cards with source row identity, target pointers, card text, and heuristic score.

        Raises:
            ValueError: If ``card_text`` is empty or ``limit`` is outside configured bounds.
        """
        return self._query.search(card_text, limit=limit, target_entities=target_entities)

    @workspace_read_locked
    def execute(self, query: "QuerySpec") -> "ResultFrame":
        """Execute a normalized query request.

        Args:
            query (QuerySpec): Immutable request exposed by `Query.spec`.

        Returns:
            ResultFrame: Detached backend-merged result container. Explicitly
                export it through `rows()`, an interoperability conversion, or
                `scalar()` rather than relying on implicit coercion.
        """
        return self._query.execute(query)

    @workspace_read_locked
    def explain(self, query: "QuerySpec") -> Dict[str, Any]:
        """Return storage, handler, and fallback diagnostics for a query.

        Args:
            query (QuerySpec): Normalized query request to inspect.

        Returns:
            Dict[str, Any]: Explanation containing storage plan, per-step handler decisions, backend grouping, and fallback notes.
        """
        return self._query.explain(query)

    @workspace_read_locked
    def health(self) -> Dict[str, Any]:
        """Return lightweight workspace health and backend metadata.

        Args:
            None.

        Returns:
            Dict[str, Any]: Workspace id, registered entities, and backend
            metadata payloads.
        """
        return self._maintenance.health()

    @workspace_locked
    def to_mcp(
        self,
        *,
        name: Optional[str] = None,
        backend: Optional[Any] = None,
        description: Optional[str] = None,
        profile: Optional[str] = None,
        tools: Optional[List[str]] = None,
    ) -> Any:
        """Return a HeavenBase-native Toolkit for this workspace or one of its backends.

        Args:
            name (Optional[str]): Toolkit name. Defaults to a workspace-derived name.
            backend (Optional[Any]): Optional backend name or backend instance to expose as a bounded Toolkit. Defaults to ``None``.
            description (Optional[str]): Toolkit description. Defaults to a workspace summary.
            profile (Optional[str]): Optional configured tool profile such as ``"full"`` or ``"agent"``. Defaults to configured MCP profile.
            tools (Optional[List[str]]): Explicit tool allowlist. When supplied, this overrides ``profile``.

        Returns:
            Any: Runtime Toolkit bound to this workspace or backend object.

        Warnings:
            Workspace-bound Toolkits close over live objects and are runtime-only; they cannot be persisted as durable Capsule Toolkits.
        """
        return self._mcp.to_mcp(name=name, backend=backend, description=description, profile=profile, tools=tools)

    @workspace_locked
    def to_mcp_json(
        self,
        *,
        name: Optional[str] = None,
        backend: Optional[Any] = None,
        description: Optional[str] = None,
        profile: Optional[str] = None,
        tools: Optional[List[str]] = None,
        transport: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
    ) -> str:
        """Return MCP client config JSON for this workspace or one of its backends.

        Args:
            name (Optional[str]): Toolkit/server name. Defaults to a workspace-derived name.
            backend (Optional[Any]): Optional backend name or backend instance to expose. Defaults to ``None``.
            description (Optional[str]): Toolkit description. Defaults to a workspace summary.
            profile (Optional[str]): Optional configured tool profile such as ``"full"`` or ``"agent"``. Defaults to configured MCP profile.
            tools (Optional[List[str]]): Explicit tool allowlist. When supplied, this overrides ``profile``.
            transport (Optional[str]): MCP transport. Defaults to configured MCP transport.
            host (Optional[str]): HTTP host. Defaults to the Toolkit MCP JSON default.
            port (Optional[int]): HTTP port. Defaults to the Toolkit MCP JSON default.

        Returns:
            str: JSON client configuration for an MCP client.
        """
        return self._mcp.to_mcp_json(
            name=name,
            backend=backend,
            description=description,
            profile=profile,
            tools=tools,
            transport=transport,
            host=host,
            port=port,
        )

    @workspace_locked
    def to_skill(
        self,
        *,
        profile: str,
        name: Optional[str] = None,
        url: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
    ) -> Any:
        """Return an installable skill for one explicit workspace MCP profile.

        Args:
            profile: MCP profile the skill is allowed to expose.
            name: Optional kebab-case skill name. Defaults to the workspace id.
            url: Optional credential-free streamable-HTTP MCP URL. When omitted,
                the workspace MCP host and port produce a local URL.
            host: Optional local MCP host override used when ``url`` is omitted.
            port: Optional local MCP port override used when ``url`` is omitted.

        Returns:
            Any: Detached :class:`WorkspaceSkill`; call ``save`` or ``install``
            to write its folder.

        Raises:
            ValueError: If the profile, skill name, URL, or port is invalid.
            RuntimeError: If Toolkit support or the selected profile is inactive.

        Notes:
            The generated artifact contains workspace/profile metadata and a
            public MCP URL only. It never serializes workspace construction or
            source-database credentials.
        """
        return self._mcp.to_skill(profile=profile, name=name, url=url, host=host, port=port)

    def serve(
        self,
        *,
        name: Optional[str] = None,
        backend: Optional[Any] = None,
        description: Optional[str] = None,
        profile: Optional[str] = None,
        tools: Optional[List[str]] = None,
        transport: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        **server_kwargs: Any,
    ) -> Any:
        """Serve this workspace or one backend as an MCP server.

        Args:
            name (Optional[str]): Toolkit/server name. Defaults to a workspace-derived name.
            backend (Optional[Any]): Optional backend name or backend instance to expose. Defaults to ``None``.
            description (Optional[str]): Toolkit description. Defaults to a workspace summary.
            profile (Optional[str]): Optional configured tool profile such as ``"full"`` or ``"agent"``. Defaults to configured MCP profile.
            tools (Optional[List[str]]): Explicit tool allowlist. When supplied, this overrides ``profile``.
            transport (Optional[str]): MCP transport. Defaults to ``heavenbase.mcp.transport``.
            host (Optional[str]): HTTP host. Defaults to ``heavenbase.mcp.host``.
            port (Optional[int]): HTTP port. Defaults to ``heavenbase.mcp.port``.
            **server_kwargs (Any): Additional FastMCP server options.

        Returns:
            Any: Result returned after the blocking Toolkit server lifecycle completes.

        Notes:
            Toolkit construction and server-option resolution run under the
            workspace lifecycle lock. Serving starts after releasing that lock,
            so blocking servers preserve synchronous failures without deadlocking
            workspace-backed tools.
        """
        from .mcp.support import mcp_host, mcp_port, mcp_transport

        with self._lock:
            self._need_live()
            with _workspace_config_scope(self.context, self.id):
                toolkit = self._mcp.to_mcp(
                    name=name,
                    backend=backend,
                    description=description,
                    profile=profile,
                    tools=tools,
                )
                config = self.context.config
                resolver = self.module_resolver
                transport = mcp_transport(transport, config=config, resolver=resolver)
                host = mcp_host(host, config=config, resolver=resolver)
                port = mcp_port(port, config=config, resolver=resolver)
        return toolkit.serve(
            transport=transport,
            host=host,
            port=port,
            config=config,
            **server_kwargs,
        )

    @workspace_locked
    def enable_extension(self, extension: str) -> Extension:
        """Enable a registered entity extension in this workspace.

        Args:
            extension (str): Registered Extension identifier.

        Returns:
            Extension: Enabled extension descriptor.
        """
        return self._extension_service.enable(extension)

    @workspace_read_locked
    def extensions(self) -> Dict[str, Dict[str, Any]]:
        """Return enabled workspace extension metadata.

        Args:
            None.

        Returns:
            Dict[str, Dict[str, Any]]: Extension metadata keyed by identifier.
        """
        return self._extension_service.to_dict()

    @workspace_read_locked
    def list_entities(self) -> List[str]:
        """Return user-visible entity identifiers in this workspace.

        Args:
            None.

        Returns:
            List[str]: Registered entity identifiers excluding system and hidden
                extension-owned entities.
        """
        return self._extension_service.list_entities()
