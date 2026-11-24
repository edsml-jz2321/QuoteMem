"""Workspace object-id lookup and match resolution."""

from __future__ import annotations

__all__ = [
    "RowMatch",
]

from heavenbase.utils import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, dataclass


@dataclass(frozen=True)
class RowMatch:
    """Describe one resolved object-id lookup result.

    Args:
        entity_cls (type): Registered entity class that owns the matched row.
        object_id (str): Normalized object identifier used for backend lookup.
        row (Dict[str, Any]): Merged logical row returned by storage.
        missing_routes (Tuple[str, ...]): Missing source routes when validation is deferred.
    """

    entity_cls: type
    object_id: str
    row: Dict[str, Any]
    missing_routes: Tuple[str, ...] = ()


class _RowLookupService:
    """Own user-facing object-id normalization, lookup, and existence checks."""

    def __init__(self, workspace: Any):
        self.workspace = workspace

    def normalize_object_id(self, schema: Any, object_id: Any) -> str:
        """Validate and normalize a user-facing ``object_id``."""
        return schema.get(schema.pk_name).dtype.validate(object_id)

    def normalized_ids_by_entity(self, object_ids: Sequence[Any], *, entity_ids: Sequence[str] | None = None) -> Iterator[Tuple[type, List[Optional[str]]]]:
        """Normalize ``object_ids`` one present entity at a time for a lookup without ``entity``.

        Each entity validates an id against its own ``object_id`` domain. An id
        that one entity's domain rejects, for example one longer than that
        entity's ``max_chars``, cannot be stored under that entity, so it is
        absent there rather than an error for the whole lookup. Only an id that
        no present entity accepts is rejected. Validate this before yielding
        any batch for backend lookup.

        Args:
            object_ids (Sequence[Any]): Requested object ids in input order.
            entity_ids (Sequence[str] | None): Present entities to consider, in
                order, such as the entities an MCP profile can see; ``None``
                considers every present entity.

        Yields:
            Tuple[type, List[Optional[str]]]: Each present entity with the
            ids normalized by its schema, aligned to ``object_ids``; ``None``
            marks an id that entity cannot hold.

        Raises:
            ValueError: If no considered entity accepts an id; the error is the one
                raised by the considered entity whose ``object_id`` domain allows
                the most characters, the first such entity on a tie.
            TypeError: If no considered entity accepts an id and that entity
                rejected it by type.
        """
        entities = self.workspace.entities
        candidates = entities.values() if entity_ids is None else [entities[entity_id] for entity_id in entity_ids]
        schemas = [(entity_cls, entity_cls.schema()) for entity_cls in candidates]

        def domain_width(schema: Any) -> int:
            max_chars = getattr(schema.get(schema.pk_name).dtype, "max_chars", None)
            return max_chars if isinstance(max_chars, int) else -1

        # Check the widest domains first so the first error names the binding limit.
        domains = sorted((schema for _, schema in schemas), key=domain_width, reverse=True)
        for object_id in object_ids:
            error: Exception | None = None
            for schema in domains:
                try:
                    self.normalize_object_id(schema, object_id)
                except (TypeError, ValueError) as exc:
                    if error is None:
                        error = exc
                else:
                    break
            else:
                if error is not None:
                    raise error

        for entity_cls, schema in schemas:
            normalized_ids: List[Optional[str]] = []
            for object_id in object_ids:
                try:
                    normalized_ids.append(self.normalize_object_id(schema, object_id))
                except (TypeError, ValueError):
                    normalized_ids.append(None)
            yield entity_cls, normalized_ids

    def id_match(
        self, object_id: Any, entity: Any = None, *, unique: bool = True, columns: Sequence[str] | None = None, for_mutation: bool = False
    ) -> Optional[RowMatch]:
        """Return the registered entity and merged row for one ``object_id``."""
        return self.id_matches_many([object_id], entity=entity, unique=unique, columns=columns, for_mutation=for_mutation)[0]

    def id_matches_many(
        self,
        object_ids: Iterable[Any],
        entity: Any = None,
        *,
        unique: bool = True,
        columns: Sequence[str] | None = None,
        for_mutation: bool = False,
    ) -> List[Optional[RowMatch]]:
        """Return registered entity matches for many object_ids using backend batch lookups.

        With ``entity`` the read is confined to that entity through
        ``entity_matches_many``. Without it every present entity, Catalog
        included, is read, and an ``object_id`` found under more than one
        entity raises whatever ``unique`` says; ``unique`` still governs
        routed-fragment disagreement inside each entity. An id that an
        entity's ``object_id`` domain rejects is absent from that entity; see
        ``normalized_ids_by_entity``.

        ``columns`` selects the logical fields to read and needs ``entity``: the
        selection is validated against that schema, whatever ``object_ids``
        holds. Each route's fragment is restricted to the selection before decoding.
        With ``for_mutation``, every entity that accepts an id but returned no
        row must show the row is absent, whether or not another entity matched:
        a mutation must neither pick one entity while another may hold the id
        nor report an id absent that an inconclusive miss may hide.
        """
        ws = self.workspace
        ordered_ids = list(object_ids)
        entity_cls = ws._schema.require(entity) if entity is not None else None
        if columns is not None and entity_cls is None:
            raise ValueError("columns requires entity=... so the selection can be validated against one schema")
        if entity_cls is not None:
            return self.entity_matches_many(entity_cls, ordered_ids, unique=unique, columns=columns)
        if not ordered_ids:
            return []

        matches_by_pos: List[Optional[RowMatch]] = [None for _ in ordered_ids]
        counts_by_pos = [0 for _ in ordered_ids]
        names_by_pos: List[List[str]] = [[] for _ in ordered_ids]
        missing_by_pos: List[List[tuple[str, str]]] = [[] for _ in ordered_ids] if for_mutation else []
        for entity_cls, normalized_ids in self.normalized_ids_by_entity(ordered_ids):
            schema = entity_cls.schema()
            wanted = [normalized_id for normalized_id in normalized_ids if normalized_id is not None]
            if not wanted:
                continue
            rows_by_id = ws._row_store.rows_for_ids(schema.entity_id, wanted)
            for pos, normalized_id in enumerate(normalized_ids):
                if normalized_id is None:
                    continue
                fragments = rows_by_id.get(normalized_id, [])
                row = ws._row_store.merge_id_rows(schema, normalized_id, fragments, unique=unique)
                if row is None:
                    if for_mutation:
                        missing_by_pos[pos].append((schema.entity_id, normalized_id))
                    continue
                count = counts_by_pos[pos] + 1
                counts_by_pos[pos] = count
                if count <= 8:
                    names_by_pos[pos].append(schema.entity_id)
                matches_by_pos[pos] = RowMatch(entity_cls, normalized_id, row) if count == 1 else None
        for original_id, count, entity_names in zip(ordered_ids, counts_by_pos, names_by_pos):
            if count > 1:
                names = ", ".join(entity_names)
                suffix = "" if count <= len(entity_names) else f", +{count - len(entity_names)} more"
                raise ValueError(f"object_id {original_id!r} exists in multiple entities: {names}{suffix}; pass entity=... to disambiguate")
        for missing in missing_by_pos:
            self.validate_entity_inference(missing)
        return matches_by_pos

    def entity_matches_many(
        self,
        entity_cls: type,
        object_ids: Iterable[Any],
        *,
        unique: bool = True,
        columns: Sequence[str] | None = None,
        for_patch: bool = False,
        validate_schema: bool = True,
    ) -> List[Optional[RowMatch]]:
        """Return matches for many object_ids of one present entity class.

        The read touches only the backends routed by the entity's storage plan,
        through batched ``get_many`` lookups and the workspace row codec; Catalog
        and the other present entities are never consulted, so cross-entity
        uniqueness is moot. ``unique`` governs only routed fragments of one row
        that disagree on a stored field: ``True`` raises, ``False`` keeps the
        first route's value. A duplicate ``object_id`` inside one entity cannot
        exist because every route stores it as the physical primary key.
        ``columns`` restricts the read to the named declared fields.
        ``for_patch`` validates missing routes before merging the row, so a
        backend with eventually consistent reads can reject a write-back. A row
        that no route returned needs no patch source once a route shows it is
        absent; see ``absent_row_error``.
        A caller that defers schema validation receives the missing source routes
        with the match and can validate the row if it needs to rewrite it.
        """
        ws = self.workspace
        schema = entity_cls.schema()
        selection = self.row_columns(schema, columns)
        normalized_ids = [self.normalize_object_id(schema, object_id) for object_id in object_ids]
        if not normalized_ids:
            return []
        rows_by_id = ws._row_store.rows_for_ids(schema.entity_id, normalized_ids, columns=selection)
        routes = ws._row_store.routed_backend_names(schema.entity_id) if for_patch or not validate_schema else []
        out: List[Optional[RowMatch]] = []
        for normalized_id in normalized_ids:
            fragments = rows_by_id.get(normalized_id, [])
            missing: Tuple[str, ...] = ()
            if routes:
                found = {fragment.backend_name for fragment in fragments}
                missing = tuple(name for name in routes if name not in found)
            if for_patch and (fragments or self.absent_row_error(schema.entity_id, normalized_id) is not None):
                self.validate_missing_fragments(schema.entity_id, normalized_id, missing)
            row = ws._row_store.merge_id_rows(schema, normalized_id, fragments, unique=unique, columns=selection, validate_schema=validate_schema)
            out.append(RowMatch(entity_cls, normalized_id, row, missing) if row is not None else None)
        return out

    def validate_missing_fragments(self, entity_id: str, object_id: str, routes: Sequence[str]) -> None:
        """Ask each route that returned no fragment whether a patch can do without that route's values.

        Args:
            entity_id: The entity whose row is patched.
            object_id: The normalized object id of the row.
            routes: Backend names of the routes that returned no fragment.

        Raises:
            ValueError: If a route cannot show that its fragment is absent.
        """
        for name in routes:
            self.workspace.backends.get(name).validate_missing_row(entity_id, object_id, for_patch=True)

    def absent_row_error(self, entity_id: str, object_id: str) -> ValueError | None:
        """Return why a row that no route returned may still exist, or ``None`` when it is absent.

        The caller has read every route of the entity and none returned the row.
        One route whose backend accepts its miss shows the row is absent, since
        such a backend keeps every row written to it and every row is written
        to every route the entity has at the time. Only the routes that joined
        first are asked (see ``_SchemaService.absence_routes``): a route that
        ``register(..., allow_replace=True)`` added later holds no row stored
        before it joined, so its miss shows nothing while an older route is
        left. Only when every route asked rejects its miss, as Pinecone's
        eventually consistent fetch does, is the absence inconclusive.

        Args:
            entity_id: The entity whose row was not found.
            object_id: The normalized object id that no route returned.

        Returns:
            ValueError | None: The first route's error when no route shows the
            row is absent, otherwise ``None``.
        """
        ws = self.workspace
        error: ValueError | None = None
        for name in ws._schema.absence_routes(entity_id):
            try:
                ws.backends.get(name).validate_missing_row(entity_id, object_id)
            except ValueError as exc:
                error = error or exc
            else:
                return None
        return error

    def validate_absent_row(self, entity_id: str, object_id: str) -> None:
        """Reject a mutation that relies on a row being absent when no route shows it is.

        Raises:
            ValueError: If every route of the entity rejects its negative read.
        """
        error = self.absent_row_error(entity_id, object_id)
        if error is not None:
            raise error

    def validate_entity_inference(self, missing: Iterable[tuple[str, str]]) -> None:
        """Reject inference when a negative read cannot exclude another entity."""
        try:
            for entity_id, object_id in missing:
                self.validate_absent_row(entity_id, object_id)
        except ValueError as exc:
            raise ValueError(f"{exc}; specify the target entity explicitly") from exc

    def row_columns(self, schema: Any, columns: Sequence[str] | None) -> tuple[str, ...] | None:
        """Validate a requested field selection against one entity schema.

        Args:
            schema: Entity schema the selection must belong to.
            columns: Requested logical field names, or ``None`` for the complete row.

        Returns:
            tuple[str, ...] | None: Unique names in request order, or ``None``.

        Raises:
            TypeError: If ``columns`` is a single string instead of a sequence of
                names, or a requested name is not a string.
            ValueError: If a requested name is not a field of the schema.
        """
        if columns is None:
            return None
        if isinstance(columns, str):
            raise TypeError(f"columns must be a sequence of field names, not a single str; pass [{columns!r}] to select one field")
        if isinstance(columns, bytes):
            raise TypeError("columns must be a sequence of str field names, not bytes")
        names = list(columns)
        if not all(isinstance(name, str) for name in names):
            raise TypeError("columns must be field names")
        unknown = [name for name in names if name != "object_id" and name not in schema.fields]
        if unknown:
            raise ValueError(f"{schema.entity_id} has no field(s) {unknown}; columns must name declared fields")
        return tuple(dict.fromkeys(names))

    def id_exists_many(self, object_ids: Iterable[Any], entity: Any = None, *, unique: bool = True) -> List[bool]:
        """Return existence flags for many object_ids using backend indexes."""
        ws = self.workspace
        ordered_ids = list(object_ids)
        entity_cls = ws._schema.require(entity) if entity is not None else None
        if not ordered_ids:
            return []
        if entity_cls is not None:
            schema = entity_cls.schema()
            normalized_ids = [self.normalize_object_id(schema, object_id) for object_id in ordered_ids]
            return self.entity_exists_many(schema, normalized_ids)

        matches_by_pos = [0 for _ in ordered_ids]
        for entity_cls, normalized_ids in self.normalized_ids_by_entity(ordered_ids):
            positions = [pos for pos, normalized_id in enumerate(normalized_ids) if normalized_id is not None]
            if not positions:
                continue
            flags = self.entity_exists_many(entity_cls.schema(), [normalized_ids[pos] for pos in positions])
            for pos, exists in zip(positions, flags):
                if exists:
                    matches_by_pos[pos] += 1
        out = []
        for original_id, count in zip(ordered_ids, matches_by_pos):
            if unique and count > 1:
                raise ValueError(f"object_id {original_id!r} exists in multiple entities; pass entity=... to disambiguate")
            out.append(count > 0)
        return out

    def entity_exists_many(self, schema: Any, normalized_ids: List[str]) -> List[bool]:
        """Return existence flags for one entity from participating backends."""
        out = [False for _ in normalized_ids]
        for backend_name in self.workspace._row_store.routed_backend_names(schema.entity_id):
            for pos, exists in enumerate(self.workspace.backends.get(backend_name).exists_many(schema.entity_id, normalized_ids)):
                out[pos] = out[pos] or exists
        return out
