"""Workspace row-fragment collection, merge, and hydration."""

from __future__ import annotations

__all__ = [
    "RowFragment",
]

from heapq import heapify, heappop, heappush
from itertools import islice

from heavenbase.utils import Any, Dict, Iterable, Iterator, List, Optional, Sequence, dataclass

from ...backends.base import Backend, accepts_keyword, is_portable_layer, project_rows, read_projected
from ...execution.contracts import QueryFragment
from ...execution.frame.result import ResultFrame


@dataclass(frozen=True)
class RowFragment:
    """Associate one backend-returned row fragment with its source.

    Args:
        backend_name (str): Workspace backend name that returned the row.
        row (Dict[str, Any]): Backend-ready row fragment containing `object_id`.
    """

    backend_name: str
    row: Dict[str, Any]

    def __post_init__(self) -> None:
        """Validate source identity and isolate the backend row mapping.

        Args:
            None.

        Returns:
            None: This initializer normalizes the fragment in place.

        Raises:
            ValueError: If `backend_name` is empty.
            TypeError: If `row` is not a mapping.
        """
        name = str(self.backend_name).strip()
        if not name:
            raise ValueError("RowFragment backend_name is required")
        try:
            row = dict(self.row)
        except (TypeError, ValueError) as exc:
            raise TypeError("RowFragment row must be a backend field mapping") from exc
        object.__setattr__(self, "backend_name", name)
        object.__setattr__(self, "row", row)


class _RowStore:
    """Own backend row fragments, merge, hydration, and scans."""

    def __init__(self, workspace: Any):
        self.workspace = workspace

    def routed_backend_names(self, entity_id: str) -> List[str]:
        """Return backend names that hold any routed fields for an entity."""
        ws = self.workspace
        with ws._lock.read():
            bindings = list(ws.storage_plans[entity_id].bindings)
        return sorted({binding.backend for binding in bindings})

    def rows_for_ids(
        self, entity_id: str, row_ids: Iterable[str], *, columns: Sequence[str] | None = None, near_fragment: QueryFragment | None = None
    ) -> Dict[str, List[RowFragment]]:
        """Return backend row fragments for many object_ids.

        With ``columns``, each fragment carries only the requested fields its route owns
        plus identity. A route can project the fields in storage or return complete
        fragments for trimming. Identity-only fragments still establish presence.
        """
        unique_ids = list(dict.fromkeys(row_ids))
        rows_by_id: Dict[str, List[RowFragment]] = {row_id: [] for row_id in unique_ids}
        if not unique_ids:
            return rows_by_id
        for backend_name in self.routed_backend_names(entity_id):
            fragment = near_fragment if near_fragment is not None and near_fragment.backend == backend_name else None
            backend_rows = self._lookup_route(backend_name, entity_id, unique_ids, columns, near_fragment=fragment)
            for row_id, row in zip(unique_ids, backend_rows):
                if row is not None:
                    rows_by_id.setdefault(row_id, []).append(RowFragment(backend_name, row))
        return rows_by_id

    def backend_columns(self, entity_id: str, backend_name: str, columns: Sequence[str]) -> List[str]:
        """Return the requested field names that one routed backend physically stores."""
        schema = self.workspace.entities[entity_id].schema()
        owned = self.workspace._row_codec.field_names_for_backend(schema, backend_name)
        return [name for name in columns if name in owned]

    def _lookup_route(
        self, backend_name: str, entity_id: str, ids: List[str], columns: Sequence[str] | None, *, near_fragment: QueryFragment | None = None
    ) -> List[Optional[Dict[str, Any]]]:
        """Read ids from one route, naming a selection only when the caller projected.

        A route whose ``get_many`` predates ``columns`` answers complete fragments, trimmed by ``read_projected``.
        """
        backend = self.workspace.backends.get(backend_name)
        selection = None if columns is None else self.backend_columns(entity_id, backend_name, columns)
        kwargs = {"_near_fragment": near_fragment} if near_fragment is not None and accepts_keyword(backend, "get_many", "_near_fragment") else {}
        return read_projected(backend, "get_many", entity_id, ids, columns=selection, **kwargs)

    def _scan_page(self, backend_name: str, entity_id: str, *, after: Any, batch_size: int, columns: Sequence[str] | None) -> list[dict[str, Any]]:
        """Fetch one keyset page from one route, naming a selection only when the caller projected."""
        backend = self.workspace.backends.get(backend_name)
        selection = None if columns is None else self.backend_columns(entity_id, backend_name, columns)
        return read_projected(backend, "scan_page", entity_id, after=after, batch_size=batch_size, columns=selection)

    def _pages_natively(self, backend_name: str) -> bool:
        """Return whether one route overrides the portable ``Backend.scan_page``, on its class or instance."""
        return not is_portable_layer(getattr(self.workspace.backends.get(backend_name), "scan_page", None), "scan_page")

    def _streamed_pages(self, backend_name: str, entity_id: str, *, batch_size: int, columns: Sequence[str] | None) -> Iterator[list[dict[str, Any]]]:
        """Cut one route's single ``iter_scan`` pass into pages; the provider may hold its lock across them."""
        backend = self.workspace.backends.get(backend_name)
        selection = None if columns is None else self.backend_columns(entity_id, backend_name, columns)
        # A provider whose iter_scan predates ``columns`` answers complete fragments; pages are trimmed below.
        pushed = selection is not None and accepts_keyword(backend, "iter_scan", "columns")
        rows = iter(backend.iter_scan(entity_id, batch_size=batch_size, **({"columns": selection} if pushed else {})))
        try:
            while batch := list(islice(rows, batch_size)):
                yield batch if pushed or selection is None else project_rows(batch, selection)
        finally:
            close = getattr(rows, "close", None)
            if callable(close):
                close()

    def _keyset_pages(self, backend_name: str, entity_id: str, *, batch_size: int, columns: Sequence[str] | None) -> Iterator[list[dict[str, Any]]]:
        """Page one route through ``Backend.scan_page``; no backend lock outlives a page fetch.

        Closed before its empty last page, it ends the scan with ``Backend.end_scan``,
        so the provider can drop what it kept for the next page.
        """
        end_scan = getattr(self.workspace.backends.get(backend_name), "end_scan", None)
        after = None
        try:
            while batch := self._scan_page(backend_name, entity_id, after=after, batch_size=batch_size, columns=columns):
                after = batch[-1]["object_id"]
                yield batch
            after = None
        finally:
            if after is not None and callable(end_scan):
                end_scan(entity_id, after=after)

    def merge_id_rows(
        self,
        schema: Any,
        row_id: str,
        rows: List[RowFragment],
        *,
        unique: bool,
        validate_schema: bool = True,
        columns: Sequence[str] | None = None,
    ) -> Optional[Dict[str, Any]]:
        """Merge routed row fragments and report field conflicts for one object_id.

        ``columns`` names the fields a projected read asked for; the schema-version
        check then covers only those fields, since the others were never fetched.
        """
        return self._merge_id_row(
            schema,
            row_id,
            rows,
            unique=unique,
            validate_schema=validate_schema,
            columns=columns,
        )[0]

    def _merge_id_row(
        self,
        schema: Any,
        row_id: str,
        rows: List[RowFragment],
        *,
        unique: bool,
        validate_schema: bool = True,
        columns: Sequence[str] | None = None,
    ) -> tuple[Optional[Dict[str, Any]], frozenset[str]]:
        """Merge one row while retaining fields supplied by authoritative routes."""
        if not rows:
            return None, frozenset()
        merged: Dict[str, Any] = {"object_id": row_id, schema.pk_name: row_id}
        conflicts = []
        soft_fields = self._soft_vector_fields(schema.entity_id)
        soft_sourced: set[str] = set()
        authoritative_sourced: set[str] = set()
        for fragment in rows:
            backend_name = fragment.backend_name
            row = fragment.row
            if row.get("object_id") != row_id:
                conflicts.append(f"{backend_name}.object_id={row.get('object_id')!r}")
            for key, value in self.workspace._row_codec.decode_row_items(schema, backend_name, row).items():
                if key == "object_id":
                    continue
                if key == schema.pk_name:
                    if value not in (None, "", row_id):
                        conflicts.append(f"{backend_name}.{key}={value!r}")
                    continue
                soft = (key, backend_name) in soft_fields
                if key not in merged or merged[key] is None:
                    if soft and key in authoritative_sourced:
                        continue
                    merged[key] = value
                    if soft:
                        soft_sourced.add(key)
                    else:
                        authoritative_sourced.add(key)
                elif value is not None and not self.workspace._row_codec.same_field_value(schema, key, merged[key], value):
                    if soft:
                        continue
                    if key in soft_sourced:
                        merged[key] = value
                        soft_sourced.discard(key)
                        authoritative_sourced.add(key)
                        continue
                    conflicts.append(f"{backend_name}.{key}={value!r} != {merged[key]!r}")
                elif key in soft_sourced and not soft:
                    merged[key] = value
                    soft_sourced.discard(key)
                    authoritative_sourced.add(key)
        if unique and conflicts:
            detail = "; ".join(conflicts[:8])
            suffix = "" if len(conflicts) <= 8 else f"; +{len(conflicts) - 8} more"
            raise ValueError(f"{schema.entity_id} object_id {row_id!r} is not unique across routed backends: {detail}{suffix}")
        entity_cls = self.workspace.entities.get(schema.entity_id)
        if validate_schema and entity_cls is not None:
            self.workspace._materializer.assert_query_row_schema(entity_cls, merged, fields=columns)
        return merged, frozenset(authoritative_sourced)

    def _soft_vector_fields(self, entity_id: str) -> frozenset[tuple[str, str]]:
        """Return vector routes backed by an authoritative non-vector copy.

        A vector store's copy of an embedding field exists for similarity search, not
        row reads: it is written through a lossy float32 round trip, and when one
        collection backs several vector fields its read-side attribution follows
        schema field order — which manifest replay does not preserve. Row merging
        treats these replicas as soft: they fill a gap but never dispute or override
        an inline copy, including an authoritative null. A vector-only field remains
        authoritative and therefore retains ordinary conflict detection.
        """
        from ...strategies import VectorIndex

        plan = self.workspace.storage_plans[entity_id]
        authoritative = {binding.field_name for binding in plan.bindings if binding.strategy != VectorIndex.identifier}
        return frozenset(
            (binding.field_name, binding.backend)
            for binding in plan.bindings
            if binding.strategy == VectorIndex.identifier and binding.field_name in authoritative
        )

    def hydrate_query_frame(self, schema: Any, frame: ResultFrame, *, near_fragment: QueryFragment | None = None) -> ResultFrame:
        """Attach logical row fields to a backend result frame by object_id.

        The backend query and this read are separate, so a row deleted in
        between is held by no route any more; it is left out rather than
        returned as a row that holds only its id and the query's metadata.
        A miss is left out only when a route shows the row is absent (see
        ``_RowLookup.absent_row_error``). Otherwise each route may read it
        again (``Backend.settle_query_misses``): Pinecone fetches the record
        again within a short bound. A row a later read returns is hydrated from
        it, and a miss that no read settles keeps the row with the values the
        backend query returned, since an eventually consistent miss never shows
        that the row was deleted.
        The near backend can reuse the fragment's resolved provider route for its read.
        """
        if len(frame) == 0:
            return frame
        route = {"near_fragment": near_fragment} if near_fragment is not None else {}
        rows_by_id = self.rows_for_ids(schema.entity_id, frame.ids, **route)
        lookup = self.workspace._row_lookup
        missed = [row_id for row_id in dict.fromkeys(frame.ids) if not rows_by_id.get(row_id)]
        absent = {row_id for row_id in missed if lookup.absent_row_error(schema.entity_id, row_id) is None}
        unsettled = [row_id for row_id in missed if row_id not in absent]
        if unsettled:
            self._settle_query_misses(schema.entity_id, unsettled, rows_by_id, near_fragment=near_fragment)
        schema_fields = set(schema.fields)
        hydrated = []
        for source in frame.rows():
            row_id = source.get("object_id")
            merged, authoritative_fields = self._merge_id_row(schema, row_id, rows_by_id.get(row_id, []), unique=True)
            if merged is None:
                if row_id in absent:
                    continue  # a route shows the row is absent: it was deleted after the query
                merged = {"object_id": row_id, schema.pk_name: row_id}
            for key, value in source.items():
                if key == "object_id":
                    continue
                if key not in schema_fields:
                    merged[key] = value
                elif key not in merged or (merged[key] is None and key not in authoritative_fields):
                    field_schema = schema.get(key)
                    merged[key] = self.workspace._row_codec.decode_field_value(field_schema, value) if field_schema else value
            hydrated.append(merged)
        return ResultFrame.from_rows(hydrated, metadata_columns=frame.metadata_columns)

    def _settle_query_misses(
        self,
        entity_id: str,
        row_ids: List[str],
        rows_by_id: Dict[str, List[RowFragment]],
        *,
        near_fragment: QueryFragment | None,
    ) -> None:
        """Ask each route to read hydration misses again; add the rows found to ``rows_by_id``."""
        for backend_name in self.routed_backend_names(entity_id):
            backend = self.workspace.backends.get(backend_name)
            fragment = near_fragment if near_fragment is not None and near_fragment.backend == backend_name else None
            for row_id, row in backend.settle_query_misses(entity_id, row_ids, _near_fragment=fragment).items():
                if row_id in rows_by_id:
                    rows_by_id[row_id].append(RowFragment(backend_name, row))

    def needs_query_hydration(self, schema: Any, frame: ResultFrame) -> bool:
        """Return whether a query frame is missing logical schema fields."""
        schema_fields = set(schema.fields)
        frame_fields = set(frame.col_names())
        if schema_fields - frame_fields:
            return True
        return any(any(self.missing_query_value(row.get(field_name)) for field_name in schema_fields) for row in frame._iter_rows())

    def missing_query_value(self, value: Any) -> bool:
        """Return whether a backend query value should be hydrated from row storage."""
        if value is None:
            return True
        try:
            return bool(value != value)
        except Exception:
            return False

    def iter_rows(
        self,
        entity_id: str,
        *,
        batch_size: int,
        columns: Sequence[str] | None = None,
        held: bool = False,
    ) -> Iterable[dict[str, Any]]:
        """Merge the exhaustive route union row by row; see ``iter_row_pages``."""
        pages = self.iter_row_pages(entity_id, batch_size=batch_size, columns=columns, held=held)
        try:
            for page in pages:
                yield from page
        finally:
            pages.close()

    def iter_row_pages(
        self,
        entity_id: str,
        *,
        batch_size: int,
        columns: Sequence[str] | None = None,
        held: bool = False,
    ) -> Iterator[list[dict[str, Any]]]:
        """Merge the exhaustive route union in bounded pages.

        With one route, each page is one ``Backend.scan_page`` keyset batch,
        decoded. With several, each page reads one keyset page from every route
        and merges them by ``object_id``; see ``_ordered_route_union``. A
        backend lock is held only inside those calls and never across the
        ``yield``, and no rows are buffered across it, so a caller may release
        shared ownership between pages while a writer takes the workspace and
        backend locks in the usual order. Duplicate identities across routes
        on the keyset path are settled by the merge and by batched point lookups.
        A route enumerates through ``iter_scan``,
        whose ``Backend`` default reads one ``scan``; a route with neither fails
        closed. With ``columns``, only the requested fields each route owns
        and its identity are passed to the row codec.

        Rows come in ascending ``object_id`` order across all routes, so a row
        that only a later route holds still takes its place in the order, and
        each row is yielded once. The merge compares ids as Python does
        (code-point order), as the in-memory backend, SQLite's default BINARY
        collation and the portable ``scan_page`` do. A route whose database
        collation orders ids otherwise, such as a case-insensitive one, still
        yields each row once while nothing writes during the scan, but the
        order across routes may then not be ascending.

        ``held=True`` is for a caller that holds the workspace read lock, or
        its write lock, until the iterator is exhausted or closed, so no other
        writer can run between pages; a caller holding the write lock may write
        between pages, but never rows of the entity being read.
        A route that inherits the portable ``Backend.scan_page``, which
        re-enumerates the route for every page, is then read through one
        ``iter_scan`` pass instead. Such a provider may hold its own lock across
        that pass, so this applies only when every other route pages natively:
        two routes that both held their locks across the merge's point lookups
        into each other could deadlock two concurrent scans. Because
        ``iter_scan`` has no order and its lock must not be held while the
        other routes are read, a streamed scan reads the routes one after
        another, and its rows are not in ``object_id`` order.
        If the streamed route also looks up ids through the portable
        ``Backend.get_many``, read it first and retain its ids. Its rows
        hydrate from the native routes, and the later pages use the retained
        ids to eliminate duplicates without looking up the streamed route.
        """
        names = self.routed_backend_names(entity_id)
        portable = [name for name in names if not self._pages_natively(name)]
        streamed = portable[0] if held and len(portable) == 1 else None
        if streamed is None and len(names) != 1:
            yield from self._ordered_route_union(entity_id, names, batch_size, columns)
            return

        scanned_ids: set[Any] | None = None
        if streamed is not None and len(names) > 1:
            lookup = self.workspace.backends.get(streamed).get_many
            if getattr(lookup, "__func__", lookup) is Backend.get_many:
                names.remove(streamed)
                names.insert(0, streamed)
                scanned_ids = set()

        for position, name in enumerate(names):
            source = self._streamed_pages if name == streamed else self._keyset_pages
            batches = source(name, entity_id, batch_size=batch_size, columns=columns)
            try:
                if scanned_ids is not None and position > 0:
                    yield from self._merge_route_pages(entity_id, names[1:], position - 1, batches, columns, excluded_ids=scanned_ids)
                else:
                    for page in self._merge_route_pages(entity_id, names, position, batches, columns):
                        if scanned_ids is not None:
                            scanned_ids.update(row["object_id"] for row in page)
                        yield page
            finally:
                batches.close()

    def _ordered_route_union(
        self,
        entity_id: str,
        names: List[str],
        batch_size: int,
        columns: Sequence[str] | None,
    ) -> Iterator[list[dict[str, Any]]]:
        """Merge every route's keyset pages by ``object_id``, one bounded output page per step.

        Each step reads one fresh ``scan_page`` per route after that route's own
        cursor, merges the pages by ``object_id``, and takes at most
        ``batch_size`` identities. It stops early once any route's page is used
        up, full or short, because that route's next id is unknown:
        ``scan_page`` returns at most ``batch_size`` rows, and only an empty
        page ends a route. Equal ids from several routes
        form one identity, owned by the earliest route that holds it. Nothing is
        buffered across a ``yield``: rows not taken are read again on the next
        step, and every fragment of an output row is read in the same step, so
        a step costs one page per route and the rest of a route is never read
        ahead. In the steady state, where every route holds every row, the
        merged pages already carry all fragments and no point lookup is made.

        Every route that orders ids as Python compares them keeps the output
        strictly ascending; an id at or behind the last one yielded is skipped,
        since a route can return it again only after a write between steps. A
        route whose pages break that order turns this skip off for the rest of
        the scan. Ownership is then settled by point lookups, as on the
        sequential path, and ids yielded from a later route are remembered.

        Closed before every route's latest page came back empty, it ends the
        scan of each route whose latest page held rows with ``Backend.end_scan``,
        after that page's last ``object_id``, as ``_keyset_pages`` does for one
        route.
        """
        cursors: list[Any] = [None] * len(names)
        # Last object_id of each route's latest page, or None once that page came back empty.
        page_ends: list[Any] = [None] * len(names)
        last: Any = None
        ordered = True
        orphans: set[Any] = set()
        try:
            while True:
                pages = []
                for position, name in enumerate(names):
                    page = self._scan_page(name, entity_id, after=cursors[position], batch_size=batch_size, columns=columns)
                    page_ends[position] = page[-1]["object_id"] if page else None
                    sequence = [row["object_id"] for row in page]
                    if cursors[position] is not None:
                        sequence.insert(0, cursors[position])
                    if ordered and any(not previous < current for previous, current in zip(sequence, sequence[1:])):
                        ordered = False
                    pages.append(page)
                heads = [(page[0]["object_id"], position, 0) for position, page in enumerate(pages) if page]
                if not heads:
                    return
                heapify(heads)
                groups: list[tuple[Any, dict[int, dict[str, Any]]]] = []
                blocked = False
                while heads and not blocked and len(groups) < batch_size:
                    row_id = heads[0][0]
                    held: dict[int, dict[str, Any]] = {}
                    while heads and heads[0][0] == row_id:
                        _, position, index = heappop(heads)
                        page = pages[position]
                        held.setdefault(position, page[index])
                        cursors[position] = row_id
                        if index + 1 < len(page):
                            heappush(heads, (page[index + 1]["object_id"], position, index + 1))
                        else:
                            # A short page need not be the route's last; only an empty one ends it.
                            blocked = True
                    groups.append((row_id, held))
                if ordered:
                    groups = [group for group in groups if last is None or last < group[0]]
                    if groups:
                        last = groups[-1][0]
                output = self._hydrate_route_groups(entity_id, names, groups, columns, orphans=None if ordered else orphans)
                if output:
                    yield output
        finally:
            self._end_route_scans(entity_id, names, page_ends)

    def _end_route_scans(self, entity_id: str, names: List[str], page_ends: List[Any]) -> None:
        """Call ``Backend.end_scan`` for every route whose latest page held rows, then raise the first failure."""
        failure: Exception | None = None
        for name, after in zip(names, page_ends):
            end_scan = getattr(self.workspace.backends.get(name), "end_scan", None)
            if after is None or not callable(end_scan):
                continue
            try:
                end_scan(entity_id, after=after)
            except Exception as exc:
                failure = failure or exc
        if failure is not None:
            raise failure

    def _hydrate_route_groups(
        self,
        entity_id: str,
        names: List[str],
        groups: List[tuple[Any, Dict[int, Dict[str, Any]]]],
        columns: Sequence[str] | None,
        *,
        orphans: Optional[set[Any]],
    ) -> list[dict[str, Any]]:
        """Complete merged identities by point lookups on the routes their pages lacked, then decode them.

        An identity is dropped when no route holds it any more, or when a route
        earlier than every route that returned it holds it: that route owns it
        and yields it itself. With ``orphans`` (routes out of Python order), an
        id yielded while no earlier route held it is remembered, and dropped if
        an earlier route returns it later.
        """
        if orphans is not None:
            groups = [group for group in groups if group[0] not in orphans]
        fragments = [dict(held) for _, held in groups]
        for position, name in enumerate(names):
            missing = [index for index, held in enumerate(fragments) if position not in held]
            if not missing:
                continue
            found = self._lookup_route(name, entity_id, [groups[index][0] for index in missing], columns)
            if len(found) != len(missing):
                raise RuntimeError(f"Backend {name!r} returned incomplete lookup results")
            for index, row in zip(missing, found):
                if row is not None:
                    fragments[index][position] = row
        schema = self.workspace.entities[entity_id].schema()
        page = []
        for (row_id, held), found in zip(groups, fragments):
            if not found or min(found) < min(held):
                continue
            if orphans is not None and min(found) > 0:
                orphans.add(row_id)
            # Preserve canonical route precedence for replica merging.
            ordered = sorted((RowFragment(names[position], row) for position, row in found.items()), key=lambda item: item.backend_name)
            row = self.merge_id_rows(schema, row_id, ordered, unique=True, columns=columns)
            if row is not None:
                page.append(row)
        return page

    def _merge_route_pages(
        self,
        entity_id: str,
        names: List[str],
        position: int,
        batches: Iterator[list[dict[str, Any]]],
        columns: Sequence[str] | None,
        *,
        excluded_ids: set[Any] | None = None,
    ) -> Iterator[list[dict[str, Any]]]:
        """Merge one route's pages with the other routes, skipping ids an earlier route owns."""
        ws = self.workspace
        name = names[position]
        for batch in batches:
            if excluded_ids is not None:
                batch = [row for row in batch if row["object_id"] not in excluded_ids]
                if not batch:
                    continue
            ids = [row["object_id"] for row in batch]
            # Cross-route duplicates are eliminated before full hydration.
            present = set()
            for prior in names[:position]:
                flags = list(ws.backends.get(prior).exists_many(entity_id, ids))
                if len(flags) != len(ids):
                    raise RuntimeError(f"Backend {prior!r} returned incomplete existence results")
                present.update(row_id for row_id, flag in zip(ids, flags) if flag)
            selected = [row for row in batch if row["object_id"] not in present]
            ids = [row["object_id"] for row in selected]
            fragments = {row["object_id"]: [RowFragment(name, row)] for row in selected}
            for other in names:
                if other == name or not ids:
                    continue
                found = self._lookup_route(other, entity_id, ids, columns)
                if len(found) != len(ids):
                    raise RuntimeError(f"Backend {other!r} returned incomplete lookup results")
                for row_id, row in zip(ids, found):
                    if row is not None:
                        fragments[row_id].append(RowFragment(other, row))
            schema = ws.entities[entity_id].schema()
            page = []
            for row_id in ids:
                # Preserve canonical route precedence for replica merging.
                ordered = sorted(fragments[row_id], key=lambda item: item.backend_name)
                row = self.merge_id_rows(schema, row_id, ordered, unique=True, columns=columns)
                if row is not None:
                    page.append(row)
            if page:
                yield page

    def count(self, entity_id: str) -> int:
        """Return the largest row count reported by one entity's routed backends.

        Routes agree in the steady state, so the maximum is the logical row
        count. It is deliberately a maximum rather than a first answer: a
        backend with eventually consistent counting can lag, and under-reporting
        a total is worse than briefly reporting the leading route's view.
        """
        ws = self.workspace
        backend_names = self.routed_backend_names(entity_id)
        return max((ws.backends.get(backend_name).count(entity_id) for backend_name in backend_names), default=0)

    def universe(
        self,
        entity_id: str,
        *,
        required_backends: Iterable[str] = (),
        backend_names: Optional[Iterable[str]] = None,
        validate_schema: bool = True,
        exhaustive: bool = False,
    ) -> ResultFrame:
        """Return merged logical rows for every stored object in an entity.

        Backends that cannot scan normally contribute no universe rows. Callers that
        use absence as an integrity proof pass `required_backends` so the operation
        fails closed instead. Integrity discovery may select only owning routes and
        defer full-schema validation until candidate rows are hydrated. Exact query
        folds pass `exhaustive=True`: every scannable `object_id` route must
        agree on the logical identity set, while selected lookup-only dependency
        routes are hydrated by those identities only after an exact fragment
        count proves that no hidden ids exist.
        """
        ws = self.workspace
        schema = ws.entities[entity_id].schema()
        required = set(required_backends)
        routed = set(self.routed_backend_names(entity_id))
        selected = routed if backend_names is None else set(backend_names)
        unknown = selected - routed
        if unknown:
            raise ValueError(f"Entity {entity_id!r} has no routed backend(s) {sorted(unknown)}")
        if required - selected:
            raise ValueError(f"Required backend(s) {sorted(required - selected)} were excluded from entity {entity_id!r} scan")
        rows_by_id: Dict[str, List[RowFragment]] = {}
        scanned_ids: Dict[str, set[str]] = {}
        unscannable = []
        for backend_name in sorted(selected):
            try:
                backend_rows = ws.backends.get(backend_name).scan(entity_id)
            except NotImplementedError as exc:
                if backend_name in required:
                    raise RuntimeError(f"GraphEdge integrity requires backend {backend_name!r} to scan entity {entity_id!r}") from exc
                unscannable.append(backend_name)
                continue
            backend_ids = set()
            for row in backend_rows:
                row_id = row.get("object_id")
                if row_id is not None:
                    backend_ids.add(row_id)
                    rows_by_id.setdefault(row_id, []).append(RowFragment(backend_name, row))
            scanned_ids[backend_name] = backend_ids

        if exhaustive:
            identity_backends = []
            identity_lookup_only = []
            for binding in ws.storage_plans[entity_id].field_bindings(schema.pk_name):
                if binding.backend not in selected and binding.backend not in identity_backends:
                    identity_backends.append(binding.backend)
            for backend_name in identity_backends:
                try:
                    backend_rows = ws.backends.get(backend_name).scan(entity_id)
                except NotImplementedError:
                    identity_lookup_only.append(backend_name)
                    continue
                backend_ids = set()
                for row in backend_rows:
                    row_id = row.get("object_id")
                    if row_id is not None:
                        backend_ids.add(row_id)
                        rows_by_id.setdefault(row_id, []).append(RowFragment(backend_name, {"object_id": row_id}))
                scanned_ids[backend_name] = backend_ids
            if not scanned_ids:
                lookup_only = sorted(set(unscannable + identity_lookup_only))
                raise RuntimeError(
                    f"Exact query fold cannot enumerate entity {entity_id!r}: no exhaustive identity route is available; "
                    f"lookup-only backend(s): {lookup_only}"
                )
            known_ids = sorted(rows_by_id)
            known_id_set = set(known_ids)
            for backend_name, backend_ids in scanned_ids.items():
                missing = sorted(known_id_set - backend_ids)
                if missing:
                    raise RuntimeError(
                        f"Exact query fold cannot enumerate entity {entity_id!r}: scannable backend {backend_name!r} "
                        f"is missing {len(missing)} routed identity fragment(s), including {missing[:3]}"
                    )
            for backend_name in unscannable:
                backend = ws.backends.get(backend_name)
                try:
                    expected = backend.exact_count(entity_id)
                except NotImplementedError as exc:
                    raise RuntimeError(
                        f"Exact query fold cannot prove completeness for entity {entity_id!r}: lookup-only backend "
                        f"{backend_name!r} has no exact fragment-count proof"
                    ) from exc
                fetched = list(backend.get_many(entity_id, known_ids))
                if len(fetched) != len(known_ids):
                    raise RuntimeError(
                        f"Exact query fold cannot hydrate entity {entity_id!r}: backend {backend_name!r} "
                        f"returned {len(fetched)} lookup result(s) for {len(known_ids)} identities"
                    )
                found = sum(row is not None for row in fetched)
                if found != expected:
                    raise RuntimeError(
                        f"Exact query fold cannot prove completeness for entity {entity_id!r}: lookup-only backend "
                        f"{backend_name!r} reported {expected} fragment(s) but {found} were reachable through "
                        f"the {len(known_ids)} enumerated identities"
                    )
                for row_id, row in zip(known_ids, fetched):
                    if row is not None:
                        rows_by_id[row_id].append(RowFragment(backend_name, row))
        rows = [
            row
            for row_id, fragments in (sorted(rows_by_id.items()) if exhaustive else rows_by_id.items())
            if (row := self.merge_id_rows(schema, row_id, fragments, unique=True, validate_schema=validate_schema)) is not None
        ]
        return ResultFrame.from_rows(rows)
