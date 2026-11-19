"""Guarded reads of stored entity names: bounded pages bound to a reader's pin (A0.3, A.6.5, D.6).

Entity and ``entity_name`` rows are read through S1b's bounded executor,
`IndexedReader.page` with the `EntityNameRows` family: one SQL statement per
page, LIMIT and keyset in SQL, at most 512 rows and 32 values per ``IN``,
header columns only. A stored row is a storage fact, not a readable result;
this module turns such reads into readable names at one pin:

- `readable_names` binds stored rows to the reader's pinned RecallSnapshot
  root. A row the root does not hold, or holds differently, makes the read
  ``delivery_stale``; only rows passing P-NAME at the reader's clock remain.
- `entity_names` reads one entity's readable names, and `named_entities` the
  entities with a readable name of some ``name_norm`` values; an unreadable
  row never takes a readable one's place.
- `reference_keys` names the rows a read used, so a delivery can register them
  as a pending-recall reference fence (``store/pending_recall.py``) that any
  later write to them invalidates.

Nothing here scores or queries HB's ``.near()``; callers read outside the
Writer mutex. Reception's active-entity check is S1b's ``entity_header``
(cross-Q15), not this module.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from types import MappingProxyType
from typing import Final

from pydantic import TypeAdapter, ValidationError

from aml.core.entities import MemorySnapshotV2, NameKind
from aml.core.errors import AMLError
from aml.core.query import ScopeV1
from aml.core.readability import Readability
from aml.core.recall_snapshot import EntityNameRowV1 as NameProjection
from aml.core.recall_snapshot import RecallSnapshot
from aml.core.snapshot_memo import decoded_snapshot
from aml.store.domain import EntityNameRowV1, EntityRowV1
from aml.store.entities import ENTITY_IDENTIFIERS
from aml.store.indexed import MAX_PAGE_SIZE, EntityNameRows, IndexedReader
from aml.store.pending_recall import PendingRecallReferenceKey

_NAME: Final[TypeAdapter[EntityNameRowV1]] = TypeAdapter(EntityNameRowV1)
_NAME_KIND_ORDER: Final[Mapping[NameKind, int]] = MappingProxyType(
    {"name": 0, "alias": 1, "former": 2},
)


def _stored(reader: IndexedReader, query: EntityNameRows) -> list[EntityNameRowV1]:
    """Drain one name query page by page, validating each row against the request."""
    rows: list[EntityNameRowV1] = []
    after: str | None = None
    while True:
        page = reader.page(query, after=after, limit=MAX_PAGE_SIZE)
        try:
            found = [_NAME.validate_python(row) for row in page]
        except ValidationError as error:
            raise AMLError("schema_invalid", "A stored entity name is invalid") from error
        if any(
            (row.scope_kind, row.scope_id) != (query.scope_kind, query.scope_id)
            or (query.entity_id is not None and row.entity_id != query.entity_id)
            or (query.name_norms and row.name_norm not in query.name_norms)
            or (after is not None and row.object_id <= after)
            for row in found
        ):
            raise AMLError("schema_invalid", "A name page left its exact request")
        rows.extend(found)
        if len(found) < MAX_PAGE_SIZE:
            return rows
        after = found[-1].object_id


def readable_names(
    root: RecallSnapshot, read: Readability, rows: Iterable[EntityNameRowV1],
) -> tuple[EntityNameRowV1, ...]:
    """Return the stored name rows that pass P-NAME at a pinned root, binding them to it.

    Each stored row must equal the root's ``entity_name`` projection;
    otherwise the storage read and the pin disagree and the read is stale. A
    row then counts only when its naming Item exists in the scope, is active,
    has the row's revision as its head and that head passes P-VER at
    ``read``'s clock (D.2).

    Args:
        root (RecallSnapshot): The reader's pinned root.
        read (Readability): The reader's readability over that root's Evidence.
        rows (Iterable[EntityNameRowV1]): Stored rows of ``read``'s scope.

    Returns:
        tuple[EntityNameRowV1, ...]: The readable rows, in their given order.

    Raises:
        AMLError: ``scope_denied`` for a row of another scope;
            ``delivery_stale`` when a row differs from the pinned projection.
    """
    readable: list[EntityNameRowV1] = []
    heads: dict[str, MemorySnapshotV2] = {}
    for row in rows:
        if (row.scope_kind, row.scope_id) != (read.scope_kind, read.scope_id):
            raise AMLError("scope_denied", "A name row lies outside the reader's scope")
        projection = root.get_entity_name(row.object_id)
        if projection is None or projection != NameProjection(
            object_id=row.object_id, scope_kind=row.scope_kind, scope_id=row.scope_id,
            entity_id=row.entity_id, name=row.name, name_norm=row.name_norm,
            name_kind=row.name_kind, naming_memory_id=row.naming_memory_id,
            naming_revision_id=row.naming_revision_id, annotation_index=row.annotation_index,
            commit_seq=row.commit_seq, created_at=row.created_at,
        ):
            raise AMLError("delivery_stale", "A name row differs from the pinned root")
        naming = root.get_memory(row.naming_memory_id)
        if naming is None or (naming.scope_kind, naming.scope_id) != (
            read.scope_kind, read.scope_id,
        ):
            continue
        head = heads.get(naming.current_revision_id)
        if head is None:
            head = heads[naming.current_revision_id] = decoded_snapshot(naming.snapshot_json)
        if read.p_name(row, naming, head):
            readable.append(row)
    return tuple(readable)


def _reader_scope(scope: ScopeV1, read: Readability) -> None:
    if (scope.scope_kind, scope.scope_id) != (read.scope_kind, read.scope_id):
        raise AMLError("scope_denied", "A name read serves the reader's exact scope")


def entity_names(
    reader: IndexedReader, root: RecallSnapshot, read: Readability, scope: ScopeV1,
    entity_id: str,
) -> tuple[EntityNameRowV1, ...]:
    """Return one entity's readable names at a reader's pin (A.3.7, E.2.5).

    Args:
        reader (IndexedReader): The workspace's bounded reader.
        root (RecallSnapshot): The reader's pinned root.
        read (Readability): The reader's readability over that root's Evidence.
        scope (ScopeV1): The reader's exact scope.
        entity_id (str): An `EntityRef`, ``"user"`` included.

    Returns:
        tuple[EntityNameRowV1, ...]: The P-NAME rows, ``name`` before ``alias``
            before ``former``, then by ``name_norm`` and row id.

    Raises:
        AMLError: ``scope_denied`` when ``scope`` is not ``read``'s;
            ``delivery_stale`` when storage and the pin disagree; an
            ``invalid_request`` or ``schema_invalid`` from the bounded read.
    """
    _reader_scope(scope, read)
    try:
        query = EntityNameRows(
            scope_kind=scope.scope_kind, scope_id=scope.scope_id, entity_id=entity_id,
        )
    except ValidationError as error:
        raise AMLError("invalid_request", "An entity name read needs an EntityRef") from error
    return tuple(sorted(
        readable_names(root, read, _stored(reader, query)),
        key=lambda row: (_NAME_KIND_ORDER[row.name_kind], row.name_norm, row.object_id),
    ))


def named_entities(
    reader: IndexedReader, root: RecallSnapshot, read: Readability, scope: ScopeV1,
    name_norms: Sequence[str],
) -> tuple[str, ...]:
    """Return the entities with a readable name of some ``name_norm`` values (A.6.5, E.2.5).

    A match is a candidate list, never identity proof (INV:D05).

    Args:
        reader (IndexedReader): The workspace's bounded reader.
        root (RecallSnapshot): The reader's pinned root.
        read (Readability): The reader's readability over that root's Evidence.
        scope (ScopeV1): The reader's exact scope.
        name_norms (Sequence[str]): 1..32 distinct ``name_norm`` values.

    Returns:
        tuple[str, ...]: The distinct named `EntityRef` values, sorted.

    Raises:
        AMLError: As for `entity_names`.
    """
    _reader_scope(scope, read)
    if not name_norms:
        raise AMLError("invalid_request", "A name lookup needs at least one name_norm")
    try:
        query = EntityNameRows(
            scope_kind=scope.scope_kind, scope_id=scope.scope_id, name_norms=tuple(name_norms),
        )
    except ValidationError as error:
        raise AMLError("invalid_request", "A name lookup holds 1..32 distinct names") from error
    return tuple(sorted({
        row.entity_id for row in readable_names(root, read, _stored(reader, query))
    }))


def reference_keys(
    rows: Iterable[EntityRowV1 | EntityNameRowV1],
) -> frozenset[PendingRecallReferenceKey]:
    """Return the storage keys of rows a read used, for a pending-recall fence.

    The Writer invalidates every registered fence whose keys a write touches,
    so a fence over these keys goes stale as soon as any of the rows changes or
    is deleted.

    Args:
        rows (Iterable[EntityRowV1 | EntityNameRowV1]): The rows the read used.

    Returns:
        frozenset[PendingRecallReferenceKey]: ``(HB entity identifier,
            object_id)`` of each row.
    """
    return frozenset(
        (ENTITY_IDENTIFIERS["entity" if isinstance(row, EntityRowV1) else "entity_name"],
         row.object_id)
        for row in rows
    )
