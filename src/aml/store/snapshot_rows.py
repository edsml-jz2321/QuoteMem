"""Read committed NL rows into the A.6.1 RecallSnapshot projections (I4).

A cold rebuild reads every committed row of the projected kinds; a commit delta
re-reads the rows its transaction wrote or deleted. Both build each projection
through the same constructors (``MemoryRowV2.from_rows`` and its peers), so a
delta root and a cold root of one commit hold the same state.

An Item is projected with its whole version chain, followed from its head
through ``previous_revision_id``; a revision or link whose Item no longer
exists is not read.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from aml.core.errors import AMLError
from aml.core.recall_snapshot import (
    EntityDeletion,
    EntityNameDeletion,
    EntityNameRowV1,
    EntityRowV1,
    HeldEvidenceRelease,
    HeldEvidenceV1,
    MemoryDeletion,
    MemoryRowV2,
    RecallSnapshot,
    SourceDeletion,
    SourceRowV2,
    SpentMintV1,
)
from aml.store.domain import DOMAIN_ADAPTERS
from aml.store.indexed import MAX_PAGE_SIZE, RevisionLinks

if TYPE_CHECKING:
    from aml.store.writer import CommittedRows, Writer

Row = Mapping[str, Any]
RowLookup = Callable[[str], Row | None]
"""Return the row of one identity, or None when it does not exist."""


@dataclass(frozen=True, slots=True)
class ColdProjections:
    """Every NL projection of a cold root (A.6.1)."""

    memories: tuple[MemoryRowV2, ...]
    entities: tuple[EntityRowV1 | SpentMintV1, ...]
    names: tuple[EntityNameRowV1, ...]
    held: tuple[HeldEvidenceV1, ...]


@dataclass(frozen=True, slots=True)
class CommitDelta:
    """The A.6.1 changes of one committed NL transaction."""

    sources: tuple[SourceRowV2 | SourceDeletion, ...]
    memories: tuple[MemoryRowV2 | MemoryDeletion, ...]
    entities: tuple[EntityRowV1 | EntityDeletion | SpentMintV1, ...]
    names: tuple[EntityNameRowV1 | EntityNameDeletion, ...]
    held: tuple[HeldEvidenceV1 | HeldEvidenceRelease, ...]


def strict_row(kind: str, raw: Row) -> dict[str, Any]:
    """Return a stored row's strict JSON image.

    Args:
        kind: The row's kind.
        raw: The row as storage returned it.

    Returns:
        The row parsed by its strict model and dumped as JSON.
    """
    return DOMAIN_ADAPTERS[kind].validate_python(raw).model_dump(mode="json")


def version_chain(item: Row, revision: RowLookup) -> list[Row]:
    """Return an Item's revisions from version 1 to its head.

    Args:
        item: The Item's strict JSON row.
        revision: A revision's strict JSON row by identity.

    Returns:
        The chain followed from the head through ``previous_revision_id``.

    Raises:
        AMLError: ``parity_mismatch`` when a revision of the chain is missing
            or the versions do not descend by one.
    """
    chain: list[Row] = []
    identity: str | None = item["current_revision_id"]
    while identity is not None:
        row = revision(identity)
        if row is None or (chain and row["version"] != chain[-1]["version"] - 1):
            raise AMLError("parity_mismatch", "A stored Item lost a version of its chain")
        chain.append(row)
        identity = row["previous_revision_id"]
    chain.reverse()
    return chain


def revision_links(writer: Writer, revision_id: str) -> list[dict[str, Any]]:
    """Return every stored link a revision carries, in object-id order.

    Args:
        writer: The sole owner, under its mutex.
        revision_id: The source revision.

    Returns:
        The links' strict JSON rows, read in bounded pages.
    """
    reader = writer.indexed_reader()
    query = RevisionLinks(source_revision_id=revision_id)
    rows: list[dict[str, Any]] = []
    after: str | None = None
    while True:
        page = reader.page(query, after=after, limit=MAX_PAGE_SIZE)
        rows.extend(strict_row("link", row) for row in page)
        if len(page) < MAX_PAGE_SIZE:
            return rows
        after = page[-1]["object_id"]


def committed_memory(writer: Writer, memory_id: str) -> MemoryRowV2 | None:
    """Project one Item from its committed rows.

    Args:
        writer: The sole owner, under its mutex.
        memory_id: The Item.

    Returns:
        The Item's projection, or None when it no longer exists.
    """
    raw = writer.get_row("item", memory_id)
    if raw is None:
        return None
    item = strict_row("item", raw)

    def revision(identity: str) -> Row | None:
        found = writer.get_row("revision", identity)
        return None if found is None else strict_row("revision", found)

    return MemoryRowV2.from_rows(
        item, version_chain(item, revision), lambda identity: revision_links(writer, identity),
    )


def cold_projections(rows: Callable[[str], Iterable[Row]]) -> ColdProjections:
    """Project every committed Item, active entity, spent declaration, name and hold.

    Args:
        rows: Every stored row of a kind.

    Returns:
        The NL projections of a cold root.
    """
    revisions = {
        row["object_id"]: row for row in (strict_row("revision", raw) for raw in rows("revision"))
    }
    links: dict[str, list[dict[str, Any]]] = {}
    for raw in rows("link"):
        link = strict_row("link", raw)
        links.setdefault(link["source_revision_id"], []).append(link)

    def carried(identity: str) -> list[dict[str, Any]]:
        return links.get(identity, [])

    memories: list[MemoryRowV2] = []
    for raw in rows("item"):
        item = strict_row("item", raw)
        memories.append(MemoryRowV2.from_rows(item, version_chain(item, revisions.get), carried))
    entities = tuple(
        EntityRowV1.from_row(row) if row["state"] == "active" else SpentMintV1.from_row(row)
        for row in (strict_row("entity", raw) for raw in rows("entity"))
        if row["state"] == "active" or row["mint_key"] is not None
    )
    names = tuple(EntityNameRowV1.from_row(strict_row("entity_name", raw))
                  for raw in rows("entity_name"))
    held = tuple(
        HeldEvidenceV1.from_candidate(row)
        for row in (strict_row("candidate", raw) for raw in rows("candidate"))
        if row["hold_evidence"]
    )
    return ColdProjections(tuple(memories), entities, names, held)


def commit_delta(writer: Writer, root: RecallSnapshot, commit: CommittedRows) -> CommitDelta:
    """Prove one committed transaction's rows against storage and project its changes.

    Every written row must equal its stored post-image and every deleted row
    must be gone. An Item is re-projected from its committed rows when the
    transaction wrote or deleted it, one of its revisions or a link one of its
    revisions carries.

    Args:
        writer: The sole owner, under its mutex, right after the commit.
        root: The root the delta applies to.
        commit: The committed transaction.

    Returns:
        The projection changes; a deletion only for a projected row.

    Raises:
        AMLError: ``parity_mismatch`` when storage differs from the committed
            rows or a written link lost its source revision.
    """
    sources: list[SourceRowV2 | SourceDeletion] = []
    entities: list[EntityRowV1 | EntityDeletion | SpentMintV1] = []
    names: list[EntityNameRowV1 | EntityNameDeletion] = []
    held: list[HeldEvidenceV1 | HeldEvidenceRelease] = []
    memory_ids: set[str] = set()
    revisions: dict[str, Row] = {}
    links: list[Row] = []
    for kind, row, deleted in (
        *((kind, row, False) for kind, row in commit.written),
        *((kind, row, True) for kind, row in commit.deleted),
    ):
        identity = row["object_id"]
        stored = writer.get_row(kind, identity)
        if (stored is not None) if deleted else (stored != row):
            raise AMLError("parity_mismatch", "Committed rows differ from storage")
        if kind == "evidence":
            sources.append(SourceDeletion(identity) if deleted else SourceRowV2.from_evidence(row))
        elif kind == "item":
            memory_ids.add(identity)
        elif kind == "revision":
            memory_ids.add(row["memory_id"])
            revisions[identity] = row
        elif kind == "link":
            links.append(row)
        elif kind == "entity":
            if not deleted and row["state"] == "active":
                entities.append(EntityRowV1.from_row(row))
            elif not deleted and row["mint_key"] is not None:
                # Also for an entity minted and erased in this one transaction.
                entities.append(SpentMintV1.from_row(row))
            elif root.get_entity(identity) is not None:
                entities.append(EntityDeletion(identity))
        elif kind == "entity_name":
            names.append(EntityNameDeletion(identity) if deleted else EntityNameRowV1.from_row(row))
        elif kind == "candidate":
            if not deleted and row["hold_evidence"]:
                held.append(HeldEvidenceV1.from_candidate(row))
            elif root.get_hold(identity) is not None:
                held.append(HeldEvidenceRelease(identity))
    for link in links:
        source = revisions.get(link["source_revision_id"]) or writer.get_row(
            "revision", link["source_revision_id"],
        )
        if source is None:
            raise AMLError("parity_mismatch", "A committed link lost its source revision")
        memory_ids.add(source["memory_id"])
    memories: list[MemoryRowV2 | MemoryDeletion] = []
    for identity in sorted(memory_ids):
        memory = committed_memory(writer, identity)
        if memory is not None:
            memories.append(memory)
        elif root.get_memory(identity) is not None:
            memories.append(MemoryDeletion(identity))
    return CommitDelta(
        tuple(sources), tuple(memories), tuple(entities), tuple(names), tuple(held),
    )
