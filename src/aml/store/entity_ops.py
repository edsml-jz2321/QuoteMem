"""Writer memory operations on Items, links, entities and names (A.3, C.9.6, D.7.3; I11).

These build the rows of one NL transaction; the single Writer commits them
(`Writer.commit_rows`). `TxnRows` collects a transaction's post-images and
deletes over the stored rows, so several operations compose in one
transaction and each reads what the earlier ones wrote. Nothing here calls a
provider or decides semantics: the commit (C.12), forget (D.3, S6a) and
privacy erase (D.7, S6b) choose what to write, and these operations write it
the one way A.3 allows:

- names: every transaction that changes an Item's head or lifecycle deletes
  the old head's ``entity_name`` rows and derives the new head's, one per
  naming annotation of an ``active`` Item's readable head (A.3.7, DM-05);
- entities: an extraction-declared entity is get-or-create on its
  ``mint_key`` and a participant on its ``participant_key`` (C.9.6, A.3.6);
- `requote` writes the chosen version's content on its surviving support and
  carries the head's ``supersedes``, ``contradicts`` and ``corrects_history``
  links; `erase_head` writes the content-free head and carries nothing
  (A.3.3, D.7.3);
- `erase_unanchored_entities` erases an extraction entity no non-erased
  memory anchors any more, and a participant that also lost its last source
  (A.3.6, INV P05); SDK entities never erase in v1;
- `audit_entity_names` rebuilds every name from the heads and compares
  (A.3.7; C.13 boot step 17).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final

from aml.core.entities import LinkType, MemorySnapshotV2, QuoteSpanV1
from aml.core.errors import AMLError
from aml.core.identity import integrity_digest, new_ulid
from aml.core.semantic import ScopeKind
from aml.core.statement import MAX_MEMORY_SPANS, MAX_SUPPORT_GROUPS, name_norm
from aml.store.domain import DOMAIN_ADAPTERS
from aml.store.indexed import (
    MAX_PAGE_SIZE,
    EntityNameRows,
    IndexedQuery,
    ItemRows,
    LinkTargetRows,
    RevisionLinks,
)
from aml.store.row_proofs import activation_mode

if TYPE_CHECKING:
    from aml.store.writer import RowDelete, RowWrite, Writer

Row = dict[str, Any]
_REQUOTE_CARRY: Final = frozenset({"supersedes", "contradicts", "corrects_history"})
_MERGE_CARRY: Final = frozenset({"supersedes", "contradicts"})


# A transaction's rows over the stored ones ------------------------------------------------------


@dataclass
class TxnRows:
    """The post-images and deletes of one NL transaction, read over the stored rows.

    Attributes:
        writer: The single Writer, whose stored rows are read.
        scope_kind: The transaction's scope kind.
        scope_id: The transaction's scope.
        commit_seq: The ``commit_seq`` the transaction will take; every row it
            writes with a ``commit_seq`` carries it.
        now: The transaction's system time in epoch milliseconds.
        run_id: The Run that commits it: revisions' ``origin_run_id`` and new
            entities' ``created_by_run_id``.
    """

    writer: Writer
    scope_kind: ScopeKind
    scope_id: str
    commit_seq: int
    now: int
    run_id: str
    written: dict[tuple[str, str], Row] = field(default_factory=dict[tuple[str, str], Row])
    deleted: dict[tuple[str, str], Row] = field(default_factory=dict[tuple[str, str], Row])

    def get(self, kind: str, object_id: str) -> Row | None:
        """Return a row as the transaction leaves it: written, deleted (None) or stored."""
        key = (kind, object_id)
        if key in self.deleted:
            return None
        found = self.written.get(key)
        return dict(found) if found is not None else self.writer.get_row(kind, object_id)

    def put(self, kind: str, row: Row) -> Row:
        """Write a complete strict post-image; a later write of one row replaces it."""
        image = DOMAIN_ADAPTERS[kind].validate_python(row).model_dump(mode="json")
        key = (kind, image["object_id"])
        if key in self.deleted:
            raise AMLError("invalid_request", "A transaction writes and deletes the same row")
        self.written[key] = image
        return image

    def delete(self, kind: str, object_id: str) -> None:
        """Delete a row: a stored one is deleted, one the transaction wrote is dropped."""
        key = (kind, object_id)
        written = self.written.pop(key, None) is not None
        stored = self.writer.get_row(kind, object_id)
        if stored is not None:
            self.deleted[key] = stored
        elif not written:
            raise AMLError("reference_missing", "A deleted row does not exist")

    def pages(self, query: IndexedQuery) -> Iterator[Row]:
        """Every stored row of one bounded indexed query, page by page (A0.3)."""
        reader = self.writer.indexed_reader()
        after: str | None = None
        while True:
            page = reader.page(query, after=after, limit=MAX_PAGE_SIZE)
            yield from page
            if len(page) < MAX_PAGE_SIZE:
                return
            after = page[-1]["object_id"]

    def merged(self, kind: str, stored: Iterable[Row], keep: Callable[[Row], bool]) -> list[Row]:
        """Stored rows as the transaction leaves them, plus its own writes, that ``keep`` admits."""
        found: dict[str, Row] = {}
        for row in stored:
            current = self.get(kind, row["object_id"])
            if current is not None and keep(current):
                found[current["object_id"]] = current
        for (written_kind, identity), row in self.written.items():
            if written_kind == kind and keep(row):
                found[identity] = dict(row)
        return [found[identity] for identity in sorted(found)]

    def links_of(self, revision_id: str) -> list[Row]:
        """Every link a revision carries, in object-id order."""
        return self.merged("link", self.pages(RevisionLinks(source_revision_id=revision_id)),
                            lambda row: row["source_revision_id"] == revision_id)

    def links_to(self, link_type: LinkType, target_key: str) -> list[Row]:
        """Every link of one type that targets one key in the scope."""
        query = LinkTargetRows(scope_kind=self.scope_kind, scope_id=self.scope_id,
                               link_type=link_type, target_key=target_key)
        return self.merged("link", self.pages(query), lambda row: (
            row["link_type"] == link_type and row["target_key"] == target_key
        ))

    def names_of(self, entity_id: str) -> list[Row]:
        """Every ``entity_name`` row of one entity in the scope."""
        query = EntityNameRows(scope_kind=self.scope_kind, scope_id=self.scope_id,
                               entity_id=entity_id)
        return self.merged("entity_name", self.pages(query),
                            lambda row: row["entity_id"] == entity_id)

    def items_of(self, entity_id: str) -> list[Row]:
        """Every Item whose primary anchor is the entity (its head is not erased)."""
        query = ItemRows(scope_kind=self.scope_kind, scope_id=self.scope_id, entity_id=entity_id)
        return self.merged("item", self.pages(query), lambda row: row["entity_id"] == entity_id)

    def row_writes(self) -> list[RowWrite]:
        """The transaction's post-images as Writer row writes, in write order."""
        from aml.store.writer import RowWrite

        return [RowWrite(kind, DOMAIN_ADAPTERS[kind].validate_python(row))
                for (kind, _identity), row in self.written.items()]

    def row_deletes(self) -> list[RowDelete]:
        """The transaction's deletes as Writer row deletes."""
        from aml.store.writer import RowDelete

        return [RowDelete(kind, identity) for kind, identity in self.deleted]

    def stamp(self) -> None:
        """Give the transaction, and every row it writes, the next ``commit_seq``.

        The Writer refuses a row naming another transaction's ``commit_seq``
        (``parity_mismatch``), and any scope's commit advances the global
        sequence, so a sequence read while the rows were built can be overtaken
        before they reach the Writer. Every producer stamps under the Writer
        mutex, immediately before the commit, or its sizing, that takes the
        transaction; a new candidate's ``created_commit_seq`` moves with it.

        Raises:
            AMLError: ``reader_busy`` outside the Writer mutex.
        """
        writer = self.writer
        if not writer.mutex._is_owned():  # pyright: ignore[reportPrivateUsage]
            raise AMLError("reader_busy", "A transaction takes its commit_seq under the Writer")
        sequence = writer.counter("commit_seq") + 1
        for (kind, identity), row in self.written.items():
            if "commit_seq" in row:
                row["commit_seq"] = sequence
            if kind == "candidate" and writer.get_row("candidate", identity) is None:
                row["created_commit_seq"] = sequence
        self.commit_seq = sequence


# Names (A.3.7, DM-05) -----------------------------------------------------------------------------


def name_rows(txn: TxnRows, item: Row, head: Row) -> list[Row]:
    """Derive the ``entity_name`` rows of an Item's head: one per naming annotation.

    Args:
        txn: The transaction; the rows carry its ``commit_seq`` and time.
        item: The Item's post-image.
        head: Its head revision's post-image.

    Returns:
        The rows of an ``active`` Item with a readable head, else none.
    """
    snapshot = head["snapshot"]
    if item["lifecycle_state"] != "active" or not snapshot["statement"]:
        return []
    return [{
        "object_id": new_ulid(), "scope_kind": item["scope_kind"], "scope_id": item["scope_id"],
        "entity_id": annotation["entity_id"], "name": annotation["name"],
        "name_norm": name_norm(annotation["name"]), "name_kind": annotation["name_kind"],
        "naming_memory_id": item["object_id"], "naming_revision_id": head["object_id"],
        "annotation_index": index, "commit_seq": txn.commit_seq, "created_at": txn.now,
    } for index, annotation in enumerate(snapshot["naming"])]


def rederive_entity_names(txn: TxnRows, memory_id: str, *, before: Row | None) -> None:
    """Re-derive one Item's names after the transaction changed its head or lifecycle (I11).

    The rows of its previous head are deleted and its current head's derived,
    so supersession, retraction, requote, erase and forget all leave the names
    equal to the A.3.7 derivation.

    Args:
        txn: The transaction, holding the Item's post-image (absent when deleted).
            One transaction changes an Item's head at most once.
        memory_id: The Item.
        before: The Item's stored row before the transaction; None for a new Item.
    """
    item = txn.get("item", memory_id)
    if before is not None and item is not None and (
        before["current_revision_id"], before["lifecycle_state"],
    ) == (item["current_revision_id"], item["lifecycle_state"]):
        return
    if before is not None:
        old = txn.writer.get_row("revision", before["current_revision_id"])
        if old is not None and old["snapshot"]["statement"]:
            for entity_id in sorted({entry["entity_id"] for entry in old["snapshot"]["naming"]}):
                for row in txn.names_of(entity_id):
                    if row["naming_revision_id"] == old["object_id"]:
                        txn.delete("entity_name", row["object_id"])
    if item is None:
        return
    head = txn.get("revision", item["current_revision_id"])
    if head is None:
        raise AMLError("reference_missing", "An Item lost its head revision")
    for row in name_rows(txn, item, head):
        txn.put("entity_name", row)


def _name_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (row["naming_revision_id"], row["annotation_index"], row["entity_id"], row["name"],
            row["name_norm"], row["name_kind"], row["naming_memory_id"], row["scope_kind"],
            row["scope_id"])


def audit_entity_names(writer: Writer) -> None:
    """Rebuild every ``entity_name`` row from the heads and compare (A.3.7; C.13 step 17).

    Args:
        writer: The single Writer, before readers open.

    Raises:
        AMLError: ``parity_mismatch`` when a stored name differs from the
            derivation, one is missing or one is extra; fail closed.
    """
    expected: set[tuple[Any, ...]] = set()
    for item in writer.iter_rows("item"):
        if item["lifecycle_state"] != "active":
            continue
        head = writer.get_row("revision", item["current_revision_id"])
        if head is None:
            raise AMLError("parity_mismatch", "An Item lost its head revision")
        snapshot = head["snapshot"]
        if not snapshot["statement"]:
            continue
        for index, annotation in enumerate(snapshot["naming"]):
            expected.add(_name_key({
                "naming_revision_id": head["object_id"], "annotation_index": index,
                "entity_id": annotation["entity_id"], "name": annotation["name"],
                "name_norm": name_norm(annotation["name"]), "name_kind": annotation["name_kind"],
                "naming_memory_id": item["object_id"], "scope_kind": item["scope_kind"],
                "scope_id": item["scope_id"],
            }))
    stored = [_name_key(row) for row in writer.iter_rows("entity_name")]
    if len(set(stored)) != len(stored) or set(stored) != expected:
        raise AMLError("parity_mismatch", "Stored entity names differ from their derivation")


# Entities (C.9.6, A.3.6) --------------------------------------------------------------------------


def get_or_mint(
    txn: TxnRows, *, mint_key: str, entity_type: str, existing: str | None,
) -> str:
    """Return the entity of an extraction declaration, minting it on first commit (C.9.6).

    Args:
        txn: The transaction.
        mint_key: The declaration's key (A0.6).
        entity_type: The declared type after D.8.4, used only when minting.
        existing: The active entity minted for the key, as the commit's pin read it.

    Returns:
        The entity id; a new active entity row is written when none exists.
    """
    if existing is not None:
        return existing
    for (kind, identity), row in txn.written.items():
        if kind == "entity" and row["mint_key"] == mint_key:
            return identity
    entity_id = new_ulid()
    txn.put("entity", {
        "object_id": entity_id, "scope_kind": txn.scope_kind, "scope_id": txn.scope_id,
        "entity_type": entity_type, "participant_key": None, "mint_key": mint_key,
        "state": "active", "created_by_run_id": txn.run_id, "commit_seq": txn.commit_seq,
        "created_at": txn.now, "updated_at": txn.now,
    })
    return entity_id


def get_or_create_participant(
    txn: TxnRows, *, participant_key: str, existing: str | None,
) -> str:
    """Return a participant's entity, creating it under its unique key (C.9.6, D.9).

    Args:
        txn: The transaction.
        participant_key: The participant's key (A0.6).
        existing: The active participant entity with the key, as the commit read it.

    Returns:
        The entity id; a new ``person`` entity is written when none exists.
    """
    if existing is not None:
        return existing
    for (kind, identity), row in txn.written.items():
        if kind == "entity" and row["participant_key"] == participant_key:
            return identity
    entity_id = new_ulid()
    txn.put("entity", {
        "object_id": entity_id, "scope_kind": txn.scope_kind, "scope_id": txn.scope_id,
        "entity_type": "person", "participant_key": participant_key, "mint_key": None,
        "state": "active", "created_by_run_id": txn.run_id, "commit_seq": txn.commit_seq,
        "created_at": txn.now, "updated_at": txn.now,
    })
    return entity_id


# Revisions and links ------------------------------------------------------------------------------


def revision_row(
    txn: TxnRows, *, memory_id: str, previous: Row | None, op: str, reason: str,
    snapshot: MemorySnapshotV2, source_candidate_id: str | None, policy_release_ref: str,
) -> Row:
    """Build one new immutable revision (A.3.2): the next version of its Item."""
    image = snapshot.model_dump(mode="json")
    return {
        "object_id": new_ulid(), "memory_id": memory_id,
        "version": 1 if previous is None else previous["version"] + 1,
        "previous_revision_id": None if previous is None else previous["object_id"],
        "op": op, "reason_code": reason, "snapshot": image,
        "content_hash": str(integrity_digest(snapshot)),
        "evidence_ids": sorted({span["evidence_id"] for span in image["quote_spans"]}),
        "source_candidate_id": source_candidate_id, "origin_run_id": txn.run_id,
        "policy_release_ref": policy_release_ref, "commit_seq": txn.commit_seq,
        "recorded_at": txn.now,
    }


def head_item(txn: TxnRows, item: Row | None, head: Row, *, memory_id: str) -> Row:
    """Return an Item's post-image with a new head: keys and activation follow the head (DM-02)."""
    snapshot = head["snapshot"]
    if item is None:
        lifecycle = "historical" if head["reason_code"] == "historical_correction" else "active"
        base: Row = {"object_id": memory_id, "scope_kind": txn.scope_kind,
                     "scope_id": txn.scope_id, "lifecycle_state": lifecycle,
                     "retired_at": None if lifecycle == "active" else txn.now,
                     "created_at": txn.now, "activation_mode": "contextual"}
    else:
        base = dict(item)
    erased = not snapshot["statement"]
    return {
        **base,
        "current_revision_id": head["object_id"],
        "entity_id": snapshot["entity_id"],
        "statement_digest": snapshot["statement_digest"],
        "activation_mode": base["activation_mode"] if erased else activation_mode(snapshot),
        "commit_seq": txn.commit_seq,
        "updated_at": txn.now,
    }


def link_row(txn: TxnRows, source: Row, link_type: str, target_key: str, **fields: Any) -> Row:
    """Build one link carried by a new revision (A.3.3)."""
    return {
        "object_id": new_ulid(), "scope_kind": txn.scope_kind, "scope_id": txn.scope_id,
        "source_revision_id": source["object_id"], "link_type": link_type,
        "target_key": target_key, "relation": None, "effective_at": None,
        "effective_basis": None, "verdict_ref": None, "correction_requested": False,
        "created_at": txn.now, **fields,
    }


def carry_links(
    txn: TxnRows, old_head: Row, new_head: Row, types: frozenset[str],
) -> list[Row]:
    """Copy the old head's links of some types onto the new head (A.3.3 carry).

    Returns:
        The carried links' post-images.
    """
    return [txn.put("link", {
        **row, "object_id": new_ulid(), "source_revision_id": new_head["object_id"],
        "created_at": txn.now,
    }) for row in txn.links_of(old_head["object_id"]) if row["link_type"] in types]


def project_about(txn: TxnRows, head: Row) -> None:
    """Write one ``about_entity`` link per snapshot ``about_entity_ids`` entry (DM-04)."""
    for entity_id in head["snapshot"]["about_entity_ids"]:
        txn.put("link", link_row(txn, head, "about_entity", entity_id))


def support_union(
    base_spans: Sequence[QuoteSpanV1], base_groups: Sequence[Sequence[int]],
    added: Iterable[Sequence[QuoteSpanV1]],
) -> tuple[list[QuoteSpanV1], list[list[int]]]:
    """The union of a version's support and added groups, re-indexed (A.3.2 ``merge``)."""
    def span_key(span: QuoteSpanV1) -> tuple[str, int, int]:
        return (span.evidence_id, span.byte_start, span.byte_end)

    groups = [[span_key(base_spans[index]) for index in group] for group in base_groups]
    groups.extend([span_key(span) for span in group] for group in added)
    keys = sorted({key for group in groups for key in group})
    index = {key: number for number, key in enumerate(keys)}
    indexed = sorted({tuple(sorted(index[key] for key in group)) for group in groups})
    ordered = [QuoteSpanV1(evidence_id=evidence, byte_start=start, byte_end=end)
               for evidence, start, end in keys]
    return ordered, [list(group) for group in indexed]


def restore_support(
    txn: TxnRows, memory_id: str, *, added: Sequence[Sequence[QuoteSpanV1]],
    source_candidate_id: str, policy_release_ref: str,
) -> Row:
    """Merge an exact restatement's groups into an unreadable head (B.8 #2b; D.2; A.3.2).

    The new head (op ``merge``, reason ``same_support``) keeps the head's
    content and ``verdict_ref`` with the union of its groups and ``added``; its
    withheld groups stay unusable, so only the new ones render. It carries the
    head's ``supersedes`` and ``contradicts`` links, re-projects its
    ``about_entity`` links and re-derives its names.

    Args:
        txn: The transaction.
        memory_id: The active Item whose head fails P-VER.
        added: The restatement's admitted groups.
        source_candidate_id: The committing candidate.
        policy_release_ref: The policy release the transaction is written under.

    Returns:
        The new head revision's post-image.

    Raises:
        AMLError: ``reference_missing`` for an Item without its head;
            ``budget_exceeded`` when the union exceeds the support bounds (A0.7),
            which settlement rules out before planning the merge.
    """
    before = txn.get("item", memory_id)
    head = None if before is None else txn.get("revision", before["current_revision_id"])
    if before is None or head is None:
        raise AMLError("reference_missing", "A restored Item lost its head")
    base = MemorySnapshotV2.model_validate(head["snapshot"])
    spans, groups = support_union(base.quote_spans, base.support_groups, added)
    if len(groups) > MAX_SUPPORT_GROUPS or len(spans) > MAX_MEMORY_SPANS:
        raise AMLError("budget_exceeded", "A restored head's support exceeds its bounds")
    snapshot = MemorySnapshotV2.model_validate({
        **base.model_dump(mode="json"),
        "quote_spans": [span.model_dump(mode="json") for span in spans],
        "support_groups": groups,
        "last_confirmed_at": confirmed_at(txn, (span.evidence_id for span in spans)),
    })
    revision = txn.put("revision", revision_row(
        txn, memory_id=memory_id, previous=head, op="merge", reason="same_support",
        snapshot=snapshot, source_candidate_id=source_candidate_id,
        policy_release_ref=policy_release_ref,
    ))
    txn.put("item", head_item(txn, before, revision, memory_id=memory_id))
    carry_links(txn, head, revision, _MERGE_CARRY)
    project_about(txn, revision)
    rederive_entity_names(txn, memory_id, before=txn.writer.get_row("item", memory_id))
    return revision


# I11: requote and erase (D.7.3) -------------------------------------------------------------------


def requote(
    txn: TxnRows, memory_id: str, *, version: Row, support: Sequence[Sequence[int]],
    policy_release_ref: str,
) -> Row:
    """Requote an Item from a readable version on its surviving support (I11; D.7.3).

    The new head (op ``requote``, reason ``privacy_requote``) is the chosen
    version's content restricted to the surviving groups, with that version's
    ``verdict_ref``. It carries the current head's ``supersedes``,
    ``contradicts`` and ``corrects_history`` links, re-projects its
    ``about_entity`` links, keeps the Item's lifecycle and re-derives its names.

    Args:
        txn: The transaction.
        memory_id: The Item.
        version: The chosen version's revision row: the head or an earlier,
            still readable version of the Item (S6b chooses it).
        support: The surviving groups, each a list of indexes into the
            version's ``quote_spans``; at least one.
        policy_release_ref: The policy release the transaction is written under.

    Returns:
        The new head revision's post-image.

    Raises:
        AMLError: ``binding_mismatch`` when the version is not one of the
            Item's, or no group survives.
    """
    source = MemorySnapshotV2.model_validate(version["snapshot"])
    return requote_on(txn, memory_id, version=version, policy_release_ref=policy_release_ref,
                      support=[[source.quote_spans[index] for index in group]
                               for group in support])


def requote_on(
    txn: TxnRows, memory_id: str, *, version: Row, support: Sequence[Sequence[QuoteSpanV1]],
    policy_release_ref: str, keywords: Sequence[str] | None = None,
) -> Row:
    """Requote an Item from a readable version on the given quote spans (I11; D.7.3).

    As `requote`, with each surviving group given as its spans: whole quotes of the
    version, or the readable parts of its quotes a forget's dependent keeps (D.7.3 #4a),
    and, when given, the keywords of the version that the dependent keeps.

    Args:
        txn: The transaction.
        memory_id: The Item.
        version: The chosen version's revision row.
        support: The surviving groups, each a list of spans inside the version's quotes'
            Evidence; at least one.
        policy_release_ref: The policy release the transaction is written under.
        keywords: The version's keywords the new head keeps (a subset, in order); None
            keeps them all.

    Returns:
        The new head revision's post-image.

    Raises:
        AMLError: ``binding_mismatch`` when the version is not one of the Item's, no
            group survives, or ``keywords`` is not a subset of the version's.
    """
    before = txn.get("item", memory_id)
    if before is None or version["memory_id"] != memory_id or not support or not all(support):
        raise AMLError("binding_mismatch", "A requote keeps a readable version of its Item")
    head = txn.get("revision", before["current_revision_id"])
    if head is None:
        raise AMLError("reference_missing", "An Item lost its head revision")
    source = MemorySnapshotV2.model_validate(version["snapshot"])
    if keywords is not None and not set(keywords) <= set(source.keywords):
        raise AMLError("binding_mismatch", "A requote keeps a subset of the version's keywords")

    def key(span: QuoteSpanV1) -> tuple[str, int, int]:
        return span.evidence_id, span.byte_start, span.byte_end

    spans = sorted({key(span): span for group in support for span in group}.values(), key=key)
    position = {key(span): index for index, span in enumerate(spans)}
    snapshot = MemorySnapshotV2.model_validate({
        **source.model_dump(mode="json"),
        "quote_spans": [span.model_dump(mode="json") for span in spans],
        "support_groups": [list(group) for group in sorted({
            tuple(sorted(position[key(span)] for span in group)) for group in support})],
        "keywords": list(source.keywords if keywords is None else keywords),
        "last_confirmed_at": confirmed_at(txn, (span.evidence_id for span in spans)),
    })
    revision = txn.put("revision", revision_row(
        txn, memory_id=memory_id, previous=head, op="requote", reason="privacy_requote",
        snapshot=snapshot, source_candidate_id=None, policy_release_ref=policy_release_ref,
    ))
    txn.put("item", head_item(txn, before, revision, memory_id=memory_id))
    carry_links(txn, head, revision, _REQUOTE_CARRY)
    project_about(txn, revision)
    rederive_entity_names(txn, memory_id, before=txn.writer.get_row("item", memory_id))
    return revision


def requoted_from(revision: Mapping[str, Any], previous: Mapping[str, Any]) -> bool:
    """Whether ``revision`` is a privacy requote of ``previous`` stating the same thing
    (D.7.3): op ``requote``, reason ``privacy_requote``, the same anchor and statement. A
    forget's compensation writes it on readable support (steps 2-4a) or keywords (#4b)."""
    new, old = revision["snapshot"], previous["snapshot"]
    return ((revision["op"], revision["reason_code"]) == ("requote", "privacy_requote")
            and revision["previous_revision_id"] == previous["object_id"]
            and (new["entity_id"], new["statement_digest"], new["statement"])
            == (old["entity_id"], old["statement_digest"], old["statement"]))


def rekey(
    txn: TxnRows, memory_id: str, *, keywords: Sequence[str], policy_release_ref: str,
) -> Row:
    """A forget's keyword-only revision of an Item's head (D.7.3 #4b, KW-1).

    The new head (op ``requote``, reason ``privacy_requote``) is the current head with
    ``keywords``, a proper subset of its keywords in their order; its statement, quotes,
    groups, validity, vector, anchors, naming, verdict and ``last_confirmed_at`` are
    unchanged. It carries the head's links and keeps the Item's lifecycle, as a requote
    does.

    Args:
        txn: The forget's P2 transaction.
        memory_id: The Item.
        keywords: The keywords the new head keeps.
        policy_release_ref: The policy release the transaction is written under.

    Returns:
        The new head revision's post-image.

    Raises:
        AMLError: ``binding_mismatch`` when the Item or its head is missing or
            ``keywords`` drops nothing or is not a subset of the head's, in order.
    """
    before = txn.get("item", memory_id)
    head = None if before is None else txn.get("revision", before["current_revision_id"])
    if before is None or head is None:
        raise AMLError("binding_mismatch", "A keyword-only revision needs its Item's head")
    source = MemorySnapshotV2.model_validate(head["snapshot"])
    if list(keywords) != [keyword for keyword in source.keywords if keyword in keywords] or (
            len(keywords) == len(source.keywords)):
        raise AMLError("binding_mismatch", "A keyword-only revision drops some of its keywords")
    snapshot = MemorySnapshotV2.model_validate({
        **source.model_dump(mode="json"), "keywords": list(keywords),
    })
    revision = txn.put("revision", revision_row(
        txn, memory_id=memory_id, previous=head, op="requote", reason="privacy_requote",
        snapshot=snapshot, source_candidate_id=None, policy_release_ref=policy_release_ref,
    ))
    txn.put("item", head_item(txn, before, revision, memory_id=memory_id))
    carry_links(txn, head, revision, _REQUOTE_CARRY)
    project_about(txn, revision)
    rederive_entity_names(txn, memory_id, before=txn.writer.get_row("item", memory_id))
    return revision


def erase_head(txn: TxnRows, memory_id: str, *, policy_release_ref: str) -> Row:
    """Write an Item's content-free head (I11; D.7.3): op ``erase``, nothing carried.

    An ``active`` Item becomes ``erased``; an ended or ``historical`` Item keeps
    its lifecycle. Earlier revisions and their links stay immutable, so a
    ``corrects_history`` row on them still hides its target (A.3.3, C28).

    Args:
        txn: The transaction.
        memory_id: The Item.
        policy_release_ref: The policy release the transaction is written under.

    Returns:
        The erased head revision's post-image.

    Raises:
        AMLError: ``binding_mismatch`` for an Item that does not exist or is
            already erased.
    """
    before = txn.get("item", memory_id)
    if before is None or before["lifecycle_state"] == "erased":
        raise AMLError("binding_mismatch", "Only a readable head is erased")
    head = txn.get("revision", before["current_revision_id"])
    if head is None:
        raise AMLError("reference_missing", "An Item lost its head revision")
    snapshot = MemorySnapshotV2.model_validate(head["snapshot"]).erase()
    revision = txn.put("revision", revision_row(
        txn, memory_id=memory_id, previous=head, op="erase", reason="privacy_erase",
        snapshot=snapshot, source_candidate_id=None, policy_release_ref=policy_release_ref,
    ))
    item = head_item(txn, before, revision, memory_id=memory_id)
    if before["lifecycle_state"] == "active":
        item.update(lifecycle_state="erased", retired_at=txn.now)
    txn.put("item", item)
    rederive_entity_names(txn, memory_id, before=txn.writer.get_row("item", memory_id))
    return revision


def anchored(txn: TxnRows, entity_id: str) -> bool:
    """Whether a non-erased memory still anchors the entity, as primary or about (A.3.6).

    Every lifecycle counts; an about anchor counts through an active
    ``about_entity`` link, one whose source revision is the head of a
    non-erased Item.
    """
    if txn.items_of(entity_id):
        return True
    for link in txn.links_to("about_entity", entity_id):
        source = txn.get("revision", link["source_revision_id"])
        item = None if source is None else txn.get("item", source["memory_id"])
        if item is not None and item["current_revision_id"] == link["source_revision_id"] and (
            item["lifecycle_state"] != "erased"
        ):
            return True
    return False


def erase_unanchored_entities(
    txn: TxnRows, entity_ids: Iterable[str], *,
    participant_has_source: Callable[[Row], bool],
) -> list[str]:
    """Erase the entities that lost their last anchor (I11; A.3.6, INV P05).

    An extraction-declared entity is erased once no non-erased memory anchors
    it; a participant once no memory anchors it and its last source in the
    scope is erased; SDK entities never erase in v1. An erased entity is
    terminal: its type is cleared, and a participant's key too.

    Args:
        txn: The transaction, holding the rows it already changed.
        entity_ids: The entities its changes may have left unanchored.
        participant_has_source: Whether a participant still has a non-erased
            source in the scope (S6b's Evidence check).

    Returns:
        The erased entities, sorted.
    """
    erased: list[str] = []
    for entity_id in sorted(set(entity_ids)):
        row = txn.get("entity", entity_id)
        if row is None or row["state"] != "active" or anchored(txn, entity_id):
            continue
        if row["mint_key"] is None and row["participant_key"] is None:
            continue  # an SDK entity
        if row["participant_key"] is not None and participant_has_source(row):
            continue
        txn.put("entity", {**row, "entity_type": "", "participant_key": None, "state": "erased",
                           "commit_seq": txn.commit_seq, "updated_at": txn.now})
        erased.append(entity_id)
    return erased


def confirmed_at(txn: TxnRows, evidence_ids: Iterable[str]) -> int | None:
    """``last_confirmed_at`` of a written version: the latest ``ingested_at`` of its
    support Evidence (A.3.2; R-Q9), read from the rows the transaction sees."""
    found: list[int] = []
    for evidence_id in sorted(set(evidence_ids)):
        row = txn.get("evidence", evidence_id)
        if row is None:
            raise AMLError("reference_missing", "A version cites Evidence that does not exist")
        found.append(row["ingested_at"])
    return max(found, default=None)


ENDED_LIFECYCLE: Final[Mapping[str, str]] = {
    "update": "superseded", "correct": "corrected", "retract": "retracted",
}
"""The lifecycle a ``supersedes`` relation ends its target in (A.3.1)."""
