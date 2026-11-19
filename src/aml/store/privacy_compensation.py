"""What a privacy erase changes beyond its planned rows: D.7.3 and C.13's manifest (S6b).

An erase's plan lists the Evidence it erases, the pending candidates it expires
and the tombstones it deletes. What depends on that Evidence is derived from
the stored rows instead (`erase_dependents`):

- every Item whose current head cites erased Evidence is compensated on S5b's
  I11 operations (`compensate_erased`): `choose_compensation`, the one D.7.3
  steps 1-4 implementation, decides; `requote` or `erase_head` applies it and
  re-derives the Item's names, so a name whose provenance expired is withdrawn
  physically (D.7.3, P-NAME). A head targeted by a historical correction keeps
  its revision identity while a complete group survives: its erased sources
  cannot appear in readable quotes, and replacing it would evade the correction
  and lose the historical stand-in. Step 5 then erases the entities left unanchored,
  including a participant whose last source is erased (A.3.6, D.9, INV P05);
- every judge record whose dependencies include erased Evidence, a version
  quoting it or a tombstone the erase deletes is deleted with its parts
  (`judge_record_groups`; C.13 erasure manifest, D.7.5, INV P06): the call
  manifest, each member's result index and accepted output, a result that
  borrows a deleted one, the ``judge_output`` locators naming them, and the
  neighbour snapshots, a judge child's or one an Add owner wrote for a quiet
  re-entered member;
- extraction replies whose inputs showed erased content, affected proposals
  and erased embedding entries leave with their unshared manifest parts. Content-free
  accounting preserves the original calls and positions for recovery; correction
  provenance stays on contradiction links. A pending or live proposal with
  surviving source support and no erased external context still continues under
  the existing fences;
- receipt-bound saved reception payloads and consent-invalidated unreceived
  payloads leave without deleting the original request's receipt or identity;
- after the planned batches, every record of an ended Add owner of the scope, or
  of its judge children, that holds a credential the reception detector
  recognizes leaves the same way, with the judge records naming it
  (`credential_records`, D.7.5): a model reply can state a credential its source
  spelled in a form the detector does not read.

Privacy erase writes no tombstone and no withheld span (UD4, P12, DM-10): the
business rows written here are Items, revisions, links, entity names, entities
and pending-candidate expiries; artifact witnesses retain no erased payload.
Nothing calls a provider. Every derivation is a function of the stored rows and
the erased set, so a transaction that did not commit is recomputed identically
on resume, and one that committed is never revisited: a compensated head cites
no erased Evidence, except a readable corrected target retaining its identity,
and a deleted record is gone. The Writer takes as many of
them as fit in each erase transaction (`Writer.apply_privacy_erase`).
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, cast

from aml.core.entities import MemorySnapshotV2
from aml.core.errors import AMLError
from aml.core.privacy import choose_compensation
from aml.core.readability import Readability
from aml.core.semantic import ScopeKind
from aml.core.statement import label_speaker_key, participant_key
from aml.store.domain import EvidenceRowV1
from aml.store.entity_ops import TxnRows, erase_head, erase_unanchored_entities, requote
from aml.store.indexed import (
    MAX_PAGE_SIZE,
    CandidateRows,
    EntityRows,
    EvidenceRows,
    IndexedQuery,
    ItemRows,
    LinkTargetRows,
    RevisionRows,
    ScopedRunRows,
    TombstoneRows,
)
from aml.store.memory_commit import readability

if TYPE_CHECKING:
    from aml.core.entities import RevisionReason
    from aml.core.extraction_context import ExtractContextV2
    from aml.core.judge_records import JudgeCallManifestV1, JudgeResultIndexV8
    from aml.core.neighbor_snapshot import NeighborSnapshotV1
    from aml.store.writer import RowDelete, RowWrite, Writer

Row = dict[str, Any]
RecordGroup = tuple[tuple[str, Row], ...]
"""Records deleted together: ``(kind, stored row)`` pairs, artifacts and Controls."""

ENDED_OWNER: Final = frozenset({"succeeded", "failed"})
"""Terminal owners no longer need their context-dependent extraction proposals."""


@dataclass(frozen=True, slots=True)
class _Version:
    """One stored revision as `choose_compensation` reads it."""

    object_id: str
    version: int
    reason_code: RevisionReason
    snapshot: MemorySnapshotV2


@dataclass(frozen=True, slots=True)
class ErasureDependents:
    """What an erase still changes because of the Evidence it erased; content-free.

    Attributes:
        items: The Items whose current head cites that Evidence, by id, except
            corrected heads with a surviving complete group.
        records: The judge and extraction records that depend on it, in groups
            that leave no surviving judge record naming a deleted one; each
            group is deleted in one transaction. Ordered by each group's first
            identity.
    """

    items: tuple[str, ...] = ()
    records: tuple[RecordGroup, ...] = ()

    def __bool__(self) -> bool:
        return bool(self.items or self.records)


@dataclass(frozen=True, slots=True)
class CompensationOutcome:
    """The content-free result of compensating some of an erase's Items.

    Attributes:
        rows: Item, revision, link, entity_name and entity post-images.
        deletes: The rows the operations deleted (old heads' names).
        items: The Items compensated, in the order given: a prefix of the
            requested ones.
        requoted: The Items requoted from a surviving group, sorted.
        erased_heads: The Items given the content-free head, sorted.
        erased_entities: The entities erased as unanchored, sorted.
        new_items: How many compensated Items no earlier transaction of this
            owner compensated (the receipt's ``item`` count is of distinct Items).
    """

    rows: tuple[RowWrite, ...]
    deletes: tuple[RowDelete, ...]
    items: tuple[str, ...]
    requoted: tuple[str, ...]
    erased_heads: tuple[str, ...]
    erased_entities: tuple[str, ...]
    new_items: int


def source_participant_keys(scope_id: str, row: Mapping[str, Any]) -> frozenset[str]:
    """The participant keys an Evidence row can be a source of (D.9).

    A row with a ``source_participant`` tuple speaks as that participant. A row
    without one may carry speaker line labels; every label the mechanical scan
    finds counts, a superset of the proven labels, so the check errs on keeping
    a labelled speaker (INV P05) rather than erasing one that still has a source.
    """
    from aml.ingestion.speaker_labels import scan_labels

    participant = row.get("source_participant")
    if participant is not None:
        return frozenset({participant_key(
            participant["namespace"], participant["conversation_id"],
            participant["participant_id"],
        )})
    text = row.get("text") or ""
    return frozenset(
        label_speaker_key(scope_id, match.name)
        for match in scan_labels(text, row.get("speaker_entity"), row.get("role"))
    )


def _pages(writer: Writer, query: IndexedQuery) -> Iterator[Row]:
    """Every stored row of one bounded indexed query, page by page (A0.3)."""
    reader = writer.indexed_reader()
    after: str | None = None
    while True:
        page = reader.page(query, after=after, limit=MAX_PAGE_SIZE)
        yield from page
        if len(page) < MAX_PAGE_SIZE:
            return
        after = page[-1]["object_id"]


def erase_dependents(
    writer: Writer, scope: tuple[str, str], erased: frozenset[str],
    tombstones: frozenset[str] = frozenset(),
    *, now: int,
) -> ErasureDependents:
    """Find what still depends on an erase's erased Evidence and deleted tombstones.

    A corrected head with a complete surviving group retains its revision
    identity. Its original erased-source locators remain immutable history;
    every read filters them through P-GROUP after the source is cleared. The
    erased set is excluded here even before this transaction clears its rows.

    Args:
        writer: The single Writer (read only).
        scope: The erase's exact ``(scope_kind, scope_id)``.
        erased: The Evidence the erase has erased, or erases in the transaction
            being built.
        tombstones: The tombstones it has deleted, or deletes in that transaction.
        now: The erase's trusted time for surviving-group readability.

    Returns:
        The Items to compensate and the judge and extraction record groups to
        delete.

    Raises:
        AMLError: ``reference_missing`` for an Item without its head;
            ``binding_mismatch`` for a judge or extraction record of another
            shape.
    """
    def evidence(identity: str) -> EvidenceRowV1 | None:
        if identity in erased:
            return None
        row = writer.get_row("evidence", identity)
        return None if row is None else EvidenceRowV1.model_validate(row)

    read = Readability(cast(ScopeKind, scope[0]), scope[1], now, evidence)
    items: list[str] = []
    for item in _pages(writer, ItemRows(scope_kind=cast(ScopeKind, scope[0]),
                                        scope_id=scope[1])):
        if item["lifecycle_state"] == "erased":
            continue
        head = writer.get_row("revision", item["current_revision_id"])
        if head is None:
            raise AMLError("reference_missing", "An Item lost its head revision")
        if not any(span["evidence_id"] in erased for span in head["snapshot"]["quote_spans"]):
            continue
        corrected = next(_pages(writer, LinkTargetRows(
            scope_kind=cast(ScopeKind, scope[0]), scope_id=scope[1],
            link_type="corrects_history", target_key=head["object_id"],
        )), None)
        # The correction targets this exact revision. Its lost source is erased,
        # and readers expose only surviving groups; a new head would revive it.
        if corrected is not None and read.p_ver(MemorySnapshotV2.model_validate(head["snapshot"])):
            continue
        items.append(item["object_id"])
    from aml.store.privacy_reception import reception_record_groups

    records = (*judge_record_groups(writer, scope, erased, tombstones),
               *extraction_record_groups(writer, scope, erased),
               *reception_record_groups(writer, scope, erased))
    return ErasureDependents(
        tuple(sorted(items)), tuple(sorted(records, key=lambda group: _identity(group[0]))),
    )


def _tombstoned(txn: TxnRows) -> Any:
    """Whether an ``(entity_id, statement_digest)`` pair has a stored tombstone (D.7.3 #3)."""
    cache: dict[str, frozenset[str]] = {}

    def check(entity_id: str, digest: str) -> bool:
        if digest not in cache:
            query = TombstoneRows(scope_kind=txn.scope_kind, scope_id=txn.scope_id,
                                  statement_digest=digest)
            cache[digest] = frozenset(
                row["entity_id"] for row in txn.pages(query)
                if row["statement_digest"] == digest
            )
        return entity_id in cache[digest]

    return check


def _versions(txn: TxnRows, memory_id: str) -> list[_Version]:
    stored = txn.pages(RevisionRows(memory_id=memory_id))
    return [
        _Version(row["object_id"], row["version"], row["reason_code"],
                 MemorySnapshotV2.model_validate(row["snapshot"]))
        for row in txn.merged("revision", stored, lambda row: row["memory_id"] == memory_id)
    ]


def _entities_of(txn: TxnRows, head: Row) -> set[str]:
    """The primary and ``about`` entities an Item's head anchors (A.3.6)."""
    found = {head["snapshot"]["entity_id"]}
    found.update(head["snapshot"].get("about_entity_ids", ()))
    found.update(
        link["target_key"] for link in txn.links_of(head["object_id"])
        if link["link_type"] == "about_entity"
    )
    found.discard("user")
    found.discard("")
    return found


def compensate_erased(
    writer: Writer,
    *,
    scope: tuple[str, str],
    commit_seq: int,
    now: int,
    run_id: str,
    evidence: Sequence[Row],
    items: Sequence[str],
    room: int,
    first: bool,
    policy_release_ref: str,
) -> CompensationOutcome:
    """Compensate an erase's Items, in order, while their rows fit (D.7.3 steps 1-5).

    Each Item whose head cites erased Evidence is requoted from a surviving
    complete group or an earlier eligible version, else gets its content-free
    head; then the entities the compensated heads anchored, and the
    participants whose source this transaction erases, are erased when left
    unanchored. An Item's rows never split: the Items are taken in order while
    the written rows, deletes and possible entity erasures stay within
    ``room``; with ``first`` the first Item is taken whatever its size.

    Args:
        writer: The single Writer, inside the transaction's write scope.
        scope: The erase's exact ``(scope_kind, scope_id)``.
        commit_seq: The transaction's ``commit_seq``.
        now: The transaction's trusted commit time.
        run_id: The erase owner, the new revisions' ``origin_run_id``.
        evidence: The Evidence post-images the transaction writes (erased
            placeholders); none for a transaction that only compensates.
        items: The Items to compensate, by id (`ErasureDependents.items`).
        room: The rows the compensation may take.
        first: Take the first Item even beyond ``room``: the transaction has
            nothing else to make progress with.
        policy_release_ref: The current extraction policy release.

    Returns:
        The rows to commit in the transaction, without the Evidence rows.

    Raises:
        AMLError: ``reference_missing`` for an Item without its head; an I11
            operation's own error.
    """
    txn = TxnRows(writer, cast(ScopeKind, scope[0]), scope[1], commit_seq, now, run_id)
    erased: set[str] = set()
    lost_keys: set[str] = set()
    for row in evidence:
        before = writer.get_row("evidence", row["object_id"])
        if row["state"] == "erased" and (before is None or before["state"] != "erased"):
            erased.add(row["object_id"])
            if before is not None:
                lost_keys |= source_participant_keys(scope[1], before)
        txn.put("evidence", row)
    touched: set[str] = set()
    if lost_keys:
        touched.update(
            row["object_id"]
            for row in txn.pages(EntityRows(scope_kind=txn.scope_kind, scope_id=txn.scope_id,
                                            state="active"))
            if row["participant_key"] in lost_keys
        )
    read = readability(txn)
    tombstoned = _tombstoned(txn)
    done: list[str] = []
    requoted: list[str] = []
    erased_heads: list[str] = []

    def size() -> int:
        return len(txn.written) - len(evidence) + len(txn.deleted) + len(touched)

    for memory_id in items:
        saved = (dict(txn.written), dict(txn.deleted), set(touched))
        item = txn.get("item", memory_id)
        head = None if item is None else txn.get("revision", item["current_revision_id"])
        if item is None or head is None:
            raise AMLError("reference_missing", "An Item lost its head revision")
        choice = choose_compensation(
            _versions(txn, memory_id), read, tombstoned=tombstoned, mode="erase",
            corrected=lambda revision_id: bool(txn.links_to("corrects_history", revision_id)),
        )
        if choice.action == "keep":
            done.append(memory_id)
            continue
        touched |= _entities_of(txn, head)
        if choice.action == "requote":
            version = txn.get("revision", cast(str, choice.revision_id))
            assert version is not None
            groups = version["snapshot"]["support_groups"]
            requote(txn, memory_id, version=version,
                    support=[groups[index] for index in choice.groups],
                    policy_release_ref=policy_release_ref)
        else:
            erase_head(txn, memory_id, policy_release_ref=policy_release_ref)
        if size() > room and (done or not first):
            txn.written, txn.deleted, touched = saved
            break
        done.append(memory_id)
        (requoted if choice.action == "requote" else erased_heads).append(memory_id)
    entities = erase_unanchored_entities(
        txn, touched, participant_has_source=participant_has_source(txn),
    ) if touched else []
    return CompensationOutcome(
        rows=tuple(write for write in txn.row_writes() if write.kind != "evidence"),
        deletes=tuple(txn.row_deletes()),
        items=tuple(done),
        requoted=tuple(sorted(requoted)),
        erased_heads=tuple(sorted(erased_heads)),
        erased_entities=tuple(entities),
        new_items=sum(
            1 for memory_id in {*requoted, *erased_heads}
            if not any(row["origin_run_id"] == run_id and row["op"] in {"requote", "erase"}
                       for row in txn.pages(RevisionRows(memory_id=memory_id)))
        ),
    )


def judge_record_groups(
    writer: Writer, scope: tuple[str, str], erased: frozenset[str],
    tombstones: frozenset[str] = frozenset(),
) -> tuple[RecordGroup, ...]:
    """Every judge record of the scope that depends on erased content, in deletion groups.

    A neighbour snapshot (``NeighborSnapshotV1``, C.4.5) depends on erased
    content when its nomination's dependencies list erased Evidence, a version
    quoting it (a shown target, history entry, entity fact or naming memory) or
    a tombstone the erase deletes: the snapshot holds their text. A judge call
    (``JudgeCallManifestV1``) does when its input spans name erased Evidence,
    one of its members' snapshots does, or a version it showed quotes erased
    Evidence: the model read the whole batch. Then every result index naming
    such a call or snapshot goes, with the accepted output it owns; a result
    borrowing a deleted output or result goes too, and so does every
    ``judge_output`` locator naming a deleted output or result (C.8 #1), so no
    later review reuses one. Manifest artifacts take their parts. Runs keep
    their input and result locators, and versions and candidates their verdict
    references, which no longer resolve (VERDICT_REF ``retain_locator``).

    Args:
        writer: The single Writer (read only).
        scope: The erase's exact ``(scope_kind, scope_id)``.
        erased: The erased Evidence ids.
        tombstones: The deleted tombstone ids.

    Returns:
        The connected groups of records to delete (``(kind, row)`` pairs).

    Raises:
        AMLError: ``binding_mismatch`` for a judge record of another shape.
    """
    if not erased and not tombstones:
        return ()
    records = _judge_records(writer, scope)
    quotes: dict[str, bool] = {}

    def quotes_erased(revision_id: str) -> bool:
        if revision_id not in quotes:
            row = writer.get_row("revision", revision_id)
            quotes[revision_id] = row is not None and any(
                span["evidence_id"] in erased for span in row["snapshot"]["quote_spans"]
            )
        return quotes[revision_id]

    def snapshot_depends(snapshot: NeighborSnapshotV1) -> bool:
        found = snapshot.dependencies
        return not erased.isdisjoint(found.evidence_ids) or (
            not tombstones.isdisjoint(found.tombstone_ids)
        ) or any(quotes_erased(revision_id)
                 for revision_id in (*found.revision_ids, *found.naming_revision_ids))

    gone_snapshots = {reference for reference, body in records.snapshots.items()
                      if snapshot_depends(body)}
    gone_calls = {
        reference for reference, call in records.calls.items()
        if any(span.evidence_id in erased for span in call.evidence_spans)
        or any(quotes_erased(revision_id) for revision_id in call.revision_ids)
    }
    return _close_judge_records(writer, scope, records, gone_snapshots, gone_calls, set())


@dataclass(frozen=True, slots=True)
class _JudgeRecords:
    """A scope's judge records by logical id: snapshots, call manifests, result indexes."""

    snapshots: dict[str, NeighborSnapshotV1]
    calls: dict[str, JudgeCallManifestV1]
    results: dict[str, JudgeResultIndexV8]


def _judge_records(writer: Writer, scope: tuple[str, str]) -> _JudgeRecords:
    """Read every judge child's snapshots, calls and results, and the owners' snapshots."""
    from aml.core.judge_records import (
        JudgeCallManifestV1,
        JudgeResultIndexV8,
        JudgmentRunInputV1,
    )
    from aml.core.neighbor_snapshot import NeighborSnapshotV1

    scope_kind = cast(ScopeKind, scope[0])
    snapshots: dict[str, NeighborSnapshotV1] = {}
    calls: dict[str, JudgeCallManifestV1] = {}
    results: dict[str, JudgeResultIndexV8] = {}
    for run in _pages(writer, ScopedRunRows(scope_kind=scope_kind, scope_id=scope[1],
                                            run_kind="evaluate")):
        if run["input_ref"] is None or not isinstance(
            writer.read_logical(run["input_ref"], "run_input"), JudgmentRunInputV1,
        ):
            continue
        for reference, body in writer.find_logical_candidates(run["object_id"], "decision_set"):
            if isinstance(body, NeighborSnapshotV1):
                snapshots[reference] = body
            elif isinstance(body, JudgeCallManifestV1):
                calls[reference] = body
            else:
                raise AMLError("binding_mismatch", "A judge child wrote another decision set")
        for reference, body in writer.find_logical_candidates(run["object_id"], "result_index"):
            if not isinstance(body, JudgeResultIndexV8):
                raise AMLError("binding_mismatch", "A judge child wrote another result")
            results[reference] = body
    for run in _pages(writer, ScopedRunRows(scope_kind=scope_kind, scope_id=scope[1],
                                            run_kind="finalize")):
        snapshots.update(
            (reference, body) for reference, body in writer.find_logical_candidates(
                run["object_id"], "decision_set")
            if isinstance(body, NeighborSnapshotV1)
        )
    return _JudgeRecords(snapshots, calls, results)


def _close_judge_records(
    writer: Writer, scope: tuple[str, str], records: _JudgeRecords,
    gone_snapshots: set[str], gone_calls: set[str], gone_outputs: set[str],
) -> tuple[RecordGroup, ...]:
    """Close deleted snapshots, calls and outputs over the records naming them, in groups.

    A call listing a deleted member snapshot goes; then every result naming a deleted
    call, snapshot or output, a result borrowing a deleted result, the accepted output a
    deleted result owns, and every ``judge_output`` locator naming a deleted output or
    result (`judge_record_groups`).
    """
    from aml.core.controls import ControlRowV1
    from aml.core.judge_records import JudgeOutputLocatorV1

    calls, results = records.calls, records.results
    gone_calls = gone_calls | {
        reference for reference, call in calls.items()
        if not gone_snapshots.isdisjoint(call.member_snapshot_refs)
    }
    gone_results: set[str] = set()
    gone_outputs = set(gone_outputs)
    grown = True
    while grown:
        grown = False
        for reference, result in results.items():
            if reference in gone_results or not (
                result.call_ref in gone_calls or result.snapshot_ref in gone_snapshots
                or result.borrowed_from in gone_results
                or result.output_ref is not None and result.output_ref in gone_outputs
            ):
                continue
            gone_results.add(reference)
            if result.output_ref is not None and result.borrowed_from is None:
                gone_outputs.add(result.output_ref)
            grown = True
    links: list[tuple[str, str]] = []
    for reference, result in results.items():
        if reference not in gone_results:
            continue
        links += [(reference, result.call_ref), (reference, result.snapshot_ref)]
        if result.output_ref is not None:
            links.append((reference, result.output_ref))
        if result.borrowed_from is not None:
            links.append((reference, result.borrowed_from))
    for reference in gone_calls:
        links += [(reference, member) for member in calls[reference].member_snapshot_refs]
    found: dict[str, tuple[str, Row]] = {}
    for reference in sorted(gone_snapshots | gone_calls | gone_results | gone_outputs):
        row = writer.get_row("artifact", reference)
        if row is not None:
            found[reference] = ("artifact", row)
    reader = writer.indexed_reader()
    after: str | None = None
    while True:
        page = reader.judge_output_controls(scope[1], after=after)
        for raw in page:
            value = ControlRowV1.model_validate(raw).value
            if not isinstance(value, JudgeOutputLocatorV1):
                raise AMLError("binding_mismatch", "A judge output locator has another value")
            if value.result_ref in gone_results or value.output_ref in gone_outputs:
                found[raw["key"]] = ("control", raw)
                links += [(raw["key"], value.result_ref), (raw["key"], value.output_ref)]
        if len(page) < 128:
            break
        after = page[-1]["key"]
    return _groups(writer, found, links)


@dataclass(frozen=True, slots=True)
class CredentialRecords:
    """The deletion groups of `credential_records` and which of their records hold one.

    Attributes:
        groups: Connected groups, each deleted in one transaction, as `ErasureDependents`
            holds them.
        holders: The logical ids of the records that hold a recognized credential;
            the other members leave because they name or contain one of them.
    """

    groups: tuple[RecordGroup, ...] = ()
    holders: frozenset[str] = frozenset()


def holds_recognized_credential(value: object) -> bool:
    """Whether a decoded value holds a credential the reception detector recognizes.

    The leaves read are the residual scan's (`PrivacyResidualScanner.inspect_value`):
    free text and undeclared strings, never declared references, local identities
    or keyed digests.
    """
    from aml.ingestion.boundary import contains_recognized_credential
    from aml.store.registry import extract_text_leaves

    return any(contains_recognized_credential(leaf.value.encode("utf-8"))
               for leaf in extract_text_leaves(value))


def credential_records(writer: Writer, scope: tuple[str, str]) -> CredentialRecords:
    """The ended records of a scope that hold a recognized credential, in deletion groups.

    Reception refuses a credential the detector recognizes and product admission a
    fact that states one, but an Add owner keeps every model reply it accepted and the
    records derived from it: a reply can state, in a form the detector reads, a
    credential its source wrote in one it does not ("I set my password to tulip-42"
    read back as "The user's password is tulip-42."). After its planned batches a
    privacy erase deletes such a record of an ended owner as C.13's erasure manifest
    deletes a record that depends on erased content (D.7.5): an extraction reply or a
    proposal with its unshared manifest parts, after its content-free accounting
    (`erased_extraction_record`), with the embedding batches that superseded vector
    snapshots replace; a judge output or neighbour snapshot with every judge record
    naming it (`_close_judge_records`). A running owner's records stay, and so do
    its judge children's: its recovery reads them, so the receipt declares them
    retained (`CredentialFindingsV1`).

    Args:
        writer: The single Writer (read only).
        scope: The erase's exact ``(scope_kind, scope_id)``.

    Returns:
        The groups to delete and the records among them that hold a credential.

    Raises:
        AMLError: ``binding_mismatch`` for a record of another shape.
    """
    from aml.core.inference import StoredInferenceOutput
    from aml.core.neighbor_snapshot import NeighborSnapshotV1
    from aml.core.proposal import ProposalV2
    from aml.store.privacy_embeddings import superseded_embedding_refs

    scope_kind = cast(ScopeKind, scope[0])
    ended = {
        run["object_id"] for run in _pages(writer, ScopedRunRows(
            scope_kind=scope_kind, scope_id=scope[1], run_kind="finalize"))
        if run["status"] in ENDED_OWNER
    }
    extraction: set[str] = set()
    proposals = False
    gone_snapshots: set[str] = set()
    gone_outputs: set[str] = set()

    def outputs(run_id: str) -> None:
        for reference, body in writer.find_logical_candidates(run_id, "llm_output"):
            if not isinstance(body, StoredInferenceOutput):
                raise AMLError("binding_mismatch", "A run stored another reply variant")
            if holds_recognized_credential(body):
                (extraction if body.purpose == "extract" else gone_outputs).add(reference)
        for reference, body in writer.find_logical_candidates(run_id, "decision_set"):
            if isinstance(body, NeighborSnapshotV1) and holds_recognized_credential(body):
                gone_snapshots.add(reference)

    for owner in sorted(ended):
        outputs(owner)
        for reference, body in writer.find_logical_candidates(owner, "proposal_snapshot"):
            if not isinstance(body, ProposalV2):
                raise AMLError("binding_mismatch", "An Add owner stored another proposal variant")
            if holds_recognized_credential(body):
                extraction.add(reference)
                proposals = True
    for run in _pages(writer, ScopedRunRows(scope_kind=scope_kind, scope_id=scope[1],
                                            run_kind="evaluate")):
        if run["parent_run_id"] in ended:
            outputs(run["object_id"])
    holders = frozenset(extraction | gone_snapshots | gone_outputs)
    if not holders:
        return CredentialRecords()
    selected = extraction | superseded_embedding_refs(writer, scope) if proposals else extraction
    groups = [*record_groups(writer, selected)]
    if gone_snapshots or gone_outputs:
        groups += _close_judge_records(writer, scope, _judge_records(writer, scope),
                                       gone_snapshots, set(), gone_outputs)
    merged: dict[str, RecordGroup] = {}
    for group in groups:
        merged.setdefault(_identity(group[0]), group)
    return CredentialRecords(
        tuple(sorted(merged.values(), key=lambda group: _identity(group[0]))), holders,
    )


def _groups(
    writer: Writer, records: dict[str, tuple[str, Row]], links: Iterable[tuple[str, str]],
) -> tuple[RecordGroup, ...]:
    """Partition the records into connected groups; a manifest artifact takes its parts."""
    parent = {identity: identity for identity in records}

    def root(identity: str) -> str:
        while parent[identity] != identity:
            parent[identity] = parent[parent[identity]]
            identity = parent[identity]
        return identity

    for left, right in links:
        if left in parent and right in parent:
            parent[root(left)] = root(right)
    artifact_ids = {identity for identity, (kind, _row) in records.items() if kind == "artifact"}
    physical = record_groups(writer, artifact_ids)
    for group in physical:
        identities = [row["object_id"] for _kind, row in group if row["object_id"] in artifact_ids]
        for identity in identities[1:]:
            parent[root(identity)] = root(identities[0])
    grouped: dict[str, list[tuple[str, Row]]] = {}
    for identity in sorted(records):
        kind, row = records[identity]
        if kind != "artifact":
            grouped.setdefault(root(identity), []).append((kind, row))
    for group in physical:
        identity = next(row["object_id"] for _kind, row in group
                        if row["object_id"] in artifact_ids)
        grouped.setdefault(root(identity), []).extend(group)
    groups = [tuple(sorted(members, key=lambda pair: (pair[0], _identity(pair))))
              for members in grouped.values()]
    return tuple(sorted(groups, key=lambda group: _identity(group[0])))


def _identity(pair: tuple[str, Row]) -> str:
    kind, row = pair
    return cast(str, row["key"] if kind == "control" else row["object_id"])


class ErasedEvidence:
    """Which Evidence of one scope is erased content (D.7).

    An erase transaction names the Evidence it erases before storing it as
    erased, so that Evidence counts as erased first; Evidence an earlier erase
    removed counts as it is stored. Only Evidence states and immutable versions
    are read, so the answer for a record never changes during an erase.
    """

    def __init__(self, writer: Writer, scope: tuple[str, str], erased: frozenset[str]) -> None:
        """Bind the scope and the Evidence the current transaction erases."""
        self._writer = writer
        self._scope = scope
        self._erased = erased
        self._states: dict[str, bool] = {}
        self._versions: dict[str, bool] = {}
        self._requests: dict[tuple[str, str], tuple[tuple[int, int], ...]] = {}

    def holds(self, evidence_id: str) -> bool:
        """Whether this Evidence is erased, or erased by the current transaction."""
        if evidence_id in self._erased:
            return True
        found = self._states.get(evidence_id)
        if found is None:
            row = self._writer.get_row("evidence", evidence_id)
            found = self._states[evidence_id] = row is not None and row["state"] == "erased"
        return found

    def any(self, evidence_ids: Iterable[str]) -> bool:
        """Whether any of these Evidence rows is erased."""
        return any(self.holds(identity) for identity in evidence_ids)

    def quoted(self, revision_id: str) -> bool:
        """Whether a version quotes erased Evidence; a version a forget deleted quotes none."""
        found = self._versions.get(revision_id)
        if found is None:
            row = self._writer.get_row("revision", revision_id)
            found = self._versions[revision_id] = row is not None and self.any(
                row["evidence_ids"])
        return found

    def in_request(self, session_id: str, request_id: str, first_ordinal: int, pin: int) -> bool:
        """Whether a message of one request, from ``first_ordinal`` on and received by
        ``pin``, is erased: what a previous batch rendered from that request (B.3)."""
        from aml.core.query import ScopeV1

        key = (session_id, request_id)
        found = self._requests.get(key)
        if found is None:
            reader = self._writer.indexed_reader()
            scope = ScopeV1(scope_kind=cast(ScopeKind, self._scope[0]), scope_id=self._scope[1])
            coordinates: list[tuple[int, int]] = []
            after: str | None = None
            while True:
                page = reader.session_source_coordinates(scope, session_id, request_id,
                                                         after=after)
                coordinates += [(row["ordinal"], row["ingested_commit_seq"]) for row in page
                                if self.holds(row["object_id"])]
                if len(page) < 128:
                    break
                after = page[-1]["object_id"]
            found = self._requests[key] = tuple(coordinates)
        return any(ordinal >= first_ordinal and ingested <= pin for ordinal, ingested in found)

    def shown_by(self, context: ExtractContextV2) -> bool:
        """Whether a unit's request showed erased content (B.4.4): one of its fresh, tail
        or bound parts, a message of its previous batch, or a context memory version or
        earlier version that quotes erased Evidence."""
        return self.any(binding.evidence_id for binding in context.unit.member_bindings) or (
            self.context_shown_by(context)
        )

    def context_shown_by(self, context: ExtractContextV2) -> bool:
        """Whether erased context outside the unit's fresh sources influenced its output.

        Fresh sources have explicit proposal support groups; catalog, previous-batch
        and other context have no per-field provenance, so their proposals must leave.
        """
        fresh = {binding.evidence_id for binding in context.unit.member_bindings}
        shown = [span.evidence_id for span in context.tail]
        shown += [ranges.evidence_id for ranges in context.withheld]
        shown += [binding.evidence_id for binding in context.bindings]
        previous = context.previous
        names = [self._writer.get_row("entity_name", identity)
                 for entry in context.catalog for identity in entry.name_row_ids]
        return self.any(identity for identity in shown if identity not in fresh) or any(
            name is None or self.quoted(name["naming_revision_id"]) for name in names
        ) or previous is not None and self.in_request(
            previous.session_id, previous.request_id, previous.first_kept_ordinal,
            context.read_view_commit_seq,
        ) or any(self.quoted(memory.revision_id) for memory in context.memories)


def record_groups(
    writer: Writer, references: Iterable[str],
) -> list[RecordGroup]:
    """Keep shared deleted parts atomic, and preserve parts a surviving manifest needs."""
    from aml.core.artifacts import ManifestV1
    from aml.store.indexed import ArtifactRows

    selected = set(references)
    rows: dict[str, Row] = {}
    for reference in sorted(selected):
        row = writer.get_row("artifact", reference)
        if row is not None:
            rows[reference] = row
    protected: set[str] = set()
    creators = {row["created_by_run_id"] for row in rows.values()
                if row["artifact_kind"] == "manifest"}
    for creator in sorted(creators):
        for row in _pages(writer, ArtifactRows(created_by_run_id=creator, kinds=("manifest",))):
            if row["object_id"] not in selected:
                protected.update(ManifestV1.model_validate_json(row["body"]).parts)
    parents = {reference: reference for reference in rows}

    def root(reference: str) -> str:
        while parents[reference] != reference:
            parents[reference] = parents[parents[reference]]
            reference = parents[reference]
        return reference

    part_owner: dict[str, str] = {}
    for reference, row in sorted(rows.items()):
        if row["artifact_kind"] != "manifest":
            continue
        for part in ManifestV1.model_validate_json(row["body"]).parts:
            if part in protected:
                continue
            previous = part_owner.setdefault(part, reference)
            parents[root(reference)] = root(previous)
            if part in rows:
                parents[root(part)] = root(reference)
    grouped: dict[str, list[tuple[str, Row]]] = {}
    claimed = set(rows)
    for reference, row in sorted(rows.items()):
        members = grouped.setdefault(root(reference), [])
        members.append(("artifact", row))
        if row["artifact_kind"] == "manifest":
            for part in ManifestV1.model_validate_json(row["body"]).parts:
                stored = writer.get_row("artifact", part)
                if stored is not None and part not in claimed and part not in protected:
                    claimed.add(part)
                    members.append(("artifact", stored))
    return sorted((tuple(sorted(members, key=_identity)) for members in grouped.values()),
                  key=lambda group: _identity(group[0]))


def needed_proposals(
    writer: Writer, scope: tuple[str, str], erased: ErasedEvidence,
) -> frozenset[str]:
    """Pending proposals whose cited sources survive and can still re-enter (C.3).

    A pending candidate citing no erased Evidence re-enters (C.3), reading its
    ``ProposalV2`` and embedding batches; one citing erased Evidence the erase
    expires (D.7.3 #5) and needs neither.

    Returns:
        The surviving pending candidates' proposal references.
    """
    pending: set[str] = set()
    query = CandidateRows(scope_kind=cast(ScopeKind, scope[0]), scope_id=scope[1],
                          status="pending")
    for row in _pages(writer, query):
        if not erased.any(row["evidence_ids"]):
            pending.add(row["proposal_ref"])
    return frozenset(pending)


def pending_record_candidates(
    writer: Writer, scope: tuple[str, str], groups: Sequence[RecordGroup],
) -> dict[str, list[Row]]:
    """Find pending rows whose selected proposal leaves in an erasure record group.

    A proposal may depend on erased context without citing its Evidence. Such a
    row must expire in the transaction deleting its proposal, before any reentry
    can observe the absent body. The Writer budgets these rows with that group.
    """
    references = {row["object_id"] for group in groups for kind, row in group
                  if kind == "artifact"}
    if not references:
        return {}
    found: dict[str, list[Row]] = {}
    for row in _pages(writer, CandidateRows(scope_kind=cast(ScopeKind, scope[0]),
                                          scope_id=scope[1], status="pending")):
        if row["proposal_ref"] in references:
            found.setdefault(row["proposal_ref"], []).append(row)
    return found


def owner_extraction_records(
    writer: Writer, owner_run_id: str, *, erased: ErasedEvidence | None = None,
    pending: frozenset[str] = frozenset(), live: bool = False,
) -> frozenset[str]:
    """Select an owner's affected replies, proposals and unused vector batches.

    A unit depends on erased content when its request showed any
    (`ErasedEvidence.shown_by`); its replies, every call's, then do. A
    proposal with no surviving support leaves even for a live owner. Erased
    non-fresh context invalidates proposals because their fields have no separate
    context provenance. Ended owners also drop fresh-source-dependent proposals.
    Mixed vector batches are replaced with exact surviving subsets before deletion.
    With no ``erased``, every record
    of the owner that holds text or vectors qualifies: the records of a cycle a
    privacy change cancelled before it reviewed or committed anything (D.5),
    which no row cites.

    Args:
        writer: The single Writer (read only).
        owner_run_id: The Add owner.
        erased: The scope's erased Evidence, or None for a cancelled cycle.
        pending: The proposals of pending candidates that re-enter.
        live: Whether surviving proposals can still be needed by this owner.

    Returns:
        The records' logical ids.

    Raises:
        AMLError: ``binding_mismatch`` for a record of another shape.
    """
    from aml.core.extraction_context import ExtractContextV2
    from aml.core.inference import StoredInferenceOutput
    from aml.core.proposal import ProposalV2
    from aml.core.semantic_embedding import StatementEmbeddingV1

    shown: dict[int, bool] = {}
    context_shown: dict[int, bool] = {}
    units: dict[str, int] = {}
    for _reference, body in writer.find_logical_candidates(owner_run_id, "decision_set"):
        if isinstance(body, ExtractContextV2):
            unit_no = body.unit.unit_no
            shown[unit_no] = erased is None or erased.shown_by(body)
            context_shown[unit_no] = erased is None or erased.context_shown_by(body)
            units[body.input_digest] = unit_no
    found: set[str] = set()
    for reference, body in writer.find_logical_candidates(owner_run_id, "llm_output"):
        if not isinstance(body, StoredInferenceOutput):
            raise AMLError("binding_mismatch", "An Add owner stored another reply variant")
        unit = units.get(body.input_digest)
        if body.purpose == "extract" and (unit is None or shown.get(unit, True)):
            found.add(reference)
    proposals: dict[tuple[str, int, int], bool] = {}
    kept: set[str] = set()
    for reference, body in writer.find_logical_candidates(owner_run_id, "proposal_snapshot"):
        if not isinstance(body, ProposalV2):
            raise AMLError("binding_mismatch", "An Add owner stored another proposal variant")
        cited_erased = erased is None or erased.any(span.evidence_id for span in body.quote_spans)
        surviving_group = erased is not None and any(
            not erased.any(body.quote_spans[index].evidence_id for index in group)
            for group in body.support_groups
        )
        leaves = context_shown.get(body.position_key[1], True) or (
            cited_erased and not (live and surviving_group)
        ) or (
            not live and reference not in pending and shown.get(body.position_key[1], True)
        )
        proposals[body.position_key] = leaves
        if leaves:
            found.add(reference)
        else:
            kept |= {use.ref for use in (body.statement_embedding, body.cue_embedding)
                     if use is not None}
    for reference, body in writer.find_logical_candidates(owner_run_id, "semantic_embedding"):
        if not isinstance(body, StatementEmbeddingV1):
            raise AMLError("binding_mismatch", "An Add owner stored another embedding variant")
        if reference not in kept and any(proposals.get(use, True)
                                         for entry in body.entries for use in entry.uses):
            found.add(reference)
    return frozenset(found)


def extraction_record_groups(
    writer: Writer, scope: tuple[str, str], erased: frozenset[str],
) -> tuple[RecordGroup, ...]:
    """Affected extraction records of all Add owners, including interrupted live owners.

    Dependencies include the current erase and previously erased Evidence.
    The Writer records content-free recovery metadata before deleting bodies.
    Contexts and settlement records hold locators and codes only and stay;
    runs and candidates keep their reply and proposal locators, which then no
    longer resolve (``EXTRACTION_REF``, A.3.4).

    Args:
        writer: The single Writer (read only).
        scope: The erase's exact ``(scope_kind, scope_id)``.
        erased: The Evidence the erase has erased, or erases in the transaction
            being built.

    Returns:
        One group per record with its manifest parts, in identity order.

    Raises:
        AMLError: ``binding_mismatch`` for a record of another shape.
    """
    if not erased:
        return ()
    evidence = ErasedEvidence(writer, scope, erased)
    pending = needed_proposals(writer, scope, evidence)
    found: set[str] = set()
    query = ScopedRunRows(scope_kind=cast(ScopeKind, scope[0]), scope_id=scope[1],
                          run_kind="finalize")
    for run in _pages(writer, query):
        found |= owner_extraction_records(writer, run["object_id"], erased=evidence,
                                          pending=pending, live=run["status"] not in ENDED_OWNER)
    from aml.store.privacy_embeddings import superseded_embedding_refs

    found |= superseded_embedding_refs(writer, scope)
    return tuple(record_groups(writer, found))


def participant_has_source(txn: TxnRows) -> Any:
    """Whether a participant keeps a non-erased source in the scope (D.9, A.3.6).

    Privacy erase and forget pass it to `erase_unanchored_entities`. Forget never
    erases Evidence, so a participant it leaves unanchored keeps its source.
    """
    keys: set[str] | None = None

    def check(entity: Row) -> bool:
        nonlocal keys
        if keys is None:
            keys = set()
            stored: Iterable[Row] = txn.pages(
                EvidenceRows(scope_kind=txn.scope_kind, scope_id=txn.scope_id)
            )
            for row in txn.merged("evidence", stored, lambda row: row["state"] != "erased"):
                keys |= source_participant_keys(txn.scope_id, row)
        return entity["participant_key"] in keys

    return check


def compensation_counts(writer: Writer, scope: tuple[str, str], owner_run_id: str) -> Row:
    """Count the Items, revisions and links an erase owner's compensation wrote (D.7.3).

    The count is derived from durable rows: every ``requote`` or ``erase``
    revision whose ``origin_run_id`` is the owner, the distinct Items they head
    and the links they carry. Each transaction writes its compensation
    atomically with the fence's progress, so after a crash the count equals the
    committed prefix's.

    Args:
        writer: The single Writer (or a read over its stored rows).
        scope: The erase's exact ``(scope_kind, scope_id)``.
        owner_run_id: The privacy-erase owner.

    Returns:
        ``{"item": n, "revision": n, "link": n}``.
    """
    txn = TxnRows(writer, cast(ScopeKind, scope[0]), scope[1], 0, 0, owner_run_id)
    items: set[str] = set()
    revisions = links = 0
    for item in txn.pages(ItemRows(scope_kind=txn.scope_kind, scope_id=txn.scope_id)):
        for revision in txn.pages(RevisionRows(memory_id=item["object_id"])):
            if revision["origin_run_id"] != owner_run_id or revision["op"] not in {
                "requote", "erase",
            }:
                continue
            items.add(item["object_id"])
            revisions += 1
            links += len(txn.links_of(revision["object_id"]))
    return {"item": len(items), "revision": revisions, "link": links}
