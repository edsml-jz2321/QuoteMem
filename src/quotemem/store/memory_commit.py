"""The Add owner's commit: settlement, rejection and component transactions.

Per owner and round the transactions run in this order:

1. **settlement** (round 1, after the barrier): the rejections, restatements
   and duplicates and the unresolved primaries (the records);
2. **rejection**: every position whose disposition writes no memory, the
   re-entry touches and expiries, the spans the forgotten gate withholds and the
   pending candidates those spans leave without a usable group;
3. **components**: one transaction per formal component (`commit_component`,
   behind the commit fence).

Settlement and rejection are chunked in canonical order within
``MAX_TX_ROWS``; positions are not atomic with each other. Each chunk writes a
create-once `CommitJournalV1` keyed by its first and last positions, and a
replay skips every position an earlier journal of the same owner, round and
kind covers, so a crash resumes at the first position without a row and a
committed chunk is never written twice. A component is never split.

This module never calls a provider and never reads a statement outside the
Writer path; candidate rows carry identities, codes and digests only.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final, Literal, cast

from quotemem.core.add_read_view import EvidenceRecord, Sigma
from quotemem.core.commit_plan import (
    CommitJournalV1,
    CommitTxn,
    DuplicatePositionV1,
    JournalMemberV1,
    JournalRowV1,
    SettledPositionV1,
    journal_key,
)
from quotemem.core.entities import PositionKey, QuoteSpanV1, ReasonCode, WithheldReason
from quotemem.core.errors import QuoteMemError
from quotemem.core.identity import ArtifactId, IntegrityDigest, integrity_digest, new_ulid
from quotemem.core.outcome_table import outcome_rule
from quotemem.core.readability import Readability
from quotemem.core.review_rounds import RoundNo
from quotemem.core.semantic import ScopeKind
from quotemem.core.statement import MAX_MEMORY_SPANS, MAX_WITHHELD_SPANS
from quotemem.ingestion.reentry import ReentryTouchV1
from quotemem.store.entity_ops import TxnRows, restore_support
from quotemem.store.indexed import CandidateRows
from quotemem.store.request_completion import require_owner_budget

if TYPE_CHECKING:
    from quotemem.store.writer import CommittedRows, ExpectedHead, Writer, WriteTicket

Row = dict[str, Any]
Publish = Callable[["CommittedRows"], None]
GroupsOf = Callable[[str], Sequence[Sequence[QuoteSpanV1]]]
"""Return the support groups of a stored ``ProposalV2`` by reference, each as its spans."""
Clock = Callable[[], int]
"""The owner's trusted action clock, in Writer milliseconds."""

def owner_budget(writer: Writer, owner_run_id: str, clock: Clock) -> None:
    """Refuse a commit the actual finalize owner's budget no longer permits.

    The clock is read at the check itself, so time spent waiting for the
    Writer mutex or building the rows counts; a caller checks before its
    pre-lock work and again under the mutex right before ``commit_rows``.

    Args:
        writer: The single Writer.
        owner_run_id: The actual finalize owner whose budget pays for the commit.
        clock: The owner's trusted action clock.

    Raises:
        QuoteMemError: ``delivery_busy`` once the owner's budget is refused,
            exhausted or past its deadline; ``binding_mismatch`` when the
            owner is not a finalize Run.
    """
    require_owner_budget(writer, owner_run_id, now=clock())


CHUNK_POSITIONS: Final = 512
"""Positions per settlement or rejection chunk at most: keeps each journal one row."""
CHUNK_ROWS: Final = 64
"""Rows per chunk without withholding or ``denied_by_forget``.

A chunk holds the Writer for its whole proof and write. At 256 rows a7's
rejection holds reached 0.83 s; the smaller bound limits that uninterrupted work.
A chunk ends before the position that would take it past this bound (a first
position always fits). A transaction that may withhold retains the 256-row
bound: withholding can expire another position the same chunk touches or writes.
The bound uses every supplied entry and the same transaction's durable journals,
so cold adoption keeps it after omitting positions already journaled.
"""
WITHHOLD_CHUNK_ROWS: Final = 256
"""The existing bound for every chunk of a transaction that may withhold."""
_OVERHEAD: Final = 3
"""RowOps every chunk adds besides its rows: its journal, ``commit_seq`` and ``scope_seq``."""


def proposal_groups(writer: Writer) -> GroupsOf:
    """Read a ``ProposalV2``'s support groups through the Writer.

    Args:
        writer: The single Writer.

    Returns:
        A function of a ``proposal_snapshot`` reference to its groups, each as
        the spans of ``quote_spans`` that ``support_groups`` names.
    """

    def groups(reference: str) -> tuple[tuple[QuoteSpanV1, ...], ...]:
        body: Any = writer.read_logical(reference, "proposal_snapshot")
        spans: Sequence[QuoteSpanV1] = body.quote_spans
        support: Sequence[Sequence[int]] = body.support_groups
        return tuple(tuple(spans[index] for index in group) for group in support)

    return groups


# Candidate rows ---------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PositionV1:
    """One position's candidate identity, as its rows are written.

    Attributes:
        position_key: The position.
        extraction_ref: Its unit's ``llm_output``.
        proposal_ref: Its ``ProposalV2``.
        statement_digest: Its statement's digest.
        evidence_ids: The Evidence its quotes cite, sorted and unique.
        candidate_id: Its stored candidate row, for a position re-entered from an
            earlier owner; None for a position of this owner, whose row is
            minted at its first persisted write.
    """

    position_key: PositionKey
    extraction_ref: str
    proposal_ref: str
    statement_digest: str
    evidence_ids: tuple[str, ...]
    candidate_id: str | None = None


def _status(code: ReasonCode, stored: Row | None) -> str:
    statuses = outcome_rule(code).statuses
    if not statuses:
        raise QuoteMemError("binding_mismatch", "A refused code has no candidate row")
    if code == "evidence_invalidated" and stored is not None and stored["status"] == "pending":
        return "expired"
    return statuses[0]


def candidate_row(
    txn: TxnRows, position: PositionV1, *, code: ReasonCode, entity_id: str | None,
    memory_id: str | None = None, revision_id: str | None = None,
    targets: Iterable[str] = (), judge_result_ref: str | None = None,
    hold_evidence: bool = False, touch: ReentryTouchV1 | None = None,
    new_id: str | None = None,
) -> Row:
    """Write one position's candidate row with its final disposition.

    A re-entered position updates its stored pending row; any other position
    gets a new row. The status follows the code: ``evidence_invalidated``
    expires a pending row and rejects any other. ``hold_evidence`` holds only
    while pending. A re-entry touch writes its count and wake key, and
    its input digest only after a review.

    Args:
        txn: The transaction.
        position: The position.
        code: Its disposition.
        entity_id: Its resolved primary at this decision; None when unresolved.
        memory_id: The memory it wrote or the one already stating the fact.
        revision_id: The revision it wrote; set exactly when published.
        targets: Ended, contradicted or judged targets (add-items).
        judge_result_ref: Its latest ``result_index:judge``.
        hold_evidence: Its latest review issued a tombstone.
        touch: Its re-entry touch at this owner, if selected.
        new_id: The id of a new row, chosen before its revision named it.

    Returns:
        The candidate row's post-image.
    """
    stored = (txn.get("candidate", position.candidate_id)
              if position.candidate_id is not None else None)
    if position.candidate_id is not None and stored is None:
        raise QuoteMemError("reference_missing", "A re-entered position lost its candidate row")
    status = _status(code, stored)
    pending = status == "pending"
    owner_run_id, unit_no, number = position.position_key
    base: Row = stored if stored is not None else {
        "object_id": new_id or new_ulid(), "scope_kind": txn.scope_kind, "scope_id": txn.scope_id,
        "owner_run_id": owner_run_id, "unit_no": unit_no, "position": number,
        "extraction_ref": position.extraction_ref, "proposal_ref": position.proposal_ref,
        "statement_digest": position.statement_digest,
        "evidence_ids": sorted(set(position.evidence_ids)),
        "last_input_digest": None, "last_wake_digest": None, "reentry_count": 0,
        "created_commit_seq": txn.commit_seq, "created_at": txn.now,
    }
    row: Row = {
        **base, "entity_id": entity_id, "status": status, "reason_code": code,
        "judge_result_ref": judge_result_ref, "memory_id": memory_id, "revision_id": revision_id,
        "target_memory_ids": sorted(set(targets)), "hold_evidence": hold_evidence and pending,
        "commit_seq": txn.commit_seq, "terminal_at": None if pending else txn.now,
    }
    if touch is not None:
        if touch.position_key != position.position_key:
            raise QuoteMemError("binding_mismatch", "A touch belongs to its own position")
        row.update(reentry_count=touch.reentry_count, last_wake_digest=touch.last_wake_digest)
        if touch.last_input_digest is not None:
            row["last_input_digest"] = touch.last_input_digest
    return txn.put("candidate", row)


def touch_row(txn: TxnRows, position: PositionV1, touch: ReentryTouchV1) -> Row:
    """Write only a re-entry touch on a stored pending row; its disposition is unchanged."""
    if position.candidate_id is None:
        raise QuoteMemError("binding_mismatch", "Only a re-entered position is touched")
    stored = txn.get("candidate", position.candidate_id)
    if stored is None or stored["status"] != "pending" or touch.position_key != (
        stored["owner_run_id"], stored["unit_no"], stored["position"]
    ):
        raise QuoteMemError("binding_mismatch", "A touch updates its own pending row")
    row: Row = {**stored, "reentry_count": touch.reentry_count,
                "last_wake_digest": touch.last_wake_digest, "commit_seq": txn.commit_seq}
    if touch.last_input_digest is not None:
        row["last_input_digest"] = touch.last_input_digest
    return txn.put("candidate", row)


# Withholding and the expiries it causes ----------------------------------------------


@dataclass(frozen=True, slots=True)
class _Range:
    byte_start: int
    byte_end: int


@dataclass(frozen=True, slots=True)
class _Header:
    """An Evidence row as `Readability` reads its header."""

    object_id: str
    scope_kind: str
    scope_id: str
    state: str
    route_state: str
    expires_at: int | None
    withheld_spans: tuple[_Range, ...]

    @classmethod
    def of(cls, row: Row) -> _Header:
        return cls(row["object_id"], row["scope_kind"], row["scope_id"], row["state"],
                   row["route_state"], row["expires_at"],
                   tuple(_Range(span["byte_start"], span["byte_end"])
                         for span in row["withheld_spans"]))


def readability(txn: TxnRows) -> Readability:
    """The predicates over the Evidence rows as this transaction leaves them."""

    def evidence(evidence_id: str) -> _Header | None:
        row = txn.get("evidence", evidence_id)
        return None if row is None else _Header.of(row)

    return Readability(txn.scope_kind, txn.scope_id, txn.now, evidence)


def _span_key(span: Mapping[str, Any]) -> tuple[int, int, str, str]:
    return (span["byte_start"], span["byte_end"], span["reason"], span["run_id"])


def _bounded(spans: list[Row], part: tuple[int, int]) -> list[Row]:
    """Keep at most 256 withheld ranges: merge overlapping and adjacent ones,
    else one range covering the part; ``forgotten`` and the smallest run id win."""
    spans = sorted(spans, key=_span_key)
    if len(spans) <= MAX_WITHHELD_SPANS:
        return spans
    merged: list[Row] = []
    for span in spans:
        if merged and span["byte_start"] <= merged[-1]["byte_end"]:
            last = merged[-1]
            last["byte_end"] = max(last["byte_end"], span["byte_end"])
            last["reason"] = ("forgotten" if "forgotten" in (last["reason"], span["reason"])
                              else span["reason"])
            last["run_id"] = min(last["run_id"], span["run_id"])
        else:
            merged.append(dict(span))
    if len(merged) <= MAX_WITHHELD_SPANS:
        return sorted(merged, key=_span_key)
    return [{
        "byte_start": part[0], "byte_end": part[1],
        "reason": "forgotten" if any(span["reason"] == "forgotten" for span in spans)
        else "restated_forgotten",
        "run_id": min(span["run_id"] for span in spans),
    }]


def withhold_spans(
    txn: TxnRows, spans: Iterable[QuoteSpanV1], *, reason: WithheldReason, run_id: str,
) -> set[str]:
    """Withhold located spans in their Evidence rows.

    Ranges are appended while the Evidence is active, sorted, and kept within
    256 entries; withholding never changes the text, hash or vector.

    Args:
        txn: The transaction.
        spans: The spans to withhold.
        reason: ``restated_forgotten`` for a rejected restatement, ``forgotten``
            for a forget.
        run_id: The Run that withholds them.

    Returns:
        The Evidence ids whose withheld ranges grew.
    """
    wanted: dict[str, set[tuple[int, int]]] = {}
    for span in spans:
        wanted.setdefault(span.evidence_id, set()).add((span.byte_start, span.byte_end))
    changed: set[str] = set()
    for evidence_id, ranges in sorted(wanted.items()):
        row = txn.get("evidence", evidence_id)
        if row is None:
            raise QuoteMemError("reference_missing", "A withheld span lies in no Evidence")
        if row["state"] != "active":
            continue
        # A part's remainder (``ForgetRemainderV1``) encodes the bytes readable before this
        # withholding: it leaves in the same transaction, so no stored vector of the part
        # still encodes a byte withheld now.
        current = [{key: value for key, value in span.items() if key != "remainder"}
                   for span in row["withheld_spans"]]
        held = {(span["byte_start"], span["byte_end"], span["reason"]) for span in current}
        added = [{"byte_start": start, "byte_end": end, "reason": reason, "run_id": run_id}
                 for start, end in sorted(ranges) if (start, end, reason) not in held]
        if added:
            txn.put("evidence", {**row, "withheld_spans": _bounded(
                current + added, (row["byte_start"], row["byte_end"]),
            )})
            changed.add(evidence_id)
    return changed


def expire_unusable(
    txn: TxnRows, evidence_ids: set[str], *, groups_of: GroupsOf,
    after: str | None = None, limit: int | None = None,
) -> list[str]:
    """Expire the pending candidates newly withheld spans leave without a usable group.

    A pending candidate that cites one of the Evidence rows and has no group
    whose every span passes ``p_span`` under the transaction's rows ends
    ``evidence_invalidated``: a stored pending row expires, a row this
    transaction writes is rejected. Its other fields and its re-entry
    state are kept. Candidates are examined in id order, so a forget expires
    them in bounded batches.

    Args:
        txn: The transaction, after its withholding.
        evidence_ids: The Evidence rows whose withheld ranges grew.
        groups_of: Each candidate's support groups by ``proposal_ref``.
        after: Examine only the candidates whose id sorts after this one.
        limit: Stop after this many expiries; the last id returned is where the
            next batch starts.

    Returns:
        The expired candidates' ids, sorted.
    """
    if not evidence_ids:
        return []
    read = readability(txn)
    query = CandidateRows(scope_kind=txn.scope_kind, scope_id=txn.scope_id, status="pending")
    pending = txn.merged("candidate", txn.pages(query), lambda row: row["status"] == "pending")
    expired: list[str] = []
    for row in pending:
        if after is not None and row["object_id"] <= after:
            continue
        if evidence_ids.isdisjoint(row["evidence_ids"]):
            continue
        if any(group and all(read.p_span(span) for span in group)
               for group in groups_of(row["proposal_ref"])):
            continue
        stored = txn.writer.get_row("candidate", row["object_id"])
        status = "expired" if stored is not None and stored["status"] == "pending" else "rejected"
        txn.put("candidate", {
            **row, "status": status, "reason_code": "evidence_invalidated",
            "hold_evidence": False, "commit_seq": txn.commit_seq, "terminal_at": txn.now,
        })
        expired.append(row["object_id"])
        if len(expired) == limit:
            break
    return sorted(expired)


# Chunked transactions ----------------------------------------------------


@dataclass(frozen=True, slots=True)
class ChunkEntryV1:
    """One position of a settlement or rejection chunk, with the duplicates written with it.

    Attributes:
        position_key: The position.
        resolution_digest: The integrity digest of what is written for it.
        build: Writes the position's rows into the transaction and returns the
            Evidence rows whose withheld ranges it grew.
        heads: Heads the chunk compares for it (``restatement_satisfied``).
        followers: The duplicates whose rows ``build`` also writes, each
            with its digest; they are journaled right after the position.
        withholds: Whether ``build`` declares withheld Evidence ranges or writes
            ``denied_by_forget``, coupling its expiry proof to this transaction.
    """

    position_key: PositionKey
    resolution_digest: IntegrityDigest
    build: Callable[[TxnRows], set[str]]
    heads: tuple[ExpectedHead, ...] = ()
    followers: tuple[tuple[PositionKey, IntegrityDigest], ...] = ()
    withholds: bool = False

    @property
    def members(self) -> tuple[tuple[PositionKey, IntegrityDigest], ...]:
        """The journal members the entry writes: the position, then its followers."""
        return ((self.position_key, self.resolution_digest), *self.followers)


def journaled_positions(
    writer: Writer, run_id: str, *, owner_run_id: str, round_no: RoundNo, txn: CommitTxn,
) -> set[PositionKey]:
    """The positions an existing journal of one owner, round and kind covers."""
    return _journaled_state(
        writer, run_id, owner_run_id=owner_run_id, round_no=round_no, txn=txn,
        inspect_withholding=False,
    )[0]


def _journaled_state(
    writer: Writer, run_id: str, *, owner_run_id: str, round_no: RoundNo,
    txn: CommitTxn, inspect_withholding: bool,
) -> tuple[set[PositionKey], bool]:
    """Read one transaction's journaled positions and, for chunk replay, withholding proof."""
    found: set[PositionKey] = set()
    withholding = False
    candidates: set[str] = set()
    for _reference, body in writer.find_logical_candidates(run_id, "write_set"):
        if isinstance(body, CommitJournalV1) and (body.owner_run_id, body.round, body.txn) == (
            owner_run_id, round_no, txn,
        ):
            found.update(member.position_key for member in body.members)
            if inspect_withholding and not withholding:
                withholding = any(row.kind == "evidence" for row in body.written)
                candidates.update(row.object_id for row in body.written if row.kind == "candidate")
    if inspect_withholding and not withholding and candidates:
        # An already-withheld denial writes no Evidence row. Its journaled
        # candidate still proves the original transaction could withhold.
        withholding = any(row is not None and row["reason_code"] == "denied_by_forget"
                          for row in writer.get_rows("candidate", sorted(candidates)))
    return found, withholding


def _journal(
    writer: Writer, txn: TxnRows, *, owner_run_id: str, round_no: RoundNo,
    kind: CommitTxn, entries: Sequence[tuple[PositionKey, IntegrityDigest]],
) -> CommitJournalV1:
    positions = [position for position, _digest in entries]
    return CommitJournalV1(
        journal_key=journal_key(writer.bind_content, owner_run_id=owner_run_id,
                                round_no=round_no, txn=kind, positions=positions),
        owner_run_id=owner_run_id, round=round_no, txn=kind, commit_seq=txn.commit_seq,
        members=[JournalMemberV1(position_key=position, resolution_digest=digest)
                 for position, digest in entries],
        written=_journal_rows(txn.written), deleted=_journal_rows(txn.deleted),
    )


def _journal_rows(keys: Iterable[tuple[str, str]]) -> list[JournalRowV1]:
    return [JournalRowV1.model_validate({"kind": kind, "object_id": identity})
            for kind, identity in sorted(keys)]


@dataclass
class _Chunk:
    txn: TxnRows
    entries: list[ChunkEntryV1] = field(default_factory=list[ChunkEntryV1])


def commit_chunked(
    writer: Writer, ticket: WriteTicket, *, owner_run_id: str, round_no: RoundNo,
    txn: Literal["settlement", "rejection"], entries: Sequence[ChunkEntryV1],
    updated_at: int, publish: Publish, groups_of: GroupsOf, clock: Clock,
) -> list[CommittedRows]:
    """Commit positions in canonical order as journaled chunks within ``CHUNK_ROWS``.

    Positions an earlier journal of this owner, round and kind covers are
    skipped, so a replay resumes at the first position without a row. A
    position's rows never split across chunks; a chunk takes positions while its rows
    stay within 256 if a supplied entry may withhold or the same transaction's
    journals prove prior withholding, otherwise ``CHUNK_ROWS``
    (its first always), and its RowOps within ``MAX_TX_ROWS``,
    measured and written from one preparation (``Writer.commit_rows_within``). After a
    chunk's positions, the pending candidates its withholding leaves without a usable
    group expire in the same chunk.

    Args:
        writer: The single Writer.
        ticket: The owner's ticket; it writes and creates the journals.
        owner_run_id: The Add owner.
        round_no: The round; settlement is round 1.
        txn: ``settlement`` or ``rejection``.
        entries: The positions in canonical order, each unique.
        updated_at: The transactions' time.
        publish: The read model's publication of each committed chunk.
        groups_of: Reads a pending candidate's support groups for the expiries.
        clock: The owner's trusted clock; the owner's budget is checked with
            it before any chunk is built and under the mutex before each
            chunk's write.

    Returns:
        The chunks committed now, in order.

    Raises:
        QuoteMemError: ``budget_exceeded`` when one position's rows alone exceed
            ``MAX_TX_ROWS``; ``target_changed`` when a compared head moved (the
            owner stops); ``delivery_busy`` once the owner's budget
            ends (the uncommitted chunks get no row); or any refusal of
            `Writer.commit_rows`.
    """
    from quotemem.store.writer import MAX_TX_ROWS

    if ticket.scope_kind is None or ticket.scope_id is None or ticket.run_id is None:
        raise QuoteMemError("scope_denied", "An owner's chunks need a scoped owner Run")
    positions = [position for entry in entries for position, _digest in entry.members]
    if len(set(positions)) != len(positions):
        raise QuoteMemError("binding_mismatch", "A chunked transaction names each position once")
    owner_budget(writer, owner_run_id, clock)
    done, prior_withholding = _journaled_state(
        writer, ticket.run_id, owner_run_id=owner_run_id, round_no=round_no, txn=txn,
        inspect_withholding=True,
    )
    row_limit = (
        WITHHOLD_CHUNK_ROWS if prior_withholding or any(entry.withholds for entry in entries)
        else CHUNK_ROWS
    )
    pending = [entry for entry in entries if entry.position_key not in done]
    committed: list[CommittedRows] = []
    limit = CHUNK_POSITIONS
    while pending:
        chunk = _Chunk(TxnRows(writer, cast(ScopeKind, ticket.scope_kind), ticket.scope_id,
                               writer.get_counter("commit_seq") + 1, updated_at, ticket.run_id))
        withheld: set[str] = set()
        for entry in pending[:limit]:
            saved = (dict(chunk.txn.written), dict(chunk.txn.deleted))
            grown = entry.build(chunk.txn)
            rows = len(chunk.txn.written) + len(chunk.txn.deleted) + _OVERHEAD
            if rows > MAX_TX_ROWS and not chunk.entries:
                raise QuoteMemError("budget_exceeded", "One position exceeds a transaction")
            if rows > row_limit and chunk.entries:
                chunk.txn.written, chunk.txn.deleted = saved
                break
            withheld |= grown
            chunk.entries.append(entry)
        expire_unusable(chunk.txn, withheld, groups_of=groups_of)
        heads = [head for entry in chunk.entries for head in entry.heads]
        with writer.mutex:
            chunk.txn.stamp()
            journal = _journal(writer, chunk.txn, owner_run_id=owner_run_id, round_no=round_no,
                               kind=txn, entries=[member for entry in chunk.entries
                                                  for member in entry.members])
            result = writer.commit_rows_within(
                ticket, chunk.txn.row_writes(), deletes=chunk.txn.row_deletes(), heads=heads,
                journal=journal, updated_at=updated_at, publish=publish,
                before_write=lambda: owner_budget(writer, owner_run_id, clock),
                limit=MAX_TX_ROWS if len(chunk.entries) > 1 else None,
            )
            if isinstance(result, int):
                limit = max(1, len(chunk.entries) // 2)  # a sharded journal took the room
                continue
        if result is not None:
            committed.append(result)
        pending = pending[len(chunk.entries):]
        limit = CHUNK_POSITIONS
    return committed


# Settlement -----------------------------------------------------------------


def _settled_entry(position: SettledPositionV1, *, owner_run_id: str) -> ChunkEntryV1:
    from quotemem.store.writer import ExpectedHead

    identity = PositionV1(position.position_key, position.extraction_ref, position.proposal_ref,
                          position.statement_digest, tuple(position.evidence_ids))

    def build(txn: TxnRows) -> set[str]:
        changed = withhold_spans(txn, position.withheld, reason="restated_forgotten",
                                 run_id=owner_run_id)
        revision_id: str | None = None
        new_id = new_ulid()
        if position.code == "merged_support":
            # New support restores a head a forget left unreadable.
            memory_id = cast(str, position.memory_id)
            head = txn.get("revision", cast(Row, txn.get("item", memory_id))["current_revision_id"])
            revision = restore_support(
                txn, memory_id, added=position.groups, source_candidate_id=new_id,
                policy_release_ref=cast(Row, head)["policy_release_ref"],
            )
            revision_id = revision["object_id"]
        candidate_row(txn, identity, code=position.code, entity_id=position.entity_id,
                      memory_id=position.memory_id, revision_id=revision_id, new_id=new_id)
        return changed

    head = position.head
    return ChunkEntryV1(
        position_key=position.position_key, resolution_digest=integrity_digest(position),
        build=build,
        heads=() if head is None else (ExpectedHead(
            head.memory_id, head.current_revision_id, head.lifecycle_state,
        ),),
        withholds=bool(position.withheld) or position.code == "denied_by_forget",
    )


def size_chunk_entry(
    writer: Writer, ticket: WriteTicket, entry: ChunkEntryV1, *, owner_run_id: str,
    round_no: RoundNo, txn: CommitTxn, updated_at: int, groups_of: GroupsOf,
) -> int:
    """The RowOps of a chunk holding only ``entry``, sized as `commit_chunked` builds it.

    The chunk is built exactly as `commit_chunked` builds one: the entry's own
    builders, the expiries its withheld spans cause, and its create-once
    journal row, stamped under the Writer mutex with the ``commit_seq`` it
    would take (``TxnRows.stamp``); `Writer.count_commit_rowops` sizes it in sizing
    mode. Nothing is written.

    Args:
        writer: The Writer.
        ticket: The owner's scoped publishing ticket.
        entry: The position.
        owner_run_id: The Add owner.
        round_no: The round.
        txn: ``settlement`` or ``rejection``.
        updated_at: The time.
        groups_of: Each candidate's support groups by ``proposal_ref``.

    Returns:
        The chunk's RowOps; ``MAX_TX_ROWS + 1`` when the Writer refuses its
        size (``budget_exceeded``), so a caller compares it with ``MAX_TX_ROWS``.

    Raises:
        QuoteMemError: ``scope_denied`` for an unscoped ticket; any other refusal
            of the build or of the sizing.
    """
    from quotemem.store.writer import MAX_TX_ROWS

    if ticket.scope_kind is None or ticket.scope_id is None or ticket.run_id is None:
        raise QuoteMemError("scope_denied", "An owner's chunks need a scoped owner Run")
    rows = TxnRows(writer, cast(ScopeKind, ticket.scope_kind), ticket.scope_id,
                   writer.get_counter("commit_seq") + 1, updated_at, ticket.run_id)
    expire_unusable(rows, entry.build(rows), groups_of=groups_of)
    with writer.mutex:
        rows.stamp()
        journal = _journal(writer, rows, owner_run_id=owner_run_id, round_no=round_no, kind=txn,
                           entries=entry.members)
        try:
            return writer.count_commit_rowops(ticket, rows.row_writes(), deletes=rows.row_deletes(),
                                              journal=journal, updated_at=updated_at,
                                              sizing_refs=frozenset({_SIZING_REF}))
        except QuoteMemError as error:
            if error.code != "budget_exceeded":
                raise
            return MAX_TX_ROWS + 1


_SIZING_REF: Final = cast(ArtifactId, "art_" + "0" * 43)
"""The artifact ids of a sized position: sizing reads none, and none is written."""


def worst_settled_position(
    parts: Sequence[EvidenceRecord], *, owner_run_id: str, unit_no: int, position: int = 0,
) -> SettledPositionV1:
    """The settlement position of one unit with the most rows.

    A forgotten restatement (``denied_by_forget``) is the only settled code
    that withholds spans; its worst case cites every part of the unit and
    withholds one span in each of up to ``MAX_MEMORY_SPANS`` of them, so its
    chunk writes its candidate row and one Evidence row per withheld part.
    Its artifact references are placeholders: sizing reads neither.

    Args:
        parts: The unit's Evidence parts.
        owner_run_id: The Add owner.
        unit_no: The unit's number.
        position: The position's number in the unit.

    Returns:
        The position, for `size_settled_position`.

    Raises:
        ValueError: ``parts`` is empty.
    """
    if not parts:
        raise ValueError("A unit has at least one part")
    cited = sorted(parts, key=lambda part: part.object_id)
    withheld = [QuoteSpanV1(evidence_id=part.object_id, byte_start=part.byte_start,
                            byte_end=part.byte_end)
                for part in cited[:MAX_MEMORY_SPANS] if part.byte_end > part.byte_start]
    return SettledPositionV1(
        position_key=(owner_run_id, unit_no, position), extraction_ref=_SIZING_REF,
        proposal_ref=_SIZING_REF, statement_digest="0" * 64, entity_id="user",
        evidence_ids=sorted({part.object_id for part in cited}), code="denied_by_forget",
        withheld=withheld,
    )


def size_settled_position(
    writer: Writer, ticket: WriteTicket, position: SettledPositionV1, *, owner_run_id: str,
    updated_at: int, groups_of: GroupsOf,
) -> int:
    """The RowOps of one settlement chunk holding only ``position``.

    The ``measure`` sizes `worst_settled_position` with it and compares the
    count with ``MAX_TX_ROWS``: chunking bounds everything but one position,
    so a plan whose worst position does not fit is refused before settlement.

    Args:
        writer: The Writer.
        ticket: The owner's scoped publishing ticket.
        position: The position.
        owner_run_id: The Add owner.
        updated_at: The time.
        groups_of: Each candidate's support groups by ``proposal_ref``.

    Returns:
        Its RowOps, or ``MAX_TX_ROWS + 1`` on the Writer's capacity refusal.
    """
    return size_chunk_entry(
        writer, ticket, _settled_entry(position, owner_run_id=owner_run_id),
        owner_run_id=owner_run_id, round_no=1, txn="settlement", updated_at=updated_at,
        groups_of=groups_of,
    )


def commit_settlement(
    writer: Writer, ticket: WriteTicket, *, owner_run_id: str,
    positions: Sequence[SettledPositionV1], updated_at: int, publish: Publish, clock: Clock,
    groups_of: GroupsOf | None = None,
) -> list[CommittedRows]:
    """Commit the settlement transaction of one owner, chunked.

    Every position gets its candidate row; a ``denied_by_forget`` position's
    located spans are withheld in the same chunk, with the expiries they cause;
    a restatement's head is compared, and a moved head stops the owner with
    ``target_changed``.

    Args:
        writer: The single Writer.
        ticket: The owner's ticket.
        owner_run_id: The Add owner.
        positions: the records in canonical order; each ``ProposalV2`` and
            ``llm_output`` they reference is already stored.
        updated_at: The transactions' time.
        publish: The read model's publication of each chunk.
        clock: The owner's trusted clock for its budget checks.
        groups_of: Reads a pending candidate's support groups; by default from
            its stored ``ProposalV2``.

    Returns:
        The chunks committed now.

    Raises:
        QuoteMemError: ``target_changed`` for a moved restatement head; else as
            `commit_chunked`.
    """
    return commit_chunked(
        writer, ticket, owner_run_id=owner_run_id, round_no=1, txn="settlement",
        entries=[_settled_entry(position, owner_run_id=owner_run_id) for position in positions],
        updated_at=updated_at, publish=publish, groups_of=groups_of or proposal_groups(writer),
        clock=clock,
    )


# Rejection --------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DispositionV1:
    """A position's final disposition as a candidate row records it.

    Attributes:
        code: The position's single reason code.
        entity_id: Its resolved primary; None when unresolved or not yet minted.
        memory_id: The memory it wrote or the one already stating its fact.
        revision_id: The revision it wrote; set exactly for a formal code.
        targets: Ended, contradicted or judged targets (add-items).
        judge_result_ref: Its latest ``result_index:judge``.
        hold_evidence: Its latest review issued a tombstone.
    """

    code: ReasonCode
    entity_id: str | None
    memory_id: str | None = None
    revision_id: str | None = None
    targets: tuple[str, ...] = ()
    judge_result_ref: str | None = None
    hold_evidence: bool = False


def mirrored(disposition: DispositionV1) -> DispositionV1:
    """A duplicate's disposition, from its representative's.

    ``duplicate_in_add`` with the representative's memory when that outcome is
    formal or unchanged; else the representative's code, its entity and review.
    """
    category = outcome_rule(disposition.code).category
    if category in ("added", "corrected", "supplemented", "unchanged"):
        return DispositionV1("duplicate_in_add", disposition.entity_id,
                             memory_id=disposition.memory_id,
                             judge_result_ref=disposition.judge_result_ref)
    return DispositionV1(disposition.code, disposition.entity_id,
                         judge_result_ref=disposition.judge_result_ref,
                         hold_evidence=disposition.hold_evidence)


def write_disposition(
    txn: TxnRows, position: PositionV1, disposition: DispositionV1, *,
    touch: ReentryTouchV1 | None = None, new_id: str | None = None,
) -> Row:
    """Write one position's candidate row with its disposition."""
    return candidate_row(
        txn, position, code=disposition.code, entity_id=disposition.entity_id,
        memory_id=disposition.memory_id, revision_id=disposition.revision_id,
        targets=disposition.targets, judge_result_ref=disposition.judge_result_ref,
        hold_evidence=disposition.hold_evidence, touch=touch, new_id=new_id,
    )


def expire_row(txn: TxnRows, position: PositionV1) -> Row:
    """Expire a re-entered pending row whose rebuild left no usable group.

    The expiry is the selection's one durable write: no touch, and the wake key
    of an earlier rebuild is nulled in the same write.
    """
    if position.candidate_id is None:
        raise QuoteMemError("binding_mismatch", "Only a re-entered position expires at its rebuild")
    stored = txn.get("candidate", position.candidate_id)
    if stored is None or stored["status"] != "pending":
        raise QuoteMemError("binding_mismatch", "A rebuild expires its own pending row")
    return txn.put("candidate", {
        **stored, "status": "expired", "reason_code": "evidence_invalidated",
        "hold_evidence": False, "last_wake_digest": None, "commit_seq": txn.commit_seq,
        "terminal_at": txn.now,
    })


@dataclass(frozen=True, slots=True)
class FollowerV1:
    """A duplicate written with its representative."""

    position: PositionV1

    @classmethod
    def of(cls, duplicate: DuplicatePositionV1) -> FollowerV1:
        """The duplicate's candidate identity; it is a position of this owner."""
        return cls(PositionV1(duplicate.position_key, duplicate.extraction_ref,
                              duplicate.proposal_ref, duplicate.statement_digest,
                              tuple(duplicate.evidence_ids)))


@dataclass(frozen=True, slots=True)
class RejectionEntryV1:
    """One position of a round's rejection transaction.

    Exactly one of ``disposition`` and ``expire`` is set, or neither for a touch
    only: a quiet re-entered member keeps its disposition and row, and a
    re-entered member whose disposition a component writes is touched here.

    Attributes:
        order: Its canonical key ``(σ, position_key)``; entries commit in this order.
        position: Its candidate identity.
        disposition: Its final disposition outside every component.
        withheld: Spans its ``denied_by_forget`` withholds as ``restated_forgotten``.
        touch: Its re-entry touch at this owner and round: every selected
            member in round 1, and a round-2 review's rewrite in round 2.
        expire: Its rebuild left no usable group: the stored pending row expires
            ``evidence_invalidated`` with a null wake key.
        followers: Duplicates that joined it; their rows mirror its
            disposition in the same chunk.
    """

    order: tuple[Sigma, PositionKey]
    position: PositionV1
    disposition: DispositionV1 | None = None
    withheld: tuple[QuoteSpanV1, ...] = ()
    touch: ReentryTouchV1 | None = None
    expire: bool = False
    followers: tuple[FollowerV1, ...] = ()

    def __post_init__(self) -> None:
        """Check that the entry writes what its kind allows, and only that."""
        if self.order[1] != self.position.position_key:
            raise QuoteMemError("binding_mismatch", "A rejection entry is ordered by its position")
        if self.expire and (self.disposition is not None or self.touch is not None):
            raise QuoteMemError("binding_mismatch", "An expiry at rebuild writes nothing else")
        if self.disposition is None and not self.expire and self.touch is None:
            raise QuoteMemError("binding_mismatch", "A rejection entry writes a row")
        if (self.withheld or self.followers) and self.disposition is None:
            raise QuoteMemError("binding_mismatch", "Only a disposition withholds or is mirrored")
        if self.withheld and self.disposition is not None and (
            self.disposition.code != "denied_by_forget"
        ):
            raise QuoteMemError("binding_mismatch",
                                "Only a forgotten restatement withholds its spans")
        if self.touch is not None and self.touch.position_key != self.position.position_key:
            raise QuoteMemError("binding_mismatch", "A touch belongs to its own position")

    def digest(self) -> IntegrityDigest:
        """The content-free integrity digest of what the entry writes."""
        disposition = self.disposition
        return integrity_digest({
            "disposition": None if disposition is None else {
                "code": disposition.code, "entity_id": disposition.entity_id,
                "memory_id": disposition.memory_id, "revision_id": disposition.revision_id,
                "targets": sorted(set(disposition.targets)),
                "judge_result_ref": disposition.judge_result_ref,
                "hold_evidence": disposition.hold_evidence,
            },
            "withheld": sorted([span.evidence_id, span.byte_start, span.byte_end]
                               for span in self.withheld),
            "touch": None if self.touch is None else [
                self.touch.reentry_count, self.touch.last_wake_digest,
                self.touch.last_input_digest,
            ],
            "expire": self.expire,
        })


def _rejection_entry(entry: RejectionEntryV1, *, run_id: str) -> ChunkEntryV1:
    disposition = entry.disposition
    mirror = None if disposition is None else mirrored(disposition)

    def build(txn: TxnRows) -> set[str]:
        if entry.expire:
            expire_row(txn, entry.position)
            return set()
        if disposition is None:
            assert entry.touch is not None
            touch_row(txn, entry.position, entry.touch)
            return set()
        changed = withhold_spans(txn, entry.withheld, reason="restated_forgotten",
                                 run_id=run_id)
        write_disposition(txn, entry.position, disposition, touch=entry.touch)
        assert mirror is not None
        for follower in entry.followers:
            write_disposition(txn, follower.position, mirror)
        return changed

    follower_digest = None if mirror is None else integrity_digest({
        "code": mirror.code, "entity_id": mirror.entity_id, "memory_id": mirror.memory_id,
        "judge_result_ref": mirror.judge_result_ref, "hold_evidence": mirror.hold_evidence,
    })
    return ChunkEntryV1(
        position_key=entry.position.position_key, resolution_digest=entry.digest(), build=build,
        followers=tuple((follower.position.position_key, cast(IntegrityDigest, follower_digest))
                        for follower in entry.followers),
        withholds=bool(entry.withheld) or (
            disposition is not None and disposition.code == "denied_by_forget"
        ),
    )


def commit_rejection(
    writer: Writer, ticket: WriteTicket, *, owner_run_id: str, round_no: RoundNo,
    entries: Sequence[RejectionEntryV1], updated_at: int, publish: Publish, clock: Clock,
    groups_of: GroupsOf | None = None,
) -> list[CommittedRows]:
    """Commit one round's rejection transaction, chunked, before its components.

    Every position whose disposition writes no memory gets its candidate row;
    every selected re-entered member is touched exactly once per round
    operation, a quiet one only touched; a rebuild without a
    usable group expires its row; a ``denied_by_forget`` withholds its spans,
    and the pending candidates those spans leave without a usable group expire
    in the same chunk. A replay skips every position a journal covers.

    Args:
        writer: The single Writer.
        ticket: The owner's ticket.
        owner_run_id: The Add owner.
        round_no: The round.
        entries: The round's rejection positions, each once; committed in
            canonical order.
        updated_at: The transactions' time.
        publish: The read model's publication of each chunk.
        clock: The owner's trusted clock for its budget checks.
        groups_of: Reads a pending candidate's support groups; by default from
            its stored ``ProposalV2``.

    Returns:
        The chunks committed now.

    Raises:
        QuoteMemError: As `commit_chunked`.
    """
    ordered = sorted(entries, key=lambda entry: entry.order)
    return commit_chunked(
        writer, ticket, owner_run_id=owner_run_id, round_no=round_no, txn="rejection",
        entries=[_rejection_entry(entry, run_id=owner_run_id) for entry in ordered],
        updated_at=updated_at, publish=publish, groups_of=groups_of or proposal_groups(writer),
        clock=clock,
    )
