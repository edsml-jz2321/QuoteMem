"""The Add owner's completion: request results, its prefix split and its record (C.2 #8).

A request's first result is the copied ``add_outcome:{scope}:{request_key}``
control (A.3.9), written by its completing owner before release and visible only
once that owner succeeded. It never changes after that first completion, so an
exact resend returns the same six counts (INV:E02). ``completion_requests`` splits
an owner's frozen prefix into the requests it completes and those an earlier
owner froze; ``owner_positions`` reads every position the owner settled from its
durable records; C.10's attribution (``core/add_outcome.attribute_outcomes``)
consumes both, and ``record_add_completion`` freezes the resulting completion
record (E.2.1's ``result_index:add_completion``) and its results before release
(``store/owner_lease.release_finalize``). ``stored_completion`` finds a record an
interrupted invocation stored but never attached, which its owner then attaches.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from aml.core.add_outcome import AddCompletionRecordV1, CompletedAddOutcomeV1, PositionOutcome
from aml.core.commit_plan import CommitJournalV1
from aml.core.controls import ControlRowV1
from aml.core.entities import MemorySnapshotV2, PositionKey, QuoteSpanV1, ReasonCode
from aml.core.errors import AMLError
from aml.core.extraction_records import ErasedProposalV1
from aml.core.identity import ArtifactId, ContentBindingDigest
from aml.core.outcome_table import outcome_rule
from aml.core.proposal import ProposalV2
from aml.core.query import ScopeV1
from aml.core.run_inputs import SessionDurableRunInputV1
from aml.ingestion.authorization import check_add_grant
from aml.store.extraction_erasure import erased_extraction_records
from aml.store.metadata import StrictModel

if TYPE_CHECKING:
    from aml.ingestion.finalize_receipts import FinalizeReceipt
    from aml.ingestion.service import AddReceipt
    from aml.providers.auth import PrincipalGrant
    from aml.store.finalize_ticket import FinalizeLeaseTicket
    from aml.store.writer import RowWrite, Writer


def request_key(
    writer: Writer, scope: ScopeV1, session_id: str, request_id: str,
) -> ContentBindingDigest:
    """Bind one external request to its scope-keyed receipt identity."""
    return writer.bind_content("add-request-key", {
        "scope_id": scope.scope_id, "session_id": session_id, "request_id": request_id,
    })


def completed_result(writer: Writer, scope: ScopeV1, key: str) -> CompletedAddOutcomeV1 | None:
    """Return a request's first result once its completing owner has succeeded.

    Raises:
        AMLError: If a stored result changed its request or completion owner.
    """
    raw = writer.get_control(f"add_outcome:{scope.scope_id}:{key}")
    if raw is None:
        return None
    value = ControlRowV1.model_validate(raw).value
    if not isinstance(value, CompletedAddOutcomeV1) or value.request_key != key:
        raise AMLError("binding_mismatch", "Add result changed its request binding")
    owner = writer.get_row("run", value.completed_by_run_id)
    if owner is None or (owner["scope_kind"], owner["scope_id"], owner["run_kind"]) != (
        scope.scope_kind, scope.scope_id, "finalize",
    ):
        raise AMLError("binding_mismatch", "Add result lost its completion owner")
    if value.state == "skipped":
        # An operator skipped the owner's quarantined cycle: the owner failed
        # ``input_abandoned`` in the same transaction (``store.quarantine_skip``).
        if (owner["status"], owner["failure_code"]) != ("failed", "input_abandoned"):
            raise AMLError("binding_mismatch", "A skipped Add result lost its skipped owner")
        return value
    if (owner["status"], owner["stage"]) != ("succeeded", "release"):
        return None
    return value


def read_add_outcome(
    writer: Writer, scope: ScopeV1, receipt: AddReceipt, *, grant: PrincipalGrant,
    clock: Callable[[], int],
) -> CompletedAddOutcomeV1 | None:
    """Point-read the exact accepted request; newer scope work cannot change it."""
    with writer.mutex:
        writer.validate_storage_owner()
        check_add_grant(grant, scope, clock())
        if writer.scope_quarantined(scope.scope_id):
            raise AMLError("scope_quarantined", "Completed Add scope requires explicit repair")
        run = writer.get_row("run", receipt.run_id)
        if run is None or (run["scope_kind"], run["scope_id"]) != (
            scope.scope_kind, scope.scope_id,
        ):
            raise AMLError("binding_mismatch", "Add result requires its reception owner")
        selected = writer.read_logical(run["input_ref"], "run_input")
        if not isinstance(selected, SessionDurableRunInputV1) or (
            selected.visibility_commit_seq != receipt.visibility_commit_seq
        ):
            raise AMLError("binding_mismatch", "Add result changed its durable receipt")
        return completed_result(writer, scope, selected.request_key)


def read_completed_add(
    writer: Writer, scope: ScopeV1, session_id: str, request_id: str,
) -> CompletedAddOutcomeV1 | None:
    """Return one request's first visible result for completion inspection.

    The caller holds the Writer mutex and has already checked its grant.
    """
    writer.validate_storage_owner()
    return completed_result(writer, scope, request_key(writer, scope, session_id, request_id))


type RequestCoordinate = tuple[str, str]
"""One accepted request: ``(session_id, request_id)``."""


@dataclass(frozen=True, slots=True)
class OwnerRequests:
    """An owner's frozen prefix split by first completion (C.10; INV:D20).

    Attributes:
        keys: Every prefix request's receipt identity, by its coordinates.
        order: Each request's reception position, its ordering commit and
            coordinates, by receipt identity.
        sources: Each prefix Evidence row's request, by Evidence id.
        pending: The requests without a visible result: this owner completes them.
        finished: The requests whose first result an earlier owner froze.
        cause: The newest pending request in reception order, the Add whose
            processing caused this owner's effects; None when it completes none.
    """

    keys: Mapping[RequestCoordinate, ContentBindingDigest]
    order: Mapping[ContentBindingDigest, tuple[int, RequestCoordinate]]
    sources: Mapping[str, ContentBindingDigest]
    pending: frozenset[ContentBindingDigest]
    finished: frozenset[ContentBindingDigest]
    cause: ContentBindingDigest | None

    def newest(self, evidence_ids: Iterable[str]) -> ContentBindingDigest:
        """Return the newest request among one position's own source spans (INV:D20).

        A position refused at admission has no admitted span; its unit's fresh
        sources name its request.

        Raises:
            AMLError: ``binding_mismatch`` when a source lies outside the owner's
                prefix or no source is named.
        """
        requests: set[ContentBindingDigest] = set()
        for evidence_id in evidence_ids:
            key = self.sources.get(evidence_id)
            if key is None:
                raise AMLError(
                    "binding_mismatch", "A position's source lies outside the owner prefix",
                )
            requests.add(key)
        if not requests:
            raise AMLError("binding_mismatch", "A position's attribution names no source")
        return max(requests, key=self.order.__getitem__)


def completion_requests(
    writer: Writer, scope: ScopeV1, receipts: Iterable[FinalizeReceipt], *,
    check: Callable[[], None],
) -> OwnerRequests:
    """Split an owner's frozen prefix into the requests it completes and frozen ones.

    A request without a visible first result is pending: this owner completes
    it. A request an earlier succeeded owner completed keeps its result, which
    froze at that first completion.

    Args:
        writer: The sole Writer, read outside its mutex.
        scope: The owner's exact scope.
        receipts: Every accepted request at or below the owner's receipt cutoff
            (``inspect_finalize_receipts(...).past``).
        check: Caller lifetime check between reads.

    Returns:
        The split, with each request's reception order and Evidence rows.

    Raises:
        AMLError: ``parity_mismatch`` when the prefix repeats a request, an
            Evidence row, or a receipt lacks its one ordering binding.
    """
    keys: dict[RequestCoordinate, ContentBindingDigest] = {}
    order: dict[ContentBindingDigest, tuple[int, RequestCoordinate]] = {}
    sources: dict[str, ContentBindingDigest] = {}
    pending: set[ContentBindingDigest] = set()
    finished: set[ContentBindingDigest] = set()
    for receipt in receipts:
        check()
        coordinate = receipt.session_id, receipt.request_id
        bindings = receipt.inventory.receipts
        if coordinate in keys or len(bindings) != 1:
            raise AMLError("parity_mismatch", "Owner prefix repeats or lost an accepted request")
        key = request_key(writer, scope, *coordinate)
        keys[coordinate] = key
        order[key] = bindings[0].visibility_commit_seq, coordinate
        for header in receipt.headers:
            if sources.setdefault(header.object_id, key) != key:
                raise AMLError("parity_mismatch", "Owner prefix repeats a source row")
        (pending if completed_result(writer, scope, key) is None else finished).add(key)
    return OwnerRequests(
        keys=MappingProxyType(keys), order=MappingProxyType(order),
        sources=MappingProxyType(sources), pending=frozenset(pending),
        finished=frozenset(finished),
        cause=max(pending, key=order.__getitem__) if pending else None,
    )


def owner_positions(
    writer: Writer, owner_run_id: str, requests: OwnerRequests, *,
    admitted: Collection[PositionKey],
    refused: Iterable[tuple[PositionKey, ReasonCode, Sequence[str]]],
) -> list[PositionOutcome]:
    """Read every position the owner settled from its durable records (C.2 #8, C.10; INV:D19).

    The owner's create-once commit journals (``CommitJournalV1``, C.13) name
    every candidate row its settlement, rejection and component transactions
    wrote, in both rounds and across its invocations: what settlement decided
    before the judge (B.8 #1-#3, B.7.3's unresolved primaries), each round's
    final code, the duplicates written with their representative (B.8 #3) and
    the re-entered rows it touched or settled (C.3). A position's final
    disposition is its row as it stands now, so a resumed owner counts what it
    committed before its stop, a component its replay skips included. A
    re-entered position of an earlier owner counts only with a formal effect,
    its other dispositions staying with its frozen receipt (C.3). A position
    admission refused has no row (B.7.5); its unit's sources name its request.
    An item's anchors (E.2.1) are entity ids only: a written version's anchors
    after the entity verdict, else the row's resolved primary. Every input is
    durable, so an invocation that repeats a completion computes it again
    exactly.

    Args:
        writer: The sole Writer.
        owner_run_id: The completing owner.
        requests: Its prefix split (``completion_requests``), which names each
            position's newest own request in reception order (INV:D20).
        admitted: Every position the owner admitted; each must have its row.
        refused: Each position admission refused, with its code and its unit's
            fresh Evidence parts.

    Returns:
        One outcome per position, refused ones included, sorted by position.

    Raises:
        AMLError: ``binding_mismatch`` when an admitted position has no row or
            a position's sources lie outside the owner's prefix;
            ``reference_missing`` for a journaled row or version that is gone.
    """
    rows: dict[PositionKey, dict[str, Any]] = {}
    for _reference, body in writer.find_logical_candidates(owner_run_id, "write_set"):
        if not isinstance(body, CommitJournalV1) or body.owner_run_id != owner_run_id:
            continue
        for written in body.written:
            if written.kind != "candidate":
                continue
            row = writer.get_row("candidate", written.object_id)
            if row is None:
                raise AMLError("reference_missing", "A journaled candidate row is gone")
            rows[(row["owner_run_id"], row["unit_no"], row["position"])] = row
    if set(admitted) - set(rows):
        raise AMLError("binding_mismatch", "An admitted position has no candidate row")
    positions: list[PositionOutcome] = []
    erased: dict[str, dict[PositionKey, ErasedProposalV1]] = {}
    for key, row in sorted(rows.items()):
        code: ReasonCode = row["reason_code"]
        revision_id: str | None = row["revision_id"]
        if key[0] != owner_run_id and not outcome_rule(code).formal:
            continue
        spans: Sequence[QuoteSpanV1]
        if writer.get_row("artifact", row["proposal_ref"]) is None:
            if key[0] not in erased:
                _replies, erased[key[0]] = erased_extraction_records(writer, key[0])
            metadata = erased[key[0]].get(key)
            if metadata is None or metadata.proposal_ref != row["proposal_ref"]:
                raise AMLError("binding_mismatch", "A candidate lost its proposal accounting")
            spans = metadata.quote_spans
        else:
            proposal = writer.read_logical(row["proposal_ref"], "proposal_snapshot")
            if not isinstance(proposal, ProposalV2):
                raise AMLError("binding_mismatch", "A candidate's proposal is not a ProposalV2")
            spans = proposal.quote_spans
        if revision_id is not None:
            version = writer.get_row("revision", revision_id)
            if version is None:
                raise AMLError("reference_missing", "A published candidate lost its revision")
            snapshot = MemorySnapshotV2.model_validate(version["snapshot"])
            entities = (snapshot.entity_id, *snapshot.about_entity_ids)
        else:
            entities = () if row["entity_id"] is None else (row["entity_id"],)
        positions.append(PositionOutcome(
            position_key=key, reason_code=code, status=row["status"], revision_id=revision_id,
            request_key=requests.newest(row["evidence_ids"]),
            memory_ids=() if row["memory_id"] is None else (row["memory_id"],),
            target_memory_ids=tuple(row["target_memory_ids"]), entity_ids=entities,
            quote_spans=tuple(sorted({(span.evidence_id, span.byte_start, span.byte_end)
                                      for span in spans})),
            judge_result_ref=row["judge_result_ref"],
        ))
    positions.extend(
        PositionOutcome(position_key=key, reason_code=code, status=None, revision_id=None,
                        request_key=requests.newest(sources))
        for key, code, sources in refused
    )
    return sorted(positions, key=lambda position: position.position_key)


_COMPLETING_STAGES = frozenset({"extract", "settle", "review", "commit"})
"""Owner stages after the frozen input is selected and before release (A.3.10)."""


def _owned_control(
    writer: Writer, key: str, value: StrictModel, owner_run_id: str, now: int,
) -> RowWrite:
    """Prepare one owner-written Control as the next version of whatever the key holds."""
    from aml.store.writer import RowWrite

    previous = writer.get_control(key)
    version = 0 if previous is None else ControlRowV1.model_validate(previous).version + 1
    return RowWrite("control", writer.materialize_control(
        key, value.model_dump(mode="json"), version, now, owner_run_id,
    ))


def stored_completion(
    writer: Writer, owner_run_id: str, recovery_count: int,
) -> AddCompletionRecordV1 | None:
    """The completion record an interrupted invocation stored but never attached.

    ``record_add_completion`` stores the record's Artifact before the transaction
    that attaches it to the owner, so a process that dies between the two leaves
    it unattached. That record froze the owner's first completion: the owner's
    next invocation attaches it, reusing the put-once Artifact, instead of
    computing another (C.13 R13). A record from before an explicit resume carries
    another ``recovery_count`` and is not this attempt's.

    Args:
        writer: The sole Writer.
        owner_run_id: The owner Run that created the record.
        recovery_count: The owner's durable automatic/explicit resume total.

    Returns:
        The stored record, or None.

    Raises:
        AMLError: ``binding_mismatch`` for a stored record naming another owner
            or for more than one record of this attempt.
    """
    found = [record for _, record in writer.find_logical_candidates(owner_run_id, "result_index")
             if isinstance(record, AddCompletionRecordV1)
             and record.recovery_count == recovery_count]
    if len(found) > 1 or any(record.owner_run_id != owner_run_id for record in found):
        raise AMLError("binding_mismatch", "The owner's stored completion cannot be proved")
    return found[0] if found else None


def record_add_completion(
    writer: Writer, ticket: FinalizeLeaseTicket, record: AddCompletionRecordV1, *,
    grant: PrincipalGrant, clock: Callable[[], int],
) -> ArtifactId:
    """Freeze the owner's completion record and its request results, before release.

    C.2 step 8's completion. The record is first stored as its put-once
    ``result_index:add_completion`` Artifact (chunked with ``part`` when large,
    A0.7) in that Artifact's own Writer holds; one Writer scope then writes one
    ``add_outcome:{scope}:{request_key}`` result per completed request, the
    ``add_completion:{scope}`` bookkeeping that release validates, and the owner
    Run's ``result_ref``; the owner moves to ``release``. The results stay
    invisible until release settles the owner (``completed_result``), and a result
    an earlier owner froze is never replaced. Repeating the same record writes
    nothing.

    Args:
        writer: The sole Writer.
        ticket: The held owner's issued cycle.
        record: The owner's record, attributed by ``core/add_outcome.attribute_outcomes``
            over ``completion_requests``.
        grant: The owner principal's grant.
        clock: Trusted millisecond action clock.

    Returns:
        The record's Artifact, now the owner's ``result_ref``.

    Raises:
        AMLError: ``binding_mismatch`` when the record names another owner or
            recovery count, or the owner's allowance is not active;
            ``operation_conflict`` when an earlier owner froze a request's result;
            ``parity_mismatch`` when the owner completed with another record;
            ``invalid_run_transition`` before the owner's frozen input;
            ``lease_lost`` for a lost cycle.
    """
    from aml.core.controls import AddCompletionV1, FinalizeBudgetV2, HeldFinalizeLeaseV1
    from aml.store.request_completion import read_budget
    from aml.store.writer import RowWrite

    staged: ArtifactId | None = None
    with writer._write_scope():  # pyright: ignore[reportPrivateUsage]
        run, lease, effect = writer._finalize_precondition(ticket)  # pyright: ignore[reportPrivateUsage]
        scope = ScopeV1(scope_kind=run["scope_kind"], scope_id=run["scope_id"])
        staged_at = clock()
        check_add_grant(grant, scope, staged_at, run["principal_id"])
        if not isinstance(lease.value, HeldFinalizeLeaseV1) or ticket.state != "held":
            raise AMLError("lease_lost", "Add completion requires the held owner")
        if record.owner_run_id != ticket.owner_run_id:
            raise AMLError("binding_mismatch", "Completion record names another owner")
        completing = run["result_ref"] is None and run["status"] == "running" and (
            run["input_ref"] is not None and run["stage"] in _COMPLETING_STAGES
        )
    if completing:
        # The record's put-once Artifact is stored in its own short Writer holds
        # before the completing transaction, which then only attaches it. Stored
        # but unattached is what an interrupted completion leaves; the owner's next
        # invocation attaches it (``stored_completion``, C.13 R13).
        staged = writer.put_logical_artifact(effect, "result_index", record, created_at=staged_at)
    with writer._write_scope():  # pyright: ignore[reportPrivateUsage]
        run, lease, effect = writer._finalize_precondition(ticket)  # pyright: ignore[reportPrivateUsage]
        scope = ScopeV1(scope_kind=run["scope_kind"], scope_id=run["scope_id"])
        now = clock()
        check_add_grant(grant, scope, now, run["principal_id"])
        if not isinstance(lease.value, HeldFinalizeLeaseV1) or ticket.state != "held":
            raise AMLError("lease_lost", "Add completion requires the held owner")
        if record.owner_run_id != ticket.owner_run_id:
            raise AMLError("binding_mismatch", "Completion record names another owner")
        if run["result_ref"] is not None:
            stored = writer._read_owned_business_artifact(  # pyright: ignore[reportPrivateUsage]
                effect, run["result_ref"], "result_index",
            )
            if stored != record:
                raise AMLError("parity_mismatch", "The owner completed with another record")
            return run["result_ref"]
        if run["status"] != "running" or run["input_ref"] is None or (
            run["stage"] not in _COMPLETING_STAGES
        ):
            raise AMLError("invalid_run_transition", "Add completion requires the frozen input")
        writer._bound_input(run)  # pyright: ignore[reportPrivateUsage]
        budget = read_budget(writer, scope, ticket.owner_run_id)
        if budget is None or not isinstance(budget.value, FinalizeBudgetV2) or (
            budget.value.state != "active"
        ):
            raise AMLError("binding_mismatch", "Add completion requires the owner's allowance")
        if budget.value.recovery_count != record.recovery_count:
            raise AMLError("binding_mismatch", "Completion record predates the owner's repair")
        owner = ticket.owner_run_id
        rows: list[RowWrite] = []
        for request in record.completed_requests:
            if completed_result(writer, scope, request.request_key) is not None:
                raise AMLError("operation_conflict", "An earlier owner already froze this result")
            rows.append(_owned_control(
                writer, f"add_outcome:{scope.scope_id}:{request.request_key}",
                CompletedAddOutcomeV1(
                    request_key=request.request_key, completed_by_run_id=owner,
                    recovery_count=record.recovery_count, outcome=request.outcome,
                ), owner, now,
            ))
        reference = staged if staged is not None and (
            writer.get_row("artifact", staged) is not None
        ) else writer.put_logical_artifact(effect, "result_index", record, created_at=now)
        rows.append(_owned_control(writer, f"add_completion:{scope.scope_id}", AddCompletionV1(
            owner_run_id=owner, result_ref=reference, recovery_count=record.recovery_count,
            request_count=len(record.completed_requests), source_count=0,
        ), owner, now))
        run.update(result_ref=reference, stage="release", updated_at=now)
        rows.append(RowWrite("run", writer._parse_row("run", run)))  # pyright: ignore[reportPrivateUsage]
        _, plan = writer._prepare_rows(rows, effect)  # pyright: ignore[reportPrivateUsage]
        check_add_grant(grant, scope, clock(), run["principal_id"])
        writer._write_plan(plan)  # pyright: ignore[reportPrivateUsage]
        return reference
