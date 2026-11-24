"""An operator's skip of a quarantined Add cycle.

A scope-local fault quarantines one user's scope (``owner_lease.quarantine_scope``).
The explicit repair resumes the same owner (``owner_lease.repair_quarantine``),
which helps only once the fault's cause is cleared. A fault that recurs with the
same input is skipped here instead, only on an operator's explicit request,
never automatically.

In one Writer transaction, the skip:

- fails the receipt's owner ``input_abandoned``, and every unended child of it;
- cancels its lease (``repair_skip``);
- writes each request the cycle froze and had not completed as completed, with
  ``state`` ``skipped`` and the counts of what the owner committed before its
  fault (attribution, as a completion counts them);
- deletes the quarantine receipt.

The read model then lifts the scope's fence (``publish_finalize_cancelled``) and,
as a release does, the in-memory Add admission stops holding the owner as running,
so the user's next Add admits a new cycle without a restart.
Nothing committed is lost: the owner's formal commits stand, and its pending
positions stay for re-entry. Afterwards the stored extraction outputs of every
unit that settled nothing are deleted in bounded delete-only transactions:
their replies, proposals and statement embeddings. No candidate cites them,
and no later owner reads them, since every request they came from has
completed. The next Add admits a new cycle once ``owner_lease.cancelled_settled``
proves this one settled; that cycle's prefix holds none of the skipped requests.

Privacy comes first. A skip is refused while a tomb, an erasure, a pending privacy
operation or a privacy epoch change holds the scope, as a repair is. A
parity-failed owner is never skipped.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from quotemem.core.add_outcome import CompletedAddOutcomeV1, attribute_outcomes
from quotemem.core.controls import (
    CancelledFinalizeLeaseV1,
    ControlRowV1,
    FinalizeBudgetRefusalV4,
    FinalizeBudgetV2,
    HeldFinalizeLeaseV1,
    PreparingFinalizeLeaseV1,
    ScopeQuarantineV1,
)
from quotemem.core.errors import QuoteMemError
from quotemem.core.query import ScopeV1
from quotemem.core.run_inputs import FinalizeOwnerRunInputV1

if TYPE_CHECKING:
    from quotemem.store.writer import RowWrite, Writer


@dataclass(frozen=True, slots=True)
class SkippedCycle:
    """What one skip did: its owner, the requests it completed and the records it dropped."""

    owner_run_id: str
    skipped_requests: int
    dropped_records: int


def skip_quarantined_cycle(
    writer: Writer, scope: ScopeV1, owner_run_id: str, *,
    budget_version: int, recovery_count: int, now: int,
) -> SkippedCycle:
    """Skip the scope's quarantined cycle and re-admit the scope (the module's transaction).

    The caller holds the Writer mutex and has proved that the owner's pool
    callbacks joined. The request names the owner and the budget version and
    resume count it observed, so a cycle that moved since is refused rather than
    skipped.

    Args:
        writer: The sole Writer.
        scope: The quarantined scope.
        owner_run_id: The owner the quarantine receipt names.
        budget_version: The observed ``finalize_budget`` version.
        recovery_count: The observed counted resumes.
        now: The skip transaction's trusted clock.

    Returns:
        The skipped owner, how many requests it completed as skipped and how
        many stored extraction records it dropped.

    Raises:
        QuoteMemError: ``operation_conflict`` when the receipt, lease or budget no
            longer name the observed cycle; ``binding_mismatch`` for an owner
            that cannot be proved; ``parity_mismatch`` for a parity-failed owner;
            ``index_rebuilding`` while privacy work owns the scope;
            ``clock_rollback`` for a clock behind the lease.
    """
    from quotemem.core.admission import admission_identity, finalize_admission
    from quotemem.ingestion.finalize_receipts import inspect_finalize_receipts
    from quotemem.store.add_outcome import completion_requests, owner_positions
    from quotemem.store.owner_lease import (
        _adopted_cycles,  # pyright: ignore[reportPrivateUsage]
        _paused_owners,  # pyright: ignore[reportPrivateUsage]
        owner_stopped,
    )
    from quotemem.store.read_model import StoreReadModel
    from quotemem.store.request_completion import read_budget
    from quotemem.store.writer import RowWrite

    with writer.write_scope():
        raw = writer.get_control(f"scope_quarantine:{scope.scope_id}")
        receipt = None if raw is None else ControlRowV1.model_validate(raw)
        raw_lease = writer.get_control(f"lease:finalize:{scope.scope_id}")
        lease = None if raw_lease is None else ControlRowV1.model_validate(raw_lease)
        value = None if lease is None else lease.value
        owner = writer.get_row("run", owner_run_id)
        if raw is None or receipt is None or not isinstance(receipt.value, ScopeQuarantineV1) or (
            lease is None or not isinstance(value, PreparingFinalizeLeaseV1 | HeldFinalizeLeaseV1)
            or owner is None
            or (receipt.value.owner_run_id, receipt.value.finalize_cycle, value.owner_run_id)
            != (owner_run_id, value.lease_generation, owner_run_id)
        ):
            raise QuoteMemError("operation_conflict",
                                "The quarantine no longer names the scope's cycle")
        family, digest, _ = admission_identity(finalize_admission(
            principal_id=owner["principal_id"], scope=scope, cycle=value.lease_generation,
        ), writer.bind_content)
        if (
            (owner["run_kind"], owner["commit_mode"], owner["target_kind"], owner["target_id"])
            != ("finalize", "orchestrator", "scope", scope.scope_id)
            or (owner["scope_kind"], owner["scope_id"]) != (scope.scope_kind, scope.scope_id)
            or owner["parent_run_id"] is not None or owner["run_generation"] != 0
            or (owner["idempotency_key"], owner["admission_digest"]) != (family, digest)
            or lease.updated_by_run_id != owner_run_id
        ):
            raise QuoteMemError("binding_mismatch",
                                "A quarantine skip cannot prove its original cycle")
        if (owner["status"], owner["failure_code"]) == ("manual_attention", "parity_mismatch"):
            raise QuoteMemError("parity_mismatch", "A parity-failed owner is never skipped")
        if owner["status"] != "running" and not owner_stopped(owner):
            raise QuoteMemError("binding_mismatch", "A skip requires a running or stopped owner")
        reader = writer.indexed_reader()
        epochs = (writer.get_counter(f"erase_epoch:{scope.scope_id}"),
                  writer.get_counter(f"consent_epoch:{scope.scope_id}"))
        if writer.get_control(f"tomb:{scope.scope_id}") is not None or (
            reader.scope_fences(scope.scope_id) or reader.pending_privacy(scope.scope_id, limit=1)
        ) or (owner["admitted_erase_epoch"], owner["admitted_consent_epoch"]) != epochs:
            raise QuoteMemError("index_rebuilding", "Privacy settlement precedes a quarantine skip")
        if now < value.heartbeat_at:
            raise QuoteMemError("clock_rollback", "A quarantine skip clock regressed")
        budget = read_budget(writer, scope, owner_run_id)
        budget_value = None if budget is None else budget.value
        if budget is None or not isinstance(
            budget_value, FinalizeBudgetV2 | FinalizeBudgetRefusalV4,
        ) or (budget.version, budget_value.recovery_count) != (budget_version, recovery_count):
            raise QuoteMemError("operation_conflict", "The observed budget is no longer current")
        # The cycle's requests: its frozen input's receipts, or, before it froze,
        # every request ordered before its admission (``request_completion.budget_row``).
        cutoff = budget_value.accepted_prefix_commit_seq
        if owner["input_ref"] is not None:
            selected = writer.read_logical(owner["input_ref"], "run_input")
            if not isinstance(selected, FinalizeOwnerRunInputV1):
                raise QuoteMemError("binding_mismatch", "A quarantine skip lost its frozen input")
            cutoff = selected.receipt_high_watermark
        census = inspect_finalize_receipts(
            writer, scope, receipt_high_watermark=cutoff, check=lambda: None,
        )
        split = completion_requests(writer, scope, census.past, check=lambda: None)
        positions = owner_positions(writer, owner_run_id, split, admitted=(), refused=())
        completed = attribute_outcomes(
            positions, pending=split.pending, finished=split.finished, cause=split.cause,
        )
        rows: list[RowWrite] = [RowWrite("run", writer.parse_row("run", dict(
            owner, status="failed", failure_code="input_abandoned", updated_at=now,
            finished_at=now,
        )))]
        after: str | None = None
        while page := reader.run_children(owner_run_id, after=after):
            for child in page:
                if (child["scope_kind"], child["scope_id"]) != (scope.scope_kind, scope.scope_id):
                    raise QuoteMemError("binding_mismatch",
                                        "A quarantine skip found a foreign child")
                if child["status"] not in {"succeeded", "failed"}:
                    rows.append(RowWrite("run", writer.parse_row("run", dict(
                        child, status="failed", failure_code="input_abandoned",
                        updated_at=now, finished_at=now,
                    ))))
            after = page[-1]["object_id"]
        cancelled = CancelledFinalizeLeaseV1(
            state="cancelled", lease_generation=value.lease_generation,
            owner_run_id=owner_run_id, writer_epoch=value.writer_epoch,
            heartbeat_at=value.heartbeat_at, cancelled_from=value.state, cancelled_at=now,
            cancel_reason="repair_skip",
        )
        rows.append(RowWrite("control", writer.materialize_control(
            lease.key, cancelled.model_dump(mode="json"), lease.version + 1, now, owner_run_id,
        )))
        for request in completed:
            key = f"add_outcome:{scope.scope_id}:{request.request_key}"
            previous = writer.get_control(key)
            rows.append(RowWrite("control", writer.materialize_control(
                key, CompletedAddOutcomeV1(
                    state="skipped", request_key=request.request_key,
                    completed_by_run_id=owner_run_id, recovery_count=recovery_count,
                    outcome=request.outcome,
                ).model_dump(mode="json"),
                0 if previous is None else ControlRowV1.model_validate(previous).version + 1,
                now, owner_run_id,
            )))
        _, plan = writer.plan_rows(rows, None)
        writer.plan_deletes(plan, [("control", raw)])
        writer.write_plan(plan)
        _adopted_cycles(writer).pop(scope.scope_id, None)
        _paused_owners(writer).pop(owner_run_id, None)
        publisher = StoreReadModel.get(writer)
        if publisher is not None:
            publisher.publish_finalize_cancelled(
                writer, scope, owner_run_id, value.lease_generation, value.writer_epoch,
            )
        writer._finalize_tickets.pop(owner_run_id, None)  # pyright: ignore[reportPrivateUsage]
        # As a release does: the in-memory admission order no longer holds the skipped
        # owner as running, so the user's next Add admits its cycle in this lifetime.
        writer.add_admission.owner_released(scope.scope_id)
        dropped = drop_unsettled_extraction_records(writer, owner_run_id)
    return SkippedCycle(owner_run_id, len(completed), dropped)


def unsettled_extraction_records(writer: Writer, owner_run_id: str) -> frozenset[str]:
    """The stored extraction outputs of every unit of the owner that settled nothing.

    A unit with any candidate row keeps all its records: a published or pending
    position cites its proposal, and re-entry and History read them. Of every
    other unit, the extraction replies, the proposals and the statement
    embeddings that only those proposals use are selected.

    Args:
        writer: The sole Writer (read only).
        owner_run_id: The ended Add owner.

    Returns:
        The records' logical ids.

    Raises:
        QuoteMemError: ``binding_mismatch`` for a record of another shape.
    """
    from quotemem.core.commit_plan import CommitJournalV1
    from quotemem.core.extraction_context import ExtractContextV2
    from quotemem.core.inference import StoredInferenceOutput
    from quotemem.core.proposal import ProposalV2
    from quotemem.core.semantic_embedding import StatementEmbeddingV1

    settled: set[int] = set()
    for _reference, body in writer.find_logical_candidates(owner_run_id, "write_set"):
        if not isinstance(body, CommitJournalV1) or body.owner_run_id != owner_run_id:
            continue
        for written in body.written:
            if written.kind != "candidate":
                continue
            row = writer.get_row("candidate", written.object_id)
            if row is not None and row["owner_run_id"] == owner_run_id:
                settled.add(int(row["unit_no"]))
    units: dict[str, int] = {}
    for _reference, body in writer.find_logical_candidates(owner_run_id, "decision_set"):
        if isinstance(body, ExtractContextV2):
            units[body.input_digest] = body.unit.unit_no
    found: set[str] = set()
    for reference, body in writer.find_logical_candidates(owner_run_id, "llm_output"):
        if not isinstance(body, StoredInferenceOutput):
            raise QuoteMemError("binding_mismatch", "An Add owner stored another reply variant")
        unit = units.get(body.input_digest)
        if body.purpose == "extract" and unit is not None and unit not in settled:
            found.add(reference)
    kept: set[str] = set()
    for reference, body in writer.find_logical_candidates(owner_run_id, "proposal_snapshot"):
        if not isinstance(body, ProposalV2):
            raise QuoteMemError("binding_mismatch", "An Add owner stored another proposal variant")
        if body.position_key[1] in settled:
            kept |= {use.ref for use in (body.statement_embedding, body.cue_embedding)
                     if use is not None}
        else:
            found.add(reference)
    for reference, body in writer.find_logical_candidates(owner_run_id, "semantic_embedding"):
        if not isinstance(body, StatementEmbeddingV1):
            raise QuoteMemError("binding_mismatch", "An Add owner stored another embedding variant")
        if reference not in kept and not any(
            use[1] in settled for entry in body.entries for use in entry.uses
        ):
            found.add(reference)
    return frozenset(found)


def drop_unsettled_extraction_records(writer: Writer, owner_run_id: str) -> int:
    """Delete ``unsettled_extraction_records`` in bounded delete-only transactions.

    The caller holds the Writer's write scope, after the owner ended. A crash
    between transactions leaves only records no reader needs.

    Returns:
        How many rows were deleted.
    """
    from quotemem.store.finalize_privacy import delete_record_groups
    from quotemem.store.privacy_compensation import record_groups

    return delete_record_groups(writer, record_groups(
        writer, unsettled_extraction_records(writer, owner_run_id),
    ))

