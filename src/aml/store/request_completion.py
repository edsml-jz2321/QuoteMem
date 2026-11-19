"""An Add owner's completion policy, budget, completed prefix and failure cause (C.5, C.16).

The Writer's admission and release paths write these projections. The owner's
budget record holds the deadline every one of its sends is bounded by: the
original one, or a counted resume's fresh allowance (ingest-Q23, cross-Q12).
``owner_deadline`` reads it for I8's fresh extraction call (D14) and for every
judge send (C.5), and ``failure_cause`` maps the owner's durable stop to C.16's
content-free cause.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Final, Literal

from aml.core.add_outcome import AddCompletionRecordV1
from aml.core.controls import (
    AddCompletionV1,
    CompletedPrefixV2,
    ControlRowV1,
    FinalizeBudgetRefusalV4,
    FinalizeBudgetV2,
    HeldFinalizeLeaseV1,
    PreparingFinalizeLeaseV1,
    ReleasedFinalizeLeaseV1,
    RequestCompletionPolicyV2,
)
from aml.core.errors import AMLError
from aml.core.extraction_bounds import ExtractionRefusalDetailV1
from aml.core.identity import integrity_digest
from aml.core.query import ScopeV1
from aml.core.request_budget import (
    DEFAULT_COMPLETION_WORK_BUDGET_MS,
    MAX_OWNER_RESUMES,
    check_request_budget,
)
from aml.core.run_inputs import (
    FinalizeOwnerRunInputV1,
    SessionDurableRunInputV1,
    parse_run_input,
)
from aml.ingestion.authorization import check_add_grant
from aml.providers.auth import PrincipalGrant
from aml.store.domain import DOMAIN_ADAPTERS

if TYPE_CHECKING:
    from aml.ingestion.service import AddReceipt
    from aml.service.bench_workers import RetiredOwnerJoined
    from aml.store.finalize_ticket import FinalizeLeaseTicket
    from aml.store.read_model import StoreReadModel
    from aml.store.writer import RowWrite, Writer


EXTRACTION_RETRY_RESERVE_MS: Final = 512_000
"""Owner time a fresh extraction call must leave for the work that follows it (D14).

A contract-valid first reply that needs it (``needs_fresh_call``: no nonblank fact for
a unit with a declaration, at least 100 fresh words no assistant wrote
(``ZERO_PROPOSAL_RETRY_WORDS``), a fresh named possessive phrase (in English or a
closed phrase of another language, ``resend_lexicon``) or a short personal assertion, or a
participant written as the user) gets one fresh call, but only while the owner's
deadline still covers that call's first send (``attempt_wall_ms``) plus this
reserve. The reserve is the previous implementation's longest measured time from a successful
extraction to the Add's response (373 s and 512 s for its live dialogue Adds).
The admitted owner's allowance must still cover both the call and this reserve.
"""

REFUSED_OWNER_FAILURE_CODE: Final = "schema_invalid"
"""The stopped owner's Run failure code for every B.7.1 whole-output fault.

Each of the four faults is a reply that broke its frozen wire schema
(``ExtractionRefusalDetailV1``); the refused budget names the exact rule.
"""

type BudgetState = Literal["active", "exhausted", "refused"]
"""The state of an owner's ``finalize_budget`` record."""

type FailureCause = Literal["whole_output", "item", "provider", "budget", "integrity"]
"""The content-free cause of a non-200 Add that V3 inspection reports (C.16, E.2)."""


def _control(writer: Writer, key: str) -> ControlRowV1 | None:
    raw = writer.get_control(key)
    return None if raw is None else ControlRowV1.model_validate(raw)


def completion_policy_row(
    writer: Writer, owner: dict[str, Any], now: int, policy: RequestCompletionPolicyV2,
) -> RowWrite | None:
    """Install the admission's completion policy with the scope's first owner admission.

    Any admission can come before an Add completion enabled the scope: a
    Search or idle trigger, a restart that interrupted the first completion,
    or an owner admitted outside the bench controller. Installing the policy
    then gives that owner its allowance (``budget_row``) and lets the later
    completion find the scope already enabled. An installed policy is never
    replaced.
    """
    from aml.store.writer import RowWrite

    key = f"request_completion:{owner['scope_id']}"
    if _control(writer, key) is not None:
        return None
    return RowWrite("control", writer.materialize_control(
        key, policy.model_dump(mode="json"), 0, now,
    ))


def effective_receipt_deadline(witness: SessionDurableRunInputV1) -> int | None:
    """Return the public receipt's original reception allowance without renewing it.

    This coordinate is immutable receipt metadata, never the deadline of a new
    owner or a later HTTP attempt. Completion-serving receipts retain it
    explicitly; receipt-only SDK inputs derive it from their original reception.
    A witness without either coordinate supplies no reported allowance.
    """
    if witness.work_deadline_at is not None:
        return witness.work_deadline_at
    if witness.received_at is None:
        return None
    return witness.received_at + DEFAULT_COMPLETION_WORK_BUDGET_MS


def budget_row(
    writer: Writer, owner: dict[str, Any], now: int, *, policy: RequestCompletionPolicyV2,
) -> RowWrite | None:
    """Prepare the first allowance with owner/lease admission; never refresh a join.

    The frozen prefix bound is the admission-time bound: every request ordered
    before this admission (the current ``commit_seq``) belongs to this owner.
    The serving queue orders one fresh head before reserving its next owner;
    SDK receipt-only staging retains its existing already-ordered prefix.
    Ordering is fenced while the owner is preparing or held, so a request
    ordered later belongs to the next owner. Historical context remains within
    the existing frozen-prefix, release and recovery proofs.

    Queue residence and receipt age consume none of the owner's work allowance.
    Its first preparing admission starts ``admission + policy`` for every
    trigger. A later joiner or background trigger never replaces that allowance.

    ``policy`` is the admission's completion policy, which the same admission
    installs when the scope has none (``completion_policy_row``); an installed
    policy is used instead. Every owner therefore has an allowance: its
    retries end and its exhaustion is a settled failure.
    """
    from aml.store.writer import RowWrite

    installed = _control(writer, f"request_completion:{owner['scope_id']}")
    if installed is None:
        selected = policy
    elif isinstance(installed.value, RequestCompletionPolicyV2):
        selected = installed.value
    else:
        raise AMLError("binding_mismatch", "Request completion policy changed its contract")
    key = f"finalize_budget:{owner['scope_id']}"
    previous = _control(writer, key)
    if previous is not None and isinstance(
        previous.value, FinalizeBudgetV2 | FinalizeBudgetRefusalV4,
    ) and previous.value.owner_run_id == owner["object_id"]:
        return None
    deadline = now + selected.work_budget_ms
    value = FinalizeBudgetV2(
        owner_run_id=owner["object_id"], owner_attempt=owner["attempt"], state="active",
        accepted_prefix_commit_seq=writer.counter("commit_seq"),
        admitted_at=now, deadline_at=deadline,
        erase_epoch=owner["admitted_erase_epoch"], consent_epoch=owner["admitted_consent_epoch"],
    )
    return RowWrite("control", writer.materialize_control(
        key, value.model_dump(mode="json"), 0 if previous is None else previous.version + 1,
        now, owner["object_id"],
    ))


def enable_completion(
    writer: Writer, scope: ScopeV1, receipt: AddReceipt, *, grant: PrincipalGrant,
    clock: Callable[[], int], work_budget_ms: int | None = None,
) -> None:
    """Prove the receipt and preserve the scope's independent owner-work policy.

    A receipt's reception coordinate remains immutable identity metadata. It
    cannot shorten the allowance of an owner that has not started yet.
    """
    from aml.store.writer import RowWrite

    policy = (RequestCompletionPolicyV2() if work_budget_ms is None
              else RequestCompletionPolicyV2(work_budget_ms=work_budget_ms))
    with writer._write_scope():  # pyright: ignore[reportPrivateUsage]
        now = clock()
        owner = writer.get_row("run", receipt.run_id)
        if owner is None or (owner["scope_kind"], owner["scope_id"]) != (
            scope.scope_kind, scope.scope_id,
        ):
            raise AMLError("binding_mismatch", "Add completion lost its reception owner")
        check_add_grant(grant, scope, now, owner["principal_id"])
        selected = writer._bound_input(owner)  # pyright: ignore[reportPrivateUsage]
        witness = parse_run_input(selected)
        if not isinstance(witness, SessionDurableRunInputV1):
            raise AMLError("binding_mismatch", "Add completion requires a durable receipt")
        if (witness.visibility_commit_seq, effective_receipt_deadline(witness)) != (
            receipt.visibility_commit_seq, receipt.work_deadline_at,
        ):
            raise AMLError("binding_mismatch", "Add completion changed its accepted receipt")
        key = f"request_completion:{scope.scope_id}"
        previous = _control(writer, key)
        if previous is not None:
            if not isinstance(previous.value, RequestCompletionPolicyV2):
                raise AMLError("binding_mismatch", "Request completion policy changed its contract")
            if work_budget_ms is not None and previous.value != policy:
                raise AMLError("operation_conflict", "Scope completion policy is already frozen")
            return
        # No owner was admitted in this scope yet: every admission installs it.
        rows = [RowWrite("control", writer.materialize_control(
            key, policy.model_dump(mode="json"), 0, now, receipt.run_id,
        ))]
        _, plan = writer._prepare_rows(rows, None)  # pyright: ignore[reportPrivateUsage]
        writer._write_plan(plan)  # pyright: ignore[reportPrivateUsage]


def _recovery_count(writer: Writer, owner: dict[str, Any]) -> int:
    """Return the owner's counted-resume total; an owner without a budget has none."""
    budget = read_budget(
        writer, ScopeV1(scope_kind=owner["scope_kind"], scope_id=owner["scope_id"]),
        owner["object_id"],
    )
    if budget is None:
        return 0
    if not isinstance(budget.value, FinalizeBudgetV2):
        raise AMLError("binding_mismatch", "A refused owner cannot complete its requests")
    return budget.value.recovery_count


def _validated_completion(writer: Writer, owner: dict[str, Any]) -> AddCompletionRecordV1:
    """Confirm the immutable attached completion against its bookkeeping and owner.

    Every owner completes with a record (``store/add_outcome.record_add_completion``),
    an owner whose frozen input holds no Evidence and no pending request included.
    A counted resume after attachment may finish release using that first record;
    its count stays frozen with the outcome, never ahead of the current allowance.
    """
    selected = writer.read_logical(owner["input_ref"], "run_input")
    if not isinstance(selected, FinalizeOwnerRunInputV1):
        raise AMLError("binding_mismatch", "Completed owner lost its frozen input")
    row = _control(writer, f"add_completion:{owner['scope_id']}")
    if row is None or not isinstance(row.value, AddCompletionV1) or (
        row.value.owner_run_id, row.value.result_ref,
    ) != (owner["object_id"], owner["result_ref"]):
        raise AMLError("incomplete_coverage", "Release lacks the owner's completion bookkeeping")
    record = writer.read_logical(owner["result_ref"], "result_index")
    if not isinstance(record, AddCompletionRecordV1) or (
        record.owner_run_id, record.recovery_count, len(record.completed_requests),
    ) != (owner["object_id"], row.value.recovery_count, row.value.request_count):
        raise AMLError("binding_mismatch", "Completion bookkeeping lost its attached record")
    if record.recovery_count > _recovery_count(writer, owner):
        raise AMLError("binding_mismatch", "Completion record exceeds the owner's repair count")
    return record


def completed_row(
    writer: Writer, owner: dict[str, Any], lease: ControlRowV1, marker: ControlRowV1,
) -> RowWrite:
    """Prepare completion in the same final transaction as actual owner settlement.

    The completion record, request results and bookkeeping were written before
    release (``store/add_outcome.record_add_completion``); this only confirms
    their owner binding and writes the completed prefix. The owner's admission
    installed the scope's policy and its allowance.
    """
    from aml.store.writer import RowWrite

    if (owner["status"], owner["stage"]) != ("succeeded", "release") or not isinstance(
        lease.value, ReleasedFinalizeLeaseV1,
    ):
        raise AMLError("binding_mismatch", "Completed prefix lacks actual release and settlement")
    record = _validated_completion(writer, owner)
    key = f"completed_prefix:{owner['scope_id']}"
    previous = _control(writer, key)
    budget = read_budget(
        writer, ScopeV1(scope_kind=owner["scope_kind"], scope_id=owner["scope_id"]),
        owner["object_id"],
    )
    if budget is None or not isinstance(budget.value, FinalizeBudgetV2):
        raise AMLError("binding_mismatch", "Completed prefix lost its frozen admission cutoff")
    value = CompletedPrefixV2.model_validate({
        "accepted_prefix_commit_seq": budget.value.accepted_prefix_commit_seq,
        "lease": lease.value.model_dump(mode="json"),
        "marker": marker.value.model_dump(mode="json"),
        "owner_digest": integrity_digest(owner), "input_ref": owner["input_ref"],
        "result_ref": owner["result_ref"],
        "input_digest": integrity_digest(writer.read_logical(owner["input_ref"], "run_input")),
        "result_digest": integrity_digest(record),
        "erase_epoch": owner["admitted_erase_epoch"],
        "consent_epoch": owner["admitted_consent_epoch"],
    })
    return RowWrite("control", writer.materialize_control(
        key, value.model_dump(mode="json"), 0 if previous is None else previous.version + 1,
        owner["finished_at"], owner["object_id"],
    ))


def completed_prefix(
    writer: Writer, publisher: StoreReadModel, scope: ScopeV1, receipt: AddReceipt, *,
    grant: PrincipalGrant, clock: Callable[[], int],
) -> CompletedPrefixV2 | None:
    """Prove a retained completed prefix without chasing later input or current leases."""
    from aml.store.read_model import StoreReadModel

    with writer.mutex:
        writer.validate_storage_owner()
        if StoreReadModel.for_writer(writer) is not publisher:
            raise AMLError("binding_mismatch", "Completed prefix requires its actual publisher")
        check_add_grant(grant, scope, clock())
        if writer.scope_quarantined(scope.scope_id):
            raise AMLError("scope_quarantined", "Completed Add scope requires explicit repair")
        row = _control(writer, f"completed_prefix:{scope.scope_id}")
        if row is None:
            return None
        value = row.value
        if not isinstance(value, CompletedPrefixV2):
            raise AMLError("binding_mismatch", "Completed prefix changed its contract")
        if (value.erase_epoch, value.consent_epoch) != (
            writer.counter(f"erase_epoch:{scope.scope_id}"),
            writer.counter(f"consent_epoch:{scope.scope_id}"),
        ):
            return None
        if value.accepted_prefix_commit_seq < receipt.visibility_commit_seq:
            return None
        owner = writer.get_row("run", value.marker.run_id)
        if owner is None or integrity_digest(owner) != value.owner_digest:
            raise AMLError("binding_mismatch", "Completed prefix lost immutable owner authority")
        DOMAIN_ADAPTERS["run"].validate_python(owner)
        if (owner["scope_kind"], owner["scope_id"], owner["status"], owner["stage"],
            owner["input_ref"], owner["result_ref"], row.updated_by_run_id) != (
            scope.scope_kind, scope.scope_id, "succeeded", "release",
            value.input_ref, value.result_ref, value.marker.run_id,
        ):
            raise AMLError("binding_mismatch", "Completed prefix changed its scope or outcome")
        check_add_grant(grant, scope, clock(), owner["principal_id"])
        selected = writer.read_logical(value.input_ref, "run_input")
        result = writer.read_logical(value.result_ref, "result_index")
        if not isinstance(selected, FinalizeOwnerRunInputV1) or (
            integrity_digest(selected), integrity_digest(result), selected.business_key.scope_id,
            selected.input_heads_digest,
        ) != (
            value.input_digest, value.result_digest, scope.scope_id,
            value.marker.input_heads_digest,
        ):
            raise AMLError("binding_mismatch", "Completed prefix lost its selected work proof")
        if publisher.current.generation < value.lease.baseline_snapshot_generation:
            raise AMLError("service_unhealthy", "Completed prefix has no published read root")
        check_add_grant(grant, scope, clock(), owner["principal_id"])
        return value


def read_budget(writer: Writer, scope: ScopeV1, owner_run_id: str) -> ControlRowV1 | None:
    """Read an exact current owner budget; waiter joins never change it."""
    with writer.mutex:
        row = _control(writer, f"finalize_budget:{scope.scope_id}")
        if row is None or not isinstance(row.value, FinalizeBudgetV2 | FinalizeBudgetRefusalV4) or (
            row.value.owner_run_id != owner_run_id
        ):
            if _control(writer, f"request_completion:{scope.scope_id}") is not None:
                owner = writer.get_row("run", owner_run_id)
                if owner is not None and owner["status"] in {
                    "queued", "running", "manual_attention",
                }:
                    raise AMLError("binding_mismatch", "Active V2 work lost its admitted budget")
            return None
        value = row.value
        owner = writer.get_row("run", owner_run_id)
        if owner is None or (owner["scope_kind"], owner["scope_id"]) != (
            scope.scope_kind, scope.scope_id,
        ) or value.owner_attempt > owner["attempt"]:
            raise AMLError("binding_mismatch", "Finalize budget lost its original owner")
        if (value.erase_epoch, value.consent_epoch) != (
            owner["admitted_erase_epoch"], owner["admitted_consent_epoch"],
        ):
            raise AMLError("binding_mismatch", "Finalize budget changed admitted privacy epochs")
        if isinstance(value, FinalizeBudgetRefusalV4) and (
            owner["status"], owner["failure_code"], owner["attempt"],
        ) != ("manual_attention", REFUSED_OWNER_FAILURE_CODE, value.owner_attempt):
            raise AMLError("binding_mismatch", "Refused budget lost its stopped owner")
        return row


def require_owner_budget(writer: Writer, owner_run_id: str, *, now: int) -> None:
    """Refuse semantic commits after the exact execution owner's allowance ends.

    Call inside the Writer transaction that will publish a semantic change. The
    cooperative request check also rejects callbacks from a retired execution
    even when an operator has explicitly funded a later repair of the same owner.

    Args:
        writer: Workspace persistence owner holding its ordinary write exclusion.
        owner_run_id: Actual finalize owner whose work produced this operation.
        now: Current wall clock from the caller's trusted action clock.

    Raises:
        AMLError: If the owner is missing, refused, exhausted or past its deadline.
    """
    check_request_budget()
    owner = writer.get_row("run", owner_run_id)
    if owner is None or owner["run_kind"] != "finalize":
        raise AMLError("binding_mismatch", "Semantic commit lost its finalize budget owner")
    row = read_budget(
        writer, ScopeV1(scope_kind=owner["scope_kind"], scope_id=owner["scope_id"]), owner_run_id,
    )
    if row is None:
        return
    if not isinstance(row.value, FinalizeBudgetV2) or row.value.state != "active" or (
        now >= row.value.deadline_at
    ):
        raise AMLError("delivery_busy", "Owner work budget no longer permits semantic commits")


def owner_deadline(writer: Writer, owner_run_id: str) -> int:
    """Return the deadline the owner's current budget record holds (C.5; cross-Q12).

    That is the original deadline, or the fresh allowance of the owner's latest
    counted resume (ingest-Q23). Every send of the owner draws on it: a judge
    send's reservation (S4b) and I8's fresh extraction call (D14). No ordinary
    join extends it. C41 may invoke the counted recovery proofs 32 times
    automatically; one final allowance remains available for explicit recovery.

    Args:
        writer: The sole Writer; the read takes its mutex.
        owner_run_id: The Add owner whose sends are bounded.

    Returns:
        The deadline as a Writer timestamp in milliseconds.

    Raises:
        AMLError: ``binding_mismatch`` when the owner is no Add owner or has no
            admitted budget (F.7 E05); ``delivery_busy`` when its allowance is
            refused or exhausted.
    """
    owner = writer.get_row("run", owner_run_id)
    if owner is None or owner["run_kind"] != "finalize":
        raise AMLError("binding_mismatch", "Owner deadline lost its Add owner")
    row = read_budget(
        writer, ScopeV1(scope_kind=owner["scope_kind"], scope_id=owner["scope_id"]), owner_run_id,
    )
    if row is None:
        raise AMLError("binding_mismatch", "The Add owner has no admitted budget")
    if not isinstance(row.value, FinalizeBudgetV2) or row.value.state != "active":
        raise AMLError("delivery_busy", "The Add owner's budget no longer permits sends")
    return row.value.deadline_at


def fresh_call_fits(deadline_at: int, now: int, *, attempt_wall_ms: int) -> bool:
    """Whether one fresh extraction call fits before the owner's deadline with the reserve.

    D14: the call's first send (``attempt_wall_ms``) plus
    ``EXTRACTION_RETRY_RESERVE_MS`` must end no later than the deadline the
    owner's budget record holds (``owner_deadline``); otherwise the first reply
    stands.

    Args:
        deadline_at: The owner's current deadline, in milliseconds.
        now: The trusted action clock, in milliseconds.
        attempt_wall_ms: The extraction call budget's per-request wall time.

    Returns:
        True when the fresh call and the reserve both fit.

    Raises:
        ValueError: If ``attempt_wall_ms`` is not positive.
    """
    if attempt_wall_ms <= 0:
        raise ValueError("A fresh extraction call needs a positive attempt budget")
    return now + attempt_wall_ms + EXTRACTION_RETRY_RESERVE_MS <= deadline_at


def failure_cause(
    *, quarantined: bool, budget_state: BudgetState | None, provider_paused: bool,
) -> FailureCause | None:
    """Map an owner's durable stop record to C.16's content-free failure cause.

    A quarantined scope (``scope_quarantine:{s}``) is ``integrity`` whatever
    else holds. Otherwise a refused budget (B.7.1's whole-output faults) is
    ``whole_output``, a provider pause (the owner's ``result_index:provider_failure``)
    ``provider`` and an exhausted budget ``budget``. A pause recorded first keeps
    its cause when the paused owner's allowance later runs out. ``item`` is never
    an owner stop: item faults are C.10 dispositions (INV:D16, E.4).

    Args:
        quarantined: Whether the scope holds a quarantine receipt.
        budget_state: The state of the owner's budget record, or None without one.
        provider_paused: Whether the owner is paused on a provider failure.

    Returns:
        The cause, or None while nothing stopped the owner.
    """
    if quarantined:
        return "integrity"
    if budget_state == "refused":
        return "whole_output"
    if provider_paused:
        return "provider"
    if budget_state == "exhausted":
        return "budget"
    return None


def owner_failure_cause(writer: Writer, scope: ScopeV1, owner_run_id: str) -> FailureCause | None:
    """Read the owner's durable stop record and map it with ``failure_cause``.

    The caller reports the result (V3 inspection, E.2); no second copy is stored.
    """
    with writer.mutex:
        row = read_budget(writer, scope, owner_run_id)
        owner = writer.get_row("run", owner_run_id)
        if owner is None or (owner["scope_kind"], owner["scope_id"]) != (
            scope.scope_kind, scope.scope_id,
        ):
            raise AMLError("binding_mismatch", "Failure cause lost its scoped owner")
        return failure_cause(
            quarantined=writer.scope_quarantined(scope.scope_id),
            budget_state=None if row is None else row.value.model_dump(mode="json")["state"],
            provider_paused=(owner["status"], owner["failure_code"]) == (
                "manual_attention", "provider_error",
            ),
        )


def _recovery_allowance(writer: Writer, scope: ScopeV1, work_budget_ms: int | None) -> int:
    """Select one explicit recovery's allowance without changing the frozen policy.

    An operator can select a positive allowance up to 1,740 seconds for this
    recovery only. Omission retains the original scope allowance.
    """
    policy = _control(writer, f"request_completion:{scope.scope_id}")
    assert policy is not None and isinstance(policy.value, RequestCompletionPolicyV2)
    return policy.value.work_budget_ms if work_budget_ms is None else (
        RequestCompletionPolicyV2(work_budget_ms=work_budget_ms).work_budget_ms
    )


def resume_refused_budget_row(
    writer: Writer, owner: dict[str, Any], observed: ControlRowV1, *, now: int,
    work_budget_ms: int | None = None,
) -> RowWrite:
    """Replace an observed refusal with a fresh counted allowance for its owner's resume.

    Explicit completion recovery resumes a refused owner as ``resume_budget``
    resumes an exhausted one: the allowance follows ``work_budget_ms`` or the
    installed scope policy, the shared ``recovery_count`` admits at most 33
    automatic/explicit resumes, and the original deadline may already have passed.
    The Writer has proved the owner's original failure history, the actual callback
    join and authorization, and commits this row in the transaction that moves
    the owner to its next attempt.

    Args:
        writer: The sole Writer, inside the owner's resume transaction.
        owner: The refused owner's current Run, before its next attempt.
        observed: The refusal Control the caller observed and now resumes.
        now: The transaction's trusted clock.
        work_budget_ms: Optional allowance for this recovery only.

    Returns:
        The active allowance that replaces the refusal.

    Raises:
        AMLError: ``operation_conflict`` when the observed refusal is no longer
            the owner's current budget, has used its counted resume cap, or the
            scope's privacy authority changed since its admission.
    """
    from aml.store.writer import RowWrite

    scope = ScopeV1(scope_kind=owner["scope_kind"], scope_id=owner["scope_id"])
    value = observed.value
    if read_budget(writer, scope, owner["object_id"]) != observed or not isinstance(
        value, FinalizeBudgetRefusalV4,
    ) or value.recovery_count >= MAX_OWNER_RESUMES:
        raise AMLError("operation_conflict", "The observed refusal cannot resume")
    if (value.erase_epoch, value.consent_epoch) != (
        writer.counter(f"erase_epoch:{scope.scope_id}"),
        writer.counter(f"consent_epoch:{scope.scope_id}"),
    ):
        raise AMLError("operation_conflict", "Refusal recovery changed privacy authority")
    active = FinalizeBudgetV2.model_validate({
        **value.model_dump(mode="json", include=set(FinalizeBudgetV2.model_fields)),
        "contract_version": 2, "state": "active", "owner_attempt": owner["attempt"] + 1,
        "recovery_count": value.recovery_count + 1, "admitted_at": now,
        "deadline_at": now + _recovery_allowance(writer, scope, work_budget_ms),
    })
    return RowWrite("control", writer.materialize_control(
        observed.key, active.model_dump(mode="json"), observed.version + 1, now,
        owner["object_id"],
    ))


def exhaust_budget(
    writer: Writer, scope: ScopeV1, owner_run_id: str, *, clock: Callable[[], int],
    early: bool = False,
) -> None:
    """Persist non-success after this owner allowance, preserving all child effects.

    Args:
        writer: The sole Writer.
        scope: The owner's scope.
        owner_run_id: The owner whose allowance ends.
        clock: Trusted millisecond action clock.
        early: End a still-live allowance the owner has proved it cannot use: an
            extraction call whose own window ended without a reply (RESUMENOW).

    Raises:
        AMLError: ``binding_mismatch`` without the selected budget;
            ``operation_conflict`` for a live allowance unless ``early``.
    """
    from aml.store.writer import RowWrite

    with writer._write_scope():  # pyright: ignore[reportPrivateUsage]
        row = read_budget(writer, scope, owner_run_id)
        if row is None or not isinstance(row.value, FinalizeBudgetV2 | FinalizeBudgetRefusalV4):
            raise AMLError("binding_mismatch", "Exhaustion requires the selected owner budget")
        if isinstance(row.value, FinalizeBudgetRefusalV4) or row.value.state == "exhausted":
            return
        now = clock()
        if now < row.value.deadline_at and not early:
            raise AMLError("operation_conflict", "The shared owner still has execution budget")
        owner = writer.get_row("run", owner_run_id)
        assert owner is not None
        if owner["status"] == "succeeded":
            return
        exhausted = FinalizeBudgetV2.model_validate({
            **row.value.model_dump(mode="json", include=set(FinalizeBudgetV2.model_fields)),
            "contract_version": 2, "state": "exhausted",
        })
        updated = writer.materialize_control(
            row.key, exhausted.model_dump(mode="json"), row.version + 1, now, owner_run_id,
        )
        _, plan = writer._prepare_rows(  # pyright: ignore[reportPrivateUsage]
            [RowWrite("control", updated)], None,
        )
        writer._write_plan(plan)  # pyright: ignore[reportPrivateUsage]


def refuse_budget(
    writer: Writer, ticket: FinalizeLeaseTicket, *, unit_no: int,
    refusal: ExtractionRefusalDetailV1, clock: Callable[[], int],
) -> ControlRowV1:
    """Stop the held owner on a unit's whole-output fault (B.7.1, C.2 step 1).

    Under one Writer scope the owner's active allowance becomes its refused
    record (``FinalizeBudgetRefusalV4``: the allowance's fields, the refused
    ``unit_no`` and its content-free detail) and the owner Run moves to
    ``manual_attention`` with ``REFUSED_OWNER_FAILURE_CODE``. The lease stays
    held and nothing in the prefix commits; the refused owner's requests get 422
    ``extraction_refused`` and its successors 409 ``repair_required`` until an
    explicit V3 resume (``resume_refused_budget_row``). The stopped owner's
    cycle no longer passes the Writer's fence, so it cannot refuse again.

    Args:
        writer: The sole Writer.
        ticket: The held owner's issued cycle.
        unit_no: The refused unit within the owner.
        refusal: That unit's whole-output fault.
        clock: Trusted millisecond action clock.

    Returns:
        The refused budget Control.

    Raises:
        AMLError: ``operation_conflict`` when the owner's allowance already ended
            (exhausted or past its deadline) or the owner is not running;
            ``binding_mismatch`` without the owner's budget or with a refused one;
            ``lease_lost`` for a lost or stopped cycle.
    """
    from aml.store.writer import RowWrite

    with writer._write_scope():  # pyright: ignore[reportPrivateUsage]
        run, _, _ = writer._finalize_precondition(ticket)  # pyright: ignore[reportPrivateUsage]
        scope = ScopeV1(scope_kind=run["scope_kind"], scope_id=run["scope_id"])
        row = read_budget(writer, scope, ticket.owner_run_id)
        if row is None:
            raise AMLError("binding_mismatch", "Refusal requires the owner's budget")
        value = row.value
        if not isinstance(value, FinalizeBudgetV2):
            raise AMLError("binding_mismatch", "Owner budget changed its contract")
        now = clock()
        if value.state != "active" or now >= value.deadline_at or run["status"] != "running":
            raise AMLError("operation_conflict", "Refusal requires the owner's live allowance")
        refused = FinalizeBudgetRefusalV4.model_validate({
            **value.model_dump(mode="json", exclude={"contract_version", "state"}),
            "unit_no": unit_no, "refusal": refusal.model_dump(mode="json"),
        })
        name = f"finalize_budget:{scope.scope_id}"
        updated = writer.materialize_control(
            name, refused.model_dump(mode="json"), row.version + 1, now, ticket.owner_run_id,
        )
        run.update(
            status="manual_attention", failure_code=REFUSED_OWNER_FAILURE_CODE,
            updated_at=now, finished_at=None,
        )
        stopped = writer._parse_row("run", run)  # pyright: ignore[reportPrivateUsage]
        rows = [RowWrite("control", updated), RowWrite("run", stopped)]
        _, plan = writer._prepare_rows(rows, None)  # pyright: ignore[reportPrivateUsage]
        writer._write_plan(plan)  # pyright: ignore[reportPrivateUsage]
        return ControlRowV1.model_validate(updated.model_dump(mode="json"))


def resume_budget(
    writer: Writer, scope: ScopeV1, observed: ControlRowV1, *, joined: RetiredOwnerJoined,
    grant: PrincipalGrant, clock: Callable[[], int], work_budget_ms: int | None = None,
) -> None:
    """Grant an exhausted owner up to the shared counted-resume cap.

    The allowance follows ``work_budget_ms`` or the installed scope policy
    (``_recovery_allowance``); a refused owner's resume counts against the same
    cap. The deadline remains persisted on the same owner; retries and
    restarts never extend it. A joined resume advances a frozen owner's attempt
    so unanswered calls can renew their expired windows; reserved sends and
    accepted replies remain authoritative. An owner still freezing its input
    remains on attempt zero and has no execution calls to renew.
    """
    from aml.store.extraction_budget import resumed_outage_allowances
    from aml.store.writer import RowWrite

    with writer._write_scope():  # pyright: ignore[reportPrivateUsage]
        current = _control(writer, observed.key)
        value = observed.value
        if current != observed or not isinstance(value, FinalizeBudgetV2) or (
            value.state != "exhausted" or value.recovery_count >= MAX_OWNER_RESUMES
        ):
            raise AMLError("operation_conflict", "The observed owner budget cannot resume")
        pool = writer._finalize_pool  # pyright: ignore[reportPrivateUsage]
        if pool is None:
            raise AMLError("binding_mismatch", "Budget recovery requires the actual worker pool")
        pool.validate_retired_owner_join(joined)
        if (joined.work.scope_id, joined.work.run_id, joined.work.kind) != (
            scope.scope_id, value.owner_run_id, "finalize",
        ):
            raise AMLError("binding_mismatch", "Budget recovery selected another joined owner")
        owner = writer.get_row("run", value.owner_run_id)
        if owner is None or (owner["status"], owner["failure_code"]) != ("running", None) or (
            writer.scope_quarantined(scope.scope_id)
        ):
            raise AMLError("operation_conflict", "Budget recovery requires its running owner")
        now = clock()
        check_add_grant(grant, scope, now, owner["principal_id"])
        if (value.erase_epoch, value.consent_epoch) != (
            writer.counter(f"erase_epoch:{scope.scope_id}"),
            writer.counter(f"consent_epoch:{scope.scope_id}"),
        ):
            raise AMLError("operation_conflict", "Budget recovery changed privacy authority")
        allowance = _recovery_allowance(writer, scope, work_budget_ms)
        lease = _control(writer, f"lease:finalize:{scope.scope_id}")
        if lease is None or not isinstance(
            lease.value, PreparingFinalizeLeaseV1 | HeldFinalizeLeaseV1,
        ) or (
            lease.value.owner_run_id, lease.value.writer_epoch,
        ) != (value.owner_run_id, owner["writer_epoch"]):
            raise AMLError("operation_conflict", "Budget recovery lost its exact current lease")
        if now < lease.value.heartbeat_at:
            raise AMLError("clock_rollback", "Budget recovery clock regressed")
        frozen = owner["input_ref"] is not None
        if not frozen and owner["attempt"] != 0:
            raise AMLError("binding_mismatch", "An unfrozen owner cannot have resumed execution")
        attempt = owner["attempt"] + int(frozen)
        renewed = writer.materialize_control(lease.key, {
            **lease.value.model_dump(mode="json"), "heartbeat_at": now,
        }, lease.version + 1, now, value.owner_run_id)
        updated = writer.materialize_control(observed.key, {
            **value.model_dump(mode="json"), "state": "active", "admitted_at": now,
            "deadline_at": now + allowance,
            "owner_attempt": attempt, "recovery_count": value.recovery_count + 1,
        }, observed.version + 1, now, value.owner_run_id)
        rows = [RowWrite("control", updated), RowWrite("control", renewed)]
        if frozen:
            calls = resumed_outage_allowances(
                writer, owner, now=now, deadline_at=now + allowance,
            )
            resumed = writer._parse_row("run", dict(  # pyright: ignore[reportPrivateUsage]
                owner, attempt=attempt, updated_at=now,
            ))
            rows.extend([RowWrite("run", resumed), *calls])
        _, plan = writer._prepare_rows(  # pyright: ignore[reportPrivateUsage]
            rows, None,
        )
        writer._write_plan(plan)  # pyright: ignore[reportPrivateUsage]

