"""The Add owner's lease: admit, drain, hold, heartbeat, freeze, release, pause.

Moved out of ``store/writer.py``. Each function takes the sole Writer and runs under
its mutex and write scope exactly as the Writer method did;
the Writer keeps a thin delegation for each function moved. The owner selects its
frozen input here and releases here after its completion
(``store/add_outcome.record_add_completion``). The scheduler binds its worker pool here and pauses
and resumes an expired owner here; integrity failures quarantine the
scope here. Admission after a released cycle proves its history (``released_history``); startup
recovery classifies each owner with ``owner_recovery.classify_cold_owner``, settles a
proved released or cancelled cycle, finishes a release that died before its last transaction
and adopts an interrupted preparing or held one under the new Writer epoch. A cycle
a forget cancelled is followed by the next one once ``cancelled_settled`` proves
it. An explicit repair lifts a quarantine around its owner's explicit resume
(``repair_quarantine``).
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal, cast

from quotemem.core.controls import (
    CancelledFinalizeLeaseV1,
    ControlRowV1,
    FinalizedValueV1,
    HeldFinalizeLeaseV1,
    PreparingFinalizeLeaseV1,
    RequestCompletionPolicyV2,
)
from quotemem.core.errors import QuoteMemError
from quotemem.core.failure_reason import FailureReason, failure_reason
from quotemem.core.identity import canonical_json, new_ulid
from quotemem.ingestion.owner_recovery import ColdOwner, classify_cold_owner
from quotemem.store.finalize_ticket import FinalizeLeaseTicket
from quotemem.store.writer import RowWrite, Writer, WriteTicket

if TYPE_CHECKING:
    from quotemem.core.identity import ArtifactId
    from quotemem.core.query import ScopeV1
    from quotemem.ingestion.finalize_freeze import FinalizeInputProof
    from quotemem.providers.auth import PrincipalGrant
    from quotemem.service.bench_workers import BenchWorkPause, BenchWorkPool
    from quotemem.service.resource_observer import ResourceObserver
    from quotemem.store.read_model import StoreReadModel

OWNER_LEASE_LIFETIME_MS = 60_000
"""The heartbeat age at which the Writer refuses an owner's lease as expired."""

OWNER_LEASE_RENEWAL_AGE_MS = OWNER_LEASE_LIFETIME_MS // 3
"""The heartbeat age from which a live lease is renewed."""


@dataclass(frozen=True, slots=True)
class _PausedOwner:
    """An expired owner reserved while its exact cancelled invocation drains.

    The owner and lease images are canonical JSON taken under the Writer mutex
    at the pause; the resume renews only an unchanged pair.
    """

    ticket: FinalizeLeaseTicket
    work: BenchWorkPause
    owner_json: str
    lease_json: str


def _paused_owners(writer: Writer) -> dict[str, _PausedOwner]:
    """Return this Writer's process-local paused owners, keyed by owner Run."""
    paused: dict[str, _PausedOwner] = writer.__dict__.setdefault("_paused_owners", {})
    return paused


type CycleIdentity = tuple[str, int, int]
"""One cycle's ``(owner_run_id, lease_generation, writer_epoch)``."""


def _adopted_cycles(writer: Writer) -> dict[str, CycleIdentity]:
    """Return the held cycles this Writer's boot adopted, keyed by scope."""
    adopted: dict[str, CycleIdentity] = writer.__dict__.setdefault("_adopted_cycles", {})
    return adopted


def adopted_cycle(writer: Writer, scope_id: str) -> CycleIdentity | None:
    """The held cycle boot step 14 adopted in a scope, until its owner holds again.

    A new process retains no baseline of an adopted held cycle. The read model's
    cold build keeps the lease's barrier closed for it, and the owner's next
    ``hold_finalize`` captures a fresh baseline.

    Args:
        writer: The sole Writer.
        scope_id: The scope.

    Returns:
        The adopted cycle, or None.
    """
    return _adopted_cycles(writer).get(scope_id)


def owner_stopped(owner: Mapping[str, Any]) -> bool:
    """Whether an owner Run is durably stopped for explicit recovery.

    A provider pause (``owner_pause``) and a refused budget
    (``request_completion.refuse_budget``) keep the owner, its lease and its
    budget until an explicit resume; adoption keeps such an owner stopped.
    """
    from quotemem.store.owner_pause import PROVIDER_PAUSE_FAILURE_CODE
    from quotemem.store.request_completion import REFUSED_OWNER_FAILURE_CODE

    return owner["status"] == "manual_attention" and owner["failure_code"] in {
        PROVIDER_PAUSE_FAILURE_CODE, REFUSED_OWNER_FAILURE_CODE,
    }


def finalize_ticket(
    writer: Writer, owner_run_id: str, *,
    clock: Callable[[], int] = lambda: time.time_ns() // 1_000_000,
) -> FinalizeLeaseTicket:
    """Bind one existing current-epoch lease to a live process identity.

    This performs no admission, epoch adoption, storage write or scheduling.
    A live ticket is reused by identity. Once every consumer releases it,
    the weak registration disappears; later discovery rechecks the actual
    Run, lease, canonical admission and ordinary Writer/privacy fences.
    """
    with writer.mutex:
        issued = writer._finalize_tickets.get(owner_run_id)
        created = issued is None
        if issued is None:
            run = writer.get_row("run", owner_run_id)
            if run is None:
                raise QuoteMemError("reference_missing", "Finalize lost its existing owner")
            writer.parse_row("run", run)
            raw = writer.get_control(f"lease:finalize:{run['scope_id']}")
            if raw is None:
                raise QuoteMemError("lease_lost", "Finalize lost its existing lease")
            value = ControlRowV1.model_validate(raw).value
            if not isinstance(value, PreparingFinalizeLeaseV1 | HeldFinalizeLeaseV1) or (
                value.owner_run_id != owner_run_id
            ):
                raise QuoteMemError("lease_lost", "Finalize owner no longer holds its cycle")
            issued = FinalizeLeaseTicket(
                run["scope_id"], value.lease_generation, owner_run_id,
                value.writer_epoch, value.state,
            )
            writer._register_finalize_ticket(issued, clock)
        try:
            writer.read_finalize_fences(issued)
        except Exception:
            if created:
                writer._finalize_tickets.pop(owner_run_id, None)
            raise
        return issued


def admit_finalize(
    writer: Writer, *, scope: ScopeV1, grant: PrincipalGrant, clock: Callable[[], int],
    resources: ResourceObserver | None = None,
    admission_trigger: Literal["first_search", "idle", "add_completion"] | None = None,
    completion_policy: RequestCompletionPolicyV2 | None = None,
) -> FinalizeLeaseTicket:
    """Admit or reuse one exact cycle and atomically acquire preparing.

    Args:
        scope: Exact registered scope selected by the service.
        grant: Fresh configured AuthProvider grant, resolved outside locks.
        clock: Trusted millisecond action clock.
        resources: Optional bounded, content-free resource observer. Observer
            failure never changes the application's admission result.
        admission_trigger: Actual controller path for a new owner; reuse or an
            earlier queued owner never acquires a new label.
        completion_policy: The scope's completion policy (the runtime
            controller passes its configured one; None selects the
            standard policy). This admission installs it when no
            completion or earlier admission has, so every owner it
            admits or joins has its own allowance.

    Returns:
        The same process-issued ticket for repeated calls in this cycle.

    Raises:
        QuoteMemError: For stale authorization, ownership or an unsettled cycle;
            ``binding_mismatch`` / ``finalize_pending`` for a released
            predecessor that ``released_history`` cannot prove; and
            ``erasure_in_progress`` / ``binding_mismatch`` for a cancelled
            predecessor ``cancelled_settled`` cannot prove yet.
    """
    from quotemem.core.admission import admission_identity, finalize_admission
    from quotemem.ingestion.authorization import check_add_grant
    from quotemem.service.resource_observer import ResourceCycle
    from quotemem.store.domain import SessionTargetRunRowV1
    from quotemem.store.indexed import IndexedReader
    from quotemem.store.released_history import prove_released_history
    from quotemem.store.request_completion import budget_row, completion_policy_row

    def observe(owner: str, cycle: int, *, new: bool, later: bool, dirty: bool) -> None:
        if resources is None:
            return
        identity = ResourceCycle(scope.scope_id, owner, cycle)
        try:
            if new and admission_trigger is not None:
                resources.cycle_admitted(
                    identity, trigger="late_add" if later and dirty else admission_trigger,
                    at_ns=time.monotonic_ns(),
                )
            else:
                resources.ensure_cycle(identity)
        except Exception:
            # No user callback, HB read, provider or application write is
            # delegated here. Measurement loss stays measurement loss.
            try:
                resources.incomplete("cycle")
            except Exception:
                pass

    policy = RequestCompletionPolicyV2() if completion_policy is None else completion_policy
    check_add_grant(grant, scope, clock())
    with writer.mutex:
        if writer.is_scope_quarantined(scope.scope_id):
            raise QuoteMemError("scope_quarantined", "The scope requires explicit recovery repair")
    with writer.write_scope():
        now = clock()
        check_add_grant(grant, scope, now)
        reader = IndexedReader(
            writer._ws,
            writer.entities,
            tenant_id=writer.tenant_id,
        )
        raw = writer.get_control(f"lease:finalize:{scope.scope_id}")
        lease = None if raw is None else ControlRowV1.model_validate(raw)
        if lease is not None and lease.value.model_dump(mode="json")["state"] in {
            "preparing", "held",
        }:
            owner = lease.value.model_dump(mode="json")["owner_run_id"]
            issued = writer.finalize_ticket(owner, clock=clock)
            # finalize_ticket just validated the full owner/lease/privacy
            # fences inside this unchanged write scope. Only the principal
            # is needed for the request's separate authorization check.
            run = writer.get_row("run", owner)
            if run is None:
                raise QuoteMemError("reference_missing", "Finalize lost its existing owner")
            check_add_grant(grant, scope, now, run["principal_id"])
            allowance = [row for row in (
                completion_policy_row(writer, run, now, policy),
                budget_row(writer, run, now, policy=policy),
            ) if row is not None]
            if allowance:
                _, budget_plan = writer.plan_rows(allowance, None)
                writer.write_plan(budget_plan)
            observe(owner, issued.lease_generation, new=False, later=False, dirty=False)
            return issued
        writer.add_admission.require_finalize_allowed(scope.scope_id)
        if lease is not None and isinstance(lease.value, CancelledFinalizeLeaseV1):
            # A forget cancelled the last cycle: the next owner reprocesses
            # every request it left incomplete.
            cancelled_settled(writer, scope, lease)
            dirty_predecessor = True
            cycle = lease.value.lease_generation + 1
        elif lease is not None:
            value = lease.value.model_dump(mode="json")
            if value["state"] != "released":
                raise QuoteMemError("unsupported_variant", "Finalize cycle is not settled")
            prior = writer.get_row("run", value["owner_run_id"])
            if prior is None or prior["status"] != "succeeded":
                raise QuoteMemError("service_unhealthy", "Released finalize requires settlement")
            # The released cycle is proved from its own records before
            # the next cycle may follow it.
            finalized = prove_released_history(writer, scope, lease)
            dirty_predecessor = finalized.finalized_scope_seq < writer.get_counter(
                f"scope_seq:{scope.scope_id}",
            )
            cycle = value["lease_generation"] + 1
        else:
            dirty_predecessor = False
            cycle = 1
        admission = finalize_admission(
            principal_id=grant.principal_id, scope=scope, cycle=cycle,
        )
        identity, digest, _ = admission_identity(admission, writer.bind_content)
        previous = reader.latest_run("finalize", identity)
        new_owner = previous is None
        if previous is None:
            writer_lease = writer.get_control("lease:writer")
            if writer_lease is None:
                raise QuoteMemError("service_unhealthy", "Finalize lost the Writer lease")
            row = SessionTargetRunRowV1.model_validate({
                "object_id": new_ulid(), "run_kind": "finalize",
                "commit_mode": "orchestrator", "status": "queued", "stage": "await_input",
                "attempt": 0, "run_generation": 0, "idempotency_key": identity,
                "admission_digest": digest, "principal_id": grant.principal_id,
                **scope.model_dump(mode="json"), "target_kind": "scope",
                "target_id": scope.scope_id,
                "writer_epoch": writer_lease["value"]["writer_epoch"],
                "admitted_erase_epoch": writer.get_counter(f"erase_epoch:{scope.scope_id}"),
                "admitted_consent_epoch": writer.get_counter(f"consent_epoch:{scope.scope_id}"),
                "created_at": now,
            })
            write_ticket = writer._admit_run(row, finalize=True)
            previous = row.model_dump(mode="json")
        else:
            writer.parse_row("run", previous)
            if previous["admission_digest"] != digest or (
                previous["status"] != "queued" or previous["stage"] != "await_input"
                or any(previous[name] is not None for name in (
                    "input_ref", "result_ref", "auth_context_ref",
                    "commit_marker", "commit_seq",
                ))
            ):
                raise QuoteMemError("operation_conflict", "Finalize admission has another state")
            write_ticket = WriteTicket(
                previous["writer_epoch"], previous["object_id"], scope.scope_kind,
                scope.scope_id, previous["admitted_erase_epoch"],
                previous["admitted_consent_epoch"],
            )
        writer.check_ticket(replace(write_ticket, run_id=None))
        now = clock()
        check_add_grant(grant, scope, now, previous["principal_id"])
        expected = None if lease is None else lease.model_dump(mode="json")
        if writer.get_control(f"lease:finalize:{scope.scope_id}") != expected:
            raise QuoteMemError("lease_lost", "Finalize cycle changed before preparing acquisition")
        owner = previous["object_id"]
        preparing = writer.materialize_control(
            f"lease:finalize:{scope.scope_id}", {
                "state": "preparing", "lease_generation": cycle, "owner_run_id": owner,
                "writer_epoch": write_ticket.writer_epoch, "heartbeat_at": now,
                "drain_state": "open", "grace_deadline": now + 5000,
            }, 0 if lease is None else lease.version + 1, now, owner,
        )
        running = writer.parse_row("run", dict(
            previous, status="running", stage="lease", started_at=now, updated_at=now,
        ))
        # The allowlisted acquire is one mixed-main upsert, not two CASes.
        # Run and lease full preimages were checked under this Writer mutex.
        rows = [RowWrite("control", preparing), RowWrite("run", running)]
        owner_row = running.model_dump(mode="json")
        rows += [row for row in (
            completion_policy_row(writer, owner_row, now, policy),
            budget_row(writer, owner_row, now, policy=policy),
        ) if row is not None]
        _, plan = writer.plan_rows(rows, None)
        writer.write_plan(plan)
        observe(owner, cycle, new=new_owner, later=cycle > 1, dirty=dirty_predecessor)
        from quotemem.store.read_model import StoreReadModel

        publisher = StoreReadModel.get(writer)
        if publisher is not None:
            publisher.publish_finalize_preparing(
                writer, scope=scope, owner_run_id=owner,
                lease_generation=cycle, writer_epoch=write_ticket.writer_epoch,
            )
        issued = FinalizeLeaseTicket(
            scope.scope_id, cycle, owner, write_ticket.writer_epoch, "preparing",
        )
        writer._register_finalize_ticket(issued, clock)
        writer.add_admission.owner_started(scope.scope_id)
        return issued


def _privacy_cancelled(writer: Writer, owner: Mapping[str, Any]) -> bool:
    """Whether an owner cancelled its own cycle at a privacy rebase.

    `finalize_privacy.rebase_finalize_privacy` cancels an owner that an erase or a
    consent withdrawal reached before its first review and before its settlement
    transaction: it fails ``erase_fenced`` or ``consent_fenced``, and the scope's
    epochs are past the ones it was admitted under.
    """
    scope_id = owner["scope_id"]
    epochs = (writer.get_counter(f"erase_epoch:{scope_id}"),
              writer.get_counter(f"consent_epoch:{scope_id}"))
    admitted = (owner["admitted_erase_epoch"], owner["admitted_consent_epoch"])
    return owner["failure_code"] in {"erase_fenced", "consent_fenced"} and admitted != epochs and (
        admitted[0] <= epochs[0] and admitted[1] <= epochs[1])


def cancelled_settled(writer: Writer, scope: ScopeV1, lease: ControlRowV1) -> None:
    """Prove a cancelled cycle settled before the next one starts.

    The forget that cancelled the cycle wrote its lease through its erasure
    controller and has finished: no erasure fence holds the scope and the
    controller succeeded. Or the owner cancelled its own cycle at a privacy rebase
    (`_privacy_cancelled`), which an erase or consent withdrawal reached before its
    first review. Or an operator skipped the quarantined cycle
    (``store.quarantine_skip``, ``repair_skip``): the owner failed
    ``input_abandoned`` and its requests completed as skipped. The owner failed
    and every child of it ended, so nothing of the cycle can act again; its
    formal commits stand and the next owner reprocesses the requests it left
    incomplete (a skip left none).

    Args:
        writer: The sole Writer.
        scope: The lease's scope.
        lease: The scope's cancelled ``lease:finalize:{s}`` Control.

    Raises:
        QuoteMemError: ``erasure_in_progress`` while that forget still holds the
            scope; ``binding_mismatch`` for a cycle that cannot be proved.
    """
    value = lease.value
    if not isinstance(value, CancelledFinalizeLeaseV1):
        raise QuoteMemError("binding_mismatch", "A cancelled cycle proof needs its cancelled lease")
    for row in writer.indexed_reader().scope_fences(scope.scope_id):
        if str(row["key"]).startswith(f"erasure:{scope.scope_id}:"):
            raise QuoteMemError("erasure_in_progress", "The forget that cancelled the cycle runs")
    controller = (None if lease.updated_by_run_id is None
                  else writer.get_row("run", lease.updated_by_run_id))
    owner = writer.get_row("run", value.owner_run_id)
    if owner is None or (owner["run_kind"], owner["status"], owner["scope_id"]) != (
        "finalize", "failed", scope.scope_id,
    ) or controller is None or not (
        (controller["run_kind"], controller["status"]) == ("erasure_controller", "succeeded")
        or controller["object_id"] == owner["object_id"] and (
            _privacy_cancelled(writer, owner) if value.cancel_reason == "hard_erase"
            # An operator skipped the quarantined cycle (``store.quarantine_skip``).
            else owner["failure_code"] == "input_abandoned"
        )
    ):
        raise QuoteMemError("binding_mismatch", "A cancelled cycle lost its forget or its owner")
    reader = writer.indexed_reader()
    after: str | None = None
    while page := reader.run_children(value.owner_run_id, after=after, limit=512):
        if any(child["status"] not in {"succeeded", "failed"} for child in page):
            raise QuoteMemError("binding_mismatch", "A cancelled cycle kept an unended child")
        after = page[-1]["object_id"]


def close_finalize_drain(
    writer: Writer, ticket: FinalizeLeaseTicket, *, clock: Callable[[], int],
) -> None:
    """Irreversibly close the preparing drain so the cycle can hold.

    Provisional callbacks, the only executors a drain waited for, left with
    the drain; the closed drain is still the hold precondition.
    """
    with writer.write_scope():
        _, lease, _ = writer.read_finalize_fences(ticket)
        value = lease.value.model_dump(mode="json")
        if value["state"] != "preparing":
            raise QuoteMemError("lease_lost", "Only preparing can close its drain")
        if value["drain_state"] == "closed":
            return
        now = clock()
        value.update(drain_state="closed", heartbeat_at=now)
        intended = writer.materialize_control(
            lease.key, value, lease.version + 1, now, ticket.owner_run_id,
        )
        writer._compare_finalize_lease(ticket, lease, intended)


def hold_finalize(
    writer: Writer, ticket: FinalizeLeaseTicket, publisher: StoreReadModel,
    *, clock: Callable[[], int],
) -> FinalizeLeaseTicket:
    """Install the input fence and its retained baseline under one Writer lock.

    Args:
        ticket: The issued preparing identity whose empty drain is closed.
        publisher: This Writer's existing registered publisher.
        clock: Trusted millisecond action clock.

    Returns:
        The identical ticket after its phase changes to held. A held ticket
        republishes its barrier; a held cycle that boot adopted first binds a
        fresh baseline.

    Raises:
        QuoteMemError: If the cycle, executor absence, storage or baseline proof
            differs. Any failed publication after CAS revokes the publisher.
    """
    from quotemem.core.query import ScopeV1
    from quotemem.store.read_model import StoreReadModel

    # A preparing owner's baseline scope view digests the scope's rows: it is built
    # before the Writer hold and lives no longer than the hold
    # (``hold_finalize_baseline``).
    with (
        publisher.hold_finalize_baseline(ticket.scope_id, prepare=ticket.state != "held"),
        writer.write_scope(),
    ):
        run, lease, _ = writer.read_finalize_fences(ticket)
        if StoreReadModel.get(writer) is not publisher:
            raise QuoteMemError("binding_mismatch", "Finalize requires its registered publisher")
        if ticket.state == "held":
            scope = ScopeV1(scope_kind=run["scope_kind"], scope_id=ticket.scope_id)
            identity = (ticket.owner_run_id, ticket.lease_generation, ticket.writer_epoch)
            if adopted_cycle(writer, ticket.scope_id) == identity:
                _rebaseline_adopted(writer, ticket, lease, publisher, scope, clock=clock)
            publisher.publish_finalize_held(writer, scope, *identity)
            return ticket
        value = lease.value.model_dump(mode="json")
        # The held-session check went with lease:session.
        if value["drain_state"] != "closed":
            raise QuoteMemError("lease_lost", "Finalize cannot hold an undrained executor set")
        scope = ScopeV1(scope_kind=run["scope_kind"], scope_id=ticket.scope_id)
        baseline = publisher.capture_finalize_baseline(
            writer, scope, ticket.owner_run_id, ticket.lease_generation, ticket.writer_epoch,
        )
        now = clock()
        intended = writer.materialize_control(lease.key, {
            "state": "held", "lease_generation": ticket.lease_generation,
            "owner_run_id": ticket.owner_run_id, "writer_epoch": ticket.writer_epoch,
            "heartbeat_at": now, "baseline_snapshot_generation": baseline.generation,
            "baseline_root_identity": baseline.root_identity,
        }, lease.version + 1, now, ticket.owner_run_id)
        writer._compare_finalize_lease(ticket, lease, intended)
        ticket.state = "held"
        publisher.publish_finalize_held(
            writer, scope, ticket.owner_run_id, ticket.lease_generation, ticket.writer_epoch,
        )
        return ticket


def _rebaseline_adopted(
    writer: Writer, ticket: FinalizeLeaseTicket, lease: ControlRowV1,
    publisher: StoreReadModel, scope: ScopeV1, *, clock: Callable[[], int],
) -> None:
    """Bind an adopted held cycle to a fresh baseline at its owner's first hold.

    The original baseline died with its process. The read model retains the
    current root, with the scope's barrier lifted, as the owner's input
    baseline; the lease CAS records it, and the caller publishes the held
    barrier again. The caller holds the Writer scope.

    Raises:
        QuoteMemError: ``lease_lost`` for another cycle or a lost lease;
            ``clock_rollback`` when the clock precedes the lease heartbeat.
    """
    value = lease.value
    if not isinstance(value, HeldFinalizeLeaseV1):
        raise QuoteMemError("lease_lost", "A cold baseline requires its adopted held lease")
    now = clock()
    if now < value.heartbeat_at:
        raise QuoteMemError("clock_rollback", "Cold baseline clock regressed")
    baseline = publisher.capture_cold_finalize_baseline(
        writer, scope, ticket.owner_run_id, ticket.lease_generation, ticket.writer_epoch,
    )
    writer._compare_finalize_lease(ticket, lease, writer.materialize_control(lease.key, {
        **value.model_dump(mode="json"), "heartbeat_at": now,
        "baseline_snapshot_generation": baseline.generation,
        "baseline_root_identity": baseline.root_identity,
    }, lease.version + 1, now, ticket.owner_run_id))
    del _adopted_cycles(writer)[ticket.scope_id]


def heartbeat_finalize(
    writer: Writer, ticket: FinalizeLeaseTicket, *, clock: Callable[[], int],
) -> None:
    """Refresh one issued cycle without extending its fixed takeover grace.

    The precondition judges the lease at the owner's action time
    (``Writer._finalize_action_time``): an owner whose action began before its
    lease expired renews it at the end of that action (``renew_due_leases``).

    Args:
        ticket: Exact process-issued preparing or held identity.
        clock: Trusted millisecond action clock.

    Raises:
        QuoteMemError: If ownership changed, the lease had expired at the action
            time or the action clock regressed.
    """
    with writer.write_scope():
        _, lease, _ = writer.read_finalize_fences(ticket)
        value = lease.value.model_dump(mode="json")
        now = clock()
        if now < value["heartbeat_at"]:
            raise QuoteMemError("invalid_request", "Finalize heartbeat clock regressed")
        value["heartbeat_at"] = writer._job_timestamp(now)
        writer._compare_finalize_lease(ticket, lease, writer.materialize_control(
            lease.key, value, lease.version + 1, now, ticket.owner_run_id,
        ))


def renew_due_leases(writer: Writer, acting: Mapping[str, FinalizeLeaseTicket]) -> None:
    """Renew due leases at the end of a completing outermost write scope.

    The scheduler's heartbeat pass renews every live lease, but it reads its lease
    page under HeavenBase's writer-preferring workspace lock and renews under the
    Writer mutex, and an owner writing back to back keeps both: the full a7
    workload's 976 judge child admissions held them for 55 s without one pass and
    expired the lease mid-admission. Renewal therefore does not wait for the pass.
    Each completing outermost scope renews, with the pass's heartbeat CAS and still
    inside its exclusion, the lease of every owner whose execution fence it proved
    once that lease is 20,000 ms old, and then the due lease of one other owner
    holding a live ticket of this Writer, such as one waiting on a provider call,
    oldest heartbeat first: one renewal per scope bounds the exclusion as the
    pass's own batch of one does. An owner that has lost its live ticket, its
    running held or preparing cycle or its unexpired lease, or whose scope refuses
    its fences, is left as it is: its next action refuses on its own fence, and the
    pass treats it as before.

    Args:
        writer: The sole Writer, inside the completing outermost write scope.
        acting: By owner Run, the live tickets whose execution fences the scope
            proved.

    Raises:
        QuoteMemError: A runtime failure of a renewal; a refusal renews nothing.
    """
    # A fixture Writer built around __init__ issues no ticket.
    tickets = cast(
        Mapping[str, FinalizeLeaseTicket] | None, writer.__dict__.get("_finalize_tickets"),
    )
    clocks = cast(
        Mapping[FinalizeLeaseTicket, Callable[[], int]] | None,
        writer.__dict__.get("_finalize_clocks"),
    )
    if not tickets or clocks is None:
        return
    seen = writer._finalize_heartbeats()
    due: list[tuple[int, str, FinalizeLeaseTicket]] = []
    for owner_run_id, issued in tuple(tickets.items()):
        clock, heartbeat_at = clocks.get(issued), seen.get(issued)
        if clock is not None and (
            heartbeat_at is None or clock() - heartbeat_at >= OWNER_LEASE_RENEWAL_AGE_MS
        ):
            due.append((-1 if heartbeat_at is None else heartbeat_at, owner_run_id, issued))
    due.sort(key=lambda entry: entry[:2])
    other = False
    for _, owner_run_id, issued in due:
        if owner_run_id in acting:
            _renew_due(writer, issued)
        elif not other:
            other = _renew_due(writer, issued)


def _renew_due(writer: Writer, issued: FinalizeLeaseTicket) -> bool:
    """Renew one live owner's lease once it is 20,000 ms old, inside the held scope.

    Returns:
        Whether the lease was renewed. A lease found younger keeps the heartbeat
        read; any other is looked at again only after the renewal age.

    Raises:
        QuoteMemError: A runtime failure of the renewal; a refusal renews nothing.
    """
    from quotemem.store.failures import is_runtime_failure

    clock = writer._finalize_clocks.get(issued)
    if clock is None or writer._finalize_tickets.get(issued.owner_run_id) is not issued:
        return False
    seen = writer._finalize_heartbeats()
    now = clock()
    raw = writer.get_control(f"lease:finalize:{issued.scope_id}")
    heartbeat_at = None if raw is None else raw["value"].get("heartbeat_at")
    if isinstance(heartbeat_at, int) and now - heartbeat_at < OWNER_LEASE_RENEWAL_AGE_MS:
        seen[issued] = heartbeat_at
        return False
    renewed = False
    try:
        if isinstance(heartbeat_at, int):
            run, lease, _ = writer.read_finalize_fences(issued)
            if run["status"] == "running" and isinstance(
                lease.value, PreparingFinalizeLeaseV1 | HeldFinalizeLeaseV1,
            ):
                heartbeat_finalize(writer, issued, clock=clock)
                renewed = True
    except QuoteMemError as error:
        if is_runtime_failure(error):
            raise
    seen[issued] = now
    return renewed


def persist_finalize_input(
    writer: Writer, proof: FinalizeInputProof, *, grant: PrincipalGrant,
    clock: Callable[[], int],
) -> ArtifactId:
    """Create the issued owner input once, select it and move the owner to ``extract``.

    The owner loop's first step ends here: the prefix is frozen and the owner's next
    stage is the extraction. Complete row scans and the source inventory
    happen outside the mutex (``prepare_finalize_input``); this fixed-key
    effect revalidates the private read lifetime, the held cycle, the owner,
    child and Artifact generations and the scope coordinates, then attaches
    the input to the owner Run. It opens no child, provider call, business
    generation or release. Repeating it for the selected input writes nothing.

    Args:
        writer: The sole Writer.
        proof: The issued input proof of this held cycle.
        grant: The owner principal's grant.
        clock: Trusted millisecond action clock.

    Returns:
        The selected ``run_input:finalize:owner`` Artifact.

    Raises:
        QuoteMemError: ``parity_mismatch`` when another input is selected or the
            stored body differs; ``stale_evidence`` or ``reception_stale`` when
            an authority changed after the proof; ``lease_lost`` for a lost cycle.
    """
    from quotemem.core.query import ScopeV1
    from quotemem.ingestion.authorization import check_add_grant
    from quotemem.ingestion.finalize_freeze import finalize_action_time, validate_finalize_input

    with writer.write_scope():
        validate_finalize_input(writer, proof, grant, clock())
        run, _, ticket = writer.read_finalize_fences(proof.ticket)
        scope = ScopeV1(scope_kind=run["scope_kind"], scope_id=run["scope_id"])

        def action_time() -> int:
            # Artifact materialization and row preparation can outlive the grant;
            # an own prewrite changes the Artifact generation, so the proof's
            # generation is not rechecked after it.
            selected_time = finalize_action_time(clock)
            check_add_grant(grant, scope, selected_time, run["principal_id"])
            guard = proof.guard()
            if guard is None:
                raise QuoteMemError("reader_busy", "Finalize input lost its content lifetime")
            guard.check_operation(writer, scope)
            if not writer.is_borrow_unchanged(proof.receipt_keys, proof.receipt_token):
                raise QuoteMemError("reception_stale",
                                    "Finalize receipt metadata changed before CAS")
            return selected_time

        reference = proof.input_ref
        if reference is None:
            reference = writer.put_logical_artifact(
                ticket, "run_input", proof.input, created_at=proof.input.freeze_time,
            )
        if writer._read_owned_business_artifact(ticket, reference, "run_input") != proof.input:
            raise QuoteMemError("parity_mismatch", "Finalize input differs from its issued body")
        now = action_time()
        if run["input_ref"] is not None:
            if run["input_ref"] != reference:
                raise QuoteMemError("parity_mismatch", "Finalize already selected another input")
            writer.read_bound_input(run)
            return reference
        run.update(input_ref=reference, stage="extract", updated_at=now)
        writer.read_bound_input(run)
        _, plan = writer.plan_rows([RowWrite("run", writer.parse_row("run", run))], ticket)
        action_time()
        writer.write_plan(plan)
        return reference


def end_abandoned_reviews(
    writer: Writer, ticket: FinalizeLeaseTicket, *, grant: PrincipalGrant,
    clock: Callable[[], int],
) -> int:
    """End the judge children a completed owner left unfinished, before its release.

    A completed owner holds every position's final disposition, and one owner
    runs its reviews serially, so a judge child still running is a review a
    stopped invocation began and a later one abandoned: every send was refused
    for the key, or the invocation stopped before the reply, and the resumed
    owner settled or carried the member without it (its rebuilt input no
    longer reproduced the child's), or no longer selected it. A confirmation
    vote sent together with an earlier vote that already decided can also be
    left running when its request failed (``JudgeStage._votes_together``). Such a child
    would refuse the release on every resume. It fails ``input_abandoned``;
    its admitted input stays. Other children keep the release's check.

    Args:
        writer: The sole Writer.
        ticket: The held owner's issued cycle.
        grant: The owner principal's grant.
        clock: Trusted millisecond action clock.

    Returns:
        int: The judge children ended.

    Raises:
        QuoteMemError: ``lease_lost`` for a lost cycle; the grant check's refusal.
    """
    from quotemem.core.judge_records import is_judgment_target
    from quotemem.core.query import ScopeV1
    from quotemem.ingestion.authorization import check_add_grant
    from quotemem.store.indexed import IndexedReader
    from quotemem.store.writer import MAX_TX_ROWS

    reader = IndexedReader(writer.runtime.workspace, writer.entities, tenant_id=writer.tenant_id)
    with writer.write_scope():
        run, _lease, effect = writer.read_finalize_fences(ticket)
        now = clock()
        check_add_grant(grant, ScopeV1(scope_kind=run["scope_kind"], scope_id=run["scope_id"]),
                        now, run["principal_id"])
        rows: list[RowWrite] = []
        after: str | None = None
        while children := reader.run_children(ticket.owner_run_id, after=after, limit=512):
            rows.extend(RowWrite("run", writer.parse_row("run", dict(
                child, status="failed", failure_code="input_abandoned", updated_at=now,
                finished_at=now,
            ))) for child in children if child["run_kind"] == "evaluate"
                and child["status"] == "running" and is_judgment_target(str(child["target_id"])))
            after = children[-1]["object_id"]
        for start in range(0, len(rows), MAX_TX_ROWS):
            _, plan = writer.plan_rows(rows[start:start + MAX_TX_ROWS], effect)
            writer.write_plan(plan)
    return len(rows)


def _require_ended_children(writer: Writer, owner_run_id: str) -> None:
    """Refuse a release while one of the owner's judge or publish children still runs."""
    from quotemem.store.indexed import IndexedReader

    reader = IndexedReader(writer.runtime.workspace, writer.entities, tenant_id=writer.tenant_id)
    after: str | None = None
    while children := reader.run_children(owner_run_id, after=after, limit=512):
        if any(child["status"] not in {"succeeded", "failed"} for child in children):
            raise QuoteMemError("invalid_run_transition",
                                "Finalize release requires ended children")
        after = children[-1]["object_id"]


def release_finalize(
    writer: Writer, ticket: FinalizeLeaseTicket, publisher: StoreReadModel, *,
    grant: PrincipalGrant, clock: Callable[[], int],
) -> None:
    """Release a completed owner: its finalized marker, released lease and settled Run.

    The owner loop's last step, after ``store/add_outcome.record_add_completion``. Under one Writer
    scope the ``finalized:{scope}`` marker records the frozen input heads and the
    scope sequence the owner settled; the read model publishes it and lifts
    the owner's barrier; the lease moves to ``released`` and the held baseline is
    dropped; the owner Run succeeds together with the ``completed_prefix:{scope}``
    projection, which makes its request results visible (``completed_result``).
    Every child of the owner has ended and the published view holds every commit,
    so the response that follows reads its own effects. A release that died after
    its marker write left that marker under the held lease; the adopted owner's
    release proves it is this cycle's own and reuses it unchanged.

    Args:
        writer: The sole Writer.
        ticket: The held owner's issued cycle.
        publisher: This Writer's registered read model.
        grant: The owner principal's grant.
        clock: Trusted millisecond action clock.

    Raises:
        QuoteMemError: ``invalid_run_transition`` without a completion record or with a
            running child; ``service_unhealthy`` when the published view lags a
            commit; ``binding_mismatch`` for another publisher or a changed
            completion binding; ``clock_rollback`` when the clock moved behind the
            last heartbeat; ``lease_lost`` for a lost cycle.
    """
    from quotemem.core.query import ScopeV1
    from quotemem.core.run_inputs import FinalizeOwnerRunInputV1
    from quotemem.ingestion.authorization import check_add_grant
    from quotemem.service.request_measurement import resource_session_event
    from quotemem.store.read_model import StoreReadModel

    with writer.write_scope():
        run, lease, effect = writer.read_finalize_fences(ticket)
        scope = ScopeV1(scope_kind=run["scope_kind"], scope_id=run["scope_id"])
        now = clock()
        check_add_grant(grant, scope, now, run["principal_id"])
        if StoreReadModel.get(writer) is not publisher:
            raise QuoteMemError("binding_mismatch",
                                "Finalize release lost its registered publisher")
        if not isinstance(lease.value, HeldFinalizeLeaseV1) or ticket.state != "held":
            raise QuoteMemError("lease_lost", "Finalize release requires the held owner")
        if (run["status"], run["stage"]) != ("running", "release") or run["result_ref"] is None:
            raise QuoteMemError("invalid_run_transition", "Finalize release requires a completion")
        if now < lease.value.heartbeat_at:
            raise QuoteMemError("clock_rollback", "Finalize release clock regressed")
        selected = FinalizeOwnerRunInputV1.model_validate(writer.read_bound_input(run))
        _require_ended_children(writer, ticket.owner_run_id)
        if publisher.current.generation < writer.get_counter("commit_seq"):
            raise QuoteMemError("service_unhealthy", "Finalize release lacks the published commits")
        marker_key = f"finalized:{scope.scope_id}"
        previous = writer.get_control(marker_key)
        value = FinalizedValueV1(
            run_id=ticket.owner_run_id, lease_generation=ticket.lease_generation, at=now,
            input_heads_digest=selected.input_heads_digest,
            finalized_scope_seq=writer.get_counter(f"scope_seq:{scope.scope_id}"),
        )
        marker = _cycle_marker(previous, value)
        write_marker = marker is None
        if marker is None:
            marker = writer.materialize_control(marker_key, value.model_dump(mode="json"), (
                0 if previous is None else ControlRowV1.model_validate(previous).version + 1
            ), now, ticket.owner_run_id)
        released = writer.materialize_control(lease.key, dict(
            lease.value.model_dump(mode="json"), state="released", released_at=now,
        ), lease.version + 1, now, ticket.owner_run_id)
        # Validate the attached first result before any marker, hold or lease
        # effect. A resume can finish release without rewriting that result.
        settled = dict(run, status="succeeded", updated_at=now, finished_at=now)
        settled_rows = _settled_rows(writer, settled, released, marker)
        if write_marker:
            holds, released_holds = _released_holds(writer, scope, run, now)
            _, plan = writer.plan_rows([RowWrite("control", marker), *holds], effect)
            if released_holds:
                writer.plan_deletes(plan, released_holds)
            writer.write_plan(plan)
        publisher.publish_finalize_marker(
            writer, scope, ticket.owner_run_id, ticket.lease_generation, ticket.writer_epoch,
        )
        writer._compare_finalize_lease(ticket, lease, released)
        publisher.release_finalize_baseline(
            writer, scope, ticket.owner_run_id, ticket.lease_generation, ticket.writer_epoch,
        )
        if writer.get_control(lease.key) != released.model_dump(mode="json") or (
            writer.get_control(marker_key) != marker.model_dump(mode="json")
        ):
            raise QuoteMemError("parity_mismatch", "Finalize release lost its released proof")
        _, plan = writer.plan_rows(settled_rows, None)
        writer.write_plan(plan)
        writer._finalize_tickets.pop(ticket.owner_run_id, None)
        resource_session_event(
            "settled", scope_id=scope.scope_id, run_id=ticket.owner_run_id,
            finalize_cycle=ticket.lease_generation, commit_seq=selected.receipt_high_watermark,
        )
        writer.add_admission.owner_released(scope.scope_id)


def _released_holds(
    writer: Writer, scope: ScopeV1, run: Mapping[str, Any], now: int,
) -> tuple[list[RowWrite], list[tuple[str, dict[str, Any]]]]:
    """The scope's outliving Evidence holds after this release settles its requests.

    A forget that cancelled an earlier owner kept its deleted rows' holds in
    ``evidence_hold:{s}`` (``ForgetOwner._settle_cohort``). The releasing owner
    completed some of those requests: their holds leave with its marker, which
    publishes the scope's held Evidence (``publish_finalize_marker``).

    Returns:
        The Control's new post-image, or its deletion once no request is left.

    Raises:
        QuoteMemError: ``binding_mismatch`` for another Control variant or completion record.
    """
    from quotemem.core.add_outcome import AddCompletionRecordV1
    from quotemem.core.controls import EvidenceHoldV1

    raw = writer.get_control(f"evidence_hold:{scope.scope_id}")
    if raw is None:
        return [], []
    hold = ControlRowV1.model_validate(raw).value
    record = writer.read_logical(run["result_ref"], "result_index")
    if not isinstance(hold, EvidenceHoldV1) or not isinstance(record, AddCompletionRecordV1):
        raise QuoteMemError("binding_mismatch", "Released holds lost their Control or completion")
    completed = {request.request_key for request in record.completed_requests}
    left = [request for request in hold.requests if request.request_key not in completed]
    if len(left) == len(hold.requests):
        return [], []
    if not left:
        return [], [("control", dict(raw))]
    return [RowWrite("control", writer.materialize_control(
        raw["key"], EvidenceHoldV1(requests=left).model_dump(mode="json"), raw["version"] + 1,
        now, run["object_id"],
    ))], []


def _cycle_marker(
    stored: Mapping[str, Any] | None, value: FinalizedValueV1,
) -> ControlRowV1 | None:
    """The stored marker when this release already wrote it, else None.

    A process killed between release's marker write and its last transaction
    leaves the marker durable under a held lease, and boot adopts the cycle.
    The marker is this cycle's own when its owner wrote it for the same cycle,
    selected input and scope sequence: only its time would differ, so the
    owner's next release publishes it as it stands. Any other marker is
    replaced by the next version, as an ordinary release does.
    """
    if stored is None:
        return None
    control = ControlRowV1.model_validate(stored)
    found = control.value
    if not isinstance(found, FinalizedValueV1) or control.updated_by_run_id != value.run_id or (
        found.run_id, found.lease_generation, found.input_heads_digest,
        found.finalized_scope_seq,
    ) != (value.run_id, value.lease_generation, value.input_heads_digest,
          value.finalized_scope_seq):
        return None
    return control


def _settled_rows(
    writer: Writer, owner: dict[str, Any], released: ControlRowV1, marker: ControlRowV1,
) -> list[RowWrite]:
    """Release's last transaction: the succeeded owner Run and its completed prefix."""
    from quotemem.store.request_completion import completed_row

    return [RowWrite("run", writer.parse_row("run", owner)),
            completed_row(writer, owner, released, marker)]


def bind_finalize_pool(writer: Writer, pool: BenchWorkPool) -> None:
    """Bind the composition root's sole owner worker pool before startup.

    Pauses and explicit recovery validate their pool proofs against this pool.

    Raises:
        QuoteMemError: ``operation_conflict`` when the Writer already has another pool.
    """
    from quotemem.service.bench_workers import BenchWorkPool

    with writer.mutex:
        if type(pool) is not BenchWorkPool or (
            writer._finalize_pool is not None and writer._finalize_pool is not pool
        ):
            raise QuoteMemError("operation_conflict", "The Writer already has another owner pool")
        writer._finalize_pool = pool


def pause_finalize(writer: Writer, work: BenchWorkPause, *, clock: Callable[[], int]) -> None:
    """Reserve an expired owner while its exact worker cancellation drains.

    The pause changes no durable row: it records the owner and lease images so
    that ``resume_finalize`` renews only the same unchanged cycle after the pool
    proves the cancelled invocation joined.

    Args:
        work: The pool's issued pause of this owner's invocation.
        clock: Trusted millisecond action clock.

    Raises:
        QuoteMemError: ``lease_lost`` unless the bound pool issued the pause and the
            owner's live lease is actually past its lifetime.
    """
    with writer.mutex:
        pool = writer._finalize_pool
        if pool is None:
            raise QuoteMemError("lease_lost", "Owner pause requires its bound worker pool")
        pool.pause_drained(work)  # Validates issuance while the invocation may still run.
        owner_id = work.work.run_id
        paused = _paused_owners(writer)
        previous = paused.get(owner_id)
        if previous is not None:
            if previous.work is not work:
                raise QuoteMemError("lease_lost", "Owner pause changed its original invocation")
            return
        issued = writer._finalize_tickets.get(owner_id)
        if issued is None:
            owner = writer.get_row("run", owner_id)
            raw = writer.get_control(f"lease:finalize:{work.work.scope_id}")
            if owner is None or raw is None:
                raise QuoteMemError("lease_lost", "Owner pause lost its durable owner")
            value = ControlRowV1.model_validate(raw).value
            if not isinstance(value, PreparingFinalizeLeaseV1 | HeldFinalizeLeaseV1):
                raise QuoteMemError("lease_lost", "Owner pause requires a live cycle")
            issued = FinalizeLeaseTicket(
                work.work.scope_id, value.lease_generation, owner_id,
                value.writer_epoch, value.state,
            )
            writer._register_finalize_ticket(issued, clock)
        owner, lease, _ = writer._finalize_identity(issued)
        now = writer._job_timestamp(clock())
        heartbeat_at = lease.value.model_dump(mode="json")["heartbeat_at"]
        if issued.scope_id != work.work.scope_id or now - heartbeat_at <= OWNER_LEASE_LIFETIME_MS:
            raise QuoteMemError("lease_lost", "Owner pause requires the exact expired cycle")
        paused[owner_id] = _PausedOwner(
            issued, work, canonical_json(owner), canonical_json(lease),
        )


def resume_finalize(
    writer: Writer, work: BenchWorkPause, *, clock: Callable[[], int],
) -> FinalizeLeaseTicket:
    """Renew a paused owner's lease only after the pool proves its cancelled invocation joined.

    No owner, input, child or baseline is replaced and the deadline is kept
    (nothing automatic extends it). A preparing cycle closes its existing
    drain. Earlier issued tickets stay invalid after this one coordinating CAS,
    so they cannot publish late effects.

    Args:
        work: The pause ``pause_finalize`` recorded.
        clock: Trusted millisecond action clock.

    Returns:
        The owner's newly issued ticket.

    Raises:
        QuoteMemError: ``reader_busy`` while the owner's executor still runs;
            ``lease_lost`` unless the recorded pause is drained and its queue slot
            reserved and the owner and lease are unchanged; ``clock_rollback``
            when the clock precedes the lease's heartbeat.
    """
    executor = writer.run_executor(work.work.run_id)
    if not executor.acquire(blocking=False):
        raise QuoteMemError("reader_busy", "Owner executor has not joined")
    try:
        with writer.write_scope():
            pool = writer._finalize_pool
            paused = _paused_owners(writer)
            saved = paused.get(work.work.run_id)
            if pool is None or saved is None or saved.work is not work or (
                not pool.pause_drained(work)
            ):
                raise QuoteMemError("lease_lost", "Owner has no proved drained invocation")
            pool.require_resume_reserved(work)
            owner, lease, _ = writer._finalize_identity(saved.ticket)
            if canonical_json(owner) != saved.owner_json or canonical_json(lease) != (
                saved.lease_json
            ):
                raise QuoteMemError("lease_lost", "Paused owner changed before continuation")
            now = writer._job_timestamp(clock())
            value = lease.value.model_dump(mode="json")
            if now < value["heartbeat_at"]:
                raise QuoteMemError("clock_rollback", "Owner resume clock regressed")
            value["heartbeat_at"] = now
            if saved.ticket.state == "preparing":
                value["drain_state"] = "closed"
            intended = writer.materialize_control(
                lease.key, value, lease.version + 1, now, saved.ticket.owner_run_id,
            )
            writer._write_finalize_lease_cas(lease, intended)
            issued = replace(saved.ticket)
            del paused[work.work.run_id]
            writer._register_finalize_ticket(issued, clock)
            return issued
    finally:
        executor.release()


def quarantine_scope(
    writer: Writer, scope: ScopeV1, owner_run_id: str, error_code: str, *, clock: Callable[[], int],
) -> ControlRowV1:
    """Fence one refused cycle without mutating its Run, lease or artifacts.

    Args:
        scope: Registered scope whose current owner refused work.
        owner_run_id: Exact finalize lease owner, or a same-scope Run when
            no active finalize lease exists.
        error_code: Content-free typed failure code, never exception text.
        clock: Trusted clock used only for the first refusal receipt.

    Returns:
        The first durable receipt. Repeated calls preserve its original bytes.

    Raises:
        QuoteMemError: If storage ownership, registration or the current owner
            changed, or the scoped receipt cannot be durably established.

    Quarantine supplies no recovery or retry permission. A new process also
    keeps it fenced until an explicit repair can prove the original cycle.
    """
    from quotemem.core.controls import ScopeQuarantineV1
    from quotemem.store.read_model import StoreReadModel

    with writer.write_scope():
        registration = writer.get_control(f"scope:{scope.scope_id}")
        raw = writer.get_control(f"lease:finalize:{scope.scope_id}")
        if registration is None or (
            registration["value"]["scope_kind"] != scope.scope_kind
        ):
            raise QuoteMemError("scope_denied", "Quarantine requires its registered scope")
        lease = None if raw is None else ControlRowV1.model_validate(raw)
        active = lease.value if lease is not None and isinstance(
            lease.value, PreparingFinalizeLeaseV1 | HeldFinalizeLeaseV1,
        ) else None
        owner = writer.get_row("run", owner_run_id)
        if active is not None and active.owner_run_id != owner_run_id:
            raise QuoteMemError("lease_lost", "Quarantine lost the observed finalize owner")
        if active is None and (owner is None or (
            owner["scope_kind"], owner["scope_id"]
        ) != (scope.scope_kind, scope.scope_id)):
            raise QuoteMemError("binding_mismatch", "Quarantine lost its exact scoped owner")
        cycle = None if active is None else active.lease_generation
        key = f"scope_quarantine:{scope.scope_id}"
        previous = writer.get_control(key)
        if previous is not None:
            receipt = ControlRowV1.model_validate(previous)
            if not isinstance(receipt.value, ScopeQuarantineV1) or (
                receipt.value.owner_run_id != owner_run_id
                or receipt.value.finalize_cycle != cycle
            ):
                raise QuoteMemError("binding_mismatch", "Quarantine belongs to another cycle")
            return receipt
        epoch = writer.get_control("lease:writer")
        if epoch is None:
            raise QuoteMemError("lease_lost", "Quarantine lost the process Writer epoch")
        receipt = writer.materialize_control(key, ScopeQuarantineV1(
            owner_run_id=owner_run_id, finalize_cycle=cycle,
            writer_epoch=epoch["value"]["writer_epoch"],
            stage="missing_owner" if owner is None else owner["stage"],
            error_code=error_code,
        ), 0, clock())
        _, plan = writer.plan_rows([RowWrite("control", receipt)], None)
        writer.write_plan(plan)
        publisher = StoreReadModel.get(writer)
        if publisher is not None:
            publisher.publish_scope_quarantine(scope)
        return receipt


def repair_quarantine(
    writer: Writer, scope: ScopeV1, owner_run_id: str, budget: RowWrite, *, now: int,
) -> None:
    """Lift a scope's quarantine around its owner's explicit resume.

    The quarantine receipt froze the refused cycle as it stood: its owner, lease,
    budget and records (``quarantine_scope``). The caller holds the Writer scope
    and has proved the explicit resume, as for a provider pause or a refusal: the
    pool's actual join, the grant, the observed budget and
    ``owner_recovery.explicit_resume_allowance``, whose fresh counted allowance
    ``budget`` writes. The receipt must name this owner and the cycle of the
    scope's active lease, and the owner must prove that cycle's admission.

    In one Writer plan the owner takes its next attempt and, with its live
    children and its lease, the current Writer epoch, as a cold adoption moves
    them (``_adopt_finalize``); the lease's heartbeat renews, the budget is
    written and the receipt is deleted. A held cycle is then recorded as adopted,
    since the quarantine discarded its baseline: its owner's next hold binds a
    fresh one (``hold_finalize``). The read model lifts the quarantine in the same
    critical section. The resumed owner continues from its durable records.

    Args:
        writer: The sole Writer; the caller holds its write scope.
        scope: The quarantined scope.
        owner_run_id: The owner the receipt names.
        budget: The owner's budget row with the resume's fresh allowance.
        now: The resume transaction's trusted clock.

    Raises:
        QuoteMemError: ``operation_conflict`` when the receipt no longer names the
            scope's active cycle; ``binding_mismatch`` for an owner, lease or
            child that cannot be proved; ``parity_mismatch`` for a parity-failed
            owner, which is never repaired; ``index_rebuilding`` while privacy
            work owns the scope; ``clock_rollback`` for a clock behind the lease.
    """
    from quotemem.core.admission import admission_identity, finalize_admission
    from quotemem.core.controls import ScopeQuarantineV1
    from quotemem.store.read_model import StoreReadModel

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
    current = writer.get_control("lease:writer")
    if current is None:
        raise QuoteMemError("lease_lost", "Quarantine repair lost the process Writer epoch")
    epoch: int = current["value"]["writer_epoch"]
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
        or not value.writer_epoch <= owner["writer_epoch"] <= epoch
    ):
        raise QuoteMemError("binding_mismatch", "Quarantine repair cannot prove its original cycle")
    if (owner["status"], owner["failure_code"]) == ("manual_attention", "parity_mismatch"):
        raise QuoteMemError("parity_mismatch", "A parity-failed owner is never repaired")
    if owner["status"] != "running" and not owner_stopped(owner):
        raise QuoteMemError("binding_mismatch",
                            "Quarantine repair requires a running or stopped owner")
    reader = writer.indexed_reader()
    if writer.get_control(f"tomb:{scope.scope_id}") is not None or (
        reader.scope_fences(scope.scope_id) or reader.pending_privacy(scope.scope_id, limit=1)
    ):
        raise QuoteMemError("index_rebuilding", "Privacy settlement precedes quarantine repair")
    if now < value.heartbeat_at:
        raise QuoteMemError("clock_rollback", "Quarantine repair clock regressed")
    rows = [budget, RowWrite("run", writer.parse_row("run", dict(
        owner, writer_epoch=epoch, status="running", failure_code=None,
        attempt=owner["attempt"] + 1, updated_at=now, finished_at=None,
    )))]
    after: str | None = None
    while page := reader.run_children(owner_run_id, after=after):
        for child in page:
            if child["writer_epoch"] > owner["writer_epoch"] or (
                child["scope_kind"], child["scope_id"],
            ) != (scope.scope_kind, scope.scope_id):
                raise QuoteMemError("binding_mismatch", "Quarantine repair found a foreign child")
            if child["status"] in {"queued", "running"} and child["writer_epoch"] != epoch:
                rows.append(RowWrite("run", writer.parse_row("run", dict(
                    child, writer_epoch=epoch,
                ))))
        after = page[-1]["object_id"]
    rows.append(RowWrite("control", writer.materialize_control(lease.key, {
        **value.model_dump(mode="json"), "writer_epoch": epoch, "heartbeat_at": now,
    }, lease.version + 1, now, owner_run_id)))
    _, plan = writer.plan_rows(rows, None)
    writer.plan_deletes(plan, [("control", raw)])
    writer.write_plan(plan)
    identity = (owner_run_id, value.lease_generation, epoch)
    if isinstance(value, HeldFinalizeLeaseV1):
        _adopted_cycles(writer)[scope.scope_id] = identity
    publisher = StoreReadModel.get(writer)
    if publisher is not None:
        publisher.publish_scope_repair(writer, scope, *identity)


def _cold_owner(writer: Writer, scope: ScopeV1, lease: Mapping[str, Any]) -> ColdOwner:
    """Read one scope's owner as the cold classification takes it (boot step 14)."""
    from quotemem.store.released_history import prove_released_history

    value = lease["value"]
    owner = writer.get_row("run", value["owner_run_id"])
    current = writer.get_control("lease:writer")
    if current is None:
        raise QuoteMemError("service_unhealthy", "Cold finalize recovery lost the Writer lease")
    children: list[tuple[Any, int]] = []
    if owner is not None:
        reader = writer.indexed_reader()
        after: str | None = None
        while page := reader.run_children(owner["object_id"], after=after):
            children.extend((child["status"], child["writer_epoch"]) for child in page)
            after = page[-1]["object_id"]
    proved = False
    if value["state"] == "released":
        try:
            prove_released_history(writer, scope, ControlRowV1.model_validate(lease))
            proved = True
        except QuoteMemError:
            proved = False
    return ColdOwner(
        scope_id=scope.scope_id, lease_state=value["state"],
        lease_owner_run_id=value["owner_run_id"], lease_writer_epoch=value["writer_epoch"],
        owner_run_id=None if owner is None else owner["object_id"],
        owner_status=None if owner is None else owner["status"],
        owner_writer_epoch=None if owner is None else owner["writer_epoch"],
        owner_scope_id=None if owner is None else owner["scope_id"],
        current_writer_epoch=current["value"]["writer_epoch"],
        children=tuple(children), history_proved=proved,
    )


def _adopt_finalize(
    writer: Writer, scope: ScopeV1, lease: Mapping[str, Any], *, clock: Callable[[], int],
) -> None:
    """Adopt one interrupted cycle under this boot's Writer epoch.

    The same owner keeps its identity, frozen prefix, selected input, recorded
    outputs and verdicts, budget and original deadline: only the Writer epoch
    of the owner, of its live children and of the lease moves, and the lease's
    heartbeat renews (a preparing cycle also closes its drain). Nothing
    re-freezes or re-extracts and no deadline extends. A paused or refused
    owner stays stopped for its explicit resume. A held cycle is recorded for
    its owner's next hold, which binds a fresh baseline (``hold_finalize``).
    The owner Run and its live children move first, in one transaction; a
    crash before the lease CAS leaves the lease at an earlier epoch than its
    owner, which the next boot adopts in the same way.

    Raises:
        QuoteMemError: ``binding_mismatch`` for an owner, lease or child that cannot
            be proved; ``parity_mismatch`` for a parity-failed owner;
            ``index_rebuilding`` while privacy work owns the scope;
            ``clock_rollback`` when the clock precedes the lease heartbeat.
    """
    from quotemem.core.admission import admission_identity, finalize_admission
    from quotemem.store.read_model import StoreReadModel

    with writer.write_scope():
        if not writer._finalize_boot_recovery_open or StoreReadModel.get(writer) is not None:
            raise QuoteMemError("invalid_write_mode", "Cold adoption requires unexposed startup")
        raw = writer.get_control(f"lease:finalize:{scope.scope_id}")
        current = writer.get_control("lease:writer")
        if raw is None or raw != lease or current is None:
            raise QuoteMemError("binding_mismatch", "Cold adoption lost its observed lease")
        control = ControlRowV1.model_validate(raw)
        value = control.value
        epoch: int = current["value"]["writer_epoch"]
        if not isinstance(value, PreparingFinalizeLeaseV1 | HeldFinalizeLeaseV1):
            raise QuoteMemError("binding_mismatch", "Cold adoption requires an active cycle")
        owner = writer.get_row("run", value.owner_run_id)
        if owner is None:
            raise QuoteMemError("binding_mismatch", "Cold adoption lost its owner")
        writer.parse_row("run", owner)
        family, digest, _ = admission_identity(finalize_admission(
            principal_id=owner["principal_id"], scope=scope, cycle=value.lease_generation,
        ), writer.bind_content)
        preparing = isinstance(value, PreparingFinalizeLeaseV1)
        if (
            (owner["run_kind"], owner["commit_mode"], owner["target_kind"], owner["target_id"])
            != ("finalize", "orchestrator", "scope", scope.scope_id)
            or (owner["scope_kind"], owner["scope_id"]) != (scope.scope_kind, scope.scope_id)
            or owner["parent_run_id"] is not None or owner["run_generation"] != 0
            or (owner["idempotency_key"], owner["admission_digest"]) != (family, digest)
            or control.updated_by_run_id != owner["object_id"]
            or not value.writer_epoch <= owner["writer_epoch"] < epoch
            or owner["attempt"] > 0 and owner["input_ref"] is None
            or preparing and (owner["attempt"], owner["stage"]) != (0, "lease")
            or preparing and any(owner[name] is not None for name in (
                "input_ref", "result_ref", "commit_marker", "commit_seq",
            ))
        ):
            raise QuoteMemError("binding_mismatch", "Cold adoption cannot prove its owner")
        if (owner["status"], owner["failure_code"]) == ("manual_attention", "parity_mismatch"):
            raise QuoteMemError("parity_mismatch", "A parity-failed owner is never adopted")
        if owner["status"] != "running" and not owner_stopped(owner):
            raise QuoteMemError("binding_mismatch",
                                "Cold adoption requires a running or stopped owner")
        reader = writer.indexed_reader()
        if (owner["admitted_erase_epoch"], owner["admitted_consent_epoch"]) != (
            writer.get_counter(f"erase_epoch:{scope.scope_id}"),
            writer.get_counter(f"consent_epoch:{scope.scope_id}"),
        ) or writer.get_control(f"tomb:{scope.scope_id}") is not None or (
            reader.scope_fences(scope.scope_id) or reader.pending_privacy(scope.scope_id, limit=1)
        ):
            raise QuoteMemError("index_rebuilding", "Privacy settlement precedes cold adoption")
        rows = [RowWrite("run", writer.parse_row("run", dict(owner, writer_epoch=epoch)))]
        after: str | None = None
        while page := reader.run_children(owner["object_id"], after=after):
            for child in page:
                if child["writer_epoch"] > owner["writer_epoch"] or (
                    child["scope_kind"], child["scope_id"],
                ) != (scope.scope_kind, scope.scope_id):
                    raise QuoteMemError("binding_mismatch", "Cold adoption found a foreign child")
                if child["status"] in {"queued", "running"}:
                    # A live child continues under its adopted owner; an ended one keeps its epoch.
                    rows.append(RowWrite("run", writer.parse_row("run", dict(
                        child, writer_epoch=epoch,
                    ))))
            after = page[-1]["object_id"]
        now = writer._job_timestamp(clock())
        if now < value.heartbeat_at:
            raise QuoteMemError("clock_rollback", "Cold adoption clock regressed")
        _, plan = writer.plan_rows(rows, None)
        writer.write_plan(plan)
        adopted = {**value.model_dump(mode="json"), "writer_epoch": epoch, "heartbeat_at": now}
        if preparing:
            adopted["drain_state"] = "closed"
        writer._write_finalize_lease_cas(control, writer.materialize_control(
            control.key, adopted, control.version + 1, now, owner["object_id"],
        ))
        if not preparing:
            _adopted_cycles(writer)[scope.scope_id] = (
                owner["object_id"], value.lease_generation, epoch,
            )


def _finish_release(writer: Writer, scope: ScopeV1, lease: Mapping[str, Any]) -> None:
    """Finish a release that died between its lease CAS and its last transaction.

    ``release_finalize`` writes the finalized marker, moves the lease to
    ``released`` and then, in a last transaction, lets the owner succeed with
    its completed prefix. A process that dies before that transaction leaves a
    released lease over an owner still running at stage ``release``. Boot
    writes that transaction from the durable proofs: the released history of
    the owner as it leaves it, at the lease's release time.

    Raises:
        QuoteMemError: ``binding_mismatch`` for a lease, owner, input, marker or
            completion that cannot be proved; ``finalize_pending`` while a
            child of the owner still runs.
    """
    from quotemem.core.controls import ReleasedFinalizeLeaseV1
    from quotemem.store.read_model import StoreReadModel
    from quotemem.store.released_history import prove_released_history

    with writer.write_scope():
        if not writer._finalize_boot_recovery_open or StoreReadModel.get(writer) is not None:
            raise QuoteMemError("invalid_write_mode", "Release recovery requires unexposed startup")
        raw = writer.get_control(f"lease:finalize:{scope.scope_id}")
        if raw is None or raw != lease:
            raise QuoteMemError("binding_mismatch", "Release recovery lost its observed lease")
        released = ControlRowV1.model_validate(raw)
        value = released.value
        if not isinstance(value, ReleasedFinalizeLeaseV1):
            raise QuoteMemError("binding_mismatch", "Release recovery requires a released lease")
        owner = writer.get_row("run", value.owner_run_id)
        if owner is None or (owner["status"], owner["stage"]) != ("running", "release"):
            raise QuoteMemError("binding_mismatch", "Release recovery requires its releasing owner")
        settled = dict(owner, status="succeeded", updated_at=value.released_at,
                       finished_at=value.released_at)
        prove_released_history(writer, scope, released, owner=settled)
        marker = ControlRowV1.model_validate(writer.get_control(f"finalized:{scope.scope_id}"))
        _, plan = writer.plan_rows(_settled_rows(writer, settled, released, marker), None)
        writer.write_plan(plan)


def recover_released_finalizes(writer: Writer, *, clock: Callable[[], int]) -> None:
    """Settle or adopt every scope's owner before readers start.

    Args:
        writer: The sole newly booted Writer.
        clock: Trusted millisecond clock for the adoption and quarantine receipts.

    Raises:
        QuoteMemError: If startup has already closed, or a cycle lost its scope
            registration or its owner.

    Each scope's owner is classified by ``classify_cold_owner`` over its
    lease, owner Run, children and released-history proof. A proved released
    cycle settles and the read model rebuilds its marker; a release that died
    before its last transaction is finished from its proofs (``_finish_release``)
    and settles alike; an interrupted preparing or held cycle is adopted by the
    same owner under the new Writer epoch (``_adopt_finalize``). A cancelled
    cycle whose owner and children ended settles. A release or an adoption
    that cannot be proved and every refused scope are quarantined here. Other
    scopes start. Recovery keeps all ordinary readers closed and uses the same
    Writer.
    """
    from quotemem.core.query import ScopeV1
    from quotemem.store.failures import is_runtime_failure

    if not writer._finalize_boot_recovery_open:
        raise QuoteMemError("invalid_write_mode", "Finalize startup recovery has already closed")
    try:
        with writer.mutex:
            leases = [
                row for row in writer.iter_rows("control")
                if row["key"].startswith("lease:finalize:")
            ]
        for lease in leases:
            scope_id = lease["key"].removeprefix("lease:finalize:")
            value = lease["value"]
            if writer.is_scope_quarantined(scope_id):
                continue
            registration = writer.get_control(f"scope:{scope_id}")
            if registration is None:
                raise QuoteMemError("scope_denied", "Cold finalize lost its scope registration")
            scope = ScopeV1(scope_kind=registration["value"]["scope_kind"], scope_id=scope_id)
            with writer.mutex:
                disposition = classify_cold_owner(_cold_owner(writer, scope, lease))
            if disposition.action == "settle" and value["state"] in {"released", "cancelled"}:
                # A released cycle with proved history is a legal end state: the
                # read model rebuilds its marker. So is a cancelled
                # cycle whose owner and children ended: the next admission proves
                # its forget complete (``cancelled_settled``).
                continue
            code = disposition.error_code or "service_unhealthy"
            reason = FailureReason("finalize_recovery", None, code, None)
            if disposition.action in {"adopt", "release"}:
                try:
                    if disposition.action == "adopt":
                        _adopt_finalize(writer, scope, lease, clock=clock)
                    else:
                        _finish_release(writer, scope, lease)
                    continue
                except Exception as error:
                    if is_runtime_failure(error):
                        raise
                    code = error.code if isinstance(error, QuoteMemError) else "internal_error"
                    reason = failure_reason("finalize_recovery", error)
            # A release, an adoption or a cancelled cycle that cannot be proved.
            quarantine_scope(writer, scope, value["owner_run_id"], code, clock=clock)
            logging.getLogger("quotemem.store.writer").warning(
                "Cold finalize recovery refused: %s", canonical_json(reason.to_dict()),
            )
    finally:
        writer._finalize_boot_recovery_open = False
