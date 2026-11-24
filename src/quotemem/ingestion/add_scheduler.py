"""Schedule Add owners: passes, heartbeats, backoff and the synchronous Add wait.

The Writer retains durable owner cycles and issues their only execution tickets.
This controller owns threads and bounded work descriptors, never storage
authority. Construction is inert; start finishes saved-owner recovery before the
runtime may activate. An Add returns only after its owner released;
Search never waits for an owner. A request's cancellation ends only
that request's wait: a shared owner continues under its own worker until it
actually joins.

Three background loops run here. The discovery loop recovers saved owners,
orders accepted queue heads without a receiver and admits an idle owner for a
scope's unfinalized prefix. The heartbeat loop renews live leases every
``HEARTBEAT_INTERVAL_MS``, one renewal per Writer-mutex acquisition, and
settles expired leases after the renewals. The pool's workers run each
owner invocation through the ``OwnerExecutor`` and classify how it ended:
a durable stop retires the owner before bounded recovery, a retryable
refusal backs off within the owner's deadline, an integrity failure
quarantines the scope and a runtime failure withdraws the runtime.
"""

from __future__ import annotations

import logging
import random
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, Never, Protocol

from quotemem.api.bench import SearchDelivery, SearchRequest
from quotemem.api.bench_recovery import (
    RecoveryOperations,
    RecoveryRequestV1,
    RecoveryResponseV1,
    ResumeRequestV1,
    ResumeResponseV1,
)
from quotemem.api.bench_service import SearchOperation
from quotemem.api.completion_recovery import (
    CompletedAddV1,
    CompletionRecoveryRequestV2,
    CompletionRecoveryRequestV3,
    CompletionRecoveryResponseV2,
    CompletionRecoveryResponseV3,
    CompletionResumeRequestV2,
    CompletionResumeRequestV3,
    QuarantineSkipRequestV1,
    QuarantineSkipResponseV1,
)
from quotemem.core.add_outcome import AddOutcomeV1
from quotemem.core.bootstrap import InitializedWorkspace
from quotemem.core.controls import (
    ControlRowV1,
    FinalizeBudgetRefusalV4,
    FinalizeBudgetV2,
    FinalizedValueV1,
    HeldFinalizeLeaseV1,
    PreparingFinalizeLeaseV1,
    ReleasedFinalizeLeaseV1,
    RequestCompletionPolicyV2,
    ScopeRegistrationV1,
)
from quotemem.core.errors import QuoteMemError, RecoveryInProgress
from quotemem.core.failure_reason import FailureReason, describe, failure_reason
from quotemem.core.identity import control_id, scope_id
from quotemem.core.provider_budget import CallAllowanceEnded
from quotemem.core.provider_diagnostics import access_refused
from quotemem.core.query import ScopeV1
from quotemem.core.request_budget import (
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
    MAX_AUTOMATIC_RESUMES,
    RequestBudget,
    request_budget,
)
from quotemem.core.run_inputs import MessagesRequestPayloadV1
from quotemem.core.timing import FinalizeCompletion
from quotemem.ingestion.add_owner import OwnerExecutor, OwnerStalled, OwnerStopped, StopCause
from quotemem.ingestion.authorization import check_add_grant
from quotemem.ingestion.boundary import AddRequestV1
from quotemem.ingestion.segment import message_chunk_index
from quotemem.ingestion.service import (
    AddReceipt,
    AddService,
    current_completion_wait_deadline,
    reconcile_add_admission,
)
from quotemem.providers.auth import AuthRequest, PrincipalGrant
from quotemem.service.auth import BenchExactUserAuth
from quotemem.service.bench_workers import (
    DEFAULT_BENCH_WORK_CAPACITY,
    BenchWork,
    BenchWorkOutcome,
    BenchWorkPool,
    RetiredOwnerJoined,
    WorkCancellation,
)
from quotemem.service.request_measurement import (
    MeasurementCollector,
    observe_add_completion,
    observe_add_result,
)
from quotemem.service.resource_observer import ResourceCycle, ResourceObserver
from quotemem.store.add_outcome import read_add_outcome, read_completed_add
from quotemem.store.domain import DOMAIN_ADAPTERS
from quotemem.store.failures import is_runtime_failure
from quotemem.store.finalize_privacy import held_ticket, privacy_rebase_scope
from quotemem.store.finalize_ticket import FinalizeLeaseTicket
from quotemem.store.indexed import IndexedReader
from quotemem.store.owner_lease import bind_finalize_pool, pause_finalize, resume_finalize
from quotemem.store.owner_pause import paused_index
from quotemem.store.quarantine_skip import skip_quarantined_cycle
from quotemem.store.read_model import StoreReadModel
from quotemem.store.request_completion import (
    completed_prefix,
    enable_completion,
    exhaust_budget,
    failure_cause,
    read_budget,
    resume_budget,
)

SCOPED_REFUSALS = frozenset({
    "index_rebuilding", "scope_denied", "erasure_in_progress", "lease_lost",
    "scope_quarantined", "authorization_expired",
})
"""The scoped refusals: they leave the owner to the next pass, never fail it."""

INTEGRITY_FAILURES = frozenset({
    "binding_mismatch", "parity_mismatch", "incomplete_coverage", "immutable_field",
    "reference_missing", "reference_kind_mismatch", "reference_cycle", "tenant_mismatch",
    "schema_invalid", "schema_mismatch", "batch_plan_conflict", "artifact_digest_collision",
    "invalid_write_mode", "invalid_run_transition", "run_terminal", "operation_conflict",
    "internal_error", "storage_conflict",
})
"""Integrity or binding violations of an executing owner: only these quarantine its scope.

Item dispositions are per-item outcomes that never leave the owner. Any
other refusal leaves the owner to a later invocation within its own deadline;
exhausting that deadline is the owner's settled failure, never a quarantine. A
non-QuoteMem exception is an unknown defect and is isolated as ``internal_error``. A
``storage_conflict`` (a constraint refused a transaction that wrote nothing,
``quotemem.store.failures``) that no component parked is quarantined the same way:
its bounded automatic resumes replay it, and every other scope keeps running.
"""

DISCOVERY_SKIP = SCOPED_REFUSALS | frozenset({
    "unsupported_variant", "reception_stale", "finalize_pending", "repair_required",
})
"""Refusals meaning a scope changed, has an owner or needs explicit completion repair.

A background pass reads each scope without holding it; an Add's owner, an
erasure or a reception can take the scope between two of its reads. The pass
skips that scope and its owner, explicit repair or the next pass carries it. Any other
scoped failure needs a durable quarantine before the pass continues.
"""

_BUDGET_REFUSALS = frozenset({"delivery_busy", "index_rebuilding", "budget_exceeded"})
"""Refusals that may mean the owner's own allowance ran out rather than a retryable fault."""

DISCOVERY_INTERVAL_SECONDS = 0.5
"""Seconds between two complete discovery passes, measured from the previous pass's end."""

RECEPTION_EXPIRY_INTERVAL_SECONDS = 60.0
"""Seconds between cleanup passes for expired roots with no accepted source."""

FINALIZE_LEASE_LIFETIME_MS = 60_000
"""Heartbeat age at which the Writer refuses an owner's lease as expired."""

HEARTBEAT_INTERVAL_MS = FINALIZE_LEASE_LIFETIME_MS // 10
"""Milliseconds between two heartbeat passes: 6,000, one tenth of the lifetime.

A pass renews every live lease at least ``HEARTBEAT_RENEWAL_AGE_MS`` old, so a
live lease is renewed at least once per renewal age plus two intervals, well
inside the lifetime.
"""

HEARTBEAT_INTERVAL_SECONDS = HEARTBEAT_INTERVAL_MS / 1000

HEARTBEAT_RENEWAL_BATCH = 1
"""Due leases one Writer-mutex acquisition renews: exactly one.

Batching renewals held the mutex long enough to starve sections with a
deadline, whose snapshot consumers then refused with ``index_rebuilding``.
"""

HEARTBEAT_RENEWAL_AGE_MS = FINALIZE_LEASE_LIFETIME_MS // 3
"""Age from which a pass renews a live lease: 20,000 ms.

A younger lease was renewed by an owner transition or a recent pass; renewing
it again only takes the Writer mutex.
"""

IDLE_INITIAL_SECONDS = 120.0
"""Quiet time after which a scope's unfinalized prefix gets an idle owner."""

IDLE_MAX_SECONDS = 1800.0
"""Longest quiet time an idle owner waits for; each completed owner doubles it."""

EXPIRED_JOIN_SECONDS = 60.0
"""How long an expired owner's cancelled invocation may take to join before it is a fault."""

RECOVERY_JOIN_SECONDS = 2.0
"""Longest wait of a request for a concurrent automatic recovery to finish."""

ACCESS_RESUME_FIRST_SECONDS = 15
"""Wait after a key refusal (401/403/429) before the first automatic resume of its owner."""

ACCESS_RESUME_MAX_SECONDS = 60
"""Longest wait between key-refusal resumes: AMLB's longest honored Retry-After.

The wait doubles from ``ACCESS_RESUME_FIRST_SECONDS`` up to this cap, so the
32 automatic resumes span about 31 minutes, longer than a key proxy's
30-minute pause of a refused key; a rotated key answers the first resume.
"""


def access_resume_delay_ms(recovery_count: int) -> int:
    """The quiet time between an owner's key-refusal pause and its next automatic resume.

    Args:
        recovery_count: The owner's counted resumes so far.

    Returns:
        int: Milliseconds; resuming sooner only spends the owner's bounded resumes
            on a key that is still refused.
    """
    seconds = ACCESS_RESUME_FIRST_SECONDS * 2 ** min(recovery_count, 8)
    return min(ACCESS_RESUME_MAX_SECONDS, seconds) * 1000

_PAGE = 128


def _trace(reason: FailureReason) -> str:
    """Render a failure's whole retained causal trace as ``file:line:function`` chains.

    Every part is code-owned (file names, lines, functions, exception types and
    codes), so the rendering stays content-free like `describe`.
    """
    return " <- ".join(
        f"{entry.relation} {entry.code or entry.exception_type} ["
        + " > ".join(f"{name}:{line}:{function}" for name, line, function in entry.frames)
        + "]" for entry in reason.trace
    )


@dataclass
class _ScopeActivity:
    """Content-free live scheduling state of one scope; no cold reconstruction is implied."""

    last_activity: float
    idle_seconds: float = IDLE_INITIAL_SECONDS


@dataclass(slots=True)
class _HeartbeatLease:
    """One page entry carried from its borrowed read to its renewal.

    ``borrowed`` is the value the page read; ``active`` is the value the
    unexcluded confirmation read returned and, once a renewal ran, the value that
    renewal proved under the mutex. ``token`` is the storage epoch captured
    before those reads, which classifies a later refusal.
    """

    selected: ControlRowV1
    scope: ScopeV1
    borrowed: PreparingFinalizeLeaseV1 | HeldFinalizeLeaseV1
    active: PreparingFinalizeLeaseV1 | HeldFinalizeLeaseV1
    token: int


@dataclass(frozen=True)
class RetryBackoff:
    """Bounded exponential delays with jitter between one owner's retryable refusals.

    The n-th consecutive refusal (from 0) waits a uniformly random time in
    [b/2, b] with b = min(``cap_seconds``, ``base_seconds`` * 2**n): the fixed half
    doubles the wait up to the cap, and the random half keeps owners that failed
    together from retrying together. ``jitter`` returns uniform values in [0, 1).
    """

    base_seconds: float = 0.5
    cap_seconds: float = 16.0
    jitter: Callable[[], float] = random.random

    def delay(self, retry: int) -> float:
        """Return the seconds to wait before retry ``retry`` (from 0) of one owner."""
        ceiling = min(self.cap_seconds, self.base_seconds * 2.0 ** min(retry, 32))
        return ceiling * (1.0 + self.jitter()) / 2.0


class OwnerScheduling:
    """The pool fences explicit recovery uses; it grants no durable authority by itself.

    Explicit recovery retires the stopped owner, proves its callbacks
    joined, commits its durable resume through the sole Writer and only then
    releases the submission fence and resubmits the same owner.
    """

    def __init__(self, controller: AddController) -> None:
        self._controller = controller

    def retire(self, work: BenchWork) -> bool:
        """Fence the owner's submission and signal its active callback."""
        return self._controller.pool.retire_owner(work)

    def issue_join_proof(self, work: BenchWork) -> RetiredOwnerJoined:
        """Issue proof that the retired owner's callbacks actually joined."""
        return self._controller.pool.retired_owner_join(work)

    def check_join_proof(self, joined: RetiredOwnerJoined) -> None:
        """Validate an issued join proof against the current pool inventory."""
        self._controller.pool.validate_retired_owner_join(joined)

    def resume(self, joined: RetiredOwnerJoined) -> None:
        """Release the fence after the Writer committed the durable resume, then resubmit."""
        self._controller.pool.resume_retired_owner(joined)
        # A refused submission costs only this promotion; the resumed owner is
        # no longer stopped, so the next complete pass re-admits it.
        self._controller.pool.submit(joined.work, urgent=True)
        self._controller.wake_waiters()


class OwnerRecovery(RecoveryOperations, Protocol):
    """Explicit recovery of a stopped owner: the proofs over this scheduler's fences.

    ``inspect_recovery``/``resume_recovery`` observe and resume a provider pause
    (``ProviderFailureV1``); ``resume_refused`` resumes a refused
    budget on the same owner and frozen prefix.
    """

    def resume_refused(self, scope: ScopeV1, request: CompletionResumeRequestV3) -> None:
        """Replace the observed refusal through the owner's explicit resume."""

    def repair_quarantine(self, scope: ScopeV1, request: CompletionResumeRequestV3) -> None:
        """Lift the observed quarantine through its owner's explicit resume."""


type RecoveryFactory = Callable[[OwnerScheduling], OwnerRecovery]
"""Build the explicit-recovery operations over this controller's pool fences."""


class AddController:
    """Own the Add owners' scheduling and the synchronous Add wait for one workspace.

    Construction is inert. ``start`` completes saved-owner recovery before the
    runtime may activate. ``completion_work_budget_ms`` selects the durable
    allowance of a fresh scope; omission uses 1,740 seconds and preserves an
    installed policy. Every owner this controller admits or rejoins has that
    allowance: an Add or the idle pass, whichever reaches a fresh scope first,
    installs the policy.

    Args:
        ready: The initialized workspace whose Writer owns every transition.
        publisher: The Writer's registered read model.
        auth: The bench's exact-user authorization.
        owner: Runs one owner invocation.
        search: The Search use case; it never waits for an owner.
        recovery: Builds explicit recovery over this controller's pool fences.
        workers: Total pool workers, reserved ones included.
        reserved: Workers that only take owners a request waits on.
        per_user: Active owner invocations per scope; also bounds each owner's
            unit calls and judge requests in flight.
        pending: Bound on queued descriptors.
        clock: Trusted millisecond action clock.
        measurement: Optional bounded observer.
        completion_work_budget_ms: Optional fresh-scope owner allowance.
    """

    def __init__(
        self, ready: InitializedWorkspace, publisher: StoreReadModel, auth: BenchExactUserAuth, *,
        owner: OwnerExecutor, search: SearchOperation, recovery: RecoveryFactory,
        workers: int = DEFAULT_BENCH_WORK_CAPACITY.workers,
        reserved: int = DEFAULT_BENCH_WORK_CAPACITY.reserved,
        per_user: int = DEFAULT_BENCH_WORK_CAPACITY.per_user,
        pending: int = DEFAULT_BENCH_WORK_CAPACITY.pending,
        clock: Callable[[], int] = lambda: time.time_ns() // 1_000_000,
        measurement: MeasurementCollector | None = None,
        completion_work_budget_ms: int | None = None,
    ) -> None:
        self._completion_work_budget_ms = (
            None if completion_work_budget_ms is None else RequestCompletionPolicyV2(
                work_budget_ms=completion_work_budget_ms,
            ).work_budget_ms
        )
        self._completion_policy = (
            RequestCompletionPolicyV2() if completion_work_budget_ms is None
            else RequestCompletionPolicyV2(work_budget_ms=completion_work_budget_ms)
        )
        self._ready, self._publisher, self._auth = ready, publisher, auth
        self._owner, self._search, self._clock = owner, search, clock
        self._per_user = per_user
        self._measurement = measurement
        self._condition = threading.Condition()
        # Serialize automatic and explicit recovery through their compare/join/
        # commit/resubmit sequence. A stale refusal must never retire a newly
        # resumed invocation before its version check rejects that selection.
        self._recovery_lock = threading.RLock()
        self._reception_changed = threading.Event()
        ready.writer.add_admission.set_change_callback(self._reception_changed.set)
        self._reception: AddService | None = None
        self._progress = 0
        self._activity: dict[str, _ScopeActivity] = {}
        # Bounded optimizations; a miss always revalidates durable state.
        self._blocked_cycles: OrderedDict[str, None] = OrderedDict()
        self._cache_limit = pending
        self._owner_budgets: OrderedDict[tuple[str, int], RequestBudget] = OrderedDict()
        self._retry_backoff = RetryBackoff()
        self._retries: OrderedDict[str, int] = OrderedDict()
        self._failure = threading.Event()
        self._failure_reason: FailureReason | None = None
        self._refusal_reason: FailureReason | None = None
        self._on_failure: Callable[[], None] = lambda: None
        self._started = self._withdrawn = self._draining = False
        self._accepting = self._starting = self._pool_sealed = False
        self._drain_recovery: Iterator[bool] | None = None
        self._drain_rejected = False
        self._expiry_due = 0.0
        self._threads: list[threading.Thread] = []
        self._heartbeat_stop = False
        self._reader = IndexedReader(
            ready.runtime.workspace, ready.writer.entities, tenant_id=ready.writer.tenant_id,
        )
        self._pool = BenchWorkPool(
            self._execute, on_failure=self._fail, workers=workers, reserved=reserved,
            per_user=per_user, pending=pending,
        )
        bind_finalize_pool(ready.writer, self._pool)
        self._recovery = recovery(OwnerScheduling(self))

    @property
    def pool(self) -> BenchWorkPool:
        """The sole worker pool this controller schedules owners on."""
        return self._pool

    @property
    def has_failed(self) -> bool:
        """Return the latched content-free failure fact, including after withdrawal."""
        return self._failure.is_set()

    @property
    def failure_reason(self) -> FailureReason | None:
        """Return the first content-free background failure, its own or the pool's."""
        with self._condition:
            own = self._failure_reason
        return own if own is not None else self._pool.failure_reason

    @property
    def refusal_reason(self) -> FailureReason | None:
        """Return the first classified background refusal this controller tolerated.

        A background pass reads scopes and leases it does not hold, so one of them
        can be owned, erased or replaced between two of its reads. Such a refusal
        belongs to that scope, not to the runtime: it is retained for diagnosis
        and never sets ``failed``.
        """
        with self._condition:
            return self._refusal_reason

    def _record_failure(self, site: str, error: BaseException | None) -> None:
        with self._condition:
            if self._failure_reason is None:
                self._failure_reason = failure_reason(site, error)

    def _record_refusal(self, site: str, error: BaseException) -> None:
        with self._condition:
            if self._refusal_reason is None:
                self._refusal_reason = failure_reason(site, error)

    def _resources(self) -> ResourceObserver | None:
        return None if self._measurement is None else self._measurement.resources

    def _grant(self, scope: ScopeV1) -> PrincipalGrant:
        """Resolve a fresh grant of one scope, outside QuoteMem locks."""
        return self._auth.resolve(AuthRequest(requested_scope=scope, now_ms=self._clock()))

    def _observe_existing_cycle(self, ticket: FinalizeLeaseTicket) -> None:
        """Retain missing startup history as incomplete, never relabel a cycle."""
        resources = self._resources()
        if resources is None:
            return
        try:
            resources.ensure_cycle(ResourceCycle(
                ticket.scope_id, ticket.owner_run_id, ticket.lease_generation,
            ))
        except Exception:
            try:
                resources.incomplete("cycle")
            except Exception:
                pass

    def set_failure_callback(self, callback: Callable[[], None]) -> None:
        """Bind the runtime's admission withdrawal before starting any worker."""
        with self._condition:
            if self._started:
                raise RuntimeError("Bind failure withdrawal before starting the controller")
            self._on_failure = callback

    def _fail(self) -> None:
        with self._condition:
            notify = not self._failure.is_set()
            self._failure.set()
            self._withdrawn = True
            self._heartbeat_stop = True
            self._progress += 1
            self._condition.notify_all()
        self._reception_changed.set()
        if notify:
            # The pool retires only metadata and keeps active callbacks until
            # they actually join. Its callback may reenter this latched method.
            self._pool.fail()
            self._on_failure()

    def wake_waiters(self) -> None:
        """Wake every waiter on this controller's condition; it carries no authority."""
        with self._condition:
            self._progress += 1
            self._condition.notify_all()
        self._ready.writer.add_admission.wake()

    def _quarantine_error(
        self, scope: ScopeV1, error: BaseException, *, owner_run_id: str | None = None,
    ) -> None:
        """Persist a scoped refusal after its operation unwinds and stop the scope's work.

        Storage and process failures propagate. A locator only selects the owner;
        the Writer re-proves it and records the receipt atomically. Failing to
        establish that receipt is not successful isolation.
        """
        if is_runtime_failure(error):
            raise error
        try:
            writer = self._ready.writer
            raw = writer.get_control(f"lease:finalize:{scope.scope_id}")
            value = None if raw is None else ControlRowV1.model_validate(raw).value
            if isinstance(value, PreparingFinalizeLeaseV1 | HeldFinalizeLeaseV1):
                owner_run_id = value.owner_run_id
            elif owner_run_id is None:
                owner = self._reader.latest_scope_owner(scope)
                owner_run_id = None if owner is None else owner["object_id"]
            if owner_run_id is None:
                raise QuoteMemError("service_unhealthy",
                                    "Scoped failure has no provable durable owner")
            writer.quarantine_scope(
                scope, owner_run_id,
                error.code if isinstance(error, QuoteMemError) else "internal_error",
                clock=self._clock,
            )
            self._pool.cancel_scope(scope.scope_id)
            # The quarantined scope's failure must not keep every other scope's
            # Search and health refused behind a closed read model.
            self._restore_reads()
        except Exception as settlement_error:
            raise QuoteMemError(
                "service_unhealthy", "Scope failure could not be durably isolated",
            ) from settlement_error
        self._record_refusal("scope", error)
        reason = failure_reason("scope", error)
        logging.getLogger(__name__).warning(
            "Scope operation refused%s; trace: %s", describe(reason), _trace(reason),
        )

    # Owner invocations ---------------------------------------------------------------------

    def _execute(self, work: BenchWork, cancellation: WorkCancellation) -> BenchWorkOutcome:
        """Return one typed disposition; only explicit runtime failures withdraw all users.

        An owner whose durable allowance ended is retired as a settled failure,
        whichever check noticed first: the cooperative budget, the Writer's
        commit-time ``require_owner_budget`` or a retirement's cancellation. Such a
        refusal is exhaustion, never a scope fault, so it neither quarantines the
        scope nor withdraws the runtime. An extraction call whose own window ended
        (``CallAllowanceEnded``) ends the owner's allowance at once: no later
        invocation of the attempt can send that call, so retrying it until the
        owner's deadline would only idle the waiting Add; the counted
        resume follows at once. Any other refusal that is no integrity
        failure holds the invocation for the owner's next backoff delay
        (``_back_off``) before it returns for a retry.
        """
        if work.kind == "reception":
            return self._execute_reception(work, cancellation)
        scope = ScopeV1(scope_kind="user", scope_id=work.scope_id)
        budget: RequestBudget | None = None
        deadline_at: int | None = None
        try:
            selected = read_budget(self._ready.writer, scope, work.run_id)
            if selected is not None and isinstance(selected.value, FinalizeBudgetRefusalV4):
                self._pool.retire_owner(work)
                return BenchWorkOutcome("settled_failure")
            if selected is not None and isinstance(selected.value, FinalizeBudgetV2):
                deadline_at = selected.value.deadline_at
                if (
                    selected.value.state == "exhausted"
                    or self._clock() >= selected.value.deadline_at
                ) and self._retire_exhausted_budget(scope, work.run_id):
                    return BenchWorkOutcome("settled_failure")
                budget = self._owner_budget(work.run_id, selected.version, deadline_at)
            # Worker ownership is independent of every HTTP waiter. Repeated
            # invocations keep the same persisted allowance and clock mapping.
            with request_budget(budget):
                if budget is not None:
                    budget.check()
                self._execute_owned(work, cancellation)
            with self._condition:
                self._retries.pop(work.run_id, None)
            current = self._ready.writer.get_row("run", work.run_id)
            if current is None:
                raise QuoteMemError("reference_missing", "Drained work lost its durable owner")
            DOMAIN_ADAPTERS["run"].validate_python(current)
            if (current["scope_kind"], current["scope_id"]) != ("user", work.scope_id):
                raise QuoteMemError("binding_mismatch", "Drained work changed its exact scope")
            if current["status"] == "succeeded":
                return BenchWorkOutcome("ok")
            if budget is not None and budget.cancelled:
                return self._settle_ended_allowance(scope, work.run_id)
            if current["status"] in {"failed", "manual_attention"} or (
                self._ready.writer.is_scope_quarantined(scope.scope_id)
            ):
                return BenchWorkOutcome("settled_failure")
            return BenchWorkOutcome("retry_later")
        except BaseException as error:
            if is_runtime_failure(error):
                return BenchWorkOutcome("fatal", failure_reason("execute", error))
            if self._cancelled(work.run_id):
                # A forget cancelled this owner while it ran: its
                # refused action is that cancellation, never a scope fault.
                return BenchWorkOutcome("settled_failure", failure_reason("execute", error))
            if budget is not None and isinstance(error, QuoteMemError) and (
                error.code in _BUDGET_REFUSALS
            ):
                try:
                    if isinstance(error, CallAllowanceEnded) and self._exhaust_stalled_owner(
                        scope, work.run_id,
                    ):
                        return BenchWorkOutcome("settled_failure")
                    if budget.cancelled or self._is_allowance_ended(scope, work.run_id):
                        return self._settle_ended_allowance(scope, work.run_id)
                except BaseException as settlement_error:
                    return BenchWorkOutcome("fatal", failure_reason("budget", settlement_error))
            if isinstance(error, QuoteMemError) and error.code not in INTEGRITY_FAILURES:
                self._record_refusal("scope", error)
                self._back_off(work.run_id, deadline_at, cancellation)
                return BenchWorkOutcome("retry_later", failure_reason("execute", error))
            try:
                self._quarantine_error(scope, error, owner_run_id=work.run_id)
            except BaseException as settlement_error:
                return BenchWorkOutcome("fatal", failure_reason("quarantine", settlement_error))
            return BenchWorkOutcome("settled_failure", failure_reason("execute", error))

    def _execute_reception(
        self, work: BenchWork, cancellation: WorkCancellation,
    ) -> BenchWorkOutcome:
        """Index only the current saved head; client cancellation cannot own this work."""
        scope = ScopeV1(scope_kind="user", scope_id=work.scope_id)
        budget = RequestBudget(DEFAULT_REQUEST_TIMEOUT_SECONDS)
        try:
            with cancellation.on_cancel(budget.cancel), request_budget(budget):
                budget.check()
                if self._scope_unavailable(scope):
                    return BenchWorkOutcome("retry_later")
                with self._condition:
                    if self._reception is None:
                        self._reception = AddService(
                            self._ready.writer, self._ready.providers.embedding,
                            self._auth, self._publisher, profile="bench", clock=self._clock,
                        )
                        self._reception.requires_completion = True
                    reception = self._reception
                reception.resume_received(work.run_id)
                self._try_reserved_finalize(scope)
            self._reception_changed.set()
            return BenchWorkOutcome("ok")
        except BaseException as error:
            if is_runtime_failure(error):
                return BenchWorkOutcome("fatal", failure_reason("reception", error))
            # Any other refusal, an integrity failure or a defect included, belongs
            # to this one scope's saved head: it is retried at the owner backoff's
            # spaced times, never by withdrawing every other user of the workspace.
            # The retry's ``resume_received`` rebuilds a read model its failed save
            # left closed before it reads anything.
            self._record_refusal("reception", error)
            self._back_off(work.run_id, None, cancellation)
            return BenchWorkOutcome("retry_later", failure_reason("reception", error))

    def _restore_reads(self) -> None:
        """Rebuild the shared read model after a quarantined scope failure failed it closed.

        A scoped write that reached storage and then failed leaves the read model
        closed for every scope (``Writer.write_scope``): every Search and the
        health check would refuse until something rebuilt it. Once the failure
        is durably quarantined, its worker rebuilds it here, outside the Writer
        mutex and every read callback, from durable state. Ordinary retryable
        refusals never rebuild: a privacy operation owns its own closed window.
        A healthy read model costs one generation read.

        Raises:
            QuoteMemError: ``service_unhealthy`` when the read model is closed, was
                withdrawn by a repeated publication failure or cannot be rebuilt:
                that is a runtime failure.
        """
        self._publisher.recover_if_needed()

    def _cancelled(self, owner_run_id: str) -> bool:
        """Whether a forget cancelled the owner: its Run failed ``erase_fenced``."""
        owner = self._ready.writer.get_row("run", owner_run_id)
        return owner is not None and (owner["status"], owner["failure_code"]) == (
            "failed", "erase_fenced",
        )

    def _owner_budget(self, owner_run_id: str, version: int, deadline_at: int) -> RequestBudget:
        """Map one durable allowance version to the cooperative budget of its invocations."""
        key = (owner_run_id, version)
        with self._condition:
            budget = self._owner_budgets.get(key)
            if budget is None:
                remaining = (deadline_at - self._clock()) / 1000
                if remaining <= 0:
                    budget = RequestBudget(0.000001)
                    budget.cancel()
                else:
                    budget = RequestBudget(remaining)
                self._owner_budgets[key] = budget
                if len(self._owner_budgets) > self._cache_limit:
                    self._owner_budgets.popitem(last=False)
            return budget

    def _execute_owned(self, work: BenchWork, cancellation: WorkCancellation) -> None:
        ready = self._ready
        scope = ScopeV1(scope_kind="user", scope_id=work.scope_id)
        if self._scope_unavailable(scope):
            return
        current = ready.writer.get_row("run", work.run_id)
        if current is None:
            raise QuoteMemError("reference_missing", "Scheduled durable owner is missing")
        DOMAIN_ADAPTERS["run"].validate_python(current)
        if current["scope_id"] != work.scope_id or current["scope_kind"] != "user":
            raise QuoteMemError("binding_mismatch", "Scheduled owner changed its exact user")
        if current["status"] in {"succeeded", "failed"}:
            # A previously selected page can outlive its completed invocation, or
            # a forget cancelled the owner. This return grants no
            # publication or completion fact.
            return
        if self._retire_stopped_owner(work):
            return
        ticket: FinalizeLeaseTicket | None = None
        try:
            ticket = held_ticket(
                ready.writer, self._publisher, work.run_id, grant=self._grant,
                cancellation=cancellation, clock=self._clock,
            )
            self._observe_existing_cycle(ticket)
            ended = self._owner.run(
                work.run_id, cancellation=cancellation, concurrency=self._per_user,
            )
            if isinstance(ended, OwnerStopped):
                if ended.owner_run_id != work.run_id:
                    raise QuoteMemError("binding_mismatch", "Owner stop changed its actual owner")
                if not self._retire_stopped_owner(work, ended.cause):
                    raise QuoteMemError(
                        "reception_stale", "Stopped owner changed before retirement",
                    )
                return
            if isinstance(ended, OwnerStalled):
                if ended.owner_run_id != work.run_id:
                    raise QuoteMemError("binding_mismatch", "Owner stall changed its actual owner")
                # ``_execute`` ends the owner's allowance.
                raise CallAllowanceEnded
        except QuoteMemError as error:
            if is_runtime_failure(error):
                raise
            if error.code in {"erase_fenced", "consent_fenced"} and self._privacy_cancelled(work):
                # A privacy change cancelled the owner before its first review:
                # an idle pass admits the owner that reprocesses
                # its requests once the cancelled cycle proves settled.
                return
            if error.code in {"index_rebuilding", "lease_lost"} and (
                self._pool.paused_invocation(work, cancellation)
                or error.code == "lease_lost" and self._pause_expired_finalize(work)
            ):
                # The pool still owns this invocation. Only its actual return
                # removes it and lets the heartbeat settle the pause.
                return
            if cancellation.cancelled and error.code == "index_rebuilding":
                # The pool cancelled this invocation (a retirement, a failed pool
                # or an abandoned drain) and the owner's clock refused its next
                # action: that refusal is the cancellation itself. The durable
                # owner keeps its lease; the heartbeat or a later pass carries it.
                return
            if self._expected_scope_refusal(scope, error):
                return
            if error.code == "authorization_expired":
                # The grant this invocation resolved outside every lock aged past
                # its lifetime. Ending the invocation is the renewal: the next one
                # resolves a fresh grant and re-proves every fence.
                return
            if ticket is None or not self._preserve_scoped_failure(ticket):
                raise
            return
        if self._measurement is not None:
            if (ended.scope_id, ended.owner_run_id, ended.finalize_cycle) != (
                work.scope_id, work.run_id, ticket.lease_generation,
            ):
                self._measurement.finalize_incomplete()
            else:
                self._measurement.finalize_completed(ended)
        self._on_owner_released(work, ended)

    def _on_owner_released(self, work: BenchWork, ended: FinalizeCompletion) -> None:
        if (ended.scope_id, ended.owner_run_id) != (work.scope_id, work.run_id):
            raise QuoteMemError("binding_mismatch", "Owner release changed its actual owner")
        with self._condition:
            self._progress += 1
            self._blocked_cycles.pop(work.run_id, None)
            state = self._activity.get(work.scope_id)
            if state is not None:
                state.last_activity = time.monotonic()
                state.idle_seconds = min(IDLE_MAX_SECONDS, state.idle_seconds * 2)
            self._condition.notify_all()

    def _back_off(
        self, run_id: str, deadline_at: int | None, cancellation: WorkCancellation,
    ) -> None:
        """Hold a retryable refusal's invocation for the owner's next backoff delay.

        The invocation stays active meanwhile, so neither a waiting Add nor
        discovery starts the owner again: an owner that keeps refusing is retried
        at exponentially spaced times (``RetryBackoff``), not at once. The delay
        never outlasts the owner's durable deadline (``deadline_at``), after which
        the next invocation retires the owner as exhausted. Cancellation,
        withdrawal, drain and a failed runtime end it early. An invocation that
        returns without a refusal forgets the owner's count.
        """
        with self._condition:
            retry = self._retries.pop(run_id, -1) + 1
            self._retries[run_id] = retry
            if len(self._retries) > self._cache_limit:
                self._retries.popitem(last=False)
        delay = self._retry_backoff.delay(retry)
        if deadline_at is not None:
            delay = min(delay, max(0.0, (deadline_at - self._clock()) / 1000))
        until = time.monotonic() + delay
        with cancellation.on_cancel(self.wake_waiters), self._condition:
            while not (
                cancellation.cancelled or self._withdrawn or self._draining
                or self._failure.is_set()
            ):
                remaining = until - time.monotonic()
                if remaining <= 0:
                    return
                self._condition.wait(remaining)

    # Allowance -----------------------------------------------------------------------------

    def _is_allowance_ended(self, scope: ScopeV1, owner_run_id: str) -> bool:
        """Read whether the owner's durable allowance is refused, exhausted or past its deadline."""
        selected = read_budget(self._ready.writer, scope, owner_run_id)
        if selected is None:
            return False
        value = selected.value
        return isinstance(value, FinalizeBudgetRefusalV4) or isinstance(
            value, FinalizeBudgetV2,
        ) and (value.state == "exhausted" or self._clock() >= value.deadline_at)

    def _settle_ended_allowance(self, scope: ScopeV1, owner_run_id: str) -> BenchWorkOutcome:
        """Settle an invocation whose cooperative budget or durable allowance ran out.

        A durably ended allowance retires the owner; its callbacks see their
        cancellation and their late results write nothing. A cooperative budget
        that ran out before the durable wall-clock deadline (the two clocks can
        drift apart) is forgotten, so the next invocation derives a fresh one from
        the same unchanged deadline instead of retiring a live owner.
        """
        selected = read_budget(self._ready.writer, scope, owner_run_id)
        if selected is not None and isinstance(selected.value, FinalizeBudgetRefusalV4):
            self._pool.retire_owner(BenchWork(scope.scope_id, owner_run_id, "finalize"))
            return BenchWorkOutcome("settled_failure")
        if self._retire_exhausted_budget(scope, owner_run_id):
            return BenchWorkOutcome("settled_failure")
        with self._condition:
            for key in [key for key in self._owner_budgets if key[0] == owner_run_id]:
                del self._owner_budgets[key]
        return BenchWorkOutcome("retry_later")

    def _retire_exhausted_budget(self, scope: ScopeV1, owner_run_id: str) -> bool:
        """Retire an owner whose allowance ran out, before admission or automatic recovery.

        An exhausted owner can keep an expired lease across a cold reopen;
        requiring a live admission first would keep its callbacks from ever
        being retired. A retired descriptor is a classified refusal, not a
        failed submission: only proved recovery grants another allowance.
        """
        writer = self._ready.writer
        # Heartbeat pages stay borrowed reads. Only a possibly ended V2 projection
        # warrants taking the Writer mutex to prove its owner.
        raw = writer.get_control(f"finalize_budget:{scope.scope_id}")
        candidate = None if raw is None else ControlRowV1.model_validate(raw).value
        if not isinstance(candidate, FinalizeBudgetV2) or candidate.owner_run_id != owner_run_id:
            return False
        if candidate.state == "active" and self._clock() < candidate.deadline_at:
            return False
        with writer.mutex:
            selected = read_budget(writer, scope, owner_run_id)
            if selected is None or not isinstance(selected.value, FinalizeBudgetV2) or (
                selected.value.state == "active" and self._clock() < selected.value.deadline_at
            ):
                return False
            owner = writer.get_row("run", owner_run_id)
            if owner is not None and owner["status"] == "succeeded":
                return False
            # Match the durable state and the pool fence under the same exclusion
            # as explicit resume. Retirement signals callbacks but never joins them.
            exhaust_budget(writer, scope, owner_run_id, clock=self._clock)
            self._pool.retire_owner(BenchWork(scope.scope_id, owner_run_id, "finalize"))
        self.wake_waiters()
        return True

    def _exhaust_stalled_owner(self, scope: ScopeV1, owner_run_id: str) -> bool:
        """End a live allowance the owner can no longer use, then retire it.

        Called when an extraction call's own window ended without a reply: no
        later invocation of the owner's attempt can send that call. The owner's
        allowance becomes exhausted before its deadline, under the same mutex and
        pool fence as ``_retire_exhausted_budget``, so automatic recovery resumes
        it as soon as its callbacks join instead of after the deadline.

        Returns:
            True when the owner's allowance is exhausted and the owner retired;
            False when it already succeeded, was refused or changed owner.
        """
        writer = self._ready.writer
        with writer.mutex:
            selected = read_budget(writer, scope, owner_run_id)
            if selected is None or not isinstance(selected.value, FinalizeBudgetV2):
                return False
            owner = writer.get_row("run", owner_run_id)
            if owner is None or owner["status"] == "succeeded":
                return False
            exhaust_budget(writer, scope, owner_run_id, clock=self._clock, early=True)
            self._pool.retire_owner(BenchWork(scope.scope_id, owner_run_id, "finalize"))
        self.wake_waiters()
        return True

    # Durable stops -------------------------------------------------------------------------

    def _read_stop_cause(self, owner_run_id: str) -> StopCause | None:
        """Read whether the owner stopped durably and why; None while it may run.

        A stopped owner left execution (``manual_attention``). ``provider``: it is
        paused on a provider failure. ``whole_output``: its budget is in the
        refused state. A running owner costs one Run read.
        """
        writer = self._ready.writer
        owner = writer.get_row("run", owner_run_id)
        if owner is None or owner["run_kind"] != "finalize" or (
            owner["status"] != "manual_attention"
        ):
            return None
        if owner["failure_code"] == "provider_error":
            return "provider"
        raw = writer.get_control(f"finalize_budget:{owner['scope_id']}")
        budget = None if raw is None else ControlRowV1.model_validate(raw).value
        if isinstance(budget, FinalizeBudgetRefusalV4) and budget.owner_run_id == owner_run_id:
            return "whole_output"
        return None

    def _retire_stopped_owner(self, work: BenchWork, cause: StopCause | None = None) -> bool:
        """Fence a durably stopped owner's submission until proved recovery.

        A borrowed read selects; the stop is then re-read under the Writer mutex,
        the same exclusion as the resume transaction, so a resumed owner is never
        fenced by its old stop. ``cause`` is what the invocation reported; a
        different durable stop is a stale report, and the caller re-reads.
        """
        found = self._read_stop_cause(work.run_id)
        if found is None or cause is not None and found != cause:
            return False
        writer = self._ready.writer
        with writer.mutex:
            found = self._read_stop_cause(work.run_id)
            if found is None or cause is not None and found != cause:
                return False
            # Retirement only notifies cancellation subscribers; it joins no
            # callback and takes no Writer lock.
            self._pool.retire_owner(work)
        self.wake_waiters()
        return True

    def _held_stop(self, scope: ScopeV1, owner_run_id: str) -> bool:
        """Leave a durably stopped owner's lease as it is: no renewal, no expiry settlement.

        A stopped owner keeps its lease until counted recovery renews it in the
        resume transaction.
        """
        return self._retire_stopped_owner(BenchWork(scope.scope_id, owner_run_id, "finalize"))

    # Synchronous Add -----------------------------------------------------------------------

    def _automatic_resume_owner(self, scope: ScopeV1, owner_run_id: str) -> bool:
        """Resume an eligible stopped owner through the existing recovery proofs.

        True means this owner may continue automatically, including a callback
        still joining or a concurrently changed selection. The HTTP caller must
        answer retryable delivery_busy, retaining the Add's original queue turn.
        False leaves a live owner alone or requires explicit repair: the shared
        resume bound, privacy, parity and corrupt recovery proofs never yield to
        automatic retries. This method never waits for an active owner to join.
        """
        if not self._recovery_lock.acquire(blocking=False):
            return True
        try:
            # HTTP recovery metadata remains within its request's cancellation
            # budget. The resumed pool invocation receives its own allowance.
            writer = self._ready.writer
            selected = read_budget(writer, scope, owner_run_id)
            if selected is None or not isinstance(
                selected.value, FinalizeBudgetV2 | FinalizeBudgetRefusalV4,
            ):
                return False
            value = selected.value
            owner = writer.get_row("run", owner_run_id)
            if owner is None or owner["status"] in {"succeeded", "failed"} or (
                owner["status"], owner["failure_code"],
            ) == ("manual_attention", "parity_mismatch"):
                return False
            quarantined = writer.is_scope_quarantined(scope.scope_id)
            provider_paused = (owner["status"], owner["failure_code"]) == (
                "manual_attention", "provider_error",
            )
            ended = isinstance(value, FinalizeBudgetRefusalV4) or (
                value.state == "exhausted" or self._clock() >= value.deadline_at
            )
            if not quarantined and not provider_paused and not ended:
                return False
            if value.recovery_count >= MAX_AUTOMATIC_RESUMES:
                return False
            if provider_paused and self._clock() < self._access_resume_at(
                owner, value.recovery_count,
            ):
                # A refused key is temporary backpressure: resuming now
                # would only spend a counted resume on the same refusal.
                return True
            if (value.erase_epoch, value.consent_epoch) != (
                writer.get_counter(f"erase_epoch:{scope.scope_id}"),
                writer.get_counter(f"consent_epoch:{scope.scope_id}"),
            ):
                return False
            if writer.get_control(f"tomb:{scope.scope_id}") is not None:
                return False
            if self._reader.scope_fences(scope.scope_id) or self._reader.pending_privacy(
                scope.scope_id, limit=1,
            ):
                # A current fence grants no repair authority. It is temporary
                # backpressure, not exhaustion of the owner's resume count.
                return True
            registration = writer.get_control(f"scope:{scope.scope_id}")
            registered = None if registration is None else ControlRowV1.model_validate(
                registration,
            ).value
            if not isinstance(registered, ScopeRegistrationV1) or (
                registered.scope_kind != "user"
                or scope_id("user", registered.raw_external_id) != scope.scope_id
            ):
                raise QuoteMemError("binding_mismatch", "Owner recovery lost its user binding")
            user_id = registered.raw_external_id
            grant = self._grant(scope)
            check_add_grant(grant, scope, self._clock(), owner["principal_id"])

            def resume_request(observed: ControlRowV1) -> CompletionResumeRequestV3:
                budget = observed.value
                assert isinstance(budget, FinalizeBudgetV2 | FinalizeBudgetRefusalV4)
                return CompletionResumeRequestV3(
                    schema_version=3, user_id=user_id, owner_run_id=owner_run_id,
                    budget_version=observed.version, recovery_count=budget.recovery_count,
                    work_budget_ms=self._completion_work_budget_ms,
                )

            try:
                if quarantined:
                    # This path proves the original quarantine cycle and
                    # rejects parity failures and any active privacy fence.
                    self._recovery.repair_quarantine(scope, resume_request(selected))
                elif provider_paused:
                    # A provider pause needs its stored original failure
                    # and call allowance, even if its work budget also ended.
                    observed = self._recovery.inspect_recovery(RecoveryRequestV1(
                        schema_version=1, user_id=user_id,
                    )).failure
                    if observed is None or observed.owner_run_id != owner_run_id:
                        return True
                    self._recovery.resume_recovery(ResumeRequestV1(
                        schema_version=1, user_id=user_id,
                        owner_run_id=observed.owner_run_id,
                        child_run_id=observed.child_run_id,
                        owner_attempt=observed.owner_attempt,
                        child_attempt=observed.child_attempt,
                    ))
                elif isinstance(value, FinalizeBudgetRefusalV4):
                    self._recovery.resume_refused(scope, resume_request(selected))
                else:
                    if not self._retire_exhausted_budget(scope, owner_run_id):
                        return True
                    exhausted = read_budget(writer, scope, owner_run_id)
                    if exhausted is None:
                        return True
                    self._resume_exhausted_owner(scope, resume_request(exhausted))
            except QuoteMemError as error:
                if error.code in {
                    "extraction_busy", "operation_conflict", "lease_lost", "index_rebuilding",
                }:
                    # Compare-and-swap races and unjoined callbacks grant
                    # nothing. The next pass/retry re-reads their proofs.
                    return True
                if error.code in INTEGRITY_FAILURES:
                    return False
                raise
            with self._condition:
                self._retries.pop(owner_run_id, None)
                self._blocked_cycles.pop(owner_run_id, None)
            self.wake_waiters()
            return True
        except QuoteMemError as error:
            if error.code in INTEGRITY_FAILURES:
                return False
            raise
        finally:
            self._recovery_lock.release()

    def _access_resume_at(self, owner: dict[str, Any], recovery_count: int) -> int:
        """When a provider-paused owner may resume automatically: at once, or after a key refusal.

        The pause wrote the owner's provider-failure index with the failed
        call's HTTP diagnostics and stamped the owner Run (``pause_owner``), so
        the wait survives a restart. Only an access or quota refusal waits.
        """
        found = paused_index(self._ready.writer, owner["object_id"])
        if found is None or not access_refused(found[1].provider_failure):
            return 0
        return int(owner["updated_at"]) + access_resume_delay_ms(recovery_count)

    def _refuse_quarantined_scope(self, scope: ScopeV1) -> Never:
        """Recheck the actual quarantine cycle before requiring explicit repair."""
        writer = self._ready.writer
        lease_key = f"lease:finalize:{scope.scope_id}"
        quarantine_key = f"scope_quarantine:{scope.scope_id}"
        with writer.mutex:
            quarantine = writer.get_control(quarantine_key)
            raw = writer.get_control(lease_key)
            owner_run_id = None if raw is None else raw["value"]["owner_run_id"]
            selected = None if owner_run_id is None else read_budget(writer, scope, owner_run_id)
        if quarantine is None or owner_run_id is not None and self._automatic_resume_owner(
            scope, owner_run_id,
        ):
            raise RecoveryInProgress("The Add scope is recovering; retry")
        with writer.mutex:
            current = None if owner_run_id is None else read_budget(writer, scope, owner_run_id)
            if writer.get_control(quarantine_key) != quarantine or (
                writer.get_control(lease_key) != raw or current != selected
            ):
                raise RecoveryInProgress("The Add scope recovery changed; retry")
        raise QuoteMemError("repair_required", "The Add scope requires explicit repair")

    def complete_add(
        self, request: AddRequestV1, receipt: AddReceipt, budget: RequestBudget,
    ) -> AddOutcomeV1:
        """Return this request's result only after its actual owner released.

        The owner that completes an ordered request froze its prefix at its own
        admission: every request ordered before that admission belongs to it, and
        reception stays fenced while it is preparing or held, so a request ordered
        later goes to the next owner. An owner this request admits ends
        at its actual admission plus the installed policy; queue residence and
        HTTP attempt age cannot consume or renew that owner's allowance.

        An exact replay of a completed Add returns its first result through the
        same covering-prefix proof as a waiting Add. Exhaustion, a provider
        pause, a whole-output refusal or a provably repairable quarantine may
        resume the same owner up to 32 times through its joined-worker proofs.
        During recovery the Add answers retryable ``delivery_busy`` (429), except
        that an exhausted owner this request resumed (or saw resumed) is simply
        waited for again within the request's own budget.
        The shared durable bound, parity failures and invalid recovery proofs
        require explicit repair (409); one final counted allowance remains
        available to an operator.

        Each HTTP attempt waits within its own response budget before retryable
        ``delivery_busy``. Deferring a waiter leaves its shared owner
        running, so an exact retry joins that owner or reads its first result.
        """
        scope = ScopeV1(scope_kind="user", scope_id=scope_id("user", request.user_id))
        try:
            return self._complete_add(request, receipt, budget, scope)
        except QuoteMemError as error:
            if error.code != "scope_quarantined":
                raise
            self._refuse_quarantined_scope(scope)

    def _complete_add(
        self, request: AddRequestV1, receipt: AddReceipt, budget: RequestBudget, scope: ScopeV1,
    ) -> AddOutcomeV1:
        writer = self._ready.writer
        enabled = noted = False
        completion_deadline = budget.deadline
        if (continued_deadline := current_completion_wait_deadline(scope.scope_id)) is not None:
            completion_deadline = min(completion_deadline, continued_deadline)
        admission_deadline = completion_deadline
        while True:
            budget.check()
            grant = self._auth.resolve(AuthRequest(requested_scope=scope, now_ms=self._clock()))
            completed = self._completed_result(scope, receipt, grant, budget)
            if completed is not None:
                return completed
            if self.has_failed:
                raise QuoteMemError(
                    "service_unhealthy", "Bench execution failed" + describe(self.failure_reason),
                )
            if time.monotonic() >= completion_deadline:
                raise QuoteMemError("delivery_busy",
                                    "The scope's completion owner is still busy; retry")
            if not enabled:
                enable_completion(
                    writer, scope, receipt, grant=grant, clock=self._clock,
                    work_budget_ms=self._completion_work_budget_ms,
                )
                enabled = True
            if self._refuse_blocked_owner(scope, budget, receipt):
                continue
            generation = writer.add_admission.generation
            try:
                ticket = writer.admit_finalize(
                    scope=scope, grant=grant, clock=self._clock, resources=self._resources(),
                    admission_trigger="add_completion",
                    completion_policy=self._completion_policy,
                )
            except QuoteMemError as error:
                if error.code != "finalize_pending":
                    raise
                with budget.on_cancel(writer.add_admission.wake):
                    writer.add_admission.wait_for_change(
                        generation, deadline=admission_deadline, check=budget.check,
                    )
                continue
            if not noted:
                self.after_add(request, receipt)
                noted = True
            if self._refuse_ended_owner(scope, ticket.owner_run_id, receipt):
                continue
            if self._preserve_scoped_failure(ticket):
                raise QuoteMemError("repair_required", "Accepted Add owner failed its parity check")
            work = BenchWork(scope.scope_id, ticket.owner_run_id, "finalize")
            while time.monotonic() < completion_deadline:
                budget.check()
                if self._pool.submit(work, urgent=True):
                    self._pool.wait(
                        ticket.owner_run_id, deadline=completion_deadline, cancellation=budget,
                    )
                    break
                if self._is_allowance_ended(scope, ticket.owner_run_id) or (
                    self._read_stop_cause(ticket.owner_run_id) is not None
                ):
                    # A retired owner refuses every submission until it is
                    # resumed; the outer pass recovers it.
                    break
                generation = writer.mutex.progress_generation
                writer.mutex.wait_for_progress(
                    generation, deadline=min(completion_deadline, time.monotonic() + 0.05),
                    cancellation=budget,
                )
            budget.check()

    def _completed_result(
        self, scope: ScopeV1, receipt: AddReceipt, grant: PrincipalGrant, budget: RequestBudget,
    ) -> AddOutcomeV1 | None:
        """Return this request's first result with the completed prefix that covers it.

        The waiter of a new request and an exact replay of a completed one take
        this same path. The prefix projection and the request results become
        visible in the owner's one release transaction: a covering prefix
        without a result is a broken completion. A result whose covering proof no
        longer exists (an erasure or consent change since that release replaced
        the scope epochs) is still the request's immutable first result.
        """
        writer = self._ready.writer
        prefix = completed_prefix(
            writer, self._publisher, scope, receipt, grant=grant, clock=self._clock,
        )
        completed = read_add_outcome(writer, scope, receipt, grant=grant, clock=self._clock)
        if completed is None:
            if prefix is not None:
                raise QuoteMemError(
                    "incomplete_coverage", "Completed Add has no attributable request result",
                )
            return None
        if prefix is None:
            # The covering owner can release between the two reads above.
            prefix = completed_prefix(
                writer, self._publisher, scope, receipt, grant=grant, clock=self._clock,
            )
        budget.check()
        if prefix is not None:
            observe_add_completion(scope.scope_id, receipt, prefix)
        observe_add_result(scope.scope_id, completed)
        budget.mark("wait_finished")
        return completed.outcome

    def _refuse_ended_owner(
        self, scope: ScopeV1, owner_run_id: str, receipt: AddReceipt | None,
    ) -> bool:
        """Return retryable backpressure while an ended owner can recover automatically.

        ``receipt`` is None for an Add the fence keeps outside reception, which
        can only follow the owner's frozen prefix. The original owner alone
        receives a counted fresh allowance after callback join. Its first 32
        resumptions answer delivery_busy, except an exhaustion resume that left
        the owner live, which the caller waits for again; the durable bound
        leaves explicit repair, without discarding this Add or its queue position.

        Returns:
            True when the owner settled successfully while its expired allowance
            was being retired, so the caller rechecks completion; False while the
            owner's allowance is live.

        Raises:
            QuoteMemError: ``delivery_busy`` during recovery, otherwise ``repair_required``.
        """
        selected = read_budget(self._ready.writer, scope, owner_run_id)
        if selected is None:
            return False
        value = selected.value
        if not isinstance(value, FinalizeBudgetV2 | FinalizeBudgetRefusalV4):
            raise QuoteMemError("binding_mismatch", "Owner budget changed its contract")
        own = receipt is not None and (
            receipt.visibility_commit_seq <= value.accepted_prefix_commit_seq
        )
        if isinstance(value, FinalizeBudgetRefusalV4):
            if self._automatic_resume_owner(scope, owner_run_id):
                raise RecoveryInProgress("The stopped Add owner is recovering; retry")
            if read_budget(self._ready.writer, scope, owner_run_id) != selected:
                return True
            raise QuoteMemError("repair_required", "A refused preceding owner blocks this Add")
        if value.state == "active" and self._clock() < value.deadline_at:
            return False
        if not self._retire_exhausted_budget(scope, owner_run_id):
            return True
        exhausted = read_budget(self._ready.writer, scope, owner_run_id)
        if exhausted is None or isinstance(exhausted.value, FinalizeBudgetV2) and (
            exhausted.value.state == "active" and self._clock() < exhausted.value.deadline_at
        ):
            return True
        if self._automatic_resume_owner(scope, owner_run_id):
            if self._allowance_renewed(scope, owner_run_id):
                # The same owner continues on a fresh allowance: this request
                # keeps waiting for it within its own budget.
                return True
            raise RecoveryInProgress("The exhausted Add owner is recovering; retry")
        if read_budget(self._ready.writer, scope, owner_run_id) != exhausted:
            return True
        raise QuoteMemError("repair_required", (
            "Accepted Add owner exhausted its allowance" if own
            else "An exhausted preceding owner blocks this Add"
        ))

    def _allowance_renewed(self, scope: ScopeV1, owner_run_id: str) -> bool:
        """Whether an exhausted owner holds a live allowance again after automatic recovery.

        A concurrent recovery (discovery's pass, another request) may hold the
        recovery lock; waiting at most ``RECOVERY_JOIN_SECONDS`` for it lets this
        request see its outcome instead of answering retryable delivery_busy.
        """
        if self._recovery_lock.acquire(timeout=RECOVERY_JOIN_SECONDS):
            self._recovery_lock.release()
        selected = read_budget(self._ready.writer, scope, owner_run_id)
        if selected is None or not isinstance(selected.value, FinalizeBudgetV2) or (
            selected.value.state != "active" or self._clock() >= selected.value.deadline_at
        ):
            return False
        return self._read_stop_cause(owner_run_id) is None and not (
            self._ready.writer.is_scope_quarantined(scope.scope_id)
        )

    def refuse_fenced_add(self, request: AddRequestV1, budget: RequestBudget) -> None:
        """Recover or refuse a proved stopped owner before this Add waits behind its fence.

        An owner holds its input fence until release, so a new Add cannot be
        ordered behind it. While that owner is live the Add waits within its own
        response allowance. A stopped owner receives at most 32 automatic
        resumptions through the same durable recovery and callback-join proofs
        as explicit recovery. A retry during that transition gets delivery_busy;
        an exhausted automatic bound or a failed proof needs explicit repair.
        """
        budget.check()
        scope = ScopeV1(scope_kind="user", scope_id=scope_id("user", request.user_id))
        writer = self._ready.writer
        if writer.is_scope_quarantined(scope.scope_id):
            self._refuse_quarantined_scope(scope)
        grant = self._grant(scope)
        with writer.mutex:
            check_add_grant(grant, scope, self._clock())
            if read_completed_add(
                writer, scope, request.session_id, request.request_id,
            ) is not None:
                # The reception caller already proved this exact payload. A
                # later stopped owner cannot delay an immutable earlier result.
                return
        raw = writer.get_control(f"lease:finalize:{scope.scope_id}")
        lease = None if raw is None else ControlRowV1.model_validate(raw).value
        if isinstance(lease, PreparingFinalizeLeaseV1 | HeldFinalizeLeaseV1):
            owner = writer.get_row("run", lease.owner_run_id)
            if owner is not None and (owner["status"], owner["failure_code"]) == (
                "manual_attention", "parity_mismatch",
            ):
                raise QuoteMemError("repair_required", "A preceding owner failed its parity check")
        self._refuse_blocked_owner(scope, budget, None)

    def _refuse_blocked_owner(
        self, scope: ScopeV1, budget: RequestBudget, receipt: AddReceipt | None,
    ) -> bool:
        """Refuse waiting on the scope's current owner when it cannot finish by itself.

        The owner's allowance is judged first: an exhausted, expired or refused
        allowance resumes only through its recovery proofs. A provider pause
        also proves its original failure and renewed call allowance.

        Returns:
            True when the owner changed concurrently and the caller rechecks
            completion; False when no current owner blocks the request.
        """
        budget.check()
        raw = self._ready.writer.get_control(f"lease:finalize:{scope.scope_id}")
        lease = None if raw is None else ControlRowV1.model_validate(raw).value
        if not isinstance(lease, PreparingFinalizeLeaseV1 | HeldFinalizeLeaseV1):
            return False
        if self._refuse_ended_owner(scope, lease.owner_run_id, receipt):
            return True
        writer = self._ready.writer
        with writer.mutex:
            stopped = self._read_stop_cause(lease.owner_run_id)
            selected = read_budget(writer, scope, lease.owner_run_id)
        if stopped == "provider":
            if self._automatic_resume_owner(scope, lease.owner_run_id):
                raise RecoveryInProgress("The paused Add owner is recovering; retry")
            with writer.mutex:
                if self._read_stop_cause(lease.owner_run_id) != stopped or read_budget(
                    writer, scope, lease.owner_run_id,
                ) != selected:
                    return True
            raise QuoteMemError("repair_required",
                                "A paused owner needs explicit provider recovery")
        return False

    def after_add(self, request: AddRequestV1, receipt: AddReceipt) -> None:
        """Refresh the scope's live activity after an actual Add, so no idle owner is due.

        A reused receipt published nothing new. HTTP owns this callback, so
        withdrawal alone cannot reject it; complete HTTP drainage precedes the
        controller's final submission seal.
        """
        if receipt.message_count != len(request.messages):
            raise QuoteMemError("binding_mismatch",
                                "Reception notification differs from its receipt")
        with self._condition:
            if self._draining or self._failure.is_set():
                raise QuoteMemError("service_unhealthy", "Bench notification admission is closed")
            if receipt.reused:
                return
            scope = scope_id("user", request.user_id)
            state = self._activity.setdefault(scope, _ScopeActivity(time.monotonic()))
            state.last_activity = time.monotonic()

    # Recovery operations -------------------------------------------------------------------

    def inspect_recovery(self, request: RecoveryRequestV1) -> RecoveryResponseV1:
        """Observe a provider pause without reviving it (explicit recovery)."""
        self.require_ready()
        return self._recovery.inspect_recovery(request)

    def resume_recovery(self, request: ResumeRequestV1) -> ResumeResponseV1:
        """Resume one observed provider pause through the original durable owner."""
        self.require_ready()
        with self._recovery_lock:
            return self._recovery.resume_recovery(request)

    def inspect_completion_budget(
        self, request: CompletionRecoveryRequestV2 | CompletionRecoveryRequestV3,
    ) -> CompletionRecoveryResponseV2 | CompletionRecoveryResponseV3:
        """Read the scope's owner allowance without creating or extending work.

        A V3 request naming one of the user's Add requests also reads that
        request's first completed result, with its explicit-repair count.
        """
        self.require_ready()
        scope = ScopeV1(scope_kind="user", scope_id=scope_id("user", request.user_id))
        grant = self._auth.resolve(AuthRequest(requested_scope=scope, now_ms=self._clock()))
        writer = self._ready.writer
        with writer.mutex:
            check_add_grant(grant, scope, self._clock())
            completed = None
            if isinstance(request, CompletionRecoveryRequestV3) and (
                request.session_id is not None and request.request_id is not None
            ):
                stored = read_completed_add(writer, scope, request.session_id, request.request_id)
                completed = None if stored is None else CompletedAddV1.from_completed(stored)
            quarantined = writer.is_scope_quarantined(scope.scope_id)
            unobserved = CompletionRecoveryResponseV3(
                budget=None, budget_version=None, completed=completed,
                failure_cause="integrity" if quarantined else None,
            ) if isinstance(request, CompletionRecoveryRequestV3) else CompletionRecoveryResponseV2(
                budget=None, budget_version=None,
            )
            raw = writer.get_control(f"finalize_budget:{scope.scope_id}")
            if raw is None:
                return unobserved
            selected = ControlRowV1.model_validate(raw)
            if not isinstance(selected.value, FinalizeBudgetV2 | FinalizeBudgetRefusalV4):
                raise QuoteMemError("binding_mismatch", "Completion owner budget is invalid")
            owner = writer.get_row("run", selected.value.owner_run_id)
            if owner is None or (owner["scope_kind"], owner["scope_id"]) != (
                scope.scope_kind, scope.scope_id,
            ):
                raise QuoteMemError("binding_mismatch", "Completion budget lost its scoped owner")
            check_add_grant(grant, scope, self._clock(), owner["principal_id"])
            if owner["status"] == "succeeded":
                return unobserved
            if read_budget(writer, scope, selected.value.owner_run_id) != selected:
                raise QuoteMemError("operation_conflict",
                                    "Completion observation changed its owner")
            if isinstance(request, CompletionRecoveryRequestV3):
                # The content-free cause of the owner's stop (the V3 contract).
                cause = failure_cause(
                    quarantined=quarantined, budget_state=selected.value.state,
                    provider_paused=(owner["status"], owner["failure_code"]) == (
                        "manual_attention", "provider_error",
                    ),
                )
                return CompletionRecoveryResponseV3(
                    budget=selected.value, budget_version=selected.version,
                    failure_cause=cause, completed=completed,
                )
            if isinstance(selected.value, FinalizeBudgetRefusalV4):
                raise QuoteMemError("operation_conflict",
                                    "A completed refusal requires V3 inspection")
            return CompletionRecoveryResponseV2(
                budget=selected.value, budget_version=selected.version,
            )

    def resume_completion_budget(
        self, request: CompletionResumeRequestV2 | CompletionResumeRequestV3,
    ) -> CompletionRecoveryResponseV2 | CompletionRecoveryResponseV3:
        """Serialize explicit recovery with automatic recovery of the same owner."""
        with self._recovery_lock:
            return self._resume_completion_budget(request)

    def _resume_completion_budget(
        self, request: CompletionResumeRequestV2 | CompletionResumeRequestV3,
    ) -> CompletionRecoveryResponseV2 | CompletionRecoveryResponseV3:
        """Resume one ended original owner only after its actual callbacks join.

        An exhausted allowance resumes in place. A refusal, which only V3
        observes, resumes through explicit recovery, which re-dispatches the
        refused unit on the owner's next attempt. A quarantined scope, whose V3
        failure cause is ``integrity``, is repaired by the explicit resume of the
        owner its receipt names, whatever its allowance. Each resume grants
        a fresh allowance of at most 1,740 seconds, counts against the owner's
        shared cap and may follow the original deadline; an active
        allowance of a scope that is not quarantined never resumes.

        Raises:
            QuoteMemError: ``operation_conflict`` when the observed budget is no longer
                current or cannot resume.
        """
        observed = self.inspect_completion_budget(request)
        budget = observed.budget
        quarantined = isinstance(observed, CompletionRecoveryResponseV3) and (
            observed.failure_cause == "integrity"
        )
        if budget is None or (
            budget.owner_run_id, observed.budget_version, budget.recovery_count,
        ) != (request.owner_run_id, request.budget_version, request.recovery_count) or (
            not quarantined and budget.state not in {"exhausted", "refused"}
        ):
            raise QuoteMemError("operation_conflict",
                                "Observed completion budget is no longer current")
        scope = ScopeV1(scope_kind="user", scope_id=scope_id("user", request.user_id))
        if quarantined:
            if not isinstance(request, CompletionResumeRequestV3):
                raise QuoteMemError("operation_conflict", "A quarantine repair requires V3 resume")
            self._recovery.repair_quarantine(scope, request)
        elif isinstance(budget, FinalizeBudgetRefusalV4):
            if not isinstance(request, CompletionResumeRequestV3):
                raise QuoteMemError("operation_conflict", "A completed refusal requires V3 resume")
            self._recovery.resume_refused(scope, request)
        else:
            self._resume_exhausted_owner(scope, request)
        resumed = self.inspect_completion_budget(request)
        if type(resumed) is not type(observed):
            raise QuoteMemError("binding_mismatch", "Completion resume changed its public contract")
        return resumed

    def skip_quarantined_cycle(self, request: QuarantineSkipRequestV1) -> QuarantineSkipResponseV1:
        """Skip one user's quarantined cycle on an operator's explicit request only.

        Nothing calls this automatically. Under the same exclusion as every
        explicit recovery, the owner's submission is retired and its callbacks
        are proved joined; the Writer then skips the cycle
        (``store.quarantine_skip``): the owner fails ``input_abandoned``, the
        requests it froze complete as skipped, the quarantine is lifted and the
        outputs of units that settled nothing are dropped. The pool fence is then
        released without resubmitting the ended owner, and waiting Adds wake:
        each skipped request answers its skipped result, and the next Add admits
        a new cycle.

        Raises:
            QuoteMemError: ``operation_conflict`` for a scope that is not quarantined
                or a cycle that moved since it was observed; ``extraction_busy``
                while the owner's worker has not joined; the Writer's refusals.
        """
        self.require_ready()
        scope = ScopeV1(scope_kind="user", scope_id=scope_id("user", request.user_id))
        grant = self._auth.resolve(AuthRequest(requested_scope=scope, now_ms=self._clock()))
        writer = self._ready.writer
        work = BenchWork(scope.scope_id, request.owner_run_id, "finalize")
        with self._recovery_lock, writer.mutex:
            check_add_grant(grant, scope, self._clock())
            if not writer.is_scope_quarantined(scope.scope_id):
                raise QuoteMemError("operation_conflict", "Only a quarantined scope can be skipped")
            owner = writer.get_row("run", request.owner_run_id)
            if owner is None:
                raise QuoteMemError("operation_conflict", "The observed owner no longer exists")
            check_add_grant(grant, scope, self._clock(), owner["principal_id"])
            if not self._pool.retire_owner(work):
                raise QuoteMemError("extraction_busy", "The quarantined owner cannot retire yet")
            joined = self._pool.retired_owner_join(work)
            skipped = skip_quarantined_cycle(
                writer, scope, request.owner_run_id, budget_version=request.budget_version,
                recovery_count=request.recovery_count, now=self._clock(),
            )
            self._pool.resume_retired_owner(joined)
        with self._condition:
            self._retries.pop(request.owner_run_id, None)
            self._blocked_cycles.pop(request.owner_run_id, None)
        self.wake_waiters()
        return QuarantineSkipResponseV1(
            owner_run_id=skipped.owner_run_id, skipped_requests=skipped.skipped_requests,
            dropped_records=skipped.dropped_records,
        )

    def _resume_exhausted_owner(
        self, scope: ScopeV1, request: CompletionResumeRequestV2 | CompletionResumeRequestV3,
    ) -> None:
        """Renew the observed exhausted allowance under the same exclusion as its pool fence."""
        writer = self._ready.writer
        work = BenchWork(scope.scope_id, request.owner_run_id, "finalize")
        grant = self._auth.resolve(AuthRequest(requested_scope=scope, now_ms=self._clock()))
        with writer.mutex:
            selected = read_budget(writer, scope, request.owner_run_id)
            if selected is None or selected.version != request.budget_version:
                raise QuoteMemError("operation_conflict", "Observed completion budget changed")
            self._pool.retire_owner(work)
            joined = self._pool.retired_owner_join(work)
            resume_budget(
                writer, scope, selected, joined=joined, grant=grant, clock=self._clock,
                work_budget_ms=request.work_budget_ms,
            )
            self._pool.resume_retired_owner(joined)
        self._pool.submit(work, urgent=True)
        self.wake_waiters()

    # Scope state ---------------------------------------------------------------------------

    def _finalize_active(self, scope: ScopeV1) -> bool:
        raw = self._ready.writer.get_control(f"lease:finalize:{scope.scope_id}")
        if raw is None:
            return False
        return isinstance(
            ControlRowV1.model_validate(raw).value, PreparingFinalizeLeaseV1 | HeldFinalizeLeaseV1,
        )

    def _scope_unavailable(self, scope: ScopeV1) -> bool:
        """Recognize a durable scope quarantine or actual scoped privacy ownership."""
        if self._ready.writer.is_scope_quarantined(scope.scope_id):
            return True
        key = f"tomb:{scope.scope_id}"
        raw = self._ready.writer.get_control(key)
        if raw is not None:
            if ControlRowV1.model_validate(raw).key != key:
                raise QuoteMemError("binding_mismatch", "Scope tomb changed its selected identity")
            return True
        controls = self._reader.scope_fences(scope.scope_id)
        for control in controls:
            ControlRowV1.model_validate(control)
        pending = self._reader.pending_privacy(scope.scope_id, limit=2)
        for run in pending:
            DOMAIN_ADAPTERS["run"].validate_python(run)
        return bool(controls or pending)

    def _privacy_cancelled(self, work: BenchWork) -> bool:
        """Whether a privacy rebase cancelled the owner's cycle and failed its Run."""
        writer = self._ready.writer
        with writer.mutex:
            run = writer.get_row("run", work.run_id)
            lease = writer.get_control(f"lease:finalize:{work.scope_id}")
        return run is not None and run["status"] == "failed" and lease is not None and (
            lease["value"].get("state") == "cancelled"
            and lease["value"].get("owner_run_id") == work.run_id
        )

    def _expected_scope_refusal(self, scope: ScopeV1, error: QuoteMemError) -> bool:
        return error.code in SCOPED_REFUSALS and self._scope_unavailable(scope)

    def _preserve_scoped_failure(self, ticket: FinalizeLeaseTicket) -> bool:
        """Keep an actually recorded takeover fault blocked without inventing repair."""
        writer = self._ready.writer
        with writer.mutex:
            row = writer.get_row("run", ticket.owner_run_id)
            if row is None:
                raise QuoteMemError("reference_missing", "Failed owner lost its Run")
            DOMAIN_ADAPTERS["run"].validate_python(row)
            if row["status"] != "manual_attention" or row["failure_code"] != "parity_mismatch":
                return False
            with self._condition:
                known = ticket.owner_run_id in self._blocked_cycles
            if not known:
                self._publisher.publish_finalize_failure(
                    writer, ScopeV1(scope_kind="user", scope_id=ticket.scope_id),
                    ticket.owner_run_id, ticket.lease_generation, ticket.writer_epoch,
                )
                with self._condition:
                    self._blocked_cycles[ticket.owner_run_id] = None
                    if len(self._blocked_cycles) > self._cache_limit:
                        self._blocked_cycles.popitem(last=False)
            return True

    # Discovery -----------------------------------------------------------------------------

    def _recover_finalize(self, raw: dict[str, Any]) -> bool:
        """Resubmit one saved owner, or classify why it waits; False means a full queue."""
        DOMAIN_ADAPTERS["run"].validate_python(raw)
        scope = ScopeV1(scope_kind=raw["scope_kind"], scope_id=raw["scope_id"])
        if scope.scope_kind != "user":
            raise QuoteMemError("scope_denied", "Bench owner recovery requires a user")
        work = BenchWork(scope.scope_id, raw["object_id"], "finalize")
        if self._pool.owns_scheduled(work):
            return True
        if self._automatic_resume_owner(scope, work.run_id):
            return True
        if self._scope_unavailable(scope):
            return True
        if self._retire_stopped_owner(work):
            return True
        if self._retire_exhausted_budget(scope, work.run_id):
            return True
        grant = self._auth.resolve(AuthRequest(requested_scope=scope, now_ms=self._clock()))
        with self._ready.writer.mutex:
            current = self._ready.writer.get_row("run", raw["object_id"])
            if current is None:
                raise QuoteMemError("reference_missing", "Saved owner disappeared")
            # An owner that settled after the pass read its page (a forget fails a live
            # owner) is no longer saved; admitting for it would admit the scope's
            # next cycle, which the identity check below refuses.
            if current["status"] in {"succeeded", "failed"}:
                return True
            ticket = self._ready.writer.admit_finalize(
                scope=scope, grant=grant, clock=self._clock,
                completion_policy=self._completion_policy,
            )
        if ticket.owner_run_id != raw["object_id"]:
            raise QuoteMemError("binding_mismatch", "Recovery selected another owner")
        if self._preserve_scoped_failure(ticket):
            return True
        with self._condition:
            self._blocked_cycles.pop(ticket.owner_run_id, None)
        return self._pool.submit(work)

    def _saved_owner_admissions(self) -> Iterator[bool]:
        """Yield one classification at a time while retaining at most one Run page.

        Suspension preserves a shutdown scan's keyset position, so short repeated
        deadlines cannot starve owners on later pages. False leaves that durable
        owner for another full pass.
        """
        after: str | None = None
        while True:
            rows = self._reader.pending_finalizes(after=after)
            for raw in rows:
                try:
                    accepted = self._recover_finalize(raw)
                except Exception as error:
                    if not isinstance(error, QuoteMemError) or error.code not in DISCOVERY_SKIP:
                        self._quarantine_error(
                            ScopeV1(scope_kind=raw["scope_kind"], scope_id=raw["scope_id"]),
                            error, owner_run_id=raw["object_id"],
                        )
                    # An unavailable scope, or an owner whose lease a live owner
                    # holds, is carried by that state; the next scan reads it afresh.
                    accepted = True
                yield accepted
            if len(rows) < _PAGE:
                return
            after = rows[-1]["object_id"]

    def _recover_work(self) -> bool:
        """Classify saved owners without retaining the durable backlog in a map."""
        admitted = True
        for selected in self._saved_owner_admissions():
            admitted &= selected
        return admitted

    def _drain_saved_owners(self, deadline: float) -> bool:
        """Continue one bounded scan before sealing the existing execution pool."""
        if self._drain_recovery is None:
            self._drain_recovery = self._saved_owner_admissions()
        while time.monotonic() < deadline:
            try:
                admitted = next(self._drain_recovery)
            except StopIteration:
                self._drain_recovery = None
                rejected, self._drain_rejected = self._drain_rejected, False
                return not rejected
            self._drain_rejected |= not admitted
        return False

    def _released_history(self, scope: ScopeV1) -> bool:
        """Whether the scope has a released owner, whose history only the proof may read.

        Admission after a released predecessor fails closed until the
        released-history proof exists (``store/owner_lease.py``), so the idle pass
        admits no owner over released history meanwhile.
        """
        raw = self._ready.writer.get_control(f"lease:finalize:{scope.scope_id}")
        return raw is not None and isinstance(
            ControlRowV1.model_validate(raw).value, ReleasedFinalizeLeaseV1,
        )

    def _idle_allowed(self, scope: ScopeV1, marker: FinalizedValueV1 | None) -> bool:
        """Whether the scope has ordered input that no live or released owner covers."""
        writer = self._ready.writer
        if self._finalize_active(scope) or self._scope_unavailable(scope):
            return False
        if self._released_history(scope):
            return False
        if not self._reader.has_available_evidence(scope, writer.tenant_id, self._clock()):
            return False
        return marker is None or marker.finalized_scope_seq != writer.get_counter(
            f"scope_seq:{scope.scope_id}",
        )

    def _finalized_marker(self, scope: ScopeV1) -> FinalizedValueV1 | None:
        raw = self._ready.writer.get_control(f"finalized:{scope.scope_id}")
        value = None if raw is None else ControlRowV1.model_validate(raw).value
        if value is not None and not isinstance(value, FinalizedValueV1):
            raise QuoteMemError("binding_mismatch", "Finalized marker changed its contract")
        return value

    def _try_idle_finalize(self, scope: ScopeV1) -> bool:
        """Admit an idle owner for the scope's unfinalized prefix; True once it is submitted.

        A full descriptor queue returns False: the admitted owner is durable and a
        later pass reaches it, but no worker started it in this pass.
        """
        writer = self._ready.writer
        with writer.mutex:
            expected = writer.ref_generation, writer.get_counter(f"scope_seq:{scope.scope_id}")
        # Body-free activity includes the durable release-to-callback gap. The
        # eligibility reads stay outside the final admission critical section.
        latest = self._reader.latest_bench_activity(scope, writer.tenant_id)
        marker = self._finalized_marker(scope)
        if latest is None or not self._idle_allowed(scope, marker):
            return False
        grant = self._auth.resolve(AuthRequest(requested_scope=scope, now_ms=self._clock()))
        with writer.mutex:
            if expected != (
                writer.ref_generation, writer.get_counter(f"scope_seq:{scope.scope_id}"),
            ) or self._finalize_active(scope) or self._scope_unavailable(scope):
                return False
            with self._condition:
                state = self._activity[scope.scope_id]
                if self._withdrawn or (
                    time.monotonic() - state.last_activity < state.idle_seconds
                    or self._clock() - latest < state.idle_seconds * 1000
                ):
                    return False
            ticket = writer.admit_finalize(
                scope=scope, grant=grant, clock=self._clock, resources=self._resources(),
                admission_trigger="idle", completion_policy=self._completion_policy,
            )
        return self._pool.submit(BenchWork(scope.scope_id, ticket.owner_run_id, "finalize"))

    def _try_reserved_finalize(self, scope: ScopeV1) -> bool:
        """Drive one ordered Add even after its receiver has left.

        The queue is only a scheduling hint. The fresh grant and existing Writer
        admission prove the released predecessor, exact committed prefix and
        admission-time owner allowance before the existing pool can execute it.
        """
        writer = self._ready.writer
        queue = writer.add_admission
        if not queue.owner_ready(scope.scope_id):
            return False
        grant = self._auth.resolve(AuthRequest(requested_scope=scope, now_ms=self._clock()))
        with writer.mutex:
            reconcile_add_admission(writer, scope.scope_id)
            if not queue.owner_ready(scope.scope_id) or self._finalize_active(scope) or (
                self._scope_unavailable(scope)
            ):
                return False
            with self._condition:
                if self._withdrawn:
                    return False
            ticket = writer.admit_finalize(
                scope=scope, grant=grant, clock=self._clock, resources=self._resources(),
                admission_trigger="add_completion",
                completion_policy=self._completion_policy,
            )
        return self._pool.submit(
            BenchWork(scope.scope_id, ticket.owner_run_id, "finalize"), urgent=True,
        )

    def _discover_scope(self, scope: ScopeV1) -> None:
        """Drive a reserved completion first, otherwise apply the ordinary idle wait."""
        try:
            if self._try_reserved_finalize(scope):
                return
            with self._condition:
                state = self._activity.get(scope.scope_id)
                due = state is not None and (
                    time.monotonic() - state.last_activity >= state.idle_seconds
                )
            if due:
                self._try_idle_finalize(scope)
        except QuoteMemError as error:
            if error.code == "parity_mismatch" and self._finalize_active(scope):
                # A live owner took the scope between this pass's eligibility
                # reads and its admission; that owner now covers the input.
                return
            raise

    def _expire_receptions(self) -> None:
        """Clean up expired unsaved roots; accepted source is retained by the Writer."""
        started = time.monotonic()
        with self._condition:
            if self._withdrawn or started < self._expiry_due:
                return
            self._expiry_due = started + RECEPTION_EXPIRY_INTERVAL_SECONDS
        writer = self._ready.writer
        after: str | None = None
        while True:
            rows = self._reader.inputless_adds(after=after)
            for raw in rows:
                with self._condition:
                    if self._withdrawn:
                        return
                scope = ScopeV1(scope_kind=raw["scope_kind"], scope_id=raw["scope_id"])
                if self._scope_unavailable(scope) or self._reader.run_children(
                    raw["object_id"], limit=1,
                ):
                    continue
                try:
                    writer.abandon_add_run(raw["object_id"], self._clock())
                except QuoteMemError as error:
                    self._record_refusal("reception_expiry", error)
            if len(rows) < _PAGE:
                return
            after = rows[-1]["object_id"]

    def _refill_received(self) -> None:
        """Restore saved overflow as directory capacity becomes free after recovery."""
        writer = self._ready.writer
        queue = writer.add_admission
        candidates: list[
            tuple[str, str, int, int | None, int | None, tuple[str, str], int | None]
        ] = []
        after: str | None = None
        while True:
            rows = self._reader.inputless_adds(after=after)
            for run in rows:
                with self._condition:
                    if self._withdrawn:
                        return
                if queue.has_saved_input(run["scope_id"], run["object_id"]):
                    continue
                scope = ScopeV1(scope_kind="user", scope_id=run["scope_id"])
                if self._scope_unavailable(scope) or self._reader.run_children(
                    run["object_id"], limit=1,
                ):
                    continue
                saved = writer.read_received_add_input(run["object_id"])
                if saved is None:
                    continue
                payload = saved.request_payload
                if not isinstance(payload, MessagesRequestPayloadV1) or payload.profile != "bench":
                    continue
                times = [message.timestamp for message in payload.messages]
                known_times = [value for value in times if value is not None]
                source = min(known_times) if len(known_times) == len(times) else None
                candidates.append((
                    run["scope_id"], run["object_id"], saved.received_at or run["created_at"],
                    source, saved.work_deadline_at,
                    (saved.business_key.session_id, saved.business_key.request_id),
                    saved.reception_sequence,
                ))
            if len(rows) < _PAGE:
                break
            after = rows[-1]["object_id"]

        def order(
            entry: tuple[str, str, int, int | None, int | None, tuple[str, str], int | None],
        ) -> tuple[str, int, int, int, int, str]:
            scope, run_id, received, source, _, identity, sequence = entry
            chunk = message_chunk_index(identity[1])
            return scope, int(chunk is None), (
                chunk if chunk is not None else source if source is not None else 2**63
            ), received, sequence or 0, run_id

        for scope, run_id, received, source, deadline, identity, sequence in sorted(
            candidates, key=order,
        ):
            try:
                queue.register(
                    scope, run_id, received, source_time=source,
                    work_deadline_at=deadline, identity=identity,
                    reception_sequence=sequence,
                )
                queue.request_completion(scope, run_id)
            except QuoteMemError as error:
                if error.code != "delivery_busy":
                    raise

    def _dispatch_pending_receptions(self) -> None:
        """Wake one saved head immediately, including work with no HTTP receiver."""
        queue = self._ready.writer.add_admission
        for queued_scope in queue.pending_scopes():
            with self._condition:
                if self._withdrawn:
                    return
            scope = ScopeV1(scope_kind="user", scope_id=queued_scope)
            try:
                reconcile_add_admission(self._ready.writer, queued_scope)
                if self._scope_unavailable(scope) or self._finalize_active(scope):
                    continue
                if self._try_reserved_finalize(scope):
                    continue
                run_id = queue.next_run(queued_scope)
                if run_id is not None:
                    self._pool.submit(BenchWork(queued_scope, run_id, "reception"), urgent=True)
            except QuoteMemError as error:
                if error.code not in DISCOVERY_SKIP | {"delivery_busy"}:
                    raise
                self._record_refusal("reception_discovery", error)

    def _discover(self) -> None:
        """Recover owners, clean unsaved roots and discover ordinary idle work."""
        self._recover_work()
        self._expire_receptions()
        self._refill_received()
        self._dispatch_pending_receptions()
        after: str | None = None
        while True:
            rows = self._reader.bench_scope_activity(self._ready.writer.tenant_id, after=after)
            for row in rows:
                with self._condition:
                    if self._withdrawn:
                        return
                    age = max(0.0, (self._clock() - row.last_ingested_at) / 1000)
                    self._activity.setdefault(
                        row.scope_id, _ScopeActivity(time.monotonic() - age),
                    )
                scope = ScopeV1(scope_kind="user", scope_id=row.scope_id)
                try:
                    self._discover_scope(scope)
                except Exception as error:
                    if not isinstance(error, QuoteMemError) or error.code not in DISCOVERY_SKIP:
                        self._quarantine_error(scope, error)
            if len(rows) < _PAGE:
                return
            after = rows[-1].scope_id

    def _schedule(self) -> None:
        """Dispatch queue changes immediately; keep full discovery on its cadence."""
        try:
            due = time.monotonic()
            while True:
                with self._condition:
                    if self._withdrawn:
                        return
                # Clear before inspecting queue state so concurrent registration
                # or release remains signalled for the next iteration.
                self._reception_changed.clear()
                try:
                    if time.monotonic() >= due:
                        self._discover()
                        due = time.monotonic() + DISCOVERY_INTERVAL_SECONDS
                    else:
                        self._dispatch_pending_receptions()
                except Exception as error:
                    if not isinstance(error, QuoteMemError) or error.code not in {
                        "index_rebuilding", "reception_stale",
                    }:
                        raise
                    # Scope-level errors are already quarantined. An enumeration
                    # conflict is retained for this pass; repeating it does not
                    # become authority to withdraw other users.
                    self._record_refusal("discovery", error)
                    due = time.monotonic() + DISCOVERY_INTERVAL_SECONDS
                self._reception_changed.wait(max(0.0, due - time.monotonic()))
        except BaseException as error:
            self._record_failure("discovery", error)
            self._fail()

    # Heartbeats ----------------------------------------------------------------------------

    def _heartbeat(self) -> None:
        try:
            due = time.monotonic()
            while True:
                with self._condition:
                    while not self._heartbeat_stop and time.monotonic() < due:
                        self._condition.wait(due - time.monotonic())
                    if self._heartbeat_stop:
                        return
                try:
                    self._heartbeat_page_pass()
                except Exception as error:
                    if not isinstance(error, QuoteMemError) or error.code not in {
                        "index_rebuilding", "reception_stale",
                    }:
                        raise
                    self._record_refusal("heartbeat", error)
                finally:
                    due = time.monotonic() + HEARTBEAT_INTERVAL_SECONDS
        except BaseException as error:
            self._record_failure("heartbeat", error)
            self._fail()

    def _heartbeat_page_pass(self) -> None:
        """Renew live owners' leases, owners outside the bounded queue included.

        Pages keep metadata only. Each entry is compared with the current lease and
        its scope's privacy state as borrowed reads, so a released, replaced,
        recently renewed or expired entry costs the Writer mutex nothing. Due
        renewals then take the mutex once each (``HEARTBEAT_RENEWAL_BATCH``); each
        reads its own preimage and rebinds its own ticket under the held mutex
        right before its heartbeat CAS.

        A page settles its expiries after its renewals: settling one inline
        delayed every lease the pass had not reached yet. The settlement re-proves
        the lease, its owner and its staleness under the mutex before it cancels
        anything. A ``lease_lost`` refusal is classified with the row ledger
        outside any Writer exclusion: a lease or owner mutated since the borrowed
        read answers a later state the next pass reads afresh.
        """
        writer = self._ready.writer
        after: str | None = None
        while True:
            with self._condition:
                if self._heartbeat_stop:
                    return
            rows = self._reader.active_finalize_leases(after=after)
            writer.probe_storage_owner()
            due: list[_HeartbeatLease] = []
            expired: list[_HeartbeatLease] = []
            for raw in rows:
                scope: ScopeV1 | None = None
                try:
                    selected = ControlRowV1.model_validate(raw)
                    value = selected.value
                    if not isinstance(value, PreparingFinalizeLeaseV1 | HeldFinalizeLeaseV1):
                        raise QuoteMemError("schema_invalid",
                                            "Heartbeat selected an inactive lease")
                    scope = ScopeV1(
                        scope_kind="user", scope_id=selected.key.removeprefix("lease:finalize:"),
                    )
                    if self._held_stop(scope, value.owner_run_id):
                        continue
                    token = writer.storage_epoch
                    if self._scope_unavailable(scope):
                        continue
                    if self._retire_exhausted_budget(scope, value.owner_run_id):
                        continue
                    raw_current = writer.get_control(selected.key)
                    if raw_current is None:
                        raise QuoteMemError("lease_lost", "Heartbeat lost its actual durable cycle")
                    active = ControlRowV1.model_validate(raw_current).value
                    if not isinstance(active, PreparingFinalizeLeaseV1 | HeldFinalizeLeaseV1):
                        continue
                    if (active.owner_run_id, active.lease_generation, active.writer_epoch) != (
                        value.owner_run_id, value.lease_generation, value.writer_epoch,
                    ):
                        continue
                    age = self._clock() - active.heartbeat_at
                    entry = _HeartbeatLease(selected, scope, value, active, token)
                    if age > FINALIZE_LEASE_LIFETIME_MS:
                        expired.append(entry)
                    elif age >= HEARTBEAT_RENEWAL_AGE_MS:
                        due.append(entry)
                except Exception as error:
                    if not isinstance(error, QuoteMemError) or error.code not in DISCOVERY_SKIP:
                        if scope is None:
                            raise
                        self._quarantine_error(scope, error)
                    # This pass reads each lease without holding it, so the owner
                    # it inspected can settle between two of its reads. Selection
                    # collects such a refusal as the renewal half does.
                    continue
            for index in range(0, len(due), HEARTBEAT_RENEWAL_BATCH):
                refused: list[tuple[_HeartbeatLease, Exception]] = []
                with writer.mutex:
                    for entry in due[index:index + HEARTBEAT_RENEWAL_BATCH]:
                        try:
                            if not self._renew_held_lease(entry):
                                expired.append(entry)
                        except Exception as error:
                            refused.append((entry, error))
                # The ledger and owner behind a refusal are read outside the
                # exclusion the refusal was raised under.
                for entry, error in refused:
                    if not isinstance(error, QuoteMemError):
                        self._quarantine_error(
                            entry.scope, error, owner_run_id=entry.active.owner_run_id,
                        )
                        continue
                    try:
                        if self._heartbeat_refusal_expired(entry, error):
                            expired.append(entry)
                    except Exception as refusal:
                        self._quarantine_error(
                            entry.scope, refusal, owner_run_id=entry.active.owner_run_id,
                        )
            for entry in expired:
                try:
                    self._settle_expired_lease(entry)
                except Exception as error:
                    if isinstance(error, QuoteMemError) and error.code == "reception_stale":
                        # A settlement whose reads a write overtook is no scope
                        # fault: the next pass reads the expired lease afresh.
                        self._record_refusal("heartbeat", error)
                        continue
                    self._quarantine_error(
                        entry.scope, error, owner_run_id=entry.active.owner_run_id,
                    )
            if len(rows) < _PAGE:
                return
            after = rows[-1]["key"].removeprefix("lease:finalize:")

    def _renew_held_lease(self, entry: _HeartbeatLease) -> bool:
        """Renew one due lease under the caller's held Writer mutex.

        Returns:
            False when the lease proved expired under the mutex and the caller
            must settle it; True when it was renewed, or no longer names this
            pass's owner, generation and Writer epoch.

        Raises:
            QuoteMemError: On a refused rebind or heartbeat CAS; the caller classifies
                it after releasing this exclusion.
        """
        writer = self._ready.writer
        # The borrowed age only chose to queue; the renewal decision is taken from
        # the lease as it is now.
        latest = writer.get_control(entry.selected.key)
        live = None if latest is None else ControlRowV1.model_validate(latest).value
        if not isinstance(live, PreparingFinalizeLeaseV1 | HeldFinalizeLeaseV1) or (
            live.owner_run_id, live.lease_generation, live.writer_epoch,
        ) != (
            entry.borrowed.owner_run_id, entry.borrowed.lease_generation,
            entry.borrowed.writer_epoch,
        ):
            return True
        entry.active = live
        if self._clock() - live.heartbeat_at > FINALIZE_LEASE_LIFETIME_MS:
            return False
        ticket = writer.finalize_ticket(live.owner_run_id, clock=self._clock)
        if ticket.scope_id != entry.scope.scope_id:
            raise QuoteMemError("binding_mismatch", "Heartbeat changed its exact scope")
        writer.heartbeat_finalize(ticket, clock=self._clock)
        return True

    def _heartbeat_refusal_expired(self, entry: _HeartbeatLease, error: QuoteMemError) -> bool:
        """Classify one refused renewal with the row ledger, outside the mutex.

        Returns:
            True when the refusal proves the lease expired and the caller must
            settle it; False when a later state answers the refusal and the next
            pass reads the lease afresh.

        Raises:
            QuoteMemError: The original refusal when neither classification holds, or
                a lost owner that no stop or explicit resume explains.
        """
        writer = self._ready.writer
        if error.code == "lease_lost" and not writer.is_borrow_unchanged(
            (("control", control_id(entry.selected.key)), ("run", entry.active.owner_run_id)),
            entry.token,
        ):
            # A release, takeover or pause by its worker changed the lease or its
            # owner after the borrowed read; the refusal answers that later state.
            return False
        if error.code in {"erase_fenced", "consent_fenced"} and privacy_rebase_scope(
            writer, entry.active.owner_run_id,
        ) is not None:
            # A privacy erase left the owner behind: its next invocation moves it
            # to the scope's current epochs and renews the lease.
            return False
        owner = writer.get_row("run", entry.active.owner_run_id)
        if error.code == "lease_lost" and owner is not None and (
            owner["status"] == "manual_attention"
        ):
            # The owner left execution between the borrowed read and the renewal.
            # A durable stop keeps its lease until explicit recovery; an explicit
            # resume may already have advanced it to a running attempt.
            owner_run_id = entry.active.owner_run_id
            if not self._held_stop(entry.scope, owner_run_id) and not self._owner_resumed(
                entry.scope, owner_run_id,
            ):
                raise QuoteMemError("lease_lost", "Heartbeat lost its selected failed owner")
            return False
        if error.code != "lease_lost" or (
            self._clock() - entry.active.heartbeat_at <= FINALIZE_LEASE_LIFETIME_MS
        ):
            raise error
        return True

    def _settle_expired_lease(self, entry: _HeartbeatLease) -> None:
        """Cancel, and once it joined resume, one expired owner.

        Runs after the page's renewals so one owner's cancellation cannot delay a
        lease the pass has not renewed yet.

        Raises:
            QuoteMemError: On an unclassified refusal, or when a cancelled owner does
                not join inside ``EXPIRED_JOIN_SECONDS``.
        """
        if self._held_stop(entry.scope, entry.active.owner_run_id):
            return
        if self._retire_exhausted_budget(entry.scope, entry.active.owner_run_id):
            return
        work = BenchWork(entry.scope.scope_id, entry.active.owner_run_id, "finalize")
        if not self._pause_expired_finalize(work):
            return
        paused = self._pool.pause_owner(work)
        if self._pool.pause_drained(paused):
            if not self._pool.reserve_resume(paused):
                return
            resume_finalize(self._ready.writer, paused, clock=self._clock)
            self._pool.resume_owner(paused)
        elif time.monotonic() - paused.started_at > EXPIRED_JOIN_SECONDS:
            raise QuoteMemError("service_unhealthy", "Expired owner did not join")

    def _owner_resumed(self, scope: ScopeV1, owner_run_id: str) -> bool:
        """Recognize an owner that an explicit resume advanced while this pass ran."""
        writer = self._ready.writer
        with writer.mutex:
            raw = writer.get_control(f"lease:finalize:{scope.scope_id}")
            owner = writer.get_row("run", owner_run_id)
            if raw is None or owner is None:
                return False
            value = ControlRowV1.model_validate(raw).value
            return isinstance(value, PreparingFinalizeLeaseV1 | HeldFinalizeLeaseV1) and (
                value.owner_run_id == owner_run_id and owner["status"] == "running"
                and owner["failure_code"] is None
            )

    def _pause_expired_finalize(self, work: BenchWork) -> bool:
        """Cancel one actually stale owner without changing its durable cycle or pin."""
        writer = self._ready.writer
        with writer.mutex:
            raw = writer.get_control(f"lease:finalize:{work.scope_id}")
            if raw is None:
                return False
            value = ControlRowV1.model_validate(raw).value
            if not isinstance(value, PreparingFinalizeLeaseV1 | HeldFinalizeLeaseV1) or (
                value.owner_run_id != work.run_id
                or self._clock() - value.heartbeat_at <= FINALIZE_LEASE_LIFETIME_MS
            ):
                return False
        # Never cancel or wait with a Writer mutex or read guard held. The exact
        # pool reservation keeps every other path from replacing the owner.
        try:
            paused = self._pool.pause_owner(work)
        except QuoteMemError as error:
            if error.code != "extraction_busy":
                raise
            return False
        pause_finalize(self._ready.writer, paused, clock=self._clock)
        return True

    # Lifecycle -----------------------------------------------------------------------------

    def start(self) -> None:
        """Recover real saved owners before making the controller available."""
        with self._condition:
            if self._started or self._withdrawn or self._failure.is_set():
                raise QuoteMemError("service_unhealthy", "Bench controller cannot restart")
            self._started = True
            self._starting = True
        try:
            self._owner.require_ready()
            self._pool.start()
            for name, target in (("heartbeat", self._heartbeat), ("discovery", self._schedule)):
                if name == "discovery":
                    self._recover_work()
                with self._condition:
                    thread = threading.Thread(target=target, name=f"quotemem-bench-{name}",
                                              daemon=False)
                    self._threads.append(thread)
                    thread.start()
            with self._condition:
                self._accepting = True
        except BaseException as error:
            self._record_failure("start", error)
            self._fail()
            raise
        finally:
            with self._condition:
                self._starting = False
                self._condition.notify_all()

    def search(self, request: SearchRequest, delivery: SearchDelivery) -> None:
        """Run the Search use case; it never waits for an Add owner."""
        self.require_ready()
        self._search(request, delivery)

    def require_ready(self) -> None:
        """Reject incomplete startup or a failed, withdrawn execution owner."""
        with self._condition:
            if not self._accepting or self._withdrawn or self._failure.is_set():
                raise QuoteMemError("service_unhealthy", "Bench controller is not accepting work")
        self._pool.require_ready()

    def withdraw(self) -> None:
        """Stop new background discovery, keeping admitted HTTP Adds."""
        with self._condition:
            self._withdrawn = True
            self._condition.notify_all()
        self._reception_changed.set()

    def abandon(self, deadline: float) -> bool:
        """Stop discovery and heartbeats, cancel every worker and join by the deadline.

        Used only after a drain missed its deadline. Nothing durable changes:
        cancelled work leaves its owner rows for a later recovery, as a process
        exit would. Returns whether every thread left.
        """
        self.withdraw()
        with self._condition:
            self._draining = True
            self._pool_sealed = True
            self._heartbeat_stop = True
            self._condition.notify_all()
            threads = tuple(self._threads)
        self._pool.abandon()
        for thread in threads:
            if thread.ident is not None and thread is not threading.current_thread():
                thread.join(max(0.0, deadline - time.monotonic()))
        workers_joined = self._pool.drain(deadline)
        return workers_joined and not any(thread.is_alive() for thread in threads)

    def drain(self, deadline: float) -> bool:
        """Join the existing pool only after HTTP and background discovery stop."""
        self.withdraw()
        with self._condition:
            self._draining = True
            if self._starting:
                return False
            threads = tuple(self._threads)
        # Discovery may still admit a saved owner; its actual join precedes
        # sealing the bounded descriptor pool.
        for thread in threads:
            if thread.name == "quotemem-bench-discovery" and thread.ident is not None:
                thread.join(max(0.0, deadline - time.monotonic()))
                if thread.is_alive():
                    return False
        if not self._pool_sealed:
            if self._started and not self._failure.is_set() and not self._pool.failed \
                    and not self._drain_saved_owners(deadline):
                return False
            self._pool_sealed = True
        if not self._pool.drain(deadline):
            return False
        with self._condition:
            self._heartbeat_stop = True
            self._condition.notify_all()
        for thread in threads:
            if thread.ident is not None:
                thread.join(max(0.0, deadline - time.monotonic()))
        return not any(thread.is_alive() for thread in threads)
