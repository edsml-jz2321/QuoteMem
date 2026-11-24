"""Content-free resource evidence backing the existing MeasurementCollector.

This object has no independent runtime attachment or seal authority. Its caller
installs all declared hooks before the registered user load, excluding startup
canaries, and finishes it only through the collector after actual owner joins.
The benchmark consumer retains these scalar records as witnesses before building
quotemem.bench.resources artifacts. Usage requires an actual validated boundary value;
missing usage and billing evidence remain unknown, never estimated zero or cost.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass, replace
from typing import Literal

from pydantic import TypeAdapter, ValidationError

from quotemem.core.identity import ULID, ScopeId
from quotemem.core.provider_diagnostics import ProviderAttemptV1, ProviderErrorClass, TimeoutPhase
from quotemem.core.timing import FinalizeCompletion
from quotemem.service.cost_attribution import (
    OWNER_PHASES,
    RESOURCE_PHASES,
    CostAttribution,
    CostCaptureMode,
    ResourcePhase,
)

type ResourceChannel = Literal["provider", "session", "cycle", "drain"]
type ProviderOutcome = Literal["succeeded", "failed", "cancelled"]
type CycleTrigger = Literal["first_search", "idle", "late_add", "add_completion"]
type PendingStage = Literal["queued"]
type SessionEvent = Literal[
    "observed", "reception", "settled", "execution_started", "execution_finished",
]

_CHANNELS: tuple[ResourceChannel, ...] = ("provider", "session", "cycle", "drain")
_SCOPE: TypeAdapter[str] = TypeAdapter(ScopeId)
_RUN: TypeAdapter[str] = TypeAdapter(ULID)


def _count(value: object) -> bool:
    return type(value) is int and value >= 0


def _identity(value: object, adapter: TypeAdapter[str]) -> bool:
    try:
        adapter.validate_python(value, strict=True)
        return True
    except ValidationError:
        return False


@dataclass(frozen=True, slots=True)
class ResourceUsage:
    """Actual verified token scalars retained in the invocation's usage witness."""

    input_tokens: int | None
    output_tokens: int | None


@dataclass(frozen=True, slots=True)
class ResourceCycle:
    """The existing durable finalize identity, without a second cycle owner."""

    scope_id: ScopeId
    owner_run_id: ULID
    finalize_cycle: int


@dataclass(frozen=True, slots=True)
class ProviderInvocation:
    """One adapter transport entry; retries have distinct collection-local numbers.

    A missing outcome means its return was not observed. Missing HTTP metadata
    means the transport did not produce a reliable attempt observation. A valid
    HTTP response with rejected provider output remains a failed adapter call.
    No request, output, response, exception or credential is retained.
    """

    number: int
    scope_id: ScopeId
    phase: ResourcePhase
    provider_snapshot: str
    cycle: ResourceCycle | None
    started_at_ns: int
    run_id: ULID | None = None
    finished_at_ns: int | None = None
    outcome: ProviderOutcome | None = None
    usage: ResourceUsage | None = None
    http_attempt_number: int | None = None
    http_outcome: ProviderErrorClass | Literal["response"] | None = None
    http_status_code: int | None = None
    http_timeout_phase: TimeoutPhase | None = None


@dataclass(frozen=True, slots=True)
class CycleObservation:
    """Original admission trigger and actual completion of one owner cycle."""

    identity: ResourceCycle
    trigger: CycleTrigger
    admitted_at_ns: int
    completed_at_ns: int | None = None


@dataclass(frozen=True, slots=True)
class SessionObservation:
    """One durable session's observed state after an actual owner transition.

    ``queued`` means received work its owner has not settled; None means no
    remaining durable work. Executing takes precedence in simultaneous counts
    while an actual extraction callback still owns the session; an owner's
    control callback is not an extracting session.
    """

    scope_id: ScopeId
    session_hash: str
    pending: PendingStage | None
    executing: bool
    at_ns: int
    event: SessionEvent = "observed"
    run_id: ULID | None = None
    reception_commit_seq: int | None = None
    execution_number: int | None = None


@dataclass(frozen=True, slots=True)
class SessionExecution:
    """Identity token for one actual callback, without owning or cancelling work."""

    number: int
    scope_id: ScopeId
    session_hash: str
    run_id: ULID
    started_at_ns: int


@dataclass(frozen=True, slots=True)
class PendingObservation:
    """Simultaneous distinct durable sessions at the highest observed total."""

    queued: int
    executing: int

    @property
    def total(self) -> int:
        """Total durable pending sessions, independent of descriptor capacity."""
        return self.queued + self.executing


@dataclass(frozen=True, slots=True)
class ResourceSnapshot:
    """Bounded raw evidence; consumers retain witnesses and bind registered cohorts.

    Completeness denotes capture only, never healthy execution or milestone
    acceptance. Pending work and incomplete cycles can remain in a fully captured
    failed experiment. Drains contain only actual marker/settlement completions;
    missing cohorts are censored using the registered origin and collection end.
    """

    started_at_ns: int
    ended_at_ns: int | None
    invalid_reasons: tuple[str, ...]
    missing_channels: tuple[ResourceChannel, ...]
    provider_history_complete: bool
    cycle_history_complete: bool
    execution_history_complete: bool
    drain_history_complete: bool
    provider_calls: tuple[ProviderInvocation, ...]
    cycles: tuple[CycleObservation, ...]
    transitions: tuple[SessionObservation, ...]
    global_executing_peak: int | None
    per_scope_execution_peaks: tuple[tuple[ScopeId, int], ...]
    pending_peak: PendingObservation | None
    drains: tuple[FinalizeCompletion, ...]


class ResourceObserver:
    """Accumulate bounded scalar evidence inside the existing collector lifetime.

    Producer errors latch incomplete evidence and return without interrupting
    application work. This backing object performs no I/O or work scheduling.
    The runtime declares only fully installed capture channels: provider means
    every attempt and return, session means every durable session transition,
    cycle means admission/selection/completion, and drain means verified complete
    scope settlement. Undeclared channels remain unknown, including zero counts.

    Args:
        scope_ids: Exact registered cohorts, already mapped to opaque user scopes.
        provider_snapshots: Registered immutable provider labels, without secrets.
        started_at_ns: Monotonic start before the first registered user operation.
        capture: Fully installed channels as documented above.
        call_capacity: Positive bound on all actual provider attempts.
        session_capacity: Positive bound on distinct durable sessions.
        cycle_capacity: Positive bound on admitted owner cycles.
        transition_capacity: Positive bound on all observed session transitions.

    Raises:
        ValueError: The preregistered configuration is malformed.
    """

    def __init__(
        self, *, scope_ids: tuple[ScopeId, ...], provider_snapshots: tuple[str, ...],
        started_at_ns: int, capture: tuple[ResourceChannel, ...], call_capacity: int,
        session_capacity: int, cycle_capacity: int, transition_capacity: int,
        attribution_capacity: int | None = None,
        cost_capture_mode: CostCaptureMode = "detailed-v1",
    ) -> None:
        if (
            type(scope_ids) is not tuple or not scope_ids
            or not all(_identity(scope, _SCOPE) for scope in scope_ids)
            or len(set(scope_ids)) != len(scope_ids)
            or type(provider_snapshots) is not tuple
            or not all(type(value) is str and 0 < len(value) <= 256
                       for value in provider_snapshots)
            or len(set(provider_snapshots)) != len(provider_snapshots)
            or not _count(started_at_ns) or type(capture) is not tuple
            or not all(type(channel) is str and channel in _CHANNELS for channel in capture)
            or len(set(capture)) != len(capture)
            or any(not _count(value) or value < 1 for value in (
                call_capacity, session_capacity, cycle_capacity, transition_capacity,
            ))
        ):
            raise ValueError("Resource capture requires valid registered bounds and identities")
        self._scopes = frozenset(scope_ids)
        self._providers = frozenset(provider_snapshots)
        self._capture = frozenset(capture)
        self._started = started_at_ns
        self._ended: int | None = None
        self._latest = started_at_ns
        self._call_capacity, self._session_capacity = call_capacity, session_capacity
        self._cycle_capacity, self._transition_capacity = cycle_capacity, transition_capacity
        self._lock = threading.Lock()
        self._invalid: set[str] = set()
        self._calls: list[ProviderInvocation] = []
        self._cycles: dict[ResourceCycle, CycleObservation] = {}
        self._sessions: dict[tuple[str, str], SessionObservation] = {}
        self._transitions: list[SessionObservation] = []
        self._receptions: dict[tuple[str, str], int] = {}
        self._executions: dict[int, SessionExecution] = {}
        self._execution_counts: dict[tuple[str, str], int] = {}
        self._execution_number = 0
        self._global_peak = 0
        self._user_peaks = dict.fromkeys(scope_ids, 0)
        self._user_active = dict.fromkeys(scope_ids, 0)
        self._current_counts = PendingObservation(0, 0)
        self._pending_peak = PendingObservation(0, 0)
        self._drains: dict[str, FinalizeCompletion] = {}
        self.costs = None if attribution_capacity is None else CostAttribution(
            capacity=attribution_capacity, scopes=scope_ids, started_at_ns=started_at_ns,
            mode=cost_capture_mode,
        )

    def _event(self, channel: ResourceChannel, at_ns: int) -> bool:
        if channel not in self._capture or not _count(at_ns) or at_ns < self._started:
            self._invalid.add(f"{channel}_event_invalid")
            return False
        if self._ended is not None:
            self._invalid.add(f"{channel}_after_finish")
        self._latest = max(self._latest, at_ns)
        return True

    def _cycle_valid(self, cycle: ResourceCycle) -> bool:
        return type(cycle) is ResourceCycle and (
            type(cycle.scope_id) is str and cycle.scope_id in self._scopes
            and _identity(cycle.owner_run_id, _RUN)
            and _count(cycle.finalize_cycle) and cycle.finalize_cycle > 0
        )

    def incomplete(self, channel: ResourceChannel) -> None:
        """Record a lost producer event without retaining its exception.

        Args:
            channel: One capture channel documented on ResourceObserver.

        Returns:
            None: Only the observation's validity changes.
        """
        with self._lock:
            if type(channel) is str and channel in _CHANNELS:
                self._invalid.add(f"{channel}_producer_failed")
            else:
                self._invalid.add("unknown_producer_failed")

    def provider_started(
        self, *, scope_id: ScopeId, phase: ResourcePhase, provider_snapshot: str,
        cycle: ResourceCycle | None, at_ns: int, run_id: ULID | None = None,
    ) -> ProviderInvocation | None:
        """Observe the actual attempt immediately before its provider HTTP entry.

        Args:
            scope_id: Registered exact user scope; startup canaries are excluded.
            phase: The request phase add or search, or the owner phase that the
                Add owner set in its context (labels, extraction, embedding,
                judge_round_1, judge_round_2, commit). Each retry enters
                independently.
            provider_snapshot: Exact registered provider binding label.
            cycle: Actual admitted owner for owner phases, otherwise None.
            at_ns: Monotonic entry timestamp.
            run_id: Actual Run performing an owner-phase call (the owner or
                its child). Optional only for the request phases add and search.

        Returns:
            ProviderInvocation | None: Identity token, or None after capture failure.
        """
        with self._lock:
            if not self._event("provider", at_ns):
                return None
            if (
                type(scope_id) is not str or scope_id not in self._scopes
                or type(phase) is not str or phase not in RESOURCE_PHASES
                or type(provider_snapshot) is not str or provider_snapshot not in self._providers
                or run_id is not None and not _identity(run_id, _RUN)
                or phase in OWNER_PHASES and run_id is None
                or (phase in OWNER_PHASES) != (cycle is not None)
                or cycle is not None and (
                    not self._cycle_valid(cycle) or cycle.scope_id != scope_id
                )
            ):
                self._invalid.add("provider_binding_invalid")
                return None
            if cycle is not None:
                admitted = self._cycles.get(cycle)
                if admitted is None:
                    # Actual calls remain observable even when startup/recovery
                    # has no captured original trigger or admission timestamp.
                    self._invalid.add("cycle_admission_missing")
                elif admitted.completed_at_ns is not None or at_ns < admitted.admitted_at_ns:
                    self._invalid.add("provider_binding_invalid")
                    return None
            if len(self._calls) >= self._call_capacity:
                self._invalid.add("provider_overflow")
                return None
            call = ProviderInvocation(
                len(self._calls) + 1, scope_id, phase, provider_snapshot, cycle, at_ns, run_id,
            )
            self._calls.append(call)
            if self.costs is not None:
                try:
                    self.costs.provider_link(call.number)
                except Exception:
                    self._invalid.add("cost_attribution_incomplete")
            return call

    def provider_finished(
        self, call: ProviderInvocation | None, *, outcome: ProviderOutcome,
        at_ns: int, usage: ResourceUsage | None = None,
        transport_attempt: ProviderAttemptV1 | None = None,
    ) -> None:
        """Retain one actual return and only its independently verified token usage.

        Args:
            call: Original identity token returned by provider_started.
            outcome: succeeded for validated output, failed for rejected/error
                output, cancelled for an actually cancelled boundary invocation.
            at_ns: Monotonic boundary return timestamp.
            usage: Verified response/billing scalars, or None when unobserved.
                The consumer must retain the containing call as a usage witness.
            transport_attempt: Sanitized transport result for this exact attempt,
                or None when the attempt did not produce a reliable observation.

        Returns:
            None: No provider response or exception is retained or changed.
        """
        with self._lock:
            if not self._event("provider", at_ns):
                return
            if (
                type(call) is not ProviderInvocation or not _count(call.number)
                or not 1 <= call.number <= len(self._calls)
                or self._calls[call.number - 1] is not call
                or type(outcome) is not str or outcome not in {"succeeded", "failed", "cancelled"}
                or at_ns < call.started_at_ns
                or transport_attempt is not None and (
                    type(transport_attempt) is not ProviderAttemptV1
                )
                or usage is not None and (
                    type(usage) is not ResourceUsage
                    or usage.input_tokens is not None and not _count(usage.input_tokens)
                    or usage.output_tokens is not None and not _count(usage.output_tokens)
                    or usage.input_tokens is usage.output_tokens is None
                )
            ):
                self._invalid.add("provider_return_invalid")
                return
            self._calls[call.number - 1] = replace(
                call, finished_at_ns=at_ns, outcome=outcome, usage=usage,
                http_attempt_number=(
                    None if transport_attempt is None else transport_attempt.attempt
                ),
                http_outcome=(
                    None if transport_attempt is None else transport_attempt.outcome
                ),
                http_status_code=(
                    None if transport_attempt is None else transport_attempt.status_code
                ),
                http_timeout_phase=(
                    None if transport_attempt is None else transport_attempt.timeout_phase
                ),
            )

    def session_changed(
        self, *, scope_id: ScopeId, session_hash: str, pending: PendingStage | None,
        executing: bool, at_ns: int,
    ) -> None:
        """Observe an atomic post-transition session state from its real owner.

        Repeated Adds to one session do not create additional pending sessions.
        Producers preserve executing until the callback actually returns, even
        after cancellation, and preserve pending until durable work is settled.

        Args:
            scope_id: Registered exact user scope.
            session_hash: Existing opaque hash, never the raw session text.
            pending: queued or None as documented on SessionObservation; never
                inferred from pool descriptor eviction.
            executing: Actual extraction invocation is still owned by a worker.
            at_ns: Monotonic transition time, sampled in the owner's transition.

        Returns:
            None: Only bounded evidence and simultaneous peaks are updated.
        """
        with self._lock:
            if not self._event("session", at_ns):
                return
            if (not self._session_valid(scope_id, session_hash)
                or pending is not None and (
                    type(pending) is not str or pending != "queued"
                )
                or type(executing) is not bool
            ):
                self._invalid.add("session_binding_invalid")
                return
            key = (scope_id, session_hash)
            if self._execution_counts.get(key, 0) and not executing:
                self._invalid.add("session_execution_conflict")
                return
            self._record_session(SessionObservation(
                scope_id, session_hash, pending, executing, at_ns,
            ))

    def _session_valid(self, scope_id: ScopeId, session_hash: str) -> bool:
        return (
            type(scope_id) is str and scope_id in self._scopes
            and type(session_hash) is str
            and re.fullmatch(r"[A-Za-z0-9_-]{22}", session_hash) is not None
        )

    def _record_session(self, state: SessionObservation) -> bool:
        key = (state.scope_id, state.session_hash)
        previous = self._sessions.get(key)
        if previous is not None and state.at_ns < previous.at_ns:
            self._invalid.add("session_clock_rollback")
            return False
        if previous is None and len(self._sessions) >= self._session_capacity:
            self._invalid.add("session_overflow")
            return False
        self._sessions[key] = state
        if len(self._transitions) < self._transition_capacity:
            self._transitions.append(state)
        else:
            self._invalid.add("session_transition_overflow")
        if state.pending is not None or state.executing:
            self._drains.pop(state.scope_id, None)
        old_executing = previous.executing if previous is not None else False
        old_pending = previous.pending if previous is not None else None
        current = self._current_counts
        counts = PendingObservation(
            current.queued + int(not state.executing and state.pending == "queued")
            - int(not old_executing and old_pending == "queued"),
            current.executing + int(state.executing) - int(old_executing),
        )
        self._current_counts = counts
        self._user_active[state.scope_id] += int(state.executing) - int(old_executing)
        self._global_peak = max(self._global_peak, counts.executing)
        self._user_peaks[state.scope_id] = max(
            self._user_peaks[state.scope_id], self._user_active[state.scope_id],
        )
        if counts.total > self._pending_peak.total:
            self._pending_peak = counts
        return True

    def session_received(
        self, *, scope_id: ScopeId, session_hash: str, run_id: ULID,
        commit_seq: int, at_ns: int,
    ) -> None:
        """Observe new durable Add reception before publication or notification.

        Args:
            scope_id: Actual registered scope from the committed Add ticket.
            session_hash: Actual admitted session target, without external text.
            run_id: Actual newly committed Add Run, never an idempotent replay.
            commit_seq: Durable receipt visibility sequence assigned under Writer.
            at_ns: Monotonic postcommit observation under that same mutex.

        Returns:
            None: The session becomes queued, preserving actual active callbacks.
        """
        with self._lock:
            if not self._event("session", at_ns):
                return
            if not self._session_valid(scope_id, session_hash) or not _identity(run_id, _RUN) or (
                not _count(commit_seq) or commit_seq < 1
            ):
                self._invalid.add("session_reception_invalid")
                return
            key = (scope_id, session_hash)
            if commit_seq <= self._receptions.get(key, 0):
                self._invalid.add("session_reception_repeated")
                return
            previous = self._sessions.get(key)
            if self._record_session(SessionObservation(
                scope_id, session_hash, "queued", previous.executing if previous else False,
                at_ns, "reception", run_id, reception_commit_seq=commit_seq,
            )):
                self._receptions[key] = commit_seq

    def scope_settled(
        self, cycle: ResourceCycle, *, settled_through: int, at_ns: int,
    ) -> None:
        """Observe an owner's succeeded settlement of its frozen prefix.

        Args:
            cycle: Actual owner whose release and settlement succeeded.
            settled_through: The reception commit sequence its frozen prefix ends
                at. Sessions received later belong to the next owner and stay
                queued.
            at_ns: Monotonic postcommit observation under the same Writer mutex.

        Returns:
            None: Settle the prefix's sessions without clearing active tokens or
                issuing a drain; the existing completion event proves the latter.
        """
        with self._lock:
            if not self._event("session", at_ns):
                return
            if not self._cycle_valid(cycle) or not _count(settled_through) or (
                settled_through < 1
            ):
                self._invalid.add("session_scope_settlement_invalid")
                return
            if cycle not in self._cycles:
                self._invalid.add("cycle_admission_missing")
            for key, previous in tuple(self._sessions.items()):
                if (
                    previous.scope_id == cycle.scope_id and previous.pending is not None
                    and self._receptions.get(key, 0) <= settled_through
                ):
                    self._record_session(replace(
                        previous, pending=None, at_ns=at_ns, event="settled",
                        run_id=cycle.owner_run_id, reception_commit_seq=None,
                        execution_number=None,
                    ))

    def execution_started(
        self, *, scope_id: ScopeId, session_hash: str, run_id: ULID, at_ns: int,
    ) -> SessionExecution | None:
        """Count a distinct session while this actual extraction callback is owned.

        Args:
            scope_id: Actual callback's registered user scope.
            session_hash: Actual selected session target.
            run_id: Actual Run that performs the extraction (the Add owner).
            at_ns: Monotonic callback entry, before provider or semantic consumption.

        Returns:
            SessionExecution | None: Exact return token, or None after invalidity
                or overflow. Overlapping callbacks share one session count.
        """
        with self._lock:
            if not self._event("session", at_ns):
                return None
            if not self._session_valid(scope_id, session_hash) or not _identity(run_id, _RUN):
                self._invalid.add("session_execution_invalid")
                return None
            if self._execution_number >= self._transition_capacity:
                self._invalid.add("session_execution_overflow")
                return None
            key = (scope_id, session_hash)
            previous = self._sessions.get(key)
            if key not in self._receptions:
                self._invalid.add("session_reception_missing")
            token = SessionExecution(
                self._execution_number + 1, scope_id, session_hash, run_id, at_ns,
            )
            if not self._record_session(SessionObservation(
                scope_id, session_hash, previous.pending if previous else None, True, at_ns,
                "execution_started", run_id, execution_number=token.number,
            )):
                return None
            self._execution_number = token.number
            self._executions[token.number] = token
            self._execution_counts[key] = self._execution_counts.get(key, 0) + 1
            return token

    def execution_finished(self, token: SessionExecution | None, *, at_ns: int) -> None:
        """Release only the exact observed callback after its actual return/unwind.

        Args:
            token: Original object returned by execution_started, never a copy.
            at_ns: Monotonic actual callback return, including cancellation unwind.

        Returns:
            None: Other active callbacks keep the session executing. Duplicate,
                foreign or missing returns invalidate only the measurement.
        """
        if token is None:
            return
        with self._lock:
            if not self._event("session", at_ns):
                return
            if type(token) is not SessionExecution or not _count(token.number) or (
                self._executions.get(token.number) is not token
            ):
                self._invalid.add("session_execution_return_invalid")
                return
            key = (token.scope_id, token.session_hash)
            previous = self._sessions[key]
            count = self._execution_counts[key] - 1
            if not self._record_session(SessionObservation(
                token.scope_id, token.session_hash, previous.pending, count > 0, at_ns,
                "execution_finished", token.run_id, execution_number=token.number,
            )):
                return
            self._executions.pop(token.number)
            self._execution_counts[key] = count

    def cycle_admitted(
        self, cycle: ResourceCycle, *, trigger: CycleTrigger, at_ns: int,
    ) -> None:
        """Observe the original durable cycle admission, including idle and redo.

        Args:
            cycle: Actual scope/owner/cycle selected by the Writer.
            trigger: Original Search, idle, Add completion or late Add admission.
                Promotion does not rewrite it.
            at_ns: Monotonic admission timestamp.

        Returns:
            None: Identical recovery observations preserve the original timestamp.
        """
        with self._lock:
            if not self._event("cycle", at_ns):
                return
            if not self._cycle_valid(cycle) or type(trigger) is not str or trigger not in {
                "first_search", "idle", "late_add", "add_completion",
            }:
                self._invalid.add("cycle_binding_invalid")
                return
            old = self._cycles.get(cycle)
            if old is not None:
                if old.trigger != trigger or at_ns < old.admitted_at_ns:
                    self._invalid.add("cycle_admission_changed")
                return
            if len(self._cycles) >= self._cycle_capacity:
                self._invalid.add("cycle_overflow")
                return
            self._cycles[cycle] = CycleObservation(cycle, trigger, at_ns)
            self._drains.pop(cycle.scope_id, None)

    def ensure_cycle(self, cycle: ResourceCycle) -> bool:
        """Check original admission capture without inventing a recovered trigger.

        Args:
            cycle: Actual existing scope/owner/cycle selected by an executor.

        Returns:
            bool: Whether its original admission is already captured. Missing
                history only invalidates cycle evidence; application work proceeds.
        """
        with self._lock:
            if not self._cycle_valid(cycle):
                self._invalid.add("cycle_binding_invalid")
                return False
            if cycle not in self._cycles:
                self._invalid.add("cycle_admission_missing")
                return False
            return True

    def finalize_completed(self, event: FinalizeCompletion) -> None:
        """Reuse the verified marker event; zero worker count never completes a scope.

        Args:
            event: Existing actual executor's marker-and-settlement observation.

        Returns:
            None: Save the original cycle completion and, only with no known
                pending/executing session, its scope drain. Later work clears it.
        """
        with self._lock:
            if type(event) is not FinalizeCompletion:
                self._invalid.add("cycle_completion_invalid")
                return
            if not self._event("cycle", event.completed_at_ns):
                return
            cycle = ResourceCycle(event.scope_id, event.owner_run_id, event.finalize_cycle)
            if not self._cycle_valid(cycle) or cycle not in self._cycles:
                self._invalid.add("cycle_completion_unbound")
                return
            old = self._cycles[cycle]
            if event.completed_at_ns < old.admitted_at_ns or (
                old.completed_at_ns is not None and event.completed_at_ns < old.completed_at_ns
            ):
                self._invalid.add("cycle_clock_rollback")
                return
            if old.completed_at_ns is not None:
                return
            self._cycles[cycle] = replace(old, completed_at_ns=event.completed_at_ns)
            if any(call.cycle == cycle and call.outcome is None for call in self._calls):
                self._invalid.add("cycle_provider_return_missing")
            if "drain" in self._capture and "session" in self._capture:
                if any(
                    item.scope_id == event.scope_id and (
                        item.pending or item.executing or item.at_ns > event.completed_at_ns
                    )
                    for item in self._sessions.values()
                ):
                    self._invalid.add("drain_pending_at_marker")
                else:
                    self._drains[event.scope_id] = event

    def finish(self, at_ns: int) -> ResourceSnapshot:
        """End capture only when the enclosing MeasurementCollector seals after joins.

        Args:
            at_ns: Monotonic collection end, after all observed transitions.

        Returns:
            ResourceSnapshot: Raw evidence with explicit invalidity/completeness.
                This method does not grant drain, storage disposal or acceptance.
        """
        with self._lock:
            if not _count(at_ns) or at_ns < self._latest:
                self._invalid.add("collection_end_invalid")
            elif self._ended is None:
                self._ended = at_ns
                if self.costs is not None:
                    try:
                        cost_snapshot = self.costs.finish(at_ns) if (
                            self.costs.mode == "detailed-v1"
                        ) else self.costs.finish_summary(at_ns)
                    except Exception:
                        self._invalid.add("cost_attribution_incomplete")
                    else:
                        if cost_snapshot.invalid_reasons:
                            self._invalid.add("cost_attribution_incomplete")
            if any(call.outcome is None for call in self._calls):
                self._invalid.add("provider_return_missing")
            if any(item.executing for item in self._sessions.values()):
                self._invalid.add("session_return_missing")
            return self._snapshot()

    def observed(self) -> ResourceSnapshot:
        """Copy bounded raw evidence without independently closing collection.

        Args:
            None.

        Returns:
            ResourceSnapshot: Immutable scalars with no acceptance assertion.
        """
        with self._lock:
            return self._snapshot()

    def _snapshot(self) -> ResourceSnapshot:
        invalid = tuple(sorted(self._invalid))

        def complete(channel: ResourceChannel) -> bool:
            return self._ended is not None and channel in self._capture and not any(
                reason.startswith((channel + "_", "collection_", "unknown_"))
                for reason in invalid
            )

        return ResourceSnapshot(
            self._started, self._ended, invalid,
            tuple(channel for channel in _CHANNELS if channel not in self._capture),
            complete("provider"), complete("cycle"), complete("session"),
            complete("drain") and complete("cycle") and complete("session"),
            tuple(self._calls), tuple(self._cycles.values()), tuple(self._transitions),
            self._global_peak if "session" in self._capture else None,
            tuple(self._user_peaks.items()) if "session" in self._capture else (),
            self._pending_peak if "session" in self._capture else None,
            tuple(self._drains.values()),
        )
