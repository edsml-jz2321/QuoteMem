"""Bounded execution ownership for durable bench work, without persistence authority."""

from __future__ import annotations

import math
import threading
import time
import weakref
from collections import deque
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Literal

from aml.core.errors import AMLError
from aml.core.failure_reason import FailureReason, describe, failure_reason
from aml.service.cost_attribution import attribute_cost, cost_count
from aml.store.writer_mutex import ProgressCancellation


class WorkCancellation:
    """Monotonic cooperative cancellation with callbacks outside its state lock."""

    def __init__(self) -> None:
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._callbacks: dict[object, Callable[[], None]] = {}

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    @contextmanager
    def on_cancel(self, wake: Callable[[], None]) -> Iterator[None]:
        registration = object()
        with self._lock:
            self._callbacks[registration] = wake
            cancelled = self.cancelled
        try:
            if cancelled:
                wake()
            yield
        finally:
            with self._lock:
                del self._callbacks[registration]

    def cancel(self) -> None:
        """Publish cancellation before waking subscribers; notify every subscriber."""
        with self._lock:
            if self.cancelled:
                return
            self._event.set()
            callbacks = tuple(self._callbacks.values())
        failed = False
        for wake in callbacks:
            try:
                wake()
            except BaseException:
                failed = True
        if failed:
            raise AMLError("service_unhealthy", "Worker cancellation notification failed")


@dataclass(frozen=True)
class BenchWork:
    """Content-free locator of saved reception or its real durable semantic owner.

    Reception indexes only the selected saved Add through its original service.
    A semantic owner runs its label, judge and commit children itself, never as
    separate pool work. Neither descriptor retains input or provider buffers.
    A reception invocation never runs beside another invocation of its scope.
    """

    scope_id: str
    run_id: str
    kind: Literal["finalize", "reception"]


@dataclass(frozen=True)
class BenchWorkOutcome:
    """Disposition of one drained invocation, never a durable completion claim.

    Only ``fatal`` withdraws the pool. A settled failure has a durable owner
    receipt; retry-later leaves work for its existing discovery policy. The
    optional reason contains safe coordinates only, never source data.
    """

    kind: Literal["ok", "settled_failure", "retry_later", "fatal"]
    reason: FailureReason | None = None


@dataclass
class _OwnedWork:
    work: BenchWork
    urgent: bool
    cancellation: WorkCancellation


@dataclass
class _Continuation:
    """One yielded owner's bounded reservation, runnable again after it unwinds."""

    owner: _OwnedWork
    retired: bool = False


@dataclass(frozen=True, eq=False)
class BenchWorkPause:
    """Exact pool-issued reservation; matching locators alone do not prove drainage."""

    work: BenchWork
    cancellation: WorkCancellation | None
    urgent: bool
    started_at: float


@dataclass(frozen=True, eq=False, slots=True, weakref_slot=True)
class RetiredOwnerJoined:
    """Pool-issued exact retirement identity, granting no durable resume by itself."""

    work: BenchWork


@dataclass
class _RetiredOwner:
    work: BenchWork
    joined: weakref.ReferenceType[RetiredOwnerJoined] | None = None


@dataclass(frozen=True)
class BenchWorkCapacity:
    """Immutable declared or actual descriptor and execution limits."""

    workers: int
    reserved: int
    per_user: int
    pending: int


DEFAULT_BENCH_WORK_CAPACITY = BenchWorkCapacity(workers=64, reserved=4, per_user=16, pending=256)


class BenchWorkPool:
    """Execute bounded owner descriptors with four reserved workers by default.

    Args:
        execute: Actual application executor. Returning only proves that its
            invocation drained; durable completion is checked by the controller.
        on_failure: Content-free readiness withdrawal, invoked outside the lock.
        workers: Total worker count including reserved workers.
        reserved: Workers that only take urgent owners, ones a request waits on.
        per_user: Maximum active owner invocations for one exact user scope.
        pending: Bound on queued descriptors, independent of durable backlog.

    A reception invocation excludes every other invocation of its scope, in
    both directions. Its scope's owner may be admitted, and a successor
    reception dispatched, from durable fences alone (the released lease, the
    ordered queue head) while the earlier invocation is still returning; the
    second parent callback waits here, atomically with selection, for the
    first one's actual return.

    Construction is inert. The controller rediscovers work from durable state
    when submit returns False. Cancelled requests never cancel shared work.
    Close HTTP admission and finish its Add notifications before drain seals
    submission. A timeout retains every thread and active invocation.
    """

    def __init__(
        self, execute: Callable[[BenchWork, WorkCancellation], BenchWorkOutcome | None], *,
        on_failure: Callable[[], None],
        workers: int = DEFAULT_BENCH_WORK_CAPACITY.workers,
        reserved: int = DEFAULT_BENCH_WORK_CAPACITY.reserved,
        per_user: int = DEFAULT_BENCH_WORK_CAPACITY.per_user,
        pending: int = DEFAULT_BENCH_WORK_CAPACITY.pending,
    ) -> None:
        if any(type(value) is not int or value < 1 for value in (
            workers, reserved, per_user, pending,
        )) or reserved >= workers:
            raise ValueError("Invalid bench worker capacity")
        self._execute, self._on_failure = execute, on_failure
        self._workers, self._reserved = workers, reserved
        self._per_user, self._pending_limit = per_user, pending
        self._condition = threading.Condition(threading.Lock())
        self._threads: list[threading.Thread] = []
        self._queued: dict[str, _OwnedWork] = {}
        self._active: dict[str, _OwnedWork] = {}
        self._continuations: dict[str, _Continuation] = {}
        self._paused: dict[str, BenchWorkPause] = {}
        self._retired: dict[str, _RetiredOwner] = {}
        self._resume_slots: set[str] = set()
        self._paused_urgent: set[str] = set()
        self._scopes: deque[str] = deque()
        self._started = False
        self._sealed = False
        self._failed = False
        self._failure_reason: FailureReason | None = None

    def start(self) -> None:
        """Start tracked non-daemon workers once; partial startup still needs drain."""
        with self._condition:
            if self._started or self._sealed or self._failed:
                raise AMLError("service_unhealthy", "Bench workers cannot restart")
            self._started = True
            failed: BaseException | None = None
            try:
                for number in range(self._workers):
                    thread = threading.Thread(
                        target=self._run, args=(number < self._reserved,),
                        name=f"aml-bench-{number}", daemon=False,
                    )
                    self._threads.append(thread)
                    thread.start()
            except BaseException as error:
                failed = error
        if failed is not None:
            self._fail("start", failed)
            raise AMLError("service_unhealthy", "Bench workers could not start") from None

    def _fail(
        self, site: str = "unspecified", error: BaseException | None = None,
        *, reason: FailureReason | None = None,
    ) -> None:
        with self._condition:
            notify = not self._failed
            self._failed = True
            if self._failure_reason is None:
                self._failure_reason = reason or failure_reason(site, error)
            self._sealed = True
            # Queued descriptors own no content and remain recoverable in HB.
            self._queued.clear()
            self._continuations.clear()
            self._paused.clear()
            self._retired.clear()
            self._resume_slots.clear()
            self._paused_urgent.clear()
            self._scopes.clear()
            self._condition.notify_all()
        if notify:
            try:
                self._on_failure()
            except BaseException:
                # Failure is already latched; no callback traceback is retained.
                pass

    def fail(self) -> None:
        """Retire abandoned descriptors while preserving every active invocation.

        A terminal controller failure cannot resume paused work in this process.
        Clearing its metadata reservations lets idle threads exit, but active
        callbacks remain in the ownership table until their actual return. Only
        a subsequent successful drain authorizes the runtime to close storage.
        """
        self._fail("requested")

    def require_ready(self) -> None:
        """Check worker ownership without starting or repairing any work."""
        with self._condition:
            if not self._started or self._sealed or self._failed or (
                len(self._threads) != self._workers
                or any(not thread.is_alive() for thread in self._threads)
            ):
                raise AMLError(
                    "service_unhealthy",
                    "Bench workers are unavailable" + describe(self._failure_reason),
                )

    @property
    def failed(self) -> bool:
        """Return the latched execution failure without exposing provider diagnostics."""
        with self._condition:
            return self._failed

    @property
    def failure_reason(self) -> FailureReason | None:
        """Return the first content-free failure site, type, code and top frame."""
        with self._condition:
            return self._failure_reason

    def capacity(self) -> BenchWorkCapacity:
        """Read the validated actual limits without starting another pool."""
        return BenchWorkCapacity(
            self._workers, self._reserved, self._per_user, self._pending_limit,
        )

    def _refresh_scopes(self) -> None:
        scopes = dict.fromkeys(value.work.scope_id for value in self._pending_work())
        self._scopes = deque(scope for scope in self._scopes if scope in scopes)
        self._scopes.extend(scope for scope in scopes if scope not in self._scopes)

    def _pending_work(self) -> list[_OwnedWork]:
        values = list(self._queued.values())
        values.extend(
            continuation.owner for key, continuation in self._continuations.items()
            if key not in self._active and key not in self._paused
            and not continuation.retired and not continuation.owner.cancellation.cancelled
        )
        return values

    def _retirement_reservations(self) -> set[str]:
        """Count admitted owners and live join proofs in the existing finite inventory."""
        owners = {
            value.work.run_id for values in (self._active, self._queued)
            for value in values.values()
        }
        return owners | set(self._continuations) | set(self._paused) | {
            key for key, value in self._retired.items()
            if value.joined is not None and value.joined() is not None
        }

    def _finish(self, selected: _OwnedWork) -> None:
        key = selected.work.run_id
        del self._active[key]
        continuation = self._continuations.get(key)
        if continuation is not None and (
            continuation.retired or continuation.owner.cancellation.cancelled
            or key in self._paused
        ):
            del self._continuations[key]
        # Otherwise the same reservation exposes the owner to _take again. It
        # never expands the separately bounded submit queue, and stays runnable
        # after external submission is sealed.
        self._refresh_scopes()
        self._condition.notify_all()

    def owns_scheduled(self, work: BenchWork) -> bool:
        """Observe exact scheduled ownership without admission or completion authority.

        Discovery may coalesce a descriptor already queued, executing, or holding
        a continuation. Cancellation, pause and retirement need their existing
        coordination paths. A true observation does not reserve work or prove
        durable state; the original callbacks retain all execution validation.
        """
        if not work.scope_id or not work.run_id:
            raise ValueError("Invalid durable work locator")
        with self._condition:
            if not self._started or self._sealed or self._failed:
                return False
            retired = self._retired.get(work.run_id)
            paused = self._paused.get(work.run_id)
            if retired is not None and retired.work != work or (
                paused is not None and paused.work != work
            ):
                raise AMLError("binding_mismatch", "Reserved owner identity changed")
            if retired is not None or paused is not None:
                return False
            continuation = self._continuations.get(work.run_id)
            owned = tuple(value for value in (
                self._active.get(work.run_id), self._queued.get(work.run_id),
                None if continuation is None else continuation.owner,
            ) if value is not None)
            if any(value.work != work for value in owned):
                raise AMLError("binding_mismatch", "Scheduled owner identity changed")
            return bool(owned) and not (continuation is not None and continuation.retired) and all(
                not value.cancellation.cancelled for value in owned
            )

    def submit(self, work: BenchWork, *, urgent: bool = False) -> bool:
        """Coalesce one durable owner, promoting a waited-on owner without duplicating it.

        False means the bounded descriptor queue is full. It does not discard
        durable work or establish milestone completion. ``urgent`` marks an
        owner a request waits on; only urgent owners use reserved workers.
        """
        if not work.scope_id or not work.run_id:
            raise ValueError("Invalid durable work locator")
        with self._condition:
            if not self._started or self._sealed or self._failed:
                raise AMLError(
                    "service_unhealthy",
                    "Bench work admission is closed" + describe(self._failure_reason),
                )
            retired = self._retired.get(work.run_id)
            if retired is not None:
                if retired.work != work:
                    raise AMLError("binding_mismatch", "Retired owner identity changed")
                return False
            paused = self._paused.get(work.run_id)
            if paused is not None:
                if paused.work != work:
                    raise AMLError("binding_mismatch", "Paused owner identity changed")
                if urgent:
                    self._paused_urgent.add(work.run_id)
                return False
            continuation = self._continuations.get(work.run_id)
            existing = self._active.get(work.run_id) or self._queued.get(work.run_id) or (
                None if continuation is None else continuation.owner
            )
            if existing is not None:
                if existing.work != work:
                    raise AMLError("binding_mismatch", "Durable owner identity changed")
                existing.urgent |= urgent
                if continuation is not None:
                    self._refresh_scopes()
                self._condition.notify_all()
                return True
            if len(self._retirement_reservations()) >= self._workers + self._pending_limit:
                # Live operator join proofs consume their existing reservation.
                # Do not admit work whose later failure could not be retired.
                return False
            if len(self._queued) + len(self._continuations) + len(self._resume_slots) \
                    >= self._pending_limit:
                victim = next((key for key in reversed(self._queued) if (
                    not self._queued[key].urgent
                )), None) if urgent else None
                if victim is None:
                    return False
                self._queued.pop(victim)
                self._refresh_scopes()
            self._queued[work.run_id] = _OwnedWork(work, urgent, WorkCancellation())
            if work.scope_id not in self._scopes:
                self._scopes.append(work.scope_id)
            self._condition.notify_all()
            return True

    def _take(self, reserved: bool) -> _OwnedWork | None:
        counts: dict[str, int] = {}
        receiving: set[str] = set()
        for value in self._active.values():
            counts[value.work.scope_id] = counts.get(value.work.scope_id, 0) + 1
            if value.work.kind == "reception":
                receiving.add(value.work.scope_id)
        eligible = [value for value in self._pending_work() if (
            (not reserved or value.urgent)
            and counts.get(value.work.scope_id, 0) < self._per_user
            and value.work.scope_id not in receiving
            and (value.work.kind != "reception" or value.work.scope_id not in counts)
        )]
        if not eligible:
            return None
        urgent = any(value.urgent for value in eligible)
        allowed = {value.work.scope_id for value in eligible if value.urgent == urgent}
        while self._scopes[0] not in allowed:
            self._scopes.rotate(-1)
        scope = self._scopes[0]
        self._scopes.rotate(-1)
        selected = next(value for value in eligible if (
            value.work.scope_id == scope and value.urgent == urgent
        ))
        if selected.work.run_id in self._queued:
            del self._queued[selected.work.run_id]
        else:
            del self._continuations[selected.work.run_id]
        self._active[selected.work.run_id] = selected
        self._refresh_scopes()
        return selected

    def _run(self, reserved: bool) -> None:
        try:
            while True:
                with self._condition:
                    while (selected := self._take(reserved)) is None:
                        # An active owner may still yield a continuation, so a
                        # sealed pool keeps its workers until no owner remains.
                        if self._failed or self._sealed and not (
                            self._queued or self._paused or self._continuations or self._active
                        ):
                            return
                        self._condition.wait()
                failed: BaseException | None = None
                outcome: BenchWorkOutcome | None = None
                try:
                    outcome = self._execute(selected.work, selected.cancellation)
                except BaseException as error:
                    failed = error
                finally:
                    # Latch withdrawal before waiters can observe the failed
                    # invocation as drained or another worker takes its successor.
                    if failed is not None:
                        self._fail("execute", failed)
                    elif outcome is not None and outcome.kind == "fatal":
                        self._fail("execute", reason=outcome.reason)
                    with self._condition:
                        self._finish(selected)
                if failed is not None or outcome is not None and outcome.kind == "fatal":
                    return
        except BaseException as error:
            self._fail("worker_loop", error)

    def _wake(self) -> None:
        with self._condition:
            self._condition.notify_all()

    def pause_owner(self, work: BenchWork) -> BenchWorkPause:
        """Block replacement submission, then cancel only this actual owner invocation.

        The reservation is bounded by worker plus queue capacity. It contains no
        storage input and survives cancellation until the Writer proves renewal.
        No callback runs while the pool lock is held.
        """
        with self._condition:
            if not self._started or self._failed:
                raise AMLError("service_unhealthy", "Bench owner pause is unavailable")
            if work.run_id in self._retired:
                raise AMLError("extraction_busy", "Retired owner requires explicit resolution")
            prior = self._paused.get(work.run_id)
            if prior is not None:
                if prior.work != work:
                    raise AMLError("binding_mismatch", "Paused owner changed its locator")
                return prior
            if len(self._paused) >= self._workers + self._pending_limit:
                raise AMLError("service_unhealthy", "Bench owner pause capacity is exhausted")
            continuation = self._continuations.get(work.run_id)
            owned = self._active.get(work.run_id) or self._queued.get(work.run_id) or (
                None if continuation is None else continuation.owner
            )
            if owned is not None and owned.work != work:
                raise AMLError("binding_mismatch", "Paused invocation changed its owner")
            reservations = self._retirement_reservations()
            if work.run_id not in reservations and len(reservations) >= (
                self._workers + self._pending_limit
            ):
                raise AMLError("extraction_busy", "Owner pause reservations are full")
            paused = BenchWorkPause(
                work, None if owned is None else owned.cancellation,
                False if owned is None else owned.urgent, time.monotonic(),
            )
            self._paused[work.run_id] = paused
            if work.run_id in self._queued:
                self._resume_slots.add(work.run_id)
            self._queued.pop(work.run_id, None)
            if continuation is not None and work.run_id not in self._active:
                del self._continuations[work.run_id]
            self._refresh_scopes()
            self._condition.notify_all()
        if paused.cancellation is not None:
            try:
                paused.cancellation.cancel()
            except BaseException as error:
                self._fail("pause_cancel", error)
                raise
        return paused

    def paused_invocation(self, work: BenchWork, cancellation: WorkCancellation) -> bool:
        """Recognize the exact still-owned invocation reserved for cancellation.

        The pause is published under the pool lock before callbacks are signalled
        outside it. A handoff in that interval must drain into this reservation;
        waiting for the cancellation event would misclassify it as pool failure.
        This does not prove callback drainage or permit Writer renewal.
        """
        with self._condition:
            paused = self._paused.get(work.run_id)
            active = self._active.get(work.run_id)
            return not self._failed and paused is not None and paused.work == work and (
                paused.cancellation is cancellation
                and active is not None and active.work == work
                and active.cancellation is cancellation
            )

    def pause_drained(self, paused: BenchWorkPause) -> bool:
        """Validate the issued reservation and observe actual invocation removal."""
        with self._condition:
            if self._failed or self._paused.get(paused.work.run_id) is not paused:
                raise AMLError("lease_lost", "Bench owner pause is no longer authoritative")
            key = paused.work.run_id
            return key not in self._active and (
                key not in self._queued and key not in self._continuations
            )

    def reserve_resume(self, paused: BenchWorkPause) -> bool:
        """Reserve existing queue capacity before the Writer renews a drained owner.

        A paused queued descriptor keeps its original slot. An active descriptor
        needs a newly free slot; only an urgent owner may evict an ordinary
        queued descriptor, using the same durable rediscovery rule as submit.
        """
        with self._condition:
            key = paused.work.run_id
            if self._failed or self._paused.get(key) is not paused or key in self._active \
                    or key in self._continuations:
                raise AMLError("lease_lost", "Owner resume has no drained reservation")
            if key in self._resume_slots:
                return True
            if len(self._queued) + len(self._continuations) + len(self._resume_slots) \
                    >= self._pending_limit:
                urgent = paused.urgent or key in self._paused_urgent
                victim = next((candidate for candidate in reversed(self._queued) if (
                    not self._queued[candidate].urgent
                )), None) if urgent else None
                if victim is None:
                    return False
                self._queued.pop(victim)
                self._refresh_scopes()
            self._resume_slots.add(key)
            return True

    def require_resume_reserved(self, paused: BenchWorkPause) -> None:
        """Check actual drainage and the still-exclusive queue reservation."""
        with self._condition:
            key = paused.work.run_id
            if self._failed or self._paused.get(key) is not paused or (
                key in self._active or key in self._continuations
                or key in self._queued or key not in self._resume_slots
            ):
                raise AMLError("lease_lost", "Owner resume lost its drained queue reservation")

    def resume_owner(self, paused: BenchWorkPause) -> None:
        """Requeue the same drained owner with a fresh cancellation token after renewal."""
        with self._condition:
            key = paused.work.run_id
            if self._failed or self._paused.get(key) is not paused or (
                key in self._active or key in self._continuations
                or key in self._queued or key not in self._resume_slots
            ):
                raise AMLError("lease_lost", "Bench owner did not finish cancellation and join")
            del self._paused[key]
            self._resume_slots.remove(key)
            self._queued[key] = _OwnedWork(
                paused.work, paused.urgent or key in self._paused_urgent, WorkCancellation(),
            )
            self._paused_urgent.discard(key)
            if paused.work.scope_id not in self._scopes:
                self._scopes.append(paused.work.scope_id)
            self._condition.notify_all()

    @attribute_cost("wait.owner")
    def wait(self, run_id: str, *, deadline: float, cancellation: ProgressCancellation) -> bool:
        """Wait for one owner invocation without transferring request cancellation to it.

        True proves only local invocation drainage. The caller must inspect the
        actual Writer/publisher marker before reporting completion.
        """
        if not math.isfinite(deadline):
            raise ValueError("Invalid work wait deadline")
        with cancellation.on_cancel(self._wake):
            with self._condition:
                while True:
                    if cancellation.cancelled:
                        cost_count("wait.cancelled")
                        return False
                    if self._failed:
                        # The pool already latched why; a caller left holding
                        # only the code has to reproduce the whole run to learn it.
                        raise AMLError(
                            "service_unhealthy",
                            "Bench execution failed" + describe(self._failure_reason),
                        )
                    if all(run_id not in group for group in (
                        self._active, self._queued, self._paused, self._continuations,
                    )):
                        cost_count("wait.completed")
                        return True
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        cost_count("wait.deadline")
                        return False
                    self._condition.wait(min(remaining, threading.TIMEOUT_MAX))

    def cancel_scope(self, scope_id: str) -> None:
        """Fence the scope's queued owners and signal its active ones outside the lock.

        The controller must first establish the real durable fence; this method
        cannot authorize takeover, cancellation settlement or privacy deletion.
        Join remains a separate obligation even after cancellation returns.
        """
        with self._condition:
            for key, value in tuple(self._queued.items()):
                if value.work.scope_id == scope_id:
                    del self._queued[key]
            for key, paused in tuple(self._paused.items()):
                if paused.work.scope_id == scope_id:
                    del self._paused[key]
                    self._resume_slots.discard(key)
                    self._paused_urgent.discard(key)
            for key, continuation in tuple(self._continuations.items()):
                if continuation.owner.work.scope_id == scope_id:
                    continuation.retired = True
                    if key not in self._active:
                        del self._continuations[key]
            self._refresh_scopes()
            active = tuple(
                value.cancellation for value in self._active.values()
                if value.work.scope_id == scope_id
            )
            self._condition.notify_all()
        failed: BaseException | None = None
        for cancellation in active:
            try:
                cancellation.cancel()
            except BaseException as error:
                failed = error
        if failed is not None:
            self._fail("scope_cancel", failed)
            raise AMLError("service_unhealthy", "Scope worker cancellation failed")
        self._wake()

    def abandon(self) -> None:
        """Seal admission, drop every queued descriptor and cancel every active invocation.

        This is the bounded way out of a drain that missed its deadline: no
        durable state is touched, active work observes its cooperative
        cancellation at its next checkpoint, and the workers exit at their
        next queue wait. Join remains a separate obligation; a later drain
        reports whether the threads actually left.
        """
        with self._condition:
            self._sealed = True
            self._queued.clear()
            for continuation in self._continuations.values():
                continuation.retired = True
            self._continuations.clear()
            self._paused.clear()
            self._refresh_scopes()
            active = tuple(value.cancellation for value in self._active.values())
            self._condition.notify_all()
        failed: BaseException | None = None
        for cancellation in active:
            try:
                cancellation.cancel()
            except BaseException as error:
                failed = error
        if failed is not None:
            self._fail("abandon", failed)
            raise AMLError("service_unhealthy", "Worker cancellation failed during abandon")
        self._wake()

    def retire_owner(self, work: BenchWork) -> bool:
        """Retire only this durably blocked owner and signal its owned callback.

        The caller must prove the actual selected failure first. This operation
        grants no storage authority and never waits for its own calling worker;
        an active entry remains visible until its real return. A later explicit
        durable resume may submit this same owner after that entry is joined.
        False leaves an unadmitted durable failure for later rediscovery when
        live operator proofs pin all reservations. Active work retains its slot.
        """
        with self._condition:
            key = work.run_id
            retired = self._retired.get(key)
            if retired is not None and retired.work != work:
                raise AMLError("binding_mismatch", "Retired owner changed its original locator")
            continuation = self._continuations.get(key)
            active = self._active.get(key)
            owner = active or self._queued.get(key) or (
                None if continuation is None else continuation.owner
            )
            paused = self._paused.get(key)
            if (owner is not None and owner.work != work) or (
                paused is not None and paused.work != work
            ):
                raise AMLError("binding_mismatch", "Retired owner changed its actual locator")
            if retired is None:
                if len(self._retired) >= self._workers + self._pending_limit:
                    victim = next((name for name, value in self._retired.items() if (
                        (value.joined is None or value.joined() is None)
                        and name not in self._active and name not in self._queued
                        and name not in self._continuations and name not in self._paused
                    )), None)
                    if victim is None:
                        return False
                    del self._retired[victim]
                self._retired[key] = _RetiredOwner(work)
            self._queued.pop(key, None)
            self._paused.pop(key, None)
            self._resume_slots.discard(key)
            self._paused_urgent.discard(key)
            if continuation is None and active is not None:
                continuation = _Continuation(active, retired=True)
                self._continuations[key] = continuation
            if continuation is not None:
                continuation.retired = True
                if key not in self._active:
                    del self._continuations[key]
            cancellation = None if active is None else active.cancellation
            self._refresh_scopes()
            self._condition.notify_all()
        if cancellation is not None:
            try:
                cancellation.cancel()
            except BaseException as error:
                self._fail("owner_cancel", error)
                raise AMLError("service_unhealthy", "Blocked owner cancellation failed") from None
        return True

    def retired_owner_join(self, work: BenchWork) -> RetiredOwnerJoined:
        """Issue actual callback quiescence while preserving its submission fence."""
        with self._condition:
            key = work.run_id
            retired = self._retired.get(key)
            if retired is None or retired.work != work or self._failed or (
                key in self._active or key in self._queued
                or key in self._continuations or key in self._paused
            ):
                raise AMLError("extraction_busy", "Retired owner has not actually joined")
            joined = None if retired.joined is None else retired.joined()
            if joined is None:
                reservations = self._retirement_reservations()
                if key not in reservations and len(reservations) >= (
                    self._workers + self._pending_limit
                ):
                    raise AMLError("extraction_busy", "Owner join reservations are full")
                joined = RetiredOwnerJoined(work)
                retired.joined = weakref.ref(joined)
            return joined

    def validate_retired_owner_join(self, proof: RetiredOwnerJoined) -> None:
        """Validate the same issued retirement and fixed actual pool inventory."""
        with self._condition:
            self._validate_retired_owner_join(proof)

    def _validate_retired_owner_join(self, proof: RetiredOwnerJoined) -> None:
        key = proof.work.run_id
        retired = self._retired.get(key)
        if self._failed or retired is None or retired.joined is None or (
            retired.joined() is not proof
        ) or (
            retired.work != proof.work or key in self._active
            or key in self._queued or key in self._continuations or key in self._paused
        ):
            raise AMLError("lease_lost", "Retired owner join proof is not current")

    def resume_retired_owner(self, proof: RetiredOwnerJoined) -> None:
        """Release this submission fence only after the Writer resumes its durable owner."""
        with self._condition:
            self._validate_retired_owner_join(proof)
            del self._retired[proof.work.run_id]
            self._condition.notify_all()

    def wait_scope(self, scope_id: str, *, deadline: float) -> bool:
        """Wait for the scope's actual owner invocations after the controller's durable fence.

        This includes queued descriptors and continuations. Cancellation alone
        never proves join; no Writer mutex, reader guard or provider buffer may be
        held by callers. A latched failure remains observable through
        failed/require_ready even after this method confirms drainage.
        """
        if not math.isfinite(deadline):
            raise ValueError("Invalid scope drain deadline")
        with self._condition:
            while any(
                value.work.scope_id == scope_id
                for values in (self._active, self._queued) for value in values.values()
            ) or any(
                continuation.owner.work.scope_id == scope_id
                for continuation in self._continuations.values()
            ):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(min(remaining, threading.TIMEOUT_MAX))
            return True

    def drain(self, deadline: float) -> bool:
        """Seal admission, finish queued work, and join actual threads by deadline.

        Call only after complete HTTP drainage. Repeated attempts retain the
        same workers. This closes no publisher, Writer, Context or file owner.
        """
        if not math.isfinite(deadline):
            raise ValueError("Invalid worker drain deadline")
        with self._condition:
            self._sealed = True
            self._condition.notify_all()
            threads = tuple(self._threads)
        if threading.current_thread() in threads:
            raise AMLError("reader_busy", "A worker cannot join its own pool")
        for thread in threads:
            if thread.ident is not None:
                thread.join(max(0.0, min(deadline - time.monotonic(), threading.TIMEOUT_MAX)))
        with self._condition:
            return not any(thread.is_alive() for thread in threads) and not (
                self._active or self._paused or self._continuations or self._queued
            )
