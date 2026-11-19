"""The Add owner's execution interface (C.2, C.2a; I15): what the scheduler runs per owner.

One admitted owner per scope runs its frozen prefix through C.2's stages. The
scheduler (``ingestion/add_scheduler.py``) owns threads, descriptors, heartbeats
and backoff; it hands each invocation of an owner to an ``OwnerExecutor``. An
invocation ends in one of three ways:

- it returns ``FinalizeCompletion``: the owner released, its completion record
  and released marker are durable (C.2 #8);
- it returns ``OwnerStopped``: the owner stopped durably and only explicit
  recovery continues it (C.16): a whole-output refusal (B.7.1) or a provider
  pause;
- it raises: a scoped refusal, an integrity failure or a runtime failure, which
  the scheduler classifies (C.16 failure classes).

Nothing here grants storage authority. The Writer re-proves the owner, its lease
and its fences before every effect.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import Literal, Protocol, get_args

from aml.core.add_outcome import OwnerPhaseSecondsV1
from aml.core.identity import ULID, ScopeId
from aml.core.timing import FinalizeCompletion
from aml.service.cost_attribution import OwnerPhase
from aml.service.request_measurement import MeasurementCollector, resource_phase
from aml.service.resource_observer import ResourceCycle
from aml.store.writer_mutex import ProgressCancellation

type StopCause = Literal["whole_output", "provider"]
"""Why an owner stopped durably: a refused budget (B.7.1) or a provider pause (C.16)."""

OWNER_PHASES: tuple[OwnerPhase, ...] = get_args(OwnerPhase)
"""C.2's six timed phases, in stage order (E.4 ``owner_phase_seconds``; eval-Q11)."""


class OwnerPhases:
    """Sum an owner's monotonic seconds per C.2 phase and tag its work with the current phase.

    The owner marks C.2's boundaries (ingest-Q13): ``labels`` for step 0's label
    stage, ``extraction`` for I8's units up to the barrier, ``embedding`` for I8's
    B.9 batches and ``ProposalV2`` writes, ``judge_round_1`` and ``judge_round_2``
    for each round's selection or rebuild, nomination and review, and ``commit``
    for the settlement transaction, resolution and every rejection and component
    transaction. I8 marks its own inner boundaries through the same ``phase``
    callable, which the owner passes it. Admission, freeze, completion and release
    are outside the six.

    Phases nest: entering one pauses the enclosing phase's clock and makes the new
    phase the key S8a's observers read for every provider call and timing inside
    it (``resource_phase``); leaving resumes the enclosing phase. A phase entered
    several times sums its intervals. Each worker thread that I8 starts must run
    inside a copy of the caller's context to inherit the key.

    Args:
        scope_id: The owner's scope.
        owner_run_id: The owner Run.
        cycle: The owner's lease generation.
        measurement: The runtime's optional observer.
        clock: Monotonic seconds.
    """

    def __init__(
        self, *, scope_id: ScopeId, owner_run_id: ULID, cycle: int,
        measurement: MeasurementCollector | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._scope_id, self._owner_run_id = scope_id, owner_run_id
        self._cycle = ResourceCycle(scope_id, owner_run_id, cycle)
        self._measurement, self._clock = measurement, clock
        self._lock = threading.Lock()
        self._totals: dict[OwnerPhase, float] = dict.fromkeys(OWNER_PHASES, 0.0)
        self._stack: list[tuple[OwnerPhase, float]] = []

    @contextmanager
    def phase(self, name: OwnerPhase) -> Iterator[None]:
        """Time one interval of ``name`` and tag the work inside it with that phase."""
        if name not in self._totals:
            raise ValueError("Unknown owner phase")
        with self._lock:
            now = self._clock()
            if self._stack:
                outer, started = self._stack[-1]
                self._totals[outer] += now - started
            self._stack.append((name, now))
        try:
            with resource_phase(
                self._measurement, scope_id=self._scope_id, phase=name,
                run_id=self._owner_run_id, cycle=self._cycle,
            ):
                yield
        finally:
            with self._lock:
                now = self._clock()
                inner, started = self._stack.pop()
                self._totals[inner] += now - started
                if self._stack:
                    outer, _ = self._stack[-1]
                    self._stack[-1] = (outer, now)

    @contextmanager
    def marking(self) -> Iterator[Callable[[OwnerPhase], None]]:
        """Yield a marker that switches between sequential phases, for I8's ``mark_phase``.

        I8 marks its boundaries (``extraction``, ``embedding``, ``commit``) rather
        than entering a block per phase (S3c ``ExtractionOwner.mark_phase``). Each
        mark closes the phase the previous mark opened and opens the new one,
        nested inside whatever phase encloses this block; leaving the block closes
        the last marked phase. Marks must come from the thread that entered it.

        Yields:
            The marker.
        """
        stack = ExitStack()

        def mark(name: OwnerPhase) -> None:
            stack.close()
            stack.enter_context(self.phase(name))

        try:
            yield mark
        finally:
            stack.close()

    def seconds(self, *, whole_run: bool) -> OwnerPhaseSecondsV1:
        """Return the sums; every phase is None when they do not cover the owner's whole run.

        A phase never entered by an owner that ran C.2 in this process is a
        measured zero. An owner adopted after a restart lost its earlier phases'
        time, so its census reports them unavailable (INV H09), never a partial sum.

        Raises:
            RuntimeError: While a phase is still open.
        """
        with self._lock:
            if self._stack:
                raise RuntimeError("Owner phase sums are read only between phases")
            totals = dict(self._totals)
        if not whole_run:
            return OwnerPhaseSecondsV1.model_validate(dict.fromkeys(OWNER_PHASES))
        return OwnerPhaseSecondsV1.model_validate(totals)


@dataclass(frozen=True, slots=True)
class OwnerStopped:
    """An owner invocation that left a durable stop; only explicit recovery continues it.

    Attributes:
        owner_run_id: The stopped owner.
        cause: ``whole_output`` when a whole-output fault refused its budget (B.7.1,
            the refused state of ``finalize_budget``); ``provider`` when a provider
            failure paused it (its ``result_index:provider_failure``).
    """

    owner_run_id: ULID
    cause: StopCause


@dataclass(frozen=True, slots=True)
class OwnerStalled:
    """An invocation whose extraction call's own window ended without a reply (RESUMENOW).

    No later invocation of the owner's attempt can send that call again, and the
    owner made no durable stop: the scheduler ends its allowance at once, so the
    counted automatic resume starts its next attempt, which renews the call's window.

    Attributes:
        owner_run_id: The stalled owner.
    """

    owner_run_id: ULID


class OwnerExecutor(Protocol):
    """Run one admitted owner's frozen prefix through C.2 (S2b's Add owner implements it)."""

    def require_ready(self) -> None:
        """Reject an unavailable stage producer before any scheduler worker starts."""

    def run(
        self, owner_run_id: str, *, cancellation: ProgressCancellation, concurrency: int,
    ) -> FinalizeCompletion | OwnerStopped | OwnerStalled:
        """Run one invocation of the owner until it releases, stops durably or raises.

        Args:
            owner_run_id: The admitted owner whose lease this process holds.
            cancellation: The invocation's own cancellation; a cancelled invocation
                stops at its next checkpoint and its late results write nothing.
            concurrency: The scope's worker bound (``per_user``); the owner runs at
                most min(16, per_user) unit calls and judge requests at once (B.3,
                C.8 #2a).

        Returns:
            The verified release, the durable stop, or the stall of an extraction
            call whose own window ended.
        """
        ...
