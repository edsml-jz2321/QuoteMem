"""The Add controller factory the runtime and CLI take as a parameter.

``service/runtime.py`` builds its controller through a ``ControllerFactory``
once the workspace is ready. ``add_controller_factory`` binds the owner's
composed stages and saved-owner recovery into one, so the runtime and the CLI
construct an ``AddController`` without knowing its stages. Each part is itself
built per workspace: the owner and recovery need the ready workspace, its read
model and the exact-user authority. Search is the runtime's: it builds the one
``SearchUseCase`` over its single ``DeliveryQueue`` and passes it in as
``search=``, which the controller admits into unchanged.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

from quotemem.api.bench_service import SearchOperation
from quotemem.core.bootstrap import InitializedWorkspace
from quotemem.core.controls import RequestCompletionPolicyV2
from quotemem.ingestion.add_owner import OwnerExecutor
from quotemem.ingestion.add_scheduler import AddController, RecoveryFactory
from quotemem.ingestion.extraction_stage import ExtractionStage
from quotemem.ingestion.judge_input import MemberInputObserver
from quotemem.ingestion.owner_measure import measure_unit
from quotemem.ingestion.owner_pipeline import ComposedOwner
from quotemem.ingestion.owner_recovery import RecoveryStore, recovery_factory
from quotemem.ingestion.owner_rounds import CommitObserver, OwnerRound, OwnerRounds
from quotemem.ingestion.owner_session import Measure, WriterOwnerSessions
from quotemem.service.auth import BenchExactUserAuth
from quotemem.service.bench_workers import DEFAULT_BENCH_WORK_CAPACITY, BenchWorkCapacity
from quotemem.service.request_measurement import MeasurementCollector
from quotemem.service.runtime import BenchController, ControllerFactory
from quotemem.store.read_model import StoreReadModel
from quotemem.store.recovery_store import WriterRecoveryStore

type OwnerFactory = Callable[
    [InitializedWorkspace, StoreReadModel, BenchExactUserAuth, MeasurementCollector | None],
    OwnerExecutor,
]
"""Build the workspace's composed owner (``owner_pipeline.ComposedOwner``)."""

@dataclass(frozen=True, slots=True)
class AddControllerFactory:
    """A ``ControllerFactory`` that builds the workspace's ``AddController``.

    Attributes:
        owner: Builds the composed owner, observed by the runtime's measurement.
        recovery: Saved-owner recovery (``RecoveryFactory``).
        capacity: The worker pool's bounds.
        clock: Trusted millisecond action clock, or None for wall time.
        completion_work_budget_ms: Optional fresh-scope owner allowance.
    """

    owner: OwnerFactory
    recovery: RecoveryFactory | WriterRecovery
    capacity: BenchWorkCapacity = DEFAULT_BENCH_WORK_CAPACITY
    clock: Callable[[], int] | None = None
    completion_work_budget_ms: int | None = None

    def __call__(
        self, ready: InitializedWorkspace, publisher: StoreReadModel,
        auth: BenchExactUserAuth, /, *, search: SearchOperation,
        measurement: MeasurementCollector | None = None,
    ) -> BenchController:
        """Build the inert controller; ``start`` recovers saved owners before activation.

        Args:
            ready: The initialized workspace.
            publisher: Its read model.
            auth: The exact-user authority.
            search: The runtime's one search operation; the controller admits
                every search into exactly it.
            measurement: The runtime's optional bounded observer.

        Returns:
            The workspace's ``AddController``.
        """
        owner = self.owner(ready, publisher, auth, measurement)
        capacity = self.capacity
        recovery = self.recovery.bind(ready, auth) if isinstance(
            self.recovery, WriterRecovery,
        ) else self.recovery
        if self.clock is None:
            return AddController(
                ready, publisher, auth, owner=owner, search=search, recovery=recovery,
                workers=capacity.workers, reserved=capacity.reserved,
                per_user=capacity.per_user, pending=capacity.pending,
                measurement=measurement,
                completion_work_budget_ms=self.completion_work_budget_ms,
            )
        return AddController(
            ready, publisher, auth, owner=owner, search=search, recovery=recovery,
            workers=capacity.workers, reserved=capacity.reserved,
            per_user=capacity.per_user, pending=capacity.pending, clock=self.clock,
            measurement=measurement, completion_work_budget_ms=self.completion_work_budget_ms,
        )


@dataclass(frozen=True)
class WriterRecovery:
    """The explicit recovery over the workspace's Writer (``WriterRecoveryStore``).

    Pass it as ``add_controller_factory(recovery=...)``; the factory binds it to
    the initialized workspace and its authority.

    Attributes:
        clock: Trusted millisecond clock, or None for wall time.
        work_budget_ms: Optional allowance for each recovery.
    """

    clock: Callable[[], int] | None = None
    work_budget_ms: int | None = None

    def bind(self, ready: InitializedWorkspace, auth: BenchExactUserAuth) -> RecoveryFactory:
        """The scheduler's ``RecoveryFactory`` over this workspace's Writer."""
        clock = self.clock or (lambda: time.time_ns() // 1_000_000)

        def store() -> RecoveryStore:
            return WriterRecoveryStore(ready.writer, auth, clock=clock,
                                       work_budget_ms=self.work_budget_ms)

        return recovery_factory(store)


def add_controller_factory(
    *, owner: OwnerFactory, recovery: RecoveryFactory | WriterRecovery,
    capacity: BenchWorkCapacity = DEFAULT_BENCH_WORK_CAPACITY,
    completion_work_budget_ms: int | None = None,
) -> ControllerFactory:
    """Bind the Add owner's parts into the runtime's ``ControllerFactory``.

    Args:
        owner: Builds the composed owner.
        recovery: Explicit recovery: ``WriterRecovery()`` in production, or a
            prebuilt ``RecoveryFactory``.
        capacity: The worker pool's bounds.
        completion_work_budget_ms: Optional fresh-scope owner allowance.

    Returns:
        The factory ``BenchRuntime`` and the CLI take.
    """
    if completion_work_budget_ms is not None:
        # Refuse an invalid allowance before any owner is constructed.
        RequestCompletionPolicyV2(work_budget_ms=completion_work_budget_ms)
    return AddControllerFactory(
        owner, recovery, capacity=capacity,
        completion_work_budget_ms=completion_work_budget_ms,
    )


def composed_owner_factory(
    *, measure: Measure = measure_unit, clock: Callable[[], int] | None = None,
    observe: MemberInputObserver | None = None, commits: CommitObserver | None = None,
) -> OwnerFactory:
    """Build the production ``OwnerFactory``: the composed owner over the real Writer.

    The owner's session (``WriterOwnerSessions``) holds, freezes and pins over the
    workspace's Writer and read model, labels with the label stage, extracts with
    ``ExtractionStage`` and judges and commits with ``OwnerRounds`` (the
    round, the judge, the commit), with the workspace's two semantic providers.

    Args:
        measure: A unit's worst admission transaction.
        clock: Trusted millisecond action clock, or None for wall time.
        observe: Sees each reviewed member input with its pin (the recorded-replay
            harness); None for none.
        commits: Sees each round's membership and each component's fence as the
            commit runs (for the recorded replay); None for none.

    Returns:
        The factory ``add_controller_factory(owner=...)`` takes.
    """

    def build(
        ready: InitializedWorkspace, publisher: StoreReadModel, auth: BenchExactUserAuth,
        measurement: MeasurementCollector | None,
    ) -> OwnerExecutor:
        providers = ready.providers
        sessions = WriterOwnerSessions(
            ready.writer, publisher, auth, providers.inference, providers.embedding, measure,
        ) if clock is None else WriterOwnerSessions(
            ready.writer, publisher, auth, providers.inference, providers.embedding, measure,
            clock,
        )
        rounds = OwnerRounds(providers.inference, observe=observe, commits=commits)
        return ComposedOwner[OwnerRound](
            sessions, ExtractionStage(), rounds, rounds, measurement,
        )

    return build

