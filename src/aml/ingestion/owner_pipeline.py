"""The composed Add owner (C.2; I15): one invocation's stages in order, with its phases.

``ComposedOwner`` is the ``OwnerExecutor`` the scheduler runs. Each invocation
opens the owner's session (its held lease, frozen prefix, policy and pins) and
runs C.2's stages in order:

0. ``labels``: the frozen turn labels of the prefix (S2a, B.3a);
1-3. ``extraction`` / ``embedding`` / ``commit``: S3c's ``ExtractionStage.run``
   up to the settlement transaction, I8 marking its own boundaries;
4-6. ``judge_round_1``: S4a's nomination and S4b's judge for round 1, then the
   round's ``commit`` (S5b);
7. ``judge_round_2`` and its ``commit``, when round 1 left a second round, then
   each settle round the fence leaves after it, timed as round 2
   (``core/review_rounds``);
8. completion and release (``record_add_completion``, ``release_finalize``),
   with the rounds' census when this invocation ran C.2 from its start.

For a cohort, steps 4-7 finish one frozen source request before the next
request's round 1 reads a fresh publication pin. Extraction, settlement and
completion stay once per owner, re-entry is selected only in the first request,
and every position is reviewed at most once per round, in at most ``MAX_ROUND`` rounds.

Every stage stays behind a typed port so a stage producer that has not landed
yet is rejected by ``require_ready`` before any scheduler worker starts. The
judge is typed against S4b's ``JudgeStage.review`` (``JudgeReview``).

A whole-output fault of a unit (B.7.1) refuses the owner's budget and stops it
(``OwnerStopped("whole_output")``); a provider failure pauses it
(``OwnerStopped("provider")``), recorded with the stage that failed: the judge
for a failure in a judge round. Any other error propagates to the scheduler's
C.16 classes. Nothing here grants storage authority: the Writer re-proves the
owner, its lease and its fences before every effect.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Protocol

from aml.core.add_outcome import (
    OwnerPhaseSecondsV1,
    ReentryCensusV1,
    ReviewedPositionsV1,
    Round2CausesV1,
    Round2CensusV1,
)
from aml.core.errors import AMLError
from aml.core.extraction_bounds import ExtractionRefusalDetailV1
from aml.core.extraction_records import ExtractionRefused
from aml.core.identity import ScopeId
from aml.core.provider_budget import CallAllowanceEnded
from aml.core.review_rounds import RoundNo
from aml.core.timing import FinalizeCompletion
from aml.ingestion.add_owner import OwnerExecutor, OwnerPhases, OwnerStalled, OwnerStopped
from aml.ingestion.extraction_stage import (
    ExtractionOutcome,
    ExtractionOwner,
    Phase,
    UnitCallAccounting,
)
from aml.ingestion.sources import LabelView
from aml.providers.adapters import ProviderError
from aml.service.request_measurement import MeasurementCollector
from aml.store.source_labels import SourceLabelsPaused
from aml.store.writer_mutex import ProgressCancellation

type Round = RoundNo


@dataclass(frozen=True, slots=True)
class RoundsCensus:
    """The judge rounds' part of the owner census (I14; ingest-Q14), every count measured.

    Attributes:
        reentry: Round 1's re-entry selection by action (C.3).
        pending_selectable: The selection's pre-cap population.
        round1_members: Every member of round 1, reviewed or not (R8-4's denominator).
        round2: Round 2's members by cause, how many ended ``target_changed``
            and its physical judge requests.
        reviewed: The positions each round sent in a physical judge request.
        judge_refused_members: The members of both rounds still without an
            accepted verdict after their solo resend (C.8 #4).
    """

    reentry: ReentryCensusV1
    pending_selectable: int
    round1_members: int
    round2: Round2CensusV1
    reviewed: ReviewedPositionsV1
    judge_refused_members: int


def _combined_census(censuses: Sequence[RoundsCensus]) -> RoundsCensus:
    """Combine disjoint request rounds without losing unavailable measurements."""
    if len(censuses) == 1:
        return censuses[0]

    def total(values: Sequence[int | None]) -> int | None:
        return None if any(value is None for value in values) else sum(
            value or 0 for value in values)

    reentry = ReentryCensusV1.model_validate({
        name: total([getattr(census.reentry, name) for census in censuses])
        for name in ReentryCensusV1.model_fields
    })
    causes = [census.round2.by_cause for census in censuses]
    by_cause = None if any(value is None for value in causes) else Round2CausesV1.model_validate({
        name: sum(getattr(value, name) for value in causes if value is not None)
        for name in Round2CausesV1.model_fields
    })
    reviewed = ReviewedPositionsV1.model_validate({
        name: None if any(getattr(census.reviewed, name) is None for census in censuses) else
        tuple(sorted({key for census in censuses for key in getattr(census.reviewed, name)}))
        for name in ReviewedPositionsV1.model_fields
    })
    return RoundsCensus(
        reentry, sum(census.pending_selectable for census in censuses),
        sum(census.round1_members for census in censuses),
        Round2CensusV1(
            members=total([census.round2.members for census in censuses]), by_cause=by_cause,
            target_changed=total([census.round2.target_changed for census in censuses]),
            judge_requests=total([census.round2.judge_requests for census in censuses]),
        ), reviewed, sum(census.judge_refused_members for census in censuses),
    )


class JudgeReview[M, J](Protocol):
    """S4b's judge for one owner, typed against ``JudgeStage.review`` until it lands.

    ``M`` is S4b's ``ReviewMemberV1`` and ``J`` its ``JudgedMemberV1``.
    """

    def review(self, members: Sequence[M], round: Round) -> tuple[J, ...]:
        """Review one round's members and return one outcome per member, in order."""
        ...


class RoundStage[R](Protocol):
    """One judge round of the owner: S4a's selection and nomination, then S4b's review.

    ``R`` is the round's result, which the round's commit and round 2 consume.
    """

    def batches(
        self, session: OwnerSession, outcome: ExtractionOutcome,
    ) -> Sequence[ExtractionOutcome]:
        """Partition fresh review by frozen source request, keeping global identities."""
        ...

    def review(
        self, session: OwnerSession, outcome: ExtractionOutcome, round: Round, previous: R | None,
        *, reentry: bool = True,
    ) -> R:
        """Select or rebuild, nominate and judge one round's members (C.3-C.8).

        Args:
            session: The invocation's owner, with its held ticket and pinned view.
            outcome: The extraction stage's proposals, settlement and members.
            round: The round.
            previous: The previous round's result when ``round`` is 2 or later.
            reentry: Select earlier-owner pending positions only for the first
                source request of this invocation; later requests select none.

        Returns:
            The round's judged members.
        """
        ...

    def next_round(self, previous: R) -> Round | None:
        """The round the previous round left members for, or None (C.13, C.11.2).

        Round 1 may leave round 2; round 2 and each settle round leave the next
        settle round the members their fence sent on, up to ``MAX_ROUND``.
        """
        ...

    def census(self, round_1: R, later: Sequence[R]) -> RoundsCensus:
        """Count what the rounds selected, reviewed and sent to round 2 (I14).

        Args:
            round_1: Round 1's result.
            later: Round 2's result and each settle round's after it, in order;
                empty when round 2 did not run.

        Returns:
            The rounds' census.
        """
        ...


class RoundCommit[R](Protocol):
    """S5b's commit of one judged round: resolutions, rejections and components (C.9-C.11)."""

    def commit(self, session: OwnerSession, outcome: ExtractionOutcome, judged: R) -> None:
        """Commit the round's judged members; nothing commits twice on a rerun."""
        ...


class OwnerSession(Protocol):
    """One invocation's owner: the held lease, its frozen inputs and its durable stops."""

    @property
    def scope_id(self) -> ScopeId:
        """The owner's scope."""
        ...

    @property
    def cycle(self) -> int:
        """The owner's lease generation."""
        ...

    @property
    def whole_run(self) -> bool:
        """Whether C.2 started in this process, so the phase sums cover the whole run."""
        ...

    @property
    def accounting(self) -> UnitCallAccounting:
        """The owner's per-unit call accounting (``OwnerUnitAccounting``)."""
        ...

    def labels(self) -> LabelView:
        """Run or replay the owner's label stage (S2a, B.3a) and return its frozen view."""
        ...

    def extraction_owner(
        self, labels: LabelView, accounting: UnitCallAccounting,
        mark_phase: Callable[[Phase], None],
    ) -> ExtractionOwner:
        """Bind the frozen inputs, the labels and the phase marker for I8."""
        ...

    def at_review_pin[T](self, body: Callable[[], T]) -> T:
        """Run a later request's rounds at the owner's current publication pin."""
        ...

    def refuse(self, unit_no: int, refusal: ExtractionRefusalDetailV1) -> None:
        """Refuse the owner's budget on a unit's whole-output fault (``refuse_budget``)."""
        ...

    def pause(self, cause: ProviderError | SourceLabelsPaused, *, judge: bool = False) -> None:
        """Pause the owner on a provider failure of its own stage or its label child (C.16).

        ``judge`` names a failure of a judge round, which pauses with kind ``judge``.
        """
        ...

    def complete(self, phases: OwnerPhaseSecondsV1, outcome: ExtractionOutcome,
                 rounds: RoundsCensus | None = None) -> FinalizeCompletion:
        """Record the completion and release the owner (C.2 step 8).

        The completion reads the owner's settled positions from its durable
        records; ``outcome`` adds the positions admission refused and the units a
        provider refusal ended. ``rounds`` is the rounds' census, None when
        this invocation's rounds do not cover the owner's whole run (INV H09).
        """
        ...


class OwnerSessions(Protocol):
    """Run one invocation inside its owner session; ending it never releases the lease.

    The session is a callback, not a context manager: the owner's pinned Add view
    exists only inside the read model's registered callback
    (``SnapshotReadGuard.consume_add_view``), so the body runs inside it.
    """

    def run[T](
        self, owner_run_id: str, body: Callable[[OwnerSession], T], *,
        cancellation: ProgressCancellation, concurrency: int,
    ) -> T:
        """Re-prove the held lease, bind the owner's frozen inputs and run ``body`` once."""
        ...

    def require_ready(self) -> None:
        """Reject a missing producer (labels, providers, read model) before any worker."""
        ...


class ExtractionRun(Protocol):
    """S3c's ``ExtractionStage``: I8 up to the settlement transaction."""

    def run(self, owner: ExtractionOwner) -> ExtractionOutcome:
        """Run the owner's units, settle, embed and write its proposals."""
        ...


@dataclass(frozen=True, slots=True)
class ComposedOwner[R]:
    """The Add owner's ``OwnerExecutor``: C.2's stages in order, each in its phase.

    Attributes:
        sessions: Opens each invocation's owner session (S2b).
        extraction: S3c's ``ExtractionStage``.
        rounds: S4a's nomination and S4b's judge per round, or None before they land.
        commit: S5b's round commit, or None before it lands.
        measurement: The runtime's optional observer, for the phase tags.
    """

    sessions: OwnerSessions
    extraction: ExtractionRun
    rounds: RoundStage[R] | None
    commit: RoundCommit[R] | None
    measurement: MeasurementCollector | None = None

    def require_ready(self) -> None:
        """Reject a stage producer that has not landed before any scheduler worker starts.

        Raises:
            AMLError: ``unsupported_variant`` without the judge or commit stage.
        """
        if self.rounds is None or self.commit is None:
            raise AMLError("unsupported_variant", "The Add owner's judge or commit stage is absent")
        self.sessions.require_ready()

    def run(
        self, owner_run_id: str, *, cancellation: ProgressCancellation, concurrency: int,
    ) -> FinalizeCompletion | OwnerStopped | OwnerStalled:
        """Run one invocation through C.2 until it releases, stops durably or raises.

        Args:
            owner_run_id: The admitted owner whose lease this process holds.
            cancellation: The invocation's own cancellation.
            concurrency: The scope's worker bound (``per_user``).

        Returns:
            The verified release, the durable stop, or ``OwnerStalled`` when an
            extraction call's own window ended (``CallAllowanceEnded``). The read
            view keeps only an exception's code, so the stall leaves it as a value,
            as a provider pause does.

        Raises:
            AMLError: ``unsupported_variant`` without a stage producer;
                ``binding_mismatch`` for a refusal whose unit is unknown; any
                stage's scoped refusal or integrity failure.
        """
        rounds, commit = self.rounds, self.commit
        if rounds is None or commit is None:
            raise AMLError("unsupported_variant", "The Add owner's judge or commit stage is absent")

        def body(session: OwnerSession) -> FinalizeCompletion | OwnerStopped | OwnerStalled:
            phases = OwnerPhases(
                scope_id=session.scope_id, owner_run_id=owner_run_id, cycle=session.cycle,
                measurement=self.measurement,
            )
            accounting = session.accounting
            judging = False
            try:
                with phases.phase("labels"):
                    labels = session.labels()
                with phases.marking() as mark:
                    outcome = self.extraction.run(
                        session.extraction_owner(labels, accounting, mark),
                    )
                censuses: list[RoundsCensus] = []

                def review_request(selected: ExtractionOutcome, *, reentry: bool) -> None:
                    nonlocal judging
                    judging = True
                    with phases.phase("judge_round_1"):
                        round_1 = rounds.review(session, selected, 1, None, reentry=reentry)
                    judging = False
                    with phases.phase("commit"):
                        commit.commit(session, selected, round_1)
                    # Round 2, then the settle rounds the fence leaves (core/review_rounds).
                    later: list[R] = []
                    previous = round_1
                    while (after := rounds.next_round(previous)) is not None:
                        judging = True
                        with phases.phase("judge_round_2"):
                            previous = rounds.review(session, selected, after, previous)
                        judging = False
                        with phases.phase("commit"):
                            commit.commit(session, selected, previous)
                        later.append(previous)
                    if session.whole_run:
                        censuses.append(rounds.census(round_1, later))

                for index, selected in enumerate(rounds.batches(session, outcome)):
                    if index == 0:
                        review_request(selected, reentry=True)
                    else:
                        session.at_review_pin(partial(review_request, selected, reentry=False))
                # An adopted or resumed owner's rounds miss its earlier invocations.
                census = _combined_census(censuses) if session.whole_run else None
            except ExtractionRefused as refused:
                if refused.unit_no is None:
                    raise AMLError(
                        "binding_mismatch", "A whole-output refusal lost its unit",
                    ) from refused
                session.refuse(refused.unit_no, ExtractionRefusalDetailV1(rule=refused.fault))
                return OwnerStopped(owner_run_id, "whole_output")
            except (ProviderError, SourceLabelsPaused) as cause:
                session.pause(cause, judge=judging)
                return OwnerStopped(owner_run_id, "provider")
            except CallAllowanceEnded:
                return OwnerStalled(owner_run_id)
            return session.complete(
                phases.seconds(whole_run=session.whole_run), outcome, census,
            )

        return self.sessions.run(
            owner_run_id, body, cancellation=cancellation, concurrency=concurrency,
        )


def composed_owner_is_an_executor(owner: ComposedOwner[object]) -> OwnerExecutor:
    """Prove statically that the composed owner satisfies the scheduler's port."""
    return owner
