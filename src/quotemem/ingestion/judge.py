"""The judge stage: one owner's round of member reviews under judge v8.

`JudgeStage.review(members, round)` takes the round's rebuilt members (the
nominations, their member inputs and persisted-ready snapshots) and returns one
`JudgedMemberV1` per member in canonical order:

1. a member whose child already holds its result (replay, recovery) keeps it
   with zero calls;
2. every other member that reuses or is sent has its judge child admitted with
   its snapshot before any dispatch;
3. a member whose ``(binding_digest, member_input_digest)`` already has an
   accepted, unscrubbed output in the scope reuses it without a request, as a
   reuse child; this is the only way the ``borrow`` happens, and an
   unavailable result is never reused;
4. the rest are packed by the pure `pack_requests` and run by
   `execute_round`; a member that fits no request alone is
   ``review_capacity_exceeded`` without a call;
5. a member whose verdict is consequential, denying it as a restated forgotten
   fact or ending an issued head (``update``, ``correct`` or
   ``retract`` on a current memory), is confirmed by fresh votes: its
   member input is sent alone again, each vote from its own judge child
   (``JudgeRecords.vote``), until two of at most three verdicts share a
   consequence, and the earliest of them is its judgment. Without a majority,
   including when a vote ends unavailable, a denial stands and a head ending is
   read ``unclear`` (``relation_unclear``). A recorded vote replays without a call.
   In the full mode, a member whose verdict a request of this round persisted
   sends its votes as soon as that verdict persists, within the round's
   concurrency, instead of after the round's slowest request; the votes, their
   requests and the decision are those the confirmation would ask after the
   round. The batched packing (targeted-v1's model stage) packs the members' votes
   after the round instead, and sends every vote the decision may still read in one
   round of votes: vote 3 starts with vote 2, not after it, for the members vote 2 has
   not decided yet, and its requests stop once vote 2 decides their members (a request
   closed in flight leaves no observed usage). The decision reads the ballots in order
   as before, so a vote it does not need is never read.

Quiet and unreviewed members never reach this stage: they make no call and
reuse no verdict. The stage settles no candidate row: resolution resolves the verdicts
and the commit stage writes the dispositions, including ``review_unavailable`` (or
``review_unavailable_repeated`` for a re-entered member whose selected code was
``review_unavailable`` and whose read timeouts are exhausted again),
which the stage only reports. A member still unaccepted after its solo resend
is always ``review_unavailable``, so later owners re-enter it.
"""

from __future__ import annotations

import contextvars
import logging
import threading
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from functools import partial
from typing import Literal, Protocol, cast

from quotemem.core.entities import PositionKey, ReasonCode
from quotemem.core.errors import QuoteMemError
from quotemem.core.identity import ArtifactId
from quotemem.core.inference import InferenceBindingV1
from quotemem.core.judge_confirmation import (
    Consequence,
    agreed_folds,
    consequence,
    consequential,
    issued_heads,
    issued_siblings,
    standing,
    unclear_endings,
)
from quotemem.core.judge_output import JudgeV8
from quotemem.core.judge_records import CONFIRMATION_VOTES
from quotemem.core.neighbor_snapshot import NeighborSnapshotV1
from quotemem.core.review_rounds import RoundNo
from quotemem.ingestion.judge_batch import (
    BATCHED_OUTPUT_TOKENS,
    Followup,
    JudgeCarrier,
    OutputBound,
    VoteFailure,
    VoteRequests,
    VotesResultV1,
    execute_round,
    execute_votes,
    plan_of,
    send_alone,
)
from quotemem.ingestion.judge_input import (
    FULL_PACKING,
    JudgeRequestPlanV1,
    MemberInputV1,
    PackingRuleV1,
    pack_requests,
)
from quotemem.ingestion.nomination import ContentBinder, NominationV1
from quotemem.providers.contracts import StructuredInferenceProvider


@dataclass(frozen=True, slots=True)
class ReviewMemberV1:
    """One member a round reviews: its nomination, input and snapshot.

    Attributes:
        nomination: the nomination at the round's pin.
        input: Its member input, built with the cited runs' frozen attribution.
        snapshot: Its ``NeighborSnapshotV1``; it names ``input.digest`` and,
            for a re-entered member, the code it was selected with.
    """

    nomination: NominationV1
    input: MemberInputV1
    snapshot: NeighborSnapshotV1

    def __post_init__(self) -> None:
        if (
            self.snapshot.member_input_digest != self.input.digest
            or self.snapshot.position.owner_run_id != self.input.position_key[0]
            or self.snapshot.position.unit_no != self.input.position_key[1]
            or self.snapshot.position.position != self.input.position_key[2]
            or self.nomination.member.position_key != self.input.position_key
        ):
            raise QuoteMemError("binding_mismatch", "A review member's snapshot or input differs")

    @property
    def position_key(self) -> PositionKey:
        """The member's position."""
        return self.input.position_key

    @property
    def repeats_unavailable(self) -> bool:
        """Whether an exhausted read timeout settles ``review_unavailable_repeated``."""
        return self.snapshot.reentry_reason_code == "review_unavailable"


@dataclass(frozen=True, slots=True)
class JudgmentV1:
    """A member's stored judgment as its child records it.

    Attributes:
        kind: ``verdict`` or ``unavailable`` (never a verdict).
        verdict: The accepted ``JudgeV8``; None when unavailable.
        reason_readable: Whether its reason is readable.
        result_ref: The ``result_index:judge``.
        borrowed: Whether the verdict was reused from an accepted output.
        refused: Whether it is unavailable because the member's solo resend was
            still not accepted, not an exhausted read timeout or spent request.
    """

    kind: Literal["verdict", "unavailable"]
    verdict: JudgeV8 | None
    reason_readable: bool
    result_ref: ArtifactId
    borrowed: bool
    refused: bool = False

    def __post_init__(self) -> None:
        if (self.kind == "verdict") != (self.verdict is not None) or (
            self.kind == "unavailable" and (self.reason_readable or self.borrowed)
        ) or (self.refused and self.kind != "unavailable"):
            raise QuoteMemError("binding_mismatch",
                                "A judgment's verdict fields differ from its kind")


@dataclass(frozen=True, slots=True)
class ReusableOutputV1:
    """An accepted, unscrubbed output of the scope for one ``(binding, input)``."""

    verdict: JudgeV8
    reason_readable: bool
    output_ref: ArtifactId
    result_ref: ArtifactId


@dataclass(frozen=True, slots=True)
class JudgedMemberV1:
    """One member's outcome of a round, in canonical order.

    Attributes:
        position_key: The member.
        judgment: Its stored judgment; None for ``review_capacity_exceeded``. After a
            confirmation it is the verdict that stands, a vote's when the votes
            overruled the first verdict; a first verdict ending a head that no majority
            confirmed keeps its record but reads each head it ends ``unclear``.
        code: The disposition the stage implies without a verdict:
            ``review_unavailable``, ``review_unavailable_repeated`` or
            ``review_capacity_exceeded``; None for a verdict, which resolution resolves.
    """

    position_key: PositionKey
    judgment: JudgmentV1 | None
    code: ReasonCode | None


class JudgeRecords(JudgeCarrier, Protocol):
    """The owner's durable judge records (``store/judge_records.py``).

    Besides the round's carrier it admits each member's child with its
    snapshot, reads a child's recorded judgment, looks up accepted outputs in
    the scope, records reuse children and gives the records of each
    confirmation vote's children.
    """

    def vote(self, number: int) -> JudgeRecords:
        """The records of confirmation vote ``number``'s children.

        Their children are admitted, sent and settled as a review's, one per
        (owner, position, round, vote); they reuse no output, write no reuse
        locator and settle an unavailable vote outside the census.
        """
        ...

    def admit(self, member: ReviewMemberV1) -> None:
        """Admit the member's child for this round with its snapshot, create-once."""
        ...

    def recorded(self, member: ReviewMemberV1) -> JudgmentV1 | None:
        """Return the judgment the member's child holds; a stored exhaustion meets the census.

        Raises:
            ProviderError: The child's exhausted read timeout is the owner's second.
        """
        ...

    def reusable(self, member: ReviewMemberV1) -> ReusableOutputV1 | None:
        """Return the scope's accepted, unscrubbed output for the member's input."""
        ...

    def reuse(self, member: ReviewMemberV1, output: ReusableOutputV1) -> JudgmentV1:
        """Record the member's reuse child for ``output`` without a request."""
        ...


_LOG = logging.getLogger(__name__)
_VOTE: ContextVar[int | None] = ContextVar("judge_confirmation_vote", default=None)


def current_judge_vote() -> int | None:
    """Name the confirmation vote this context is sending: 2 or 3, else None.

    Content-free attribution for an opted-in HTTP observer, as the recorded-replay
    transport is: a vote sends its member's solo request bytes again, so only
    this tells the votes apart.

    Returns:
        int | None: The sending vote, or None outside a confirmation vote.
    """
    return _VOTE.get()


@contextmanager
def _sending_vote(number: int) -> Iterator[None]:
    token = _VOTE.set(number)
    try:
        yield
    finally:
        _VOTE.reset(token)


def unavailable_code(member: ReviewMemberV1, judgment: JudgmentV1) -> ReasonCode:
    """The disposition of an unavailable judgment.

    A refused member stays re-enterable; only an exhaustion that repeats a re-entered
    member's ``review_unavailable`` is ``review_unavailable_repeated``.
    """
    if not judgment.refused and member.repeats_unavailable:
        return "review_unavailable_repeated"
    return "review_unavailable"


class JudgeStage:
    """The judge stage for one owner: reuse, packing, dispatch and records.

    Construction does no I/O; ``review`` performs the round. It holds no lock
    around a provider call and makes no call for replayed or reused work.
    ``requests`` lists the physical requests of the last ``review``, each as
    its members' positions, for the owner census; ``votes`` lists the
    confirmation votes it asked, as (position, vote), each once its request started;
    ``cancelled`` lists the confirmation vote requests it closed in flight, as (vote,
    positions): no reply was read for them, so their token usage is not observed, and the
    provider may still bill them. The agent service's provider counters count every
    asynchronous call cancelled after its request went out, these among them, as a call
    without usage.
    """

    def __init__(self, records: JudgeRecords, provider: StructuredInferenceProvider, *,
                 binding: InferenceBindingV1, bind: ContentBinder, owner_run_id: str,
                 packing: PackingRuleV1 = FULL_PACKING,
                 output_bound: OutputBound = BATCHED_OUTPUT_TOKENS) -> None:
        """Bind the owner's records, provider, frozen judge binding and keyed binder.

        ``packing`` is the full mode's rule, or the batched packing's
        (``judge_input.BATCHED_PACKING``): one Add's members in as few requests as the
        bounds allow, their unaccepted members and confirmation votes kept together.
        ``output_bound`` bounds the batched packing's replies (``judge_batch``'s
        ``AGENT_OUTPUT_TOKENS`` for an ``agent`` release).
        """
        if binding.purpose != "judge" or binding.concurrency is None:
            raise QuoteMemError("binding_mismatch",
                                "The judge stage needs the frozen judge binding")
        self._records, self._provider, self._binding = records, provider, binding
        self._bind, self._owner, self._packing = bind, owner_run_id, packing
        self._batched = packing != FULL_PACKING
        self._output_bound = output_bound
        self._ballots: dict[int, JudgeRecords] = {}
        self._ballots_lock = threading.Lock()
        self.requests: tuple[tuple[PositionKey, ...], ...] = ()
        self.votes: tuple[tuple[PositionKey, int], ...] = ()
        self.cancelled: tuple[tuple[int, tuple[PositionKey, ...]], ...] = ()

    def _vote_records(self, number: int) -> JudgeRecords:
        """The records of vote ``number``, one object per stage for every thread."""
        with self._ballots_lock:
            if number not in self._ballots:
                self._ballots[number] = self._records.vote(number)
            return self._ballots[number]

    def review(self, members: Sequence[ReviewMemberV1], round: RoundNo,
               ) -> tuple[JudgedMemberV1, ...]:
        """Review one round's members.

        Args:
            members: The round's members to review (``review``/``borrow``
                actions and fresh members); never quiet or unreviewed ones.
            round: The round; every member's snapshot names it.

        Returns:
            One outcome per member, in canonical order.

        Raises:
            ProviderError: A provider failure outside the bounded read-timeout
                settlement policy.
            QuoteMemError: ``binding_mismatch`` for inconsistent members; revoked
                authority or an exhausted allowance.
        """
        ordered = sorted(members, key=lambda member: member.input.order)
        with self._ballots_lock:
            self._ballots = {}
        self.cancelled = ()
        if len({member.position_key for member in ordered}) != len(ordered) or any(
            member.snapshot.round != round or member.snapshot.owner_run_id != self._owner
            for member in ordered
        ):
            raise QuoteMemError("binding_mismatch",
                                "A round's members repeat or differ in owner/round")
        judged: dict[PositionKey, JudgedMemberV1] = {}
        pending: list[ReviewMemberV1] = []
        for member in ordered:
            recorded = self._records.recorded(member)
            if recorded is None and (found := self._records.reusable(member)) is not None:
                self._records.admit(member)
                recorded = self._records.reuse(member, found)
            if recorded is None:
                pending.append(member)
                continue
            judged[member.position_key] = JudgedMemberV1(
                member.position_key, recorded,
                unavailable_code(member, recorded) if recorded.kind == "unavailable" else None,
            )
        by_position = {member.position_key: member for member in pending}
        packing = pack_requests((member.input for member in pending), self._packing)
        self.requests = tuple(tuple(item.position_key for item in plan.members)
                              for plan in packing.requests)
        for unfit in packing.unfit:
            # No request can carry it: no child, no call.
            judged[unfit.position_key] = JudgedMemberV1(
                unfit.position_key, None, "review_capacity_exceeded",
            )
        for plan in packing.requests:
            for item in plan.members:
                self._records.admit(by_position[item.position_key])
        # The batched packing packs each vote round after the round (`_vote_plans`).
        ahead = None if self._batched else _VotesAhead(self, by_position)
        execute_round(packing.requests, self._provider, self._records, binding=self._binding,
                      bind=self._bind, owner=self._owner,
                      after=None if ahead is None else ahead.release, halving=self._batched,
                      output_bound=self._output_bound)
        for plan in packing.requests:
            for item in plan.members:
                member = by_position[item.position_key]
                recorded = self._records.recorded(member)
                if recorded is None:
                    raise QuoteMemError("binding_mismatch",
                                        "A reviewed member has no recorded result")
                judged[item.position_key] = JudgedMemberV1(
                    item.position_key, recorded,
                    unavailable_code(member, recorded) if recorded.kind == "unavailable" else None,
                )
        self._confirm(ordered, judged, () if ahead is None else ahead.sent)
        return tuple(judged[member.position_key] for member in ordered)

    def _send_vote(self, member: ReviewMemberV1, number: int) -> bool:
        """Admit and send one vote alone unless its child holds a result; True when sent."""
        records = self._vote_records(number)
        if records.recorded(member) is not None:
            return False
        records.admit(member)
        with _sending_vote(number):
            send_alone(member.input, self._provider, records, binding=self._binding,
                       bind=self._bind, owner=self._owner, vote=number)
        return True

    def _vote_plans(self, missing: Sequence[ReviewMemberV1]) -> list[JudgeRequestPlanV1]:
        """A vote round's requests: each member alone, or, batched, packed together."""
        return self._input_plans([member.input for member in missing])

    def _input_plans(self, inputs: Sequence[MemberInputV1]) -> list[JudgeRequestPlanV1]:
        """`_vote_plans` over member inputs."""
        if not self._batched:
            return [plan_of([item]) for item in inputs]
        packing = pack_requests(inputs, self._packing)
        return [*packing.requests, *(plan_of([member]) for member in packing.unfit)]

    def _votes_together(self, missing: Sequence[ReviewMemberV1], number: int,
                        ballots: Mapping[PositionKey, Sequence[JudgmentV1]],
                        issued: Mapping[PositionKey, frozenset[str]],
                        ) -> tuple[list[tuple[PositionKey, int]], VotesResultV1]:
        """Send votes ``number`` to ``CONFIRMATION_VOTES`` of these members in one round.

        The batched packing's confirmation: each vote's unrecorded members are
        packed together, and all the votes run within the frozen concurrency
        (`execute_votes`), vote ``number``'s requests first, so a later vote does not wait
        for an earlier one's round. Each request is taken from the vote's unstarted plans
        as it is about to start; once the earlier votes have decided some of those members
        (`_still_open`), the undecided ones are packed anew, as the vote-by-vote round packs
        them. A request admits its members' children then, and stops once they are
        decided, so the round lasts no longer than the votes the decision reads.

        Args:
            missing: The members still undecided whose vote ``number`` is unrecorded.
            number: The first vote to send.
            ballots: Each member's verdicts before vote ``number``, the first one first.
            issued: Each member's ``issued_heads``.

        Returns:
            The (position, vote) pairs whose request started, and the round's failures
            and closed requests.
        """
        rounds: list[VoteRequests] = []
        for later in range(number, CONFIRMATION_VOTES + 1):
            records = self._vote_records(later)
            asked = [member for member in missing if records.recorded(member) is None]
            if not asked:
                continue
            with _sending_vote(later):
                context = contextvars.copy_context()
            rounds.append(VoteRequests(later, self._vote_plans(asked), records, context))
        by_position = {member.position_key: member for member in missing}
        started: list[tuple[PositionKey, int]] = []
        lock = threading.Lock()

        def needed(later: int, key: PositionKey) -> bool:
            return self._still_open(by_position[key], ballots[key], issued[key], number, later)

        def start(later: int, inputs: Sequence[MemberInputV1]) -> None:
            records = self._vote_records(later)
            for item in inputs:
                records.admit(by_position[item.position_key])
            with lock:
                started.extend((item.position_key, later) for item in inputs)

        result = execute_votes(rounds, self._provider, binding=self._binding, bind=self._bind,
                               owner=self._owner, halving=self._batched,
                               output_bound=self._output_bound, needed=needed, start=start,
                               pack=self._input_plans)
        return started, result

    def _still_open(self, member: ReviewMemberV1, ballots: Sequence[JudgmentV1],
                    issued: frozenset[str], number: int, later: int) -> bool:
        """Whether the decision may still read vote ``later`` of this member.

        ``ballots`` are its verdicts before vote ``number``. Votes ``number`` to
        ``later - 1`` are read in order as `_confirm` reads them: one not recorded yet
        keeps the confirmation open, one unavailable closes it, and `standing` deciding
        closes it.
        """
        found: list[Consequence] = [consequence(cast(JudgeV8, ballot.verdict), issued)
                                    for ballot in ballots]
        supported = [cast(JudgeV8, ballot.verdict).support == "supported" for ballot in ballots]
        for earlier in range(number, later):
            vote = self._vote_records(earlier).recorded(member)
            if vote is None:
                return True
            if vote.verdict is None:
                return False
            found.append(consequence(vote.verdict, issued))
            supported.append(vote.verdict.support == "supported")
            if standing(found, closed=False, supported=supported) is not None:
                return False
        return True

    def _confirm(self, ordered: Sequence[ReviewMemberV1],
                 judged: dict[PositionKey, JudgedMemberV1],
                 sent_ahead: Sequence[tuple[PositionKey, int]] = ()) -> None:
        """Confirm every consequential verdict by fresh votes, a majority of three.

        A verdict is consequential when it denies its member or ends an issued head
        (``judge_confirmation``). Each vote round sends every still undecided member
        alone, concurrently within the frozen concurrency; a vote its child
        already holds replays without a call. The batched packing sends a member's later
        votes in the same round (`_votes_together`), each only while the earlier ones leave
        the member undecided; the decision still reads each vote only while the earlier
        ones leave it undecided. A failed request stops the owner only when the decision
        reads that vote of one of its members; a member whose vote the round did not reach
        for another member's failure is sent again, as the vote-by-vote rounds would have
        sent it. The earliest verdict whose consequence
        another verdict shares stands. A vote that ends unavailable closes the member's
        confirmation; without a majority a denial stands, and a member whose verdicts
        disagree on the heads it ends keeps its first verdict with those heads read
        ``unclear``, so it settles ``relation_unclear`` and ends none. A sibling fold of the
        verdict that stands holds only when every agreeing verdict folds that sibling too;
        otherwise it reads ``independent`` (``agreed_folds``). ``sent_ahead`` lists the
        votes `_VotesAhead` already sent during the round; they read here as recorded.
        """
        ballots: dict[PositionKey, list[JudgmentV1]] = {}
        found: dict[PositionKey, list[Consequence]] = {}
        issued: dict[PositionKey, frozenset[str]] = {}
        for member in ordered:
            judgment = judged[member.position_key].judgment
            if judgment is None or judgment.verdict is None:
                continue
            key = member.position_key
            issued[key] = issued_heads(member.input.member.targets, member.input.catalog)
            first = consequence(judgment.verdict, issued[key])
            if consequential(first):
                ballots[key], found[key] = [judgment], [first]
        sent: list[tuple[PositionKey, int]] = []
        failures: list[VoteFailure] = []
        cancelled: list[tuple[int, tuple[PositionKey, ...]]] = []
        read: dict[int, set[PositionKey]] = {}
        settled: set[PositionKey] = set()
        for number in range(2, CONFIRMATION_VOTES + 1):
            asked = [member for member in ordered
                     if member.position_key in ballots and member.position_key not in settled]
            if not asked:
                break
            read[number] = {member.position_key for member in asked}
            records = self._vote_records(number)
            missing = [member for member in asked if records.recorded(member) is None]
            if missing and self._batched:
                _raise_blocking(failures, number, missing)
                together, result = self._votes_together(missing, number, ballots, issued)
                sent.extend(together)
                failures.extend(result.failures)
                cancelled.extend(result.cancelled)
                _raise_blocking(failures, number, missing)
            elif missing:
                for member in missing:
                    records.admit(member)
                with _sending_vote(number):
                    execute_round(self._vote_plans(missing), self._provider, records,
                                  binding=self._binding, bind=self._bind, owner=self._owner,
                                  halving=self._batched,
                                  output_bound=self._output_bound, vote=number)
                sent.extend((member.position_key, number) for member in missing)
            for member in asked:
                key = member.position_key
                vote = records.recorded(member)
                closed = vote is None or vote.verdict is None
                if vote is not None and vote.verdict is not None:
                    ballots[key].append(vote)
                    found[key].append(consequence(vote.verdict, issued[key]))
                decision = standing(found[key], closed=closed, supported=[
                    cast(JudgeV8, ballot.verdict).support == "supported"
                    for ballot in ballots[key]])
                if decision is None:
                    continue
                settled.add(key)
                if decision == "unclear":
                    first = ballots[key][0]
                    judged[key] = JudgedMemberV1(key, replace(first, verdict=unclear_endings(
                        cast(JudgeV8, first.verdict), issued[key])), None)
                    continue
                ballot = ballots[key][decision]
                stood = agreed_folds(
                    [cast(JudgeV8, each.verdict) for each in ballots[key]], found[key],
                    decision, issued_siblings(member.input.member.targets, member.input.catalog))
                if stood is not ballot.verdict:
                    ballot = replace(ballot, verdict=stood)
                if decision or ballot is not ballots[key][decision]:
                    judged[key] = JudgedMemberV1(key, ballot, None)
        for failure in failures:
            if failure.members & read.get(failure.number, set()):
                continue
            # The decision read this vote of none of the request's members. A child the
            # failure left running ends as abandoned before the owner's release
            # (`end_abandoned_reviews`).
            _LOG.warning("A confirmation vote request failed after its members were decided: "
                         "owner=%s vote=%d members=%d error=%s", self._owner, failure.number,
                         len(failure.members),
                         getattr(failure.error, "code", type(failure.error).__name__))
        if cancelled:
            _LOG.info("Confirmation vote requests closed in flight (token usage not "
                      "observed): owner=%s requests=%d", self._owner, len(cancelled))
        rank = {member.position_key: index for index, member in enumerate(ordered)}
        self.votes = tuple(sorted({*sent, *sent_ahead}, key=lambda vote: (vote[1], rank[vote[0]])))
        self.cancelled = tuple(cancelled)


def _raise_blocking(failures: Sequence[VoteFailure], number: int,
                    missing: Sequence[ReviewMemberV1]) -> None:
    """Raise the first failure of vote ``number`` that carried a member still missing it.

    The decision is about to read that vote for those members, as the vote-by-vote
    rounds would after the same failure, so the owner pauses.
    """
    keys = {member.position_key for member in missing}
    for failure in failures:
        if failure.number == number and failure.members & keys:
            raise failure.error


class _VotesAhead:
    """Send a consequential verdict's votes as soon as the verdict persists.

    `JudgeStage._confirm` asks a member's vote 2 whenever its first verdict is
    consequential, and vote 3 whenever the first two leave `standing` undecided.
    This asks exactly those votes, with the same solo requests and records, but
    each starts when the ballot before it persists rather than after the round's
    slowest request; `_confirm` then reads them as recorded votes and decides as
    before. Only members whose verdict a request of this round persisted are
    asked here; a replayed or reused verdict is confirmed after the round.

    Args:
        stage: The reviewing stage.
        members: This round's sent members by position.
    """

    def __init__(self, stage: JudgeStage, members: Mapping[PositionKey, ReviewMemberV1]) -> None:
        self._stage, self._members = stage, members
        self._lock = threading.Lock()
        self.sent: list[tuple[PositionKey, int]] = []

    def release(self, keys: Sequence[PositionKey]) -> list[Followup]:
        """Return the first vote of every persisted verdict that needs confirmation."""
        work: list[Followup] = []
        for key in keys:
            member = self._members.get(key)
            if member is None:
                continue
            judgment = self._stage._records.recorded(member)  # pyright: ignore[reportPrivateUsage]
            if judgment is None or judgment.verdict is None:
                continue
            issued = issued_heads(member.input.member.targets, member.input.catalog)
            first = consequence(judgment.verdict, issued)
            if consequential(first):
                work.append(partial(self._vote, member, issued, (judgment,), (first,), 2))
        return work

    def _vote(self, member: ReviewMemberV1, issued: frozenset[str],
              ballots: tuple[JudgmentV1, ...], found: tuple[Consequence, ...],
              number: int) -> tuple[Followup, ...]:
        """Send vote ``number``; return the next vote while `standing` stays undecided."""
        if self._stage._send_vote(member, number):  # pyright: ignore[reportPrivateUsage]
            with self._lock:
                self.sent.append((member.position_key, number))
        records = self._stage._vote_records(number)  # pyright: ignore[reportPrivateUsage]
        vote = records.recorded(member)
        if vote is None or vote.verdict is None or number >= CONFIRMATION_VOTES:
            return ()
        ballots = (*ballots, vote)
        found = (*found, consequence(vote.verdict, issued))
        if standing(list(found), closed=False, supported=[
            cast(JudgeV8, ballot.verdict).support == "supported" for ballot in ballots
        ]) is not None:
            return ()
        return (partial(self._vote, member, issued, ballots, found, number + 1),)
