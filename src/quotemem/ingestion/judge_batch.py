"""Judge-v8 physical requests: acceptance, once-only accounting and the round executor.

`prepare_member_outputs` is pure. `execute_round` runs one owner's
round of packed requests over the provider: it holds no
lock around a call, sends the members a request leaves to go alone concurrently
with the round's other requests, and its `JudgeCarrier` owns
authorization, durable send reservations, deadlines and the one-transaction
persistence of each reply. Work that waits only on one member's persisted
verdict (its confirmation votes) can follow that verdict within the round's
concurrency instead of after the round's slowest request (`Followup`).
`execute_votes` runs several confirmation votes' requests as one round within the
same concurrency, each vote with its own records, and stops a vote's requests once
the votes before it decide their members.
"""

from __future__ import annotations

import contextvars
import heapq
import itertools
import logging
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import AbstractContextManager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import partial
from typing import Final, Literal, Protocol

from quotemem.core.entities import PositionKey
from quotemem.core.errors import QuoteMemError
from quotemem.core.identity import IntegrityDigest, integrity_digest
from quotemem.core.inference import (
    InferenceBindingV1,
    JudgeBatchAccountingV1,
    ProviderUsage,
    RetryPolicyV1,
    StoredInferenceOutput,
    judge_timeout_policy_digest,
)
from quotemem.core.judge_output import (
    JUDGE_BATCH_V8_SCHEMA_ID,
    JUDGE_V8_SCHEMA_ID,
    JudgeBatchOutputV8,
    JudgeMemberAcceptanceV8,
    accept_judge_batch,
    judge_messages,
)
from quotemem.core.judge_records import JudgeMemberUnavailableV1, JudgeReplyRefusalV1
from quotemem.core.neighbor_snapshot import PositionV1
from quotemem.core.provider_budget import JudgeTimeShortfall, SendsSpent
from quotemem.core.request_budget import RequestBudget, current_budget, request_budget
from quotemem.ingestion.judge_input import (
    JudgeRequestPlanV1,
    MemberInputV1,
    physical_request,
    request_input_bytes,
)
from quotemem.ingestion.nomination import ContentBinder
from quotemem.providers.adapters import ProviderError
from quotemem.providers.attempts import (
    current_provider_attempt,
    provider_attempts,
    provider_attempts_from,
)
from quotemem.providers.contracts import (
    InferenceRequest,
    InferenceResult,
    Message,
    StructuredInferenceProvider,
    validate_binding,
)


@dataclass(frozen=True, slots=True)
class AcceptedJudgeMemberV8:
    """One accepted member ready for the reply's single Writer transaction."""

    position_key: PositionKey
    output: StoredInferenceOutput
    reason_readable: bool


@dataclass(frozen=True, slots=True)
class JudgeReplyV1:
    """Every acceptance and refusal from one reply, in physical request order.

    If ``accepted`` is empty, usage belongs to the transport record. Otherwise
    only its first member bears the physical request's usage. Refused members
    need the solo resend; they are never converted to timeout dispositions.
    """

    accepted: tuple[AcceptedJudgeMemberV8, ...]
    refused: tuple[JudgeMemberAcceptanceV8, ...]


def prepare_member_outputs(
    plan: JudgeRequestPlanV1, result: InferenceResult, *, binding: InferenceBindingV1,
    request_digest: IntegrityDigest, timeout_ms: int | None = None,
) -> JudgeReplyV1:
    """Validate one reply and materialize accepted stored member outputs.

    Args:
        plan: The actual packed request and its ordered member inputs.
        result: The physical provider result, with usage for this request only.
        binding: The owner's frozen physical judge binding.
        request_digest: Integrity digest of the complete physical InferenceRequest.
        timeout_ms: Actual wait carried by that request, including its owner cap.

    Returns:
        All accepted outputs and refusals. The output identity uses each member's
        logical input digest and the unchanged physical binding digest; neither
        the physical grouping nor another member becomes its input identity.

    Raises:
        QuoteMemError: ``binding_mismatch`` for mismatched plans, schemas, bindings,
            or a result already carrying member accounting.
    """
    if (
        binding.purpose != "judge" or binding.schema_id != JUDGE_BATCH_V8_SCHEMA_ID
        or result.schema_id != JUDGE_BATCH_V8_SCHEMA_ID
        or not isinstance(result.output, JudgeBatchOutputV8)
        or result.usage.judge_batch is not None or result.provider_attempts
        or plan.request != physical_request(plan.members)
    ):
        raise QuoteMemError("binding_mismatch", "Judge reply differs from its physical request")
    acceptances = accept_judge_batch(plan.request, result.output)
    accepted: list[AcceptedJudgeMemberV8] = []
    refused: list[JudgeMemberAcceptanceV8] = []
    for member, acceptance in zip(plan.members, acceptances, strict=True):
        if acceptance.verdict is None:
            refused.append(acceptance)
            continue
        charged = not accepted
        usage = result.usage
        stored = StoredInferenceOutput.model_validate({
            "purpose": "judge", "provider": binding.provider,
            "requested_model": binding.requested_model, "resolved_model": result.resolved_model,
            "schema_id": JUDGE_V8_SCHEMA_ID, "prompt_version": binding.prompt_version,
            "timeout_ms": binding.timeout_ms if timeout_ms is None else timeout_ms,
            "retry_policy_digest": binding.retry_policy_digest,
            "inference_binding_digest": integrity_digest(binding),
            "input_digest": member.digest, "context_digest": member.digest,
            "output_schema": JUDGE_V8_SCHEMA_ID, "output": acceptance.verdict,
            "usage": ProviderUsage(
                input_tokens=usage.input_tokens if charged else 0,
                output_tokens=usage.output_tokens if charged else 0,
                judge_batch=JudgeBatchAccountingV1(
                    request_digest=request_digest, charged=charged,
                    batch_input_tokens=usage.input_tokens, batch_output_tokens=usage.output_tokens,
                ),
            ),
        })
        accepted.append(AcceptedJudgeMemberV8(
            member.position_key, stored, acceptance.reason_readable,
        ))
    return JudgeReplyV1(tuple(accepted), tuple(refused))


# Physical requests and the round executor ------------------------------

REQUEST_PURPOSE: Final = "judge-batch-v8-request"
"""Binding purpose of a physical request's input digest: its members' input digests in order."""


_LOG = logging.getLogger(__name__)


def adapter_refusal(members: int, error: ProviderError) -> JudgeReplyRefusalV1:
    """Record a reply the adapter refused as a whole (``schema_invalid``), content-free."""
    detail = error.diagnostics
    return JudgeReplyRefusalV1(
        members=members, fault=None, schema_refusal=error.schema_refusal,
        output_tokens=None if detail is None else detail.attempts[-1].output_tokens,
    )


def describe_refusal(refusal: JudgeReplyRefusalV1) -> str:
    """Render one refused reply for the service log: closed codes and counts only."""
    cause = (f"fault={refusal.fault}" if refusal.fault is not None
             else "schema_refusal=" + ("none" if refusal.schema_refusal is None
                                       else refusal.schema_refusal.reason))
    return f"{cause} members={refusal.members} output_tokens={refusal.output_tokens}"


def _log_refusal(owner: str, member: MemberInputV1, refusal: JudgeReplyRefusalV1) -> None:
    _, unit_no, position = member.position_key
    _LOG.warning("Judge reply not accepted: owner=%s member=%s.%d.%d %s", owner,
                 member.position_key[0], unit_no, position, describe_refusal(refusal))


def _log_unavailable(owner: str, refusal: JudgeMemberUnavailableV1) -> None:
    position = refusal.position
    _LOG.warning("Judge member unavailable after its solo resend: owner=%s member=%s.%d.%d "
                 "replies=[%s]", owner, position.owner_run_id, position.unit_no,
                 position.position, "; ".join(describe_refusal(reply)
                                              for reply in refusal.replies))


type Followup = Callable[[], Sequence["Followup"]]
"""Work that follows persisted verdicts within a round's concurrency.

It carries at most one physical request at a time and returns the work it leaves.
"""

type AcceptedHook = Callable[[Sequence[PositionKey]], Sequence[Followup]]
"""Called after a reply's accepted members persist; returns the work they release."""


class _InFlight:
    """At most one in-flight request per (vote, member input digest) in this process.

    A review is vote 1. Each confirmation vote sends its member's same input again from its
    own child, so the votes of one member may be in flight together.
    """

    def __init__(self) -> None:
        self._keys: set[tuple[int, str]] = set()
        self._lock = threading.Lock()

    @contextmanager
    def holding(self, keys: Sequence[tuple[int, str]]) -> Iterator[None]:
        with self._lock:
            if self._keys.intersection(keys):
                raise QuoteMemError("delivery_busy",
                                    "A member input already has a request in flight")
            self._keys.update(keys)
        try:
            yield
        finally:
            with self._lock:
                self._keys.difference_update(keys)


IN_FLIGHT = _InFlight()


def exhausted_read_timeouts(error: ProviderError) -> bool:
    """Whether the phase ended at its final read timeout, after any transient failures.

    A preceding 429 or 503 does not consume the second read phase. A final
    connect, pool, write or deadline timeout, or a stopped logical budget,
    does not qualify.
    """
    detail = error.diagnostics
    return error.error_class == "timeout" and detail is not None and (
        detail.stop_reason is None and bool(detail.attempts)
        and detail.attempts[-1].outcome == "timeout"
        and detail.attempts[-1].timeout_phase == "read"
    )


BATCHED_OUTPUT_TOKENS: Final = (16_384, 3_072, 65_536)
"""A batched request's completion bound: 16,384 plus 3,072 a member, at most 65,536.

Mode B's first LoCoMo hours: a 14-member reply's completion p50 was 12.9k and p90 37k
tokens, reasoning included, while two runaway replies spent the provider's whole
131,072 tokens and ended ``length``. The bound leaves ordinary replies room (a
14-member request may use 59,392) and stops a runaway at half the provider's limit; a
reply it cuts off is a refused whole reply, so the request goes on as two halves.
"""


AGENT_OUTPUT_TOKENS: Final = (3_072, 1_024, 8_192)
"""The ``agent`` release's completion bound (``extraction_policy.agent_release``): 3,072 plus
1,024 a member, at most 8,192, so a one-member request gets 4,096.

With thinking off (``QUOTEMEM_CHAT_REASONING=off``) an end-to-end test saw judge replies
of at most 1,131 completion tokens (four members), against bounds of 19,456 to 28,672. A
cut-off reply is refused whole and its request goes on as two halves, as under
`BATCHED_OUTPUT_TOKENS`, so a runaway reply costs a few thousand tokens, not tens of
thousands. Under ``high`` routing adds the reasoning room to this bound
(``quotemem.agent.routing``), so it still bounds the answer.
"""

type OutputBound = tuple[int, int, int]
"""A batched request's completion bound: base, per member and cap."""


def batched_output_tokens(members: int, bound: OutputBound = BATCHED_OUTPUT_TOKENS) -> int:
    """The completion bound of a batched request of this many members under ``bound``."""
    base, per_member, cap = bound
    return min(cap, base + per_member * members)


def judge_request(
    plan: JudgeRequestPlanV1, binding: InferenceBindingV1, *, bind: ContentBinder,
    attempt: Literal[1, 2] = 1, timeout_ms: int | None = None,
    max_output_tokens: int | None = None,
) -> InferenceRequest:
    """Build the physical request of one plan under the frozen binding.

    The request is validated against the binding first. A request carrying
    more than one member then drops ``timeout`` from its retryable classes, as
    the batch contract requires: a shared timeout is never resent. The numbered
    phase and effective timeout bind the exact request identity. Read-timeout
    transitions belong to the round executor; transport retries only handle
    the remaining transient failures.

    Args:
        plan: The packed request.
        binding: The owner's frozen judge-v8 binding.
        bind: The workspace's keyed content binder.
        attempt: First planned request or second solo request.
        timeout_ms: Effective owner-capped wait; None selects the phase's policy limit.
        max_output_tokens: The batched packing's completion bound; None sends none.

    Returns:
        The request the provider receives; its integrity digest is the
        accounting ``request_digest``.

    Raises:
        QuoteMemError: ``binding_mismatch`` if the request differs from the binding.
    """
    policy_timeout = binding.timeout_ms if attempt == 1 else binding.judge_second_timeout_ms
    if policy_timeout is None:
        raise QuoteMemError("binding_mismatch", "The judge binding lost its second attempt timeout")
    system, user = judge_messages(plan.request)
    digest = bind(REQUEST_PURPOSE, [member.digest for member in plan.members])
    request = InferenceRequest.model_validate({
        "purpose": "judge", "provider": binding.provider,
        "requested_model": binding.requested_model, "schema_id": binding.schema_id,
        "prompt_version": binding.prompt_version, "input_digest": digest,
        "context_digest": digest,
        "messages": [Message(role="system", content=system), Message(role="user", content=user)],
        "timeout_ms": policy_timeout if timeout_ms is None else timeout_ms,
        "judge_attempt": attempt,
        "judge_timeout_policy_digest": judge_timeout_policy_digest(binding),
        "retry_policy": binding.retry_policy,
        "reasoning_effort": binding.reasoning_effort,
        **({} if max_output_tokens is None else {"max_output_tokens": max_output_tokens}),
    })
    validate_binding(request, binding)
    if len(plan.members) == 1:
        return request
    shared = binding.retry_policy.model_copy(update={"retryable_error_classes": [
        kind for kind in binding.retry_policy.retryable_error_classes if kind != "timeout"
    ]})
    return request.model_copy(update={"retry_policy": RetryPolicyV1.model_validate(
        shared.model_dump(mode="json"),
    )})


class JudgeCarrier(Protocol):
    """The durable side of one owner's round: authority, reservations and persistence."""

    def attempt(self, member: MemberInputV1) -> Literal[1, 2, 3]:
        """Return the next safe phase; a reserved second phase returns 3 (spent)."""
        ...

    def timeout(self, members: Sequence[MemberInputV1], attempt: Literal[1, 2]) -> int | None:
        """Cap the phase by owner time less the commit reserve; None permits no wait."""
        ...

    def authorize(self, members: Sequence[MemberInputV1]) -> None:
        """Recheck grant, lease, cancellation and privacy for exactly these members."""
        ...

    def sends(self, members: Sequence[MemberInputV1], request: InferenceRequest,
              ) -> AbstractContextManager[None]:
        """Bind the request's durable per-send reservation and deadline.

        Each physical send is reserved in every carried member's child before
        HTTP. Transient retries share its three-send phase allowance; recovery
        advances past a reserved phase. Every wait leaves the owner's commit
        reserve. An exhausted phase raises ``SendsSpent``; too little useful
        time raises ``JudgeTimeShortfall``. A carrier that binds the send a request
        budget of its own (``current_budget()`` inside it) lets a round of votes close
        the send once no decision needs it (`_Round.closing`).
        """
        ...

    def spent(self, plan: JudgeRequestPlanV1, request: InferenceRequest) -> None:
        """Settle the members of a request whose sends are spent as the exhaustion.

        No call is made. Every member settles pending; a later Add re-enters it.
        """
        ...

    def persist(self, plan: JudgeRequestPlanV1, request: InferenceRequest,
                reply: JudgeReplyV1) -> None:
        """Store every accepted member of one reply in one Writer transaction."""
        ...

    def refused(self, member: MemberInputV1, request: InferenceRequest,
                refusal: JudgeMemberUnavailableV1) -> None:
        """Settle a member still unaccepted after its solo resend as ``unavailable``.

        Its result records the refusal and remains pending. A later Add
        re-enters the member. ``request`` is its actual solo request.
        """
        ...



@dataclass(frozen=True, slots=True)
class RoundResultV1:
    """What one round's physical requests settled, by position.

    Attributes:
        accepted: Every accepted member, persisted.
        unavailable: Members pending after exhausted reads or phases, insufficient
            useful time, or a solo resend without an accepted verdict.
    """

    accepted: Mapping[PositionKey, AcceptedJudgeMemberV8]
    unavailable: frozenset[PositionKey]


@dataclass(frozen=True, order=True, slots=True)
class _Job:
    """One step of a round that has at most one physical request in flight at a time.

    Steps start in ``order``: ``(plan, -1)`` runs a planned request and ``(plan, k)``
    sends its ``k``-th member alone, so the solo sends a request leaves start before
    any later request, in canonical order. ``run`` returns the steps it
    leaves to start. ``members`` are the positions it carries; none for released work.
    """

    order: tuple[int, int]
    run: Callable[[], tuple[_Job, ...]] = field(compare=False)
    members: tuple[PositionKey, ...] = field(default=(), compare=False)


class _Stopped(Exception):
    """A step stopped by its round because the votes before it decided its members."""


class _Stop:
    """A running step's stop signal (`execute_votes`).

    Once set, the step sends nothing more, and the request it has in flight is closed
    through the request budget its carrier gave that send (``JudgeCarrier.sends``).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._set = False
        self._wakes: list[Callable[[], None]] = []

    def set(self) -> None:
        """Stop the step and wake its in-flight send."""
        with self._lock:
            if self._set:
                return
            self._set = True
            wakes = list(self._wakes)
        for wake in wakes:
            try:
                wake()
            except QuoteMemError:
                # A wake that could not be delivered leaves the request to finish.
                _LOG.warning("A stopped judge send could not be woken")

    def is_set(self) -> bool:
        """Whether the step was stopped."""
        return self._set

    def subscribe(self, wake: Callable[[], None]) -> bool:
        """Wake ``wake`` on a later stop; False when the step is already stopped."""
        with self._lock:
            if self._set:
                return False
            self._wakes.append(wake)
            return True

    def unsubscribe(self, wake: Callable[[], None]) -> None:
        """Stop waking ``wake``."""
        with self._lock:
            if wake in self._wakes:
                self._wakes.remove(wake)


_STOP: ContextVar[_Stop | None] = ContextVar("judge_step_stop", default=None)


class _Round:
    """One owner's round of requests while they run."""

    def __init__(self, provider: StructuredInferenceProvider, carrier: JudgeCarrier, *,
                 binding: InferenceBindingV1, bind: ContentBinder, owner: str,
                 after: AcceptedHook | None = None, halving: bool = False,
                 output_bound: OutputBound = BATCHED_OUTPUT_TOKENS, vote: int = 1) -> None:
        self.provider, self.carrier, self.binding = provider, carrier, binding
        self.bind, self.owner, self.after, self.halving = bind, owner, after, halving
        self.output_bound, self.vote = output_bound, vote
        self.cancelled: list[tuple[PositionKey, ...]] = []
        self.accepted: dict[PositionKey, AcceptedJudgeMemberV8] = {}
        self.unavailable: set[PositionKey] = set()
        self.released: list[Followup] = []
        self.enclosing = current_provider_attempt()
        self.lock = threading.Lock()

    def take_released(self) -> list[Followup]:
        """Hand the executor the work persisted verdicts released so far."""
        with self.lock:
            released, self.released = self.released, []
        return released

    def request(self, members: Sequence[MemberInputV1], attempt: Literal[1, 2],
                ) -> InferenceRequest:
        """Freeze the actual wait in the physical request before reservation.

        A no-send disposition uses the smallest representable wait; it is below
        the useful minimum and never reaches a provider.
        """
        timeout = self.carrier.timeout(members, attempt)
        bound = batched_output_tokens(len(members), self.output_bound) if self.halving else None
        return judge_request(plan_of(members), self.binding, bind=self.bind,
                             attempt=attempt, timeout_ms=max(1, timeout or 0),
                             max_output_tokens=bound)

    def send(self, members: Sequence[MemberInputV1], request: InferenceRequest,
             ) -> tuple[JudgeReplyV1, dict[str, JudgeReplyRefusalV1]]:
        """Send one numbered phase, reauthorizing each transient transport retry."""
        plan = plan_of(members)

        def authorize() -> None:
            self.carrier.authorize(members)

        authorize()
        minimum = self.binding.judge_min_attempt_ms
        if minimum is None:
            raise QuoteMemError("binding_mismatch",
                                "The judge binding lost its useful attempt limit")
        if request.timeout_ms < minimum:
            raise JudgeTimeShortfall
        before = current_budget()
        with provider_attempts_from(self.enclosing), IN_FLIGHT.holding(
            [(self.vote, member.digest) for member in members],
        ), provider_attempts(authorize), self.carrier.sends(members, request), self.closing(
            members, before,
        ):
            result = self.provider.infer(request)
        authorize()
        reply = prepare_member_outputs(plan, result, binding=self.binding,
                                       request_digest=integrity_digest(request),
                                       timeout_ms=request.timeout_ms)
        if reply.accepted:
            self.carrier.persist(plan, request, reply)
            with self.lock:
                self.accepted.update((item.position_key, item) for item in reply.accepted)
            if self.after is not None:
                released = self.after([item.position_key for item in reply.accepted])
                with self.lock:
                    self.released.extend(released)
        refused = {acceptance.member_id: JudgeReplyRefusalV1(
            members=len(members), fault=acceptance.fault, schema_refusal=None,
            output_tokens=result.usage.output_tokens,
        ) for acceptance in reply.refused}
        for member in members:
            if member.digest in refused:
                _log_refusal(self.owner, member, refused[member.digest])
        return reply, refused

    @contextmanager
    def closing(self, members: Sequence[MemberInputV1], before: RequestBudget | None,
                ) -> Iterator[None]:
        """Let the step's stop close this send (`execute_votes`).

        The carrier gives the send a request budget of its own (one that is not
        ``before``, the budget the round runs under). The provider call runs under a
        closer budget bound by that one and ending with it, and the stop cancels only the
        closer, so the transport closes the HTTP request; the carrier's own budget is
        never cancelled by the stop, so a reply that came back as the stop fired is kept
        as an ordinary reply. A stopped step sends nothing; a closed send raises
        `_Stopped` and is listed in ``cancelled``. No reply is read for it, so its token
        usage is not observed, and the provider may still bill what it generated. A send
        whose carrier gave it no budget of its own cannot be closed and finishes as usual.
        """
        stop = _STOP.get()
        if stop is None:
            yield
            return
        if stop.is_set():
            raise _Stopped
        owned = current_budget()
        remaining = 0.0 if owned is None or owned is before else owned.remaining()
        if owned is None or remaining <= 0:
            yield
            return
        closer = RequestBudget(remaining)
        wake = closer.cancel
        if not stop.subscribe(wake):
            raise _Stopped
        try:
            with owned.on_cancel(wake), request_budget(closer):
                yield
        except BaseException:
            if stop.is_set():
                with self.lock:
                    self.cancelled.append(tuple(member.position_key for member in members))
                raise _Stopped from None
            raise
        finally:
            stop.unsubscribe(wake)

    def settle_spent(self, members: Sequence[MemberInputV1], request: InferenceRequest) -> None:
        """Settle all unsent or exhausted members pending, without pausing the owner."""
        self.carrier.spent(plan_of(members), request)
        with self.lock:
            self.unavailable.update(member.position_key for member in members)

    def settle_refused(self, member: MemberInputV1, request: InferenceRequest,
                       replies: Sequence[JudgeReplyRefusalV1]) -> None:
        """Keep the exact final request and content-free refusal on the pending member."""
        refusal = JudgeMemberUnavailableV1(position=PositionV1.of(member.position_key),
                                           replies=list(replies))
        _log_unavailable(self.owner, refusal)
        self.carrier.refused(member, request, refusal)
        with self.lock:
            self.unavailable.add(member.position_key)

    def alone(self, member: MemberInputV1, *, first: bool,
              earlier: tuple[JudgeReplyRefusalV1, ...] = ()) -> None:
        """Send the first or second phase; a first read timeout gets only phase two."""
        request = self.request([member], 1 if first else 2)
        try:
            reply, refused = self.send([member], request)
        except (SendsSpent, JudgeTimeShortfall):
            self.settle_spent([member], request)
            return
        except ProviderError as error:
            if exhausted_read_timeouts(error):
                if first:
                    self.alone(member, first=False, earlier=earlier)
                else:
                    self.settle_spent([member], request)
                return
            if error.code != "schema_invalid":
                raise
            replies = (*earlier, adapter_refusal(1, error))
            _log_refusal(self.owner, member, replies[-1])
        else:
            if reply.accepted:
                return
            replies = (*earlier, refused[member.digest])
        if first:
            self.alone(member, first=False, earlier=replies)
        else:
            self.settle_refused(member, request, replies)

    def solo(self, order: tuple[int, int], member: MemberInputV1, *, first: bool,
             earlier: tuple[JudgeReplyRefusalV1, ...] = ()) -> _Job:
        """The step that sends one member alone, as `alone` does; it leaves no step."""
        def run() -> tuple[_Job, ...]:
            self.alone(member, first=first, earlier=earlier)
            return ()

        return _Job(order, run, (member.position_key,))

    def together(self, order: tuple[int, int], members: Sequence[MemberInputV1],
                 earlier: Mapping[str, tuple[JudgeReplyRefusalV1, ...]]) -> _Job:
        """The step that sends several members' second phase in one request (batched packing).

        The phase is each member's last: a timeout or spent sends settle them pending,
        and a member still without an accepted verdict settles ``unavailable`` with its
        refusals, as a solo second phase would.
        """
        def run() -> tuple[_Job, ...]:
            request = self.request(members, 2)
            try:
                _, refused = self.send(members, request)
            except (SendsSpent, JudgeTimeShortfall):
                self.settle_spent(members, request)
                return ()
            except ProviderError as error:
                if exhausted_read_timeouts(error):
                    self.settle_spent(members, request)
                    return ()
                if error.code != "schema_invalid":
                    raise
                refusal = adapter_refusal(len(members), error)
                for member in members:
                    _log_refusal(self.owner, member, refusal)
                    self.settle_refused(member, request, (*earlier[member.digest], refusal))
                return ()
            for member in members:
                if member.digest in refused:
                    self.settle_refused(member, request,
                                        (*earlier[member.digest], refused[member.digest]))
            return ()

        return _Job(order, run, tuple(member.position_key for member in members))

    def second(self, index: int, members: Sequence[MemberInputV1],
               positions: Mapping[str, int],
               earlier: Mapping[str, tuple[JudgeReplyRefusalV1, ...]], *, halve: bool,
               ) -> tuple[_Job, ...]:
        """The second phase of a packed request's unaccepted members.

        The full mode sends each alone. The batched packing keeps them together:
        after a shared timeout or a refused whole reply it sends the two halves of the
        request, each as one request; members a reply refused one by one go on in one
        request. A single member goes alone either way.
        """
        if not members:
            return ()
        if not self.halving or len(members) == 1:
            return tuple(self.solo((index, positions[member.digest]), member, first=False,
                                   earlier=earlier[member.digest]) for member in members)
        parts = ([members[:len(members) // 2], members[len(members) // 2:]] if halve
                 else [members])
        return tuple(
            self.solo((index, positions[part[0].digest]), part[0], first=False,
                      earlier=earlier[part[0].digest]) if len(part) == 1
            else self.together((index, positions[part[0].digest]), part, earlier)
            for part in parts)

    def group(self, index: int, members: Sequence[MemberInputV1]) -> tuple[_Job, ...]:
        """Dispatch fresh packed members and recover already-reserved phases safely.

        Reservation consumes a phase across recovery. A recovered first phase
        proceeds alone to phase two; a recovered second phase settles pending.
        Members without a reservation retain their planned packing.
        """
        fresh: list[MemberInputV1] = []
        following: list[_Job] = []
        positions = {member.digest: position for position, member in enumerate(members)}
        for member in members:
            attempt = self.carrier.attempt(member)
            if attempt == 1:
                fresh.append(member)
            elif attempt == 2:
                following.append(self.solo((index, positions[member.digest]), member, first=False))
            else:
                self.settle_spent([member], self.request([member], 2))
        if not fresh:
            return tuple(following)
        if len(fresh) == 1:
            self.alone(fresh[0], first=True)
            return tuple(following)
        request = self.request(fresh, 1)
        try:
            _, refused = self.send(fresh, request)
        except (SendsSpent, JudgeTimeShortfall):
            self.settle_spent(fresh, request)
        except ProviderError as error:
            earlier: tuple[JudgeReplyRefusalV1, ...] = ()
            if error.error_class != "timeout":
                if error.code != "schema_invalid":
                    raise
                earlier = (adapter_refusal(len(fresh), error),)
                for member in fresh:
                    _log_refusal(self.owner, member, earlier[0])
            following.extend(self.second(index, fresh, positions,
                                         {member.digest: earlier for member in fresh},
                                         halve=True))
        else:
            unaccepted = [member for member in fresh if member.digest in refused]
            following.extend(self.second(
                index, unaccepted, positions,
                {member.digest: (refused[member.digest],) for member in unaccepted},
                halve=False))
        return tuple(following)


@dataclass(frozen=True, slots=True)
class _Lane:
    """One carrier's round inside `_run`.

    Attributes:
        state: The round's state over its carrier.
        plans: Its packed requests.
        context: The context its steps run in.
        needed: Whether a position still needs this lane's request; None keeps all.
        start: Called with a planned request's members as it starts, before any send.
        pack: Packs members into requests. With it, when one of the lane's requests is
            about to start after some of its members not sent yet were decided, the
            still-needed ones are packed anew (`next_request`), so they are grouped as a
            round of just those members would be.
    """

    state: _Round
    plans: Sequence[JudgeRequestPlanV1]
    context: contextvars.Context
    needed: Callable[[PositionKey], bool] | None = None
    start: Callable[[Sequence[MemberInputV1]], None] | None = None
    pack: Callable[[Sequence[MemberInputV1]], Sequence[JudgeRequestPlanV1]] | None = None

    def next_request(self, number: int, plans: Sequence[JudgeRequestPlanV1],
                     ) -> tuple[_Job | None, list[JudgeRequestPlanV1]]:
        """Start the lane's request ``number`` from its unstarted ``plans``.

        While every member of them is still needed, the next plan starts as packed. Once
        an earlier vote has decided some member, the still-needed members are packed anew,
        as the vote-by-vote round packs the members it asks, not along requests planned
        for the larger set.

        Returns:
            The request's step, None when no member is needed any more, and the plans
            left for the lane's later requests, in canonical order.
        """
        assert self.pack is not None
        members = [member for plan in plans for member in plan.members]
        left = [member for member in members
                if self.needed is None or self.needed(member.position_key)]
        if not left:
            return None, []
        if len(left) < len(members):
            plans = self.pack(left)
        first, *rest = plans
        return self.planned(number, first), list(rest)

    def planned(self, index: int, plan: JudgeRequestPlanV1) -> _Job:
        """The step of planned request ``index``: its still-needed members, packed."""
        def run() -> tuple[_Job, ...]:
            members: Sequence[MemberInputV1] = plan.members
            if self.needed is not None:
                members = [member for member in members if self.needed(member.position_key)]
                if not members:
                    return ()
            if self.start is not None:
                self.start(members)
            return self.state.group(index, members)

        return _Job((index, -1), run, tuple(member.position_key for member in plan.members))

    def wanted(self, job: _Job) -> bool:
        """Whether a step still carries a position this lane's request is needed for."""
        return self.needed is None or not job.members or any(
            self.needed(key) for key in job.members)


@dataclass(frozen=True, slots=True)
class _Failed:
    """A step's failure in `_run`: its lane, order, error and members, and whether it stopped
    its lane (it carried a position still needing the request)."""

    lane: int
    order: tuple[int, int]
    error: BaseException
    members: frozenset[PositionKey]
    stopping: bool


def _stopped_by(stop: _Stop | None, run: Callable[[], tuple[_Job, ...]]) -> tuple[_Job, ...]:
    """Run one step with its stop signal bound (in the step's own context copy)."""
    _STOP.set(stop)
    return run()


def plan_of(members: Sequence[MemberInputV1]) -> JudgeRequestPlanV1:
    """The plan of exactly these members in this order, measured like `pack_requests`."""
    request = physical_request(members)
    return JudgeRequestPlanV1(tuple(members), request, request_input_bytes(request),
                              sum(member.output_bytes for member in members))


def send_alone(
    member: MemberInputV1, provider: StructuredInferenceProvider, carrier: JudgeCarrier, *,
    binding: InferenceBindingV1, bind: ContentBinder, owner: str, vote: int = 1,
) -> None:
    """Send one member alone through its safe phases, as a round's solo step does.

    One request is in flight at a time: the phase the carrier selects first, then
    any solo phase it leaves, so a caller may run it as one step of a round.

    Args:
        member: The member, already admitted by ``carrier``.
        provider: the structured provider.
        carrier: The member's durable records (a confirmation vote's records).
        binding: The frozen judge-v8 binding.
        bind: The workspace's keyed content binder.
        owner: The reviewing owner run.
        vote: The confirmation vote it sends (1 for a review).

    Raises:
        ProviderError: A stopping provider failure.
        QuoteMemError: Revoked authority, exhausted allowance or a binding mismatch.
    """
    state = _Round(provider, carrier, binding=binding, bind=bind, owner=owner, vote=vote)
    steps = list(state.group(0, [member]))
    while steps:
        steps.extend(steps.pop(0).run())


def execute_round(
    plans: Sequence[JudgeRequestPlanV1], provider: StructuredInferenceProvider,
    carrier: JudgeCarrier, *, binding: InferenceBindingV1, bind: ContentBinder, owner: str,
    after: AcceptedHook | None = None, halving: bool = False,
    output_bound: OutputBound = BATCHED_OUTPUT_TOKENS, vote: int = 1,
) -> RoundResultV1:
    """Run one owner's round of packed requests.

    Requests start in canonical order with at most ``binding.concurrency`` in
    flight. A shared request that times out is never resent: its
    members go alone, and the owner's later requests are still sent as packed. A
    member a reply leaves without an accepted verdict is sent once
    more alone; still none settles ``unavailable`` with its refusal codes,
    and the owner goes on. Members sent alone are steps of
    their own that start before any later request, concurrently within the same
    bound. Every first read waits at most 900 s; its single solo second
    phase waits at most 500 s. Both leave the frozen commit reserve. Exhausted,
    spent or budget-limited members settle pending without pausing the owner.
    Recovery advances past a reserved phase and never starts a third. Every other failure
    stops further starts; in-flight requests finish and the first failure in
    canonical order propagates (the owner pauses). Accepted members persist
    whatever their neighbours do, and the result lists them in canonical order
    whatever order the replies came in. No lock is held around a provider call.
    ``after`` sees each reply's persisted members; the work it returns
    runs within the same bound, after every planned request and solo step has
    started, and the round returns only once that work has finished too.

    Args:
        plans: The round's packed requests, ordered by their first member.
        provider: the structured provider.
        carrier: The owner's durable round state.
        binding: The frozen judge-v8 binding (its ``concurrency``).
        bind: The workspace's keyed content binder.
        owner: The reviewing owner run, for the in-process alone tracking.
        after: Optional hook releasing work that waits only on persisted verdicts.
        halving: The batched packing's second phase (``_Round.second``): a request's
            unaccepted members go on together, halved after a shared timeout or a
            refused whole reply, instead of alone.
        output_bound: The batched requests' completion bound (`AGENT_OUTPUT_TOKENS` for an
            ``agent`` release, else `BATCHED_OUTPUT_TOKENS`).
        vote: The confirmation vote the round sends (1 for a review), which keys its
            in-flight requests (`IN_FLIGHT`).

    Returns:
        The persisted accepted members and the settled unavailable members.

    Raises:
        ProviderError: The first stopping provider failure.
        QuoteMemError: Revoked authority, exhausted allowance or a binding mismatch.
    """
    if binding.concurrency is None:
        raise QuoteMemError("binding_mismatch", "A judge round requires its frozen concurrency")
    state = _Round(provider, carrier, binding=binding, bind=bind, owner=owner, after=after,
                   halving=halving, output_bound=output_bound, vote=vote)
    failures = _run([_Lane(state, plans, contextvars.copy_context())], binding.concurrency)
    if failures:
        raise failures[0].error  # a lane without ``needed``: every failure stopped it
    accepted = {member.position_key: state.accepted[member.position_key]
                for plan in plans for member in plan.members
                if member.position_key in state.accepted}
    return RoundResultV1(accepted, frozenset(state.unavailable))


@dataclass(frozen=True, slots=True)
class VoteRequests:
    """One confirmation vote's packed requests in a round of votes.

    Attributes:
        number: The vote, 2 to ``CONFIRMATION_VOTES``.
        plans: Its packed requests, ordered by their first member.
        carrier: The vote's durable records.
        context: The context its requests run in, which names the vote to an observer.
    """

    number: int
    plans: Sequence[JudgeRequestPlanV1]
    carrier: JudgeCarrier
    context: contextvars.Context


@dataclass(frozen=True, slots=True)
class VoteFailure:
    """A failed step of a round of votes.

    Attributes:
        number: Its vote.
        error: Its failure.
        members: The positions the step carried.
        stopping: Whether it stopped its vote: it carried a position the vote was still
            needed for. Otherwise its members were already decided and the round went on.
    """

    number: int
    error: BaseException
    members: frozenset[PositionKey]
    stopping: bool


@dataclass(frozen=True, slots=True)
class VotesResultV1:
    """What a round of votes left besides its records.

    Attributes:
        failures: Every failed step, by vote and then in canonical order.
        cancelled: Every request closed in flight, as (vote, its positions). No reply was
            read for these, so their token usage is not observed, and the provider may
            still bill them. (The agent service's provider counters count every
            asynchronous call cancelled after its request went out, these among them, as
            a call without usage.)
    """

    failures: tuple[VoteFailure, ...]
    cancelled: tuple[tuple[int, tuple[PositionKey, ...]], ...]


def execute_votes(
    votes: Sequence[VoteRequests], provider: StructuredInferenceProvider, *,
    binding: InferenceBindingV1, bind: ContentBinder, owner: str, halving: bool = False,
    output_bound: OutputBound = BATCHED_OUTPUT_TOKENS,
    needed: Callable[[int, PositionKey], bool] | None = None,
    start: Callable[[int, Sequence[MemberInputV1]], None] | None = None,
    pack: Callable[[Sequence[MemberInputV1]], Sequence[JudgeRequestPlanV1]] | None = None,
) -> VotesResultV1:
    """Run several confirmation votes' requests as one round.

    Each vote's requests run as `execute_round` runs a round's, through its own records
    and under the same request budget as the caller, and all of them share the frozen
    concurrency. An earlier vote's steps start before a later vote's.

    ``needed(vote, position)`` says whether a position may still need that vote; it
    turns false once the votes before it decide the position, and stays false. With
    ``pack``, each of a vote's requests is taken from its unstarted plans only when it is
    about to start; once the earlier votes have decided some of those members, the
    still-needed ones are packed anew, so a vote that starts after that is grouped as the
    vote-by-vote round groups it, and a vote whose members are all still needed keeps
    its packing.
    A planned request starts with only its still-needed members (``start`` sees them
    first, so the caller admits their children and lists them as asked); a step
    whose positions are all decided does not start; a running one is stopped, and the
    request it has in flight is closed (`_Round.closing`). A closed request's usage is
    not observed (``VotesResultV1.cancelled``).

    A step's failure stops its vote, and the later votes, only when the step carried a
    still-needed position; a failure of a step whose positions were all decided is
    recorded and the round goes on. Nothing is raised: the caller raises a failure when
    its decision reads that vote for one of the failure's positions.

    Args:
        votes: The votes' requests, each vote once.
        provider: the structured provider.
        binding: The frozen judge-v8 binding (its ``concurrency``).
        bind: The workspace's keyed content binder.
        owner: The reviewing owner run.
        halving: As `execute_round`'s.
        output_bound: As `execute_round`'s.
        needed: Whether a position may still need a vote; None keeps every request.
        start: Called with each planned request's members as it starts.
        pack: Packs a vote's still-needed members into its requests.

    Returns:
        The round's failures and closed requests.
    """
    if binding.concurrency is None:
        raise QuoteMemError("binding_mismatch", "A judge round requires its frozen concurrency")
    if len({vote.number for vote in votes}) != len(votes):
        raise QuoteMemError("binding_mismatch", "A round of votes repeats a vote")
    ordered = sorted(votes, key=lambda vote: vote.number)
    lanes = [_Lane(
        _Round(provider, vote.carrier, binding=binding, bind=bind, owner=owner,
               halving=halving, output_bound=output_bound, vote=vote.number),
        vote.plans, vote.context,
        None if needed is None else partial(needed, vote.number),
        None if start is None else partial(start, vote.number),
        pack,
    ) for vote in ordered]
    failed = _run(lanes, binding.concurrency)
    return VotesResultV1(
        tuple(VoteFailure(ordered[each.lane].number, each.error, each.members, each.stopping)
              for each in failed),
        tuple((vote.number, members) for vote, lane in zip(ordered, lanes, strict=True)
              for members in lane.state.cancelled),
    )


def _next_request() -> tuple[_Job, ...]:
    """Marks a lane's next request in `_run`, which packs it only as it starts; never run."""
    raise AssertionError("A lane's next request is packed as it starts, not run")


def _run(lanes: Sequence[_Lane], concurrency: int) -> list[_Failed]:
    """Run each lane's planned requests and the steps they leave within one bound.

    Steps start in ``(lane, order)`` order with at most ``concurrency`` running, so an
    earlier lane's steps start first. A lane with ``pack`` keeps its unstarted plans and
    takes its next request from them only when that request is about to start, packing
    the still-needed members anew once some were decided (`_Lane.next_request`). A step
    of a lane with ``needed`` whose positions are all decided is not started, and a
    running one is stopped. A failure stops further starts of its lane and every later
    lane, unless the failed step's positions were all decided; running steps finish.
    Returns every failure, by lane and then in canonical order.
    """
    pending: list[tuple[int, tuple[int, int], int, _Job]] = []
    ties = itertools.count()

    def push(lane: int, job: _Job) -> None:
        heapq.heappush(pending, (lane, job.order, next(ties), job))

    waiting: dict[int, list[JudgeRequestPlanV1]] = {}

    def mark(lane: int, number: int) -> None:
        """Queue the lane's request ``number``, taken from its waiting plans as it starts."""
        push(lane, _Job((number, -1), _next_request, tuple(
            member.position_key for plan in waiting[lane] for member in plan.members)))

    for number, lane in enumerate(lanes):
        if lane.pack is None:
            for index, plan in enumerate(lane.plans):
                push(number, lane.planned(index, plan))
            continue
        waiting[number] = list(lane.plans)
        if waiting[number]:
            mark(number, 0)
    # Released work orders after every planned request and its solo steps, in release order.
    sequences = [itertools.count() for _ in lanes]

    def follow(lane: int, work: Followup) -> _Job:
        def run() -> tuple[_Job, ...]:
            return tuple(follow(lane, each) for each in work())

        return _Job((len(lanes[lane].plans), next(sequences[lane])), run)
    failed: list[_Failed] = []
    halted = len(lanes)
    # Each step carries at most one request at a time, so at most ``concurrency``
    # steps running keeps the round's requests in flight within the bound.
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        running: dict[Future[tuple[_Job, ...]], tuple[int, _Job, _Stop | None]] = {}
        while True:
            while pending and len(running) < concurrency:
                index, _, _, job = heapq.heappop(pending)
                lane = lanes[index]
                if index >= halted or not lane.wanted(job):
                    continue
                if job.run is _next_request:
                    number = job.order[0]
                    found, waiting[index] = lane.next_request(number, waiting[index])
                    if found is None:
                        continue
                    if waiting[index]:
                        mark(index, number + 1)
                    job = found
                stop = None if lane.needed is None else _Stop()
                context = lane.context.copy()
                running[pool.submit(context.run, _stopped_by, stop, job.run)] = (
                    index, job, stop)
            if not running:
                break
            done, _ = wait(running, return_when=FIRST_COMPLETED)
            for future in done:
                index, job, _ = running.pop(future)
                if (error := future.exception()) is not None:
                    if isinstance(error, _Stopped):
                        continue
                    stopping = lanes[index].wanted(job)
                    failed.append(_Failed(index, job.order, error, frozenset(job.members),
                                          stopping))
                    if stopping:
                        halted = min(halted, index)
                    continue
                for step in future.result():
                    push(index, step)
            for index, lane in enumerate(lanes):
                for work in lane.state.take_released():
                    push(index, follow(index, work))
            for index, job, stop in running.values():
                if stop is not None and not lanes[index].wanted(job):
                    stop.set()
    return sorted(failed, key=lambda each: (each.lane, each.order))
