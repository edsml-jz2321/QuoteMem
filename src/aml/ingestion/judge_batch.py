"""Judge-v8 physical requests: acceptance, once-only accounting and the round executor.

`prepare_member_outputs` is pure (C.7, C.8 #7). `execute_round` runs one owner's
round of packed requests over S1d's provider (C.8 #2a-#6, C.14): it holds no
lock around a call, sends the members a request leaves to go alone concurrently
with the round's other requests (JX-2), and its `JudgeCarrier` owns
authorization, durable send reservations, deadlines and the one-transaction
persistence of each reply. Work that waits only on one member's persisted
verdict (its confirmation votes) can follow that verdict within the round's
concurrency instead of after the round's slowest request (`Followup`).
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
from dataclasses import dataclass, field
from functools import partial
from typing import Final, Literal, Protocol

from aml.core.entities import PositionKey
from aml.core.errors import AMLError
from aml.core.identity import IntegrityDigest, integrity_digest
from aml.core.inference import (
    InferenceBindingV1,
    JudgeBatchAccountingV1,
    ProviderUsage,
    RetryPolicyV1,
    StoredInferenceOutput,
    judge_timeout_policy_digest,
)
from aml.core.judge_output import (
    JUDGE_BATCH_V8_SCHEMA_ID,
    JUDGE_V8_SCHEMA_ID,
    JudgeBatchOutputV8,
    JudgeMemberAcceptanceV8,
    accept_judge_batch,
    judge_messages,
)
from aml.core.judge_records import JudgeMemberUnavailableV1, JudgeReplyRefusalV1
from aml.core.neighbor_snapshot import PositionV1
from aml.core.provider_budget import JudgeTimeShortfall, SendsSpent
from aml.ingestion.judge_input import (
    JudgeRequestPlanV1,
    MemberInputV1,
    physical_request,
    request_input_bytes,
)
from aml.ingestion.nomination import ContentBinder
from aml.providers.adapters import ProviderError
from aml.providers.attempts import (
    current_provider_attempt,
    provider_attempts,
    provider_attempts_from,
)
from aml.providers.contracts import (
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
    need the C.8 solo resend; they are never converted to timeout dispositions.
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
        AMLError: ``binding_mismatch`` for mismatched plans, schemas, bindings,
            or a result already carrying member accounting.
    """
    if (
        binding.purpose != "judge" or binding.schema_id != JUDGE_BATCH_V8_SCHEMA_ID
        or result.schema_id != JUDGE_BATCH_V8_SCHEMA_ID
        or not isinstance(result.output, JudgeBatchOutputV8)
        or result.usage.judge_batch is not None or result.provider_attempts
        or plan.request != physical_request(plan.members)
    ):
        raise AMLError("binding_mismatch", "Judge reply differs from its physical request")
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


# Physical requests and the round executor (C.5, C.8 #2a-#8, C.14) ------------------------------

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
    """At most one in-flight request per member input digest in this process (C.14)."""

    def __init__(self) -> None:
        self._digests: set[str] = set()
        self._lock = threading.Lock()

    @contextmanager
    def holding(self, digests: Sequence[str]) -> Iterator[None]:
        with self._lock:
            if self._digests.intersection(digests):
                raise AMLError("delivery_busy", "A member input already has a request in flight")
            self._digests.update(digests)
        try:
            yield
        finally:
            with self._lock:
                self._digests.difference_update(digests)


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


def batched_output_tokens(members: int) -> int:
    """The completion bound of a batched request of this many members."""
    base, per_member, cap = BATCHED_OUTPUT_TOKENS
    return min(cap, base + per_member * members)


def judge_request(
    plan: JudgeRequestPlanV1, binding: InferenceBindingV1, *, bind: ContentBinder,
    attempt: Literal[1, 2] = 1, timeout_ms: int | None = None,
    max_output_tokens: int | None = None,
) -> InferenceRequest:
    """Build the physical request of one plan under the frozen binding (C.5, C.6.2).

    The request is validated against the binding first. A request carrying
    more than one member then drops ``timeout`` from its retryable classes, as
    C.5 requires: a shared timeout is never resent (C.8 #5). The numbered
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
        AMLError: ``binding_mismatch`` if the request differs from the binding.
    """
    policy_timeout = binding.timeout_ms if attempt == 1 else binding.judge_second_timeout_ms
    if policy_timeout is None:
        raise AMLError("binding_mismatch", "The judge binding lost its second attempt timeout")
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
        """Bind the request's durable per-send reservation and deadline (C.5).

        Each physical send is reserved in every carried member's child before
        HTTP. Transient retries share its three-send phase allowance; recovery
        advances past a reserved phase. Every wait leaves the owner's commit
        reserve. An exhausted phase raises ``SendsSpent``; too little useful
        time raises ``JudgeTimeShortfall``.
        """
        ...

    def spent(self, plan: JudgeRequestPlanV1, request: InferenceRequest) -> None:
        """Settle the members of a request whose sends are spent as C.8's exhaustion (W3-RT-19).

        No call is made. Every member settles pending; a later Add re-enters it.
        """
        ...

    def persist(self, plan: JudgeRequestPlanV1, request: InferenceRequest,
                reply: JudgeReplyV1) -> None:
        """Store every accepted member of one reply in one Writer transaction (C.8 #8)."""
        ...

    def refused(self, member: MemberInputV1, request: InferenceRequest,
                refusal: JudgeMemberUnavailableV1) -> None:
        """Settle a member still unaccepted after its solo resend as ``unavailable`` (C.8 #4).

        Its result records the refusal and remains pending. A later Add
        re-enters the member (C.3). ``request`` is its actual solo request.
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
    any later request, in canonical order (C.8 #2a). ``run`` returns the steps it
    leaves to start.
    """

    order: tuple[int, int]
    run: Callable[[], tuple[_Job, ...]] = field(compare=False)


class _Round:
    """One owner's round of requests while they run."""

    def __init__(self, provider: StructuredInferenceProvider, carrier: JudgeCarrier, *,
                 binding: InferenceBindingV1, bind: ContentBinder, owner: str,
                 after: AcceptedHook | None = None, halving: bool = False) -> None:
        self.provider, self.carrier, self.binding = provider, carrier, binding
        self.bind, self.owner, self.after, self.halving = bind, owner, after, halving
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
        return judge_request(plan_of(members), self.binding, bind=self.bind,
                             attempt=attempt, timeout_ms=max(1, timeout or 0),
                             max_output_tokens=(batched_output_tokens(len(members))
                                                if self.halving else None))

    def send(self, members: Sequence[MemberInputV1], request: InferenceRequest,
             ) -> tuple[JudgeReplyV1, dict[str, JudgeReplyRefusalV1]]:
        """Send one numbered phase, reauthorizing each transient transport retry."""
        plan = plan_of(members)

        def authorize() -> None:
            self.carrier.authorize(members)

        authorize()
        minimum = self.binding.judge_min_attempt_ms
        if minimum is None:
            raise AMLError("binding_mismatch", "The judge binding lost its useful attempt limit")
        if request.timeout_ms < minimum:
            raise JudgeTimeShortfall
        with provider_attempts_from(self.enclosing), IN_FLIGHT.holding(
            [member.digest for member in members],
        ), provider_attempts(authorize), self.carrier.sends(members, request):
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

        return _Job(order, run)

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

        return _Job(order, run)

    def second(self, index: int, members: Sequence[MemberInputV1],
               positions: Mapping[str, int],
               earlier: Mapping[str, tuple[JudgeReplyRefusalV1, ...]], *, halve: bool,
               ) -> tuple[_Job, ...]:
        """The second phase of a packed request's unaccepted members.

        The full mode sends each alone (C.8 #4). The batched packing keeps them together:
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


def plan_of(members: Sequence[MemberInputV1]) -> JudgeRequestPlanV1:
    """The plan of exactly these members in this order, measured like `pack_requests`."""
    request = physical_request(members)
    return JudgeRequestPlanV1(tuple(members), request, request_input_bytes(request),
                              sum(member.output_bytes for member in members))


def send_alone(
    member: MemberInputV1, provider: StructuredInferenceProvider, carrier: JudgeCarrier, *,
    binding: InferenceBindingV1, bind: ContentBinder, owner: str,
) -> None:
    """Send one member alone through its safe phases, as a round's solo step does.

    One request is in flight at a time: the phase the carrier selects first, then
    any solo phase it leaves, so a caller may run it as one step of a round.

    Args:
        member: The member, already admitted by ``carrier``.
        provider: S1d's structured provider.
        carrier: The member's durable records (a confirmation vote's records).
        binding: The frozen judge-v8 binding.
        bind: The workspace's keyed content binder.
        owner: The reviewing owner run.

    Raises:
        ProviderError: A stopping provider failure.
        AMLError: Revoked authority, exhausted allowance or a binding mismatch.
    """
    state = _Round(provider, carrier, binding=binding, bind=bind, owner=owner)
    steps = list(state.group(0, [member]))
    while steps:
        steps.extend(steps.pop(0).run())


def execute_round(
    plans: Sequence[JudgeRequestPlanV1], provider: StructuredInferenceProvider,
    carrier: JudgeCarrier, *, binding: InferenceBindingV1, bind: ContentBinder, owner: str,
    after: AcceptedHook | None = None, halving: bool = False,
) -> RoundResultV1:
    """Run one owner's round of packed requests (C.8 #2a-#8).

    Requests start in canonical order with at most ``binding.concurrency`` in
    flight (C.8 #2a). A shared request that times out is never resent: its
    members go alone, and the owner's later requests are still sent as packed. A
    member a reply leaves without an accepted verdict is sent once
    more alone; still none settles ``unavailable`` with its refusal codes,
    and the owner goes on (C.8 #4). Members sent alone are steps of
    their own that start before any later request, concurrently within the same
    bound (JX-2). Every first read waits at most 900 s; its single solo second
    phase waits at most 500 s. Both leave the frozen commit reserve. Exhausted,
    spent or budget-limited members settle pending without pausing the owner.
    Recovery advances past a reserved phase and never starts a third. Every other failure
    stops further starts; in-flight requests finish and the first failure in
    canonical order propagates (the owner pauses). Accepted members persist
    whatever their neighbours do, and the result lists them in canonical order
    whatever order the replies came in. No lock is held around a provider call
    (C.14). ``after`` sees each reply's persisted members; the work it returns
    runs within the same bound, after every planned request and solo step has
    started, and the round returns only once that work has finished too.

    Args:
        plans: The round's packed requests, ordered by their first member.
        provider: S1d's structured provider.
        carrier: The owner's durable round state.
        binding: The frozen judge-v8 binding (its ``concurrency``).
        bind: The workspace's keyed content binder.
        owner: The reviewing owner run, for the in-process alone tracking.
        after: Optional hook releasing work that waits only on persisted verdicts.
        halving: The batched packing's second phase (``_Round.second``): a request's
            unaccepted members go on together, halved after a shared timeout or a
            refused whole reply, instead of alone.

    Returns:
        The persisted accepted members and the settled unavailable members.

    Raises:
        ProviderError: The first stopping provider failure.
        AMLError: Revoked authority, exhausted allowance or a binding mismatch.
    """
    if binding.concurrency is None:
        raise AMLError("binding_mismatch", "A judge round requires its frozen concurrency")
    state = _Round(provider, carrier, binding=binding, bind=bind, owner=owner, after=after,
                   halving=halving)
    pending = [_Job((index, -1), partial(state.group, index, plan.members))
               for index, plan in enumerate(plans)]
    heapq.heapify(pending)
    # Released work orders after every planned request and its solo steps, in release order.
    sequence = itertools.count()

    def follow(work: Followup) -> _Job:
        def run() -> tuple[_Job, ...]:
            return tuple(follow(each) for each in work())

        return _Job((len(plans), next(sequence)), run)
    failures: dict[tuple[int, int], BaseException] = {}
    # Each step carries at most one request at a time, so at most ``concurrency``
    # steps running keeps the round's requests in flight within the bound.
    with ThreadPoolExecutor(max_workers=binding.concurrency) as pool:
        running: dict[Future[tuple[_Job, ...]], tuple[int, int]] = {}
        while True:
            while pending and not failures and len(running) < binding.concurrency:
                job = heapq.heappop(pending)
                context = contextvars.copy_context()
                running[pool.submit(context.run, job.run)] = job.order
            if not running:
                break
            done, _ = wait(running, return_when=FIRST_COMPLETED)
            for future in done:
                order = running.pop(future)
                if (error := future.exception()) is not None:
                    failures[order] = error
                    continue
                for step in future.result():
                    heapq.heappush(pending, step)
            for work in state.take_released():
                heapq.heappush(pending, follow(work))
    if failures:
        raise failures[min(failures)]
    accepted = {member.position_key: state.accepted[member.position_key]
                for plan in plans for member in plan.members
                if member.position_key in state.accepted}
    return RoundResultV1(accepted, frozenset(state.unavailable))
