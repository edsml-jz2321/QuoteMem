"""Frozen provider call budgets, independent of request/work ownership."""

from collections.abc import Mapping
from typing import Annotated, Final, Literal, Self

from pydantic import ConfigDict, Field, model_validator

from quotemem.core.errors import QuoteMemError
from quotemem.core.identity import ULID, IntegrityDigest
from quotemem.core.judge_timeout import JUDGE_SECOND_TIMEOUT_MS
from quotemem.core.request_budget import MAX_OWNER_RESUMES
from quotemem.store.metadata import OpaqueIdSpec, RefSpec, StrictModel

ExtractionCallBudgetTrial = Literal[
    "extraction-wall-120-300-v2", "extraction-wall-240-300-v2", "extraction-wall-480-600-v2",
    "extraction-wall-900-1500-v1",
]
DEFAULT_EXTRACTION_CALL_BUDGET: ExtractionCallBudgetTrial = "extraction-wall-480-600-v2"
LONG_EXTRACTION_CALL_BUDGET: ExtractionCallBudgetTrial = "extraction-wall-900-1500-v1"
"""The lean-local Add mode's extraction budget: 900 s per request, 1,500 s per call.

That mode's judge makes no model call, so the extraction call is the Add's only long
wait. Answered lean-local extraction requests took at most 340 s (a cost run, 404 calls,
p90 244 s; a deployment run, 71 calls, at most 288 s; a later run, no Add over 260 s), at no fewer
than 123 completion tokens per second, for replies of up to 52,632 completion tokens.
900 s carries that reply at 58 tokens per second, under half the slowest observed rate,
and the 600 s left for a retry carry the slowest observed reply. The 1,500 s call leaves
240 s of the Add's 1,740 s for its source labels (at most 84 s observed), embeddings
and commit.
"""
EXTENDED_EXTRACTION_ATTEMPT_MS: Final[Mapping[ExtractionCallBudgetTrial, int]] = {
    "extraction-wall-480-600-v2": 480000, "extraction-wall-900-1500-v1": 900000,
}
"""The trials whose requests may wait more than 300 s, with each one's per-request cap."""
MAX_EXTRACTION_ATTEMPT_MS = max(EXTENDED_EXTRACTION_ATTEMPT_MS.values())
MAX_EXTRACTION_CALL_MS = 1_500_000
"""The longest logical extraction call of any trial."""
BudgetMilliseconds = Annotated[int, Field(ge=1, le=MAX_EXTRACTION_CALL_MS)]
EXTRACTION_REQUEST_CAP = 3
"""Total physical HTTP requests of one budgeted logical extraction call.

Network retries under the frozen retry policy and output-schema regenerations
share this cap and the call's logical budget. The explicit outage resume
adds one send, at most once per counted owner resume.
"""


class JudgeTimeShortfall(QuoteMemError):
    """A judge attempt cannot finish usefully while preserving the commit reserve.

    This is a member disposition, not exhaustion of the owner's full budget.
    The judge executor settles its members pending without starting another send.
    """

    def __init__(self) -> None:
        super().__init__("delivery_busy", "The judge must leave the owner's commit reserve")


class CallAllowanceEnded(QuoteMemError):
    """An extraction call's own logical allowance ran out before any reply.

    Its window covers every send of the call in the owner's attempt, so no later
    invocation of that attempt can send it again: the owner cannot progress until a
    counted resume starts its next attempt. The scheduler ends the owner's allowance
    at once instead of retrying the refusal until the owner's own deadline.
    Its code is ``delivery_busy`` for every other reader.
    """

    def __init__(self) -> None:
        """Create the refusal; it carries no payload."""
        super().__init__("delivery_busy", "The extraction call's allowance is exhausted")


class SendsSpent(QuoteMemError):
    """A logical call or judge request whose physical sends are all reserved.

    Reserved sends stay spent across crashes and explicit resumes, so no send can
    start and no reply was accepted. The caller settles the call without one: an
    extraction call ends its unit with a provider refusal, and a judge
    request's members settle pending as exhausted read timeouts do.
    Its code is ``delivery_busy`` for every other reader.
    """

    def __init__(self) -> None:
        """Create the refusal; it carries no payload."""
        super().__init__("delivery_busy", "The call's physical sends are spent")


OwnerBudgetPurpose = Literal["extract", "judge"]
"""The provider child a send allowance belongs to: an extraction call or a judge request."""

OWNER_BUDGET_SPAN_MS: Final[Mapping[OwnerBudgetPurpose, int]] = {
    "extract": MAX_EXTRACTION_CALL_MS,
    "judge": 1800000,
}
"""Longest allowance per purpose: one logical extraction call (at most
``MAX_EXTRACTION_CALL_MS``), or the
owner's 1,800 s that every judge send draws on; the reservation logic
further caps a judge deadline at the one the owner's current budget record holds."""


UnitCallEndCause = Literal["truncated", "output_bound", "owner_time"]
"""Why a unit's extraction call ended for good without an accepted reply."""

UnitCallEndOutcome = Literal[
    "split", "single_part", "unit_bound", "first_stands", "not_started",
]
"""What the extraction stage made of that end.

``split`` admits the unit's halves; ``single_part`` and ``unit_bound`` name why an
unsplittable unit ended spent; ``first_stands`` keeps call 0's reply beside a
spent fresh call; ``not_started`` ends a split piece the owner's deadline no longer fits."""


class UnitCallEndV1(StrictModel):
    """The durable end of one unit call: recorded before the stage acts on it.

    A replay reads it before any send, so an interrupted owner takes the same path
    again: a call cut short is never resent, and a piece that did not fit the owner's
    deadline never starts.
    """

    cause: UnitCallEndCause
    outcome: UnitCallEndOutcome
    elapsed_ms: Annotated[int, Field(ge=0)] | None = Field(
        default=None, exclude_if=lambda value: value is None,
    )
    """How long a split call took in the invocation that cut it: its pieces start only
    while the owner's deadline fits that long again (``admits_piece``)."""

    @model_validator(mode="after")
    def consistent(self) -> Self:
        """Only the owner's deadline leaves a call unstarted; only a split records a duration."""
        if (self.cause == "owner_time") != (self.outcome == "not_started"):
            raise ValueError("Only a piece the owner deadline does not fit is not started")
        if (self.outcome == "split") != (self.elapsed_ms is not None):
            raise ValueError("A split records how long its call took; no other end does")
        return self


class ExtractionOwnerBudgetV1(StrictModel):
    """One provider child attempt's durable deadline and physical-send count.

    Every physical send is reserved before HTTP and remains spent after a crash.
    This is execution state, not a history of calls or an authorization shortcut.
    An explicit retry of a failed child is a new logical call: its attempt gets
    its own allowance, still bounded by the original owner's deadline.

    ``outage_resumes`` counts the extra physical sends explicitly granted to
    an exhausted Add extraction call; its reservations stay cumulative.
    Automatic recovery preserves both counts. Judges never receive this grant.
    ``ended`` marks an Add extraction call that ended for good (`UnitCallEndV1`):
    it never sends again, and no resume renews it.

    ``purpose`` names the child. An ``extract`` allowance spans at
    most one logical extraction call, 600 s, or 1,500 s under
    ``LONG_EXTRACTION_CALL_BUDGET``. A ``judge`` allowance holds
    the sends a judge child reserves, at most ``EXTRACTION_REQUEST_CAP``, within
    at most 1,800 s; how a request's sends map onto its members' children
    is the judge stage's reservation rule.
    """

    schema_version: Literal[1] = 1
    purpose: OwnerBudgetPurpose
    source_run_id: Annotated[ULID, RefSpec("run", scope_rule="same_scope")]
    source_attempt: Annotated[int, Field(ge=0)]
    policy_digest: Annotated[IntegrityDigest, OpaqueIdSpec("digest")]
    started_at: Annotated[int, Field(ge=0)]
    deadline_at: Annotated[int, Field(gt=0)]
    attempts_reserved: Annotated[
        int, Field(ge=0, le=EXTRACTION_REQUEST_CAP + MAX_OWNER_RESUMES),
    ] = 0
    outage_resumes: Annotated[int, Field(ge=0, le=MAX_OWNER_RESUMES)] = 0
    judge_attempt: Literal[1, 2] | None = Field(
        default=None, exclude_if=lambda value: value is None,
    )
    judge_timeout_ms: Annotated[int, Field(ge=1, le=900000)] | None = Field(
        default=None, exclude_if=lambda value: value is None,
    )
    ended: UnitCallEndV1 | None = Field(default=None, exclude_if=lambda value: value is None)

    @model_validator(mode="after")
    def bounded_deadline(self) -> Self:
        """Keep the allowance inside the bound of its purpose."""
        if not 0 < self.deadline_at - self.started_at <= OWNER_BUDGET_SPAN_MS[self.purpose]:
            raise ValueError("Owner budget deadline exceeds the allowance of its purpose")
        if (self.judge_attempt is None) != (self.judge_timeout_ms is None) or (
            self.purpose != "judge" and self.judge_attempt is not None
        ):
            raise ValueError("Only a judge allowance pins its numbered attempt and timeout")
        if self.judge_attempt == 2 and (
            self.judge_timeout_ms is None or self.judge_timeout_ms > JUDGE_SECOND_TIMEOUT_MS
        ):
            raise ValueError("The second judge allowance exceeds its timeout policy")
        if self.purpose != "extract" and (self.outage_resumes or self.ended is not None):
            raise ValueError("Only extraction calls receive outage resume sends or end marks")
        if self.attempts_reserved > EXTRACTION_REQUEST_CAP + self.outage_resumes:
            raise ValueError("Owner budget sends exceed its granted allowance")
        return self


class ProviderCallBudgetV2(StrictModel):
    """Bound attempts and all retries of one logical call without resetting its parent.

    The call's physical requests, at most ``EXTRACTION_REQUEST_CAP`` of them,
    and every backoff between them fit in ``logical_wall_ms``; each request
    fits in ``attempt_wall_ms``.
    """

    model_config = ConfigDict(frozen=True)

    schema_version: Literal[2]
    policy_id: Annotated[ExtractionCallBudgetTrial, OpaqueIdSpec("local")]
    attempt_wall_ms: BudgetMilliseconds
    logical_wall_ms: BudgetMilliseconds
    connect_ms: BudgetMilliseconds
    pool_ms: BudgetMilliseconds
    write_ms: BudgetMilliseconds
    read_ms: BudgetMilliseconds

    @model_validator(mode="after")
    def coherent_limits(self) -> Self:
        """Reject phase limits exceeding the attempt or logical wall-clock budget."""
        if self.attempt_wall_ms > self.logical_wall_ms:
            raise ValueError("Provider attempt exceeds its logical call budget")
        if max(self.connect_ms, self.pool_ms, self.write_ms, self.read_ms) > self.attempt_wall_ms:
            raise ValueError("Provider phase exceeds its attempt budget")
        return self


def extraction_call_budget(trial: ExtractionCallBudgetTrial) -> ProviderCallBudgetV2:
    """Resolve the installed wall-budget identity without changing its original bytes."""
    match trial:
        case "extraction-wall-120-300-v2":
            attempt_ms, logical_ms = 120000, 300000
        case "extraction-wall-240-300-v2":
            attempt_ms, logical_ms = 240000, 300000
        case "extraction-wall-480-600-v2":
            attempt_ms, logical_ms = 480000, 600000
        case "extraction-wall-900-1500-v1":
            attempt_ms, logical_ms = 900000, MAX_EXTRACTION_CALL_MS
        case _:
            raise ValueError("Unknown extraction call-budget trial")
    return ProviderCallBudgetV2(
        schema_version=2,
        policy_id=trial,
        attempt_wall_ms=attempt_ms,
        logical_wall_ms=logical_ms,
        connect_ms=10000,
        pool_ms=10000,
        write_ms=30000,
        read_ms=attempt_ms,
    )


def source_labels_call_budget(trial: ExtractionCallBudgetTrial) -> ProviderCallBudgetV2:
    """Select the independent label budget without extending its 240/300-second cap.

    Args:
        trial: The governing extraction policy's installed call-budget identity.

    Returns:
        The existing shorter trial, or the label policy's 240/300-second budget
        when extraction selects a longer budget (``EXTENDED_EXTRACTION_ATTEMPT_MS``).

    Raises:
        ValueError: If the extraction call-budget identity is unknown.
    """
    return extraction_call_budget(
        "extraction-wall-240-300-v2" if trial in EXTENDED_EXTRACTION_ATTEMPT_MS else trial,
    )


def extended_attempt_allowed(call_budget: ProviderCallBudgetV2 | None, timeout_ms: int) -> bool:
    """Whether an extraction request may wait ``timeout_ms`` (over 300 s) under its budget.

    Args:
        call_budget: The request's logical call budget; None has no extension.
        timeout_ms: The request's read timeout.

    Returns:
        True when the budget names an extended trial whose per-request cap covers
        the timeout.
    """
    return call_budget is not None and timeout_ms <= EXTENDED_EXTRACTION_ATTEMPT_MS.get(
        call_budget.policy_id, 0,
    )
