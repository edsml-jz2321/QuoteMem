"""Explicit bounded recovery of an Add owner's completion allowance, never an Add retry.

V2 observes and resumes an exhausted allowance. V3 also observes a refused
budget (the whole-output faults) and reports the content-free
``failure_cause`` of the owner's stop.
"""

from __future__ import annotations

from typing import Annotated, Literal, Protocol, cast, runtime_checkable

from pydantic import ConfigDict, Field, ValidationError, field_validator, model_validator

from quotemem.core.add_outcome import AddOutcomeV1, CompletedAddOutcomeV1
from quotemem.core.controls import FinalizeBudgetRefusalV4, FinalizeBudgetV2
from quotemem.core.errors import QuoteMemError
from quotemem.core.identity import ULID
from quotemem.core.request_budget import DEFAULT_COMPLETION_WORK_BUDGET_MS, MAX_OWNER_RESUMES
from quotemem.core.technical_types import ExternalId
from quotemem.store.metadata import StrictModel
from quotemem.store.request_completion import FailureCause


class CompletionRecoveryRequestV2(StrictModel):
    """Select one exact external user without accepting another owner's locators."""

    model_config = ConfigDict(frozen=True, hide_input_in_errors=True)
    schema_version: Literal[2]
    user_id: Annotated[ExternalId, Field(repr=False)]

    @field_validator("user_id")
    @classmethod
    def valid_user(cls, value: str) -> str:
        try:
            value.encode("utf-8")
        except UnicodeEncodeError:
            raise ValueError("User identity must contain valid Unicode") from None
        return value


class CompletionResumeRequestV2(CompletionRecoveryRequestV2):
    """Compare the exact observed projection and bounded recovery generation.

    It resumes an exhausted allowance only; a refusal needs V3.
    """

    owner_run_id: ULID
    budget_version: Annotated[int, Field(ge=0)]
    recovery_count: Annotated[int, Field(ge=0, le=MAX_OWNER_RESUMES)]
    work_budget_ms: Annotated[int, Field(ge=1, le=DEFAULT_COMPLETION_WORK_BUDGET_MS)] | None = None


class CompletionRecoveryRequestV3(StrictModel):
    """Opt in to refused-budget inspection and its failure cause for one exact external user.

    The caller may also name one of its own Add requests by its exact
    ``session_id`` and ``request_id`` (both or neither) to read that request's
    first completed result, including whether explicit repair completed it.
    """

    model_config = ConfigDict(frozen=True, hide_input_in_errors=True)
    schema_version: Literal[3]
    user_id: Annotated[ExternalId, Field(repr=False)]
    session_id: Annotated[ExternalId | None, Field(repr=False)] = None
    request_id: Annotated[ExternalId | None, Field(repr=False)] = None

    @field_validator("user_id", "session_id", "request_id")
    @classmethod
    def valid_identity(cls, value: str | None) -> str | None:
        try:
            if value is not None:
                value.encode("utf-8")
        except UnicodeEncodeError:
            raise ValueError("Request identities must contain valid Unicode") from None
        return value

    @model_validator(mode="after")
    def exact_request(self) -> CompletionRecoveryRequestV3:
        if (self.session_id is None) != (self.request_id is None):
            raise ValueError("A selected Add request needs both its session and request IDs")
        return self


class CompletionResumeRequestV3(CompletionRecoveryRequestV3):
    """Compare the exact observed V3 projection and bounded recovery generation.

    It resumes an exhausted allowance or a refused budget (the
    whole-output fault on one unit); the refused unit runs again on
    the same owner and frozen prefix, never re-freezing a recorded unit. It also
    repairs a quarantined scope (failure cause ``integrity``) by resuming the
    owner its receipt names, which lifts the quarantine. Each
    grants a fresh allowance (``work_budget_ms``, or the scope's own when
    omitted) of at most 1,740 seconds from the resume transaction and counts
    against the owner's shared 33-resume cap, also after the original deadline.
    Per-item stops never fail the owner, so they never resume here.
    """

    owner_run_id: ULID
    budget_version: Annotated[int, Field(ge=0)]
    recovery_count: Annotated[int, Field(ge=0, le=MAX_OWNER_RESUMES)]
    work_budget_ms: Annotated[int, Field(ge=1, le=DEFAULT_COMPLETION_WORK_BUDGET_MS)] | None = None


class CompletedAddV1(StrictModel):
    """One selected request's first completed result, never rewritten by later work.

    ``recovery_count`` is the owner's counted-resume total at completion: 0 for
    an ordinary completion, 1 through 3 after automatic or explicit recovery.
    """

    model_config = ConfigDict(frozen=True)
    owner_run_id: ULID
    outcome: AddOutcomeV1
    recovery_count: Annotated[int, Field(ge=0, le=MAX_OWNER_RESUMES)]
    skipped: Annotated[bool, Field(exclude_if=lambda value: not value)] = False
    """An operator skipped the request's quarantined cycle (``store.quarantine_skip``)."""

    @classmethod
    def from_completed(cls, value: CompletedAddOutcomeV1) -> CompletedAddV1:
        """Project a stored result without its internal request binding digest."""
        return cls(
            owner_run_id=value.completed_by_run_id, outcome=value.outcome,
            recovery_count=value.recovery_count, skipped=value.state == "skipped",
        )


class CompletionRecoveryResponseV2(StrictModel):
    """Expose exhausted work independently of outer-process or Run liveness."""

    model_config = ConfigDict(frozen=True)
    schema_version: Literal[2] = 2
    budget: FinalizeBudgetV2 | None
    budget_version: Annotated[int, Field(ge=0)] | None

    @model_validator(mode="after")
    def coherent_observation(self) -> CompletionRecoveryResponseV2:
        if (self.budget is None) != (self.budget_version is None):
            raise ValueError("Budget and its observed version must be present together")
        return self


class CompletionRecoveryResponseV3(StrictModel):
    """Observe original V2 allowances or a refused budget; observing grants no recovery.

    A refusal carries the refused ``unit_no`` and that unit's content-free
    whole-output fault (``FinalizeBudgetRefusalV4``).
    ``failure_cause`` is the content-free cause of the owner's stop, from
    ``store.request_completion.failure_cause``; it is omitted while
    nothing stopped the owner. ``completed`` is the selected request's first
    completed result. It is omitted when no request was selected or the
    selected one has not completed.
    """

    model_config = ConfigDict(frozen=True)
    schema_version: Literal[3] = 3
    budget: FinalizeBudgetV2 | FinalizeBudgetRefusalV4 | None
    budget_version: Annotated[int, Field(ge=0)] | None
    failure_cause: Annotated[
        FailureCause | None, Field(exclude_if=lambda value: value is None),
    ] = None
    completed: Annotated[
        CompletedAddV1 | None, Field(exclude_if=lambda value: value is None),
    ] = None

    @model_validator(mode="after")
    def coherent_observation(self) -> CompletionRecoveryResponseV3:
        if (self.budget is None) != (self.budget_version is None):
            raise ValueError("Budget and its observed version must be present together")
        if self.failure_cause not in _CAUSES_BY_STATE[
            None if self.budget is None else self.budget.state
        ]:
            raise ValueError("Failure cause contradicts the observed budget state")
        return self


_CAUSES_BY_STATE: dict[str | None, frozenset[FailureCause | None]] = {
    None: frozenset({None, "item", "provider", "budget", "integrity"}),
    "active": frozenset({None, "item", "provider", "budget", "integrity"}),
    "exhausted": frozenset({None, "budget", "provider", "integrity"}),
    "refused": frozenset({None, "whole_output", "integrity"}),
}
"""The causes ``failure_cause`` can map from each budget state.

A refused budget is ``whole_output`` and an exhausted one ``budget`` unless a
provider pause came first; a quarantined scope is ``integrity`` whatever else
holds. ``whole_output`` exists only with a refused budget. A selected request
can stop on its own ``budget`` cause (request budget, ``delivery_busy``, an
``index_rebuilding`` timeout, ``budget_exceeded``) while the owner's allowance
is still active. ``item`` is never an owner stop: ``failure_cause`` never maps
it, but the wire keeps the value so that ``item_fault_non_200_adds`` can
count one (it must stay 0) instead of failing to parse; a stopped budget never
carries it. Until the scheduler's V3 producer reports the cause, a stopped budget
may still omit it; a later change removes ``None`` from the stopped states.
"""


def parse_completion_response(
    raw: object,
) -> CompletionRecoveryResponseV2 | CompletionRecoveryResponseV3:
    """Parse an exact declared V2 or V3 observation without coercing wire values."""
    if not isinstance(raw, dict):
        raise QuoteMemError("schema_invalid", "Completion observation must be an object")
    fields = cast(dict[str, object], raw)
    version = fields.get("schema_version")
    if type(version) is not int:
        raise QuoteMemError("schema_invalid",
                            "Completion observation needs an exact schema version")
    model = {
        2: CompletionRecoveryResponseV2, 3: CompletionRecoveryResponseV3,
    }.get(version)
    if model is None:
        raise QuoteMemError("unsupported_variant", "Completion observation version is unsupported")
    try:
        return model.model_validate(fields, strict=True)
    except ValidationError:
        raise QuoteMemError("schema_invalid", "Completion observation fields are invalid") from None


def parse_completion_request[RequestT: CompletionRecoveryRequestV2 | CompletionRecoveryRequestV3](
    raw: object, model: type[RequestT],
) -> RequestT:
    """Validate exact wire and SDK fields without retaining payload errors."""
    if isinstance(raw, CompletionRecoveryRequestV2 | CompletionRecoveryRequestV3):
        raw = {**raw.__dict__, **(raw.__pydantic_extra__ or {})}
    try:
        return model.model_validate(raw)
    except ValidationError:
        raise QuoteMemError("invalid_request", "Completion recovery fields are invalid") from None


class QuarantineSkipRequestV1(StrictModel):
    """An operator's explicit skip of one user's quarantined cycle (``store.quarantine_skip``).

    Operator-only: nothing skips automatically. It names the owner, budget
    version and resume count a V3 completion inspection observed, so a cycle
    that moved since is refused (``operation_conflict``) rather than skipped.
    """

    model_config = ConfigDict(frozen=True, hide_input_in_errors=True)
    schema_version: Literal[1]
    user_id: Annotated[ExternalId, Field(repr=False)]
    owner_run_id: ULID
    budget_version: Annotated[int, Field(ge=0)]
    recovery_count: Annotated[int, Field(ge=0, le=MAX_OWNER_RESUMES)]

    @field_validator("user_id")
    @classmethod
    def valid_user(cls, value: str) -> str:
        try:
            value.encode("utf-8")
        except UnicodeEncodeError:
            raise ValueError("User identity must contain valid Unicode") from None
        return value


def parse_skip_request(raw: object) -> QuarantineSkipRequestV1:
    """Validate an exact skip request without retaining payload errors."""
    if isinstance(raw, QuarantineSkipRequestV1):
        raw = {**raw.__dict__, **(raw.__pydantic_extra__ or {})}
    try:
        return QuarantineSkipRequestV1.model_validate(raw)
    except ValidationError:
        raise QuoteMemError("invalid_request", "Quarantine skip fields are invalid") from None


class QuarantineSkipResponseV1(StrictModel):
    """What one skip did: the skipped owner, its requests completed as skipped and the
    stored extraction records it dropped."""

    model_config = ConfigDict(frozen=True)
    schema_version: Literal[1] = 1
    owner_run_id: ULID
    skipped_requests: Annotated[int, Field(ge=0)]
    dropped_records: Annotated[int, Field(ge=0)]


@runtime_checkable
class QuarantineSkipOperations(Protocol):
    """The controller's operator-only skip of a quarantined cycle (``store.quarantine_skip``)."""

    def skip_quarantined_cycle(self, request: QuarantineSkipRequestV1) -> QuarantineSkipResponseV1:
        ...


@runtime_checkable
class CompletionRecoveryOperations(Protocol):
    """The existing controller owns authorization, drainage and sole-Writer mutation."""

    def inspect_completion_budget(
        self, request: CompletionRecoveryRequestV2 | CompletionRecoveryRequestV3,
    ) -> CompletionRecoveryResponseV2 | CompletionRecoveryResponseV3: ...

    def resume_completion_budget(
        self, request: CompletionResumeRequestV2 | CompletionResumeRequestV3,
    ) -> CompletionRecoveryResponseV2 | CompletionRecoveryResponseV3: ...
