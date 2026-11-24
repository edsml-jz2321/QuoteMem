"""Explicit provider recovery contracts over the Add owner.

Locators identify an observed provider pause; they never replace its original
stored proof (the owner's ``result_index:provider_failure``). A successful resume
acknowledges the durable transition, not the later provider call or publication.
"""

from __future__ import annotations

from typing import Annotated, Final, Literal, Protocol, Self, cast, get_args

from pydantic import ConfigDict, Field, ValidationError, field_validator, model_validator

from quotemem.api.completion_recovery import (
    CompletionRecoveryOperations,
    CompletionRecoveryRequestV2,
    CompletionRecoveryRequestV3,
    CompletionRecoveryResponseV2,
    CompletionRecoveryResponseV3,
    CompletionResumeRequestV2,
    CompletionResumeRequestV3,
    QuarantineSkipOperations,
    QuarantineSkipRequestV1,
    QuarantineSkipResponseV1,
    parse_completion_request,
    parse_skip_request,
)
from quotemem.core.errors import QuoteMemError
from quotemem.core.identity import ULID
from quotemem.core.technical_types import ExternalId
from quotemem.service.lifecycle import ServiceLifetime, ServiceUnavailable
from quotemem.store.metadata import StrictModel


class RecoveryRequestV1(StrictModel):
    """Inspect only the current finalize owner of this exact external user."""

    model_config = ConfigDict(frozen=True, hide_input_in_errors=True)
    schema_version: Literal[1]
    user_id: Annotated[ExternalId, Field(repr=False)]

    @field_validator("user_id")
    @classmethod
    def valid_user(cls, value: str) -> str:
        """Preserve exact IDs while rejecting non-scalar Unicode."""
        try:
            value.encode("utf-8")
        except UnicodeEncodeError:
            raise ValueError("User identity must contain valid Unicode") from None
        return value


class ResumeRequestV1(RecoveryRequestV1):
    """Compare all four observed locators before consuming the original proof."""

    owner_run_id: ULID
    child_run_id: ULID
    owner_attempt: Annotated[int, Field(ge=0)]
    child_attempt: Annotated[int, Field(ge=0)]


type ProviderFailureKind = Literal["source_labels", "extract", "embedding", "judge"]
"""The owner stages whose provider failure pauses the owner.

They are the call kinds of the owner's ``result_index:provider_failure``.
``source_labels`` fails in its label child Run. ``extract``, ``embedding`` and
``judge`` are calls the owner makes itself: a judge request carries up to eight
members' ``evaluate:judgment`` children, so its failure is the owner's own and
the owner Run records it.
"""

PROVIDER_FAILURE_KINDS: Final[tuple[ProviderFailureKind, ...]] = cast(
    "tuple[ProviderFailureKind, ...]", get_args(ProviderFailureKind.__value__),
)


class ProviderFailureV1(StrictModel):
    """Content-free identity of a strictly proved current provider pause.

    Attributes:
        owner_run_id (str): The paused Add owner.
        child_run_id (str): The failed label child Run for ``source_labels``;
            for ``extract``, ``embedding`` and ``judge``, which the owner calls
            itself, the owner's own Run id.
        owner_attempt (int): The owner's attempt that observed the failure.
        child_attempt (int): The failed child's attempt; the owner's own
            attempt for ``extract``, ``embedding`` and ``judge``.
        kind (str): The paused stage.
        failure_code (str): Always ``provider_error``.
    """

    model_config = ConfigDict(frozen=True)
    owner_run_id: ULID
    child_run_id: ULID
    owner_attempt: Annotated[int, Field(ge=0)]
    child_attempt: Annotated[int, Field(ge=0)]
    kind: ProviderFailureKind
    failure_code: Literal["provider_error"] = "provider_error"

    @model_validator(mode="after")
    def owner_stage_names_its_owner(self) -> Self:
        """An owner-run stage names the owner itself as its failed Run."""
        if self.kind in _OWNER_STAGES and (
            (self.child_run_id, self.child_attempt) != (self.owner_run_id, self.owner_attempt)
        ):
            raise ValueError("An owner-run stage names the owner as its failed Run")
        if self.kind not in _OWNER_STAGES and self.child_run_id == self.owner_run_id:
            raise ValueError("A child stage names a child Run distinct from its owner")
        return self


_OWNER_STAGES: Final[frozenset[ProviderFailureKind]] = frozenset(
    {"extract", "embedding", "judge"},
)
"""Stages the owner Run calls itself, not a child."""

type RecoveryFailureV1 = ProviderFailureV1
"""The only recoverable failure: a provider pause."""


class RecoveryResponseV1(StrictModel):
    """An observation only; null means no eligible failure in the current owner."""

    model_config = ConfigDict(frozen=True)
    schema_version: Literal[1] = 1
    failure: RecoveryFailureV1 | None


class ResumeResponseV1(StrictModel):
    """The selected failure was durably resumed; execution may still fail later."""

    model_config = ConfigDict(frozen=True)
    schema_version: Literal[1] = 1
    status: Literal["resume_committed"] = "resume_committed"
    selection: RecoveryFailureV1


class RecoveryOperations(Protocol):
    """The existing controller owns selection, authorization and atomic resume."""

    def inspect_recovery(self, request: RecoveryRequestV1) -> RecoveryResponseV1:
        """Observe a proved current failure without reviving it."""
        ...

    def resume_recovery(self, request: ResumeRequestV1) -> ResumeResponseV1:
        """Commit one explicitly selected recovery through the existing owner."""
        ...


def parse_recovery_request[RequestT: RecoveryRequestV1](
    raw: object, model: type[RequestT],
) -> RequestT:
    """Validate wire or SDK fields without retaining payload-bearing errors."""
    if isinstance(raw, RecoveryRequestV1):
        payload = dict(raw.__dict__)
        payload.update(raw.__pydantic_extra__ or {})
        raw = payload
    try:
        return model.model_validate(raw)
    except ValidationError:
        raise QuoteMemError("invalid_request", "Provider recovery fields are invalid") from None


class BenchRecoveryService:
    """Borrow the same serving lifetime as Add/Search without creating workers."""

    def __init__(self, operations: RecoveryOperations, lifetime: ServiceLifetime) -> None:
        self._operations = operations
        self._lifetime = lifetime

    def inspect_recovery(self, request: RecoveryRequestV1) -> RecoveryResponseV1:
        """Authorize and inspect through the initialized controller."""
        try:
            with self._lifetime.admission():
                return self._operations.inspect_recovery(
                    parse_recovery_request(request, RecoveryRequestV1),
                )
        except ServiceUnavailable:
            raise QuoteMemError("service_unhealthy", "Provider recovery is unavailable") from None

    def resume_recovery(self, request: ResumeRequestV1) -> ResumeResponseV1:
        """Acknowledge only the existing owner's completed durable transition."""
        try:
            with self._lifetime.admission():
                return self._operations.resume_recovery(
                    parse_recovery_request(request, ResumeRequestV1),
                )
        except ServiceUnavailable:
            raise QuoteMemError("service_unhealthy", "Provider recovery is unavailable") from None

    def inspect_completion_budget(
        self, request: CompletionRecoveryRequestV2 | CompletionRecoveryRequestV3,
    ) -> CompletionRecoveryResponseV2 | CompletionRecoveryResponseV3:
        """Inspect the selected version through the same initialized controller."""
        try:
            with self._lifetime.admission():
                if not isinstance(self._operations, CompletionRecoveryOperations):
                    raise QuoteMemError("service_unhealthy", "Completion recovery is unavailable")
                return self._operations.inspect_completion_budget(
                    parse_completion_request(
                        request, CompletionRecoveryRequestV3
                        if isinstance(request, CompletionRecoveryRequestV3)
                        else CompletionRecoveryRequestV2,
                    ),
                )
        except ServiceUnavailable:
            raise QuoteMemError("service_unhealthy", "Completion recovery is unavailable") from None

    def resume_completion_budget(
        self, request: CompletionResumeRequestV2 | CompletionResumeRequestV3,
    ) -> CompletionRecoveryResponseV2 | CompletionRecoveryResponseV3:
        """Grant only the explicitly selected owner's bounded recovery allowance."""
        try:
            with self._lifetime.admission():
                if not isinstance(self._operations, CompletionRecoveryOperations):
                    raise QuoteMemError("service_unhealthy", "Completion recovery is unavailable")
                return self._operations.resume_completion_budget(
                    parse_completion_request(
                        request, CompletionResumeRequestV3
                        if isinstance(request, CompletionResumeRequestV3)
                        else CompletionResumeRequestV2,
                    ),
                )
        except ServiceUnavailable:
            raise QuoteMemError("service_unhealthy", "Completion recovery is unavailable") from None

    def skip_quarantined_cycle(self, request: QuarantineSkipRequestV1) -> QuarantineSkipResponseV1:
        """Skip only the explicitly selected quarantined cycle (``store.quarantine_skip``)."""
        try:
            with self._lifetime.admission():
                if not isinstance(self._operations, QuarantineSkipOperations):
                    raise QuoteMemError("service_unhealthy", "Quarantine skip is unavailable")
                return self._operations.skip_quarantined_cycle(parse_skip_request(request))
        except ServiceUnavailable:
            raise QuoteMemError("service_unhealthy", "Quarantine skip is unavailable") from None
