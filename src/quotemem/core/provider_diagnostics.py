"""Bounded provider observations without prompts, responses or retry authority."""

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from quotemem.core.judge_timeout import JUDGE_FIRST_TIMEOUT_MS, JUDGE_SECOND_TIMEOUT_MS
from quotemem.core.provider_budget import MAX_EXTRACTION_ATTEMPT_MS, MAX_EXTRACTION_CALL_MS
from quotemem.store.metadata import OpaqueIdSpec, StrictModel

type ProviderErrorClass = Literal[
    "rate_limited", "server_error", "timeout", "client_error", "schema_invalid", "unavailable",
]
type TimeoutPhase = Literal["connect", "pool", "write", "read", "deadline", "unknown"]
type ProviderSchemaReason = Literal[
    "response_json", "response_shape", "message_refusal", "truncated",
    "content_json", "output_schema", "output_bound", "usage_metadata", "model_metadata",
    "request_metadata",
]
type ProviderFieldPath = Literal[
    "$", "choices", "choices[0]", "choices[0].message", "choices[0].finish_reason",
    "choices[0].message.refusal", "choices[0].message.content", "usage", "model", "id",
    "data", "data[].index", "data[].embedding",
]
type ProviderOutputSchema = Literal[
    "source-labels-v2", "extract-v12",
]
"""Registered output contracts whose structural refusals keep a content-free description.

A judge-batch-v8 envelope has none: its members are accepted or refused one by one,
and a refused envelope is never persisted."""
# Length bounds of text and arrays and invalid Unicode keep their own categories;
# ``value_constraint`` covers the remaining numeric, pattern and parsing bounds.
type ProviderOutputErrorCategory = Literal[
    "missing", "extra_field", "type_mismatch", "too_short", "too_long", "invalid_unicode",
    "value_constraint", "literal_mismatch", "union_tag", "custom_validation", "schema_invalid",
]

OUTPUT_DIAGNOSTIC_ISSUES = 8
OUTPUT_DIAGNOSTIC_DEPTH = 8
OUTPUT_DIAGNOSTIC_SEGMENT_CHARS = 64


class ProviderOutputIssueV1(StrictModel):
    """One safe category and schema-owned path; array positions never retain indices."""

    category: ProviderOutputErrorCategory
    field_path: Annotated[
        list[Annotated[str, Field(min_length=1, max_length=OUTPUT_DIAGNOSTIC_SEGMENT_CHARS)]],
        Field(max_length=OUTPUT_DIAGNOSTIC_DEPTH),
    ]
    path_limited: bool


class ProviderOutputValidationV1(StrictModel):
    """Bound output errors to the selected immutable schema, including on artifact reads."""

    schema_id: Annotated[ProviderOutputSchema, OpaqueIdSpec("local")]
    issues: Annotated[
        list[ProviderOutputIssueV1], Field(min_length=1, max_length=OUTPUT_DIAGNOSTIC_ISSUES),
    ]
    truncated: bool

    @model_validator(mode="after")
    def validate_paths(self) -> Self:
        from quotemem.core.provider_output_diagnostics import output_path_is_owned

        if any(not output_path_is_owned(self.schema_id, issue.field_path) for issue in self.issues):
            raise ValueError("Output diagnostic path is not owned by its selected schema")
        return self


class ProviderSchemaRefusalV1(StrictModel):
    """Only program-owned reason and path constants cross the diagnostic boundary."""

    reason: ProviderSchemaReason
    field_path: ProviderFieldPath
    output_validation: ProviderOutputValidationV1 | None = Field(
        default=None, exclude_if=lambda value: value is None,
    )

    @model_validator(mode="after")
    def validate_output_detail(self) -> Self:
        if self.reason in _CONTENT_REASONS and self.field_path != "choices[0].message.content":
            raise ValueError("A content refusal requires its exact content path")
        if self.output_validation is not None and (
            self.reason != "output_schema" or self.field_path != "choices[0].message.content"
        ):
            raise ValueError("Output validation details require an actual output-schema refusal")
        return self


_CONTENT_REASONS = frozenset({"content_json", "output_schema", "output_bound"})
_REGENERABLE: frozenset[tuple[str, str]] = frozenset({
    ("content_json", "choices[0].message.content"),
    ("output_schema", "choices[0].message.content"),
    ("truncated", "choices[0].finish_reason"),
})


def regenerable_refusal(refusal: ProviderSchemaRefusalV1 | None) -> bool:
    """Whether a reply's refusal is one a regeneration may fix.

    Content that is not JSON, the wrong structure and a truncated reply (a
    finish reason other than stop) may be regenerated within the call's shared
    send cap. An output beyond its bound never is, nor is a model
    refusal or a wrong reply envelope.

    Args:
        refusal (ProviderSchemaRefusalV1 | None): The refusal, if any.

    Returns:
        bool: True for a regenerable refusal.
    """
    return refusal is not None and (refusal.reason, refusal.field_path) in _REGENERABLE


class ProviderAttemptV1(StrictModel):
    """One actual HTTP send and its decision under the original retry binding.

    ``timeout_ms`` is the existing HTTPX inactivity setting, not a measured
    wall-clock limit. Backoff is the scheduled delay, not an extra attempt.
    ``attempt`` numbers the physical requests of one logical call, network
    retries and output-schema regenerations together. The transport never
    records a budgeted attempt, one with ``attempt_wall_ms``, beyond
    ``EXTRACTION_REQUEST_CAP`` (3); the Finalize failure and stored-output
    readers also refuse more attempts than the frozen retry policy allows.
    The field's bound of 8 belongs to unbudgeted policies. ``judge_attempt``
    identifies the first or second memory-judge attempt and permits only that
    attempt's extended timeout. Other calls keep their original timeout bounds.
    """

    attempt: Annotated[int, Field(ge=1, le=8)]
    elapsed_ms: Annotated[int, Field(ge=0, le=2**63 - 1)]
    timeout_ms: Annotated[int, Field(ge=1, le=JUDGE_FIRST_TIMEOUT_MS)]
    judge_attempt: Literal[1, 2] | None = Field(
        default=None, exclude_if=lambda value: value is None,
    )
    outcome: ProviderErrorClass | Literal["response"]
    timeout_phase: TimeoutPhase | None = None
    status_code: Annotated[int, Field(ge=100, le=599)] | None = None
    decision: Literal["retry", "exhausted", "budget_exhausted", "not_retryable", "response"]
    backoff_ms: Annotated[int, Field(ge=0, le=2**63 - 1)] = 0
    attempt_wall_ms: Annotated[int, Field(ge=1, le=MAX_EXTRACTION_ATTEMPT_MS)] | None = Field(
        default=None, exclude_if=lambda value: value is None,
    )
    logical_remaining_ms: Annotated[int, Field(ge=1, le=MAX_EXTRACTION_CALL_MS)] | None = Field(
        default=None, exclude_if=lambda value: value is None,
    )
    schema_refusal: ProviderSchemaRefusalV1 | None = Field(
        default=None, exclude_if=lambda value: value is None,
    )
    input_tokens: Annotated[int, Field(ge=0, le=2**63 - 1)] | None = Field(
        default=None, exclude_if=lambda value: value is None,
    )
    output_tokens: Annotated[int, Field(ge=0, le=2**63 - 1)] | None = Field(
        default=None, exclude_if=lambda value: value is None,
    )

    @model_validator(mode="after")
    def validate_observation(self) -> Self:
        """Reject diagnostics that imply an unobserved timeout or retry."""
        if self.judge_attempt is not None and self.timeout_ms > (
            JUDGE_FIRST_TIMEOUT_MS if self.judge_attempt == 1 else JUDGE_SECOND_TIMEOUT_MS
        ):
            raise ValueError("Judge timeout diagnostics exceed their numbered attempt")
        if self.judge_attempt is None and self.timeout_ms > MAX_EXTRACTION_ATTEMPT_MS:
            raise ValueError("Only judge timeout diagnostics allow more than 900000 ms")
        if self.judge_attempt is None and self.timeout_ms > 300000 and (
            self.attempt_wall_ms is None or self.logical_remaining_ms is None
        ):
            raise ValueError("Extended timeout diagnostics require the actual call budget")
        if (self.outcome == "timeout") != (self.timeout_phase is not None):
            raise ValueError("Timeout observations require their actual phase")
        if (self.outcome == "response") != (self.decision == "response"):
            raise ValueError("Only accepted transport responses have a response decision")
        if self.backoff_ms and self.decision not in {"retry", "budget_exhausted"}:
            raise ValueError("Only a scheduled retry has backoff")
        if self.schema_refusal is not None and self.outcome != "schema_invalid":
            raise ValueError("Only a schema-invalid attempt has a schema refusal")
        if self.schema_refusal is not None and self.schema_refusal.reason == (
            "output_schema"
        ) and (self.status_code is None or not 200 <= self.status_code <= 299):
            raise ValueError("Output-schema refusal requires an accepted HTTP response")
        return self


class ProviderFailureDetailV1(StrictModel):
    """The final HTTP call's bounded attempts and subsequent adapter refusal.

    ``attempts`` keeps every physical request of the refused logical call in
    order; a budgeted call records at most ``EXTRACTION_REQUEST_CAP``. A final
    ``exhausted`` decision means the attempt cap ended a retryable failure;
    ``budget_exhausted`` means the logical budget left no room for another
    request.
    """

    error_class: ProviderErrorClass
    attempts: Annotated[list[ProviderAttemptV1], Field(min_length=1, max_length=8)]
    schema_refusal: ProviderSchemaRefusalV1 | None = None
    stop_reason: Literal["logical_budget_exhausted"] | None = Field(
        default=None, exclude_if=lambda value: value is None,
    )

    @model_validator(mode="after")
    def validate_terminal(self) -> Self:
        """Keep sequence and terminal classification consistent with actual events."""
        if [item.attempt for item in self.attempts] != list(range(1, len(self.attempts) + 1)):
            raise ValueError("Provider attempts must retain their original contiguous order")
        if any(item.decision != "retry" for item in self.attempts[:-1]):
            raise ValueError("Only retries can precede another attempt")
        last = self.attempts[-1]
        if (self.stop_reason is not None) != (last.decision == "budget_exhausted"):
            raise ValueError("Logical budget exhaustion must stop actual retries")
        if last.decision == "retry" or last.outcome not in {self.error_class, "response"}:
            raise ValueError("Provider failure lost its actual final attempt")
        if last.outcome == "response" and self.error_class != "schema_invalid":
            raise ValueError("An accepted response can only produce an adapter schema refusal")
        if (self.error_class == "schema_invalid") != (self.schema_refusal is not None):
            raise ValueError("Schema failure requires its exact adapter reason")
        return self


ACCESS_REFUSAL_STATUSES: frozenset[int] = frozenset({401, 403, 429})
"""HTTP statuses of a provider that refused the key itself: authentication, permission or quota.

The provider answered without evaluating the request, so the refusal says
nothing about the unit or member it carried. It ends when the key works again
(or a key proxy rotates to another key), never by spending the request.
"""


def access_refused(detail: ProviderFailureDetailV1 | None) -> bool:
    """Tell whether a failed call's final send was refused for its key's access or quota.

    Args:
        detail: The failed call's bounded HTTP diagnostics, when it has them.

    Returns:
        bool: True when the call's final physical send answered 401, 403 or 429.
    """
    return detail is not None and detail.attempts[-1].status_code in ACCESS_REFUSAL_STATUSES


def access_refused_sends(detail: ProviderFailureDetailV1 | None) -> int:
    """Count a failed call's physical sends the provider refused for access or quota.

    Args:
        detail: The failed call's bounded HTTP diagnostics, when it has them.

    Returns:
        int: The sends that answered 401, 403 or 429; 0 without diagnostics.
    """
    if detail is None:
        return 0
    return sum(1 for attempt in detail.attempts
               if attempt.status_code in ACCESS_REFUSAL_STATUSES)
