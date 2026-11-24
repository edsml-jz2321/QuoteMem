"""Frozen inference bindings and provider output schemas."""

from __future__ import annotations

from typing import Annotated, Literal, Self, get_args

from pydantic import (
    Field,
    ValidationError,
    model_validator,
)

from quotemem.core.conflict_judge import ConflictVerdictV1
from quotemem.core.errors import QuoteMemError
from quotemem.core.extraction_output import EXTRACT_V12_SCHEMA_ID, ExtractOutputV12
from quotemem.core.extraction_records import ExtractionAdmissionV2
from quotemem.core.forget_pick import (
    ForgetPickClassifyV1,
    ForgetPickRemaindersV1,
    ForgetPickTellsV1,
)
from quotemem.core.identity import ContentBindingDigest, IntegrityDigest, integrity_digest
from quotemem.core.judge_output import (
    JUDGE_BATCH_V8_SCHEMA_ID,
    JUDGE_V8_SCHEMA_ID,
    JudgeBatchOutputV8,
    JudgeV8,
)
from quotemem.core.judge_timeout import (
    JUDGE_FIRST_TIMEOUT_MS,
    JUDGE_TIMEOUT_POLICY_ID,
)
from quotemem.core.pool_rerank import RerankOrderV1
from quotemem.core.premise_judge import PremiseVerdictV1
from quotemem.core.provider_budget import (
    EXTRACTION_REQUEST_CAP,
    MAX_EXTRACTION_ATTEMPT_MS,
    ProviderCallBudgetV2,
    extended_attempt_allowed,
)
from quotemem.core.provider_diagnostics import ProviderAttemptV1, regenerable_refusal
from quotemem.core.reader_note import ReaderNoteV1
from quotemem.core.session_rollup import SessionRollupV1
from quotemem.core.source_labels import SourceLabelsOutputV1
from quotemem.store.metadata import OpaqueIdSpec, StrictModel

MAX_JUDGE_CONCURRENCY = 16
"""Upper bound of ``judge_concurrency`` = min(16, per_user)."""
NonNegativeInt = Annotated[int, Field(ge=0)]
PositiveInt = Annotated[int, Field(gt=0)]
ProviderTimeoutMs = Annotated[int, Field(ge=1, le=300000)]
InferenceTimeoutMs = Annotated[int, Field(ge=1, le=JUDGE_FIRST_TIMEOUT_MS)]
DEFAULT_PROVIDER_TIMEOUT_MS = 30000
Digest = Annotated[IntegrityDigest, OpaqueIdSpec("digest")]
# Canonical retry class order. ``unavailable`` is a transport failure other than
# a timeout, such as a refused connection; the published initializer omits it.
RetryError = Literal["rate_limited", "server_error", "timeout", "unavailable"]
InferencePurpose = Literal["extract", "judge"]
ReasoningEffort = Literal["low", "medium", "high"]


class RetryPolicyV1(StrictModel):
    """Freeze attempts, backoff, and retryable boundary failures for one call.

    ``max_read_timeouts`` caps the attempts that may end in a read timeout: the one
    that reaches it is final even while ``timeout`` is retryable and attempts remain.
    The judge binding sets 1 when more than one send is allowed; a single-send
    policy already stops on its first timeout. The round owns the separate
    second attempt. None leaves read timeouts to ``max_attempts`` and is not part
    of the policy's digest.
    """

    max_attempts: Annotated[int, Field(ge=1, le=8)]
    initial_backoff_ms: Annotated[int, Field(ge=0, le=60000)]
    max_backoff_ms: Annotated[int, Field(ge=0, le=60000)]
    multiplier: Annotated[float, Field(ge=1, le=10, allow_inf_nan=False)]
    jitter_mode: Literal["none", "full"]
    retryable_error_classes: list[RetryError]
    max_read_timeouts: Annotated[int, Field(ge=1, le=8)] | None = Field(
        default=None, exclude_if=lambda value: value is None,
    )

    @model_validator(mode="after")
    def validate_policy(self) -> Self:
        """Reject ambiguous class ordering, impossible backoff intervals and idle caps."""
        if self.initial_backoff_ms > self.max_backoff_ms:
            raise ValueError("Initial backoff exceeds maximum backoff")
        expected = [kind for kind in get_args(RetryError) if kind in self.retryable_error_classes]
        if self.retryable_error_classes != expected:
            raise ValueError("Retry classes must be unique and in canonical order")
        if self.max_read_timeouts is not None and self.max_read_timeouts >= self.max_attempts:
            raise ValueError("A read-timeout cap must stop before the attempts run out")
        return self


def default_retry_policy() -> RetryPolicyV1:
    """Return the published initializer policy, frozen explicitly by each request."""
    return RetryPolicyV1(
        max_attempts=3,
        initial_backoff_ms=250,
        max_backoff_ms=2000,
        multiplier=2.0,
        jitter_mode="full",
        retryable_error_classes=["rate_limited", "server_error", "timeout"],
    )


class InferenceBindingV1(StrictModel):
    """Bind a Run to the complete extract or judge execution policy.

    ``reasoning_effort`` is the explicit reasoning mode that the adapter sends.
    Every binding of a current extraction pipeline carries the pipeline's effort,
    whether it binds extraction, source labels or the judge, so each
    binding digest covers it. None sends no reasoning parameter.

    A ``judge-batch-v8`` binding also pins ``prompt_digest``, the SHA-256 of
    the system prompt every request of the binding sends, which covers the
    shared definitions block; ``concurrency``, the ``judge_concurrency`` of
    at most that many physical requests in flight per owner and round;
    and ``schema_revision``, the revision of its per-member result schema and
    acceptance, which the wire's schema name does not show. All three are part
    of the binding digest, so no verdict is reused across a changed prompt,
    concurrency or revision; no other binding carries them. The judge timeout policy also binds the
    timeout policy id, its 900 s first and 500 s second limits, the 60 s commit
    reserve and the 10 s minimum useful attempt. Physical requests carry the
    selected attempt, the policy digest and their effective capped timeout.
    """

    purpose: Literal["extract", "judge"]
    provider: Annotated[str, Field(min_length=1, max_length=128)]
    requested_model: Annotated[str, Field(min_length=1, max_length=256)]
    schema_id: Annotated[str, Field(min_length=1, max_length=128), OpaqueIdSpec("local")]
    prompt_version: Annotated[str, Field(min_length=1, max_length=128)]
    timeout_ms: InferenceTimeoutMs
    retry_policy: RetryPolicyV1
    retry_policy_digest: Digest
    call_budget: ProviderCallBudgetV2 | None = Field(
        default=None, exclude_if=lambda value: value is None,
    )
    retry_output_schema: bool = Field(default=False, exclude_if=lambda value: not value)
    reasoning_effort: ReasoningEffort | None = Field(
        default=None, exclude_if=lambda value: value is None,
    )
    prompt_digest: Digest | None = Field(default=None, exclude_if=lambda value: value is None)
    concurrency: Annotated[int, Field(ge=1, le=MAX_JUDGE_CONCURRENCY)] | None = Field(
        default=None, exclude_if=lambda value: value is None,
    )
    schema_revision: Annotated[int, Field(ge=1)] | None = Field(
        default=None, exclude_if=lambda value: value is None,
    )

    judge_timeout_policy_id: Annotated[
        Literal["judge-read-900-500-v1"], OpaqueIdSpec("local"),
    ] | None = Field(
        default=None, exclude_if=lambda value: value is None,
    )
    judge_second_timeout_ms: Literal[500000] | None = Field(
        default=None, exclude_if=lambda value: value is None,
    )
    judge_commit_reserve_ms: Literal[60000] | None = Field(
        default=None, exclude_if=lambda value: value is None,
    )
    judge_min_attempt_ms: Literal[10000] | None = Field(
        default=None, exclude_if=lambda value: value is None,
    )

    @model_validator(mode="after")
    def validate_digest(self) -> Self:
        """Require the frozen retry digest to match every retry-policy field."""
        if self.retry_policy_digest != integrity_digest(self.retry_policy):
            raise ValueError("Retry policy digest does not match its frozen policy")
        if self.reasoning_effort is not None and self.provider != "openai":
            raise ValueError("Reasoning effort binds only the OpenAI-compatible provider")
        batch = self.purpose == "judge" and self.schema_id == JUDGE_BATCH_V8_SCHEMA_ID
        if any(batch != (pinned is not None)
               for pinned in (self.prompt_digest, self.concurrency, self.schema_revision,
                              self.judge_timeout_policy_id, self.judge_second_timeout_ms,
                              self.judge_commit_reserve_ms, self.judge_min_attempt_ms)):
            raise ValueError(
                "Exactly a judge-batch-v8 binding pins its prompt, concurrency, revision "
                "and timeout policy",
            )
        if (batch and self.retry_policy.max_attempts != 1
                and self.retry_policy.max_read_timeouts != 1):
            raise ValueError("Memory judge attempts stop on their first read timeout")
        if batch and self.timeout_ms != JUDGE_FIRST_TIMEOUT_MS:
            raise ValueError("The current judge timeout policy requires a 900000 ms first attempt")
        if not batch and self.timeout_ms > MAX_EXTRACTION_ATTEMPT_MS:
            raise ValueError("Only memory judge timeouts may exceed 900000 ms")
        if not batch and self.timeout_ms > 300000 and (
            self.purpose != "extract" or self.schema_id != EXTRACT_V12_SCHEMA_ID
            or not extended_attempt_allowed(self.call_budget, self.timeout_ms)
        ):
            raise ValueError("Extended timeout requires bounded extract-v12")
        if self.call_budget is not None and (
            self.purpose != "extract" or self.timeout_ms != self.call_budget.read_ms
            or self.retry_policy.max_attempts > EXTRACTION_REQUEST_CAP
        ):
            raise ValueError(
                "Extraction call budget requires its read timeout and at most 3 attempts",
            )
        if self.retry_output_schema and (
            self.purpose != "extract" or self.schema_id != EXTRACT_V12_SCHEMA_ID
            or self.call_budget is None
        ):
            raise ValueError("Output-schema regeneration requires bounded extract-v12")
        return self


def judge_timeout_policy_digest(binding: InferenceBindingV1) -> IntegrityDigest:
    """Bind both judge limits and its commit reserve into each physical request.

    Args:
        binding: The owner's frozen memory-judge binding.

    Returns:
        IntegrityDigest: Identity of the complete judge timeout policy.

    Raises:
        QuoteMemError: If this is not the current memory-judge timeout policy.
    """
    if binding.judge_timeout_policy_id != JUDGE_TIMEOUT_POLICY_ID:
        raise QuoteMemError("binding_mismatch",
                            "A judge request requires its frozen timeout policy")
    return integrity_digest({
        "policy_id": binding.judge_timeout_policy_id,
        "first_timeout_ms": binding.timeout_ms,
        "second_timeout_ms": binding.judge_second_timeout_ms,
        "commit_reserve_ms": binding.judge_commit_reserve_ms,
        "min_attempt_ms": binding.judge_min_attempt_ms,
    })


class JudgeBatchAccountingV1(StrictModel):
    """Whether one member's stored judgment bears its physical batch request's usage.

    Each physical request's tokens are recorded once by the transport-level
    observation, including a response without any accepted member. A member
    judgment binds only its own logical input; it records which physical
    request answered it and whether it carries that request's tokens, never
    the other members of the batch.
    """

    request_digest: Digest
    charged: bool
    batch_input_tokens: NonNegativeInt
    batch_output_tokens: NonNegativeInt


class ProviderUsageV1(StrictModel):
    """Record token counts without a redundant independently writable total."""

    input_tokens: NonNegativeInt
    output_tokens: NonNegativeInt
    judge_batch: JudgeBatchAccountingV1 | None = Field(
        default=None, exclude_if=lambda value: value is None,
    )

    @model_validator(mode="after")
    def charge_batch_once(self) -> Self:
        batch = self.judge_batch
        if batch is not None:
            expected = ((batch.batch_input_tokens, batch.batch_output_tokens)
                        if batch.charged else (0, 0))
            if (self.input_tokens, self.output_tokens) != expected:
                raise ValueError("Physical batch usage belongs only to its charged member")
        return self


ProviderUsage = ProviderUsageV1


InferenceOutput = (SourceLabelsOutputV1 | ExtractOutputV12 | JudgeBatchOutputV8 | JudgeV8
                   | ForgetPickClassifyV1 | ForgetPickRemaindersV1 | ForgetPickTellsV1
                   | PremiseVerdictV1 | RerankOrderV1 | ConflictVerdictV1 | ReaderNoteV1
                   | SessionRollupV1)
"""Every model registered with `quotemem.providers.contracts.register_output`."""

InferenceRecord = ExtractionAdmissionV2
"""Every admission record an output contract registers to keep beside its reply.

Only extract-v12 registers one: the content-free ``ExtractionAdmissionV2`` that
program admission computed from the verbatim reply.
"""


class StoredInferenceOutput(StrictModel):
    """Strict llm_output body; the Artifact envelope supplies the creating Run.

    ``provider_attempts`` is present only after an output-schema regeneration.
    It keeps every physical request of the one logical extraction call, at most
    ``EXTRACTION_REQUEST_CAP``, and ends with the accepted reply.

    ``admission`` is the reply's admission record, present only for an
    output contract that registers one and only when admission changed the
    reply; the verbatim ``output`` is never edited. The inline bound covers the
    wrapper without it; a body the record makes larger is chunked with ``part``.

    ``call_no`` is the logical extraction call of the unit whose reply this is:
    0 for the unit's call, 1 for its one fresh call after a reply
    without facts. It belongs to the output identity, so each call's accepted
    reply has its own create-once body; only an extract-v12 output is a fresh
    call, and call 0 is the body without the key.
    """

    purpose: Literal["extract", "judge"]
    provider: Literal["model_studio", "openai"]
    requested_model: Annotated[str, OpaqueIdSpec("local"), Field(min_length=1)]
    resolved_model: Annotated[str, OpaqueIdSpec("local"), Field(min_length=1)]
    schema_id: Annotated[str, OpaqueIdSpec("local"), Field(min_length=1)]
    prompt_version: Annotated[str, OpaqueIdSpec("local"), Field(min_length=1)]
    timeout_ms: InferenceTimeoutMs
    retry_policy_digest: Digest
    inference_binding_digest: Digest
    input_digest: Annotated[ContentBindingDigest, OpaqueIdSpec("digest")]
    context_digest: Annotated[ContentBindingDigest, OpaqueIdSpec("digest")]
    summary_context_digest: Annotated[ContentBindingDigest, OpaqueIdSpec("digest")] | None = None
    output_schema: Annotated[str, OpaqueIdSpec("local"), Field(min_length=1)]
    output: InferenceOutput
    usage: ProviderUsage
    provider_attempts: list[ProviderAttemptV1] = Field(
        default_factory=list[ProviderAttemptV1], exclude_if=lambda value: not value,
    )
    admission: InferenceRecord | None = Field(
        default=None, exclude_if=lambda value: value is None,
    )
    call_no: Literal[0, 1] = Field(default=0, exclude_if=lambda value: value == 0)

    @model_validator(mode="after")
    def output_parity(self) -> Self:
        """Reject a purpose/schema mismatch even after loading archived data."""
        if self.call_no and self.schema_id != EXTRACT_V12_SCHEMA_ID:
            raise ValueError("Only an extraction reply has a fresh logical call")
        if self.timeout_ms > 300000 and (
            (self.purpose, self.schema_id) not in {
                ("extract", EXTRACT_V12_SCHEMA_ID), ("judge", JUDGE_V8_SCHEMA_ID),
            } or (self.purpose == "extract" and self.timeout_ms > MAX_EXTRACTION_ATTEMPT_MS)
        ):
            raise ValueError("Extended timeout requires bounded extraction or a judge member")
        if self.output_schema != self.schema_id:
            raise ValueError("Stored output schema must match its frozen schema identity")
        self.output = validate_output(
            self.purpose, self.schema_id, self.output.model_dump(mode="json"),
        )
        validate_record(self.purpose, self.schema_id, self.output, self.admission)
        if self.purpose == "judge" and self.schema_id != JUDGE_V8_SCHEMA_ID:
            raise ValueError("Only accepted judge members are stored, never physical batches")
        member = self.purpose == "judge" and self.schema_id == JUDGE_V8_SCHEMA_ID
        if member != (self.usage.judge_batch is not None):
            raise ValueError("Exactly a stored judge-v8 member carries batch accounting")
        if self.schema_id == EXTRACT_V12_SCHEMA_ID:
            from quotemem.core.extraction_bounds import (
                INLINE_ARTIFACT_CHARS,
                MAX_USAGE,
                METADATA_CANONICAL_CHARS,
                canonical_limit,
            )

            for value in (
                self.requested_model,
                self.resolved_model,
                self.schema_id,
                self.prompt_version,
                self.output_schema,
            ):
                canonical_limit(value, METADATA_CANONICAL_CHARS, "inference metadata")
            if max(self.usage.input_tokens, self.usage.output_tokens) > MAX_USAGE:
                raise QuoteMemError("schema_invalid",
                                    "Extraction usage exceeds the v2 integer bound")
            canonical_limit(
                self.model_dump(mode="json", exclude={"admission"}), INLINE_ARTIFACT_CHARS,
                "stored inference wrapper",
            )
        if self.purpose != "extract" and self.summary_context_digest is not None:
            raise ValueError("Summary context provenance is exclusive to extraction")
        if any(
            attempt.schema_refusal is not None
            and attempt.schema_refusal.output_validation is not None
            and attempt.schema_refusal.output_validation.schema_id != self.schema_id
            for attempt in self.provider_attempts
        ):
            raise ValueError("A stored attempt describes another output schema")
        if self.provider_attempts and (
            self.schema_id != EXTRACT_V12_SCHEMA_ID
            or len(self.provider_attempts) > EXTRACTION_REQUEST_CAP
            or [attempt.attempt for attempt in self.provider_attempts]
            != list(range(1, len(self.provider_attempts) + 1))
            or any(attempt.decision != "retry" for attempt in self.provider_attempts[:-1])
            or self.provider_attempts[-1].decision != "response"
            or not any(
                regenerable_refusal(attempt.schema_refusal)
                for attempt in self.provider_attempts[:-1]
            )
        ):
            raise ValueError("Stored regeneration attempts must end in a valid extraction")
        return self


def validate_output(purpose: str, schema_id: str, raw: object) -> InferenceOutput:
    """Parse one registered provider output contract without accepting unknown variants.

    The contract comes from the provider output registry
    (`quotemem.providers.contracts.output_contract`), which imports this module, so
    the registry is read when the function runs.

    Args:
        purpose: One supported inference purpose.
        schema_id: The frozen versioned output contract identifier.
        raw: The decoded JSON object of an output.

    Returns:
        A strict typed output corresponding to the requested purpose and schema.

    Raises:
        QuoteMemError: unsupported_variant if the pair is not registered;
            schema_invalid if the output violates its strict model.
    """
    from quotemem.providers.contracts import output_contract

    model = output_contract(purpose, schema_id).model
    try:
        return model.model_validate(raw)
    except ValidationError as error:
        raise QuoteMemError("schema_invalid",
                            "Provider output violates its strict schema") from error



def validate_record(
    purpose: str, schema_id: str, output: InferenceOutput, record: InferenceRecord | None,
) -> None:
    """Require an admission record to belong to its registered output contract.

    The contract comes from the provider output registry
    (`quotemem.providers.contracts.output_contract`), read when the function runs.

    Args:
        purpose: One supported inference purpose.
        schema_id: The frozen versioned output contract identifier.
        output: The strict typed reply the record describes.
        record: The admission record kept beside it, or None.

    Raises:
        QuoteMemError: unsupported_variant if the pair is not registered;
            schema_invalid if the contract keeps no record or the record does
            not describe this reply.
    """
    if record is None:
        return
    from quotemem.providers.contracts import output_contract

    bind = output_contract(purpose, schema_id).bind_record
    if bind is None:
        raise QuoteMemError("schema_invalid", "This output contract keeps no admission record")
    bind(output, record)
