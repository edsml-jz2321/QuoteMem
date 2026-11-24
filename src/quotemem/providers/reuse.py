"""Create-once inference reuse through the one authoritative Writer interface."""

from __future__ import annotations

from collections.abc import Callable

from pydantic import ValidationError

from quotemem.core.errors import QuoteMemError
from quotemem.core.identity import ContentBindingDigest, integrity_digest
from quotemem.core.inference import (
    InferenceBindingV1,
    InferenceOutput,
    ProviderUsageV1,
    validate_output,
    validate_record,
)
from quotemem.core.provider_diagnostics import ProviderAttemptV1, regenerable_refusal
from quotemem.providers.contracts import (
    InferenceOutputStore,
    InferenceRequest,
    InferenceResult,
    OutputIdentity,
    StoredInferenceOutput,
    StructuredInferenceProvider,
    validate_binding,
)


def stored_result(output: StoredInferenceOutput) -> InferenceResult:
    """Return the provider result one stored output keeps, for a parity rebuild.

    Every rebuild passes it to ``stored_output``, which binds it to the current
    request's contract.
    """
    return InferenceResult(
        resolved_model=output.resolved_model,
        schema_id=output.schema_id,
        output=output.output,
        usage=output.usage,
        provider_attempts=output.provider_attempts,
        admission=output.admission,
    )


def _usage_holds(accepted: ProviderAttemptV1, usage: ProviderUsageV1) -> bool:
    """Whether a reply's usage includes its accepted send's tokens.

    The usage may hold more than that send: a lean-local reply also counts the
    supplement and rollup calls that went beside it (``ingestion.extract.extract``).
    """
    return accepted.input_tokens is not None and accepted.output_tokens is not None and (
        accepted.input_tokens <= usage.input_tokens
        and accepted.output_tokens <= usage.output_tokens
    )


def stored_output(
    identity: OutputIdentity,
    request: InferenceRequest,
    result: InferenceResult,
    summary_context_digest: ContentBindingDigest | None = None,
) -> StoredInferenceOutput:
    """Materialize the exact llm_output body for the authoritative Writer.

    This function creates no durable state. It is shared with Writer so fixture
    reuse and persisted output cannot drift into different body conventions.
    Every stored-output rebuild passes through here too. Fresh replies meet
    their request validators first, as schema faults. An admission record the
    result carries is kept beside the verbatim reply only when the
    request's registered output contract keeps one. The identity's logical
    call number is the body's.
    """
    if identity.purpose != request.purpose or identity.input_digest != request.input_digest:
        raise QuoteMemError("binding_mismatch",
                            "Output identity differs from its inference request")
    if result.schema_id != request.schema_id:
        raise QuoteMemError("binding_mismatch", "Provider response changed the requested schema")
    if result.provider_attempts and not request.retry_output_schema:
        raise QuoteMemError("binding_mismatch", "Unbound output carries regeneration attempts")
    if result.provider_attempts:
        attempts = result.provider_attempts
        budget = request.call_budget
        if budget is None or len(attempts) > request.retry_policy.max_attempts or any(
            attempt.timeout_ms != request.timeout_ms
            or attempt.attempt_wall_ms is None or attempt.logical_remaining_ms is None
            or attempt.attempt_wall_ms > min(
                budget.attempt_wall_ms, attempt.logical_remaining_ms,
            )
            or attempt.logical_remaining_ms > budget.logical_wall_ms
            for attempt in attempts
        ) or any(
            before.logical_remaining_ms is None or after.logical_remaining_ms is None
            or after.logical_remaining_ms > before.logical_remaining_ms
            for before, after in zip(attempts, attempts[1:], strict=False)
        ) or any(
            attempt.outcome not in request.retry_policy.retryable_error_classes
            and not (
                attempt.schema_refusal is not None
                and regenerable_refusal(attempt.schema_refusal)
                and attempt.status_code is not None
                and 200 <= attempt.status_code <= 299
                and (
                    attempt.schema_refusal.output_validation is None
                    or attempt.schema_refusal.output_validation.schema_id == request.schema_id
                )
            )
            for attempt in attempts[:-1]
        ) or not _usage_holds(attempts[-1], result.usage):
            raise QuoteMemError("binding_mismatch",
                                "Regeneration attempts changed their frozen call")
    output = validate_output(
        request.purpose, request.schema_id, result.output.model_dump(mode="json"),
    )
    validate_record(request.purpose, request.schema_id, output, result.admission)
    try:
        return StoredInferenceOutput(
            purpose=request.purpose,
            provider=request.provider,
            requested_model=request.requested_model,
            resolved_model=result.resolved_model,
            schema_id=request.schema_id,
            prompt_version=request.prompt_version,
            timeout_ms=request.timeout_ms,
            retry_policy_digest=integrity_digest(request.retry_policy),
            inference_binding_digest=identity.inference_binding_digest,
            input_digest=request.input_digest,
            context_digest=request.context_digest,
            summary_context_digest=summary_context_digest,
            output_schema=result.schema_id,
            output=result.output,
            usage=result.usage,
            provider_attempts=result.provider_attempts,
            admission=result.admission,
            call_no=identity.call_no,
        )
    except ValidationError as error:
        raise QuoteMemError("schema_invalid",
                            "Stored inference violates its complete schema") from error


def infer_once(
    identity: OutputIdentity,
    binding: InferenceBindingV1,
    request: InferenceRequest,
    provider: StructuredInferenceProvider,
    store: InferenceOutputStore,
    *,
    summary_context_digest: ContentBindingDigest | None = None,
    validate_result: Callable[[InferenceOutput], None] | None = None,
) -> StoredInferenceOutput:
    """Reuse an exact durable output before crossing the actual provider boundary.

    The caller supplies the already admitted Run's frozen binding. Workflow
    recovery and Run admission remain outside this contract coordinator.

    Args:
        identity: Complete output lookup identity of an already admitted Run.
        binding: Exact immutable provider/model/schema/retry policy.
        request: Frozen complete request bytes and provenance digests.
        provider: Actual structured-inference boundary, invoked only on a miss.
        store: The sole Writer output interface.
        summary_context_digest: Optional extraction-only summary provenance.
        validate_result: Code-owned request validator receiving the whole typed
            output before its first persistence and again on durable reuse.

    Returns:
        The completely validated create-once output with its original accounting.

    Raises:
        QuoteMemError: If identity, schema, request-specific validation, or reuse parity fails.
    """
    binding_digest = validate_binding(request, binding)
    if identity.inference_binding_digest != binding_digest:
        raise QuoteMemError("binding_mismatch", "Output lookup uses a different frozen binding")
    if identity.purpose != request.purpose or identity.input_digest != request.input_digest:
        raise QuoteMemError("binding_mismatch", "Output lookup uses a different inference input")
    existing = store.get_output(identity)
    if existing is None:
        result = provider.infer(request)
        if validate_result is not None:
            validate_result(result.output)
        existing = store.put_output_once(identity, request, result, summary_context_digest)
    # Rebuild all parity fields from the frozen request while retaining the
    # original resolved output/accounting. Summary context is provenance only.
    expected = stored_output(
        identity, request, stored_result(existing), existing.summary_context_digest,
    )
    if existing != expected:
        raise QuoteMemError(
            "binding_mismatch", "Stored output does not match its complete frozen binding"
        )
    if validate_result is not None:
        validate_result(existing.output)
    return existing
