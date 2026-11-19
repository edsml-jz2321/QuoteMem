"""Pure complete job/provider bindings for plan.md §5.9 and §11."""

from aml.core.controls import EmbeddingProviderBindingV1, JobProviderBindingsV1
from aml.core.errors import AMLError
from aml.core.extraction_policy import ExtractionPolicyConfigV2, extraction_pipeline
from aml.core.identity import integrity_digest
from aml.core.inference import (
    MAX_JUDGE_CONCURRENCY,
    InferenceBindingV1,
    RetryPolicyV1,
    default_retry_policy,
)
from aml.core.judge_output import JUDGE_BATCH_V8_SCHEMA_ID, JUDGE_BATCH_V8_SCHEMA_REVISION
from aml.core.judge_template_v8 import MEMORY_JUDGE_PROMPT_V8_DIGEST, MEMORY_JUDGE_PROMPT_VERSION
from aml.core.judge_timeout import (
    JUDGE_COMMIT_RESERVE_MS,
    JUDGE_MIN_ATTEMPT_MS,
    JUDGE_SECOND_TIMEOUT_MS,
    JUDGE_TIMEOUT_POLICY_ID,
)
from aml.core.provider_budget import extraction_call_budget
from aml.providers.config import EMBEDDING_MODEL, ProviderConfig

JUDGE_MAX_READ_TIMEOUTS = 1
"""The round owns the one 500 s retry; each transport call stops on its first read timeout."""

LEAN_JUDGE_MODEL = "aml-lean-local-judge-v1"
"""The requested and resolved model of every lean-mode verdict (``ingestion/lean_judge.py``)."""


def memory_judge_binding(
    extract: InferenceBindingV1, *, read_timeout_ms: int, concurrency: int,
) -> InferenceBindingV1:
    """Bind the memory judge to its frozen extraction pipeline (C.5).

    Provider, model, reasoning effort and retry policy come from the pipeline's
    extraction binding; ``JUDGE_MAX_READ_TIMEOUTS`` stops transport read retries,
    because the round sends the explicit second attempt itself. A single-send
    extraction policy remains single-send and needs no redundant read cap. The judge's own
    fields are purpose ``judge``, the
    physical schema ``judge-batch-v8`` at ``JUDGE_BATCH_V8_SCHEMA_REVISION`` (its
    per-member result schema and acceptance), the prompt ``memory-judge-v8`` with its
    pinned digest (which covers the shared definitions block), the per-attempt
    read timeout and ``judge_concurrency`` (C.8 #2a). The extraction call
    budget, timeout and output-schema regeneration never carry over, and at
    run time each attempt is further bounded by the owner's remaining time.

    Args:
        extract (InferenceBindingV1): The pipeline's frozen extraction binding.
        read_timeout_ms (int): The pipeline's judge read timeout per attempt;
            C57 fixes the first attempt at 900,000 ms.
        concurrency (int): ``judge_concurrency`` = min(16, per_user).

    Returns:
        InferenceBindingV1: The judge binding. A change to any of its fields
            changes its digest, so no verdict is reused across bindings.

    Raises:
        AMLError: binding_mismatch if ``extract`` is not an extraction binding.
        ValidationError: If the timeout or the concurrency is out of bounds.
    """
    if extract.purpose != "extract":
        raise AMLError("binding_mismatch", "A judge binding follows its extraction binding")
    retry = RetryPolicyV1.model_validate({
        **extract.retry_policy.model_dump(mode="json"),
        "max_read_timeouts": (None if extract.retry_policy.max_attempts == 1
                              else JUDGE_MAX_READ_TIMEOUTS),
    })
    return InferenceBindingV1(
        purpose="judge",
        provider=extract.provider,
        requested_model=extract.requested_model,
        schema_id=JUDGE_BATCH_V8_SCHEMA_ID,
        prompt_version=MEMORY_JUDGE_PROMPT_VERSION,
        timeout_ms=read_timeout_ms,
        retry_policy=retry,
        retry_policy_digest=integrity_digest(retry),
        reasoning_effort=extract.reasoning_effort,
        prompt_digest=MEMORY_JUDGE_PROMPT_V8_DIGEST,
        concurrency=concurrency,
        schema_revision=JUDGE_BATCH_V8_SCHEMA_REVISION,
        judge_timeout_policy_id=JUDGE_TIMEOUT_POLICY_ID,
        judge_second_timeout_ms=JUDGE_SECOND_TIMEOUT_MS,
        judge_commit_reserve_ms=JUDGE_COMMIT_RESERVE_MS,
        judge_min_attempt_ms=JUDGE_MIN_ATTEMPT_MS,
    )


def job_provider_bindings(
    config: ProviderConfig, extraction: ExtractionPolicyConfigV2,
) -> JobProviderBindingsV1:
    """Bind a bench job to its actual retained extraction policy and configuration.

    An omitted Add mode is validated as the configured default, just like an explicit mode.
    The job's judge is C.5's memory judge of the same pipeline, at the
    pipeline's judge read timeout and the full ``judge_concurrency``.

    Args:
        config (ProviderConfig): The workspace's bench provider configuration.
        extraction (ExtractionPolicyConfigV2): The retained extraction policy.

    Returns:
        JobProviderBindingsV1: The embedding, extraction and judge bindings.

    Raises:
        AMLError: unsupported_variant for another profile; binding_mismatch when
            the retained policy differs from the effective call budget,
            contract, bench pipeline or model.
    """
    if config.profile != "bench":
        raise AMLError("unsupported_variant", "Job bindings require the bench profile")
    parameters = extraction.parameters
    if parameters.call_budget != extraction_call_budget(config.extraction_call_budget):
        raise AMLError("binding_mismatch", "Retained extraction differs from call-budget trial")
    if (
        parameters.judge_mode != config.judge_mode
        or parameters.chat_forget != config.chat_forget
        or parameters.extraction_supplement != config.extraction_supplement
    ):
        raise AMLError("binding_mismatch", "Retained extraction differs from the selected Add mode")
    selected_contract = config.extraction_contract_version
    if selected_contract is not None and not parameters.pipeline_version.startswith(
        f"extract-v{selected_contract}-",
    ):
        raise AMLError("binding_mismatch", "Retained extraction differs from selected contract")
    pipeline = extraction_pipeline(parameters.pipeline_version)
    if pipeline.profile != "bench":
        raise AMLError("binding_mismatch", "Retained extraction differs from the bench profile")
    extract = inference_binding_for_policy(extraction)
    if (extract.provider, extract.requested_model) != (
        config.inference_provider, config.inference_model,
    ):
        raise AMLError("binding_mismatch", "Retained extraction differs from the configured model")
    return JobProviderBindingsV1(
        embedding=EmbeddingProviderBindingV1(
            provider="model_studio",
            requested_model=EMBEDDING_MODEL,
            dimension=config.embedding_dimensions,
        ),
        extract=extract,
        judge=judge_binding_for_policy(extraction),
    )


def judge_binding_for_policy(policy: ExtractionPolicyConfigV2) -> InferenceBindingV1:
    """The owner's frozen judge binding from its frozen extraction policy (C.5).

    The model judge modes' binding (``model``, and ``targeted-v1``'s model stage) is
    ``memory_judge_binding``'s: they send the same prompt, member inputs and verdicts, only
    packed differently, so their verdicts are the model's. The lean mode's (``judge_mode``
    ``local-lean-v1``) is `lean_judge_binding`, the same with ``LEAN_JUDGE_MODEL`` as its
    model: the round executor runs unchanged, and the binding's own digest keeps lean
    verdicts from being reused as model verdicts, or the reverse (C.8 #1). The targeted mode
    reviews the members its gate leaves out under `lean_judge_binding` too.

    Args:
        policy (ExtractionPolicyConfigV2): The owner's frozen extraction policy.

    Returns:
        InferenceBindingV1: The judge binding.
    """
    if policy.parameters.judge_mode == "local-lean-v1":
        return lean_judge_binding(policy)
    pipeline = extraction_pipeline(policy.parameters.pipeline_version)
    return memory_judge_binding(
        inference_binding_for_policy(policy), read_timeout_ms=pipeline.judge_read_timeout_ms,
        concurrency=MAX_JUDGE_CONCURRENCY,
    )


def lean_judge_binding(policy: ExtractionPolicyConfigV2) -> InferenceBindingV1:
    """The local judge's binding: the policy's memory-judge binding with ``LEAN_JUDGE_MODEL``.

    The lean mode reviews every member with it; the targeted mode (``targeted-v1``) every
    member its gate leaves out, so a lean verdict has one binding in both modes.

    Args:
        policy (ExtractionPolicyConfigV2): The owner's frozen extraction policy.

    Returns:
        InferenceBindingV1: The lean judge binding.
    """
    pipeline = extraction_pipeline(policy.parameters.pipeline_version)
    model = memory_judge_binding(
        inference_binding_for_policy(policy), read_timeout_ms=pipeline.judge_read_timeout_ms,
        concurrency=MAX_JUDGE_CONCURRENCY,
    )
    return InferenceBindingV1.model_validate(
        {**model.model_dump(mode="json"), "requested_model": LEAN_JUDGE_MODEL},
    )


def inference_binding_for_policy(config: ExtractionPolicyConfigV2) -> InferenceBindingV1:
    """Bind a frozen extraction policy to its pipeline's provider and call budget (B.5).

    The extraction call binds it; source labels, the memory judge and forget follow it.

    Args:
        config (ExtractionPolicyConfigV2): The owner's frozen extraction policy.

    Returns:
        InferenceBindingV1: Purpose ``extract`` with the pipeline's provider,
            model and reasoning effort, the policy's schema, pipeline, call
            budget and output-schema regeneration, and the published retry policy.
    """
    parameters = config.parameters
    pipeline = extraction_pipeline(parameters.pipeline_version)
    retry = default_retry_policy()
    return InferenceBindingV1(
        purpose="extract",
        provider=pipeline.provider,
        requested_model=pipeline.requested_model,
        schema_id=parameters.extractor_schema_id,
        prompt_version=parameters.pipeline_version,
        timeout_ms=parameters.call_budget.read_ms,
        retry_policy=retry,
        retry_policy_digest=integrity_digest(retry),
        call_budget=parameters.call_budget,
        retry_output_schema=parameters.retry_output_schema,
        reasoning_effort=pipeline.reasoning_effort,
    )
