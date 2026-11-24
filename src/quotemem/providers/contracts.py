"""Transport-neutral provider contracts and the provider output registry.

Requests, results, bindings and the embedding shape live here, together with the
one registry of provider output contracts. `register_output` records a
``(purpose, schema_id)`` pair with its strict output model, the advertiser that
builds the strict JSON Schema of one actual request, and the decoder of a reply.
Every registration is a call in the static registration site at the end of this
module, which imports each contract owner's core module directly; an owner never
registers itself on import. A pair registered twice fails at import, and an
unregistered pair fails with ``unsupported_variant``.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, Any, Literal, NoReturn, Protocol, cast

from pydantic import Field, ValidationError, model_validator

from quotemem.core.conflict_judge import CONFLICT_SCHEMA_ID, ConflictVerdictV1, advertise_conflict
from quotemem.core.errors import QuoteMemError
from quotemem.core.extraction_output import (
    EXTRACT_V12_SCHEMA_ID,
    ExtractOutputV12,
    ExtractReplyRefused,
    advertise_extract_v12,
    decode_extract_reply,
)
from quotemem.core.forget_pick import (
    FORGET_PICK_CLASSIFY_SCHEMA_ID,
    FORGET_PICK_REMAINDERS_SCHEMA_ID,
    FORGET_PICK_TELLS_SCHEMA_ID,
    ForgetPickClassifyV1,
    ForgetPickRemaindersV1,
    ForgetPickTellsV1,
    advertise_forget_pick_classify,
    advertise_forget_pick_remainders,
    advertise_forget_pick_tells,
)
from quotemem.core.identity import ULID, ContentBindingDigest, IntegrityDigest, integrity_digest
from quotemem.core.inference import (
    InferenceBindingV1,
    InferenceOutput,
    InferenceRecord,
    InferenceTimeoutMs,
    ProviderTimeoutMs,
    ProviderUsage,
    ReasoningEffort,
    RetryPolicyV1,
    judge_timeout_policy_digest,
)
from quotemem.core.inference import (
    StoredInferenceOutput as StoredInferenceOutput,
)
from quotemem.core.judge_output import (
    JUDGE_BATCH_V8_SCHEMA_ID,
    JUDGE_BATCH_V8_SCHEMA_REVISION,
    JUDGE_V8_SCHEMA_ID,
    JudgeBatchOutputV8,
    JudgeV8,
    advertise_judge,
    advertise_judge_batch_v8,
)
from quotemem.core.judge_timeout import JUDGE_FIRST_TIMEOUT_MS, JUDGE_SECOND_TIMEOUT_MS
from quotemem.core.pool_rerank import RERANK_SCHEMA_ID, RerankOrderV1, advertise_rerank
from quotemem.core.premise_judge import PREMISE_SCHEMA_ID, PremiseVerdictV1, advertise_premise
from quotemem.core.provider_budget import (
    EXTRACTION_REQUEST_CAP,
    MAX_EXTRACTION_ATTEMPT_MS,
    ProviderCallBudgetV2,
    extended_attempt_allowed,
)
from quotemem.core.provider_diagnostics import ProviderAttemptV1
from quotemem.core.reader_note import READER_SCHEMA_ID, ReaderNoteV1, advertise_reader
from quotemem.core.session_rollup import ROLLUP_SCHEMA_ID, SessionRollupV1, advertise_rollup
from quotemem.core.source_labels import (
    SOURCE_LABEL_LIMIT,
    SOURCE_LABELS_VERSION,
    SourceLabelsOutputV1,
)
from quotemem.store.metadata import OpaqueIdSpec, RefSpec, StrictModel

Purpose = Literal["extract", "judge"]
ProviderName = Literal["model_studio", "openai"]
LocalId = Annotated[str, OpaqueIdSpec(kind="local"), Field(min_length=1)]
ContentDigest = Annotated[ContentBindingDigest, OpaqueIdSpec(kind="digest")]
BindingDigest = Annotated[IntegrityDigest, OpaqueIdSpec(kind="digest")]


class Message(StrictModel):
    """Frozen message bytes delivered to a provider, with a closed role set."""

    role: Literal["system", "user", "assistant"]
    content: Annotated[str, Field(min_length=1)]


class InferenceRequest(StrictModel):
    """The complete request shape; no ambient configuration lookup."""

    purpose: Purpose
    trial_side: Literal["baseline", "proposed"] | None = None
    provider: ProviderName
    requested_model: LocalId
    schema_id: LocalId
    prompt_version: LocalId
    input_digest: ContentDigest
    context_digest: ContentDigest
    messages: Annotated[list[Message], Field(min_length=1)]
    timeout_ms: InferenceTimeoutMs
    retry_policy: RetryPolicyV1
    call_budget: ProviderCallBudgetV2 | None = Field(
        default=None, exclude_if=lambda value: value is None,
    )
    retry_output_schema: bool = Field(default=False, exclude_if=lambda value: not value)
    reasoning_effort: ReasoningEffort | None = Field(
        default=None, exclude_if=lambda value: value is None,
    )
    judge_attempt: Literal[1, 2] | None = Field(
        default=None, exclude_if=lambda value: value is None,
    )
    judge_timeout_policy_digest: BindingDigest | None = Field(
        default=None, exclude_if=lambda value: value is None,
    )
    batch_id: LocalId | None = None
    max_output_tokens: Annotated[int, Field(ge=1, le=131_072)] | None = Field(
        default=None, exclude_if=lambda value: value is None,
    )
    """The reply's completion bound, reasoning included (the batched judge and Search's
    premise judge)."""
    thinking: Literal["disabled"] | None = Field(
        default=None, exclude_if=lambda value: value is None,
    )
    """``disabled`` sends the call without model reasoning, in place of its reasoning effort
    (Search's premise judge, ``core/premise_judge.py``, its pool re-rank,
    ``core/pool_rerank.py``, its conflict check, ``core/conflict_judge.py``, its reader
    note, ``core/reader_note.py``, the ``agent`` mode's forget picks,
    ``service/forget_statement.py``, the extraction supplement,
    ``core/extraction_supplement.py``, and the session rollup, ``core/session_rollup.py``)."""

    @model_validator(mode="after")
    def current_variant(self) -> InferenceRequest:
        """Reserve paired policy trials for their future implementation milestone."""
        if self.trial_side is not None:
            raise QuoteMemError("unsupported_variant", "Policy trial inference is not implemented")
        if (self.reasoning_effort is not None or self.thinking is not None) and (
            self.provider != "openai"
        ):
            raise QuoteMemError(
                "binding_mismatch", "Reasoning effort binds only the OpenAI-compatible provider",
            )
        batch = self.purpose == "judge" and self.schema_id == JUDGE_BATCH_V8_SCHEMA_ID
        if (batch and self.retry_policy.max_attempts != 1
                and self.retry_policy.max_read_timeouts != 1):
            raise QuoteMemError("binding_mismatch",
                                "Memory judge attempts stop on their first read timeout")
        if self.thinking is not None and not (
            (self.purpose == "judge"
             and self.schema_id in {PREMISE_SCHEMA_ID, RERANK_SCHEMA_ID, CONFLICT_SCHEMA_ID,
                                    READER_SCHEMA_ID, FORGET_PICK_CLASSIFY_SCHEMA_ID,
                                    FORGET_PICK_REMAINDERS_SCHEMA_ID,
                                    FORGET_PICK_TELLS_SCHEMA_ID})
            or (self.purpose == "extract"
                and self.schema_id in {EXTRACT_V12_SCHEMA_ID, ROLLUP_SCHEMA_ID})
        ):
            raise QuoteMemError(
                "binding_mismatch",
                "Only Search's premise, re-rank and conflict checks, its reader note, the"
                " agent mode's forget picks, the extraction supplement and the session rollup"
                " go without reasoning",
            )
        if (self.judge_attempt is None) != (self.judge_timeout_policy_digest is None):
            raise QuoteMemError("binding_mismatch",
                                "Judge attempt and timeout identity belong together")
        if self.judge_attempt is not None and not batch:
            raise QuoteMemError("binding_mismatch",
                                "Only memory judge requests carry a timeout policy")
        if batch and self.timeout_ms > (
            JUDGE_FIRST_TIMEOUT_MS if self.judge_attempt == 1 else JUDGE_SECOND_TIMEOUT_MS
        ):
            raise QuoteMemError("binding_mismatch", "Judge timeout exceeds its numbered attempt")
        if not batch and self.timeout_ms > MAX_EXTRACTION_ATTEMPT_MS:
            raise QuoteMemError("binding_mismatch",
                                "Only memory judge timeouts may exceed 900000 ms")
        if self.timeout_ms > 300000 and not (batch and self.judge_attempt is not None) and (
            self.purpose != "extract" or self.schema_id != EXTRACT_V12_SCHEMA_ID
            or not extended_attempt_allowed(self.call_budget, self.timeout_ms)
        ):
            raise QuoteMemError("binding_mismatch", "Extended timeout requires bounded extract-v12")
        if self.call_budget is not None and (
            self.purpose != "extract" or self.timeout_ms != self.call_budget.read_ms
            or self.retry_policy.max_attempts > EXTRACTION_REQUEST_CAP
        ):
            raise QuoteMemError(
                "binding_mismatch", "Extraction call budget differs from timeout/retries",
            )
        if self.retry_output_schema and (
            self.purpose != "extract" or self.schema_id != EXTRACT_V12_SCHEMA_ID
            or self.call_budget is None
        ):
            raise QuoteMemError(
                "binding_mismatch", "Output-schema regeneration requires bounded extract-v12",
            )
        return self


class InferenceResult(StrictModel):
    """Validated semantic output plus frozen model resolution and accounting.

    ``provider_attempts`` is present only after an output-schema regeneration
    and keeps every physical request of the logical call, at most
    ``EXTRACTION_REQUEST_CAP``.

    ``admission`` is never set by a provider. The extraction call attaches the
    reply's admission record after admitting it, so the result is the
    complete ``llm_output`` content that ``stored_output`` binds to its request.
    """

    resolved_model: LocalId
    schema_id: LocalId
    output: InferenceOutput
    usage: ProviderUsage
    provider_request_id: Annotated[str, OpaqueIdSpec(kind="external")] | None = None
    provider_attempts: list[ProviderAttemptV1] = Field(
        default_factory=list[ProviderAttemptV1], exclude_if=lambda value: not value,
    )
    admission: InferenceRecord | None = Field(
        default=None, exclude_if=lambda value: value is None,
    )


class OutputIdentity(StrictModel):
    """Complete durable lookup key, with the owning Run distinct from input content.

    ``call_no`` is the logical extraction call of a unit: its fresh
    call (1) sends the identical request as its call (0), so the call number
    keeps each call's accepted reply apart.
    """

    run_id: Annotated[ULID, RefSpec(target_kind="run", scope_rule="same_scope")]
    purpose: Literal["extract", "judge"]
    input_digest: ContentDigest
    inference_binding_digest: BindingDigest
    call_no: Literal[0, 1] = Field(default=0, exclude_if=lambda value: value == 0)


class InferenceOutputStore(Protocol):
    """The authoritative Writer supplies persistence; providers never write HeavenBase."""

    def get_output(self, identity: OutputIdentity) -> StoredInferenceOutput | None:
        """Load the creating Run's exact output identity, if already durable."""
        ...

    def put_output_once(
        self,
        identity: OutputIdentity,
        request: InferenceRequest,
        result: InferenceResult,
        summary_context_digest: ContentBindingDigest | None = None,
    ) -> StoredInferenceOutput:
        """Persist an immutable llm_output or reject conflicting envelope bytes."""
        ...


class StructuredInferenceProvider(Protocol):
    """One external structured inference boundary."""

    def infer(self, request: InferenceRequest) -> InferenceResult:
        """Invoke the frozen request under its retry policy and strict output model."""
        ...


class EmbeddingRequest(StrictModel):
    """Bounded embedding input with the workspace's immutable dimension."""

    requested_model: Literal["qwen3.7-text-embedding"] = "qwen3.7-text-embedding"
    dimensions: Literal[1024, 2048]
    texts: Annotated[list[str], Field(min_length=1)]
    timeout_ms: ProviderTimeoutMs
    retry_policy: RetryPolicyV1

    @model_validator(mode="after")
    def nonempty_texts(self) -> EmbeddingRequest:
        """Reject empty provider inputs without replacing them with fabricated vectors."""
        if any(not text for text in self.texts):
            raise ValueError("Embedding texts must be nonempty")
        return self


class EmbeddingResult(StrictModel):
    """Input-ordered finite vectors and the provider's resolved model name."""

    resolved_model: LocalId
    dimensions: Literal[1024, 2048]
    vectors: Annotated[list[list[float]], Field(min_length=1)]
    usage: ProviderUsage

    @model_validator(mode="after")
    def finite_dimensions(self) -> EmbeddingResult:
        """Validate dimension, finite components, and usable nonzero vector norms."""
        for vector in self.vectors:
            if (
                len(vector) != self.dimensions
                or any(not math.isfinite(value) for value in vector)
                or not any(vector)
            ):
                raise ValueError("Embedding vector must be finite, nonzero, and dimension-correct")
        return self


class EmbeddingProvider(Protocol):
    """One embedding boundary; batching is an adapter detail."""

    def embed(self, request: EmbeddingRequest) -> EmbeddingResult:
        """Return exactly one vector per input in its original order."""
        ...


def validate_binding(request: InferenceRequest, binding: InferenceBindingV1) -> IntegrityDigest:
    """Require every invocation field to come from the Run's frozen single binding.

    A binding that pins its prompt also requires the request's system
    message to be exactly that prompt, and one that pins a judge-batch-v8 schema
    revision requires the revision this code advertises and accepts.

    Args:
        request (InferenceRequest): The request about to be sent.
        binding (InferenceBindingV1): The Run's frozen binding.

    Returns:
        IntegrityDigest: The binding digest that identifies reusable outputs.

    Raises:
        QuoteMemError: binding_mismatch if any bound field, the pinned prompt or the
            pinned schema revision differs.
    """
    if binding.schema_revision not in (None, JUDGE_BATCH_V8_SCHEMA_REVISION):
        raise QuoteMemError("binding_mismatch", "Judge binding pins another schema revision")
    if binding.prompt_digest is not None and (
        request.messages[0].role != "system"
        or hashlib.sha256(request.messages[0].content.encode("utf-8")).hexdigest()
        != binding.prompt_digest
    ):
        raise QuoteMemError("binding_mismatch", "Inference request changed its pinned prompt")
    fields = (
        "purpose", "provider", "requested_model", "schema_id", "prompt_version",
        "reasoning_effort",
    )
    if any(getattr(request, field) != getattr(binding, field) for field in fields):
        raise QuoteMemError("binding_mismatch",
                            "Inference request differs from its frozen Run binding")
    if binding.judge_timeout_policy_id is None:
        if request.timeout_ms != binding.timeout_ms:
            raise QuoteMemError("binding_mismatch",
                                "Inference timeout differs from its frozen binding")
    else:
        if request.judge_attempt is None or request.judge_timeout_policy_digest != (
            judge_timeout_policy_digest(binding)
        ):
            raise QuoteMemError("binding_mismatch", "Judge attempt differs from its frozen policy")
        limit = (binding.timeout_ms if request.judge_attempt == 1
                 else binding.judge_second_timeout_ms)
        assert limit is not None
        if not 1 <= request.timeout_ms <= limit:
            raise QuoteMemError("binding_mismatch",
                                "Judge timeout exceeds its frozen attempt limit")
    if request.retry_policy != binding.retry_policy:
        raise QuoteMemError("binding_mismatch",
                            "Inference retry policy differs from its frozen binding")
    if request.call_budget != binding.call_budget:
        raise QuoteMemError("binding_mismatch",
                            "Inference call budget differs from its frozen binding")
    if request.retry_output_schema != binding.retry_output_schema:
        raise QuoteMemError("binding_mismatch",
                            "Output-schema retry differs from its frozen binding")
    if integrity_digest(request.retry_policy) != binding.retry_policy_digest:
        raise QuoteMemError("binding_mismatch", "Inference retry policy digest does not match")
    return integrity_digest(binding)


OutputFault = Literal["content_json", "output_schema", "output_bound"]
"""Why a reply's content was refused: not decodable JSON, the wrong structure, or
an output beyond its bound, which is never regenerated."""

Advertise = Callable[[InferenceRequest], dict[str, Any]]
"""Build the strict JSON Schema of one actual request, its enums read from it."""

Decode = Callable[[str], InferenceOutput]
"""Decode the content of one reply; refusals raise `OutputRefused`."""

BindRecord = Callable[[InferenceOutput, InferenceRecord], None]
"""Require an admission record to describe one decoded reply; raise ``schema_invalid``."""


class OutputRefused(QuoteMemError):
    """A reply's content refused by its output contract.

    Attributes:
        fault (OutputFault): ``content_json``, ``output_schema`` or ``output_bound``.
        validation (ValidationError | None): The model's refusal of a wrong
            structure, kept only for content-free diagnostics.
    """

    def __init__(self, fault: OutputFault, validation: ValidationError | None = None) -> None:
        """Create the refusal with its fault and, for a structure, the model's errors."""
        super().__init__("schema_invalid", f"Provider output refused: {fault}")
        self.fault: OutputFault = fault
        self.validation = validation


@dataclass(frozen=True, slots=True)
class OutputContract:
    """One registered provider output contract.

    Attributes:
        purpose (Purpose): The inference purpose that requests it.
        schema_id (str): The frozen output schema identity.
        model (type[InferenceOutput]): The strict model of a decoded reply and of
            a stored output.
        advertise (Advertise): The strict JSON Schema of one actual request.
        decode (Decode): The decoder of one reply's content.
        bind_record (BindRecord | None): For a contract whose ``llm_output``
            keeps an admission record beside the verbatim reply, the
            check that a record describes its reply; None keeps no record.
    """

    purpose: Purpose
    schema_id: str
    model: type[InferenceOutput]
    advertise: Advertise
    decode: Decode
    bind_record: BindRecord | None = None


_OUTPUTS: dict[tuple[str, str], OutputContract] = {}


def _distinct_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    decoded: dict[str, Any] = {}
    for key, value in pairs:
        if key in decoded:
            raise ValueError("Duplicate JSON object key")
        decoded[key] = value
    return decoded


def _finite_only(constant: str) -> NoReturn:
    raise ValueError(f"Non-finite JSON number {constant}")


def json_decoder(model: type[InferenceOutput], *, distinct_keys: bool = False) -> Decode:
    """Return a decoder that parses JSON and validates it with ``model``.

    Args:
        model (type[InferenceOutput]): The strict output model.
        distinct_keys (bool): Refuse a repeated object key anywhere and any
            non-finite number, as acceptance requires of a judge envelope.

    Returns:
        Decode: The decoder. It raises `OutputRefused` with ``content_json``
            for content that does not decode and ``output_schema`` for a
            structure the model refuses.
    """
    def decode(content: str) -> InferenceOutput:
        try:
            raw: object = json.loads(
                content, object_pairs_hook=_distinct_keys, parse_constant=_finite_only,
            ) if distinct_keys else json.loads(content)
        except (ValueError, RecursionError):
            raise OutputRefused("content_json") from None
        try:
            return model.model_validate(raw)
        except ValidationError as error:
            raise OutputRefused("output_schema", error) from None

    return decode


def register_output(
    purpose: Purpose, schema_id: str, model: type[InferenceOutput], advertise: Advertise,
    decode: Decode | None = None, *, bind_record: BindRecord | None = None,
) -> OutputContract:
    """Register one provider output contract; only the static site below calls this.

    Args:
        purpose (Purpose): The inference purpose that requests it.
        schema_id (str): The frozen output schema identity.
        model (type[InferenceOutput]): The strict model of the decoded output.
        advertise (Advertise): Builds the strict JSON Schema of one request.
        decode (Decode | None): The reply decoder; None decodes plain JSON and
            validates it with ``model``.
        bind_record (BindRecord | None): Accepts an admission record beside the
            reply and checks that it describes the reply; None keeps no record.

    Returns:
        OutputContract: The registered contract.

    Raises:
        ValueError: If the pair is already registered.
    """
    if (purpose, schema_id) in _OUTPUTS:
        raise ValueError(f"Output contract {purpose}/{schema_id} is registered twice")
    contract = OutputContract(
        purpose, schema_id, model, advertise, json_decoder(model) if decode is None else decode,
        bind_record,
    )
    _OUTPUTS[purpose, schema_id] = contract
    return contract


def output_contract(purpose: str, schema_id: str) -> OutputContract:
    """Return the registered contract of a pair.

    Args:
        purpose (str): The inference purpose.
        schema_id (str): The output schema identity.

    Returns:
        OutputContract: The registered contract.

    Raises:
        QuoteMemError: unsupported_variant if the pair is not registered.
    """
    contract = _OUTPUTS.get((purpose, schema_id))
    if contract is None:
        raise QuoteMemError("unsupported_variant", "Inference output schema is not implemented")
    return contract


def registered_outputs() -> tuple[OutputContract, ...]:
    """Return every registered contract in registration order."""
    return tuple(_OUTPUTS.values())


def strict_model_schema(model: type[StrictModel]) -> dict[str, Any]:
    """Return the strict JSON Schema of a pydantic output model.

    Every object is closed and requires all its properties, no property keeps
    a default, and tagged alternatives become ``anyOf``.

    Args:
        model (type[StrictModel]): The output model.

    Returns:
        dict[str, Any]: The schema.
    """
    schema = model.model_json_schema()
    _close_objects(schema)
    return schema


def _close_objects(node: Any) -> None:
    if isinstance(node, dict):
        fields = cast(dict[str, Any], node)
        if fields.get("type") == "object":
            fields["additionalProperties"] = False
            fields["required"] = list(fields.get("properties", {}))
        # OpenAI requires nullable fields to be required and rejects default.
        fields.pop("default", None)
        fields.pop("discriminator", None)
        if "oneOf" in fields:
            # Tagged alternatives remain disjoint through their const fields.
            fields["anyOf"] = fields.pop("oneOf")
        for child in fields.values():
            _close_objects(child)
    elif isinstance(node, list):
        for child in cast(list[Any], node):
            _close_objects(child)


def _source_label_decision_ids(request: InferenceRequest) -> list[int]:
    """Derive schema choices only from the frozen first-party label prompt.

    The prompt lists undecided labels only, each as original text without byte
    positions or a state, so every listed label_id is a pending decision.
    """
    if len(request.messages) != 2 or [message.role for message in request.messages] != [
        "system", "user",
    ]:
        raise QuoteMemError("binding_mismatch", "Source-label request changed its message roles")
    try:
        payload: object = json.loads(request.messages[1].content)
    except ValueError:
        raise QuoteMemError("binding_mismatch", "Source-label request lost its payload") from None
    if not isinstance(payload, dict):
        raise QuoteMemError("binding_mismatch", "Source-label request lost its payload")
    body = cast(dict[str, object], payload)
    if body.get("rule_version") != SOURCE_LABELS_VERSION:
        raise QuoteMemError("binding_mismatch", "Source-label request changed its rule version")
    messages = body.get("messages")
    if not isinstance(messages, list):
        raise QuoteMemError("binding_mismatch", "Source-label request lost its messages")
    pending: list[int] = []
    previous = -1
    for message in cast(list[object], messages):
        if not isinstance(message, dict):
            raise QuoteMemError("binding_mismatch", "Source-label request lost its labels")
        labels = cast(dict[str, object], message).get("labels")
        if not isinstance(labels, list):
            raise QuoteMemError("binding_mismatch", "Source-label request lost its labels")
        for label in cast(list[object], labels):
            if not isinstance(label, dict):
                raise QuoteMemError("binding_mismatch", "Source-label request changed a label")
            row = cast(dict[str, object], label)
            identity = row.get("label_id")
            if type(identity) is not int or set(row) != {
                "label_id", "label", "line", "line_number",
            }:
                raise QuoteMemError("binding_mismatch",
                                    "Source-label request changed label positions")
            position = identity
            if not previous < position < SOURCE_LABEL_LIMIT:
                raise QuoteMemError("binding_mismatch",
                                    "Source-label request changed label positions")
            previous = position
            pending.append(position)
    if not pending:
        raise QuoteMemError("binding_mismatch", "Source-label request has no pending decisions")
    return pending


def _advertise_source_labels(request: InferenceRequest) -> dict[str, Any]:
    """Advertise source-labels-v2 with exactly the request's pending label ids."""
    schema = SourceLabelsOutputV1.model_json_schema()
    pending = _source_label_decision_ids(request)
    decisions = schema["properties"]["decisions"]
    decisions["minItems"] = decisions["maxItems"] = len(pending)
    schema["$defs"]["SourceLabelDecisionV1"]["properties"]["label_id"]["enum"] = pending
    _close_objects(schema)
    return schema


def _decode_extract(content: str) -> ExtractOutputV12:
    """Decode an extract-v12 reply, keeping its fault as an `OutputRefused` fault."""
    try:
        return decode_extract_reply(content)
    except ExtractReplyRefused as refused:
        if refused.fault == "output_bound":
            raise OutputRefused("output_bound") from None
        cause = refused.__cause__
        if isinstance(cause, ValidationError):
            raise OutputRefused("output_schema", cause) from None
        raise OutputRefused("content_json") from None


def _bind_extraction_admission(output: InferenceOutput, record: InferenceRecord) -> None:
    """Require an extract-v12 admission record to count exactly its reply's facts."""
    if not isinstance(output, ExtractOutputV12) or record.reply_facts != len(output.facts):
        raise QuoteMemError("schema_invalid", "Admission record describes another reply")


# The static registration site. Each row's content belongs to its
# contract owner: source labels, extract-v12 and judge-batch-v8 each to their stage.
# query-rewrite-v1 left with the recall v4 cutover.
register_output("extract", SOURCE_LABELS_VERSION, SourceLabelsOutputV1, _advertise_source_labels)
register_output(
    "extract", EXTRACT_V12_SCHEMA_ID, ExtractOutputV12, advertise_extract_v12, _decode_extract,
    bind_record=_bind_extraction_admission,
)
register_output(
    "judge", JUDGE_BATCH_V8_SCHEMA_ID, JudgeBatchOutputV8, advertise_judge_batch_v8,
    json_decoder(JudgeBatchOutputV8, distinct_keys=True),
)

register_output(
    "judge", JUDGE_V8_SCHEMA_ID, JudgeV8, advertise_judge,
    json_decoder(JudgeV8, distinct_keys=True),
)

# Forget by statement's pick (core/forget_pick): two strict replies of the
# product judge's ``memory-forget-pick-v1`` prompt.
register_output(
    "judge", FORGET_PICK_CLASSIFY_SCHEMA_ID, ForgetPickClassifyV1,
    advertise_forget_pick_classify,
    json_decoder(ForgetPickClassifyV1, distinct_keys=True),
)
register_output(
    "judge", FORGET_PICK_REMAINDERS_SCHEMA_ID, ForgetPickRemaindersV1,
    advertise_forget_pick_remainders,
    json_decoder(ForgetPickRemaindersV1, distinct_keys=True),
)
register_output(
    "judge", FORGET_PICK_TELLS_SCHEMA_ID, ForgetPickTellsV1,
    advertise_forget_pick_tells,
    json_decoder(ForgetPickTellsV1, distinct_keys=True),
)

# Search's premise judge (core/premise_judge): one strict verdict per vote.
register_output(
    "judge", PREMISE_SCHEMA_ID, PremiseVerdictV1, advertise_premise,
    json_decoder(PremiseVerdictV1, distinct_keys=True),
)

# Search's pool re-rank (core/pool_rerank): one strict order per batch of candidates.
register_output(
    "judge", RERANK_SCHEMA_ID, RerankOrderV1, advertise_rerank,
    json_decoder(RerankOrderV1, distinct_keys=True),
)

# Search's conflict check (core/conflict_judge): one strict reply per vote.
register_output(
    "judge", CONFLICT_SCHEMA_ID, ConflictVerdictV1, advertise_conflict,
    json_decoder(ConflictVerdictV1, distinct_keys=True),
)

# Search's reader note (core/reader_note): the delivered facts a question needs.
register_output(
    "judge", READER_SCHEMA_ID, ReaderNoteV1, advertise_reader,
    json_decoder(ReaderNoteV1, distinct_keys=True),
)

# The lean Add's session rollup (core/session_rollup): one strict rollup text per call.
register_output(
    "extract", ROLLUP_SCHEMA_ID, SessionRollupV1, advertise_rollup,
    json_decoder(SessionRollupV1, distinct_keys=True),
)
