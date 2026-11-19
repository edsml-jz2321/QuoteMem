"""Bound benchmark answer and rubric calls, outside Search and core Run inference.

Remote adapters reuse the authoritative JSON transport without persistence,
memoization, fallback, or recovery. Fixture providers are explicit immutable test
artifacts keyed by complete requests; they never call or populate a remote cache.
The rubric output follows BEAM's published prompt with a stricter local parser,
not its permissive JSON repair/coercion or event-ordering evaluator.
"""

from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from pathlib import Path
from typing import Annotated, Any, Literal, Protocol, Self, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_validator,
)

from aml.adapters.http.models import SearchRequestV1, parse_search_response
from aml.bench.answer_packing import AnswerPackingPolicy
from aml.bench.datasets import BenchmarkSource, source_identity, source_identity_kind
from aml.bench.mem0_binding import Mem0Registration, Mem0RetrievalBinding, mem0_search_response
from aml.bench.replay import ReplayConfig
from aml.core.errors import AMLError
from aml.core.identity import canonical_json
from aml.core.inference import ProviderTimeoutMs, RetryPolicyV1
from aml.core.provider_diagnostics import ProviderSchemaRefusalV1
from aml.providers.adapters import JsonTransport, ProviderError
from aml.providers.contracts import ProviderName

Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Nonempty = Annotated[str, Field(min_length=1)]
_SOURCE: TypeAdapter[BenchmarkSource] = TypeAdapter(BenchmarkSource)


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, hide_input_in_errors=True)


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, value in pairs:
        if name in result:
            raise ValueError("Duplicate JSON field")
        result[name] = value
    return result


def _nonfinite(_: str) -> None:
    raise ValueError("Nonfinite JSON value")


def _object(value: str, *, exact_numbers: bool = False) -> dict[str, Any]:
    raw: object = json.loads(
        value,
        object_pairs_hook=_unique,
        parse_constant=_nonfinite,
        parse_float=Decimal if exact_numbers else float,
    )
    if not isinstance(raw, dict):
        raise ValueError("Expected a JSON object")
    return cast(dict[str, Any], raw)


def benchmark_digest(value: object) -> str:
    """Hash canonical benchmark artifact values, without borrowing core Run identities."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def prompt_digest(value: str) -> str:
    """Bind the exact UTF-8 template bytes separately from rendered messages."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class HttpRetrievalBinding(_Frozen):
    """Normalized Add inputs; actual Search wire bytes are bound separately."""

    kind: Literal["aml_http"]
    add_input_sha256: Digest
    replay_config_json: str

    @model_validator(mode="after")
    def explicit_replay(self) -> Self:
        """Require a closed replay configuration independently supplied by orchestration."""
        ReplayConfig.model_validate_json(canonical_json(_object(self.replay_config_json)))
        return self


class VerbatimRetrievalBinding(_Frozen):
    """Explicit local comparator input and method identity, never an HTTP observation."""

    kind: Literal["verbatim_bm25_v1"]
    add_input_sha256: Digest
    replay_config_json: str
    source_units_sha256: Digest
    method_json: str

    @model_validator(mode="after")
    def complete_method(self) -> Self:
        """Retain a supplied nonempty method object without interpreting its owner’s schema."""
        if not _object(self.method_json):
            raise ValueError("A verbatim comparator requires its exact method descriptor")
        ReplayConfig.model_validate_json(canonical_json(_object(self.replay_config_json)))
        return self


_RETRIEVAL_BINDING: TypeAdapter[
    HttpRetrievalBinding | VerbatimRetrievalBinding | Mem0RetrievalBinding
] = TypeAdapter(HttpRetrievalBinding | VerbatimRetrievalBinding | Mem0RetrievalBinding)


class BenchmarkContext(_Frozen):
    """Complete case provenance, separating HTTP wire bytes from local comparator artifacts."""

    dataset: Nonempty
    source_revision: Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
    source_identity_kind: Literal[
        "raw_file_sha256", "selected_file_manifest_v1", "bundle_manifest_v1",
        "locomo_conversation_selection_v1",
        "received_message_prefix_v1",
    ]
    source_identity: Digest
    retrieval_origin: Literal["aml_http", "verbatim_bm25_v1", "mem0_oss_v2_1_0"]
    retrieval_binding_json: Annotated[str, Field(repr=False)]
    source_selection_json: Annotated[str, Field(repr=False)]
    run_key: Nonempty
    trial_key: Nonempty
    case_id: Nonempty
    query: Annotated[str, Field(min_length=1, repr=False)]
    query_source_date: Annotated[Nonempty | None, Field(repr=False)]
    user_id: Annotated[str, Field(min_length=1, repr=False)]
    source_question_json: Annotated[str | None, Field(repr=False)]
    search_request_json: Annotated[str, Field(repr=False)]
    search_response_json: Annotated[str, Field(repr=False)]

    @model_validator(mode="after")
    def complete_source_and_search(self) -> Self:
        """Reject source substitutions and mismatched query/user Search request bytes."""
        source = _SOURCE.validate_json(canonical_json(_object(self.source_selection_json)))
        binding = _RETRIEVAL_BINDING.validate_json(
            canonical_json(_object(self.retrieval_binding_json))
        )
        if binding.kind != self.retrieval_origin:
            raise ValueError("Benchmark retrieval origin differs from its exact binding")
        if (
            source.dataset != self.dataset
            or source.revision != self.source_revision
            or source_identity(source) != self.source_identity
            or source_identity_kind(source) != self.source_identity_kind
        ):
            raise ValueError("Benchmark source provenance differs from its selected descriptor")
        request = SearchRequestV1.from_body(self.search_request_json.encode(), max_top_k=1000)
        response = parse_search_response(self.search_response_json.encode())
        if (
            request.query != self.query
            or request.user_id != self.user_id
            or len(response.data) > request.top_k
        ):
            raise ValueError("Benchmark Search request/result differs from its bound case")
        if self.source_question_json is not None:
            _object(self.source_question_json)
        if isinstance(binding, Mem0RetrievalBinding):
            registration = Mem0Registration.model_validate_json(binding.registration_json)
            if (
                registration.source != source or self.case_id not in registration.case_ids
                or registration.run_key != self.run_key or registration.trial_key != self.trial_key
                or binding.receipt.case_id != self.case_id
                or binding.receipt.user_id != self.user_id
                or binding.receipt.request_sha256 != prompt_digest(self.search_request_json)
                or mem0_search_response(binding.raw_result_json, top_k=request.top_k) != response
            ):
                raise ValueError("Mem0 context differs from its registered local receipt")
        return self


class BenchmarkMessage(_Frozen):
    """Exact caller-rendered prompt messages; the evaluator owns prompt composition."""

    role: Literal["system", "user", "assistant"]
    content: Annotated[str, Field(min_length=1, repr=False)]


class BenchmarkPolicy(_Frozen):
    """Frozen remote policy; shared mutable retry models are revalidated at dispatch."""

    provider: ProviderName
    requested_model: Nonempty
    expected_resolved_model: Nonempty | None
    schema_id: Literal["bench-answer-v1", "beam-rubric-score-v1"]
    prompt_version: Nonempty
    prompt_template: Annotated[str, Field(min_length=1, repr=False)]
    prompt_sha256: Digest
    context_packing_json: Annotated[str | None, Field(repr=False)]
    timeout_ms: ProviderTimeoutMs
    retry_policy: RetryPolicyV1
    retry_policy_sha256: Digest

    @model_validator(mode="after")
    def complete_policy(self) -> Self:
        """Require exact retry fields and template bytes to match registered digests."""
        if self.prompt_sha256 != prompt_digest(self.prompt_template):
            raise ValueError("Benchmark prompt bytes differ from their registered digest")
        if self.retry_policy_sha256 != benchmark_digest(self.retry_policy.model_dump(mode="json")):
            raise ValueError("Benchmark retry policy differs from its registered digest")
        if self.context_packing_json is not None:
            if self.schema_id != "bench-answer-v1" or not _object(self.context_packing_json):
                raise ValueError("Only answer policies can bind a nonempty context packing policy")
        return self


class _Request(_Frozen):
    context: BenchmarkContext
    policy: BenchmarkPolicy
    messages: Annotated[tuple[BenchmarkMessage, ...], Field(min_length=1, repr=False)]
    request_sha256: Digest

    @model_validator(mode="after")
    def complete_request(self) -> Self:
        """Bind every nested field and message byte, including retry and trial coordinates."""
        if (
            benchmark_digest(self.model_dump(mode="json", exclude={"request_sha256"}))
            != self.request_sha256
        ):
            raise ValueError("Benchmark request differs from its complete registered identity")
        return self


class AnswerRequest(_Request):
    """One external answer-generation call, after Search has delivered its response."""

    purpose: Literal["benchmark_answer"]

    @model_validator(mode="after")
    def answer_schema(self) -> Self:
        """Keep answer generation separate from rubric and core inference schemas."""
        if self.policy.schema_id != "bench-answer-v1":
            raise ValueError("An answer request requires its own benchmark output schema")
        return self


class RubricRequest(_Request):
    """One BEAM criterion ordinal, retaining duplicates and the complete ordered rubric."""

    purpose: Literal["beam_rubric_item"]
    prediction: Annotated[str, Field(repr=False)]
    criteria: Annotated[tuple[Nonempty, ...], Field(min_length=1, repr=False)]
    criterion_index: Annotated[int, Field(ge=0)]

    @model_validator(mode="after")
    def rubric_schema(self) -> Self:
        """Reject an absent criterion before transport rather than shrinking its denominator."""
        if self.policy.schema_id != "beam-rubric-score-v1" or self.criterion_index >= len(
            self.criteria
        ):
            raise ValueError(
                "Rubric call requires its exact schema and a registered criterion ordinal"
            )
        return self


def answer_request(
    context: BenchmarkContext, policy: BenchmarkPolicy, messages: tuple[BenchmarkMessage, ...]
) -> AnswerRequest:
    """Seal fully supplied answer inputs; this helper performs no rendering or model call."""
    value = {
        "purpose": "benchmark_answer",
        "context": context.model_dump(mode="python", warnings=False),
        "policy": policy.model_dump(mode="python", warnings=False),
        "messages": tuple(item.model_dump(mode="python", warnings=False) for item in messages),
    }
    return AnswerRequest.model_validate(value | {"request_sha256": benchmark_digest(value)})


def rubric_request(
    context: BenchmarkContext,
    policy: BenchmarkPolicy,
    messages: tuple[BenchmarkMessage, ...],
    *,
    prediction: str,
    criteria: tuple[str, ...],
    criterion_index: int,
) -> RubricRequest:
    """Seal one rubric item without deduplicating criteria or changing prompt bytes."""
    value = {
        "purpose": "beam_rubric_item",
        "context": context.model_dump(mode="python", warnings=False),
        "policy": policy.model_dump(mode="python", warnings=False),
        "messages": tuple(item.model_dump(mode="python", warnings=False) for item in messages),
        "prediction": prediction,
        "criteria": criteria,
        "criterion_index": criterion_index,
    }
    return RubricRequest.model_validate(value | {"request_sha256": benchmark_digest(value)})


class AnswerOutput(_Frozen):
    """The answer payload consumed by the local evaluator, never returned by Search."""

    answer: Annotated[str, Field(min_length=1, repr=False)]

    @field_validator("answer")
    @classmethod
    def nonblank(cls, value: str) -> str:
        """Reject an empty response without trimming meaningful answer bytes."""
        if not value.strip():
            raise ValueError("A generated answer cannot contain only whitespace")
        return value


class RubricScore(_Frozen):
    """BEAM prompt output, using a stricter local numeric parser than upstream repair."""

    score: Annotated[
        float, Field(ge=0, le=1, allow_inf_nan=False, json_schema_extra={"enum": [0.0, 0.5, 1.0]})
    ]
    reason: Annotated[str, Field(min_length=1, repr=False)]

    @field_validator("score", mode="before")
    @classmethod
    def strict_numeric_score(cls, value: object) -> float:
        """Reject booleans, strings, nonfinite numbers, and values outside the rubric scale."""
        if (
            type(value) not in (int, float, Decimal)
            or (isinstance(value, Decimal) and not value.is_finite())
            or value not in (0, 0.5, 1)
        ):
            raise ValueError("A rubric score must be the JSON number zero, one half, or one")
        return float(cast(int | float | Decimal, value))

    @field_validator("reason")
    @classmethod
    def nonblank_reason(cls, value: str) -> str:
        """Require explanation content without changing its exact response bytes."""
        if not value.strip():
            raise ValueError("A rubric explanation cannot contain only whitespace")
        return value


class BenchmarkUsage(_Frozen):
    """Only observed generation token counts; missing accounting remains unknown."""

    input_tokens: Annotated[int | None, Field(ge=0)]
    output_tokens: Annotated[int | None, Field(ge=0)]


class _Result(_Frozen):
    request_sha256: Digest
    resolved_model: Nonempty
    resolution_status: Literal["matched", "unverified"]
    provider_request_id: Nonempty | None
    usage: BenchmarkUsage | None


class AnswerResult(_Result):
    """A bound answer with actual provider resolution and available accounting."""

    output: AnswerOutput


class RubricResult(_Result):
    """One bound criterion result; aggregation and event-order diagnostics live elsewhere."""

    output: RubricScore
    criterion_index: Annotated[int, Field(ge=0)]


class AnswerProvider(Protocol):
    """An answer-generation boundary independent of Search and durable core inference."""

    def answer(self, request: AnswerRequest) -> AnswerResult:
        """Generate or replay exactly one request-bound external benchmark answer."""
        ...


class RubricJudge(Protocol):
    """An item-wise rubric boundary whose caller retains the complete case denominator."""

    def judge(self, request: RubricRequest) -> RubricResult:
        """Judge one exact criterion ordinal without changing the case denominator."""
        ...


class ModelResolutionMismatch(AMLError):
    """A response from a different frozen model, distinct from unknown resolution."""

    def __init__(self, expected: str, actual: str) -> None:
        super().__init__("binding_mismatch", "Benchmark provider resolved a different frozen model")
        self.expected_resolved_model = expected
        self.actual_resolved_model = actual


def _resolution(policy: BenchmarkPolicy, actual: object) -> Literal["matched", "unverified"]:
    if not isinstance(actual, str) or not actual:
        raise ProviderError("schema_invalid")
    expected = policy.expected_resolved_model
    if expected is None:
        return "unverified"
    if actual != expected:
        raise ModelResolutionMismatch(expected, actual)
    return "matched"


class BenchmarkOutputLengthError(ProviderError):
    """The actual answer envelope stopped at its model/provider output boundary."""

    def __init__(self) -> None:
        super().__init__("schema_invalid", schema_refusal=ProviderSchemaRefusalV1(
            reason="truncated", field_path="choices[0].finish_reason",
        ))


class RemoteBenchmarkProviders:
    """Uncached opt-in chat adapters sharing the existing caller-owned JSON transport."""

    def __init__(
        self, transport: JsonTransport, provider: ProviderName, *, enabled: bool = False
    ) -> None:
        if type(enabled) is not bool:
            raise ValueError("Remote benchmark opt-in must be an explicit boolean")
        self._transport = transport
        self._provider: ProviderName = provider
        self._enabled = enabled

    def _invoke(
        self,
        request: AnswerRequest | RubricRequest,
        output_model: type[AnswerOutput] | type[RubricScore],
    ) -> dict[str, Any]:
        if not self._enabled:
            raise AMLError("provider_unavailable", "Remote benchmark calls require explicit opt-in")
        if request.policy.provider != self._provider:
            raise AMLError(
                "binding_mismatch", "Benchmark request differs from the selected provider"
            )
        policy = request.policy
        payload: dict[str, Any] = {
            "model": policy.requested_model,
            "messages": [message.model_dump(mode="json") for message in request.messages],
            "stream": False,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": policy.schema_id.replace("-", "_"),
                    "strict": True,
                    "schema": output_model.model_json_schema(),
                },
            },
        }
        if self._provider == "model_studio":
            payload["enable_thinking"] = False
        full_evidence = (
            isinstance(request, AnswerRequest) and policy.context_packing_json is not None
            and _object(policy.context_packing_json).get("strategy") == "all_returned_content_v2"
        )
        if full_evidence:
            assert policy.context_packing_json is not None
            AnswerPackingPolicy.model_validate_json(policy.context_packing_json)
            raw = self._transport.post(
                "/chat/completions", payload, policy.timeout_ms, policy.retry_policy,
                context_limit_provider=self._provider,
            )
        else:
            raw = self._transport.post(
                "/chat/completions", payload, policy.timeout_ms, policy.retry_policy,
            )
        try:
            choices = raw["choices"]
            if not isinstance(choices, list) or len(cast(list[Any], choices)) != 1:
                raise ValueError("Incomplete provider choices")
            choice: Any = cast(list[Any], choices)[0]
            message: Any = choice["message"]
            if full_evidence and choice.get("finish_reason") == "length":
                raise BenchmarkOutputLengthError()
            if choice.get("finish_reason") != "stop" or message.get("refusal"):
                raise ValueError("Incomplete provider response")
            content = message["content"]
            if not isinstance(content, str):
                raise ValueError("Provider content must be a string")
            output = output_model.model_validate(_object(content, exact_numbers=True))
            usage = raw.get("usage")
            if usage is not None and not isinstance(usage, dict):
                raise ValueError("Provider usage must be an object when present")
            counts: dict[str, Any] | None = cast(dict[str, Any] | None, usage)
            return {
                "request_sha256": request.request_sha256,
                "resolved_model": raw["model"],
                "resolution_status": _resolution(policy, raw["model"]),
                "provider_request_id": raw.get("id"),
                "output": output,
                "usage": None
                if counts is None
                else BenchmarkUsage(
                    input_tokens=counts.get("prompt_tokens"),
                    output_tokens=counts.get("completion_tokens"),
                ),
            }
        except (ModelResolutionMismatch, BenchmarkOutputLengthError):
            raise
        except (KeyError, TypeError, AttributeError, ValueError, ValidationError):
            raise ProviderError("schema_invalid") from None

    def answer(self, request: AnswerRequest) -> AnswerResult:
        """Revalidate the full request snapshot before reaching the actual transport."""
        checked = AnswerRequest.model_validate(request.model_dump(mode="python", warnings=False))
        try:
            return AnswerResult.model_validate(self._invoke(checked, AnswerOutput))
        except ValidationError:
            raise ProviderError("schema_invalid") from None

    def judge(self, request: RubricRequest) -> RubricResult:
        """Judge one criterion with the published score/reason schema and no repair."""
        checked = RubricRequest.model_validate(request.model_dump(mode="python", warnings=False))
        try:
            return RubricResult.model_validate(
                self._invoke(checked, RubricScore) | {"criterion_index": checked.criterion_index}
            )
        except ValidationError:
            raise ProviderError("schema_invalid") from None


class FrozenBenchmarkFixture(_Frozen):
    """A fully bound synthetic fixture; it is never produced by a remote memoization path."""

    request: AnswerRequest | RubricRequest
    response: AnswerResult | RubricResult

    @model_validator(mode="after")
    def response_parity(self) -> Self:
        """Require exact request identity, output kind, and rubric ordinal."""
        if self.request.request_sha256 != self.response.request_sha256:
            raise ValueError("Frozen benchmark response differs from its complete request")
        status = _resolution(self.request.policy, self.response.resolved_model)
        if self.response.resolution_status != status:
            raise ValueError("Frozen benchmark model-resolution status differs from its binding")
        if isinstance(self.request, AnswerRequest) != isinstance(self.response, AnswerResult):
            raise ValueError("Frozen benchmark request and output kinds differ")
        if isinstance(self.request, RubricRequest) and isinstance(self.response, RubricResult):
            if self.request.criterion_index != self.response.criterion_index:
                raise ValueError("Frozen rubric response differs from its requested ordinal")
        return self


class FrozenBenchmarkProviders:
    """Explicit frozen responses only, with no remote client, storage, or fallback."""

    def __init__(self, fixtures: tuple[FrozenBenchmarkFixture, ...]) -> None:
        self._fixtures: dict[str, str] = {}
        for fixture in fixtures:
            checked = FrozenBenchmarkFixture.model_validate(
                fixture.model_dump(mode="python", warnings=False)
            )
            key = canonical_json(checked.request.model_dump(mode="json"))
            if key in self._fixtures:
                raise ValueError("Duplicate complete benchmark fixture request")
            self._fixtures[key] = checked.model_dump_json()

    @classmethod
    def load(cls, path: Path) -> FrozenBenchmarkProviders:
        """Load strict fixtures from one explicit file, with no environment lookup."""
        body = path.read_text(encoding="utf-8")
        raw: object = json.loads(body, object_pairs_hook=_unique, parse_constant=_nonfinite)
        # Validate closed-scale scores before normal JSON parsing can round them.
        exact: Any = json.loads(body, parse_float=Decimal)
        if isinstance(exact, list):
            for item in cast(list[Any], exact):
                if isinstance(item, dict):
                    response: Any = cast(dict[str, Any], item).get("response")
                    if isinstance(response, dict):
                        output: Any = cast(dict[str, Any], response).get("output")
                        if isinstance(output, dict) and "score" in output:
                            RubricScore.model_validate(output)
        fixtures = TypeAdapter(tuple[FrozenBenchmarkFixture, ...]).validate_json(
            canonical_json(raw)
        )
        return cls(fixtures)

    def _lookup(self, request: AnswerRequest | RubricRequest) -> AnswerResult | RubricResult:
        checked = type(request).model_validate(request.model_dump(mode="python", warnings=False))
        body = self._fixtures.get(canonical_json(checked.model_dump(mode="json")))
        if body is None:
            raise AMLError("fixture_missing", "No fixture matches the complete benchmark request")
        return FrozenBenchmarkFixture.model_validate_json(body).response

    def answer(self, request: AnswerRequest) -> AnswerResult:
        """Return a fresh validated answer solely for its complete frozen identity."""
        response = self._lookup(request)
        if not isinstance(response, AnswerResult):
            raise AMLError("fixture_missing", "The fixture is not an answer result")
        return response

    def judge(self, request: RubricRequest) -> RubricResult:
        """Return the exact ordinal-bound rubric output without any transport access."""
        response = self._lookup(request)
        if not isinstance(response, RubricResult):
            raise AMLError("fixture_missing", "The fixture is not a rubric result")
        return response
