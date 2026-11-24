"""Local external answer generation and complete-denominator BEAM rubric judgment.

These answer consumers do not generate within Search, retry providers, or
create caches. Caller-supplied Search contexts and provider implementations retain
their own execution provenance. A complete report alone establishes neither real
model quality nor publisher event-ordering evaluator parity.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from time import perf_counter_ns
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, model_validator

from quotemem.adapters.http.models import parse_search_response
from quotemem.bench.answer_packing import AnswerPackingPolicy as AnswerPackingPolicy
from quotemem.bench.datasets import (
    BenchmarkCase,
    BenchmarkCorpus,
    BenchmarkSource,
    decode_source_json,
    source_matches_cases,
)
from quotemem.bench.providers import (
    AnswerProvider,
    AnswerRequest,
    AnswerResult,
    BenchmarkContext,
    BenchmarkMessage,
    BenchmarkOutputLengthError,
    BenchmarkPolicy,
    Digest,
    ModelResolutionMismatch,
    Nonempty,
    RubricJudge,
    RubricRequest,
    RubricResult,
    answer_request,
    benchmark_digest,
    prompt_digest,
    rubric_request,
)
from quotemem.bench.replay import ReplayConfig, validate_replay_contexts
from quotemem.core.errors import QuoteMemError
from quotemem.core.identity import canonical_json
from quotemem.providers.adapters import ProviderContextLimitError, ProviderError

BEAM_PROMPT_VERSION = "beam-unified-llm-judge-b2da22eac88b-v1"
BEAM_PROMPT_SHA256 = "593373c642a288a7b590577d8a8fc92c3f9a2b70e2f64ad6e59a040a6c56b7f5"
_SOURCE: TypeAdapter[BenchmarkSource] = TypeAdapter(BenchmarkSource)
_ANSWER_PLACEHOLDERS = re.compile(r"<question>|<search_response>|<question_date>")
Duration = Annotated[int | None, Field(ge=0)]


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, hide_input_in_errors=True)


class CallFailure(_Frozen):
    """A payload-free classification; unknown usage and model metadata remain absent."""

    kind: Literal[
        "fixture_missing",
        "provider_timeout",
        "provider_schema_invalid",
        "provider_context_limit",
        "provider_output_length",
        "provider_error",
        "model_resolution_mismatch",
        "response_binding_mismatch",
        "unexpected_error",
    ]
    actual_resolved_model: str | None


class PackedHit(_Frozen):
    """An included source hit with its measured byte retention and truncation witness."""

    index: Annotated[int, Field(ge=0)]
    hit_id: str
    original_content_utf8_bytes: Annotated[int, Field(ge=0)]
    supplied_content_utf8_bytes: Annotated[int, Field(ge=0)]
    truncated: bool


class PackedContext(_Frozen):
    """Exact supplied content-only JSON plus original result and omission identities."""

    original_response_sha256: Digest
    content_json: str = Field(repr=False)
    content_utf8_bytes: Annotated[int, Field(ge=2)]
    content_sha256: Digest
    included: tuple[PackedHit, ...]
    omitted_indices: tuple[Annotated[int, Field(ge=0)], ...]


def _packing_policy(policy: BenchmarkPolicy) -> AnswerPackingPolicy:
    if policy.context_packing_json is None:
        raise QuoteMemError(
            "evaluation_invalid", "Answer orchestration requires a frozen packing policy"
        )
    return AnswerPackingPolicy.model_validate_json(policy.context_packing_json)


def pack_search_context(context: BenchmarkContext, policy: AnswerPackingPolicy) -> PackedContext:
    """Apply the exact registered packing strategy without changing Search order.

    V2 includes every complete returned content string, with no byte cap or
    summary. V1 retains its frozen final-prefix truncation and omission rules.
    IDs, scores and gold labels outside rendered content remain provenance-only.
    """
    checked = AnswerPackingPolicy.model_validate(policy.model_dump(mode="python", warnings=False))
    limit = checked.max_context_utf8_bytes
    hits = parse_search_response(context.search_response_json.encode("utf-8")).data
    contents: list[str] = []
    included: list[PackedHit] = []
    for index, hit in enumerate(hits):
        text = hit.content
        if limit is not None and len(canonical_json([*contents, text]).encode("utf-8")) > limit:
            low, high = 0, len(text)
            while low < high:
                middle = (low + high + 1) // 2
                if (
                    len(canonical_json([*contents, text[:middle]]).encode("utf-8"))
                    <= limit
                ):
                    low = middle
                else:
                    high = middle - 1
            text = text[:low]
            if not text:
                break
        contents.append(text)
        included.append(
            PackedHit(
                index=index,
                hit_id=hit.id,
                original_content_utf8_bytes=len(hit.content.encode("utf-8")),
                supplied_content_utf8_bytes=len(text.encode("utf-8")),
                truncated=text != hit.content,
            )
        )
        if text != hit.content:
            break
    body = canonical_json(contents)
    return PackedContext(
        original_response_sha256=prompt_digest(context.search_response_json),
        content_json=body,
        content_utf8_bytes=len(body.encode("utf-8")),
        content_sha256=prompt_digest(body),
        included=tuple(included),
        omitted_indices=tuple(range(len(included), len(hits))),
    )


def _check_result(
    request: AnswerRequest | RubricRequest, result: AnswerResult | RubricResult
) -> None:
    if result.request_sha256 != request.request_sha256 or isinstance(
        request, AnswerRequest
    ) != isinstance(result, AnswerResult):
        raise QuoteMemError("binding_mismatch",
                            "Benchmark result differs from the dispatched request")
    expected = request.policy.expected_resolved_model
    if expected is not None and expected != result.resolved_model:
        raise ModelResolutionMismatch(expected, result.resolved_model)
    if result.resolution_status != ("unverified" if expected is None else "matched"):
        raise QuoteMemError("binding_mismatch", "Benchmark model resolution status differs")
    if isinstance(request, RubricRequest) and isinstance(result, RubricResult):
        if request.criterion_index != result.criterion_index:
            raise QuoteMemError(
                "binding_mismatch", "Benchmark result differs from its criterion ordinal"
            )


class AnswerCaseObservation(_Frozen):
    """One original corpus case, including absent Search and failed provider outcomes."""

    case_id: Nonempty
    status: Literal["answered", "missing_search", "failed"]
    request: AnswerRequest | None = Field(repr=False)
    result: AnswerResult | None = Field(repr=False)
    packing: PackedContext | None = Field(repr=False)
    duration_ns: Duration
    failure: CallFailure | None

    @model_validator(mode="after")
    def coherent_state(self) -> Self:
        """Only successful bound results may supply evaluator predictions."""
        if self.status == "missing_search":
            if any(
                value is not None
                for value in (
                    self.request,
                    self.result,
                    self.packing,
                    self.duration_ns,
                    self.failure,
                )
            ):
                raise ValueError("An absent Search cannot claim a provider observation")
        elif self.request is None or self.request.context.case_id != self.case_id:
            raise ValueError("Answer observation lacks its exact case request")
        elif self.status == "answered":
            if self.result is None or self.failure is not None:
                raise ValueError("An answered case requires a bound result and no failure")
            _check_result(self.request, self.result)
        elif self.failure is None:
            raise ValueError("A failed answer requires an explicit failure classification")
        if self.request is not None and (
            self.packing
            != pack_search_context(self.request.context, _packing_policy(self.request.policy))
            or self.request.messages
            != render_answer_messages(self.request.context, self.request.policy)
        ):
            raise ValueError(
                "Answer packing or rendered prompt differs from the frozen request policy"
            )
        return self


class AnswerRunReport(_Frozen):
    """Serializable complete case denominator with sealed source and call observations."""

    corpus_sha256: Digest
    source_selection_json: str = Field(repr=False)
    run_key: Nonempty
    trial_key: Nonempty
    policy: BenchmarkPolicy
    replay_config: ReplayConfig
    case_count: Annotated[int, Field(gt=0)]
    answered_count: Annotated[int, Field(ge=0)]
    status: Literal["complete", "incomplete"]
    cases: tuple[AnswerCaseObservation, ...] = Field(repr=False)
    report_sha256: Digest

    @model_validator(mode="after")
    def complete_denominator(self) -> Self:
        """Reject altered counts, report bytes, and mixed trial or prompt policies."""
        source = _SOURCE.validate_json(
            canonical_json(decode_source_json(self.source_selection_json.encode("utf-8")))
        )
        if (
            self.case_count != len(self.cases)
            or not source_matches_cases(source, [case.case_id for case in self.cases])
            or len({case.case_id for case in self.cases}) != self.case_count
            or self.answered_count != sum(case.status == "answered" for case in self.cases)
            or self.status
            != ("complete" if self.answered_count == self.case_count else "incomplete")
            or self.policy.schema_id != "bench-answer-v1"
        ):
            raise ValueError("Answer report denominator or status differs from its cases")
        for case in self.cases:
            if case.request is not None and (
                case.request.context.run_key != self.run_key
                or case.request.context.trial_key != self.trial_key
                or case.request.policy != self.policy
                or ReplayConfig.model_validate_json(
                    json.loads(case.request.context.retrieval_binding_json)["replay_config_json"]
                )
                != self.replay_config
                or json.loads(case.request.context.source_selection_json)
                != source.model_dump(mode="json")
            ):
                raise ValueError(
                    "Answer report mixes source, run, trial, or provider policy bindings"
                )
        if (
            len(
                {
                    case.request.context.retrieval_origin
                    for case in self.cases
                    if case.request is not None
                }
            )
            > 1
        ):
            raise ValueError("An answer report cannot mix retrieval arms")
        if self.report_sha256 != benchmark_digest(
            self.model_dump(mode="json", exclude={"report_sha256"})
        ):
            raise ValueError("Answer report differs from its sealed observations")
        return self


class RubricItemObservation(_Frozen):
    """One criterion ordinal; duplicates and unexecuted items retain separate rows."""

    criterion_index: Annotated[int, Field(ge=0)]
    criterion: str = Field(repr=False)
    status: Literal["complete", "missing_answer", "invalid_rubric", "failed"]
    request: RubricRequest | None = Field(repr=False)
    result: RubricResult | None = Field(repr=False)
    duration_ns: Duration
    failure: CallFailure | None

    @model_validator(mode="after")
    def coherent_state(self) -> Self:
        """Unexecuted items cannot carry scores; complete items bind their exact ordinal."""
        if self.status in ("missing_answer", "invalid_rubric"):
            if any(
                value is not None
                for value in (self.request, self.result, self.duration_ns, self.failure)
            ):
                raise ValueError("An unexecuted rubric item cannot claim a provider observation")
        elif self.request is None or self.request.criterion_index != self.criterion_index:
            raise ValueError("Rubric item lacks its registered ordinal request")
        elif self.request.criteria[self.criterion_index] != self.criterion:
            raise ValueError("Rubric item text differs from its registered criterion")
        elif self.status == "complete":
            if self.result is None or self.failure is not None:
                raise ValueError("A complete rubric item requires its result and no failure")
            _check_result(self.request, self.result)
        elif self.failure is None:
            raise ValueError("A failed rubric item requires its failure classification")
        if self.request is not None and self.request.messages != render_beam_messages(
            self.request.policy, criterion=self.criterion, prediction=self.request.prediction
        ):
            raise ValueError(
                "Rubric request messages differ from the exact pinned publisher prompt"
            )
        return self


class RubricCaseObservation(_Frozen):
    """An item-wise mean exists only when the entire original rubric completed."""

    case_id: Nonempty
    category: str
    status: Literal["complete", "incomplete"]
    incomplete_reason: Literal["missing_answer", "invalid_rubric", "failed_items"] | None
    criterion_count: Annotated[int, Field(ge=0)]
    complete_count: Annotated[int, Field(ge=0)]
    items: tuple[RubricItemObservation, ...] = Field(repr=False)
    mean_score: Annotated[float | None, Field(ge=0, le=1, allow_inf_nan=False)]
    event_ordering_status: Literal["unsupported", "not_applicable"]

    @model_validator(mode="after")
    def complete_mean(self) -> Self:
        """Never shrink the denominator after missing or malformed criterion results."""
        completed = sum(item.status == "complete" for item in self.items)
        complete = bool(self.items) and completed == len(self.items)
        mean = (
            sum(item.result.output.score for item in self.items if item.result is not None)
            / len(self.items)
            if complete
            else None
        )
        if (
            self.criterion_count != len(self.items)
            or tuple(item.criterion_index for item in self.items) != tuple(range(len(self.items)))
            or self.complete_count != completed
            or self.status != ("complete" if complete else "incomplete")
            or self.mean_score != mean
            or (self.incomplete_reason is None) != complete
            or self.event_ordering_status
            != ("unsupported" if self.category == "event_ordering" else "not_applicable")
        ):
            raise ValueError(
                "Rubric case mean or completeness differs from its complete denominator"
            )
        return self


class RubricRunReport(_Frozen):
    """Structurally checked rubric rows; use validate_beam_report with both references.

    Parsing alone cannot verify the original corpus or referenced answer report.
    The macro mean covers these rows and excludes the unsupported event-order metric.
    """

    corpus_sha256: Digest
    answer_report_sha256: Digest
    run_key: Nonempty
    trial_key: Nonempty
    policy: BenchmarkPolicy
    metric: Literal["beam_rubric_macro_case_mean_v1_strict_local_parser"]
    case_count: Annotated[int, Field(gt=0)]
    complete_case_count: Annotated[int, Field(ge=0)]
    criterion_count: Annotated[int, Field(ge=0)]
    complete_criterion_count: Annotated[int, Field(ge=0)]
    status: Literal["complete", "incomplete"]
    mean_score: Annotated[float | None, Field(ge=0, le=1, allow_inf_nan=False)]
    event_ordering_metric: Literal["unsupported"]
    cases: tuple[RubricCaseObservation, ...] = Field(repr=False)

    @model_validator(mode="after")
    def complete_mean(self) -> Self:
        """Keep the case macro denominator separate from unequal rubric lengths."""
        complete = bool(self.cases) and all(case.status == "complete" for case in self.cases)
        mean = (
            sum(case.mean_score for case in self.cases if case.mean_score is not None)
            / len(self.cases)
            if complete
            else None
        )
        if (
            self.case_count != len(self.cases)
            or len({case.case_id for case in self.cases}) != self.case_count
            or self.complete_case_count != sum(case.status == "complete" for case in self.cases)
            or self.criterion_count != sum(case.criterion_count for case in self.cases)
            or self.complete_criterion_count != sum(case.complete_count for case in self.cases)
            or self.status != ("complete" if complete else "incomplete")
            or self.mean_score != mean
        ):
            raise ValueError(
                "Rubric report mean or counts differ from its complete case denominator"
            )
        for case in self.cases:
            criteria = tuple(item.criterion for item in case.items)
            for item in case.items:
                if item.request is not None and (
                    item.request.context.case_id != case.case_id
                    or item.request.context.run_key != self.run_key
                    or item.request.context.trial_key != self.trial_key
                    or item.request.policy != self.policy
                    or item.request.criteria != criteria
                ):
                    raise ValueError(
                        "Rubric report mixes case, trial, policy, or criterion bindings"
                    )
        return self


def _corpus_identity(corpus: BenchmarkCorpus) -> str:
    source = _SOURCE.validate_python(corpus.source.model_dump(mode="python", warnings=False))
    if (
        not corpus.cases
        or not source_matches_cases(source, [case.case_id for case in corpus.cases])
        or len({case.case_id for case in corpus.cases}) != len(corpus.cases)
        or any(not case.case_id or not case.query for case in corpus.cases)
    ):
        raise QuoteMemError(
            "evaluation_invalid", "External answers require the complete declared corpus"
        )
    cases = [asdict(case) for case in corpus.cases]
    # This historical corpus identity describes the original text projection.
    # New caller metadata is separately bound by the selected replay contract
    # and exact Add digest; adding it here would invalidate frozen answer reports.
    for case in cases:
        for session in case["sessions"]:
            for message in session["messages"]:
                message.pop("speaker", None)
                message.pop("source_participant", None)
    return benchmark_digest({"source": source.model_dump(mode="json"), "cases": cases})


def _context_matches(
    corpus: BenchmarkCorpus,
    case: BenchmarkCase,
    context: BenchmarkContext,
    run_key: str,
    trial_key: str,
) -> None:
    if (
        json.loads(context.source_selection_json) != corpus.source.model_dump(mode="json")
        or context.case_id != case.case_id
        or context.query != case.query
        or context.query_source_date != case.query_source_date
        or context.source_question_json != case.source_question_json
        or context.run_key != run_key
        or context.trial_key != trial_key
    ):
        raise QuoteMemError("binding_mismatch",
                            "Search context differs from its corpus case or trial")


def render_answer_messages(
    context: BenchmarkContext, policy: BenchmarkPolicy
) -> tuple[BenchmarkMessage, ...]:
    """Render the versioned caller template once, without interpreting inserted placeholders.

    The template must contain both <question> and <search_response>. The latter
    inserts the registered content-only JSON list; full original result bytes stay
    provenance-only. No answers, rubrics, or source labels are supplied.
    Optional <question_date> inserts the exact corpus-provided date, or the
    literal 'not provided' when absent. This is a versioned local prompt convention.
    """
    if policy.schema_id != "bench-answer-v1" or any(
        placeholder not in policy.prompt_template
        for placeholder in ("<question>", "<search_response>")
    ):
        raise QuoteMemError("evaluation_invalid",
                            "Answer template requires its schema and both inputs")
    packed = pack_search_context(context, _packing_policy(policy))
    values = {
        "<question>": context.query,
        "<search_response>": packed.content_json,
        "<question_date>": context.query_source_date
        if context.query_source_date is not None
        else "not provided",
    }
    text = _ANSWER_PLACEHOLDERS.sub(lambda match: values[match.group()], policy.prompt_template)
    return (BenchmarkMessage(role="user", content=text),)


def render_beam_messages(
    policy: BenchmarkPolicy, *, criterion: str, prediction: str
) -> tuple[BenchmarkMessage, ...]:
    """Use the exact pinned publisher template and its two ordered substitutions only."""
    if (
        policy.schema_id != "beam-rubric-score-v1"
        or policy.prompt_version != BEAM_PROMPT_VERSION
        or policy.prompt_sha256 != BEAM_PROMPT_SHA256
    ):
        raise QuoteMemError("evaluation_invalid",
                            "BEAM rubric scoring requires its exact pinned prompt")
    text = policy.prompt_template.replace("<rubric_item>", criterion).replace(
        "<llm_response>", prediction
    )
    return (BenchmarkMessage(role="user", content=text),)


def _failure(error: Exception) -> CallFailure:
    if isinstance(error, ProviderContextLimitError):
        return CallFailure(kind="provider_context_limit", actual_resolved_model=None)
    if isinstance(error, BenchmarkOutputLengthError):
        return CallFailure(kind="provider_output_length", actual_resolved_model=None)
    if isinstance(error, ModelResolutionMismatch):
        return CallFailure(
            kind="model_resolution_mismatch", actual_resolved_model=error.actual_resolved_model
        )
    if isinstance(error, ProviderError):
        kind = (
            "provider_timeout"
            if error.error_class == "timeout"
            else "provider_schema_invalid"
            if error.error_class == "schema_invalid"
            else "provider_error"
        )
        return CallFailure(kind=kind, actual_resolved_model=None)
    if isinstance(error, ValidationError):
        return CallFailure(kind="provider_schema_invalid", actual_resolved_model=None)
    if isinstance(error, QuoteMemError):
        kind = (
            "fixture_missing"
            if error.code == "fixture_missing"
            else "response_binding_mismatch"
            if error.code == "binding_mismatch"
            else "provider_error"
        )
        return CallFailure(kind=kind, actual_resolved_model=None)
    return CallFailure(kind="unexpected_error", actual_resolved_model=None)


def _duration(start: int, end: int) -> int | None:
    return end - start if type(start) is int and type(end) is int and end >= start else None


def generate_answers(
    corpus: BenchmarkCorpus,
    contexts: tuple[BenchmarkContext, ...],
    *,
    run_key: str,
    trial_key: str,
    policy: BenchmarkPolicy,
    replay_config: ReplayConfig,
    provider: AnswerProvider,
    clock_ns: Callable[[], int] = perf_counter_ns,
    answer_workers: int = 1,
) -> AnswerRunReport:
    """Generate external answers after validating every supplied Search association.

    Missing contexts and provider failures retain original corpus rows. Timings
    cover each provider invocation plus response validation, including provider-
    owned retries; they do not claim an HTTP-attempt count or complete usage trace.
    A caller using multiple workers must supply a concurrency-safe provider.
    """
    if type(answer_workers) is not int or answer_workers not in (1, 4, 8, 16, 28):
        raise QuoteMemError("evaluation_invalid", "Answer workers must be 1, 4, 8, 16, or 28")
    identity = _corpus_identity(corpus)
    checked_policy = BenchmarkPolicy.model_validate(
        policy.model_dump(mode="python", warnings=False)
    )
    checked_replay = ReplayConfig.model_validate(
        replay_config.model_dump(mode="python", warnings=False)
    )
    packing_policy = _packing_policy(checked_policy)
    by_id = {case.case_id: case for case in corpus.cases}
    requests: dict[str, AnswerRequest] = {}
    if not run_key or not trial_key:
        raise QuoteMemError(
            "evaluation_invalid", "Answer generation requires explicit run and trial keys"
        )
    for supplied in contexts:
        context = BenchmarkContext.model_validate(
            supplied.model_dump(mode="python", warnings=False)
        )
        if context.case_id not in by_id or context.case_id in requests:
            raise QuoteMemError(
                "binding_mismatch", "Search contexts contain an unknown or repeated case"
            )
        _context_matches(corpus, by_id[context.case_id], context, run_key, trial_key)
        requests[context.case_id] = answer_request(
            context, checked_policy, render_answer_messages(context, checked_policy)
        )
    if len({request.context.retrieval_origin for request in requests.values()}) > 1:
        raise QuoteMemError("binding_mismatch", "One answer run cannot mix retrieval arms")
    validate_replay_contexts(
        corpus,
        tuple(request.context for request in requests.values()),
        run_id=run_key,
        config=checked_replay,
    )

    def observe(case: BenchmarkCase) -> AnswerCaseObservation:
        request = requests.get(case.case_id)
        if request is None:
            return AnswerCaseObservation(
                case_id=case.case_id,
                status="missing_search",
                request=None,
                result=None,
                packing=None,
                duration_ns=None,
                failure=None,
            )
        result: AnswerResult | None = None
        failure: CallFailure | None = None
        start = clock_ns()
        try:
            result = AnswerResult.model_validate(
                provider.answer(
                    AnswerRequest.model_validate(request.model_dump(mode="python", warnings=False))
                ).model_dump(mode="python", warnings=False)
            )
            _check_result(request, result)
        except Exception as error:
            # Independent cases continue; cancellation and process termination do not.
            failure = _failure(error)
        duration = _duration(start, clock_ns())
        return AnswerCaseObservation(
            case_id=case.case_id,
            status="answered" if failure is None else "failed",
            request=request,
            result=result,
            packing=pack_search_context(request.context, packing_policy),
            duration_ns=duration,
            failure=failure,
        )

    if answer_workers == 1:
        rows = [observe(case) for case in corpus.cases]
    else:
        pool = ThreadPoolExecutor(max_workers=answer_workers)
        try:
            pending = {
                pool.submit(observe, case): index
                for index, case in enumerate(corpus.cases)
            }
            completed: dict[int, AnswerCaseObservation] = {}
            for future in as_completed(pending):
                completed[pending[future]] = future.result()
            rows = [completed[index] for index in range(len(corpus.cases))]
        except BaseException:
            pool.shutdown(cancel_futures=True)
            raise
        else:
            pool.shutdown()
    count = sum(row.status == "answered" for row in rows)
    body = {
        "corpus_sha256": identity,
        "source_selection_json": canonical_json(corpus.source.model_dump(mode="json")),
        "run_key": run_key,
        "trial_key": trial_key,
        "policy": checked_policy.model_dump(mode="json"),
        "replay_config": checked_replay.model_dump(mode="json"),
        "case_count": len(rows),
        "answered_count": count,
        "status": "complete" if count == len(rows) else "incomplete",
        "cases": [row.model_dump(mode="json") for row in rows],
    }
    return AnswerRunReport.model_validate_json(
        canonical_json(body | {"report_sha256": benchmark_digest(body)})
    )


def validate_answer_report(corpus: BenchmarkCorpus, answers: AnswerRunReport) -> AnswerRunReport:
    """Bind sealed answer observations to their complete verified source and replay.

    Args:
        corpus: Complete normalized corpus loaded from its verified source.
        answers: Report retaining one observation per original case, including failures.

    Returns:
        AnswerRunReport: Revalidated source, query, policy, packing and replay bindings.
            Consistency does not establish actual remote or HTTP execution.

    Raises:
        QuoteMemError: The report or a request differs from the complete source or replay.
        ValidationError: A serialized observation or its seal is invalid.
    """
    identity = _corpus_identity(corpus)
    checked = AnswerRunReport.model_validate(answers.model_dump(mode="python", warnings=False))
    if (
        checked.corpus_sha256 != identity
        or json.loads(checked.source_selection_json) != corpus.source.model_dump(mode="json")
        or tuple(row.case_id for row in checked.cases)
        != tuple(case.case_id for case in corpus.cases)
    ):
        raise QuoteMemError("binding_mismatch", "Answers differ from the complete selected corpus")
    for case, answer in zip(corpus.cases, checked.cases, strict=True):
        if answer.request is not None:
            _context_matches(
                corpus, case, answer.request.context, checked.run_key, checked.trial_key
            )
            if answer.request.messages != render_answer_messages(
                answer.request.context, checked.policy
            ):
                raise QuoteMemError(
                    "binding_mismatch", "Answer messages differ from their frozen template"
                )
    validate_replay_contexts(
        corpus,
        tuple(row.request.context for row in checked.cases if row.request is not None),
        run_id=checked.run_key,
        config=checked.replay_config,
    )
    return checked


def _beam_answers(corpus: BenchmarkCorpus, answers: AnswerRunReport) -> tuple[str, AnswerRunReport]:
    """Revalidate the exact complete BEAM corpus and referenced answer observations."""
    if corpus.source.dataset != "beam" or any(
        case.evaluation_kind != "beam_rubric" for case in corpus.cases
    ):
        raise QuoteMemError("binding_mismatch", "BEAM answers require the complete BEAM corpus")
    checked = validate_answer_report(corpus, answers)
    return checked.corpus_sha256, checked


def validate_beam_report(
    corpus: BenchmarkCorpus, answers: AnswerRunReport, report: RubricRunReport
) -> RubricRunReport:
    """Validate a serialized rubric report against its complete source and answers.

    Args:
        corpus: Complete normalized corpus loaded from its verified source.
        answers: The exact answer report named by the rubric report's digest.
        report: Structurally parsed observations, including every missing/failed item.

    Returns:
        A revalidated report whose cases, criteria, predictions, and dispositions
        match both references. This checks consistency, not remote execution.

    Raises:
        QuoteMemError: A source, answer, request, or denominator binding differs.
        ValidationError: A supplied serialized model is structurally invalid.
    """
    identity, checked_answers = _beam_answers(corpus, answers)
    checked = RubricRunReport.model_validate(report.model_dump(mode="python", warnings=False))
    render_beam_messages(checked.policy, criterion="", prediction="")
    if (
        checked.corpus_sha256 != identity
        or checked.answer_report_sha256 != checked_answers.report_sha256
        or checked.run_key != checked_answers.run_key
        or checked.trial_key != checked_answers.trial_key
        or tuple(row.case_id for row in checked.cases)
        != tuple(case.case_id for case in corpus.cases)
    ):
        raise QuoteMemError("binding_mismatch", "Rubric report differs from its source or answers")
    for case, answer, row in zip(corpus.cases, checked_answers.cases, checked.cases, strict=True):
        if (
            row.category != case.category
            or tuple(item.criterion for item in row.items) != case.evaluation_rubric
        ):
            raise QuoteMemError("binding_mismatch", "Rubric case differs from its source criteria")
        invalid = not case.evaluation_rubric or any(
            not criterion.strip() for criterion in case.evaluation_rubric
        )
        reason = (
            "invalid_rubric"
            if invalid
            else "missing_answer"
            if answer.status != "answered"
            else "failed_items"
            if row.status != "complete"
            else None
        )
        if row.incomplete_reason != reason:
            raise QuoteMemError("binding_mismatch",
                                "Rubric disposition differs from its source answer")
        for item in row.items:
            if invalid or answer.status != "answered":
                expected_status = "invalid_rubric" if invalid else "missing_answer"
                if item.status != expected_status:
                    raise QuoteMemError("binding_mismatch",
                                        "Unexecuted rubric item differs from source")
                continue
            assert answer.request is not None and answer.result is not None
            prediction = answer.result.output.answer
            expected = rubric_request(
                answer.request.context,
                checked.policy,
                render_beam_messages(
                    checked.policy, criterion=item.criterion, prediction=prediction
                ),
                prediction=prediction,
                criteria=case.evaluation_rubric,
                criterion_index=item.criterion_index,
            )
            if item.status not in ("complete", "failed") or item.request != expected:
                raise QuoteMemError("binding_mismatch",
                                    "Rubric request differs from its exact answer")
    return checked


def score_beam_answers(
    corpus: BenchmarkCorpus,
    answers: AnswerRunReport,
    *,
    policy: BenchmarkPolicy,
    judge: RubricJudge,
    clock_ns: Callable[[], int] = perf_counter_ns,
) -> RubricRunReport:
    """Judge every BEAM rubric ordinal and macro-average only complete case results.

    Failed or missing items never reduce denominators. Independent item failures
    do not prevent later items from running. The result is reference-validated
    before return; use validate_beam_report again when loading it independently.
    Event-ordering alignment, tau, F1, and combined scores remain unsupported.
    """
    identity, checked = _beam_answers(corpus, answers)
    checked_policy = BenchmarkPolicy.model_validate(
        policy.model_dump(mode="python", warnings=False)
    )
    render_beam_messages(checked_policy, criterion="", prediction="")
    rows: list[RubricCaseObservation] = []
    for case, answer in zip(corpus.cases, checked.cases, strict=True):
        items: list[RubricItemObservation] = []
        invalid = not case.evaluation_rubric or any(
            not criterion.strip() for criterion in case.evaluation_rubric
        )
        for index, criterion in enumerate(case.evaluation_rubric):
            if invalid or answer.status != "answered":
                items.append(
                    RubricItemObservation(
                        criterion_index=index,
                        criterion=criterion,
                        status="invalid_rubric" if invalid else "missing_answer",
                        request=None,
                        result=None,
                        duration_ns=None,
                        failure=None,
                    )
                )
                continue
            assert answer.request is not None and answer.result is not None
            prediction = answer.result.output.answer
            request = rubric_request(
                answer.request.context,
                checked_policy,
                render_beam_messages(checked_policy, criterion=criterion, prediction=prediction),
                prediction=prediction,
                criteria=case.evaluation_rubric,
                criterion_index=index,
            )
            result: RubricResult | None = None
            failure: CallFailure | None = None
            start = clock_ns()
            try:
                result = RubricResult.model_validate(
                    judge.judge(
                        RubricRequest.model_validate(
                            request.model_dump(mode="python", warnings=False)
                        )
                    ).model_dump(mode="python", warnings=False)
                )
                _check_result(request, result)
            except Exception as error:
                failure = _failure(error)
            duration = _duration(start, clock_ns())
            items.append(
                RubricItemObservation(
                    criterion_index=index,
                    criterion=criterion,
                    status="complete" if failure is None else "failed",
                    request=request,
                    result=result,
                    duration_ns=duration,
                    failure=failure,
                )
            )
        completed = sum(item.status == "complete" for item in items)
        complete = bool(items) and completed == len(items)
        mean = (
            sum(item.result.output.score for item in items if item.result is not None) / len(items)
            if complete
            else None
        )
        reason = (
            None
            if complete
            else "invalid_rubric"
            if invalid
            else "missing_answer"
            if answer.status != "answered"
            else "failed_items"
        )
        rows.append(
            RubricCaseObservation(
                case_id=case.case_id,
                category=case.category,
                status="complete" if complete else "incomplete",
                incomplete_reason=reason,
                criterion_count=len(items),
                complete_count=completed,
                items=tuple(items),
                mean_score=mean,
                event_ordering_status="unsupported"
                if case.category == "event_ordering"
                else "not_applicable",
            )
        )
    complete = all(row.status == "complete" for row in rows)
    mean = (
        sum(row.mean_score for row in rows if row.mean_score is not None) / len(rows)
        if complete
        else None
    )
    report = RubricRunReport(
        corpus_sha256=identity,
        answer_report_sha256=checked.report_sha256,
        run_key=checked.run_key,
        trial_key=checked.trial_key,
        policy=checked_policy,
        metric="beam_rubric_macro_case_mean_v1_strict_local_parser",
        case_count=len(rows),
        complete_case_count=sum(row.status == "complete" for row in rows),
        criterion_count=sum(row.criterion_count for row in rows),
        complete_criterion_count=sum(row.complete_count for row in rows),
        status="complete" if complete else "incomplete",
        mean_score=mean,
        event_ordering_metric="unsupported",
        cases=tuple(rows),
    )
    return validate_beam_report(corpus, checked, report)
