"""Descriptive paired answer diagnostics over registered AML and local baseline arms.

Registration and evidence digests bind supplied artifacts; they do not certify
execution, preregistration time, reviewed truth, cost attribution or independence.
This consumer never makes a statistical superiority or production-policy decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from aml.bench.answering import (
    AnswerPackingPolicy,
    AnswerRunReport,
    validate_answer_report,
)
from aml.bench.datasets import BenchmarkCorpus, DatasetSource, source_identity, source_identity_kind
from aml.bench.evaluation import AnswerScore, EvaluationReport, Prediction, score_predictions
from aml.bench.mem0_binding import Mem0Registration, Mem0RetrievalBinding
from aml.bench.providers import BenchmarkPolicy, Digest, Nonempty, benchmark_digest
from aml.bench.replay import ReplayConfig
from aml.core.errors import AMLError


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, hide_input_in_errors=True)


class ComparisonArmRegistration(_Frozen):
    """Preselected arm coordinates and code provenance, without outcome fields."""

    retrieval_origin: Literal["aml_http", "verbatim_bm25_v1", "mem0_oss_v2_1_0"]
    run_key: Nonempty
    trial_key: Nonempty
    source_commit: Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
    execution_registration_sha256: Digest
    provider_mode: Literal["remote_live_model", "frozen_fixture"]


class AnswerComparisonRegistration(_Frozen):
    """One fixed ordered denominator and shared answer/replay conditions.

    Only descriptive analysis is supported. A separate approved preregistration
    and dependence-aware analysis are required before any benefit decision.
    """

    schema_version: Literal[1]
    source: DatasetSource
    case_ids: Annotated[tuple[Nonempty, ...], Field(min_length=1)]
    answer_policy: BenchmarkPolicy
    replay_config: ReplayConfig
    arrival: Literal["all_adds_then_search", "immediate_first_search"]
    aml: ComparisonArmRegistration
    baseline: ComparisonArmRegistration
    analysis: Literal["descriptive_only"]

    @model_validator(mode="after")
    def compatible_arms(self) -> Self:
        """Reject mixed provider modes, arm roles, budgets and ambiguous denominators."""
        if (
            self.aml.retrieval_origin != "aml_http"
            or self.baseline.retrieval_origin not in {"verbatim_bm25_v1", "mem0_oss_v2_1_0"}
            or self.aml.provider_mode != self.baseline.provider_mode
            or self.answer_policy.schema_id != "bench-answer-v1"
            or self.answer_policy.expected_resolved_model is None
            or self.answer_policy.context_packing_json is None
            or len(set(self.case_ids)) != len(self.case_ids)
            or len(self.case_ids) != self.source.expected_cases
            or self.source.conversation_selection is not None
            and tuple(self.source.conversation_selection.case_ids) != self.case_ids
        ):
            raise ValueError("Comparison arms or denominator differ from the registered conditions")
        AnswerPackingPolicy.model_validate_json(self.answer_policy.context_packing_json)
        return self


class AnswerComparisonArm(_Frozen):
    """Answer artifact with mandatory retained evidence and explicit completion claims.

    AML chain completion means Adds, publication/release, Search and shutdown.
    Baseline completion means local retrieval; its AML HTTP chain is inapplicable.
    Evidence custody must be checked separately; these declarations are not proof.
    """

    registration: ComparisonArmRegistration
    execution_evidence_sha256: Digest
    completion: Literal["complete", "incomplete", "unknown"]
    http_chain: Literal["complete", "incomplete", "unknown", "not_applicable"]
    answers: AnswerRunReport

    @model_validator(mode="after")
    def completion_contract(self) -> Self:
        """Source-only results cannot claim AML HTTP delivery or publication."""
        if self.registration.retrieval_origin != "aml_http":
            if self.http_chain != "not_applicable":
                raise ValueError("Local baseline retrieval has no AML HTTP chain")
        elif self.http_chain == "not_applicable" or (
            self.completion == "complete" and self.http_chain != "complete"
        ):
            raise ValueError("Complete AML requires its explicit complete HTTP chain")
        return self


@dataclass(frozen=True)
class PairedAnswerScore:
    """Original case metadata and AML-minus-baseline local metric differences."""

    case_id: str
    conversation_id: str
    category: str
    aml: AnswerScore
    baseline: AnswerScore
    f1_delta: float
    exact_match_delta: int


@dataclass(frozen=True)
class PairedAnswerSummary:
    """Fixed-denominator aggregate; paired presence never changes the denominator."""

    case_count: int
    both_answered_count: int
    aml_answered_count: int
    baseline_answered_count: int
    aml_mean_f1: float
    baseline_mean_f1: float
    mean_f1_delta: float
    aml_exact_match_rate: float
    baseline_exact_match_rate: float
    exact_match_rate_delta: float


@dataclass(frozen=True)
class AnswerComparisonReport:
    """Descriptive effects with retained arm identities and explicitly unknown quality."""

    registration_sha256: str
    dataset_identity: str
    dataset_identity_kind: str
    metric: str
    aml_report_sha256: str
    baseline_report_sha256: str
    aml_execution_evidence_sha256: str
    baseline_execution_evidence_sha256: str
    experiment_status: str
    quality_status: str
    decision: str
    decision_eligibility: str
    limitations: tuple[str, ...]
    summary: PairedAnswerSummary
    categories: dict[str, PairedAnswerSummary]
    conversations: dict[str, PairedAnswerSummary]
    cases: tuple[PairedAnswerScore, ...]


def validate_comparison_arm(
    corpus: BenchmarkCorpus,
    registration: AnswerComparisonRegistration,
    expected: ComparisonArmRegistration,
    arm: AnswerComparisonArm,
) -> AnswerRunReport:
    """Verify an arm's actual answer report and registered conditions before scoring.

    This shared validator intentionally supplies no source or checkpoint truth.
    The selected scorer owns its separately bound expected-outcome contract.
    """
    answers = validate_answer_report(corpus, arm.answers)
    for row in answers.cases:
        if row.request is not None and row.request.context.retrieval_origin == "mem0_oss_v2_1_0":
            binding = Mem0RetrievalBinding.model_validate_json(
                row.request.context.retrieval_binding_json,
            )
            mem0 = Mem0Registration.model_validate_json(binding.registration_json)
            if (
                binding.registration_sha256 != expected.execution_registration_sha256
                or mem0.arrival != registration.arrival
            ):
                raise AMLError(
                    "binding_mismatch", "Mem0 arm differs from its registered method or arrival",
                )
    if (
        arm.registration != expected
        or answers.run_key != expected.run_key
        or answers.trial_key != expected.trial_key
        or answers.policy != registration.answer_policy
        or answers.replay_config != registration.replay_config
        or any(
            row.request is not None
            and row.request.context.retrieval_origin != expected.retrieval_origin
            for row in answers.cases
        )
    ):
        raise AMLError("binding_mismatch", "Answer arm differs from its comparison registration")
    return answers


def _score_arm(
    corpus: BenchmarkCorpus, registration: AnswerComparisonRegistration,
    expected: ComparisonArmRegistration, arm: AnswerComparisonArm,
) -> EvaluationReport:
    answers = validate_comparison_arm(corpus, registration, expected, arm)
    return score_predictions(corpus, [
        Prediction(case_id=row.case_id, answer=row.result.output.answer)
        for row in answers.cases
        if row.status == "answered" and row.result is not None
    ])


def _summary(rows: list[PairedAnswerScore]) -> PairedAnswerSummary:
    count = len(rows)
    return PairedAnswerSummary(
        case_count=count,
        both_answered_count=sum(row.aml.prediction_present and row.baseline.prediction_present
                                for row in rows),
        aml_answered_count=sum(row.aml.prediction_present for row in rows),
        baseline_answered_count=sum(row.baseline.prediction_present for row in rows),
        aml_mean_f1=sum(row.aml.f1 for row in rows) / count,
        baseline_mean_f1=sum(row.baseline.f1 for row in rows) / count,
        mean_f1_delta=sum(row.f1_delta for row in rows) / count,
        aml_exact_match_rate=sum(row.aml.exact_match for row in rows) / count,
        baseline_exact_match_rate=sum(row.baseline.exact_match for row in rows) / count,
        exact_match_rate_delta=sum(row.exact_match_delta for row in rows) / count,
    )


def compare_answers(
    corpus: BenchmarkCorpus,
    registration: AnswerComparisonRegistration,
    aml: AnswerComparisonArm,
    baseline: AnswerComparisonArm,
) -> AnswerComparisonReport:
    """Compare compatible sealed arms without dropping absent or failed predictions.

    Args:
        corpus: Complete verified LoCoMo or LongMemEval source, including references.
        registration: Frozen ordered cohort, arm provenance and shared answer conditions.
        aml: AML observations and explicit retained chain-evidence identity.
        baseline: Registered BM25 or Mem0 observations and local execution-evidence identity.

    Returns:
        AnswerComparisonReport: AML-minus-baseline paired lexical F1/exact effects, by
            category and conversation, with an always-inconclusive benefit decision.

    Raises:
        AMLError: Source, case order, policy, replay configuration or arm bindings differ.
        ValidationError: A registration, provenance field or sealed report is malformed.
    """
    registration = AnswerComparisonRegistration.model_validate(
        registration.model_dump(mode="python", warnings=False)
    )
    aml = AnswerComparisonArm.model_validate(aml.model_dump(mode="python", warnings=False))
    baseline = AnswerComparisonArm.model_validate(
        baseline.model_dump(mode="python", warnings=False)
    )
    if (
        corpus.source != registration.source
        or tuple(case.case_id for case in corpus.cases) != registration.case_ids
    ):
        raise AMLError("binding_mismatch", "Comparison differs from its complete registered source")
    aml_scores = _score_arm(corpus, registration, registration.aml, aml)
    baseline_scores = _score_arm(corpus, registration, registration.baseline, baseline)
    rows = [
        PairedAnswerScore(
            case.case_id, case.cluster_id, case.category, left, right,
            left.f1 - right.f1, int(left.exact_match) - int(right.exact_match),
        )
        for case, left, right in zip(
            corpus.cases, aml_scores.cases, baseline_scores.cases, strict=True,
        )
    ]
    categories: dict[str, list[PairedAnswerScore]] = {}
    conversations: dict[str, list[PairedAnswerScore]] = {}
    for row in rows:
        categories.setdefault(row.category, []).append(row)
        conversations.setdefault(row.conversation_id, []).append(row)
    limitations = [
        "descriptive_only_no_preregistered_benefit_criterion_or_paired_uncertainty",
        "local_lexical_metrics_not_official_judge",
        "independent_source_truth_and_stage_error_labels_unknown",
        "held_out_status_and_sequential_prefix_quality_unknown",
        "cost_attribution_not_joined",
        "execution_evidence_and_preregistration_time_not_verified_by_this_consumer",
        (
            "source_only_baseline_has_no_http_publication_or_release"
            if registration.baseline.retrieval_origin == "verbatim_bm25_v1"
            else "mem0_baseline_has_no_aml_http_publication_or_release"
        ),
    ]
    if len(conversations) == 1:
        limitations.append("single_conversation_cannot_establish_independent_uncertainty")
    else:
        limitations.append("conversation_groups_do_not_establish_statistical_independence")
    complete = all(
        arm.completion == "complete" and arm.answers.status == "complete"
        for arm in (aml, baseline)
    )
    if not complete:
        limitations.append("incomplete_or_unknown_chain_or_answers")
    if registration.aml.provider_mode == "frozen_fixture":
        limitations.append("frozen_fixture_answers_are_not_real_model_quality")
    if registration.baseline.retrieval_origin == "mem0_oss_v2_1_0":
        limitations.append("mem0_extraction_embedding_and_metadata_time_semantics_differ_from_aml")
    return AnswerComparisonReport(
        benchmark_digest(registration.model_dump(mode="json")),
        source_identity(corpus.source), source_identity_kind(corpus.source), aml_scores.metric,
        aml.answers.report_sha256, baseline.answers.report_sha256,
        aml.execution_evidence_sha256, baseline.execution_evidence_sha256,
        "declared_complete" if complete else "incomplete_or_unknown",
        "unknown", "inconclusive", "ineligible", tuple(limitations), _summary(rows),
        {key: _summary(group) for key, group in categories.items()},
        {key: _summary(group) for key, group in conversations.items()}, tuple(rows),
    )
