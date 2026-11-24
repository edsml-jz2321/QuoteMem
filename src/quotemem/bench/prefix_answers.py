"""Reviewed checkpoint outcomes and descriptive comparisons with explicit unknowns.

No source answers are promoted to checkpoint labels. Review witnesses are retained
claims; their custody, independence and pre-execution freeze require external proof.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Annotated, Literal, Self, cast

from pydantic import Field, ValidationInfo, field_validator, model_validator

from quotemem.bench.answer_comparison import (
    AnswerComparisonArm,
    AnswerComparisonRegistration,
    validate_comparison_arm,
)
from quotemem.bench.datasets import (
    BenchmarkCorpus,
    DatasetSource,
    Digest,
    Nonempty,
    source_identity,
)
from quotemem.bench.evaluation import Prediction, score_answer_alternatives
from quotemem.bench.mechanism_quality import ReviewWitness, SourceReference
from quotemem.bench.providers import benchmark_digest
from quotemem.bench.quality_evaluation import FrozenTuple, ReviewRegistration
from quotemem.bench.received_prefix import validate_received_prefix
from quotemem.core.errors import QuoteMemError
from quotemem.core.identity import canonical_json
from quotemem.store.metadata import StrictModel


class CheckpointOutcome(StrictModel):
    """One source-only judgment; unanswerable alternatives explicitly describe abstention."""

    case_id: Nonempty
    query_sha256: Digest
    state: Literal["answerable", "unanswerable", "unknown"]
    reference_answers: FrozenTuple[Nonempty]
    sources: FrozenTuple[SourceReference]
    review: ReviewWitness | None

    @model_validator(mode="after")
    def labels_are_explicit(self) -> Self:
        if self.state == "unknown":
            if self.reference_answers:
                raise ValueError("Unknown checkpoint truth cannot carry scoring alternatives")
        elif not self.reference_answers or self.review is None or (
            self.state == "answerable" and not self.sources
        ):
            raise ValueError("Known checkpoint truth requires reviewed explicit alternatives")
        return self


class CheckpointExpectedOutcomes(StrictModel):
    """A complete independently reviewed prefix denominator, including undecided cases."""

    review: ReviewRegistration
    independence_witness_sha256: Digest
    exposure_witness_sha256: Digest
    held_out_status: Literal["declared_held_out", "exposed", "unknown"]
    outcomes: Annotated[FrozenTuple[CheckpointOutcome], Field(min_length=1)]


class PrefixAnswerComparisonRegistration(StrictModel):
    """Opt-in paired report branch; the nested legacy registration retains all arm conditions."""

    kind: Literal["received_prefix_answer_comparison_v1"]
    comparison: AnswerComparisonRegistration
    expected_outcomes: CheckpointExpectedOutcomes

    @field_validator("comparison", mode="before")
    @classmethod
    def comparison_json(cls, value: object, info: ValidationInfo) -> object:
        """Preserve the existing strict registration's native JSON tuple boundary."""
        if info.mode == "json" and isinstance(value, dict):
            return AnswerComparisonRegistration.model_validate_json(
                canonical_json(cast(dict[str, object], value)),
            )
        return value


def validate_checkpoint_outcomes(
    corpus: BenchmarkCorpus, expected: CheckpointExpectedOutcomes,
) -> CheckpointExpectedOutcomes:
    """Bind all reviewed outcomes and every support coordinate to only the received source."""
    if not isinstance(corpus.source, DatasetSource) or corpus.source.received_prefix is None:
        raise QuoteMemError("binding_mismatch", "Checkpoint outcomes require a received prefix")
    validate_received_prefix(corpus.source, corpus.cases)
    expected = CheckpointExpectedOutcomes.model_validate(expected.model_dump(mode="python"))
    if (
        expected.review.population_sha256 != source_identity(corpus.source)
        or [row.case_id for row in expected.outcomes] != corpus.source.received_prefix.case_ids
        or [case.case_id for case in corpus.cases] != corpus.source.received_prefix.case_ids
    ):
        raise QuoteMemError("binding_mismatch", "Checkpoint outcomes differ from the prefix census")
    reviewers = [row.reviewer_id for row in expected.review.reviewers]
    if len(reviewers) != len(set(reviewers)):
        raise QuoteMemError("binding_mismatch", "Checkpoint review repeats a reviewer")
    for case, outcome in zip(corpus.cases, expected.outcomes, strict=True):
        if outcome.query_sha256 != benchmark_digest({
            "case_id": case.case_id, "query": case.query,
            "query_source_date": case.query_source_date,
        }):
            raise QuoteMemError("binding_mismatch", "Checkpoint review changed its source question")
        known = {
            (session.session_id, message.message_id): hashlib.sha256(
                message.content.encode("utf-8"),
            ).hexdigest() for session in case.sessions for message in session.messages
        }
        references = [(row.session_id, row.message_id) for row in outcome.sources]
        if len(references) != len(set(references)) or any(
            known.get((row.session_id, row.message_id)) != row.content_sha256
            for row in outcome.sources
        ):
            raise QuoteMemError("binding_mismatch",
                                "Checkpoint review cites changed or future source")
        review = outcome.review
        if review is not None and (
            review.method != "source_only"
            or len(review.reviewer_ids) != len(set(review.reviewer_ids))
            or not set(review.reviewer_ids) <= set(reviewers)
            or review.reviewed_at_ns > expected.review.frozen_at_ns
        ):
            raise QuoteMemError("binding_mismatch",
                                "Checkpoint review used an incompatible boundary")
    return expected


@dataclass(frozen=True)
class CheckpointAnswerScore:
    """Unknown labels remain null; known-label missing predictions are measured zeroes."""

    case_id: str
    state: Literal["answerable", "unanswerable", "unknown"]
    f1: float | None
    exact_match: bool | None
    prediction_present: bool


@dataclass(frozen=True)
class CheckpointEvaluationReport:
    """Complete denominator and bounds, never an unknown-as-zero point estimate."""

    source_identity: str
    expected_outcomes_sha256: str
    metric: str
    case_count: int
    prediction_count: int
    known_outcome_count: int
    unknown_outcome_count: int
    unanswerable_count: int
    f1_numerator_known: float
    exact_match_numerator_known: int
    mean_f1: float | None
    exact_match_rate: float | None
    full_denominator_f1_bounds: tuple[float, float]
    full_denominator_exact_bounds: tuple[float, float]
    cases: tuple[CheckpointAnswerScore, ...]


def score_checkpoint_predictions(
    corpus: BenchmarkCorpus, predictions: list[Prediction], expected: CheckpointExpectedOutcomes,
) -> CheckpointEvaluationReport:
    """Score only reviewed checkpoint alternatives, retaining the original full denominator."""
    expected = validate_checkpoint_outcomes(corpus, expected)
    predictions = [Prediction.model_validate(row.model_dump(mode="python")) for row in predictions]
    by_id = {row.case_id: row for row in predictions}
    if len(by_id) != len(predictions) or not set(by_id) <= {case.case_id for case in corpus.cases}:
        raise QuoteMemError("binding_mismatch",
                            "Checkpoint predictions changed the case population")
    rows: list[CheckpointAnswerScore] = []
    for outcome in expected.outcomes:
        prediction = by_id.get(outcome.case_id)
        f1, exact = (None, None) if outcome.state == "unknown" else score_answer_alternatives(
            prediction.answer if prediction is not None else None, outcome.reference_answers,
        )
        rows.append(CheckpointAnswerScore(
            outcome.case_id, outcome.state, f1, exact, prediction is not None,
        ))
    count = len(rows)
    unknown = sum(row.state == "unknown" for row in rows)
    f1_sum = sum(row.f1 for row in rows if row.f1 is not None)
    exact_sum = sum(row.exact_match for row in rows if row.exact_match is not None)
    return CheckpointEvaluationReport(
        source_identity(corpus.source), benchmark_digest(expected.model_dump(mode="json")),
        "reviewed_checkpoint_token_f1_v1_not_official_judge", count, len(predictions),
        count - unknown, unknown, sum(row.state == "unanswerable" for row in rows),
        f1_sum, exact_sum, f1_sum / count if not unknown else None,
        exact_sum / count if not unknown else None,
        (f1_sum / count, (f1_sum + unknown) / count),
        (exact_sum / count, (exact_sum + unknown) / count), tuple(rows),
    )


@dataclass(frozen=True)
class PrefixAnswerComparisonReport:
    """Descriptive cumulative-prefix comparison; no causal or superiority inference."""

    kind: str
    registration_sha256: str
    aml_answer_report_sha256: str
    baseline_answer_report_sha256: str
    aml_execution_evidence_sha256: str
    baseline_execution_evidence_sha256: str
    experiment_status: str
    decision: str
    aml: CheckpointEvaluationReport
    baseline: CheckpointEvaluationReport
    paired_f1_delta_bounds: tuple[float, float]
    category_f1_delta_bounds: dict[str, tuple[float, float]]
    conversation_f1_delta_bounds: dict[str, tuple[float, float]]
    limitations: tuple[str, ...]


def compare_prefix_answers(
    corpus: BenchmarkCorpus, registration: PrefixAnswerComparisonRegistration,
    aml: AnswerComparisonArm, baseline: AnswerComparisonArm,
) -> PrefixAnswerComparisonReport:
    """Reuse exact answer/arm validation with separately reviewed prefix outcomes."""
    registration = PrefixAnswerComparisonRegistration.model_validate(
        registration.model_dump(mode="python"),
    )
    comparison = registration.comparison
    expected = validate_checkpoint_outcomes(corpus, registration.expected_outcomes)
    if corpus.source != comparison.source or tuple(
        case.case_id for case in corpus.cases
    ) != comparison.case_ids:
        raise QuoteMemError("binding_mismatch",
                            "Checkpoint comparison changed its registered source")
    scores: list[CheckpointEvaluationReport] = []
    for arm, arm_registration in ((aml, comparison.aml), (baseline, comparison.baseline)):
        arm = AnswerComparisonArm.model_validate(arm.model_dump(mode="python"))
        answers = validate_comparison_arm(corpus, comparison, arm_registration, arm)
        scores.append(score_checkpoint_predictions(corpus, [
            Prediction(case_id=row.case_id, answer=row.result.output.answer)
            for row in answers.cases if row.status == "answered" and row.result is not None
        ], expected))
    left, right = scores
    known_delta = left.f1_numerator_known - right.f1_numerator_known
    unknown = left.unknown_outcome_count
    categories: dict[str, list[tuple[CheckpointAnswerScore, CheckpointAnswerScore]]] = {}
    conversations: dict[str, list[tuple[CheckpointAnswerScore, CheckpointAnswerScore]]] = {}
    for case, a, b in zip(corpus.cases, left.cases, right.cases, strict=True):
        categories.setdefault(case.category, []).append((a, b))
        conversations.setdefault(case.cluster_id, []).append((a, b))

    def bounds(
        rows: list[tuple[CheckpointAnswerScore, CheckpointAnswerScore]],
    ) -> tuple[float, float]:
        delta = sum(a.f1 - b.f1 for a, b in rows if a.f1 is not None and b.f1 is not None)
        unknown = sum(a.f1 is None for a, _ in rows)
        return (delta - unknown) / len(rows), (delta + unknown) / len(rows)

    return PrefixAnswerComparisonReport(
        registration.kind, benchmark_digest(registration.model_dump(mode="json")),
        aml.answers.report_sha256, baseline.answers.report_sha256,
        aml.execution_evidence_sha256, baseline.execution_evidence_sha256,
        "declared_complete" if all(
            arm.completion == "complete" and arm.answers.status == "complete"
            for arm in (aml, baseline)
        ) else "incomplete_or_unknown",
        "inconclusive", left, right,
        ((known_delta - unknown) / left.case_count, (known_delta + unknown) / left.case_count),
        {key: bounds(rows) for key, rows in categories.items()},
        {key: bounds(rows) for key, rows in conversations.items()},
        (
            "fresh_cumulative_prefix_replay_not_uninterrupted_runtime_transitions",
            "source_only_review_and_preexecution_freeze_need_external_custody_verification",
            "unanswerable_scored_only_against_explicit_reviewed_abstention_alternatives",
            "unknown_outcomes_remain_in_full_denominator_bounds",
            "local_lexical_metric_not_semantic_judge_or_superiority_test",
            "conversation_dependence_and_costs_require_separate_analysis",
        ),
    )
