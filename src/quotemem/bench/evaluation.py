"""Local answer metrics with complete alternative answers and fixed denominators.

These lexical diagnostics do not reproduce either upstream LLM judge and are
never evidence for the real-model two-run quality target by themselves.
"""

from __future__ import annotations

import re
import string
import unicodedata
from collections import Counter
from dataclasses import dataclass

from pydantic import ValidationError

from quotemem.bench.datasets import (
    BenchmarkCorpus,
    DatasetSource,
    Nonempty,
    source_identity,
    source_identity_kind,
    source_matches_cases,
)
from quotemem.bench.memoryagentbench import SELECTED_SOURCES
from quotemem.core.errors import QuoteMemError
from quotemem.store.metadata import StrictModel


class Prediction(StrictModel):
    """An externally generated answer; generation is never performed by Search."""

    case_id: Nonempty
    answer: str


@dataclass(frozen=True)
class AnswerScore:
    """One case's local lexical F1 and exact match, retaining zero-output failures."""

    case_id: str
    f1: float
    exact_match: bool
    prediction_present: bool


@dataclass(frozen=True)
class EvaluationReport:
    """A complete-denominator local diagnostic, separate from official judge scores."""

    dataset_sha256: str
    metric: str
    case_count: int
    prediction_count: int
    mean_f1: float
    exact_match_rate: float
    cases: tuple[AnswerScore, ...]
    dataset_identity_kind: str = "raw_file_sha256"
    dataset_variant: str | None = None


@dataclass(frozen=True)
class SubstringScore:
    """One MAB case's publisher substring metric with explicit missing prediction state."""

    case_id: str
    substring_exact_match: bool
    prediction_present: bool


@dataclass(frozen=True)
class SubstringReport:
    """MAB Conflict Resolution subset accuracy, without a claim about full MAB."""

    dataset_sha256: str
    dataset_identity_kind: str
    dataset_variant: str
    metric: str
    case_count: int
    prediction_count: int
    accuracy: float
    empty_normalized_reference_cases: tuple[str, ...]
    cases: tuple[SubstringScore, ...]


def _tokens(value: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return re.findall(r"\w+", normalized)


def _f1(prediction: list[str], reference: list[str]) -> float:
    overlap = sum((Counter(prediction) & Counter(reference)).values())
    return 2 * overlap / (len(prediction) + len(reference)) if overlap else 0.0


def score_answer_alternatives(
    answer: str | None, references: tuple[str, ...],
) -> tuple[float, bool]:
    """Apply the existing lexical metric to explicitly supplied complete alternatives.

    This pure metric provides no authority for the reference labels. Callers must
    bind either the original complete source or independently reviewed checkpoint
    outcomes before supplying them. A missing prediction remains zero.
    """
    if not references:
        raise QuoteMemError("evaluation_invalid", "Lexical scoring requires explicit alternatives")
    tokens = _tokens(answer) if answer is not None else []
    alternatives = [_tokens(reference) for reference in references]
    return max(_f1(tokens, reference) for reference in alternatives), bool(tokens) and (
        tokens in alternatives
    )


def _complete_source(corpus: BenchmarkCorpus, *, mab: bool) -> DatasetSource:
    if not isinstance(corpus.source, DatasetSource):
        raise QuoteMemError("evaluation_invalid",
                            "This metric requires a declared single-file corpus")
    if corpus.source.received_prefix is not None:
        raise QuoteMemError("evaluation_invalid",
                            "Prefix scoring requires reviewed checkpoint outcomes")
    try:
        source = DatasetSource.model_validate(corpus.source.model_dump(mode="python"))
    except ValidationError:
        raise QuoteMemError("evaluation_invalid",
                            "The corpus source descriptor is invalid") from None
    variants = (
        {"memoryagentbench": {"factconsolidation_6k_sh_mh"}}
        if mab
        else {"locomo_refined": {"refined"}, "longmemeval": {"oracle", "s_cleaned", "m_cleaned"}}
    )
    if (
        source.variant not in variants.get(source.dataset, set())
        or not source_matches_cases(source, [case.case_id for case in corpus.cases])
        or len({case.case_id for case in corpus.cases}) != len(corpus.cases)
        or any(not case.case_id or not case.query for case in corpus.cases)
        or mab
        and {case.category for case in corpus.cases} != set(SELECTED_SOURCES)
    ):
        raise QuoteMemError("evaluation_invalid",
                            "Scoring requires the complete declared source cohort")
    return source


def score_predictions(corpus: BenchmarkCorpus, predictions: list[Prediction]) -> EvaluationReport:
    """Score external answers against complete upstream alternatives with a fixed denominator.

    Args:
        corpus (BenchmarkCorpus): Verified dataset whose complete case set is evaluated.
        predictions (list[Prediction]): At most one answer per dataset query. Missing
            predictions remain zero-score cases; unknown or repeated IDs are errors.

    Returns:
        EvaluationReport: Macro lexical F1 and normalized exact-match diagnostics.
            Alternative answers are independent complete answers, never merged as a set.

    Raises:
        QuoteMemError: If the source cohort is incomplete or inconsistent, predictions
            repeat IDs, or IDs are unknown.
    """
    source = _complete_source(corpus, mab=False)
    by_id = {prediction.case_id: prediction for prediction in predictions}
    if (
        not corpus.cases
        or any(case.evaluation_kind != "local_token_f1" for case in corpus.cases)
        or len(by_id) != len(predictions)
        or not set(by_id) <= {case.case_id for case in corpus.cases}
    ):
        raise QuoteMemError("evaluation_invalid",
                            "Prediction identities differ from the frozen corpus")
    scores: list[AnswerScore] = []
    for case in corpus.cases:
        predicted = by_id.get(case.case_id)
        f1, exact = score_answer_alternatives(
            predicted.answer if predicted is not None else None, case.answers,
        )
        scores.append(
            AnswerScore(
                case.case_id,
                f1,
                exact,
                predicted is not None,
            )
        )
    count = len(scores)
    return EvaluationReport(
        source_identity(source),
        "local_answer_token_f1_v1_not_official_judge",
        count,
        len(predictions),
        sum(score.f1 for score in scores) / count,
        sum(score.exact_match for score in scores) / count,
        tuple(scores),
        source_identity_kind(source),
        source.variant,
    )


def _mab_normalize(value: str) -> str:
    value = "".join(character for character in value.lower() if character not in string.punctuation)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", value).split())


def score_mab_predictions(
    corpus: BenchmarkCorpus, predictions: list[Prediction]
) -> SubstringReport:
    """Apply the publisher's exact normalized substring rule to a fixed local denominator.

    Args:
        corpus (BenchmarkCorpus): Complete declared two-context Conflict Resolution subset.
        predictions (list[Prediction]): Unique external answers; missing cases receive zero.

    Returns:
        SubstringReport: Macro accuracy and explicit source-label normalization anomalies.
            This does not establish real-model quality or evaluate other MAB competencies.

    Raises:
        QuoteMemError: For an incomplete/wrong corpus, duplicate predictions or unknown case IDs.
    """
    source = _complete_source(corpus, mab=True)
    by_id = {prediction.case_id: prediction for prediction in predictions}
    if (
        not corpus.cases
        or any(case.evaluation_kind != "mab_substring_exact_match" for case in corpus.cases)
        or len(by_id) != len(predictions)
        or not set(by_id) <= {case.case_id for case in corpus.cases}
    ):
        raise QuoteMemError("evaluation_invalid", "MAB predictions differ from the selected corpus")
    scores: list[SubstringScore] = []
    anomalies: list[str] = []
    for case in corpus.cases:
        references = tuple(_mab_normalize(answer) for answer in case.answers)
        if not references:
            raise QuoteMemError("evaluation_invalid",
                                "MAB source case lacks reference alternatives")
        if "" in references:
            anomalies.append(case.case_id)
        prediction = by_id.get(case.case_id)
        matched = prediction is not None and any(
            answer in _mab_normalize(prediction.answer) for answer in references
        )
        scores.append(SubstringScore(case.case_id, matched, prediction is not None))
    return SubstringReport(
        source_identity(source),
        source_identity_kind(source),
        source.variant,
        "mab_conflict_resolution_normalized_substring_exact_match_v1",
        len(scores),
        len(predictions),
        sum(score.substring_exact_match for score in scores) / len(scores),
        tuple(anomalies),
        tuple(scores),
    )
