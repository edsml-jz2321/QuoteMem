"""Source-reviewed quality diagnostics with frozen labels and explicit missing data.

These pure scorers do not produce labels, collect provider output, run retrieval,
or certify reviewer independence. Their provenance is a recorded claim that the
experiment owner must establish from retained review and execution receipts.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Annotated, Literal, cast, get_args

from pydantic import BeforeValidator, ConfigDict, Field, ValidationError

from quotemem.bench.datasets import Digest, Nonempty
from quotemem.core.errors import QuoteMemError
from quotemem.core.identity import canonical_json
from quotemem.core.sensitivity import SensitivityClass
from quotemem.store.metadata import StrictModel

Timestamp = Annotated[int, Field(ge=0)]
KeywordArm = Literal["model", "legacy_tokens", "disabled"]


def _freeze_tuple(value: object) -> object:
    return tuple(cast(list[object], value)) if isinstance(value, list) else value


type FrozenTuple[T] = Annotated[tuple[T, ...], BeforeValidator(_freeze_tuple)]


class _QualityModel(StrictModel):
    model_config = ConfigDict(frozen=True, hide_input_in_errors=True)


class Reviewer(_QualityModel):
    """Declared reviewer identity and origin; AI review is never human truth."""

    reviewer_id: Nonempty
    kind: Literal["human", "ai"]


class ReviewRegistration(_QualityModel):
    """Review conditions frozen before evaluated output collection.

    Use separate registrations for representative samples and directed cases,
    and for each source-cluster split. The population and split manifests are
    external immutable artifacts; this object records their exact byte hashes.
    """

    population_sha256: Digest
    sampling_protocol: Nonempty
    sample_kind: Literal["representative", "directed"]
    split: Nonempty
    cluster_split_sha256: Digest
    rubric_sha256: Digest
    reviewers: Annotated[FrozenTuple[Reviewer], Field(min_length=1)]
    frozen_at_ns: Timestamp


class SourceCase(_QualityModel):
    """Exact reviewed source identity within the registered source-cluster split."""

    case_id: Nonempty
    source_sha256: Digest
    source_cluster: Nonempty


class SensitivityLabel(SourceCase):
    """Exhaustive disjoint present, absent and undecided category labels."""

    present: FrozenTuple[SensitivityClass]
    absent: FrozenTuple[SensitivityClass]
    undecided: FrozenTuple[SensitivityClass]


class SensitivityLabels(_QualityModel):
    """Source-only labels; every case must cover all seven existing categories."""

    registration: ReviewRegistration
    cases: Annotated[FrozenTuple[SensitivityLabel], Field(min_length=1)]

    @property
    def digest(self) -> str:
        """Canonical label and registration hash, bound into each observation."""
        return _digest(self)


class SensitivityObservation(_QualityModel):
    """Model nominations and final code-floor union from one exact source case.

    None means no usable output for that stage; an empty tuple means an actual
    completed assessment with no nominated category. Missing output remains in
    the gold-positive recall denominator and is separately counted.
    """

    case_id: Nonempty
    source_sha256: Digest
    labels_sha256: Digest
    execution_sha256: Digest
    collected_at_ns: Timestamp
    model_categories: FrozenTuple[SensitivityClass] | None
    final_categories: FrozenTuple[SensitivityClass] | None


@dataclass(frozen=True)
class CategoryScore:
    """Counts and micro-compatible rates; undefined denominators produce None."""

    category: str
    true_positive: int
    false_positive: int
    false_negative: int
    true_negative: int
    undecided_gold: int
    positives_on_undecided: int
    missing_predictions: int
    precision: float | None
    recall: float | None
    f1: float | None


@dataclass(frozen=True)
class SensitivityCaseScore:
    """Exact category errors for one stage; missing is distinct from empty."""

    case_id: str
    prediction_present: bool
    false_positives: tuple[str, ...]
    false_negatives: tuple[str, ...]
    undecided_gold: tuple[str, ...]
    positives_on_undecided: tuple[str, ...]


@dataclass(frozen=True)
class SensitivityStageScore:
    """Complete case denominator, category counts, micro and defined macro F1."""

    stage: str
    case_count: int
    prediction_count: int
    categories: tuple[CategoryScore, ...]
    micro: CategoryScore
    macro_f1: float | None
    macro_f1_category_count: int
    cases: tuple[SensitivityCaseScore, ...]


@dataclass(frozen=True)
class SensitivityReport:
    """Paired model/final diagnostics with preserved declared review provenance."""

    labels_sha256: str
    observations_sha256: str
    registration: ReviewRegistration
    model: SensitivityStageScore
    final: SensitivityStageScore


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _records_digest(records: tuple[StrictModel, ...]) -> str:
    return _digest([record.model_dump(mode="json") for record in records])


def _validated[T: StrictModel](value: T) -> T:
    # Revalidate model_copy/model_construct inputs at the public scoring boundary.
    try:
        return type(value).model_validate_json(value.model_dump_json())
    except ValidationError as error:
        raise QuoteMemError("evaluation_invalid", "Quality input schema is invalid") from error


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise QuoteMemError("evaluation_invalid", message)


def _unique(values: tuple[str, ...], name: str) -> None:
    _require(len(values) == len(set(values)), f"Duplicate {name}")


def _registration(registration: ReviewRegistration, cases: tuple[SourceCase, ...]) -> None:
    _unique(tuple(reviewer.reviewer_id for reviewer in registration.reviewers), "reviewer")
    _unique(tuple(case.case_id for case in cases), "source case")


def _observation(
    case: SourceCase, *, source_sha256: str, labels_sha256: str, expected_digest: str,
    collected_at_ns: int, frozen_at_ns: int,
) -> None:
    _require(case.source_sha256 == source_sha256, "Observation source differs from reviewed source")
    _require(labels_sha256 == expected_digest, "Observation differs from frozen labels")
    _require(collected_at_ns > frozen_at_ns, "Output collection must follow the label freeze")


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _category_score(category: str, counts: tuple[int, ...]) -> CategoryScore:
    tp, fp, fn, tn, undecided, unknown_positive, missing = counts
    return CategoryScore(
        category, tp, fp, fn, tn, undecided, unknown_positive, missing,
        _ratio(tp, tp + fp), _ratio(tp, tp + fn), _ratio(2 * tp, 2 * tp + fp + fn),
    )


def _sensitivity_stage(
    labels: SensitivityLabels, observations: dict[str, SensitivityObservation], *,
    final: bool,
) -> SensitivityStageScore:
    counts = {category: [0] * 7 for category in get_args(SensitivityClass)}
    cases: list[SensitivityCaseScore] = []
    for case in labels.cases:
        observation = observations.get(case.case_id)
        output = None if observation is None else (
            observation.final_categories if final else observation.model_categories
        )
        prediction = set(output or ())
        positive, negative, unknown = set(case.present), set(case.absent), set(case.undecided)
        for category, count in counts.items():
            predicted = category in prediction
            if category in unknown:
                count[4] += 1
                count[5] += predicted
            elif category in positive:
                count[0 if predicted else 2] += 1
            else:
                count[1 if predicted else 3] += 1
            count[6] += output is None
        cases.append(SensitivityCaseScore(
            case.case_id, output is not None, tuple(sorted(prediction & negative)),
            tuple(sorted(positive - prediction)), tuple(sorted(unknown)),
            tuple(sorted(prediction & unknown)),
        ))
    categories = tuple(
        _category_score(category, tuple(count)) for category, count in counts.items()
    )
    micro = _category_score("micro", tuple(sum(count[i] for count in counts.values())
                                           for i in range(7)))
    defined_f1 = [category.f1 for category in categories if category.f1 is not None]
    return SensitivityStageScore(
        "final_code_floor_union" if final else "model_nominations", len(cases),
        sum(case.prediction_present for case in cases), categories, micro,
        math.fsum(defined_f1) / len(defined_f1) if defined_f1 else None,
        len(defined_f1), tuple(cases),
    )


def score_sensitivity(
    labels: SensitivityLabels, observations: tuple[SensitivityObservation, ...],
) -> SensitivityReport:
    """Score source-labelled nominations and the final union independently.

    Args:
        labels (SensitivityLabels): Complete frozen review cohort. Undecided
            categories are excluded from confusion counts and explicitly retained.
        observations (tuple[SensitivityObservation, ...]): Unique actual outputs.
            Missing cases/stages count as no positive prediction, so missing
            output cannot shrink the positive-gold recall denominator.

    Returns:
        SensitivityReport: Counts, per-case errors and rates, without a passing
            threshold or claim that declared reviewers are independent humans.

    Raises:
        QuoteMemError: If categories, identities, chronology or label bindings differ.
    """
    labels = _validated(labels)
    observations = tuple(_validated(observation) for observation in observations)
    digest = labels.digest
    _registration(labels.registration, labels.cases)
    categories = set(get_args(SensitivityClass))
    for case in labels.cases:
        values = case.present + case.absent + case.undecided
        _unique(values, "sensitivity label")
        _require(set(values) == categories, "Source labels must cover all sensitivity categories")
    _unique(tuple(observation.case_id for observation in observations), "sensitivity observation")
    _require(len({observation.execution_sha256 for observation in observations}) <= 1,
             "Sensitivity observations must share one frozen execution manifest")
    by_id = {case.case_id: case for case in labels.cases}
    _require({item.case_id for item in observations} <= by_id.keys(), "Unknown sensitivity case")
    for observation in observations:
        _observation(
            by_id[observation.case_id], source_sha256=observation.source_sha256,
            labels_sha256=observation.labels_sha256, expected_digest=digest,
            collected_at_ns=observation.collected_at_ns,
            frozen_at_ns=labels.registration.frozen_at_ns,
        )
        for output in (observation.model_categories, observation.final_categories):
            if output is not None:
                _unique(output, "predicted sensitivity category")
        if observation.model_categories is not None and observation.final_categories is not None:
            _require(set(observation.model_categories) <= set(observation.final_categories),
                     "The final code-floor union cannot omit a model nomination")
    mapped = {observation.case_id: observation for observation in observations}
    return SensitivityReport(
        digest, _records_digest(observations), labels.registration,
        _sensitivity_stage(labels, mapped, final=False),
        _sensitivity_stage(labels, mapped, final=True),
    )


class KeywordConcept(_QualityModel):
    """One source-reviewed concept; wording may include entities or multiword terms."""

    concept_id: Nonempty
    description: Nonempty


class KeywordLabel(SourceCase):
    """Concepts selected from source without seeing evaluated keywords."""

    concepts: FrozenTuple[KeywordConcept]


class KeywordLabels(_QualityModel):
    """Independent source concepts and their original sampling/review conditions."""

    registration: ReviewRegistration
    cases: Annotated[FrozenTuple[KeywordLabel], Field(min_length=1)]

    @property
    def digest(self) -> str:
        """Canonical source-concept and registration hash."""
        return _digest(self)


class KeywordObservation(_QualityModel):
    """Exact emitted terms; no stemming or token splitting is done by this scorer."""

    case_id: Nonempty
    source_sha256: Digest
    labels_sha256: Digest
    execution_sha256: Digest
    collected_at_ns: Timestamp
    keywords: FrozenTuple[Nonempty]

    @property
    def digest(self) -> str:
        """Exact output binding used by independent term reviews."""
        return _digest(self)


class KeywordTermReview(_QualityModel):
    """Explicit review of one emitted position, including synonyms and repetition.

    useful maps to frozen source concept IDs; filler is supported but not useful;
    unsupported adds a meaning absent from source; undecided records uncertainty.
    An unreviewed position is also undecided, never automatically unsupported.
    """

    index: Annotated[int, Field(ge=0)]
    verdict: Literal["useful", "filler", "unsupported", "undecided"]
    concept_ids: FrozenTuple[Nonempty]


class KeywordReview(_QualityModel):
    """Output-aware independent review bound to exact terms and declared reviewer."""

    case_id: Nonempty
    observation_sha256: Digest
    reviewer_id: Nonempty
    reviewed_at_ns: Timestamp
    terms: FrozenTuple[KeywordTermReview]


@dataclass(frozen=True)
class KeywordCaseScore:
    """Complete emitted and expected denominators, retaining unreviewed positions."""

    case_id: str
    prediction_present: bool
    terms: tuple[str, ...]
    useful_indices: tuple[int, ...]
    filler_indices: tuple[int, ...]
    unsupported_indices: tuple[int, ...]
    undecided_indices: tuple[int, ...]
    expected_concept_count: int
    covered_concepts: tuple[str, ...]
    uncovered_concepts: tuple[str, ...]


@dataclass(frozen=True)
class KeywordReport:
    """Term relevance bounds and known concept coverage; no automatic semantic judge."""

    labels_sha256: str
    observations_sha256: str
    reviews_sha256: str
    registration: ReviewRegistration
    case_count: int
    prediction_count: int
    emitted_term_count: int
    useful_term_count: int
    filler_term_count: int
    unsupported_term_count: int
    undecided_term_count: int
    expected_concept_count: int
    covered_concept_count: int
    useful_rate_lower: float | None
    useful_rate_upper: float | None
    concept_coverage_lower: float | None
    cases: tuple[KeywordCaseScore, ...]


def score_keywords(
    labels: KeywordLabels, observations: tuple[KeywordObservation, ...],
    reviews: tuple[KeywordReview, ...],
) -> KeywordReport:
    """Score explicit term reviews against concepts frozen from original source.

    Args:
        labels (KeywordLabels): Source concepts frozen before keyword collection.
        observations (tuple[KeywordObservation, ...]): Exact outputs for any subset
            of the cohort; missing outputs keep all expected concept denominators.
        reviews (tuple[KeywordReview, ...]): At most one independent review per
            case. Missing positions remain undecided. Reviewer origin is retained.

    Returns:
        KeywordReport: Useful-rate bounds, supported concept coverage and explicit
            filler/unsupported/undecided positions. Uncovered concepts can be
            called missing only after their emitted terms have been decided.

    Raises:
        QuoteMemError: If source, output, reviewer, timing or concept bindings differ.
    """
    labels = _validated(labels)
    observations = tuple(_validated(observation) for observation in observations)
    reviews = tuple(_validated(review) for review in reviews)
    digest = labels.digest
    _registration(labels.registration, labels.cases)
    _unique(tuple(observation.case_id for observation in observations), "keyword observation")
    _require(len({observation.execution_sha256 for observation in observations}) <= 1,
             "Keyword observations must share one frozen execution manifest")
    _unique(tuple(review.case_id for review in reviews), "keyword review")
    source = {case.case_id: case for case in labels.cases}
    output = {observation.case_id: observation for observation in observations}
    judgments = {review.case_id: review for review in reviews}
    _require(output.keys() <= source.keys(), "Unknown keyword case")
    _require(judgments.keys() <= output.keys(), "Keyword review lacks an actual output")
    reviewers = {reviewer.reviewer_id for reviewer in labels.registration.reviewers}
    cases: list[KeywordCaseScore] = []
    for case in labels.cases:
        expected = tuple(concept.concept_id for concept in case.concepts)
        _unique(expected, "source concept")
        observation, review = output.get(case.case_id), judgments.get(case.case_id)
        if observation is not None:
            _observation(
                case, source_sha256=observation.source_sha256,
                labels_sha256=observation.labels_sha256, expected_digest=digest,
                collected_at_ns=observation.collected_at_ns,
                frozen_at_ns=labels.registration.frozen_at_ns,
            )
        keywords = observation.keywords if observation is not None else ()
        terms: dict[int, KeywordTermReview] = {}
        if review is not None:
            assert observation is not None
            _require(review.observation_sha256 == observation.digest,
                     "Keyword review output changed")
            _require(review.reviewer_id in reviewers, "Keyword reviewer was not registered")
            _require(review.reviewed_at_ns >= observation.collected_at_ns,
                     "Keyword review predates its output")
            terms = {term.index: term for term in review.terms}
            _require(len(terms) == len(review.terms), "Duplicate keyword position review")
            for term in review.terms:
                _require(term.index < len(keywords), "Keyword review position exceeds its output")
                _unique(term.concept_ids, "reviewed concept")
                _require(set(term.concept_ids) <= set(expected), "Review invents a source concept")
                _require(bool(term.concept_ids) == (term.verdict == "useful"),
                         "Only useful terms must identify supported source concepts")
        positions: dict[str, list[int]] = {
            verdict: [] for verdict in ("useful", "filler", "unsupported", "undecided")
        }
        covered: set[str] = set()
        for index in range(len(keywords)):
            term = terms.get(index)
            positions[term.verdict if term is not None else "undecided"].append(index)
            if term is not None:
                covered.update(term.concept_ids)
        cases.append(KeywordCaseScore(
            case.case_id, observation is not None, keywords,
            tuple(positions["useful"]), tuple(positions["filler"]),
            tuple(positions["unsupported"]), tuple(positions["undecided"]), len(expected),
            tuple(sorted(covered)), tuple(sorted(set(expected) - covered)),
        ))
    emitted = sum(len(case.terms) for case in cases)
    useful = sum(len(case.useful_indices) for case in cases)
    undecided = sum(len(case.undecided_indices) for case in cases)
    expected_count = sum(case.expected_concept_count for case in cases)
    covered_count = sum(len(case.covered_concepts) for case in cases)
    return KeywordReport(
        digest, _records_digest(observations), _records_digest(reviews),
        labels.registration, len(cases), len(output), emitted, useful,
        sum(len(case.filler_indices) for case in cases),
        sum(len(case.unsupported_indices) for case in cases), undecided,
        expected_count, covered_count, _ratio(useful, emitted),
        _ratio(useful + undecided, emitted), _ratio(covered_count, expected_count), tuple(cases),
    )


class RetrievalLabel(SourceCase):
    """Frozen query relevance judgments; omitted targets remain unjudged."""

    query_sha256: Digest
    relevant_targets: FrozenTuple[Nonempty]
    nonrelevant_targets: FrozenTuple[Nonempty]
    undecided_targets: FrozenTuple[Nonempty]


class RetrievalControls(_QualityModel):
    """Common ablation inputs, independently checked by the execution owner.

    The corpus manifest includes exact claims, summaries, vectors, object IDs,
    permissions and other retrieval inputs while excluding the varied keywords.
    A fixed-route diagnostic cannot establish the production route-selection effect.
    """

    source_code_sha256: Digest
    corpus_without_keywords_sha256: Digest
    query_set_sha256: Digest
    policy_sha256: Digest
    top_k: Annotated[int, Field(ge=1)]
    route_selection: Literal["recomputed_per_arm", "fixed_route_diagnostic"]
    fixed_routes: FrozenTuple[Nonempty] = ()


class RetrievalLabels(_QualityModel):
    """Query/source judgments and comparison controls frozen before all three arms."""

    registration: ReviewRegistration
    controls: RetrievalControls
    cases: Annotated[FrozenTuple[RetrievalLabel], Field(min_length=1)]

    @property
    def digest(self) -> str:
        """Canonical query judgments and frozen comparison identity."""
        return _digest(self)


class RetrievalObservation(_QualityModel):
    """Observed candidate pool and returned ordering, bound to one frozen query."""

    case_id: Nonempty
    query_sha256: Digest
    collected_at_ns: Timestamp
    pool_targets: FrozenTuple[Nonempty]
    ranked_targets: FrozenTuple[Nonempty]
    active_routes: FrozenTuple[Nonempty]


class RetrievalArm(_QualityModel):
    """One isolated keyword arm; missing cases remain explicit nonpairs."""

    arm: KeywordArm
    labels_sha256: Digest
    controls: RetrievalControls
    keyword_payload_sha256: Digest
    execution_sha256: Digest
    observations: FrozenTuple[RetrievalObservation]


@dataclass(frozen=True)
class RetrievalCaseScore:
    """Known relevance counts with unjudged/undecided output kept separate."""

    case_id: str
    prediction_present: bool
    relevant_target_count: int
    pool_size: int
    pool_relevant: int
    pool_unjudged: int
    returned_count: int
    top_k_relevant: int
    top_k_unjudged: int
    top_k: int
    pool_coverage: float | None
    recall_at_k: float | None
    precision_at_k_lower: float
    precision_at_k_upper: float


@dataclass(frozen=True)
class RetrievalArmScore:
    """Complete query denominator and missing outputs for one diagnostic arm."""

    arm: str
    case_count: int
    observation_count: int
    mean_precision_at_k_lower: float
    mean_precision_at_k_upper: float
    pooled_known_relevant_coverage: float | None
    pooled_known_relevant_recall_at_k: float | None
    cases: tuple[RetrievalCaseScore, ...]


@dataclass(frozen=True)
class RetrievalPairScore:
    """Wins/losses use known relevant counts and only genuinely observed pairs."""

    comparator: str
    paired_case_count: int
    unpaired_case_ids: tuple[str, ...]
    top_k_wins: tuple[str, ...]
    top_k_losses: tuple[str, ...]
    top_k_ties: tuple[str, ...]
    pool_wins: tuple[str, ...]
    pool_losses: tuple[str, ...]
    pool_ties: tuple[str, ...]


@dataclass(frozen=True)
class RetrievalReport:
    """Controlled diagnostic comparisons, never an automatic quality pass."""

    labels_sha256: str
    arms_sha256: str
    registration: ReviewRegistration
    controls: RetrievalControls
    arms: tuple[RetrievalArmScore, ...]
    comparisons: tuple[RetrievalPairScore, ...]


def _retrieval_arm(labels: RetrievalLabels, arm: RetrievalArm) -> RetrievalArmScore:
    output = {observation.case_id: observation for observation in arm.observations}
    cases: list[RetrievalCaseScore] = []
    k = labels.controls.top_k
    for case in labels.cases:
        observation = output.get(case.case_id)
        relevant = set(case.relevant_targets)
        judged = relevant | set(case.nonrelevant_targets)
        pool = set(observation.pool_targets) if observation is not None else set[str]()
        top = set(observation.ranked_targets[:k]) if observation is not None else set[str]()
        pool_hits, top_hits = len(pool & relevant), len(top & relevant)
        unknown_top = len(top - judged)
        cases.append(RetrievalCaseScore(
            case.case_id, observation is not None, len(relevant), len(pool), pool_hits,
            len(pool - judged), len(top), top_hits, unknown_top, k,
            _ratio(pool_hits, len(relevant)), _ratio(top_hits, len(relevant)),
            top_hits / k, (top_hits + unknown_top) / k,
        ))
    denominator = sum(case.relevant_target_count for case in cases)
    return RetrievalArmScore(
        arm.arm, len(cases), len(output),
        math.fsum(case.precision_at_k_lower for case in cases) / len(cases),
        math.fsum(case.precision_at_k_upper for case in cases) / len(cases),
        _ratio(sum(case.pool_relevant for case in cases), denominator),
        _ratio(sum(case.top_k_relevant for case in cases), denominator), tuple(cases),
    )


def _retrieval_pair(model: RetrievalArmScore, other: RetrievalArmScore) -> RetrievalPairScore:
    top: dict[int, list[str]] = {1: [], -1: [], 0: []}
    pool: dict[int, list[str]] = {1: [], -1: [], 0: []}
    missing: list[str] = []
    for left, right in zip(model.cases, other.cases, strict=True):
        if not left.prediction_present or not right.prediction_present:
            missing.append(left.case_id)
            continue
        delta = left.top_k_relevant - right.top_k_relevant
        top[(delta > 0) - (delta < 0)].append(left.case_id)
        delta = left.pool_relevant - right.pool_relevant
        pool[(delta > 0) - (delta < 0)].append(left.case_id)
    return RetrievalPairScore(
        other.arm, len(model.cases) - len(missing), tuple(missing),
        tuple(top[1]), tuple(top[-1]), tuple(top[0]),
        tuple(pool[1]), tuple(pool[-1]), tuple(pool[0]),
    )


def score_keyword_retrieval(
    labels: RetrievalLabels, arms: tuple[RetrievalArm, ...],
) -> RetrievalReport:
    """Compare three keyword arms on one registered query/source denominator.

    Args:
        labels (RetrievalLabels): Frozen relevance and exact shared input controls.
        arms (tuple[RetrievalArm, ...]): Exactly model, legacy_tokens and disabled.
            Every arm must share controls and freeze binding. Missing observations
            score zero known coverage but do not create a win/loss pair.

    Returns:
        RetrievalReport: Per-case pool/top-k diagnostics, known relevance bounds
            and paired wins/losses. Execution receipts must independently prove
            the input hashes and production route-selection behavior.

    Raises:
        QuoteMemError: If arm controls, review, query, ordering or freeze bindings differ.
    """
    labels = _validated(labels)
    arms = tuple(_validated(arm) for arm in arms)
    digest = labels.digest
    _registration(labels.registration, labels.cases)
    _require(len(arms) == 3 and {arm.arm for arm in arms} == set(get_args(KeywordArm)),
             "Retrieval comparison requires exactly three distinct keyword arms")
    _unique(labels.controls.fixed_routes, "fixed route")
    _require(bool(labels.controls.fixed_routes)
             == (labels.controls.route_selection == "fixed_route_diagnostic"),
             "Only the fixed-route diagnostic must declare fixed routes")
    source = {case.case_id: case for case in labels.cases}
    for case in labels.cases:
        _unique(case.relevant_targets + case.nonrelevant_targets + case.undecided_targets,
                "retrieval relevance label")
    for arm in arms:
        _require(arm.labels_sha256 == digest, "Retrieval arm differs from frozen labels")
        _require(arm.controls == labels.controls, "Retrieval arm changes a shared control")
        _unique(tuple(observation.case_id for observation in arm.observations), "retrieval case")
        _require({item.case_id for item in arm.observations} <= source.keys(),
                 "Unknown retrieval case")
        for observation in arm.observations:
            case = source[observation.case_id]
            _require(observation.query_sha256 == case.query_sha256, "Retrieval query changed")
            _require(observation.collected_at_ns > labels.registration.frozen_at_ns,
                     "Retrieval collection must follow label freeze")
            _unique(observation.pool_targets, "candidate pool target")
            _unique(observation.ranked_targets, "ranked target")
            _unique(observation.active_routes, "active route")
            _require(set(observation.ranked_targets) <= set(observation.pool_targets),
                     "Ranked output is outside the observed candidate pool")
            if labels.controls.route_selection == "fixed_route_diagnostic":
                _require(observation.active_routes == labels.controls.fixed_routes,
                         "Diagnostic observation changed the frozen routes")
    by_arm = {arm.arm: arm for arm in arms}
    scores = tuple(_retrieval_arm(labels, by_arm[arm])
                   for arm in ("model", "legacy_tokens", "disabled"))
    return RetrievalReport(
        digest, _records_digest(arms), labels.registration, labels.controls, scores,
        tuple(_retrieval_pair(scores[0], other) for other in scores[1:]),
    )
