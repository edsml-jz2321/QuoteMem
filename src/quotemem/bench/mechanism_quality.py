"""Consume independent review and execution evidence without generating truth.

The complete-source dataset, answer and resource owners remain authoritative.
This arithmetic consumer verifies bindings, not witness custody, reviewer
independence or a causal effect of disabling a publication mechanism.

Each arm's evidence is its NL (natural-language) census: the frozen census of the statements and
positions it emitted, reviewed against the registered facts, and the record of
its provider calls. The QuoteMem arm's calls are its runtime call census
(``quotemem.bench.call_census``), whose attempts carry the request or owner phase that
the runtime recorded; a baseline's calls are its own producer's report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated, Literal, Self, cast, get_args

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)

from quotemem.bench.answer_comparison import (
    AnswerComparisonArm,
    AnswerComparisonRegistration,
    AnswerComparisonReport,
    compare_answers,
)
from quotemem.bench.baseline_resources import (
    BaselineResourceBindingV1,
    BaselineResourceRegistrationV1,
    validate_baseline_resources,
)
from quotemem.bench.datasets import (
    BenchmarkCorpus,
    Digest,
    Nonempty,
    decode_source_json,
    load_dataset,
    read_source_json,
    source_identity,
)
from quotemem.bench.providers import BenchmarkUsage, benchmark_digest
from quotemem.bench.quality_evaluation import FrozenTuple, ReviewRegistration
from quotemem.bench.replay import prepare_replay
from quotemem.bench.resources import ProviderCallObservation
from quotemem.core.errors import QuoteMemError
from quotemem.core.identity import canonical_json, integrity_digest, scope_id
from quotemem.service.cost_attribution import RESOURCE_PHASES, ResourcePhase
from quotemem.store.metadata import StrictModel

_CENSUS_MAX_BYTES = 32 * 1024 * 1024
Commit = Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
Support = Literal["supported", "unsupported", "unknown"]
Permission = Literal["permitted", "denied", "unknown"]
State = Literal["present", "absent", "unknown"]
Stage = Literal["extraction", "refusal", "publication", "retrieval", "answering", "unknown", "none"]


class _Record(StrictModel):
    model_config = ConfigDict(frozen=True, hide_input_in_errors=True)


class SourceReference(_Record):
    """An original message reference whose exact bytes are checked against the corpus."""

    session_id: Nonempty
    message_id: Nonempty
    content_sha256: Digest


class ReviewWitness(_Record):
    """Declared review boundary and retained witness; custody needs external verification."""

    reviewer_ids: Annotated[FrozenTuple[Nonempty], Field(min_length=1)]
    method: Literal["source_only", "blinded_output_alignment"]
    witness_sha256: Digest
    reviewed_at_ns: Annotated[int, Field(ge=0)]


class FactExpectation(_Record):
    """One preregistered fact, including undecided and policy-denied expectations."""

    fact_id: Nonempty
    fact_content_sha256: Digest
    conversation_id: Nonempty
    sources: Annotated[FrozenTuple[SourceReference], Field(min_length=1)]
    support: Support
    permission: Permission
    review: ReviewWitness | None


class QuestionExpectation(_Record):
    """Every original question, with separately reviewed current-correction scope."""

    case_id: Nonempty
    fact_ids: FrozenTuple[Nonempty]
    stale_current_eligible: Literal["yes", "no", "unknown"]
    review: ReviewWitness | None


class MechanismRegistration(_Record):
    """Frozen full-source questions/facts and exact resource joins for two existing arms."""

    schema_version: Literal[1]
    scope: Literal["complete_registered_source"]
    comparison: AnswerComparisonRegistration
    review: ReviewRegistration
    independence_witness_sha256: Digest
    exposure_witness_sha256: Digest
    held_out_status: Literal["declared_held_out", "exposed", "unknown"]
    fact_inventory_complete: bool
    facts: FrozenTuple[FactExpectation]
    questions: Annotated[FrozenTuple[QuestionExpectation], Field(min_length=1)]
    aml_resource_registration_sha256: Digest
    baseline_resource_registration_sha256: Digest

    @field_validator("comparison", mode="before")
    @classmethod
    def comparison_json(cls, value: object, info: ValidationInfo) -> object:
        """Preserve the nested strict model's JSON boundary after literal validation."""
        if info.mode == "json" and isinstance(value, dict):
            return AnswerComparisonRegistration.model_validate_json(
                canonical_json(cast(dict[str, object], value)),
            )
        return value


class ObservedState(_Record):
    """Known absence requires evidence just as presence does; None means unknown."""

    state: State
    witness_sha256: Digest | None

    @model_validator(mode="after")
    def witnessed(self) -> Self:
        if self.state != "unknown" and self.witness_sha256 is None:
            raise ValueError("A known stage state requires its actual observation witness")
        return self


class PositionIdentity(_Record):
    """One member of the complete census of emitted positions, frozen before review."""

    position_id: Nonempty
    conversation_id: Nonempty
    statement_sha256: Digest


class PositionObservation(_Record):
    """Independent support review and separately witnessed system decisions."""

    position_id: Nonempty
    fact_ids: FrozenTuple[Nonempty]
    statement_sha256: Digest
    sources: FrozenTuple[SourceReference]
    support: Support
    permission: Permission
    review: ReviewWitness | None
    decision: Literal["accepted", "refused", "unknown"]
    decision_witness_sha256: Digest | None
    published: ObservedState
    revision_id: Nonempty | None

    @model_validator(mode="after")
    def coherent(self) -> Self:
        if self.decision != "unknown" and self.decision_witness_sha256 is None:
            raise ValueError("An observed decision requires its witness")
        if (self.published.state == "present") != (self.revision_id is not None):
            raise ValueError("Published positions require their exact Revision identity")
        if self.decision == "refused" and self.published.state == "present":
            raise ValueError("A refused position cannot also be published")
        return self


class FactObservation(_Record):
    """Whole-prefix processing and fact-stage evidence, independent of answer success."""

    fact_id: Nonempty
    fact_content_sha256: Digest
    alignment_complete: ObservedState
    processing_complete: ObservedState
    extracted: ObservedState
    published: ObservedState
    position_ids: FrozenTuple[Nonempty]

    @model_validator(mode="after")
    def coherent(self) -> Self:
        if self.extracted.state == "absent" and (
            self.position_ids or self.published.state == "present"
        ):
            raise ValueError("An unextracted fact cannot name positions or published state")
        if self.extracted.state == "present" and not self.position_ids:
            raise ValueError("An extracted fact requires an observed position identity")
        return self


class FactRetrievalObservation(_Record):
    """One expected fact's eligibility and representation in the actual packed context."""

    fact_id: Nonempty
    eligible: ObservedState
    retrieved: ObservedState
    revision_ids: FrozenTuple[Nonempty]
    context_hit_ids: FrozenTuple[Nonempty]

    @model_validator(mode="after")
    def coherent(self) -> Self:
        if (self.eligible.state == "present") != bool(self.revision_ids):
            raise ValueError("Known eligible facts require exact published Revision identities")
        if (self.retrieved.state == "present") != bool(self.context_hit_ids):
            raise ValueError("Known retrieved facts require actual packed context hit identities")
        return self


class QuestionObservation(_Record):
    """Reviewed answer meaning plus separate observations for each expected fact."""

    case_id: Nonempty
    facts: FrozenTuple[FactRetrievalObservation]
    context_sufficient: ObservedState
    answer_assessment: Literal[
        "correct", "incorrect", "stale_current", "qualified_historical", "abstained", "unknown",
    ]
    answer_result_sha256: Digest | None
    review: ReviewWitness | None


class AnswerCostObservation(_Record):
    """Actual answer attempt, including failed attempts absent from successful results."""

    call_key: Nonempty
    case_id: Nonempty
    request_sha256: Digest
    result_sha256: Digest | None
    outcome: Literal["succeeded", "failed", "cancelled"]
    observation_witness_sha256: Digest
    usage: BenchmarkUsage | None
    cost_usd: Annotated[float | None, Field(ge=0, allow_inf_nan=False)]
    billing_witness_sha256: Digest | None

    @model_validator(mode="after")
    def witnessed(self) -> Self:
        if self.cost_usd is not None and self.billing_witness_sha256 is None:
            raise ValueError("Answer billing requires its retained actual charge witness")
        if (self.outcome == "succeeded") != (self.result_sha256 is not None):
            raise ValueError("Only an observed successful answer attempt binds a result")
        return self


class AmlCensusResourceRegistrationV1(_Record):
    """Prior join of the QuoteMem arm to the runtime call census of its own execution.

    The harness opens a ``quotemem.bench.call_census`` census with the ``runtime``
    boundary around the service's provider clients before the arm runs; the census
    records every HTTP attempt of those clients with the request or owner phase
    that the runtime recorded. Whole-process coverage and census custody need
    external verification; this registration binds the census to the arm.
    """

    schema_version: Literal[1]
    kind: Literal["aml_runtime_call_census_v1"]
    run_key: Nonempty
    source_commit: Commit
    registered_at_ms: Annotated[int, Field(ge=0)]
    preregistration_witness_sha256: Digest
    execution_registration_sha256: Digest


class AmlCensusBindingV1(_Record):
    """The QuoteMem arm's retained runtime call census file, byte for byte."""

    schema_version: Literal[1]
    census_file_sha256: Digest
    census_file_utf8: Annotated[
        str, Field(min_length=1, max_length=_CENSUS_MAX_BYTES, repr=False),
    ]

    @model_validator(mode="after")
    def exact_file(self) -> Self:
        payload = self.census_file_utf8.encode("utf-8")
        if len(payload) > _CENSUS_MAX_BYTES or hashlib.sha256(payload).hexdigest() != (
            self.census_file_sha256
        ):
            raise ValueError("QuoteMem census binding differs from the bounded original file")
        return self


class MechanismArmEvidence(_Record):
    """One exact arm's census, partial observations and separately recorded resource calls.

    The QuoteMem arm binds its runtime call census; a baseline binds its producer's
    report. Either binding may be absent, which leaves the arm's calls unknown.
    """

    schema_version: Literal[4]
    execution_evidence_sha256: Digest
    answer_report_sha256: Digest
    collected_at_ns: Annotated[int, Field(ge=0)]
    position_census_frozen_at_ns: Annotated[int, Field(ge=0)]
    position_census_witness_sha256: Digest
    position_census_complete: bool
    positions: FrozenTuple[PositionIdentity]
    position_observations: FrozenTuple[PositionObservation]
    fact_observations: FrozenTuple[FactObservation]
    question_observations: FrozenTuple[QuestionObservation]
    resource_binding_witness_sha256: Digest
    resource_registration: AmlCensusResourceRegistrationV1 | BaselineResourceRegistrationV1
    answer_history_complete: bool
    answer_calls: FrozenTuple[AnswerCostObservation]
    aml_census_binding: AmlCensusBindingV1 | None = Field(
        default=None, exclude_if=lambda value: value is None,
    )
    baseline_resource_binding: BaselineResourceBindingV1 | None = Field(
        default=None, exclude_if=lambda value: value is None,
    )

    @model_validator(mode="after")
    def resource_variant(self) -> Self:
        aml = isinstance(self.resource_registration, AmlCensusResourceRegistrationV1)
        if (self.aml_census_binding is not None and not aml) or (
            self.baseline_resource_binding is not None and aml
        ):
            raise ValueError("Resources require the binding of their own evidence variant")
        return self


@dataclass(frozen=True)
class QualityRate:
    """Full registered population, decided events and explicit missing-label bounds."""

    population: int
    eligible: int
    events: int
    non_events: int
    unknown: int
    excluded: int
    decided_rate: float | None
    lower_bound: float | None
    upper_bound: float | None


@dataclass(frozen=True)
class Attribution:
    identity: str
    stage: Stage


@dataclass(frozen=True)
class _ResourceCall:
    """One recorded provider attempt outside the answer calls; None is unknown."""

    call_key: str
    phase: ResourcePhase | None
    input_tokens: int | None
    output_tokens: int | None
    cost_usd: float | None


@dataclass(frozen=True)
class _ArmResources:
    """An arm's recorded provider attempts and how much of its process they cover."""

    calls: tuple[_ResourceCall, ...]
    history_complete: bool
    scope: Literal["runtime_call_census", "recorded_add_search_only_whole_process_unknown"]
    evidence_file_sha256: str | None


@dataclass(frozen=True)
class FactRetrievalScore:
    case_id: str
    fact_id: str
    status: str
    eligibility_observed: State
    retrieval_observed: State


@dataclass(frozen=True)
class MechanismArmScore:
    evidence_sha256: str
    position_count: int
    fact_count: int
    question_count: int
    position_census_complete: bool
    unsupported_publication: QualityRate
    false_refusal: QualityRate
    missing_facts: QualityRate
    retrieval_miss: QualityRate
    retrieval_scope: Literal["registered_case_fact_pairs_in_answer_packed_context"]
    retrieval_cases: tuple[FactRetrievalScore, ...]
    stale_current_answers: QualityRate
    fact_attribution: tuple[Attribution, ...]
    question_attribution: tuple[Attribution, ...]
    costs: dict[str, object]


@dataclass(frozen=True)
class MechanismQualityReport:
    registration_sha256: str
    answers: AnswerComparisonReport
    equal_conversation_f1_delta: float
    equal_conversation_exact_match_delta: float
    conversation_count: int
    fact_inventory_complete: bool
    held_out_status: str
    total_cost_delta_usd: float | None
    aml: MechanismArmScore
    baseline: MechanismArmScore
    decision: Literal["inconclusive"]
    limitations: tuple[str, ...]


def _require(value: bool, message: str) -> None:
    if not value:
        raise QuoteMemError("evaluation_invalid", message)


def _unique(values: tuple[str, ...], name: str) -> None:
    _require(len(values) == len(set(values)), f"Duplicate {name}")


def _review(
    witness: ReviewWitness | None, registration: MechanismRegistration, *, source_only: bool,
) -> None:
    if witness is None:
        return
    _unique(witness.reviewer_ids, "reviewer identity")
    _require(
        set(witness.reviewer_ids) <= {row.reviewer_id for row in registration.review.reviewers},
        "Review identity is outside the independent review registration",
    )
    _require(
        witness.method == ("source_only" if source_only else "blinded_output_alignment"),
        "Review used the wrong information boundary",
    )
    if source_only:
        _require(witness.reviewed_at_ns <= registration.review.frozen_at_ns,
                 "Source expectations were not reviewed before their freeze")


def _sources(
    corpus: BenchmarkCorpus, conversation: str, references: tuple[SourceReference, ...],
) -> None:
    known = {
        (session.session_id, message.message_id): hashlib.sha256(
            message.content.encode(),
        ).hexdigest()
        for case in corpus.cases if case.cluster_id == conversation
        for session in case.sessions for message in session.messages
    }
    identities = tuple(f"{row.session_id}\0{row.message_id}" for row in references)
    _unique(identities, "source reference")
    _require(bool(known), "Review conversation is outside the registered source")
    _require(all(known.get((row.session_id, row.message_id)) == row.content_sha256
                 for row in references), "Review source bytes or message identity differ")


def _registration(corpus: BenchmarkCorpus, registration: MechanismRegistration) -> None:
    _require(registration.review.population_sha256 == source_identity(corpus.source),
             "Review population differs from the complete registered source")
    _unique(tuple(row.reviewer_id for row in registration.review.reviewers), "reviewer")
    _unique(tuple(row.fact_id for row in registration.facts), "expected fact")
    _require(
        tuple(row.case_id for row in registration.questions) == registration.comparison.case_ids,
        "Question expectations must preserve the full ordered question denominator",
    )
    facts = {row.fact_id: row for row in registration.facts}
    for fact in registration.facts:
        _sources(corpus, fact.conversation_id, fact.sources)
        _review(fact.review, registration, source_only=True)
        _require(fact.review is not None or (fact.support == fact.permission == "unknown"),
                 "A decided expected fact requires independent review")
    for question, case in zip(registration.questions, corpus.cases, strict=True):
        _unique(question.fact_ids, "question fact reference")
        _require(all(key in facts and facts[key].conversation_id == case.cluster_id
                     for key in question.fact_ids),
                 "Question refers to unknown or cross-scope facts")
        _review(question.review, registration, source_only=True)
        _require(question.review is not None or question.stale_current_eligible == "unknown",
                 "Current-correction eligibility requires independent source review")


def _rate(states: list[str]) -> QualityRate:
    events, negative, unknown, excluded = (states.count(key) for key in
                                         ("event", "non_event", "unknown", "excluded"))
    eligible = events + negative + unknown
    decided = events + negative
    return QualityRate(
        len(states), eligible, events, negative, unknown, excluded,
        events / decided if decided else None, events / eligible if eligible else None,
        (events + unknown) / eligible if eligible else None,
    )


def _eligibility(support: Support, permission: Permission) -> str:
    if support == "unsupported" or permission == "denied":
        return "excluded"
    return "eligible" if support == "supported" and permission == "permitted" else "unknown"


_CENSUS_OPENED = frozenset({
    "event", "schema_version", "kind", "census_id", "boundary", "started_epoch_ns", "usd",
    "retry_owner",
})
_CENSUS_STARTED = frozenset({
    "event", "attempt", "call_key", "call", "endpoint", "method", "started_epoch_ns",
    "runtime_scope_id", "runtime_phase", "runtime_run_id",
})
_CENSUS_TAGS = frozenset({"purpose", "embedding_role"})
"""Optional call tags of an attempt, which the grading metrics read."""
_CENSUS_FINISHED = frozenset({
    "event", "attempt", "elapsed_ns", "status_code", "outcome", "timeout_phase", "usage",
    "response_model", "cost_usd", "billing_witness_sha256",
})
_CENSUS_CLOSED = frozenset({
    "event", "census_id", "logical_calls", "http_attempts", "unfinished_calls",
    "unfinished_attempts", "unattributed_answer_attempts", "capture_complete",
    "finished_epoch_ns", "cost_usd",
})
_CENSUS_USAGE = frozenset({"prompt_tokens", "completion_tokens", "total_tokens", "cached_tokens"})
_CENSUS_ENDPOINTS = frozenset({"chat/completions", "embeddings", "rerank", "other"})


def _count(value: object) -> bool:
    return type(value) is int and value >= 0


def _census_rows(text: str) -> list[dict[str, object]]:
    """Decode every census line, refusing any line the census itself did not write."""
    _require(text.endswith("\n"), "QuoteMem census file does not end with a complete line")
    rows: list[dict[str, object]] = []
    for line in text[:-1].split("\n"):
        try:
            row = decode_source_json(line.encode("utf-8"))
        except ValueError:
            raise QuoteMemError("evaluation_invalid",
                                "QuoteMem census line is not strict JSON") from None
        _require(isinstance(row, dict)
                 and json.dumps(row, sort_keys=True, ensure_ascii=True) == line,
                 "QuoteMem census line differs from the census's own encoding")
        rows.append(cast(dict[str, object], row))
    return rows


def _census_attempt(
    row: dict[str, object], *, number: int, census_id: str, scopes: frozenset[str],
) -> None:
    _require(_CENSUS_STARTED <= set(row) <= _CENSUS_STARTED | _CENSUS_TAGS,
             "QuoteMem census attempt differs from a recorded HTTP attempt")
    scope, phase, run = row["runtime_scope_id"], row["runtime_phase"], row["runtime_run_id"]
    _require(row["attempt"] == number and type(row["attempt"]) is int
             and row["call_key"] == f"{census_id}:{number}" and row["call"] is None
             and isinstance(row["endpoint"], str) and row["endpoint"] in _CENSUS_ENDPOINTS
             and isinstance(row["method"], str) and _count(row["started_epoch_ns"])
             and (scope is None or isinstance(scope, str) and scope in scopes)
             and (phase is None or isinstance(phase, str) and phase in RESOURCE_PHASES)
             and (run is None or isinstance(run, str)),
             "QuoteMem census attempt leaves the arm's registered runtime capture")


def _census_return(row: dict[str, object]) -> None:
    _require(_CENSUS_FINISHED <= set(row) <= _CENSUS_FINISHED | {"provider_error"},
             "QuoteMem census return differs from an attempt")
    if "provider_error" in row:
        diagnostic = row["provider_error"]
        fields = cast(dict[str, object], diagnostic) if isinstance(diagnostic, dict) else {}
        _require(type(row["status_code"]) is int and row["status_code"] != 200
                 and bool(fields) and set(fields) <= {"code", "message"}
                 and all(isinstance(value, str) for value in fields.values())
                 and len(json.dumps(fields, ensure_ascii=True).encode()) <= 2048,
                 "QuoteMem census provider error differs from a bounded code/message diagnostic")
    usage, cost = row["usage"], row["cost_usd"]
    values = cast(dict[str, object], usage) if isinstance(usage, dict) else None
    _require(_count(row["elapsed_ns"])
             and (row["status_code"] is None or _count(row["status_code"]))
             and isinstance(row["outcome"], str)
             and (row["response_model"] is None or isinstance(row["response_model"], str))
             and (row["timeout_phase"] is None or isinstance(row["timeout_phase"], str))
             and (usage is None or values is not None and frozenset(values) == _CENSUS_USAGE
                  and all(value is None or _count(value) for value in values.values())),
             "QuoteMem census return differs from a recorded HTTP attempt")
    witness = row["billing_witness_sha256"]
    _require((cost is None and witness is None) or (
        type(cost) in (int, float) and math.isfinite(cast(float, cost)) and cast(float, cost) >= 0
        and isinstance(witness, str) and re.fullmatch(r"[0-9a-f]{64}", witness) is not None
    ), "A QuoteMem census charge requires its retained billing witness")


def _census_calls(
    binding: AmlCensusBindingV1, *, scopes: frozenset[str], registered_at_ms: int,
) -> _ArmResources:
    """Read a runtime call census as the QuoteMem arm's provider attempts.

    Every attempt counts, finished or not; its phase is the one the runtime
    recorded, and an attempt without one stays unclassified. The
    history is complete only when the sealed census reports complete capture.

    Raises:
        QuoteMemError: The census does not belong to the registered runtime capture.
    """
    rows = _census_rows(binding.census_file_utf8)
    opened = rows[0]
    census_id = opened.get("census_id")
    _require(frozenset(opened) == _CENSUS_OPENED and opened["event"] == "opened"
             and type(opened["schema_version"]) is int and opened["schema_version"] == 1
             and opened["kind"] == "provider_http_call_census" and opened["boundary"] == "runtime"
             and isinstance(census_id, str) and re.fullmatch(r"[0-9a-f]{32}", census_id) is not None
             and opened["usd"] == "unknown" and opened["retry_owner"] == "existing_json_transport"
             and _count(opened["started_epoch_ns"])
             and cast(int, opened["started_epoch_ns"]) >= registered_at_ms * 1_000_000,
             "QuoteMem census does not open a runtime capture after its registration")
    assert isinstance(census_id, str)
    sealed = rows[-1] if len(rows) > 1 and rows[-1].get("event") == "closed" else None
    started: dict[int, dict[str, object]] = {}
    finished: dict[int, dict[str, object]] = {}
    for row in rows[1:-1] if sealed is not None else rows[1:]:
        if row.get("event") == "attempt_started":
            _census_attempt(row, number=len(started) + 1, census_id=census_id, scopes=scopes)
            started[len(started) + 1] = row
            continue
        number = row.get("attempt")
        _require(row.get("event") == "attempt_finished" and type(number) is int
                 and number in started and number not in finished,
                 "QuoteMem census holds an event outside one runtime attempt")
        assert type(number) is int
        _census_return(row)
        finished[number] = row
    complete = False
    if sealed is not None:
        unfinished = sorted(set(started) - set(finished))
        _require(frozenset(sealed) == _CENSUS_CLOSED and sealed["census_id"] == census_id
                 and type(sealed["logical_calls"]) is int and sealed["logical_calls"] == 0
                 and type(sealed["http_attempts"]) is int
                 and sealed["http_attempts"] == len(started)
                 and sealed["unfinished_calls"] == []
                 and sealed["unfinished_attempts"] == unfinished
                 and type(sealed["unattributed_answer_attempts"]) is int
                 and sealed["unattributed_answer_attempts"] == 0
                 and sealed["capture_complete"] is (not unfinished)
                 and _count(sealed["finished_epoch_ns"])
                 and cast(int, sealed["finished_epoch_ns"])
                 >= cast(int, opened["started_epoch_ns"])
                 and sealed["cost_usd"] is None,
                 "QuoteMem census seal differs from its recorded attempts")
        complete = not unfinished
    calls: list[_ResourceCall] = []
    for number, row in started.items():
        result = finished.get(number, {})
        usage = cast(dict[str, int | None] | None, result.get("usage"))
        cost = cast(float | None, result.get("cost_usd"))
        prompt = None if usage is None else usage["prompt_tokens"]
        if prompt is None and usage is not None and row["endpoint"] == "embeddings":
            prompt = usage["total_tokens"]
        calls.append(_ResourceCall(
            str(row["call_key"]), cast(ResourcePhase | None, row["runtime_phase"]), prompt,
            None if usage is None else usage["completion_tokens"],
            None if cost is None else float(cost),
        ))
    return _ArmResources(tuple(calls), complete, "runtime_call_census", binding.census_file_sha256)


def _arm_resources(
    corpus: BenchmarkCorpus, registration: MechanismRegistration,
    arm: AnswerComparisonArm, evidence: MechanismArmEvidence,
) -> _ArmResources:
    """Join each arm to the resource variant of its own retrieval origin."""
    resource = evidence.resource_registration
    if isinstance(resource, BaselineResourceRegistrationV1):
        recorded: tuple[ProviderCallObservation, ...] = validate_baseline_resources(
            corpus, registration.comparison, arm, resource, evidence.baseline_resource_binding,
        )
        binding = evidence.baseline_resource_binding
        return _ArmResources(
            tuple(_ResourceCall(row.call_key, row.phase, row.input_tokens, row.output_tokens,
                                row.cost_usd) for row in recorded),
            False, "recorded_add_search_only_whole_process_unknown",
            None if binding is None else binding.report_file_sha256,
        )
    _require(arm.registration.retrieval_origin == "aml_http"
             and resource.run_key == arm.registration.run_key
             and resource.source_commit == arm.registration.source_commit
             and resource.execution_registration_sha256
             == arm.registration.execution_registration_sha256,
             "QuoteMem census resources differ from their registered QuoteMem arm")
    if evidence.aml_census_binding is None:
        return _ArmResources((), False, "runtime_call_census", None)
    replay = registration.comparison.replay_config
    groups = prepare_replay(
        corpus, run_id=arm.registration.run_key, top_k=replay.top_k, max_top_k=replay.max_top_k,
        isolate_queries=replay.isolate_queries,
        source_metadata_version=replay.source_metadata_version,
    )
    return _census_calls(
        evidence.aml_census_binding, registered_at_ms=resource.registered_at_ms,
        scopes=frozenset(scope_id("user", add.user_id) for group in groups for add in group.adds),
    )


def _validate_arm(
    corpus: BenchmarkCorpus, registration: MechanismRegistration,
    arm: AnswerComparisonArm, evidence: MechanismArmEvidence,
) -> _ArmResources:
    resources = _arm_resources(corpus, registration, arm, evidence)
    _require(evidence.answer_report_sha256 == arm.answers.report_sha256
             and evidence.execution_evidence_sha256 == arm.execution_evidence_sha256,
             "Mechanism evidence differs from its exact answer/execution artifacts")
    _require(evidence.collected_at_ns > registration.review.frozen_at_ns,
             "Evaluated outputs must follow the source-expectation freeze")
    _require(evidence.position_census_frozen_at_ns >= evidence.collected_at_ns,
             "Position census must bind the collected outputs before their review")
    positions = {row.position_id: row for row in evidence.positions}
    facts = {row.fact_id: row for row in registration.facts}
    questions = {row.case_id: row for row in arm.answers.cases}
    for values, name in (
        (tuple(row.position_id for row in evidence.positions), "position census identity"),
        (tuple(row.position_id for row in evidence.position_observations), "position observation"),
        (tuple(row.fact_id for row in evidence.fact_observations), "fact observation"),
        (tuple(row.case_id for row in evidence.question_observations), "question observation"),
    ):
        _unique(values, name)
    for position in evidence.positions:
        _sources(corpus, position.conversation_id, ())
    for row in evidence.position_observations:
        _require(row.position_id in positions,
                 "Position observation is outside its registered census")
        conversation = positions[row.position_id].conversation_id
        _require(row.statement_sha256 == positions[row.position_id].statement_sha256,
                 "Position review differs from the frozen statement")
        _sources(corpus, conversation, row.sources)
        _unique(row.fact_ids, "position fact reference")
        _require(all(key in facts and facts[key].conversation_id == conversation
                     for key in row.fact_ids), "Position refers to an unknown or cross-scope fact")
        _review(row.review, registration, source_only=False)
        _require(row.review is None
                 or row.review.reviewed_at_ns >= evidence.position_census_frozen_at_ns,
                 "Output alignment must follow the collected position-census freeze")
        _require(row.review is not None or row.support == row.permission == "unknown",
                 "Position correctness cannot be labelled by its system decision")
        _require(not row.fact_ids or row.review is not None,
                 "Position-to-fact alignment requires independent blinded review")
        _require(row.support != "supported" or bool(row.sources),
                 "Supported positions require original source references")
    observations = {row.position_id: row for row in evidence.position_observations}
    for row in evidence.fact_observations:
        _require(row.fact_id in facts, "Fact observation is outside the expected inventory")
        _require(row.fact_content_sha256 == facts[row.fact_id].fact_content_sha256,
                 "Fact stage trace differs from the independently expected assertion")
        _unique(row.position_ids, "fact position reference")
        _require(all(key in observations and row.fact_id in observations[key].fact_ids
                     for key in row.position_ids),
                 "Fact trace lacks its reviewed position alignment")
        aligned = [position for position in observations.values()
                   if row.fact_id in position.fact_ids]
        _require(set(row.position_ids) == {position.position_id for position in aligned},
                 "Fact trace must retain every observed aligned position")
        if row.alignment_complete.state == "present":
            conversation = facts[row.fact_id].conversation_id
            _require(evidence.position_census_complete and all(
                position.position_id in observations
                and observations[position.position_id].review is not None
                for position in evidence.positions if position.conversation_id == conversation
            ), "Complete fact alignment requires the full independently reviewed position census")
        _require(row.extracted.state != "absent" or not aligned,
                 "An absent extraction contradicts an aligned observed position")
        _require(row.published.state != "absent"
                 or not any(position.published.state == "present" for position in aligned),
                 "An absent publication contradicts an aligned published Revision")
        if row.published.state == "present":
            _require(any(observations[key].published.state == "present"
                         for key in row.position_ids),
                     "Published fact lacks its aligned published Revision")
    for row in evidence.question_observations:
        _require(row.case_id in questions, "Question observation is outside the full denominator")
        _review(row.review, registration, source_only=False)
        _require(row.review is None
                 or row.review.reviewed_at_ns >= evidence.position_census_frozen_at_ns,
                 "Answer review must follow the collected-output census freeze")
        answer = questions[row.case_id]
        result_digest = (
            benchmark_digest(answer.result.model_dump(mode="json")) if answer.result else None
        )
        _require(row.answer_result_sha256 == result_digest,
                 "Answer review differs from the actual result bytes")
        _require(row.answer_assessment == "unknown" or (
            row.review is not None and answer.result is not None and answer.status == "answered"
        ), "A missing answer or unreviewed answer cannot supply a meaning label")
        expected = next(value for value in registration.questions if value.case_id == row.case_id)
        _unique(tuple(value.fact_id for value in row.facts), "question fact observation")
        _require(row.review is not None or (
            row.context_sufficient.state == "unknown"
            and all(value.retrieved.state == "unknown" for value in row.facts)
        ), "Fact representation and context sufficiency require independent blinded review")
        packed_ids: set[str] = (
            {hit.hit_id for hit in answer.packing.included} if answer.packing else set()
        )
        traces = {value.fact_id: value for value in evidence.fact_observations}
        for value in row.facts:
            _require(value.fact_id in expected.fact_ids,
                     "Retrieval observation is outside the registered question/fact pairs")
            _unique(value.revision_ids, "eligible Revision reference")
            _unique(value.context_hit_ids, "retrieved context reference")
            _require(set(value.context_hit_ids) <= packed_ids,
                     "Retrieved fact references a hit absent from the actual packed answer context")
            if value.eligible.state == "present":
                trace = traces.get(value.fact_id)
                published = {position.revision_id for position in observations.values()
                             if value.fact_id in position.fact_ids
                             and position.published.state == "present"}
                _require(trace is not None and trace.published.state == "present"
                         and set(value.revision_ids) <= published,
                         "Eligible fact lacks its aligned observed published Revision")
        returned = {value.fact_id for value in row.facts if value.retrieved.state == "present"}
        _require(row.context_sufficient.state != "present" or (
            answer.request is not None and set(expected.fact_ids) <= returned
        ), "Sufficient answer context requires each expected fact in actual delivered retrieval")
        _require(row.answer_assessment != "stale_current"
                 or expected.stale_current_eligible == "yes",
                 "Stale current labels require a proved current-correction expectation")
    return resources


def _costs(
    arm: AnswerComparisonArm, evidence: MechanismArmEvidence, expected_digest: str,
    resources: _ArmResources,
) -> dict[str, object]:
    resource = evidence.resource_registration
    resource_digest = integrity_digest(resource)
    _require(resource_digest == expected_digest and resource.run_key == arm.registration.run_key,
             "Resource join differs from its preregistered arm/run")
    calls = {row.call_key: row for row in resources.calls}
    _require(len(calls) == len(resources.calls), "Conflicting resource provider-call identity")
    _unique(tuple(row.call_key for row in evidence.answer_calls), "answer provider call")
    _require(not set(calls) & {row.call_key for row in evidence.answer_calls},
             "Answer and ingestion/retrieval accounting double-count the same actual call")
    answers = {row.case_id: row for row in arm.answers.cases}
    successes: set[str] = set()
    for row in evidence.answer_calls:
        _require(row.case_id in answers, "Answer cost names an unknown question")
        answer = answers[row.case_id]
        _require(answer.request is not None
                 and row.request_sha256 == answer.request.request_sha256,
                 "Answer cost differs from the actual request")
        if row.outcome == "succeeded":
            _require(answer.result is not None and row.case_id not in successes,
                     "Answer accounting repeats or invents a successful result")
            assert answer.result is not None
            _require(row.result_sha256 == benchmark_digest(answer.result.model_dump(mode="json")),
                     "Answer billing differs from the exact successful result")
            _require(answer.result.usage is None or row.usage == answer.result.usage,
                     "Answer usage differs from the retained provider response")
            successes.add(row.case_id)
    if evidence.answer_history_complete:
        _require(successes == {row.case_id for row in arm.answers.cases
                               if row.status == "answered"},
                 "Complete answer accounting omits an observed successful call")
        _require({row.case_id for row in evidence.answer_calls}
                 == {row.case_id for row in arm.answers.cases if row.request is not None},
                 "Complete answer accounting omits a failed or attempted request")
    costs = [row.cost_usd for row in calls.values()] + [
        row.cost_usd for row in evidence.answer_calls
    ]
    entered = [row.input_tokens for row in calls.values()] + [
        row.usage.input_tokens if row.usage else None for row in evidence.answer_calls
    ]
    emitted = [row.output_tokens for row in calls.values()] + [
        row.usage.output_tokens if row.usage else None for row in evidence.answer_calls
    ]
    complete = resources.history_complete and evidence.answer_history_complete
    # An unclassified call may belong to any phase, so no phase total is then complete.
    phased = resources.history_complete and all(row.phase is not None for row in calls.values())
    return {
        "resource_registration_sha256": resource_digest,
        "resource_evidence_file_sha256": resources.evidence_file_sha256,
        "resource_binding_witness_sha256": evidence.resource_binding_witness_sha256,
        "provider_history_scope": resources.scope,
        "history_complete": complete,
        "observed_unique_calls": len(costs),
        "unknown_cost_calls": sum(value is None for value in costs),
        "observed_known_cost_usd": math.fsum(value for value in costs if value is not None),
        "total_cost_usd": math.fsum(value for value in costs if value is not None)
        if complete and all(value is not None for value in costs) else None,
        "observed_known_input_tokens": sum(value for value in entered if value is not None),
        "observed_known_output_tokens": sum(value for value in emitted if value is not None),
        "input_tokens": sum(value for value in entered if value is not None)
        if complete and all(value is not None for value in entered) else None,
        "output_tokens": sum(value for value in emitted if value is not None)
        if complete and all(value is not None for value in emitted) else None,
        "answered_case_calls": len(successes),
        "answer_attempt_count": len(evidence.answer_calls),
        "provider_cost_by_phase": {
            phase: {
                "observed_unique_calls": len(selected),
                "observed_known_cost_usd": math.fsum(
                    row.cost_usd for row in selected if row.cost_usd is not None
                ),
                "total_cost_usd": math.fsum(
                    row.cost_usd for row in selected if row.cost_usd is not None
                ) if (resources.history_complete if phase == "unclassified" else phased)
                and all(row.cost_usd is not None for row in selected) else None,
            }
            for phase in (*get_args(ResourcePhase), "unclassified")
            for selected in [[row for row in calls.values()
                              if (row.phase or "unclassified") == phase]]
        },
        "answer_observed_known_cost_usd": math.fsum(
            row.cost_usd for row in evidence.answer_calls if row.cost_usd is not None
        ),
        "workload_completion": arm.completion,
    }


def _arm_score(
    registration: MechanismRegistration, arm: AnswerComparisonArm,
    evidence: MechanismArmEvidence, resource_digest: str, resources: _ArmResources,
) -> MechanismArmScore:
    positions = {row.position_id: row for row in evidence.position_observations}
    facts = {row.fact_id: row for row in evidence.fact_observations}
    questions = {row.case_id: row for row in evidence.question_observations}
    expected_facts = {row.fact_id: row for row in registration.facts}
    unsupported: list[str] = []
    refused: list[str] = []
    for identity in evidence.positions:
        position = positions.get(identity.position_id)
        if position is None:
            unsupported.append("unknown")
            refused.append("unknown")
            continue
        unsupported.append(
            "excluded" if position.published.state == "absent" else
            "unknown" if position.published.state == "unknown" or position.support == "unknown"
            else "event" if position.support == "unsupported" else "non_event"
        )
        eligible = _eligibility(position.support, position.permission)
        refused.append(eligible if eligible != "eligible" else
                       "event" if position.decision == "refused" else
                       "non_event" if position.decision == "accepted" else "unknown")
    missing: list[str] = []
    fact_stages: dict[str, Stage] = {}
    for expected in registration.facts:
        eligible = _eligibility(expected.support, expected.permission)
        row = facts.get(expected.fact_id)
        state = "unknown"
        fact_stage: Stage = "unknown"
        if eligible == "excluded":
            state, fact_stage = "excluded", "none"
        elif eligible == "eligible" and row is not None:
            if row.published.state == "present":
                state, fact_stage = "non_event", "none"
            elif (
                row.processing_complete.state == "present" and row.published.state == "absent"
                and row.alignment_complete.state == "present"
            ):
                state = "event"
                if row.extracted.state == "absent":
                    fact_stage = "extraction"
                elif row.extracted.state == "present":
                    aligned = [positions[key] for key in row.position_ids]
                    if all(value.decision == "refused" and value.support == "supported"
                           and value.permission == "permitted" for value in aligned):
                        fact_stage = "refusal"
                    elif all(value.decision == "accepted" for value in aligned):
                        fact_stage = "publication"
        missing.append(state)
        fact_stages[expected.fact_id] = fact_stage
    retrieval: list[str] = []
    retrieval_cases: list[FactRetrievalScore] = []
    stale: list[str] = []
    question_stages: list[Attribution] = []
    for expected in registration.questions:
        row = questions.get(expected.case_id)
        fact_rows = {value.fact_id: value for value in row.facts} if row else {}
        retrieval_states: list[str] = []
        for fact_id in expected.fact_ids:
            fact = expected_facts[fact_id]
            value = fact_rows.get(fact_id)
            eligibility = _eligibility(fact.support, fact.permission)
            state = eligibility if eligibility != "eligible" else (
                "unknown" if value is None or value.eligible.state == "unknown" else
                "excluded" if value.eligible.state == "absent" else
                "event" if value.retrieved.state == "absent" else
                "non_event" if value.retrieved.state == "present" else "unknown"
            )
            retrieval_states.append(state)
            retrieval_cases.append(FactRetrievalScore(
                expected.case_id, fact_id, state,
                value.eligible.state if value else "unknown",
                value.retrieved.state if value else "unknown",
            ))
        retrieval.extend(retrieval_states)
        stale.append(
            "excluded" if expected.stale_current_eligible == "no" else
            "unknown" if expected.stale_current_eligible == "unknown" or row is None
            or row.answer_assessment == "unknown" else
            "event" if row.answer_assessment == "stale_current" else "non_event"
        )
        related = [fact_stages[key] for key in expected.fact_ids]
        stage_order: tuple[Stage, ...] = ("extraction", "refusal", "publication")
        selected_stage: Stage | None = next((name for name in stage_order if name in related), None)
        stage: Stage = "none" if selected_stage is None else selected_stage
        if stage == "none":
            if "unknown" in related or row is None:
                stage = "unknown"
            elif "event" in retrieval_states:
                stage = "retrieval"
            elif row.context_sufficient.state == "present" and row.answer_assessment in (
                "incorrect", "stale_current",
            ):
                stage = "answering"
            elif row.answer_assessment != "correct":
                stage = "unknown"
        question_stages.append(Attribution(expected.case_id, stage))
    return MechanismArmScore(
        benchmark_digest(evidence.model_dump(mode="json")), len(evidence.positions),
        len(registration.facts), len(registration.questions), evidence.position_census_complete,
        _rate(unsupported), _rate(refused), _rate(missing), _rate(retrieval),
        "registered_case_fact_pairs_in_answer_packed_context", tuple(retrieval_cases), _rate(stale),
        tuple(Attribution(key, stage) for key, stage in fact_stages.items()),
        tuple(question_stages), _costs(arm, evidence, resource_digest, resources),
    )


def evaluate_mechanism_quality(
    corpus: BenchmarkCorpus, registration: MechanismRegistration,
    aml: AnswerComparisonArm, baseline: AnswerComparisonArm,
    aml_evidence: MechanismArmEvidence, baseline_evidence: MechanismArmEvidence,
) -> MechanismQualityReport:
    """Join reviewed full-denominator quality with existing paired answers and costs.

    Args:
        corpus: Complete corpus from the maintained source loader, never a mutated prefix.
        registration: Frozen source expectations, review provenance and arm conditions.
        aml: Existing QuoteMem answer/chain envelope.
        baseline: Existing separately registered BM25 or Mem0 answer envelope.
        aml_evidence: QuoteMem's complete declared position census, available observed rows
            and its runtime call census.
        baseline_evidence: Baseline census, available reviews and exact resource joins.

    Returns:
        Descriptive counts, unknown-label bounds, earliest proved attribution,
        equal-conversation lexical effects and observed costs. No superiority decision.

    Raises:
        QuoteMemError: Source, review, census, stage, answer or resource identities conflict.
        ValueError: Any supplied model has an invalid strict schema.
    """
    registration = MechanismRegistration.model_validate(registration.model_dump(mode="python"))
    aml_evidence = MechanismArmEvidence.model_validate(aml_evidence.model_dump(mode="python"))
    baseline_evidence = MechanismArmEvidence.model_validate(
        baseline_evidence.model_dump(mode="python"),
    )
    paired = compare_answers(corpus, registration.comparison, aml, baseline)
    _registration(corpus, registration)
    aml_resources = _validate_arm(corpus, registration, aml, aml_evidence)
    baseline_resources = _validate_arm(corpus, registration, baseline, baseline_evidence)
    conversations = tuple(paired.conversations.values())
    aml_score = _arm_score(
        registration, aml, aml_evidence, registration.aml_resource_registration_sha256,
        aml_resources,
    )
    baseline_score = _arm_score(
        registration, baseline, baseline_evidence,
        registration.baseline_resource_registration_sha256, baseline_resources,
    )
    aml_cost = aml_score.costs["total_cost_usd"]
    baseline_cost = baseline_score.costs["total_cost_usd"]
    return MechanismQualityReport(
        benchmark_digest(registration.model_dump(mode="json")), paired,
        math.fsum(row.mean_f1_delta for row in conversations) / len(conversations),
        math.fsum(row.exact_match_rate_delta for row in conversations) / len(conversations),
        len(conversations),
        registration.fact_inventory_complete, registration.held_out_status,
        aml_cost - baseline_cost if isinstance(aml_cost, float)
        and isinstance(baseline_cost, float) else None,
        aml_score, baseline_score,
        "inconclusive", (
            "descriptive_reviewed_labels_are_not_automatic_ground_truth",
            "witness_custody_and_reviewer_independence_require_external_verification",
            "unknown_labels_and_incomplete_censuses_are_not_negative_observations",
            "complete_registered_source_only_prefix_and_overlay_experiments_are_separate",
            "conversation_groups_do_not_establish_independent_population_sampling",
            "no_business_utility_threshold_or_gate_disabled_arm_is_inferred",
        ),
    )


def _load[T: BaseModel](model: type[T], path: Path) -> T:
    return model.model_validate_json(canonical_json(read_source_json(path)))


def main(argv: Sequence[str] | None = None) -> int:
    """Read exact existing artifacts and write one new descriptive quality report.

    Args:
        argv: Optional command arguments; otherwise use the process arguments.

    Returns:
        Zero for a valid descriptive report, including negative/incomplete results;
        one for invalid input or an existing output. Zero is not quality acceptance.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "registration", "quotemem-arm", "baseline-arm",
                 "quotemem-evidence", "baseline-evidence", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    options = parser.parse_args(argv)
    try:
        registration = _load(MechanismRegistration, options.registration)
        corpus = load_dataset(options.dataset, registration.comparison.source)
        report = evaluate_mechanism_quality(
            corpus, registration, _load(AnswerComparisonArm, options.quotemem_arm),
            _load(AnswerComparisonArm, options.baseline_arm),
            _load(MechanismArmEvidence, options.quotemem_evidence),
            _load(MechanismArmEvidence, options.baseline_evidence),
        )
        with options.output.open("x", encoding="utf-8") as output:
            output.write(canonical_json(asdict(report)) + "\n")
        return 0
    except (QuoteMemError, OSError, ValueError):
        print(json.dumps({"status": "failed", "code": "mechanism_quality_invalid"}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
