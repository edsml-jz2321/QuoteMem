"""Bind ordinary baseline artifacts without inventing GoldLoad or observer history.

These projections verify retained bytes and internal joins. Installation, native
execution and whole-process custody remain independently witnessed requirements.
"""

from __future__ import annotations

import hashlib
import math
from typing import Annotated, Literal, Self

from pydantic import ConfigDict, Field, TypeAdapter, model_validator

from quotemem.bench.answer_comparison import AnswerComparisonArm, AnswerComparisonRegistration
from quotemem.bench.datasets import (
    BenchmarkCorpus,
    DatasetSource,
    Digest,
    Nonempty,
    decode_source_json,
    source_identity,
    source_identity_kind,
    source_matches_cases,
)
from quotemem.bench.mem0_binding import Mem0Registration, Mem0RetrievalBinding
from quotemem.bench.mem0_oss import Mem0ReplayReport, mem0_report_json, register_mem0
from quotemem.bench.providers import BenchmarkContext, BenchmarkPolicy
from quotemem.bench.quality_evaluation import FrozenTuple
from quotemem.bench.replay import ReplayConfig, prepare_replay, validate_replay_contexts
from quotemem.bench.resources import ProviderCallObservation
from quotemem.bench.verbatim import METHOD_JSON, METHOD_SHA256
from quotemem.core.errors import QuoteMemError
from quotemem.core.identity import canonical_json, integrity_digest
from quotemem.store.metadata import StrictModel

_MAX_BYTES = 8 * 1024 * 1024
ArtifactText = Annotated[str, Field(min_length=1, max_length=_MAX_BYTES, repr=False)]
Commit = Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
Origin = Literal["verbatim_bm25_v1", "mem0_oss_v2_1_0"]


class _Record(StrictModel):
    model_config = ConfigDict(frozen=True, hide_input_in_errors=True)


class VerbatimExecutionRegistrationV1(_Record):
    """Preregister the existing local BM25 producer, without asserting execution."""

    schema_version: Literal[1]
    kind: Literal["verbatim_bm25_v1"]
    source_commit: Commit
    source: DatasetSource
    case_ids: FrozenTuple[Nonempty]
    run_key: Nonempty
    trial_key: Nonempty
    replay_config: ReplayConfig
    arrival: Literal["all_adds_then_search"]
    answer_policy: BenchmarkPolicy
    method_json: str
    method_sha256: Digest


def register_verbatim_execution(
    corpus: BenchmarkCorpus, *, run_key: str, trial_key: str, source_commit: str,
    config: ReplayConfig, arrival: Literal["all_adds_then_search", "immediate_first_search"],
    answer_policy: BenchmarkPolicy,
) -> VerbatimExecutionRegistrationV1:
    """Freeze BM25 inputs before producing retrieval, answers or the paired registration.

    Args:
        corpus: Complete original source from the maintained dataset loader.
        run_key: Fresh local producer namespace.
        trial_key: Answer-trial coordinate.
        source_commit: Exact producer checkout.
        config: Shared replay conditions.
        arrival: Only `all_adds_then_search` is supported by the ordinary full-source producer.
        answer_policy: Identical paired answer generation and packing conditions.

    Returns:
        The ordinary execution registration; it contains no GoldLoad population.

    Raises:
        QuoteMemError: Source is incomplete or arrival requires an unsupported incremental producer.
    """
    _require(isinstance(corpus.source, DatasetSource) and source_matches_cases(
        corpus.source, [case.case_id for case in corpus.cases],
    ), "BM25 registration requires the complete declared single-file source")
    assert isinstance(corpus.source, DatasetSource)
    if arrival != "all_adds_then_search":
        raise QuoteMemError("binding_mismatch", "Ordinary BM25 has no immediate-arrival receipt")
    return VerbatimExecutionRegistrationV1(
        schema_version=1, kind="verbatim_bm25_v1", source_commit=source_commit,
        source=corpus.source, case_ids=tuple(case.case_id for case in corpus.cases),
        run_key=run_key, trial_key=trial_key, replay_config=config, arrival=arrival,
        answer_policy=answer_policy,
        method_json=METHOD_JSON, method_sha256=METHOD_SHA256,
    )


class BaselineResourceRegistrationV1(_Record):
    """Prior resource join to an original ordinary baseline execution registration."""

    schema_version: Literal[1]
    kind: Origin
    run_key: Nonempty
    source_commit: Commit
    registered_at_ms: Annotated[int, Field(ge=0)]
    preregistration_witness_sha256: Digest
    execution_registration_json: ArtifactText
    execution_registration_sha256: Digest

    @model_validator(mode="after")
    def bounded_registration(self) -> Self:
        if len(self.execution_registration_json.encode("utf-8")) > _MAX_BYTES:
            raise ValueError("Baseline registration exceeds the original artifact byte bound")
        return self


def register_baseline_resources(
    execution: VerbatimExecutionRegistrationV1 | Mem0Registration,
    *, source_commit: str, registered_at_ms: int, preregistration_witness_sha256: str,
) -> BaselineResourceRegistrationV1:
    """Bind an existing baseline registration before its execution.

    Args:
        execution: BM25 execution inputs or the existing Mem0 registrar's output.
        source_commit: Exact QuoteMem producer/adapter checkout, separate from Mem0's upstream pin.
        registered_at_ms: Actual registration time supplied by the receipt owner.
        preregistration_witness_sha256: Retained independent timing/custody receipt.

    Returns:
        An ordinary resource registration. Timing and witness custody are not certified here.
    """
    return BaselineResourceRegistrationV1(
        schema_version=1,
        kind="mem0_oss_v2_1_0" if isinstance(execution, Mem0Registration) else "verbatim_bm25_v1",
        run_key=execution.run_key, source_commit=source_commit, registered_at_ms=registered_at_ms,
        preregistration_witness_sha256=preregistration_witness_sha256,
        execution_registration_json=canonical_json(execution),
        execution_registration_sha256=integrity_digest(execution),
    )


class BaselineResourceBindingV1(_Record):
    """Original producer report and standalone contexts, without invented observation times."""

    schema_version: Literal[1]
    report_file_sha256: Digest
    report_file_utf8: ArtifactText
    contexts_file_sha256: Digest
    contexts_file_utf8: ArtifactText

    @model_validator(mode="after")
    def exact_files(self) -> Self:
        for content, digest in ((self.report_file_utf8, self.report_file_sha256),
                                (self.contexts_file_utf8, self.contexts_file_sha256)):
            payload = content.encode("utf-8")
            if len(payload) > _MAX_BYTES or hashlib.sha256(payload).hexdigest() != digest:
                raise ValueError("Baseline resource input differs from bounded original bytes")
        return self


def _require(value: bool, message: str) -> None:
    if not value:
        raise QuoteMemError("binding_mismatch", message)


def _mem0_calls(
    report: Mem0ReplayReport, registration: Mem0Registration,
) -> tuple[ProviderCallObservation, ...]:
    """Validate the complete declared Add/Search census; do not infer process-wide capture."""
    per_group = [
        [("add", identity, group.user_id) for identity in group.add_request_ids]
        + [("search", identity, group.user_id) for identity in group.case_ids]
        for group in registration.groups
    ]
    scheduled = [row for group in per_group for row in group]
    if registration.arrival == "all_adds_then_search":
        scheduled.sort(key=lambda row: row[0] != "add")
    _require([(row.operation, row.identity, row.user_id) for row in report.receipts] == scheduled
             and [row.sequence for row in report.receipts] == list(range(len(scheduled))),
             "Mem0 receipt census differs from the registered complete schedule")
    calls = {row.observation.call_key: row for row in report.provider_calls}
    referenced = [key for row in report.receipts for key in row.provider_call_keys]
    _require(len(calls) == len(report.provider_calls) and len(set(referenced)) == len(referenced)
             and set(referenced) == set(calls)
             and sorted(row.sequence for row in calls.values()) == list(range(1, len(calls) + 1)),
             "Mem0 provider attempts are duplicated, missing or outside operation receipts")
    config = registration.method.configuration
    for purpose in ("extraction", "embedding"):
        endpoint = config.extraction if purpose == "extraction" else config.embedding
        _require(sum(row.purpose == purpose for row in calls.values()) <= endpoint.max_calls,
                 "Mem0 provider attempts exceed the registered call bound")
    contexts = {row.case_id: row for row in report.contexts}
    expected_contexts = {row.identity for row in report.receipts
                         if row.operation == "search" and row.status == "succeeded"}
    _require(set(contexts) == expected_contexts, "Mem0 Search result census differs from receipts")
    for receipt in report.receipts:
        started = receipt.status != "not_started"
        _require((receipt.elapsed_ms is not None) == started
                 and (receipt.elapsed_ms is None or (
                     math.isfinite(receipt.elapsed_ms) and receipt.elapsed_ms >= 0
                 ))
                 and (receipt.status == "succeeded") == (receipt.failure is None)
                 and (started or (
                     receipt.raw_result_json is None and not receipt.provider_call_keys
                 ))
                 and (receipt.status != "succeeded" or receipt.raw_result_json is not None),
                 "Mem0 operation disposition contradicts its observed fields")
        if receipt.operation == "search" and receipt.status == "succeeded":
            binding = Mem0RetrievalBinding.model_validate_json(
                contexts[receipt.identity].retrieval_binding_json,
            )
            _require(binding.raw_result_json == receipt.raw_result_json
                     and binding.registration_sha256 == report.registration_sha256
                     and binding.receipt.sequence == receipt.sequence
                     and binding.receipt.case_id == receipt.identity
                     and binding.receipt.user_id == receipt.user_id
                     and binding.receipt.provider_call_keys == receipt.provider_call_keys,
                     "Mem0 Search context differs from its actual operation receipt")
        for key in receipt.provider_call_keys:
            call = calls[key]
            endpoint = config.extraction if call.purpose == "extraction" else config.embedding
            observation = call.observation
            witness = {
                "purpose": call.purpose, "operation_id": call.operation_id,
                "sequence": call.sequence, "request_sha256": call.request_sha256,
                "response_sha256": call.response_sha256, "requested_model": call.requested_model,
                "resolved_model": call.resolved_model,
                "provider_request_id": call.provider_request_id,
                "status_code": call.status_code, "elapsed_ms": call.elapsed_ms,
                "failure": call.failure, "input_tokens": observation.input_tokens,
                "output_tokens": observation.output_tokens,
            }
            digest = integrity_digest(witness)
            TypeAdapter(Digest).validate_python(call.request_sha256)
            if call.response_sha256 is not None:
                TypeAdapter(Digest).validate_python(call.response_sha256)
            _require(call.operation_id == receipt.identity
                     and observation.phase == receipt.operation
                     and observation.call_key == f"mem0-http-{call.sequence:06d}"
                     and call.requested_model == endpoint.requested_model
                     and observation.provider_snapshot
                     == f"{endpoint.provider}:{endpoint.requested_model}"
                     and observation.cycle_key is None
                     and math.isfinite(call.elapsed_ms) and call.elapsed_ms >= 0
                     and observation.witness_sha256 == digest
                     and observation.usage_witness_sha256 == (
                         digest if observation.input_tokens is not None
                         or observation.output_tokens is not None else None
                     )
                     and observation.outcome == ("failed" if call.failure else "succeeded")
                     and observation.cost_usd is observation.cost_witness_sha256 is None,
                     "Mem0 provider cost projection differs from its actual scalar witness")
            _require(call.failure is not None or (
                call.status_code is not None and 200 <= call.status_code < 300
                and call.resolved_model == endpoint.expected_resolved_model
                and call.response_sha256 is not None and observation.input_tokens is not None
                and (call.purpose == "embedding" or observation.output_tokens is not None)
            ), "Successful Mem0 provider attempt contradicts its actual response fields")
    complete = all(row.status == "succeeded" for row in report.receipts)
    _require((report.status == "complete") == (complete and report.close_status != "failed"),
             "Mem0 replay completion contradicts its operation or close dispositions")
    return tuple(row.observation for row in report.provider_calls)


def validate_baseline_resources(
    corpus: BenchmarkCorpus, comparison: AnswerComparisonRegistration, arm: AnswerComparisonArm,
    resource: BaselineResourceRegistrationV1, binding: BaselineResourceBindingV1 | None,
) -> tuple[ProviderCallObservation, ...]:
    """Validate ordinary execution/report joins and return only actually recorded calls.

    Args:
        corpus: Original selected complete source from the maintained loader.
        comparison: Shared answer/replay conditions.
        arm: Actual baseline answer envelope; its execution evidence is the original report hash.
        resource: Prior ordinary resource registration, never a GoldLoad registration.
        binding: Original producer report and context files, or absent observations.

    Returns:
        Recorded Mem0 Add/Search provider calls, or an empty tuple when none are observed.
        This never establishes complete whole-process provider history or total billing.

    Raises:
        QuoteMemError: A source, method, report, receipt, answer or provider join disagrees.
        ValueError: Original bytes or strict typed fields are malformed.
    """
    raw = decode_source_json(resource.execution_registration_json.encode("utf-8"))
    _require(resource.kind == arm.registration.retrieval_origin
             and resource.run_key == arm.registration.run_key
             and resource.source_commit == arm.registration.source_commit
             and integrity_digest(raw) == resource.execution_registration_sha256
             == arm.registration.execution_registration_sha256,
             "Baseline resources differ from their original registered execution")
    if resource.kind == "mem0_oss_v2_1_0":
        execution = Mem0Registration.model_validate_json(canonical_json(raw))
        expected = register_mem0(
            corpus, run_key=resource.run_key, trial_key=arm.registration.trial_key,
            config=comparison.replay_config, arrival=comparison.arrival, method=execution.method,
        )
    else:
        execution = VerbatimExecutionRegistrationV1.model_validate_json(canonical_json(raw))
        expected = register_verbatim_execution(
            corpus, run_key=resource.run_key, trial_key=arm.registration.trial_key,
            source_commit=arm.registration.source_commit, config=comparison.replay_config,
            arrival=comparison.arrival, answer_policy=comparison.answer_policy,
        )
    _require(canonical_json(execution) == canonical_json(expected),
             "Baseline execution differs from the exact source, method or common conditions")
    if binding is None:
        return ()
    _require(binding.report_file_sha256 == arm.execution_evidence_sha256,
             "Baseline execution evidence is not the exact retained original report")
    raw_contexts = decode_source_json(binding.contexts_file_utf8.encode("utf-8"))
    contexts = TypeAdapter(tuple[BenchmarkContext, ...]).validate_json(canonical_json(raw_contexts))
    validate_replay_contexts(
        corpus, contexts, run_id=resource.run_key, config=comparison.replay_config,
    )
    _require(all(row.trial_key == arm.registration.trial_key
                 and row.retrieval_origin == resource.kind
                 and row.source_identity == source_identity(corpus.source)
                 and row.source_selection_json == canonical_json(corpus.source)
                 for row in contexts), "Baseline contexts differ from their arm")
    by_id = {row.case_id: row for row in contexts}
    _require(all(row.request is None or by_id.get(row.case_id) == row.request.context
                 for row in arm.answers.cases),
             "Answer context differs from original baseline output")
    raw_report = decode_source_json(binding.report_file_utf8.encode("utf-8"))
    if isinstance(execution, Mem0Registration):
        report = TypeAdapter(Mem0ReplayReport).validate_json(
            canonical_json(raw_report), strict=True,
        )
        _require(mem0_report_json(report) == canonical_json(raw_report)
                 and report.registration == execution
                 and report.registration_sha256 == resource.execution_registration_sha256
                 and report.contexts == contexts,
                 "Mem0 report differs from original registration, contexts or closed schema")
        return _mem0_calls(report, execution)
    groups = prepare_replay(
        corpus, run_id=resource.run_key, top_k=comparison.replay_config.top_k,
        max_top_k=comparison.replay_config.max_top_k,
        isolate_queries=comparison.replay_config.isolate_queries,
        source_metadata_version=comparison.replay_config.source_metadata_version,
    )
    expected_report = {
        "status": "local_retrieval_completed", "retrieval_origin": "verbatim_bm25_v1",
        "dataset_identity": source_identity(corpus.source),
        "dataset_identity_kind": source_identity_kind(corpus.source),
        "dataset_variant": corpus.source.variant, "run_id": resource.run_key,
        "isolate_queries": comparison.replay_config.isolate_queries,
        **({"source_metadata_version": comparison.replay_config.source_metadata_version}
           if comparison.replay_config.source_metadata_version is not None else {}),
        "group_count": len(groups), "case_count": len(corpus.cases),
        "method_json": METHOD_JSON, "method_sha256": METHOD_SHA256,
        "http_delivery": "not_executed", "real_model_quality": "unverified",
        "milestone_acceptance": "unverified",
    }
    _require(canonical_json(raw_report) == canonical_json(expected_report)
             and tuple(row.case_id for row in contexts)
             == tuple(case_id for group in groups for case_id, _ in group.searches),
             "BM25 report differs from the maintained method or complete source denominator")
    return ()
