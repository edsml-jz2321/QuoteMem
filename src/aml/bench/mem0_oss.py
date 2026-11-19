"""Sequential replay producer for the pinned optional Mem0 OSS memory-system arm."""

from __future__ import annotations

import hashlib
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import ValidationError

from aml.bench.datasets import (
    BenchmarkCorpus,
    DatasetSource,
    decode_source_json,
    source_identity,
    source_identity_kind,
    source_matches_cases,
)
from aml.bench.mem0_binding import (
    BaselineArrival,
    Mem0InputGroup,
    Mem0Method,
    Mem0Registration,
    Mem0RetrievalBinding,
    Mem0SearchReceipt,
    mem0_digest,
    mem0_search_response,
)
from aml.bench.mem0_runtime import Mem0Ledger, Mem0Memory, Mem0ProviderCall
from aml.bench.providers import BenchmarkContext
from aml.bench.replay import ReplayConfig, ReplayGroup, prepare_replay, replay_add_input_sha256
from aml.core.errors import AMLError
from aml.core.identity import canonical_json

if TYPE_CHECKING:
    from aml.ingestion.boundary import AddRequestV1


def mem0_adapter_digest() -> str:
    """Bind the complete producer, binding and optional-runtime implementation bytes."""
    root = Path(__file__).parent
    return mem0_digest({
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in ("mem0_binding.py", "mem0_oss.py", "mem0_runtime.py")
    })


def mem0_input_group(group: ReplayGroup) -> Mem0InputGroup:
    """Reconstruct exact ordered source-unit metadata from authoritative replay inputs."""
    if not group.adds or not group.searches:
        raise AMLError("baseline_invalid", "Mem0 requires nonempty prepared Add and Search groups")
    users = {request.user_id for request in group.adds} | {
        request.user_id for _, request in group.searches
    }
    if len(users) != 1:
        raise AMLError("baseline_invalid", "Mem0 source group contains inconsistent user scopes")
    return Mem0InputGroup(
        cluster_id=group.cluster_id, user_id=group.adds[0].user_id,
        case_ids=tuple(case_id for case_id, _ in group.searches),
        add_request_ids=tuple(request.request_id for request in group.adds),
        add_input_sha256=replay_add_input_sha256(group),
        source_units_sha256=mem0_digest([
            {
                "request_id": request.request_id, "session_id": request.session_id,
                "message_index": index, "role": message.role, "content": message.content,
                "timestamp": message.timestamp,
                **({"speaker": message.speaker,
                    "source_participant": (
                        message.source_participant.model_dump(mode="json")
                        if message.source_participant is not None else None
                    )} if request.source_metadata_version is not None else {}),
            }
            for request in group.adds for index, message in enumerate(request.messages)
        ]),
    )


def register_mem0(
    corpus: BenchmarkCorpus, *, run_key: str, trial_key: str, config: ReplayConfig,
    arrival: BaselineArrival, method: Mem0Method,
) -> Mem0Registration:
    """Prepare complete source identities before any optional runtime or provider work.

    Args:
        corpus: Verified complete source, including an explicit selected conversation cohort.
        run_key: Fresh replay namespace.
        trial_key: Frozen answer-trial coordinate.
        config: Shared replay limits and isolation policy.
        arrival: `all_adds_then_search` completes all groups' Adds before any Search;
            `immediate_first_search` searches each group immediately after its own Adds.
        method: Explicit pinned build, configuration and bounded provider conditions.

    Returns:
        Mem0Registration: Serializable preflight inputs; saving and timing custody are caller-owned.

    Raises:
        AMLError: The source is not complete or is outside the supported single-file cohort.
        ValidationError: A configuration or prepared identity is malformed.
    """
    ids = [case.case_id for case in corpus.cases]
    if not isinstance(corpus.source, DatasetSource) or not source_matches_cases(corpus.source, ids):
        raise AMLError("baseline_invalid", "Mem0 requires the complete declared single-file source")
    groups = prepare_replay(
        corpus, run_id=run_key, top_k=config.top_k, max_top_k=config.max_top_k,
        isolate_queries=config.isolate_queries,
        source_metadata_version=config.source_metadata_version,
    )
    return Mem0Registration(
        schema_version=1, source=corpus.source, case_ids=tuple(ids), run_key=run_key,
        trial_key=trial_key, replay_config=config, arrival=arrival,
        groups=tuple(mem0_input_group(group) for group in groups), method=method,
    )


def validate_mem0_context(
    corpus: BenchmarkCorpus, context: BenchmarkContext, group: ReplayGroup,
) -> None:
    """Reject re-sealed context substitutions using the verified corpus and prepared inputs."""
    binding = Mem0RetrievalBinding.model_validate_json(context.retrieval_binding_json)
    registration = Mem0Registration.model_validate_json(binding.registration_json)
    matches = [item for item in registration.groups if context.case_id in item.case_ids]
    if len(matches) != 1:
        raise AMLError("binding_mismatch", "Mem0 context lacks a unique registered source group")
    group_index = registration.groups.index(matches[0])
    previous = registration.groups[:group_index]
    sequence = (
        sum(len(item.add_request_ids) + len(item.case_ids) for item in previous)
        + len(matches[0].add_request_ids)
        if registration.arrival == "immediate_first_search"
        else sum(len(item.add_request_ids) for item in registration.groups)
        + sum(len(item.case_ids) for item in previous)
    ) + matches[0].case_ids.index(context.case_id)
    expected_response = mem0_search_response(
        binding.raw_result_json, top_k=registration.replay_config.top_k,
    )
    if (
        registration.source != corpus.source
        or registration.case_ids != tuple(case.case_id for case in corpus.cases)
        or registration.run_key != context.run_key or registration.trial_key != context.trial_key
        or len(matches) != 1 or matches[0] != mem0_input_group(group)
        or binding.receipt.sequence != sequence
        or binding.receipt.case_id != context.case_id or binding.receipt.user_id != context.user_id
        or binding.receipt.request_sha256
        != hashlib.sha256(context.search_request_json.encode()).hexdigest()
        or canonical_json(expected_response.model_dump(mode="json", exclude_none=True))
        != context.search_response_json
    ):
        raise AMLError("binding_mismatch", "Mem0 context differs from its registered source replay")


@dataclass(frozen=True)
class Mem0OperationReceipt:
    """One local operation's result; failed and unstarted work stays visible."""

    sequence: int
    operation: Literal["add", "search"]
    identity: str
    user_id: str
    status: Literal["succeeded", "failed", "not_started"]
    elapsed_ms: float | None
    raw_result_json: str | None
    failure: str | None
    provider_call_keys: tuple[str, ...]


@dataclass(frozen=True)
class Mem0ReplayReport:
    """Complete local API denominator plus actual bounded HTTP accounting."""

    registration: Mem0Registration
    registration_sha256: str
    status: Literal["complete", "incomplete"]
    receipts: tuple[Mem0OperationReceipt, ...]
    contexts: tuple[BenchmarkContext, ...]
    provider_calls: tuple[Mem0ProviderCall, ...]
    http_delivery: Literal["not_executed"] = "not_executed"
    quality_status: Literal["unknown"] = "unknown"
    cost_status: Literal["unknown_without_billing_witness"] = "unknown_without_billing_witness"
    close_status: Literal["not_observed", "complete", "failed"] = "not_observed"


def mem0_add_metadata(request: AddRequestV1, source_revision: str) -> dict[str, object]:
    """Retain source metadata without rewriting text or claiming Mem0 uses AML identity rules."""
    return {
        "source_revision": source_revision, "request_id": request.request_id,
        "session_id": request.session_id,
        "message_timestamps": [message.timestamp for message in request.messages],
        "add_request_sha256": mem0_digest(request.model_dump(mode="json")),
        **({
            "source_metadata_version": request.source_metadata_version,
            "message_source_metadata": [
                {
                    "speaker": message.speaker,
                    "source_participant": (
                        message.source_participant.model_dump(mode="json")
                        if message.source_participant is not None else None
                    ),
                } for message in request.messages
            ],
        } if request.source_metadata_version is not None else {}),
    }


def replay_mem0(
    corpus: BenchmarkCorpus, registration: Mem0Registration,
    *, memory: Mem0Memory, ledger: Mem0Ledger,
) -> Mem0ReplayReport:
    """Execute one registered sequential local replay, preserving every source case.

    Args:
        corpus: Complete verified source.
        registration: Frozen run, method and exact prepared Add/Search identities.
        memory: Caller-owned pinned Memory instance; fixtures must be labeled separately.
        ledger: The runtime's actual HTTP observer and budget owner.

    Returns:
        Mem0ReplayReport: Every Add/Search disposition and successful bound contexts.
            A failed Add withholds that group's Search; missing contexts score zero later.

    Raises:
        AMLError: The registration, adapter source or observer does not match the run.
        ValidationError: A frozen input is malformed before any operation.
    """
    registration = Mem0Registration.model_validate(registration.model_dump(mode="python"))
    expected = register_mem0(
        corpus, run_key=registration.run_key, trial_key=registration.trial_key,
        config=registration.replay_config, arrival=registration.arrival, method=registration.method,
    )
    if (
        registration != expected or ledger.registration != registration
        or registration.method.adapter_source_sha256 != mem0_adapter_digest()
    ):
        raise AMLError("binding_mismatch", "Mem0 execution differs from its frozen registration")
    groups = prepare_replay(
        corpus, run_id=registration.run_key, top_k=registration.replay_config.top_k,
        max_top_k=registration.replay_config.max_top_k,
        isolate_queries=registration.replay_config.isolate_queries,
        source_metadata_version=registration.replay_config.source_metadata_version,
    )
    scheduled: list[tuple[int, Literal["add", "search"], int]] = []
    if registration.arrival == "immediate_first_search":
        for index, group in enumerate(groups):
            scheduled.extend((index, "add", offset) for offset in range(len(group.adds)))
            scheduled.extend((index, "search", offset) for offset in range(len(group.searches)))
    else:
        operations: tuple[Literal["add", "search"], ...] = ("add", "search")
        for operation in operations:
            for index, group in enumerate(groups):
                scheduled.extend((index, operation, offset) for offset in range(
                    len(group.adds) if operation == "add" else len(group.searches),
                ))
    failed_groups: set[int] = set()
    completed: dict[int, list[str]] = {index: [] for index in range(len(groups))}
    receipts: list[Mem0OperationReceipt] = []
    contexts: dict[str, BenchmarkContext] = {}
    cases = {case.case_id: case for case in corpus.cases}
    registration_json = canonical_json(registration.model_dump(mode="json"))
    registration_digest = mem0_digest(registration.model_dump(mode="json"))
    for sequence, (index, operation, offset) in enumerate(scheduled):
        group = groups[index]
        identity = (
            group.adds[offset].request_id if operation == "add" else group.searches[offset][0]
        )
        user_id = registration.groups[index].user_id
        if index in failed_groups or ledger.failure is not None:
            receipts.append(Mem0OperationReceipt(
                sequence, operation, identity, user_id, "not_started", None, None,
                "incomplete_source" if index in failed_groups else "provider_boundary_stopped", (),
            ))
            continue
        call_start = ledger.begin(operation, identity)
        start = time.monotonic()
        raw_json: str | None = None
        failure: str | None = None
        search = group.searches[offset][1] if operation == "search" else None
        try:
            ledger.check()
            if operation == "add":
                add = group.adds[offset]
                raw = memory.add(
                    [{"role": message.role, "content": message.content}
                     for message in add.messages],
                    user_id=add.user_id, metadata=mem0_add_metadata(add, corpus.source.revision),
                    infer=True,
                )
            else:
                assert search is not None
                raw = memory.search(
                    search.query, top_k=search.top_k, filters={"user_id": search.user_id},
                    threshold=0.1, rerank=False, explain=False,
                )
            raw_json = canonical_json(raw)
            ledger.check()
            if operation == "add":
                decoded = decode_source_json(raw_json.encode())
                if not isinstance(decoded, dict) or not isinstance(decoded.get("results"), list):
                    raise ValueError("Mem0 Add returned an invalid result")
                completed[index].append(identity)
            else:
                assert search is not None
                request_json = canonical_json(search.model_dump(mode="json", exclude_none=True))
                response = mem0_search_response(raw_json, top_k=search.top_k)
                receipt = Mem0SearchReceipt(
                    operation="search", sequence=sequence, case_id=identity, user_id=user_id,
                    request_sha256=hashlib.sha256(request_json.encode()).hexdigest(),
                    raw_result_sha256=hashlib.sha256(raw_json.encode()).hexdigest(),
                    completed_add_ids=tuple(completed[index]),
                    elapsed_ms=(time.monotonic() - start) * 1000,
                    provider_call_keys=tuple(
                        call.observation.call_key for call in ledger.calls[call_start:]
                    ),
                )
                binding = Mem0RetrievalBinding(
                    kind="mem0_oss_v2_1_0",
                    add_input_sha256=registration.groups[index].add_input_sha256,
                    replay_config_json=canonical_json(
                        registration.replay_config.model_dump(mode="json"),
                    ),
                    source_units_sha256=registration.groups[index].source_units_sha256,
                    registration_json=registration_json, registration_sha256=registration_digest,
                    raw_result_json=raw_json, receipt=receipt,
                )
                case = cases[identity]
                contexts[identity] = BenchmarkContext.model_validate({
                    "dataset": corpus.source.dataset, "source_revision": corpus.source.revision,
                    "source_identity_kind": source_identity_kind(corpus.source),
                    "source_identity": source_identity(corpus.source),
                    "source_selection_json": canonical_json(corpus.source.model_dump(mode="json")),
                    "retrieval_origin": "mem0_oss_v2_1_0",
                    "retrieval_binding_json": canonical_json(binding.model_dump(mode="json")),
                    "run_key": registration.run_key, "trial_key": registration.trial_key,
                    "case_id": identity, "query": case.query,
                    "query_source_date": case.query_source_date, "user_id": user_id,
                    "source_question_json": case.source_question_json,
                    "search_request_json": request_json,
                    "search_response_json": canonical_json(
                        response.model_dump(mode="json", exclude_none=True),
                    ),
                })
        except Exception as error:
            failure = ledger.failure or (
                "invalid_response" if isinstance(error, ValueError | TypeError | ValidationError)
                else "local_api_error"
            )
            if operation == "add":
                failed_groups.add(index)
        receipts.append(Mem0OperationReceipt(
            sequence, operation, identity, user_id, "failed" if failure else "succeeded",
            (time.monotonic() - start) * 1000, raw_json, failure,
            tuple(call.observation.call_key for call in ledger.calls[call_start:]),
        ))
    return Mem0ReplayReport(
        registration, registration_digest,
        "complete" if all(row.status == "succeeded" for row in receipts) else "incomplete",
        tuple(receipts),
        tuple(contexts[case.case_id] for case in corpus.cases if case.case_id in contexts),
        tuple(ledger.calls),
    )


def mem0_report_json(report: Mem0ReplayReport) -> str:
    """Serialize Pydantic bindings through their JSON owners and keep scalar accounting."""
    body = asdict(report)
    body["registration"] = report.registration.model_dump(mode="json")
    body["contexts"] = [context.model_dump(mode="json") for context in report.contexts]
    body["provider_calls"] = [
        {**asdict(call), "observation": call.observation.model_dump(mode="json")}
        for call in report.provider_calls
    ]
    return canonical_json(body)
