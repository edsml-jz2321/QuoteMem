"""AMLB-compatible local replay through the authoritative HTTP models.

This client does not implement storage, extraction, finalization, or provider
reuse. HTTP observations alone cannot prove internal delivery or provider-call
counts. The service's real observers supply those measurements.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import logging
import math
import re
import time
from collections import deque
from collections.abc import Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import TYPE_CHECKING, Annotated, Literal, Self

import httpx
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, model_validator

from quotemem.adapters.http import (
    AddResponseV1,
    PublicErrorV1,
    SearchRequestV1,
    SearchResponseV1,
    parse_add_body,
)
from quotemem.adapters.http.errors import HTTPErrorCode
from quotemem.adapters.http.models import parse_search_response
from quotemem.bench.datasets import (
    BenchmarkCase,
    BenchmarkCorpus,
    BenchmarkMessage,
    BenchmarkSession,
    BenchmarkSource,
    source_identity,
    source_identity_kind,
    source_matches_cases,
)
from quotemem.bench.received_prefix import validate_received_prefix
from quotemem.core.errors import QuoteMemError
from quotemem.core.identity import canonical_json
from quotemem.core.source_participants import SourceMetadataVersion
from quotemem.ingestion.boundary import MAX_ADD_UTF8_BYTES, AddRequestV1

if TYPE_CHECKING:
    from quotemem.bench.providers import BenchmarkContext


_SOURCE: TypeAdapter[BenchmarkSource] = TypeAdapter(BenchmarkSource)
_LOGGER = logging.getLogger(__name__)
_RETRY_STATUSES = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 524})


def replay_retry_delay(
    operation: Literal["add", "search"], status_code: int | None, attempt: int,
    retry_after: str | None = None,
) -> float | None:
    """Return AMLB's delay after a failed HTTP attempt, or stop retrying.

    Args:
        operation: Add permits 32 total attempts; Search permits six and never
            retries 409. Other retryable statuses are 408/425/429/500/502/503/504/524.
        status_code: The received HTTP status; transport failures have no status.
        attempt: The one-based attempt that just completed.
        retry_after: A 429 response's Retry-After seconds or HTTP date. Missing or
            invalid values use 60 seconds; valid waits are capped at 60 seconds.

    Returns:
        float | None: Seconds before the next identical request, or None when the
            status is terminal or the attempt limit has been reached.
    """
    if (status_code not in _RETRY_STATUSES
            or attempt >= (32 if operation == "add" else 6)
            or (operation == "search" and status_code == 409)):
        return None
    if status_code == 429:
        try:
            delay = float(retry_after) if retry_after is not None else 60.0
        except ValueError:
            try:
                date = parsedate_to_datetime(retry_after or "")
                delay = (date - datetime.now(UTC)).total_seconds()
            except (TypeError, ValueError, OverflowError):
                return 60.0
        return min(60.0, max(0.0, delay)) if math.isfinite(delay) else 60.0
    delays = (10.0, 20.0, 40.0, 60.0) if operation == "add" else (1.5, 3.0, 6.0, 12.0, 20.0)
    return delays[min(attempt - 1, len(delays) - 1)]


class ReplayConfig(BaseModel):
    """Explicit immutable replay conditions, independent of imported context claims.

    Current replay producers send no options. Opening a non-null options branch
    requires matching producer and validation support rather than inferring it
    from a supplied result artifact.
    """

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, hide_input_in_errors=True)
    isolate_queries: bool
    top_k: Annotated[int, Field(ge=1, le=1000)]
    max_top_k: Annotated[int, Field(ge=1, le=1000)]
    options: None
    source_metadata_version: SourceMetadataVersion | None = Field(
        default=None, exclude_if=lambda value: value is None,
    )

    @model_validator(mode="after")
    def within_policy(self) -> Self:
        """Require the requested limit to fit the explicitly selected active policy."""
        if self.top_k > self.max_top_k:
            raise ValueError("Replay top_k exceeds the selected policy limit")
        return self


@dataclass(frozen=True)
class ReplayGroup:
    """One isolated source cluster, with all Add chunks and its reference queries."""

    cluster_id: str
    adds: tuple[AddRequestV1, ...]
    searches: tuple[tuple[str, SearchRequestV1], ...]


@dataclass(frozen=True)
class OperationReceipt:
    """Payload-free client observation; elapsed time includes HTTP attempts and backoff.

    ``error_code`` is the closed public code of an ``http_error`` whose body
    matches its status. ``attempts`` counts physical HTTP attempts for this
    logical request, including bounded AMLB platform retries.
    """

    operation: Literal["add", "search"]
    identity: str
    status_code: int | None
    elapsed_ms: float
    response_sha256: str | None
    result_ids: tuple[str, ...] = ()
    failure_code: Literal["http_error", "timeout", "transport_error", "invalid_response"] | None = (
        None
    )
    response_contract_version: Literal[1, 2] | None = None
    error_code: HTTPErrorCode | None = None
    attempts: int = 1


@dataclass(frozen=True)
class SearchResult:
    """Transient successful Search context, including the actual HTTP entity bytes."""

    case_id: str
    response: SearchResponseV1
    request_body: bytes = field(repr=False)
    response_body: bytes = field(repr=False)
    add_input_sha256: str


@dataclass(frozen=True)
class ReplayReport:
    """Observed transport results with no invented application or quality measurements."""

    dataset_sha256: str
    run_id: str
    arrival: Literal["all_adds_then_search", "immediate_first_search"]
    receipts: tuple[OperationReceipt, ...]
    results: tuple[SearchResult, ...]
    isolate_queries: bool = False
    add_concurrency: int = 1
    search_concurrency: int = 1
    dataset_identity_kind: str = "raw_file_sha256"
    dataset_variant: str | None = None


def _oversize(text: str) -> bool:
    return len(text.split()) > 2000 or len(text.encode()) > MAX_ADD_UTF8_BYTES


def _within_budget(sentence: str) -> Iterator[str]:
    """A sentence as it is, or, when it alone exceeds the Add chunk budget, its pieces.

    Taken unchanged from an earlier research replay. BEAM (code listings, logs) and
    LongMemEval-S (9 instances: one pasted user turn of 2,144-5,873 words) hold single
    "sentences" above 2,000 words or the wire byte cap, which the replay refused
    (``replay_input_oversize``). Such a sentence is cut at whitespace into the longest pieces
    within the budget, each keeping its leading whitespace, so the pieces concatenate to the
    sentence; a single token above the byte cap is cut by characters. A sentence within the
    budget is yielded unchanged, so every input the replay accepted before produces the same
    requests.
    """
    if not _oversize(sentence):
        yield sentence
        return
    current = ""
    for token in re.findall(r"\s*\S+|\s+$", sentence):
        while len(token.encode()) > MAX_ADD_UTF8_BYTES:
            if current:
                yield current
                current = ""
            cut = len(token)
            while len(token[:cut].encode()) > MAX_ADD_UTF8_BYTES:
                cut = cut * 3 // 4
            yield token[:cut]
            token = token[cut:]
        if current and _oversize(current + token):
            yield current
            current = token
        else:
            current += token
    if current:
        yield current


def _segments(message: BenchmarkMessage) -> Iterator[BenchmarkMessage]:
    if len(message.content.split()) <= 2000 and len(message.content.encode()) <= MAX_ADD_UTF8_BYTES:
        yield message
        return
    sentences = [
        piece
        for sentence in re.split(r"(?<=[.!?])(?=\s)", message.content)
        for piece in _within_budget(sentence)
    ]
    current = ""
    part = 0
    for sentence in sentences:
        combined = current + sentence
        if current and (
            len(combined.split()) > 2000 or len(combined.encode()) > MAX_ADD_UTF8_BYTES
        ):
            yield replace(message, message_id=f"{message.message_id}:part-{part}", content=current)
            part += 1
            current = sentence
        else:
            current = combined
    if current:
        yield replace(message, message_id=f"{message.message_id}:part-{part}", content=current)


def _chunks(session: BenchmarkSession) -> Iterator[tuple[BenchmarkMessage, ...]]:
    chunk: list[BenchmarkMessage] = []
    words = size = 0
    for source in session.messages:
        for message in _segments(source):
            message_words, message_size = (
                len(message.content.split()),
                len(message.content.encode()),
            )
            if chunk and (
                len(chunk) >= 20
                or words + message_words > 2000
                or size + message_size > MAX_ADD_UTF8_BYTES
            ):
                yield tuple(chunk)
                chunk, words, size = [], 0, 0
            chunk.append(message)
            words += message_words
            size += message_size
    if chunk:
        yield tuple(chunk)


def prepare_replay(
    corpus: BenchmarkCorpus,
    *,
    run_id: str,
    top_k: int = 10,
    max_top_k: int = 100,
    isolate_queries: bool = False,
    source_metadata_version: SourceMetadataVersion | None = None,
) -> tuple[ReplayGroup, ...]:
    """Build deterministic isolated Add/Search requests without adding reference answers.

    Args:
        corpus (BenchmarkCorpus): Verified source cases. Shared clusters ingest once.
        run_id (str): Explicit nonempty namespace. Use a new value for a fresh measured
            run; reuse only to deliberately exercise request idempotence.
        top_k (int): Requested result count within the service's active policy.
        max_top_k (int): Trusted active policy limit, validated by the HTTP authority.
        isolate_queries (bool): Give every query its own fresh user and replay all
            its source sessions. Required when each query is a first-Search observation.
        source_metadata_version: Explicit source metadata payload contract; absence
            preserves historical Add bytes. Version 1 carries verified loader metadata.

    Returns:
        tuple[ReplayGroup, ...]: Whole-message or sentence-boundary chunks of at most
            20 messages and 2000 whitespace words, respecting the absolute wire byte cap.

    Raises:
        QuoteMemError: For invalid requests, inconsistent shared context or empty namespaces. A
            source sentence above the chunk budget is cut losslessly (``_within_budget``).
    """
    return _prepare_cases(
        corpus.source,
        corpus.cases,
        run_id=run_id,
        top_k=top_k,
        max_top_k=max_top_k,
        isolate_queries=isolate_queries,
        source_metadata_version=source_metadata_version,
    )


def _prepare_cases(
    source: BenchmarkSource,
    selected_cases: tuple[BenchmarkCase, ...],
    *,
    run_id: str,
    top_k: int,
    max_top_k: int,
    isolate_queries: bool,
    source_metadata_version: SourceMetadataVersion | None = None,
) -> tuple[ReplayGroup, ...]:
    """Share request preparation while retaining the original complete source identity."""
    validate_received_prefix(source, selected_cases)
    try:
        ReplayConfig(
            isolate_queries=isolate_queries, top_k=top_k, max_top_k=max_top_k, options=None,
            source_metadata_version=source_metadata_version,
        )
    except ValidationError:
        raise QuoteMemError("replay_invalid", "Replay configuration is invalid") from None
    try:
        namespace_bytes = run_id.encode("utf-8")
    except UnicodeError:
        raise QuoteMemError("replay_invalid", "Replay namespace must contain valid UTF-8") from None
    if not namespace_bytes:
        raise QuoteMemError("replay_invalid", "Replay requires an explicit nonempty run namespace")
    grouped: dict[tuple[str, str | None], list[BenchmarkCase]] = {}
    for case in selected_cases:
        grouped.setdefault((case.cluster_id, case.case_id if isolate_queries else None), []).append(
            case
        )
    groups: list[ReplayGroup] = []
    for (cluster_id, isolated_case_id), cases in grouped.items():
        namespace = hashlib.sha256(
            canonical_json(
                [source_identity(source), run_id, cluster_id, isolated_case_id]
            ).encode()
        ).hexdigest()
        user_id = f"bench:{namespace}"
        sessions = cases[0].sessions
        if any(case.sessions != sessions for case in cases):
            raise QuoteMemError(
                "replay_invalid", "Shared cluster cases have inconsistent source context"
            )
        adds: list[AddRequestV1] = []
        for session in sessions:
            session_id = f"{namespace}:{session.session_id}"
            for index, chunk in enumerate(_chunks(session)):
                raw = {
                    "user_id": user_id,
                    "session_id": session_id,
                    "request_id": f"{session_id}:chunk-{index}",
                    **({"source_metadata_version": source_metadata_version}
                       if source_metadata_version is not None else {}),
                    "messages": [
                        {
                            "role": message.role,
                            "content": message.content,
                            **({"speaker": message.speaker}
                               if source_metadata_version is not None
                               and message.speaker is not None else {}),
                            **({"source_participant": message.source_participant.model_dump(
                                mode="json",
                            )} if source_metadata_version is not None
                               and message.source_participant is not None else {}),
                            **(
                                {"timestamp": message.timestamp_ms}
                                if message.timestamp_ms is not None
                                else {"timestamp": session.timestamp_ms}
                                if session.timestamp_ms is not None
                                else {}
                            ),
                        }
                        for message in chunk
                    ],
                }
                adds.append(parse_add_body(canonical_json(raw).encode("utf-8")))
        searches = tuple(
            (
                case.case_id,
                SearchRequestV1.from_body(
                    canonical_json(
                        {
                            "query": case.query,
                            "user_id": user_id,
                            "top_k": top_k,
                        }
                    ).encode("utf-8"),
                    max_top_k=max_top_k,
                ),
            )
            for case in cases
        )
        groups.append(ReplayGroup(cluster_id, tuple(adds), searches))
    return tuple(groups)


def replay_add_input_sha256(group: ReplayGroup) -> str:
    """Bind normalized Add inputs by request ID, separately from actual Search entity bytes."""
    checked = tuple(
        AddRequestV1.model_validate(request.model_dump(mode="python", warnings=False))
        for request in group.adds
    )
    return hashlib.sha256(
        canonical_json(
            [
                request.model_dump(mode="json")
                for request in sorted(checked, key=lambda request: request.request_id)
            ]
        ).encode()
    ).hexdigest()


def validate_replay_contexts(
    corpus: BenchmarkCorpus,
    contexts: tuple[BenchmarkContext, ...],
    *,
    run_id: str,
    config: ReplayConfig,
) -> None:
    """Check imported Add/user/Search bindings against explicitly selected source replay.

    Group the corpus once and prepare one group at a time. Shared clusters are
    prepared once, while isolated cases retain their complete source histories.
    Missing contexts remain absent; no response is manufactured. These checks
    establish source/configuration consistency, never actual HTTP execution,
    storage visibility, freshness of an existing workspace, or retrieval quality.
    """
    checked = ReplayConfig.model_validate(config.model_dump(mode="python", warnings=False))
    by_id = {context.case_id: context for context in contexts}
    source_ids = {case.case_id for case in corpus.cases}
    if (
        not run_id
        or len(source_ids) != len(corpus.cases)
        or len(by_id) != len(contexts)
        or not set(by_id) <= source_ids
    ):
        raise QuoteMemError(
            "binding_mismatch", "Replay contexts contain unknown or repeated source cases"
        )
    grouped: dict[tuple[str, str | None], list[BenchmarkCase]] = {}
    for case in corpus.cases:
        grouped.setdefault(
            (case.cluster_id, case.case_id if checked.isolate_queries else None), []
        ).append(case)
    try:
        for cases in grouped.values():
            present = [by_id[case.case_id] for case in cases if case.case_id in by_id]
            if not present:
                continue
            groups = prepare_replay(
                BenchmarkCorpus(corpus.source, tuple(cases)),
                run_id=run_id,
                top_k=checked.top_k,
                max_top_k=checked.max_top_k,
                isolate_queries=checked.isolate_queries,
                source_metadata_version=checked.source_metadata_version,
            )
            if len(groups) != 1:
                raise ValueError(
                    "Prepared replay grouping differs from the registered configuration"
                )
            group = groups[0]
            add_digest = replay_add_input_sha256(group)
            expected = dict(group.searches)
            baseline_binding: tuple[str, str] | None = None
            if any(context.retrieval_origin == "verbatim_bm25_v1" for context in present):
                # Verbatim uses ReplayGroup; resolve its input authority after module loading.
                from quotemem.bench.verbatim import verbatim_input_binding

                baseline_binding = verbatim_input_binding(group)
            for context in present:
                if context.retrieval_origin == "mem0_oss_v2_1_0":
                    from quotemem.bench.mem0_oss import validate_mem0_context

                    validate_mem0_context(corpus, context, group)
                binding = json.loads(context.retrieval_binding_json)
                declared = ReplayConfig.model_validate_json(binding["replay_config_json"])
                request = SearchRequestV1.from_body(
                    context.search_request_json.encode("utf-8"), max_top_k=checked.max_top_k
                )
                if (
                    context.run_key != run_id
                    or declared != checked
                    or binding["add_input_sha256"] != add_digest
                    or context.user_id != expected[context.case_id].user_id
                    or context.query != expected[context.case_id].query
                    or request != expected[context.case_id]
                    or (
                        context.retrieval_origin == "verbatim_bm25_v1"
                        and (binding["source_units_sha256"], binding["method_json"])
                        != baseline_binding
                    )
                ):
                    raise ValueError(
                        "Imported context differs from its exact prepared replay inputs"
                    )
    except (KeyError, TypeError, ValueError, ValidationError):
        raise QuoteMemError(
            "binding_mismatch", "Imported context differs from the selected Add/user/Search replay"
        ) from None


@dataclass(frozen=True)
class _Job:
    order: tuple[int, int, int]
    group: int
    operation: Literal["add", "search"]
    identity: str
    request: AddRequestV1 | SearchRequestV1
    add_input_sha256: str


@dataclass(frozen=True)
class _Outcome:
    receipt: OperationReceipt
    result: SearchResult | None


@dataclass(frozen=True)
class _GroupJobs:
    adds: tuple[_Job, ...]
    searches: tuple[_Job, ...]


class _DispatchQueue:
    """Keep each session's source chunks serial, preferring logical input order."""

    def __init__(self, jobs: tuple[_Job, ...]) -> None:
        self._streams: dict[str, deque[_Job]] = {}
        for job in jobs:
            self._streams.setdefault(self._key(job), deque()).append(job)
        self._ready = [(stream[0].order, key) for key, stream in self._streams.items()]
        heapq.heapify(self._ready)

    @staticmethod
    def _key(job: _Job) -> str:
        return job.request.session_id if isinstance(job.request, AddRequestV1) else job.identity

    def take(self) -> _Job | None:
        if not self._ready:
            return None
        _, key = heapq.heappop(self._ready)
        return self._streams[key].popleft()

    def complete(self, job: _Job) -> None:
        key = self._key(job)
        stream = self._streams[key]
        if stream:
            heapq.heappush(self._ready, (stream[0].order, key))


@dataclass
class _GroupCursor:
    jobs: _GroupJobs
    adds: _DispatchQueue = field(init=False)
    completed_adds: int = 0
    next_search: int = 0
    search_pending: bool = False
    failed: bool = False
    submitted: int = 0

    def __post_init__(self) -> None:
        self.adds = _DispatchQueue(self.jobs.adds)


def _public_code(response: httpx.Response) -> HTTPErrorCode | None:
    try:
        return PublicErrorV1.from_response(response.status_code, response.content).code
    except QuoteMemError:
        return None


def _send(client: httpx.Client, job: _Job) -> _Outcome:
    start = time.perf_counter()
    response: httpx.Response | None = None
    result: SearchResult | None = None
    result_ids: tuple[str, ...] = ()
    failure: Literal["http_error", "timeout", "transport_error", "invalid_response"] | None = None
    error_code: HTTPErrorCode | None = None
    attempts = 1
    try:
        request = client.build_request(
            "POST", f"/{job.operation}", json=job.request.model_dump(mode="json", exclude_none=True)
        )
        for attempts in range(1, 33 if job.operation == "add" else 7):
            response = None
            response = client.send(request)
            delay = replay_retry_delay(
                job.operation, response.status_code, attempts, response.headers.get("retry-after")
            )
            if delay is None:
                break
            _LOGGER.warning(
                "Replay retry operation=%s identity=%s status=%s attempt=%s delay_seconds=%s",
                job.operation, job.identity, response.status_code, attempts, delay,
            )
            response.close()
            time.sleep(delay)
        assert response is not None
        if response.status_code != 200:
            failure = "http_error"
            error_code = _public_code(response)
        elif isinstance(job.request, AddRequestV1):
            if not AddResponseV1.from_body(response.content).acknowledges(job.request):
                failure = "invalid_response"
        else:
            body = parse_search_response(response.content)
            if len(body.data) > job.request.top_k:
                failure = "invalid_response"
            else:
                result_ids = tuple(hit.id for hit in body.data)
                result = SearchResult(
                    job.identity, body, request.content, response.content, job.add_input_sha256
                )
    except httpx.TimeoutException:
        failure = "timeout"
    except httpx.DecodingError:
        failure = "invalid_response"
    except httpx.RequestError:
        failure = "transport_error"
    except (ValidationError, QuoteMemError):
        failure = "invalid_response"
    return _Outcome(
        OperationReceipt(
            job.operation,
            job.identity,
            response.status_code if response is not None else None,
            (time.perf_counter() - start) * 1000,
            hashlib.sha256(response.content).hexdigest() if response is not None else None,
            result_ids,
            failure,
            (2 if response.headers.get("x-aml-request-contract") == "2" else
             1 if response.headers.get("x-aml-request-contract") == "1" else None)
            if response is not None else None,
            error_code,
            attempts,
        ),
        result,
    )


def _jobs(groups: tuple[ReplayGroup, ...], *, immediate: bool) -> tuple[_GroupJobs, ...]:
    result: list[_GroupJobs] = []
    for group_index, group in enumerate(groups):
        add_input_sha256 = replay_add_input_sha256(group)
        add_prefix: tuple[int, int] = (group_index, 0) if immediate else (0, group_index)
        search_prefix: tuple[int, int] = (group_index, 1) if immediate else (1, group_index)
        result.append(
            _GroupJobs(
                tuple(
                    _Job(
                        (add_prefix[0], add_prefix[1], index),
                        group_index,
                        "add",
                        request.request_id,
                        request,
                        add_input_sha256,
                    )
                    for index, request in enumerate(group.adds)
                ),
                tuple(
                    _Job(
                        (search_prefix[0], search_prefix[1], index),
                        group_index,
                        "search",
                        identity,
                        request,
                        add_input_sha256,
                    )
                    for index, (identity, request) in enumerate(group.searches)
                ),
            )
        )
    return tuple(result)


def _bounded_phase(
    jobs: Iterator[_Job],
    pool: ThreadPoolExecutor,
    client: httpx.Client,
    *,
    limit: int,
    stop_on_failure: bool,
) -> dict[tuple[int, int, int], _Outcome]:
    pending: dict[Future[_Outcome], _Job] = {}
    outcomes: dict[tuple[int, int, int], _Outcome] = {}
    queue = _DispatchQueue(tuple(jobs))
    failed = False
    while True:
        while not failed and len(pending) < limit:
            job = queue.take()
            if job is None:
                break
            pending[pool.submit(_send, client, job)] = job
        if not pending:
            break
        done, _ = wait(pending, return_when=FIRST_COMPLETED)
        # Consume the entire completed batch before deciding whether to replenish.
        for future in done:
            job = pending.pop(future)
            outcome = future.result()
            outcomes[job.order] = outcome
            queue.complete(job)
            failed |= stop_on_failure and outcome.receipt.failure_code is not None
    return outcomes


def _immediate_groups(
    groups: tuple[_GroupJobs, ...],
    add_pool: ThreadPoolExecutor,
    search_pool: ThreadPoolExecutor,
    client: httpx.Client,
    *,
    add_limit: int,
    search_limit: int,
) -> dict[tuple[int, int, int], _Outcome]:
    # Each admitted group reserves one Search-worker slot until all its queries
    # finish. Its first Search therefore never queues behind unrelated searches.
    active: dict[int, _GroupCursor] = {}
    pending: dict[Future[_Outcome], _Job] = {}
    outcomes: dict[tuple[int, int, int], _Outcome] = {}
    next_group = 0
    while True:
        while next_group < len(groups) and len(active) < search_limit:
            active[next_group] = _GroupCursor(groups[next_group])
            next_group += 1
        # Dispatch ready Searches before admitting any further Add requests.
        for cursor in active.values():
            if (
                not cursor.failed
                and cursor.completed_adds == len(cursor.jobs.adds)
                and not cursor.search_pending
            ):
                job = cursor.jobs.searches[cursor.next_search]
                cursor.next_search += 1
                cursor.search_pending = True
                cursor.submitted += 1
                pending[search_pool.submit(_send, client, job)] = job
        add_pending = sum(job.operation == "add" for job in pending.values())
        for cursor in active.values():
            while not cursor.failed and add_pending < add_limit:
                job = cursor.adds.take()
                if job is None:
                    break
                cursor.submitted += 1
                pending[add_pool.submit(_send, client, job)] = job
                add_pending += 1
        if not pending:
            break
        done, _ = wait(pending, return_when=FIRST_COMPLETED)
        for future in done:
            job = pending.pop(future)
            outcome = future.result()
            outcomes[job.order] = outcome
            cursor = active[job.group]
            cursor.submitted -= 1
            if job.operation == "add":
                cursor.adds.complete(job)
                cursor.completed_adds += 1
                cursor.failed |= outcome.receipt.failure_code is not None
            else:
                cursor.search_pending = False
            # A failed cohort withholds its own remaining Adds and its first Search
            # because only its own source history is incomplete; isolated cohorts share
            # no state, so the rest keep their planned dispatch. Retiring it once its
            # already submitted requests return releases its reserved Search slot.
            if cursor.submitted == 0 and (
                cursor.failed or cursor.next_search == len(cursor.jobs.searches)
            ):
                del active[job.group]
    return outcomes


def replay(
    corpus: BenchmarkCorpus,
    client: httpx.Client,
    *,
    run_id: str,
    arrival: Literal["all_adds_then_search", "immediate_first_search"] = "all_adds_then_search",
    top_k: int = 10,
    max_top_k: int = 100,
    isolate_queries: bool = False,
    source_metadata_version: SourceMetadataVersion | None = None,
    add_concurrency: int = 1,
    search_concurrency: int = 1,
) -> ReplayReport:
    """Replay bounded concurrent HTTP requests with AMLB's default retry policy.

    Args:
        corpus (BenchmarkCorpus): Complete verified source corpus; partial cohorts are rejected.
        client (httpx.Client): Caller-owned thread-safe client with an explicit base URL,
            credentials, finite timeout, and enough connection capacity for configured
            parallelism. A transport double proves only the client contract.
        run_id (str): Namespace documented by `prepare_replay`.
        arrival (Literal["all_adds_then_search", "immediate_first_search"]): Bulk mode
            requires every Add to succeed before any Search. Immediate mode reserves
            a Search slot per active group and dispatches its first Search immediately
            after all its Adds succeed; same-user follow-up queries remain sequential.
        top_k (int): Requested result count.
        max_top_k (int): Trusted active recall policy limit.
        isolate_queries (bool): Use a fresh user per query as documented by `prepare_replay`.
        source_metadata_version: Explicit Add metadata contract passed to `prepare_replay`.
        add_concurrency (int): Maximum submitted unfinished Adds, from 1 through 64.
            This workload keeps one session's source chunks serial; distinct sessions
            may overlap. That choice does not imply a server ordering guarantee.
        search_concurrency (int): Maximum submitted unfinished Searches, from 1 through 256.
            In immediate mode this also bounds active groups. Both defaults preserve
            sequential execution. Configured limits are not measured service concurrency.

    Returns:
        ReplayReport: HTTP observations in stable logical input order, irrespective of
            completion order. Adds retry the platform's transient HTTP statuses up to
            32 attempts, with 10/20/40/60-second waits. Searches permit six attempts,
            with 1.5/3/6/12/20-second waits, and do not retry 409. A 429 uses Retry-After
            up to 60 seconds, defaulting to 60. Every retry is logged and resends the
            same bytes and identity. An exhausted Add failure stops further dispatch in bulk
            mode and only the failing group's own dispatch in immediate mode; every
            already submitted request is drained. Search failures remain in the
            denominator. No service-time, queue-wait, provider, or milestone
            measurement is inferred.

    Raises:
        QuoteMemError: If the source cohort, preparation, arrival mode, or concurrency is invalid.
    """
    _validate_dispatch(arrival, add_concurrency, search_concurrency)
    corpus = _complete_corpus(corpus)
    groups = prepare_replay(
        corpus,
        run_id=run_id,
        top_k=top_k,
        max_top_k=max_top_k,
        isolate_queries=isolate_queries,
        source_metadata_version=source_metadata_version,
    )
    return _dispatch_replay(
        corpus.source,
        groups,
        client,
        run_id=run_id,
        arrival=arrival,
        isolate_queries=isolate_queries,
        add_concurrency=add_concurrency,
        search_concurrency=search_concurrency,
    )


def _validate_dispatch(arrival: str, add_concurrency: int, search_concurrency: int) -> None:
    if arrival not in ("all_adds_then_search", "immediate_first_search"):
        raise QuoteMemError("replay_invalid", "Replay arrival mode is unsupported")
    if (
        type(add_concurrency) is not int
        or not 1 <= add_concurrency <= 64
        or type(search_concurrency) is not int
        or not 1 <= search_concurrency <= 256
    ):
        raise QuoteMemError("replay_invalid", "Replay concurrency limits are invalid")


def _complete_corpus(corpus: BenchmarkCorpus) -> BenchmarkCorpus:
    """Apply the same complete-source admission before either public execution path."""
    try:
        source = _SOURCE.validate_python(corpus.source.model_dump(mode="python", warnings=False))
    except ValidationError:
        raise QuoteMemError("replay_invalid", "Replay source descriptor is invalid") from None
    if (
        not corpus.cases
        or not source_matches_cases(source, [case.case_id for case in corpus.cases])
        or any(
            type(case.case_id) is not str
            or not case.case_id
            or type(case.query) is not str
            or not case.query
            for case in corpus.cases
        )
        or len({case.case_id for case in corpus.cases}) != len(corpus.cases)
    ):
        raise QuoteMemError("replay_invalid", "Replay requires the complete declared source cohort")
    return replace(corpus, source=source)


def _dispatch_replay(
    source: BenchmarkSource,
    prepared: tuple[ReplayGroup, ...],
    client: httpx.Client,
    *,
    run_id: str,
    arrival: Literal["all_adds_then_search", "immediate_first_search"],
    isolate_queries: bool,
    add_concurrency: int,
    search_concurrency: int,
) -> ReplayReport:
    """Run the one shared scheduler over fully admitted, completely prepared inputs."""
    groups = _jobs(prepared, immediate=arrival == "immediate_first_search")
    with (
        ThreadPoolExecutor(max_workers=add_concurrency) as add_pool,
        ThreadPoolExecutor(max_workers=search_concurrency) as search_pool,
    ):
        if arrival == "immediate_first_search":
            outcomes = _immediate_groups(
                groups,
                add_pool,
                search_pool,
                client,
                add_limit=add_concurrency,
                search_limit=search_concurrency,
            )
        else:
            outcomes = _bounded_phase(
                (job for group in groups for job in group.adds),
                add_pool,
                client,
                limit=add_concurrency,
                stop_on_failure=True,
            )
            if all(outcome.receipt.failure_code is None for outcome in outcomes.values()):
                outcomes.update(
                    _bounded_phase(
                        (job for group in groups for job in group.searches),
                        search_pool,
                        client,
                        limit=search_concurrency,
                        stop_on_failure=False,
                    )
                )
    ordered = [outcomes[order] for order in sorted(outcomes)]
    return ReplayReport(
        source_identity(source),
        run_id,
        arrival,
        tuple(outcome.receipt for outcome in ordered),
        tuple(outcome.result for outcome in ordered if outcome.result is not None),
        isolate_queries,
        add_concurrency,
        search_concurrency,
        source_identity_kind(source),
        source.variant,
    )
