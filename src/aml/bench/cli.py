"""Explicit local benchmark preparation, HTTP replay, and diagnostic scoring."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
from collections import Counter
from contextlib import ExitStack, nullcontext
from dataclasses import asdict, replace
from pathlib import Path
from threading import Lock, local
from time import perf_counter_ns
from typing import Protocol
from urllib.parse import urlsplit

import httpx
from pydantic import TypeAdapter, ValidationError

from aml.adapters.http import SearchRequestV1
from aml.bench.acquire import download_dataset
from aml.bench.answer_comparison import (
    AnswerComparisonArm,
    AnswerComparisonRegistration,
    compare_answers,
)
from aml.bench.answering import AnswerRunReport, generate_answers, score_beam_answers
from aml.bench.call_census import CensusClient, ProviderCallCensus
from aml.bench.datasets import (
    BenchmarkCorpus,
    BenchmarkSource,
    DatasetSource,
    decode_source_json,
    load_dataset,
    read_source_json,
    select_conversations,
    source_identity,
    source_identity_kind,
)
from aml.bench.evaluation import Prediction, score_mab_predictions, score_predictions
from aml.bench.mem0_binding import Mem0Registration
from aml.bench.mem0_oss import mem0_adapter_digest, mem0_report_json, register_mem0, replay_mem0
from aml.bench.mem0_runtime import open_mem0
from aml.bench.prefix_answers import (
    CheckpointExpectedOutcomes,
    PrefixAnswerComparisonRegistration,
    compare_prefix_answers,
    score_checkpoint_predictions,
    validate_checkpoint_outcomes,
)
from aml.bench.providers import (
    AnswerProvider,
    AnswerRequest,
    AnswerResult,
    BenchmarkContext,
    BenchmarkPolicy,
    FrozenBenchmarkProviders,
    RemoteBenchmarkProviders,
    RubricJudge,
    RubricRequest,
    RubricResult,
    benchmark_digest,
)
from aml.bench.received_prefix import (
    ReceivedPrefixRegistration,
    apply_received_prefix,
    register_received_prefix,
)
from aml.bench.replay import ReplayConfig, ReplayReport, replay
from aml.bench.verbatim import retrieve_verbatim
from aml.core.errors import AMLError
from aml.core.identity import canonical_json
from aml.core.request_budget import PLATFORM_REQUEST_TIMEOUT_SECONDS
from aml.providers.adapters import JsonTransport, ProviderError
from aml.providers.contracts import ProviderName

DASHSCOPE_COMPATIBLE_ROOT = "https://dashscope.aliyuncs.com/compatible-mode/v1"


class _BenchmarkProviders(AnswerProvider, RubricJudge, Protocol):
    """Both benchmark call boundaries, as supplied by one bound provider object."""


def _replay_contexts(
    corpus: BenchmarkCorpus, report: ReplayReport, *, trial_key: str, config: ReplayConfig
) -> tuple[BenchmarkContext, ...]:
    cases = {case.case_id: case for case in corpus.cases}
    contexts: list[BenchmarkContext] = []
    for result in report.results:
        case = cases[result.case_id]
        request = SearchRequestV1.from_body(result.request_body, max_top_k=config.max_top_k)
        contexts.append(
            BenchmarkContext.model_validate(
                {
                    "dataset": corpus.source.dataset,
                    "source_revision": corpus.source.revision,
                    "source_identity_kind": source_identity_kind(corpus.source),
                    "source_identity": source_identity(corpus.source),
                    "source_selection_json": canonical_json(corpus.source.model_dump(mode="json")),
                    "retrieval_origin": "aml_http",
                    "retrieval_binding_json": canonical_json(
                        {
                            "kind": "aml_http",
                            "add_input_sha256": result.add_input_sha256,
                            "replay_config_json": canonical_json(config.model_dump(mode="json")),
                        }
                    ),
                    "run_key": report.run_id,
                    "trial_key": trial_key,
                    "case_id": result.case_id,
                    "query": case.query,
                    "query_source_date": case.query_source_date,
                    "user_id": request.user_id,
                    "source_question_json": case.source_question_json,
                    "search_request_json": result.request_body.decode("utf-8"),
                    "search_response_json": result.response_body.decode("utf-8"),
                }
            )
        )
    return tuple(contexts)


def _answer_policy(corpus: BenchmarkCorpus, options: argparse.Namespace) -> BenchmarkPolicy:
    """Return the frozen answer policy of a report that names its output and rubric arm."""
    if options.output is None:
        raise AMLError("evaluation_invalid", "Answer reports require an explicit output file")
    if (corpus.source.dataset == "beam") != (options.rubric_policy is not None):
        raise AMLError("evaluation_invalid", "A rubric policy is required exactly for BEAM")
    return BenchmarkPolicy.model_validate(read_source_json(options.answer_policy))


def _score_answers(
    corpus: BenchmarkCorpus,
    options: argparse.Namespace,
    *,
    policy: BenchmarkPolicy,
    providers: _BenchmarkProviders,
) -> tuple[AnswerRunReport, object, bool]:
    """Generate answers from one bound provider and score them with the local evaluators.

    The evaluators, the complete case denominator, and every supplied artifact are
    identical for frozen fixtures and for an opted-in live provider; only the bound
    provider object differs.
    """
    expected = _checkpoint_outcomes(corpus, options)
    contexts = TypeAdapter(list[BenchmarkContext]).validate_python(
        read_source_json(options.contexts)
    )
    replay_config = ReplayConfig.model_validate(read_source_json(options.replay_config))
    answers = generate_answers(
        corpus,
        tuple(contexts),
        run_key=options.run_key,
        trial_key=options.trial_key,
        policy=policy,
        replay_config=replay_config,
        provider=providers,
        answer_workers=getattr(options, "answer_workers", 1),
    )
    failed = answers.status != "complete"
    if corpus.source.dataset == "beam":
        rubric = score_beam_answers(
            corpus,
            answers,
            policy=BenchmarkPolicy.model_validate(read_source_json(options.rubric_policy)),
            judge=providers,
        )
        return answers, rubric.model_dump(mode="json"), failed or rubric.status != "complete"
    predictions = [
        Prediction(case_id=row.case_id, answer=row.result.output.answer)
        for row in answers.cases
        if row.status == "answered" and row.result is not None
    ]
    if expected is not None:
        return answers, asdict(score_checkpoint_predictions(corpus, predictions, expected)), failed
    scorer = (
        score_mab_predictions if corpus.source.dataset == "memoryagentbench" else score_predictions
    )
    return answers, asdict(scorer(corpus, predictions)), failed


def _checkpoint_outcomes(
    corpus: BenchmarkCorpus, options: argparse.Namespace,
) -> CheckpointExpectedOutcomes | None:
    """Require separate reviewed prefix truth before generating or scoring any answers."""
    path = getattr(options, "checkpoint_outcomes", None)
    prefix = isinstance(corpus.source, DatasetSource) and corpus.source.received_prefix is not None
    if prefix != (path is not None):
        raise AMLError("evaluation_invalid", "Prefix evaluation needs explicit checkpoint outcomes")
    return None if path is None else validate_checkpoint_outcomes(
        corpus, CheckpointExpectedOutcomes.model_validate(read_source_json(path)),
    )


def _evaluate_frozen(corpus: BenchmarkCorpus, options: argparse.Namespace) -> tuple[str, bool]:
    answers, metric, failed = _score_answers(
        corpus,
        options,
        policy=_answer_policy(corpus, options),
        providers=FrozenBenchmarkProviders.load(options.fixtures),
    )
    return canonical_json(
        {
            "status": "frozen_incomplete" if failed else "frozen_complete",
            "provider_mode": "fully_bound_frozen_fixtures",
            "real_model_quality": "unverified",
            "milestone_acceptance": "unverified",
            "answers": answers.model_dump(mode="json"),
            "metric": metric,
        }
    ), failed


def _outcome(error: Exception) -> str:
    """Classify one failed live call without retaining any provider payload."""
    if isinstance(error, ProviderError):
        return f"provider_{error.error_class}"
    if isinstance(error, AMLError):
        return f"failed_{error.code}"
    return "failed_unexpected"


class _ThreadLocalRemoteProviders:
    """Give each answer worker its own transport and close every created client."""

    def __init__(
        self, base_url: str, credential: str, provider: ProviderName, timeout_ms: int,
        *, census: ProviderCallCensus | None = None,
    ) -> None:
        self._base_url = base_url
        self._credential = credential
        self._provider: ProviderName = provider
        self._timeout_ms = timeout_ms
        self._census = census
        self._thread = local()
        self._clients: list[httpx.Client] = []
        self._lock = Lock()

    def _current(self) -> RemoteBenchmarkProviders:
        current: RemoteBenchmarkProviders | None = getattr(self._thread, "provider", None)
        if current is None:
            timeout = httpx.Timeout(self._timeout_ms / 1000, connect=10.0)
            client = httpx.Client(timeout=timeout) if self._census is None else (
                CensusClient(self._census, timeout=timeout)
            )
            with self._lock:
                self._clients.append(client)
            current = RemoteBenchmarkProviders(
                JsonTransport(client, self._base_url, self._credential),
                self._provider,
                enabled=True,
            )
            self._thread.provider = current
        return current

    def answer(self, request: AnswerRequest) -> AnswerResult:
        return self._current().answer(request)

    def judge(self, request: RubricRequest) -> RubricResult:
        return self._current().judge(request)

    def close(self) -> None:
        with self._lock:
            clients = self._clients[:]
            self._clients.clear()
        for client in clients:
            client.close()


class _RecordedRemoteProviders:
    """Retain every live benchmark call as a licensed gzip JSON-lines transcript.

    One line per call holds the complete sealed request, including its rendered
    prompt messages and packed Search context, plus the validated result or the
    failure class of that attempt. Those lines reconstruct benchmark source text
    and inherit the source dataset's license conditions, and they are the input a
    later fixture build would need for a frozen replay of the same run. Recording
    changes no answer, score, or denominator.

    Args:
        providers (_BenchmarkProviders): The opted-in live answer/rubric adapter.
        path (Path): New gzip JSON-lines file; an existing path is refused.

    Raises:
        OSError: The transcript file exists or cannot be created.
    """

    def __init__(
        self, providers: _BenchmarkProviders, path: Path,
        *, census: ProviderCallCensus | None = None,
    ) -> None:
        self._providers = providers
        self._census = census
        self._stream = gzip.open(path, "xt", encoding="utf-8")
        self._lock = Lock()
        self.count = 0

    def _record(
        self,
        request: AnswerRequest | RubricRequest,
        result: AnswerResult | RubricResult | None,
        outcome: str,
        duration_ns: int,
    ) -> None:
        line = (
            json.dumps(
                {
                    "purpose": request.purpose,
                    "case_id": request.context.case_id,
                    "request_sha256": request.request_sha256,
                    "request": request.model_dump(mode="json"),
                    "response": None if result is None else result.model_dump(mode="json"),
                    "outcome": outcome,
                    "duration_ns": duration_ns,
                },
                ensure_ascii=True,
                sort_keys=True,
            )
            + "\n"
        )
        with self._lock:
            self._stream.write(line)
            self.count += 1

    def answer(self, request: AnswerRequest) -> AnswerResult:
        """Dispatch one live answer call and retain its exact request and output."""
        started = perf_counter_ns()
        try:
            with self._census.call(request) if self._census is not None else nullcontext():
                result = self._providers.answer(request)
        except Exception as error:
            # The caller retains each independent case; the transcript keeps the attempt.
            self._record(request, None, _outcome(error), perf_counter_ns() - started)
            raise
        self._record(request, result, "answered", perf_counter_ns() - started)
        return result

    def judge(self, request: RubricRequest) -> RubricResult:
        """Dispatch one live rubric call and retain its exact request and output."""
        started = perf_counter_ns()
        try:
            with self._census.call(request) if self._census is not None else nullcontext():
                result = self._providers.judge(request)
        except Exception as error:
            self._record(request, None, _outcome(error), perf_counter_ns() - started)
            raise
        self._record(request, result, "judged", perf_counter_ns() - started)
        return result

    def close(self) -> None:
        """Flush and close the transcript file."""
        self._stream.close()


def _remote_endpoint(options: argparse.Namespace) -> tuple[str, str]:
    """Check the live root and read its credential before any transport exists.

    Returns:
        tuple[str, str]: The base-URL host, which reports may retain, and the
            credential, which is passed only to the transport.

    Raises:
        AMLError: With ``evaluation_invalid`` for a root that is not a plain HTTPS
            URL, and with ``provider_unavailable`` when the named environment
            variable holds no credential. Neither failure has a fallback path.
    """
    base_url: str = options.remote_base_url
    parts = urlsplit(base_url)
    if (
        parts.scheme != "https"
        or not parts.hostname
        or parts.username
        or parts.password
        or parts.query
        or parts.fragment
    ):
        raise AMLError("evaluation_invalid", "Remote answering requires a plain HTTPS root")
    credential = os.environ.get(options.remote_credential_env, "")
    if not credential.strip():
        raise AMLError(
            "provider_unavailable", "Remote answering requires its explicit credential variable"
        )
    return parts.hostname, credential


def _evaluate_remote(corpus: BenchmarkCorpus, options: argparse.Namespace) -> tuple[str, bool]:
    """Answer every case with the opted-in live model and score it with the local evaluators.

    The client keeps the httpx default environment-proxy behavior used by the
    remote reviewer, so ``HTTPS_PROXY`` and ``NO_PROXY`` still select the route.
    """
    host, credential = _remote_endpoint(options)
    policy = _answer_policy(corpus, options)
    answer_workers: int = getattr(options, "answer_workers", 1)
    with ExitStack() as stack:
        census_path: Path | None = getattr(options, "remote_call_census", None)
        census = None if census_path is None else ProviderCallCensus(
            census_path, boundary="answers",
        )
        if census is not None:
            stack.callback(census.close)
        if answer_workers == 1:
            timeout = httpx.Timeout(policy.timeout_ms / 1000, connect=10.0)
            client = stack.enter_context(httpx.Client(timeout=timeout) if census is None else (
                CensusClient(census, timeout=timeout)
            ))
            remote: _BenchmarkProviders = RemoteBenchmarkProviders(
                JsonTransport(client, options.remote_base_url, credential),
                policy.provider,
                enabled=True,
            )
        else:
            threaded = _ThreadLocalRemoteProviders(
                options.remote_base_url, credential, policy.provider, policy.timeout_ms,
                census=census,
            )
            stack.callback(threaded.close)
            remote = threaded
        providers = _RecordedRemoteProviders(
            remote,
            options.remote_transcript,
            census=census,
        )
        stack.callback(providers.close)
        answers, metric, failed = _score_answers(
            corpus, options, policy=policy, providers=providers
        )
        calls = providers.count
    # The credential never enters this report; only its variable name and the host do.
    return canonical_json(
        {
            "status": "remote_incomplete" if failed else "remote_complete",
            "provider_mode": "remote_live_model",
            "real_model_quality": (
                "incomplete_development_baseline" if failed else "measured_development_baseline"
            ),
            "milestone_acceptance": "unverified",
            "answer_policy_sha256": benchmark_digest(policy.model_dump(mode="json")),
            "remote_base_url_host": host,
            "remote_credential_env": options.remote_credential_env,
            "environment_proxy_mode": "httpx_default_trust_env",
            "remote_transcript": str(options.remote_transcript),
            "remote_call_count": calls,
            "answer_workers": answer_workers,
            "answers": answers.model_dump(mode="json"),
            "metric": metric,
        }
    ), failed


def _verbatim(corpus: BenchmarkCorpus, options: argparse.Namespace) -> str:
    config = ReplayConfig(
        isolate_queries=options.isolate_queries,
        top_k=options.top_k,
        max_top_k=options.max_top_k,
        options=None,
        source_metadata_version=options.source_metadata_version,
    )
    if options.output is not None and options.output.resolve() == (
        options.search_contexts_output.resolve()
    ):
        raise AMLError("baseline_invalid", "Context and report outputs must be distinct")
    report = retrieve_verbatim(
        corpus,
        run_id=options.run_id,
        top_k=options.top_k,
        max_top_k=options.max_top_k,
        isolate_queries=options.isolate_queries,
        source_metadata_version=options.source_metadata_version,
    )
    cases = {case.case_id: case for case in corpus.cases}
    contexts = [
        BenchmarkContext.model_validate(
            {
                "dataset": corpus.source.dataset,
                "source_revision": corpus.source.revision,
                "source_identity_kind": report.dataset_identity_kind,
                "source_identity": report.dataset_identity,
                "source_selection_json": report.source_selection_json,
                "retrieval_origin": "verbatim_bm25_v1",
                "retrieval_binding_json": canonical_json(
                    {
                        "kind": "verbatim_bm25_v1",
                        "add_input_sha256": result.add_input_sha256,
                        "source_units_sha256": result.source_units_sha256,
                        "method_json": report.method_json,
                        "replay_config_json": canonical_json(config.model_dump(mode="json")),
                    }
                ),
                "run_key": report.run_id,
                "trial_key": options.trial_key,
                "case_id": result.case_id,
                "query": cases[result.case_id].query,
                "query_source_date": cases[result.case_id].query_source_date,
                "user_id": result.user_id,
                "source_question_json": cases[result.case_id].source_question_json,
                "search_request_json": result.request_json,
                "search_response_json": result.response_json,
            }
        )
        for result in report.results
    ]
    options.search_contexts_output.write_text(
        canonical_json([context.model_dump(mode="json") for context in contexts]) + "\n",
        encoding="utf-8",
    )
    return canonical_json(
        {
            "status": "local_retrieval_completed",
            "retrieval_origin": "verbatim_bm25_v1",
            "dataset_identity": report.dataset_identity,
            "dataset_identity_kind": report.dataset_identity_kind,
            "dataset_variant": report.dataset_variant,
            "run_id": report.run_id,
            "isolate_queries": report.isolate_queries,
            **({"source_metadata_version": config.source_metadata_version}
               if config.source_metadata_version is not None else {}),
            "group_count": report.group_count,
            "case_count": len(report.results),
            "method_json": report.method_json,
            "method_sha256": report.method_sha256,
            "http_delivery": "not_executed",
            "real_model_quality": "unverified",
            "milestone_acceptance": "unverified",
        }
    )


def _mem0(corpus: BenchmarkCorpus, options: argparse.Namespace) -> tuple[str, bool]:
    """Execute only an explicit complete registration in an already installed optional runtime."""
    if (
        options.output is None or options.output.exists()
        or options.search_contexts_output.exists()
        or options.output.resolve() == options.search_contexts_output.resolve()
    ):
        raise AMLError("baseline_invalid", "Mem0 requires distinct new report and context outputs")
    registration = Mem0Registration.model_validate_json(
        canonical_json(read_source_json(options.registration)),
    )
    expected = register_mem0(
        corpus, run_key=registration.run_key, trial_key=registration.trial_key,
        config=registration.replay_config, arrival=registration.arrival, method=registration.method,
    )
    if (
        registration != expected
        or registration.method.adapter_source_sha256 != mem0_adapter_digest()
    ):
        raise AMLError("binding_mismatch", "Mem0 registration differs from the selected source")
    report = None
    try:
        with open_mem0(
            registration, wheel=options.mem0_wheel, dependency_lock=options.dependency_lock,
            environment=os.environ,
        ) as (memory, ledger):
            report = replay_mem0(corpus, registration, memory=memory, ledger=ledger)
        report = replace(report, close_status="complete")
    except Exception:
        if report is None:
            raise
        report = replace(report, status="incomplete", close_status="failed")
    with options.search_contexts_output.open("x", encoding="utf-8") as output:
        output.write(canonical_json([
            context.model_dump(mode="json") for context in report.contexts
        ]) + "\n")
    return mem0_report_json(report), report.status != "complete"


def _cause(error: BaseException) -> str:
    """Name the failure's own type, message and raising site for a one-line report.

    A driver that reports only ``benchmark_failed`` leaves nothing to act on: a
    run that dies in five seconds and one that dies after an hour read the same,
    and finding the real cause took three separate instrumented runs on
    2026-09-12. The type, the message and the innermost frame identify it
    without a traceback, and no argument, path or environment value is read, so
    nothing a caller supplied can reach the output.
    """
    frame = error.__traceback__
    while frame is not None and frame.tb_next is not None:
        frame = frame.tb_next
    where = (
        ""
        if frame is None
        else f" at {Path(frame.tb_frame.f_code.co_filename).name}"
        f":{frame.tb_lineno}:{frame.tb_frame.f_code.co_name}"
    )
    return f"{type(error).__name__}: {error}{where}"


def main() -> int:
    """Run explicit local benchmark commands, calling a model API only when opted in.

    Args:
        None. Arguments are read from the command line.

    Returns:
        int: Zero on successful preparation/scoring or valid HTTP replay; one on
            invalid inputs, failed replay, or I/O. Success is not milestone acceptance.
    """
    parser = argparse.ArgumentParser(prog="python -m aml.bench.cli")
    # Exception: evaluate-remote answers through the explicitly selected live provider.
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--variant",
        choices=[
            "refined",
            "oracle",
            "s_cleaned",
            "m_cleaned",
            "factconsolidation_6k_sh_mh",
            "128k",
        ],
        required=True,
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--received-prefix", type=Path)
    parser.add_argument("--checkpoint-outcomes", type=Path)
    parser.add_argument(
        "--conversation-id", action="append", dest="conversation_ids",
        help="Select one whole LoCoMo conversation; repeat for an explicit subset",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("download")
    commands.add_parser("inspect")
    prefix = commands.add_parser("register-prefix")
    prefix.add_argument("--checkpoint-id", required=True)
    prefix.add_argument("--message-counts", type=Path, required=True)
    run = commands.add_parser("replay")
    run.add_argument("--base-url", required=True)
    run.add_argument("--run-id", required=True)
    run.add_argument("--auth-mode", choices=["token", "bearer", "x-api-key"], required=True)
    run.add_argument("--credential-env", default="AML_BENCH_TOKEN")
    run.add_argument("--timeout", type=float, default=PLATFORM_REQUEST_TIMEOUT_SECONDS)
    run.add_argument("--top-k", type=int, default=10)
    run.add_argument("--max-top-k", type=int, default=100)
    run.add_argument("--isolate-queries", action="store_true")
    run.add_argument("--source-metadata-version", type=int, choices=[1])
    run.add_argument("--add-concurrency", type=int, default=1)
    run.add_argument("--search-concurrency", type=int, default=1)
    run.add_argument("--search-contexts-output", type=Path)
    run.add_argument("--trial-key")
    run.add_argument(
        "--arrival",
        choices=["all_adds_then_search", "immediate_first_search"],
        default="all_adds_then_search",
    )
    score = commands.add_parser("score")
    score.add_argument("--predictions", type=Path, required=True)
    comparison = commands.add_parser("compare-answers")
    comparison.add_argument("--registration", type=Path, required=True)
    comparison.add_argument("--aml-arm", type=Path, required=True)
    comparison.add_argument("--baseline-arm", type=Path, required=True)
    evaluate = commands.add_parser("evaluate-frozen")
    evaluate.add_argument("--contexts", type=Path, required=True)
    evaluate.add_argument("--fixtures", type=Path, required=True)
    evaluate.add_argument("--answer-policy", type=Path, required=True)
    evaluate.add_argument("--replay-config", type=Path, required=True)
    evaluate.add_argument("--rubric-policy", type=Path)
    evaluate.add_argument("--run-key", required=True)
    evaluate.add_argument("--trial-key", required=True)
    live = commands.add_parser("evaluate-remote")
    live.add_argument("--contexts", type=Path, required=True)
    live.add_argument("--answer-policy", type=Path, required=True)
    live.add_argument("--replay-config", type=Path, required=True)
    live.add_argument("--rubric-policy", type=Path)
    live.add_argument("--run-key", required=True)
    live.add_argument("--trial-key", required=True)
    live.add_argument("--remote-base-url", default=DASHSCOPE_COMPATIBLE_ROOT)
    live.add_argument("--remote-credential-env", default="DASHSCOPE_API_KEY")
    live.add_argument("--remote-transcript", type=Path, required=True)
    live.add_argument("--remote-call-census", type=Path)
    live.add_argument("--answer-workers", type=int, choices=(1, 4, 8, 16, 28), default=1)
    baseline = commands.add_parser("verbatim")
    baseline.add_argument("--run-id", required=True)
    baseline.add_argument("--trial-key", required=True)
    baseline.add_argument("--top-k", type=int, default=10)
    baseline.add_argument("--max-top-k", type=int, default=100)
    baseline.add_argument("--isolate-queries", action="store_true")
    baseline.add_argument("--source-metadata-version", type=int, choices=[1])
    baseline.add_argument("--search-contexts-output", type=Path, required=True)
    mem0 = commands.add_parser("mem0-oss")
    mem0.add_argument("--registration", type=Path, required=True)
    mem0.add_argument("--mem0-wheel", type=Path, required=True)
    mem0.add_argument("--dependency-lock", type=Path, required=True)
    mem0.add_argument("--search-contexts-output", type=Path, required=True)
    options = parser.parse_args()
    failed = False
    try:
        if options.conversation_ids is not None and options.command not in {
            "inspect", "replay", "score", "evaluate-frozen", "evaluate-remote",
            "verbatim", "compare-answers", "mem0-oss", "register-prefix",
        }:
            raise AMLError("dataset_invalid", "This command does not accept a conversation subset")
        sources = TypeAdapter(list[BenchmarkSource]).validate_python(
            read_source_json(options.sources)
        )
        matches = [source for source in sources if source.variant == options.variant]
        if len(matches) != 1:
            raise AMLError("dataset_invalid", "The source variant must resolve exactly once")
        if options.command == "download":
            with httpx.Client(
                timeout=httpx.Timeout(30.0, connect=10.0), follow_redirects=True
            ) as client:
                download_dataset(matches[0], options.dataset, client)
        corpus = load_dataset(options.dataset, matches[0])
        if options.conversation_ids is not None:
            corpus = select_conversations(corpus, options.conversation_ids)
        if options.received_prefix is not None:
            corpus = apply_received_prefix(corpus, ReceivedPrefixRegistration.model_validate(
                read_source_json(options.received_prefix),
            ))
        is_prefix = (
            isinstance(corpus.source, DatasetSource) and corpus.source.received_prefix is not None
        )
        if is_prefix and options.command not in {
            "inspect", "replay", "verbatim", "mem0-oss", "score",
            "evaluate-frozen", "evaluate-remote", "compare-answers",
        }:
            raise AMLError("dataset_invalid", "This command does not accept received-prefix input")
        if options.checkpoint_outcomes is not None and options.command not in {
            "score", "evaluate-frozen", "evaluate-remote",
        }:
            raise AMLError("dataset_invalid", "Checkpoint outcomes require a score command")
        if options.command in ("inspect", "download"):
            session_groups: dict[str, set[str]] = {}
            for case in corpus.cases:
                for session in case.sessions:
                    session_groups.setdefault(session.session_id, set()).add(case.cluster_id)
            payload = canonical_json(
                {
                    "dataset_sha256": source_identity(corpus.source),
                    "dataset_identity_kind": source_identity_kind(corpus.source),
                    "variant": corpus.source.variant,
                    "cases": len(corpus.cases),
                    "replay_groups": len({c.cluster_id for c in corpus.cases}),
                    "shared_source_sessions": sum(
                        len(groups) > 1 for groups in session_groups.values()
                    ),
                    "categories": dict(Counter(case.category for case in corpus.cases)),
                    "evaluation_kinds": sorted({case.evaluation_kind for case in corpus.cases}),
                    "unresolved_evidence_cases": sum(
                        bool(c.unresolved_evidence_ids) for c in corpus.cases
                    ),
                    "quality_status": "baseline_pending",
                }
            )
        elif options.command == "register-prefix":
            counts = TypeAdapter(dict[str, int]).validate_python(
                read_source_json(options.message_counts), strict=True,
            )
            payload = canonical_json(register_received_prefix(
                corpus, checkpoint_id=options.checkpoint_id, message_counts=counts,
            ).model_dump(mode="json"))
        elif options.command == "score":
            predictions = [
                Prediction.model_validate(decode_source_json(line))
                for line in options.predictions.read_bytes().splitlines()
            ]
            expected = _checkpoint_outcomes(corpus, options)
            scorer = (
                score_mab_predictions
                if corpus.source.dataset == "memoryagentbench"
                else score_predictions
            )
            payload = canonical_json(asdict(
                score_checkpoint_predictions(corpus, predictions, expected)
                if expected is not None else scorer(corpus, predictions),
            ))
        elif options.command == "compare-answers":
            registration_json = canonical_json(read_source_json(options.registration))
            aml_arm = AnswerComparisonArm.model_validate_json(
                canonical_json(read_source_json(options.aml_arm)),
            )
            baseline_arm = AnswerComparisonArm.model_validate_json(
                canonical_json(read_source_json(options.baseline_arm)),
            )
            comparison_report = (
                compare_prefix_answers(
                    corpus,
                    PrefixAnswerComparisonRegistration.model_validate_json(registration_json),
                    aml_arm, baseline_arm,
                ) if is_prefix else compare_answers(
                    corpus, AnswerComparisonRegistration.model_validate_json(registration_json),
                    aml_arm, baseline_arm,
                )
            )
            payload = canonical_json(asdict(comparison_report))
            failed = comparison_report.experiment_status != "declared_complete"
        elif options.command == "evaluate-frozen":
            payload, failed = _evaluate_frozen(corpus, options)
        elif options.command == "evaluate-remote":
            payload, failed = _evaluate_remote(corpus, options)
        elif options.command == "verbatim":
            payload = _verbatim(corpus, options)
        elif options.command == "mem0-oss":
            payload, failed = _mem0(corpus, options)
        else:
            base_url: str = options.base_url
            parts = urlsplit(base_url)
            credential = os.environ.get(options.credential_env)
            if (
                parts.scheme not in ("http", "https")
                or not parts.hostname
                or parts.username
                or parts.password
                or parts.query
                or parts.fragment
                or parts.path not in ("", "/")
                or not credential
                or not math.isfinite(options.timeout)
                or options.timeout <= 0
                or not 1 <= options.add_concurrency <= 64
                or not 1 <= options.search_concurrency <= 256
                or (options.search_contexts_output is not None and not options.trial_key)
                or (
                    options.search_contexts_output is not None
                    and options.output is not None
                    and options.search_contexts_output.resolve() == options.output.resolve()
                )
            ):
                raise AMLError(
                    "replay_invalid",
                    "Replay endpoint, credential, timeout, or concurrency is invalid",
                )
            header = "X-Api-Key" if options.auth_mode == "x-api-key" else "Authorization"
            value = (
                credential
                if header == "X-Api-Key"
                else (f"{'Token' if options.auth_mode == 'token' else 'Bearer'} {credential}")
            )
            with httpx.Client(
                base_url=options.base_url,
                headers={header: value},
                timeout=httpx.Timeout(options.timeout, connect=min(options.timeout, 30.0)),
                follow_redirects=False,
                limits=httpx.Limits(
                    max_connections=options.add_concurrency + options.search_concurrency,
                    max_keepalive_connections=options.add_concurrency + options.search_concurrency,
                ),
            ) as client:
                report = replay(
                    corpus,
                    client,
                    run_id=options.run_id,
                    arrival=options.arrival,
                    top_k=options.top_k,
                    max_top_k=options.max_top_k,
                    isolate_queries=options.isolate_queries,
                    source_metadata_version=options.source_metadata_version,
                    add_concurrency=options.add_concurrency,
                    search_concurrency=options.search_concurrency,
                )
            failed = any(receipt.failure_code is not None for receipt in report.receipts)
            if options.search_contexts_output is not None:
                contexts = _replay_contexts(
                    corpus,
                    report,
                    trial_key=options.trial_key,
                    config=ReplayConfig(
                        isolate_queries=options.isolate_queries,
                        top_k=options.top_k,
                        max_top_k=options.max_top_k,
                        options=None,
                        source_metadata_version=options.source_metadata_version,
                    ),
                )
                options.search_contexts_output.write_text(
                    canonical_json([context.model_dump(mode="json") for context in contexts])
                    + "\n",
                    encoding="utf-8",
                )
            # Body persistence requires the separate explicit context output option.
            # The default artifact contains no source text, credentials, or bodies.
            payload = canonical_json(
                {
                    "dataset_sha256": report.dataset_sha256,
                    "dataset_identity_kind": report.dataset_identity_kind,
                    "dataset_variant": report.dataset_variant,
                    "run_id": report.run_id,
                    "arrival": report.arrival,
                    "isolate_queries": report.isolate_queries,
                    **({"source_metadata_version": options.source_metadata_version}
                       if options.source_metadata_version is not None else {}),
                    "add_concurrency": report.add_concurrency,
                    "search_concurrency": report.search_concurrency,
                    "status": "transport_failed" if failed else "transport_passed",
                    "milestone_acceptance": "unverified",
                    "receipts": [
                        {key: value for key, value in asdict(receipt).items()
                         if key != "response_contract_version" or value is not None}
                        for receipt in report.receipts
                    ],
                }
            )
        if options.conversation_ids is not None:
            selected_ids = list(dict.fromkeys(case.cluster_id for case in corpus.cases))
            payload = canonical_json({
                **json.loads(payload),
                "conversation_selection": {
                    "source_identity": source_identity(corpus.source),
                    "conversation_ids": selected_ids,
                    "case_ids": [case.case_id for case in corpus.cases],
                },
            })
        if options.output is None:
            print(payload)
        elif options.command == "mem0-oss":
            with options.output.open("x", encoding="utf-8") as output:
                output.write(payload + "\n")
        else:
            options.output.write_text(payload + "\n", encoding="utf-8")
        return int(failed)
    except AMLError as error:
        print(json.dumps({"status": "failed", "code": error.code, "cause": _cause(error)}))
        return 1
    except (OSError, ValueError, ValidationError, httpx.HTTPError) as error:
        print(json.dumps(
            {"status": "failed", "code": "benchmark_failed", "cause": _cause(error)},
        ))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
