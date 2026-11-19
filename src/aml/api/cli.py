"""Initialization and owned benchmark HTTP service composition.

``aml serve`` runs ``BenchRuntime`` on S2b's production Add owner
(``add_controller_factory`` over ``composed_owner_factory``), the
runtime-owned Search, the memory reads with ``/memory/add-items``, and S6a's
``ForgetService`` behind ``forget_factory``. Provider keys and the HTTP
secret come only from the environment; argv carries nonsecret configuration.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import signal
import sys
import time
from collections.abc import Mapping
from dataclasses import asdict
from functools import partial
from pathlib import Path
from types import FrameType
from typing import Literal, cast, get_args

import httpx
from pydantic import TypeAdapter, ValidationError

from aml.adapters.http.auth import HTTPCredentials
from aml.api.bench import DEFAULT_REQUEST_TIMEOUT_SECONDS
from aml.api.job_cli import add_job_commands, run_job_command
from aml.bench.call_census import CensusAsyncClient, CensusClient, ProviderCallCensus
from aml.core.bootstrap import open_initialized_workspace
from aml.core.errors import AMLError
from aml.core.extraction_policy import AddMode, ExtractionContractVersion
from aml.core.provider_budget import ExtractionCallBudgetTrial
from aml.core.request_budget import DEFAULT_REQUEST_CONTRACT_VERSION, RequestContractVersion
from aml.core.technical_types import ExternalId
from aml.ingestion.add_controller_factory import (
    WriterRecovery,
    add_controller_factory,
    composed_owner_factory,
)
from aml.ingestion.judge_input import MemberInputObserver
from aml.ingestion.owner_measure import measure_unit
from aml.ingestion.owner_rounds import CommitObserver
from aml.providers.adapters import EXTRACT_REQUEST_EFFORT_ENV, extract_request_effort
from aml.providers.config import DEFAULT_ADD_MODE, AddTiersV1, ProviderConfig
from aml.providers.deadline_transport import AsyncClientFactory
from aml.service.forget import ForgetService
from aml.service.heap_policy import (
    freeze_started_heap,
    limit_malloc_arenas,
    tune_collector,
)
from aml.service.listener import UvicornListener
from aml.service.request_measurement import MeasurementCollector
from aml.service.runtime import (
    BenchRuntime,
    ControllerFactory,
    RuntimeDrainResult,
    RuntimeInspection,
)

_DRAIN_RETRY_SECONDS = 1.0
_FAILED_DRAIN_SECONDS = 60.0
_DEFAULT_MEASUREMENT_CAPACITY = 100_000
_JOB_ID: TypeAdapter[str] = TypeAdapter(ExternalId)
_EXTRACTION_CALL_BUDGET: TypeAdapter[ExtractionCallBudgetTrial | None] = TypeAdapter(
    ExtractionCallBudgetTrial | None,
)


def _notice(value: dict[str, object], *, error: bool = False) -> bool:
    """Write content-free status without letting a broken output bypass drain."""
    try:
        print(json.dumps(value), file=sys.stderr if error else sys.stdout, flush=True)
        return True
    except BaseException:
        return False


def _service_failure(error: BaseException, stage: str) -> None:
    """Retain bounded code locations without exception messages, locals or source lines."""
    def frames_for(caught: BaseException) -> list[dict[str, object]]:
        frames: list[dict[str, object]] = []
        current = caught.__traceback__
        while current is not None:
            frames.append({
                "file": Path(current.tb_frame.f_code.co_filename).name,
                "function": current.tb_frame.f_code.co_name, "line": current.tb_lineno,
            })
            if len(frames) > 8:
                frames.pop(0)
            current = current.tb_next
        return frames

    causes: list[dict[str, object]] = []
    seen = {id(error)}
    cause = error.__cause__
    while cause is not None and id(cause) not in seen and len(causes) < 4:
        seen.add(id(cause))
        detail: dict[str, object] = {
            "error_type": type(cause).__name__[:80], "frames": frames_for(cause),
        }
        if isinstance(cause, OSError):
            for name in ("errno", "winerror"):
                try:
                    number = getattr(cause, name, None)
                except BaseException:
                    continue
                if type(number) is int and 0 <= number <= 0xFFFFFFFF:
                    detail[name] = number
        causes.append(detail)
        cause = cause.__cause__
    _notice({
        "status": "failed", "code": "service_failed", "stage": stage,
        "error_type": type(error).__name__,
        "error_code": error.code if isinstance(error, AMLError) else None,
        "frames": frames_for(error), "causes": causes,
    }, error=True)


async def _listener_stopped(listener: UvicornListener) -> bool:
    try:
        await listener.wait_stopped()
        return True
    except BaseException:
        return False


async def _runtime_failed(runtime: BenchRuntime) -> bool:
    while not runtime.failed:
        await asyncio.sleep(0.05)
    return False


def _traffic_shutdown_detail(result: RuntimeDrainResult) -> dict[str, object]:
    """Project only closed drain state; never report watermarks or private causes."""
    if result.stage != "traffic":
        return {}
    if result.traffic is None:
        return {"traffic_call_pending": not result.failed}
    traffic = result.traffic
    state = traffic.traffic
    retry_ms = None
    if state is not None:
        wait = state.wait_seconds
        if wait is not None and math.isfinite(wait):
            retry_ms = math.ceil(wait * 1000)
    return {
        "traffic_outcome": traffic.outcome,
        "traffic_joined": traffic.joined,
        "traffic_failure": None if traffic.failure is None else traffic.failure.value,
        "traffic_pending": None if state is None else state.pending_at is not None,
        "traffic_in_flight": None if state is None else state.in_flight,
        "traffic_retry_remaining_ms": retry_ms,
    }


async def serve_bench(runtime: BenchRuntime, *, host: str, port: int) -> int:
    """Own signals and the event loop until the real runtime acknowledges closure.

    A timed-out drain keeps the workspace owned and unavailable. A failed
    service gets a bounded final drain window; expiry terminates this process
    without pretending its live workers or storage handles were closed. The OS
    releases process ownership and the next startup uses normal epoch recovery.
    """
    loop = asyncio.get_running_loop()
    stop = asyncio.Event()
    previous = {signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)}
    installed: list[signal.Signals] = []
    waiters: list[asyncio.Task[bool]] = []
    failed = False
    stage = "signal_setup"

    def request_stop(_signum: int, _frame: FrameType | None) -> None:
        loop.call_soon_threadsafe(stop.set)

    try:
        for signum in previous:
            if os.name == "nt":
                signal.signal(signum, request_stop)
            else:
                loop.add_signal_handler(signum, stop.set)
            installed.append(signum)
        stage = "runtime_start"
        limit_malloc_arenas()
        tune_collector()
        app = runtime.start()
        freeze_started_heap()
        stage = "listener_attach"
        listener = UvicornListener(app, host=host, port=port)
        runtime.attach_listener(listener)
        if not stop.is_set():
            stage = "listener_start"
            await listener.start(time.monotonic() + 30)
            if not stop.is_set():
                stage = "runtime_activate"
                runtime.activate()
                stage = "serving"
                if not _notice({"status": "serving", "port": listener.port}):
                    failed = True
                    stop.set()
                waiters = [asyncio.create_task(stop.wait()),
                           asyncio.create_task(_listener_stopped(listener)),
                           asyncio.create_task(_runtime_failed(runtime))]
                await asyncio.wait(waiters, return_when=asyncio.FIRST_COMPLETED)
                failed = failed or not stop.is_set()
    except BaseException as error:
        failed = True
        _service_failure(error, stage)

    # This code intentionally sits outside an exception handler: a long drain
    # must not retain a bootstrap/request traceback in sys.exception().
    previous_notice: tuple[object, ...] | None = None
    termination: asyncio.TimerHandle | None = None
    while True:
        if (runtime.failed or failed) and termination is None:
            # This is process ownership, not a worker cancellation or a storage
            # close. A blocking worker cannot keep a failed foreground process
            # alive indefinitely. Startup, listener and close failures also
            # enter this same bounded window. Keep it across cancelled drains.
            termination = loop.call_later(_FAILED_DRAIN_SECONDS, os._exit, 1)
        stage = "runtime"
        attempt_failed = False
        traffic_detail: dict[str, object] = {}
        try:
            result = await runtime.drain(time.monotonic() + 2)
            stage, attempt_failed = result.stage, result.failed
            traffic_detail = _traffic_shutdown_detail(result)
            if result.complete:
                stage = "runtime"
                if await runtime.close_async(time.monotonic() + 2):
                    if termination is not None:
                        termination.cancel()
                    break
                attempt_failed = True
        except BaseException:
            attempt_failed = True
        failed = failed or attempt_failed
        runtime_failed = runtime.failed
        watchdog_armed = termination is not None
        notice = (stage, attempt_failed, runtime_failed, watchdog_armed, tuple(
            (name, value) for name, value in traffic_detail.items()
            if name != "traffic_retry_remaining_ms"
        ))
        if notice != previous_notice:
            written = _notice({
                "status": "shutdown_blocked", "stage": stage,
                "failed": attempt_failed, "workspace_retained": True,
                "runtime_failed": runtime_failed, "failed_drain_watchdog_armed": watchdog_armed,
                **traffic_detail,
            }, error=True)
            failed = failed or not written
            previous_notice = notice
        try:
            await asyncio.sleep(_DRAIN_RETRY_SECONDS)
        except asyncio.CancelledError:
            failed = True

    for number, waiter in enumerate(waiters):
        # Listener drainage already joined its server loop. Let that observer
        # report an actual failure even when a signal arrived simultaneously.
        cancelled_here = number != 1 and not waiter.done()
        if cancelled_here:
            waiter.cancel()
        try:
            if not await waiter:
                failed = True
        except BaseException:
            failed = failed or not cancelled_here
    for signum in installed:
        if os.name != "nt":
            loop.remove_signal_handler(signum)
        signal.signal(signum, previous[signum])
    return 1 if failed or runtime.failed else 0


def _provider_configuration(options: argparse.Namespace) -> ProviderConfig:
    """Resolve one nonsecret configuration before application ownership or provider calls."""
    if options.provider_config is not None:
        if any(value is not None for value in (
            options.profile, options.embedding_dimensions, options.extraction_contract_version,
            options.extraction_call_budget_trial, options.add_mode, options.add_tier_fast_mode,
            options.add_tier_fast_from_chunk, options.add_tier_fast_from_source_bytes,
        )):
            raise AMLError("schema_invalid", "Provider config conflicts with selector flags")
        raw = options.provider_config.read_bytes()
        if options.provider_config_sha256 is not None and (
            hashlib.sha256(raw).hexdigest() != options.provider_config_sha256
        ):
            raise AMLError("binding_mismatch", "Provider config differs from its frozen bytes")
        config = ProviderConfig.model_validate_json(raw)
        if config.profile not in {"product", "bench"} or (
            options.command == "serve" and config.profile != "bench"
        ):
            raise AMLError("schema_invalid", "Provider config differs from the command profile")
        return config
    if options.provider_config_sha256 is not None:
        raise AMLError("schema_invalid", "Provider config digest requires its explicit file")
    serving = options.command == "serve"
    thresholds = options.add_tier_fast_from_chunk, options.add_tier_fast_from_source_bytes
    if options.add_tier_fast_mode is None and any(value is not None for value in thresholds):
        raise AMLError("schema_invalid", "Add tier thresholds require --add-tier-fast-mode")
    return ProviderConfig(
        profile=options.profile or ("bench" if serving else "product"),
        openai_base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        embedding_dimensions=(
            options.embedding_dimensions
            if options.embedding_dimensions is not None else 2048 if serving else 1024
        ),
        extraction_contract_version=cast(
            ExtractionContractVersion | None, options.extraction_contract_version,
        ),
        extraction_call_budget_trial=_EXTRACTION_CALL_BUDGET.validate_python(
            options.extraction_call_budget_trial, strict=True,
        ),
        add_mode=options.add_mode,
        add_tiers=None if options.add_tier_fast_mode is None else AddTiersV1(
            fast_mode=options.add_tier_fast_mode,
            fast_from_chunk=options.add_tier_fast_from_chunk,
            fast_from_source_bytes=options.add_tier_fast_from_source_bytes,
        ),
    )


def _serve_options_valid(options: argparse.Namespace) -> bool:
    """Check listener, timeout and capture options before any ownership or provider call."""
    return bool(options.host) and 0 <= options.port <= 65535 and math.isfinite(
        options.search_timeout_seconds,
    ) and 0 < options.search_timeout_seconds <= DEFAULT_REQUEST_TIMEOUT_SECONDS and (
        options.completion_work_budget_ms is None or options.completion_work_budget_ms > 0
    ) and options.measurement_capacity > 0 and all(
        path is None or not path.exists() for path in (
            options.provider_census, options.measurement_output,
        )
    ) and (
        options.provider_census is None or options.provider_census != options.measurement_output
    )


def _write_measurement(
    path: Path, measurement: MeasurementCollector, *, closed: bool, service_exit: int | None,
) -> None:
    """Write the collector's scalar records; seal only after the runtime closed."""
    sealed = False
    if closed:
        try:
            observed, finalizes = measurement.seal()
            sealed = True
        except ValueError:
            observed, finalizes = measurement.observed()
    else:
        observed, finalizes = measurement.observed()
    document = {
        "schema_version": 1, "kind": "aml_serve_measurement", "sealed": sealed,
        "service_exit": service_exit,
        "requests": [asdict(row) for row in observed],
        "finalizes": [asdict(row) for row in finalizes],
        "budgets": [asdict(row) for row in measurement.observed_budgets()],
        **measurement.extraction_summary(),
    }
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(document, sort_keys=True, default=str) + "\n")


def serving_controller_factory(
    completion_work_budget_ms: int | None = None, *,
    judge_members: MemberInputObserver | None = None, commits: CommitObserver | None = None,
) -> ControllerFactory:
    """The Add owner ``aml serve`` runs (S8a F2).

    S2b's composed owner (``composed_owner_factory`` with ``measure_unit``) under S2c's
    ``WriterRecovery``; an invalid completion allowance is refused before any owner exists.
    ``judge_members`` sees each reviewed member input with its pin (the recorded-replay
    harness's member registration, E.5); ``commits`` sees each round's membership and
    each component's fence as the commit runs (FENCE-1's recording).
    """
    return add_controller_factory(
        owner=composed_owner_factory(measure=measure_unit, observe=judge_members,
                                     commits=commits),
        recovery=WriterRecovery(), completion_work_budget_ms=completion_work_budget_ms,
    )


def bench_runtime(
    workspace: Path, config: ProviderConfig, environment: Mapping[str, str], *, job_id: str,
    credentials: HTTPCredentials, search_timeout_seconds: float,
    request_contract_version: RequestContractVersion, client: httpx.Client | None,
    async_client_factory: AsyncClientFactory | None, measurement: MeasurementCollector | None,
    inspection: RuntimeInspection | None, controller_factory: ControllerFactory | None = None,
    judge_members: MemberInputObserver | None = None,
) -> BenchRuntime:
    """Compose ``BenchRuntime`` as ``aml serve`` serves it (S8a F2).

    The serving Add owner (`serving_controller_factory`, unless the caller built it first
    to validate its allowance), the runtime-owned Search, the memory reads and S6a's
    ``ForgetService``. ``aml serve`` and the research harness's service children
    build their runtime here, so
    every served lifetime runs one composition. A recorded or replayed lifetime brings
    its provider transport in ``client`` and ``async_client_factory`` and its member
    registration in ``judge_members`` (E.5).

    Raises:
        ValueError: ``judge_members`` beside a caller-built ``controller_factory``.
    """
    if controller_factory is not None and judge_members is not None:
        raise ValueError("judge_members_require_the_serving_controller")
    return BenchRuntime(
        workspace, config, environment, job_id=job_id, credentials=credentials,
        controller_factory=serving_controller_factory(judge_members=judge_members)
        if controller_factory is None else controller_factory,
        search_timeout_seconds=search_timeout_seconds,
        request_contract_version=request_contract_version, client=client,
        async_client_factory=async_client_factory, measurement=measurement,
        inspection=inspection, forget_factory=ForgetService,
    )


def _serve(
    options: argparse.Namespace, config: ProviderConfig, *,
    inspection: RuntimeInspection | None, measurement: MeasurementCollector | None,
) -> int:
    """Run the owned service on the production Add owner, Search and forget (S8a F2).

    Configuration is validated before the census file, the runtime or the
    workspace is touched. Keys are read by the runtime from ``os.environ``, and so is
    ``AML_EXTRACT_REQUEST_EFFORT`` (``extract_request_effort``), named in one startup
    line when set.
    The census and measurement file are completed after the foreground returns.
    """
    if not _serve_options_valid(options):
        raise AMLError("schema_invalid", "Invalid HTTP service configuration")
    credentials = HTTPCredentials(
        cast(Literal["token", "bearer", "x-api-key"],
             os.environ.get("AML_HTTP_AUTH_MODE", "bearer")),
        os.environ.get("AML_HTTP_AUTH_SECRET", ""),
    )
    job_id = _JOB_ID.validate_python(os.environ.get("AML_JOB_ID", ""), strict=True)
    effort = extract_request_effort()
    if effort is not None:
        _notice({"status": "extract_request_effort", "level": "info",
                 "variable": EXTRACT_REQUEST_EFFORT_ENV, "effort": effort})
    controller_factory = serving_controller_factory(options.completion_work_budget_ms)
    collector = measurement
    if collector is None:
        collector = MeasurementCollector(
            request_capacity=options.measurement_capacity,
            finalize_capacity=options.measurement_capacity,
        )
    census = None if options.provider_census is None else ProviderCallCensus(
        options.provider_census, boundary="runtime",
    )
    client = None if census is None else CensusClient(census)
    service_exit: int | None = None
    try:
        runtime = bench_runtime(
            options.workspace, config, os.environ, job_id=job_id, credentials=credentials,
            search_timeout_seconds=options.search_timeout_seconds,
            request_contract_version=cast(
                RequestContractVersion, options.request_contract_version,
            ),
            client=client,
            async_client_factory=None if census is None else partial(CensusAsyncClient, census),
            measurement=collector, inspection=inspection, controller_factory=controller_factory,
        )
        service_exit = asyncio.run(serve_bench(runtime, host=options.host, port=options.port))
        return service_exit
    finally:
        if client is not None:
            client.close()
        if census is not None:
            census.close()
        if options.measurement_output is not None:
            try:
                _write_measurement(
                    options.measurement_output, collector,
                    closed=service_exit == 0, service_exit=service_exit,
                )
            except (OSError, TypeError, ValueError):
                # An observer failure never replaces the service's own result.
                _notice({"status": "measurement_failed"}, error=True)


def _route(options: argparse.Namespace) -> int:
    """Serve the user_id router until SIGINT or SIGTERM (``aml.adapters.http.router``)."""
    import uvicorn

    from aml.adapters.http.router import RouterConfig, router_app

    try:
        config = RouterConfig(tuple(cast(list[str], options.shards)))
    except ValueError:
        _notice({"status": "failed", "code": "configuration_failed"}, error=True)
        return 1
    shape_path = cast(Path | None, options.shape_log)
    shape_log = None
    if shape_path is not None:
        try:
            # Owner-only, append-only: the records hold no values, but they are evaluation metadata.
            shape_log = os.fdopen(os.open(
                shape_path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600,
            ), "a", encoding="utf-8")
        except OSError:
            _notice({"status": "failed", "code": "shape_log_unwritable"}, error=True)
            return 1
    _notice({"status": "routing", "port": cast(int, options.port), "shards": len(config.shards),
             "shape_log": shape_log is not None})
    try:
        uvicorn.run(
            router_app(config, shape_log=shape_log),
            host=cast(str, options.host), port=cast(int, options.port),
            access_log=False, log_level="warning", timeout_keep_alive=int(config.request_seconds),
        )
    finally:
        if shape_log is not None:
            shape_log.close()
    return 0


def main(
    *, inspection: RuntimeInspection | None = None,
    measurement: MeasurementCollector | None = None,
) -> int:
    """Adapt CLI configuration to the actual initialization or service owner."""
    parser = argparse.ArgumentParser(prog="aml")
    commands = parser.add_subparsers(dest="command", required=True)
    add_job_commands(commands)
    initialize = commands.add_parser("init", help="Initialize one private AML workspace")
    initialize.add_argument("--workspace", type=Path, required=True)
    initialize.add_argument("--profile", choices=("product", "bench"), default=None)
    initialize.add_argument("--embedding-dimensions", type=int, choices=(1024, 2048), default=None)
    serve = commands.add_parser(
        "serve", help="Run the owned benchmark HTTP service (keys only from the environment)",
    )
    serve.add_argument("--workspace", type=Path, required=True)
    serve.add_argument("--profile", choices=("bench",), default=None)
    serve.add_argument("--embedding-dimensions", type=int, choices=(1024, 2048), default=None)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000, help="0 selects a free port")
    serve.add_argument(
        "--search-timeout-seconds", type=float, default=DEFAULT_REQUEST_TIMEOUT_SECONDS,
    )
    serve.add_argument(
        "--request-contract-version", type=int, choices=(2,),
        default=DEFAULT_REQUEST_CONTRACT_VERSION,
    )
    serve.add_argument(
        "--completion-work-budget-ms", type=int, default=None,
        help="Request-completion work allowance; omitted keeps the stored or default policy",
    )
    serve.add_argument(
        "--provider-census", type=Path, default=None,
        help="Create this content-free provider HTTP call census (boundary runtime)",
    )
    serve.add_argument(
        "--measurement-output", type=Path, default=None,
        help="Create this scalar request/finalize measurement JSON after shutdown",
    )
    serve.add_argument(
        "--measurement-capacity", type=int, default=_DEFAULT_MEASUREMENT_CAPACITY,
        help="Bound on measured requests and finalizes; overflow only marks it unsealed",
    )
    route = commands.add_parser(
        "route", help="Route user-scoped requests to N user_id-sharded aml serve processes",
    )
    route.add_argument(
        "--shard", action="append", required=True, dest="shards",
        help="A shard's base URL, repeated in frozen shard order (http://127.0.0.1:8001)",
    )
    route.add_argument("--host", default="127.0.0.1")
    route.add_argument("--port", type=int, default=7999)
    route.add_argument(
        "--shape-log", type=Path, default=None,
        help="Append one value-free JSON shape record per user-route request (off by default)",
    )
    for command in (initialize, serve):
        command.add_argument(
            "--provider-config", type=Path,
            help="Read nonsecret ProviderConfig JSON; keep API keys in environment variables",
        )
        command.add_argument(
            "--provider-config-sha256",
            help="Require the exact provider-config bytes frozen by a registered caller",
        )
        command.add_argument(
            "--extraction-contract-version", type=int,
            choices=get_args(ExtractionContractVersion), default=None,
            help="Select an explicit frozen extraction contract for a new workspace",
        )
        command.add_argument(
            "--extraction-call-budget-trial", choices=get_args(ExtractionCallBudgetTrial),
            default=None, help="Select a frozen extraction wall budget for a new workspace",
        )
        command.add_argument(
            "--add-mode", choices=get_args(AddMode), default=None,
            help=f"Select the Add mode of a new workspace (default: {DEFAULT_ADD_MODE})",
        )
        command.add_argument(
            "--add-tier-fast-mode", choices=get_args(AddMode), default=None,
            help="Run a user's Adds past a tier threshold in this faster Add mode",
        )
        command.add_argument(
            "--add-tier-fast-from-chunk", type=int, default=None,
            help="The 1-based Add ordinal of a user from which the fast tier runs",
        )
        command.add_argument(
            "--add-tier-fast-from-source-bytes", type=int, default=None,
            help="The cumulative source bytes of a user from which the fast tier runs",
        )
    options = parser.parse_args()
    if options.command == "serve":
        # Before the workspace opens and any worker thread allocates.
        limit_malloc_arenas()
    if options.command == "route":
        return _route(options)
    job_result = run_job_command(options)
    if job_result is not None:
        return job_result
    try:
        config = _provider_configuration(options)
        if options.command == "serve":
            return _serve(options, config, inspection=inspection, measurement=measurement)
        with open_initialized_workspace(options.workspace, config, os.environ) as workspace:
            print(
                json.dumps(
                    {
                        "status": "initialized",
                        "generation": workspace.projection.generation,
                        "embedding_state": workspace.drift_state,
                        "mean_cosine_deviation": workspace.mean_cosine_deviation,
                    },
                    sort_keys=True,
                )
            )
    except (AMLError, ValidationError, OSError) as error:
        code = error.code if isinstance(error, AMLError) else (
            "initialization_failed" if options.command == "init" else "configuration_failed"
        )
        _notice({"status": "failed", "code": code}, error=options.command == "serve")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
