"""Own initialized bench resources until every explicit serving owner has drained.

This component opens no listener. It builds the one guarded Search use case
(``api/bench_search.SearchUseCase``) over the Writer, its registered
publisher, the bench authorization, the bound embedding provider and one
application-owned delivery queue, and hands it to the required controller, which
the supplied factory builds; the controller runs the Add owners and admits each
Search into that use case, which never waits for an owner.

``concurrency="interactive"`` serves a plugin: one user whose Add runs in the
background between turns while the user waits for Search. Its Searches deliver
without the Writer mutex, write their ``recall`` rows after the response, queue
per user and run ahead of background Add work, and its row caches have smaller
budgets (``PluginDelivery``, ``service/search_priority.py``,
``store/search_rows.py``, ``INTERACTIVE_ROW_CACHE_BYTES``). ``bench`` keeps the
bench delivery unchanged. Every runtime caps glibc's malloc arenas and tunes the
collector when it starts (``service/heap_policy.py``), however it was started.
"""

from __future__ import annotations

import asyncio
import json
import math
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

import httpx
from starlette.applications import Starlette

from quotemem.adapters.http.app import create_app
from quotemem.adapters.http.auth import HTTPCredentials
from quotemem.adapters.http.forget import MemoryForgetHTTP
from quotemem.adapters.http.memory import memory_read_http
from quotemem.adapters.http.recovery import BenchRecoveryHTTP
from quotemem.api.bench import (
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
    AcceptingAddDispatch,
    AddDispatch,
    SearchDelivery,
    SearchRequest,
)
from quotemem.api.bench_conflict_judge import ConflictJudge
from quotemem.api.bench_premise import PremiseJudge
from quotemem.api.bench_query import QueryVectorCache
from quotemem.api.bench_reader import ReaderNote
from quotemem.api.bench_recovery import BenchRecoveryService, RecoveryOperations
from quotemem.api.bench_rerank import PoolReranker
from quotemem.api.bench_search import PluginDelivery, SearchUseCase
from quotemem.api.bench_service import BenchService, SearchOperation
from quotemem.api.interactive_search import InteractiveSearch
from quotemem.api.memory_forget import (
    MemoryForgetOperations,
    MemoryForgetService,
    StatementForgetOperations,
    StatementForgetService,
)
from quotemem.core.add_outcome import AddOutcomeV1
from quotemem.core.bootstrap import (
    InitializedWorkspace,
    WorkspaceBootstrapResources,
    open_initialized_workspace,
)
from quotemem.core.controls import JobValueV1
from quotemem.core.errors import QuoteMemError
from quotemem.core.extraction_policy import ExtractionPolicyConfigV2, extraction_pipeline
from quotemem.core.failure_reason import FailureReason, failure_reason
from quotemem.core.identity import ContentBindingDigest
from quotemem.core.inference import MAX_JUDGE_CONCURRENCY
from quotemem.core.recall_policy import RecallPolicyConfigV4
from quotemem.core.request_budget import (
    DEFAULT_REQUEST_CONTRACT_VERSION,
    RequestBudget,
    RequestContractVersion,
    current_budget,
)
from quotemem.ingestion.boundary import AddRequestV1, parse_add_request
from quotemem.ingestion.service import AddReceipt, AddService
from quotemem.providers.auth import SingleUserAuth
from quotemem.providers.bindings import (
    inference_binding_for_policy,
    job_provider_bindings,
    memory_judge_binding,
)
from quotemem.providers.config import ProviderConfig
from quotemem.providers.deadline_transport import AsyncClientFactory
from quotemem.publication.history import MemoryReads
from quotemem.service.add_followup import AddFollowups, FollowedDispatch, FollowupStore
from quotemem.service.auth import BenchExactUserAuth
from quotemem.service.chat_forget import ChatForget, ChatForgetOperations
from quotemem.service.delivery import DeliveryQueue, ScopedDeliveryQueues
from quotemem.service.heap_policy import limit_malloc_arenas, tune_collector
from quotemem.service.lifecycle import ServiceLifetime
from quotemem.service.ownership import WorkspaceOwner
from quotemem.service.request_measurement import MeasurementCollector
from quotemem.service.search_priority import SearchActivity
from quotemem.service.traffic import JobTraffic
from quotemem.service.traffic_worker import JobTrafficWorker, TrafficDrainResult
from quotemem.store.erasure_gate import known_gate
from quotemem.store.read_model import StoreReadModel
from quotemem.store.row_cache import (
    INTERACTIVE_ROW_BINDING_BYTES,
    INTERACTIVE_ROW_CACHE_BYTES,
    cache_budgets,
)
from quotemem.store.search_rows import DeferredRecallRows
from quotemem.store.writer import Writer

type Concurrency = Literal["bench", "interactive"]
"""How one service process schedules Search against background Add work.

``bench`` is the delivery under the Writer mutex, one FIFO for all users.
``interactive`` is the plugin delivery (``api/bench_search.PluginDelivery``).
"""


class BenchController(RecoveryOperations, Protocol):
    """Required real Search and background-work owner; construction starts no work."""

    def start(self) -> None:
        """Start tracked workers; failed acknowledgement still requires drain."""

    def set_failure_callback(self, callback: Callable[[], None]) -> None:
        """Bind content-free runtime withdrawal before any worker starts."""

    @property
    def failure_reason(self) -> FailureReason | None:
        """Return the first content-free background failure, or None."""
        ...

    def search(self, request: SearchRequest, delivery: SearchDelivery) -> None:
        """Hand off to the factory's ``search`` operation; it never waits for an owner."""

    def refuse_fenced_add(self, request: AddRequestV1, budget: RequestBudget) -> None:
        """Recover or report a proved owner stop before this Add waits on its fence."""

    def require_ready(self) -> None:
        """Reject unavailable worker state without starting or repairing work."""

    def withdraw(self) -> None:
        """Stop new background admission without abandoning admitted work."""

    def drain(self, deadline: float) -> bool:
        """Join actual workers by an absolute monotonic deadline; retain on timeout."""
        ...

    def abandon(self, deadline: float) -> bool:
        """Cancel every worker and join by the deadline after a drain missed it."""
        ...


class BenchListener(Protocol):
    """The attached real listener's shutdown boundary, including its ASGI tasks."""

    def withdraw(self) -> None:
        """Stop accepting connections without waiting or abandoning request tasks."""

    def require_ready(self) -> None:
        """Verify that the attached listener still owns a running server."""

    async def drain(self, deadline: float) -> bool:
        """Cancel/wake as needed and join all listener/request tasks by the deadline."""
        ...


class ControllerFactory(Protocol):
    """Construct the existing controller with an optional bounded observer.

    ``search`` is the runtime's one ``SearchUseCase``; the controller's
    ``search`` admits a request into exactly that operation.
    """

    def __call__(
        self, ready: InitializedWorkspace, publisher: StoreReadModel,
        auth: BenchExactUserAuth, /, *, search: SearchOperation,
        measurement: MeasurementCollector | None = None,
    ) -> BenchController: ...


class ForgetFactory(Protocol):
    """Construct the forget owner (``ForgetService``) over this runtime.

    It receives the runtime's Writer resources, its one publisher and bench
    authorization, and the controller whose Add owners a forget
    must fence; construction starts no work. The runtime serves the returned
    operations on ``/memory/forget/prepare`` and ``/memory/forget``
    under its request lifetime; without a factory neither route is mounted.
    """

    def __call__(
        self, ready: InitializedWorkspace, publisher: StoreReadModel,
        auth: BenchExactUserAuth, /, *, controller: BenchController,
    ) -> MemoryForgetOperations: ...


type RuntimeInspection = Callable[[Writer, StoreReadModel, BenchExactUserAuth], None]
type DrainStage = Literal[
    "listener", "controller", "http", "traffic", "inspection", "publisher", "drained",
]


@dataclass(frozen=True)
class RuntimeDrainResult:
    """Describe actual shutdown progress, never authorize closure on an incomplete result."""

    stage: DrainStage
    complete: bool
    failed: bool = False
    traffic: TrafficDrainResult | None = None


@dataclass(frozen=True)
class _CallResult:
    value: bool | TrafficDrainResult | None = None
    failed: bool = False


def _call(operation: Callable[[], bool | TrafficDrainResult | None]) -> _CallResult:
    try:
        return _CallResult(operation())
    except BaseException:
        # No raw exception or traceback is stored in the completed Future.
        return _CallResult(failed=True)


def _close_publisher(
    publisher: StoreReadModel, deadline: float, rows: DeferredRecallRows | None = None,
) -> bool:
    """Write the queued interactive ``recall`` rows, then close the publisher.

    Returns:
        bool: False, closing nothing, while the queued rows are not yet written.
    """
    if rows is not None and not rows.close(deadline):
        return False
    publisher.close(timeout=max(0.0, deadline - time.monotonic()))
    return True


async def _drain_listener(listener: BenchListener, deadline: float) -> _CallResult:
    try:
        return _CallResult(await listener.drain(deadline))
    except BaseException:
        return _CallResult(failed=True)


@runtime_checkable
class _CompletionController(Protocol):
    def complete_add(
        self, request: AddRequestV1, receipt: AddReceipt, budget: RequestBudget,
    ) -> AddOutcomeV1: ...


class _Reception(AddService):
    """Keep durable reception and its required trigger notification in one admission."""

    controller: BenchController
    requires_completion = True
    chat_forget: ChatForget | None = None
    """Carries out the completed Add's in-chat forget requests when the policy names a rule."""
    followups: AddFollowups | None = None
    """Completes acknowledged Adds in the background (interactive concurrency, a plugin's)."""

    @contextmanager
    def add_reception(self, request: AddRequestV1) -> Iterator[AddDispatch]:
        """The reception's dispatch; an accepting one answers only once its follow-up ran."""
        with super().add_reception(request) as dispatch:
            followups = self.followups
            if followups is None or not isinstance(dispatch, AcceptingAddDispatch):
                yield dispatch
            else:
                yield FollowedDispatch(dispatch, followups, request)

    def _check_finalize_pending(self, request: AddRequestV1) -> None:
        budget = current_budget()
        if budget is None:
            raise QuoteMemError("invalid_request", "Completed-prefix Add requires an owned budget")
        self.controller.refuse_fenced_add(request, budget)

    def add(self, raw: object) -> AddReceipt:
        request = parse_add_request(raw)
        budget = current_budget()
        if budget is None:
            raise QuoteMemError("invalid_request", "Completed-prefix Add requires an owned budget")
        budget.check()
        receipt = super().add(request)
        controller = self.controller
        if not isinstance(controller, _CompletionController):
            raise QuoteMemError("unsupported_variant",
                                "Controller has no completed-prefix contract")
        outcome = controller.complete_add(request, receipt, budget)
        if self.chat_forget is not None:
            forgets = self.chat_forget.carry_out(request, budget)
            if forgets is not None and forgets.taken:
                # A memory this Add made that its own in-chat forget took (the record of the
                # request, a restatement of a forgotten fact) is no memory it added.
                # The frozen receipt keeps its counts; the answer reports what stays.
                outcome = outcome.model_copy(
                    update={"added": max(0, outcome.added - forgets.taken)})
        return replace(receipt, outcome=outcome)


class _FollowedService(BenchService):
    """The bench application whose Add answers its background completion's result.

    An exact resend of an Add a background run completes, with or without ``Prefer:
    respond-async``, reads that run's result, waiting for it while it runs; only an Add
    without a run is completed here (`complete`).
    """

    followups: AddFollowups | None = None

    def add(self, request: AddRequestV1) -> AddOutcomeV1:
        followups, budget = self.followups, current_budget()
        if followups is not None and budget is not None and (
                outcome := followups.wait(request, budget)) is not None:
            return outcome
        return self.complete(request)

    def complete(self, request: AddRequestV1) -> AddOutcomeV1:
        """The synchronous Add itself; a background run calls it."""
        return super().add(request)


class BenchRuntime:
    """Retain one workspace's resource stack across incomplete shutdown attempts.

    Construction performs no I/O. Start is synchronous bootstrap; attach the
    actual listener and activate separately. Use drain from one event loop; its
    blocking phase Future survives cancellation and concurrent attempts serialize.
    No context-manager exit can bypass the explicit drain/close acknowledgement.

    ``interactive`` selects the interactive Search profile (``api/interactive_search``):
    Search then makes no model call besides its query embedding, defaults and caps
    ``top_k``, renders compact hits and keeps to its deadline. None serves the bench
    Search contract unchanged. ``concurrency`` selects how Search is scheduled against
    background Add work (module documentation); it is independent of ``interactive``.
    """

    def __init__(
        self, root: Path, config: ProviderConfig, environment: Mapping[str, str], *,
        job_id: str, credentials: HTTPCredentials, controller_factory: ControllerFactory,
        search_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
        request_contract_version: RequestContractVersion = DEFAULT_REQUEST_CONTRACT_VERSION,
        client: httpx.Client | None = None,
        async_client_factory: AsyncClientFactory | None = None,
        clock_ms: Callable[[], int] = lambda: time.time_ns() // 1_000_000,
        measurement: MeasurementCollector | None = None,
        inspection: RuntimeInspection | None = None,
        forget_factory: ForgetFactory | None = None,
        interactive: InteractiveSearch | None = None,
        concurrency: Concurrency = "bench",
    ) -> None:
        if concurrency not in ("bench", "interactive"):
            raise QuoteMemError("schema_invalid", "Service concurrency is not released")
        if config.profile != "bench":
            raise QuoteMemError("unsupported_variant", "Bench runtime requires the bench profile")
        if type(request_contract_version) is not int or request_contract_version != 2:
            raise QuoteMemError("schema_invalid", "Request contract version is invalid")
        self._root, self._config = root, config
        self._environment: Mapping[str, str] = environment
        self._job_id, self._credentials = job_id, credentials
        self._factory, self._search_timeout = controller_factory, search_timeout_seconds
        self._request_contract_version = request_contract_version
        self._client, self._clock_ms = client, clock_ms
        self._async_client_factory = async_client_factory
        self._measurement = measurement
        self._inspection = inspection
        self._forget_factory = forget_factory
        self._interactive = interactive
        self._concurrency: Concurrency = concurrency
        self._recall_rows: DeferredRecallRows | None = None
        self._followups: AddFollowups | None = None
        self._chat_forget: ChatForget | None = None
        self.followup_hold: Callable[[], bool] | None = None
        """While it answers True no background Add completion starts (``agent serve``'s
        spend cap)."""
        self.followup_store: FollowupStore | None = None
        """What background Add completions read of the store (``agent serve``'s
        ``AgentStore``; ``service/add_followup.py``)."""
        self._search_activity: SearchActivity | None = None
        self._inspection_done = False
        self._inspection_auth: BenchExactUserAuth | None = None
        self._stack = ExitStack()
        self._bootstrap_resources = WorkspaceBootstrapResources()
        self._owner = WorkspaceOwner(root, profile="bench")
        self._workspace: InitializedWorkspace | None = None
        self._publisher: StoreReadModel | None = None
        # The store whose erasure waits `withdraw` ends (`quotemem.store.erasure_gate`).
        self._erasure_store: object | None = None
        self._controller: BenchController | None = None
        self._traffic: JobTrafficWorker | None = None
        self._listener: BenchListener | None = None
        self._lifetime = ServiceLifetime(self._verify)
        self._failure = threading.Event()
        self._failure_reason: FailureReason | None = None
        self._started = self._withdrawn = self._drained = self._closed = False
        self._abandoned = False
        self._stage: DrainStage = "listener"
        self._lock = asyncio.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1,
                                            thread_name_prefix="quotemem-service-close")
        self._pending: Future[_CallResult] | None = None
        self._close_pending: Future[_CallResult] | None = None
        self._close_thread: threading.Thread | None = None
        self._executor_stopping = False
        self._listener_task: asyncio.Task[_CallResult] | None = None
        self._resource_state = threading.Lock()
        self._workspace_borrowed = self._closing = self._close_started = False

    @property
    def lifetime(self) -> ServiceLifetime:
        """The request lifetime: an operator call admitted here is drained like a request."""
        return self._lifetime

    @property
    def failed(self) -> bool:
        """Observe a worker failure without inspecting content or triggering repair."""
        return self._failure.is_set()

    @property
    def failure_reason(self) -> FailureReason | None:
        """Return the first content-free failure source: controller, traffic, drain or close."""
        with self._resource_state:
            return self._failure_reason

    @property
    def search_activity(self) -> SearchActivity | None:
        """The interactive service's running Searches; None for ``bench`` or before start."""
        return self._search_activity

    @property
    def recall_rows(self) -> DeferredRecallRows | None:
        """The interactive service's deferred ``recall`` rows; None for ``bench``."""
        return self._recall_rows

    def _followup_held(self) -> bool:
        hold = self.followup_hold
        return hold is not None and hold()

    def add_forgets(self, body: bytes) -> list[dict[str, object]]:
        """What the in-chat forget requests of one completed Add did.

        Args:
            body: The Add's normalized request body, as the bench application received it.

        Returns:
            One entry per request (``ChatForget.report``); empty when the policy carries out
            no complete in-chat forget, the Add holds no request or the body is not an Add.
        """
        chat_forget = self._chat_forget
        if chat_forget is None:
            return []
        try:
            request = parse_add_request(json.loads(body))
        except (QuoteMemError, ValueError):
            return []
        return chat_forget.report(request)

    def followups_running(self) -> int:
        """Acknowledged Adds this process is still completing in the background."""
        followups = self._followups
        return 0 if followups is None else followups.running()

    @property
    def followups(self) -> AddFollowups | None:
        """The background completions of acknowledged Adds; None for ``bench``."""
        return self._followups

    def _fail(self, site: str = "unspecified", error: BaseException | None = None) -> None:
        with self._resource_state:
            if self._failure_reason is None:
                if site == "inspection" and error is not None:
                    # Optional observers contribute categories, not their
                    # messages, arguments or potentially user-defined frames.
                    reason = FailureReason(
                        site, type(error).__name__,
                        error.code if isinstance(error, QuoteMemError) else None, None,
                    )
                else:
                    reason = failure_reason(site, error)
                controller = self._controller
                if site == "controller" and controller is not None:
                    # A failure callback never raises; a controller without a
                    # recorded reason contributes only its site.
                    nested = getattr(controller, "failure_reason", None)
                    if isinstance(nested, FailureReason):
                        reason = nested._replace(site=f"controller:{nested.site}")
                self._failure_reason = reason
        self._failure.set()
        self._lifetime.stop_accepting()

    def _verify(self) -> None:
        ready, publisher, controller, traffic = (
            self._workspace, self._publisher, self._controller, self._traffic,
        )
        if ready is None or publisher is None or controller is None or traffic is None:
            raise QuoteMemError("service_unhealthy", "Bench startup has not completed")
        if ready.runtime.owner is not self._owner:
            raise QuoteMemError("service_unhealthy", "Bench runtime lost its storage owner")
        with ready.writer.mutex:
            ready.writer.validate_storage_owner()
            if StoreReadModel.get(ready.writer) is not publisher:
                raise QuoteMemError("service_unhealthy", "Bench runtime lost its publisher")
            _ = publisher.current
        controller.require_ready()
        traffic.require_ready()
        if self._listener is None:
            raise QuoteMemError("service_unhealthy", "Bench runtime has no listener owner")
        self._listener.require_ready()

    def start(self) -> Starlette:
        """Bootstrap and construct the app without publishing service readiness.

        Failure withdraws admission and retains acquired resources for explicit
        drain/close. The supplied controller must not start work in its factory.
        """
        if self._started or self._closed:
            raise RuntimeError("Bench runtime can start only once")
        self._started = True
        # Before the workspace opens and any worker thread allocates; a second
        # runtime in the same process changes nothing (``heap_policy``).
        limit_malloc_arenas()
        tune_collector()
        published = self._concurrency == "interactive"
        try:
            with cache_budgets(
                INTERACTIVE_ROW_CACHE_BYTES, INTERACTIVE_ROW_BINDING_BYTES,
            ) if published else nullcontext():
                ready = self._stack.enter_context(open_initialized_workspace(
                    self._root, self._config, self._environment, client=self._client,
                    owner=self._owner, resources=self._bootstrap_resources,
                    async_client_factory=self._async_client_factory,
                ))
            self._workspace = ready
            extraction_pointer = ready.writer.get_control("policy:extraction")
            recall_pointer = ready.writer.get_control("policy:recall")
            if extraction_pointer is None or recall_pointer is None:
                raise QuoteMemError("service_unhealthy", "Bench startup has no retained policies")
            extraction_release = ready.writer.read_logical(
                extraction_pointer["value"]["current_release_ref"], "policy_release",
            )
            recall_release = ready.writer.read_logical(
                recall_pointer["value"]["current_release_ref"], "policy_release",
            )
            extraction = ready.writer.read_logical(
                extraction_release.model_dump(mode="json")["config_ref"], "policy_config",
            )
            recall = ready.writer.read_logical(
                recall_release.model_dump(mode="json")["config_ref"], "policy_config",
            )
            if not isinstance(extraction, ExtractionPolicyConfigV2) or not isinstance(
                recall, RecallPolicyConfigV4,
            ):
                raise QuoteMemError("service_unhealthy", "Bench retained policy kinds differ")
            job = ready.writer.bind_bench_job(
                self._job_id, job_provider_bindings(self._config, extraction), clock=self._clock_ms,
            )
            if not isinstance(job.value, JobValueV1):
                raise QuoteMemError("service_unhealthy", "Bench job binding has another kind")
            publisher = self._publisher = StoreReadModel(ready.writer)
            self._erasure_store = ready.writer
            # A check inside a Writer section cannot renew, so each uses a grant
            # resolved outside it: an owner's borrow renews its grant once half
            # the lifetime has passed, a commit resolves one for its whole
            # component transaction, and extraction and judge outputs resolve a
            # fresh one after their provider call. The lifetime thus bounds one
            # commit or the wait for the commit mutex,
            # minutes under normal load, never a model call. An expired grant
            # ends the invocation with `authorization_expired`; the next resolves
            # a fresh grant and re-proves every fence. The provider's default
            # minute leaves a renewed borrow half a minute for that wait; fifteen
            # minutes leaves seven and a half and stays inside the one-hour
            # maximum of this provider.
            auth = BenchExactUserAuth(
                SingleUserAuth(ready.keyring, []).principal_id, ttl_ms=900_000,
            )
            self._inspection_auth = auth
            # One use case and one FIFO delivery queue serve every Search of
            # this runtime; construction starts no worker. Its query vectors
            # are reused through the service-owned cache of this embedding provider.
            # Plain Searches of short recall questions also get the premise judge
            # (bench_premise) on the pipeline's memory judge model.
            # Plain Searches of very large scopes also re-rank their fused pool
            # (bench_rerank) on the same model, and plain Searches the reader note
            # (bench_reader), which closes the last hit.
            judge = memory_judge_binding(
                inference_binding_for_policy(extraction), concurrency=MAX_JUDGE_CONCURRENCY,
                read_timeout_ms=extraction_pipeline(
                    extraction.parameters.pipeline_version).judge_read_timeout_ms,
            )

            def bind(purpose: str, value: object) -> ContentBindingDigest:
                # Bound per call: the keyring is the Writer's, read when a request is built.
                return ready.writer.bind_content(purpose, value)

            # An interactive service delivers its Searches without the Writer and
            # lets them run ahead of background Add work (plugin profile).
            plugin_delivery: PluginDelivery | None = None
            if published:
                activity = self._search_activity = SearchActivity()
                ready.writer.mutex.before_contend(activity.yield_to_searches)
                rows = self._recall_rows = DeferredRecallRows(ready.writer, publisher)
                plugin_delivery = PluginDelivery(rows, activity, ScopedDeliveryQueues())
            # Plain Searches get the model conflict check (bench_conflict_judge) on
            # the same judge model; the lexical rule only nominates its pairs.
            # The interactive profile (a memory plugin's Search) takes none of the four
            # model notes: its only provider call is the query embedding.
            interactive = self._interactive
            search = SearchUseCase(
                ready.writer, publisher, auth, ready.providers.embedding, DeliveryQueue(),
                profile="bench", clock=self._clock_ms,
                vector_cache=QueryVectorCache(ready.providers.embedding),
                premise=PremiseJudge(ready.providers.inference, judge, bind)
                if interactive is None else None,
                rerank=PoolReranker(ready.providers.inference, judge, bind)
                if interactive is None else None,
                conflicts=ConflictJudge(ready.providers.inference, judge, bind)
                if interactive is None else None,
                reader=ReaderNote(ready.providers.inference, judge, bind)
                if interactive is None else None,
                interactive=interactive,
                plugin_delivery=plugin_delivery,
            )
            if self._measurement is None:
                controller = self._factory(ready, publisher, auth, search=search)
            else:
                controller = self._factory(
                    ready, publisher, auth, search=search, measurement=self._measurement,
                )
            self._controller = controller
            if not isinstance(
                controller, _CompletionController,
            ):
                raise QuoteMemError("unsupported_variant",
                                    "Controller has no completed-prefix contract")
            controller.set_failure_callback(lambda: self._fail("controller"))
            controller.start()

            def persist(at: int) -> None:
                ready.writer.flush_job_traffic(self._job_id, at, clock=self._clock_ms)

            traffic = self._traffic = JobTrafficWorker(
                JobTraffic(persist, acknowledged_at=job.value.last_platform_traffic_at),
                clock_ms=self._clock_ms,
                on_failure=lambda reason: self._fail(f"traffic:{reason}"),
            )
            traffic.start()
            reception = _Reception(
                ready.writer, ready.providers.embedding, auth, publisher,
                profile="bench", clock=self._clock_ms,
            )
            reception.controller = controller
            application = _FollowedService(
                reception, search=controller.search, lifetime=self._lifetime)
            if published:
                # A plugin's acknowledged Add completes in the background, the forget work
                # after completion included, so its resend reads a stored result.
                application.followups = reception.followups = self._followups = AddFollowups(
                    application.complete, self._search_timeout)
                self._followups.hold = self._followup_held
                self._followups.store = self.followup_store
            # forget (the ForgetService) through the adapter,
            # admitted by the same lifetime as every other request; forget by
            # statement is mounted when the operations implement it.
            memory_forget: MemoryForgetHTTP | None = None
            operations = None if self._forget_factory is None else self._forget_factory(
                ready, publisher, auth, controller=controller)
            if operations is not None:
                memory_forget = MemoryForgetHTTP(
                    MemoryForgetService(operations, self._lifetime),
                    statements=StatementForgetService(operations, self._lifetime)
                    if isinstance(operations, StatementForgetOperations) else None,
                )
            # A policy that names an in-chat forget rule carries out its users' forget
            # requests after each Add; without forget by statement it could
            # not, so the runtime refuses to start rather than ignore them.
            if (rule := extraction.parameters.chat_forget) is not None:
                if not isinstance(operations, ChatForgetOperations):
                    raise QuoteMemError("unsupported_variant",
                                        "In-chat forget requests need forget by statement")
                reception.chat_forget = self._chat_forget = ChatForget(operations, rule)
            return create_app(
                application, self._credentials,
                max_top_k=recall.parameters.max_top_k if interactive is None
                else min(recall.parameters.max_top_k, interactive.max_top_k),
                default_top_k=None if interactive is None
                else min(recall.parameters.max_top_k, interactive.default_top_k),
                search_timeout_seconds=self._search_timeout,
                request_contract_version=self._request_contract_version,
                on_successful_traffic=traffic.observe_success, worker_lifetime=self._lifetime,
                add_reception=application.add_reception,
                measurement=self._measurement,
                recovery=BenchRecoveryHTTP(
                    BenchRecoveryService(controller, self._lifetime),
                    completion_enabled=True,
                ),
                # History v3 and Why v2 over the stored judge records;
                # a verdict_ref naming no stored record fails
                # closed as service_unhealthy. /memory/add-items reads the
                # completion records the Add owner writes before release
                # (``record_add_completion``).
                memory_forget=memory_forget,
                memory_reads=memory_read_http(
                    MemoryReads(ready.writer, auth, publisher, clock=self._clock_ms),
                    add_items=True,
                ),
            )
        except BaseException:
            self._lifetime.stop_accepting()
            raise
        finally:
            # Bootstrap resolves credentials into its provider owner. Do not keep
            # a second environment mapping in this long-lived resource owner.
            self._environment = {}

    def attach_listener(self, listener: BenchListener) -> None:
        """Attach exactly one constructed listener before activation or shutdown."""
        if not self._started or self._listener is not None or self._withdrawn or self._closed:
            raise RuntimeError("Listener attachment is no longer available")
        self._listener = listener

    def activate(self) -> None:
        """Publish readiness only after the actual listener owner is attached."""
        if self._listener is None or self._withdrawn or self._closed:
            raise RuntimeError("Activation requires the attached listener owner")
        self._lifetime.activate()

    def withdraw(self) -> None:
        """Permanently stop admission; do not wait or close resources here."""
        self._lifetime.stop_accepting()
        self._withdrawn = True
        if self._followups is not None:
            self._followups.stop()
        # A forget or read waiting for its scope's erasure turn answers at once now; a
        # turn already given runs to its end.
        gate = None if self._erasure_store is None else known_gate(self._erasure_store)
        if gate is not None:
            gate.close()
        # Attempt every owned withdrawal even when one reports failure.
        failed = False
        for owner in (self._listener, self._controller):
            if owner is not None:
                try:
                    owner.withdraw()
                except BaseException:
                    failed = True
        if failed:
            raise RuntimeError("A serving owner could not acknowledge withdrawal") from None

    async def _blocking(
        self, operation: Callable[[], bool | TrafficDrainResult | None], deadline: float,
    ) -> _CallResult | None:
        resumed = self._pending is not None
        if self._pending is None:
            self._pending = self._executor.submit(_call, operation)
        pending = self._pending
        wrapped = asyncio.wrap_future(pending)
        done, _ = await asyncio.wait({wrapped}, timeout=max(0.0, deadline - time.monotonic()))
        if not done:
            return None
        self._pending = None
        result = pending.result()
        timed_out = result.value is False or isinstance(result.value, TrafficDrainResult) and (
            result.value.outcome in {"pending_timeout", "active_timeout", "join_timeout"}
        )
        if resumed and timed_out and time.monotonic() < deadline:
            # Join the old call before starting exactly one attempt with this
            # invocation's fresh deadline. Never overlap the two operations.
            return await self._blocking(operation, deadline)
        return result

    async def _listener_drain(
        self, listener: BenchListener, deadline: float,
    ) -> _CallResult | None:
        """Join a retained listener drain, then attempt once under this deadline.

        A retained task keeps the deadline of the call that started it. When it
        has already expired under that earlier deadline, its False result must
        not stand in for an attempt under this invocation's budget; the other
        blocking stages make the same distinction in ``_blocking``.
        """
        resumed = self._listener_task is not None
        if self._listener_task is None:
            self._listener_task = asyncio.create_task(_drain_listener(listener, deadline))
        task = self._listener_task
        done, _ = await asyncio.wait({task}, timeout=max(0.0, deadline - time.monotonic()))
        if not done:
            return None
        self._listener_task = None
        try:
            result = task.result()
        except asyncio.CancelledError:
            result = _CallResult(failed=True)
        if resumed and result.value is False and time.monotonic() < deadline:
            # The retained attempt is joined; make exactly one fresh attempt.
            return await self._listener_drain(listener, deadline)
        return result

    async def drain(self, deadline: float) -> RuntimeDrainResult:
        """Drain each owner in order, retaining in-progress calls across cancellation.

        The deadline is absolute monotonic time. Cancellation or timeout never
        cancels a submitted blocking shutdown call or closes the resource stack.
        Retry this method to join that same call before attempting later phases.
        """
        if type(deadline) not in {int, float} or not math.isfinite(deadline):
            raise ValueError("Drain requires a finite absolute monotonic deadline")
        if self._lock.locked():
            try:
                await asyncio.wait_for(self._lock.acquire(), max(0.0, deadline - time.monotonic()))
            except TimeoutError:
                return RuntimeDrainResult(self._stage, False)
        else:
            await self._lock.acquire()
        try:
            if self._closed or self._drained:
                return RuntimeDrainResult("drained", True, failed=self.failed)
            try:
                self.withdraw()
            except RuntimeError:
                return RuntimeDrainResult(self._stage, False, failed=True)
            stages: tuple[DrainStage, ...] = (
                "listener", "http", "controller", "traffic", "inspection", "publisher",
            )
            for stage in stages[stages.index(self._stage):]:
                self._stage = stage
                result: _CallResult | None
                if stage == "listener":
                    if self._listener is None:
                        continue  # A failed pre-listener startup has no listener to stop.
                    result = await self._listener_drain(self._listener, deadline)
                elif stage == "controller":
                    if self._controller is None:
                        continue
                    controller = self._controller
                    result = await self._blocking(partial(controller.drain, deadline), deadline)
                elif stage == "http":
                    result = await self._blocking(
                        partial(self._lifetime.wait_drained, deadline), deadline,
                    )
                elif stage == "traffic":
                    if self._traffic is None:
                        continue
                    traffic = self._traffic
                    result = await self._blocking(partial(traffic.drain, deadline), deadline)
                elif stage == "inspection":
                    if self._inspection is None or self._inspection_done:
                        continue
                    result = await self._blocking(self._inspect_before_publisher_close, deadline)
                    if result is None:
                        return RuntimeDrainResult(stage, False, failed=self.failed)
                    self._inspection_done = True
                    if result.failed:
                        self._fail("inspection")
                    # Inspection cannot prevent the already joined service from
                    # closing its publisher and native resources on a failure.
                    continue
                else:
                    if self._publisher is None:
                        continue
                    publisher = self._publisher
                    result = await self._blocking(
                        partial(_close_publisher, publisher, deadline, self._recall_rows),
                        deadline,
                    )
                if result is None or result.failed or result.value is False:
                    return RuntimeDrainResult(stage, False, failed=bool(result and result.failed))
                if isinstance(result.value, TrafficDrainResult):
                    if result.value.failure is not None:
                        self._fail(f"drain:{stage}:{result.value.failure}")
                    if result.value.outcome != "settled" or not result.value.joined:
                        return RuntimeDrainResult(
                            stage, False, failed=result.value.failure is not None,
                            traffic=result.value,
                        )
                    # A successful due retry can settle and join a worker whose
                    # original failure remains latched. Preserve failed health,
                    # but do not strand resources after acknowledged drainage.
            self._stage = "drained"
            self._drained = True
            return RuntimeDrainResult("drained", True, failed=self.failed)
        finally:
            self._lock.release()

    def _inspect_before_publisher_close(self) -> None:
        """Lend existing authorities once, after traffic joins and before pins close.

        The optional internal observer may only read its registered exact scopes.
        It must not retain these owners, perform inference, or mutate storage.
        The ordinary retained blocking-call machinery joins cancellation and
        deadline retries before disposal can advance.
        """
        ready, publisher, auth = self._workspace, self._publisher, self._inspection_auth
        if ready is None or publisher is None or auth is None or self._inspection is None:
            raise RuntimeError("Inspection requires the actual initialized service owners")
        # All serving owners have joined. The retained blocking call owns this
        # observation until it unwinds; read guards must be released outside the
        # Writer mutex, including when the observer raises or is cancelled.
        try:
            with ready.writer.mutex:
                ready.writer.validate_storage_owner()
            try:
                self._inspection(ready.writer, publisher, auth)
            finally:
                with ready.writer.mutex:
                    ready.writer.validate_storage_owner()
        except BaseException as error:
            # Capture safe attribution before _call discards the exception.
            self._fail("inspection", error)
            raise

    def consume_drained_workspace(
        self, consumer: Callable[[Writer], None], *,
        measurement: MeasurementCollector | None = None,
    ) -> None:
        """Borrow this runtime's actual Writer after successful complete drainage.

        The synchronous callback runs outside runtime and Writer locks and may
        read retained bodies. It must not retain or dispose of the Writer. An
        optional collector must be this runtime's configured observer. This
        lifetime proves neither a healthy service run nor measurement acceptance.
        Close and overlapping borrows reject until the callback has unwound;
        a partial close permanently ends this read opportunity.
        """
        with self._resource_state:
            if self._closed or self._close_started:
                raise RuntimeError("Drained workspace is unavailable after disposal starts")
            if self._workspace_borrowed:
                raise RuntimeError("Drained workspace already has an active consumer")
            if not self._drained or self._lock.locked() or self._pending is not None or (
                self._listener_task is not None or self._workspace is None
            ):
                raise RuntimeError("Workspace consumption requires completed successful drainage")
            if measurement is not None and measurement is not self._measurement:
                raise RuntimeError("Measurement collector does not belong to this runtime")
            ready = self._workspace
            self._workspace_borrowed = True
        try:
            if ready.runtime.owner is not self._owner:
                raise QuoteMemError("service_unhealthy", "Drained workspace lost its storage owner")
            with ready.writer.mutex:
                ready.writer.validate_storage_owner()
            consumer(ready.writer)
            with ready.writer.mutex:
                ready.writer.validate_storage_owner()
        finally:
            with self._resource_state:
                self._workspace_borrowed = False

    async def abandon(self, deadline: float) -> bool:
        """Cancel the retained owners' work and join their threads by the deadline.

        The bounded way out after ``drain`` missed its deadline: admission is
        withdrawn, every worker is cancelled and joined, and the traffic
        flusher is drained so no non-daemon thread outlives the caller. The
        resource stack stays retained and is never closed here; durable state
        is left for a later recovery. Returns whether every thread left.
        """
        if type(deadline) not in {int, float} or not math.isfinite(deadline):
            raise ValueError("Abandon requires a finite absolute monotonic deadline")
        async with self._lock:
            if self._closed or self._drained:
                return True
            try:
                self.withdraw()
            except RuntimeError:
                pass
            joined = True
            controller = self._controller
            if controller is not None:
                result = await self._blocking(partial(controller.abandon, deadline), deadline)
                joined = bool(result is not None and result.value is True and not result.failed)
            traffic = self._traffic
            if traffic is not None:
                result = await self._blocking(partial(traffic.drain, deadline), deadline)
                joined = joined and result is not None and not result.failed and (
                    isinstance(result.value, TrafficDrainResult) and result.value.joined
                )
            self._abandoned = True
            return joined

    def close(self) -> None:
        """Close the drained runtime before releasing its context and flock.

        An incomplete phase or runtime-close error retains the owning stack.
        The original initialized context performs its normal idempotent runtime
        exit only after that first close has acknowledged actual resource drain.
        """
        with self._resource_state:
            if self._closed:
                return
            if self._workspace_borrowed or self._closing:
                raise RuntimeError("Bench resources have an active workspace consumer or close")
            if not self._drained or self._pending is not None or self._listener_task is not None:
                raise RuntimeError("Bench resources must completely drain before close")
            self._closing = self._close_started = True
        try:
            self._close_resources()
            # Drainage and the close exclusion leave no executor operation
            # active. Keep it reusable if native disposal needs an async retry.
            self._executor.shutdown(wait=True)
            with self._resource_state:
                self._closed = True
        finally:
            with self._resource_state:
                self._closing = False

    def _close_resources(self) -> None:
        if self._workspace is not None:
            self._workspace.runtime.close()
        self._bootstrap_resources.close()
        self._stack.close()
        self._lifetime.close()

    async def close_async(self, deadline: float) -> bool:
        """Join native disposal without blocking the foreground loop's failure timer.

        Use only after complete drainage and any drained-workspace consumer.
        The existing single close executor performs the same ordered disposal
        as ``close``. Timeout or cancellation retains that exact Future; another
        call joins it before any retry. No submitted close is cancelled, and
        synchronous close/borrows remain excluded while it is owned here.
        """
        if type(deadline) not in {int, float} or not math.isfinite(deadline):
            raise ValueError("Close requires a finite absolute monotonic deadline")
        if self._lock.locked():
            try:
                await asyncio.wait_for(self._lock.acquire(), max(0.0, deadline - time.monotonic()))
            except TimeoutError:
                return False
        else:
            await self._lock.acquire()
        try:
            with self._resource_state:
                if self._closed:
                    return True
                if self._workspace_borrowed or self._closing and self._close_pending is None:
                    raise RuntimeError("Bench resources have an active workspace consumer or close")
                if not self._drained or self._pending is not None or (
                    self._listener_task is not None
                ):
                    raise RuntimeError("Bench resources must completely drain before close")
                if self._close_pending is None:
                    self._closing = self._close_started = True

                    def dispose() -> None:
                        self._close_thread = threading.current_thread()
                        self._close_resources()

                    try:
                        self._close_pending = self._executor.submit(_call, dispose)
                    except BaseException:
                        self._closing = False
                        raise
                pending = self._close_pending
            done, _ = await asyncio.wait(
                {asyncio.wrap_future(pending)}, timeout=max(0.0, deadline - time.monotonic()),
            )
            if not done:
                return False
            if pending.result().failed:
                with self._resource_state:
                    self._close_pending = None
                    self._closing = False
                self._fail("close")
                raise RuntimeError("Bench resource disposal did not complete") from None
            if not self._executor_stopping:
                self._executor.shutdown(wait=False)
                self._executor_stopping = True
            thread = self._close_thread
            if thread is None:
                raise RuntimeError("Bench resource disposal has no actual worker acknowledgement")
            while thread.is_alive():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                await asyncio.sleep(min(0.001, remaining))
            # The captured sole executor thread has exited. Joining now cannot
            # block on native disposal or a still-running shutdown callback.
            self._executor.shutdown(wait=True)
            with self._resource_state:
                self._close_pending = None
                self._closing = False
                self._closed = True
            return True
        finally:
            self._lock.release()
