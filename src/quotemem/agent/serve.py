"""``quotemem serve``: compose and run one agent memory service.

Order of a start:

1. Resolve the provider endpoints, the data directory, the store binding and the
   token (``quotemem.agent.settings``). Configuration errors stop here, naming the
   variable to fix, and so does a store another process is using.
2. Bind the HTTP port at once with only ``AgentGateway`` behind it and probe both
   providers, retrying with backoff (1 s doubling to 60 s) until they answer or a
   signal arrives. Meanwhile ``/agent/status`` explains what fails and the memory
   routes answer 503 with ``Retry-After``, so a plugin keeps chatting without memory.
3. Open the store with the bench composition over the routing clients, on a worker
   thread, while the same listener keeps answering 503; then the opened runtime
   takes that listener over. The port is never closed between the phases, so a
   client never sees a refused connection (closing the probing listener before the
   open would refuse connections for as long as the store takes to open).

A provider that fails after startup is handled by the core: Adds wait durably and
resume, and ``/agent/status`` shows each role's latest failure.

Data controls are composed here too: the daily spend guard (``quotemem.agent.spend``),
PII scrubbing (``QUOTEMEM_SCRUB_PII``, ``quotemem.agent.pii``), the operator routes and
store details (``quotemem.agent.admin``) and the opt-in automatic quarantine repair
(``QUOTEMEM_AUTO_REPAIR=resume|skip``), which logs one line per action.

``check`` resolves the same configuration read-only: it creates, binds and writes
nothing, and probes the providers once.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import signal
import sys
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import httpx
from starlette.types import ASGIApp

from quotemem.adapters.http.auth import HTTPCredentials
from quotemem.agent.admin import AgentStore, RepairPolicy, repair_policy
from quotemem.agent.gateway import AgentGateway, AgentState
from quotemem.agent.pii import scrub_enabled
from quotemem.agent.probe import ProbeResult, probe_providers
from quotemem.agent.routing import ProviderRoutes, RoutedAsyncClient, RoutedClient
from quotemem.agent.settings import (
    AGENT_JOB_ID,
    LOOPBACK_HOSTS,
    ROUTED_ENVIRONMENT,
    bind_store,
    describe_binding,
    inspect_store,
    load_token,
    prepare_data_dir,
    resolve_endpoints,
    resolve_review,
    store_provider_config,
    token_source,
)
from quotemem.agent.spend import SPEND_FILE, SpendGuard, SpendLimits
from quotemem.api.bench import DEFAULT_REQUEST_TIMEOUT_SECONDS
from quotemem.api.interactive_search import InteractiveSearch, SearchProfile
from quotemem.core.errors import QuoteMemError
from quotemem.core.extraction_policy import AddMode, ReviewMode
from quotemem.core.platform_files import exclusive_owner, own_new_files
from quotemem.core.request_budget import DEFAULT_REQUEST_CONTRACT_VERSION
from quotemem.providers.config import DEFAULT_ADD_MODE, ProviderConfig
from quotemem.service.heap_policy import limit_malloc_arenas
from quotemem.service.listener import UvicornListener
from quotemem.service.runtime import BenchRuntime, Concurrency

DEFAULT_PORT = 8765
AUTO_REPAIR_INTERVAL_SECONDS = 30.0
DEFAULT_REQUEST_TIMEOUT = 600.0
"""Seconds one Add or Search may take end to end; a plugin applies shorter deadlines."""
MAX_PROBE_BACKOFF_SECONDS = 60.0
STORE_IN_USE = ("Another QuoteMem process is using this data directory; stop it, or give "
                "this service another --data-dir")


@dataclass(frozen=True)
class AgentServeOptions:
    """Nonsecret options of one agent service.

    Attributes:
        data_dir: The agent data directory.
        host: The listen address; non-loopback requires ``allow_remote``.
        port: The listen port; 0 picks a free one, reported on stdout.
        default_user_id: The user of requests that name none.
        default_top_k: The Search result count when a request names none.
        request_timeout_seconds: The budget of one Add or Search.
        dimensions: The vector dimension of a new store; None keeps the store's.
        add_mode: The Add mode of a new store (None: ``agent``); None keeps an existing store's.
        review: The review of a new ``agent`` store (``--review``; None reads
            ``QUOTEMEM_REVIEW``, then ``standard``); an existing store keeps its own and
            refuses another (``settings.store_provider_config``).
        allow_remote: Permit a non-loopback listen address.
        search_profile: How Search answers; ``interactive`` (the default, a plugin's
            profile: no model call besides the query embedding, compact hits, a deadline
            with a keyword-only fallback) or ``bench``.
        search_deadline_seconds: The interactive profile's Search deadline.
        concurrency: ``interactive`` (the default: Search never waits for an Add, per-user
            delivery queues, recall rows written after the response) or ``bench``.
    """

    data_dir: Path
    host: str = "127.0.0.1"
    port: int = DEFAULT_PORT
    default_user_id: str = "default"
    default_top_k: int = 10
    request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT
    dimensions: Literal[1024, 2048] | None = None
    add_mode: AddMode | None = None
    review: ReviewMode | None = None
    allow_remote: bool = False
    search_profile: SearchProfile = "interactive"
    search_deadline_seconds: float = 2.5
    concurrency: Concurrency = "interactive"


@dataclass(frozen=True)
class AgentService:
    """A composed agent service before it serves."""

    options: AgentServeOptions
    workspace: Path
    config: ProviderConfig
    routes: ProviderRoutes
    credentials: HTTPCredentials
    state: AgentState
    gateway: AgentGateway
    store: AgentStore
    repair_policy: RepairPolicy = "off"


def notice(value: dict[str, object], *, error: bool = False) -> None:
    """Write one content-free JSON status line."""
    try:
        print(json.dumps(value, sort_keys=True), file=sys.stderr if error else sys.stdout,
              flush=True)
    except (OSError, ValueError):
        pass


def compose(options: AgentServeOptions, environment: Mapping[str, str]) -> AgentService:
    """Resolve and bind everything the service needs, without network access.

    Raises:
        QuoteMemError: For configuration that cannot serve; the message names the fix.
            ``workspace_locked`` (`STORE_IN_USE`) for a store another process holds,
            before its binding or token is written.
        OSError: When the data directory cannot be made private.
    """
    if not options.host or not 0 <= options.port <= 65535:
        raise QuoteMemError("schema_invalid", "The listen address is invalid")
    if options.host not in LOOPBACK_HOSTS and not options.allow_remote:
        raise QuoteMemError("schema_invalid", "A non-loopback --host requires --allow-remote")
    if not 0 < options.request_timeout_seconds <= DEFAULT_REQUEST_TIMEOUT_SECONDS:
        raise QuoteMemError("schema_invalid", "The request timeout is out of range")
    if not 1 <= options.default_top_k <= 100 or not options.default_user_id:
        raise QuoteMemError("schema_invalid", "The Search defaults are invalid")
    if not (math.isfinite(options.search_deadline_seconds)
            and 0 < options.search_deadline_seconds <= options.request_timeout_seconds):
        raise QuoteMemError("schema_invalid", "The Search deadline is out of range")
    chat, embedding = resolve_endpoints(environment)
    limits = SpendLimits.from_environment(environment)
    policy = repair_policy(environment.get("QUOTEMEM_AUTO_REPAIR", ""))
    scrub = scrub_enabled(environment)
    review = resolve_review(options.review, environment)
    own_new_files()
    workspace = prepare_data_dir(options.data_dir)
    # Before the binding or the token is written: a refused start writes nothing.
    require_store_free(workspace)
    config = store_provider_config(
        workspace, dimensions=options.dimensions, add_mode=options.add_mode, review=review,
    )
    binding = bind_store(
        options.data_dir, embedding_model=embedding.model,
        dimensions=config.embedding_dimensions, chat_model=chat.model,
    )
    token, token_source = load_token(options.data_dir, environment)
    credentials = HTTPCredentials("bearer", token)
    spend = SpendGuard(limits, options.data_dir / SPEND_FILE) if limits.capped else None
    routes = ProviderRoutes(chat, embedding, config.embedding_dimensions, spend=spend, scrub=scrub)
    state = AgentState(
        store={
            **describe_binding(binding), "data_dir": str(options.data_dir),
            "add_mode": config.add_mode or DEFAULT_ADD_MODE, "review": config.review_mode,
            "token_source": token_source,
        },
        health=routes.health,
        endpoints={
            "chat": {"base_url": chat.base_url, "model": chat.model,
                     "structured_output": chat.structured_output,
                     "reasoning": chat.reasoning,
                     **({"reasoning_room": chat.reasoning_room,
                         "reasoning_wait": chat.reasoning_wait}
                        if chat.reasoning == "high" else {})},
            "embedding": {"base_url": embedding.base_url, "model": embedding.model,
                          "send_dimensions": embedding.send_dimensions},
        },
        controls={"scrub_pii": scrub, "auto_repair": policy,
                  "spend_capped": limits.capped},
    )
    store = AgentStore()
    gateway = AgentGateway(
        credentials, state, default_user_id=options.default_user_id,
        default_top_k=options.default_top_k, store=store, spend=spend,
        self_identity=config.self_identity,
    )
    return AgentService(options, workspace, config, routes, credentials, state, gateway,
                        store, policy)


def check(
    options: AgentServeOptions, environment: Mapping[str, str], *,
    transport: httpx.BaseTransport | None = None,
) -> dict[str, object]:
    """Resolve the service's configuration and probe the providers, read-only.

    Nothing is created or written: no data directory, binding, token or spend
    ledger. Configuration errors raise exactly as ``compose`` would, and so does a
    Windows token under which the store's new files would not be owned by the user.

    Returns:
        The report: ``ok``, the failing role and hint, the store and providers.
    """
    if options.host not in LOOPBACK_HOSTS and not options.allow_remote:
        raise QuoteMemError("schema_invalid", "A non-loopback --host requires --allow-remote")
    chat, embedding = resolve_endpoints(environment)
    limits = SpendLimits.from_environment(environment)
    policy = repair_policy(environment.get("QUOTEMEM_AUTO_REPAIR", ""))
    own_new_files()
    store, dimensions = inspect_store(
        options.data_dir, embedding_model=embedding.model, dimensions=options.dimensions,
        add_mode=options.add_mode, review=resolve_review(options.review, environment),
    )
    store["token_source"] = token_source(options.data_dir, environment)
    routes = ProviderRoutes(chat, embedding, dimensions, scrub=scrub_enabled(environment))
    with RoutedClient(routes, transport=transport) as client:
        result = probe_providers(client, routes)
    providers = routes.health.snapshot()
    ok, role, code, hint = result.ok, result.role, result.code, result.hint
    refused = [name for name, record in providers.items() if record.get("model_mismatch")]
    if ok and refused:
        # Never report ok over a reply the routing refused, however the probe read it.
        ok, role, code = False, refused[0], "model_mismatch"
        hint = (f"The {refused[0]} provider answered as another model; see "
                "/agent/status providers or set the model or its pattern")
    return {
        "status": "checked", "ok": ok, "role": role, "code": code,
        "hint": hint, "provider_code": result.provider_code, "store": store,
        "controls": {"scrub_pii": routes.scrub, "auto_repair": policy,
                     "spend": limits.describe() if limits.capped else None},
        "providers": {
            "chat": {"base_url": chat.base_url, "model": chat.model, **providers["chat"]},
            "embedding": {"base_url": embedding.base_url, "model": embedding.model,
                          **providers["embedding"]},
        },
    }


class _AutoRepair(threading.Thread):
    """Every interval: sweep the forget intents that expired unconfirmed, keeping only their
    binding (``AgentStore.expire_forget_intents``), and apply the opt-in quarantine policy;
    one log line per repair action."""

    def __init__(self, store: AgentStore, policy: RepairPolicy, interval: float) -> None:
        super().__init__(name="quotemem-auto-repair", daemon=True)
        self._store = store
        self._policy: RepairPolicy = policy
        self._interval = interval
        self.stopped = threading.Event()

    def run(self) -> None:
        while not self.stopped.wait(self._interval):
            try:
                self._store.expire_forget_intents()
            except (QuoteMemError, RuntimeError) as error:
                notice({"status": "forget_intent_sweep", "ok": False,
                        "code": getattr(error, "code", "service_unhealthy")}, error=True)
            try:
                actions = self._store.auto_repair(self._policy)
            except QuoteMemError as error:
                actions = [{"ok": False, "code": error.code}]
            for action in actions:
                notice({"status": "auto_repair", "policy": self._policy, **{
                    key: value for key, value in action.items() if key != "user_id"
                }, "scope": "user"}, error=not action.get("ok"))


async def bind_gateway(service: AgentService) -> UvicornListener:
    """Bind the port with only the gateway behind it; every memory route answers 503.

    The same listener serves through the probing and opening phases and is then
    taken over by the opened runtime, so the port never refuses a connection.

    Raises:
        QuoteMemError: When the port cannot be bound within 30 seconds.
    """
    options = service.options
    listener = UvicornListener(service.gateway, host=options.host, port=options.port)
    try:
        await listener.start(time.monotonic() + 30)
    except BaseException:
        await drain_listener(listener)
        raise
    return listener


async def drain_listener(listener: UvicornListener) -> bool:
    """Withdraw and join a listener the runtime has not taken over; False on timeout."""
    listener.withdraw()
    deadline = time.monotonic() + 10
    while not await listener.drain(deadline):
        if time.monotonic() >= deadline:
            return False
        await asyncio.sleep(0.05)
    return True


async def wait_for_providers(
    service: AgentService, client: httpx.Client, *,
    probe: Callable[[httpx.Client, ProviderRoutes], ProbeResult] = probe_providers,
    first_backoff_seconds: float = 1.0,
) -> bool:
    """Probe both providers, with backoff, until they answer or a signal arrives.

    The caller keeps the gateway listener bound meanwhile (`bind_gateway`).

    Returns:
        True once both providers answered; False when a signal stopped the wait.
    """
    loop = asyncio.get_running_loop()
    stop = asyncio.Event()
    installed: list[signal.Signals] = []
    previous = {signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)}
    for signum in previous:
        if os.name == "nt":
            signal.signal(signum, lambda _signum, _frame: loop.call_soon_threadsafe(stop.set))
        else:
            loop.add_signal_handler(signum, stop.set)
        installed.append(signum)
    try:
        backoff = first_backoff_seconds
        while not stop.is_set():
            result = await asyncio.to_thread(probe, client, service.routes)
            service.state.probed(ok=result.ok, role=result.role, code=result.code,
                                 hint=result.hint)
            if result.ok:
                return True
            notice({"status": "providers_unavailable", "role": result.role,
                    "code": result.code, "hint": result.hint,
                    "retry_seconds": backoff}, error=True)
            try:
                await asyncio.wait_for(stop.wait(), timeout=backoff)
            except TimeoutError:
                pass
            backoff = min(backoff * 2, MAX_PROBE_BACKOFF_SECONDS)
        return False
    finally:
        for signum in installed:
            if os.name != "nt":
                loop.remove_signal_handler(signum)
            signal.signal(signum, previous[signum])


def agent_runtime(
    service: AgentService, client: httpx.Client, *,
    async_transport: httpx.AsyncBaseTransport | None = None,
) -> BenchRuntime:
    """The bench runtime over the routing clients, its controller captured by the store.

    It serves the options' Search profile and concurrency, by default the plugin's:
    the interactive Search profile and interactive concurrency. Its background Add
    completions are counted in the status (``adds.completing``), held while the daily
    spend cap is reached and read the store's deletions, results and erase epochs
    (``service/add_followup.py``).
    """
    from quotemem.api.cli import bench_runtime, serving_controller_factory

    options = service.options
    runtime = bench_runtime(
        service.workspace, service.config, ROUTED_ENVIRONMENT, job_id=AGENT_JOB_ID,
        credentials=service.credentials,
        search_timeout_seconds=service.options.request_timeout_seconds,
        request_contract_version=DEFAULT_REQUEST_CONTRACT_VERSION, client=client,
        async_client_factory=lambda: RoutedAsyncClient(service.routes, transport=async_transport),
        measurement=None, inspection=None,
        controller_factory=service.store.capturing(serving_controller_factory()),
        interactive=InteractiveSearch.within(options.search_deadline_seconds)
        if options.search_profile == "interactive" else None,
        concurrency=options.concurrency,
    )
    service.state.followups_running = runtime.followups_running
    service.state.add_forgets = runtime.add_forgets
    if service.gateway.spend is not None:
        runtime.followup_hold = service.gateway.spend.exhausted

    runtime.followup_store = service.store

    def overtaken() -> dict[str, int]:
        followups = runtime.followups
        return {"readmitted": 0, "unstored": 0} if followups is None else followups.readmissions()

    def refused() -> list[tuple[str, str, str]]:
        followups = runtime.followups
        return [] if followups is None else followups.refused()

    service.state.followups_overtaken = overtaken
    service.state.followups_refused = refused
    return runtime


def agent_front(
    service: AgentService, runtime: BenchRuntime, *,
    repair_interval_seconds: float = AUTO_REPAIR_INTERVAL_SECONDS,
) -> tuple[Callable[[ASGIApp], ASGIApp], _AutoRepair | None]:
    """The ``wrap_app`` that binds the store to the started runtime and the gateway.

    Returns:
        (the wrapper, the housekeeping thread it starts: the forget intent sweep, and the
        automatic repair unless that is off). The caller sets the thread's ``stopped`` once
        the runtime stops.
    """
    repair = _AutoRepair(service.store, service.repair_policy, repair_interval_seconds)

    def front(app: ASGIApp) -> ASGIApp:
        service.store.attach(runtime.lifetime)
        repair.start()
        return service.gateway.serving(app)

    return front, repair


async def serve_agent(
    service: AgentService, client: httpx.Client, open_runtime: Callable[[], BenchRuntime], *,
    probe: Callable[[httpx.Client, ProviderRoutes], ProbeResult] = probe_providers,
    first_backoff_seconds: float = 1.0,
) -> int:
    """Bind, probe, open and serve on one port; the process exit code.

    The port is bound first and stays bound: while the providers are probed and the
    store opens, ``/memory/add``, ``/memory/search`` and the other memory routes answer 503
    with ``Retry-After``, and ``/agent/ready`` and ``/agent/status`` answer at once.

    Args:
        service: The composed service.
        client: The provider client the probes and the runtime share.
        open_runtime: Builds the (unstarted) runtime once the providers answer.
        probe: The provider probe (tests replace it).
        first_backoff_seconds: The first wait after a failed probe.
    """
    from quotemem.api.cli import serve_bench

    listener = await bind_gateway(service)
    port = listener.port
    notice({"status": "probing", "port": port})
    try:
        ready = await wait_for_providers(
            service, client, probe=probe, first_backoff_seconds=first_backoff_seconds,
        )
        runtime = open_runtime() if ready else None
    except BaseException:
        await drain_listener(listener)
        raise
    if runtime is None:
        return 0 if await drain_listener(listener) else 1
    service.state.enter("opening")
    notice({"status": "opening", "port": port})
    front, repair = agent_front(service, runtime)
    try:
        return await serve_bench(
            runtime, host=service.options.host, port=port, wrap_app=front,
            listener=listener, on_failure=explain_failure,
        )
    finally:
        if repair is not None:
            repair.stopped.set()


def run_agent(
    options: AgentServeOptions, environment: Mapping[str, str], *,
    transport: httpx.BaseTransport | None = None,
    async_transport: httpx.AsyncBaseTransport | None = None,
) -> int:
    """Run one agent service until a signal; the process exit code.

    Args:
        options: The service options.
        environment: Provider and token variables.
        transport: A provider transport replacing the network (tests).
        async_transport: Its asynchronous counterpart.
    """
    limit_malloc_arenas()
    service = compose(options, environment)
    notice({"status": "agent_configured", "data_dir": str(options.data_dir),
            "token_source": service.state.store["token_source"],
            "chat_model": service.routes.chat.model,
            "embedding_model": service.routes.embedding.model,
            "dimensions": service.config.embedding_dimensions,
            "review": service.config.review_mode})
    with RoutedClient(service.routes, transport=transport) as client:
        return asyncio.run(serve_agent(
            service, client,
            lambda: agent_runtime(service, client, async_transport=async_transport),
        ))


def require_store_free(workspace: Path) -> None:
    """Refuse at once a store another process holds (`compose` calls it).

    The service's own storage owner takes the same lock when the store opens; this
    early look names the cause before anything is written, the port is bound or a
    provider is called.

    Raises:
        QuoteMemError: ``workspace_locked`` with `STORE_IN_USE`.
    """
    try:
        with exclusive_owner(workspace):
            pass
    except QuoteMemError as error:
        if error.code == "workspace_locked":
            raise QuoteMemError("workspace_locked", STORE_IN_USE) from None
        raise


def _store_locked(error: BaseException) -> bool:
    """Whether the storage owner failed because another process holds the store's lock:
    an `OwnershipError` caused by the lock's `BlockingIOError` (as `read_offline` reads it)."""
    from quotemem.service.ownership import OwnershipError

    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        if isinstance(current, OwnershipError) and isinstance(current.__cause__, BlockingIOError):
            return True
        seen.add(id(current))
        current = current.__cause__
    return False


def explain_failure(error: BaseException) -> None:
    """Explain a store that cannot open: one in use by another process, or one this
    release cannot open (``quotemem rebuild``)."""
    from quotemem.agent.rebuild import describe_upgrade

    if _store_locked(error):
        notice({"status": "failed", "code": "workspace_locked", "message": STORE_IN_USE},
               error=True)
    elif isinstance(error, QuoteMemError) and (hint := describe_upgrade(error.code)) is not None:
        notice({"status": "store_unopenable", "code": error.code, **hint}, error=True)
