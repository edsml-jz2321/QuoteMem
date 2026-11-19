"""ASGI adaptation of the agreed benchmark application seam from plan.md §2/§9.

The application owns authorization, source publication, finalize, and delivery
admission. A successful Search callback validates and queues its complete encoded
body while the application still owns the final delivery guard. This module never
re-reads result objects after that callback or opens a storage/provider owner.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, ExitStack, contextmanager, nullcontext
from threading import Lock
from time import monotonic, monotonic_ns
from typing import cast

import anyio
from starlette.applications import Starlette
from starlette.exceptions import HTTPException
from starlette.middleware import Middleware
from starlette.requests import ClientDisconnect, Request
from starlette.responses import Response
from starlette.routing import Route
from starlette.types import Receive, Scope, Send

from aml.adapters.http._ownership import (
    RequestOwnership,
    admit,
    discard_exception,
    observe,
    submit,
)
from aml.adapters.http.auth import HTTPCredentials
from aml.adapters.http.errors import (
    HTTPErrorCode,
    HTTPErrorResponse,
    MappedFailure,
    public_error,
    recode,
    recovery_refusal,
)
from aml.adapters.http.forget import MemoryForgetHTTP
from aml.adapters.http.memory import MemoryReadHTTP
from aml.adapters.http.models import (
    AddResponseV1,
    SearchRequestV1,
    parse_add_body,
)
from aml.adapters.http.recovery import BenchRecoveryHTTP
from aml.adapters.http.search_response import encode_search_response
from aml.api.bench import (
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
    AddDispatch,
    BenchmarkApplication,
    SearchRequest,
    SearchResult,
)
from aml.core.add_outcome import AddOutcomeV1
from aml.core.errors import AMLError
from aml.core.identity import scope_id
from aml.core.request_budget import (
    RequestBudget,
    RequestContractVersion,
    check_request_budget,
    current_budget,
)
from aml.ingestion.boundary import AddRequestV1
from aml.service.lifecycle import ServiceLifetime
from aml.service.request_measurement import (
    MeasurementCollector,
    current_request_measurement,
    measure_request,
)


class _ContentResponse(Response):
    """Release the owned response body after either send completion or cancellation."""

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            self.body = b""


def _clear_request(request: Request) -> None:
    # Starlette caches body/json on the Request itself. Route frames and error
    # tracebacks may outlive the endpoint, so clear these before ownership leaves.
    request.__dict__.pop("_body", None)
    request.__dict__.pop("_json", None)


def _mapped(error: Exception) -> HTTPErrorResponse:
    """Map a failure, crediting the time its request already waited."""
    budget = current_budget()
    return public_error(error, waited_seconds=0.0 if budget is None else budget.elapsed())


def _failure(error: Exception, *, observe_response: bool = True) -> Response:
    try:
        mapped = _mapped(error)
    finally:
        discard_exception(error)
    return _respond(mapped, observe_response=observe_response)


def _respond(mapped: HTTPErrorResponse, *, observe_response: bool = True) -> Response:
    response = _ContentResponse(
        mapped.body.model_dump_json().encode("utf-8"),
        status_code=mapped.status_code,
        headers=mapped.headers,
        media_type="application/json",
    )
    observer = current_request_measurement()
    if observe_response and observer is not None:
        observer.response("error_response_ready", mapped.status_code, monotonic_ns())
    return response


def _add_failure(error: Exception) -> Response:
    """Answer Add's temporary fences as retryable429 ``delivery_busy``.

    A whole-output extraction refusal resumes its owner automatically, so the
    Add waits the recovery delay. Maintenance and erasure keep their own delay
    under the ``delivery_busy`` code Add has always answered for them.
    """
    try:
        mapped = _mapped(error)
    finally:
        discard_exception(error)
    if mapped.body.code == "extraction_refused":
        mapped = recovery_refusal()
    elif mapped.body.code in {"index_rebuilding", "erasure_in_progress"}:
        mapped = recode(mapped, "delivery_busy")
    # Add's observer stays owned through send. The transport records an error
    # only after its work-budget check, so an unsent response and the owned
    # deadline fallback cannot both report the same response boundary.
    return _respond(mapped, observe_response=False)


_SEARCH_CODES: frozenset[HTTPErrorCode] = frozenset({
    "invalid_request", "authentication_failed", "provider_unavailable",
    "authorization_unavailable", "index_rebuilding", "delivery_busy", "service_unhealthy",
})
"""The public codes /search may answer (E.3); it never answers 409."""


def _search_failure(error: Exception) -> Response:
    try:
        mapped = _mapped(error)
    finally:
        discard_exception(error)
    if mapped.body.code in _SEARCH_CODES:
        return _respond(mapped)
    # erasure_in_progress belongs to the write side; a Search waits for the
    # visibility fence instead. No other write-side code has a Search
    # meaning, so it fails closed like any unmapped failure.
    if mapped.body.code == "erasure_in_progress":
        return _respond(recode(mapped, "index_rebuilding"))
    return _failure(AMLError("service_unhealthy", "Search failed outside its public codes"))


def _request_budget() -> RequestBudget:
    # RequestOwnership installs exactly one budget for every /add and /search request.
    budget = current_budget()
    if budget is None:
        raise RuntimeError("HTTP request budget ownership was not installed")
    return budget


async def _json_body(request: Request) -> bytes:
    content_type = request.headers.get("content-type", "").partition(";")[0].strip().lower()
    if content_type != "application/json":
        raise AMLError("invalid_request", "HTTP requests require JSON content")
    return await request.body()


class _CancellationWake:
    """Retire or join one content-free notification outside the receiver lock."""

    def __init__(self, callback: Callable[[], None]) -> None:
        self._lock = Lock()
        self._callback: Callable[[], None] | None = callback

    def invoke(self) -> None:
        with self._lock:
            callback, self._callback = self._callback, None
            if callback is not None:
                callback()

    def retire(self) -> None:
        with self._lock:
            self._callback = None


class _DeliveryReceiver:
    """Synchronize the callback's queue point with ASGI receiver closure."""

    def __init__(
        self, expected_scope: str, top_k: int, received_at: float, deadline: float,
        *, on_successful_traffic: Callable[[], None] | None = None,
    ) -> None:
        self._lock = Lock()
        self._closed = False
        self._cancelled = False
        self._cleanup_failed = False
        self._body: bytes | None = None
        self._expected_scope = expected_scope
        self._top_k = top_k
        self._received_at = received_at
        self._deadline = deadline
        self._on_successful_traffic = on_successful_traffic
        self._wakes: dict[object, _CancellationWake] = {}

    @property
    def received_at(self) -> float:
        return self._received_at

    @property
    def deadline(self) -> float:
        return self._deadline

    @property
    def cancelled(self) -> bool:
        with self._lock:
            return self._cancelled

    @contextmanager
    def on_cancel(self, wake: Callable[[], None]) -> Iterator[None]:
        token = object()
        notification = _CancellationWake(wake)
        with self._lock:
            cancelled = self._cancelled
            if not self._closed:
                self._wakes[token] = notification
        try:
            if cancelled:
                notification.invoke()
            yield
        finally:
            with self._lock:
                self._wakes.pop(token, None)
            notification.retire()

    def _check_open(self) -> None:
        deadline_reached = monotonic() >= self._deadline
        if deadline_reached:
            observer = current_request_measurement()
            if observer is not None:
                observer.timeout("delivery_deadline")
        if self._closed or deadline_reached:
            raise AMLError("delivery_busy", "The HTTP receiver is unavailable")

    def check_admission(self) -> None:
        with self._lock:
            self._check_open()

    def deliver(self, result: SearchResult) -> None:
        self.check_admission()
        if len(result.hits) > self._top_k or any(
            hit.scope_id != self._expected_scope for hit in result.hits
        ):
            raise AMLError("service_unhealthy", "Search delivery failed its exit contract")
        encoded = encode_search_response(result)
        self.enqueue(encoded)

    def enqueue(self, body: bytes) -> None:
        with self._lock:
            self._check_open()
            if self._body is not None:
                raise AMLError("service_unhealthy", "Search delivered more than one response")
            self._body = body
            enqueued_at_ns = monotonic_ns()
        # Accepted enqueue is the event, even if a later disconnect discards
        # unsent bytes. Observe synchronously inside the worker's owned lifetime,
        # after releasing the transport lock; duplicate/rejected attempts do not
        # reach this point. The composition callback must never persist or block.
        observer = current_request_measurement()
        if observer is not None:
            observer.response("search_enqueue", 200, enqueued_at_ns)
        if self._on_successful_traffic is not None:
            self._on_successful_traffic()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._cancelled = True
            self._body = None
            wakes, self._wakes = tuple(self._wakes.values()), {}
        for wake in wakes:
            try:
                wake.invoke()
            except Exception:
                # One defective observer must neither strand the remaining
                # waits nor turn a disconnected receiver into an error response.
                with self._lock:
                    self._cleanup_failed = True

    def take(self) -> bytes | None:
        with self._lock:
            self._closed = True
            self._wakes.clear()
            body, self._body = self._body, None
            if self._cleanup_failed:
                raise AMLError("service_unhealthy", "Cancellation notification cleanup failed")
            return body


class _DisconnectedResponse(_ContentResponse):
    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        # The receiver already disconnected. In particular, do not emit a new
        # error body after discarding an accepted but unsent source response.
        return None


async def _search_body(
    request: Request,
    application: BenchmarkApplication,
    admitted: SearchRequestV1,
    *,
    received_at: float,
    deadline: float,
    on_successful_traffic: Callable[[], None] | None = None,
) -> bytes:
    expected_scope = scope_id("user", admitted.user_id)
    receiver = _DeliveryReceiver(
        expected_scope, admitted.top_k, received_at, deadline,
        on_successful_traffic=on_successful_traffic,
    )
    selected: SearchRequestV1 | None = admitted
    del admitted
    failure: HTTPErrorResponse | None = None
    disconnected = False
    application_finished = False

    def search() -> None:
        # Threadpool admission is part of the same fixed response budget.
        receiver.check_admission()
        if selected is None:
            raise RuntimeError("The owned Search request was released before execution")
        application.search(
            SearchRequest(
                query=selected.query,
                user_id=selected.user_id,
                top_k=selected.top_k,
                options=selected.options,
                time_order=selected.time_order,
                as_of=selected.as_of,
            ),
            receiver,
        )

    try:
        receiver.check_admission()
        async with anyio.create_task_group() as group:

            async def run() -> None:
                nonlocal failure, application_finished
                try:
                    await submit(search)
                except Exception as error:
                    try:
                        failure = _mapped(error)
                    finally:
                        discard_exception(error)
                finally:
                    application_finished = True
                    group.cancel_scope.cancel()

            async def watch_disconnect() -> None:
                nonlocal disconnected
                try:
                    while True:
                        if (await request.receive())["type"] == "http.disconnect":
                            disconnected = True
                            receiver.close()
                            group.cancel_scope.cancel()
                            return
                finally:
                    # On external cancellation this monitor closes admission
                    # while the task group still joins its shielded worker.
                    if not application_finished:
                        receiver.close()

            group.start_soon(watch_disconnect)
            group.start_soon(run)
        if disconnected:
            raise ClientDisconnect
        body = receiver.take()
        if body is not None:
            # Enqueue is the accepted delivery point. A subsequent application
            # exception cannot replace or reconstruct the queued public body.
            return body
        if failure is not None:
            raise MappedFailure(failure)
        raise AMLError("service_unhealthy", "Search did not deliver a response")
    finally:
        receiver.close()
        selected = None


def create_app(
    application: BenchmarkApplication,
    credentials: HTTPCredentials,
    *,
    max_top_k: int,
    search_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    request_contract_version: RequestContractVersion = 2,
    on_successful_traffic: Callable[[], None] | None = None,
    add_reception: (
        Callable[[AddRequestV1], AbstractContextManager[AddDispatch | None]] | None
    ) = None,
    worker_lifetime: ServiceLifetime | None = None,
    measurement: MeasurementCollector | None = None,
    recovery: BenchRecoveryHTTP | None = None,
    memory_forget: MemoryForgetHTTP | None = None,
    memory_reads: MemoryReadHTTP | None = None,
) -> Starlette:
    """Construct HTTP handlers over an explicitly supplied application owner.

    Args:
        application: The agreed use-case seam; Add returns only after durable
            publication, and Search invokes its delivery callback under its
            final guard. Test doubles establish transport behavior only.
        credentials: Exactly one selected deployment authentication mode.
        max_top_k: The immutable active RecallPolicy.max_top_k for this instance.
        search_timeout_seconds: Application-selected complete request budget for POST
            /add and POST /search, default 1740 seconds, measured from initial ASGI
            request ownership through actual send; body receipt, parsing, threadpool
            queueing, application work and send all spend it. It is AML policy,
            not an inferred platform timeout.
        request_contract_version: The current contract bounds both endpoints from
            initial ASGI request ownership through actual send. The application
            separately guarantees completed-prefix Add acknowledgement.
        on_successful_traffic: Optional application-owned, argument-free observer
            called once after durable Add succeeds or Search enqueue is accepted.
            Empty HTTP-only Add acknowledgements do not record durable traffic.
            It must be content-free, nonblocking, non-throwing and perform no HB,
            provider or other I/O. Supply the bound job's in-memory observation
            from the production composition; omission supports component callers
            and does not establish a complete benchmark runtime. The callback
            runs synchronously in the owned application worker, outside transport
            locks, and is included in request-worker draining. Later Search
            failure or disconnect cannot retract an accepted traffic event.
        add_reception: Optional application-owned reception reservation for a
            validated Add. Production enters it before worker submission and
            retains it until the existing joined worker has actually finished.
            The context performs no persistence or provider I/O and carries no
            source bytes. Omission supports isolated transport applications.
        worker_lifetime: The exact same ServiceLifetime used by the real
            application. Admission starts before body reading/threadpool waiting
            and ends after complete call, callback and response cleanup. Omit
            only for component callers; no substitute lifecycle is constructed.
            Existing application admission remains and verifies readiness again.
        measurement: Optional bounded, content-free collector selected before
            the measured load. It observes complete request receipt, actual
            worker admission and response events without changing HTTP output.
            Add response readiness is distinct from transport enqueue.
        recovery: Explicit provider recovery over the existing controller and
            request lifetime. Its acknowledgement means durable resume only.
        memory_forget: Explicit selective erasure preparation and confirmation
            through the same authenticated application and request lifetime.
        memory_reads: The AML memory reads (E.2): ``POST /memory/history``, and
            ``/memory/why`` and ``/memory/add-items`` when their readers are
            wired, under the same credentials, request ownership and the
            Search delivery budget.

    Returns:
        An ASGI application with POST /add, POST /search, and unauthenticated
        GET /health. Construction starts no listener or persistence lifecycle.
        The composition root owns application shutdown and process resources.

    Raises:
        AMLError: ``schema_invalid`` if trusted transport configuration is invalid.
    """
    if type(max_top_k) is not int or not 1 <= max_top_k <= 1000:
        raise AMLError("schema_invalid", "The configured recall limit is invalid")
    if type(request_contract_version) is not int or request_contract_version != 2:
        raise AMLError("schema_invalid", "The request contract version is invalid")
    if type(search_timeout_seconds) not in (float, int):
        raise AMLError("schema_invalid", "The Search response budget is invalid")
    try:
        search_timeout_seconds = float(search_timeout_seconds)
    except OverflowError:
        raise AMLError("schema_invalid", "The Search response budget is invalid") from None
    if not math.isfinite(search_timeout_seconds) or not (
        0 < search_timeout_seconds <= DEFAULT_REQUEST_TIMEOUT_SECONDS
    ):
        raise AMLError("schema_invalid", "The Search response budget is invalid")
    if on_successful_traffic is not None and not callable(on_successful_traffic):
        raise AMLError("schema_invalid", "The traffic observation callback is invalid")
    if worker_lifetime is not None and not isinstance(
        cast(object, worker_lifetime), ServiceLifetime
    ):
        raise AMLError("schema_invalid", "The HTTP worker lifetime is invalid")
    if measurement is not None and not isinstance(cast(object, measurement), MeasurementCollector):
        raise AMLError("schema_invalid", "The request measurement collector is invalid")

    async def add(request: Request) -> Response:
        admitted: AddRequestV1 | None = None
        raw: bytes | None = None
        try:
            credentials.verify(request.headers.raw)
            admit(request.scope)
            budget = _request_budget()
            raw = await _json_body(request)
            budget.identify("add", hashlib.sha256(raw).hexdigest())
            received_at_ns = monotonic_ns()
            observer = None if measurement is None else measurement.begin(
                "add", hashlib.sha256(raw).hexdigest(), received_at_ns,
            )
            if observer is not None:
                observe(request.scope, observer)
            admitted = parse_add_body(raw)
            check_request_budget()
            if observer is not None:
                observer.scope(scope_id("user", admitted.user_id))

            outcome: AddOutcomeV1 | None = None

            def persist_add() -> None:
                nonlocal outcome
                if admitted is None:
                    raise RuntimeError("The owned Add request was released before execution")
                # Dropped blank turns have no source to admit or persist. Their
                # deterministic zero result is identical on every exact replay.
                outcome = application.add(admitted) if admitted.messages else AddOutcomeV1()
                if not isinstance(cast(object, outcome), AddOutcomeV1):
                    raise AMLError("incomplete_coverage", "Add returned no completed result")
                if admitted.messages and on_successful_traffic is not None:
                    on_successful_traffic()

            disconnected = False
            failure: HTTPErrorResponse | None = None
            with (
                nullcontext() if add_reception is None or not admitted.messages
                else add_reception(admitted)
            ) as dispatch:
                async with anyio.create_task_group() as group:
                    async def run_add() -> None:
                        nonlocal failure

                        async def joined_phase(operation: Callable[[], bool | None]) -> bool | None:
                            await anyio.lowlevel.checkpoint()
                            check_request_budget()
                            result = await submit(operation)
                            # submit joins under its own shield. Deliver pending
                            # parent cancellation before starting another phase.
                            await anyio.lowlevel.checkpoint()
                            check_request_budget()
                            return result

                        try:
                            if dispatch is not None:
                                while not await joined_phase(dispatch.step):
                                    await dispatch.wait_for_change()
                                while not await joined_phase(dispatch.completion_ready):
                                    await dispatch.wait_for_change()
                            await joined_phase(persist_add)
                        except Exception as error:
                            try:
                                failure = _mapped(error)
                            finally:
                                discard_exception(error)
                        finally:
                            group.cancel_scope.cancel()

                    async def watch_add_disconnect() -> None:
                        nonlocal disconnected
                        try:
                            while True:
                                if (await request.receive())["type"] == "http.disconnect":
                                    disconnected = True
                                    budget.cancel()
                                    group.cancel_scope.cancel()
                                    return
                        except AMLError:
                            if not budget.cancelled:
                                raise
                            group.cancel_scope.cancel()

                    group.start_soon(watch_add_disconnect)
                    group.start_soon(run_add)
            if disconnected:
                raise ClientDisconnect
            if failure is not None:
                raise MappedFailure(failure)
            check_request_budget()
            if outcome is None:
                raise AMLError("incomplete_coverage", "Add returned no completed result")
            response = _ContentResponse(
                AddResponseV1.from_request(admitted, outcome).model_dump_json().encode("utf-8"),
                media_type="application/json",
            )
            if observer is not None:
                observer.add_response_ready(response.body, monotonic_ns())
            check_request_budget()
            return response
        except ClientDisconnect as error:
            discard_exception(error)
            return _DisconnectedResponse()
        except Exception as error:
            return _add_failure(error)
        finally:
            raw = admitted = None
            _clear_request(request)

    async def search(request: Request) -> Response:
        raw: bytes | None = None
        admitted: SearchRequestV1 | None = None
        body: bytes | None = None
        observations = ExitStack()
        try:
            credentials.verify(request.headers.raw)
            admit(request.scope)
            budget = _request_budget()
            raw = await _json_body(request)
            received_at_ns = monotonic_ns()
            budget.identify("search", hashlib.sha256(raw).hexdigest())
            observer = None if measurement is None else measurement.begin(
                "search", hashlib.sha256(raw).hexdigest(), received_at_ns,
            )
            if observer is not None:
                observations.enter_context(measure_request(observer))
            admitted = SearchRequestV1.from_body(raw, max_top_k=max_top_k)
            if observer is not None:
                observer.scope(scope_id("user", admitted.user_id))
            body = await _search_body(
                request,
                application,
                admitted,
                received_at=budget.started_at,
                deadline=budget.deadline,
                on_successful_traffic=on_successful_traffic,
            )
            return _ContentResponse(body, media_type="application/json")
        except ClientDisconnect as error:
            discard_exception(error)
            return _DisconnectedResponse()
        except Exception as error:
            return _search_failure(error)
        finally:
            raw = admitted = body = None
            _clear_request(request)
            observations.close()

    async def health(request: Request) -> Response:
        try:
            admit(request.scope)
            if await request.body():
                raise AMLError("invalid_request", "The health request must have no body")
            if not await submit(application.health):
                raise AMLError("service_unhealthy", "The application cannot accept traffic")
            return _ContentResponse(b"{}", media_type="application/json")
        except Exception as error:
            return _failure(error)
        finally:
            _clear_request(request)

    async def routing_failure(request: Request, error: Exception) -> Response:
        if request.method == "HEAD" and request.url.path in {"/add", "/search"}:
            # Some clients probe both routes with HEAD.
            return Response(status_code=405, headers={"Allow": "POST"})
        return _failure(AMLError("invalid_request", "The HTTP route or method is invalid"))

    app = Starlette(
        routes=[
            Route("/add", add, methods=["POST"]),
            Route("/search", search, methods=["POST"]),
            Route("/health", health, methods=["GET"]),
            *([] if recovery is None else recovery.routes(credentials)),
            *([] if memory_forget is None else memory_forget.routes(credentials)),
            *([] if memory_reads is None else memory_reads.routes(
                credentials, timeout_seconds=search_timeout_seconds,
            )),
        ],
        exception_handlers={HTTPException: routing_failure},
        middleware=[Middleware(
            RequestOwnership, lifetime=worker_lifetime,
            contract_version=request_contract_version,
            request_timeout_seconds=search_timeout_seconds,
            observe_budget=None if measurement is None else measurement.observe_budget,
        )],
    )
    app.router.redirect_slashes = False
    if measurement is not None:
        # The selected listener reads this exact callback before Starlette wraps
        # send. Other ASGI transports retain readiness as an explicit nonpass.
        app.state.aml_add_transport_measurement = current_request_measurement
    return app
