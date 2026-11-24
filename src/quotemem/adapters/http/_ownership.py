"""Keep transport submissions and content alive only inside their actual owner."""

from __future__ import annotations

import asyncio
import time
import traceback
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass
from threading import Event, Lock
from typing import Literal, cast

import anyio
from starlette.concurrency import run_in_threadpool
from starlette.responses import Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from quotemem.adapters.http.errors import HTTPErrorResponse, MappedFailure, public_error
from quotemem.core.errors import QuoteMemError
from quotemem.core.request_budget import (
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
    RequestBudget,
    RequestBudgetObservation,
    RequestContractVersion,
    check_request_budget,
    current_budget,
    request_budget,
)
from quotemem.core.timing import RequestTiming
from quotemem.providers.adapters import ProviderError
from quotemem.service.lifecycle import ServiceLifetime
from quotemem.service.request_measurement import (
    RequestMeasurementOwner,
    current_request_measurement,
    measure_request,
)

type _Stop = Literal[
    "cancelled", "keyboard_interrupt", "system_exit", "generator_exit", "budget_rejected", "failed",
]
_SCOPE_KEY = "aml.http.ownership"
_RESPONSE_SEND_RESERVE_SECONDS = 2.0
"""Time inside the receipt deadline for a retryable refusal and owned send cleanup."""


def discard_exception(error: BaseException) -> None:
    """Release completed frames and chained errors without forwarding their payloads."""
    pending = [error]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        pending.extend(
            linked for linked in (current.__cause__, current.__context__) if linked is not None
        )
        if isinstance(current, BaseExceptionGroup):
            pending.extend(cast(BaseExceptionGroup[BaseException], current).exceptions)
        frames = current.__traceback__
        current.__traceback__ = None
        current.__cause__ = None
        current.__context__ = None
        if frames is not None:
            traceback.clear_frames(frames)


def _stop_kind(error: BaseException) -> _Stop:
    if isinstance(error, asyncio.CancelledError):
        return "cancelled"
    if isinstance(error, KeyboardInterrupt):
        return "keyboard_interrupt"
    if isinstance(error, SystemExit):
        return "system_exit"
    if isinstance(error, GeneratorExit):
        return "generator_exit"
    return "failed"


def _raise_stop(kind: _Stop) -> None:
    if kind == "cancelled":
        raise asyncio.CancelledError() from None
    if kind == "keyboard_interrupt":
        raise KeyboardInterrupt() from None
    if kind == "system_exit":
        raise SystemExit(1) from None
    if kind == "generator_exit":
        raise GeneratorExit() from None
    if kind == "budget_rejected":
        raise QuoteMemError("delivery_busy",
                            "The request budget no longer permits delivery") from None
    raise RuntimeError("The HTTP operation terminated unexpectedly") from None


async def _send_stopping_refusal(
    scope: Scope, receive: Receive, send: Send, budget: RequestBudget,
    response_deadline: float,
) -> None:
    """Send a retryable 429 for a request the service's stop cut.

    Uvicorn answers a cancelled request 500, which AMLB retries for under a minute. A
    service restart can take far longer; a 429 waits out Retry-After.
    """
    await _send_refusal(
        scope, receive, send, budget, response_deadline,
        public_error(QuoteMemError("index_rebuilding", "The service is stopping")),
    )


async def _send_budget_refusal(
    scope: Scope, receive: Receive, send: Send, budget: RequestBudget,
    response_deadline: float,
) -> None:
    """Send the closed 429 inside the receipt allowance, retaining sender ownership."""
    await _send_refusal(
        scope, receive, send, budget, response_deadline,
        public_error(
            QuoteMemError("delivery_busy", "Request delivery budget expired"),
            waited_seconds=budget.elapsed(),
        ),
    )


async def _send_refusal(
    scope: Scope, receive: Receive, send: Send, budget: RequestBudget,
    response_deadline: float, mapped: HTTPErrorResponse,
) -> None:
    remaining = response_deadline - asyncio.get_running_loop().time()
    if remaining <= 0:
        raise QuoteMemError("delivery_busy", "No response allowance remains")
    response = Response(
        mapped.body.model_dump_json().encode("utf-8"),
        status_code=mapped.status_code,
        headers={**mapped.headers, "X-AML-Request-Contract": "2"},
        media_type="application/json",
    )
    if (observer := current_request_measurement()) is not None:
        if time.monotonic() >= budget.deadline:
            observer.timeout("delivery_deadline")
        # An earlier 200 readiness is invalidated rather than counted as a
        # successful delivery when this 429 replaces it before headers.
        observer.response("error_response_ready", mapped.status_code, time.monotonic_ns())
    # Keep part of the reserve for cancellation cleanup. Ownership still joins a
    # send that suppresses cancellation; it cannot escape this request owner.
    await asyncio.wait_for(response(scope, receive, send), min(1.0, remaining / 2))


@dataclass(frozen=True)
class _Outcome:
    value: bool | None = None
    error: HTTPErrorResponse | None = None
    stop: _Stop | None = None


class _Call:
    """Leave only this cleared holder in AnyIO's pending tuple after completion."""

    def __init__(
        self, operation: Callable[[], bool | None], *, timing: RequestTiming | None = None,
        submitted_at_ns: int | None = None,
    ) -> None:
        self._operation: Callable[[], bool | None] | None = operation
        self._lock = Lock()
        self._started = False
        self.finished = Event()
        self._timing, self._submitted_at_ns = timing, submitted_at_ns

    def cancel_pending(self) -> None:
        """Drop a cancelled pre-start submission even if AnyIO skips its callable."""
        with self._lock:
            if not self._started:
                self._operation = None
                self.finished.set()

    def __call__(self) -> _Outcome:
        if self._timing is not None and self._submitted_at_ns is not None:
            self._timing.elapsed_queue(self._submitted_at_ns)
        with self._lock:
            self._started = True
            if self._operation is None:
                self._timing = self._submitted_at_ns = None
                self.finished.set()
                return _Outcome(stop="cancelled")
        try:
            check_request_budget()
            if (budget := current_budget()) is not None:
                budget.mark("worker_started")
            return _Outcome(value=self._operation())
        except Exception as error:
            try:
                observer = current_request_measurement()
                if observer is not None and isinstance(error, ProviderError) and (
                    error.error_class == "timeout"
                ):
                    observer.timeout("provider_timeout")
                budget = current_budget()
                return _Outcome(error=public_error(
                    error, waited_seconds=0.0 if budget is None else budget.elapsed(),
                ))
            finally:
                discard_exception(error)
        except BaseException as error:
            try:
                return _Outcome(stop=_stop_kind(error))
            finally:
                discard_exception(error)
        finally:
            self._operation = None
            self._timing = self._submitted_at_ns = None
            self.finished.set()


async def submit(operation: Callable[[], bool | None]) -> bool | None:
    """Join actual call cleanup despite repeated native or AnyIO cancellation.

    The enclosing request owns its ServiceLifetime reservation before calling
    this function. No shutdown deadline can turn an unfinished call into drain.
    The callable must return only a content-free boolean or None; responses stay
    with the request/receiver owner until their send or discard completes.
    """
    observer = current_request_measurement()
    holder = _Call(
        operation, timing=None if observer is None else observer.timing,
        submitted_at_ns=None if observer is None else time.monotonic_ns(),
    )
    del operation
    task = asyncio.create_task(run_in_threadpool(holder))
    cancelled = False
    outcome: _Outcome | None = None
    with ExitStack() as notifications, anyio.CancelScope(shield=True):
        if (budget := current_budget()) is not None:
            loop = asyncio.get_running_loop()
            cancel_task = task.cancel

            def cancel_worker() -> None:
                loop.call_soon_threadsafe(cancel_task)

            notifications.enter_context(budget.on_cancel(cancel_worker))
        while True:
            try:
                outcome = await asyncio.shield(task)
                break
            except asyncio.CancelledError as error:
                if (budget := current_budget()) is not None:
                    budget.cancel()
                discard_exception(error)
                cancelled = True
                if task.cancelled():
                    # A runtime-wide cancellation can also target the private
                    # task. Pending work is cleared; already running work must
                    # still finish before complete request ownership can leave.
                    holder.cancel_pending()
                    while not holder.finished.is_set():
                        try:
                            await asyncio.sleep(0.001)
                        except asyncio.CancelledError as repeated:
                            discard_exception(repeated)
                    outcome = _Outcome(stop="cancelled")
                    break
    del task, holder
    if cancelled:
        raise asyncio.CancelledError() from None
    if outcome is None:
        raise RuntimeError("The HTTP worker ended without a completion outcome")
    if outcome.error is not None:
        raise MappedFailure(outcome.error) from None
    if outcome.stop is not None:
        _raise_stop(outcome.stop)
    return outcome.value


class _RequestOwner:
    """One request's context on the existing lifecycle, never a separate coordinator."""

    def __init__(self, lifetime: ServiceLifetime | None) -> None:
        self._lifetime = lifetime
        self._contexts = ExitStack()
        self._admitted = False
        self._observed = False

    def admit(self) -> None:
        if self._admitted:
            raise RuntimeError("HTTP request admission cannot be repeated")
        if self._lifetime is not None:
            self._contexts.enter_context(self._lifetime.admission())
        self._admitted = True

    def observe(self, observer: RequestMeasurementOwner) -> None:
        if not self._admitted or self._observed:
            raise RuntimeError("HTTP observation requires one admitted request owner")
        self._contexts.enter_context(measure_request(observer))
        self._observed = True

    def close(self) -> None:
        self._contexts.close()


def admit(scope: Scope) -> None:
    """Admit after credential validation and before body reads or worker submission."""
    owner = cast(object, scope.get(_SCOPE_KEY))
    if not isinstance(owner, _RequestOwner):
        raise RuntimeError("HTTP request ownership was not installed")
    owner.admit()


def observe(scope: Scope, observer: RequestMeasurementOwner) -> None:
    """Keep Add observation in the existing owner through actual send/discard."""
    owner = cast(object, scope.get(_SCOPE_KEY))
    if not isinstance(owner, _RequestOwner):
        raise RuntimeError("HTTP request ownership was not installed")
    owner.observe(observer)


class RequestOwnership:
    """Hold admitted ownership through complete response send/discard and cleanup."""

    def __init__(
        self, app: ASGIApp, *, lifetime: ServiceLifetime | None,
        contract_version: RequestContractVersion = 2,
        request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
        observe_budget: Callable[[RequestBudgetObservation], None] | None = None,
    ) -> None:
        self._app = app
        self._lifetime = lifetime
        if type(contract_version) is not int or contract_version != 2:
            raise ValueError("Only the current request contract is supported")
        self._request_timeout_seconds = request_timeout_seconds
        self._observe_budget = observe_budget

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return
        owner = _RequestOwner(self._lifetime)
        scope = dict(scope)
        scope[_SCOPE_KEY] = owner
        stopped: _Stop | None = None
        response_started = False
        disconnected = False
        budget = (
            RequestBudget(self._request_timeout_seconds)
            if scope.get("path") in {"/add", "/search"}
            else None
        )
        loop = asyncio.get_running_loop()
        response_deadline = None if budget is None else loop.time() + budget.remaining()
        if budget is not None:
            budget.reserve_response(min(
                _RESPONSE_SEND_RESERVE_SECONDS, self._request_timeout_seconds / 4,
            ))
        work_deadline = None if budget is None else loop.time() + budget.remaining()
        timer = None if budget is None or work_deadline is None else (
            loop.call_at(work_deadline, budget.cancel)
        )
        work_timeout = asyncio.timeout_at(work_deadline)

        def work_remaining() -> float:
            assert work_deadline is not None
            assert budget is not None
            return min(budget.remaining(), max(0.0, work_deadline - loop.time()))

        async def bounded_receive() -> Message:
            nonlocal disconnected
            assert budget is not None
            budget.check()
            try:
                message = await asyncio.wait_for(receive(), work_remaining())
            except TimeoutError:
                budget.cancel()
                budget.check()
                raise
            if message["type"] == "http.disconnect":
                disconnected = True
                budget.mark("disconnected")
                budget.cancel()
            elif message["type"] == "http.request" and not message.get("more_body", False):
                budget.mark("body_complete")
            return message

        async def bounded_send(message: Message) -> None:
            nonlocal response_started
            assert budget is not None
            budget.check()
            if message["type"] == "http.response.start":
                status = message.get("status")
                if scope.get("path") == "/add" and type(status) is int and status >= 400:
                    if (observer := current_request_measurement()) is not None:
                        observer.response("error_response_ready", status, time.monotonic_ns())
                # A timed-out send may still have placed headers on the wire.
                # Never attempt a second status once the first send begins.
                response_started = True
                budget.mark("response_ready")
                budget.mark("send_started")
                message = {
                    **message,
                    "headers": [*message.get("headers", []), (b"x-aml-request-contract", b"2")],
                }
            try:
                await asyncio.wait_for(send(message), work_remaining())
            except TimeoutError:
                budget.cancel()
                budget.check()
                raise
            budget.check()
            if message["type"] == "http.response.body" and not message.get("more_body", False):
                budget.mark("send_finished")

        try:
            with request_budget(budget):
                async with work_timeout:
                    await self._app(
                        scope, receive if budget is None else bounded_receive,
                        send if budget is None else bounded_send,
                    )
        except BaseException as error:
            # An owned deadline/cancellation check may escape during response
            # sending, after the route's error handler has already returned.
            # Preserve that closed rejection after cleanup; it is not an
            # unexpected runtime failure. A separate bounded 429 send attempt
            # below remains owned and joined by this request.
            owned_expiry = work_timeout.expired() and isinstance(
                error, TimeoutError | asyncio.CancelledError,
            )
            stopped = "budget_rejected" if budget is not None and (
                owned_expiry or budget.cancelled and isinstance(error, QuoteMemError)
                and error.code == "delivery_busy"
            ) else _stop_kind(error)
            if owned_expiry:
                assert budget is not None
                budget.mark("exhausted")
            discard_exception(error)
        finally:
            if timer is not None:
                timer.cancel()
            try:
                if stopped == "budget_rejected" and not response_started and not disconnected:
                    try:
                        assert budget is not None
                        assert response_deadline is not None
                        await _send_budget_refusal(
                            scope, receive, send, budget, response_deadline,
                        )
                    except BaseException as error:
                        kind = _stop_kind(error)
                        if kind != "failed":
                            stopped = kind
                        discard_exception(error)
                    else:
                        stopped = None
                elif (
                    stopped == "cancelled" and budget is not None
                    and response_deadline is not None and not response_started
                    and not disconnected
                    and self._lifetime is not None and self._lifetime.stopping()
                ):
                    # The cancellation still propagates; the client just gets an answer it
                    # can retry instead of Uvicorn's 500.
                    try:
                        await _send_stopping_refusal(
                            scope, receive, send, budget, response_deadline,
                        )
                    except BaseException as error:
                        discard_exception(error)
                if budget is not None:
                    budget.mark("closed")
                    if self._observe_budget is not None:
                        self._observe_budget(budget.observed())
            finally:
                scope.pop(_SCOPE_KEY, None)
                owner.close()
        if stopped is not None:
            _raise_stop(stopped)
