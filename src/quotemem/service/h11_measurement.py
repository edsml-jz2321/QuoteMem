"""Confirm a bounded Add response at the selected original H11 send boundary.

This optional observer delegates the existing send exactly once and creates no
task, queue, transport, or request owner. An open transport after final send is
required: closing cannot distinguish a lawful close from a swallowed fatal
socket write. Unsupported or interrupted observations remain unmeasured.
"""

from __future__ import annotations

import asyncio.selector_events
import hashlib
import importlib
import sys
import time
from collections.abc import Callable
from contextlib import AbstractContextManager, ExitStack
from dataclasses import dataclass
from operator import attrgetter
from types import MethodType
from typing import Protocol, cast

import uvicorn
from starlette.types import Message, Send
from uvicorn.protocols.http.h11_impl import H11Protocol, RequestResponseCycle

_H11_SEND = RequestResponseCycle.send
_PINNED_UVICORN = "0.52.4"
def _native_proactor_types() -> tuple[
    type[asyncio.Transport] | None, type[asyncio.Future[object]] | None, type[object] | None,
]:
    """Return the Windows-only native transport, overlapped future and IOCP types.

    Absent elsewhere; the identity checks below compare against ``None`` there,
    so a Proactor branch is unreachable on POSIX and the Selector path is
    unchanged.
    """
    if sys.platform != "win32":
        return None, None, None
    transport = cast(type[asyncio.Transport], attrgetter("_ProactorSocketTransport")(
        importlib.import_module("asyncio.proactor_events"),
    ))
    windows_events = importlib.import_module("asyncio.windows_events")
    future = cast(type[asyncio.Future[object]], attrgetter("_OverlappedFuture")(windows_events))
    return transport, future, cast(type[object], windows_events.IocpProactor)


_PROACTOR_TRANSPORT, _OVERLAPPED_FUTURE, _IOCP_TYPE = _native_proactor_types()
_TRANSPORT_TYPES: tuple[type[asyncio.Transport], ...] = (
    cast(type[asyncio.Transport], attrgetter("_SelectorSocketTransport")(asyncio.selector_events)),
) + ((_PROACTOR_TRANSPORT,) if _PROACTOR_TRANSPORT is not None else ())
_PROACTOR_WRITE = getattr(_PROACTOR_TRANSPORT, "_loop_writing", None)
_IOCP_SEND = getattr(_IOCP_TYPE, "send", None)
_TRANSPORT_BINDINGS = tuple(
    (selected, selected.write, selected.is_closing, selected.get_protocol)
    for selected in _TRANSPORT_TYPES
)


@dataclass(frozen=True, slots=True)
class ResponseBinding:
    """The exact immutable successful Add body prepared by the existing adapter."""

    status_code: int
    body_sha256: str
    body_bytes: int

    def __post_init__(self) -> None:
        if (
            type(self.status_code) is not int or self.status_code != 200
            or type(self.body_bytes) is not int or self.body_bytes < 1
            or type(self.body_sha256) is not str or len(self.body_sha256) != 64
            or any(char not in "0123456789abcdef" for char in self.body_sha256)
        ):
            raise ValueError("Transport measurement requires an exact nonempty Add 200 body")

    @classmethod
    def from_body(cls, body: bytes) -> ResponseBinding:
        """Bind the already constructed response without retaining its bytes."""
        if type(body) is not bytes:
            raise ValueError("Transport measurement requires immutable response bytes")
        return cls(200, hashlib.sha256(body).hexdigest(), len(body))


@dataclass(frozen=True, slots=True)
class H11Enqueue:
    """Successful original final send, after writes and synchronous bookkeeping.

    This confirmation is a conservative timestamp after transport enqueue,
    not a client-receipt or socket-drain acknowledgement. Build custody still
    binds the selected Python/Uvicorn files; these scalar names do not replace it.
    """

    response: ResponseBinding
    confirmed_at_ns: int
    uvicorn_version: str
    transport_type: str


class RequestTransportObserver(Protocol):
    """Borrow the existing request's observer until its actual owner closes.

    The implementation must preserve a prior accepted enqueue when later cleanup
    fails. Notifications never perform I/O or alter the application response.
    """

    def response_binding(self) -> ResponseBinding | None:
        """Return the actual prepared Add body, or None for another response kind."""
        ...

    def suspend_context(self) -> AbstractContextManager[None]:
        """Clear observation ContextVars only while the original server send runs."""
        ...

    def transport_enqueued(self, event: H11Enqueue) -> None:
        """Record actual confirmation on this still-owned request."""
        ...

    def transport_unmeasured(self) -> None:
        """Latch a pending Add observation as incomplete, without retracting success."""
        ...


def _bound(value: object, owner: object, function: object) -> bool:
    return isinstance(value, MethodType) and value.__self__ is owner and value.__func__ is function


def _native_write_open(transport: asyncio.Transport) -> bool:
    """Reject known IOCP failure before its scheduled closing callback runs.

    Only inspect the original native transport/futures; do not await, drain,
    retrieve an exception, replace callbacks or infer a peer acknowledgement.
    Pending native writes retain the same transport-enqueue meaning as Selector.
    """
    if type(transport) is not _PROACTOR_TRANSPORT:
        return True
    (proactor, loop_writing, conn_lost, eof_written, empty_waiter,
     read_future, write_future, buffer, pending_write) = attrgetter(
         "_loop._proactor", "_loop_writing", "_conn_lost", "_eof_written", "_empty_waiter",
         "_read_fut", "_write_fut", "_buffer", "_pending_write",
     )(transport)
    if (
        type(proactor) is not _IOCP_TYPE
        or not _bound(attrgetter("send")(proactor), proactor, _IOCP_SEND)
        or not _bound(loop_writing, transport, _PROACTOR_WRITE)
        or conn_lost != 0 or eof_written is not False or empty_waiter is not None
    ):
        return False
    for future in (read_future, write_future):
        if future is None:
            continue
        if type(future) is not _OVERLAPPED_FUTURE:
            return False
        state, exception = attrgetter("_state", "_exception")(future)
        if state not in {"PENDING", "FINISHED"} or exception is not None:
            return False
    if write_future is None and (buffer is not None or pending_write != 0):
        return False
    return True


def _transport_name(transport: asyncio.Transport) -> str | None:
    try:
        for selected, write, is_closing, get_protocol in _TRANSPORT_BINDINGS:
            if type(transport) is selected and (
                _bound(transport.write, transport, write)
                and _bound(transport.is_closing, transport, is_closing)
                and _bound(transport.get_protocol, transport, get_protocol)
                and _native_write_open(transport)
            ):
                return f"{selected.__module__}.{selected.__qualname__}"
    except Exception:
        # Changed/unsupported native state affects observation, never the response.
        return None
    return None


def _unmeasured(observer: RequestTransportObserver | None) -> None:
    if observer is not None:
        try:
            observer.transport_unmeasured()
        except Exception:
            # A broken metric consumer cannot replace the actual HTTP outcome.
            # No enqueue event is issued by this helper.
            pass


def _select_cycle(
    send: Send,
) -> tuple[RequestResponseCycle, asyncio.Transport, H11Protocol, str] | None:
    try:
        if uvicorn.__version__ != _PINNED_UVICORN or not isinstance(send, MethodType):
            return None
        candidate = send.__self__
        if type(candidate) is not RequestResponseCycle or send.__func__ is not _H11_SEND:
            return None
        cycle = candidate
        transport = cycle.transport
        selected = _transport_name(transport)
        if selected is None:
            return None
        protocol = transport.get_protocol()
        if (
            type(protocol) is H11Protocol and protocol.transport is transport
            and protocol.cycle is cycle and cycle.scope["method"] == "POST"
            and cycle.scope["path"] == "/add"
            and not cycle.response_started and not cycle.response_complete
        ):
            return cycle, transport, protocol, selected
    except Exception:
        # Unsupported inspection changes measurement availability only.
        return None
    return None


class H11SendObservation:
    """Observe one existing ASGI request at the listener's original send boundary.

    Args:
        send: The original callable supplied by Uvicorn, before Starlette wraps it.
        observer: Read the optional existing request observer. It remains owned
            by RequestOwnership through response send and cleanup. No observer is
            constructed here and no context survives the original request.

    A wrapped or alternate send is still delegated normally but grants no
    measurement. Only a single complete final body matching the prepared Add
    acknowledgement is accepted; streaming bodies are outside this adapter path.
    """

    def __init__(
        self, send: Send, *, observer: Callable[[], RequestTransportObserver | None],
    ) -> None:
        self._send, self._observer = send, observer
        self._cycle: RequestResponseCycle | None = None
        self._transport: asyncio.Transport | None = None
        self._protocol: H11Protocol | None = None
        self._transport_type: str | None = None
        self._status: int | None = None
        self._expected: ResponseBinding | None = None
        self._body_seen = False
        selected = _select_cycle(send)
        if selected is not None:
            self._cycle, self._transport, self._protocol, self._transport_type = selected

    async def __call__(self, message: Message) -> None:
        observer: RequestTransportObserver | None = None
        try:
            observer = self._observer()
            expected = None if observer is None else observer.response_binding()
        except Exception:
            _unmeasured(observer)
            expected = None
        cycle, transport = self._cycle, self._transport
        started = False if cycle is None else cycle.response_started
        completed = False if cycle is None else cycle.response_complete
        first_body = not self._body_seen
        if message["type"] == "http.response.body":
            self._body_seen = True
        supported = cycle is not None and transport is not None
        contexts = ExitStack()
        try:
            if observer is not None:
                try:
                    contexts.enter_context(observer.suspend_context())
                except Exception:
                    supported = False
                    _unmeasured(observer)
            # Preserve cancellation, errors, backpressure, and the exact message.
            await self._send(message)
            try:
                confirmed_at_ns = time.monotonic_ns()
            except Exception:
                confirmed_at_ns = None
                supported = False
                _unmeasured(observer)
        except BaseException:
            _unmeasured(observer)
            raise
        finally:
            try:
                contexts.close()
            except Exception:
                supported = False
                _unmeasured(observer)
        if not supported or cycle is None or transport is None:
            _unmeasured(observer)
            return
        if message["type"] == "http.response.start":
            status = message.get("status")
            if not started and cycle.response_started and type(status) is int:
                self._status = status
                if type(expected) is ResponseBinding:
                    self._expected = expected
            else:
                _unmeasured(observer)
            return
        if message["type"] != "http.response.body" or expected is None:
            return
        body = message.get("body", b"")
        if (
            type(expected) is not ResponseBinding or type(body) is not bytes
            or expected != self._expected
            or message.get("more_body", False) is not False or not first_body
            or self._status != expected.status_code or not started or completed
            or not cycle.response_complete or cycle.disconnected
            or cycle.transport is not transport
            or _transport_name(transport) != self._transport_type
            or transport.is_closing() or transport.get_protocol() is not self._protocol
            or len(body) != expected.body_bytes
            or hashlib.sha256(body).hexdigest() != expected.body_sha256
        ):
            _unmeasured(observer)
            return
        assert observer is not None and self._transport_type is not None
        assert confirmed_at_ns is not None
        try:
            observer.transport_enqueued(H11Enqueue(
                expected, confirmed_at_ns, _PINNED_UVICORN, self._transport_type,
            ))
        except Exception:
            _unmeasured(observer)
