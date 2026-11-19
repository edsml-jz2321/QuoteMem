"""Explicit ownership of the pinned Uvicorn listener and all its ASGI tasks."""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Iterator
from contextlib import contextmanager

import uvicorn
from starlette.applications import Starlette
from starlette.types import ASGIApp, Receive, Scope, Send

from aml.core.errors import AMLError


class _Server(uvicorn.Server):
    def __init__(self, config: uvicorn.Config) -> None:
        super().__init__(config)
        # Uvicorn otherwise adds this attribute only after startup reaches bind.
        self.servers = []

    @contextmanager
    def capture_signals(self) -> Iterator[None]:
        # The process composition owns signals and the full resource shutdown.
        # Uvicorn must not re-raise a signal before application workers drain.
        yield


def _remaining(deadline: float) -> float:
    if type(deadline) not in {int, float} or not math.isfinite(deadline):
        raise ValueError("Listener deadline must be finite absolute monotonic time")
    return max(0.0, deadline - time.monotonic())


def _measured_app(app: ASGIApp) -> ASGIApp:
    if not isinstance(app, Starlette):
        return app
    observer = getattr(app.state, "aml_add_transport_measurement", None)
    if observer is None:
        return app
    from aml.service.h11_measurement import H11SendObservation
    from aml.service.request_measurement import current_request_measurement

    if observer is not current_request_measurement:
        # Unknown instrumentation cannot grant transport evidence.
        return app

    async def observed(scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and scope["method"] == "POST" and scope["path"] == "/add":
            await app(scope, receive, H11SendObservation(send, observer=observer))
        else:
            await app(scope, receive, send)

    return observed


KEEP_ALIVE_IDLE_SECONDS = 75
"""Idle seconds before the server closes a keep-alive connection.

Uvicorn's default of five seconds closed pooled connections between a
client's phases, and a request the client had already put on such a
connection surfaced as a read error with no response (M2.2 acceptance run
under 64 sessions, 2026-09-11). The value exceeds the bench client timeouts
(45 s Add, 30 s Search) so an idle gap shorter than a request budget never
meets a server-side close; it is a server resource setting, not a request or
measurement timeout.
"""


class UvicornListener:
    """Run one TCP HTTP server in the composition's existing asyncio loop.

    Construction is inert. Attach this listener to BenchRuntime before start,
    then activate that runtime only after start acknowledges the bound socket.
    There is no automatic lifespan-driven application resource closure. The
    configured response budget belongs to the HTTP/application composition.
    """

    def __init__(self, app: ASGIApp, *, host: str, port: int) -> None:
        if type(host) is not str or not host or type(port) is not int or not 0 <= port <= 65535:
            raise ValueError("Listener requires a host and a valid TCP port")
        self._server = _Server(uvicorn.Config(
            _measured_app(app), host=host, port=port, loop="asyncio", http="h11", ws="none",
            lifespan="off", access_log=False, log_config=None,
            timeout_graceful_shutdown=0, timeout_keep_alive=KEEP_ALIVE_IDLE_SECONDS,
        ))
        self._task: asyncio.Task[None] | None = None
        self._withdrawn = False
        self._failed = False

    async def _serve(self) -> None:
        try:
            await self._server.serve()
        except BaseException:
            # Retain a closed failure fact, not an exception with application or
            # server frames. Actual sockets/tasks must still positively drain.
            self._failed = True

    async def start(self, deadline: float) -> None:
        """Start once and wait for actual bound sockets within the supplied budget.

        Any failed, cancelled or incomplete startup remains owned. Its caller
        must withdraw and drain before closing the application resource owner.
        """
        _remaining(deadline)
        if self._task is not None or self._withdrawn:
            raise RuntimeError("Listener can start only once")
        self._task = asyncio.create_task(self._serve(), name="aml-http-listener")
        while not self._server.started:
            if self._task.done() or not _remaining(deadline):
                self.withdraw()
                raise AMLError("service_unhealthy", "HTTP listener did not start")
            await asyncio.sleep(min(0.01, _remaining(deadline)))
        if self._withdrawn or self._failed or self._task.done():
            raise AMLError("service_unhealthy", "HTTP listener startup was withdrawn")

    @property
    def port(self) -> int:
        """Return the actual sole bound port, including an assigned ephemeral port."""
        if not self._server.started or self._withdrawn:
            raise RuntimeError("Listener is not serving")
        ports = {
            int(sock.getsockname()[1]) for server in self._server.servers
            for sock in server.sockets or ()
        }
        if len(ports) != 1:
            raise RuntimeError("Listener does not have one bound TCP port")
        return ports.pop()

    def require_ready(self) -> None:
        """Reject a stopped, withdrawn or failed server without any network request."""
        if self._task is None or self._task.done() or self._failed or self._withdrawn \
                or not self._server.started or not all(
                    server.is_serving() for server in self._server.servers
                ):
            raise AMLError("service_unhealthy", "HTTP listener is not accepting connections")

    async def wait_stopped(self) -> None:
        """Wait only for the server loop; the caller must still drain ASGI ownership."""
        if self._task is None:
            raise RuntimeError("Listener has not started")
        await asyncio.shield(self._task)
        if self._failed:
            raise AMLError("service_unhealthy", "HTTP listener stopped unexpectedly")

    def withdraw(self) -> None:
        """Stop connection admission immediately without mistaking cancellation for join."""
        self._withdrawn = True
        self._server.should_exit = True
        for server in self._server.servers:
            server.close()
        for connection in tuple(self._server.server_state.connections):
            connection.shutdown()

    async def drain(self, deadline: float) -> bool:
        """Join actual server, ASGI tasks and connections; retain everything on timeout.

        Uvicorn 0.52.4 shutdown cancels timed-out tasks and can return before they
        finish. Its serve-task completion is therefore only one of these checks.
        This wait never cancels the serving task or a previous drain attempt.
        A caller cancellation leaves the same live owners available to retry.
        """
        _remaining(deadline)
        self.withdraw()
        task = self._task
        if task is None:
            return True
        while True:
            # Startup can finish binding after withdrawal. Reapply the same
            # shutdown to any sockets/connections it has since established.
            self.withdraw()
            active = {pending for pending in self._server.server_state.tasks if not pending.done()}
            if task.done():
                # An interrupted server task may have skipped Uvicorn shutdown.
                # Wake its surviving ASGI requests but retain them until joined.
                for request in active:
                    request.cancel()
            if task.done() and not active and not self._server.server_state.connections:
                servers = tuple(self._server.servers)
                if any(server.is_serving() for server in servers):
                    return False
                for server in servers:
                    try:
                        await asyncio.wait_for(server.wait_closed(), _remaining(deadline))
                    except TimeoutError:
                        return False
                return True
            remaining = _remaining(deadline)
            if not remaining:
                return False
            pending: set[asyncio.Task[None]] = set(active)
            if not task.done():
                pending.add(task)
            if pending:
                await asyncio.wait(pending, timeout=min(0.02, remaining))
            else:
                await asyncio.sleep(min(0.01, remaining))
