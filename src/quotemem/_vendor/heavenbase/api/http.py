"""Starlette transport for the dashboard application service."""

from __future__ import annotations

__all__ = ["build_app", "resolve_bind", "run_server"]

import asyncio
import inspect
import socket
import sys
import sysconfig
import threading
import warnings
from contextvars import Context, copy_context
from ipaddress import ip_address
from urllib.parse import urlparse

from heavenbase.utils.files import exists_dir, pj
from heavenbase.utils.serialize import dumps_json
from heavenbase.utils.typing import Any, AsyncGenerator, Callable, Dict, Iterator, Optional, Set

from .models import DASHBOARD_OPERATIONS, ApiOperation, AssistEvent, EmptyParameters, openapi_schema
from .service import DashboardAPI, _ASSIST_STREAM_STOP

LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
# One assist tool round can hold the provider for a minute with nothing to emit, and a
# silent socket reads the same as a dead one to the browser and to any proxy that reaps
# idle connections. A beat well inside the usual 30-60s reaper window keeps a working
# turn visibly alive for the cost of a few bytes. The shipped client aborts a stream
# after 90s of silence (``ASSIST_STREAM_IDLE_MS``, apps/gui/ui/src/api.ts), so this
# interval must stay well under that window.
ASSIST_KEEPALIVE_SECONDS = 15.0


def _ui_dir() -> str:
    """Resolve the dashboard UI from a source checkout or installed wheel.

    Args:
        None.

    Returns:
        str: Absolute path to the distributable UI tree.

    Raises:
        FileNotFoundError: If neither the source nor installed data-file tree
            contains the dashboard UI.
    """
    source = pj(__file__, "..", "..", "..", "..", "apps", "gui", "ui", abs=True)
    installed = pj(sysconfig.get_path("data"), "share", "heavenbase", "gui", "ui", abs=True)
    for candidate in (source, installed):
        if exists_dir(candidate):
            return candidate
    raise FileNotFoundError("dashboard UI assets are missing; reinstall HeavenBase from a complete wheel")


def _host_name(raw: str) -> str:
    """Return the lowercase host name from a ``Host``/``Origin`` value, dropping any port."""
    raw = raw.strip().lower()
    if raw.startswith("[") and "]" in raw:  # bracketed IPv6, e.g. [::1]:42777
        return raw[1 : raw.index("]")]
    if raw.count(":") > 1:  # unbracketed IPv6 bind address has no port suffix
        return raw
    return raw.split(":", 1)[0]


def _host_port(raw: str) -> str:
    """Return the port suffix of a ``Host``/``Origin`` netloc, or ``""`` when absent."""
    raw = raw.strip().lower()
    if raw.startswith("[") and "]" in raw:
        raw = raw[raw.index("]") + 1 :]
        return raw[1:] if raw.startswith(":") else ""
    if raw.count(":") > 1:
        return ""
    _, sep, port = raw.partition(":")
    return port if sep else ""


def _is_loopback(host: str) -> bool:
    """Return whether ``host`` names a loopback-only listener."""
    name = _host_name(host)
    if name == "localhost":
        return True
    try:
        return ip_address(name).is_loopback
    except ValueError:
        return False


def _url_host(host: str) -> str:
    """Return ``host`` formatted for use in an HTTP URL authority."""
    name = _host_name(host)
    return f"[{name}]" if ":" in name else name


def _same_local_origin(origin_netloc: str, host: str) -> bool:
    """Return whether an ``Origin`` netloc may write to a ``Host`` netloc.

    Exact match passes. Loopback aliases (``localhost``/``127.0.0.1``/``::1``)
    on the **same port** also pass: they name the same single listener, so the
    requesting page is this server's own UI reached under another alias. A
    different port is a different local app and stays blocked, as does any
    non-loopback or opaque (``"null"``) origin.
    """
    if origin_netloc == host:
        return True
    if _host_port(origin_netloc) != _host_port(host):
        return False
    return _host_name(origin_netloc) in LOOPBACK_HOSTS and _host_name(host) in LOOPBACK_HOSTS


class LocalGuard:
    """ASGI guard against DNS-rebinding reads and cross-origin writes.

    When ``allowed_hosts`` is set, requests must carry a ``Host`` header naming
    an allowed host, which defeats DNS-rebinding attacks against a
    loopback-bound server. State-changing requests are additionally required to
    be same-origin: any ``Origin`` header must match the request host **and
    port** exactly, which blocks cross-site writes both from remote pages and
    from other local apps served on a different loopback port.
    """

    def __init__(self, app: Any, allowed_hosts: Optional[Set[str]] = None) -> None:
        """Attach the guard to one ASGI application.

        Args:
            app (Any): Inner ASGI application.
            allowed_hosts (Set[str] | None): Host names accepted in the ``Host``
                header. ``None`` skips the host check.

        Returns:
            None: This function does not return a value.
        """
        self.app = app
        self.allowed_hosts = allowed_hosts

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = {key.decode().lower(): value.decode() for key, value in scope.get("headers") or []}
        host = headers.get("host", "").strip().lower()
        if self.allowed_hosts is not None and _host_name(host) not in self.allowed_hosts:
            await self._deny(send, 403, "host not allowed")
            return
        if scope.get("method", "GET").upper() not in SAFE_METHODS:
            origin = headers.get("origin")
            # Same-netloc comparison with loopback-alias equivalence: the page at
            # "http://localhost:42777" may write to "127.0.0.1:42777" (same
            # listener), but "http://localhost:3000" and opaque "null" origins
            # must not write here.
            if origin and not _same_local_origin((urlparse(origin).netloc or origin).strip().lower(), host):
                await self._deny(send, 403, "cross-origin request blocked")
                return
        await self.app(scope, receive, send)

    async def _deny(self, send: Any, status_code: int, message: str) -> None:
        body = dumps_json({"error": message}).encode()
        await send({"type": "http.response.start", "status": status_code, "headers": [(b"content-type", b"application/json")]})
        await send({"type": "http.response.body", "body": body})


def _json(data: Any, status_code: int = 200) -> Any:
    import starlette
    from starlette.responses import Response

    return Response(dumps_json(data), status_code=status_code, media_type="application/json")


async def _operation_arguments(request: Any, operation: ApiOperation, method: Callable[..., Any]) -> Dict[str, Any]:
    """Validate and merge one operation's path, query, and body arguments for ``method``.

    A field the request model lists in ``advisory_fields`` is passed only when
    ``method`` accepts it (a named parameter or ``**kwargs``), whatever its
    value. 0.1.2.3 added the session-log identifiers ``conversation_id`` and
    ``turn_id`` to the assist requests, and the bundled UI sends both on every
    chat turn; a ``DashboardAPI`` override with the 0.1.2.2 parameter list
    records no session log, so it is served without them. Every other field
    is always passed, so an override that cannot bind one (a required field,
    or an optional one such as a config ``scope`` or ``read_only`` that changes
    what the request does) fails with the binding ``TypeError``, which the
    endpoints report as HTTP 400, instead of silently ignoring the request.

    Args:
        request (Any): Starlette request carrying the path, query, and body.
        operation (ApiOperation): Operation whose models validate the request.
        method (Callable[..., Any]): Bound service method the arguments are for.

    Returns:
        Dict[str, Any]: Keyword arguments for ``method``.

    Raises:
        ValueError: If the body is not a JSON object.
        pydantic.ValidationError: If a part of the request fails its model.
    """
    arguments: Dict[str, Any] = {}
    advisory: Set[str] = set()

    def merge(validated: Any) -> None:
        arguments.update(validated.model_dump())
        advisory.update(type(validated).advisory_fields)

    if operation.path_model is not EmptyParameters:
        merge(operation.path_model.model_validate(dict(request.path_params)))
    if operation.query_model is not EmptyParameters:
        merge(operation.query_model.model_validate(dict(request.query_params)))
    if operation.body_model is not None:
        try:
            body = await request.json()
        except ValueError as error:
            raise ValueError("request body must be a JSON object") from error
        if not isinstance(body, dict):
            raise ValueError("request body must be a JSON object")
        merge(operation.body_model.model_validate(body))
    accepted = _keyword_names(method)
    if accepted is None:
        return arguments
    return {name: value for name, value in arguments.items() if name in accepted or name not in advisory}


def _keyword_names(method: Callable[..., Any]) -> Optional[Set[str]]:
    """Return the keyword names ``method`` binds, or ``None`` when it takes any keyword.

    Args:
        method (Callable[..., Any]): Callable to inspect.

    Returns:
        Set[str] | None: Parameter names passable by keyword, or ``None`` for a
        ``**kwargs`` callable or one whose signature cannot be read.
    """
    try:
        parameters = inspect.signature(method).parameters.values()
    except (TypeError, ValueError):
        return None
    if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters):
        return None
    return {parameter.name for parameter in parameters if parameter.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)}


def _validation_message(error: Any) -> str:
    """Return one stable client-facing Pydantic validation message."""
    errors = error.errors()
    if not errors:
        return str(error)
    first = errors[0]
    location = ".".join(str(part) for part in first.get("loc") or ())
    message = str(first.get("msg") or "invalid request")
    return f"{location}: {message}" if location else message


def _endpoint(service: DashboardAPI, operation: ApiOperation) -> Callable[..., Any]:
    async def endpoint(request: Any) -> Any:
        import starlette
        from starlette.concurrency import run_in_threadpool

        try:
            method = getattr(service, operation.service_method)
            arguments = await _operation_arguments(request, operation, method)
            result = await run_in_threadpool(method, **arguments)
        except KeyError as error:
            return _json({"error": str(error.args[0] if error.args else error)}, status_code=404)
        except _pydantic_validation_error() as error:
            return _json({"error": _validation_message(error)}, status_code=400)
        except (TypeError, ValueError) as error:
            return _json({"error": str(error)}, status_code=400)
        except RuntimeError as error:
            return _json({"error": str(error)}, status_code=500)
        try:
            payload = operation.response_model.model_validate(result).model_dump(mode="json", by_alias=True)
        except _pydantic_validation_error() as error:
            raise RuntimeError(f"{operation.operation_id} produced an invalid response: {_validation_message(error)}") from error
        return _json(payload)

    return endpoint


def _pydantic_validation_error() -> type[Exception]:
    """Load Pydantic's validation exception without widening module imports."""
    from pydantic import ValidationError

    return ValidationError


async def _assist_sse(iterator: Iterator[Dict[str, Any]], stream_context: Context) -> AsyncGenerator[str, None]:
    """Stream one sync assist generator as SSE lines with deterministic teardown.

    A dedicated daemon thread owns every resumption of ``iterator`` (one
    persistent ``contextvars`` context, so tracing survives suspensions) and
    hands events to the running loop. The response side therefore only ever
    awaits a queue — a client disconnect cancels that await immediately, the
    thread closes the generator at its next yield boundary, and a provider
    that never yields again parks only this daemon thread, not an anyio
    limiter token or a half-open response, until the provider's own timeout.
    A one-event acknowledgement preserves response backpressure instead of
    allowing a fast producer to grow an unbounded in-memory queue.

    Args:
        iterator (Iterator[Dict[str, Any]]): Sync iterator of assist event
            dictionaries.
        stream_context (Context): ``contextvars`` context the pump thread runs in,
            so tracing and other request ContextVars reach the generator.

    Yields:
        str: ``data: {json}\\n\\n`` server-sent-event lines, interleaved with
            ``: keepalive\\n\\n`` comment frames whenever the provider goes quiet.
    """
    loop = asyncio.get_running_loop()
    feed: "asyncio.Queue[Any]" = asyncio.Queue(maxsize=1)
    stopped = threading.Event()
    consumed = threading.Event()
    done = object()

    def emit(item: Any) -> bool:
        if stopped.is_set():
            return False
        consumed.clear()
        try:
            loop.call_soon_threadsafe(feed.put_nowait, item)
        except RuntimeError:  # event loop already closed during shutdown
            stopped.set()
            return False
        while not stopped.is_set():
            if consumed.wait(timeout=0.05):
                return True
        return False

    def pump() -> None:
        token = _ASSIST_STREAM_STOP.set(stopped)
        try:
            while not stopped.is_set():
                try:
                    event = next(iterator)
                    payload = AssistEvent.model_validate(event).model_dump(mode="json", exclude_none=True)
                except StopIteration:
                    break
                except _pydantic_validation_error() as error:
                    emit({"kind": "error", "text": f"assist failed: invalid stream event: {_validation_message(error)}"})
                    break
                except Exception as error:  # defensive: assist_stream wraps its own errors
                    emit({"kind": "error", "text": f"assist failed: {error}"})
                    break
                if not emit(payload):
                    break
        finally:
            try:
                close = getattr(iterator, "close", None)  # plain iterators (test stubs) have nothing to close
                if close is not None:
                    close()
            except Exception as error:
                if not stopped.is_set():
                    emit({"kind": "error", "text": f"assist failed during stream teardown: {error}"})
            finally:
                _ASSIST_STREAM_STOP.reset(token)
                emit(done)

    threading.Thread(target=lambda: stream_context.run(pump), name="assist-sse", daemon=True).start()
    try:
        while True:
            try:
                event = await asyncio.wait_for(feed.get(), timeout=ASSIST_KEEPALIVE_SECONDS)
            except asyncio.TimeoutError:
                # An SSE comment frame: the shipped client keeps only ``data:`` lines, so a
                # heartbeat proves liveness without touching the event contract. Timing out
                # the queue read instead of beating from a second task keeps the heartbeat
                # bound to this generator, so it cannot outlive the response; a cancelled
                # ``get`` leaves a concurrently queued event in place for the next round.
                yield ": keepalive\n\n"
                continue
            if event is done:
                consumed.set()
                break
            try:
                yield f"data: {dumps_json(event, compact=True)}\n\n"
            finally:
                consumed.set()
    finally:
        stopped.set()
        consumed.set()


def _sse_endpoint(service: "DashboardAPI", operation: ApiOperation) -> Callable[..., Any]:
    """Build the server-sent-events endpoint streaming assist events."""

    async def endpoint(request: Any) -> Any:
        import starlette
        from starlette.responses import StreamingResponse

        try:
            method = getattr(service, operation.service_method)
            arguments = await _operation_arguments(request, operation, method)
            # Inside the try: an override that cannot bind a required field is a
            # 400, like the blocking endpoint, not an unhandled 500.
            iterator = iter(method(**arguments))
        except _pydantic_validation_error() as error:
            return _json({"error": _validation_message(error)}, status_code=400)
        except (TypeError, ValueError) as error:
            return _json({"error": str(error)}, status_code=400)

        return StreamingResponse(
            _assist_sse(iterator, copy_context()),
            media_type="text/event-stream",
            headers={"cache-control": "no-cache"},
        )

    return endpoint


def _no_cache(app: Any) -> Any:
    """Force revalidation of static assets so upgraded frontend files are never served stale from the browser cache."""

    async def wrapped(scope: Any, receive: Any, send: Any) -> None:
        async def send_no_cache(message: Any) -> None:
            if message["type"] == "http.response.start":
                headers = [(key, value) for key, value in message.get("headers", []) if key.lower() != b"cache-control"]
                message["headers"] = headers + [(b"cache-control", b"no-cache")]
            await send(message)

        await app(scope, receive, send_no_cache)

    return wrapped


def build_app(
    api: Optional[DashboardAPI] = None,
    allowed_hosts: Optional[Set[str]] = LOOPBACK_HOSTS,
    static_dir: Optional[str] = None,
) -> Any:
    """Build the dashboard Starlette application.

    Args:
        api (DashboardAPI | None): Service instance backing the routes.
            Defaults to a fresh ``DashboardAPI()`` over the process default
            Context.
        allowed_hosts (Set[str] | None): Host names accepted in the ``Host``
            header. Defaults to the loopback set so a DNS-rebinding page cannot
            read config secrets. Pass ``None`` to skip the host check when the
            server is bound to a broad interface on purpose.
        static_dir (str | None): Optional distributable UI tree. Defaults to
            `apps/gui/ui` in a source checkout or the wheel-installed data path.

    Returns:
        Any: A ``starlette.applications.Starlette`` app serving the JSON API
        under ``/api`` and the static workbench at ``/``, wrapped in
        :class:`LocalGuard`.
    """
    import starlette
    from starlette.applications import Starlette
    from starlette.middleware import Middleware
    from starlette.routing import Mount, Route
    from starlette.staticfiles import StaticFiles

    service = api or DashboardAPI()
    routes = []
    for operation in DASHBOARD_OPERATIONS:
        endpoint = _sse_endpoint(service, operation) if operation.stream else _endpoint(service, operation)
        routes.append(Route(operation.path, endpoint, methods=[operation.method], name=operation.operation_id))
    routes.extend(
        [
            Route("/openapi.json", lambda request: _json(openapi_schema()), methods=["GET"], name="openapi"),
            Mount("/", app=_no_cache(StaticFiles(directory=static_dir or _ui_dir(), html=True))),
        ]
    )
    return Starlette(routes=routes, middleware=[Middleware(LocalGuard, allowed_hosts=allowed_hosts)])


def resolve_bind(cm: Any = None, host: Optional[str] = None, port: Optional[int] = None) -> tuple[str, int]:
    """Resolve the dashboard bind address from explicit values and config.

    Args:
        cm (Any): Config manager for ``heavenbase.dashboard.*`` tunables.
            Defaults to the process default Context's ConfigManager:
            ``CM_HVNB``, unless ``set_default_context`` installed a Context
            with its own ConfigManager.
        host (str | None): Explicit bind host. Defaults to
            ``heavenbase.dashboard.host`` (``127.0.0.1``).
        port (int | None): Explicit bind port. Zero requests an OS-assigned
            port. Defaults to ``heavenbase.dashboard.port`` (``42777``).

    Returns:
        tuple[str, int]: ``(host, port)`` with config defaults applied.
    """
    if cm is None:
        from ..context.authority import omitted_config

        cm = omitted_config()
    return str(host or cm.get("heavenbase.dashboard.host", "127.0.0.1")), int(cm.get("heavenbase.dashboard.port", 42777) if port is None else port)


def run_server(
    host: Optional[str] = None,
    port: Optional[int] = None,
    api: Optional[DashboardAPI] = None,
    allow_network: bool = False,
    static_dir: Optional[str] = None,
) -> None:
    """Run the dashboard HTTP service and block until interrupted.

    Args:
        host (str | None): Bind host. Defaults to
            ``heavenbase.dashboard.host`` (``127.0.0.1``).
        port (int | None): Bind port. Defaults to
            ``heavenbase.dashboard.port`` (``42777``).
        api (DashboardAPI | None): Service instance backing the routes.
            Defaults to a fresh ``DashboardAPI()`` over the process default
            Context. The boot check, host and port read the ConfigManager of
            the service that serves the routes.
        allow_network (bool): Allow a non-loopback bind that exposes the
            unauthenticated dashboard API to other machines. Defaults to
            ``False`` and must be opted into explicitly.
        static_dir (str | None): Optional distributable UI tree. Defaults to
            the source or installed GUI data path.

    Warns:
        UserWarning: When a requested non-loopback bind is explicitly allowed.

    Raises:
        ValueError: If a non-loopback bind is requested without
            ``allow_network=True``.
        SystemExit: With status 1 if Uvicorn cannot bind the requested address
            (such as a busy port), or with Uvicorn's startup-failure status 3
            if the server stops before serving for another reason, such as a
            failed application startup.

    Returns:
        None: This function does not return a value.
    """
    _run_server(host, port, api, allow_network, static_dir)


def _run_server(
    host: Optional[str],
    port: Optional[int],
    api: Optional[DashboardAPI],
    allow_network: bool,
    static_dir: Optional[str] = None,
    *,
    on_started: Callable[[int], None] | None = None,
) -> None:
    """Run the dashboard server and report its listening port to the startup callback."""
    import uvicorn
    from uvicorn.main import STARTUP_FAILURE

    # One service backs the routes, and the boot check and bind read its
    # config, so an installed default Context with its own ConfigManager is
    # checked and bound by that ConfigManager, not by ``CM_HVNB``.
    service = api if api is not None else DashboardAPI()
    # Every server entry point (CLI serve, desktop app, direct run_server)
    # refuses to boot on a session-log configuration that cannot keep its
    # durability or credential promises; a no-op while the log is off.
    from .assist_log import AssistSessionLog

    AssistSessionLog.validate(service.cm, service.context)
    host, port = resolve_bind(service.cm, host, port)
    broad = _host_name(host) in {"0.0.0.0", "::", ""}
    network = not _is_loopback(host)
    if network and not allow_network:
        raise ValueError(f"refusing non-loopback dashboard bind {host!r}; pass --allow-network only if you accept exposing the unauthenticated config API")
    if network:
        warnings.warn(
            f"SECURITY WARNING: dashboard bound to {host}; the unauthenticated config API may expose secrets to the network",
            UserWarning,
            stacklevel=2,
        )
    allowed_hosts = None if broad else LOOPBACK_HOSTS | {_host_name(host)}
    app = build_app(service, allowed_hosts=allowed_hosts, static_dir=static_dir)

    class DashboardServer(uvicorn.Server):
        async def startup(self, sockets: list[socket.socket] | None = None) -> None:
            await super().startup(sockets=sockets)
            if on_started is not None and self.started and not self.should_exit:
                bound_port = port if port else self.servers[0].sockets[0].getsockname()[1]
                on_started(bound_port)

    # One run path for every entry point, with ``uvicorn.run``'s outcome:
    # Uvicorn re-raises the captured Ctrl+C after its graceful shutdown, which
    # ends the run normally. A bind failure exits with status 1 inside
    # ``server.run()``; a server that stopped before serving for another
    # reason (such as a failed application startup) exits with Uvicorn's
    # startup-failure status.
    server = DashboardServer(uvicorn.Config(app, host=host, port=port, log_level="warning"))
    try:
        server.run()
    except KeyboardInterrupt:
        pass
    if not server.started:
        sys.exit(STARTUP_FAILURE)
