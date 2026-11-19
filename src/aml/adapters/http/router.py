"""User-sharded front router over N independent ``aml serve`` processes (PERF-1).

One ``aml serve`` owns one workspace with one Writer and runs Python under one
interpreter lock, so it uses about one core however many requests it holds.
A host with more cores runs N services, each on its own workspace and port,
behind this router::

    client --> aml route :7999 --> shard i = SHA-256(user_id)[:8] mod N --> aml serve

Why routing by ``user_id`` keeps every contract:

- Every Add, Search and memory route names exactly one ``user_id``, and that
  value is the request's whole retrieval and write scope. A user therefore
  lives entirely on one shard: its strict per-conversation queue, idempotency
  ledger, binding checks and single Writer are that shard's, unchanged.
- No request reads or writes two shards. The shard function depends only on
  the bytes of ``user_id`` and the shard count, so it survives restarts. The
  shard list is frozen with its workspaces: changing N or reordering the list
  moves users to workspaces that never saw them, so it needs fresh workspaces.

Behaviour:

- ``POST`` on the user routes forwards the body byte for byte with the
  credential and content-type headers; each shard authenticates and validates
  the request itself. A body without a usable string ``user_id`` goes to shard
  0, which answers the request's own 401 or 422: the router never invents a
  contract response.
- The shard's status, body, content type, ``Retry-After`` and request-contract
  headers return unchanged. The router never retries: a POST is replayed only
  by its client, with its own idempotency.
- A shard that cannot be reached is starting or draining: ``index_rebuilding``
  (429, retryable, ``Retry-After: 60``). A client that honours Retry-After only
  on a 429 and retries a 503 quickly (AMLB allows 32 Add attempts) would spend
  its attempts long before a large shard has restarted.
- ``HEAD`` on a user route answers 405, so a client probing ``HEAD /add`` or
  ``HEAD /search`` learns that only POST is allowed.
- ``GET /health`` is 200 only when every shard's ``/health`` answers 2xx
  within the health timeout; otherwise 503 ``service_unhealthy``.
- Request and response bodies, user ids and credentials are never logged.
  With ``shape_log`` (``aml route --shape-log``, off by default) the router
  appends one value-free shape record per user-route POST
  (``aml.adapters.http.wire_shape``): key paths, types, sizes and header names.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import time
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Final, TextIO, cast

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route

from aml.adapters.http.errors import public_error
from aml.adapters.http.wire_shape import request_shape
from aml.core.errors import AMLError
from aml.core.request_budget import DEFAULT_REQUEST_TIMEOUT_SECONDS

USER_ROUTES: Final = (
    "/add", "/search", "/memory/history", "/memory/why", "/memory/add-items",
    "/memory/forget/prepare", "/memory/forget", "/memory/forget/statement/prepare",
    "/memory/forget/statement",
)
"""Every route whose body names the one ``user_id`` it reads or writes."""

FORWARDED_REQUEST_HEADERS: Final = frozenset({"authorization", "x-api-key", "content-type"})
FORWARDED_RESPONSE_HEADERS: Final = frozenset({
    "content-type", "retry-after", "x-aml-request-contract",
})
MAX_BODY_BYTES: Final = 64 * 1024 * 1024
"""Largest request body the router buffers; a larger one is refused as ``invalid_request``."""


def shard_index(user_id: str, shards: int) -> int:
    """Return the shard that owns ``user_id``: SHA-256 of its UTF-8 bytes, first 8 bytes, mod N.

    Args:
        user_id (str): The request's exact external user id.
        shards (int): The frozen number of shards, at least 1.

    Returns:
        int: The shard index in ``range(shards)``.

    Raises:
        ValueError: When ``shards`` is not a positive integer.
    """
    if type(shards) is not int or shards < 1:
        raise ValueError("The shard count must be a positive integer")
    digest = hashlib.sha256(user_id.encode("utf-8", "surrogatepass")).digest()
    return int.from_bytes(digest[:8], "big") % shards


def _user_of(body: bytes) -> str | None:
    """The body's top-level string ``user_id``, or None when it has none."""
    try:
        parsed: object = json.loads(body)
    except (ValueError, UnicodeDecodeError, RecursionError):
        return None
    if not isinstance(parsed, dict):
        return None
    user = cast(dict[str, object], parsed).get("user_id")
    return user if isinstance(user, str) and user else None


@dataclass(frozen=True)
class RouterConfig:
    """The frozen shard list and the router's upstream timeouts.

    Attributes:
        shards: Base URLs of the shard services, in shard order.
        connect_seconds: Upstream connect timeout.
        request_seconds: Upstream read and write timeout; longer than one
            request's own budget, so the shard's answer always arrives.
        health_seconds: Timeout of one shard's health check.
    """

    shards: tuple[str, ...]
    connect_seconds: float = 10.0
    request_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS + 110.0
    health_seconds: float = 10.0

    def __post_init__(self) -> None:
        if not self.shards:
            raise ValueError("A router needs at least one shard")
        if len(set(self.shards)) != len(self.shards):
            raise ValueError("Each shard is listed once")
        for url in self.shards:
            parsed = httpx.URL(url)
            if parsed.scheme not in ("http", "https") or not parsed.host or (
                parsed.path not in ("", "/") or parsed.query or parsed.fragment
            ):
                raise ValueError("A shard is an http(s) origin without a path")
        if min(self.connect_seconds, self.request_seconds, self.health_seconds) <= 0:
            raise ValueError("Router timeouts are positive")


UNREACHABLE_RETRY_AFTER_SECONDS: Final = 60
"""AMLB's Retry-After cap; a shard's restart takes minutes, so retry at the cap."""


def _error(code: str, retry_after_seconds: int = 1) -> Response:
    mapped = public_error(AMLError(code, code), retry_after_seconds=retry_after_seconds)
    return Response(
        mapped.body.model_dump_json().encode("utf-8"), status_code=mapped.status_code,
        headers=mapped.headers, media_type="application/json",
    )


def router_app(
    config: RouterConfig, *, transport: httpx.AsyncBaseTransport | None = None,
    shape_log: TextIO | None = None,
) -> Starlette:
    """Build the router's ASGI application.

    Args:
        config (RouterConfig): The frozen shards and timeouts.
        transport (httpx.AsyncBaseTransport | None): Upstream transport; None
            uses real connections. Tests pass a mock.
        shape_log (TextIO | None): Where to append one JSON shape record per
            user-route POST, flushed per line; None (the default) records nothing.
            A failed write never changes the answer.

    Returns:
        Starlette: The application. Its lifespan owns the upstream client.
    """
    shards = tuple(url.rstrip("/") for url in config.shards)
    timeout = httpx.Timeout(
        connect=config.connect_seconds, read=config.request_seconds,
        write=config.request_seconds, pool=None,
    )
    holder: list[httpx.AsyncClient] = []

    def client() -> httpx.AsyncClient:
        if not holder:
            holder.append(httpx.AsyncClient(
                timeout=timeout, transport=transport, trust_env=False, follow_redirects=False,
                limits=httpx.Limits(max_connections=None, max_keepalive_connections=1024),
            ))
        return holder[0]

    def record(request: Request, body: bytes, status: int) -> None:
        if shape_log is None:
            return
        try:
            line = {"t": round(time.time(), 3),
                    **request_shape(request.url.path, body, request.headers.keys(), status)}
            shape_log.write(json.dumps(line, sort_keys=True) + "\n")
            shape_log.flush()
        except (OSError, ValueError):
            pass

    async def forward(request: Request) -> Response:
        declared = request.headers.get("content-length")
        if declared is not None and (not declared.isdigit() or int(declared) > MAX_BODY_BYTES):
            return _error("invalid_request")
        body = await request.body()
        if len(body) > MAX_BODY_BYTES:
            return _error("invalid_request")
        user = _user_of(body)
        shard = 0 if user is None else shard_index(user, len(shards))
        headers = [(name, value) for name, value in request.headers.items()
                   if name.lower() in FORWARDED_REQUEST_HEADERS]
        try:
            upstream = await client().post(
                shards[shard] + request.url.path, content=body, headers=headers,
            )
        except httpx.TransportError:
            refusal = _error("index_rebuilding", UNREACHABLE_RETRY_AFTER_SECONDS)
            record(request, body, refusal.status_code)
            return refusal
        record(request, body, upstream.status_code)
        passed = {name: value for name, value in upstream.headers.items()
                  if name.lower() in FORWARDED_RESPONSE_HEADERS}
        return Response(upstream.content, status_code=upstream.status_code, headers=passed)

    async def one_healthy(url: str) -> bool:
        try:
            answer = await client().get(url + "/health", timeout=config.health_seconds)
        except httpx.TransportError:
            return False
        return 200 <= answer.status_code < 300

    async def health(_: Request) -> Response:
        states = await asyncio.gather(*(one_healthy(url) for url in shards))
        if all(states):
            return Response(b"{}", media_type="application/json")
        return _error("service_unhealthy")

    async def missing(request: Request) -> Response:
        if request.method == "HEAD" and request.url.path in USER_ROUTES:
            return Response(status_code=405, headers={"Allow": "POST"})
        return _error("invalid_request")

    @contextlib.asynccontextmanager
    async def lifespan(_: Starlette) -> AsyncIterator[None]:
        try:
            yield
        finally:
            if holder:
                await holder.pop().aclose()

    routes: Sequence[Route] = [
        *(Route(path, forward, methods=["POST"]) for path in USER_ROUTES),
        Route("/health", health, methods=["GET"]),
        Route("/{path:path}", missing, methods=["GET", "POST", "PUT", "DELETE", "PATCH"]),
    ]
    return Starlette(routes=list(routes), lifespan=lifespan)
