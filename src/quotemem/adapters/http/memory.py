"""Authenticated QuoteMem memory reads over HTTP: History v3, Why v2 and add-items v1.

The readers own authorization, the pinned guard and the predicates. This
adapter validates the strict request bodies, runs each read on the request's
worker and encodes the response inside the reader's callback, while the
guard still holds the scope: a privacy commit that revokes the
pin fails the delivery instead of releasing withdrawn bytes. Encoded bytes
stay with a receiver that drops them on client disconnect, and the response
body is cleared once it is sent.

Each route answers only its codes: History and Why ``invalid_request``,
``authentication_failed``, ``scope_denied``, ``index_rebuilding``,
``delivery_busy``, ``authorization_unavailable`` and ``service_unhealthy``;
add-items the same without ``scope_denied`` and ``index_rebuilding``. Every
other failure fails closed to ``service_unhealthy``.
"""

from __future__ import annotations

from collections.abc import Callable
from threading import Lock
from time import monotonic
from typing import Annotated, Final, Literal

import anyio
from pydantic import ConfigDict, Field, ValidationError, field_validator
from starlette.requests import ClientDisconnect, Request
from starlette.responses import Response
from starlette.routing import Route
from starlette.types import Receive, Scope, Send

from quotemem.adapters.http._ownership import admit, discard_exception, submit
from quotemem.adapters.http.auth import HTTPCredentials
from quotemem.adapters.http.errors import HTTPErrorCode, public_error
from quotemem.adapters.http.models import (
    _object_body,  # pyright: ignore[reportPrivateUsage]
    _unicode,  # pyright: ignore[reportPrivateUsage]
)
from quotemem.api.quote_speakers import FrozenLabelSpeakers
from quotemem.core.errors import QuoteMemError
from quotemem.core.identity import ULID, scope_id
from quotemem.core.query import ScopeV1
from quotemem.core.semantic import SystemTimestamp
from quotemem.core.technical_types import ExternalId
from quotemem.publication.add_completions import StoredAddCompletions
from quotemem.publication.add_items import (
    MAX_ADD_ITEMS_LIMIT,
    AddItemsReader,
    AddItemsV1,
    ItemRef,
)
from quotemem.publication.history import (
    MAX_HISTORY_LIMIT,
    MemoryHistoryReader,
    MemoryHistoryV3,
    MemoryReads,
)
from quotemem.publication.judge_verdicts import StoredJudgeVerdicts
from quotemem.publication.why import MemoryWhyV2, WhyReader, WhyTargetV2
from quotemem.store.metadata import StrictModel

_READ_CODES: Final[frozenset[HTTPErrorCode]] = frozenset({
    "invalid_request", "authentication_failed", "scope_denied", "index_rebuilding",
    "delivery_busy", "authorization_unavailable", "service_unhealthy",
})
"""The public codes of ``/memory/history`` and ``/memory/why``."""

_ADD_ITEMS_CODES: Final[frozenset[HTTPErrorCode]] = frozenset({
    "invalid_request", "authentication_failed", "delivery_busy", "authorization_unavailable",
    "service_unhealthy",
})
"""The public codes of ``/memory/add-items``."""


class _Wire(StrictModel):
    model_config = ConfigDict(frozen=True, hide_input_in_errors=True)


class _UserRequest(_Wire):
    user_id: Annotated[ExternalId, Field(repr=False)]

    @field_validator("user_id")
    @classmethod
    def valid_user(cls, value: str) -> str:
        """Keep the exact external user id after checking it is valid Unicode."""
        return _unicode(value)

    @property
    def scope(self) -> ScopeV1:
        """The caller's exact user scope."""
        return ScopeV1(scope_kind="user", scope_id=scope_id("user", self.user_id))


class MemoryHistoryRequestV3(_UserRequest):
    """Read one memory's History, or its version recorded by an instant."""

    schema_version: Literal[3]
    memory_id: ULID
    limit: Annotated[int, Field(ge=1, le=MAX_HISTORY_LIMIT)] = 32
    recorded_as_of: SystemTimestamp | None = None


class MemoryWhyRequestV2(_UserRequest):
    """Explain one memory version or one Add item."""

    schema_version: Literal[2]
    target: WhyTargetV2


class MemoryAddItemsRequestV1(_UserRequest):
    """List one completed Add request's items."""

    schema_version: Literal[1]
    session_id: Annotated[ExternalId, Field(repr=False)]
    request_id: Annotated[ExternalId, Field(repr=False)]
    after_item_ref: ItemRef | None = None
    limit: Annotated[int, Field(ge=1, le=MAX_ADD_ITEMS_LIMIT)] = MAX_ADD_ITEMS_LIMIT

    @field_validator("session_id", "request_id")
    @classmethod
    def valid_ids(cls, value: str) -> str:
        """Keep the exact external ids after checking they are valid Unicode."""
        return _unicode(value)


class MemoryHistoryResponseV3(_Wire):
    """``POST /memory/history`` response, schema 3."""

    schema_version: Literal[3] = 3
    data: MemoryHistoryV3


class MemoryWhyResponseV2(_Wire):
    """``POST /memory/why`` response, schema 2."""

    schema_version: Literal[2] = 2
    data: MemoryWhyV2


class MemoryAddItemsResponseV1(_Wire):
    """``POST /memory/add-items`` response, schema 1."""

    schema_version: Literal[1] = 1
    data: AddItemsV1


MemoryReadRequest = MemoryHistoryRequestV3 | MemoryWhyRequestV2 | MemoryAddItemsRequestV1
"""One parsed memory-read request."""

_REQUESTS: Final[dict[str, type[MemoryReadRequest]]] = {
    "/memory/history": MemoryHistoryRequestV3,
    "/memory/why": MemoryWhyRequestV2,
    "/memory/add-items": MemoryAddItemsRequestV1,
}


def parse_memory_read(path: str, body: bytes) -> MemoryReadRequest:
    """Parse one memory-read request body for its route.

    Args:
        path (str): ``/memory/history``, ``/memory/why`` or ``/memory/add-items``.
        body (bytes): The UTF-8 JSON body; duplicate fields are refused.

    Returns:
        MemoryReadRequest: The strict request.

    Raises:
        QuoteMemError: ``invalid_request`` for a body the route does not accept.
    """
    model = _REQUESTS.get(path)
    if model is None:
        raise QuoteMemError("invalid_request", "Unknown memory read route")
    try:
        return model.model_validate(_object_body(body))
    except ValidationError:
        raise QuoteMemError("invalid_request", "Memory read request is invalid") from None


class _Response(Response):
    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            self.body = b""


class _Disconnected(Response):
    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        self.body = b""


class _Receiver:
    """Own the bytes a reader's callback encoded until they are taken or dropped."""

    def __init__(self, deadline: float) -> None:
        self.lock = Lock()
        self.deadline = deadline
        self.closed = False
        self.body: bytes | None = None

    def enqueue(self, body: bytes) -> None:
        with self.lock:
            if self.closed or monotonic() >= self.deadline:
                raise QuoteMemError("delivery_busy", "Memory read delivery is unavailable")
            if self.body is not None:
                raise QuoteMemError("service_unhealthy", "Memory reader delivered more than once")
            self.body = body

    def close(self) -> None:
        with self.lock:
            self.closed = True
            self.body = None

    def take(self) -> bytes:
        with self.lock:
            if self.body is None:
                raise QuoteMemError("service_unhealthy", "Memory reader did not deliver a response")
            self.closed = True
            body, self.body = self.body, None
            return body


async def _consume_body(
    request: Request, operation: Callable[[Callable[[bytes], None]], None], deadline: float,
) -> bytes:
    """Run a read on the request's worker and take the bytes its callback enqueued."""
    receiver = _Receiver(deadline)
    disconnected = False
    finished = False
    failure: str | None = None

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
            if not finished:
                receiver.close()

    def consume() -> None:
        if monotonic() >= deadline:
            raise QuoteMemError("delivery_busy", "Memory read admission budget elapsed")
        operation(receiver.enqueue)

    async def run() -> None:
        nonlocal finished, failure
        try:
            await submit(consume)
        except Exception as error:
            try:
                failure = public_error(error).body.code
            finally:
                discard_exception(error)
        finally:
            finished = True
            group.cancel_scope.cancel()

    try:
        async with anyio.create_task_group() as group:
            group.start_soon(watch_disconnect)
            group.start_soon(run)
        if disconnected:
            raise ClientDisconnect()
        if failure is not None:
            raise QuoteMemError(failure, failure)
        return receiver.take()
    finally:
        receiver.close()


class MemoryReadHTTP:
    """Adapt the History, Why and add-items readers to their authenticated routes.

    Construct one per serving application, over readers built on the
    registered read model; the enclosing application installs its request
    ownership middleware and serving lifetime. Only the routes of the supplied
    readers are mounted, so a serving application mounts Why and add-items once
    their judge-record and completion readers are wired; a route that is not
    mounted is never answered with an empty result.

    Args:
        history (MemoryHistoryReader): The History v3 reader.
        why (WhyReader | None): The Why v2 reader, or None to leave the route out.
        add_items (AddItemsReader | None): The add-items reader, or None to leave
            the route out.
    """

    def __init__(
        self, history: MemoryHistoryReader, *, why: WhyReader | None = None,
        add_items: AddItemsReader | None = None,
    ) -> None:
        self._history, self._why, self._add_items = history, why, add_items

    def _execute(self, selected: MemoryReadRequest, enqueue: Callable[[bytes], None]) -> None:
        """Run one read and encode its response inside the reader's callback."""
        if isinstance(selected, MemoryHistoryRequestV3):
            def history(value: MemoryHistoryV3) -> None:
                if value.memory_id != selected.memory_id:
                    raise QuoteMemError("binding_mismatch", "History changed its selected Item")
                enqueue(MemoryHistoryResponseV3(data=value).model_dump_json().encode("utf-8"))

            self._history.consume(
                selected.memory_id, scope=selected.scope, consumer=history,
                limit=selected.limit, recorded_as_of=selected.recorded_as_of,
            )
        elif isinstance(selected, MemoryWhyRequestV2):
            def why(value: MemoryWhyV2) -> None:
                if value.target != selected.target:
                    raise QuoteMemError("binding_mismatch", "Why changed its selected target")
                enqueue(MemoryWhyResponseV2(data=value).model_dump_json().encode("utf-8"))

            if self._why is None:
                raise QuoteMemError("service_unhealthy", "The Why route is not mounted")
            self._why.consume(selected.target, scope=selected.scope, consumer=why)
        else:
            def items(value: AddItemsV1) -> None:
                if (value.session_id, value.request_id) != (
                    selected.session_id, selected.request_id,
                ):
                    raise QuoteMemError("binding_mismatch", "Add items changed their request")
                enqueue(MemoryAddItemsResponseV1(data=value).model_dump_json().encode("utf-8"))

            if self._add_items is None:
                raise QuoteMemError("service_unhealthy", "The add-items route is not mounted")
            self._add_items.consume(
                scope=selected.scope, session_id=selected.session_id,
                request_id=selected.request_id, consumer=items,
                after_item_ref=selected.after_item_ref, limit=selected.limit,
            )

    def routes(self, credentials: HTTPCredentials, *, timeout_seconds: float) -> list[Route]:
        """Build the authenticated memory-read routes.

        Args:
            credentials (HTTPCredentials): The deployment's credential boundary.
            timeout_seconds (float): The serving application's delivery budget.

        Returns:
            list[Route]: ``POST /memory/history``, and ``/memory/why`` and
                ``/memory/add-items`` when their readers are supplied.
        """
        async def handle(request: Request) -> Response:
            raw: bytes | None = None
            selected: MemoryReadRequest | None = None
            path = request.url.path
            try:
                credentials.verify(request.headers.raw)
                admit(request.scope)
                content_type = request.headers.get("content-type", "").partition(";")[0]
                if content_type.strip().lower() != "application/json":
                    raise QuoteMemError("invalid_request", "Memory reads require JSON content")
                raw = await request.body()
                deadline = monotonic() + timeout_seconds
                selected = parse_memory_read(path, raw)

                def consume(enqueue: Callable[[bytes], None]) -> None:
                    if selected is None:
                        raise RuntimeError("Memory read request was released before execution")
                    self._execute(selected, enqueue)

                return _Response(
                    await _consume_body(request, consume, deadline),
                    media_type="application/json",
                )
            except ClientDisconnect as error:
                discard_exception(error)
                return _Disconnected()
            except Exception as error:
                try:
                    mapped = public_error(error)
                    if mapped.body.code not in public_codes(path):
                        mapped = public_error(QuoteMemError("service_unhealthy",
                                                            "service_unhealthy"))
                finally:
                    discard_exception(error)
                return _Response(
                    mapped.body.model_dump_json().encode("utf-8"),
                    status_code=mapped.status_code, headers=mapped.headers,
                    media_type="application/json",
                )
            finally:
                raw = selected = None
                request.__dict__.pop("_body", None)
                request.__dict__.pop("_json", None)

        return [
            Route("/memory/history", handle, methods=["POST"]),
            *([] if self._why is None else [Route("/memory/why", handle, methods=["POST"])]),
            *([] if self._add_items is None else [
                Route("/memory/add-items", handle, methods=["POST"]),
            ]),
        ]


def memory_read_http(reads: MemoryReads, *, add_items: bool = False) -> MemoryReadHTTP:
    """Build the production memory-read routes over one shared `MemoryReads`.

    History and Why share one `FrozenLabelSpeakers`; Why reads the stored
    judge records (`StoredJudgeVerdicts`) and, for item targets, the
    completion records (`StoredAddCompletions`).

    Args:
        reads (MemoryReads): The runtime's authority, read model and clock.
        add_items (bool): Mount ``/memory/add-items``; ``BenchRuntime`` mounts it
            over the completion records its Add owner writes.

    Returns:
        MemoryReadHTTP: The routes, passed to the app as ``memory_reads=``.
    """
    speakers, completions = FrozenLabelSpeakers(), StoredAddCompletions()
    return MemoryReadHTTP(
        MemoryHistoryReader(reads, speakers),
        why=WhyReader(reads, speakers, StoredJudgeVerdicts(), completions),
        add_items=AddItemsReader(reads, completions) if add_items else None,
    )


def public_codes(path: str) -> frozenset[HTTPErrorCode]:
    """The public error codes one memory-read route answers."""
    return _ADD_ITEMS_CODES if path == "/memory/add-items" else _READ_CODES
