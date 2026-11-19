"""Authenticated explicit prepare/confirm routes for selective memory erasure.

``/memory/forget/prepare`` and ``/memory/forget`` select one memory (E.2.4).
``/memory/forget/statement/prepare`` and ``/memory/forget/statement`` forget a
statement (FS-1): they are mounted only for operations that implement them.
"""

from __future__ import annotations

from typing import Literal

from starlette.requests import ClientDisconnect, Request
from starlette.responses import Response
from starlette.routing import Route
from starlette.types import Receive, Scope, Send

from aml.adapters.http._ownership import admit, discard_exception, submit
from aml.adapters.http.auth import HTTPCredentials
from aml.adapters.http.errors import forget_public_error
from aml.adapters.http.models import _object_body  # pyright: ignore[reportPrivateUsage]
from aml.api.memory_forget import (
    ForgetRequest,
    MemoryForgetOperations,
    MemoryForgetPreparedV2,
    MemoryForgetPrepareRequestV1,
    MemoryForgetRequestV1,
    MemoryForgetResponseV2,
    MemoryForgetStatementPreparedV1,
    MemoryForgetStatementPrepareRequestV1,
    MemoryForgetStatementRequestV1,
    MemoryForgetStatementResponseV1,
    StatementForgetOperations,
    parse_memory_forget_request,
)
from aml.core.errors import AMLError

type _Kind = Literal["prepare", "confirm", "statement_prepare", "statement"]
_MODELS: dict[_Kind, type[ForgetRequest]] = {
    "prepare": MemoryForgetPrepareRequestV1,
    "confirm": MemoryForgetRequestV1,
    "statement_prepare": MemoryForgetStatementPrepareRequestV1,
    "statement": MemoryForgetStatementRequestV1,
}


class _Response(Response):
    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            self.body = b""


class _Disconnected(Response):
    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        self.body = b""


class MemoryForgetHTTP:
    """Share existing authentication and joined request ownership without retrying writes.

    Args:
        operations: The memory forget operations (E.2.4).
        statements: Forget by statement (FS-1); None mounts no statement route.
    """

    def __init__(
        self, operations: MemoryForgetOperations,
        statements: StatementForgetOperations | None = None,
    ) -> None:
        self._operations = operations
        self._statements = statements

    def _answer(self, selected: ForgetRequest) -> bytes:
        # E.2.4 wire: prepare answers MemoryForgetPreparedV2 and confirm
        # MemoryForgetResponseV2 (with ForgetReceiptV3); FS-1 answers its own
        # plan and outcomes. Any other value fails closed rather than reaching
        # the caller.
        result: (MemoryForgetPreparedV2 | MemoryForgetResponseV2 | MemoryForgetStatementPreparedV1
                 | MemoryForgetStatementResponseV1)
        expected: type
        if isinstance(selected, MemoryForgetRequestV1):
            result, expected = self._operations.forget_memory(selected), MemoryForgetResponseV2
        elif isinstance(selected, MemoryForgetPrepareRequestV1):
            result = self._operations.prepare_memory_forget(selected)
            expected = MemoryForgetPreparedV2
        elif self._statements is None:
            raise AMLError("invalid_request", "Forget by statement is not served")
        elif isinstance(selected, MemoryForgetStatementPrepareRequestV1):
            result = self._statements.prepare_statement_forget(selected)
            expected = MemoryForgetStatementPreparedV1
        else:
            result = self._statements.forget_statement(selected)
            expected = MemoryForgetStatementResponseV1
        if type(result) is not expected:
            raise AMLError("service_unhealthy", "Memory forget returned another wire")
        return result.model_dump_json().encode("utf-8")

    def routes(self, credentials: HTTPCredentials) -> list[Route]:
        async def handle(request: Request, *, kind: _Kind) -> Response:
            raw: bytes | None = None
            selected: ForgetRequest | None = None
            body: bytes | None = None
            statement = kind in ("statement_prepare", "statement")
            try:
                credentials.verify(request.headers.raw)
                admit(request.scope)
                content_type = request.headers.get("content-type", "").partition(";")[0]
                if content_type.strip().lower() != "application/json":
                    raise AMLError("invalid_request", "Memory forget requires JSON content")
                raw = await request.body()
                selected = parse_memory_forget_request(_object_body(raw), _MODELS[kind])

                def execute() -> None:
                    nonlocal body
                    try:
                        if selected is None:
                            raise RuntimeError("Forget request was released before execution")
                        body = self._answer(selected)
                    except Exception as error:
                        # The worker returns only a public code; take the forget
                        # routes' code here, where the internal one is still known
                        # (E.3: intent_expired, reception_stale, budget_exceeded).
                        try:
                            code = forget_public_error(error, statement=statement).body.code
                        finally:
                            discard_exception(error)
                        raise AMLError(code, code) from None

                # A lost response never cancels or repeats a possibly committed
                # erasure. Existing ownership joins the actual worker on shutdown.
                await submit(execute)
                if body is None:
                    raise AMLError("service_unhealthy", "Memory forget returned no response")
                return _Response(body, media_type="application/json")
            except ClientDisconnect as error:
                discard_exception(error)
                return _Disconnected()
            except Exception as error:
                try:
                    # E.3: the forget routes' own code set and their
                    # resend-cannot-succeed mappings (S8a's errors table).
                    mapped = forget_public_error(error, statement=statement)
                finally:
                    discard_exception(error)
                return _Response(
                    mapped.body.model_dump_json().encode("utf-8"),
                    status_code=mapped.status_code, headers=mapped.headers,
                    media_type="application/json",
                )
            finally:
                raw = selected = body = None
                request.__dict__.pop("_body", None)
                request.__dict__.pop("_json", None)

        async def prepare(request: Request) -> Response:
            return await handle(request, kind="prepare")

        async def confirm(request: Request) -> Response:
            return await handle(request, kind="confirm")

        async def statement_prepare(request: Request) -> Response:
            return await handle(request, kind="statement_prepare")

        async def statement_confirm(request: Request) -> Response:
            return await handle(request, kind="statement")

        routes = [
            Route("/memory/forget/prepare", prepare, methods=["POST"]),
            Route("/memory/forget", confirm, methods=["POST"]),
        ]
        if self._statements is not None:
            routes += [
                Route("/memory/forget/statement/prepare", statement_prepare, methods=["POST"]),
                Route("/memory/forget/statement", statement_confirm, methods=["POST"]),
            ]
        return routes
