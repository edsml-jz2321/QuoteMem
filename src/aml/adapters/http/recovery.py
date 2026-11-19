"""Authenticated explicit provider recovery on the shared ASGI request lifetime."""

from __future__ import annotations

from starlette.requests import ClientDisconnect, Request
from starlette.responses import Response
from starlette.routing import Route
from starlette.types import Receive, Scope, Send

from aml.adapters.http._ownership import admit, discard_exception, submit
from aml.adapters.http.auth import HTTPCredentials
from aml.adapters.http.errors import public_error
from aml.adapters.http.models import _object_body  # pyright: ignore[reportPrivateUsage]
from aml.api.bench_recovery import (
    RecoveryOperations,
    RecoveryRequestV1,
    ResumeRequestV1,
    parse_recovery_request,
)
from aml.api.completion_recovery import (
    CompletionRecoveryOperations,
    CompletionRecoveryRequestV2,
    CompletionRecoveryRequestV3,
    CompletionResumeRequestV2,
    CompletionResumeRequestV3,
    QuarantineSkipOperations,
    QuarantineSkipRequestV1,
    parse_completion_request,
    parse_skip_request,
)
from aml.core.errors import AMLError


class _Response(Response):
    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            self.body = b""


class _Disconnected(Response):
    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        self.body = b""


class BenchRecoveryHTTP:
    """Adapt strict metadata DTOs; authorization and recovery stay in the owner."""

    def __init__(
        self, application: RecoveryOperations, *, completion_enabled: bool = False,
    ) -> None:
        self._application = application
        self._completion = application if completion_enabled and isinstance(
            application, CompletionRecoveryOperations,
        ) else None
        # The operator-only quarantine skip rides on completion recovery's
        # enablement and loopback route family; nothing calls it automatically.
        self._skip = application if self._completion is not None and isinstance(
            application, QuarantineSkipOperations,
        ) else None
        if completion_enabled and self._completion is None:
            raise AMLError("schema_invalid", "V2 completion recovery requires its existing owner")

    def routes(self, credentials: HTTPCredentials) -> list[Route]:
        """Share credentials, admission and joined submission with ordinary HTTP."""
        async def handle(
            request: Request, *, resume: bool, completion: bool = False, refusal: bool = False,
        ) -> Response:
            raw: bytes | None = None
            selected: (
                RecoveryRequestV1 | CompletionRecoveryRequestV2 | CompletionRecoveryRequestV3 | None
            ) = None
            body: bytes | None = None
            try:
                credentials.verify(request.headers.raw)
                admit(request.scope)
                content_type = request.headers.get("content-type", "").partition(";")[0]
                if content_type.strip().lower() != "application/json":
                    raise AMLError("invalid_request", "HTTP requests require JSON content")
                raw = await request.body()
                selected = parse_completion_request(
                    _object_body(raw), (
                        CompletionResumeRequestV3 if resume else CompletionRecoveryRequestV3
                    ) if refusal else (
                        CompletionResumeRequestV2 if resume else CompletionRecoveryRequestV2
                    ),
                ) if completion else parse_recovery_request(
                    _object_body(raw), ResumeRequestV1 if resume else RecoveryRequestV1,
                )

                def execute() -> None:
                    nonlocal body
                    if selected is None:
                        raise RuntimeError("Recovery request was released before execution")
                    if isinstance(
                        selected, CompletionRecoveryRequestV2 | CompletionRecoveryRequestV3,
                    ):
                        if self._completion is None:
                            raise AMLError(
                                "service_unhealthy", "V2 completion recovery is disabled",
                            )
                        result = (
                            self._completion.resume_completion_budget(selected)
                            if isinstance(
                                selected, CompletionResumeRequestV2 | CompletionResumeRequestV3,
                            )
                            else self._completion.inspect_completion_budget(selected)
                        )
                    else:
                        result = (
                            self._application.resume_recovery(selected)
                            if isinstance(selected, ResumeRequestV1)
                            else self._application.inspect_recovery(selected)
                        )
                    body = result.model_dump_json().encode("utf-8")

                # submit joins the real synchronous operation even if the request
                # is cancelled. Never repeat a possibly committed mutation.
                await submit(execute)
                if body is None:
                    raise AMLError("service_unhealthy", "Recovery returned no response")
                return _Response(body, media_type="application/json")
            except ClientDisconnect as error:
                discard_exception(error)
                return _Disconnected()
            except Exception as error:
                try:
                    mapped = public_error(error)
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

        async def skip(request: Request) -> Response:
            raw: bytes | None = None
            selected: QuarantineSkipRequestV1 | None = None
            body: bytes | None = None
            try:
                credentials.verify(request.headers.raw)
                admit(request.scope)
                content_type = request.headers.get("content-type", "").partition(";")[0]
                if content_type.strip().lower() != "application/json":
                    raise AMLError("invalid_request", "HTTP requests require JSON content")
                raw = await request.body()
                selected = parse_skip_request(_object_body(raw))

                def execute() -> None:
                    nonlocal body
                    if selected is None or self._skip is None:
                        raise RuntimeError("Skip request was released before execution")
                    body = self._skip.skip_quarantined_cycle(
                        selected,
                    ).model_dump_json().encode("utf-8")

                # Joined like every recovery: a possibly committed skip is never repeated.
                await submit(execute)
                if body is None:
                    raise AMLError("service_unhealthy", "Quarantine skip returned no response")
                return _Response(body, media_type="application/json")
            except ClientDisconnect as error:
                discard_exception(error)
                return _Disconnected()
            except Exception as error:
                try:
                    mapped = public_error(error)
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

        async def inspect(request: Request) -> Response:
            return await handle(request, resume=False)

        async def resume(request: Request) -> Response:
            return await handle(request, resume=True)

        async def inspect_completion(request: Request) -> Response:
            return await handle(request, resume=False, completion=True)

        async def resume_completion(request: Request) -> Response:
            return await handle(request, resume=True, completion=True)

        async def inspect_refusal(request: Request) -> Response:
            return await handle(request, resume=False, completion=True, refusal=True)

        async def resume_refusal(request: Request) -> Response:
            return await handle(request, resume=True, completion=True, refusal=True)

        return [
            Route("/bench/recovery", inspect, methods=["POST"]),
            Route("/bench/resume", resume, methods=["POST"]),
            *([] if self._completion is None else [
                Route("/bench/v2/completion", inspect_completion, methods=["POST"]),
                Route("/bench/v2/completion/resume", resume_completion, methods=["POST"]),
                Route("/bench/v3/completion", inspect_refusal, methods=["POST"]),
                Route("/bench/v3/completion/resume", resume_refusal, methods=["POST"]),
            ]),
            *([] if self._skip is None else [
                Route("/bench/v1/quarantine/skip", skip, methods=["POST"]),
            ]),
        ]
