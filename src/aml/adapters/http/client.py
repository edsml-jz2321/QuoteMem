"""User-facing client for the existing authenticated HTTP application boundary.

This adapter does not own storage, start a service, retry mutations, or infer
publication from an Add acknowledgement. Selective forget requires an explicit
prepared selection and preserves that operation across uncertain responses. The
memory reads (History v3, Why v2, add-items v1; E.2) are checked against the
selection they answer.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Literal, NamedTuple, cast, get_args

import httpx
from pydantic import ValidationError

from aml.adapters.http.auth import HTTPCredentials
from aml.adapters.http.errors import PublicErrorV1
from aml.adapters.http.memory import (
    MemoryAddItemsRequestV1,
    MemoryAddItemsResponseV1,
    MemoryHistoryRequestV3,
    MemoryHistoryResponseV3,
    MemoryWhyRequestV2,
    MemoryWhyResponseV2,
)
from aml.adapters.http.models import (
    AddResponseV1,
    SearchRequestV1,
    _object_body,  # pyright: ignore[reportPrivateUsage]
    parse_add_body,
    parse_search_response,
)
from aml.api.bench_recovery import (
    RecoveryRequestV1,
    RecoveryResponseV1,
    ResumeRequestV1,
    ResumeResponseV1,
)
from aml.api.completion_recovery import (
    CompletionRecoveryRequestV2,
    CompletionRecoveryRequestV3,
    CompletionRecoveryResponseV2,
    CompletionRecoveryResponseV3,
    CompletionResumeRequestV2,
    CompletionResumeRequestV3,
    parse_completion_response,
)
from aml.api.memory_forget import (
    MemoryForgetPreparedV2,
    MemoryForgetPrepareRequestV1,
    MemoryForgetRequestV1,
    MemoryForgetResponseV2,
    MemoryForgetStatementPreparedV1,
    MemoryForgetStatementPrepareRequestV1,
    MemoryForgetStatementRequestV1,
    MemoryForgetStatementResponseV1,
)
from aml.core.errors import AMLError
from aml.core.identity import canonical_json, integrity_digest
from aml.core.request_budget import PLATFORM_REQUEST_TIMEOUT_SECONDS
from aml.ingestion.boundary import AddRequestV1
from aml.store.metadata import StrictModel

Operation = Literal[
    "add", "search", "recovery", "resume",
    "completion", "resume-completion", "prepare-forget", "forget",
    "prepare-forget-statement", "forget-statement", "history", "why", "add-items",
]
_MAX_REQUEST_BYTES = 8 * 1024 * 1024
_MAX_RESPONSE_BYTES = 16 * 1024 * 1024
_OPERATIONS: dict[Operation, tuple[str, type[StrictModel], type[StrictModel]]] = {
    "recovery": ("/bench/recovery", RecoveryRequestV1, RecoveryResponseV1),
    "resume": ("/bench/resume", ResumeRequestV1, ResumeResponseV1),
    "completion": (
        "/bench/v2/completion", CompletionRecoveryRequestV2, CompletionRecoveryResponseV2,
    ),
    "resume-completion": (
        "/bench/v2/completion/resume", CompletionResumeRequestV2, CompletionRecoveryResponseV2,
    ),
    "prepare-forget": (
        "/memory/forget/prepare", MemoryForgetPrepareRequestV1, MemoryForgetPreparedV2,
    ),
    "forget": ("/memory/forget", MemoryForgetRequestV1, MemoryForgetResponseV2),
    "prepare-forget-statement": (
        "/memory/forget/statement/prepare", MemoryForgetStatementPrepareRequestV1,
        MemoryForgetStatementPreparedV1,
    ),
    "forget-statement": (
        "/memory/forget/statement", MemoryForgetStatementRequestV1,
        MemoryForgetStatementResponseV1,
    ),
    "history": ("/memory/history", MemoryHistoryRequestV3, MemoryHistoryResponseV3),
    "why": ("/memory/why", MemoryWhyRequestV2, MemoryWhyResponseV2),
    "add-items": ("/memory/add-items", MemoryAddItemsRequestV1, MemoryAddItemsResponseV1),
}


class EncodedRequest(NamedTuple):
    """One validated operation with the exact route and body bytes that ``request`` sends.

    ``content`` is the canonical UTF-8 JSON of ``model`` without null fields, so it
    also carries the non-null declared defaults of fields the caller left unset.
    """

    path: str
    model: StrictModel
    content: bytes


def _request(
    operation: Operation, body: bytes,
) -> tuple[EncodedRequest, type[StrictModel] | None]:
    if len(body) > _MAX_REQUEST_BYTES:
        raise AMLError("invalid_request", "Client request exceeds its byte limit")
    selected: StrictModel
    response_model: type[StrictModel] | None = None
    if operation == "add":
        path, selected, response_model = "/add", parse_add_body(body), AddResponseV1
    elif operation == "search":
        path, selected = "/search", SearchRequestV1.from_body(body, max_top_k=1000)
    elif operation in _OPERATIONS:
        path, request_model, response_model = _OPERATIONS[operation]
        raw = _object_body(body, exact_numbers=False)
        if operation == "completion" and raw.get("schema_version") == 3:
            path = "/bench/v3/completion"
            request_model = CompletionRecoveryRequestV3
            response_model = CompletionRecoveryResponseV3
        if operation == "resume-completion" and raw.get("schema_version") == 3:
            path = "/bench/v3/completion/resume"
            request_model = CompletionResumeRequestV3
            response_model = CompletionRecoveryResponseV3
        try:
            selected = request_model.model_validate(raw)
        except ValidationError:
            raise AMLError("invalid_request", "Client request fields are invalid") from None
    else:
        raise AMLError("unsupported_variant", "HTTP client operation is unsupported")
    content = canonical_json(selected.model_dump(mode="json", exclude_none=True)).encode("utf-8")
    return EncodedRequest(path, selected, content), response_model


def _response(
    operation: Operation, body: bytes, selected: StrictModel, model: type[StrictModel] | None,
) -> StrictModel:
    try:
        if operation == "search":
            result = parse_search_response(body)
            if not isinstance(selected, SearchRequestV1) or len(result.data) > selected.top_k:
                raise ValueError("Search response exceeds its requested count")
            return result
        assert model is not None
        parsed = _object_body(body, exact_numbers=False)
        if operation in {"completion", "resume-completion"}:
            result = parse_completion_response(parsed)
            if result.schema_version != selected.model_dump()["schema_version"]:
                raise ValueError("Completion response changed its requested version")
            unselected = not isinstance(selected, CompletionRecoveryRequestV3) or (
                selected.request_id is None
            )
            if isinstance(result, CompletionRecoveryResponseV3) and (
                unselected and result.completed is not None
            ):
                raise ValueError("Completion response reports an unselected request")
        else:
            result = model.model_validate_json(canonical_json(parsed))
        if operation == "add" and not (
            isinstance(result, AddResponseV1) and isinstance(selected, AddRequestV1)
            and result.acknowledges(selected)
        ):
            raise ValueError("Add acknowledgement differs from the selected operation")
        if isinstance(result, ResumeResponseV1):
            observed = result.selection.model_dump(
                include={"owner_run_id", "child_run_id", "owner_attempt", "child_attempt"},
            )
            if any(selected.model_dump().get(key) != value for key, value in observed.items()):
                raise ValueError("Recovery acknowledgement differs from the selected failure")
        if isinstance(result, MemoryForgetPreparedV2):
            if result.confirmation.model_dump(include=set(selected.model_fields_set)) != (
                selected.model_dump()
            ):
                raise ValueError("Forget preparation changed the selected operation")
        if isinstance(result, MemoryForgetResponseV2 | MemoryForgetStatementResponseV1) and (
            result.request_digest != integrity_digest(selected)
        ):
            raise ValueError("Forget receipt differs from the confirmed operation")
        if isinstance(result, MemoryForgetStatementPreparedV1) and (
            result.confirmation is not None and result.confirmation.model_dump(
                include={"user_id", "client_session_id", "request_id"},
            ) != selected.model_dump(include={"user_id", "client_session_id", "request_id"})
        ):
            raise ValueError("Forget preparation changed the selected operation")
        _check_memory_read(selected, result)
        return result
    except (AMLError, ValueError):
        raise AMLError("service_unhealthy", "HTTP response contract is invalid") from None


def _check_memory_read(selected: StrictModel, result: StrictModel) -> None:
    """Refuse a memory-read response that answers another selection than the one sent."""
    if isinstance(result, MemoryHistoryResponseV3) and not (
        isinstance(selected, MemoryHistoryRequestV3)
        and result.data.memory_id == selected.memory_id
        and result.data.recorded_as_of == selected.recorded_as_of
        and len(result.data.entries) <= selected.limit
    ):
        raise ValueError("History answered another selection")
    if isinstance(result, MemoryWhyResponseV2) and not (
        isinstance(selected, MemoryWhyRequestV2) and result.data.target == selected.target
    ):
        raise ValueError("Why explained another target")
    if isinstance(result, MemoryAddItemsResponseV1) and not (
        isinstance(selected, MemoryAddItemsRequestV1)
        and (result.data.session_id, result.data.request_id) == (
            selected.session_id, selected.request_id,
        )
        and len(result.data.items) <= selected.limit
    ):
        raise ValueError("Add items answered another request")


def encode_request(operation: Operation, body: bytes) -> EncodedRequest:
    """Validate one operation and return the exact route and bytes ``request`` sends.

    Args:
        operation: One existing application route, selected without fallback.
        body: Original request JSON, at most 8 MiB.

    Returns:
        The route, the validated typed request and its wire bytes. The bytes omit
        null fields and add the declared defaults of fields the caller left unset,
        so they can differ from ``body``.

    Raises:
        AMLError: ``invalid_request`` for oversize, malformed or invalid input, or
            ``unsupported_variant`` for an unknown operation. Nothing is sent.
    """
    return _request(operation, body)[0]


def request(client: httpx.Client, operation: Operation, body: bytes) -> StrictModel:
    """Send one validated operation and return only its complete validated response.

    Args:
        client: Caller-owned authenticated HTTP client with an explicit base URL
            and finite timeout. The caller closes this client after use.
        operation: One existing application route, selected without fallback.
        body: Original request JSON. Add source text and operation IDs are retained.
            The bytes sent are exactly ``encode_request(operation, body).content``.

    Returns:
        The route's authoritative response model. ``success`` confirms the active
        server Add contract and its six completed counts; ``resume_committed``
        confirms only recovery admission. A null eligible failure does not
        establish that publication is complete. Whether explicit repair completed
        an Add is read from the V3 completion inspection that names its request.

    Raises:
        AMLError: Invalid input, a safe server error, invalid response, timeout or
            unavailable transport. No operation is automatically resent: a timeout
            may follow a committed mutation. Reuse the same Add IDs and exact
            content only when deliberately checking that operation's idempotence.
            ``repair_required`` and ``extraction_refused`` are terminal until an
            explicit completion resume; only a V3 resume resumes a refusal.
    """
    encoded, response_model = _request(operation, body)
    # Validate the caller's timeout instead of silently accepting httpx's None.
    if any(value is None or not math.isfinite(value) or value <= 0 for value in (
        client.timeout.connect, client.timeout.read, client.timeout.write, client.timeout.pool,
    )):
        raise AMLError("invalid_request", "Client requires finite positive timeout bounds")
    received = bytearray()
    try:
        with client.stream(
            "POST", encoded.path, content=encoded.content,
            headers={"Content-Type": "application/json"}, follow_redirects=False,
        ) as response:
            for chunk in response.iter_bytes():
                if len(received) + len(chunk) > _MAX_RESPONSE_BYTES:
                    raise AMLError("service_unhealthy", "HTTP response exceeds its byte limit")
                received.extend(chunk)
            if response.status_code != 200:
                failure = PublicErrorV1.from_response(response.status_code, bytes(received))
                raise AMLError(failure.code, failure.detail)
            return _response(operation, bytes(received), encoded.model, response_model)
    except httpx.TimeoutException:
        raise AMLError(
            "delivery_busy", "HTTP timeout; operation outcome is not established",
        ) from None
    except httpx.HTTPError:
        raise AMLError("service_unhealthy", "HTTP transport unavailable") from None
    finally:
        received.clear()


def main() -> int:
    """Run one local host operation using environment-held credentials and JSON input."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=get_args(Operation))
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=PLATFORM_REQUEST_TIMEOUT_SECONDS)
    options = parser.parse_args()
    try:
        endpoint = httpx.URL(options.url)
        if endpoint.username or endpoint.password or endpoint.query or endpoint.fragment or (
            endpoint.path not in ("", "/") or endpoint.scheme not in ("http", "https")
        ):
            raise AMLError("invalid_request", "Use one HTTP origin without embedded credentials")
        credentials = HTTPCredentials(
            cast(Literal["token", "bearer", "x-api-key"],
                 os.environ.get("AML_HTTP_AUTH_MODE", "bearer")),
            os.environ.get("AML_HTTP_AUTH_SECRET", ""),
        )
        header = {"X-API-Key": credentials.secret} if credentials.mode == "x-api-key" else {
            "Authorization": f"{credentials.mode} {credentials.secret}",
        }
        with options.request.open("rb") as source:
            body = source.read(_MAX_REQUEST_BYTES + 1)
        with httpx.Client(
            base_url=endpoint, headers=header, timeout=options.timeout_seconds,
            trust_env=False, follow_redirects=False,
        ) as client:
            result = request(client, cast(Operation, options.operation), body)
        print(result.model_dump_json())
        return 0
    except (AMLError, OSError, ValueError) as error:
        code = error.code if isinstance(error, AMLError) else "invalid_request"
        print(json.dumps({"status": "failed", "code": code}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
