"""Explicit HTTP attempt evidence for opted-in benchmark callers.

The existing transport still owns dispatch, retry, validation and credentials.
This observer records each client send, including failures before a response and
usage on responses later rejected by the model adapter. It does not price calls.
Runtime failures retain a bounded, credential-redacted provider code/message.
"""

from __future__ import annotations

import asyncio
import json
import secrets
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Literal, cast

import httpx

from quotemem.bench.providers import AnswerRequest, RubricRequest
from quotemem.providers.adapters import current_inference_purpose
from quotemem.providers.error_diagnostics import provider_error_body
from quotemem.service.request_measurement import current_resource_context

_CALL: ContextVar[tuple[object, int] | None] = ContextVar("aml_bench_census_call", default=None)


class ProviderCallCensus:
    """Append and flush scalar observations; seal only after the caller joins its clients."""

    def __init__(self, path: Path, *, boundary: Literal["answers", "runtime"]) -> None:
        self._stream = path.open("x", encoding="utf-8", newline="\n")
        self._lock = threading.Lock()
        self._boundary = boundary
        self._identity = secrets.token_hex(16)
        self._logical = self._attempts = self._unattributed = 0
        self._active_calls: set[int] = set()
        self._active_attempts: set[int] = set()
        self._judge_deadlines: dict[int, Callable[[], bool]] = {}
        self._closed = False
        self._write({"event": "opened", "schema_version": 1,
                     "kind": "provider_http_call_census", "census_id": self._identity,
                     "boundary": boundary, "started_epoch_ns": time.time_ns(),
                     "usd": "unknown", "retry_owner": "existing_json_transport"})

    def _write(self, value: dict[str, object]) -> None:
        self._stream.write(json.dumps(value, sort_keys=True, ensure_ascii=True) + "\n")
        self._stream.flush()

    @contextmanager
    def call(self, request: AnswerRequest | RubricRequest) -> Iterator[None]:
        """Bind transport attempts to one existing sealed answer or rubric request."""
        with self._lock:
            if self._closed:
                raise ValueError("provider_census_closed")
            self._logical += 1
            number = self._logical
            self._active_calls.add(number)
            self._write({"event": "call_started", "call": number,
                         "case_id": request.context.case_id,
                         "request_sha256": request.request_sha256,
                         "purpose": request.purpose, "provider": request.policy.provider,
                         "requested_model": request.policy.requested_model,
                         "started_epoch_ns": time.time_ns()})
        token = _CALL.set((self, number))
        started = time.perf_counter_ns()
        outcome = "returned"
        try:
            yield
        except BaseException as error:
            outcome = type(error).__name__
            raise
        finally:
            _CALL.reset(token)
            with self._lock:
                self._write({"event": "call_finished", "call": number,
                             "outcome": outcome, "elapsed_ns": time.perf_counter_ns() - started})
                self._active_calls.remove(number)

    def started(self, request: httpx.Request) -> tuple[int, int]:
        """Observe client entry without retaining a URL, headers, body or credential.

        A runtime structured inference send also names its role as ``purpose``
        (``extraction``, ``judge`` or ``labels``); no other send carries the tag.
        """
        current = _CALL.get()
        purpose = current_inference_purpose()
        call = current[1] if current is not None and current[0] is self else None
        context = current_resource_context()
        endpoint = next((name for name in ("chat/completions", "embeddings", "rerank")
                         if request.url.path.endswith("/" + name)), "other")
        with self._lock:
            if self._closed:
                raise ValueError("provider_census_closed")
            self._attempts += 1
            number = self._attempts
            self._active_attempts.add(number)
            expired = request.extensions.get("aml_judge_deadline_expired")
            if "aml_judge_attempt" in request.extensions and callable(expired):
                self._judge_deadlines[number] = cast(Callable[[], bool], expired)
            if call is None and self._boundary == "answers":
                self._unattributed += 1
            row: dict[str, object] = {
                "event": "attempt_started", "attempt": number,
                "call_key": f"{self._identity}:{number}", "call": call, "endpoint": endpoint,
                "method": request.method, "started_epoch_ns": time.time_ns(),
                "runtime_scope_id": None if context is None else context.scope_id,
                "runtime_phase": None if context is None else context.phase,
                "runtime_run_id": None if context is None else context.run_id,
            }
            if purpose is not None:
                row["purpose"] = purpose
            for extension, field in (
                ("aml_judge_attempt", "judge_attempt"),
                ("aml_judge_request_timeout_ms", "judge_request_timeout_ms"),
                ("aml_judge_timeout_policy_digest", "judge_timeout_policy_digest"),
                ("aml_judge_timeout_ms", "timeout_ms"),
                ("aml_request_sha256", "request_sha256"),
            ):
                if extension in request.extensions:
                    row[field] = request.extensions[extension]
            self._write(row)
        return number, time.perf_counter_ns()

    def finished(
        self, started: tuple[int, int], response: httpx.Response | None,
        error: BaseException | None,
    ) -> None:
        """Retain actual usage even when a later adapter rejects the response schema.

        ``cached_tokens`` is the part of ``prompt_tokens`` the provider reports it
        served from its prompt cache (``prompt_tokens_details``), or None.
        ``response_model`` is the model the response reports, the snapshot it actually
        resolved to, or None.
        """
        usage: dict[str, int | None] | None = None
        model: str | None = None
        if response is not None and response.is_stream_consumed and (
            len(response.content) <= 2097152
        ):
            try:
                raw: object = response.json()
            except (ValueError, UnicodeError):
                raw = None
            if isinstance(raw, dict):
                reported: object = cast(dict[str, object], raw).get("model")
                if isinstance(reported, str):
                    model = reported
                supplied: object = cast(dict[str, object], raw).get("usage")
                if isinstance(supplied, dict):
                    values = cast(dict[str, object], supplied)
                    details = values.get("prompt_tokens_details")
                    cached = (cast(dict[str, object], details).get("cached_tokens")
                              if isinstance(details, dict) else None)
                    usage = {key: value if type(value) is int and value >= 0 else None
                             for key, value in (
                                 *((key, values.get(key)) for key in (
                                     "prompt_tokens", "completion_tokens", "total_tokens")),
                                 ("cached_tokens", cached))}
        number, at_ns = started
        timeout = next((name for name, cls in (
            ("connect", httpx.ConnectTimeout), ("pool", httpx.PoolTimeout),
            ("write", httpx.WriteTimeout), ("read", httpx.ReadTimeout),
        ) if isinstance(error, cls)), None)
        with self._lock:
            expired = self._judge_deadlines.pop(number, None)
            if isinstance(error, asyncio.CancelledError) and expired is not None and expired():
                timeout = "deadline"
            row: dict[str, object] = {"event": "attempt_finished", "attempt": number,
                         "elapsed_ns": time.perf_counter_ns() - at_ns,
                         "status_code": None if response is None else response.status_code,
                         "outcome": "response" if error is None else type(error).__name__,
                         "timeout_phase": timeout, "usage": usage, "response_model": model,
                         "cost_usd": None, "billing_witness_sha256": None}
            if self._boundary == "runtime" and response is not None and (
                response.status_code != 200 and response.is_stream_consumed
            ):
                row["provider_error"] = provider_error_body(response)
            self._write(row)
            self._active_attempts.remove(number)

    def close(self) -> None:
        """Seal a joined call census; unfinished work remains explicitly incomplete."""
        with self._lock:
            if self._closed:
                return
            try:
                self._write({"event": "closed", "census_id": self._identity,
                             "logical_calls": self._logical, "http_attempts": self._attempts,
                             "unfinished_calls": sorted(self._active_calls),
                             "unfinished_attempts": sorted(self._active_attempts),
                             "unattributed_answer_attempts": self._unattributed,
                             "capture_complete": not (self._active_calls or self._active_attempts
                                                       or self._unattributed),
                             "finished_epoch_ns": time.time_ns(), "cost_usd": None})
            finally:
                self._closed = True
                self._stream.close()


class CensusClient(httpx.Client):
    """Observe the existing synchronous client boundary, retaining default proxy semantics."""

    def __init__(self, census: ProviderCallCensus, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._census = census

    def send(self, request: httpx.Request, **kwargs: Any) -> httpx.Response:
        started = self._census.started(request)
        try:
            response = super().send(request, **kwargs)
        except BaseException as error:
            self._census.finished(started, None, error)
            raise
        self._census.finished(started, response, None)
        return response


class CensusAsyncClient(httpx.AsyncClient):
    """Observe bounded asynchronous provider sends through the runtime's existing factory."""

    def __init__(self, census: ProviderCallCensus, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._census = census

    async def send(self, request: httpx.Request, **kwargs: Any) -> httpx.Response:
        started = self._census.started(request)
        try:
            response = await super().send(request, **kwargs)
        except BaseException as error:
            self._census.finished(started, None, error)
            raise
        self._census.finished(started, response, None)
        return response
