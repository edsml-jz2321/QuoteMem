"""Direct Model Studio/OpenAI HTTP adapters, with frozen retry policies.

External envelopes are parsed here; only strict application output crosses the
boundary. Official endpoint evidence is recorded in ADR 0004.
"""

from __future__ import annotations

import logging
import math
import os
import random
import re
import time
from asyncio import CancelledError as AsyncCancelledError
from collections.abc import Callable, Iterator
from concurrent.futures import CancelledError as FutureCancelledError
from contextlib import contextmanager
from contextvars import ContextVar
from functools import cache
from typing import Any, Final, Literal, cast, get_args

import httpx
from pydantic import ValidationError

from aml.core.errors import AMLError
from aml.core.extraction_bounds import MAX_USAGE
from aml.core.extraction_output import EXTRACT_V12_SCHEMA_ID
from aml.core.identity import integrity_digest
from aml.core.inference import InferenceOutput, ProviderUsage, RetryPolicyV1
from aml.core.judge_timeout import (
    JUDGE_MIN_ATTEMPT_MS,
    JUDGE_NON_READ_TIMEOUT_MS,
    judge_owner_wait_ms,
)
from aml.core.model_identity import DEEPSEEK_FLASH_MODEL, admits_inference_model
from aml.core.provider_budget import (
    EXTRACTION_REQUEST_CAP,
    JudgeTimeShortfall,
    ProviderCallBudgetV2,
)
from aml.core.provider_diagnostics import (
    ProviderAttemptV1,
    ProviderErrorClass,
    ProviderFailureDetailV1,
    ProviderFieldPath,
    ProviderOutputSchema,
    ProviderOutputValidationV1,
    ProviderSchemaReason,
    ProviderSchemaRefusalV1,
    TimeoutPhase,
    regenerable_refusal,
)
from aml.core.provider_output_diagnostics import describe_output_validation
from aml.core.request_budget import RequestBudget, current_budget
from aml.core.semantic import CURRENT_EXTRACTION_PIPELINES
from aml.providers.attempts import check_provider_attempt
from aml.providers.contracts import (
    EmbeddingRequest,
    EmbeddingResult,
    InferenceRequest,
    InferenceResult,
    OutputRefused,
    ProviderName,
    output_contract,
)
from aml.providers.deadline_transport import (
    AsyncClientFactory,
    ProviderDeadlineExpired,
    post_with_budget,
    wait_backoff,
)
from aml.service.cost_attribution import CostOperation, cost_span, current_cost_attribution
from aml.service.request_measurement import current_resource_context
from aml.service.resource_observer import ProviderInvocation, ProviderOutcome, ResourceUsage

ErrorClass = ProviderErrorClass

type CallPurpose = Literal["extraction", "judge", "labels"]
"""The E.4 role of a structured inference call (G-Q3 adds ``labels``)."""

_INFERENCE_PURPOSE: ContextVar[CallPurpose | None] = ContextVar(
    "aml_inference_purpose", default=None,
)

EXTRACT_REQUEST_EFFORT_ENV: Final = "AML_EXTRACT_REQUEST_EFFORT"
"""Environment variable that lowers the main extraction call's sent effort (see below)."""

type ExtractRequestEffort = Literal["low", "none"]
"""The main extraction call's sent effort override: ``low`` sends reasoning effort
``low`` in place of the binding's, ``none`` sends it without reasoning."""

_EXTRACT_REQUEST_EFFORTS = frozenset(get_args(ExtractRequestEffort.__value__))


@cache
def extract_request_effort() -> ExtractRequestEffort | None:
    """Read this process's main extraction effort override once, at its first use.

    The override changes only the outgoing HTTP payload of a main extraction call
    (``ChatCompletionsInference``); the request, its binding, prompt version, cost
    digest and every persisted record stay those of the frozen policy, so a workspace
    served with it reopens without it.

    Returns:
        ExtractRequestEffort | None: The override, or None when the variable is unset
        or empty.

    Raises:
        AMLError: schema_invalid for any other value.
    """
    value = os.environ.get(EXTRACT_REQUEST_EFFORT_ENV, "")
    if not value:
        return None
    if value not in _EXTRACT_REQUEST_EFFORTS:
        raise AMLError("schema_invalid", f"{EXTRACT_REQUEST_EFFORT_ENV} is low or none")
    return cast(ExtractRequestEffort, value)


DEADLINE_HEADER = "x-aml-deadline-ms"
"""Milliseconds left in one provider attempt, so a key proxy never waits past its caller."""


def current_inference_purpose() -> CallPurpose | None:
    """Name the E.4 role of the structured inference request this context is sending.

    Content-free attribution for an opted-in HTTP observer such as the benchmark call
    census (I14): ``extraction``, ``judge`` or ``labels`` inside
    ``ChatCompletionsInference.infer``, and None for every other send (embeddings,
    answers).

    Returns:
        CallPurpose | None: The sending request's role, or None outside inference.
    """
    return _INFERENCE_PURPOSE.get()


def _call_purpose(request: InferenceRequest) -> CallPurpose:
    """A source-label request is ``labels``; otherwise extraction or the judge."""
    if request.schema_id.startswith("source-labels-"):
        return "labels"
    return "extraction" if request.purpose == "extract" else "judge"


class ProviderError(AMLError):
    """A sanitized boundary failure whose classification controls frozen retries.

    An explicit deterministic embedding-input refusal uses ``invalid_request``;
    it retains the client-error diagnostic without becoming a retryable outage.
    """

    def __init__(
        self, error_class: ErrorClass, *, schema_refusal: ProviderSchemaRefusalV1 | None = None,
        invalid_input: bool = False, regenerable: bool = True,
    ) -> None:
        super().__init__(
            "invalid_request" if invalid_input else (
                "schema_invalid" if error_class == "schema_invalid" else "provider_unavailable"
            ),
            f"Provider request failed: {error_class}",
        )
        self.error_class: ErrorClass = error_class
        self.schema_refusal = schema_refusal
        self.regenerable = regenerable
        self.diagnostics: ProviderFailureDetailV1 | None = None


class ProviderContextLimitError(ProviderError):
    """An opt-in actual HTTP context refusal, without retaining provider text."""

    def __init__(self) -> None:
        super().__init__("client_error")


_EMBEDDING_INPUT_CODES = frozenset({
    "context_length_exceeded", "input_length_exceeded", "input_too_long", "empty_input",
    "invalid_input", "invalid_input_type", "invalid_input_length", "invalid_input_size",
    "string_above_max_length",
    "string_below_min_length", "array_above_max_length", "array_below_min_length",
})
_EMBEDDING_INPUT_MESSAGES = re.compile(
    r"\brange of (?:input length|text length|batch size) should be\s*\["
    r"|\b(?:input|texts?|prompt|content|batch)(?:\s+(?:value|parameter|list|string))?"
    r"['\"]?\s+(?:is|are|must(?: not)? be|cannot be|can not be|can't be|should(?: not)? be)\s+"
    r"(?:an?\s+)?(?:empty|blank|non-empty)\b"
    r"|\b(?:input|texts?|prompt|content)['\"]?\s+(?:must|should) be\s+"
    r"(?:an?\s+)?(?:string|list|array)\b"
    r"|\b(?:input|texts?|prompt|content)(?:\s+parameter)?['\"]?\s+"
    r"(?:is required|is missing|must be provided|contains? empty strings?)\b"
    r"|\b(?:input|text|prompt|context|sequence|batch)[ _-](?:length|size|count|tokens?)\b"
    r".{0,32}\b(?:exceeds?|exceeded|over|above|greater than|larger than|less than|"
    r"at most|at least|between|must be|should be)\b"
    r"|\b(?:input|text|prompt|context|sequence|batch)\s+(?:is\s+)?too (?:long|large)\b",
    re.IGNORECASE,
)


def _embedding_input_error(response: httpx.Response) -> bool:
    """Recognize explicit input validation; generic InvalidParameter is inconclusive.

    Model Studio also uses HTTP 400 InvalidParameter for temporary backend
    unavailability. Only an explicit input code or input-validation message makes
    the unchanged request terminal; unknown failures remain transient.
    """
    try:
        raw: object = response.json()
    except ValueError:
        raw = response.text
    if isinstance(raw, str):
        return _EMBEDDING_INPUT_MESSAGES.search(raw) is not None
    if not isinstance(raw, dict):
        return False
    body = cast(dict[str, Any], raw)
    error: object = body.get("error", body)
    if not isinstance(error, dict):
        return False
    detail = cast(dict[str, Any], error)
    code: object = detail.get("code", detail.get("Code"))
    message: object = detail.get("message", detail.get("Message"))
    return (
        isinstance(code, str) and code.casefold() in _EMBEDDING_INPUT_CODES
        or isinstance(message, str) and _EMBEDDING_INPUT_MESSAGES.search(message) is not None
    )


def _context_limit_response(response: httpx.Response, provider: ProviderName) -> bool:
    if response.status_code != 400:
        return False
    try:
        body: object = response.json()
    except ValueError:
        return False
    if not isinstance(body, dict):
        return False
    error: object = cast(dict[str, Any], body).get("error")
    if not isinstance(error, dict):
        return False
    detail = cast(dict[str, Any], error)
    code: object = detail.get("code")
    if code == "context_length_exceeded":
        return True
    message: object = detail.get("message")
    # Model Studio documents this specific input range under a generic parameter
    # code. Other invalid parameters, max_tokens ranges and arbitrary text fail closed.
    return provider == "model_studio" and code in (
        "InvalidParameter", "InternalError.Algo.InvalidParameter",
    ) and isinstance(message, str) and re.fullmatch(
        r"Range of input length should be \[1,\s*[1-9][0-9]*\]\.?(?: "
        r"\(InternalError\.Algo\.InvalidParameter\))?", message,
    ) is not None


def _schema_failure(
    reason: ProviderSchemaReason, path: ProviderFieldPath,
    *, output_validation: ProviderOutputValidationV1 | None = None, regenerable: bool = True,
) -> ProviderError:
    return ProviderError(
        "schema_invalid", schema_refusal=ProviderSchemaRefusalV1(
            reason=reason, field_path=path, output_validation=output_validation,
        ), regenerable=regenerable,
    )


_DESCRIBED_OUTPUTS = frozenset(get_args(ProviderOutputSchema.__value__))


def _output_validation(
    request: InferenceRequest, error: BaseException | None,
) -> ProviderOutputValidationV1 | None:
    """Describe a refused output only for a contract with a bounded diagnostic projection.

    The batch judge envelope has none: its members are bound and refused at the
    transport split, and a refused envelope is never persisted.
    """
    if request.schema_id not in _DESCRIBED_OUTPUTS:
        return None
    return describe_output_validation(request.purpose, request.schema_id, error)


def _regenerable(failure: ProviderError, status_code: int | None) -> bool:
    """Return whether an accepted HTTP reply failed in a way a regeneration may fix (B.5)."""
    return (
        failure.regenerable and regenerable_refusal(failure.schema_refusal)
        and status_code is not None and 200 <= status_code <= 299
    )


def _timeout_phase(error: httpx.TimeoutException) -> TimeoutPhase:
    if isinstance(error, httpx.ConnectTimeout):
        return "connect"
    if isinstance(error, httpx.PoolTimeout):
        return "pool"
    if isinstance(error, httpx.WriteTimeout):
        return "write"
    if isinstance(error, httpx.ReadTimeout):
        return "read"
    return "unknown"


class _ProviderObservation:
    """Scalar attempt accounting, owned only by the current adapter invocation.

    HTTP retries finish individually. Successful HTTP responses remain pending
    until the adapter validates its complete result, including aggregate embedding
    shape/model checks. Rejected aggregate results do not publish successful
    provider results for the contributing HTTP responses. No response is retained.
    """

    def __init__(self, snapshot: str) -> None:
        self._context = current_resource_context()
        self._snapshot = snapshot
        self._current: ProviderInvocation | None = None
        self._usage: ResourceUsage | None = None
        self._transport_attempt: ProviderAttemptV1 | None = None
        self._pending: list[
            tuple[ProviderInvocation, ResourceUsage | None, ProviderAttemptV1 | None]
        ] = []
        self.transport_attempts: list[ProviderAttemptV1] = []

    def _invalidate(self) -> None:
        context = self._context
        if context is not None and context.resources is not None:
            try:
                context.resources.incomplete("provider")
            except Exception:
                # Observation failure cannot replace the adapter's actual result.
                pass

    def begin(self) -> None:
        context = self._context
        if self._current is not None:
            self._pending.append((self._current, self._usage, self._transport_attempt))
        self._current, self._usage, self._transport_attempt = None, None, None
        if context is None or context.resources is None:
            return
        try:
            self._current = context.resources.provider_started(
                scope_id=context.scope_id, phase=context.phase, provider_snapshot=self._snapshot,
                cycle=context.cycle, run_id=context.run_id, at_ns=time.monotonic_ns(),
            )
        except Exception:
            self._invalidate()

    def response(self, raw: dict[str, Any], *, generation: bool = False) -> None:
        # Count only exact scalars present in this actual HTTP usage envelope.
        # Application _usage's non-generative zero is not a provider observation.
        values: object = raw.get("usage")
        if not isinstance(values, dict):
            return
        usage = cast(dict[str, Any], values)
        entered: object = usage.get("prompt_tokens") if generation else usage.get(
            "prompt_tokens", usage.get("total_tokens"),
        )
        emitted: object = usage.get("completion_tokens")
        input_tokens = entered if type(entered) is int and 0 <= entered <= MAX_USAGE else None
        output_tokens = emitted if type(emitted) is int and 0 <= emitted <= MAX_USAGE else None
        if (entered is not None and input_tokens is None) or (
            emitted is not None and output_tokens is None
        ):
            self._invalidate()
        if input_tokens is not None or output_tokens is not None:
            self._usage = ResourceUsage(input_tokens, output_tokens)

    @property
    def observed_usage(self) -> ResourceUsage | None:
        """Actual scalar usage from the current response, even without a capture sink."""
        return self._usage

    def transport_attempt(self, attempt: ProviderAttemptV1) -> None:
        """Keep only the current send's validated scalar observation."""
        if self._current is not None:
            if attempt.outcome == "timeout" and attempt.timeout_phase == "deadline" and (
                attempt.status_code is None
            ):
                # The budget can expire before client.post begins. Diagnostics
                # retain the attempt, but HTTP dispatch cannot be asserted.
                return
            self._transport_attempt = attempt

    def failed_attempt(self) -> None:
        if self._current is not None:
            self._finish_call(
                self._current, "failed", self._usage, self._transport_attempt,
            )
            self._current, self._usage, self._transport_attempt = None, None, None

    def _finish_call(
        self, call: ProviderInvocation, outcome: ProviderOutcome, usage: ResourceUsage | None,
        transport_attempt: ProviderAttemptV1 | None,
    ) -> None:
        context = self._context
        if context is not None and context.resources is not None:
            try:
                context.resources.provider_finished(
                    call, outcome=outcome, at_ns=time.monotonic_ns(), usage=usage,
                    transport_attempt=transport_attempt,
                )
            except Exception:
                self._invalidate()

    def finish(self, outcome: ProviderOutcome) -> None:
        if self._current is not None:
            self._pending.append((self._current, self._usage, self._transport_attempt))
            self._current, self._usage, self._transport_attempt = None, None, None
        for call, usage, transport_attempt in self._pending:
            self._finish_call(call, outcome, usage, transport_attempt)
        self._pending.clear()


@contextmanager
def _observe_provider(snapshot: str) -> Iterator[_ProviderObservation]:
    observation = _ProviderObservation(snapshot)
    try:
        yield observation
    except (AsyncCancelledError, FutureCancelledError):
        observation.finish("cancelled")
        raise
    except ProviderError as error:
        if error.diagnostics is None and observation.transport_attempts:
            error.diagnostics = ProviderFailureDetailV1(
                error_class=error.error_class, attempts=observation.transport_attempts,
                schema_refusal=error.schema_refusal,
            )
        observation.finish("failed")
        raise
    except BaseException:
        observation.finish("failed")
        raise
    else:
        observation.finish("succeeded")


def _provider_cost_digest(request: InferenceRequest | EmbeddingRequest) -> str | None:
    """Hash only a nonsecret semantic request when its existing collector opted in."""
    observer = current_cost_attribution()
    if observer is None:
        return None
    try:
        return integrity_digest(request)
    except Exception:
        try:
            observer.incomplete("producer_failed")
        except Exception:
            pass
        return None


class JsonTransport:
    """An explicit HTTP client and credential; no ambient SDK/configuration state."""

    def __init__(
        self,
        client: httpx.Client,
        base_url: str,
        api_key: str,
        *,
        sleep: Callable[[float], None] = time.sleep,
        random_fraction: Callable[[], float] = random.random,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
        async_client_factory: AsyncClientFactory | None = None,
    ) -> None:
        if not api_key.strip():
            raise ProviderError("unavailable")
        self._client = client
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._sleep = sleep
        self._random_fraction = random_fraction
        self._monotonic_ns = monotonic_ns
        self._async_client_factory = async_client_factory

    def post(
        self, path: str, payload: dict[str, Any], timeout_ms: int, policy: RetryPolicyV1,
        *, observation: _ProviderObservation | None = None,
        call_budget: ProviderCallBudgetV2 | None = None,
        context_limit_provider: ProviderName | None = None,
        accept_response: Callable[[dict[str, Any]], None] | None = None,
        retry_output_schema: bool = False,
        judge_attempt: Literal[1, 2] | None = None,
        request_extensions: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """POST with only the request's retry classes and bounded backoff schedule.

        ``policy.max_attempts`` is the one cap on the physical HTTP requests of
        this call, and a policy's ``max_read_timeouts`` makes the read timeout that
        reaches it final. A ``call_budget`` limits that cap to ``EXTRACTION_REQUEST_CAP``
        (3): network retries of the policy's retryable classes and output-schema
        regenerations count against it together, within the one logical
        deadline, and a fourth request never starts.

        ``accept_response`` checks each decoded 2xx reply inside its own attempt,
        so a refusal is recorded on the send that produced it. Only with
        ``retry_output_schema`` does an output-schema refusal of
        ``choices[0].message.content`` on a 2xx reply send the identical payload
        again. Every other refusal is final, and a reply that passes these checks
        returns without another request.
        """
        if call_budget is not None and policy.max_attempts > EXTRACTION_REQUEST_CAP:
            raise AMLError("binding_mismatch", "Extraction call budget permits at most three sends")
        if retry_output_schema and (call_budget is None or accept_response is None):
            raise AMLError("binding_mismatch", "Output-schema retry requires bounded validation")
        request_url = self._base_url + path
        delay = float(policy.initial_backoff_ms)
        budget = current_budget()
        logical_deadline = None if call_budget is None else (
            self._monotonic_ns() + call_budget.logical_wall_ms * 1_000_000
        )
        if budget is not None:
            budget.check()
        if budget is not None or call_budget is not None:
            if self._async_client_factory is None:
                raise AMLError(
                    "unsupported_variant", "Bounded provider requires an explicit async transport",
                )
        attempts: list[ProviderAttemptV1] = []
        if observation is not None:
            observation.transport_attempts = attempts
        failure: ProviderError | None = None

        def exhausted() -> ProviderError:
            assert failure is not None and attempts
            attempts[-1] = attempts[-1].model_copy(update={"decision": "budget_exhausted"})
            failure.diagnostics = ProviderFailureDetailV1(
                error_class=failure.error_class, attempts=attempts,
                schema_refusal=failure.schema_refusal, stop_reason="logical_budget_exhausted",
            )
            return failure

        for attempt in range(policy.max_attempts):
            if budget is not None:
                budget.check()
            remaining_ms: int | None = None
            wall_ms: int | None = None
            attempt_budget = budget
            phase_timeout: float | httpx.Timeout = timeout_ms / 1000
            if call_budget is not None:
                assert logical_deadline is not None
                remaining_ns = logical_deadline - self._monotonic_ns()
                if remaining_ns <= 0:
                    if attempts:
                        raise exhausted() from None
                    raise ProviderError("timeout")
                remaining_ms = math.ceil(remaining_ns / 1_000_000)
                seconds = min(call_budget.attempt_wall_ms / 1000, remaining_ns / 1e9)
                if budget is not None:
                    seconds = min(seconds, budget.remaining())
                    budget.check()
                wall_ms = max(1, math.ceil(seconds * 1000))
                attempt_budget = RequestBudget(seconds, clock=lambda: self._monotonic_ns() / 1e9)
                phase_timeout = httpx.Timeout(
                    connect=call_budget.connect_ms / 1000, pool=call_budget.pool_ms / 1000,
                    write=call_budget.write_ms / 1000, read=call_budget.read_ms / 1000,
                )
            check_provider_attempt()
            attempt_timeout_ms = timeout_ms
            extensions = request_extensions
            if judge_attempt is not None:
                # Reservation consumes owner time. Cap the read policy by the
                # remaining owner allowance; setup and backoff do not subtract
                # from 900/500 s independently. The owner also bounds whole HTTP.
                owner_wait_ms = judge_owner_wait_ms()
                if owner_wait_ms is None and budget is not None:
                    owner_wait_ms = math.floor(budget.remaining() * 1000)
                if owner_wait_ms is not None:
                    attempt_timeout_ms = min(timeout_ms, owner_wait_ms)
                    if attempt_timeout_ms < JUDGE_MIN_ATTEMPT_MS:
                        raise JudgeTimeShortfall
                if extensions is not None:
                    extensions = {**extensions, "aml_judge_timeout_ms": attempt_timeout_ms}
                phase_timeout = httpx.Timeout(
                    JUDGE_NON_READ_TIMEOUT_MS / 1000, read=attempt_timeout_ms / 1000,
                )
            deadline_ms = attempt_timeout_ms if attempt_budget is None else min(
                attempt_timeout_ms, math.floor(attempt_budget.remaining() * 1000),
            )
            headers = {
                "Authorization": f"Bearer {self._api_key}",
                # The key proxy caps its queueing and backoff below this attempt's own
                # deadline; the header carries no content and is ignored by gateways.
                DEADLINE_HEADER: str(max(0, deadline_ms)),
            }
            if observation is not None:
                observation.begin()
            started_ns = self._monotonic_ns()
            status_code: int | None = None
            timeout_phase: TimeoutPhase | None = None
            try:
                if attempt_budget is None:
                    response = self._client.post(
                        request_url,
                        json=payload,
                        headers=headers,
                        timeout=phase_timeout, extensions=extensions,
                    )
                else:
                    assert self._async_client_factory is not None
                    response = post_with_budget(
                        self._async_client_factory, attempt_budget, request_url, payload,
                        headers, phase_timeout,
                        parent=budget if call_budget is not None else None,
                        extensions=extensions,
                    )
                status_code = response.status_code
                if not 200 <= status_code <= 299 and observation is not None:
                    # Error replies may still report billed usage. Keep those
                    # actual scalars on this physical attempt; absent usage stays
                    # unknown and is never replaced with an estimated zero.
                    try:
                        error_body: object = response.json()
                    except ValueError:
                        error_body = None
                    if isinstance(error_body, dict):
                        observation.response(
                            cast(dict[str, Any], error_body),
                            generation=path == "/chat/completions",
                        )
                if response.status_code == 429:
                    raise ProviderError("rate_limited")
                if 500 <= response.status_code <= 599:
                    raise ProviderError("server_error")
                if response.status_code == 400 and path == "/embeddings":
                    if _embedding_input_error(response):
                        raise ProviderError("client_error", invalid_input=True)
                    raise ProviderError("server_error")
                if not 200 <= response.status_code <= 299:
                    if context_limit_provider is not None and _context_limit_response(
                        response, context_limit_provider,
                    ):
                        raise ProviderContextLimitError()
                    raise ProviderError("client_error")
                try:
                    raw: object = response.json()
                except ValueError:
                    raise _schema_failure("response_json", "$") from None
                if not isinstance(raw, dict):
                    raise _schema_failure("response_shape", "$")
                if budget is not None:
                    budget.check()
                if call_budget is not None:
                    assert attempt_budget is not None
                    attempt_budget.check()
                if accept_response is not None:
                    accept_response(cast(dict[str, Any], raw))
                if budget is not None:
                    budget.check()
                if call_budget is not None:
                    assert attempt_budget is not None
                    attempt_budget.check()
                observed = None if observation is None else observation.observed_usage
                recorded = ProviderAttemptV1(
                    attempt=attempt + 1,
                    elapsed_ms=max(0, (self._monotonic_ns() - started_ns) // 1_000_000),
                    timeout_ms=attempt_timeout_ms, judge_attempt=judge_attempt,
                    outcome="response", status_code=status_code,
                    decision="response", attempt_wall_ms=wall_ms,
                    logical_remaining_ms=remaining_ms,
                    input_tokens=None if observed is None else observed.input_tokens,
                    output_tokens=None if observed is None else observed.output_tokens,
                )
                attempts.append(recorded)
                if observation is not None:
                    observation.transport_attempt(recorded)
                return cast(dict[str, Any], raw)
            except httpx.TimeoutException as error:
                failure = ProviderError("timeout")
                timeout_phase = _timeout_phase(error)
            except httpx.HTTPError:
                failure = ProviderError("unavailable")
            except ProviderError as error:
                failure = error
            except AMLError as error:
                if judge_attempt is not None and isinstance(error, ProviderDeadlineExpired):
                    recorded = ProviderAttemptV1(
                        attempt=attempt + 1,
                        elapsed_ms=max(0, (self._monotonic_ns() - started_ns) // 1_000_000),
                        timeout_ms=attempt_timeout_ms, judge_attempt=judge_attempt,
                        outcome="timeout", timeout_phase="deadline", status_code=status_code,
                        decision="budget_exhausted",
                    )
                    attempts.append(recorded)
                    error.diagnostics = ProviderFailureDetailV1(
                        error_class="timeout", attempts=attempts,
                        stop_reason="logical_budget_exhausted",
                    )
                    if observation is not None:
                        observation.transport_attempt(recorded)
                        observation.failed_attempt()
                    raise
                if call_budget is None or error.code != "delivery_busy":
                    if status_code is not None:
                        observed = None if observation is None else observation.observed_usage
                        recorded = ProviderAttemptV1(
                            attempt=attempt + 1,
                            elapsed_ms=max(0, (self._monotonic_ns() - started_ns) // 1_000_000),
                            timeout_ms=attempt_timeout_ms, judge_attempt=judge_attempt,
                            outcome="response", status_code=status_code,
                            decision="response", attempt_wall_ms=wall_ms,
                            logical_remaining_ms=remaining_ms,
                            input_tokens=None if observed is None else observed.input_tokens,
                            output_tokens=None if observed is None else observed.output_tokens,
                        )
                        attempts.append(recorded)
                        if observation is not None:
                            observation.transport_attempt(recorded)
                    raise
                if budget is not None:
                    budget.check()
                failure = ProviderError("timeout")
                timeout_phase = "deadline"
            # Network retries and output-schema regenerations share this one
            # loop, so both draw on the same frozen attempts and deadline.
            retryable = failure.error_class in policy.retryable_error_classes or (
                retry_output_schema and _regenerable(failure, status_code)
            )
            # A policy may cap the attempts that end in a read timeout (C.5, JX-3):
            # the one that reaches the cap is final, as the last attempt is.
            capped = timeout_phase == "read" and policy.max_read_timeouts is not None and (
                1 + sum(1 for earlier in attempts if earlier.timeout_phase == "read")
                >= policy.max_read_timeouts
            )
            retry = retryable and not capped and attempt + 1 < policy.max_attempts
            backoff = 0.0
            if retry:
                factor = self._random_fraction() if policy.jitter_mode == "full" else 1.0
                backoff = delay * factor
            recorded = ProviderAttemptV1(
                attempt=attempt + 1,
                elapsed_ms=max(0, (self._monotonic_ns() - started_ns) // 1_000_000),
                timeout_ms=attempt_timeout_ms, judge_attempt=judge_attempt,
                outcome=failure.error_class, status_code=status_code,
                timeout_phase=timeout_phase,
                decision="retry" if retry else "exhausted" if retryable else "not_retryable",
                backoff_ms=round(backoff), attempt_wall_ms=wall_ms,
                logical_remaining_ms=remaining_ms,
                schema_refusal=failure.schema_refusal,
                input_tokens=(None if observation is None or observation.observed_usage is None
                              else observation.observed_usage.input_tokens),
                output_tokens=(None if observation is None or observation.observed_usage is None
                               else observation.observed_usage.output_tokens),
            )
            attempts.append(recorded)
            if observation is not None:
                observation.transport_attempt(recorded)
                observation.failed_attempt()
            if logical_deadline is not None and (
                self._monotonic_ns() >= logical_deadline or (
                    retry and self._monotonic_ns() + backoff * 1_000_000 >= logical_deadline
                )
            ):
                # No later send can fit after the selected backoff. Do not start
                # an attempt, reset the clock, or invent a successful response.
                attempts[-1] = attempts[-1].model_copy(update={"backoff_ms": 0})
                raise exhausted() from None
            if not retry:
                failure.diagnostics = ProviderFailureDetailV1(
                    error_class=failure.error_class, attempts=attempts,
                    schema_refusal=failure.schema_refusal,
                )
                raise failure from None
            if path == "/embeddings":
                logging.getLogger(__name__).warning(
                    "Provider retry endpoint=%s attempt=%d max_attempts=%d "
                    "status=%s error_class=%s backoff_ms=%d",
                    path, attempt + 1, policy.max_attempts, status_code,
                    failure.error_class, round(backoff),
                )
            with cost_span("provider.backoff"):
                if budget is None:
                    self._sleep(backoff / 1000)
                else:
                    wait_backoff(budget, backoff / 1000)
            delay = min(float(policy.max_backoff_ms), delay * policy.multiplier)
        raise AssertionError("A validated retry policy always has at least one attempt")


def _usage(raw: dict[str, Any], *, generation: bool = False) -> ProviderUsage:
    """Normalize documented token accounting without storing arbitrary provider JSON."""
    usage = raw.get("usage")
    if not isinstance(usage, dict):
        raise _schema_failure("usage_metadata", "usage")
    values = cast(dict[str, Any], usage)
    input_tokens = (
        values.get("prompt_tokens")
        if generation
        else values.get("prompt_tokens", values.get("total_tokens"))
    )
    output_tokens = values.get("completion_tokens") if generation else 0
    try:
        parsed = ProviderUsage.model_validate(
            {"input_tokens": input_tokens, "output_tokens": output_tokens}
        )
    except ValidationError:
        raise _schema_failure("usage_metadata", "usage") from None
    if max(parsed.input_tokens, parsed.output_tokens) > MAX_USAGE:
        raise _schema_failure("usage_metadata", "usage")
    return parsed


def _model(raw: dict[str, Any]) -> str:
    """Require actual model resolution; a requested model is not response evidence."""
    resolved: object = raw.get("model")
    if not isinstance(resolved, str) or not resolved:
        raise _schema_failure("model_metadata", "model")
    return resolved


class ChatCompletionsInference:
    """Structured chat over the OpenAI-compatible gateway, the one chat transport (B.0)."""

    def __init__(
        self, transport: JsonTransport, provider: ProviderName,
        *, extract_effort: ExtractRequestEffort | None = None,
    ) -> None:
        """Bind the gateway transport.

        Args:
            transport: The OpenAI-compatible gateway transport.
            provider: The selected inference provider; only ``openai``.
            extract_effort: The sent effort of main extraction calls
                (``extract_request_effort``); None sends each request's own.
        """
        if provider != "openai":
            raise AMLError(
                "unsupported_variant", "Chat runs only through the OpenAI-compatible gateway",
            )
        self._transport = transport
        self._provider: ProviderName = provider
        self._extract_effort = extract_effort

    def infer(self, request: InferenceRequest) -> InferenceResult:
        """Validate schema identity, invoke strict JSON output, and parse semantic data.

        One invocation is one logical call with one ``JsonTransport.post``. A
        contract-valid reply returns whatever its semantic content, so grounding,
        judge, identity, reference and correction verdicts of the caller never
        cause another request. After a regenerated extraction reply,
        ``provider_attempts`` keeps every physical request of the call.
        """
        if request.provider != self._provider:
            raise AMLError("binding_mismatch", "Request provider differs from the selected adapter")
        deepseek = request.requested_model == DEEPSEEK_FLASH_MODEL
        if request.prompt_version in CURRENT_EXTRACTION_PIPELINES and (
            not deepseek or request.purpose != "extract"
        ):
            raise AMLError("binding_mismatch", "DeepSeek extraction differs from its frozen model")
        if deepseek and request.reasoning_effort is None:
            raise AMLError("binding_mismatch", "A DeepSeek call binds its reasoning effort")
        contract = output_contract(request.purpose, request.schema_id)
        payload: dict[str, Any] = {
            "model": request.requested_model,
            "messages": [message.model_dump(mode="json") for message in request.messages],
            "stream": False,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": request.schema_id.replace("-", "_"),
                    "strict": True,
                    "schema": contract.advertise(request),
                },
            },
        }
        if request.thinking == "disabled":
            # The call goes without reasoning; the gateway reads no effort then.
            payload["thinking"] = {"type": "disabled"}
        elif request.reasoning_effort is not None:
            # Every binding of a current pipeline carries its reasoning effort (C.5).
            payload["reasoning_effort"] = request.reasoning_effort
        if (self._extract_effort is not None and request.purpose == "extract"
                and request.schema_id == EXTRACT_V12_SCHEMA_ID and request.thinking is None):
            # The main extraction call alone: its supplement goes without reasoning
            # already, and the request itself keeps its frozen binding.
            if self._extract_effort == "none":
                payload.pop("reasoning_effort", None)
                payload["thinking"] = {"type": "disabled"}
            else:
                payload["reasoning_effort"] = self._extract_effort
        if request.max_output_tokens is not None:
            payload["max_tokens"] = request.max_output_tokens
        snapshot = (
            f"{request.provider}:{request.requested_model}:chat/completions:{request.schema_id}"
        )
        digest = _provider_cost_digest(request)
        with cost_span(
            cast(CostOperation, f"provider.{request.purpose}"), logical_request_sha256=digest,
        ), _observe_provider(snapshot) as observation:
            accepted: list[tuple[str, ProviderUsage, str | None, InferenceOutput]] = []

            def parse_output(raw: dict[str, Any]) -> InferenceOutput:
                choices: object = raw.get("choices")
                if not isinstance(choices, list) or len(cast(list[Any], choices)) != 1:
                    raise _schema_failure("response_shape", "choices")
                choice: object = cast(list[Any], choices)[0]
                if not isinstance(choice, dict):
                    raise _schema_failure("response_shape", "choices[0]")
                choice_values = cast(dict[str, Any], choice)
                message: object = choice_values.get("message")
                if not isinstance(message, dict):
                    raise _schema_failure("response_shape", "choices[0].message")
                message_values = cast(dict[str, Any], message)
                finish_reason = choice_values.get("finish_reason")
                if finish_reason != "stop":
                    # Filtering is deterministic for this input. Keep the length-fault
                    # split path, but do not regenerate the identical request first.
                    raise _schema_failure(
                        "truncated", "choices[0].finish_reason",
                        regenerable=finish_reason != "content_filter",
                    )
                if message_values.get("refusal"):
                    raise _schema_failure("message_refusal", "choices[0].message.refusal")
                content: object = message_values.get("content")
                if not isinstance(content, str):
                    raise _schema_failure("response_shape", "choices[0].message.content")
                try:
                    return contract.decode(content)
                except OutputRefused as refused:
                    # The contract's fault becomes the refusal reason; only
                    # schema-owned paths and categories cross this boundary.
                    raise _schema_failure(
                        refused.fault, "choices[0].message.content",
                        output_validation=_output_validation(request, refused.validation),
                    ) from None
                except TypeError:
                    raise _schema_failure("output_schema", "choices[0].message.content") from None

            def accept_response(raw: dict[str, Any]) -> None:
                # Every send checks usage and the resolved model before its
                # output, so each recorded attempt, a schema refusal included,
                # carries actual usage from a verified model identity.
                observation.response(raw, generation=True)
                if not admits_inference_model(
                    request.provider, request.requested_model, raw.get("model"),
                ):
                    raise AMLError(
                        "binding_mismatch", "Gateway returned a different model identity",
                    )
                resolved = _model(raw)
                usage = _usage(raw, generation=True)
                request_id: object = raw.get("id")
                if request_id is not None and not isinstance(request_id, str):
                    raise _schema_failure("request_metadata", "id")
                accepted.append((resolved, usage, request_id, parse_output(raw)))

            purpose = _INFERENCE_PURPOSE.set(_call_purpose(request))
            try:
                self._transport.post(
                    "/chat/completions", payload, request.timeout_ms, request.retry_policy,
                    observation=observation, call_budget=request.call_budget,
                    accept_response=accept_response,
                    retry_output_schema=request.retry_output_schema,
                    judge_attempt=request.judge_attempt,
                    request_extensions=None if request.judge_attempt is None else {
                        "aml_judge_attempt": request.judge_attempt,
                        "aml_judge_timeout_policy_digest": request.judge_timeout_policy_digest,
                        "aml_judge_request_timeout_ms": request.timeout_ms,
                        "aml_judge_timeout_ms": request.timeout_ms,
                        "aml_request_sha256": integrity_digest(request),
                    },
                )
            finally:
                _INFERENCE_PURPOSE.reset(purpose)
            # The transport returns only after its final send passed every check
            # above; an earlier accepted send can still miss its attempt deadline.
            resolved_model, usage, request_id, output = accepted[-1]
            return InferenceResult(
                resolved_model=resolved_model, schema_id=request.schema_id, output=output,
                usage=usage, provider_request_id=request_id,
                provider_attempts=(
                    list(observation.transport_attempts)
                    if request.retry_output_schema and any(
                        regenerable_refusal(attempt.schema_refusal)
                        for attempt in observation.transport_attempts[:-1]
                    ) else []
                ),
            )


class ModelStudioEmbedding:
    """Qwen embedding adapter with the documented twenty-input maximum per request."""

    def __init__(self, transport: JsonTransport) -> None:
        self._transport = transport

    def embed(self, request: EmbeddingRequest) -> EmbeddingResult:
        """Split bounded batches, restore input order, and reject inconsistent resolution."""
        snapshot = f"model_studio:{request.requested_model}:embeddings:{request.dimensions}"
        digest = _provider_cost_digest(request)
        with cost_span(
            "provider.embedding", logical_request_sha256=digest,
        ), _observe_provider(snapshot) as observation:
            vectors: list[list[float]] = []
            models: set[str] = set()
            tokens = 0
            for start in range(0, len(request.texts), 20):
                texts = request.texts[start : start + 20]
                raw = self._transport.post(
                    "/embeddings",
                    {
                        "model": request.requested_model,
                        "input": texts,
                        "dimensions": request.dimensions,
                        "encoding_format": "float",
                    },
                    request.timeout_ms,
                    request.retry_policy,
                    observation=observation,
                )
                observation.response(raw)
                models.add(_model(raw))
                tokens += _usage(raw).input_tokens
                entries: object = raw.get("data")
                if not isinstance(entries, list) or len(cast(list[Any], entries)) != len(texts):
                    raise _schema_failure("response_shape", "data")
                ordered: dict[int, list[float]] = {}
                for entry in cast(list[Any], entries):
                    if not isinstance(entry, dict):
                        raise _schema_failure("response_shape", "data")
                    item = cast(dict[str, Any], entry)
                    index = item.get("index")
                    raw_vector = item.get("embedding")
                    if not isinstance(raw_vector, list):
                        raise _schema_failure("response_shape", "data[].embedding")
                    vector = cast(list[Any], raw_vector)
                    if (
                        type(index) is not int
                        or index not in range(len(texts))
                        or index in ordered
                    ):
                        raise _schema_failure("response_shape", "data[].index")
                    if len(vector) != request.dimensions or any(
                        type(value) not in (int, float) for value in vector
                    ):
                        raise _schema_failure("response_shape", "data[].embedding")
                    try:
                        converted = [float(value) for value in vector]
                    except OverflowError:
                        raise _schema_failure("response_shape", "data[].embedding") from None
                    if any(not math.isfinite(value) for value in converted):
                        raise _schema_failure("response_shape", "data[].embedding")
                    ordered[index] = converted
                vectors.extend(ordered[index] for index in range(len(texts)))
            if len(models) != 1:
                raise _schema_failure("model_metadata", "model")
            try:
                return EmbeddingResult(
                    resolved_model=models.pop(),
                    dimensions=request.dimensions,
                    vectors=vectors,
                    usage=ProviderUsage(input_tokens=tokens, output_tokens=0),
                )
            except ValidationError:
                raise _schema_failure("response_shape", "data") from None
