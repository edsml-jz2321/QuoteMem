"""Closed safe HTTP error mapping from plan.md §2 PublicErrorV1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Self

from pydantic import ConfigDict, model_validator

from aml.core.errors import AMLError, RecoveryInProgress
from aml.store.metadata import StrictModel

HTTPErrorCode = Literal[
    "invalid_request",
    "authentication_failed",
    "scope_denied",
    "secret_rejected",
    "provider_unavailable",
    "authorization_unavailable",
    "index_rebuilding",
    "delivery_busy",
    "erasure_in_progress",
    "service_unhealthy",
    "operation_conflict",
    "extraction_refused",
    "repair_required",
]

_POLICY: dict[HTTPErrorCode, tuple[int, bool]] = {
    "invalid_request": (422, False),
    "authentication_failed": (401, False),
    "scope_denied": (403, False),
    "secret_rejected": (422, False),
    # Every transient state answers 429: AMLB honors Retry-After only on 429
    # (up to 60 seconds) and retries 503/425/409 on its own short schedule,
    # which spends the 32 Add attempts long before a recovery ends.
    "provider_unavailable": (429, True),
    "authorization_unavailable": (429, True),
    "index_rebuilding": (429, True),
    "delivery_busy": (429, True),
    "erasure_in_progress": (429, True),
    # Genuine unavailability of this service, and every unmapped internal fault.
    "service_unhealthy": (503, True),
    "operation_conflict": (422, False),
    "extraction_refused": (422, False),
    # Terminal until an explicit completion repair; resending cannot succeed.
    "repair_required": (409, False),
}

_INTERNAL: dict[str, HTTPErrorCode] = {
    # A quarantined scope, or a completed owner that lacks this request's
    # result, can change only through explicit repair, never by resending.
    "scope_quarantined": "repair_required",
    "incomplete_coverage": "repair_required",
    # A terminal request identity requires a new request ID.
    "run_terminal": "operation_conflict",
    # Scheduling races and spent allowances that the same request outlives:
    # a later attempt joins the next turn, lease or renewed allowance.
    "finalize_pending": "delivery_busy",
    "reception_stale": "delivery_busy",
    "authorization_expired": "delivery_busy",
    "budget_exceeded": "delivery_busy",
    "extraction_busy": "delivery_busy",
    "lease_lost": "delivery_busy",
}
"""Internal codes with a deliberate public meaning; any other code fails closed.

The forget routes answer ``reception_stale`` and ``budget_exceeded`` through
``_FORGET_INTERNAL`` first, so their terminal meaning there is unchanged.
"""

QUEUE_RETRY_AFTER_SECONDS = 15
"""An immediate queue or turn refusal: the next owner release or race outcome is near."""

WAITED_RETRY_AFTER_SECONDS = 5
"""A ``delivery_busy`` after a long in-request wait: the wait itself was the backoff."""

WAITED_REFUSAL_SECONDS = 60.0
"""In-request seconds after which a ``delivery_busy`` counts as having waited."""

RECOVERY_RETRY_AFTER_SECONDS = 60
"""Owner resumption, maintenance, erasure and provider outages: AMLB's 60-second cap.

A resumed owner holds a fresh 1,740-second allowance only after its previous
callback joins. Retrying at the cap lets the next attempt's own 1,740-second
wait outlast that allowance, so one attempt spans one whole resumption cycle.
"""

_RETRY_AFTER: dict[HTTPErrorCode, int] = {
    "provider_unavailable": RECOVERY_RETRY_AFTER_SECONDS,
    "authorization_unavailable": RECOVERY_RETRY_AFTER_SECONDS,
    "index_rebuilding": RECOVERY_RETRY_AFTER_SECONDS,
    "delivery_busy": QUEUE_RETRY_AFTER_SECONDS,
    "erasure_in_progress": RECOVERY_RETRY_AFTER_SECONDS,
    "service_unhealthy": RECOVERY_RETRY_AFTER_SECONDS,
}
"""Retry-After of each retryable public code for an immediate refusal."""

_FORGET_INTERNAL: dict[str, HTTPErrorCode] = {
    # An expired intent, a selection that changed under the prepared intent,
    # and an exhausted erasure budget each need a new prepare: a resend of the
    # same confirmation can never succeed (E.3, forget routes only).
    "intent_expired": "operation_conflict",
    "reception_stale": "operation_conflict",
    "budget_exceeded": "operation_conflict",
}
"""Internal codes the forget routes, and only they, answer as ``operation_conflict``."""

FORGET_CODES: frozenset[HTTPErrorCode] = frozenset({
    "invalid_request", "authentication_failed", "scope_denied", "operation_conflict",
    "erasure_in_progress", "index_rebuilding", "authorization_unavailable", "service_unhealthy",
})
"""The public codes ``/memory/forget/prepare`` and ``/memory/forget`` answer (E.3)."""

FORGET_STATEMENT_CODES: frozenset[HTTPErrorCode] = FORGET_CODES | frozenset[HTTPErrorCode]({
    "provider_unavailable", "delivery_busy",
})
"""The statement routes (FS-1) also answer their Search's and judge's retryable codes."""


class PublicErrorV1(StrictModel):
    """Only a closed HTTP code, retryability, and the same safe reason code.

    Details intentionally contain no original exception message or locator.
    This prevents body bytes, scope existence, and internal identities escaping
    even when a dependency raises an exception with sensitive diagnostics.
    """

    model_config = ConfigDict(frozen=True, hide_input_in_errors=True)
    code: HTTPErrorCode
    retryable: bool
    detail: HTTPErrorCode

    @model_validator(mode="after")
    def canonical_failure(self) -> Self:
        """Prevent mismatched retryability or arbitrary detail text."""
        if self.retryable is not _POLICY[self.code][1] or self.detail != self.code:
            raise ValueError("The public error fields do not match the closed policy")
        return self

    @property
    def status_code(self) -> int:
        """Return the one HTTP status the closed policy assigns to this code."""
        return _POLICY[self.code][0]

    @classmethod
    def from_response(cls, status_code: int, body: bytes) -> Self:
        """Validate a remote failure body together with its actual HTTP status.

        Args:
            status_code: The received HTTP status.
            body: The complete received JSON body; duplicate fields are rejected.

        Returns:
            The closed public failure, whose policy status equals ``status_code``.

        Raises:
            AMLError: ``service_unhealthy`` for an invalid body or a status that
                differs from the code's policy, such as a proxy's own failure page.
        """
        from aml.adapters.http.models import _object_body  # pyright: ignore[reportPrivateUsage]

        try:
            failure = cls.model_validate(_object_body(body, exact_numbers=False))
        except (AMLError, ValueError):
            raise AMLError("service_unhealthy", "HTTP failure body is invalid") from None
        if failure.status_code != status_code:
            raise AMLError("service_unhealthy", "HTTP failure status differs from its code")
        return failure


@dataclass(frozen=True)
class HTTPErrorResponse:
    """Safe public body and its mandatory HTTP status and retry delay."""

    status_code: int
    body: PublicErrorV1
    retry_after_seconds: int | None

    @property
    def headers(self) -> dict[str, str]:
        """Return Retry-After exactly when the public error is retryable.

        Args:
            None.

        Returns:
            The response headers, without credentials or application identifiers.
        """
        if self.retry_after_seconds is None:
            return {}
        return {"Retry-After": str(self.retry_after_seconds)}


class MappedFailure(AMLError):
    """A failure already mapped where it happened, re-raised across a worker boundary.

    The worker sees the internal cause and the request's elapsed time; the
    route answers later. Carrying the complete mapping keeps the chosen
    Retry-After instead of re-deriving it from the bare public code.
    """

    def __init__(self, mapped: HTTPErrorResponse) -> None:
        """Carry one complete public mapping without any diagnostic text."""
        super().__init__(mapped.body.code, mapped.body.code)
        self.mapped = mapped


def public_error(
    error: Exception, *, retry_after_seconds: int | None = None, waited_seconds: float = 0.0,
) -> HTTPErrorResponse:
    """Map application failures to safe HTTP fields without exposing diagnostics.

    Args:
        error: An application or unexpected dependency failure. Only recognized
            HTTP AMLError codes pass through; all other failures fail closed.
        retry_after_seconds: Trusted positive integer retry hint in seconds;
            None selects the state's policy delay.
        waited_seconds: Seconds the failing request already spent in its own
            budget; a long wait shortens a ``delivery_busy`` policy delay.

    Returns:
        The plan's status and retryability mapping. Internal codes listed in
        ``_INTERNAL`` take their deliberate public code; other internal failures
        and non-HTTP tool codes become service_unhealthy, never a successful empty
        result. Original exception messages are neither returned nor logged.
        A ``MappedFailure`` returns its carried mapping unchanged.

    Raises:
        ValueError: If the trusted retry delay is not a positive integer.
    """
    if isinstance(error, MappedFailure):
        return error.mapped
    code: HTTPErrorCode = "service_unhealthy"
    if isinstance(error, AMLError):
        code = error.code if error.code in _POLICY else _INTERNAL.get(error.code, code)
    if retry_after_seconds is None:
        retry_after_seconds = _policy_delay(
            code, recovering=isinstance(error, RecoveryInProgress), waited_seconds=waited_seconds,
        )
    return _response(code, retry_after_seconds)


def recode(mapped: HTTPErrorResponse, code: HTTPErrorCode) -> HTTPErrorResponse:
    """Answer a retryable mapping under another public code, keeping its retry delay.

    Args:
        mapped: The retryable mapping a route answers under its own code.
        code: The public code the route answers instead.

    Returns:
        The closed status and body for ``code`` with ``mapped``'s Retry-After.
    """
    return _response(code, mapped.retry_after_seconds or _RETRY_AFTER.get(code, 1))


def recovery_refusal() -> HTTPErrorResponse:
    """Return the ``delivery_busy`` an Add answers while its owner recovers."""
    return _response("delivery_busy", RECOVERY_RETRY_AFTER_SECONDS)


def _policy_delay(code: HTTPErrorCode, *, recovering: bool, waited_seconds: float) -> int:
    if recovering:
        return RECOVERY_RETRY_AFTER_SECONDS
    if code == "delivery_busy" and waited_seconds >= WAITED_REFUSAL_SECONDS:
        return WAITED_RETRY_AFTER_SECONDS
    return _RETRY_AFTER.get(code, 1)


def _response(code: HTTPErrorCode, retry_after_seconds: int) -> HTTPErrorResponse:
    if type(retry_after_seconds) is not int or retry_after_seconds < 1:
        raise ValueError("HTTP retry delay must be a positive integer")
    status, retryable = _POLICY[code]
    return HTTPErrorResponse(
        status,
        PublicErrorV1(code=code, retryable=retryable, detail=code),
        retry_after_seconds if retryable else None,
    )


def forget_public_error(
    error: Exception, *, retry_after_seconds: int | None = None, statement: bool = False,
) -> HTTPErrorResponse:
    """Map a forget route failure to its safe HTTP fields (E.3, forget routes).

    Args:
        error: A prepare or confirm failure from the forget operations or the
            route's own authentication and body validation.
        retry_after_seconds: Trusted positive integer retry hint in seconds;
            None selects the state's policy delay.
        statement: A statement route (FS-1), which also answers
            ``provider_unavailable`` and ``delivery_busy``.

    Returns:
        ``public_error``'s mapping, except that ``intent_expired``,
        ``reception_stale`` and ``budget_exceeded`` answer 422
        ``operation_conflict`` (never retryable), and a public code outside
        ``FORGET_CODES`` (``FORGET_STATEMENT_CODES``) fails closed to
        ``service_unhealthy``.

    Raises:
        ValueError: If the trusted retry delay is not a positive integer.
    """
    if isinstance(error, AMLError) and error.code in _FORGET_INTERNAL:
        return _response(_FORGET_INTERNAL[error.code], retry_after_seconds or 1)
    mapped = public_error(error, retry_after_seconds=retry_after_seconds)
    if mapped.body.code in (FORGET_STATEMENT_CODES if statement else FORGET_CODES):
        return mapped
    return public_error(
        AMLError("service_unhealthy", "service_unhealthy"), retry_after_seconds=retry_after_seconds,
    )
