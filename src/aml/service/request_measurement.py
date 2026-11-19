"""Bounded, content-free observations from the existing HTTP and finalize owners.

These records are measurement inputs. They neither authorize a response nor
prove gold matching. The real replay consumer must bind them to its registered
workload and source/build identity; missing or invalid collection is a nonpass.
"""

from __future__ import annotations

import threading
import time
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from pydantic import TypeAdapter, ValidationError

from aml.core.identity import ULID, ScopeId, canonical_json, integrity_digest
from aml.core.request_budget import RequestBudgetObservation
from aml.core.timing import FinalizeCompletion as FinalizeCompletion
from aml.core.timing import RequestDurations, RequestTiming, request_timing
from aml.service.cost_attribution import ResourcePhase
from aml.service.resource_observer import (
    ResourceChannel,
    ResourceCycle,
    ResourceObserver,
    ResourceSnapshot,
)

if TYPE_CHECKING:
    from aml.core.add_outcome import CompletedAddOutcomeV1
    from aml.core.controls import CompletedPrefixV2
    from aml.core.extraction_records import ExtractionAdmissionV2
    from aml.ingestion.service import AddReceipt
    from aml.service.h11_measurement import H11Enqueue, ResponseBinding

type RequestOperation = Literal["add", "search"]
type ResponseEvent = Literal[
    "search_enqueue", "add_response_ready", "add_enqueue_confirmed", "error_response_ready",
]
type TimeoutReason = Literal["provider_timeout", "delivery_deadline"]
_SCOPE: TypeAdapter[str] = TypeAdapter(ScopeId)
_RUN: TypeAdapter[str] = TypeAdapter(ULID)


def _scope_matches(value: str) -> bool:
    try:
        _SCOPE.validate_python(value, strict=True)
        return True
    except ValidationError:
        return False


def _run_matches(value: str) -> bool:
    try:
        _RUN.validate_python(value, strict=True)
        return True
    except ValidationError:
        return False


@dataclass(frozen=True, slots=True)
class AddPrefixObservation:
    """The exact validated durable prefix observed before this Add response.

    The immutable JSON contains identities and digests, never source text. It is
    captured when the controller proves completion; a terminal observer cannot
    reconstruct this earlier observation from the final scope marker.
    """

    reception_run_id: ULID
    visibility_commit_seq: int
    verified_at_ns: int
    completed_prefix_json: str
    completed_prefix_sha256: str


@dataclass(frozen=True, slots=True)
class ExtractionObservation:
    """Content-free outcome of one fresh extraction reply at program admission (B.7).

    ``notes`` counts the notes of the reply's admission record
    (``ExtractionAdmissionV2``) as ``(key, count)``: ``declaration:{rule}``,
    ``fact:{rule}``, ``group:{rule}``, ``anchor:{rule}``,
    ``field:{field}:{rule}`` and ``merge`` for an in-reply duplicate.
    ``overflow`` counts facts beyond the reply's fact bound
    (``fact_count_exceeded``). ``refusal`` instead names the B.7.1 whole-output
    fault that refused the reply before storage: ``provider_schema_invalid``,
    ``truncated``, ``output_bound`` or ``no_fresh_source``.
    """

    notes: tuple[tuple[str, int], ...] = ()
    overflow: int = 0
    refusal: str | None = None

    @classmethod
    def from_admission(cls, record: ExtractionAdmissionV2 | None) -> ExtractionObservation:
        """Count one admitted reply's record; None means admission changed nothing."""
        if record is None:
            return cls()
        notes: Counter[str] = Counter()
        notes.update(f"declaration:{note.rule}" for note in record.declarations)
        notes.update(f"fact:{note.rule}" for note in record.facts)
        notes.update(f"group:{note.rule}" for note in record.groups)
        notes.update(f"anchor:{note.rule}" for note in record.anchors)
        notes.update(f"field:{note.field}:{note.rule}" for note in record.fields)
        notes.update("merge" for _ in record.merges)
        return cls(
            tuple(sorted(notes.items())),
            sum(note.rule == "fact_count_exceeded" for note in record.facts),
        )


_TRUNCATION_FAULTS = ("truncated", "output_bound")
"""B.7.1 output-length faults; ``truncation_count`` is their sum (E.4, G-Q5)."""


def _extraction_view(counts: Counter[str]) -> dict[str, object]:
    """Name one tally's counts as the E.4 metrics read them."""

    def group(prefix: str) -> dict[str, int]:
        return {key.removeprefix(prefix): value for key, value in sorted(counts.items())
                if key.startswith(prefix)}

    return {
        "admitted_replies": counts["replies"],
        "admission_counts": group("note:"),
        "overflow_count": counts["overflow"],
        "truncation_count": sum(counts[f"refusal:{fault}"] for fault in _TRUNCATION_FAULTS),
        "whole_output_refusals": group("refusal:"),
    }


@dataclass(frozen=True, slots=True)
class RequestMeasurement:
    """An actual response event, or an explicitly incomplete request.

    The SHA-256 binds complete body bytes to an independently registered replay
    request. It excludes headers and credentials. Add's readiness event is named
    exactly and remains separate from an optional original H11 final-send
    confirmation. The confirmation is after transport writes, not a peer receipt.
    No missing response gets a synthetic status. ``add_recovery_count`` is the
    explicit-resume count of the completed result the controller returned for
    this Add (0 for an ordinary completion), or None when none was observed.
    """

    operation: RequestOperation
    request_sha256: str
    scope_id: ScopeId | None
    received_at_ns: int
    response_event: ResponseEvent | None
    response_at_ns: int | None
    status_code: int | None
    durations: RequestDurations | None
    timeout_reason: TimeoutReason | None
    add_response_ready_at_ns: int | None = None
    add_enqueue: H11Enqueue | None = None
    add_completion: AddPrefixObservation | None = None
    add_outcome_json: str | None = None
    add_recovery_count: int | None = None


class MeasurementCollector:
    """Collect bounded scalar records without I/O, background work or durable state.

    Capacity is selected before the load from its complete registered request and
    retry bounds. Overflow or a malformed/duplicate producer event only latches
    observation failure; application execution remains unchanged. Seal only after
    the caller has joined its real request and finalize owners. The collector
    checks its observed request lifetimes, never substitutes for that drain.
    """

    def __init__(
        self, *, request_capacity: int, finalize_capacity: int,
        resources: ResourceObserver | None = None,
    ) -> None:
        if any(type(value) is not int or value < 1 for value in (
            request_capacity, finalize_capacity,
        )):
            raise ValueError("Measurement capacities must be positive registered bounds")
        self._request_capacity = request_capacity
        self._finalize_capacity = finalize_capacity
        self._resources = resources
        self._lock = threading.Lock()
        self._active: set[RequestMeasurementOwner] = set()
        self._requests: list[RequestMeasurement] = []
        self._request_budgets: list[RequestBudgetObservation] = []
        self._finalizes: dict[tuple[str, str, int], FinalizeCompletion] = {}
        self._extraction: dict[str, Counter[str]] = {}
        self._invalid = self._sealed = False

    @property
    def resources(self) -> ResourceObserver | None:
        """Optional resource backing sharing this collector's capture lifetime."""
        return self._resources

    def resource_observation(self) -> ResourceSnapshot | None:
        """Read raw resource evidence without granting an independent seal or drain.

        Args:
            None.

        Returns:
            ResourceSnapshot | None: Content-free records, or None when unconfigured.
        """
        return None if self._resources is None else self._resources.observed()

    def begin(
        self, operation: RequestOperation, request_sha256: str, received_at_ns: int,
    ) -> RequestMeasurementOwner | None:
        """Reserve one complete-byte request before parsing or thread submission."""
        with self._lock:
            if (
                self._sealed or len(self._active) + len(self._requests) >= self._request_capacity
                or type(operation) is not str or operation not in {"add", "search"}
                or type(request_sha256) is not str or len(request_sha256) != 64
                or any(char not in "0123456789abcdef" for char in request_sha256)
                or type(received_at_ns) is not int or received_at_ns < 0
            ):
                self._invalid = True
                return None
            owner = RequestMeasurementOwner(self, operation, request_sha256, received_at_ns)
            self._active.add(owner)
            return owner

    def finalize_completed(self, event: FinalizeCompletion) -> None:
        """Observe successful marker validation, never infer it from a pool join."""
        with self._lock:
            if (
                self._sealed or type(event) is not FinalizeCompletion
                or not _scope_matches(event.scope_id) or not _run_matches(event.owner_run_id)
                or type(event.finalize_cycle) is not int or event.finalize_cycle < 1
                or type(event.completed_at_ns) is not int or event.completed_at_ns < 0
            ):
                self._invalid = True
                return
            key = (event.scope_id, event.owner_run_id, event.finalize_cycle)
            previous = self._finalizes.get(key)
            if previous is not None:
                # A genuine later replay of the same completed owner must not
                # replace its original observed completion with a later time.
                if event.completed_at_ns < previous.completed_at_ns:
                    self._invalid = True
                return
            if len(self._finalizes) >= self._finalize_capacity:
                self._invalid = True
                return
            self._finalizes[key] = event
        if self._resources is not None:
            self._resources.finalize_completed(event)

    def finalize_incomplete(self) -> None:
        """Latch an absent actual completion fact without changing application work."""
        self._invalidate()

    def _finish(self, owner: RequestMeasurementOwner, record: RequestMeasurement) -> None:
        with self._lock:
            if owner not in self._active:
                self._invalid = True
                return
            self._active.remove(owner)
            if (
                self._sealed or owner.invalid
                or record.response_event is None or record.durations is None
                or record.response_event == "add_response_ready"
            ):
                self._invalid = True
            self._requests.append(record)

    def _invalidate(self) -> None:
        with self._lock:
            self._invalid = True

    def seal(self) -> tuple[tuple[RequestMeasurement, ...], tuple[FinalizeCompletion, ...]]:
        """Return complete scalar observations after actual caller-owned draining.

        Invalid, incomplete, overflowing or still-active collection raises rather
        than yielding an apparently complete acceptance population. Records stay
        available through ``observed`` for raw nonpass evidence.
        """
        with self._lock:
            self._sealed = True
            if self._resources is not None:
                resources = self._resources.finish(time.monotonic_ns())
                if resources.invalid_reasons:
                    self._invalid = True
            if self._invalid or self._active or not self._requests:
                raise ValueError("Measurement collection is incomplete or invalid")
            return tuple(self._requests), tuple(self._finalizes.values())

    def observed(self) -> tuple[tuple[RequestMeasurement, ...], tuple[FinalizeCompletion, ...]]:
        """Copy raw scalar evidence after execution, without claiming completeness."""
        with self._lock:
            return tuple(self._requests), tuple(self._finalizes.values())

    def observe_budget(self, observation: RequestBudgetObservation) -> None:
        """Retain V2 request boundaries, including failures before a complete body."""
        with self._lock:
            if self._sealed or len(self._request_budgets) >= self._request_capacity:
                self._invalid = True
                return
            self._request_budgets.append(observation)

    def observed_budgets(self) -> tuple[RequestBudgetObservation, ...]:
        """Return server boundary facts without claiming client-side receipt."""
        with self._lock:
            return tuple(self._request_budgets)

    def observe_extraction(self, phase: str, observation: ExtractionObservation) -> None:
        """Count one fresh extraction reply under its phase; never gates extraction."""
        if type(observation) is not ExtractionObservation or any(
            value < 0 for _, value in observation.notes
        ) or observation.overflow < 0:
            self._invalidate()
            return
        with self._lock:
            if self._sealed:
                self._invalid = True
                return
            counts = self._extraction.setdefault(phase, Counter())
            if observation.refusal is not None:
                counts[f"refusal:{observation.refusal}"] += 1
                return
            counts["replies"] += 1
            counts["overflow"] += observation.overflow
            for key, value in observation.notes:
                counts["note:" + key] += value

    def extraction_summary(self) -> dict[str, object]:
        """Return admission and refusal counts over every phase, then per phase.

        The totals carry the keys acceptance metrics read from a measurement
        (``admission_counts``, ``overflow_count``, ``truncation_count``).
        """
        with self._lock:
            phases = {phase: Counter(counts) for phase, counts in self._extraction.items()}
        total: Counter[str] = Counter()
        for counts in phases.values():
            total.update(counts)
        return {**_extraction_view(total), "extraction_by_phase": {
            phase: _extraction_view(counts) for phase, counts in sorted(phases.items())
        }}


class RequestMeasurementOwner:
    """One existing request's scalar lifetime; no borrowed content crosses it."""

    def __init__(
        self, collector: MeasurementCollector, operation: RequestOperation,
        request_sha256: str, received_at_ns: int,
    ) -> None:
        self.timing = RequestTiming(received_at_ns)
        self._collector = collector
        self._operation: RequestOperation = operation
        self._request_sha256 = request_sha256
        self._scope_id: ScopeId | None = None
        self._response: tuple[ResponseEvent, int, int, RequestDurations | None] | None = None
        self._add_ready_at_ns: int | None = None
        self._add_binding: ResponseBinding | None = None
        self._add_outcome_json: str | None = None
        self._add_result: CompletedAddOutcomeV1 | None = None
        self._add_enqueue: H11Enqueue | None = None
        self._add_completion: AddPrefixObservation | None = None
        self._timeout_reason: TimeoutReason | None = None
        self._closed = self.invalid = False

    def _invalidate(self) -> None:
        self.invalid = True
        self._collector._invalidate()  # pyright: ignore[reportPrivateUsage]

    def scope(self, scope_id: ScopeId) -> None:
        """Observe exact parsed user scope; a second different scope invalidates only metrics."""
        if self._closed or not _scope_matches(scope_id) or (
            self._scope_id is not None and self._scope_id != scope_id
        ):
            self._invalidate()
            return
        self._scope_id = scope_id

    def response(self, event: ResponseEvent, status_code: int, at_ns: int) -> None:
        """Freeze the actual immutable response event before later owner cleanup."""
        if (
            self._closed or self._response is not None
            or type(status_code) is not int or not 100 <= status_code <= 599
            or type(at_ns) is not int or at_ns < self.timing.received_at_ns
            or status_code == 200 and self._scope_id is None
            or event not in {"search_enqueue", "add_response_ready", "error_response_ready"}
            or event == "search_enqueue" and (self._operation != "search" or status_code != 200)
            or event == "add_response_ready" and (self._operation != "add" or status_code != 200)
            or event == "error_response_ready" and status_code < 400
            or status_code == 200 and self._timeout_reason is not None
        ):
            self._invalidate()
            return
        if event == "add_response_ready":
            self._add_ready_at_ns = at_ns
            # Final send still belongs to this request. Readiness is retained
            # for diagnostics if the actual transport cannot confirm enqueue.
            self._response = (event, status_code, at_ns, None)
        else:
            self._response = (event, status_code, at_ns, self.timing.finish(at_ns))

    def add_response_ready(self, body: bytes, at_ns: int) -> None:
        """Bind exact response bytes and actual counts before the ASGI send.

        The acknowledgement must name this request's parsed user scope, and its
        six counts must equal an observed completed result. Either difference
        latches observation failure; the HTTP response itself is unchanged.
        """
        from aml.adapters.http.models import AddResponseV1
        from aml.core.errors import AMLError
        from aml.core.identity import scope_id
        from aml.service.h11_measurement import ResponseBinding

        try:
            binding = ResponseBinding.from_body(body)
            acknowledgement = AddResponseV1.from_body(body)
        except (AMLError, TypeError, ValueError):
            self._invalidate()
            return
        if scope_id("user", acknowledgement.user_id) != self._scope_id or (
            self._add_result is not None and acknowledgement.outcome != self._add_result.outcome
        ):
            self._invalidate()
            return
        self.response("add_response_ready", 200, at_ns)
        if not self.invalid and self._add_ready_at_ns == at_ns:
            self._add_binding = binding
            self._add_outcome_json = acknowledgement.outcome.model_dump_json()

    def completed_result(self, scope_id: ScopeId, completed: CompletedAddOutcomeV1) -> None:
        """Copy the completed result the controller returns, before the response.

        Args:
            scope_id: Actual proved scope, equal to this request's parsed scope.
            completed: This request's first completed result. Observed at most
                once; it never changes application authority.
        """
        from aml.core.add_outcome import CompletedAddOutcomeV1

        if (
            self._closed or self._operation != "add" or self._response is not None
            or self._add_result is not None or self._scope_id != scope_id
            or type(completed) is not CompletedAddOutcomeV1
        ):
            self._invalidate()
            return
        self._add_result = completed

    def completed_add(
        self, scope_id: ScopeId, receipt: AddReceipt, prefix: CompletedPrefixV2,
    ) -> None:
        """Copy an already validated completion, without changing application authority."""
        from aml.core.controls import CompletedPrefixV2
        from aml.ingestion.service import AddReceipt

        if (
            self._closed or self._operation != "add" or self._response is not None
            or self._add_completion is not None or self._scope_id != scope_id
            or type(receipt) is not AddReceipt or not _run_matches(receipt.run_id)
            or type(receipt.visibility_commit_seq) is not int or receipt.visibility_commit_seq < 1
            or type(prefix) is not CompletedPrefixV2
            or prefix.accepted_prefix_commit_seq < receipt.visibility_commit_seq
        ):
            self._invalidate()
            return
        payload = canonical_json(prefix)
        if len(payload.encode("utf-8")) > 16_384:
            self._invalidate()
            return
        self._add_completion = AddPrefixObservation(
            receipt.run_id, receipt.visibility_commit_seq, time.monotonic_ns(), payload,
            integrity_digest(prefix),
        )

    def response_binding(self) -> ResponseBinding | None:
        """Borrow only the exact pending Add acknowledgement's scalar binding."""
        if self._closed or self._operation != "add" or self._add_enqueue is not None:
            return None
        return self._add_binding

    @contextmanager
    def suspend_context(self) -> Iterator[None]:
        """Keep a pipelined server task from inheriting this request's observers."""
        token = _CURRENT.set(None)
        resource_token = _RESOURCE.set(None)
        try:
            with request_timing(None):
                yield
        finally:
            _RESOURCE.reset(resource_token)
            _CURRENT.reset(token)

    def transport_enqueued(self, event: H11Enqueue) -> None:
        """Finish timing only after exact original H11 final-send confirmation."""
        from aml.service.h11_measurement import H11Enqueue

        selected = self._response
        if (
            self._closed or self.invalid or self._operation != "add"
            or self._add_enqueue is not None or self._add_binding is None
            or selected is None or selected[0] != "add_response_ready"
            or type(event) is not H11Enqueue or event.response != self._add_binding
            or type(event.confirmed_at_ns) is not int or event.confirmed_at_ns < selected[2]
            or not event.uvicorn_version or not event.transport_type
        ):
            self._invalidate()
            return
        self._add_enqueue = event
        self._response = (
            "add_enqueue_confirmed", 200, event.confirmed_at_ns,
            self.timing.finish(event.confirmed_at_ns),
        )

    def transport_unmeasured(self) -> None:
        """Latch a pending successful Add's transport gap without altering HTTP."""
        if self._operation == "add" and self._add_ready_at_ns is not None \
                and self._add_enqueue is None:
            self._invalidate()

    def timeout(self, reason: TimeoutReason) -> None:
        """Observe a terminal timeout at its real boundary before error scrubbing.

        The worker supplies provider_timeout only for the final actual
        ProviderError timeout. The delivery owner supplies delivery_deadline
        only after its real deadline check. A generic 503, exhausted contention
        retry, cancellation, or an earlier retried provider attempt grants
        neither fact. Work after accepted enqueue cannot retract that response.
        """
        if self._response is not None:
            return
        if self._closed or reason not in {"provider_timeout", "delivery_deadline"}:
            self._invalidate()
            return
        if self._timeout_reason is None:
            self._timeout_reason = reason

    def close(self) -> None:
        """Observe the joined request exit; absent enqueue remains explicitly incomplete."""
        if self._closed:
            self._invalidate()
            return
        self._closed = True
        selected = self._response
        if selected is not None and selected[0] == "add_response_ready":
            selected = (selected[0], selected[1], selected[2], self.timing.finish(selected[2]))
        self._collector._finish(self, RequestMeasurement(  # pyright: ignore[reportPrivateUsage]
            operation=self._operation, request_sha256=self._request_sha256,
            scope_id=self._scope_id, received_at_ns=self.timing.received_at_ns,
            response_event=None if selected is None else selected[0],
            response_at_ns=None if selected is None else selected[2],
            status_code=None if selected is None else selected[1],
            durations=None if selected is None else selected[3],
            timeout_reason=self._timeout_reason,
            add_response_ready_at_ns=self._add_ready_at_ns,
            add_enqueue=self._add_enqueue,
            add_completion=self._add_completion,
            add_outcome_json=self._add_outcome_json,
            add_recovery_count=(
                None if self._add_result is None else self._add_result.recovery_count
            ),
        ))


_CURRENT: ContextVar[RequestMeasurementOwner | None] = ContextVar(
    "aml_request_measurement", default=None,
)


def observe_add_completion(
    scope_id: ScopeId, receipt: AddReceipt, prefix: CompletedPrefixV2,
) -> None:
    """Observe the real controller proof inside the current optional HTTP lifetime."""
    owner = _CURRENT.get()
    if owner is not None:
        owner.completed_add(scope_id, receipt, prefix)


def observe_add_result(scope_id: ScopeId, completed: CompletedAddOutcomeV1) -> None:
    """Observe the completed result the controller returns for the current Add.

    Args:
        scope_id: Actual proved scope of the request.
        completed: The request's first completed result, read from storage. A
            same-request replay after explicit repair returns it with its
            ``recovery_count``; the six-count HTTP response must equal its outcome.
    """
    owner = _CURRENT.get()
    if owner is not None:
        owner.completed_result(scope_id, completed)


@dataclass(frozen=True, slots=True)
class ResourceContext:
    """Scalar provider attribution inside an existing request or worker callback.

    ``measurement`` is the runtime collector that counts extraction admission
    for this phase, with or without resource backing.
    """

    resources: ResourceObserver | None
    scope_id: ScopeId
    phase: ResourcePhase
    run_id: ULID | None = None
    cycle: ResourceCycle | None = None
    measurement: MeasurementCollector | None = None


_RESOURCE: ContextVar[ResourceContext | None] = ContextVar(
    "aml_resource_measurement", default=None,
)


def current_resource_context() -> ResourceContext | None:
    """Read explicit worker attribution or the existing parsed HTTP request scope.

    Args:
        None.

    Returns:
        ResourceContext | None: Scalar attribution only; startup canaries and
            unmeasured callers have no registered capture context.
    """
    context = _RESOURCE.get()
    if context is not None:
        return context if context.resources is not None else None
    owner = _CURRENT.get()
    if owner is None or owner._closed or owner._scope_id is None:  # pyright: ignore[reportPrivateUsage]
        return None
    collector = owner._collector  # pyright: ignore[reportPrivateUsage]
    if collector.resources is None:
        return None
    return ResourceContext(
        collector.resources, owner._scope_id, owner._operation,  # pyright: ignore[reportPrivateUsage]
        measurement=collector,
    )


@contextmanager
def resource_phase(
    measurement: MeasurementCollector | None, *, scope_id: ScopeId, phase: ResourcePhase,
    run_id: ULID | None = None, cycle: ResourceCycle | None = None,
) -> Iterator[None]:
    """Attribute provider calls within an existing worker's joined callback.

    Args:
        measurement: Existing runtime collector, or None to disable this context.
        scope_id: Actual opaque user scope.
        phase: The request phase add or search, or the owner phase the Add owner
            is in (labels, extraction, embedding, judge_round_1, judge_round_2,
            commit). The owner sets it at each phase boundary, so every provider
            call and timing inside carries that key (eval-Q11).
        run_id: Actual Run doing the work; required for the owner phases.
        cycle: Actual owner scope/run/cycle; required for the owner phases.

    Yields:
        None: Calls retain scalar attribution without creating a worker or task.
    """
    token = _RESOURCE.set(ResourceContext(
        None if measurement is None else measurement.resources, scope_id, phase, run_id, cycle,
        measurement,
    ))
    try:
        from aml.service.cost_attribution import cost_span

        with cost_span("phase"):
            yield
    finally:
        _RESOURCE.reset(token)


@contextmanager
def resource_run(*, scope_id: ScopeId, run_id: ULID) -> Iterator[None]:
    """Bind an inherited worker phase to its verified provider-owning child.

    Args:
        scope_id: Actual child scope, which must equal the inherited phase scope.
        run_id: Actual child read and authorized by the existing executor.

    Yields:
        None: Attribution only; no context is invented for an unmeasured caller.
    """
    context = current_resource_context()
    if context is None:
        yield
        return
    if scope_id != context.scope_id:
        if context.resources is not None:
            context.resources.incomplete("provider")
        token = _RESOURCE.set(ResourceContext(
            None, scope_id, context.phase, run_id, measurement=context.measurement,
        ))
    else:
        token = _RESOURCE.set(ResourceContext(
            context.resources, scope_id, context.phase, run_id, context.cycle,
            context.measurement,
        ))
    try:
        yield
    finally:
        _RESOURCE.reset(token)


def observe_extraction(observation: ExtractionObservation) -> None:
    """Count one fresh extraction reply for the current capture, if one is configured.

    Args:
        observation: Content-free counts of one reply's admission or whole refusal.

    Returns:
        None: Measurement only; an unmeasured caller records nothing.
    """
    context = _RESOURCE.get()
    if context is not None:
        collector, phase = context.measurement, context.phase
    else:
        owner = _CURRENT.get()
        if owner is None or owner._closed:  # pyright: ignore[reportPrivateUsage]
            return
        collector = owner._collector  # pyright: ignore[reportPrivateUsage]
        phase = owner._operation  # pyright: ignore[reportPrivateUsage]
    if collector is not None:
        collector.observe_extraction(phase, observation)


def _resource_failed(resources: ResourceObserver, channel: ResourceChannel) -> None:
    try:
        resources.incomplete(channel)
    except Exception:
        pass


def resource_session_event(
    event: Literal["reception", "settled"], *,
    scope_id: ScopeId, run_id: ULID, session_hash: str | None = None,
    commit_seq: int | None = None, finalize_cycle: int | None = None,
) -> None:
    """Capture only scalar facts at the existing Writer's successful commit tails.

    Args:
        event: ``reception`` after a new Add write, or ``settled`` after the
            owner's actual release and settlement of its frozen prefix.
        scope_id: Actual proved scope, which must match the existing context.
        run_id: Actual committed Add (reception) or the Add owner (settled).
        session_hash: Actual admitted session target, only for reception.
        commit_seq: For reception, the new reception visibility sequence; for
            settled, the reception sequence the owner's frozen prefix ends at.
        finalize_cycle: Actual held lease generation, only for settled.

    Returns:
        None: No I/O, storage access, scheduling or callback authority. Invalidity
            and telemetry exceptions affect measurement only, never the commit.
    """
    context = current_resource_context()
    if context is None or context.resources is None:
        return
    resources = context.resources
    try:
        if scope_id != context.scope_id:
            resources.incomplete("session")
        elif event == "reception" and session_hash is not None and commit_seq is not None:
            resources.session_received(
                scope_id=scope_id, session_hash=session_hash, run_id=run_id,
                commit_seq=commit_seq, at_ns=time.monotonic_ns(),
            )
        elif event == "settled" and finalize_cycle is not None and commit_seq is not None:
            resources.scope_settled(
                ResourceCycle(scope_id, run_id, finalize_cycle), settled_through=commit_seq,
                at_ns=time.monotonic_ns(),
            )
        else:
            resources.incomplete("session")
    except Exception:
        _resource_failed(resources, "session")


@contextmanager
def resource_execution(
    *, scope_id: ScopeId, session_hash: str, run_id: ULID,
) -> Iterator[None]:
    """Count one existing extraction callback until its actual return or unwind.

    Args:
        scope_id: Actual verified child scope, equal to the current owner context.
        session_hash: Actual selected session target without external session text.
        run_id: Actual admitted extraction child, never a queue descriptor.

    Yields:
        None: Identity-bound telemetry retains concurrent callbacks of one session
            until each returns. The context owns no worker, task or cancellation.
    """
    context = current_resource_context()
    if context is None or context.resources is None:
        yield
        return
    resources = context.resources
    execution = None
    try:
        if scope_id != context.scope_id:
            resources.incomplete("session")
        else:
            execution = resources.execution_started(
                scope_id=scope_id, session_hash=session_hash, run_id=run_id,
                at_ns=time.monotonic_ns(),
            )
    except Exception:
        _resource_failed(resources, "session")
    try:
        yield
    finally:
        if execution is not None:
            try:
                resources.execution_finished(execution, at_ns=time.monotonic_ns())
            except Exception:
                _resource_failed(resources, "session")


def current_request_measurement() -> RequestMeasurementOwner | None:
    """Return the optional current request's observer, never application authority."""
    return _CURRENT.get()


@contextmanager
def measure_request(owner: RequestMeasurementOwner) -> Iterator[None]:
    """Install through the existing joined worker and release at actual request exit."""
    token = _CURRENT.set(owner)
    try:
        with request_timing(owner.timing):
            yield
    finally:
        _CURRENT.reset(token)
        owner.close()
