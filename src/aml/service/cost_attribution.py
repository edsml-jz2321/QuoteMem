"""Bounded scalar cost spans backing the existing resource measurement lifetime."""

from __future__ import annotations

import threading
import time
from asyncio import CancelledError
from collections.abc import Callable, Iterator
from concurrent.futures import CancelledError as FutureCancelledError
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from functools import wraps
from typing import TYPE_CHECKING, Literal, get_args

from pydantic import TypeAdapter, ValidationError

from aml.core.identity import ULID, ScopeId

if TYPE_CHECKING:
    from aml.service.resource_observer import ProviderInvocation, ResourceSnapshot

OwnerPhase = Literal[
    "labels", "extraction", "embedding", "judge_round_1", "judge_round_2", "commit",
]
"""The Add owner's six phases (E.4 ``owner_phase_seconds``), set by the owner context."""
ResourcePhase = Literal[
    "add", "search",
    "labels", "extraction", "embedding", "judge_round_1", "judge_round_2", "commit",
]
"""A request phase (``add``, ``search``) or one of the owner's phases."""
CostOperation = Literal[
    "phase", "proof.finalize_sources", "storage.indexed_read", "storage.decode",
    "storage.row_copy", "cache.row_lookup", "cache.scan_lookup", "provider.extract",
    "provider.judge", "provider.embedding", "provider.backoff", "wait.owner",
]
CostCounter = Literal[
    "storage.rows_decoded", "cache.row_hit", "cache.row_absent", "cache.scan_hit",
    "cache.scan_absent", "cache.scan_invalidated",
    "wait.completed", "wait.deadline", "wait.cancelled",
]
CostOutcome = Literal["succeeded", "failed", "cancelled"]
CostCaptureMode = Literal["detailed-v1", "essential-summary-v1"]
OWNER_PHASES: frozenset[OwnerPhase] = frozenset(get_args(OwnerPhase))
RESOURCE_PHASES: frozenset[ResourcePhase] = frozenset(get_args(ResourcePhase))
_OPERATIONS = frozenset(get_args(CostOperation))
_COUNTERS = frozenset(get_args(CostCounter))
_RUN: TypeAdapter[str] = TypeAdapter(ULID)


def _run_id(value: ULID | None) -> bool:
    if value is None:
        return True
    try:
        _RUN.validate_python(value, strict=True)
        return True
    except ValidationError:
        return False


@dataclass(frozen=True, slots=True)
class CostSpan:
    """One actual interval; absent finish is censored rather than zero duration."""

    number: int
    parent: int | None
    operation: CostOperation
    scope_id: ScopeId
    phase: str
    run_id: ULID | None
    owner_run_id: ULID | None
    finalize_cycle: int | None
    thread_id: int
    started_at_ns: int
    logical_request_sha256: str | None = None
    finished_at_ns: int | None = None
    outcome: CostOutcome | None = None
    counters: tuple[tuple[CostCounter, int], ...] = ()


@dataclass(frozen=True, slots=True)
class ProviderCostLink:
    """Join existing actual provider-attempt/usage records to their observed span."""

    call_number: int
    span_number: int | None


@dataclass(frozen=True, slots=True)
class CostSnapshot:
    """Raw bounded attribution; no execution success or billing assertion."""

    schema_version: Literal[1]
    started_at_ns: int
    ended_at_ns: int | None
    invalid_reasons: tuple[str, ...]
    spans: tuple[CostSpan, ...]
    provider_links: tuple[ProviderCostLink, ...]


@dataclass(frozen=True, slots=True)
class CostOperationSummary:
    """Exact observed counts for one closed operation and scoped phase."""

    scope_id: ScopeId
    phase: str
    operation: CostOperation
    entered: int
    succeeded: int
    failed: int
    cancelled: int
    unfinished: int
    counters: tuple[tuple[CostCounter, int], ...]


@dataclass(frozen=True, slots=True)
class EssentialCostSnapshot:
    """Versioned counts and phase intervals, explicitly without fine-grained history."""

    schema_version: Literal[2]
    capture_mode: Literal["essential-summary-v1"]
    fine_grained_history: Literal["not_collected"]
    started_at_ns: int
    ended_at_ns: int | None
    invalid_reasons: tuple[str, ...]
    operations: tuple[CostOperationSummary, ...]
    phase_timings: tuple[CostSpan, ...]
    provider_call_count: int


def operation_delta(
    before: EssentialCostSnapshot, after: EssentialCostSnapshot,
) -> list[dict[str, object]]:
    """Difference cumulative counts without subtracting overlapping phase intervals."""
    if before.started_at_ns != after.started_at_ns or (
        before.invalid_reasons or after.invalid_reasons
        or after.provider_call_count < before.provider_call_count
    ):
        raise ValueError("Essential snapshots do not describe one valid observation lifetime")
    previous = {(row.scope_id, row.phase, row.operation): row for row in before.operations}
    current = {(row.scope_id, row.phase, row.operation): row for row in after.operations}
    if not previous.keys() <= current.keys():
        raise ValueError("Essential operation population regressed")
    result: list[dict[str, object]] = []
    for key, row in current.items():
        old = previous.get(key)
        values = {
            name: getattr(row, name) - (0 if old is None else getattr(old, name))
            for name in ("entered", "succeeded", "failed", "cancelled")
        }
        counts = dict(row.counters)
        for name, count in (() if old is None else old.counters):
            counts[name] = counts.get(name, 0) - count
        if any(value < 0 for value in (*values.values(), *counts.values())):
            raise ValueError("Essential cumulative counts regressed")
        result.append({
            "scope_id": row.scope_id, "phase": row.phase, "operation": row.operation,
            **values, "counters": counts,
            "unfinished_before": 0 if old is None else old.unfinished,
            "unfinished_after": row.unfinished,
        })
    return result


@dataclass(slots=True)
class _OperationCounts:
    entered: int = 0
    succeeded: int = 0
    failed: int = 0
    cancelled: int = 0
    counters: dict[CostCounter, int] = field(default_factory=dict[CostCounter, int])


class CostAttribution:
    """Observation backing only; the existing resource owner controls its lifetime.

    Detailed mode bounds all retained spans. Essential mode bounds simultaneous
    active tokens and retained phase intervals; completed operation counts use
    the fixed registered-scope/phase/operation domain, not a growing event list.
    """

    def __init__(
        self, *, capacity: int, scopes: tuple[ScopeId, ...], started_at_ns: int,
        clock: Callable[[], int] = time.monotonic_ns,
        mode: CostCaptureMode = "detailed-v1",
    ) -> None:
        if type(capacity) is not int or capacity < 1:
            raise ValueError("Attribution requires a positive registered capacity")
        self._capacity, self._scopes, self._started = capacity, frozenset(scopes), started_at_ns
        self._clock = clock
        if mode not in {"detailed-v1", "essential-summary-v1"}:
            raise ValueError("Unknown cost capture mode")
        self.mode: CostCaptureMode = mode
        self._lock = threading.Lock()
        self._spans: list[CostSpan] = []
        self._counts: dict[int, dict[CostCounter, int]] = {}
        self._links: list[ProviderCostLink] = []
        self._invalid: set[str] = set()
        self._ended: int | None = None
        self._summary_counts: dict[tuple[ScopeId, str, CostOperation], _OperationCounts] = {}
        self._summary_active: dict[int, CostSpan] = {}
        self._summary_phases: list[CostSpan] = []
        self._summary_number = self._summary_provider_calls = 0
        self._summary_latest_return = started_at_ns

    def incomplete(self, reason: Literal["producer_failed", "invalid_event"]) -> None:
        """Latch only fixed content-free failure codes, leaving business work unchanged."""
        with self._lock:
            self._invalid.add(reason if reason in {"producer_failed", "invalid_event"}
                              else "invalid_event")

    def begin(
        self, operation: CostOperation, *, scope_id: ScopeId, phase: str,
        run_id: ULID | None, owner_run_id: ULID | None, finalize_cycle: int | None,
        logical_request_sha256: str | None = None,
    ) -> CostSpan | None:
        """Copy only validated scalar context, never a request, key, row or exception."""
        now, thread = self._clock(), threading.get_ident()
        with self._lock:
            used = len(self._spans) if self.mode == "detailed-v1" else max(
                len(self._summary_active),
                len(self._summary_phases) if operation == "phase" else 0,
            )
            if self._ended is not None or used >= self._capacity:
                self._invalid.add("span_overflow_or_after_finish")
                return None
            if operation not in _OPERATIONS or scope_id not in self._scopes or (
                type(now) is not int or now < self._started
                or phase not in RESOURCE_PHASES or not _run_id(run_id) or not _run_id(owner_run_id)
                or (owner_run_id is None) != (finalize_cycle is None)
                or finalize_cycle is not None and (
                    type(finalize_cycle) is not int or finalize_cycle < 1
                )
            ) or logical_request_sha256 is not None and (
                len(logical_request_sha256) != 64
                or any(char not in "0123456789abcdef" for char in logical_request_sha256)
            ):
                self._invalid.add("invalid_event")
                return None
            if self.mode == "essential-summary-v1":
                self._summary_number += 1
                span = CostSpan(
                    self._summary_number, None, operation, scope_id, phase, run_id,
                    owner_run_id, finalize_cycle, thread, now, logical_request_sha256,
                )
                self._summary_active[span.number] = span
                self._counts[span.number] = {}
                key = scope_id, phase, operation
                counts = self._summary_counts.get(key)
                if counts is None:
                    counts = self._summary_counts[key] = _OperationCounts()
                counts.entered += 1
                return span
            active = _ACTIVE.get()
            parent = None
            if active is not None and active[0] is self and active[2] == thread:
                previous = self._spans[active[1] - 1]
                if previous.finished_at_ns is None and previous.scope_id == scope_id:
                    parent = previous.number
            span = CostSpan(
                len(self._spans) + 1, parent, operation, scope_id, phase, run_id,
                owner_run_id, finalize_cycle, thread, now, logical_request_sha256,
            )
            self._spans.append(span)
            self._counts[span.number] = {}
            return span

    def end(self, span: CostSpan, outcome: CostOutcome) -> None:
        """Close only the exact original token after the measured operation exits."""
        now = self._clock()
        with self._lock:
            if self.mode == "essential-summary-v1":
                self._end_summary(span, outcome, now)
                return
            if not 1 <= span.number <= len(self._spans) or (
                self._spans[span.number - 1] is not span or now < span.started_at_ns
                or self._ended is not None
                or outcome not in {"succeeded", "failed", "cancelled"}
            ):
                self._invalid.add("span_return_invalid")
                return
            self._spans[span.number - 1] = replace(
                span, finished_at_ns=now, outcome=outcome,
                counters=tuple(sorted(self._counts.pop(span.number).items())),
            )

    def _end_summary(self, span: CostSpan, outcome: CostOutcome, now: int) -> None:
        if self._summary_active.get(span.number) is not span or type(now) is not int or (
            now < span.started_at_ns or self._ended is not None
            or outcome not in {"succeeded", "failed", "cancelled"}
        ):
            self._invalid.add("span_return_invalid")
            return
        del self._summary_active[span.number]
        self._summary_latest_return = max(self._summary_latest_return, now)
        counters = self._counts.pop(span.number)
        counts = self._summary_counts[span.scope_id, span.phase, span.operation]
        if outcome == "succeeded":
            counts.succeeded += 1
        elif outcome == "failed":
            counts.failed += 1
        else:
            counts.cancelled += 1
        for name, value in counters.items():
            counts.counters[name] = counts.counters.get(name, 0) + value
        if span.operation == "phase":
            if len(self._summary_phases) >= self._capacity:
                self._invalid.add("phase_overflow_or_after_finish")
            else:
                self._summary_phases.append(replace(
                    span, finished_at_ns=now, outcome=outcome,
                    counters=tuple(sorted(counters.items())),
                ))

    def count(self, number: int, counter: CostCounter, amount: int) -> None:
        """Count bounded code-owned events inside the actual current interval."""
        with self._lock:
            if counter not in _COUNTERS or type(amount) is not int or amount < 0 or (
                number not in self._counts or self._ended is not None
            ):
                self._invalid.add("counter_invalid")
                return
            counts = self._counts[number]
            counts[counter] = counts.get(counter, 0) + amount

    def provider_link(self, call_number: int) -> None:
        """Link the resource owner's actual new attempt without inventing another call."""
        with self._lock:
            if self.mode == "essential-summary-v1":
                active = _ACTIVE.get()
                if self._ended is not None or call_number != self._summary_provider_calls + 1 or (
                    active is None or active[0] is not self
                    or active[2] != threading.get_ident() or active[1] not in self._summary_active
                ):
                    self._invalid.add("provider_span_missing")
                    return
                self._summary_provider_calls += 1
                return
            if self._ended is not None or len(self._links) >= self._capacity:
                self._invalid.add("provider_link_overflow_or_after_finish")
                return
            active = _ACTIVE.get()
            number = active[1] if (
                active is not None and active[0] is self and active[2] == threading.get_ident()
                and self._spans[active[1] - 1].finished_at_ns is None
            ) else None
            if number is None:
                self._invalid.add("provider_span_missing")
            self._links.append(ProviderCostLink(call_number, number))

    def finish(self, at_ns: int) -> CostSnapshot:
        """Called only by the enclosing resource collector after actual owner joins."""
        if self.mode != "detailed-v1":
            raise ValueError("Essential capture requires its separate summary contract")
        with self._lock:
            if self._ended is not None or type(at_ns) is not int or at_ns < self._started or any(
                span.finished_at_ns is not None and span.finished_at_ns > at_ns
                for span in self._spans
            ):
                self._invalid.add("collection_end_invalid")
            else:
                self._ended = at_ns
            if self._counts:
                self._invalid.add("span_return_missing")
            return self._snapshot()

    def observed(self) -> CostSnapshot:
        """Copy evidence without claiming complete capture or draining any work."""
        if self.mode != "detailed-v1":
            raise ValueError("Fine-grained cost history was not collected")
        with self._lock:
            return self._snapshot()

    def finish_summary(self, at_ns: int) -> EssentialCostSnapshot:
        """Seal the registered summary after the enclosing owner has joined its callbacks."""
        if self.mode != "essential-summary-v1":
            raise ValueError("Detailed capture requires its original snapshot contract")
        with self._lock:
            if self._ended is not None or type(at_ns) is not int or (
                at_ns < self._summary_latest_return
            ):
                self._invalid.add("collection_end_invalid")
            else:
                self._ended = at_ns
            if self._summary_active:
                self._invalid.add("span_return_missing")
            return self._summary_snapshot()

    def summary_observed(self) -> EssentialCostSnapshot:
        """Read registered essential evidence without inventing fine-grained traces."""
        if self.mode != "essential-summary-v1":
            raise ValueError("Detailed capture requires its original snapshot contract")
        with self._lock:
            return self._summary_snapshot()

    def _summary_snapshot(self) -> EssentialCostSnapshot:
        counters = {key: dict(counts.counters) for key, counts in self._summary_counts.items()}
        for number, span in self._summary_active.items():
            aggregate = counters[span.scope_id, span.phase, span.operation]
            for name, value in self._counts[number].items():
                aggregate[name] = aggregate.get(name, 0) + value
        return EssentialCostSnapshot(
            2, "essential-summary-v1", "not_collected", self._started, self._ended,
            tuple(sorted(self._invalid)),
            tuple(CostOperationSummary(
                scope, phase, operation, counts.entered, counts.succeeded, counts.failed,
                counts.cancelled,
                counts.entered - counts.succeeded - counts.failed - counts.cancelled,
                tuple(sorted(counters[scope, phase, operation].items())),
            ) for (scope, phase, operation), counts in sorted(self._summary_counts.items())),
            tuple(sorted((*self._summary_phases, *(
                replace(span, counters=tuple(sorted(self._counts[span.number].items())))
                for span in self._summary_active.values() if span.operation == "phase"
            )), key=lambda span: span.number)),
            self._summary_provider_calls,
        )

    def _snapshot(self) -> CostSnapshot:
        return CostSnapshot(
            1, self._started, self._ended, tuple(sorted(self._invalid)),
            tuple(
                replace(span, counters=tuple(sorted(self._counts[span.number].items())))
                if span.number in self._counts else span for span in self._spans
            ),
            tuple(self._links),
        )


_ACTIVE: ContextVar[tuple[CostAttribution, int, int] | None] = ContextVar(
    "aml_cost_span", default=None,
)


def essential_costs_complete(
    snapshot: EssentialCostSnapshot, *, scopes: frozenset[ScopeId], phase_capacity: int,
    provider_call_count: int,
) -> bool:
    """Validate registered summary completeness, never full cost-history completeness.

    Counts cover observed operation entries and outcomes. Only phase intervals
    retain timing/context; no per-read ancestry or latency can be reconstructed.
    Artifact custody and authenticity remain the enclosing evidence owner's job.
    """
    end = snapshot.ended_at_ns
    if end is None or end < snapshot.started_at_ns or snapshot.invalid_reasons or (
        snapshot.provider_call_count != provider_call_count
        or snapshot.provider_call_count < 0 or len(snapshot.phase_timings) > phase_capacity
    ):
        return False
    groups: dict[tuple[ScopeId, str, CostOperation], CostOperationSummary] = {}

    def valid_counters(values: tuple[tuple[CostCounter, int], ...]) -> bool:
        return len({name for name, _ in values}) == len(values) and all(
            name in _COUNTERS and type(amount) is int and amount >= 0 for name, amount in values
        )

    for row in snapshot.operations:
        key = row.scope_id, row.phase, row.operation
        if key in groups or row.scope_id not in scopes or row.phase not in RESOURCE_PHASES or (
            row.operation not in _OPERATIONS or row.entered < 1 or row.unfinished != 0
            or min(row.succeeded, row.failed, row.cancelled) < 0
            or row.entered != row.succeeded + row.failed + row.cancelled
            or not valid_counters(row.counters)
        ):
            return False
        groups[key] = row
    phases: dict[tuple[ScopeId, str, CostOperation], _OperationCounts] = {}
    previous_number = 0
    for span in snapshot.phase_timings:
        if (
            span.number <= previous_number or span.operation != "phase" or span.parent is not None
            or span.scope_id not in scopes or span.phase not in RESOURCE_PHASES
            or span.thread_id < 1
            or span.started_at_ns < snapshot.started_at_ns or span.finished_at_ns is None
            or not span.started_at_ns <= span.finished_at_ns <= end
            or span.outcome not in {"succeeded", "failed", "cancelled"}
            or not _run_id(span.run_id) or not _run_id(span.owner_run_id)
            or (span.owner_run_id is None) != (span.finalize_cycle is None)
            or span.finalize_cycle is not None and span.finalize_cycle < 1
            or not valid_counters(span.counters)
            or span.logical_request_sha256 is not None and (
                len(span.logical_request_sha256) != 64 or any(
                    char not in "0123456789abcdef" for char in span.logical_request_sha256
                )
            )
        ):
            return False
        previous_number = span.number
        key = span.scope_id, span.phase, span.operation
        count = phases.setdefault(key, _OperationCounts())
        count.entered += 1
        if span.outcome == "succeeded":
            count.succeeded += 1
        elif span.outcome == "failed":
            count.failed += 1
        else:
            count.cancelled += 1
        for name, value in span.counters:
            count.counters[name] = count.counters.get(name, 0) + value
    expected = {key: row for key, row in groups.items() if row.operation == "phase"}
    return phases.keys() == expected.keys() and all(
        (count.entered, count.succeeded, count.failed, count.cancelled, count.counters)
        == (expected[key].entered, expected[key].succeeded, expected[key].failed,
            expected[key].cancelled, dict(expected[key].counters))
        for key, count in phases.items()
    )


def current_cost_attribution() -> CostAttribution | None:
    """Resolve only an already installed resource observer; never create one implicitly."""
    from aml.service.request_measurement import current_resource_context

    try:
        context = current_resource_context()
        return None if context is None or context.resources is None else context.resources.costs
    except Exception:
        return None


def _producer_failed(observer: CostAttribution) -> None:
    try:
        observer.incomplete("producer_failed")
    except Exception:
        pass


@contextmanager
def cost_span(
    operation: CostOperation, *, logical_request_sha256: str | None = None,
) -> Iterator[None]:
    """Measure one existing operation; disabled capture reads no clock or request bytes."""
    from aml.service.request_measurement import current_resource_context

    try:
        context = current_resource_context()
        observer = None if context is None or context.resources is None else context.resources.costs
    except Exception:
        context = observer = None
    if observer is None or context is None:
        absent = _ACTIVE.set(None)
        try:
            yield
        finally:
            _ACTIVE.reset(absent)
        return
    span = None
    try:
        span = observer.begin(
            operation, scope_id=context.scope_id, phase=context.phase, run_id=context.run_id,
            owner_run_id=None if context.cycle is None else context.cycle.owner_run_id,
            finalize_cycle=None if context.cycle is None else context.cycle.finalize_cycle,
            logical_request_sha256=logical_request_sha256,
        )
    except Exception:
        _producer_failed(observer)
    if span is None:
        absent = _ACTIVE.set(None)
        try:
            yield
        finally:
            _ACTIVE.reset(absent)
        return
    token = _ACTIVE.set((observer, span.number, span.thread_id))
    outcome: CostOutcome = "succeeded"
    try:
        yield
    except BaseException as error:
        outcome = (
            "cancelled" if isinstance(error, CancelledError | FutureCancelledError) else "failed"
        )
        raise
    finally:
        _ACTIVE.reset(token)
        try:
            observer.end(span, outcome)
        except Exception:
            _producer_failed(observer)


def cost_count(counter: CostCounter, amount: int = 1) -> None:
    """Count only inside an observed span, never retaining a cache key or row."""
    active = _ACTIVE.get()
    if active is not None and active[2] == threading.get_ident():
        try:
            active[0].count(active[1], counter, amount)
        except Exception:
            _producer_failed(active[0])


def _capture_off() -> bool:
    """Whether a span here would be disabled with no enclosing span to suspend.

    ``cost_span`` then only sets the active span to None, which it already
    is, so the boundary can run directly. A failed context read is left to
    ``cost_span``, which owns that fallback.
    """
    from aml.service.request_measurement import current_resource_context

    if _ACTIVE.get() is not None:
        return False
    try:
        return current_resource_context() is None
    except Exception:
        return False


def attribute_cost[**P, T](
    operation: CostOperation,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Wrap a synchronous existing boundary without examining arguments or results.

    Row copies and indexed reads run millions of times in one large finalize,
    nearly always without capture; such a call skips the span's context manager.
    """
    def decorate(function: Callable[P, T]) -> Callable[P, T]:
        @wraps(function)
        def measured(*args: P.args, **kwargs: P.kwargs) -> T:
            if _capture_off():
                return function(*args, **kwargs)
            with cost_span(operation):
                return function(*args, **kwargs)
        return measured
    return decorate


def summarize_costs(snapshot: CostSnapshot) -> dict[str, object]:
    """Report distributions without adding nested or concurrent spans into wall time."""
    children: dict[int, list[CostSpan]] = {}
    groups: dict[tuple[str, str], list[CostSpan]] = {}
    for span in snapshot.spans:
        groups.setdefault((span.phase, span.operation), []).append(span)
        if span.parent is not None:
            children.setdefault(span.parent, []).append(span)

    def union(intervals: list[tuple[int, int]]) -> int:
        total, end = 0, -1
        for start, finish in sorted(intervals):
            total += max(0, finish - max(start, end))
            end = max(end, finish)
        return total

    def distribution(values: list[int]) -> dict[str, int | None]:
        ordered = sorted(values)
        count = len(ordered)
        return {
            "count": count, "sum_ns": sum(ordered) if ordered else None,
            "p50_ns": ordered[(count - 1) // 2] if ordered else None,
            "p95_ns": ordered[(95 * count + 99) // 100 - 1] if ordered else None,
            "max_ns": ordered[-1] if ordered else None,
        }

    result: list[dict[str, object]] = []
    for (phase, operation), spans in sorted(groups.items()):
        inclusive: list[int] = []
        exclusive: list[int] = []
        counters: dict[str, int] = {}
        for span in spans:
            for name, value in span.counters:
                counters[name] = counters.get(name, 0) + value
            if span.finished_at_ns is None:
                continue
            duration = span.finished_at_ns - span.started_at_ns
            inclusive.append(duration)
            nested = children.get(span.number, [])
            if all(child.finished_at_ns is not None for child in nested):
                intervals = [
                    (max(span.started_at_ns, child.started_at_ns),
                     min(span.finished_at_ns, child.finished_at_ns))
                    for child in nested if child.finished_at_ns is not None
                ]
                exclusive.append(duration - union(intervals))
        result.append({
            "phase": phase, "operation": operation, "entered": len(spans),
            "failed": sum(span.outcome == "failed" for span in spans),
            "cancelled": sum(span.outcome == "cancelled" for span in spans),
            "unfinished": sum(span.finished_at_ns is None for span in spans),
            "inclusive": distribution(inclusive), "exclusive": distribution(exclusive),
            "counters": counters,
        })
    return {
        "schema_version": 1,
        "capture_complete": snapshot.ended_at_ns is not None and not snapshot.invalid_reasons,
        "invalid_reasons": list(snapshot.invalid_reasons), "groups": result,
        "note": "Inclusive and exclusive sums are work attribution, not cohort wall time.",
    }


def summarize_judge_calls(
    snapshot: ResourceSnapshot, costs: CostSnapshot | None = None,
) -> dict[str, object]:
    """Count actual judge-batch-v8 attempts per owner phase (C.8; E.4 judge diagnostics).

    The adapter names the real output schema in ``provider_snapshot``; every
    judge request is a judge-batch-v8 batch, reported under the phase it ran in
    (``judge_round_1`` or ``judge_round_2``). Each physical attempt counts once.
    Logical calls use the existing provider cost links; absent or incomplete
    links leave them unknown rather than counting one logical call per retry.
    Members per request come from the owner's call manifests, not from here.
    These counts grant no semantic or billing authority, and a reused verdict
    performs no new observed physical attempt.
    """
    from aml.core.judge_output import JUDGE_BATCH_V8_SCHEMA_ID

    phases: dict[str, list[ProviderInvocation]] = {}
    for call in snapshot.provider_calls:
        endpoint, separator, schema = call.provider_snapshot.rpartition(":")
        if separator and endpoint.endswith(":chat/completions") and (
            schema == JUDGE_BATCH_V8_SCHEMA_ID
        ):
            phases.setdefault(call.phase, []).append(call)
    complete = snapshot.provider_history_complete
    links = {} if costs is None else {
        link.call_number: link.span_number for link in costs.provider_links
    }
    spans = {} if costs is None else {span.number: span for span in costs.spans}
    result: dict[str, object] = {}
    for phase, calls in sorted(phases.items()):
        logical: set[int] = set()
        logical_complete = complete and costs is not None and (
            costs.ended_at_ns is not None and not costs.invalid_reasons
        )
        for call in calls:
            number = links.get(call.number)
            span = None if number is None else spans.get(number)
            if span is None or span.operation != "provider.judge" or (
                span.logical_request_sha256 is None or span.finished_at_ns is None
            ):
                logical_complete = False
            elif number is not None:
                logical.add(number)
        input_tokens = [
            call.usage.input_tokens if call.usage is not None else None for call in calls
        ]
        output_tokens = [
            call.usage.output_tokens if call.usage is not None else None for call in calls
        ]
        result[phase] = {
            "observed_physical_attempts": len(calls),
            "physical_attempts": len(calls) if complete else None,
            "logical_calls": len(logical) if logical_complete else None,
            "succeeded": sum(call.outcome == "succeeded" for call in calls),
            "failed": sum(call.outcome == "failed" for call in calls),
            "cancelled": sum(call.outcome == "cancelled" for call in calls),
            "unfinished": sum(call.outcome is None for call in calls),
            "input_tokens": sum(value for value in input_tokens if value is not None)
            if complete and all(value is not None for value in input_tokens) else None,
            "output_tokens": sum(value for value in output_tokens if value is not None)
            if complete and all(value is not None for value in output_tokens) else None,
        }
    return {
        "schema_version": 2, "call_trace_complete": complete, "phases": result,
        "note": "Actual attempt counts only; no measured live cost or quality is implied.",
    }
