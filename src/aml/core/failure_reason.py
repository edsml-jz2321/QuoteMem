"""Content-free attribution of a background failure for later diagnosis.

A failure reason keeps its short origin summary and a bounded causal trace of
file names, line numbers and functions. It never retains exception messages,
arguments, locals, source lines, exception objects or traceback objects.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Literal, NamedTuple

from aml.core.errors import AMLError

_MAX_TRACE_EXCEPTIONS = 8
_MAX_TRACE_FRAMES = 64


class FailureTrace(NamedTuple):
    """One exception's safe coordinates, linked to its preceding trace entry.

    Frames retain at most the innermost 64 coordinates in caller-to-callee
    order. The origin coordinate may survive an earlier sanitizing wrapper
    even when its original traceback does not. Neither reconstructs lost data.
    """

    exception_type: str
    code: str | None
    relation: Literal["root", "cause", "context"]
    frames: tuple[tuple[str, int, str], ...]
    origin_frame: tuple[str, int, str] | None
    context_suppressed: bool
    frames_truncated: bool


class FailureReason(NamedTuple):
    """Where and what failed, without content; old four-argument fixtures remain valid."""

    site: str
    exception_type: str | None
    code: str | None
    frame: tuple[str, int, str] | None
    trace: tuple[FailureTrace, ...] = ()
    trace_truncated: bool = False
    trace_cycle: bool = False

    def to_dict(self) -> dict[str, object]:
        """Return the safe summary and trace as JSON-compatible named fields.

        Returns:
            The existing site/type/code/frame fields plus trace entries and
            explicit chain/frame truncation and cycle indicators. An empty
            trace on a legacy fixture means no trace was recorded.
        """
        return {
            "site": self.site, "exception_type": self.exception_type, "code": self.code,
            "frame": None if self.frame is None else list(self.frame),
            "trace": [{
                "exception_type": entry.exception_type, "code": entry.code,
                "relation": entry.relation, "frames": [list(frame) for frame in entry.frames],
                "origin_frame": None if entry.origin_frame is None else list(entry.origin_frame),
                "context_suppressed": entry.context_suppressed,
                "frames_truncated": entry.frames_truncated,
            } for entry in self.trace],
            "trace_truncated": self.trace_truncated, "trace_cycle": self.trace_cycle,
        }


def _exception_trace(
    error: BaseException, relation: Literal["root", "cause", "context"],
) -> FailureTrace:
    frames: deque[tuple[str, int, str]] = deque(maxlen=_MAX_TRACE_FRAMES)
    count = 0
    traceback = error.__traceback__
    while traceback is not None:
        code = traceback.tb_frame.f_code
        frames.append((Path(code.co_filename).name, traceback.tb_lineno, code.co_name))
        count += 1
        traceback = traceback.tb_next
    return FailureTrace(
        type(error).__name__, error.code if isinstance(error, AMLError) else None, relation,
        tuple(frames), error.origin_frame if isinstance(error, AMLError) else None,
        error.__suppress_context__, count > len(frames),
    )


def failure_reason(site: str, error: BaseException | None) -> FailureReason:
    """Capture the available causal trace without retaining exception content.

    Args:
        site: Code-owned name of the boundary that latched the failure.
        error: The actual exception, or None for a failure callback without one.

    Returns:
        The unchanged short summary and at most eight distinct exceptions,
        starting with the supplied error. Explicit cause takes precedence over
        context. Suppressed context is still captured as safe coordinates and
        labeled as suppressed; no hidden message or argument is rendered.
        Cycles and retention limits are explicit. All traceback frames are
        traversed to preserve the innermost summary, but only the innermost 64
        per exception are retained, without consulting source lines or locals.
        Already discarded causes/frames cannot be reconstructed.
    """
    if error is None:
        return FailureReason(site, None, None, None)
    trace: list[FailureTrace] = []
    seen: set[int] = set()
    current: BaseException | None = error
    relation: Literal["root", "cause", "context"] = "root"
    truncated = cycle = False
    while current is not None:
        if id(current) in seen:
            truncated = cycle = True
            break
        if len(trace) == _MAX_TRACE_EXCEPTIONS:
            truncated = True
            break
        seen.add(id(current))
        trace.append(_exception_trace(current, relation))
        if current.__cause__ is not None:
            current, relation = current.__cause__, "cause"
        else:
            current, relation = current.__context__, "context"
    first = trace[0]
    frame = first.origin_frame
    if frame is None and first.frames:
        frame = first.frames[-1]
    return FailureReason(
        site, first.exception_type, first.code, frame, tuple(trace), truncated, cycle,
    )


def describe(reason: FailureReason | None) -> str:
    """Render a latched reason for an error message, still without content.

    A pool that latches a failure and then raises a generic code leaves every
    later caller holding a symptom: twenty-eight `service_unhealthy` errors on
    2026-09-13 all came from one cause the pool had already recorded and did
    not pass on. Every field here is content-free by construction, so attaching
    it to the message adds attribution and nothing else.
    """
    if reason is None:
        return ""
    where = "" if reason.frame is None else (
        f" at {reason.frame[0]}:{reason.frame[1]}:{reason.frame[2]}"
    )
    named = reason.code or reason.exception_type
    return f" ({reason.site}: {named}{where})" if named else f" ({reason.site}{where})"
