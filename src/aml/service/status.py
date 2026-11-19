"""Pure job-traffic reminders from plan.md §5.9 and §9; no completion authority."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import TypeAdapter

from aml.core.semantic import SystemTimestamp

_TIMESTAMP: TypeAdapter[int] = TypeAdapter(SystemTimestamp)
_DAY_MS = 86_400_000


@dataclass(frozen=True)
class JobTrafficStatus:
    """An operational reminder, never a completion or compliance determination.

    Attributes:
        state: Never observed, current, day-20 reminder, day-28 warning, or a
            clock rollback relative to the recorded traffic watermark.
        elapsed_ms: Exact observed age; absent when no age can be established.
    """

    state: Literal["never_observed", "current", "reminder", "warning", "clock_rollback"]
    elapsed_ms: int | None

    @property
    def exit_code(self) -> Literal[0, 1]:
        """Warn on day 28 or an indeterminate age caused by clock rollback."""
        return 1 if self.state in {"warning", "clock_rollback"} else 0


def job_traffic_status(last_platform_traffic_at: int, *, now: int) -> JobTrafficStatus:
    """Project the supplied durable watermark using a trusted epoch-ms clock.

    Args:
        last_platform_traffic_at: The bound job's durable traffic watermark;
            zero means no successful platform traffic has been observed.
        now: Current trusted system time in epoch milliseconds.

    Returns:
        A reminder at 20 days and a nonzero warning at 28 days. A future
        watermark reports clock rollback with no fabricated zero/negative age.
        There is no automatic deletion, completion time or compliance verdict.

    Raises:
        ValidationError: If either input violates the shared strict timestamp.
    """
    observed = _TIMESTAMP.validate_python(last_platform_traffic_at, strict=True)
    current = _TIMESTAMP.validate_python(now, strict=True)
    if observed == 0:
        return JobTrafficStatus("never_observed", None)
    if current < observed:
        return JobTrafficStatus("clock_rollback", None)
    elapsed = current - observed
    if elapsed >= 28 * _DAY_MS:
        return JobTrafficStatus("warning", elapsed)
    if elapsed >= 20 * _DAY_MS:
        return JobTrafficStatus("reminder", elapsed)
    return JobTrafficStatus("current", elapsed)
