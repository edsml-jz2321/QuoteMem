"""The memory-judge timeout policy; result schemas and prompts are independent."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Final

JUDGE_TIMEOUT_POLICY_ID: Final = "judge-read-900-500-v1"
JUDGE_FIRST_TIMEOUT_MS: Final = 900_000
JUDGE_SECOND_TIMEOUT_MS: Final = 500_000
JUDGE_NON_READ_TIMEOUT_MS: Final = 300_000
JUDGE_COMMIT_RESERVE_MS: Final = 60_000
JUDGE_MIN_ATTEMPT_MS: Final = 10_000


_OWNER_WAIT_MS: ContextVar[Callable[[], int] | None] = ContextVar(
    "judge_owner_wait_ms", default=None,
)


@contextmanager
def judge_owner_wait(remaining_ms: Callable[[], int]) -> Iterator[None]:
    """Expose the owner's remaining allowance across numbered judge read attempts.

    Args:
        remaining_ms: Read the owner's current time to its fixed deadline minus
            the commit reserve. The provider samples it after each send reservation;
            setup and backoff consume owner time without reducing the read policy limit.

    Yields:
        None: The callback applies only within this request-owned context.
    """
    token = _OWNER_WAIT_MS.set(remaining_ms)
    try:
        yield
    finally:
        _OWNER_WAIT_MS.reset(token)


def judge_owner_wait_ms() -> int | None:
    """Read the current owner allowance after its commit reserve.

    Returns:
        The callback's current allowance in milliseconds, or None when this
        provider caller has not supplied an owner. A nonpositive value leaves
        no time for another judge request.
    """
    remaining = _OWNER_WAIT_MS.get()
    return None if remaining is None else remaining()
