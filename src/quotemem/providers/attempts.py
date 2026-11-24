"""Keep an executor's authorization check on each actual HTTP attempt."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar

_CHECK: ContextVar[Callable[[], None] | None] = ContextVar("aml_provider_attempt", default=None)


@contextmanager
def provider_attempts(check: Callable[[], None]) -> Iterator[None]:
    """Bind an action check to this execution context until provider return.

    Args:
        check: Current cancellation and action authorization, without a model call.

    Yields:
        None: The transport checks each initial attempt and retry before sending.

    Raises:
        Exception: An authorization or provider error propagates unchanged.
    """
    previous = _CHECK.get()

    def combined() -> None:
        if previous is not None:
            previous()
        check()

    token = _CHECK.set(combined)
    try:
        yield
    finally:
        _CHECK.reset(token)


def current_provider_attempt() -> Callable[[], None] | None:
    """Return the attempt check bound to this execution context, if any."""
    return _CHECK.get()


@contextmanager
def provider_attempts_from(check: Callable[[], None] | None) -> Iterator[None]:
    """Run with exactly an attempt check captured earlier in this context.

    Checks bound after that capture do not apply inside. A batch uses this so a
    suspended member's own attempt check never rides on a request that does not
    carry that member, while every enclosing owner check still applies.

    Args:
        check: The captured check, or None when none was bound.

    Yields:
        None: Transport attempts inside use exactly ``check``.
    """
    token = _CHECK.set(check)
    try:
        yield
    finally:
        _CHECK.reset(token)


def check_provider_attempt() -> None:
    """Check the current executor immediately before an actual transport send."""
    if (check := _CHECK.get()) is not None:
        check()
