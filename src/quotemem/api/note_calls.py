"""The process's shared worker threads for Search's model-note calls (premise, reader).

Each bench Search used to start a fresh thread pool, and so fresh threads, for its
premise votes and for its reader note: about 7% of a warm Search's local time on a
2,000-turn store. Every note call now runs on one shared pool whose
idle threads are reused. Its bound, `MAX_NOTE_CALLS`, is far above the concurrent
note calls a service makes (four per Search in flight), so a call starts as soon as
it is submitted, as with a pool of its own; a call still in flight after its
Search stopped waiting finishes on its own read timeout, and a call that has not
started is cancelled.
"""

from __future__ import annotations

import contextvars
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Final

MAX_NOTE_CALLS: Final = 256
"""Threads the shared pool starts at most; idle ones are reused."""

_pool = ThreadPoolExecutor(max_workers=MAX_NOTE_CALLS, thread_name_prefix="quotemem-search-note")


def submit_note_call[T](call: Callable[[], T]) -> Future[T]:
    """Run ``call`` on the shared pool in a copy of the caller's context.

    Args:
        call: One model-note call; it handles its own provider failures.

    Returns:
        The call's future; cancel it when its Search stops waiting.
    """
    return _pool.submit(contextvars.copy_context().run, call)
