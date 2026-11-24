"""Cancel and join request-owned HTTP work within its existing executor thread."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from contextlib import ExitStack
from typing import Any

import httpx

from quotemem.core.errors import QuoteMemError
from quotemem.core.provider_diagnostics import ProviderFailureDetailV1
from quotemem.core.request_budget import RequestBudget

type AsyncClientFactory = Callable[[], httpx.AsyncClient]


class ProviderDeadlineExpired(QuoteMemError):
    """The owned HTTP wall timer expired; parent cancellation is a different event."""

    def __init__(self) -> None:
        super().__init__("delivery_busy", "The owned provider deadline expired")
        self.diagnostics: ProviderFailureDetailV1 | None = None


def post_with_budget(
    factory: AsyncClientFactory,
    budget: RequestBudget,
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout_seconds: float | httpx.Timeout,
    *, parent: RequestBudget | None = None,
    extensions: dict[str, Any] | None = None,
) -> httpx.Response:
    """Use a local event loop and close the actual HTTP request before returning.

    The caller's existing worker owns this loop; no detached provider thread is
    introduced. Phase inactivity settings retain their original binding. The
    request budget also bounds the entire send, headers and body, even when a
    peer keeps an inactivity timer alive by sending occasional bytes. Client
    cleanup is joined separately and is never reported complete from a timeout.
    """
    budget.check()
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        raise QuoteMemError("unsupported_variant", "Bounded sync provider needs its owned worker")

    async def send() -> httpx.Response:
        async with factory() as client:
            budget.check()
            loop = asyncio.get_running_loop()
            deadline = asyncio.timeout(budget.remaining())
            observed_extensions = extensions
            if extensions is not None and "aml_judge_attempt" in extensions:
                observed_extensions = {**extensions,
                                       "aml_judge_deadline_expired": deadline.expired}
            pending = asyncio.create_task(client.post(
                url, json=payload, headers=headers, timeout=timeout_seconds,
                extensions=observed_extensions,
            ))

            def cancel() -> None:
                loop.call_soon_threadsafe(pending.cancel)

            try:
                with ExitStack() as notifications:
                    notifications.enter_context(budget.on_cancel(cancel))
                    if parent is not None:
                        notifications.enter_context(parent.on_cancel(budget.cancel))
                    async with deadline:
                        response = await pending
            except TimeoutError:
                if deadline.expired():
                    # Event-loop deadlines can fire before a lower-resolution
                    # monotonic clock advances. Retain the actual timer outcome
                    # instead of leaking TimeoutError or starting another send.
                    if parent is not None and parent.deadline <= budget.deadline:
                        parent.cancel()
                    budget.cancel()
                    if extensions is not None and "aml_judge_attempt" in extensions:
                        raise ProviderDeadlineExpired from None
                    budget.check()
                raise
            except asyncio.CancelledError:
                budget.check()
                raise
            finally:
                if not pending.done():
                    pending.cancel()
                # Awaiting cancellation acknowledges the actual transport; the
                # client context then closes before this worker returns.
                await asyncio.gather(pending, return_exceptions=True)
            budget.check()
            return response

    with asyncio.Runner() as runner:
        return runner.run(send())


def wait_backoff(budget: RequestBudget, seconds: float) -> None:
    """Backoff consumes the same request allowance and wakes on cancellation."""
    budget.check()
    wake = threading.Event()
    with budget.on_cancel(wake.set):
        wake.wait(min(seconds, budget.remaining()))
    budget.check()
