"""Search's reader note call (``search-reader-v1``).

`ReaderNote` sends a Search query and its top delivered results to the pipeline's model
without reasoning, once, under one read timeout and no retry, and turns a valid reply into
the note body (``core/reader_note.reader_note``). `ReaderNote.begin` starts the call on its
own thread so that it overlaps Search's other checks (conflict, premise); its wait returns
None on any provider failure, timeout or refused reply: Search never fails because of the
note. `reader_noted` closes the last delivered hit with the note, after every other note of
the response. Nothing stored, ranked, scored or selected changes.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import replace
from html import escape
from time import monotonic
from typing import Literal, cast

from quotemem.api.bench import SearchResult
from quotemem.api.note_calls import submit_note_call
from quotemem.core.errors import QuoteMemError
from quotemem.core.identity import ContentBindingDigest
from quotemem.core.inference import InferenceBindingV1, RetryPolicyV1
from quotemem.core.reader_note import (
    READER_HITS,
    READER_PROMPT,
    READER_PROMPT_VERSION,
    READER_SCHEMA_ID,
    ReaderNoteV1,
    reader_message,
    reader_note,
    reader_question,
)
from quotemem.providers.contracts import InferenceRequest, Message, StructuredInferenceProvider

READER_TIMEOUT_MS = 15_000
"""Read timeout of the call; Search waits at most this long plus `_GRACE_S` for the note."""
READER_MAX_TOKENS = 1_500
_GRACE_S = 0.5
_CLOSE = "\n</memory>"
_ONE_SEND = RetryPolicyV1(
    max_attempts=1, initial_backoff_ms=0, max_backoff_ms=0, multiplier=1.0,
    jitter_mode="none", retryable_error_classes=[],
)


class ReaderNote:
    """The reader note of one Search runtime: the provider and the pipeline's judge model.

    Args:
        provider: The structured inference provider of the runtime.
        binding: The pipeline's memory judge binding (provider, model, effort).
        bind: The tenant keyring's content binding for request digests.
    """

    def __init__(
        self, provider: StructuredInferenceProvider, binding: InferenceBindingV1,
        bind: Callable[[str, object], ContentBindingDigest],
    ) -> None:
        if binding.purpose != "judge" or binding.provider != "openai":
            raise QuoteMemError("binding_mismatch",
                                "The reader note runs on the chat judge binding")
        self._provider = provider
        self._binding = binding
        self._bind = bind

    @staticmethod
    def message(query: str, contents: Sequence[str]) -> str | None:
        """The call's user message for one Search, or None when the Search gets no note.

        Args:
            query: The original Search text.
            contents: Every delivered hit's rendered text, in delivery order.

        Returns:
            The message for a gated query with at least one hit; else None.
        """
        if not contents or not reader_question(query):
            return None
        return reader_message(query, contents)

    def begin(self, message: str, hits: int) -> Callable[[], str | None]:
        """Start the call for `message` and return its wait.

        Args:
            message: A `message` result.
            hits: How many delivered hits the Search has.

        Returns:
            A function that waits for the call until its deadline and returns the note
            body, or None on any failure, refusal, empty reply or late answer.
        """
        request = InferenceRequest(
            purpose="judge", provider=cast(Literal["openai"], self._binding.provider),
            requested_model=self._binding.requested_model, schema_id=READER_SCHEMA_ID,
            prompt_version=READER_PROMPT_VERSION,
            input_digest=self._bind("search-reader-input", {"message": message}),
            context_digest=self._bind("search-reader-context", {
                "schema": READER_SCHEMA_ID, "prompt": READER_PROMPT}),
            messages=[Message(role="system", content=READER_PROMPT),
                      Message(role="user", content=message)],
            timeout_ms=READER_TIMEOUT_MS, retry_policy=_ONE_SEND,
            reasoning_effort=self._binding.reasoning_effort, thinking="disabled",
            max_output_tokens=READER_MAX_TOKENS,
        )
        shown = min(hits, READER_HITS)

        def call() -> str | None:
            # The note is advisory: whatever the call raises (a provider, transport or
            # reply failure) is no note, never a Search failure.
            try:
                output = self._provider.infer(request).output
            except Exception:  # noqa: BLE001
                return None
            return reader_note(output, shown) if isinstance(output, ReaderNoteV1) else None

        future = submit_note_call(call)
        deadline = monotonic() + READER_TIMEOUT_MS / 1000 + _GRACE_S

        def wait() -> str | None:
            try:
                return future.result(timeout=max(0.0, deadline - monotonic()))
            except FutureTimeout:
                return None
            finally:
                # A call still in flight finishes on its own read timeout; nothing waits for it.
                future.cancel()

        return wait


def reader_noted(result: SearchResult, note: str) -> SearchResult:
    """Close the last hit with one escaped ``Note:`` line just before its closing tag.

    The note follows every delivered result and every other note of the response
    (premise, conflict, temporal, interval, standing-instruction and forget-mark notes
    keep their places), so a reader meets the results first and the reading aid last.

    Args:
        result: Delivered hits rendered by ``bench_render.render_projections``.
        note: A `ReaderNote.begin` wait result.

    Returns:
        The same hits, ids, order and scores; only the last hit's text gains the line.

    Raises:
        QuoteMemError: service_unhealthy when the last hit lacks its closing tag.
    """
    *rest, last = result.hits
    if not last.content.endswith(_CLOSE):
        raise QuoteMemError("service_unhealthy", "Search hit lacks its closing tag")
    content = (f"{last.content[:-len(_CLOSE)]}\n{escape('Note: ' + note, quote=False)}"
               f"{_CLOSE}")
    return SearchResult((*rest, replace(last, content=content)))
