"""Search's pool re-rank call (``search-rerank-v1``) for very large scopes.

`PoolReranker` sends a plain read's question and its leading fused pool candidates to the
pipeline's model without reasoning, one call per batch of `RERANK_BATCH` candidates, all
in parallel under one read timeout and no retry. Recall moves the candidates the replies
name ahead of the rest (``aml.core.recall``). Any provider failure, timeout or refused
reply in any batch keeps the fused order: Search never fails because of the re-rank.
"""

from __future__ import annotations

import contextvars
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from time import monotonic
from typing import Literal, cast

from aml.core.errors import AMLError
from aml.core.identity import ContentBindingDigest
from aml.core.inference import InferenceBindingV1, RetryPolicyV1
from aml.core.pool_rerank import (
    RERANK_PROMPT,
    RERANK_PROMPT_VERSION,
    RERANK_SCHEMA_ID,
    RerankCandidate,
    RerankOrderV1,
    batch_order,
    merged_order,
    rerank_batches,
    rerank_message,
)
from aml.providers.contracts import InferenceRequest, Message, StructuredInferenceProvider

RERANK_TIMEOUT_MS = 15_000
"""Read timeout of each call; the re-rank waits at most this long plus `_GRACE_S` in all."""
RERANK_MAX_TOKENS = 1_000
"""Completion bound of one reply: at most `RERANK_BATCH` indices."""
_GRACE_S = 0.5
_ONE_SEND = RetryPolicyV1(
    max_attempts=1, initial_backoff_ms=0, max_backoff_ms=0, multiplier=1.0,
    jitter_mode="none", retryable_error_classes=[],
)


class PoolReranker:
    """The pool re-rank of one Search runtime: the provider and the pipeline's judge model.

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
            raise AMLError("binding_mismatch", "The pool re-rank runs on the chat judge binding")
        self._provider = provider
        self._binding = binding
        self._bind = bind

    def _request(self, message: str) -> InferenceRequest:
        return InferenceRequest(
            purpose="judge", provider=cast(Literal["openai"], self._binding.provider),
            requested_model=self._binding.requested_model, schema_id=RERANK_SCHEMA_ID,
            prompt_version=RERANK_PROMPT_VERSION,
            input_digest=self._bind("search-rerank-input", {"message": message}),
            context_digest=self._bind("search-rerank-context", {
                "schema": RERANK_SCHEMA_ID, "prompt": RERANK_PROMPT}),
            messages=[Message(role="system", content=RERANK_PROMPT),
                      Message(role="user", content=message)],
            timeout_ms=RERANK_TIMEOUT_MS, retry_policy=_ONE_SEND,
            reasoning_effort=self._binding.reasoning_effort, thinking="disabled",
            max_output_tokens=RERANK_MAX_TOKENS,
        )

    def __call__(self, query: str, candidates: Sequence[RerankCandidate]) -> list[int] | None:
        """Re-rank one pool (``aml.core.pool_rerank.Rerank``).

        Args:
            query: The frozen query text.
            candidates: The pool's leading candidates, in fused order.

        Returns:
            Positions of the candidates the replies name, interleaved across batches
            (`merged_order`); None when any call fails, is refused or does not
            answer in time.
        """
        batches = rerank_batches(len(candidates))
        if not batches:
            return None
        requests = [self._request(rerank_message(query, candidates, batch)) for batch in batches]

        def call(request: InferenceRequest) -> RerankOrderV1 | None:
            # The re-rank is advisory: whatever a call raises (a provider, transport or
            # reply failure) keeps the fused order, never a Search failure.
            try:
                output = self._provider.infer(request).output
            except Exception:  # noqa: BLE001
                return None
            return output if isinstance(output, RerankOrderV1) else None

        pool = ThreadPoolExecutor(max_workers=len(requests), thread_name_prefix="aml-rerank")
        try:
            futures = [pool.submit(contextvars.copy_context().run, call, request)
                       for request in requests]
            deadline = monotonic() + RERANK_TIMEOUT_MS / 1000 + _GRACE_S
            replies = [future.result(timeout=max(0.0, deadline - monotonic()))
                       for future in futures]
        except FutureTimeout:
            return None
        finally:
            # A call still in flight finishes on its own read timeout; nothing waits for it.
            pool.shutdown(wait=False, cancel_futures=True)
        orders: list[list[int]] = []
        for reply, batch in zip(replies, batches, strict=True):
            if reply is None:
                return None
            orders.append(batch_order(reply, batch))
        return merged_order(orders)
