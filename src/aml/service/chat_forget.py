"""Carry out a completed Add's in-chat forget requests: retire the fact, mark what states it.

A user who writes "Please forget that I ..." in a chat asks the memory to forget a
fact. When the workspace's frozen extraction policy names a ``chat_forget`` rule,
`ChatForget` runs once the Add's owner has released, before the Add answers
(`aml.service.forget_statement.StatementForget.chat_forget`): every request's statement
is Searched; the product's judge decides, for all of them at once, which current
memories and which readable Evidence sentences state it or a part of it. Every such
memory, a paraphrase as much as the matched statement, goes through the D.3 forget
(tombstone, no relearning); the memory that only records the request ("The user asked
to forget that ...") stays, as does the request sentence. Every such sentence stays
readable and is marked, so Search shows that sentence as something the user later asked
not to use. A summary-only memory (an overview or the assistant's work) that states it, or
states a memory forgotten for it, is withdrawn, so not even a summary read delivers it
again. A request names the fact in the user's own words, so a stored "father has diabetes"
goes with a request to forget "a family history of type 2 diabetes".

A retryable refusal runs the Add's requests again, up to ``ATTEMPTS`` times in the
Add's own budget (a fact already forgotten no longer matches). Requests still not
carried out fail the Add with the retryable ``delivery_busy``: the Add's memories are
committed, and its exact replay answers the first result again and then carries out
the requests once more. An Add is never acknowledged while a request is left undone.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Final, Protocol, runtime_checkable

from aml.core.errors import AMLError
from aml.core.extraction_policy import ChatForgetRule
from aml.core.request_budget import RequestBudget
from aml.ingestion.boundary import AddRequestV1
from aml.ingestion.forget_requests import forget_requests
from aml.service.forget_statement import ChatForgetOutcome

ATTEMPTS: Final = 3
"""Runs of one Add's requests before the Add answers ``delivery_busy``."""
RETRY_SECONDS: Final = 2.0
"""The pause before the requests run again."""
_RETRYABLE: Final = frozenset({
    "provider_unavailable", "operation_conflict", "intent_expired", "erasure_in_progress",
    "index_rebuilding", "delivery_busy",
})
"""Refusals that running the requests again can get past."""


@runtime_checkable
class ChatForgetOperations(Protocol):
    """The in-chat forget of one Add's requests (``service/forget.ForgetService``)."""

    def forget_in_chat(
        self, *, user_id: str, session_id: str, statements: Sequence[str],
    ) -> ChatForgetOutcome: ...


class ChatForget:
    """Carry out the in-chat forget requests of completed Adds in one workspace.

    Args:
        operations: The runtime's forget operations (``service/forget.ForgetService``).
        rule: The frozen policy's in-chat forget rule.
        sleep: Waits between runs (injected by tests).
    """

    def __init__(
        self, operations: ChatForgetOperations, rule: ChatForgetRule, *,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._operations, self.rule, self._sleep = operations, rule, sleep

    def carry_out(self, request: AddRequestV1, budget: RequestBudget) -> ChatForgetOutcome | None:
        """Forget what the requests of a completed Add ask to forget.

        Args:
            request: The completed Add.
            budget: The Add's own request budget.

        Returns:
            ChatForgetOutcome | None: What the requests did; None when the Add holds none.

        Raises:
            AMLError: ``delivery_busy`` when the requests are still not carried out after
                ``ATTEMPTS`` runs or the budget runs out; any error that running them again
                cannot get past.
        """
        statements = [found.statement for found in forget_requests(request.messages)]
        if not statements:
            return None
        for attempt in range(ATTEMPTS):
            if attempt:
                self._sleep(min(RETRY_SECONDS, budget.remaining()))
            budget.check()
            try:
                return self._operations.forget_in_chat(
                    user_id=request.user_id, session_id=request.session_id,
                    statements=statements)
            except AMLError as error:
                if error.code not in _RETRYABLE:
                    raise
        raise AMLError("delivery_busy", "An in-chat forget request is not carried out yet; retry")
