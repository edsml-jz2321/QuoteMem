"""Carry out a completed Add's in-chat forget requests: retire the fact, mark what states it.

A user who writes "Please forget that I ..." in a chat asks the memory to forget a
fact. When the workspace's frozen extraction policy names a ``chat_forget`` rule,
`ChatForget` runs once the Add's owner has released, before the Add answers
(`quotemem.service.forget_statement.StatementForget.chat_forget`): every request's statement
is Searched; the product's judge decides, for all of them at once, which current
memories and which readable Evidence sentences state it or a part of it. Every such
memory, a paraphrase as much as the matched statement, goes through the forget
(tombstone, no relearning); the memory that only records the request ("The user asked
to forget that ...") stays, as does the request sentence. Every such sentence stays
readable and is marked, so Search shows that sentence as something the user later asked
not to use. A summary-only memory (an overview or the assistant's work) that states it, or
states a memory forgotten for it, is withdrawn, so not even a summary read delivers it
again. A request names the fact in the user's own words, so a stored "father has diabetes"
goes with a request to forget "a family history of type 2 diabetes".

A forget that meets another erasure of the scope waits for its turn inside the request
(`quotemem.store.erasure_gate`, at most ``FORGET_WAIT_SECONDS`` and never past the Add's budget).
A retryable refusal, a wait for a turn that passed its bound among them, runs the Add's
requests again, up to ``ATTEMPTS`` times in the Add's own budget (a fact already forgotten
no longer matches); a run that already waited for its turn is not paused before the next
one, whose wait is bounded by what is left of the budget. A run that had withheld its
targets and then failed showed them again, so the Add never runs its requests again in
the same request: it answers ``delivery_busy`` at once and a forget never erases what it
has shown again (`quotemem.store.erasure_gate.noting_holds`). Requests still not
carried out fail the Add with the retryable ``delivery_busy``: the Add's memories are
committed, and its exact replay answers the first result again and then carries out
the requests once more. An Add is never acknowledged while a request is left undone.

A workspace whose rule is ``explicit-request-v2`` (the ``agent`` Add mode) forgets
completely (``service/forget_statement.py``): the record of the request and
the request sentence go with the fact, the sentences restating it are withheld rather than
marked, and every Add, with requests or not, is checked for memories and sentences restating
a forgotten fact, so an assistant reply recounting it in a later session is forgotten again;
what the user states again in their own words after the forget is learned again.
A request the plugin's forget by statement already handled is read only for what changed since.
Its requests also take the Chinese forms the ``agent`` plugins' users write
(``forget_requests``'s ``complete``).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Final, Protocol, runtime_checkable

from quotemem.core.errors import QuoteMemError
from quotemem.core.extraction_policy import CHAT_FORGET_COMPLETE, ChatForgetRule
from quotemem.core.request_budget import RequestBudget
from quotemem.ingestion.boundary import AddRequestV1
from quotemem.ingestion.forget_requests import forget_requests
from quotemem.service.forget_statement import ChatForgetOutcome, ChatForgetRequest
from quotemem.store.erasure_gate import noting_holds, waiting_until

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
        request_id: str | None = None,
    ) -> ChatForgetOutcome: ...


@runtime_checkable
class ChatForgetReports(Protocol):
    """What one Add's in-chat forget requests did (``service/forget.ForgetService``)."""

    def add_forgets(
        self, *, user_id: str, request_id: str, statements: Sequence[str],
    ) -> tuple[list[ChatForgetRequest], int]: ...


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
            ChatForgetOutcome | None: What the requests did; None when the Add holds none
                and the rule is not the complete one (a complete rule checks every Add).

        Raises:
            QuoteMemError: ``delivery_busy`` when the requests are still not carried out after
                ``ATTEMPTS`` runs or the budget runs out; any error that running them again
                cannot get past.
        """
        complete = self.rule == CHAT_FORGET_COMPLETE
        statements = [found.statement
                      for found in forget_requests(request.messages, complete=complete)]
        if not statements and not complete:
            return None
        waited = False
        for attempt in range(ATTEMPTS):
            if attempt and not waited:
                self._sleep(min(RETRY_SECONDS, budget.remaining()))
            budget.check()
            started = time.monotonic()
            with noting_holds() as held:
                try:
                    # Every wait for the scope's erasure turn ends within the Add's own budget.
                    with waiting_until(time.monotonic() + budget.remaining()):
                        if complete:
                            return self._operations.forget_in_chat(
                                user_id=request.user_id, session_id=request.session_id,
                                statements=statements, request_id=request.request_id)
                        return self._operations.forget_in_chat(
                            user_id=request.user_id, session_id=request.session_id,
                            statements=statements)
                except QuoteMemError as error:
                    if error.code not in _RETRYABLE:
                        raise
                    if held():
                        # The run withheld its targets and showed them again: they are never
                        # erased later in this request.
                        raise QuoteMemError(
                            "delivery_busy",
                            "An in-chat forget request is not carried out yet; retry",
                        ) from error
                    # A run that waited for its turn until the bound passed has paused already.
                    waited = error.code == "erasure_in_progress" and (
                        time.monotonic() - started >= RETRY_SECONDS)
        raise QuoteMemError("delivery_busy",
                            "An in-chat forget request is not carried out yet; retry")

    def report(self, request: AddRequestV1) -> list[dict[str, object]]:
        """What each in-chat forget request of a completed Add did.

        Only a complete rule (``explicit-request-v2``, the ``agent`` mode) records outcomes
        (``forget_add:{s}``, ``StatementForget.add_forgets``); the request words come from the
        Add itself (``forget_requests``), as the record keeps only their keyed digests.

        Returns:
            One entry per request, in message order: ``request`` (its words), ``status``
            (``forgotten``, ``nothing_matched``, or ``pending`` while no outcome is recorded
            or the plugin's forget by statement holds a plan for it not yet confirmed),
            ``forgotten``, ``withheld_spans``, ``bundled`` and ``earlier`` (the plugin's
            forget by statement of the same request had forgotten it before the Add read
            it). Empty when the Add holds no request or the rule is not the complete one.
        """
        if self.rule != CHAT_FORGET_COMPLETE:
            return []
        statements = [found.statement
                      for found in forget_requests(request.messages, complete=True)]
        if not statements:
            return []
        operations = self._operations
        if not isinstance(operations, ChatForgetReports):
            return [{"request": statement, "status": "pending", "forgotten": 0,
                     "withheld_spans": 0, "bundled": 0, "earlier": False}
                    for statement in statements]
        found, _ = operations.add_forgets(user_id=request.user_id, request_id=request.request_id,
                                          statements=statements)
        return [{
            "request": row.statement, "status": row.status, "forgotten": row.forgotten,
            "withheld_spans": row.withheld, "bundled": row.bundled, "earlier": row.earlier,
        } for row in found]
