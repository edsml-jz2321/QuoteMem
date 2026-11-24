"""The forget operations the runtime mounts on the routes, and forget by statement.

``ForgetService`` matches the ``ForgetFactory`` seam
(``service/runtime.py``): ``(ready, publisher, auth, /, *, controller)``. It
resolves the exact user grant through the runtime's bench authorization and runs
the single Writer's ``ForgetOwner`` (which keys the client session; no raw
external identity enters a Control). Codes pass through unchanged; the forget
routes map ``intent_expired``, ``reception_stale`` and ``budget_exceeded`` to
422 ``operation_conflict``.

Forget by statement (`StatementForget`) runs on the same owner: its prepare
Searches through the controller's Search and asks the workspace's judge, and its
confirm forgets every picked memory through the same forget.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import cast

from quotemem.api.bench_search import bench_ranked
from quotemem.api.bench_service import SearchOperation
from quotemem.api.memory_forget import (
    ForgetStatementAffectedV1,
    MemoryForgetPreparedV2,
    MemoryForgetPrepareRequestV1,
    MemoryForgetRequestV1,
    MemoryForgetResponseV2,
    MemoryForgetStatementPreparedV1,
    MemoryForgetStatementPrepareRequestV1,
    MemoryForgetStatementRequestV1,
    MemoryForgetStatementResponseV1,
)
from quotemem.core.bootstrap import InitializedWorkspace
from quotemem.core.entities import QuoteSpanV1
from quotemem.core.identity import ArtifactId, IntegrityDigest, integrity_digest, scope_id
from quotemem.core.query import ScopeV1
from quotemem.providers.auth import AuthRequest, PrincipalGrant
from quotemem.service.auth import BenchExactUserAuth
from quotemem.service.forget_statement import ChatForgetOutcome, ChatForgetRequest, StatementForget
from quotemem.store.forget_run import ForgetOwner, workspace_embedding
from quotemem.store.read_model import StoreReadModel
from quotemem.store.writer import Writer


def _now_ms() -> int:
    return time.time_ns() // 1_000_000


class ForgetService:
    """The single Writer's forget owner behind the prepare/confirm routes.

    Args:
        ready: The runtime's initialized workspace (its sole Writer and providers).
        publisher: The runtime's one read-model publisher.
        auth: The runtime's exact-user authorization.
        controller: The runtime's Add controller. Accepted for the seam; the
            erasure fence itself refuses its owners' writes in the fenced scope
            (``erasure_in_progress``), and forget admission refuses while a
            finalize lease is preparing or held there (the Add owner handles cancellation).
            Its ``search`` is the Search forget by statement runs.
        clock: The trusted millisecond clock.
    """

    def __init__(
        self, ready: InitializedWorkspace, publisher: StoreReadModel,
        auth: BenchExactUserAuth, /, *, controller: object = None,
        clock: Callable[[], int] | None = None,
    ) -> None:
        self._writer: Writer = ready.writer
        self._auth = auth
        self._controller = controller
        self._clock: Callable[[], int] = clock or _now_ms
        self._owner = ForgetOwner(ready.writer, publisher, clock=self._now)
        if self._owner.forgets_completely():
            embedder = ready.providers.embedding
            # A complete forget's run embeds the readable rest of each part it withholds
            # bytes of, so Search still finds those bytes (``ForgetRemainderV1``).
            self._owner.embed_remainders_with(
                lambda texts: workspace_embedding(ready.writer, embedder, texts))
        self._ready = ready
        self._statements: StatementForget | None = None

    def _now(self) -> int:
        return self._clock()

    def _statement_forget(self) -> StatementForget:
        """Forget by statement over the controller's Search and the workspace's judge provider.

        Built at its first use. A memory forget touches the providers only in a complete
        (``agent``) workspace: its prepare asks the judge what states the forgotten fact
        (``StatementForget.memory_plan``) and its run embeds the readable rest of what it
        withholds (``ForgetRemainderV1``).
        """
        if self._statements is None:
            search = getattr(self._controller, "search", None)
            # The interactive Search withholds what forget requests name; a forget must
            # find it, so it searches with the bench ranking (``bench_ranked``).
            self._statements = StatementForget(
                self._writer, self._owner,
                search=None if search is None else bench_ranked(cast(SearchOperation, search)),
                provider=self._ready.providers.inference, clock=self._now,
                embedder=self._ready.providers.embedding,
            )
        return self._statements

    def _scope(self, user_id: str) -> tuple[ScopeV1, PrincipalGrant]:
        scope = ScopeV1(scope_kind="user", scope_id=scope_id("user", user_id))
        grant = self._auth.resolve(AuthRequest(requested_scope=scope, now_ms=self._now()))
        return scope, grant

    def _scope_state(self, scope: ScopeV1) -> bool:
        """Whether a maintenance successor holds the scope after the forget."""
        return any(
            str(row["key"]).startswith(f"maintenance:{scope.scope_id}:")
            for row in self._writer.indexed_reader().scope_fences(scope.scope_id)
        )

    def prepare_memory_forget(
        self, request: MemoryForgetPrepareRequestV1,
    ) -> MemoryForgetPreparedV2:
        """Bind an intent to the selected head and disclose the retention, and in a
        complete (``agent``) workspace every other memory the forget takes or may leave
        unreadable (``ForgetOwner.affected``). A complete forget also withholds the sentences
        that state or name what it forgets and the assistant's echoes of it
        (``StatementForget.memory_sentences``), as a forget by statement does, and refuses
        with the retryable ``provider_unavailable`` when the judge cannot settle what else
        tells the fact (``StatementForget.memory_plan``)."""
        scope, grant = self._scope(request.user_id)
        complete = self._owner.forgets_completely()
        plan = (self._statement_forget().memory_plan(scope, request.memory_id)
                if complete else None)
        restating: list[QuoteSpanV1] = [] if plan is None else list(plan.sentences.spans)
        # A complete prepare names everything its forget takes before the user confirms,
        # and the forget refuses to take anything else (``ForgetOwner.prepare``'s ``listed``).
        found = [] if plan is None else self._owner.affected(
            scope, [request.memory_id], restating=restating, allowed=plan.allowed)
        prepared = self._owner.prepare(
            scope, grant, session_id=request.client_session_id,
            memory_id=request.memory_id, expected_revision_id=request.expected_revision_id,
            restating=restating,
            allowed=None if plan is None else sorted(plan.allowed),
            listed=None if plan is None else [request.memory_id,
                                              *(memory_id for memory_id, *_ in found)],
        )
        if plan is not None:
            self._statement_forget().remember_memory_plan(
                scope, prepared.intent_key, prepared.expires_at, plan.sentences,
                {request.memory_id, *plan.chain})
        confirmation = MemoryForgetRequestV1(
            schema_version=1, user_id=request.user_id,
            client_session_id=request.client_session_id, request_id=request.request_id,
            memory_id=prepared.memory_id, expected_revision_id=prepared.expected_revision_id,
            expected_content_hash=IntegrityDigest(prepared.expected_content_hash),
            intent_key=prepared.intent_key,
        )
        affected = [ForgetStatementAffectedV1(memory_id=memory_id, effect=effect,
                                              statement=statement)
                    for memory_id, effect, statement in found]
        return MemoryForgetPreparedV2(confirmation=confirmation, expires_at=prepared.expires_at,
                                      affected=affected)

    def forget_memory(self, request: MemoryForgetRequestV1) -> MemoryForgetResponseV2:
        """Forget one memory; a repeated confirm returns the original verified receipt."""
        scope, grant = self._scope(request.user_id)
        started = self._now()
        outcome = self._owner.forget(
            scope, grant, session_id=request.client_session_id,
            memory_id=request.memory_id, expected_revision_id=request.expected_revision_id,
            expected_content_hash=str(request.expected_content_hash),
            intent=request.intent_key,
        )
        if self._owner.forgets_completely():
            # What the prepare judged and kept stays found under the names this forget
            # stamped.
            self._statement_forget().record_memory_plan(scope, request.intent_key, started)
        return MemoryForgetResponseV2(
            request_digest=integrity_digest(request.model_dump(mode="json")),
            controller_run_id=outcome.controller_run_id,
            receipt_ref=ArtifactId(outcome.receipt_ref), receipt=outcome.receipt,
            scope_state="maintenance_pending" if self._scope_state(scope) else "ready",
        )

    def prepare_statement_forget(
        self, request: MemoryForgetStatementPrepareRequestV1,
    ) -> MemoryForgetStatementPreparedV1:
        """Search the statement, let the judge pick, and return the plan to confirm."""
        scope, grant = self._scope(request.user_id)
        return self._statement_forget().prepare(scope, grant, request)

    def forget_in_chat(
        self, *, user_id: str, session_id: str, statements: Sequence[str],
        request_id: str | None = None,
    ) -> ChatForgetOutcome:
        """Forget what one Add's in-chat requests ask to forget (``StatementForget.chat_forget``).
        """
        scope, _ = self._scope(user_id)
        return self._statement_forget().chat_forget(
            scope, lambda: self._scope(user_id)[1], user_id=user_id, session_id=session_id,
            statements=statements, request_id=request_id)

    def add_forgets(
        self, *, user_id: str, request_id: str, statements: Sequence[str],
    ) -> tuple[list[ChatForgetRequest], int]:
        """What one Add's in-chat forget requests came to (``agent`` mode), read only:
        each request of ``statements`` (the Add's own, ``forget_requests``'s ``complete``) in
        order, ``pending`` until its in-chat forget recorded it, and the memories the Add
        itself made that its forget took."""
        scope, _ = self._scope(user_id)
        return self._statement_forget().add_forgets(scope, request_id, statements)

    def forget_statement(
        self, request: MemoryForgetStatementRequestV1,
    ) -> MemoryForgetStatementResponseV1:
        """Forget every target of the confirmed plan; resumable."""
        scope, grant = self._scope(request.user_id)
        statements = self._statement_forget()
        report = statements.confirm_report(scope, grant, request)
        outcomes = report.outcomes
        return MemoryForgetStatementResponseV1(
            status="verified" if all(row.outcome != "changed" for row in outcomes)
            else "incomplete",
            request_digest=integrity_digest(request.model_dump(mode="json")),
            outcomes=list(outcomes),
            scope_state="maintenance_pending" if self._scope_state(scope) else "ready",
            earlier=statements.earlier(scope, request.request_id),
            forgotten=report.forgotten, withheld_spans=report.withheld,
            affected=list(report.affected),
        )
