"""I12: the forget operations the runtime mounts on the E.2.4 routes (D.3), and FS-1.

``ForgetService`` matches S8a's ``ForgetFactory`` seam
(``service/runtime.py``): ``(ready, publisher, auth, /, *, controller)``. It
resolves the exact user grant through the runtime's bench authorization and runs
the single Writer's ``ForgetOwner`` (which keys the client session; no raw
external identity enters a Control). Codes pass through unchanged; the forget
routes map ``intent_expired``, ``reception_stale`` and ``budget_exceeded`` to
422 ``operation_conflict`` (E.3).

Forget by statement (FS-1, `StatementForget`) runs on the same owner: its prepare
Searches through the controller's Search and asks the workspace's judge, and its
confirm forgets every picked memory through the same D.3 forget.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import cast

from aml.api.bench_service import SearchOperation
from aml.api.memory_forget import (
    MemoryForgetPreparedV2,
    MemoryForgetPrepareRequestV1,
    MemoryForgetRequestV1,
    MemoryForgetResponseV2,
    MemoryForgetStatementPreparedV1,
    MemoryForgetStatementPrepareRequestV1,
    MemoryForgetStatementRequestV1,
    MemoryForgetStatementResponseV1,
)
from aml.core.bootstrap import InitializedWorkspace
from aml.core.identity import ArtifactId, IntegrityDigest, integrity_digest, scope_id
from aml.core.query import ScopeV1
from aml.providers.auth import AuthRequest, PrincipalGrant
from aml.service.auth import BenchExactUserAuth
from aml.service.forget_statement import ChatForgetOutcome, StatementForget
from aml.store.forget_run import ForgetOwner
from aml.store.read_model import StoreReadModel
from aml.store.writer import Writer


def _now_ms() -> int:
    return time.time_ns() // 1_000_000


class ForgetService:
    """The single Writer's forget owner behind the E.2.4 prepare/confirm routes.

    Args:
        ready: The runtime's initialized workspace (its sole Writer and providers).
        publisher: The runtime's one read-model publisher.
        auth: The runtime's exact-user authorization.
        controller: The runtime's Add controller. Accepted for the seam; the
            erasure fence itself refuses its owners' writes in the fenced scope
            (``erasure_in_progress``), and forget admission refuses while a
            finalize lease is preparing or held there (cancellation is S2c's).
            Its ``search`` is the Search forget by statement runs (FS-1).
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
        self._ready = ready
        self._statements: StatementForget | None = None

    def _now(self) -> int:
        return self._clock()

    def _statement_forget(self) -> StatementForget:
        """Forget by statement over the controller's Search and the workspace's judge provider.

        Built at its first use, so a memory forget never touches the providers.
        """
        if self._statements is None:
            search = getattr(self._controller, "search", None)
            self._statements = StatementForget(
                self._writer, self._owner,
                search=None if search is None else cast(SearchOperation, search),
                provider=self._ready.providers.inference, clock=self._now,
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
        """D.3 P0: bind an intent to the selected head and disclose the retention."""
        scope, grant = self._scope(request.user_id)
        prepared = self._owner.prepare(
            scope, grant, session_id=request.client_session_id,
            memory_id=request.memory_id, expected_revision_id=request.expected_revision_id,
        )
        confirmation = MemoryForgetRequestV1(
            schema_version=1, user_id=request.user_id,
            client_session_id=request.client_session_id, request_id=request.request_id,
            memory_id=prepared.memory_id, expected_revision_id=prepared.expected_revision_id,
            expected_content_hash=IntegrityDigest(prepared.expected_content_hash),
            intent_key=prepared.intent_key,
        )
        return MemoryForgetPreparedV2(confirmation=confirmation, expires_at=prepared.expires_at)

    def forget_memory(self, request: MemoryForgetRequestV1) -> MemoryForgetResponseV2:
        """D.3 P1–P4; a repeated confirm returns the original verified receipt."""
        scope, grant = self._scope(request.user_id)
        outcome = self._owner.forget(
            scope, grant, session_id=request.client_session_id,
            memory_id=request.memory_id, expected_revision_id=request.expected_revision_id,
            expected_content_hash=str(request.expected_content_hash),
            intent=request.intent_key,
        )
        return MemoryForgetResponseV2(
            request_digest=integrity_digest(request.model_dump(mode="json")),
            controller_run_id=outcome.controller_run_id,
            receipt_ref=ArtifactId(outcome.receipt_ref), receipt=outcome.receipt,
            scope_state="maintenance_pending" if self._scope_state(scope) else "ready",
        )

    def prepare_statement_forget(
        self, request: MemoryForgetStatementPrepareRequestV1,
    ) -> MemoryForgetStatementPreparedV1:
        """FS-1: Search the statement, let the judge pick, and return the plan to confirm."""
        scope, grant = self._scope(request.user_id)
        return self._statement_forget().prepare(scope, grant, request)

    def forget_in_chat(
        self, *, user_id: str, session_id: str, statements: Sequence[str],
    ) -> ChatForgetOutcome:
        """Forget what one Add's in-chat requests ask to forget (``StatementForget.chat_forget``).
        """
        scope, _ = self._scope(user_id)
        return self._statement_forget().chat_forget(
            scope, lambda: self._scope(user_id)[1], user_id=user_id, session_id=session_id,
            statements=statements)

    def forget_statement(
        self, request: MemoryForgetStatementRequestV1,
    ) -> MemoryForgetStatementResponseV1:
        """FS-1: forget every target of the confirmed plan through D.3; resumable."""
        scope, grant = self._scope(request.user_id)
        outcomes = self._statement_forget().confirm(scope, grant, request)
        return MemoryForgetStatementResponseV1(
            status="verified" if all(row.outcome != "changed" for row in outcomes)
            else "incomplete",
            request_digest=integrity_digest(request.model_dump(mode="json")),
            outcomes=list(outcomes),
            scope_state="maintenance_pending" if self._scope_state(scope) else "ready",
        )
