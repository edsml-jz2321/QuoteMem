"""S2c's ``RecoveryStore`` on the real Writer (C.13 R08, ingest-Q12, ingest-Q17; I15).

``WriterRecoveryStore`` is the Writer side of explicit owner recovery. It reads
the owner's provider pause from its stored proof (the owner's own
``result_index:provider_failure`` for a stage the owner calls itself, with or
without HTTP diagnostics: extraction, statement embedding or the judge; the
failed label child's for ``source_labels``, or the owner's own when the
provider refused the label call's key and the child still runs) and commits
each resume in one Writer transaction: the pool's join proof is validated,
``explicit_resume_allowance`` selects the counted fresh allowance, the owner
moves to its next attempt and the exact lease is renewed. A quarantined scope
is repaired the same way: the resume of the owner its receipt names lifts the
quarantine (``owner_lease.repair_quarantine``).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Any

from aml.core.controls import (
    ControlRowV1,
    FinalizeBudgetRefusalV4,
    FinalizeBudgetV2,
    HeldFinalizeLeaseV1,
    PreparingFinalizeLeaseV1,
)
from aml.core.errors import AMLError
from aml.core.provider_diagnostics import access_refused
from aml.core.query import ScopeV1
from aml.ingestion.authorization import check_add_grant
from aml.ingestion.owner_recovery import explicit_resume_allowance, provider_failure
from aml.providers.auth import AuthProvider, AuthRequest
from aml.service.bench_workers import RetiredOwnerJoined
from aml.store.extraction_budget import resumed_call_allowance, resumed_outage_allowances
from aml.store.owner_pause import PROVIDER_PAUSE_FAILURE_CODE, paused_index
from aml.store.request_completion import (
    REFUSED_OWNER_FAILURE_CODE,
    read_budget,
    resume_refused_budget_row,
)

if TYPE_CHECKING:
    from aml.api.bench_recovery import ProviderFailureKind, ProviderFailureV1
    from aml.api.completion_recovery import CompletionResumeRequestV3
    from aml.store.writer import RowWrite, Writer


class WriterRecoveryStore:
    """The ``RecoveryStore`` over the sole Writer (``owner_recovery.recovery_factory``).

    Args:
        writer: The sole Writer.
        auth: The bench's exact-user authorization.
        clock: Trusted millisecond clock.
        work_budget_ms: Optional allowance for each recovery; omission keeps the
            installed scope policy.
    """

    def __init__(
        self, writer: Writer, auth: AuthProvider, *, clock: Callable[[], int],
        work_budget_ms: int | None = None,
    ) -> None:
        self._writer = writer
        self._auth = auth
        self._clock = clock
        self._work_budget_ms = work_budget_ms

    def exclusive(self) -> AbstractContextManager[object]:
        """The sole Writer's mutex."""
        return self._writer.mutex

    def authorize(self, scope: ScopeV1) -> None:
        """Check a fresh Add grant for the scope before any private read."""
        now = self._clock()
        check_add_grant(self._auth.resolve(AuthRequest(requested_scope=scope, now_ms=now)),
                        scope, now)

    def paused_failure(self, scope: ScopeV1) -> ProviderFailureV1 | None:
        """The scope owner's current provider pause from its stored proof, or None."""
        owner = self._owner(scope)
        if owner is None or (owner["status"], owner["failure_code"]) != (
            "manual_attention", PROVIDER_PAUSE_FAILURE_CODE,
        ):
            return None
        own = paused_index(self._writer, owner["object_id"])
        kind: ProviderFailureKind = "source_labels" if own is None else (
            own[1].provider_call_kind or "extract"
        )
        if kind != "source_labels":
            return provider_failure(owner, owner, kind)
        child = self._label_child(owner)
        if child is None:
            raise AMLError("binding_mismatch", "A provider pause has no stored proof")
        # A refused key leaves the child running and indexes the pause on the
        # owner with the refused call's diagnostics (LABELS-RESUME); any other
        # label failure is the failed child's own.
        refused = child["status"] == "running"
        if refused and (own is None or not access_refused(own[1].provider_failure)):
            raise AMLError("binding_mismatch", "A running label child paused without a key refusal")
        return provider_failure(owner, child, "source_labels", key_refused=refused)

    def commit_provider_resume(
        self, scope: ScopeV1, selection: ProviderFailureV1, joined: RetiredOwnerJoined,
    ) -> None:
        """Resume the paused owner on its next attempt in one Writer transaction."""
        writer = self._writer
        with writer._write_scope():  # pyright: ignore[reportPrivateUsage]
            current = self.paused_failure(scope)
            if current != selection:
                raise AMLError("operation_conflict", "The observed provider pause changed")
            owner, budget, now = self._resumable(scope, selection.owner_run_id, joined)
            if not isinstance(budget.value, FinalizeBudgetV2):
                raise AMLError("operation_conflict", "A provider pause keeps a live budget")
            value = budget.value
            allowance = explicit_resume_allowance(
                recovery_count=value.recovery_count,
                admitted_epochs=(value.erase_epoch, value.consent_epoch),
                current_epochs=self._epochs(scope), heartbeat_at=self._heartbeat(scope, owner),
                now=now, allowance_ms=self._allowance(scope),
            )
            active = FinalizeBudgetV2.model_validate({
                **value.model_dump(mode="json"), "state": "active",
                "owner_attempt": owner["attempt"] + 1, "admitted_at": allowance.admitted_at,
                "deadline_at": allowance.deadline_at,
                "recovery_count": allowance.recovery_count,
            })
            from aml.store.writer import RowWrite

            updated = RowWrite("control", writer.materialize_control(
                budget.key, active.model_dump(mode="json"), budget.version + 1, now,
                owner["object_id"],
            ))
            calls: tuple[RowWrite, ...] = ()
            if selection.kind == "extract":
                calls = resumed_outage_allowances(
                    writer, owner, now=now, deadline_at=active.deadline_at,
                )
            elif selection.kind == "source_labels":
                from aml.store.source_labels import resumed_label_allowance

                label = resumed_label_allowance(
                    writer, owner, now=now, deadline_at=active.deadline_at,
                )
                calls = () if label is None else (label,)
            self._commit(scope, owner, updated, now, *calls)

    def commit_refused_resume(
        self, scope: ScopeV1, request: CompletionResumeRequestV3, joined: RetiredOwnerJoined,
    ) -> None:
        """Replace the observed refusal and move the owner to its next attempt (ingest-Q12).

        The fresh allowance is the request's ``work_budget_ms`` when it names one,
        else this store's, else the scope's installed policy (C.13, ingest-Q23).
        The same transaction grants the refused unit's call its fresh send
        allowance, so the resumed owner re-runs that call (B.7.1,
        ``resumed_call_allowance``).
        """
        writer = self._writer
        work_budget_ms = (
            self._work_budget_ms if request.work_budget_ms is None else request.work_budget_ms
        )
        with writer._write_scope():  # pyright: ignore[reportPrivateUsage]
            owner, budget, now = self._resumable(scope, request.owner_run_id, joined)
            if not isinstance(budget.value, FinalizeBudgetRefusalV4) or (
                budget.version, budget.value.recovery_count,
            ) != (request.budget_version, request.recovery_count) or (
                owner["status"], owner["failure_code"],
            ) != ("manual_attention", REFUSED_OWNER_FAILURE_CODE):
                raise AMLError("operation_conflict", "The observed refusal is no longer current")
            explicit_resume_allowance(
                recovery_count=budget.value.recovery_count,
                admitted_epochs=(budget.value.erase_epoch, budget.value.consent_epoch),
                current_epochs=self._epochs(scope), heartbeat_at=self._heartbeat(scope, owner),
                now=now, allowance_ms=self._allowance(scope, work_budget_ms),
            )
            updated = resume_refused_budget_row(
                writer, owner, budget, now=now, work_budget_ms=work_budget_ms,
            )
            resumed = ControlRowV1.model_validate(updated.row.model_dump(mode="json")).value
            if not isinstance(resumed, FinalizeBudgetV2):
                raise AMLError("binding_mismatch", "A refused resume lost its fresh allowance")
            call = resumed_call_allowance(
                writer, owner, budget.value.unit_no, now=now, deadline_at=resumed.deadline_at,
            )
            self._commit(scope, owner, updated, now, *(() if call is None else (call,)))

    def commit_quarantine_repair(
        self, scope: ScopeV1, request: CompletionResumeRequestV3, joined: RetiredOwnerJoined,
    ) -> None:
        """Repair the observed quarantine through its owner's explicit resume (C.13).

        The observed budget must still be the owner's current one; the resume
        grants the requested (else this store's, else the scope's) allowance and
        counts against the owner's shared resume cap, as every counted resume does.
        """
        from aml.store.owner_lease import repair_quarantine
        from aml.store.writer import RowWrite

        writer = self._writer
        work_budget_ms = (
            self._work_budget_ms if request.work_budget_ms is None else request.work_budget_ms
        )
        with writer._write_scope():  # pyright: ignore[reportPrivateUsage]
            owner, budget, now = self._resumable(scope, request.owner_run_id, joined)
            value = budget.value
            if not isinstance(value, FinalizeBudgetV2) or (
                budget.version, value.recovery_count,
            ) != (request.budget_version, request.recovery_count):
                raise AMLError("operation_conflict", "The observed budget is no longer current")
            allowance = explicit_resume_allowance(
                recovery_count=value.recovery_count,
                admitted_epochs=(value.erase_epoch, value.consent_epoch),
                current_epochs=self._epochs(scope), heartbeat_at=self._heartbeat(scope, owner),
                now=now, allowance_ms=self._allowance(scope, work_budget_ms),
            )
            active = FinalizeBudgetV2.model_validate({
                **value.model_dump(mode="json"), "state": "active",
                "owner_attempt": owner["attempt"] + 1, "admitted_at": allowance.admitted_at,
                "deadline_at": allowance.deadline_at,
                "recovery_count": allowance.recovery_count,
            })
            repair_quarantine(writer, scope, owner["object_id"], RowWrite(
                "control", writer.materialize_control(
                    budget.key, active.model_dump(mode="json"), budget.version + 1, now,
                    owner["object_id"],
                ),
            ), now=now)

    # ------------------------------------------------------------------ helpers

    def _owner(self, scope: ScopeV1) -> dict[str, Any] | None:
        lease = self._writer.get_control(f"lease:finalize:{scope.scope_id}")
        if lease is None:
            return None
        return self._writer.get_row("run", lease["value"]["owner_run_id"])

    def _label_child(self, owner: Mapping[str, Any]) -> dict[str, Any] | None:
        from aml.store.source_labels import (
            _family_child,  # pyright: ignore[reportPrivateUsage]
        )

        return _family_child(self._writer, owner)

    def _resumable(
        self, scope: ScopeV1, owner_run_id: str, joined: RetiredOwnerJoined,
    ) -> tuple[dict[str, Any], ControlRowV1, int]:
        writer = self._writer
        pool = writer._finalize_pool  # pyright: ignore[reportPrivateUsage]
        if pool is None:
            raise AMLError("binding_mismatch", "Owner recovery requires the actual worker pool")
        pool.validate_retired_owner_join(joined)
        if (joined.work.scope_id, joined.work.run_id, joined.work.kind) != (
            scope.scope_id, owner_run_id, "finalize",
        ):
            raise AMLError("binding_mismatch", "Owner recovery selected another joined owner")
        owner = writer.get_row("run", owner_run_id)
        budget = read_budget(writer, scope, owner_run_id)
        if owner is None or budget is None:
            raise AMLError("operation_conflict", "Owner recovery lost its stopped owner")
        now = self._clock()
        grant = self._auth.resolve(AuthRequest(requested_scope=scope, now_ms=now))
        check_add_grant(grant, scope, now, owner["principal_id"])
        return owner, budget, now

    def _epochs(self, scope: ScopeV1) -> tuple[int, int]:
        return (self._writer.counter(f"erase_epoch:{scope.scope_id}"),
                self._writer.counter(f"consent_epoch:{scope.scope_id}"))

    def _lease(self, scope: ScopeV1, owner: Mapping[str, Any]) -> ControlRowV1:
        raw = self._writer.get_control(f"lease:finalize:{scope.scope_id}")
        lease = None if raw is None else ControlRowV1.model_validate(raw)
        if lease is None or not isinstance(
            lease.value, PreparingFinalizeLeaseV1 | HeldFinalizeLeaseV1,
        ) or (lease.value.owner_run_id, lease.value.writer_epoch) != (
            owner["object_id"], owner["writer_epoch"],
        ):
            raise AMLError("operation_conflict", "Owner recovery lost its exact current lease")
        return lease

    def _heartbeat(self, scope: ScopeV1, owner: Mapping[str, Any]) -> int:
        value = self._lease(scope, owner).value
        assert isinstance(value, PreparingFinalizeLeaseV1 | HeldFinalizeLeaseV1)
        return value.heartbeat_at

    def _allowance(self, scope: ScopeV1, work_budget_ms: int | None = None) -> int:
        from aml.store.request_completion import (
            _recovery_allowance,  # pyright: ignore[reportPrivateUsage]
        )

        return _recovery_allowance(
            self._writer, scope, self._work_budget_ms if work_budget_ms is None else work_budget_ms,
        )

    def _commit(
        self, scope: ScopeV1, owner: dict[str, Any], budget: RowWrite, now: int,
        *granted: RowWrite,
    ) -> None:
        """Move the owner to its next attempt and renew the lease with the budget.

        ``granted`` are the call allowances the resume grants that attempt.
        """
        from aml.store.writer import RowWrite

        writer = self._writer
        lease = self._lease(scope, owner)
        renewed = writer.materialize_control(lease.key, {
            **lease.value.model_dump(mode="json"), "heartbeat_at": now,
        }, lease.version + 1, now, owner["object_id"])
        resumed = writer._parse_row("run", dict(  # pyright: ignore[reportPrivateUsage]
            owner, status="running", failure_code=None, attempt=owner["attempt"] + 1,
            updated_at=now, finished_at=None,
        ))
        _, plan = writer._prepare_rows(  # pyright: ignore[reportPrivateUsage]
            [budget, RowWrite("control", renewed), RowWrite("run", resumed), *granted], None,
        )
        writer._write_plan(plan)  # pyright: ignore[reportPrivateUsage]
