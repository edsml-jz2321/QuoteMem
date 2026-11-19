"""Owner recovery: explicit resume, provider-pause selection, cold classification (C.13).

This module holds S2c's recovery proofs over the Add owner. It decides; it never
writes by itself. Every durable change runs in the sole Writer through the
``RecoveryStore`` seam, which S2b's owner lease (I15) and S1a's Writer bind; the
pool fences come from the scheduler's ``OwnerScheduling``.

- **Counted resume (R08, ingest-Q23).** At most 33 per owner, each a fresh
  allowance of at most 1,740 seconds from the resume transaction, also after
  the original deadline; only while the scope's privacy epochs equal the owner's
  admitted ones, after a real join of the previous worker, with the lease
  renewed in the same transaction and a clock regression refused as
  ``clock_rollback``. C41's Add controller may invoke these same proofs for
  the first 32 resumptions; the last remains available for explicit repair.
- **Provider pause (ingest-Q17).** The only recoverable failure is a provider
  pause of ``source_labels``, ``extract``, ``embedding`` or ``judge``; the
  owner's ``result_index:provider_failure`` is its original stored proof.
- **Cold classification (R13).** A legal state change settles (a release that
  died before its last transaction is finished first), a recoverable
  interruption is adopted by the same owner under the new Writer epoch with its
  prefix and deadline unchanged, and only an integrity break quarantines its own
  scope.
- **Zero-effect proof (R17).** An interrupted child is retried or abandoned
  only after proving it wrote nothing; a witness is ``parity_mismatch``, and a
  witnessed main commit whose catalog step failed is ``catalog_repair_required``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, Literal, Protocol

from aml.api.bench_recovery import (
    ProviderFailureKind,
    ProviderFailureV1,
    RecoveryRequestV1,
    RecoveryResponseV1,
    ResumeRequestV1,
    ResumeResponseV1,
)
from aml.core.errors import AMLError
from aml.core.identity import scope_id
from aml.core.query import ScopeV1
from aml.core.request_budget import DEFAULT_COMPLETION_WORK_BUDGET_MS, MAX_OWNER_RESUMES
from aml.service.bench_workers import BenchWork

if TYPE_CHECKING:
    from aml.api.completion_recovery import CompletionResumeRequestV3
    from aml.ingestion.add_scheduler import OwnerScheduling
    from aml.service.bench_workers import RetiredOwnerJoined

MAX_RESUME_ALLOWANCE_MS: Final = DEFAULT_COMPLETION_WORK_BUDGET_MS
"""The longest fresh allowance one explicit resume grants (C.13, ingest-Q23)."""

_OWNER_STAGES: Final[frozenset[ProviderFailureKind]] = frozenset(
    {"extract", "embedding", "judge"},
)

_LABEL_CHILD_FAILURES: Final[frozenset[str]] = frozenset(
    {"provider_error", "schema_invalid", "budget_exceeded"},
)
"""Every failure a label child settles with that pauses its owner (``store.source_labels``).

The child's own code says why its one call ended. The owner is paused alike on
each, and its resume goes on with that call's labels ambiguous; a child ended by a
source loss (``stale_evidence``) never pauses.
"""


# ---------------------------------------------------------------- explicit resume


@dataclass(frozen=True, slots=True)
class ResumeAllowance:
    """The counted fresh allowance one proved resume grants its owner.

    Attributes:
        admitted_at: The resume transaction's trusted time.
        deadline_at: ``admitted_at`` plus the allowance; it may follow the
            original deadline (ingest-Q23).
        recovery_count: The owner's resume count after this resume.
    """

    admitted_at: int
    deadline_at: int
    recovery_count: int


def explicit_resume_allowance(
    *,
    recovery_count: int,
    admitted_epochs: tuple[int, int],
    current_epochs: tuple[int, int],
    heartbeat_at: int,
    now: int,
    allowance_ms: int,
) -> ResumeAllowance:
    """Check one explicit resume's authority and return its fresh allowance (C.13 R08).

    The caller holds the Writer's write scope, has validated the pool's actual
    join proof and commits the returned allowance with the renewed lease in one
    transaction. Sends reserved before the resume stay spent.

    Args:
        recovery_count: The owner's resumes so far.
        admitted_epochs: The owner's admitted ``(erase_epoch, consent_epoch)``.
        current_epochs: The scope's current ``(erase_epoch, consent_epoch)``.
        heartbeat_at: The owner's lease heartbeat, the latest trusted time it holds.
        now: The resume transaction's trusted clock.
        allowance_ms: The selected allowance (the request's, or the scope policy's).

    Returns:
        The allowance from ``now``, with the incremented count.

    Raises:
        AMLError: ``operation_conflict`` at the shared resume cap or when the scope's
            privacy authority changed since admission; ``invalid_request`` for an
            allowance outside 1 ms to 1,740 seconds; ``clock_rollback`` when
            ``now`` precedes the lease heartbeat (C.14).
    """
    if recovery_count >= MAX_OWNER_RESUMES:
        raise AMLError("operation_conflict", "The owner has used its counted resume allowances")
    if admitted_epochs != current_epochs:
        raise AMLError("operation_conflict", "Explicit resume changed privacy authority")
    if not 1 <= allowance_ms <= MAX_RESUME_ALLOWANCE_MS:
        raise AMLError("invalid_request", "An explicit resume grants at most 1,740 seconds")
    if now < heartbeat_at:
        raise AMLError("clock_rollback", "Explicit resume clock regressed")
    return ResumeAllowance(now, now + allowance_ms, recovery_count + 1)


def reply_admissible(
    *, sent_attempt: int, current_attempt: int, deadline_at: int, arrived_at: int,
) -> bool:
    """Whether a provider reply may be stored and settled (C.13, INV:E07, I08).

    A reply belongs to the owner attempt that reserved its send and to that
    attempt's deadline. A late reply, after its allowance or after an explicit
    resume moved the owner to a new attempt, is neither stored nor settled; its
    reserved send stays spent.
    """
    return sent_attempt == current_attempt and arrived_at <= deadline_at


# ---------------------------------------------------------------- provider pause


def provider_failure(
    owner: Mapping[str, Any], failed: Mapping[str, Any], kind: ProviderFailureKind, *,
    key_refused: bool = False,
) -> ProviderFailureV1 | None:
    """Select the owner's current provider pause from its proved Runs (ingest-Q17).

    ``failed`` is the Run that wrote the owner's ``result_index:provider_failure``:
    the failed label child for ``source_labels``, the owner itself for
    ``extract``, ``embedding`` and ``judge``. A label call whose key the
    provider refused leaves its child running and the owner indexes the pause
    (``key_refused``; LABELS-RESUME): ``failed`` is then that running child.

    Args:
        owner: The owner's current Run row.
        failed: The failed Run's current row.
        kind: The index's call kind.
        key_refused: The label child still runs after a refused key.

    Returns:
        The content-free locator, or None when the owner is not paused on a
        provider failure (it completed, or a resume already moved it on).

    Raises:
        AMLError: ``binding_mismatch`` when the index names a Run that is not
            this owner's failed stage.
    """
    if (owner["status"], owner["failure_code"]) != ("manual_attention", "provider_error"):
        return None
    if kind in _OWNER_STAGES:
        if failed["object_id"] != owner["object_id"]:
            raise AMLError("binding_mismatch", "An owner-run stage failed in another Run")
    elif failed["object_id"] == owner["object_id"] or failed["parent_run_id"] != (
        owner["object_id"]
    ) or not (
        (failed["status"], failed["failure_code"]) == ("running", None) if key_refused else
        failed["status"] == "failed" and failed["failure_code"] in _LABEL_CHILD_FAILURES
    ):
        raise AMLError("binding_mismatch", "A provider pause names no failed child of its owner")
    return ProviderFailureV1(
        owner_run_id=owner["object_id"], child_run_id=failed["object_id"],
        owner_attempt=owner["attempt"], child_attempt=failed["attempt"], kind=kind,
    )


def require_selected(
    selection: ProviderFailureV1 | None, request: ResumeRequestV1,
) -> ProviderFailureV1:
    """Compare all four observed locators before the original proof is consumed.

    Raises:
        AMLError: ``operation_conflict`` when no pause is current or any locator changed.
    """
    if selection is None or (
        selection.owner_run_id, selection.child_run_id,
        selection.owner_attempt, selection.child_attempt,
    ) != (
        request.owner_run_id, request.child_run_id, request.owner_attempt, request.child_attempt,
    ):
        raise AMLError("operation_conflict", "The observed provider pause is no longer current")
    return selection


# ---------------------------------------------------------------- per position


type PositionAction = Literal[
    "settled", "rebuild_and_commit", "evidence_invalidated", "round_2", "target_changed",
    "reuse_output", "resend", "await_allowance", "action_rules",
]
"""What cold replay does with one judge position (C.13, C.3, C.11.1)."""

type ChildStatus = Literal["none", "queued", "running", "succeeded", "failed"]


@dataclass(frozen=True, slots=True)
class PositionState:
    """The content-free durable state of one judge position at cold replay.

    Attributes:
        candidate_status: The position's Candidate status (``pending`` or terminal).
        committed: Whether the position's create-once commit journal exists.
        child_status: The latest judge child's status for this round, ``none``
            when no child was created.
        usable_support_groups: Own-support groups still usable after the
            rebuild dropped erased or invalidated ones (C.3).
        review_dependency_erased: Whether an erased dependency invalidated the
            recorded review (C.13's round-2 route).
        round_2_available: Whether the position may still take round 2.
        reusable_accepted_output: Whether an unscrubbed stored accepted output
            matches the child's ``OutputIdentity`` (C.8; never an unavailable one).
        allowance_left: Whether the owner's original allowance admits a send.
    """

    candidate_status: Literal["pending", "published", "unchanged", "rejected", "expired"]
    committed: bool
    child_status: ChildStatus
    usable_support_groups: int
    review_dependency_erased: bool = False
    round_2_available: bool = False
    reusable_accepted_output: bool = False
    allowance_left: bool = True


def classify_position(state: PositionState) -> PositionAction:
    """Classify one judge position for cold replay (C.13 per position; C.3; C.11.1).

    A terminal or committed position is settled. A succeeded child is rebuilt
    from its recorded snapshot, digest-verified, resolved and committed behind
    the fence; the rebuild first drops unusable own-support groups, and none
    left is ``evidence_invalidated``, which the erased-review route never
    overrides. That route takes round 2 while available, else
    ``target_changed``. A running or failed child first reuses an unscrubbed
    stored accepted output; only otherwise does it resend within the original
    allowance (the unknown earlier attempt stays charged), and with no
    allowance left it waits for the owner's budget stop. A position with no
    child is not a recovery dispatch: an absent verdict is no permission to
    send, so the v1.2 action and snapshot rules (S4a, S5b) decide it.
    """
    if state.committed or state.candidate_status != "pending":
        return "settled"
    if state.child_status == "succeeded":
        if state.usable_support_groups == 0:
            return "evidence_invalidated"
        if state.review_dependency_erased:
            return "round_2" if state.round_2_available else "target_changed"
        return "rebuild_and_commit"
    if state.child_status == "none":
        return "action_rules"
    if state.reusable_accepted_output:
        return "reuse_output"
    return "resend" if state.allowance_left else "await_allowance"


# ---------------------------------------------------------------- cold classification


type LeaseState = Literal["preparing", "held", "released", "cancelled"]
type RunStatus = Literal["queued", "running", "succeeded", "failed", "manual_attention"]
type ColdAction = Literal["settle", "release", "adopt", "quarantine"]


@dataclass(frozen=True, slots=True)
class ColdOwner:
    """The content-free durable state of one scope's owner at boot step 14.

    Attributes:
        scope_id: The scope.
        lease_state: The ``lease:finalize:{s}`` state.
        lease_owner_run_id: The owner the lease names.
        lease_writer_epoch: The lease's Writer epoch.
        owner_run_id: The owner Run found, or None when it is missing.
        owner_status: The owner Run's status.
        owner_writer_epoch: The owner Run's Writer epoch.
        owner_scope_id: The owner Run's scope.
        current_writer_epoch: This boot's Writer epoch.
        children: ``(status, writer_epoch)`` of each judge and publish child.
        history_proved: Whether the released history validated (finalized
            marker, frozen input digest and baseline) for a released cycle.
    """

    scope_id: str
    lease_state: LeaseState
    lease_owner_run_id: str
    lease_writer_epoch: int
    owner_run_id: str | None
    owner_status: RunStatus | None
    owner_writer_epoch: int | None
    owner_scope_id: str | None
    current_writer_epoch: int
    children: Sequence[tuple[RunStatus, int]] = ()
    history_proved: bool = False


@dataclass(frozen=True, slots=True)
class ColdDisposition:
    """What boot step 14 does with one scope; ``error_code`` only for a quarantine."""

    action: ColdAction
    error_code: Literal["binding_mismatch"] | None = None


_TERMINAL: Final[frozenset[RunStatus]] = frozenset({"succeeded", "failed"})
_LIVE: Final[frozenset[RunStatus]] = frozenset({"queued", "running", "manual_attention"})


def classify_cold_owner(state: ColdOwner) -> ColdDisposition:
    """Classify one scope's owner after a restart (C.13 R13, PD:P1).

    A released cycle with its succeeded owner, ended children and proved history
    is a legal state change: settle. A released cycle whose owner still runs is
    a release that died between its lease CAS and its last transaction, the
    owner's success (``owner_lease.release_finalize``, C.2a): boot finishes it
    from its proofs (``release``). A cancelled cycle settles once its
    children ended; its owner did not succeed. A preparing or held cycle is a
    recoverable interruption: the same owner is adopted under the new Writer
    epoch, whether it still ran, was paused or reached its bookkeeping before
    release. Anything that cannot be proven quarantines this scope only.

    The epochs follow ``publication.finalize_authority``: every child's epoch
    is at most the owner's, which is at most the current Writer's.
    """
    if state.owner_run_id is None or state.owner_status is None or (
        state.owner_writer_epoch is None
    ):
        return _quarantine()
    if state.owner_run_id != state.lease_owner_run_id or state.owner_scope_id != state.scope_id:
        return _quarantine()
    if not state.owner_writer_epoch <= state.current_writer_epoch or not (
        state.lease_writer_epoch <= state.current_writer_epoch
    ) or any(epoch > state.owner_writer_epoch for _, epoch in state.children):
        return _quarantine()
    ended = all(status in _TERMINAL for status, _ in state.children)
    if state.lease_state == "released":
        if state.owner_status == "succeeded" and ended and state.history_proved:
            return ColdDisposition("settle")
        if state.owner_status == "running" and ended:
            return ColdDisposition("release")
        return _quarantine()
    if state.lease_state == "cancelled":
        if state.owner_status != "succeeded" and ended:
            return ColdDisposition("settle")
        return _quarantine()
    if state.owner_status in _LIVE or state.owner_status == "succeeded":
        return ColdDisposition("adopt")
    return _quarantine()


def _quarantine() -> ColdDisposition:
    return ColdDisposition("quarantine", "binding_mismatch")


@dataclass(frozen=True, slots=True)
class OwnerBudgetImage:
    """The owner budget fields cold adoption must keep (C.13, ``finalize_adoption``)."""

    owner_run_id: str
    accepted_prefix_commit_seq: int
    deadline_at: int
    recovery_count: int
    erase_epoch: int
    consent_epoch: int


def require_adoption_preserves(before: OwnerBudgetImage, after: OwnerBudgetImage) -> None:
    """Refuse an adoption that changed the owner, prefix, deadline or resume count.

    Adoption keeps the same owner identity, frozen prefix, original deadline and
    recorded outputs; it never extends the deadline or counts as a resume.

    Raises:
        AMLError: ``binding_mismatch`` for any change.
    """
    if before != after:
        raise AMLError("binding_mismatch", "Cold adoption changed its owner's frozen budget")


# ---------------------------------------------------------------- zero-effect proof


class EffectProbe(Protocol):
    """The bounded reads of the zero-effect proof (C.13 R17).

    ``Writer.commit_journal`` and ``IndexedReader.run_effects`` provide the
    first two; ``commit_seq_effects`` reads the main rows stamped with one
    ``commit_seq`` in the scope (``ix_*_commit`` indexes).
    """

    def commit_journal(self, run_id: str, journal_key: str) -> object | None:
        """The ``write_set`` journal the Run committed under this key, or None."""
        ...

    def run_effects(self, run_id: str, *, include_children: bool = True) -> bool:
        """Whether a main-backend row names this Run as its producer."""
        ...

    def commit_seq_effects(self, scope: ScopeV1, commit_seq: int) -> bool:
        """Whether a main-backend row in the scope carries this ``commit_seq``."""
        ...


def prove_zero_effect(
    probe: EffectProbe, *, scope: ScopeV1, run_id: str, journal_key: str,
    commit_seq: int | None, catalog_pending: bool,
) -> None:
    """Prove an interrupted publish, forget-plan or erase child wrote nothing (C.13 R17).

    Only after this returns may the child be retried or abandoned.

    Args:
        probe: The bounded reads.
        scope: The child's scope.
        run_id: The child Run.
        journal_key: Its create-once journal (or plan block) key.
        commit_seq: The ``commit_seq`` its transaction reserved, when recorded.
        catalog_pending: Whether the catalog step of its transaction is recorded
            as failed and awaiting repair.

    Raises:
        AMLError: ``catalog_repair_required`` when the catalog step is recorded
            as failed and a durable witness (its journal, a row naming the
            child as producer, or a row with its ``commit_seq``) shows the main
            transaction committed, including a journal-less one such as an
            erase batch: repair the catalog, never re-run the transaction.
            ``parity_mismatch`` for a witness without that record, and for a
            recorded catalog failure that no witness proves.
    """
    witnessed = probe.commit_journal(run_id, journal_key) is not None or probe.run_effects(
        run_id, include_children=True,
    ) or (commit_seq is not None and probe.commit_seq_effects(scope, commit_seq))
    if catalog_pending:
        if witnessed:
            raise AMLError(
                "catalog_repair_required", "The transaction committed; only its catalog failed",
            )
        raise AMLError("parity_mismatch", "A recorded catalog failure has no committed main rows")
    if witnessed:
        raise AMLError("parity_mismatch", "An interrupted child left a durable effect")


# ---------------------------------------------------------------- explicit recovery


class RecoveryStore(Protocol):
    """The Writer-side seam of explicit recovery (I15 hooks; bound in S2b's W4 merge).

    Every method runs under ``exclusive``; each commit re-validates the selected
    original proof and the pool's join proof inside its own Writer transaction,
    renews the exact lease there and applies ``explicit_resume_allowance``.
    """

    def exclusive(self) -> AbstractContextManager[object]:
        """The sole Writer's mutex."""
        ...

    def authorize(self, scope: ScopeV1) -> None:
        """Check the caller's Add grant for the scope; refuse before any private read."""
        ...

    def paused_failure(self, scope: ScopeV1) -> ProviderFailureV1 | None:
        """The scope owner's current provider pause from its stored proof, or None."""
        ...

    def commit_provider_resume(
        self, scope: ScopeV1, selection: ProviderFailureV1, joined: RetiredOwnerJoined,
    ) -> None:
        """Resume the paused owner on its next attempt in one Writer transaction."""
        ...

    def commit_refused_resume(
        self, scope: ScopeV1, request: CompletionResumeRequestV3, joined: RetiredOwnerJoined,
    ) -> None:
        """Replace the observed refusal and move the owner to its next attempt (ingest-Q12)."""
        ...

    def commit_quarantine_repair(
        self, scope: ScopeV1, request: CompletionResumeRequestV3, joined: RetiredOwnerJoined,
    ) -> None:
        """Lift the scope's quarantine around its owner's explicit resume (C.13)."""
        ...


class ExplicitRecovery:
    """Explicit recovery of a stopped owner over the scheduler's fences (``OwnerRecovery``).

    C41's bounded automatic controller reuses this exact proof seam.
    Observation is read-only. A resume selects the same original proof the
    caller observed, retires the owner's submission, proves its callbacks
    actually joined, commits the durable resume and only then releases the
    fence and resubmits the same owner, outside the Writer mutex. A refused
    commit keeps the fence; the owner stays stopped for a later explicit resume.
    """

    def __init__(self, scheduling: OwnerScheduling, store: RecoveryStore) -> None:
        self._scheduling = scheduling
        self._store = store

    def inspect_recovery(self, request: RecoveryRequestV1) -> RecoveryResponseV1:
        """Observe the scope owner's provider pause without reviving it."""
        scope = _user_scope(request.user_id)
        with self._store.exclusive():
            self._store.authorize(scope)
            return RecoveryResponseV1(failure=self._store.paused_failure(scope))

    def resume_recovery(self, request: ResumeRequestV1) -> ResumeResponseV1:
        """Resume the exact observed provider pause through its original owner."""
        scope = _user_scope(request.user_id)
        with self._store.exclusive():
            self._store.authorize(scope)
            selected = require_selected(self._store.paused_failure(scope), request)
            joined = self._join(BenchWork(scope.scope_id, selected.owner_run_id, "finalize"))
            self._store.commit_provider_resume(scope, selected, joined)
        # Outside the Writer mutex, as the scheduler's own resume submits (C.2a).
        self._scheduling.resume(joined)
        return ResumeResponseV1(selection=selected)

    def resume_refused(self, scope: ScopeV1, request: CompletionResumeRequestV3) -> None:
        """Resume the observed refused budget on the same owner and frozen prefix.

        The scheduler has authorized the caller and compared the observed budget.
        """
        with self._store.exclusive():
            joined = self._join(BenchWork(scope.scope_id, request.owner_run_id, "finalize"))
            self._store.commit_refused_resume(scope, request, joined)
        self._scheduling.resume(joined)

    def repair_quarantine(self, scope: ScopeV1, request: CompletionResumeRequestV3) -> None:
        """Repair the scope's observed quarantine through its owner's explicit resume (C.13).

        The quarantine left the owner, its lease and its budget as they stood; the
        resume proves that cycle, lifts the receipt and resubmits the same owner,
        which continues from its durable records. The scheduler has authorized the
        caller and compared the observed budget.
        """
        with self._store.exclusive():
            joined = self._join(BenchWork(scope.scope_id, request.owner_run_id, "finalize"))
            self._store.commit_quarantine_repair(scope, request, joined)
        self._scheduling.resume(joined)

    def _join(self, work: BenchWork) -> RetiredOwnerJoined:
        if not self._scheduling.retire(work):
            raise AMLError("extraction_busy", "The stopped owner cannot retire yet")
        return self._scheduling.retired_join(work)


def recovery_factory(
    store: Callable[[], RecoveryStore],
) -> Callable[[OwnerScheduling], ExplicitRecovery]:
    """Build the scheduler's ``RecoveryFactory`` over one Writer-side store."""

    def build(scheduling: OwnerScheduling) -> ExplicitRecovery:
        return ExplicitRecovery(scheduling, store())

    return build


def _user_scope(user_id: str) -> ScopeV1:
    return ScopeV1(scope_kind="user", scope_id=scope_id("user", user_id))
