"""The ``fence_1`` finding for a held finalize owner after a privacy change: the controlled rebase.

A privacy erase advances the scope's ``erase_epoch``; a batch that deletes
a tombstone advances ``tombstone_seq`` with it; a consent withdrawal
advances ``consent_epoch``. An owner holding the scope's finalize lease keeps the
epochs it was admitted under, so every Writer action it takes is then refused
``erase_fenced`` or ``consent_fenced`` (``Writer.check_ticket``), and the read model
has dropped its baseline root, which held the erased content.
The ``fence_1`` finding makes that change a target change for what the owner reviewed before
it instead: a member goes to the next review round, up to ``MAX_ROUND``;
only a stale review in the last round settles ``target_changed``.

`rebase_finalize_privacy` is the one transition that lets such an owner act
again; each invocation of an owner takes it when it is due (`held_ticket`).
Under the Writer mutex it proves the actual held cycle (the identity,
admission, lease, heartbeat and ordinary workspace and scope fences of
``Writer.read_finalize_fences``, at the scope's current epochs), the grant
and the invocation's cancellation, and that its epochs only advanced: the finding reads
an ``erase_epoch`` and a ``consent_epoch`` change alike, while a pending privacy
owner, a maintenance or erasure fence, a tomb or a quarantine still refuse. An owner it
reached before both its first review and its settlement transaction has
nothing reviewed or settled to keep, while its stored extraction contexts
were rendered at the old pin and no longer reconstruct (``rebased_outcome``):
the rebase cancels its cycle instead (during extraction or review, a
cancellation), and the next owner reprocesses its requests
(`_cancel_unsettled`). No row cites anything that owner stored, so its replies,
proposals and embedding batches, which hold text and vectors the erase may
have removed, leave first, in transactions of their own before the
cancellation (`delete_record_groups`): an interruption leaves
the owner held and behind, and its next invocation cancels it again. Every
other owner, one that has reviewed or whose settlement transaction committed,
it moves. It then

1. CASes the lease once, naming the scope's view of the current root as its
   baseline;
2. moves the owner Run and its budget to the current epochs and ends the
   owner's judge children admitted before them (``failed``, ``erase_fenced``):
   they can never act again, so a late reply is neither stored nor settled;
3. retains that view as the owner's baseline and republishes the held barrier;
4. issues a new ticket: every ticket issued before is ``lease_lost``, and every
   write ticket minted under the old epochs stays ``erase_fenced``.

Nothing reviewed before the transition commits. `privacy_resume` tells the
owner's rounds what the ``fence_1`` finding and recovery leave of each round-1 member, its own
and the pending rows of earlier owners it re-entered, from the epoch its judge
children were admitted under; the current-Evidence checks of the review
(``judge_input.build_member_input``) and of the commit (``p_span``) apply
at the new pin as everywhere else.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final, cast

from quotemem.core.commit_plan import CommitJournalV1
from quotemem.core.controls import (
    CancelledFinalizeLeaseV1,
    ControlRowV1,
    FinalizeBudgetV2,
    HeldFinalizeLeaseV1,
)
from quotemem.core.entities import PositionKey
from quotemem.core.errors import QuoteMemError
from quotemem.core.judge_records import is_judgment_target, judgment_targets
from quotemem.core.query import ScopeV1
from quotemem.core.review_rounds import MAX_ROUND, RoundNo
from quotemem.store.finalize_ticket import FinalizeLeaseTicket

if TYPE_CHECKING:
    from quotemem.providers.auth import PrincipalGrant
    from quotemem.store.domain import CandidateRowV2
    from quotemem.store.privacy_compensation import RecordGroup
    from quotemem.store.read_model import StoreReadModel
    from quotemem.store.writer import RowWrite, Writer
    from quotemem.store.writer_mutex import ProgressCancellation

# pyright: reportPrivateUsage=false

_ACTIVE: Final = frozenset({"queued", "running"})
_RECORD_BATCH: Final = 512
"""Stored rows one delete-only transaction takes (`delete_record_groups`), well within
``MAX_TX_ROWS``."""


def delete_record_groups(writer: Writer, groups: Sequence[RecordGroup]) -> int:
    """Delete stored record groups in bounded delete-only transactions.

    The caller holds the Writer's write scope and derives the groups from the
    stored rows, so an interruption leaves exactly the records a later call
    derives again. A group is never split: each transaction takes whole groups
    up to ``_RECORD_BATCH`` rows. Artifacts change no scope visibility, so no
    counter advances.

    Args:
        writer: The single Writer, inside its write scope.
        groups: The groups to delete, each ``(kind, stored row)`` pairs.

    Returns:
        How many rows were deleted.

    Raises:
        QuoteMemError: ``budget_exceeded`` for a group beyond ``MAX_TX_ROWS`` RowOps;
            ``parity_mismatch`` for a row that disappeared before its commit.
    """
    from quotemem.store.writer import MAX_TX_ROWS

    deleted = 0
    batch: list[tuple[str, dict[str, Any]]] = []

    def flush() -> None:
        plan = writer._ws._writer.prepare_upsert_rows([], overwrite=True, catalog_mode="merge")
        plan.entity_ops.setdefault("main", [])
        writer.plan_deletes(plan, batch)
        if len(plan.entity_ops["main"]) > MAX_TX_ROWS:
            raise QuoteMemError("budget_exceeded",
                                "A record deletion exceeds its transaction budget")
        writer.write_plan(plan)

    for group in groups:
        if batch and len(batch) + len(group) > _RECORD_BATCH:
            flush()
            deleted += len(batch)
            batch = []
        batch.extend(group)
    if batch:
        flush()
        deleted += len(batch)
    if deleted:
        writer.release_erased_caches()
    return deleted


def _judge_target(position_key: PositionKey, round_no: int) -> str:
    """The target id of a position's judge child in one round (``judgment_target``)."""
    owner_run_id, unit_no, position = position_key
    return f"{owner_run_id}.{unit_no}.{position}.r{round_no}"


def _judge_child(child: Mapping[str, Any], owner_run_id: str) -> bool:
    """Whether a child of the owner is one of its judge children (``evaluate:judgment``).

    Its target names the reviewed position with that position's original owner:
    the reviewing owner for its own members, an earlier one for a pending row it
    re-entered, and a confirmation vote's its vote (``judgment_target``).
    """
    return (child["run_kind"] == "evaluate" and child["parent_run_id"] == owner_run_id
            and is_judgment_target(str(child["target_id"])))


def _behind(child: Mapping[str, Any], epochs: tuple[int, int]) -> bool:
    """Whether a Run was admitted before the owner's ``(erase_epoch, consent_epoch)``."""
    return (child["admitted_erase_epoch"], child["admitted_consent_epoch"]) != epochs and (
        child["admitted_erase_epoch"] <= epochs[0] and child["admitted_consent_epoch"] <= epochs[1]
    )


def _children(writer: Writer, owner_run_id: str) -> Iterator[dict[str, Any]]:
    """Every child Run of the owner, in bounded pages."""
    from quotemem.store.indexed import IndexedReader

    reader = IndexedReader(writer.runtime.workspace, writer.entities, tenant_id=writer.tenant_id)
    after: str | None = None
    while page := reader.run_children(owner_run_id, after=after, limit=512):
        yield from page
        after = page[-1]["object_id"]


def privacy_rebase_scope(writer: Writer, owner_run_id: str) -> ScopeV1 | None:
    """The scope of a held finalize owner that a privacy erase left behind, else None.

    A content-free probe: a running held owner with a selected input whose
    scope's ``erase_epoch`` or ``consent_epoch`` advanced past its admitted one.
    `rebase_finalize_privacy` proves all of it again.
    """
    with writer.mutex:
        run = writer.get_row("run", owner_run_id)
        if run is None or run["run_kind"] != "finalize" or run["status"] != "running" or (
            run["input_ref"] is None or run["commit_marker"] is not None
        ):
            return None
        raw = writer.get_control(f"lease:finalize:{run['scope_id']}")
        value = None if raw is None else ControlRowV1.model_validate(raw).value
        if not isinstance(value, HeldFinalizeLeaseV1) or value.owner_run_id != owner_run_id:
            return None
        scope_id = run["scope_id"]
        epochs = (writer.get_counter(f"erase_epoch:{scope_id}"),
                  writer.get_counter(f"consent_epoch:{scope_id}"))
        if not _behind(run, epochs):
            return None
        return ScopeV1(scope_kind=run["scope_kind"], scope_id=scope_id)


def held_ticket(
    writer: Writer, publisher: StoreReadModel, owner_run_id: str, *,
    grant: Callable[[ScopeV1], PrincipalGrant], cancellation: ProgressCancellation,
    clock: Callable[[], int],
) -> FinalizeLeaseTicket:
    """The owner's issued ticket, after its privacy rebase when an erase left it behind.

    Args:
        writer: The single Writer.
        publisher: This Writer's registered read model.
        owner_run_id: The owner an invocation is about to run.
        grant: Resolves a fresh grant of the owner's scope, outside QuoteMem locks.
        cancellation: The invocation's cancellation.
        clock: The trusted millisecond action clock its ticket keeps.

    Returns:
        The owner's live ticket (``Writer.finalize_ticket``), or the one its
        rebase issued.
    """
    scope = privacy_rebase_scope(writer, owner_run_id)
    if scope is None:
        return writer.finalize_ticket(owner_run_id, clock=clock)
    return rebase_finalize_privacy(
        writer, publisher, owner_run_id, grant=grant(scope), cancellation=cancellation,
        clock=clock,
    )


def _ended_children(
    writer: Writer, owner_run_id: str, children: Sequence[Mapping[str, Any]],
    epochs: tuple[int, int], now: int,
) -> list[RowWrite]:
    """End the owner's judge children admitted before ``epochs``.

    Raises:
        QuoteMemError: ``erase_fenced`` for an unfinished non-judge child.
    """
    from quotemem.store.writer import RowWrite

    rows: list[RowWrite] = []
    for child in children:
        if child["status"] not in _ACTIVE or not _behind(child, epochs):
            continue
        if not _judge_child(child, owner_run_id):
            raise QuoteMemError("erase_fenced", "An unfinished non-judge child keeps the old epoch")
        ended = dict(child, status="failed", failure_code="erase_fenced", updated_at=now,
                     finished_at=now)
        rows.append(RowWrite("run", writer.parse_row("run", ended)))
    return rows


def _settled(writer: Writer, owner_run_id: str) -> bool:
    """Whether a transaction of the owner committed: settlement, rejection or component."""
    return any(
        isinstance(body, CommitJournalV1) and body.owner_run_id == owner_run_id
        for _reference, body in writer.find_logical_candidates(owner_run_id, "write_set")
    )


def _drop_cancelled_records(writer: Writer, owner_run_id: str) -> None:
    """Delete what an owner cancelled before its first review or commit stored.

    Nothing of it committed, so no candidate row cites its proposals, and no
    owner but itself reads its replies or embedding batches: each holds text or
    vectors of sources the privacy change may have erased, and none is ever
    read again. Its contexts and settlement record hold locators and codes only.
    """
    from quotemem.store.privacy_compensation import owner_extraction_records, record_groups

    delete_record_groups(writer, record_groups(writer, owner_extraction_records(
        writer, owner_run_id)))


def _cancel_unsettled(
    writer: Writer, publisher: StoreReadModel, run: Mapping[str, Any], lease: ControlRowV1,
    children: Sequence[Mapping[str, Any]], epochs: tuple[int, int], now: int,
) -> None:
    """Cancel the cycle of an owner the privacy change reached before it reviewed or settled.

    Nothing of it committed, so no candidate row of it exists ("uncommitted
    positions have no rows"). In one transaction under the Writer mutex the lease
    becomes ``cancelled`` (``hard_erase``) under the owner itself, the owner fails
    ``erase_fenced`` (``consent_fenced`` for a consent change alone) and every
    unfinished child fails ``erase_fenced``, so no late reply is stored or settled.
    The read model lifts the scope's barrier and the admission queue
    releases its running input. The next admission proves the cancellation
    settled (``owner_lease.cancelled_settled``) and reprocesses
    the requests, reusing an extraction output only by its unchanged digests.
    """
    from quotemem.store.writer import RowWrite

    value = lease.value
    assert isinstance(value, HeldFinalizeLeaseV1)
    code = "erase_fenced" if run["admitted_erase_epoch"] < epochs[0] else "consent_fenced"
    rows = [RowWrite("run", writer.parse_row("run", dict(
        run, status="failed", failure_code=code, updated_at=now, finished_at=now,
    )))]
    rows += [RowWrite("run", writer.parse_row("run", dict(
        child, status="failed", failure_code="erase_fenced", updated_at=now, finished_at=now,
    ))) for child in children if child["status"] in _ACTIVE]
    writer._parse_rows(rows, None)
    cancelled = CancelledFinalizeLeaseV1(
        state="cancelled", lease_generation=value.lease_generation,
        owner_run_id=value.owner_run_id, writer_epoch=value.writer_epoch,
        heartbeat_at=value.heartbeat_at, cancelled_from="held", cancelled_at=now,
        cancel_reason="hard_erase",
    )
    writer._write_finalize_lease_cas(lease, writer.materialize_control(
        lease.key, cancelled.model_dump(mode="json"), lease.version + 1, now, run["object_id"],
    ))
    _, plan = writer.plan_rows(rows, None)
    writer.write_plan(plan)
    publisher.publish_finalize_cancelled(
        writer, ScopeV1(scope_kind=run["scope_kind"], scope_id=run["scope_id"]),
        value.owner_run_id, value.lease_generation, value.writer_epoch,
    )
    writer.add_admission.owner_released(run["scope_id"])


def rebase_finalize_privacy(
    writer: Writer, publisher: StoreReadModel, owner_run_id: str, *,
    grant: PrincipalGrant, cancellation: ProgressCancellation, clock: Callable[[], int],
) -> FinalizeLeaseTicket:
    """Move a held finalize owner to its scope's current epochs.

    Args:
        writer: The single Writer; called outside its mutex.
        publisher: This Writer's registered read model.
        owner_run_id: The held owner whose admitted epochs are behind.
        grant: The owner principal's fresh grant.
        cancellation: The owner invocation's cancellation.
        clock: The trusted millisecond action clock the new ticket keeps.

    Returns:
        The owner's newly issued held ticket.

    Raises:
        QuoteMemError: ``lease_lost`` for a cancelled invocation, a lost or expired
            lease or an owner that is not a running held finalize with a
            selected input; ``clock_rollback`` for a clock behind the lease;
            ``erase_fenced`` when no epoch advanced (or one went back), when
            another of its children is unfinished, or after cancelling an owner
            the change reached before its first review and settlement;
            ``binding_mismatch`` for another publisher or a budget of another
            owner; ``service_unhealthy`` for a root that lags storage; and every
            ordinary Writer fence (quarantine, tomb, pending privacy owner,
            maintenance or erasure fence) and grant refusal.
    """
    from quotemem.ingestion.authorization import check_add_grant
    from quotemem.store.read_model import StoreReadModel
    from quotemem.store.writer import RowWrite

    if cancellation.cancelled:
        raise QuoteMemError("lease_lost", "The owner invocation was cancelled")
    cancelled = False
    with writer.write_scope():
        if StoreReadModel.get(writer) is not publisher:
            raise QuoteMemError("binding_mismatch",
                                "A privacy rebase requires its registered publisher")
        run = writer.get_row("run", owner_run_id)
        if run is None:
            raise QuoteMemError("reference_missing", "The finalize owner does not exist")
        writer.parse_row("run", run)
        scope = ScopeV1(scope_kind=run["scope_kind"], scope_id=run["scope_id"])
        raw = writer.get_control(f"lease:finalize:{scope.scope_id}")
        held = None if raw is None else ControlRowV1.model_validate(raw).value
        if not isinstance(held, HeldFinalizeLeaseV1) or held.owner_run_id != owner_run_id or (
            run["run_kind"] != "finalize" or run["status"] != "running"
            or run["input_ref"] is None or run["commit_marker"] is not None
            or run["result_ref"] is not None
        ):
            raise QuoteMemError("lease_lost", "A privacy rebase requires a running held owner")
        erase = writer.get_counter(f"erase_epoch:{scope.scope_id}")
        consent = writer.get_counter(f"consent_epoch:{scope.scope_id}")
        if not _behind(run, (erase, consent)):
            raise QuoteMemError("erase_fenced", "No privacy change advanced the owner's scope")
        issued = FinalizeLeaseTicket(
            scope.scope_id, held.lease_generation, owner_run_id, held.writer_epoch, "held",
        )
        # Registration replaces the owner's earlier ticket, which can then no
        # longer act; the write tickets minted under the old epochs stay fenced.
        writer._register_finalize_ticket(issued, clock)
        try:
            run, lease, _ = writer._finalize_identity(issued, rebased=(erase, consent))
            assert isinstance(lease.value, HeldFinalizeLeaseV1)
            now = writer._job_timestamp(clock())
            writer._check_finalize_heartbeat(lease, now)
            if now < lease.value.heartbeat_at:
                raise QuoteMemError("clock_rollback", "A privacy rebase clock regressed")
            check_add_grant(grant, scope, now, run["principal_id"])
            if cancellation.cancelled:
                raise QuoteMemError("lease_lost", "The owner invocation was cancelled")
            children = list(_children(writer, owner_run_id))
            if not any(_judge_child(child, owner_run_id) for child in children) and not (
                _settled(writer, owner_run_id)
            ):
                # Its records leave before the cancellation: an interruption
                # leaves the owner held and behind, so its next invocation comes here.
                _drop_cancelled_records(writer, owner_run_id)
                # Raised after the write scope ends: an error inside it after a write
                # would fail the read model closed.
                _cancel_unsettled(writer, publisher, run, lease, children, (erase, consent), now)
                cancelled = True
            else:
                ended = _ended_children(writer, owner_run_id, children, (erase, consent), now)
                raw_budget = writer.get_control(f"finalize_budget:{scope.scope_id}")
                budget = None if raw_budget is None else ControlRowV1.model_validate(raw_budget)
                if budget is None or not isinstance(budget.value, FinalizeBudgetV2) or (
                    budget.value.owner_run_id != owner_run_id
                ):
                    raise QuoteMemError("binding_mismatch",
                                        "A privacy rebase lost its owner budget")
                moved = dict(run, admitted_erase_epoch=erase, admitted_consent_epoch=consent,
                             updated_at=now)
                rows = [
                    RowWrite("run", writer.parse_row("run", moved)),
                    RowWrite("control", writer.materialize_control(
                        budget.key, budget.value.model_copy(update={
                            "erase_epoch": erase, "consent_epoch": consent,
                        }),
                        budget.version + 1, now, owner_run_id,
                    )),
                    *ended,
                ]
                # The Run still names its old epochs until this batch lands, so the
                # batch is owner-less; `_finalize_identity` admitted the scope at
                # the new ones under the same mutex. It is validated before the
                # lease CAS and written after it: an interrupted rebase leaves the
                # owner behind, and its next invocation rebases it again.
                writer._parse_rows(rows, None)
                view = publisher.baseline_view(writer, scope)
                value = lease.value.model_copy(update={
                    "heartbeat_at": now, "baseline_snapshot_generation": view.generation,
                    "baseline_root_identity": view.root_identity,
                })
                writer._write_finalize_lease_cas(lease, writer.materialize_control(
                    lease.key, value.model_dump(mode="json"), lease.version + 1, now, owner_run_id,
                ))
                _, plan = writer.plan_rows(rows, None)
                writer.write_plan(plan)
                publisher.rebase_finalize_baseline(
                    writer, scope, owner_run_id, issued.lease_generation, issued.writer_epoch,
                )
                publisher.publish_finalize_held(
                    writer, scope, owner_run_id, issued.lease_generation, issued.writer_epoch,
                )
                writer.read_finalize_fences(issued)
        except BaseException:
            writer._finalize_tickets.pop(owner_run_id, None)
            raise
        if cancelled:
            writer._finalize_tickets.pop(owner_run_id, None)
    if cancelled:
        raise QuoteMemError("erase_fenced", "A privacy change cancelled an owner before its first "
                            "review; the next owner reprocesses its requests")
    return issued


@dataclass(frozen=True, slots=True)
class PrivacyResumeV1:
    """What the ``fence_1`` finding and recovery leave of a rebased owner's round-1 members.

    A member an earlier invocation committed is in none of these (a
    committed candidate is settled).

    Attributes:
        round1: Its own members no review of which predates the rebase: round 1
            as usual.
        round2: Members with an earlier review to route through round 2, cause
            ``fence_1``; stale later reviews go on to the next settle round.
        invalidated: Each review admitted before the current privacy epochs,
            with its retained ``result_index:judge`` or None after erasure.
        rows: The stored row of each pending row of an earlier owner that it
            re-entered and routes here, in ``round2``;
            round 1 does not select them again.
    """

    round1: tuple[PositionKey, ...]
    round2: tuple[PositionKey, ...]
    invalidated: Mapping[tuple[PositionKey, RoundNo], str | None]
    rows: Mapping[PositionKey, CandidateRowV2] = field(
        default_factory=dict[PositionKey, "CandidateRowV2"])


def committed_positions(writer: Writer, owner_run_id: str) -> set[PositionKey]:
    """The positions a rejection or component transaction of the owner wrote.

    Of the owner's own members these are settled: an earlier invocation
    committed their final disposition in round 1 or 2. A re-entered row of an
    earlier owner is also listed once the owner touched it.
    """
    found: set[PositionKey] = set()
    for _reference, body in writer.find_logical_candidates(owner_run_id, "write_set"):
        if isinstance(body, CommitJournalV1) and body.owner_run_id == owner_run_id and (
            body.txn in ("rejection", "component")
        ):
            found.update(member.position_key for member in body.members)
    return found


def privacy_resume(
    writer: Writer, owner_run_id: str, members: Sequence[PositionKey],
    pending: Mapping[PositionKey, CandidateRowV2] | None = None,
) -> PrivacyResumeV1 | None:
    """Route a rebased owner's round-1 members by the epoch of their reviews.

    A judge child is admitted under its owner's epochs, so a child admitted
    under an older ``erase_epoch`` or ``consent_epoch`` than its owner's is a
    review the rebase left behind. A pending row of an earlier owner that the owner reviewed is
    routed like its own members, unless one of those reviews already settled
    it (its row holds that review's result, or a confirmation vote's).

    Args:
        writer: The single Writer.
        owner_run_id: The owner.
        members: Its round-1 members (``ExtractionOutcome.review``).
        pending: The scope's pending rows by position, which round 1 re-enters.

    Returns:
        The routing, or None for an owner none of whose reviews predates its epoch.
    """
    with writer.mutex:
        owner = writer.get_row("run", owner_run_id)
        if owner is None:
            raise QuoteMemError("lease_lost", "The finalize owner does not exist")
        epochs = (owner["admitted_erase_epoch"], owner["admitted_consent_epoch"])
        children = {
            child["target_id"]: child for child in _children(writer, owner_run_id)
            if _judge_child(child, owner_run_id)
        }
        if not any(_behind(child, epochs) for child in children.values()):
            return None
        committed = committed_positions(writer, owner_run_id)
    round1: list[PositionKey] = []
    round2: list[PositionKey] = []
    invalidated: dict[tuple[PositionKey, RoundNo], str | None] = {}

    def route(key: PositionKey) -> bool:
        first = children.get(_judge_target(key, 1))
        reviews = [(round_no, children.get(_judge_target(key, round_no)))
                   for round_no in range(1, MAX_ROUND + 1)]
        if not any(child is not None for number, child in reviews if number >= 2) and not (
            first is not None and _behind(first, epochs)
        ):
            return False
        round2.append(key)
        for number, child in reviews:
            if child is not None and _behind(child, epochs):
                # The erase may have deleted that review's result.
                result = child["result_ref"]
                invalidated[(key, cast(RoundNo, number))] = (
                    None if result is None or writer.get_row("artifact", result) is None
                    else result)
        return True

    for key in members:
        if key not in committed and not route(key):
            round1.append(key)
    rows: dict[PositionKey, CandidateRowV2] = {}
    for key, row in (pending or {}).items():
        reviews = [children[target] for target in judgment_targets(key) if target in children]
        if key[0] == owner_run_id or not reviews or row.judge_result_ref is not None and (
            row.judge_result_ref in {child["result_ref"] for child in reviews}
        ):
            continue
        if route(key):
            rows[key] = row
    return PrivacyResumeV1(tuple(round1), tuple(round2), invalidated, rows)


__all__ = [
    "PrivacyResumeV1",
    "committed_positions",
    "held_ticket",
    "privacy_rebase_scope",
    "privacy_resume",
    "rebase_finalize_privacy",
]
