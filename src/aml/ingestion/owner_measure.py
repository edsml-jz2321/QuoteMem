"""The production ``measure`` of ``units.plan_units`` (W3-RT-14; B.3 as amended, C.12).

C.12's chunked settlement replaces B.3's per-unit 128-fact proof: settlement is
split across positions within ``MAX_TX_ROWS``, so only two transactions must
still fit for a unit, and both are measured on the real Writer, never estimated:

1. ``artifact_rowops``: the unit's largest pre-call Artifact, its
   ``decision_set`` extraction context, the one the stage stores
   (``ExtractionOwner.unit_context``: ``build_context`` with the same preceding
   parts and the owner's previous batch) and sized by
   ``Writer.logical_artifact_rowops`` (sizing rows, nothing stored).
2. ``business_rowops``: the unit's worst settlement position, S5b's
   ``worst_settled_position`` (a forgotten restatement citing every part and
   withholding a span in each), sized as one settlement chunk by
   ``size_settled_position`` (``Writer.commit_rowops`` in sizing mode; nothing
   is written).

The measurement depends only on the unit's parts, so counts are monotone as
parts are appended (``plan_units``). ``facts`` is not a factor: positions are
chunked (C.12). A unit whose worst position does not fit is refused before any
call (``budget_exceeded``), never at settlement.
"""

from __future__ import annotations

from aml.core.errors import AMLError
from aml.core.identity import integrity_digest
from aml.ingestion.extraction_stage import ExtractionOwner
from aml.ingestion.units import StorageBudget, Unit
from aml.store.memory_commit import (
    proposal_groups,
    size_settled_position,
    worst_settled_position,
)
from aml.store.writer import MAX_TX_ROWS

_WORST_POSITION = 511
"""The largest position number (A.3.4); the worst position takes it."""


def measure_unit(owner: ExtractionOwner, unit: Unit, facts: int) -> StorageBudget:
    """Measure the unit's largest pre-call Artifact and its worst position (W3-RT-14).

    Args:
        owner: The Add owner, at its pinned view.
        unit: The candidate unit.
        facts: The policy's fact maximum; chunking makes it no factor (C.12).

    Returns:
        StorageBudget: Both measured RowOp counts and the measured plan's identity.

    Raises:
        AMLError: ``budget_exceeded`` when the worst position's rows alone exceed
            ``MAX_TX_ROWS``; the Writer's other refusals.
    """
    del facts
    writer, ticket = owner.writer, owner.ticket
    built = owner.unit_context(unit)
    now = owner.clock()
    if ticket.scope_id is None or ticket.run_id is None:
        raise AMLError("binding_mismatch", "Unit measurement requires the scoped owner ticket")
    artifact = 0 if built is None else writer.logical_artifact_rowops(
        ticket, "decision_set", built.record, created_at=now, security_level="internal",
    )
    worst = worst_settled_position(
        unit.parts, owner_run_id=owner.owner_run_id, unit_no=unit.unit_no,
        position=_WORST_POSITION,
    )
    business = size_settled_position(
        writer, ticket, worst, owner_run_id=owner.owner_run_id, updated_at=now,
        groups_of=proposal_groups(writer),
    )
    if business > MAX_TX_ROWS:
        raise AMLError("budget_exceeded", "One position's rows exceed the transaction maximum")
    return StorageBudget(artifact, business, integrity_digest(
        ["owner_measure", artifact, business, len(worst.evidence_ids)],
    ))
