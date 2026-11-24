"""The provider-pause producer: the owner's ``result_index:provider_failure``.

A provider failure of a stage the owner calls itself (extraction, statement
embedding or the judge round) pauses the owner: the owner Run itself writes a
``SessionProviderFailureResultIndexV1`` bound to its selected input, naming the
failed stage, with the failure's bounded HTTP diagnostics when it has them,
and moves to ``manual_attention`` with ``failure_code="provider_error"``. The
lease and the budget stay: the scheduler retires the stopped owner
(``_read_stop_cause`` reads ``provider``) and only the explicit resume moves it
on (``owner_recovery.RecoveryStore``).

The seam is narrow on purpose. The producer writes two things and the recovery
reads them back: ``paused_index`` returns the index of the owner's latest pause
(its ``paused_at``, never the listing order of content-addressed indices), and
``owner_recovery.provider_failure`` decides from the owner Run alone whether the
pause is still current. A label child's pause
(``source_labels``) is written by its own child Run; its owner stop goes
through ``pause_owner`` without an error, so the index is the child's and the
owner only changes state.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from quotemem.core.artifacts import SessionProviderFailureResultIndexV1
from quotemem.core.errors import QuoteMemError
from quotemem.core.identity import ArtifactId
from quotemem.core.provider_diagnostics import ProviderFailureDetailV1
from quotemem.providers.adapters import ProviderError
from quotemem.store.finalize_ticket import FinalizeLeaseTicket
from quotemem.store.writer import Writer

type OwnerStageKind = Literal["extract", "embedding", "judge"]
"""The stages the owner calls itself, whose provider failure the owner Run records."""

type PauseKind = Literal["extract", "embedding", "source_labels", "judge"]
"""Every stage whose provider failure pauses the owner; a child stage holds its own index."""

PROVIDER_PAUSE_FAILURE_CODE: Literal["provider_error"] = "provider_error"
"""The paused owner's failure code; the scheduler and the Add owner read exactly this."""


def pause_owner(
    writer: Writer, ticket: FinalizeLeaseTicket, error: ProviderError | None, *,
    kind: PauseKind = "extract", clock: Callable[[], int],
) -> ArtifactId | None:
    """Pause the held owner on a provider failure of one of its own stages.

    The index and the state change share the owner's lease precondition. The
    index is written first: a crash between the two leaves an index whose owner
    is still running, which ``owner_recovery.provider_failure`` ignores, and the
    retried invocation pauses again with a fresh index.

    Args:
        writer: The sole Writer.
        ticket: The held owner's issued cycle.
        error: The provider failure that ended a stage the owner calls itself;
            None for the label child (``source_labels``), whose failed child Run
            already wrote its own index. A label call whose key the provider
            refused passes its error: its child still runs, so the owner
            indexes the pause with the refusal's diagnostics.
        kind: The failed stage's call kind.
        clock: Trusted millisecond action clock.

    Returns:
        The owner's written index; None for a failed label child's pause.

    Raises:
        QuoteMemError: ``lease_lost`` for a lost or stopped cycle; ``binding_mismatch``
            for an owner without a selected input.
    """
    from quotemem.store.writer import RowWrite

    with writer.write_scope():
        run, _, write = writer.read_finalize_fences(ticket)
        if run["input_ref"] is None:
            raise QuoteMemError("binding_mismatch",
                                "A provider pause needs the owner's selected input")
        now = clock()
        reference: ArtifactId | None = None
        if kind != "source_labels" and error is None:
            raise QuoteMemError("binding_mismatch", "An owner stage's pause needs its failure")
        if error is not None:
            diagnostics = error.diagnostics
            reference = writer.put_logical_artifact(
                write, "result_index", SessionProviderFailureResultIndexV1(
                    input_ref=ArtifactId(run["input_ref"]), paused_at=now,
                    provider_failure=(diagnostics if isinstance(diagnostics,
                                                                ProviderFailureDetailV1)
                                      else None),
                    provider_call_kind=kind,
                ), created_at=now,
            )
        current = writer.get_row("run", ticket.owner_run_id)
        if current is None or current["status"] != "running":
            raise QuoteMemError("lease_lost", "A provider pause needs the running owner")
        paused = dict(
            current, status="manual_attention", failure_code=PROVIDER_PAUSE_FAILURE_CODE,
            updated_at=now, finished_at=None,
        )
        rows = [RowWrite("run", writer.parse_row("run", paused))]
        _, plan = writer.plan_rows(rows, None)
        writer.write_plan(plan)
        return reference


def paused_index(
    writer: Writer, owner_run_id: str,
) -> tuple[ArtifactId, SessionProviderFailureResultIndexV1] | None:
    """Read the owner's own latest ``result_index:provider_failure`` for its selected input.

    The reader side of the seam for ``RecoveryStore.paused_failure`` and the
    scheduler's V3 ``failure_cause``. Only an index bound to the owner's current
    ``input_ref`` counts, and of those the one with the latest ``paused_at``: an
    index's address is its content's digest, so the listing order says nothing of
    which pause came last, and an earlier stage's index (an extraction pause before a
    judge pause) would otherwise select another stage's recovery. Whether the pause
    is still current is decided from the owner Run (``owner_recovery.provider_failure``).

    Args:
        writer: The sole Writer.
        owner_run_id: The owner Run.

    Returns:
        The index and its address, or None. Indices of one millisecond (no real
        pause and resume fit in one) are ordered by address, so the choice is stable.
    """
    run = writer.get_row("run", owner_run_id)
    if run is None or run["input_ref"] is None:
        return None
    found: tuple[ArtifactId, SessionProviderFailureResultIndexV1] | None = None
    for reference, body in writer.find_logical_candidates(owner_run_id, "result_index"):
        if not isinstance(body, SessionProviderFailureResultIndexV1) or (
            body.input_ref != run["input_ref"]
        ):
            continue
        if found is None or (body.paused_at, reference) > (found[1].paused_at, found[0]):
            found = reference, body
    return found
