"""The judge records' reader for Why (C.13, E.2.3; DM-06, read-Q19, W3-RT-11).

`StoredJudgeVerdicts` explains one accepted verdict from the three content-free
records the single Writer stored with the reply, read through the memory
read's guarded rows so the delivery recheck covers them:

- the ``result_index:judge`` that admitted the version: its kind, the
  ``reason_readable`` flag, whether it borrowed a recorded verdict and its
  resolved target map (ids and closed codes, written with the judgment);
- the accepted member output (``llm_output`` holding one ``JudgeV8``): the
  support verdict and the reason, the only verdict text Why may show;
- the call's ``decision_set:judge_call`` manifest: the whole-call summary
  (any tombstone issued, every input Evidence span, every input version).

It never loads a neighbour snapshot, a proposal, an extraction reply or a
tombstone, and rebuilds nothing from a call's inputs.
"""

from __future__ import annotations

from typing import Any

from aml.core.artifacts import parse_artifact_body
from aml.core.controls import CounterV1, parse_control_value
from aml.core.errors import AMLError
from aml.core.identity import control_id
from aml.core.inference import StoredInferenceOutput
from aml.core.judge_output import JudgeV8
from aml.core.judge_records import JudgeCallManifestV1, JudgeResultIndexV8
from aml.publication.history import MemoryRows
from aml.publication.why import JudgeCallSummary, JudgeVerdictRecord
from aml.store.metadata import StrictModel

__all__ = ["StoredJudgeVerdicts"]


def _artifact(rows: MemoryRows, reference: str, kind: str) -> tuple[dict[str, Any], StrictModel]:
    """Read one artifact of the read's exact scope through the guarded rows and parse it."""
    raw = rows.row("artifact", reference)
    if raw is None:
        raise AMLError("reference_missing", "A judge record is missing")
    if raw.get("artifact_kind") != kind:
        raise AMLError("binding_mismatch", "A judge record has another artifact kind")
    return raw, parse_artifact_body(kind, raw["body"], raw["schema_version"])


def _erased_in_scope(rows: MemoryRows) -> bool:
    """Whether an erasure advanced the read scope's ``erase_epoch`` (D.1, D.7.5).

    Only an erasure deletes a judge result, and every privacy erase advances
    ``erase_epoch`` in its first transaction, before any judge record leaves
    (C.13, INV P06). The counter is read through the guarded rows, so the
    delivery recheck covers it. A scope it never advanced deleted no result.
    """
    key = f"erase_epoch:{rows.scope.scope_id}"
    raw = rows.row("control", control_id(key))
    if raw is None:
        return False
    value = parse_control_value(key, raw["value"])
    return isinstance(value, CounterV1) and value.seq > 0


class StoredJudgeVerdicts:
    """Read accepted judge verdicts from the stored C.13 records (the `JudgeVerdicts` seam)."""

    def verdict(self, rows: MemoryRows, verdict_ref: str) -> JudgeVerdictRecord | None:
        """Read one accepted verdict, its member output and its call summary.

        Args:
            rows (MemoryRows): The read's guarded rows; every record is recorded
                for the delivery recheck.
            verdict_ref (str): The ``result_index:judge`` that admitted a version,
                or that a position's frozen item names.

        Returns:
            JudgeVerdictRecord | None: The verdict's readable parts, or None
                when the result is a valid ``unavailable`` one (C.8 #4/#6),
                which names a call but no verdict, or when the
                result is gone from a scope an erasure advanced: a privacy
                erase deleted it with its call (C.13). ``slot0`` is None
                when slot 0 was bound mechanically; ``relations`` holds the
                issued committed targets only (siblings have no memory id and
                tombstone verdicts are never recorded in the map).

        Raises:
            AMLError: ``reference_missing`` when a record is gone, other than
                a result in a scope an erasure advanced; ``binding_mismatch``
                when a record is not an accepted judge verdict, or the records
                disagree.
        """
        if rows.row("artifact", verdict_ref) is None and _erased_in_scope(rows):
            return None  # Deleted with its call by a privacy erase (C.13, INV P06).
        raw, result = _artifact(rows, verdict_ref, "result_index")
        if not isinstance(result, JudgeResultIndexV8):
            raise AMLError("binding_mismatch", "A verdict reference is not a judge result")
        _, call = _artifact(rows, result.call_ref, "decision_set")
        if not isinstance(call, JudgeCallManifestV1) or (
            call.binding_digest != result.binding_digest
        ):
            raise AMLError("binding_mismatch", "A judge result names another call")
        if result.kind == "unavailable":
            return None
        if result.output_ref is None or result.resolved is None:
            raise AMLError("binding_mismatch", "A judged version names no accepted verdict")
        _, stored = _artifact(rows, result.output_ref, "llm_output")
        if not (
            isinstance(stored, StoredInferenceOutput) and stored.purpose == "judge"
            and isinstance(stored.output, JudgeV8)
        ):
            raise AMLError("binding_mismatch", "A judge result names no member verdict")
        slot0 = next((slot for slot in result.resolved.slots if slot.slot == 0), None)
        return JudgeVerdictRecord(
            verdict_ref=verdict_ref, judged_at=int(raw["created_at"]),
            replayed=result.borrowed_from is not None, support=stored.output.support,
            slot0=None if slot0 is None or slot0.verdict == "mechanical" else slot0.verdict,
            relations=tuple(
                (relation.memory_id, relation.revision_id, relation.relation)
                for relation in result.resolved.relations
            ),
            reason=stored.output.reason, reason_readable=result.reason_readable,
            call=JudgeCallSummary(
                call.tombstone_issued, tuple(call.evidence_spans), tuple(call.revision_ids),
            ),
        )
