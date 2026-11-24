"""Prove loss of one admitted label source through receipt and eligibility metadata.

The caller proves the original label admission and current recovery authority.
This reader binds that attached input and proves a lost conjunct of its original
message set. It grants neither source access nor a replacement call budget.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

from quotemem.core.artifacts import SourceLabelsResultIndexV1
from quotemem.core.errors import QuoteMemError
from quotemem.core.identity import canonical_json
from quotemem.core.query import ScopeV1
from quotemem.core.run_inputs import SessionInventoryDurableRunInputV2, SourceLabelsRunInputV1
from quotemem.core.source_labels import (
    SourceLabelMessageBindingV1,
    SourceLabelsSourceInterruptionV1,
    SourceLabelsSourceLossV1,
)
from quotemem.ingestion.inventory import (
    SourceCoordinateV1,
    freeze_source_coordinates,
    load_request_coordinates,
)
from quotemem.ingestion.source_history import (
    cleanup_completion_sequence,
    resolve_erased_source_history,
    validate_erased_source_history,
)
from quotemem.store.artifact_read import read_logical_artifact
from quotemem.store.indexed import IndexedReader
from quotemem.store.metadata import StrictModel

if TYPE_CHECKING:
    from quotemem.store.writer import Writer

type Record = Callable[[str, dict[str, Any]], None]
_VISITING: ContextVar[frozenset[str]] = ContextVar("source_labels_source_loss", default=frozenset())


def _noop(_kind: str, _row: dict[str, Any]) -> None:
    pass


class _Read:
    """Retain physical metadata rows separately from actual Evidence projections."""

    def __init__(self, writer: Writer, run: Mapping[str, Any], record: Record) -> None:
        self.writer = writer
        self.run = run
        self.record = record
        self.scope = ScopeV1(scope_kind=run["scope_kind"], scope_id=run["scope_id"])
        self.index = IndexedReader(
            writer.runtime.workspace, writer.entities, tenant_id=writer.tenant_id,
        )
        self.receipts: dict[str, tuple[SourceCoordinateV1, ...]] = {}

    def scoped(self, row: Mapping[str, Any]) -> None:
        if (row.get("scope_kind"), row.get("scope_id")) != (
            self.scope.scope_kind, self.scope.scope_id,
        ) or ("tenant_id" in row and row["tenant_id"] != self.writer.tenant_id):
            raise QuoteMemError("binding_mismatch", "Source-label loss crosses its original scope")

    def row(self, kind: str, identity: str) -> dict[str, Any]:
        if kind == "evidence":
            header = self.header(identity)
            if header["state"] != "erased":
                raise QuoteMemError("binding_mismatch",
                                    "Source loss cannot hydrate original Evidence")
        row = self.writer.get_row(kind, identity)
        if row is None or row.get("object_id") != identity:
            raise QuoteMemError("binding_mismatch", "Source-label loss lost a fixed authority")
        # Controls can be tenant-wide. Their exact key and digest are verified
        # by the original receipt/history reader before being retained here.
        if kind != "control":
            self.scoped(row)
        self.record(kind, row)
        return row

    def logical(
        self, reference: str, creator: str | None = None, *, kind: str = "run_input",
    ) -> StrictModel:
        def read(kind: str, identity: str) -> dict[str, Any]:
            row = self.row(kind, identity)
            if creator is not None and row["created_by_run_id"] != creator:
                raise QuoteMemError("binding_mismatch",
                                    "Source-label loss has another input creator")
            return row

        return read_logical_artifact(read, reference, kind)

    def header(self, identity: str) -> dict[str, Any]:
        row = self.index.evidence_eligibility_header(identity)
        if row is None or row.get("object_id") != identity:
            raise QuoteMemError("binding_mismatch", "Source-label loss has no eligibility metadata")
        self.scoped(row)
        self.record("evidence_eligibility", row)
        return row

    def parts(self, message: SourceLabelMessageBindingV1) -> tuple[SourceCoordinateV1, ...]:
        reference = message.receipt_ref
        if reference not in self.receipts:
            receipt = self.logical(reference)
            if not isinstance(receipt, SessionInventoryDurableRunInputV2):
                raise QuoteMemError("binding_mismatch", "Source-label loss requires a V2 reception")
            envelope = self.row("artifact", reference)
            owner = self.row("run", envelope["created_by_run_id"])
            if owner["input_ref"] != reference or (
                owner["run_kind"] != "session_ingestion"
                or owner["admission_digest"] != receipt.admission_digest
            ):
                raise QuoteMemError("binding_mismatch",
                                    "Source-label receipt lost its attached owner")
            proof = load_request_coordinates(
                self.writer, self.scope, receipt.business_key.session_id,
                [receipt.business_key.request_id],
            )
            authorities = {(value.entity_kind, value.object_id) for value in proof.dependencies}
            if ("run", owner["object_id"]) not in authorities or (
                "artifact", reference
            ) not in authorities:
                raise QuoteMemError(
                    "binding_mismatch", "Original receipt is outside its accepted family",
                )
            for dependency in proof.dependencies:
                if dependency.entity_kind == "evidence":
                    raise QuoteMemError("binding_mismatch",
                                        "Receipt metadata included an Evidence body")
                row = self.row(dependency.entity_kind, dependency.object_id)
                if self.writer.bind_content("row-state", row) != dependency.digest:
                    raise QuoteMemError(
                        "stale_evidence", "Source-label receipt changed during inspection",
                    )
            for dependency in proof.coordinate_dependencies:
                row = self.index.source_coordinate(dependency.object_id)
                if row is None or self.writer.bind_content("source-coordinate", row) != (
                    dependency.digest
                ):
                    raise QuoteMemError(
                        "stale_evidence", "Source-label coordinates changed during inspection",
                    )
                self.scoped(row)
                self.record("source_coordinate", row)
            coordinates = dict(proof.values).get(receipt.business_key.request_id)
            if coordinates is None or freeze_source_coordinates(
                coordinates, message_count=receipt.message_count, bind=self.writer.bind_content,
            ) != receipt.source_inventory or any(
                (part.session_id, part.request_id, part.ingested_commit_seq) != (
                    receipt.business_key.session_id, receipt.business_key.request_id,
                    receipt.visibility_commit_seq,
                ) for part in coordinates
            ):
                raise QuoteMemError(
                    "binding_mismatch", "Source-label loss changed its original inventory",
                )
            self.receipts[reference] = coordinates
        parts = tuple(part for part in self.receipts[reference] if part.ordinal == message.ordinal)
        if not parts:
            raise QuoteMemError("binding_mismatch",
                                "Source-label loss names an absent received message")
        return parts


def _attached(read: _Read, selected: SourceLabelsRunInputV1, observed_at: int) -> int:
    run = read.run
    started = run["started_at"]
    if type(observed_at) is not int or type(started) is not int or not 0 <= started <= observed_at:
        raise QuoteMemError("binding_mismatch", "Source-label loss predates the original call")
    if read.row("run", run["object_id"]) != dict(run) or run["input_ref"] is None or (
        run["run_kind"] != "evaluate" or run["parent_run_id"] != selected.parent_run_id
        or run["admission_digest"] != selected.admission_digest
        or read.logical(run["input_ref"], run["object_id"]) != selected
    ):
        raise QuoteMemError("binding_mismatch",
                            "Source-label loss changed its attached original input")
    return started


def _cleanup(
    read: _Read, message: SourceLabelMessageBindingV1, identity: str, observed_at: int,
    *, after_observation: bool = False,
) -> SourceLabelsSourceLossV1:
    history = resolve_erased_source_history(read.writer, read.scope, [identity])
    validate_erased_source_history(read.writer, history)
    if len(history.sources) != 1 or history.sources[0].evidence_id != identity:
        raise QuoteMemError("binding_mismatch", "Source-label cleanup has another lost source")
    origin = history.sources[0]
    owner = read.row("run", origin.owner_run_id)
    finished = owner["finished_at"]
    if origin.metadata.state not in {"active", "expired"} or type(finished) is not int or not (
        read.run["started_at"] < finished
    ) or (finished <= observed_at if after_observation else finished > observed_at) or (
        owner["status"] != "succeeded"
    ):
        raise QuoteMemError("binding_mismatch", "Source-label cleanup predates its original call")
    part = next(part for part in read.parts(message) if part.object_id == identity)
    if cleanup_completion_sequence(origin) <= part.ingested_commit_seq:
        raise QuoteMemError("binding_mismatch", "Source-label cleanup predates its received source")
    for dependency in history.dependencies:
        row = read.row(dependency.entity_kind, dependency.object_id)
        if read.writer.bind_content("row-state", row) != dependency.digest:
            raise QuoteMemError("stale_evidence", "Source-label cleanup metadata changed")
    validate_erased_source_history(read.writer, history)
    return SourceLabelsSourceLossV1(
        reason="cleanup", receipt_ref=message.receipt_ref, ordinal=message.ordinal,
        evidence_id=origin.evidence_id, cleanup_run_id=origin.owner_run_id,
    )


def _issued_interruption(read: _Read, interruption: SourceLabelsSourceInterruptionV1) -> None:
    """Keep only the exact original TTL event when its source was erased later.

    Writer permits interruption results only through its sealed recovery
    prewrite. A bare DTO or another event cannot borrow that original issuance.
    The selected cleanup must separately prove the current tombstone, whose
    eligibility header retains the original expiration under the privacy owner.
    """
    run = read.run
    candidates = read.writer.find_logical_candidates(run["object_id"], "result_index")
    if len(candidates) != 1:
        raise QuoteMemError("binding_mismatch", "Erased TTL proof requires its unique issued event")
    reference, _ = candidates[0]
    result = read.logical(reference, run["object_id"], kind="result_index")
    envelope = read.row("artifact", reference)
    if not isinstance(result, SourceLabelsResultIndexV1) or (
        result.input_ref != run["input_ref"]
    ) or (
        result.interruption != interruption or result.llm_output_ref is not None
        or envelope["created_at"] != interruption.observed_at
        or run["result_ref"] not in {None, reference}
        or (run["status"], run["stage"]) not in {
            ("running", "run_cases"), ("failed", "run_cases"),
        }
        or run["status"] == "failed" and (
            run["failure_code"] != "stale_evidence" or run["result_ref"] != reference
            or run["finished_at"] != interruption.observed_at
        )
        or read.index.artifact_headers(run["object_id"], "llm_output", limit=1)
    ):
        raise QuoteMemError("binding_mismatch",
                            "Erased TTL proof changed its original issued event")


def _loss(
    read: _Read, message: SourceLabelMessageBindingV1, identity: str, observed_at: int,
) -> SourceLabelsSourceLossV1 | None:
    header = read.header(identity)
    expiry = header["expires_at"]
    if header["state"] in {"active", "expired"} and type(expiry) is int and (
        read.run["started_at"] < expiry <= observed_at
    ):
        return SourceLabelsSourceLossV1(
            reason="ttl", receipt_ref=message.receipt_ref, ordinal=message.ordinal,
            evidence_id=identity, expires_at=expiry,
        )
    if header["state"] == "erased":
        return _cleanup(read, message, identity, observed_at)
    return None


def _enter(run: Mapping[str, Any]) -> frozenset[str]:
    visiting = _VISITING.get()
    identity = run["object_id"]
    if identity in visiting:
        raise QuoteMemError("reference_cycle",
                            "Source-label cleanup history returns to its own loss")
    return visiting | {identity}


def inspect_source_loss(
    writer: Writer, run: Mapping[str, Any], selected: SourceLabelsRunInputV1,
    *, observed_at: int, record: Record = _noop,
) -> SourceLabelsSourceInterruptionV1 | None:
    """Find one sufficient real source loss, never decoding expired or erased text.

    Discovery chooses one lost part of an original complete message. This proves
    that the original combined call is unusable; a continuation independently
    proves all messages it will actually send. Existing output has priority and
    must be checked by the caller before choosing this interruption.
    """
    token = _VISITING.set(_enter(run))
    try:
        read = _Read(writer, run, record)
        _attached(read, selected, observed_at)
        for message in selected.message_bindings:
            for part in read.parts(message):
                loss = _loss(read, message, part.object_id, observed_at)
                if loss is not None:
                    return SourceLabelsSourceInterruptionV1(
                        kind="source_unavailable", observed_at=observed_at, losses=[loss],
                    )
        return None
    finally:
        _VISITING.reset(token)


def validate_source_loss(
    writer: Writer, run: Mapping[str, Any], selected: SourceLabelsRunInputV1,
    interruption: SourceLabelsSourceInterruptionV1, *, record: Record = _noop,
) -> None:
    """Reprove the exact persisted losses at their original observation frontier."""
    token = _VISITING.set(_enter(run))
    try:
        interruption = SourceLabelsSourceInterruptionV1.model_validate(
            interruption.model_dump(mode="json"),
        )
        read = _Read(writer, run, record)
        _attached(read, selected, interruption.observed_at)
        messages = {
            (value.receipt_ref, value.ordinal): value for value in selected.message_bindings
        }
        for expected in interruption.losses:
            message = messages.get((expected.receipt_ref, expected.ordinal))
            if message is None or expected.evidence_id not in {
                part.object_id for part in read.parts(message)
            }:
                raise QuoteMemError(
                    "binding_mismatch", "Source-label interruption changed its original message",
                )
            header = read.header(expected.evidence_id)
            if expected.reason == "ttl" and header["state"] == "erased":
                if (
                    header["expires_at"] != expected.expires_at or expected.expires_at is None
                ) or not (
                    run["started_at"] < expected.expires_at <= interruption.observed_at
                ):
                    raise QuoteMemError(
                        "binding_mismatch", "Erased TTL proof changed its retained expiration",
                    )
                _issued_interruption(read, interruption)
                _cleanup(
                    read, message, expected.evidence_id, interruption.observed_at,
                    after_observation=True,
                )
                continue
            if canonical_json(_loss(
                read, message, expected.evidence_id, interruption.observed_at,
            )) != canonical_json(expected):
                raise QuoteMemError(
                    "binding_mismatch", "Source-label interruption changed its actual loss",
                )
    finally:
        _VISITING.reset(token)
