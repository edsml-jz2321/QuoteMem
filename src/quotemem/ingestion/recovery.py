"""Read-only ordinary Add replay proof.

This implements no destructive cleanup or Run GC. Unexpected raw transport and
unclassified parts therefore fail closed; a missing family cannot borrow a
future orphan-scan proof. Historical receipt digests never read current Evidence
or cursor state. The single Writer remains the logical Artifact authority.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from quotemem.core.artifacts import ManifestV1
from quotemem.core.errors import QuoteMemError
from quotemem.core.run_inputs import (
    MessagesRequestPayloadV1,
    SessionAdmissionRunInputV1,
    SessionBusinessKeyV1,
    SessionDurableRunInputV1,
)
from quotemem.core.technical_types import RequestProgressV1
from quotemem.ingestion.boundary import AddAdmissionBinding
from quotemem.store.domain import SessionTargetRunRowV1
from quotemem.store.indexed import ArtifactRows, IndexedQuery, IndexedReader, RunRows
from quotemem.store.writer import Writer


@dataclass(frozen=True)
class DurableReplayProof:
    """A complete surviving-family proof at one observed Writer generation.

    Attributes:
        latest_run: Highest surviving Run generation, independent of ULID order.
        witness: Attached historical receipt from the highest generation that has one.
        family_count: Number of surviving family rows checked.
        durable_witness_count: Number of attached durable receipts checked.
        ref_generation: Observed in-process mutation token; callers retry if stale.
    """

    latest_run: SessionTargetRunRowV1
    witness: SessionDurableRunInputV1
    family_count: int
    durable_witness_count: int
    ref_generation: int


def _pages(reader: IndexedReader, query: IndexedQuery) -> Iterator[dict[str, Any]]:
    after = None
    while page := reader.page(query, after=after):
        yield from page
        after = page[-1]["object_id"]


def _creator_transport(writer: Writer, reader: IndexedReader, run_id: str) -> None:
    parts: set[str] = set()
    protected: set[str] = set()
    for artifact in _pages(reader, ArtifactRows(created_by_run_id=run_id)):
        kind = artifact["artifact_kind"]
        if kind == "part":
            parts.add(artifact["object_id"])
            continue
        if kind == "manifest":
            manifest = ManifestV1.model_validate_json(artifact["body"])
            logical = writer.read_logical(artifact["object_id"], manifest.logical_kind)
            if manifest.logical_kind == "run_input" and (
                isinstance(logical, SessionAdmissionRunInputV1)
            ):
                _received_transport(writer, run_id, logical)
            # read_logical proves every part has exactly this creator and envelope;
            # another creator's manifest cannot legitimately protect these parts.
            protected.update(manifest.parts)
        elif kind == "run_input":
            logical = writer.read_logical(artifact["object_id"], "run_input")
            if isinstance(logical, SessionAdmissionRunInputV1):
                _received_transport(writer, run_id, logical)
    if parts - protected:
        raise QuoteMemError("service_unhealthy", "Ordinary Add has unclassified transport parts")


def _received_transport(
    writer: Writer, run_id: str, received: SessionAdmissionRunInputV1,
) -> None:
    """Allow only the same request's saved bytes pending bounded transport cleanup."""
    owner = writer.get_row("run", run_id)
    if owner is None or received.admission_digest != owner["admission_digest"] or (
        received.received_at != owner["created_at"] or received.work_deadline_at is None
        or not isinstance(received.request_payload, MessagesRequestPayloadV1)
        or received.payload_digest != writer.bind_content(
            "run-input-payload", received.request_payload,
        )
    ):
        raise QuoteMemError("binding_mismatch", "Saved Add source changed its original receipt")


def _verify(
    writer: Writer, binding: AddAdmissionBinding, ledger_binding: RequestProgressV1, generation: int
) -> DurableReplayProof:
    reader = IndexedReader(
        writer.runtime.workspace,
        writer.entities,
        tenant_id=writer.tenant_id,
    )
    latest: SessionTargetRunRowV1 | None = None
    selected: SessionDurableRunInputV1 | None = None
    selected_generation = -1
    payload = None
    message_count = None
    principal = None
    count, witnesses = 0, 0
    query = RunRows(run_kind="session_ingestion", idempotency_key=binding.idempotency_key)
    assert isinstance(binding.admission.raw_business_key, SessionBusinessKeyV1)
    business = binding.admission.raw_business_key.model_dump(mode="json")
    for raw in _pages(reader, query):
        run = SessionTargetRunRowV1.model_validate(raw)
        count += 1
        if latest is None or run.run_generation > latest.run_generation:
            if latest is not None:
                _previous_generation(latest)
            latest = run
        else:
            _previous_generation(run)
        if principal is None:
            principal = run.principal_id
        if (
            run.admission_digest != ledger_binding.admission_digest
            or run.principal_id != principal
            or ledger_binding.admission_digest == binding.admission_digest
            and run.principal_id != binding.admission.principal_id
            or run.scope_kind != binding.admission.scope_kind
            or run.scope_id != binding.admission.scope_id
            or run.target_id != binding.admission.target_locator
            or run.commit_mode != "orchestrator"
            or run.run_generation == 0
            and run.parent_run_id is not None
            or run.run_generation > 0
            and run.parent_run_id is None
        ):
            raise QuoteMemError("service_unhealthy", "Durable Run family identity is inconsistent")
        if run.run_generation > 0:
            previous = writer.get_row("run", run.parent_run_id or "")
            if previous is not None:
                parent = SessionTargetRunRowV1.model_validate(previous)
                if parent.run_generation != run.run_generation - 1 or any(
                    getattr(parent, field) != getattr(run, field)
                    for field in (
                        "run_kind",
                        "commit_mode",
                        "idempotency_key",
                        "admission_digest",
                        "scope_kind",
                        "scope_id",
                        "principal_id",
                        "target_kind",
                        "target_id",
                    )
                ):
                    raise QuoteMemError("service_unhealthy", "Durable generation lineage differs")
                _previous_generation(parent)
        if run.failure_code == "input_abandoned" and run.input_ref is not None:
            raise QuoteMemError("service_unhealthy", "Abandoned inputless Run acquired an input")
        if run.input_ref is None:
            if run.status == "succeeded" or run.failure_code == "stale_evidence":
                raise QuoteMemError("service_unhealthy", "Executed Add lost its durable witness")
        else:
            envelope = writer.get_row("artifact", run.input_ref)
            body = writer.read_logical(run.input_ref, "run_input")
            if (
                envelope is None
                or envelope["created_by_run_id"] != run.object_id
                or envelope["tenant_id"] != writer.tenant_id
                or envelope["scope_kind"] != run.scope_kind
                or envelope["scope_id"] != run.scope_id
                or not isinstance(body, SessionDurableRunInputV1)
                or body.request_key != binding.request_key
                or body.admission_digest != ledger_binding.admission_digest
                or body.business_key.model_dump(mode="json") != business
                or body.confirmed_request != ledger_binding.confirmed_request
                or payload is not None
                and body.payload_digest != payload
                or message_count is not None
                and body.message_count != message_count
            ):
                raise QuoteMemError("service_unhealthy",
                                    "Attached durable witness fails binding parity")
            payload = body.payload_digest
            message_count = body.message_count
            witnesses += 1
            if run.run_generation > selected_generation:
                selected, selected_generation = body, run.run_generation
        _creator_transport(writer, reader, run.object_id)
    if latest is None or selected is None:
        raise QuoteMemError("service_unhealthy", "Ledger has no proved surviving durable family")
    return DurableReplayProof(latest, selected, count, witnesses, generation)


def _previous_generation(run: SessionTargetRunRowV1) -> None:
    if run.status != "failed" or run.failure_code not in {
        "input_abandoned",
        "erase_fenced",
        "consent_fenced",
        "stale_evidence",
    }:
        raise QuoteMemError("service_unhealthy",
                            "A previous Run generation lacks legal termination")


def verify_durable_replay(
    writer: Writer, *, binding: AddAdmissionBinding, ledger_binding: RequestProgressV1
) -> DurableReplayProof:
    """Prove stored reception and transport absence before any replay outcome.

    Args:
        writer (Writer): Existing initialized persistence authority, used only for reads.
        binding (AddAdmissionBinding): Complete current canonical request bindings.
        ledger_binding (RequestProgressV1): Entry from the already verified ledger chain.

    Returns:
        DurableReplayProof: Latest family identity and its verified historical receipt.

    Raises:
        QuoteMemError: service_unhealthy for missing/corrupt witness or unexpected raw
            staging; operation_conflict for different admission bytes only after
            proof succeeds; reception_stale when concurrent writes require a retry;
            unsupported_variant for requests outside ordinary messages Add.

    Notes:
        Indexed scans and logical reads occur outside the Writer commit mutex.
        A short before/after mutation-token check detects concurrent changes;
        reception_stale is a retry signal, never corruption evidence. The caller
        retains session ownership and rechecks current grants/fences before reply.
        No provider is called, no bytes are deleted, and no receipt is adopted.
    """
    if not isinstance(binding.admission.raw_business_key, SessionBusinessKeyV1) or not isinstance(
        binding.admission.request_payload, MessagesRequestPayloadV1
    ):
        raise QuoteMemError("unsupported_variant", "Replay proof requires an ordinary messages Add")
    with writer.mutex:
        generation = writer.ref_generation
    error: Exception | None = None
    proof: DurableReplayProof | None = None
    try:
        ledger = RequestProgressV1.model_validate(vars(ledger_binding))
        proof = _verify(writer, binding, ledger, generation)
    except Exception as caught:
        error = caught
    with writer.mutex:
        if writer.ref_generation != generation:
            raise QuoteMemError("reception_stale", "Concurrent writes require another replay proof")
    if error is not None:
        if isinstance(error, QuoteMemError) and error.code == "service_unhealthy":
            raise error
        raise QuoteMemError("service_unhealthy",
                            "Stored replay evidence cannot be verified") from None
    assert proof is not None
    if ledger_binding.admission_digest != binding.admission_digest:
        raise QuoteMemError("operation_conflict",
                            "This request identity binds different input bytes")
    if proof.witness.payload_digest != binding.payload_digest:
        raise QuoteMemError("service_unhealthy", "Stored request payload binding differs")
    # Ordinary Add admissions include the full request payload, so its count can
    # be checked without rereading mutable Evidence or extraction cursor state.
    if proof.witness.message_count != len(binding.admission.request_payload.messages):
        raise QuoteMemError("service_unhealthy", "Stored request message count differs")
    return proof
