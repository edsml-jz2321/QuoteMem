"""Read-only recovery proofs for dedicated privacy-erase owners (C.13).

Rejected reception parents settle only after their bound cleanup completes.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import ValidationError

from aml.core.admission import split_secret_parent_binding
from aml.core.artifacts import (
    ManifestV1,
    ReceiptResultIndexV1,
    RequestLedgerV1,
)
from aml.core.controls import RequestLedgerValueV1, ScopeRegistrationV1, maintenance_fence_key
from aml.core.errors import AMLError
from aml.core.identity import (
    ArtifactId,
    ContentBindingDigest,
    base64url,
    canonical_json,
)
from aml.core.identity import scope_id as canonical_scope_id
from aml.core.ingestion_cleanup import CleanupRunInput, RootRunInput
from aml.core.privacy_persistence import (
    IngestionErasurePlanV1,
    PrivacyEraseProgressV1,
    RedactionReceiptV1,
    validate_ingestion_erasure_chain,
)
from aml.core.run_inputs import (
    SessionDurableRunInputV1,
    SplitSecretChildRunInputV1,
)
from aml.core.semantic import ScopeKind
from aml.core.technical_types import RequestProgressV1
from aml.ingestion.receipts import inspect_ledger
from aml.store.domain import DOMAIN_ADAPTERS, SessionTargetRunRowV1
from aml.store.indexed import ArtifactRows, EvidenceRows, IndexedQuery, IndexedReader
from aml.store.metadata import StrictModel
from aml.store.privacy_plan import (
    PreparedPrivacyEffects,
    load_privacy_plan,
    privacy_effect_counts,
    verify_privacy_completion,
    verify_privacy_prefix,
)
from aml.store.writer import Writer

_EMPTY_FIELDS = (
    "input_ref",
    "result_ref",
    "auth_context_ref",
    "commit_marker",
    "commit_seq",
    "started_at",
)
PrivacyRecoveryState = Literal["awaiting_client_replay", "resume"]


@dataclass(frozen=True)
class PendingPrivacy:
    """A startup reservation, never execution authority or permission to attach.

    Inputless split-secret owners await client replay and remain barred.
    Running owners have a verified persisted prefix; ``erase_epoch_bumps`` are
    the scope's erase_epoch advances that prefix committed (D.1, D.7.5).
    The caller rechecks ref_generation before adoption or publishing reservations.
    """

    owner_run_id: str
    scope_kind: ScopeKind
    scope_id: str
    parent_run_id: str | None
    state: PrivacyRecoveryState
    ref_generation: int
    erase_epoch_bumps: int


def _row(writer: Writer, kind: str, identity: str) -> dict[str, Any]:
    row = writer.get_row(kind, identity)
    if row is None or row["object_id"] != identity:
        raise AMLError("service_unhealthy", "Privacy recovery lost a required row")
    return row


def _pages(reader: IndexedReader, query: IndexedQuery) -> Iterator[dict[str, Any]]:
    after = None
    while page := reader.page(query, after=after):
        yield from page
        after = page[-1]["object_id"]


def _registration(writer: Writer, owner: dict[str, Any]) -> None:
    row = writer.get_control(f"scope:{owner['scope_id']}")
    if row is None:
        raise AMLError("service_unhealthy", "Privacy owner has no registered scope")
    value = ScopeRegistrationV1.model_validate(row["value"])
    if (
        value.scope_kind != owner["scope_kind"]
        or canonical_scope_id(value.scope_kind, value.raw_external_id) != owner["scope_id"]
    ):
        raise AMLError("service_unhealthy", "Privacy owner scope registration differs")


def _parent_header(
    writer: Writer, parent_id: str, *, allow_terminal: bool
) -> SessionTargetRunRowV1:
    parent = SessionTargetRunRowV1.model_validate(_row(writer, "run", parent_id))
    _registration(writer, parent.model_dump(mode="json"))
    pending = parent.status == "queued" and parent.stage == "await_input"
    rejected = (
        allow_terminal
        and parent.status == "failed"
        and parent.failure_code == "policy_reject"
        and parent.stage == "settle"
    )
    if (
        parent.run_kind != "session_ingestion"
        or parent.commit_mode != "orchestrator"
        or parent.target_kind != "session"
        or parent.scope_kind != "user"
        or parent.attempt != 0
        or any(getattr(parent, field) is not None for field in _EMPTY_FIELDS)
        or not (pending or rejected)
    ):
        raise AMLError("service_unhealthy", "Privacy rejection parent has execution state")
    return parent


def _only_child(
    writer: Writer, reader: IndexedReader, parent: SessionTargetRunRowV1, child_id: str
) -> dict[str, Any]:
    children = reader.run_children(parent.object_id, limit=2)
    if len(children) != 1 or children[0]["object_id"] != child_id:
        raise AMLError("service_unhealthy", "Privacy rejection requires its sole exact child")
    child = DOMAIN_ADAPTERS["run"].validate_python(children[0]).model_dump(mode="json")
    if (
        child["run_kind"] != "retention"
        or child["commit_mode"] != "destructive_scan"
        or child["target_kind"] != "affected_set"
        or child["run_generation"] != 0
        or child["principal_id"] != parent.principal_id
        or child["scope_kind"] != parent.scope_kind
        or child["scope_id"] != parent.scope_id
        or child["admitted_erase_epoch"] != parent.admitted_erase_epoch
        or child["admitted_consent_epoch"] != parent.admitted_consent_epoch
        or child["created_at"] < parent.created_at
        or child["commit_marker"] is not None
        or child["commit_seq"] is not None
    ):
        raise AMLError("service_unhealthy", "Privacy rejection child binding differs")
    if reader.run_children(child_id, limit=1):
        raise AMLError("service_unhealthy", "Privacy cleanup unexpectedly owns a child")
    return child


def _logical(writer: Writer, owner: dict[str, Any], reference: str, kind: str) -> StrictModel:
    envelope = _row(writer, "artifact", reference)
    if (
        envelope["created_by_run_id"] != owner["object_id"]
        or envelope["tenant_id"] != writer.tenant_id
        or envelope["scope_kind"] != owner["scope_kind"]
        or envelope["scope_id"] != owner["scope_id"]
    ):
        raise AMLError("service_unhealthy", "Privacy artifact envelope differs from its owner")
    return writer.read_logical(reference, kind)


def _input(
    writer: Writer, child: dict[str, Any], parent: SessionTargetRunRowV1 | None
) -> CleanupRunInput:
    if child["input_ref"] is None:
        raise AMLError("input_required", "Privacy owner awaits its admitted selection")
    body = _logical(writer, child, child["input_ref"], "run_input")
    if parent is None:
        if not isinstance(body, RootRunInput) or child["parent_run_id"] is not None:
            raise AMLError("service_unhealthy", "Independent privacy owner has a child input")
    elif not isinstance(
        body, SplitSecretChildRunInputV1
    ) or body.business_key.parent_binding_digest != split_secret_parent_binding(
        parent.model_dump(mode="json"), writer.bind_content
    ):
        raise AMLError("service_unhealthy", "Privacy child lost its immutable parent binding")
    return body


def _plan(
    writer: Writer,
    child: dict[str, Any],
    body: CleanupRunInput,
) -> PreparedPrivacyEffects:
    reverse: list[tuple[ArtifactId, IngestionErasurePlanV1]] = []
    seen: set[ArtifactId] = set()
    reference: ArtifactId | None = body.plan_ref
    while reference is not None:
        if reference in seen:
            raise AMLError("service_unhealthy", "Privacy recovery found a plan cycle")
        seen.add(reference)
        block = _logical(writer, child, reference, "erasure_plan")
        if not isinstance(block, IngestionErasurePlanV1):
            raise AMLError("service_unhealthy", "Privacy recovery found another plan variant")
        reverse.append((reference, block))
        reference = block.previous_ref
    summary = validate_ingestion_erasure_chain(reversed(reverse))
    return load_privacy_plan(
        writer,
        body.plan_ref,
        owner_run_id=child["object_id"],
        supplied=body.business_key,
        expected_plan_digest=summary.plan_digest,
    )


def _parent_transport(writer: Writer, reader: IndexedReader, parent: SessionTargetRunRowV1) -> None:
    parts: set[str] = set()
    protected: set[str] = set()
    owner = parent.model_dump(mode="json")
    for artifact in _pages(reader, ArtifactRows(created_by_run_id=parent.object_id)):
        kind = artifact["artifact_kind"]
        if kind == "part":
            parts.add(artifact["object_id"])
            continue
        if kind == "manifest":
            manifest = ManifestV1.model_validate_json(artifact["body"])
            kind = manifest.logical_kind
            protected.update(manifest.parts)
        if kind not in {"run_input", "request_ledger"}:
            raise AMLError("service_unhealthy", "Rejected reception has nonexempt artifacts")
        body = _logical(writer, owner, artifact["object_id"], kind)
        if kind == "run_input":
            if not isinstance(body, SessionDurableRunInputV1) or body.confirmed_request is not None:
                raise AMLError("service_unhealthy", "Rejected reception has raw or confirmed input")
            identity = _family(parent, body.business_key.session_id, body.business_key.request_id)
            session = base64url(hashlib.sha256(body.business_key.session_id.encode()).digest())[:22]
            if (
                body.admission_digest != parent.admission_digest
                or identity != parent.idempotency_key
                or session != parent.target_id
            ):
                raise AMLError("service_unhealthy", "Rejected reception orphan binding differs")
    if parts - protected:
        raise AMLError("service_unhealthy", "Rejected reception has ambiguous transport parts")


def _family(parent: SessionTargetRunRowV1, session: str, request: str) -> str:
    value = {
        "scope_kind": parent.scope_kind,
        "scope_id": parent.scope_id,
        "business_key": {"session_id": session, "request_id": request},
    }
    return "idem_" + base64url(hashlib.sha256(canonical_json(value).encode()).digest())


def _direct_effects(writer: Writer, reader: IndexedReader, parent: SessionTargetRunRowV1) -> None:
    if reader.run_effects(parent.object_id, include_children=False):
        raise AMLError("service_unhealthy", "Rejected reception has direct domain effects")
    for evidence in _pages(
        reader,
        EvidenceRows.model_validate({"scope_kind": parent.scope_kind, "scope_id": parent.scope_id}),
    ):
        if (
            _family(parent, evidence["session_id"], evidence["request_id"])
            == parent.idempotency_key
        ):
            raise AMLError("service_unhealthy", "Rejected reception family already has Evidence")
    if reader.control_provenance(parent.object_id, limit=1):
        raise AMLError("service_unhealthy", "Rejected reception has Control effects")
    _parent_transport(writer, reader, parent)
    ledger = writer.get_control(f"request_ledger:{parent.scope_id}:{parent.target_id}")
    if ledger is None:
        return
    value = RequestLedgerValueV1.model_validate(ledger["value"])

    def read(reference: ArtifactId) -> RequestLedgerV1:
        body = writer.read_logical(reference, "request_ledger")
        if not isinstance(body, RequestLedgerV1):
            raise AMLError("service_unhealthy", "Rejected reception ledger variant differs")
        return body

    def observe(_key: ContentBindingDigest, progress: RequestProgressV1) -> None:
        if progress.admission_digest == parent.admission_digest:
            raise AMLError("service_unhealthy", "Rejected reception already has a ledger binding")

    inspect_ledger(value, parent.admission_digest, read, writer.bind_content, observe=observe)


def _progress(
    writer: Writer,
    child: dict[str, Any],
    body: CleanupRunInput,
) -> PrivacyEraseProgressV1:
    key = maintenance_fence_key(
        child["scope_id"], body.business_key.affected[0].evidence_id, writer.bind_content
    )
    control = writer.get_control(key)
    if control is None:
        raise AMLError("service_unhealthy", "Running privacy owner lost its durable fence")
    progress = PrivacyEraseProgressV1.model_validate(control["value"])
    if progress.owner_run_id != child["object_id"]:
        raise AMLError("service_unhealthy", "Privacy progress belongs to another owner")
    expected_stage = {"prepared": "select", "applying": "compensate", "verifying": "scan"}[
        progress.phase
    ]
    if child["stage"] != expected_stage:
        raise AMLError("service_unhealthy", "Privacy owner stage differs from its progress")
    return progress


def privacy_rejected_parent(
    writer: Writer,
    parent_id: str,
    child_id: str,
    *,
    allow_terminal: bool = False,
    allow_active_child: bool = False,
) -> SessionTargetRunRowV1:
    """Prove a rejected reception parent without changing any durable state.

    Args:
        writer: The sole Writer, used only for decoded reads and tenant bindings.
        parent_id: Exact ordinary Add root whose boundary was never accepted.
        child_id: Its sole dedicated same-scope privacy child.
        allow_terminal: Permit an already settled failed(policy_reject) parent.
        allow_active_child: Explicit precommit mode: require the running child
            at its fully verified cursor. The Writer can then settle both rows
            atomically; this option never accepts an inputless or partial child.

    Returns:
        The strict stored parent after child, input, plan, receipt and direct-effect
        proofs. Later historical replay validates immutable completion evidence;
        it does not demand that subsequently changed domain rows remain frozen.

    Raises:
        AMLError: On missing, partial, mismatched or concurrently changed proof.

    Notes:
        Call outside the commit mutex. The mutation owner must retain its own
        ref_generation token and recheck it and both Run rows before committing.
    """
    generation = writer.ref_generation
    try:
        parent = _parent_header(writer, parent_id, allow_terminal=allow_terminal)
        reader = IndexedReader(
            writer.runtime.workspace,
            writer.entities,
            tenant_id=writer.tenant_id,
        )
        child = _only_child(writer, reader, parent, child_id)
        body = _input(writer, child, parent)
        proof = _plan(writer, child, body)
        _direct_effects(writer, reader, parent)
        if child["status"] == "running" and allow_active_child and parent.status == "queued":
            progress = _progress(writer, child, body)
            if progress.phase != "verifying" or progress.cursor != len(proof.batches):
                raise AMLError("service_unhealthy", "Privacy child has incomplete progress")
            verify_privacy_prefix(writer, proof, progress)
            if verify_privacy_completion(writer, proof) != progress.affected_counts:
                raise AMLError("service_unhealthy", "Privacy child counts differ from its effects")
        elif child["status"] == "succeeded" and child["stage"] == "scan":
            key = maintenance_fence_key(
                child["scope_id"], body.business_key.affected[0].evidence_id, writer.bind_content
            )
            fence = writer.get_control(key)
            if fence is not None and fence["value"].get("owner_run_id") == child_id:
                raise AMLError("service_unhealthy", "Completed privacy child still owns its fence")
            if child["result_ref"] is None:
                raise AMLError("service_unhealthy", "Completed privacy child lost its receipt")
            result = _logical(writer, child, child["result_ref"], "result_index")
            if not isinstance(result, ReceiptResultIndexV1):
                raise AMLError("service_unhealthy", "Privacy completion has another result variant")
            receipt = _logical(writer, child, result.receipt_ref, "receipt")
            if not isinstance(receipt, RedactionReceiptV1):
                raise AMLError(
                    "service_unhealthy", "Privacy completion has another receipt variant"
                )
            if (
                receipt.counts != privacy_effect_counts(writer, proof)
                or receipt.reason_code != body.business_key.reason_code
                or receipt.affected_digests
                != sorted(target.expected_row_digest for target in body.business_key.affected)
                or parent.finished_at is not None
                and parent.finished_at < child["finished_at"]
            ):
                raise AMLError(
                    "service_unhealthy", "Privacy receipt does not cover the selected plan"
                )
        else:
            raise AMLError("service_unhealthy", "Privacy child has no successful completion proof")
    except ValidationError as error:
        raise AMLError(
            "service_unhealthy", "Privacy rejection proof has an invalid schema"
        ) from error
    if writer.ref_generation != generation:
        raise AMLError("reception_stale", "Privacy rejection proof changed during its scan")
    return parent


def classify_pending_privacy(writer: Writer) -> tuple[PendingPrivacy, ...]:
    """Reserve unresolved cleanup scopes before ordinary inputless Add adoption.

    Args:
        writer: Read-only access to the startup workspace before listeners open.

    Returns:
        Immutable reservations. Inputless split-secret owners need client replay
        and no reservation attaches an orphan plan. Running owners prove their
        full prefix and suffix.

    Raises:
        AMLError: On malformed owners, unrelated children, effects, or stale proof.

    Notes:
        This function does not adopt epochs, attach candidates, create a fence,
        resume cleanup, or make the service healthy. The composition owns those
        steps and keeps every listed scope barred in the interim.
    """
    generation = writer.ref_generation
    reader = IndexedReader(
        writer.runtime.workspace,
        writer.entities,
        tenant_id=writer.tenant_id,
    )
    result: list[PendingPrivacy] = []
    scopes: set[str] = set()
    after = None
    try:
        while page := reader.pending_privacy(after=after):
            after = page[-1]["object_id"]
            for raw in page:
                owner = DOMAIN_ADAPTERS["run"].validate_python(raw).model_dump(mode="json")
                _registration(writer, owner)
                if owner["scope_id"] in scopes:
                    raise AMLError("service_unhealthy", "Scope has competing privacy owners")
                scopes.add(owner["scope_id"])
                parent = None
                if owner["parent_run_id"] is not None:
                    parent = _parent_header(writer, owner["parent_run_id"], allow_terminal=False)
                    _only_child(writer, reader, parent, owner["object_id"])
                    _direct_effects(writer, reader, parent)
                if owner["status"] == "queued":
                    if (
                        owner["stage"] != "await_input"
                        or owner["input_ref"] is not None
                        or any(owner[field] is not None for field in _EMPTY_FIELDS)
                        or owner["attempt"] != 0
                        or reader.run_effects(owner["object_id"])
                        or reader.run_children(owner["object_id"], limit=1)
                        or reader.control_provenance(owner["object_id"], limit=1)
                    ):
                        raise AMLError(
                            "service_unhealthy", "Queued privacy owner has execution effects"
                        )
                    for fence in reader.scope_fences(owner["scope_id"]):
                        if fence["value"].get("owner_run_id") == owner["object_id"]:
                            raise AMLError(
                                "service_unhealthy", "Queued privacy owner already owns a fence"
                            )
                    state: PrivacyRecoveryState = "awaiting_client_replay"
                    bumps = 0
                else:
                    body = _input(writer, owner, parent)
                    proof = _plan(writer, owner, body)
                    progress = _progress(writer, owner, body)
                    verify_privacy_prefix(writer, proof, progress)
                    state = "resume"
                    bumps = progress.erase_epoch_bumps
                result.append(
                    PendingPrivacy(
                        owner["object_id"],
                        owner["scope_kind"],
                        owner["scope_id"],
                        owner["parent_run_id"],
                        state,
                        generation,
                        bumps,
                    )
                )
    except ValidationError as error:
        raise AMLError(
            "service_unhealthy", "Pending privacy owner has an invalid schema"
        ) from error
    if writer.ref_generation != generation:
        raise AMLError("reception_stale", "Pending privacy classification changed during its scan")
    return tuple(result)
