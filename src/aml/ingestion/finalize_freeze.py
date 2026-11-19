"""Issued input-only freeze authority for the Add owner's frozen prefix (C.2 step 0, C.2a).

The owner scans within an existing private held-input callback. This module
issues the compact input selection, including nonempty inputs, but grants no
child, provider, release or cross-process recovery authority;
``store/owner_lease.persist_finalize_input`` selects the input and moves the
owner to ``extract``.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import weakref
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from itertools import islice
from typing import Any, cast

from pydantic import TypeAdapter, ValidationError

from aml.core.admission import admission_identity, finalize_admission
from aml.core.artifacts import ArtifactRowV1
from aml.core.controls import (
    FAST_EXTRACTION_POLICY_KEY,
    ControlRowV1,
    HeldFinalizeLeaseV1,
    PolicyPointerV1,
)
from aml.core.errors import AMLError
from aml.core.extraction_policy import ExtractionPolicyConfigV2, normalize_extraction_policy
from aml.core.identity import ArtifactId, ContentBindingDigest, canonical_json, integrity_digest
from aml.core.query import ScopeV1
from aml.core.run_inputs import FinalizeBusinessKeyV1, FinalizeOwnerRunInputV1
from aml.core.semantic import SystemTimestamp
from aml.core.technical_types import MessageRangeKeyV2
from aml.core.work import (
    FinalizeSourceInventoryBlockV1,
    FinalizeSourceInventoryMemberV1,
    PlanMessageRangeKeyV1,
    PlanMessageRangeKeyV2,
)
from aml.ingestion.authorization import check_add_grant
from aml.ingestion.finalize_heads import (
    finalize_input_heads_digest,
    inspect_finalize_evidence_digest,
    row_elements,
    support_headers,
)
from aml.ingestion.finalize_receipts import FinalizeReceiptCensus, inspect_finalize_receipts
from aml.ingestion.request_order import RequestOrder, load_request_order
from aml.ingestion.segment import checkpoint_partitions, message_key
from aml.providers.auth import PrincipalGrant
from aml.store.artifact_read import read_logical_artifact
from aml.store.finalize_ticket import FinalizeLeaseTicket
from aml.store.indexed import (
    ArtifactRows,
    FinalizeEvidenceQuery,
    IndexedReader,
    finalize_source_order,
)
from aml.store.metadata import StrictModel
from aml.store.read_model import SnapshotReadGuard, StoreReadModel
from aml.store.row_cache import MEMBERSHIP_ANY, MISSING, membership_family, membership_key
from aml.store.writer import Writer

_ISSUANCE_KEY = secrets.token_bytes(32)
_TIME: TypeAdapter[int] = TypeAdapter(SystemTimestamp)


def finalize_action_time(clock: Callable[[], int]) -> int:
    """Read a strict trusted action timestamp for input discovery or persistence."""
    try:
        return _TIME.validate_python(clock(), strict=True)
    except ValidationError as error:
        raise AMLError("schema_invalid", "Invalid finalize action timestamp") from error


@dataclass(frozen=True)
class FinalizeInputProof:
    """Keep fixed metadata and compact input bytes, with process-owned issuance."""

    writer: weakref.ReferenceType[Writer] = field(repr=False)
    guard: weakref.ReferenceType[SnapshotReadGuard] = field(repr=False)
    ticket: FinalizeLeaseTicket = field(repr=False)
    owner_json: str = field(repr=False)
    input_json: str = field(repr=False)
    policy_json: str = field(repr=False)
    scope_token: tuple[int, int, int]
    artifact_generation: int
    child_token: int
    input_ref: ArtifactId | None
    seal: str = field(default="", repr=False)
    receipt_keys: tuple[tuple[str, str], ...] = ()
    receipt_token: int = 0

    @property
    def input(self) -> FinalizeOwnerRunInputV1:
        return FinalizeOwnerRunInputV1.model_validate_json(self.input_json)


def _seal(proof: FinalizeInputProof) -> str:
    payload = {
        "writer": id(proof.writer()), "guard": id(proof.guard()),
        "ticket": id(proof.ticket),
        "cycle": (
            proof.ticket.owner_run_id, proof.ticket.scope_id,
            proof.ticket.lease_generation, proof.ticket.writer_epoch, proof.ticket.state,
        ),
        "owner": proof.owner_json, "input": proof.input_json, "policy": proof.policy_json,
        "scope_token": proof.scope_token, "artifact_generation": proof.artifact_generation,
        "child_token": proof.child_token, "input_ref": proof.input_ref,
        "receipt_keys": proof.receipt_keys, "receipt_token": proof.receipt_token,
    }
    return hmac.new(_ISSUANCE_KEY, canonical_json(payload).encode(), hashlib.sha256).hexdigest()


def _state(
    writer: Writer, ticket: FinalizeLeaseTicket, guard: SnapshotReadGuard,
    grant: PrincipalGrant, at_time: int,
) -> tuple[dict[str, Any], tuple[int, int, int], int, int]:
    try:
        grant = PrincipalGrant.model_validate(grant.model_dump(mode="json"))
    except (AttributeError, ValidationError) as error:
        raise AMLError("schema_invalid", "Invalid finalize input authorization") from error
    with writer.mutex:
        owner, lease, _ = writer._finalize_precondition(ticket)  # pyright: ignore[reportPrivateUsage]
        if not isinstance(lease.value, HeldFinalizeLeaseV1) or ticket.state != "held":
            raise AMLError("lease_lost", "Finalize input requires actual held ownership")
        if (
            owner["result_ref"] is not None or owner["auth_context_ref"] is not None
            or owner["commit_marker"] is not None or owner["commit_seq"] is not None
            or owner["stage"] not in (
                {"lease", "freeze"} if owner["input_ref"] is None else {"extract"}
            )
        ):
            raise AMLError("unsupported_variant", "Finalize has advanced past input-only freeze")
        scope = ScopeV1(scope_kind=owner["scope_kind"], scope_id=owner["scope_id"])
        check_add_grant(grant, scope, at_time, owner["principal_id"])
        publisher = StoreReadModel.for_writer(writer)
        if publisher is None:
            raise AMLError("index_rebuilding", "Finalize input has no registered publisher")
        publisher.validate_finalize_held(
            writer, scope, ticket.owner_run_id, ticket.lease_generation, ticket.writer_epoch,
        )
        guard.check_operation(writer, scope)
        if (
            guard.metadata.generation != lease.value.baseline_snapshot_generation
            or guard.metadata.root_identity != lease.value.baseline_root_identity
        ):
            raise AMLError("binding_mismatch", "Finalize input guard has another baseline")
        token = tuple(writer.counter(f"{name}:{scope.scope_id}") for name in (
            "scope_seq", "erase_epoch", "consent_epoch",
        ))
        return (
            owner, (token[0], token[1], token[2]),
            writer._artifact_generations.get(owner["object_id"], 0),  # pyright: ignore[reportPrivateUsage]
            writer.child_change_token(owner["object_id"]),
        )


def _policy(writer: Writer, reference: ArtifactId) -> ExtractionPolicyConfigV2:
    """Read fixed logical policy references through the existing full validator."""
    def global_body(selected: ArtifactId, kind: str) -> dict[str, Any]:
        raw = writer.get_row("artifact", selected)
        if raw is None:
            raise AMLError("reference_missing", "Finalize governing policy is unavailable")
        try:
            row = ArtifactRowV1.model_validate(raw)
        except ValidationError as error:
            raise AMLError("schema_invalid", "Invalid stored finalize policy envelope") from error
        if row.tenant_id != writer.tenant_id or (
            row.scope_kind is not None or row.scope_id is not None
        ):
            raise AMLError("scope_denied", "Finalize policy has another ownership scope")
        return writer.read_logical(selected, kind).model_dump(mode="json")

    release = global_body(reference, "policy_release")
    if release["policy_kind"] != "extraction":
        raise AMLError("binding_mismatch", "Finalize selected another policy kind")
    try:
        return ExtractionPolicyConfigV2.model_validate(
            global_body(ArtifactId(release["config_ref"]), "policy_config"),
        )
    except ValidationError as error:
        raise AMLError("schema_invalid", "Invalid stored finalize extraction policy") from error


def _governing_policy_ref(writer: Writer, receipts: FinalizeReceiptCensus) -> ArtifactId:
    """Select the new owner's extraction policy release by its user's Add tier.

    The tier counts the scope's accepted Adds and their source bytes up to the owner's
    frozen receipt cutoff, from reception metadata alone (``AddTiersV1``). The owner's
    input freezes the selected release, so a restart, a recovery, a resent Add and a
    recorded replay run the same tier; without tiers it is always ``policy:extraction``.
    """
    key = "policy:extraction"
    tiers = writer.add_tiers
    if tiers is not None and tiers.tier(
        chunks=len(receipts.past), source_bytes=sum(
            header.byte_end - header.byte_start
            for receipt in receipts.past for header in receipt.headers
        ),
    ) == "fast":
        key = FAST_EXTRACTION_POLICY_KEY
    pointer = writer.get_control(key)
    if pointer is None:
        raise AMLError("reference_missing", "Finalize has no governing extraction policy")
    try:
        control = ControlRowV1.model_validate(pointer)
    except ValidationError as error:
        raise AMLError("schema_invalid", "Invalid stored extraction pointer") from error
    if control.key != key or not isinstance(control.value, PolicyPointerV1):
        raise AMLError("binding_mismatch", "Finalize extraction pointer is inconsistent")
    return control.value.current_release_ref


def _selected_input(
    writer: Writer, owner: dict[str, Any],
) -> tuple[ArtifactId, FinalizeOwnerRunInputV1] | None:
    selected = writer.find_logical_candidates(owner["object_id"], "run_input")
    if len(selected) > 1 or (owner["input_ref"] is not None and (
        not selected or selected[0][0] != owner["input_ref"]
    )):
        raise AMLError("parity_mismatch", "Finalize has conflicting or missing selected inputs")
    if not selected:
        return None
    reference, body = selected[0]
    if not isinstance(body, FinalizeOwnerRunInputV1):
        raise AMLError("binding_mismatch", "Finalize input has another strict worker role")
    return reference, body


def _source_inventory_blocks(
    writer: Writer, ticket: FinalizeLeaseTicket, guard: SnapshotReadGuard,
    reader: IndexedReader, scope: ScopeV1,
    *, observe: Callable[[str, str], None] | None = None,
) -> Iterator[tuple[ArtifactId, FinalizeSourceInventoryBlockV1]]:
    """Discover only this owner's bounded metadata blocks without retaining a chain index."""
    after: str | None = None
    query = ArtifactRows(created_by_run_id=ticket.owner_run_id, kinds=("decision_set", "manifest"))
    while rows := reader.page(query, after=after):
        for raw in rows:
            guard.check_operation(writer, scope)
            reference = ArtifactId(raw["object_id"])
            if observe is not None:
                observe("artifact", reference)

            def read(kind: str, identity: str) -> dict[str, Any] | None:
                assert observe is not None
                observe(kind, identity)
                return writer.get_row(kind, identity)

            def logical(kind: str, reference: ArtifactId = reference) -> StrictModel:
                return (writer.read_logical(reference, kind) if observe is None
                        else writer.read_logical_observed(reference, kind, read))

            if raw["artifact_kind"] == "manifest":
                manifest = logical("manifest")
                if manifest.model_dump()["logical_kind"] != "decision_set":
                    continue
            value = logical("decision_set")
            if not isinstance(value, FinalizeSourceInventoryBlockV1):
                continue
            if (raw["tenant_id"], raw["scope_kind"], raw["scope_id"], value.owner_run_id) != (
                writer.tenant_id, scope.scope_kind, scope.scope_id, ticket.owner_run_id,
            ) or value.finalize_cycle != ticket.lease_generation:
                raise AMLError("binding_mismatch", "Source inventory changed its original owner")
            yield reference, value
        after = rows[-1]["object_id"]


class _SourceInventoryIndex:
    """One freeze pass's ordered block lookup, never input or write authority.

    The first lookup consumes the original discovery synchronously, including
    its refusal order. A hit retains only bounded canonical block bytes and
    physical row dependencies. The Writer alone extends the membership proof
    over its exact own prewrite, inside its existing exclusive storage scope.
    """

    def __init__(
        self, writer: Writer, ticket: FinalizeLeaseTicket, guard: SnapshotReadGuard,
        reader: IndexedReader, scope: ScopeV1,
    ) -> None:
        self.writer, self.ticket, self.guard = writer, ticket, guard
        self.reader, self.scope = reader, scope
        self.token: int | None = None
        self.keys: set[tuple[str, str]] = set()
        self.blocks: dict[int, list[tuple[int, ArtifactId, str]]] = {}
        self.order: list[int] = []
        self.size = 0

    def _clear(self) -> None:
        self.token = None
        self.keys.clear()
        self.blocks.clear()
        self.order.clear()
        self.size = 0

    def _current(self) -> bool:
        return self.token is not None and self.writer.borrowed_rows_unchanged(
            self.keys, self.token,
        )

    @staticmethod
    def _match(
        reference: ArtifactId, value: str, body: str, existing: ArtifactId | None,
    ) -> ArtifactId:
        if value != body or existing is not None and existing != reference:
            raise AMLError("parity_mismatch", "Source inventory changed its original block")
        return reference

    def find(self, body: FinalizeSourceInventoryBlockV1) -> ArtifactId | None:
        encoded = canonical_json(body)
        existing: ArtifactId | None = None
        if not self._current():
            self._clear()
            token = self.writer.storage_epoch
            self.keys.update((
                MEMBERSHIP_ANY, membership_family("artifact"),
                membership_key("artifact", self.ticket.owner_run_id),
            ))
            overflow = False

            def observe(kind: str, identity: str) -> None:
                nonlocal overflow
                if not overflow:
                    self.keys.add((kind, identity))
                    if self.size + 64 * len(self.keys) > self.writer.scan_memo().budget:
                        self._clear()
                        overflow = True

            for reference, value in _source_inventory_blocks(
                self.writer, self.ticket, self.guard, self.reader, self.scope, observe=observe,
            ):
                if (value.freeze_time, value.root_input_heads_digest,
                    value.receipt_high_watermark) != (
                    body.freeze_time, body.root_input_heads_digest, body.receipt_high_watermark,
                ) or body.is_head and value.block_no > body.block_no:
                    raise AMLError(
                        "parity_mismatch", "Source inventory has competing frozen roots",
                    )
                value_json = canonical_json(value)
                if value.block_no == body.block_no:
                    existing = self._match(reference, value_json, encoded, existing)
                if not overflow:
                    self.size += 200 + len(value_json.encode())
                    if self.size + 64 * len(self.keys) > self.writer.scan_memo().budget:
                        self._clear()
                        overflow = True
                    else:
                        self.blocks.setdefault(value.block_no, []).append((
                            len(self.order), reference, value_json,
                        ))
                        self.order.append(value.block_no)
            if (not overflow
                and self.size + 64 * len(self.keys) <= self.writer.scan_memo().budget
                and self.writer.borrowed_rows_unchanged(self.keys, token)):
                self.token = token
            return existing

        try:
            existing = self._find_current(body, encoded)
        except AMLError:
            if self._current():
                raise
        else:
            if self._current():
                return existing
        self._clear()
        return self.find(body)

    def _find_current(
        self, body: FinalizeSourceInventoryBlockV1, encoded: str,
    ) -> ArtifactId | None:
        existing: ArtifactId | None = None
        # The first later block competes with this head in discovery order;
        # retain that position relative to same-number duplicate/mismatch checks.
        later = next((position for position, number in enumerate(self.order)
                      if number > body.block_no), None) if body.is_head else None
        for position, reference, value in self.blocks.get(body.block_no, ()):
            self.guard.check_operation(self.writer, self.scope)
            if later is not None and later < position:
                raise AMLError("parity_mismatch", "Source inventory has competing frozen roots")
            existing = self._match(reference, value, encoded, existing)
        self.guard.check_operation(self.writer, self.scope)
        if later is not None:
            raise AMLError("parity_mismatch", "Source inventory has competing frozen roots")
        return existing

    def before_write(
        self, writer: Writer, ticket: FinalizeLeaseTicket, guard: SnapshotReadGuard,
    ) -> int | None:
        """Capture a still-current proof inside the Writer's exclusive write scope."""
        if (writer is not self.writer or ticket is not self.ticket or guard is not self.guard
            or not self._current()):
            self._clear()
            return None
        return self.token

    def after_write(
        self, token: int, following: int | None, reference: ArtifactId,
        body: FinalizeSourceInventoryBlockV1,
    ) -> None:
        """Extend only the Writer's own exact prewrite before storage exclusion ends."""
        if token != self.token or following is None:
            self._clear()
            return
        keys: set[tuple[str, str]] = set()
        overflow = False

        def read(kind: str, identity: str) -> dict[str, Any] | None:
            nonlocal overflow
            if not overflow:
                keys.add((kind, identity))
                if self.size + 64 * (len(self.keys) + len(keys)) > self.writer.scan_memo().budget:
                    keys.clear()
                    overflow = True
            return self.writer.get_row(kind, identity)

        try:
            actual = self.writer.read_logical_observed(reference, "decision_set", read)
        except AMLError:
            # A cache extension supplies no new acceptance or refusal. The
            # ordinary subsequent read/final fence remains the error owner.
            self._clear()
            return
        if overflow or actual != body or not self.writer.borrowed_rows_unchanged(
            self.keys | keys, following,
        ):
            self._clear()
            return
        self.keys.update(keys)
        self.token = following


def _inventory_snapshot(
    writer: Writer, scope: ScopeV1, reference: ArtifactId, count: int,
    *, check: Callable[[], None], observe: Callable[[str, dict[str, Any]], None] | None = None,
) -> Iterator[FinalizeSourceInventoryMemberV1]:
    """Apply immutable deltas oldest first, then restore the canonical support order."""
    seen: set[str] = set()
    creators: dict[str, str] = {}
    snapshots: list[tuple[FinalizeSourceInventoryBlockV1,
                          list[FinalizeSourceInventoryMemberV1]]] = []
    newer: FinalizeSourceInventoryBlockV1 | None = None

    def read(kind: str, identity: str) -> dict[str, Any] | None:
        check()
        raw = writer.get_row(kind, identity)
        if kind != "artifact" or raw is None or (
            raw["tenant_id"], raw["scope_kind"], raw["scope_id"],
        ) != (writer.tenant_id, scope.scope_kind, scope.scope_id):
            raise AMLError("binding_mismatch", "Inventory snapshot crossed its original scope")
        if observe is not None:
            observe(identity, raw)
        creators[identity] = raw["created_by_run_id"]
        return raw

    while True:
        if reference in seen:
            raise AMLError("parity_mismatch", "Inventory snapshot repeats an ancestor")
        seen.add(reference)
        head = read_logical_artifact(read, reference, "decision_set")
        if not isinstance(head, FinalizeSourceInventoryBlockV1) or not head.is_head or (
            count != head.total_record_count or head.owner_run_id != creators[reference]
        ) or newer is not None and (
            head.owner_run_id == newer.owner_run_id
            or head.receipt_high_watermark > newer.receipt_high_watermark
            or head.freeze_time > newer.freeze_time
        ):
            raise AMLError("parity_mismatch", "Inventory snapshot lost its complete head")
        chunks = [head]
        block = head
        while block.previous_ref is not None:
            if block.previous_ref in seen:
                raise AMLError("parity_mismatch", "Inventory snapshot repeats a delta block")
            seen.add(block.previous_ref)
            previous = read_logical_artifact(read, block.previous_ref, "decision_set")
            if not isinstance(previous, FinalizeSourceInventoryBlockV1) or previous.is_head or (
                previous.owner_run_id != creators[block.previous_ref]
                or previous.base_ref != head.base_ref
                or previous.base_count != head.base_count
                or previous.removed_ids != head.removed_ids
            ) or (
                previous.owner_run_id, previous.finalize_cycle, previous.freeze_time,
                previous.receipt_high_watermark, previous.root_input_heads_digest,
                previous.block_no + 1, previous.total_record_count + block.record_count,
            ) != (
                head.owner_run_id, head.finalize_cycle, head.freeze_time,
                head.receipt_high_watermark, head.root_input_heads_digest,
                block.block_no, block.total_record_count - (
                    block.base_count - len(block.removed_ids) if block.is_head else 0
                ),
            ):
                raise AMLError("parity_mismatch", "Inventory snapshot changed its delta chain")
            chunks.append(previous)
            block = previous
        delta: list[FinalizeSourceInventoryMemberV1] = []
        binding = writer.content_array_binding("finalize-source-inventory")
        for chunk in reversed(chunks):
            for member in chunk.records:
                if member.index != binding.count or (
                    member.header.tenant_id, member.header.scope_kind, member.header.scope_id,
                ) != (writer.tenant_id, scope.scope_kind, scope.scope_id):
                    raise AMLError("parity_mismatch", "Inventory delta changed its coordinates")
                binding.append(member)
                delta.append(member)
            if not chunk.is_head and (chunk.total_record_count, chunk.cumulative_digest) != (
                binding.count, binding.digest(),
            ):
                raise AMLError("parity_mismatch", "Inventory delta changed its cumulative binding")
        if binding.count + head.base_count - len(head.removed_ids) != head.total_record_count:
            raise AMLError("parity_mismatch", "Inventory delta changed its complete count")
        snapshots.append((head, delta))
        if head.base_ref is None:
            break
        reference, count, newer = head.base_ref, head.base_count, head
    members: dict[str, FinalizeSourceInventoryMemberV1] = {}
    for head, delta in reversed(snapshots):
        for identity in head.removed_ids:
            if members.pop(identity, None) is None:
                raise AMLError("parity_mismatch", "Inventory removed an absent original member")
        for member in delta:
            identity = member.header.object_id
            if identity in members:
                raise AMLError("parity_mismatch", "Inventory delta repeats an unchanged member")
            members[identity] = member
        if len(members) != head.total_record_count:
            raise AMLError("parity_mismatch", "Inventory snapshot changed its member count")
    for index, member in enumerate(sorted(
        members.values(), key=lambda value: finalize_source_order(value.header),
    )):
        yield member.model_copy(update={"index": index})


def inspect_finalize_source_inventory_artifacts(
    writer: Writer, owner: dict[str, Any], selected: FinalizeOwnerRunInputV1 | None,
    *, check: Callable[[], None],
    observe: Callable[[str, dict[str, Any]], None] | None = None,
) -> None:
    """Prove the source-freeze physical closure for cold adoption without reading sources.

    The caller proves the real owner and lease and keeps its fixed view. A
    partial exact prewrite chain is allowed only before input selection; it
    grants neither a complete source inventory nor content access.
    """
    scope = ScopeV1(scope_kind=owner["scope_kind"], scope_id=owner["scope_id"])
    reader = IndexedReader(writer.runtime.workspace, writer.entities, tenant_id=writer.tenant_id)
    physical: set[str] = set()

    def read(kind: str, identity: str) -> dict[str, Any] | None:
        check()
        if kind != "artifact":
            raise AMLError("binding_mismatch", "Freeze closure cannot read business content")
        raw = writer.get_row(kind, identity)
        if raw is None:
            raise AMLError("reference_missing", "Freeze closure lost an original Artifact")
        row = ArtifactRowV1.model_validate(raw)
        if (row.object_id, row.tenant_id, row.scope_kind, row.scope_id, row.created_by_run_id) != (
            identity, writer.tenant_id, scope.scope_kind, scope.scope_id, owner["object_id"],
        ):
            raise AMLError("binding_mismatch", "Freeze closure crossed its original owner")
        physical.add(identity)
        if observe is not None:
            observe(identity, raw)
        check()
        return raw

    blocks: dict[int, ArtifactId] = {}
    input_ref: ArtifactId | None = None
    actual_input: FinalizeOwnerRunInputV1 | None = None
    after: str | None = None
    query = ArtifactRows(created_by_run_id=owner["object_id"])
    total = 0
    while rows := reader.page(query, after=after):
        for raw in rows:
            check()
            total += 1
            kind = raw["artifact_kind"]
            if kind == "part":
                continue
            reference = ArtifactId(raw["object_id"])
            if kind == "manifest":
                manifest = read_logical_artifact(read, reference, "manifest")
                kind = manifest.model_dump()["logical_kind"]
            if kind not in {"decision_set", "run_input"}:
                raise AMLError(
                    "unsupported_variant", "Unselected Finalize has other work artifacts",
                )
            value = read_logical_artifact(read, reference, kind)
            if isinstance(value, FinalizeSourceInventoryBlockV1):
                if value.block_no in blocks:
                    raise AMLError(
                        "parity_mismatch", "Freeze closure has competing inventory blocks",
                    )
                blocks[value.block_no] = reference
            elif isinstance(value, FinalizeOwnerRunInputV1) and actual_input is None:
                input_ref, actual_input = reference, value
            else:
                raise AMLError(
                    "unsupported_variant", "Freeze closure has another logical role",
                )
        after = rows[-1]["object_id"]
    if len(physical) != total:
        raise AMLError("parity_mismatch", "Freeze closure has unattached physical parts")
    if owner["input_ref"] is not None and owner["input_ref"] != input_ref or (
        selected is not None and selected != actual_input
    ):
        raise AMLError("binding_mismatch", "Freeze closure changed the selected original input")
    stream = writer.content_array_binding("finalize-source-inventory")
    previous: ArtifactId | None = None
    first: FinalizeSourceInventoryBlockV1 | None = None
    last: FinalizeSourceInventoryBlockV1 | None = None
    for number in range(len(blocks)):
        reference = blocks.get(number)
        if reference is None:
            raise AMLError("parity_mismatch", "Freeze inventory has a gap in its original chain")
        value = read_logical_artifact(read, reference, "decision_set")
        if not isinstance(value, FinalizeSourceInventoryBlockV1):
            raise AMLError("binding_mismatch", "Freeze inventory changed its logical role")
        if first is None:
            first = value
        if value.previous_ref != previous or value.owner_run_id != owner["object_id"] or (
            value.finalize_cycle, value.freeze_time, value.root_input_heads_digest,
            value.receipt_high_watermark, value.base_ref, value.base_count, value.removed_ids,
        ) != (
            first.finalize_cycle, first.freeze_time, first.root_input_heads_digest,
            first.receipt_high_watermark, first.base_ref, first.base_count, first.removed_ids,
        ) or (
            value.is_head and number != len(blocks) - 1
        ):
            raise AMLError("parity_mismatch", "Freeze inventory changed its original chain roots")
        for member in value.records:
            if (member.header.tenant_id, member.header.scope_kind, member.header.scope_id) != (
                writer.tenant_id, scope.scope_kind, scope.scope_id,
            ):
                raise AMLError("scope_denied", "Freeze inventory contains another source scope")
            stream.append(member)
        if value.is_head:
            def base_observe(identity: str, raw: dict[str, Any]) -> None:
                if observe is not None:
                    observe(identity, raw)

            stream = writer.content_array_binding("finalize-source-inventory")
            for member in _inventory_snapshot(
                writer, scope, reference, value.total_record_count,
                check=check, observe=base_observe,
            ):
                if member.header.expires_at is not None and (
                    member.header.expires_at <= value.freeze_time
                ) or member.header.ingested_commit_seq > value.receipt_high_watermark:
                    raise AMLError("parity_mismatch", "Inventory snapshot changed its eligibility")
                stream.append(member)
        if (value.total_record_count, value.cumulative_digest) != (stream.count, stream.digest()):
            raise AMLError("parity_mismatch", "Freeze inventory changed its cumulative binding")
        previous, last = reference, value
    cycle = None if first is None else first.finalize_cycle
    if actual_input is not None:
        cycle = actual_input.business_key.finalize_cycle
        if (actual_input.source_inventory_ref, actual_input.source_inventory_count,
            actual_input.source_inventory_digest) != (previous, stream.count, stream.digest()) or (
            last is not None and not last.is_head
        ) or first is not None and (
            first.finalize_cycle, first.freeze_time, first.root_input_heads_digest,
            first.receipt_high_watermark,
        ) != (cycle, actual_input.freeze_time, actual_input.input_heads_digest,
              actual_input.receipt_high_watermark):
            raise AMLError("parity_mismatch", "Frozen input changed its complete source inventory")
    if cycle is not None:
        family, digest, _ = admission_identity(finalize_admission(
            principal_id=owner["principal_id"], scope=scope, cycle=cycle,
        ), writer.bind_content)
        if (family, digest) != (owner["idempotency_key"], owner["admission_digest"]):
            raise AMLError("binding_mismatch", "Freeze inventory changed the actual owner cycle")
    check()


def _inventory_deadline(
    writer: Writer, reader: IndexedReader, scope: ScopeV1, check: Callable[[], None],
    query: FinalizeEvidenceQuery,
) -> int | None:
    """Read only source headers before any original-time body scan, including partial resumes.

    The deadline is the earliest expiry among the freeze's eligible rows,
    read from the same support headers its digest and inventory use.
    """
    earliest: int | None = None
    check()
    for header in support_headers(writer, reader, query):
        if (header.tenant_id, header.scope_kind, header.scope_id) != (
            writer.tenant_id, scope.scope_kind, scope.scope_id,
        ):
            raise AMLError("scope_denied", "Freeze source header crossed its exact scope")
        if header.state == "active" and header.expires_at is not None and (
            header.expires_at > query.freeze_time
        ):
            earliest = (
                header.expires_at if earliest is None else min(earliest, header.expires_at)
            )
    check()
    return earliest


def _inventory_origin(
    writer: Writer, ticket: FinalizeLeaseTicket, guard: SnapshotReadGuard,
    reader: IndexedReader, scope: ScopeV1, at_time: int,
) -> tuple[int, int, FinalizeSourceInventoryBlockV1 | None]:
    """Resume only the actual first prewrite; a new clock cannot replace its source set."""
    first: tuple[ArtifactId, FinalizeSourceInventoryBlockV1] | None = None
    found = False
    for reference, block in _source_inventory_blocks(writer, ticket, guard, reader, scope):
        found = True
        if block.block_no != 0:
            continue
        if first is not None and first != (reference, block):
            raise AMLError("parity_mismatch", "Finalize has competing source inventory roots")
        first = reference, block
    if first is None:
        if found:
            raise AMLError("parity_mismatch", "Source inventory lost its original first block")
        return at_time, guard.metadata.generation, None
    if first[1].freeze_time > at_time:
        raise AMLError("binding_mismatch", "Source inventory freeze time is in the future")
    if first[1].receipt_high_watermark > guard.metadata.generation:
        raise AMLError("binding_mismatch", "Source inventory receipt cutoff is from the future")
    return first[1].freeze_time, first[1].receipt_high_watermark, first[1]


@dataclass(frozen=True, slots=True)
class _InventoryMember:
    """One source inventory member and its two serializations, without its position.

    A member differs between Adds only by ``index``: every new row ahead of it
    in the support order moves it. ``canonical`` splices the position into the
    member's canonical JSON, whose top-level keys are sorted and among which
    ``index`` occurs once (no header field is named ``index``), so the bytes
    equal ``canonical_json`` of the member at that position.
    """

    member: FinalizeSourceInventoryMemberV1
    """The member at the position it was first projected at; copy before use."""
    encoded: str
    """``model_dump_json(exclude={"index"})``, the inventory snapshot's form."""
    head: str
    tail: str

    @classmethod
    def of(cls, member: FinalizeSourceInventoryMemberV1) -> _InventoryMember:
        """Serialize ``member`` once."""
        marker = f',"index":{member.index},'
        canonical = canonical_json(member)
        head, found, tail = canonical.partition(marker)
        if not found or marker in tail:
            raise AMLError("schema_invalid", "Inventory member has no single index")
        return cls(member, member.model_dump_json(exclude={"index"}),
                   head + ',"index":', "," + tail)

    def canonical(self, index: int) -> str:
        """``canonical_json`` of the member at ``index``."""
        return f"{self.head}{index}{self.tail}"


def _source_inventory_members(
    writer: Writer, reader: IndexedReader, query: FinalizeEvidenceQuery,
    check: Callable[[], None],
) -> Iterator[_InventoryMember]:
    """Project original rows while the actual freeze still authorizes their content.

    The rows are the support headers; each row-state digest comes from the
    scope index or the binding memo, and only rows both miss are read in full
    (``row_elements``). A member and its serializations are memoized per row
    and receipt coordinates, so a late Add serializes only its new members.
    """
    session_id: str | None = None
    orders: dict[str, RequestOrder] = {}
    partitions: dict[tuple[str, str], int] = {}
    session_time: int | None = None
    # Each trust tier lists the scope's sessions again; one scan derives each
    # session's order once.
    sessions: dict[str, tuple[
        dict[str, RequestOrder], int | None, dict[tuple[str, str], int],
    ]] = {}
    check()
    token = writer.storage_epoch
    headers = support_headers(writer, reader, query)
    elements = row_elements(writer, reader, query, headers, check)
    check()
    member_memo = writer.binding_memo()
    for index, (header, (digest, _element)) in enumerate(zip(headers, elements, strict=True)):
        if header.session_id != session_id:
            check()
            session_id = header.session_id
            known = sessions.get(session_id)
            if known is None:
                orders = load_request_order(
                    writer, query.scope, session_id, at_time=query.freeze_time,
                    input_high_watermark=query.receipt_high_watermark,
                )[0]
                session_time = min((value.earliest_source_time
                                    for value in orders.values()
                                    if value.earliest_source_time is not None), default=None)
                partitions = checkpoint_partitions(
                    {session_id: session_time},
                    {(session_id, key): (*value.first, value.chunk_event_time)
                     for key, value in orders.items()},
                    {(session_id, key): (value.received_commit_seq,
                                         value.source_time_present)
                     for key, value in orders.items()},
                )
                sessions[session_id] = orders, session_time, partitions
            else:
                orders, session_time, partitions = known
        order = orders.get(header.request_id)
        if order is None:
            raise AMLError("parity_mismatch", "Frozen inventory lost its original receipt")
        member_key = ("source-inventory-member", header.object_id, order, session_time,
                      partitions.get((header.session_id, header.request_id), 0))
        cached_member = member_memo.lookup(member_key)
        if cached_member is not MISSING:
            yield cast(_InventoryMember, cached_member)
            continue
        key = message_key(
            header, session_event_time=session_time,
            chunk_event_time=order.chunk_event_time, coordinate_version=2,
            received_commit_seq=order.received_commit_seq,
            source_time_present=order.source_time_present,
            receipt_partition_seq=partitions.get((header.session_id, header.request_id), 0),
        )
        key_type = PlanMessageRangeKeyV2 if isinstance(key, MessageRangeKeyV2) else (
            PlanMessageRangeKeyV1
        )
        member = FinalizeSourceInventoryMemberV1(
            index=index, key=key_type.model_validate(key.model_dump(mode="python")),
            row_binding_digest=digest, header=header,
        )
        projected = _InventoryMember.of(member)
        member_memo.store(member_key, token, (("evidence", header.object_id),), projected,
                          value_bytes=512 + 2 * (len(projected.encoded) + len(projected.head)
                                                 + len(projected.tail)))
        yield projected
    check()


def _source_inventory(
    writer: Writer, ticket: FinalizeLeaseTicket, guard: SnapshotReadGuard,
    grant: PrincipalGrant, reader: IndexedReader, scope: ScopeV1,
    freeze_time: int, heads: ContentBindingDigest, receipt_high_watermark: int,
    stored: FinalizeOwnerRunInputV1 | None, artifact_generation: int,
    clock: Callable[[], int],
    check: Callable[[], None],
    base_keys: set[tuple[str, str]] | None = None,
    original: FinalizeSourceInventoryBlockV1 | None = None,
) -> tuple[ArtifactId | None, int, ContentBindingDigest, int]:
    """Prewrite bounded metadata or compare the already selected original chain read-only."""
    stream = writer.content_array_binding("finalize-source-inventory")
    records = list(_source_inventory_members(
        writer, reader,
        FinalizeEvidenceQuery(
            tenant_id=writer.tenant_id, scope=scope, freeze_time=freeze_time,
            receipt_high_watermark=receipt_high_watermark,
        ),
        check,
    ))
    # Trust tiers interleave new sources with historical members. Persist only
    # changed identities; expansion restores the original support order.
    encoded = {record.member.header.object_id: record.encoded for record in records}
    # One scope's snapshot is megabytes late in a long history: the scan table,
    # whose whole budget is 16 MB, evicted it, or the census evicted it, on
    # every Add, and every Add then read the prior inventory again.
    memo = writer.binding_memo()
    memo_key = ("source-inventory-snapshot", scope.scope_kind, scope.scope_id)
    cached = memo.lookup(memo_key)
    snapshot: tuple[str, ArtifactId, dict[str, str], tuple[tuple[str, str], ...]] | None = (
        None if cached is MISSING else cast(
            tuple[str, ArtifactId, dict[str, str], tuple[tuple[str, str], ...]], cached,
        )
    )
    base_ref: ArtifactId | None = None
    base_count = 0
    removed_ids: list[str] = []
    retained: set[str] = set()
    dependencies: set[tuple[str, str]] = {MEMBERSHIP_ANY}
    base_token = writer.storage_epoch

    def observe(identity: str, _raw: dict[str, Any]) -> None:
        dependencies.add(("artifact", identity))

    if original is not None:
        base_ref, base_count = original.base_ref, original.base_count
        removed_ids = list(original.removed_ids)
        if base_ref is not None:
            old = {member.header.object_id: member.model_dump_json(exclude={"index"})
                   for member in _inventory_snapshot(
                       writer, scope, base_ref, base_count, check=check, observe=observe,
                   )}
            retained = old.keys() - set(removed_ids)
            if not set(removed_ids) <= old.keys() or any(
                encoded.get(identity) != old[identity] for identity in retained
            ):
                raise AMLError("parity_mismatch", "Original inventory base changed its members")
    else:
        if snapshot is None or snapshot[0] == ticket.owner_run_id:
            owner = writer.get_row("run", ticket.owner_run_id)
            assert owner is not None
            prior = reader.latest_finalize_input(scope, before=owner["created_at"])
            if prior is not None:
                def read(kind: str, identity: str) -> dict[str, Any] | None:
                    raw = writer.get_row(kind, identity)
                    if raw is None or (raw["tenant_id"], raw["scope_kind"], raw["scope_id"]) != (
                        writer.tenant_id, scope.scope_kind, scope.scope_id,
                    ):
                        raise AMLError("binding_mismatch", "Prior inventory changed its scope")
                    observe(identity, raw)
                    return raw

                prior_input = read_logical_artifact(read, ArtifactId(prior), "run_input")
                if not isinstance(prior_input, FinalizeOwnerRunInputV1):
                    raise AMLError("binding_mismatch", "Prior inventory has another input role")
                if prior_input.source_inventory_ref is not None:
                    old_members = tuple(_inventory_snapshot(
                        writer, scope, prior_input.source_inventory_ref,
                        prior_input.source_inventory_count, check=check, observe=observe,
                    ))
                    old_binding = writer.content_array_binding("finalize-source-inventory")
                    for index, member in enumerate(old_members):
                        old_binding.append(member.model_copy(update={"index": index}))
                    if old_binding.digest() != prior_input.source_inventory_digest:
                        raise AMLError("parity_mismatch", "Prior inventory changed its binding")
                    snapshot = (
                        "", prior_input.source_inventory_ref,
                        {member.header.object_id: member.model_dump_json(exclude={"index"})
                         for member in old_members},
                        tuple(sorted(dependencies)),
                    )
        if snapshot is not None and snapshot[0] != ticket.owner_run_id:
            retained = {identity for identity, value in snapshot[2].items()
                        if encoded.get(identity) == value}
            if retained:
                base_ref, base_count = snapshot[1], len(snapshot[2])
                removed_ids = sorted(snapshot[2].keys() - retained)
                dependencies.update(snapshot[3])
    if not writer.borrowed_rows_unchanged(dependencies, base_token):
        raise AMLError("stale_evidence", "Inventory base changed during discovery")
    if base_keys is not None:
        base_keys.update(dependencies)
    for index, record in enumerate(records):
        stream.append_canonical(record.canonical(index))
    delta = iter(record.member.model_copy(update={"index": index}) for index, record in enumerate(
        record for record in records if record.member.header.object_id not in retained
    ))
    delta_binding = writer.content_array_binding("finalize-source-inventory")
    previous: ArtifactId | None = None
    number = 0
    inventory = _SourceInventoryIndex(writer, ticket, guard, reader, scope)
    block = list(islice(delta, 128))
    while block or number == 0 and retained:
        following = list(islice(delta, 128))
        for record in block:
            delta_binding.append(record)
        binding = delta_binding if following else stream
        body = FinalizeSourceInventoryBlockV1(
            finalize_source_inventory=1, owner_run_id=ticket.owner_run_id,
            finalize_cycle=ticket.lease_generation, freeze_time=freeze_time,
            receipt_high_watermark=receipt_high_watermark,
            root_input_heads_digest=heads, previous_ref=previous, block_no=number,
            records=block, record_count=len(block), total_record_count=binding.count,
            cumulative_digest=binding.digest(), is_head=not following,
            base_ref=base_ref, base_count=base_count, removed_ids=removed_ids,
        )
        existing = inventory.find(body)
        if stored is not None:
            if existing is None:
                raise AMLError("parity_mismatch", "Selected source inventory omits original rows")
            previous = existing
        else:
            # The Writer discovers outside its mutex, then returns only the
            # generation produced by this exact prewrite under its final fence.
            previous, artifact_generation = writer.put_finalize_source_inventory_block(
                ticket, guard, body, grant=grant, clock=clock,
                artifact_generation=artifact_generation, inventory_index=inventory,
            )
            if existing is not None and previous != existing:
                raise AMLError("parity_mismatch", "Source inventory failed exact create-once reuse")
        block, number = following, number + 1
    if number == 0 and next(
        _source_inventory_blocks(writer, ticket, guard, reader, scope), None,
    ) is not None:
        raise AMLError("parity_mismatch", "An empty input cannot replace a source inventory")
    if stored is not None and (
        (previous, stream.count, stream.digest()) != (
            stored.source_inventory_ref, stored.source_inventory_count,
            stored.source_inventory_digest,
        )
    ):
        raise AMLError("parity_mismatch", "Selected source inventory has another complete head")
    if not writer.borrowed_rows_unchanged(dependencies, base_token):
        raise AMLError("stale_evidence", "Inventory base changed before selection")
    if previous is not None:
        # Physical fixed rows only: adding this owner's selected input later
        # does not change any source block already written by this freeze.
        dependencies.update(key for key in inventory.keys if key[0] == "artifact")
        proof = tuple(sorted(dependencies))
        memo.store(
            memo_key, writer.storage_epoch, proof,
            (ticket.owner_run_id, previous, encoded, proof),
            value_bytes=256 + sum(128 + len(value.encode()) for value in encoded.values())
            + 128 * len(proof),
        )
    return previous, stream.count, stream.digest(), artifact_generation


def prepare_finalize_input(
    writer: Writer, ticket: FinalizeLeaseTicket, guard: SnapshotReadGuard,
    grant: PrincipalGrant, *, clock: Callable[[], int],
) -> FinalizeInputProof:
    """Prewrite original bounded metadata and issue the compact input selection proof.

    Existing input bytes select the original time and policy. The eligible
    rows come from the scope's header index and its per-row digests; the
    inventory is a streaming HMAC. Own heartbeat and
    unrelated-scope writes do not invalidate the scan; owner, child, Artifact
    discovery and scope coordinates must remain unchanged.
    New metadata blocks are create-once prewrites without dispatch authority.
    An already selected input is compared read-only; its inventory is never
    replaced. No provider or business effect occurs here.
    """
    if writer.mutex._is_owned():  # pyright: ignore[reportPrivateUsage]
        raise AMLError("reader_busy", "Finalize input discovery must run outside the Writer mutex")
    at_time = finalize_action_time(clock)
    owner, token, artifact_generation, child_token = _state(
        writer, ticket, guard, grant, at_time,
    )
    scope = ScopeV1(scope_kind=owner["scope_kind"], scope_id=owner["scope_id"])
    reader = writer.indexed_reader()
    if reader.run_children(owner["object_id"], limit=1):
        raise AMLError("unsupported_variant", "Finalize descendants require complete plan recovery")
    for kind in ("finalize_plan", "result_index"):
        if writer.find_logical_candidates(owner["object_id"], kind):
            raise AMLError("unsupported_variant", "Finalize selected work requires plan recovery")
    stored = _selected_input(writer, owner)
    policy_ref: ArtifactId | None = None
    original: FinalizeSourceInventoryBlockV1 | None = None
    if stored is None:
        freeze_time, receipt_high_watermark, original = _inventory_origin(
            writer, ticket, guard, reader, scope, at_time,
        )
    else:
        policy_ref, freeze_time = stored[1].extraction_policy_release_ref, stored[1].freeze_time
        receipt_high_watermark = stored[1].receipt_high_watermark
        if stored[1].source_inventory_ref is not None:
            head = writer.read_logical(stored[1].source_inventory_ref, "decision_set")
            if not isinstance(head, FinalizeSourceInventoryBlockV1):
                raise AMLError("parity_mismatch", "Selected inventory lost its original head")
            original = head
    if receipt_high_watermark > guard.metadata.generation:
        raise AMLError("binding_mismatch", "Finalize receipt cutoff exceeds its held baseline")
    receipts = inspect_finalize_receipts(
        writer, scope, receipt_high_watermark=receipt_high_watermark,
        check=lambda: guard.check_operation(writer, scope),
    )
    if policy_ref is None:
        policy_ref = _governing_policy_ref(writer, receipts)
    config = _policy(writer, policy_ref)
    normalized = normalize_extraction_policy(config)
    # One freeze's support headers serve its deadline, digest and inventory.
    query = FinalizeEvidenceQuery(
        tenant_id=writer.tenant_id, scope=scope, freeze_time=freeze_time,
        receipt_high_watermark=receipt_high_watermark,
    )
    deadline = _inventory_deadline(
        writer, reader, scope,
        lambda: receipts.validate(writer, lambda: guard.check_operation(writer, scope)), query,
    )

    def check_content() -> None:
        receipts.validate(writer, lambda: guard.check_operation(writer, scope))
        now = finalize_action_time(clock)
        if now < at_time or deadline is not None and now >= deadline:
            raise AMLError("stale_evidence", "Source lifetime ended before input freeze completed")

    check_content()
    evidence = inspect_finalize_evidence_digest(
        writer, scope, freeze_time=freeze_time, receipt_high_watermark=receipt_high_watermark,
        check=check_content,
    )
    heads = finalize_input_heads_digest(writer, evidence)
    base_keys: set[tuple[str, str]] = set()
    inventory_ref, inventory_count, inventory_digest, artifact_generation = _source_inventory(
        writer, ticket, guard, grant, reader, scope, freeze_time, heads, receipt_high_watermark,
        None if stored is None else stored[1], artifact_generation, clock, check_content,
        base_keys, original,
    )
    final_evidence = inspect_finalize_evidence_digest(
        writer, scope, freeze_time=freeze_time, receipt_high_watermark=receipt_high_watermark,
        check=check_content,
    )
    following = _state(writer, ticket, guard, grant, finalize_action_time(clock))
    if final_evidence != evidence or following != (
        owner, token, artifact_generation, child_token,
    ):
        raise AMLError("stale_evidence", "Finalize input changed during complete discovery")
    family, admission, payload = admission_identity(finalize_admission(
        principal_id=owner["principal_id"], scope=scope, cycle=ticket.lease_generation,
    ), writer.bind_content)
    if (family, admission) != (owner["idempotency_key"], owner["admission_digest"]):
        raise AMLError("binding_mismatch", "Finalize owner differs from canonical admission")
    body = FinalizeOwnerRunInputV1(
        admission_digest=admission, payload_digest=payload,
        run_kind="finalize", worker_role="owner",
        business_key=FinalizeBusinessKeyV1(
            scope_id=scope.scope_id, finalize_cycle=ticket.lease_generation,
        ),
        extraction_policy_release_ref=policy_ref,
        extraction_config_digest=normalized.config_digest,
        extraction_policy_snapshot_digest=integrity_digest(normalized),
        freeze_time=freeze_time, finalize_input_scope_seq=token[0],
        receipt_high_watermark=receipt_high_watermark,
        evidence_count=evidence.count, evidence_rows_digest=evidence.digest,
        input_heads_digest=finalize_input_heads_digest(writer, evidence),
        source_inventory_ref=inventory_ref, source_inventory_count=inventory_count,
        source_inventory_digest=inventory_digest,
    )
    if stored is not None and stored[1] != body:
        raise AMLError("parity_mismatch", "Finalize input no longer matches its original freeze")
    proof = FinalizeInputProof(
        weakref.ref(writer), weakref.ref(guard), ticket, canonical_json(owner),
        canonical_json(body), canonical_json(config), token, artifact_generation,
        child_token, None if stored is None else stored[0],
        receipt_keys=tuple(sorted(set(receipts.keys) | base_keys)), receipt_token=receipts.token,
    )
    object.__setattr__(proof, "seal", _seal(proof))
    return proof


def validate_finalize_input(
    writer: Writer, proof: FinalizeInputProof, grant: PrincipalGrant, at_time: int,
) -> None:
    """Recheck fixed authorities before the sole Writer's input-only effect."""
    if proof.writer() is not writer or not hmac.compare_digest(proof.seal, _seal(proof)):
        raise AMLError("binding_mismatch", "Finalize input proof is not issued and unchanged")
    guard = proof.guard()
    if guard is None:
        raise AMLError("reader_busy", "Finalize input lost its registered content lifetime")
    state = _state(writer, proof.ticket, guard, grant, finalize_action_time(lambda: at_time))
    owner, token, artifact_generation, child_token = state
    if (
        canonical_json(owner) != proof.owner_json or token != proof.scope_token
        or artifact_generation != proof.artifact_generation or child_token != proof.child_token
        or not writer.borrowed_rows_unchanged(proof.receipt_keys, proof.receipt_token)
    ):
        raise AMLError("stale_evidence", "Finalize fixed attachment authorities changed")
    config = _policy(writer, proof.input.extraction_policy_release_ref)
    if canonical_json(config) != proof.policy_json:
        raise AMLError("parity_mismatch", "Finalize governing policy bytes changed")
    if proof.input_ref is not None:
        if writer.read_logical(proof.input_ref, "run_input") != proof.input:
            raise AMLError("parity_mismatch", "Finalize selected immutable input changed")
