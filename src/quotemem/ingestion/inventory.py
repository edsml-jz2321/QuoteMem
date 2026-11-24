"""Complete privacy-stable request inventories.

All reads use the existing HeavenBase owner. This module grants no execution
rights and writes neither receipts nor cursor state; the sole Writer rechecks
the returned fixed keys and current authorization at the actual transaction.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, cast

from pydantic import TypeAdapter, ValidationError

from quotemem.core.admission import session_request_family
from quotemem.core.artifacts import RequestLedgerV1
from quotemem.core.controls import ControlRowV1, RequestLedgerValueV1
from quotemem.core.errors import QuoteMemError
from quotemem.core.identity import (
    ULID,
    ArtifactId,
    ContentBindingDigest,
    ScopeId,
    base64url,
    canonical_json,
)
from quotemem.core.ingestion_limits import MAX_ADD_MESSAGES
from quotemem.core.query import ScopeV1
from quotemem.core.run_inputs import (
    MAX_ADD_SOURCE_PARTS,
    SessionInventoryDurableRunInputV2,
    SourceInventoryV1,
)
from quotemem.core.semantic import ScopeKind, ShortText
from quotemem.core.source_boundary import EvidenceBoundaryProof, MessageBoundaryProof
from quotemem.core.technical_types import ExternalId, NonNegativeInt, RequestProgressV1
from quotemem.ingestion.receipts import inspect_ledger
from quotemem.store.domain import EvidenceRowV1, SessionTargetRunRowV1, TenantId
from quotemem.store.indexed import EvidenceRows, IndexedReader, RunRows
from quotemem.store.metadata import StrictModel
from quotemem.store.row_cache import MEMBERSHIP_ANY, MISSING, membership_family, membership_key

if TYPE_CHECKING:
    from quotemem.store.writer import Writer

BindContent = Callable[[str, object], ContentBindingDigest]
SOURCE_INVENTORY_PURPOSE = "add-source-inventory"
MAX_SOURCE_PARTS = MAX_ADD_SOURCE_PARTS
MAX_CURSOR_REQUESTS = 2064
_EXTERNAL: TypeAdapter[str] = TypeAdapter(ExternalId)


class SourceCoordinateV1(StrictModel):
    """Exact immutable Evidence inventory projection; no source content or eligibility."""

    object_id: ULID
    tenant_id: TenantId
    scope_kind: ScopeKind
    scope_id: ScopeId
    session_id: ShortText
    request_id: ShortText
    ordinal: NonNegativeInt
    part_no: NonNegativeInt
    ingested_commit_seq: NonNegativeInt


def source_coordinate(row: EvidenceRowV1) -> SourceCoordinateV1:
    """Project one strict Evidence row onto its immutable inventory coordinate.

    Args:
        row: A validated Evidence row carrying its receipt stamps.

    Returns:
        The coordinate ``freeze_source_coordinates`` binds; no source content.
    """
    return SourceCoordinateV1.model_validate(
        {name: getattr(row, name) for name in SourceCoordinateV1.model_fields}
    )


def freeze_source_inventory(
    rows: Iterable[EvidenceRowV1], *, message_count: int, bind: BindContent
) -> SourceInventoryV1:
    """Bind the complete received part geometry after strict full-row validation.

    The legacy full-row consumer keeps its original validation. The shared
    projection deliberately excludes byte bounds, source content and liveness.
    """
    def coordinates() -> Iterable[SourceCoordinateV1]:
        for supplied in rows:
            yield source_coordinate(EvidenceRowV1.model_validate(supplied.model_dump(mode="json")))

    return freeze_source_coordinates(coordinates(), message_count=message_count, bind=bind)


def freeze_source_coordinates(
    rows: Iterable[SourceCoordinateV1], *, message_count: int, bind: BindContent
) -> SourceInventoryV1:
    """Bind the exact same V2 received inventory using actual metadata projections."""
    if type(message_count) is not int or not 1 <= message_count <= 256:
        raise QuoteMemError("schema_invalid",
                            "Source inventory requires the accepted message count")
    selected: list[SourceCoordinateV1] = []
    for supplied in rows:
        if len(selected) == MAX_SOURCE_PARTS:
            raise QuoteMemError("budget_exceeded",
                                "Source inventory exceeds the complete Add bound")
        selected.append(SourceCoordinateV1.model_validate(supplied.model_dump(mode="json")))
    if not selected:
        raise QuoteMemError("parity_mismatch", "Accepted request lost every source part")
    first = selected[0]
    fields = (
        "tenant_id",
        "scope_kind",
        "scope_id",
        "session_id",
        "request_id",
        "ingested_commit_seq",
    )
    if any(any(getattr(row, key) != getattr(first, key) for key in fields) for row in selected):
        raise QuoteMemError("parity_mismatch",
                            "Source inventory combines different received boundaries")
    coordinates = {(row.ordinal, row.part_no) for row in selected}
    if len(coordinates) != len(selected) or len({row.object_id for row in selected}) != len(
        selected
    ):
        raise QuoteMemError("parity_mismatch", "Source inventory repeats a coordinate or identity")
    if {row.ordinal for row in selected} != set(range(message_count)):
        raise QuoteMemError("parity_mismatch", "Source inventory lost a logical message")
    for ordinal in range(message_count):
        numbers = sorted(row.part_no for row in selected if row.ordinal == ordinal)
        if numbers != list(range(len(numbers))):
            raise QuoteMemError("parity_mismatch", "Source inventory lost an interior part")
    projection = {
        "scope_kind": first.scope_kind,
        "scope_id": first.scope_id,
        "session_id": first.session_id,
        "request_id": first.request_id,
        "message_count": message_count,
        "part_count": len(selected),
        "parts": [
            {
                "object_id": row.object_id,
                "ordinal": row.ordinal,
                "part_no": row.part_no,
                "ingested_commit_seq": row.ingested_commit_seq,
            }
            for row in sorted(selected, key=lambda row: (row.ordinal, row.part_no, row.object_id))
        ],
    }
    return SourceInventoryV1(
        part_count=len(selected), digest=bind(SOURCE_INVENTORY_PURPOSE, projection)
    )


def verify_source_inventory(
    rows: Iterable[EvidenceRowV1], receipt: SessionInventoryDurableRunInputV2, bind: BindContent
) -> tuple[EvidenceRowV1, ...]:
    """Verify complete current membership against the immutable coordinate receipt."""
    selected: list[EvidenceRowV1] = []
    for row in rows:
        if len(selected) == MAX_SOURCE_PARTS:
            raise QuoteMemError("budget_exceeded",
                                "Source inventory exceeds the complete Add bound")
        selected.append(EvidenceRowV1.model_validate(row.model_dump(mode="json")))
    observed = freeze_source_inventory(selected, message_count=receipt.message_count, bind=bind)
    if any(
        row.session_id != receipt.business_key.session_id
        or row.request_id != receipt.business_key.request_id
        or row.ingested_commit_seq != receipt.visibility_commit_seq
        for row in selected
    ):
        raise QuoteMemError("parity_mismatch",
                            "Source inventory differs from its immutable receipt")
    if receipt.source_inventory != observed:
        raise QuoteMemError("parity_mismatch", "Accepted source inventory lost or replaced a part")
    return tuple(selected)


@dataclass(frozen=True)
class InventoryDependency:
    """A captured physical row identity and tenant-bound complete-row digest."""

    entity_kind: str
    object_id: str
    digest: ContentBindingDigest


@dataclass(frozen=True)
class ReceiptBinding:
    """Selected immutable reception witness and authorization classification."""

    request_id: str
    run_id: str
    input_ref: ArtifactId
    admission_digest: ContentBindingDigest
    visibility_commit_seq: int
    confirmed_request_json: str | None


@dataclass(frozen=True)
class RequestInventoriesProof:
    """Current part eligibility and exact complete-set authorities, without text."""

    scope_json: str = field(repr=False)
    session_id: str
    at_time: int
    values: tuple[tuple[str, RequestInventory], ...]
    receipts: tuple[ReceiptBinding, ...]
    dependencies: tuple[InventoryDependency, ...]
    boundary_values: tuple[tuple[str, EvidenceBoundaryProof], ...] = ()

    @property
    def inventories(self) -> Mapping[str, RequestInventory]:
        """Return an immutable lookup consumed by the existing cursor algorithm."""
        return MappingProxyType(dict(self.values))

    @property
    def boundaries(self) -> Mapping[str, EvidenceBoundaryProof]:
        """Return original-message finality from complete receipt-proven parts."""
        return MappingProxyType(dict(self.boundary_values))


def _boundaries(
    rows: Sequence[EvidenceRowV1 | SourceCoordinateV1],
    message_count: int,
    binding: ReceiptBinding,
    inventory_digest: ContentBindingDigest,
) -> tuple[tuple[str, EvidenceBoundaryProof], ...]:
    """Derive metadata only after the caller proved the entire immutable Add set."""
    first = rows[0]
    result: list[tuple[str, EvidenceBoundaryProof]] = []
    for ordinal in range(message_count):
        parts = tuple(
            sorted((row for row in rows if row.ordinal == ordinal), key=lambda row: row.part_no)
        )
        message = MessageBoundaryProof(
            first.tenant_id,
            first.scope_kind,
            first.scope_id,
            first.session_id,
            first.request_id,
            ordinal,
            first.ingested_commit_seq,
            tuple(part.object_id for part in parts),
            binding.input_ref,
            binding.admission_digest,
            inventory_digest,
        )
        result.extend(
            (part.object_id, EvidenceBoundaryProof(message, part.object_id, part.part_no))
            for part in parts
        )
    return tuple(result)


class _Read:
    def __init__(self, writer: Writer, scope: ScopeV1) -> None:
        self.writer = writer
        self.scope = scope
        self.dependencies: dict[tuple[str, str], InventoryDependency] = {}

    def row(self, kind: str, identity: str) -> dict[str, Any]:
        try:
            raw = self.writer.get_row(kind, identity)
        except QuoteMemError as error:
            if error.code != "schema_invalid":
                raise
            raise QuoteMemError(
                "parity_mismatch", "Request inventory authority row is malformed"
            ) from error
        return self.row_from(kind, identity, raw)

    def row_from(self, kind: str, identity: str, raw: dict[str, Any] | None) -> dict[str, Any]:
        """Record one already-read row as a fixed dependency without reading it again.

        A request's Evidence page already returns each complete row; reading
        every row a second time through ``get_row`` doubled the largest
        inventory read site under normal benchmark load (2,225 second reads, 103 MB, 411 s of read
        wait in seventeen minutes).
        """
        if raw is None or raw.get("object_id") != identity:
            raise QuoteMemError("parity_mismatch",
                                "Request inventory lost a selected authority row")
        if "tenant_id" in raw and raw["tenant_id"] != self.writer.tenant_id:
            raise QuoteMemError(
                "parity_mismatch", "Request inventory authority belongs to another tenant"
            )
        key = kind, identity
        item = InventoryDependency(kind, identity, self.writer.bind_content("row-state", raw))
        if key in self.dependencies and self.dependencies[key] != item:
            raise QuoteMemError(
                "parity_mismatch", "Request inventory fixed row changed during inspection"
            )
        self.dependencies[key] = item
        return raw

    def artifact(self, reference: str, kind: str, creator: str | None = None) -> Any:
        import json

        raw = self.row("artifact", reference)
        if (raw["scope_kind"], raw["scope_id"]) != (self.scope.scope_kind, self.scope.scope_id):
            raise QuoteMemError("parity_mismatch",
                                "Request inventory Artifact crosses its exact scope")
        if creator is not None and raw["created_by_run_id"] != creator:
            raise QuoteMemError("parity_mismatch", "Receipt Artifact belongs to another producer")
        try:
            value = self.writer.read_logical(reference, kind)
        except QuoteMemError as error:
            if error.code == "unsupported_variant":
                raise
            raise QuoteMemError(
                "parity_mismatch", "Request inventory Artifact failed logical proof"
            ) from None
        if raw["artifact_kind"] == "manifest":
            for part in json.loads(raw["body"])["parts"]:
                self.row("artifact", part)
        return value


def _receipt(
    read: _Read,
    reader: IndexedReader,
    scope: ScopeV1,
    session_id: str,
    request_id: str,
    request_key: ContentBindingDigest,
    classification: RequestProgressV1,
) -> tuple[SessionInventoryDurableRunInputV2, ReceiptBinding]:
    family = session_request_family(scope, session_id, request_id)
    query = RunRows(run_kind="session_ingestion", idempotency_key=family)
    after = None
    selected: tuple[int, SessionInventoryDurableRunInputV2, ReceiptBinding] | None = None
    expected_payload = None
    expected_principal = None
    while page := reader.page(query, after=after):
        for raw in page:
            run = SessionTargetRunRowV1.model_validate(read.row("run", raw["object_id"]))
            if (
                run.commit_mode != "orchestrator"
                or run.target_kind != "session"
                or (
                    run.scope_kind != scope.scope_kind
                    or run.scope_id != scope.scope_id
                    or run.admission_digest != classification.admission_digest
                    or run.target_id != base64url(hashlib.sha256(session_id.encode()).digest())[:22]
                    or expected_principal is not None
                    and expected_principal != run.principal_id
                )
            ):
                raise QuoteMemError(
                    "parity_mismatch", "Request family differs from its reception binding"
                )
            expected_principal = run.principal_id
            if run.input_ref is None:
                continue
            body = read.artifact(run.input_ref, "run_input", run.object_id)
            if not isinstance(body, SessionInventoryDurableRunInputV2) or (
                body.request_key != request_key
                or body.admission_digest != run.admission_digest
                or body.business_key.session_id != session_id
                or body.business_key.request_id != request_id
                or body.confirmed_request != classification.confirmed_request
                or expected_payload is not None
                and expected_payload != body.payload_digest
            ):
                raise QuoteMemError(
                    "parity_mismatch", "Selected receipt differs from complete admission"
                )
            expected_payload = body.payload_digest
            binding = ReceiptBinding(
                request_id,
                run.object_id,
                run.input_ref,
                run.admission_digest,
                body.visibility_commit_seq,
                canonical_json(body.confirmed_request)
                if body.confirmed_request is not None
                else None,
            )
            if selected is None or run.run_generation > selected[0]:
                selected = run.run_generation, body, binding
        if len(page) < 128:
            break
        after = page[-1]["object_id"]
    if selected is None:
        raise QuoteMemError("parity_mismatch", "Accepted request has no surviving attached receipt")
    return selected[1], selected[2]


def _ledger_classifications(
    read: _Read, scope: ScopeV1, session_id: str,
) -> Mapping[ContentBindingDigest, str]:
    """Reuse one validated session ledger while its exact physical rows survive."""
    writer = read.writer
    memo = writer.scan_memo()
    key = ("request-classifications", scope.scope_kind, scope.scope_id, session_id)
    cached = memo.lookup(key)
    if cached is not MISSING:
        entries, dependencies = cast(tuple[
            Mapping[ContentBindingDigest, str], tuple[InventoryDependency, ...],
        ], cached)
        read.dependencies.update((
            (value.entity_kind, value.object_id), value,
        ) for value in dependencies)
        return entries
    token = writer.storage_epoch
    selected = _Read(writer, scope)
    result = _read_ledger_classifications(selected, scope, session_id)
    read.dependencies.update(selected.dependencies)
    # A caller normally needs one request. Re-parsing the complete session
    # ledger for each lookup would make the census quadratic in request count.
    entries = MappingProxyType({key: value.model_dump_json() for key, value in result.items()})
    dependencies = tuple(selected.dependencies.values())
    memo.store(key, token, (MEMBERSHIP_ANY, *selected.dependencies),
               (entries, dependencies), value_bytes=sum(
                   len(value.encode()) + 256 for value in entries.values()
               ) + 512 * len(dependencies))
    return entries


def _read_ledger_classifications(
    read: _Read, scope: ScopeV1, session_id: str,
) -> dict[ContentBindingDigest, RequestProgressV1]:
    writer = read.writer
    session_hash = base64url(hashlib.sha256(session_id.encode()).digest())[:22]
    ledger_key = f"request_ledger:{scope.scope_id}:{session_hash}"
    ledger = writer.get_control(ledger_key)
    if ledger is None:
        raise QuoteMemError("parity_mismatch", "Session lost its reception classification")
    control = ControlRowV1.model_validate(read.row("control", ledger["object_id"]))
    if control.key != ledger_key or not isinstance(control.value, RequestLedgerValueV1):
        raise QuoteMemError("parity_mismatch", "Request inventory selected another ledger Control")
    value = control.value
    found: dict[ContentBindingDigest, RequestProgressV1] = {}

    def observe(key: ContentBindingDigest, progress: RequestProgressV1) -> None:
        found[key] = progress

    def block(reference: ArtifactId) -> RequestLedgerV1:
        value = read.artifact(reference, "request_ledger")
        if not isinstance(value, RequestLedgerV1):
            raise QuoteMemError("parity_mismatch", "Reception chain has another strict body kind")
        return value

    inspect_ledger(
        value,
        writer.bind_content("add-request-key", {"lookup": "inventory classification"}),
        block,
        writer.bind_content,
        observe=observe,
    )
    return found


def _classifications(
    read: _Read, scope: ScopeV1, session_id: str, wanted: set[str],
) -> tuple[dict[ContentBindingDigest, str], dict[str, RequestProgressV1]]:
    request_keys = {read.writer.bind_content("add-request-key", {
        "scope_id": scope.scope_id, "session_id": session_id, "request_id": request,
    }): request for request in wanted}
    accepted = _ledger_classifications(read, scope, session_id)
    found = {request: RequestProgressV1.model_validate_json(accepted[key])
             for key, request in request_keys.items() if key in accepted}
    if set(found) != wanted:
        raise QuoteMemError("parity_mismatch", "Cursor input lacks an accepted reception binding")
    return request_keys, found


@dataclass(frozen=True)
class SessionRequestDiscovery:
    """Complete keyed request-set equality; actual caller still owns the guard and lifetime."""

    request_ids: tuple[str, ...]
    dependencies: tuple[InventoryDependency, ...]


def discover_session_requests(
    writer: Writer, scope: ScopeV1, session_id: str, *,
    additional_request_ids: Iterable[str] = (),
) -> SessionRequestDiscovery:
    """Join all-state request metadata to the complete actual reception ledger.

    Optional historical cursor IDs only broaden discovery; every resulting ID
    must still belong to the exact accepted keyed set. Missing all source locators
    therefore cannot silently disappear. Per-request V2 geometry and original
    message ordering remain separate owners, and this value grants no execution.
    """
    scope = ScopeV1.model_validate(scope.model_dump(mode="json"))
    session_id = _EXTERNAL.validate_python(session_id)
    read = _Read(writer, scope)
    accepted = _ledger_classifications(read, scope, session_id)
    requests = {_EXTERNAL.validate_python(value) for value in additional_request_ids}
    reader = IndexedReader(
        writer.runtime.workspace,
        writer.entities,
        tenant_id=writer.tenant_id,
    )
    after = None
    while page := reader.session_request_ids(scope, session_id, after=after):
        requests.update(page)
        after = page[-1]
    keyed = {writer.bind_content("add-request-key", {
        "scope_id": scope.scope_id, "session_id": session_id, "request_id": value,
    }): value for value in requests}
    if len(keyed) != len(requests) or set(keyed) != set(accepted):
        raise QuoteMemError(
            "parity_mismatch", "Session reception set is incomplete or has extra requests",
        )
    return SessionRequestDiscovery(
        tuple(sorted(requests)),
        tuple(read.dependencies[key] for key in sorted(read.dependencies)),
    )


def load_request_inventories(
    writer: Writer, scope: ScopeV1, session_id: str, request_ids: Iterable[str], *, at_time: int,
    input_high_watermark: int | None = None,
) -> RequestInventoriesProof:
    """Read complete current requests and their independently persisted witnesses.

    Call outside the commit mutex under the caller's scoped privacy read lifetime.
    The result's ledger row dependency guards both matched classifications and
    absence; exact Evidence predicates are paged to exhaustion without a state
    filter so redacted parts remain counted. Provider/lease/epoch authorization
    stays with the caller and must be repeated at its actual action boundary.
    """
    try:
        scope = ScopeV1.model_validate(scope.model_dump(mode="json"))
        session_id = _EXTERNAL.validate_python(session_id)
        if type(at_time) is not int or at_time < 0:
            raise QuoteMemError("invalid_request",
                                "Request inventory requires a trusted attempt time")
        if input_high_watermark is not None and (
            type(input_high_watermark) is not int or input_high_watermark < 0
        ):
            raise QuoteMemError("invalid_request", "Request inventory has an invalid receipt bound")
        wanted: set[str] = set()
        for identifier in request_ids:
            wanted.add(_EXTERNAL.validate_python(identifier))
            if len(wanted) > MAX_CURSOR_REQUESTS:
                raise QuoteMemError(
                    "budget_exceeded", "Cursor request inventory exceeds its bounded input"
                )
        read = _Read(writer, scope)
        reader = IndexedReader(
            writer.runtime.workspace,
            writer.entities,
            tenant_id=writer.tenant_id,
        )
        request_keys, found = _classifications(read, scope, session_id, wanted)
        inventories: list[tuple[str, RequestInventory]] = []
        receipts: list[ReceiptBinding] = []
        boundaries: list[tuple[str, EvidenceBoundaryProof]] = []
        for key, request_id in sorted(request_keys.items(), key=lambda pair: pair[1]):
            receipt, binding = _receipt(
                read, reader, scope, session_id, request_id, key, found[request_id]
            )
            if input_high_watermark is not None and (
                binding.visibility_commit_seq > input_high_watermark
            ):
                continue
            query = EvidenceRows(
                scope_kind=scope.scope_kind,
                scope_id=scope.scope_id,
                session_id=session_id,
                request_id=request_id,
            )
            rows: list[EvidenceRowV1] = []
            after = None
            while page := reader.page(query, after=after):
                for raw in page:
                    if len(rows) == MAX_SOURCE_PARTS:
                        raise QuoteMemError(
                            "parity_mismatch", "Stored request exceeds its accepted part bound"
                        )
                    rows.append(EvidenceRowV1.model_validate(
                        read.row_from("evidence", raw["object_id"], raw)
                    ))
                if len(page) < 128:
                    break
                after = page[-1]["object_id"]
            selected = verify_source_inventory(rows, receipt, writer.bind_content)
            inventory = freeze_request_inventory(
                selected,
                message_count=receipt.message_count,
                now_ms=at_time,
                bind=writer.bind_content,
            )
            inventories.append((request_id, inventory))
            receipts.append(binding)
            witness = freeze_source_inventory(
                selected, message_count=receipt.message_count, bind=writer.bind_content,
            )
            boundaries.extend(_boundaries(
                selected, receipt.message_count, binding, witness.digest,
            ))
        return RequestInventoriesProof(
            canonical_json(scope),
            session_id,
            at_time,
            tuple(inventories),
            tuple(receipts),
            tuple(read.dependencies[key] for key in sorted(read.dependencies)),
            tuple(boundaries),
        )
    except QuoteMemError:
        raise
    except (ValidationError, KeyError, ValueError):
        raise QuoteMemError(
            "parity_mismatch", "Request inventory failed its strict stored proof"
        ) from None


@dataclass(frozen=True)
class SourceCoordinateDependency:
    """Actual nine-field source projection, distinct from a complete-row digest."""

    object_id: str
    digest: ContentBindingDigest


@dataclass(frozen=True)
class MetadataRequestInventories:
    """Complete V2 receipt geometry with no Evidence body materialization."""

    values: tuple[tuple[str, tuple[SourceCoordinateV1, ...]], ...]
    receipts: tuple[ReceiptBinding, ...]
    dependencies: tuple[InventoryDependency, ...]
    coordinate_dependencies: tuple[SourceCoordinateDependency, ...]
    boundary_values: tuple[tuple[str, EvidenceBoundaryProof], ...] = ()


_METADATA_INVENTORIES = TypeAdapter(MetadataRequestInventories)


def _request_coordinates(
    read: _Read, reader: IndexedReader, scope: ScopeV1, session_id: str,
    request_id: str, key: ContentBindingDigest, classification: RequestProgressV1,
    input_high_watermark: int | None,
) -> MetadataRequestInventories:
    """Load one classified request without copying the caller's shared ledger proof."""
    writer = read.writer
    token = writer.storage_epoch
    memo = writer.scan_memo()
    memo_key = ("request-coordinates", scope.scope_kind, scope.scope_id,
                session_id, request_id, canonical_json(classification))
    try:
        cached = memo.lookup(memo_key)
        if cached is not MISSING:
            result = _METADATA_INVENTORIES.validate_json(cast(str, cached))
            read.dependencies.update((
                (value.entity_kind, value.object_id), value,
            ) for value in result.dependencies)
            if input_high_watermark is not None and (
                result.receipts[0].visibility_commit_seq > input_high_watermark
            ):
                return MetadataRequestInventories((), (), result.dependencies, ())
            return result
        request_read = _Read(writer, scope)
        receipt, binding = _receipt(
            request_read, reader, scope, session_id, request_id, key, classification,
        )
        read.dependencies.update(request_read.dependencies)
        dependencies = tuple(request_read.dependencies[k] for k in sorted(
            request_read.dependencies,
        ))
        if input_high_watermark is not None and (
            binding.visibility_commit_seq > input_high_watermark
        ):
            return MetadataRequestInventories((), (), dependencies, ())
        rows: list[SourceCoordinateV1] = []
        coordinate_dependencies: list[SourceCoordinateDependency] = []
        after = None
        while page := reader.session_source_coordinates(
            scope, session_id, request_id, after=after,
        ):
            for raw in page:
                if len(rows) == MAX_SOURCE_PARTS:
                    raise QuoteMemError("parity_mismatch", "Stored request exceeds its Add bound")
                row = SourceCoordinateV1.model_validate(raw)
                if (row.tenant_id, row.scope_kind, row.scope_id, row.session_id,
                    row.request_id, row.ingested_commit_seq) != (
                    writer.tenant_id, scope.scope_kind, scope.scope_id, session_id,
                    request_id, receipt.visibility_commit_seq,
                ):
                    raise QuoteMemError("parity_mismatch", "Source metadata changed its receipt")
                rows.append(row)
                coordinate_dependencies.append(SourceCoordinateDependency(
                    row.object_id, writer.bind_content("source-coordinate", raw),
                ))
            after = page[-1]["object_id"]
        observed = freeze_source_coordinates(
            rows, message_count=receipt.message_count, bind=writer.bind_content,
        )
        if observed != receipt.source_inventory:
            raise QuoteMemError("parity_mismatch", "Source inventory lost or replaced a part")
        result = MetadataRequestInventories(
            ((request_id, tuple(sorted(
                rows, key=lambda row: (row.ordinal, row.part_no, row.object_id),
            ))),), (binding,), dependencies, tuple(coordinate_dependencies),
            _boundaries(rows, receipt.message_count, binding, observed.digest),
        )
        memo.store(memo_key, token, (
            MEMBERSHIP_ANY, membership_family("evidence"), membership_family("run"),
            membership_key("evidence", canonical_json([
                scope.scope_kind, scope.scope_id, session_id, request_id,
            ])),
            membership_key("run", session_request_family(scope, session_id, request_id)),
            *request_read.dependencies,
            *(("evidence", row.object_id) for row in rows),
        ), _METADATA_INVENTORIES.dump_json(result).decode("utf-8"))
        return result
    except QuoteMemError:
        raise
    except (ValidationError, KeyError, ValueError):
        raise QuoteMemError("parity_mismatch", "Source metadata failed its strict proof") from None


def load_request_coordinates(
    writer: Writer, scope: ScopeV1, session_id: str, request_ids: Iterable[str],
    *, input_high_watermark: int | None = None,
) -> MetadataRequestInventories:
    """Read full receipt-proven V2 geometry using metadata-only Evidence queries.

    This read grants no close or execution authority. The caller must hold its
    actual scoped guard, bind complete discovery to it, and validate the returned
    fixed metadata projections at the final action.
    """
    try:
        if input_high_watermark is not None and (
            type(input_high_watermark) is not int or input_high_watermark < 0
        ):
            raise QuoteMemError("invalid_request",
                                "Source coordinates have an invalid receipt bound")
        scope = ScopeV1.model_validate(scope.model_dump(mode="json"))
        session_id = _EXTERNAL.validate_python(session_id)
        wanted: set[str] = set()
        for identifier in request_ids:
            wanted.add(_EXTERNAL.validate_python(identifier))
            if len(wanted) > MAX_CURSOR_REQUESTS:
                raise QuoteMemError("budget_exceeded", "Inventory batch exceeds its bounded input")
        # Cutoffs select receipts, not coordinates. Cache one request at a time
        # below, so the next finalize reuses every unchanged past request even
        # though its tenant commit cutoff advanced.
        read = _Read(writer, scope)
        reader = IndexedReader(
            writer.runtime.workspace,
            writer.entities,
            tenant_id=writer.tenant_id,
        )
        request_keys, found = _classifications(read, scope, session_id, wanted)
        values: list[tuple[str, tuple[SourceCoordinateV1, ...]]] = []
        receipts: list[ReceiptBinding] = []
        dependencies: list[SourceCoordinateDependency] = []
        boundaries: list[tuple[str, EvidenceBoundaryProof]] = []
        for key, request_id in sorted(request_keys.items(), key=lambda pair: pair[1]):
            inventory = _request_coordinates(
                read, reader, scope, session_id, request_id, key, found[request_id],
                input_high_watermark,
            )
            values.extend(inventory.values)
            receipts.extend(inventory.receipts)
            dependencies.extend(inventory.coordinate_dependencies)
            boundaries.extend(inventory.boundary_values)
        return MetadataRequestInventories(
            tuple(values), tuple(receipts),
            tuple(read.dependencies[key] for key in sorted(read.dependencies)),
            tuple(dependencies),
            tuple(boundaries),
        )
    except QuoteMemError:
        raise
    except (ValidationError, KeyError, ValueError):
        raise QuoteMemError("parity_mismatch", "Source metadata failed its strict proof") from None


def load_request_boundaries(
    writer: Writer, scope: ScopeV1, session_id: str, request_id: str, *, at_time: int,
) -> Mapping[str, EvidenceBoundaryProof]:
    """Prove message membership without materializing unrelated source content.

    V2 receipts independently bind the complete immutable coordinate set. Their
    geometry proof needs no text, vector, or action-time eligibility inventory.
    The scoped caller still checks the actual selected rows, TTL and authority;
    a boundary grants none of those rights.
    Immutable coordinates may be reused while their exact row and request
    membership keys remain current; no action authority is cached or persisted.
    """
    if type(at_time) is not int or at_time < 0:
        raise QuoteMemError("invalid_request", "Request boundaries require a trusted attempt time")
    proof = load_request_coordinates(writer, scope, session_id, [request_id])
    return MappingProxyType(dict(proof.boundary_values))


@dataclass(frozen=True)
class InventoryPart:
    """One present immutable Add coordinate and its current source eligibility proof."""

    evidence_id: str
    ordinal: int
    part_no: int
    row_binding_digest: ContentBindingDigest
    usable: bool


@dataclass(frozen=True)
class RequestInventory:
    """A complete caller-proven Add set; retain these prerequisites until Writer commit.

    The caller must prove the full indexed Add set and durable message count,
    never substitute a scan of only processable or currently visible sources.
    All rows must still exist, including fixed tombstones. No missing-row waiver
    or caller-supplied eligibility Boolean is accepted by the factory.
    """

    tenant_id: str
    scope_kind: str
    scope_id: str
    session_id: str
    request_id: str
    ingested_commit_seq: int
    message_count: int
    now_ms: int
    parts: tuple[InventoryPart, ...]

    def completed_prefix(self, completed: set[tuple[int, int]]) -> int:
        """Count only consecutive fully consumed messages, never omitted sources."""
        maximum = -1
        for ordinal in range(self.message_count):
            if any(
                (part.ordinal, part.part_no) not in completed
                for part in self.parts
                if part.ordinal == ordinal
            ):
                break
            maximum = ordinal
        return maximum

    def exhausted(self, completed: set[tuple[int, int]]) -> bool:
        """True only once every still-usable part was actually consumed."""
        return all(
            not part.usable or (part.ordinal, part.part_no) in completed for part in self.parts
        )


def freeze_request_inventory(
    rows: Iterable[EvidenceRowV1],
    *,
    message_count: int,
    now_ms: int,
    bind: BindContent,
) -> RequestInventory:
    """Bind the complete present Add set at a trusted monotone attempt time.

    Fresh inactive/elapsed-TTL rows permit omission, not completion. Writer must
    revalidate every retained row binding, immutable Add provenance and the
    trusted time before it commits anything this inventory selected. Changing
    TTL later must never resurrect a part an earlier selection skipped.
    """
    if (
        type(message_count) is not int
        or not 1 <= message_count <= MAX_ADD_MESSAGES
        or type(now_ms) is not int
        or now_ms < 0
    ):
        raise QuoteMemError("invalid_request",
                            "Request inventory has invalid message or time bounds")
    first: EvidenceRowV1 | None = None
    parts: list[InventoryPart] = []
    coordinates: set[tuple[int, int]] = set()
    identifiers: set[str] = set()
    for supplied in rows:
        row = EvidenceRowV1.model_validate(supplied.model_dump(mode="json"))
        if len(parts) == MAX_ADD_SOURCE_PARTS:
            raise QuoteMemError("budget_exceeded",
                                "Request inventory exceeds the complete Add bound")
        if first is None:
            first = row
        if any(
            getattr(first, name) != getattr(row, name)
            for name in (
                "tenant_id",
                "scope_kind",
                "scope_id",
                "session_id",
                "request_id",
                "ingested_commit_seq",
            )
        ):
            raise QuoteMemError("parity_mismatch",
                                "Request inventory combines different immutable Adds")
        coordinate = row.ordinal, row.part_no
        if coordinate in coordinates or row.object_id in identifiers:
            raise QuoteMemError("parity_mismatch", "Request inventory repeats an Evidence part")
        coordinates.add(coordinate)
        identifiers.add(row.object_id)
        parts.append(
            InventoryPart(
                row.object_id,
                row.ordinal,
                row.part_no,
                bind("row-state", row.model_dump(mode="json")),
                row.state == "active" and (row.expires_at is None or row.expires_at > now_ms),
            )
        )
    if first is None or {part.ordinal for part in parts} != set(range(message_count)):
        raise QuoteMemError("parity_mismatch", "Request inventory is missing a logical message")
    for ordinal in range(message_count):
        numbers = sorted(part.part_no for part in parts if part.ordinal == ordinal)
        if numbers != list(range(len(numbers))):
            raise QuoteMemError(
                "parity_mismatch", "Request inventory is missing an interior source part",
            )
    return RequestInventory(
        first.tenant_id,
        first.scope_kind,
        first.scope_id,
        first.session_id,
        first.request_id,
        first.ingested_commit_seq,
        message_count,
        now_ms,
        tuple(sorted(parts, key=lambda part: (part.ordinal, part.part_no))),
    )


