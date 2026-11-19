"""One metadata-only reception census before Finalize opens any source body."""

from __future__ import annotations

import hashlib
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, cast

from pydantic import TypeAdapter

from aml.core.admission import session_request_family
from aml.core.artifacts import RequestLedgerV1
from aml.core.controls import ControlRowV1, RequestLedgerValueV1
from aml.core.errors import AMLError
from aml.core.identity import ArtifactId, ContentBindingDigest, base64url, canonical_json
from aml.core.query import ScopeV1
from aml.core.run_inputs import SessionInventoryDurableRunInputV2
from aml.core.technical_types import RequestProgressV1
from aml.core.work import FinalizeSourceHeaderV1
from aml.ingestion.inventory import (
    MetadataRequestInventories,
    _Read,  # pyright: ignore[reportPrivateUsage]
    _request_coordinates,  # pyright: ignore[reportPrivateUsage]
)
from aml.ingestion.receipts import inspect_ledger
from aml.store.domain import SessionTargetRunRowV1
from aml.store.indexed import MAX_PAGE_SIZE, IndexedReader, ScopedRunRows
from aml.store.row_cache import (
    MEMBERSHIP_ANY,
    MISSING,
    BorrowWatch,
    membership_family,
    membership_key,
)
from aml.store.writer import Writer

_MAX_RECORDS = 100_000
_INVENTORY = TypeAdapter(MetadataRequestInventories)


class FinalizeReceiptPreparationChanged(AMLError):
    """The census borrow changed; rebuild it before using any source content.

    This signal carries no authority to accept changed receipt geometry. A
    caller may retry only after ending its current borrow and rechecking the
    same finalize owner, cutoff and complete reception directory.
    """

    def __init__(self) -> None:
        super().__init__("reception_stale", "Finalize reception metadata changed before use")


@dataclass(frozen=True)
class FinalizeReceipt:
    """One fixed request; shared ledger authority belongs to the parent census."""

    session_id: str
    request_id: str
    _inventory_json: str
    _header_json: tuple[str, ...]

    @property
    def inventory(self) -> MetadataRequestInventories:
        """Give each consumer its own mutable DTOs; cached authority remains immutable."""
        return _INVENTORY.validate_json(self._inventory_json)

    @property
    def headers(self) -> tuple[FinalizeSourceHeaderV1, ...]:
        """Decode only the requested receipt's fixed headers into independent objects."""
        return tuple(FinalizeSourceHeaderV1.model_validate_json(value)
                     for value in self._header_json)

    def resident_bytes(self) -> int:
        """Measure retained receipt text without decoding or copying its JSON."""
        return (
            sys.getsizeof(self) + sys.getsizeof(self.__dict__)
            + sys.getsizeof(self.session_id) + sys.getsizeof(self.request_id)
            + sys.getsizeof(self._inventory_json) + sys.getsizeof(self._header_json)
            + sum(sys.getsizeof(value) for value in self._header_json)
        )


@dataclass(frozen=True)
class FinalizeReceiptCensus:
    """A borrowed metadata result; it grants neither content access nor a write."""

    receipt_high_watermark: int
    past: tuple[FinalizeReceipt, ...]
    keys: tuple[tuple[str, str], ...]
    token: int
    # The finalize scans recheck the census between every row they read; the
    # watch answers each recheck in O(1) instead of once per census key.
    watch: BorrowWatch = field(compare=False, repr=False)

    def matches(self, writer: Writer) -> bool:
        """Include absent membership and omitted peers in every cache recheck."""
        return writer.borrowed_rows_watched(self.watch)

    def validate(self, writer: Writer, check: Callable[[], None]) -> None:
        """Recheck the actual borrow and metadata before opening or reusing content."""
        check()
        if not self.matches(writer):
            raise FinalizeReceiptPreparationChanged()
        check()


@dataclass(frozen=True)
class _SessionReceipts:
    """One session's part of the reception census, memoized while its rows are unchanged.

    ``runs`` is the session's complete Run directory as the census enumerated
    it; a later census reuses this part only for an identical directory, and
    ``proof`` names every row the part read, so an unchanged directory and
    proof mean the same rows produce the same part. Set membership is proven
    by those enumerations and by the census's own complete source cover, not
    by membership keys: every Add retires the scope-wide Run and Evidence
    sets, which made every session's part be rebuilt by every Add.
    """

    runs: tuple[tuple[str, str, str | None, str | None], ...]
    accepted: int
    entries: tuple[tuple[int, FinalizeReceipt, tuple[str, ...]], ...]
    keys: frozenset[tuple[str, str]]
    proof: tuple[tuple[str, str], ...]
    bound: int | None
    """The cutoff this part was read at when it left out a later receipt, else None."""

    def resident_bytes(self) -> int:
        """Estimate retained bytes: the receipts plus the census and proof keys."""
        return 512 + sum(entry.resident_bytes() + 64 * len(parts)
                         for _visibility, entry, parts in self.entries) + 128 * (
            len(self.keys) + len(self.proof) + len(self.runs)
        )


def _scope_session_runs(
    reader: IndexedReader, scope: ScopeV1, check: Callable[[], None],
) -> dict[str, tuple[tuple[str, str, str | None, str | None], ...]]:
    """Enumerate the scope's ingestion Runs by session: identity, commit mode, admission and input.

    One scan of the scope's ``session_ingestion`` Runs, each session's in
    ascending id order as a scan of that session alone returns them. A census
    read each session's Runs with its own statement, one per session of the
    user's history, at every Add.
    """
    runs: dict[str, list[tuple[str, str, str | None, str | None]]] = {}
    count = 0
    after: str | None = None
    while page := reader.page(ScopedRunRows(
        scope_kind=scope.scope_kind, scope_id=scope.scope_id, run_kind="session_ingestion",
    ), after=after, limit=MAX_PAGE_SIZE):
        check()
        for raw in page:
            runs.setdefault(raw["target_id"], []).append((
                raw["object_id"], raw["commit_mode"], raw["admission_digest"], raw["input_ref"],
            ))
        count += len(page)
        if count > _MAX_RECORDS:
            raise AMLError("budget_exceeded", "Finalize reception history is too large")
        after = page[-1]["object_id"]
    return {target: tuple(values) for target, values in runs.items()}


def _inspect_session(
    writer: Writer, reader: IndexedReader, scope: ScopeV1, reference: dict[str, Any],
    runs: tuple[tuple[str, str, str | None, str | None], ...],
    headers: Mapping[str, FinalizeSourceHeaderV1], receipt_high_watermark: int,
    check: Callable[[], None],
) -> _SessionReceipts:
    """Read one session's ledger, attached receipts and request coordinates.

    Its per-request entries live in the Writer's per-row table: the census
    itself, one entry per cutoff, is larger than the scan table's whole
    budget and evicted them on every Add.
    """
    memo = writer.binding_memo()
    token = writer.storage_epoch
    read = _Read(writer, scope)
    keys: set[tuple[str, str]] = {("run", run[0]) for run in runs}
    proof: set[tuple[str, str]] = {MEMBERSHIP_ANY, *keys}
    control = ControlRowV1.model_validate(read.row("control", reference["object_id"]))
    if (control.key, control.version) != (reference["key"], reference["version"]) or (
        not isinstance(control.value, RequestLedgerValueV1)
    ):
        raise AMLError("parity_mismatch", "Finalize reception directory changed its row")
    session_hash = control.key.rsplit(":", 1)[1]
    accepted: dict[ContentBindingDigest, RequestProgressV1] = {}

    def observe(key: ContentBindingDigest, progress: RequestProgressV1) -> None:
        if len(accepted) == _MAX_RECORDS:
            raise AMLError("budget_exceeded", "Finalize reception exceeds its Add bound")
        accepted[key] = progress

    def ledger_block(reference: ArtifactId) -> RequestLedgerV1:
        body = read.artifact(reference, "request_ledger")
        if not isinstance(body, RequestLedgerV1):
            raise AMLError("parity_mismatch", "Finalize reception changed its ledger kind")
        return body

    inspect_ledger(
        control.value, writer.bind_content("add-request-key", {"lookup": "finalize"}),
        ledger_block, writer.bind_content, observe=observe,
    )
    admissions = {value.admission_digest for value in accepted.values()}
    identities: dict[ContentBindingDigest, tuple[str, str]] = {}
    for object_id, commit_mode, admission_digest, input_ref in runs:
        check()
        if commit_mode != "orchestrator" or (
            admission_digest not in admissions or input_ref is None
        ):
            continue
        run = SessionTargetRunRowV1.model_validate(read.row("run", object_id))
        if run.input_ref is None:
            raise AMLError("reception_stale", "Finalize receipt lost its attached input")
        body = read.artifact(run.input_ref, "run_input", run.object_id)
        if not isinstance(body, SessionInventoryDurableRunInputV2):
            raise AMLError("unsupported_variant", "Finalize reception needs V2 receipts")
        identity = body.business_key.session_id, body.business_key.request_id
        request_key = writer.bind_content("add-request-key", {
            "scope_id": scope.scope_id, "session_id": identity[0],
            "request_id": identity[1],
        })
        if (
            base64url(hashlib.sha256(identity[0].encode()).digest())[:22] != session_hash
            or run.target_kind != "session" or run.target_id != session_hash
            or (run.scope_kind, run.scope_id) != (scope.scope_kind, scope.scope_id)
            or body.request_key != request_key or request_key not in accepted
            or body.admission_digest != run.admission_digest
            or accepted[request_key].admission_digest != run.admission_digest
            or request_key in identities and identities[request_key] != identity
        ):
            raise AMLError("parity_mismatch", "Finalize receipt changed its directory")
        identities[request_key] = identity
    if set(identities) != set(accepted):
        raise AMLError("parity_mismatch", "Finalize reception lost an attached receipt")
    entries: list[tuple[int, FinalizeReceipt, tuple[str, ...]]] = []
    bound: int | None = None
    for key, (session_id, request_id) in sorted(identities.items()):
        check()
        membership = {
            MEMBERSHIP_ANY, membership_family("evidence"), membership_family("run"),
            membership_key("evidence", canonical_json([
                scope.scope_kind, scope.scope_id, session_id, request_id,
            ])),
            membership_key("run", session_request_family(scope, session_id, request_id)),
        }
        keys.update(membership)
        entry_key = (
            "finalize-receipt", scope.scope_kind, scope.scope_id,
            session_id, request_id, canonical_json(accepted[key]),
        )
        cached_entry = memo.lookup(entry_key)
        if cached_entry is not MISSING:
            visibility, entry, entry_keys = cast(tuple[
                int, FinalizeReceipt, tuple[tuple[str, str], ...],
            ], cached_entry)
            keys.update(entry_keys)
            proof.update(key for key in entry_keys if key[0] != "membership")
            parts = tuple(header.object_id for header in entry.headers)
            if visibility <= receipt_high_watermark:
                for header in entry.headers:
                    if headers.get(header.object_id) != header:
                        raise AMLError(
                            "parity_mismatch", "Finalize receipt changed an original part",
                        )
            entries.append((visibility, entry, parts))
            continue
        inventory = _request_coordinates(
            read, reader, scope, session_id, request_id, key, accepted[key],
            receipt_high_watermark,
        )
        keys.update(
            (value.entity_kind, value.object_id) for value in inventory.dependencies
        )
        if not inventory.receipts:
            bound = receipt_high_watermark
            continue
        if len(inventory.receipts) != 1 or len(inventory.values) != 1:
            raise AMLError("reception_stale", "Finalize receipt changed during its census")
        parts = inventory.values[0][1]
        selected_headers: list[FinalizeSourceHeaderV1] = []
        for part in parts:
            header = headers.get(part.object_id)
            if header is None or any(
                getattr(header, name) != getattr(part, name)
                for name in type(part).model_fields
            ):
                raise AMLError("parity_mismatch", "Finalize receipt lost an original part")
            selected_headers.append(header)
        inventory_json = _INVENTORY.dump_json(inventory).decode("utf-8")
        header_json = tuple(header.model_dump_json() for header in selected_headers)
        entry = FinalizeReceipt(session_id, request_id, inventory_json, header_json)
        entry_keys = tuple(sorted({
            *membership,
            *((value.entity_kind, value.object_id) for value in inventory.dependencies),
            *(("evidence", part.object_id) for part in parts),
        }))
        memo.store(
            entry_key, token, entry_keys,
            (inventory.receipts[0].visibility_commit_seq, entry, entry_keys),
            value_bytes=512 + len(inventory_json.encode()) + sum(
                64 + len(value.encode()) for value in header_json
            ) + 128 * len(entry_keys),
        )
        proof.update(key for key in entry_keys if key[0] != "membership")
        entries.append((
            inventory.receipts[0].visibility_commit_seq, entry,
            tuple(part.object_id for part in parts),
        ))
    keys.update(proof)
    keys.update((value.entity_kind, value.object_id) for value in read.dependencies.values())
    proof.update((value.entity_kind, value.object_id) for value in read.dependencies.values())
    return _SessionReceipts(
        runs, len(accepted), tuple(entries), frozenset(keys), tuple(sorted(proof)), bound,
    )


def _census_headers(
    writer: Writer, reader: IndexedReader, scope: ScopeV1, check: Callable[[], None],
) -> Mapping[str, FinalizeSourceHeaderV1]:
    """Read every Evidence header of the exact scope, whatever its state or receipt.

    One Add reads its census three times, at its freeze, its session and its
    completion, and every Add adds rows to the scope. The headers come from
    the Writer's per-scope index (``ScopeEvidenceIndex.snapshot``), which
    re-reads only the rows retired since its last read, so a late Add of a
    long user does not read every earlier header again. The returned mapping
    is shared: callers must not change it.
    """
    snapshot = writer.scope_evidence().snapshot(reader, scope, check)
    check()
    headers = snapshot.headers
    if len(headers) > _MAX_RECORDS:
        raise AMLError("budget_exceeded", "Finalize reception exceeds its source bound")
    if any(
        (header.tenant_id, header.scope_kind, header.scope_id)
        != (writer.tenant_id, scope.scope_kind, scope.scope_id) for header in headers.values()
    ):
        raise AMLError("scope_denied", "Finalize reception crossed its exact tenant scope")
    snapshot.note(scope)
    return headers


def inspect_finalize_receipts(
    writer: Writer, scope: ScopeV1, *, receipt_high_watermark: int,
    check: Callable[[], None],
) -> FinalizeReceiptCensus:
    """Join the complete reception directory to original V2 receipt coordinates.

    The cutoff is an already-authorized tenant commit sequence. Current ledger
    membership is never mistaken for the original freeze: future receipts are
    authenticated but their source coordinates are not included. Missing past
    parts, whole requests or sessions refuse before any Evidence body is read.
    """
    if type(receipt_high_watermark) is not int or receipt_high_watermark < 0:
        raise AMLError("invalid_request", "Finalize reception requires its exact commit cutoff")
    check()
    scope = ScopeV1.model_validate(scope.model_dump(mode="json"))
    # A long history's census is megabytes; the scan table's 16 MB budget let
    # it evict, or refuse, it before the same Add's later reads asked for it.
    memo = writer.binding_memo()
    memo_key = (
        "finalize-receipts", scope.scope_kind, scope.scope_id, receipt_high_watermark,
    )
    cached = memo.lookup(memo_key)
    if cached is not MISSING:
        result = cast(FinalizeReceiptCensus, cached)
        result.validate(writer, check)
        return result
    token = writer.storage_epoch
    keys: set[tuple[str, str]] = {
        MEMBERSHIP_ANY, membership_family("evidence"), membership_family("control"),
        membership_family("run-scope"), membership_family("run"),
        membership_key("evidence", f"{scope.scope_kind}:{scope.scope_id}"),
        membership_key("control", f"request_ledger:{scope.scope_id}"),
        membership_key("run-scope", f"{scope.scope_kind}:{scope.scope_id}"),
    }
    reader = IndexedReader(writer.runtime.workspace, writer.entities, tenant_id=writer.tenant_id)
    headers = _census_headers(writer, reader, scope, check)
    keys.update(("evidence", identifier) for identifier in headers)

    # Session parts live in the per-row table (``_inspect_session``).
    session_memo = writer.binding_memo()
    past: list[FinalizeReceipt] = []
    covered: set[str] = set()
    accepted_count = 0
    ledger_count = 0
    run_count = 0
    scope_runs: dict[str, tuple[tuple[str, str, str | None, str | None], ...]] | None = None
    after = None
    while page := reader.scoped_request_ledger_headers(scope, after=after):
        for reference in page:
            check()
            ledger_count += 1
            if ledger_count > _MAX_RECORDS:
                raise AMLError("budget_exceeded", "Finalize reception directory is too large")
            session_hash = reference["key"].rsplit(":", 1)[1]
            if scope_runs is None:
                scope_runs = _scope_session_runs(reader, scope, check)
            runs = scope_runs[session_hash] if session_hash in scope_runs else ()
            run_count += len(runs)
            if run_count > _MAX_RECORDS:
                raise AMLError("budget_exceeded", "Finalize reception history is too large")
            session_key = (
                "finalize-receipt-session", scope.scope_kind, scope.scope_id,
                reference["object_id"], reference["version"],
            )
            cached = session_memo.lookup(session_key)
            session = None if cached is MISSING else cast(_SessionReceipts, cached)
            if session is None or session.runs != runs or (
                session.bound is not None and session.bound != receipt_high_watermark
            ):
                session_token = writer.storage_epoch
                session = _inspect_session(
                    writer, reader, scope, reference, runs, headers, receipt_high_watermark,
                    check,
                )
                session_memo.store(
                    session_key, session_token, session.proof, session,
                    value_bytes=session.resident_bytes(),
                )
            accepted_count += session.accepted
            if accepted_count > _MAX_RECORDS:
                raise AMLError("budget_exceeded", "Finalize reception exceeds its Add bound")
            keys.update(session.keys)
            for visibility, entry, parts in session.entries:
                if visibility > receipt_high_watermark:
                    continue
                for part in parts:
                    if part not in headers:
                        raise AMLError("parity_mismatch", "Finalize receipt lost an original part")
                    covered.add(part)
                past.append(entry)
        after = page[-1]["key"].rsplit(":", 1)[1]
    if covered != {identifier for identifier, header in headers.items()
                   if header.ingested_commit_seq <= receipt_high_watermark}:
        raise AMLError("parity_mismatch", "Finalize sources differ from complete accepted receipts")
    result = FinalizeReceiptCensus(
        receipt_high_watermark, tuple(sorted(past, key=lambda value: (
            value.session_id, value.request_id,
        ))), tuple(sorted(keys)), token, writer.watch_borrowed_rows(keys, token),
    )
    result.validate(writer, check)
    size = (
        sys.getsizeof(result) + sys.getsizeof(result.__dict__)
        + sys.getsizeof(result.past) + sys.getsizeof(result.keys)
        + sum(receipt.resident_bytes() for receipt in result.past)
        # The census and its watch retain the dependency keys as well as the
        # memo's proof. Charge their strings and the watch's set conservatively.
        + sum(sys.getsizeof(key) + sum(sys.getsizeof(part) for part in key)
              for key in result.keys)
        + sys.getsizeof(frozenset(result.keys))
    )
    memo.store(memo_key, token, result.keys, result, value_bytes=size)
    return result
