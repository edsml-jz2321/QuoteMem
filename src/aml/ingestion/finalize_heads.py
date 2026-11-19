"""Bounded guarded row streams for finalize input heads in plan.md §5.9 and §9.

These read-only aggregates grant no lease, frozen-input or attachment authority.
The owner must double-sample its per-scope token under the held fence, use one
fixed freeze time, and validate its issued ticket before any effect. No source
body escapes the registered workflow callback.

The digest is memoized on the sole Writer (ADR 0041): a scan records the
admission token read before it started, every row it hashed and the membership
key of the set it enumerated, and is served again while the Writer's mutation
ledger proves none of them retired since. A served digest is exactly what the
scan would hash now; the caller's borrow check still runs on every call.

The eligible rows come from the Writer's per-scope header index
(``aml.store.scope_evidence``) rather than a support-order SQL scan, and the
digest is a keyed multiset hash of the rows (``evidence_set_digest``), so an
Add derives values only for the rows it added or changed.
"""

from __future__ import annotations

import base64
from collections.abc import Callable, Container, Sequence
from dataclasses import dataclass
from typing import cast

from pydantic import ValidationError

from aml.core.errors import AMLError
from aml.core.identity import ContentBindingDigest, base64url, canonical_json
from aml.core.query import ScopeV1
from aml.core.technical_types import SupportRangeKeyV1
from aml.core.work import FinalizeSourceHeaderV1
from aml.ingestion.finalize_receipts import inspect_finalize_receipts
from aml.store.domain import EvidenceRowV1
from aml.store.indexed import (
    MAX_PAGE_SIZE,
    FinalizeEvidenceCursor,
    FinalizeEvidenceQuery,
    IndexedReader,
)
from aml.store.read_model import SnapshotReadGuard
from aml.store.row_cache import (
    MEMBERSHIP_ANY,
    MISSING,
    ScanMemo,
    membership_family,
    membership_key,
    note_unprovable_read,
)
from aml.store.scope_evidence import RowElement
from aml.store.writer import Writer


@dataclass(frozen=True)
class FinalizeRowDigest:
    """Constant-size aggregate of complete canonical rows, without row content."""

    count: int
    digest: ContentBindingDigest


@dataclass(frozen=True)
class FinalizeEvidenceDigest(FinalizeRowDigest):
    """Retain only the support-order boundaries needed by compact work metadata."""

    first_key: SupportRangeKeyV1 | None
    last_key: SupportRangeKeyV1 | None


def _check_borrow(writer: Writer, scope: ScopeV1, guard: SnapshotReadGuard) -> None:
    if writer.mutex._is_owned():  # pyright: ignore[reportPrivateUsage]
        raise AMLError("reader_busy", "Finalize row scans must run outside the Writer mutex")
    guard.check_operation(writer, scope)


def _row_binding(
    writer: Writer, memo: ScanMemo, token: int, row: EvidenceRowV1,
) -> tuple[str, ContentBindingDigest, int, int]:
    """Return one Evidence row's canonical bytes and row-state digest, memoized per row.

    The same rows are canonicalized by the input digest and bound by the
    source inventory, and every Add opens a finalize cycle that repeats both.
    Canonicalizing one row of a 2,048-value embedding space costs 0.61 ms, so
    the bytes are memoized beside the row-state digest under the row's key;
    any mutation of the row retires both (ADR 0036). ``token`` must be the
    admission token the caller read before the row.
    """
    key = ("row-binding", row.object_id)
    memoized = memo.lookup(key)
    if memoized is not MISSING:
        return cast(tuple[str, ContentBindingDigest, int, int], memoized)
    dumped = row.model_dump(mode="json")
    value = (
        canonical_json(dumped), writer.bind_content("row-state", dumped),
        row.byte_start, row.byte_end,
    )
    memo.store(key, token, (("evidence", row.object_id),), value)
    return value


def read_finalize_evidence_digest(
    writer: Writer,
    scope: ScopeV1,
    guard: SnapshotReadGuard,
    *,
    freeze_time: int,
    receipt_high_watermark: int,
) -> FinalizeEvidenceDigest:
    """Bind every eligible complete Evidence row of the fixed support set.

    The rows are selected by the Writer tenant, exact scope and fixed-time
    eligibility (``support_headers``); rows read in full are checked against
    their headers. Pending and indexed routes both participate. The caller
    must establish unchanged input tokens around the complete scan.
    """
    return inspect_finalize_evidence_digest(
        writer, scope, freeze_time=freeze_time, receipt_high_watermark=receipt_high_watermark,
        check=lambda: _check_borrow(writer, scope, guard),
    )


_SUM_MODULUS = 1 << 256


def support_headers(
    writer: Writer, reader: IndexedReader, query: FinalizeEvidenceQuery,
) -> tuple[FinalizeSourceHeaderV1, ...]:
    """Return exactly the rows ``finalize_evidence_page`` pages return, as headers, in its order.

    They are selected from the scope's header index
    (``ScopeEvidenceIndex.snapshot``), current as of a token read during this
    call, by the same predicate and support order as the SQL scan; like that
    scan, the read is not provable by row keys. Every Add of a long user read
    them three times per freeze, each time sorting the scope's rows in SQL and
    validating every header again.

    Raises:
        AMLError: ``schema_invalid`` for a row of another tenant, as the scan
            refuses one.
    """
    snapshot = writer.scope_evidence().snapshot(reader, query.scope)
    note_unprovable_read()
    headers = snapshot.eligible(query.freeze_time, query.receipt_high_watermark, query.tenant_id)
    if any(header.tenant_id != query.tenant_id for header in headers):
        raise AMLError("schema_invalid", "Invalid stored finalize Evidence header")
    return headers


_KNOWN = cast(tuple[str, ContentBindingDigest, int, int], ("", "", -1, -1))
"""Placeholder of a row whose binding the caller already holds."""


def evidence_element(writer: Writer, digest: ContentBindingDigest) -> int:
    """The secret evidence-set element of one row: a keyed PRF of its row-state digest.

    The row-state digest is stored in source inventories; the element derived
    from it under the tenant key is never stored or returned, so a sum of
    elements reveals nothing an attacker could combine (``evidence_set_digest``).
    """
    value = writer.bind_content("finalize-evidence-element", digest)
    return int.from_bytes(base64.urlsafe_b64decode(value + "=" * (-len(value) % 4)), "big")


def evidence_set_digest(writer: Writer, count: int, total: int) -> ContentBindingDigest:
    """Bind the count and the element sum of an eligible Evidence row set.

    The finalize input digest is a keyed multiset hash (MSet-Add-Hash, Clarke
    et al., ASIACRYPT 2003): each row contributes ``evidence_element`` of its
    row-state digest, a 256-bit HMAC-SHA256 output under the tenant key, and
    the elements are added modulo 2**256. The sum and the count are then bound
    by one more HMAC under a distinct purpose, so only that final value is
    ever stored. A different row set, or any changed row, yields the same
    value only if an HMAC collides or if unknown pseudorandom elements of two
    different sets sum to the same value, each with probability about 2**-256
    per attempt for an attacker without the key. The support order is not
    hashed: it is a function of the bound rows' own trust, time and identity
    fields (``finalize_source_order``). The sum composes, so a scope's
    elements are derived once per row (``row_elements``), not once per Add.
    """
    return writer.bind_content("finalize-evidence-set", {
        "count": count, "sum": base64url((total % _SUM_MODULUS).to_bytes(32, "big")),
    })


def row_elements(
    writer: Writer, reader: IndexedReader, query: FinalizeEvidenceQuery,
    headers: Sequence[FinalizeSourceHeaderV1], recheck: Callable[[], None],
) -> list[RowElement]:
    """Return each eligible header's row-state digest and set element, in order.

    ``headers`` is ``support_headers(writer, reader, query)``. Values the
    scope index already holds are reused; the others come from the row
    bindings (``evidence_bindings``), which read in full only rows the
    binding memo lacks, and are kept in the index for later Adds while the
    ledger proves their rows unchanged since the token read before them.
    """
    index = writer.scope_evidence()
    known = index.elements(query.scope)
    token = writer.storage_epoch
    bindings = evidence_bindings(writer, reader, query, None, headers, recheck, known)
    result: list[RowElement] = []
    fresh: dict[str, RowElement] = {}
    for header, binding in zip(headers, bindings, strict=True):
        if binding is None:
            result.append(known[header.object_id])
            continue
        value = (binding[1], evidence_element(writer, binding[1]))
        fresh[header.object_id] = value
        result.append(value)
    if fresh:
        index.store_elements(query.scope, token, fresh)
    return result


def evidence_bindings(
    writer: Writer, reader: IndexedReader, query: FinalizeEvidenceQuery,
    after: FinalizeEvidenceCursor | None, headers: Sequence[FinalizeSourceHeaderV1],
    recheck: Callable[[], None], known: Container[str] = frozenset(),
) -> tuple[tuple[str, ContentBindingDigest] | None, ...]:
    """Return each header's row canonical bytes and row-state digest, in page order.

    ``headers`` is ``support_headers``'s rows from ``after`` on. A row whose
    identity is in ``known`` is neither looked up nor read; its value is None.
    A row whose bytes the binding memo still proves unchanged is not read
    again; each run of consecutive memo misses is read in full by
    ``finalize_evidence_page`` statements of at most ``MAX_PAGE_SIZE`` rows at
    its position, which must be exactly the headers' rows. A late Add of a
    long user therefore reads only its new rows in full, instead of every
    earlier row of the scope.
    ``recheck`` runs after that read, as after any page, so a mutation that
    landed in between is reported by the caller's own census or lifetime
    check before this function reports the changed rows.
    """
    memo = writer.binding_memo()
    found: list[tuple[str, ContentBindingDigest, int, int] | None] = []
    for header in headers:
        if header.object_id in known:
            found.append(_KNOWN)
            continue
        memoized = memo.lookup(("row-binding", header.object_id))
        found.append(None if memoized is MISSING else cast(
            tuple[str, ContentBindingDigest, int, int], memoized,
        ))
    first = 0
    while first < len(found):
        if found[first] is not None:
            first += 1
            continue
        last = first
        while (last + 1 < len(found) and found[last + 1] is None
               and last + 1 - first < MAX_PAGE_SIZE):
            last += 1
        token = writer.storage_epoch
        rows = reader.finalize_evidence_page(
            query, after=after if first == 0 else FinalizeEvidenceCursor.at(headers[first - 1]),
            limit=last - first + 1,
        )
        recheck()
        if [row.object_id for row in rows] != [
            header.object_id for header in headers[first:last + 1]
        ] or any(
            FinalizeSourceHeaderV1.model_validate(row.model_dump(
                mode="python", include=set(FinalizeSourceHeaderV1.model_fields),
            )) != header for row, header in zip(rows, headers[first:last + 1], strict=True)
        ):
            raise AMLError("stale_evidence", "Finalize Evidence changed during its header scan")
        for index, row in enumerate(rows, first):
            found[index] = _row_binding(writer, memo, token, row)
        first = last + 1
    result: list[tuple[str, ContentBindingDigest] | None] = []
    for header, value in zip(headers, found, strict=True):
        if value is _KNOWN:
            result.append(None)
            continue
        if value is None or (value[2], value[3]) != (header.byte_start, header.byte_end):
            raise AMLError("stale_evidence", "Finalize Evidence changed during its header scan")
        result.append((value[0], value[1]))
    return tuple(result)


def inspect_finalize_evidence_digest(
    writer: Writer, scope: ScopeV1, *, freeze_time: int, receipt_high_watermark: int,
    check: Callable[[], None],
) -> FinalizeEvidenceDigest:
    """Read the same exact fixed-time row set for guarded or cold verification.

    The caller owns its actual lifetime check. No raw rows escape this
    function; a cold proof cannot be used to execute a workflow or access a
    Search root. The rows are the support headers (``support_headers``), each
    contributing its set element (``row_elements``) to ``evidence_set_digest``.
    """
    if writer.mutex._is_owned():  # pyright: ignore[reportPrivateUsage]
        raise AMLError("reader_busy", "Finalize row scans require Writer lock release")
    check()
    try:
        query = FinalizeEvidenceQuery(
            tenant_id=writer.tenant_id, scope=scope, freeze_time=freeze_time,
            receipt_high_watermark=receipt_high_watermark,
        )
    except ValidationError as error:
        raise AMLError("schema_invalid", "Invalid finalize Evidence scan identity") from error
    receipts = inspect_finalize_receipts(
        writer, scope, receipt_high_watermark=receipt_high_watermark, check=check,
    )
    memo = writer.scan_memo()
    memo_key = (
        "finalize-evidence", query.scope.scope_kind, query.scope.scope_id, freeze_time,
        receipt_high_watermark,
    )
    memoized = memo.lookup(memo_key)
    if memoized is not MISSING:
        return cast(FinalizeEvidenceDigest, memoized)
    token = writer.storage_epoch
    reader = writer.indexed_reader()
    receipts.validate(writer, check)
    headers = support_headers(writer, reader, query)
    receipts.validate(writer, check)
    elements = row_elements(
        writer, reader, query, headers, lambda: receipts.validate(writer, check),
    )
    check()
    # The digest depends on every eligible row's content and on the set of
    # Evidence rows in the exact scope: a new or re-scoped row is an upsert
    # whose post-image names this scope, whatever its route state.
    keys: list[tuple[str, str]] = [
        MEMBERSHIP_ANY, membership_family("evidence"),
        membership_key("evidence", f"{query.scope.scope_kind}:{query.scope.scope_id}"),
        *receipts.keys, *(("evidence", header.object_id) for header in headers),
    ]
    first: SupportRangeKeyV1 | None = None
    last: SupportRangeKeyV1 | None = None
    if headers:
        position = FinalizeEvidenceCursor.at(headers[0])
        first = SupportRangeKeyV1(
            key_kind="support", trust_tier=headers[0].trust_tier,
            observed_at=position.observed_at, evidence_id=position.evidence_id,
        )
        position = FinalizeEvidenceCursor.at(headers[-1])
        last = SupportRangeKeyV1(
            key_kind="support", trust_tier=headers[-1].trust_tier,
            observed_at=position.observed_at, evidence_id=position.evidence_id,
        )
    receipts.validate(writer, check)
    digest = FinalizeEvidenceDigest(
        len(headers),
        evidence_set_digest(writer, len(headers), sum(element for _digest, element in elements)),
        first, last,
    )
    memo.store(memo_key, token, keys, digest)
    return digest


def finalize_input_heads_digest(
    writer: Writer, evidence: FinalizeRowDigest,
) -> ContentBindingDigest:
    """Bind the frozen-input envelope: the Evidence count and its row-stream digest."""
    return writer.bind_content("finalize-input-heads", {
        "evidence_count": evidence.count,
        "evidence_rows_digest": evidence.digest,
    })
