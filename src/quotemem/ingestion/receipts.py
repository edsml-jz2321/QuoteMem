"""Bounded reception ledger proofs.

These functions compute and validate values. Only the authoritative Writer
persists a block, its receipt, or the corresponding visibility transaction.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from quotemem.core.artifacts import RequestLedgerEntryV1, RequestLedgerV1
from quotemem.core.controls import RequestLedgerValueV1
from quotemem.core.errors import QuoteMemError
from quotemem.core.identity import ArtifactId, ContentBindingDigest
from quotemem.core.technical_types import RequestProgressV1

BindContent = Callable[[str, object], ContentBindingDigest]


@dataclass(frozen=True)
class LedgerHead:
    """Carry a validated immutable overflow head and its cumulative count."""

    reference: ArtifactId | None
    digest: ContentBindingDigest | None
    total_count: int


def block_digest(
    bind: BindContent,
    entries: list[RequestLedgerEntryV1],
    previous_digest: ContentBindingDigest | None,
    total_count: int,
) -> ContentBindingDigest:
    """Bind a chain prefix without copying its prior entries into a new block.

    Args:
        bind: The tenant's purpose-separated content binding function.
        entries: Canonically ordered entries in this bounded block.
        previous_digest: Verified digest of the preceding prefix, if present.
        total_count: Number of entries in the entire resulting chain.

    Returns:
        The cumulative digest for this block.

    Raises:
        QuoteMemError: If the supplied digest function rejects its input.
    """
    return bind(
        "add-request-ledger",
        {
            "previous_digest": previous_digest,
            "entries": [entry.model_dump(mode="json") for entry in entries],
            "entry_count": len(entries),
            "total_count": total_count,
        },
    )


def inspect_ledger(
    value: RequestLedgerValueV1,
    request_key: ContentBindingDigest,
    read: Callable[[ArtifactId], RequestLedgerV1],
    bind: BindContent,
    *,
    observe: Callable[[ContentBindingDigest, RequestProgressV1], None] | None = None,
) -> tuple[RequestProgressV1 | None, LedgerHead]:
    """Verify every linked prefix before returning an accepted request binding.

    Args:
        value: The captured Control value, read under session ownership.
        request_key: The exact tenant-bound request identity to find.
        read: A strict logical Artifact reader with envelope verification.
        bind: The tenant's content binding function.
        observe: Optional read-only observer of each current or historical binding.

    Returns:
        A matching binding, if any, and the validated current overflow head.

    Raises:
        QuoteMemError: If the chain cycles, loses a block, repeats a request, or fails parity.
    """
    found = value.recent.get(request_key)
    if observe is not None:
        for key, progress in value.recent.items():
            observe(key, progress)
    reference, expected = value.ledger_ref, value.ledger_digest
    visited: set[ArtifactId] = set()
    seen = set(value.recent)
    first_count = 0
    expected_count: int | None = None
    while reference is not None:
        if reference in visited:
            raise QuoteMemError("service_unhealthy", "Reception ledger contains a cycle")
        visited.add(reference)
        block = read(reference)
        if not first_count:
            first_count = block.total_count
        if expected != block.cumulative_digest or (
            expected_count is not None and expected_count != block.total_count
        ):
            raise QuoteMemError("service_unhealthy", "Reception ledger head or count differs")
        previous = None if block.previous_ref is None else read(block.previous_ref)
        prior_digest = None if previous is None else previous.cumulative_digest
        prior_count = 0 if previous is None else previous.total_count
        if block.total_count != prior_count + block.entry_count or block.cumulative_digest != (
            block_digest(bind, block.entries, prior_digest, block.total_count)
        ):
            raise QuoteMemError("service_unhealthy", "Reception ledger cumulative proof differs")
        for entry in block.entries:
            if entry.request_key in seen:
                raise QuoteMemError("service_unhealthy", "Reception ledger repeats a request")
            seen.add(entry.request_key)
            if observe is not None:
                observe(
                    entry.request_key,
                    RequestProgressV1(
                        admission_digest=entry.admission_digest,
                        confirmed_request=entry.confirmed_request,
                    ),
                )
            if entry.request_key == request_key:
                found = RequestProgressV1(
                    admission_digest=entry.admission_digest,
                    confirmed_request=entry.confirmed_request,
                )
        expected_count = prior_count
        reference, expected = block.previous_ref, prior_digest
    return found, LedgerHead(value.ledger_ref, value.ledger_digest, first_count)


def overflow_block(
    value: RequestLedgerValueV1, head: LedgerHead, bind: BindContent
) -> RequestLedgerV1 | None:
    """Freeze the complete old recent window when the next request would overflow.

    Args:
        value: Current accepted request bindings.
        head: Verified overflow identity from that exact Control value.
        bind: The tenant's content binding function.

    Returns:
        One create-once block, or None when the recent window still has room.

    Raises:
        QuoteMemError: If the supplied head does not belong to the captured value.
    """
    if (head.reference, head.digest) != (value.ledger_ref, value.ledger_digest):
        raise QuoteMemError("service_unhealthy", "Reception ledger projection changed")
    if len(value.recent) < 64:
        return None
    entries = [
        RequestLedgerEntryV1(request_key=key, **progress.model_dump(mode="json"))
        for key, progress in sorted(value.recent.items())
    ]
    total = head.total_count + len(entries)
    return RequestLedgerV1(
        previous_ref=head.reference,
        entries=entries,
        entry_count=len(entries),
        total_count=total,
        cumulative_digest=block_digest(bind, entries, head.digest, total),
    )
