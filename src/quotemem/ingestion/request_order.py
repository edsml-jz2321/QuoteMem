"""Receipt-proven request ordering metadata, including content-free tombstones."""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from quotemem.core.admission import session_request_family
from quotemem.core.errors import QuoteMemError
from quotemem.core.identity import canonical_json
from quotemem.core.query import ScopeV1
from quotemem.store.indexed import BatchedHeaders, IndexedReader
from quotemem.store.row_cache import MISSING, membership_key

if TYPE_CHECKING:
    from quotemem.store.writer import Writer


@dataclass(frozen=True)
class RequestOrder:
    """Original receipt and source-time anchors; neither grants content access."""

    first: tuple[int, int, str]
    chunk_event_time: int | None
    received_commit_seq: int
    source_time_present: bool
    earliest_source_time: int | None


@dataclass(frozen=True)
class RequestCoordinate:
    """One complete request's independently proved chronology, without content."""

    key: tuple[str, str]
    source_order: tuple[object, ...]
    received_commit_seq: int
    has_source_coordinate: bool


def request_partitions(requests: Sequence[RequestCoordinate]) -> dict[tuple[str, str], int]:
    """Preserve source order within receipt blocks separated by unclocked Adds.

    A partition is its preceding unclocked Add's actual receipt sequence, never
    a fabricated source timestamp. Source order must remain monotone across
    blocks; contradictory chronology refuses instead of selecting a winner.
    """
    if len({request.key for request in requests}) != len(requests) or len({
        request.received_commit_seq for request in requests
    }) != len(requests):
        raise QuoteMemError("parity_mismatch", "Request order has duplicate identities or receipts")
    result: dict[tuple[str, str], int] = {}
    block: list[RequestCoordinate] = []
    partition = 0
    prior: tuple[object, ...] | None = None

    def close_block() -> None:
        nonlocal prior
        ordered = sorted(block, key=lambda request: request.source_order)
        if ordered and prior is not None and ordered[0].source_order <= prior:
            raise QuoteMemError("binding_mismatch",
                                "Source chronology contradicts receipt boundaries")
        for request in ordered:
            result[request.key] = partition
        if ordered:
            prior = ordered[-1].source_order
        block.clear()

    for request in sorted(requests, key=lambda request: request.received_commit_seq):
        if request.has_source_coordinate:
            block.append(request)
            continue
        close_block()
        partition = request.received_commit_seq
        result[request.key] = partition
    close_block()
    return result


def load_request_order(
    writer: Writer, scope: ScopeV1, session_id: str, request_ids: Sequence[str] | None = None,
    *, at_time: int, input_high_watermark: int,
) -> tuple[dict[str, RequestOrder], tuple[tuple[str, str], ...]]:
    """Join complete immutable receipts to exact source ordering headers.

    The caller holds its existing scoped read and uses the returned row keys
    to retire cached anchors. No plaintext or eligibility decision is returned.
    Missing or inconsistent receipt metadata fails closed.
    """
    # The pure partition owner is imported by segmentation; receipt inventory
    # also uses segmentation through cursor validation. Load that read owner
    # only when a caller actually authenticates persisted request metadata.
    from quotemem.ingestion.inventory import (
        SourceCoordinateV1,
        discover_session_requests,
        load_request_coordinates,
    )

    if type(input_high_watermark) is not int or input_high_watermark < 0:
        raise QuoteMemError("invalid_request", "Request ordering requires its frozen receipt bound")
    reader = IndexedReader(writer.runtime.workspace, writer.entities, tenant_id=writer.tenant_id)
    # Orders are per-row values: the scan table's budget is spent on whole
    # reception censuses, which evicted them on every Add.
    memo = writer.binding_memo()
    result: dict[str, RequestOrder] = {}
    admitted: set[str] = set()
    keys: set[tuple[str, str]] = set()
    session_key: Hashable | None = None
    session_token = writer.storage_epoch
    if request_ids is None:
        # Every Add's freeze orders every session of its scope; a session no
        # request reached since is served from its last order (_SessionOrder).
        session_key = ("request-order-session", scope.scope_kind, scope.scope_id, session_id)
        cached = memo.lookup(session_key)
        if cached is not MISSING:
            known = cast(_SessionOrder, cached)
            if known.serves(input_high_watermark):
                return dict(known.orders), known.keys
        discovery = discover_session_requests(writer, scope, session_id)
        request_ids = discovery.request_ids
        keys.update((value.entity_kind, value.object_id) for value in discovery.dependencies)
    visibility: dict[str, int] = {}
    excluded = False
    for offset in range(0, len(request_ids), 64):
        wanted = request_ids[offset:offset + 64]
        inventory = load_request_coordinates(
            writer, scope, session_id, wanted, input_high_watermark=input_high_watermark,
        )
        values = inventory.values
        receipts = inventory.receipts
        keys.update((value.entity_kind, value.object_id) for value in inventory.dependencies)
        admitted.update(value.request_id for value in receipts
                        if value.visibility_commit_seq <= input_high_watermark)
        visibility.update((value.request_id, value.visibility_commit_seq) for value in receipts
                          if value.visibility_commit_seq <= input_high_watermark)
        excluded = excluded or len(admitted) < offset + len(wanted)
        pending: list[tuple[
            str, tuple[SourceCoordinateV1, ...], Hashable, tuple[tuple[str, str], ...],
        ]] = []
        for request_id, parts in values:
            if request_id not in admitted:
                continue
            part_keys = tuple(("evidence", part.object_id) for part in parts)
            keys.update(part_keys)
            order_key = ("request-order", scope.scope_kind, scope.scope_id, session_id,
                         request_id, part_keys)
            cached = memo.lookup(order_key)
            if cached is MISSING:
                pending.append((request_id, parts, order_key, part_keys))
            else:
                result[request_id] = cast(RequestOrder, cached)
        token = writer.storage_epoch
        # One point query per part was a scope's whole history per finalize freeze.
        headers = BatchedHeaders(reader.source_ordering_headers, reader.source_ordering_header, (
            part.object_id for _request_id, parts, _key, _dependencies in pending
            for part in parts
        ))
        for request_id, parts, order_key, part_keys in pending:
            observations: list[int] = []
            first_time: int | None = None
            for index, part in enumerate(parts):
                header = headers(part.object_id)
                if header is None:
                    raise QuoteMemError("reference_missing", "Request order lost an original part")
                if any(header[name] != getattr(part, name) for name in type(part).model_fields):
                    raise QuoteMemError("parity_mismatch", "Request order differs from its receipt")
                keys.add(("evidence", part.object_id))
                observed = header["source_observed_at"]
                if index == 0:
                    first_time = observed
                if observed is not None:
                    observations.append(observed)
            first = parts[0]
            result[request_id] = RequestOrder(
                (first.ordinal, first.part_no, first.object_id), first_time,
                first.ingested_commit_seq, bool(observations), min(observations, default=None),
            )
            memo.store(order_key, token, part_keys, result[request_id], value_bytes=512)
    if set(result) != admitted:
        raise QuoteMemError("parity_mismatch", "Request order lost an admitted request")
    ordered = tuple(sorted(keys))
    if session_key is not None:
        proof = {*keys, *(
            proof_key for request_id in request_ids for proof_key in (
                membership_key("run", session_request_family(scope, session_id, request_id)),
                membership_key("evidence", canonical_json([
                    scope.scope_kind, scope.scope_id, session_id, request_id,
                ])),
            )
        )}
        memo.store(session_key, session_token, tuple(proof), _SessionOrder(
            dict(result), ordered, input_high_watermark, excluded,
            max(visibility.values(), default=0),
        ), value_bytes=512 + 256 * len(result) + 128 * len(proof))
    return result, ordered


@dataclass(frozen=True)
class _SessionOrder:
    """One session's complete request order and the cutoffs it holds for.

    The memo is proven by every row it read, by each request's Run family and
    by each request's Evidence set: a new request rewrites the session's
    ledger row, and a new receipt or part joins one of those sets. An order
    read at one cutoff serves another only when it left no receipt out and
    every receipt it kept is visible at the other cutoff.
    """

    orders: dict[str, RequestOrder]
    keys: tuple[tuple[str, str], ...]
    cutoff: int
    excluded: bool
    latest: int

    def serves(self, cutoff: int) -> bool:
        """Whether the order read at ``self.cutoff`` is the order at ``cutoff``."""
        return cutoff == self.cutoff or not self.excluded and cutoff >= self.latest
