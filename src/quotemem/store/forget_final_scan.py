"""Shared privacy quiescence after a forget's admission commit or a privacy erase.

Forget and privacy erase call the same function after their
commit. It confers no completion: the caller's final scan and receipt remain
its own proof obligations.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

from quotemem.core.errors import QuoteMemError
from quotemem.core.query import ScopeV1
from quotemem.ingestion.tombstone_index import TombstoneIndex
from quotemem.store.read_model import StoreReadModel
from quotemem.store.writer import Writer


def privacy_quiescence(
    writer: Writer,
    publisher: StoreReadModel,
    scope: ScopeV1,
    *,
    barrier: Callable[[], None],
    tombstone_indexes: Iterable[TombstoneIndex] = (),
) -> None:
    """Drain old readers, then release the plaintext the erasure left in process.

    Steps, in order: (1) revoke guards on older pins and wait, outside
    the Writer mutex, until registered readers drain; (2) their plaintext
    buffers are released with their guards; (3) under the Writer, recheck the
    caller's durable barrier, release exactly the memos the erasure's own
    mutations retired, invalidate HeavenBase's near cache for every entity,
    and close each given TombstoneIndex of this scope built for an older
    ``tombstone_seq``. The publisher drops the scope's recall corpora with the
    drain. There is no label-scan cache to clear in v1.

    Args:
        writer: The single Writer that committed the erasure.
        publisher: That Writer's registered read model.
        scope: The exact scope whose barrier is already published.
        barrier: Raises unless the caller's durable fence or reservation still
            holds; called under the Writer mutex before any cache release.
        tombstone_indexes: The Add owners' loaded indexes for this scope; one
            at the current ``tombstone_seq`` stays open.

    Raises:
        QuoteMemError: ``binding_mismatch`` for another publisher or another scope's
            index; the publisher's drain errors (Writer mutex held, active
            callback, missing barrier, changed root); the barrier's own error.
    """
    if StoreReadModel.get(writer) is not publisher:
        raise QuoteMemError("binding_mismatch", "Privacy quiescence has another read-model owner")
    indexes = tuple(tombstone_indexes)
    if any(
        (index.scope_kind, index.scope_id) != (scope.scope_kind, scope.scope_id)
        for index in indexes
    ):
        raise QuoteMemError("binding_mismatch", "A tombstone index belongs to another scope")
    publisher.quiesce_privacy(scope)
    with writer.mutex:
        barrier()
        writer.release_erased_caches()
        current = writer.get_counter(f"tombstone_seq:{scope.scope_id}")
        for index in indexes:
            try:
                retired = index.tombstone_seq != current
            except QuoteMemError as error:
                # A revoked or stale index has already dropped its content.
                if error.code != "delivery_stale":
                    raise
                retired = True
            if retired:
                index.close()
