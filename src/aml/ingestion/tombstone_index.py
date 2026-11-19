"""Owner-private, exact-scope tombstones for Add only (A.6.6, D.6).

Loading is explicit and outside the Writer lock. The owner supplies its read
fence and the scope counter; every read checks both. A changed sequence drops
all retained content and refuses the old index. Privacy quiescence must drain
the owner's callbacks before calling ``close``; no index enters RecallSnapshot.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from itertools import batched
from typing import Final

from pydantic import ValidationError

from aml.core.add_read_view import TombstoneRecord
from aml.core.errors import AMLError
from aml.core.query import ScopeV1
from aml.core.semantic import ScopeKind
from aml.store.domain import TombstoneRowV1
from aml.store.indexed import MAX_PAGE_SIZE, IndexedReader, TombstoneRows

TOMBSTONE_CHUNK_SIZE: Final = 1024
"""Rows prepared together during the cold load (A.6.6)."""


class TombstoneIndex:
    """One owner's tombstones at one scope sequence; implements ``TombstoneSource``.

    Use ``load`` to build it and ``close`` after the owner's readers drain.
    Failed loads never expose a partial or empty substitute. Storage failures
    propagate to the owner, which pauses; stale generations raise
    ``delivery_stale`` and malformed stored rows raise ``parity_mismatch``.
    """

    def __init__(
        self, scope: ScopeV1, tombstone_seq: int, *, sequence: Callable[[], int],
        check: Callable[[], None],
    ) -> None:
        """Bind lifetime checks without loading content; use ``load`` for readable indexes."""
        self._scope = scope.model_copy()
        self._sequence = sequence
        self._check = check
        self._tombstone_seq = tombstone_seq
        self._rows: tuple[TombstoneRecord, ...] = ()
        self._keys: dict[tuple[str, str], TombstoneRecord] = {}
        self._ready = False
        self._closed = False

    @classmethod
    def load(
        cls, reader: IndexedReader, scope: ScopeV1, *, sequence: Callable[[], int],
        check: Callable[[], None],
    ) -> TombstoneIndex:
        """Stream a complete scope in 1,024-row chunks and verify its sequence.

        Args:
            reader: The owner's bounded storage reader. Its physical page cap
                is 512, so two pages supply one logical chunk.
            scope: Exact scope; no parent expansion.
            sequence: Reads the required ``tombstone_seq:{scope_id}`` counter.
            check: Validates the owner's read guard/lease; called before and
                after storage reads and whenever content is requested.

        Returns:
            A fully loaded index implementing ``TombstoneSource``.

        Raises:
            AMLError: A revoked guard, changed counter, or invalid stored row.
            Exception: Storage failures propagate, never producing an empty index.
        """
        check()
        current = sequence()
        if type(current) is not int or current < 0:
            raise AMLError("parity_mismatch", "Invalid tombstone sequence")
        index = cls(scope, current, sequence=sequence, check=check)
        try:
            rows: list[TombstoneRecord] = []
            for chunk in batched(index._stream(reader), TOMBSTONE_CHUNK_SIZE):
                for row in chunk:
                    key = (row.entity_id, row.statement_digest)
                    if key in index._keys:
                        raise AMLError("parity_mismatch", "Duplicate tombstone key")
                    index._keys[key] = row
                    rows.append(row)
            index._validate()
            index._rows = tuple(rows)
            index._ready = True
            return index
        except BaseException:
            index.close()
            raise

    def _stream(self, reader: IndexedReader) -> Iterator[TombstoneRecord]:
        query = TombstoneRows(scope_kind=self.scope_kind, scope_id=self.scope_id)
        after: str | None = None
        while True:
            self._validate()
            page = reader.page(query, after=after, limit=MAX_PAGE_SIZE)
            self._validate()
            for raw in page:
                try:
                    row = TombstoneRowV1.model_validate(raw)
                except ValidationError:
                    raise AMLError("parity_mismatch", "Invalid stored tombstone") from None
                if (row.scope_kind, row.scope_id) != (self.scope_kind, self.scope_id):
                    raise AMLError("scope_denied", "Tombstone belongs to another scope")
                if row.tombstone_seq > self._tombstone_seq or (
                    after is not None and row.object_id <= after
                ):
                    raise AMLError("parity_mismatch", "Tombstone page differs from its pin")
                after = row.object_id
                yield TombstoneRecord(
                    object_id=row.object_id, scope_kind=row.scope_kind, scope_id=row.scope_id,
                    entity_id=row.entity_id, about_entity_ids=tuple(row.about_entity_ids),
                    statement=row.statement, statement_digest=row.statement_digest, kind=row.kind,
                    valid_from=row.valid_from, valid_to=row.valid_to,
                    time_granularity=row.time_granularity, embedding=tuple(row.embedding),
                    embedding_space=row.embedding_space, tombstone_seq=row.tombstone_seq,
                )
            if len(page) < MAX_PAGE_SIZE:
                return

    @property
    def scope_kind(self) -> ScopeKind:
        """The exact scope's kind."""
        return self._scope.scope_kind

    @property
    def scope_id(self) -> str:
        """The exact scope's id."""
        return self._scope.scope_id

    @property
    def tombstone_seq(self) -> int:
        """The sequence this index represents; revoked indexes refuse even the pin."""
        self._readable()
        return self._tombstone_seq

    def tombstones(self) -> tuple[TombstoneRecord, ...]:
        """Return the complete scope in ascending object-id order after checking its fence."""
        self._readable()
        return self._rows

    def tombstone(self, entity_id: str, statement_digest: str) -> TombstoneRecord | None:
        """Read the unique ``(entity_id, statement_digest)`` pair for settlement (B.8)."""
        self._readable()
        return self._keys.get((entity_id, statement_digest))

    def close(self) -> None:
        """Drop retained plaintext and vectors after the owner's callbacks drain (D.6)."""
        self._closed = True
        self._ready = False
        self._rows = ()
        self._keys.clear()

    def _readable(self) -> None:
        self._validate()
        if not self._ready:
            raise AMLError("delivery_stale", "Tombstone index is not loaded")

    def _validate(self) -> None:
        if self._closed:
            raise AMLError("delivery_stale", "Tombstone index is closed")
        try:
            self._check()
            if self._sequence() != self._tombstone_seq:
                raise AMLError("delivery_stale", "Tombstone sequence changed")
        except BaseException:
            self.close()
            raise
