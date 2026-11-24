"""Workspace-owned, explicitly versioned candidate reuse; never a result cache."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Iterable, Iterator, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from sys import getsizeof
import threading
from typing import Any

from ...execution.handlers.near import _PreparedVectors


def _size(value: Any) -> int:
    """Conservatively account Python-owned candidate payload bytes."""
    if isinstance(value, Mapping):
        return getsizeof(value) + sum(_size(key) + _size(item) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return getsizeof(value) + sum(_size(item) for item in value)
    return getsizeof(value)


@dataclass
class _Candidates:
    version: str
    rows: list[dict[str, Any]]
    size: int
    vectors: dict[str, _PreparedVectors] = field(default_factory=dict)


class _CandidateCache:
    """Retain one complete candidate generation per entity for concurrent readers.

    Readers of one workspace share its lock, so this cache guards its own
    bookkeeping: eviction, reuse, matrix preparation, and publication happen
    under ``_lock``; row iteration and candidate building happen outside it.
    Owned writes invalidate under the workspace write lock.
    """

    def __init__(self) -> None:
        self.entries: OrderedDict[str, _Candidates] = OrderedDict()
        self.bytes = 0
        self._lock = threading.Lock()

    def invalidate(self, entity_id: str | None = None) -> None:
        """Discard an entity generation or every owned candidate generation."""
        with self._lock:
            self._discard(entity_id)

    def _discard(self, entity_id: str | None = None) -> None:
        if entity_id is None:
            self.entries.clear()
            self.bytes = 0
        elif entry := self.entries.pop(entity_id, None):
            self.bytes -= entry.size

    def _evict(self, max_bytes: int, max_entries: int) -> None:
        while self.entries and (self.bytes > max_bytes or len(self.entries) > max_entries):
            self._discard(next(iter(self.entries)))

    def prepare(self, entity_id: str, version: str | None, vector_field: str, *, max_bytes: int, max_entries: int) -> _PreparedVectors | None:
        """Reuse a coordinate matrix only while it fits the same cache budget."""
        if max_bytes < 0 or max_entries < 0:
            raise ValueError("near cache bounds must be non-negative")
        with self._lock:
            self._evict(max_bytes, max_entries)
            entry = self.entries.get(entity_id)
            if not max_entries or entry is None or version is None or entry.version != version or self.bytes > max_bytes:
                return None
            if vector_field in entry.vectors:
                return entry.vectors[vector_field]
            try:
                import numpy  # noqa: F401
            except ImportError:
                return None
            # Check a conservative allocation estimate before constructing the matrix.
            try:
                estimate = sum((len(row[vector_field]) * 8 + 256) for row in entry.rows if row.get(vector_field) is not None)
                if self.bytes + estimate > max_bytes:
                    return None
                prepared = _PreparedVectors(entry.rows, vector_field)
            except (TypeError, ValueError):
                # Preparation is an optimization over the whole entity. Invalid
                # vectors outside this query's filter must not change its behavior;
                # admitted batches still validate and reject invalid coordinates.
                return None
            size = getsizeof(prepared) + getsizeof(prepared.rows) + _size(prepared.positions) + getsizeof(prepared.values) + getsizeof(prepared.norms) + 256
            if self.bytes + size > max_bytes:
                return None
            entry.vectors[vector_field] = prepared
            entry.size += size
            self.bytes += size
            return prepared

    def rows(
        self,
        entity_id: str,
        version: str | None,
        source: Callable[[], Iterable[dict[str, Any]]],
        *,
        max_bytes: int,
        max_entries: int,
    ) -> Iterator[dict[str, Any]]:
        """Borrow candidates internally; publish only an exhausted, bounded source.

        Overflow, cancellation, and provider errors never publish partial data.
        Concurrent readers may build the same generation; the last complete
        build wins and earlier retained rows keep serving their iterators.
        Ordinary queries bypass this cache even when a generation is retained.
        """
        if max_bytes < 0 or max_entries < 0:
            raise ValueError("near cache bounds must be non-negative")
        bypass = version is None or not max_bytes or not max_entries
        reuse: list[dict[str, Any]] | None = None
        with self._lock:
            self._evict(max_bytes, max_entries)
            if not bypass:
                entry = self.entries.get(entity_id)
                if entry is not None and entry.version == version:
                    self.entries.move_to_end(entity_id)
                    reuse = entry.rows
                else:
                    self._discard(entity_id)
        if bypass:
            yield from source()
            return
        if reuse is not None:
            yield from reuse
            return
        retained: list[dict[str, Any]] | None = []
        size = getsizeof(retained)
        iterator = iter(source())
        try:
            for row in iterator:
                if retained is not None:
                    owned = deepcopy(row)
                    row_size = _size(owned)
                    # Reserve list pointers and allocation slack conservatively.
                    if size + row_size + 16 > max_bytes:
                        retained = None
                    else:
                        retained.append(owned)
                        size += row_size + 16
                        row = owned
                yield row
        finally:
            close = getattr(iterator, "close", None)
            if callable(close):
                close()
        if retained is not None and size <= max_bytes:
            with self._lock:
                self._discard(entity_id)
                while self.entries and (len(self.entries) >= max_entries or self.bytes + size > max_bytes):
                    self._discard(next(iter(self.entries)))
                self.entries[entity_id] = _Candidates(version, retained, size)
                self.bytes += size
