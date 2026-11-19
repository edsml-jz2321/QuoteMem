"""Process-wide memos of decoded memory snapshots and their unit vectors (PERF-1).

Every Add and Search view decodes the head snapshot of each Item of its scope
(``RecallSnapshot`` keeps ``snapshot_json``, the canonical text), and every
nomination, novelty and dense score divides each head's embedding by its norm.
Each view used to repeat both for every head of the scope, so an Add cost time
linear in its user's history several times over: in a 10-user LoCoMo run
(fixed provider) decoding alone was 8% of the service's CPU.

A snapshot is a pure function of its canonical text and a unit vector of its
embedding, so both are remembered here:

- ``decoded_snapshot`` keeps at most ``SNAPSHOT_MEMO_SIZE`` decoded snapshots by
  their exact text. Projection rows keep the same text object while their
  head is unchanged, so a lookup hashes it once. Every view reads all heads of
  its scope in turn, so a scope with more heads than the memo holds would make
  any recency eviction miss on every read (a single-user chain past 4,096
  Items did, and its service's memory ran away). The memo therefore evicts
  only snapshots not read for ``_STALE_READS`` reads, sweeps for them at most
  once per ``_SWEEP_INTERVAL`` refused admissions, and otherwise decodes
  without keeping: an oversized scope then hits for the heads it holds.
- ``snapshot_unit`` keeps the unit vector of a decoded snapshot for as long as
  that very snapshot object lives (``IdentityMemo``).

Callers must treat returned snapshots as read-only, as every view already had
to: one decoded snapshot was already shared by all readers of its view.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from typing import Final, cast

from aml.core.entities import MemorySnapshotV2
from aml.core.gc_safe import IdentityMemo
from aml.core.vector_score import UnitArray, UnitRows, UnitVector, unit_array, unit_vector

SNAPSHOT_MEMO_SIZE: Final = 131072
"""Decoded snapshots kept per process; each holds its embedding (about 38 KiB at 1024).

A full memo holds about 4.8 GiB. At 4,096 a single scope of 12.9k Items (in a
deployment run) decoded about 8.8k heads again on every view of it, roughly 0.25-0.5 ms
each. 32,768 held the largest scopes until a single textual user passed 59k Items
(in a deployment run) and every Search decoded its overflow again; 131,072 holds it.
"""

_SWEEP_INTERVAL: Final = SNAPSHOT_MEMO_SIZE // 8
"""Refused admissions between two sweeps of a full memo."""

_STALE_READS: Final = 16 * SNAPSHOT_MEMO_SIZE
"""Memo reads after its last read that make a kept snapshot evictable.

A scope's views read each of its heads once per pass, so a head of a scope
with fewer heads than this is read again before it turns stale; a head
replaced by a newer version is never read again and leaves at a sweep.
"""

_decoded: dict[str, list[MemorySnapshotV2 | int]] = {}
"""Each kept text's ``[snapshot, memo read count at its last read]``."""
_decoded_lock = threading.Lock()
_reads = 0
_refused = 0
_units: IdentityMemo[MemorySnapshotV2, UnitVector] = IdentityMemo()
_arrays: IdentityMemo[MemorySnapshotV2, UnitArray] = IdentityMemo()


def decoded_snapshot(snapshot_json: str) -> MemorySnapshotV2:
    """Return the snapshot that canonical text decodes to, decoding it at most once while kept.

    Args:
        snapshot_json (str): A head's stored canonical snapshot text.

    Returns:
        MemorySnapshotV2: Its decoded snapshot, shared read-only.

    Raises:
        ValidationError: When the text is not a valid snapshot.
    """
    global _reads
    # The count is advisory: a lost increment under concurrent readers only
    # delays an eviction.
    _reads += 1
    entry = _decoded.get(snapshot_json)
    if entry is not None:
        entry[1] = _reads
        return cast(MemorySnapshotV2, entry[0])
    snapshot = MemorySnapshotV2.model_validate_json(snapshot_json)
    remember_snapshot(snapshot_json, snapshot)
    return snapshot


def remember_snapshot(snapshot_json: str, snapshot: MemorySnapshotV2) -> None:
    """Keep a snapshot just decoded from exactly ``snapshot_json``, when the memo has room.

    Args:
        snapshot_json (str): The canonical text.
        snapshot (MemorySnapshotV2): What that text decodes to.
    """
    global _refused
    evicted: list[list[MemorySnapshotV2 | int]] = []
    with _decoded_lock:
        if len(_decoded) >= SNAPSHOT_MEMO_SIZE:
            if _refused < _SWEEP_INTERVAL:
                _refused += 1
                return
            _refused = 0
            oldest = _reads - _STALE_READS
            for key, entry in list(_decoded.items()):
                if cast(int, entry[1]) < oldest:
                    evicted.append(_decoded.pop(key))
            if len(_decoded) >= SNAPSHOT_MEMO_SIZE:
                return
        _decoded[snapshot_json] = [snapshot, _reads]
    # Released outside the lock: a dying snapshot runs the unit memo's callback.
    evicted.clear()


def snapshot_unit(snapshot: MemorySnapshotV2) -> UnitVector:
    """Return the unit vector of a snapshot's embedding, computed once per snapshot object.

    Args:
        snapshot (MemorySnapshotV2): A content-bearing snapshot.

    Returns:
        UnitVector: The exact ``unit_vector(snapshot.embedding)`` values in
            immutable packed float64 storage.

    Raises:
        AMLError: ``binding_mismatch`` for a zero or non-finite embedding, as
            ``unit_vector`` raises.
    """
    unit = _units.recall(snapshot)
    if unit is None:
        unit = unit_vector(snapshot.embedding)
        _units.remember(snapshot, unit)
    return unit


def snapshot_unit_array(snapshot: MemorySnapshotV2) -> UnitArray:
    """Return `snapshot_unit` as a read-only float64 array, converted once per snapshot object.

    Args:
        snapshot (MemorySnapshotV2): A content-bearing snapshot.

    Returns:
        numpy.ndarray: ``unit_array(snapshot_unit(snapshot))``, to stack into ``UnitRows``.
    """
    array = _arrays.recall(snapshot)
    if array is None:
        array = unit_array(snapshot_unit(snapshot))
        _arrays.remember(snapshot, array)
    return array


def snapshot_rows(snapshots: Sequence[MemorySnapshotV2]) -> UnitRows:
    """Stack the unit vectors of content-bearing snapshots, in order, for batched cosines.

    Args:
        snapshots (Sequence[MemorySnapshotV2]): Snapshots that are not erased.

    Returns:
        UnitRows: Their unit vectors, scored by one product per query.
    """
    return UnitRows([snapshot_unit(snapshot) for snapshot in snapshots],
                    [snapshot_unit_array(snapshot) for snapshot in snapshots])


def clear_snapshot_memos() -> None:
    """Forget every kept snapshot and unit vector (tests and memory pressure)."""
    global _refused, _reads
    with _decoded_lock:
        released = list(_decoded.values())
        _decoded.clear()
        _refused = _reads = 0
    released.clear()
    _units.clear()
    _arrays.clear()
