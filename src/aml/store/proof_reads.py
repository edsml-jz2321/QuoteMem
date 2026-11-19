"""Decide whether a write changed a proof's reads per row, not for the whole workspace.

``Writer.ref_generation`` is one counter for every scope: each reference-bearing
write reserves the next value, whichever user it serves. A proof fenced on it
refused whenever any other user wrote anything, several times a second under
plan section 9 normal load, and refused with an integrity code where its fence
assumed only its own owner could move it: the bench controller then quarantined
a healthy scope. A proof records instead every row and enumerated set its
discovery reads, and the Writer's retirement ledger decides per key whether a
later row mutation may have changed one of them (ADR 0036, 0041). Another
scope's write retires none of them, so the proof stays valid; a write that did
is an overtaken read, retried with ``reception_stale``. A fence still rechecks
the rows its proof binds first and keeps its own integrity code for a real
change of those.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Protocol

from aml.core.errors import AMLError
from aml.store.row_cache import MEMBERSHIP_ANY, recording_reads


class ProofLedger(Protocol):
    """The Writer's generation, storage epoch and per-key retirement proof (ADR 0036)."""

    @property
    def ref_generation(self) -> int: ...

    @property
    def storage_epoch(self) -> int: ...

    def borrowed_rows_unchanged(self, keys: Iterable[tuple[str, str]], token: int) -> bool: ...


@dataclass(frozen=True)
class ProofReads:
    """Every row and enumerated set one proof's discovery read.

    Attributes:
        generation: The Writer's reference generation before the first read;
            while it is unchanged no reference-bearing write ran.
        token: The storage epoch read before the first read.
        keys: Each row and membership key the discovery read, with
            ``MEMBERSHIP_ANY``.
        complete: False when the discovery also enumerated a set that no key
            names; then only an unchanged generation proves the reads.
    """

    generation: int
    token: int
    keys: frozenset[tuple[str, str]]
    complete: bool = True


@contextmanager
def recording_proof_reads(writer: ProofLedger) -> Iterator[Callable[[], ProofReads]]:
    """Record the rows one proof's discovery reads in this execution context.

    The generation and storage epoch are read before the block, so every row
    the block reads is current at or after them. Reads on other threads are not
    recorded; a discovery must make its own reads. A read that only decides
    whether the discovery may proceed, or a set its proof rechecks explicitly,
    belongs in ``suspended_reads``.

    Args:
        writer: The sole Writer whose ledger later decides the recorded keys.

    Yields:
        A function returning the reads recorded so far; call it once the
        discovery has made every read its proof depends on.
    """
    generation, token = writer.ref_generation, writer.storage_epoch
    with recording_reads() as record:
        yield lambda: ProofReads(
            generation, token, frozenset((MEMBERSHIP_ANY, *record.keys)), record.complete,
        )


def proof_reads_unchanged(writer: ProofLedger, reads: ProofReads) -> bool:
    """Report whether no write since ``reads`` may have changed what they read.

    A write that reserves no reference generation (a heartbeat-only lease
    renewal, a job's traffic watermark) never counts, as before. Once another
    write ran, the ledger decides per recorded key.

    Args:
        writer: The Writer that issued ``reads``.
        reads: A proof's recorded reads.

    Returns:
        True when no reference-bearing write ran since, or when the ledger
        proves that none of the recorded rows and sets was retired after the
        recorded storage epoch. False otherwise, including when the discovery
        read an unnamed set or the ledger no longer covers that epoch.
    """
    if writer.ref_generation == reads.generation:
        return True
    return reads.complete and writer.borrowed_rows_unchanged(reads.keys, reads.token)


def require_proof_reads(writer: ProofLedger, reads: ProofReads, message: str) -> None:
    """Refuse, as retryable, a proof whose recorded reads a later write may have changed.

    A fence calls this after rechecking the rows its proof binds, which keep
    their own integrity refusals. What remains is an overtaken read, never an
    integrity fault: the caller prepares the proof again.

    Args:
        writer: The Writer that issued ``reads``.
        reads: The proof's recorded reads.
        message: The payload-free explanation for the refusal.

    Raises:
        AMLError: ``reception_stale`` when ``proof_reads_unchanged`` is False.
    """
    if not proof_reads_unchanged(writer, reads):
        raise AMLError("reception_stale", message)
