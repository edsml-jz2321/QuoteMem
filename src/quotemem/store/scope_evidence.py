"""One process-local index of each scope's Evidence headers, kept current by journal.

Every Add of a long user read its scope's complete Evidence headers three
times: the reception census, the previous-batch selection and the freeze's
support-order pages. Each read was one or more SQL statements over every row
of the scope, a HeavenBase row copy and a pydantic validation per row, so the
Add cost grew with the user's whole history.

This index reads a scope's headers once, then applies only the rows the
Writer's retirement ledger journals as retired since (``RowCache
.evidence_retired_since``): a new row, a changed row, a re-scoped row and a
deleted row each name their key, so an Add re-reads only its own rows. A full
retirement or a forgotten journal prefix rescans the scope. A snapshot is
exactly what a complete scan started at its token returns, the same guarantee
a fresh SQL scan gives its caller; every consumer keeps its own borrow checks.

Nothing here is durable, shared across processes or a source of authority.
"""

from __future__ import annotations

import threading
from bisect import bisect_left
from collections import OrderedDict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from quotemem.core.query import ScopeV1
from quotemem.core.work import FinalizeSourceHeaderV1
from quotemem.store.indexed import IndexedReader, finalize_source_order
from quotemem.store.row_cache import (
    MEMBERSHIP_ANY,
    RowCache,
    membership_family,
    membership_key,
    note_reads,
    suspended_reads,
)

if TYPE_CHECKING:
    from quotemem.core.identity import ContentBindingDigest

SCOPE_EVIDENCE_BYTES = 128 * 1024 * 1024
"""Approximate resident bytes of indexed headers before the least recent scopes go.

A header and its support position are charged 1 KB; a 6,000-row BEAM-1M user
is about 6 MB. A scope larger than the budget is not retained and every read
of it scans, as before the index.
"""

_HEADER_BYTES = 1024

type ScopeKey = tuple[str, str]
type RowElement = tuple[ContentBindingDigest, int]
"""A row's row-state digest and its evidence-set element (``finalize_heads``)."""


def scope_keys(scope: ScopeV1) -> tuple[tuple[str, str], ...]:
    """The membership keys a complete scan of ``scope``'s Evidence enumerates."""
    return (
        MEMBERSHIP_ANY, membership_family("evidence"),
        membership_key("evidence", f"{scope.scope_kind}:{scope.scope_id}"),
    )


@dataclass(frozen=True, slots=True)
class ScopeEvidence:
    """Every Evidence header of one scope, current as of ``token``.

    Attributes:
        token: The admission token read before the snapshot's reads; a row the
            ledger has not retired since holds exactly the header here.
        headers: Every row of the scope, whatever its state, in ascending
            ``object_id`` order (``IndexedReader.scope_headers``). Shared:
            callers must not change it.
        support: The active rows in the finalize support order
            (``finalize_source_order``).
        order: Each ``support`` row's order key, for bisection.
        timed: The rows of every state that carry an expiry.
        max_seq: The largest ``ingested_commit_seq`` of any row, or -1.
    """

    token: int
    headers: Mapping[str, FinalizeSourceHeaderV1]
    support: tuple[FinalizeSourceHeaderV1, ...]
    order: tuple[tuple[int, int, str], ...]
    timed: tuple[FinalizeSourceHeaderV1, ...]
    max_seq: int

    def note(self, scope: ScopeV1) -> None:
        """Name what a complete ``scope_headers`` scan names in active read records."""
        note_reads((*scope_keys(scope), *(("evidence", key) for key in self.headers)))

    def eligible(self, freeze_time: int, receipt_high_watermark: int, tenant_id: str,
                 ) -> tuple[FinalizeSourceHeaderV1, ...]:
        """The rows ``finalize_evidence_page`` returns at one time and commit cutoff.

        An active row ingested at or before the cutoff that has not expired,
        and no row of whose original message (scope, session, request,
        ordinal), of any state and of ``tenant_id``, ingested at or before the
        cutoff, has expired (``_finalize_evidence_predicate``), in the support
        order.
        """
        expired = {
            (header.session_id, header.request_id, header.ordinal) for header in self.timed
            if header.tenant_id == tenant_id
            and header.ingested_commit_seq <= receipt_high_watermark
            and header.expires_at is not None and header.expires_at <= freeze_time
        }
        if not expired and self.max_seq <= receipt_high_watermark:
            return self.support
        return tuple(
            header for header in self.support
            if header.ingested_commit_seq <= receipt_high_watermark
            and (header.expires_at is None or header.expires_at > freeze_time)
            and (header.session_id, header.request_id, header.ordinal) not in expired
        )


def _build(token: int, headers: dict[str, FinalizeSourceHeaderV1]) -> ScopeEvidence:
    support = sorted(
        (header for header in headers.values() if header.state == "active"),
        key=finalize_source_order,
    )
    return ScopeEvidence(
        token, headers, tuple(support), tuple(finalize_source_order(h) for h in support),
        tuple(header for header in headers.values() if header.expires_at is not None),
        max((header.ingested_commit_seq for header in headers.values()), default=-1),
    )


def _apply(
    token: int, previous: ScopeEvidence,
    changed: Mapping[str, FinalizeSourceHeaderV1 | None],
) -> ScopeEvidence:
    """Replace, add or remove the ``changed`` rows of ``previous``; None removes one.

    The result equals ``_build`` over the changed rows: the identity order of
    ``headers``, the support order of ``support`` and the identity order of
    ``timed`` are kept, each changed row moved by bisection.
    """
    headers = dict(previous.headers)
    support, order = list(previous.support), list(previous.order)
    last = next(reversed(headers), None)
    ascending = True
    olds: list[FinalizeSourceHeaderV1] = []
    for identifier, header in changed.items():
        old = headers.get(identifier)
        if old is not None:
            olds.append(old)
            if old.state == "active":
                position = bisect_left(order, finalize_source_order(old))
                if position == len(support) or support[position].object_id != identifier:
                    return _build(token, _rebuilt(previous.headers, changed))
                del support[position], order[position]
        if header is None:
            if old is not None:
                del headers[identifier]
            continue
        if old is None:
            if last is not None and identifier < last:
                ascending = False
            else:
                last = identifier
        # An existing row keeps its place in the identity order.
        headers[identifier] = header
        if header.state == "active":
            key = finalize_source_order(header)
            position = bisect_left(order, key)
            support.insert(position, header)
            order.insert(position, key)
    if not ascending:
        headers = dict(sorted(headers.items()))
    news = [header for header in changed.values() if header is not None]
    timed = previous.timed
    if any(header.expires_at is not None for header in (*olds, *news)):
        timed = tuple(header for header in headers.values() if header.expires_at is not None)
    max_seq = max((previous.max_seq, *(header.ingested_commit_seq for header in news)))
    if any(old.ingested_commit_seq >= max_seq for old in olds):
        max_seq = max((header.ingested_commit_seq for header in headers.values()), default=-1)
    return ScopeEvidence(token, headers, tuple(support), tuple(order), timed, max_seq)


def _rebuilt(
    headers: Mapping[str, FinalizeSourceHeaderV1],
    changed: Mapping[str, FinalizeSourceHeaderV1 | None],
) -> dict[str, FinalizeSourceHeaderV1]:
    merged = {key: value for key, value in headers.items() if key not in changed}
    merged.update((key, value) for key, value in changed.items() if value is not None)
    return dict(sorted(merged.items()))


class ScopeEvidenceIndex:
    """The per-scope header snapshots and row elements of one Writer's process.

    ``snapshot`` refreshes one scope under that scope's lock; scopes refresh
    concurrently. Row elements are per-row values a consumer derived from the
    complete row (``store_elements``); a row's element is dropped whenever its
    row is journaled, and is stored only while the ledger proves the row
    unchanged since the token its reader read before the row.
    """

    def __init__(self, cache: RowCache, budget: int = SCOPE_EVIDENCE_BYTES) -> None:
        self._cache = cache
        self.budget = budget
        self._lock = threading.Lock()
        self._locks: dict[ScopeKey, threading.Lock] = {}
        self._scopes: OrderedDict[ScopeKey, ScopeEvidence] = OrderedDict()
        self._elements: dict[ScopeKey, dict[str, RowElement]] = {}
        self._bytes = 0

    def _scope_lock(self, key: ScopeKey) -> threading.Lock:
        with self._lock:
            lock = self._locks.get(key)
            if lock is None:
                lock = self._locks[key] = threading.Lock()
            return lock

    def snapshot(
        self, reader: IndexedReader, scope: ScopeV1,
        check: Callable[[], None] = lambda: None,
    ) -> ScopeEvidence:
        """Return ``scope``'s headers current as of a token read during this call.

        Args:
            reader: An indexed reader of the Writer's workspace.
            scope: The exact scope.
            check: Called after each page of a complete scan.

        Returns:
            The snapshot. Its reads are suspended from active read records;
            a consumer names what it used (``ScopeEvidence.note``).
        """
        key = (scope.scope_kind, scope.scope_id)
        with self._scope_lock(key), suspended_reads():
            token = self._cache.token
            with self._lock:
                previous = self._scopes.get(key)
            retired = None if previous is None else self._cache.evidence_retired_since(
                previous.token,
            )
            if previous is not None and retired is not None:
                if not retired:
                    self._touch(key, previous)
                    return previous
                changed: dict[str, FinalizeSourceHeaderV1 | None] = {}
                for identifier, header in reader.source_headers(retired).items():
                    if header is not None and (header.scope_kind, header.scope_id) == key:
                        changed[identifier] = header
                    elif identifier in previous.headers:
                        changed[identifier] = None
                current = _apply(token, previous, changed)
                self._forget_elements(key, retired)
            else:
                headers: dict[str, FinalizeSourceHeaderV1] = {}
                after: str | None = None
                while page := reader.scope_headers(scope, after=after):
                    check()
                    for header in page:
                        headers[header.object_id] = header
                    after = page[-1].object_id
                current = _build(token, headers)
                with self._lock:
                    self._elements.pop(key, None)
            self._keep(key, current)
            return current

    def _touch(self, key: ScopeKey, value: ScopeEvidence) -> None:
        with self._lock:
            if self._scopes.get(key) is value:
                self._scopes.move_to_end(key)

    def _keep(self, key: ScopeKey, value: ScopeEvidence) -> None:
        size = _HEADER_BYTES * len(value.headers)
        with self._lock:
            previous = self._scopes.pop(key, None)
            if previous is not None:
                self._bytes -= _HEADER_BYTES * len(previous.headers)
            if size > self.budget:
                self._elements.pop(key, None)
                return
            while self._scopes and self._bytes + size > self.budget:
                oldest, evicted = self._scopes.popitem(last=False)
                self._bytes -= _HEADER_BYTES * len(evicted.headers)
                self._elements.pop(oldest, None)
            self._scopes[key] = value
            self._bytes += size

    def _forget_elements(self, key: ScopeKey, identifiers: Iterable[str]) -> None:
        with self._lock:
            elements = self._elements.get(key)
            if elements:
                for identifier in identifiers:
                    elements.pop(identifier, None)

    def elements(self, scope: ScopeV1) -> dict[str, RowElement]:
        """A copy of the scope's stored row elements by Evidence id."""
        with self._lock:
            return dict(self._elements.get((scope.scope_kind, scope.scope_id), {}))

    def store_elements(
        self, scope: ScopeV1, token: int, values: Mapping[str, RowElement],
    ) -> None:
        """Keep each row's element unless its row was retired after ``token``.

        ``token`` must be the admission token read before the rows were read.
        A row retired since is left out: its element may describe an older row.
        """
        key = (scope.scope_kind, scope.scope_id)
        with self._scope_lock(key), self._lock:
            if key not in self._scopes:
                return
            elements = self._elements.setdefault(key, {})
            for identifier, value in values.items():
                if not self._cache.changed_since((("evidence", identifier),), token):
                    elements[identifier] = value
