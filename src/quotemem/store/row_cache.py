"""Process-local memoization of Entity rows read by the sole Writer.

QuoteMem's Writer owns the only process that mutates its HeavenBase workspace, so
between two in-process row mutations every by-identity read returns the same
row, and the finalize consumers repeat those reads by the tens of thousands
under a serialized storage lock. This cache memoizes rows for exactly that
interval: the Writer retires the whole cache immediately before and
immediately after each row mutation reaching the workspace row writer or a
registered backend, whether issued by the Writer, by HeavenBase's own
maintenance or by a test injecting a fault, and a read that overlapped a
mutation is never kept because its admission token no longer matches.

Every hit returns a private copy; a borrower's in-place edit reaches no other
read. Nothing here is durable, shared across processes, or a substitute for
the Writer mutex, reader guards or storage-owner validation.

The same retirement ledger proves complete scans current: a scan
over every row that satisfies a predicate records the rows it read and the
membership keys of the sets it enumerated, and ``ScanMemo`` serves its result
while none of them was retired after the scan's admission token. Those proofs
read the ledger without the cache lock; a sequence counter bracketing every
ledger write makes a prober that overlapped one report a change instead of
trusting the partial write it may have read.
"""

from __future__ import annotations

import threading
import weakref
from array import array
from bisect import bisect_right
from collections.abc import Hashable, Iterable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from typing import Any, cast

from quotemem.service.cost_attribution import attribute_cost, cost_count

ROW_CACHE_CAPACITY = 32768
"""Distinct rows memoized before the least recently used are forgotten.

The byte budget below bounds memory; a finalize over a large scope reads
thousands of Evidence rows while the heartbeat and Adds keep reading theirs,
and a smaller count evicted rows that were still in use.
"""

ROW_CACHE_BYTES = 192 * 1024 * 1024
"""Approximate resident bytes of memoized rows before the least used are forgotten.

A budget below the working set is worse than none: every consumer of a frozen
plan walks its scope's eligible rows in plan order, so a table that holds part
of them forgets each row exactly before the next walk asks for it. The
workspace of a normal-load run holds 10,611 Evidence rows across
seventeen scopes, the largest holding 689, and the cache
probe measures one packed row of a 2,048-value embedding space at 9.2 KB of
this estimate (6,493 rows in 60 MB), so every scope at once is
about 98 MB. At 48 MB that walk re-read 2.3 GB of rows in eighteen minutes;
this value holds every scope with room to spare. A larger workspace still
evicts, and the bound is still a bound.

An earlier note here blamed a host with 8 GB for a tuple allocation that
failed inside a snapshot callback at 192, 192 and 128 MB. That failure was
the observer probe walking the heap with ``gc.get_objects``, which takes a
reference to a tuple another thread is still building so CPython refuses to
resize it; it was not memory pressure, and the budgets it argued down are
restored here. The per-consumer parse memo that multiplied this residency by
every concurrent consumer is bounded separately (``_ROW_MEMO_ENTRIES``).
"""

MISSING: Any = object()
"""Sentinel distinguishing an absent memoized row from a memoized absence."""

_EVERY_KEY: tuple[str, str] = ("", "")
"""Retirement ledger entry standing for every key after a full retirement."""

MEMBERSHIP_ANY: tuple[str, str] = ("membership", "")
"""Ledger key retired by a mutation whose rows cannot be inspected at all.

Every membership proof includes it, so a drop, an iterator or an unknown
mutator shape retires every memoized scan.
"""


def membership_family(family: str) -> tuple[str, str]:
    """Return the wildcard retired when a row of ``family`` joined an unknown set.

    An upsert whose post-image does not carry the field naming its set (a Run
    progress fragment, an Evidence route change) could have joined any set of
    that family, but it cannot have joined a set of another family: a Run row
    is never an Evidence row. A scan includes the wildcard of every family it
    enumerated, so it is refused by such a fragment of its own family only.
    Under normal benchmark load the Add path and the finalize workers
    write Run fragments continuously, and one global wildcard retired every
    memoized scan of every family several times a second.
    """
    return ("membership", f"{family}:")

SCAN_MEMO_CAPACITY = 32768
"""Distinct scans memoized before the least recently used are forgotten."""

SCAN_MEMO_BYTES = 16 * 1024 * 1024
"""Approximate resident bytes of memoized scan values before the least used are forgotten.

A memoized scan holds a session's ordered descriptors or a scope's Candidate
page, so a count alone does not bound this memo. Per-row values are the many
and would evict the few scans they are read for, so they are memoized in
their own table (``ROW_BINDING_BYTES``) rather than here.
"""

ROW_BINDING_BYTES = 448 * 1024 * 1024
"""Approximate resident bytes of memoized per-row values before the least used go.

One Evidence row's canonical bytes and row-state digest are about 42 KB
together in a 2,048-value embedding space (745 entries in
31 MB), and a miss recomputes both from the memoized row without reading
storage. That recomputation is not cheap: it was the single largest consumer of
worker time in the baseline, 95% of every hashing sample and about 23% of every
non-idle sample, because sixteen scopes finalize at
once while this table held roughly three of their working sets.

The largest scope of that workspace holds 689 Evidence rows, so one scope is
about 29 MB and every scope finalizing at once is about 445 MB. This value
holds them. Unlike the rows, these values cannot be packed: they are the exact
canonical bytes their digest is taken over.
"""

INTERACTIVE_ROW_CACHE_BYTES = 64 * 1024 * 1024
"""``ROW_CACHE_BYTES`` of an interactive (plugin) service, which serves one user.

The bench budgets above hold every scope of a sixteen-user shard finalizing at
once. A plugin process holds one user whose finalize and Search walk one scope;
at 3,000 facts the two bench tables filled to 640 MB, 41% of the process.
"""

INTERACTIVE_ROW_BINDING_BYTES = 128 * 1024 * 1024
"""``ROW_BINDING_BYTES`` of an interactive (plugin) service (see above)."""

_BUDGETS: ContextVar[tuple[int, int] | None] = ContextVar("aml_cache_budgets", default=None)


@contextmanager
def cache_budgets(rows: int, bindings: int) -> Iterator[None]:
    """Give the row caches constructed inside this context these byte budgets.

    A service profile opens its workspace inside it, so the Writer's row cache and
    per-row value memo are bounded from their first row. Caches built elsewhere
    keep ``ROW_CACHE_BYTES`` and ``ROW_BINDING_BYTES``.

    Args:
        rows: Positive byte budget of the row cache.
        bindings: Positive byte budget of the per-row value memo.
    """
    if rows < 1 or bindings < 1:
        raise ValueError("Cache budgets must be positive")
    token = _BUDGETS.set((rows, bindings))
    try:
        yield
    finally:
        _BUDGETS.reset(token)


def row_cache_bytes() -> int:
    """The row cache budget of the current context (``cache_budgets``)."""
    budgets = _BUDGETS.get()
    return ROW_CACHE_BYTES if budgets is None else budgets[0]


def row_binding_bytes() -> int:
    """The per-row value memo budget of the current context (``cache_budgets``)."""
    budgets = _BUDGETS.get()
    return ROW_BINDING_BYTES if budgets is None else budgets[1]


FINALIZE_METADATA_MEMO_CAPACITY = 16
"""Verified source-TTL metadata views memoized before the least recently used go.

One entry is one finalize owner's complete source and unit inventories, its
selected disposition ancestry and the live header fence over every member,
under one input, one selected result and one history mode. A few owners whose
dispatch rounds interleave in one process each keep theirs; a forgotten entry
only makes the next call verify the immutable chains and read every current
header again.
"""

FINALIZE_METADATA_MEMO_BYTES = 64 * 1024 * 1024
"""Resident bytes of memoized source-TTL views before the least recently used go.

A view retains about one to three kilobytes of content-free metadata per
original member: its frozen header, its key and row binding, its unit and its
disposition records. That is about three megabytes per thousand frozen sources.
The producer measures each view and passes ``value_bytes``; a view larger than
this budget is not retained, and its caller keeps the uncached verification.
"""


EVIDENCE_JOURNAL_CAPACITY = 1 << 16
"""Evidence row retirements the journal names before its oldest half is forgotten.

A per-scope header index (``quotemem.store.scope_evidence``) asks which Evidence
rows were retired after the token its snapshot was read at. An Add retires a
few dozen; a reader whose token predates the forgotten half rescans its scope.
"""


class ReadRecord:
    """The rows one in-process proof read, so a memo of it is proven by the ledger.

    ``complete`` turns false when the proof enumerated a set it cannot name (an
    index scan): row keys alone cannot cover a member inserted later, so such a
    proof is never memoized by its recorded reads.
    """

    __slots__ = ("complete", "keys")

    def __init__(self) -> None:
        self.keys: set[tuple[str, str]] = set()
        self.complete = True


_READS: ContextVar[tuple[ReadRecord, ...]] = ContextVar("aml_read_records", default=())


@contextmanager
def recording_reads() -> Iterator[ReadRecord]:
    """Record every row this execution context reads until exit, nested records included."""
    record = ReadRecord()
    token = _READS.set((*_READS.get(), record))
    try:
        yield record
    finally:
        _READS.reset(token)


@contextmanager
def suspended_reads() -> Iterator[None]:
    """Keep a gate's reads out of active records.

    A borrow, lease, grant or owner check decides whether a read may proceed,
    not what it returns; its caller reruns that gate on every reuse, so the
    rows it reads are not dependencies of a recorded value.
    """
    token = _READS.set(())
    try:
        yield
    finally:
        _READS.reset(token)


def note_reads(keys: Iterable[tuple[str, str]]) -> None:
    """Name in every active record the rows a memo served without reading them."""
    records = _READS.get()
    if records:
        values = tuple(keys)
        for record in records:
            record.keys.update(values)


def note_unprovable_read() -> None:
    """Mark every active record as having read a set no row key can name."""
    for record in _READS.get():
        record.complete = False


def membership_key(family: str, member: str) -> tuple[str, str]:
    """Return the ledger key retired when a row may have joined ``member`` of ``family``.

    A memoized scan proves the rows it read unchanged per key
    (``RowCache.changed_since``) but cannot name a row that did not exist when
    it ran. The Writer retires the membership key of the scope, key namespace
    or Run family an upsert's post-image names, before and after the mutation,
    so a proof that includes the key covers insertion as well as change. A
    delete retires the deleted row's own key, which every scan that read the
    row already names, and cannot add a row to any set.

    Args:
        family: The enumerated set's kind: ``evidence`` for the Evidence rows
            of a scope, ``control`` for the Controls of one key namespace, or
            ``run`` for the Runs of one idempotency family.
        member: The exact set within the family: ``{scope_kind}:{scope_id}``,
            the Control key without its last segment, or the family key.
    """
    return ("membership", f"{family}:{member}")

_IMMUTABLE = (str, bytes, int, float, bool, type(None))

_EXACT_IMMUTABLE = frozenset(_IMMUTABLE)
"""The exact types of ``_IMMUTABLE``: a decoded row's scalars, handed out as they are.

``copy_row`` serves every row read and every memoized input hand-out, millions
of calls in a 976-position finalize. A row is nearly all such scalars, so
``_thaw`` copies a ``dict`` or ``list`` by checking each element's exact type
inline and calls itself only for a nested container; a subclass of these types
takes the general checks, in their original order.
"""

_FLAT = (int, float, bool, type(None))
"""Scalars whose own object is a fixed size, so a vector of them prices per element.

``str`` and ``bytes`` are immutable too but carry their length, and pricing a
list of them at a fixed size per element understated one of 1,000 one-kilobyte
strings by 29.7 times; they take the recursive branch, which prices each by its
own length.
"""


_VECTOR_DOC = """A stored list of immutable scalars is held as a plain tuple.

An Evidence row carries a 2,048-value embedding, so copying its list one
element at a time cost 0.39 ms every time the row was memoized or handed out;
under normal load the Writer serves about 35,000 row reads a minute. A tuple
cannot be edited in place, so one instance is shared by every stored copy and
rebuilt as a fresh list for each caller (0.006 ms). A decoded storage row
holds only JSON containers, so a tuple in a stored row is one this module made;
``copy_row`` turns every tuple it finds back into a list. An earlier version
used a tuple subclass, which fails CPython's exact-tuple check inside its own
tuple routines and raised ``SystemError`` from a snapshot callback under load.
"""


@attribute_cost("storage.row_copy")
def copy_row(row: dict[str, Any]) -> dict[str, Any]:
    """Return a private plain copy of a stored or freshly decoded row.

    Values a caller could edit in place are copied; stored vectors are
    rebuilt as lists, so the returned row is a complete plain structure
    whatever form it was memoized in.
    """
    return cast(dict[str, Any], _thaw(row))


def sized_copy(row: dict[str, Any]) -> tuple[dict[str, Any], int]:
    """Return the storable copy of a decoded row and its estimated bytes."""
    size = [0]
    return _copy(row, size), size[0]


def _thaw(value: Any) -> Any:
    if type(value) is dict:
        return {key: item if type(item) in _EXACT_IMMUTABLE else _thaw(item)
                for key, item in cast(dict[Any, Any], value).items()}
    if type(value) in _EXACT_IMMUTABLE:
        return value
    if type(value) is list:
        return [item if type(item) in _EXACT_IMMUTABLE else _thaw(item)
                for item in cast(list[Any], value)]
    if isinstance(value, _IMMUTABLE):
        return value
    if isinstance(value, array):
        # A stored embedding vector; ``tolist`` rebuilds the exact doubles
        # ``_copy`` packed, including signed zero, infinities and NaN.
        return cast("array[float]", value).tolist()
    if isinstance(value, tuple):
        # Only ``_copy`` puts a tuple in a stored row and only for a vector of
        # immutable scalars, so one shallow rebuild hands out a private list.
        return list(cast(tuple[Any, ...], value))
    if isinstance(value, dict):
        mapping = cast(dict[Any, Any], value)
        return {key: _thaw(item) for key, item in mapping.items()}
    if isinstance(value, list):
        return [_thaw(item) for item in cast(list[Any], value)]
    return deepcopy(value)


def _copy(value: Any, size: list[int]) -> Any:
    if isinstance(value, str):
        size[0] += 49 + len(value)
        return value
    if isinstance(value, bytes):
        size[0] += 33 + len(value)
        return value
    if isinstance(value, _IMMUTABLE):
        size[0] += 28
        return value
    if isinstance(value, array):
        # Already the stored form of an embedding vector; nothing mutates a
        # stored row and the hand-out rebuilds a private list, so share it.
        vector = cast("array[float]", value)
        size[0] += 64 + 8 * len(vector)
        return vector
    if isinstance(value, dict):
        mapping = cast(dict[Any, Any], value)
        size[0] += 64 + 16 * len(mapping)
        return {key: _copy(item, size) for key, item in mapping.items()}
    if isinstance(value, list):
        items = cast(list[Any], value)
        if items and all(type(item) is float for item in items):
            # An embedding vector. Held as Python floats it costs a pointer
            # and a boxed object per value, 65 KB for the 2,048 of one
            # Evidence row and 615 MB for one normal-load
            # scope, which no bound this cache can carry would hold; packed
            # as doubles it is 16 KB and the same scope is 190 MB. ``array``
            # stores the exact double, so the hand-out rebuilds the vector
            # the row was decoded with. Only an exact ``float`` is packed: a
            # vector carrying an ``int`` would hand back a float and change
            # the row's canonical bytes.
            size[0] += 64 + 8 * len(items)
            return array("d", items)
        size[0] += 56 + 8 * len(items)
        if items and all(isinstance(item, _FLAT) for item in items):
            # A vector of fixed-size scalars: store one immutable copy and let
            # every stored copy share it, rather than recursing per element.
            size[0] += 28 * len(items)
            return tuple(items)
        return [_copy(item, size) for item in items]
    if isinstance(value, tuple):
        stored = cast(tuple[Any, ...], value)
        size[0] += 40 + 8 * len(stored)
        if stored and all(isinstance(item, _FLAT) for item in stored):
            # Already the stored form of a scalar vector; share it as it is.
            size[0] += 28 * len(stored)
            return stored
        # Every tuple a stored row holds is a scalar vector, so the hand-out
        # rebuilds one as a list without inspecting its elements; a tuple that
        # holds a container is stored as a list of private copies instead.
        return [_copy(item, size) for item in stored]
    size[0] += 256
    return deepcopy(value)


class ScanExtension:
    """Result of observing one exact artifact write while extending a scan proof.

    ``token`` stays absent unless every retirement in the window is either
    unrelated to the scan or an expected new Artifact and its owner membership
    in the same mutation event. This is cache validity, never write authority.
    """

    __slots__ = ("token", "_keys", "_rows", "_membership", "_valid")

    def __init__(
        self, keys: frozenset[tuple[str, str]], rows: frozenset[tuple[str, str]],
        membership: tuple[str, str], valid: bool,
    ) -> None:
        self.token: int | None = None
        self._keys, self._rows, self._membership = keys, rows, membership
        self._valid = valid

    def _observe(self, keys: list[tuple[str, str]]) -> None:
        changed = set(keys)
        if any(key in self._keys and key != self._membership for key in changed) or any(
            key[0] == "artifact" and key not in self._rows for key in changed
        ) or self._membership in changed and not changed.intersection(self._rows):
            self._valid = False


class RetirementWatch:
    """Whether every retirement since one admission token named a permitted key.

    The opener names the rows its own writes and heartbeats retire. Any other
    retirement, and any full retirement, ends the watch's validity for good;
    so does ending the watch. This is cache validity, never write authority.
    """

    __slots__ = ("__weakref__", "token", "violation", "_allowed", "_valid")

    def __init__(self, token: int, allowed: frozenset[tuple[str, str]]) -> None:
        self.token = token
        # The first key retired outside ``allowed``, for diagnosis only.
        self.violation: tuple[str, str] | None = None
        self._allowed = allowed
        self._valid = True

    @property
    def valid(self) -> bool:
        """Whether only permitted keys were retired since ``token`` while watched."""
        return self._valid

    def _observe(self, keys: list[tuple[str, str]]) -> None:
        for key in keys:
            if key not in self._allowed:
                if self._valid:
                    self.violation = key
                self._valid = False
                return


class BorrowWatch:
    """Whether none of one borrow's keys was retired after its admission token.

    ``RowCache.watch_borrow`` opens it with the exact ``changed_since`` answer
    for the borrow at that instant; every later retirement of a borrowed key,
    and any full retirement, ends its validity for good. A borrower that
    rechecks the same large key set between every row it reads (the finalize
    reception census) then pays once per mutation instead of once per key and
    check. This is cache validity, never write authority.
    """

    __slots__ = ("__weakref__", "token", "_keys", "_noted", "_valid")

    def __init__(self, token: int, keys: frozenset[tuple[str, str]], valid: bool) -> None:
        self.token = token
        self._keys = keys
        # The active read records last given ``keys``; a record's set only grows.
        self._noted: tuple[ReadRecord, ...] = ()
        self._valid = valid

    def _note(self) -> None:
        """Name the borrowed keys in every active read record, once per record set."""
        records = _READS.get()
        if records and records is not self._noted:
            for record in records:
                record.keys.update(self._keys)
            self._noted = records

    def _observe(self, keys: list[tuple[str, str]]) -> None:
        if self._valid and not self._keys.isdisjoint(keys):
            self._valid = False


class RowCache:
    """Memoize rows between this owner's raw storage mutations.

    ``lookup`` returns the admission token current at the read together with
    the memoized row, or ``MISSING``. ``store`` keeps a row only while that
    token is still current, so a read that started before a mutation and
    finished after it is discarded. ``retire`` advances the token and drops
    every row; the Writer calls it around each raw mutation.

    ``changed_since`` takes no lock. Every ledger write (``retire``,
    ``invalidate``, ``_prune``) runs under the cache lock and increments
    ``_seq`` to an odd value before its first ledger store and to an even
    value after its last; a prober reads ``_seq`` before and after its ledger
    reads and reports a change whenever either value is odd or the two
    differ. Its reads therefore all fall inside one quiescent interval of
    the ledger, or the answer is a change.
    """

    __slots__ = (
        "_bytes", "_floor", "_lock", "_retired", "_rows", "_seq", "_token", "budget", "capacity",
        "_borrows", "_extensions", "_watches", "_journal_ids", "_journal_tokens", "_journal_floor",
    )

    def __init__(
        self, capacity: int = ROW_CACHE_CAPACITY, budget: int | None = None,
    ) -> None:
        if budget is None:
            budget = row_cache_bytes()
        if capacity < 1 or budget < 1:
            raise ValueError("Row cache capacity and budget must be positive")
        self._lock = threading.Lock()
        self._token = 0
        self._bytes = 0
        self._rows: dict[tuple[str, str], tuple[dict[str, Any] | None, int]] = {}
        self._retired: dict[tuple[str, str], int] = {}
        self._floor = 0
        self._seq = 0
        self.capacity = capacity
        self.budget = budget
        self._extensions: set[ScanExtension] = set()
        # Held weakly: a watch whose opener is gone observes nothing for anyone.
        self._watches: weakref.WeakSet[RetirementWatch] = weakref.WeakSet()
        # Only borrows still valid; one ends for good on its first observed change.
        self._borrows: weakref.WeakSet[BorrowWatch] = weakref.WeakSet()
        # Every retired Evidence row key in token order, for ``evidence_retired_since``.
        self._journal_tokens: list[int] = []
        self._journal_ids: list[str] = []
        self._journal_floor = 0

    def evidence_retired_since(self, token: int) -> tuple[str, ...] | None:
        """Name every Evidence row retired after ``token``, or None when that is unknowable.

        Each ``invalidate`` that names an ``("evidence", object_id)`` key
        journals the object id under the token it publishes, in the same
        locked ledger write; a full retirement and a forgotten journal prefix
        answer None. Since every in-process mutation of an Evidence row names
        its key or retires everything, a row absent from the answer was not
        mutated after ``token``, which is the proof ``changed_since`` gives for
        a named key, extended to rows the asker cannot name (a new row).

        Args:
            token: An admission token read before the asker's own reads.

        Returns:
            The distinct object ids retired after ``token``, oldest first, or
            None when a full retirement or the journal floor covers ``token``.
        """
        with self._lock:
            if token < self._journal_floor or token < self._retired.get(_EVERY_KEY, 0):
                return None
            start = bisect_right(self._journal_tokens, token)
            return tuple(dict.fromkeys(self._journal_ids[start:]))

    def watch_retirements(self, allowed: Iterable[tuple[str, str]]) -> RetirementWatch:
        """Open a watch over every later retirement, until ``end_watch`` or its collection."""
        with self._lock:
            watch = RetirementWatch(self._token, frozenset(allowed))
            self._watches.add(watch)
            return watch

    def end_watch(self, watch: RetirementWatch) -> None:
        """Stop observing for ``watch``; it is no longer valid afterwards."""
        with self._lock:
            self._watches.discard(watch)
            watch._valid = False  # pyright: ignore[reportPrivateUsage]

    def watch_borrow(self, keys: Iterable[tuple[str, str]], token: int) -> BorrowWatch:
        """Open a watch whose validity equals ``not changed_since(keys, token)`` from now on.

        The opening answer is decided under the cache lock, where no ledger
        write is open, by the same rules as ``changed_since``; a borrow that has
        already changed is returned invalid and never observes anything.
        """
        borrowed = frozenset(keys)
        with self._lock:
            ledger = self._retired
            valid = token >= max(self._floor, ledger.get(_EVERY_KEY, 0)) and not any(
                ledger.get(key, 0) > token for key in borrowed
            )
            watch = BorrowWatch(token, borrowed, valid)
            if valid:
                self._borrows.add(watch)
            return watch

    def borrow_unchanged(self, watch: BorrowWatch) -> bool:
        """Answer ``not changed_since`` for ``watch``'s borrow in O(1), without the lock.

        Every ledger write observes the open borrows before its closing
        ``_seq`` increment, so a probe that reads an even, unchanged ``_seq``
        around the watch's validity read the validity of one quiescent ledger.
        Anything else is a possible partial write and is reported as a change,
        exactly as ``changed_since`` reports it.
        """
        before = self._seq
        if before & 1:
            return False
        valid = watch._valid  # pyright: ignore[reportPrivateUsage]
        return valid and self._seq == before

    @contextmanager
    def extend_artifact_scan(
        self, keys: Iterable[tuple[str, str]], token: int,
        rows: Iterable[tuple[str, str]], membership: tuple[str, str],
    ) -> Iterator[ScanExtension]:
        """Observe an exact own write without absorbing reentrant storage mutations.

        The Writer supplies physical identities from its ordinary strict
        materializer. It retains its existing storage exclusion and business
        fences; a failed observation only denies cache reuse. The retirement
        observer sees whole events, including an unknown membership invalidation
        that a later expected write would otherwise hide with a newer token.
        """
        dependencies, expected = frozenset(keys), frozenset(rows)
        with self._lock:
            receipt = ScanExtension(
                dependencies, expected, membership,
                not self.changed_since(dependencies, token),
            )
            self._extensions.add(receipt)
        completed = False
        try:
            yield receipt
            completed = True
        finally:
            with self._lock:
                self._extensions.discard(receipt)
                if (
                    completed and receipt._valid  # pyright: ignore[reportPrivateUsage]
                    and token >= self._floor
                ):
                    receipt.token = self._token

    @property
    def token(self) -> int:
        """Return the current admission token; it only advances.

        The read takes no lock: an integer attribute read is atomic under the
        interpreter lock, and a borrower that reads it before its rows only
        needs a token no later than the read.
        """
        return self._token

    @property
    def resident_bytes(self) -> int:
        """Return the estimated resident size of every memoized row."""
        with self._lock:
            return self._bytes

    def __len__(self) -> int:
        with self._lock:
            return len(self._rows)

    @attribute_cost("cache.row_lookup")
    def lookup(self, key: tuple[str, str]) -> tuple[int, dict[str, Any] | None]:
        """Return the admission token and the memoized row or ``MISSING``.

        A hit moves its row to the most recently used end, so the eviction in
        ``store`` forgets rows no reader has asked for lately. Every lookup is a
        read of ``key`` for the active read records.
        """
        for record in _READS.get():
            record.keys.add(key)
        with self._lock:
            entry = self._rows.get(key)
            if entry is None:
                cost_count("cache.row_absent")
                return self._token, MISSING
            cost_count("cache.row_hit")
            self._rows[key] = self._rows.pop(key)
            return self._token, entry[0]

    def store(
        self, token: int, key: tuple[str, str], row: dict[str, Any] | None, size: int = 0,
    ) -> bool:
        """Memoize ``row`` unless a mutation retired the cache since ``token``.

        A row larger than an eighth of the byte budget is never memoized. When
        the row count or the estimated resident bytes would exceed their
        bounds, the least recently used rows are forgotten one at a time until
        the new row fits: a finalize over a large scope and the Adds of other
        scopes share the cache, and clearing all of it on every overflow made
        each of them reread rows the other had just admitted (measured twelve full restarts in six
        minutes, one every thirty seconds).
        Forgetting a row is always sound: it only turns a later read
        of that row into a storage read.
        """
        with self._lock:
            retired = max(self._retired.get(key, 0), self._retired.get(_EVERY_KEY, 0))
            if token < max(retired, self._floor) or size > self.budget // 8:
                return False
            previous = self._rows.pop(key, None)
            if previous is not None:
                self._bytes -= previous[1]
            while self._rows and (
                len(self._rows) >= self.capacity or self._bytes + size > self.budget
            ):
                oldest = next(iter(self._rows))
                self._bytes -= self._rows.pop(oldest)[1]
            self._rows[key] = (row, size)
            self._bytes += size
            return True

    def retire(self) -> None:
        """Drop every memoized row and refuse admissions issued before now."""
        with self._lock:
            for receipt in self._extensions:
                receipt._valid = False  # pyright: ignore[reportPrivateUsage]
            for watch in self._watches:
                watch._valid = False  # pyright: ignore[reportPrivateUsage]
            self._seq += 1  # Odd: a ledger write is open.
            for borrow in self._borrows:
                borrow._valid = False  # pyright: ignore[reportPrivateUsage]
            self._borrows.clear()
            self._token += 1
            self._bytes = 0
            self._rows.clear()
            # Published in place: the entry carries the newest full-retirement
            # token at every instant; the bracketing _seq refuses a prober
            # that overlaps the write.
            self._retired[_EVERY_KEY] = self._token
            self._seq += 1  # Even: the ledger is quiescent again.

    def invalidate(self, keys: Iterable[tuple[str, str]]) -> None:
        """Drop exactly ``keys`` and refuse their admissions issued before now.

        Other rows stay memoized: a HeavenBase row mutation names the entity
        and object it replaces or deletes, and no other main row changes with
        it. Reads of other keys that overlap the mutation remain admissible.
        """
        with self._lock:
            observed: list[tuple[str, str]] | None = (
                [] if self._extensions or self._watches or self._borrows else None
            )
            self._seq += 1  # Odd: a ledger write is open.
            try:
                self._token += 1
                for key in keys:
                    if observed is not None:
                        observed.append(key)
                    if key[0] == "evidence":
                        self._journal_tokens.append(self._token)
                        self._journal_ids.append(key[1])
                    previous = self._rows.pop(key, None)
                    if previous is not None:
                        self._bytes -= previous[1]
                    # Published in place, as retire() publishes its entry, so
                    # the ledger is a valid retirement record at every instant
                    # and _prune ranks entries by token rather than insertion
                    # order. The bracketing _seq, not the order of these
                    # stores, refuses a prober that lands between the first
                    # and the last key of one mutation.
                    self._retired[key] = self._token
            finally:
                if observed is not None:
                    for receipt in self._extensions:
                        receipt._observe(observed)  # pyright: ignore[reportPrivateUsage]
                    for watch in self._watches:
                        watch._observe(observed)  # pyright: ignore[reportPrivateUsage]
                    ended: list[BorrowWatch] = []
                    for borrow in self._borrows:
                        borrow._observe(observed)  # pyright: ignore[reportPrivateUsage]
                        if not borrow._valid:  # pyright: ignore[reportPrivateUsage]
                            ended.append(borrow)
                    for borrow in ended:
                        self._borrows.discard(borrow)
                self._seq += 1  # Even: every key of the mutation is published.
            if len(self._journal_tokens) > EVIDENCE_JOURNAL_CAPACITY:
                half = len(self._journal_tokens) // 2
                # Every entry at the new floor's token is forgotten with it, so
                # an asker at or above the floor still sees every later one.
                floor = self._journal_tokens[half - 1]
                start = bisect_right(self._journal_tokens, floor)
                self._journal_floor = floor
                del self._journal_tokens[:start], self._journal_ids[:start]
            if len(self._retired) > self.capacity:
                self._prune()

    def _prune(self) -> None:
        """Forget the older half of the ledger under a raised floor; called locked.

        The ledger is bounded like the rows. Forgetting the entries whose
        retirement token is at or below the median raises the floor to that
        token: an admission or a borrower's check issued before the floor can
        no longer be answered per key and is refused, later ones keep their
        exact answer. The rank is computed before the write opens; the floor
        raise and the removals then form one ``_seq``-bracketed ledger write,
        and every forgotten entry's token is at most the floor, so a lock-free
        prober never sees a pruned key under an old floor. The sort ranks
        every ledger entry, O(n log n) under the cache lock; it runs only once
        the ledger exceeds ``capacity`` entries and forgets at least half of
        them, so it cannot run on every mutation.
        """
        tokens = sorted(token for key, token in self._retired.items() if key != _EVERY_KEY)
        if not tokens:
            return
        threshold = tokens[(len(tokens) - 1) // 2]
        forgotten = [
            key for key, token in self._retired.items()
            if token <= threshold and key != _EVERY_KEY
        ]
        self._seq += 1  # Odd: a ledger write is open.
        self._floor = max(self._floor, threshold)
        for key in forgotten:
            del self._retired[key]
        self._seq += 1  # Even: the floor covers every forgotten entry.

    def changed_since(self, keys: Iterable[tuple[str, str]], token: int) -> bool:
        """Report whether any of ``keys`` may have been mutated after ``token``.

        ``False`` proves that no in-process row mutation named one of the keys
        after the admission token, so a borrower that read those rows at or
        after that token still holds their current content. ``True`` is also
        the answer whenever the ledger no longer covers the token, a full
        retirement ran, the borrower's token predates the ledger floor, or a
        ledger write was open at, or completed between, this call's reads.
        """
        # Lock-free: the finalize plan chain asks millions of times per plan,
        # and taking the contended cache lock here cost more than the check.
        # Each read below is one atomic attribute or dictionary read under the
        # interpreter lock; what makes their combination sound is the
        # sequence counter, read first and last. Every ledger write holds it
        # odd from its first store to its last, so an even and unchanged value
        # proves that no write was open at either read or completed between
        # them and every read saw one quiescent ledger. Anything else is a
        # possible partial view and is reported as a change.
        before = self._seq
        if before & 1:
            return True
        if token == self._token:
            # No key can have a newer retirement at the current token. Keep
            # the sequence fence: pruning can change the ledger without
            # advancing the token, and an invalidation may still be open.
            return self._seq != before
        ledger = self._retired
        if any(ledger.get(key, 0) > token for key in keys):
            return True
        if token < max(self._floor, ledger.get(_EVERY_KEY, 0)):
            return True
        return self._seq != before


def _value_bytes(value: Any) -> int:
    """Estimate one memoized scan value's resident bytes, without walking deeply.

    Only the shapes this memo actually holds are measured exactly: strings, and
    tuples or lists of them or of small records. Anything else is charged a
    fixed estimate, which bounds the table without a full traversal, and a
    vector of more than 256 items is charged a fixed size per item rather than
    walked. A table whose values are records larger than those fixed sizes
    measures its own entries and stores them with an explicit ``value_bytes``:
    this estimate never sees inside such a record and would not bound it.
    """
    if isinstance(value, str):
        return 49 + len(value)
    if isinstance(value, tuple | list):
        items = cast("tuple[Any, ...] | list[Any]", value)
        if len(items) > 256:
            return 56 + 256 * len(items)
        return 56 + sum(_value_bytes(item) for item in items)
    if isinstance(value, _IMMUTABLE):
        return 28
    return 512


class ScanMemo:
    """Memoize complete scans proven current by a ``RowCache`` retirement ledger.

    An entry records the admission token read before its scan started, every
    ``(kind, object_id)`` row the scan read and the membership keys of the sets
    it enumerated. ``lookup`` serves the value while the cache proves that none
    of those keys was retired after the token, exactly as a borrower proves
    its fixed rows unchanged, and ``store`` refuses a scan whose keys were
    retired while it ran. A served value is exactly what the scan would read
    now; the caller still runs its borrow checks around every use and copies
    a mutable value before handing it out. Nothing here is durable or shared
    across processes. When the entry count or the resident bytes would exceed
    their bounds the least recently used entries are forgotten one at a time:
    the memos of one scope's rows and of another scope's scans share this
    table, and clearing all of it on every overflow made each of them
    recompute what the other had just admitted. Forgetting an entry only turns
    a later lookup into a fresh scan. An entry's bytes are ``_value_bytes``'s
    estimate unless its own producer measured the value and passed
    ``value_bytes``, which a table holding records this module cannot see
    inside must do for the budget to bound anything.
    """

    __slots__ = ("_bytes", "_cache", "_entries", "_lock", "budget", "capacity")

    def __init__(
        self, cache: RowCache, capacity: int = SCAN_MEMO_CAPACITY,
        budget: int = SCAN_MEMO_BYTES,
    ) -> None:
        if capacity < 1 or budget < 1:
            raise ValueError("Scan memo capacity and budget must be positive")
        self._cache = cache
        self._entries: dict[Hashable, tuple[int, tuple[tuple[str, str], ...], Any, int]] = {}
        self._lock = threading.Lock()
        self._bytes = 0
        self.capacity = capacity
        self.budget = budget

    @property
    def resident_bytes(self) -> int:
        """Return the estimated resident size of every memoized value."""
        with self._lock:
            return self._bytes

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def clear(self) -> None:
        """Release all payload references after the owner has retired their row proofs.

        Clearing by itself is cache eviction, not read revocation.
        """
        with self._lock:
            self._entries.clear()
            self._bytes = 0

    def release_stale(self) -> int:
        """Release every entry the ledger no longer proves current; keep the rest.

        An erasure mutates every row whose content it removes, and the mutation
        retires exactly those keys, so an entry that read removed content can
        no longer be served (``lookup``) and dropping it releases the payload.
        An entry whose every key is unretired is exactly what a lookup serves
        now and holds nothing the erasure removed. ``store`` refuses a scan
        whose keys were retired while it ran, so none can return afterwards.

        Returns:
            How many entries were released.
        """
        with self._lock:
            stale = [
                key for key, (token, keys, _value, _size) in self._entries.items()
                if self._cache.token != token and self._cache.changed_since(keys, token)
            ]
            for key in stale:
                self._bytes -= self._entries.pop(key)[3]
        return len(stale)

    @attribute_cost("cache.scan_lookup")
    def lookup(self, key: Hashable) -> Any:
        """Return the memoized scan result for ``key``, or ``MISSING``.

        Lock-free: the entry is one dictionary read, and the proof is the
        ledger's own lock-free ``changed_since`` over the entry's keys, which
        refuses a proof that overlapped a ledger write.
        """
        entry = self._entries.get(key)
        if entry is None:
            cost_count("cache.scan_absent")
            return MISSING
        token, keys, value, _size = entry
        if self._cache.token == token or not self._cache.changed_since(keys, token):
            cost_count("cache.scan_hit")
            # A served scan read its rows earlier; active records still need them.
            note_reads(keys)
            with self._lock:
                if key in self._entries:
                    # Most recently used; the eviction below forgets the other end.
                    self._entries[key] = self._entries.pop(key)
            return value
        cost_count("cache.scan_invalidated")
        with self._lock:
            if self._entries.get(key) is entry:
                self._bytes -= self._entries.pop(key)[3]
        return MISSING

    def store(
        self, key: Hashable, token: int, keys: Iterable[tuple[str, str]], value: Any,
        *, value_bytes: int | None = None,
    ) -> bool:
        """Memoize ``value`` unless one of ``keys`` was retired since ``token``.

        Args:
            key: The scan's identity, including everything its result depends
                on besides storage (its scope, its fixed query parameters).
            token: The cache's admission token read before the scan started.
            keys: Every row the scan read and every membership key of the sets
                it enumerated, including ``MEMBERSHIP_ANY``.
            value: The scan result; immutable, or copied by the caller on use.
            value_bytes: The value's own resident bytes, when its producer
                measured them. ``_value_bytes`` measures only the shapes this
                module holds by identity and charges a fixed size to any other
                record, so a table whose values are records it cannot see
                inside passes its own measurement or its budget bounds
                nothing. Omit it for a value of the measured shapes.

        Returns:
            Whether the value was memoized.
        """
        proof = tuple(keys)
        if self._cache.changed_since(proof, token):
            return False
        measured = _value_bytes(value) if value_bytes is None else value_bytes
        size = measured + 64 * len(proof)
        if size > self.budget:
            return False
        with self._lock:
            # A privacy retirement may happen after the optimistic check and
            # before this lock. Never repopulate a freshly scrubbed table from
            # a pre-retirement scan.
            if self._cache.changed_since(proof, token):
                return False
            previous = self._entries.pop(key, None)
            if previous is not None:
                self._bytes -= previous[3]
            while self._entries and (
                len(self._entries) >= self.capacity or self._bytes + size > self.budget
            ):
                oldest = next(iter(self._entries))
                self._bytes -= self._entries.pop(oldest)[3]
            self._entries[key] = (token, proof, value, size)
            self._bytes += size
        return True
