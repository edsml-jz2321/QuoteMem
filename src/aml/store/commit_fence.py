"""The commit fence's would-be test: which rows committed after a member's pin it would
have been issued (C.11.3 with R8-4's D-34 tightening and the A2 amendment).

A member was nominated at its round's pin with frozen corpus statistics (C.4).
Rows committed later, by an earlier component of the same owner or by anyone
else, are its *delta*. The fence must send the member's component to round 2
when re-nominating the member at the fence pin, with the same frozen
statistics, would issue it a delta row that no issued sibling produced
(C.11.1 #3). These functions decide that conservatively: every delta row such a
re-nomination selects is returned (zero false negatives, C.18 #2); a returned
row need not be selected (a false positive costs one re-review).

Each route of C.4.3 that ranks a pool is tested on its own pool:

- **Exact re-rank** (A2) when the route was recorded whole at the pin: an entity
  route that was ``complete`` or ``empty_complete`` held at most 16 rows and a
  ``complete`` secondary route at most 8, all in the stored top-16 lists with
  their scores. The fence pool is then known exactly (those rows, minus the
  removed ones, plus the delta), and re-ranking it is the brute force itself.
- **Dominance** (D-34 (i), amended by A2): on an entity route larger than its
  cap at the fence pin, a delta row ``h`` is not selected when at least K rows
  of the fence pool precede it on the dense list and, unless ``h``'s BM25 is 0,
  on the BM25 list. Lists are ordered by score, then commit and content, as nomination
  orders them; such a row has a strictly higher RRF in any pool holding both.
- **Counted shift** (D-34 (ii)): on an originally truncated route and on
  corrects, each selected row keeps at least ``1/(60 + r_d + c_d) + 1/(60 + r_b
  + c_b)``, where ``c_L`` counts the rows that may join the pool above it on list
  L; ``h`` is not selected when its optimistic RRF, ranked against the stored
  top-16 lists without the removed rows, stays below the lowest such bound.
- **Floors and counted rows**: scope dense keeps its recorded floor, and a
  delta row that ``scope_dense_top_k`` rows the route's pool certainly holds
  strictly precede (by cosine, then commit and content) cannot be selected.
- **Certain pools** (FENCE-1): a route whose fence pool is known exactly (every
  row's membership is certain, and its pin pool is whole) is re-ranked as
  nomination ranks it; scope dense then tells the later routes exactly what it
  takes, and corrects re-ranks a pin pool of at most ``corrects_top_k`` rows
  exactly. Otherwise a corrects delta row that ``corrects_top_k`` certain pool
  rows dominate (as on an entity route) cannot be selected.

A cutoff is void when one of the route's selected rows may have left the pool:
it was removed, or an earlier route may now select it. Every delta row is then
returned unless an exact or certain-pool proof above excludes it. RRF ranks are
not monotone under pool changes, so a
top-K route whose pool changed at all may select a different set of pin rows:
the pin rows it selected may join later pools, and pin rows it did not select
may leave them. `member_would_be` applies these rules across one member's routes.

A corrects cutoff is also void when a selected row has no frozen scores: a
retired version the cue selected once ``history_targets`` were full was never
issued, so the fence never scored it (C.4.3 #4).

The uniform-m bound that D-34 replaced is deleted (R8-4 recorded, latest only).
"""

from __future__ import annotations

import bisect
from collections.abc import Callable, Iterable, Mapping, Sequence, Set
from dataclasses import dataclass, field
from fractions import Fraction
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final, Literal, Protocol

from aml.core.add_read_view import AddReadView, HistoryEntry, MemoryRecord, VersionRecord
from aml.core.entities import USER, PositionKey, QuoteSpanV1
from aml.core.errors import AMLError
from aml.core.neighbor_snapshot import (
    CorrectsRouteV1,
    EntityRouteV1,
    RankedV1,
    RankRecordV1,
    RevisesRouteV1,
    RouteRecordV1,
    ScopeDenseRouteV1,
    SecondaryRouteV1,
)
from aml.core.readability import Readability
from aml.core.request_order import VersionOrder, version_order
from aml.core.semantic import ScopeKind
from aml.core.statement import token_seq
from aml.ingestion.nomination import (
    NominationPolicyV1,
    NominationV1,
    RrfRanking,
    ScoredRow,
    identity_reach,
    score_rows,
)
from aml.store.indexed import (
    MAX_PAGE_SIZE,
    EntityNameRows,
    EntityRows,
    IndexedQuery,
    IndexedReader,
    ItemRows,
    RevisionRows,
)

if TYPE_CHECKING:
    from aml.store.writer import Writer

_DEFAULT_POLICY: Final = NominationPolicyV1()

type WouldBeRoute = EntityRouteV1 | SecondaryRouteV1 | CorrectsRouteV1 | ScopeDenseRouteV1


class Scored(Protocol):
    """A row's frozen scores for one of the member's queries (C.4.2)."""

    @property
    def order(self) -> VersionOrder:
        """The row's deterministic commit and content tie order."""
        ...

    @property
    def cosine(self) -> float:
        """The dense score: cosine rounded half-even to 6 decimals."""
        ...

    @property
    def bm25(self) -> float:
        """The BM25 score at the pin's statistics, rounded to 6 decimals; 0 has no rank."""
        ...


@dataclass(frozen=True, slots=True)
class PinScoreV1:
    """A row's frozen cosine and BM25 for one query of one member (C.4.2)."""

    cosine: float
    bm25: float
    order: VersionOrder


_NONE: Final[Mapping[str, Scored]] = MappingProxyType({})


# The single-route test -----------------------------------------------------------------------


def _term(rank: int | None, shift: int, k: int) -> Fraction:
    return Fraction(0) if rank is None else Fraction(1, k + rank + shift)


def _dense_before(one_key: str, one: Scored, other_key: str, other: Scored) -> bool:
    """A higher cosine precedes a lower one; commit and content, then the key, break ties.

    The key ends the order, as in nomination's `RrfRanking`: two rows can share every
    commit and content coordinate (a row made readable again without a new version),
    and the order must still be total for a counted shift to be exact.
    """
    return (-one.cosine, one.order, one_key) < (-other.cosine, other.order, other_key)


def _lexical_before(one_key: str, one: Scored, other_key: str, other: Scored) -> bool:
    """A higher positive BM25 precedes a lower one, then commit, content and the key."""
    return one.bm25 > 0 and (other.bm25 == 0 or (
        (-one.bm25, one.order, one_key) < (-other.bm25, other.order, other_key)))


def _upper_rank(
    score: float, order: VersionOrder, key: str, top: Sequence[RankedV1],
    removed: Set[str], pinned: Mapping[str, Scored],
) -> int:
    # Unissued top-list rows have no content coordinate in the snapshot. An equal
    # score then cannot prove precedence: ignoring it keeps this bound optimistic.
    return 1 + sum(
        row.score > score or (row.score == score and row.object_id in pinned
                              and (pinned[row.object_id].order, row.object_id) < (order, key))
        for row in top if row.object_id not in removed
    )


def _upper(
    key: str, score: Scored, ranks: RankRecordV1, removed: Set[str], k: int,
    pinned: Mapping[str, Scored],
) -> Fraction:
    dense = _upper_rank(score.cosine, score.order, key, ranks.top_dense, removed, pinned)
    lexical = (_upper_rank(score.bm25, score.order, key, ranks.top_bm25, removed, pinned)
               if score.bm25 > 0 else None)
    return _term(dense, 0, k) + _term(lexical, 0, k)


def _counted_lower(
    ranks: RankRecordV1, joining: Mapping[str, Scored], pinned: Mapping[str, Scored], k: int,
) -> Fraction:
    bounds: list[Fraction] = []
    for selected in ranks.selected_ranks:
        key, score = selected.object_id, pinned[selected.object_id]
        # Every row that may join the pool above the selected row on a list pushes it
        # down by one there; equal scores follow commit and content order, then the key.
        dense_shift = sum(_dense_before(other, row, key, score)
                          for other, row in joining.items())
        lexical_shift = sum(_lexical_before(other, row, key, score)
                            for other, row in joining.items())
        bounds.append(_term(selected.dense_rank, dense_shift, k)
                      + _term(selected.bm25_rank, lexical_shift, k))
    return min(bounds, default=Fraction(0))


def _dominated(key: str, row: Scored, pool: Mapping[str, Scored], needed: int) -> bool:
    """At least ``needed`` other pool rows have a strictly higher RRF than ``key`` (A2).

    Such a row precedes ``key`` on the dense list and, unless ``key``'s BM25 is
    0 (no BM25 rank, so no BM25 term), on the BM25 list.
    """
    found = 0
    for other, score in pool.items():
        if other != key and _dense_before(other, score, key, row) and (
            row.bm25 == 0 or _lexical_before(other, score, key, row)
        ):
            found += 1
            if found >= needed:
                return True
    return False


def ranked_selection(pool: Mapping[str, Scored], *, whole: int, top: int, k: int) -> list[str]:
    """Select from a pool as one RRF route does, with nomination's `RrfRanking` (C.4.2, C.4.3 #1).

    Nomination and the fence share that one selection, so the exact re-rank of a
    route recorded whole selects what re-nomination at the fence pin would (A2).

    Args:
        pool: Every row of the pool with its frozen scores, keyed by object id.
        whole: The pool is selected whole when it holds at most this many rows.
        top: Otherwise its first ``top`` rows by RRF are selected.
        k: The RRF constant.

    Returns:
        The selected object ids in selection order: RRF descending, then cosine
        descending, then commit and content, then object id; a zero BM25 has no
        BM25 rank.
    """
    rows = [ScoredRow(key, row.cosine, row.bm25, row.order) for key, row in pool.items()]
    return [row.key for row in RrfRanking.of(rows, k).select(whole=whole, top=top)]


def _limits(route: EntityRouteV1 | SecondaryRouteV1, policy: NominationPolicyV1) -> tuple[int, int]:
    if isinstance(route, EntityRouteV1):
        return policy.primary_all_cap, policy.primary_top_k
    return policy.secondary_top_k, policy.secondary_top_k


def _size(route: EntityRouteV1 | SecondaryRouteV1) -> int:
    return route.partition_size if isinstance(route, EntityRouteV1) else route.pool_size


def recorded_whole(route: RouteRecordV1) -> bool:
    """Whether the snapshot holds every row of the route's pin pool with its scores.

    An entity route ``complete`` or ``empty_complete`` and a ``complete``
    secondary route were selected whole; their pools are at most 16 rows, all in
    the stored top-16 dense list.
    """
    if isinstance(route, EntityRouteV1):
        return route.mode in ("complete", "empty_complete")
    return isinstance(route, SecondaryRouteV1) and route.mode == "complete"


def would_be_neighbors(
    route: WouldBeRoute, delta: Mapping[str, Scored], *,
    pinned: Mapping[str, Scored],
    removed: Set[str] = frozenset(),
    joined: Mapping[str, Scored] = _NONE,
    uncertain: Set[str] = frozenset(),
    shrink: int = 0,
    void: bool = False,
    policy: NominationPolicyV1 = _DEFAULT_POLICY,
) -> tuple[str, ...]:
    """Return the delta rows that one route may select at the fence pin (C.11.3, A2).

    Args:
        route: The route as recorded at the member's pin.
        delta: Rows new since the pin that are (or, when in ``uncertain``, may be)
            in the route's pool at the fence pin, scored at the pin's frozen
            statistics for the route's query. Keys are memory ids for heads and
            revision ids for retired versions. Issued siblings' rows stay in;
            the caller exempts them when it applies finding #3.
        pinned: Pin-pool rows with their frozen scores: every row of a route
            recorded whole (`recorded_whole`) and every selected row of a
            truncated entity, secondary or corrects route; a corrects route
            missing one has a void cutoff. Any other pin-pool row still in the
            pool sharpens dominance; a pin row outside the pool must not be given.
        removed: Pin-pool rows that may have left the pool by the fence pin (a
            new version of an Item is a removal plus a delta row). A superset is
            allowed: it only makes the test more conservative.
        joined: Pin rows outside the route's pin pool that are (or, when in
            ``uncertain``, may be) in it at the fence pin: rows an earlier route
            selected at the pin and may no longer select. They compete but are
            never returned.
        uncertain: Keys of ``delta`` and ``joined`` whose membership of the pool
            at the fence pin is not certain (an earlier route may take them).
        shrink: How many pin-pool rows outside ``removed`` may leave the pool to
            an earlier route; it only lowers the smallest possible pool size.
        void: The route's cutoff is invalid; every delta row is returned, or for
            scope dense every row at or above the policy floor.
        policy: The member's frozen nomination policy.

    Returns:
        Sorted delta keys the route may select. False positives are allowed;
        every delta row a re-nomination at the fence pin selects is included.

    Raises:
        ValueError: A route recorded whole lacks one of its rows in ``pinned``,
            or a truncated entity or secondary route lacks a selected row's
            frozen scores.
    """
    if isinstance(route, ScopeDenseRouteV1):
        taken = {row.object_id for row in route.selected}
        floor = policy.scope_dense_floor if void or taken & removed else route.floor
        return tuple(sorted(key for key, row in delta.items() if row.cosine >= floor))
    if isinstance(route, CorrectsRouteV1):
        return _corrects(route, delta, pinned, removed, joined, void, policy)
    if isinstance(route, EntityRouteV1) and route.mode == "empty_incomplete":
        return ()
    if void:
        return tuple(sorted(delta))
    if recorded_whole(route):
        return _recorded(route, delta, pinned, removed, joined, uncertain, policy)
    return _truncated(route, delta, pinned, removed, joined, uncertain, shrink, policy)


def _recorded(
    route: EntityRouteV1 | SecondaryRouteV1, delta: Mapping[str, Scored],
    pinned: Mapping[str, Scored], removed: Set[str], joined: Mapping[str, Scored],
    uncertain: Set[str], policy: NominationPolicyV1,
) -> tuple[str, ...]:
    """Exact re-rank of a route recorded whole, unless an uncertain member prevents it."""
    recorded = sorted(row.object_id for row in route.ranks.top_dense)
    if len(recorded) != _size(route) or not set(recorded) <= set(pinned):
        raise ValueError("A route recorded whole needs every pin row's frozen scores")
    whole, top = _limits(route, policy)
    certain: dict[str, Scored] = {key: pinned[key] for key in recorded if key not in removed}
    certain.update((key, row) for key, row in joined.items() if key not in uncertain)
    certain.update((key, row) for key, row in delta.items() if key not in uncertain)
    if not uncertain & (delta.keys() | joined.keys()):
        chosen = ranked_selection(certain, whole=whole, top=top, k=policy.rrf_k)
        return tuple(sorted(set(chosen) & delta.keys()))
    # The pool may hold more rows than the certain ones: while it may be small, it is
    # selected whole; otherwise only dominance by certain members clears a row.
    if len(certain) <= whole:
        return tuple(sorted(delta))
    return tuple(sorted(key for key, row in delta.items()
                        if not _dominated(key, row, certain, top)))


def _truncated(
    route: EntityRouteV1 | SecondaryRouteV1, delta: Mapping[str, Scored],
    pinned: Mapping[str, Scored], removed: Set[str], joined: Mapping[str, Scored],
    uncertain: Set[str], shrink: int, policy: NominationPolicyV1,
) -> tuple[str, ...]:
    """D-34 on a route truncated at its pin: dominance, then the counted shift."""
    whole, top = _limits(route, policy)
    certain_new = sum(key not in uncertain for key in (*delta, *joined))
    # The fewest rows the pool can hold: every pin row that may leave has left and no
    # uncertain row joined. A pool that may be small is selected whole.
    if _size(route) - len(removed) - shrink + certain_new <= whole:
        return tuple(sorted(delta))
    ranks = route.ranks
    selected = {row.object_id for row in ranks.selected_ranks}
    shifted = not selected & removed
    if shifted and not selected <= set(pinned):
        raise ValueError("D-34 needs frozen scores for every selected row")
    pool: dict[str, Scored] = {key: row for key, row in pinned.items() if key not in removed}
    pool.update((key, row) for key, row in joined.items() if key not in uncertain)
    pool.update((key, row) for key, row in delta.items() if key not in uncertain)
    lower = (_counted_lower(ranks, {**joined, **delta}, pinned, policy.rrf_k) if shifted
             else None)
    result: list[str] = []
    for key, score in sorted(delta.items()):
        if _dominated(key, score, pool, top):
            continue
        if lower is not None and _upper(key, score, ranks, removed, policy.rrf_k,
                                        pinned) < lower:
            continue
        result.append(key)
    return tuple(result)


def _corrects(
    route: CorrectsRouteV1, delta: Mapping[str, Scored], pinned: Mapping[str, Scored],
    removed: Set[str], joined: Mapping[str, Scored], void: bool, policy: NominationPolicyV1,
) -> tuple[str, ...]:
    """D-34 (ii) on the corrects route; dominance stays an entity-route rule."""
    if policy.corrects_top_k == 0:
        return ()
    ranks = route.ranks
    selected = {row.object_id for row in ranks.selected_ranks}
    # Without K original witnesses the minimum selected score cannot bound the future
    # Kth score: even a weak insertion can fill a vacant slot. A selected row without
    # frozen scores (selected past `history_targets`, so never issued) bounds nothing.
    if void or len(selected) < policy.corrects_top_k or selected & removed or (
        not selected <= set(pinned)
    ):
        return tuple(sorted(delta))
    lower = _counted_lower(ranks, {**joined, **delta}, pinned, policy.rrf_k)
    return tuple(sorted(key for key, score in delta.items()
                        if _upper(key, score, ranks, removed, policy.rrf_k, pinned) >= lower))


# One member across its routes ----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FenceRowV1:
    """One head or retired version as the fence sees it for one member.

    Attributes:
        key: The memory id of a head, or the revision id of a retired version.
        anchors: The version's anchors, primary included.
        statement: Its frozen scores against the member's statement query.
        cue: Its frozen scores against the member's ``corrects_text`` cue; None
            when the member has no cue.
        retired: A retired version (C.1), which only the corrects route ranks;
            False for a head.
    """

    key: str
    anchors: frozenset[str]
    statement: PinScoreV1
    cue: PinScoreV1 | None = None
    retired: bool = False


@dataclass(frozen=True, slots=True)
class MemberFenceV1:
    """What the would-be test reads of one member at one fence pin.

    Attributes:
        routes: The member's route records at its pin (C.4.5 ``routes[]``).
        primary: The primary slot's entity at the fence pin: the pin's, or the
            entity minted since the pin for an ``empty_complete`` route's new
            declaration; None for an open slot or a declaration still unminted.
        secondary: The secondary slots' entities at the fence pin: the pin's,
            plus any minted since the pin.
        pinned: Pin heads with their pin versions' scores and anchors: every row
            of a route recorded whole and every selected row of every route
            (with cue scores when the member has a cue), except a retired version
            the corrects route selected but never issued, which voids its cutoff.
            Other pin heads still in a pool sharpen dominance.
        removed: Pin heads and retired versions that changed or left since the
            pin, keyed like rows, with the anchors of their pin versions; a
            superset is allowed.
        heads: P-MEM heads at the fence pin whose version is new since the pin.
        retired: Readable retired versions at the fence pin that are new since
            the pin; only the corrects route reads them.
    """

    routes: tuple[RouteRecordV1, ...]
    primary: str | None
    secondary: frozenset[str]
    pinned: Mapping[str, FenceRowV1]
    removed: Mapping[str, frozenset[str]]
    heads: Mapping[str, FenceRowV1]
    retired: Mapping[str, FenceRowV1] = field(default_factory=dict[str, FenceRowV1])


type _Grab = tuple[Callable[[FenceRowV1], bool], int]
"""An earlier route that may select pin rows it did not select: which rows, how many."""


def _cue_score(row: FenceRowV1) -> Scored:
    if row.cue is None:
        raise ValueError("A corrects route needs every row's cue scores")
    return row.cue


def _cue(rows: Iterable[FenceRowV1]) -> dict[str, Scored]:
    return {row.key: _cue_score(row) for row in rows}


@dataclass
class _Routes:
    """The state one member's fence carries from each route to the later ones."""

    fence: MemberFenceV1
    policy: NominationPolicyV1
    found: set[str] = field(default_factory=set[str])
    # Delta rows an earlier route selects for certain, or may select.
    taken: set[str] = field(default_factory=set[str])
    maybe_taken: set[str] = field(default_factory=set[str])
    # Pin rows an earlier route selected and no longer selects: for certain, or maybe.
    left: dict[str, FenceRowV1] = field(default_factory=dict[str, FenceRowV1])
    maybe_left: set[str] = field(default_factory=set[str])
    # Left rows a later route takes for certain (`held`) or may take (`contested`).
    held: set[str] = field(default_factory=set[str])
    contested: set[str] = field(default_factory=set[str])
    # Earlier routes that may select pin rows they did not select at the pin.
    grabs: list[_Grab] = field(default_factory=list[_Grab])

    def row(self, key: str) -> FenceRowV1:
        found = self.fence.pinned.get(key)
        if found is None:
            raise ValueError("The fence needs every selected pin row's scores and anchors")
        return found

    def grabbed(self, key: str) -> bool:
        """A pin head an earlier route may now select; retired versions only rank in corrects."""
        row = self.row(key)
        return not row.retired and any(matches(row) for matches, _count in self.grabs)

    def leaving(self, keys: Iterable[str]) -> None:
        """The pin rows a route selected may leave its selection for later pools."""
        for key in keys:
            if key not in self.fence.removed and key not in self.left:
                self.left[key] = self.row(key)
                self.maybe_left.add(key)

    def head_delta(self, member: Callable[[FenceRowV1], bool]) -> dict[str, Scored]:
        return {key: row.statement for key, row in self.fence.heads.items()
                if member(row) and key not in self.taken}

    def primary(self, route: EntityRouteV1) -> None:
        entity = self.fence.primary
        if entity is None or route.mode == "empty_incomplete":
            return
        delta = self.head_delta(lambda row: entity in row.anchors)
        gone = {key for key, anchors in self.fence.removed.items() if entity in anchors}
        pinned = {key: row.statement for key, row in self.fence.pinned.items()
                  if not row.retired and entity in row.anchors}
        selected = set(route.ranks.selected)
        if recorded_whole(route):
            recorded = sorted(row.object_id for row in route.ranks.top_dense)
            if not set(recorded) <= set(pinned):
                raise ValueError("A route recorded whole needs every pin row's frozen scores")
            whole, top = _limits(route, self.policy)
            pool: dict[str, Scored] = {key: pinned[key] for key in recorded if key not in gone}
            pool.update(delta)
            chosen = set(ranked_selection(pool, whole=whole, top=top, k=self.policy.rrf_k))
            self.found |= chosen & delta.keys()
            self.taken |= chosen & delta.keys()
            # Every pin row was selected, so none can join; the ones re-ranked out leave.
            self.left.update((key, self.row(key)) for key in selected - gone - chosen)
            return
        flagged = would_be_neighbors(route, delta, pinned=pinned, removed=gone,
                                     policy=self.policy)
        self.found |= set(flagged)
        self.maybe_taken |= set(flagged)
        if delta or gone:
            self.grabs.append((lambda row: entity in row.anchors, self.policy.primary_top_k))
            self.leaving(selected)

    def secondary(self, route: SecondaryRouteV1 | None) -> None:
        entities = self.fence.secondary
        if not entities:
            return
        if route is None:  # every secondary entity was minted after the pin
            route = SecondaryRouteV1(
                route="secondary", slots=[1], entity_ids=sorted(entities), mode="complete",
                pool_size=0, ranks=RankRecordV1(selected=[], top_dense=[], top_bm25=[],
                                                selected_ranks=[], s_k=None),
            )
        primary = self.fence.routes
        primary_selected = {key for record in primary if isinstance(record, EntityRouteV1)
                            for key in record.ranks.selected}
        delta = self.head_delta(lambda row: bool(row.anchors & entities))
        gone = {key for key, anchors in self.fence.removed.items() if anchors & entities}
        # The pin pool excludes what the primary selected; those rows may join as `left`.
        pinned = {key: row.statement for key, row in self.fence.pinned.items()
                  if not row.retired and row.anchors & entities and key not in primary_selected}
        selected = set(route.ranks.selected)
        void = any(self.grabbed(key) for key in selected)
        maybe_gone = {key for key in pinned if self.grabbed(key)}
        joined = {key: row.statement for key, row in self.left.items() if row.anchors & entities}
        # The route may take a row other routes left; later pools hold it only maybe.
        self.contested |= joined.keys()
        uncertain = (self.maybe_taken & delta.keys()) | (self.maybe_left & joined.keys())
        shrink = sum(count for _matches, count in self.grabs)
        flagged = would_be_neighbors(
            route, delta, pinned=pinned, removed=gone | maybe_gone, joined=joined,
            uncertain=uncertain, shrink=shrink, void=void, policy=self.policy,
        )
        self.found |= set(flagged)
        self.maybe_taken |= set(flagged)
        changed = bool(delta or gone or joined or void or maybe_gone)
        if changed:
            if not recorded_whole(route):
                self.grabs.append((lambda row: bool(row.anchors & entities),
                                   self.policy.secondary_top_k))
            self.leaving(selected)

    def scope_dense(self, route: ScopeDenseRouteV1) -> None:
        selected = {row.object_id for row in route.selected}
        void = any(self.grabbed(key) for key in selected)
        delta = {key: row.statement for key, row in self.fence.heads.items()}
        ahead = self.dense_pool(selected, delta)
        flagged = [key for key in would_be_neighbors(
            route, delta, pinned={}, removed=self.fence.removed.keys(), void=void,
            policy=self.policy,
        ) if ahead(delta[key]) < self.policy.scope_dense_top_k]
        self.found |= set(flagged)
        floor = self.policy.scope_dense_floor
        if void or selected & self.fence.removed.keys():
            self.grabs.append((lambda row: row.statement.cosine >= floor,
                               self.policy.scope_dense_top_k))
        chosen = self.dense_exact(route, selected, delta, void=void)
        if chosen is not None:
            # Known exactly: later pools lose what it takes and gain the pin rows it gave up.
            self.taken |= chosen & delta.keys()
            self.held |= chosen & self.left.keys()
            self.left.update((key, self.row(key)) for key in selected - chosen
                             if key not in self.fence.removed)
            return
        self.maybe_taken |= set(flagged)
        self.contested |= {key for key, row in self.left.items() if row.statement.cosine >= floor}
        # New rows and pin rows other routes left may take its slots.
        if flagged or void or any(row.statement.cosine >= route.floor
                                  for row in self.left.values()):
            self.leaving(selected)

    def dense_exact(
        self, route: ScopeDenseRouteV1, selected: Set[str], delta: Mapping[str, Scored], *,
        void: bool,
    ) -> set[str] | None:
        """Scope dense's selection at the fence pin when its pool is known exactly, else None.

        The pool's rows at or above the policy floor are known when no selected
        pin row may have been taken by an earlier route, every delta and left row
        at the floor is certainly in or out of the pool, and the pin rows the route
        did not select cannot return: either it selected every pin row at the
        floor (fewer than ``scope_dense_top_k``), or none of its selected rows left.
        The selection is then nomination's: the first ``scope_dense_top_k`` such
        rows by cosine, then commit and content order. A tie on both at the cutoff
        leaves it unknown.

        Args:
            route: The route as recorded at the member's pin.
            selected: Its selected pin rows.
            delta: Every delta head with its frozen statement scores.
            void: An earlier route may take one of its selected rows.

        Returns:
            The delta and pin rows it selects, or None when the pool is uncertain.
        """
        floor, count = self.policy.scope_dense_floor, self.policy.scope_dense_top_k
        if count == 0:
            return set()
        removed = self.fence.removed.keys()
        if void or (len(route.selected) >= count and selected & removed):
            return None
        pool: dict[str, Scored] = {key: self.row(key).statement for key in selected
                                   if key not in removed}
        for key, row in delta.items():
            if key in self.taken or row.cosine < floor:
                continue
            if key in self.maybe_taken:
                return None
            pool[key] = row
        for key, row in self.left.items():
            if row.statement.cosine < floor:
                continue
            if key in self.maybe_left or key in self.contested:
                return None
            pool[key] = row.statement
        ordered = sorted(((-row.cosine, row.order, key) for key, row in pool.items()
                          if row.cosine >= floor))
        if len(ordered) > count and ordered[count - 1][:2] == ordered[count][:2]:
            return None
        return {key for _cosine, _order, key in ordered[:count]}

    def dense_pool(
        self, selected: Set[str], delta: Mapping[str, Scored],
    ) -> Callable[[Scored], int]:
        """Count the rows the scope-dense pool certainly holds ahead of a delta row (C.11.3).

        Scope dense takes its first ``scope_dense_top_k`` rows at or above the
        policy floor by cosine, then commit and content order, from the heads no
        earlier route selects. Its pool at the fence pin certainly holds: each
        selected pin row that is unchanged and that no earlier route may now take;
        each pin row the primary route's exact re-rank certainly gave up, unless a
        secondary route could take it; and each delta row that no earlier route
        selects or may select. A delta row with ``scope_dense_top_k`` of these
        strictly ahead of it cannot be selected, whatever else the pool holds, so
        only the rows the route could still take are would-be neighbours.

        Args:
            selected: The route's selected pin rows.
            delta: Every delta head with its frozen statement scores.

        Returns:
            A function giving, for a delta row, how many such rows precede it.
        """
        floor = self.policy.scope_dense_floor
        certain = [self.row(key).statement for key in selected
                   if key not in self.fence.removed and not self.grabbed(key)]
        certain += [row.statement for key, row in self.left.items()
                    if key not in self.maybe_left and not row.anchors & self.fence.secondary]
        held = {key: row for key, row in delta.items()
                if key not in self.taken and key not in self.maybe_taken}
        ranks = sorted([(-row.cosine, row.order) for row in certain if row.cosine >= floor]
                       + [(-row.cosine, row.order) for row in held.values()
                          if row.cosine >= floor])

        def ahead(row: Scored) -> int:
            # Strictly ahead only: a row never counts itself, nor a row with equal coordinates.
            return bisect.bisect_left(ranks, (-row.cosine, row.order))

        return ahead

    def corrects(self, route: CorrectsRouteV1) -> None:
        # A retired version the cue selected once `history_targets` were full was never
        # issued, so the fence has no row for it; `_corrects` voids the cutoff (C.4.3 #4).
        selected = [key for key in route.ranks.selected if key in self.fence.pinned]
        heads_open, retired_open, _kept = self.revises_open()
        void = heads_open or retired_open or any(self.grabbed(key) for key in selected)
        delta = _cue([*self.fence.heads.values(), *self.fence.retired.values()])
        pinned = _cue(self.row(key) for key in selected)
        flagged = would_be_neighbors(
            route, delta, pinned=pinned, removed=self.fence.removed.keys(),
            joined=_cue(self.left.values()), void=void, policy=self.policy,
        )
        count = self.policy.corrects_top_k
        pool, exact = self.corrects_pool(route, selected)
        if exact:
            chosen = set(ranked_selection(pool, whole=count, top=count, k=self.policy.rrf_k))
            flagged = tuple(key for key in flagged if key in chosen)
        else:
            flagged = tuple(key for key in flagged
                            if not _dominated(key, delta[key], pool, count))
        self.found |= set(flagged)

    def corrects_pool(
        self, route: CorrectsRouteV1, selected: Sequence[str],
    ) -> tuple[dict[str, Scored], bool]:
        """The rows the corrects pool certainly holds at the fence pin, and whether that is all.

        The pool is every head no earlier route selects (the ``revises`` route
        keeps the live heads of its references) and every readable retired
        version ``revises`` did not issue. It certainly holds: each selected pin
        row that is unchanged and that no earlier route may now take; each pin
        row an earlier route certainly gave up and no later route may take; each
        delta head no earlier route selects or may select; and each delta retired
        version, unless an unavailable ``revises`` reference could name it. Those
        are all its rows when the route selected its whole pin pool (at most
        ``corrects_top_k`` rows, each scored) and no row's membership is uncertain:
        it is then re-ranked exactly. Otherwise a delta row that ``corrects_top_k``
        of these rows dominate (they precede it on the cue's dense list and, unless
        its cue BM25 is 0, on its BM25 list) has a lower RRF than each of them in any
        pool, so it cannot be selected.

        Args:
            route: The route as recorded at the member's pin.
            selected: Its selected pin rows that were issued, so scored.

        Returns:
            The certain rows with their cue scores, and whether the pool is exact.
        """
        removed = self.fence.removed.keys()
        heads_open, retired_open, references = self.revises_open()
        pool: dict[str, Scored] = {}
        certain = not (heads_open or retired_open) and (
            route.pool_size <= self.policy.corrects_top_k
            and len(selected) == len(route.ranks.selected))
        for key in selected:
            if key in removed:
                continue
            unknown = retired_open if self.row(key).retired else heads_open
            if self.grabbed(key) or unknown:
                certain = False
                continue
            pool[key] = _cue_score(self.row(key))
        for key, row in self.left.items():
            if key in self.held or key in references:
                continue
            if heads_open or key in self.maybe_left or key in self.contested:
                certain = False
                continue
            pool[key] = _cue_score(row)
        for key, row in self.fence.heads.items():
            if key in self.taken or key in references:
                continue
            if heads_open or key in self.maybe_taken:
                certain = False
                continue
            pool[key] = _cue_score(row)
        for key, row in self.fence.retired.items():
            if key in references:
                continue
            if retired_open:
                certain = False
                continue
            pool[key] = _cue_score(row)
        return pool, certain

    def revises_open(self) -> tuple[bool, bool, frozenset[str]]:
        """What the ``revises`` route may take at the fence pin (C.4.3 #3).

        Returns:
            Whether it may take a delta head (a referenced head changed, or a
            reference issued none), whether it may take a delta retired version
            (an ``earlier`` reference issued none), and the head or retired keys it keeps.
        """
        removed = self.fence.removed.keys()
        heads_open = retired_open = False
        kept: set[str] = set()
        for record in self.fence.routes:
            if not isinstance(record, RevisesRouteV1):
                continue
            for reference in record.references:
                if reference.revision_id is not None:
                    retired_open = retired_open or reference.issued_revision_id is None
                    if reference.issued_revision_id is not None:
                        kept.add(reference.issued_revision_id)
                elif reference.issued_memory_id is None or reference.issued_memory_id in removed:
                    heads_open = True
                else:
                    kept.add(reference.issued_memory_id)
            # `round2_issue` is recorded here but actually runs after corrects;
            # it never removes a row from the corrects ranking pool.
        return heads_open, retired_open, frozenset(kept)


def member_would_be(
    fence: MemberFenceV1, *, policy: NominationPolicyV1 = _DEFAULT_POLICY,
) -> frozenset[str]:
    """Return every delta row that re-nominating one member at the fence pin may issue.

    The routes run in nomination order: the primary entity route, the
    secondary route, scope dense and corrects (the ``revises`` route issues
    live heads, whose moves are finding #2). A route's pool excludes the rows
    an earlier route selects, so changes travel forward:

    - a primary route recorded whole is re-ranked exactly: the delta rows it
      selects leave the later pools and the pin rows it no longer selects join
      them, both for certain;
    - a truncated route whose pool changed at all may select other pin rows
      (RRF is not monotone): the delta rows it may select and the pin rows it
      selected become possible members of later pools, and a later route whose
      selected pin row it may take has a void cutoff;
    - scope dense whose cutoff is void, or lost a selected row, may take any pin
      head at or above the policy floor;
    - scope dense whose pool is known exactly selects exactly, so corrects
      knows which rows it lost and gained; corrects re-ranks a whole pin pool of
      at most ``corrects_top_k`` rows exactly when every row's membership is
      certain, and otherwise clears the delta rows its certain rows dominate.

    Args:
        fence: The member's routes and rows at the fence pin.
        policy: The member's frozen nomination policy.

    Returns:
        The keys of the delta heads and retired versions that may be issued: a
        superset of what re-nomination with the frozen statistics issues
        through these routes.

    Raises:
        ValueError: A recorded route lacks a pin row's scores or anchors, or a
            corrects route lacks cue scores.
    """
    routes = {route.route: route for route in fence.routes}
    state = _Routes(fence, policy)
    primary = routes.get("entity")
    if isinstance(primary, EntityRouteV1):
        state.primary(primary)
    secondary = routes.get("secondary")
    state.secondary(secondary if isinstance(secondary, SecondaryRouteV1) else None)
    dense = routes.get("scope_dense")
    if isinstance(dense, ScopeDenseRouteV1):
        state.scope_dense(dense)
    corrects = routes.get("corrects")
    if isinstance(corrects, CorrectsRouteV1):
        state.corrects(corrects)
    return frozenset(state.found)


# Finding #4: entities and names committed after the pin (C.11.1 #4, W3-RT-7) ------------------


@dataclass(frozen=True, slots=True)
class IdentityFenceV1:
    """What finding #4 reads of one member: the identity part of its judge input.

    Attributes:
        anchors: The existing entities its slots are bound or nominated to, and
            its judged slots' existing candidates, at the pin.
        mint_keys: Its own slots' new declarations (C.9.6): the entity minted
            for one of these is the member's own (W3-RT-7 (a)).
        candidate_mint_keys: Other declarations its judged slots were offered
            as candidates (C.4.3 #8); once minted they are candidates.
        terms: The tokens of its judged slots' texts: mentions, declared names,
            naming names and the nominees' readable names. A name sharing one
            is an exact-phrase or name-BM25 candidate source (C.4.3 #8).
        types: The casefolded types of its judged slots that have one.
        open_slot: A judged slot is open, so every new entity could fit it.
        judged_slot: At least one slot is judged. Any new participant may become
            its speaker-name candidate, including across entity types.
        complete_types: The casefolded types of its judged slots whose
            ``catalog_complete`` was true at the pin. A new entity of such a type
            that is no candidate turns it false (decision Q2, N-Q2); one that was
            already false stays false, since the new entity is no candidate.
        complete_open: An open judged slot's ``catalog_complete`` was true at the
            pin, so any new entity turns it false.
        siblings: Its issued siblings, whose commits it saw (W3-RT-7 (b)).
    """

    anchors: frozenset[str]
    mint_keys: frozenset[str]
    candidate_mint_keys: frozenset[str]
    terms: frozenset[str]
    types: frozenset[str]
    open_slot: bool
    judged_slot: bool
    complete_types: frozenset[str]
    complete_open: bool
    siblings: frozenset[PositionKey]


@dataclass(frozen=True, slots=True)
class EntityChangeV1:
    """An entity row committed after the pin.

    Attributes:
        entity_id: The entity.
        entity_type: Its type after the change; ``""`` when erased.
        mint_key: Its declaration's key for an extraction-declared entity.
        producers: The positions of every component of this round whose
            commit wrote it; empty for every other writer.
        participant: A participant's entity, which a slot's texts may name as a
            speaker whatever its ``catalog_complete`` (C.4.3 #8).
    """

    entity_id: str
    entity_type: str
    mint_key: str | None
    producers: frozenset[PositionKey] = frozenset()
    participant: bool = False


@dataclass(frozen=True, slots=True)
class NameChangeV1:
    """An ``entity_name`` row committed after the pin, and the positions whose commits wrote it.

    Attributes:
        entity_id: The named entity.
        name: The name.
        producers: The positions of every component of this round whose
            commit wrote it; empty for every other writer.
    """

    entity_id: str
    name: str
    producers: frozenset[PositionKey] = frozenset()


def _issued(producers: frozenset[PositionKey], siblings: frozenset[PositionKey]) -> bool:
    """Whether every contributing producer of a row was issued to the member (W3-RT-7, -18)."""
    return bool(producers) and producers <= siblings


def identity_finding(
    member: IdentityFenceV1, entities: Iterable[EntityChangeV1], names: Iterable[NameChangeV1],
) -> bool:
    """Tell whether an entity or name committed after the pin is finding #4 (C.11.1 #4).

    A row counts when its entity is among the member's anchors or candidates,
    when a new name shares a token with a judged slot's texts, or, for an
    entity row, when it turns a judged slot's ``catalog_complete`` (decision
    Q2) false: its type is the type of a judged slot whose ``catalog_complete``
    was true at the pin, or such a slot is open. A new entity that is none of
    the slot's candidates leaves a false ``catalog_complete`` false and its
    candidates unchanged (FENCE-1); a participant's entity still counts whenever
    a slot is judged, as speaker-name candidates are not restricted by entity type.
    Exempt (W3-RT-7): the entity minted for
    one of the member's own declarations, which its commit binds to anyway
    (C.9.6), and every entity and name row whose contributing producers were
    all issued siblings, which the member saw as those siblings; a producer
    is a whole component, so a multi-position component's row is exempt when
    each of its positions was issued (W3-RT-18). A row with an unissued
    contributor still counts, and so do names written by unissued siblings,
    also on the member's own minted entity.

    Args:
        member: The member's identity input at its pin.
        entities: Entity rows with ``commit_seq`` after the pin.
        names: ``entity_name`` rows with ``commit_seq`` after the pin.

    Returns:
        True when some row is a finding.
    """
    rows = tuple(entities)
    own = {row.entity_id for row in rows
           if row.mint_key is not None and row.mint_key in member.mint_keys}
    named = member.anchors | own | {
        row.entity_id for row in rows
        if row.mint_key is not None and row.mint_key in member.candidate_mint_keys
    }
    for row in rows:
        if row.entity_id in own or _issued(row.producers, member.siblings):
            continue
        kind = row.entity_type.casefold()
        if (row.entity_id in named or (row.participant and member.judged_slot)
                or member.complete_open or (kind and kind in member.complete_types)):
            return True
    for name in names:
        if _issued(name.producers, member.siblings):
            continue
        if name.entity_id in named or not member.terms.isdisjoint(token_seq(name.name)):
            return True
    return False


def identity_fence(nomination: NominationV1, siblings: Iterable[PositionKey]) -> IdentityFenceV1:
    """Collect finding #4's view of one member from its nomination at the pin.

    Args:
        nomination: The member's nomination at its round's pin.
        siblings: The positions of its issued siblings.

    Returns:
        The member's anchors, own and candidate declarations, slot-text terms,
        judged types and open slots, and those whose ``catalog_complete`` was true.
    """
    reach = identity_reach(nomination.slots, nomination.member.naming)
    return IdentityFenceV1(
        anchors=reach.anchors, mint_keys=reach.mint_keys,
        candidate_mint_keys=reach.candidate_mint_keys, terms=reach.terms,
        types=reach.types, open_slot=reach.open_slot, judged_slot=reach.judged_slot,
        complete_types=reach.complete_types,
        complete_open=reach.complete_open, siblings=frozenset(siblings),
    )


# The fence of one component (C.11.1, C.11.2) ------------------------------------------------------

FenceCause = Literal["fence_1", "fence_2", "fence_3", "fence_4", "fence_5", "fence_6"]
"""Finding #n of C.11.1 that sends a component to round 2."""
CAUSE_ORDER: Final[tuple[FenceCause, ...]] = (
    "fence_1", "fence_2", "fence_4", "fence_5", "fence_6", "fence_3",
)
"""store-Q3: a moved component takes the first of these found on any member."""


def component_cause(causes: Iterable[FenceCause]) -> FenceCause | None:
    """The single cause a moved component gives every member (store-Q3), or None."""
    found = set(causes)
    return next((cause for cause in CAUSE_ORDER if cause in found), None)


@dataclass(frozen=True, slots=True)
class FenceMemberV1:
    """One member of a component as its fence reads it (C.11.1).

    Attributes:
        position_key: The member.
        nomination: Its nomination at its round's pin.
        siblings: Its issued siblings.
        relations: Its judged relation to each issued head, by memory id; an
            absent head was judged ``independent``.
        write_heads: The heads its write targets, ends or contradicts
            (C ∪ D ∪ S ∪ X, ``WriteV1.fence``); empty for a member that writes nothing.
        corrects: The retired versions its ``annotate`` corrects.
        groups: The groups its write stores, each as its spans.
        final: Its final primary and statement digest, for the tombstone guard;
            None when it writes nothing.
    """

    position_key: PositionKey
    nomination: NominationV1
    siblings: frozenset[PositionKey]
    relations: Mapping[str, str] = field(default_factory=dict[str, str])
    write_heads: tuple[str, ...] = ()
    corrects: tuple[str, ...] = ()
    groups: tuple[tuple[QuoteSpanV1, ...], ...] = ()
    final: FinalV1 | None = None


@dataclass(frozen=True, slots=True)
class FinalV1:
    """A writing member's final primary and statement digest (C.11.1 #6).

    The primary is an existing entity, or the declaration or participant the
    commit gets or creates; exactly one of the three is set.
    """

    statement_digest: str
    entity_id: str | None = None
    mint_key: str | None = None
    participant_key: str | None = None


@dataclass
class MemberFindingsV1:
    """What the fence found for one member (C.11.1).

    Attributes:
        position_key: The member.
        causes: The findings that move its component (TC).
        rebind: Moved heads the member keeps, by memory id, with their current
            head revision: digest-preserving moves of still-active heads, and
            change relations mapped onto a sibling's result (#2's exemptions).
        discard: Issued heads whose relation is discarded: an issued sibling
            ended them, or changed the statement the member's ``same`` restated.
        tombstoned: The tombstone guard refuses it (#6): ``denied_by_forget``.
        unusable: Indexes of its write's groups with a span that is no longer
            usable (P-SPAN); every group unusable is ``evidence_invalidated``.
    """

    position_key: PositionKey
    causes: set[FenceCause] = field(default_factory=set[FenceCause])
    rebind: dict[str, str] = field(default_factory=dict[str, str])
    discard: set[str] = field(default_factory=set[str])
    tombstoned: bool = False
    unusable: frozenset[int] = frozenset()


@dataclass(frozen=True, slots=True)
class FencePinV1:
    """The fence pin s1 of a component's pre-lock read, which its in-lock recheck compares."""

    commit_seq: int
    erase_epoch: int
    consent_epoch: int
    tombstone_seq: int


@dataclass(frozen=True, slots=True)
class MemberWouldBeV1:
    """The would-be test of one member as `ComponentFence.settle` ran it (C.11.3; FENCE-1).

    The operands are the objects the test read, kept so a recording can rerun
    the rule and compare others on the same event; nothing is read again.

    Attributes:
        fence: The test's operands at the fence pin.
        policy: The member's frozen nomination policy.
        lost: A pin head left the pool without a committed change of its Item,
            so every delta row was returned without running the test.
        flagged: The delta rows the test returned.
        issued: The issued heads (by memory id) and retired versions (by
            revision id), never would-be neighbours (finding #2's).
        items: Each delta retired version's Item, whose producers exempt it.
        siblings: The member's issued siblings, which exempt what they produced.
    """

    fence: MemberFenceV1
    policy: NominationPolicyV1
    lost: bool
    flagged: frozenset[str]
    issued: frozenset[str]
    items: Mapping[str, str]
    siblings: frozenset[PositionKey]


@dataclass
class _MemberRead:
    findings: MemberFindingsV1
    primary: str | None
    secondary: frozenset[str]
    pinned: dict[str, FenceRowV1]
    heads: dict[str, FenceRowV1]
    retired: dict[str, FenceRowV1]
    retired_items: dict[str, str]
    gone: dict[str, frozenset[str]]


def _producers(produced: Mapping[str, frozenset[PositionKey]], key: str) -> frozenset[PositionKey]:
    return produced.get(key, frozenset())


class ComponentFence:
    """The pre-lock fence of one component at its pin s1 (C.11.2 #1).

    `read` runs inside the owner's read callback on the view at s1 and keeps
    only content-free results; `settle` then reads the Writer's rows changed
    since the round's pin and runs the would-be test. A member's findings
    never read a provider. ``produced`` names, by memory, entity and
    ``entity_name`` id, the positions whose commits in this round changed the
    row; a row changed by a commit this process did not make has no producer,
    so it is never exempt. ``would_be`` keeps each member's test as it ran.
    """

    def __init__(
        self, members: Sequence[FenceMemberV1], *,
        produced: Mapping[str, frozenset[PositionKey]],
    ) -> None:
        """Bind the members and the round's producers."""
        if not members:
            raise ValueError("A fence reads at least one member")
        pins = {member.nomination.pin for member in members}
        if len(pins) != 1:
            raise AMLError("binding_mismatch", "A component's members share their round's pin")
        self.members = tuple(members)
        self.pin = members[0].nomination.pin
        self.produced = produced
        self.s1: FencePinV1 | None = None
        self.reads: dict[PositionKey, _MemberRead] = {}
        self.would_be: dict[PositionKey, MemberWouldBeV1] = {}
        self.unchanged = 0
        self._unchanged: frozenset[str] = frozenset()

    # Inside the read callback ------------------------------------------------------------------

    def read(self, view: AddReadView) -> None:
        """Collect every finding the view at s1 decides, and the delta rows' frozen scores."""
        pin = view.pin
        self.s1 = FencePinV1(pin.commit_seq, pin.erase_epoch, pin.consent_epoch,
                             pin.tombstone_seq)
        if (pin.erase_epoch, pin.consent_epoch) == (self.pin.erase_epoch,
                                                    self.pin.consent_epoch) and (
            pin.tombstone_seq != self.pin.tombstone_seq
        ):
            raise AMLError("binding_mismatch", "tombstone_seq changed under a valid lease")
        heads = view.heads()
        unchanged = frozenset(record.memory_id for record in heads
                              if record.commit_seq <= self.pin.commit_seq)
        self.unchanged = len(unchanged)
        self._unchanged = unchanged
        pinned_heads = self.members[0].nomination.readable_heads
        delta = [record for record in heads if record.commit_seq > self.pin.commit_seq
                 or record.memory_id not in pinned_heads]
        readable = view.retired()
        retired = [entry for entry in readable if self._changed(view, entry.version.memory_id)]
        present = unchanged | {entry.version.revision_id for entry in readable}
        changes = self._identity_changes(view)
        read = Readability(view.scope_kind, view.scope_id, view.now, view.evidence)
        for member in self.members:
            self.reads[member.position_key] = self._member(
                view, member, delta, retired, changes, read, present,
            )

    def _changed(self, view: AddReadView, memory_id: str) -> bool:
        record = view.memory(memory_id)
        return memory_id in self.produced or record is None or (
            record.commit_seq > self.pin.commit_seq
        )

    def _identity_changes(
        self, view: AddReadView,
    ) -> tuple[list[EntityChangeV1], list[NameChangeV1]]:
        entities: list[EntityChangeV1] = []
        names: list[NameChangeV1] = []
        user = view.entity(USER)
        for record in ((user,) if user is not None else ()) + view.entities():
            if record.commit_seq > self.pin.commit_seq:
                entities.append(EntityChangeV1(
                    record.entity_id, record.entity_type, record.mint_key,
                    _producers(self.produced, record.entity_id),
                    participant=record.participant_key is not None,
                ))
            names.extend(
                NameChangeV1(name.entity_id, name.name,
                             _producers(self.produced, name.object_id))
                for name in record.names if name.commit_seq > self.pin.commit_seq
            )
        return entities, names

    def _entity(self, view: AddReadView, entity_id: str | None, mint_key: str | None,
                participant_key: str | None) -> str | None:
        if entity_id is not None:
            return entity_id
        record = (view.minted(mint_key) if mint_key is not None
                  else view.participant(participant_key) if participant_key is not None
                  else None)
        return None if record is None else record.entity_id

    def _member(
        self, view: AddReadView, member: FenceMemberV1, delta: Sequence[MemoryRecord],
        retired: Sequence[HistoryEntry], changes: tuple[list[EntityChangeV1], list[NameChangeV1]],
        read: Readability, present: frozenset[str],
    ) -> _MemberRead:
        nomination = member.nomination
        found = MemberFindingsV1(member.position_key)
        pin = view.pin
        if (pin.erase_epoch, pin.consent_epoch) != (self.pin.erase_epoch,
                                                    self.pin.consent_epoch):
            found.causes.add("fence_1")
        self._moved_heads(view, member, found)
        for memory_id in member.write_heads:
            if view.memory(memory_id) is None:
                found.causes.add("fence_5")
        if not set(member.corrects) <= present:
            found.causes.add("fence_5")
        self._tombstones(view, member, found)
        identity = identity_fence(nomination, member.siblings)
        # A declaration whose minted entity was erased is never minted again:
        # its erased row keeps the unique key (A.3.6, C.9.6). Nomination opens
        # such a slot; one nominated before the erasure is rebuilt next round.
        if identity_finding(identity, *changes) or any(
            view.spent(key) for key in identity.mint_keys | identity.candidate_mint_keys
        ):
            found.causes.add("fence_4")
        found.unusable = frozenset(index for index, group in enumerate(member.groups)
                                   if not all(read.p_span(span) for span in group))
        slots = [self._entity(view, item.slot.entity_id, item.slot.mint_key,
                              item.slot.participant_key) for item in nomination.slots]
        return _MemberRead(
            findings=found, primary=slots[0] if slots else None,
            secondary=frozenset(entity for entity in slots[1:] if entity is not None),
            pinned=self._pinned(nomination),
            heads=_scored(nomination, [record.head for record in delta]),
            retired=(_scored(nomination, [entry.version for entry in retired], retired=True)
                     if nomination.member.cue is not None else {}),
            retired_items={entry.version.revision_id: entry.version.memory_id
                           for entry in retired},
            gone=_corrects_gone(nomination, present),
        )

    def _moved_heads(self, view: AddReadView, member: FenceMemberV1,
                     found: MemberFindingsV1) -> None:
        """Finding #2 and its exemptions over every issued head (C.11.1 #2)."""
        for target in member.nomination.targets:
            if target.kind != "head" or target.revision_id is None:
                continue
            record = view.memory(target.object_id)
            producers = _producers(self.produced, target.object_id)
            if record is None:
                moved = bool(producers)
            else:
                moved = (record.current_revision_id, record.lifecycle_state) != (
                    target.revision_id, "active",
                )
            if not moved:
                continue
            active = record is not None and record.lifecycle_state == "active"
            if active and record is not None and (
                record.head.snapshot.statement_digest == target.statement_digest
            ):
                found.rebind[target.object_id] = record.current_revision_id
                continue
            if not producers or not producers <= member.siblings:
                found.causes.add("fence_2")
                continue
            relation = member.relations.get(target.object_id, "independent")
            if relation in ("independent", "unclear"):
                continue
            if not active or relation == "same" or record is None:
                found.discard.add(target.object_id)
            else:
                found.rebind[target.object_id] = record.current_revision_id

    def _tombstones(self, view: AddReadView, member: FenceMemberV1,
                    found: MemberFindingsV1) -> None:
        """Finding #6: the tombstone guard (C.11.1 #6)."""
        final = member.final
        if final is None:
            return
        digest = final.statement_digest
        entity_id = self._entity(view, final.entity_id, final.mint_key, final.participant_key)
        if entity_id is not None and view.tombstone(entity_id, digest) is not None:
            found.tombstoned = True
            return
        issued = {item.record.object_id for item in member.nomination.tombstones}
        if any(record.statement_digest == digest and record.entity_id != entity_id
               and record.object_id not in issued for record in view.tombstones()):
            found.causes.add("fence_6")

    @staticmethod
    def _pinned(nomination: NominationV1) -> dict[str, FenceRowV1]:
        """The pin rows the would-be test needs: every issued head and retired version."""
        heads = [target.version for target in nomination.targets
                 if target.kind == "head" and target.version is not None]
        retired = [target.version for target in nomination.targets
                   if target.kind == "retired" and target.version is not None]
        return {**_scored(nomination, heads), **_scored(nomination, retired, retired=True)}

    # After the read callback ---------------------------------------------------------------------

    def settle(
        self, writer: Writer, scope_kind: ScopeKind, scope_id: str,
    ) -> list[MemberFindingsV1]:
        """Read the Items changed since the round's pin and run the would-be test (C.11.3).

        An issued head or retired version is never a would-be neighbour: a
        moved issued head is finding #2's (C.11.1 #2), with its exemptions.
        A pin head that left the pool without a committed change of its Item
        (its support became unusable) cannot be named from the rows; when the
        pin's readable-head identity set shows one, every delta row is returned as a would-be
        neighbour.

        Args:
            writer: The single Writer, outside its mutex.
            scope_kind: The component's scope kind.
            scope_id: The component's scope.

        Returns:
            Each member's findings, in member order.
        """
        if self.s1 is None:
            raise AMLError("binding_mismatch", "A fence settles after its read")
        removed = _pin_removals(writer, scope_kind, scope_id, after=self.pin.commit_seq,
                                upto=self.s1.commit_seq)
        return self.settle_changes(removed)

    def settle_changes(
        self, removed: Mapping[str, frozenset[str]],
    ) -> list[MemberFindingsV1]:
        """Settle a completed fence read with the pin's changed-Item inventory.

        Args:
            removed: Items changed between the nomination pin and ``s1``, mapped
                to their last pin version's anchors. Items created after the pin
                are excluded. The Writer and lab supply the same inventory.

        Returns:
            Findings in member order, with the would-be tests retained.

        Raises:
            AMLError: ``binding_mismatch`` if no read preceded settlement.
        """
        if self.s1 is None:
            raise AMLError("binding_mismatch", "A fence settles after its read")
        results: list[MemberFindingsV1] = []
        for member in self.members:
            state = self.reads[member.position_key]
            nomination = member.nomination
            # A source can become indexed without changing its Item. Its restored
            # head is a delta above, and must not hide another head's source loss
            # merely by keeping the total readable-head count unchanged.
            lost = bool(nomination.readable_heads - self._unchanged - removed.keys())
            fence = MemberFenceV1(
                routes=nomination.routes, primary=state.primary, secondary=state.secondary,
                pinned=state.pinned, removed={**state.gone, **removed}, heads=state.heads,
                retired=state.retired,
            )
            flagged = (set(state.heads) | set(state.retired) if lost
                       else member_would_be(fence, policy=nomination.policy))
            issued = {target.object_id if target.kind == "head" else target.revision_id or ""
                      for target in nomination.targets if target.kind != "sibling"}
            for key in sorted(set(flagged) - issued):
                producers = _producers(self.produced, state.retired_items.get(key, key))
                if not producers or not producers <= member.siblings:
                    state.findings.causes.add("fence_3")
                    break
            self.would_be[member.position_key] = MemberWouldBeV1(
                fence=fence, policy=nomination.policy, lost=lost, flagged=frozenset(flagged),
                issued=frozenset(issued), items=state.retired_items, siblings=member.siblings,
            )
            results.append(state.findings)
        return results


def _scored(
    nomination: NominationV1, versions: Sequence[VersionRecord], *, retired: bool = False,
) -> dict[str, FenceRowV1]:
    """Fence rows of versions scored with the member's frozen statistics (C.4.2, I9b)."""
    if not versions:
        return {}
    member, policy = nomination.member, nomination.policy
    found = score_rows(member, versions, nomination.lexical, policy)
    cues = (score_rows(member, versions, nomination.lexical, policy, query="cue")
            if member.cue is not None else None)
    rows: dict[str, FenceRowV1] = {}
    for index, version in enumerate(versions):
        key = version.revision_id if retired else version.memory_id
        cue = (None if cues is None else
               PinScoreV1(cues[index].cosine, cues[index].bm25, version_order(version)))
        rows[key] = FenceRowV1(
            key=key, anchors=frozenset(version.anchors),
            statement=PinScoreV1(found[index].cosine, found[index].bm25, version_order(version)),
            cue=cue,
            retired=retired,
        )
    return rows


def _pin_removals(
    writer: Writer, scope_kind: ScopeKind, scope_id: str, *, after: int, upto: int,
) -> dict[str, frozenset[str]]:
    """Pin heads whose Item changed after the pin, with their pin versions' anchors.

    An Item changed after the pin that has a version at or before the pin was a
    pin head: under a valid lease only this owner's commits change Items, and
    they change issued heads only. Rows changed after ``upto`` (s1) are left to
    the in-lock recheck.
    """
    reader = writer.indexed_reader()
    query = ItemRows(scope_kind=scope_kind, scope_id=scope_id, after_commit_seq=after)
    removed: dict[str, frozenset[str]] = {}
    last: str | None = None
    while True:
        page = reader.page(query, after=last, limit=MAX_PAGE_SIZE)
        for item in page:
            if item["commit_seq"] > upto:
                continue
            versions = [row for row in _pages(reader, RevisionRows(memory_id=item["object_id"]))
                        if row["commit_seq"] <= after]
            if not versions:
                continue  # created after the pin
            snapshot = max(versions, key=lambda row: row["version"])["snapshot"]
            removed[item["object_id"]] = frozenset(
                [snapshot["entity_id"], *snapshot["about_entity_ids"]],
            ) - {""}
        if len(page) < MAX_PAGE_SIZE:
            return removed
        last = page[-1]["object_id"]


def _pages(reader: IndexedReader, query: IndexedQuery) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    last: str | None = None
    while True:
        page = reader.page(query, after=last, limit=MAX_PAGE_SIZE)
        rows.extend(page)
        if len(page) < MAX_PAGE_SIZE:
            return rows
        last = page[-1]["object_id"]


def recheck(
    writer: Writer, scope_kind: ScopeKind, scope_id: str, s1: FencePinV1,
    members: Sequence[FenceMemberV1],
) -> FenceCause | None:
    """Re-run C.11.1 under the Writer mutex over the rows committed after s1 (C.11.2 #3).

    No provider is called. Under a valid lease no Item of the scope changes
    between the pre-lock read and the lock, so any such row is a moved head;
    entity and name rows run finding #4 again; changed epochs are finding #1.

    Args:
        writer: The single Writer, under its mutex.
        scope_kind: The component's scope kind.
        scope_id: The component's scope.
        s1: The pre-lock pin.
        members: The component's members.

    Returns:
        The first new cause in store-Q3 order, or None.

    Raises:
        AMLError: ``binding_mismatch`` for a ``tombstone_seq`` changed under a
            valid lease without an ``erase_epoch`` change.
    """
    erase = writer.counter(f"erase_epoch:{scope_id}")
    consent = writer.counter(f"consent_epoch:{scope_id}")
    tombstones = writer.counter(f"tombstone_seq:{scope_id}")
    if (erase, consent) != (s1.erase_epoch, s1.consent_epoch):
        return "fence_1"
    if tombstones != s1.tombstone_seq:
        raise AMLError("binding_mismatch", "tombstone_seq changed under a valid lease")
    reader = writer.indexed_reader()
    causes: set[FenceCause] = set()
    if reader.page(ItemRows(scope_kind=scope_kind, scope_id=scope_id,
                            after_commit_seq=s1.commit_seq), limit=1):
        causes.add("fence_2")
    entities = [EntityChangeV1(row["object_id"], row["entity_type"], row["mint_key"],
                               participant=row.get("participant_key") is not None)
                for row in _pages(reader, EntityRows(scope_kind=scope_kind, scope_id=scope_id,
                                                     after_commit_seq=s1.commit_seq))]
    names = [NameChangeV1(row["entity_id"], row["name"])
             for row in _pages(reader, EntityNameRows(scope_kind=scope_kind, scope_id=scope_id,
                                                      after_commit_seq=s1.commit_seq))]
    if (entities or names) and any(
        identity_finding(identity_fence(member.nomination, member.siblings), entities, names)
        for member in members
    ):
        causes.add("fence_4")
    return component_cause(causes)


def _corrects_gone(nomination: NominationV1, present: frozenset[str]) -> dict[str, frozenset[str]]:
    """The corrects route's recorded rows that are no longer readable and unchanged at s1.

    A recorded row is a head (by memory id) or a retired version (by revision
    id); one absent from ``present`` changed or left its pool after the pin.
    """
    found: dict[str, frozenset[str]] = {}
    for route in nomination.routes:
        if isinstance(route, CorrectsRouteV1):
            ranks = route.ranks
            keys = {row.object_id for row in (*ranks.top_dense, *ranks.top_bm25)}
            found.update((key, frozenset()) for key in keys | set(ranks.selected)
                         if key not in present)
    return found
