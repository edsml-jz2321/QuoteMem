"""Deterministic nomination of a member's neighbours at one pin (C.4; I9, I9b).

Each member of an owner's round is reviewed together with the stored memories,
same-Add siblings, retired versions, forgotten statements and entity candidates
it could relate to. `nominate` picks them from one `AddReadView` pin, route by
route (C.4.3); a target's recorded route is the first route that selects it:

1. entity: all of the primary slot's partition P(E) when it has at most 16
   heads, else its top 8 by RRF; then the top 8 by RRF over the union of the
   secondary slots' partitions;
2. scope dense: the top 4 other heads at cosine 0.80 or more;
3. revises: the live head each referenced memory reached, or the retired
   version an ``earlier`` reference names; in round 2 also each target that
   resolution asked to issue (``Round2RequestV1.issue``, L-Q3), a retired one
   to a history correction only;
4. corrects: the top 4 other heads and retired versions by RRF of the
   correction cue;
5. history: retired targets and each issued head's two latest history entries,
   only for a member that corrects history (PD:P8);
6. siblings: the round's earlier members sharing an anchor (all up to 4, else
   the top 4 by RRF), the top 2 others at cosine 0.80 or more, every earlier
   member the entity, secondary, scope-dense and corrects routes would select
   were each earlier member a head committed after the pin, which is what the
   commit fence re-ranks once they commit (C.11.3), and every earlier member
   whose commit would write an entity or name the fence's finding #4 counts
   for this member (C.11.1 #4);
7. tombstones: same-digest tombstones of other entities, then same-entity and
   scope-wide near matches, 8 in all;
8. entity candidates for every judged slot and whether the candidate list is
   complete (decision Q2); every existing entity the input names shows its most
   recent readable heads as facts, at most three and newest first, the same for
   every member at the pin (L-Q1);
9. sources: the cited messages and up to four messages before the earliest.

Scoring follows C.4.2: float64 cosine over L2-normalised vectors, rounded
half-even to 6 decimals; BM25 (k1 1.2, b 0.75) of the member's statement and
keywords over the scope's P-MEM heads at the pin; and RRF
``1/(60 + r_dense) + 1/(60 + r_bm25)``, compared as exact fractions, with a
zero BM25 ranked at infinity. Every order ends in an id tie-break and nothing
reads a clock or a random source, so one pin, policy and member always give one
nomination (C.17 C3).

`neighbor_snapshot` records a nomination as the content-free
`NeighborSnapshotV1` of its judge child run (C.4.5): ids, ranks, scores, keyed
term digests and the member's judge handles (C.4.4), and for a re-entered
member the code it was selected with and its wake key (C.3), never a statement
or a name. For the commit fence's would-be test (C.11.3), `score_rows` scores
rows the way nomination did, and `RrfRanking` is the one RRF selection that
nomination's ranked routes and the fence's re-rank share.
"""

from __future__ import annotations

import base64
import functools
import hashlib
import unicodedata
from collections import Counter
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from fractions import Fraction
from operator import attrgetter
from typing import Annotated, Final, Literal, Self

import numpy as np
from pydantic import ConfigDict, Field, model_validator

from aml.core.add_read_view import (
    AddReadView,
    EndedBy,
    EntityRecord,
    HistoryEntry,
    MemoryRecord,
    MessageKey,
    MessageRecord,
    ReadPin,
    Sigma,
    TombstoneRecord,
    VersionRecord,
)
from aml.core.entities import (
    USER,
    EntityChoiceV1,
    ExistingChoiceV1,
    MemoryKind,
    MemorySnapshotV2,
    NameKind,
    NewEntityChoiceV1,
    ParticipantChoiceV1,
    PositionKey,
    QuoteSpanV1,
    ReasonCode,
    UserChoiceV1,
)
from aml.core.errors import AMLError
from aml.core.identity import canonical_json
from aml.core.naming_syntax import bare_naming
from aml.core.neighbor_snapshot import (
    FENCE_LIST,
    Binding,
    CandidateSource,
    CorrectsRouteV1,
    EntityRouteV1,
    LexicalRecordV1,
    NeighborSnapshotV1,
    NominationDependenciesV1,
    NominationFlagsV1,
    PartitionMode,
    PositionV1,
    RankedV1,
    RankRecordV1,
    RevisesReferenceV1,
    RevisesRouteV1,
    Round2IssueV1,
    RouteRecordV1,
    ScopeDenseRouteV1,
    SecondaryRouteV1,
    SelectedRankV1,
    SiblingSelectionV1,
    SiblingsRouteV1,
    SnapshotCandidateV1,
    SnapshotEntityFactsV1,
    SnapshotHistoryV1,
    SnapshotPinV1,
    SnapshotQuoteV1,
    SnapshotSlotV1,
    SnapshotSourceV1,
    SnapshotTargetV1,
    SnapshotTombstoneV1,
    TargetKind,
    TargetRoute,
    TombstoneFloorsV1,
    TombstoneRoute,
    TombstoneSelectionV1,
    TombstonesRouteV1,
)
from aml.core.recall_corpora import (
    EntityRetrieval,
    SparseDocument,
    SparseStatistics,
    query_terms,
    sparse_terms,
    statement_document,
)
from aml.core.request_order import VersionOrder, entity_order, tombstone_order, version_order
from aml.core.resolution import IssueV1
from aml.core.review_rounds import RoundNo
from aml.core.semantic import SourceTimeExpressionV1, TimeGranularity, validate_support_groups
from aml.core.sensitivity import SensitivityClass
from aml.core.snapshot_memo import snapshot_rows, snapshot_unit
from aml.core.statement import (
    MAX_ANCHORS,
    MAX_ENTITY_CANDIDATES,
    MAX_GROUP_SPANS,
    MAX_JUDGE_HISTORY,
    MAX_JUDGE_TARGETS,
    MAX_JUDGE_TOMBSTONES,
    MAX_MEMORY_SPANS,
    MAX_REVISES,
    MAX_SUPPORT_GROUPS,
    mint_key,
    name_norm,
    named_participants,
    statement_digest,
    token_seq,
)
from aml.core.vector_score import UnitRows, UnitVector, cosine, unit_vector
from aml.store.metadata import StrictModel

_Floor = Annotated[float, Field(ge=-1.0, le=1.0)]

HANDLE_PURPOSE: Final = "judge-handle-v1"
TERM_PURPOSE: Final = "bm25-term-v1"
HANDLE_CHARS: Final = 10
LONG_HANDLE_CHARS: Final = 16

ContentBinder = Callable[[str, object], str]
"""The workspace's keyed content binder: ``bind(purpose, value)`` returns the unpadded
base64url HMAC-SHA256 of the canonical value, as ``Keyring.digest`` does."""

HandleKind = Literal[
    "memory", "history", "sibling", "forgotten", "entity", "new_entity", "quote", "message",
]
HandleKey = tuple[HandleKind, str, str | None]
"""A judge handle's identity ``(kind, object_id, version_id)`` (C.4.4)."""

_PREFIX: Final[Mapping[HandleKind, str]] = {
    "memory": "M", "history": "H", "sibling": "S", "forgotten": "T", "entity": "E",
    "new_entity": "N", "quote": "C", "message": "Q",
}


def handle_key(kind: HandleKind, object_id: str, version_id: str | None = None) -> HandleKey:
    """Return the identity a judge handle is derived from (C.4.4).

    Args:
        kind (HandleKind): The object kind; it fixes the handle's prefix letter.
            Supported values:
            - ``memory`` (M): a current head; ``version_id`` is its revision.
            - ``history`` (H): a history entry or retired version; ``version_id``
              is the version.
            - ``sibling`` (S): an earlier member; ``object_id`` is
              ``owner:unit:position``.
            - ``forgotten`` (T): a tombstone.
            - ``entity`` (E): an existing entity, the user included.
            - ``new_entity`` (N): a new declaration; ``object_id`` is its mint key.
            - ``quote`` (C): a member span; ``version_id`` is ``start-end``.
            - ``message`` (Q): a message by its first part; ``version_id`` is
              ``cited`` or ``context``.
        object_id (str): The object.
        version_id (str | None): The version where the kind has one.

    Returns:
        HandleKey: ``(kind, object_id, version_id)``.
    """
    return (kind, object_id, version_id)


# Policy (C.4.1) ------------------------------------------------------------------------------


class NominationPolicyV1(StrictModel):
    """The frozen nomination parameters (C.4.1); G1a calibrates them (open item O7).

    The defaults are the contract's values. A variant may change any of them
    within the judge wire's bounds (A0.7): at most 64 targets, 80 history
    entries, 8 tombstones and 6 entity candidates besides the nominee per slot.

    Attributes:
        primary_all_cap (int): A primary partition this small is issued whole.
        primary_top_k (int): Heads issued from a larger primary partition.
        secondary_top_k (int): Heads issued from the secondary slots' partitions.
        scope_dense_top_k (int): Heads issued by the scope-wide dense route.
        scope_dense_floor (float): Their minimum cosine.
        corrects_top_k (int): Heads and retired versions issued by a correction cue.
        same_anchor_siblings (int): Siblings sharing an anchor are issued whole up to
            this count, else this many by RRF. The would-be-head and identity-producer
            routes can issue more siblings, so the commit fence keeps the same inputs.
        cross_sibling_top_k (int): Other siblings issued by cosine.
        cross_sibling_floor (float): Their minimum cosine.
        history_targets (int): Retired versions the revises and corrects routes issue
            as targets, in all (PD:P8); those resolution asks round 2 to issue come on
            top, within the wire's 64 targets.
        history_per_head (int): Latest history entries shown beside each issued head;
            at most 2, as a memory entry lists them (C.6.2).
        tombstone_exact_newest (int): Same-digest tombstones of other entities.
        tombstone_entity_top_k (int): Same-entity tombstones by cosine.
        tombstone_entity_floor (float): Their minimum cosine.
        tombstone_scope_top_k (int): Scope-wide tombstones by cosine.
        tombstone_scope_floor (float): Their minimum cosine.
        tombstone_cross_script_entity_floor (float): The same-entity floor for a
            tombstone whose statement writes letters of other scripts than the
            member's, as a name copied in its source script does (ML2).
        tombstone_cross_script_scope_floor (float): The scope floor for such a tombstone.
        tombstone_total (int): Tombstones issued in all.
        entity_candidates (int): Candidates per judged slot besides the nominee.
        entity_facts (int): Readable facts shown per entity: its most recent heads.
        context_messages (int): Context budget: complete small same-request gaps
            between citations first, then messages before the earliest citation.
        context_message_chars (int): Characters shown from each context message's end.
        declaration_jaccard (float): Descriptor-token Jaccard that joins two unnamed
            new-entity declarations (C.4.3 #8).
        rrf_k (int): The RRF constant.
        bm25_k1 (float): BM25 term-frequency saturation.
        bm25_b (float): BM25 length normalisation.

    Examples:
        ```python
        policy = NominationPolicyV1()
        variant = NominationPolicyV1(tombstone_scope_top_k=3)  # an O7 variant (decision C8)
        assert variant.digest != policy.digest
        ```
    """

    model_config = ConfigDict(frozen=True)

    schema_version: Literal[1] = 1
    primary_all_cap: Annotated[int, Field(ge=1, le=MAX_JUDGE_TARGETS)] = 16
    primary_top_k: Annotated[int, Field(ge=1, le=MAX_JUDGE_TARGETS)] = 8
    secondary_top_k: Annotated[int, Field(ge=0, le=MAX_JUDGE_TARGETS)] = 8
    scope_dense_top_k: Annotated[int, Field(ge=0, le=MAX_JUDGE_TARGETS)] = 4
    scope_dense_floor: _Floor = 0.80
    corrects_top_k: Annotated[int, Field(ge=0, le=MAX_JUDGE_TARGETS)] = 4
    same_anchor_siblings: Annotated[int, Field(ge=0, le=MAX_JUDGE_TARGETS)] = 4
    cross_sibling_top_k: Annotated[int, Field(ge=0, le=MAX_JUDGE_TARGETS)] = 2
    cross_sibling_floor: _Floor = 0.80
    history_targets: Annotated[int, Field(ge=0, le=MAX_JUDGE_TARGETS)] = 4
    history_per_head: Annotated[int, Field(ge=0, le=2)] = 2
    tombstone_exact_newest: Annotated[int, Field(ge=0, le=MAX_JUDGE_TOMBSTONES)] = 3
    tombstone_entity_top_k: Annotated[int, Field(ge=0, le=MAX_JUDGE_TOMBSTONES)] = 3
    tombstone_entity_floor: _Floor = 0.60
    tombstone_scope_top_k: Annotated[int, Field(ge=0, le=MAX_JUDGE_TOMBSTONES)] = 2
    tombstone_scope_floor: _Floor = 0.75
    tombstone_cross_script_entity_floor: _Floor = 0.50
    tombstone_cross_script_scope_floor: _Floor = 0.60
    tombstone_total: Annotated[int, Field(ge=0, le=MAX_JUDGE_TOMBSTONES)] = 8
    entity_candidates: Annotated[int, Field(ge=0, le=MAX_ENTITY_CANDIDATES)] = 6
    entity_facts: Annotated[int, Field(ge=0, le=3)] = 3
    context_messages: Annotated[int, Field(ge=0, le=4)] = 4
    context_message_chars: Annotated[int, Field(ge=1)] = 1500
    declaration_jaccard: Annotated[float, Field(gt=0.0, le=1.0)] = 0.5
    rrf_k: Annotated[int, Field(ge=1)] = 60
    bm25_k1: Annotated[float, Field(gt=0.0, allow_inf_nan=False)] = 1.2
    bm25_b: Annotated[float, Field(ge=0.0, le=1.0)] = 0.75

    @model_validator(mode="after")
    def within_wire_bounds(self) -> Self:
        """Refuse a variant that could issue more than the judge wire admits (A0.7)."""
        heads = (self.primary_all_cap + self.secondary_top_k + self.scope_dense_top_k
                 + MAX_REVISES + self.corrects_top_k)
        targets = heads + self.same_anchor_siblings + self.cross_sibling_top_k
        if targets + self.history_targets > MAX_JUDGE_TARGETS:
            raise ValueError(f"A member is issued at most {MAX_JUDGE_TARGETS} targets")
        if heads * self.history_per_head > MAX_JUDGE_HISTORY:
            raise ValueError(f"A member is shown at most {MAX_JUDGE_HISTORY} history entries")
        if self.primary_top_k > self.primary_all_cap:
            raise ValueError("A truncated partition issues fewer heads than a complete one")
        return self

    @property
    def digest(self) -> str:
        """The policy identity: sha256 of the canonical ``["aml-nomination-policy-v1", policy]``."""
        payload = ["aml-nomination-policy-v1", self.model_dump(mode="json")]
        return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


# Scoring (C.4.2) -----------------------------------------------------------------------------


def _same_space(space: str, expected: str) -> None:
    if space != expected:
        raise AMLError("embedding_space_mismatch", "A nominated vector is of another space")


def _terms(tokens: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted(set(tokens)))


@dataclass(frozen=True, slots=True, eq=False)
class _Doc:
    """A head, retired version or sibling scored with the shared sparse terms."""

    key: str
    unit: UnitVector
    document: SparseDocument

    @classmethod
    def of(cls, key: str, vector: Sequence[float], tokens: Sequence[str]) -> _Doc:
        return cls(key, unit_vector(vector), SparseDocument.of(tokens))

    @classmethod
    def of_snapshot(cls, key: str, snapshot: MemorySnapshotV2) -> _Doc:
        """A stored version's document; its unit vector and terms are computed once per snapshot."""
        return cls(key, snapshot_unit(snapshot), statement_document(snapshot))


def _bm25(stats: SparseStatistics, terms: Collection[str], doc: _Doc,
          policy: NominationPolicyV1) -> float:
    """C.4.2's BM25 of one document over the pin's heads, with the policy's ``k1`` and ``b``."""
    return stats.bm25(terms, doc.document, k1=policy.bm25_k1, b=policy.bm25_b)


@dataclass(frozen=True, slots=True, eq=False)
class ScoredRow:
    """One row of a ranked pool with its scores for one query of a member (C.4.2).

    Attributes:
        key (str): The row's object id: a head's memory id, a retired version's
            revision id or a sibling's position.
        order (VersionOrder): The commit/source/content coordinate for tied scores.
        cosine (float): Its cosine, rounded half-even to 6 decimals.
        bm25 (float): Its BM25 at the pin's statistics, rounded to 6 decimals; 0
            has no BM25 rank.
    """

    key: str
    cosine: float
    bm25: float
    order: VersionOrder


_FUSION_BITS: Final = 128
"""Fixed-point bits of a fusion's sort key (`RrfRanking.of`).

A fusion is ``1/a + 1/b`` with ``a, b <= k + n``, so it is a fraction of denominator at
most ``(k + n)^2``, and two different fusions differ by at least ``(k + n)^-4``. Below
2^31 rows and constant, that gap is above 2^-124: multiplied by 2^128 and floored, two
different fusions keep their order and equal ones get the same integer.
"""


class _Fusion(Mapping[str, Fraction]):
    """Each row's exact fusion, as the small numerator and denominator it was summed from."""

    __slots__ = ("_terms",)

    def __init__(self, terms: dict[str, tuple[int, int]]) -> None:
        self._terms = terms

    def __getitem__(self, key: str) -> Fraction:
        return Fraction(*self._terms[key])

    def __iter__(self) -> Iterator[str]:
        return iter(self._terms)

    def __len__(self) -> int:
        return len(self._terms)

    def __repr__(self) -> str:
        return repr(dict(self))


_COSINE: Final = attrgetter("cosine")
_BM25: Final = attrgetter("bm25")


@dataclass(frozen=True, slots=True)
class RrfRanking:
    """One pool ranked by dense, by BM25 and by their RRF fusion (C.4.2).

    The one RRF selection of nomination's ranked routes (entity, secondary,
    corrects and same-anchor siblings) and of the commit fence's re-rank
    (`aml.store.commit_fence.ranked_selection`), which must select exactly as
    nomination did (C.11.3, A2). The dense list orders the pool by cosine and the
    BM25 list its rows with a positive BM25 by BM25, each descending and then by
    commit and source order. The fusion ``1/(k + r_dense) + 1/(k + r_bm25)`` is an exact fraction
    without a BM25 term for a row off the BM25 list. The pool's order is fusion
    descending, then cosine descending, then commit and source order.

    Attributes:
        order (tuple[ScoredRow, ...]): The pool in RRF order.
        dense_rank (Mapping[str, int]): Each row's 1-based rank on the dense list.
        bm25_rank (Mapping[str, int]): Each listed row's 1-based rank on the BM25 list.
        fused (Mapping[str, Fraction]): Each row's fusion.
        top_dense (tuple[ScoredRow, ...]): The first ``FENCE_LIST`` rows of the dense list.
        top_bm25 (tuple[ScoredRow, ...]): The first ``FENCE_LIST`` rows of the BM25 list.

    Examples:
        ```python
        ranking = RrfRanking.of(rows, policy.rrf_k)
        chosen = ranking.select(whole=policy.primary_all_cap, top=policy.primary_top_k)
        record = ranking.record(chosen)
        ```
    """

    order: tuple[ScoredRow, ...]
    dense_rank: Mapping[str, int]
    bm25_rank: Mapping[str, int]
    fused: Mapping[str, Fraction]
    top_dense: tuple[ScoredRow, ...]
    top_bm25: tuple[ScoredRow, ...]

    @classmethod
    def of(cls, rows: Sequence[ScoredRow], k: int) -> RrfRanking:
        """Rank one pool.

        Args:
            rows (Sequence[ScoredRow]): The pool, each key once.
            k (int): The RRF constant (``rrf_k``).

        Returns:
            RrfRanking: The pool's lists, ranks, fusion and order.
        """
        # The key ends every order: two rows can share all commit and content
        # coordinates (one made readable again without a new version), and the
        # commit fence counts shifts against this same total order. Every list below
        # is a stable sort of this one, so it breaks its ties the same way. A float or
        # small integer key sorts far faster than a tuple key, and a long user's Add
        # ranks each member's whole partition several times.
        settled = sorted(rows, key=lambda row: (row.order, row.key))
        dense = sorted(settled, key=_COSINE, reverse=True)
        lexical = sorted((row for row in settled if row.bm25 > 0), key=_BM25, reverse=True)
        dense_rank = {row.key: rank for rank, row in enumerate(dense, 1)}
        bm25_rank = {row.key: rank for rank, row in enumerate(lexical, 1)}
        # A fusion is one exact fraction of small integers, and the sort key is its
        # fixed-point value (`_FUSION_BITS`), which orders the pool exactly as the
        # fractions do, ties included. The common denominator lcm(k + 1, ..., k + n) of
        # every term grows with the pool: at 5k heads its arithmetic was 22% of an Add.
        terms: dict[str, tuple[int, int]] = {}
        scaled: dict[str, int] = {}
        for row in dense:
            first = k + dense_rank[row.key]
            rank = bm25_rank.get(row.key)
            if rank is None:
                numerator, denominator = 1, first
            else:
                second = k + rank
                numerator, denominator = first + second, first * second
            terms[row.key] = numerator, denominator
            scaled[row.key] = (numerator << _FUSION_BITS) // denominator
        order = sorted(dense, key=lambda row: scaled[row.key], reverse=True)
        return cls(tuple(order), dense_rank, bm25_rank, _Fusion(terms),
                   tuple(dense[:FENCE_LIST]), tuple(lexical[:FENCE_LIST]))

    def select(self, *, whole: int, top: int) -> tuple[ScoredRow, ...]:
        """Select from the pool as a ranked route does (C.4.3).

        Args:
            whole (int): The pool is selected whole while it holds at most this
                many rows; a top-k route passes its k.
            top (int): Otherwise its first ``top`` rows in RRF order are selected.

        Returns:
            tuple[ScoredRow, ...]: The selected rows in RRF order.
        """
        return self.order if len(self.order) <= whole else self.order[:top]

    def record(self, chosen: Sequence[ScoredRow]) -> RankRecordV1:
        """The fence's view of a selection: the top-16 lists, selected ranks and ``s_k``."""
        return RankRecordV1(
            selected=[row.key for row in chosen],
            top_dense=[RankedV1(object_id=row.key, score=row.cosine) for row in self.top_dense],
            top_bm25=[RankedV1(object_id=row.key, score=row.bm25) for row in self.top_bm25],
            selected_ranks=[SelectedRankV1(
                object_id=row.key, dense_rank=self.dense_rank[row.key],
                bm25_rank=self.bm25_rank.get(row.key),
            ) for row in chosen],
            s_k=round(float(min(self.fused[row.key] for row in chosen)), 6) if chosen else None,
        )


# Route records (content-free; C.4.5) ---------------------------------------------------------


_ROUTE_ORDER: Final[Mapping[str, int]] = {
    "entity": 0, "secondary": 1, "scope_dense": 2, "revises": 3, "corrects": 4, "siblings": 5,
    "tombstones": 6,
}


# Member input (C.1) ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MemberSlotV1:
    """One anchor slot of a member, resolved at the round's pin (C.1, C.3).

    Attributes:
        slot (int): 0 for the primary anchor, then 1..7.
        binding (Binding): How the slot is bound.
            Supported values:
            - ``self``: the authenticated user; never judged.
            - ``participant``: a source or labelled participant; never judged.
            - ``sdk``: an entity an SDK binding fixed; never judged.
            - ``nominated``: an existing entity or a new declaration (``mint_key``)
              that the judge confirms or re-picks.
            - ``open``: an unresolved referent or an erased entity; candidates only.
        entity_id (str | None): The bound or nominated entity; None for an open
            slot, an unminted new declaration or an unminted participant.
        mint_key (str | None): An unminted new declaration's key (handle N).
        participant_key (str | None): A participant's key.
        name (str | None): The declared name of a new entity.
        descriptor (str | None): The declared descriptor of a new entity.
        entity_type (str | None): The declared type of a new entity.
        mention (str | None): The text an unresolved referent was written as.
    """

    slot: int
    binding: Binding
    entity_id: str | None = None
    mint_key: str | None = None
    participant_key: str | None = None
    name: str | None = None
    descriptor: str | None = None
    entity_type: str | None = None
    mention: str | None = None

    def __post_init__(self) -> None:
        """Check that the binding and the identities agree."""
        keys = (self.entity_id, self.mint_key, self.participant_key)
        valid = {
            "self": self.entity_id == USER and keys[1:] == (None, None),
            "participant": self.participant_key is not None and self.mint_key is None,
            "sdk": self.entity_id is not None and keys[1:] == (None, None),
            "nominated": (self.entity_id is None) != (self.mint_key is None)
            and self.participant_key is None,
            "open": keys == (None, None, None),
        }[self.binding]
        if not 0 <= self.slot < MAX_ANCHORS or not valid:
            raise ValueError(f"Slot {self.slot}: binding {self.binding} disagrees with its ids")

    @property
    def anchor(self) -> str | None:
        """The slot's identity for anchor tests: its entity, mint or participant key."""
        return self.entity_id or self.mint_key or self.participant_key

    @property
    def judged(self) -> bool:
        """Whether the judge decides this slot (``nominated`` or ``open``, C.7.2 #3)."""
        return self.binding in ("nominated", "open")


@dataclass(frozen=True, slots=True)
class MemberNamingV1:
    """One naming annotation of the member on one of its slots."""

    slot: int
    name: str
    name_kind: NameKind


@dataclass(frozen=True, slots=True)
class RevisesRefV1:
    """One ``revises`` reference of the member (B.6.3).

    Attributes:
        memory_id (str): The referenced Item.
        revision_id (str | None): The retired version an ``earlier`` handle named
            (B.4.3); None for a reference to a current memory.
    """

    memory_id: str
    revision_id: str | None = None


@dataclass(frozen=True, slots=True)
class CueV1:
    """The member's ``corrects_text`` and its embedding (role ``correction_cue``, A.7)."""

    text: str
    vector: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class FactTimeV1:
    """A fact's time: its expression as written and its anchored validity (A.3.2).

    Attributes:
        text (str | None): The time expression as written; None when the fact
            states no time.
        valid_from (int | None): The anchored start in epoch milliseconds.
        valid_to (int | None): The anchored end; events only.
        granularity (TimeGranularity): The granularity of the anchored bounds;
            ``unanchored`` when there are none.
    """

    text: str | None = None
    valid_from: int | None = None
    valid_to: int | None = None
    granularity: TimeGranularity = "unanchored"

    @classmethod
    def of(cls, snapshot: MemorySnapshotV2) -> FactTimeV1:
        """Read a stored version's time.

        Args:
            snapshot (MemorySnapshotV2): A version that is not erased.

        Returns:
            FactTimeV1: Its expression text, bounds and granularity.
        """
        expression = snapshot.time_expr
        text = expression.text if isinstance(expression, SourceTimeExpressionV1) else None
        return cls(text, snapshot.valid_from, snapshot.valid_to, snapshot.time_granularity)


NO_TIME: Final = FactTimeV1()
"""The time of a fact that states none."""


def slot_key(slot: MemberSlotV1) -> HandleKey | None:
    """The handle identity of a slot's entity: E for an entity or participant, N for a new one.

    Args:
        slot (MemberSlotV1): A member or sibling slot.

    Returns:
        HandleKey | None: ``entity`` keyed by the entity id, or by the
            ``participant_key`` of a participant not minted yet; ``new_entity``
            keyed by the ``mint_key`` of an unminted declaration; None for an
            open slot.
    """
    if slot.entity_id is not None:
        return handle_key("entity", slot.entity_id)
    if slot.mint_key is not None:
        return handle_key("new_entity", slot.mint_key)
    if slot.participant_key is not None:
        return handle_key("entity", slot.participant_key)
    return None


@dataclass(frozen=True, slots=True)
class SiblingV1:
    """An earlier member of the same owner and round, as the sibling route and its entry read it.

    Attributes:
        position_key (PositionKey): The sibling's position.
        statement (str): Its statement.
        kind (MemoryKind): ``state`` or ``event``.
        time (FactTimeV1): Its time.
        tokens (tuple[str, ...]): Shared sparse terms of its statement and keywords.
        vector (tuple[float, ...]): Its statement vector.
        embedding_space (str): The vector's space.
        slots (tuple[MemberSlotV1, ...]): Its resolved slots, slot 0 first.
        naming (tuple[str, ...]): The anchors its naming annotations name.
        sigma (Sigma): Its σ.
        evidence_ids (tuple[str, ...]): The Evidence its quotes cite.
    """

    position_key: PositionKey
    statement: str
    kind: MemoryKind
    time: FactTimeV1
    tokens: tuple[str, ...]
    vector: tuple[float, ...]
    embedding_space: str
    slots: tuple[MemberSlotV1, ...]
    naming: tuple[str, ...]
    sigma: Sigma
    evidence_ids: tuple[str, ...]

    @property
    def statement_digest(self) -> str:
        """The statement's digest (A0.6)."""
        return statement_digest(self.statement)

    @property
    def anchors(self) -> tuple[str, ...]:
        """Its slot anchors (`MemberSlotV1.anchor`), primary first; open slots are absent."""
        return tuple(slot.anchor for slot in self.slots if slot.anchor is not None)


@dataclass(frozen=True, slots=True)
class DeclarationV1:
    """A new-entity declaration of another unit of this owner or of a re-entered position.

    Attributes:
        owner_run_id (str): The owner that extracted it.
        unit_no (int): Its unit.
        local_key (str): Its handle in that reply, ``N1``..``N32``.
        name (str | None): Its declared name.
        descriptor (str): Its descriptor.
        entity_type (str): Its declared type.
        order (tuple[Sigma, PositionKey]): The canonical key of the first position
            anchored to it; declarations are offered in this order.
    """

    owner_run_id: str
    unit_no: int
    local_key: str
    name: str | None
    descriptor: str
    entity_type: str
    order: tuple[Sigma, PositionKey]

    @property
    def mint_key(self) -> str:
        """The key a commit mints it under (A0.6)."""
        return mint_key(self.owner_run_id, self.unit_no, self.local_key)


@dataclass(frozen=True, slots=True)
class MemberV1:
    """One position under review in one round (C.1), built from its ``ProposalV2``.

    Attributes:
        position_key (PositionKey): The position.
        round (RoundNo): The review round.
        statement (str): The canonical statement.
        kind (MemoryKind): ``state`` or ``event``.
        keywords (tuple[str, ...]): The member's keywords.
        vector (tuple[float, ...]): The statement's embedding.
        embedding_space (str): Its space; it must be the view's.
        slots (tuple[MemberSlotV1, ...]): Resolved anchors, slot 0 first.
        quotes (tuple[QuoteSpanV1, ...]): Every located quote span; at least one.
        groups (tuple[tuple[int, ...], ...]): The admitted support groups over
            ``quotes``, canonical (A0.6): sorted, unique, covering every quote.
        sigma (Sigma): The member's σ, the maximum over its spans.
        time (FactTimeV1): The member's time.
        sensitivity_classes (tuple[SensitivityClass, ...]): Its protected classes.
        naming (tuple[MemberNamingV1, ...]): Naming annotations.
        revises (tuple[RevisesRefV1, ...]): ``revises`` references, at most 8.
        cue (CueV1 | None): ``corrects_text`` and its vector.
        siblings (tuple[SiblingV1, ...]): Earlier members of the round in
            canonical order; positions settled in B.8 are never among them.
        declarations (tuple[DeclarationV1, ...]): New-entity declarations of the
            owner's other units and of re-entered positions.
    """

    position_key: PositionKey
    round: RoundNo
    statement: str
    kind: MemoryKind
    keywords: tuple[str, ...]
    vector: tuple[float, ...]
    embedding_space: str
    slots: tuple[MemberSlotV1, ...]
    quotes: tuple[QuoteSpanV1, ...]
    groups: tuple[tuple[int, ...], ...]
    sigma: Sigma
    time: FactTimeV1 = NO_TIME
    sensitivity_classes: tuple[SensitivityClass, ...] = ()
    naming: tuple[MemberNamingV1, ...] = ()
    revises: tuple[RevisesRefV1, ...] = ()
    cue: CueV1 | None = None
    siblings: tuple[SiblingV1, ...] = ()
    declarations: tuple[DeclarationV1, ...] = ()

    def __post_init__(self) -> None:
        """Check slot numbering, support groups, bounds and the siblings' canonical order."""
        numbers = [slot.slot for slot in self.slots]
        if not numbers or numbers != list(range(len(numbers))):
            raise ValueError("Slots are numbered from 0 in order")
        if len(self.revises) > MAX_REVISES or not self.quotes or (
            len(self.quotes) > MAX_MEMORY_SPANS
        ):
            raise ValueError("A member has 1..256 quotes and at most 8 revises references")
        validate_support_groups([list(group) for group in self.groups], len(self.quotes))
        if len(self.groups) > MAX_SUPPORT_GROUPS or any(
            len(group) > MAX_GROUP_SPANS for group in self.groups
        ):
            raise ValueError("A member has at most 8 groups of at most 32 spans")
        order = [(sibling.sigma, sibling.position_key) for sibling in self.siblings]
        if order != sorted(set(order)) or (
            order and order[-1] >= (self.sigma, self.position_key)
        ):
            raise ValueError("Siblings are earlier members, in canonical order")

    @property
    def statement_digest(self) -> str:
        """The statement's digest (A0.6)."""
        return statement_digest(self.statement)

    @property
    def tokens(self) -> tuple[str, ...]:
        """Shared sparse terms of the statement and keywords: the BM25 query (C.4.2)."""
        return sparse_terms(" ".join((self.statement, *self.keywords)))

    def as_sibling(self) -> SiblingV1:
        """Return this member as a later member of its round sees it.

        Args:
            None.

        Returns:
            SiblingV1: Its identity, statement, kind, time, tokens, vector, slots,
                naming, σ and Evidence.
        """
        by_slot = {slot.slot: slot.anchor for slot in self.slots}
        return SiblingV1(
            position_key=self.position_key, statement=self.statement, kind=self.kind,
            time=self.time, tokens=self.tokens, vector=self.vector,
            embedding_space=self.embedding_space, slots=self.slots,
            naming=tuple(dict.fromkeys(
                anchor for naming in self.naming
                if (anchor := by_slot.get(naming.slot)) is not None
            )),
            sigma=self.sigma,
            evidence_ids=tuple(sorted({span.evidence_id for span in self.quotes})),
        )


def resolve_slots(
    view: AddReadView, position_key: PositionKey, choices: Sequence[EntityChoiceV1],
) -> tuple[MemberSlotV1, ...]:
    """Resolve a position's admitted anchors at a pin (C.1, C.3).

    The user is ``self``. A participant is bound to its entity, looked up by
    key when it has none (or lost it); without one it stays an unminted
    participant. An SDK binding is ``sdk`` and a catalog choice ``nominated``.
    A new declaration is ``nominated`` under its ``mint_key`` until a commit
    mints it, then as that existing entity. An unresolved referent, and any
    anchor whose entity is no longer active, is ``open``: so is a declaration
    whose minted entity was erased (``AddReadView.spent``), which is never
    minted again.

    Args:
        view (AddReadView): The round's pin.
        position_key (PositionKey): The position; its owner and unit key the
            position's new declarations.
        choices (Sequence[EntityChoiceV1]): The admitted anchors, primary first.

    Returns:
        tuple[MemberSlotV1, ...]: One slot per choice, numbered from 0.
    """
    owner_run_id, unit_no, _position = position_key
    slots: list[MemberSlotV1] = []
    for number, choice in enumerate(choices):
        if isinstance(choice, UserChoiceV1):
            slot = MemberSlotV1(number, "self", USER)
        elif isinstance(choice, ParticipantChoiceV1):
            found = view.entity(choice.entity_id) if choice.entity_id is not None else None
            found = found or view.participant(choice.participant_key)
            if found is None and choice.entity_id is not None:
                slot = MemberSlotV1(number, "open")
            else:
                slot = MemberSlotV1(
                    number, "participant", found.entity_id if found is not None else None,
                    participant_key=choice.participant_key,
                )
        elif isinstance(choice, ExistingChoiceV1):
            binding: Binding = "sdk" if choice.via == "sdk" else "nominated"
            if view.entity(choice.entity_id) is None:
                slot = MemberSlotV1(number, "open")
            else:
                slot = MemberSlotV1(number, binding, choice.entity_id)
        elif isinstance(choice, NewEntityChoiceV1):
            key = mint_key(owner_run_id, unit_no, choice.local_key)
            minted = view.minted(key)
            if minted is None and view.spent(key):
                slot = MemberSlotV1(number, "open")
            else:
                slot = MemberSlotV1(
                    number, "nominated", minted.entity_id if minted is not None else None,
                    None if minted is not None else key, name=choice.name,
                    descriptor=choice.descriptor, entity_type=choice.entity_type,
                )
        else:
            slot = MemberSlotV1(number, "open", mention=choice.mention)
        slots.append(slot)
    return tuple(slots)


def round_members(members: Iterable[MemberV1]) -> tuple[MemberV1, ...]:
    """Put one owner's round in canonical order and give each member its siblings (C.4.3 #6).

    Args:
        members (Iterable[MemberV1]): Every member of the round; their own
            ``siblings`` are replaced.

    Returns:
        tuple[MemberV1, ...]: The members in canonical order ``(σ, position_key)``,
            each with every earlier member as a sibling. A member's ``SiblingV1``
            is built once and shared by every later member: it is immutable and
            reads nothing of the siblings it is given, so a round holds one per
            member rather than one per pair.
    """
    ordered = sorted(members, key=lambda member: (member.sigma, member.position_key))
    result: list[MemberV1] = []
    earlier: list[SiblingV1] = []
    for member in ordered:
        result.append(replace(member, siblings=tuple(earlier)))
        earlier.append(member.as_sibling())
    return tuple(result)


# Nomination output ----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class NominatedTargetV1:
    """One issued target: a current head, a retired version or a sibling.

    Attributes:
        kind (TargetKind): ``head`` (handle M), ``retired`` (handle H, issued to a
            history correction only) or ``sibling`` (handle S).
        route (TargetRoute): The first route that selected it; ``revises`` also for
            a target issued because resolution asked round 2 to (L-Q3).
        object_id (str): The memory id, or ``owner:unit:position`` for a sibling.
        revision_id (str | None): The head revision or retired version; None for
            a sibling.
        statement_digest (str): Its statement's digest.
        sigma (Sigma): Its σ.
        primary (str | None): Its primary anchor.
        anchors (tuple[str, ...]): Its anchors, primary first.
        naming (tuple[str, ...]): The entities its naming annotations name.
        version (VersionRecord | None): The head or retired version at the pin.
        ended_by (EndedBy | None): How a retired version was ended.
        sibling (SiblingV1 | None): The sibling.
    """

    kind: TargetKind
    route: TargetRoute
    object_id: str
    revision_id: str | None
    statement_digest: str
    sigma: Sigma
    primary: str | None
    anchors: tuple[str, ...]
    naming: tuple[str, ...]
    version: VersionRecord | None = None
    ended_by: EndedBy | None = None
    sibling: SiblingV1 | None = None

    @property
    def handle_key(self) -> HandleKey:
        """The target's handle identity (C.4.4)."""
        if self.kind == "sibling":
            return handle_key("sibling", self.object_id)
        return handle_key("memory" if self.kind == "head" else "history", self.object_id,
                          self.revision_id)


@dataclass(frozen=True, slots=True)
class HistoryContextV1:
    """A history entry shown beside an issued head (C.4.3 #5).

    Attributes:
        head_memory_id (str): The issued head it is shown beside.
        entry (HistoryEntry): The earlier version and how it was ended.
    """

    head_memory_id: str
    entry: HistoryEntry

    @property
    def handle_key(self) -> HandleKey:
        """The entry's handle identity (C.4.4)."""
        version = self.entry.version
        return handle_key("history", version.memory_id, version.revision_id)


@dataclass(frozen=True, slots=True)
class NominatedTombstoneV1:
    """One issued tombstone: its route, cosine and row (read by Add only, UD3)."""

    route: TombstoneRoute
    cosine: float
    record: TombstoneRecord


@dataclass(frozen=True, slots=True)
class _ScoredTombstone:
    """A route-7 candidate: its row, its cosine to the member and whether the two
    statements write letters of different scripts."""

    record: TombstoneRecord
    cosine: float
    cross_script: bool


@functools.lru_cache(maxsize=4096)
def _letter_scripts(text: str) -> frozenset[str]:
    """The writing systems of a text's letters other than Latin (C.4.3 #7, ML2).

    A letter's system is the first word of its Unicode name after NFKC: ``CYRILLIC``,
    ``ARABIC``, ``DEVANAGARI``, ``THAI``, ``HANGUL``, ``HIRAGANA``, ``KATAKANA``,
    ``CJK`` and so on. Statements are English, so another system marks a name, a
    place or a term copied in its source script (B.6.3).

    Args:
        text (str): A statement.

    Returns:
        frozenset[str]: The systems; empty for a text whose letters are all Latin or
            unnamed in the running Unicode database.
    """
    return frozenset(
        unicodedata.name(char, "").partition(" ")[0]
        for char in unicodedata.normalize("NFKC", text) if char.isalpha()
    ) - {"LATIN", ""}


def _bare_member(member: MemberV1) -> bool:
    """Whether a member names its one non-user entity by a neutral kind alone (ML2).

    Its statement has the bare-name syntax (`bare_naming`) and one of its naming
    annotations gives that slot the statement's name: "A person is named Rowan."

    Args:
        member (MemberV1): The member.

    Returns:
        bool: True for a bare-name member.
    """
    found = bare_naming(member.statement)
    if found is None or len(member.slots) != 1 or member.slots[0].binding == "self":
        return False
    name = name_norm(found[1])
    return any(naming.slot == 0 and name_norm(naming.name) == name for naming in member.naming)


@dataclass(frozen=True, slots=True)
class EntityCandidateV1:
    """One candidate of a judged slot: an existing entity (E) or a new declaration (N).

    Attributes:
        source (CandidateSource): The route-8 source that offered it first.
        entity (EntityRecord | None): An existing entity.
        mint_key (str | None): A new declaration's key.
        declaration (DeclarationV1 | None): Another unit's declaration, for a
            candidate a matching declaration offered.
    """

    source: CandidateSource
    entity: EntityRecord | None
    mint_key: str | None
    declaration: DeclarationV1 | None

    @property
    def key(self) -> str:
        """The entity id or mint key."""
        return self.entity.entity_id if self.entity is not None else self.mint_key or ""

    @property
    def handle_key(self) -> HandleKey:
        """The candidate's handle identity (C.4.4)."""
        if self.entity is not None:
            return handle_key("entity", self.entity.entity_id)
        return handle_key("new_entity", self.key)


@dataclass(frozen=True, slots=True)
class SlotNominationV1:
    """One slot's entity and, for a judged slot, its candidates.

    Attributes:
        slot (MemberSlotV1): The slot.
        entity (EntityRecord | None): The bound or nominated existing entity.
        candidates (tuple[EntityCandidateV1, ...]): For a judged slot, the nominee
            first, then up to six more in route-8 source order; empty otherwise.
        catalog_complete (bool | None): For a judged slot, whether every live
            entity whose type or names match it was issued (decision Q2); None
            for a bound slot.
    """

    slot: MemberSlotV1
    entity: EntityRecord | None
    candidates: tuple[EntityCandidateV1, ...]
    catalog_complete: bool | None


@dataclass(frozen=True, slots=True)
class IdentityReachV1:
    """Identity inputs shared by sibling nomination and finding #4 (C.11.1).

    Anchors include existing candidates; mint keys distinguish the member's
    own declarations from other offered declarations. Terms and types come
    only from judged slots. Complete types and open slots record where a new
    entity could change ``catalog_complete`` from true to false. A participant
    can become a speaker-name candidate whenever there is a judged slot.
    """

    anchors: frozenset[str]
    mint_keys: frozenset[str]
    candidate_mint_keys: frozenset[str]
    terms: frozenset[str]
    judged_slot: bool
    types: frozenset[str]
    open_slot: bool
    complete_types: frozenset[str]
    complete_open: bool


def identity_reach(
    slots: Sequence[SlotNominationV1], naming: Sequence[MemberNamingV1],
) -> IdentityReachV1:
    """Collect the identity inputs that a new entity or name can change.

    Args:
        slots: The member's slots, candidates and catalog completeness at its pin.
        naming: The member's naming annotations, included in judged slot texts.

    Returns:
        IdentityReachV1: The anchors, declarations, slot-text terms, judged types
        and completeness flags used both before judging and at the commit fence.
    """
    anchors: set[str] = set()
    mint_keys: set[str] = set()
    offered: set[str] = set()
    terms: set[str] = set()
    types: set[str] = set()
    complete: set[str] = set()
    judged_slot = open_slot = complete_open = False
    for item in slots:
        slot = item.slot
        if slot.entity_id is not None:
            anchors.add(slot.entity_id)
        if slot.mint_key is not None:
            mint_keys.add(slot.mint_key)
        if not slot.judged:
            continue
        judged_slot = True
        if slot.binding == "open":
            open_slot = True
            complete_open = complete_open or bool(item.catalog_complete)
        texts = [slot.mention, slot.name, *(entry.name for entry in naming
                                         if entry.slot == slot.slot)]
        kind = (item.entity.entity_type if item.entity is not None
                else slot.entity_type or "").casefold()
        if item.entity is not None:
            texts += [row.name for row in item.entity.names]
        if kind:
            types.add(kind)
            if item.catalog_complete:
                complete.add(kind)
        for text in texts:
            if text:
                terms.update(token_seq(text))
        for candidate in item.candidates:
            if candidate.entity is not None:
                anchors.add(candidate.entity.entity_id)
            elif candidate.mint_key is not None and candidate.mint_key != slot.mint_key:
                offered.add(candidate.mint_key)
    return IdentityReachV1(
        anchors=frozenset(anchors), mint_keys=frozenset(mint_keys),
        candidate_mint_keys=frozenset(offered - mint_keys), terms=frozenset(terms),
        judged_slot=judged_slot, types=frozenset(types), open_slot=open_slot,
        complete_types=frozenset(complete),
        complete_open=complete_open,
    )


@dataclass(frozen=True, slots=True)
class SourceMessageV1:
    """A message the member input shows: cited in full, or as context before them."""

    message: MessageRecord
    cited: bool

    @property
    def handle_key(self) -> HandleKey:
        """The message's handle identity: its first part and how it is shown."""
        return handle_key("message", self.message.parts[0].object_id,
                          "cited" if self.cited else "context")


@dataclass(frozen=True, slots=True)
class NominationV1:
    """Everything one member is issued at one pin, and how it was chosen (C.4).

    Attributes:
        member (MemberV1): The member.
        pin (ReadPin): The view's pin.
        embedding_space (str): The member's space.
        policy (NominationPolicyV1): The parameters it was nominated with.
        lexical (SparseStatistics): The pin's statistics with ``frequencies``
            for the member's query terms (statement, keywords and cue).
        readable_heads (frozenset[str]): The pin's P-MEM identities, shared by
            every member of its index. Source indexing can make an existing
            version readable without changing its Item's commit sequence.
        routes (tuple[RouteRecordV1, ...]): The route records, in route order.
        targets (tuple[NominatedTargetV1, ...]): Heads in route order, then
            siblings, then retired versions.
        history (tuple[HistoryContextV1, ...]): Context history entries.
        tombstones (tuple[NominatedTombstoneV1, ...]): Issued tombstones.
        slots (tuple[SlotNominationV1, ...]): One per member slot.
        entity_facts (Mapping[str, tuple[MemoryRecord, ...]]): Every existing
            entity the input names, by entity id in id order, with the facts its
            entry shows: `NominationIndex.facts`, which depends on the entity and
            the pin only, so every member of a round shows an entity alike (L-Q1).
        sources (tuple[SourceMessageV1, ...]): Cited messages, then context.
        flags (NominationFlagsV1): The member flags.
        dependencies (NominationDependenciesV1): The rows the member input reads.
    """

    member: MemberV1
    pin: ReadPin
    embedding_space: str
    policy: NominationPolicyV1
    lexical: SparseStatistics
    readable_heads: frozenset[str]
    routes: tuple[RouteRecordV1, ...]
    targets: tuple[NominatedTargetV1, ...]
    history: tuple[HistoryContextV1, ...]
    tombstones: tuple[NominatedTombstoneV1, ...]
    slots: tuple[SlotNominationV1, ...]
    entity_facts: Mapping[str, tuple[MemoryRecord, ...]]
    sources: tuple[SourceMessageV1, ...]
    flags: NominationFlagsV1
    dependencies: NominationDependenciesV1

    @property
    def policy_digest(self) -> str:
        """`NominationPolicyV1.digest` of its policy."""
        return self.policy.digest

    @property
    def reviewable(self) -> bool:
        """False when an open slot has no candidate: the member is not reviewed (C.3)."""
        return all(slot.candidates for slot in self.slots if slot.slot.binding == "open")


# The index of one pin -------------------------------------------------------------------------


class NominationIndex:
    """One pin's heads, retired versions, entities and tombstones, prepared once.

    Build one index per round pin and nominate every member of the round with
    it; the function `nominate` builds a fresh index for a single member.

    Examples:
        ```python
        index = NominationIndex(view)
        nominations = [index.nominate(member, policy) for member in round_members(members)]
        ```
    """

    def __init__(self, view: AddReadView) -> None:
        """Read and prepare the pin's P-MEM heads, entities and tombstones.

        Args:
            view (AddReadView): The round's pin.

        Raises:
            AMLError: ``embedding_space_mismatch`` when a head or tombstone has
                another space than the view.
        """
        self.view = view
        self.space = view.embedding_space
        heads = view.heads()
        self.records: Mapping[str, MemoryRecord] = {record.memory_id: record for record in heads}
        self.readable_heads = frozenset(self.records)
        self.heads: tuple[_Doc, ...] = tuple(
            self._doc(record.memory_id, record.head) for record in heads
        )
        self.docs: Mapping[str, _Doc] = {doc.key: doc for doc in self.heads}
        partitions: dict[str, list[str]] = {}
        for record in heads:
            for anchor in dict.fromkeys(record.head.anchors):
                partitions.setdefault(anchor, []).append(record.memory_id)
        self.partitions: Mapping[str, tuple[str, ...]] = {
            anchor: tuple(members) for anchor, members in partitions.items()
        }
        self.stats = SparseStatistics.of(doc.document for doc in self.heads)
        self.entities: Mapping[str, EntityRecord] = {
            record.entity_id: record for record in view.entities()
        }
        self.user = view.entity(USER)
        self.entity_retrieval = EntityRetrieval.of_view(view, include_user=False)
        self.participants: Mapping[str, str] = {
            record.participant_key: entity_id for entity_id, record in self.entities.items()
            if record.participant_key is not None
        }
        self.speakers = view.speakers()
        tombstones = view.tombstones()
        for record in tombstones:
            _same_space(record.embedding_space, self.space)
        self.tombstones: tuple[tuple[TombstoneRecord, UnitVector], ...] = tuple(
            (record, unit_vector(record.embedding)) for record in tombstones
        )
        self._siblings: dict[str, tuple[tuple[float, ...], tuple[str, ...], _Doc]] = {}
        self._facts: dict[tuple[str, int], tuple[MemoryRecord, ...]] = {}
        self._head_scores: dict[bytes, list[float]] = {}

    @functools.cached_property
    def _postings(self) -> Mapping[str, set[str]]:
        """Each sparse term with the heads whose document counts it."""
        postings: dict[str, set[str]] = {}
        for doc in self.heads:
            for term, frequency in doc.document.counts.items():
                if frequency:
                    postings.setdefault(term, set()).add(doc.key)
        return postings

    def lexical_heads(self, terms: Iterable[str]) -> set[str]:
        """The heads sharing a term with ``terms``, the only ones with a BM25 above 0."""
        postings, found = self._postings, set[str]()
        for term in terms:
            found.update(postings.get(term, ()))
        return found

    @functools.cached_property
    def head_rows(self) -> UnitRows:
        """The heads' unit vectors in ``heads`` order, stacked for each member's dense scores."""
        return snapshot_rows([self.records[doc.key].head.snapshot for doc in self.heads])

    def prime(self, members: Sequence[MemberV1]) -> None:
        """Score every head against all of a round's members at once (``head_scores``).

        Each member's dense scores against every head of the user are one row
        of a products matrix; computing the whole round's matrix reads the
        heads once (``UnitRows.cosines_many``) instead of once per member. A
        member of another space is left to ``nominate``'s own refusal.
        """
        pending: dict[bytes, UnitVector] = {}
        for member in members:
            if member.embedding_space != self.space:
                continue
            try:
                unit = unit_vector(member.vector)
            except AMLError:
                continue  # Its own nomination refuses it, in member order.
            pending[bytes(unit)] = unit
        for key in self._head_scores:
            pending.pop(key, None)
        if len(pending) > 1:
            self._head_scores.update(zip(
                pending, self.head_rows.cosines_many(list(pending.values())), strict=True,
            ))

    def head_scores(self, query: UnitVector) -> list[float]:
        """The cosine of a member's unit vector with every head, in ``heads`` order.

        Exactly ``head_rows.cosines(query)``, kept per query for the round.
        """
        key = bytes(query) if isinstance(query, memoryview) else None
        scores = None if key is None else self._head_scores.get(key)
        if scores is None:
            scores = self.head_rows.cosines(query)
            if key is not None:
                self._head_scores[key] = scores
        return scores

    def _doc(self, key: str, version: VersionRecord) -> _Doc:
        _same_space(version.snapshot.embedding_space, self.space)
        return _Doc.of_snapshot(key, version.snapshot)

    @functools.cached_property
    def retired(self) -> tuple[tuple[HistoryEntry, _Doc], ...]:
        """The scope's readable retired versions (C.4.3 #4), keyed by revision id."""
        return tuple((entry, self._doc(entry.version.revision_id, entry.version))
                     for entry in self.view.retired())

    @functools.cached_property
    def retired_rows(self) -> UnitRows:
        """The retired versions' unit vectors in ``retired`` order, for a cue's dense scores."""
        return snapshot_rows([entry.version.snapshot for entry, _doc in self.retired])

    def sibling(self, sibling: SiblingV1) -> _Doc:
        """A sibling prepared for scoring, kept for the later members of its round."""
        _same_space(sibling.embedding_space, self.space)
        key = _position(sibling.position_key)
        kept = self._siblings.get(key)
        if kept is None or not (kept[0] is sibling.vector or kept[0] == sibling.vector) or (
            kept[1] != sibling.tokens
        ):
            kept = (sibling.vector, sibling.tokens, _Doc.of(key, sibling.vector, sibling.tokens))
            self._siblings[key] = kept
        return kept[2]

    def entity(self, entity_id: str) -> EntityRecord | None:
        """Return an active entity of the pin, or the virtual user."""
        return self.user if entity_id == USER else self.entities.get(entity_id)

    def live_head(self, memory_id: str) -> str | None:
        """Follow an Item to its live head: its own head, else ``supersedes`` forward (C.4.3 #3)."""
        queue, seen = [memory_id], set[str]()
        while queue:
            current = queue.pop(0)
            if current in seen:
                continue
            seen.add(current)
            if current in self.docs:
                return current
            record = self.view.memory(current)
            if record is not None:
                successors = [(link, self.view.memory(link.memory_id))
                              for link in record.successors]
                queue.extend(link.memory_id for link, _successor in sorted(
                    successors, key=lambda pair: version_order(pair[1].head)
                    if pair[1] is not None else (0, (0, 0, 0, 0, 0), "", 0),
                ))
        return None

    def facts(self, entity_id: str, limit: int) -> tuple[MemoryRecord, ...]:
        """Return the facts an entity's entry shows: its most recent readable heads (L-Q1).

        The heads anchored to the entity at the pin, newest first by
        ``recorded_at`` and then by commit sequence descending, as the view orders
        history. The result depends on the entity and the pin only, never on the
        member, its slot roles or its re-entry, so one round shows an entity alike
        to every member.

        Args:
            entity_id (str): An entity id, ``user`` included.
            limit (int): The most facts to return (``entity_facts``).

        Returns:
            tuple[MemoryRecord, ...]: Up to ``limit`` memories.
        """
        key = (entity_id, limit)
        found = self._facts.get(key)
        if found is None:
            heads = sorted((self.records[memory_id].head
                            for memory_id in self.partitions.get(entity_id, ())),
                           key=lambda head: (head.recorded_at, version_order(head)), reverse=True)
            found = tuple(self.records[head.memory_id] for head in heads[:limit])
            self._facts[key] = found
        return found

    def nominate(self, member: MemberV1, policy: NominationPolicyV1, *,
                 issued: Sequence[IssueV1] = ()) -> NominationV1:
        """Nominate one member's neighbours at this index's pin (C.4.3).

        Args:
            member (MemberV1): The member, with its siblings and declarations.
            policy (NominationPolicyV1): The parameters.
            issued (Sequence[IssueV1]): For a round-2 member, the targets resolution
                asked round 2 to issue (``Round2RequestV1.issue``, C.9.3 ``?``). Each
                is issued under route ``revises`` when it is still that kind at the
                pin: a head while it is the Item's current head, a retired version
                only to a member that corrects history (N-Q3). Round 1 issues none.

        Returns:
            NominationV1: The issued targets, history, tombstones, entity
                candidates and facts, and sources with their route records.

        Raises:
            AMLError: ``embedding_space_mismatch`` when the member, a sibling, a
                head, a retired version or a tombstone has another space;
                ``binding_mismatch`` when a slot's entity is not active at the
                pin, a quote cites Evidence outside the view, or a round-1
                member is given targets to issue.
        """
        return _Run(self, member, policy, tuple(issued)).result()


def nominate(view: AddReadView, member: MemberV1, policy: NominationPolicyV1, *,
             issued: Sequence[IssueV1] = ()) -> NominationV1:
    """Nominate one member's neighbours at a pin (C.4; I9).

    Args:
        view (AddReadView): The round's pin.
        member (MemberV1): The member, with its siblings and declarations.
        policy (NominationPolicyV1): The parameters.
        issued (Sequence[IssueV1]): A round-2 member's targets to issue; see
            `NominationIndex.nominate`.

    Returns:
        NominationV1: See `NominationIndex.nominate`.

    Raises:
        AMLError: See `NominationIndex.nominate`.

    Examples:
        ```python
        nomination = nominate(view, member, NominationPolicyV1())
        snapshot = neighbor_snapshot(nomination, bind=keyring.digest,
                                     owner_run_id=owner, member_input_digest=digest)
        again = nominate(view2, member2, NominationPolicyV1(), issued=request.issue)
        ```
    """
    return NominationIndex(view).nominate(member, policy, issued=issued)


@dataclass(frozen=True, slots=True)
class _SlotNames:
    """A judged slot's texts and their matches on the pin's name index (A.6.5; C.4.3 #8).

    Attributes:
        texts: The slot's mention and name, its naming names and, for an existing
            entity, that entity's readable names.
        exact: The entities with a readable name token-contained in one of the
            texts (exact phrase).
        scores: The entities scoring above zero by name BM25 over the texts.
        speakers: The participants one of the texts names by its label or by a
            speaker name of its readable Evidence, as catalog route 4 lists
            them (B.4.2). A participant has no entity name (PD:P4), so neither
            name source above can offer it.
    """

    texts: tuple[str, ...]
    exact: frozenset[str]
    scores: Mapping[str, float]
    speakers: frozenset[str]


class _Run:
    """One member's nomination while its routes run."""

    def __init__(self, index: NominationIndex, member: MemberV1,
                 policy: NominationPolicyV1, issued: tuple[IssueV1, ...]) -> None:
        _same_space(member.embedding_space, index.space)
        for slot in member.slots:
            if slot.entity_id is not None and index.entity(slot.entity_id) is None:
                raise AMLError("binding_mismatch", "A slot's entity is not active at the pin")
        if issued and member.round == 1:
            raise AMLError("binding_mismatch", "A round-1 member is issued no requested targets")
        self.index, self.member, self.policy, self.issued = index, member, policy, issued
        self.query = unit_vector(member.vector)
        self.terms = _terms(member.tokens)
        self.cue_terms = query_terms(member.cue.text) if member.cue is not None else ()
        self._term_set = frozenset(self.terms)
        self._lexical: set[str] | None = None
        self._scored: dict[str, ScoredRow] = {}
        scores = index.head_scores(self.query)
        self.dense = dict(zip((doc.key for doc in index.heads), scores, strict=True))
        # The heads at or above the scope-dense floor, in ``heads`` order: the
        # only heads the scope-dense route and the siblings' dense pool can take.
        self.near: tuple[_Doc, ...] = tuple(index.heads[position] for position in np.flatnonzero(
            np.asarray(scores, dtype=np.float64) >= policy.scope_dense_floor,
        ).tolist())
        self._cue_dense: tuple[dict[str, float], list[float]] | None = None
        self.heads: dict[str, NominatedTargetV1] = {}
        self.retired: dict[str, NominatedTargetV1] = {}
        self.routes: list[RouteRecordV1] = []
        self.references: list[RevisesReferenceV1] = []
        self.history_requested = False
        self.unavailable = False
        self.mode: PartitionMode = "empty_incomplete"

    def cue_dense(self) -> tuple[dict[str, float], list[float]]:
        """The cue's cosine with every head by key and with every retired version in order.

        Each is `cosine` of the cue's unit vector exactly (``UnitRows``), scored
        once for the corrects route and the sibling fence's cue pool.
        """
        if self._cue_dense is None:
            cue = self.member.cue
            assert cue is not None
            unit, index = unit_vector(cue.vector), self.index
            self._cue_dense = (
                dict(zip((doc.key for doc in index.heads), index.head_rows.cosines(unit),
                         strict=True)),
                index.retired_rows.cosines(unit),
            )
        return self._cue_dense

    def scored(self, keys: Iterable[str]) -> list[ScoredRow]:
        """Each head's scores for this member, each computed once however many routes read it.

        Only a head sharing a term with the member has a BM25 above 0, so the others skip it.
        """
        kept = self._scored
        if self._lexical is None:
            self._lexical = self.index.lexical_heads(self.terms)
        lexical, stats, docs = self._lexical, self.index.stats, self.index.docs
        found: list[ScoredRow] = []
        for key in keys:
            row = kept.get(key)
            if row is None:
                row = kept[key] = ScoredRow(
                    key, self.dense[key],
                    _bm25(stats, self._term_set, docs[key], self.policy)
                    if key in lexical else 0.0,
                    version_order(self.index.records[key].head),
                )
            found.append(row)
        return found

    def add_head(self, memory_id: str, route: TargetRoute) -> None:
        if memory_id in self.heads:
            return
        version = self.index.records[memory_id].head
        self.heads[memory_id] = NominatedTargetV1(
            kind="head", route=route, object_id=memory_id, revision_id=version.revision_id,
            statement_digest=version.snapshot.statement_digest, sigma=version.sigma,
            primary=version.snapshot.entity_id, anchors=version.anchors,
            naming=_named(version), version=version,
        )

    def add_retired(self, entry: HistoryEntry, route: TargetRoute, *, capped: bool = True) -> bool:
        version = entry.version
        if version.revision_id in self.retired:
            return True
        if capped and len(self.retired) >= self.policy.history_targets:
            return False
        self.retired[version.revision_id] = NominatedTargetV1(
            kind="retired", route=route, object_id=version.memory_id,
            revision_id=version.revision_id, statement_digest=version.snapshot.statement_digest,
            sigma=version.sigma, primary=version.snapshot.entity_id, anchors=version.anchors,
            naming=_named(version), version=version, ended_by=entry.ended_by,
        )
        return True

    # Route 1 ---------------------------------------------------------------------------------

    def primary(self) -> None:
        slot, policy = self.member.slots[0], self.policy
        pool: tuple[str, ...] = ()
        if slot.binding == "open":
            mode: PartitionMode = "empty_incomplete"
        elif slot.entity_id is None:
            mode = "empty_complete"
        else:
            pool = self.index.partitions.get(slot.entity_id, ())
            mode = "complete" if len(pool) <= policy.primary_all_cap else "truncated"
        ranking = RrfRanking.of(self.scored(pool), policy.rrf_k)
        chosen = ranking.select(whole=policy.primary_all_cap, top=policy.primary_top_k)
        for row in chosen:
            self.add_head(row.key, "entity")
        self.mode = mode
        self.routes.append(EntityRouteV1(
            route="entity", slot=0, entity_id=slot.entity_id, mode=mode,
            partition_size=len(pool), ranks=ranking.record(chosen),
        ))

    def secondary(self) -> None:
        slots = [(slot.slot, slot.entity_id) for slot in self.member.slots[1:]
                 if slot.entity_id is not None]
        if not slots:
            return
        pool = sorted({
            memory_id for _slot, entity_id in slots
            for memory_id in self.index.partitions.get(entity_id, ())
            if memory_id not in self.heads
        })
        limit = self.policy.secondary_top_k
        ranking = RrfRanking.of(self.scored(pool), self.policy.rrf_k)
        chosen = ranking.select(whole=limit, top=limit)
        for row in chosen:
            self.add_head(row.key, "secondary")
        self.routes.append(SecondaryRouteV1(
            route="secondary", slots=[number for number, _entity in slots],
            entity_ids=[entity_id for _slot, entity_id in slots],
            mode="complete" if len(chosen) == len(pool) else "truncated", pool_size=len(pool),
            ranks=ranking.record(chosen),
        ))

    # Route 2 ---------------------------------------------------------------------------------

    def scope_dense(self) -> None:
        policy, floor = self.policy, self.policy.scope_dense_floor
        found = sorted(
            ((self.dense[doc.key], doc.key) for doc in self.near
             if doc.key not in self.heads and self.dense[doc.key] >= floor),
            key=lambda pair: (-pair[0], version_order(self.index.records[pair[1]].head)),
        )[:policy.scope_dense_top_k]
        for _score, memory_id in found:
            self.add_head(memory_id, "scope_dense")
        full = bool(found) and len(found) == policy.scope_dense_top_k
        self.routes.append(ScopeDenseRouteV1(
            route="scope_dense",
            selected=[RankedV1(object_id=memory_id, score=score) for score, memory_id in found],
            floor=max(floor, found[-1][0]) if full else floor,
        ))

    # Route 3 ---------------------------------------------------------------------------------

    def revises(self) -> None:
        if not self.member.revises:
            return
        retired = {entry.version.revision_id: entry for entry, _doc in self.index.retired}
        for reference in self.member.revises:
            issued: VersionRecord | None = None
            if reference.revision_id is not None:
                self.history_requested = True
                entry = retired.get(reference.revision_id)
                if entry is None:
                    self.unavailable = True
                elif self.add_retired(entry, "revises"):
                    issued = entry.version
            elif (memory_id := self.index.live_head(reference.memory_id)) is None:
                self.unavailable = True
            else:
                self.add_head(memory_id, "revises")
                issued = self.index.records[memory_id].head
            self.references.append(RevisesReferenceV1(
                memory_id=reference.memory_id, revision_id=reference.revision_id,
                issued_memory_id=issued.memory_id if issued is not None else None,
                issued_revision_id=issued.revision_id if issued is not None else None,
            ))

    def issue(self, siblings: int) -> None:
        """Route 3 for round 2: issue the targets resolution asked for (L-Q3), then record
        the route with the member's own references.

        A head is issued while it is still the Item's current head at the pin; a
        retired version only to a history correction and while readable (N-Q3),
        beyond ``history_targets`` but, like every target, within the wire's 64.
        """
        requested: list[Round2IssueV1] = []
        seen: set[tuple[str, str]] = set()
        for item in self.issued:
            if (item.kind, item.revision_id) in seen:
                continue
            seen.add((item.kind, item.revision_id))
            room = len(self.heads) + siblings + len(self.retired) < MAX_JUDGE_TARGETS
            record = self.index.records.get(item.memory_id)
            if item.kind == "head":
                done = record is not None and record.head.revision_id == item.revision_id and (
                    item.memory_id in self.heads or room
                )
                if done:
                    self.add_head(item.memory_id, "revises")
            else:
                entry = next((entry for entry, _doc in self.index.retired
                              if entry.version.revision_id == item.revision_id
                              and entry.version.memory_id == item.memory_id), None)
                done = entry is not None and self.history_requested and (
                    item.revision_id in self.retired or room
                ) and self.add_retired(entry, "revises", capped=False)
            requested.append(Round2IssueV1(kind=item.kind, memory_id=item.memory_id,
                                           revision_id=item.revision_id, issued=done))
        if self.references or requested:
            self.routes.append(RevisesRouteV1(route="revises", references=self.references,
                                              round2_issue=requested))

    # Route 4 ---------------------------------------------------------------------------------

    def corrects(self) -> None:
        cue, policy, stats = self.member.cue, self.policy, self.index.stats
        if cue is None:
            return
        dense, retired = self.cue_dense()
        rows = [ScoredRow(doc.key, dense[doc.key],
                          _bm25(stats, self.cue_terms, doc, policy),
                          version_order(self.index.records[doc.key].head))
                for doc in self.index.heads if doc.key not in self.heads]
        entries: dict[str, HistoryEntry] = {}
        for (entry, doc), score in zip(self.index.retired, retired, strict=True):
            if doc.key not in self.retired:
                entries[doc.key] = entry
                rows.append(ScoredRow(doc.key, score,
                                      _bm25(stats, self.cue_terms, doc, policy),
                                      version_order(entry.version)))
        ranking = RrfRanking.of(rows, policy.rrf_k)
        chosen = ranking.select(whole=policy.corrects_top_k, top=policy.corrects_top_k)
        retired_selected: list[str] = []
        for row in chosen:
            if row.key not in entries:
                self.add_head(row.key, "corrects")
                continue
            self.history_requested = True
            if self.add_retired(entries[row.key], "corrects"):
                retired_selected.append(row.key)
        self.routes.append(CorrectsRouteV1(
            route="corrects", pool_size=len(rows), retired_selected=retired_selected,
            ranks=ranking.record(chosen),
        ))

    # Route 5 ---------------------------------------------------------------------------------

    def history(self) -> tuple[HistoryContextV1, ...]:
        if not self.history_requested:
            return ()
        shown = set(self.retired)
        context: list[HistoryContextV1] = []
        for memory_id in self.heads:
            for entry in sorted(
                # C.9.4a: a description a rewording replaced is no history context.
                (entry for entry in self.index.view.history(memory_id) if not entry.reworded),
                key=lambda entry: (entry.version.recorded_at, version_order(entry.version)),
                reverse=True,
            )[:self.policy.history_per_head]:
                if entry.version.revision_id not in shown:
                    shown.add(entry.version.revision_id)
                    context.append(HistoryContextV1(memory_id, entry))
        return tuple(context[:MAX_JUDGE_HISTORY])

    # Route 6 ---------------------------------------------------------------------------------

    def siblings(self) -> tuple[NominatedTargetV1, ...]:
        member, policy, stats = self.member, self.policy, self.index.stats
        anchors = {slot.anchor for slot in member.slots if slot.anchor is not None}
        rows: dict[str, tuple[SiblingV1, ScoredRow]] = {}
        for sibling in member.siblings:
            doc = self.index.sibling(sibling)
            rows[doc.key] = (sibling, ScoredRow(doc.key, cosine(self.query, doc.unit),
                                                _bm25(stats, self.terms, doc, policy),
                                                (0, sibling.sigma, sibling.statement,
                                                 sibling.position_key[2])))
        same = [key for key, (sibling, _row) in rows.items() if anchors & set(sibling.anchors)]
        limit = policy.same_anchor_siblings
        ranking = RrfRanking.of([rows[key][1] for key in same], policy.rrf_k)
        chosen = {row.key for row in ranking.select(whole=limit, top=limit)}
        cross = sorted(
            (key for key, (_sibling, row) in rows.items()
             if key not in same and row.cosine >= policy.cross_sibling_floor),
            key=lambda key: (-rows[key][1].cosine, rows[key][0].sigma,
                             rows[key][0].position_key[1:]),
        )[:policy.cross_sibling_top_k]
        chosen.update(cross)
        # Round 2's requested targets keep their room (`issue`).
        room = (MAX_JUDGE_TARGETS - len(self.heads) - len(self.retired) - len(chosen)
                - len(self.issued))
        wanted = dict.fromkeys(key for key in self.as_heads(rows) if key not in chosen)
        # Requested heads also supply entity candidates. Resolve them before the
        # identity producers read catalog completeness, reserving the base siblings
        # and the same requested-target room used by the extra sibling routes.
        self.issue(len(chosen))
        wanted.update((key, None) for key in self.as_producers(rows) if key not in chosen)
        chosen.update(list(wanted)[:max(room, 0)])
        if member.siblings:
            self.routes.append(SiblingsRouteV1(
                route="siblings", same_anchor_count=len(same),
                same_anchor_rrf=[row.key for row in ranking.order],
                selected=[SiblingSelectionV1(position=PositionV1.of(rows[key][0].position_key),
                                             same_anchor=key in same)
                          for key in rows if key in chosen],
            ))
        return tuple(NominatedTargetV1(
            kind="sibling", route="siblings", object_id=key, revision_id=None,
            statement_digest=sibling.statement_digest, sigma=sibling.sigma,
            primary=sibling.anchors[0] if sibling.anchors else None, anchors=sibling.anchors,
            naming=sibling.naming, sibling=sibling,
        ) for key, (sibling, _row) in rows.items() if key in chosen)

    def as_heads(self, rows: Mapping[str, tuple[SiblingV1, ScoredRow]]) -> list[str]:
        """The siblings routes 1, 2 and 4 would issue were every earlier member a head (C.11.3).

        Earlier members commit before this member's fence, each as a head ranked
        after every pin head in canonical order. Ranking them with the pin's heads
        as the entity, secondary, scope-dense and corrects routes rank (their
        pools, exclusions, cutoffs and floor; corrects by the cue, beside the
        retired versions ``revises`` did not issue) gives the siblings the fence's
        re-rank selects once they all committed; issuing them leaves the fence
        nothing to flag but what an earlier member that did not commit changed.

        Args:
            rows: Every earlier member of the round with its statement scores.

        Returns:
            The siblings those routes select, in route and then selection order.
        """
        member, policy, index = self.member, self.policy, self.index
        pin = index.view.pin.commit_seq
        virtual = {key: (sibling, ScoredRow(key, row.cosine, row.bm25,
                                            (pin + number, sibling.sigma, sibling.statement, 1)))
                   for number, (key, (sibling, row)) in enumerate(rows.items(), 1)}
        taken: dict[str, None] = {}

        def rank(heads: Iterable[str], anchored: Callable[[SiblingV1], bool], *,
                 whole: int, top: int) -> None:
            pool = [*self.scored(key for key in heads if key not in taken),
                    *(row for key, (sibling, row) in virtual.items()
                      if key not in taken and anchored(sibling))]
            taken.update((row.key, None) for row in RrfRanking.of(pool, policy.rrf_k).select(
                whole=whole, top=top))

        primary = member.slots[0]
        if primary.binding != "open" and primary.anchor is not None:
            rank(index.partitions.get(primary.entity_id, ()) if primary.entity_id else (),
                 lambda sibling: primary.anchor in sibling.anchors,
                 whole=policy.primary_all_cap, top=policy.primary_top_k)
        secondary = {slot.anchor for slot in member.slots[1:] if slot.anchor is not None}
        if secondary:
            rank(sorted({key for entity in secondary for key in index.partitions.get(entity, ())}),
                 lambda sibling: bool(secondary & set(sibling.anchors)),
                 whole=policy.secondary_top_k, top=policy.secondary_top_k)
        floor = policy.scope_dense_floor
        dense = sorted(
            [(-self.dense[doc.key], version_order(index.records[doc.key].head), doc.key)
             for doc in self.near if doc.key not in taken and self.dense[doc.key] >= floor]
            + [(-row.cosine, row.order, key) for key, (_sibling, row) in virtual.items()
               if key not in taken and row.cosine >= floor],
        )[:policy.scope_dense_top_k]
        taken.update((key, None) for _score, _order, key in dense)
        cue = member.cue
        if cue is not None:
            unit, stats, terms = unit_vector(cue.vector), index.stats, self.cue_terms
            kept = {reference.issued_memory_id for reference in self.references
                    if reference.revision_id is None}
            issued = {reference.issued_revision_id for reference in self.references
                      if reference.revision_id is not None}
            dense, retired = self.cue_dense()
            pool = [ScoredRow(doc.key, dense[doc.key], _bm25(stats, terms, doc, policy),
                              version_order(index.records[doc.key].head))
                    for doc in index.heads if doc.key not in taken and doc.key not in kept]
            pool += [ScoredRow(doc.key, score, _bm25(stats, terms, doc, policy),
                               version_order(entry.version))
                     for (entry, doc), score in zip(index.retired, retired, strict=True)
                     if doc.key not in issued]
            for key, (sibling, row) in virtual.items():
                if key not in taken:
                    shown = index.sibling(sibling)
                    pool.append(ScoredRow(key, cosine(unit, shown.unit),
                                          _bm25(stats, terms, shown, policy), row.order))
            taken.update((row.key, None) for row in RrfRanking.of(pool, policy.rrf_k).select(
                whole=policy.corrects_top_k, top=policy.corrects_top_k))
        return [key for key in taken if key in virtual]

    def as_producers(self, rows: Mapping[str, tuple[SiblingV1, ScoredRow]]) -> list[str]:
        """The siblings whose commits would write an entity or name finding #4 counts (C.11.1 #4).

        An earlier member's commit mints its new declarations and an unminted
        participant, and writes the names its naming annotations give. For this
        member the commit fence counts
        such a row unless an issued sibling wrote it (W3-RT-7): a new entity of
        the type of a judged slot whose ``catalog_complete`` is true, or any new
        entity while an open slot's is; a declaration offered as a candidate; a
        participant, which can become a speaker-name candidate for any judged
        slot irrespective of its type; and a name of an anchor, a candidate or
        its own declaration, or one sharing a token with a
        judged slot's texts, read here as the earlier member's statement tokens.
        Every potential producer is included: an earlier member may not publish,
        leaving a later member to mint a shared participant or declaration.

        Args:
            rows: Every earlier member of the round, in canonical order.

        Returns:
            The earlier members whose commits the fence would count, in order.
        """
        if not rows:
            return []
        reach = identity_reach(self.slots(), self.member.naming)
        named = reach.anchors | reach.mint_keys | reach.candidate_mint_keys
        participants = reach.judged_slot
        found: list[str] = []
        for key, (sibling, _row) in rows.items():
            hit = bool(sibling.naming) and (
                bool(named & set(sibling.naming)) or not reach.terms.isdisjoint(sibling.tokens)
            )
            for slot in sibling.slots:
                if slot.binding == "nominated" and slot.mint_key is not None and (
                    slot.mint_key not in reach.mint_keys
                ):
                    kind = (slot.entity_type or "").casefold()
                    hit = hit or reach.complete_open or (kind in reach.complete_types) or (
                        slot.mint_key in reach.candidate_mint_keys)
                elif slot.binding == "participant" and slot.entity_id is None and (
                    slot.participant_key is not None
                ):
                    hit = hit or participants
            if hit:
                found.append(key)
        return found

    # Route 7 ---------------------------------------------------------------------------------

    def tombstones(self) -> tuple[NominatedTombstoneV1, ...]:
        """Route 7: same-digest, same-entity and scope tombstones (C.4.3 #7, D.4).

        A tombstone whose statement writes letters of other scripts than the member's
        is compared under the cross-script floors: a name copied in its source script
        ("married to 罗文" against "married to Rowan") lowers the cosine of a restatement
        that keeps every other word. A bare-name member ("A person is named Rowan.") is
        issued only its own entity's tombstones and bare-name ones: it asserts nothing
        but a name, and an equal name never proves that its entity is another
        tombstone's (C.15 #5), so a forgotten "The user's spouse is named Rowan." is
        restated by a member that says whose spouse, never by the bare name alone.
        """
        member, policy = self.member, self.policy
        digest, primary = member.statement_digest, member.slots[0].entity_id
        anchors = {slot.entity_id for slot in member.slots if slot.entity_id is not None}
        scripts, bare = _letter_scripts(member.statement), _bare_member(member)
        scored = [
            _ScoredTombstone(record, cosine(self.query, unit),
                             _letter_scripts(record.statement) != scripts)
            for record, unit in self.index.tombstones
            if not bare or record.entity_id in anchors
            or bare_naming(record.statement) is not None
        ]
        issued: list[NominatedTombstoneV1] = []

        def take(route: TombstoneRoute, found: list[_ScoredTombstone], limit: int) -> None:
            for row in found[:limit]:
                if len(issued) < policy.tombstone_total:
                    issued.append(NominatedTombstoneV1(route, row.cosine, row.record))

        def open_rows() -> list[_ScoredTombstone]:
            taken = {item.record.object_id for item in issued}
            return [row for row in scored if row.record.object_id not in taken]

        def ranked(row: _ScoredTombstone) -> tuple[float, tuple[int, str, str]]:
            return -row.cosine, tombstone_order(row.record)

        take("exact_digest", sorted(
            (row for row in scored
             if row.record.statement_digest == digest and row.record.entity_id != primary),
            key=lambda row: (-row.record.tombstone_seq, tombstone_order(row.record)),
        ), policy.tombstone_exact_newest)
        take("same_entity", sorted(
            (row for row in open_rows()
             if anchors & {row.record.entity_id, *row.record.about_entity_ids}
             and row.cosine >= (policy.tombstone_cross_script_entity_floor if row.cross_script
                                else policy.tombstone_entity_floor)),
            key=ranked,
        ), policy.tombstone_entity_top_k)
        take("scope", sorted(
            (row for row in open_rows()
             if row.cosine >= (policy.tombstone_cross_script_scope_floor if row.cross_script
                               else policy.tombstone_scope_floor)),
            key=ranked,
        ), policy.tombstone_scope_top_k)
        self.routes.append(TombstonesRouteV1(
            route="tombstones",
            selected=[TombstoneSelectionV1(tombstone_id=item.record.object_id,
                                           route=item.route, cosine=item.cosine)
                      for item in issued],
            floors=TombstoneFloorsV1(
                same_entity=policy.tombstone_entity_floor, scope=policy.tombstone_scope_floor,
                cross_script_same_entity=policy.tombstone_cross_script_entity_floor,
                cross_script_scope=policy.tombstone_cross_script_scope_floor,
            ),
        ))
        return tuple(issued)

    # Route 8 ---------------------------------------------------------------------------------

    def slot_texts(self, slot: MemberSlotV1, entity: EntityRecord | None) -> tuple[str, ...]:
        """The texts a slot is matched on: its mention and name, its naming names and,
        for an existing entity, that entity's readable names."""
        texts = [slot.mention, slot.name]
        texts += [naming.name for naming in self.member.naming if naming.slot == slot.slot]
        texts += [row.name for row in entity.names] if entity is not None else []
        return tuple(dict.fromkeys(text for text in texts if text and token_seq(text)))

    def slot_names(self, slot: MemberSlotV1, entity: EntityRecord | None) -> _SlotNames:
        """Match a slot's texts on the pin's name index once, for its candidates and its
        ``catalog_complete``: A.6.5's exact phrase and name BM25."""
        texts = self.slot_texts(slot, entity)
        index = self.index
        retrieval, participants = index.entity_retrieval, index.participants
        return _SlotNames(texts, frozenset(entity_id for text in texts
                                           for entity_id in retrieval.exact(text)),
                          retrieval.scores(query_terms(*texts)),
                          frozenset(participants[key] for text in texts
                                    for key, _ in named_participants(
                                        index.view.scope_id, text, participants, index.speakers,
                                    )))

    def declaration_matches(self, slot: MemberSlotV1, declaration: DeclarationV1,
                            texts: Sequence[str]) -> bool:
        """An equal ``name_norm``, or both unnamed with descriptor-token Jaccard ≥ 0.5."""
        if declaration.name is not None:
            return name_norm(declaration.name) in {name_norm(text) for text in texts}
        described = slot.descriptor or slot.mention
        if slot.name is not None or described is None:
            return False
        first, second = set(token_seq(described)), set(token_seq(declaration.descriptor))
        union = first | second
        return bool(union) and len(first & second) / len(union) >= self.policy.declaration_jaccard

    def candidates(self, slot: MemberSlotV1, entity: EntityRecord | None,
                   names: _SlotNames) -> tuple[EntityCandidateV1, ...]:
        member, policy, index = self.member, self.policy, self.index
        chosen: dict[str, EntityCandidateV1] = {}
        if slot.binding == "nominated":
            nominee = EntityCandidateV1("nominated", entity, None if entity is not None
                                        else slot.mint_key, None)
            chosen[nominee.key] = nominee
        limit = len(chosen) + policy.entity_candidates

        def room(key: str) -> bool:
            return key not in chosen and len(chosen) < limit

        def existing(source: CandidateSource, entity_id: str) -> None:
            if room(entity_id):
                chosen[entity_id] = EntityCandidateV1(source, index.entities[entity_id], None, None)

        # The id is the last key: hits equal in commit, names and type otherwise keep
        # the frozenset's iteration order, which follows the process's hash seed.
        for entity_id in sorted(names.exact,
                                key=lambda key: (entity_order(index.entities[key]), key)):
            existing("exact_name", entity_id)
        for entity_id in sorted(names.speakers,
                                key=lambda key: (entity_order(index.entities[key]), key)):
            existing("speaker_name", entity_id)
        own_unit = member.position_key[:2]
        for declaration in sorted(member.declarations,
                                  key=lambda item: (item.order[0], item.order[1][1:],
                                                    int(item.local_key[1:]))):
            key = declaration.mint_key
            if index.view.minted(key) is not None or index.view.spent(key):
                continue
            if (declaration.owner_run_id, declaration.unit_no) != own_unit and room(key) and (
                self.declaration_matches(slot, declaration, names.texts)
            ):
                chosen[key] = EntityCandidateV1("declaration", None, key, declaration)
        scores = names.scores
        for entity_id in sorted(scores, key=lambda key: (
            -scores[key], entity_order(index.entities[key]),
        )):
            existing("name_bm25", entity_id)
        for target in self.heads.values():
            for anchor in (*target.anchors[:1], *sorted(
                target.anchors[1:], key=lambda anchor: entity_order(index.entities[anchor])
                if anchor in index.entities else (0, (), ""),
            )):
                record = index.entities.get(anchor)
                if record is not None and record.participant_key is None:
                    existing("third_party", anchor)
        return tuple(chosen.values())

    def catalog_complete(self, slot: MemberSlotV1, entity: EntityRecord | None,
                         names: _SlotNames, candidates: tuple[EntityCandidateV1, ...]) -> bool:
        """Decision Q2: every live entity whose type or names match the slot was issued.

        A live entity matches when its type equals the slot's type (casefolded),
        or one of its readable names matches the slot's texts by exact phrase or
        by a positive name BM25, or a text names it as a speaker. An open slot
        has no type, so every live entity could fit it.
        """
        issued = {candidate.key for candidate in candidates if candidate.entity is not None}
        wanted = entity.entity_type if entity is not None else slot.entity_type
        return all(
            entity_id in issued for entity_id, record in self.index.entities.items()
            if wanted is None or record.entity_type.casefold() == wanted.casefold()
            or entity_id in names.scores or entity_id in names.exact
            or entity_id in names.speakers
        )

    def slots(self) -> tuple[SlotNominationV1, ...]:
        result: list[SlotNominationV1] = []
        for slot in self.member.slots:
            entity = self.index.entity(slot.entity_id) if slot.entity_id is not None else None
            if not slot.judged:
                result.append(SlotNominationV1(slot, entity, (), None))
                continue
            names = self.slot_names(slot, entity)
            candidates = self.candidates(slot, entity, names)
            result.append(SlotNominationV1(
                slot, entity, candidates, self.catalog_complete(slot, entity, names, candidates),
            ))
        return tuple(result)

    # Route 9 ---------------------------------------------------------------------------------

    def sources(self) -> tuple[SourceMessageV1, ...]:
        view = self.index.view
        keys: dict[MessageKey, None] = {}
        for span in self.member.quotes:
            part = view.evidence(span.evidence_id)
            if part is None:
                raise AMLError("binding_mismatch", "A member quote cites Evidence of another scope")
            keys[MessageKey(part.session_id, part.request_id, part.ordinal)] = None
        cited = sorted((message for key in keys if (message := view.message(key)) is not None),
                       key=lambda message: (message.sigma, message.key.request_id,
                                            message.key.ordinal))
        budget = self.policy.context_messages
        requests: dict[tuple[str, str], list[MessageRecord]] = {}
        for message in cited:
            requests.setdefault((message.key.session_id, message.key.request_id), []).append(
                message)
        gaps = sorted(
            (right.key.ordinal - left.key.ordinal - 1, left.sigma,
             left.key.session_id, left.key.request_id, left.key.ordinal, left.key, right.key)
            for messages in requests.values()
            for ordered in [sorted(messages, key=lambda message: message.key.ordinal)]
            for left, right in zip(ordered, ordered[1:], strict=False)
            if 0 < right.key.ordinal - left.key.ordinal - 1 <= budget
        )
        context: list[MessageRecord] = []
        for size, _, _, _, _, left, right in gaps:
            if size > budget - len(context):
                continue
            bridge = [view.message(MessageKey(left.session_id, left.request_id, ordinal))
                      for ordinal in range(left.ordinal + 1, right.ordinal)]
            if all(message is not None for message in bridge):
                context.extend(message for message in bridge if message is not None)
        if cited and len(context) < budget:
            selected = set(keys) | {message.key for message in context}
            remaining = budget - len(context)
            previous = [message for message in view.messages_before(cited[0].key, budget)
                        if message.key not in selected]
            context.extend(previous[-remaining:])
        context.sort(key=lambda message: (message.sigma, message.key.request_id,
                                           message.key.ordinal))
        return (*(SourceMessageV1(message, True) for message in cited),
                *(SourceMessageV1(message, False) for message in context))

    # Result ----------------------------------------------------------------------------------

    def result(self) -> NominationV1:
        self.primary()
        self.secondary()
        self.scope_dense()
        self.revises()
        self.corrects()
        siblings = self.siblings()
        history = self.history()
        tombstones = self.tombstones()
        slots = self.slots()
        sources = self.sources()
        targets = (*self.heads.values(), *siblings, *self.retired.values())
        entities = self.entities(targets, tombstones, slots)
        facts = {entity_id: self.index.facts(entity_id, self.policy.entity_facts)
                 for entity_id in entities}
        member = self.member
        correction = bool(member.revises) or member.cue is not None
        complete = self.mode in ("complete", "empty_complete")
        return NominationV1(
            member=member, pin=self.index.view.pin, embedding_space=member.embedding_space,
            policy=self.policy,
            lexical=self.index.stats.restricted((*self.terms, *self.cue_terms)),
            readable_heads=self.index.readable_heads,
            routes=tuple(sorted(self.routes, key=lambda record: _ROUTE_ORDER[record.route])),
            targets=targets, history=history, tombstones=tombstones, slots=slots,
            entity_facts=facts, sources=sources,
            flags=NominationFlagsV1(
                correction_requested=correction,
                history_correction_requested=self.history_requested,
                entity_partition_complete=complete,
                external_only_allowed=correction and complete,
                revises_unavailable=self.unavailable,
            ),
            dependencies=self.dependencies(targets, history, tombstones, entities, facts,
                                           sources),
        )

    def entities(
        self, targets: Sequence[NominatedTargetV1], tombstones: Sequence[NominatedTombstoneV1],
        slots: Sequence[SlotNominationV1],
    ) -> dict[str, EntityRecord]:
        """Every existing entity the member input names, by id: slot entities, candidates,
        and the anchors of issued heads, siblings and tombstones."""
        found = {slot.entity.entity_id for slot in slots if slot.entity is not None}
        found |= {candidate.key for slot in slots for candidate in slot.candidates
                  if candidate.entity is not None}
        found |= {anchor for target in targets if target.kind != "retired"
                  for anchor in target.anchors}
        found |= {anchor for item in tombstones
                  for anchor in (item.record.entity_id, *item.record.about_entity_ids)}
        return {entity_id: record for entity_id in sorted(found)
                if (record := self.index.entity(entity_id)) is not None}

    def dependencies(
        self, targets: Sequence[NominatedTargetV1], history: Sequence[HistoryContextV1],
        tombstones: Sequence[NominatedTombstoneV1], entities: Mapping[str, EntityRecord],
        facts: Mapping[str, tuple[MemoryRecord, ...]], sources: Sequence[SourceMessageV1],
    ) -> NominationDependenciesV1:
        versions = [target.version for target in targets if target.version is not None]
        versions += [item.entry.version for item in history]
        versions += [fact.head for shown in facts.values() for fact in shown]
        evidence = {span.evidence_id for span in self.member.quotes}
        evidence |= {part.object_id for source in sources for part in source.message.parts}
        evidence |= {span.evidence_id for version in versions
                     for span in version.snapshot.quote_spans}
        evidence |= {evidence_id for target in targets if target.sibling is not None
                     for evidence_id in target.sibling.evidence_ids}
        return NominationDependenciesV1(
            evidence_ids=sorted(evidence),
            revision_ids=sorted({version.revision_id for version in versions}),
            tombstone_ids=sorted(item.record.object_id for item in tombstones),
            entity_ids=sorted(entities),
            naming_revision_ids=sorted({row.naming_revision_id for record in entities.values()
                                        for row in record.names}),
        )


def _named(version: VersionRecord) -> tuple[str, ...]:
    return tuple(dict.fromkeys(item.entity_id for item in version.snapshot.naming))


def _position(position_key: PositionKey) -> str:
    owner_run_id, unit_no, position = position_key
    return f"{owner_run_id}:{unit_no}:{position}"


# Fence scoring (I9b) --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RowScoreV1:
    """A row's scores against a member's query, as nomination computes them."""

    memory_id: str
    revision_id: str
    cosine: float
    bm25: float


def score_rows(
    member: MemberV1, rows: Sequence[VersionRecord], stats: SparseStatistics,
    policy: NominationPolicyV1, *, query: Literal["statement", "cue"] = "statement",
) -> tuple[RowScoreV1, ...]:
    """Score rows for one member with frozen corpus statistics (C.4.2; I9b).

    The commit fence scores rows committed after the pin with the statistics
    the member's snapshot recorded, so a would-be neighbour is ranked exactly
    as nomination ranked its pool (C.11.3).

    Args:
        member (MemberV1): The member.
        rows (Sequence[VersionRecord]): Heads or retired versions to score.
        stats (SparseStatistics): The pin's statistics; ``frequencies`` needs the
            query terms only.
        policy (NominationPolicyV1): The member's policy.
        query (Literal["statement", "cue"]): Which of the member's queries to score.
            Supported values:
            - ``statement``: the statement vector and statement and keyword tokens.
            - ``cue``: the ``corrects_text`` vector and tokens (route 4).

    Returns:
        tuple[RowScoreV1, ...]: One score per row, in row order.

    Raises:
        AMLError: ``embedding_space_mismatch`` for a row of another space;
            ``binding_mismatch`` for a cue query of a member without a cue.
    """
    if query == "cue":
        if member.cue is None:
            raise AMLError("binding_mismatch", "A member without corrects_text has no cue query")
        vector, tokens = member.cue.vector, sparse_terms(member.cue.text)
    else:
        vector, tokens = member.vector, member.tokens
    unit, terms = unit_vector(vector), _terms(tokens)
    scores: list[RowScoreV1] = []
    for row in rows:
        _same_space(row.snapshot.embedding_space, member.embedding_space)
        doc = _Doc.of_snapshot(row.memory_id, row.snapshot)
        scores.append(RowScoreV1(row.memory_id, row.revision_id, cosine(unit, doc.unit),
                                 _bm25(stats, terms, doc, policy)))
    return tuple(scores)


# Handles (C.4.4) ------------------------------------------------------------------------------


def judge_handle(bind: ContentBinder, key: HandleKey, *, chars: int = HANDLE_CHARS) -> str:
    """Return one judge handle: a prefix letter and lowercase base32 of a keyed binding (C.4.4).

    Args:
        bind (ContentBinder): The workspace's keyed content binder.
        key (HandleKey): The object's identity from `handle_key`.
        chars (int): 10, or 16 after a collision.

    Returns:
        str: ``prefix + base32(bind("judge-handle-v1", key))[:chars]`` in lowercase,
            which matches ``^[MHSTENCQ][a-z2-7]{10}([a-z2-7]{6})?$``.
    """
    digest = bind(HANDLE_PURPOSE, list(key))
    raw = base64.urlsafe_b64decode(digest + "=" * (-len(digest) % 4))
    return _PREFIX[key[0]] + base64.b32encode(raw).decode("ascii").lower()[:chars]


def handle_keys(nomination: NominationV1) -> tuple[HandleKey, ...]:
    """Return the identity of every handle one member input uses, each once.

    Args:
        nomination (NominationV1): The member's nomination.

    Returns:
        tuple[HandleKey, ...]: Targets, history entries, tombstones, the
            entities and new declarations the input names (slot entities,
            candidates, and the anchors of issued heads, siblings and
            tombstones, an unminted participant by its key), quotes and
            messages, in that order.
    """
    keys: dict[HandleKey, None] = {}
    keys.update(dict.fromkeys(target.handle_key for target in nomination.targets))
    keys.update(dict.fromkeys(item.handle_key for item in nomination.history))
    keys.update(dict.fromkeys(handle_key("forgotten", item.record.object_id)
                              for item in nomination.tombstones))
    keys.update(dict.fromkeys(handle_key("entity", entity_id)
                              for entity_id in nomination.dependencies.entity_ids))
    keys.update(dict.fromkeys(key for slot in nomination.member.slots
                              if (key := slot_key(slot)) is not None))
    keys.update(dict.fromkeys(candidate.handle_key for slot in nomination.slots
                              for candidate in slot.candidates))
    keys.update(dict.fromkeys(
        key for target in nomination.targets if target.sibling is not None
        for slot in target.sibling.slots
        if slot.entity_id is None and (key := slot_key(slot)) is not None
    ))
    keys.update(dict.fromkeys(
        handle_key("quote", span.evidence_id, f"{span.byte_start}-{span.byte_end}")
        for span in nomination.member.quotes
    ))
    keys.update(dict.fromkeys(source.handle_key for source in nomination.sources))
    return tuple(keys)


def assign_handles(nomination: NominationV1, bind: ContentBinder) -> dict[HandleKey, str]:
    """Give every object of one member input its judge handle (C.4.4).

    A handle depends on its object and the binder only, never on the batch.
    When two objects of the member input share a 10-character handle, both
    take 16 characters.

    Args:
        nomination (NominationV1): The member's nomination.
        bind (ContentBinder): The workspace's keyed content binder.

    Returns:
        dict[HandleKey, str]: The handle of every key of `handle_keys`.

    Raises:
        AMLError: ``binding_mismatch`` if two objects still share a handle at
            16 characters.
    """
    short = {key: judge_handle(bind, key) for key in handle_keys(nomination)}
    clashing = {handle for handle, count in Counter(short.values()).items() if count > 1}
    handles = {key: judge_handle(bind, key, chars=LONG_HANDLE_CHARS) if handle in clashing
               else handle for key, handle in short.items()}
    if len(set(handles.values())) != len(handles):
        raise AMLError("binding_mismatch", "Two objects of one member input share a handle")
    return handles


def _entity_handle(handles: Mapping[HandleKey, str], slot: MemberSlotV1) -> str | None:
    key = slot_key(slot)
    return handles[key] if key is not None and key[0] == "entity" else None


def term_key(bind: ContentBinder, term: str) -> str:
    """Return the keyed digest a snapshot records a BM25 term under (C.4.5)."""
    return bind(TERM_PURPOSE, term)


# NeighborSnapshotV1 (C.4.5) -------------------------------------------------------------------


def neighbor_snapshot(
    nomination: NominationV1, *, bind: ContentBinder, owner_run_id: str,
    member_input_digest: str, wake_digest: str | None = None,
    reentry_reason_code: ReasonCode | None = None,
) -> NeighborSnapshotV1:
    """Record one nomination as its content-free `NeighborSnapshotV1` (C.4.5).

    Args:
        nomination (NominationV1): The member's nomination.
        bind (ContentBinder): The workspace's keyed content binder, for handles
            and term digests.
        owner_run_id (str): The owner run that reviews the member.
        member_input_digest (str): ``bind_content("judge-v8-member-input",
            closure)`` of the member input built from this nomination (C.6.1).
        wake_digest (str | None): The member's wake key at this rebuild
            (`aml.ingestion.reentry.wake_digest`); None for a member that did not
            re-enter, or that round 2 built with a pick or issued targets.
        reentry_reason_code (ReasonCode | None): For a re-entered member, the
            candidate's code read at selection; None otherwise.

    Returns:
        NeighborSnapshotV1: The snapshot.

    Raises:
        ValidationError: If a wake key is given without its code.
    """
    handles = assign_handles(nomination, bind)
    member, stats, pin = nomination.member, nomination.lexical, nomination.pin

    def candidate(item: EntityCandidateV1) -> SnapshotCandidateV1:
        return SnapshotCandidateV1(
            handle=handles[item.handle_key],
            entity_id=item.entity.entity_id if item.entity is not None else None,
            mint_key=item.mint_key,
            commit_seq=item.entity.commit_seq if item.entity is not None else None,
            source=item.source,
        )

    heads = {target.object_id: handles[target.handle_key] for target in nomination.targets
             if target.kind == "head"}
    return NeighborSnapshotV1(
        schema_version=1, policy_digest=nomination.policy_digest, owner_run_id=owner_run_id,
        round=member.round, position=PositionV1.of(member.position_key),
        member_input_digest=member_input_digest, wake_digest=wake_digest,
        reentry_reason_code=reentry_reason_code,
        pin=SnapshotPinV1(
            commit_seq=pin.commit_seq, erase_epoch=pin.erase_epoch,
            consent_epoch=pin.consent_epoch, tombstone_seq=pin.tombstone_seq,
            recall_snapshot_version=pin.recall_snapshot_version,
        ),
        embedding_space=nomination.embedding_space,
        lexical=LexicalRecordV1(
            doc_count=stats.count, total_length=stats.length,
            df={term_key(bind, term): count for term, count in stats.frequencies.items()},
        ),
        routes=list(nomination.routes),
        targets=[SnapshotTargetV1(
            handle=handles[target.handle_key], kind=target.kind, object_id=target.object_id,
            revision_id=target.revision_id, statement_digest=target.statement_digest,
            sigma=list(target.sigma), primary_entity_id=target.primary,
            naming=list(target.naming),
            route=target.route,
        ) for target in nomination.targets],
        history=[SnapshotHistoryV1(
            handle=handles[item.handle_key], head_handle=heads[item.head_memory_id],
            of_handle=heads[item.head_memory_id]
            if item.entry.version.memory_id == item.head_memory_id else None,
            revision_id=item.entry.version.revision_id, ended_by=item.entry.ended_by,
        ) for item in nomination.history],
        tombstones=[SnapshotTombstoneV1(
            handle=handles[handle_key("forgotten", item.record.object_id)],
            tombstone_id=item.record.object_id, route=item.route,
        ) for item in nomination.tombstones],
        entities=[SnapshotSlotV1(
            slot=item.slot.slot, binding=item.slot.binding,
            entity_handle=_entity_handle(handles, item.slot),
            nominated_handle=handles[item.candidates[0].handle_key]
            if item.slot.binding == "nominated" else None,
            candidates=[candidate(entry) for entry in item.candidates],
            catalog_complete=item.catalog_complete,
        ) for item in nomination.slots],
        entity_facts=[SnapshotEntityFactsV1(
            handle=handles[handle_key("entity", entity_id)], entity_id=entity_id,
            facts=[fact.head.revision_id for fact in facts],
        ) for entity_id, facts in nomination.entity_facts.items()],
        quotes=[SnapshotQuoteV1(
            handle=handles[handle_key("quote", span.evidence_id,
                                      f"{span.byte_start}-{span.byte_end}")],
            evidence_id=span.evidence_id, byte_start=span.byte_start, byte_end=span.byte_end,
        ) for span in member.quotes],
        sources=[SnapshotSourceV1(
            handle=handles[source.handle_key],
            evidence_ids=[part.object_id for part in source.message.parts], cited=source.cited,
        ) for source in nomination.sources],
        flags=nomination.flags, dependencies=nomination.dependencies,
    )
