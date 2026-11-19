"""Prepared per-scope corpora of a RecallSnapshot root and entity retrieval (A.6.1, A.6.5; I7).

A RecallSnapshot root (S1a) holds a scope's projections; this module prepares
what Search, the catalog and nomination read from them, and the one
entity-retrieval implementation they share:

- the sparse-term function `sparse_terms` and the keyword-form function
  `keyword_forms` (C6, read-Q8): document and query sides call these same
  function objects;
- `EntityRetrieval`, A.6.5 over one name view: exact phrase (a readable name,
  indexed whole, token-contained in the text), BM25 over each entity's readable
  names, the casefolded type filter, the participant point lookup and the
  Search expansion bound. `EntityRetrieval.of_view` builds the current name
  view of an Add read view, `EntityRetrieval.of_versions` the as-of view of a
  mode's eligible versions (read-Q9);
- `MemoryCorpus`: every Item of a scope with its head decoded once, its unit
  vector, sparse documents and keyword forms, the anchor index, the
  incoming-update index with each source Item's lifecycle and head reason, and
  the corrected-version map built from every historical Item's
  ``corrected_targets``, erased correcting heads included (D-14, T-S7a-1). It
  serves `aml.core.readability.WorldIndex`, so S1d's world functions read it;
- `EntityCorpus` and `EvidenceCorpus`: the scope's current name view and the
  bench Evidence documents that pass P-EVR;
- `SnapshotRecallCache`: at most 4 corpora and 64 MiB per published root,
  carried to the next root only while the delta left their inputs unchanged;
  `built_corpus` keeps the few corpora above that bound for the very root they
  were built at, so a long user's Search and its delivery checks build each once;
- `memory_text_parts` and `evidence_parts`: each head's lexical document and
  keyword forms and each Evidence part's unit vector and lexical document,
  computed once per snapshot or row object and shared by every corpus built
  from it (SEARCH-CACHE);
- `cold_build`: C.13 step 17's build of every corpus of a cold root.

Corpora are never eligibility or authority. Their statistics count the
documents that pass D.2 at `STATISTICS_CLOCK`, so clock-only expiry still
counts until its erase commits (D.6) while withheld, erased, unindexed, held
and forgotten content never does; every reader applies its mode's predicates
at its own clock. A corpus holds only its own scope's values and never a root,
so a carried corpus retains no other scope's pre-image. Tombstones never enter
a corpus (A.3.8).
"""

from __future__ import annotations

import math
import threading
import unicodedata
from collections import Counter, OrderedDict
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import Future, wait
from dataclasses import dataclass
from functools import cached_property
from itertools import pairwise
from types import MappingProxyType
from typing import Final, Literal, Protocol, cast
from weakref import ReferenceType, ref

from aml.core.add_read_view import AddReadView, EntityRecord
from aml.core.category_terms import category_terms
from aml.core.entities import (
    USER,
    MemoryLifecycle,
    MemorySnapshotV2,
    NameKind,
    QuoteSpanV1,
    RevisionReason,
)
from aml.core.errors import AMLError
from aml.core.gc_safe import CheckedLock, HeldIdentityMemo, IdentityMemo
from aml.core.readability import CorrectionRef, IncomingLink, Readability, WorldMemory
from aml.core.recall_snapshot import (
    EntityNameRowV1,
    EntityRowV1,
    MemoryRowV2,
    RecallSnapshot,
    ScopeFence,
    SourceRowV2,
)
from aml.core.semantic import ScopeKind, TimeGranularity
from aml.core.snapshot_memo import decoded_snapshot, snapshot_unit
from aml.core.statement import (
    is_name,
    name_norm,
    own_token,
    participant_key,
    token_seq,
    token_spans,
)
from aml.core.vector_score import UnitRows, UnitVector, centroid_typicality, unit_vector

CorpusKind = Literal["memory", "entity", "evidence"]
"""The three per-scope corpora of A.6.1."""

BM25_K1: Final = 1.2
"""BM25 term-frequency saturation of every sparse index (A.6.3, A.6.5)."""

BM25_B: Final = 0.75
"""BM25 length normalisation of every sparse index (A.6.3, A.6.5)."""

MAX_CACHE_WEIGHT: Final = 64 * 1024 * 1024
"""The estimated bytes a `SnapshotRecallCache` retains at most (A.6.1)."""

MAX_CACHED_CORPORA: Final = 4
"""The corpora a `SnapshotRecallCache` retains at most (A.6.1)."""

STATISTICS_CLOCK: Final = -1
"""The clock at which corpora apply D.2 to choose their statistics populations.

It precedes every ``SystemTimestamp``, so no Evidence is expired at it: a
corpus is a function of its root alone, and clock-only expiry counts in its
statistics until the erase commits, as D.6 allows. Withheld, erased, unindexed,
held and forgotten content fails at every clock and never counts.
"""

_FINEST_FIRST: Final[Mapping[TimeGranularity, int]] = MappingProxyType({
    "HOUR": 0, "DAY": 1, "MONTH": 2, "YEAR": 3, "unanchored": 4,
})
_NAME_KIND_ORDER: Final[Mapping[NameKind, int]] = MappingProxyType(
    {"name": 0, "alias": 1, "former": 2},
)


def sparse_terms(text: str) -> tuple[str, ...]:
    """Return the terms of one sparse document or query text (A.6.3, A.6.5; C6, read-Q8, ML-1).

    The terms are the `token_seq` unigrams, unchanged and in text order,
    followed in text order by the bigram of every two tokens of their own
    (`own_token`) that are adjacent in the folded text (``statement.folded``): two Han, Kana
    or Hangul characters (C6), or two grapheme clusters of an unspaced
    Southeast Asian script such as Thai (ML-1). A bigram never crosses a
    space, punctuation or any other character. Search's lexical
    route, the bench Evidence route, name BM25 and Add-time route 8 call this
    one function on both their document and query sides. Exact phrase stays
    token containment. Context selection (B.4.3), nomination's memory BM25
    (C.4.2) and commit-fence scoring (C.11.3) use these same sparse terms;
    ``memory_tokens`` and statement/name identity keep unigrams.

    Args:
        text (str): A statement with its keywords, an Evidence part, a name or a
            query.

    Returns:
        tuple[str, ...]: The unigrams, then the bigrams; empty when the text has
            no token.

    Examples:
        ```python
        sparse_terms("王小明 likes 北京")
        # ('王', '小', '明', 'likes', '北', '京', '王小', '小明', '北京')
        sparse_terms("ชื่อสมศรี")
        # ('ชื่', 'อ', 'ส', 'ม', 'ศ', 'รี', 'ชื่อ', 'อส', 'สม', 'มศ', 'ศรี')
        ```
    """
    spans = token_spans(text)
    bigrams = (
        first + second for (first, _, end), (second, start, _) in pairwise(spans)
        if end == start and own_token(first) and own_token(second)
    )
    return (*(token for token, _, _ in spans), *bigrams)


def query_terms(*texts: str) -> tuple[str, ...]:
    """Return the distinct sparse terms of query texts, sorted (A.6.3, A.6.5).

    Each text contributes its own `sparse_terms`, so no bigram spans two texts.

    Args:
        *texts (str): The query texts, such as a Search query or a slot's
            mention, name and naming names.

    Returns:
        tuple[str, ...]: The union of their terms in sorted order.
    """
    return tuple(sorted({term for text in texts for term in sparse_terms(text)}))


def keyword_forms(keyword: str) -> frozenset[str]:
    """Return the folded forms the trigger route matches for one keyword or term (A.6.3).

    The form of a keyword is its NFKC-casefolded text; an ASCII alphabetic
    keyword of four or more characters also folds its regular English plural:
    ``-ies`` to ``-y``, ``-es`` after ``s``, ``x``, ``z``, ``ch`` or ``sh``,
    and a final ``-s`` not after ``s``. The memory corpus stores the forms of
    each memory's keywords; Search's trigger route applies this same function
    to each `sparse_terms` term of the query and nominates the memories whose
    stored forms meet the query's (read-Q8).

    Args:
        keyword (str): A stored keyword or one query term.

    Returns:
        frozenset[str]: Its forms.

    Examples:
        ```python
        keyword_forms("Cherries")  # frozenset({'cherries', 'cherry'})
        keyword_forms("北京")       # frozenset({'北京'})
        ```
    """
    normalized = unicodedata.normalize("NFKC", keyword).casefold()
    if len(normalized) < 4 or not normalized.isascii() or not normalized.isalpha():
        return frozenset({normalized})
    if normalized.endswith("ies"):
        return frozenset({normalized, normalized[:-3] + "y"})
    if normalized.endswith("es") and normalized[:-2].endswith(("s", "x", "z", "ch", "sh")):
        return frozenset({normalized, normalized[:-2]})
    if normalized.endswith("s") and not normalized.endswith("ss"):
        return frozenset({normalized, normalized[:-1]})
    return frozenset({normalized})


@dataclass(frozen=True, slots=True)
class SparseDocument:
    """The term frequencies and length of one sparse document.

    Attributes:
        counts (Mapping[str, int]): How often each term occurs.
        length (int): The number of terms, repeats included.
    """

    counts: Mapping[str, int]
    length: int

    @classmethod
    def of(cls, terms: Sequence[str]) -> SparseDocument:
        """Count one document's terms.

        Args:
            terms (Sequence[str]): The document's terms, such as `sparse_terms`.

        Returns:
            SparseDocument: Its frequencies and length.
        """
        return cls(MappingProxyType(dict(Counter(terms))), len(terms))

    @classmethod
    def annotated(cls, terms: Sequence[str], annotations: Sequence[str]) -> SparseDocument:
        """Count one document's terms plus annotation terms that match but add no length.

        An annotation, such as a `category_terms` word, is counted like a term
        but leaves the document's length, and so BM25's length normalization
        and the corpus's average length, those of its own text.

        Args:
            terms (Sequence[str]): The document's terms, such as `sparse_terms`.
            annotations (Sequence[str]): The annotation terms.

        Returns:
            SparseDocument: The frequencies of both and the length of ``terms``.
        """
        return cls(MappingProxyType(dict(Counter((*terms, *annotations)))), len(terms))


def statement_document(snapshot: MemorySnapshotV2) -> SparseDocument:
    """Return the sparse document of a version's statement and keywords, once per snapshot object.

    Context selection and nomination score every head of the scope at each pin
    with BM25 over ``sparse_terms`` of its statement and keywords (B.4.3,
    C.4.2), so every Add would tokenize the user's whole history again. A
    snapshot is immutable and the decoded snapshot of an unchanged head is the
    same object across pins (``decoded_snapshot``), so its document is kept
    for as long as that object lives.

    Args:
        snapshot (MemorySnapshotV2): A content-bearing snapshot.

    Returns:
        SparseDocument: ``SparseDocument.of(sparse_terms(statement and keywords))``.
    """
    document = _statement_documents.recall(snapshot)
    if document is None:
        document = SparseDocument.of(sparse_terms(
            " ".join((snapshot.statement, *snapshot.keywords)),
        ))
        _statement_documents.remember(snapshot, document)
    return document


_statement_documents: IdentityMemo[MemorySnapshotV2, SparseDocument] = IdentityMemo()


@dataclass(frozen=True, slots=True)
class SparseStatistics:
    """BM25 corpus statistics: documents, total length and document frequencies.

    `bm25` is the one float64 BM25 of name retrieval (A.6.5), context selection
    (B.4.3) and nomination (C.4.2). Each caller keeps its own terms and
    population: names score their `sparse_terms` over the name view; context
    selection and nomination score `sparse_terms` over the scope's P-MEM
    heads at the pin. Search's memory lexical route keeps its Decimal
    arithmetic (A.6.3).

    Attributes:
        count (int): The documents of the population.
        length (int): Their terms in all.
        frequencies (Mapping[str, int]): The documents holding each term; an
            absent term has frequency 0.
    """

    count: int
    length: int
    frequencies: Mapping[str, int]

    @classmethod
    def of(cls, documents: Iterable[SparseDocument]) -> SparseStatistics:
        """Count a population of documents.

        Args:
            documents (Iterable[SparseDocument]): Every document of the population.

        Returns:
            SparseStatistics: Their count, total length and term frequencies.
        """
        frequencies: Counter[str] = Counter()
        count = length = 0
        for document in documents:
            count += 1
            length += document.length
            frequencies.update(document.counts.keys())
        return cls(count, length, MappingProxyType(dict(frequencies)))

    def restricted(self, terms: Iterable[str]) -> SparseStatistics:
        """Keep the frequencies of some terms only, as a nomination snapshot records them (C.4.5).

        Args:
            terms (Iterable[str]): The terms to keep, typically a member's query terms.

        Returns:
            SparseStatistics: The same count and length with ``frequencies``
                limited to ``terms``, in sorted order.
        """
        return SparseStatistics(self.count, self.length, MappingProxyType({
            term: self.frequencies.get(term, 0) for term in sorted(set(terms))
        }))

    def bm25(
        self, terms: Iterable[str], document: SparseDocument, *,
        k1: float = BM25_K1, b: float = BM25_B,
    ) -> float:
        """Score one document for distinct query terms (C.4.2).

        The sum ``Σ idf·tf·(k1+1) / (tf + k1·(1 − b + b·length/avgdl))`` with
        ``idf = ln(1 + (N − df + 0.5)/(df + 0.5))`` is taken exactly and
        rounded half-even to 6 decimals, so a score is the same in every process.

        Args:
            terms (Iterable[str]): The distinct query terms.
            document (SparseDocument): The scored document.
            k1 (float): Term-frequency saturation; nomination passes its policy's.
            b (float): Length normalisation; nomination passes its policy's.

        Returns:
            float: The score; 0.0 over an empty population or without a shared
                term.
        """
        if not self.count or not self.length:
            return 0.0
        average = self.length / self.count
        # Only shared terms contribute, and ``math.fsum`` is exactly rounded, so
        # the order the terms are visited in cannot change the score: visit the
        # smaller side. Context selection and nomination score every head of a
        # long history against a unit's hundreds of query terms.
        wanted = terms if isinstance(terms, set | frozenset) else set(terms)
        counts = document.counts
        shared = (
            [term for term in counts if term in wanted] if len(counts) < len(wanted)
            else [term for term in wanted if term in counts]
        )
        parts: list[float] = []
        for term in shared:
            frequency = counts[term]
            if frequency:
                frequent = self.frequencies.get(term, 0)
                idf = math.log(1.0 + (self.count - frequent + 0.5) / (frequent + 0.5))
                parts.append(idf * frequency * (k1 + 1.0) / (
                    frequency + k1 * (1.0 - b + b * document.length / average)
                ))
        return round(math.fsum(parts), 6) + 0.0


@dataclass(frozen=True, slots=True)
class NameEntry:
    """One readable name of an entity, indexed whole (A.3.7, A.6.5; read-Q8).

    Attributes:
        entity_id (str): The named `EntityRef`.
        name (str): The name as written.
        name_norm (str): Its ``name_norm``.
        name_kind (NameKind): ``name``, ``alias`` or ``former``.
        tokens (tuple[str, ...]): Its ``token_seq``, of `bound_length` 1..24: the
            whole name, never truncated, which exact phrase matches.
        terms (tuple[str, ...]): Its `sparse_terms`, which name BM25 scores.
    """

    entity_id: str
    name: str
    name_norm: str
    name_kind: NameKind
    tokens: tuple[str, ...]
    terms: tuple[str, ...]

    @classmethod
    def of(cls, entity_id: str, name: str, name_kind: NameKind) -> NameEntry:
        """Index one name of an entity.

        Args:
            entity_id (str): The named `EntityRef`.
            name (str): A valid A0.6 name.
            name_kind (NameKind): Its kind.

        Returns:
            NameEntry: The name with its tokens and sparse terms.

        Raises:
            ValueError: If the name is not a valid name (A0.6, ``is_name``).
        """
        if not is_name(name):
            raise ValueError("An indexed name is a valid A0.6 name")
        tokens = tuple(token_seq(name))
        return cls(entity_id, name, " ".join(tokens), name_kind, tokens, sparse_terms(name))


@dataclass(frozen=True, slots=True)
class EntityEntry:
    """An active entity of one name view with its readable names (A.3.6, A.6.5).

    Attributes:
        entity_id (str): The `EntityRef`; ``"user"`` for the virtual user.
        entity_type (str): Its stored type; ``"user"`` for the virtual user.
        participant_key (str | None): A participant's key, else None.
        names (tuple[NameEntry, ...]): Its readable names in this view.
    """

    entity_id: str
    entity_type: str
    participant_key: str | None
    names: tuple[NameEntry, ...]


@dataclass(frozen=True, slots=True)
class NameExpansion:
    """What the readable names in a Search query nominate and add (A.6.5).

    Attributes:
        matched (frozenset[str]): Every entity a readable name in the query
            names; each one is nominated.
        expanding (frozenset[str]): The matched entities named by a matched name
            that at most ``max_entities`` entities bear.
        names (tuple[NameEntry, ...]): The other readable names of the expanding
            entities, whose terms the query gains.
    """

    matched: frozenset[str]
    expanding: frozenset[str]
    names: tuple[NameEntry, ...]


def _entry(record: EntityRecord) -> EntityEntry:
    return EntityEntry(
        record.entity_id, record.entity_type, record.participant_key,
        tuple(NameEntry.of(record.entity_id, row.name, row.name_kind) for row in record.names),
    )


class EntityRetrieval:
    """A.6.5 entity retrieval over one name view, shared by the catalog, nomination and Search.

    A name view is the active entities of one exact scope with the names that
    are readable in the caller's mode: the current view (`of_view`) for
    current, plain, ``auto``/``core_hook`` and Add-time callers, an as-of view
    (`of_versions`) for world(t) and knowledge(t) (read-Q9). Every name is
    indexed whole, so exact phrase holds for every readable name of up to 24
    tokens. BM25 scores each entity's document, the `sparse_terms` of its
    readable names, against statistics over the entities that have a readable
    name.

    Args:
        entities (Iterable[EntityEntry]): The view's entities, each listed once.

    Raises:
        ValueError: If an entity or a participant key is listed twice, or a name
            names another entity.

    Examples:
        ```python
        retrieval = EntityRetrieval.of_view(view, include_user=False)
        named = retrieval.exact("I met Hazel today")
        scores = retrieval.scores(query_terms(slot.mention, slot.name))
        ```
    """

    def __init__(self, entities: Iterable[EntityEntry]) -> None:
        """Index the view; see the class documentation."""
        ordered = sorted(entities, key=lambda entry: entry.entity_id)
        self._entities: Mapping[str, EntityEntry] = MappingProxyType({
            entry.entity_id: entry for entry in ordered
        })
        if len(self._entities) != len(ordered):
            raise ValueError("A name view lists each entity once")
        starts: dict[str, list[NameEntry]] = {}
        owners: dict[str, set[str]] = {}
        participants: dict[str, str] = {}
        documents: dict[str, SparseDocument] = {}
        postings: dict[str, list[str]] = {}
        for entry in ordered:
            if entry.participant_key is not None:
                if entry.participant_key in participants:
                    raise ValueError("A participant key names one entity")
                participants[entry.participant_key] = entry.entity_id
            for name in entry.names:
                if name.entity_id != entry.entity_id:
                    raise ValueError("A name view lists names under their own entity")
                starts.setdefault(name.tokens[0], []).append(name)
                owners.setdefault(name.name_norm, set()).add(entry.entity_id)
            terms = [term for name in entry.names for term in name.terms]
            if terms:
                document = SparseDocument.of(terms)
                documents[entry.entity_id] = document
                for term in document.counts:
                    postings.setdefault(term, []).append(entry.entity_id)
        self._starts: Mapping[str, tuple[NameEntry, ...]] = {
            token: tuple(names) for token, names in starts.items()
        }
        self._owners: Mapping[str, frozenset[str]] = {
            norm: frozenset(ids) for norm, ids in owners.items()
        }
        self._participants: Mapping[str, str] = participants
        self._documents: Mapping[str, SparseDocument] = documents
        self._postings: Mapping[str, tuple[str, ...]] = {
            term: tuple(ids) for term, ids in postings.items()
        }
        self.statistics: SparseStatistics = SparseStatistics.of(documents.values())
        """Name BM25 statistics over the entities that have a readable name."""

    @classmethod
    def of_view(cls, view: AddReadView, *, include_user: bool) -> EntityRetrieval:
        """Return the current name view of an Add read view's pin (A.6.5, read-Q9).

        Args:
            view (AddReadView): The pin; it lists active entities with their
                P-NAME names at its clock.
            include_user (bool): Whether the virtual user is part of the view.
                Catalog route 4 and route 8 list the user through their own
                routes and pass False.

        Returns:
            EntityRetrieval: The view's active entities and readable names.
        """
        records = list(view.entities())
        user = view.entity(USER) if include_user else None
        if user is not None:
            records.append(user)
        return cls(_entry(record) for record in records)

    @classmethod
    def of_versions(
        cls,
        versions: Iterable[MemorySnapshotV2],
        entities: Iterable[EntityRowV1],
        *,
        include_user: bool,
    ) -> EntityRetrieval:
        """Return the as-of name view of a mode's eligible versions (A.6.4, read-Q9).

        world(t) and knowledge(t) name entities from their eligible memories:
        each shown version's ``naming`` annotations, kept for the active
        entities of the exact scope. Their eligibility already applies P-VER
        and leaves out forgotten Items, so it is the as-of P-NAME. This view is
        built per Search and is never cached.

        Args:
            versions (Iterable[MemorySnapshotV2]): The shown version of every
                eligible memory: eligible heads and Corrected-history stand-ins
                in world(t), the versions the knowledge reader returns in
                knowledge(t).
            entities (Iterable[EntityRowV1]): The scope's active entities.
            include_user (bool): Whether the virtual user is part of the view.

        Returns:
            EntityRetrieval: The active entities with the names their eligible
                versions give them, one name per annotation.
        """
        active = {row.object_id: row for row in entities}
        names: dict[str, list[NameEntry]] = {entity_id: [] for entity_id in active}
        if include_user:
            names[USER] = []
        for version in versions:
            for annotation in version.naming:
                if annotation.entity_id in names:
                    names[annotation.entity_id].append(NameEntry.of(
                        annotation.entity_id, annotation.name, annotation.name_kind,
                    ))
        return cls(
            EntityEntry(
                entity_id,
                "user" if entity_id == USER else active[entity_id].entity_type,
                None if entity_id == USER else active[entity_id].participant_key,
                tuple(sorted(found, key=lambda name: (
                    _NAME_KIND_ORDER[name.name_kind], name.name_norm, name.name,
                ))),
            )
            for entity_id, found in names.items()
        )

    def entity(self, entity_id: str) -> EntityEntry | None:
        """Return one entity of the view, or None when it is not in the view."""
        return self._entities.get(entity_id)

    def participant(self, participant_key: str) -> EntityEntry | None:
        """Return the participant with a key: the point lookup on ``participant_key``."""
        entity_id = self._participants.get(participant_key)
        return None if entity_id is None else self._entities[entity_id]

    def entities(self, *, entity_type: str | None = None) -> tuple[EntityEntry, ...]:
        """Return the view's entities, optionally of one casefolded type, by id.

        Args:
            entity_type (str | None): Keep only entities whose type equals this
                one after casefolding; None keeps all.

        Returns:
            tuple[EntityEntry, ...]: The entities, sorted by ``entity_id``.
        """
        return tuple(
            entry for entry in self._entities.values() if self._typed(entry, entity_type)
        )

    def exact_names(self, text: str, *, entity_type: str | None = None) -> tuple[NameEntry, ...]:
        """Return the readable names token-contained in a text (A.6.5 exact phrase).

        A name matches when its whole ``token_seq`` is a contiguous run of the
        text's tokens, the A0.6 token containment; a partial token never
        matches, and a Chinese name matches inside Chinese text because each
        Han character is a token.

        Args:
            text (str): The text, such as one readable stretch of a source, a
                slot's mention or a Search query.
            entity_type (str | None): Keep only names of entities of this
                casefolded type.

        Returns:
            tuple[NameEntry, ...]: The distinct matching names, sorted by
                ``name_norm``, entity, kind and name.
        """
        tokens = token_seq(text)
        found: dict[NameEntry, None] = {}
        for index, token in enumerate(tokens):
            for name in self._starts.get(token, ()):
                if tuple(tokens[index:index + len(name.tokens)]) == name.tokens and (
                    self._typed(self._entities[name.entity_id], entity_type)
                ):
                    found[name] = None
        return tuple(sorted(found, key=lambda name: (
            name.name_norm, name.entity_id, _NAME_KIND_ORDER[name.name_kind], name.name,
        )))

    def exact(self, text: str, *, entity_type: str | None = None) -> frozenset[str]:
        """Return the entities with a readable name token-contained in a text (A.6.5).

        Args:
            text (str): The text.
            entity_type (str | None): Keep only entities of this casefolded type.

        Returns:
            frozenset[str]: Their ids.
        """
        return frozenset(
            name.entity_id for name in self.exact_names(text, entity_type=entity_type)
        )

    def scores(
        self, terms: Iterable[str], *, entity_type: str | None = None,
    ) -> dict[str, float]:
        """Return the positive BM25 scores of the view's entities for query terms (A.6.5).

        Args:
            terms (Iterable[str]): The query's terms, normally `query_terms` of
                its texts.
            entity_type (str | None): Score only entities of this casefolded
                type; the statistics stay those of the whole view.

        Returns:
            dict[str, float]: Each entity scoring above zero, by id.
        """
        query = sorted(set(terms))
        candidates = sorted({
            entity_id for term in query for entity_id in self._postings.get(term, ())
        })
        result: dict[str, float] = {}
        for entity_id in candidates:
            if self._typed(self._entities[entity_id], entity_type):
                score = self.statistics.bm25(query, self._documents[entity_id])
                if score > 0:
                    result[entity_id] = score
        return result

    def expansion(self, text: str, *, max_entities: int) -> NameExpansion:
        """Return what the readable names in a Search query nominate and add (A.6.5).

        Every entity a contained name names is nominated. A contained name that
        more than ``max_entities`` entities bear nominates them without
        expanding; otherwise the query gains its entities' other readable names.

        Args:
            text (str): The query.
            max_entities (int): The policy's ``entity_expansion_max_entities``.

        Returns:
            NameExpansion: The nominated entities, the expanding ones and the
                names that expand the query.

        Raises:
            ValueError: If ``max_entities`` is negative.
        """
        if max_entities < 0:
            raise ValueError("max_entities is 0 or more")
        names = self.exact_names(text)
        norms = {name.name_norm for name in names}
        expanding = frozenset(
            entity_id for norm in norms if len(self._owners[norm]) <= max_entities
            for entity_id in self._owners[norm]
        )
        added = dict.fromkeys(
            name for entity_id in sorted(expanding) for name in self._entities[entity_id].names
            if name.name_norm not in norms
        )
        return NameExpansion(
            frozenset(name.entity_id for name in names), expanding,
            tuple(sorted(added, key=lambda name: (
                name.name_norm, name.entity_id, _NAME_KIND_ORDER[name.name_kind], name.name,
            ))),
        )

    @staticmethod
    def _typed(entry: EntityEntry, entity_type: str | None) -> bool:
        return entity_type is None or entry.entity_type.casefold() == entity_type.casefold()


def _closed(fence: ScopeFence) -> AMLError | None:
    """The refusal of a scope whose fence closes its content, as ``source_corpus`` refuses it."""
    if fence.tomb:
        return AMLError("scope_denied", "Snapshot scope is tombstoned")
    if (
        fence.visibility_barrier or fence.active_erasure or fence.maintenance
        or fence.finalize_barrier is not None
    ):
        return AMLError("index_rebuilding", "Snapshot scope has an active visibility fence")
    return None


def _open_fence(root: RecallSnapshot, scope_id: str) -> ScopeFence:
    """Return the scope's fence, refusing an unknown scope or one whose fence closes it."""
    fence = root.scope_fence(scope_id)
    refusal = _closed(fence)
    if refusal is not None:
        raise refusal
    return fence


def _statistics_read(root: RecallSnapshot, fence: ScopeFence) -> Readability:
    return Readability(fence.scope_kind, fence.scope_id, STATISTICS_CLOCK, root.get_source)


def memory_text_parts(snapshot: MemorySnapshotV2) -> tuple[SparseDocument, frozenset[str]]:
    """Return a head's lexical document and keyword forms, computed once per snapshot object.

    Every Search builds the memory corpus of its scope, and a corpus too large
    for its root's cache (`MAX_CACHE_WEIGHT`) is built again by every Search
    and every delivery check: a user of 59k Items tokenized all of them three
    times per Search. Both values are pure functions of the immutable snapshot,
    and the decoded snapshot of an unchanged head is the same object across
    builds (``decoded_snapshot``), so they are kept for as long as that object
    lives (``IdentityMemo``); a snapshot decoded again computes them again.

    Args:
        snapshot (MemorySnapshotV2): A head snapshot.

    Returns:
        tuple[SparseDocument, frozenset[str]]: ``SparseDocument.annotated`` of the
            `sparse_terms` of its statement and keywords and of its keywords'
            `category_terms`, and the `keyword_forms` of its keywords and those
            categories: `MemoryDocument`'s ``sparse`` and ``keyword_forms``. Both
            are immutable and shared by every reader.
    """
    parts = _memory_text_parts.recall(snapshot)
    if parts is None:
        categories = category_terms(snapshot.keywords)
        parts = (
            SparseDocument.annotated(
                sparse_terms(" ".join((snapshot.statement, *snapshot.keywords))),
                sparse_terms(" ".join(categories)),
            ),
            frozenset(
                form for keyword in (*snapshot.keywords, *categories)
                for form in keyword_forms(keyword)
            ),
        )
        _memory_text_parts.remember(snapshot, parts)
    return parts


_memory_text_parts: IdentityMemo[MemorySnapshotV2, tuple[SparseDocument, frozenset[str]]] = (
    IdentityMemo()
)


EVIDENCE_MEMO_SIZE: Final = 131072
"""Evidence parts whose unit vector and lexical document a process keeps (`evidence_parts`).

Each entry holds its part's row (its text and embedding, shared with the root
while the part is current), a packed unit vector of 8 KiB at 1024 dimensions
and the part's term counts.
"""


def evidence_parts(row: SourceRowV2) -> tuple[UnitVector, SparseDocument]:
    """Return an Evidence part's unit vector and lexical document, computed once per row object.

    The bench Evidence corpus of a long user (18k parts in a
    deployment run) outweighs its root's cache, so every Search normalized and
    tokenized every part again. Both values are pure functions of the
    immutable row's ``embedding`` and ``text``. A root keeps an unchanged
    part's row object from root to root, so the memo keys on that object and
    holds it (``HeldIdentityMemo``): no other row can take its identity while
    the entry lives. A part replaced by a new row object computes them again.

    Args:
        row (SourceRowV2): An Evidence part.

    Returns:
        tuple[UnitVector, SparseDocument]: ``unit_vector(row.embedding)`` in
            immutable packed storage and ``SparseDocument.of(sparse_terms(row.text))``.

    Raises:
        AMLError: ``binding_mismatch`` for a zero or non-finite embedding, as
            ``unit_vector`` raises; nothing is kept then.
    """
    parts = _evidence_parts.recall(row)
    if parts is None:
        parts = unit_vector(row.embedding), SparseDocument.of(sparse_terms(row.text))
        _evidence_parts.remember(row, parts)
    return parts


_evidence_parts: HeldIdentityMemo[SourceRowV2, tuple[UnitVector, SparseDocument]] = (
    HeldIdentityMemo(EVIDENCE_MEMO_SIZE)
)


def clear_corpus_memos() -> None:
    """Forget every kept lexical document, Evidence vector and oversized corpus (tests)."""
    _memory_text_parts.clear()
    _evidence_parts.clear()
    _statement_documents.clear()
    drop_built_corpora()


class _HasVector(Protocol):
    @property
    def vector(self) -> UnitVector | None: ...


@dataclass(frozen=True, slots=True, eq=False)
class CorpusVectors:
    """A corpus's unit vectors, stacked once, for one Search's dense scores.

    Every Search stacked the same vectors again for each dense route: 480 MB of copying
    per route for a user of 59k Items. A document's position is by object identity, so a
    document that is not this corpus's own (a knowledge read's) is scored apart.

    Attributes:
        rows (UnitRows): The vectors in document order.
        positions (Mapping[int, int]): Each document's ``id`` with its row.
    """

    rows: UnitRows
    positions: Mapping[int, int]

    @classmethod
    def of(cls, documents: Sequence[_HasVector]) -> CorpusVectors:
        """Stack the vectors of the documents that have one."""
        held = [document for document in documents if document.vector is not None]
        return cls(
            UnitRows([cast(UnitVector, document.vector) for document in held]),
            {id(document): position for position, document in enumerate(held)},
        )

    def cosines(self, documents: Sequence[object], query: UnitVector) -> list[float] | None:
        """Each document's cosine with ``query``, or None to score them apart.

        None when a document is not this corpus's, or when so few are wanted that one
        product over all the rows costs more than stacking those few.

        Args:
            documents (Sequence[object]): The documents to score, each kept alive by the corpus.
            query (UnitVector): The query's unit vector.

        Returns:
            list[float] | None: One `UnitRows.cosines` value per document, in order.
        """
        positions = self.positions
        if 4 * len(documents) < len(positions):
            return None
        try:
            wanted = [positions[id(document)] for document in documents]
        except KeyError:
            return None
        scores = self.rows.cosines(query)
        return [scores[position] for position in wanted]


@dataclass(frozen=True, slots=True)
class IncomingUpdate:
    """One active ``supersedes`` link that ends an Item, with its source (A.6.1, A.6.4; T-S7a-1).

    Attributes:
        target_id (str): The Item the link ends.
        link (IncomingLink): The link as S1d's world functions read it; for
            basis ``source_observed_at`` it carries the Evidence granularity of
            its end.
        source_lifecycle (MemoryLifecycle): The lifecycle of the Item whose head
            carries the link.
        source_reason (RevisionReason): The ``reason_code`` of that head.
    """

    target_id: str
    link: IncomingLink
    source_lifecycle: MemoryLifecycle
    source_reason: RevisionReason


@dataclass(frozen=True, slots=True)
class MemoryDocument:
    """One Item of a memory corpus, its current head decoded once (A.6.1).

    Attributes:
        row (MemoryRowV2): The Item's projection, in any lifecycle.
        snapshot (MemorySnapshotV2): Its head snapshot.
        world (WorldMemory): The Item as S1d's world functions read it.
        vector (UnitVector | None): The head's unit vector; None for an erased
            head, which supplies no vector.
        sparse (SparseDocument): The `sparse_terms` of the head's statement
            and keywords, annotated with the `category_terms` of its keywords,
            Search's lexical document; empty when erased.
        keyword_forms (frozenset[str]): The `keyword_forms` of the head's
            keywords and their `category_terms`.
    """

    row: MemoryRowV2
    snapshot: MemorySnapshotV2
    world: WorldMemory
    vector: UnitVector | None
    sparse: SparseDocument
    keyword_forms: frozenset[str]

    @classmethod
    def of(cls, row: MemoryRowV2) -> MemoryDocument:
        """Decode one Item's head and prepare its documents.

        Args:
            row (MemoryRowV2): The Item.

        Returns:
            MemoryDocument: The prepared Item.
        """
        snapshot = decoded_snapshot(row.snapshot_json)
        sparse, forms = memory_text_parts(snapshot)
        return cls(
            row=row,
            snapshot=snapshot,
            world=WorldMemory(
                row.object_id, row.lifecycle_state, row.current_revision_id, row.reason_code,
                snapshot, row.links, row.corrected_targets,
            ),
            vector=None if snapshot.erased else snapshot_unit(snapshot),
            sparse=sparse,
            keyword_forms=forms,
        )

    @property
    def memory_id(self) -> str:
        """The Item's id."""
        return self.row.object_id

    @property
    def weight(self) -> int:
        """The estimated bytes the document retains: vectors, terms and keys."""
        vectors = len(self.snapshot.embedding) * (1 if self.vector is None else 2)
        return 2048 + 40 * vectors + 64 * self.sparse.length


def _observed_granularity(
    root: RecallSnapshot, fence: ScopeFence, head: MemorySnapshotV2, effective_at: int,
) -> TimeGranularity:
    """The granularity of a ``source_observed_at`` end: its finest cited Evidence, else DAY.

    The finest ``time_granularity`` among the successor head's cited parts
    whose ``source_observed_at`` equals ``effective_at``; DAY when no cited
    part still carries that time (W2-RT-9).
    """
    scope = fence.scope_kind, fence.scope_id
    found: list[TimeGranularity] = [
        part.time_granularity
        for part in (root.get_source(span.evidence_id) for span in head.quote_spans)
        if part is not None and (part.scope_kind, part.scope_id) == scope
        and part.source_observed_at == effective_at
    ]
    return min(found, key=_FINEST_FIRST.__getitem__) if found else "DAY"


class MemoryCorpus:
    """Every Item of one scope at one root, prepared for Search and world visibility (A.6.1).

    It holds the scope's Items in every lifecycle and nothing of a forgotten
    Item, with:

    - each head decoded once, its unit vector, its sparse document and its
      keyword forms (`MemoryDocument`);
    - ``search_statistics``: Search's BM25 statistics over the Items whose head
      passes P-VER, in any lifecycle, superseded heads included (A.6.3);
    - the anchor index ``EntityRef → memory ids`` of every non-erased head;
    - the incoming-update index: the active ``supersedes`` links that end each
      Item, with their source's lifecycle and head reason (A.6.4 witnesses);
    - the corrected-version map ``revision_id → [(historical Item id,
      created_commit_seq)]`` from every historical Item's ``corrected_targets``,
      erased correcting heads included, ordered by ``created_commit_seq`` then id.

    It serves `aml.core.readability.WorldIndex`: S1d's world functions read it
    and nothing here decides visibility. Erased heads, correction rows among
    them, supply no text, vector or statistics. Build one with `build`.
    """

    def __init__(
        self,
        scope_kind: ScopeKind,
        scope_id: str,
        documents: Sequence[MemoryDocument],
        updates: Mapping[str, tuple[IncomingUpdate, ...]],
        corrections: Mapping[str, tuple[CorrectionRef, ...]],
        search_statistics: SparseStatistics,
        readable: Sequence[str],
        rollups: Sequence[MemoryDocument] = (),
    ) -> None:
        """Hold prepared parts; `build` prepares them from a root."""
        self.scope_kind: ScopeKind = scope_kind
        self.scope_id = scope_id
        self.documents = tuple(sorted(documents, key=lambda document: document.memory_id))
        """Every Item of the scope in any lifecycle, by id, except the session rollups."""
        self.rollups = tuple(sorted(rollups, key=lambda document: document.memory_id))
        """The scope's session rollups (origin ``session_rollup``) in any lifecycle, by id.
        They are no part of the documents, statistics, typicality or indexes, so a Search
        ranks as if they were not stored; only a summary or ordering read delivers them."""
        self.search_statistics = search_statistics
        """Search's BM25 statistics: readable heads of any lifecycle (A.6.3)."""
        self._readable = tuple(readable)
        self._documents: Mapping[str, MemoryDocument] = {
            document.memory_id: document for document in self.documents
        }
        self._heads: Mapping[str, MemoryDocument] = {
            document.row.current_revision_id: document for document in self.documents
        }
        anchors: dict[str, list[str]] = {}
        for document in self.documents:
            for anchor in document.snapshot.anchors:
                anchors.setdefault(anchor, []).append(document.memory_id)
        self._anchors: Mapping[str, tuple[str, ...]] = {
            anchor: tuple(ids) for anchor, ids in anchors.items()
        }
        self._updates = updates
        self._incoming: Mapping[str, tuple[IncomingLink, ...]] = {
            target: tuple(update.link for update in found) for target, found in updates.items()
        }
        self._corrections = corrections
        self.weight: int = sum(
            document.weight for document in (*self.documents, *self.rollups)
        ) + 96 * len(self._readable)
        """The estimated bytes the corpus retains, typicality included."""

    @cached_property
    def typicality(self) -> Mapping[str, float]:
        """Each readable head's `centroid_typicality` among the readable heads (A.6.3).

        The population is the one of ``search_statistics``: the heads with a
        vector that pass P-VER at `STATISTICS_CLOCK`. Computed at the first
        dense Search of the corpus and kept with it.
        """
        return MappingProxyType(dict(centroid_typicality({
            memory_id: vector for memory_id in self._readable
            if (vector := self._documents[memory_id].vector) is not None
        })))

    @classmethod
    def build(cls, root: RecallSnapshot, scope_id: str) -> MemoryCorpus:
        """Prepare the memory corpus of one scope at a root.

        Args:
            root (RecallSnapshot): The published root.
            scope_id (str): The exact scope.

        Returns:
            MemoryCorpus: The scope's corpus.

        Raises:
            AMLError: ``scope_denied`` when the scope is unknown or tombstoned;
                ``index_rebuilding`` while a visibility fence closes it.
        """
        fence = _open_fence(root, scope_id)
        read = _statistics_read(root, fence)
        documents: list[MemoryDocument] = []
        rollups: list[MemoryDocument] = []
        for row in root.memory_rows(scope_id):
            document = MemoryDocument.of(row)
            (rollups if document.snapshot.origin == "session_rollup" else documents).append(
                document,
            )
        updates: dict[str, list[IncomingUpdate]] = {}
        corrections: dict[str, list[CorrectionRef]] = {}
        for document in documents:
            row = document.row
            for link in row.links:
                if link.relation is None:
                    continue
                observed = link.effective_basis == "source_observed_at"
                granularity = (
                    _observed_granularity(root, fence, document.snapshot, link.effective_at)
                    if observed and link.effective_at is not None else None
                )
                updates.setdefault(link.target_key, []).append(IncomingUpdate(
                    target_id=link.target_key,
                    link=IncomingLink(
                        link.object_id, row.object_id, link.relation, link.effective_at,
                        link.effective_basis, granularity,
                    ),
                    source_lifecycle=row.lifecycle_state,
                    source_reason=row.reason_code,
                ))
            if row.created_commit_seq is not None:
                for target in row.corrected_targets:
                    corrections.setdefault(target, []).append(
                        CorrectionRef(row.object_id, row.created_commit_seq),
                    )
        readable = [document for document in documents if read.p_ver(document.snapshot)]
        return cls(
            fence.scope_kind,
            scope_id,
            documents,
            {
                target: tuple(sorted(found, key=lambda update: update.link.link_id))
                for target, found in updates.items()
            },
            {
                target: tuple(sorted(
                    found, key=lambda ref: (ref.created_commit_seq, ref.memory_id),
                ))
                for target, found in corrections.items()
            },
            SparseStatistics.of(document.sparse for document in readable),
            tuple(document.memory_id for document in readable),
            rollups,
        )

    @cached_property
    def readable_documents(self) -> tuple[MemoryDocument, ...]:
        """The documents readable at `STATISTICS_CLOCK`, by id: those `search_statistics` counts."""
        return tuple(self._documents[memory_id] for memory_id in self._readable)

    @cached_property
    def vectors(self) -> CorpusVectors:
        """The unit vectors of the documents that have one, stacked once (`CorpusVectors`)."""
        return CorpusVectors.of(self.documents)

    def document(self, memory_id: str) -> MemoryDocument | None:
        """Return one Item of the scope, or None when it does not exist."""
        return self._documents.get(memory_id)

    def anchored(self, entity_id: str) -> tuple[str, ...]:
        """Return the Items whose non-erased head anchors an entity, primary or about, by id.

        Args:
            entity_id (str): An `EntityRef`.

        Returns:
            tuple[str, ...]: The Items in any lifecycle; readers apply eligibility.
        """
        return self._anchors.get(entity_id, ())

    def updates(self, memory_id: str) -> tuple[IncomingUpdate, ...]:
        """Return the active ``supersedes`` links that end an Item, with their sources, by id."""
        return self._updates.get(memory_id, ())

    def memory(self, memory_id: str) -> WorldMemory | None:
        """Return an Item of the scope as world visibility reads it, or None (`WorldIndex`)."""
        document = self._documents.get(memory_id)
        return None if document is None else document.world

    def head_memory(self, revision_id: str) -> WorldMemory | None:
        """Return the Item whose current head is a revision, or None (`WorldIndex`)."""
        document = self._heads.get(revision_id)
        return None if document is None else document.world

    def incoming(self, memory_id: str) -> tuple[IncomingLink, ...]:
        """Return the active ``supersedes`` links that target an Item, by link id (`WorldIndex`)."""
        return self._incoming.get(memory_id, ())

    def corrections(self, revision_id: str) -> tuple[CorrectionRef, ...]:
        """Return the historical Items correcting a revision, oldest first (`WorldIndex`).

        Args:
            revision_id (str): A revision of any Item of the scope.

        Returns:
            tuple[CorrectionRef, ...]: Every historical Item whose
                ``corrects_history`` rows target it, erased correcting heads
                included, by ``created_commit_seq`` then id; a forgotten
                correction is absent.
        """
        return self._corrections.get(revision_id, ())


@dataclass(frozen=True, slots=True)
class _NamingVersion:
    """What P-NAME reads of a naming Item: its lifecycle, its head and the head's support."""

    lifecycle_state: MemoryLifecycle
    current_revision_id: str
    statement: str
    quote_spans: tuple[QuoteSpanV1, ...]
    support_groups: tuple[tuple[int, ...], ...]

    @classmethod
    def of(cls, row: MemoryRowV2) -> _NamingVersion:
        snapshot = decoded_snapshot(row.snapshot_json)
        return cls(
            row.lifecycle_state, row.current_revision_id, snapshot.statement,
            tuple(snapshot.quote_spans), tuple(tuple(group) for group in snapshot.support_groups),
        )


class EntityCorpus:
    """The current name view of one scope at one root (A.6.1, A.6.5; read-Q8, read-Q9).

    Its entities are the scope's active entities and the virtual user. Its
    names are the ``entity_name`` rows that pass P-NAME at `STATISTICS_CLOCK`,
    each indexed whole. Every Search in current or plain mode reads it through
    `retrieval`, which applies P-NAME at the reader's clock. Build one with
    `build`. Source-participant speaker names share this bounded corpus: one
    name/key pair retains the last expiry of its P-SRC-readable Evidence, and
    query tokens select names through a prefix index without scanning Evidence.
    """

    def __init__(
        self,
        scope_kind: ScopeKind,
        scope_id: str,
        entities: Sequence[EntityRowV1],
        names: Sequence[tuple[EntityNameRowV1, _NamingVersion]],
        horizon: int | None,
        speakers: Mapping[str, Mapping[str, int | None]],
    ) -> None:
        """Hold prepared parts; `build` prepares them from a root."""
        self.scope_kind: ScopeKind = scope_kind
        self.scope_id = scope_id
        self._entities = tuple(entities)
        self._names = tuple(names)
        self._horizon = horizon
        self.participant_entities: Mapping[str, str] = MappingProxyType({
            row.participant_key: row.object_id for row in entities
            if row.participant_key is not None
        })
        self._speakers = {
            norm: tuple(sorted(expiries.items())) for norm, expiries in speakers.items()
        }
        self._speaker_prefixes = frozenset(
            " ".join(tokens[:end]) for norm in speakers
            if (tokens := norm.split(" ")) for end in range(1, len(tokens) + 1)
        )
        self._retrieval = self._view(self._names)
        self.weight: int = 1024 + sum(
            512 + 64 * (len(entry.tokens) + len(entry.terms))
            for entity in self._retrieval.entities() for entry in entity.names
        ) + 256 * len(self._entities) + sum(
            256 + len(norm.encode("utf-8")) + 192 * len(expiries)
            for norm, expiries in self._speakers.items()
        ) + sum(128 + len(prefix.encode("utf-8")) for prefix in self._speaker_prefixes)
        """The estimated bytes the corpus retains."""

    @classmethod
    def build(cls, root: RecallSnapshot, scope_id: str) -> EntityCorpus:
        """Prepare the current name view of one scope at a root.

        Args:
            root (RecallSnapshot): The published root.
            scope_id (str): The exact scope.

        Returns:
            EntityCorpus: The scope's name view.

        Raises:
            AMLError: ``scope_denied`` when the scope is unknown or tombstoned;
                ``index_rebuilding`` while a visibility fence closes it.
        """
        fence = _open_fence(root, scope_id)
        read = _statistics_read(root, fence)
        entities = root.entity_rows(scope_id)
        active = {row.object_id for row in entities} | {USER}
        versions: dict[str, _NamingVersion] = {}
        names: list[tuple[EntityNameRowV1, _NamingVersion]] = []
        for row in root.entity_name_rows(scope_id):
            naming = root.get_memory(row.naming_memory_id)
            if row.entity_id not in active or naming is None or (
                (naming.scope_kind, naming.scope_id) != (fence.scope_kind, scope_id)
            ):
                continue
            version = versions.get(naming.object_id)
            if version is None:
                version = versions[naming.object_id] = _NamingVersion.of(naming)
            if read.p_name(row, version, version):
                names.append((row, version))
        naming = {row.naming_memory_id for row, _ in names}
        expiries = [
            part.expires_at
            for memory_id in naming for span in versions[memory_id].quote_spans
            if (part := root.get_source(span.evidence_id)) is not None
            and part.expires_at is not None
        ]
        participants = {row.participant_key for row in entities if row.participant_key is not None}
        speakers: dict[str, dict[str, int | None]] = {}
        if participants:
            for row in root.scope_sources(scope_id):
                source = row.source_participant
                if source is None or row.speaker_entity is None or not read.p_src(row):
                    continue
                key = participant_key(
                    source.namespace, source.conversation_id, source.participant_id,
                )
                if key not in participants or not (norm := name_norm(row.speaker_entity)):
                    continue
                supported = speakers.setdefault(norm, {})
                if key not in supported or row.expires_at is None:
                    supported[key] = row.expires_at
                elif (previous := supported[key]) is not None:
                    supported[key] = max(previous, row.expires_at)
        return cls(
            fence.scope_kind, scope_id, entities, names, min(expiries, default=None), speakers,
        )

    def speakers(self, read: Readability, texts: Sequence[str]) -> Mapping[str, tuple[str, ...]]:
        """Return the speaker names in query texts with P-SRC support at this clock.

        Args:
            read (Readability): The reader's exact scope and clock, on this root.
            texts (Sequence[str]): The query and its string choices.

        Returns:
            Mapping[str, tuple[str, ...]]: Matching normalized names mapped to
                participant keys. A pair survives while at least one source
                remains unexpired; Evidence was checked when the corpus built.

        Raises:
            AMLError: ``scope_denied`` if the read belongs to another scope.
        """
        if (read.scope_kind, read.scope_id) != (self.scope_kind, self.scope_id):
            raise AMLError("scope_denied", "A speaker index serves its own exact scope")
        found: dict[str, tuple[str, ...]] = {}
        for text in texts:
            tokens = token_seq(text)
            for start in range(len(tokens)):
                for end in range(start + 1, len(tokens) + 1):
                    norm = " ".join(tokens[start:end])
                    if norm not in self._speaker_prefixes:
                        break
                    if norm in self._speakers and norm not in found:
                        found[norm] = tuple(
                            key for key, expiry in self._speakers[norm]
                            if expiry is None or expiry > read.now
                        )
        return found

    def retrieval(self, read: Readability) -> EntityRetrieval:
        """Return the scope's current name view at a reader's clock (A.6.5).

        Each name is kept only while its row passes P-NAME at ``read``'s clock.
        Until the earliest expiry among the Evidence its naming heads cite, that
        is every prepared name, and the prepared view is returned.

        Args:
            read (Readability): The reader's readability, over the same root's
                Evidence.

        Returns:
            EntityRetrieval: The current view.

        Raises:
            AMLError: ``scope_denied`` when ``read`` binds another scope.
        """
        if (read.scope_kind, read.scope_id) != (self.scope_kind, self.scope_id):
            raise AMLError("scope_denied", "A name view serves its own exact scope")
        if self._horizon is None or read.now < self._horizon:
            return self._retrieval
        return self._view(tuple(
            (row, version) for row, version in self._names if read.p_name(row, version, version)
        ))

    def _view(self, names: Sequence[tuple[EntityNameRowV1, _NamingVersion]]) -> EntityRetrieval:
        grouped: dict[str, list[EntityNameRowV1]] = {}
        for row, _ in names:
            grouped.setdefault(row.entity_id, []).append(row)

        def entries(entity_id: str) -> tuple[NameEntry, ...]:
            rows = sorted(grouped.get(entity_id, ()), key=lambda row: (
                _NAME_KIND_ORDER[row.name_kind], -row.commit_seq, row.name_norm,
                row.naming_revision_id, row.annotation_index,
            ))
            return tuple(NameEntry.of(row.entity_id, row.name, row.name_kind) for row in rows)

        return EntityRetrieval((
            *(EntityEntry(row.object_id, row.entity_type, row.participant_key,
                          entries(row.object_id)) for row in self._entities),
            EntityEntry(USER, "user", None, entries(USER)),
        ))


@dataclass(frozen=True, slots=True)
class EvidenceDocument:
    """One Evidence part of the bench Evidence corpus (A.6.1, PD:P5).

    Attributes:
        row (SourceRowV2): The part.
        vector (UnitVector): Its unit vector.
        sparse (SparseDocument): The `sparse_terms` of its text.
    """

    row: SourceRowV2
    vector: UnitVector
    sparse: SparseDocument


class EvidenceCorpus:
    """The bench Evidence documents of one scope at one root (A.6.1, D.2 P-EVR).

    Its documents are the parts that pass P-EVR at `STATISTICS_CLOCK`: active,
    indexed, without a withheld range and not held by a pending candidate. A
    held or partly withheld part contributes nothing while the message's other
    parts remain (PD:P5). Folding never removes a part from this corpus.
    Build one with `build`.
    """

    @cached_property
    def vectors(self) -> CorpusVectors:
        """The parts' unit vectors, stacked once (`CorpusVectors`)."""
        return CorpusVectors.of(self.documents)

    def __init__(
        self, scope_kind: ScopeKind, scope_id: str, documents: Sequence[EvidenceDocument],
    ) -> None:
        """Hold prepared documents; `build` prepares them from a root."""
        self.scope_kind: ScopeKind = scope_kind
        self.scope_id = scope_id
        self.documents = tuple(sorted(documents, key=lambda document: document.row.object_id))
        """The parts, by id."""
        self.statistics = SparseStatistics.of(document.sparse for document in self.documents)
        """Evidence BM25 statistics over the parts."""
        self._documents: Mapping[str, EvidenceDocument] = {
            document.row.object_id: document for document in self.documents
        }
        self.weight: int = sum(
            1120 + 40 * len(document.vector) + 64 * document.sparse.length
            for document in self.documents
        )
        """The estimated bytes the corpus retains, typicality included."""

    @cached_property
    def typicality(self) -> Mapping[str, float]:
        """Each part's `centroid_typicality` among the corpus's parts (A.6.3).

        Computed at the first dense Search of the corpus and kept with it.
        """
        return MappingProxyType(dict(centroid_typicality({
            document.row.object_id: document.vector for document in self.documents
        })))

    @classmethod
    def build(cls, root: RecallSnapshot, scope_id: str) -> EvidenceCorpus:
        """Prepare the Evidence corpus of one scope at a root.

        Args:
            root (RecallSnapshot): The published root.
            scope_id (str): The exact scope.

        Returns:
            EvidenceCorpus: The scope's corpus.

        Raises:
            AMLError: ``scope_denied`` when the scope is unknown or tombstoned;
                ``index_rebuilding`` while a visibility fence closes it.
        """
        fence = _open_fence(root, scope_id)
        read = _statistics_read(root, fence)
        held = root.held_evidence(scope_id)
        return cls(fence.scope_kind, scope_id, [
            EvidenceDocument(row, *evidence_parts(row))
            for row in root.source_corpus(scope_id)
            if read.p_evr(row, held=row.object_id in held)
        ])

    def document(self, evidence_id: str) -> EvidenceDocument | None:
        """Return one part of the corpus, or None when it is not in the corpus."""
        return self._documents.get(evidence_id)


def cold_build(root: RecallSnapshot, scope_ids: Iterable[str]) -> int:
    """Build every corpus of each open scope of a cold root, as C.13 step 17 verifies them.

    A scope whose fence closes it serves no corpus and is skipped. An unknown
    scope, or a projection or corpus that cannot be built, raises, which aborts
    startup before readiness (INV:T01). Nothing is retained.

    Args:
        root (RecallSnapshot): A root built cold from committed rows.
        scope_ids (Iterable[str]): Every registered scope.

    Returns:
        int: The corpora built, three per open scope.

    Raises:
        AMLError: ``scope_denied`` for a scope the root lacks; any refusal of a
            corpus build.
    """
    built = 0
    for scope_id in scope_ids:
        if _closed(root.scope_fence(scope_id)) is not None:
            continue
        MemoryCorpus.build(root, scope_id)
        EntityCorpus.build(root, scope_id)
        EvidenceCorpus.build(root, scope_id)
        built += 3
    return built


type _Corpus = MemoryCorpus | EntityCorpus | EvidenceCorpus
type _Key = tuple[CorpusKind, str]


_CORPUS_TYPES: Final[Mapping[CorpusKind, type[_Corpus]]] = MappingProxyType({
    "memory": MemoryCorpus, "entity": EntityCorpus, "evidence": EvidenceCorpus,
})

MAX_OVERSIZED_CORPORA: Final = 4
"""Corpora above `MAX_CACHE_WEIGHT` a process keeps for the root they were built at."""

_oversized: OrderedDict[_Key, tuple[ReferenceType[RecallSnapshot], _Corpus]] = OrderedDict()
"""Each kept oversized corpus with a weak reference to the root it was built at, by use."""
_oversized_drops: dict[str | None, int] = {}
"""How often `drop_built_corpora` ran for each scope (None: for every scope)."""
_oversized_lock = CheckedLock()


def _drops(scope_id: str) -> tuple[int, int]:
    """The drop counts that a build of the scope must still see to be kept; the caller locks."""
    return _oversized_drops.get(None, 0), _oversized_drops.get(scope_id, 0)


def built_corpus[T: (MemoryCorpus, EntityCorpus, EvidenceCorpus)](
    kind: CorpusKind,
    root: RecallSnapshot,
    scope_id: str,
    build: Callable[[RecallSnapshot, str], T],
) -> T:
    """Return ``build(root, scope_id)``, reusing the one built at this very root when oversized.

    A `SnapshotRecallCache` never retains a corpus above its weight bound, and
    delivery reads no such cache, so every Search built its scope's memory corpus
    again for delivery and for the delivery recheck. A long user's Search (59k
    Items in a deployment run) also built it for ranking, and its Evidence
    corpus for ranking and for each model note's completeness check. A corpus is
    a pure function of its immutable root and scope, so the last
    `MAX_OVERSIZED_CORPORA` built corpora are kept here by kind and scope, each with
    a weak reference to its root: one is returned only to a reader of that very
    root, while it lives, and no root is kept alive. A corpus holds only its own
    scope's values; one whose root has died leaves at the next build, and
    `drop_built_corpora` (`SnapshotRecallCache.drop` and ``clear``) drops them
    and keeps nothing a build in flight returns.

    Args:
        kind (CorpusKind): The corpus kind ``build`` prepares.
        root (RecallSnapshot): The root to build at.
        scope_id (str): The exact scope.
        build (Callable[[RecallSnapshot, str], T]): The kind's ``build``.

    Returns:
        T: The corpus, shared read-only with every other reader of the root.

    Raises:
        AMLError: Whatever ``build`` raises; a failed build keeps nothing.
    """
    key: _Key = (kind, scope_id)
    with _oversized_lock:
        entry = _oversized.get(key)
        kept = None
        if entry is not None and entry[0]() is root:
            _oversized.move_to_end(key)
            kept = entry[1]
        drops = _drops(scope_id)
    if kept is not None:
        if not isinstance(kept, _CORPUS_TYPES[kind]):
            raise AMLError("parity_mismatch", "A corpus of another kind answered the request")
        return cast(T, kept)
    corpus = build(root, scope_id)
    released: list[tuple[ReferenceType[RecallSnapshot], _Corpus]] = []
    with _oversized_lock:
        if _drops(scope_id) == drops:
            for dead in [name for name, (held, _) in _oversized.items() if held() is None]:
                released.append(_oversized.pop(dead))
            replaced = _oversized.pop(key, None)
            if replaced is not None:
                released.append(replaced)
            _oversized[key] = (ref(root), corpus)
            while len(_oversized) > MAX_OVERSIZED_CORPORA:
                released.append(_oversized.popitem(last=False)[1])
    # Released outside the lock: a dying corpus's snapshots run memo callbacks.
    released.clear()
    return corpus


def memory_corpus(root: RecallSnapshot, scope_id: str) -> MemoryCorpus:
    """Return ``MemoryCorpus.build(root, scope_id)`` through `built_corpus`.

    Args:
        root (RecallSnapshot): The root to build at.
        scope_id (str): The exact scope.

    Returns:
        MemoryCorpus: The scope's corpus at that root.

    Raises:
        AMLError: As ``MemoryCorpus.build`` raises.
    """
    return built_corpus("memory", root, scope_id, MemoryCorpus.build)


def drop_built_corpora(scope_id: str | None = None) -> None:
    """Drop the kept oversized corpora of one scope, or of every scope when None.

    A build of that scope in flight keeps nothing it returns. The scope's kept Evidence
    parts go too: each holds its row's text and embedding, and D.6 (3) leaves none of
    them behind privacy quiescence.

    Args:
        scope_id (str | None): The scope whose corpora privacy quiescence drops.
    """
    with _oversized_lock:
        _oversized_drops[scope_id] = _oversized_drops.get(scope_id, 0) + 1
        released = [
            _oversized.pop(key) for key in list(_oversized)
            if scope_id is None or key[1] == scope_id
        ]
    released.clear()
    if scope_id is None:
        _evidence_parts.clear()
    else:
        _evidence_parts.discard(lambda row: row.scope_id == scope_id)


def _unchecked() -> None:
    return None


class SnapshotRecallCache:
    """The bounded per-scope corpora of one published root (A.6.1).

    It retains at most ``max_corpora`` corpora of at most ``max_weight``
    estimated bytes in all, dropping the least recently used first; a corpus
    larger than the bound is built for its caller and not retained here
    (`built_corpus` keeps it for its root while it lives). A corpus is
    built at most once at a time, outside the cache's lock, and concurrent
    callers of the same corpus share that build. A corpus grants nothing: it
    never decides eligibility and never hides a fence, because every access
    rechecks the root's fence for the scope.

    The read model holds one cache per published root. `advance` carries a
    corpus to the next root only when that root's delta left its inputs
    unchanged: Item, revision and link changes invalidate the memory corpus,
    entity and name changes the entity corpus, Evidence and hold changes the
    Evidence corpus, Evidence withholding, state and expiry changes the memory
    and entity corpora too, and an embedding or privacy change every corpus of
    the scope (``RecallSnapshot.shares_corpus``). The entity corpus also
    follows its naming heads, so memory changes invalidate it. A corpus with
    participants also follows Evidence changes for its speaker-name index. `drop` removes
    one scope's corpora during privacy quiescence (D.6) and `clear` every corpus.

    Args:
        root (RecallSnapshot): The published root.
        max_weight (int): The estimated bytes retained at most.
        max_corpora (int): The corpora retained at most.

    Raises:
        ValueError: If a bound is negative.
    """

    def __init__(
        self,
        root: RecallSnapshot,
        *,
        max_weight: int = MAX_CACHE_WEIGHT,
        max_corpora: int = MAX_CACHED_CORPORA,
    ) -> None:
        """Start an empty cache; see the class documentation."""
        if max_weight < 0 or max_corpora < 0:
            raise ValueError("Cache bounds are 0 or more")
        self._root = root
        self._max_weight = max_weight
        self._max_corpora = max_corpora
        self._weight = 0
        self._condition = threading.Condition(threading.Lock())
        self._entries: OrderedDict[_Key, _Corpus] = OrderedDict()
        self._pending: dict[_Key, Future[_Corpus]] = {}
        self._generation = 0
        self._scope_generations: dict[str, int] = {}

    @property
    def root(self) -> RecallSnapshot:
        """The root whose corpora this cache prepares."""
        return self._root

    @property
    def weight(self) -> int:
        """The estimated bytes of the retained corpora."""
        with self._condition:
            return self._weight

    def retained(self) -> tuple[tuple[CorpusKind, str], ...]:
        """Return the kind and scope of every retained corpus, least recently used first."""
        with self._condition:
            return tuple(self._entries)

    def memory(self, scope_id: str, *, check: Callable[[], None] = _unchecked) -> MemoryCorpus:
        """Return the memory corpus of a scope, building it when it is not retained.

        Args:
            scope_id (str): The exact scope.
            check (Callable[[], None]): Raises when the caller's read is no longer
                live; called before and while waiting, and after a shared build.

        Returns:
            MemoryCorpus: The corpus at this cache's root.

        Raises:
            AMLError: ``scope_denied`` or ``index_rebuilding`` when the root's
                fence closes the scope; any error of ``check``.
        """
        return self._get("memory", scope_id, MemoryCorpus.build, MemoryCorpus, check)

    def entity(self, scope_id: str, *, check: Callable[[], None] = _unchecked) -> EntityCorpus:
        """Return the entity corpus of a scope, building it when it is not retained.

        Args:
            scope_id (str): The exact scope.
            check (Callable[[], None]): As for `memory`.

        Returns:
            EntityCorpus: The corpus at this cache's root.

        Raises:
            AMLError: As for `memory`.
        """
        return self._get("entity", scope_id, EntityCorpus.build, EntityCorpus, check)

    def evidence(self, scope_id: str, *, check: Callable[[], None] = _unchecked) -> EvidenceCorpus:
        """Return the bench Evidence corpus of a scope, building it when it is not retained.

        Args:
            scope_id (str): The exact scope.
            check (Callable[[], None]): As for `memory`.

        Returns:
            EvidenceCorpus: The corpus at this cache's root.

        Raises:
            AMLError: As for `memory`.
        """
        return self._get("evidence", scope_id, EvidenceCorpus.build, EvidenceCorpus, check)

    def advance(self, root: RecallSnapshot) -> SnapshotRecallCache:
        """Return the cache of the next published root, carrying every unchanged corpus.

        Args:
            root (RecallSnapshot): The root published after this cache's root.

        Returns:
            SnapshotRecallCache: A new cache of the same bounds. It shares no
                build in flight with this one.
        """
        following = SnapshotRecallCache(
            root, max_weight=self._max_weight, max_corpora=self._max_corpora,
        )
        with self._condition:
            entries = tuple(self._entries.items())
        with following._condition:
            for (kind, scope_id), corpus in entries:
                if root.shares_corpus(self._root, scope_id, kind=kind) and (
                    kind != "entity" or root.shares_corpus(self._root, scope_id, kind="memory")
                ) and (
                    not isinstance(corpus, EntityCorpus) or not corpus.participant_entities
                    or root.shares_corpus(self._root, scope_id, kind="evidence")
                ):
                    following._retain((kind, scope_id), corpus)
        return following

    def drop(self, scope_id: str) -> None:
        """Drop every corpus of one scope, and keep none a build in flight returns (D.6).

        Args:
            scope_id (str): The scope whose corpora privacy quiescence drops.
        """
        with self._condition:
            self._scope_generations[scope_id] = self._scope_generations.get(scope_id, 0) + 1
            for key in [key for key in self._entries if key[1] == scope_id]:
                self._weight -= self._entries.pop(key).weight
        drop_built_corpora(scope_id)

    def clear(self) -> None:
        """Drop every corpus, and keep none a build in flight returns."""
        with self._condition:
            self._generation += 1
            self._entries.clear()
            self._weight = 0
        drop_built_corpora()

    def _retain(self, key: _Key, corpus: _Corpus) -> None:
        """Retain a corpus within the bounds; the caller holds the lock."""
        if corpus.weight > self._max_weight or self._max_corpora < 1:
            return
        while self._entries and (
            self._weight + corpus.weight > self._max_weight
            or len(self._entries) >= self._max_corpora
        ):
            _, removed = self._entries.popitem(last=False)
            self._weight -= removed.weight
        self._entries[key] = corpus
        self._weight += corpus.weight

    def _get[T: (MemoryCorpus, EntityCorpus, EvidenceCorpus)](
        self,
        kind: CorpusKind,
        scope_id: str,
        build: Callable[[RecallSnapshot, str], T],
        expected: type[T],
        check: Callable[[], None],
    ) -> T:
        key: _Key = (kind, scope_id)
        while True:
            check()
            _open_fence(self._root, scope_id)
            with self._condition:
                cached = self._entries.get(key)
                if cached is not None:
                    self._entries.move_to_end(key)
                    corpus: _Corpus = cached
                    break
                pending = self._pending.get(key)
                generation: tuple[int, int] | None = None
                if pending is None:
                    if len(self._pending) >= max(1, self._max_corpora):
                        # At most max_corpora builds run at once. The waiting
                        # reader rechecks its read outside the lock.
                        self._condition.wait(timeout=0.05)
                        continue
                    pending = self._pending[key] = Future[_Corpus]()
                    generation = self._generation, self._scope_generations.get(scope_id, 0)
            if generation is not None:
                corpus = self._build(key, pending, generation, build)
                break
            # Another reader builds this corpus. A waiting reader may lose its
            # read; its cancellation never cancels the shared build.
            while not wait((pending,), timeout=0.05).done:
                check()
            corpus = pending.result()
            check()
            break
        if not isinstance(corpus, expected):
            raise AMLError("parity_mismatch", "A corpus of another kind answered the request")
        return corpus

    def _build[T: (MemoryCorpus, EntityCorpus, EvidenceCorpus)](
        self,
        key: _Key,
        pending: Future[_Corpus],
        generation: tuple[int, int],
        build: Callable[[RecallSnapshot, str], T],
    ) -> T:
        try:
            corpus = built_corpus(key[0], self._root, key[1], build)
            with self._condition:
                if generation == (self._generation, self._scope_generations.get(key[1], 0)):
                    self._retain(key, corpus)
            pending.set_result(corpus)
            return corpus
        except BaseException as error:
            pending.set_exception(error)
            raise
        finally:
            with self._condition:
                del self._pending[key]
                self._condition.notify_all()
