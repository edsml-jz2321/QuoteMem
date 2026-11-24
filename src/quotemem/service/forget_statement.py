"""Forget by statement on the single Writer: Search, the judge's pick, one forget per target.

Extraction can store one statement as several memories, so forgetting one memory id
can leave the rest of the statement current. ``StatementForget`` forgets a statement:

``prepare``
    1. Searches the statement in the user's scope with the runtime's own Search, at
       ``top_k`` = ``MAX_FORGET_CANDIDATES``, exactly as a client Search runs.
    2. Its candidates are the memory hits whose memory is current (``p_mem``) in the scope,
       in rank order, each read by its current statement under the Writer mutex.
    3. The product's judge picks the targets (`quotemem.core.forget_pick`, prompt
       ``memory-forget-pick-v1``); every vote is a call outside every lock.
    4. The plan names each target's exact head, each candidate's head, the scope's
       ``erase_epoch`` and an expiry ``INTENT_TTL_MS`` away, under the workspace's keyed
       ``forget-statement-plan`` binding. Preparing writes nothing; the Search records
       its own delivery row, as every Search does.

``confirm``
    Checks the binding, then forgets the targets in plan order, each through the
    forget (``ForgetOwner``: prepare, then admit, delete, scan and release) bound to the plan's
    head. The first confirmation also requires the scope unchanged since the plan: the same
    ``erase_epoch`` and every candidate at its planned head, else ``operation_conflict``
    (prepare again). Each target's forget is atomic; the statement is not. A repeated
    confirmation resumes: a target already gone is ``forgotten_earlier``, one whose
    head moved is ``changed`` and is left alone, the rest are forgotten. A head moved
    only by privacy requotes stating the same thing (an earlier target's privacy
    compensation: readable support, keywords) still states what the plan bound: that
    target is forgotten at its new head. Expiry refuses
    a confirmation that still has a target to forget (``intent_expired``); one with
    every target gone answers again. Forget itself calls no provider.

A workspace whose released extraction policy freezes ``explicit-request-v2`` (the ``agent``
Add mode) forgets completely: after a confirmation, and after an in-chat
forget, every memory a forget left without readable support is forgotten too
(`StatementForget._cascade`), and the in-chat forget also takes the memories recording its
requests, the other memories stating the fact, and withholds the sentences restating it
(`StatementForget.chat_forget`).

Such a workspace also prepares completely (`StatementForget._prepare_complete`):

- the candidates are Search's memory hits and, by one embedding call of the statement, the
  current memories nearest it (``DENSE_MEMORIES`` of them at cosine ``DENSE_FLOOR`` or more),
  so a Chinese request finds an English statement; every memory the judge decides states
  the statement or part of it is a target, and so is the record of an earlier request;
- a target the judge picked whose clause carrying the claim names another person or animal,
  or states the opposite claim, is no target (`quotemem.core.forget_match`); one that bundles a
  clause the judge finds unrelated with one that states the statement is a target all the
  same (its words state the fact) and is reported as ``bundled``;
- every readable sentence of the user's Evidence sharing terms with the statement or with
  what a target says or quotes, and the sentences of the Evidence parts near the
  statement's vector, is judged, and those stating it are the plan's ``spans``, with the
  sentences of this user's forget requests asking it; the confirmation withholds them with
  the first forgotten memory, through the same keeper paths as the memory's own quotes
  (`ForgetOwner.admit`): another memory quoting a clause of such a sentence that states
  only its own fact keeps that clause readable, and one that keeps nothing is left
  unreadable but never forgotten (`_leftovers`);
- a fact only the user's sentences state (no memory states it) is forgotten by them alone:
  the prepare issues that forget's intent (`ForgetOwner.prepare_spans`) with the statement
  and its vector as the forget's tombstone, so a later restatement is checked against it;
- a memory an ``update`` or ``correct`` link ties to what the forget takes goes with it when
  its own change names the forgotten value as one it replaced, ended or left
  (``names_replaced``, by the words of the value's own predicate, never a shared place or
  time); beyond the judge's picks, the base links and those linked replacers, a word
  rule only nominates: a memory no link ties to them that names the asked value so is taken
  only once the forget judge confirms it states or presupposes the fact (`_confirmed`,
  ``forget_pick.tells``). Without that answer, or with a contested one, every complete
  forget refuses, retryably, and forgets nothing: by statement, by memory id (where a
  judge-less forget goes on only when no sentence its judge would read still names the
  value, ``_unsettled``) and in chat. What a word rule withholds alone is only the words
  naming the value (``naming_parts``'s ``strict``) and the assistant's clauses stating it
  (``forget_admission.echoes``), never a sentence whole because one clause matched; a
  confirmation takes nothing its prepare did not name (``ForgetIntentV1.listed``).

A target that bundles the statement's fact with a clause the judge finds unrelated is
forgotten whole, but its tombstone is the user's words of the fact and their vector, not the
bundle: the plan names it in ``bundled`` with the user's ``statement``, and the
confirmation embeds that statement once for the tombstone. The in-chat forget does the same
for its requests, and it embeds them once too, so a memory or an Evidence part near a
request's vector (cosine ``DENSE_FLOOR`` or more) is a candidate in another language.

Every Add of such a workspace is checked for memories and sentences restating a tombstone by
words shared with it or by their vectors' cosine to the tombstone's (``RELEARN_PICKS``
tombstones at most), so a restatement in another language, or one the extraction stored
nowhere (a question, an acknowledgement), is judged too; what states another fact is never
taken (`forget_match.exclusion`).

What the user states again, in their own words, after a forget is learned again
(``core/relearn.py``): a dated user sentence received and dated after the
tombstone's forget, that is no forget request, names no forgetting or memory and is no
question, and whose own words restate that tombstone, is never taken by the relearn check
for it, review is not issued that tombstone (nomination, settlement and the commit
fence's guard), and the interactive Search does not shadow it by the terms it states. The
assistant's text, the request and the words it quotes, a question, a message without a date
or dated before the forget, and a later sentence about something else are restatements as
before; the forgotten old Evidence stays withheld. Competition workspaces never relearn.

One request, two paths: an ``agent`` plugin sends the user's "please forget
..." both as a forget by statement and inside its Add. Both paths pick with the same
candidates (`_matched`: Search's memory hits and the memories nearest the request's vector
from ``CANDIDATE_FLOOR``, the request read without its pointing words, `forget_match.probe`)
and the same sender (`_pick_sender`), and record what they did in ``forget_handled:{s}``,
keyed by the session and the request's words: the later path reads only what changed since
the earlier one read the scope, so it never contradicts it, and a prepare or confirmation
reports both (``earlier``), so the plugin can tell the user the final outcome.

A complete forget changes only what it says it changes:

- a memory the judge kept that only says the user has the thing the forgotten fact is about
  ("The user has a tabby cat." beside "The user's tabby cat is named Pixel.") keeps its own
  clause, and a sentence withheld as restating the fact forbids it no word of that clause
  (``forget_admission.predicate_words``'s ``existence``); a kept memory the forget must still
  change (its only quote lies in what it withholds) is listed by the prepare beforehand
  (``affected``, `ForgetOwner.preview`) and reported by the confirmation;
- the confirmation counts every memory it removed (``forgotten``), a memory whose only
  sentence was withheld as stating the fact included, not only its targets;
- an Add's in-chat forget records, per request, what it came to (``forget_add:{s}``, by keyed
  digests only), so the Add's answer reports it (``add_forgets``), and a memory the Add itself
  made that its own forget took is no memory it added (``ChatForgetOutcome.taken``);
- a forget that takes a naming memory takes, by rule and whatever its judge decided, every
  readable memory still showing that name as copied (`_name_holders`: "李伟
  attends a university in Beijing." after "The user's younger brother is named 李伟."), and
  the prepare lists each in ``affected``; a bundle's names are not forgotten and take nothing;
- a request to forget the name of a person or an animal of the user's ("forget my brother's
  name", "忘掉我弟弟的名字") takes, by rule, every memory tying a name to them and their naming
  memories (`_referent_namers`), listed as targets or in ``affected``.

A complete forget is fast: the prepare's Search and its one embedding run
together; its memory pick, the clause pick of its splittable candidates and the sentence
picks of the user's words run together, then only the picks needing the targets; every vote
goes without model reasoning (``thinking`` disabled), under ``AGENT_PICK_TIMEOUT_MS``,
hedged after ``AGENT_PICK_HEDGE_SECONDS``; no pick asks a ``remainders`` stage
(`forget_pick.pick_targets`'s ``minimal``); and the confirmation reuses the prepared
statement's vector. A memory taken as ``part`` of the fact whose quote is a fragment of the
quote of another candidate, judged unrelated, stays (`_spared`): withholding those words
would forget the other fact.
"""

from __future__ import annotations

import contextvars
import hashlib
import json
import re
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence, Set
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Final, Literal, cast, overload

from quotemem.api.bench import SearchRequest, SearchResult
from quotemem.api.bench_service import SearchOperation
from quotemem.api.memory_forget import (
    MAX_FORGET_TARGETS,
    ForgetStatementAffectedV1,
    ForgetStatementCallsV1,
    ForgetStatementCandidateRefV1,
    ForgetStatementCandidateV1,
    ForgetStatementDecisionV1,
    ForgetStatementEarlierV1,
    ForgetStatementOutcomeV1,
    ForgetStatementSpanV1,
    ForgetStatementTargetV1,
    ForgetStatementVoteV1,
    MemoryForgetStatementPreparedV1,
    MemoryForgetStatementPrepareRequestV1,
    MemoryForgetStatementRequestV1,
    PickPromptVersion,
)
from quotemem.core.controls import (
    HANDLED_PURPOSE,
    MAX_FORGET_ADD_REQUESTS,
    MAX_FORGET_ADDS,
    MAX_FORGET_HANDLED,
    MAX_FORGET_KEPT,
    MAX_FORGET_KEPT_HITS,
    MAX_FORGET_KEPT_PLANS,
    MAX_FORGET_MARKS,
    MAX_FORGET_WITHDRAWN,
    ForgetAddEntryV1,
    ForgetAddRequestV1,
    ForgetAddV1,
    ForgetHandledEntryV1,
    ForgetHandledV1,
    ForgetKeptEntryV1,
    ForgetKeptPlanV1,
    ForgetKeptV1,
    ForgetMarksV1,
    ForgetRewordPlanV1,
    ForgetShadowV1,
    ForgetWithdrawnV1,
    HandledInChat,
    HandledStatement,
)
from quotemem.core.entities import USER, MemorySnapshotV2, QuoteSpanV1
from quotemem.core.errors import QuoteMemError
from quotemem.core.extraction_policy import CHAT_FORGET_COMPLETE, extraction_pipeline
from quotemem.core.forget_detail import (
    DetailRequest,
    attribute,
    detail_request,
    detail_reworded,
    owned,
    quote_forms,
    quote_ranges,
    with_thing,
    workplace_values,
)
from quotemem.core.forget_detail import detail_values as detail_request_values
from quotemem.core.forget_match import (
    LINKED,
    PARENTHESIZED_FORM,
    TIES,
    Exclusion,
    NameRequest,
    asked_by,
    asks_name,
    clauses,
    exclusion,
    given_request,
    name_forms,
    name_only,
    name_request,
    naming_tie,
    nickname_request,
    other_script,
    other_subject,
    probe,
    tied_name,
    words_exclusion,
)
from quotemem.core.forget_pick import (
    FORGET_PICK_CLASSIFY_SCHEMA_ID,
    FORGET_PICK_MAX_READ_TIMEOUTS,
    FORGET_PICK_REMAINDERS_SCHEMA_ID,
    FORGET_PICK_TELLS_SCHEMA_ID,
    FORGET_PICK_TIMEOUT_MS,
    MAX_FORGET_CANDIDATES,
    ForgetPickClassifyV1,
    ForgetPickOutput,
    ForgetPickRemaindersV1,
    ForgetPickTellsV1,
    ForgetPickV1,
    Role,
    Send,
    Stage,
    forget_prompt,
    pick_targets,
    tells,
)
from quotemem.core.forget_remainder import readable_text, stored_remainder
from quotemem.core.forget_reword import (
    MAX_REWORDINGS,
    asked_kinds,
    asked_sources,
    asked_values,
    detail_values,
    existence_of,
    forgotten_values,
    han_forms,
    kind_of,
    left_out,
    reworded_statement,
    shows,
    tie_descriptions,
    value_pattern,
)
from quotemem.core.forget_state import (
    INTENT_TTL_MS,
    MAX_FORGET_KEPT_LISTED,
    MAX_RESTATING_SPANS,
    ForgetRewordV1,
    ForgetSeedV1,
)
from quotemem.core.identity import ArtifactId, ContentBindingDigest, IntegrityDigest
from quotemem.core.inference import (
    MAX_JUDGE_CONCURRENCY,
    InferenceBindingV1,
    ReasoningEffort,
)
from quotemem.core.privacy import (
    english,
    fact_terms,
    naming_value,
    support_words,
)
from quotemem.core.query import ScopeV1
from quotemem.core.readmission import admitted_request_ids
from quotemem.core.recall import SUMMARY_ONLY_ORIGINS
from quotemem.core.recall_agent import (
    NAME_KEEP_MEMORIES,
    SHADOW_PURPOSE,
    ForgetShadow,
    ForgetShadowTerms,
    content_terms,
    content_words,
    forget_shadowed,
    replacement_ranges,
    request_sentences,
    sentences,
    shadow_terms,
    surviving_names,
)
from quotemem.core.recall_corpora import sparse_terms
from quotemem.core.relearn import Forgotten, Told, group_told, restates
from quotemem.core.relearn import relearned as relearns
from quotemem.core.relearn import told as told_quote
from quotemem.core.retained_policy import read_released_extraction_config
from quotemem.core.session_rollup import RESTATED_SHARE
from quotemem.core.statement import (
    MAX_KEYWORDS,
    canonical_statement,
    name_norm,
    own_token,
    statement_digest,
    token_contained,
    token_seq,
)
from quotemem.core.vector_score import UnitVector, cosine, unit_vector
from quotemem.ingestion.forget_requests import (
    asked_part,
    is_request,
    lived_value,
    old_value,
    request_reach,
    requested,
    tells_new,
    whole_referent,
)
from quotemem.ingestion.relearn import asserts
from quotemem.providers.auth import PrincipalGrant
from quotemem.providers.bindings import inference_binding_for_policy, memory_judge_binding
from quotemem.providers.contracts import (
    EmbeddingProvider,
    InferenceRequest,
    Message,
    StructuredInferenceProvider,
)
from quotemem.store.entity_ops import TxnRows, requoted_from
from quotemem.store.forget_admission import (
    asks_modifier,
    echoes,
    forgotten_keywords,
    names_another,
    names_replaced,
    names_value,
    naming_parts,
    predicate_words,
)
from quotemem.store.forget_run import (
    ForgetOwner,
    ForgottenFact,
    asks_everything,
    bare_name_stays,
    names_tied,
    naming_only,
    workspace_embedding,
)
from quotemem.store.indexed import EvidenceRows, MemoryTableRows, TombstoneRows
from quotemem.store.memory_commit import readability
from quotemem.store.writer import RowWrite, Writer

type TargetState = Literal["pending", "gone", "changed"]

SEARCH_SECONDS = 120.0
"""The statement's own Search budget, queue time included."""
CHAT_SEARCH_TOP_K = 30
"""An in-chat request's Search depth: its memory and sentence candidates both come from it."""
_PARTS: frozenset[Role] = frozenset({"same_fact", "part"})
_TELLING: frozenset[Role] = frozenset({"same_fact", "part", "identity"})
"""The roles of a candidate that names or states what the request is about: the memories a
rewording uses to identify the forgotten values."""
_SHARED_LIGHT: Final = frozenset({"user", "last", "week", "month", "year", "now", "also"})
"""Words a later memory kept as it is may share with what is forgotten."""
"""The roles an in-chat request targets (`StatementForget.chat_forget`): the request names
the fact in the user's own words, so every stored piece of it goes, not only the minimal set
an exact statement forget takes; a candidate that only identifies an entity stays."""
CHAT_FORGET_REASONING_EFFORT: ReasoningEffort = "low"
"""The in-chat forget pick's judge effort.

At the pipeline's effort, the picks' reasoning length dominated the time of an Add that
holds a forget request. In replays of recorded Adds, ``low`` about halved the forget's
time, and its picks agreed with the pipeline effort's about as often as the pipeline
effort agreed with itself. Batching the lists into one call cut calls but made each
vote's reasoning longer. The prepare path keeps the pipeline's effort."""
CHAT_SCAN_CHUNKS = 4
"""A complete in-chat forget's sentence picks per request: each judges up
to ``MAX_FORGET_CANDIDATES`` sentences of the user's Evidence, those sharing the largest share
of a probe's terms first."""
CHAT_SCAN_SHARED = 2
"""The terms (`content_terms`) a sentence or memory shares with a probe to be judged, or all
of a probe's terms when it has fewer."""
NAMED_SHARE = 1e-6
"""The rank of a sentence judged only because it names a name of a probe (its shadow name):
after every sentence sharing terms, before the cap of ``CHAT_SCAN_CHUNKS`` picks."""
CASCADE_ROUNDS = 3
"""Passes that forget the memories a forget left without readable support
(`StatementForget._cascade`); one usually leaves none."""
_FORGET_REASONS = frozenset({"forgotten", "restated_forgotten"})
"""Withheld-span reasons a forget writes: a forget's own spans and a denied restatement's."""
_VOTE_ERRORS = frozenset({"provider_unavailable", "schema_invalid"})
"""Call failures that leave one vote unanswered; any other error propagates."""
_SEED_FALLBACK = frozenset({"provider_unavailable", "schema_invalid", "embedding_space_mismatch"})
"""Embedding failures after which a confirmed plan's bundled targets are tombstoned by their
own statements instead of the user's words (``confirm``)."""
DENSE_FLOOR = 0.6
"""The cosine to the statement's vector (or to a tombstone's) at which a memory or an
Evidence part is a candidate in another language than the request (complete forgets): the
judge still decides; the floor only bounds what it reads."""
DENSE_MEMORIES = 4
"""Candidate places a complete prepare keeps for the memories nearest the statement's vector
that Search did not return."""
RELEARN_PICKS = 8
"""Tombstones a complete forget judges one Add's memories against (`_relearned`), those
restated by shared words first, then by the highest cosine."""
_LEXICAL = 2.0
"""The rank of a lexical restatement among cosines (`_relearned`)."""
CANDIDATE_FLOOR = 0.5
"""The cosine to a request's vector from which a current memory is a complete forget's
candidate (`StatementForget._matched`): lower than ``DENSE_FLOOR``, which bounds the
sentences, as only ``DENSE_MEMORIES`` memories are taken this way."""
AGENT_PICK_TIMEOUT_MS = 20_000
"""A complete forget's vote timeout: a vote is a short classification."""
AGENT_PICK_HEDGE_SECONDS = 4.0
"""When a complete forget sends a still unanswered vote again."""
PHRASE_CACHE = 64
"""Prepared statements whose vector a complete prepare keeps for the confirmation."""
class _Delivery:
    """The statement's own Search delivery, answered inside the prepare call."""

    def __init__(self, seconds: float) -> None:
        self._received = time.monotonic()
        self._deadline = self._received + seconds
        self.result: SearchResult | None = None

    @property
    def received_at(self) -> float:
        return self._received

    @property
    def deadline(self) -> float:
        return self._deadline

    @property
    def cancelled(self) -> bool:
        return False

    def on_cancel(self, wake: Callable[[], None]) -> AbstractContextManager[None]:
        del wake

        @contextmanager
        def registered() -> Iterator[None]:
            yield

        return registered()

    def deliver(self, result: SearchResult) -> None:
        if self.result is not None:
            raise QuoteMemError("service_unhealthy", "Search delivered more than one response")
        self.result = result


@dataclass(frozen=True)
class _Candidate:
    memory_id: str
    revision_id: str
    content_hash: str
    statement: str
    quotes: tuple[QuoteSpanV1, ...]
    vector: tuple[float, ...] = ()
    space: str = ""
    keywords: tuple[str, ...] = ()
    groups: tuple[tuple[int, ...], ...] = ()


@dataclass(frozen=True)
class _Part:
    """An Evidence row as `core.relearn` reads it."""

    role: str | None
    source_observed_at: int | None
    ingested_at: int
    byte_start: int
    text: str
    embedding: tuple[float, ...] = ()
    embedding_space: str = ""

    @classmethod
    def of(cls, row: Mapping[str, object]) -> _Part:
        return cls(cast(str | None, row["role"]), cast(int | None, row["source_observed_at"]),
                   cast(int, row["ingested_at"]), cast(int, row["byte_start"]),
                   cast(str, row["text"]), tuple(cast(Sequence[float], row["embedding"])),
                   cast(str, row["embedding_space"]))


@dataclass(frozen=True)
class _Judged:
    """One sentence a complete forget judged, or a forget request it took without a vote."""

    span: QuoteSpanV1
    text: str
    role: Role | None
    target: bool
    excluded: Exclusion | None = None
    asked: frozenset[int] = frozenset()
    """The requests (positions in the statements judged against) that took it."""


@dataclass(frozen=True)
class _Phrase:
    """A complete prepare's statement vector in the workspace's space (one embedding call)."""

    vector: UnitVector
    raw: tuple[float, ...]
    space: str


@dataclass(frozen=True)
class _Stone:
    """One of the scope's tombstones as a relearn check reads it."""

    statement: str
    raw: tuple[float, ...]
    space: str
    created_at: int = 0
    entity_id: str = USER

    @property
    def unit(self) -> UnitVector | None:
        return unit_vector(self.raw) if any(self.raw) else None

    def forgotten(self) -> Forgotten:
        """This tombstone as a later statement is compared with it."""
        return Forgotten(self.statement, self.created_at, self.raw, self.space)

    def seed(self) -> ForgetSeedV1:
        """This tombstone as the seed of a forget of sentences restating it: the same
        tombstone (``ForgetOwner._seed_tombstone`` finds it by its entity and statement), so
        enforcing it writes no new one."""
        return ForgetSeedV1(statement=canonical_statement(self.statement),
                            embedding=list(self.raw), embedding_space=self.space,
                            entity_id=self.entity_id)


@dataclass(frozen=True)
class StatementConfirmed:
    """What one confirmation of a forget by statement did (`StatementForget.confirm_report`).

    Attributes:
        outcomes: Each target's outcome, then the plan's sentences' when no memory forget
            withheld them.
        forgotten: The memories it removed (``agent`` mode; None otherwise): every memory
            readable before it and not after it that the plan did not keep.
        withheld: The sentences it withheld (``agent`` mode).
        affected: The memories the plan kept that it changed all the same.
    """

    outcomes: tuple[ForgetStatementOutcomeV1, ...]
    forgotten: int | None = None
    withheld: int | None = None
    affected: tuple[ForgetStatementAffectedV1, ...] = ()


@dataclass(frozen=True)
class ChatForgetOutcome:
    """What one Add's in-chat forget requests did.

    Attributes:
        forgotten: The memories forgotten now (paraphrases included).
        changed: Picked memories whose head moved or that were gone before their forget.
        marked: Sentences newly marked as stating a fact the user asked to forget.
        withdrawn: Summary-only memories newly withdrawn from every read.
        calls: The judge calls the picks made.
        withheld: Restating sentences a complete forget withheld with a forgotten memory.
        cascaded: Memories a complete forget left without readable support, forgotten too
            (counted in ``forgotten`` as well).
    """

    forgotten: int
    changed: int
    marked: int
    withdrawn: int
    calls: int
    withheld: int = 0
    cascaded: int = 0
    affected: int = 0
    requests: tuple[ChatForgetRequest, ...] = ()
    taken: int = 0


type ChatForgetStatus = Literal["forgotten", "nothing_matched", "pending"]


@dataclass(frozen=True)
class ChatForgetRequest:
    """What one in-chat forget request of an Add came to (``agent`` mode).

    Attributes:
        statement: The request's words as extracted (``forget_requests``); held in memory
            only, never stored.
        status: ``forgotten`` when either path forgot something for it, ``pending`` while
            the plugin's forget by statement has a plan not yet confirmed, else
            ``nothing_matched``.
        forgotten: The memories removed for it (both paths' when ``earlier``).
        withheld: The sentences withheld for it (both paths' when ``earlier``).
        bundled: Its forgotten memories that bundled an unrelated fact.
        earlier: Whether the plugin's prepare and confirmation of the same request had
            forgotten it before this Add's in-chat forget read it.
    """

    statement: str
    status: ChatForgetStatus
    forgotten: int
    withheld: int
    bundled: int
    earlier: bool


def _sharing(statement: str, rows: Sequence[_Candidate]) -> list[_Candidate]:
    """The rows whose statement shares a content word with the statement, the
    ``MAX_FORGET_CANDIDATES`` sharing the largest share of its content words."""
    wanted = content_words(statement)
    shares = [(len(wanted & content_words(row.statement)) / max(1, len(wanted)), row)
              for row in rows]
    ranked = sorted((entry for entry in shares if entry[0] > 0), key=lambda entry: -entry[0])
    return [row for _, row in ranked[:MAX_FORGET_CANDIDATES]]


def _shared_terms(statement: str, rows: Sequence[_Candidate]) -> list[_Candidate]:
    """The rows sharing at least ``CHAT_SCAN_SHARED`` `content_terms` with the statement (all of
    them when it has fewer), the ``MAX_FORGET_CANDIDATES`` sharing the largest share first."""
    wanted = content_terms(statement)
    need = min(CHAT_SCAN_SHARED, len(wanted))
    if not need:
        return []
    shares = [(len(wanted & content_terms(row.statement)), row) for row in rows]
    ranked = sorted((entry for entry in shares if entry[0] >= need), key=lambda entry: -entry[0])
    return [row for _, row in ranked[:MAX_FORGET_CANDIDATES]]


def _restates(statement: str, forgotten: Sequence[frozenset[str]]) -> bool:
    """Whether a statement holds ``CHAT_SCAN_SHARED`` or more of a forgotten statement's
    `content_terms` and at least ``RESTATED_SHARE`` of them (``session_rollup``'s test, with CJK
    bigrams)."""
    terms = content_terms(statement)
    return any(len(terms & fact) >= CHAT_SCAN_SHARED
               and len(terms & fact) >= RESTATED_SHARE * len(fact) for fact in forgotten)


def _spared(rows: Sequence[_Candidate], roles: Sequence[Role]) -> set[int]:
    """The candidates a complete forget leaves although the judge took them as ``part`` of the
    fact: a memory one of whose quotes is a fragment of a quote of a candidate judged
    ``unrelated`` (inside it and shorter). Its words belong to that other fact's sentence
    ("The user has a cat." quoting "我的猫" of "我的猫叫 Mochi"), and forgetting the memory
    would withhold them and so forget the other fact. A memory quoting the very span an
    unrelated candidate quotes is still taken: the two then tell one sentence, and the judge's
    sentence pick decides that sentence; when it is withheld, the cascade takes what is left
    without support."""
    unrelated = [span for row, role in zip(rows, roles, strict=True) if role == "unrelated"
                 for span in row.quotes]
    return {index for index, (row, role) in enumerate(zip(rows, roles, strict=True))
            if role == "part" and any(
                span.evidence_id == other.evidence_id and other.byte_start <= span.byte_start
                and span.byte_end <= other.byte_end
                and (span.byte_start, span.byte_end) != (other.byte_start, other.byte_end)
                for span in row.quotes for other in unrelated)}


def _words(statement: str) -> str:
    """A statement's tokens (`token_seq`), joined: its wording without case or punctuation."""
    return " ".join(token_seq(statement))


def _about(snapshot: MemorySnapshotV2) -> set[str]:
    """The non-user entities a naming tie of ``snapshot`` may name: its primary entity, else
    the entities its naming annotations name."""
    if snapshot.entity_id != USER:
        return {snapshot.entity_id}
    return {entry.entity_id for entry in snapshot.naming if entry.entity_id != USER}


_EX_TIE = r"ex(?:-?(?:boyfriend|girlfriend|husband|wife|partner|fianc[eé]e?))?"
_EX = re.compile(
    rf"\b(?:(?P<name>[A-Z][a-z]+) is the user's {_EX_TIE}\b|[Tt]he user's {_EX_TIE},? "
    rf"(?:is named |is called )?(?P<after>[A-Z][a-z]+))")
_OLD_KINDS: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType({
    "车": ("car", "车"), "汽车": ("car", "车"), "车子": ("car", "车"),
    "自行车": ("bike", "bicycle"),
    "手机": ("phone",), "电脑": ("computer", "laptop"), "笔记本": ("laptop",),
    "房子": ("house", "home", "apartment"), "公寓": ("apartment",), "学校": ("school",),
    "公司": ("company", "employer"), "工作": ("job",), "宠物": ("pet",),
    "car": ("car",), "cars": ("car",), "bike": ("bike", "bicycle"), "phone": ("phone",),
    "address": ("address", "lives at", "lives in", "lived at", "lived in"),
    "地址": ("address", "lives at", "lives in", "lived at", "lived in"),
    "旧地址": ("address", "lives at", "lives in"),
    "laptop": ("laptop",), "house": ("house", "home"), "apartment": ("apartment",),
    "school": ("school",), "job": ("job",), "company": ("company", "employer"),
    "number": ("number",), "phone number": ("number",), "email": ("email",),
    "email address": ("email",),
})
"""The words a past value's statement shows for the kind an old-value request names, when
the judge takes no past value (`StatementForget._old_values`): "我以前那辆车" finds "The user's
car is a white 思域.", a value a newer one superseded."""
_USERS_OWN = re.compile(
    r"The user(?!'s)\b|The user's (?:(?:old|former|previous|current|new|first|home|phone|"
    r"email|mailing|work) )?(?:car|cars|vehicle|bike|bicycle|address|home|house|apartment|flat|"
    r"job|company|employer|school|phone|number|email|e-mail|laptop|computer|pet)\b")
"""A statement of the user's own value ("The user lives at 14 Elm Street.", "The user's car
is a white 思域."), never of someone of the user's ("The user's sister lives in Boston."): the
only past values an old-value request of the user's finds by its kind's words."""
_OLD_JUDGED: Final[Mapping[str, str]] = MappingProxyType({
    "地址": "address", "住址": "address", "家庭住址": "address", "车": "car", "汽车": "car",
    "车子": "car", "工作": "job", "公司": "company", "学校": "school", "电话": "phone number",
    "电话号码": "phone number", "手机号": "phone number", "手机号码": "phone number",
    "邮箱": "email address", "房子": "house", "公寓": "apartment", "手机": "phone"})
"""The English words a Chinese old-value kind is judged by (`StatementForget._old_values`)."""
_OLD_KIND_WORDS: Final[Mapping[str, re.Pattern[str]]] = MappingProxyType({
    "car": re.compile(r"\b(?:car|cars|drives?|drove|vehicle|SUV)\b", re.IGNORECASE),
    "job": re.compile(r"\b(?:job|works? (?:as|at|for)|working (?:as|at|for)|employed|position|"
                      r"profession|career)\b", re.IGNORECASE),
    "company": re.compile(r"\b(?:company|works? (?:at|for)|employer)\b", re.IGNORECASE),
    "school": re.compile(r"\b(?:school|studies at|attends)\b", re.IGNORECASE),
    "phone": re.compile(r"\bphone\b", re.IGNORECASE),
    "phone number": re.compile(r"\b(?:phone|number)\b", re.IGNORECASE),
    "number": re.compile(r"\b(?:phone|number)\b", re.IGNORECASE),
    "email address": re.compile(r"\be-?mail\b", re.IGNORECASE),
    "house": re.compile(r"\b(?:house|home)\b", re.IGNORECASE),
    "address": re.compile(r"\b(?:address|lives? (?:at|in)|living (?:at|in)|moved to|resides)\b",
                          re.IGNORECASE),
    "apartment": re.compile(r"\b(?:apartment|flat)\b", re.IGNORECASE)})
"""What a current memory of an old-value kind says, read only when the request told the new
value and the judge took none (`StatementForget._old_values`)."""
_EX_WORD = re.compile(r"\bex\b|\bex-|前男友|前女友|前夫|前妻|前任")
"""An ex of the user's, by the words of a memory ("Jordan is the user's ex-boyfriend.", "The
user's ex-boyfriend Jordan moved to Seattle.") and of a request ("my ex", "我前男友")."""


def _shows_own(statement: str, name: str, longer: Set[str]) -> bool:
    """Whether a statement shows a name as whole words outside every longer name the store
    knows for another person (``longer``: "Michael Jordan" and "Jordan Lee" beside an ex called
    Jordan): never the "Jordan" of "The user is a huge fan of Michael Jordan.", always the one
    of "Yesterday Jordan texted the user." (`StatementForget._whole_people`)."""
    for other in sorted(longer, key=len, reverse=True):
        statement = re.sub(rf"(?<![\w-]){re.escape(other)}(?![\w-])",
                           lambda found: " " * len(found.group()), statement)
    return _shows_name(statement, name)


def _shows_name(statement: str, name: str) -> bool:
    """Whether a statement shows a name as whole words (any run of Han characters holding
    it, for a Han name)."""
    if not name.isascii():
        return name in statement
    return re.search(rf"(?<![\w-]){re.escape(name)}(?![\w-])", statement) is not None


def _told(snapshots: Iterable[MemorySnapshotV2]) -> set[str]:
    """The entities the memories are about: their primary and other anchored entities."""
    return {entity for snapshot in snapshots
            for entity in (snapshot.entity_id, *snapshot.about_entity_ids)}


Within = Mapping[str, frozenset[tuple[str, int, int]]]
"""For an entity two people share, the sentences its forgotten name is forgotten in
(`StatementForget._referent_namers`)."""


def _inside(span: QuoteSpanV1, sentences_: Set[tuple[str, int, int]]) -> bool:
    """Whether ``span`` lies inside one of ``sentences_``."""
    return any(span.evidence_id == evidence_id and start <= span.byte_start
               and span.byte_end <= end for evidence_id, start, end in sentences_)


def _said(forgotten: str, survivor: str, *, modifier: bool = False) -> frozenset[str]:
    """What a forgotten statement says of a survivor's subject (``predicate_words``): of one
    only naming someone, the name alone, as the dependent trim reads it
    (``ForgetOwner._compensate``): "Ollie" of "The user's son Rowan is nicknamed Ollie.", so
    "Rowan is the user's son." restates none of it."""
    name = naming_value(forgotten) if english(forgotten) and english(survivor) else None
    if name is not None:
        return support_words(name) | fact_terms(name)
    return predicate_words(forgotten, survivor, existence=True, modifier=modifier)


def _scoped(given: Set[tuple[str, str]], found: Set[tuple[str, str]],
            ) -> frozenset[tuple[str, str]]:
    """The ``(entity_id, name)`` pairs a forget takes: once a name request resolved to an
    entity (``given``, `StatementForget._referent_namers`), only that entity's; else every
    pair its targets give."""
    entities = {entity for entity, _ in given}
    return frozenset(given) | frozenset(pair for pair in found
                                        if not entities or pair[0] in entities)


_HAN_RUN: Final = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")
"""A run of Han characters, as an English statement copies a Han word ("芝麻汤圆")."""


_BEARERS: Final = frozenset({
    "person", "people", "pet", "animal", "man", "woman", "boy", "girl", "child", "baby",
})
"""Entity types whose names the name rules withhold, from their own entity's sentences."""


def _han(name: str) -> bool:
    """Whether a name is written in Han characters alone ("芝麻", "李伟")."""
    return _HAN_RUN.fullmatch(name) is not None


def _vocabulary(statements: Iterable[str]) -> frozenset[str]:
    """The Han words the extraction wrote into English statements, each a Han run of one
    ("芝麻汤圆" of "The user likes eating 芝麻汤圆 for breakfast."): how a Han text no space
    cuts is cut into words."""
    return frozenset(match.group() for statement in statements if not _mostly_own(statement)
                     for match in _HAN_RUN.finditer(statement))


def _whole(name: str, text: str, start: int, end: int, words: Set[str]) -> bool:
    """Whether the occurrence ``text[start:end]`` of a Han name is the name itself, not a
    piece of a longer Han word: in a mostly Latin text, its Han run is the name alone (the
    芝麻 of "The user's cat 芝麻", not of "The user likes eating 芝麻汤圆"); in a Han text, no
    longer word of ``words`` (`_vocabulary`) covers it there ("我早餐喜欢吃芝麻汤圆")."""
    if not _mostly_own(text):
        return any(match.start() <= start and end <= match.end() and match.group() == name
                   for match in _HAN_RUN.finditer(text))
    return not any(
        found.start() <= start and end <= found.end()
        for word in words if len(word) > len(name) and name in word
        for found in re.finditer(re.escape(word), text))


def _shows(name: str, text: str, words: Set[str] | None = None) -> bool:
    """Whether ``text`` shows ``name`` as whole tokens: a capitalized Latin name in a Latin
    text as written only ("Pepper", not the "pepper" on eggs), any other in any case
    (`token_contained`). Given ``words`` (`_vocabulary`), a Han name counts only where it is
    no piece of a longer Han word (`_whole`): the 芝麻 of "芝麻汤圆" is no cat called 芝麻."""
    if not token_contained(name, text):
        return False
    if _han(name):
        return words is None or any(_whole(name, text, match.start(), match.end(), words)
                                    for match in re.finditer(re.escape(name), text))
    if name == name.casefold() or _mostly_own(text) or any(own_token(token)
                                                           for token in token_seq(name)):
        return True
    body = r"\s+".join(re.escape(word) for word in name.split())
    return re.search(rf"(?<![0-9A-Za-z]){body}(?![0-9A-Za-z])", text) is not None


_OWNS: Final = re.compile(r"The user has (?:a|an) [a-z]+(?: [a-z]+){0,2}\.")
"""A statement saying only that the user has someone or something ("The user has a dog.")."""
_ALIAS: Final = re.compile(
    r"\b(?:named|called|nicknamed|known as|goes by|go by|nickname|alias)\b|叫|名字|小名|昵称|外号",
    re.IGNORECASE)
"""A word that gives someone a name ("is also called", "nicknamed", 小名)."""
_ALIAS_FORM: Final = re.compile(
    r"\b(?:also (?:known as|called|named)|nicknamed|known as|goes by|go by|nick ?names?|"
    r"pet names?)\b|小名|昵称|外号|乳名|绰号|爱称", re.IGNORECASE)
"""A statement giving someone another name besides the name ("林可欣 is also known as 安安.")."""


def _aliases(snapshot: MemorySnapshotV2) -> set[tuple[str, str]]:
    """The ``(entity_id, name)`` pairs of another name than the name ``snapshot`` gives
    someone: its alias naming annotations, and the name a statement giving one states ("安安"
    of "林可欣 is also known as 安安."), for each entity it is about."""
    found = {(entry.entity_id, entry.name) for entry in snapshot.naming
             if entry.entity_id != USER and entry.name_kind == "alias"}
    alias = naming_value(snapshot.statement) if _ALIAS_FORM.search(snapshot.statement) else None
    if alias is not None:
        found |= {(entity, alias) for entity in {
            snapshot.entity_id, *snapshot.about_entity_ids, *_about(snapshot)} - {USER, ""}}
    return found


_CALENDAR: Final = frozenset({
    "january", "february", "march", "april", "may", "june", "july", "august", "september",
    "october", "november", "december", "monday", "tuesday", "wednesday", "thursday", "friday",
    "saturday", "sunday"})
"""Names that are also a month or a weekday: their only non-name reading the text alone shows."""
_TEMPORAL: Final = re.compile(
    r"\b(?:in|on|since|until|during|next|last|early|late|mid|this|every)\s+$", re.IGNORECASE)
"""A word before a month or a weekday that makes it a date ("in May", "next June")."""


def _dated(name: str, text: str, start: int, end: int) -> bool:
    """Whether the ``name`` at ``text[start:end]`` is a month or a weekday there, no one's name:
    after a temporal word or before a number ("in May", "May 5"), never a possessive ("in
    May's car"). Any other name stands as a name wherever it shows: a place or a thing reading
    is told by another entity bearing the name (`StatementForget._name_readers`)."""
    return (name.casefold() in _CALENDAR and re.match(r"['\u2019]s\b", text[end:]) is None
            and (_TEMPORAL.search(text[:start]) is not None
                 or re.match(r"\s*\d", text[end:]) is not None))


def _named_at(name: str, text: str, words: Set[str] | None = None) -> bool:
    """Whether ``text`` shows ``name`` where a person's or an animal's name stands (`_shows`):
    a Latin name anywhere but as a date (`_dated`: "The user's birthday is in May."); a Han
    name as `_shows` reads it."""
    if not _shows(name, text, words):
        return False
    if not name.isascii():
        return True
    body = r"\s+".join(re.escape(word) for word in name.split())
    return any(not _dated(name, text, match.start(), match.end())
               for match in re.finditer(rf"(?<![0-9A-Za-z]){body}(?![0-9A-Za-z])", text,
                                        re.IGNORECASE))


def _longer_name(name: str, full: str) -> bool:
    """Whether ``full`` is a longer name with ``name`` as one part ("Michael Jordan" for
    "Jordan", "陈小明" for "小明")."""
    return name.casefold() != full.casefold() and token_contained(name, full)


def _masked(text: str, longer: Sequence[str]) -> str:
    """``text`` with every occurrence of the ``longer`` names of other people
    (`StatementForget._longer_names`) blanked at its own length, so a part of one is no
    showing of the forgotten name."""
    for full in longer:
        body = r"\s+".join(re.escape(word) for word in full.split())
        first, last = full[0], full[-1]
        text = re.sub(
            rf"{'(?<![0-9A-Za-z])' if first.isascii() else ''}{body}"
            rf"{'(?![0-9A-Za-z])' if last.isascii() else ''}",
            lambda match: " " * len(match.group()), text, flags=re.IGNORECASE)
    return text


def _owned_tie(words: Set[str], statement: str) -> bool:
    """Whether ``statement`` names the user's own tie of one of ``words`` ("The user's cat",
    "their cat", 我的猫)."""
    return any(re.search(rf"{_OWNERS_EN}\s+(?:[A-Za-z-]+\s+){{0,2}}?{re.escape(word)}"
                         rf"(?![A-Za-z])", statement, re.IGNORECASE) if word.isascii()
               else re.search(rf"(?:{_OWNERS_ZH}|{_OWNERS_EN}\s+){re.escape(word)}",
                              statement)
               for word in words)


_CALLED: Final = re.compile(
    r"(?:叫|喊|称呼|管)(?:她|他|它)(?:做|为|作)?"
    r"(?P<han>(?:(?![的了是在和与跟也都很因为所以但而她他它们我你])"
    r"[\u3400-\u4dbf\u4e00-\u9fff]){2,6})"
    r"|\b(?:call|calls|called)\s+(?:her|him|it)\s+"
    r"(?P<latin>[A-Z][\w'\u2019-]*(?:\s+[A-Z][\w'\u2019-]*)?)")
"""Another name given to someone by a pronoun ("我们都叫她兰兰奶奶", "we call her Granny Lan")."""
_OWNERS_EN: Final = r"(?:\b(?:the\s+)?user['\u2019]s|\b(?:my|our|their))"
"""The user's possessive before a Latin tie word ("the user's cat", "my cat", "their cat")."""
_OWNERS_ZH: Final = r"(?:我家的?|我的?|用户的)"
"""The user's possessive before a Han tie word ("我的猫", "我家的猫", "我女儿", "用户的猫")."""


_TIE_WORDS: Final = "|".join(sorted((re.escape(word) for word in TIES), key=len, reverse=True))


def _tied_before(statement: str, name: str) -> bool:
    """Whether ``statement`` names the user's tie right before ``name``: "The user's colleague
    志强", "the user's older male cousin Tom", "我同事志强"."""
    body = re.escape(name)
    return re.search(rf"{_OWNERS_EN}\s+(?:[A-Za-z-]+\s+){{0,2}}?(?:{_TIE_WORDS})\s*{body}"
                     rf"(?![0-9A-Za-z])|{_OWNERS_ZH}(?:{_TIE_WORDS}){body}", statement,
                     re.IGNORECASE) is not None


def _refers(words: Set[str], texts: Sequence[str], name: str) -> bool:
    """Whether one of ``texts`` names the user's own tie of ``words`` (an entity's tie words,
    `StatementForget._tie_words`): a Latin word after the user's possessive with at most two
    words between ("the user's cat", "their old cat") or right before the name ("cat Bob"), a
    Han word after 我, 我的, 我家的 or 用户的 or right before or after the name ("我的猫", "猫芝
    麻"). "The user's coworker Max has a cat." names no cat of the user's: its "cat" is the
    coworker's."""
    for word in words:
        body = re.escape(word)
        if word.isascii():
            pattern = re.compile(
                rf"{_OWNERS_EN}\s+(?:[A-Za-z-]+\s+){{0,2}}?{body}(?![A-Za-z])"
                rf"|(?<![A-Za-z]){body}\s+{re.escape(name)}(?![0-9A-Za-z])", re.IGNORECASE)
        else:
            # A Han tie word may stand in an English statement too ("The user's 表哥 志强").
            pattern = re.compile(rf"{_OWNERS_ZH}{body}|{_OWNERS_EN}\s+{body}"
                                 rf"|{body}\s?{re.escape(name)}|{re.escape(name)}{body}")
        if any(pattern.search(text) for text in texts):
            return True
    return False


def _covered(start: int, end: int, held: Sequence[tuple[int, int]]) -> bool:
    """Whether the union of ``held`` ranges covers ``[start, end)``."""
    position = start
    for low, high in sorted(held):
        if low > position:
            break
        position = max(position, high)
        if position >= end:
            return True
    return position >= end


def _with_names(spans: Sequence[QuoteSpanV1], names: Sequence[QuoteSpanV1],
                ) -> list[QuoteSpanV1]:
    """``spans`` and the forgotten names' own spans (`StatementForget._name_token_spans`)
    withheld with them, at most ``MAX_RESTATING_SPANS``.

    Raises:
        QuoteMemError: ``budget_exceeded`` when the bound would leave out a name's span: the forget
            would report itself done with the forgotten name still readable.
    """
    merged = list({_key(span): span for span in (*spans, *names)}.values())
    kept = merged[:MAX_RESTATING_SPANS]
    if len(merged) > len(kept) and {_key(span) for span in names} - {_key(span)
                                                                      for span in kept}:
        raise QuoteMemError("budget_exceeded",
                            "The forget would leave a forgotten name readable past its bound")
    return kept


def _rounds(calls: Sequence[Callable[[], ForgetPickOutput | None]]) -> list[
        ForgetPickOutput | None]:
    """Run one round's votes concurrently, each in a copy of the caller's context."""
    with ThreadPoolExecutor(max_workers=len(calls),
                            thread_name_prefix="quotemem-forget-pick") as pool:
        futures = [pool.submit(contextvars.copy_context().run, call) for call in calls]
        return [future.result() for future in futures]


@overload
def _together[A, B](first: Callable[[], A], second: Callable[[], B], /) -> tuple[A, B]: ...


@overload
def _together[A, B, C](first: Callable[[], A], second: Callable[[], B],
                       third: Callable[[], C], /) -> tuple[A, B, C]: ...


def _together(*calls: Callable[[], object]) -> tuple[object, ...]:
    """Run independent steps of one forget at once, each in a copy of the caller's context,
    and return their results in order; the first error propagates."""
    with ThreadPoolExecutor(max_workers=len(calls),
                            thread_name_prefix="quotemem-forget-wave") as pool:
        futures = [pool.submit(contextvars.copy_context().run, call) for call in calls]
        return tuple(future.result() for future in futures)


type _Picked = tuple[list[_Candidate], list[QuoteSpanV1], int, list[_Judged], set[str]]


_CLAUSE_MARKS: Final = re.compile(r"[,;，；、。！？!?\n]|\.(?=\s|$)")
_CONJUNCTS: Final = re.compile(r"\s+(?:and|but|while|whereas)\s+|[和并而且但]", re.IGNORECASE)
"""What joins two conjuncts of one clause, each read on its own as an echo (``_echoes``)."""


def _key(span: QuoteSpanV1) -> tuple[str, int, int]:
    return span.evidence_id, span.byte_start, span.byte_end


def _meets(span: QuoteSpanV1, cover: Sequence[QuoteSpanV1]) -> bool:
    return any(span.evidence_id == other.evidence_id and span.byte_start < other.byte_end
               and other.byte_start < span.byte_end for other in cover)


def _rewordable(row: _Candidate, cover: Sequence[QuoteSpanV1]) -> bool:
    """Whether withholding ``cover`` leaves a memory no whole support group, so its present
    wording is read no more, and still some readable byte of a quote, which a new wording
    may rest on."""
    held: dict[str, list[tuple[int, int]]] = {}
    for span in cover:
        held.setdefault(span.evidence_id, []).append((span.byte_start, span.byte_end))
    return bool(row.groups) and all(
        any(_meets(row.quotes[index], cover) for index in group) for group in row.groups
    ) and not all(_covered(span.byte_start, span.byte_end, held.get(span.evidence_id, []))
                  for span in row.quotes)


def _mostly_own(text: str) -> bool:
    """Whether a text is mostly Han, Kana or Hangul (more such characters than Latin words)."""
    tokens = token_seq(text)
    own = sum(1 for token in tokens if own_token(token))
    return own > sum(1 for token in tokens if not own_token(token) and token.isascii()
                     and any(character.isalpha() for character in token))


@dataclass(frozen=True)
class MemorySentences:
    """What a complete forget by memory id withholds besides the memory's own quotes
    (``StatementForget.memory_sentences``), and every sentence its judge read."""

    spans: tuple[QuoteSpanV1, ...]
    judged: tuple[_Judged, ...]


@dataclass(frozen=True)
class MemoryPlan:
    """What a complete forget by memory id takes and withholds (``StatementForget.memory_plan``):
    the memories its forget takes besides it (``chain``), the sentences it withholds and the
    nominations the forget judge confirmed (``allowed``)."""

    chain: tuple[str, ...]
    sentences: MemorySentences
    allowed: frozenset[str]


@dataclass(frozen=True)
class _PlanRecord:
    """What a complete prepare recorded of its plan (``ForgetKeptPlanV1``): whether a record
    is ``held``, the memories it names, the targets it read past, what the judge confirmed
    it takes (``allowed``), whether its request asked a nickname, the memories it rewords
    (``rewordings``) and what its request asked (``asked``, ``name_asked``)."""

    listed: frozenset[str]
    past: frozenset[str]
    allowed: frozenset[str] = frozenset()
    held: bool = False
    nickname: bool = False
    residue: frozenset[str] = frozenset()
    modifier: bool = False
    name_or_all: bool = False
    rewordings: tuple[ForgetRewordPlanV1, ...] = ()
    asked: frozenset[ContentBindingDigest] = frozenset()
    name_asked: bool = False


_ASKED_WORDS: Final = frozenset(content_terms(
    "forget remember old previous former address number name phone email "
    "忘掉 忘记 旧地址 地址 号码 名字 手机号"))
"""Words a request's own part shares with a replacement's statement that name no value."""


def _value_words(text: str) -> set[str]:
    """The value words of a replacement ("I live at 27 Birch Lane now", "我现在住在银杏路6号"): its
    terms that hold a digit, an @ or a capital, and its Han character pairs, less those that
    name no value (现在, 住在, ...)."""
    words = {term for term in content_terms(text)
             if any(char.isdigit() or char == "@" for char in term) or term in {
                 word.lower() for word in re.findall(r"\b[A-Z][a-z]+\b", text)}
             or all("一" <= char <= "鿿" for char in term)}
    return words - _VALUE_STOP


_VALUE_STOP: Final = frozenset({"现在", "目前", "如今", "住在", "我现", "在住", "搬到", "我们",
                                "新的", "我的", "的新", "已经"})


_LIVES_NOW = re.compile(r"\b(?:lives|live|is living|are living|resides|reside)\b",
                        re.IGNORECASE)
"""A statement of where someone lives now: never taken by a request for where the user
lived (`lived_value`, `StatementForget._kept_homes`)."""


def _overlaps(span: QuoteSpanV1, quotes: Sequence[QuoteSpanV1]) -> bool:
    """Whether ``span`` shares bytes with one of ``quotes``."""
    return any(span.evidence_id == quote.evidence_id and span.byte_start < quote.byte_end
               and quote.byte_start < span.byte_end for quote in quotes)


def _in_context(asked: str, words: str) -> str:
    """What the judge reads of a request: its asked part, with the plugin's whole words beside
    it when they say more ("my old address (said as: "my old address since I moved to 27 Birch
    Lane")"): the asked part is what to forget, the rest only tells which value is old."""
    return asked if asked == words.strip() else f'{asked} (said as: "{words.strip()}")'


_GAVE_UP: Final = re.compile(
    r"卖|辞|离开|搬走|搬离|搬出|退掉|退了|"
    r"\b(?:sold|traded in|left|quit|gave up|got rid of|moved out of)\b", re.IGNORECASE)
"""What tells that a value was given up, never that one was taken up ("换了一辆", "got a")."""
_CHANGE_CUT: Final = re.compile(_CLAUSE_MARKS.pattern + r"|\s+(?:and|but|then)\s+", re.IGNORECASE)


def _given_up(text: str, low: int, high: int, gone: Sequence[str], stated: Sequence[str],
              ) -> list[tuple[int, int]]:
    """The clauses of the sentence a kept memory's quote (``text[low:high]``) lies in that tell
    a value of ``gone`` was given up ("我把思域卖了" of "我把思域卖了，换了一辆蓝色的
    卡罗拉", "I sold my Civic" of "I sold my Civic and got a Corolla"): each has a give-up
    verb, names a value of ``gone`` and none of what kept memories state (``stated``), and the
    quote lies in another clause of the sentence too, one with no such verb. Any other sentence
    gives none."""
    found: list[tuple[int, int]] = []
    for start, end in sentences(text):
        if high <= start or end <= low or is_request(text[start:end], complete=True):
            continue
        cuts = [(cut.start(), cut.end()) for cut in _CHANGE_CUT.finditer(text, start, end)]
        clauses = [(at, stop) for at, stop in zip(
            [start, *(stop for _, stop in cuts)], [*(at for at, _ in cuts), end], strict=True)
            if text[at:stop].strip()]
        if any(at < high and low < stop and not _GAVE_UP.search(text[at:stop])
               for at, stop in clauses):
            found += [(stop - len(text[at:stop].lstrip()), at + len(text[at:stop].rstrip()))
                      for at, stop in clauses if _GAVE_UP.search(text[at:stop])
                      and names_value(text[at:stop], gone)
                      and not names_value(text[at:stop], stated)]
    return found


def _states_new(statement: str, words: str) -> bool:
    """Whether a memory states the new value a request's words told after its asked part
    ("The user lives at 27 Birch Lane." of "my old address since I moved to 27 Birch Lane", or
    of the sentence "Forget my old address, I live at 27 Birch Lane now."): such a memory is
    never one the request takes. A value told whole ("555-9876", "jon@example.com") is stated
    only whole: the part another value shares ("555" of "The user's phone number is
    555-1234.", the old one) states nothing new."""
    reach = request_reach(words)
    rest = words[reach:] if reach is not None else words.strip()[len(asked_part(words)):]
    shared = (_value_words(rest) - _ASKED_WORDS) & _value_words(statement)
    partial = {term for token in _whole_values(rest) - _whole_values(statement)
               for term in content_terms(token)}
    return bool(shared - partial)


def _whole_values(text: str) -> set[str]:
    """The Latin values of a text as written, each whole: "555-9876", "27", "jon@example.com"."""
    return {token.strip(".+_-").lower() for token in re.findall(r"[A-Za-z0-9@.+_-]+", text)
            if any(char.isdigit() or char == "@" for char in token)}


def _naming_clauses(span: QuoteSpanV1, text: str, statements: Sequence[str],
                    ) -> list[QuoteSpanV1]:
    """The clauses of a sentence (``text`` of ``span``), cut at clause marks in any script,
    that name a value of one of ``statements`` as replaced or ended
    (``forget_admission.names_replaced``), each cut to its parts naming the value when the
    rest of it names none (``forget_admission.naming_parts``): "不再打羽毛球" of "搬家后我改去
    西湖附近的攀岩馆攀岩了，不再打羽毛球。", "instead of playing badminton" of "Since the move I go
    climbing at a gym near West Lake instead of playing badminton.". A complete forget
    withholds them whatever its pick decided."""
    found: list[QuoteSpanV1] = []
    left: set[str] = set()
    start = 0
    for end in [*(match.start() for match in _CLAUSE_MARKS.finditer(text)), len(text)]:
        clause = text[start:end]
        lead = start + len(clause) - len(clause.lstrip())
        body = clause.strip()
        if body and any(names_replaced(body, statement, unsure=False)
                        for statement in statements):
            for low, high in naming_parts(body, statements, strict=True):
                first = span.byte_start + len(text[:lead + low].encode("utf-8"))
                found.append(span.model_copy(update={
                    "byte_start": first,
                    "byte_end": first + len(body[low:high].encode("utf-8"))}))
                if _han(body[low:high]) and len(body[low:high]) > 1:
                    left.add(body[low:high])
        start = end + 1
    # A Chinese place a move left goes wherever the sentence names it, in its other clauses
    # too: "上海" of "我从北京搬到上海，又从上海搬到了杭州".
    for value in left:
        for match in re.finditer(re.escape(value), text):
            first = span.byte_start + len(text[:match.start()].encode("utf-8"))
            part = span.model_copy(update={
                "byte_start": first, "byte_end": first + len(value.encode("utf-8"))})
            if _key(part) not in {_key(other) for other in found}:
                found.append(part)
    return found


def _merged_sentences(
    first: _Picked, later: _Picked, quotes: Sequence[QuoteSpanV1],
) -> tuple[list[QuoteSpanV1], list[_Judged]]:
    """A complete prepare's sentences from both of its waves (`_complete_picks`): the forget
    requests first, then the picked sentences in position order, ``MAX_RESTATING_SPANS`` in
    all, and every sentence judged. A sentence the first wave picked before the targets were
    known that a target quotes is left to that target's own forget, as the second wave
    leaves it."""
    def key(span: QuoteSpanV1) -> tuple[str, int, int]:
        return span.evidence_id, span.byte_start, span.byte_end

    held: dict[str, list[tuple[int, int]]] = {}
    for span in quotes:
        held.setdefault(span.evidence_id, []).append((span.byte_start, span.byte_end))
    taken = {key(entry.span) for entry in (*first[3], *later[3]) if entry.role is None}

    def quoted(span: QuoteSpanV1) -> bool:
        return key(span) not in taken and any(
            span.byte_start < high and low < span.byte_end
            for low, high in held.get(span.evidence_id, ()))

    spans: dict[tuple[str, int, int], QuoteSpanV1] = {}
    for span in (*first[1], *later[1]):
        if not quoted(span):
            spans.setdefault(key(span), span)
    judged: dict[tuple[str, int, int], _Judged] = {}
    for entry in (*first[3], *later[3]):
        if not quoted(entry.span) and (key(entry.span) not in judged or entry.target):
            judged[key(entry.span)] = entry
    requests = [spans[found] for found in spans if found in taken]
    picked = sorted(found for found in spans if found not in taken)
    return ([*requests, *(spans[found] for found in picked)][:MAX_RESTATING_SPANS],
            list(judged.values()))


class StatementForget:
    """Prepare and confirm forget by statement for one workspace.

    Args:
        writer: The single Writer.
        owner: The workspace's forget owner; every target goes through it.
        search: The runtime's Search; None refuses ``prepare`` (``service_unhealthy``).
        provider: The workspace's structured inference provider (the judge's).
        clock: The trusted millisecond clock.
        judge: The judge binding whose provider, model, effort, timeout and retries
            the pick's calls use; None reads the workspace's released extraction policy
            (the memory judge of that pipeline).
        complete: Whether forgets leave nothing that states the fact readable
            (``explicit-request-v2``); None reads the released extraction
            policy's in-chat forget rule.
        embedder: The workspace's embedding provider: a complete prepare embeds its
            statement once for the candidates in another language; None finds them by
            words only.
    """

    def __init__(
        self, writer: Writer, owner: ForgetOwner, *, search: SearchOperation | None,
        provider: StructuredInferenceProvider | None, clock: Callable[[], int],
        judge: InferenceBindingV1 | None = None, complete: bool | None = None,
        embedder: EmbeddingProvider | None = None,
    ) -> None:
        self._writer, self._owner, self._search = writer, owner, search
        self._provider, self._clock, self._judge = provider, clock, judge
        self._complete_rule = complete
        self._embedder = embedder
        if complete and embedder is not None:
            # A forget told to forget completely keeps its runs' remainders as a complete
            # workspace's service does (``ForgetService``).
            owner.embed_remainders_with(
                lambda texts: workspace_embedding(writer, embedder, texts))
        self._phrase_cache: dict[tuple[str, str], _Phrase] = {}
        self._wording_cache: dict[tuple[str, str], tuple[ForgetRewordV1, ...]] = {}

    # Prepare ---------------------------------------------------------------------------
    def prepare(
        self, scope: ScopeV1, grant: PrincipalGrant,
        request: MemoryForgetStatementPrepareRequestV1,
    ) -> MemoryForgetStatementPreparedV1:
        """Search, pick and return the plan to confirm, or ``nothing_matched``.

        Raises:
            QuoteMemError: Search's own errors (``index_rebuilding`` under a fence,
                ``delivery_busy``); ``provider_unavailable`` when a pick decision got
                fewer than two answered votes; ``service_unhealthy`` without a
                Search or provider.
        """
        if self._search is None or self._provider is None:
            raise QuoteMemError("service_unhealthy", "Forget by statement has no Search or judge")
        if self._complete():
            return self._prepare_complete(scope, grant, request)
        hits = self._hits(scope, request.user_id, request.statement, MAX_FORGET_CANDIDATES)
        candidates, erase_epoch = self._candidates(scope, hits)
        if not candidates:
            return MemoryForgetStatementPreparedV1(
                status="nothing_matched", confirmation=None, expires_at=None,
                candidates=[], decision=None)
        pick = pick_targets(request.statement, [row.statement for row in candidates],
                            self._sender(), rounds=_rounds)
        chosen = sorted(pick.targets)
        rows = [ForgetStatementCandidateV1(
            memory_id=row.memory_id, revision_id=row.revision_id, statement=row.statement,
            role=pick.roles[index], target=index in chosen,
        ) for index, row in enumerate(candidates)]
        decision = _decision(pick, request.statement)
        if not chosen:
            return MemoryForgetStatementPreparedV1(
                status="nothing_matched", confirmation=None, expires_at=None,
                candidates=rows, decision=decision)
        expires_at = self._clock() + INTENT_TTL_MS
        targets = tuple(ForgetStatementTargetV1(
            memory_id=candidates[index].memory_id,
            expected_revision_id=candidates[index].revision_id,
            expected_content_hash=IntegrityDigest(candidates[index].content_hash),
        ) for index in chosen)
        read = tuple(ForgetStatementCandidateRefV1(memory_id=row.memory_id,
                                                   revision_id=row.revision_id)
                     for row in candidates)
        binding = self._binding(scope, grant, request.client_session_id, request.request_id,
                                targets, read, erase_epoch, expires_at)
        confirmation = MemoryForgetStatementRequestV1(
            schema_version=1, user_id=request.user_id,
            client_session_id=request.client_session_id, request_id=request.request_id,
            targets=list(targets), candidates=list(read), erase_epoch=erase_epoch,
            expires_at=expires_at,
            plan_binding=binding,
        )
        return MemoryForgetStatementPreparedV1(
            status="confirmation_required", confirmation=confirmation, expires_at=expires_at,
            candidates=rows, decision=decision)

    def _prepare_complete(
        self, scope: ScopeV1, grant: PrincipalGrant,
        request: MemoryForgetStatementPrepareRequestV1,
    ) -> MemoryForgetStatementPreparedV1:
        """A complete prepare (``explicit-request-v2``; module docstring): memories and
        sentences, by words and by vector, the judge's picks less what states another fact.

        The statement's Search and its embedding run together; the memory pick,
        the clause pick of every candidate that splits into clauses and the sentence picks
        of the statement's own words run together; only the picks that need the targets
        (the other memories sharing a target's words, the sentences a target's words or
        quotes find that were not judged yet) follow. The picks are the in-chat forget's
        (`_matched`, `_pick_sender`), so both paths of one request decide alike;
        after an in-chat forget of the same request of the session, only what
        changed since it read the scope is read, and ``earlier`` reports what it did.
        """
        # A request sentence sent as it was said ("Forget my old address, I live at 27 Birch
        # Lane now.") is read as the in-chat forget reads it: its words of the
        # fact, without the asking words or a clear replacement after them.
        asked = requested(request.statement, complete=True)
        statement = asked[0] if len(asked) == 1 else request.statement
        sender = self._pick_sender()
        handled = self._handled_key(scope, request.client_session_id, statement)
        earlier = self._earlier(scope, handled)
        entry = self._entry(scope, key=handled)
        # The plugin's whole statement keys the request; the judge and the
        # matching read only what it asks to forget, never the new value told after it
        # ("my old address" of "my old address since I moved to 27 Birch Lane"); the first pick
        # sees the whole words beside it, which tell which stored value is the old one.
        words = statement
        statement = asked_part(statement)
        # A request asking a name: a target tying a name to someone by a role gives it too.
        asks = asks_name(statement)
        # A request for a detail of a thing (its colour, its breed, the place the user works
        # or which place of a kind) forgets the value, never the thing.
        detail = self._detail(scope, statement)
        # A colour or a breed is a value of the thing itself: the names its memories give
        # are not what the request forgets.
        own_names = not attribute(detail)
        # A request for a nickname forgets another name, never the name (`_names`).
        nick = nickname_request(statement)
        # An in-chat forget of the same request already judged the scope: only what changed
        # after it is read, so the two paths never contradict each other. It
        # read the whole scope itself unless a forget by statement had settled it first. A
        # forget that matched nothing settles nothing: the user asking again is read whole.
        after = None if entry is None or "forgotten" not in (entry.statement, entry.in_chat) \
            else max((seq for seq in (entry.statement_seq, entry.chat_seq) if seq is not None),
                     default=None)
        hits, (phrases, embedded) = _together(
            lambda: self._hits(scope, request.user_id, probe(statement), CHAT_SEARCH_TOP_K),
            lambda: self._phrases([probe(statement)]))
        phrase = phrases[0] if phrases else None
        candidates, erase_epoch, read_seq = self._matched(
            scope, hits, phrase, self._memories(scope) if phrase is not None else [],
            after=after)
        summaries = {row.memory_id for row in self._summaries(scope)}
        indexed = dict(enumerate(candidates))

        def first_pick() -> ForgetPickV1 | None:
            return (pick_targets(_in_context(statement, words),
                                 [row.statement for row in candidates], sender,
                                 rounds=_rounds, minimal=False) if candidates else None)

        pick, (early_roles, early_calls), first = _together(
            first_pick, lambda: self._clause_roles(statement, indexed, sender),
            lambda: self._complete_picks(scope, [statement], {}, summaries, sender,
                                         phrases=phrases, veto=exclusion, after=after))
        calls = early_calls + first[2]
        chosen: list[int] = []
        excluded: dict[int, Exclusion] = {}
        bundled: set[int] = set()
        if pick is not None:
            calls += pick.calls
            spared = _spared(candidates, pick.roles)
            chosen = sorted({*pick.targets,
                             *(i for i, role in enumerate(pick.roles) if role in _PARTS)}
                            - spared)
            excluded = {i: why for i in chosen
                        if (why := exclusion(statement, candidates[i].statement)) is not None}
        picked = {i: candidates[i] for i in chosen if i not in excluded}
        targets = {row.memory_id: row for row in picked.values()}
        changed = None if after is None else {
            row.memory_id for row in self._memories(scope, after=after)}
        records = self._request_records(scope, [statement])
        for row in records:
            if changed is None or row.memory_id in changed:
                targets.setdefault(row.memory_id, row)
        # A request to forget a person's or an animal's name takes every memory giving that
        # name, whatever the judge picked (`_referent_namers`).
        named_rows, given_names, uncertain, within = self._referent_namers(scope, [statement])
        for row in named_rows:
            targets.setdefault(row.memory_id, row)
        # A request for an old value takes the values a later one replaced, and one for a
        # person as a whole every memory of that one person (`_old_values`, `_whole_people`).
        old_rows, old_calls = self._old_values(
            scope, [(statement, candidates, words if tells_new(words) else request.statement)],
            sender)
        calls += old_calls
        ruled_rows = [row for row in (*old_rows, *self._whole_people(scope, [statement]),
                                      *self._detail_rows(scope, detail))
                      if changed is None or row.memory_id in changed]
        for row in ruled_rows:
            targets.setdefault(row.memory_id, row)
        # Nor does it take a pick the request does not ask for: a fact not showing the name,
        # or the user's having the one named (`_unnamed_picks`).
        ruled_ids = {row.memory_id for row in (*named_rows, *records, *ruled_rows)}
        for memory_id in self._unnamed_picks(
                scope, targets, {row.memory_id for row in picked.values()} - ruled_ids,
                given_names, [statement]):
            del targets[memory_id]
        later_roles: dict[int, list[str]] = {}
        later: _Picked = ([], [], 0, [], set())
        if targets:
            missing = {i: row for i, row in picked.items() if i not in early_roles}
            judged_keys = frozenset(
                (entry.span.evidence_id, entry.span.byte_start, entry.span.byte_end)
                for entry in first[3])
            (later_roles, later_calls), later = _together(
                lambda: (self._clause_roles(statement, missing, sender) if missing
                         else ({}, 0)),
                lambda: self._complete_picks(
                    scope, [statement], targets, summaries, sender, phrases=phrases,
                    veto=exclusion, skip=judged_keys, after=after,
                    against_request=detail is not None))
            calls += later_calls + later[2]
        roles = {**early_roles, **later_roles}
        bundled = {index for index in picked if index in roles
                   and any(role in _PARTS for role in roles[index])
                   and "unrelated" in roles[index]}
        for row in later[0]:
            targets.setdefault(row.memory_id, row)
        spans, judged = _merged_sentences(first, later, [span for row in targets.values()
                                                         for span in row.quotes])
        if lived_value(statement) is not None:
            # Where the user lived withholds no sentence a current home's memory quotes, so
            # that memory stays readable (fail closed).
            held = self._home_quotes(scope)
            homes = {_key(entry.span) for entry in judged
                     if entry.target and _overlaps(entry.span, held)}
            spans = [span for span in spans if _key(span) not in homes]
            judged = [replace(entry, target=False) if _key(entry.span) in homes else entry
                      for entry in judged]
        # A memory naming the asked value as one it replaced or ended ("... instead of playing
        # badminton"), about whom the fact is (``_naming_subjects``) and about no other person
        # (``names_replaced``), may still tell it whatever the pick decided: a word rule only
        # nominates it, and it is taken, as a bundle of the request's fact (its
        # other value told again), once the forget judge confirms it states or presupposes the
        # fact (``_confirmed``); the prepare names each among its targets. So may a past
        # memory, which is no candidate (``_naming``).
        replacing = self._owner.forgets_completely()
        subjects = self._naming_subjects(scope, targets) if replacing else frozenset[str]()
        about = self._subjects(scope, [row.memory_id for row in candidates]) if replacing \
            else {}
        nominated = {index: row for index, row in enumerate(candidates if replacing else [])
                     if row.memory_id not in targets and index not in excluded
                     and about.get(row.memory_id) in subjects
                     and names_replaced(row.statement, statement, unsure=False)}
        past_rows = self._naming(scope, [statement], targets, subjects) if replacing else []
        named = {_key(entry.span): parts for entry in (judged if replacing else [])
                 if not entry.target and entry.excluded is None
                 and (parts := _naming_clauses(entry.span, entry.text, [statement]))}
        if named:
            judged = [replace(entry, target=True) if _key(entry.span) in named else entry
                      for entry in judged]
            spans = list({_key(span): span for span in (
                *spans, *(part for parts in named.values() for part in parts))}.values())[
                    :MAX_RESTATING_SPANS]
        if replacing:
            # A sentence the judge never read (one sharing only the value word) still tells
            # the asked value when it names it as replaced (``_naming_sentences``).
            spans = list({_key(span): span for span in (
                *spans, *self._naming_sentences(scope, [statement]))}.values())[
                    :MAX_RESTATING_SPANS]
        # What a word rule nominates is taken only once the judge confirms it; without an
        # answer, or with a contested one, the prepare refuses, to be asked again
        # (``provider_unavailable``). What the targets' links take by the rule is not asked.
        allowed = set[str]()
        declined: list[str] = []
        if replacing and (targets or spans or nominated or past_rows):
            ruled = self._ruled(scope, targets, spans, {
                candidates[index].memory_id for index in bundled}, [statement])
            trial = {**targets, **{row.memory_id: row for row in (*nominated.values(),
                                                                     *past_rows)}}
            trial_bundles = {*(candidates[index].memory_id for index in (*bundled, *nominated)),
                             *(row.memory_id for row in past_rows)}
            nominees = list(dict.fromkeys(
                nominee for nominee in (
                    *(row.memory_id for row in (*nominated.values(), *past_rows)),
                    *self._nominations(scope, trial, spans, None if trial or not spans
                                       else self._seed(scope, statement, spans, phrase),
                                       bundled=trial_bundles, words=[statement]))
                if nominee not in ruled))
            confirmed = self._confirmed(scope, [statement], nominees, sender)
            if confirmed is None:
                raise QuoteMemError("provider_unavailable",
                                    "The forget judge could not settle what the forget takes")
            allowed, judge_calls = confirmed
            calls += judge_calls
            declined = [nominee for nominee in nominees if nominee not in allowed]
        for index, row in nominated.items():
            if row.memory_id in allowed:
                targets[row.memory_id] = row
                bundled.add(index)
        named_past = [row.memory_id for row in past_rows if row.memory_id in allowed]
        targets.update((row.memory_id, row) for row in past_rows if row.memory_id in allowed)
        # A judge's pick naming another entity than the one a name request resolved to goes
        # back to the judge's kept memories: neither it nor its name goes; so
        # does a later pick the request does not ask for (`_unnamed_picks`).
        for memory_id in {*self._other_namers(scope, targets, given_names),
                          *self._unnamed_picks(
                              scope, targets, {row.memory_id for row in later[0]} - ruled_ids,
                              given_names, [statement])}:
            del targets[memory_id]
        # Where the user lived takes no current home, and no request takes the new value it
        # told (`_kept_homes`, `_states_new`).
        for memory_id in {*self._kept_homes(scope, targets, [statement],
                                            [words, request.statement]),
                          *self._current_values(scope, [statement], targets,
                                                {row.memory_id for row in old_rows})}:
            del targets[memory_id]
        # A request for the name of a place or organisation of someone's ("the name of my
        # daughter's school") asks that one's names alone: the name rules take no name of
        # its owner ("Ruby"), whatever the targets give besides (`_kind_names`).
        if asks and not nick and not given_names:
            given_names = self._kind_names(scope, [statement], list(targets))
        # A bundle goes for the user's words, not for its own value (``ForgetOwner.chains``).
        bundle_ids = {*(candidates[index].memory_id for index in bundled), *named_past}
        # An old value's own clause of the sentence telling the change goes with it.
        spans = list({_key(span): span for span in (*spans, *self._given_up_clauses(
            scope, [statement], targets, spans))}.values())[:MAX_RESTATING_SPANS]
        if replacing and (targets or spans):
            # The assistant's echoes of what is forgotten state it too.
            spans = list({_key(span): span for span in (*spans, *self._echoes(
                scope, [*(span for row in targets.values() for span in row.quotes), *spans],
                self._echoed(scope, [statement], targets, bundle_ids)))}.values())[
                    :MAX_RESTATING_SPANS]
        # What tells the request's referent: the request, what the forget
        # takes and the memories the judge found to name or state it.
        context = [*targets, *(row.memory_id for index, row in enumerate(candidates)
                               if pick is not None and pick.roles[index] in _TELLING)]
        sources = [statement, *(row.statement for row in self._rows(scope, context))]
        # Only a request to forget a name lets a tie stand for it (`_reword_plans`): "forget
        # my wife's name", "forget Marcus's name", never "forget that Sofia is my wife".
        tie_names = asks
        wordings: dict[str, ForgetRewordV1] = {}
        bundle_parts: dict[str, list[QuoteSpanV1]] = {}
        form_parts: dict[str, list[QuoteSpanV1]] = {}
        if replacing and targets:
            # A target that tells the forgotten detail beside a fact of its own ("The user plays
            # badminton at 星河体育馆 on weekends." for "which sports hall I play at") is reworded
            # without it; only the detail's words are withheld, as sentences of the request.
            target_names = self._forgotten_names(
                scope, given_names, self._names(scope, list(targets), ties=asks, aliases=nick)
                if own_names else frozenset(), uncertain=uncertain, within=within)
            # A target showing the asked colour or breed of the thing is reworded without it,
            # its other facts and the thing's name kept (`_detail_plans`).
            attributes, used = self._detail_plans(scope, statement, detail, list(targets),
                                                  sender)
            calls += used
            plans, used = self._reword_plans(
                scope, statement, [memory_id for memory_id in targets
                                   if memory_id not in attributes],
                target_names, sources, context, sender, ties=tie_names,
                exists={row.statement.casefold() for row in self._memories(scope)},
                detail=detail)
            calls += used
            located: dict[str, ForgetRewordPlanV1] = {}
            for row in self._rows(scope, [*plans, *attributes]):
                if row.memory_id in attributes:
                    plan, forms = attributes[row.memory_id]
                    parts = self._form_spans(row.quotes, forms)
                    if parts and _rewordable(row, parts):
                        # Its value goes from every other readable sentence telling it of the
                        # thing, as its forget whole would have withheld it there too.
                        located[row.memory_id] = plan
                        bundle_parts[row.memory_id] = [
                            *parts, *self._detail_token_spans(scope, detail, forms)]
                    continue
                plan, removed = plans[row.memory_id]
                parts = self._value_spans(row.quotes, self._han_located(
                    row.quotes, removed, [statement, *self._quote_texts([
                        span for other in targets.values() if other.memory_id != row.memory_id
                        for span in other.quotes])]))
                # Every group of its support must lose a byte, or its present wording, which
                # shows the detail, would stay readable should its rewording fail.
                if plan.statement != row.statement and parts and _rewordable(row, parts):
                    # What it leaves out goes from every other readable sentence showing it,
                    # as its forget whole would have withheld it there too.
                    parts = [*parts, *self._value_token_spans(scope, [
                        value for value in removed if name_norm(value) not in {
                            name_norm(name) for _, name in target_names}])]
                    located[row.memory_id], bundle_parts[row.memory_id] = plan, parts
            found_bundles, used = self._embedded(scope, located)
            embedded += used
            wordings.update(found_bundles)
        # A value the request asks of a target ("青柠科技" of "The user works at 青柠科技." for
        # "忘掉我在青柠科技工作过") goes from every readable sentence of the scope showing it,
        # whatever forgets the target, reworded or whole; a name request's
        # names go by the name rules instead.
        # Only the value an English target states, an employer or a thing, never a place or
        # time it holds at ("Xuhui" of "... at a club in Xuhui.", `names_value`), a place it
        # is in, goes to or comes from ("lives in Shanghai": a sentence sharing only the
        # place tells another fact), nor a name a naming verb gives.
        # A request asking a detail ("forget where I work") asks a target's own value too,
        # where the target holds at it ("a designer at Bluefin Labs").
        asked_spans = [] if asks or not replacing else self._value_token_spans(scope, [
            run for row in targets.values() if english(row.statement)
            for run in dict.fromkeys([
                *(run for run in asked_sources(statement, [row.statement])[1:]
                  if names_value(run, [row.statement])),
                *detail_values(row.statement, statement)])
            if run != row.statement and not re.search(
                rf"\b(?:in|to|from|near|around|named|called)\s+(?:the\s+)?{re.escape(run)}",
                row.statement, re.IGNORECASE)])
        # The values asked of a target naming another too: the forget shadow and the rewording
        # keep the other (`_asked_shadow`, ``ForgetOwner._kept_names``).
        asked_names = self._asked_shadow(statement, [row.statement for row in targets.values()])
        base_targets, base_spans = dict(targets), list(spans)
        base_bundled, base_bundle_ids = set(bundled), set(bundle_ids)
        rounds_left: int = MAX_REWORDINGS + 2
        planned_more = False
        left: list[str] = []
        taken_left = False
        seed: ForgetSeedV1 | None = None
        planned: tuple[ForgetStatementTargetV1, ...] = ()
        read: tuple[ForgetStatementCandidateRefV1, ...] = ()
        planned_ids: set[str] = set()
        bundles: list[str] = []
        names = frozenset[tuple[str, str]]()
        affected: list[ForgetStatementAffectedV1] = []
        admitted: list[ForgetRewordV1] = []
        plan_withheld: tuple[QuoteSpanV1, ...] = ()
        while True:
            # The plan with its reworded targets kept and their details withheld. Read again until
            # every rewording it plans is one its forgets write (``ForgetOwner.admit_rewordings``),
            # so the confirmation does what the prepare says.
            targets, spans = dict(base_targets), list(base_spans)
            bundled, bundle_ids = set(base_bundled), set(base_bundle_ids)
            for memory_id, parts in bundle_parts.items():
                if memory_id in wordings:
                    merged = list({_key(span): span for span in (*spans, *parts)}.values())
                    if len(merged) > MAX_RESTATING_SPANS:
                        # Not every sentence showing what it leaves out could be withheld: it
                        # goes whole, as before (fail closed).
                        del wordings[memory_id]
                        continue
                    del targets[memory_id]
                    bundled -= {index for index, row in enumerate(candidates)
                                if row.memory_id == memory_id}
                    bundle_ids.discard(memory_id)
                    spans = merged
            # Every form of what a reworded memory leaves out goes from its own quotes too,
            # another script's in parentheses as well ("罗晓东（Luo Xiaodong）").
            spans = list({_key(span): span for span in (*spans, *(
                span for memory_id in wordings if memory_id not in bundle_parts
                for span in form_parts.get(memory_id, ())))}.values())[:MAX_RESTATING_SPANS]
            # A past memory the plan leaves unreadable is taken whole when a later memory that
            # replaced it still names its value, so its forget takes that memory too
            # (``promoted``).
            listed_ids = {row.memory_id for row in candidates}
            promoted = [*named_past, *(row.memory_id for row in (*named_rows, *ruled_rows)
                                       if row.memory_id not in listed_ids
                                       and row.memory_id in targets), *self._promoted(
                scope, targets, spans, None if targets or not spans
                else self._seed(scope, statement, spans, phrase), about=[statement],
                bundled=bundle_ids, allowed=allowed)]
            # A memory the plan rewords names the value no more: it is not taken.
            for memory_id in promoted:
                if memory_id in wordings:
                    targets.pop(memory_id, None)
            promoted = [memory_id for memory_id in promoted if memory_id not in wordings]
            if not targets and not spans:
                break
            # Only sentences state the fact: its forget is prepared below, while the
            # statement's vector is at hand for its tombstone.
            seed = None if targets else self._seed(scope, statement, spans, phrase)
            planned = tuple(ForgetStatementTargetV1(
                memory_id=row.memory_id, expected_revision_id=row.revision_id,
                expected_content_hash=IntegrityDigest(row.content_hash),
            ) for row in list(targets.values())[:MAX_FORGET_TARGETS])
            reworded_targets = [memory_id for memory_id in bundle_parts if memory_id in wordings]
            if targets or reworded_targets:
                # Every other sentence showing a name the forget takes loses that name alone
                # (`_name_token_spans`), a reworded target's too.
                kept_names = [*(target.memory_id for target in planned
                                if target.memory_id not in bundle_ids), *reworded_targets]
                spans = _with_names(spans, self._name_token_spans(
                    scope, kept_names, self._forgotten_names(
                        scope, given_names, self._names(scope, kept_names, ties=asks,
                                                        aliases=nick,
                                                        every=asks and not nick)
                        if own_names else frozenset(), uncertain=uncertain, within=within),
                    self._sessions(scope, [*targets.values(), *self._rows(
                        scope, reworded_targets)], request.client_session_id), within))
            # The asked values' every occurrence too, but where the plan withholds it already:
            # the prepare refuses rather than leave one readable (``budget_exceeded``).
            spans = _with_names(spans, [span for span in asked_spans if not any(
                span.evidence_id == other.evidence_id and other.byte_start <= span.byte_start
                and span.byte_end <= other.byte_end for other in spans)])
            # A promoted target was read by the plan too, so the confirmation checks its head.
            read = tuple(ForgetStatementCandidateRefV1(memory_id=row.memory_id,
                                                       revision_id=row.revision_id)
                         for row in [*candidates, *(targets[memory_id] for memory_id in promoted
                                                    if memory_id not in listed_ids)])
            # A bundled target's forget tombstones the user's words.
            planned_ids = {target.memory_id for target in planned}
            bundles = [candidates[index].memory_id for index in sorted(bundled)
                       if candidates[index].memory_id in planned_ids]
            # A past memory naming the asked value bundles it with its own new value: it is
            # tombstoned by the user's words, so its new value may be told again.
            bundles += [memory_id for memory_id in named_past
                        if memory_id in planned_ids and memory_id not in bundles]
            names = self._forgotten_names(
                scope, given_names, self._names(
                    scope, [memory_id for memory_id in planned_ids if memory_id not in bundles],
                    ties=asks, aliases=nick, every=asks and not nick)
                if own_names else frozenset(), uncertain=uncertain, within=within)
            # A reworded target whose left-out words the plan could not withhold everywhere
            # goes whole (fail closed), and the plan is read again.
            held_keys = {_key(span) for span in spans}
            unheld = [memory_id for memory_id, parts in bundle_parts.items()
                      if memory_id in wordings
                      and any(_key(part) not in held_keys for part in parts)]
            if unheld:
                # Each pass drops one wording at least, so the loop ends.
                wordings = {memory_id: wording for memory_id, wording in wordings.items()
                            if memory_id not in unheld}
                continue
            affected, taken, admitted, plan_withheld = self._affected(
                scope, [target.memory_id for target in planned], spans, seed, promoted,
                bundled=set(bundles), words=[statement], allowed=allowed, names=given_names,
                uncertain=uncertain, tie_names=asks, aliases=nick, within=within,
                rewordings=list(wordings.values()), asked=asked_names, own_names=own_names)
            listed = {entry.memory_id for entry in affected}
            if replacing and (affected or declined) and not planned_more:
                # A memory the plan leaves unreadable or forgets for still showing a forgotten
                # name or value, and a later one it takes for naming the forgotten value, is
                # reworded without the forgotten detail when it tells another fact too;
                # a current one taken only for its link to what is
                # forgotten, which shows none of it, stays as it is.
                planned_more = True
                current = {row.memory_id for row in self._memories(scope)}
                chained = [memory_id for memory_id in taken if memory_id in current]
                # A nomination the judge declined that still names a value of the request or
                # of a target as the one it replaced or left ("The user moved from 上海 to 杭州
                # last month." beside a forgotten "The user lives in 上海."), and that the plan
                # would leave readable and unlisted, is reworded without the value, or taken
                # when it cannot be.
                stated = [statement, *(row.statement for row in targets.values())]
                asked = asked_sources(statement, stated[1:])
                left = [row.memory_id for row in self._rows(scope, declined)
                        if row.memory_id in current and row.memory_id not in listed
                        and row.memory_id not in targets and any(
                            names_replaced(row.statement, value, unsure=False)
                            for value in stated) and forgotten_values(row.statement, asked)]
                # The targets the plan promoted tell the forgotten values too ("The user
                # lives in Shanghai." behind "The user moved from Shanghai to Hangzhou last
                # month.").
                context = list(dict.fromkeys([*targets, *context]))
                sources = [statement, *(row.statement for row in self._rows(scope, context))]
                rest = [*(entry.memory_id for entry in affected
                          if entry.memory_id in current and entry.memory_id not in wordings),
                        *left]
                # One showing the asked colour or breed is reworded without it, as a target
                # is (`_detail_plans`).
                attributes_more, used = self._detail_plans(scope, statement, detail, rest,
                                                           sender)
                calls += used
                more, used = self._reword_plans(
                    scope, statement, [memory_id for memory_id in rest
                                       if memory_id not in attributes_more],
                    names, sources, context, sender, ties=tie_names, keep=frozenset(chained),
                    detail=detail)
                calls += used
                more = {**more, **attributes_more}
                found_more, used = self._embedded(
                    scope, {memory_id: plan for memory_id, (plan, _) in more.items()})
                embedded += used
                rows_more = {row.memory_id: row for row in self._rows(scope, list(found_more))}
                form_parts.update((memory_id, [*self._form_spans(
                    rows_more[memory_id].quotes, self._han_located(
                        rows_more[memory_id].quotes, more[memory_id][1], [
                            statement, *self._quote_texts([
                                span for row in targets.values() for span in row.quotes])])),
                    *(self._detail_token_spans(scope, detail, more[memory_id][1])
                      if memory_id in attributes_more else ())])
                    for memory_id in found_more if memory_id in rows_more)
                if found_more:
                    wordings.update(found_more)
                    continue
            # What the plan's forgets do not write is not planned, and the plan is read again
            # without it.
            admitted_ids = {wording.memory_id for wording in admitted}
            if admitted_ids == set(wordings):
                wordings = {wording.memory_id: wording for wording in admitted}
                unworded = {memory_id for memory_id in left if memory_id not in wordings}
                if unworded and not taken_left:
                    # Read again with them taken: the forget judge's no settles only that
                    # they do not state the fact whole.
                    taken_left = True
                    allowed |= unworded
                    continue
                break
            rounds_left -= 1
            wordings = {memory_id: wording for memory_id, wording in wordings.items()
                        if memory_id in admitted_ids and rounds_left > 0}
        if not targets and not spans:
            rows = [ForgetStatementCandidateV1(
                memory_id=row.memory_id, revision_id=row.revision_id, statement=row.statement,
                role=pick.roles[index], target=False, excluded=excluded.get(index),
                bundled=False) for index, row in enumerate(candidates)] if pick is not None \
                else []
            self._record_handled(scope, handled, statement="nothing_matched",
                                 statement_seq=read_seq, asked=request.request_id)
            return MemoryForgetStatementPreparedV1(
                status="nothing_matched", confirmation=None, expires_at=None,
                candidates=rows, decision=None if pick is None else _decision(pick, statement),
                spans=[ForgetStatementSpanV1(
                    evidence_id=entry.span.evidence_id, byte_start=entry.span.byte_start,
                    byte_end=entry.span.byte_end, text=entry.text, role=entry.role,
                    target=entry.target, excluded=entry.excluded) for entry in judged],
                calls=ForgetStatementCallsV1(judge=calls, embedding=embedded),
                earlier=earlier)
        rows = [ForgetStatementCandidateV1(
            memory_id=row.memory_id, revision_id=row.revision_id, statement=row.statement,
            role=pick.roles[index], target=row.memory_id in targets,
            excluded=excluded.get(index), bundled=index in bundled,
        ) for index, row in enumerate(candidates)] if pick is not None else []
        sentences = [ForgetStatementSpanV1(
            evidence_id=entry.span.evidence_id, byte_start=entry.span.byte_start,
            byte_end=entry.span.byte_end, text=entry.text, role=entry.role,
            target=entry.target, excluded=entry.excluded) for entry in judged]
        decision = None if pick is None else _decision(pick, statement)
        span_intent: str | None = None
        if targets:
            expires_at = self._clock() + INTENT_TTL_MS
        else:
            assert seed is not None
            prepared = self._owner.prepare_spans(
                scope, grant, session_id=request.client_session_id, spans=spans, seed=seed,
                rewordings=list(wordings.values()))
            span_intent, expires_at = prepared.intent_key, prepared.expires_at
        # A target no candidate shows (a later pick, a record of the request) is named among
        # them too, as one the judge picked: the prepare names all the confirmation takes.
        shown_ids = {*(row.memory_id for row in rows), *promoted}
        rows += [ForgetStatementCandidateV1(
            memory_id=target.memory_id, revision_id=target.expected_revision_id,
            statement=targets[target.memory_id].statement, role="same_fact", target=True)
            for target in planned if target.memory_id not in shown_ids]
        if bundles and phrase is not None:
            # The confirmation tombstones the bundles by this vector.
            self._remember_phrase(scope, statement, phrase)
        binding = self._binding(scope, grant, request.client_session_id, request.request_id,
                                planned, read, erase_epoch, expires_at, spans=spans,
                                span_intent=span_intent, bundled=bundles,
                                statement=statement if bundles else None)
        if wordings:
            # The confirmation rewords by these vectors.
            cache = self._wording_cache
            cache[scope.scope_id, str(binding)] = tuple(wordings.values())
            while len(cache) > PHRASE_CACHE:
                cache.pop(next(iter(cache)))
        confirmation = MemoryForgetStatementRequestV1(
            schema_version=1, user_id=request.user_id,
            client_session_id=request.client_session_id, request_id=request.request_id,
            targets=list(planned), candidates=list(read), erase_epoch=erase_epoch,
            expires_at=expires_at, plan_binding=binding, spans=list(spans),
            span_intent=span_intent, bundled=bundles,
            statement=statement if bundles else None,
        )
        # What the picks read and left stays found after the confirmation; the
        # confirmation takes nothing the plan does not name, and only a target read
        # past may be past then.
        # A memory showing a forgotten name about another entity, or as an ordinary word, is
        # kept from the shadow's names as the picks' own are.
        namers = [*(memory_id for memory_id in planned_ids if memory_id not in bundles),
                  *wordings] if own_names else []
        bystanders = {row.memory_id for rows_ in self._name_readers(
            scope, given_names | self._names(scope, namers, ties=asks, aliases=nick,
                                             every=asks and not nick),
            ties=self._tie_words(scope, namers), skip=planned_ids, within=within)
            for row in rows_} - {
            row.memory_id for row in affected}
        read_ids = (({row.memory_id for row in candidates} if pick is not None else set[str]())
                    | first[4] | later[4] | bystanders) - set(targets)
        # What the names the confirmation stamps will hide beside what its kept record
        # exempts, it reports (`_hidden`): the prepare names it too.
        # What the forgets tombstone: the targets' plans (no replacer's), and each memory the
        # cascade forgets on its own.
        order = [target.memory_id for target in planned]
        chained = set(self._owner.chains(
            scope, order, restating=spans, bundled=set(bundles),
            words=[statement], allowed=allowed)) if self._owner.forgets_completely() \
            else set[str]()
        stones = [*self._owner.tombstoned(
            scope, order, restating=spans, bundled=set(bundles),
            words=[statement], allowed=allowed),
            *((snapshot.statement, snapshot.keywords, snapshot.embedding,
               snapshot.embedding_space) for snapshot in self._heads(scope, [
                row.memory_id for row in affected if row.effect == "forgotten"
                and row.memory_id not in chained | planned_ids]).values())]
        hidden = self._foreseen_hidden(
            scope, [*planned_ids, *(row.memory_id for row in affected
                                    if row.effect == "forgotten")],
            read_ids, judged, plan_withheld, stones, lost={row.memory_id for row in affected},
            targets=sorted(planned_ids), sentences=spans) - {
            *planned_ids, *(row.memory_id for row in affected)}
        hidden_heads = self._heads(scope, sorted(hidden))
        affected += [ForgetStatementAffectedV1(
            memory_id=memory_id, effect="unreadable",
            statement=hidden_heads[memory_id].statement if memory_id in hidden_heads else None)
            for memory_id in sorted(hidden)]
        # What the plan's forget shadow would hide from view besides is listed too.
        heads_planned = self._heads(scope, [*planned_ids, *(
            row.memory_id for row in affected if row.effect == "forgotten")])
        reworded_heads = self._heads(scope, list(wordings))
        listed_now = {*planned_ids, *(row.memory_id for row in affected), *wordings,
                      *(row.memory_id for row in candidates)}
        left_forms = [form for memory_id, wording in wordings.items()
                      if (old := reworded_heads.get(memory_id)) is not None
                      for form in left_out(old.statement, wording.statement)]
        shadowed = self._hidden_by(
            scope, [(head.statement, head.keywords) for head in heads_planned.values()],
            asked=asked_names, left=left_forms,
            seed=statement if bundles or not targets else None, skip=listed_now,
            gone={*planned_ids, *(row.memory_id for row in affected
                                  if row.effect != "reworded")})
        affected += [ForgetStatementAffectedV1(memory_id=row.memory_id, effect="unreadable",
                                               statement=row.statement) for row in shadowed]
        self._remember_kept(
            scope, binding, expires_at, read_ids, judged,
            listed={*planned_ids, *(row.memory_id for row in affected)},
            past=self._past(scope, planned_ids), allowed=allowed, nickname=nick,
            residue={row.memory_id for row in affected if row.effect == "forgotten"},
            modifier=self._modifier([statement]),
            name_or_all=asks or asks_everything(statement),
            rewordings=[ForgetRewordPlanV1(
                memory_id=wording.memory_id, base_digest=wording.base_digest,
                statement=wording.statement, tie=wording.tie,
                support=[list(group) for group in wording.support],
                keywords=list(wording.keywords)) for wording in wordings.values()],
            asked=asked_names, name_asked=asks)
        self._record_handled(scope, handled, statement="prepared", statement_seq=read_seq,
                             asked=request.request_id)
        return MemoryForgetStatementPreparedV1(
            status="confirmation_required", confirmation=confirmation, expires_at=expires_at,
            candidates=rows, decision=decision, spans=sentences,
            calls=ForgetStatementCallsV1(judge=calls, embedding=embedded),
            earlier=earlier, affected=affected)

    def _promoted(self, scope: ScopeV1, targets: dict[str, _Candidate],
                  spans: Sequence[QuoteSpanV1], seed: ForgetSeedV1 | None = None,
                  about: Sequence[str] = (), bundled: Set[str] = frozenset(),
                  allowed: Set[str] = frozenset()) -> list[str]:
        """Add to ``targets`` the memories a complete plan of them withholding ``spans`` (with no
        target, under ``seed``; the ``bundled`` ones forgotten for the ``about`` words) takes
        as well (``ForgetOwner.promotions``): one it leaves unreadable that a readable later
        memory replaced and still names ("The user plays badminton every Saturday morning at
        a club in Xuhui." behind "... instead of playing badminton"), once the forget judge
        confirmed it (``allowed``, ``_confirmed``); their ids."""
        try:
            found = self._owner.promotions(scope, list(targets), restating=spans, seed=seed,
                                           about=about, bundled=bundled, allowed=allowed)
        except QuoteMemError as error:
            if error.code not in ("reference_missing", "budget_exceeded"):
                raise
            return []
        rows = [row for row in self._rows(scope, [memory_id for memory_id in found
                                                  if memory_id in allowed])
                if row.memory_id not in targets]
        targets.update((row.memory_id, row) for row in rows)
        return [row.memory_id for row in rows]

    def _nominations(
        self, scope: ScopeV1, trial: Mapping[str, _Candidate], spans: Sequence[QuoteSpanV1],
        seed: ForgetSeedV1 | None = None, *, bundled: Set[str] = frozenset(),
        words: Sequence[str] = (),
    ) -> list[str]:
        """What a complete plan of the ``trial`` targets withholding ``spans`` may take beyond
        its base links, were every word rule's nomination confirmed: the memories it would
        promote (``ForgetOwner.promotions``) and the replacers and unlinked memories naming a
        forgotten value its forgets would take (``ForgetOwner.nominations``), for the forget
        judge to confirm (``_confirmed``)."""
        try:
            promoted = self._owner.promotions(scope, list(trial), restating=spans, seed=seed,
                                              about=words, bundled=bundled)
            found = self._owner.nominations(scope, [*trial, *promoted], restating=spans,
                                            bundled=bundled, words=words)
        except QuoteMemError as error:
            if error.code not in ("reference_missing", "budget_exceeded"):
                raise
            return []
        return [*promoted, *found]

    def _confirmed(self, scope: ScopeV1, statements: Sequence[str], nominees: Sequence[str],
                   sender: Send) -> tuple[set[str], int] | None:
        """The ``nominees`` the forget judge confirms state or presuppose one of ``statements``
        (``forget_pick.tells``), with the calls it made; no call without a nominee. None when
        it cannot answer or leaves one contested: the forget then refuses, since a nominee
        names the forgotten value. A word rule only nominates a memory no link ties to what
        the forget takes; only the judge takes it ("The user works in Pudong now, not
        Xuhui." shares the place of a forgotten badminton club and tells nothing of the
        badminton)."""
        rows = self._rows(scope, list(dict.fromkeys(nominees)))
        found: set[str] = set()
        calls = 0
        for statement in statements if rows else ():
            told = tells(statement, [row.statement for row in rows], sender, rounds=_rounds)
            if told is None or None in told[0]:
                return None
            answers, used = told
            calls += used
            found |= {row.memory_id for row, yes in zip(rows, answers, strict=True) if yes}
        return found, calls

    def _ruled(self, scope: ScopeV1, targets: Mapping[str, _Candidate],
               spans: Sequence[QuoteSpanV1], bundled: Set[str],
               words: Sequence[str]) -> set[str]:
        """What a complete plan of ``targets`` takes by its links alone: their base links
        and the linked replacers naming a forgotten value (``ForgetOwner.chains`` with
        nothing confirmed), never asked of the judge."""
        if not targets:
            return set()
        try:
            return set(self._owner.chains(scope, list(targets), restating=spans,
                                          bundled=bundled, words=words, allowed=frozenset()))
        except QuoteMemError as error:
            if error.code not in ("reference_missing", "budget_exceeded"):
                raise
            return set()

    def _affected(
        self, scope: ScopeV1, targets: Sequence[str], spans: Sequence[QuoteSpanV1],
        seed: ForgetSeedV1 | None, promoted: Sequence[str] = (), *,
        bundled: Set[str] = frozenset(), words: Sequence[str] = (),
        allowed: Set[str] = frozenset(), names: Set[tuple[str, str]] = frozenset(),
        uncertain: bool = False, tie_names: bool = False, aliases: bool = False,
        within: Within | None = None, rewordings: Sequence[ForgetRewordV1] = (),
        asked: Set[ContentBindingDigest] = frozenset(), own_names: bool = True,
    ) -> tuple[list[ForgetStatementAffectedV1], list[str], list[ForgetRewordV1],
               tuple[QuoteSpanV1, ...]]:
        """The memories outside the judge's picks a plan's confirmation may change, each
        with its statement, which the plan's reader already may read: what
        ``ForgetOwner.preview`` of the targets, the first withholding ``spans``, finds may be
        left unreadable, and, when the workspace forgets completely, first the ``promoted``
        targets and what the targets' forgets take with them (``ForgetOwner.chains``), the
        ``bundled`` ones' for the user's ``words``, beyond the base links only what the judge
        confirmed (``allowed``). ``names`` are the names a name request's rule gives besides
        the targets' naming annotations (`_referent_namers`); when its referent was
        ``uncertain``, no name is taken by rule at all. With ``tie_names`` (a request asking a
        name), a target tying a name to someone of the user's by any description gives its
        name too ("Priya is the user's manager.", `_names`). Also the memories the targets'
        forgets take with them by their links (``ForgetOwner.chains``), and the planned
        ``rewordings`` the confirmation's forgets write (``ForgetOwner.admit_rewordings``,
        read with every forget the cascade adds), each listed ``reworded`` with its new
        statement; the others are read as kept and are planned again without them. Without
        ``own_names`` (a request for a colour or a breed), the names the
        targets and rewordings give are none the plan takes."""
        modifier = self._modifier(words)
        planned = {wording.memory_id for wording in rewordings}
        try:
            found, withheld = self._owner.preview_plan(
                scope, targets, restating=spans, seed=seed, bundled=bundled, words=words,
                cascade=True, allowed=allowed, requests=words, modifier=modifier,
                rewordings=rewordings)
            found = [(memory_id, covered) for memory_id, covered in found
                     if memory_id not in planned]
            taken = ([memory_id for memory_id in self._owner.chains(
                scope, targets, restating=spans, bundled=bundled, words=words,
                allowed=allowed)
                      if memory_id not in promoted and memory_id not in planned]
                     if self._owner.forgets_completely() else [])
        except QuoteMemError as error:
            if error.code not in ("reference_missing", "binding_mismatch", "budget_exceeded"):
                raise
            # The confirmation still reports what it did.
            return [], [], [], ()
        listed = [*promoted, *taken]
        heads = {row.memory_id: row.statement for row in self._rows(scope, listed)}
        current = self._memories(scope)
        statements = {row.memory_id: row.statement for row in current}
        # A past memory the plan may leave unreadable is named by its head's statement too.
        statements.update((row.memory_id, row.statement) for row in self._rows(
            scope, [memory_id for memory_id, _ in found if memory_id not in statements]))
        named = {*targets, *listed, *(memory_id for memory_id, _ in found), *planned}
        # What the cascade may take besides (`_leftovers`): a survivor privacy compensation requotes
        # that still restates a tombstoned statement (no replacer's or bundle's: those keep none
        # of their own), then the bare names only it and the rest quoted.
        shown = self._memories(scope, past=True)
        restating = self._restating_survivors(
            scope, shown, named, [*withheld, *(span for row in self._rows(
                scope, list(promoted)) for span in row.quotes)],
            [*(memory_id for memory_id in (*targets, *promoted) if memory_id not in bundled)],
            words)
        named |= {row.memory_id for row in restating}
        # A memory that still shows a name the forget takes goes too (`_leftovers`' fourth
        # kind), whatever the judge decided of it.
        # A quote meeting what the plan withholds shows no forgotten name afterwards (no keeper
        # keeps one): only the others are read for it.
        # A memory the plan rewords still gives the names it gave, as it did as a target.
        namers = [*(memory_id for memory_id in (*targets, *promoted) if memory_id not in bundled),
                  *planned] if own_names else []
        # A memory the plan leaves unreadable that ties a name by the asked description gives it
        # too, as a target would (`_asked_names`).
        lost = [memory_id for memory_id, _ in found]
        holders = [row for row in self._name_holders(
            scope, self._forgotten_names(scope, names, self._names(
                scope, namers, ties=tie_names, aliases=aliases,
                every=tie_names and not aliases) | (
                    self._asked_names(scope, lost, words) if tie_names and not aliases
                    else frozenset[tuple[str, str]]()), uncertain=uncertain, within=within),
            cover=[*spans, *(span for row in self._rows(scope, list(targets))
                             for span in row.quotes)],
            ties=self._tie_words(scope, [*namers, *lost]), skip={*targets, *promoted},
            within=within)
                   if row.memory_id not in named]
        # A memory that goes only for showing a forgotten name says what it says of others:
        # its quotes are no residue of the forgotten fact, nor does it keep a name told.
        named |= {row.memory_id for row in holders}
        gone = [*targets, *listed, *(memory_id for memory_id, covered in found if covered),
                *(row.memory_id for row in restating)]
        residue = self._bare_residue(scope, current, named, gone, withheld, words)
        taken_names = {memory_id for memory_id, snapshot in self._heads(scope, [
            *(memory_id for memory_id, covered in found if covered),
            *(row.memory_id for row in residue)]).items() if naming_only(snapshot)} \
            if tie_names or any(asks_everything(word) for word in words) else set[str]()
        if taken_names:
            # A name asked goes whole, as does everything about someone: the memories still
            # showing a bare name taken as the asked name's residue go with it ("小红 is a
            # teacher." beside "小红 is 小明's wife." for "忘掉小明老婆的名字"), as the cascade
            # takes them (`_cascade`).
            holders += [row for row in self._name_holders(
                scope, self._names(scope, sorted(taken_names)),
                cover=[*spans, *(span for row in self._rows(scope, list(targets))
                                 for span in row.quotes)],
                ties=self._tie_words(scope, [*namers, *lost]), skip={*targets, *promoted},
                within=within) if row.memory_id not in named | taken_names]
            named |= {row.memory_id for row in holders}
        # What the cascade takes besides withholds its own readable quotes, so it may leave
        # still other memories unreadable ("Ruby is the user's daughter." quoting only what a
        # bare "A person is named Ruby." taken as residue quotes): the plan of all of them is
        # read again, as the cascade's rounds take them (``CASCADE_ROUNDS``).
        taken_more = [*(row.memory_id for row in (*restating, *holders, *residue)),
                      *(memory_id for memory_id, covered in found if covered)]
        named |= set(taken_more)
        later: list[tuple[str, bool]] = []
        for _ in range(CASCADE_ROUNDS if taken_more else 0):
            try:
                again = [(memory_id, covered) for memory_id, covered in self._owner.preview_plan(
                    scope, [*targets, *taken_more], restating=spans, seed=seed,
                    bundled=bundled, words=words, cascade=True, allowed=allowed,
                    requests=words, modifier=modifier, rewordings=rewordings)[0]
                         if memory_id not in named]
            except QuoteMemError as error:
                if error.code not in ("reference_missing", "binding_mismatch",
                                      "budget_exceeded"):
                    raise
                break
            if not again:
                break
            later += again
            named |= {memory_id for memory_id, _ in again}
            taken_more += [memory_id for memory_id, covered in again if covered]
        statements.update((row.memory_id, row.statement) for row in self._rows(
            scope, [memory_id for memory_id, _ in later if memory_id not in statements]))
        try:
            admitted = self._owner.admit_rewordings(
                scope, [*targets, *(memory_id for memory_id in taken_more
                                    if memory_id not in targets)],
                restating=spans, seed=seed, bundled=bundled, words=words, allowed=allowed,
                rewordings=rewordings, modifier=modifier, asked=asked) if rewordings else []
        except QuoteMemError as error:
            if error.code not in ("reference_missing", "binding_mismatch", "budget_exceeded"):
                raise
            admitted = []
        olds = {row.memory_id: row.statement for row in self._rows(scope, list(planned))}
        return [*(ForgetStatementAffectedV1(memory_id=memory_id, effect="forgotten",
                                            statement=heads.get(memory_id))
                  for memory_id in listed),
                *(ForgetStatementAffectedV1(memory_id=memory_id,
                                            effect="forgotten" if covered else "unreadable",
                                            statement=statements.get(memory_id))
                  for memory_id, covered in found if memory_id not in listed),
                *(ForgetStatementAffectedV1(memory_id=row.memory_id, effect="forgotten",
                                            statement=row.statement)
                  for row in (*restating, *holders, *residue)),
                *(ForgetStatementAffectedV1(memory_id=memory_id,
                                            effect="forgotten" if covered else "unreadable",
                                            statement=statements.get(memory_id))
                  for memory_id, covered in later),
                # A memory reworded is listed with what it will say (no forgotten word).
                *(ForgetStatementAffectedV1(memory_id=wording.memory_id, effect="reworded",
                                            statement=wording.statement)
                  for wording in admitted
                  if olds.get(wording.memory_id) not in (None, wording.statement))], \
            taken, admitted, withheld

    def _reword_plans(
        self, scope: ScopeV1, statement: str, memory_ids: Sequence[str],
        names: Set[tuple[str, str]], sources: Sequence[str], context: Sequence[str],
        sender: Send, *, ties: bool, keep: Set[str] = frozenset(),
        exists: Set[str] | None = None, detail: DetailRequest | None = None,
    ) -> tuple[dict[str, tuple[ForgetRewordPlanV1, tuple[str, ...]]], int]:
        """How each of ``memory_ids`` reads once what the forget takes is gone
        (``core/forget_reword.py``), with the names and values it leaves out, and the judge's calls.

        A memory is reworded when its statement shows a forgotten name of the entity it is
        about or gives that name for (``names``), or a value it shares with the ``sources``
        (the request and what the forget takes, ``forgotten_values``), and only then; one
        showing a forgotten name of another bearer (a colleague named like the brother) is
        not. Only a request to forget a name (``ties``) lets the tie the ``context`` memories
        state of the bearer stand for the name ("the user's wife"): forgetting that Sofia is
        the user's wife must not bring "wife" back. Any other name is a value its phrase takes
        out. A memory of ``keep`` that shows neither stays as it is. With ``exists`` (the
        statements of the scope's current memories, casefolded), a target that only ties the
        forgotten name to the user ("The user's cat is named Bob.") keeps what it says besides
        ("The user has a cat.", `existence_of`) when no other memory tells of the one it
        names. A request for the place the user works, or which place of a kind (``detail``),
        asks the place a target shows as a name request does (`asked_values`), and the place
        the user works at (`workplace_values`). Each new statement must
        be one the forget judge finds does not state or presuppose ``statement``, the user's
        request (``tells``); without an answer none is (fail closed: each is taken as
        before). Whether the plan's forgets let it be written is the store's to tell
        (``ForgetOwner.admit_rewordings``)."""
        heads = self._heads(scope, [*memory_ids, *context])
        current = {row.statement.casefold() for row in self._memories(scope)}
        # Only the value the request asks is forgotten of a target naming two: the other is
        # the one the user kept ("杭州" of "moved from 杭州 to 上海" for "忘掉我在上海住过").
        sources = asked_sources(statement, [source for source in sources
                                            if source != statement])
        # The targets (planned with ``exists``) are the memories the request names.
        exists_ids = set(memory_ids) if exists is not None else set[str]()
        places = detail is not None and detail.aspect in ("place", "workplace")
        workplace = detail is not None and detail.aspect == "workplace"
        tied = self._entity_ties([heads[memory_id] for memory_id in (*context, *memory_ids)
                                  if memory_id in heads]) if ties else {}
        forgotten = {(entity, name_norm(name)) for entity, name in names}
        plans: dict[str, tuple[ForgetRewordPlanV1, tuple[str, ...]]] = {}
        for memory_id in dict.fromkeys(memory_ids):
            snapshot = heads.get(memory_id)
            if (snapshot is None or snapshot.erased or snapshot.sensitivity_classes
                    or len(plans) >= MAX_REWORDINGS):
                continue
            text = snapshot.statement
            # The forgotten names it shows of an entity it is about, or gives the name for.
            about = {*snapshot.anchors, *(entry.entity_id for entry in snapshot.naming
                                          if (entry.entity_id, name_norm(entry.name))
                                          in forgotten)}
            # A name right after the bearer's tie is the bearer's too, anchor or none ("the
            # user's cat bob").
            own = {name: entity for entity, name in names if shows(text, name) and (
                entity in about or (entity in tied and re.search(
                    rf"\bthe user['’]s {re.escape(tied[entity])} {re.escape(name)}\b", text,
                    re.IGNORECASE)))}
            if any(shows(text, name) and name not in own for _, name in names):
                continue
            given = {name: tied[entity] for name, entity in own.items() if entity in tied}
            # A request for the name of a place or organisation asks the one of that kind
            # a target shows, annotated or not ("忘掉我女儿学校的名字"): its kind stays.
            kinded = asked_values(text, statement) if ties and memory_id in exists_ids else []
            placed = asked_values(text, statement) if places and not ties and (
                memory_id in exists_ids) else []
            values = list(dict.fromkeys([
                *(name for name, entity in own.items() if entity not in tied),
                *(value for value in forgotten_values(
                    text, [source for source in sources if source != text])
                  if name_norm(value) not in {name_norm(name) for _, name in names}),
                *kinded,
                # A request for which place it is asks the one of that kind a statement shows
                # ("忘掉我在哪个体育馆打球").
                *placed,
                # A request for the place the user works asks where a target says they work
                # ("Pinecrest Bakery" of "The user is a pastry chef at Pinecrest Bakery.").
                *(workplace_values(text) if workplace and memory_id in exists_ids
                  else ())]))
            if not given and not values:
                # Kept as it is only when it shares no word with what is forgotten: a later
                # memory taken for its link alone ("The user joined Fabrikam last week.").
                if memory_id in keep and not any(
                        support_words(text) & support_words(source) - _SHARED_LIGHT
                        for source in sources if source != text):
                    plans[memory_id] = (ForgetRewordPlanV1(
                        memory_id=memory_id, base_digest=snapshot.statement_digest,
                        statement=text), ())
                continue
            found = reworded_statement(text, names=given, values=values, kinded=kinded)
            if found is not None and any(
                    (kind := kind_of(value)) is not None
                    and (noun := kind.split(" ", 1)[1]) not in asked_kinds(statement)
                    and shows(found[0], noun) for value in placed):
                # Which place it is goes whole, a kind the request does not name too: "attends
                # a primary school" would still tell which school of "实验小学" for "忘掉我女儿在
                # 哪个学校上学"; "works at a company" tells no more than "哪家公司".
                found = None
            existence = existence_of(text) if given else None
            tie = next(iter(given.values()), "")
            # The one the tie names is kept only when no other memory tells of it ("The user
            # has a cat and a dog." beside "The user's dog is named 豆豆.").
            if found is None and existence is not None and exists is not None and not any(
                    shows(other, tie.rpartition("'s ")[2])
                    for other in (*exists, *(plan.statement for plan, _ in plans.values()))
                    if other != text.casefold()):
                # Its tie's words are what it keeps, not what the request forgets.
                found = existence, f"the user's {tie}"
            # A wording another current memory states already adds nothing: the memory goes
            # as before ("Last month I moved to Hangzhou." beside the move).
            if found is not None and found[0].casefold() != text.casefold() and (
                    found[0].casefold() in current):
                found = None
            if found is not None:
                plans[memory_id] = (ForgetRewordPlanV1(
                    memory_id=memory_id, base_digest=snapshot.statement_digest,
                    statement=found[0], tie=found[1]), tuple(dict.fromkeys([
                        *left_out(text, found[0]), *(value for value in (*given, *values)
                                                     if not shows(found[0], value))])))
        asked = {memory_id: plan for memory_id, plan in plans.items()
                 if plan[0].statement != heads[memory_id].statement}
        if not asked:
            return plans, 0
        told = tells(statement, [plan.statement for plan, _ in asked.values()], sender,
                     rounds=_rounds)
        if told is None:
            return {memory_id: plan for memory_id, plan in plans.items()
                    if memory_id not in asked}, 0
        answers, calls = told
        refused = {memory_id for memory_id, stated in zip(asked, answers, strict=True)
                   if stated is not False}
        return {memory_id: plan for memory_id, plan in plans.items()
                if memory_id not in refused}, calls

    def _detail(self, scope: ScopeV1, statement: str) -> DetailRequest | None:
        """The detail ``statement`` asks to forget (`detail_request`), the thing a request
        names by its name alone read from the scope's naming memories (`with_thing`)."""
        detail = detail_request(statement)
        if detail is None or detail.thing is not None or detail.name is None:
            return detail
        return with_thing(detail, [row.statement for row in self._memories(scope)])

    def _detail_rows(self, scope: ScopeV1, detail: DetailRequest | None) -> list[_Candidate]:
        """The current memories showing the asked detail's value of the user's thing, which a
        request for it takes whatever the judge picked, to reword or take: a
        colour or breed of the thing (`detail_request_values`, `owned`), the place a statement says
        the user works at (`workplace_values`); none for any other request."""
        if detail is None or detail.aspect == "place":
            return []
        return [row for row in self._memories(scope)
                if (attribute(detail) and detail_request_values(row.statement, detail)
                    and owned(row.statement, detail))
                or (detail.aspect == "workplace" and workplace_values(row.statement))]

    def _detail_plans(
        self, scope: ScopeV1, statement: str, detail: DetailRequest | None,
        memory_ids: Sequence[str], sender: Send,
    ) -> tuple[dict[str, tuple[ForgetRewordPlanV1, tuple[str, ...]]], int]:
        """How each of ``memory_ids`` showing the asked colour or breed of the thing reads
        without it (`detail_reworded`: "The user's cat is named bob." of "The user's orange cat
        is named bob."), with every form of the value its statement and quotes show ("orange",
        "橘"), and the judge's calls. One whose value is what it says of the thing ("The user's
        cat is orange."), whose new wording another current memory states already, or whose
        new wording the forget judge finds still tells the detail (``tells``; without an
        answer, none) has no plan, and is taken as before (fail closed)."""
        if not attribute(detail):
            return {}, 0
        heads = self._heads(scope, list(memory_ids))
        current = {row.statement.casefold() for row in self._memories(scope)}
        rows = {row.memory_id: row for row in self._rows(scope, list(memory_ids))}
        plans: dict[str, tuple[ForgetRewordPlanV1, tuple[str, ...]]] = {}
        for memory_id in dict.fromkeys(memory_ids):
            snapshot, row = heads.get(memory_id), rows.get(memory_id)
            if (snapshot is None or row is None or snapshot.erased
                    or snapshot.sensitivity_classes or len(plans) >= MAX_REWORDINGS):
                continue
            values = detail_request_values(snapshot.statement, detail)
            found = detail_reworded(snapshot.statement, detail, values)
            if found is None or found.casefold() in current:
                continue
            forms = [*values, *(form for text in self._quote_texts(row.quotes)
                                for form in quote_forms(text, values, detail))]
            plans[memory_id] = (ForgetRewordPlanV1(
                memory_id=memory_id, base_digest=snapshot.statement_digest, statement=found),
                tuple(dict.fromkeys(forms)))
        if not plans:
            return plans, 0
        told = tells(statement, [plan.statement for plan, _ in plans.values()], sender,
                     rounds=_rounds)
        if told is None:
            return {}, 0
        answers, calls = told
        return {memory_id: plan for (memory_id, plan), stated in zip(
            plans.items(), answers, strict=True) if stated is False}, calls

    def _detail_token_spans(self, scope: ScopeV1, detail: DetailRequest | None,
                            forms: Sequence[str]) -> list[QuoteSpanV1]:
        """The occurrences of the forms of an asked colour or breed in the scope's readable
        sentences, telling it of the thing (`quote_forms`: "橘" of the assistant's "你的橘猫叫
        bob", never of "橘子"), each form's bytes alone: what a reworded memory leaves out goes
        wherever it is told, as its forget whole would have withheld it there too."""
        if not attribute(detail) or not forms:
            return []
        writer = self._writer
        with writer.write_scope():
            txn = TxnRows(writer, scope.scope_kind, scope.scope_id, 0, self._clock(), "")
            parts = [row for row in txn.pages(EvidenceRows(scope_kind=scope.scope_kind,
                                                           scope_id=scope.scope_id))
                     if row["state"] == "active"]
        found: list[QuoteSpanV1] = []
        for part in parts:
            for sentence, text in self._readable(part):
                for start, end in quote_ranges(text, forms, detail):
                    first = sentence.byte_start + len(text[:start].encode("utf-8"))
                    found.append(sentence.model_copy(update={
                        "byte_start": first,
                        "byte_end": first + len(text[start:end].encode("utf-8"))}))
        return found

    @staticmethod
    def _entity_ties(snapshots: Sequence[MemorySnapshotV2]) -> dict[str, str]:
        """The tie each entity the ``snapshots`` name is the user's, as it reads after "the
        user's" ("wife"), from the statements that tie its name to it (`tie_descriptions`):
        the entity the statement gives that name for, else the one it is about ("林晓雨 is the
        user's wife." of the person it anchors); none for an entity two statements tie
        differently."""
        found: dict[str, set[str]] = {}
        for snapshot in snapshots:
            for name, description in tie_descriptions([snapshot.statement]).items():
                entities = {entry.entity_id for entry in snapshot.naming
                            if entry.entity_id != USER
                            and name_norm(entry.name) == name_norm(name)} or _about(snapshot)
                if len(entities) == 1:
                    found.setdefault(next(iter(entities)), set()).add(description)
        # One tie extending another by its modifiers is the same tie ("paternal grandmother"
        # of "grandmother"): the longest stands.
        merged: dict[str, str] = {}
        for entity, ties in found.items():
            longest = max(ties, key=len)
            if all(longest == tie or longest.endswith(f" {tie}") for tie in ties):
                merged[entity] = longest
        return merged

    def _embedded(self, scope: ScopeV1, plans: Mapping[str, ForgetRewordPlanV1],
                  ) -> tuple[dict[str, ForgetRewordV1], int]:
        """The ``plans`` with their statements' vectors, by one embedding call, and the calls
        made: a plan keeping its memory's statement keeps its memory's vector. Without an
        embedder, or when the call fails, only those are kept (fail closed). The support and
        keywords a plan records are kept."""
        heads = self._heads(scope, list(plans))
        fresh = [plan for memory_id, plan in plans.items()
                 if memory_id in heads and heads[memory_id].statement != plan.statement]
        try:
            phrases, calls = self._phrases([plan.statement for plan in fresh])
        except QuoteMemError as error:
            if error.code not in _SEED_FALLBACK:
                raise
            phrases, calls = [], 1
        vectors = {plan.memory_id: (list(phrase.raw), phrase.space)
                   for plan, phrase in zip(fresh, phrases, strict=False)}
        found: dict[str, ForgetRewordV1] = {}
        for memory_id, plan in plans.items():
            head = heads.get(memory_id)
            if head is not None and plan.statement == head.statement:
                vector, space = list(head.embedding), head.embedding_space
            elif memory_id in vectors:
                vector, space = vectors[memory_id]
            else:
                continue
            found[memory_id] = ForgetRewordV1(
                memory_id=memory_id, base_digest=plan.base_digest, statement=plan.statement,
                embedding=vector, embedding_space=space, tie=plan.tie,
                support=tuple(tuple(group) for group in plan.support),
                keywords=tuple(plan.keywords))
        return found, calls

    @staticmethod
    def _value_spans_text(text: str, value: str) -> list[tuple[int, int]]:
        """The UTF-8 byte ranges of ``text`` where it shows ``value`` (`value_pattern`)."""
        return [(len(text[:match.start()].encode("utf-8")),
                 len(text[:match.end()].encode("utf-8")))
                for match in value_pattern(value).finditer(text)]

    def _han_located(self, quotes: Sequence[QuoteSpanV1], values: Sequence[str],
                     forgotten: Sequence[str]) -> list[str]:
        """``values`` with each Latin one no quote shows located in the Han quotes by what
        they share with the ``forgotten`` texts (`han_forms`): "上海" for "Shanghai" of "The
        user moved from Shanghai to Hangzhou." over "我上个月从上海搬到了杭州"."""
        texts = self._quote_texts(quotes)
        found: list[str] = []
        for value in values:
            if value.isascii() and not any(shows(text, value) for text in texts):
                forms = [form for text in texts if text and not text.isascii()
                         for form in han_forms(text, forgotten)]
                found += forms or [value]
            else:
                found.append(value)
        return list(dict.fromkeys(found))

    def _form_spans(self, quotes: Sequence[QuoteSpanV1],
                    forms: Sequence[str]) -> list[QuoteSpanV1]:
        """Where a memory's ``quotes`` show any of the ``forms`` its rewording leaves out."""
        return [span.model_copy(update={"byte_start": span.byte_start + start,
                                         "byte_end": span.byte_start + end})
                for span, text in zip(quotes, self._quote_texts(quotes), strict=True)
                for form in forms for start, end in self._value_spans_text(text, form)]

    def _value_spans(self, quotes: Sequence[QuoteSpanV1],
                     values: Sequence[str]) -> list[QuoteSpanV1]:
        """Where a memory's ``quotes`` show each of ``values``: the words a reworded bundle
        withholds. Empty when a value is in none of them, since it may then
        be told in other words, or when a quote shows a value in a way its text does not."""
        found: list[QuoteSpanV1] = []
        located: set[str] = set()
        for span, text in zip(quotes, self._quote_texts(quotes), strict=True):
            if not text:
                return []
            for value in values:
                ranges = self._value_spans_text(text, value)
                if shows(text, value) and not ranges:
                    return []
                located |= {value} if ranges else set()
                found += [span.model_copy(update={"byte_start": span.byte_start + start,
                                                  "byte_end": span.byte_start + end})
                          for start, end in ranges]
        return found if values and located == set(values) else []

    def _names(self, scope: ScopeV1, memory_ids: Sequence[str], *,
               ties: bool = False, aliases: bool = False,
               every: bool = False) -> frozenset[tuple[str, str]]:
        """The ``(entity_id, name)`` pairs the heads of ``memory_ids`` give (their naming
        annotations), read before they are forgotten: what no memory about that entity may
        show once the forget took them; with ``ties``, also the name a head ties to someone or
        something of the user's by any description, a tie word or a role (`tied_name`: "Tom"
        of "Tom is the user's younger brother.", "Priya" of "Priya is the user's manager."),
        for the entity it is about, or else for the one entity a naming memory gives that
        name. A name of the user, or of no known entity, gives none: which memories are about
        it cannot be told. With ``aliases`` (a request for a nickname), only another name a
        head gives someone counts ("安安" of "林可欣 is also known as 安安."; an alias naming
        annotation), never the name itself. With ``every`` (a request for the name), every
        name the scope's memories give an entity a head names counts too: a name request whose
        pick is "Rowan's nickname is Ollie." forgets "Rowan" with "Ollie"."""
        found: set[tuple[str, str]] = set()
        if aliases:
            for snapshot in self._heads(scope, memory_ids).values():
                found |= _aliases(snapshot)
            return frozenset(found)
        bearers: dict[str, set[str]] | None = None
        for snapshot in self._heads(scope, memory_ids).values():
            found |= {(entry.entity_id, entry.name) for entry in snapshot.naming
                      if entry.entity_id != USER}
            tie = tied_name(snapshot.statement) if ties else None
            if tie is None:
                continue
            about = _about(snapshot)
            if not about:
                if bearers is None:
                    bearers = {}
                    for head in self._heads(scope, [row.memory_id for row in self._memories(
                            scope)]).values():
                        for entry in head.naming:
                            if entry.entity_id != USER:
                                bearers.setdefault(entry.name.casefold(), set()).add(
                                    entry.entity_id)
                about = bearers.get(tie[1].casefold(), set[str]())
                about = about if len(about) == 1 else set[str]()
            found |= {(entity, tie[1]) for entity in about}
        if every and found:
            named = {entity for entity, _ in found}
            for head in self._heads(scope, [row.memory_id for row in self._memories(
                    scope)]).values():
                found |= {(entry.entity_id, entry.name) for entry in head.naming
                          if entry.entity_id in named}
        return frozenset(found)

    def _kind_names(self, scope: ScopeV1, statements: Sequence[str],
                    memory_ids: Sequence[str]) -> frozenset[tuple[str, str]]:
        """The ``(entity_id, name)`` pairs of the names the requests ask of a place or
        organisation the heads of ``memory_ids`` show (`asked_values`: "Maple Grove Elementary"
        of "The user's daughter Ruby goes to Maple Grove Elementary." for "forget the name of
        my daughter's school"), as their naming annotations give them; none when no head
        gives such a name an entity."""
        found: set[tuple[str, str]] = set()
        for snapshot in self._heads(scope, memory_ids).values():
            asked = {name_norm(value) for statement in statements
                     for value in asked_values(snapshot.statement, statement)}
            found |= {(entry.entity_id, entry.name) for entry in snapshot.naming
                      if entry.entity_id != USER and name_norm(entry.name) in asked}
            # A memory only naming it ("A school is named Maple Grove Elementary.") gives it
            # its own entity, annotated or not.
            named = naming_value(snapshot.statement)
            if named is not None and name_norm(named) in asked and snapshot.entity_id != USER:
                found.add((snapshot.entity_id, named))
        return frozenset(found)

    def _asked_names(self, scope: ScopeV1, memory_ids: Sequence[str],
                     statements: Sequence[str]) -> frozenset[tuple[str, str]]:
        """The ``(entity_id, name)`` pairs of the heads of ``memory_ids`` that tie a name to
        someone by the description a request asks the name of, or tie the name it gives
        (`asked_by`): "Priya" of "Priya is the user's manager." under "forget my manager's
        name", when the forget took only the sentence telling it (`_names`)."""
        asked = [memory_id for memory_id, snapshot in self._heads(scope, memory_ids).items()
                 if (tie := tied_name(snapshot.statement)) is not None and any(
                     asked_by(statement, *tie) for statement in statements)]
        return self._names(scope, asked, ties=True) if asked else frozenset()

    def _referent_namers(self, scope: ScopeV1, statements: Sequence[str],
                         ) -> tuple[list[_Candidate], frozenset[tuple[str, str]], bool, Within]:
        """The current memories giving the name the user asks to forget, the
        ``(entity_id, name)`` pairs they give, whether a request's referent was uncertain
        (several people or animals), and, for an entity the extraction made of two people
        (`_ambiguous`: "Daniel is the user's brother." and "Daniel is the user's colleague."),
        the sentences its asked tie is told in: the name is forgotten there
        alone, with the naming memories and name holders quoting nothing else
        ("My brother Daniel lives in Oslo." but not "My colleague Daniel helps me").

        For a request naming a person or an animal by their tie to the user
        (`forget_match.name_request`), the memories tying a name to such a tie
        (`forget_match.naming_tie`) say who they may be. Only when they are about exactly one
        person or animal (the one the request names, when it gives a name: "my friend Alice's
        name") are those memories and that entity's naming memories taken. Two friends, two
        brothers or a name no memory ties leave the request to the judge; no other fact about
        the referent is taken here (one still showing the name goes by `_name_holders`), and a
        request about anything else takes nothing."""
        requests: list[tuple[NameRequest, bool]] = []
        for statement in statements:
            nick = nickname_request(statement)
            # A request naming the one whose name it asks by that name alone ("忘掉Emma的名字")
            # resolves to the one entity bearing it, as a request giving the name beside a tie
            # does: two bearing it leave the request to the judge.
            bare = None if nick else given_request(statement)
            requests += [(request, nick) for request in (
                name_request(statement, nickname=nick) or (() if bare is None else (bare,)))]
        if not requests:
            return [], frozenset(), False, {}
        rows = self._memories(scope)
        heads = self._heads(scope, [row.memory_id for row in rows])
        named_by: dict[str, set[str]] = {}
        for snapshot in heads.values():
            for entry in snapshot.naming:
                if entry.entity_id != USER:
                    named_by.setdefault(entry.name.casefold(), set()).add(entry.entity_id)
        taken: dict[str, _Candidate] = {}
        pairs: set[tuple[str, str]] = set()
        uncertain = False
        within: dict[str, frozenset[tuple[str, int, int]]] = {}
        ambiguous = self._ambiguous(scope)
        for request, nick in requests:
            # Each memory tying a name to the request's tie, by the referent it is about: its
            # entity, or the one entity a naming memory gives that name, or the name alone.
            referents: dict[str, list[tuple[_Candidate, str]]] = {}
            for row in rows:
                snapshot = heads.get(row.memory_id)
                tie = None if snapshot is None else naming_tie(snapshot.statement)
                if snapshot is None or tie is None or not tie[0] & request.ties:
                    continue
                about = sorted(_about(snapshot)) or sorted(named_by.get(tie[1].casefold(), ()))
                key = about[0] if len(about) == 1 else f"name:{tie[1].casefold()}"
                referents.setdefault(key, []).append((row, tie[1]))
            if request.given is not None:
                given = request.given.casefold()
                referents = {key: found for key, found in referents.items()
                             if any(name.casefold() == given for _, name in found)}
                if not referents and len(named_by.get(given, ())) == 1:
                    (entity,) = named_by[given]
                    referents = {entity: []}
            if len(referents) != 1 or (next(iter(referents)) in ambiguous and not all(
                    found for found in referents.values())):
                # No referent or several: the judge decides, and of a referent it cannot
                # tell, the judge's picks alone (`_forgotten_names`).
                uncertain |= bool(referents)
                continue
            ((key, found),) = referents.items()
            if key in ambiguous:
                # Two people in one entity: the asked one is the one its tie's sentences tell.
                within[key] = within.get(key, frozenset()) | self._sentence_scope(
                    scope, [row for row, _ in found])
            if nick:
                # A nickname goes, not the name: the memories giving the referent another name
                # alone ("林可欣 is also known as 安安."), never its name or its tie.
                for row in [] if key.startswith("name:") else rows:
                    snapshot = heads.get(row.memory_id)
                    other = set[tuple[str, str]]() if snapshot is None else {
                        pair for pair in _aliases(snapshot) if pair[0] == key}
                    if other and (key not in within or all(
                            _inside(span, within[key]) for span in row.quotes)):
                        taken.setdefault(row.memory_id, row)
                        pairs.update(other)
                continue
            for row, name in found:
                taken.setdefault(row.memory_id, row)
                if not key.startswith("name:"):
                    pairs.add((key, name))
            if key.startswith("name:"):
                continue
            for row in rows:
                snapshot = heads.get(row.memory_id)
                given_names = [] if snapshot is None else [
                    entry.name for entry in snapshot.naming if entry.entity_id == key]
                if given_names and (key not in within or all(
                        _inside(span, within[key]) for span in row.quotes)):
                    taken.setdefault(row.memory_id, row)
                    pairs.update((key, name) for name in given_names)
        return list(taken.values()), frozenset(pairs), uncertain, within

    def _sentence_scope(self, scope: ScopeV1, rows: Iterable[_Candidate],
                        ) -> frozenset[tuple[str, int, int]]:
        """The sentences of the scope's Evidence the quotes of ``rows`` lie in, as
        ``(evidence_id, byte_start, byte_end)``."""
        quotes = [span for row in rows for span in row.quotes]
        found: set[tuple[str, int, int]] = set()
        writer = self._writer
        with writer.write_scope():
            for evidence_id in sorted({span.evidence_id for span in quotes}):
                part = writer.get_row("evidence", evidence_id)
                if part is None or (part["scope_kind"], part["scope_id"]) != (
                        scope.scope_kind, scope.scope_id):
                    continue
                text, base = cast(str, part["text"]), cast(int, part["byte_start"])
                for start, end in sentences(text):
                    first = base + len(text[:start].encode("utf-8"))
                    last = base + len(text[:end].encode("utf-8"))
                    if any(span.evidence_id == evidence_id and span.byte_start < last
                           and first < span.byte_end for span in quotes):
                        found.add((evidence_id, first, last))
        return frozenset(found)

    def _home_quotes(self, scope: ScopeV1) -> list[QuoteSpanV1]:
        """The quotes of the current memories of where someone lives now (`_LIVES_NOW`) that no
        later one replaced or ended: a request for where the user lived withholds none."""
        rows = self._memories(scope)
        past = {row.memory_id for row in self._replaced(scope, rows)}
        return [span for row in rows if row.memory_id not in past
                and _LIVES_NOW.search(row.statement) for span in row.quotes]

    def _current_values(self, scope: ScopeV1, statements: Sequence[str],
                        targets: Mapping[str, _Candidate], old_ids: Set[str]) -> set[str]:
        """For a request for an old value whose past value was found (``old_ids`` holding a
        value a later one replaced), the ``targets`` stating a current value of that kind:
        "forget my old address" takes "The user lives at 14 Elm Street." (replaced), never
        "The user lives at 27 Birch Lane." the judge read as an address too."""
        rows = self._memories(scope)
        past = {row.memory_id for row in self._replaced(scope, rows)}
        if not old_ids & past:
            return set()
        found: set[str] = set()
        for statement in statements:
            plain = old_value(statement)
            if plain is None:
                continue
            kind = plain.split("的", 1)[-1] if not plain.isascii() else plain.split(
                maxsplit=1)[-1]
            pattern = _OLD_KIND_WORDS.get(_OLD_JUDGED.get(kind, kind))
            if pattern is not None:
                found |= {memory_id for memory_id, row in targets.items()
                          if memory_id not in past and memory_id not in old_ids
                          and pattern.search(row.statement)}
        return found

    def _given_up_clauses(self, scope: ScopeV1, statements: Sequence[str],
                          targets: Mapping[str, _Candidate], spans: Sequence[QuoteSpanV1],
                          ) -> list[QuoteSpanV1]:
        """For a request for an old value ("我以前那辆车", `old_value`), the clauses telling that
        value was given up beside the new one a kept memory of the kind quotes (`_given_up`):
        the forget takes such a clause as its span, and the new value's clause stays. What it
        takes already (``targets``, ``spans``) gives the old value. A clause a kept memory of
        the kind quotes is left: withholding it would leave that memory unreadable, new value
        and all ("The user's car is a blue 卡罗拉." quoting the whole sentence)."""
        taken = [*(span for row in targets.values() for span in row.quotes), *spans]
        gone = self._quoted(scope, taken)
        kinds = [plain.split("的", 1)[-1] if not plain.isascii() else plain.split(maxsplit=1)[-1]
                 for statement in statements if (plain := old_value(statement)) is not None]
        patterns = [pattern for kind in kinds
                    if (pattern := _OLD_KIND_WORDS.get(_OLD_JUDGED.get(kind, kind))) is not None]
        if not gone or not patterns:
            return []
        kept = [row for row in self._memories(scope) if row.memory_id not in targets
                and any(pattern.search(row.statement) for pattern in patterns)
                and not any(_overlaps(span, taken) for span in row.quotes)]
        stated = [row.statement for row in kept]
        writer = self._writer
        found: dict[tuple[str, int, int], QuoteSpanV1] = {}
        for quote in (span for row in kept for span in row.quotes):
            with writer.write_scope():
                part = writer.get_row("evidence", quote.evidence_id)
            if part is None or part["role"] != "user":
                continue
            raw = part["text"].encode("utf-8")
            low, high = (len(raw[:at - part["byte_start"]].decode("utf-8", "ignore"))
                         for at in (quote.byte_start, quote.byte_end))
            for start, end in _given_up(part["text"], low, high, gone, stated):
                span = QuoteSpanV1(
                    evidence_id=quote.evidence_id,
                    byte_start=part["byte_start"] + len(part["text"][:start].encode("utf-8")),
                    byte_end=part["byte_start"] + len(part["text"][:end].encode("utf-8")))
                if not any(_overlaps(span, row.quotes) for row in kept):
                    found.setdefault(_key(span), span)
        return list(found.values())

    def _hidden_by_words(self, scope: ScopeV1, statements: Sequence[str]) -> set[str]:
        """The current memories the requests' own words hide from Search and the list: those
        showing a name the words give, in the statement, the keywords or a quote, unless
        enough memories still hold that name (`recall_agent.surviving_names`)."""
        given = [shadow_terms(asked_part(statement)).names for statement in statements]
        shadow = ForgetShadow.of(ForgetShadowTerms(frozenset[str]().union(*given)))
        if not shadow.names:
            return set()
        held = {row.memory_id: frozenset(term for text in (
            " ".join((row.statement, *row.keywords)), *self._quoted(scope, row.quotes))
            for term in sparse_terms(text)) for row in self._memories(scope)}
        names = shadow.names - surviving_names(shadow, held.values())
        return {memory_id for memory_id, terms in held.items() if terms & names}

    def _kept_homes(self, scope: ScopeV1, targets: Mapping[str, _Candidate],
                    statements: Sequence[str], words: Sequence[str]) -> set[str]:
        """The ``targets`` a request may not take: for where the user lived (`lived_value`),
        a current memory of where someone lives (`_LIVES_NOW`) no later one replaced or ended,
        so "forget that I lived in Shanghai as a kid" never takes "The user lives in
        Shanghai." (fail closed: it stays, and nothing is reported removed); for any request,
        a memory stating the new value its words told (`_states_new`)."""
        if not targets:
            return set()
        lived = any(lived_value(statement) is not None for statement in statements)
        past = ({row.memory_id for row in self._replaced(scope, self._memories(scope))}
                if lived else set[str]())
        return {memory_id for memory_id, row in targets.items()
                if (lived and memory_id not in past and _LIVES_NOW.search(row.statement))
                or any(_states_new(row.statement, said) for said in words)}

    def _old_values(self, scope: ScopeV1, asked: Sequence[tuple[str, Sequence[_Candidate], str]],
                    sender: Send) -> tuple[list[_Candidate], int]:
        """The memories a request for an old value of the user's asks to forget ("my old
        address", "我以前那辆车", `forget_requests.old_value`), and the judge calls spent.

        ``asked`` holds each request's words it asks to forget (`forget_requests.asked_part`),
        its candidates (Search and vectors, as its pick reads them) and the words it was said
        in (its sentence, or the plugin's whole statement). A plain request for an old car
        ("forget my old car", no new one told) is left to the judge's own pick. The judge reads
        the request's plain words ("my address", a Chinese kind by its English words,
        ``_OLD_JUDGED``) against those candidates, the memories sharing a word with them and
        the memories a later one replaced or that ended (`_replaced`). Of those it takes, the
        replaced ones are the old values; with none replaced, the replaced values of the user's
        own whose statements name the kind (`_OLD_KINDS`, `_USERS_OWN`: "The user lives at 14 Elm
        Street." when the judge took only "The user lives at 27 Birch Lane."), else all it took
        when the request's sentence said the new value ("..., I live at 27 Birch Lane now."),
        else the one memory it takes, the only value the user called old. Several current values
        and no replaced one leave the request to the judge's own pick (nothing is taken here);
        a found past value makes `_current_values` keep the current ones the judge took."""
        found: dict[str, _Candidate] = {}
        calls = 0
        if not any(old_value(statement) or lived_value(statement) for statement, *_ in asked):
            return [], 0
        rows = self._memories(scope)
        past = self._replaced(scope, rows)
        replaced = {row.memory_id for row in past}
        for statement, candidates, sentence in asked:
            if (home := lived_value(statement)) is not None:
                # Where the user lived: only a value a later one replaced or ended, judged as
                # the place it stated; never a current home (fail closed).
                pool = past[:MAX_FORGET_CANDIDATES]
                if pool:
                    pick = pick_targets(home, [row.statement for row in pool], sender,
                                        rounds=_rounds, minimal=False)
                    calls += pick.calls
                    for i in sorted({*pick.targets, *(
                            i for i, role in enumerate(pick.roles) if role in _PARTS)}
                            - _spared(pool, pick.roles)):
                        found.setdefault(pool[i].memory_id, pool[i])
                continue
            plain = old_value(statement)
            if plain is None:
                continue
            kind = plain.split("的", 1)[-1] if not plain.isascii() else plain.split(
                maxsplit=1)[-1]
            reach = request_reach(sentence)
            told_new = tells_new(sentence) or (
                reach is not None and reach < len(sentence.rstrip()))
            # A plain request for an old car is left to the judge's own pick: a car is stored as
            # several memories (its make, its colour, its sale), the rule took one, and the rest
            # stayed readable behind a verified forget.
            if not told_new and _OLD_KIND_WORDS["car"].search(_OLD_JUDGED.get(kind, kind)):
                continue
            pool = list({row.memory_id: row for row in (
                *past, *candidates, *_sharing(plain, rows))}.values())[:MAX_FORGET_CANDIDATES]
            if not pool:
                continue
            # The stored statements are English: a Chinese kind is judged by its English
            # words ("我的地址" as "my address"), which the judge matches to "The user lives
            # at 梧桐路18号."; it read the bare Chinese kind as unrelated.
            judged = f"my {_OLD_JUDGED[kind]}" if not plain.isascii() and kind in _OLD_JUDGED \
                else plain
            pick = pick_targets(judged, [row.statement for row in pool], sender, rounds=_rounds,
                                minimal=False)
            calls += pick.calls
            chosen = [pool[i] for i in sorted(
                {*pick.targets, *(i for i, role in enumerate(pick.roles) if role in _PARTS)}
                - _spared(pool, pick.roles))
                if exclusion(plain, pool[i].statement) is None]
            old = [row for row in chosen if row.memory_id in replaced]
            if not old:
                # The judge reads a bare kind ("my car") as no fact, or takes only a current
                # value for it ("The user lives at 27 Birch Lane." as "my address"): a past
                # value of the user's own whose statement names the kind is the old one ("The
                # user's car is a white 思域.", "The user lives at 14 Elm Street.").
                words = _OLD_KINDS.get(kind, (kind,))
                old = [row for row in past if _USERS_OWN.match(row.statement)
                       and any(_shows_name(row.statement, word) for word in words)]
            if not chosen and not old:
                # With none and the new value told, the one current memory of that kind is the
                # old one ("The user drives a Honda Civic." of "my old car, I drive a Tesla
                # now").
                pattern = _OLD_KIND_WORDS.get(_OLD_JUDGED.get(kind, kind))
                current = [row for row in rows if row.memory_id not in replaced
                           and pattern is not None and pattern.search(row.statement)
                           and not any(self._told_new(span) for span in row.quotes)]
                if told_new and len(current) == 1:
                    old = current
            for row in old or (chosen if told_new or len(chosen) == 1 else []):
                if not _states_new(row.statement, sentence):
                    found.setdefault(row.memory_id, row)
        return list(found.values()), calls

    def _replaced(self, scope: ScopeV1, rows: Sequence[_Candidate]) -> list[_Candidate]:
        """The user's past values: the readable memories a newer one superseded (the list's
        ``past``: "The user's car is a white Civic." after "The user's car is a blue 卡罗拉."),
        and the current ``rows`` a later current memory replaced (an ``update`` or ``correct``
        link) or whose state has ended."""
        current = {row.memory_id for row in rows}
        now = self._clock()
        found = {memory_id for memory_id, snapshot in self._heads(scope, sorted(current)).items()
                 if snapshot.kind == "state" and snapshot.valid_to is not None
                 and snapshot.valid_to <= now}
        writer = self._writer
        with writer.write_scope():
            txn = TxnRows(writer, scope.scope_kind, scope.scope_id, 0, self._clock(), "")
            read = readability(txn)
            for item in txn.pages(MemoryTableRows(scope_kind=scope.scope_kind,
                                                  scope_id=scope.scope_id)):
                head = writer.get_row("revision", item["current_revision_id"])
                if item["lifecycle_state"] == "superseded" and head is not None and read.p_ver(
                        MemorySnapshotV2.model_validate(head["snapshot"])):
                    found.add(item["object_id"])
            for row in rows:
                for link in txn.links_to("supersedes", row.memory_id):
                    source = txn.get("revision", link["source_revision_id"])
                    if link["relation"] in ("update", "correct") and source is not None \
                            and source["memory_id"] in current \
                            and source["memory_id"] != row.memory_id:
                        found.add(row.memory_id)
                        break
        return self._rows(scope, sorted(found))

    def _whole_people(self, scope: ScopeV1, statements: Sequence[str]) -> list[_Candidate]:
        """The memories of the one person a request asks to forget as a whole ("everything
        about my ex", "忘掉阿杰", `forget_requests.whole_referent`).

        The person is the one entity the scope names so (a name) or ties to the user so (a
        tie: `forget_match.naming_tie`, or an ex the memories name as "ex-boyfriend", 前男友
        ...). Every current memory anchoring that entity is taken, and one anchoring no entity
        that shows one of its names outside a longer name the store knows for another person
        (`_shows_own`: never "Jordan" of "Michael Jordan", always that of "Yesterday Jordan
        texted the user."). A memory that also anchors, or names, any other named entity, a
        person of the user's or not, is another person's too and stays (the judge may still
        take it, and the plan lists it); with no such entity, or several, nothing is taken
        and the judge decides."""
        wanted = [found for statement in statements if (found := whole_referent(statement))]
        if not wanted:
            return []
        rows = self._memories(scope)
        heads = self._heads(scope, [row.memory_id for row in rows])
        named_by: dict[str, set[str]] = {}
        names_of: dict[str, set[str]] = {}
        for snapshot in heads.values():
            for entry in snapshot.naming:
                if entry.entity_id != USER:
                    named_by.setdefault(entry.name.casefold(), set()).add(entry.entity_id)
                    names_of.setdefault(entry.entity_id, set()).add(entry.name)
        people: dict[str, set[str]] = {}
        exes: set[str] = set()
        for snapshot in heads.values():
            tie = naming_tie(snapshot.statement)
            about = _about(snapshot) or (set() if tie is None else named_by.get(
                tie[1].casefold(), set()))
            if tie is not None:
                for entity in about:
                    people.setdefault(entity, set()).update(tie[0])
            if (ex := _EX.search(snapshot.statement)) is not None:
                # The ex the memory is about; by name only when it anchors none ("Max is the
                # user's ex-husband." is no tie of a coworker Max).
                exes |= _about(snapshot) or named_by.get(
                    (ex["name"] or ex["after"] or "").casefold(), set())
        ambiguous = self._ambiguous(scope)
        taken: dict[str, _Candidate] = {}
        for kind, who in wanted:
            if kind == "name":
                entities = set(named_by.get(who.casefold(), ()))
            else:
                ties = {tie for request in name_request(
                    f"forget {who}'s name" if who.isascii() else f"忘掉{who}的名字")
                    for tie in request.ties}
                entities = {entity for entity, found in people.items() if found & ties}
                if _EX_WORD.search(who):
                    entities |= exes
            if len(entities) != 1 or entities & ambiguous:
                continue
            (entity,) = entities
            others = ({named for found in named_by.values() for named in found}
                      | set(people)) - {entity}
            names = names_of.get(entity, set())
            other_names = {name for other in others for name in names_of.get(other, ())}
            # A name another entity bears too tells no memory of this one's (a coworker Max
            # beside the ex Max): only anchoring does.
            own = {name for name in names if name not in other_names}
            known = people.get(entity, set())
            for row in rows:
                snapshot = heads.get(row.memory_id)
                if snapshot is None:
                    continue
                anchored = {snapshot.entity_id, *snapshot.about_entity_ids,
                            *(entry.entity_id for entry in snapshot.naming)} - {USER}
                if anchored & others or any(_shows_name(snapshot.statement, name)
                                            for name in other_names if name not in names):
                    continue  # another person's too (阿杰和小李一起…), or another Jordan
                tie = naming_tie(snapshot.statement)
                if entity not in anchored and tie is not None and known and not (
                        tie[0] & known):
                    continue  # the name tied to another of the user's ("my coworker Max")
                if entity in anchored or any(
                        _shows_own(snapshot.statement, name, other_names) for name in own):
                    taken.setdefault(row.memory_id, row)
        return list(taken.values())

    def _ambiguous(self, scope: ScopeV1) -> frozenset[str]:
        """The entities the scope's memories tie to the user by ties that share none ("Daniel
        is the user's brother." and "Daniel is the user's colleague." of one entity): two
        people the extraction made one, whose facts no name rule can tell apart."""
        heads = self._heads(scope, [row.memory_id for row in self._memories(scope)])
        ties: dict[str, list[frozenset[str]]] = {}
        for snapshot in heads.values():
            tie = naming_tie(snapshot.statement)
            if tie is not None:
                for entity in _about(snapshot):
                    ties.setdefault(entity, []).append(tie[0])
        return frozenset(entity for entity, found in ties.items()
                         if any(not first & second for first in found for second in found))

    def _forgotten_names(self, scope: ScopeV1, given: Set[tuple[str, str]],
                         found: Set[tuple[str, str]], *, uncertain: bool,
                         within: Within | None = None) -> frozenset[tuple[str, str]]:
        """The ``(entity_id, name)`` pairs whose holders a forget takes by rule:
        a name request's resolved entity's alone (`_scoped`), none when the request's referent
        was ``uncertain`` (several people, or two in one entity; the judge's picks decide
        alone), and a name of an entity two people may share (`_ambiguous`) only where its
        asked tie is told (``within``)."""
        if uncertain and not given:
            return frozenset()
        ambiguous = self._ambiguous(scope)
        # A form of the name in parentheses beside it is the same name ("林晓雨（Lin Xiaoyu）"
        # is "林晓雨" and "Lin Xiaoyu" too): it goes with it.
        return frozenset((entity, form) for entity, name in _scoped(given, found)
                         if entity not in ambiguous or entity in (within or {})
                         for form in name_forms(name))

    def _other_namers(self, scope: ScopeV1, targets: Mapping[str, _Candidate],
                      given: Set[tuple[str, str]]) -> list[str]:
        """The targets that only name another entity than the one a name request resolved to
        (``given``): the judge's pick of the coworker's "A person is named Max." for "forget my
        cat's name" is dropped, so neither it nor its name goes."""
        entities = {entity for entity, _ in given}
        if not entities:
            return []
        return [memory_id for memory_id, snapshot in self._heads(scope, list(targets)).items()
                if snapshot.naming and not {entry.entity_id for entry in snapshot.naming}
                & entities and not {snapshot.entity_id, *snapshot.about_entity_ids} & entities]

    def _unnamed_picks(self, scope: ScopeV1, targets: Mapping[str, _Candidate],
                       picks: Set[str], given: Set[tuple[str, str]],
                       statements: Sequence[str]) -> list[str]:
        """The judge's ``picks`` among ``targets`` that a request for a name alone does not ask
        for (every statement a name phrase and nothing else, `name_only`): once
        the request resolved to an entity (``given``, `_referent_namers`), each whose statement
        shows none of its names ("The user has a dog." for "忘掉我的狗的名字"; never a Han name
        inside a longer word, `_shows`); for any such request, each saying only that the user
        has someone or something ("The user has a sister." for "forget my sister's husband's
        name"). A pick giving the referent another name stays ("The user's cat is also called
        Bobo.": a naming memory of the entity, or one naming the user's own tie with a naming
        word): a nickname goes with the name. "忘掉我的狗和它的名字" asks for the dog too, so
        every pick stays. A request for a nickname ("忘掉我女儿的小名") drops instead every pick
        only naming or tying the referent by its name, and, once a memory gives someone a
        nickname, every pick that neither gives one nor shows one ("Rowan loves dinosaurs."
        beside "Rowan's nickname is Ollie.")."""
        if statements and any(nickname_request(statement) for statement in statements):
            # A nickname goes, not the name: a pick only naming the referent by its name or
            # tying it ("A person is named 林可欣.", "林可欣 is the user's daughter.") stays, and
            # so does one about them that shows no nickname.
            known = {name for snapshot in self._heads(
                scope, [row.memory_id for row in self._memories(scope)]).values()
                     for _, name in _aliases(snapshot)}
            return [memory_id for memory_id in picks if memory_id in targets
                    and _ALIAS_FORM.search(targets[memory_id].statement) is None and (
                        tied_name(targets[memory_id].statement) is not None
                        or naming_value(targets[memory_id].statement) is not None
                        or (bool(known) and not any(
                            _shows(name, targets[memory_id].statement) for name in known)))]
        if not statements or not all(name_only(statement) for statement in statements):
            return []
        rows = self._memories(scope)
        vocabulary = _vocabulary(row.statement for row in rows)
        names = {name for _, name in given}
        entities = {entity for entity, _ in given}
        tie_words = frozenset(word for word, covered in TIES.items() if covered & {
            tie.rsplit(LINKED, 1)[-1] for statement in statements
            for request in name_request(statement) for tie in request.ties})
        heads = self._heads(scope, [memory_id for memory_id in picks if memory_id in targets])

        def aliases(memory_id: str) -> bool:
            snapshot = heads.get(memory_id)
            return snapshot is not None and (
                any(entry.entity_id in entities for entry in snapshot.naming)
                or (_ALIAS.search(snapshot.statement) is not None
                    and _owned_tie(tie_words, snapshot.statement)))

        return [memory_id for memory_id in picks if memory_id in targets and not aliases(
            memory_id) and (
            _OWNS.fullmatch(targets[memory_id].statement.strip()) is not None or (
                bool(names) and not any(_shows(name, targets[memory_id].statement, vocabulary)
                                        for name in names)))]

    def _name_holders(self, scope: ScopeV1, names: Set[tuple[str, str]],
                      cover: Sequence[QuoteSpanV1] = (),
                      ties: Mapping[str, frozenset[str]] | None = None,
                      skip: Set[str] = frozenset(),
                      within: Within | None = None) -> list[_Candidate]:
        """The scope's readable memories about an entity whose name the forget takes that
        still show it as copied: a memory anchoring the entity of an
        ``(entity_id, name)`` pair of ``names`` whose statement holds the name as whole tokens
        ("李伟 attends a university in Beijing."), or a readable quote outside ``cover`` does
        ("I took Bob to the vet yesterday"; but a single Han character, a token of many Chinese
        words). A forget taking a naming memory takes them too, by rule: the judge may have
        found the memory unrelated, but it still tells the forgotten name. A memory about
        another entity of the same name, or using the word as an ordinary word ("The user
        drinks cola every day." for a dog named 可乐), anchors no such entity and stays."""
        return self._name_readers(scope, names, cover, ties, skip, within)[0]

    def _name_readers(self, scope: ScopeV1, names: Set[tuple[str, str]],
                      cover: Sequence[QuoteSpanV1] = (),
                      ties: Mapping[str, frozenset[str]] | None = None,
                      skip: Set[str] = frozenset(),
                      within: Within | None = None,
                      ) -> tuple[list[_Candidate], list[_Candidate]]:
        """The readable memories showing a name of ``names`` (`_name_holders`'s test), split
        into holders, which go with the name, and bystanders, which the forget shadow's names
        must not hide (`_record_kept`). A memory showing the forgotten name is a holder when it
        anchors that entity, names the user's own tie to it (`_refers`: "The user took the
        user's cat Bob to the vet.", anchored to the user alone), anchors another entity of
        that name and kind whose tie is unknown and that the user has no tie to, or anchors no
        one while no other person or animal bears the name and states the name itself ("The
        user calls Bob every day."); one anchoring another of the user's people, or naming no
        one beside another bearer (the coworker Max beside the ex-husband Max, a place named
        Paris), is none, and one anchoring no one holds it only where a name stands
        (`_named_at`: not "The user's birthday is in May." beside a friend May). It is
        a bystander only when it anchors another entity of another kind or tie (a coworker
        called Max beside the cat Max, "The user's coworker Max has a cat."), or none beside
        another bearer of the name, shows the name in its quotes alone, as a
        word ("The user drinks one bottle of cola every day." beside the dog 可乐), or shows
        it only inside a longer Han word ("The user likes eating 芝麻汤圆 for breakfast."
        beside the cat 芝麻, `_whole`) or inside another person's longer name ("The user is
        a huge fan of Michael Jordan." beside the ex Jordan, `_longer_names`). A capitalized
        Latin name in a Latin text counts only as written ("pepper" on eggs is no dog called
        Pepper). An entity's tie words are ``ties``' (`_tie_words`, read before the forget took
        its targets); ``skip`` names memories not to read (the forget's own targets). The name
        of an entity two people share is held only by a memory quoting nothing but the
        sentences ``within`` gives it."""
        if not names:
            return [], []
        rows = self._memories(scope)
        heads = self._heads(scope, [row.memory_id for row in rows])
        bearers: dict[str, set[str]] = {}
        for snapshot in heads.values():
            for entry in snapshot.naming:
                bearers.setdefault(entry.name.casefold(), set()).add(entry.entity_id)
        tie_words = ties or {}
        kind = self._kind
        longer = self._longer_names(scope, heads, names, tie_words)
        vocabulary = _vocabulary(row.statement for row in rows)
        named: dict[str, set[str]] = {}
        for snapshot in heads.values():
            for entry in snapshot.naming:
                named.setdefault(entry.entity_id, set()).add(entry.name.casefold())
        # The user's other people: an entity a memory ties to the user, by a naming tie ("Max
        # is the user's coworker.") or by the user's tie word right before its name ("The
        # user's colleague 志强 is a programmer."). One of them is never the forgotten one.
        tied = {entity for snapshot in heads.values()
                if tied_name(snapshot.statement) is not None for entity in _about(snapshot)}
        for snapshot in heads.values():
            for entity in {snapshot.entity_id, *snapshot.about_entity_ids} - {USER, ""}:
                if entity not in tied and any(_tied_before(snapshot.statement, name)
                                              for name in named.get(entity, ())):
                    tied.add(entity)

        def another(other: str, entity: str) -> bool:
            """Whether ``other`` is another person or animal than ``entity``: one the user has
            a tie to, or one of another kind (an entity of the same kind whose tie is unknown
            may be the forgotten one drawn twice). The forgotten one's kind may be gone with its
            last naming memory (the confirmation's cascade): then only a tie tells."""
            own = kind(entity)
            return other in tied or (bool(own) and kind(other) != own)

        def twice(other: str, entity: str, name: str) -> bool:
            """Whether ``other`` may be ``entity`` drawn twice: not another (`another`), and
            bearing ``name`` or no name at all ("May moved to Lyon in March." of a second,
            untied May; never "Michael Jordan" beside a forgotten Jordan)."""
            return not another(other, entity) and (
                other not in named or name.casefold() in named[other])

        holders: list[_Candidate] = []
        bystanders: list[_Candidate] = []
        for row in rows:
            snapshot = heads.get(row.memory_id)
            if snapshot is None or row.memory_id in skip:
                continue
            anchored = {snapshot.entity_id, *snapshot.about_entity_ids}
            texts = self._quoted(scope, [span for span in row.quotes if not any(
                span.evidence_id == other.evidence_id and span.byte_start < other.byte_end
                and other.byte_start < span.byte_end for other in cover)])
            shown = [(entity, name) for entity, name in names
                     if _shows(name, row.statement) or any(
                         _shows(name, text) for text in texts
                         if len(token_seq(name)) >= 2 or not own_token(name))]
            if not shown:
                continue
            holder = False
            for entity, name in shown:
                if within and entity in within and not all(
                        _inside(span, within[entity]) for span in row.quotes):
                    continue  # The other person of a shared entity.
                words = tie_words.get(entity, frozenset())
                parts = longer.get((entity, name), ())
                others = anchored - {entity, USER}
                rivals = {other for other in bearers.get(name.casefold(), set()) - {
                    entity, USER} if another(other, entity)}
                if entity in anchored or _refers(words, [row.statement, *texts], name):
                    holder = True
                elif parts and not any(
                        _shows(name, _masked(text, parts)) for text in (row.statement, *(
                            text for text in texts
                            if len(token_seq(name)) >= 2 or not own_token(name)))):
                    # The name shows only as a part of another person's longer name
                    # ("The user is a huge fan of Michael Jordan." beside the ex Jordan).
                    holder = False
                elif others:
                    # A same-kind entity drawn twice holds it, the one's tie known or not.
                    holder = any(twice(other, entity, name) for other in others)
                elif rivals:
                    # Another bears the name: a memory naming no one is told from it only by
                    # the forgotten one's anchor or tie ("The user's coworker Max sends the
                    # weekly report." beside the ex-husband Max stays).
                    holder = False
                else:
                    # Anchoring no one, from any conversation, it holds the name where a name
                    # stands ("The user took Bob to the vet."), never where the word is a month,
                    # a place or a thing (`_named_at`: "The user's birthday is in May." beside a
                    # friend May) or inside a longer Han word ("芝麻汤圆", `_whole`).
                    holder = _named_at(name, _masked(row.statement, parts), vocabulary)
                if holder:
                    break
            (holders if holder else bystanders).append(row)
        return holders, bystanders

    def _kind(self, entity: str) -> str | None:
        """The stored type of ``entity`` (its kind), empty once it is erased."""
        writer = self._writer
        with writer.write_scope():
            row = writer.get_row("entity", entity)
        return None if row is None else cast(str, row["entity_type"])

    def _longer_names(
        self, scope: ScopeV1, heads: Mapping[str, MemorySnapshotV2],
        names: Set[tuple[str, str]], ties: Mapping[str, frozenset[str]],
    ) -> dict[tuple[str, str], tuple[str, ...]]:
        """For each ``(entity, name)`` of ``names``, the longer names of other people the scope
        knows that hold ``name`` as one part ("Michael Jordan" beside the ex Jordan): another
        person's of the forgotten one's kind (an unknown kind, erased with its last naming
        memory, tells none apart) that shares no tie word with it (sharing one, the longer name
        is its own: "Jordan Lee" of the ex Jordan). Their memories keep the name, and the
        forget takes no token of it."""
        known: dict[str, set[str]] = {}
        for snapshot in heads.values():
            for entry in snapshot.naming:
                if entry.entity_id != USER:
                    known.setdefault(entry.entity_id, set()).add(entry.name)
        found: dict[tuple[str, str], tuple[str, ...]] = {}
        tie_words: dict[str, frozenset[str]] = {}
        for entity, name in names:
            own = self._kind(entity)
            mine = ties.get(entity, frozenset())
            longer: set[str] = set()
            for other, full in ((other, full) for other, named in known.items()
                                if other != entity for full in named
                                if _longer_name(name, full)):
                if own and self._kind(other) != own:
                    continue
                if mine:
                    if not tie_words:
                        tie_words = self._tie_words(scope, list(heads))
                    if mine & tie_words.get(other, frozenset()):
                        continue  # The forgotten one's own longer name.
                longer.add(full)
            if longer:
                found[(entity, name)] = tuple(sorted(longer))
        return found

    def _tie_words(self, scope: ScopeV1, targets: Sequence[str],
                   ) -> dict[str, frozenset[str]]:
        """The words naming each entity's tie to the user (猫, cat, 弟弟, brother; husband for
        the user's sister's husband, a chain's last link) that the ``targets``' naming ties give
        it (`naming_tie`), or the role word of a tie no tie word names ("manager" of "Priya is
        the user's manager.", `tied_name`), read before they are forgotten."""
        ties: dict[str, set[str]] = {}
        words: dict[str, set[str]] = {}
        heads = self._heads(scope, list(targets))
        for memory_id in targets:
            snapshot = heads.get(memory_id)
            parts = None if snapshot is None else tied_name(snapshot.statement)
            if snapshot is None or parts is None:
                continue
            tie = naming_tie(snapshot.statement)
            for entity in _about(snapshot):
                # A possessive chain's entity is told by its last link ("husband" of "the
                # user's sister's husband"); a role no tie word names, by its own word.
                ties.setdefault(entity, set()).update(
                    found.rsplit(LINKED, 1)[-1] for found in (() if tie is None else tie[0]))
                if tie is None:
                    words.setdefault(entity, set()).add(parts[0].split()[-1])
        return {entity: frozenset({*(word for word, covered in TIES.items()
                                     if covered & ties.get(entity, set())),
                                   *words.get(entity, ())})
                for entity in {*ties, *words}}

    def _sessions(self, scope: ScopeV1, rows: Iterable[_Candidate], session_id: str,
                  ) -> frozenset[str]:
        """The conversation sessions of the Evidence ``rows`` quote, and ``session_id``."""
        found = {session_id}
        writer = self._writer
        with writer.write_scope():
            for evidence_id in {span.evidence_id for row in rows for span in row.quotes}:
                part = writer.get_row("evidence", evidence_id)
                if part is not None and (part["scope_kind"], part["scope_id"]) == (
                        scope.scope_kind, scope.scope_id):
                    found.add(part["session_id"])
        return frozenset(found)

    def _hidden_by(
        self, scope: ScopeV1, forgotten: Sequence[tuple[str, Sequence[str]]], *,
        asked: Set[ContentBindingDigest] = frozenset(), left: Sequence[str] = (),
        seed: str | None = None, skip: Set[str] = frozenset(), gone: Set[str] = frozenset(),
    ) -> list[_Candidate]:
        """The current memories outside ``skip`` that the forget shadow a plan adds would hide
        from the list and the interactive Search (`recall_agent.forget_shadowed`, read as
        `_shadowed` reads a memory): the shadow of the ``forgotten`` statements and keywords,
        a statement naming an ``asked`` value beside another giving only the asked names as
        ``ForgetOwner._shadow`` does, of the names the rewordings leave out (``left``) and of
        the ``seed``. As Search reads it, a name `NAME_KEEP_MEMORIES` memories the plan leaves
        (all but ``gone``) still hold hides none (`recall_agent.surviving_names`). A forget
        reports them, so none goes from view silently."""
        terms = ForgetShadowTerms()
        writer = self._writer
        for statement, keywords in forgotten:
            found = shadow_terms(statement, keywords)
            if asked:
                own = {name for name in found.names
                       if writer.bind_content(SHADOW_PURPOSE, name) in asked}
                if own and own != set(found.names):
                    found = ForgetShadowTerms(frozenset(own), found.topics)
            terms = terms | found
        for form in left:
            terms = terms | shadow_terms(f"the name {form}")
        if seed is not None:
            terms = terms | shadow_terms(seed)
        shadow = ForgetShadow.of(terms)
        if not shadow:
            return []
        rows = self._memories(scope)
        # Read by its statement and keywords: its quotes are the forget's to cut, and a part
        # it withholds shows nothing afterwards. A Han run counts only within a forgotten one,
        # never as a piece of a longer word ("芝麻" of "芝麻汤圆" beside a cat named 芝麻).
        forgotten_runs = {run for statement, _ in forgotten
                          for run in _HAN_RUN.findall(statement)} | {
            run for form in left for run in _HAN_RUN.findall(form)}

        def read(row: _Candidate) -> frozenset[str]:
            text = " ".join((row.statement, *row.keywords))
            runs = _HAN_RUN.findall(text)
            return frozenset(term for term in sparse_terms(text) if term.isascii() or any(
                term in run and run in other for run in runs for other in forgotten_runs))

        held = {row.memory_id: read(row) for row in rows}
        shadow = shadow.without_names(surviving_names(shadow, (
            terms_ for memory_id, terms_ in held.items() if memory_id not in gone)))
        return [row for row in rows if row.memory_id not in skip
                and forget_shadowed(held[row.memory_id], None, True, shadow)]

    def _asked_shadow(self, statement: str, statements: Sequence[str],
                      ) -> frozenset[ContentBindingDigest]:
        """The keyed digests (``SHADOW_PURPOSE``) of the shadow names of the values the request
        asks of ``statements`` (the targets) that name another value too (`asked_sources`):
        the forget shadow keeps the value the user kept out (``ForgetOwner._shadow``)."""
        asked = {run for source in asked_sources(statement, statements)[1:]
                 for run in [source] if source not in statements}
        if not asked:
            return frozenset()
        names = {name for value in asked for name in shadow_terms(f"the name {value}").names}
        writer = self._writer
        return frozenset(writer.bind_content(SHADOW_PURPOSE, name) for name in names)

    def _value_token_spans(self, scope: ScopeV1, values: Sequence[str]) -> list[QuoteSpanV1]:
        """The occurrences of what a reworded target leaves out in the scope's readable
        sentences, its bytes alone, but a name some memory gives: "青柠科技"
        of "我上周离开了青柠科技，去了蓝鲸数据" beside "The user works at 青柠科技."
        reworded to "The user works at a company.", as the target's forget whole would have
        withheld it. A Latin value
        matches as whole words in any case; a Han one of two characters or more, never inside
        a longer Han word (`_whole`)."""
        rows = self._memories(scope)
        # A person's or an animal's name goes by the name rules, from its own entity's
        # sentences alone (`_name_token_spans`): another may bear it ("志强" the colleague).
        # A company's or a place's goes wherever it stands, a naming memory or none.
        heads = self._heads(scope, [row.memory_id for row in rows])
        writer = self._writer
        with writer.write_scope():
            kinds = {entry.entity_id: str(entity["entity_type"]).casefold()
                     for snapshot in heads.values() for entry in snapshot.naming
                     if (entity := writer.get_row("entity", entry.entity_id)) is not None}
        named = {name_norm(entry.name) for snapshot in heads.values()
                 for entry in snapshot.naming
                 if kinds.get(entry.entity_id, "person") in _BEARERS
                 or kinds.get(entry.entity_id, "") in TIES}
        wanted = [value for value in dict.fromkeys(values)
                  if value.strip() and not (len(value) < 2 and _han(value))
                  and name_norm(value) not in named]
        if not wanted:
            return []
        vocabulary = _vocabulary(row.statement for row in rows)
        with writer.write_scope():
            txn = TxnRows(writer, scope.scope_kind, scope.scope_id, 0, self._clock(), "")
            parts = [row for row in txn.pages(EvidenceRows(scope_kind=scope.scope_kind,
                                                           scope_id=scope.scope_id))
                     if row["state"] == "active"]
        found: list[QuoteSpanV1] = []
        for part in parts:
            for sentence, text in self._readable(part):
                for value in wanted:
                    # A capitalized Latin value matches as written: "Apple", not "apple pie".
                    pattern = re.compile(
                        rf"(?<![0-9A-Za-z_-]){re.escape(value)}(?![0-9A-Za-z_-])"
                    ) if value[:1].isupper() else value_pattern(value)
                    for match in pattern.finditer(text):
                        if _han(value) and not _whole(value, text, match.start(), match.end(),
                                                      vocabulary):
                            continue
                        first = sentence.byte_start + len(text[:match.start()].encode("utf-8"))
                        found.append(sentence.model_copy(update={
                            "byte_start": first,
                            "byte_end": first + len(match.group().encode("utf-8"))}))
        return found

    def _name_token_spans(
        self, scope: ScopeV1, targets: Sequence[str], names: Set[tuple[str, str]],
        sessions: Set[str], within: Within | None = None,
    ) -> list[QuoteSpanV1]:
        """The occurrences of a forgotten name in the scope's readable sentences, the name's
        bytes alone, for the forget to withhold with its targets: "bob" of
        "我昨天带bob去看兽医了，", "Bob" of "I took Bob to the vet yesterday", whatever memory
        quotes the sentence, so the rest stays readable and found and the name does not.

        Only a name exactly one known entity bears (``names``' pairs: no user's name; the
        naming annotations of the scope's memories and of the ``targets`` give the others, and
        so does a memory saying the name of another entity it is about) is withheld, never a
        single Han character. A Latin name matches as whole words, in any
        case in a Han text and as written or capitalized in a Latin one, so "Bobby" is no
        "Bob". A Han name, which may be an ordinary word too (a dog named 可乐 and "我每天都要
        喝一瓶可乐"), is withheld only in a sentence of the forget's own ``sessions`` (those
        the targets quote and the request's) or one that also names the user's own tie to the
        entity (`_refers`: 我的猫, my cat, 我弟弟, from what the targets say of it), and never
        inside a longer Han word ("我早餐喜欢吃芝麻汤圆" beside a cat named 芝麻, `_whole`) or
        another person's longer name ("Michael Jordan" beside the ex Jordan, `_longer_names`)."""
        if not names:
            return []
        rows = self._memories(scope)
        heads = self._heads(scope, [*{row.memory_id for row in rows}, *targets])
        bearers: dict[str, set[str]] = {}
        for snapshot in heads.values():
            for entry in snapshot.naming:
                bearers.setdefault(entry.name.casefold(), set()).add(entry.entity_id)
        ties = self._tie_words(scope, targets)
        # Never a part of another person's longer name ("Michael Jordan" beside the ex Jordan).
        longer = self._longer_names(scope, heads, names, ties)
        # Another entity a memory says the name of is a bearer too, named or not ("The user's
        # coworker Max has a cat." beside the cat Max, `_name_readers`' bystanders).
        for row in self._name_readers(scope, names, ties=ties)[1]:
            snapshot = heads.get(row.memory_id)
            if snapshot is None:
                continue
            for entity, name in names:
                if _shows(name, _masked(row.statement, longer.get((entity, name), ()))):
                    bearers.setdefault(name.casefold(), set()).update(
                        {snapshot.entity_id, *snapshot.about_entity_ids} - {entity, USER})
        scoped = within or {}
        wanted: list[tuple[str, str, frozenset[str]]] = []
        for entity, name in sorted(names):
            tokens = token_seq(name)
            if (entity == USER or not tokens or bearers.get(name.casefold(), {entity}) | {
                    entity} != {entity} or (len(tokens) == 1 and own_token(tokens[0]))):
                continue
            wanted.append((entity, name, ties.get(entity, frozenset())))
        if not wanted:
            return []
        vocabulary = _vocabulary(snapshot.statement for snapshot in heads.values())
        # What each entity's own memories quote: there its name goes wherever it shows.
        quoted_by: dict[str, list[QuoteSpanV1]] = {}
        listed = {row.memory_id for row in rows}
        for row in (*rows, *self._rows(scope, [memory_id for memory_id in targets
                                               if memory_id not in listed])):
            snapshot = heads.get(row.memory_id)
            if snapshot is None:
                continue
            for entity in {snapshot.entity_id, *snapshot.about_entity_ids,
                           *(entry.entity_id for entry in snapshot.naming)}:
                quoted_by.setdefault(entity, []).extend(row.quotes)
        writer = self._writer
        with writer.write_scope():
            txn = TxnRows(writer, scope.scope_kind, scope.scope_id, 0, self._clock(), "")
            parts = [row for row in txn.pages(EvidenceRows(scope_kind=scope.scope_kind,
                                                           scope_id=scope.scope_id))
                     if row["state"] == "active"]
        found: list[QuoteSpanV1] = []
        for part in parts:
            for sentence, text in self._readable(part):
                han = _mostly_own(text)
                for entity, name, words in wanted:
                    if entity in scoped and not _inside(sentence, scoped[entity]):
                        continue  # The other person of a shared entity.
                    # What the one's own memories quote loses the name wherever it shows;
                    # elsewhere a Latin name goes by the one's tie beside it or where a name
                    # stands ("I took Bob to the vet", never "in May", `_named_at`).
                    quoted = [span for span in quoted_by.get(entity, ())
                              if span.evidence_id == sentence.evidence_id
                              and span.byte_start < sentence.byte_end
                              and sentence.byte_start < span.byte_end]
                    placed = _refers(words, [text], name)
                    if any(own_token(token) for token in token_seq(name)):
                        if not quoted and not placed and part["session_id"] not in sessions:
                            continue
                        placed = True
                        pattern = re.compile(re.escape(name))
                    else:
                        body = r"\s+".join(re.escape(word) for word in name.split())
                        forms = {name, " ".join(word[:1].upper() + word[1:]
                                                for word in name.split())}
                        pattern = re.compile(
                            rf"(?<![0-9A-Za-z])(?:{body})(?![0-9A-Za-z])",
                            re.IGNORECASE) if han else re.compile(
                            rf"(?<![0-9A-Za-z])(?:{'|'.join(map(re.escape, forms))})"
                            r"(?![0-9A-Za-z])")
                    for match in pattern.finditer(_masked(text, longer.get(
                            (entity, name), ()))):
                        if _han(name) and not _whole(name, text, match.start(), match.end(),
                                                     vocabulary):
                            continue  # A piece of a longer Han word ("芝麻汤圆"), no name.
                        start = sentence.byte_start + len(text[:match.start()].encode("utf-8"))
                        stop = start + len(match.group().encode("utf-8"))
                        if not placed and not any(
                                span.byte_start <= start and stop <= span.byte_end
                                for span in quoted) and (
                                _dated(name, text, match.start(), match.end())):
                            continue  # A month or a weekday ("in May"), no name.
                        # Another form of the name in parentheses right after it goes too
                        # ("林晓雨（Lin Xiaoyu）", "Li Wei (李伟)"); one in the same script is
                        # no form of it ("李伟（我弟弟）", "Bob (Berlin)").
                        form = PARENTHESIZED_FORM.match(text, match.end())
                        end = match.end() if form is None or not other_script(
                            match.group(), form.group().strip()[1:-1]) else form.end()
                        first = sentence.byte_start + len(text[:match.start()].encode("utf-8"))
                        found.append(sentence.model_copy(update={
                            "byte_start": first,
                            "byte_end": first + len(text[match.start():end].encode("utf-8"))}))
            # A message telling the name may give the one named another, by a pronoun: "我们都
            # 叫她兰兰奶奶", "we call her Granny Lan". That name goes with it.
            if any(span.evidence_id == part["object_id"] for span in found):
                for sentence, text in self._readable(part):
                    for match in _CALLED.finditer(text):
                        group = "han" if match["han"] is not None else "latin"
                        first = sentence.byte_start + len(
                            text[:match.start(group)].encode("utf-8"))
                        found.append(sentence.model_copy(update={
                            "byte_start": first,
                            "byte_end": first + len(match[group].encode("utf-8"))}))
        return found

    def _bare_residue(
        self, scope: ScopeV1, current: Sequence[_Candidate], named: Set[str],
        gone: Sequence[str], spans: Sequence[QuoteSpanV1], requests: Sequence[str],
    ) -> list[_Candidate]:
        """The bare naming memories outside ``named`` (one naming its single non-user entity
        and anchoring nothing else) every quote of which lies in what the plan withholds
        (``spans``) or a memory of ``gone`` quotes: the cascade takes them as the forgotten
        fact's residue (`_leftovers`' second kind), unless another memory outside ``named`` is
        about the entity and its name is no part of what ``gone`` states or ``requests`` ask,
        or no forgotten statement shows it (`bare_name_stays`)."""
        cover: dict[str, list[tuple[int, int]]] = {}
        for span in (*spans, *(span for row in self._rows(scope, list(gone))
                               for span in row.quotes)):
            cover.setdefault(span.evidence_id, []).append((span.byte_start, span.byte_end))
        every = self._heads(scope, [row.memory_id for row in current])
        heads = {memory_id: snapshot for memory_id, snapshot in every.items()
                 if memory_id not in named}
        bare = {memory_id for memory_id, snapshot in heads.items() if naming_only(snapshot)}
        about = _told(snapshot for memory_id, snapshot in heads.items() if memory_id not in bare)
        names = {entry.name for snapshot in every.values() for entry in snapshot.naming}
        facts: list[ForgottenFact] = [
            *((snapshot.statement, snapshot.entity_id, snapshot.about_entity_ids)
              for snapshot in self._heads(scope, list(gone)).values()),
            *self._tombstone_facts(scope)]
        tied = names_tied((snapshot.statement, snapshot.entity_id, snapshot.about_entity_ids)
                          for snapshot in every.values())
        return [row for row in current if row.memory_id in bare and row.quotes
                and not bare_name_stays(heads[row.memory_id], requests, names, facts, tied,
                                        told=heads[row.memory_id].entity_id in about) and all(
            _covered(span.byte_start, span.byte_end, cover.get(span.evidence_id, []))
            for span in row.quotes)]

    def _heads(self, scope: ScopeV1, memory_ids: Sequence[str]) -> dict[str, MemorySnapshotV2]:
        """Each memory's head snapshot."""
        writer = self._writer
        found: dict[str, MemorySnapshotV2] = {}
        with writer.write_scope():
            for memory_id in memory_ids:
                item = writer.get_row("item", memory_id)
                if item is None or (item["scope_kind"], item["scope_id"]) != (
                        scope.scope_kind, scope.scope_id):
                    continue
                head = writer.get_row("revision", item["current_revision_id"])
                if head is not None:
                    found[memory_id] = MemorySnapshotV2.model_validate(head["snapshot"])
        return found

    def _restating_survivors(
        self, scope: ScopeV1, current: Sequence[_Candidate], named: Set[str],
        held: Sequence[QuoteSpanV1], stones_of: Sequence[str], words: Sequence[str],
    ) -> list[_Candidate]:
        """The memories outside ``named`` the confirmation's cascade may take as still
        restating what is forgotten (`_leftovers`' third kind: a memory privacy compensation
        requotes whose statement still restates a forgotten statement), for the prepare to name: a
        current memory privacy compensation may requote (a quote of it meets ``held``, what the plan
        withholds and the quotes of the memories forgotten beside it, or it carries a keyword
        the forget drops, ``forgotten_keywords``) whose statement restates a tombstoned
        statement (one of ``stones_of``'s, or the user's ``words``). A quote the plan leaves
        readable loses no byte, so its memory is not requoted and stays as it is ("Tom is the
        user's younger brother." on "My younger brother Tom" beside a forgotten "goes to
        college in Boston"; "The user's son is named Leo." on "My son Leo" beside a forgotten
        "at Willowbrook Primary")."""
        forgotten = list(dict.fromkeys([
            *(row.statement for row in self._rows(scope, list(stones_of))), *words]))
        stones = [(statement, terms) for statement in forgotten
                  if len(terms := content_terms(statement)) >= CHAT_SCAN_SHARED]
        return [row for row in current
                if row.memory_id not in named
                and (any(span.evidence_id == other.evidence_id
                         and span.byte_start < other.byte_end
                         and other.byte_start < span.byte_end
                         for span in row.quotes for other in held)
                     or forgotten_keywords(row.statement, row.keywords, forgotten,
                                           self._quote_texts(row.quotes)))
                and any(_restates(row.statement, [terms]) and support_words(
                    row.statement) & _said(statement, row.statement,
                                           modifier=self._modifier(words))
                        and not (names_another(statement, row.statement) and not all(
                            text.isascii() for text in self._quote_texts(row.quotes)))
                        for statement, terms in stones)]

    def _clause_roles(
        self, statement: str, picked: Mapping[int, _Candidate], sender: Send,
    ) -> tuple[dict[int, list[str]], int]:
        """The judge's role for each clause of the picked memories that split into clauses
        (`forget_match.clauses`), in one pick of at most ``MAX_FORGET_CANDIDATES`` clauses,
        by memory, and the calls; a memory of one clause is left out."""
        pieces: list[tuple[int, str]] = []
        for index, row in picked.items():
            parts = clauses(row.statement)
            if len(parts) > 1 and len(pieces) + len(parts) <= MAX_FORGET_CANDIDATES:
                pieces += [(index, part) for part in parts]
        if not pieces:
            return {}, 0
        pick = pick_targets(statement, [text for _, text in pieces], sender, rounds=_rounds,
                            minimal=False)
        roles: dict[int, list[str]] = {}
        for (index, _), role in zip(pieces, pick.roles, strict=True):
            roles.setdefault(index, []).append(role)
        return roles, pick.calls

    def _chat_seeds(
        self, statements: Sequence[str], phrases: Sequence[_Phrase],
        asked: Mapping[int, Mapping[int, _Candidate]], sender: Send,
    ) -> tuple[dict[str, ForgetSeedV1], int]:
        """The in-chat forget's bundled targets' tombstones, and the calls.

        Every memory a request picked is judged by clauses against every request of the
        Add, as one Add may ask to forget two facts one memory bundles ("forget that I play
        tennis, and that I am learning Japanese"). A memory is bundled, and tombstoned by a
        request's words, only when its clauses state that request alone and one clause is
        ``unrelated`` to every request; otherwise its own statement is tombstoned.
        """
        rows = {row.memory_id: row for picked in asked.values() for row in picked.values()}
        if not rows:
            return {}, 0
        indexed = dict(enumerate(rows.values()))
        calls = 0
        stating: dict[int, set[int]] = {}
        other: dict[int, list[bool]] = {}
        for position, statement in enumerate(statements):
            roles, more = self._clause_roles(statement, indexed, sender)
            calls += more
            for index, found in roles.items():
                if any(role in _PARTS for role in found):
                    stating.setdefault(index, set()).add(position)
                unrelated = [role == "unrelated" for role in found]
                other[index] = ([a and b for a, b in zip(other[index], unrelated, strict=True)]
                                if index in other else unrelated)
        seeds: dict[str, ForgetSeedV1] = {}
        for index, positions in stating.items():
            if len(positions) != 1 or not any(other.get(index, [])):
                continue
            (position,) = positions
            seed = self._bundle_seed(statements[position], phrases[position])
            if seed is not None:
                seeds[indexed[index].memory_id] = seed
        return seeds, calls

    def _phrase(self, statement: str) -> tuple[_Phrase | None, int]:
        """The statement's vector in the workspace's active space and the embedding calls
        made (none without an embedder)."""
        phrases, calls = self._phrases([statement])
        return (phrases[0] if phrases else None), calls

    def _phrases(self, statements: Sequence[str]) -> tuple[list[_Phrase], int]:
        """The statements' vectors in the workspace's active space, by one embedding call, and
        the calls made (none, and no vectors, without an embedder or a statement)."""
        if self._embedder is None or not statements:
            return [], 0
        space, vectors = workspace_embedding(self._writer, self._embedder, statements)
        return [_Phrase(unit_vector(raw), raw, space) for raw in vectors], 1

    @staticmethod
    def _bundle_seed(statement: str, phrase: _Phrase | None) -> ForgetSeedV1 | None:
        """A bundled target's tombstone: the user's words of the fact and their
        vector; None, so the bundle's own statement is tombstoned, without a vector or for
        words that are no canonical statement."""
        if phrase is None:
            return None
        try:
            text = canonical_statement(asked_part(statement))
        except ValueError:
            return None
        return ForgetSeedV1(statement=text, embedding=list(phrase.raw),
                            embedding_space=phrase.space)

    @staticmethod
    def _nearest(rows: Sequence[_Candidate], phrase: _Phrase,
                 floor: float = DENSE_FLOOR) -> list[_Candidate]:
        """The memories at ``floor`` or more to the statement's vector, nearest first."""
        scored = [(cosine(phrase.vector, unit_vector(row.vector)), row) for row in rows
                  if row.vector and row.space == phrase.space]
        return [row for score, row in sorted(scored, key=lambda entry: -entry[0])
                if score >= floor]

    def _hits(self, scope: ScopeV1, user_id: str, query: str, top_k: int) -> list[str]:
        """The ids a statement's own Search returns, in rank order (bench ranking)."""
        search = self._search
        if search is None:
            raise QuoteMemError("service_unhealthy", "Forget by statement has no Search")
        delivery = _Delivery(SEARCH_SECONDS)
        search(SearchRequest(query=query, user_id=user_id, top_k=top_k), delivery)
        if delivery.result is None or any(hit.scope_id != scope.scope_id
                                          for hit in delivery.result.hits):
            raise QuoteMemError("service_unhealthy", "The statement's Search answered no result")
        return [hit.id for hit in delivery.result.hits]

    def _matched(
        self, scope: ScopeV1, hit_ids: Sequence[str], phrase: _Phrase | None,
        current: Sequence[_Candidate], *, after: int | None = None,
    ) -> tuple[list[_Candidate], int, int]:
        """A complete forget's memory candidates for one request, the prepare's and the
        in-chat forget's alike: the current memory hits of its Search that do
        not record a forget request, then the current memories nearest its vector
        (``CANDIDATE_FLOOR``, ``DENSE_MEMORIES`` places) that Search did not return,
        ``MAX_FORGET_CANDIDATES`` in all; with ``after``, only memories a transaction after
        that commit changed. Also the scope's ``erase_epoch`` and the commit read.

        The floor is lower than the sentences': a restatement in other words or another
        language lies at 0.59-0.73 to the user's words in the agent mode's embedding space,
        an unrelated memory of the same user at 0.27-0.51; the judge decides, and the
        confusable-target veto still removes another subject or the opposite claim.
        """
        read_seq = self._commit_seq()
        rows, erase_epoch = self._candidates(scope, hit_ids, limit=False)
        changed = None if after is None else {
            row.memory_id for row in self._memories(scope, after=after)}
        rows = [row for row in rows if (changed is None or row.memory_id in changed)
                and not self._records_request(scope, row.quotes, complete=True)]
        dense: list[_Candidate] = []
        if phrase is not None:
            known = {row.memory_id for row in rows}
            dense = [row for row in self._nearest(current, phrase, CANDIDATE_FLOOR)
                     if row.memory_id not in known
                     and (changed is None or row.memory_id in changed)
                     and not self._records_request(scope, row.quotes, complete=True)
                     ][:DENSE_MEMORIES]
        return [*rows[:MAX_FORGET_CANDIDATES - len(dense)], *dense], erase_epoch, read_seq

    def _pick_sender(self) -> Send:
        """The sender of a forget's picks: a complete forget's (the prepare and the in-chat
        forget alike) without model reasoning where the provider takes it,
        ``AGENT_PICK_TIMEOUT_MS`` and hedged after ``AGENT_PICK_HEDGE_SECONDS``;
        any other forget's at the judge binding's own effort."""
        if not self._complete():
            return self._sender()
        # The effort stays bound (an endpoint that drops ``thinking`` reads it); the
        # adapter sends ``thinking`` in its place.
        binding = self._judge_binding().model_copy(
            update={"reasoning_effort": CHAT_FORGET_REASONING_EFFORT})
        thinking = binding.provider == "openai"
        return forget_pick_sender(cast(StructuredInferenceProvider, self._provider), binding,
                                  self._writer.bind_content, thinking=thinking,
                                  timeout_ms=AGENT_PICK_TIMEOUT_MS,
                                  hedge_seconds=AGENT_PICK_HEDGE_SECONDS)

    def _remember_phrase(self, scope: ScopeV1, statement: str, phrase: _Phrase) -> None:
        """Keep a prepared plan's statement vector, when it bundles a target, for its
        confirmation (``PHRASE_CACHE`` of them, in this process only, each dropped once
        used), which then embeds nothing."""
        cache = self._phrase_cache
        cache.pop((scope.scope_id, statement), None)
        cache[scope.scope_id, statement] = phrase
        while len(cache) > PHRASE_CACHE:
            cache.pop(next(iter(cache)))

    # Both paths of one request ------------------------------------------
    def _handled_key(self, scope: ScopeV1, session_id: str, statement: str) -> str:
        """The keyed digest naming one request of one session in ``forget_handled:{s}``: its
        words without case, punctuation or pointing words (`forget_match.probe`), and without
        the asking words a plugin's forget by statement may still carry ("请忘掉我的猫的名字",
        "forget my cat's name"): what the Add's in-chat forget reads of the same turn
        ("我的猫的名字", "my cat's name", `forget_requests.requested`), so
        both paths name it alike."""
        asked = requested(statement, complete=True)
        return self._writer.bind_content(HANDLED_PURPOSE, {
            "scope": scope.scope_id, "session": session_id,
            "words": _words(probe(asked[0] if len(asked) == 1 else statement))})

    def _asked_key(self, scope: ScopeV1, request_id: str) -> str:
        """The keyed digest of a forget by statement's ``request_id``."""
        return self._writer.bind_content(HANDLED_PURPOSE, {
            "scope": scope.scope_id, "request": request_id})

    def _handled_entries(self, scope: ScopeV1) -> list[ForgetHandledEntryV1]:
        stored = self._writer.get_control(f"forget_handled:{scope.scope_id}")
        return [] if stored is None else list(
            ForgetHandledV1.model_validate(stored["value"]).entries)

    def _entry(self, scope: ScopeV1, *, key: str | None = None,
               asked: str | None = None) -> ForgetHandledEntryV1 | None:
        """The handled record of a request, by its words' key or its ``request_id``."""
        wanted = None if asked is None else self._asked_key(scope, asked)
        for entry in reversed(self._handled_entries(scope)):
            if entry.request == key or (wanted is not None and entry.asked == wanted):
                return entry
        return None

    def _earlier(self, scope: ScopeV1, key: str | None = None, *,
                 asked: str | None = None) -> ForgetStatementEarlierV1 | None:
        """What both paths did for a request, as a response reports it (complete only)."""
        if not self._complete():
            return None
        entry = self._entry(scope, key=key, asked=asked)
        return None if entry is None else ForgetStatementEarlierV1(
            statement=entry.statement, in_chat=entry.in_chat, forgotten=entry.forgotten,
            withheld=entry.withheld, at=entry.at)

    def earlier(self, scope: ScopeV1, request_id: str) -> ForgetStatementEarlierV1 | None:
        """Both paths' record of the request a confirmation names (``agent`` mode)."""
        return self._earlier(scope, asked=request_id)

    def _record_handled(
        self, scope: ScopeV1, key: str, *, statement_seq: int | None = None,
        chat_seq: int | None = None, statement: HandledStatement | None = None,
        in_chat: HandledInChat | None = None, asked: str | None = None, forgotten: int = 0,
        withheld: int = 0,
    ) -> None:
        """Record what a path did for one request (complete forgets only): a prepare's or a
        confirmation's result and the commit it read, an in-chat forget's forgets."""
        if not self._complete():
            return
        writer = self._writer
        control_key = f"forget_handled:{scope.scope_id}"
        with writer.write_scope():
            stored = writer.get_control(control_key)
            entries = [] if stored is None else list(
                ForgetHandledV1.model_validate(stored["value"]).entries)
            found = next((row for row in entries if row.request == key), None)
            now = self._clock()
            update: dict[str, object] = {
                "at": now, "forgotten": (0 if found is None else found.forgotten) + forgotten,
                "withheld": (0 if found is None else found.withheld) + withheld}
            if found is not None and found.statement == "forgotten" and statement != "forgotten":
                # A forgotten request stays forgotten, at its confirmation's time;
                # a later prepare finding nothing more says nothing new.
                update["at"], statement = found.at, None
            if found is not None and found.in_chat == "forgotten":
                in_chat = None
            for field_name, value in (("statement", statement), ("in_chat", in_chat),
                                      ("statement_seq", statement_seq),
                                      ("chat_seq", chat_seq)):
                if value is not None:
                    update[field_name] = value
            if asked is not None:
                update["asked"] = self._asked_key(scope, asked)
            row = (ForgetHandledEntryV1.model_validate({"request": key, **update})
                   if found is None else found.model_copy(update=update))
            kept = [other for other in entries if other.request != row.request]
            value = ForgetHandledV1(entries=[*kept, row][-MAX_FORGET_HANDLED:])
            control = writer.materialize_control(
                control_key, value.model_dump(mode="json"),
                0 if stored is None else stored["version"] + 1, now)
            _, plan = writer.plan_rows(
                [RowWrite("control", control)], None)
            writer.write_plan(plan)

    def _created_by(self, scope: ScopeV1, request_id: str) -> set[str]:
        """The memories the Add ``request_id`` made: created since its Evidence was
        received and supported by that Evidence. A memory another Add of the same user made
        meanwhile quotes its own Add's Evidence and is not counted."""
        writer = self._writer
        with writer.write_scope():
            txn = TxnRows(writer, scope.scope_kind, scope.scope_id, 0, self._clock(), "")
            parts = list(txn.pages(EvidenceRows(
                scope_kind=scope.scope_kind, scope_id=scope.scope_id, request_id=request_id)))
            if not parts:
                return set()
            since = min(row["ingested_at"] for row in parts)
            mine = {row["object_id"] for row in parts}
            found: set[str] = set()
            for item in txn.pages(MemoryTableRows(
                    scope_kind=scope.scope_kind, scope_id=scope.scope_id,
                    after_commit_seq=max(0, min(row["ingested_commit_seq"] for row in parts) - 1))):
                if item["created_at"] < since:
                    continue
                head = writer.get_row("revision", item["current_revision_id"])
                if head is not None and mine.intersection(head["evidence_ids"]):
                    found.add(item["object_id"])
            return found

    def _fresh(self, scope: ScopeV1, reads: Iterable[int | None],
               request_id: str | None) -> tuple[set[str], set[str]]:
        """The memories quoting Evidence ingested after the commits forgets by statement read
        the scope up to (``reads``), and that Evidence: what the user told since,
        never what those forgets read, kept or changed. With a commit unknown, only the Add
        ``request_id``'s own memories (`_created_by`) and no Evidence."""
        seqs = list(reads)
        if not seqs or any(seq is None for seq in seqs):
            return (set() if request_id is None else self._created_by(scope, request_id)), set()
        after = min(seq for seq in seqs if seq is not None)
        writer = self._writer
        with writer.write_scope():
            txn = TxnRows(writer, scope.scope_kind, scope.scope_id, 0, self._clock(), "")
            told = {row["object_id"] for row in txn.pages(EvidenceRows(
                scope_kind=scope.scope_kind, scope_id=scope.scope_id))
                if row["ingested_commit_seq"] > after}
        return {row.memory_id for row in self._memories(scope)
                if any(span.evidence_id in told for span in row.quotes)}, told

    def _told_new(self, span: QuoteSpanV1) -> bool:
        """Whether a span lies in what a user told after a request in the request's sentence
        (`recall_agent.replacement_ranges`): "I moved to 27 Birch Lane" of "Forget my old
        address, I moved to 27 Birch Lane.", or "我现在住在银杏路6号" after "忘掉我的旧地址，"."""
        writer = self._writer
        with writer.write_scope():
            row = writer.get_row("evidence", span.evidence_id)
        if row is None or row["role"] != "user":
            return False
        text: str = row["text"]
        return any(row["byte_start"] + low <= span.byte_start
                   and span.byte_end <= row["byte_start"] + high
                   for low, high in replacement_ranges(
                       text, lambda words: requested(words, complete=True), request_reach))

    def _request_said(self, scope: ScopeV1, request_id: str,
                      statements: Sequence[str]) -> list[str]:
        """The sentences of the Add ``request_id`` that make its requests (``statements``), as
        the user said them, else the statements themselves: what a request told with its
        words is read from them (`_states_new`)."""
        writer = self._writer
        with writer.write_scope():
            txn = TxnRows(writer, scope.scope_kind, scope.scope_id, 0, self._clock(), "")
            texts = [part["text"] for part in txn.pages(EvidenceRows(
                scope_kind=scope.scope_kind, scope_id=scope.scope_id, request_id=request_id))
                if part["role"] == "user"]
        said: dict[str, str] = {}
        for text in texts:
            for start, end in request_sentences(text):
                for statement in requested(text[start:end], complete=True):
                    said.setdefault(statement, text[start:end])
        return [said.get(statement, statement) for statement in statements]

    def _replacements(self, scope: ScopeV1, request_id: str) -> set[str]:
        """The memories the Add ``request_id`` made of a clear replacement said with its
        request ("Forget my old address, I live at 27 Birch Lane now."): every quote lies in a
        request sentence that a replacement ends (`forget_requests.request_reach`) and reaches
        past the request, and the statement gives a value word of the replacement
        ("27 Birch Lane", "银杏路6号") and none of the request's own words. Its own forget never
        takes them: the request asks to forget the old value, and once the Add corrected it
        the new value is the only current one ("my old address" would read as the new
        address)."""
        own = self._created_by(scope, request_id)
        found: set[str] = set()
        writer = self._writer
        for row in self._memories(scope):
            if row.memory_id not in own or not row.quotes:
                continue
            values: set[str] = set()
            asked: set[str] = set()
            inside = True
            for span in row.quotes:
                with writer.write_scope():
                    part = writer.get_row("evidence", span.evidence_id)
                if part is None or part["role"] != "user":
                    inside = False
                    break
                text: str = part["text"]
                low, high = span.byte_start - part["byte_start"], span.byte_end - part["byte_start"]
                inside = False
                for start, end in request_sentences(text):
                    reach = request_reach(text[start:end])
                    if reach is None or start + reach >= end:
                        continue
                    first = len(text[:start].encode("utf-8"))
                    if (first <= low and len(text[:start + reach].encode("utf-8")) < high
                            <= len(text[:end].encode("utf-8"))):
                        inside = True
                        values |= _value_words(text[start + reach:end])
                        asked |= content_terms(text[start:start + reach])
                        break
                if not inside:
                    break
            said = content_terms(row.statement)
            if inside and said & values and not said & asked - _ASKED_WORDS:
                found.add(row.memory_id)
        return found

    def _record_add(
        self, scope: ScopeV1, request_id: str | None, requests: Sequence[ChatForgetRequest],
        taken: int,
    ) -> tuple[list[ChatForgetRequest], int]:
        """Record what one Add's in-chat forget came to (``forget_add:{s}``) and return it
        merged with what an earlier run of the same Add recorded.

        An Add is run again on every resend, and a later run reads only what changed, so a
        request once ``forgotten`` stays so with its counts, a ``pending`` one takes the
        later result, and ``taken`` (the Add's own memories its forget took) keeps its
        largest value. The record names the Add and each request by keyed digests only.
        """
        if request_id is None:
            return list(requests), taken
        writer = self._writer
        control_key = f"forget_add:{scope.scope_id}"
        add_key = self._writer.bind_content(HANDLED_PURPOSE, {
            "scope": scope.scope_id, "add": request_id})
        keys = [self._writer.bind_content(HANDLED_PURPOSE, {
            "scope": scope.scope_id, "add": request_id, "request": position,
            "words": _words(probe(row.statement))}) for position, row in enumerate(requests)]
        with writer.write_scope():
            stored = writer.get_control(control_key)
            entries = [] if stored is None else list(
                ForgetAddV1.model_validate(stored["value"]).entries)
            found = next((entry for entry in entries if entry.add == add_key), None)
            earlier = {} if found is None else {row.request: row for row in found.requests}
            merged: list[ChatForgetRequest] = []
            for key, row in zip(keys, requests, strict=True):
                old = earlier.get(key)
                if old is not None and old.status == "forgotten":
                    row = ChatForgetRequest(
                        statement=row.statement, status="forgotten",
                        forgotten=max(old.forgotten, row.forgotten),
                        withheld=max(old.withheld, row.withheld),
                        bundled=max(old.bundled, row.bundled),
                        earlier=old.earlier or row.earlier)
                merged.append(row)
            taken = max(taken, 0 if found is None else found.taken)
            entry = ForgetAddEntryV1(add=add_key, at=self._clock(), taken=taken, requests=[
                ForgetAddRequestV1(request=key, status=row.status, forgotten=row.forgotten,
                                   withheld=row.withheld, bundled=row.bundled,
                                   earlier=row.earlier)
                for key, row in zip(keys, merged, strict=True)][:MAX_FORGET_ADD_REQUESTS])
            if found is None or found.model_dump(exclude={"at"}) != entry.model_dump(
                    exclude={"at"}):
                kept = [other for other in entries if other.add != add_key]
                value = ForgetAddV1(entries=[*kept, entry][-MAX_FORGET_ADDS:])
                control = writer.materialize_control(
                    control_key, value.model_dump(mode="json"),
                    0 if stored is None else stored["version"] + 1, self._clock())
                _, plan = writer.plan_rows(
                    [RowWrite("control", control)], None)
                writer.write_plan(plan)
        return merged, taken

    def _write_kept(self, scope: ScopeV1, change: Callable[[ForgetKeptV1], ForgetKeptV1]) -> None:
        """Rewrite the scope's judged-and-kept record (``forget_kept:{s}``) by ``change``,
        expired plans dropped; nothing is written when it is unchanged."""
        writer = self._writer
        control_key = f"forget_kept:{scope.scope_id}"
        with writer.write_scope():
            stored = writer.get_control(control_key)
            current = ForgetKeptV1() if stored is None else ForgetKeptV1.model_validate(
                stored["value"])
            now = self._clock()
            value = change(current.model_copy(update={
                "plans": [plan for plan in current.plans if plan.expires_at > now]}))
            if value == current:
                return
            control = writer.materialize_control(
                control_key, value.model_dump(mode="json"),
                0 if stored is None else stored["version"] + 1, now)
            _, plan = writer.plan_rows(
                [RowWrite("control", control)], None)
            writer.write_plan(plan)

    def _modifier(self, words: Sequence[str]) -> bool:
        """Whether the user's ``words`` ask to forget only a modifier of a thing of theirs
        (``asks_modifier``: "忘掉我的猫是橘猫"), in a workspace that forgets completely: only
        then does the plan keep a survivor about that thing (``ForgetIntentV1.modifier_only``);
        any other forget of "The user has an orange cat." forgets the possession."""
        return self._owner.forgets_completely() and asks_modifier(words)

    def _remember_kept(
        self, scope: ScopeV1, plan: ContentBindingDigest, expires_at: int, memories: Set[str],
        judged: Sequence[_Judged], *, listed: Set[str] = frozenset(),
        past: Set[str] = frozenset(), allowed: Set[str] = frozenset(), nickname: bool = False,
        residue: Set[str] = frozenset(), modifier: bool = False, name_or_all: bool = False,
        rewordings: Sequence[ForgetRewordPlanV1] = (),
        asked: Set[ContentBindingDigest] = frozenset(), name_asked: bool = False,
    ) -> None:
        """Keep what a complete prepare judged and did not take until its plan is confirmed
        (``forget_kept:{s}``'s ``plans``): its confirmation records it (`_record_kept`), takes
        no memory outside ``listed`` (`_cascade`, ``ForgetOwner.prepare``), beyond the base
        links only what the judge confirmed (``allowed``), of the bare names only its
        ``residue`` (with the memories still showing them when its words ask a name or
        everything about someone, ``name_or_all``), forgets a modifier alone when the prepare
        did (``modifier``, `_modifier`), lets only a target in ``past`` be past
        (`_check_unchanged`) and rewords the ``rewordings``' memories (keeping the user's
        ``asked`` values out of the shadow). Every plan is kept, empty or not, so its
        confirmation knows a record is gone (``confirm_report``).

        Raises:
            QuoteMemError: ``budget_exceeded`` when the plan names more than
                ``MAX_FORGET_KEPT_HITS`` memories, which the record would cut.
        """
        spans = sorted({(entry.span.evidence_id, entry.span.byte_start, entry.span.byte_end):
                        entry.span for entry in judged
                        if entry.role is not None and not entry.target}.items())
        if max(len(listed), len(past), len(allowed), len(residue)) > MAX_FORGET_KEPT_HITS:
            # Cut, the record would let the confirmation take less than the plan names.
            raise QuoteMemError("budget_exceeded", "The forget plan names too many memories")
        entry = ForgetKeptPlanV1(
            plan=plan, expires_at=expires_at,
            memories=sorted(memories)[:MAX_FORGET_KEPT_HITS],
            spans=[span for _, span in spans][:MAX_FORGET_KEPT_HITS],
            listed=sorted(listed)[:MAX_FORGET_KEPT_HITS],
            past=sorted(past)[:MAX_FORGET_KEPT_HITS],
            allowed=sorted(allowed)[:MAX_FORGET_KEPT_HITS], nickname=nickname,
            residue=sorted(residue)[:MAX_FORGET_KEPT_HITS], modifier=modifier,
            name_or_all=name_or_all, rewordings=list(rewordings)[:MAX_REWORDINGS],
            asked=sorted(asked)[:MAX_KEYWORDS], name_asked=name_asked)
        self._write_kept(scope, lambda value: value.model_copy(update={"plans": [
            *(other for other in value.plans if other.plan != plan), entry,
        ][-MAX_FORGET_KEPT_PLANS:]}))

    def _planned(self, scope: ScopeV1, plan: ContentBindingDigest) -> _PlanRecord:
        """What a complete prepare recorded of its plan (`_remember_kept`): the memories it
        named and the targets it read past; none once the record is gone."""
        writer = self._writer
        with writer.write_scope():
            stored = writer.get_control(f"forget_kept:{scope.scope_id}")
        rows = [] if stored is None else [
            row for row in ForgetKeptV1.model_validate(stored["value"]).plans if row.plan == plan]
        return _PlanRecord(frozenset(memory_id for row in rows for memory_id in row.listed),
                           frozenset(memory_id for row in rows for memory_id in row.past),
                           frozenset(memory_id for row in rows for memory_id in row.allowed),
                           bool(rows), any(row.nickname for row in rows),
                           frozenset(memory_id for row in rows for memory_id in row.residue),
                           any(row.modifier for row in rows), any(row.name_or_all for row in rows),
                           tuple(plan for row in rows for plan in row.rewordings),
                           frozenset(digest for row in rows for digest in row.asked),
                           any(row.name_asked for row in rows))

    def _past(self, scope: ScopeV1, memory_ids: Set[str]) -> set[str]:
        """The memories of ``memory_ids`` that are not current."""
        writer = self._writer
        with writer.write_scope():
            return {memory_id for memory_id in memory_ids
                    if (item := writer.get_row("item", memory_id)) is not None
                    and (item["scope_kind"], item["scope_id"]) == (
                        scope.scope_kind, scope.scope_id)
                    and item["lifecycle_state"] != "active"}

    def _record_kept(
        self, scope: ScopeV1, start: int, *, plan: ContentBindingDigest | None = None,
        memories: Set[str] = frozenset(), judged: Sequence[_Judged] = (),
        cover: Sequence[QuoteSpanV1] = (), ours: Set[str] | None = None,
        held: Mapping[str, Sequence[tuple[int, int]]] | None = None, write: bool = True,
    ) -> tuple[frozenset[str], frozenset[str]]:
        """Record what a complete forget that began at ``start`` judged and kept, so the
        names it stamped do not hide it from the interactive Search; the names it
        stamped (keys) and the memories recorded kept, which the caller reports from
        (`_hidden`). A prepare foresees them (`_foreseen_hidden`): given the names its forgets
        will stamp (``ours``) and the ranges its plan withholds (``held``), it records
        nothing (``write`` False).

        The forget's names are the shadow names stamped since ``start``. A kept memory is
        recorded when each of its quotes naming one of them lies in a sentence the forget
        judged and kept, or is the readable rest of a quote of its targets (``cover``) that
        the forget's own cut left readable for it and that shows no stamped name but the names
        of what the memory is about ("My daughter Ruby" of a forgotten "My daughter Ruby goes
        to Maple Grove Elementary." for "The user's daughter is named Ruby."); a part holding a
        kept sentence when each of its readable sentences naming one of them is such a
        sentence. Anything else naming them stays hidden: the forget never read it (an older
        message naming the forgotten pet). A confirmation reads its prepare's judgments from
        ``plan`` (`_remember_kept`).
        """
        writer = self._writer
        found: list[ForgetKeptPlanV1] = []
        if plan is not None:
            with writer.write_scope():
                stored_kept = writer.get_control(f"forget_kept:{scope.scope_id}")
            found = [] if stored_kept is None else [
                row for row in ForgetKeptV1.model_validate(stored_kept["value"]).plans
                if row.plan == plan]
            memories = set(memories) | {memory_id for row in found for memory_id in row.memories}
            spans = [span for row in found for span in row.spans]
        else:
            spans = [entry.span for entry in judged if entry.role is not None
                     and not entry.target]
        end = self._clock()
        if ours is None:
            with writer.write_scope():
                stored = writer.get_control(f"forget_shadow:{scope.scope_id}")
            shadow = None if stored is None else ForgetShadowV1.model_validate(stored["value"])
            ours = set[str]() if shadow is None else {
                name for name, at in zip(shadow.names, shadow.name_times, strict=False)
                if start <= at <= end}
        planned = held or {}
        kept_memories: list[str] = []
        kept_parts: list[str] = []
        if ours and (memories or spans):
            keys = {(span.evidence_id, span.byte_start, span.byte_end) for span in spans}

            def naming(text: str) -> bool:
                return any(writer.bind_content(SHADOW_PURPOSE, term) in ours
                           for term in sparse_terms(text))

            def within(span: QuoteSpanV1, others: Sequence[QuoteSpanV1]) -> bool:
                return any(span.evidence_id == other.evidence_id
                           and other.byte_start <= span.byte_start
                           and span.byte_end <= other.byte_end for other in others)

            def meets(span: QuoteSpanV1, others: Sequence[QuoteSpanV1]) -> bool:
                return any(span.evidence_id == other.evidence_id
                           and other.byte_start < span.byte_end
                           and span.byte_start < other.byte_end for other in others)

            def outside(span: QuoteSpanV1, others: Sequence[QuoteSpanV1]) -> list[QuoteSpanV1]:
                pieces = [(span.byte_start, span.byte_end)]
                for other in others:
                    if other.evidence_id != span.evidence_id:
                        continue
                    pieces = [piece for low, high in pieces for piece in (
                        (low, min(high, other.byte_start)), (max(low, other.byte_end), high))
                        if piece[0] < piece[1]]
                return [QuoteSpanV1(evidence_id=span.evidence_id, byte_start=low,
                                    byte_end=high) for low, high in pieces]

            def stamped(text: str) -> set[str]:
                return {bound for term in sparse_terms(text)
                        if (bound := writer.bind_content(SHADOW_PURPOSE, term)) in ours}

            def inside(span: QuoteSpanV1, text: str, own: Set[str]) -> bool:
                if within(span, spans):
                    return True
                with writer.write_scope():
                    part = writer.get_row("evidence", span.evidence_id)
                if part is not None and not any(
                        span.byte_start < high and low < span.byte_end
                        for low, high in (*planned.get(span.evidence_id, ()), *(
                            (cast(int, range_["byte_start"]), cast(int, range_["byte_end"]))
                            for range_ in cast(list[dict[str, object]],
                                               part["withheld_spans"])))
                ) and stamped(text) <= own and (within(span, cover) or (
                        meets(span, cover) and not any(
                            naming(rest) for rest in self._quote_texts(outside(span, cover))))):
                    # The readable rest of a target's quote the forget's cut left for it, which
                    # shows no stamped name but the memory's own ("My daughter Ruby" for "The
                    # user's daughter is named Ruby."), with words of its own outside it that
                    # name none ("我女儿" of "我女儿大名叫林可欣" beside a forgotten "林可欣 is
                    # also known as 安安." quoting "大名叫林可欣，小名叫安安").
                    return True
                # A quote of several sentences ("I no longer work in Xuhui; I work in Pudong
                # now"): each of its readable sentences naming a forgotten name was judged
                # and kept.
                pieces = [] if part is None else [
                    (sentence, text) for sentence, text in self._readable(part)
                    if sentence.byte_start < span.byte_end and span.byte_start < sentence.byte_end]
                return bool(pieces) and all(
                    (sentence.evidence_id, sentence.byte_start, sentence.byte_end) in keys
                    or not naming(text) for sentence, text in pieces)

            with writer.write_scope():
                parts = {evidence_id: row for evidence_id in sorted(
                    {span.evidence_id for span in spans})
                    if (row := writer.get_row("evidence", evidence_id)) is not None
                    and (row["scope_kind"], row["scope_id"]) == (
                        scope.scope_kind, scope.scope_id) and row["state"] == "active"}
            kept_parts = [evidence_id for evidence_id, row in parts.items() if all(
                (span.evidence_id, span.byte_start, span.byte_end) in keys or not naming(text)
                for span, text in self._readable(row))]
            current = self._memories(scope)
            rows = [row for row in current if row.memory_id in memories]
            # Each entity's names, as the scope's naming memories give them: a kept memory's
            # own names are those of the entities it is about.
            entity_names: dict[str, set[str]] = {}
            for snapshot in self._heads(scope, [row.memory_id for row in current]).values():
                for entry in snapshot.naming:
                    entity_names.setdefault(entry.entity_id, set()).add(entry.name)
            heads = self._heads(scope, [row.memory_id for row in rows])
            for row in rows:
                snapshot = heads.get(row.memory_id)
                names_ = set[str]() if snapshot is None else {
                    *(entry.name for entry in snapshot.naming),
                    *((tie[1],) if (tie := tied_name(snapshot.statement)) else ()),
                    *(name for entity in {snapshot.entity_id, *snapshot.about_entity_ids}
                      - {USER} for name in entity_names.get(entity, ()))}
                own = {writer.bind_content(SHADOW_PURPOSE, term) for name in names_
                       for term in sparse_terms(name)}
                # A prepare reads a quote as its plan leaves it: the readable parts used
                # to requote it.
                quotes = [piece for span in row.quotes for piece in outside(span, [
                    QuoteSpanV1(evidence_id=span.evidence_id, byte_start=low, byte_end=high)
                    for low, high in planned.get(span.evidence_id, ())])]
                if all(inside(span, text, own) or not naming(text) for span, text in zip(
                        quotes, self._quote_texts(quotes), strict=True)):
                    kept_memories.append(row.memory_id)
        entry = ForgetKeptEntryV1(
            start=start, end=end, memories=sorted(kept_memories)[:MAX_FORGET_KEPT_HITS],
            parts=sorted(kept_parts)[:MAX_FORGET_KEPT_HITS]) if (
            kept_memories or kept_parts) else None
        if not write or entry is None and not found:
            return frozenset(ours), frozenset(kept_memories)
        self._write_kept(scope, lambda value: value.model_copy(update={
            "plans": [other for other in value.plans if other.plan != plan],
            "entries": [*value.entries, *([] if entry is None else [entry])][
                -MAX_FORGET_KEPT:]}))
        return frozenset(ours), frozenset(kept_memories)

    def _kept_windows(self, scope: ScopeV1, *, before: int | None = None,
                      ) -> dict[tuple[str, str], list[tuple[int, int]]]:
        """Each memory and Evidence part the scope's complete forgets recorded judged and
        kept (``forget_kept:{s}``), with those forgets' clock windows; with ``before``, only
        the forgets that ended before it."""
        with self._writer.write_scope():
            stored = self._writer.get_control(f"forget_kept:{scope.scope_id}")
        windows: dict[tuple[str, str], list[tuple[int, int]]] = {}
        if stored is None:
            return windows
        for entry in ForgetKeptV1.model_validate(stored["value"]).entries:
            if before is not None and entry.end >= before:
                continue
            for key in (*(("memory", memory_id) for memory_id in entry.memories),
                        *(("evidence", part_id) for part_id in entry.parts)):
                windows.setdefault(key, []).append((entry.start, entry.end))
        return windows

    def shadow_times(self, scope: ScopeV1) -> dict[str, int]:
        """The scope's shadow names (keys) with when a complete forget last stamped each."""
        with self._writer.write_scope():
            stored = self._writer.get_control(f"forget_shadow:{scope.scope_id}")
        if stored is None:
            return {}
        shadow = ForgetShadowV1.model_validate(stored["value"])
        return dict(zip(shadow.names, shadow.name_times, strict=False))

    def _hidden(self, scope: ScopeV1, memory_ids: Set[str], ours: Set[str], kept: Set[str],
                held: Mapping[str, Sequence[tuple[int, int]]] | None = None,
                gone: Set[str] = frozenset(), prior: Mapping[str, int] | None = None,
                since: int | None = None, asked: Set[str] = frozenset()) -> set[str]:
        """The readable memories of ``memory_ids`` the names a forget stamped (``ours``,
        keys) hide from the interactive Search and the memory list
        (``recall_agent.forget_shadowed``): each showing one in its statement, keywords or quotes
        that the forget did not record kept (``kept``), unless ``NAME_KEEP_MEMORIES``
        surviving memories still hold that name (``recall_agent.surviving_names``). The
        forget reports them, so no memory leaves the user's view silently; a name lifted
        again errs toward reporting. A prepare reads the quotes as its plan leaves them,
        ``held`` withheld, and the memories it names as ``gone`` as surviving no more. Only
        what this forget newly hides counts: a name stamped before it (``prior``) outside
        every window of a forget that kept the memory (those ended before ``since``) already
        hid it, so a later turn does not report it again. A name of the request turn's own
        words (``asked``, keys) hides every memory showing it, kept or not (``recall``'s
        request shadow): the turn reports a memory an earlier forget stamped that name for
        and kept, which those words now hide ("The user is an analyst at Northwind." that the
        plugin's confirmation kept, hidden by the request turn's "Northwind"); what a request
        alone hides before any forget ran is no forget's doing."""
        names = {*ours, *asked}
        if not memory_ids or not names:
            return set()
        writer = self._writer

        def readable(quotes: Sequence[QuoteSpanV1]) -> list[QuoteSpanV1]:
            if not held:
                return list(quotes)
            pieces: list[QuoteSpanV1] = []
            for span in quotes:
                parts = [(span.byte_start, span.byte_end)]
                for low, high in held.get(span.evidence_id, ()):
                    parts = [part for start, end in parts for part in (
                        (start, min(end, low)), (max(start, high), end)) if part[0] < part[1]]
                pieces += [QuoteSpanV1(evidence_id=span.evidence_id, byte_start=start,
                                       byte_end=end) for start, end in parts]
            return pieces

        # A past memory the list shows is hidden as a current one is; only current ones are
        # counted as holding a name, which errs toward hiding.
        current = {row.memory_id for row in self._memories(scope)}
        survivors = [row for row in self._memories(scope, past=True)
                     if row.memory_id not in gone]
        terms = {row.memory_id: {writer.bind_content(SHADOW_PURPOSE, term) for text in (
            row.statement, *row.keywords, *self._quote_texts(readable(row.quotes)))
            for term in sparse_terms(text)} & names for row in survivors}
        held_by = {name: sum(1 for memory_id, found in terms.items()
                             if name in found and memory_id in current) for name in names}
        hiding = {name for name, count in held_by.items() if count < NAME_KEEP_MEMORIES}
        earlier = self._kept_windows(scope, before=since) if prior else {}

        def kept_before(memory_id: str, name: str) -> bool:
            return prior is not None and name in prior and any(
                low <= prior[name] <= high
                for low, high in earlier.get(("memory", memory_id), ()))

        def newly(memory_id: str, found: Set[str]) -> bool:
            return any(kept_before(memory_id, name) if name in asked and name not in ours
                       else memory_id not in kept and (
                           prior is None or name not in prior or kept_before(memory_id, name))
                       for name in found)

        return {memory_id for memory_id, found in terms.items()
                if memory_id in memory_ids and newly(memory_id, found & hiding)}

    def _foreseen_hidden(
        self, scope: ScopeV1, gone: Sequence[str], read: Set[str], judged: Sequence[_Judged],
        held: Sequence[QuoteSpanV1],
        stones: Sequence[tuple[str, Sequence[str], Sequence[float], str]],
        lost: Set[str] = frozenset(), targets: Sequence[str] = (),
        sentences: Sequence[QuoteSpanV1] = (),
    ) -> set[str]:
        """The readable memories a confirmation forgetting ``gone`` will report hidden by the
        names it stamps (`_hidden`), as a prepare foresees them: the shadow names of what its
        forgets tombstone (``stones``: statements and keywords, `ForgetOwner.tombstoned`), the
        kept record its prepare's reads (``read``, ``judged``), the ``targets``' quotes and the
        plan's ``sentences`` give (`_record_kept`, as the confirmation's cover), with the plan's
        withheld ranges (``held``) withheld, and the memories
        it leaves unreadable (``lost``) surviving no more. It errs toward naming one: a name
        lifted again, or a sentence its forget leaves readable that no pick judged, is not
        read here."""
        heads = self._heads(scope, list(targets))
        terms = [term for statement, keywords, _, _ in stones
                 for term in shadow_terms(statement, keywords).names]
        writer = self._writer
        ours = {writer.bind_content(SHADOW_PURPOSE, term) for term in terms}
        if not ours:
            return set()
        planned: dict[str, list[tuple[int, int]]] = {}
        for span in held:
            planned.setdefault(span.evidence_id, []).append((span.byte_start, span.byte_end))
        prior = self.shadow_times(scope)
        _, kept = self._record_kept(
            scope, 0, memories=read, judged=judged,
            cover=[*(span for snapshot in heads.values() for span in snapshot.quote_spans),
                   *sentences],
            ours=ours, held=planned, write=False)
        out = {*gone, *lost}
        return self._hidden(scope, {row.memory_id for row in self._memories(
            scope, past=True)} - out, ours, kept, planned, out, prior=prior)

    def _quote_texts(self, quotes: Sequence[QuoteSpanV1]) -> list[str]:
        """Each quote's text, empty where its part is gone."""
        texts: list[str] = []
        writer = self._writer
        with writer.write_scope():
            for span in quotes:
                row = writer.get_row("evidence", span.evidence_id)
                raw = b"" if row is None else row["text"].encode("utf-8")
                texts.append("" if row is None else raw[
                    span.byte_start - row["byte_start"]:span.byte_end - row["byte_start"]
                ].decode("utf-8", "replace"))
        return texts

    def add_forgets(
        self, scope: ScopeV1, request_id: str, statements: Sequence[str],
    ) -> tuple[list[ChatForgetRequest], int]:
        """What an Add's in-chat forget recorded (``forget_add:{s}``), read only, with each
        request's words from ``statements`` (the Add's own requests, in order); a request
        with no record yet is ``pending``. Also the Add's own memories its forget took. An
        Add a forget overtook was carried out under a re-admission's id
        (``core/readmission.py``), whose record answers for it."""
        stored = self._writer.get_control(f"forget_add:{scope.scope_id}")
        entries = [] if stored is None else ForgetAddV1.model_validate(stored["value"]).entries
        found = None
        for admitted in admitted_request_ids(request_id):
            add_key = self._writer.bind_content(HANDLED_PURPOSE, {
                "scope": scope.scope_id, "add": admitted})
            found = next((entry for entry in entries if entry.add == add_key), None)
            if found is not None:
                request_id = admitted
                break
        rows = {} if found is None else {row.request: row for row in found.requests}
        result: list[ChatForgetRequest] = []
        for position, statement in enumerate(statements):
            row = rows.get(self._writer.bind_content(HANDLED_PURPOSE, {
                "scope": scope.scope_id, "add": request_id, "request": position,
                "words": _words(probe(statement))}))
            result.append(ChatForgetRequest(
                statement=statement, status="pending", forgotten=0, withheld=0, bundled=0,
                earlier=False) if row is None else ChatForgetRequest(
                statement=statement, status=row.status, forgotten=row.forgotten,
                withheld=row.withheld, bundled=row.bundled, earlier=row.earlier))
        return result, 0 if found is None else found.taken

    def _commit_seq(self) -> int:
        """The Writer's current ``commit_seq``."""
        writer = self._writer
        with writer.write_scope():
            return writer.get_counter("commit_seq")

    def _seed(
        self, scope: ScopeV1, statement: str, spans: Sequence[QuoteSpanV1],
        phrase: _Phrase | None,
    ) -> ForgetSeedV1:
        """A forget of sentences' tombstone: the statement and its vector, or, without one,
        the vector of the first sentence's Evidence part."""
        try:
            text = canonical_statement(asked_part(statement))
        except ValueError:
            text = canonical_statement(self._quoted(scope, spans[:1])[0])
        if phrase is not None:
            return ForgetSeedV1(statement=text, embedding=list(phrase.raw),
                                embedding_space=phrase.space)
        writer = self._writer
        with writer.write_scope():
            row = writer.get_row("evidence", spans[0].evidence_id)
        if row is None:
            raise QuoteMemError("reference_missing", "A sentence to forget lies in no Evidence")
        return ForgetSeedV1(statement=text, embedding=row["embedding"],
                            embedding_space=row["embedding_space"])

    def _forget_spans(
        self, scope: ScopeV1, grant: PrincipalGrant, session_id: str,
        spans: Sequence[QuoteSpanV1], *, statement: str | None = None,
        intent: str | None = None, seed: ForgetSeedV1 | None = None,
        enforces: int | None = None, rewordings: Sequence[ForgetRewordV1] = (),
    ) -> ForgetStatementOutcomeV1:
        """Forget sentences no forgotten memory withheld: through ``intent``,
        else through one prepared now under ``seed``, else seeded with ``statement`` (else
        the first sentence) and the first sentence's Evidence vector; ``enforces`` and
        ``rewordings`` as `_forget`'s."""
        try:
            if intent is None:
                seed = seed or self._seed(scope, statement or "", spans, None)
                intent = self._owner.prepare_spans(
                    scope, grant, session_id=session_id, spans=spans, seed=seed,
                    enforces=enforces, rewordings=rewordings).intent_key
            done = self._owner.forget_spans(scope, grant, session_id=session_id, intent=intent)
        except QuoteMemError as error:
            if error.code not in ("operation_conflict", "scope_denied", "reference_missing"):
                raise
            return ForgetStatementOutcomeV1(memory_id=None, outcome="changed")
        return ForgetStatementOutcomeV1(
            memory_id=None, withheld_spans=len(spans), outcome="verified",
            controller_run_id=done.controller_run_id, receipt_ref=ArtifactId(done.receipt_ref),
            receipt=done.receipt)

    def forget_sentences(
        self, scope: ScopeV1, grant: PrincipalGrant, *, session_id: str, statement: str,
        spans: Sequence[QuoteSpanV1],
    ) -> ForgetStatementOutcomeV1:
        """Forget ``spans`` under one tombstone of ``statement`` and its vector.

        A rebuild's replay of a forget whose words its journal withholds
        (``quotemem.agent.rebuild``): no judge is asked, the statement is embedded once (else the
        first sentence's vector stands for it), and the sentences are withheld by one
        forget of sentences, so the tombstone, its forget shadow and the withheld ranges are
        made as the old store's forget made them.
        """
        phrase, _ = self._phrase(statement)
        return self._forget_spans(scope, grant, session_id, spans,
                                  seed=self._seed(scope, statement, spans, phrase))

    def forget_earlier(
        self, scope: ScopeV1, authorize: Callable[[], PrincipalGrant], *, session_id: str,
        statement: str, sources: Set[tuple[str, str]],
    ) -> int:
        """Forget ``statement`` again over the Evidence of ``sources`` alone; how many
        memories went (a rebuild's safety pass, ``quotemem.agent.rebuild``).

        ``sources`` are the (session id, request id) of the user's Adds the old store took
        at or before the forget. Only a current memory every quote of which lies in their
        Evidence is a candidate, so nothing drawn from a later Add (a restatement the
        relearn check admitted) is reached. A candidate whose words are the statement's
        (`_words`; a journal statement is the forgotten memory's own) is taken without a
        vote, and so is one the statement's forget shadow matches (`_shadowed`: a name it
        names, two of its topic keywords, or a vector within ``SHADOW_COSINE`` of its own,
        the rule by which the interactive Search withholds a hit), so a memory the rebuild
        drew again, in the statement's words or in other words, is forgotten whatever its
        judge decides. This may forget an earlier memory the old store kept and its Search
        withheld behind the same shadow, or one about another fact that names the same
        name: a rebuild errs toward forgetting what was said before a forget. Each other
        candidate restating the statement (`_restates`) or at ``DENSE_FLOOR`` or more to
        its vector is judged against it, ``MAX_FORGET_CANDIDATES`` at most, and taken when
        it states it or part of it and states no other fact (`forget_match.exclusion`).
        Each target goes, which withholds what it quotes, then every memory
        known only from what was forgotten (`_cascade`). A repeat finds nothing left to
        forget.

        Raises:
            QuoteMemError: ``provider_unavailable`` when a pick decision got fewer than two
                answered votes; the forget's own errors.
        """
        earlier = self._evidence_of(scope, sources)
        rows = [row for row in self._memories(scope)
                if row.quotes and all(span.evidence_id in earlier for span in row.quotes)]
        if not rows:
            return 0
        words = _words(statement)
        targets = {row.memory_id: row for row in rows if _words(row.statement) == words}
        rest = [row for row in rows if row.memory_id not in targets]
        phrase, _ = self._phrase(statement) if rest else (None, 0)
        for row in self._shadowed(scope, statement, phrase, rest):
            targets[row.memory_id] = row
        rest = [row for row in rest if row.memory_id not in targets]
        terms = content_terms(statement)
        shown = [row for row in rest if _restates(row.statement, [terms])]
        if phrase is not None and rest:
            known = {row.memory_id for row in shown}
            shown += [row for row in self._nearest(rest, phrase) if row.memory_id not in known]
        shown = shown[:MAX_FORGET_CANDIDATES]
        if shown and self._provider is not None:
            pick = pick_targets(statement, [row.statement for row in shown],
                                self._sender(CHAT_FORGET_REASONING_EFFORT), rounds=_rounds)
            for i in sorted({*pick.targets,
                             *(i for i, role in enumerate(pick.roles) if role in _PARTS)}):
                if exclusion(statement, shown[i].statement) is None:
                    targets.setdefault(shown[i].memory_id, shown[i])
        forgotten = 0
        for row in targets.values():
            target = ForgetStatementTargetV1(
                memory_id=row.memory_id, expected_revision_id=row.revision_id,
                expected_content_hash=IntegrityDigest(row.content_hash))
            if self._forget(scope, authorize(), session_id, target,
                            modifier=self._modifier([statement])).outcome == "verified":
                forgotten += 1
        return forgotten + (self._cascade(
            scope, authorize, session_id, [span for row in targets.values()
                                           for span in row.quotes],
            requests=[statement], modifier=self._modifier([statement])) if forgotten else 0)

    def _shadowed(
        self, scope: ScopeV1, statement: str, phrase: _Phrase | None,
        rows: Sequence[_Candidate],
    ) -> list[_Candidate]:
        """The ``rows`` the forget shadow of ``statement`` alone matches
        (`recall_agent.forget_shadowed`), each read as the interactive Search reads a
        memory: the `sparse_terms` of its statement, keywords and quotes, and its vector
        when it lies in the statement's space. A name the store's other memories still
        hold is not let off here, unlike in Search (`recall_agent.surviving_names`)."""
        shadow = ForgetShadow.of(shadow_terms(statement),
                                 vectors=() if phrase is None else (phrase.vector,))
        if not shadow:
            return []
        found: list[_Candidate] = []
        for row in rows:
            texts = (" ".join((row.statement, *row.keywords)), *self._quoted(scope, row.quotes))
            held = frozenset(term for text in texts for term in sparse_terms(text))
            vector = (unit_vector(row.vector) if phrase is not None and any(row.vector)
                      and row.space == phrase.space else None)
            if forget_shadowed(held, vector, True, shadow):
                found.append(row)
        return found

    def _evidence_of(self, scope: ScopeV1, sources: Set[tuple[str, str]]) -> set[str]:
        """The ids of the scope's Evidence parts of ``sources`` (session id, request id)."""
        if not sources:
            return set()
        writer = self._writer
        with writer.write_scope():
            txn = TxnRows(writer, scope.scope_kind, scope.scope_id, 0, self._clock(), "")
            return {cast(str, row["object_id"]) for row in txn.pages(EvidenceRows(
                scope_kind=scope.scope_kind, scope_id=scope.scope_id))
                if (row["session_id"], row["request_id"]) in sources}

    # In-chat requests ---------------------------------------------------------------
    def chat_forget(
        self, scope: ScopeV1, authorize: Callable[[], PrincipalGrant], *, user_id: str,
        session_id: str, statements: Sequence[str], request_id: str | None = None,
    ) -> ChatForgetOutcome:
        """Forget what one Add's in-chat requests ask to forget, and mark what still states it.

        Every statement is Searched (``CHAT_SEARCH_TOP_K``). Its memory candidates are the
        current memory hits; its sentence candidates are the readable sentences of the
        Evidence hits that share a content word with it, the ``MAX_FORGET_CANDIDATES``
        sharing the most. The record of a request is neither: a memory every quote of which
        is a forget request ("The user asked to forget that ...") and a sentence that is one
        stay as they are, so what the user asked stays known. The judge
        (``memory-forget-pick-v1``, at ``CHAT_FORGET_REASONING_EFFORT``) classifies both
        lists of every statement, all picks concurrent, and every candidate it decides
        states the statement or part of it (``same_fact``, ``part``) is a target. Each
        memory target, a paraphrase as much as the matched statement, goes through the forget one
        by one, under authority ``authorize``
        resolves fresh for it, as the picks can outlast a grant; one whose head moved or
        that is gone is skipped. Each sentence target stays readable and its span is marked
        (``forget_marks:{s}``), so Search shows that sentence as something the user later
        asked not to use.

        No Search corpus holds a memory of a summary-only origin (an overview or the
        assistant's work, ``recall.SUMMARY_ONLY_ORIGINS``), so the scope's current ones are
        read directly; those whose statement shares a content word with the request, the
        ``MAX_FORGET_CANDIDATES`` sharing the most, are a third list. Each the judge decides
        states the fact or part of it is withdrawn (``forget_withdrawn:{s}``): it recounts a
        whole session, so a forget would withhold every span it quotes, and instead no read
        delivers it again. The request's words find only the summaries worded like it, and
        its memory targets come from Search: a memory the judge takes as stating part of the
        request ("a playlist full of Spanish guitar" for "I enjoy a wide variety of music
        genres") goes, while an overview telling the same detail shares no word with the
        request or does not state the request's own claim. So every summary-only memory
        sharing a content word with a memory target's statement is judged against that
        statement too, and one that states it or part of it is withdrawn
        (`StatementForget._restating`). Every pick precedes every forget, so a judge failure
        leaves nothing done and the retried run judges them all again.

        A complete forget (``explicit-request-v2``) differs in these ways:

        - the memories recording these requests (`_request_records`) are targets, not kept:
          the request restates the fact, so a memory the same Add made of it would outlive
          the forget it asked for;
        - sentences are not taken from the Search hits: after the memory picks, every
          readable sentence of the user's Evidence sharing terms with a request, a target's
          statement or what a target quotes (CJK bigrams included, `content_terms`) is
          judged,
          assistant replies and earlier sessions alike, and each sentence that is one of
          these requests is taken without a vote. The first forgotten memory withholds them
          all (``ForgetOwner.prepare``'s ``restating``), and the requests' statements join
          the scope's forget shadow (``ForgetOwner.record_shadow``), so the interactive
          Search still withholds what they named; only when no memory is forgotten are the
          sentences marked instead;
        - every other current memory sharing terms with a target's statement is judged
          against it, and one stating it or part of it is a target, so a memory whose own
          wording still states the fact goes even when its quotes stay readable;
        - after the forgets, every memory left without readable support goes too
          (`_cascade`);
        - given the Add's ``request_id`` (every Add of such a workspace, with requests or
          not, ``service/chat_forget.py``), the memories that Add wrote and its sentences
          are checked against the scope's tombstones first (`_relearned`): a restatement of
          a forgotten fact, such as an assistant reply recounting it in a later session, is
          forgotten too, and its sentences are withheld (under that tombstone when no
          memory is forgotten);
        - a memory or sentence pick whose clause carrying the claim names another person or
          states the opposite (`forget_match.exclusion`) is no target;
        - the candidates and picks are the prepare's: a request this session's
          forget by statement already handled is read only for what changed since it read
          the scope, and what the user told again after its confirmation is left;
          both paths' record names what this one did.

        Raises:
            QuoteMemError: ``service_unhealthy`` without a Search or provider;
                ``provider_unavailable`` when a pick decision got fewer than two answered
                votes; the forget's own errors.
        """
        if self._search is None or self._provider is None:
            raise QuoteMemError("service_unhealthy", "Forget by statement has no Search or judge")
        complete = self._complete()
        # The plugins' whole words of each request key it and report it; the judge
        # and the matching read only what it asks to forget (`forget_requests.asked_part`), as
        # the prepare reads it, never the new value told after it.
        said = list(statements)
        statements = [asked_part(statement) for statement in said] if complete else said
        started = self._clock()
        # The shadow's names as earlier forgets left them (`_hidden`).
        prior = self.shadow_times(scope) if complete else {}
        sender = self._pick_sender() if complete else self._sender(CHAT_FORGET_REASONING_EFFORT)
        relearned: list[_Candidate] = []
        relearned_spans: list[QuoteSpanV1] = []
        stone: _Stone | None = None
        enforced: dict[str, int] = {}
        calls = 0
        if complete and request_id is not None:
            relearned, relearned_spans, stone, calls, enforced = self._relearned(
                scope, request_id, sender)
            # A clear replacement told with a request is the user's new value, never a
            # restatement of what the request forgets: its words stay (`_told_new`).
            relearned_spans = [span for span in relearned_spans if not self._told_new(span)]
            if not statements and not relearned and not relearned_spans:
                # An Add restating nothing forgotten and asking nothing reads no more.
                return ChatForgetOutcome(forgotten=0, changed=0, marked=0, withdrawn=0,
                                         calls=calls)
        picks: list[tuple[str, list[_Candidate], list[QuoteSpanV1], list[str],
                          list[_Candidate]]] = []
        everything = self._summaries(scope)
        summaries = [row for row in everything
                     if not self._records_request(scope, row.quotes, complete=complete)]
        # A complete forget embeds the requests once, so a fact told in another language than
        # the request is a candidate too (the judge still decides). Its candidates and picks
        # are the prepare's (`_matched`, `_pick_sender`): one request decides alike on both
        # paths. A request its forget by statement already handled in this
        # session reads only what changed after that prepare read the scope.
        phrases = self._phrases([probe(statement) for statement in statements])[0] if (
            complete) else []
        current = ([row for row in self._memories(scope)
                    if row.memory_id not in {summary.memory_id for summary in everything}]
                   if phrases else [])
        handled = [self._handled_key(scope, session_id, statement) for statement in said
                   ] if complete else []
        covered: dict[int, int] = {}
        confirmed: dict[int, Forgotten] = {}
        ran: dict[int, int | None] = {}
        for position, key in enumerate(handled):
            entry = self._entry(scope, key=key)
            if entry is not None and entry.statement in ("forgotten", "nothing_matched"):
                ran[position] = entry.statement_seq
                if entry.statement_seq is not None:
                    covered[position] = max(covered.get(position, 0), entry.statement_seq)
            # What either path already forgot of this request is not read again: a plan still
            # unconfirmed settled nothing (its prepare may never be confirmed), and neither did
            # a forget that matched nothing (the user asking again is read whole).
            seen = [] if entry is None else [seq for seq, done in (
                (entry.statement_seq, entry.statement == "forgotten"),
                (entry.chat_seq, entry.in_chat == "forgotten")) if seq is not None and done]
            if entry is not None and seen:
                covered[position] = max([*seen, *([covered[position]] if position in covered
                                                  else [])])
            if entry is not None and entry.statement == "forgotten":
                # Its confirmation forgot it then: what the user told after it, restating it
                # in their own words, is new.
                phrase = phrases[position] if phrases else None
                confirmed[position] = Forgotten(
                    statements[position], entry.at, () if phrase is None else phrase.raw,
                    "" if phrase is None else phrase.space)
        # A request its forget by statement already ran for (confirmed, or matched nothing)
        # is that forget's: its own turn takes no memory that forget read, left or
        # changed, nor any other by a rule of its own (a name, a correction, what a
        # confirmation left unreadable); it reads only the memories and sentences of the
        # Evidence ingested after that forget read the scope (`_fresh`).
        active = [statement for position, statement in enumerate(statements)
                  if position not in ran]
        lead = next((position for position in range(len(statements)) if position not in ran),
                    None)
        fresh, told_since = (self._fresh(scope, ran.values(), request_id) if complete and ran
                             else (set[str](), set[str]()))
        for position, statement in enumerate(statements):
            if complete:
                hit_ids = self._hits(scope, user_id, probe(statement), CHAT_SEARCH_TOP_K)
                candidates = [row for row in self._matched(
                    scope, hit_ids, phrases[position] if phrases else None, current,
                    after=covered.get(position))[0]
                    if (position not in ran or row.memory_id in fresh) and (
                        position not in confirmed
                        or not relearns(self._tolds(row), confirmed[position], row.statement))]
                picks.append((statement, candidates, [], [], [] if position in ran
                              else _sharing(statement, summaries)))
                continue
            hit_ids = self._hits(scope, user_id, statement, CHAT_SEARCH_TOP_K)
            candidates = [row for row in self._candidates(scope, hit_ids)[0]
                          if not self._records_request(scope, row.quotes)]
            spans, texts = self._sentences(scope, statement, hit_ids)
            picks.append((statement, candidates, spans, texts, _sharing(statement, summaries)))
        jobs = [(index, kind, _in_context(statement, said[index]) if complete and kind == "memory"
                 else statement, texts)
                for index, (statement, candidates, _, sentences, shared) in enumerate(picks)
                for kind, texts in (("memory", [row.statement for row in candidates]),
                                    ("sentence", sentences),
                                    ("summary", [row.statement for row in shared])) if texts]

        def decide(job: tuple[int, str, str, list[str]]) -> tuple[int, str, ForgetPickV1]:
            return job[0], job[1], pick_targets(job[2], job[3], sender, rounds=_rounds,
                                                minimal=not complete)

        with ThreadPoolExecutor(max_workers=max(1, min(len(jobs), MAX_JUDGE_CONCURRENCY)),
                                thread_name_prefix="quotemem-chat-forget") as pool:
            futures = [pool.submit(contextvars.copy_context().run, decide, job) for job in jobs]
            decided = [future.result() for future in futures]
        targets: dict[str, _Candidate] = {row.memory_id: row for row in relearned}
        # The targets these requests asked for: a forget of them is new, while the relearn
        # check's own targets only enforce a forget already made.
        requested: set[str] = set()
        marked: dict[tuple[str, int, int], QuoteSpanV1] = {}
        withdrawn: set[str] = set()
        asked: dict[int, dict[int, _Candidate]] = {}
        for index, kind, pick in decided:
            calls += pick.calls
            chosen = {*pick.targets, *(i for i, role in enumerate(pick.roles) if role in _PARTS)}
            if complete and kind == "memory":
                chosen -= _spared(picks[index][1], pick.roles)
            for i in sorted(chosen):
                if kind == "memory":
                    row = picks[index][1][i]
                    if not complete or exclusion(picks[index][0], row.statement) is None:
                        targets.setdefault(row.memory_id, row)
                        requested.add(row.memory_id)
                        asked.setdefault(index, {})[i] = row
                elif kind == "summary":
                    withdrawn.add(picks[index][4][i].memory_id)
                else:
                    span = picks[index][2][i]
                    marked[span.evidence_id, span.byte_start, span.byte_end] = span
        given_names = frozenset[tuple[str, str]]()
        uncertain = False
        within: Within = {}
        old_rows: list[_Candidate] = []
        ruled_ids: set[str] = {row.memory_id for row in relearned}
        if complete and active:
            # A request to forget a person's or an animal's name takes every memory giving it,
            # as the prepare does.
            # Only for a request no earlier forget settled: what that forget took was the
            # referent; the rule finds no new one among what it left ("forget my friend's
            # name" after the plugin took Alice of two friends leaves Bruno).
            named_rows, given_names, uncertain, within = self._referent_namers(
                scope, [statement for position, statement in enumerate(statements)
                        if position not in ran and position not in covered])
            # An old value, a person as a whole: by rule too, as the prepare takes them.
            old_rows, old_calls = self._old_values(
                scope, [(statement, picks[position][1], said[position])
                        for position, statement in enumerate(statements)
                        if position not in ran], sender)
            calls += old_calls
            for row in (*named_rows, *old_rows, *self._whole_people(scope, active)):
                targets.setdefault(row.memory_id, row)
                requested.add(row.memory_id)
                ruled_ids.add(row.memory_id)
            # Nor does it take a pick the request does not ask for (`_unnamed_picks`).
            unasked = self._unnamed_picks(
                scope, targets, {row.memory_id for rows_ in asked.values()
                                 for row in rows_.values()} - ruled_ids,
                given_names, active)
            for memory_id in unasked:
                del targets[memory_id]
                requested.discard(memory_id)
            for rows_ in asked.values():
                for i in [i for i, row in rows_.items() if row.memory_id in unasked]:
                    del rows_[i]
        replacing = complete and self._owner.forgets_completely()
        # A memory naming the asked value as one it replaced or ended is nominated whatever the
        # pick decided, and taken once the forget judge confirms it (as the prepare does,
        # ``_prepare_complete``).
        nominated: dict[str, tuple[int, int, _Candidate]] = {}
        if replacing:
            subjects = self._naming_subjects(scope, targets)
            about = self._subjects(scope, [row.memory_id for _, candidates, *_ in picks
                                           for row in candidates])
            for index, (statement, candidates, *_) in enumerate(picks):
                for i, row in enumerate(candidates):
                    if row.memory_id not in targets and about.get(row.memory_id) in subjects \
                            and exclusion(statement, row.statement) is None and names_replaced(
                                row.statement, statement, unsure=False):
                        nominated.setdefault(row.memory_id, (index, i, row))
        restating_spans: list[QuoteSpanV1] = []
        # Which requests took each sentence, and what the picks read and left.
        takers: dict[tuple[str, int, int], set[int]] = {}
        read: set[str] = set()
        left: list[_Judged] = []
        record_ids: set[str] = set()
        more_ids: set[str] = set()
        if complete and statements:
            # Requests read alike are picked together; a confirmed one alone, against its own
            # words.
            keys = {(covered.get(position), position if position in confirmed else -1,
                     position in ran) for position in range(len(statements))}
            for after, alone, done in sorted(keys, key=lambda key: (
                    -1 if key[0] is None else key[0], key[1], key[2])):
                group = [position for position in range(len(statements))
                         if (covered.get(position), position if position in confirmed else -1,
                             position in ran) == (after, alone, done)]
                since = confirmed.get(alone)
                asked_now = [statements[position] for position in group]
                changed = None if after is None else {
                    row.memory_id for row in self._memories(scope, after=after)}
                for row in self._request_records(scope, asked_now):
                    if (changed is None or row.memory_id in changed) and (
                            not done or row.memory_id in fresh):
                        targets.setdefault(row.memory_id, row)
                        requested.add(row.memory_id)
                        record_ids.add(row.memory_id)
                more, found_spans, more_calls, judged_now, read_now = self._complete_picks(
                    scope, asked_now, targets, {row.memory_id for row in summaries}, sender,
                    phrases=[phrases[position] for position in group] if phrases else [],
                    veto=exclusion, after=after, told_after=since, memory_picks=not done)
                calls += more_calls
                if any(lived_value(statement) is not None for statement in asked_now):
                    # Where the user lived withholds no sentence a current home quotes.
                    held = self._home_quotes(scope)
                    homes = {_key(entry.span) for entry in judged_now
                             if entry.target and _overlaps(entry.span, held)}
                    found_spans = [span for span in found_spans if _key(span) not in homes]
                    judged_now = [replace(entry, target=False) if _key(entry.span) in homes
                                  else entry for entry in judged_now]
                restating_spans += found_spans
                read |= read_now
                for entry in judged_now:
                    parts = ([] if not replacing or entry.target or entry.excluded is not None
                             else _naming_clauses(entry.span, entry.text, asked_now))
                    if parts:
                        # The clauses of a sentence naming the asked value as replaced still
                        # tell it (as the prepare reads them, ``_naming_clauses``).
                        restating_spans += parts
                        entry = replace(entry, target=True)
                    if entry.target:
                        takers.setdefault((entry.span.evidence_id, entry.span.byte_start,
                                           entry.span.byte_end), set()).update(
                            group[index] for index in entry.asked)
                    else:
                        left.append(entry)
                for row in more:
                    targets.setdefault(row.memory_id, row)
                    requested.add(row.memory_id)
                    more_ids.add(row.memory_id)
        # A past memory naming the asked value is no candidate; it is nominated all the same.
        named_rows = (self._naming(scope, active, targets,
                                   self._naming_subjects(scope, targets))
                      if replacing and active else [])
        naming = self._naming_sentences(scope, active) if replacing and active else []
        allowed = set[str]()
        if replacing and active:
            trial = {**targets, **{row.memory_id: row for *_, row in nominated.values()},
                     **{row.memory_id: row for row in named_rows}}
            planned_spans = list({_key(span): span for span in (
                *relearned_spans, *restating_spans, *naming)}.values())[:MAX_RESTATING_SPANS]
            bundles = {*nominated, *(row.memory_id for row in named_rows)}
            ruled = self._ruled(scope, targets, planned_spans, set(), active)
            # Without the judge's answer, or with a contested one, the in-chat forget refuses
            # as its pick does (``provider_unavailable``): it never reports a forget that
            # left a nomination unread.
            assert lead is not None
            verdict = self._confirmed(scope, active, [
                nominee for nominee in (*bundles, *self._nominations(
                    scope, trial, planned_spans,
                    None if trial or not (phrases and planned_spans) else self._seed(
                        scope, statements[lead], planned_spans, phrases[lead]),
                    bundled=bundles, words=active)) if nominee not in ruled], sender)
            if verdict is None:
                raise QuoteMemError("provider_unavailable",
                                    "The forget judge could not settle what the forget takes")
            allowed, judge_calls = verdict
            calls += judge_calls
        for memory_id, (index, i, row) in nominated.items():
            if memory_id in allowed:
                targets.setdefault(memory_id, row)
                requested.add(memory_id)
                asked.setdefault(index, {})[i] = row
        named_rows = [row for row in named_rows if row.memory_id in allowed]
        targets.update((row.memory_id, row) for row in named_rows)
        restating, restating_calls = self._restating(
            [row.statement for row in targets.values()],
            [row for row in summaries if row.memory_id not in withdrawn], sender)
        withdrawn |= restating
        calls += restating_calls
        # A memory target bundling the request's fact with an unrelated one is tombstoned by
        # the request's words.
        seeds: dict[str, ForgetSeedV1] = {}
        if complete and phrases:
            seeds, bundle_calls = self._chat_seeds(statements, phrases, asked, sender)
            calls += bundle_calls
        if covered:
            # A memory the forget of a handled request requoted or reworded since that forget
            # read the scope is what it decided to keep: the same
            # request read again does not take it.
            for memory_id in self._revised_since(scope, list(targets), min(covered.values())):
                if memory_id not in {row.memory_id for row in relearned}:
                    targets.pop(memory_id)
                    requested.discard(memory_id)
        forgotten = changed = withheld = 0
        # An Add asking nothing only enforces the forgets its restatements broke: the forget
        # shadow keeps its stamps, so the user's own restatement after them is still found.
        times = [*enforced.values(), *([] if stone is None else [stone.created_at])]
        enforced_at: int | None = None if active or not times else min(times)
        carry = list({(span.evidence_id, span.byte_start, span.byte_end): span for span in (
            *relearned_spans, *restating_spans)}.values())[:MAX_RESTATING_SPANS]
        requested.update(row.memory_id for row in named_rows)
        if named_rows and phrases and lead is not None:
            # Tombstoned by the request's words, as a bundle is.
            for row in named_rows:
                seeds.setdefault(row.memory_id, self._seed(
                    scope, statements[lead], [], phrases[lead]))
        if replacing and active:
            carry = list({_key(span): span for span in (*carry, *naming)}.values())[
                :MAX_RESTATING_SPANS]
            # An old value's own clause of the sentence telling the change goes with it.
            carry = list({_key(span): span for span in (*carry, *self._given_up_clauses(
                scope, active, targets, carry))}.values())[:MAX_RESTATING_SPANS]
        if complete and statements and (targets or carry):
            # The assistant's echoes of what is forgotten state it too; of
            # requests the forget by statement ran for, only those told since it read.
            carry = list({_key(span): span for span in (*carry, *(
                span for span in self._echoes(
                    scope, [*(span for row in targets.values() for span in row.quotes), *carry],
                    self._echoed(scope, statements, targets,
                                 {*seeds, *(row.memory_id for row in named_rows)}))
                if active or span.evidence_id in told_since))}.values())[:MAX_RESTATING_SPANS]
        if complete and carry and (active or not statements):
            # A memory this forget leaves unreadable is taken whole when a later memory that
            # replaced it still names its value (``_promoted``).
            requested.update(self._promoted(scope, targets, carry, None if targets or not (
                lead is not None and phrases) else self._seed(
                    scope, statements[lead], carry, phrases[lead]), about=active,
                                            bundled=set(seeds), allowed=allowed))
        # What the judge read and did not take is reported apart from the targets; what
        # this Add itself made and the forget takes is no memory it added. A memory
        # recording a request (its own sentence) forgets nothing the user told.
        kept = {row.memory_id for pick in picks for row in pick[1]} - set(targets)
        before = {row.memory_id for row in self._memories(scope, past=True)} if complete \
            else set[str]()
        own = (self._created_by(scope, request_id) & before
               if complete and request_id is not None else set[str]())
        records = ({row.memory_id for row in self._request_records(scope, statements)}
                   if complete and statements else set[str]())
        done_ids: set[str] = set()
        told: set[tuple[str, int, int]] = set()
        # The names a request's targets give, but a bundle's (tombstoned by the request's
        # words): no memory may show them after the forget (`_leftovers`' fourth kind).
        # A judge's pick naming another entity than the request's stays, and so
        # does a later pick the request does not ask for (`_unnamed_picks`).
        for memory_id in {*self._other_namers(scope, targets, given_names),
                          *self._unnamed_picks(scope, targets, more_ids - record_ids - ruled_ids,
                                               given_names, active)}:
            del targets[memory_id]
            requested.discard(memory_id)
        # A request asking a name: a target tying a name to someone by a role gives it too.
        asks = any(asks_name(statement) for statement in active)
        nick = any(nickname_request(statement) for statement in active)
        if complete and asks and not nick and not given_names:
            # The name of a place or organisation of someone's, as the prepare reads it.
            given_names = self._kind_names(scope, active, list(targets))
        namers = [memory_id for memory_id in targets if memory_id not in seeds]
        every_name = given_names | self._names(scope, namers, ties=asks, aliases=nick,
                                               every=asks and not nick)
        names = self._forgotten_names(
            scope, given_names, self._names(scope, namers, ties=asks, aliases=nick,
                                            every=asks and not nick),
            uncertain=uncertain, within=within) if complete and active \
            else frozenset[tuple[str, str]]()
        name_ties = self._tie_words(scope, namers) if names else {}
        if names and targets:
            # Every other sentence showing a name the forget takes loses that name alone
            # (`_name_token_spans`), withheld with the first forgotten memory.
            kept_names = [memory_id for memory_id in targets if memory_id not in seeds]
            # Past the bound of one forget, the name's spans are withheld over several, never
            # cut (the Add has no refusal the plugin would not retry).
            carry = list({_key(span): span for span in (*carry, *self._name_token_spans(
                scope, kept_names, self._forgotten_names(
                    scope, given_names, self._names(scope, kept_names, ties=asks,
                                                    aliases=nick, every=asks and not nick),
                    uncertain=uncertain, within=within),
                self._sessions(scope, targets.values(), session_id), within))}.values())
        if complete and active:
            # Where the user lived takes no current home, and no request takes the new value
            # it told (`_kept_homes`, `_states_new`).
            spared_homes = self._kept_homes(
                scope, targets, active, [said[position] for position in range(len(statements))
                                         if position not in ran]) | self._current_values(
                scope, active, targets, {row.memory_id for row in old_rows})
            targets = {memory_id: row for memory_id, row in targets.items()
                       if memory_id not in spared_homes}
        if complete and request_id is not None and statements:
            # A clear replacement said with a request is no target of it, nor of the relearn
            # check of its Add (`_replacements`).
            spared = self._replacements(scope, request_id)
            targets = {memory_id: row for memory_id, row in targets.items()
                       if memory_id not in spared}
            # No path of the request's turn takes a memory stating the new value its sentence
            # told ("The user's phone number is 555-9876." of "Forget my old number, my new
            # number is 555-9876.", stored before it), whatever the judge picked (`_states_new`).
            sentences = self._request_said(scope, request_id, statements)
            targets = {memory_id: row for memory_id, row in targets.items()
                       if not any(_states_new(row.statement, words) for words in sentences)}
        # The sentences withheld as stating the fact: a bare name quoted in one is the
        # forgotten fact's residue as one a forgotten memory quotes is (`_leftovers`).
        sentences_held: list[QuoteSpanV1] = []
        for row in targets.values():
            target = ForgetStatementTargetV1(
                memory_id=row.memory_id, expected_revision_id=row.revision_id,
                expected_content_hash=IntegrityDigest(row.content_hash))
            batch = carry[:MAX_RESTATING_SPANS]
            if self._forget(scope, authorize(), session_id, target, restating=batch,
                            seed=seeds.get(row.memory_id), allowed=allowed,
                            modifier=self._modifier(statements),
                            enforces=None if row.memory_id in requested
                            else enforced.get(row.memory_id, enforced_at)
                            ).outcome == "verified":
                forgotten += 1
                done_ids.add(row.memory_id)
                withheld += len(batch)
                told |= self._facts(batch)
                sentences_held += batch
                carry = carry[len(batch):]
            else:
                changed += 1
        unattached: list[QuoteSpanV1] = []
        while complete and carry:
            # With no memory left to forget, a complete forget withholds the sentences alone,
            # the request sentences with them: under the request's words, or
            # under the tombstone a sentence of this Add restated; a bound's worth at a time.
            batch, carry = carry[:MAX_RESTATING_SPANS], carry[MAX_RESTATING_SPANS:]
            done = self._forget_spans(
                scope, authorize(), session_id, batch,
                statement=statements[0] if statements else None,
                seed=None if statements or stone is None else stone.seed(),
                enforces=enforced_at)
            if done.outcome == "verified":
                withheld += len(batch)
                told |= self._facts(batch)
                unattached += batch
                sentences_held += batch
        asked_only = bool(unattached) and all(
            self._quoted_request(span) for span in unattached)
        # With no memory forgotten, no forget withholds the sentences: the judged ones are
        # marked, the request sentences are left as the benchmark rule leaves them.
        for span in carry:
            if not self._quoted_request(span):
                marked[span.evidence_id, span.byte_start, span.byte_end] = span
        if complete and active and asks and not uncertain and not nick:
            # A memory this forget left unreadable that ties a name by the asked description
            # gives it too, as a target would: the request was answered by its sentence alone
            # ("Priya is the user's manager." under "forget my manager's name").
            lost = sorted(before - {row.memory_id for row in self._memories(scope, past=True)}
                          - set(targets))
            extra = self._asked_names(scope, lost, active)
            if extra - names:
                names = names | self._forgotten_names(scope, given_names, extra,
                                                      uncertain=False, within=within)
                every_name |= extra
                name_ties = {**self._tie_words(scope, lost), **name_ties}
        # A request the forget by statement already ran for was cascaded by its confirmation,
        # within what its prepare named: an in-chat forget of it that forgot no memory of its
        # own takes no other, and one that forgot this Add's own takes only this Add's.
        inherited = bool(statements) and not active and not done_ids
        cascaded = self._cascade(
            scope, authorize, session_id,
            [span for row in targets.values() for span in row.quotes],
            enforces=enforced_at, names=names, ties=name_ties, sentences=sentences_held,
            allowed=fresh if statements and not active else None,
            told=[row.statement for row in targets.values()], within=within,
            requests=statements, modifier=self._modifier(statements),
        ) if complete and not inherited else 0
        if complete and active and withheld and not (asked_only and not cascaded):
            # The request sentences went with the fact: Search reads what they asked to
            # forget from the shadow instead (``ForgetOwner.record_shadow``). Requests that
            # matched nothing, whose own sentences alone went, hide nothing more: no memory
            # the forget kept is lost without a word ("Hangzhou" of a move from Shanghai).
            # A request the forget by statement ran for adds no words to the shadow: that
            # forget's confirmation hid what it forgot, and the request's own words would hide
            # what it kept ("Alice" of "my friend's name" after Bruno's went).
            self._owner.record_shadow(scope, [asked_part(statement) for statement in active])
        added = self._mark(scope, list(marked.values()))
        withdrawn_count = self._withdraw(scope, withdrawn)
        if not complete:
            return ChatForgetOutcome(forgotten=forgotten + cascaded, changed=changed,
                                     marked=added, withdrawn=withdrawn_count, calls=calls,
                                     withheld=withheld, cascaded=cascaded)
        # What it read and left stays found under the names it stamped; a memory they
        # hide all the same is reported (`_hidden`).
        bystanders = {row.memory_id for rows_ in self._name_readers(
            scope, every_name if complete and statements else names, ties=name_ties,
            within=within) for row in rows_}
        stamped, recorded = self._record_kept(
            scope, started, memories=(kept | read | bystanders) - set(targets), judged=left,
            cover=[span for row in targets.values() for span in row.quotes])
        # The request turn's own names hide what shows them too (`_hidden`'s ``asked``).
        asked_names = {self._writer.bind_content(SHADOW_PURPOSE, name)
                       for statement in statements
                       for name in shadow_terms(statement).names} if complete else set[str]()
        removed, affected = self._lost(scope, before, kept, self._hidden(
            scope, before, stamped, recorded, prior=prior, since=started, asked=asked_names))
        # The Add's answer has no separate list: a kept memory this forget changed is
        # counted with what it removed, so nothing it did is silent.
        removed |= {row.memory_id for row in affected}
        # The words of a request that matched something hide what still shows a name they give
        # (`recall._agent_scope`), but for what this Add stored: nothing the user had stored
        # leaves their list unreported.
        if confirmed or removed or told:
            removed |= self._hidden_by_words(scope, statements) & before - own
        read_seq = self._commit_seq()
        requests = self._outcomes(
            scope, said, handled, confirmed, read_seq, removed=removed - records,
            asked=asked, done_ids=done_ids, seeds=seeds, told=told, takers=takers)
        taken = len(own - {row.memory_id for row in self._memories(scope)})
        merged, taken = self._record_add(scope, request_id, requests, taken)
        return ChatForgetOutcome(forgotten=len(removed), changed=changed, marked=added,
                                 withdrawn=withdrawn_count, calls=calls, withheld=withheld,
                                 cascaded=cascaded, affected=len(affected),
                                 requests=tuple(merged), taken=taken)

    def _revised_since(self, scope: ScopeV1, memory_ids: Sequence[str], after: int) -> set[str]:
        """The memories of ``memory_ids`` a forget revised after commit ``after``: their head is
        a privacy requote written since (a trim on readable parts, or a rewording)."""
        writer = self._writer
        found: set[str] = set()
        with writer.write_scope():
            for memory_id in memory_ids:
                item = writer.get_row("item", memory_id)
                head = None if item is None or (item["scope_kind"], item["scope_id"]) != (
                    scope.scope_kind, scope.scope_id) else writer.get_row(
                        "revision", item["current_revision_id"])
                if head is not None and head["op"] == "requote" and head["commit_seq"] > after:
                    found.add(memory_id)
        return found

    def _facts(self, spans: Sequence[QuoteSpanV1]) -> set[tuple[str, int, int]]:
        """The withheld sentences that state something, not a forget request."""
        return {(span.evidence_id, span.byte_start, span.byte_end) for span in spans
                if not self._quoted_request(span)}

    def _outcomes(
        self, scope: ScopeV1, statements: Sequence[str], handled: Sequence[str],
        confirmed: Mapping[int, Forgotten], read_seq: int, *, removed: Set[str],
        asked: Mapping[int, Mapping[int, _Candidate]], done_ids: Set[str],
        seeds: Mapping[str, ForgetSeedV1], told: Set[tuple[str, int, int]],
        takers: Mapping[tuple[str, int, int], Set[int]],
    ) -> list[ChatForgetRequest]:
        """What each in-chat request came to, recorded with both paths' record.

        A request's own counts are the memories its pick took and the forget removed, and
        the sentences withheld that its sentence pick took; its own words (the request
        sentence, a memory recording the request: ``removed`` leaves those out) are never
        counted. What no request took alone (the cascade, a memory whose only support
        stated the fact, a restated earlier fact, a sentence naming the asked value as
        replaced) counts with the first request that forgot something itself, else with the
        first request when it removed a memory or withheld words the user said that are no
        request (`_said_fact`): a forget that changed what the user told never reports
        ``nothing_matched`` ("from Shanghai" withheld and the move memory gone), and one that
        withheld only the request sentences or the assistant's acknowledgement of them ("Okay,
        I won't remember that your sister is pregnant.") still does. A request that forgot
        nothing itself and changed nothing is ``nothing_matched``,
        ``pending`` while the plugin's plan for it waits for confirmation, and
        ``forgotten`` with ``earlier`` when the plugin's confirmation forgot it first.
        """
        picked = {position: {row.memory_id for row in rows.values()}
                  for position, rows in asked.items()}
        mine = [len(picked.get(position, set()) & removed) for position in range(len(handled))]
        facts = [sum(1 for key in told if position in takers.get(key, ()))
                 for position in range(len(handled))]
        extra_memories = len(removed - {memory_id for ids in picked.values() for memory_id in ids})
        extra = [QuoteSpanV1(evidence_id=key[0], byte_start=key[1], byte_end=key[2])
                 for key in told if not takers.get(key)]
        extra_facts = len(extra)
        first = next((position for position in range(len(handled))
                      if mine[position] or facts[position]),
                     0 if handled and (extra_memories or any(
                         self._said_fact(span) for span in extra)) else None)
        if first is not None:
            mine[first] += extra_memories
            facts[first] += extra_facts
        requests: list[ChatForgetRequest] = []
        for position, key in enumerate(handled):
            prior = self._entry(scope, key=key)
            self._record_handled(
                scope, key, chat_seq=read_seq,
                in_chat="forgotten" if mine[position] or facts[position] else "nothing_matched",
                forgotten=mine[position], withheld=facts[position])
            entry = self._entry(scope, key=key)
            assert entry is not None
            # A request the forget by statement forgot first is reported as that forget's
            # (``earlier``) unless this Add's own forget removed memories or withheld words
            # too: then it says what it forgot anew itself, so no removal of the request's
            # turn goes unreported (a memory the confirmation left unreadable that this
            # forget's cascade takes).
            earlier = not (mine[position] or facts[position]) and (
                position in confirmed or (prior is not None and prior.statement == "forgotten"))
            status: ChatForgetStatus = (
                "forgotten" if earlier or entry.in_chat == "forgotten"
                else "pending" if entry.statement == "prepared" else "nothing_matched")
            requests.append(ChatForgetRequest(
                statement=statements[position], status=status,
                forgotten=entry.forgotten if earlier else mine[position],
                withheld=entry.withheld if earlier else facts[position],
                bundled=sum(1 for row in asked.get(position, {}).values()
                            if row.memory_id in seeds and row.memory_id in done_ids),
                earlier=earlier))
        return requests

    # Complete forget (explicit-request-v2) -----------------------------------------------
    def _complete(self) -> bool:
        """Whether this workspace forgets completely (``explicit-request-v2``)."""
        if self._complete_rule is None:
            pointer = self._writer.get_control("policy:extraction")
            if pointer is None:
                raise QuoteMemError("service_unhealthy",
                                    "Forget by statement has no released policy")
            policy = read_released_extraction_config(
                self._writer, pointer["value"]["current_release_ref"])
            self._complete_rule = policy.parameters.chat_forget == CHAT_FORGET_COMPLETE
        return self._complete_rule

    def _complete_picks(
        self, scope: ScopeV1, statements: Sequence[str], targets: Mapping[str, _Candidate],
        summaries: Set[str], sender: Send, *, phrases: Sequence[_Phrase] = (),
        veto: Callable[[str, str], Exclusion | None] | None = None,
        skip: Set[tuple[str, int, int]] = frozenset(), after: int | None = None,
        told_after: Forgotten | None = None, memory_picks: bool = True,
        against_request: bool = False,
    ) -> _Picked:
        """A complete forget's second picks: the other memories and the sentences that state it.

        Probes are the requests, the targets' statements and what the targets quote. Every
        current memory outside the targets (summary-only ones aside, which ``_restating``
        judges) sharing ``CHAT_SCAN_SHARED`` terms with a target's statement is judged against
        it; one stating it or part of it is a target too, whatever its quotes. Every readable
        sentence of the user's Evidence that shares terms with a probe, assistant replies and
        other sessions included, is judged against each request in up to ``CHAT_SCAN_CHUNKS``
        picks; the sentences picked, and every sentence that is one of these requests, are
        returned to withhold with the first forgotten memory. A sentence a target already
        quotes is left to that target's own forget.

        A complete prepare, and a complete in-chat forget, add the sentences of the Evidence
        parts near the requests' vectors (``phrases``). Each pick ``veto`` names a reason
        against, given the statement it was judged against and its text
        (`forget_match.exclusion`), is dropped. A sentence in ``skip`` (judged by an earlier
        wave of the same prepare) is not judged again; with ``after`` (the in-chat forget of a
        request its forget by statement already handled), only memories changed
        and Evidence ingested after that commit are read; with ``told_after`` (that forget
        by statement's request and confirmation time), what the user told again after it as their
        own statement, restating it, is left. Without ``memory_picks`` (a forget
        by memory id) only the sentences are judged. With ``against_request`` (a request for a
        detail), the other memories are judged against the request itself, not
        a target's statement: one that only says the thing exists or names it ("The user has
        a cat." beside "The user's orange cat is named bob.") states no value of the detail.

        Returns:
            The further memory targets, the sentences to withhold, the judge calls, every
            sentence judged or taken (each with the requests that took it) and the memories
            judged and not taken.
        """
        quoted = [span for row in targets.values() for span in row.quotes]
        probes = [*statements, *(row.statement for row in targets.values()),
                  *self._quoted(scope, quoted)]
        requests, sentences = self._scan(scope, statements, probes, quoted, phrases=phrases,
                                         after=after, told_after=told_after)
        sentences = [entry for entry in sentences
                     if (entry[0].evidence_id, entry[0].byte_start, entry[0].byte_end)
                     not in skip]
        others = [row for row in self._memories(scope, after=after)
                  if row.memory_id not in targets and row.memory_id not in summaries
                  and (told_after is None
                       or not relearns(self._tolds(row), told_after, row.statement))]
        jobs: list[tuple[str, str, list[_Candidate] | list[tuple[QuoteSpanV1, str]]]] = []
        judging: set[str] = set()
        for statement in dict.fromkeys(row.statement for row in targets.values()
                                       if memory_picks):
            if not against_request:
                if shared := _shared_terms(statement, others):
                    jobs.append(("memory", statement, shared))
                continue
            shared = [row for row in _shared_terms(statement, others)
                      if row.memory_id not in judging]
            judging |= {row.memory_id for row in shared}
            jobs += [("memory", asked, shared) for asked in statements if shared]
        size = MAX_FORGET_CANDIDATES
        for statement in statements:
            for chunk in range(0, min(len(sentences), size * CHAT_SCAN_CHUNKS), size):
                jobs.append(("sentence", statement, sentences[chunk:chunk + size]))
        requests = requests[:MAX_RESTATING_SPANS]
        taken = [_Judged(span, text, None, True) for span, text in zip(
            requests, self._quoted(scope, requests), strict=True)]
        if not jobs:
            return [], requests, 0, taken, set()

        def decide(job: tuple[str, str, list[_Candidate] | list[tuple[QuoteSpanV1, str]]]
                   ) -> ForgetPickV1:
            _, statement, rows = job
            texts = [row.statement if isinstance(row, _Candidate) else row[1] for row in rows]
            return pick_targets(statement, texts, sender, rounds=_rounds, minimal=False)

        with ThreadPoolExecutor(max_workers=max(1, min(len(jobs), MAX_JUDGE_CONCURRENCY)),
                                thread_name_prefix="quotemem-chat-forget") as pool:
            futures = [pool.submit(contextvars.copy_context().run, decide, job)
                       for job in jobs]
            decided = [future.result() for future in futures]
        memories: dict[str, _Candidate] = {}
        read: set[str] = set()
        spans: dict[tuple[str, int, int], QuoteSpanV1] = {}
        judged: dict[tuple[str, int, int], _Judged] = {}
        takers: dict[tuple[str, int, int], set[int]] = {}
        for (kind, judged_against, rows), pick in zip(jobs, decided, strict=True):
            chosen = {*pick.targets, *(i for i, role in enumerate(pick.roles) if role in _PARTS)}
            if kind == "memory":
                chosen -= _spared([row for row in rows if isinstance(row, _Candidate)],
                                  pick.roles)
            for i, row in enumerate(rows):
                text = row.statement if isinstance(row, _Candidate) else row[1]
                # A memory judged against a target's statement must not plainly state another
                # fact than the user's words either (another subject, the opposite claim).
                why: Exclusion | None = None
                if veto is not None and i in chosen:
                    why = veto(judged_against, text)
                    for asked in statements if kind == "memory" else ():
                        why = why or words_exclusion(asked, text)
                if kind == "memory" and isinstance(row, _Candidate):
                    read.add(row.memory_id)
                    if i in chosen and why is None:
                        memories.setdefault(row.memory_id, row)
                elif isinstance(row, tuple):
                    span = row[0]
                    key = (span.evidence_id, span.byte_start, span.byte_end)
                    target = i in chosen and why is None
                    if target:
                        spans[key] = span
                        takers.setdefault(key, set()).add(statements.index(judged_against))
                    if key not in judged or target:
                        judged[key] = _Judged(span, text, pick.roles[i], target, why)
        picked = [spans[key] for key in sorted(spans)][:MAX_RESTATING_SPANS - len(requests)]
        return (list(memories.values()), [*requests, *picked],
                sum(pick.calls for pick in decided),
                [*taken, *(replace(entry, asked=frozenset(takers.get(key, ())))
                           for key, entry in judged.items())],
                read - set(memories))

    def memory_sentences(self, scope: ScopeV1, memory_ids: Sequence[str],
                         chain: Sequence[str] = ()) -> MemorySentences:
        """The sentences a complete forget of memories by their ids withholds with them,
        as a forget by statement does: every readable sentence the judge
        finds stating one of their statements (``_complete_picks``, sentences only, judged
        against each statement), every clause naming one of their values as replaced
        (``_naming_clauses``: "不再打羽毛球了" of a change sentence no memory quotes, when the
        extraction kept only its climbing), and the assistant's echoes of them (``_echoes``)
        in the sessions of what the forget withholds, ``chain`` (the memories their forgets
        take) included. At most ``MAX_RESTATING_SPANS``, in order; with every sentence the
        judge read, so what it kept stays found (`remember_memory_plan`).

        Without the judge, what the word rules withhold is enough only when no sentence the
        judge would have read still names a forgotten value once it is withheld
        (``_unsettled``).

        Raises:
            QuoteMemError: ``provider_unavailable`` when the judge cannot answer and such a
                sentence is left; the judge's own errors.
        """
        rows = {row.memory_id: row for row in self._rows(scope, memory_ids)}
        if not rows:
            return MemorySentences((), ())
        statements = list(dict.fromkeys(row.statement for row in rows.values()))
        quotes = [span for row in rows.values() for span in row.quotes]
        quoted = self._quoted(scope, quotes)
        values = [*statements, *quoted]
        # A clause is read against the statements, and against the quoted words only in a
        # script no statement is in: an English quote may bundle another fact the forget
        # keeps ("..., and my sister Mei lives in Xuhui").
        scripts = {_mostly_own(statement) for statement in statements}
        naming = [*statements, *(text for text in quoted if _mostly_own(text) not in scripts)]
        summaries = {row.memory_id for row in self._summaries(scope)}
        unjudged: QuoteMemError | None = None
        try:
            _, spans, _, judged, _ = self._complete_picks(
                scope, statements, rows, summaries, self._pick_sender(), veto=exclusion,
                memory_picks=False)
        except QuoteMemError as error:
            if error.code != "provider_unavailable":
                raise
            spans, judged, unjudged = [], [], error
        named = [*(part for entry in judged if not entry.target and entry.excluded is None
                   for part in _naming_clauses(entry.span, entry.text, naming)),
                 *self._naming_sentences(scope, naming)]
        taken = [span for row in self._rows(scope, chain) for span in row.quotes]
        echoes = self._echoes(scope, [*quotes, *taken, *spans, *named], values)
        if unjudged is not None and self._unsettled(
                scope, statements, rows, [*quotes, *taken, *named, *echoes], naming):
            raise QuoteMemError("provider_unavailable",
                                "The forget judge could not read what tells the fact") from unjudged
        named_keys = {_key(span) for span in named}
        return MemorySentences(
            tuple(list({_key(span): span for span in (*spans, *named, *echoes)}.values())[
                :MAX_RESTATING_SPANS]),
            tuple(entry for entry in judged if _key(entry.span) not in named_keys))

    def _unsettled(self, scope: ScopeV1, statements: Sequence[str],
                   rows: Mapping[str, _Candidate], withheld: Sequence[QuoteSpanV1],
                   values: Sequence[str]) -> bool:
        """Whether a sentence a judge of ``statements`` would read (``_scan``) still names one
        of the forgotten ``values`` (``forget_admission.names_value``) once ``withheld`` is, or
        is in a script no value is in, which no word rule reads: without the judge, the forget
        may not go then."""
        quoted = [span for row in rows.values() for span in row.quotes]
        probes = [*statements, *(row.statement for row in rows.values()),
                  *self._quoted(scope, quoted)]
        _, sentences = self._scan(scope, statements, probes, quoted)
        held: dict[str, list[tuple[int, int]]] = {}
        for span in withheld:
            held.setdefault(span.evidence_id, []).append((span.byte_start, span.byte_end))
        scripts = {_mostly_own(value) for value in values}
        for span, text in sentences:
            readable = readable_text(text, span.byte_start, [
                (max(low, span.byte_start), min(high, span.byte_end))
                for low, high in held.get(span.evidence_id, ())
                if low < span.byte_end and span.byte_start < high])
            if any(character.isalnum() for character in readable) and (
                    _mostly_own(readable) not in scripts or names_value(readable, values)):
                return True
        return False

    def memory_plan(self, scope: ScopeV1, memory_id: str) -> MemoryPlan:
        """What a complete forget of one memory by its id takes and withholds.

        Its forget takes its base links and the linked replacers naming the forgotten
        value by the rule; an unlinked memory a word rule nominates (``ForgetOwner.nominations``)
        only once the forget judge confirms it states or presupposes the memory's statement
        (``_confirmed``), asked once, after the sentences it withholds are known
        (``memory_sentences``).

        Raises:
            QuoteMemError: ``provider_unavailable`` when the judge cannot answer, or leaves one
                contested, about a nomination or a sentence that could still tell the value;
                nothing is forgotten, and a later prepare asks again.
        """
        words = [row.statement for row in self._rows(scope, [memory_id])]
        sentences = self.memory_sentences(
            scope, [memory_id], self._owner.chains(scope, [memory_id], allowed=frozenset()))
        found = self._confirmed(scope, words, self._owner.nominations(
            scope, [memory_id], restating=sentences.spans), self._pick_sender())
        if found is None:
            raise QuoteMemError("provider_unavailable",
                                "The forget judge could not settle what the forget takes")
        chain = self._owner.chains(scope, [memory_id], restating=sentences.spans,
                                   allowed=found[0])
        return MemoryPlan(tuple(chain), sentences, frozenset(found[0]))

    def remember_memory_plan(self, scope: ScopeV1, intent_key: str, expires_at: int,
                             sentences: MemorySentences, taken: Set[str]) -> None:
        """Keep what a complete forget by memory id judged and kept (`memory_sentences`) until
        its intent is confirmed, as a forget by statement keeps its plan's: the
        confirmation records it (`record_memory_plan`), so the forget's names do not hide
        from the interactive Search and the memory list a memory it read and left ("My coach
        left the Xuhui club." beside a forgotten badminton club in Xuhui)."""
        self._remember_kept(scope, self._memory_plan(intent_key), expires_at,
                            {row.memory_id for row in self._memories(scope)} - set(taken),
                            sentences.judged)

    def record_memory_plan(self, scope: ScopeV1, intent_key: str, started: int) -> None:
        """Record, after a complete forget by memory id that began at ``started``, what its
        prepare judged and kept (`remember_memory_plan`)."""
        self._record_kept(scope, started, plan=self._memory_plan(intent_key))

    def _memory_plan(self, intent_key: str) -> ContentBindingDigest:
        """The key a forget by memory id keeps its judgments under: its intent's."""
        return self._writer.bind_content("forget-memory-plan", intent_key)

    def _naming_sentences(self, scope: ScopeV1, words: Sequence[str]) -> list[QuoteSpanV1]:
        """The clauses of every readable sentence of the scope, the user's and the assistant's,
        that name a value of ``words`` as one replaced, ended or left (``_naming_clauses``): a
        complete forget withholds them whether or not its judge read the sentence ("Since the
        move I go climbing at a gym near West Lake instead of playing badminton." shares only
        the value word with the user's "I play badminton every Saturday morning")."""
        writer = self._writer
        with writer.write_scope():
            txn = TxnRows(writer, scope.scope_kind, scope.scope_id, 0, self._clock(), "")
            rows = [row for row in txn.pages(EvidenceRows(scope_kind=scope.scope_kind,
                                                          scope_id=scope.scope_id))
                    if row["state"] == "active"]
        found: list[QuoteSpanV1] = []
        for row in rows:
            text: str = row["text"]
            held = [(span["byte_start"], span["byte_end"]) for span in row["withheld_spans"]]
            for start, end in sentences(text):
                first = row["byte_start"] + len(text[:start].encode("utf-8"))
                last = row["byte_start"] + len(text[:end].encode("utf-8"))
                sentence = text[start:end]
                if _covered(first, last, held) or is_request(sentence, complete=True):
                    continue
                # Only the readable bytes are read: a withheld word names nothing again.
                sentence = readable_text(sentence, first, [
                    (max(low, first), min(high, last)) for low, high in held
                    if first < high and low < last])
                span = QuoteSpanV1(evidence_id=row["object_id"], byte_start=first, byte_end=last)
                found += [part for part in _naming_clauses(span, sentence, words)
                          if not _covered(part.byte_start, part.byte_end, held)]
        return found

    def _echoed(self, scope: ScopeV1, statements: Sequence[str],
                targets: Mapping[str, _Candidate], bundles: Set[str]) -> list[str]:
        """What the assistant's echoes are read against (``_echoes``): the user's words, and
        the statements and quoted words of the targets that are no bundle (a bundle's other
        fact is not forgotten)."""
        own = [row for row in targets.values() if row.memory_id not in bundles]
        return [*statements, *(row.statement for row in own),
                *self._quoted(scope, [span for row in own for span in row.quotes])]

    def _echoes(self, scope: ScopeV1, withheld: Sequence[QuoteSpanV1],
                values: Sequence[str]) -> list[QuoteSpanV1]:
        """The clauses of the assistant's replies, in the sessions of the Evidence a complete
        forget withholds (``withheld``), that say nothing but the forgotten fact
        (``forget_admission.echoes``): "Shanghai" of "Shanghai, noted." once "I live in
        Shanghai." is forgotten, never "数据平台开发很有意思" for a forgotten interview at
        青川数据. Such an echo states the fact as the user's sentence did;
        a reply saying more is the judge's to read, as every sentence is. A clause already
        withheld, or inside ``withheld``, is left out."""
        writer = self._writer
        with writer.write_scope():
            sessions = {row["session_id"] for evidence_id in {span.evidence_id
                                                              for span in withheld}
                        if (row := writer.get_row("evidence", evidence_id)) is not None}
            txn = TxnRows(writer, scope.scope_kind, scope.scope_id, 0, self._clock(), "")
            rows = [row for row in txn.pages(EvidenceRows(scope_kind=scope.scope_kind,
                                                          scope_id=scope.scope_id))
                    if row["state"] == "active" and row["kind"] == "assistant_message"
                    and row["session_id"] in sessions]
        planned: dict[str, list[tuple[int, int]]] = {}
        for span in withheld:
            planned.setdefault(span.evidence_id, []).append((span.byte_start, span.byte_end))
        found: list[QuoteSpanV1] = []
        for row in rows:
            text: str = row["text"]
            held = [(span["byte_start"], span["byte_end"]) for span in row["withheld_spans"]]
            held += planned.get(row["object_id"], [])
            start = 0
            for end in [*(match.start() for match in _CLAUSE_MARKS.finditer(text)), len(text)]:
                clause = text[start:end]
                lead = start + len(clause) - len(clause.lstrip())
                body = clause.strip()
                # "Badminton on Saturdays and a job at Tencent": each conjunct on its own.
                at = 0
                for cut in [*(match for match in _CONJUNCTS.finditer(body)), None]:
                    piece = body[at:cut.start() if cut is not None else len(body)]
                    part = piece.strip()
                    if part and echoes(part, values):
                        offset = lead + at + len(piece) - len(piece.lstrip())
                        first = row["byte_start"] + len(text[:offset].encode("utf-8"))
                        last = first + len(part.encode("utf-8"))
                        if not _covered(first, last, held):
                            found.append(QuoteSpanV1(evidence_id=row["object_id"],
                                                     byte_start=first, byte_end=last))
                    if cut is not None:
                        at = cut.end()
                start = end + 1
        return found

    def _scan(
        self, scope: ScopeV1, statements: Sequence[str], probes: Sequence[str],
        quoted: Sequence[QuoteSpanV1], *, phrases: Sequence[_Phrase] = (),
        after: int | None = None, told_after: Forgotten | None = None,
    ) -> tuple[list[QuoteSpanV1], list[tuple[QuoteSpanV1, str]]]:
        """The user's sentences that are these requests, and the sentences to judge.

        Every sentence of the scope's active Evidence is read. A sentence that is a forget
        request is one of these requests (its asked statement is one of ``statements``, a
        named speaker's prefix aside), taken whole unless already withheld, even where a
        memory recording it keeps a name readable, or another request, never a candidate.
        Any other sentence overlapping a withheld range or a target's quote is skipped. Any
        other sentence sharing ``CHAT_SCAN_SHARED`` `content_terms` with a probe (all of a short
        probe's) is a candidate, ranked by the largest share of a probe's terms it holds; so
        is one naming a shadow name of a probe (`recall_agent.shadow_terms`), after those
        (``NAMED_SHARE``): the forget's names hide every sentence it did not judge and keep
        (`recall_agent.forget_shadowed`), so it judges what they would hide. Given
        ``phrases``, the requests' vectors, every other sentence of every part at
        ``DENSE_FLOOR`` or more to one of them is a candidate too, nearest part first: a fact
        told in another language than the request shares no term with it. Of the
        ``CHAT_SCAN_CHUNKS * MAX_FORGET_CANDIDATES`` places, those by vector take up to half,
        those by terms the rest, and either takes what the other leaves; they are returned
        with their text. With ``after``, only the parts ingested after that commit are read;
        with ``told_after``, a sentence the user told as their own statement after it,
        restating it, is no candidate.
        """
        wanted = [terms for probe in probes if (terms := content_terms(probe))]
        # Every sentence naming a name the forget's shadow will hold is read, so what that
        # name hides from the interactive Search is what the judge took or never saw, not
        # what it would have kept (a correction naming the old city).
        named = {name for probe in probes for name in shadow_terms(probe).names}
        asked = {asked_part(statement) for statement in statements}
        held_quotes: dict[str, list[tuple[int, int]]] = {}
        for span in quoted:
            held_quotes.setdefault(span.evidence_id, []).append((span.byte_start, span.byte_end))
        writer = self._writer
        with writer.write_scope():
            txn = TxnRows(writer, scope.scope_kind, scope.scope_id, 0, self._clock(), "")
            rows = [row for row in txn.pages(EvidenceRows(scope_kind=scope.scope_kind,
                                                          scope_id=scope.scope_id))
                    if row["state"] == "active"
                    and (after is None or row["ingested_commit_seq"] > after)]
        complete = self._complete()
        near: dict[str, float] = {}
        for phrase in phrases:
            for row in rows:
                # A complete forget reads a part an earlier forget withheld bytes of by its
                # remainder's vector (``ForgetRemainderV1``), never by the stored one.
                vector = (stored_remainder(row) if complete and row["withheld_spans"]
                          else row["embedding"])
                if vector is not None and row["embedding_space"] == phrase.space and any(
                        vector) and (score := cosine(phrase.vector, unit_vector(vector))
                                     ) >= max(DENSE_FLOOR, near.get(row["object_id"], 0.0)):
                    near[row["object_id"]] = score
        requests: list[QuoteSpanV1] = []
        found: list[tuple[float, QuoteSpanV1, str]] = []
        dense: list[tuple[float, QuoteSpanV1, str]] = []
        for row in rows:
            text: str = row["text"]
            withheld = [(span["byte_start"], span["byte_end"]) for span in row["withheld_spans"]]
            quotes = held_quotes.get(row["object_id"], [])
            held = [*withheld, *quotes]
            # A request sentence, as the plugins split a message ("... Dr. Sato."), is never
            # judged; it is taken when it asks one of these requests, only as far as the
            # request goes: a clear replacement said after it ("..., I live at 27 Birch Lane
            # now.") stays (`request_reach`).
            asking_ranges: list[tuple[int, int]] = []
            for start, end in request_sentences(text):
                sentence = text[start:end]
                if not is_request(sentence, complete=True):
                    continue
                asking_ranges.append((start, end))
                first = row["byte_start"] + len(text[:start].encode("utf-8"))
                stop = start + (request_reach(sentence) or len(sentence))
                last = row["byte_start"] + len(text[:stop].encode("utf-8"))
                if not _covered(first, last, withheld) and any(
                        statement == own or statement.endswith(f": {own}")
                        for own in map(asked_part, requested(sentence, complete=True))
                        for statement in asked):
                    requests.append(QuoteSpanV1(evidence_id=row["object_id"], byte_start=first,
                                                byte_end=last))
            for start, end in sentences(text):
                first = row["byte_start"] + len(text[:start].encode("utf-8"))
                last = row["byte_start"] + len(text[:end].encode("utf-8"))
                sentence = text[start:end]
                span = QuoteSpanV1(evidence_id=row["object_id"], byte_start=first, byte_end=last)
                if any(low < end and start < high for low, high in asking_ranges):
                    continue
                if any(first < high and low < last for low, high in held):
                    # A complete forget judges the readable rest of a sentence an earlier
                    # forget withheld part of, as Search finds it (its remainder), never the
                    # withheld bytes: "... moved to Xiamen, and I live in Xiamen now."
                    if not complete or _covered(first, last, withheld) or any(
                            first < high and low < last for low, high in quotes):
                        continue
                    sentence = readable_text(sentence, first, [
                        (max(low, first), min(high, last)) for low, high in withheld
                        if first < high and low < last])
                if told_after is not None and restates(told_quote(
                        _Part.of(row), first, last, asserts), told_after):
                    continue
                terms = content_terms(sentence)
                share = max((len(terms & probe) / len(probe) for probe in wanted
                             if len(terms & probe) >= min(CHAT_SCAN_SHARED, len(probe))),
                            default=0.0)
                if not share and named and not named.isdisjoint(sparse_terms(sentence)):
                    share = NAMED_SHARE
                if share:
                    found.append((share, span, sentence))
                elif row["object_id"] in near:
                    dense.append((near[row["object_id"]], span, sentence))
        found.sort(key=lambda entry: (-entry[0], entry[1].evidence_id, entry[1].byte_start))
        dense.sort(key=lambda entry: (-entry[0], entry[1].evidence_id, entry[1].byte_start))
        bound = CHAT_SCAN_CHUNKS * MAX_FORGET_CANDIDATES
        by_vector = min(len(dense), max(bound // 2, bound - len(found)))
        return requests, [(span, sentence) for _, span, sentence in
                          (*found[:bound - by_vector], *dense[:by_vector])]

    def _said_fact(self, span: QuoteSpanV1) -> bool:
        """Whether a span lies in a user message and is no forget request."""
        writer = self._writer
        with writer.write_scope():
            row = writer.get_row("evidence", span.evidence_id)
        return row is not None and row["role"] == "user" and not self._quoted_request(
            span)

    def _quoted_request(self, span: QuoteSpanV1) -> bool:
        """Whether a sentence span is a forget request (a complete forget reads the ``agent``
        mode's forms too: "别记我小时候住哪了")."""
        writer = self._writer
        with writer.write_scope():
            row = writer.get_row("evidence", span.evidence_id)
        if row is None:
            return False
        raw = row["text"].encode("utf-8")
        return is_request(raw[span.byte_start - row["byte_start"]:
                              span.byte_end - row["byte_start"]].decode("utf-8", "replace"),
                          complete=self._complete())

    def _relearned(
        self, scope: ScopeV1, request_id: str, sender: Send,
    ) -> tuple[list[_Candidate], list[QuoteSpanV1], _Stone | None, int, dict[str, int]]:
        """What one Add restated of a forgotten statement: its memories, its sentences, the
        tombstone the first sentence restates, the judge calls, and for each memory the
        admission time of the forget its tombstone records (the forget enforced on it:
        `ForgetOwner.prepare`'s ``enforces``).

        The Add's memories are the current ones changed since its earliest Evidence was
        ingested; its sentences are the readable sentences of its own Evidence, user and
        assistant alike, so a restatement the extraction stored nowhere (a question about the
        forgotten fact, an acknowledgement of the forget) is found too. A memory whose
        statement is a tombstone's statement word for word (an entity forgotten with its last
        memory is minted anew, so the tombstone's own key no longer matches) is taken
        without a vote. Each other memory or sentence that restates a tombstone's statement
        (`_restates`), or whose vector (a sentence's: its Evidence part's) is at
        ``DENSE_FLOOR`` or more to the tombstone's (a restatement in another language), is
        judged against it, the ``MAX_FORGET_CANDIDATES`` nearest in one pick, and taken when
        it states it or part of it and states no other fact (`forget_match.exclusion`):
        ``RELEARN_PICKS`` tombstones at most, those restated in words first, then the nearest.
        No tombstone, no Evidence of the Add or no such memory or sentence makes no call.

        What the user told again after a tombstone's forget, as their own statement restating
        it (``core/relearn.py``: a dated user sentence received and dated after
        it, no forget request, no memory cue, no question, whose words state the tombstone; a
        memory one of whose support groups is only such sentences, or is such sentences
        stating the memory in full while it tells part of the tombstone), is never taken
        for that tombstone: the user says it now. The assistant's text, the request and its quoted
        words, a question, anything undated or dated before the forget, and a later sentence
        about something else still are.
        """
        stones = self._tombstones(scope)
        if not stones:
            return [], [], None, 0, {}
        writer = self._writer
        with writer.write_scope():
            txn = TxnRows(writer, scope.scope_kind, scope.scope_id, 0, self._clock(), "")
            parts = list(txn.pages(EvidenceRows(
                scope_kind=scope.scope_kind, scope_id=scope.scope_id, request_id=request_id)))
        if not parts:
            return [], [], None, 0, {}
        written = self._memories(scope, after=max(
            0, min(row["ingested_commit_seq"] for row in parts) - 1))
        # What the user told again, in their own words, after a tombstone's
        # forget states the fact anew and is not taken for that tombstone.
        told = {row.memory_id: self._tolds(row) for row in written}
        exact = {_words(stone.statement): stone for stone in stones}
        found = {row.memory_id: row for row in written
                 if (stone := exact.get(_words(row.statement))) is not None
                 and not relearns(told[row.memory_id], stone.forgotten(), row.statement)}
        enforced = {memory_id: exact[_words(row.statement)].created_at
                    for memory_id, row in found.items()}
        rest: list[tuple[_Candidate | tuple[QuoteSpanV1, str], str, UnitVector | None,
                         str, list[Told | None]]] = [
            (row, row.statement, unit_vector(row.vector) if row.vector else None, row.space,
             told[row.memory_id])
            for row in written if row.memory_id not in found]
        for part in parts:
            if part["state"] != "active":
                continue
            unit = unit_vector(part["embedding"]) if any(part["embedding"]) else None
            readable = _Part.of(part)
            for span, text in self._readable(part):
                rest.append(((span, text), text, unit, part["embedding_space"],
                             [told_quote(readable, span.byte_start, span.byte_end, asserts)]))
        scored: dict[str, list[tuple[float, int]]] = {}
        for stone in stones:
            terms, vector = content_terms(stone.statement), stone.unit
            forgotten = stone.forgotten()
            for position, (entry, text, unit, space, when) in enumerate(rest):
                # A memory the user's later words state in full is relearned for the part
                # of the tombstone it tells; a sentence only when its words restate it.
                if relearns(when, forgotten,
                            text if isinstance(entry, _Candidate) else None):
                    continue
                score = _LEXICAL if _restates(text, [terms]) else (
                    cosine(vector, unit) if vector is not None and unit is not None
                    and space == stone.space else 0.0)
                if score >= DENSE_FLOOR:
                    scored.setdefault(stone.statement, []).append((score, position))
        by_statement = {stone.statement: stone for stone in stones}
        ranked = sorted(scored.items(), key=lambda entry: -max(score for score, _ in entry[1]))
        jobs = [(by_statement[statement], [position for _, position in sorted(
            rows, key=lambda entry: (-entry[0], entry[1]))][:MAX_FORGET_CANDIDATES])
            for statement, rows in ranked[:RELEARN_PICKS]]
        if not jobs:
            return list(found.values()), [], None, 0, enforced

        def decide(job: tuple[_Stone, list[int]]) -> ForgetPickV1:
            return pick_targets(job[0].statement, [rest[position][1] for position in job[1]],
                                sender, rounds=_rounds, minimal=False)

        with ThreadPoolExecutor(max_workers=max(1, min(len(jobs), MAX_JUDGE_CONCURRENCY)),
                                thread_name_prefix="quotemem-chat-forget") as pool:
            futures = [pool.submit(contextvars.copy_context().run, decide, job)
                       for job in jobs]
            decided = [future.result() for future in futures]
        spans: dict[tuple[str, int, int], QuoteSpanV1] = {}
        seeded: _Stone | None = None
        for (stone, positions), pick in zip(jobs, decided, strict=True):
            for i in sorted({*pick.targets, *(i for i, role in enumerate(pick.roles)
                                              if role in _PARTS)}):
                entry, text, _, _, _ = rest[positions[i]]
                if exclusion(stone.statement, text) is not None:
                    continue
                if isinstance(entry, _Candidate):
                    found.setdefault(entry.memory_id, entry)
                    enforced.setdefault(entry.memory_id, stone.created_at)
                else:
                    spans.setdefault((entry[0].evidence_id, entry[0].byte_start,
                                      entry[0].byte_end), entry[0])
                    seeded = seeded or stone
        return (list(found.values()), [spans[key] for key in sorted(spans)], seeded,
                sum(pick.calls for pick in decided), enforced)

    def _tolds(self, row: _Candidate) -> list[Told | None]:
        """What each support group of a memory says as the user's own statement
        (`relearn.group_told`)."""
        writer = self._writer
        with writer.write_scope():
            parts = {span.evidence_id: writer.get_row("evidence", span.evidence_id)
                     for span in row.quotes}
        return [group_told(
            ((None if (part := parts[row.quotes[index].evidence_id]) is None
              else _Part.of(part), row.quotes[index].byte_start, row.quotes[index].byte_end)
             for index in group), asserts) for group in row.groups]

    @staticmethod
    def _readable(part: Mapping[str, object]) -> Iterator[tuple[QuoteSpanV1, str]]:
        """The sentences of an Evidence part that no withheld range meets, with their text."""
        text = cast(str, part["text"])
        base = cast(int, part["byte_start"])
        held = [(cast(int, span["byte_start"]), cast(int, span["byte_end"]))
                for span in cast(list[dict[str, object]], part["withheld_spans"])]
        for start, end in sentences(text):
            first = base + len(text[:start].encode("utf-8"))
            last = base + len(text[:end].encode("utf-8"))
            if not any(first < high and low < last for low, high in held):
                yield (QuoteSpanV1(evidence_id=cast(str, part["object_id"]), byte_start=first,
                                   byte_end=last), text[start:end])

    def _request_records(self, scope: ScopeV1, statements: Sequence[str]) -> list[_Candidate]:
        """The current memories recording one of these requests: every quote is a forget
        request and one of them asks one of ``statements`` (a named speaker's prefix aside).

        Such a memory ("The user asked to forget that ...") restates what the user asked to
        forget; a complete forget forgets it with the fact. One stating the new value the
        sentence told after its request ("The user lives in Hangzhou, not Shanghai." of
        "Forget that I lived in Shanghai because I moved to Hangzhou.", `_states_new`) records
        a fact the user told, never the request.
        """
        asked = {asked_part(statement) for statement in statements}
        found: list[_Candidate] = []
        for row in self._memories(scope):
            texts = self._quoted(scope, row.quotes)
            if not texts or len(texts) != len(row.quotes) or not all(
                    is_request(text, complete=True) for text in texts) or any(
                    _states_new(row.statement, text) for text in texts):
                continue
            if any(statement == own or statement.endswith(f": {own}")
                   for text in texts for own in map(asked_part, requested(text, complete=True))
                   for statement in asked):
                found.append(row)
        return found

    def _naming(self, scope: ScopeV1, words: Sequence[str], targets: Mapping[str, _Candidate],
                subjects: Set[str] = frozenset({USER}),
                ) -> list[_Candidate]:
        """Every readable memory outside ``targets``, current or past, about one of
        ``subjects`` and no other person, whose statement names a value of the user's
        ``words`` as one it replaced or ended (``forget_admission.names_replaced``, never
        failing closed on a guess): the nominations the forget judge confirms or not
        (``_confirmed``)."""
        writer = self._writer
        ids: list[str] = []
        with writer.write_scope():
            read = readability(TxnRows(writer, scope.scope_kind, scope.scope_id, 0,
                                       self._clock(), ""))
            for item in writer.iter_rows("item"):
                if (item["scope_kind"], item["scope_id"]) != (scope.scope_kind, scope.scope_id) \
                        or item["object_id"] in targets or item["lifecycle_state"] == "erased":
                    continue
                head = writer.get_row("revision", item["current_revision_id"])
                if head is None:
                    continue
                snapshot = MemorySnapshotV2.model_validate(head["snapshot"])
                if read.p_ver(snapshot) and snapshot.entity_id in subjects and any(
                        not other_subject(asked, snapshot.statement)
                        and names_replaced(snapshot.statement, asked, unsure=False)
                        for asked in words):
                    ids.append(item["object_id"])
        return self._rows(scope, ids)

    def _naming_subjects(self, scope: ScopeV1, targets: Mapping[str, _Candidate]) -> frozenset[str]:
        """Whom a memory found naming the asked value may be about: the user, or the primary
        entity of a memory the judge picked (`_naming`)."""
        return frozenset({USER, *self._subjects(scope, list(targets)).values()})

    def _subjects(self, scope: ScopeV1, memory_ids: Sequence[str]) -> dict[str, str]:
        """The primary entity of each memory's head."""
        writer = self._writer
        found: dict[str, str] = {}
        with writer.write_scope():
            for memory_id in memory_ids:
                item = writer.get_row("item", memory_id)
                if item is None or (item["scope_kind"], item["scope_id"]) != (
                        scope.scope_kind, scope.scope_id):
                    continue
                head = writer.get_row("revision", item["current_revision_id"])
                if head is not None:
                    found[memory_id] = MemorySnapshotV2.model_validate(
                        head["snapshot"]).entity_id
        return found

    def _rows(self, scope: ScopeV1, memory_ids: Sequence[str]) -> list[_Candidate]:
        """The scope's memories of ``memory_ids``, current or past, by their head, in order."""
        writer = self._writer
        found: list[_Candidate] = []
        with writer.write_scope():
            for memory_id in memory_ids:
                item = writer.get_row("item", memory_id)
                head = None if item is None or (item["scope_kind"], item["scope_id"]) != (
                    scope.scope_kind, scope.scope_id) else writer.get_row(
                        "revision", item["current_revision_id"])
                if item is None or head is None:
                    continue
                snapshot = MemorySnapshotV2.model_validate(head["snapshot"])
                found.append(_Candidate(memory_id, head["object_id"], head["content_hash"],
                                        snapshot.statement, tuple(snapshot.quote_spans),
                                        tuple(snapshot.embedding), snapshot.embedding_space,
                                        tuple(snapshot.keywords),
                                        tuple(map(tuple, snapshot.support_groups))))
        return found

    def _memories(self, scope: ScopeV1, *, after: int | None = None,
                  past: bool = False) -> list[_Candidate]:
        """The scope's current (``p_mem``) memories, in id order; with ``after``, only those a
        transaction after that ``commit_seq`` changed; with ``past``, the past ones whose
        head is readable (``p_ver``) too, which the memory list shows as past."""
        writer = self._writer
        found: list[_Candidate] = []
        with writer.write_scope():
            txn = TxnRows(writer, scope.scope_kind, scope.scope_id, 0, self._clock(), "")
            read = readability(txn)
            for item in txn.pages(MemoryTableRows(scope_kind=scope.scope_kind,
                                                  scope_id=scope.scope_id, after_commit_seq=after)):
                head = writer.get_row("revision", item["current_revision_id"])
                if head is None:
                    continue
                snapshot = MemorySnapshotV2.model_validate(head["snapshot"])
                if read.p_mem(item["lifecycle_state"], snapshot) or (
                        past and item["lifecycle_state"] not in ("active", "erased")
                        and read.p_ver(snapshot)):
                    found.append(_Candidate(item["object_id"], head["object_id"],
                                            head["content_hash"], snapshot.statement,
                                            tuple(snapshot.quote_spans),
                                            tuple(snapshot.embedding), snapshot.embedding_space,
                                            tuple(snapshot.keywords),
                                            tuple(map(tuple, snapshot.support_groups))))
        return found

    def _quoted(self, scope: ScopeV1, quotes: Sequence[QuoteSpanV1]) -> list[str]:
        """The text of every quote that lies in the scope's Evidence, in order."""
        texts: list[str] = []
        writer = self._writer
        with writer.write_scope():
            for span in quotes:
                row = writer.get_row("evidence", span.evidence_id)
                if row is None or (row["scope_kind"], row["scope_id"]) != (
                        scope.scope_kind, scope.scope_id):
                    continue
                raw = row["text"].encode("utf-8")
                texts.append(raw[span.byte_start - row["byte_start"]:
                                 span.byte_end - row["byte_start"]].decode("utf-8", "replace"))
        return texts

    def _leftovers(self, scope: ScopeV1, cover: Sequence[QuoteSpanV1],
                   names: Set[tuple[str, str]] = frozenset(),
                   ties: Mapping[str, frozenset[str]] | None = None,
                   told: Sequence[str] | None = None,
                   within: Within | None = None,
                   requests: Sequence[str] | None = None,
                   residue: Set[str] | None = None,
                   modifier: bool = False,
                   sentences: Sequence[QuoteSpanV1] = ()) -> list[_Candidate]:
        """The scope's live memories known only from what the user asked to forget.

        - A memory whose head fails ``p_ver``, every quote of which lies inside ranges withheld
          by a forget (``forgotten``, or ``restated_forgotten`` for a denied restatement) and
          one of which meets ``cover``. A forget leaves such a dependent active and unreadable
          (``forget_dependent``), so new support, such as the request itself said again,
          could make it readable again. A memory that lost its support only to a sentence
          withheld as restating the fact (``restating``) quotes no forgotten memory's words:
          it states another fact told in the same sentence and is not forgotten (it stays
          unreadable until it is told again, and no tombstone keeps it from being told).
        - A bare naming memory, one that names its single non-user entity and anchors nothing
          else ("A company is named 星河科技."), every quote of which lies inside those ranges
          or inside ``cover``, the quotes of the memories just forgotten. A forget keeps such a
          name readable so the entity keeps its name (``plan_stating_spans``); when the
          entity was only ever told as part of the forgotten fact, its name is that fact's
          residue: so one of the forgotten statements (``told``, the memories just forgotten,
          or the scope's tombstones) shows the name ("The user works at 星河科技."); another
          person's name beside a forgotten one ("A person is named 小王." in "我的两个同事小王、
          小李" under a forgotten "A person is named 小李.") is no residue. With the user's
          ``requests`` (`bare_name_stays`), a name they give, or a name request does not keep,
          is residue whatever shows it, and a shown name stays when another readable memory is
          about the entity and no fact forgotten from its sentence (the tombstones citing it)
          states it of another entity. A confirmation passes no ``requests``: its cascade
          takes the bare names its prepare named as forgotten (``residue``). A naming memory
          that also
          anchors the user or another entity ("The user's partner is named Alex.") states a
          relation the user did not ask to forget and stays.
        - A memory a forget requoted (``privacy_requote``: compensation moved its quotes or dropped
          its forgotten keywords) whose own statement still restates a forgotten statement
          (`_restates`): the requote changes what it quotes, not what it says.
        - A readable memory that still shows one of ``names``, the names the forgotten
          memories gave (`_name_holders`): it tells the forgotten name whatever it says.
        """
        covering: dict[str, list[tuple[int, int]]] = {}
        for span in cover:
            covering.setdefault(span.evidence_id, []).append((span.byte_start, span.byte_end))
        # A bare name quoted in a sentence the forget withheld as stating the fact is its residue
        # as one a forgotten memory quotes is (``sentences``).
        naming_cover = {evidence: list(ranges) for evidence, ranges in covering.items()}
        for span in sentences:
            naming_cover.setdefault(span.evidence_id, []).append(
                (span.byte_start, span.byte_end))
        tombstones = self._tombstone_statements(scope)
        stones = [(statement, terms) for statement in tombstones
                  if len(terms := content_terms(statement)) >= CHAT_SCAN_SHARED]
        # The facts forgotten: the tombstones', with their entities, and ``told``.
        facts: list[ForgottenFact] = [*self._tombstone_facts(scope),
                                      *((statement, "", ()) for statement in told or ())]
        writer = self._writer
        found: list[_Candidate] = []
        with writer.write_scope():
            txn = TxnRows(writer, scope.scope_kind, scope.scope_id, 0, self._clock(), "")
            read = readability(txn)
            live: list[tuple[dict[str, Any], dict[str, Any], MemorySnapshotV2]] = []
            for item in txn.pages(MemoryTableRows(scope_kind=scope.scope_kind,
                                                  scope_id=scope.scope_id)):
                if item["lifecycle_state"] == "erased":
                    continue
                head = writer.get_row("revision", item["current_revision_id"])
                if head is not None:
                    live.append((item, head, MemorySnapshotV2.model_validate(head["snapshot"])))
            about = _told(snapshot for _, _, snapshot in live
                          if read.p_ver(snapshot) and not naming_only(snapshot))
            known = {entry.name for _, _, snapshot in live for entry in snapshot.naming}
            tied = names_tied((snapshot.statement, snapshot.entity_id, snapshot.about_entity_ids)
                              for _, _, snapshot in live)
            for item, head, snapshot in live:
                if not snapshot.quote_spans:
                    continue
                if read.p_ver(snapshot):
                    # With the user's words, a bare name is residue unless it stays
                    # (`bare_name_stays`); a confirmation takes those its prepare named.
                    bare = naming_only(snapshot) and (
                        not bare_name_stays(snapshot, requests, known, facts, tied,
                                            told=snapshot.entity_id in about)
                        if requests is not None else residue is None
                        or item["object_id"] in residue)
                    # A survivor kept on its own clause ("The user has a tabby cat." after
                    # "The user's tabby cat is named Pixel.") shares the forgotten statement's
                    # description but shows none of what it says of it.
                    # A rewording already showed none of it (``ForgetOwner``).
                    restates = (head["op"], head["reason_code"]) == (
                        "requote", "privacy_requote") and not self._reworded_head(head) and any(
                        _restates(snapshot.statement, [terms]) and support_words(
                            snapshot.statement) & _said(statement, snapshot.statement,
                                                        modifier=modifier)
                        # Over a Han quote, another person's name restates no forgotten one
                        # ("A person is named 小王." after "A person is named 小李.").
                        and not (names_another(statement, snapshot.statement) and not all(
                            text.isascii() for text in self._quote_texts(
                                list(snapshot.quote_spans))))
                        for statement, terms in stones)
                    if not restates and (not bare or not all(
                            self._forget_withheld(span, naming_cover)
                            for span in snapshot.quote_spans)):
                        continue
                elif not all(self._forget_withheld(span)
                             for span in snapshot.quote_spans) or not any(
                        start < span.byte_end and span.byte_start < end
                        for span in snapshot.quote_spans
                        for start, end in covering.get(span.evidence_id, ())):
                    continue
                found.append(_Candidate(item["object_id"], head["object_id"],
                                        head["content_hash"], snapshot.statement,
                                        tuple(snapshot.quote_spans)))
        known = {row.memory_id for row in found}
        return [*found, *(row for row in self._name_holders(scope, names, ties=ties,
                                                            within=within)
                          if row.memory_id not in known)]

    def _reworded_head(self, head: Mapping[str, object]) -> bool:
        """Whether a head is a forget's rewording: a privacy requote stating something else
        than the version before it (``publication.history.reworded_through``); under the
        Writer scope."""
        previous = head["previous_revision_id"]
        before = None if not isinstance(previous, str) else self._writer.get_row(
            "revision", previous)
        snapshot = cast(Mapping[str, object], head["snapshot"])
        return head["op"] == "requote" and before is not None and (
            before["snapshot"]["statement_digest"] != snapshot["statement_digest"])

    def _tombstones(self, scope: ScopeV1) -> list[_Stone]:
        """The scope's tombstones, one per statement, each with its vector and space."""
        writer = self._writer
        found: dict[str, _Stone] = {}
        with writer.write_scope():
            txn = TxnRows(writer, scope.scope_kind, scope.scope_id, 0, self._clock(), "")
            for row in txn.pages(TombstoneRows(scope_kind=scope.scope_kind,
                                               scope_id=scope.scope_id, after_tombstone_seq=0)):
                if row["statement"] and (
                        row["statement"] not in found
                        or row["created_at"] > found[row["statement"]].created_at):
                    found[row["statement"]] = _Stone(row["statement"], tuple(row["embedding"]),
                                                     row["embedding_space"], row["created_at"],
                                                     row["entity_id"])
        return list(found.values())

    def _tombstone_facts(self, scope: ScopeV1) -> list[ForgottenFact]:
        """The facts the scope's forgets took, from their tombstones: each statement, with its
        entity and its other anchored entities."""
        writer = self._writer
        with writer.write_scope():
            txn = TxnRows(writer, scope.scope_kind, scope.scope_id, 0, self._clock(), "")
            return [(row["statement"], row["entity_id"], tuple(row["about_entity_ids"]))
                    for row in txn.pages(TombstoneRows(scope_kind=scope.scope_kind,
                                                       scope_id=scope.scope_id,
                                                       after_tombstone_seq=0))
                    if row["statement"]]

    def _tombstone_statements(self, scope: ScopeV1) -> list[str]:
        """The statements of the scope's tombstones."""
        writer = self._writer
        with writer.write_scope():
            txn = TxnRows(writer, scope.scope_kind, scope.scope_id, 0, self._clock(), "")
            return [row["statement"] for row in txn.pages(TombstoneRows(
                scope_kind=scope.scope_kind, scope_id=scope.scope_id, after_tombstone_seq=0))
                if row["statement"]]

    def _forget_withheld(
        self, span: QuoteSpanV1, cover: Mapping[str, Sequence[tuple[int, int]]] | None = None,
    ) -> bool:
        """Whether ranges a forget withheld, with ``cover``'s, cover the span (under the
        Writer scope)."""
        row = self._writer.get_row("evidence", span.evidence_id)
        if row is None or row["state"] != "active":
            return False
        return _covered(span.byte_start, span.byte_end, [
            *((held["byte_start"], held["byte_end"]) for held in row["withheld_spans"]
              if held["reason"] in _FORGET_REASONS),
            *(() if cover is None else cover.get(span.evidence_id, ()))])

    def _cascade(
        self, scope: ScopeV1, authorize: Callable[[], PrincipalGrant], session_id: str,
        cover: Sequence[QuoteSpanV1], *, enforces: int | None = None,
        allowed: Set[str] | None = None, names: Set[tuple[str, str]] = frozenset(),
        ties: Mapping[str, frozenset[str]] | None = None, told: Sequence[str] | None = None,
        within: Within | None = None,
        requests: Sequence[str] | None = None, residue: Set[str] | None = None,
        modifier: bool = False, asked: bool = False, sentences: Sequence[QuoteSpanV1] = (),
        rewordings: Sequence[ForgetRewordV1] = (),
    ) -> int:
        """Forget every memory known only from what was forgotten
        (`_leftovers`, ``cover`` the quotes of the memories just forgotten, growing with each
        memory the cascade forgets), up to ``CASCADE_ROUNDS`` passes; how many were
        forgotten. A memory the judge kept may be one of them (a name only the forgotten
        sentence told): the caller reports it. ``enforces`` is the relearn check's
        (`_forget`): the cascade of an Add that only restated forgotten facts forgets
        nothing new either. A confirmation forgets only the memories its prepare named
        (``allowed``): another one stays as the forget left it. ``names`` are the names
        the forgotten memories gave (`_leftovers`' fourth kind); ``told`` their statements,
        one of which must show a bare name for it to be their residue (None: any).
        ``requests`` are the user's words, which decide a bare name's residue instead
        (`_leftovers`' second kind): the in-chat forget and the forget of sentences pass
        them; a confirmation passes none and takes the bare names its prepare named as
        forgotten (``residue``). With ``modifier`` the user asked to forget a modifier alone
        (`_forget`); ``asked``, or ``requests`` asking a name (``asks_name``) or everything
        about someone (``asks_everything``), takes the memories still showing a bare name
        taken as the asked name's residue. A memory the forget reworded (``rewordings``)
        says no forgotten word, so it stays whatever its revision is."""
        forgotten = 0
        cover = list(cover)
        told_now = None if told is None else list(told)
        # A name asked goes whole: a readable bare name taken as its residue gives its name to
        # the memories still showing it (`_leftovers`' fourth kind), found while they are
        # still readable, before that name is withheld; a confirmation takes only those its
        # prepare named (``allowed``, `_affected`).
        asked = asked or any(asks_name(text) or asks_everything(text) for text in requests or ())
        names = set(names)

        def left(names: Set[tuple[str, str]], reworded: Set[str]) -> list[_Candidate]:
            # What the cascade may take: a memory its prepare did not name, and a reworded one,
            # give no name either.
            return [row for row in self._leftovers(scope, cover, names, ties, told_now, within,
                                                   requests, residue, modifier, sentences)
                    if (allowed is None or row.memory_id in allowed)
                    and row.memory_id not in reworded]

        for _ in range(CASCADE_ROUNDS):
            reworded = {row.memory_id for row in self._rows(scope, [
                wording.memory_id for wording in rewordings]) if row.statement in {
                    wording.statement for wording in rewordings
                    if wording.memory_id == row.memory_id
                    and wording.base_digest != statement_digest(wording.statement)}}
            leftovers = left(names, reworded)
            shown = {row.memory_id for row in self._memories(scope)} if asked else set[str]()
            grown = names | {(entry.entity_id, entry.name) for snapshot in self._heads(
                scope, [row.memory_id for row in leftovers if row.memory_id in shown]).values()
                if naming_only(snapshot) for entry in snapshot.naming}
            if grown != names:
                names = grown
                leftovers = left(names, reworded)
            done = 0
            for row in leftovers:
                target = ForgetStatementTargetV1(
                    memory_id=row.memory_id, expected_revision_id=row.revision_id,
                    expected_content_hash=IntegrityDigest(row.content_hash))
                # It is forgotten for its support alone: its own value is not, so no memory
                # that replaced it goes with it.
                if self._forget(scope, authorize(), session_id, target, enforces=enforces,
                                support_only=True, modifier=modifier, rewordings=rewordings,
                                ).outcome == "verified":
                    done += 1
                    cover += row.quotes
                    if told_now is not None:
                        told_now.append(row.statement)
            forgotten += done
            if not done:
                break
        return forgotten

    def _restating(
        self, statements: Sequence[str], summaries: Sequence[_Candidate], sender: Send,
    ) -> tuple[set[str], int]:
        """The summary-only memories that state a memory target's statement, and the calls.

        Each target's statement (a stored paraphrase of what is forgotten, worded like the
        summaries extracted from the same messages) is a pick target of its own over the
        ``MAX_FORGET_CANDIDATES`` summaries sharing the most of its content words; every
        summary the judge decides states it or part of it (``same_fact``, ``part``) is
        returned. A summary that only names the subject, or tells other facts about it,
        stays (``identity``, ``unrelated``).
        """
        jobs = [(statement, shared) for statement in dict.fromkeys(statements)
                if (shared := _sharing(statement, summaries))]
        if not jobs:
            return set(), 0

        def decide(job: tuple[str, list[_Candidate]]) -> ForgetPickV1:
            return pick_targets(job[0], [row.statement for row in job[1]], sender,
                                rounds=_rounds)

        with ThreadPoolExecutor(max_workers=max(1, min(len(jobs), MAX_JUDGE_CONCURRENCY)),
                                thread_name_prefix="quotemem-chat-forget") as pool:
            futures = [pool.submit(contextvars.copy_context().run, decide, job)
                       for job in jobs]
            decided = [future.result() for future in futures]
        found: set[str] = set()
        for (_, shared), pick in zip(jobs, decided, strict=True):
            for i in {*pick.targets, *(i for i, role in enumerate(pick.roles)
                                       if role in _PARTS)}:
                found.add(shared[i].memory_id)
        return found, sum(pick.calls for pick in decided)

    def _summaries(self, scope: ScopeV1) -> list[_Candidate]:
        """The scope's current memories of a summary-only origin not yet withdrawn."""
        writer = self._writer
        stored = writer.get_control(f"forget_withdrawn:{scope.scope_id}")
        withdrawn: frozenset[str] = frozenset() if stored is None else frozenset(
            ForgetWithdrawnV1.model_validate(stored["value"]).memory_ids)
        found: list[_Candidate] = []
        with writer.write_scope():
            read = readability(TxnRows(writer, scope.scope_kind, scope.scope_id, 0,
                                       self._clock(), ""))
            for item in writer.iter_rows("item"):
                if (item["scope_kind"], item["scope_id"]) != (
                        scope.scope_kind, scope.scope_id) or item["object_id"] in withdrawn:
                    continue
                head = writer.get_row("revision", item["current_revision_id"])
                if head is None or head["snapshot"].get("origin") not in SUMMARY_ONLY_ORIGINS:
                    continue
                snapshot = MemorySnapshotV2.model_validate(head["snapshot"])
                if read.p_mem(item["lifecycle_state"], snapshot):
                    found.append(_Candidate(item["object_id"], head["object_id"],
                                            head["content_hash"], snapshot.statement,
                                            tuple(snapshot.quote_spans)))
        return found

    def _withdraw(self, scope: ScopeV1, memory_ids: Set[str]) -> int:
        """Add memories to the scope's withdrawn set (``forget_withdrawn:{s}``); how many new."""
        writer = self._writer
        key = f"forget_withdrawn:{scope.scope_id}"
        with writer.write_scope():
            stored = writer.get_control(key)
            current: set[str] = set() if stored is None else set(
                ForgetWithdrawnV1.model_validate(stored["value"]).memory_ids)
            merged = sorted(current | memory_ids)[:MAX_FORGET_WITHDRAWN]
            if len(merged) == len(current):
                return 0
            control = writer.materialize_control(
                key, ForgetWithdrawnV1(memory_ids=merged).model_dump(mode="json"),
                0 if stored is None else stored["version"] + 1, self._clock())
            _, plan = writer.plan_rows(
                [RowWrite("control", control)], None)
            writer.write_plan(plan)
        return len(merged) - len(current)

    def _mark(self, scope: ScopeV1, spans: Sequence[QuoteSpanV1]) -> int:
        """Add sentence spans to the scope's forget marks (``forget_marks:{s}``); how many new.

        The marks name spans only; nothing of what they say is copied. A span the forgets
        later withhold or erase is still named: Search marks only what it delivers.
        """
        writer = self._writer
        key = f"forget_marks:{scope.scope_id}"
        with writer.write_scope():
            stored = writer.get_control(key)
            current = {} if stored is None else {
                (span.evidence_id, span.byte_start, span.byte_end): span
                for span in ForgetMarksV1.model_validate(stored["value"]).spans}
            merged = current | {(span.evidence_id, span.byte_start, span.byte_end): span
                                for span in spans}
            if len(merged) == len(current):
                return 0
            kept = [merged[position] for position in sorted(merged)][:MAX_FORGET_MARKS]
            control = writer.materialize_control(
                key, ForgetMarksV1(spans=kept).model_dump(mode="json"),
                0 if stored is None else stored["version"] + 1, self._clock())
            _, plan = writer.plan_rows(
                [RowWrite("control", control)], None)
            writer.write_plan(plan)
        return len(kept) - len(current)

    def _records_request(
        self, scope: ScopeV1, quotes: Sequence[QuoteSpanV1], *, complete: bool = False,
    ) -> bool:
        """Whether every quote of a memory is a forget request: it records the request.

        Such a memory ("The user asked to forget that ...") says what the user asked, not
        the fact; an in-chat forget keeps it, as the request sentence itself stays readable.
        """
        writer = self._writer
        texts: list[str] = []
        with writer.write_scope():
            for span in quotes:
                row = writer.get_row("evidence", span.evidence_id)
                if row is None or (row["scope_kind"], row["scope_id"]) != (
                        scope.scope_kind, scope.scope_id):
                    return False
                raw = row["text"].encode("utf-8")
                texts.append(raw[span.byte_start - row["byte_start"]:
                                 span.byte_end - row["byte_start"]].decode("utf-8", "replace"))
        return bool(texts) and all(is_request(text, complete=complete) for text in texts)

    def _sentences(self, scope: ScopeV1, statement: str, hit_ids: Sequence[str]) -> tuple[
            list[QuoteSpanV1], list[str]]:
        """The readable sentences of Evidence hits that share a content word with the statement.

        A sentence that is itself a forget request is the request's record and never a
        candidate. Returns the ``MAX_FORGET_CANDIDATES`` sharing the largest share of the
        statement's content words, each as its span (original-message byte coordinates) and
        its text.
        """
        wanted = content_words(statement)
        found: list[tuple[float, QuoteSpanV1, str]] = []
        writer = self._writer
        with writer.write_scope():
            rows = [writer.get_row("evidence", hit_id) for hit_id in dict.fromkeys(hit_ids)]
        for row in rows:
            if row is None or row["state"] != "active" or (row["scope_kind"], row["scope_id"]) != (
                    scope.scope_kind, scope.scope_id):
                continue
            text: str = row["text"]
            held = [(span["byte_start"], span["byte_end"]) for span in row["withheld_spans"]]
            for start, end in sentences(text):
                first = row["byte_start"] + len(text[:start].encode("utf-8"))
                last = row["byte_start"] + len(text[:end].encode("utf-8"))
                if any(first < high and low < last for low, high in held) or is_request(
                        text[start:end]):
                    continue
                shared = len(wanted & content_words(text[start:end]))
                if shared:
                    found.append((shared / max(1, len(wanted)), QuoteSpanV1(
                        evidence_id=row["object_id"], byte_start=first, byte_end=last),
                        text[start:end]))
        found.sort(key=lambda entry: -entry[0])
        kept = found[:MAX_FORGET_CANDIDATES]
        return [span for _, span, _ in kept], [text for _, _, text in kept]

    def _candidates(self, scope: ScopeV1, hit_ids: Sequence[str], *, limit: bool = True,
                    ) -> tuple[list[_Candidate], int]:
        """The hits whose memory is current (``p_mem``) in the scope (the first
        ``MAX_FORGET_CANDIDATES`` unless ``limit`` is False), and the scope's
        ``erase_epoch``."""
        writer = self._writer
        found: dict[str, _Candidate] = {}
        with writer.write_scope():
            read = readability(TxnRows(writer, scope.scope_kind, scope.scope_id, 0,
                                       self._clock(), ""))
            for hit_id in hit_ids:
                item = writer.get_row("item", hit_id)
                if item is None or hit_id in found or (item["scope_kind"], item["scope_id"]) != (
                        scope.scope_kind, scope.scope_id):
                    continue
                head = writer.get_row("revision", item["current_revision_id"])
                if head is None:
                    continue
                snapshot = MemorySnapshotV2.model_validate(head["snapshot"])
                if read.p_mem(item["lifecycle_state"], snapshot):
                    found[hit_id] = _Candidate(hit_id, head["object_id"], head["content_hash"],
                                               snapshot.statement, tuple(snapshot.quote_spans),
                                               tuple(snapshot.embedding),
                                               snapshot.embedding_space)
            erase_epoch = writer.get_counter(f"erase_epoch:{scope.scope_id}")
        rows = list(found.values())
        return (rows[:MAX_FORGET_CANDIDATES] if limit else rows), erase_epoch

    def _judge_binding(self) -> InferenceBindingV1:
        """The memory judge of the workspace's released extraction pipeline."""
        if self._judge is None:
            pointer = self._writer.get_control("policy:extraction")
            if pointer is None:
                raise QuoteMemError("service_unhealthy",
                                    "Forget by statement has no released policy")
            policy = read_released_extraction_config(
                self._writer, pointer["value"]["current_release_ref"])
            self._judge = memory_judge_binding(
                inference_binding_for_policy(policy), concurrency=MAX_JUDGE_CONCURRENCY,
                read_timeout_ms=extraction_pipeline(
                    policy.parameters.pipeline_version).judge_read_timeout_ms,
            )
        return self._judge

    def _sender(self, effort: ReasoningEffort | None = None) -> Send:
        """The pick's sender, at ``effort`` when given, else at the judge binding's own."""
        binding = self._judge_binding()
        if effort is not None:
            binding = binding.model_copy(update={"reasoning_effort": effort})
        return forget_pick_sender(cast(StructuredInferenceProvider, self._provider), binding,
                                  self._writer.bind_content)

    def _binding(
        self, scope: ScopeV1, grant: PrincipalGrant, client_session_id: str, request_id: str,
        targets: Sequence[ForgetStatementTargetV1],
        candidates: Sequence[ForgetStatementCandidateRefV1], erase_epoch: int, expires_at: int,
        *, spans: Sequence[QuoteSpanV1] = (), span_intent: str | None = None,
        bundled: Sequence[str] = (), statement: str | None = None,
    ) -> ContentBindingDigest:
        bind = self._writer.bind_content
        plan: dict[str, object] = {
            "scope": scope.scope_id, "principal": grant.principal_id,
            "session": bind("forget-session", {"scope": scope.scope_id,
                                               "session": client_session_id}),
            "request": bind("forget-statement-request", {"request": request_id}),
            "targets": [target.model_dump(mode="json") for target in targets],
            "candidates": [row.model_dump(mode="json") for row in candidates],
            "erase_epoch": erase_epoch, "expires_at": expires_at,
        }
        if spans:
            # A complete plan's sentences and their prepared forget.
            plan["spans"] = [span.model_dump(mode="json") for span in spans]
            plan["span_intent"] = span_intent
        if bundled:
            # A complete plan's bundled targets and the user's words they are forgotten for.
            plan["bundled"] = list(bundled)
            plan["statement"] = statement
        return bind("forget-statement-plan", plan)

    # Confirm ---------------------------------------------------------------------------
    def confirm(
        self, scope: ScopeV1, grant: PrincipalGrant, request: MemoryForgetStatementRequestV1,
    ) -> tuple[ForgetStatementOutcomeV1, ...]:
        """`confirm_report`'s outcomes."""
        return self.confirm_report(scope, grant, request).outcomes

    def confirm_report(
        self, scope: ScopeV1, grant: PrincipalGrant, request: MemoryForgetStatementRequestV1,
    ) -> StatementConfirmed:
        """Forget every target of the confirmed plan in order; resume on repeat.

        A complete forget (``agent`` mode) reports everything it changed: every
        memory readable before the confirmation and not after it, a candidate the plan kept in
        ``affected`` (the prepare listed it beforehand) and any other in ``forgotten``.

        Raises:
            QuoteMemError: ``operation_conflict`` for a plan this workspace did not prepare or a
                scope that changed before any target was forgotten; ``intent_expired``
                when a target is left after the expiry; the forget's own errors
                (``erasure_in_progress``, ``index_rebuilding``, ``budget_exceeded``…),
                after which a repeated confirmation resumes.
        """
        if self._binding(scope, grant, request.client_session_id, request.request_id,
                         request.targets, request.candidates, request.erase_epoch,
                         request.expires_at, spans=request.spans,
                         span_intent=request.span_intent, bundled=request.bundled,
                         statement=request.statement) != request.plan_binding:
            raise QuoteMemError("operation_conflict", "The statement plan was not prepared here")
        states = [self._state(scope, target)[0] for target in request.targets]
        writer = self._writer
        with writer.write_scope():
            carry = [span for span in request.spans if not self._forget_withheld(span)]
        if ("pending" in states or carry) and self._clock() >= request.expires_at:
            if request.span_intent is not None:
                # The prepared forget of the sentences keeps only its binding now.
                self._owner.expire_intent(request.span_intent)
            raise QuoteMemError("intent_expired", "The statement plan expired")
        complete = self._complete()
        record = self._planned(scope, request.plan_binding) if complete else None
        if record is not None and not record.held and ("pending" in states or carry):
            # A complete plan's record (what it named and what the judge confirmed) is gone,
            # evicted by newer plans: without it the confirmation would take less than the
            # prepare named, so it refuses and the user prepares again.
            raise QuoteMemError("intent_expired", "The statement plan is no longer held")
        if request.targets and all(state == "pending" for state in states):
            self._check_unchanged(scope, request)
        started = self._clock()
        # The shadow's names as earlier forgets left them (`_hidden`).
        prior = self.shadow_times(scope) if complete else {}
        cover = self._target_quotes(request.targets) if complete else []
        # A request for a modifier alone keeps the thing's other facts, as its prepare planned
        # (`_modifier`).
        modifier = record is not None and record.held and record.modifier
        planned = {target.memory_id for target in request.targets}
        # The statements of what the plan forgets, read before it is (`_leftovers`).
        told = [row.statement for row in self._rows(scope, sorted(planned))] if complete \
            else []
        # The memories the plan rewords instead of taking; the sentences it
        # withholds of one known only from them are the forgotten memories' quotes too.
        wordings = self._wordings(scope, request.plan_binding, record) \
            if record is not None and record.held and ("pending" in states or carry) else ()
        if wordings:
            quoted = [span for row in self._rows(scope, [
                wording.memory_id for wording in wordings]) for span in row.quotes]
            cover += [span for span in request.spans if any(
                span.evidence_id == quote.evidence_id and span.byte_start < quote.byte_end
                and quote.byte_start < span.byte_end for quote in quoted)]
        # The tie names too: the cascade takes only what the prepare named (``allowed``). A
        # memory the plan rewords still gives the names it gave (as `_affected` reads them).
        namers = [*(memory_id for memory_id in planned if memory_id not in request.bundled),
                  *(wording.memory_id for wording in wordings)]
        # A request for a nickname forgets the nickname alone, never the name (`_names`): the
        # request's words say so when the confirmation carries them (a bundle's), else what
        # its prepare recorded; neither saying, it is read as a request for the name, which
        # forgets more, never less.
        nick = nickname_request(request.statement) if request.statement is not None else (
            record is not None and record.held and record.nickname)
        found_names = self._names(scope, namers, ties=not nick, aliases=nick,
                                  every=not nick) if complete \
            else frozenset[tuple[str, str]]()
        # The name of an entity two people share goes only where the plan's targets about it
        # are told (``within``, as its prepare scoped it).
        planned_rows = self._rows(scope, sorted(planned)) if complete else []
        planned_heads = self._heads(scope, sorted(planned)) if complete else {}
        within: Within = {entity: self._sentence_scope(scope, [
            row for row in planned_rows if (head := planned_heads.get(row.memory_id))
            is not None and entity in {head.entity_id, *head.about_entity_ids,
                                       *(entry.entity_id for entry in head.naming)}])
            for entity in {entity for entity, _ in found_names} & self._ambiguous(scope)
        } if complete else {}
        names = self._forgotten_names(scope, frozenset(), found_names, uncertain=False,
                                      within=within) if complete \
            else frozenset[tuple[str, str]]()
        name_ties = self._tie_words(scope, namers) if names else {}
        kept = {row.memory_id for row in request.candidates} - planned
        before = {row.memory_id for row in self._memories(scope, past=True)} if complete \
            else set[str]()
        seed: ForgetSeedV1 | None = None
        if request.statement is not None and "pending" in states:
            # The user's words of the fact, embedded once, tombstone the bundled targets.
            # Without their vector a bundled target's own statement is tombstoned: safe,
            # though its unrelated fact is then checked as forgotten too.
            # The prepared vector is used once and dropped: no copy of the words outlives it.
            phrase = self._phrase_cache.pop((scope.scope_id, request.statement), None)
            try:
                if phrase is None:
                    phrase = self._phrase(probe(request.statement))[0]
            except QuoteMemError as error:
                if error.code not in _SEED_FALLBACK:
                    raise
                phrase = None
            seed = self._bundle_seed(request.statement, phrase)
        outcomes: list[ForgetStatementOutcomeV1] = []
        withheld = 0
        # What the forgets tombstone, read before they do: the shadow they add (`_hidden_by`).
        planned_heads = self._heads(scope, [target.memory_id for target in request.targets])
        reworded_heads = self._heads(scope, [wording.memory_id for wording in wordings])
        for target in request.targets:
            outcome = self._forget(
                scope, grant, request.client_session_id, target, restating=carry,
                seed=seed if target.memory_id in request.bundled else None,
                allowed=frozenset() if record is None else record.allowed,
                listed=None if record is None or not record.held else record.listed | planned,
                modifier=modifier, rewordings=wordings,
                asked=frozenset() if record is None else record.asked,
                name_asked=record is not None and record.name_asked)
            if outcome.outcome == "verified":
                # The first forgotten memory withholds the plan's sentences with its quotes.
                withheld += len(carry)
                carry = []
            outcomes.append(outcome)
        if carry:
            # No memory forget withheld the sentences: they are forgotten alone.
            outcomes.append(self._forget_spans(
                scope, grant, request.client_session_id, carry, intent=request.span_intent,
                seed=None if request.span_intent is not None else self._planned_seed(
                    request.targets), rewordings=wordings))
        elif request.spans and not request.targets:
            outcomes.append(ForgetStatementOutcomeV1(memory_id=None,
                                                     outcome="forgotten_earlier"))
        withheld += sum(row.withheld_spans or 0 for row in outcomes
                        if row.outcome == "verified")
        if not complete:
            return StatementConfirmed(tuple(outcomes))
        # A memory the forgets left unreadable that ties a name gives it too; the cascade takes
        # only what the prepare named (`_asked_names`, ``allowed``).
        lost = sorted(before - {row.memory_id for row in self._memories(scope, past=True)}
                      - planned)
        more_names = frozenset[tuple[str, str]]() if nick else self._names(scope, [
            memory_id for memory_id, snapshot in self._heads(scope, lost).items()
            if tied_name(snapshot.statement) is not None], ties=True)
        if more_names - names:
            names = names | self._forgotten_names(scope, frozenset(), more_names,
                                                  uncertain=False)
            name_ties = {**self._tie_words(scope, lost), **name_ties}
        self._cascade(scope, lambda: grant, request.client_session_id, cover,
                      allowed=(frozenset[str]() if record is None else record.listed) | planned,
                      names=names, ties=name_ties, told=told, within=within,
                      residue=record.residue if record is not None and record.held else None,
                      modifier=modifier,
                      asked=record is not None and record.held and record.name_or_all,
                      sentences=request.spans, rewordings=wordings)
        # What the prepare judged and kept stays found under the names this confirmation
        # stamped; a memory they hide all the same is reported (`_hidden`).
        stamped, recorded = self._record_kept(scope, started, plan=request.plan_binding,
                                              cover=[*cover, *request.spans])
        removed, affected = self._lost(scope, before, kept, self._hidden(
            scope, before, stamped, recorded, prior=prior, since=started), wordings)
        # A memory the plan's forget shadow hides from view is reported, never silent.
        reported = {row.memory_id for row in affected}
        left_forms = [form for wording in wordings
                      if (old := reworded_heads.get(wording.memory_id)) is not None
                      for form in left_out(old.statement, wording.statement)]
        hidden = self._hidden_by(
            scope, [(head.statement, head.keywords) for head in planned_heads.values()],
            asked=frozenset() if record is None else record.asked, left=left_forms,
            seed=request.statement if request.bundled else None,
            skip={*kept, *planned, *reported, *removed,
                  *(wording.memory_id for wording in wordings)})
        # (The memories gone already are no longer current: `_memories` reads none of them.)
        affected += [ForgetStatementAffectedV1(memory_id=row.memory_id, effect="unreadable")
                     for row in hidden if row.memory_id in before]
        entry = self._entry(scope, asked=request.request_id)
        if entry is not None:
            # The statement stays read up to the prepare's commit: what another Add wrote
            # between the prepare and this confirmation is left to the in-chat forget.
            self._record_handled(
                scope, entry.request, statement="forgotten",
                forgotten=len(removed), withheld=withheld)
        return StatementConfirmed(tuple(outcomes), len(removed), withheld, tuple(affected))

    def _lost(
        self, scope: ScopeV1, before: Set[str], kept: Set[str], hidden: Set[str] = frozenset(),
        wordings: Sequence[ForgetRewordV1] = (),
    ) -> tuple[set[str], list[ForgetStatementAffectedV1]]:
        """What a complete forget did to the memories readable ``before`` it.

        Returns the memories no longer readable that the forget's plan did not keep, each a
        removal of the fact (a target, a memory the cascade forgot, or one whose only support
        stated the fact and is withheld now), and the kept ones it changed all the same, each
        ``forgotten`` when its memory is gone or erased and ``unreadable`` otherwise; a memory
        still readable that the names the forget stamped now hide (``hidden``, `_hidden`) is
        ``unreadable`` too, whether the plan read it or not; and each memory it reworded
        (``wordings``: readable now under its new statement) ``reworded``; a memory the
        prepare named as reworded that is lost all the same counts as kept.
        """
        current = {row.memory_id: row for row in self._memories(scope)}
        lost = set(before) - {row.memory_id for row in self._memories(scope, past=True)}
        # A memory the prepare named as reworded is reported as such or with what befell it.
        kept = {*kept, *(wording.memory_id for wording in wordings)}
        affected: list[ForgetStatementAffectedV1] = []
        writer = self._writer
        with writer.write_scope():
            for memory_id in sorted(lost & kept):
                item = writer.get_row("item", memory_id)
                affected.append(ForgetStatementAffectedV1(
                    memory_id=memory_id,
                    effect="unreadable" if item is not None
                    and item["lifecycle_state"] == "active" else "forgotten"))
        affected += [ForgetStatementAffectedV1(memory_id=memory_id, effect="unreadable")
                     for memory_id in sorted(set(hidden) - lost)]
        affected += [ForgetStatementAffectedV1(memory_id=wording.memory_id, effect="reworded")
                     for wording in sorted(wordings, key=lambda wording: wording.memory_id)
                     if wording.memory_id in before and wording.memory_id in current
                     and current[wording.memory_id].statement == wording.statement
                     and wording.base_digest != statement_digest(wording.statement)]
        return lost - set(kept), affected

    def _wordings(self, scope: ScopeV1, plan: ContentBindingDigest,
                  record: _PlanRecord) -> tuple[ForgetRewordV1, ...]:
        """The rewordings a plan's prepare recorded (`_remember_kept`), with the vectors it
        kept for this confirmation, or embedded again when this process holds none."""
        cached = self._wording_cache.pop((scope.scope_id, str(plan)), ())
        wanted = {wording.memory_id for wording in record.rewordings}
        if cached and {wording.memory_id for wording in cached} == wanted:
            return cached
        found, _ = self._embedded(scope, {wording.memory_id: wording
                                          for wording in record.rewordings})
        return tuple(found.values())

    def _planned_seed(self, targets: Sequence[ForgetStatementTargetV1]) -> ForgetSeedV1 | None:
        """A plan's sentences' tombstone when none of its memories withheld them: the first
        planned head's statement and vector, a statement of the fact the plan bound."""
        writer = self._writer
        with writer.write_scope():
            for target in targets:
                head = writer.get_row("revision", target.expected_revision_id)
                if head is None:
                    continue
                snapshot = MemorySnapshotV2.model_validate(head["snapshot"])
                if snapshot.embedding_space and any(snapshot.embedding):
                    return ForgetSeedV1(statement=canonical_statement(snapshot.statement),
                                        embedding=list(snapshot.embedding),
                                        embedding_space=snapshot.embedding_space)
        return None

    def _target_quotes(self, targets: Sequence[ForgetStatementTargetV1]) -> list[QuoteSpanV1]:
        """The quotes of the targets' current heads, read before they are forgotten."""
        found: list[QuoteSpanV1] = []
        writer = self._writer
        with writer.write_scope():
            for target in targets:
                item = writer.get_row("item", target.memory_id)
                head = None if item is None else writer.get_row(
                    "revision", item["current_revision_id"])
                if head is not None:
                    found += MemorySnapshotV2.model_validate(head["snapshot"]).quote_spans
        return found

    def _forget(
        self, scope: ScopeV1, grant: PrincipalGrant, session_id: str,
        target: ForgetStatementTargetV1, *, restating: Sequence[QuoteSpanV1] = (),
        seed: ForgetSeedV1 | None = None, enforces: int | None = None,
        allowed: Set[str] = frozenset(), listed: Set[str] | None = None,
        support_only: bool = False, modifier: bool = False,
        rewordings: Sequence[ForgetRewordV1] = (),
        asked: Set[ContentBindingDigest] = frozenset(), name_asked: bool = False,
    ) -> ForgetStatementOutcomeV1:
        """One target's forget; ``restating`` sentences join its S, a bundled target's
        ``seed`` is its tombstone, and ``enforces`` is the admission time of the forget a
        relearn check enforces on it (`ForgetOwner.prepare`: the forget shadow's stamps then
        stay where that forget put them). A complete forget takes beyond the
        base links only what the judge confirmed (``allowed``) and, with ``listed``, refuses
        to take a memory its prepare did not name; with ``support_only`` (the cascade's) it
        takes no replacer; with ``modifier`` the user asked to forget only a modifier of a thing
        of theirs (``asks_modifier``, `ForgetOwner.prepare`'s ``modifier_only``). It rewords the
        ``rewordings``' memories it settles."""
        state, head, content_hash = self._state(scope, target)
        if state != "pending":
            return ForgetStatementOutcomeV1(
                memory_id=target.memory_id,
                outcome="forgotten_earlier" if state == "gone" else "changed")
        try:
            prepared = self._owner.prepare(
                scope, grant, session_id=session_id, memory_id=target.memory_id,
                expected_revision_id=head, restating=restating, seed=seed,
                enforces=enforces, allowed=sorted(allowed)[:MAX_FORGET_KEPT_LISTED],
                listed=None if listed is None else sorted(listed), support_only=support_only,
                modifier_only=modifier,
                rewordings=[wording for wording in rewordings
                            if wording.memory_id != target.memory_id], asked=sorted(asked),
                name_asked=name_asked)
            done = self._owner.forget(
                scope, grant, session_id=session_id, memory_id=target.memory_id,
                expected_revision_id=head, expected_content_hash=content_hash,
                intent=prepared.intent_key)
        except QuoteMemError as error:
            if error.code not in ("operation_conflict", "scope_denied"):
                raise
            state = self._state(scope, target)[0]
            return ForgetStatementOutcomeV1(
                memory_id=target.memory_id,
                outcome="forgotten_earlier" if state == "gone" else "changed")
        return ForgetStatementOutcomeV1(
            memory_id=target.memory_id, outcome="verified",
            controller_run_id=done.controller_run_id, receipt_ref=ArtifactId(done.receipt_ref),
            receipt=done.receipt)

    def _state(
        self, scope: ScopeV1, target: ForgetStatementTargetV1,
    ) -> tuple[TargetState, str, str]:
        """A target's state, with the head and content hash its forget binds when pending.

        An earlier target's forget may requote this one on its readable support or drop
        its keywords: the statement the plan bound is unchanged, so the
        target is pending at its new head (``requoted_from``).
        """
        writer = self._writer
        with writer.write_scope():
            item = writer.get_row("item", target.memory_id)
            if item is None or (item["scope_kind"], item["scope_id"]) != (
                    scope.scope_kind, scope.scope_id):
                return "gone", "", ""
            head = writer.get_row("revision", item["current_revision_id"])
            planned = head
            while planned is not None and planned["object_id"] != target.expected_revision_id:
                previous = (None if planned["previous_revision_id"] is None
                            else writer.get_row("revision", planned["previous_revision_id"]))
                planned = (previous if previous is not None and requoted_from(planned, previous)
                           else None)
        if head is None or planned is None or (
                planned["content_hash"] != target.expected_content_hash):
            return "changed", "", ""
        return "pending", head["object_id"], head["content_hash"]

    def _check_unchanged(self, scope: ScopeV1, request: MemoryForgetStatementRequestV1) -> None:
        """Before any target is forgotten, the scope must read as the plan read it."""
        writer = self._writer
        # A target a complete prepare read past (a promoted one, ``_promoted``) need not be
        # current: its own forget checks its head as every target's does. Any other memory
        # the plan read, and in every other mode every one, must still be current.
        past = (self._planned(scope, request.plan_binding).past if self._complete()
                else frozenset[str]())
        with writer.write_scope():
            if writer.get_counter(f"erase_epoch:{scope.scope_id}") != request.erase_epoch:
                raise QuoteMemError("operation_conflict", "The scope changed since the plan")
            for row in request.candidates:
                item = writer.get_row("item", row.memory_id)
                if item is None or item["current_revision_id"] != row.revision_id or (
                        item["lifecycle_state"] != "active" and row.memory_id not in past):
                    raise QuoteMemError("operation_conflict", "The scope changed since the plan")


def _target(message: str) -> str:
    """The target a vote's user message asks about (`forget_pick._message`)."""
    try:
        return str(cast(dict[str, object], json.loads(message))["target"])
    except (KeyError, TypeError, ValueError):
        raise QuoteMemError("binding_mismatch", "A forget-pick vote lost its target") from None


def _decision(pick: ForgetPickV1, statement: str) -> ForgetStatementDecisionV1:
    """The pick's votes, with the prompt its classify votes of ``statement`` sent."""
    _, version, digest = forget_prompt("classify", statement)
    return ForgetStatementDecisionV1(
        prompt_version=cast(PickPromptVersion, version),
        prompt_digest=digest, method=pick.method, calls=pick.calls,
        answered=sum(stage.answered for stage in pick.stages), fallbacks=pick.fallbacks,
        votes=[ForgetStatementVoteV1(
            stage=stage.stage, key=decision.key, votes=dict(decision.votes),
            decided=decision.decided, fallback=decision.fallback,
        ) for stage in pick.stages for decision in stage.decisions],
    )


def forget_pick_sender(
    provider: StructuredInferenceProvider, binding: InferenceBindingV1,
    bind: Callable[[str, object], ContentBindingDigest], *, thinking: bool = False,
    timeout_ms: int = FORGET_PICK_TIMEOUT_MS, hedge_seconds: float | None = None,
) -> Send:
    """Send forget-pick votes through the judge's provider under its binding.

    Each vote is one ``memory-forget-pick-v1`` request with the judge binding's
    provider, model and reasoning effort, the timeout and retries, and the pinned
    prompt. A call that ends ``provider_unavailable`` or ``schema_invalid`` is an
    unanswered vote; any other error propagates.

    The ``agent`` mode's forgets send their votes without model reasoning
    (``thinking``, when the binding's provider takes it), under a shorter ``timeout_ms``,
    and hedged: a vote still unanswered after ``hedge_seconds`` is sent once more with the
    same message, and the first valid reply is the vote's. A hedge never changes what a
    vote says, only how long its slowest call holds the round.

    Args:
        provider: The structured inference provider.
        binding: The memory judge's binding.
        bind: The keyed content binding for the request digests.
        thinking: Send each vote with its reasoning disabled.
        timeout_ms: Each vote's call timeout.
        hedge_seconds: When a vote is sent again; None never.

    Returns:
        Send: The sender `pick_targets` takes.
    """

    def send(stage: Stage, message: str) -> ForgetPickOutput | None:
        schema = {"classify": FORGET_PICK_CLASSIFY_SCHEMA_ID,
                  "remainders": FORGET_PICK_REMAINDERS_SCHEMA_ID,
                  "tells": FORGET_PICK_TELLS_SCHEMA_ID}[stage]
        # A complete forget's ``tells`` votes send their own prompt, and a detail request's
        # votes theirs; the pick's stays pinned for every other target.
        prompt, version, digest = forget_prompt(stage, _target(message))
        request = InferenceRequest(
            purpose="judge", provider=cast(Literal["openai"], binding.provider),
            requested_model=binding.requested_model, schema_id=schema,
            prompt_version=version,
            input_digest=bind("forget-pick-input", {"message": message}),
            context_digest=bind("forget-pick-context", {
                "schema": schema, "prompt": digest}),
            messages=[Message(role="system", content=prompt),
                      Message(role="user", content=message)],
            timeout_ms=timeout_ms,
            retry_policy=binding.retry_policy.model_copy(update={
                "max_read_timeouts": FORGET_PICK_MAX_READ_TIMEOUTS,
            }),
            reasoning_effort=binding.reasoning_effort,
            thinking="disabled" if thinking else None,
        )
        if hashlib.sha256(request.messages[0].content.encode("utf-8")).hexdigest() != digest:
            raise QuoteMemError("binding_mismatch", "The forget pick changed its pinned prompt")
        try:
            output = provider.infer(request).output
        except QuoteMemError as error:
            if error.code not in _VOTE_ERRORS:
                raise
            return None
        if isinstance(output, ForgetPickClassifyV1 | ForgetPickRemaindersV1 | ForgetPickTellsV1):
            return output
        return None

    if hedge_seconds is None:
        return send

    def hedged(stage: Stage, message: str) -> ForgetPickOutput | None:
        pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="quotemem-forget-hedge")
        try:
            first = pool.submit(contextvars.copy_context().run, send, stage, message)
            done, _ = wait([first], timeout=hedge_seconds)
            if done:
                return first.result()
            pending = {first, pool.submit(contextvars.copy_context().run, send, stage,
                                          message)}
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    if (reply := future.result()) is not None:
                        return reply
            return None
        finally:
            # A slower duplicate is left to finish on its own; its reply is not read.
            pool.shutdown(wait=False, cancel_futures=True)

    return hedged
