"""Forget by statement (FS-1) on the single Writer: Search, the judge's pick, D.3 per target.

Extraction can store one statement as several memories, so forgetting one memory id
can leave the rest of the statement current. ``StatementForget`` forgets a statement:

``prepare``
    1. Searches the statement in the user's scope with the runtime's own Search, at
       ``top_k`` = ``MAX_FORGET_CANDIDATES``, exactly as a client Search runs.
    2. Its candidates are the memory hits whose Item is current (P-MEM) in the scope,
       in rank order, each read by its current statement under the Writer mutex.
    3. The product's judge picks the targets (`aml.core.forget_pick`, prompt
       ``memory-forget-pick-v1``); every vote is a call outside every lock.
    4. The plan names each target's exact head, each candidate's head, the scope's
       ``erase_epoch`` and an expiry ``INTENT_TTL_MS`` away, under the workspace's keyed
       ``forget-statement-plan`` binding. Preparing writes nothing; the Search records
       its own delivery row, as every Search does.

``confirm``
    Checks the binding, then forgets the targets in plan order, each through the D.3
    forget (``ForgetOwner`` P0, then P1–P4) bound to the plan's head. The first
    confirmation also requires the scope unchanged since the plan: the same
    ``erase_epoch`` and every candidate at its planned head, else ``operation_conflict``
    (prepare again). Each target's forget is atomic; the statement is not. A repeated
    confirmation resumes: a target already gone is ``forgotten_earlier``, one whose
    head moved is ``changed`` and is left alone, the rest are forgotten. A head moved
    only by privacy requotes stating the same thing (an earlier target's D.7.3
    compensation: readable support, keywords) still states what the plan bound: that
    target is forgotten at its new head. Expiry refuses
    a confirmation that still has a target to forget (``intent_expired``); one with
    every target gone answers again. Forget itself calls no provider (D.3).
"""

from __future__ import annotations

import contextvars
import hashlib
import time
from collections.abc import Callable, Iterator, Sequence, Set
from concurrent.futures import ThreadPoolExecutor
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from typing import Literal, cast

from aml.api.bench import SearchRequest, SearchResult
from aml.api.bench_service import SearchOperation
from aml.api.memory_forget import (
    ForgetStatementCandidateRefV1,
    ForgetStatementCandidateV1,
    ForgetStatementDecisionV1,
    ForgetStatementOutcomeV1,
    ForgetStatementTargetV1,
    ForgetStatementVoteV1,
    MemoryForgetStatementPreparedV1,
    MemoryForgetStatementPrepareRequestV1,
    MemoryForgetStatementRequestV1,
)
from aml.core.controls import (
    MAX_FORGET_MARKS,
    MAX_FORGET_WITHDRAWN,
    ForgetMarksV1,
    ForgetWithdrawnV1,
)
from aml.core.entities import MemorySnapshotV2, QuoteSpanV1
from aml.core.errors import AMLError
from aml.core.extraction_policy import extraction_pipeline
from aml.core.forget_pick import (
    FORGET_PICK_CLASSIFY_SCHEMA_ID,
    FORGET_PICK_MAX_READ_TIMEOUTS,
    FORGET_PICK_PROMPT,
    FORGET_PICK_PROMPT_DIGEST,
    FORGET_PICK_PROMPT_VERSION,
    FORGET_PICK_REMAINDERS_SCHEMA_ID,
    FORGET_PICK_TIMEOUT_MS,
    MAX_FORGET_CANDIDATES,
    ForgetPickClassifyV1,
    ForgetPickOutput,
    ForgetPickRemaindersV1,
    ForgetPickV1,
    Role,
    Send,
    Stage,
    pick_targets,
)
from aml.core.forget_state import INTENT_TTL_MS
from aml.core.identity import ArtifactId, ContentBindingDigest, IntegrityDigest
from aml.core.inference import MAX_JUDGE_CONCURRENCY, InferenceBindingV1, ReasoningEffort
from aml.core.query import ScopeV1
from aml.core.recall import SUMMARY_ONLY_ORIGINS
from aml.core.retained_policy import read_released_extraction_config
from aml.ingestion.forget_requests import content_words, is_request, sentence_ranges
from aml.providers.auth import PrincipalGrant
from aml.providers.bindings import inference_binding_for_policy, memory_judge_binding
from aml.providers.contracts import InferenceRequest, Message, StructuredInferenceProvider
from aml.store.entity_ops import TxnRows, requoted_from
from aml.store.forget_run import ForgetOwner
from aml.store.memory_commit import readability
from aml.store.writer import RowWrite, Writer

type TargetState = Literal["pending", "gone", "changed"]

SEARCH_SECONDS = 120.0
"""The statement's own Search budget, queue time included."""
CHAT_SEARCH_TOP_K = 30
"""An in-chat request's Search depth: its memory and sentence candidates both come from it."""
_PARTS: frozenset[Role] = frozenset({"same_fact", "part"})
"""The roles an in-chat request targets (`StatementForget.chat_forget`): the request names
the fact in the user's own words, so every stored piece of it goes, not only the minimal set
an exact statement forget takes; a candidate that only identifies an entity stays."""
CHAT_FORGET_REASONING_EFFORT: ReasoningEffort = "low"
"""The in-chat forget pick's judge effort.

At the pipeline's effort, the picks' reasoning length dominated the time of an Add that
holds a forget request. In replays of recorded Adds, ``low`` about halved the forget's
time, and its picks agreed with the pipeline effort's about as often as the pipeline
effort agreed with itself. Batching the lists into one call cut calls but made each
vote's reasoning longer. The FS-1 prepare path keeps the pipeline's effort."""
_VOTE_ERRORS = frozenset({"provider_unavailable", "schema_invalid"})
"""Call failures that leave one vote unanswered; any other error propagates."""


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
            raise AMLError("service_unhealthy", "Search delivered more than one response")
        self.result = result


@dataclass(frozen=True)
class _Candidate:
    memory_id: str
    revision_id: str
    content_hash: str
    statement: str
    quotes: tuple[QuoteSpanV1, ...]


@dataclass(frozen=True)
class ChatForgetOutcome:
    """What one Add's in-chat forget requests did (CHAT-FORGET, FORGET-TARGET).

    Attributes:
        forgotten: The memories forgotten now through D.3 (paraphrases included).
        changed: Picked memories whose head moved or that were gone before their forget.
        marked: Sentences newly marked as stating a fact the user asked to forget.
        withdrawn: Summary-only memories newly withdrawn from every read.
        calls: The judge calls the picks made.
    """

    forgotten: int
    changed: int
    marked: int
    withdrawn: int
    calls: int


def _sharing(statement: str, rows: Sequence[_Candidate]) -> list[_Candidate]:
    """The rows whose statement shares a content word with the statement, the
    ``MAX_FORGET_CANDIDATES`` sharing the largest share of its content words."""
    wanted = content_words(statement)
    shares = [(len(wanted & content_words(row.statement)) / max(1, len(wanted)), row)
              for row in rows]
    ranked = sorted((entry for entry in shares if entry[0] > 0), key=lambda entry: -entry[0])
    return [row for _, row in ranked[:MAX_FORGET_CANDIDATES]]


def _rounds(calls: Sequence[Callable[[], ForgetPickOutput | None]]) -> list[
        ForgetPickOutput | None]:
    """Run one round's votes concurrently, each in a copy of the caller's context."""
    with ThreadPoolExecutor(max_workers=len(calls), thread_name_prefix="aml-forget-pick") as pool:
        futures = [pool.submit(contextvars.copy_context().run, call) for call in calls]
        return [future.result() for future in futures]


class StatementForget:
    """Prepare and confirm forget by statement for one workspace (FS-1).

    Args:
        writer: The single Writer.
        owner: The workspace's D.3 forget owner; every target goes through it.
        search: The runtime's Search; None refuses ``prepare`` (``service_unhealthy``).
        provider: The workspace's structured inference provider (the judge's).
        clock: The trusted millisecond clock.
        judge: The judge binding whose provider, model, effort, timeout and retries
            the pick's calls use; None reads the workspace's released extraction policy
            (C.5's memory judge of that pipeline).
    """

    def __init__(
        self, writer: Writer, owner: ForgetOwner, *, search: SearchOperation | None,
        provider: StructuredInferenceProvider | None, clock: Callable[[], int],
        judge: InferenceBindingV1 | None = None,
    ) -> None:
        self._writer, self._owner, self._search = writer, owner, search
        self._provider, self._clock, self._judge = provider, clock, judge

    # Prepare ---------------------------------------------------------------------------
    def prepare(
        self, scope: ScopeV1, grant: PrincipalGrant,
        request: MemoryForgetStatementPrepareRequestV1,
    ) -> MemoryForgetStatementPreparedV1:
        """Search, pick and return the plan to confirm, or ``nothing_matched``.

        Raises:
            AMLError: Search's own errors (``index_rebuilding`` under a fence,
                ``delivery_busy``); ``provider_unavailable`` when a pick decision got
                fewer than two answered votes; ``service_unhealthy`` without a
                Search or provider.
        """
        if self._search is None or self._provider is None:
            raise AMLError("service_unhealthy", "Forget by statement has no Search or judge")
        delivery = _Delivery(SEARCH_SECONDS)
        self._search(SearchRequest(query=request.statement, user_id=request.user_id,
                                   top_k=MAX_FORGET_CANDIDATES), delivery)
        if delivery.result is None or any(hit.scope_id != scope.scope_id
                                          for hit in delivery.result.hits):
            raise AMLError("service_unhealthy", "The statement's Search answered no result")
        candidates, erase_epoch = self._candidates(scope, [hit.id for hit in
                                                           delivery.result.hits])
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
        decision = _decision(pick)
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

    # In-chat requests ---------------------------------------------------------------
    def chat_forget(
        self, scope: ScopeV1, authorize: Callable[[], PrincipalGrant], *, user_id: str,
        session_id: str, statements: Sequence[str],
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
        memory target, a paraphrase as much as the matched statement, goes through D.3 one
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
        whole session, so D.3 would withhold every span it quotes, and instead no read
        delivers it again. The request's words find only the summaries worded like it, and
        its memory targets come from Search: a memory the judge takes as stating part of the
        request ("a playlist full of Spanish guitar" for "I enjoy a wide variety of music
        genres") goes, while an overview telling the same detail shares no word with the
        request or does not state the request's own claim. So every summary-only memory
        sharing a content word with a memory target's statement is judged against that
        statement too, and one that states it or part of it is withdrawn
        (`StatementForget._restating`). Every pick precedes every forget, so a judge failure
        leaves nothing done and the retried run judges them all again.

        Raises:
            AMLError: ``service_unhealthy`` without a Search or provider;
                ``provider_unavailable`` when a pick decision got fewer than two answered
                votes; the D.3 forget's own errors.
        """
        if self._search is None or self._provider is None:
            raise AMLError("service_unhealthy", "Forget by statement has no Search or judge")
        picks: list[tuple[str, list[_Candidate], list[QuoteSpanV1], list[str],
                          list[_Candidate]]] = []
        summaries = [row for row in self._summaries(scope)
                     if not self._records_request(scope, row.quotes)]
        for statement in statements:
            delivery = _Delivery(SEARCH_SECONDS)
            self._search(SearchRequest(query=statement, user_id=user_id,
                                       top_k=CHAT_SEARCH_TOP_K), delivery)
            if delivery.result is None or any(hit.scope_id != scope.scope_id
                                              for hit in delivery.result.hits):
                raise AMLError("service_unhealthy", "The statement's Search answered no result")
            hit_ids = [hit.id for hit in delivery.result.hits]
            candidates = [row for row in self._candidates(scope, hit_ids)[0]
                          if not self._records_request(scope, row.quotes)]
            spans, texts = self._sentences(scope, statement, hit_ids)
            picks.append((statement, candidates, spans, texts, _sharing(statement, summaries)))
        sender = self._sender(CHAT_FORGET_REASONING_EFFORT)
        jobs = [(index, kind, statement, texts)
                for index, (statement, candidates, _, sentences, shared) in enumerate(picks)
                for kind, texts in (("memory", [row.statement for row in candidates]),
                                    ("sentence", sentences),
                                    ("summary", [row.statement for row in shared])) if texts]

        def decide(job: tuple[int, str, str, list[str]]) -> tuple[int, str, ForgetPickV1]:
            return job[0], job[1], pick_targets(job[2], job[3], sender, rounds=_rounds)

        with ThreadPoolExecutor(max_workers=max(1, min(len(jobs), MAX_JUDGE_CONCURRENCY)),
                                thread_name_prefix="aml-chat-forget") as pool:
            futures = [pool.submit(contextvars.copy_context().run, decide, job) for job in jobs]
            decided = [future.result() for future in futures]
        targets: dict[str, _Candidate] = {}
        marked: dict[tuple[str, int, int], QuoteSpanV1] = {}
        withdrawn: set[str] = set()
        calls = 0
        for index, kind, pick in decided:
            calls += pick.calls
            chosen = {*pick.targets, *(i for i, role in enumerate(pick.roles) if role in _PARTS)}
            for i in sorted(chosen):
                if kind == "memory":
                    row = picks[index][1][i]
                    targets.setdefault(row.memory_id, row)
                elif kind == "summary":
                    withdrawn.add(picks[index][4][i].memory_id)
                else:
                    span = picks[index][2][i]
                    marked[span.evidence_id, span.byte_start, span.byte_end] = span
        restating, restating_calls = self._restating(
            [row.statement for row in targets.values()],
            [row for row in summaries if row.memory_id not in withdrawn], sender)
        withdrawn |= restating
        calls += restating_calls
        forgotten = changed = 0
        for row in targets.values():
            target = ForgetStatementTargetV1(
                memory_id=row.memory_id, expected_revision_id=row.revision_id,
                expected_content_hash=IntegrityDigest(row.content_hash))
            if self._forget(scope, authorize(), session_id, target).outcome == "verified":
                forgotten += 1
            else:
                changed += 1
        added = self._mark(scope, list(marked.values()))
        return ChatForgetOutcome(forgotten=forgotten, changed=changed, marked=added,
                                 withdrawn=self._withdraw(scope, withdrawn), calls=calls)

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
                                thread_name_prefix="aml-chat-forget") as pool:
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
        with writer._write_scope():  # pyright: ignore[reportPrivateUsage]
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
        with writer._write_scope():  # pyright: ignore[reportPrivateUsage]
            stored = writer.get_control(key)
            current: set[str] = set() if stored is None else set(
                ForgetWithdrawnV1.model_validate(stored["value"]).memory_ids)
            merged = sorted(current | memory_ids)[:MAX_FORGET_WITHDRAWN]
            if len(merged) == len(current):
                return 0
            control = writer.materialize_control(
                key, ForgetWithdrawnV1(memory_ids=merged).model_dump(mode="json"),
                0 if stored is None else stored["version"] + 1, self._clock())
            _, plan = writer._prepare_rows(  # pyright: ignore[reportPrivateUsage]
                [RowWrite("control", control)], None)
            writer._write_plan(plan)  # pyright: ignore[reportPrivateUsage]
        return len(merged) - len(current)

    def _mark(self, scope: ScopeV1, spans: Sequence[QuoteSpanV1]) -> int:
        """Add sentence spans to the scope's forget marks (``forget_marks:{s}``); how many new.

        The marks name spans only; nothing of what they say is copied. A span the forgets
        later withhold or erase is still named: Search marks only what it delivers.
        """
        writer = self._writer
        key = f"forget_marks:{scope.scope_id}"
        with writer._write_scope():  # pyright: ignore[reportPrivateUsage]
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
            _, plan = writer._prepare_rows(  # pyright: ignore[reportPrivateUsage]
                [RowWrite("control", control)], None)
            writer._write_plan(plan)  # pyright: ignore[reportPrivateUsage]
        return len(kept) - len(current)

    def _records_request(self, scope: ScopeV1, quotes: Sequence[QuoteSpanV1]) -> bool:
        """Whether every quote of a memory is a forget request: it records the request.

        Such a memory ("The user asked to forget that ...") says what the user asked, not
        the fact; an in-chat forget keeps it, as the request sentence itself stays readable.
        """
        writer = self._writer
        texts: list[str] = []
        with writer._write_scope():  # pyright: ignore[reportPrivateUsage]
            for span in quotes:
                row = writer.get_row("evidence", span.evidence_id)
                if row is None or (row["scope_kind"], row["scope_id"]) != (
                        scope.scope_kind, scope.scope_id):
                    return False
                raw = row["text"].encode("utf-8")
                texts.append(raw[span.byte_start - row["byte_start"]:
                                 span.byte_end - row["byte_start"]].decode("utf-8", "replace"))
        return bool(texts) and all(is_request(text) for text in texts)

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
        with writer._write_scope():  # pyright: ignore[reportPrivateUsage]
            rows = [writer.get_row("evidence", hit_id) for hit_id in dict.fromkeys(hit_ids)]
        for row in rows:
            if row is None or row["state"] != "active" or (row["scope_kind"], row["scope_id"]) != (
                    scope.scope_kind, scope.scope_id):
                continue
            text: str = row["text"]
            held = [(span["byte_start"], span["byte_end"]) for span in row["withheld_spans"]]
            for start, end in sentence_ranges(text):
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

    def _candidates(self, scope: ScopeV1, hit_ids: Sequence[str]) -> tuple[
            list[_Candidate], int]:
        """The hits whose Item is current (P-MEM) in the scope, and the scope's ``erase_epoch``."""
        writer = self._writer
        found: dict[str, _Candidate] = {}
        with writer._write_scope():  # pyright: ignore[reportPrivateUsage]
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
                                               snapshot.statement, tuple(snapshot.quote_spans))
            erase_epoch = writer.counter(f"erase_epoch:{scope.scope_id}")
        return list(found.values())[:MAX_FORGET_CANDIDATES], erase_epoch

    def _judge_binding(self) -> InferenceBindingV1:
        """C.5's memory judge of the workspace's released extraction pipeline."""
        if self._judge is None:
            pointer = self._writer.get_control("policy:extraction")
            if pointer is None:
                raise AMLError("service_unhealthy", "Forget by statement has no released policy")
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
    ) -> ContentBindingDigest:
        bind = self._writer.bind_content
        return bind("forget-statement-plan", {
            "scope": scope.scope_id, "principal": grant.principal_id,
            "session": bind("forget-session", {"scope": scope.scope_id,
                                               "session": client_session_id}),
            "request": bind("forget-statement-request", {"request": request_id}),
            "targets": [target.model_dump(mode="json") for target in targets],
            "candidates": [row.model_dump(mode="json") for row in candidates],
            "erase_epoch": erase_epoch, "expires_at": expires_at,
        })

    # Confirm ---------------------------------------------------------------------------
    def confirm(
        self, scope: ScopeV1, grant: PrincipalGrant, request: MemoryForgetStatementRequestV1,
    ) -> tuple[ForgetStatementOutcomeV1, ...]:
        """Forget every target of the confirmed plan in order (D.3 each); resume on repeat.

        Raises:
            AMLError: ``operation_conflict`` for a plan this workspace did not prepare or a
                scope that changed before any target was forgotten; ``intent_expired``
                when a target is left after the expiry; the D.3 forget's own errors
                (``erasure_in_progress``, ``index_rebuilding``, ``budget_exceeded``…),
                after which a repeated confirmation resumes.
        """
        if self._binding(scope, grant, request.client_session_id, request.request_id,
                         request.targets, request.candidates, request.erase_epoch,
                         request.expires_at) != request.plan_binding:
            raise AMLError("operation_conflict", "The statement plan was not prepared here")
        states = [self._state(scope, target)[0] for target in request.targets]
        if "pending" in states and self._clock() >= request.expires_at:
            raise AMLError("intent_expired", "The statement plan expired")
        if all(state == "pending" for state in states):
            self._check_unchanged(scope, request)
        outcomes: list[ForgetStatementOutcomeV1] = []
        for target in request.targets:
            outcomes.append(self._forget(scope, grant, request.client_session_id, target))
        return tuple(outcomes)

    def _forget(
        self, scope: ScopeV1, grant: PrincipalGrant, session_id: str,
        target: ForgetStatementTargetV1,
    ) -> ForgetStatementOutcomeV1:
        state, head, content_hash = self._state(scope, target)
        if state != "pending":
            return ForgetStatementOutcomeV1(
                memory_id=target.memory_id,
                outcome="forgotten_earlier" if state == "gone" else "changed")
        try:
            prepared = self._owner.prepare(
                scope, grant, session_id=session_id, memory_id=target.memory_id,
                expected_revision_id=head)
            done = self._owner.forget(
                scope, grant, session_id=session_id, memory_id=target.memory_id,
                expected_revision_id=head, expected_content_hash=content_hash,
                intent=prepared.intent_key)
        except AMLError as error:
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
        its keywords (D.7.3 steps 2-4b): the statement the plan bound is unchanged, so the
        target is pending at its new head (``requoted_from``).
        """
        writer = self._writer
        with writer._write_scope():  # pyright: ignore[reportPrivateUsage]
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
        with writer._write_scope():  # pyright: ignore[reportPrivateUsage]
            if writer.counter(f"erase_epoch:{scope.scope_id}") != request.erase_epoch:
                raise AMLError("operation_conflict", "The scope changed since the plan")
            for row in request.candidates:
                item = writer.get_row("item", row.memory_id)
                if item is None or item["current_revision_id"] != row.revision_id or (
                        item["lifecycle_state"] != "active"):
                    raise AMLError("operation_conflict", "The scope changed since the plan")


def _decision(pick: ForgetPickV1) -> ForgetStatementDecisionV1:
    return ForgetStatementDecisionV1(
        prompt_digest=FORGET_PICK_PROMPT_DIGEST, method=pick.method, calls=pick.calls,
        answered=sum(stage.answered for stage in pick.stages), fallbacks=pick.fallbacks,
        votes=[ForgetStatementVoteV1(
            stage=stage.stage, key=decision.key, votes=dict(decision.votes),
            decided=decision.decided, fallback=decision.fallback,
        ) for stage in pick.stages for decision in stage.decisions],
    )


def forget_pick_sender(
    provider: StructuredInferenceProvider, binding: InferenceBindingV1,
    bind: Callable[[str, object], ContentBindingDigest],
) -> Send:
    """Send forget-pick votes through the judge's provider under its binding (FS-1).

    Each vote is one ``memory-forget-pick-v1`` request with the judge binding's
    provider, model and reasoning effort, the FS-1 timeout and retries, and the pinned
    prompt. A call that ends ``provider_unavailable`` or ``schema_invalid`` is an
    unanswered vote; any other error propagates.

    Args:
        provider: The structured inference provider.
        binding: The memory judge's binding (C.5).
        bind: The keyed content binding for the request digests.

    Returns:
        Send: The sender `pick_targets` takes.
    """

    def send(stage: Stage, message: str) -> ForgetPickOutput | None:
        schema = (FORGET_PICK_CLASSIFY_SCHEMA_ID if stage == "classify"
                  else FORGET_PICK_REMAINDERS_SCHEMA_ID)
        request = InferenceRequest(
            purpose="judge", provider=cast(Literal["openai"], binding.provider),
            requested_model=binding.requested_model, schema_id=schema,
            prompt_version=FORGET_PICK_PROMPT_VERSION,
            input_digest=bind("forget-pick-input", {"message": message}),
            context_digest=bind("forget-pick-context", {
                "schema": schema, "prompt": FORGET_PICK_PROMPT_DIGEST}),
            messages=[Message(role="system", content=FORGET_PICK_PROMPT),
                      Message(role="user", content=message)],
            timeout_ms=FORGET_PICK_TIMEOUT_MS,
            retry_policy=binding.retry_policy.model_copy(update={
                "max_read_timeouts": FORGET_PICK_MAX_READ_TIMEOUTS,
            }),
            reasoning_effort=binding.reasoning_effort,
        )
        if hashlib.sha256(request.messages[0].content.encode("utf-8")).hexdigest() != (
                FORGET_PICK_PROMPT_DIGEST):
            raise AMLError("binding_mismatch", "The forget pick changed its pinned prompt")
        try:
            output = provider.infer(request).output
        except AMLError as error:
            if error.code not in _VOTE_ERRORS:
                raise
            return None
        if isinstance(output, ForgetPickClassifyV1 | ForgetPickRemaindersV1):
            return output
        return None

    return send
