"""Complete acknowledged (``Prefer: respond-async``) Adds in the background.

A plugin sends an Add with ``Prefer: respond-async``, gets 202 once its source is saved,
and reads the result by resending the exact same request. Its owner completes in the
scheduler without the client, but what follows completion before an Add may answer 200
ran only inside a request: the in-chat forget of the Add's requests and, in the ``agent``
mode, the check of every Add against the scope's forgotten facts
(``service/chat_forget.py``). Each resend that found the Add complete ran that work again,
model calls included, inside its own request. A resend therefore took as long as a forget
(tens of seconds to minutes), and a plugin whose resend times out sooner never got its
200: every resend started the work over (a hang an end-to-end test found).

`AddFollowups` runs that completion once per Add, in a background thread that acts as a
waiting client: the first acknowledgement starts it (`FollowedDispatch.completed`), and
it calls the ordinary synchronous Add (``BenchService.add``), which waits for the owner,
reads the stored result and carries out the forget work, all under the Add's own budget
and the service lifetime. Until it ends an exact resend answers 202 at once. Afterwards a
resend reads the stored result (the follow-up is recorded as done, so the forget work is
not repeated) and answers 200, or answers the failure the background run ended with,
once and as raised (a ``RecoveryInProgress`` keeps its longer ``Retry-After``); the
resend after that starts a new background run. Nothing is extracted or stored twice: the
background run is the same request identity, which the core completes once.

A run's budget is one request timeout, but an owner queued behind long work can take
longer. When only that budget ran out, the run waits again under a fresh one, up to
``WAIT_ROUNDS`` budgets, so a slow owner is never told as a failure: the resends keep
answering 202 (``RequestBudget``'s expiry is ``delivery_busy``).
An exact resend without ``Prefer: respond-async`` while a run is under way waits for that
run (`AddFollowups.wait`) instead of completing the Add a second time beside it.

While ``hold`` says so (the agent service's daily spend cap is reached) a run still
starts: reading a stored result spends nothing, and a model call its forget work would
make is refused locally by the provider route. A run that fails meanwhile is not told
(the Add keeps answering 202); the first resend after the reset starts it again. So an
Add whose forget work needs no model call (one finished before a restart, or a run
forgotten past ``MAX_REMEMBERED``) still answers its stored result under the cap.

What a background run has done lives in this process only. After a restart an exact
resend finds the Add complete without a record, starts a new background run and answers
202; that run carries out the forget work again, which finds nothing already forgotten.

A forget of the same user that commits after an Add's input was saved and before it was
ordered deletes that input, and the Writer refuses the Add's Run for good
(``erase_fenced``). In the agent service the forget is often the service's own: the
in-chat forget work of the user's previous Add, which runs in the background while the
plugin sends the next batch. The core's answer, 422
``operation_conflict``, tells a client to send the content again
under a new request id, but a plugin never re-sends a batch the service acknowledged, so
the batch was dropped. The background run therefore does that itself: it completes the
same messages under a derived request id (``core/readmission.py``), admitted after the
forget, at most ``MAX_READMISSIONS`` times, and the original request answers that result;
every read of the original id (its result, add-items, Why, its in-chat forget report)
finds the result under the derived one. The re-admitted messages are an Add after the
forget like any other: the ``agent`` mode checks them against the scope's forgotten facts.
A resend whose own attempt meets the refusal (no run known, after a restart) starts the
same run (`FollowedDispatch.step`).

Three rules keep a re-admission from doing more than the dropped Add would have:

- A deletion of the user is never undone. ``delete_user`` withdraws consent before it
  forgets, and the core then refuses an Add received before it ``consent_fenced``, which
  is not re-admitted; while a deletion runs, nothing is re-admitted
  (`FollowupStore.deleting`).
- The session keeps its order. While a background run of an Add is overtaken, or the
  user's erase epoch moved since it started (a forget committed; the run may yet be
  refused), a later Add of the same session is not received: it answers 429
  ``delivery_busy`` and its resend tries again (`AddFollowups.held_back`). Two overtaken
  Adds of one session are re-admitted in the order their runs started.
- An Add whose re-admissions ran out, or that a deletion or consent change refused, is
  refused for good: every resend answers that refusal without a new run, and
  ``/agent/status`` stops counting it in flight (`AddFollowups.refused`).

``readmissions`` counts, for ``/agent/status``, the Adds a re-admission stored (a replay of
a result stored before is not counted again) and those refused once re-admissions ran out.
"""

from __future__ import annotations

import contextvars
import hashlib
import itertools
import logging
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Final, Literal, Protocol

from quotemem.api.bench import AcceptingAddDispatch
from quotemem.core.add_outcome import AddOutcomeV1
from quotemem.core.errors import QuoteMemError
from quotemem.core.identity import canonical_json, scope_id
from quotemem.core.readmission import MAX_READMISSIONS, readmitted_request_id
from quotemem.core.request_budget import RequestBudget, request_budget
from quotemem.ingestion.boundary import AddRequestV1

MAX_RUNNING: Final = 64
"""Background runs at most at once; beyond it an Add waits for its resend to start one."""
MAX_REMEMBERED: Final = 4096
"""Finished runs remembered, oldest forgotten first; a forgotten one is simply run again."""
WAIT_ROUNDS: Final = 12
"""Budgets one run spends waiting before its expiry is told (two hours at the default)."""
WAIT_POLL_SECONDS: Final = 0.25
"""How often a synchronous resend waiting for a run checks its own budget."""
OVERTAKEN: Final = "erase_fenced"
"""The Writer's refusal of an Add Run admitted before a forget of its user committed."""
FINAL_REFUSALS: Final = frozenset({OVERTAKEN, "consent_fenced"})
"""Refusals of an Add's request id that no later run of it can change."""

__all__ = ["MAX_READMISSIONS", "AddFollowups", "FollowedDispatch", "FollowupStore"]

_LOG = logging.getLogger(__name__)

type FollowupKey = tuple[str, str, str, str]
"""``(scope_id, session_id, request_id, body digest)``: one exact request. Another body
under the same identity is no resend of it; the core answers that conflict itself."""


def followup_key(request: AddRequestV1) -> FollowupKey:
    """The identity of one Add's background completion."""
    digest = hashlib.sha256(canonical_json(request.model_dump(mode="json")).encode()).hexdigest()
    return scope_id("user", request.user_id), request.session_id, request.request_id, digest


def readmitted_request(request: AddRequestV1, admission: int) -> AddRequestV1:
    """The same Add under the request id of its ``admission``-th re-admission.

    The id is a function of the original id alone, so a restart re-admits under the same
    id and the core replays a result it already stored instead of extracting it again.
    """
    return AddRequestV1.model_validate({
        **request.model_dump(mode="json"),
        "request_id": readmitted_request_id(request.request_id, admission),
    })


type AddCoordinate = tuple[str, str, str]
"""``(user_id, session_id, request_id)``: an Add as ``/agent/status`` counts it."""


class FollowupStore(Protocol):
    """What the background runs read of the store (``agent/admin.AgentStore``)."""

    def deleting(self, user_id: str) -> bool:
        """Whether the user is being deleted."""
        ...

    def completed_adds(self, requests: Iterable[AddCoordinate]) -> frozenset[AddCoordinate]:
        """Which of ``requests`` have a stored result."""
        ...

    def erase_epoch(self, user_id: str) -> int | None:
        """The user's erase epoch (each committed forget advances it); None when unread."""
        ...


@dataclass
class _Refusal:
    """An Add refused for good: every resend answers ``error`` without a new run."""

    coordinate: AddCoordinate
    error: QuoteMemError


@dataclass
class _Run:
    state: Literal["running", "done", "failed"]
    budget: RequestBudget
    user_id: str = ""
    seq: int = 0
    epoch: int | None = None
    """The user's erase epoch when the run started; None when unread."""
    overtaken: bool = False
    """Whether a forget overtook the Add: from its first refusal until the run ends."""
    final: bool = False
    """Whether ``error`` refuses the Add for good (`_Refusal`)."""
    outcome: AddOutcomeV1 | None = None
    error: QuoteMemError | None = None
    ended: threading.Event = field(default_factory=threading.Event)


class AddFollowups:
    """The background completions of acknowledged Adds in one service process.

    Args:
        complete: The synchronous Add (``BenchService.add``): it waits for the owner and
            carries out what follows completion, under the caller's request budget.
        seconds: The budget of one wait of a background run (``WAIT_ROUNDS`` at most).
    """

    hold: Callable[[], bool] | None = None
    """True while a failed run is not told (the agent service's spend cap;
    ``agent/serve.py``)."""
    store: FollowupStore | None = None
    """The store's deletion state, stored results and erase epochs (``agent/serve.py``);
    without it a run re-admits while its user is deleted and only an overtaken run holds
    its session's later Adds."""

    def __init__(self, complete: Callable[[AddRequestV1], AddOutcomeV1], seconds: float) -> None:
        self._complete = complete
        self._seconds = seconds
        self._runs: OrderedDict[FollowupKey, _Run] = OrderedDict()
        self._refused: OrderedDict[FollowupKey, _Refusal] = OrderedDict()
        self._lock = threading.Lock()
        self._changed = threading.Condition(self._lock)
        self._seq = itertools.count(1)
        self._stopped = False
        self._threads: set[threading.Thread] = set()
        self._readmitted = 0
        self._unstored = 0

    def running(self) -> int:
        """Background runs not yet ended: the Adds this process is still completing."""
        with self._lock:
            return sum(run.state == "running" for run in self._runs.values())

    def readmissions(self) -> dict[str, int]:
        """Adds a forget overtook since this process started: ``readmitted`` (stored by a
        re-admission; a replay of a stored result is not counted) and ``unstored``
        (re-admissions ran out; the Add is refused for good)."""
        with self._lock:
            return {"readmitted": self._readmitted, "unstored": self._unstored}

    def refused(self) -> list[AddCoordinate]:
        """The Adds refused for good (the most recent ``MAX_REMEMBERED``): none of them is
        still in flight."""
        with self._lock:
            return [refusal.coordinate for refusal in self._refused.values()]

    def held_back(self, request: AddRequestV1) -> bool:
        """Whether this Add must wait before it is received: an earlier Add of its session
        runs in the background and is overtaken, or a forget committed since it started
        (it may yet be refused). Receiving this Add first would order it before that
        Add's re-admission."""
        key = followup_key(request)
        with self._lock:
            earlier = [run for other, run in self._runs.items() if run.state == "running"
                       and other[:2] == key[:2] and other[2] != key[2]]
        if not earlier:
            return False
        if any(run.overtaken for run in earlier):
            return True
        started = {run.epoch for run in earlier if run.epoch is not None}
        store = self.store
        if not started or store is None:
            return False
        current = store.erase_epoch(request.user_id)
        return current is not None and any(epoch != current for epoch in started)

    def outcome(self, request: AddRequestV1) -> AddOutcomeV1 | None:
        """The result a finished background run of this Add stored; None otherwise."""
        with self._lock:
            run = self._runs.get(followup_key(request))
            return None if run is None or run.state != "done" else run.outcome

    def known(self, request: AddRequestV1) -> bool:
        """Whether a background run of this Add runs or finished (a failure not yet told),
        or the Add is refused for good."""
        key = followup_key(request)
        with self._lock:
            return key in self._runs or key in self._refused

    def refusal(self, request: AddRequestV1) -> QuoteMemError | None:
        """The refusal of an Add refused for good; None otherwise."""
        with self._lock:
            refusal = self._refused.get(followup_key(request))
            return None if refusal is None else refusal.error

    def wait(self, request: AddRequestV1, budget: RequestBudget) -> AddOutcomeV1 | None:
        """The result of this Add's background run, once it ends; None when none is known.

        A synchronous resend (no ``Prefer: respond-async``) reads it here rather than
        completing the Add, and its forget work, a second time beside the run.

        Raises:
            QuoteMemError: The failure the run ended with, once; ``delivery_busy`` when the
                caller's own budget runs out first.
        """
        key = followup_key(request)
        with self._lock:
            run = self._runs.get(key)
            refusal = self._refused.get(key)
        if refusal is not None:
            raise refusal.error
        if run is None:
            return None
        while not run.ended.wait(min(WAIT_POLL_SECONDS, max(budget.remaining(), 0.001))):
            budget.check()
        with self._lock:
            if run.state == "done":
                return run.outcome
            if self._runs.get(key) is run and not run.final:
                del self._runs[key]
        assert run.error is not None
        raise run.error

    def finished(self, request: AddRequestV1) -> bool:
        """Whether this Add may answer its result now; starts a background run when none.

        Returns:
            True when a background run finished it; False while one runs or was just
            started (or none could start).

        Raises:
            QuoteMemError: The failure the last background run ended with, once; the next
                call starts a new run. The refusal of an Add refused for good, each time.
        """
        key = followup_key(request)
        with self._lock:
            refusal = self._refused.get(key)
            if refusal is not None:
                raise refusal.error
            run = self._runs.get(key)
            if run is not None:
                if run.state == "done":
                    self._runs.move_to_end(key)
                    return True
                if run.state == "running":
                    return False
                del self._runs[key]
                assert run.error is not None
                raise run.error
        self.start(request)
        return False

    def start(self, request: AddRequestV1) -> bool:
        """Start a background run of this Add unless one is known.

        Returns:
            Whether a run of it is known now: False only when none could start (the
            service is stopping, or ``MAX_RUNNING`` runs are under way).
        """
        key = followup_key(request)
        with self._lock:
            if key in self._runs or key in self._refused:
                return True
        store = self.store
        epoch = None if store is None else store.erase_epoch(request.user_id)
        with self._lock:
            if key in self._runs or key in self._refused:
                return True
            if self._stopped or sum(
                    entry.state == "running" for entry in self._runs.values()) >= MAX_RUNNING:
                return False
            run = self._runs[key] = _Run(
                "running", RequestBudget(self._seconds), user_id=request.user_id,
                seq=next(self._seq), epoch=epoch)
            self._forget_oldest()
        thread = threading.Thread(
            target=contextvars.Context().run, args=(self._run, key, request, run),
            name="quotemem-add-followup", daemon=True,
        )
        with self._lock:
            self._threads.add(thread)
        thread.start()
        return True

    def _held(self) -> bool:
        hold = self.hold
        return hold is not None and hold()

    def _renewed(self, run: _Run, budget: RequestBudget, round_no: int) -> bool:
        """Whether the run waits again: only its own budget ran out, the service still
        runs and rounds are left; the fresh budget replaces it (``stop`` cancels that)."""
        with self._lock:
            if self._stopped or not budget.cancelled or round_no + 1 >= WAIT_ROUNDS:
                return False
            run.budget = RequestBudget(self._seconds)
            return True

    def _completed(self, key: FollowupKey, request: AddRequestV1, run: _Run) -> AddOutcomeV1:
        """Complete the Add, again under its re-admission when a forget overtook it.

        Raises:
            QuoteMemError: The completion's failure; a refusal for good marks the run final.
        """
        admitted = request
        for admission in range(MAX_READMISSIONS + 1):
            try:
                if admission == 0:
                    return self._waited(admitted, run)
                stored = self._stored(admitted)
                outcome = self._waited(admitted, run)
                if not stored:
                    with self._lock:
                        self._readmitted += 1
                return outcome
            except QuoteMemError as error:
                if error.code not in FINAL_REFUSALS:
                    raise
                exhausted = error.code == OVERTAKEN and admission == MAX_READMISSIONS
                # A deletion of the user drops what it found, this Add's input included.
                if error.code != OVERTAKEN or exhausted or self._deleting(request):
                    with self._lock:
                        run.final = True
                        self._unstored += exhausted
                    raise
                with self._lock:
                    run.overtaken = True
                self._wait_turn(key, run)
                _LOG.warning("Acknowledged Add overtaken by a forget of its user; admitted "
                             "again (admission %d of %d)", admission + 1, MAX_READMISSIONS)
                admitted = readmitted_request(request, admission + 1)
        raise AssertionError("unreachable: the last admission returns or raises")

    def _deleting(self, request: AddRequestV1) -> bool:
        store = self.store
        return store is not None and store.deleting(request.user_id)

    def _stored(self, request: AddRequestV1) -> bool:
        """Whether a result of this exact request id is stored already (a replay)."""
        store = self.store
        coordinate = (request.user_id, request.session_id, request.request_id)
        try:
            return store is not None and coordinate in store.completed_adds((coordinate,))
        except QuoteMemError:
            return False

    def _wait_turn(self, key: FollowupKey, run: _Run) -> None:
        """Wait until the runs of the same session that started earlier ended, so their
        re-admissions keep the session's order; never longer than ``WAIT_ROUNDS``
        budgets, nor once the service stops."""
        deadline = time.monotonic() + self._seconds * WAIT_ROUNDS
        with self._changed:
            while not self._stopped and any(
                    other.state == "running" and other.seq < run.seq and other_key[:2] == key[:2]
                    for other_key, other in self._runs.items()):
                left = deadline - time.monotonic()
                if left <= 0:
                    return
                self._changed.wait(min(left, WAIT_POLL_SECONDS))

    def _waited(self, request: AddRequestV1, run: _Run) -> AddOutcomeV1:
        for round_no in range(WAIT_ROUNDS):
            budget = run.budget
            try:
                with request_budget(budget):
                    return self._complete(request)
            except QuoteMemError:
                if not self._renewed(run, budget, round_no):
                    raise
                _LOG.info("Background Add completion waits again (round %d)", round_no + 2)
        raise AssertionError("unreachable: the last round raises")

    def _run(self, key: FollowupKey, request: AddRequestV1, run: _Run) -> None:
        try:
            outcome = self._completed(key, request, run)
        except QuoteMemError as error:
            self._end(key, run, error=error)
        except Exception as error:  # every failure must end the run
            _LOG.warning("Background Add completion failed: %s", type(error).__name__)
            self._end(key, run, error=QuoteMemError(
                "service_unhealthy", "The background Add completion failed"))
        else:
            self._end(key, run, outcome=outcome)
        finally:
            with self._lock:
                self._threads.discard(threading.current_thread())

    def _end(self, key: FollowupKey, run: _Run, *, outcome: AddOutcomeV1 | None = None,
             error: QuoteMemError | None = None) -> None:
        held = error is not None and not run.final and self._held()
        with self._changed:
            run.state, run.outcome, run.error = (
                ("done", outcome, None) if error is None else ("failed", None, error))
            if self._runs.get(key) is run and error is not None and run.final:
                del self._runs[key]
                self._refused[key] = _Refusal((run.user_id, key[1], key[2]), error)
                while len(self._refused) > MAX_REMEMBERED:
                    self._refused.popitem(last=False)
            elif self._runs.get(key) is run:
                if held:
                    # Under the spend cap the failure is the refused model calls: not told,
                    # the first resend after the reset runs it again.
                    del self._runs[key]
                else:
                    self._runs.move_to_end(key)
                    self._forget_oldest()
            self._changed.notify_all()
        run.ended.set()

    def _forget_oldest(self) -> None:
        excess = len(self._runs) - MAX_REMEMBERED
        if excess > 0:
            for key in [key for key, run in self._runs.items() if run.state != "running"][
                    :excess]:
                del self._runs[key]

    def stop(self) -> None:
        """Start no further run and cancel the budgets of those running (service drain).

        A cancelled run ends at its next budget check; its Add stays saved, and a resend
        after the restart completes it.
        """
        with self._lock:
            self._stopped = True
            budgets = [run.budget for run in self._runs.values() if run.state == "running"]
        for budget in budgets:
            budget.cancel()

    def join(self, timeout: float) -> bool:
        """Wait for the running threads (tests); True when none is left."""
        with self._lock:
            threads = list(self._threads)
        for thread in threads:
            thread.join(timeout)
        with self._lock:
            return not any(thread.is_alive() for thread in self._threads)


class FollowedDispatch:
    """An accepting Add dispatch whose result waits for its background completion.

    ``completed`` is True only once the Add's background run finished, so an exact resend
    answers 202 while it runs and the stored result afterwards; the HTTP Add then reads
    that result through the synchronous Add, which no longer repeats the forget work.
    """

    def __init__(self, inner: AcceptingAddDispatch, followups: AddFollowups,
                 request: AddRequestV1) -> None:
        self._inner, self._followups, self._request = inner, followups, request

    def step(self) -> bool:
        """The inner step; an Add a forget overtook is completed by its background run.

        Raises:
            QuoteMemError: The inner step's failure; ``delivery_busy`` while an earlier Add of
                the session may yet be re-admitted (`AddFollowups.held_back`), or when an
                overtaken Add's background run cannot start now (the resend starts it
                later); the refusal of an Add refused for good.
        """
        followups = self._followups
        if not self._inner.accepted and followups.held_back(self._request):
            raise QuoteMemError("delivery_busy",
                                "An earlier Add of this session awaits its re-admission; retry")
        try:
            return self._inner.step()
        except QuoteMemError as error:
            if error.code != OVERTAKEN:
                raise
            refusal = followups.refusal(self._request)
            if refusal is not None:
                raise refusal from None
            if not followups.start(self._request):
                raise QuoteMemError("delivery_busy",
                                    "The Add awaits its re-admission; retry") from None
            return True

    def completion_ready(self) -> bool:
        # A background run waits for the owner; the Add reads (or waits for) its result.
        return self._followups.known(self._request) or self._inner.completion_ready()

    def in_background(self) -> bool:
        """Whether a background run of this Add runs or ended, so a resend reads it,
        with or without ``Prefer: respond-async``, instead of ordering it again."""
        return self._followups.known(self._request)

    async def wait_for_change(self) -> None:
        await self._inner.wait_for_change()

    @property
    def accepted(self) -> bool:
        return self._inner.accepted

    def completed(self) -> bool:
        """Whether the Add's background run finished; starts one when none runs.

        Raises:
            QuoteMemError: The failure the Add's last background run ended with.
        """
        return self._followups.finished(self._request)

