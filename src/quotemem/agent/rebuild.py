"""``quotemem rebuild``: make a new agent store from an old one's source journal.

A store from another release, or one bound to another embedding model, is never
migrated in place. The rebuild replays what the user said through the current
code into a new data directory, so every derived record (memories, vectors, judge
records, indexes) is made again by this release:

1. The old store is exported (``quotemem.agent.export``) into
   ``<new>/rebuild/journal.jsonl``, its forgotten statements sealed under a new
   key in ``<new>/rebuild/journal.key``; or a journal file is verified, its
   sealed statements opened with the key the caller names (``--key-file`` or
   ``--passphrase-env``), and copied there. The old store is read only. A journal
   whose forgets cannot be opened is refused: a rebuild never skips a forget.
2. The plan is printed before any provider call: Adds, forget statements, content
   bytes and an estimate of the model calls.
3. The new store is opened with the bench composition behind the agent gateway, on
   a loopback port with the store's token, exactly as ``quotemem serve`` runs it. The
   gateway normalizes each replayed Add under the new store's Add mode, so a user
   message's speaker metadata follows that mode's rule (``quotemem.agent.ingest``).
   The records are sent in the order the old store took them (``in_time_order``):
   every Add in the journal's order, with its original user, session and request ids
   and message timestamps, and every forget at its place in time among them, before
   the first Add ingested after it. A forget is never moved after an Add the old
   store took in later, so a fact the user stated again after a forget meets the
   forget the old store had carried out, never one replayed after its restatement.
   Each forget statement goes through forget by statement (prepare, then confirm the
   plan it returns). The old store's forget withheld the words that stated it, so the
   replay mostly finds nothing. Each statement, found or not, is then tombstoned over
   the user's withheld markers
   (``AgentStore.forget_withheld``, a forget of sentences seeded with it), so the new
   store holds the old one's tombstones and forget shadow (a later restatement is
   treated as the old store treated it: the shadow's names take the old forget's time, or
   the latest date of its user's earlier messages when later (``restamp_times``,
   ``AgentStore.restamp_names``), which a restatement dated after it lifts) and reads a
   part holding a marker, as the old
   store read the part whose words it withheld, as no Search hit. An Add after the
   user's forget whose words the old store withheld as a restatement, or whose part an
   erasure emptied, has that tombstoning repeated right after it (``late_markers``), so
   its markers are no Search hit either. Right after
   the Add a journal ``retired`` range names (``after``), the memories that quote
   those words are ended as the old store held them (``AgentStore.retire``): a
   memory whose successor the user asked to forget stays a past state, not a
   current one, although the words that ended it are withheld from the replay,
   and the later Adds meet it ended, as they did in the old store. When the old
   store is a data directory, the export's skipped counts are printed first
   (``unplaced_retired`` and ``ambiguous_retired`` are ended memories the rebuild
   cannot name, which the replay may bring back as current). After the last record a
   safety pass (``safety_pass``) replays every forget once more, in time order: each
   statement is forgotten again over the Evidence of its user's Adds the old store
   took at or before it alone (``AgentStore.forget_earlier``: a current memory drawn
   only from there goes without a vote when its words are the statement's or the
   statement's forget shadow matches it, as Search's would; another the judge finds
   stating it goes too) and tombstoned over every withheld marker of the
   user (``AgentStore.forget_withheld``). So nothing said before a forget is current
   after the rebuild, however the rebuild's chat model re-decided an extraction, a
   pick or a relearn check; only what the user said after it can survive, as the
   relearn check admitted it, and a memory drawing on a later Add is never reached.
4. After each record ``<new>/rebuild/state.json`` records the next one. An
   interrupted rebuild resumes there when run again with the same ``--to``; the
   record in flight is sent again, which an Add's request id makes idempotent, and
   its retirement runs again, which finds nothing left to end. The safety pass is
   complete only with the rebuild and runs again whole when resumed; a repeat finds
   nothing left to forget. A
   resumed rebuild of a journal file needs the same key again. On completion the
   journal copy and its rebuild key are deleted and the state is marked complete.

A rebuild never writes into a data directory that holds anything but its own
unfinished rebuild.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any, Final, Literal, cast

import httpx
from pydantic import ConfigDict, Field

from quotemem.agent.admin import AgentStore
from quotemem.agent.export import export_store
from quotemem.agent.journal import (
    Forget,
    Journal,
    JournalAddV1,
    ReplayEntry,
    read_journal,
)
from quotemem.agent.journal_key import JournalKey, KeySource, key_for, write_key_file
from quotemem.agent.probe import probe_providers
from quotemem.agent.routing import RoutedAsyncClient, RoutedClient
from quotemem.agent.serve import AgentServeOptions, AgentService, compose, notice
from quotemem.agent.settings import (
    AGENT_JOB_ID,
    ROUTED_ENVIRONMENT,
    agent_provider_config,
    resolve_review,
    store_provider_config,
)
from quotemem.api.memory_retire import QuotedRange
from quotemem.core.errors import QuoteMemError
from quotemem.core.extraction_policy import AddMode, ReviewMode
from quotemem.core.forget_pick import MAX_FORGET_STATEMENT_CHARS
from quotemem.core.platform_files import ensure_private_directory, open_private_file
from quotemem.core.request_budget import DEFAULT_REQUEST_CONTRACT_VERSION
from quotemem.providers.config import DEFAULT_ADD_MODE
from quotemem.service.listener import UvicornListener
from quotemem.store.metadata import StrictModel

STATE_FORMAT: Final = "quotemem-rebuild-state"
CHAT_CALLS_PER_ADD: Final = 2
"""Extraction plus the judge's state-change check; supplements and retries add more."""
EMBEDDING_CALLS_PER_ADD: Final = 2
"""The stored sentences and the memories made from them."""
CALLS_PER_FORGET_STATEMENT: Final = 1
"""One judge pick and one query vector per statement."""
RETRY_ATTEMPTS: Final = 8
"""Sends of one record before the rebuild stops and leaves it for a resume."""
_RETRYABLE_STATUS: Final = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
_MAX_RETRY_SECONDS: Final = 60.0


class RebuildStateV1(StrictModel):
    """``<new>/rebuild/state.json``: where an unfinished rebuild resumes."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    format: Literal["quotemem-rebuild-state"] = STATE_FORMAT
    version: Literal[1] = 1
    journal_sha256: str
    total: int = Field(ge=0)
    next: int = Field(ge=0)
    complete: bool = False
    started_at_ms: int
    updated_at_ms: int


@dataclass(frozen=True)
class RebuildPlan:
    """What a rebuild will send and roughly what it costs."""

    adds: int
    forget_statements: int
    messages: int
    content_bytes: int
    resume_at: int
    total: int

    def estimate(self) -> dict[str, object]:
        """The plan as a content-free JSON document, with call estimates."""
        remaining = self.total - self.resume_at
        return {
            "status": "rebuild_plan", "adds": self.adds,
            "forget_statements": self.forget_statements, "messages": self.messages,
            "content_bytes": self.content_bytes, "records": self.total,
            "resume_at": self.resume_at, "remaining_records": remaining,
            "estimated_chat_calls": self.adds * CHAT_CALLS_PER_ADD
            + self.forget_statements * CALLS_PER_FORGET_STATEMENT,
            "estimated_embedding_calls": self.adds * EMBEDDING_CALLS_PER_ADD
            + self.forget_statements * CALLS_PER_FORGET_STATEMENT,
            "estimated_input_tokens": self.content_bytes // 4 * CHAT_CALLS_PER_ADD,
            "note": "Estimates assume about two chat calls per Add; resumed records are "
                    "answered from the new store without new calls",
        }


@dataclass(frozen=True)
class RebuildOptions:
    """Nonsecret options of one rebuild.

    Attributes:
        source: The old data directory, or a journal file.
        target: The new data directory; empty, absent or an unfinished rebuild.
        dimensions: The new store's vector dimension; None takes the journal's.
        add_mode: The new store's Add mode; None takes ``AGENT_ADD_MODE``.
        review: The new store's review; None reads ``QUOTEMEM_REVIEW``, then ``standard``.
            A rebuild is how a store moves to another review.
        dry_run: Print the plan and stop before opening the new store.
        request_timeout_seconds: The budget of one Add or forget.
    """

    source: Path
    target: Path
    dimensions: Literal[1024, 2048] | None = None
    add_mode: AddMode | None = None
    review: ReviewMode | None = None
    dry_run: bool = False
    request_timeout_seconds: float = 600.0


def _now_ms() -> int:
    return time.time_ns() // 1_000_000


def _rebuild_dir(target: Path) -> Path:
    return target / "rebuild"


def _write_state(target: Path, state: RebuildStateV1) -> None:
    path = _rebuild_dir(target) / "state.json"
    staged = path.with_name("state.json.tmp")
    staged.unlink(missing_ok=True)
    descriptor = open_private_file(staged, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(state.model_dump_json() + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(staged, path)


def _read_state(target: Path) -> RebuildStateV1 | None:
    path = _rebuild_dir(target) / "state.json"
    if not (path.exists() or path.is_symlink()):
        return None
    descriptor = open_private_file(path, os.O_RDONLY)
    with os.fdopen(descriptor, "rb") as stream:
        return RebuildStateV1.model_validate_json(stream.read())


def in_time_order(entries: tuple[ReplayEntry, ...]) -> tuple[ReplayEntry, ...]:
    """The records in the order the store took them: the order a rebuild replays.

    The Adds keep the journal's order (the store's commit order). The forgets, by
    ``at_ms`` (stably, so two at one time keep the journal's order), each go before the
    first Add ingested after it (``ingested_at_ms`` greater than its ``at_ms``); an Add
    ingested at the same millisecond came first. A journal lists every forget after
    every Add, and a statement forget, a forget of sentences and a forget a rebuild
    tombstoned over withheld markers are all forgets alike. A deleted user leaves no
    record to place: the deletion erased their Adds and tombstones.
    """
    waiting = deque(sorted((entry for entry in entries if isinstance(entry, Forget)),
                           key=lambda forget: forget.at_ms))
    ordered: list[ReplayEntry] = []
    for entry in entries:
        if isinstance(entry, Forget):
            continue
        while waiting and waiting[0].at_ms < entry.ingested_at_ms:
            ordered.append(waiting.popleft())
        ordered.append(entry)
    ordered.extend(waiting)
    return tuple(ordered)


RESTATED_FORGOTTEN: Final = "restated_forgotten"
"""The reason of a range an Add withheld as a restatement of a forgotten fact."""


def late_markers(entries: tuple[ReplayEntry, ...]) -> dict[int, Forget]:
    """The Adds whose withheld markers no replayed forget reaches, by replay position.

    A forget tombstoned over withheld markers (``AgentStore.forget_withheld``) withholds
    the markers the user's Adds hold when it is replayed. The old store also withheld,
    after a forget, the words of a later Add that restated the forgotten fact (reason
    ``restated_forgotten``) and the parts a privacy erasure emptied; replayed in time
    order, such an Add comes after every forget that could withhold its markers. Each
    one after a forget of its user maps to the user's latest forget before it, whose last
    statement's replay is repeated right after the Add, so its markers are withheld as the
    old store withheld those words, under a tombstone the user already has.
    ``entries`` are in replay order (``in_time_order``).
    """
    latest: dict[str, Forget] = {}
    found: dict[int, Forget] = {}
    for index, entry in enumerate(entries):
        if isinstance(entry, Forget):
            latest[entry.user_id] = entry
        elif entry.user_id in latest and (entry.erased_parts or any(
                span.reason == RESTATED_FORGOTTEN for span in entry.withheld)):
            found[index] = latest[entry.user_id]
    return found


type Safety = tuple[Forget, frozenset[tuple[str, str]]]
"""(a journal forget, the (session id, request id) of its user's Adds replayed before it)."""


def safety_pass(entries: tuple[ReplayEntry, ...]) -> tuple[Safety, ...]:
    """The forgets the safety pass replays once more after every Add, in replay order.

    Each comes with the Adds of its user that the old store took at or before it (those
    ``in_time_order`` places before it): the Evidence it is forgotten over again
    (``AgentStore.forget_earlier``). ``entries`` are in replay order.
    """
    before: dict[str, set[tuple[str, str]]] = {}
    found: list[Safety] = []
    for entry in entries:
        if isinstance(entry, Forget):
            found.append((entry, frozenset(before.get(entry.user_id, ()))))
        else:
            before.setdefault(entry.user_id, set()).add((entry.session_id, entry.request_id))
    return tuple(found)


def _open(journal: Journal, keys: KeySource, target: Path) -> tuple[ReplayEntry, ...]:
    """The journal's records with its forgets opened, in time order (``in_time_order``);
    a rebuild's own key comes first."""
    if not journal.sealed:
        return in_time_order(journal.open(None))
    local = _rebuild_dir(target) / "journal.key"
    if local.exists() or local.is_symlink():
        keys = KeySource(key_file=local)
    return in_time_order(journal.open(key_for(journal.header.encryption, keys)))


def _same_place(first: Path, second: Path) -> bool:
    one, two = first.expanduser().resolve(), second.expanduser().resolve()
    return one == two or one in two.parents or two in one.parents


def prepare(
    options: RebuildOptions, keys: KeySource | None = None,
) -> tuple[Journal, tuple[ReplayEntry, ...], RebuildStateV1]:
    """Start or resume a rebuild: the verified journal, its opened records and the state.

    Args:
        options: The rebuild options.
        keys: Where the key of a journal file is (not needed for a data directory,
            whose export the rebuild seals under its own key).

    Raises:
        QuoteMemError: ``operation_conflict`` when the target holds anything but this
            rebuild, or names the source; ``input_required`` or ``secret_rejected``
            when a journal's forgets cannot be opened; the export's and journal's
            refusals.
    """
    keys = keys or KeySource()
    source, target = options.source.expanduser(), options.target.expanduser()
    if _same_place(source, target):
        raise QuoteMemError("operation_conflict", "--to must be a new directory outside --from")
    state = _read_state(target) if target.is_dir() else None
    journal_path = _rebuild_dir(target) / "journal.jsonl"
    if state is not None:
        if state.complete:
            raise QuoteMemError("operation_conflict", "This rebuild has already completed")
        journal = read_journal(journal_path)
        if journal.sha256 != state.journal_sha256:
            raise QuoteMemError("parity_mismatch", "The rebuild's journal copy changed")
        if source.is_file() and read_journal(source).sha256 != journal.sha256:
            raise QuoteMemError("operation_conflict",
                                "--to holds an unfinished rebuild of another journal")
        return journal, _open(journal, keys, target), state
    if target.exists() and (not target.is_dir() or any(target.iterdir())):
        raise QuoteMemError(
            "operation_conflict",
            "--to is not empty; a rebuild never overwrites a store. Name a new directory",
        )
    copied: Journal | None = None
    opened: tuple[ReplayEntry, ...] | None = None
    if source.is_file():
        # Refuse a journal whose forgets cannot be opened before creating anything.
        copied = read_journal(source)
        opened = _open(copied, keys, target)
    target.parent.mkdir(parents=True, exist_ok=True)
    ensure_private_directory(target)
    ensure_private_directory(_rebuild_dir(target))
    if source.is_file():
        staged = journal_path.with_name("journal.jsonl.tmp")
        staged.unlink(missing_ok=True)
        descriptor = open_private_file(staged, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(descriptor, "wb") as stream, source.open("rb") as original:
            shutil.copyfileobj(original, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(staged, journal_path)
    else:
        key = JournalKey.generate()
        write_key_file(_rebuild_dir(target) / "journal.key", key)
        exported = export_store(source, journal_path, key=key)
        notice({"status": "exported", "skipped": asdict(exported.skips)})
    journal = read_journal(journal_path)
    if copied is not None and copied.sha256 != journal.sha256:
        raise QuoteMemError("parity_mismatch", "The journal changed while it was copied")
    if opened is None:
        opened = _open(journal, keys, target)
    now = _now_ms()
    state = RebuildStateV1(
        journal_sha256=journal.sha256, total=len(journal.entries), next=0,
        started_at_ms=now, updated_at_ms=now,
    )
    _write_state(target, state)
    return journal, opened, state


def plan(journal: Journal, state: RebuildStateV1) -> RebuildPlan:
    """The rebuild plan of ``journal`` from ``state``."""
    counts = journal.end.counts
    return RebuildPlan(
        adds=counts.adds, forget_statements=counts.forget_statements,
        messages=counts.messages, content_bytes=counts.content_bytes,
        resume_at=state.next, total=state.total,
    )


def add_body(entry: JournalAddV1) -> dict[str, object]:
    """The ``/memory/add`` body that replays one journal Add."""
    messages: list[dict[str, object]] = []
    versioned = False
    for message in entry.messages:
        fields: dict[str, object] = {"role": message.role, "content": message.content}
        if message.timestamp is not None:
            fields["timestamp"] = message.timestamp
        if message.speaker is not None:
            fields["speaker"] = message.speaker
            versioned = True
        if message.source_participant is not None:
            fields["source_participant"] = message.source_participant
            versioned = True
        messages.append(fields)
    body: dict[str, object] = {
        "user_id": entry.user_id, "session_id": entry.session_id,
        "request_id": entry.request_id, "messages": messages,
    }
    if versioned:
        body["source_metadata_version"] = 1
    return body


class RebuildStopped(QuoteMemError):
    """A record could not be sent; the rebuild resumes at it when run again."""


def _retry_seconds(response: httpx.Response, attempt: int) -> float:
    header = response.headers.get("retry-after", "")
    try:
        seconds = float(header)
    except ValueError:
        seconds = float(2 ** attempt)
    return min(max(seconds, 0.5), _MAX_RETRY_SECONDS)


def _code(response: httpx.Response) -> str:
    try:
        value: object = response.json()
    except ValueError:
        return f"http_{response.status_code}"
    if isinstance(value, dict):
        code = cast(dict[str, Any], value).get("code")
        if isinstance(code, str):
            return code
    return f"http_{response.status_code}"


async def _post(
    http: httpx.AsyncClient, path: str, body: Mapping[str, object], *, record: int,
    sleep: Callable[[float], Any] = asyncio.sleep,
) -> dict[str, Any]:
    """POST with retries for transient refusals; the JSON answer."""
    for attempt in range(RETRY_ATTEMPTS):
        try:
            response = await http.post(path, json=body)
        except httpx.TransportError:
            if attempt + 1 == RETRY_ATTEMPTS:
                raise RebuildStopped(
                    "provider_unavailable", f"Record {record} could not be sent; run again",
                ) from None
            await sleep(min(2.0 ** attempt, _MAX_RETRY_SECONDS))
            continue
        if response.status_code == 200:
            return cast(dict[str, Any], response.json())
        if response.status_code not in _RETRYABLE_STATUS or attempt + 1 == RETRY_ATTEMPTS:
            code = _code(response)
            raise RebuildStopped(
                code, f"Record {record} was refused ({code}); fix the cause and run again",
            )
        await sleep(_retry_seconds(response, attempt))
    raise RebuildStopped("delivery_busy", f"Record {record} was not accepted; run again")


@dataclass
class _Forgotten:
    """What the replayed forgets did: memories forgotten, and statements tombstoned over
    the user's withheld markers (those of a user whose Evidence held any)."""

    memories: int = 0
    tombstoned: int = 0


def restamp_times(entries: tuple[ReplayEntry, ...]) -> dict[int, int]:
    """The time each replayed forget stamps its shadow names with, by replay position.

    A name is lifted only by a restatement received after its stamp and dated after it
    (``core/relearn.after``). In a rebuilt store every part is received on the rebuild's
    clock, after any journal time, so only the date tells a restatement from a message the
    old store took before the forget. The stamp is therefore the later of the forget's own
    time and the latest date among its user's messages replayed before it: a message dated
    ahead of the forget by a client clock running ahead never lifts the name, as the old
    store's receipt check kept it from doing. ``entries`` are in replay order.
    """
    latest: dict[str, int] = {}
    found: dict[int, int] = {}
    for index, entry in enumerate(entries):
        if isinstance(entry, Forget):
            found[index] = max(entry.at_ms, latest.get(entry.user_id, entry.at_ms))
            continue
        dates = [message.timestamp for message in entry.messages if message.timestamp is not None]
        if dates:
            latest[entry.user_id] = max(latest.get(entry.user_id, dates[0]), *dates)
    return found


async def _forget(
    http: httpx.AsyncClient, entry: Forget, *, record: int,
    withheld: Callable[[str, str], int] | None = None,
    store: AgentStore | None = None, stamp: tuple[int, int] | None = None,
) -> _Forgotten:
    """Forget each statement of one journal forget; what that did.

    A plan whose target changed before its confirmation is prepared again, up to
    ``RETRY_ATTEMPTS`` times. Each statement is then tombstoned over the user's
    withheld markers by ``withheld`` (``AgentStore.forget_withheld``) when given: the
    old store's forget withheld its words, so the replay holds the marker where they
    were, and a marker not withheld again would be read back as a literal
    ``[withheld]`` hit. That holds whether or not the prepare found a memory in the new
    store (a later Add may state the fact again, and a rebuild's chat model may pick it
    as the forgotten one): every journal statement is one of the old store's
    tombstones. Without ``withheld`` (a store whose forgets do not forget sentences
    alone) it is done. With ``store`` and ``stamp`` (since, at), the names stamped since
    ``since`` (the end of the previous record, so also by an attempt a crash cut short)
    are then stamped with ``at`` (`restamp_times`, ``AgentStore.restamp_names``): the
    rebuild's clock is later than every replayed message's date, and a name is lifted
    only by a restatement dated after its stamp.
    """
    done_by = _Forgotten()
    for number, statement in enumerate(entry.statements):
        for attempt in range(RETRY_ATTEMPTS):
            prepared = await _post(http, "/memory/forget/prepare", {
                "schema_version": 1, "user_id": entry.user_id,
                "client_session_id": "quotemem-rebuild",
                "request_id": f"rebuild-forget-{record}-{number}-{attempt}",
                "statement": statement[:MAX_FORGET_STATEMENT_CHARS],
            }, record=record)
            confirmation = prepared.get("confirmation")
            if not isinstance(confirmation, dict):
                break
            done = await _post(http, "/memory/forget/confirm",
                               cast(dict[str, object], confirmation), record=record)
            done_by.memories += sum(
                1 for outcome in cast(list[dict[str, Any]], done.get("outcomes", []))
                if outcome.get("outcome") == "verified" and outcome.get("memory_id")
            )
            if done.get("status") == "verified":
                break
        else:
            raise RebuildStopped(
                "operation_conflict", f"Record {record}'s forget did not complete; run again",
            )
        if withheld is not None and await asyncio.to_thread(withheld, entry.user_id, statement):
            done_by.tombstoned += 1
    if store is not None and stamp is not None:
        await asyncio.to_thread(partial(store.restamp_names, entry.user_id, since=stamp[0],
                                        at=stamp[1]))
    return done_by


type Progress = Callable[[int, RebuildStateV1], None]


type Retirement = tuple[str, Literal["update", "correct", "retract"], QuotedRange]
"""(user id, the relation that ended the memory, its quoted range in the new store)."""


def retirements(entries: tuple[ReplayEntry, ...]) -> dict[int, list[Retirement]]:
    """The journal's ``retired`` words, by the record after which each memory ends.

    A replayed message's position in its Add is its ordinal in the new store.

    Raises:
        QuoteMemError: ``schema_invalid`` when a range names no Add of its user at or
            after the Add that quotes it.
    """
    places: dict[tuple[str, str, str], int] = {
        (entry.user_id, entry.session_id, entry.request_id): index
        for index, entry in enumerate(entries) if isinstance(entry, JournalAddV1)
    }
    found: dict[int, list[Retirement]] = {}
    for index, entry in enumerate(entries):
        if not isinstance(entry, JournalAddV1):
            continue
        for span in entry.retired:
            after = places.get((entry.user_id, span.after.session_id, span.after.request_id))
            if after is None or after < index:
                raise QuoteMemError("schema_invalid",
                               f"Record {index} names a retirement point that is not a later Add")
            found.setdefault(after, []).append((entry.user_id, span.ended, QuotedRange(
                entry.session_id, entry.request_id, span.message, span.byte_start,
                span.byte_end,
            )))
    return found


def _retire(service: AgentService, retiring: list[Retirement]) -> int:
    """End the memories of the new store that quote ``retiring``; how many this ended."""
    grouped: dict[tuple[str, Literal["update", "correct", "retract"]], list[QuotedRange]] = {}
    for user_id, relation, quoted in retiring:
        grouped.setdefault((user_id, relation), []).append(quoted)
    return sum(service.store.retire(user_id, ranges, relation)
               for (user_id, relation), ranges in sorted(grouped.items()))


async def replay(
    service: AgentService, client: httpx.Client, entries: tuple[ReplayEntry, ...],
    state: RebuildStateV1,
    *, async_transport: httpx.AsyncBaseTransport | None = None,
    progress: Progress | None = None,
) -> RebuildStateV1:
    """Send the journal's opened records from ``state.next`` to a runtime over ``service``.

    ``entries`` are in replay order (``in_time_order``); ``state.next`` counts in it.

    Returns:
        The final state, ``complete`` when every record was sent.
    """
    from quotemem.api.cli import bench_runtime, serving_controller_factory

    ending = retirements(entries)
    rewithholding = late_markers(entries)
    stamps = restamp_times(entries)
    target = service.options.data_dir
    runtime = bench_runtime(
        service.workspace, service.config, ROUTED_ENVIRONMENT, job_id=AGENT_JOB_ID,
        credentials=service.credentials,
        search_timeout_seconds=service.options.request_timeout_seconds,
        request_contract_version=DEFAULT_REQUEST_CONTRACT_VERSION, client=client,
        async_client_factory=lambda: RoutedAsyncClient(
            service.routes, transport=async_transport,
        ),
        measurement=None, inspection=None,
        controller_factory=service.store.capturing(serving_controller_factory()),
    )
    try:
        service.state.enter("opening")
        app = service.gateway.serving(runtime.start())
        listener = UvicornListener(app, host="127.0.0.1", port=0)
        runtime.attach_listener(listener)
        await listener.start(time.monotonic() + 30)
        runtime.activate()
        service.store.attach(runtime.lifetime)
        async with httpx.AsyncClient(
            base_url=f"http://127.0.0.1:{listener.port}", trust_env=False,
            timeout=service.options.request_timeout_seconds + 30,
            headers={"Authorization": f"Bearer {service.credentials.secret}"},
        ) as http:
            forgotten = retired = tombstoned = 0
            # Only a complete forget (the agent Add mode) forgets sentences alone.
            withheld = service.store.forget_withheld if service.config.add_mode == "agent" \
                else None
            while state.next < state.total:
                entry = entries[state.next]
                if isinstance(entry, JournalAddV1):
                    await _post(http, "/memory/add", add_body(entry), record=state.next)
                    if state.next in ending:
                        retired += await asyncio.to_thread(
                            _retire, service, ending[state.next])
                    if withheld is not None and state.next in rewithholding:
                        late = rewithholding[state.next]
                        await asyncio.to_thread(partial(
                            service.store.forget_withheld, late.user_id, late.statements[-1],
                            enforces=late.at_ms))
                else:
                    # The previous record ended at updated_at_ms: every name stamped since
                    # is this forget's, an interrupted attempt's included.
                    done_by = await _forget(
                        http, entry, record=state.next, withheld=withheld,
                        store=None if withheld is None else service.store,
                        stamp=(state.updated_at_ms, stamps[state.next]))
                    forgotten += done_by.memories
                    tombstoned += done_by.tombstoned
                state = state.model_copy(update={
                    "next": state.next + 1, "updated_at_ms": _now_ms(),
                })
                _write_state(target, state)
                if progress is not None:
                    progress(state.next, state)
                if state.next == state.total or state.next % 25 == 0:
                    notice({"status": "rebuilding", "done": state.next, "total": state.total,
                            "memories_forgotten": forgotten, "memories_retired": retired,
                            "statements_tombstoned": tombstoned})
            if not state.complete and withheld is not None:
                # The safety pass, run again whole by a resumed rebuild (it is idempotent).
                # A repeat adds to what the replay forgot: the forget shadow keeps the stamps
                # the replay gave it, and what it adds takes the forget's own time.
                for forget, sources in safety_pass(entries):
                    for statement in forget.statements:
                        forgotten += await asyncio.to_thread(partial(
                            service.store.forget_earlier, forget.user_id, statement, sources,
                            enforces=forget.at_ms))
                        if await asyncio.to_thread(partial(
                                service.store.forget_withheld, forget.user_id, statement,
                                enforces=forget.at_ms)):
                            tombstoned += 1
                notice({"status": "safety_pass", "memories_forgotten": forgotten,
                        "statements_tombstoned": tombstoned})
            if not state.complete:
                state = state.model_copy(update={"complete": True, "updated_at_ms": _now_ms()})
                _write_state(target, state)
    finally:
        deadline = time.monotonic() + 60
        while not (await runtime.drain(min(deadline, time.monotonic() + 2))).complete:
            if time.monotonic() >= deadline:
                raise QuoteMemError("service_unhealthy", "The new store did not drain")
        while not await runtime.close_async(min(deadline, time.monotonic() + 2)):
            if time.monotonic() >= deadline:
                raise QuoteMemError("service_unhealthy", "The new store did not close")
    return state


def target_review(
    options: RebuildOptions, environment: Mapping[str, str],
) -> tuple[AddMode, ReviewMode]:
    """The new store's Add mode and review, checked before anything is created.

    The review comes from ``--review``, else ``QUOTEMEM_REVIEW``, never from the old
    store; a resumed rebuild keeps the review its new store was created with.

    Raises:
        QuoteMemError: ``schema_invalid`` for an unknown review or one the Add mode cannot
            take; ``workspace_binding_mismatch`` when a resumed rebuild's new store was
            created with another Add mode or review.
    """
    review = resolve_review(options.review, environment)
    target = options.target.expanduser()
    if target.is_dir() and _read_state(target) is not None:
        config = store_provider_config(target / "workspace", dimensions=None,
                                       add_mode=options.add_mode, review=review)
    else:
        config = agent_provider_config(add_mode=options.add_mode, review=review)
    return config.add_mode or DEFAULT_ADD_MODE, config.review_mode


def run_rebuild(
    options: RebuildOptions, environment: Mapping[str, str], *,
    keys: KeySource | None = None,
    transport: httpx.BaseTransport | None = None,
    async_transport: httpx.AsyncBaseTransport | None = None,
    progress: Progress | None = None,
) -> int:
    """Run one rebuild; the process exit code.

    Args:
        options: The rebuild options.
        environment: Provider and token variables, as for ``quotemem serve``.
        keys: Where a journal file's key is.
        transport: A provider transport replacing the network (tests).
        async_transport: Its asynchronous counterpart.
        progress: Called after each record (tests).
    """
    add_mode, review = target_review(options, environment)
    journal, entries, state = prepare(options, keys)
    plan_document = {**plan(journal, state).estimate(), "add_mode": add_mode, "review": review}
    notice(plan_document)
    if options.dry_run:
        notice({"status": "dry_run", "resume_with": "the same command without --dry-run"})
        return 0
    store = journal.header.store
    try:
        service = compose(AgentServeOptions(
            data_dir=options.target.expanduser(), port=0,
            dimensions=options.dimensions or store.embedding_dimensions,
            add_mode=options.add_mode, review=options.review,
            request_timeout_seconds=options.request_timeout_seconds,
        ), environment)
    except QuoteMemError as error:
        if error.code != "workspace_locked":
            raise
        # compose names serve's --data-dir; the rebuild's store is its --to directory.
        raise QuoteMemError("workspace_locked", TARGET_IN_USE) from None
    with RoutedClient(service.routes, transport=transport) as client:
        probed = probe_providers(client, service.routes)
        service.state.probed(ok=probed.ok, role=probed.role, code=probed.code, hint=probed.hint)
        if not probed.ok:
            raise QuoteMemError(
                "provider_unavailable",
                f"The {probed.role} provider is unavailable ({probed.code}): "
                f"{probed.hint or 'check its configuration'}; run again to resume",
            )
        final = asyncio.run(replay(service, client, entries, state,
                                   async_transport=async_transport, progress=progress))
    for name in ("journal.jsonl", "journal.key"):
        (_rebuild_dir(options.target.expanduser()) / name).unlink(missing_ok=True)
    notice({"status": "rebuilt", "records": final.total,
            "data_dir": str(options.target.expanduser()),
            "next": "Serve it with: quotemem serve --data-dir <the --to directory>"})
    return 0


TARGET_IN_USE = ("Another QuoteMem process is using the --to directory; stop it, or "
                 "rebuild into another --to directory")
"""Why a rebuild whose new store another process holds stops before anything is written."""


def describe_upgrade(code: str) -> dict[str, object] | None:
    """The fix-it hint for a store this release cannot open, else None."""
    if code not in {"workspace_binding_mismatch", "schema_mismatch"}:
        return None
    return {
        "upgrade": "This store was made by another QuoteMem release or for another "
                   "embedding model. It is never migrated in place; rebuild it from its "
                   "own evidence into a new directory: `quotemem rebuild --from "
                   "<this data dir> --to <new data dir>`. `quotemem export --data-dir "
                   "<this data dir> --out <file>` writes a backup journal first.",
    }

