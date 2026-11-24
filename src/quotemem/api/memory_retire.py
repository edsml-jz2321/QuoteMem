"""Retire a memory without a successor: the rebuild's replay of an ended memory.

A memory ends when a later memory supersedes it (an update, a correction or a
retraction the Add judge decided): its memory leaves ``active`` and a ``supersedes``
link of the successor names it. When the user then asks to forget the successor,
the forget deletes the successor with its links and withholds the words that
stated it, and the old memory stays ended: the user said it stopped being
so, and forgetting what replaced it does not make it so again. Search reads such
a memory as a past state without a witnessed end and does not deliver it; its
History still answers.

A source-journal rebuild (``quotemem rebuild``) replays only what the user said,
so the successor's words are withheld there too and the old memory comes back
``active``. The journal therefore names the quoted words of each memory that
ended without a surviving successor (``quotemem.agent.journal``), and the rebuild ends
the memories of the new store that quote them through this module:

- `memories_quoting` finds the active memories of one scope most of whose quoted
  bytes lie in the named ranges: words of the ended memory, not a memory that
  merely shares a few of them;
- `MemoryRetire.retire` ends one of them, as the relation the old successor named,
  while its head is still the observed one. It is one ``entity_update`` run
  admitted with its ``entity_update:retire`` input: idempotent per (principal,
  scope, request id), refused ``operation_conflict`` when the same request id
  carries another body, and committed through the single Writer's NL (natural-language) transaction
  with the memory's head and lifecycle as its expected head. It writes the memory
  row only (``lifecycle_state``, ``retired_at``); no revision, link or name
  changes, so the memory's History is the History it had.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Final

from quotemem.core.admission import admission_identity, canonicalize_admission
from quotemem.core.entities import SupersedeRelation
from quotemem.core.errors import QuoteMemError
from quotemem.core.identity import ContentBindingDigest, new_ulid
from quotemem.core.query import ScopeV1
from quotemem.core.run_inputs import EntityUpdateKeyV1, MemoryRetireRunInputV1, MemoryRetireV1
from quotemem.ingestion.authorization import check_add_grant
from quotemem.providers.auth import AuthProvider, AuthRequest, PrincipalGrant
from quotemem.store.domain import DOMAIN_ADAPTERS
from quotemem.store.entity_ops import ENDED_LIFECYCLE
from quotemem.store.indexed import MAX_PAGE_SIZE, EvidenceRows, IndexedQuery, MemoryTableRows
from quotemem.store.read_model import StoreReadModel
from quotemem.store.writer import ExpectedHead, RowWrite, Writer, WriteTicket

RUN_KIND: Final = "entity_update"
COMMIT_MODE: Final = "single_compensatable"


@dataclass(frozen=True, slots=True)
class QuotedRange:
    """Bytes of one stored message: its request, its position and a half-open range.

    Offsets are the original message's UTF-8 bytes, as ``QuoteSpanV1`` counts them.
    """

    session_id: str
    request_id: str
    ordinal: int
    byte_start: int
    byte_end: int


@dataclass(frozen=True, slots=True)
class QuotingMemory:
    """An active memory whose head quotes a named range, and that head."""

    memory_id: str
    revision_id: str


def _pages(writer: Writer, query: IndexedQuery) -> Iterable[dict[str, Any]]:
    reader = writer.indexed_reader()
    after: str | None = None
    while True:
        page = reader.page(query, after=after, limit=MAX_PAGE_SIZE)
        yield from page
        if len(page) < MAX_PAGE_SIZE:
            return
        after = page[-1]["object_id"]


def memories_quoting(
    writer: Writer, scope: ScopeV1, ranges: Sequence[QuotedRange],
) -> list[QuotingMemory]:
    """The active memories of ``scope`` more than half of whose quoted bytes lie in ``ranges``.

    The caller holds the Writer mutex and has checked its grant. The named
    requests' Evidence parts and the scope's memories are read page by page, and each
    active memory's head quotes are measured against the ranges of their message.

    Returns:
        The memories, in memory-id order.
    """
    wanted: dict[tuple[str, str], list[QuotedRange]] = {}
    for item in ranges:
        wanted.setdefault((item.session_id, item.request_id), []).append(item)
    parts: dict[str, tuple[int, list[QuotedRange]]] = {}
    for (session_id, request_id), named in wanted.items():
        for row in _pages(writer, EvidenceRows(
            scope_kind=scope.scope_kind, scope_id=scope.scope_id,
            session_id=session_id, request_id=request_id,
        )):
            if (row["session_id"], row["request_id"]) == (session_id, request_id):
                parts[row["object_id"]] = (row["ordinal"], named)
    if not parts:
        return []
    found: list[QuotingMemory] = []
    for item in _pages(
        writer, MemoryTableRows(scope_kind=scope.scope_kind, scope_id=scope.scope_id)
    ):
        if item["lifecycle_state"] != "active":
            continue
        head = writer.get_row("revision", item["current_revision_id"])
        if head is None:
            continue
        quoted = inside = 0
        for span in head["snapshot"]["quote_spans"]:
            start, end = span["byte_start"], span["byte_end"]
            quoted += end - start
            located = parts.get(span["evidence_id"])
            if located is not None:
                inside += _covered(start, end, [
                    (named.byte_start, named.byte_end) for named in located[1]
                    if named.ordinal == located[0]])
        if inside * 2 > quoted:
            found.append(QuotingMemory(item["object_id"], item["current_revision_id"]))
    return sorted(found, key=lambda memory: memory.memory_id)


def _covered(start: int, end: int, ranges: Iterable[tuple[int, int]]) -> int:
    """How many bytes of ``[start, end)`` the union of ``ranges`` covers."""
    total, cursor = 0, start
    for low, high in sorted(ranges):
        low, high = max(low, cursor), min(high, end)
        if high > low:
            total += high - low
            cursor = high
    return total


def _run(values: dict[str, Any]) -> Any:
    """Parse one ``entity_update`` run row with the store's strict run model."""
    return DOMAIN_ADAPTERS["run"].validate_python(values)


class MemoryRetire:
    """End active memories without a successor, one ``entity_update`` run each.

    Args:
        writer: The sole Writer.
        auth: Resolves the exact scope's grant.
        publisher: The read model registered for ``writer``.
        clock: The trusted wall clock in epoch milliseconds.
    """

    def __init__(
        self, writer: Writer, auth: AuthProvider, publisher: StoreReadModel,
        *, clock: Callable[[], int],
    ) -> None:
        self._writer, self._auth, self._publisher, self._clock = writer, auth, publisher, clock

    def grant(self, scope: ScopeV1) -> PrincipalGrant:
        """The exact scope's grant, resolved outside QuoteMem locks and checked once."""
        grant = self._auth.resolve(AuthRequest(requested_scope=scope, now_ms=self._clock()))
        check_add_grant(grant, scope, self._clock())
        return grant

    def retire(
        self, *, scope: ScopeV1, request_id: str, memory_id: str, expected_revision_id: str,
        relation: SupersedeRelation,
    ) -> bool:
        """End one active memory as ``relation`` ended it, while its head is the observed one.

        Args:
            scope: The memory's exact scope.
            request_id: The caller's idempotency key within the scope.
            memory_id: The memory to end.
            expected_revision_id: Its head when the caller chose it.
            relation: ``update`` (superseded), ``correct`` (corrected) or ``retract``.

        Returns:
            True when the memory is ended by this request (now or by its earlier
            attempt); False when it was no longer active at the observed head.

        Raises:
            QuoteMemError: ``operation_conflict`` when the request id carries another body;
                the Writer's refusals.
        """
        writer = self._writer
        exact = ScopeV1.model_validate(scope.model_dump(mode="json"))
        request = MemoryRetireV1(
            op="retire", request_id=request_id, memory_id=memory_id,
            expected_revision_id=expected_revision_id, relation=relation,
        )
        grant = self.grant(exact)
        admission = canonicalize_admission({
            "v": 1, "run_kind": RUN_KIND, "commit_mode": COMMIT_MODE,
            "principal_id": grant.principal_id, **exact.model_dump(mode="json"),
            "target_kind": "scope", "target_locator": exact.scope_id,
            "raw_business_key": EntityUpdateKeyV1(request_id=request_id).model_dump(),
            "request_payload": request.model_dump(mode="json"), "options": {},
        })
        family, digest, payload = admission_identity(admission, writer.bind_content)
        with writer.mutex:
            check_add_grant(grant, exact, self._clock())
            run = writer.indexed_reader().latest_run(RUN_KIND, family)
            if run is not None and run["admission_digest"] != digest:
                raise QuoteMemError("operation_conflict", "The request id carries another body")
            if run is not None and run["status"] == "succeeded":
                return self._ended(memory_id, relation)
            item = writer.get_row("item", memory_id)
            if item is None or (item["scope_kind"], item["scope_id"]) != (
                exact.scope_kind, exact.scope_id,
            ):
                raise QuoteMemError("scope_denied", "Memory is unavailable")
            if run is None and (
                item["lifecycle_state"] != "active"
                or item["current_revision_id"] != expected_revision_id
            ):
                return False
            if run is None:
                run = self._admit(grant, exact, family, digest)
            ticket = writer.issue_ticket(run["object_id"])
            self._input(ticket, run, request, digest, payload)
            ended = self._effect(ticket, request)
            now = self._clock()
            current = writer.get_row("run", run["object_id"])
            if current is None:
                raise QuoteMemError("reference_missing", "A memory retirement lost its run")
            finished = {**current, "status": "succeeded", "updated_at": now, "finished_at": now}
            writer.write_bookkeeping(ticket, [RowWrite("run", _run(finished))])
            return ended

    def _ended(self, memory_id: str, relation: SupersedeRelation) -> bool:
        item = self._writer.get_row("item", memory_id)
        return item is not None and item["lifecycle_state"] == ENDED_LIFECYCLE[relation]

    def _admit(
        self, grant: PrincipalGrant, scope: ScopeV1, family: str, digest: ContentBindingDigest,
    ) -> dict[str, Any]:
        """Admit the request's run, empty and queued, at the scope's current epochs."""
        writer = self._writer
        lease = writer.get_control("lease:writer")
        if lease is None:
            raise QuoteMemError("lease_lost", "A memory retirement needs the active Writer")
        ticket = writer.admit_run(_run({
            "object_id": new_ulid(), "run_kind": RUN_KIND, "commit_mode": COMMIT_MODE,
            "status": "queued", "stage": "await_input", "attempt": 0,
            "idempotency_key": family, "run_generation": 0, "admission_digest": digest,
            "principal_id": grant.principal_id, "scope_kind": scope.scope_kind,
            "scope_id": scope.scope_id, "target_kind": "scope", "target_id": scope.scope_id,
            "writer_epoch": lease["value"]["writer_epoch"],
            "admitted_erase_epoch": writer.get_counter(f"erase_epoch:{scope.scope_id}"),
            "admitted_consent_epoch": writer.get_counter(f"consent_epoch:{scope.scope_id}"),
            "created_at": self._clock(),
        }))
        run = writer.get_row("run", ticket.run_id or "")
        if run is None:
            raise QuoteMemError("reference_missing", "An admitted memory retirement was lost")
        return run

    def _input(
        self, ticket: WriteTicket, run: dict[str, Any], request: MemoryRetireV1,
        digest: ContentBindingDigest, payload: ContentBindingDigest,
    ) -> None:
        """Write the run's input once and attach it; on a repeat, check the attached one."""
        writer = self._writer
        found = writer.find_logical_candidates(run["object_id"], "run_input")
        if len(found) > 1:
            raise QuoteMemError("binding_mismatch", "A memory retirement has two inputs")
        if found:
            reference, body = found[0]
            if not isinstance(body, MemoryRetireRunInputV1) or body.request != request:
                raise QuoteMemError("binding_mismatch", "A memory retirement changed its input")
        else:
            reference = writer.put_logical_artifact(ticket, "run_input", MemoryRetireRunInputV1(
                admission_digest=digest, payload_digest=payload, run_kind=RUN_KIND,
                source="retire", business_key=EntityUpdateKeyV1(request_id=request.request_id),
                request=request,
            ), created_at=self._clock())
        if run["input_ref"] is None:
            now = self._clock()
            attached = {**run, "input_ref": reference, "status": "running", "stage": "commit",
                        "started_at": now, "updated_at": now}
            writer.write_bookkeeping(ticket, [RowWrite("run", _run(attached))])
        elif run["input_ref"] != reference:
            raise QuoteMemError("binding_mismatch", "A memory retirement attached another input")

    def _effect(self, ticket: WriteTicket, request: MemoryRetireV1) -> bool:
        """End the memory, unless an earlier attempt of this run already did."""
        writer = self._writer
        item = writer.get_row("item", request.memory_id)
        ended = ENDED_LIFECYCLE[request.relation]
        if item is None:
            raise QuoteMemError("scope_denied", "Memory is unavailable")
        if item["lifecycle_state"] == ended and item["current_revision_id"] == (
            request.expected_revision_id
        ):
            return True
        if item["lifecycle_state"] != "active" or item["current_revision_id"] != (
            request.expected_revision_id
        ):
            return False
        now = self._clock()
        sequence = writer.get_counter("commit_seq") + 1
        writer.commit_rows(
            ticket, [RowWrite("item", DOMAIN_ADAPTERS["item"].validate_python({
                **item, "lifecycle_state": ended, "retired_at": now, "commit_seq": sequence,
                "updated_at": now,
            }))],
            heads=[ExpectedHead(request.memory_id, request.expected_revision_id, "active")],
            updated_at=now,
            publish=lambda committed: self._publisher.publish_commit(writer, committed),
        )
        return True
