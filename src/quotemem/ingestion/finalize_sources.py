"""Read the exact frozen finalize input within its existing held content borrow.

The fallback ordering walks bounded SQL pages repeatedly instead of retaining
an entire scope or session. It favors bounded retained content over scan cost;
no throughput or normal-load latency claim follows from this implementation.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple, cast

from quotemem.core.controls import ControlRowV1, HeldFinalizeLeaseV1
from quotemem.core.errors import QuoteMemError
from quotemem.core.identity import ArtifactId, base64url, canonical_json, control_id
from quotemem.core.query import ScopeV1
from quotemem.core.request_budget import check_request_budget
from quotemem.core.run_inputs import FinalizeOwnerRunInputV1
from quotemem.ingestion.authorization import check_add_grant
from quotemem.ingestion.finalize_heads import (
    FinalizeEvidenceDigest,
    finalize_input_heads_digest,
    read_finalize_evidence_digest,
)
from quotemem.ingestion.request_order import load_request_order
from quotemem.ingestion.segment import (
    CheckpointMember,
    checkpoint_partitions,
    message_key,
    message_order,
)
from quotemem.providers.auth import AuthProvider, AuthRequest, PrincipalGrant
from quotemem.service.cost_attribution import attribute_cost
from quotemem.store.domain import EvidenceRowV1
from quotemem.store.finalize_ticket import FinalizeLeaseTicket
from quotemem.store.indexed import FinalizeEvidenceCursor, FinalizeEvidenceQuery
from quotemem.store.read_model import SnapshotReadGuard
from quotemem.store.row_cache import (
    MEMBERSHIP_ANY,
    MISSING,
    membership_family,
    membership_key,
    note_unprovable_read,
    suspended_reads,
)
from quotemem.store.writer import Writer


@dataclass(frozen=True)
class FinalizeSession:
    """One exact external session identity, with no retained source contents."""

    session_hash: str
    session_id: str


class BoundSource(NamedTuple):
    """One live source's canonical bytes, row, row-state digest and byte interval.

    ``body`` and ``digest`` are derived from the row the interval came from,
    which is not necessarily ``row``: the two derived values are memoized while
    the ledger proves the row unchanged, and ``row`` is read live on every
    call so its scope, state and expiry decisions run against the current
    clock. A consumer that relies on their agreement compares the interval.
    """

    body: str
    row: EvidenceRowV1
    digest: str
    byte_start: int
    byte_end: int


_HEADER_MEMO_ENTRIES = 4096
"""Eligibility projections one source consumer keeps: a few fixed fields each."""

_HEADER_PAGE = 512
"""Scope Evidence lifetime headers read per page for the expiry bound."""

_ROW_MEMO_ENTRIES = 128
"""Rows one source consumer keeps parsed; each carries its embedding vector (about 65 KiB).

This memo is private to one consumer, and a cohort runs one consumer per
concurrent invocation, so every entry is paid for again by each of them: at
1,024 the twenty-five consumers of a normal-load run could hold
1.6 GB of the same rows. A miss falls through to the Writer's shared row cache,
which the consumers do share and which now holds the scope, so the entries kept
here need only cover the unit being consumed.
"""

class FinalizeSources:
    """Consume the selected input using the sole Writer and its issued held pin.

    Construct and consume inside ``guard.consume_operation``. The caller retains
    the owner executor lock for the complete operation and must not export rows
    or iterators beyond the callback. This reader creates no storage or provider
    effects. Child publications may advance scope_seq but cannot change the
    input protected by this same held owner/cycle.
    An explicitly supplied AuthProvider renews expired borrowed grants outside
    Writer locks. A caller-supplied fixed grant is never automatically renewed.
    """

    def __init__(
        self, writer: Writer, ticket: FinalizeLeaseTicket, guard: SnapshotReadGuard,
        grant: PrincipalGrant, *, clock: Callable[[], int], auth: AuthProvider | None = None,
    ) -> None:
        self.writer, self.ticket, self.guard = writer, ticket, guard
        self.grant, self.clock, self.auth = grant, clock, auth
        self._checked_generation: int | None = None
        self._checked_token: int | None = None
        self._owner_parent_run_id: str = ""
        self._page_memo: dict[
            tuple[str | None, str | None],
            tuple[int, tuple[EvidenceRowV1, ...], tuple[tuple[str, str], ...]],
        ] = {}
        self._member_memo: tuple[str, int, tuple[CheckpointMember, ...]] | None = None
        self._session_memo: tuple[int, tuple[FinalizeSession, ...]] | None = None
        self._row_memo: dict[str, tuple[int, EvidenceRowV1]] = {}
        with writer.mutex:
            owner, lease, _ = writer.read_finalize_fences(
                ticket,
            )
            if not isinstance(lease.value, HeldFinalizeLeaseV1) or owner["input_ref"] is None:
                raise QuoteMemError("input_required", "Finalize requires its selected held input")
            self.scope = ScopeV1(scope_kind=owner["scope_kind"], scope_id=owner["scope_id"])
            guard.check_operation(writer, self.scope)
            check_add_grant(grant, self.scope, clock(), owner["principal_id"])
            self.input_ref = ArtifactId(owner["input_ref"])
            self.principal_id = owner["principal_id"]
            self.input = FinalizeOwnerRunInputV1.model_validate(
                writer.read_bound_input(owner),
            )
            self._ticket_identity = (
                ticket.scope_id, ticket.lease_generation, ticket.owner_run_id, ticket.writer_epoch,
            )
        self.reader = writer.indexed_reader()
        self.query = FinalizeEvidenceQuery(
            tenant_id=writer.tenant_id, scope=self.scope, freeze_time=self.input.freeze_time,
            receipt_high_watermark=self.input.receipt_high_watermark,
        )
        self.check()

    def _registered(self) -> None:
        saved = self.writer._finalize_tickets.get(  # pyright: ignore[reportPrivateUsage]
            self.ticket.owner_run_id,
        )
        if saved is not self.ticket or self.ticket.state != "held" or (
            self.ticket.scope_id, self.ticket.lease_generation,
            self.ticket.owner_run_id, self.ticket.writer_epoch,
        ) != self._ticket_identity:
            raise QuoteMemError("lease_lost", "Finalize source ticket lost its Writer registration")

    @attribute_cost("proof.finalize_sources")
    def check(self) -> None:
        """Check the active borrow and refresh durable identity after every write.

        This gate's reads are never recorded as dependencies of a memoized
        read (``suspended_reads``): every reuse runs this check again.

        Ticket identity, authorization and guard cancellation are checked every
        time. The fixed owner and lease rows are reread only when an in-process
        mutation named one of them since the admission token this borrow last
        read them at, which a heartbeat or a takeover does and an
        unrelated Add does not; process ownership is probed on the path that
        rereads nothing. The reread goes through ``Writer.refresh_borrowed``
        with those two keys, so it runs through active Writer critical sections
        and is accepted on the same per-key proof, and only bounded torn
        attempts reread under the Writer. Concurrent owners must not queue
        behind the Writer merely to learn that nothing changed, nor to learn
        that only their lease row did.

        The guard is checked first, before any other step, and again before
        every return. On the path that rereads nothing this thread's
        remaining steps only read (the grant, the ticket registration, the
        per-key proof, the owner probe), but another thread may revoke the
        borrow inside that window: a publisher failing closed and discarding
        the finalize baselines, a publication, a privacy quiesce or a close,
        and the owner probe may stat the filesystem meanwhile. The trailing
        check refuses those; while nothing moved it is the guard's lock-free
        repeat (``SnapshotReadGuard.check_operation``), so it costs no lock.
        """
        with suspended_reads():
            self._check_borrow()

    def _check_borrow(self) -> None:
        check_request_budget()
        writer = self.writer
        self.guard.check_operation(writer, self.scope)
        now = self.clock()
        if self.auth is not None and now >= self._renewal_due():
            # A grant is renewed once half its lifetime has passed and this
            # check runs outside the Writer, so the checks that later run
            # under a lock, which cannot renew, always hold a grant with at
            # least half a lifetime left. An expired grant under a lock is a
            # failure of that invariant, never renewed in place.
            if writer.mutex._is_owned():  # pyright: ignore[reportPrivateUsage]
                if now >= self.grant.expires_at:
                    raise QuoteMemError(
                        "authorization_expired", "Finalize must renew outside Writer locks",
                    )
            else:
                renewed = self.auth.resolve(AuthRequest(requested_scope=self.scope, now_ms=now))
                check_add_grant(renewed, self.scope, self.clock(), self.principal_id)
                self.guard.check_operation(writer, self.scope)
                self.grant = renewed
        check_add_grant(self.grant, self.scope, self.clock(), self.principal_id)
        self._registered()
        # Fixture-constructed sources may bypass __init__; they start unchecked.
        checked_token = self.__dict__.get("_checked_token")
        if checked_token is not None and writer.is_borrow_unchanged(
            self._fixed_keys(), checked_token,
        ):
            # Unrelated writes moved the reservation generation, but no
            # in-process mutation named the owner or lease row since this
            # borrow last read them: nothing to reread, no queueing
            # behind the Writer.
            writer.probe_storage_owner()
            self.guard.check_operation(writer, self.scope)
            return
        token = writer.storage_epoch

        def refresh() -> None:
            self._registered()
            owner, lease, _ = writer.read_finalize_fences(
                self.ticket,
            )
            self._owner_parent_run_id = str(owner.get("parent_run_id") or "")
            if not isinstance(lease.value, HeldFinalizeLeaseV1) or (
                owner["input_ref"] != self.input_ref
                or owner["principal_id"] != self.principal_id
            ):
                raise QuoteMemError("lease_lost", "Finalize source selection lost its held owner")

        self._checked_generation = writer.refresh_borrowed(
            self._checked_generation, refresh, keys=self._fixed_keys(),
        )
        self._checked_token = token
        self.guard.check_operation(writer, self.scope)

    def _renewal_due(self) -> int:
        """Return the clock value from which an unlocked check renews the grant."""
        grant = self.grant
        return grant.expires_at - max(1, (grant.expires_at - grant.issued_at) // 2)

    def _fixed_keys(self) -> tuple[tuple[str, str], ...]:
        """Name every row the refresh reads, so the per-key proof covers all of them.

        The refresh runs ``read_finalize_fences``, which reads the owner Run
        and the finalize lease and re-admits the ticket: the Writer lease, the
        scope registration, tombstone and the erase and consent epoch counters,
        and the owner's parent Run when it has one.
        """
        # The owner's parent Run, when the refresh has seen one, is read by
        # the execution fence check; the refresh records it without a read.
        parent = self.__dict__.get("_owner_parent_run_id") or ""
        memo = self.__dict__.get("_fixed_keys_memo")
        if memo is not None and memo[0] == parent:
            return memo[1]
        scope_id = self.ticket.scope_id
        keys: list[tuple[str, str]] = [
            ("run", self.ticket.owner_run_id),
            ("control", control_id(f"lease:finalize:{scope_id}")),
            ("control", control_id("lease:writer")),
            ("control", control_id(f"scope:{scope_id}")),
            ("control", control_id(f"tomb:{scope_id}")),
            ("control", control_id(f"erase_epoch:{scope_id}")),
            ("control", control_id(f"consent_epoch:{scope_id}")),
        ]
        if parent:
            keys.append(("run", parent))
        # The ticket's scope and owner never change for this borrow; only the
        # parent Run can, so the seven derived Control ids are computed once
        # per parent rather than on every per-row check.
        fixed = tuple(keys)
        self.__dict__["_fixed_keys_memo"] = (parent, fixed)
        return fixed

    def _page(
        self, after: FinalizeEvidenceCursor | None, session_id: str | None = None,
    ) -> tuple[EvidenceRowV1, ...]:
        """Read one evidence page, reusing it while no in-process row mutation ran.

        The scope's frozen evidence is paged once per session unit, so the
        plan-chain consumer re-reads the same few pages many times per owner.
        A page is reused only while the Writer's storage epoch is
        unchanged, every caller still runs ``check`` around it, and each reuse
        hands out fresh copies so no consumer shares a row object. A caller
        wanting one session narrows the statement instead of discarding the
        other sessions' rows after decoding their embedding vectors.
        """
        from quotemem.ingestion.finalize_receipts import inspect_finalize_receipts

        census = inspect_finalize_receipts(
            self.writer, self.scope, receipt_high_watermark=self.input.receipt_high_watermark,
            check=self.check,
        )
        census.validate(self.writer, self.check)
        key = (
            session_id,
            None if after is None else canonical_json(after.model_dump(mode="json")),
        )
        epoch = self.writer.storage_epoch
        memo = self._page_memo.get(key)
        if memo is not None and self._memo_current(memo[0], memo[2]):
            census.validate(self.writer, self.check)
            return tuple(row.model_copy() for row in memo[1])
        page = self.reader.finalize_evidence_page(self.query, after=after, session_id=session_id)
        census.validate(self.writer, self.check)
        if memo is None and len(self._page_memo) >= 64:
            self._page_memo.clear()
        # Excluded message peers and receipt membership also select this page.
        # Include their original keys, including rows a later delete can retire.
        keys = tuple(sorted({self._scope_seq_key(), *census.keys,
                             *(("evidence", row.object_id) for row in page)}))
        self._page_memo[key] = (epoch, tuple(row.model_copy() for row in page), keys)
        return page

    def _memo_current(self, token: int, keys: tuple[tuple[str, str], ...]) -> bool:
        """Reuse a memo taken at ``token`` while none of ``keys`` was mutated since."""
        writer = self.writer
        return writer.storage_epoch == token or writer.is_borrow_unchanged(keys, token)

    def _scope_seq_key(self) -> tuple[str, str]:
        return ("control", control_id(f"scope_seq:{self.scope.scope_id}"))

    def rows(self, session_id: str | None = None) -> Iterator[EvidenceRowV1]:
        """Stream complete fixed-time eligible rows through bounded actual SQL pages."""
        self.check()
        after: FinalizeEvidenceCursor | None = None
        while True:
            self.check()
            page = self._page(after, session_id)
            self.check()
            if not page:
                return
            for row in page:
                self.check()
                if row.session_id != session_id and session_id is not None:
                    raise QuoteMemError("parity_mismatch",
                                        "Finalize page left its selected session")
                yield row
                self.check()
            after = FinalizeEvidenceCursor.from_evidence(page[-1])

    def verify_heads(self) -> FinalizeEvidenceDigest:
        """Prove the frozen input heads from the fixed-time eligible Evidence rows."""
        self.check()
        evidence = read_finalize_evidence_digest(
            self.writer, self.scope, self.guard, freeze_time=self.input.freeze_time,
            receipt_high_watermark=self.input.receipt_high_watermark,
        )
        self.check()
        if (
            evidence.count, evidence.digest, finalize_input_heads_digest(self.writer, evidence),
        ) != (
            self.input.evidence_count, self.input.evidence_rows_digest,
            self.input.input_heads_digest,
        ):
            raise QuoteMemError("parity_mismatch", "Held finalize input heads changed after freeze")
        return evidence

    def joint_input_high_watermark(self) -> int:
        """Keep the original receipt boundary across a legitimately rebased cold lease."""
        self.check()
        raw = self.writer.get_control(f"lease:finalize:{self.scope.scope_id}")
        if raw is None:
            raise QuoteMemError("lease_lost", "Joint sources lost their held baseline")
        lease = ControlRowV1.model_validate(raw).value
        if not isinstance(lease, HeldFinalizeLeaseV1) or (
            lease.owner_run_id, lease.lease_generation, lease.writer_epoch,
            lease.baseline_snapshot_generation, lease.baseline_root_identity,
        ) != (
            self.ticket.owner_run_id, self.ticket.lease_generation, self.ticket.writer_epoch,
            self.guard.metadata.generation, self.guard.metadata.root_identity,
        ):
            raise QuoteMemError("binding_mismatch", "Joint sources changed their held baseline")
        if self.input.receipt_high_watermark > lease.baseline_snapshot_generation:
            raise QuoteMemError("binding_mismatch",
                                "Joint receipt boundary exceeds the held baseline")
        return self.input.receipt_high_watermark

    def _session_identities(
        self, *, keys: list[tuple[str, str]] | None = None,
    ) -> Iterator[FinalizeSession]:
        for row in self.rows():
            if keys is not None:
                keys.append(("evidence", row.object_id))
            key = base64url(hashlib.sha256(row.session_id.encode("utf-8")).digest())[:22]
            yield FinalizeSession(key, row.session_id)

    def session_identity(self, session_hash: str) -> FinalizeSession:
        """Resolve an original plan session hash to its one session identity."""
        found = self._historical_sessions().get(session_hash)
        if found is None:
            raise QuoteMemError("reference_missing",
                                "Frozen session lost all original identity proofs")
        return found

    def _historical_sessions(self) -> dict[str, FinalizeSession]:
        """Map every session hash the scope can still prove to its one identity.

        A frozen plan names one session per record, and consuming it resolved
        each of them by streaming the scope's rows again: a plan over 485
        sessions read 10,611 Evidence rows per record, once per cohort child.
        The map is a function of the same rows the ordered
        sessions are, and is memoized under the same
        membership proof, so a row joining the scope retires it exactly as it
        retires them. A hash naming two identities is refused
        while the map is built, as it was while the sessions were.
        """
        receipt_keys = self._receipt_keys()
        scan_memo = self.writer.scan_memo()
        memo_key = (
            "finalize-session-identities", self.scope.scope_kind, self.scope.scope_id,
            self.input.freeze_time, self.input.receipt_high_watermark,
        )
        memoized = scan_memo.lookup(memo_key)
        if memoized is not MISSING:
            return cast(dict[str, FinalizeSession], memoized)
        epoch = self.writer.storage_epoch
        keys: list[tuple[str, str]] = [
            *receipt_keys,
            MEMBERSHIP_ANY, membership_family("evidence"),
            membership_key("evidence", f"{self.scope.scope_kind}:{self.scope.scope_id}"),
        ]
        found: dict[str, FinalizeSession] = {}
        for candidate in self._session_identities(keys=keys):
            known = found.get(candidate.session_hash)
            if known is None:
                found[candidate.session_hash] = candidate
            elif known != candidate:
                raise QuoteMemError("binding_mismatch", "Finalize session hash has two identities")
        scan_memo.store(memo_key, epoch, keys, found)
        return found

    def sessions(self) -> Iterator[FinalizeSession]:
        """Yield exact session hashes in plan order from one bounded identity pass.

        One pass over the scope's rows retains one
        identity per distinct session hash (never a row inventory), rejects a
        hash that names two session identities, and orders the hashes. The
        ordered identities are reused while the Writer's storage epoch is
        unchanged and discarded if it moved during the pass; each
        yielded session still runs ``check`` first.
        """
        for session in self._ordered_sessions():
            self.check()
            yield session

    def _ordered_sessions(self) -> tuple[FinalizeSession, ...]:
        receipt_keys = self._receipt_keys()
        memo = self._session_memo
        epoch = self.writer.storage_epoch
        if memo is not None and memo[0] == epoch:
            return memo[1]
        scan_memo = self.writer.scan_memo()
        memo_key = (
            "finalize-sessions", self.scope.scope_kind, self.scope.scope_id,
            self.input.freeze_time, self.input.receipt_high_watermark,
        )
        memoized = scan_memo.lookup(memo_key)
        if memoized is not MISSING:
            ordered = cast(tuple[FinalizeSession, ...], memoized)
            self._session_memo = epoch, ordered
            return ordered
        # The identities depend on which eligible rows the scope holds and on
        # each row's state; a row's session never changes.
        keys: list[tuple[str, str]] = [
            *receipt_keys,
            MEMBERSHIP_ANY, membership_family("evidence"),
            membership_key("evidence", f"{self.scope.scope_kind}:{self.scope.scope_id}"),
        ]
        by_hash: dict[str, FinalizeSession] = {}
        for candidate in self._session_identities(keys=keys):
            known = by_hash.get(candidate.session_hash)
            if known is None:
                by_hash[candidate.session_hash] = candidate
            elif known.session_id != candidate.session_id:
                raise QuoteMemError("binding_mismatch", "Finalize session hash has two identities")
        ordered = tuple(by_hash[key] for key in sorted(by_hash))
        scan_memo.store(memo_key, epoch, keys, ordered)
        if self.writer.storage_epoch == epoch:
            self._session_memo = epoch, ordered
        return ordered

    def _receipt_keys(self) -> tuple[tuple[str, str], ...]:
        """Revalidate complete receipt membership before an ordering memo is reused."""
        from quotemem.ingestion.finalize_receipts import inspect_finalize_receipts

        census = inspect_finalize_receipts(
            self.writer, self.scope, receipt_high_watermark=self.input.receipt_high_watermark,
            check=self.check,
        )
        census.validate(self.writer, self.check)
        return census.keys

    def _anchors(self, session_id: str) -> tuple[
        int | None, dict[str, tuple[tuple[int, int, str], int | None]],
        dict[str, tuple[int, bool]], tuple[tuple[str, str], ...],
    ]:
        """Read one session's frozen receipt order (``load_request_order``).

        Each ``request_id`` anchors on its receipt-proven first part and that
        part's signed observation, and carries its receipt sequence and whether
        any of its parts has a source time; the session time is the earliest
        source time among its requests. A session with no eligible row has none.
        """
        if next(self.rows(session_id), None) is None:
            return None, {}, {}, ()
        original, extra_keys = load_request_order(
            self.writer, self.scope, session_id, at_time=self.input.freeze_time,
            input_high_watermark=self.joint_input_high_watermark(),
        )
        anchors = {key: (value.first, value.chunk_event_time) for key, value in original.items()}
        receipts = {key: (value.received_commit_seq, value.source_time_present)
                    for key, value in original.items()}
        event_time = min((value.earliest_source_time for value in original.values()
                          if value.earliest_source_time is not None), default=None)
        return event_time, anchors, receipts, extra_keys

    def _ordered_members(self, session_id: str) -> tuple[CheckpointMember, ...]:
        """Order one session's complete frozen descriptors in one bounded pass pair.

        The first pass confirms that the session has an eligible row, whose
        anchors then come from its frozen receipt order; the second binds each
        row and orders the result by ``message_order``. Both passes walk the
        same bounded SQL pages ``rows`` already walks, so no scope or
        session contents are retained beyond the ordered descriptors and one
        anchor per request. The result is reused while the Writer's storage
        epoch is unchanged, which proves that no in-process row
        mutation ran and that another pass would read the same rows; an
        ordering that overlapped a mutation is never kept, exactly as the row
        cache refuses a read that overlapped one. Callers still run ``check``
        around every descriptor they consume.
        """
        receipt_keys = self._receipt_keys()
        memo = self._member_memo
        epoch = self.writer.storage_epoch
        if memo is not None and memo[:2] == (session_id, epoch):
            return memo[2]
        scan_memo = self.writer.scan_memo()
        binding_memo = self.writer.binding_memo()
        memo_key = (
            "finalize-members", self.scope.scope_kind, self.scope.scope_id,
            self.input.freeze_time, self.input.receipt_high_watermark, session_id,
        )
        memoized = scan_memo.lookup(memo_key)
        if memoized is not MISSING:
            ordered = cast(tuple[CheckpointMember, ...], memoized)
            self._member_memo = session_id, epoch, ordered
            return ordered
        # Every child of a cohort and every plan step built its own source
        # consumer and walked the whole scope twice for one session's members
        # (842 MB of scope pages in seventeen minutes). The ordered
        # descriptors are a function of this session's eligible rows alone:
        # a row joining the scope retires the membership key, a row changing
        # state retires its own key.
        keys: list[tuple[str, str]] = [
            *receipt_keys,
            MEMBERSHIP_ANY, membership_family("evidence"),
            membership_key("evidence", f"{self.scope.scope_kind}:{self.scope.scope_id}"),
        ]
        event_time, anchors, receipts, extra_keys = self._anchors(session_id)
        keys.extend(extra_keys)
        partitions = checkpoint_partitions(
            {session_id: event_time},
            {(session_id, request): (*value[0], value[1]) for request, value in anchors.items()},
            {(session_id, request): value for request, value in receipts.items()},
        )
        members: list[CheckpointMember] = []
        for row in self.rows(session_id):
            keys.append(("evidence", row.object_id))
            anchor = anchors.get(row.request_id)
            if anchor is None:
                raise QuoteMemError("parity_mismatch", "Finalize chunk disappeared from held input")
            key = message_key(
                row, session_event_time=event_time, chunk_event_time=anchor[1],
                coordinate_version=2,
                received_commit_seq=receipts[row.request_id][0],
                source_time_present=receipts[row.request_id][1],
                receipt_partition_seq=partitions.get((session_id, row.request_id), 0),
            )
            dumped = row.model_dump(mode="json")
            digest = self.writer.bind_content("row-state", dumped)
            # This pass already canonicalizes and rekeys every member row, so
            # it publishes both under the row's key for the plan consumers
            # that would otherwise recompute them (``binding``).
            binding_memo.store(
                ("row-binding", row.object_id), epoch,
                (("evidence", row.object_id),),
                (canonical_json(dumped), digest, row.byte_start, row.byte_end),
            )
            members.append(CheckpointMember(row.object_id, digest, canonical_json(key)))
        members.sort(key=lambda member: message_order(member.key))
        ordered = tuple(members)
        scan_memo.store(memo_key, epoch, keys, ordered)
        if self.writer.storage_epoch == epoch:
            self._member_memo = session_id, epoch, ordered
        return ordered

    def members(self, session: FinalizeSession) -> Iterator[CheckpointMember]:
        """Yield complete descriptors in the existing stable semantic message order.

        Bounded page scans retain one anchor per request and this session's
        ordered descriptors, never a scope inventory or any source contents.
        Each key carries its request's frozen receipt order
        (``MessageRangeKeyV2``), as ``checkpoint_order`` derives it.
        Every descriptor is still subject to current usability when rehydrated.
        """
        expected = base64url(hashlib.sha256(session.session_id.encode("utf-8")).digest())[:22]
        if session.session_hash != expected:
            raise QuoteMemError("binding_mismatch",
                                "Finalize session identity differs from its hash")
        for member in self._ordered_members(session.session_id):
            self.check()
            yield member

    def _row(self, identifier: str) -> EvidenceRowV1 | None:
        """Read one evidence row, reusing it while no in-process row mutation ran.

        Planning rehydrates the same session members once per candidate
        segmentation step, and each read reparses the row's embedding vector.
        A row is reused only while the Writer's storage epoch is
        unchanged, which proves that another read would return the same row,
        and each reuse hands out a fresh copy so no consumer shares an object.
        The caller still applies its scope, state and expiry decisions to every
        returned row against the current clock.
        """
        epoch = self.writer.storage_epoch
        memo = self._row_memo.get(identifier)
        if memo is not None and self._memo_current(memo[0], (("evidence", identifier),)):
            return memo[1].model_copy()
        raw = self.writer.get_row("evidence", identifier)
        if raw is None:
            return None
        row = EvidenceRowV1.model_validate(raw)
        if self.writer.storage_epoch == epoch:
            if memo is None and len(self._row_memo) >= _ROW_MEMO_ENTRIES:
                # Forget the oldest entry only; clearing the whole memo made a
                # scope with more than 256 live rows reload every one of them
                # on each candidate segmentation step.
                self._row_memo.pop(next(iter(self._row_memo)))
            self._row_memo[identifier] = epoch, row.model_copy()
        return row

    def load_many(self, identifiers: Sequence[str]) -> None:
        """Warm the row memo for several sources with one batched storage read.

        A frozen plan or a segmentation step touches every member of a session
        in order; loading them one at a time made each row a separate storage
        read queued behind the Writer and every other reader (62k
        single-row reads, 2.9 GB of rows, 59% of all read wait). The misses
        are read together through ``Writer.get_rows`` and memoized exactly as
        ``_row`` memoizes a single row. Fixed eligibility headers are checked
        before the batched content read; ``load`` rechecks the copied row.
        """
        self.check()
        wanted: list[str] = []
        for identifier in identifiers:
            if identifier in wanted:
                continue
            if self._live_header(identifier) is None:
                continue
            memo = self._row_memo.get(identifier)
            if memo is None or not self._memo_current(memo[0], (("evidence", identifier),)):
                wanted.append(identifier)
        if not wanted:
            return
        epoch = self.writer.storage_epoch
        raws = self.writer.get_rows("evidence", wanted)
        if self.writer.storage_epoch != epoch:
            return
        for identifier, raw in zip(wanted, raws, strict=True):
            if raw is None:
                continue
            row = EvidenceRowV1.model_validate(raw)
            header = self._live_header(identifier)
            if header is None or row.model_dump(mode="json", include=set(header)) != header:
                raise QuoteMemError("stale_evidence", "Finalize source changed after its header")
            while identifier not in self._row_memo and len(self._row_memo) >= _ROW_MEMO_ENTRIES:
                self._row_memo.pop(next(iter(self._row_memo)))
            self._row_memo[identifier] = epoch, row
        self.check()

    def _live_header(self, identifier: str) -> dict[str, Any] | None:
        """Reject unavailable content using only its fixed eligibility projection."""
        header = self._eligibility_header(identifier)
        if header is None:
            return None
        if (header["object_id"], header["tenant_id"], header["scope_kind"], header["scope_id"]) != (
            identifier, self.writer.tenant_id, self.scope.scope_kind, self.scope.scope_id,
        ):
            raise QuoteMemError("scope_denied", "Finalize source differs from its exact scope")
        if header["state"] != "active" or (
            header["expires_at"] is not None and header["expires_at"] <= self.clock()
        ):
            raise QuoteMemError("stale_evidence", "Finalize source is no longer usable")
        return header

    def evidence_expiry_bound(self) -> int | None:
        """Return the first future expiry among every Evidence row of this scope.

        Every source, comparison and review check of this owner consults only
        Evidence of its own scope, and a row past its expiry stays so until the
        row changes. A proof reused while no row changed is therefore refused
        by no expiry before this time. Only lifetime headers are paged, never a
        body: a row already expired may not be hydrated again. The bound is
        kept while no row mutation happened since it was read except this
        owner's own lease heartbeats (``Writer.is_heartbeat_only_since``),
        so it is current whenever such a reuse is.
        """
        writer = self.writer
        owner = self.ticket.owner_run_id
        held = cast(tuple[int, int, int | None] | None, self.__dict__.get("_expiry_bound"))
        now = self.clock()
        if held is not None and held[1] <= now and (
            held[2] is None or now < held[2]
        ) and writer.is_heartbeat_only_since(owner, held[0]):
            return held[2]
        token = writer.storage_epoch
        bound: int | None = None
        after: str | None = None
        with suspended_reads():
            watermark = writer.get_counter("commit_seq")
            while True:
                page = self.reader.scope_source_headers(
                    self.scope, high_watermark=watermark, after=after, limit=_HEADER_PAGE,
                )
                for header in page:
                    expires_at = header["expires_at"]
                    if header["state"] == "active" and expires_at is not None and (
                        expires_at > now
                    ):
                        bound = expires_at if bound is None else min(bound, expires_at)
                if len(page) < _HEADER_PAGE:
                    break
                after = page[-1]["object_id"]
        if writer.is_heartbeat_only_since(owner, token):
            self.__dict__["_expiry_bound"] = token, now, bound
        return bound

    def _eligibility_header(self, identifier: str) -> dict[str, Any] | None:
        """Read one source's eligibility projection, again only after its row changed.

        Every authority check of a source call loads each of its sources, and
        each load reread this projection with its own storage read: a sixth of
        an a7 consume walk. The projection is reused while no in-process
        mutation named the row since it was read, the proof the row
        memo uses; the caller's scope, state and expiry decisions still run on
        every use against the current clock. A reuse marks active read records
        as the storage read does.
        """
        # A consumer built around __init__ (a fixture premise) starts empty too.
        headers = cast(
            dict[str, tuple[int, dict[str, Any]]],
            self.__dict__.setdefault("_header_memo", {}),
        )
        memo = headers.get(identifier)
        if memo is not None and self._memo_current(memo[0], (("evidence", identifier),)):
            note_unprovable_read()
            return dict(memo[1])
        token = self.writer.storage_epoch
        header = self.reader.evidence_eligibility_header(identifier)
        if header is not None:
            while identifier not in headers and len(headers) >= _HEADER_MEMO_ENTRIES:
                headers.pop(next(iter(headers)))
            headers[identifier] = token, dict(header)
        return header

    def binding(self, identifier: str) -> tuple[str, EvidenceRowV1, str]:
        """Return one live source's canonical bytes, its row and its row-state digest."""
        bound = self.bound(identifier)
        return bound.body, bound.row, bound.digest

    def bound(self, identifier: str) -> BoundSource:
        """Return one live source's bytes, row, digest and the interval those bytes carry.

        Both derived values are pure functions of the stored row, and every
        consumer of a frozen plan recomputes them for the same rows: ordering
        a session's descriptors, verifying a frozen unit and segmenting a
        candidate step each canonicalized the row (0.61 ms) and rekeyed its
        row-state digest (about 1.2 ms) for every member, and one cohort runs
        sixteen children over the same plan. They are memoized under the row's
        own cache key, so any mutation of that row retires them,
        and the scope, state and expiry decisions of ``load`` still run on
        every call.

        The interval is memoized beside them because it is the one thing a
        consumer compares between the memoized bytes and the live row. A
        consumer that seeds a window with the live row instead of reparsing
        the bytes keeps that comparison by reading it here; two integers cost
        nothing beside a 41 KB body.
        """
        memo = self.writer.binding_memo()
        key = ("row-binding", identifier)
        memoized = memo.lookup(key)
        if memoized is not MISSING:
            body, digest, start, end = cast(tuple[str, str, int, int], memoized)
            return BoundSource(body, self.load(identifier), digest, start, end)
        token = self.writer.storage_epoch
        row = self.load(identifier)
        dumped = row.model_dump(mode="json")
        value = (
            canonical_json(dumped), self.writer.bind_content("row-state", dumped),
            row.byte_start, row.byte_end,
        )
        memo.store(key, token, (("evidence", identifier),), value)
        return BoundSource(value[0], row, value[1], value[2], value[3])

    def load(self, identifier: str) -> EvidenceRowV1:
        """Rehydrate one same-scope live source before rendering or segmentation."""
        self.check()
        header = self._live_header(identifier)
        if header is None:
            raise QuoteMemError("reference_missing", "Finalize source is missing")
        row = self._row(identifier)
        if row is None:
            raise QuoteMemError("reference_missing", "Finalize source is missing")
        if row.model_dump(mode="json", include=set(header)) != header:
            raise QuoteMemError("stale_evidence", "Finalize source changed after its header")
        if row.object_id != identifier or row.tenant_id != self.writer.tenant_id or (
            row.scope_kind, row.scope_id
        ) != (self.scope.scope_kind, self.scope.scope_id):
            raise QuoteMemError("scope_denied", "Finalize source differs from its exact scope")
        if row.state != "active" or row.expires_at is not None and row.expires_at <= self.clock():
            raise QuoteMemError("stale_evidence", "Finalize source is no longer usable")
        self.check()
        return row
