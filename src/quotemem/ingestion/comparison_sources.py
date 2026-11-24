"""Scope-bound source access that retains exact row witnesses instead of plaintext."""

from __future__ import annotations

import weakref
from collections.abc import Callable, Iterator, Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING

from quotemem.core.errors import QuoteMemError
from quotemem.core.identity import ContentBindingDigest, IntegrityDigest, integrity_digest
from quotemem.core.query import ScopeV1
from quotemem.store.domain import EvidenceRowV1

if TYPE_CHECKING:
    from quotemem.store.writer import Writer


class ComparisonEvidenceView(Mapping[str, EvidenceRowV1]):
    """Read authoritative compared rows first and bounded original windows second."""

    def __init__(
        self, primary: Mapping[str, EvidenceRowV1], fallback: Mapping[str, EvidenceRowV1],
    ) -> None:
        self._primary, self._fallback = primary, fallback

    def __getitem__(self, identity: str) -> EvidenceRowV1:
        try:
            return self._primary[identity]
        except KeyError:
            return self._fallback[identity]

    def __iter__(self) -> Iterator[str]:
        return iter(dict.fromkeys((*self._primary, *self._fallback)))

    def __len__(self) -> int:
        return len(set(self._primary) | set(self._fallback))


class BorrowedComparisonEvidence(Mapping[str, EvidenceRowV1]):
    """Read one original source per access within the caller's live privacy borrow.

    Hydration records each actual complete row once, then freezes the keyset.
    Neither the collector nor this mapping keeps an Evidence body. Every later
    lookup rereads the sole Writer's row and checks its exact original witness;
    membership does not authorize access after the caller's read lifetime ends.
    """

    def __init__(self, writer: Writer, scope: ScopeV1, check: Callable[[], None]) -> None:
        self._writer = weakref.ref(writer)
        self._scope = scope.model_copy(deep=True)
        self._check = check
        self._digests: Mapping[str, ContentBindingDigest] = {}
        self._frozen = False
        # Each source's content digest, with the admission token read before
        # the complete read that proved it (see content_digest).
        self._content: dict[str, tuple[int, IntegrityDigest]] = {}

    def _owner(self) -> Writer:
        self._check()
        writer = self._writer()
        if writer is None:
            raise QuoteMemError("stale_evidence", "Comparison source owner is no longer available")
        return writer

    def observe(self, raw: dict[str, object]) -> None:
        """Retain only the digest of one source already read by actual hydration."""
        writer = self._owner()
        identity = raw.get("object_id")
        if not isinstance(identity, str) or (
            raw.get("tenant_id") != writer.tenant_id
            or (raw.get("scope_kind"), raw.get("scope_id"))
            != (self._scope.scope_kind, self._scope.scope_id)
        ):
            raise QuoteMemError("scope_denied", "Comparison source leaves its original exact scope")
        digest = writer.bind_content("row-state", raw)
        previous = self._digests.get(identity)
        if previous is not None and previous != digest:
            raise QuoteMemError("stale_evidence", "Comparison observed a changing source row")
        if self._frozen:
            if previous is None:
                raise QuoteMemError("binding_mismatch", "Frozen comparison added a new source")
            return
        assert isinstance(self._digests, dict)
        self._digests[identity] = digest

    def freeze(self) -> None:
        """Close the complete source keyset before the comparison proof is issued."""
        self._owner()
        if not self._frozen:
            self._digests = MappingProxyType(dict(self._digests))
            self._frozen = True

    def __getitem__(self, identity: str) -> EvidenceRowV1:
        writer = self._owner()
        expected = self._digests[identity]
        raw = writer.get_row("evidence", identity)
        if raw is None or writer.bind_content("row-state", raw) != expected:
            raise QuoteMemError("stale_evidence",
                                "Borrowed comparison source changed or disappeared")
        row = EvidenceRowV1.model_validate(raw)
        if row.object_id != identity or row.tenant_id != writer.tenant_id or (
            (row.scope_kind, row.scope_id) != (self._scope.scope_kind, self._scope.scope_id)
        ):
            raise QuoteMemError("scope_denied", "Borrowed comparison source has another identity")
        self._check()
        return row

    def check(self) -> None:
        """Check the caller's live borrow and the sole Writer without reading a source."""
        self._owner()

    def content_digest(self, identity: str) -> IntegrityDigest:
        """Return ``integrity_digest(self[identity])``, rereading only a possibly changed row.

        A comparison's issuance seal binds the content digest of every source.
        Every validation of an issued comparison recomputed it from a complete
        reread, keyed check and strict parse of each row, embedding included:
        about 60 ms per comparison, repeated for each reconciled position.
        The first call proves the row exactly as ``__getitem__`` does. Later
        calls reuse its digest while the Writer's mutation ledger proves the
        row unchanged since the token read before that proof; any
        mutation of the row retires it and the next call reads and proves it
        again, refusing a changed row as ``__getitem__`` does. The borrow is
        checked on every call.
        """
        writer = self._owner()
        if identity not in self._digests:
            raise KeyError(identity)
        proven = self._content.get(identity)
        if proven is not None and writer.is_borrow_unchanged(
            (("evidence", identity),), proven[0],
        ):
            self._check()
            return proven[1]
        token = writer.storage_epoch
        digest = integrity_digest(self[identity])
        self._content[identity] = token, digest
        return digest

    def __iter__(self) -> Iterator[str]:
        self._owner()
        return iter(self._digests)

    def __len__(self) -> int:
        self._owner()
        return len(self._digests)
