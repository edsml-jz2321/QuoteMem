"""Reference fences for pending recall preparations, invalidated by Writer mutations.

The registry retains only active fences and their exact storage keys. It holds
no row content, prepared rows or storage authority.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

from aml.core.errors import AMLError

type PendingRecallReferenceKey = tuple[str, str]


@dataclass(frozen=True, slots=True, eq=False)
class PendingRecallReferenceFence:
    """Opaque identity issued for one callback's actual reference preparation.

    The issuer's active membership, not these public object fields, determines
    validity. Copying or reconstructing a fence never copies its registration.
    No content, row snapshot, keys or storage authority lives in this handle.
    """

    _issuer: object = field(repr=False)


class PendingRecallReferenceRegistry:
    """Invalidate only live preparations whose exact storage keys may change.

    Callers serialize registration, mutation invalidation and final validation
    through their existing Writer. The registry's separate lock allows guard
    cleanup to release a fence without acquiring Writer or reader locks. It
    never invokes a callback, reads storage, or keeps historical row metadata.

    Keys identify actual entity rows, including the separately identified
    Catalog entity. Their finite set comes from the already bounded receipt
    and reference readers; this registry grants no additional query budget.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._issuer = object()
        self._closed = False
        self._active: dict[
            PendingRecallReferenceFence, frozenset[PendingRecallReferenceKey] | None,
        ] = {}
        self._by_key: dict[PendingRecallReferenceKey, set[PendingRecallReferenceFence]] = {}

    @staticmethod
    def _check_keys(keys: frozenset[PendingRecallReferenceKey], *, nonempty: bool) -> None:
        if type(keys) is not frozenset or (nonempty and not keys) or any(
            type(key) is not tuple or len(key) != 2
            or any(type(value) is not str or not value for value in key)
            for key in keys
        ):
            raise AMLError("schema_invalid", "Pending receipt reference keys are invalid")

    def register(
        self, keys: frozenset[PendingRecallReferenceKey],
    ) -> PendingRecallReferenceFence:
        """Register finite actual keys before reads, while the caller holds Writer.

        A closed registry cannot issue new fences. Empty or malformed key sets
        fail rather than creating a preparation immune to invalidation.
        """
        self._check_keys(keys, nonempty=True)
        with self._lock:
            if self._closed:
                raise AMLError("service_unhealthy", "Pending receipt references are closed")
            fence = PendingRecallReferenceFence(self._issuer)
            self._active[fence] = keys
            for key in keys:
                self._by_key.setdefault(key, set()).add(fence)
            return fence

    def validate(
        self, fence: PendingRecallReferenceFence, *,
        keys: frozenset[PendingRecallReferenceKey] | None = None,
    ) -> bool:
        """Check exact issued, active, unstale identity under caller Writer ordering."""
        if keys is not None:
            self._check_keys(keys, nonempty=True)
        with self._lock:
            return (
                not self._closed and type(fence) is PendingRecallReferenceFence
                and getattr(fence, "_issuer", None) is self._issuer
                and self._active.get(fence) is not None
                and (keys is None or self._active[fence] == keys)
            )

    def _remove_keys(
        self, fence: PendingRecallReferenceFence, keys: frozenset[PendingRecallReferenceKey],
    ) -> None:
        for key in keys:
            members = self._by_key[key]
            members.remove(fence)
            if not members:
                del self._by_key[key]

    def invalidate(self, keys: frozenset[PendingRecallReferenceKey]) -> None:
        """Mark exact intersecting preparations stale before a physical attempt.

        Invoke under the existing Writer before main or Catalog writes, even
        when their acknowledgement may be lost. Unrelated keys do nothing;
        stale fences immediately shed their reverse-index keys.
        """
        self._check_keys(keys, nonempty=False)
        with self._lock:
            affected: set[PendingRecallReferenceFence] = set()
            for key in keys:
                affected.update(self._by_key.get(key, ()))
            for fence in affected:
                original = self._active[fence]
                if original is not None:
                    self._remove_keys(fence, original)
                    self._active[fence] = None

    def invalidate_all(self) -> None:
        """Invalidate live proofs before an unenumerated repair; allow later reprepare."""
        with self._lock:
            self._by_key.clear()
            for fence in self._active:
                self._active[fence] = None

    def release(self, fence: PendingRecallReferenceFence) -> None:
        """Forget one registration on unwind; repeated or foreign releases are harmless."""
        with self._lock:
            if type(fence) is not PendingRecallReferenceFence:
                return
            keys = self._active.pop(fence, None)
            if keys is not None:
                self._remove_keys(fence, keys)

    def close(self) -> None:
        """Permanently close issuance and discard all active keys and handles."""
        with self._lock:
            self._closed = True
            self._by_key.clear()
            self._active.clear()
