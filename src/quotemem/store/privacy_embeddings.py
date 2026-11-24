"""Scrub mixed embedding batches without resending or rewriting surviving proposals."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING, Any, cast

from quotemem.core.entities import PositionKey
from quotemem.core.errors import QuoteMemError
from quotemem.core.identity import ArtifactId
from quotemem.core.privacy_embeddings import PrivacyEmbeddingV1
from quotemem.core.proposal import ProposalV2
from quotemem.core.semantic import ScopeKind
from quotemem.core.semantic_embedding import StatementEmbeddingV1
from quotemem.store.indexed import MAX_PAGE_SIZE, IndexedQuery, ScopedRunRows

if TYPE_CHECKING:
    from quotemem.store.writer import Writer


def _runs(writer: Writer, query: IndexedQuery) -> Iterator[dict[str, Any]]:
    reader = writer.indexed_reader()
    after: str | None = None
    while True:
        page = reader.page(query, after=after, limit=MAX_PAGE_SIZE)
        yield from page
        if len(page) < MAX_PAGE_SIZE:
            return
        after = page[-1]["object_id"]


def _records(
    writer: Writer, scope: tuple[str, str] | None,
) -> dict[str, list[tuple[str, PrivacyEmbeddingV1]]]:
    if scope is None:
        runs = (row for row in writer.iter_rows("run") if row["run_kind"] == "retention")
    else:
        runs = _runs(writer, ScopedRunRows(
            scope_kind=cast(ScopeKind, scope[0]), scope_id=scope[1], run_kind="retention",
        ))
    found: dict[str, list[tuple[str, PrivacyEmbeddingV1]]] = {}
    for run in runs:
        for reference, body in writer.find_logical_candidates(run["object_id"], "decision_set"):
            if isinstance(body, PrivacyEmbeddingV1):
                owner = writer.get_row("run", body.owner_run_id)
                if owner is None or owner["run_kind"] != "finalize" or (
                    owner["scope_kind"], owner["scope_id"]
                ) != (run["scope_kind"], run["scope_id"]):
                    raise QuoteMemError("binding_mismatch",
                                        "Retained embeddings changed their scope")
                found.setdefault(body.original_ref, []).append((reference, body))
    return found


def _subset(left: PrivacyEmbeddingV1, right: PrivacyEmbeddingV1) -> bool:
    if (left.original_ref, left.owner_run_id, left.embedding_space, left.resolved_model,
        left.dimensions) != (right.original_ref, right.owner_run_id, right.embedding_space,
                             right.resolved_model, right.dimensions):
        return False
    entries = dict(zip(right.indices, right.entries, strict=True))
    for index, entry in zip(left.indices, left.entries, strict=True):
        previous = entries.get(index)
        if previous is None or not set(entry.uses) <= set(previous.uses) or (
            entry.model_dump(exclude={"uses"}) != previous.model_dump(exclude={"uses"})
        ):
            return False
    return True


def _minimal(records: list[tuple[str, PrivacyEmbeddingV1]]) -> tuple[str, PrivacyEmbeddingV1]:
    selected = [record for record in records
                if all(_subset(record[1], other[1]) for other in records)]
    if not selected:
        raise QuoteMemError("binding_mismatch", "Retained embedding snapshots are not monotone")
    return min(selected, key=lambda record: record[0])


def retained_embedding(
    writer: Writer, original_ref: str, owner_run_id: str | None = None,
) -> PrivacyEmbeddingV1 | None:
    """Resolve a missing original batch to its narrowest durable retained snapshot.

    Args:
        writer: The single Writer, used read-only.
        original_ref: The immutable original batch address a proposal names.
        owner_run_id: Its extraction owner when the caller has one; narrows the scan.

    Returns:
        The retained entries, including an empty erasure marker, or None without a record.

    Raises:
        QuoteMemError: For missing ownership or conflicting, non-monotone snapshots.
    """
    scope = None
    if owner_run_id is not None:
        owner = writer.get_row("run", owner_run_id)
        if owner is None:
            raise QuoteMemError("reference_missing", "Retained embeddings lost their owner")
        scope = owner["scope_kind"], owner["scope_id"]
    records = _records(writer, scope).get(original_ref)
    if not records:
        return None
    result = _minimal(records)[1]
    if owner_run_id is not None and result.owner_run_id != owner_run_id:
        raise QuoteMemError("binding_mismatch", "Retained embeddings name another owner")
    return result


def _original(reference: ArtifactId, batch: StatementEmbeddingV1) -> PrivacyEmbeddingV1:
    return PrivacyEmbeddingV1(
        privacy_embedding=1, original_ref=reference, owner_run_id=batch.owner_run_id,
        embedding_space=batch.embedding_space, resolved_model=batch.resolved_model,
        dimensions=batch.dimensions, indices=list(range(len(batch.entries))),
        entries=batch.entries,
    )


def _retain(
    source: PrivacyEmbeddingV1, uses: Mapping[int, set[PositionKey]],
) -> PrivacyEmbeddingV1:
    kept = [
        (index, entry.model_copy(update={"uses": sorted(set(entry.uses) & uses.get(index, set()))}))
        for index, entry in zip(source.indices, source.entries, strict=True)
        if set(entry.uses) & uses.get(index, set())
    ]
    return PrivacyEmbeddingV1(
        privacy_embedding=1, original_ref=source.original_ref, owner_run_id=source.owner_run_id,
        embedding_space=source.embedding_space, resolved_model=source.resolved_model,
        dimensions=source.dimensions, indices=[index for index, _ in kept],
        entries=[entry for _, entry in kept],
    )


def prepare_embedding_scrubs(
    writer: Writer, scope: tuple[str, str], deleting: frozenset[str],
) -> tuple[PrivacyEmbeddingV1, ...]:
    """Build subset prewrites before the selected proposals or original batches leave.

    Args:
        writer: The single Writer, under its erase write scope.
        scope: The erase's exact scope.
        deleting: Logical artifact ids selected for deletion, including proposals and batches.

    Returns:
        Changed retained snapshots to persist under the privacy owner before deletion.
        The next pass re-derives the same subset after a crash.
    """
    replacements = _records(writer, scope)
    originals: dict[str, PrivacyEmbeddingV1] = {}
    needed: dict[str, dict[int, set[PositionKey]]] = {}
    affected = set(replacements)
    for run in _runs(writer, ScopedRunRows(scope_kind=cast(ScopeKind, scope[0]),
                                          scope_id=scope[1], run_kind="finalize")):
        for reference, body in writer.find_logical_candidates(run["object_id"],
                                                              "semantic_embedding"):
            if not isinstance(body, StatementEmbeddingV1) or body.owner_run_id != run["object_id"]:
                raise QuoteMemError("binding_mismatch", "An embedding batch changed its owner")
            originals[reference] = _original(reference, body)
            if reference in deleting:
                affected.add(reference)
        for reference, body in writer.find_logical_candidates(run["object_id"],
                                                              "proposal_snapshot"):
            if not isinstance(body, ProposalV2):
                raise QuoteMemError("binding_mismatch",
                                    "An extraction owner stored another proposal")
            for use in (body.statement_embedding, body.cue_embedding):
                if use is None:
                    continue
                if reference in deleting:
                    affected.add(use.ref)
                else:
                    needed.setdefault(use.ref, {}).setdefault(use.index, set()).add(
                        body.position_key,
                    )
    changed: list[PrivacyEmbeddingV1] = []
    for reference in sorted(affected):
        records = replacements.get(reference, [])
        previous = _minimal(records)[1] if records else None
        original = originals.get(reference)
        if previous is not None and original is not None and not _subset(previous, original):
            raise QuoteMemError("binding_mismatch", "Retained embeddings differ from the original")
        source = previous or original
        if source is None:
            raise QuoteMemError("reference_missing", "An erased proposal lost its embedding proof")
        following = _retain(source, needed.get(reference, {}))
        if following != previous:
            changed.append(following)
    return tuple(changed)


def superseded_embedding_refs(writer: Writer, scope: tuple[str, str]) -> frozenset[str]:
    """Find every original and superseded subset that a durable replacement supersedes.

    Args:
        writer: The single Writer, used read-only under the erase barrier.
        scope: The erase's exact scope.

    Returns:
        Logical artifact ids to delete with their manifest parts. The minimal snapshot,
        including an empty marker, remains so erased vectors never revive after a crash.
    """
    found: set[str] = set()
    for original_ref, records in _records(writer, scope).items():
        selected, _ = _minimal(records)
        found.add(original_ref)
        found.update(reference for reference, _ in records if reference != selected)
    return frozenset(found)
