"""Reconstruct knowledge(t) from immutable revisions and current memory authority."""

from collections.abc import Callable
from dataclasses import replace

from pydantic import ValidationError

from quotemem.core.errors import QuoteMemError
from quotemem.core.identity import canonical_json
from quotemem.core.knowledge import KnowledgeReadView
from quotemem.core.query import ScopeV1
from quotemem.core.readability import Readability
from quotemem.core.recall_snapshot import MemoryLinkV2, MemoryRowV2, RecallSnapshot
from quotemem.store.domain import MemoryRow, RevisionRowV2
from quotemem.store.indexed import IndexedReader, RevisionLinks
from quotemem.store.writer import Writer


def read_knowledge_view(
    writer: Writer,
    root: RecallSnapshot,
    scope: ScopeV1,
    *,
    recorded_as_of: int,
    query_time: int,
    check_active: Callable[[], None],
    baseline: bool = False,
) -> KnowledgeReadView:
    """Choose the newest revision recorded at/before t, testing ``p_ver`` now.

    Args:
        writer: Sole persistence owner; this function writes nothing.
        root: Current guarded root, excluding forgotten memories.
        scope: Exact authorized scope.
        recorded_as_of: Inclusive, nonfuture recorded-time instant.
        query_time: Current expiry clock, independent of recorded time.
        check_active: Guard cancellation/authority check before every storage read.
        baseline: ``root`` is a held cycle's retained baseline: each
            chain starts at the baseline's own head, whatever the owner has
            committed since, so the memory rows the owner moves are not read. The
            immutable revisions and the links they carry are.

    Returns:
        Guard-owned selection with current Evidence and no historical memories.

    Raises:
        QuoteMemError: Invalid time, broken chronology, changed storage, scope mismatch
            or unavailable embedding space. Storage failures never become empties.
    """
    if type(recorded_as_of) is not int or not 0 <= recorded_as_of <= query_time:
        raise QuoteMemError("invalid_request", "Knowledge time must be nonfuture")
    fence = root.scope_fence(scope.scope_id)
    if fence.scope_kind != scope.scope_kind or fence.tomb:
        raise QuoteMemError("scope_denied", "Knowledge read crossed its exact scope")
    check_active()
    generation = writer.ref_generation
    indexed = IndexedReader(writer.runtime.workspace, writer.entities, tenant_id=writer.tenant_id)
    read = Readability(scope.scope_kind, scope.scope_id, query_time, root.get_source)
    selected: list[MemoryRowV2] = []
    try:
        for head in root.memory_rows(scope.scope_id):
            check_active()
            if not baseline:
                raw = writer.get_row("item", head.object_id)
                if raw is None:
                    raise QuoteMemError("delivery_stale", "Knowledge Item disappeared")
                item = MemoryRow.model_validate(raw)
                if (item.scope_kind, item.scope_id) != (scope.scope_kind, scope.scope_id):
                    raise QuoteMemError("scope_denied", "Knowledge Item crossed its scope")
                if (item.current_revision_id, item.lifecycle_state, item.retired_at) != (
                    head.current_revision_id,
                    head.lifecycle_state,
                    head.retired_at,
                ):
                    raise QuoteMemError("delivery_stale", "Knowledge Item changed")
            if (
                head.op == "erase"
                or head.lifecycle_state in {"historical", "erased"}
                or (head.retired_at is not None and head.retired_at <= recorded_as_of)
            ):
                continue
            identifier: str | None = head.current_revision_id
            newer_version, newer_time = head.version + 1, query_time
            chosen: RevisionRowV2 | None = None
            seen: set[str] = set()
            while identifier is not None:
                check_active()
                if identifier in seen:
                    raise QuoteMemError("parity_mismatch", "Knowledge revision chain cycles")
                seen.add(identifier)
                raw_revision = writer.get_row("revision", identifier)
                if raw_revision is None:
                    raise QuoteMemError("service_unhealthy", "Knowledge revision is missing")
                revision = RevisionRowV2.model_validate(raw_revision)
                if (
                    revision.memory_id != head.object_id
                    or revision.object_id != identifier
                    or (revision.version != newer_version - 1 or revision.recorded_at > newer_time)
                ):
                    raise QuoteMemError("parity_mismatch", "Knowledge chronology is inconsistent")
                if identifier == head.current_revision_id and (
                    revision.content_hash != head.content_hash
                    or canonical_json(revision.snapshot) != head.snapshot_json
                ):
                    raise QuoteMemError("delivery_stale", "Knowledge head differs from its pin")
                if chosen is None and revision.recorded_at <= recorded_as_of:
                    chosen = revision
                newer_version, newer_time = revision.version, revision.recorded_at
                identifier = revision.previous_revision_id
            if newer_version != 1:
                raise QuoteMemError("parity_mismatch", "Knowledge chain lacks its first revision")
            if chosen is None or not read.p_ver(chosen.snapshot):
                continue
            if chosen.snapshot.embedding_space != root.embedding_space:
                raise QuoteMemError("embedding_space_mismatch",
                                    "Knowledge embedding is unavailable")
            links: list[MemoryLinkV2] = []
            after: str | None = None
            while True:
                check_active()
                page = indexed.page(
                    RevisionLinks(source_revision_id=chosen.object_id), after=after, limit=128
                )
                links.extend(MemoryLinkV2.from_row(row) for row in page)
                if len(page) < 128:
                    break
                after = page[-1]["object_id"]
            selected.append(
                replace(
                    head,
                    current_revision_id=chosen.object_id,
                    lifecycle_state="active",
                    retired_at=None,
                    entity_id=chosen.snapshot.entity_id,
                    statement_digest=chosen.snapshot.statement_digest,
                    version=chosen.version,
                    op=chosen.op,
                    reason_code=chosen.reason_code,
                    recorded_at=chosen.recorded_at,
                    head_commit_seq=chosen.commit_seq,
                    content_hash=chosen.content_hash,
                    snapshot_json=canonical_json(chosen.snapshot),
                    links=tuple(sorted(links, key=lambda link: link.object_id)),
                    replaced=(),
                )
            )
    except QuoteMemError:
        # A QuoteMemError is a ValueError: its own code (delivery_stale, scope_denied,
        # parity_mismatch ...) stands, never "malformed rows".
        raise
    except (ValidationError, ValueError) as error:
        raise QuoteMemError("service_unhealthy", "Knowledge rows are malformed") from error
    check_active()
    if not baseline and writer.ref_generation != generation:
        raise QuoteMemError("delivery_stale", "Knowledge storage changed during reconstruction")
    projected = RecallSnapshot.rebuild(
        generation=root.generation,
        scopes=(root.scope_fence(scope.scope_id),),
        embedding_space=root.embedding_space,
        embedding_token=root.embedding_token,
        recall_policy=root.recall_policy,
        sources=root.scope_sources(scope.scope_id),
        memories=selected,
        entities=root.entity_rows(scope.scope_id),
        held=root.held_rows(scope.scope_id),
    )
    return KnowledgeReadView(projected, scope.scope_id, recorded_as_of, generation)
