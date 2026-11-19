"""Erase raw reception inputs left behind by a publication crash or withdrawn consent.

An ordered Add no longer needs its admission payload: its attached durable
receipt contains no source text. Normal reception deletes the payload after
publication, but a crash can leave it stored. Privacy erase takes those records
when their exact received source is erased, using the same receipt binding as
``Writer.discard_received_add_input``. A never-ordered reception keeps its input
unless its admission predates the scope's consent epoch and cannot continue.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, cast

from aml.core.errors import AMLError
from aml.core.run_inputs import SessionAdmissionRunInputV1, SessionDurableRunInputV1
from aml.core.semantic import ScopeKind
from aml.store.indexed import MAX_PAGE_SIZE, ScopedRunRows

if TYPE_CHECKING:
    from aml.store.privacy_compensation import RecordGroup
    from aml.store.writer import Writer


def _receptions(writer: Writer, scope: tuple[str, str]) -> Iterator[dict[str, Any]]:
    """Page ordinary reception roots in the erase's exact scope."""
    query = ScopedRunRows(scope_kind=cast(ScopeKind, scope[0]), scope_id=scope[1],
                          run_kind="session_ingestion")
    reader = writer.indexed_reader()
    after: str | None = None
    while True:
        page = reader.page(query, after=after, limit=MAX_PAGE_SIZE)
        yield from (row for row in page if row["commit_mode"] == "orchestrator")
        if len(page) < MAX_PAGE_SIZE:
            return
        after = page[-1]["object_id"]


def reception_record_groups(
    writer: Writer, scope: tuple[str, str], erased: frozenset[str],
) -> tuple[RecordGroup, ...]:
    """Select obsolete raw admissions with their physical manifest parts (D.7, P06).

    An attached receipt proves ordering and binds the erased source's session,
    request and visibility generation. Each saved admission must match that
    receipt's admission digest, payload digest and business key. With no receipt,
    only an admission under withdrawn consent qualifies; an interrupted request
    under the current consent epoch keeps its source for its own retry.

    Args:
        writer: The sole Writer, read only within the erase's write scope.
        scope: The erase's exact scope kind and id.
        erased: Evidence erased by this plan, including its current transaction.

    Returns:
        Deterministically ordered groups of complete stored admission records.

    Raises:
        AMLError: ``reference_missing`` for missing selected Evidence,
            ``scope_denied`` for Evidence outside the erase's scope, or
            ``binding_mismatch`` for a receipt or admission that changed identity.
    """
    from aml.store.privacy_compensation import record_groups

    coordinates: set[tuple[str, str, int]] = set()
    for evidence_id in erased:
        row = writer.get_row("evidence", evidence_id)
        if row is None:
            raise AMLError("reference_missing", "Reception erasure lost selected Evidence")
        if (row["scope_kind"], row["scope_id"]) != scope:
            raise AMLError("scope_denied", "Reception erasure selected another scope")
        coordinates.add((row["session_id"], row["request_id"], row["ingested_commit_seq"]))

    consent = writer.counter(f"consent_epoch:{scope[1]}")
    references: set[str] = set()
    for run in _receptions(writer, scope):
        receipt: SessionDurableRunInputV1 | None = None
        if run["input_ref"] is not None:
            body = writer.read_logical(run["input_ref"], "run_input")
            if not isinstance(body, SessionDurableRunInputV1):
                raise AMLError("binding_mismatch", "Reception erasure selected another receipt")
            if (body.business_key.session_id, body.business_key.request_id,
                body.visibility_commit_seq) not in coordinates:
                continue
            receipt = body
        elif run["admitted_consent_epoch"] >= consent:
            continue
        for reference, body in writer.find_logical_candidates(run["object_id"], "run_input"):
            if not isinstance(body, SessionAdmissionRunInputV1):
                continue
            if body.admission_digest != run["admission_digest"] or receipt is not None and (
                body.admission_digest, body.payload_digest, body.business_key,
            ) != (receipt.admission_digest, receipt.payload_digest, receipt.business_key):
                raise AMLError("binding_mismatch", "Reception erasure changed saved input")
            references.add(reference)
    return tuple(record_groups(writer, references))
