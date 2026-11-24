"""Explicit fresh-scope checkpoints containing only received original source turns."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from typing import TYPE_CHECKING, Annotated, Literal, Self

from pydantic import BeforeValidator, Field, model_validator

from quotemem.core.errors import QuoteMemError
from quotemem.core.identity import canonical_json
from quotemem.store.metadata import StrictModel

if TYPE_CHECKING:
    from quotemem.bench.datasets import (
        BenchmarkCase,
        BenchmarkCorpus,
        BenchmarkSession,
        BenchmarkSource,
    )

Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Name = Annotated[str, Field(min_length=1)]


def _version(value: object) -> object:
    if type(value) is not int or value != 1:
        raise ValueError("Received-prefix version must be the exact integer 1")
    return value


class ReceivedPrefixCut(StrictModel):
    """A complete-message boundary and hashes of its full and received inventories."""

    cluster_id: Name
    received_message_count: Annotated[int, Field(ge=1)]
    source_message_count: Annotated[int, Field(ge=1)]
    source_inventory_sha256: Digest
    received_inventory_sha256: Digest
    last_session_id: Name
    last_message_id: Name

    @model_validator(mode="after")
    def bounded(self) -> Self:
        if self.received_message_count > self.source_message_count:
            raise ValueError("Received prefix exceeds the original message population")
        return self


class ReceivedPrefixRegistration(StrictModel):
    """Shared source selection for QuoteMem/BM25/Mem0, frozen before any checkpoint run.

    Counts follow original session/message order, not timestamp order. Every
    checkpoint replays its cumulative prefix into a fresh scope; it is not an
    observation of an uninterrupted mutable runtime. Original answer labels are
    source inventory only and cannot score this checkpoint.
    """

    schema_version: Annotated[Literal[1], BeforeValidator(_version)]
    checkpoint_id: Name
    original_source_identity: Digest
    execution: Literal["fresh_scope_cumulative_prefix"]
    labels: Literal["original_answers_are_not_checkpoint_truth"]
    case_ids: Annotated[list[Name], Field(min_length=1)]
    cuts: Annotated[list[ReceivedPrefixCut], Field(min_length=1)]

    @model_validator(mode="after")
    def unique(self) -> Self:
        if len(set(self.case_ids)) != len(self.case_ids) or len({
            cut.cluster_id for cut in self.cuts
        }) != len(self.cuts):
            raise ValueError("Received-prefix population repeats an identity")
        return self


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _clusters(cases: tuple[BenchmarkCase, ...]) -> dict[str, tuple[BenchmarkSession, ...]]:
    grouped: dict[str, tuple[BenchmarkSession, ...]] = {}
    for case in cases:
        if case.cluster_id in grouped and grouped[case.cluster_id] != case.sessions:
            raise QuoteMemError("binding_mismatch",
                                "Prefix cluster has inconsistent source sessions")
        grouped[case.cluster_id] = case.sessions
    for sessions in grouped.values():
        if not sessions or len({item.session_id for item in sessions}) != len(sessions) or any(
            not session.messages or len({message.message_id for message in session.messages})
            != len(session.messages) for session in sessions
        ):
            raise QuoteMemError("binding_mismatch",
                                "Prefix source repeats or omits a source coordinate")
    return grouped


def _inventory(sessions: tuple[BenchmarkSession, ...]) -> list[dict[str, object]]:
    return [{
        "session_id": session.session_id, "source_date": session.source_date,
        "timestamp_ms": session.timestamp_ms,
        "messages": [{
            "message_id": message.message_id, "role": message.role,
            "content_sha256": hashlib.sha256(message.content.encode("utf-8")).hexdigest(),
            "source_date": message.source_date, "timestamp_ms": message.timestamp_ms,
            "speaker": message.speaker,
            "source_participant": (message.source_participant.model_dump(mode="json")
                                   if message.source_participant is not None else None),
        } for message in session.messages],
    } for session in sessions]


def _take(sessions: tuple[BenchmarkSession, ...], count: int) -> tuple[BenchmarkSession, ...]:
    selected: list[BenchmarkSession] = []
    for session in sessions:
        if count <= 0:
            break
        messages = session.messages[:count]
        selected.append(replace(session, messages=messages))
        count -= len(messages)
    if count:
        raise QuoteMemError("binding_mismatch", "Received prefix exceeds the actual source")
    return tuple(selected)


def register_received_prefix(
    corpus: BenchmarkCorpus, *, checkpoint_id: str, message_counts: dict[str, int],
) -> ReceivedPrefixRegistration:
    """Freeze exact whole-message cuts from a complete already verified source.

    Every cluster must have one positive count. Empty-source checkpoints are not
    supported by the existing baseline producers and are refused explicitly.
    This function creates no expected outcomes and performs no execution.
    """
    from quotemem.bench.datasets import DatasetSource, source_identity, source_matches_cases

    if not isinstance(corpus.source, DatasetSource) or corpus.source.received_prefix is not None:
        raise QuoteMemError("binding_mismatch", "Prefix registration requires an uncut file source")
    case_ids = [case.case_id for case in corpus.cases]
    if not source_matches_cases(corpus.source, case_ids) or len(set(case_ids)) != len(case_ids):
        raise QuoteMemError("binding_mismatch",
                            "Prefix registration requires its complete case census")
    clusters = _clusters(corpus.cases)
    if set(message_counts) != set(clusters) or any(
        type(count) is not int or count < 1 for count in message_counts.values()
    ):
        raise QuoteMemError("binding_mismatch",
                            "Prefix counts must name every source cluster exactly")
    cuts: list[ReceivedPrefixCut] = []
    for cluster_id, sessions in clusters.items():
        count = message_counts[cluster_id]
        selected = _take(sessions, count)
        cuts.append(ReceivedPrefixCut(
            cluster_id=cluster_id, received_message_count=count,
            source_message_count=sum(len(session.messages) for session in sessions),
            source_inventory_sha256=_digest(_inventory(sessions)),
            received_inventory_sha256=_digest(_inventory(selected)),
            last_session_id=selected[-1].session_id,
            last_message_id=selected[-1].messages[-1].message_id,
        ))
    return ReceivedPrefixRegistration(
        schema_version=1, checkpoint_id=checkpoint_id,
        original_source_identity=source_identity(corpus.source),
        execution="fresh_scope_cumulative_prefix",
        labels="original_answers_are_not_checkpoint_truth", case_ids=case_ids, cuts=cuts,
    )


def apply_received_prefix(
    corpus: BenchmarkCorpus, registration: ReceivedPrefixRegistration,
) -> BenchmarkCorpus:
    """Recheck a frozen cut against the full source, then remove every later turn.

    Original source question/answer records are retained for custody only. Both
    source identity and replay scopes change; existing producers receive only
    the returned shortened sessions and never need a special retrieval branch.
    """
    from quotemem.bench.datasets import BenchmarkCorpus, DatasetSource

    registration = ReceivedPrefixRegistration.model_validate(registration.model_dump(mode="python"))
    expected = register_received_prefix(
        corpus, checkpoint_id=registration.checkpoint_id,
        message_counts={cut.cluster_id: cut.received_message_count for cut in registration.cuts},
    )
    if registration != expected:
        raise QuoteMemError("binding_mismatch",
                            "Received prefix differs from its frozen source cut")
    by_cluster = {cut.cluster_id: cut for cut in registration.cuts}
    source = DatasetSource.model_validate({
        **corpus.source.model_dump(mode="python"), "received_prefix": registration,
    })
    return BenchmarkCorpus(source, tuple(replace(
        case, sessions=_take(case.sessions, by_cluster[case.cluster_id].received_message_count),
    ) for case in corpus.cases))


def validate_received_prefix(source: BenchmarkSource, cases: tuple[BenchmarkCase, ...]) -> None:
    """Reject injected future turns or changed source metadata before producing requests.

    Internal context verification may supply a subset of cases, but every supplied
    cluster must still carry exactly its complete registered received prefix.
    """
    from quotemem.bench.datasets import DatasetSource, source_identity

    if not isinstance(source, DatasetSource) or source.received_prefix is None:
        return
    registration = ReceivedPrefixRegistration.model_validate(
        source.received_prefix.model_dump(mode="python"),
    )
    original = DatasetSource.model_validate(source.model_dump(
        mode="python", exclude={"received_prefix"},
    ))
    by_cluster = {cut.cluster_id: cut for cut in registration.cuts}
    case_ids = [case.case_id for case in cases]
    if registration.original_source_identity != source_identity(original) or (
        len(case_ids) != len(set(case_ids)) or not set(case_ids) <= set(registration.case_ids)
    ):
        raise QuoteMemError("binding_mismatch", "Prefix source or case identity changed")
    for cluster_id, sessions in _clusters(cases).items():
        cut = by_cluster.get(cluster_id)
        if cut is None or not sessions or (
            sum(len(session.messages) for session in sessions) != cut.received_message_count
            or _digest(_inventory(sessions)) != cut.received_inventory_sha256
            or sessions[-1].session_id != cut.last_session_id
            or sessions[-1].messages[-1].message_id != cut.last_message_id
        ):
            raise QuoteMemError("binding_mismatch",
                                "Prefix contains changed or unreceived source turns")
