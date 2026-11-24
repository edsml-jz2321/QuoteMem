"""Hash-bound adapters for the published LoCoMo-Refined and LongMemEval formats.

Upstream QA answers establish retrieval references, not QuoteMem publishability gold.
Dates without a timezone use UTC solely as a declared local replay convention.
LoCoMo has two human participants: both retain user role and an explicit speaker
label. Image bytes are never fetched; available captions remain labeled context.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from io import TextIOWrapper
from pathlib import Path
from typing import IO, Annotated, Literal, Self

from pydantic import Field, JsonValue, TypeAdapter, ValidationError, model_validator

from quotemem.bench.bundle import DatasetBundleSource, verified_source
from quotemem.bench.received_prefix import ReceivedPrefixRegistration
from quotemem.core.errors import QuoteMemError
from quotemem.core.identity import canonical_json
from quotemem.core.source_participants import SourceParticipantV1
from quotemem.store.metadata import StrictModel

Dataset = Literal["locomo_refined", "longmemeval", "memoryagentbench"]
Nonempty = Annotated[str, Field(min_length=1)]
Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
_JSON: TypeAdapter[JsonValue] = TypeAdapter(JsonValue)
_MAX_RECORD_CHARS = 32 * 1024 * 1024


class ConversationSelection(StrictModel):
    """Explicit whole-conversation membership within the original verified raw file."""

    original_expected_cases: Annotated[int, Field(gt=0)]
    conversation_ids: Annotated[list[Nonempty], Field(min_length=1)]
    case_ids: Annotated[list[Nonempty], Field(min_length=1)]

    @model_validator(mode="after")
    def unique_membership(self) -> Self:
        if (
            len(set(self.conversation_ids)) != len(self.conversation_ids)
            or len(set(self.case_ids)) != len(self.case_ids)
        ):
            raise ValueError("Conversation selection must contain unique original IDs")
        return self


class DatasetSource(StrictModel):
    """Pin raw source bytes and the complete declared question denominator.

    An explicit conversation selection keeps the raw descriptor unchanged and
    stores its original count beside the selected count and complete membership.
    """

    dataset: Dataset
    variant: Literal["refined", "oracle", "s_cleaned", "m_cleaned", "factconsolidation_6k_sh_mh"]
    revision: Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
    url: Nonempty
    sha256: Digest
    size_bytes: Annotated[int, Field(gt=0)]
    license: Nonempty
    license_url: Nonempty
    expected_cases: Annotated[int, Field(gt=0)]
    conversation_selection: ConversationSelection | None = Field(
        default=None, exclude_if=lambda value: value is None,
    )
    received_prefix: ReceivedPrefixRegistration | None = Field(
        default=None, exclude_if=lambda value: value is None,
    )

    @model_validator(mode="after")
    def selected_denominator(self) -> Self:
        selection = self.conversation_selection
        if self.received_prefix is not None and (
            self.expected_cases != len(self.received_prefix.case_ids)
            or selection is not None and selection.case_ids != self.received_prefix.case_ids
        ):
            raise ValueError("Received prefix differs from its declared question denominator")
        if selection is not None and (
            (self.dataset, self.variant) != ("locomo_refined", "refined")
            or self.expected_cases != len(selection.case_ids)
            or self.expected_cases > selection.original_expected_cases
        ):
            raise ValueError("Conversation selection differs from its declared LoCoMo denominator")
        return self


BenchmarkSource = DatasetSource | DatasetBundleSource


def source_identity_kind(source: BenchmarkSource) -> str:
    """Disclose whether a source identity binds raw bytes or an explicit selection."""
    if isinstance(source, DatasetBundleSource):
        return "bundle_manifest_v1"
    if source.received_prefix is not None:
        return "received_message_prefix_v1"
    if source.conversation_selection is not None:
        return "locomo_conversation_selection_v1"
    if source.dataset == "memoryagentbench":
        return "selected_file_manifest_v1"
    return "raw_file_sha256"


def source_identity(source: BenchmarkSource) -> str:
    """Preserve existing corpus identities while binding newly declared source selections."""
    if isinstance(source, DatasetBundleSource):
        return source.manifest_sha256
    if source.conversation_selection is not None or source.received_prefix is not None:
        return hashlib.sha256(canonical_json(source.model_dump(mode="json")).encode()).hexdigest()
    if source.dataset == "memoryagentbench":
        return hashlib.sha256(
            canonical_json(
                {
                    "kind": "selected_file_manifest_v1",
                    "dataset": source.dataset,
                    "variant": source.variant,
                    "revision": source.revision,
                    "raw_file_sha256": source.sha256,
                    "selection": ["factconsolidation_sh_6k", "factconsolidation_mh_6k"],
                }
            ).encode()
        ).hexdigest()
    return source.sha256


def source_matches_cases(source: BenchmarkSource, case_ids: list[str]) -> bool:
    """Check complete ordinary membership or the exact declared selected question order."""
    if len(case_ids) != source.expected_cases:
        return False
    if isinstance(source, DatasetSource) and source.received_prefix is not None:
        return case_ids == source.received_prefix.case_ids
    return not isinstance(source, DatasetSource) or source.conversation_selection is None or (
        case_ids == source.conversation_selection.case_ids
    )


@dataclass(frozen=True)
class BenchmarkMessage:
    """One source turn, retaining its source identity and replay text."""

    message_id: str
    role: Literal["user", "assistant"]
    content: str
    source_date: str | None = None
    timestamp_ms: int | None = None
    speaker: str | None = None
    source_participant: SourceParticipantV1 | None = None


@dataclass(frozen=True)
class BenchmarkSession:
    """One source session with its original date and deterministic replay time."""

    session_id: str
    source_date: str | None
    timestamp_ms: int | None
    messages: tuple[BenchmarkMessage, ...]


@dataclass(frozen=True)
class BenchmarkCase:
    """A query, replay grouping, and complete source-bound reference answers.

    `cluster_id` groups identical contexts for ingestion. It does not certify
    statistical independence: LongMemEval can share session IDs across questions.
    Original session IDs remain available for later source-overlap analysis.
    """

    case_id: str
    cluster_id: str
    sessions: tuple[BenchmarkSession, ...]
    query: str
    answers: tuple[str, ...]
    evidence_message_ids: tuple[str, ...]
    evidence_session_ids: tuple[str, ...]
    category: str
    is_multimodal: bool = False
    is_abstention: bool = False
    unresolved_evidence_ids: tuple[str, ...] = ()
    query_source_date: str | None = None
    evaluation_kind: Literal["local_token_f1", "mab_substring_exact_match", "beam_rubric"] = (
        "local_token_f1"
    )
    evaluation_rubric: tuple[str, ...] = ()
    source_question_json: str | None = None


@dataclass(frozen=True)
class BenchmarkCorpus:
    """Normalized cases bound to an exact verified source manifest."""

    source: BenchmarkSource
    cases: tuple[BenchmarkCase, ...]


def select_conversations(
    corpus: BenchmarkCorpus, conversation_ids: list[str],
) -> BenchmarkCorpus:
    """Select complete LoCoMo conversations from an already verified source.

    Args:
        corpus: Entire loaded source, whose raw bytes must already be verified.
        conversation_ids: Nonempty, distinct original conversation IDs.

    Returns:
        BenchmarkCorpus: Every original case and complete context in the selected
        conversations, preserving source order. Its source retains the original
        raw descriptor and binds the exact selected denominator separately.

    Raises:
        QuoteMemError: If the source is not LoCoMo or any selection ID is invalid.
    """
    requested = set(conversation_ids)
    available = {case.cluster_id for case in corpus.cases}
    if (
        not isinstance(corpus.source, DatasetSource)
        or corpus.source.dataset != "locomo_refined" or not requested
        or len(requested) != len(conversation_ids) or not requested <= available
        or corpus.source.conversation_selection is not None
        or corpus.source.received_prefix is not None
        or len(corpus.cases) != corpus.source.expected_cases
    ):
        raise QuoteMemError("dataset_invalid", "Select distinct existing LoCoMo conversations")
    cases = tuple(case for case in corpus.cases if case.cluster_id in requested)
    source = DatasetSource.model_validate({
        **corpus.source.model_dump(mode="python"), "expected_cases": len(cases),
        "conversation_selection": {
            "original_expected_cases": corpus.source.expected_cases,
            "conversation_ids": list(dict.fromkeys(case.cluster_id for case in cases)),
            "case_ids": [case.case_id for case in cases],
        },
    })
    return BenchmarkCorpus(source, cases)


class _LoCoMoQuestion(StrictModel):
    question: Nonempty
    answer: Annotated[list[str | int], Field(min_length=1)]
    evidence: list[Nonempty]
    category: Literal[1, 2, 3, 4]
    is_multi_modality: bool


class _LoCoMoMessage(StrictModel):
    speaker: Nonempty
    dia_id: Nonempty
    text: Nonempty
    blip_caption: str | None = None
    img_url: list[str] | str | None = None
    query: str | None = None
    redownload: JsonValue = Field(default=None, alias="re-download")


class _LoCoMoRecord(StrictModel):
    sample_id: Nonempty
    qa: Annotated[list[_LoCoMoQuestion], Field(min_length=1)]
    conversation: dict[str, JsonValue]
    event_summary: JsonValue
    observation: JsonValue
    session_summary: JsonValue


class _LongMemEvalMessage(StrictModel):
    role: Literal["user", "assistant"]
    content: str
    has_answer: bool | None = None


class _LongMemEvalRecord(StrictModel):
    question_id: Nonempty
    question_type: Literal[
        "single-session-user",
        "single-session-assistant",
        "single-session-preference",
        "temporal-reasoning",
        "knowledge-update",
        "multi-session",
    ]
    question: Nonempty
    answer: Nonempty | int
    question_date: Nonempty
    haystack_dates: list[Nonempty]
    haystack_session_ids: list[Nonempty]
    haystack_sessions: list[list[_LongMemEvalMessage]]
    answer_session_ids: list[Nonempty]


def _object(pairs: list[tuple[str, JsonValue]]) -> dict[str, JsonValue]:
    value: dict[str, JsonValue] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("Dataset JSON repeats an object key")
        value[key] = item
    return value


def _constant(_: str) -> JsonValue:
    raise ValueError("Dataset JSON contains a non-finite number")


def read_source_json(path: Path) -> JsonValue:
    """Decode one bounded source JSON file with duplicate and non-finite rejection."""
    with path.open("rb") as stream:
        return decode_source_json(stream.read(_MAX_RECORD_CHARS + 1))


def read_verified_json(path: Path, *, size_bytes: int, sha256: str) -> JsonValue:
    """Decode bounded JSON from the exact verified bytes of one source part.

    Args:
        path (Path): Local source pathname, opened once for a private snapshot.
        size_bytes (int): Exact manifest size, no greater than the 32 MiB JSON limit.
        sha256 (str): Expected digest of the raw source bytes.

    Returns:
        JsonValue: Strictly decoded JSON; duplicate keys and non-finite numbers fail.

    Raises:
        QuoteMemError: If source size or digest differs from its manifest.
        OSError: If the source or temporary snapshot cannot be read or written.
        ValueError: If the declared size exceeds the JSON bound or decoding fails.
    """
    if size_bytes > _MAX_RECORD_CHARS:
        raise ValueError("Dataset JSON file exceeds the bounded parser size")
    with verified_source(path, size_bytes=size_bytes, sha256=sha256) as snapshot:
        return decode_source_json(snapshot.read(_MAX_RECORD_CHARS + 1))


def decode_source_json(payload: bytes) -> JsonValue:
    """Decode bounded source/manifest bytes without duplicate-key or numeric ambiguity."""
    if len(payload) > _MAX_RECORD_CHARS:
        raise ValueError("Dataset JSON payload exceeds the bounded parser size")
    return _JSON.validate_python(
        json.loads(payload, object_pairs_hook=_object, parse_constant=_constant)
    )


def _records(source: IO[bytes]) -> Iterator[JsonValue]:
    """Decode one top-level array record at a time without a full-file allocation."""
    decoder = json.JSONDecoder(object_pairs_hook=_object, parse_constant=_constant)
    with TextIOWrapper(source, encoding="utf-8") as stream:
        buffer = ""
        eof = False

        def refill() -> None:
            nonlocal buffer, eof
            chunk = stream.read(65536)
            buffer += chunk
            eof = not chunk
            if len(buffer) > _MAX_RECORD_CHARS:
                raise ValueError("Dataset record exceeds the bounded parser size")

        def trim() -> None:
            nonlocal buffer
            buffer = buffer.lstrip()
            while not buffer and not eof:
                refill()
                buffer = buffer.lstrip()

        trim()
        if not buffer.startswith("["):
            raise ValueError("Dataset root must be an array")
        buffer = buffer[1:]
        first = True
        while True:
            trim()
            if buffer.startswith("]"):
                buffer = buffer[1:]
                break
            if not first:
                if not buffer.startswith(","):
                    raise ValueError("Dataset array separator is missing")
                buffer = buffer[1:]
                trim()
                if buffer.startswith("]"):
                    raise ValueError("Dataset array has a trailing comma")
            while True:
                try:
                    record, end = decoder.raw_decode(buffer)
                    break
                except json.JSONDecodeError:
                    if eof:
                        raise
                    refill()
            yield _JSON.validate_python(record)
            buffer = buffer[end:]
            first = False
        trim()
        if buffer:
            raise ValueError("Dataset contains trailing content")


def _date(value: str, pattern: str) -> int:
    return int(datetime.strptime(value, pattern).replace(tzinfo=UTC).timestamp() * 1000)


def _locomo(value: JsonValue, *, source_namespace: str) -> tuple[BenchmarkCase, ...]:
    record = _LoCoMoRecord.model_validate(value)
    conversation = record.conversation
    speaker_a, speaker_b = conversation.get("speaker_a"), conversation.get("speaker_b")
    if not isinstance(speaker_a, str) or not isinstance(speaker_b, str):
        raise ValueError("LoCoMo requires two distinct named speakers")
    speakers = {speaker_a, speaker_b}
    if len(speakers) != 2 or any(not item for item in speakers):
        raise ValueError("LoCoMo requires two distinct named speakers")
    indices = sorted(int(key[8:]) for key in conversation if re.fullmatch(r"session_\d+", key))
    expected_keys = {"speaker_a", "speaker_b"} | {
        key for index in indices for key in (f"session_{index}", f"session_{index}_date_time")
    }
    # The refined source retains date-only keys for removed later sessions.
    extra_keys = set(conversation) - expected_keys
    if (
        not indices
        or not expected_keys <= set(conversation)
        or any(
            not re.fullmatch(r"session_\d+_date_time", key)
            or not isinstance(conversation[key], str)
            for key in extra_keys
        )
    ):
        raise ValueError("LoCoMo session keys are incomplete or unknown")
    sessions: list[BenchmarkSession] = []
    message_sessions: dict[str, str] = {}
    for index in indices:
        session_id = f"{record.sample_id}:session-{index}"
        source_date = conversation[f"session_{index}_date_time"]
        if not isinstance(source_date, str):
            raise ValueError("LoCoMo session date must be a string")
        source_messages = TypeAdapter(list[_LoCoMoMessage]).validate_python(
            conversation[f"session_{index}"]
        )
        messages: list[BenchmarkMessage] = []
        for message in source_messages:
            if message.speaker not in speakers or message.dia_id in message_sessions:
                raise ValueError("LoCoMo speaker or turn identity is inconsistent")
            message_sessions[message.dia_id] = session_id
            content = f"{message.speaker}: {message.text}"
            if message.blip_caption:
                content += f"\n[Image caption: {message.blip_caption}]"
            messages.append(BenchmarkMessage(
                message.dia_id, "user", content, speaker=message.speaker,
                source_participant=SourceParticipantV1(
                    namespace=source_namespace, conversation_id=record.sample_id,
                    participant_id="speaker_a" if message.speaker == speaker_a else "speaker_b",
                ),
            ))
        if not messages:
            raise ValueError("LoCoMo session cannot be empty")
        sessions.append(
            BenchmarkSession(
                session_id,
                source_date,
                _date(source_date, "%I:%M %p on %d %B, %Y"),
                tuple(messages),
            )
        )
    cases: list[BenchmarkCase] = []
    for index, question in enumerate(record.qa):
        answers = tuple(str(answer) for answer in question.answer)
        if any(not answer for answer in answers):
            raise ValueError("LoCoMo answer cannot be empty")
        cases.append(
            BenchmarkCase(
                case_id=f"{record.sample_id}#q{index:04d}",
                cluster_id=record.sample_id,
                sessions=tuple(sessions),
                query=question.question,
                answers=answers,
                evidence_message_ids=tuple(question.evidence),
                evidence_session_ids=tuple(
                    sorted(
                        {message_sessions[e] for e in question.evidence if e in message_sessions}
                    )
                ),
                category=str(question.category),
                is_multimodal=question.is_multi_modality,
                unresolved_evidence_ids=tuple(
                    e for e in question.evidence if e not in message_sessions
                ),
            )
        )
    return tuple(cases)


def _session_ids(source_ids: list[str]) -> list[str]:
    """Name each haystack session once: a repeated id's k-th occurrence becomes ``<id>#k``.

    LongMemEval-S ``s_cleaned`` repeats one distractor session id in 13 records
    (58bf7951, 1e043500, 001be529, d23cf73b, caf03d32, 91b15a6e, 078150f1, gpt4_4929293b,
    gpt4_76048e76, gpt4_c27434e8_abs, 18bc8abd, c7dc5443, 1d4da289): identical turns at two
    different dates. Both occurrences are kept, each at its own date, as the haystack holds
    them; the first keeps the source id, which answer references name.
    """
    seen: dict[str, int] = {}
    named: list[str] = []
    for session_id in source_ids:
        seen[session_id] = seen.get(session_id, 0) + 1
        named.append(session_id if seen[session_id] == 1 else f"{session_id}#{seen[session_id]}")
    if len(set(named)) != len(named):
        raise ValueError("LongMemEval repeated session ids collide with their renamed occurrences")
    return named


def _longmemeval(value: JsonValue) -> tuple[BenchmarkCase, ...]:
    """Normalize one record; source turns keep their order, content and index.

    Two deterministic normalizations keep every content-bearing turn: a repeated session id
    is renamed per occurrence (``_session_ids``), and a turn whose content is the empty
    string (12 turns in 7 ``s_cleaned`` records: dd2973ad, gpt4_7fce9456, gpt4_d6585ce9,
    gpt4_4edbafa2, gpt4_fe651585_abs, affe2881, 41275add; none carries ``has_answer``) is
    dropped, the others keeping their source ``turn-<index>``. A record with neither
    normalizes exactly as before.
    """
    record = _LongMemEvalRecord.model_validate(value)
    _date(record.question_date, "%Y/%m/%d (%a) %H:%M")
    count = len(record.haystack_session_ids)
    if not count or count != len(record.haystack_dates) or count != len(record.haystack_sessions):
        raise ValueError("LongMemEval parallel session arrays must have equal nonzero length")
    if not set(record.answer_session_ids) <= set(record.haystack_session_ids):
        raise ValueError("LongMemEval session identities or answer references are invalid")
    sessions: list[BenchmarkSession] = []
    evidence: list[str] = []
    for session_id, date, source_turns in zip(
        _session_ids(record.haystack_session_ids), record.haystack_dates,
        record.haystack_sessions, strict=True,
    ):
        turns = [(index, turn) for index, turn in enumerate(source_turns) if turn.content]
        if not turns:
            raise ValueError("LongMemEval session cannot be empty")
        messages = tuple(
            BenchmarkMessage(f"{session_id}:turn-{index}", turn.role, turn.content)
            for index, turn in turns
        )
        evidence.extend(
            message.message_id
            for (_, turn), message in zip(turns, messages, strict=True)
            if turn.has_answer
        )
        sessions.append(
            BenchmarkSession(session_id, date, _date(date, "%Y/%m/%d (%a) %H:%M"), messages)
        )
    # Oracle arrays are explicitly not ordered upstream. Stable date/session sorting
    # defines local replay order without changing message order inside any session.
    sessions.sort(key=lambda session: (session.timestamp_ms or 0, session.session_id))
    return (
        BenchmarkCase(
            case_id=record.question_id,
            cluster_id=record.question_id,
            sessions=tuple(sessions),
            query=record.question,
            answers=(str(record.answer),),
            evidence_message_ids=tuple(evidence),
            evidence_session_ids=tuple(record.answer_session_ids),
            category=record.question_type,
            is_abstention=record.question_id.endswith("_abs"),
            query_source_date=record.question_date,
        ),
    )


def load_dataset(path: Path, source: BenchmarkSource) -> BenchmarkCorpus:
    """Verify exact source bytes before normalizing published benchmark records.

    Args:
        path (Path): Explicit local source file; this function never downloads data.
        source (DatasetSource): Frozen upstream revision, byte hash, size, and variant.

    Returns:
        BenchmarkCorpus: Complete validated cases in upstream question order.

    Raises:
        QuoteMemError: For missing, changed, malformed, inconsistent, or oversized data.
            Errors omit dataset content and paths.
    """
    try:
        if isinstance(source, DatasetBundleSource):
            from quotemem.bench.beam import load_beam_bundle

            return load_beam_bundle(path, source)
        source = DatasetSource.model_validate(source.model_dump(mode="python", warnings=False))
        if source.received_prefix is not None:
            from quotemem.bench.received_prefix import apply_received_prefix

            original = DatasetSource.model_validate(source.model_dump(
                mode="python", exclude={"received_prefix"},
            ))
            return apply_received_prefix(load_dataset(path, original), source.received_prefix)
        if source.conversation_selection is not None:
            selection = source.conversation_selection
            original = DatasetSource.model_validate({
                **source.model_dump(mode="python", exclude={"conversation_selection"}),
                "expected_cases": selection.original_expected_cases,
            })
            corpus = select_conversations(load_dataset(path, original), selection.conversation_ids)
            if corpus.source != source:
                raise ValueError("Selected source differs from its complete original conversations")
            return corpus
        variants = {
            "locomo_refined": {"refined"},
            "longmemeval": {"oracle", "s_cleaned", "m_cleaned"},
            "memoryagentbench": {"factconsolidation_6k_sh_mh"},
        }
        if source.variant not in variants[source.dataset]:
            raise ValueError("Dataset and variant disagree")
        with verified_source(path, size_bytes=source.size_bytes, sha256=source.sha256) as snapshot:
            if source.dataset == "memoryagentbench":
                from quotemem.bench.memoryagentbench import load_memoryagentbench

                cases = load_memoryagentbench(snapshot)
            else:
                source_namespace = f"{source.dataset}:{source.revision}:{source.sha256}"
                cases = tuple(
                    case for record in _records(snapshot)
                    for case in (
                        _locomo(record, source_namespace=source_namespace)
                        if source.dataset == "locomo_refined" else _longmemeval(record)
                    )
                )
        if len(cases) != source.expected_cases or len({case.case_id for case in cases}) != len(
            cases
        ):
            raise ValueError("Dataset case count or unique identities differ from the manifest")
        return BenchmarkCorpus(source, cases)
    except QuoteMemError:
        raise
    except (OSError, UnicodeError, ValueError, TypeError, ValidationError, RecursionError):
        raise QuoteMemError("dataset_invalid", "Dataset failed strict loading") from None
