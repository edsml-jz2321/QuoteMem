"""Normalize the publisher's complete 128K BEAM selection without generating answers."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, JsonValue, TypeAdapter

from quotemem.bench.bundle import DatasetBundleSource, bundle_part_path
from quotemem.bench.datasets import (
    BenchmarkCase,
    BenchmarkCorpus,
    BenchmarkMessage,
    BenchmarkSession,
    Nonempty,
    read_verified_json,
)
from quotemem.core.errors import QuoteMemError
from quotemem.core.identity import canonical_json
from quotemem.store.metadata import StrictModel

_CATEGORIES = (
    "abstention",
    "contradiction_resolution",
    "event_ordering",
    "information_extraction",
    "instruction_following",
    "knowledge_update",
    "multi_session_reasoning",
    "preference_following",
    "summarization",
    "temporal_reasoning",
)
_MONTHS = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)
_REFERENCE_FIELDS = (
    "answer",
    "ideal_answer",
    "ideal_response",
    "ideal_summary",
    "expected_compliance",
)
_TEXT: TypeAdapter[str] = TypeAdapter(Nonempty)
_RUBRIC: TypeAdapter[list[str]] = TypeAdapter(Annotated[list[Nonempty], Field(min_length=1)])


class _Message(StrictModel):
    role: Literal["user", "assistant"]
    id: Annotated[int, Field(ge=0)]
    content: Nonempty
    time_anchor: Nonempty | None = None
    index: str | None = None
    question_type: str | None = None


class _Batch(StrictModel):
    batch_number: Annotated[int, Field(ge=1)]
    turns: Annotated[list[Annotated[list[_Message], Field(min_length=1)]], Field(min_length=1)]
    time_anchor: str | None = None


def _anchor(value: str | None) -> int | None:
    if value is None:
        return None
    month, day, year = value.split("-")
    return int(
        datetime(int(year), _MONTHS.index(month) + 1, int(day), tzinfo=UTC).timestamp() * 1000
    )


def _references(value: JsonValue) -> tuple[int, ...]:
    groups = value.values() if isinstance(value, dict) else (value,)
    result: list[int] = []
    for group in groups:
        if not isinstance(group, list):
            raise ValueError("BEAM references must be lists of source message IDs")
        for item in group:
            # Some publisher ordering concepts cite several messages as one list item.
            # Flatten lookup IDs only; source_question_json retains the complete grouping.
            if isinstance(item, list) and not item:
                raise ValueError("BEAM grouped source references cannot be empty")
            identifiers = item if isinstance(item, list) else (item,)
            for identifier in identifiers:
                if type(identifier) is not int or identifier < 0:
                    raise ValueError("BEAM source message IDs must be nonnegative integers")
                if identifier not in result:
                    result.append(identifier)
    return tuple(result)


def normalize_beam_conversation(
    chat: JsonValue, probes: JsonValue, *, conversation_id: int
) -> tuple[BenchmarkCase, ...]:
    """Preserve full final-chat history, original anchors, and every ordered rubric item.

    Args:
        chat (JsonValue): Exact selected publisher chat JSON, with nested batches and turns.
        probes (JsonValue): Corresponding publisher question mapping, never ingested as history.
        conversation_id (int): Numeric source conversation identity within the 128K scale.

    Returns:
        tuple[BenchmarkCase, ...]: Cases in publisher category/question order. Missing
            evidence references remain unresolved; absent message dates remain absent.

    Raises:
        ValueError: For malformed source structures, duplicate identities or unsupported dates.
    """
    if type(conversation_id) is not int or not 1 <= conversation_id <= 20:
        raise ValueError("BEAM conversation is outside the declared 128K selection")
    batches = TypeAdapter(list[_Batch]).validate_python(chat)
    questions = TypeAdapter(dict[str, list[dict[str, JsonValue]]]).validate_python(probes)
    if not batches or set(questions) != set(_CATEGORIES):
        raise ValueError("BEAM requires chat history and all ten probing abilities")
    cluster_id = f"beam:128k:{conversation_id}"
    sessions: list[BenchmarkSession] = []
    message_sessions: dict[str, str] = {}
    for batch in batches:
        session_id = f"{cluster_id}:batch-{batch.batch_number}"
        if session_id in {session.session_id for session in sessions}:
            raise ValueError("BEAM repeats a source batch identity")
        messages: list[BenchmarkMessage] = []
        for turn in batch.turns:
            for source in turn:
                message_id = f"{cluster_id}:message-{source.id}"
                if message_id in message_sessions:
                    raise ValueError("BEAM repeats a source message identity")
                message_sessions[message_id] = session_id
                messages.append(
                    BenchmarkMessage(
                        message_id,
                        source.role,
                        source.content,
                        source.time_anchor,
                        _anchor(source.time_anchor),
                    )
                )
        # Batch anchors do not establish timestamps for every contained message.
        # Only explicit message anchors are replayed; no later anchor is backfilled.
        sessions.append(BenchmarkSession(session_id, None, None, tuple(messages)))
    cases: list[BenchmarkCase] = []
    for category, items in questions.items():
        if not items:
            raise ValueError("BEAM probing abilities cannot be empty")
        for index, question in enumerate(items):
            query = _TEXT.validate_python(question.get("question"))
            rubric = _RUBRIC.validate_python(question.get("rubric"))
            answers = tuple(
                _TEXT.validate_python(question[field])
                for field in _REFERENCE_FIELDS
                if field in question
            )
            if not answers:
                raise ValueError("BEAM question lacks its publisher reference prose")
            references = tuple(
                f"{cluster_id}:message-{identifier}"
                for identifier in _references(question.get("source_chat_ids", []))
            )
            cases.append(
                BenchmarkCase(
                    case_id=f"{cluster_id}:{category}:{index}",
                    cluster_id=cluster_id,
                    sessions=tuple(sessions),
                    query=query,
                    answers=answers,
                    evidence_message_ids=references,
                    evidence_session_ids=tuple(
                        dict.fromkeys(
                            message_sessions[ref] for ref in references if ref in message_sessions
                        )
                    ),
                    category=category,
                    is_abstention=category == "abstention",
                    unresolved_evidence_ids=tuple(
                        ref for ref in references if ref not in message_sessions
                    ),
                    evaluation_kind="beam_rubric",
                    evaluation_rubric=tuple(rubric),
                    source_question_json=canonical_json(question),
                )
            )
    return tuple(cases)


def load_beam_bundle(directory: Path, source: DatasetBundleSource) -> BenchmarkCorpus:
    """Verify and load all 20 selected chats and all probes, with no source substitution.

    Args:
        directory (Path): Local source directory; no network access occurs here.
        source (DatasetBundleSource): Frozen publisher selection with every raw file hash.

    Returns:
        BenchmarkCorpus: Complete source-bound 128K subset with rubric evaluation metadata.

    Raises:
        QuoteMemError: For incomplete, malformed or changed parts and inconsistent case counts.
    """
    try:
        source = DatasetBundleSource.model_validate(source.model_dump())
        parts = {part.path: part for part in source.parts}
        used: set[str] = set()
        cases: list[BenchmarkCase] = []
        for identifier in range(1, 21):
            base = f"chats/100K/{identifier}/"
            candidates = [
                base + name
                for name in ("chat_trunecated.json", "chat.json")
                if base + name in parts
            ]
            question_path = base + "probing_questions/probing_questions.json"
            if len(candidates) != 1 or question_path not in parts:
                raise ValueError("BEAM bundle must declare one selected chat and probes per ID")
            chat_path = candidates[0]
            used.update((chat_path, question_path))
            decoded: list[JsonValue] = []
            for name in (chat_path, question_path):
                part = parts[name]
                decoded.append(
                    read_verified_json(
                        bundle_part_path(directory, part),
                        size_bytes=part.size_bytes,
                        sha256=part.sha256,
                    )
                )
            cases.extend(
                normalize_beam_conversation(decoded[0], decoded[1], conversation_id=identifier)
            )
        if used != set(parts) or len(cases) != source.expected_cases:
            raise ValueError("BEAM selection or case count differs from its manifest")
        return BenchmarkCorpus(source, tuple(cases))
    except QuoteMemError:
        raise
    except (OSError, ValueError, TypeError, UnicodeError, RecursionError):
        raise QuoteMemError("dataset_invalid", "BEAM source failed strict loading") from None
