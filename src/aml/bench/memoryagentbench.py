"""The explicitly selected two-context MemoryAgentBench Conflict Resolution subset."""

from __future__ import annotations

import importlib
from typing import IO, Annotated

from pydantic import Field, JsonValue, TypeAdapter

from aml.bench.datasets import BenchmarkCase, BenchmarkMessage, BenchmarkSession, Nonempty
from aml.core.errors import AMLError
from aml.core.identity import canonical_json
from aml.store.metadata import StrictModel

SELECTED_SOURCES = ("factconsolidation_sh_6k", "factconsolidation_mh_6k")
_DATES: TypeAdapter[list[str] | None] = TypeAdapter(list[Nonempty] | None)
_JSON: TypeAdapter[JsonValue] = TypeAdapter(JsonValue)


class _Record(StrictModel):
    context: Nonempty
    questions: Annotated[list[Nonempty], Field(min_length=1)]
    answers: Annotated[list[Annotated[list[Nonempty], Field(min_length=1)]], Field(min_length=1)]
    metadata: dict[str, JsonValue]


def normalize_memoryagentbench(rows: JsonValue) -> tuple[BenchmarkCase, ...]:
    """Keep every QA and all context text for the two named publisher selections.

    Args:
        rows (JsonValue): Decoded Parquet rows containing context, questions, answers and metadata.

    Returns:
        tuple[BenchmarkCase, ...]: Cases in declared source order, then original QA order.
            Absent source dates remain absent and no supporting message labels are invented.

    Raises:
        ValueError: For absent or repeated selected contexts and inconsistent QA identities.
    """
    records = TypeAdapter(list[dict[str, JsonValue]]).validate_python(rows)
    selected: dict[str, _Record] = {}
    for row in records:
        metadata = row.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError("MemoryAgentBench row lacks source metadata")
        source = metadata.get("source")
        if not isinstance(source, str):
            raise ValueError("MemoryAgentBench row lacks its publisher source name")
        if source not in SELECTED_SOURCES:
            continue
        if source in selected:
            raise ValueError("MemoryAgentBench repeats a selected publisher context")
        selected[source] = _Record.model_validate(row)
    if set(selected) != set(SELECTED_SOURCES):
        raise ValueError("MemoryAgentBench is missing a selected publisher context")
    cases: list[BenchmarkCase] = []
    for source in SELECTED_SOURCES:
        record = selected[source]
        ids = TypeAdapter(list[Nonempty]).validate_python(record.metadata.get("qa_pair_ids"))
        dates = _DATES.validate_python(record.metadata.get("question_dates"))
        if (
            len(record.questions) != len(record.answers)
            or len(ids) != len(record.questions)
            or len(set(ids)) != len(ids)
            or dates
            and len(dates) != len(ids)
        ):
            raise ValueError("MemoryAgentBench parallel QA fields are inconsistent")
        cluster_id = f"memoryagentbench:{source}"
        # The source is document-style memory input, not an invented conversation.
        # Existing replay splits only at its declared whole-message/sentence boundaries.
        session = BenchmarkSession(
            f"{cluster_id}:context",
            None,
            None,
            (BenchmarkMessage(f"{cluster_id}:context", "user", record.context),),
        )
        for index, (identifier, query, answers) in enumerate(
            zip(ids, record.questions, record.answers, strict=True)
        ):
            cases.append(
                BenchmarkCase(
                    case_id=f"{cluster_id}:qa-{identifier}",
                    cluster_id=cluster_id,
                    sessions=(session,),
                    query=query,
                    answers=tuple(answers),
                    evidence_message_ids=(),
                    evidence_session_ids=(),
                    category=source,
                    query_source_date=dates[index] if dates else None,
                    evaluation_kind="mab_substring_exact_match",
                    source_question_json=canonical_json(
                        {
                            "qa_pair_id": identifier,
                            "question": query,
                            "answers": answers,
                            "metadata": record.metadata,
                        }
                    ),
                )
            )
    return tuple(cases)


def load_memoryagentbench(source: IO[bytes]) -> tuple[BenchmarkCase, ...]:
    """Decode the verified local Parquet shard using the explicit optional bench extra.

    Args:
        source (IO[bytes]): Borrowed seekable snapshot, already hash-verified by the
            central dataset loader. The reader never reopens the original pathname.

    Returns:
        tuple[BenchmarkCase, ...]: All cases in the two declared publisher contexts.

    Raises:
        AMLError: If the pinned optional dependency is unavailable or decoding fails.
    """
    try:
        parquet = importlib.import_module("pyarrow.parquet")
        arrow = importlib.import_module("pyarrow")
    except ImportError:
        raise AMLError(
            "benchmark_dependency_missing", "MemoryAgentBench requires the pinned bench extra"
        ) from None
    if arrow.__version__ != "25.0.1":
        raise AMLError("benchmark_dependency_mismatch", "Parquet reader differs from its pin")
    try:
        table = parquet.read_table(
            source, columns=["context", "questions", "answers", "metadata"], use_threads=False
        )
        return normalize_memoryagentbench(_JSON.validate_python(table.to_pylist()))
    except (OSError, ValueError, TypeError):
        raise AMLError(
            "dataset_invalid", "MemoryAgentBench Parquet failed strict loading"
        ) from None
