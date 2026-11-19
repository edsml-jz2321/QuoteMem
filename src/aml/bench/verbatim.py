"""A local raw-source BM25 comparator, separate from AML Search and persistence.

Only the exact message segments submitted by the replay builder become documents.
Results are local serialized artifacts, never evidence of HTTP delivery, source
visibility, publication, or recovery. Answer generation and its common context
budget belong to the benchmark answer runner.
"""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field, replace
from decimal import ROUND_HALF_EVEN, Context, Decimal

from pydantic import ValidationError

from aml.adapters.http.models import SearchRequestV1, SearchResponseV1
from aml.bench.datasets import BenchmarkCorpus, source_identity, source_identity_kind
from aml.bench.replay import ReplayGroup, prepare_replay, replay_add_input_sha256
from aml.core.errors import AMLError
from aml.core.identity import canonical_json
from aml.core.source_participants import SourceMetadataVersion
from aml.ingestion.boundary import AddRequestV1

METHOD_JSON = canonical_json(
    {
        "method": "verbatim_bm25_v1",
        "documents": "one exact Add message segment; distinct duplicates retained",
        "tokenizer": "Python 3.12 Unicode NFKC, casefold, regex \\w+",
        "k1": 1.2,
        "b": 0.75,
        "idf": "log1p((N - df + 0.5) / (df + 0.5))",
        "query_terms": "distinct tokens sorted lexicographically; no multiplicity weight",
        "options": "retained in request provenance; ignored by this comparator",
        "statistics_scope": "one exact replay user and group",
        "admission": "strictly positive unrounded BM25 score",
        "score": "Decimal(str(math.fsum(term_scores))), explicit precision 40, 6 places, half even",
        "order": "quantized score descending, request_id ascending, message_index ascending",
        "result_budget": "Search top_k; full selected segments, no content truncation",
        "answer_context_budget": "applied separately by the same answer runner for both arms",
    }
)
METHOD_SHA256 = hashlib.sha256(METHOD_JSON.encode()).hexdigest()


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def _tokens(value: str) -> tuple[str, ...]:
    return tuple(re.findall(r"\w+", unicodedata.normalize("NFKC", value).casefold()))


@dataclass(frozen=True)
class VerbatimSource:
    """A ranked Add segment, with no guessed publisher message/span correspondence."""

    result_id: str
    request_id: str
    message_index: int
    content_sha256: str


@dataclass(frozen=True)
class VerbatimResult:
    """Exact local request/result artifact bytes and their source/configuration binding."""

    case_id: str
    user_id: str
    add_input_sha256: str
    source_units_sha256: str
    request_json: str = field(repr=False)
    response_json: str = field(repr=False)
    sources: tuple[VerbatimSource, ...]
    document_count: int
    matched_document_count: int


@dataclass(frozen=True)
class VerbatimReport:
    """Complete local retrieval cohort; it has no HTTP or application timing receipts."""

    dataset_identity: str
    dataset_identity_kind: str
    dataset_variant: str
    run_id: str
    isolate_queries: bool
    group_count: int
    results: tuple[VerbatimResult, ...]
    source_selection_json: str = field(repr=False)
    method_json: str = METHOD_JSON
    method_sha256: str = METHOD_SHA256


@dataclass(frozen=True)
class _Document:
    source: VerbatimSource
    content: str = field(repr=False)
    terms: Counter[str] = field(repr=False)


def _validate_group(group: ReplayGroup) -> ReplayGroup:
    """Detach and validate every request before deriving identities or scoring."""
    try:
        adds = tuple(
            AddRequestV1.model_validate(request.model_dump(mode="python", warnings=False))
            for request in group.adds
        )
        searches = tuple(
            (case_id, SearchRequestV1.model_validate(
                request.model_dump(mode="python", warnings=False)
            ))
            for case_id, request in group.searches
        )
    except ValidationError:
        raise AMLError("baseline_invalid", "Verbatim requests contain invalid fields") from None
    group = replace(group, adds=adds, searches=searches)
    if (
        not group.adds
        or not group.searches
        or len({item.request_id for item in group.adds}) != len(group.adds)
        or any(type(case_id) is not str or not case_id for case_id, _ in group.searches)
        or len({case_id for case_id, _ in group.searches}) != len(group.searches)
        or any(message.role not in {"user", "assistant"}
               for request in group.adds for message in request.messages)
    ):
        raise AMLError(
            "baseline_invalid", "Verbatim input has invalid requests or repeated identities"
        )
    users = {item.user_id for item in group.adds} | {
        request.user_id for _, request in group.searches
    }
    if len(users) != 1:
        raise AMLError("baseline_invalid", "Verbatim input must have one exact replay user")
    return group


def _segments(group: ReplayGroup) -> Iterator[tuple[VerbatimSource, str]]:
    """Derive source identities without tokenization or relevance scoring."""
    # Canonical order makes the corpus binding independent of request arrival order.
    for request in sorted(group.adds, key=lambda request: request.request_id):
        for index, message in enumerate(request.messages):
            content_hash = hashlib.sha256(message.content.encode()).hexdigest()
            result_id = "verbatim:" + _digest(
                [request.user_id, request.request_id, index, content_hash]
            )
            yield (
                VerbatimSource(result_id, request.request_id, index, content_hash), message.content
            )


def _source_digest(sources: Iterable[VerbatimSource]) -> str:
    """Bind the complete ordered source-unit metadata using the existing projection."""
    return _digest(
        [
            {
                "result_id": source.result_id,
                "request_id": source.request_id,
                "message_index": source.message_index,
                "content_sha256": source.content_sha256,
            }
            for source in sources
        ]
    )


def verbatim_input_binding(group: ReplayGroup) -> tuple[str, str]:
    """Reconstruct comparator input identity without scoring or calling providers.

    Args:
        group (ReplayGroup): Complete prepared Add/Search cohort for one exact user.

    Returns:
        tuple[str, str]: Source-unit SHA256 and exact method JSON, in that order.
            These validate declared local inputs, not actual HTTP execution.

    Raises:
        AMLError: For mixed users, repeated identities, invalid requests or empty input.
    """
    group = _validate_group(group)
    return _source_digest(source for source, _ in _segments(group)), METHOD_JSON


def retrieve_group(group: ReplayGroup) -> tuple[VerbatimResult, ...]:
    """Rank only one prepared group's exact Add segments using the frozen local method.

    Args:
        group (ReplayGroup): Matching Add and Search requests for one exact user.
            Every request ID must occur once, including identical repeated requests.

    Returns:
        tuple[VerbatimResult, ...]: One result per query in input order. No lexical
            match produces an empty strict response. Raw content remains unchanged.

    Raises:
        AMLError: For mixed users, repeated identities, invalid requests or empty input.
    """
    group = _validate_group(group)
    add_input_sha256 = replay_add_input_sha256(group)
    documents = [
        _Document(source, content, Counter(_tokens(content)))
        for source, content in _segments(group)
    ]
    source_units_sha256 = _source_digest(document.source for document in documents)
    count = len(documents)
    document_frequency: Counter[str] = Counter()
    for document in documents:
        document_frequency.update(document.terms.keys())
    average_length = sum(sum(document.terms.values()) for document in documents) / count
    score_context = Context(
        prec=40, rounding=ROUND_HALF_EVEN, Emin=-999999, Emax=999999, capitals=1, clamp=0,
        flags=[], traps=[],
    )
    results: list[VerbatimResult] = []
    for case_id, request in group.searches:
        query = sorted(set(_tokens(request.query)))
        ranked: list[tuple[Decimal, _Document]] = []
        for document in documents:
            if not average_length:
                continue
            length = sum(document.terms.values())
            terms: list[float] = []
            for term in query:
                frequency = document.terms[term]
                if frequency:
                    df = document_frequency[term]
                    idf = math.log1p((count - df + 0.5) / (df + 0.5))
                    denominator = frequency + 1.2 * (0.25 + 0.75 * length / average_length)
                    terms.append(idf * frequency * 2.2 / denominator)
            score = math.fsum(terms)
            if score > 0:
                rounded = score_context.quantize(Decimal(str(score)), Decimal("0.000001"))
                if not rounded.is_finite():
                    raise AMLError("baseline_invalid", "Verbatim score exceeds its numeric domain")
                ranked.append((rounded, document))
        ranked.sort(
            key=lambda item: (
                item[0].copy_negate(), item[1].source.request_id, item[1].source.message_index
            )
        )
        selected = ranked[: request.top_k]
        response_json = canonical_json(
            {
                "data": [
                    {
                        "id": document.source.result_id,
                        "content": document.content,
                        "score": float(score),
                    }
                    for score, document in selected
                ]
            }
        )
        SearchResponseV1.from_body(response_json.encode())
        results.append(
            VerbatimResult(
                case_id,
                request.user_id,
                add_input_sha256,
                source_units_sha256,
                canonical_json(request.model_dump(mode="json", exclude_none=True)),
                response_json,
                tuple(document.source for _, document in selected),
                count,
                len(ranked),
            )
        )
    return tuple(results)


def retrieve_verbatim(
    corpus: BenchmarkCorpus,
    *,
    run_id: str,
    top_k: int = 10,
    max_top_k: int = 100,
    isolate_queries: bool = False,
    source_metadata_version: SourceMetadataVersion | None = None,
) -> VerbatimReport:
    """Build the same replay groups as HTTP and retrieve every local comparator query.

    Args:
        corpus (BenchmarkCorpus): Complete source-verified corpus with unique case IDs.
        run_id (str): Same explicit namespace as the compared HTTP run.
        top_k (int): Same requested result limit as the compared HTTP run.
        max_top_k (int): Trusted active HTTP policy limit.
        isolate_queries (bool): Same query isolation setting as the HTTP run.
        source_metadata_version: Same explicit Add metadata contract as the HTTP run.

    Returns:
        VerbatimReport: Source/configuration-bound local results for all corpus cases.

    Raises:
        AMLError: For an incomplete corpus or invalid replay/baseline inputs.
    """
    if (
        len(corpus.cases) != corpus.source.expected_cases
        or not corpus.cases
        or len({case.case_id for case in corpus.cases}) != len(corpus.cases)
    ):
        raise AMLError("baseline_invalid", "Verbatim retrieval requires the complete source cohort")
    groups = prepare_replay(
        corpus, run_id=run_id, top_k=top_k, max_top_k=max_top_k, isolate_queries=isolate_queries,
        source_metadata_version=source_metadata_version,
    )
    return VerbatimReport(
        source_identity(corpus.source),
        source_identity_kind(corpus.source),
        corpus.source.variant,
        run_id,
        isolate_queries,
        len(groups),
        tuple(result for group in groups for result in retrieve_group(group)),
        canonical_json(corpus.source.model_dump(mode="json")),
    )
