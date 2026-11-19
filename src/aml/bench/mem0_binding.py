"""Pinned Mem0 OSS conditions and source-bound local retrieval artifacts.

The optional package is never imported here. These checks establish consistency,
not installation, provider execution, store freshness or memory quality.
"""

from __future__ import annotations

import hashlib
from typing import Annotated, Literal, Self
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from aml.adapters.http.models import SearchResponseV1
from aml.bench.datasets import DatasetSource, Digest, Nonempty, decode_source_json
from aml.bench.replay import ReplayConfig
from aml.core.identity import canonical_json

MEM0_VERSION = "2.1.0"
MEM0_COMMIT = "19f713408273fb1d657daa38d7b82ccf496d36d5"
Mem0Origin = Literal["mem0_oss_v2_1_0"]
BaselineArrival = Literal["all_adds_then_search", "immediate_first_search"]
Positive = Annotated[int, Field(gt=0)]
Count = Annotated[int, Field(ge=0)]


def mem0_digest(value: object) -> str:
    """Return the canonical SHA-256 of one non-secret baseline artifact."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, hide_input_in_errors=True)


class Mem0Endpoint(_Frozen):
    """Explicit compatible provider; no model, endpoint or credential fallback.

    `provider` is a provenance label. Both supported labels use Mem0's OpenAI
    wire adapter with the exact HTTPS endpoint and named credential environment.
    """

    provider: Literal["model_studio", "openai"]
    base_url: Nonempty
    credential_env: Annotated[str, Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")]
    requested_model: Nonempty
    expected_resolved_model: Nonempty
    timeout_ms: Annotated[int, Field(gt=0, le=600_000)]
    max_calls: Positive
    max_request_utf8_bytes: Positive
    max_response_utf8_bytes: Positive
    max_observed_input_tokens: Positive

    @field_validator("base_url")
    @classmethod
    def explicit_https(cls, value: str) -> str:
        parsed = urlsplit(value)
        if (
            parsed.scheme != "https" or not parsed.hostname or parsed.username
            or parsed.password or parsed.query or parsed.fragment or value.endswith("/")
        ):
            raise ValueError("Mem0 endpoints require exact HTTPS roots without credentials")
        return value


class Mem0Configuration(_Frozen):
    """Complete supported configuration; extraction is separate from answer generation.

    Paths derive from a fresh registered workspace. SDK retries, telemetry,
    reranking, graph, fastembed and spaCy extras are disabled for this method.
    Observed-token limits stop later calls; they are not prepaid billing caps.
    """

    workspace: Nonempty
    collection_name: Annotated[str, Field(pattern=r"^[A-Za-z0-9_-]{1,128}$")]
    proxy_env: Annotated[str | None, Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")]
    extraction: Mem0Endpoint
    embedding: Mem0Endpoint
    embedding_dimensions: Annotated[int, Field(gt=0, le=65536)]
    temperature: Annotated[float, Field(ge=0, le=2, allow_inf_nan=False)]
    top_p: Annotated[float, Field(gt=0, le=1, allow_inf_nan=False)]
    max_output_tokens_per_call: Positive
    max_observed_output_tokens: Positive
    max_run_seconds: Annotated[int, Field(gt=0, le=86400)]
    threshold: Annotated[float, Field(ge=0.1, le=0.1, allow_inf_nan=False)]
    infer: Literal[True]
    rerank: Literal[False]
    explain: Literal[False]
    sdk_max_retries: Literal[0]
    optional_nlp: Literal["absent"]
    telemetry: Literal[False]

    @field_validator("infer", "rerank", "explain", "telemetry", mode="before")
    @classmethod
    def exact_boolean_flags(cls, value: object) -> bool:
        if type(value) is not bool:
            raise ValueError("Mem0 execution flags require exact boolean values")
        return value

    @field_validator("sdk_max_retries", mode="before")
    @classmethod
    def exact_retry_count(cls, value: object) -> int:
        if type(value) is not int:
            raise ValueError("Mem0 retry count requires an exact integer")
        return value


class Mem0Method(_Frozen):
    """Exact external build and adapter inputs, without credentials or answer labels."""

    package: Literal["mem0ai"]
    version: Literal["2.1.0"]
    source_commit: Literal["19f713408273fb1d657daa38d7b82ccf496d36d5"]
    wheel_sha256: Digest
    dependency_lock_sha256: Digest
    installed_distributions_sha256: Digest
    adapter_source_sha256: Digest
    configuration: Mem0Configuration
    temporal_semantics: Literal["timestamps_retained_as_metadata_only"]
    usage_limit_semantics: Literal["calls_and_bytes_bounded_tokens_stop_after_response"]
    cost_accounting: Literal["unknown_without_billing_witness"]


class Mem0InputGroup(_Frozen):
    """Exact prepared group, including ordered source units and Add identities."""

    cluster_id: Nonempty
    user_id: Nonempty
    case_ids: Annotated[tuple[Nonempty, ...], Field(min_length=1)]
    add_request_ids: Annotated[tuple[Nonempty, ...], Field(min_length=1)]
    add_input_sha256: Digest
    source_units_sha256: Digest


class Mem0Registration(_Frozen):
    """Frozen complete source, replay, arrival and optional-runtime conditions."""

    schema_version: Literal[1]
    source: DatasetSource
    case_ids: Annotated[tuple[Nonempty, ...], Field(min_length=1)]
    run_key: Nonempty
    trial_key: Nonempty
    replay_config: ReplayConfig
    arrival: BaselineArrival
    groups: Annotated[tuple[Mem0InputGroup, ...], Field(min_length=1)]
    method: Mem0Method

    @field_validator("schema_version", mode="before")
    @classmethod
    def exact_schema_version(cls, value: object) -> int:
        if type(value) is not int:
            raise ValueError("Mem0 registration version requires an exact integer")
        return value

    @model_validator(mode="after")
    def complete_denominator(self) -> Self:
        selected = self.source.conversation_selection
        if (
            len(set(self.case_ids)) != len(self.case_ids)
            or len(self.case_ids) != self.source.expected_cases
            or selected is not None and tuple(selected.case_ids) != self.case_ids
            or sorted(case for group in self.groups for case in group.case_ids)
            != sorted(self.case_ids)
            or len({group.user_id for group in self.groups}) != len(self.groups)
            or any(len(set(group.add_request_ids)) != len(group.add_request_ids)
                   for group in self.groups)
        ):
            raise ValueError("Mem0 registration requires the full ordered source denominator")
        return self


class Mem0SearchReceipt(_Frozen):
    """Successful local API receipt; source and raw-result identities are independently bound."""

    operation: Literal["search"]
    sequence: Count
    case_id: Nonempty
    user_id: Nonempty
    request_sha256: Digest
    raw_result_sha256: Digest
    completed_add_ids: tuple[Nonempty, ...]
    elapsed_ms: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    provider_call_keys: tuple[Nonempty, ...]


def mem0_search_response(raw_json: str, *, top_k: int) -> SearchResponseV1:
    """Project finite ordered Mem0 hits without altering content, IDs or membership.

    Args:
        raw_json: Canonical JSON of the actual local API return value.
        top_k: Registered maximum result count.

    Returns:
        SearchResponseV1: Ordered stable IDs, original memory text and observed scores.

    Raises:
        ValueError: The result is malformed, nonfinite, or exceeds the requested count.
    """
    raw = decode_source_json(raw_json.encode("utf-8"))
    if not isinstance(raw, dict) or set(raw) != {"results"}:
        raise ValueError("Mem0 search requires its exact results envelope")
    results = raw["results"]
    if not isinstance(results, list) or len(results) > top_k:
        raise ValueError("Mem0 search results exceed the registered shape or count")
    hits: list[dict[str, object]] = []
    for item in results:
        if not isinstance(item, dict) or type(item.get("score")) not in (int, float):
            raise ValueError("Mem0 search hit requires an observed finite numeric score")
        hits.append({"id": item.get("id"), "content": item.get("memory"), "score": item["score"]})
    return SearchResponseV1.model_validate({"data": hits})


class Mem0RetrievalBinding(_Frozen):
    """Sealed local result plus the exact registered external-memory experiment."""

    kind: Mem0Origin
    add_input_sha256: Digest
    replay_config_json: str
    source_units_sha256: Digest
    registration_json: str = Field(repr=False)
    registration_sha256: Digest
    raw_result_json: str = Field(repr=False)
    receipt: Mem0SearchReceipt

    @model_validator(mode="after")
    def bound_artifacts(self) -> Self:
        registration = Mem0Registration.model_validate_json(self.registration_json)
        groups = [group for group in registration.groups if self.receipt.case_id in group.case_ids]
        if (
            len(groups) != 1
            or self.registration_sha256 != mem0_digest(registration.model_dump(mode="json"))
            or self.registration_json != canonical_json(registration.model_dump(mode="json"))
            or ReplayConfig.model_validate_json(self.replay_config_json)
            != registration.replay_config
            or self.replay_config_json
            != canonical_json(registration.replay_config.model_dump(mode="json"))
            or self.receipt.raw_result_sha256
            != hashlib.sha256(self.raw_result_json.encode()).hexdigest()
        ):
            raise ValueError("Mem0 result differs from its complete registration or raw result")
        group = groups[0]
        if (
            self.add_input_sha256 != group.add_input_sha256
            or self.source_units_sha256 != group.source_units_sha256
            or self.receipt.user_id != group.user_id
            or self.receipt.completed_add_ids != group.add_request_ids
        ):
            raise ValueError("Mem0 search lacks its exact completed source inputs")
        mem0_search_response(self.raw_result_json, top_k=registration.replay_config.top_k)
        return self
