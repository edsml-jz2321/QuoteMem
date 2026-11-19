"""Immutable resource registrations and observations that the bench evaluators consume.

These file artifacts do not observe providers or scheduler transitions. The real
collector must retain and verify their referenced witnesses. Parsing supplied
observations cannot establish storage or milestone correctness.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Annotated, Literal, Self, cast

from pydantic import ConfigDict, Field, ValidationInfo, field_validator, model_validator

from aml.bench.datasets import Digest
from aml.bench.quality_evaluation import FrozenTuple
from aml.core.identity import canonical_json, scope_id
from aml.core.query import ScopeV1
from aml.service.cost_attribution import OWNER_PHASES, ResourcePhase
from aml.store.metadata import StrictModel

Count = Annotated[int, Field(ge=0)]
Key = Annotated[str, Field(min_length=1)]
Cost = Annotated[float, Field(ge=0, allow_inf_nan=False)]
Milliseconds = Annotated[float, Field(ge=0, allow_inf_nan=False)]


class _Artifact(StrictModel):
    model_config = ConfigDict(frozen=True, hide_input_in_errors=True)

    def _validated(self) -> Self:
        """Validate copied fields before JSON serialization can normalize their types."""
        return type(self).model_validate(self.model_dump(mode="python", warnings=False))

    @property
    def digest(self) -> str:
        """Return the canonical identity of this complete immutable artifact.

        Args:
            None.

        Returns:
            SHA-256 of the canonical JSON bytes, excluding this derived property.
        """
        checked = self._validated()
        return hashlib.sha256(canonical_json(checked.model_dump(mode="json")).encode()).hexdigest()

    def save(self, path: Path) -> None:
        """Write a new evidence artifact without replacing an earlier registration.

        Args:
            path: New output path whose parent directory already exists.

        Returns:
            None after writing canonical JSON. This does not verify witness files
            or prove the collection's timing; the real collector owns those checks.

        Raises:
            ValueError: A copied or constructed artifact fails strict validation.
            OSError: The path exists or cannot be written.
        """
        checked = self._validated()
        with path.open("x", encoding="utf-8") as output:
            output.write(canonical_json(checked.model_dump(mode="json")) + "\n")


class CorpusCostArtifactEncodingV1(StrictModel):
    """Lossless cost-file encoding with a separately registered decoded-byte bound."""

    model_config = ConfigDict(frozen=True)
    encoding: Literal["canonical-json-gzip-v1"]
    decoded_max_bytes: Annotated[int, Field(ge=1024, le=1024 * 1024 * 1024)]


class CorpusResourceRegistrationV1(StrictModel):
    """Exact ordinary trial provenance and finite capture budgets, fixed before calls.

    An absent cost mode preserves the original detailed capture. Essential mode
    bounds active cost tokens and retained phase intervals by attribution_capacity;
    its fixed operation-count table is not a full fine-grained history.
    """

    model_config = ConfigDict(frozen=True)
    schema_version: Literal[1]
    run_key: Annotated[str, Field(min_length=1)]
    source_commit: Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
    source_identity: Digest
    trial_sha256: Digest
    provider_bindings_sha256: Digest
    registered_at_ms: Annotated[int, Field(ge=1)]
    scopes: Annotated[FrozenTuple[ScopeV1], Field(min_length=1)]
    cohorts: Annotated[FrozenTuple[str], Field(min_length=1)]
    provider_snapshots: Annotated[FrozenTuple[str], Field(min_length=1)]
    call_capacity: Annotated[int, Field(ge=1, le=10_000_000)]
    session_capacity: Annotated[int, Field(ge=1, le=10_000_000)]
    cycle_capacity: Annotated[int, Field(ge=1, le=10_000_000)]
    transition_capacity: Annotated[int, Field(ge=1, le=10_000_000)]
    attribution_capacity: Annotated[int, Field(ge=1, le=10_000_000)]
    artifact_max_bytes: Annotated[int, Field(ge=1024, le=1024 * 1024 * 1024)]
    cost_artifact_encoding: CorpusCostArtifactEncodingV1 | None = Field(
        default=None, exclude_if=lambda value: value is None,
    )
    cost_capture_mode: Literal["essential-summary-v1"] | None = Field(
        default=None, exclude_if=lambda value: value is None,
    )

    @model_validator(mode="after")
    def exact_population(self) -> Self:
        if self.cost_capture_mode is not None and self.cost_artifact_encoding is not None:
            raise ValueError("Essential summaries do not encode an absent detailed history")
        if len(set(self.cohorts)) != len(self.cohorts) or self.scopes != tuple(
            ScopeV1(scope_kind="user", scope_id=scope_id("user", cohort)) for cohort in self.cohorts
        ) or len(set(self.provider_snapshots)) != len(self.provider_snapshots):
            raise ValueError("Corpus resources require exact unique cohorts and providers")
        return self


class ProviderCallObservation(_Artifact):
    """One actual provider-boundary invocation, including failed attempts.

    A shared single-flight invocation keeps the same call_key wherever referenced.
    Retries are distinct invocations. Usage and USD charges come from retained
    actual boundary/billing witnesses; missing values are never estimated from
    model names, reuse labels, successful responses, or configured prices.
    """

    call_key: Key
    witness_sha256: Digest
    provider_snapshot: Key
    phase: ResourcePhase
    cycle_key: Key | None
    outcome: Literal["succeeded", "failed", "cancelled"]
    input_tokens: Count | None
    output_tokens: Count | None
    usage_witness_sha256: Digest | None
    cost_usd: Cost | None
    cost_witness_sha256: Digest | None

    @model_validator(mode="after")
    def observed_usage(self) -> Self:
        """Reject unsupported usage/cost claims and owner phases without their cycle."""
        if (self.input_tokens is not None or self.output_tokens is not None) and (
            self.usage_witness_sha256 is None
        ):
            raise ValueError("Reported provider usage requires a retained usage witness")
        if self.cost_usd is not None and self.cost_witness_sha256 is None:
            raise ValueError("Reported provider cost requires a retained charge witness")
        if (self.phase in OWNER_PHASES) != (self.cycle_key is not None):
            raise ValueError("Exactly the owner-phase provider calls carry their owner cycle")
        return self


class FinalizeCycleObservation(_Artifact):
    """One observed owner cycle: its admission trigger and whether it completed."""

    cycle_key: Key
    witness_sha256: Digest
    trigger: Literal["first_search", "idle", "late_add", "add_completion"]
    outcome: Literal["complete", "incomplete"]


class PendingCounts(_Artifact):
    """Simultaneous session counts at an observed pending peak, not worker slots."""

    queued: Count
    executing: Count

    @property
    def total(self) -> int:
        """Count every durable pending state without imposing the execution cap.

        Args:
            None.

        Returns:
            Queued plus executing sessions.
        """
        checked = self._validated()
        return checked.queued + checked.executing


class UserExecutionPeak(_Artifact):
    """One registered fresh workspace/user cohort's observed execution peak."""

    cohort_key: Key
    executing_peak: Count


class ExecutionObservation(_Artifact):
    """Observed global/user peaks, with exact transition or sampled provenance.

    Complete transition capture must cover the whole registered experiment,
    including every owner cycle and the last drain.
    Sampled values are only observed lower bounds, even when below configured caps.
    """

    witness_sha256: Digest
    capture: Literal["complete_transitions", "sampled"]
    global_executing_peak: Count
    per_user: tuple[UserExecutionPeak, ...]
    pending_peak: PendingCounts

    @field_validator("per_user", mode="before")
    @classmethod
    def json_arrays(cls, value: object, info: ValidationInfo) -> object:
        """Restore JSON array handling after shared exact-literal validation."""
        if info.mode == "json" and isinstance(value, list):
            return tuple(cast(list[object], value))
        return value

    @model_validator(mode="after")
    def consistent_peaks(self) -> Self:
        """Reject duplicate user peaks and contradictory observed process totals."""
        if len({user.cohort_key for user in self.per_user}) != len(self.per_user):
            raise ValueError("Each execution observation requires unique user cohorts")
        if (
            max((user.executing_peak for user in self.per_user), default=0)
            > (self.global_executing_peak)
            or self.pending_peak.executing > self.global_executing_peak
        ):
            raise ValueError(
                "A simultaneous user or pending execution count exceeds the global peak"
            )
        if self.capture == "complete_transitions" and (
            self.global_executing_peak > sum(user.executing_peak for user in self.per_user)
            or self.global_executing_peak > self.pending_peak.total
        ):
            raise ValueError("Complete execution peaks exceed their user or pending totals")
        return self


class DrainObservation(_Artifact):
    """A completed drain or right-censored elapsed observation for one cohort.

    Complete means every expected unit is durably settled by its owner's commit
    and final marker. Zero active workers alone does not mean complete drain.
    Censored elapsed time is only a lower bound.
    """

    cohort_key: Key
    witness_sha256: Digest
    outcome: Literal["complete", "censored"]
    elapsed_ms: Milliseconds


class ResourceObservation(_Artifact):
    """Explicit observations for one resource registration, without synthetic defaults.

    Complete provider history includes the request phases (add, search) and every
    owner phase of every cycle.
    Completeness is an assertion by the retained real observer, not this parser.
    """

    schema_version: Literal[1] = 1
    resource_registration_digest: Digest
    observer_witness_sha256: Digest
    collection_started_at_ms: Count
    collection_ended_at_ms: Count
    provider_history_complete: bool
    provider_calls: tuple[ProviderCallObservation, ...]
    cycle_history_complete: bool
    cycles: tuple[FinalizeCycleObservation, ...]
    execution: ExecutionObservation | None
    drains: tuple[DrainObservation, ...]

    @field_validator("provider_calls", "cycles", "drains", mode="before")
    @classmethod
    def json_arrays(cls, value: object, info: ValidationInfo) -> object:
        """Adapt only JSON arrays; Python containers and nested values stay strict."""
        if info.mode == "json" and isinstance(value, list):
            return tuple(cast(list[object], value))
        return value

    @model_validator(mode="after")
    def ordered_collection(self) -> Self:
        """Require a nondecreasing retained collection interval."""
        if self.collection_ended_at_ms < self.collection_started_at_ms:
            raise ValueError("Resource collection ends before it starts")
        return self


