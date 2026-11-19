"""Closed source-label decisions and durable logical-call budget coordinates.

Model output may classify code-owned positions only. It carries no source text,
speaker name, entity identifier or authority to rewrite an original message.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal, Self

from pydantic import ConfigDict, Field, model_validator

from aml.core.errors import AMLError
from aml.core.identity import ULID, ArtifactId, ContentBindingDigest, integrity_digest
from aml.core.model_identity import admits_inference_model
from aml.core.provider_budget import EXTRACTION_REQUEST_CAP
from aml.core.semantic import SystemTimestamp
from aml.core.statement import MAX_MESSAGE_BYTES
from aml.store.metadata import OpaqueIdSpec, RefSpec, StrictModel

if TYPE_CHECKING:
    from aml.core.inference import InferenceBindingV1, StoredInferenceOutput

SOURCE_LABELS_VERSION = "source-labels-v2"
SOURCE_LABEL_LIMIT = 4096
type SourceLabelState = Literal["speaker", "content", "ambiguous"]


class SourceLabelDecisionV1(StrictModel):
    """Classify exactly one position selected by code, without naming an identity."""

    label_id: Annotated[int, Field(ge=0, lt=SOURCE_LABEL_LIMIT), OpaqueIdSpec("local")]
    state: SourceLabelState


class SourceLabelsOutputV1(StrictModel):
    """Return the complete ordered semantic decisions; ambiguous is a final result."""

    decisions: Annotated[list[SourceLabelDecisionV1], Field(max_length=SOURCE_LABEL_LIMIT)]

    @model_validator(mode="after")
    def ordered_positions(self) -> Self:
        """Reject repeated or reordered positions before source-specific coverage checks."""
        positions = [row.label_id for row in self.decisions]
        if positions != sorted(set(positions)):
            raise ValueError("Source-label decisions must be unique and in position order")
        return self


class SourceLabelsTargetV1(StrictModel):
    """Locate the original admitted semantic input, independently of extract output."""

    source_run_id: Annotated[ULID, RefSpec("run", scope_rule="same_scope")]
    source_input_ref: Annotated[
        ArtifactId, RefSpec("artifact", scope_rule="same_scope", artifact_kinds=("run_input",)),
    ]


class SourceLabelMessageOriginV1(StrictModel):
    """Locate the first classifier without granting its source's present eligibility."""

    child_run_id: Annotated[ULID, RefSpec("run", scope_rule="same_scope")]
    input_ref: Annotated[
        ArtifactId, RefSpec("artifact", scope_rule="same_scope", artifact_kinds=("run_input",)),
    ]


class SourceLabelMessageContinuationV1(StrictModel):
    """Freeze one reservation handoff from a terminal owner child that left no decision.

    ``reservation_version`` is the reservation's version when the handoff read
    it; the handoff writes the next one. The new owner child classifies with
    its own budget (ingest-Q-S2a-1).
    """

    previous_owner: SourceLabelMessageOriginV1
    first_origin: SourceLabelMessageOriginV1
    reservation_version: Annotated[int, Field(ge=0)]


class SourceLabelSpanV1(StrictModel):
    """One sent label's message byte range, which identifies its durable decision."""

    model_config = ConfigDict(frozen=True)
    byte_start: Annotated[int, Field(ge=0, lt=MAX_MESSAGE_BYTES)]
    byte_end: Annotated[int, Field(gt=0, le=MAX_MESSAGE_BYTES)]

    @model_validator(mode="after")
    def nonempty(self) -> Self:
        """Require a nonempty half-open range."""
        if self.byte_start >= self.byte_end:
            raise ValueError("Source-label span must be a nonempty byte range")
        return self


class SourceLabelMessageBindingV1(StrictModel):
    """Bind the labels one call classifies in one message, without retaining its text.

    The content digest covers the receipt, the message's parts, its rendered
    text and metadata and every located label. ``labels`` are the byte ranges
    of the labels this call sends, in message order; their positions in the
    call are ``label_start`` onwards, so a decision maps back to its exact range.
    """

    receipt_ref: Annotated[
        ArtifactId, RefSpec("artifact", scope_rule="same_scope", artifact_kinds=("run_input",)),
    ]
    ordinal: Annotated[int, Field(ge=0)]
    content_digest: Annotated[ContentBindingDigest, OpaqueIdSpec("digest")]
    label_start: Annotated[int, Field(ge=0, lt=SOURCE_LABEL_LIMIT)]
    labels: Annotated[list[SourceLabelSpanV1], Field(min_length=1, max_length=SOURCE_LABEL_LIMIT)]
    continuation: SourceLabelMessageContinuationV1 | None = None

    @model_validator(mode="after")
    def exact_positions(self) -> Self:
        """Require ordered, disjoint ranges whose positions stay inside the call's bound."""
        if self.label_start + len(self.labels) > SOURCE_LABEL_LIMIT or any(
            before.byte_end > after.byte_start
            for before, after in zip(self.labels, self.labels[1:], strict=False)
        ):
            raise ValueError("Source-label message has invalid label positions")
        return self

    @property
    def label_ids(self) -> range:
        """The call positions of this message's labels, in message order."""
        return range(self.label_start, self.label_start + len(self.labels))


class SourceLabelMessageReservationV1(StrictModel):
    """Reserve one received message for its classifying owner child (CAS owner).

    The fixed key excludes owners and call positions. The first binding never
    changes; a handoff from a terminal owner child that left no decision changes
    only ``owner`` and the Control version (ingest-Q-S2a-1).
    """

    schema_version: Literal[1]
    rule_version: Literal["source-labels-v2"]
    message: SourceLabelMessageBindingV1
    first_origin: SourceLabelMessageOriginV1
    owner: SourceLabelMessageOriginV1

    @model_validator(mode="after")
    def first_owner(self) -> Self:
        """The reservation keeps its first classifier's own binding."""
        if self.message.continuation is not None:
            raise ValueError("A source-label reservation keeps its first binding")
        return self


class SourceLabelsSourceLossV1(StrictModel):
    """Name a real original-message TTL or completed-cleanup boundary without text."""

    reason: Literal["ttl", "cleanup"]
    receipt_ref: Annotated[
        ArtifactId, RefSpec("artifact", scope_rule="same_scope", artifact_kinds=("run_input",)),
    ]
    ordinal: Annotated[int, Field(ge=0)]
    evidence_id: Annotated[ULID, RefSpec("evidence", scope_rule="same_scope")]
    expires_at: SystemTimestamp | None = None
    cleanup_run_id: Annotated[ULID, RefSpec("run", scope_rule="same_scope")] | None = None

    @model_validator(mode="after")
    def exact_reason(self) -> Self:
        if self.reason == "ttl" and (self.expires_at is None or self.cleanup_run_id is not None):
            raise ValueError("A TTL interruption requires only its original expiration")
        if self.reason == "cleanup" and (
            self.cleanup_run_id is None or self.expires_at is not None
        ):
            raise ValueError("A cleanup interruption requires only its completed owner")
        return self


class SourceLabelsSourceInterruptionV1(StrictModel):
    """Retire an unusable original batch without inventing a provider verdict."""

    kind: Literal["source_unavailable"]
    observed_at: SystemTimestamp
    losses: Annotated[list[SourceLabelsSourceLossV1], Field(min_length=1, max_length=6000)]

    @model_validator(mode="after")
    def unique_losses(self) -> Self:
        identities = [value.evidence_id for value in self.losses]
        if identities != sorted(set(identities)) or any(
            value.expires_at is not None and value.expires_at > self.observed_at
            for value in self.losses
        ):
            raise ValueError("Source interruption requires canonical actually observed losses")
        return self


SourceLabelsInterruptionV1 = SourceLabelsSourceInterruptionV1
"""The one terminal interruption of a label call: an observed source loss (closure stage)."""


class SourceLabelsBudgetV1(StrictModel):
    """Persist one owner child's deadline and reserved sends across crashes and resumes.

    The Writer creates this once, bounded by the owner's remaining budget, and
    atomically reserves each send. A reserved send remains spent after a crash.
    This counter is independent of the owner's attempt number: an adopted or
    resumed owner continues its child, never renews it (ingest-Q-S2a-1). The
    one exception is a key the provider refused (401/403/429): the call never
    reached a model, so each explicit resume of that pause grants the child one
    more call's sends in a fresh window from the resume (``outage_resumes``,
    ``resumed_at``; LABELS-RESUME). A whole call, not OR-1's one send: the
    transport's retry of a 429 must meet the provider again, never the cap.
    """

    schema_version: Literal[1]
    child_run_id: Annotated[ULID, RefSpec("run", scope_rule="same_scope")]
    parent_run_id: Annotated[ULID, RefSpec("run", scope_rule="same_scope")]
    started_at: SystemTimestamp
    deadline_at: SystemTimestamp
    attempts_reserved: Annotated[int, Field(ge=0)]
    outage_resumes: int = Field(default=0, ge=0, exclude_if=lambda value: value == 0)
    resumed_at: SystemTimestamp | None = Field(
        default=None, exclude_if=lambda value: value is None,
    )

    @model_validator(mode="after")
    def bounded_deadline(self) -> Self:
        """Reject a restored state that could grant more than one logical budget per window."""
        opened = self.started_at if self.resumed_at is None else self.resumed_at
        if (self.resumed_at is None) != (self.outage_resumes == 0) or (
            opened < self.started_at
        ) or not 0 < self.deadline_at - opened <= 300000:
            raise ValueError("Source-label deadline exceeds its single logical call budget")
        if self.attempts_reserved > EXTRACTION_REQUEST_CAP * (1 + self.outage_resumes):
            raise ValueError("Source-label sends exceed their call and its resume grants")
        return self


def checked_source_labels(
    binding: InferenceBindingV1, output: StoredInferenceOutput, *,
    input_digest: ContentBindingDigest, context_digest: ContentBindingDigest,
) -> SourceLabelsOutputV1:
    """Read an immutable result only under the exact frozen classification request.

    Args:
        binding: The admitted child's original inference policy.
        output: Its actual persisted output after envelope and creator verification.
        input_digest: The independently reconstructed complete source-label request.
        context_digest: The original complete-message provenance binding.

    Returns:
        The strict label decisions. Complete label coverage is checked against
        the independently reconstructed view by the consumer.

    Raises:
        AMLError: The schema, request, model or frozen policy differs. This check
            grants no source authorization or child ownership by itself.
    """
    fields = (
        "purpose", "provider", "requested_model", "schema_id", "prompt_version",
        "timeout_ms", "retry_policy_digest",
    )
    if (binding.purpose, binding.schema_id, binding.prompt_version) != (
        "extract", SOURCE_LABELS_VERSION, SOURCE_LABELS_VERSION,
    ) or any(getattr(output, name) != getattr(binding, name) for name in fields) or (
        output.inference_binding_digest != integrity_digest(binding)
        or output.input_digest != input_digest or output.context_digest != context_digest
        or output.summary_context_digest is not None
        or output.output_schema != SOURCE_LABELS_VERSION
        or not admits_inference_model(
            binding.provider, binding.requested_model, output.resolved_model,
        )
        or not isinstance(output.output, SourceLabelsOutputV1)
    ):
        raise AMLError("binding_mismatch", "Source-label output changed its frozen request")
    return SourceLabelsOutputV1.model_validate(output.output.model_dump(mode="json"))
