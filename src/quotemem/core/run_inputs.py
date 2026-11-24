"""Strict admitted inputs for the Run variants."""

from __future__ import annotations

from typing import Annotated, Any, Literal, Self, cast

from pydantic import (
    AfterValidator,
    Field,
    SerializerFunctionWrapHandler,
    ValidationError,
    model_serializer,
    model_validator,
)

from quotemem.core.controls import ControlRowV1, PolicyTier, policy_pointer_key
from quotemem.core.entities import EntityBindingV1, PositionKey, SupersedeRelation
from quotemem.core.errors import QuoteMemError
from quotemem.core.identity import ULID, ArtifactId, IntegrityDigest, ScopeId, integrity_digest
from quotemem.core.inference import InferenceBindingV1
from quotemem.core.ingestion_limits import MAX_ADD_SOURCE_PARTS
from quotemem.core.judge_records import JudgmentRunInputV1
from quotemem.core.privacy_persistence import (
    ConsentWithdrawalInputV1,
    IngestionSplitSecretInputV1,
    PreAddSplitSecretInputV1,
    SourceExpiryInputV1,
)
from quotemem.core.provider_budget import EXTRACTION_REQUEST_CAP, source_labels_call_budget
from quotemem.core.review_rounds import RoundNo
from quotemem.core.source_labels import (
    SOURCE_LABELS_VERSION,
    SourceLabelMessageBindingV1,
    SourceLabelsTargetV1,
)
from quotemem.core.source_participants import SourceMetadataVersion, SourceParticipantV1
from quotemem.core.statement import canonical_entity_type
from quotemem.core.technical_types import (
    ConfirmedRequestV1,
    ContentDigest,
    ExternalId,
    GoverningExtractionV1,
    HistoricalTimestamp,
    Integrity,
    LocalId,
    NonNegativeInt,
    PositiveInt,
    Timestamp,
)
from quotemem.store.metadata import OpaqueIdSpec, RefSpec, ScopeMembershipSpec, StrictModel


class AbsentControlPreconditionV1(StrictModel):
    """Require create-if-absent for the targeted Control key."""

    exists: Literal[False]


class ExistingControlPreconditionV1(StrictModel):
    """Require a specific existing materialized Control row."""

    exists: Literal[True]
    version: NonNegativeInt
    digest: Integrity


class ControlCasIntentV1(StrictModel):
    """Persist the entire CAS precondition and full post-image before execution."""

    control_key: LocalId
    expected_pre: Annotated[
        AbsentControlPreconditionV1 | ExistingControlPreconditionV1, Field(discriminator="exists")
    ]
    intended_post_image: ControlRowV1
    intended_post_digest: Integrity

    @model_validator(mode="after")
    def validate_post_image(self) -> Self:
        """Prove the intended bytes and revision match the frozen CAS target."""
        if self.control_key != self.intended_post_image.key:
            raise ValueError("CAS intent key does not match its full post-image")
        if self.intended_post_digest != integrity_digest(self.intended_post_image):
            raise ValueError("CAS intent digest does not match its full post-image")
        expected_version = (
            self.expected_pre.version + 1
            if isinstance(self.expected_pre, ExistingControlPreconditionV1)
            else 0
        )
        if self.intended_post_image.version != expected_version:
            raise ValueError("CAS post-image version must advance exactly once")
        return self


class RunInputBaseV1(StrictModel):
    """Carry the immutable admission and payload bindings of the creating Run."""

    admission_digest: ContentDigest
    payload_digest: ContentDigest
    target_locator: ContentDigest | None = None


class SessionBusinessKeyV1(StrictModel):
    """Identify a stable external request family independently of its content."""

    session_id: ExternalId
    request_id: ExternalId


class SourceRequestLocatorV1(SessionBusinessKeyV1):
    """Retain external source locators without creating QuoteMem references."""


class InputMessageV1(StrictModel):
    """Preserve canonical request message bytes, optional historical time, and speaker.

    ``speaker`` is the caller's own name for whoever wrote this message, which
    the source contract makes the sole origin of ``Evidence.speaker_entity``. It is
    never derived from the content, so a caller that names nobody is admitted
    exactly as before.

    ``entity_bindings`` are the Entity SDK's anchors: each binds
    a half-open range of this original message's UTF-8 bytes to one existing
    entity. Only the SDK supplies them; reception checks that each entity is
    active in the scope and splits them across the message's parts.
    """

    role: Literal["system", "user", "assistant", "tool"]
    timestamp: HistoricalTimestamp | None = None
    content: Annotated[str, Field(min_length=1)]
    speaker: Annotated[str, Field(min_length=1, max_length=1024)] | None = None
    source_participant: SourceParticipantV1 | None = None
    entity_bindings: Annotated[
        list[EntityBindingV1], Field(min_length=1, max_length=128),
    ] | None = None

    @model_validator(mode="after")
    def validate_message_bindings(self) -> Self:
        """Require a named speaker for participant metadata and exact entity anchors.

        Each entity binding is unique and lies inside this message, on UTF-8
        code-point boundaries at both ends.
        """
        if self.source_participant is not None and self.speaker is None:
            raise ValueError("Source participant metadata requires the original named speaker")
        raw = self.content.encode("utf-8")
        seen: set[EntityBindingV1] = set()
        for binding in self.entity_bindings or ():
            if binding in seen or binding.byte_end > len(raw):
                raise ValueError("Entity binding is duplicate or outside its source message")
            try:
                raw[:binding.byte_start].decode("utf-8")
                raw[binding.byte_start:binding.byte_end].decode("utf-8")
            except UnicodeDecodeError:
                raise ValueError("Entity binding splits a UTF-8 character") from None
            seen.add(binding)
        return self

    @model_serializer(mode="wrap")
    def serialize_named_speaker(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        """Keep a message without optional metadata byte-identical to its plain form."""
        values: dict[str, Any] = handler(self)
        for name in ("speaker", "source_participant", "entity_bindings"):
            if values.get(name) is None:
                values.pop(name, None)
        return values


class MessagesRequestPayloadV1(SessionBusinessKeyV1):
    """Bound a canonical Add request before any persistent side effect."""

    profile: Literal["product", "bench"]
    messages: Annotated[list[InputMessageV1], Field(min_length=1, max_length=256)]
    source_metadata_version: SourceMetadataVersion | None = Field(
        default=None, exclude_if=lambda value: value is None,
    )

    @model_validator(mode="after")
    def validate_bytes(self) -> Self:
        """Enforce the one-megabyte total source-text limit."""
        if sum(len(message.content.encode("utf-8")) for message in self.messages) > 1_048_576:
            raise ValueError("Add message content exceeds the UTF-8 byte budget")
        if self.source_metadata_version is None and any(
            message.source_participant is not None for message in self.messages
        ):
            raise ValueError("Source participant metadata requires its explicit payload version")
        return self


class TextRequestPayloadV1(SessionBusinessKeyV1):
    """Preserve an adapter-generated remember request without model options."""

    profile: Literal["product", "bench"]
    text: Annotated[str, Field(min_length=1)]

    @model_validator(mode="after")
    def validate_bytes(self) -> Self:
        """Apply the same request byte budget as the messages form."""
        if len(self.text.encode("utf-8")) > 1_048_576:
            raise ValueError("Remember text exceeds the UTF-8 byte budget")
        return self


class SessionAdmissionRunInputV1(RunInputBaseV1):
    """Retain accepted source bytes until the Evidence index is durable."""

    run_kind: Literal["session_ingestion"]
    input_phase: Literal["admission"]
    business_key: SessionBusinessKeyV1
    request_payload: MessagesRequestPayloadV1 | TextRequestPayloadV1
    received_at: Timestamp | None = Field(default=None, exclude_if=lambda value: value is None)
    reception_sequence: PositiveInt | None = Field(
        default=None, exclude_if=lambda value: value is None,
    )
    """Original admission-directory order, preserving same-millisecond ties at restart."""
    work_deadline_at: Timestamp | None = Field(default=None, exclude_if=lambda value: value is None)

    @model_validator(mode="after")
    def validate_request_identity(self) -> Self:
        """Reject a payload from a different admitted request family."""
        if (self.received_at is None) != (self.work_deadline_at is None) or (
            self.received_at is not None and self.work_deadline_at is not None
            and self.work_deadline_at <= self.received_at
        ):
            raise ValueError("Reception must retain one positive original work allowance")
        if (self.business_key.session_id, self.business_key.request_id) != (
            self.request_payload.session_id,
            self.request_payload.request_id,
        ):
            raise ValueError("Admission payload does not match its business key")
        return self


class SessionDurableRunInputV1(RunInputBaseV1):
    """Freeze a durable reception witness that contains no source message text."""

    run_kind: Literal["session_ingestion"]
    input_phase: Literal["durable"]
    business_key: SessionBusinessKeyV1
    visibility_commit_seq: PositiveInt
    source_request_locator: SourceRequestLocatorV1
    request_key: ContentDigest
    message_count: PositiveInt
    evidence_set_digest: ContentDigest
    confirmed_request: ConfirmedRequestV1 | None = None
    received_at: Timestamp | None = Field(default=None, exclude_if=lambda value: value is None)
    reception_sequence: PositiveInt | None = Field(
        default=None, exclude_if=lambda value: value is None,
    )
    work_deadline_at: Timestamp | None = Field(default=None, exclude_if=lambda value: value is None)

    @model_validator(mode="after")
    def validate_request_identity(self) -> Self:
        """Bind the historical witness to the same external request locators."""
        if self.business_key.model_dump() != self.source_request_locator.model_dump():
            raise ValueError("Durable witness does not match its business key")
        return self


class SourceInventoryV1(StrictModel):
    """Bind the complete immutable part set without retaining source text or IDs."""

    part_count: Annotated[int, Field(ge=1, le=MAX_ADD_SOURCE_PARTS)]
    digest: ContentDigest


class SessionInventoryDurableRunInputV2(SessionDurableRunInputV1):
    """Retain a privacy-stable witness of the accepted request's complete part set."""

    receipt_version: Literal[2]
    source_inventory: SourceInventoryV1

    @model_validator(mode="after")
    def validate_inventory_count(self) -> Self:
        """Require every accepted nonempty message to contribute a physical part."""
        if not self.message_count <= self.source_inventory.part_count or self.message_count > 256:
            raise ValueError("Durable inventory part count differs from the message boundary")
        return self


class EvidenceMemberBindingV1(StrictModel):
    """Bind a source row and an optional physical byte window."""

    evidence_id: Annotated[
        ULID, RefSpec("evidence", ref_mode="snapshot_locator", target_forget_action="delete_owner")
    ]
    row_binding_digest: ContentDigest
    byte_start: NonNegativeInt | None = None
    byte_end: PositiveInt | None = None

    @model_validator(mode="after")
    def validate_window(self) -> Self:
        """Require paired, increasing byte offsets when a window is used."""
        if (self.byte_start is None) != (self.byte_end is None):
            raise ValueError("Member byte windows require both offsets")
        if self.byte_start is not None and self.byte_end is not None:
            if self.byte_start >= self.byte_end:
                raise ValueError("Member byte window must be nonempty")
        return self


class FinalizeBusinessKeyV1(StrictModel):
    """Identify one finalize cycle for an exact registered scope."""

    scope_id: Annotated[ScopeId, OpaqueIdSpec("local"), ScopeMembershipSpec()]
    finalize_cycle: PositiveInt


class FinalizeOwnerRunInputV1(RunInputBaseV1, GoverningExtractionV1):
    """Freeze compact complete-scope input counts under the held input fence."""

    run_kind: Literal["finalize"]
    worker_role: Literal["owner"]
    business_key: FinalizeBusinessKeyV1
    freeze_time: Timestamp
    finalize_input_scope_seq: NonNegativeInt
    receipt_high_watermark: NonNegativeInt
    evidence_count: NonNegativeInt
    evidence_rows_digest: ContentDigest
    input_heads_digest: ContentDigest
    # The input contains only counts/digests and the original policy binding.
    # Erasing inventory content must not erase a published survivor's policy.
    # Resuming a cycle still requires the actual inventory, independently.
    source_inventory_ref: Annotated[
        ArtifactId,
        RefSpec(
            "artifact", ref_mode="snapshot_locator", scope_rule="same_scope",
            gc_root="while_owner_nonterminal", target_forget_action="retain_locator",
            artifact_kinds=("decision_set",),
        ),
    ] | None
    source_inventory_count: NonNegativeInt
    source_inventory_digest: ContentDigest

    @model_validator(mode="after")
    def validate_source_inventory(self) -> Self:
        """Original metadata accounts for every frozen Evidence row, including empty scopes."""
        if self.source_inventory_count != self.evidence_count or (
            (self.source_inventory_ref is not None) != (self.evidence_count > 0)
        ):
            raise ValueError("Finalize source inventory must cover its exact frozen input")
        return self


class PolicyBootstrapKeyV1(StrictModel):
    """Identify installation of one exact built-in policy configuration."""

    policy_kind: Literal["recall", "extraction"]
    action: Literal["bootstrap"]
    builtin_config_digest: Integrity
    tier: PolicyTier | None = Field(default=None, exclude_if=lambda value: value is None)
    """The Add tier whose extraction policy this installs; None (left out of the body, so a
    default policy's admission is unchanged) is the default tier's."""

    @model_validator(mode="after")
    def validate_tier(self) -> Self:
        """Only the extraction policy has a fast tier."""
        if self.tier is not None and self.policy_kind != "extraction":
            raise ValueError("Only the extraction policy has a fast tier")
        return self

    @property
    def control_key(self) -> str:
        """The pointer this bootstrap installs (``POLICY_POINTERS``)."""
        return policy_pointer_key(self.policy_kind, self.tier)


class PolicyBootstrapRunInputV1(RunInputBaseV1):
    """Persist the first current-policy CAS under the authoritative Writer."""

    run_kind: Literal["policy_release"]
    business_key: PolicyBootstrapKeyV1
    control_cas_intent: ControlCasIntentV1

    @model_validator(mode="after")
    def validate_bootstrap(self) -> Self:
        """Limit bootstrap to an absent matching policy Control."""
        if self.control_cas_intent.expected_pre.exists:
            raise ValueError("Policy bootstrap requires an absent Control")
        if self.control_cas_intent.control_key != self.business_key.control_key:
            raise ValueError("Policy bootstrap intent targets a different policy")
        return self


class EmbeddingBootstrapKeyV1(StrictModel):
    """Identify the first embedding state by its frozen space and canary."""

    action: Literal["bootstrap"]
    builtin_space: Annotated[str, Field(min_length=1, max_length=1024)]
    builtin_canary_digest: Integrity


class EmbeddingBootstrapRunInputV1(RunInputBaseV1, EmbeddingBootstrapKeyV1):
    """Freeze baseline references before the first active-space CAS."""

    run_kind: Literal["embedding_switch"]
    business_key: EmbeddingBootstrapKeyV1
    canary_baseline_ref: Annotated[
        ArtifactId,
        RefSpec(
            "artifact",
            scope_rule="global",
            target_forget_action="retain_locator",
            artifact_kinds=("policy_config",),
        ),
    ]
    reserved_generation: PositiveInt
    control_cas_intent: ControlCasIntentV1

    @model_validator(mode="after")
    def validate_bootstrap(self) -> Self:
        """Prove the duplicated canonical key and intended active generation."""
        from quotemem.core.controls import ActiveEmbeddingSpaceV1

        for field in ("action", "builtin_space", "builtin_canary_digest"):
            if getattr(self, field) != getattr(self.business_key, field):
                raise ValueError("Embedding bootstrap body differs from its business key")
        intent = self.control_cas_intent
        value = intent.intended_post_image.value
        if intent.expected_pre.exists or intent.control_key != "policy:embedding_space":
            raise ValueError("Embedding bootstrap requires an absent embedding Control")
        if not isinstance(value, ActiveEmbeddingSpaceV1):
            raise ValueError("Embedding bootstrap requires the first active state")
        if (
            value.state_commit_seq != self.reserved_generation
            or value.embedding_space != self.builtin_space
            or value.canary_digest != self.builtin_canary_digest
            or value.canary_baseline_ref != self.canary_baseline_ref
        ):
            raise ValueError("Embedding bootstrap intent does not match its frozen baseline")
        return self


class _PrivacyEraseRunInputV1(RunInputBaseV1):
    """Attach only a complete content-free plan after its privacy-erase owner is admitted."""

    run_kind: Literal["retention"]
    plan_ref: Annotated[
        ArtifactId,
        RefSpec("artifact", scope_rule="same_scope", artifact_kinds=("erasure_plan",)),
    ]

    @model_validator(mode="after")
    def check_target(self) -> Self:
        """Every erase binds its affected set without copying an external source locator."""
        if self.target_locator is None:
            raise ValueError("Privacy erase input requires its admitted affected-set binding")
        return self


class SplitSecretPreAddRunInputV1(_PrivacyEraseRunInputV1):
    """Freeze the independent cleanup root before the rejected Add is admitted."""

    retention_role: Literal["split_secret"]
    trigger: Literal["pre_add"]
    business_key: PreAddSplitSecretInputV1


class SplitSecretChildRunInputV1(_PrivacyEraseRunInputV1):
    """Freeze the cleanup child's original ingestion-parent binding."""

    retention_role: Literal["split_secret"]
    trigger: Literal["ingestion_child"]
    business_key: IngestionSplitSecretInputV1


class SourceExpiryRunInputV1(_PrivacyEraseRunInputV1):
    """Freeze an independent source-expiry erase root (test fixtures only)."""

    retention_role: Literal["source_expiry"]
    trigger: Literal["clock"]
    business_key: SourceExpiryInputV1


class ConsentWithdrawalRunInputV1(_PrivacyEraseRunInputV1):
    """Freeze an independent consent-withdrawal erase root (test fixtures only)."""

    retention_role: Literal["consent_withdrawal"]
    trigger: Literal["consent_epoch"]
    business_key: ConsentWithdrawalInputV1



class SourceLabelsAdmissionSeedV1(StrictModel):
    """Freeze a complete-message classification under its original semantic owner."""

    labels_version: Literal[1]
    parent_run_id: Annotated[ULID, RefSpec("run", scope_rule="same_scope")]
    parent_attempt: NonNegativeInt
    target: SourceLabelsTargetV1
    message_bindings: Annotated[
        list[SourceLabelMessageBindingV1], Field(min_length=1, max_length=6000),
    ]
    inference_binding: InferenceBindingV1
    input_digest: ContentDigest
    context_digest: ContentDigest

    @model_validator(mode="after")
    def validate_source_binding(self) -> Self:
        """Require the exact label schema, source owner and bounded extract policy."""
        binding = self.inference_binding
        budget = binding.call_budget
        limits = source_labels_call_budget(budget.policy_id) if budget is not None else None
        if self.parent_run_id != self.target.source_run_id or (
            binding.purpose, binding.schema_id, binding.prompt_version
        ) != ("extract", SOURCE_LABELS_VERSION, SOURCE_LABELS_VERSION) or (
            budget is None or limits is None or budget.policy_id != limits.policy_id
            or budget.attempt_wall_ms > limits.attempt_wall_ms
            or budget.logical_wall_ms > limits.logical_wall_ms
            or binding.retry_policy.max_attempts > EXTRACTION_REQUEST_CAP
        ):
            raise ValueError("Source labels require their original owner and exact bounded policy")
        identities = [(message.receipt_ref, message.ordinal) for message in self.message_bindings]
        offset = 0
        for message in self.message_bindings:
            if message.label_start != offset:
                raise ValueError("Source-label message ranges must be contiguous from zero")
            offset += len(message.labels)
        if len(set(identities)) != len(identities):
            raise ValueError("Source labels require unique messages")
        return self


def source_labels_business_key(
    seed: SourceLabelsAdmissionSeedV1 | SourceLabelsTargetV1,
) -> Integrity:
    """Name one original source-input family across owner recovery attempts.

    Complete policy and source content remain in the separately authenticated
    admission payload. A retry cannot allocate a second logical-call budget by
    changing its parent attempt or losing the first child's input attachment.
    """
    target = seed.target if isinstance(seed, SourceLabelsAdmissionSeedV1) else seed
    return integrity_digest({
        "kind": "source_labels", "version": 1, "target": target.model_dump(mode="json"),
    })


class SourceLabelsRunInputV1(RunInputBaseV1, SourceLabelsAdmissionSeedV1):
    """Admit one artifact-only evaluate child before its parent's extract output."""

    run_kind: Literal["evaluate"]
    evaluation_kind: Literal["source_labels"]
    business_key: Integrity

    @model_validator(mode="after")
    def validate_source_labels(self) -> Self:
        """Reject a changed source family; admission authenticates the complete payload."""
        seed = SourceLabelsAdmissionSeedV1.model_validate({
            name: getattr(self, name) for name in SourceLabelsAdmissionSeedV1.model_fields
        })
        if self.business_key != source_labels_business_key(seed):
            raise ValueError("Source-label business key changed its original source target")
        return self


def _stored_type(value: str) -> str:
    """Require an entity type in its stored form."""
    if canonical_entity_type(value) != value:
        raise ValueError("An entity type is stored after NFC and trimming")
    return value


EntityTypeText = Annotated[str, Field(min_length=1, max_length=64), AfterValidator(_stored_type)]
"""A stored entity type: free text after NFC, trimming and the type rules."""


class EntityUpdateKeyV1(StrictModel):
    """Identify one Entity SDK call by its request id.

    The admission family adds the exact scope and the principal, so a request
    id is idempotent per (principal, scope, request_id).
    """

    request_id: ExternalId


class EntityCreateV1(StrictModel):
    """Create one SDK entity of a stored type; it has no names (``create``)."""

    op: Literal["create"]
    request_id: ExternalId
    entity_type: EntityTypeText


class EntitySetTypeV1(StrictModel):
    """Retype one SDK entity while it still has the observed type (``set_type``)."""

    op: Literal["set_type"]
    request_id: ExternalId
    entity_id: Annotated[ULID, OpaqueIdSpec("local")]
    expected_type: EntityTypeText
    entity_type: EntityTypeText


EntityUpdateRequestV1 = Annotated[EntityCreateV1 | EntitySetTypeV1, Field(discriminator="op")]
"""One Entity SDK operation, as admitted (its request payload)."""


class EntityUpdateSdkRunInputV1(RunInputBaseV1):
    """The frozen input of one Entity SDK ``entity_update`` run.

    ``entity_id`` is the one entity the run writes: the entity it creates,
    chosen once when the input is written, or the entity it retypes.
    """

    run_kind: Literal["entity_update"]
    source: Literal["sdk"]
    business_key: EntityUpdateKeyV1
    request: EntityUpdateRequestV1
    entity_id: Annotated[ULID, OpaqueIdSpec("local")]

    @model_validator(mode="after")
    def validate_request(self) -> Self:
        """Bind the request to its key and a retype to the entity it names."""
        if self.request.request_id != self.business_key.request_id:
            raise ValueError("An SDK request differs from its business key")
        if isinstance(self.request, EntitySetTypeV1) and self.request.entity_id != self.entity_id:
            raise ValueError("An SDK retype writes the entity it names")
        return self


class MemoryRetireV1(StrictModel):
    """End one active memory without a successor (``api.memory_retire``).

    A source-journal rebuild (``quotemem rebuild``) replays what the user said, so
    a memory whose successor the user later asked to forget comes back current: the
    words that ended it are withheld and never reach the extractor again. The
    exported store held that memory ended; this request ends it again, as the
    relation its old successor named, while its head is still the observed one.
    """

    op: Literal["retire"]
    request_id: ExternalId
    memory_id: Annotated[ULID, OpaqueIdSpec("local")]
    expected_revision_id: Annotated[ULID, OpaqueIdSpec("local")]
    relation: SupersedeRelation


class MemoryRetireRunInputV1(RunInputBaseV1):
    """The frozen input of one ``entity_update`` run that retires a memory (``retire``)."""

    run_kind: Literal["entity_update"]
    source: Literal["retire"]
    business_key: EntityUpdateKeyV1
    request: MemoryRetireV1

    @model_validator(mode="after")
    def validate_request(self) -> Self:
        """Bind the request to its key."""
        if self.request.request_id != self.business_key.request_id:
            raise ValueError("A retire request differs from its business key")
        return self


class PublishCommitSeedV1(StrictModel):
    """The component one ``publish`` child commits: its admission request.

    Attributes:
        owner_run_id: The Add owner, the child's parent.
        round: The round.
        positions: The component's members, sorted by position key.
        journal_key: The component's journal key (``quotemem.core.commit_plan``).
    """

    owner_run_id: Annotated[ULID, RefSpec("run", target_forget_action="retain_locator")]
    round: RoundNo
    positions: Annotated[list[PositionKey], Field(min_length=1)]
    journal_key: ContentDigest

    @model_validator(mode="after")
    def sorted_positions(self) -> Self:
        """Name each member once, sorted by position key."""
        if self.positions != sorted(set(self.positions)):
            raise ValueError("A publish component lists its members sorted and unique")
        return self


def publish_commit_business_key(seed: PublishCommitSeedV1) -> IntegrityDigest:
    """Name one component's ``publish`` family: the owner, round and members."""
    value = seed.model_dump(mode="json")
    return integrity_digest(["publish-commit-v1", value["owner_run_id"], value["round"],
                             value["positions"]])


class PublishCommitRunInputV1(RunInputBaseV1, PublishCommitSeedV1):
    """The input of one ``publish`` child committing one formal component (``publish:commit``).

    Recorded contribution of the ``entity_name`` audit: the child's journal key and
    members fix the one component transaction it may write; its
    ``CommitPlanV1`` (``publish_plan``) and ``CommitJournalV1`` follow.
    """

    run_kind: Literal["publish"]
    publish_kind: Literal["commit"]
    business_key: Integrity

    @model_validator(mode="after")
    def bound_component(self) -> Self:
        """Bind the business key to the component it names."""
        seed = PublishCommitSeedV1.model_validate({
            name: getattr(self, name) for name in PublishCommitSeedV1.model_fields
        })
        if self.business_key != publish_commit_business_key(seed):
            raise ValueError("A publish input's business key names another component")
        return self


RUN_INPUT_MODELS: dict[str, type[StrictModel]] = {
    "session_ingestion:admission": SessionAdmissionRunInputV1,
    "session_ingestion:durable": SessionDurableRunInputV1,
    "session_ingestion:durable:inventory_v2": SessionInventoryDurableRunInputV2,
    "finalize:owner": FinalizeOwnerRunInputV1,
    "policy_release:bootstrap": PolicyBootstrapRunInputV1,
    "embedding_switch:bootstrap": EmbeddingBootstrapRunInputV1,
    "retention:split_secret:pre_add": SplitSecretPreAddRunInputV1,
    "retention:split_secret:ingestion_child": SplitSecretChildRunInputV1,
    "retention:source_expiry:clock": SourceExpiryRunInputV1,
    "retention:consent_withdrawal:consent_epoch": ConsentWithdrawalRunInputV1,
    "evaluate:source_labels:single": SourceLabelsRunInputV1,
    "entity_update:sdk": EntityUpdateSdkRunInputV1,
    "entity_update:retire": MemoryRetireRunInputV1,
    "publish:commit": PublishCommitRunInputV1,
    "evaluate:judgment:single": JudgmentRunInputV1,
}


def parse_run_input(raw: object) -> StrictModel:
    """Dispatch the admitted variant and reject every undeclared or future form.

    Args:
        raw: Decoded logical run_input body.

    Returns:
        A strict model in the implemented schema set.

    Raises:
        QuoteMemError: If the discriminant is unsupported or the body is invalid.
    """
    if not isinstance(raw, dict):
        raise QuoteMemError("schema_invalid", "Run input must be an object")
    raw_dict = cast(dict[str, object], raw)
    kind = raw_dict.get("run_kind")
    key = raw_dict.get("business_key")
    if kind == "session_ingestion":
        variant = f"{kind}:{raw_dict.get('input_phase')}"
        if raw_dict.get("input_phase") == "durable" and "receipt_version" in raw_dict:
            if type(raw_dict["receipt_version"]) is not int:
                raise QuoteMemError("schema_invalid", "Receipt version must be an exact integer")
            if raw_dict["receipt_version"] != 2:
                raise QuoteMemError("unsupported_variant",
                                    "Durable receipt version is not implemented")
            variant += ":inventory_v2"
    elif kind == "finalize":
        variant = f"{kind}:{raw_dict.get('worker_role')}"
    elif kind in {"policy_release", "embedding_switch"}:
        action = cast(dict[str, object], key).get("action") if isinstance(key, dict) else None
        variant = f"{kind}:{action}"
    elif kind == "retention":
        variant = f"{kind}:{raw_dict.get('retention_role')}:{raw_dict.get('trigger')}"
    elif kind == "evaluate":
        if raw_dict.get("evaluation_kind") == "source_labels" and (
            type(raw_dict.get("labels_version")) is not int or raw_dict["labels_version"] != 1
        ):
            raise QuoteMemError("unsupported_variant", "Source-label version is not implemented")
        variant = f"{kind}:{raw_dict.get('evaluation_kind')}:single"
    elif kind == "entity_update":
        variant = f"{kind}:{raw_dict.get('source')}"
    elif kind == "publish":
        variant = f"{kind}:{raw_dict.get('publish_kind')}"
    else:
        variant = str(kind)
    model = RUN_INPUT_MODELS.get(variant)
    if model is None:
        raise QuoteMemError("unsupported_variant",
                            f"Run input variant is not implemented: {variant}")
    try:
        return model.model_validate(raw_dict)
    except ValidationError as error:
        raise QuoteMemError("schema_invalid", "Run input violates its strict schema") from error

