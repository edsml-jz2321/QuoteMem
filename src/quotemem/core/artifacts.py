"""Strict logical Artifact bodies and physical envelopes."""

from __future__ import annotations

import base64
import binascii
import json
import math
import struct
from typing import Annotated, Literal, Self, cast, get_args

from pydantic import Field, ValidationError, ValidationInfo, model_validator

from quotemem.core.add_outcome import AddCompletionRecordV1
from quotemem.core.commit_plan import CommitJournalV1, CommitPlanV1
from quotemem.core.errors import QuoteMemError
from quotemem.core.extraction_bounds import ExtractionRefusalDetailV1
from quotemem.core.extraction_context import ExtractContextV2
from quotemem.core.extraction_policy import ExtractionPolicyConfigV2
from quotemem.core.extraction_records import DigestSettlementV1, ErasedExtractionV1
from quotemem.core.forget import ForgetPrepareReceiptV3, ForgetReceiptV3
from quotemem.core.forget_receipts import ErasureReceiptResultIndexV1
from quotemem.core.identity import (
    ULID,
    ArtifactId,
    PrincipalId,
    ScopeId,
    artifact_id,
    canonical_json,
)
from quotemem.core.inference import StoredInferenceOutput
from quotemem.core.judge_records import JudgeCallManifestV1, JudgeResultIndexV8
from quotemem.core.neighbor_snapshot import NeighborSnapshotV1
from quotemem.core.privacy_embeddings import PrivacyEmbeddingV1
from quotemem.core.privacy_persistence import (
    REDACTION_REASONS,
    IngestionErasurePlanV1,
    RedactionReceiptV1,
)
from quotemem.core.proposal import ProposalV2
from quotemem.core.provider_diagnostics import ProviderFailureDetailV1
from quotemem.core.query import ScopeGrantV1, ScopeV1
from quotemem.core.recall_policy import RecallPolicyConfigV4
from quotemem.core.references import ReferenceProjectionV1
from quotemem.core.run_inputs import RUN_INPUT_MODELS, parse_run_input
from quotemem.core.semantic_embedding import StatementEmbeddingV1
from quotemem.core.source_labels import SOURCE_LABEL_LIMIT, SourceLabelsInterruptionV1
from quotemem.core.technical_types import (
    ContentDigest,
    Integrity,
    LocalId,
    NonNegativeInt,
    PositiveInt,
    RequestProgressV1,
    Timestamp,
)
from quotemem.core.work import FinalizeSourceInventoryBlockV1, FinalizeSourceTTLBlockV1
from quotemem.store.metadata import DerivedSpec, OpaqueIdSpec, RefSpec, StrictModel

ArtifactKind = Literal[
    "llm_output",
    "proposal_snapshot",
    "semantic_embedding",
    "publish_plan",
    "write_set",
    "result_index",
    "finalize_plan",
    "request_ledger",
    "decision_set",
    "erasure_plan",
    "part",
    "manifest",
    "auth_context",
    "receipt",
    "policy_config",
    "policy_release",
    "run_input",
]
# Kinds whose logical body may exceed the 262,144-character row bound and is then
# stored as one `manifest` and its `part`s. `decision_set` carries the
# extraction context (withheld ranges for up to 600 sources) and the settlement
# set; `llm_output` carries a verbatim reply with its admission record.
SHARDABLE_KINDS = frozenset(
    {
        "decision_set",
        "llm_output",
        "write_set",
        "result_index",
        "finalize_plan",
        "policy_config",
        "run_input",
        "erasure_plan",
        "semantic_embedding",
    }
)


class SourceLabelCoverageDiagnosticV1(StrictModel):
    """Only code-owned positions from a complete-response coverage mismatch."""

    expected_count: Annotated[int, Field(ge=0, le=SOURCE_LABEL_LIMIT)]
    received_count: Annotated[int, Field(ge=0, le=SOURCE_LABEL_LIMIT)]
    first_mismatch_position: Annotated[int, Field(ge=0, lt=SOURCE_LABEL_LIMIT)]
    expected_label_id: Annotated[
        int, Field(ge=0, lt=SOURCE_LABEL_LIMIT), OpaqueIdSpec("local"),
    ] | None
    received_label_id: Annotated[
        int, Field(ge=0, lt=SOURCE_LABEL_LIMIT), OpaqueIdSpec("local"),
    ] | None

    @model_validator(mode="after")
    def actual_mismatch(self) -> Self:
        """Reject a fabricated coordinate that describes matching decisions."""
        position = self.first_mismatch_position
        if position >= max(self.expected_count, self.received_count) or (
            (self.expected_label_id is None) != (position >= self.expected_count)
        ) or (
            (self.received_label_id is None) != (position >= self.received_count)
        ) or self.expected_label_id == self.received_label_id:
            raise ValueError("Source-label coverage diagnostic has no actual mismatch")
        return self


class SourceLabelsFailureDiagnosticV1(StrictModel):
    """A sanitized original-call refusal, distinct from inherited failed claims."""

    kind: Literal["adapter_schema", "decision_coverage"]
    provider_diagnostics: ProviderFailureDetailV1 | None = Field(
        default=None, exclude_if=lambda value: value is None,
    )
    coverage: SourceLabelCoverageDiagnosticV1 | None = Field(
        default=None, exclude_if=lambda value: value is None,
    )

    @model_validator(mode="after")
    def exact_kind(self) -> Self:
        if self.kind == "adapter_schema":
            if self.coverage is not None or (
                self.provider_diagnostics is not None
                and self.provider_diagnostics.error_class != "schema_invalid"
            ):
                raise ValueError("Adapter refusal has unrelated source-label details")
        elif self.provider_diagnostics is not None or self.coverage is None:
            raise ValueError("Coverage refusal cannot claim adapter diagnostics")
        return self


class SourceLabelsResultIndexV1(StrictModel):
    """Close one label request; fully reused inputs have no new provider output."""

    source_labels: Literal[1]
    input_ref: Annotated[
        ArtifactId, RefSpec("artifact", scope_rule="same_scope", artifact_kinds=("run_input",)),
    ]
    llm_output_ref: Annotated[
        ArtifactId, RefSpec("artifact", scope_rule="same_scope", artifact_kinds=("llm_output",)),
    ] | None
    interruption: SourceLabelsInterruptionV1 | None = None
    failure_diagnostics: SourceLabelsFailureDiagnosticV1 | None = Field(
        default=None, exclude_if=lambda value: value is None,
    )

    @model_validator(mode="after")
    def no_interrupted_verdict(self) -> Self:
        """An interrupted request has no provider output or copied label verdict."""
        if sum(value is not None for value in (
            self.interruption, self.failure_diagnostics, self.llm_output_ref,
        )) > 1:
            raise ValueError("Source-label result cannot mix verdict, interruption and failure")
        return self


class SessionExtractionRefusalResultIndexV1(StrictModel):
    """Diagnostic-only original request failure; never recovery or factual authority."""

    input_ref: Annotated[ArtifactId, RefSpec("artifact", artifact_kinds=("run_input",))]
    inference_binding_digest: Integrity
    extraction_refusal: ExtractionRefusalDetailV1
    provider_diagnostics: ProviderFailureDetailV1 | None = Field(
        default=None, exclude_if=lambda value: value is None,
    )
    provider_call_kind: Literal["extract"] | None = Field(
        default=None, exclude_if=lambda value: value is None,
    )

    @model_validator(mode="after")
    def validate_provider_refusal(self) -> Self:
        """Request-output and provider-envelope refusals remain separate boundaries."""
        if self.provider_call_kind is not None and self.provider_diagnostics is None:
            raise ValueError("Provider call kind requires its actual diagnostics")
        if self.provider_diagnostics is not None and (
            self.provider_diagnostics.error_class != "schema_invalid"
            or self.extraction_refusal.rule != "provider_schema_invalid"
        ):
            raise ValueError("Provider diagnostics require a provider schema refusal")
        return self


class SessionProviderFailureResultIndexV1(StrictModel):
    """An actual session provider failure with no successful output or recovery grant.

    ``provider_failure`` holds the failure's bounded HTTP diagnostics, or null
    for a failure without them: a judge member with no accepted verdict after
    its solo resend, or a refusal before any request. The failure is
    recorded either way, so recovery can select the pause.

    ``paused_at`` is the pause's trusted time, the one the pause stamps on the owner
    Run. An index's address is its content's digest, so neither the listing order
    nor an equal earlier body tells which pause is current: each pause writes its
    own index, and the latest ``paused_at`` names the current one (``paused_index``).
    """

    input_ref: Annotated[ArtifactId, RefSpec("artifact", artifact_kinds=("run_input",))]
    paused_at: NonNegativeInt
    provider_failure: ProviderFailureDetailV1 | None
    provider_call_kind: Literal["extract", "embedding", "source_labels", "judge"] | None = Field(
        default=None, exclude_if=lambda value: value is None,
    )


class ReceiptResultIndexV1(StrictModel):
    """Select the verified scoped receipt of one ingestion cleanup."""

    receipt_ref: Annotated[
        ArtifactId,
        RefSpec("artifact", scope_rule="same_scope", artifact_kinds=("receipt",)),
    ]


class RequestLedgerEntryV1(RequestProgressV1):
    """Keep one permanent admission conflict binding and optional confirmation."""

    request_key: ContentDigest


class RequestLedgerV1(StrictModel):
    """Freeze one bounded accepted-request chain block in canonical key order."""

    previous_ref: (
        Annotated[ArtifactId, RefSpec("artifact", artifact_kinds=("request_ledger",))] | None
    ) = None
    entries: Annotated[list[RequestLedgerEntryV1], Field(min_length=1, max_length=128)]
    entry_count: PositiveInt
    total_count: PositiveInt
    cumulative_digest: ContentDigest

    @model_validator(mode="after")
    def validate_entries(self) -> Self:
        """Require the exact local count and unique sorted request keys."""
        keys = [entry.request_key for entry in self.entries]
        if keys != sorted(set(keys)):
            raise ValueError("Request ledger keys must be sorted and unique")
        if self.entry_count != len(self.entries) or self.total_count < self.entry_count:
            raise ValueError("Request ledger counts do not match its entries")
        if self.previous_ref is None and self.total_count != self.entry_count:
            raise ValueError("First request ledger block has an inconsistent total")
        return self


class AuthContextV1(StrictModel):
    """Freeze the provider-resolved principal and exact authorization grants."""

    principal_id: Annotated[PrincipalId, OpaqueIdSpec("local")]
    client_session_digest: ContentDigest
    requested_scope: ScopeV1
    scope_grants: Annotated[list[ScopeGrantV1], Field(min_length=1)]
    issued_at: Timestamp
    expires_at: Timestamp

    @model_validator(mode="after")
    def validate_grants(self) -> Self:
        """Require one exact requested grant and a valid authorization lifetime."""
        keys = [(grant.scope_kind, grant.scope_id, grant.relation) for grant in self.scope_grants]
        expected = (self.requested_scope.scope_kind, self.requested_scope.scope_id, "exact")
        if keys != sorted(set(keys)) or expected not in keys:
            raise ValueError("Auth context grants must be canonical and include its exact scope")
        if self.issued_at >= self.expires_at:
            raise ValueError("Auth context lifetime must be positive")
        return self


class EmbeddingCanaryParametersV1(StrictModel):
    """Freeze the 32 finite float32LE baseline vectors independently of providers."""

    text_version: Annotated[str, Field(min_length=1, max_length=128)]
    model: Annotated[str, Field(min_length=1, max_length=256)]
    dim: Literal[1024, 2048]
    vectors: Annotated[list[str], Field(min_length=32, max_length=32)]

    @model_validator(mode="after")
    def validate_vectors(self) -> Self:
        """Reject malformed encoding, nonfinite components, and zero vectors."""
        for encoded in self.vectors:
            try:
                raw = base64.b64decode(encoded, validate=True)
            except (ValueError, binascii.Error) as error:
                raise ValueError("Canary vectors require canonical base64") from error
            if len(raw) != self.dim * 4 or base64.b64encode(raw).decode("ascii") != encoded:
                raise ValueError("Canary vector encoding or dimension is invalid")
            values = struct.unpack(f"<{self.dim}f", raw)
            if not all(math.isfinite(value) for value in values) or not any(values):
                raise ValueError("Canary vectors must be finite and nonzero")
        return self


class EmbeddingCanaryConfigV1(StrictModel):
    """Use the same strict policy-config envelope as recall and extraction policies."""

    policy_kind: Literal["embedding_canary"]
    config_version: Literal[1]
    parameters: EmbeddingCanaryParametersV1


class BootstrapPolicyReleaseV1(StrictModel):
    """Publish only the installed-default policy bootstrap in the schema set."""

    action: Literal["bootstrap"]
    policy_kind: Literal["recall", "extraction"]
    release_id: LocalId
    config_ref: Annotated[
        ArtifactId, RefSpec("artifact", scope_rule="global", artifact_kinds=("policy_config",))
    ]
    reason_code: Literal["installed_default"]


class PartV1(StrictModel):
    """Store one physical byte fragment; logical validation precedes part creation."""

    encoding: Literal["base64"]
    data: Annotated[str, Field(min_length=1, max_length=250000)]

    @model_validator(mode="after")
    def validate_base64(self) -> Self:
        """Require canonical standard base64 without ignored characters."""
        try:
            raw = base64.b64decode(self.data, validate=True)
        except (ValueError, binascii.Error) as error:
            raise ValueError("Artifact part requires valid base64") from error
        if base64.b64encode(raw).decode("ascii") != self.data:
            raise ValueError("Artifact part base64 must be canonical")
        return self


class ManifestV1(StrictModel):
    """Order physical parts while preserving one validated logical body identity."""

    logical_kind: Literal[
        "decision_set",
        "llm_output",
        "write_set",
        "result_index",
        "finalize_plan",
        "policy_config",
        "run_input",
        "erasure_plan",
        "semantic_embedding",
    ]
    schema_version: Literal["1"]
    body_digest: Integrity
    parts: Annotated[
        list[Annotated[ArtifactId, RefSpec("artifact", artifact_kinds=("part",))]],
        Field(min_length=1),
    ]
    total_bytes: PositiveInt


class DigestSettlementSetV1(StrictModel):
    """One owner's settled positions as an artifact body.

    An artifact body is a JSON object, so this wraps the settlement list
    (`decision_set:settlement`). It adds no settlement semantics: each entry is
    `DigestSettlementV1`. Entries are sorted by `position_key`
    and name each position once, so the same settlements have the same bytes.
    The variant is registered, with marker key `settlements`, together with the
    store metadata of `DigestSettlementV1`'s identity fields.
    """

    settlements: list[DigestSettlementV1]

    @model_validator(mode="after")
    def ordered_positions(self) -> Self:
        """Require unique position keys in sorted order."""
        keys = [settlement.position_key for settlement in self.settlements]
        if keys != sorted(set(keys)):
            raise ValueError("Settlements are sorted by position_key and name each position once")
        return self


ARTIFACT_MODELS: dict[str, type[StrictModel]] = {
    "llm_output": StoredInferenceOutput,
    "decision_set:judge_call": JudgeCallManifestV1,
    "result_index:judge": JudgeResultIndexV8,
    "semantic_embedding": StatementEmbeddingV1,
    "proposal_snapshot": ProposalV2,
    "result_index:erasure_receipt": ErasureReceiptResultIndexV1,
    "result_index:source_labels": SourceLabelsResultIndexV1,
    "result_index:extraction_refusal": SessionExtractionRefusalResultIndexV1,
    "result_index:provider_failure": SessionProviderFailureResultIndexV1,
    "result_index:receipt": ReceiptResultIndexV1,
    "result_index:add_completion": AddCompletionRecordV1,
    "erasure_plan:evidence_set": IngestionErasurePlanV1,
    "receipt:forget_prepare": ForgetPrepareReceiptV3,
    "receipt:forget_final": ForgetReceiptV3,
    "receipt:redaction": RedactionReceiptV1,
    "request_ledger": RequestLedgerV1,
    "decision_set:neighbor_snapshot": NeighborSnapshotV1,
    "decision_set:extraction_context": ExtractContextV2,
    "decision_set:privacy_embedding": PrivacyEmbeddingV1,
    "decision_set:settlement": DigestSettlementSetV1,
    "decision_set:erased_extraction": ErasedExtractionV1,
    "decision_set:finalize_source_inventory": FinalizeSourceInventoryBlockV1,
    "decision_set:finalize_source_ttl": FinalizeSourceTTLBlockV1,
    "auth_context": AuthContextV1,
    "policy_config:recall:v4": RecallPolicyConfigV4,
    "policy_config:extraction:v2": ExtractionPolicyConfigV2,
    "policy_config:embedding_canary": EmbeddingCanaryConfigV1,
    "policy_release:bootstrap": BootstrapPolicyReleaseV1,
    "part": PartV1,
    "manifest": ManifestV1,
    "publish_plan": CommitPlanV1,
    "write_set:commit_journal": CommitJournalV1,
    **{f"run_input:{key}": value for key, value in RUN_INPUT_MODELS.items()},
}

# Logical variants of one physical kind are told apart by a marker key:
# a required field of the variant's model that no other registered model of the
# same kind has. A body selects the variant whose marker it carries; none or more
# than one is `unsupported_variant`. Kinds with a semantic discriminator keep it:
# `policy_config` (policy kind and version), `receipt` (`receipt_kind`) and
# `erasure_plan` (`target.kind`).
#
# Registering a variant. The change that adds it:
# 1. defines the model in a core leaf module and imports it here directly; there
#    is no runtime registration hook. Every identity-named field (`id`, `*_id`,
#    `*_ids`, `*_ref`, `*_refs`, `*_digest`, `*_hash`) carries its store
#    metadata, a RefSpec or an OpaqueIdSpec: the schema registry
#    (`store/schemas.py` schema_registry) refuses an undeclared identity path;
# 2. adds its ARTIFACT_MODELS row and, for decision_set, result_index and
#    write_set, its marker below; a policy row is `policy_config:<kind>:v<n>`;
# 3. when a body can exceed 262,144 characters, adds the kind to SHARDABLE_KINDS
#    and ManifestV1.logical_kind together, and lets every reference to such a
#    body accept `manifest` beside the kind in its RefSpec `artifact_kinds`;
# 4. adds the variant to the artifact registry tests.
MARKER_KEYS: dict[str, dict[str, str]] = {
    "decision_set": {
        "decision_set:judge_call": "member_snapshot_refs",
        "decision_set:neighbor_snapshot": "lexical",
        "decision_set:extraction_context": "render_config_digest",
        "decision_set:privacy_embedding": "privacy_embedding",
        "decision_set:settlement": "settlements",
        "decision_set:erased_extraction": "erased_extraction",
        "decision_set:finalize_source_inventory": "finalize_source_inventory",
        "decision_set:finalize_source_ttl": "finalize_source_ttl",
    },
    "result_index": {
        "result_index:judge": "member_input_digest",
        "result_index:erasure_receipt": "erasure_receipt_ref",
        "result_index:source_labels": "source_labels",
        "result_index:extraction_refusal": "extraction_refusal",
        "result_index:provider_failure": "provider_failure",
        "result_index:receipt": "receipt_ref",
        "result_index:add_completion": "completed_requests",
    },
    "write_set": {"write_set:commit_journal": "journal_key"},
}


def marker_key_refusal(registry: dict[str, dict[str, str]]) -> str | None:
    """Say why a marker table cannot dispatch unambiguously, or nothing.

    Every registered variant of a marked kind needs exactly one marker, which
    is a required field of its model and a field of no other registered model
    of that kind.
    """
    for kind, markers in registry.items():
        variants = {
            variant: model for variant, model in ARTIFACT_MODELS.items()
            if variant == kind or variant.startswith(kind + ":")
        }
        if set(markers) != set(variants):
            return f"Every {kind} variant needs exactly one marker key"
        for variant, marker in markers.items():
            field = variants[variant].model_fields.get(marker)
            if field is None or not field.is_required():
                return f"Marker {marker} is not a required field of {variant}"
            if any(
                marker in model.model_fields
                for other, model in variants.items() if other != variant
            ):
                return f"Marker {marker} of {variant} is a field of another {kind} variant"
    return None


def _marked_variant(kind: str, raw: dict[str, object]) -> str:
    """Select the one registered variant whose marker key the body carries."""
    present = [variant for variant, key in MARKER_KEYS[kind].items() if key in raw]
    if len(present) != 1:
        raise QuoteMemError("unsupported_variant", f"Artifact variant is not implemented: {kind}")
    return present[0]


def parse_artifact_body(kind: str, raw: object, schema_version: str = "1") -> StrictModel:
    """Parse one direct or reconstructed logical Artifact body.

    Args:
        kind: The physical or logical artifact kind.
        raw: Decoded JSON or a canonical physical JSON string.
        schema_version: The exact version of the logical schema.

    Returns:
        The implemented strict body model.

    Raises:
        QuoteMemError: For unsupported schema variants, malformed JSON, or invalid fields.
    """
    if schema_version != "1":
        raise QuoteMemError("unsupported_variant", "Artifact schema version is not implemented")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (ValueError, TypeError) as error:
            raise QuoteMemError("schema_invalid", "Artifact body is not valid JSON") from error
    if not isinstance(raw, dict):
        raise QuoteMemError("schema_invalid", "Artifact body must be an object")
    raw_dict = cast(dict[str, object], raw)
    if kind == "run_input":
        return parse_run_input(raw_dict)
    if kind == "policy_config":
        policy_kind = raw_dict.get("policy_kind")
        if policy_kind == "embedding_canary":
            variant = f"{kind}:embedding_canary"
        else:
            version = raw_dict.get("config_version")
            if type(version) is not int:
                raise QuoteMemError("schema_invalid",
                                    "Policy config version must be an exact integer")
            variant = f"{kind}:{policy_kind}:v{version}"
    elif kind == "policy_release":
        variant = f"{kind}:{raw_dict.get('action')}"
    elif kind in MARKER_KEYS:
        variant = _marked_variant(kind, raw_dict)
        if variant == "result_index:source_labels" and (
            type(raw_dict["source_labels"]) is not int or raw_dict["source_labels"] != 1
        ):
            raise QuoteMemError("unsupported_variant",
                                "Source-label result version is not implemented")
    elif kind == "erasure_plan":
        target = raw_dict.get("target")
        target_kind = (
            cast(dict[str, object], target).get("kind") if isinstance(target, dict) else None
        )
        variant = f"{kind}:{target_kind}"
    elif kind == "receipt" and raw_dict.get("receipt_kind") in {
        "erasure_prepare", "erasure_final",
    }:
        variant = ("receipt:forget_prepare" if raw_dict["receipt_kind"] == "erasure_prepare"
                   else "receipt:forget_final")
    elif kind == "receipt":
        variant = f"{kind}:{raw_dict.get('receipt_kind')}"
        reason = raw_dict.get("reason_code")
        if variant == "receipt:redaction" and (
            not isinstance(reason, str) or reason not in REDACTION_REASONS
        ):
            raise QuoteMemError("unsupported_variant", "This receipt producer is not implemented")
    else:
        variant = kind
    model = ARTIFACT_MODELS.get(variant)
    if model is None:
        raise QuoteMemError("unsupported_variant",
                            f"Artifact variant is not implemented: {variant}")
    if kind == "llm_output" and "trial_side" in raw_dict:
        raise QuoteMemError("unsupported_variant", "Paired inference outputs are not implemented")
    try:
        return model.model_validate(raw_dict)
    except ValidationError as error:
        raise QuoteMemError("schema_invalid", "Artifact body violates its strict schema") from error


if frozenset(get_args(ManifestV1.model_fields["logical_kind"].annotation)) != SHARDABLE_KINDS:
    # Two lists decide sharding and neither derives from the other:
    # `SHARDABLE_KINDS` decides whether a body may be split at all, and this
    # `Literal` decides whether the manifest that records the split can be
    # written. A kind added to only the first starts sharding and then fails
    # its own manifest validation, which surfaces as `schema_invalid` on a
    # body far from the list that is actually wrong. Disagreeing lists are an
    # import-time failure instead.
    raise AssertionError("Shardable kinds and manifest kinds must name the same set")
_MARKER_REFUSAL = marker_key_refusal(MARKER_KEYS)
if _MARKER_REFUSAL is not None:
    # A marker shared by two variants of one kind, or missing from its model,
    # would let a body parse as another variant; refuse at import instead.
    raise AssertionError(_MARKER_REFUSAL)


_SCOPED_KINDS = frozenset({
    "proposal_snapshot", "semantic_embedding", "publish_plan", "decision_set",
    "llm_output", "finalize_plan", "erasure_plan", "receipt",
})


def placement_refusal(
    kind: str, scope_id: str | None, security_level: str, parsed: object,
) -> str | None:
    """Say why this kind may not sit at this scope and protection, or nothing.

    These rules must be applied to the LOGICAL kind, not only to the kind of the
    row that carries it. A body over `MAX_BODY_CHARS` is stored as a manifest
    and its parts, and those rows name themselves `manifest` and `part`, so a
    rule matched against the row's own kind stops applying to exactly the
    artifacts that grew large enough to be split. `Writer.put_logical_artifact` therefore asks
    this before it decides how to store anything, and the envelope validator
    asks it again for the row it is about to write.
    """
    if isinstance(
        parsed, ErasureReceiptResultIndexV1 | ForgetPrepareReceiptV3 | ForgetReceiptV3,
    ):
        if scope_id is not None or security_level != "restricted":
            return "Erasure receipts require a restricted global controller"
        return None
    if kind in _SCOPED_KINDS and scope_id is None:
        return f"A {kind} requires an exact creator scope"
    if kind == "erasure_plan" and security_level != "restricted":
        return "Erasure plans require restricted security"
    if isinstance(parsed, ReceiptResultIndexV1 | JudgeResultIndexV8) and scope_id is None:
        return "Ingestion receipt results require an exact creator scope"
    if kind in {"policy_config", "policy_release"} and scope_id is not None:
        return f"A {kind} must belong to a global creator"
    return None


class ArtifactEnvelopeV1(StrictModel):
    """Define exactly the canonical fields used for the content address."""

    artifact_kind: ArtifactKind
    schema_version: Literal["1"]
    tenant_id: Annotated[str, DerivedSpec("workspace.tenant_id"), OpaqueIdSpec("local")]
    scope_kind: (
        Annotated[
            Literal["user", "project", "agent", "organization"], DerivedSpec("creator.scope_kind")
        ]
        | None
    )
    scope_id: Annotated[ScopeId, DerivedSpec("creator.scope_id"), OpaqueIdSpec("local")] | None
    security_level: Literal["internal", "confidential", "restricted"]
    body_refs: Annotated[list[ReferenceProjectionV1], DerivedSpec("body")]
    created_by_run_id: Annotated[
        ULID,
        RefSpec(
            "run",
            ref_mode="snapshot_locator",
            fulfillment="pre_state",
            target_forget_action="retain_locator",
        ),
    ]
    body: Annotated[str, Field(max_length=262144)]

    @model_validator(mode="after")
    def validate_body(self, info: ValidationInfo) -> Self:
        """Validate physical canonical bytes and the artifact's scope category.

        A caller that also needs the strict body passes the validation context
        ``{"parsed_body": []}``: the body this validation parsed is appended to that
        list, so the caller does not parse the same bytes a second time.
        """
        if (self.scope_kind is None) != (self.scope_id is None):
            raise ValueError("Artifact scope kind and ID must appear together")
        parsed = parse_artifact_body(self.artifact_kind, self.body, self.schema_version)
        if canonical_json(parsed) != self.body:
            raise ValueError("Artifact body must contain the canonical strict JSON bytes")
        misplaced = placement_refusal(
            self.artifact_kind, self.scope_id, self.security_level, parsed,
        )
        if misplaced is not None:
            raise ValueError(misplaced)
        context = cast(dict[str, object] | None, info.context)
        if isinstance(context, dict):
            out = context.get("parsed_body")
            if isinstance(out, list):
                cast(list[StrictModel], out).append(parsed)
        return self


class ArtifactRowV1(ArtifactEnvelopeV1):
    """Materialize the complete physical Artifact row without another body authority."""

    object_id: Annotated[ArtifactId, DerivedSpec("envelope"), OpaqueIdSpec("local")]
    artifact_refs: Annotated[
        list[ArtifactId], DerivedSpec("body"), RefSpec("artifact", cardinality="many")
    ]
    size_bytes: Annotated[NonNegativeInt, DerivedSpec("body")]
    created_at: Timestamp

    @model_validator(mode="after")
    def validate_address(self, info: ValidationInfo) -> Self:
        """Check the complete envelope address and UTF-8 body byte count.

        A row the Writer materializes derives its address from the very
        envelope it validates here; that caller passes the validation context
        ``{"artifact_address": "derived"}`` and the address recomputation is
        skipped (the byte count is still checked). Every other construction,
        a stored row above all, recomputes the address: under the normal-load
        registration the second hash of every materialized body was 8 percent
        of the process's running samples.
        """
        context = cast(dict[str, object] | None, info.context)
        derived = isinstance(context, dict) and context.get("artifact_address") == "derived"
        if not derived:
            envelope = self.model_dump(mode="json", include=set(ArtifactEnvelopeV1.model_fields))
            if self.object_id != artifact_id(envelope):
                raise ValueError("Artifact address does not match its canonical envelope")
        if self.size_bytes != len(self.body.encode("utf-8")):
            raise ValueError("Artifact size does not match its canonical UTF-8 body")
        return self

