"""Canonical admission producers first exercised in M0, derived from plan.md §5.6.

These strict producers cover bootstrap, external session admission, source-label
inputs, finalize roots and ingestion-only cleanup. Other workflows retain their
strict run_input schemas but do not acquire an unrestricted admission producer here.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from typing import Annotated, Literal, Self, cast, get_args

from pydantic import Field, ValidationError, model_validator

from aml.core.entities import PositionKey
from aml.core.errors import AMLError
from aml.core.identity import (
    ContentBindingDigest,
    IntegrityDigest,
    PrincipalId,
    ScopeId,
    base64url,
    canonical_json,
)
from aml.core.judge_records import (
    JudgmentAdmissionSeedV1,
    judgment_business_key,
    judgment_family_key,
)
from aml.core.privacy_persistence import (
    PRIVACY_ERASE_INPUT,
    ConsentWithdrawalInputV1,
    IngestionSplitSecretInputV1,
    PreAddSplitSecretInputV1,
    PrivacyEraseInputV1,
    SourceExpiryInputV1,
)
from aml.core.query import ScopeV1
from aml.core.review_rounds import RoundNo
from aml.core.run_inputs import (
    EmbeddingBootstrapKeyV1,
    EntityCreateV1,
    EntitySetTypeV1,
    EntityUpdateKeyV1,
    FinalizeBusinessKeyV1,
    MessagesRequestPayloadV1,
    PolicyBootstrapKeyV1,
    PublishCommitSeedV1,
    SessionBusinessKeyV1,
    SourceLabelsAdmissionSeedV1,
    TextRequestPayloadV1,
    publish_commit_business_key,
    source_labels_business_key,
)
from aml.core.source_labels import SourceLabelsTargetV1
from aml.core.technical_types import PositiveInt
from aml.store.metadata import StrictModel


class InternalOptionsV1(StrictModel):
    """Mark the closed internal dispatcher option without accepting arbitrary flags."""

    internal: Literal[True]


class EmptyOptionsV1(StrictModel):
    """The current external message admission has no additional execution options."""


class FinalizeRequestPayloadV1(StrictModel):
    """Bind only the finalize cycle selected by the existing lease owner (§9)."""

    finalize_cycle: PositiveInt


# The run kinds of A.3.10; the store's `RunKind` names the same set.
AdmittedRunKind = Literal[
    "session_ingestion",
    "finalize",
    "publish",
    "evaluate",
    "retention",
    "forget",
    "erasure_controller",
    "entity_update",
    "policy_release",
    "embedding_switch",
]
_ADMITTED_RUN_KINDS = frozenset(get_args(AdmittedRunKind))
_OPENED_RETENTION = frozenset({
    ("split_secret", "pre_add"),
    ("split_secret", "ingestion_child"),
    ("source_expiry", "clock"),
    ("consent_withdrawal", "consent_epoch"),
})
"""The privacy-erase producers (D.7): retention role and trigger of each opened input."""


class CanonicalRunAdmission(StrictModel):
    """Canonical input frozen before new server allocation for the implemented producers."""

    v: Literal[1]
    run_kind: AdmittedRunKind
    commit_mode: Literal["orchestrator", "single_compensatable", "cas_only", "destructive_scan"]
    principal_id: PrincipalId
    scope_kind: Literal["user", "project", "agent", "organization"] | None = None
    scope_id: ScopeId | None = None
    target_kind: Literal["session", "memory", "candidate", "none", "affected_set", "scope"]
    target_locator: Annotated[str, Field(min_length=1, max_length=63)] | None = None
    raw_business_key: (
        SessionBusinessKeyV1
        | PolicyBootstrapKeyV1
        | EmbeddingBootstrapKeyV1
        | IntegrityDigest
        | PrivacyEraseInputV1
        | FinalizeBusinessKeyV1
        | EntityUpdateKeyV1
    )
    request_payload: (
        MessagesRequestPayloadV1
        | TextRequestPayloadV1
        | PolicyBootstrapKeyV1
        | EmbeddingBootstrapKeyV1
        | PrivacyEraseInputV1
        | FinalizeRequestPayloadV1
        | SourceLabelsAdmissionSeedV1
        | EntityCreateV1
        | EntitySetTypeV1
        | PublishCommitSeedV1
        | JudgmentAdmissionSeedV1
    )
    options: InternalOptionsV1 | EmptyOptionsV1

    @model_validator(mode="after")
    def validate_branch(self) -> Self:
        """Reject crossed keys, scopes, options, and execution modes before hashing."""
        scoped = self.scope_id is not None
        if scoped != (self.scope_kind is not None):
            raise ValueError("Admission scope fields must be present together")
        if (self.target_kind == "none") != (self.target_locator is None):
            raise ValueError("Admission target locator must match target kind")
        key, payload = self.raw_business_key, self.request_payload
        if self.run_kind in {"policy_release", "embedding_switch"}:
            expected = (
                PolicyBootstrapKeyV1
                if self.run_kind == "policy_release"
                else EmbeddingBootstrapKeyV1
            )
            if (
                not isinstance(key, expected)
                or not isinstance(payload, expected)
                or key != payload
                or scoped
                or self.target_kind != "none"
                or self.commit_mode != "cas_only"
                or not isinstance(self.options, InternalOptionsV1)
            ):
                raise ValueError("Bootstrap admission must bind its exact global installed default")
        elif self.run_kind == "session_ingestion":
            if not scoped or self.target_kind != "session":
                raise ValueError("Session admission requires an exact scoped session target")
            if isinstance(key, SessionBusinessKeyV1):
                if (
                    not isinstance(payload, MessagesRequestPayloadV1 | TextRequestPayloadV1)
                    or (key.session_id, key.request_id) != (payload.session_id, payload.request_id)
                    or self.commit_mode != "orchestrator"
                    or not isinstance(self.options, EmptyOptionsV1)
                ):
                    raise ValueError("External session admission has inconsistent request identity")
            else:
                raise AMLError(
                    "unsupported_variant", "This session admission producer is not implemented"
                )
        elif self.run_kind == "finalize":
            if (
                not isinstance(key, FinalizeBusinessKeyV1)
                or not isinstance(payload, FinalizeRequestPayloadV1)
                or not scoped
                or self.target_kind != "scope"
                or self.target_locator != self.scope_id
                or key.scope_id != self.scope_id
                or key.finalize_cycle != payload.finalize_cycle
                or self.commit_mode != "orchestrator"
                or not isinstance(self.options, InternalOptionsV1)
            ):
                raise ValueError("Finalize root admission requires its exact internal scope cycle")
        elif self.run_kind == "retention":
            if (
                not isinstance(
                    key,
                    PreAddSplitSecretInputV1
                    | IngestionSplitSecretInputV1
                    | SourceExpiryInputV1
                    | ConsentWithdrawalInputV1,
                )
                or type(payload) is not type(key)
                or payload != key
                or not scoped
                or self.target_kind != "affected_set"
                or self.commit_mode != "destructive_scan"
                or not isinstance(self.options, InternalOptionsV1)
            ):
                raise AMLError(
                    "unsupported_variant", "Only the bound privacy-erase producers are opened"
                )
            ContentBindingDigest(self.target_locator)
        elif isinstance(payload, SourceLabelsAdmissionSeedV1):
            if (
                self.run_kind != "evaluate" or not scoped
                or self.target_kind != "scope" or self.target_locator != self.scope_id
                or self.commit_mode != "single_compensatable"
                or not isinstance(self.options, InternalOptionsV1)
                or key != source_labels_business_key(payload)
            ):
                raise ValueError("Source labels require their exact scoped source seed")
        elif self.run_kind == "entity_update":
            if (
                not isinstance(key, EntityUpdateKeyV1)
                or not isinstance(payload, EntityCreateV1 | EntitySetTypeV1)
                or key.request_id != payload.request_id
                or not scoped
                or self.target_kind != "scope"
                or self.target_locator != self.scope_id
                or self.commit_mode != "single_compensatable"
                or not isinstance(self.options, EmptyOptionsV1)
            ):
                raise ValueError("An Entity SDK call binds its exact scope and request id")
        elif isinstance(payload, PublishCommitSeedV1):
            if (
                self.run_kind != "publish" or not scoped
                or self.target_kind != "scope" or self.target_locator != self.scope_id
                or self.commit_mode != "single_compensatable"
                or not isinstance(self.options, InternalOptionsV1)
                or key != publish_commit_business_key(payload)
            ):
                raise ValueError("A publish child requires its exact scoped component")
        elif isinstance(payload, JudgmentAdmissionSeedV1):
            # One evaluate:judgment child per reviewing owner, position and round (C.13).
            if (
                self.run_kind != "evaluate" or not scoped
                or self.target_kind != "scope" or self.target_locator != self.scope_id
                or self.commit_mode != "single_compensatable"
                or not isinstance(self.options, InternalOptionsV1)
                or key != judgment_business_key(payload)
            ):
                raise ValueError("A judgment requires its exact scoped position and round seed")
        else:
            raise AMLError("unsupported_variant", "Admission producer is not implemented")
        return self


def canonicalize_admission(raw: object) -> CanonicalRunAdmission:
    """Parse only implemented producers and normalize optional absence to explicit null."""
    if isinstance(raw, dict) and (
        cast(dict[str, object], raw).get("run_kind") not in _ADMITTED_RUN_KINDS
    ):
        raise AMLError("unsupported_variant", "This admission producer is not implemented")
    if isinstance(raw, dict) and cast(dict[str, object], raw).get("run_kind") == "retention":
        key = cast(dict[str, object], raw).get("raw_business_key")
        if isinstance(key, dict):
            key_fields = cast(dict[str, object], key)
            role, trigger = key_fields.get("retention_role"), key_fields.get("trigger")
            opened = (
                isinstance(role, str) and isinstance(trigger, str)
                and (role, trigger) in _OPENED_RETENTION
            )
            if not opened:
                raise AMLError("unsupported_variant", "This retention producer is not implemented")
    try:
        return CanonicalRunAdmission.model_validate(raw)
    except ValidationError as error:
        raise AMLError("schema_invalid", "Canonical Run admission is invalid") from error


def privacy_erase_input_projection(value: PrivacyEraseInputV1) -> dict[str, object]:
    """Bind the complete content-free admission payload before any plan Artifact exists."""
    validated = PRIVACY_ERASE_INPUT.validate_python(value.model_dump(mode="json"))
    return cast(dict[str, object], validated.model_dump(mode="json"))


def privacy_erase_target_locator(
    value: PrivacyEraseInputV1,
    bind_content: Callable[[str, object], ContentBindingDigest],
) -> ContentBindingDigest:
    """Derive the affected-set binding from the exact affected identities and keyed pre-states."""
    projection = privacy_erase_input_projection(value)
    return bind_content("privacy-erase-target", projection["affected"])


def privacy_erase_admission(
    scope: ScopeV1,
    value: PrivacyEraseInputV1,
    principal_id: str,
    bind_content: Callable[[str, object], ContentBindingDigest],
) -> CanonicalRunAdmission:
    """Return the one canonical admission of a privacy erase (D.7).

    The Writer admits from it, the plan loader re-proves its owner against it and
    a replay of a completed selection finds its owner by it: one family per
    principal, scope and complete selection.

    Args:
        scope: The erase's exact scope.
        value: Its complete selection.
        principal_id: The admitting principal.
        bind_content: The tenant Keyring's canonical HMAC callback.

    Returns:
        The canonical retention admission, target ``affected_set``.
    """
    projection = privacy_erase_input_projection(value)
    return canonicalize_admission(
        {
            "v": 1,
            "run_kind": "retention",
            "commit_mode": "destructive_scan",
            "principal_id": principal_id,
            **scope.model_dump(mode="json"),
            "target_kind": "affected_set",
            "target_locator": privacy_erase_target_locator(value, bind_content),
            "raw_business_key": projection,
            "request_payload": projection,
            "options": {"internal": True},
        }
    )


def split_secret_parent_binding(
    parent_row: Mapping[str, object],
    bind_content: Callable[[str, object], ContentBindingDigest],
) -> ContentBindingDigest:
    """Bind immutable ingestion admission identity while allowing legitimate worker recovery.

    Current writer epoch, lifecycle, execution stage, and fence authority must
    still be checked separately by the sole Writer before any child effect.
    """
    from aml.store.domain import SessionTargetRunRowV1

    parent = SessionTargetRunRowV1.model_validate(dict(parent_row))
    if parent.run_kind != "session_ingestion" or parent.commit_mode != "orchestrator":
        raise AMLError("scope_denied", "Cleanup parent must be the admitted ingestion root")
    fields = (
        "object_id",
        "run_kind",
        "commit_mode",
        "principal_id",
        "scope_kind",
        "scope_id",
        "target_kind",
        "target_id",
        "idempotency_key",
        "run_generation",
        "admission_digest",
        "parent_run_id",
        "created_at",
    )
    value = parent.model_dump(mode="json")
    return bind_content("split-secret-parent", {field: value[field] for field in fields})


def admission_identity(
    admission: CanonicalRunAdmission,
    bind_content: Callable[[str, object], ContentBindingDigest],
) -> tuple[str, ContentBindingDigest, ContentBindingDigest]:
    """Return the exact idempotency, admission, and request-payload bindings."""
    value = admission.model_dump(mode="json")
    family = {
        "scope_kind": value["scope_kind"],
        "scope_id": value["scope_id"],
        "business_key": value["raw_business_key"],
    }
    if not isinstance(admission.raw_business_key, SessionBusinessKeyV1):
        family["principal_id"] = value["principal_id"]
    identity = _family_identity(family)
    return (
        identity,
        bind_content("run-admission", value),
        bind_content("run-input-payload", value["request_payload"]),
    )


def _family_identity(family: Mapping[str, object]) -> str:
    return "idem_" + base64url(hashlib.sha256(canonical_json(dict(family)).encode()).digest())


def judgment_family(
    *, principal_id: str, scope: ScopeV1, owner_run_id: str, position_key: PositionKey,
    round_no: RoundNo,
) -> str:
    """Locate one owner's judge child family for a position and round before its input.

    The family of ``admission_identity`` over the judgment admission, which names
    the reviewing owner, the position and the round but not the member input its
    admission digest binds: one child exists per family (C.13).
    """
    exact = ScopeV1.model_validate(scope.model_dump(mode="json"))
    return _family_identity({
        "scope_kind": exact.scope_kind, "scope_id": exact.scope_id, "principal_id": principal_id,
        "business_key": judgment_family_key(owner_run_id, position_key, round_no),
    })


def source_labels_family(
    *, principal_id: str, scope: ScopeV1, target: SourceLabelsTargetV1,
) -> str:
    """Locate an original source-label family even before its input is attached.

    This content-free selector shares the exact canonical admission projection.
    It grants neither execution nor completed-result authority to an empty child.
    """
    exact = ScopeV1.model_validate(scope.model_dump(mode="json"))
    selected = SourceLabelsTargetV1.model_validate(target.model_dump(mode="json"))
    return _family_identity({
        "scope_kind": exact.scope_kind, "scope_id": exact.scope_id,
        "principal_id": principal_id, "business_key": source_labels_business_key(selected),
    })


def session_request_family(scope: ScopeV1, session_id: str, request_id: str) -> str:
    """Locate the canonical external session family without inventing a principal.

    This is a read selector, not admission or authorization. External request
    identity excludes the principal; full admission digest still includes it.
    The projection is shared with admission_identity (plan.md §5.6).
    """
    scope = ScopeV1.model_validate(scope.model_dump(mode="json"))
    key = SessionBusinessKeyV1(session_id=session_id, request_id=request_id)
    return _family_identity(
        {
            "scope_kind": scope.scope_kind,
            "scope_id": scope.scope_id,
            "business_key": key.model_dump(mode="json"),
        }
    )


def finalize_admission(
    *,
    principal_id: str,
    scope: ScopeV1,
    cycle: int,
) -> CanonicalRunAdmission:
    """Bind one internal finalize root without reading or acquiring its lease.

    Args:
        principal_id: Principal independently authorized for this exact scope.
        scope: Registered scope whose input and publication cycle will finalize.
        cycle: Positive cycle independently selected under plan.md §9: one for
            an absent lease, or the settled terminal lease generation plus one.
            An existing preparing or held owner must be reused by the caller.

    Returns:
        CanonicalRunAdmission: Strict scope-targeted orchestrator identity for
        the sole Writer, with the existing scope/cycle business key and only
        finalize_cycle in its request payload. This helper allocates no Run.

    Raises:
        AMLError: If the scope, principal or cycle is malformed. This function
            does not prove authorization, cycle availability or lease ownership.
    """
    return canonicalize_admission(
        {
            "v": 1,
            "run_kind": "finalize",
            "commit_mode": "orchestrator",
            "principal_id": principal_id,
            **scope.model_dump(mode="json"),
            "target_kind": "scope",
            "target_locator": scope.scope_id,
            "raw_business_key": {"scope_id": scope.scope_id, "finalize_cycle": cycle},
            "request_payload": {"finalize_cycle": cycle},
            "options": {"internal": True},
        }
    )



def publish_admission(
    *, principal_id: str, scope: ScopeV1, seed: PublishCommitSeedV1,
) -> CanonicalRunAdmission:
    """Bind one ``publish`` child of an Add owner to the component it commits (C.12; I10).

    Args:
        principal_id: The owner's principal.
        scope: The owner's exact scope.
        seed: The component: owner, round, members and journal key.

    Returns:
        CanonicalRunAdmission: The scoped ``single_compensatable`` admission whose
        business key names the owner, round and members. This helper allocates
        no Run and proves neither the owner's lease nor its plan.

    Raises:
        AMLError: If the scope, principal or seed is malformed.
    """
    return canonicalize_admission(
        {
            "v": 1,
            "run_kind": "publish",
            "commit_mode": "single_compensatable",
            "principal_id": principal_id,
            **scope.model_dump(mode="json"),
            "target_kind": "scope",
            "target_locator": scope.scope_id,
            "raw_business_key": publish_commit_business_key(seed),
            "request_payload": seed.model_dump(mode="json"),
            "options": {"internal": True},
        }
    )
