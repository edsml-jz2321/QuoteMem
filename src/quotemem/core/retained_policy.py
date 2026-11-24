"""Read-only original policy/CAS authority shared by bootstrap and offline status."""

from typing import Any, Literal, Protocol

from quotemem.core.admission import admission_identity, canonicalize_admission
from quotemem.core.artifacts import BootstrapPolicyReleaseV1
from quotemem.core.controls import PolicyTier
from quotemem.core.errors import QuoteMemError
from quotemem.core.extraction_policy import ExtractionPolicyConfigV2
from quotemem.core.identity import integrity_digest
from quotemem.core.recall_policy import RecallPolicyConfigV4
from quotemem.core.run_inputs import PolicyBootstrapKeyV1, PolicyBootstrapRunInputV1
from quotemem.providers.keyring import Keyring
from quotemem.store.domain import DOMAIN_ADAPTERS
from quotemem.store.metadata import StrictModel


class PolicyReader(Protocol):
    """Read existing typed content without granting initialization or mutation."""

    @property
    def tenant_id(self) -> str: ...

    def get_row(self, kind: str, object_id: str) -> dict[str, Any] | None: ...

    def get_control(self, key: str) -> dict[str, Any] | None: ...

    def read_logical(self, reference: str, expected_kind: str) -> StrictModel: ...


def read_retained_bootstrap_policy(
    reader: PolicyReader, keyring: Keyring, principal: str,
    kind: Literal["recall", "extraction"], release_ref: str, *,
    pending_input: PolicyBootstrapRunInputV1 | None = None, tier: PolicyTier | None = None,
) -> tuple[StrictModel, PolicyBootstrapKeyV1]:
    """Read the selected original config and independently bind its bootstrap intent.

    ``tier`` selects the Add tier's pointer (``POLICY_POINTERS``); None is the default's.
    """
    release = reader.read_logical(release_ref, "policy_release")
    if not isinstance(release, BootstrapPolicyReleaseV1) or release.policy_kind != kind:
        raise QuoteMemError("parity_mismatch", "Installed policy release has an invalid kind")
    saved = reader.read_logical(release.config_ref, "policy_config")
    if saved.model_dump(mode="json").get("policy_kind") != kind:
        raise QuoteMemError("binding_mismatch", "Retained policy config has another kind")
    # Only the released families are retained: recall v4 and extraction
    # v2. No reader of a retired generation exists.
    family = RecallPolicyConfigV4 if kind == "recall" else ExtractionPolicyConfigV2
    if not isinstance(saved, family):
        raise QuoteMemError("binding_mismatch", "Retained policy config has a retired family")
    owner = reader.get_row("run", release.release_id)
    if owner is None:
        raise QuoteMemError("reference_missing", "Retained policy lost its original bootstrap")
    DOMAIN_ADAPTERS["run"].validate_python(owner)
    for reference in (release_ref, release.config_ref):
        envelope = reader.get_row("artifact", reference)
        if envelope is None or (
            envelope["created_by_run_id"] != owner["object_id"]
            or envelope["tenant_id"] != reader.tenant_id
            or envelope["scope_kind"] is not None or envelope["scope_id"] is not None
        ):
            raise QuoteMemError("binding_mismatch", "Retained policy has another artifact owner")
    key = PolicyBootstrapKeyV1(
        policy_kind=kind, action="bootstrap", builtin_config_digest=integrity_digest(saved),
        tier=tier,
    )
    canonical = canonicalize_admission({
        "v": 1, "run_kind": "policy_release", "commit_mode": "cas_only",
        "principal_id": principal, "target_kind": "none",
        "raw_business_key": key.model_dump(mode="json"),
        "request_payload": key.model_dump(mode="json"), "options": {"internal": True},
    })
    family, digest, _ = admission_identity(canonical, keyring.digest)
    if (owner["run_kind"], owner["commit_mode"], owner["principal_id"],
        owner["idempotency_key"], owner["admission_digest"]) != (
        "policy_release", "cas_only", principal, family, digest,
    ):
        raise QuoteMemError("binding_mismatch", "Retained policy changed canonical admission")
    if pending_input is None:
        if owner["status"] != "succeeded" or owner["input_ref"] is None:
            raise QuoteMemError("parity_mismatch", "Installed policy has no completed bootstrap")
        envelope = reader.get_row("artifact", owner["input_ref"])
        if envelope is None or envelope["created_by_run_id"] != owner["object_id"]:
            raise QuoteMemError("binding_mismatch", "Retained policy input has another creator")
        source = reader.read_logical(owner["input_ref"], "run_input")
    else:
        source = pending_input
    if not isinstance(source, PolicyBootstrapRunInputV1) or (
        source.business_key != key or source.admission_digest != digest
        or source.payload_digest != keyring.digest(
            "run-input-payload", key.model_dump(mode="json"),
        )
        or source.control_cas_intent.control_key != key.control_key
        or source.control_cas_intent.intended_post_image.value.model_dump(mode="json")
        != {"current_release_ref": release_ref}
        or source.control_cas_intent.intended_post_image.updated_by_run_id
        != owner["object_id"]
    ):
        raise QuoteMemError("binding_mismatch", "Retained policy lost its exact original intent")
    if pending_input is None:
        current = reader.get_control(key.control_key)
        if current is None or integrity_digest(current) != (
            source.control_cas_intent.intended_post_digest
        ):
            raise QuoteMemError("parity_mismatch", "Installed policy differs from its original CAS")
    return saved, key


def read_released_extraction_config(
    reader: PolicyReader, release_ref: str,
) -> ExtractionPolicyConfigV2:
    """Read the exact extraction release and config family selected by its caller.

    The consuming owner independently checks its frozen policy digest, creator
    and scope dependencies; this family read grants no execution or write rights.
    """
    release = reader.read_logical(release_ref, "policy_release")
    if not isinstance(release, BootstrapPolicyReleaseV1) or release.policy_kind != "extraction":
        raise QuoteMemError("binding_mismatch", "Governing release is not an extraction policy")
    config = reader.read_logical(release.config_ref, "policy_config")
    if not isinstance(config, ExtractionPolicyConfigV2):
        raise QuoteMemError("binding_mismatch", "Governing release selected another config family")
    return config


def read_released_recall_config(
    reader: PolicyReader, release_ref: str,
) -> RecallPolicyConfigV4:
    """Read the exact recall release and its v4 config family.

    The released recall policy has exactly one generation, v4; a release that
    selects any other config family is refused, never converted.
    """
    release = reader.read_logical(release_ref, "policy_release")
    if not isinstance(release, BootstrapPolicyReleaseV1) or release.policy_kind != "recall":
        raise QuoteMemError("binding_mismatch", "Governing release is not a recall policy")
    config = reader.read_logical(release.config_ref, "policy_config")
    if not isinstance(config, RecallPolicyConfigV4):
        raise QuoteMemError("binding_mismatch", "Governing release selected another config family")
    return config
