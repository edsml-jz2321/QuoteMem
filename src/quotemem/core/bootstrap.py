"""Exclusive-owner initialization and startup recovery through the single Writer.

`open_initialized_workspace` runs the boot order (`BOOT_STEPS`):
steps 1-12 initialize the owner, storage, providers, Writer, catalog, canary
and policies; steps 13-17 run the recovery steps of `RECOVERY_STEPS` that have
landed. The service reports readiness (step 18) after the yield. This module
does not start a listener or run Add or Search work.
"""

from __future__ import annotations

import base64
import contextlib
import os
import struct
import time
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final, Literal, get_args

import httpx

from quotemem.core.admission import admission_identity, canonicalize_admission
from quotemem.core.artifacts import (
    BootstrapPolicyReleaseV1,
    EmbeddingCanaryConfigV1,
    EmbeddingCanaryParametersV1,
)
from quotemem.core.controls import ControlRowV1, PolicyTier
from quotemem.core.embedding import embedding_state_token
from quotemem.core.errors import QuoteMemError
from quotemem.core.extraction_policy import (
    CHAT_FORGET_RULES,
    EXTRACTION_CALL_BUDGETS,
    EXTRACTION_SUPPLEMENTS,
    EXTRACTION_TEMPLATE_DIGESTS,
    JUDGE_GATES,
    JUDGE_MODES,
    QUOTE_GROUNDING,
    SELF_IDENTITY_RULES,
    build_extraction_policy,
)
from quotemem.core.identity import (
    ArtifactId,
    IntegrityDigest,
    canonical_json,
    integrity_digest,
    new_ulid,
)
from quotemem.core.inference import DEFAULT_PROVIDER_TIMEOUT_MS, default_retry_policy
from quotemem.core.platform_files import ensure_private_directory, open_private_file
from quotemem.core.platform_files import exclusive_owner as _exclusive_owner
from quotemem.core.platform_files import sync_directory as _sync_directory
from quotemem.core.recall_policy import build_recall_policy
from quotemem.core.run_inputs import (
    AbsentControlPreconditionV1,
    ControlCasIntentV1,
    EmbeddingBootstrapKeyV1,
    EmbeddingBootstrapRunInputV1,
    PolicyBootstrapKeyV1,
    PolicyBootstrapRunInputV1,
)
from quotemem.providers.auth import SingleUserAuth
from quotemem.providers.canary import (
    CANARY_TEXT_VERSION,
    CANARY_TEXTS,
    compare_canaries,
    encode_canary_vectors,
)
from quotemem.providers.config import EMBEDDING_MODEL, ProviderConfig, load_credentials
from quotemem.providers.contracts import EmbeddingRequest, EmbeddingResult
from quotemem.providers.deadline_transport import AsyncClientFactory
from quotemem.providers.factory import Providers, create_providers
from quotemem.providers.keyring import LocalKeyring
from quotemem.store.catalog_ledger import take_catalog_ledger
from quotemem.store.catalog_proof import leave_catalog_proof, take_catalog_proof
from quotemem.store.domain import DOMAIN_ADAPTERS, UntargetedRunRowV1
from quotemem.store.hb_runtime import HBRuntime, StorageOwner, open_runtime
from quotemem.store.metadata import StrictModel
from quotemem.store.writer import RowWrite, Writer, WriteTicket

if TYPE_CHECKING:
    from quotemem.ingestion.privacy_recovery import PendingPrivacy

BootstrapInput = PolicyBootstrapRunInputV1 | EmbeddingBootstrapRunInputV1
BootstrapHook = Callable[[str, str, Writer], None]


class WorkspaceConfiguration(StrictModel):
    """Persist nonsecret composition identity independently of personal settings."""

    format_version: Literal[1] = 1
    workspace_id: str = "aml"
    db_uuid: str
    tenant_id: str = "solo"
    providers: ProviderConfig


@dataclass(frozen=True)
class BootstrapProjection:
    """The initialized control projection; no retrieval index is claimed."""

    generation: int
    recall_release_ref: ArtifactId
    embedding_state_token: IntegrityDigest
    global_barrier: bool = False


@dataclass(frozen=True)
class InitializedWorkspace:
    """Hold the sole Writer and its lock after initialization and startup recovery."""

    runtime: HBRuntime
    writer: Writer
    keyring: LocalKeyring
    providers: Providers
    projection: BootstrapProjection
    drift_state: Literal["stable", "drift_warning", "space_mismatch"]
    mean_cosine_deviation: float


class WorkspaceCleanupPending(QuoteMemError):
    """Expose retained bootstrap ownership when storage shutdown did not finish.

    Attributes:
        resources: The same acquired resources. Retry its close after resolving
            the shutdown failure; a failed attempt has not released the owner.
    """

    def __init__(self, resources: WorkspaceBootstrapResources) -> None:
        self.resources = resources
        super().__init__("service_unhealthy", "Workspace cleanup remains owned and incomplete")


class WorkspaceBootstrapResources:
    """Retain acquired storage, transport and process ownership across failed cleanup.

    Construction performs no I/O. A service retains this object before entering
    open_initialized_workspace, so even a pre-yield bootstrap failure remains
    available to its explicit shutdown retry. Ordinary callers receive this same
    object in WorkspaceCleanupPending when automatic cleanup cannot finish.
    """

    def __init__(self) -> None:
        self._stack = ExitStack()
        self._storage_close: Callable[[], None] | None = None
        self._after_storage: Callable[[], object] | None = None
        self._used = False
        self._closed = False

    def _retain_storage(self, close: Callable[[], None]) -> None:
        if self._storage_close is not None or self._closed:
            raise RuntimeError("Workspace bootstrap already owns storage cleanup")
        self._storage_close = close

    def _after_storage_closed(self, callback: Callable[[], object]) -> None:
        """Run ``callback`` once storage acknowledges closure, before ownership is released."""
        self._after_storage = callback

    def close(self) -> None:
        """Close storage before transport/ownership, retaining failures for retry.

        Application workers and callbacks must already have joined. This method
        creates no retry loop and does not cancel or drain application work.
        Once storage has acknowledged closure, and while the process still owns
        the workspace, it records the lifetime's Catalog proof when the bootstrap
        registered one; that step's failure only leaves no proof.

        Raises:
            WorkspaceCleanupPending: If HB storage has not acknowledged closure.
            Exception: If subsequent transport or owner disposal reports failure.
        """
        if self._closed:
            return
        if self._storage_close is not None:
            try:
                self._storage_close()
            except BaseException:
                raise WorkspaceCleanupPending(self) from None
            self._storage_close = None
            after, self._after_storage = self._after_storage, None
            if after is not None:
                with contextlib.suppress(Exception):
                    after()
        self._stack.close()
        self._closed = True


def _configuration(
    root: Path, config: ProviderConfig | None, *, existing_only: bool = False,
) -> tuple[WorkspaceConfiguration, LocalKeyring]:
    path = root / "workspace.json"
    key_path = root / "tenant.key"
    if path.exists() or path.is_symlink():
        descriptor = open_private_file(path, os.O_RDONLY)
        with os.fdopen(descriptor, "rb") as stream:
            stored = WorkspaceConfiguration.model_validate_json(stream.read())
        if config is not None and stored.providers != config:
            raise QuoteMemError("workspace_binding_mismatch",
                                "Workspace provider configuration differs")
        return stored, LocalKeyring.load(key_path)
    if existing_only:
        raise QuoteMemError("workspace_binding_mismatch", "Existing workspace identity is missing")
    if config is None:
        raise QuoteMemError(
            "workspace_binding_mismatch", "New workspace requires provider configuration",
        )
    if any((root / name).exists() for name in ("memory.db", "hb-system.db")):
        raise QuoteMemError(
            "workspace_binding_mismatch", "Existing databases have no workspace identity"
        )
    keyring = LocalKeyring.load(key_path) if key_path.exists() else LocalKeyring.create(key_path)
    stored = WorkspaceConfiguration(db_uuid=uuid.uuid4().hex, providers=config)
    descriptor = open_private_file(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(canonical_json(stored) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    _sync_directory(root)
    return stored, keyring


class _Bootstrap:
    """Coordinate installed-default CAS Runs; every mutation uses the supplied Writer."""

    def __init__(
        self, writer: Writer, keyring: LocalKeyring, now: int, hook: BootstrapHook | None
    ) -> None:
        self.writer = writer
        self.keyring = keyring
        self.now = now
        self.hook = hook
        self.principal = SingleUserAuth(keyring, []).principal_id
        self.barrier = False

    def event(self, key: str, event: str) -> None:
        if self.hook is not None:
            self.hook(key, event, self.writer)

    def update(self, ticket: WriteTicket, **changes: object) -> dict[str, Any]:
        if ticket.run_id is None:
            raise QuoteMemError("reference_missing", "Bootstrap operation requires an admitted Run")
        row = self.writer.get_row("run", ticket.run_id)
        if row is None:
            raise QuoteMemError("reference_missing", "Bootstrap Run disappeared")
        row.update(changes)
        row["updated_at"] = self.now
        typed = DOMAIN_ADAPTERS["run"].validate_python(row)
        self.writer.write_bookkeeping(ticket, [RowWrite("run", typed)])
        return row

    def ambiguous(self, ticket: WriteTicket, code: str = "cas_outcome_ambiguous") -> None:
        self.update(ticket, status="manual_attention", failure_code=code, finished_at=None)
        raise QuoteMemError(code, "Bootstrap recovery cannot prove the frozen CAS outcome")

    def run(
        self, kind: Literal["policy_release", "embedding_switch"], key: StrictModel
    ) -> WriteTicket:
        canonical = canonicalize_admission(
            {
                "v": 1,
                "run_kind": kind,
                "commit_mode": "cas_only",
                "principal_id": self.principal,
                "target_kind": "none",
                "raw_business_key": key.model_dump(mode="json"),
                "request_payload": key.model_dump(mode="json"),
                "options": {"internal": True},
            }
        )
        family, admission, _ = admission_identity(canonical, self.keyring.digest)
        matches = [
            row
            for row in self.writer.iter_rows("run")
            if row["run_kind"] == kind and row["idempotency_key"] == family
        ]
        if len(matches) > 1:
            raise QuoteMemError("parity_mismatch", "Bootstrap has multiple Run identities")
        if matches:
            row = matches[0]
            if row["admission_digest"] != admission:
                raise QuoteMemError("parity_mismatch", "Bootstrap Run admission binding differs")
            if row["status"] not in {"queued", "running"}:
                raise QuoteMemError("service_unhealthy", "An existing bootstrap Run cannot execute")
            return self.writer.adopt_bootstrap_run(row["object_id"], self.now)
        lease = self.writer.get_control("lease:writer")
        if lease is None:
            raise QuoteMemError("service_unhealthy", "Writer lease is absent")
        row = UntargetedRunRowV1(
            object_id=new_ulid(),
            run_kind=kind,
            commit_mode="cas_only",
            status="queued",
            stage="await_input",
            attempt=0,
            idempotency_key=family,
            run_generation=0,
            admission_digest=admission,
            principal_id=self.principal,
            target_kind="none",
            writer_epoch=lease["value"]["writer_epoch"],
            created_at=self.now,
        )
        return self.writer.admit_run(row)

    def attach(self, ticket: WriteTicket, expected: BootstrapInput) -> BootstrapInput:
        row = self.writer.get_row("run", ticket.run_id or "")
        if row is None:
            raise QuoteMemError("reference_missing", "Bootstrap Run is absent")
        if row["input_ref"] is not None:
            actual = self.writer.read_logical(row["input_ref"], "run_input")
            if canonical_json(actual) != canonical_json(expected):
                self.ambiguous(ticket, "parity_mismatch")
            return expected
        found = self.writer.find_logical_candidates(ticket.run_id or "", "run_input")
        if len(found) > 1 or (found and canonical_json(found[0][1]) != canonical_json(expected)):
            self.ambiguous(ticket, "parity_mismatch")
        if isinstance(expected, EmbeddingBootstrapRunInputV1) and (
            self.writer.get_counter("commit_seq") != expected.reserved_generation - 1
        ):
            self.ambiguous(ticket)
        ref = (
            found[0][0]
            if found
            else self.writer.put_logical_artifact(
                ticket, "run_input", expected, created_at=row["created_at"]
            )
        )
        key = expected.control_cas_intent.control_key
        self.event(key, "intent_persisted")
        self.update(
            ticket,
            input_ref=ref,
            status="running",
            stage="persist_intent"
            if isinstance(expected, EmbeddingBootstrapRunInputV1)
            else "cas_current",
            started_at=row["started_at"] or self.now,
        )
        self.event(key, "intent_attached")
        return expected

    def apply(self, ticket: WriteTicket, payload: BootstrapInput) -> None:
        intent = payload.control_cas_intent
        key = intent.control_key
        intended = intent.intended_post_image
        with self.writer.mutex:
            current = self.writer.get_control(key)
            post = current is not None and integrity_digest(current) == intent.intended_post_digest
            pre = current is None
            if not post and not pre:
                self.ambiguous(ticket)
            if isinstance(payload, EmbeddingBootstrapRunInputV1):
                self.barrier = True
                generation = payload.reserved_generation
                sequence = self.writer.get_counter("commit_seq")
                if (
                    sequence > generation
                    or sequence < generation - 1
                    or (post and sequence != generation)
                ):
                    self.ambiguous(ticket)
                if sequence == generation - 1:
                    counter = self.writer.get_control("commit_seq")
                    if counter is None:
                        self.ambiguous(ticket)
                        raise AssertionError("Unreachable recovery state")
                    reserved = self.writer.materialize_control(
                        "commit_seq",
                        {"seq": generation},
                        counter["version"] + 1,
                        self.now,
                        ticket.run_id,
                    )
                    if not self.writer.swap_control(ticket, reserved, counter["version"]):
                        self.ambiguous(ticket)
                self.event(key, "generation_reserved")
            if pre:
                changed = self.writer.swap_control(ticket, intended, None)
                current = self.writer.get_control(key)
                if not changed and (
                    current is None or integrity_digest(current) != intent.intended_post_digest
                ):
                    self.ambiguous(ticket)
            self.event(key, "pointer_committed")
            # Initialization builds no retrieval index. This verified control projection is rebuilt
            # before initialization returns; a caller cannot observe the barrier.
            verified = self.writer.get_control(key)
            if verified is None or integrity_digest(verified) != intent.intended_post_digest:
                self.ambiguous(ticket)
            self.barrier = False
            self.update(
                ticket,
                status="succeeded",
                finished_at=self.now,
                stage="publish_snapshot"
                if isinstance(payload, EmbeddingBootstrapRunInputV1)
                else "cas_current",
            )

    def policy(
        self, kind: Literal["recall", "extraction"], config: StrictModel, *,
        tier: PolicyTier | None = None,
    ) -> None:
        """Hold the single Writer mutex through current-policy selection and CAS.

        ``tier`` installs an Add tier's own extraction policy at its own pointer
        (``POLICY_POINTERS``); None installs the default policy.
        """
        with self.writer.mutex:
            self._policy_locked(kind, config, tier)

    def _retained_policy_default(
        self, kind: Literal["recall", "extraction"], release_ref: str, *,
        pending_input: PolicyBootstrapRunInputV1 | None = None, tier: PolicyTier | None = None,
    ) -> tuple[StrictModel, PolicyBootstrapKeyV1]:
        """Read the original policy through the shared read-only authority."""
        from quotemem.core.retained_policy import read_retained_bootstrap_policy

        return read_retained_bootstrap_policy(
            self.writer, self.keyring, self.principal, kind, release_ref,
            pending_input=pending_input, tier=tier,
        )

    def _policy_locked(
        self, kind: Literal["recall", "extraction"], config: StrictModel,
        tier: PolicyTier | None,
    ) -> None:
        key = PolicyBootstrapKeyV1(
            policy_kind=kind, action="bootstrap", builtin_config_digest=integrity_digest(config),
            tier=tier,
        )
        pending = self.pending("policy_release", key.control_key)
        if pending is not None:
            candidates = self.writer.find_logical_candidates(pending["object_id"], "run_input")
            if len(candidates) != 1 or not isinstance(candidates[0][1], PolicyBootstrapRunInputV1):
                raise QuoteMemError("parity_mismatch",
                                    "Pending policy bootstrap input is ambiguous")
            original = candidates[0][1]
            if original.control_cas_intent.intended_post_image.updated_by_run_id != (
                pending["object_id"]
            ):
                raise QuoteMemError("binding_mismatch", "Pending policy input names another owner")
            reference = original.control_cas_intent.intended_post_image.value.model_dump(
                mode="json",
            ).get("current_release_ref")
            if not isinstance(reference, str):
                raise QuoteMemError("binding_mismatch", "Pending policy has another pointer shape")
            saved, retained_key = self._retained_policy_default(
                kind, reference, pending_input=original, tier=tier,
            )
            if saved != config:
                raise QuoteMemError(
                    "workspace_binding_mismatch", "A prior policy intent blocks this default",
                )
            config, key = saved, retained_key
        current = self.writer.get_control(key.control_key)
        if current is not None and pending is None:
            saved, _retained_key = self._retained_policy_default(
                kind, current["value"]["current_release_ref"], tier=tier,
            )
            if saved != config:
                raise QuoteMemError("workspace_binding_mismatch",
                                    "Installed default policy differs")
            return
        ticket = self.run("policy_release", key)
        row = self.writer.get_row("run", ticket.run_id or "")
        assert row is not None
        config_ref = self.writer.put_artifact(
            ticket, "policy_config", config, created_at=row["created_at"]
        )
        release = BootstrapPolicyReleaseV1(
            action="bootstrap",
            policy_kind=kind,
            release_id=ticket.run_id or "",
            config_ref=config_ref,
            reason_code="installed_default",
        )
        release_ref = self.writer.put_artifact(
            ticket, "policy_release", release, created_at=row["created_at"]
        )
        intended = self.writer.materialize_control(
            key.control_key,
            {"current_release_ref": release_ref},
            0,
            row["created_at"],
            ticket.run_id,
        )
        payload = PolicyBootstrapRunInputV1(
            run_kind="policy_release",
            business_key=key,
            admission_digest=row["admission_digest"],
            payload_digest=self.keyring.digest("run-input-payload", key.model_dump(mode="json")),
            control_cas_intent=self.intent(intended),
        )
        self.apply(ticket, self.attach(ticket, payload))

    def pending(self, kind: str, control_key: str) -> dict[str, Any] | None:
        found: list[dict[str, Any]] = []
        for row in self.writer.iter_rows("run"):
            if row["run_kind"] != kind or row["status"] in {"succeeded", "failed"}:
                continue
            candidates = self.writer.find_logical_candidates(row["object_id"], "run_input")
            if row["input_ref"] is not None:
                body = self.writer.read_logical(row["input_ref"], "run_input")
                candidates = [(row["input_ref"], body)]
            if not candidates:
                if kind == "embedding_switch":
                    found.append(row)
                continue
            for _, body in candidates:
                data = body.model_dump(mode="json")
                if data.get("control_cas_intent", {}).get("control_key") == control_key:
                    found.append(row)
                    break
        if len(found) > 1:
            raise QuoteMemError("parity_mismatch",
                                "Multiple nonterminal CAS intents target one key")
        if found and found[0]["status"] == "manual_attention":
            raise QuoteMemError("service_unhealthy", "Bootstrap recovery requires manual attention")
        return found[0] if found else None

    @staticmethod
    def intent(intended: ControlRowV1) -> ControlCasIntentV1:
        return ControlCasIntentV1(
            control_key=intended.key,
            expected_pre=AbsentControlPreconditionV1(exists=False),
            intended_post_image=intended,
            intended_post_digest=integrity_digest(intended),
        )

    def embedding(self, current_vectors: EmbeddingResult) -> EmbeddingCanaryConfigV1:
        """Keep one global barrier and mutex from intent preparation through projection."""
        with self.writer.mutex:
            self.barrier = True
            result = self._embedding_locked(current_vectors)
            self.barrier = False
            return result

    def _embedding_locked(self, current_vectors: EmbeddingResult) -> EmbeddingCanaryConfigV1:
        current = self.writer.get_control("policy:embedding_space")
        pending = self.pending("embedding_switch", "policy:embedding_space")
        if current is not None and pending is None:
            saved = self.writer.read_logical(
                current["value"]["canary_baseline_ref"], "policy_config"
            )
            if not isinstance(saved, EmbeddingCanaryConfigV1):
                raise QuoteMemError("parity_mismatch", "Embedding baseline has an invalid schema")
            self.check_baseline(current, saved, current_vectors)
            return saved
        baseline = EmbeddingCanaryConfigV1(
            policy_kind="embedding_canary",
            config_version=1,
            parameters=EmbeddingCanaryParametersV1(
                text_version=CANARY_TEXT_VERSION,
                model=current_vectors.resolved_model,
                dim=current_vectors.dimensions,
                vectors=encode_canary_vectors(current_vectors),
            ),
        )
        if pending is not None:
            candidates = self.writer.find_logical_candidates(pending["object_id"], "policy_config")
            baselines = [
                body for _, body in candidates if isinstance(body, EmbeddingCanaryConfigV1)
            ]
            if len(baselines) > 1:
                raise QuoteMemError("parity_mismatch",
                                    "Bootstrap has multiple frozen canary baselines")
            if baselines:
                baseline = baselines[0]
        self.check_canary_identity(baseline, current_vectors)
        digest = integrity_digest(baseline.parameters)
        space = f"{baseline.parameters.model}@{digest}@{baseline.parameters.dim}"
        key = EmbeddingBootstrapKeyV1(
            action="bootstrap", builtin_space=space, builtin_canary_digest=digest
        )
        ticket = self.run("embedding_switch", key)
        row = self.writer.get_row("run", ticket.run_id or "")
        assert row is not None
        baseline_ref = self.writer.put_logical_artifact(
            ticket, "policy_config", baseline, created_at=row["created_at"]
        )
        candidates = self.writer.find_logical_candidates(ticket.run_id or "", "run_input")
        if candidates:
            if len(candidates) != 1 or not isinstance(
                candidates[0][1], EmbeddingBootstrapRunInputV1
            ):
                self.ambiguous(ticket, "parity_mismatch")
            payload = candidates[0][1]
            assert isinstance(payload, EmbeddingBootstrapRunInputV1)
            if payload.business_key != key or payload.canary_baseline_ref != baseline_ref:
                self.ambiguous(ticket, "parity_mismatch")
        else:
            generation = self.writer.get_counter("commit_seq") + 1
            intended = self.writer.materialize_control(
                "policy:embedding_space",
                {
                    "state": "active",
                    "embedding_space": space,
                    "canary_digest": digest,
                    "canary_baseline_ref": baseline_ref,
                    "state_commit_seq": generation,
                },
                0,
                row["created_at"],
                ticket.run_id,
            )
            payload = EmbeddingBootstrapRunInputV1(
                run_kind="embedding_switch",
                business_key=key,
                action="bootstrap",
                builtin_space=space,
                builtin_canary_digest=digest,
                canary_baseline_ref=baseline_ref,
                reserved_generation=generation,
                admission_digest=row["admission_digest"],
                payload_digest=self.keyring.digest(
                    "run-input-payload", key.model_dump(mode="json")
                ),
                control_cas_intent=self.intent(intended),
            )
        self.apply(ticket, self.attach(ticket, payload))
        return baseline

    def check_canary_identity(
        self, baseline: EmbeddingCanaryConfigV1, current_vectors: EmbeddingResult
    ) -> None:
        """Bind persisted canaries to the installed request and workspace identity."""
        parameters = baseline.parameters
        if parameters.dim != self.writer.dimension or parameters.dim != current_vectors.dimensions:
            raise QuoteMemError(
                "embedding_dimension_mismatch", "Canary dimension differs from workspace"
            )
        if (
            parameters.model != EMBEDDING_MODEL
            or parameters.model != current_vectors.resolved_model
            or parameters.text_version != CANARY_TEXT_VERSION
        ):
            raise QuoteMemError(
                "binding_mismatch", "Canary model or text version differs from initialization"
            )

    def check_baseline(
        self,
        control: dict[str, Any],
        baseline: EmbeddingCanaryConfigV1,
        current_vectors: EmbeddingResult,
    ) -> None:
        self.check_canary_identity(baseline, current_vectors)
        digest = integrity_digest(baseline.parameters)
        parameters = baseline.parameters
        if (
            control["value"]["state"] != "active"
            or control["value"]["canary_digest"] != digest
            or control["value"]["embedding_space"]
            != f"{parameters.model}@{digest}@{parameters.dim}"
        ):
            raise QuoteMemError("parity_mismatch",
                                "Embedding pointer differs from its frozen baseline")


BootStepName = Literal[
    "credentials",
    "storage_owner",
    "workspace_keyring",
    "owner_storage_probe",
    "layout_probe",
    "providers",
    "hb_runtime",
    "writer_epoch",
    "catalog_repair",
    "bench_invariants",
    "embedding_canary",
    "policies",
    "forget_recovery",
    "owner_settlement",
    "privacy_recovery",
    "inputless_add_recovery",
    "cold_builds",
    "ready",
]
BOOT_STEPS: Final[tuple[BootStepName, ...]] = get_args(BootStepName)
"""The 18 boot steps, in order.

Steps 1-12 run inline in `open_initialized_workspace`, each marked with its
number. Steps 13-17 run as `RECOVERY_STEPS`. Step 18 (``ready``) belongs to
the service, after the yield: readers stay closed until it reports readiness.
"""

RecoveryStepName = Literal[
    "forget_recovery",
    "owner_settlement",
    "privacy_recovery",
    "inputless_add_recovery",
    "entity_name_audit",
    "tombstone_index_cold_build",
    "recall_snapshot_cold_build",
]
RECOVERY_STEPS: Final[tuple[RecoveryStepName, ...]] = get_args(RecoveryStepName)
"""Boot recovery steps 13-17 in order. Step 17 runs as its three parts, in this order.

Each step adds its implementation to ``_RECOVERY``:
forget recovery (13), owner settlement and cold
adoption (14), privacy recovery and admission (15), inputless Add
recovery (16), the ``entity_name`` audit, the TombstoneIndex cold
build, and the RecallSnapshot cold build.
"""


@dataclass(frozen=True, slots=True)
class BootRecovery:
    """What one recovery step runs with.

    Attributes:
        writer (Writer): The initialized sole Writer, after the policy load.
        now (int): The boot clock in epoch milliseconds.
        hook (BootstrapHook | None): The crash and evidence observer.
    """

    writer: Writer
    now: int
    hook: BootstrapHook | None


RecoveryStep = Callable[[BootRecovery], None]


def _recover_forgets(context: BootRecovery) -> None:
    """Step 13: complete every forget whose erasure fence survived the restart.

    Then end every erasure controller a crash left before its admission
    (``abandon_orphan_controllers``).

    Each fence resumes from its durable phase under its own controller
    (``recover_forgets``): no committed write repeats, and the receipt pair is
    written and verified once, so a repeated confirm returns it. A cycle the
    forget cancelled settles at step 14 and its requests are reprocessed by the
    next owner. The publisher is this step's own
    (``StoreReadModel.for_forget_recovery``) and is dropped before step 14,
    which adopts with no publisher registered.
    """
    from quotemem.core.query import ScopeV1
    from quotemem.store.forget_run import ForgetOwner, abandon_orphan_controllers, recover_forgets
    from quotemem.store.read_model import StoreReadModel

    writer, hook = context.writer, context.hook
    scopes: list[ScopeV1] = []
    for scope_id in sorted({
        str(row["key"]).split(":")[1] for row in writer.iter_rows("control")
        if str(row["key"]).startswith("erasure:")
    }):
        registration = writer.get_control(f"scope:{scope_id}")
        if registration is None:
            raise QuoteMemError("scope_denied", "An erasure fence lost its registered scope")
        scopes.append(ScopeV1(scope_kind=registration["value"]["scope_kind"], scope_id=scope_id))
    if scopes:
        publisher = StoreReadModel.for_forget_recovery(writer)
        try:
            recover_forgets(ForgetOwner(
                writer, publisher, clock=lambda: time.time_ns() // 1_000_000,
            ), scopes)
        finally:
            publisher.fail_closed()
    # A controller a crash left before its admission holds no fence.
    abandon_orphan_controllers(writer, now=time.time_ns() // 1_000_000)
    if hook is not None:
        hook("forget_recovery", "forgets_recovered", writer)


def _settle_released_owners(context: BootRecovery) -> None:
    """Step 14: settle released owners and adopt interrupted ones before any publisher exists.

    The cold classification: a proved released cycle settles, a
    preparing or held one is adopted by its owner under the new Writer epoch,
    and every other scope is quarantined.
    """
    writer, hook = context.writer, context.hook
    writer.recover_released_finalizes(clock=lambda: time.time_ns() // 1_000_000)
    if hook is not None:
        for row in writer.iter_rows("control"):
            if row["key"].startswith("scope_quarantine:"):
                scope_id = row["key"].removeprefix("scope_quarantine:")
                hook("finalize_recovery", f"quarantined:{scope_id}", writer)
        hook("finalize_recovery", "released_owners_settled", writer)


def _recover_privacy(context: BootRecovery) -> None:
    """Step 15: reserve privacy parents and resume proven cleanup plans.

    Inputless split-secret owners still need their matching client boundary,
    so they are only reserved here.
    """
    writer, hook = context.writer, context.hook
    pending = writer.recover_privacy_admissions()
    if hook is not None:
        hook("privacy_recovery", "admissions_recovered", writer)
    resumable = tuple(owner for owner in pending if owner.state != "awaiting_client_replay")
    if resumable:
        _resume_privacy(writer, resumable, hook)
    if hook is not None:
        hook("privacy_recovery", "attached_plans_recovered", writer)


def _recover_inputless_adds(context: BootRecovery) -> None:
    """Step 16: recover inputless Adds and void inputs past their deadline."""
    context.writer.recover_inputless_adds(context.now)


def _cold_build_recall_snapshot(context: BootRecovery) -> None:
    """Step 17: build the RecallSnapshot cold and every open scope's corpora.

    ``StoreReadModel.build_cold_root`` projects every committed row without
    registering a publisher, and ``cold_build`` builds the memory, entity and
    Evidence corpora of each registered scope whose fence admits reads. A
    projection or corpus that cannot be built aborts startup before readiness.
    The corpora are dropped. The root is retained for the
    service's publisher, which adopts it after the job binding when no row
    changed since and otherwise builds its own: a large store is
    then projected once per boot, not twice. The step, a recorded
    contributor edit.
    """
    from quotemem.core.recall_corpora import cold_build
    from quotemem.store.read_model import StoreReadModel

    writer, hook = context.writer, context.hook
    cold_build(StoreReadModel.build_cold_root(writer, retain=True), [
        row["key"].removeprefix("scope:") for row in writer.iter_rows("control")
        if row["key"].startswith("scope:")
    ])
    if hook is not None:
        hook("recall_snapshot_cold_build", "corpora_verified", writer)


def _cold_build_tombstone_index(context: BootRecovery) -> None:
    """Verify every registered scope's private index, then discard it.

    Startup holds exclusive storage ownership. Runtime owners load their own
    guarded index; no plaintext is retained by bootstrap or the read model.
    """
    from quotemem.core.query import ScopeV1
    from quotemem.ingestion.tombstone_index import TombstoneIndex

    writer = context.writer

    def check() -> None:
        with writer.mutex:
            writer.validate_storage_owner()

    for row in writer.iter_rows("control"):
        if not row["key"].startswith("scope:"):
            continue
        scope = ScopeV1(scope_kind=row["value"]["scope_kind"],
                        scope_id=row["key"].removeprefix("scope:"))
        counter = f"tombstone_seq:{scope.scope_id}"
        index = TombstoneIndex.load(
            writer.indexed_reader(), scope, sequence=partial(writer.get_counter, counter),
            check=check,
        )
        index.close()
    if context.hook is not None:
        context.hook("tombstone_index_cold_build", "tombstones_verified", writer)


def _audit_entity_names(context: BootRecovery) -> None:
    """Step 17, first part: rebuild every ``entity_name`` row from the heads and compare.

    A stored name the derivation lacks, a derived name that is not stored, or
    a duplicate aborts startup with ``parity_mismatch`` before any reader opens.
    """
    from quotemem.store.entity_ops import audit_entity_names

    audit_entity_names(context.writer)
    if context.hook is not None:
        context.hook("entity_name_audit", "names_verified", context.writer)


_RECOVERY: Mapping[RecoveryStepName, RecoveryStep] = MappingProxyType({
    "forget_recovery": _recover_forgets,
    "owner_settlement": _settle_released_owners,
    "privacy_recovery": _recover_privacy,
    "inputless_add_recovery": _recover_inputless_adds,
    "entity_name_audit": _audit_entity_names,
    "tombstone_index_cold_build": _cold_build_tombstone_index,
    "recall_snapshot_cold_build": _cold_build_recall_snapshot,
})
"""The recovery steps that have landed. A step whose owner has not landed is absent:
there is no stub and no no-op in its place."""


def _run_recovery(context: BootRecovery) -> None:
    """Run every landed recovery step in `RECOVERY_STEPS` order."""
    for name in RECOVERY_STEPS:
        step = _RECOVERY.get(name)
        if step is not None:
            step(context)


def _resume_privacy(
    writer: Writer, pending: tuple[PendingPrivacy, ...],
    hook: BootstrapHook | None,
) -> None:
    """Resume selected cleanup authority before normal consumers obtain a publisher."""
    from quotemem.ingestion.privacy_cleanup import PrivacyCleanup
    from quotemem.store.read_model import StoreReadModel

    publisher = StoreReadModel(writer)
    active = ""

    def checkpoint(event: str) -> None:
        if hook is not None:
            hook(active, event, writer)

    cleanup = PrivacyCleanup(
        writer,
        publisher,
        clock=lambda: time.time_ns() // 1_000_000,
        checkpoint=checkpoint,
    )
    try:
        for owner in pending:
            active = owner.owner_run_id
            if owner.state == "resume":
                cleanup.resume(active)
    finally:
        publisher.fail_closed()
    # Returning drops both strong owner references before the normal service
    # constructs its own publisher. No startup guard is exported to that service.


@contextmanager
def open_initialized_workspace(
    root: Path,
    config: ProviderConfig,
    environment: Mapping[str, str],
    *,
    client: httpx.Client | None = None,
    async_client_factory: AsyncClientFactory | None = None,
    hook: BootstrapHook | None = None,
    owner: StorageOwner | None = None,
    resources: WorkspaceBootstrapResources | None = None,
) -> Iterator[InitializedWorkspace]:
    """Initialize a workspace and complete proven privacy recovery before yielding.

    Credentials are checked before filesystem effects. Injecting an HTTP client
    permits frozen boundary tests; configuration never enables fabricated vectors.

    Args:
        root: The private explicit workspace directory.
        config: The pinned nonsecret provider and workspace profile configuration.
        environment: Credential values resolved before any persistent side effect.
        client: Optional transport for provider boundaries, including offline fixtures.
        async_client_factory: Explicit asynchronous counterpart for V2 bounded calls.
            Without an injected client, normal HTTPX clients are used. An injected
            synchronous client never silently falls back to live asynchronous HTTP.
        hook: Optional crash/evidence observer for bootstrap and privacy checkpoints.
        owner: Optional strict process owner supplied by the service. It replaces
            the legacy lock and remains held until the runtime has shut down.
        resources: Optional caller-retained bootstrap resources for explicit
            shutdown retry, including initialization failures before the yield.

    Yields:
        The sole initialized Writer after selected cleanup plans finish, including
        inputless denial reconstructed within its registered maintenance borrow.
        Inputless split-secret scopes still await the matching client boundary.

    Raises:
        QuoteMemError: If workspace ownership, startup binding, or recovery proof fails.
        WorkspaceCleanupPending: If cleanup fails while retaining storage and
            process ownership. Its resources remain available for a close retry.
        OSError: If the private workspace cannot be safely opened or synchronized.
    """
    from quotemem.core.platform_files import validate_private_path
    from quotemem.ingestion.nomination import NominationPolicyV1
    from quotemem.store.offline import probe_workspace_layout
    from quotemem.store.schemas import schema_fingerprint

    # Boot steps 1-12 in order (`BOOT_STEPS`), then the recovery steps 13-17.
    # 1. Validate the endpoint and load credentials, before any filesystem effect.
    config.validate_inference_endpoint()
    credentials = load_credentials(config, environment)
    root = root.parent.resolve() / root.name
    acquired = resources if resources is not None else WorkspaceBootstrapResources()
    if acquired._used or acquired._closed:  # pyright: ignore[reportPrivateUsage]
        raise RuntimeError("Workspace bootstrap resources cannot be reused")
    acquired._used = True  # pyright: ignore[reportPrivateUsage]
    stack = acquired._stack  # pyright: ignore[reportPrivateUsage]
    try:
        # 2. Take the exclusive storage owner.
        if owner is None:
            stack.enter_context(_exclusive_owner(root))
        else:
            if owner.root.parent.resolve() / owner.root.name != root:
                raise QuoteMemError("workspace_binding_mismatch", "Storage owner root differs")
            ensure_private_directory(root)
            if stack.enter_context(owner) is not owner or owner.root != root:
                raise QuoteMemError("workspace_binding_mismatch", "Storage owner root differs")
        # 3. Load workspace.json and the keyring.
        stored, keyring = _configuration(root, config)
        # The previous lifetime's Catalog proof and Catalog ledger are taken and
        # retired before any storage is probed or opened (step 9;
        # ``quotemem.store.catalog_proof``, ``quotemem.store.catalog_ledger``).
        proof = take_catalog_proof(root, stored.workspace_id)
        ledger = take_catalog_ledger(root, stored.workspace_id)
        # 4. Probe owner storage.
        if owner is not None:
            owner.probe_storage()
            owner.prepare_databases()
        # 5. The offline layout and schema-fingerprint probe, before HB opens.
        probe_workspace_layout(
            root, workspace_id=stored.workspace_id, db_uuid=stored.db_uuid,
            tenant_id=stored.tenant_id, dimension=config.embedding_dimensions,
            validate_owner=(owner.validate if owner is not None
                            else lambda: validate_private_path(root, directory=True)),
        )
        # 6. Create the providers.
        http = client if client is not None else stack.enter_context(httpx.Client())
        providers = create_providers(
            config, credentials, http,
            async_client_factory=(
                async_client_factory if async_client_factory is not None
                else httpx.AsyncClient if client is None else None
            ),
        )
        # 7. Open the HB runtime.
        runtime = open_runtime(
            root, workspace_id=stored.workspace_id, owner=owner,
            retain_cleanup=acquired._retain_storage,  # pyright: ignore[reportPrivateUsage]
        )
        # 8. Create the Writer and run initialize_base (writer-epoch CAS).
        writer = Writer(
            runtime,
            stored.tenant_id,
            config.embedding_dimensions,
            keyring.digest,
            provider_config=config,
            bind_content_array=keyring.array_binding,
        )
        now = time.time_ns() // 1_000_000
        writer.initialize_base(
            stored.workspace_id,
            stored.db_uuid,
            schema_fingerprint(
                config.embedding_dimensions,
            ),
            now,
        )
        # 9. Repair the catalog, unless the previous lifetime's clean close or its
        # clean ledger proved it; from here this lifetime keeps its own ledger.
        writer.recover_catalog(proof=proof, ledger=ledger)
        stack.callback(writer.close_catalog_ledger)
        writer.start_catalog_ledger(root, stored.workspace_id)
        # 10. Check the bench-profile invariants.
        if config.profile == "bench":
            writer.read_bench_profile()
        if hook is not None:
            hook("catalog_recovery", "repaired", writer)
        # 11. Run the embedding canary outside the Writer mutex and compare it with
        # the durable baseline, on every start including a complete reopen.
        vectors = providers.embedding.embed(
            initialization_canary_request(config.embedding_dimensions)
        )
        bootstrap = _Bootstrap(writer, keyring, now, hook)
        baseline = bootstrap.embedding(vectors)
        saved_vectors = [
            list(struct.unpack(f"<{baseline.parameters.dim}f", base64.b64decode(value)))
            for value in baseline.parameters.vectors
        ]
        drift, deviation = compare_canaries(saved_vectors, vectors.vectors)
        writer.embedding_write_state = drift
        embedding = writer.get_control("policy:embedding_space")
        if embedding is None or bootstrap.barrier:
            raise QuoteMemError("service_unhealthy", "Bootstrap embedding projection is incomplete")
        bootstrap.check_baseline(embedding, baseline, vectors)
        # 12. Load the recall and extraction policies. The extraction policy pins the
        # frozen nomination policy by its digest.
        profile = "bench" if config.profile == "bench" else "product"
        bootstrap.policy("recall", build_recall_policy(profile))
        bootstrap.policy("extraction", build_extraction_policy(
            profile, nomination_policy_digest=NominationPolicyV1().digest,
            call_budget_trial=config.extraction_call_budget,
            judge_mode=config.judge_mode, judge_gate=config.judge_gate,
            chat_forget=config.chat_forget,
            extraction_supplement=config.extraction_supplement,
            self_identity=config.self_identity,
            quote_grounding=config.quote_grounding,
            prompt_template_digest=config.extraction_template,
        ))
        # A configured fast tier releases its own extraction policy beside the default
        # one; each finalize owner freezes one of the two (``ingestion/finalize_freeze.py``).
        if config.add_tiers is not None:
            fast = config.add_tiers.fast_mode
            bootstrap.policy("extraction", build_extraction_policy(
                profile, nomination_policy_digest=NominationPolicyV1().digest,
                call_budget_trial=(
                    config.extraction_call_budget_trial or EXTRACTION_CALL_BUDGETS[fast]
                ),
                judge_mode=JUDGE_MODES[fast], judge_gate=JUDGE_GATES[fast],
                chat_forget=CHAT_FORGET_RULES[fast],
                extraction_supplement=EXTRACTION_SUPPLEMENTS[fast],
                self_identity=SELF_IDENTITY_RULES[fast],
                quote_grounding=QUOTE_GROUNDING[fast],
                prompt_template_digest=EXTRACTION_TEMPLATE_DIGESTS[fast],
            ), tier="fast")
        recall = writer.get_control("policy:recall")
        if recall is None:
            raise QuoteMemError("service_unhealthy", "Bootstrap policy projection is incomplete")
        # 13-17. The recovery steps, before any ordinary publisher can exist.
        _run_recovery(BootRecovery(writer, now, hook))
        token = embedding_state_token(ControlRowV1.model_validate(embedding))
        projection = BootstrapProjection(
            writer.get_counter("commit_seq"), ArtifactId(recall["value"]["current_release_ref"]),
            token
        )
        if owner is not None:
            owner.validate()
        # Whoever closes this storage records the lifetime's Catalog proof once it
        # has closed, while this process still owns the workspace.
        acquired._after_storage_closed(  # pyright: ignore[reportPrivateUsage]
            partial(_leave_catalog_proof, writer, root, stored.workspace_id),
        )
        yield InitializedWorkspace(
            runtime, writer, keyring, providers, projection, drift, deviation
        )
    finally:
        acquired.close()


def _leave_catalog_proof(writer: Writer, root: Path, workspace_id: str) -> None:
    """Seal and record a closed lifetime's Catalog proof (``quotemem.store.catalog_proof``).

    Runs once storage has acknowledged closure, so no mutation can follow the seal;
    a lifetime whose Catalog is unproven records nothing.
    """
    sealed = writer.seal_catalog_proof()
    writer.close_catalog_ledger()
    if sealed is not None and writer.is_catalog_proof_intact():
        leave_catalog_proof(root, workspace_id, sealed)


def initialization_canary_request(dimension: Literal[1024, 2048]) -> EmbeddingRequest:
    """Freeze the complete startup request independently of fixture or transport behavior."""
    return EmbeddingRequest(
        dimensions=dimension,
        texts=list(CANARY_TEXTS),
        timeout_ms=DEFAULT_PROVIDER_TIMEOUT_MS,
        retry_policy=default_retry_policy(),
    )
