"""Privacy-erase orchestration through the sole Writer and read model (D.7).

Split-secret cleanup is privacy erase's one product producer; source expiry and
consent withdrawal are selected only by test fixtures (INV P13, Open item O10).
This consumer has no provider dependency. Durable plans contain only locators,
keyed states, and closed transforms; current incoming boundary bytes stay out.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import cast

from aml.core.admission import admission_identity, privacy_erase_admission
from aml.core.controls import maintenance_fence_key
from aml.core.errors import AMLError
from aml.core.identity import ArtifactId, IntegrityDigest
from aml.core.ingestion_cleanup import (
    RUN_INPUT_OF_TRIGGER,
    CleanupInput,
    validate_cleanup_run_input,
)
from aml.core.privacy_persistence import (
    ZERO_COUNTS,
    IngestionErasurePlanV1,
    PrivacyEraseInputV1,
    RedactionReceiptV1,
)
from aml.core.query import ScopeV1
from aml.ingestion.authorization import check_add_grant
from aml.ingestion.tombstone_index import TombstoneIndex
from aml.providers.auth import PrincipalGrant
from aml.store.domain import DOMAIN_ADAPTERS
from aml.store.forget_final_scan import privacy_quiescence
from aml.store.privacy_plan import (
    PlanBlockBuilder,
    load_privacy_plan,
    prepare_privacy_effects,
    privacy_effect_counts,
)
from aml.store.read_model import StoreReadModel
from aml.store.writer import (
    MAX_TX_ROWS,
    CommittedRows,
    RowDelete,
    RowWrite,
    Writer,
    WriteTicket,
)


class PrivacyCleanup:
    """Own the privacy-erase use case: admit, plan, apply, verify and receipt one erase.

    Args:
        writer: The process's sole durable mutation authority.
        publisher: Its sole runtime RecallSnapshot owner.
        clock: Trusted current time in integer milliseconds.
        checkpoint: Optional crash observer; never a provider or scheduler.
        tombstone_indexes: The Add owners' loaded TombstoneIndexes of the scope;
            the shared quiescence closes those an erase batch made stale.
    """

    def __init__(
        self,
        writer: Writer,
        publisher: StoreReadModel,
        *,
        clock: Callable[[], int],
        checkpoint: Callable[[str], None] | None = None,
        tombstone_indexes: Iterable[TombstoneIndex] = (),
    ) -> None:
        self.writer, self.publisher = writer, publisher
        self.clock, self.checkpoint = clock, checkpoint
        self.tombstone_indexes = tuple(tombstone_indexes)

    def _ensure_publisher(self) -> None:
        """Repair uncertain Catalog/root acknowledgement before any success response."""
        try:
            _ = self.publisher.current
        except AMLError as error:
            if error.code != "service_unhealthy":
                raise
            self.writer.recover_catalog()
            self.publisher.recover_privacy(self.writer)

    def _receipt(self, run: dict[str, object]) -> ArtifactId:
        """Read only the actual terminal owner's selected immutable receipt."""
        from aml.core.artifacts import ReceiptResultIndexV1

        if run["status"] != "succeeded" or not isinstance(run["result_ref"], str):
            raise AMLError("parity_mismatch", "Cleanup has no terminal selected receipt")
        result_ref = run["result_ref"]
        envelope = self.writer.get_row("artifact", result_ref)
        result = self.writer.read_logical(result_ref, "result_index")
        if (
            envelope is None or envelope["created_by_run_id"] != run["object_id"]
            or not isinstance(result, ReceiptResultIndexV1)
        ):
            raise AMLError("binding_mismatch", "Cleanup result has another owner or variant")
        envelope = self.writer.get_row("artifact", result.receipt_ref)
        receipt = self.writer.read_logical(result.receipt_ref, "receipt")
        body = validate_cleanup_run_input(self.writer.read_logical(
            cast(str, run["input_ref"]), "run_input"
        ))
        if envelope is None or envelope["created_by_run_id"] != run["object_id"]:
            raise AMLError("binding_mismatch", "Cleanup receipt has another owner")
        proof = load_privacy_plan(
            self.writer, body.plan_ref, owner_run_id=cast(str, run["object_id"]),
            supplied=body.business_key, expected_plan_digest=None,
        )
        if (
            not isinstance(receipt, RedactionReceiptV1)
            or receipt.reason_code != body.business_key.reason_code
        ):
            raise AMLError("binding_mismatch", "Cleanup receipt has another closed variant")
        if receipt.counts != privacy_effect_counts(self.writer, proof):
            raise AMLError("parity_mismatch", "Cleanup receipt counts differ from selected plan")
        if receipt.affected_digests != sorted(
            target.expected_row_digest for target in body.business_key.affected
        ):
            raise AMLError("binding_mismatch", "Erase receipt changed its targets")
        return result.receipt_ref

    def execute(
        self,
        scope: ScopeV1,
        supplied: PrivacyEraseInputV1,
        grant: PrincipalGrant,
        *,
        parent_run_id: str | None = None,
    ) -> ArtifactId:
        """Admit, freeze, apply and verify one source-bound cleanup before returning.

        Args:
            scope: Already authorized exact scope, without ancestor expansion.
            supplied: Content-free selection built from current old Evidence states.
            grant: Fresh principal grant; admission checks it again under the mutex.
            parent_run_id: Required only for the already-admitted ingestion child.

        Returns:
            Its verified redaction receipt, after complete clean-root publication.
            Executing a completed selection again returns the same receipt and
            writes nothing.

        Raises:
            AMLError: If authority, source state, capacity, or cleanup verification fails.
                A durable fence remains effective until explicit recovery succeeds.
        """
        self._ensure_publisher()
        completed = self._completed(scope, supplied, grant, parent_run_id)
        if completed is not None:
            return self._receipt(completed)
        ticket = self.writer.admit_privacy_erase(
            scope, supplied, grant, now=self.clock(), parent_run_id=parent_run_id
        )
        run_id = cast(str, ticket.run_id)
        try:
            run = self.writer.get_row("run", run_id)
            if run is None:
                raise AMLError("service_unhealthy", "Cleanup admission lost its owner")
            if run["input_ref"] is None:
                self._prepare(ticket, supplied, scope, run)
            return self.resume(run_id)
        except BaseException:
            # Includes an uncertain main commit or Catalog acknowledgement. No
            # lock-free reader may keep using an unpublished prior generation.
            self.publisher.fail_closed(discard_finalize_baselines=True)
            raise

    def _completed(
        self,
        scope: ScopeV1,
        supplied: PrivacyEraseInputV1,
        grant: PrincipalGrant,
        parent_run_id: str | None,
    ) -> dict[str, object] | None:
        """Find the succeeded owner of exactly this selection, after reauthorizing it."""
        check_add_grant(grant, scope, self.clock())
        admission = privacy_erase_admission(
            scope, supplied, grant.principal_id, self.writer.bind_content,
        )
        family, digest, _ = admission_identity(admission, self.writer.bind_content)
        existing = self.writer.indexed_reader().latest_run("retention", family)
        if existing is None or existing["status"] != "succeeded":
            return None
        if (existing["admission_digest"], existing["parent_run_id"]) != (digest, parent_run_id):
            raise AMLError("operation_conflict", "Privacy family binds another input")
        return existing

    def _prepare(
        self,
        ticket: WriteTicket,
        supplied: CleanupInput,
        scope: ScopeV1,
        run: dict[str, object],
    ) -> None:
        writer = self.writer
        run_id = cast(str, ticket.run_id)
        frozen_time = cast(int, run["created_at"])
        targets = tuple(supplied.affected)
        if not targets:
            raise AMLError("input_required", "Cleanup preparation requires its exact targets")
        key = maintenance_fence_key(
            scope.scope_id, targets[0].evidence_id, writer.bind_content
        )
        # Count the final attached Run geometry before its content-addressed
        # input exists. This placeholder is only a StoragePlan sizing value.
        budget_run = {
            **run,
            "input_ref": ArtifactId("art_" + "A" * 43),
            "status": "running",
            "stage": "compensate",
            "started_at": frozen_time,
            "updated_at": frozen_time,
        }
        progress = writer.materialize_control(
            key,
            {
                "state": "active",
                "fence_generation": 1,
                "owner_run_id": run_id,
                "evidence_id": targets[0].evidence_id,
                "reason_code": "privacy_erase",
                "phase": "prepared",
                "plan_digest": IntegrityDigest("0" * 64),
                "cursor": 0,
                "affected_counts": ZERO_COUNTS.model_dump(mode="json"),
                "erase_epoch_bumps": 0,
            },
            0,
            frozen_time,
            run_id,
        )
        owner = [
            RowWrite("run", DOMAIN_ADAPTERS["run"].validate_python(budget_run)),
            RowWrite("control", progress),
        ]

        def rowops(
            rows: Sequence[RowWrite], deletes: Sequence[RowDelete], bumps: frozenset[str],
        ) -> int:
            names = ["commit_seq", f"scope_seq:{scope.scope_id}"]
            names += [f"{name}:{scope.scope_id}" for name in sorted(bumps)]
            counters = [
                RowWrite("control", writer.materialize_control(
                    name, {"seq": writer.counter(name) + 1}, 1, frozen_time, run_id,
                ))
                for name in names
            ]
            return writer.privacy_rowops(rows, deletes=deletes, overhead=[*owner, *counters])

        proof = prepare_privacy_effects(
            writer,
            supplied,
            owner_run_id=run_id,
            effect_time=frozen_time,
            rowops=rowops,
        )

        def fits(block: IngestionErasurePlanV1) -> bool:
            try:
                return (
                    writer.logical_artifact_rowops(
                        ticket, "erasure_plan", block, created_at=frozen_time
                    )
                    <= MAX_TX_ROWS
                )
            except AMLError as error:
                if error.code == "artifact_too_large":
                    return False
                raise

        builder = PlanBlockBuilder(proof, fits=fits)
        previous: ArtifactId | None = None
        while not builder.complete:
            block = builder.next_block(previous)
            previous = writer.put_logical_artifact(
                ticket, "erasure_plan", block, security_level="restricted", created_at=frozen_time
            )
            if self.checkpoint is not None:
                self.checkpoint("cleanup_plan_block_prewritten")
        if previous is None:
            raise AMLError("parity_mismatch", "Cleanup has no complete plan head")
        body = RUN_INPUT_OF_TRIGGER[supplied.trigger].model_validate(
            {
                "run_kind": "retention",
                "retention_role": supplied.retention_role,
                "trigger": supplied.trigger,
                "business_key": supplied,
                "plan_ref": previous,
                "admission_digest": run["admission_digest"],
                "target_locator": run["target_id"],
                "payload_digest": writer.bind_content(
                    "run-input-payload", supplied.model_dump(mode="json")
                ),
            }
        )
        input_ref = writer.put_logical_artifact(
            ticket, "run_input", body, security_level="restricted", created_at=frozen_time
        )
        if self.checkpoint is not None:
            self.checkpoint("cleanup_input_prewritten")
        writer.start_privacy_erase(
            ticket,
            input_ref,
            proof.plan_digest,
            now=self.clock(),
            publish=lambda generation: self.publisher.publish_privacy(
                writer, scope, None, generation, maintenance=True
            ),
            checkpoint=self.checkpoint,
        )

    def resume(self, run_id: str) -> ArtifactId:
        """Resume only the attached current-epoch plan; never infer new targets or call providers.

        Args:
            run_id: The exact existing cleanup owner. Startup adoption is separate.

        Returns:
            Its verified receipt after the unique maintenance fence is safely cleared.

        Raises:
            AMLError: On missing input, changed ownership, inconsistent progress, or residuals.
        """
        writer = self.writer
        try:
            self._ensure_publisher()
            run = writer.get_row("run", run_id)
            if run is None or run["input_ref"] is None:
                raise AMLError("input_required", "Cleanup resume requires its durable selection")
            if run["status"] == "succeeded":
                return self._receipt(run)
            ticket = writer.ticket(run_id)
            body = validate_cleanup_run_input(writer.read_logical(run["input_ref"], "run_input"))
            scope = ScopeV1.model_validate(
                {"scope_kind": run["scope_kind"], "scope_id": run["scope_id"]}
            )
            key = maintenance_fence_key(
                scope.scope_id, body.business_key.affected[0].evidence_id, writer.bind_content,
            )

            def publish(committed: CommittedRows) -> None:
                self.publisher.publish_privacy(
                    writer, scope, committed, committed.commit_seq, maintenance=True,
                )

            def barrier() -> None:
                fence = writer.get_control(key)
                if fence is None or fence["value"].get("owner_run_id") != run_id:
                    raise AMLError("reception_stale", "Privacy quiescence lost its erase fence")

            # D.6 (3): every erase commit is followed by the shared quiescence,
            # so no reader or cache keeps a batch's erased content past it. A
            # resume whose batches all committed before the interruption still
            # quiesces once, since the crash may have preceded that quiescence.
            applied = False
            while writer.apply_privacy_erase(
                ticket, key, now=self.clock(), publish=publish, checkpoint=self.checkpoint
            ):
                applied = True
                privacy_quiescence(
                    writer, self.publisher, scope, barrier=barrier,
                    tombstone_indexes=self.tombstone_indexes,
                )
            if not applied:
                privacy_quiescence(
                    writer, self.publisher, scope, barrier=barrier,
                    tombstone_indexes=self.tombstone_indexes,
                )

            return writer.finish_privacy_erase(
                ticket,
                key,
                now=self.clock(),
                publisher=self.publisher,
                checkpoint=self.checkpoint,
            )
        except BaseException:
            self.publisher.fail_closed(discard_finalize_baselines=True)
            raise
