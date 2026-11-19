"""The released-history proof (C.13 R13; C.2a): what lets a new cycle or a cold root use it.

S0b failed every released predecessor closed. A released cycle is a legal end
state only when its own durable records agree: the lease names its succeeded
owner at the release stage, the owner's admission and selected input are this
scope's exact cycle, the ``finalized:{s}`` marker was written by that owner for
that cycle and input and is the marker its release projected into
``completed_prefix:{s}`` (so its ``finalized_scope_seq`` is the one the release
recorded, never a later sequence), and every child the owner ran has ended.
Nothing here writes; admission (``owner_lease.admit_finalize``), the cold root
(``StoreReadModel._rebuild``) and boot step 14 consume the proof.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from aml.core.admission import admission_identity, finalize_admission
from aml.core.controls import (
    CompletedPrefixV2,
    ControlRowV1,
    FinalizedValueV1,
    ReleasedFinalizeLeaseV1,
)
from aml.core.errors import AMLError
from aml.core.query import ScopeV1
from aml.core.run_inputs import FinalizeOwnerRunInputV1
from aml.store.input_fence import (
    _require_released_owner,  # pyright: ignore[reportPrivateUsage]
)
from aml.store.writer import Writer

_TERMINAL = frozenset({"succeeded", "failed"})


def prove_released_history(
    writer: Writer, scope: ScopeV1, lease: ControlRowV1, *,
    owner: Mapping[str, Any] | None = None,
) -> FinalizedValueV1:
    """Prove one released cycle from its own records; return its finalized marker.

    Args:
        writer: The sole Writer (the caller holds its mutex or write scope).
        scope: The exact registered scope.
        lease: The scope's current ``lease:finalize:{s}`` Control.
        owner: The owner Run row to prove instead of the stored one: the row
            release's last transaction writes, for a release that died before
            it (``owner_lease``, boot step 14). That transaction also writes the
            ``completed_prefix:{s}`` projection, so only the stored owner's
            history is compared with it.

    Returns:
        The cycle's ``finalized:{s}`` value. Its ``finalized_scope_seq`` may lag
        the scope's sequence: later receptions leave the scope dirty, not unproved.

    Raises:
        AMLError: ``binding_mismatch`` for any record that disagrees,
            ``finalize_pending`` while the owner or one of its children still runs.
    """
    value = lease.value
    if not isinstance(value, ReleasedFinalizeLeaseV1) or lease.updated_by_run_id != (
        value.owner_run_id
    ):
        raise AMLError("binding_mismatch", "Released history requires its released lease")
    settled = owner is None
    if owner is None:
        owner = writer.get_row("run", value.owner_run_id)
    current = writer.get_control("lease:writer")
    if current is None:
        raise AMLError("service_unhealthy", "Released history requires the Writer lease")
    _require_released_owner(value.model_dump(mode="json"), {
        "scope_kind": scope.scope_kind, "scope_id": scope.scope_id,
        "writer_epoch": current["value"]["writer_epoch"],
    }, owner)
    assert owner is not None
    family, digest, payload = admission_identity(finalize_admission(
        principal_id=owner["principal_id"], scope=scope, cycle=value.lease_generation,
    ), writer.bind_content)
    if (owner["idempotency_key"], owner["admission_digest"]) != (family, digest):
        raise AMLError("binding_mismatch", "Released owner has another admission")
    envelope = writer.get_row("artifact", owner["input_ref"])
    if envelope is None or (envelope["created_by_run_id"], envelope["scope_id"]) != (
        owner["object_id"], scope.scope_id,
    ):
        raise AMLError("binding_mismatch", "Released input has another creator or scope")
    selected = writer.read_logical(owner["input_ref"], "run_input")
    marker = writer.get_control(f"finalized:{scope.scope_id}")
    if marker is None:
        raise AMLError("binding_mismatch", "Released cycle lacks its finalized marker")
    control = ControlRowV1.model_validate(marker)
    finalized = control.value
    if (
        not isinstance(selected, FinalizeOwnerRunInputV1)
        or (selected.business_key.scope_id, selected.business_key.finalize_cycle)
        != (scope.scope_id, value.lease_generation)
        or (selected.admission_digest, selected.payload_digest) != (digest, payload)
        or not isinstance(finalized, FinalizedValueV1)
        or control.updated_by_run_id != owner["object_id"]
        or (finalized.run_id, finalized.lease_generation)
        != (owner["object_id"], value.lease_generation)
        or finalized.input_heads_digest != selected.input_heads_digest
        or finalized.finalized_scope_seq > writer.counter(f"scope_seq:{scope.scope_id}")
    ):
        raise AMLError("binding_mismatch", "Released marker differs from its owner's input")
    if settled:
        # Release's last transaction projected the marker it wrote, with the
        # released lease, into the completed prefix (C.2a): a marker changed
        # after it, e.g. a sequence promoted after a later Add, is not its own.
        raw = writer.get_control(f"completed_prefix:{scope.scope_id}")
        prefix = None if raw is None else ControlRowV1.model_validate(raw)
        if prefix is None or not isinstance(prefix.value, CompletedPrefixV2) or (
            prefix.updated_by_run_id != owner["object_id"]
            or prefix.value.marker != finalized or prefix.value.lease != value
        ):
            raise AMLError("binding_mismatch", "Released marker is not its release's own")
    after: str | None = None
    reader = writer.indexed_reader()
    while True:
        children = reader.run_children(owner["object_id"], after=after)
        for child in children:
            if child["status"] not in _TERMINAL:
                raise AMLError("finalize_pending", "A released owner's child still runs")
        if len(children) < 128:
            return finalized
        after = children[-1]["object_id"]
