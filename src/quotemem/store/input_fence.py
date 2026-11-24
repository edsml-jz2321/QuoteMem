"""The Add owner's finalize fences for actual scope input mutations."""

from collections.abc import Mapping
from typing import Any

from quotemem.core.errors import QuoteMemError


def _require_released_owner(
    value: Mapping[str, Any],
    run: Mapping[str, Any],
    owner: Mapping[str, Any] | None,
) -> None:
    """Keep the released interval closed until its exact owner has settled."""
    if owner is None:
        raise QuoteMemError("finalize_pending", "Released finalize awaits owner settlement")
    if (
        owner.get("object_id") != value["owner_run_id"]
        or run.get("scope_id") is None
        or any(owner.get(key) != run.get(key) for key in ("scope_kind", "scope_id"))
        or owner.get("run_kind") != "finalize"
        or owner.get("commit_mode") != "orchestrator"
        or owner.get("target_kind") != "scope"
        or owner.get("target_id") != run["scope_id"]
        or owner.get("parent_run_id") is not None
        or type(owner.get("attempt")) is not int
        or owner["attempt"] < 0
        or owner.get("run_generation") != 0
    ):
        raise QuoteMemError("binding_mismatch", "Released finalize changed its owner binding")
    lease_epoch = value.get("writer_epoch")
    owner_epoch = owner.get("writer_epoch")
    run_epoch = run.get("writer_epoch")
    if not (
        type(lease_epoch) is int and type(owner_epoch) is int and type(run_epoch) is int
        and 0 < lease_epoch <= owner_epoch <= run_epoch
    ):
        raise QuoteMemError("binding_mismatch", "Released finalize has impossible Writer epochs")
    if owner.get("status") in {"queued", "running", "manual_attention"}:
        raise QuoteMemError("finalize_pending", "Released finalize awaits owner settlement")
    if (
        owner.get("status") != "succeeded"
        or owner.get("stage") != "release"
        or any(owner.get(key) is None for key in (
            "input_ref", "result_ref", "started_at", "finished_at",
        ))
        or any(owner.get(key) is not None for key in (
            "commit_marker", "commit_seq", "failure_code",
        ))
    ):
        raise QuoteMemError("binding_mismatch", "Released finalize lacks its terminal release")


def require_input_mutation_allowed(
    lease: Mapping[str, Any] | None,
    run: Mapping[str, Any],
    *,
    execution_root: Mapping[str, Any] | None = None,
    finalize_owner: Mapping[str, Any] | None = None,
) -> None:
    """Check selected rows under the existing Writer mutex before input changes.

    The Writer supplies its actual same-scope lease, executing Run and released
    owner row. Its admission owner validates canonical cycle identity; only its
    validated ancestry walk may supply execution_root. This check performs no
    reads and grants no mutation authority. Held refuses every input mutation
    except the held owner's own execution: the finalize owner and its publish
    and evaluate children, whose ancestry root it is. Preparing passes this
    check; Add ordering (``Writer.commit_add``) keeps its own preparing fence,
    so no request is ordered into an owner's frozen prefix after that owner's
    admission. Release permits input only after terminal owner settlement,
    including historical owners whose Writer epoch precedes the current
    executing Run. A recovered owner can have a positive attempt; its settled
    release, not an initial-attempt restriction, ends the input fence. Recovery
    history remains with its existing admission and publication validators.

    Raises:
        QuoteMemError: With finalize_pending for held or unsettled released fences,
            or binding_mismatch for a substituted released owner or invalid
            terminal release binding.
    """
    if lease is None:
        return
    value = lease["value"]
    if value["state"] == "released":
        _require_released_owner(value, run, finalize_owner)
        return
    if value["state"] != "held":
        return
    if (
        execution_root is not None
        and execution_root["run_kind"] == "finalize"
        and execution_root["object_id"] == value["owner_run_id"]
        and execution_root["writer_epoch"] == run["writer_epoch"] == value["writer_epoch"]
        and run.get("scope_id") is not None
        and all(
            execution_root.get(key) == run.get(key)
            for key in ("tenant_id", "scope_kind", "scope_id")
        )
    ):
        return
    raise QuoteMemError("finalize_pending", "Held finalize ownership blocks this input mutation")
