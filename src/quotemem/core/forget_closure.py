"""Bounded reverse-reference closure for selective hard-forget.

The caller supplies a proved inventory from the existing schema registry. This
pure planner cannot authorize reads or writes, invent source ownership, or treat
an incomplete inventory as a completed erasure. Compensation and source-message
selection precede this step; their surviving rows replace their original edges.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

from quotemem.core.errors import QuoteMemError
from quotemem.store.registry import Reference

RowKey = tuple[str, str]
ScopeKey = tuple[str, str]


@dataclass(frozen=True)
class ForgetClosureNode:
    """Content-free lifecycle edges of one exact, already validated row.

    Args:
        key: Existing physical kind and object identity.
        scope: Effective owner scope, resolved by the existing scope authority.
        references: Actual RefSpec-derived edges after selected compensation.
        forget_memberships: Proved selective-privacy deletion scopes. These are
            replayable-delivery grants or an admission payload's actual scope,
            not every authorization-only ScopeMembershipSpec in the registry.
        source_memory: For a Link only, the proved owner Revision's memory identity.
        aggregate: Existing members of a memory retention unit. They must be
            selected from actual owner relationships, never arbitrary caller IDs.
        physical_parts: A verified logical Artifact's actual manifest parts.
            Removing a manifest must not leave its original encoded bytes behind.
        call_outputs: For a call's input Artifact, the outputs that call
            produced. Whichever edge deletes the input, the call showed erased
            content, so its outputs are derived material and leave with it.
    """

    key: RowKey
    scope: ScopeKey | None
    references: tuple[Reference, ...]
    forget_memberships: frozenset[str] = frozenset()
    source_memory: RowKey | None = None
    aggregate: tuple[RowKey, ...] = ()
    physical_parts: tuple[RowKey, ...] = ()
    call_outputs: tuple[RowKey, ...] = ()


@dataclass(frozen=True)
class ForgetClearReference:
    """A surviving owner needs a closed transform, not an arbitrary JSON patch."""

    owner: RowKey
    path: str
    target: RowKey


@dataclass(frozen=True)
class ForgetClosure:
    """A deterministic complete fixed point over one proved inventory."""

    delete: tuple[RowKey, ...]
    clear: tuple[ForgetClearReference, ...]
    traversed_edges: int


def plan_forget_closure(
    nodes: tuple[ForgetClosureNode, ...],
    *,
    scope: ScopeKey,
    seeds: frozenset[RowKey],
    retained: frozenset[RowKey] = frozenset(),
    include_privacy_roots: bool = True,
    max_rows: int = 2_000_000,
    max_rounds: int = 20,
) -> ForgetClosure:
    """Expand the existing registry's forget axes once per relevant edge.

    Args:
        nodes: Complete caller-proved inventory, with already selected survivor
            compensation represented by its post-state references.
        scope: The administrator-authorized exact scope.
        seeds: Actual selected memory unit and selected source-bearing artifact targets.
        retained: Independently supported unrelated provenance. A conflicting
            delete edge fails closed instead of silently deleting another fact.
        include_privacy_roots: Include the separately proved conservative query
            and admission roots, and the outputs of every call whose input the
            closure deletes. Initial retention-unit selection disables these so a
            privacy dependency cannot redefine unrelated facts as selected memories.
        max_rows: Positive fixed inventory/deletion bound, at most two million.
        max_rounds: Positive root-expansion depth bound, at most twenty.

    Returns:
        Sorted deletion identities and surviving clear-ref requirements. A
        requirement still needs its own registered closed transform; this plan
        is not permission to patch an arbitrary persisted model path.

    Raises:
        QuoteMemError: On incomplete/ambiguous identity, unauthorized cross-scope
            propagation, unsupported cascade ownership, retention conflict or
            closure-budget exhaustion. No prefix is returned as a successful plan.
    """
    if type(include_privacy_roots) is not bool:
        raise QuoteMemError("invalid_request",
                            "Selective erasure privacy selection must be boolean")
    if type(max_rows) is not int or type(max_rounds) is not int or (
        not 1 <= max_rows <= 2_000_000 or not 1 <= max_rounds <= 20
    ):
        raise QuoteMemError("invalid_request",
                            "Selective erasure budget is outside its fixed bounds")
    if not nodes or not seeds or len(nodes) > max_rows:
        raise QuoteMemError("closure_incomplete",
                            "Selective erasure needs a bounded complete inventory")
    by_key: dict[RowKey, ForgetClosureNode] = {}
    reverse: dict[RowKey, list[tuple[ForgetClosureNode, Reference]]] = defaultdict(list)
    for node in nodes:
        if node.key in by_key:
            raise QuoteMemError("parity_mismatch",
                                "Erasure inventory contains a duplicate identity")
        by_key[node.key] = node
        if node.aggregate and node.key[0] != "item":
            raise QuoteMemError("binding_mismatch", "Only an actual Item owns a retention unit")
        if node.physical_parts and (
            node.key[0] != "artifact"
            or any(key[0] != "artifact" for key in node.physical_parts)
        ):
            raise QuoteMemError("binding_mismatch", "Only an actual manifest owns physical parts")
        if node.call_outputs and (
            node.key[0] != "artifact"
            or any(key[0] != "artifact" for key in node.call_outputs)
        ):
            raise QuoteMemError("binding_mismatch", "Only an actual call input owns call outputs")
        if node.source_memory is not None and (
            node.key[0] != "link" or node.source_memory[0] != "item"
        ):
            raise QuoteMemError("binding_mismatch", "Only a proved Link names its source Item")
        for ref in node.references:
            reverse[(ref.spec.target_kind, ref.target_id)].append((node, ref))
    if not seeds <= by_key.keys() or not retained <= by_key.keys():
        raise QuoteMemError("closure_incomplete", "Erasure selection is absent from its inventory")

    pending: deque[tuple[RowKey, int]] = deque()
    deleted: set[RowKey] = set()
    clear: set[ForgetClearReference] = set()
    edges = 0
    recall_authority = {
        node.key for node in nodes
        if node.key[0] == "recall" and scope[1] in node.forget_memberships
    }

    def select(key: RowKey, depth: int) -> None:
        if key in deleted:
            return
        node = by_key.get(key)
        if node is None:
            raise QuoteMemError("closure_incomplete", "Erasure closure lost a selected owner")
        if key in retained:
            raise QuoteMemError("closure_incomplete",
                                "Erasure would remove an unrelated retained fact")
        if node.scope != scope and key not in recall_authority:
            raise QuoteMemError("scope_denied",
                                "Erasure closure crossed its authorized content scope")
        if depth > max_rounds or len(deleted) >= max_rows:
            raise QuoteMemError("closure_incomplete", "Erasure closure exceeded its fixed budget")
        deleted.add(key)
        pending.append((key, depth))

    for seed in sorted(seeds):
        select(seed, 0)
    # Conservative privacy roots include zero-hit replayable deliveries and
    # abnormal admission payloads. General auth scope membership alone is not a
    # single-memory privacy root; scope forget retains its broader rules.
    for node in nodes:
        if include_privacy_roots and scope[1] in node.forget_memberships:
            if node.key[0] not in {"recall", "artifact"}:
                raise QuoteMemError("unsupported_variant",
                                    "Unopened content membership erasure owner")
            # An admission Artifact authorizes its own scoped bytes and verified
            # manifest parts, never a global or foreign owner's content.
            select(node.key, 0)
    while pending:
        key, depth = pending.popleft()
        node = by_key[key]
        for member in node.aggregate:
            select(member, depth + 1)
        for part in node.physical_parts:
            select(part, depth + 1)
        if include_privacy_roots:
            for output in node.call_outputs:
                select(output, depth + 1)
        for owner, ref in reverse.get(key, ()):
            edges += 1
            if owner.key in deleted:
                continue
            action = ref.spec.target_forget_action
            if action == "retain_locator":
                continue
            if action == "clear_ref":
                if owner.scope != scope:
                    raise QuoteMemError("scope_denied",
                                        "Erasure cannot patch another content scope")
                clear.add(ForgetClearReference(owner.key, ref.path, key))
                continue
            if key in recall_authority and owner.key[0] == "recall" and (
                ref.path == "parent_recall_id" and action == "delete_owner"
                and owner.scope == node.scope
            ):
                # The real parent's authorized deletion covers its own feedback
                # descendants; another scope's unrelated Recall still cannot join.
                recall_authority.add(owner.key)
            select(owner.key, depth + 1)
            if action == "cascade_source_item":
                if owner.source_memory is None:
                    raise QuoteMemError("binding_mismatch",
                                        "Derived Link lost its actual source Item")
                select(owner.source_memory, depth + 1)
            elif action != "delete_owner":
                raise QuoteMemError("unsupported_variant",
                                    "Reference has no supported forget action")
    return ForgetClosure(
        tuple(sorted(deleted)),
        tuple(sorted(
            (entry for entry in clear if entry.owner not in deleted),
            key=lambda entry: (entry.owner, entry.path, entry.target),
        )),
        edges,
    )
