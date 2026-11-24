"""Implemented schema admission and generated conformance reports.

The inventory compiles every strict row, artifact and control variant, and must
cover all twelve tables; the physical fingerprint binds each workspace.
"""

from __future__ import annotations

import copy
from functools import lru_cache
from typing import Any, cast, get_args, get_origin

from pydantic import BaseModel, create_model

from quotemem.core.artifacts import ARTIFACT_MODELS, ArtifactRowV1
from quotemem.core.controls import CONTROL_MODELS, ControlRowV1
from quotemem.core.errors import QuoteMemError
from quotemem.core.identity import IntegrityDigest, integrity_digest
from quotemem.store import domain
from quotemem.store.entities import SCHEMA_SPECS
from quotemem.store.metadata import StrictModel
from quotemem.store.registry import Registry, SchemaVariant, compile_inventory


def _variants(annotation: Any) -> list[type[StrictModel]]:
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return [cast(type[StrictModel], annotation)]
    found: list[type[StrictModel]] = []
    if get_origin(annotation) is not None:
        for argument in get_args(annotation):
            found.extend(_variants(argument))
    return found


def _body_view(
    base: type[StrictModel], name: str, field: str, body: type[StrictModel]
) -> type[StrictModel]:
    # This inventory-only model projects a physical JSON/string column into its
    # actual strict logical branch. It is never used as another write parser.
    fields: dict[str, Any] = {field: (body, ...)}
    return create_model(name, __base__=base, **fields)


@lru_cache(maxsize=1)
def implemented_variants() -> tuple[SchemaVariant, ...]:
    """Enumerate actual strict models; unsupported declared values get no placeholder."""
    row_types: dict[str, Any] = {
        "item": domain.MemoryRow,
        "revision": domain.RevisionRowV2,
        "link": domain.LinkRowV2,
        "candidate": domain.CandidateRowV2,
        "evidence": domain.EvidenceRowV1,
        "entity": domain.EntityRowV1,
        "entity_name": domain.EntityNameRowV1,
        "tombstone": domain.TombstoneRowV1,
        "run": domain.EvolutionRunRowV1,
        "recall": domain.RecallRowV1,
    }
    result = [
        SchemaVariant(kind, model.__name__, model)
        for kind, annotation in row_types.items()
        for model in _variants(annotation)
    ]
    for branch, model in ARTIFACT_MODELS.items():
        name = "ArtifactInventory_" + branch.replace(":", "_")
        result.append(
            SchemaVariant("artifact", branch, _body_view(ArtifactRowV1, name, "body", model))
        )
    for branch, model in CONTROL_MODELS.items():
        name = "ControlInventory_" + branch.replace(":", "_")
        # Preserve column-level derived metadata when narrowing externally
        # discriminated Control.value to one actual namespace branch.
        field = copy.deepcopy(ControlRowV1.model_fields["value"])
        view = cast(
            type[StrictModel], create_model(name, __base__=ControlRowV1, value=(model, field))
        )
        result.append(SchemaVariant("control", branch, view))
    return tuple(result)


IMPLEMENTED_SCHEMA_SET = frozenset((item.entity, item.branch) for item in implemented_variants())

_FINALIZE_INVENTORY_CONDITIONAL_ROOTS = frozenset({
    ("run_input:finalize:owner", "body.source_inventory_ref"),
    *{("decision_set:finalize_source_inventory", "body." + field) for field in (
        "owner_run_id", "previous_ref", "base_ref", "removed_ids[]", "records[].header.object_id",
    )},
    *{
        ("decision_set:finalize_source_inventory/body.records[].key=" + variant,
         "body.records[].key.evidence_id")
        for variant in ("PlanMessageRangeKeyV1", "PlanMessageRangeKeyV2")
    },
    *{("decision_set:finalize_source_ttl", "body." + field) for field in (
        "owner_run_id", "source_inventory_ref", "unit_inventory_ref",
        "previous_projection_ref", "previous_ref", "records[].retained_member_ids[]",
        "records[].expired_sources[].evidence_id", "records[].omitted_member_ids[]",
    )},
})


def schema_registry() -> Registry:
    """Build the exhaustive leaf inventory and fail closed on metadata gaps."""
    registry = compile_inventory(implemented_variants())
    if {leaf.entity for leaf in registry.leaves} != set(SCHEMA_SPECS):
        raise RuntimeError("Implemented inventory does not cover all admitted physical schemas")
    for leaf in registry.leaves:
        if leaf.ref is None:
            continue
        if leaf.ref.gc_root == "while_owner_nonterminal" and not (
            leaf.entity == "artifact"
            and (leaf.branch, leaf.path) in _FINALIZE_INVENTORY_CONDITIONAL_ROOTS
        ):
            raise QuoteMemError(
                "registry_invalid", "Conditional root is outside the implemented allowlist"
            )
        if leaf.ref.scope_rule == "protocol" and not leaf.path.endswith("updated_by_run_id"):
            raise QuoteMemError("registry_invalid",
                                "Protocol scope exception is not Control provenance")
    return registry


def schema_fingerprint(dimension: int) -> IntegrityDigest:
    """Bind the current normalized physical declaration."""
    from quotemem.store.entities import physical_schema_description
    from quotemem.store.normalization import NORMALIZED_SCHEMA_SPECS

    if dimension not in {1024, 2048}:
        raise ValueError("Workspace embedding dimension must be 1024 or 2048")
    return integrity_digest(
        physical_schema_description(dimension, specs=NORMALIZED_SCHEMA_SPECS),
    )


def conformance_digest() -> IntegrityDigest:
    """Identify a generated logical report without modifying the workspace binding."""
    return integrity_digest(
        {
            "schema_version": 1,
            "domain": {
                kind: adapter.json_schema() for kind, adapter in domain.DOMAIN_ADAPTERS.items()
            },
            "artifacts": {
                branch: model.model_json_schema() for branch, model in ARTIFACT_MODELS.items()
            },
            "controls": {
                branch: model.model_json_schema() for branch, model in CONTROL_MODELS.items()
            },
            "inventory": schema_registry().report(),
        }
    )
