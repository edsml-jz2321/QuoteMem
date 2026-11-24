"""Declarative HeavenBase schemas of the twelve QuoteMem entities.

Each entity is one closed physical field map with exact HeavenBase types and
widths, its unique constraints (all enforced by the ``main`` backend) and its
composite lookup indexes. Every identifier, short-text, integer and float field
opts out of HeavenBase's default single-column index, so the indexes a
workspace resolves are exactly the declared ones. Startup compares fields,
routes, constraints and indexes of the registered workspace with this
declaration and fails ``schema_mismatch`` on any difference.
"""

from __future__ import annotations

import inspect
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final, cast, get_args

from quotemem.core.entities import (
    ActivationMode,
    CandidateStatus,
    EffectiveBasis,
    EntityState,
    LinkType,
    MemoryKind,
    MemoryLifecycle,
    NameKind,
    ReasonCode,
    RevisionOp,
    RevisionReason,
    SupersedeRelation,
)
from quotemem.core.errors import QuoteMemError
from quotemem.store.hb_runtime import HBRuntime, hb


@dataclass(frozen=True)
class PhysicalSchemaSpec:
    """One closed physical field map, its unique owners and its lookup indexes.

    Attributes:
        identifier: The HeavenBase entity identifier, exactly as the schema names it.
        fields: Each column's logical type, a key of the declared HeavenBase types.
        optional: The nullable columns.
        unique: The unique constraints, each enforced by the ``main`` backend.
        indexes: The composite lookup indexes as ``(contract name, fields)``.
    """

    identifier: str
    fields: dict[str, str]
    optional: frozenset[str] = frozenset()
    unique: tuple[tuple[str, ...], ...] = ()
    indexes: tuple[tuple[str, tuple[str, ...]], ...] = ()


def _fields(**groups: str) -> dict[str, str]:
    """Expand grouped field declarations without silently overwriting a column."""
    result: dict[str, str] = {}
    for logical_type, names in groups.items():
        for name in names.split():
            if name in result:
                raise ValueError(f"Duplicate physical field: {name}")
            result[name] = logical_type
    return result


ENTITY_IDENTIFIERS: Final[Mapping[str, str]] = MappingProxyType({
    "item": "agent-memory-item",
    "revision": "agent-memory-revision",
    "link": "agent-memory-link",
    "candidate": "agent-memory-candidate",
    "evidence": "agent-memory-evidence",
    "entity": "agent-memory-entity",
    "entity_name": "agent-memory-entity-name",
    "tombstone": "agent-memory-tombstone",
    "control": "agent-memory-control",
    "run": "agent-memory-evolution-run",
    "artifact": "agent-memory-artifact",
    "recall": "agent-memory-recall",
})
"""The HeavenBase identifier of each table kind, exactly as the schema names it.

The ``agent-memory-<kind>`` is shorthand for this table; no other spelling exists.
"""

DEFAULT_INDEXED_TYPES: Final = frozenset({
    "identifier", "short", "short64", "short128", "short256", "integer", "float",
})
"""Logical types HeavenBase indexes by default (identifier, short-text, integer, float)."""

NULLABLE_UNIQUE_OWNERS: Final = frozenset({
    ("revision", "source_candidate_id"),
    ("entity", "participant_key"),
    ("entity", "mint_key"),
})
"""The only nullable fields a unique constraint may cover; NULL never collides."""


def entity_class_name(kind: str, dimension: int) -> str:
    """Return the class name ``AML<Kind>D<dim>`` of one table kind.

    Args:
        kind: A table kind such as ``item`` or ``entity_name``.
        dimension: The workspace embedding dimension.

    Returns:
        The PascalCase kind between ``AML`` and ``D<dim>``, e.g. ``AMLEntityNameD1024``.
    """
    return "AML" + "".join(part.title() for part in kind.split("_")) + f"D{dimension}"


ENUMS: dict[str, tuple[str, ...]] = {
    "scope_kind": ("user", "project", "agent", "organization"),
    "security_level": ("internal", "confidential", "restricted"),
    "trust_tier": ("verified", "reported", "inferred", "untrusted"),
    "time_granularity": ("HOUR", "DAY", "MONTH", "YEAR", "unanchored"),
    "memory_lifecycle": get_args(MemoryLifecycle),
    "activation_mode": get_args(ActivationMode),
    "memory_kind": get_args(MemoryKind),
    "revision_op": get_args(RevisionOp),
    "revision_reason": get_args(RevisionReason),
    "evidence_kind": (
        "user_message",
        "assistant_message",
        "system_message",
        "tool_result",
        "document",
        "file",
        "external",
        "import",
    ),
    "role": ("user", "assistant", "tool", "system"),
    "evidence_state": ("active", "invalidated", "expired", "erased"),
    "route_state": ("pending", "indexed"),
    "link_type": get_args(LinkType),
    "supersede_relation": get_args(SupersedeRelation),
    "effective_basis": get_args(EffectiveBasis),
    "candidate_status": get_args(CandidateStatus),
    "reason_code": get_args(ReasonCode),
    "entity_state": get_args(EntityState),
    "name_kind": get_args(NameKind),
    "run_kind": (
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
    ),
    "commit_mode": ("single_compensatable", "orchestrator", "destructive_scan", "cas_only"),
    "run_status": ("queued", "running", "succeeded", "failed", "manual_attention"),
    "run_target": ("session", "candidate", "memory", "scope", "affected_set", "none"),
    "record_class": ("minimal_ref_delivery", "replayable_delivery", "interaction", "telemetry"),
    "environment": ("production", "test", "eval", "shadow", "benchmark"),
    "trigger": (
        "core_hook",
        "auto",
        "model_tool",
        "before_plan",
        "before_tool",
        "error",
        "retry",
        "low_confidence",
        "benchmark_adapter",
        "manual",
    ),
    "delivered_kind": ("core", "menu", "deep", "rendered_evidence", "none"),
    "recall_target": ("memory", "evidence"),
    "interaction": ("expanded", "used", "feedback"),
    "outcome": ("success", "failure", "helpful", "irrelevant", "wrong", "outdated", "unsafe"),
    "artifact_kind": (
        "llm_output",
        "decision_set",
        "proposal_snapshot",
        "semantic_embedding",
        "publish_plan",
        "write_set",
        "result_index",
        "request_ledger",
        "run_input",
        "auth_context",
        "policy_config",
        "policy_release",
        "erasure_plan",
        "receipt",
        "finalize_plan",
        "manifest",
        "part",
    ),
}
"""The closed physical enums, each an ``hb.Categorical``.

The memory-model enums are the ``Literal`` types of ``quotemem.core.entities``, so a
row model and its column admit the same values. ``run_kind``, ``commit_mode``,
``run_target`` and ``artifact_kind`` mirror the run and artifact registries.
"""


def _schema(
    kind: str,
    fields: dict[str, str],
    *,
    optional: str = "",
    unique: tuple[tuple[str, ...], ...] = (),
    indexes: tuple[tuple[str, tuple[str, ...]], ...] = (),
) -> PhysicalSchemaSpec:
    """Bind a table kind's schema identifier to its field map, constraints and indexes."""
    return PhysicalSchemaSpec(
        ENTITY_IDENTIFIERS[kind], fields, frozenset(optional.split()), unique, indexes,
    )


SCHEMA_SPECS: dict[str, PhysicalSchemaSpec] = {
    "item": _schema(
        "item",
        {
            **_fields(
                identifier="scope_id current_revision_id",
                short64="entity_id statement_digest",
                integer="commit_seq",
                timestamp="retired_at created_at updated_at",
            ),
            "scope_kind": "scope_kind",
            "lifecycle_state": "memory_lifecycle",
            "activation_mode": "activation_mode",
        },
        optional="retired_at",
        indexes=(
            ("ix_item_entity", ("scope_kind", "scope_id", "entity_id", "lifecycle_state")),
            ("ix_item_digest", ("scope_kind", "scope_id", "statement_digest")),
            ("ix_item_commit", ("scope_kind", "scope_id", "commit_seq")),
        ),
    ),
    "revision": _schema(
        "revision",
        {
            **_fields(
                identifier=(
                    "memory_id previous_revision_id source_candidate_id origin_run_id "
                    "policy_release_ref"
                ),
                integer="version commit_seq",
                json="snapshot evidence_ids",
                short64="content_hash",
                timestamp="recorded_at",
            ),
            "op": "revision_op",
            "reason_code": "revision_reason",
        },
        optional="previous_revision_id source_candidate_id",
        unique=(("memory_id", "version"), ("source_candidate_id",)),
        indexes=(("ix_revision_run", ("origin_run_id",)),),
    ),
    "link": _schema(
        "link",
        {
            **_fields(
                identifier="scope_id source_revision_id",
                short64="target_key verdict_ref",
                integer="effective_at",
                boolean="correction_requested",
                timestamp="created_at",
            ),
            "scope_kind": "scope_kind",
            "link_type": "link_type",
            "relation": "supersede_relation",
            "effective_basis": "effective_basis",
        },
        optional="relation effective_at effective_basis verdict_ref",
        unique=(("source_revision_id", "link_type", "target_key"),),
        indexes=(("ix_link_target", ("scope_kind", "scope_id", "link_type", "target_key")),),
    ),
    "candidate": _schema(
        "candidate",
        {
            **_fields(
                identifier="scope_id owner_run_id memory_id revision_id",
                short64=(
                    "extraction_ref proposal_ref statement_digest entity_id judge_result_ref "
                    "last_input_digest last_wake_digest"
                ),
                integer="unit_no position reentry_count created_commit_seq commit_seq",
                boolean="hold_evidence",
                json="target_memory_ids evidence_ids",
                timestamp="created_at terminal_at",
            ),
            "scope_kind": "scope_kind",
            "status": "candidate_status",
            "reason_code": "reason_code",
        },
        optional=(
            "entity_id judge_result_ref memory_id revision_id last_input_digest "
            "last_wake_digest terminal_at"
        ),
        unique=(("owner_run_id", "unit_no", "position"),),
        indexes=(
            (
                "ix_candidate_status",
                ("scope_kind", "scope_id", "status", "reentry_count", "created_commit_seq"),
            ),
            ("ix_candidate_commit", ("scope_kind", "scope_id", "commit_seq")),
            ("ix_candidate_entity", ("scope_kind", "scope_id", "entity_id", "status")),
        ),
    ),
    "evidence": _schema(
        "evidence",
        {
            **_fields(
                identifier="tenant_id scope_id origin_run_id",
                short="session_id request_id speaker_entity content_hash embedding_space",
                integer=(
                    "ordinal byte_start byte_end source_observed_at ingested_commit_seq part_no"
                ),
                timestamp="ingested_at expires_at",
                json="source_descriptor entity_bindings withheld_spans",
                long="text",
                vector="embedding",
            ),
            **{
                name: name
                for name in (
                    "scope_kind",
                    "role",
                    "time_granularity",
                    "trust_tier",
                    "security_level",
                    "route_state",
                )
            },
            "kind": "evidence_kind",
            "state": "evidence_state",
        },
        optional=(
            "role speaker_entity source_descriptor origin_run_id source_observed_at expires_at "
            "entity_bindings"
        ),
        unique=(("scope_kind", "scope_id", "session_id", "request_id", "ordinal", "part_no"),),
        indexes=(
            ("ix_evidence_state", ("scope_kind", "scope_id", "state", "ingested_commit_seq")),
            ("ix_evidence_expiry", ("scope_kind", "scope_id", "expires_at")),
            ("ix_evidence_run", ("origin_run_id", "state")),
        ),
    ),
    "entity": _schema(
        "entity",
        {
            **_fields(
                identifier="scope_id created_by_run_id",
                short64="entity_type participant_key mint_key",
                integer="commit_seq",
                timestamp="created_at updated_at",
            ),
            "scope_kind": "scope_kind",
            "state": "entity_state",
        },
        optional="participant_key mint_key",
        unique=(
            ("scope_kind", "scope_id", "participant_key"),
            ("scope_kind", "scope_id", "mint_key"),
        ),
        indexes=(
            ("ix_entity_state", ("scope_kind", "scope_id", "state")),
            ("ix_entity_commit", ("scope_kind", "scope_id", "commit_seq")),
            ("ix_entity_run", ("created_by_run_id",)),
        ),
    ),
    "entity_name": _schema(
        "entity_name",
        {
            **_fields(
                identifier="scope_id naming_memory_id naming_revision_id",
                short64="entity_id",
                short128="name",
                short256="name_norm",
                integer="annotation_index commit_seq",
                timestamp="created_at",
            ),
            "scope_kind": "scope_kind",
            "name_kind": "name_kind",
        },
        unique=(("naming_revision_id", "annotation_index"),),
        indexes=(
            ("ix_name_lookup", ("scope_kind", "scope_id", "name_norm")),
            ("ix_name_entity", ("scope_kind", "scope_id", "entity_id")),
            ("ix_name_commit", ("scope_kind", "scope_id", "commit_seq")),
        ),
    ),
    "tombstone": _schema(
        "tombstone",
        {
            **_fields(
                identifier="scope_id forget_run_id",
                short64="entity_id statement_digest",
                short="embedding_space",
                long="statement",
                integer="valid_from valid_to tombstone_seq",
                json="about_entity_ids source_evidence_ids",
                vector="embedding",
                timestamp="created_at",
            ),
            "scope_kind": "scope_kind",
            "kind": "memory_kind",
            "time_granularity": "time_granularity",
        },
        optional="valid_from valid_to",
        unique=(("scope_kind", "scope_id", "entity_id", "statement_digest"),),
        indexes=(
            ("ix_tomb_digest", ("scope_kind", "scope_id", "statement_digest")),
            ("ix_tomb_seq", ("scope_kind", "scope_id", "tombstone_seq")),
            ("ix_tomb_run", ("forget_run_id",)),
        ),
    ),
    "control": _schema(
        "control",
        {
            **_fields(
                short="key",
                json="control_refs value",
                integer="version",
                timestamp="updated_at",
                identifier="updated_by_run_id",
            ),
        },
        optional="updated_by_run_id",
        unique=(("key",),),
        indexes=(("ix_control_run", ("updated_by_run_id",)),),
    ),
    "run": _schema(
        "run",
        {
            **_fields(
                identifier=(
                    "parent_run_id principal_id scope_id auth_context_ref target_id "
                    "input_ref result_ref"
                ),
                short="stage idempotency_key admission_digest commit_marker failure_code",
                integer=(
                    "attempt run_generation commit_seq writer_epoch "
                    "admitted_erase_epoch admitted_consent_epoch"
                ),
                timestamp="created_at started_at updated_at finished_at",
            ),
            **{name: name for name in ("run_kind", "commit_mode", "scope_kind")},
            "status": "run_status",
            "target_kind": "run_target",
        },
        optional=(
            "parent_run_id scope_kind scope_id auth_context_ref target_id "
            "input_ref result_ref commit_marker commit_seq "
            "admitted_erase_epoch admitted_consent_epoch failure_code "
            "started_at updated_at finished_at"
        ),
        unique=(("run_kind", "idempotency_key", "run_generation"),),
        indexes=(
            ("ix_run_scope", ("scope_kind", "scope_id", "run_kind", "target_id")),
            ("ix_run_parent", ("parent_run_id", "run_kind")),
            ("ix_run_status", ("status", "run_kind")),
        ),
    ),
    "artifact": _schema(
        "artifact",
        {
            **_fields(
                identifier="tenant_id scope_id created_by_run_id",
                short="schema_version",
                json="body_refs artifact_refs",
                integer="size_bytes",
                artifact_body="body",
                timestamp="created_at",
            ),
            **{name: name for name in ("artifact_kind", "scope_kind", "security_level")},
        },
        optional="scope_kind scope_id",
        indexes=(
            ("ix_artifact_run", ("created_by_run_id", "artifact_kind")),
            ("ix_artifact_scope", ("scope_kind", "scope_id", "artifact_kind")),
        ),
    ),
    "recall": _schema(
        "recall",
        {
            **_fields(
                identifier=(
                    "tenant_id scope_id principal_id actual_release_ref "
                    "parent_recall_id target_object_id target_revision_id "
                ),
                short="query_digest",
                integer="snapshot_generation latency_ms injected_tokens",
                json="query_bundle allowed_scopes returned task_ref score_parts",
                timestamp="created_at",
            ),
            **{
                name: name
                for name in (
                    "scope_kind",
                    "record_class",
                    "environment",
                    "security_level",
                    "trigger",
                    "delivered_kind",
                    "interaction",
                    "outcome",
                )
            },
            "target_kind": "recall_target",
        },
        optional=(
            "trigger query_digest query_bundle actual_release_ref "
            "allowed_scopes snapshot_generation returned parent_recall_id "
            "target_kind target_object_id target_revision_id interaction "
            "task_ref outcome latency_ms injected_tokens "
            "score_parts"
        ),
        indexes=(
            ("ix_recall_parent", ("parent_recall_id", "record_class")),
            ("ix_recall_scope", ("scope_kind", "scope_id", "record_class", "created_at")),
        ),
    ),
}
"""The twelve physical tables in schema order, before the tenant normalization of
``quotemem.store.normalization`` removes the duplicated Evidence, Recall and Artifact
columns."""


def _entity_declarations(
    dimension: int, *, specs: Mapping[str, PhysicalSchemaSpec] | None = None,
) -> Iterator[tuple[str, str, dict[str, Any]]]:
    """Share exact namespaces between actual classes and passive schema descriptions."""
    if dimension not in {1024, 2048}:
        raise ValueError("QuoteMem supports only the pinned 1024 and 2048 embedding spaces")
    logical: dict[str, Any] = {
        "identifier": hb.Identifier,
        "short": hb.ShortText(max_chars=1024),
        "short64": hb.ShortText(max_chars=64),
        "short128": hb.ShortText(max_chars=128),
        "short256": hb.ShortText(max_chars=256),
        "long": hb.LongText(max_chars=65535),
        "artifact_body": hb.LongText(max_chars=262144),
        "integer": hb.Integer,
        "float": hb.Float,
        "boolean": hb.Boolean,
        "timestamp": hb.Timestamp,
        "json": hb.Json,
        "vector": hb.Vector(dimension),
        **{name: hb.Categorical(values) for name, values in ENUMS.items()},
    }
    from quotemem.store.normalization import NORMALIZED_SCHEMA_SPECS

    for kind, spec in (NORMALIZED_SCHEMA_SPECS if specs is None else specs).items():
        # A single-column unique or index already owns its field's lookup path; every
        # other default-indexed field opts out, so only the declared indexes exist.
        keyed = {fields[0] for fields in spec.unique if len(fields) == 1} | {
            fields[0] for _, fields in spec.indexes if len(fields) == 1
        }
        namespace: dict[str, Any] = {
            "identifier": spec.identifier,
            "__doc__": f"Physical {kind} schema.",
            "__constraints__": [hb.unique(*fields, to="main") for fields in spec.unique],
            "__indexes__": [hb.index(*fields) for _, fields in spec.indexes],
        }
        for name, logical_name in spec.fields.items():
            field = hb.field(logical[logical_name])
            if logical_name in DEFAULT_INDEXED_TYPES and name not in keyed:
                field = field.index(enabled=False)
            namespace[name] = field.optional() if name in spec.optional else field.required()
        yield kind, entity_class_name(kind, dimension), namespace


def build_entity_classes(
    dimension: int, *, specs: Mapping[str, PhysicalSchemaSpec] | None = None,
) -> dict[str, type[hb.Entity]]:
    """Build independent physical declarations for an admitted embedding dimension."""
    return {
        kind: type(name, (hb.Entity,), namespace)
        for kind, name, namespace in _entity_declarations(dimension, specs=specs)
    }


QUOTEMEM_ENTITY_OPTIONS: dict[str, object] = {"catalog": "identity", "gram": False}
"""Per-entity HeavenBase maintenance options QuoteMem selects for a workspace it initializes.

QuoteMem always reads its rows by Entity and never discovers them through the
HeavenBase Catalog or its keyword index, so a new workspace registers every
QuoteMem Entity with the identity-only Catalog projection and without sparse-gram
maintenance. An Entity already present in a reopened workspace keeps the
options it was registered with: HeavenBase rejects a silent change, and a
stored workspace is never migrated at startup.
"""


def registration_options(workspace: Any, identifier: str) -> dict[str, object]:
    """Return the options to pass for ``identifier``, or none when unsupported or present."""
    register = getattr(workspace, "register", None)
    if not callable(register):
        return {}
    try:
        parameters: Mapping[str, inspect.Parameter] = inspect.signature(register).parameters
    except (TypeError, ValueError):
        return {}
    if not all(name in parameters for name in QUOTEMEM_ENTITY_OPTIONS):
        return {}
    present = getattr(workspace, "entities", None)
    if isinstance(present, Mapping) and identifier in cast(Mapping[str, object], present):
        return {}
    return dict(QUOTEMEM_ENTITY_OPTIONS)


def register_schemas(
    runtime: HBRuntime, dimension: int, *,
    specs: Mapping[str, PhysicalSchemaSpec] | None = None,
) -> dict[str, type[hb.Entity]]:
    """Register and verify the complete current physical schema set."""
    from quotemem.store.normalization import NORMALIZED_SCHEMA_SPECS

    selected = NORMALIZED_SCHEMA_SPECS if specs is None else specs
    if set(selected) != set(NORMALIZED_SCHEMA_SPECS):
        raise QuoteMemError("schema_mismatch",
                            "Registration requires a complete admitted schema set")
    entities = build_entity_classes(dimension, specs=selected)
    for kind, entity in entities.items():
        options = registration_options(runtime.workspace, selected[kind].identifier)
        entities[kind] = runtime.workspace.register(entity, **options)
    assert_registered_schemas(runtime, entities, dimension, specs=selected)
    return entities


def physical_schema_description(
    dimension: int, *, specs: Mapping[str, PhysicalSchemaSpec] | None = None,
) -> dict[str, Any]:
    """Describe the selected physical schema set for workspace binding.

    This function declares no Entity classes, opens no Context or database, and
    excludes logical JSON variants. The description, and so the schema
    fingerprint, covers fields, unique owners, routes and lookup indexes.
    """
    from heavenbase.entity.assembly import assemble_entity

    result: dict[str, Any] = {}
    from quotemem.store.normalization import NORMALIZED_SCHEMA_SPECS

    selected = NORMALIZED_SCHEMA_SPECS if specs is None else specs
    for kind, name, namespace in _entity_declarations(dimension, specs=selected):
        spec = selected[kind]
        schema = assemble_entity(name, namespace).schema
        fields = cast(dict[str, Any], cast(Any, schema).to_dict()["fields"])
        result[kind] = {
            "identifier": spec.identifier,
            "fields": {
                name: {key: field[key] for key in ("dtype", "pk", "required", "has_default")}
                for name, field in fields.items()
            },
            "unique": [{"fields": list(names), "backend": "main"} for names in spec.unique],
            "indexes": [{"fields": list(names), "backend": "main"} for _, names in spec.indexes],
            "routes": {
                name: {
                    "backend": "main",
                    "strategy": logical if logical in {"json", "vector"} else "inline",
                }
                for name, logical in {"object_id": "identifier", **spec.fields}.items()
            },
        }
    return result


def assert_registered_schemas(
    runtime: HBRuntime,
    entities: dict[str, type[hb.Entity]],
    dimension: int,
    *,
    specs: Mapping[str, PhysicalSchemaSpec] | None = None,
) -> None:
    """Fail startup when actual HB schemas, routing, constraints or indexes differ.

    Read the registered entities and resolved storage plans from this runtime;
    comparison of source declarations alone cannot establish startup conformance.
    """
    from quotemem.store.normalization import NORMALIZED_SCHEMA_SPECS

    selected = NORMALIZED_SCHEMA_SPECS if specs is None else specs
    expected = build_entity_classes(dimension, specs=selected)
    workspace = cast(Any, runtime.workspace)
    if set(entities) != set(selected) or set(selected) != set(NORMALIZED_SCHEMA_SPECS):
        raise QuoteMemError("schema_mismatch", "Startup requires a complete admitted schema set")
    for kind, spec in selected.items():
        registered = workspace.entities.get(spec.identifier)
        plan = workspace.storage_plans.get(spec.identifier)
        if registered is None or plan is None:
            raise QuoteMemError(
                "schema_mismatch", f"Registered schema or storage plan is missing: {kind}"
            )
        expected_fields = cast(
            dict[str, Any], cast(Any, expected[kind]).schema().to_dict()["fields"]
        )
        actual_fields = cast(dict[str, Any], registered.schema().to_dict()["fields"])
        if set(actual_fields) != set(expected_fields):
            raise QuoteMemError("schema_mismatch", f"Registered physical columns differ: {kind}")
        for constraint in plan.constraints:
            for name in constraint.fields:
                if (
                    not actual_fields[name]["required"]
                    and (kind, name) not in NULLABLE_UNIQUE_OWNERS
                ):
                    raise QuoteMemError(
                        "schema_mismatch", f"Unique owner fields must be required: {kind}.{name}"
                    )
        for name, field in actual_fields.items():
            if any(
                field[key] != expected_fields[name][key]
                for key in ("dtype", "pk", "required", "has_default")
            ):
                raise QuoteMemError(
                    "schema_mismatch", f"Registered field contract differs: {kind}.{name}"
                )
        routes = {
            (binding.field_name, binding.backend, binding.strategy) for binding in plan.bindings
        }
        expected_routes = {
            (name, "main", logical if logical in {"json", "vector"} else "inline")
            for name, logical in {"object_id": "identifier", **spec.fields}.items()
        }
        if routes != expected_routes or len(plan.bindings) != len(expected_routes):
            raise QuoteMemError(
                "schema_mismatch", f"Physical routing differs from the main owner: {kind}"
            )
        unique = tuple(
            (tuple(constraint.fields), constraint.backend) for constraint in plan.constraints
        )
        if unique != tuple((fields, "main") for fields in spec.unique):
            raise QuoteMemError("schema_mismatch", f"Unique enforcement owners differ: {kind}")
        indexes = sorted(
            (tuple(index.fields), index.backend, index.source) for index in plan.indexes
        )
        if indexes != sorted((fields, "main", "entity") for _, fields in spec.indexes):
            raise QuoteMemError("schema_mismatch", f"Resolved lookup indexes differ: {kind}")
