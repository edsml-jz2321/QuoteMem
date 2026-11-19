"""Recall v4 payloads, fixed-point snapshots and shared released values (A.6.7).

The parameters are pure data. Release installation belongs to C.13 step 12;
measurement callers pass an explicit snapshot to the same ranking entry.
"""

from __future__ import annotations

from typing import Annotated, Any, Final, Literal, Self

from pydantic import Field, TypeAdapter, model_validator

from aml.core.identity import IntegrityDigest, integrity_digest
from aml.core.policy import (
    Cosine,
    FixedNumber,
    LexicalFloor,
    Number01,
    Number02,
    Number04,
    Number13,
    from_fixed_point,
    to_fixed_point,
)
from aml.store.metadata import OpaqueIdSpec, StrictModel

Route = Literal["dense", "lexical", "trigger", "entity", "evidence_dense", "evidence_lexical"]
VECTOR_ROUTES: Final[frozenset[Route]] = frozenset({"dense", "evidence_dense", "entity"})
"""Routes scored by cosine against the frozen query vector (A.6.3, A.6.5)."""


class RouteWeightsV4(StrictModel):
    """Weights for the six A.6.3 routes; nomination routes do not vote in RRF."""

    dense: Number04
    lexical: Number04
    trigger: Number04
    entity: Number04
    evidence_dense: Number04
    evidence_lexical: Number04


class ImportanceV4(StrictModel):
    """Bounded importance multiplier parameters."""

    base: Number02
    coefficient: Number02
    min: Number02
    max: Number02

    @model_validator(mode="after")
    def check_bounds(self) -> Self:
        """Reject an inverted clipping interval."""
        if self.min > self.max:
            raise ValueError("importance.min must not exceed importance.max")
        return self


class TrustWeightsV4(StrictModel):
    """Exact provenance multiplier keys."""

    verified: Number02
    reported: Number02
    inferred: Number02
    untrusted: Number02


class RecallPolicyV4(StrictModel):
    """Complete caller-supplied recall parameters; no implicit defaults."""

    enabled_routes: list[Route]
    route_quota_multiplier: Annotated[int, Field(ge=1, le=16)]
    candidate_pool_limit: Annotated[int, Field(ge=1, le=2000)]
    rrf_k: Annotated[int, Field(ge=1, le=1000)]
    dense_floor: Cosine
    evidence_dense_floor: Cosine
    lexical_floor: LexicalFloor
    evidence_lexical_floor: LexicalFloor
    route_weights: RouteWeightsV4
    importance: ImportanceV4
    trust_weights: TrustWeightsV4
    boost_cap: Annotated[float, Field(ge=0.1, le=4, allow_inf_nan=False)]
    absolute_threshold: Number13
    relative_threshold: Number01
    inject_threshold: Number13
    mmr_lambda: Number01
    max_top_k: Annotated[int, Field(ge=1, le=1000)]
    history_mode: Literal["explicit_as_of"]
    entity_expansion_max_entities: Annotated[int, Field(ge=0, le=8)]
    dense_scoring: Literal["float64_fsum_half_even_v1"]
    plain_history: Literal["none", "past_states"]
    evidence_placement: Literal["after_memories", "fused_interleave", "context_interleave"] | None
    evidence_folding: Literal["exact_whole_part"] | None

    @model_validator(mode="after")
    def check_routes(self) -> Self:
        """Validate released placement and explicit E.10 measurement parameters."""
        if len(self.enabled_routes) != len(set(self.enabled_routes)):
            raise ValueError("Enabled routes must be unique")
        if not any(getattr(self.route_weights, route) > 0 for route in self.enabled_routes):
            raise ValueError("At least one enabled route must have positive weight")
        evidence = bool(set(self.enabled_routes) & {"evidence_dense", "evidence_lexical"})
        if evidence != (self.evidence_placement is not None) or evidence != (
            self.evidence_folding is not None
        ):
            raise ValueError("Evidence routes require placement and folding, exclusively")
        memory = any(
            route in {"dense", "lexical", "trigger", "entity"}
            and getattr(self.route_weights, route) > 0
            for route in self.enabled_routes
        )
        evidence_active = any(
            route in {"evidence_dense", "evidence_lexical"}
            and getattr(self.route_weights, route) > 0
            for route in self.enabled_routes
        )
        if (memory and evidence_active and self.evidence_placement == "after_memories"
            and self.absolute_threshold == self.relative_threshold == 0):
            raise ValueError("Two-channel after_memories requires L3 thresholds")
        return self


class RecallPolicyConfigV4(StrictModel):
    """Content ranking whose trigger route nominates without an independent RRF vote."""

    policy_kind: Literal["recall"]
    config_version: Literal[4]
    parameters: RecallPolicyV4


RecallPolicyConfig = RecallPolicyConfigV4
_RECALL_CONFIG: TypeAdapter[RecallPolicyConfig] = TypeAdapter(RecallPolicyConfig)


def parse_recall_policy_config(raw: object) -> RecallPolicyConfig:
    """Decode a declared immutable recall config without upgrading its version.

    Args:
        raw: A strict config model, decoded JSON object or JSON string.

    Returns:
        The exact versioned configuration, with no defaults or conversion.

    Raises:
        ValidationError: If the version or complete configuration is invalid.
    """
    return (
        _RECALL_CONFIG.validate_json(raw)
        if isinstance(raw, str)
        else (_RECALL_CONFIG.validate_python(raw))
    )


class NormalizedRouteWeightsV4(StrictModel):
    """The fixed-point representation of RouteWeightsV4."""

    dense: FixedNumber
    lexical: FixedNumber
    trigger: FixedNumber
    entity: FixedNumber
    evidence_dense: FixedNumber
    evidence_lexical: FixedNumber


class NormalizedImportanceV4(StrictModel):
    """The fixed-point representation of ImportanceV4."""

    base: FixedNumber
    coefficient: FixedNumber
    min: FixedNumber
    max: FixedNumber


class NormalizedTrustWeightsV4(StrictModel):
    """The fixed-point representation of TrustWeightsV4."""

    verified: FixedNumber
    reported: FixedNumber
    inferred: FixedNumber
    untrusted: FixedNumber


class NormalizedRecallPolicyV4(StrictModel):
    """The fixed-point representation of RecallPolicyV4."""

    enabled_routes: list[Route]
    route_quota_multiplier: Annotated[int, Field(ge=1, le=16)]
    candidate_pool_limit: Annotated[int, Field(ge=1, le=2000)]
    rrf_k: Annotated[int, Field(ge=1, le=1000)]
    dense_floor: FixedNumber
    evidence_dense_floor: FixedNumber
    lexical_floor: FixedNumber
    evidence_lexical_floor: FixedNumber
    route_weights: NormalizedRouteWeightsV4
    importance: NormalizedImportanceV4
    trust_weights: NormalizedTrustWeightsV4
    boost_cap: FixedNumber
    absolute_threshold: FixedNumber
    relative_threshold: FixedNumber
    inject_threshold: FixedNumber
    mmr_lambda: FixedNumber
    max_top_k: Annotated[int, Field(ge=1, le=1000)]
    history_mode: Literal["explicit_as_of"]
    entity_expansion_max_entities: Annotated[int, Field(ge=0, le=8)]
    dense_scoring: Literal["float64_fsum_half_even_v1"]
    plain_history: Literal["none", "past_states"]
    evidence_placement: Literal["after_memories", "fused_interleave", "context_interleave"] | None
    evidence_folding: Literal["exact_whole_part"] | None


class RecallPolicySnapshotV4(StrictModel):
    """Frozen content ranking with trigger nomination excluded from RRF scoring."""

    schema_version: Literal[4]
    config_digest: Annotated[IntegrityDigest, OpaqueIdSpec("digest")]
    parameters: NormalizedRecallPolicyV4

    @model_validator(mode="after")
    def check_parameters(self) -> Self:
        """Validate the complete frozen parameter contract."""
        RecallPolicyV4.model_validate(
            from_fixed_point(self.parameters.model_dump(), RecallPolicyV4),
        )
        routes = self.parameters.enabled_routes
        if routes != sorted(set(routes)):
            raise ValueError("Normalized enabled_routes must be sorted and unique")
        return self


RecallPolicySnapshot = RecallPolicySnapshotV4
_RECALL_SNAPSHOT: TypeAdapter[RecallPolicySnapshot] = TypeAdapter(RecallPolicySnapshot)


def parse_recall_policy_snapshot(raw: object) -> RecallPolicySnapshot:
    """Decode frozen recall semantics without replacing them with current defaults.

    Args:
        raw: A strict snapshot model, decoded JSON object or canonical JSON string.

    Returns:
        The exact supported snapshot version and its frozen parameters.

    Raises:
        ValidationError: If the version, fixed-point values or invariants are invalid.
    """
    return (
        _RECALL_SNAPSHOT.validate_json(raw)
        if isinstance(raw, str)
        else (_RECALL_SNAPSHOT.validate_python(raw))
    )


def normalize_recall_policy(
    config: RecallPolicyConfig | dict[str, Any],
) -> RecallPolicySnapshot:
    """Normalize a complete config without supplying any missing field."""
    parsed = parse_recall_policy_config(config)
    parameters = to_fixed_point(parsed.parameters.model_dump(mode="json"))
    parameters["enabled_routes"] = sorted(set(parameters["enabled_routes"]))
    return parse_recall_policy_snapshot(
        {
            "schema_version": parsed.config_version,
            "config_digest": integrity_digest(parsed),
            "parameters": parameters,
        }
    )


def build_recall_policy(profile: Literal["product", "bench"]) -> RecallPolicyConfigV4:
    """Build the shared released A.6.7 policy without provider calls.

    Args:
        profile (Literal["product", "bench"]): Valid serving profile. Both profiles
            use the same ranking parameters and readable Evidence routes.

    Returns:
        RecallPolicyConfigV4: Explicit shared configuration for freezing.

    Raises:
        ValueError: If the profile is unknown.
    """
    if profile not in {"product", "bench"}:
        raise ValueError("Unknown policy profile")
    routes: list[Route] = [
        "dense", "lexical", "trigger", "entity", "evidence_dense", "evidence_lexical",
    ]
    return RecallPolicyConfigV4.model_validate(
        {
            "policy_kind": "recall",
            "config_version": 4,
            "parameters": {
                "enabled_routes": routes,
                "route_quota_multiplier": 4,
                "candidate_pool_limit": 200,
                "rrf_k": 60,
                "dense_floor": 0.2,
                "evidence_dense_floor": 0.2,
                "lexical_floor": 0.0,
                "evidence_lexical_floor": 0.0,
                "route_weights": {
                    "dense": 1.0,
                    "lexical": 0.6,
                    "trigger": 0.8,
                    "entity": 1.0,
                    "evidence_dense": 0.6,
                    "evidence_lexical": 0.6,
                },
                "importance": {"base": 0.9, "coefficient": 0.25, "min": 0.9, "max": 1.15},
                "trust_weights": {
                    "verified": 1.0,
                    "reported": 0.95,
                    "inferred": 0.85,
                    "untrusted": 0.7,
                },
                "boost_cap": 1.3,
                "absolute_threshold": 0.20,
                "relative_threshold": 0.30,
                "inject_threshold": 0.30,
                "mmr_lambda": 0.95,
                "max_top_k": 100,
                "history_mode": "explicit_as_of",
                "entity_expansion_max_entities": 8,
                "dense_scoring": "float64_fsum_half_even_v1",
                "plain_history": "past_states",
                "evidence_placement": "context_interleave",
                "evidence_folding": "exact_whole_part",
            },
        }
    )
