"""Frozen Search authorization, locators and the recall v4 query bundle (A.3.11, E.1.3)."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, JsonValue, model_validator

from aml.core.identity import (
    ULID,
    ArtifactId,
    IntegrityDigest,
    PrincipalId,
    ScopeId,
    canonical_json,
)
from aml.core.recall_policy import VECTOR_ROUTES, RecallPolicySnapshotV4, Route
from aml.core.semantic import (
    EmbeddingVector,
    HistoricalTimestamp,
    ScopeKind,
    ShortText,
    SystemTimestamp,
)
from aml.core.time_order import read_time_order
from aml.store.metadata import (
    OpaqueIdSpec,
    OpaquePayloadSpec,
    RefSpec,
    ScopeMembershipSpec,
    StrictModel,
)

MAX_QUERY_BUNDLE_BYTES = 65536
QUERY_OMISSION = "\n[... query middle omitted ...]\n"
"""The marker a bounded Search query keeps between its head and tail (E.1.3)."""
MAX_CARRIED_PARTS = 256
"""Most Evidence parts a bounded query's bundle records as carried by its whole text."""
CARRIED_RESERVE_BYTES = (
    len(',"carried":[]') + MAX_CARRIED_PARTS * len('"01ARZ3NDEKTSV4RRFFQ69G5FAV",')
)
"""Bundle bytes a bounded query leaves for its carried parts (E.1.3)."""


class ScopeV1(StrictModel):
    """An exact internal scope identity with privacy membership semantics."""

    scope_kind: ScopeKind
    scope_id: Annotated[ScopeId, OpaqueIdSpec("local"), ScopeMembershipSpec()]


class ScopeGrantV1(ScopeV1):
    """A frozen authorization relation and both deletion fences."""

    relation: Literal["exact", "ancestor"]
    erase_epoch: Annotated[int, Field(ge=0)]
    consent_epoch: Annotated[int, Field(ge=0)]


class FrozenScopeGrantSetV1(StrictModel):
    """An immutable grant set; fresh action authorization never overwrites it."""

    principal_id: Annotated[PrincipalId, OpaqueIdSpec("local")]
    requested_scope: ScopeV1
    grants: Annotated[list[ScopeGrantV1], Field(min_length=1)]
    issued_at: SystemTimestamp
    expires_at: SystemTimestamp

    @model_validator(mode="after")
    def check_grants(self) -> Self:
        """Require one exact requested grant, canonical identities, and a valid lifetime."""
        keys = [(grant.scope_kind, grant.scope_id, grant.relation) for grant in self.grants]
        if keys != sorted(set(keys)):
            raise ValueError("Scope grants must be sorted and unique")
        requested = (self.requested_scope.scope_kind, self.requested_scope.scope_id, "exact")
        if requested not in keys or self.issued_at >= self.expires_at:
            raise ValueError("Scope grants require the exact requested scope and a valid lifetime")
        return self


class MemoryLocatorV1(StrictModel):
    """A non-rooting frozen Item locator."""

    kind: Literal["memory"]
    object_id: Annotated[ULID, RefSpec("item", ref_mode="snapshot_locator", gc_root=False)]


class EvidenceLocatorV1(StrictModel):
    """A non-rooting frozen Evidence locator."""

    kind: Literal["evidence"]
    object_id: Annotated[ULID, RefSpec("evidence", ref_mode="snapshot_locator", gc_root=False)]


ObjectLocatorV1 = Annotated[MemoryLocatorV1 | EvidenceLocatorV1, Field(discriminator="kind")]


class ReturnedMemoryV1(MemoryLocatorV1):
    """An exact delivered Item version and content binding."""

    revision_id: Annotated[ULID, RefSpec("revision", ref_mode="snapshot_locator", gc_root=False)]
    content_hash: Annotated[IntegrityDigest, OpaqueIdSpec("digest")]
    rank: Annotated[int, Field(ge=1)]
    score: Annotated[float, Field(allow_inf_nan=False)]


class ReturnedEvidenceV1(EvidenceLocatorV1):
    """Delivered Evidence cannot carry an Item Revision identity."""

    content_hash: Annotated[IntegrityDigest, OpaqueIdSpec("digest")]
    rank: Annotated[int, Field(ge=1)]
    score: Annotated[float, Field(allow_inf_nan=False)]


ReturnedItemV1 = Annotated[ReturnedMemoryV1 | ReturnedEvidenceV1, Field(discriminator="kind")]


class AsOfV1(StrictModel):
    """A frozen explicit world-time or knowledge-time view."""

    mode: Literal["world", "knowledge"]
    at: HistoricalTimestamp


class QueryBundleV1(StrictModel):
    """The complete retrieval-only replay input of one Search (A.3.11, E.1.3).

    ``query`` and ``options`` serve retrieval only (INV A16). When ``query`` is
    bounded around `QUERY_OMISSION`, ``carried`` lists the Evidence parts that
    the whole original query contained verbatim, so ranking treats them as
    already said (A.6.3) without the omitted text. The bundle freezes
    the released recall v4 snapshot, the routes active for the resolved mode and
    the provider vector those routes need; replay never depends on the current
    provider configuration. A failed embedding is a Search error (E.1.3), so no
    bundle ever records a degraded route set.
    """

    schema_version: Literal[1]
    query: str
    options: Annotated[dict[str, JsonValue] | list[str], OpaquePayloadSpec(max_bytes=16384)]
    query_time: SystemTimestamp
    as_of: AsOfV1 | None = None
    exclude: list[ObjectLocatorV1]
    session_lru: list[ObjectLocatorV1]
    active_routes: list[Route]
    policy_release_ref: Annotated[
        ArtifactId,
        RefSpec(
            "artifact",
            ref_mode="snapshot_locator",
            gc_root=True,
            artifact_kinds=("policy_release",),
        ),
    ]
    policy_snapshot: RecallPolicySnapshotV4
    embedding_space: Annotated[ShortText, OpaqueIdSpec("local")]
    embedding_epoch: Annotated[IntegrityDigest, OpaqueIdSpec("digest")]
    query_vector: EmbeddingVector | None = None
    top_k: Annotated[int, Field(ge=1, le=1000)]
    carried: list[
        Annotated[ULID, RefSpec("evidence", ref_mode="snapshot_locator", gc_root=False)]
    ] = Field(
        default_factory=list, max_length=MAX_CARRIED_PARTS, exclude_if=lambda value: not value,
    )

    @model_validator(mode="after")
    def check_frozen_inputs(self) -> Self:
        """Require a frozen vector for every active vector route and the policy limits."""
        if len(canonical_json(self.options).encode("utf-8")) > 16384:
            raise ValueError("Query options exceed their external payload budget")
        read_time_order(self.options, self.policy_snapshot.schema_version)
        if len(self.active_routes) != len(set(self.active_routes)):
            raise ValueError("active_routes must be unique")
        if set(self.active_routes) - set(self.policy_snapshot.parameters.enabled_routes):
            raise ValueError("An active route is disabled by the frozen policy")
        vector_routes = bool(VECTOR_ROUTES & set(self.active_routes))
        if vector_routes and (self.query_vector is None or not any(self.query_vector)):
            raise ValueError("Vector routes require a nonzero frozen query vector")
        if self.top_k > self.policy_snapshot.parameters.max_top_k:
            raise ValueError("top_k exceeds the frozen policy limit")
        if len(canonical_json(self).encode("utf-8")) > MAX_QUERY_BUNDLE_BYTES:
            raise ValueError("QueryBundle exceeds 64 KiB")
        return self
