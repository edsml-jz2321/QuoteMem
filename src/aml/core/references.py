"""Typed target identities for the derived reference indexes of artifacts and controls.

These models are embedded only beneath DerivedSpec in materialized index fields
(``body_refs``, ``control_refs``). Their type metadata preserves target kinds
without creating another authority; every table a reference may target (A.1)
has one branch.
"""

from typing import Annotated, Literal

from pydantic import Field

from aml.core.identity import ULID, ArtifactId, ControlId
from aml.store.metadata import RefSpec, StrictModel


class ItemReference(StrictModel):
    """Identify an Item in a generated reverse index."""

    kind: Literal["item"]
    object_id: Annotated[ULID, RefSpec("item")]


class RevisionReference(StrictModel):
    """Identify a Revision in a generated reverse index."""

    kind: Literal["revision"]
    object_id: Annotated[ULID, RefSpec("revision")]


class EvidenceReference(StrictModel):
    """Identify Evidence in a generated reverse index."""

    kind: Literal["evidence"]
    object_id: Annotated[ULID, RefSpec("evidence")]


class LinkReference(StrictModel):
    """Identify a Link in a generated reverse index."""

    kind: Literal["link"]
    object_id: Annotated[ULID, RefSpec("link")]


class CandidateReference(StrictModel):
    """Identify a Candidate in a generated reverse index."""

    kind: Literal["candidate"]
    object_id: Annotated[ULID, RefSpec("candidate")]


class EntityReference(StrictModel):
    """Identify an entity, such as an SDK binding's, in a generated reverse index."""

    kind: Literal["entity"]
    object_id: Annotated[ULID, RefSpec("entity")]


class EntityNameReference(StrictModel):
    """Identify a catalog's frozen name row in a generated reverse index."""

    kind: Literal["entity_name"]
    object_id: Annotated[ULID, RefSpec("entity_name")]


class TombstoneReference(StrictModel):
    """Identify a tombstone, such as a settlement's forgotten digest, in a reverse index."""

    kind: Literal["tombstone"]
    object_id: Annotated[ULID, RefSpec("tombstone")]


class RunReference(StrictModel):
    """Identify an EvolutionRun in a generated reverse index."""

    kind: Literal["run"]
    object_id: Annotated[ULID, RefSpec("run")]


class RecallReference(StrictModel):
    """Identify a Recall record in a generated reverse index."""

    kind: Literal["recall"]
    object_id: Annotated[str, Field(min_length=1, max_length=63), RefSpec("recall")]


class ArtifactReference(StrictModel):
    """Identify an Artifact in a generated reverse index."""

    kind: Literal["artifact"]
    object_id: Annotated[ArtifactId, RefSpec("artifact")]


class ControlReference(StrictModel):
    """Identify a Control row in a generated reverse index."""

    kind: Literal["control"]
    object_id: Annotated[ControlId, RefSpec("control")]


ReferenceProjectionV1 = Annotated[
    ItemReference | RevisionReference | EvidenceReference | LinkReference | CandidateReference
    | EntityReference | EntityNameReference | TombstoneReference | RunReference | RecallReference
    | ArtifactReference | ControlReference,
    Field(discriminator="kind"),
]
