"""Surviving entries of an erased embedding batch, retaining its original indices."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from quotemem.core.identity import ULID, ArtifactId
from quotemem.core.semantic_embedding import MAX_EMBEDDING_BATCH, StatementEmbeddingEntryV1
from quotemem.core.technical_types import LocalId
from quotemem.store.metadata import OpaqueIdSpec, StrictModel


class PrivacyEmbeddingV1(StrictModel):
    """A monotone subset of one original batch, written before its predecessor is deleted.

    Empty entries are the durable proof that none of the original vectors survives.
    Original positions and indices never change, so surviving proposals need no
    rewrite and no provider resend. Locators identify deleted records, not live refs.
    """

    privacy_embedding: Literal[1]
    original_ref: Annotated[ArtifactId, OpaqueIdSpec("local")]
    owner_run_id: Annotated[ULID, OpaqueIdSpec("local")]
    embedding_space: LocalId
    resolved_model: LocalId
    dimensions: Literal[1024, 2048]
    indices: Annotated[
        list[Annotated[int, Field(ge=0, lt=MAX_EMBEDDING_BATCH)]],
        Field(max_length=MAX_EMBEDDING_BATCH),
    ]
    entries: Annotated[list[StatementEmbeddingEntryV1], Field(max_length=MAX_EMBEDDING_BATCH)]

    @model_validator(mode="after")
    def validate_subset(self) -> Self:
        """Keep original indices unique and every surviving vector in its owner's space."""
        if self.indices != sorted(set(self.indices)) or len(self.indices) != len(self.entries):
            raise ValueError("Retained embedding indices are ordered, unique and complete")
        try:
            model, _, dimension = self.embedding_space.rsplit("@", 2)
        except ValueError:
            raise ValueError("Retained embedding space is malformed") from None
        if model != self.resolved_model or dimension != str(self.dimensions):
            raise ValueError("Retained embedding identity differs from its space")
        if any(len(entry.vector) != self.dimensions or not any(entry.vector)
               or any(key[0] != self.owner_run_id for key in entry.uses)
               for entry in self.entries):
            raise ValueError("Retained vectors belong to their original owner and space")
        return self
