"""One logical embedding batch of an Add owner's statements and correction cues.

``StatementEmbeddingV1`` is the ``semantic_embedding`` artifact: the
vectors of one successful batch of at most 32 unique texts, persisted before
the next batch starts, so a resumed owner reuses every completed batch without
a call. It holds no text: each entry names its role, the SHA-256 of its exact
UTF-8 text and the positions that use it. A ``ProposalV2`` points at one entry
by ``{ref, index}``.
"""

from __future__ import annotations

from typing import Annotated, Final, Literal, Self

from pydantic import Field, field_validator, model_validator

from quotemem.core.entities import PositionKey
from quotemem.core.identity import ULID, IntegrityDigest
from quotemem.core.semantic import EmbeddingVector
from quotemem.core.statement import HexDigest
from quotemem.core.technical_types import LocalId
from quotemem.store.metadata import OpaqueIdSpec, RefSpec, StrictModel

MAX_EMBEDDING_BATCH: Final = 32
"""Texts per logical embedding batch; the adapter sends at most 20 per HTTP request."""

EmbeddingRole = Literal["statement", "correction_cue"]
"""What an embedded text is: a statement, or a ``corrects_text`` cue."""


class StatementEmbeddingEntryV1(StrictModel):
    """One unique text of one role in a batch, and every position that uses it.

    Attributes:
        role: ``statement`` or ``correction_cue``; a vector is reused only
            within its role and space.
        input_digest: SHA-256 hex of the exact UTF-8 text.
        uses: The positions whose statement or cue this text is, sorted.
        embedding_token: The active embedding state the vector was made under.
        vector: The vector, finite and nonzero.
    """

    role: EmbeddingRole
    input_digest: Annotated[HexDigest, OpaqueIdSpec("digest")]
    uses: Annotated[list[PositionKey], Field(min_length=1)]
    embedding_token: Annotated[IntegrityDigest, OpaqueIdSpec("digest")]
    vector: EmbeddingVector

    @field_validator("uses")
    @classmethod
    def sorted_uses(cls, value: list[tuple[str, int, int]]) -> list[tuple[str, int, int]]:
        """Require sorted, unique positions, so one batch has one encoding."""
        if value != sorted(set(value)):
            raise ValueError("An embedding entry's uses are sorted and unique")
        return value


class StatementEmbeddingV1(StrictModel):
    """One successful logical batch of an owner's embeddings.

    Attributes:
        owner_run_id: The Add owner that embedded the batch.
        embedding_space: ``{model}@{canary_digest}@{dimensions}`` of the workspace.
        resolved_model: The model the provider reported.
        dimensions: The vector dimension.
        entries: 1..32 entries, unique by role and text.
    """

    owner_run_id: Annotated[ULID, RefSpec("run", scope_rule="same_scope")]
    embedding_space: LocalId
    resolved_model: LocalId
    dimensions: Literal[1024, 2048]
    entries: Annotated[
        list[StatementEmbeddingEntryV1], Field(min_length=1, max_length=MAX_EMBEDDING_BATCH),
    ]

    @model_validator(mode="after")
    def check_batch(self) -> Self:
        """Bind every vector to the declared model, space and dimension; unique texts."""
        try:
            model, _, raw_dimension = self.embedding_space.rsplit("@", 2)
            space_dimension = int(raw_dimension)
        except ValueError:
            raise ValueError("The embedding space is malformed") from None
        if model != self.resolved_model or space_dimension != self.dimensions:
            raise ValueError("The embedding model or space differs from its vectors")
        if any(len(entry.vector) != self.dimensions or not any(entry.vector)
               for entry in self.entries):
            raise ValueError("Every vector has the batch dimension and a nonzero norm")
        keys = [(entry.role, entry.input_digest) for entry in self.entries]
        if len(set(keys)) != len(keys):
            raise ValueError("A batch embeds each text of each role once")
        return self
