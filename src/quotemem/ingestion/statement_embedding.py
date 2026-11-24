"""Statement and correction-cue embedding of one Add owner.

After settlement, the owner embeds the statement of every unsettled position
(unresolved primaries and their review followers included) and every non-null
``corrects_text``, once per
unique text and role. The texts go in canonical order in logical batches of at
most 32; the adapter splits a batch into HTTP requests of at most 20 inputs, 30 s
each. Each successful batch is stored as one ``StatementEmbeddingV1`` before the
next starts, so a resumed owner finds every completed batch and makes no call for
it. A statement that a live, non-erased head of the scope already states in the
same space reuses that head's vector without a call; a cue never does, since no
revision stores a cue vector. Tombstones and erased material are never a source.
A provider failure propagates and pauses the owner. Nothing here runs under the
Writer lock: a batch is embedded first and stored afterwards. The owner's
authority is rechecked before each batch, before every physical send and after
each reply, so a revoked, cancelled or expired owner neither calls nor stores
another batch.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Final, Literal, Protocol

from quotemem.core.add_read_view import AddReadView
from quotemem.core.entities import PositionKey
from quotemem.core.errors import QuoteMemError
from quotemem.core.identity import ArtifactId, IntegrityDigest
from quotemem.core.inference import DEFAULT_PROVIDER_TIMEOUT_MS, default_retry_policy
from quotemem.core.proposal import EmbeddingUseV1
from quotemem.core.semantic_embedding import (
    MAX_EMBEDDING_BATCH,
    EmbeddingRole,
    StatementEmbeddingEntryV1,
    StatementEmbeddingV1,
)
from quotemem.core.statement import statement_digest
from quotemem.providers.attempts import provider_attempts
from quotemem.providers.contracts import EmbeddingProvider, EmbeddingRequest, EmbeddingResult
from quotemem.store.metadata import StrictModel
from quotemem.store.writer import WriteTicket

EMBEDDING_REQUEST_MS: Final = DEFAULT_PROVIDER_TIMEOUT_MS
"""Each HTTP embedding request's limit, 30 s."""

TextKey = tuple[EmbeddingRole, str]
"""One embedded text: its role and its exact text."""


@dataclass(frozen=True, slots=True)
class EmbeddingText:
    """One unique text of one role and the positions that use it.

    Attributes:
        role (EmbeddingRole): ``statement`` or ``correction_cue``.
        text (str): The exact text.
        uses (tuple[PositionKey, ...]): The positions it belongs to, in order.
    """

    role: EmbeddingRole
    text: str
    uses: tuple[PositionKey, ...]

    @property
    def key(self) -> TextKey:
        """The text's role and text."""
        return (self.role, self.text)

    @property
    def digest(self) -> str:
        """SHA-256 hex of the text's UTF-8 bytes (``StatementEmbeddingV1`` ``input_digest``)."""
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


def embedding_texts(
    positions: Sequence[tuple[PositionKey, str, str | None]],
) -> tuple[EmbeddingText, ...]:
    """Return the unique texts of the positions to embed, in order.

    Args:
        positions (Sequence[tuple[PositionKey, str, str | None]]): Each position
            to embed with its statement and its ``corrects_text`` or None, in
            canonical order.

    Returns:
        tuple[EmbeddingText, ...]: Each unique ``(role, text)`` once, at its first
            use: a position's statement, then its cue.
    """
    found: dict[TextKey, list[PositionKey]] = {}
    for key, statement, cue in positions:
        found.setdefault(("statement", statement), []).append(key)
        if cue is not None:
            found.setdefault(("correction_cue", cue), []).append(key)
    return tuple(EmbeddingText(role, text, tuple(uses)) for (role, text), uses in found.items())


class BatchStore(Protocol):
    """The Writer's create-once logical artifacts, as the embedding stage uses them."""

    def find_logical_candidates(
        self, creator: str, kind: str,
    ) -> list[tuple[ArtifactId, StrictModel]]:
        """Return a Run's stored logical artifacts of one kind."""
        ...

    def put_logical_artifact(
        self, ticket: WriteTicket, kind: str, body: object, *, security_level: str = "internal",
        created_at: int,
    ) -> ArtifactId:
        """Store one logical artifact create-once and return its id."""
        ...


@dataclass(frozen=True, slots=True)
class OwnerEmbedding:
    """What one owner's embeddings are made and stored under.

    Attributes:
        owner_run_id (str): The Add owner.
        ticket (WriteTicket): The owner's ticket, for storing batches.
        space (str): The workspace's active space, ``{model}@{canary}@{dimensions}``.
        token (IntegrityDigest): The active embedding state token.
        clock (Callable[[], int]): The trusted clock, epoch milliseconds.
        authorize (Callable[[], None]): The owner's recheck of grant, lease,
            cancellation and privacy validity; it raises the owner's own error.
    """

    owner_run_id: str
    ticket: WriteTicket
    space: str
    token: IntegrityDigest
    clock: Callable[[], int]
    authorize: Callable[[], None]

    @property
    def model(self) -> str:
        """The space's model."""
        return self.space.rsplit("@", 2)[0]

    @property
    def dimensions(self) -> Literal[1024, 2048]:
        """The space's vector dimension."""
        dimensions = int(self.space.rsplit("@", 2)[2])
        if dimensions not in (1024, 2048):
            raise QuoteMemError("embedding_space_mismatch",
                                "The embedding space has no valid dimension")
        return dimensions


def _stored(owner: OwnerEmbedding, store: BatchStore) -> dict[tuple[str, str], EmbeddingUseV1]:
    """The owner's stored batches, by role and input digest."""
    found: dict[tuple[str, str], EmbeddingUseV1] = {}
    for reference, body in store.find_logical_candidates(owner.owner_run_id, "semantic_embedding"):
        if not isinstance(body, StatementEmbeddingV1) or body.owner_run_id != owner.owner_run_id:
            raise QuoteMemError("binding_mismatch",
                                "An owner's embedding batch names another owner")
        if body.embedding_space != owner.space:
            raise QuoteMemError("embedding_space_mismatch", "A stored batch is of another space")
        for index, entry in enumerate(body.entries):
            found.setdefault((entry.role, entry.input_digest), EmbeddingUseV1(
                ref=reference, index=index,
            ))
    return found


def _live_vector(text: EmbeddingText, view: AddReadView) -> list[float] | None:
    """The vector of a live, non-erased head that states exactly this statement."""
    if text.role != "statement":
        return None
    for record in view.heads_by_digest(statement_digest(text.text)):
        snapshot = record.head.snapshot
        if not snapshot.erased and snapshot.statement == text.text and (
            snapshot.embedding_space == view.embedding_space
        ):
            return list(snapshot.embedding)
    return None


def _embedded(
    owner: OwnerEmbedding, provider: EmbeddingProvider, texts: Sequence[EmbeddingText],
) -> EmbeddingResult:
    """Embed one batch's texts in one logical request (the adapter splits it by 20)."""
    result = provider.embed(EmbeddingRequest(
        dimensions=owner.dimensions, texts=[text.text for text in texts],
        timeout_ms=EMBEDDING_REQUEST_MS, retry_policy=default_retry_policy(),
    ))
    result = EmbeddingResult.model_validate(result.model_dump(mode="json"))
    if len(result.vectors) != len(texts):
        raise QuoteMemError("provider_error", "The embedding reply lost an input")
    if result.dimensions != owner.dimensions or result.resolved_model != owner.model:
        raise QuoteMemError("embedding_space_mismatch", "The embedding reply is of another space")
    return result


def embed_texts(
    texts: Sequence[EmbeddingText],
    owner: OwnerEmbedding,
    *,
    view: AddReadView,
    provider: EmbeddingProvider,
    store: BatchStore,
) -> dict[TextKey, EmbeddingUseV1]:
    """Embed an owner's texts in stored batches and return where each vector is.

    Args:
        texts (Sequence[EmbeddingText]): The unique texts, in canonical order
            (`embedding_texts`).
        owner (OwnerEmbedding): The owner, its ticket, space, token and clock.
        view (AddReadView): The read view after settlement, for reused head vectors.
        provider (EmbeddingProvider): The single embedding provider.
        store (BatchStore): The Writer, for the create-once batches.

    Returns:
        dict[TextKey, EmbeddingUseV1]: Each text's batch entry.

    Raises:
        ProviderError: The provider failed; the owner pauses and a resume reuses
            every stored batch.
        QuoteMemError: ``embedding_space_mismatch`` for a reply or batch of another
            space; ``binding_mismatch`` for another owner's batch; or the
            owner's own error from ``owner.authorize``, before the batch it
            interrupts is stored.
    """
    if view.embedding_space != owner.space:
        raise QuoteMemError("embedding_space_mismatch", "The read view is of another space")
    located = _stored(owner, store)
    pending = [text for text in texts if (text.role, text.digest) not in located]
    for start in range(0, len(pending), MAX_EMBEDDING_BATCH):
        batch = pending[start:start + MAX_EMBEDDING_BATCH]
        owner.authorize()
        vectors: dict[int, list[float]] = {}
        for index, text in enumerate(batch):
            if (vector := _live_vector(text, view)) is not None:
                vectors[index] = vector
        called = [index for index in range(len(batch)) if index not in vectors]
        resolved = owner.model
        if called:
            with provider_attempts(owner.authorize):
                result = _embedded(owner, provider, [batch[index] for index in called])
            owner.authorize()
            resolved = result.resolved_model
            vectors.update(zip(called, result.vectors, strict=True))
        body = StatementEmbeddingV1(
            owner_run_id=owner.owner_run_id, embedding_space=owner.space,
            resolved_model=resolved, dimensions=owner.dimensions,
            entries=[StatementEmbeddingEntryV1(
                role=text.role, input_digest=text.digest, uses=sorted(text.uses),
                embedding_token=owner.token, vector=vectors[index],
            ) for index, text in enumerate(batch)],
        )
        reference = store.put_logical_artifact(
            owner.ticket, "semantic_embedding", body, security_level="restricted",
            created_at=owner.clock(),
        )
        located.update(((text.role, text.digest), EmbeddingUseV1(ref=reference, index=index))
                       for index, text in enumerate(batch))
    return {text.key: located[(text.role, text.digest)] for text in texts}

