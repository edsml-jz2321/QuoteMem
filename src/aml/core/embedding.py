"""The single active embedding-state token projection in plan.md §5.9."""

from aml.core.controls import ActiveEmbeddingSpaceV1, ControlRowV1
from aml.core.errors import AMLError
from aml.core.identity import IntegrityDigest, integrity_digest


def embedding_state_token(control: ControlRowV1) -> IntegrityDigest:
    """Bind the active embedding authority without copying unrelated provenance.

    Args:
        control: Strict current policy:embedding_space Control row.

    Returns:
        The canonical token used by initialization and every active-space consumer.

    Raises:
        AMLError: If the Control is not the implemented active embedding variant.
    """
    value = control.value
    if control.key != "policy:embedding_space" or not isinstance(value, ActiveEmbeddingSpaceV1):
        raise AMLError("unsupported_variant", "Embedding token requires the active state")
    return integrity_digest(
        {
            "version": control.version,
            "state": value.state,
            "state_commit_seq": value.state_commit_seq,
            "embedding_space": value.embedding_space,
            "canary_digest": value.canary_digest,
        }
    )
