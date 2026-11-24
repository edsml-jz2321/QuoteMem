"""Global content-free erasure receipt selection, separate from scoped cleanup."""

from typing import Annotated

from quotemem.core.identity import ArtifactId
from quotemem.store.metadata import RefSpec, StrictModel


class ErasureReceiptResultIndexV1(StrictModel):
    """Select only this global controller's exact prepare or final receipt."""

    erasure_receipt_ref: Annotated[
        ArtifactId, RefSpec("artifact", scope_rule="global", artifact_kinds=("receipt",)),
    ]
