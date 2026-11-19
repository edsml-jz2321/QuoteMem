"""Extract the typed references the authoritative Writer validates (plan.md §7, §10).

This pure helper prepares references for the Writer. It neither reads
HeavenBase nor implements publication, ingestion or recovery workflows.
"""

from __future__ import annotations

from aml.core.errors import AMLError
from aml.store.metadata import StrictModel
from aml.store.registry import Reference, extract_references


def contextual_references(model: StrictModel) -> tuple[Reference, ...]:
    """Extract references, refusing a symbolic target.

    Every declared reference keeps its declared contract. A ``$`` target names
    a placeholder of an unresolved plan, which no written row may carry.

    Args:
        model: A strict validated model.

    Returns:
        References with their declared axes and original registry paths.

    Raises:
        AMLError: ``schema_invalid`` for a symbolic target.
    """
    result: list[Reference] = []
    for ref in extract_references(model):
        if ref.target_id.startswith("$"):
            raise AMLError("schema_invalid", "Reference requires its resolved target")
        result.append(ref)
    return tuple(result)
