"""Shared canonical bounds and content-free whole-output extraction refusals."""

from typing import Literal

from aml.core.errors import AMLError
from aml.core.identity import canonical_json
from aml.store.metadata import StrictModel

OUTPUT_CANONICAL_CHARS = 200_000
PROVIDER_PROPOSAL_CANONICAL_CHARS = 32_768
MAX_SOURCE_INDEX = 5_999
MAX_PROPOSALS = 512
MAX_USAGE = (1 << 63) - 1
METADATA_CANONICAL_CHARS = 256
INLINE_ARTIFACT_CHARS = 262_144


def canonical_limit(value: object, maximum: int, label: str) -> None:
    """Reject a value whose complete canonical representation exceeds its bound.

    Args:
        value: A JSON-compatible value or strict model, without conversion rules.
        maximum: Inclusive number of canonical Unicode characters.
        label: Code-owned sanitized field description; never source content.

    Returns:
        None: The complete representation fits the declared bound.

    Raises:
        AMLError: With schema_invalid when the entire value is too large.
    """
    if len(canonical_json(value)) > maximum:
        raise AMLError("schema_invalid", f"Extraction {label} exceeds its canonical bound")


RefusalRule = Literal["provider_schema_invalid", "truncated", "output_bound", "no_fresh_source"]


class ExtractionRefusalDetailV1(StrictModel):
    """Name the whole-output refusal of a reply that broke its frozen wire schema.

    Item faults never refuse an output: program admission records them
    (``ExtractionAdmissionV2``). The detail copies no model text.
    """

    rule: RefusalRule
