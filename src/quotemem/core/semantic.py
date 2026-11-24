"""Retained leaf value types: scopes, time, text bounds, vectors and support.

The NL (natural-language) memory model is in ``quotemem.core.entities`` and
``quotemem.core.statement``.
These primitives are the ones it and the copied infrastructure share: scope and
time types, the bounded text and unit types, the embedding vector, the time
expression and anchor of a fact, source spans and support-group validation.
"""

from __future__ import annotations

import math
from typing import Annotated, Final, Literal, Self

from pydantic import (
    AfterValidator,
    Field,
    model_validator,
)

from quotemem.core.identity import ULID
from quotemem.store.metadata import RefSpec, StrictModel

ScopeKind = Literal["user", "project", "agent", "organization"]
TimeGranularity = Literal["HOUR", "DAY", "MONTH", "YEAR", "unanchored"]
HistoricalTimestamp = Annotated[int, Field(ge=-(2**63), le=2**63 - 1)]
SystemTimestamp = Annotated[int, Field(ge=0, le=253402300799000)]
ShortText = Annotated[str, Field(max_length=1024)]
LongText = Annotated[str, Field(max_length=65535)]
UnitFloat = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
SecurityLevel = Literal["internal", "confidential", "restricted"]
TrustTier = Literal["verified", "reported", "inferred", "untrusted"]
DEEPSEEK_FLASH_PIPELINE_V48: Final = "extract-v48-gateway-deepseek-v4-1-flash-260910-medium"
BENCH_DEEPSEEK_FLASH_PIPELINE_V48: Final = (
    "extract-v48-bench-gateway-deepseek-v4-1-flash-260910-medium"
)
CURRENT_EXTRACTION_PIPELINES: Final = frozenset({
    DEEPSEEK_FLASH_PIPELINE_V48, BENCH_DEEPSEEK_FLASH_PIPELINE_V48,
})
"""The extraction pipelines, product and bench: the only pipeline identities."""


def validate_vector(vector: list[float]) -> list[float]:
    """Check the two admitted dimensions without calling an embedding provider."""
    # A finite total proves every element finite, and costs a tenth of scanning
    # them one at a time. Addition can also reach an infinity of its own, so a
    # non-finite total only sends the vector to the elementwise check that tells
    # a value that was never finite from one the summation produced.
    if len(vector) not in {1024, 2048} or (
        not math.isfinite(sum(vector)) and not all(math.isfinite(value) for value in vector)
    ):
        raise ValueError("Embedding must contain 1024 or 2048 finite values")
    return vector


EmbeddingVector = Annotated[list[float], AfterValidator(validate_vector)]


class NoTimeExpressionV1(StrictModel):
    """A proposal with no source time expression."""

    kind: Literal["none"]


class SourceTimeExpressionV1(StrictModel):
    """An absolute or relative expression preserved from source text."""

    kind: Literal["absolute", "relative"]
    text: Annotated[str, Field(min_length=1, max_length=1024)]


TimeExpressionV1 = Annotated[
    NoTimeExpressionV1 | SourceTimeExpressionV1, Field(discriminator="kind")
]


class AnchorV1(StrictModel):
    """Describe time anchoring without duplicating the source-span Evidence authority."""

    kind: Literal["source_observed_at", "absolute_source_text", "unanchored", "not_applicable"]
    source_time: HistoricalTimestamp | None = None

    @model_validator(mode="after")
    def check_source_time(self) -> Self:
        """Only an observed-source-time anchor carries its frozen timestamp."""
        if (self.kind == "source_observed_at") != (self.source_time is not None):
            raise ValueError("source_time is required only for source_observed_at")
        return self


class SourceSpanV1(StrictModel):
    """An original UTF-8 half-open span; the only semantic Evidence reference."""

    evidence_id: Annotated[ULID, RefSpec("evidence")]
    byte_start: Annotated[int, Field(ge=0)]
    byte_end: Annotated[int, Field(gt=0)]

    @model_validator(mode="after")
    def check_range(self) -> Self:
        """Require a nonempty half-open interval."""
        if self.byte_end <= self.byte_start:
            raise ValueError("Source span must be nonempty")
        return self


def validate_support_groups(
    groups: list[list[int]], size: int, *, allow_empty: bool = False
) -> None:
    """Validate canonical OR-of-AND groups and full coverage of a local index domain."""
    if size == 0 and allow_empty and groups == []:
        return
    if (
        not groups
        or groups != sorted(groups)
        or len({tuple(group) for group in groups}) != len(groups)
    ):
        raise ValueError("Support groups must be nonempty, sorted, and unique")
    covered: set[int] = set()
    for group in groups:
        if not group or group != sorted(set(group)):
            raise ValueError("Support group positions must be increasing and unique")
        if any(index < 0 or index >= size for index in group):
            raise ValueError("Support group position is outside its local domain")
        covered.update(group)
    if covered != set(range(size)):
        raise ValueError("Every support position must belong to a complete group")
