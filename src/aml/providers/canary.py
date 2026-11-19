"""Fixed global embedding canaries and vector comparison from plan.md §11."""

from __future__ import annotations

import base64
import math
import struct
from typing import Literal

from aml.core.errors import AMLError
from aml.providers.contracts import EmbeddingResult

CANARY_TEXT_VERSION = "aml-canary-v1"
CANARY_TEXTS = (
    "A red bicycle stands beside the library.",
    "The river flows north beyond the old bridge.",
    "Please schedule the meeting for next Tuesday.",
    "She prefers tea without milk or sugar.",
    "The package arrived two days before the deadline.",
    "Three children are building a sand castle.",
    "A triangle has three sides and three corners.",
    "The engineer replaced a damaged power cable.",
    "This recipe requires flour, water, and salt.",
    "The train leaves from platform seven at noon.",
    "I have never visited the northern research station.",
    "If it rains tomorrow, the picnic will move indoors.",
    "The document describes how to reset the device.",
    "A quiet room makes careful reading easier.",
    "The blue notebook belongs to the mathematics teacher.",
    "Yesterday the team completed its final experiment.",
    "The first attempt failed, but the second succeeded.",
    "Keep the spare batteries in a dry container.",
    "The musician practiced a difficult passage slowly.",
    "Both reports describe the same historical event.",
    "The smallest box contains the heaviest object.",
    "A reliable clock is essential for this measurement.",
    "The garden receives sunlight only in the morning.",
    "The volunteer translated the letter into another language.",
    "Do not confuse the departure date with the arrival date.",
    "The customer changed the delivery address last week.",
    "Two independent witnesses confirmed the result.",
    "The old policy remains valid until the replacement is approved.",
    "The scientist recorded an unexpected decrease in temperature.",
    "A shared resource must be released after each use.",
    "The museum is closed on public holidays.",
    "The final answer depends on which assumption is correct.",
)


def encode_canary_vectors(result: EmbeddingResult) -> list[str]:
    """Freeze exactly 32 finite vectors as base64 little-endian float32 bytes."""
    if len(result.vectors) != len(CANARY_TEXTS):
        raise AMLError("schema_invalid", "A canary baseline requires exactly 32 vectors")
    try:
        encoded: list[str] = []
        for vector in result.vectors:
            raw = struct.pack(f"<{result.dimensions}f", *vector)
            if not any(struct.unpack(f"<{result.dimensions}f", raw)):
                raise AMLError("schema_invalid", "Canary vector underflows to zero in float32")
            encoded.append(base64.b64encode(raw).decode("ascii"))
        return encoded
    except (OverflowError, struct.error):
        raise AMLError("schema_invalid", "Canary values must be representable as float32") from None


def compare_canaries(
    baseline: list[list[float]], current: list[list[float]]
) -> tuple[Literal["stable", "drift_warning", "space_mismatch"], float]:
    """Compare vectors, never their digests; return the mean cosine deviation."""
    if len(baseline) != 32 or len(current) != 32:
        raise AMLError("schema_invalid", "Canary comparison requires both complete vector sets")
    deviations: list[float] = []
    for before, after in zip(baseline, current, strict=True):
        if len(before) != len(after) or len(before) not in (1024, 2048):
            raise AMLError("schema_invalid", "Canary dimensions differ")
        if any(not math.isfinite(value) for value in before + after):
            raise AMLError("schema_invalid", "Canary values must be finite")
        left = math.hypot(*before)
        right = math.hypot(*after)
        if left == 0 or right == 0:
            raise AMLError("schema_invalid", "Canary vectors must have nonzero norms")
        cosine = math.fsum((x / left) * (y / right) for x, y in zip(before, after, strict=True))
        deviations.append(max(0.0, min(2.0, 1.0 - cosine)))
    mean = math.fsum(deviations) / 32
    state = "space_mismatch" if mean > 2e-2 else "drift_warning" if mean > 1e-3 else "stable"
    return state, mean
