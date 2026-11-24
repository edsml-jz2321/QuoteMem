"""Potential content sensitivity of a source text.

Reception uses the shared classifier's base terms without admission floors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from quotemem.core.errors import QuoteMemError
from quotemem.core.semantic import SecurityLevel
from quotemem.core.sensitivity_rules import classify


@dataclass(frozen=True)
class SourceSecurity:
    """Potential content sensitivity, never inferred identity or authorization."""

    security_level: SecurityLevel
    sensitivity_classes: tuple[str, ...]


def classify_source_security(text: str, *, profile: Literal["product", "bench"]) -> SourceSecurity:
    """Classify source content without attributing any identity to its speaker.

    Args:
        text: Exact source text; admission floors do not affect reception.
        profile: Trusted deployment profile; model output cannot select it.

    Returns:
        Product restricted by default; bench confidential unless recognized
        sensitive topics require restricted. This does not replace secret rejection.

    Raises:
        QuoteMemError: If a caller supplies an unsupported profile.
    """
    if profile not in {"product", "bench"}:
        raise QuoteMemError("schema_invalid", "Source security profile is invalid")
    categories = tuple(sorted(classify((text,), include_floors=False)))
    return SourceSecurity(
        "restricted" if profile == "product" or categories else "confidential", categories
    )
