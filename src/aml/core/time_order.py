"""Explicit fact-time ordering carried by the existing frozen query options.

This is an ordering request, not a historical visibility filter or a rewrite
hint. Undated results keep their relevance slots; only dated results reorder.
No recording timestamp supplies a missing fact date.
Answer choices remain separate from explicit ordering instructions. The order
is honoured under the released recall policy (A.6.3 keeps ``time_order``); it is
never inferred from query text (D-24).
"""

from __future__ import annotations

from typing import Annotated, Final, Literal, get_args

from pydantic import ConfigDict, Field, JsonValue, TypeAdapter

from aml.core.recall_policy import RecallPolicySnapshot
from aml.core.semantic import HistoricalTimestamp
from aml.store.metadata import StrictModel

RECALL_POLICY_VERSION: Final[int] = get_args(
    RecallPolicySnapshot.model_fields["schema_version"].annotation,
)[0]
"""The released recall policy version, read from its snapshot model: explicit ordering
follows the current policy and never a retired one."""


class LatestTimeOrderV1(StrictModel):
    """Order eligible facts by descending occurrence/effective start time."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["latest"]


class NearestTimeOrderV1(StrictModel):
    """Order eligible facts by distance from one explicit UTC millisecond anchor."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["nearest"]
    at: HistoricalTimestamp


TimeOrderV1 = Annotated[
    LatestTimeOrderV1 | NearestTimeOrderV1, Field(discriminator="kind")
]
_TIME_ORDER: TypeAdapter[TimeOrderV1] = TypeAdapter(TimeOrderV1)


def read_time_order(
    options: dict[str, JsonValue] | list[str], policy_version: int,
) -> TimeOrderV1 | None:
    """Validate the reserved current option without normalizing frozen bytes.

    A present ``time_order`` must be one complete strict variant; null, unknown
    fields and an implicit nearest-date anchor are refused.
    """
    if policy_version != RECALL_POLICY_VERSION:
        raise ValueError("Explicit time ordering requires the released recall policy")
    if not isinstance(options, dict) or "time_order" not in options:
        return None
    return _TIME_ORDER.validate_python(options["time_order"], strict=True)


def freeze_query_options(
    options: tuple[str, ...] | None, time_order: TimeOrderV1 | None, policy_version: int,
) -> dict[str, JsonValue] | list[str]:
    """Keep answer choices verbatim, adding an envelope only for an explicit order.

    The new envelope is ``{"choices": [...], "time_order": {"kind": ...}}``.
    It preserves the answer choices separately from the retrieval instruction.
    The complete envelope is already covered by query and execution digests.
    """
    if policy_version != RECALL_POLICY_VERSION:
        raise ValueError("Explicit time ordering requires the released recall policy")
    choices = list(options or ())
    if time_order is None:
        return choices
    if type(time_order) not in (LatestTimeOrderV1, NearestTimeOrderV1):
        raise ValueError("Explicit time ordering requires a declared request model")
    # Reparse actual fields: a copied/subclassed DTO cannot hide additional
    # instructions or bypass strict timestamp validation through serialization.
    frozen: LatestTimeOrderV1 | NearestTimeOrderV1
    if type(time_order) is LatestTimeOrderV1:
        frozen = LatestTimeOrderV1.model_validate(dict(time_order.__dict__), strict=True)
    else:
        frozen = NearestTimeOrderV1.model_validate(dict(time_order.__dict__), strict=True)
    choice_values: list[JsonValue] = list(choices)
    return {"choices": choice_values, "time_order": frozen.model_dump(mode="json")}
