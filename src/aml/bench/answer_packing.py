"""Frozen answer input policies shared by prompt rendering and provider dispatch."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class AnswerPackingPolicy(BaseModel):
    """Select legacy prefix packing or every returned content byte without a cap.

    V1 retains its original budget and wire fields. V2 has no local byte/token
    limit: actual provider capacity refusals remain explicit failed observations.
    """

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, hide_input_in_errors=True)

    strategy: Literal["ordered_content_utf8_prefix_v1", "all_returned_content_v2"]
    max_context_utf8_bytes: Annotated[int | None, Field(ge=2)] = Field(
        default=None, exclude_if=lambda value: value is None,
    )

    @model_validator(mode="after")
    def strategy_budget(self) -> Self:
        """Do not smuggle a replacement cap into full-returned-evidence packing."""
        if (self.strategy == "ordered_content_utf8_prefix_v1") != (
            self.max_context_utf8_bytes is not None
        ):
            raise ValueError("Only legacy prefix packing requires a context byte budget")
        return self


@dataclass(frozen=True)
class AMLBCut:
    """Complete items retained by the rank-order AMLB Answer prefix."""

    contents: tuple[str, ...]
    tokens: int
    cut_bound: bool


def amlb_cut(contents: Sequence[str], count_tokens: Callable[[str], int]) -> AMLBCut:
    """Take the first 100 items, then the prefix fitting 117,760 tokens.

    Args:
        contents: Ordered Search content, including its rendering wrappers.
        count_tokens: Counter from bench.tokenizer.checked_encoding; scripted in tests.

    Returns:
        Retained whole items and the token count of their newline-joined text.
        Stops at the first non-fitting item; never skips ahead to shorter hits.
    """
    kept: list[str] = []
    tokens = 0
    for content in contents[:100]:
        candidate_tokens = count_tokens("\n".join((*kept, content)))
        if candidate_tokens < 0:
            raise ValueError("Token count must be nonnegative")
        if candidate_tokens > 117_760:
            break
        kept.append(content)
        tokens = candidate_tokens
    return AMLBCut(tuple(kept), tokens, len(kept) < len(contents))
