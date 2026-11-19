"""Profile-specific provider settings and explicit credential selection."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Final, Literal
from urllib.parse import urlsplit

from pydantic import Field, PositiveInt, field_validator, model_validator

from aml.core.controls import PolicyTier
from aml.core.errors import AMLError
from aml.core.extraction_policy import (
    CHAT_FORGET_RULES,
    EXTRACTION_CALL_BUDGETS,
    EXTRACTION_SUPPLEMENTS,
    JUDGE_MODES,
    AddMode,
    ChatForgetRule,
    ExtractionContractVersion,
    JudgeMode,
)
from aml.core.extraction_supplement import SupplementRule
from aml.core.model_identity import DEEPSEEK_FLASH_MODEL
from aml.core.provider_budget import ExtractionCallBudgetTrial
from aml.store.metadata import StrictModel

EMBEDDING_MODEL = "qwen3.7-text-embedding"
DEFAULT_ADD_MODE: Final[AddMode] = "lean-local"


class AddTiersV1(StrictModel):
    """Per-user Add tiers: a user's later Adds run a faster Add mode.

    Tier Q, the default, is the workspace's ``add_mode``. A finalize owner whose frozen
    reception has reached either threshold runs tier F, ``fast_mode``. Both counts cover
    the owner's whole scope (one AMLB user) up to its frozen receipt cutoff, every
    session included, and come from reception metadata alone: AMLB sends no dataset
    name, so how much of the user has arrived is the only signal.

    Attributes:
        fast_mode (AddMode): Tier F's Add mode; it differs from the default mode.
        fast_from_chunk (PositiveInt | None): The 1-based Add (chunk) ordinal from which
            the user is tier F.
        fast_from_source_bytes (PositiveInt | None): The cumulative UTF-8 source bytes
            from which the user is tier F; English chat runs about six bytes a word.
    """

    fast_mode: AddMode
    fast_from_chunk: PositiveInt | None = Field(
        default=None, exclude_if=lambda value: value is None,
    )
    fast_from_source_bytes: PositiveInt | None = Field(
        default=None, exclude_if=lambda value: value is None,
    )

    @model_validator(mode="after")
    def some_threshold(self) -> AddTiersV1:
        """A tier without a threshold would never be selected."""
        if self.fast_from_chunk is None and self.fast_from_source_bytes is None:
            raise ValueError("Add tiers require a chunk or source-byte threshold")
        return self

    def tier(self, *, chunks: int, source_bytes: int) -> PolicyTier | None:
        """The tier of a reception of ``chunks`` Adds and ``source_bytes`` source bytes.

        Returns:
            PolicyTier | None: ``fast`` at or past either threshold; None is tier Q.
        """
        if self.fast_from_chunk is not None and chunks >= self.fast_from_chunk or (
            self.fast_from_source_bytes is not None
            and source_bytes >= self.fast_from_source_bytes
        ):
            return "fast"
        return None


class ProviderConfig(StrictModel):
    """Nonsecret configuration; no profile implicitly falls back to fake outputs."""

    profile: Literal["product", "bench", "development", "test"] = "product"
    embedding_dimensions: Literal[1024, 2048] = 1024
    model_studio_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    openai_base_url: str = "https://api.openai.com/v1"
    extraction_contract_version: ExtractionContractVersion | None = Field(
        default=None, exclude_if=lambda value: value is None,
    )
    extraction_call_budget_trial: ExtractionCallBudgetTrial | None = Field(
        default=None, exclude_if=lambda value: value is None,
    )
    add_mode: AddMode | None = Field(
        default=None, exclude_if=lambda value: value is None,
    )
    """The Add mode a new workspace's extraction policy is released with (``JUDGE_MODES``);
    None selects ``DEFAULT_ADD_MODE`` (``lean-local``); ``full`` must be explicit.
    Reopening must match the frozen policy; a changed default never rewrites it."""
    add_tiers: AddTiersV1 | None = Field(default=None, exclude_if=lambda value: value is None)
    """Per-user Add tiers; None runs every Add in ``add_mode``."""

    @field_validator("extraction_contract_version", mode="before")
    @classmethod
    def strict_contract_version(cls, value: object) -> object:
        """Do not coerce an explicit policy version from a boolean, float or text."""
        if value is not None and type(value) is not int:
            raise ValueError("Extraction contract version must be an exact integer")
        return value

    @field_validator("model_studio_base_url", "openai_base_url")
    @classmethod
    def safe_endpoint(cls, value: str) -> str:
        """Keep credentials outside endpoints and require explicit HTTPS service roots."""
        parsed = urlsplit(value)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("Provider endpoint must be an HTTPS URL without embedded credentials")
        return value.rstrip("/")

    @model_validator(mode="after")
    def valid_profile(self) -> ProviderConfig:
        """Reserve wider vectors for bench workspaces and call-budget trials for real profiles."""
        if self.profile != "bench" and self.embedding_dimensions != 1024:
            raise ValueError("Only a new bench workspace may select 2048 embedding dimensions")
        if self.profile not in {"product", "bench"} and (
            self.extraction_call_budget_trial is not None
        ):
            raise ValueError("Extraction wall-budget selection requires a product or bench profile")
        if self.profile not in {"product", "bench"} and (
            self.add_mode is not None or self.add_tiers is not None
        ):
            raise ValueError("Add mode selection requires a product or bench profile")
        if self.add_tiers is not None and self.add_tiers.fast_mode == (
            self.add_mode or DEFAULT_ADD_MODE
        ):
            raise ValueError("The fast Add tier must differ from the default Add mode")
        return self

    @property
    def judge_mode(self) -> JudgeMode:
        """The extraction policy's judge mode that this configuration releases."""
        return JUDGE_MODES[self.add_mode or DEFAULT_ADD_MODE]

    @property
    def extraction_call_budget(self) -> ExtractionCallBudgetTrial:
        """The extraction call budget a new workspace releases: the configured trial, else
        the Add mode's (``EXTRACTION_CALL_BUDGETS``)."""
        if self.extraction_call_budget_trial is not None:
            return self.extraction_call_budget_trial
        return EXTRACTION_CALL_BUDGETS[self.add_mode or DEFAULT_ADD_MODE]

    @property
    def chat_forget(self) -> ChatForgetRule | None:
        """The in-chat forget rule that this configuration releases, else None."""
        return CHAT_FORGET_RULES[self.add_mode or DEFAULT_ADD_MODE]

    @property
    def extraction_supplement(self) -> SupplementRule | None:
        """The supplement call that this configuration's policy release freezes, else None."""
        return EXTRACTION_SUPPLEMENTS[self.add_mode or DEFAULT_ADD_MODE]

    @property
    def inference_provider(self) -> Literal["openai"]:
        """Name the one chat transport, the OpenAI-compatible gateway (B.0).

        Embeddings alone use Model Studio.
        """
        return "openai"

    @property
    def inference_key_variable(self) -> str:
        """Select one credential name explicitly; never rotate or fall back between keys."""
        return "OPENAI_API_KEY"

    @property
    def inference_model(self) -> str:
        """Return the chat model of every current extraction pipeline (B.0).

        This is not an existing workspace's resolved extraction model. Its retained
        policy owns the complete binding through ``inference_binding_for_policy``.
        """
        return DEEPSEEK_FLASH_MODEL

    def validate_inference_endpoint(self) -> None:
        """Require an explicitly configured gateway before executing DeepSeek calls."""
        if self.inference_model == DEEPSEEK_FLASH_MODEL and (
            urlsplit(self.openai_base_url).hostname == "api.openai.com"
        ):
            raise AMLError(
                "provider_unavailable", "DeepSeek inference requires a gateway openai_base_url",
            )


@dataclass(frozen=True)
class ProviderCredentials:
    """In-memory secrets excluded from representations and persistent configuration."""

    model_studio: str = field(repr=False)
    openai: str | None = field(default=None, repr=False)


def load_credentials(config: ProviderConfig, environment: Mapping[str, str]) -> ProviderCredentials:
    """Validate only keys needed by the selected profile, without network access."""
    studio = environment.get("DASHSCOPE_API_KEY", "").strip()
    if not studio:
        raise AMLError("provider_unavailable", "Set DASHSCOPE_API_KEY for Model Studio")
    openai = environment.get(config.inference_key_variable, "").strip()
    if not openai:
        raise AMLError(
            "provider_unavailable", f"Set {config.inference_key_variable} for inference",
        )
    return ProviderCredentials(studio, openai)
