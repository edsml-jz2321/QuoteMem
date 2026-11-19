"""Extraction policy v2 (B.12): the parameters every Add owner's extraction is frozen to.

Bootstrap releases one ``policy_config:extraction:v2`` config per workspace, and
each owner freezes it: the extract-v12 schema, template v50 and the shared
definitions block (by digest), the calendar-unit time reading, the pipeline, the
extraction and source-label call budgets, the unit, context, catalog and support
limits, B.10's importance inputs, the nomination policy (by digest) and the re-entry
cap. Any change is a new policy release.

Only B.0's two DeepSeek pipelines exist, the members of
``core/semantic.CURRENT_EXTRACTION_PIPELINES``. ``EXTRACTION_PIPELINES`` holds the
provider facts that each one binds; C.5's judge takes its model, reasoning and
retries from the same frozen pipeline. This module never imports ingestion, so
the builder takes the digest of the frozen ``NominationPolicyV1`` as an argument.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from types import MappingProxyType
from typing import Annotated, Any, Final, Literal, Self

from pydantic import Field, model_validator

from aml.core.extraction_output import EXTRACT_V12_SCHEMA_ID, EXTRACTION_TEMPLATES
from aml.core.extraction_supplement import SupplementRule
from aml.core.extraction_template_v50 import WORKFLOW_TEMPLATE_V50_DIGEST
from aml.core.identity import IntegrityDigest, integrity_digest
from aml.core.inference import ReasoningEffort
from aml.core.judge_timeout import JUDGE_FIRST_TIMEOUT_MS
from aml.core.memory_definitions import MEMORY_DEFINITIONS_DIGEST
from aml.core.model_identity import DEEPSEEK_FLASH_MODEL
from aml.core.policy import FixedNumber, Number01, from_fixed_point, to_fixed_point
from aml.core.provider_budget import (
    DEFAULT_EXTRACTION_CALL_BUDGET,
    LONG_EXTRACTION_CALL_BUDGET,
    ExtractionCallBudgetTrial,
    ProviderCallBudgetV2,
    extraction_call_budget,
    source_labels_call_budget,
)
from aml.core.semantic import (
    BENCH_DEEPSEEK_FLASH_PIPELINE_V48,
    DEEPSEEK_FLASH_PIPELINE_V48,
    ShortText,
)
from aml.core.statement import (
    MAX_CATALOG_ENTITIES,
    MAX_REENTRY,
    MAX_RELEVANCE_MEMORIES,
    MAX_RESERVED_MEMORIES,
    MAX_SUPPORT_GROUPS,
    MAX_TAIL_SOURCES,
    MAX_UNIT_FACTS,
)
from aml.store.metadata import OpaqueIdSpec, StrictModel

ExtractionContractVersion = Literal[48]
"""The extraction contract that a provider configuration may pin: v48 (B.0).

Its number is the one that every current pipeline name carries (``extract-v48-``).
"""
DEFAULT_EXTRACTION_CONTRACT_VERSION: Final[ExtractionContractVersion] = 48
TEMPLATE_ALLOWLIST: Final = frozenset(EXTRACTION_TEMPLATES)
"""The extraction templates that a policy may name: template v50 (B.0, SYS-EXTRACT)."""
MAX_EXTRACTION_CONCURRENCY: Final = 16
"""B.3's cap on one owner's unit calls in flight, min(16, per_user)."""
MAX_UNIT_WORDS: Final = 16_000
"""B.3's word bound of a unit, which also bounds the previous batch and the tail."""
MAX_UNIT_PARTS: Final = 2_000
"""B.3's part bound of a unit, the bound that the RowOp doubling proof covers."""

JudgeMode = Literal["model", "local-lean-v1", "targeted-v1"]
"""How an owner reviews its members: ``model``, the memory judge model at four members
a request (the full Add mode); ``local-lean-v1``, deterministic local verdicts
(``ingestion/lean_judge.py``); ``targeted-v1``, the memory judge model for the members the
policy's ``judge_gate`` selects, packed in as few requests as the bounds allow
(``judge_input.BATCHED_PACKING``), and the local verdict for every other member
(``ingestion/judge_gate.py``). No Add mode releases ``targeted-v1``; the selective-review
research modes freeze it with their own gates."""

AddMode = Literal["full", "lean-local"]
"""The Add mode a deployment selects for a new workspace. Both share one extraction
(FULL-UNIFY); ``full`` reviews every member with the memory judge model, ``lean-local``,
the default, with the local judge."""

JUDGE_MODES: Final[Mapping[str, JudgeMode]] = MappingProxyType({
    "full": "model", "lean-local": "local-lean-v1",
})
"""The extraction policy's judge mode of each Add mode."""

EXTRACTION_CALL_BUDGETS: Final[Mapping[str, ExtractionCallBudgetTrial]] = MappingProxyType({
    "full": DEFAULT_EXTRACTION_CALL_BUDGET, "lean-local": LONG_EXTRACTION_CALL_BUDGET,
})
"""The extraction call budget each Add mode releases when none is configured explicitly.

Only ``lean-local``, whose judge makes no model call, spends most of the Add's 1,740 s
on its extraction call (``LONG_EXTRACTION_CALL_BUDGET``). ``full``, although it shares
lean-local's extraction otherwise, keeps the 480/600-second budget: its judge model reads
after extraction, and 1,500 s of extraction would leave its 900 s first read at most
240 s of the Add."""


class JudgeGateV1(StrictModel):
    """Which members the targeted judge mode sends to the memory judge model (TJUDGE).

    A member goes to the model when any rule holds; every other member takes the lean
    local verdict (``ingestion/lean_judge.py``). The rules read only the member input
    and the cosines nomination already has, so the gate makes no call
    (``ingestion/judge_gate.py``):

    - a current memory among its targets at cosine ``head_cosine_permille / 1000`` or
      more, or a same-Add sibling at ``sibling_cosine_permille / 1000`` or more;
    - a change cue (``cue_lexicon``) in its statement, or a correction the extractor
      flagged, while a target memory shares one of its entities;
    - an event whose shared-entity target memory is in plan or future form;
    - a sensitivity class, or any issued tombstone (nomination's own tombstone floors,
      lower than the head bar, already chose them).

    C69 arm A (LoCoMo, 15,083 member inputs) at the default bars: 43% of members skip
    the model; recall of A's update+correct verdicts 0.97, same+supplement 0.90.
    The bars are per mille, so the policy snapshot keeps them as integers.
    """

    head_cosine_permille: Annotated[int, Field(ge=0, le=1000)]
    sibling_cosine_permille: Annotated[int, Field(ge=0, le=1000)]
    cue_lexicon: Literal["change-cues-v1"]


DEFAULT_JUDGE_GATE: Final = JudgeGateV1(
    head_cosine_permille=855, sibling_cosine_permille=865, cue_lexicon="change-cues-v1",
)
"""The targeted judge mode's gate, fit on C69 arm A's judge verdicts (TJUDGE MEASURE); the
selective-review research gates are derived from it."""

TimeReading = Literal["calendar-unit-v1"]
"""How time resolution reads a count of weeks, months or years from the source time
(``ingestion/temporal.py``): ``calendar-unit-v1`` reads a week count as its whole ISO
week, as "last week" is read, and a month or year count as its calendar month or year.
Every policy freezes it in its body."""

ChatForgetRule = Literal["explicit-request-v1"]
"""The one in-chat forget rule a policy may freeze (``ingestion/forget_requests.py``)."""

CHAT_FORGET_RULES: Final[Mapping[str, ChatForgetRule | None]] = MappingProxyType({
    "full": "explicit-request-v1", "lean-local": "explicit-request-v1",
})
"""The in-chat forget rule of each Add mode (CHAT-FORGET); None carries out no in-chat
request. Both Add modes carry them out (``ingestion/forget_requests.py``,
``service/chat_forget.py``)."""

EXTRACTION_SUPPLEMENTS: Final[Mapping[str, SupplementRule | None]] = MappingProxyType({
    "full": "overview-assistant-rollup-health-v4",
    "lean-local": "overview-assistant-rollup-health-v4",
})
"""The supplement call of each Add mode (``core/extraction_supplement.py``)
and the session rollup after it (``core/session_rollup.py``); None sends neither. Both Add
modes send them beside each extraction call; the full mode's memory judge reviews their
members like any other. Rule v4 is the rollup rule v3 whose supplement
also states implied health conditions."""

Digest = Annotated[IntegrityDigest, OpaqueIdSpec("digest")]
LocalId = Annotated[ShortText, OpaqueIdSpec("local")]


@dataclass(frozen=True, slots=True)
class ExtractionPipeline:
    """The provider facts that one extraction pipeline binds (B.0, B.5, C.5).

    Attributes:
        profile (Literal["product", "bench"]): The deployment profile whose
            policy selects the pipeline.
        provider (Literal["openai"]): The transport: the OpenAI-compatible gateway.
        requested_model (str): The model alias that every request names.
        reasoning_effort (ReasoningEffort): The reasoning effort of every call
            that the pipeline binds: extraction, source labels and the judge.
        judge_read_timeout_ms (int): The judge's first read timeout; the
            owner's remaining time bounds each attempt further (C.5).
    """

    profile: Literal["product", "bench"]
    provider: Literal["openai"]
    requested_model: str
    reasoning_effort: ReasoningEffort
    judge_read_timeout_ms: int


EXTRACTION_PIPELINES: Final[Mapping[str, ExtractionPipeline]] = MappingProxyType({
    DEEPSEEK_FLASH_PIPELINE_V48: ExtractionPipeline(
        profile="product", provider="openai", requested_model=DEEPSEEK_FLASH_MODEL,
        reasoning_effort="medium", judge_read_timeout_ms=JUDGE_FIRST_TIMEOUT_MS,
    ),
    BENCH_DEEPSEEK_FLASH_PIPELINE_V48: ExtractionPipeline(
        profile="bench", provider="openai", requested_model=DEEPSEEK_FLASH_MODEL,
        reasoning_effort="medium", judge_read_timeout_ms=JUDGE_FIRST_TIMEOUT_MS,
    ),
})
"""B.0's pipelines and the provider facts of each; no other pipeline is accepted."""


def extraction_pipeline(pipeline_version: str) -> ExtractionPipeline:
    """Return the provider facts of a current extraction pipeline.

    Args:
        pipeline_version (str): A policy's ``pipeline_version``.

    Returns:
        ExtractionPipeline: The facts that every binding of the pipeline takes.

    Raises:
        ValueError: If the pipeline is not one of B.0's pipelines.
    """
    pipeline = EXTRACTION_PIPELINES.get(pipeline_version)
    if pipeline is None:
        raise ValueError("Unknown extraction pipeline")
    return pipeline


class ImportanceWeightsV1(StrictModel):
    """B.10's importance weights; they sum to exactly one.

    Resolution weighs a new revision's novelty, durability and actionability
    with them (``core/resolution.py`` ``importance``).
    """

    novelty: Number01
    durability: Number01
    actionability: Number01

    @model_validator(mode="after")
    def check_sum(self) -> Self:
        """Require an exact decimal sum of one."""
        weights = (self.novelty, self.durability, self.actionability)
        if sum(Decimal(str(weight)) for weight in weights) != 1:
            raise ValueError("Importance weights must sum exactly to one")
        return self


class DurabilityV1(StrictModel):
    """B.10's durability input of importance: a naming memory's, else by memory kind."""

    naming: Number01
    state: Number01
    event: Number01


class NormalizedImportanceWeightsV1(StrictModel):
    """The fixed-point representation of ``ImportanceWeightsV1``."""

    novelty: FixedNumber
    durability: FixedNumber
    actionability: FixedNumber


class NormalizedDurabilityV1(StrictModel):
    """The fixed-point representation of ``DurabilityV1``."""

    naming: FixedNumber
    state: FixedNumber
    event: FixedNumber


class _ExtractionParametersV2(StrictModel):
    """The B.12 parameters that a config and its fixed-point snapshot hold alike.

    Each upper bound is the A0.7 or B.3 bound of what the parameter limits, so a
    valid policy never asks for more than the extract-v12 wire carries.
    """

    prompt_template_digest: Digest
    definitions_digest: Digest
    source_labels_budget: ProviderCallBudgetV2
    extraction_concurrency: Annotated[int, Field(ge=1, le=MAX_EXTRACTION_CONCURRENCY)]
    pipeline_version: LocalId
    extractor_schema_id: LocalId
    overlap_messages: Annotated[int, Field(ge=0, le=MAX_TAIL_SOURCES)]
    tail_words: Annotated[int, Field(ge=1, le=MAX_UNIT_WORDS)]
    previous_batch_words: Annotated[int, Field(ge=1, le=MAX_UNIT_WORDS)]
    segment_words: Annotated[int, Field(ge=1, le=MAX_UNIT_WORDS)]
    max_parts_per_unit: Annotated[int, Field(ge=1, le=MAX_UNIT_PARTS)]
    max_facts_per_unit: Annotated[int, Field(ge=1, le=MAX_UNIT_FACTS)]
    context_limit: Annotated[int, Field(ge=0, le=MAX_RELEVANCE_MEMORIES)]
    reserved_limit: Annotated[int, Field(ge=0, le=MAX_RESERVED_MEMORIES)]
    catalog_limit: Annotated[int, Field(ge=1, le=MAX_CATALOG_ENTITIES)]
    support_limit: Annotated[int, Field(ge=1, le=MAX_SUPPORT_GROUPS)]
    nomination_policy_digest: Digest
    reentry_limit: Annotated[int, Field(ge=1, le=MAX_REENTRY)]
    call_budget: ProviderCallBudgetV2
    retry_output_schema: bool
    require_source_spans: Literal[True]
    source_time_only: Literal[True]
    sensitive_explicit_source_only: Literal[True]
    judge_mode: JudgeMode = Field(default="model", exclude_if=lambda value: value == "model")
    """The full mode's ``model`` is left out of the body, so its digest is unchanged."""
    judge_gate: JudgeGateV1 | None = Field(default=None, exclude_if=lambda value: value is None)
    """The targeted judge mode's gate, and only its (left out of every other body)."""
    time_reading: TimeReading
    """How time resolution reads week, month and year counts (``TimeReading``)."""
    chat_forget: ChatForgetRule | None = Field(default=None, exclude_if=lambda value: value is None)
    """The rule by which a completed Add carries out its user's in-chat forget requests
    (``CHAT_FORGET_RULES``); None (left out of the body, so a mode without one keeps its
    digest) carries out none."""
    extraction_supplement: SupplementRule | None = Field(
        default=None, exclude_if=lambda value: value is None,
    )
    """The supplement call sent beside each extraction call (``EXTRACTION_SUPPLEMENTS``);
    None (left out of the body, so a mode without one keeps its digest) sends none."""


class ExtractionPolicyV2(_ExtractionParametersV2):
    """Complete extraction parameters (B.12); its safety conditions cannot be disabled.

    One logical extraction call makes at most ``EXTRACTION_REQUEST_CAP`` (3)
    physical requests, network retries and output-schema regenerations
    together, all inside ``call_budget`` (B.5, INV:E16). ``retry_output_schema``
    lets only an HTTP 2xx reply that is not JSON, breaks the strict schema or
    was cut off send the identical request again (INV:E17); an ``output_bound``
    fault never does (Q6). ``call_budget`` is exactly the call-budget preset that
    its ``policy_id`` names, and the source-label pre-pass binds
    ``source_labels_budget``, the label budget of that same preset (B.3a).
    """

    importance_weights: ImportanceWeightsV1
    durability: DurabilityV1

    @model_validator(mode="after")
    def check_bindings(self) -> Self:
        """Bind a current pipeline, schema, template, definitions block and budgets."""
        extraction_pipeline(self.pipeline_version)
        if self.extractor_schema_id != EXTRACT_V12_SCHEMA_ID:
            raise ValueError("Unknown extraction output schema")
        if self.prompt_template_digest not in TEMPLATE_ALLOWLIST:
            raise ValueError("Unknown extraction prompt template")
        if self.definitions_digest != MEMORY_DEFINITIONS_DIGEST:
            raise ValueError("Unknown memory definitions")
        trial = self.call_budget.policy_id
        if self.call_budget != extraction_call_budget(trial):
            raise ValueError("Extraction call budget differs from its named preset")
        if self.source_labels_budget != source_labels_call_budget(trial):
            raise ValueError("Source-label budget differs from its extraction call budget")
        if (self.judge_mode == "targeted-v1") != (self.judge_gate is not None):
            raise ValueError("The targeted judge mode, and only it, freezes a judge gate")
        return self


class NormalizedExtractionPolicyV2(_ExtractionParametersV2):
    """The fixed-point representation of ``ExtractionPolicyV2``."""

    importance_weights: NormalizedImportanceWeightsV1
    durability: NormalizedDurabilityV1


class ExtractionPolicyConfigV2(StrictModel):
    """The immutable ``policy_config:extraction:v2`` body (A.3.12, B.12)."""

    policy_kind: Literal["extraction"]
    config_version: Literal[2]
    parameters: ExtractionPolicyV2


class ExtractionPolicySnapshotV2(StrictModel):
    """An owner's frozen extraction parameters, with numbers at six decimal places.

    ``config_digest`` is the integrity digest of the config that the snapshot
    normalizes, so every parameter is bound into it.
    """

    schema_version: Literal[2]
    config_digest: Digest
    parameters: NormalizedExtractionPolicyV2

    @model_validator(mode="after")
    def check_parameters(self) -> Self:
        """Reapply every range, binding and invariant of the complete policy."""
        ExtractionPolicyV2.model_validate(
            from_fixed_point(self.parameters.model_dump(), ExtractionPolicyV2),
        )
        return self


def normalize_extraction_policy(
    config: ExtractionPolicyConfigV2 | dict[str, Any],
) -> ExtractionPolicySnapshotV2:
    """Freeze a complete extraction config without supplying any missing field.

    Args:
        config (ExtractionPolicyConfigV2 | dict[str, Any]): The config, or its
            decoded JSON object.

    Returns:
        ExtractionPolicySnapshotV2: Its parameters at six decimal places, bound
            to the config's integrity digest.

    Raises:
        ValidationError: If the config is incomplete or breaks a policy rule.
    """
    parsed = ExtractionPolicyConfigV2.model_validate(config)
    return ExtractionPolicySnapshotV2.model_validate({
        "schema_version": 2,
        "config_digest": integrity_digest(parsed),
        "parameters": to_fixed_point(parsed.parameters.model_dump(mode="json")),
    })


def build_extraction_policy(
    profile: Literal["product", "bench"], *, nomination_policy_digest: str,
    call_budget_trial: ExtractionCallBudgetTrial | None = None,
    judge_mode: JudgeMode = "model",
    judge_gate: JudgeGateV1 | None = None,
    chat_forget: ChatForgetRule | None = None,
    extraction_supplement: SupplementRule | None = None,
) -> ExtractionPolicyConfigV2:
    """Create the current extraction policy of a deployment profile (B.12).

    Every policy names template v50 and the ``calendar-unit-v1`` time reading.

    Args:
        profile (Literal["product", "bench"]): Selects the product or the bench
            v48 pipeline; the two policies differ in nothing else.
        nomination_policy_digest (str): ``NominationPolicyV1.digest`` of the
            frozen nomination policy (C.4.1, ``ingestion/nomination.py``).
        call_budget_trial (ExtractionCallBudgetTrial | None): An explicit
            extraction call-budget trial; None selects
            ``extraction-wall-480-600-v2``. The source-label budget follows the
            trial (``source_labels_call_budget``).
        judge_mode (JudgeMode): ``model`` for the full Add mode, ``local-lean-v1``
            for the lean mode, whose judge makes no model call.
        judge_gate (JudgeGateV1 | None): The targeted judge mode's gate; required with
            ``targeted-v1`` and refused with every other judge mode.
        chat_forget (ChatForgetRule | None): The Add mode's in-chat forget rule
            (``CHAT_FORGET_RULES``); None carries out no in-chat request.
        extraction_supplement (SupplementRule | None): The Add mode's supplement call
            (``EXTRACTION_SUPPLEMENTS``); None sends none.

    Returns:
        ExtractionPolicyConfigV2: The complete config.

    Raises:
        ValueError: If the profile or the call-budget trial is unknown.
        ValidationError: If the nomination digest is not a SHA-256 hex digest.

    Examples:
        ```python
        from aml.ingestion.nomination import NominationPolicyV1

        config = build_extraction_policy(
            "bench", nomination_policy_digest=NominationPolicyV1().digest,
        )
        assert config.parameters.call_budget.policy_id == "extraction-wall-480-600-v2"
        ```
    """
    if profile not in {"product", "bench"}:
        raise ValueError("Unknown policy profile")
    trial = DEFAULT_EXTRACTION_CALL_BUDGET if call_budget_trial is None else call_budget_trial
    return ExtractionPolicyConfigV2.model_validate({
        "policy_kind": "extraction",
        "config_version": 2,
        "parameters": {
            "prompt_template_digest": WORKFLOW_TEMPLATE_V50_DIGEST,
            "definitions_digest": MEMORY_DEFINITIONS_DIGEST,
            "source_labels_budget": source_labels_call_budget(trial),
            "extraction_concurrency": MAX_EXTRACTION_CONCURRENCY,
            "pipeline_version": (
                DEEPSEEK_FLASH_PIPELINE_V48 if profile == "product"
                else BENCH_DEEPSEEK_FLASH_PIPELINE_V48
            ),
            "extractor_schema_id": EXTRACT_V12_SCHEMA_ID,
            "overlap_messages": 6,
            "tail_words": 600,
            "previous_batch_words": 16_000,
            "segment_words": 16_000,
            "max_parts_per_unit": 2_000,
            "max_facts_per_unit": 128,
            "context_limit": 30,
            "reserved_limit": 10,
            "catalog_limit": 64,
            "support_limit": 8,
            "importance_weights": {"novelty": 0.45, "durability": 0.35, "actionability": 0.2},
            "durability": {"naming": 1.0, "state": 0.85, "event": 0.5},
            "nomination_policy_digest": nomination_policy_digest,
            "reentry_limit": 64,
            "call_budget": extraction_call_budget(trial),
            "retry_output_schema": True,
            "require_source_spans": True,
            "source_time_only": True,
            "sensitive_explicit_source_only": True,
            "judge_mode": judge_mode,
            "judge_gate": None if judge_gate is None else judge_gate.model_dump(mode="json"),
            "time_reading": "calendar-unit-v1",
            "chat_forget": chat_forget,
            "extraction_supplement": extraction_supplement,
        },
    })
