"""Extraction policy v2: the parameters every Add owner's extraction is frozen to.

Bootstrap releases one ``policy_config:extraction:v2`` config per workspace, and
each owner freezes it: the extract-v12 schema, template v50 and the shared
definitions block (by digest), the calendar-unit time reading, the pipeline, the
extraction and source-label call budgets, the unit, context, catalog and support
limits, the importance inputs, the nomination policy (by digest) and the re-entry
cap. Any change is a new policy release.

Only the two DeepSeek pipelines exist, the members of
``core/semantic.CURRENT_EXTRACTION_PIPELINES``. ``EXTRACTION_PIPELINES`` holds the
provider facts that each one binds; the judge takes its model, reasoning and
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

from quotemem.core.extraction_output import EXTRACT_V12_SCHEMA_ID, EXTRACTION_TEMPLATES
from quotemem.core.extraction_supplement import SupplementRule
from quotemem.core.extraction_template_agent import WORKFLOW_TEMPLATE_AGENT_V1_DIGEST
from quotemem.core.extraction_template_v50 import WORKFLOW_TEMPLATE_V50_DIGEST
from quotemem.core.identity import IntegrityDigest, integrity_digest
from quotemem.core.inference import ReasoningEffort
from quotemem.core.judge_timeout import JUDGE_FIRST_TIMEOUT_MS
from quotemem.core.memory_definitions import MEMORY_DEFINITIONS_DIGEST
from quotemem.core.model_identity import DEEPSEEK_FLASH_MODEL
from quotemem.core.policy import FixedNumber, Number01, from_fixed_point, to_fixed_point
from quotemem.core.provider_budget import (
    DEFAULT_EXTRACTION_CALL_BUDGET,
    LONG_EXTRACTION_CALL_BUDGET,
    ExtractionCallBudgetTrial,
    ProviderCallBudgetV2,
    extraction_call_budget,
    source_labels_call_budget,
)
from quotemem.core.semantic import (
    BENCH_DEEPSEEK_FLASH_PIPELINE_V48,
    DEEPSEEK_FLASH_PIPELINE_V48,
    ShortText,
)
from quotemem.core.statement import (
    MAX_CATALOG_ENTITIES,
    MAX_REENTRY,
    MAX_RELEVANCE_MEMORIES,
    MAX_RESERVED_MEMORIES,
    MAX_SUPPORT_GROUPS,
    MAX_TAIL_SOURCES,
    MAX_UNIT_FACTS,
)
from quotemem.store.metadata import OpaqueIdSpec, StrictModel

ExtractionContractVersion = Literal[48]
"""The extraction contract that a provider configuration may pin: v48.

Its number is the one that every current pipeline name carries (``extract-v48-``).
"""
DEFAULT_EXTRACTION_CONTRACT_VERSION: Final[ExtractionContractVersion] = 48
TEMPLATE_ALLOWLIST: Final = frozenset(EXTRACTION_TEMPLATES)
"""The extraction templates that a policy may name: template v50."""
MAX_EXTRACTION_CONCURRENCY: Final = 16
"""The cap on one owner's unit calls in flight, min(16, per_user)."""
MAX_UNIT_WORDS: Final = 16_000
"""The word bound of a unit, which also bounds the previous batch and the tail."""
MAX_UNIT_PARTS: Final = 2_000
"""The part bound of a unit, the bound that the RowOp doubling proof covers."""

JudgeMode = Literal["model", "local-lean-v1", "targeted-v1"]
"""How an owner reviews its members: ``model``, the memory judge model at four members
a request (the full Add mode); ``local-lean-v1``, deterministic local verdicts
(``ingestion/lean_judge.py``); ``targeted-v1``, the memory judge model for the members the
policy's ``judge_gate`` selects, packed in as few requests as the bounds allow
(``judge_input.BATCHED_PACKING``), and the local verdict for every other member
(``ingestion/judge_gate.py``). The ``agent`` Add mode releases ``targeted-v1`` with
``DEFAULT_JUDGE_GATE``."""

AddMode = Literal["full", "lean-local", "agent"]
"""The Add mode a deployment selects for a new workspace.

- ``full`` and ``lean-local`` share one extraction; ``full`` reviews every
  member with the memory judge model, ``lean-local``, the default, with the local judge.
  Both are the benchmark modes and send the supplement call and the session rollup.
- ``agent`` is the mode for a long-lived memory that an agent fills from live chat a few
  turns at a time, the mode of ``quotemem serve`` and of every memory
  plugin:

  - By default it reviews with the targeted judge (``targeted-v1``, ``AGENT_JUDGE_GATE``).
    A member that may repeat, update or correct a stored memory goes to the memory judge
    model; the rest take the local verdict. Without it, "the user moved to Berlin" leaves
    "the user lives in Paris" current, because the local judge needs an explicit "X, not
    Y", and a paraphrase said again in a later session becomes a second memory. A store may
    choose another review when it is created (``ReviewMode``: ``fast``, the local judge
    alone; ``full``, the model for every member); its other rules stay the same.
  - It sends no supplement call, so it states no overview, no assistant's-work summary and
    no implied health condition, and makes no session rollup: one extraction call an Add
    besides the judge calls the gate selects. Capturing every three turns would otherwise
    add two model calls and one overview memory per Add, and infer special-category data
    about a real user.
  - It binds the user's own first person to the user (``SELF_IDENTITY_RULES``).
  - It carries out in-chat forget requests like the other modes, and forgets completely
    (``explicit-request-v2``): nothing that states the forgotten fact stays readable.
  - It never grounds a factual statement about the user on a quote that is only a question
    (``no-question-support-v1``, ``QUOTE_GROUNDING``).
  - It extracts with template agent-v1 (``EXTRACTION_TEMPLATE_DIGESTS``): a value corrected
    within one batch gives only the latest value, each fact states one claim, and a question
    gives no fact.
"""

JUDGE_MODES: Final[Mapping[str, JudgeMode]] = MappingProxyType({
    "full": "model", "lean-local": "local-lean-v1", "agent": "targeted-v1",
})
"""The extraction policy's judge mode of each Add mode."""

EXTRACTION_CALL_BUDGETS: Final[Mapping[str, ExtractionCallBudgetTrial]] = MappingProxyType({
    "full": DEFAULT_EXTRACTION_CALL_BUDGET, "lean-local": LONG_EXTRACTION_CALL_BUDGET,
    "agent": DEFAULT_EXTRACTION_CALL_BUDGET,
})
"""The extraction call budget each Add mode releases when none is configured explicitly.

Only ``lean-local``, whose judge makes no model call, spends most of the Add's 1,740 s
on its extraction call (``LONG_EXTRACTION_CALL_BUDGET``). ``full``, although it shares
lean-local's extraction otherwise, keeps the 480/600-second budget: its judge model reads
after extraction, and 1,500 s of extraction would leave its 900 s first read at most
240 s of the Add. ``agent``'s targeted judge also reads after extraction, so it keeps the
same budget: a plugin batch is a few turns, and an extraction stuck for 25 minutes would
hold every later batch of the user behind it."""


class JudgeGateV1(StrictModel):
    """Which members the targeted judge mode sends to the memory judge model.

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

    On the LoCoMo calibration run (15,083 member inputs) at the default bars, 43% of members skip
    the model; recall of its update+correct verdicts 0.97, same+supplement 0.90.
    The bars are per mille, so the policy snapshot keeps them as integers.
    """

    head_cosine_permille: Annotated[int, Field(ge=0, le=1000)]
    sibling_cosine_permille: Annotated[int, Field(ge=0, le=1000)]
    cue_lexicon: Literal["change-cues-v1"]
    sibling_cues: bool = Field(default=False, exclude_if=lambda value: not value)
    """A change cue also counts with a same-Add sibling that shares one of the member's
    entities, not only with a stored memory (``AGENT_JUDGE_GATE``); False is left out of the
    body, so the default gate keeps its digest."""


DEFAULT_JUDGE_GATE: Final = JudgeGateV1(
    head_cosine_permille=855, sibling_cosine_permille=865, cue_lexicon="change-cues-v1",
)
"""The targeted judge mode's gate, fit on the LoCoMo calibration run's judge verdicts; the
selective-review research gates are derived from it."""

AGENT_JUDGE_GATE: Final = DEFAULT_JUDGE_GATE.model_copy(update={"sibling_cues": True})
"""The ``agent`` mode's gate: the default bars, and a change cue counts with a same-Add
sibling too. A plugin batch states a value and its change together
("我住在成都武侯区。" then "I live in Xiamen now"); under an ``agent`` release the change
carries no ``corrects_text`` (``extraction_admission.admit``'s ``plain_corrections``), so
without this the change reached only the local judge, which found the two independent and
left the old city current beside the new one."""

JUDGE_GATES: Final[Mapping[str, JudgeGateV1 | None]] = MappingProxyType({
    "full": None, "lean-local": None, "agent": AGENT_JUDGE_GATE,
})
"""The judge gate of each Add mode: the ``agent`` mode's, and None for every other mode."""

ReviewMode = Literal["fast", "standard", "full"]
"""How strongly an ``agent`` Add mode store reviews its new memories.

The review step decides whether a new memory repeats, updates or corrects a stored one.
Every review keeps the ``agent`` mode's other rules (in-chat forget, quote grounding, the
self-identity rule, no supplement call); only the judge differs:

- ``fast``: the local review rules of the benchmark modes (``local-lean-v1``). No
  judge model call; a change of state needs an explicit "X, not Y" to retire the old one.
- ``standard`` (the default): the targeted judge (``targeted-v1``, ``AGENT_JUDGE_GATE``).
  Only a member that may repeat, update or correct a stored memory goes to the memory judge
  model.
- ``full``: the memory judge model reviews every new memory (``model``).

The review is frozen into the store's extraction policy when the store is created: the
bootstrap installs one policy release, and a reopen with another one is refused. A store
moves to another review by ``quotemem rebuild``.
"""
DEFAULT_REVIEW: Final[ReviewMode] = "standard"
"""The review of an ``agent`` store that names none: the targeted judge."""
REVIEW_JUDGE_MODES: Final[Mapping[str, JudgeMode]] = MappingProxyType({
    "fast": "local-lean-v1", "standard": JUDGE_MODES["agent"], "full": "model",
})
"""The extraction policy's judge mode of each review (``ReviewMode``)."""
JUDGE_MODE_REVIEWS: Final[Mapping[str, ReviewMode]] = MappingProxyType({
    "local-lean-v1": "fast", "targeted-v1": "standard", "model": "full",
})
"""The review a judge mode carries out, so every store reports one (``lean-local`` is
``fast``, ``full`` is ``full``)."""

TimeReading = Literal["calendar-unit-v1"]
"""How time resolution reads a count of weeks, months or years from the source time
(``ingestion/temporal.py``): ``calendar-unit-v1`` reads a week count as its whole ISO
week, as "last week" is read, and a month or year count as its calendar month or year.
Every policy freezes it in its body."""

ChatForgetRule = Literal["explicit-request-v1", "explicit-request-v2"]
"""The in-chat forget rules a policy may freeze (``ingestion/forget_requests.py``).

Both find the same requests. ``explicit-request-v1`` (the benchmark modes) forgets the
memories the judge picks, keeps the memory recording the request and the request sentence,
and marks the sentences that still state the fact, which Search shows with a note.

``explicit-request-v2`` (the ``agent`` mode) also takes what still states the
forgotten fact (``service/forget_statement.py``):

- the memories recording this Add's requests ("The user asked to forget that ...") are
  forgotten with the fact, so a memory made in the same Add never outlives the request;
- the judge reads every readable sentence of the user's Evidence that shares words with the
  request or with what it forgets (assistant replies and the request sentence included), not
  only Search's hits, and the sentences it picks are withheld by the forget instead of
  marked;
- every other current memory that shares words with a forgotten statement is judged against
  it, and one the judge finds still stating the fact goes too, whatever its quotes;
- a memory left with no readable support, every quote withheld by a forget, is forgotten as
  well (also after a confirmed forget by statement), so new support cannot bring it back;
- what the user states again in their own words after the forget (no request, no question,
  dated after it, and showing no forgotten name) is not refused by its tombstone, and
  Search lifts its shadow terms for it; review judges it (``core/relearn.py``); every other
  restatement is still checked against the tombstone.
"""
CHAT_FORGET_COMPLETE: Final[ChatForgetRule] = "explicit-request-v2"
"""The in-chat forget rule that forgets completely (``ChatForgetRule``)."""

CHAT_FORGET_RULES: Final[Mapping[str, ChatForgetRule | None]] = MappingProxyType({
    "full": "explicit-request-v1", "lean-local": "explicit-request-v1",
    "agent": CHAT_FORGET_COMPLETE,
})
"""The in-chat forget rule of each Add mode; None carries out no in-chat
request. Every Add mode carries them out (``ingestion/forget_requests.py``,
``service/chat_forget.py``); the ``agent`` mode forgets completely (``explicit-request-v2``)."""

SelfIdentityRule = Literal["user-role-v1"]
"""How an Add binds the speaker of its user's own messages.

``user-role-v1`` is the ``agent`` Add mode's rule. A plugin sends only the turns of one
person, the user it remembers, so:

- A line label in a user-role message without a metadata speaker or participant
  (``更正一下：``, ``Quick update:``) that stays ``ambiguous`` after the label call reads
  as text, not as a turn of an unknown speaker. The message stays the authenticated user's,
  and its first person is the user. A label the call settles ``speaker`` still opens that
  speaker's turn, so a pasted transcript keeps its speakers
  (``ingestion/speaker_labels.py``).
- A fact whose primary is an unresolved singular first-person mention ("我", "I", "my")
  and whose every quote lies in the user's own turns is admitted as the user's
  (``ingestion/extraction_admission.py``, ``ingestion/first_person.py``).
- A re-entered ``identity_unresolved`` position still unresolved at its
  ``IDENTITY_SETTLE_AFTER``-th selection expires instead of waiting for ever
  (``ingestion/reentry.py``).

None (the benchmark modes, left out of the body so their digests are unchanged) keeps
the speaker-label rules: an ambiguous label blocks every inheritance of speaker identity."""

SELF_IDENTITY_RULES: Final[Mapping[str, SelfIdentityRule | None]] = MappingProxyType({
    "full": None, "lean-local": None, "agent": "user-role-v1",
})
"""The self-identity rule of each Add mode (``SelfIdentityRule``)."""
QuoteGroundingRule = Literal["no-question-support-v1"]
"""How admission grounds a statement about the user on its quotes, beyond the fact rules
(``ingestion/extraction_admission.py``).

``no-question-support-v1`` (the ``agent`` mode): only a user message that states
something grounds a factual statement about the user ("The user lives in Haidian,
Beijing."). Admission drops a support group whose user quotes are only questions ("Where do
I live", ``quote_question``), even beside the assistant's echo of the answer, and a group
quoting no user message, only the assistant's reply (``quote_unstated``), and refuses a
fact left without a group (``no_valid_group``). A group quoting a tool result or a document
grounds what it reports, beside a question. A statement that the user asked or
wondered something keeps such a group (the question is its evidence), as does one whose
subject is the assistant (what it says it did)."""
QUESTION_SUPPORT_RULE: Final[QuoteGroundingRule] = "no-question-support-v1"

QUOTE_GROUNDING: Final[Mapping[str, QuoteGroundingRule | None]] = MappingProxyType({
    "full": None, "lean-local": None, "agent": QUESTION_SUPPORT_RULE,
})
"""The quote grounding rule of each Add mode; None keeps the fact rules alone. The two benchmark
modes keep their pinned releases."""

EXTRACTION_SUPPLEMENTS: Final[Mapping[str, SupplementRule | None]] = MappingProxyType({
    "full": "overview-assistant-rollup-health-v4",
    "lean-local": "overview-assistant-rollup-health-v4",
    "agent": None,
})
"""The supplement call of each Add mode (``core/extraction_supplement.py``)
and the session rollup after it (``core/session_rollup.py``); None sends neither. ``full``
and ``lean-local`` send both beside each extraction call; the full mode's memory judge
reviews their members like any other. Rule v4 is the rollup rule v3 whose
supplement also states implied health conditions. ``agent`` sends neither (``AddMode``)."""

EXTRACTION_TEMPLATE_DIGESTS: Final[Mapping[str, IntegrityDigest]] = MappingProxyType({
    "full": WORKFLOW_TEMPLATE_V50_DIGEST, "lean-local": WORKFLOW_TEMPLATE_V50_DIGEST,
    "agent": WORKFLOW_TEMPLATE_AGENT_V1_DIGEST,
})
"""The extraction template each Add mode's policy names. The benchmark modes keep v50; the
``agent`` mode names agent-v1, v50 with its latest-value, atomic-statement and question rules
(``core/extraction_template_agent.py``)."""

Digest = Annotated[IntegrityDigest, OpaqueIdSpec("digest")]
LocalId = Annotated[ShortText, OpaqueIdSpec("local")]


@dataclass(frozen=True, slots=True)
class ExtractionPipeline:
    """The provider facts that one extraction pipeline binds.

    Attributes:
        profile (Literal["product", "bench"]): The deployment profile whose
            policy selects the pipeline.
        provider (Literal["openai"]): The transport: the OpenAI-compatible gateway.
        requested_model (str): The model alias that every request names.
        reasoning_effort (ReasoningEffort): The reasoning effort of every call
            that the pipeline binds: extraction, source labels and the judge.
        judge_read_timeout_ms (int): The judge's first read timeout; the
            owner's remaining time bounds each attempt further.
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
"""The pipelines and the provider facts of each; no other pipeline is accepted."""


def agent_release(policy: ExtractionPolicyConfigV2) -> bool:
    """Whether a frozen policy is an ``agent`` release with template agent-v1.

    Such an owner also admits a correction as a plain fact, without ``corrects_text``
    (``ingestion/extraction_admission.admit``'s ``plain_corrections``), resolves a state and an
    event of one Add as two memories (``core/resolution.resolve``'s ``kinds``), bounds its
    judge replies to the agent completion bound (``ingestion/judge_batch.AGENT_OUTPUT_TOKENS``)
    and logs every fact admission refuses (``ingestion/extract.log_refusals``). An ``agent``
    workspace released before agent-v1 keeps v50 and the earlier rules, so its replay is
    unchanged.
    """
    return policy.parameters.prompt_template_digest == WORKFLOW_TEMPLATE_AGENT_V1_DIGEST


def extraction_pipeline(pipeline_version: str) -> ExtractionPipeline:
    """Return the provider facts of a current extraction pipeline.

    Args:
        pipeline_version (str): A policy's ``pipeline_version``.

    Returns:
        ExtractionPipeline: The facts that every binding of the pipeline takes.

    Raises:
        ValueError: If the pipeline is not one of the pipelines.
    """
    pipeline = EXTRACTION_PIPELINES.get(pipeline_version)
    if pipeline is None:
        raise ValueError("Unknown extraction pipeline")
    return pipeline


class ImportanceWeightsV1(StrictModel):
    """The importance weights; they sum to exactly one.

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
    """The durability input of importance: a naming memory's, else by memory kind."""

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
    """The parameters that a config and its fixed-point snapshot hold alike.

    Each upper bound is the bound of what the parameter limits, so a
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
    self_identity: SelfIdentityRule | None = Field(
        default=None, exclude_if=lambda value: value is None,
    )
    """How the user's own messages bind their speaker (``SELF_IDENTITY_RULES``); None (left
    out of the body, so a mode without one keeps its digest) keeps the speaker-label rules."""
    quote_grounding: QuoteGroundingRule | None = Field(
        default=None, exclude_if=lambda value: value is None,
    )
    """How admission grounds a statement about the user on its quotes (``QUOTE_GROUNDING``);
    None (left out of the body, so a mode without one keeps its digest) adds no rule."""


class ExtractionPolicyV2(_ExtractionParametersV2):
    """Complete extraction parameters; its safety conditions cannot be disabled.

    One logical extraction call makes at most ``EXTRACTION_REQUEST_CAP`` (3)
    physical requests, network retries and output-schema regenerations
    together, all inside ``call_budget``. ``retry_output_schema``
    lets only an HTTP 2xx reply that is not JSON, breaks the strict schema or
    was cut off send the identical request again; an ``output_bound``
    fault never does. ``call_budget`` is exactly the call-budget preset that
    its ``policy_id`` names, and the source-label pre-pass binds
    ``source_labels_budget``, the label budget of that same preset.
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
    """The immutable ``policy_config:extraction:v2`` body."""

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
    self_identity: SelfIdentityRule | None = None,
    quote_grounding: QuoteGroundingRule | None = None,
    prompt_template_digest: IntegrityDigest = WORKFLOW_TEMPLATE_V50_DIGEST,
) -> ExtractionPolicyConfigV2:
    """Create the current extraction policy of a deployment profile.

    Every policy names the ``calendar-unit-v1`` time reading, and template v50 unless
    ``prompt_template_digest`` names another allowed template.

    Args:
        profile (Literal["product", "bench"]): Selects the product or the bench
            v48 pipeline; the two policies differ in nothing else.
        nomination_policy_digest (str): ``NominationPolicyV1.digest`` of the
            frozen nomination policy (``ingestion/nomination.py``).
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
        self_identity (SelfIdentityRule | None): The Add mode's self-identity rule
            (``SELF_IDENTITY_RULES``); None keeps the speaker-label rules.
        quote_grounding (QuoteGroundingRule | None): The Add mode's quote grounding rule
            (``QUOTE_GROUNDING``); None adds none.
        prompt_template_digest (IntegrityDigest): The Add mode's extraction template
            (``EXTRACTION_TEMPLATE_DIGESTS``); template v50 by default.

    Returns:
        ExtractionPolicyConfigV2: The complete config.

    Raises:
        ValueError: If the profile or the call-budget trial is unknown.
        ValidationError: If the nomination digest is not a SHA-256 hex digest.

    Examples:
        ```python
        from quotemem.ingestion.nomination import NominationPolicyV1

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
            "prompt_template_digest": prompt_template_digest,
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
            "self_identity": self_identity,
            "quote_grounding": quote_grounding,
        },
    })
