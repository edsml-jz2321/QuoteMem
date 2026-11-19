"""Mechanical admission of one extract-v12 reply (B.7) and its program-owned fields (B.10).

Admission is a pure function of the reply, the unit's frozen context and the
deployment profile, so replay recomputes it exactly. It applies mechanical
rules only, in the order of B.7: the whole output, the declarations, each fact
in reply order (its groups, statement, secret and sensitivity floor, user
attestation, anchors and fields), then in-reply duplicates and the 128-fact
cap. Language and referents are the judge's (C.5).

A fault of one item costs that item: a group, an anchor or a field is dropped
or masked, and a fact whose own rule fails is refused with its outcome code
(``extraction_item_invalid``, ``secret_rejected``,
``unsupported_sensitive_inference`` or ``extraction_capacity_exceeded``) and
gets no candidate row. Only a whole-output fault (B.7.1) fails the unit.

A fact that refers to the user, through the ``user`` anchor or through the
wording the template prescribes for the user (`USER_FORMS`) or a closed
translation of it (`TRANSLATED_USER_FORMS`), stands only when the unit renders
an authenticated-self source and, for that wording, the user is among the
anchors admission keeps; otherwise it is refused ``user_unattested`` (B.7.3), so
a statement the extractor misattributed to the user is never stored, in English
or in another language.

A declaration whose name no rendered source writes keeps its descriptor and
type without the name (B.7.2 ``name_unlocated``): its facts stay admitted
about an unnamed new entity, and the unlocated text never becomes a name.

An admitted fact carries every field the program derives (B.10): validity
from its time expression and source time, origin and trust tier from its cited
Evidence, and security and sensitivity (D.8.1). Its importance inputs come from
the owner's frozen policy when its ``ProposalV2`` is built (``core/proposal.py``).
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence, Set
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Final, Literal, cast, get_args

from aml.core.add_read_view import EvidenceRecord
from aml.core.entities import (
    USER,
    EntityChoiceV1,
    ExistingChoiceV1,
    LocalKeyRefV1,
    MemoryKind,
    NameKind,
    NewEntityChoiceV1,
    Origin,
    ParticipantChoiceV1,
    QuoteSpanV1,
    ReasonCode,
    SensitivityBasis,
    UnresolvedChoiceV1,
    UserChoiceV1,
)
from aml.core.errors import AMLError
from aml.core.extraction_bounds import PROVIDER_PROPOSAL_CANONICAL_CHARS
from aml.core.extraction_output import (
    NEW_ENTITY_HANDLES,
    UNRESOLVED_HANDLES,
    ExtractOutputV12,
    FactV12,
    NamingV12,
    NewEntityV12,
    SupportGroupV12,
)
from aml.core.extraction_records import (
    AnchorNoteV1,
    AnchorRule,
    DeclarationNoteV1,
    DeclarationRule,
    ExtractionAdmissionV2,
    ExtractionRefused,
    FactNoteV1,
    FactRule,
    FieldName,
    FieldNoteV1,
    FieldRule,
    GroupNoteV1,
    GroupRule,
    MergeV1,
)
from aml.core.extraction_template_v50 import WORKFLOW_EXTRACTION_TEMPLATE_V50
from aml.core.identity import canonical_json
from aml.core.name_forms import name_located, name_pattern
from aml.core.naming_syntax import NEUTRAL_KINDS
from aml.core.proposal import Concern
from aml.core.quote_location import locate_quote
from aml.core.recall_corpora import sparse_terms
from aml.core.run_inputs import InputMessageV1
from aml.core.semantic import (
    AnchorV1,
    NoTimeExpressionV1,
    SecurityLevel,
    SourceTimeExpressionV1,
    TimeExpressionV1,
    TimeGranularity,
    TrustTier,
)
from aml.core.sensitivity import SensitivityClass
from aml.core.sensitivity_rules import (
    PROTECTED_CLASSES,
    Profile,
    classify,
    security_level,
    stored_entity_type,
)
from aml.core.statement import (
    MAX_ANCHORS,
    MAX_CORRECTS_TEXT_CHARS,
    MAX_DESCRIPTOR_CHARS,
    MAX_ENTITY_TYPE_CHARS,
    MAX_GROUP_QUOTES,
    MAX_GROUP_SPANS,
    MAX_KEYWORD_CHARS,
    MAX_KEYWORDS,
    MAX_MEMORY_SPANS,
    MAX_MENTION_CHARS,
    MAX_NAMING,
    MAX_QUOTE_CHARS,
    MAX_REFERENT_CANDIDATES,
    MAX_REVISES,
    MAX_SENSITIVITY_FINDINGS,
    MAX_SUPPORT_GROUPS,
    MAX_TIME_TEXT_CHARS,
    MAX_UNIT_FACTS,
    canonical_statement,
    is_name,
    name_norm,
    statement_digest,
    token_contained,
    token_seq,
)
from aml.ingestion.alias_naming import normalize_alias_fact
from aml.ingestion.boundary import contains_recognized_credential
from aml.ingestion.catalog import CatalogEntry
from aml.ingestion.context import ExtractionContext, RenderedBinding
from aml.ingestion.forget_requests import forget_requests
from aml.ingestion.sources import RenderedSource
from aml.ingestion.temporal import resolve_time

REFUSALS: Final[dict[FactRule, ReasonCode]] = {
    "no_valid_group": "extraction_item_invalid",
    "statement_invalid": "extraction_item_invalid",
    "kind_invalid": "extraction_item_invalid",
    "fact_bound": "extraction_item_invalid",
    "secret": "secret_rejected",
    "sensitive_inference": "unsupported_sensitive_inference",
    "user_unattested": "extraction_item_invalid",
    "fact_count_exceeded": "extraction_capacity_exceeded",
}
"""The outcome code of each refused fact (B.7.3, C.10); a refused fact has no row."""

_PRESCRIBED_USER_FORM: Final = re.compile(r'the authenticated user is "([^"]+)"')
_IMPLIED_HEALTH: Final = re.compile(r'The user likely has ([^"()]+) \(implied by: "([^"]+)"\)\.?')
"""The supplement's implied health statement; its groups are the condition it names and the
quote it copies."""
_IMPLIED_CONDITION: Final = re.compile(
    r"(?<!\w)(?:"
    # what names a condition as such
    r"conditions?|disorders?|diseases?|syndromes?|illness(?:es)?|infections?|injur(?:y|ies)"
    r"|deficienc(?:y|ies)|deficient|intoleran(?:ce|t)|allerg(?:y|ies|ic)"
    r"|(?:low|high)\s+(?:vitamin\s+\w+|iron|ferritin|b12|potassium|magnesium|calcium"
    r"|blood\s+(?:pressure|sugar)|cholesterol)"
    # mental-health states
    r"|anxiety|anxious|panic|phobias?|depress(?:ion|ive)|burn-?out|chronic\s+stress|insomnia|ptsd"
    r"|ocd|adhd|autis(?:m|tic)|bipolar|eating\s+disorder|seasonal\s+affective"
    # common conditions and symptoms the floor's terms leave out
    r"|asthma|eczema|psoriasis|acne|rosacea|migraines?|headaches?|arthritis|tendinitis|sciatica"
    r"|scoliosis|diabet(?:es|ic)|hypertension|hypothyroidism|thyroid|anaemi(?:a|c)|anemi(?:a|c)"
    r"|coeliac|celiac|ibs|irritable\s+bowel|acid\s+reflux|reflux|gerd|heartburn|ulcers?"
    r"|myopia|(?:near|far|short|long)-?sighted(?:ness)?|astigmatism|glaucoma|cataracts?"
    r"|colou?r\s*blind(?:ness)?|hearing\s+loss|tinnitus|sleep\s+apnea|(?:back|joint|neck)\s+pain"
    r"|fatigue|dyslexi(?:a|c)|epilep(?:sy|tic)|seizures?|obesity|pcos|endometriosis"
    r"|pregnan(?:t|cy)|hay\s+fever|pollen|sinusitis|lactose|gluten"
    # named injuries
    r"|sprain(?:s|ed)?|fractur(?:e|es|ed)|torn\s+(?:ligament|acl|meniscus|tendon)s?"
    r"|broken\s+(?:bone|arm|leg|wrist|ankle|foot|hand|rib)s?"
    r")(?!\w)",
    re.IGNORECASE,
)
"""Words that name a health condition in an implied statement's ``<condition>``.

The floor's health terms (``core/sensitivity_rules``) name few conditions (diabetes,
asthma, depression), so without this lexicon a grounded implied statement (anxiety,
lactose intolerance, low vitamin D, burnout) would be refused. This lexicon is read
only for the implied form, only on its condition, and only lets a condition stand that
the copied quote implies; ``a liking for galleries`` still names none."""
_IMPLIED_MARK: Final = "(implied by:"
"""The words that mark a supplement statement as an implied inference."""
_IMPLIED_EXEMPT: Final = frozenset[SensitivityClass]({"health"})
"""The one protected class an implied statement may name without a quote stating it."""


def implied_health(statement: str, quoted: Iterable[str]) -> bool:
    """Whether a statement is the supplement's implied health form copying one of its quotes.

    Args:
        statement (str): The canonical statement.
        quoted (Iterable[str]): The located quotes of the fact's support groups.

    Returns:
        bool: True when the statement reads ``The user likely has <condition> (implied
        by: "<quote>").`` and ``<quote>`` lies within one located quote.
    """
    match = _IMPLIED_HEALTH.fullmatch(statement)
    return match is not None and any(match.group(2) in quote for quote in quoted)


def _implied_grounded(
    statement: str, stated: Set[SensitivityClass], quoted: Iterable[str],
) -> bool:
    """Whether an implied health statement may stand on its quotes.

    It must name a health class, by the floor's terms or by a condition word of its
    ``<condition>`` (`_IMPLIED_CONDITION`), or it implies nothing the floor would hold
    back and is no health condition; and no quote of it may ask to forget
    (``forget_requests``): the words "Please forget that I have mild lactose intolerance"
    withdraw the condition, they never imply it.
    """
    match = _IMPLIED_HEALTH.fullmatch(statement)
    named = "health" in stated or (
        match is not None and _IMPLIED_CONDITION.search(match.group(1)) is not None
    )
    return named and not any(
        forget_requests([InputMessageV1(role="user", content=quote)]) for quote in quoted
    )


def user_forms(template: str) -> tuple[str, ...]:
    """Return the wording an extraction template prescribes for the user in statements.

    Template v50 states its rule once, in its own words: the authenticated user
    is "the user". The forms are read from the frozen template text, so the
    ``user_unattested`` guard follows exactly the user wording the extractor is
    told to write; `TRANSLATED_USER_FORMS` adds the closed translations a statement
    the extractor writes in another language uses instead.

    Args:
        template (str): The frozen extraction template text.

    Returns:
        tuple[str, ...]: The prescribed forms in template order, without repeats.

    Raises:
        ValueError: If the template prescribes no form for the user.
    """
    forms = tuple(dict.fromkeys(_PRESCRIBED_USER_FORM.findall(template)))
    if not forms:
        raise ValueError("The extraction template prescribes no wording for the user")
    return forms


USER_FORMS: Final = user_forms(WORKFLOW_EXTRACTION_TEMPLATE_V50)
"""Template v50's wording for the user (B.7.3 ``user_unattested``): ``("the user",)``."""

TRANSLATED_USER_FORMS: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType({
    "es": ("usuario", "usuaria"),
    "pt": ("usuário", "usuária", "utilizador", "utilizadora"),
    "fr": ("utilisateur", "utilisatrice"),
    "it": ("utente",),
    "de": ("nutzer", "nutzers", "nutzerin", "benutzer", "benutzers", "benutzerin",
           "anwender", "anwenders", "anwenderin"),
    "el": ("χρήστης", "χρήστρια"),
    "ru": ("пользователь", "пользовательница"),
    "ar": ("المستخدم", "المستخدمة"),
    "hi": ("उपयोगकर्ता", "यूज़र", "यूजर"),
    "hinglish": ("upayogakarta", "upyogkarta"),
    "th": ("ผู้ใช้",),
    "ko": ("사용자",),
    "ja": ("ユーザー", "ユーザ", "利用者"),
    "zh": ("用户", "用戶", "使用者"),
    "tr": ("kullanıcı", "kullanıcının", "kullanıcıya", "kullanıcıyı", "kullanıcıda",
           "kullanıcıdan", "kullanıcıyla"),
    "vi": ("người dùng",),
    "id": ("pengguna",),
    "bn": ("ব্যবহারকারী", "ব্যবহারকারীর", "ব্যবহারকারীকে"),
    "ta": ("பயனர்", "பயனரின்"),
    "lo": ("ຜູ້ໃຊ້", "ຜູ້ໃຊ້ງານ"),
    "km": ("អ្នកប្រើ", "អ្នកប្រើប្រាស់"),
    "my": ("အသုံးပြုသူ",),
})
"""The closed translations of the user wording, per language (B.7.3, ML2, C17).

Statements are English (B.6.3), but admission keeps a statement written in another
language, so a statement that names the user only as, say, "el usuario" or "用户"
must face ``user_unattested`` as "the user" does. Each form is the word that
translates "user" in a statement of that language, written as the extractor writes
it there, and is matched as a token run (A0.6), never as a prefix: German and Turkish
forms are listed whole, while a Russian or Arabic word is also matched in its closed
case endings and proclitics (``пользователя``, ``للمستخدم``), the forms under which ML-1
locates a name (`aml.core.name_forms`). The table covers the credential lexicon's
sixteen languages or variants and attested Bengali, Tamil, Lao, Khmer and Burmese
forms. Hinglish lists Romanized Hindi words; bare English "user" is not added, so
English wording keeps the template's "the user" rule. A word for "user" elsewhere
in a statement errs toward refusing, never toward storing. C24 adds German
Anwender singular forms and Greek χρήστης/χρήστρια with their closed singular
cases. Pronouns do not identify authenticated self and remain outside this table.
"""

_USER_PATTERNS: Final = tuple(
    pattern
    for form in (*USER_FORMS, *(form for forms in TRANSLATED_USER_FORMS.values()
                                for form in forms))
    if (pattern := name_pattern(form)) is not None
)


def uses_user_wording(statement: str) -> bool:
    """Tell whether a statement uses a wording for the user (B.7.3).

    A form of `USER_FORMS` or of `TRANSLATED_USER_FORMS` counts when it is
    token-contained in the statement (A0.6), a Russian or Arabic token also in its
    closed inflected forms (`aml.core.name_forms`), so ``The user's dog`` uses ``the
    user`` while ``the users`` and ``the username`` do not, and ``Rowan es el marido
    del usuario`` uses ``usuario``.

    Args:
        statement (str): A statement.

    Returns:
        bool: True when some user form is token-contained in it.
    """
    joined = "\0" + "\0".join(token_seq(statement)) + "\0"
    return any(pattern.search(joined) is not None for pattern in _USER_PATTERNS)


def refers_to_user(fact: FactV12) -> bool:
    """Tell whether a fact refers to the user (B.7.3 ``user_unattested``).

    It does when the ``user`` handle is among its anchors (the first 8 distinct
    handles, as B.7.3's anchor rows keep them) or its statement uses a wording for
    the user, the template's or a closed translation of it (`uses_user_wording`).

    Args:
        fact (FactV12): A reply fact.

    Returns:
        bool: True when the fact refers to the user.
    """
    return _user_anchored(fact) or uses_user_wording(fact.statement)


def _user_anchored(fact: FactV12) -> bool:
    return USER in list(dict.fromkeys(fact.anchors))[:MAX_ANCHORS]


_KINDS: Final = frozenset(get_args(MemoryKind))
_NAME_KINDS: Final = frozenset(get_args(NameKind))
_TIME_KINDS: Final = frozenset({"none", "absolute", "relative"})
_REASONS: Final = frozenset({"ambiguous", "insufficient_context"})
_CATEGORIES: Final = frozenset(get_args(SensitivityClass))
_TRUST: Final[dict[TrustTier, int]] = {"untrusted": 0, "inferred": 1, "reported": 2, "verified": 3}


@dataclass(frozen=True, slots=True)
class AdmittedNaming:
    """A naming annotation that admission kept (``ProposalV2.naming``).

    Attributes:
        anchor_index (int): The named anchor's index in the fact's anchors.
        name (str): The name as written.
        name_kind (NameKind): ``name``, ``alias`` or ``former``.
    """

    anchor_index: int
    name: str
    name_kind: NameKind


@dataclass(frozen=True, slots=True)
class AdmittedTime:
    """A fact's time and the validity the program anchors it to (B.10).

    Attributes:
        expression (TimeExpressionV1): The source time expression, or ``none``.
        valid_from (int | None): Start of validity in epoch milliseconds.
        valid_to (int | None): End of validity; only events have one.
        granularity (TimeGranularity): The precision of the anchored window.
        anchored_from (AnchorV1): How the window was anchored.
    """

    expression: TimeExpressionV1
    valid_from: int | None
    valid_to: int | None
    granularity: TimeGranularity
    anchored_from: AnchorV1


NO_TIME: Final = AdmittedTime(
    NoTimeExpressionV1(kind="none"), None, None, "unanchored", AnchorV1(kind="not_applicable"),
)


@dataclass(frozen=True, slots=True)
class RevisedMemory:
    """A memory a fact's ``revises`` names, resolved from its handle.

    Attributes:
        memory_id (str): The Item.
        revision_id (str): The head shown, or the earlier version for an
            ``earlier`` handle (a correction of history, PD:P8).
        earlier (bool): The handle named an earlier version.
    """

    memory_id: str
    revision_id: str
    earlier: bool


@dataclass(frozen=True, slots=True)
class AdmittedFact:
    """One admitted position of a reply, with every program-owned field (B.7, B.10).

    Attributes:
        position (int): Its index in the reply's facts.
        statement (str): The canonical statement.
        statement_digest (str): Its digest (A0.6).
        kind (MemoryKind): ``state`` or ``event``.
        anchors (tuple[EntityChoiceV1, ...]): The primary anchor first; only the
            primary may be unresolved.
        naming (tuple[AdmittedNaming, ...]): Its kept naming annotations.
        groups (tuple[tuple[QuoteSpanV1, ...], ...]): Its admitted support
            groups in reply order, each its distinct located spans in quote
            order.
        time (AdmittedTime): Its time and validity.
        keywords (tuple[str, ...]): Its canonical keywords.
        revises (tuple[RevisedMemory, ...]): The memories it may revise.
        corrects_text (str | None): The repaired assertion no memory shows.
        security_level (SecurityLevel): Its security level.
        sensitivity_classes (tuple[SensitivityClass, ...]): Sorted classes.
        sensitivity_basis (tuple[SensitivityBasis, ...]): Non-empty exactly
            when classes are.
        origin (Origin): From its cited Evidence kinds.
        trust_tier (TrustTier): The lowest tier of its cited Evidence.
        concerns (frozenset[Concern]): Admission's concerns.
        duplicate_of (int | None): The earlier position an in-reply duplicate
            merges into (B.7.3 ``duplicate_in_reply``).
    """

    position: int
    statement: str
    statement_digest: str
    kind: MemoryKind
    anchors: tuple[EntityChoiceV1, ...]
    naming: tuple[AdmittedNaming, ...]
    groups: tuple[tuple[QuoteSpanV1, ...], ...]
    time: AdmittedTime
    keywords: tuple[str, ...]
    revises: tuple[RevisedMemory, ...]
    corrects_text: str | None
    security_level: SecurityLevel
    sensitivity_classes: tuple[SensitivityClass, ...]
    sensitivity_basis: tuple[SensitivityBasis, ...]
    origin: Origin
    trust_tier: TrustTier
    concerns: frozenset[Concern]
    duplicate_of: int | None

    @property
    def spans(self) -> tuple[QuoteSpanV1, ...]:
        """Every distinct located span of its groups, in group order."""
        unique = {span_key(span): span for group in self.groups for span in group}
        return tuple(unique.values())

    @property
    def evidence_ids(self) -> tuple[str, ...]:
        """The Evidence parts it cites, sorted."""
        return tuple(sorted({span.evidence_id for span in self.spans}))


@dataclass(frozen=True, slots=True)
class RefusedFact:
    """A reply position admission refused: it gets no candidate row (B.7.5).

    Attributes:
        position (int): Its index in the reply's facts.
        rule (FactRule): The rule that refused it.
        code (ReasonCode): Its disposition (C.10).
    """

    position: int
    rule: FactRule
    code: ReasonCode


@dataclass(frozen=True, slots=True)
class Admission:
    """The result of admitting one reply.

    Attributes:
        facts (tuple[AdmittedFact, ...]): Admitted positions in reply order,
            in-reply duplicates included.
        refused (tuple[RefusedFact, ...]): Refused positions in reply order.
        record (ExtractionAdmissionV2 | None): The admission record; None when
            admission changed nothing.
    """

    facts: tuple[AdmittedFact, ...]
    refused: tuple[RefusedFact, ...]
    record: ExtractionAdmissionV2 | None


def _joined(tokens: Sequence[str]) -> str:
    # No token contains NUL, so a contiguous token match is a substring match.
    return "\0" + "\0".join(tokens) + "\0"


@dataclass(frozen=True, slots=True)
class _Texts:
    """Token forms of every readable stretch of the context's sources."""

    by_source: dict[str, tuple[str, ...]]
    raw_texts: tuple[str, ...]

    @staticmethod
    def of(sources: Sequence[RenderedSource]) -> _Texts:
        raw = {source.source_id: tuple(source.readable_texts()) for source in sources}
        return _Texts(
            {source_id: tuple(_joined(token_seq(text)) for text in texts)
             for source_id, texts in raw.items()},
            tuple(text for texts in raw.values() for text in texts),
        )

    def contains(self, needle: str, source_ids: Iterable[str] | None = None) -> bool:
        """Token containment (A0.6) of a needle in the readable text of some source."""
        tokens = token_seq(needle)
        if not tokens:
            return False
        joined = _joined(tokens)
        ids = self.by_source if source_ids is None else source_ids
        return any(joined in text for source_id in ids for text in self.by_source[source_id])

    def locates_name(self, name: str) -> bool:
        """Whether a name occurs in some readable text, each token whole or in one of its
        closed inflected forms (`aml.core.name_forms`, ML-1)."""
        return name_located(name, self.raw_texts)


@dataclass(slots=True)
class _Notes:
    """The admission record's notes as admission finds them."""

    declarations: list[DeclarationNoteV1] = field(default_factory=list[DeclarationNoteV1])
    facts: list[FactNoteV1] = field(default_factory=list[FactNoteV1])
    groups: list[GroupNoteV1] = field(default_factory=list[GroupNoteV1])
    anchors: list[AnchorNoteV1] = field(default_factory=list[AnchorNoteV1])
    fields: list[FieldNoteV1] = field(default_factory=list[FieldNoteV1])
    merges: list[MergeV1] = field(default_factory=list[MergeV1])

    def note_anchor(self, position: int, anchor: int, rule: AnchorRule) -> None:
        self.anchors.append(AnchorNoteV1(position=position, anchor=anchor, rule=rule))

    def note_field(
        self, position: int, name: FieldName, index: int | None, rule: FieldRule,
    ) -> None:
        self.fields.append(FieldNoteV1(position=position, field=name, index=index, rule=rule))

    def record(self, reply_facts: int) -> ExtractionAdmissionV2 | None:
        if not any((self.declarations, self.facts, self.groups, self.anchors, self.fields,
                    self.merges)):
            return None
        return ExtractionAdmissionV2(
            reply_facts=reply_facts,
            declarations=sorted(self.declarations, key=lambda n: (n.list, n.index, n.rule)),
            facts=sorted(self.facts, key=lambda note: note.position),
            groups=sorted(self.groups, key=lambda note: (note.position, note.group)),
            anchors=sorted(self.anchors, key=lambda n: (n.position, n.anchor, n.rule)),
            fields=sorted(self.fields, key=lambda n: (
                n.position, n.field, -1 if n.index is None else n.index, n.rule,
            )),
            merges=sorted(self.merges, key=lambda note: note.position),
        )


def _mention(text: str | None) -> str | None:
    """Return a mention cut to its bound, or None when nothing is left."""
    if text is None:
        return None
    cut = text.strip()[:MAX_MENTION_CHARS].strip()
    return cut or None


@dataclass(frozen=True, slots=True)
class _Declarations:
    """The reply's declarations after B.7.2."""

    new: dict[str, NewEntityChoiceV1]
    masked_new: dict[str, NewEntityV12]
    unresolved: dict[str, UnresolvedChoiceV1]
    masked_unresolved: dict[str, UnresolvedChoiceV1]
    raw_new: dict[str, NewEntityV12]


def _candidate(handle: str, context: ExtractionContext, new: dict[str, NewEntityChoiceV1]) -> (
    str | LocalKeyRefV1 | None
):
    if handle in new:
        return LocalKeyRefV1(local_key=handle)
    entry = context.catalog.entry(handle)
    return None if entry is None else entry.entity_id


def _declarations(
    reply: ExtractOutputV12, context: ExtractionContext, texts: _Texts, notes: _Notes,
) -> _Declarations:
    """Apply B.7.2 to the declarations.

    A valid declaration whose name no rendered source writes keeps its descriptor
    and type without the name (``name_unlocated``): the name may be a
    transliteration ("Maya" for 마야), a translation or an invention, so it never
    becomes the entity's name, while the facts that anchor the entity are kept.
    Two declarations with one ``name_norm`` stay masked (``name_shared``) even when
    the name is unlocated.
    """
    rules: dict[int, DeclarationRule] = {}
    unlocated: set[int] = set()
    counts = Counter(declaration.handle for declaration in reply.new_entities)
    for index, declaration in enumerate(reply.new_entities):
        descriptor = declaration.descriptor.strip()
        type_text = unicodedata.normalize("NFC", declaration.type).strip()
        if declaration.handle not in NEW_ENTITY_HANDLES:
            rules[index] = "declaration_invalid"
        elif counts[declaration.handle] > 1:
            rules[index] = "declaration_repeated"
        elif (
            declaration.name is not None and not is_name(declaration.name)
            or not 1 <= len(descriptor) <= MAX_DESCRIPTOR_CHARS
            or not 1 <= len(type_text) <= MAX_ENTITY_TYPE_CHARS
        ):
            rules[index] = "declaration_invalid"
        elif declaration.name is not None and not texts.locates_name(declaration.name):
            unlocated.add(index)
    norms = [
        (index, norm) for index, declaration in enumerate(reply.new_entities)
        if declaration.name is not None and (norm := name_norm(declaration.name))
    ]
    shared = {norm for norm, count in Counter(norm for _, norm in norms).items() if count > 1}
    for index, norm in norms:
        if norm in shared:
            rules.setdefault(index, "name_shared")
    new: dict[str, NewEntityChoiceV1] = {}
    masked_new: dict[str, NewEntityV12] = {}
    raw_new: dict[str, NewEntityV12] = {}
    for index, declaration in enumerate(reply.new_entities):
        raw_new.setdefault(declaration.handle, declaration)
        if index in rules:
            notes.declarations.append(DeclarationNoteV1(list="new", index=index, rule=rules[index]))
            masked_new.setdefault(declaration.handle, declaration)
            continue
        if index in unlocated:
            notes.declarations.append(DeclarationNoteV1(list="new", index=index,
                                                        rule="name_unlocated"))
        new[declaration.handle] = NewEntityChoiceV1(
            kind="new", local_key=declaration.handle,
            name=None if index in unlocated else declaration.name,
            descriptor=declaration.descriptor.strip(),
            entity_type=stored_entity_type(declaration.type),
        )
    unresolved: dict[str, UnresolvedChoiceV1] = {}
    masked_unresolved: dict[str, UnresolvedChoiceV1] = {}
    counts = Counter(referent.handle for referent in reply.unresolved_referents)
    for index, referent in enumerate(reply.unresolved_referents):
        rule: DeclarationRule | None = None
        if referent.handle not in UNRESOLVED_HANDLES:
            rule = "declaration_invalid"
        elif counts[referent.handle] > 1:
            rule = "declaration_repeated"
        elif not 1 <= len(referent.mention.strip()) <= MAX_MENTION_CHARS:
            rule = "declaration_invalid"
        resolved = [_candidate(handle, context, new) for handle in referent.candidates]
        kept = list(dict.fromkeys(
            candidate for candidate in resolved if candidate is not None
        ))
        choice = UnresolvedChoiceV1(
            kind="unresolved", mention=_mention(referent.mention),
            reason=cast(Literal["ambiguous", "insufficient_context"], referent.reason)
            if referent.reason in _REASONS else "insufficient_context",
            candidates=kept[:MAX_REFERENT_CANDIDATES],
        )
        changed: list[DeclarationRule] = [] if rule is None else [rule]
        if rule is None and referent.reason not in _REASONS:
            changed.append("reason_unknown")
        if None in resolved or len(kept) > MAX_REFERENT_CANDIDATES:
            changed.append("candidates_dropped")
        notes.declarations.extend(
            DeclarationNoteV1(list="unresolved", index=index, rule=item) for item in changed
        )
        if rule is None:
            unresolved[referent.handle] = choice
        else:
            masked_unresolved.setdefault(referent.handle, choice.model_copy(update={
                "reason": "insufficient_context",
            }))
    return _Declarations(new, masked_new, unresolved, masked_unresolved, raw_new)


SpanKey = tuple[str, int, int]
"""A located span as a hashable key: ``(evidence_id, byte_start, byte_end)``."""


def span_key(span: QuoteSpanV1) -> SpanKey:
    """Return the hashable key of a located span."""
    return (span.evidence_id, span.byte_start, span.byte_end)


@dataclass(frozen=True, slots=True)
class _Quote:
    source: RenderedSource
    text: str
    spans: tuple[QuoteSpanV1, ...]
    char_start: int
    char_end: int


@dataclass(frozen=True, slots=True)
class _Group:
    index: int
    quotes: tuple[_Quote, ...]

    @property
    def spans(self) -> tuple[QuoteSpanV1, ...]:
        unique = {span_key(span): span for quote in self.quotes for span in quote.spans}
        return tuple(unique.values())


def _group(
    group: SupportGroupV12, context: ExtractionContext, seen: set[tuple[SpanKey, ...]],
) -> GroupRule | tuple[_Quote, ...]:
    """Locate one support group's quotes, or return the rule that drops it (B.7.3)."""
    if len(group.quotes) > MAX_GROUP_QUOTES:
        return "span_bound"
    quotes: list[_Quote] = []
    located_here: set[tuple[SpanKey, ...]] = set()
    for quote in group.quotes:
        source = context.source(quote.source_id)
        if source is None:
            return "source_unknown"
        if not quote.quote.strip():
            return "quote_blank"
        if len(quote.quote) > MAX_QUOTE_CHARS:
            return "quote_bound"
        located = locate_quote(quote.quote, source.masked)
        if isinstance(located, str):
            return located
        keys = tuple(span_key(span) for span in located.spans)
        if keys in seen or keys in located_here:
            return "duplicate_span"
        located_here.add(keys)
        quotes.append(_Quote(source, source.text[located.char_start:located.char_end],
                             located.spans, located.char_start, located.char_end))
    if not any(quote.source.fresh for quote in quotes):
        return "no_fresh_quote"
    if len({span_key(span) for quote in quotes for span in quote.spans}) > MAX_GROUP_SPANS:
        return "span_bound"
    return tuple(quotes)


def _groups(
    position: int, fact: FactV12, context: ExtractionContext, notes: _Notes,
) -> list[_Group]:
    """Return the fact's admitted groups (B.7.3 group rows)."""
    kept: list[_Group] = []
    seen: set[tuple[SpanKey, ...]] = set()
    for index, group in enumerate(fact.support_groups):
        result = _group(group, context, seen)
        if isinstance(result, str):
            notes.groups.append(GroupNoteV1(position=position, group=index, rule=result))
            continue
        kept.append(_Group(index, result))
        seen.update(tuple(span_key(span) for span in quote.spans) for quote in result)
    spans: set[SpanKey] = set()
    admitted: list[_Group] = []
    for group in kept:
        keys = {span_key(span) for span in group.spans}
        if len(admitted) == MAX_SUPPORT_GROUPS or len(spans | keys) > MAX_MEMORY_SPANS:
            notes.groups.append(GroupNoteV1(position=position, group=group.index,
                                            rule="group_bound"))
            continue
        admitted.append(group)
        spans.update(keys)
    return admitted


def _secret(fact: FactV12, declarations: _Declarations) -> bool:
    """Whether a recognized credential appears in what the fact would store (B.7.3):
    its statement, keywords, naming names, ``corrects_text`` (its proposal keeps it) or
    an anchored declaration."""
    texts = [fact.statement, *fact.keywords, *(naming.name for naming in fact.naming)]
    if fact.corrects_text is not None:
        texts.append(fact.corrects_text)
    for handle in fact.anchors:
        declaration = declarations.raw_new.get(handle)
        if declaration is not None:
            texts.extend(
                text for text in (declaration.name, declaration.descriptor, declaration.type)
                if text is not None
            )
    return any(contains_recognized_credential(text.encode("utf-8")) for text in texts)


def _credentials_finding(fact: FactV12) -> bool:
    """Whether a finding of the model, valid or not, says ``credentials`` (B.7.3).

    The extractor recognizes a credential in any language; the signatures of
    `contains_recognized_credential` cover only listed forms.
    """
    return any(finding.category == "credentials" for finding in fact.sensitivity)


def _user_unattested(
    fact: FactV12, statement: str, groups: Sequence[_Group], context: ExtractionContext,
) -> bool:
    """B.7.3 ``user_unattested``: the fact refers to the user, who does not support it.

    A fact refers to the user when the ``user`` handle is among its anchors (the
    first 8 distinct handles, as B.7.3's anchor rows keep them) or its statement
    uses a wording for the user (`uses_user_wording`). It is refused when
    no rendered source of the unit is authenticated self (B.4.2 (1)), or when its
    statement uses that wording and no admitted group quotes an authenticated-self
    source (B.7.4): the user never said it. With such a quote the wording is the
    user's own, and the fact keeps the anchors the extraction gave it, the user among
    them or not: a fact about someone the user names by their tie to the user ("the
    user's cousin") is about that person. A ``user`` anchor of a statement without
    that wording in a unit that has a self source keeps the ``self_not_proven``
    degradation.
    """
    anchored = _user_anchored(fact)
    worded = uses_user_wording(statement)
    if not anchored and not worded:
        return False
    if not any(source.attribution.authenticated_self for source in context.sources):
        return True
    return worded and not _self_proven(groups)


def _self_proven(groups: Sequence[_Group]) -> bool:
    return any(
        quote.source.attribution.authenticated_self for group in groups for quote in group.quotes
    )


def _choice(entry: CatalogEntry) -> EntityChoiceV1:
    if entry.kind == "user":
        return UserChoiceV1(kind="user")
    if entry.participant_key is not None and entry.route in ("participant", "label"):
        return ParticipantChoiceV1(
            kind="participant", participant_key=entry.participant_key, entity_id=entry.entity_id,
        )
    if entry.entity_id is None:
        raise ValueError("A catalog entity without an id is a participant")
    return ExistingChoiceV1(
        kind="existing", entity_id=entry.entity_id,
        via="sdk" if entry.route == "sdk" else "catalog",
    )


def _speaker_candidates(
    mention: str | None, context: ExtractionContext,
) -> list[str | LocalKeyRefV1]:
    """Catalog entities whose names or speaker names equal a mention (B.7.3 ``self_not_proven``)."""
    if mention is None or not (norm := name_norm(mention)):
        return []
    found: list[str | LocalKeyRefV1] = [
        entry.entity_id for entry in context.catalog.entries
        if entry.entity_id is not None and (
            any(row.name_norm == norm for row in entry.names)
            or any(name_norm(name) == norm for name in entry.speaker_names)
        )
    ]
    return found[:MAX_REFERENT_CANDIDATES]


@dataclass(frozen=True, slots=True)
class _Anchors:
    choices: tuple[EntityChoiceV1, ...]
    handles: tuple[str | None, ...]


def _anchors(
    position: int, fact: FactV12, groups: Sequence[_Group], context: ExtractionContext,
    declarations: _Declarations, notes: _Notes,
) -> _Anchors:
    """Resolve the fact's anchors (B.7.3 anchor rows, B.7.4 identity checks)."""
    seen: set[str] = set()
    slots: list[tuple[int, str]] = []
    for index, handle in enumerate(fact.anchors):
        if handle in seen:
            notes.note_anchor(position, index, "duplicate")
        elif len(slots) == MAX_ANCHORS:
            notes.note_anchor(position, index, "anchor_bound")
        else:
            seen.add(handle)
            slots.append((index, handle))
    if not slots:
        # A fact without any anchor degrades like an unknown primary handle.
        notes.note_anchor(position, 0, "handle_unknown")
        return _Anchors((_degraded("handle_unknown", "", groups, context, declarations),), (None,))
    proven = _self_proven(groups)
    choices: list[EntityChoiceV1] = []
    handles: list[str | None] = []
    for slot, (index, handle) in enumerate(slots):
        primary = slot == 0
        rule: AnchorRule | None = None
        choice: EntityChoiceV1 | None = None
        entry = context.catalog.entry(handle)
        if handle in declarations.new:
            choice = declarations.new[handle]
        elif handle in declarations.unresolved:
            choice = declarations.unresolved[handle]
            rule = None if primary else "secondary_unresolved"
        elif handle in declarations.masked_new or handle in declarations.masked_unresolved:
            rule = "declaration_invalid"
        elif handle in NEW_ENTITY_HANDLES or handle in UNRESOLVED_HANDLES:
            rule = "declaration_missing"
        elif entry is None:
            rule = "handle_unknown"
        elif entry.kind == "user" and not proven:
            rule = "self_not_proven"
        else:
            choice = _choice(entry)
        if rule is not None and not primary:
            notes.note_anchor(position, index, rule)
            continue
        if rule is not None:
            notes.note_anchor(position, index, rule)
            choice = _degraded(rule, handle, groups, context, declarations)
            handle = None
        assert choice is not None
        choices.append(choice)
        handles.append(handle)
    conflict = _binding_conflict(groups, context, choices)
    if conflict is not None and choices[0].kind != "unresolved":
        notes.note_anchor(position, slots[0][0], "anchor_conflict")
        choices[0], handles[0] = conflict, None
    return _Anchors(tuple(choices), tuple(handles))


def _degraded(
    rule: AnchorRule, handle: str, groups: Sequence[_Group], context: ExtractionContext,
    declarations: _Declarations,
) -> UnresolvedChoiceV1:
    """The unresolved primary a failed anchor rule leaves (B.7.3)."""
    if rule == "self_not_proven":
        attribution = groups[0].quotes[0].source.attribution
        mention = _mention(attribution.label or attribution.metadata_speaker)
        return UnresolvedChoiceV1(
            kind="unresolved", mention=mention, reason="insufficient_context",
            candidates=list(_speaker_candidates(mention, context)),
        )
    if rule == "declaration_invalid" and handle in declarations.masked_unresolved:
        return declarations.masked_unresolved[handle]
    if rule == "declaration_invalid":
        declaration = declarations.masked_new[handle]
        return UnresolvedChoiceV1(
            kind="unresolved",
            mention=_mention(declaration.name) or _mention(declaration.descriptor),
            reason="insufficient_context", candidates=[],
        )
    return UnresolvedChoiceV1(
        kind="unresolved", mention=None, reason="insufficient_context", candidates=[],
    )


def _binding_conflict(
    groups: Sequence[_Group], context: ExtractionContext, choices: Sequence[EntityChoiceV1],
) -> UnresolvedChoiceV1 | None:
    """B.7.4: a quoted SDK binding whose entity the fact does not anchor."""
    anchored = {
        choice.entity_id for choice in choices
        if isinstance(choice, ExistingChoiceV1 | ParticipantChoiceV1) and choice.entity_id
    }
    conflicts: list[RenderedBinding] = []
    for group in groups:
        for quote in group.quotes:
            for span in quote.spans:
                conflicts.extend(
                    binding for binding in context.bindings
                    if binding.evidence_id == span.evidence_id
                    and binding.byte_start < span.byte_end and span.byte_start < binding.byte_end
                    and binding.entity_id not in anchored
                )
    if not conflicts:
        return None
    candidates: list[str | LocalKeyRefV1] = list(dict.fromkeys(
        binding.entity_id for binding in conflicts
    ))
    return UnresolvedChoiceV1(
        kind="unresolved", mention=_mention(conflicts[0].text), reason="insufficient_context",
        candidates=candidates[:MAX_REFERENT_CANDIDATES],
    )


def _time(
    position: int, fact: FactV12, memory_kind: MemoryKind, groups: Sequence[_Group],
    texts: _Texts, notes: _Notes, pipeline_version: str,
) -> tuple[AdmittedTime, bool]:
    """Apply B.7.3's time rows and anchor validity (B.10); also return whether it was masked."""
    kind, text = fact.time.kind, fact.time.text
    if kind not in _TIME_KINDS or (kind == "none") != (text is None) or text is not None and (
        not text.strip() or len(text) > MAX_TIME_TEXT_CHARS
    ):
        notes.note_field(position, "time", None, "time_inconsistent")
        return NO_TIME, True
    if text is None:
        return NO_TIME, False
    cited = {
        quote.source.source_id: quote.source for group in groups for quote in group.quotes
    }
    anchoring = next(
        (source for source in cited.values() if texts.contains(text, (source.source_id,))), None,
    )
    if anchoring is None:
        notes.note_field(position, "time", None, "time_unlocated")
        return NO_TIME, True
    expression = SourceTimeExpressionV1(
        kind=cast(Literal["absolute", "relative"], kind), text=text.strip(),
    )
    window = resolve_time(
        expression, source_observed_at=anchoring.observed_at, pipeline_version=pipeline_version,
    )
    valid_from, valid_to = window.validity(memory_kind)
    return AdmittedTime(
        expression, valid_from, valid_to, window.granularity, window.anchored_from,
    ), False


def _naming(
    position: int, fact: FactV12, statement: str, anchors: _Anchors, groups: Sequence[_Group],
    context: ExtractionContext, notes: _Notes,
) -> tuple[tuple[AdmittedNaming, ...], bool]:
    """Apply B.7.3's naming rows; also return whether an annotation was masked."""
    quoted = [quote.text for group in groups for quote in group.quotes]
    kept: list[AdmittedNaming] = []
    seen: set[tuple[int, str, str]] = set()
    for index, naming in enumerate(fact.naming):
        slot = next((
            slot for slot, handle in enumerate(anchors.handles)
            if handle == naming.entity and anchors.choices[slot].kind != "unresolved"
        ), None)
        in_quotes = name_located(naming.name, quoted)
        rule: FieldRule | None = None
        if index >= MAX_NAMING:
            rule = "naming_bound"
        elif naming.name_kind not in _NAME_KINDS or not is_name(naming.name):
            rule = "naming_invalid"
        elif slot is None:
            rule = "naming_not_anchor"
        elif not token_contained(naming.name, statement):
            rule = "naming_not_in_statement"
        elif naming.name_kind != "former" and not in_quotes:
            rule = "naming_unquoted"
        elif naming.name_kind == "former" and not in_quotes and not _live_name(
            naming.name, naming.entity, context,
        ):
            rule = "naming_former_unknown"
        elif (slot, name_norm(naming.name), naming.name_kind) in seen:
            rule = "naming_duplicate"
        if rule is not None or slot is None:
            notes.note_field(position, "naming", index, rule or "naming_not_anchor")
            continue
        seen.add((slot, name_norm(naming.name), naming.name_kind))
        kept.append(AdmittedNaming(slot, naming.name, cast(NameKind, naming.name_kind)))
    return tuple(kept), len(kept) < len(fact.naming)


def _live_name(name: str, handle: str, context: ExtractionContext) -> bool:
    entry = context.catalog.entry(handle)
    return entry is not None and any(row.name_norm == name_norm(name) for row in entry.names)


def _alias_fact(
    fact: FactV12, naming: Sequence[AdmittedNaming], anchors: _Anchors,
    groups: Sequence[_Group], context: ExtractionContext,
) -> FactV12 | None:
    """Give C45 only admitted anchors/names and already located source quotes."""
    names: dict[str, list[str]] = {}
    validated: list[NamingV12] = []
    for slot, handle in enumerate(anchors.handles):
        if handle is None or anchors.choices[slot].kind == "unresolved":
            continue
        entry = context.catalog.entry(handle)
        names[handle] = [] if entry is None else [
            *(row.name for row in entry.names), *entry.speaker_names,
        ]
        choice = anchors.choices[slot]
        if isinstance(choice, NewEntityChoiceV1) and choice.name is not None:
            names[handle].append(choice.name)
        if handle == "user":
            names[handle].extend(("I", "my", "the user"))
        for item in naming:
            if item.anchor_index == slot:
                validated.append(NamingV12(
                    entity=handle, name=item.name, name_kind=item.name_kind,
                ))
        # A model-invented former annotation cannot become a readable former
        # name. A source-written subject can still ground the alias's anchor
        # when the minimal quote omitted that subject and the declaration used
        # the nickname instead (the original field refusal remains recorded).
        names[handle].extend(
            item.name for item in fact.naming
            if isinstance(choice, NewEntityChoiceV1)
            and item.entity == handle and item.name_kind == "former" and is_name(item.name)
            and token_contained(item.name, fact.statement)
            and name_located(item.name, [quote.source.text
                                        for group in groups for quote in group.quotes])
        )
    return normalize_alias_fact(
        fact.model_copy(update={"naming": validated}), names=names,
        quoted=[(quote.source.masked, quote.char_start, quote.char_end)
                for group in groups for quote in group.quotes],
    )


def _revises(position: int, fact: FactV12, context: ExtractionContext, notes: _Notes) -> (
    tuple[RevisedMemory, ...]
):
    kept: list[RevisedMemory] = []
    seen: set[str] = set()
    for index, handle in enumerate(fact.revises):
        memory = context.memory(handle)
        if memory is None:
            notes.note_field(position, "revises", index, "revises_unknown")
        elif handle in seen:
            notes.note_field(position, "revises", index, "revises_duplicate")
        elif len(kept) == MAX_REVISES:
            notes.note_field(position, "revises", index, "revises_bound")
        else:
            seen.add(handle)
            kept.append(RevisedMemory(
                memory.memory_id, memory.version.revision_id, memory.origin == "earlier",
            ))
    return tuple(kept)


def _corrects_text(position: int, fact: FactV12, notes: _Notes) -> str | None:
    if fact.corrects_text is None:
        return None
    text = " ".join(unicodedata.normalize("NFC", fact.corrects_text).split())
    if not 1 <= len(text) <= MAX_CORRECTS_TEXT_CHARS:
        notes.note_field(position, "corrects_text", None, "corrects_text_invalid")
        return None
    return text


def _keywords(position: int, fact: FactV12, notes: _Notes) -> tuple[str, ...]:
    kept: dict[str, str] = {}
    for index, keyword in enumerate(fact.keywords):
        text = keyword.strip()
        if not 1 <= len(text) <= MAX_KEYWORD_CHARS or not any(char.isalnum() for char in text):
            notes.note_field(position, "keywords", index, "keyword_invalid")
            continue
        key = unicodedata.normalize("NFKC", text).casefold()
        if key in kept:
            continue
        if len(kept) == MAX_KEYWORDS:
            notes.note_field(position, "keywords", index, "keyword_bound")
            continue
        kept[key] = text
    return tuple(kept.values())


_SOURCE_SCRIPT_RESERVE: Final = 8


def _source_script_keywords(
    keywords: tuple[str, ...], anchors: tuple[EntityChoiceV1, ...],
    declarations: _Declarations, groups: Sequence[_Group],
) -> tuple[str, ...]:
    """Keep quote terms searchable when B.7.2 dropped an unlocated declared name.

    No term identifies or names an entity. Only the fact's admitted, located
    quotes contribute non-Latin terms from Search's own tokenizer. Shorter
    quotes come first. When model keywords occupy the eight reserved slots,
    multi-character terms precede single characters within each quote; otherwise
    the source term order is unchanged. Up to eight source-only terms reserve
    room within the existing keyword bound. Model terms also present in those
    quotes take priority over other model terms, retaining their original relative
    order. A credential-bearing
    quote contributes nothing.
    """
    if not any(
        choice.kind == "new" and choice.name is None
        and declarations.raw_new[choice.local_key].name is not None
        for choice in anchors
    ):
        return keywords
    model = {unicodedata.normalize("NFKC", word).casefold(): word for word in keywords}
    quoted_terms: dict[str, str] = {}
    quoted = sorted((quote.text for group in groups for quote in group.quotes), key=len)
    prioritize = len(model) > MAX_KEYWORDS - _SOURCE_SCRIPT_RESERVE
    for text in quoted:
        if contains_recognized_credential(text.encode("utf-8")):
            continue
        terms = sparse_terms(text)
        if prioritize:
            terms = tuple(sorted(terms, key=lambda word: len(word) == 1))
        for term in terms:
            if len(term) <= MAX_KEYWORD_CHARS and any(
                char.isalpha() and "LATIN" not in unicodedata.name(char, "") for char in term
            ):
                quoted_terms.setdefault(term, term)
            if len(quoted_terms) == MAX_KEYWORDS:
                break
        if len(quoted_terms) == MAX_KEYWORDS:
            break
    generated = [term for term in quoted_terms if term not in model]
    reserved = min(_SOURCE_SCRIPT_RESERVE, len(generated))
    model_limit = MAX_KEYWORDS - reserved
    prioritized = [key for key in model if key in quoted_terms]
    prioritized.extend(key for key in model if key not in quoted_terms)
    selected = set(prioritized[:model_limit])
    kept = {key: word for key, word in model.items() if key in selected}
    for term in generated:
        if len(kept) == MAX_KEYWORDS:
            break
        kept[term] = term
    return tuple(kept.values())


@dataclass(frozen=True, slots=True)
class _Findings:
    """The model's sensitivity findings after B.7.3: valid ones and dropped indices."""

    categories: frozenset[SensitivityClass]
    quotes: tuple[str, ...]
    invalid: tuple[int, ...]


def _findings(fact: FactV12) -> _Findings:
    """Keep valid findings; an invalid one is dropped and forces ``restricted``."""
    categories: set[SensitivityClass] = set()
    quotes: list[str] = []
    invalid: list[int] = []
    for index, finding in enumerate(fact.sensitivity):
        if (
            index >= MAX_SENSITIVITY_FINDINGS or finding.category not in _CATEGORIES
            or not finding.quote.strip() or len(finding.quote) > MAX_QUOTE_CHARS
        ):
            invalid.append(index)
            continue
        categories.add(cast(SensitivityClass, finding.category))
        quotes.append(finding.quote)
    return _Findings(frozenset(categories), tuple(quotes), tuple(invalid))


def _evidence(context: ExtractionContext) -> dict[str, EvidenceRecord]:
    return {
        part.object_id: part for source in context.sources for part in source.run.parts
    }


def _origin(kinds: set[str]) -> Origin:
    """B.10 origin: all cited Evidence tool results or all documents, else inference."""
    if kinds == {"tool_result"}:
        return "tool_output"
    if kinds == {"document"}:
        return "external_document"
    return "agent_inference"


_NAMING_FORM = re.compile(r"^(?P<descriptor>[^,;:]+?) is (?:named|called) "
                          r"(?P<name>[^,.;:]+)\.$")
"""A naming fact's statement: ``<descriptor> is named <name>.``"""
_RESTATED_FORM = re.compile(r"^(?P<name>[^,.;:]+?) is (?P<descriptor>[^,;:]+)\.$")
"""A statement that only gives a named entity a descriptor: ``<name> is <descriptor>.``"""


def _naming_restatement_key(fact: AdmittedFact, descriptor: str, name: str) -> str:
    """The identity a naming fact shares with the restatement of its descriptor (B.7.3)."""
    return canonical_json([
        name_norm(descriptor), name_norm(name), fact.kind,
        sorted(canonical_json(choice.model_dump(mode="json")) for choice in fact.anchors),
        sorted((memory.memory_id, memory.revision_id) for memory in fact.revises),
        fact.corrects_text,
    ])


def _naming_key(fact: AdmittedFact) -> str | None:
    """The key of a naming fact whose statement gives its one name after its descriptor."""
    match = _NAMING_FORM.match(fact.statement)
    if match is None or [name_norm(naming.name) for naming in fact.naming
                         if naming.name_kind == "name"] != [name_norm(match["name"])]:
        return None
    return _naming_restatement_key(fact, match["descriptor"], match["name"])


def _restatement_key(fact: AdmittedFact) -> str | None:
    """The key of a fact without naming that only says the named entity is the descriptor
    a naming fact names it by (``<name> is <descriptor>.``)."""
    match = None if (fact.naming or fact.time.expression.kind != "none"
                     or "time_unlocated" in fact.concerns) else (
        _RESTATED_FORM.match(fact.statement)
    )
    return None if match is None else _naming_restatement_key(
        fact, match["descriptor"], match["name"])


def _duplicate_key(fact: AdmittedFact) -> str:
    """The B.7.3 ``duplicate_in_reply`` identity: everything but the support."""
    return canonical_json([
        fact.statement_digest, fact.kind, fact.anchors[0].model_dump(mode="json"),
        sorted(canonical_json(choice.model_dump(mode="json")) for choice in fact.anchors[1:]),
        sorted((memory.memory_id, memory.revision_id) for memory in fact.revises),
        fact.corrects_text,
    ])


def admit(
    reply: ExtractOutputV12, context: ExtractionContext, profile: Profile, *,
    pipeline_version: str, implied_health_rule: bool = False,
) -> Admission:
    """Admit one structurally valid extract-v12 reply against its frozen context (B.7).

    Args:
        reply (ExtractOutputV12): The verbatim reply, already decoded (B.6.2).
        context (ExtractionContext): The unit's frozen context.
        profile (Profile): ``product`` refuses facts with credentials and stores
            every memory ``restricted``; ``bench`` admits them ``restricted``.
        pipeline_version (str): The owner policy's extraction pipeline, which
            anchors each fact's time expression (``resolve_time``, B.10).
        implied_health_rule (bool): The owner policy's supplement rule asks for implied
            health statements (``IMPLIED_HEALTH_RULES``). A supplement fact in that form
            (`implied_health`) must then name a health class and quote no forget
            request, and it may name a health class that only its copied quote
            implies; it is stored with origin ``implied_health``. Any other supplement
            statement marked "(implied by:" is refused. The extraction call's own
            facts keep the floor whatever they say.

    Returns:
        Admission: The admitted and refused positions and the admission record.

    Raises:
        ExtractionRefused: ``no_fresh_source`` when the reply has facts but the
            context renders no fresh source (B.7.1).
        ValueError: If a fact's time is resolved under a pipeline that is not current.
    """
    if reply.facts and not any(source.fresh for source in context.sources):
        raise ExtractionRefused("no_fresh_source", "Extraction reply has facts but no fresh source")
    notes = _Notes()
    texts = _Texts.of(context.sources)
    declarations = _declarations(reply, context, texts, notes)
    evidence = _evidence(context)
    admitted: list[AdmittedFact] = []
    refused: list[RefusedFact] = []
    start = reply.supplement_from
    rollup = len(reply.facts) if reply.rollup_from is None else reply.rollup_from
    implied: set[int] = set()

    def refuse(position: int, rule: FactRule) -> None:
        notes.facts.append(FactNoteV1(position=position, rule=rule))
        refused.append(RefusedFact(position, rule, REFUSALS[rule]))

    for position, fact in enumerate(reply.facts):
        if not fact.statement.strip():
            refuse(position, "statement_invalid")
            continue
        groups = _groups(position, fact, context, notes)
        if not groups:
            refuse(position, "no_valid_group")
            continue
        try:
            statement = canonical_statement(fact.statement)
        except ValueError:
            refuse(position, "statement_invalid")
            continue
        if fact.kind not in _KINDS:
            refuse(position, "kind_invalid")
            continue
        kind = cast(MemoryKind, fact.kind)
        if len(canonical_json(fact)) > PROVIDER_PROPOSAL_CANONICAL_CHARS:
            refuse(position, "fact_bound")
            continue
        secret = _secret(fact, declarations)
        if profile == "product" and (secret or _credentials_finding(fact)):
            refuse(position, "secret")
            continue
        findings = _findings(fact)
        quoted = [quote.text for group in groups for quote in group.quotes]
        statement_classes = classify([statement])
        stated = statement_classes & PROTECTED_CLASSES
        sourced = classify(quoted) | classify(
            quote for quote in findings.quotes if texts.contains(quote)
        )
        if (implied_health_rule and start is not None and start <= position < rollup
                and _IMPLIED_MARK in statement):
            # Only a grounded implied health statement stands; any other "implied by"
            # inference of the supplement (a preference read off a forget request) is none.
            if not (implied_health(statement, quoted)
                    and _implied_grounded(statement, stated, quoted)):
                refuse(position, "sensitive_inference")
                continue
            implied.add(position)
            stated -= _IMPLIED_EXEMPT
        if not stated <= sourced:
            refuse(position, "sensitive_inference")
            continue
        if _user_unattested(fact, statement, groups, context):
            refuse(position, "user_unattested")
            continue
        for index in findings.invalid:
            notes.note_field(position, "sensitivity", index, "sensitivity_invalid")
        if secret:
            notes.note_field(position, "sensitivity", None, "secret")
        anchors = _anchors(position, fact, groups, context, declarations, notes)
        naming, naming_masked = _naming(position, fact, statement, anchors, groups, context, notes)
        alias = _alias_fact(fact, naming, anchors, groups, context)
        if alias is not None:
            try:
                alias_statement = canonical_statement(alias.statement)
            except ValueError:
                refuse(position, "statement_invalid")
                continue
            alias_secret = _secret(alias, declarations)
            if profile == "product" and (alias_secret or _credentials_finding(alias)):
                refuse(position, "secret")
                continue
            alias_classes = classify([alias_statement])
            if not alias_classes & PROTECTED_CLASSES <= sourced:
                refuse(position, "sensitive_inference")
                continue
            if _user_unattested(alias, alias_statement, groups, context):
                refuse(position, "user_unattested")
                continue
            if alias_secret and not secret:
                notes.note_field(position, "sensitivity", None, "secret")
            secret = secret or alias_secret
            for name in ("statement", "kind", "naming", "corrects_text"):
                if getattr(alias, name) != getattr(fact, name):
                    notes.note_field(position, cast(FieldName, name), None, "alias_normalized")
            fact = alias
            statement = alias_statement
            statement_classes = alias_classes
            kind = "state"
            slot = anchors.handles.index(fact.naming[0].entity)
            naming = (AdmittedNaming(slot, fact.naming[0].name, "alias"),)
        time, time_masked = _time(
            position, fact, kind, groups, texts, notes, pipeline_version,
        )
        # An implied statement is a health condition, whatever the floor's terms name.
        implied_class = _IMPLIED_EXEMPT if position in implied else frozenset[SensitivityClass]()
        classes = tuple(sorted(
            classify([statement, *quoted]) | findings.categories | implied_class
        ))
        cited = [evidence[span.evidence_id] for group in groups for span in group.spans]
        concerns: set[Concern] = set()
        if time_masked:
            concerns.add("time_unlocated")
        if naming_masked:
            concerns.add("naming_masked")
        if context.catalog.truncated and any(choice.kind == "new" for choice in anchors.choices):
            concerns.add("catalog_truncated")
        forced = secret or bool(findings.invalid)
        admitted.append(AdmittedFact(
            position=position,
            statement=statement,
            statement_digest=statement_digest(statement),
            kind=kind,
            anchors=anchors.choices,
            naming=naming,
            groups=tuple(group.spans for group in groups),
            time=time,
            keywords=_source_script_keywords(
                _keywords(position, fact, notes), anchors.choices, declarations, groups,
            ),
            revises=_revises(position, fact, context, notes),
            corrects_text=_corrects_text(position, fact, notes),
            security_level=security_level(
                profile, statement_classes | findings.categories | implied_class, forced=forced,
            ),
            sensitivity_classes=classes,
            sensitivity_basis=("policy_classification",) if classes else (),
            origin=_origin({part.kind for part in cited}),
            trust_tier=min((part.trust_tier for part in cited), key=_TRUST.__getitem__),
            concerns=frozenset(concerns),
            duplicate_of=None,
        ))
    fresh = {part.object_id for source in context.sources if source.fresh
             for part in source.run.parts}
    facts = _merge_and_cap(_bare_naming(admitted, evidence, fresh), notes, refuse)
    if start is not None:
        # The supplement call's facts, and the session rollup after them, carry their
        # own origin, whatever they cite; an implied health statement its own, so Search
        # reads it as an ordinary memory.
        facts = [
            replace(fact, origin="session_rollup") if fact.position >= rollup
            else replace(fact, origin="implied_health") if fact.position in implied
            else replace(fact, origin="extraction_supplement") if fact.position >= start
            else fact for fact in facts
        ]
    refused.sort(key=lambda item: item.position)
    return Admission(tuple(facts), tuple(refused), notes.record(len(reply.facts)))


def _merge_and_cap(
    admitted: Sequence[AdmittedFact], notes: _Notes, refuse: Callable[[int, FactRule], None],
) -> list[AdmittedFact]:
    """Merge in-reply duplicates and refuse facts beyond the cap (B.7.3), in reply order.

    A fact equal to an earlier kept fact (statement digest, primary anchor,
    secondary anchors, kind, ``revises`` and ``corrects_text``; an event must
    also share a located span) is a duplicate of it and counts against no cap.
    Every other fact beyond the first 128 is refused.
    """
    kept: list[AdmittedFact] = []
    representatives: dict[str, list[AdmittedFact]] = {}
    count: int = 0
    for fact in admitted:
        key = _duplicate_key(fact)
        spans = {span_key(span) for span in fact.spans}
        primary = next((
            earlier for earlier in representatives.get(key, ())
            if fact.kind == "state" or spans & {span_key(span) for span in earlier.spans}
        ), None)
        if primary is not None:
            notes.merges.append(MergeV1(position=fact.position, primary=primary.position))
            kept.append(replace(fact, duplicate_of=primary.position))
            continue
        if count == MAX_UNIT_FACTS:
            refuse(fact.position, "fact_count_exceeded")
            continue
        count += 1
        representatives.setdefault(key, []).append(fact)
        kept.append(fact)
    return kept


def _name_groups(
    fact: AdmittedFact, name: str, evidence: Mapping[str, EvidenceRecord],
    fresh: Set[str],
) -> tuple[tuple[QuoteSpanV1, ...], ...]:
    """Locate only whole, literally written names inside this fact's admitted support.

    An inflected or normalized-only occurrence keeps the original fact: a bare-name
    keeper must quote the name itself, rather than an occupation or relationship.
    """
    pattern = re.compile(r"(?<!\w)" + re.escape(name) + r"(?!\w)")
    groups: list[tuple[QuoteSpanV1, ...]] = []
    for group in fact.groups:
        names: dict[tuple[str, int, int], QuoteSpanV1] = {}
        for span in group:
            part = evidence[span.evidence_id]
            text = part.text.encode()[
                span.byte_start - part.byte_start:span.byte_end - part.byte_start
            ].decode()
            for match in pattern.finditer(text):
                start = span.byte_start + len(text[:match.start()].encode())
                located = QuoteSpanV1(evidence_id=span.evidence_id, byte_start=start,
                                      byte_end=start + len(name.encode()))
                names[span_key(located)] = located
        if names and any(span.evidence_id in fresh for span in names.values()):
            groups.append(tuple(names.values()))
    return tuple(groups)


def _bare_naming(
    admitted: Sequence[AdmittedFact], evidence: Mapping[str, EvidenceRecord],
    fresh: Set[str],
) -> list[AdmittedFact]:
    """A naming fact whose descriptor another fact of the reply restates gives only the name.

    ``<descriptor> is named <name>.`` beside ``<name> is <descriptor>.`` with the same
    anchors, kind, ``revises`` and ``corrects_text`` states the relationship twice.
    The naming fact of a new entity becomes ``A <neutral kind> is named <name>.``
    only for a neutral declared type. It anchors only that
    entity and quotes only its name, while the relationship keeps its original support.
    This is the genuine bare-name keeper of D.3: forgetting the relationship withholds
    its words without hiding the name that binds later mentions to its tombstones.
    """
    restated = {key for fact in admitted if (key := _restatement_key(fact)) is not None}
    found: list[AdmittedFact] = []
    for fact in admitted:
        key = _naming_key(fact)
        names = [naming for naming in fact.naming if naming.name_kind == "name"]
        anchor = fact.anchors[names[0].anchor_index] if key is not None else None
        if (key is None or key not in restated or not isinstance(anchor, NewEntityChoiceV1)
                or len(fact.naming) != 1 or fact.kind != "state"
                or fact.time.expression.kind != "none"
                or "time_unlocated" in fact.concerns
                or fact.revises or fact.corrects_text is not None
                or anchor.entity_type not in NEUTRAL_KINDS):
            found.append(fact)
            continue
        groups = _name_groups(fact, names[0].name, evidence, fresh)
        if not groups:
            found.append(fact)
            continue
        kind = anchor.entity_type
        article = "An" if kind[:1].lower() in "aeiou" else "A"
        try:
            statement = canonical_statement(f"{article} {kind} is named {names[0].name}.")
        except ValueError:
            found.append(fact)
            continue
        cited = [evidence[span.evidence_id] for group in groups for span in group]
        trust: TrustTier = min((part.trust_tier for part in cited), key=_TRUST.__getitem__)
        found.append(replace(
            fact, statement=statement, statement_digest=statement_digest(statement),
            anchors=(anchor,), naming=(replace(names[0], anchor_index=0),),
            groups=groups, keywords=(),
            origin=_origin({part.kind for part in cited}), trust_tier=trust,
        ))
    return found


def replay_admission(
    reply: ExtractOutputV12, context: ExtractionContext, profile: Profile,
    stored: ExtractionAdmissionV2 | None, *, pipeline_version: str,
    implied_health_rule: bool = False,
) -> Admission:
    """Recompute the admission of a stored reply and require its stored record (B.7.5).

    Args:
        reply (ExtractOutputV12): The stored verbatim reply.
        context (ExtractionContext): The unit's frozen context.
        profile (Profile): The deployment profile the reply was admitted under.
        stored (ExtractionAdmissionV2 | None): The record stored beside it.
        pipeline_version (str): The owner policy's extraction pipeline.
        implied_health_rule (bool): The owner policy's supplement rule asks for
            implied health statements (``admit``).

    Returns:
        Admission: The recomputed admission.

    Raises:
        AMLError: ``binding_mismatch`` when the recomputed record differs.
    """
    admission = admit(
        reply, context, profile, pipeline_version=pipeline_version,
        implied_health_rule=implied_health_rule,
    )
    if admission.record != stored:
        raise AMLError("binding_mismatch", "Admission replay computed a different record")
    return admission
