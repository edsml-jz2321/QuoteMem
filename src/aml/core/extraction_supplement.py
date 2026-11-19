"""The extraction supplement call: an overview and the assistant's work, beside extraction.

The lean extraction templates keep the facts a source states about a person or entity,
so what an assistant produced for the user (a plan it drew up, a recommendation the user
took, a deadline agreed with it) and the order in which a conversation moved from one
topic to the next live only in raw Evidence parts. A broad question ("how has my project
progressed") needs many of those parts, and the 100 Search hits reach few of them.

A policy that names a supplement rule (``extraction_supplement``) sends a second
extract-v12 request with every extraction call, at the same time and over the same user
message: this module's template asks it for only three kinds of fact, so the extraction
template itself is unchanged.

- The assistant's work for the user: its plans, schedules and lists for the user, the
  recommendations the user accepted and the deadlines and decisions agreed with it, each
  stated as the assistant's and quoted verbatim; general knowledge stays out.
- One overview of the fresh sources: what was raised, asked, decided, explained or
  recommended, in source order and by whom, quoting the key words of every point.

Its facts join the extraction reply before admission (``ingestion/extract.merged_reply``),
so they are grounded, admitted, stored and replayed like any other fact. As soon as it
answers, the session rollup call reads its facts (``core/session_rollup.py``), still beside
the extraction call. The template is
template v50 with the supplement passages before section 4, so both calls read the same
rules. Inside the extraction call these two passages helped summary questions, but a
serial call made the Add twice as slow; the separate call keeps the Add's latency near
the slower of the two calls.

The rule ``overview-assistant-rollup-health-v4`` (the rollup rule with the implied
health passage) asks for a third kind:

- An implied health condition: a health or medical condition of the user, or a medical
  history in the user's family, that the user's own words clearly imply and no source
  names, stated ``The user likely has <condition> (implied by: "<quote>").`` with the
  user's words as ``<quote>``. Admission lets such a statement, and only one from this
  call under this rule, name a health class that only its copied quote implies
  (``ingestion/extraction_admission.implied_health``); every other protected class keeps
  the sensitivity floor. It must name a health class and quote no forget request, and any
  other statement of this call marked "(implied by:" is refused. Its memory has origin
  ``implied_health``, which Search reads as an ordinary memory while the overview and
  assistant's-work memories stay summary-only.

Putting the same rule into the extraction template itself (v52) made it fire rarely and
plausibly, but the longer extraction prompt stored fewer memories. Here the extraction
call's prompt and reply are unchanged.
"""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final, Literal

from aml.core.extraction_template_v50 import WORKFLOW_EXTRACTION_TEMPLATE_V50
from aml.core.identity import IntegrityDigest
from aml.core.inference import ReasoningEffort

SupplementRule = Literal["overview-assistant-rollup-health-v4"]
"""The one supplement rule a policy may freeze: this template, sent without reasoning, then
the session rollup call over its facts (``core/session_rollup.py``). v4 is the rollup
rule v3 with the implied health passage."""

BEFORE_SECTION_4: Final = "4. Statements."
"""The opening words of section 4; the supplement passages precede them."""

SCOPE: Final = (
    "Supplement. Another call extracts every fact that sections 3 and 3a ask for from these "
    "same sources; this call returns only the three kinds of fact below and none of those. "
    "new_entities and unresolved_referents are both []: anchor each fact to user and to "
    "handles that entities lists, never to a new or unresolved handle. Skip a kind the fresh "
    "sources do not hold.\n"
)
"""The passage that limits the supplement call to its three kinds of fact."""

ASSISTANT_WORK: Final = (
    "Assistant's work for the user: what an assistant source produced for this user or agreed "
    "with the user: a schedule, plan, timeline or list it drew up for the user, a "
    "recommendation or solution the user accepted or applied, and a deadline, decision or next "
    "step agreed with the user. State each as the assistant's, with its specific names, "
    "figures, versions and dates (\"The assistant drew up a four-week plan for the user: two "
    "runs a week, then three.\", \"The assistant recommended a standing desk, and the user "
    "agreed to buy one.\"), quoting the assistant's words verbatim and, for an acceptance or "
    "agreement, the user's words too. Never state an assistant's proposal as the user's fact "
    "or decision unless the user's words accept it, and leave out general knowledge an "
    "assistant explains (definitions, how something works, sample code).\n"
)
"""The passage asking for the assistant's user-specific substantive content."""

OVERVIEW: Final = (
    "Overview: one fact that recounts the fresh sources as a whole, unless they hold only "
    "greetings, thanks or small talk; it is the one fact about the conversation itself. Its "
    "statement, at most 700 characters, says in source order what was discussed: each topic "
    "raised, question asked and problem met, and each decision, plan, explanation, "
    "recommendation, figure or result given, naming who gave it (\"The user asked how to wax a "
    "canoe; the assistant recommended three thin coats of paste wax; the user planned to start "
    "on Saturday.\"). It keeps specific names, figures, versions and dates, and states what an "
    "assistant or another participant said as theirs, never as a fact about the user. Its one "
    "support group quotes, verbatim, the key words of each point it states (at most 16 "
    "quotes). kind is event, time kind none, anchors user and the listed entities it names, "
    "and keywords its topics.\n"
)
"""The passage asking for one overview fact of the fresh sources."""

IMPLIED_HEALTH: Final = (
    "Implied health: when the user's own words clearly imply a health or medical condition "
    "of the user, or a medical history in the user's family, that no source names, one state "
    "fact 'The user likely has <condition> (implied by: \"<quote>\").' that copies its "
    "support quote, the user's own words, as <quote>, with a health finding on that quote "
    "(\"I squint at bus numbers until my specs are on\": \"The user likely has "
    "nearsightedness (implied by: \\\"I squint at bus numbers until my specs are "
    "on\\\").\"). Name only the mildest condition those words alone make clear; never imply "
    "one from a topic only mentioned, a guess, an assistant's or another speaker's words, or "
    "the user's background, culture, gender, age or work, and never imply religion, political "
    "affiliation, sexual orientation, gender identity or a preference this way.\n"
)
"""The passage asking for implied health conditions, the template v52 rule."""


def _insert_before(template: str, passage: str, sections: str) -> str:
    """Insert ``sections`` right before the single occurrence of ``passage`` in ``template``."""
    if template.count(passage) != 1:
        raise ValueError("A template section must precede exactly one passage")
    at = template.index(passage)
    return template[:at] + sections + template[at:]


WORKFLOW_SUPPLEMENT_TEMPLATE_V2: Final = _insert_before(
    WORKFLOW_EXTRACTION_TEMPLATE_V50, BEFORE_SECTION_4,
    SCOPE + ASSISTANT_WORK + OVERVIEW + IMPLIED_HEALTH,
)
"""The supplement call's whole system message: template v50 with its four passages (v1, the
same without the implied health passage, is retired)."""

WORKFLOW_SUPPLEMENT_TEMPLATE_V2_DIGEST: Final = IntegrityDigest(
    "452abf47dec9c4f91c0bacfbc3f88904dcb7b0b64bbe8e2c80cd17b731824182"
)
"""SHA-256 of ``WORKFLOW_SUPPLEMENT_TEMPLATE_V2`` in UTF-8."""

SUPPLEMENT_REASONING_EFFORT: Final[ReasoningEffort] = "low"
"""The effort the supplement request binds; with its thinking disabled the gateway reads none.

The supplement only recounts and quotes, so it goes without reasoning. With reasoning at low
effort it was often the slower of the Add's two calls, so the Add waited for the slower of
two heavy-tailed calls. Every statement stays within the strict
schema's 1,024 characters; the overview's 700 is the prompt's aim, and a longer statement is
cut at its last clause within the 1,024 when it joins the reply
(``ingestion/extract.merged_reply``), since the model does not keep to either."""

SUPPLEMENTS: Final[Mapping[SupplementRule, tuple[str, ReasoningEffort]]] = MappingProxyType({
    "overview-assistant-rollup-health-v4": (
        WORKFLOW_SUPPLEMENT_TEMPLATE_V2, SUPPLEMENT_REASONING_EFFORT,
    ),
})
"""Each supplement rule's system message and the effort its request binds."""

IMPLIED_HEALTH_RULES: Final = frozenset[SupplementRule]({"overview-assistant-rollup-health-v4"})
"""The supplement rules that ask for implied health statements, whose admission lifts the
health floor for them alone (``ingestion/extraction_admission.admit``)."""
