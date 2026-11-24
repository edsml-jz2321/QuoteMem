"""Extraction prompt template agent-v1: template v50 with the ``agent`` Add mode's rules.

The ``agent`` Add mode (``core/extraction_policy.AddMode``) fills a long-lived memory from live
chat a few turns at a time, so a batch often holds a fact and its correction, a mixed
sentence, or a question the user asks the assistant. Template v50 (``extraction_template_v50``)
stays the benchmark modes' template unchanged; this template is v50 with three of its
sentences replaced (`AGENT_RULES`), and only the ``agent`` policy names its digest
(``EXTRACTION_TEMPLATE_DIGESTS``):

- Latest value. v50 says "When the fresh sources state an assertion and
  then its correction, extract both." In one plugin batch ("我住在成都武侯区。" then
  "Correction: last month I moved from Chengdu to Xiamen, and I live in Xiamen now.") both
  facts went to one judge round as siblings, and in four Adds of eight the old city stayed
  current beside the new one or the new state was folded into the move event, so no
  residence was current. The agent template asks for the latest value only, as a plain fact:
  with ``corrects_text`` naming a value no memory holds, the fact waited as
  ``correction_target_unresolved`` or lost its review (three Adds of twelve).
- Atomic statements. "I only drink green tea — 我从来不喝咖啡" was stored as one
  statement, so forgetting the tea also forgot the coffee, and the judge refused other
  compounds ("lives in Xiamen now, having moved from Chengdu") as ``not_atomic``, losing both
  claims. A sentence joining claims that are not items of one list gives one fact per claim,
  across languages.
- Questions. "What sport do I play?" gave "The user plays a sport". A question states
  nothing about the user, not even what it presupposes; a fact said in passing inside a
  request is still a fact.

The user wording ("the authenticated user is \"the user\"") is v50's, so admission's
``user_unattested`` guard (``ingestion/extraction_admission.USER_FORMS``) reads it unchanged.
"""

from __future__ import annotations

import hashlib
from typing import Final

from quotemem.core.extraction_template_v50 import WORKFLOW_EXTRACTION_TEMPLATE_V50
from quotemem.core.identity import IntegrityDigest

_V50_ATOMIC: Final = "One fact per statement, every clause's fact once. "
_V50_SKIP: Final = (
    "Skip greetings, thanks, small talk, filler, remarks about the conversation, unanswered "
    "questions and claims about people in general. "
)
_V50_CORRECTION: Final = (
    "When the fresh sources state an assertion and then its correction, extract both. "
)

AGENT_RULES: Final = {
    _V50_ATOMIC: (
        "One fact per statement, every clause's fact once. Each fact states one claim that can "
        "be corrected on its own: a sentence that joins claims which are not items of one list "
        "(a positive and a negative claim, an activity and a preference, a move and the new "
        "home, an employer and a role, an event and what someone said about it, a name and "
        "where that person lives, clauses in two languages) gives one fact per claim (\"I only "
        "drink green tea — 我从来不喝咖啡\": \"The user only drinks green tea.\" and \"The user "
        "never drinks coffee.\"), each quoting only its own clause. "
    ),
    _V50_SKIP: (
        "Skip greetings, thanks, small talk, filler, remarks about the conversation, unanswered "
        "questions and claims about people in general. A question states nothing about who "
        "asks it, not even what it presupposes: \"What sport do I play?\" gives no fact (never "
        "\"The user plays a sport.\"); a fact the user states in passing inside a question or "
        "request is still a fact. "
    ),
    _V50_CORRECTION: (
        "When the sources state a value and a later fresh source changes or corrects it (a "
        "move, a switch, \"Correction:\", \"actually\", 更正), extract only the latest value, "
        "quoting the later source; never a separate fact for the replaced value, not even in "
        "the past tense (no \"The user lived in A.\" beside \"The user lives in B now.\"). A "
        "dated change "
        "(\"last month I moved from A to B\") may also be an event fact, but the current state "
        "the user states (\"I live in B now\") is its own state fact. A value the fresh sources "
        "replace is no memory, so such a fact has corrects_text null; revises still names a "
        "listed memory that it changes. "
    ),
}
"""Each v50 sentence the agent template replaces, with its replacement."""


def _agent_template() -> str:
    text = WORKFLOW_EXTRACTION_TEMPLATE_V50
    for old, new in AGENT_RULES.items():
        if text.count(old) != 1:
            raise RuntimeError("Template v50 lost a sentence the agent template replaces")
        text = text.replace(old, new)
    return text


WORKFLOW_EXTRACTION_TEMPLATE_AGENT_V1: Final = _agent_template()
"""The ``agent`` Add mode's extraction template (module docstring)."""

WORKFLOW_TEMPLATE_AGENT_V1_DIGEST: Final = IntegrityDigest(
    hashlib.sha256(WORKFLOW_EXTRACTION_TEMPLATE_AGENT_V1.encode("utf-8")).hexdigest()
)
"""SHA-256 of ``WORKFLOW_EXTRACTION_TEMPLATE_AGENT_V1`` in UTF-8; pinned by its test."""
