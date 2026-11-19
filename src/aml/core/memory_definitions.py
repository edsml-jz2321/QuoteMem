"""The definitions block that both memory prompts render (B.13, C.15; INV O17).

Extraction (template v50) and review (judge prompt v8) must read one meaning of
fact, statement, assertion, naming and time: a definition that differed between
the two once produced zero memories. Both prompts embed
``MEMORY_DEFINITIONS_V1`` byte for byte, and ``MEMORY_DEFINITIONS_DIGEST`` is the
``definitions_digest`` that the extraction policy and the judge binding record,
so changing a definition re-binds both calls.
"""

from aml.core.identity import IntegrityDigest

MEMORY_DEFINITIONS_V1 = (
    "Definitions (shared by extraction and review; they define terms and are never "
    "evidence):\n"
    "- Fact: one proposition about one or more entities that can be corrected, withdrawn "
    "or forgotten on its own. Atomic: no independently asserted part can be removed while "
    "the rest stays true. Independently correctable parts are separate facts even in one "
    "sentence: \"Ana likes chess and opera\" gives one fact about chess and one about opera. "
    "Indivisible: parts that only hold together, such as one event with its participants, "
    "roles, place, purpose, conditions and time, or a name such as \"Research and "
    "Development\". Compound: independently correctable facts packed into one.\n"
    "- Self-contained statement: one third-person sentence that a reader who never saw "
    "the conversation understands and can check. It names each entity by name or by a "
    "distinguishing descriptor (\"the user's cousin Ines\", \"Ana's tortoise\"), never "
    "as I, you, he, she, they or the speaker; it states the fact's polarity, modality and "
    "time; it adds nothing the sources do not say.\n"
    "- Assertion: the speaker presents the content as true. Quoted or reported speech "
    "(words in quotation marks, even a whole turn after a speaker label), questions, "
    "wishes, suggestions, hypotheticals, conditionals, jokes and role-play do "
    "not assert their content; a report holds only as a report (\"Lee said that ...\"). "
    "A question and its explicit answer can assert a fact together.\n"
    "- Intention versus completion: wanting, planning, deciding, signing up, applying, "
    "trying or starting to do something is not having done it; a plan stays a plan until "
    "a source says it happened.\n"
    "- Negation, modality and period: not, no longer, might, should, used to, as a kid and "
    "since 2020 are part of the fact. A fact limited to a period holds only for that "
    "period; the past does not describe the present.\n"
    "- Naming versus owning: a name says what one entity is called; owning, possession or "
    "a relationship links two entities. \"I have a tortoise named Pip\" asserts both: the "
    "user has a tortoise, and that tortoise is named Pip. A possessive that only identifies "
    "the entity is a descriptor, so \"My canoe is called Wren\" is one naming fact. Equal "
    "names never show that two mentions are one entity.\n"
    "- State versus event: a state holds over time (lives in Porto, is left-handed, owns a "
    "kayak, likes opera, a habit, rule or method); an event happens at a time (moved to "
    "Porto, climbed a volcano, sold a piano).\n"
    "- Time expression: the fact's own time words as written (yesterday, in May 2021, "
    "every Monday, as a kid): absolute when it names a calendar date, month or year, "
    "relative otherwise, none when the fact states no time. A message date tells when "
    "something was said, not when the fact holds.\n"
)

MEMORY_DEFINITIONS_DIGEST = IntegrityDigest(
    "f223afbf968bb82162fd0d0abf5fd5b1bf3d94b792dfea9a40dc613eb3eec087"
)
"""SHA-256 of ``MEMORY_DEFINITIONS_V1`` in UTF-8, pinned; a test proves the bytes match."""
