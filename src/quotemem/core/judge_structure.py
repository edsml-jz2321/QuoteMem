"""What a judge member's own form fixes about its verdict.

Some parts of a judge verdict follow from the member's own statement, its targets
and the contract rather than from a reading of its sources. They are applied in two
ways, both deterministic functions of the member input, so a member that meets no
rule is sent and accepted exactly as before:

- **A narrower question** for a member-level label. The member's advertised result
  schema omits the values its form rules out, and acceptance refuses a
  result that still uses one as ``result_schema``, which the batch resends.

  1. *Paired repair.* A correction ``X, not Y`` (or ``X instead of Y``, ``X rather than
     Y``) whose ``Y`` names exactly what a current target (a memory or a sibling)
     states in place of ``X`` is one replacement proposition, so ``compound`` is ruled
     out. A core with a conjunction, one that adds more content words than it
     replaces, or one that adds an ownership or naming verb is not recognized. A
     repair of a retired version is a history correction, whose judged
     atomicity was never in question, so it is asked as before.
- **A settled relation** where the judge's decision splits in two. The
  judge decides whether a target held and ended or was wrong from the start; whether
  the member states a replacement value is a property of its own statement. A member
  that only denies a target never states one.

  3. *Denial.* A statement with one negation whose content words (at least two) all
     occur in a positive target statement, without that statement's trailing
     ``, not ...`` contrast, only denies the target: an accepted ``correct`` of it is
     stored as ``retract``.
  4. *Dropped name.* For a naming head denied by a member with no naming annotation,
     an accepted ``update`` is stored as ``retract``: a name dropped with no new one
     is retract, and the resolution gates would void the update.

- **A settled attribution** for a speaker's own explicit report or one about someone else.
  The judge reads such a report as another person's statement,
  ``inferred``; the policy publishes it as an attributed fact.

  2. *Literal self-report.* An explicit report of the speaker's own experience,
     feeling, event or attribute is direct support, including a participant named by
     a transcript line label. Acceptance settles its ``inferred`` as ``direct``;
     it does not narrow the advertised schema. ``self_report`` preserves predicate
     roles, polarity and reporting or uncertainty wording. It also reads a definite
     event reference after one explicit attendance event, an evaluation of that
     event or its described content, and a coordinated predicate of that event.
     Mentioning a protected topic or someone else's attribute never supplies the
     speaker's own attribute. Rule 5 separately settles third-party reports.

  5. *Explicit third-party report.* A sensitive statement (any class but
     ``credentials``, whatever anchor the extractor bound) is the speaker's explicit
     report when it restates the sentence holding one of its user-message quotes (quote
     fragments are read in their sentence): a subject phrase followed by a contiguous
     run of that sentence right after the sentence's opening subject phrase (``is gay``
     after ``My sister``), the subject phrases corresponding. The sentence's ``my`` and
     a description of at most three tokens is restated as the reporter's possessive
     (``the user's`` for an unlabelled message, ``<speaker>'s`` for a labelled one)
     with that description or its ending (``my brother`` as ``the user's brother``; an
     informal kinship word may be spelled formally, ``my mom`` as ``the user's
     mother``); a capitalized name of at most three words is restated as said or as the
     name ending that description (``Anna`` for ``Anna`` or ``my friend Anna``). The
     sentence has no hedge, quotation mark or question mark; a negation inside the
     predicate is part of the run, so polarity is restated as said. An accepted
     ``inferred`` is stored as ``direct``, which publication still allows only with a
     supported group of user messages. A guessed, hedged, relayed, questioned,
     subordinate, paraphrased or inferred trait is not recognized and stays as judged;
     a self-report opens with ``I`` and never corresponds.

     The same restatement is read in more places of a fully shown message: a clause after ``and``,
     ``but`` or a semicolon whose subject is a description, a name or ``he``/``she``
     (``I love my sister and she is gay``); a predicate coordinated with the subject's first one
     (``My sister is a nurse and is gay``); a relative clause after ``who`` or an appositive
     ``, a ...,`` attached to a description or name
     (``My brother, who is allergic to peanuts, is visiting``), and the opening subject's own
     predicate after one; and a sentence opening with ``he`` or ``she``
     (``I visited my mom. She is a devout Hindu.``). A pronoun stands for the one third party its
     line mentions before it that agrees in number and gender, unknown gender agreeing with both;
     with two or more such candidates, or withheld text or a quotation mark before it, nothing is
     settled. No clause after a subordinating word is read, and a sentence with a relayed, doubted
     or alternative wording (``according to``, ``hope``, ``or``) is never a report. A possessive
     chain is restated whole: ``my brother's wife`` is never ``the user's wife``.

- **A settled tombstone verdict** for a bounded earlier episode. The
  judge decides whether a member restates a tombstone "for the same occurrence"; when
  the member is that state's own predicate placed in a bounded earlier period, it
  concerns another period (``other``), which its form shows.

  6. *Bounded earlier episode.* A ``restates`` verdict on a tombstone that holds a
     state as said (kind ``state``, untimed: no time text or bound) is stored as
     ``other`` when the member's own time text names a bounded earlier period (a life
     stage: ``as a student``, ``when I was ten``, ``in college``, ``growing up``; or a
     short span: ``for a summer``, ``for two weeks``) sharing no content word with the
     tombstone, and the member states every content word of the tombstone with its
     present predicate in the past (``lived`` for ``lives``, ``was`` for ``is``,
     ``had`` for ``has``, ``used to``), with no negation and nothing that keeps it
     current or recurring (``still``, ``now``, ``since``, ``again``, a present-tense
     ``is`` or ``has``). "The user lived in a treehouse for a month as an intern."
     against "The user lives in a treehouse." is recognized; "The user has lived in a
     treehouse for years.", "The user was born on the 9th of June." against "The
     user's birthday is June 9." (the period is the forgotten content) and every timed
     tombstone are not.
  8. *Bounded earlier episode of a habit.* A ``restates`` verdict on a tombstone that
     states a recurring present habit with its own schedule (``every Friday evening``,
     ``on weekends``, ``weekly``, ``twice a week``, in its statement or time text, with no
     past marker and no end) is stored as ``other`` when the member's own time text names
     a bounded past period (a life stage or short span as in rule 6, ``last summer``,
     ``three years ago``), the member names no schedule, has no negation and nothing
     that keeps it current, and shares a topic word with the habit. "The
     user's uncle taught the user to play chess when the user was eight." against "The
     user plays chess every Friday evening." is recognized; "The user
     played chess every Friday as a child.", "The user has played chess since
     childhood." and every tombstone without a schedule, such as "The user adopted a
     greyhound." (an event whose resulting state restates it), are not.
  9. *Request about a topic.* A ``restates`` verdict on a tombstone is stored as
     ``other`` when the member is one request or preference about the assistant raising
     the tombstone's topic and states none of its asserted value: "The
     user" asked, requested, told, wants, prefers or would like the assistant to (stop)
     ask, talk about, mention, discuss, bring up, comment on, remind or question
     something, as one clause (no comma, semicolon, ``because``, ``since``, ``but``,
     ``anyway``), with no number and no ``that`` clause for its topic. The tombstone is
     read as "The user's <attribute> is <value>", "The user is <value>" (after a number
     its last word is the attribute: "2.1 meters tall") or "The user <verb> <value>"
     (the verb is the attribute); any other shape is not settled. The value must hold a
     topic word, no value word may be in the member (``_kin``: equal, or the same first
     five letters), and an attribute word the member shares must lie in the request's
     topic. "The user wants to stop being asked how much they weigh." against "The user
     weighs 61 kilograms." is recognized (the weight is the topic, its value is absent);
     "The user wants to stop being asked whether they are a pilot." against "The user is
     a pilot.", "The user wants to talk about their move to Lisbon." against "The user
     lives in Lisbon." (the value is named) and any request stating a number or a reason
     are not.

- **A settled sibling relation** for unrelated members of one Add. A
  ``same`` makes the member a duplicate of its sibling: the sibling's memory carries
  the member's support, so forgetting either withholds the other's quotes.
  Two statements that share no topic word cannot state one fact, which their forms
  show.

  7. *Unrelated sibling.* A ``same`` relation to a sibling is stored as
     ``independent`` when both statements have a topic word and they share none. A
     topic word is a content word (an informal kinship word read formally, ``mom`` as
     ``mother``) other than a preposition, the user referent (``user``, ``I``, ``my``,
     ``me``), a personal pronoun or ``have``. "The user has a cousin." against "The
     user's birthday is June 9." is recognized; "The user was born on June 9." against
     the same sibling (``june``, ``9``) and every ``same`` of a current memory are not.

- **A settled atomicity** for one attribute of an identified subject.

  10. *Identifying subject descriptor.* A singular person or pet relation followed by
     its capitalized name (``the user's sister Zarela``), or the same relation as a
     possessive appositive (``Zarela, the user's sister,``), identifies the subject of
     one attribute; the descriptor is not another assertion. A judged ``compound`` is
     stored as ``atomic`` for a full single-attribute form: a residence, work or study
     location; one copular quality or class; or one simple possession, preference or
     health attribute. The descriptor has no modifier or relative clause, and the
     predicate has no coordination, second clause, naming or added attribute. Other
     forms and other atomicity verdicts stay as judged.

- **A named replacement** that an ``agent`` release's resolution and forget read, never
  acceptance: the stored verdict stays as judged.

  11. *Replaced value beside the new one.* ``X instead of Y`` (or ``X rather than Y``, ``X,
     not Y``, ``Instead of Y, X``) is one replacement of a current target when ``Y`` names
     only what the target states, with a word besides a generic verb, and ``X`` is the
     target's subject with one plain predicate, opened or closed at most by a time phrase of
     a closed list (``since the move``, ``now``, ``these days``) (``one_replacement``):
     resolution then reads a ``compound`` verdict whose one change is the update of that
     target as one fact, the update recording ``Y``. "Since the move, the user goes
     climbing at a gym near West Lake instead of playing badminton." against "The user
     plays badminton every Saturday morning at a club in Xuhui." is recognized. Any other
     clause, cause, event, person, possessive or name in ``X`` is not: "After moving to
     Hangzhou for a job, ...", "... since the user broke an ankle", "... with the user's
     girlfriend Lily ...", nor is a ``Y`` the target does not state ("instead of tennis")
     or one naming only a verb ("instead of playing"). ``X`` replaces like with like:
     every word it adds to the target is an activity, food or drink, way of getting about,
     place or schedule (the allowlist ``_VALUE_WORDS``), so no condition, status, role,
     person or cause ("on bail", "asthmatic", "a felon") passes; it states no condition in
     place of an activity ("is pregnant", ``_STATE_VERBS``) and nothing
     ``sensitivity_rules.classify`` finds; and a capitalized name is only a two-word natural
     place (``_NATURE_HEADS``: "near West Lake", never "near Lily" or "Rehab Park") or one
     word in place of the replaced value's own ("Hangzhou, not Shanghai", never "Hangzhou
     Prison, not Shanghai"). A complete forget reads the parts of a
     statement that name a value as replaced, ended or denied (``ended_parts``) by their
     words that name a value of their own (``distinctive_words``): a memory that replaced
     a forgotten one and still names its value is forgotten with it
     (``store.forget_admission.names_replaced``, ``store.forget_run.ForgetOwner``).

  A relation, an attribution or a tombstone verdict is settled rather than narrowed in
  the schema because the judge decides it while reasoning, before the strict output is
  decoded: a narrowed enum made such members emit incoherent relations,
  and the request bytes stay those every safety measurement used.

The forms are recognized in plain English only; a statement or quote in another
script is never constrained or settled.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Final

from quotemem.core.add_read_view import Sigma
from quotemem.core.readability import WITHHELD_MARKER
from quotemem.core.sensitivity_rules import classify

_OTHER_SCRIPT: Final = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af\uf900-\ufaff]")
_WORD: Final = re.compile(r"\w+")
_CONTRAST: Final = re.compile(
    r"(?:,\s*not|,?\s+rather\s+than|,?\s+instead\s+of)\s+(?P<replaced>[^,;]+?)\s*\.?\s*$",
    re.IGNORECASE,
)
_NEGATIONS: Final = frozenset({
    "not", "never", "no", "nobody", "nothing", "none", "neither", "nor", "cannot",
})
_CONTRACTED: Final = frozenset({
    "don", "doesn", "didn", "isn", "aren", "wasn", "weren", "hasn", "haven", "hadn", "won",
    "wouldn", "can", "couldn", "shouldn", "mustn", "mightn", "needn", "ain",
})
"""Tokens that a following ``t`` (from ``n't``) turns into a negation."""
_FUNCTION: Final = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "am", "be", "been", "being", "do", "does",
    "did", "s", "will", "would", "shall", "should", "can", "could", "may", "might", "must",
    "ever", "any", "more", "longer", "anymore", "now", "it", "its", "he", "she", "they", "him",
    "her", "them", "his", "their", "this", "that", "these", "those",
})
"""Words a content comparison ignores: articles, auxiliaries, pronouns, time particles."""
_HAVE: Final = {"has": "have", "have": "have", "had": "have", "having": "have"}
_SUFFIXES: Final = ("ing", "ies", "ied", "es", "ed", "s", "e", "d")
_ADDED_PREDICATES: Final = frozenset({"have", "nam", "call", "own"})
"""Stems that add an ownership or naming fact when a repair core brings them in."""
_CONJUNCTIONS: Final = frozenset({"and", "but", "or"})
_LEADING_CONTRAST: Final = re.compile(
    r"^\s*(?:instead\s+of|rather\s+than)\s+(?P<replaced>[^,;]+?)\s*,\s*(?P<core>.+?)\s*$",
    re.IGNORECASE,
)
"""A replaced value named before the new one: "Instead of Y, X" (rule 11)."""
_REPLACED_MARKER: Final = re.compile(
    r"(?:,\s*not|,?\s+rather\s+than|,?\s+instead\s+of)\s+", re.IGNORECASE)
"""What names a replaced value after the new one: "X instead of Y" (rule 11)."""
_TIME_PHRASES: Final = tuple(phrase.replace(" ", r"\s+") for phrase in (
    r"(?:since|after) (?:the|my|the user['\u2019]s) move", r"at the moment", r"these days",
    r"since then", r"at present", r"now", r"nowadays", r"lately", r"recently", r"currently",
    r"today",
))
"""The only phrases a single replacement may open or close with (rule 11). Each names when
the new value holds and no other fact: "since the user broke an ankle" or "after moving to
Hangzhou for a job" carries one, so an open time phrase is never read."""
_REPLACEMENT_JOINS: Final = _CONJUNCTIONS | frozenset({
    "who", "whom", "whose", "which", "that", "where", "when", "while", "whereas", "because",
    "although", "though", "if", "unless", "named", "called", "with", "without", "since",
    "after", "before", "as", "until", "till", "following", "during", "despite", "besides",
    "also", "too", "plus", "for", "from", "by", "having", "being", "so", "then", "than",
    "instead", "rather", "except", "unlike", "whenever", "once", "via",
})
"""Words of a single replacement's new value that join another clause, person, cause, event
or naming fact to it (rule 11)."""
_PLACE_PREPOSITIONS: Final = frozenset({
    "at", "in", "near", "on", "around", "outside", "inside", "across", "along", "beside",
    "behind", "opposite",
})
"""Words a capitalized place name of a single replacement follows ("near West Lake")."""
_DESTINATIONS: Final = frozenset({
    "the", "a", "an", "work", "school", "class", "college", "university", "campus", "town",
    "bed",
})
"""What ``to`` may lead to in a single replacement: a place, never a purpose ("to lose
weight") (rule 11)."""
_POSSESSIVES: Final = frozenset({"s", "my", "his", "her", "their", "our", "your", "its"})
"""Possessive tokens; ``user's`` reads ``user`` ``s`` (rule 11)."""
_THIRD_PARTIES: Final = frozenset({
    "he", "she", "him", "her", "they", "them", "we", "us", "someone", "somebody", "anyone",
    "everyone", "people",
})
"""Pronouns and indefinites that bring another person into a single replacement (rule 11)."""
_GENERIC_VERBS: Final = frozenset({
    "play", "plays", "played", "playing", "go", "goes", "went", "going", "gone", "do", "does",
    "did", "doing", "done", "have", "has", "had", "having", "use", "uses", "used", "using",
    "take", "takes", "took", "taking", "taken", "make", "makes", "made", "making", "get",
    "gets", "got", "getting", "gotten",
})
"""Verbs that name no value of their own: a replaced value needs another word ("instead of
playing badminton", never "instead of playing") (rule 11)."""
_CALENDAR: Final = frozenset({
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "mondays",
    "tuesdays", "wednesdays", "thursdays", "fridays", "saturdays", "sundays", "january",
    "february", "march", "april", "may", "june", "july", "august", "september", "october",
    "november", "december",
})
"""Capitalized days and months a single replacement's new value may name (rule 11)."""
_REPLACEMENT_WORDS: Final = 8
_CLAUSE_OPENER: Final = (r"\s+(?:and|but|then|so|or|while|because|after|before|since|when|"
                         r"to|with)\b")
"""What ends an ended or left value inside one clause: a coordinated clause or an adjunct
("left Northwind and joined Fabrikam last week", "quit the band with my friends")."""
_LEFT_FROM: Final = re.compile(
    r"\b(?:moved|moves|moving|switched|switches|switching|changed|changes|changing|"
    r"transferred|relocated|went)\b[^,;.]*?(?P<part>\bfrom\s+(?P<old>[^,;.]+?))"
    rf"(?={_CLAUSE_OPENER}|[,;.]|$)",
    re.IGNORECASE)
"""What a move or a switch left: "moved from Shanghai to Hangzhou" (rule 11)."""
_ENDED_OBJECT: Final = re.compile(
    r"\b(?:sold|sells|selling|gave\s+away|gave\s+up|gives\s+up|giving\s+up|quit|quits|"
    r"left|leaves|dropped|drops|ditched|replaced|replaces|stopped|stops|retired\s+from|"
    r"broke\s+up\s+with|got\s+rid\s+of)\s+(?P<old>[^,;.]+?)"
    rf"(?={_CLAUSE_OPENER}|[,;.]|$)",
    re.IGNORECASE)
"""What an ending verb took: "sold the blue Toyota", "has replaced Saturday badminton"; never
a coordinated clause after it ("and joined Fabrikam last week")."""
_ENDING_WORDS: Final = frozenset({
    "longer", "anymore", "stopped", "stops", "quit", "quits", "formerly", "previously",
    "former", "gave",
})
"""Words that end a value: a part holding one names that value as ended (rule 11)."""
_CATEGORY_WORDS: Final = frozenset({
    "food", "foods", "dish", "dishes", "meal", "meals", "drink", "drinks", "beverage", "car",
    "cars", "vehicle", "vehicles", "sport", "sports", "game", "games", "club", "clubs", "team",
    "teams", "place", "places", "thing", "things", "stuff", "activity", "activities", "hobby",
    "hobbies", "job", "jobs", "work", "company", "companies", "business", "music", "movie",
    "movies", "film", "films", "show", "shows", "book", "books", "school", "class", "classes",
    "house", "home", "city", "town", "country", "language", "pet", "pets", "animal",
    "animals", "person", "people", "other", "another", "any", "some", "kind", "type", "sort",
    "way", "time", "times", "day", "days", "week", "weeks", "weekend", "weekends", "weekday",
    "weekdays", "morning", "mornings", "evening", "evenings", "afternoon", "afternoons",
    "night", "nights", "month", "months", "year", "years", "routine", "schedule", "habit",
    "old", "new", "usual", "regular", "one", "every", "each",
})
"""Category, time and quantity words: a value they share names no value of its own ("sushi
rather than any other food" against "The user's favourite food is pizza.") (rule 11)."""
_STATE_VERBS: Final = frozenset({
    "is", "are", "am", "was", "were", "be", "been", "become", "becomes", "became", "get",
    "gets", "got", "stay", "stays", "stayed", "remain", "remains", "remained", "feel", "feels",
    "felt", "seem", "seems", "seemed",
})
"""Verbs that state a condition of the subject: a single replacement states one only in
place of one ("is Cantonese food rather than Sichuan food"), never in place of an activity
("is pregnant instead of playing badminton") (rule 11)."""
_NATURE_HEADS: Final = frozenset({
    "lake", "river", "mountain", "mountains", "hill", "hills", "bay", "beach", "island",
    "valley", "coast",
})
"""The place heads a two-word name a single replacement brings may end in: no person's or
institution's name ends in one ("Tom Park", "Rehab Park", "Hangzhou Prison") (rule 11)."""
_VALUE_WORDS: Final = frozenset({
    # activities, as verbs and as nouns
    "go", "goes", "went", "play", "plays", "played", "practice", "practices", "practise",
    "practises", "do", "does", "take", "takes", "took", "ride", "rides", "drive", "drives",
    "climb", "climbs", "climbing", "swim", "swims", "swimming", "run", "runs", "running",
    "jog", "jogs", "jogging", "hike", "hikes", "hiking", "walk", "walks", "walking", "cycle",
    "cycles", "cycling", "bike", "bikes", "biking", "row", "rows", "rowing", "sail", "sails",
    "sailing", "surf", "surfs", "surfing", "ski", "skis", "skiing", "skate", "skates",
    "skating", "dive", "dives", "diving", "fish", "fishes", "fishing", "dance", "dances",
    "dancing", "sing", "sings", "singing", "paint", "paints", "painting", "draw", "draws",
    "drawing", "cook", "cooks", "cooking", "bake", "bakes", "baking", "garden", "gardens",
    "gardening", "read", "reads", "reading", "write", "writes", "writing", "bouldering",
    "boxing", "yoga", "pilates", "tennis", "badminton", "squash", "football", "soccer",
    "basketball", "volleyball", "baseball", "golf", "chess", "guitar", "piano", "violin",
    "drums", "lessons", "classes", "class", "workouts", "workout", "rock",
    # food and drink
    "drink", "drinks", "eat", "eats", "tea", "green", "black", "coffee", "juice", "water",
    "milk", "beer", "wine", "soda", "food", "cuisine", "dish", "dishes", "rice", "noodles",
    "bread", "pizza", "sushi", "salad", "breakfast", "lunch", "dinner", "cup", "cups",
    # getting about and living
    "commute", "commutes", "bus", "bicycle", "car", "train", "subway", "metro", "tram",
    "taxi", "scooter", "motorbike", "motorcycle", "ferry", "foot", "live", "lives", "work",
    "works", "study", "studies", "favourite", "favorite",
    # places
    "gym", "club", "park", "lake", "pool", "court", "courts", "field", "studio", "track",
    "trail", "trails", "beach", "mountain", "mountains", "hill", "hills", "river", "hall",
    "stadium", "rink", "office", "home", "school", "campus", "library", "cafe", "restaurant",
    "downtown", "city", "district", "neighborhood", "neighbourhood", "nearby", "near",
    "local", "indoor", "outdoor",
    # schedules
    "every", "each", "daily", "weekly", "weekend", "weekends", "weekday", "weekdays",
    "morning", "mornings", "evening", "evenings", "afternoon", "afternoons", "night",
    "nights", "day", "days", "week", "weeks", "month", "months", "time", "times", "twice",
    "sometimes", "often", "usually", "regularly", "mostly", "currently", "nowadays", "lately",
    "recently", "today",
})
"""The only words a single replacement's new value may add to its target (rule 11): an
activity, food or drink, a way of getting about, a place or a schedule, the kinds a replaced
value is. Any other word (a condition, status, role, person or cause: "on bail",
"asthmatic", "on insulin", "a felon") is another fact, so the statement is refused; a
capitalized name is read apart (``_names_in_place``)."""
"""Topic words a single replacement's new value may add to its target's (rule 11): "goes
climbing at a gym near West Lake" adds six."""
_HEDGES: Final = frozenset({
    "think", "thinks", "thought", "guess", "guessed", "maybe", "perhaps", "probably",
    "possibly", "might", "may", "could", "believe", "believes", "believed", "seem", "seems",
    "seemed", "apparently", "suppose", "supposedly", "rumor", "rumour", "heard", "says", "said",
    "told", "tells", "claims", "claimed", "if", "would", "wonder", "whether", "suspect",
    "suspects",
})
"""Quote words that make a report a guess, a relayed claim or a supposition."""
_DESCRIPTION_TOKENS: Final = 3
"""Tokens a reported third party's description or name may hold (rule 5)."""
_REPORT_LEAD: Final = frozenset({"and", "but", "so", "also", "oh", "well", "plus"})
"""Words a reported sentence may open with before its subject phrase (rule 5)."""
_NOT_A_NAME: Final = frozenset({
    "i", "we", "you", "he", "she", "they", "it", "me", "us", "him", "her", "them", "my", "our",
    "your", "his", "their", "its", "the", "a", "an", "this", "that", "these", "those",
    "someone", "somebody", "everyone", "everybody", "people", "user",
})
"""Words that never stand in a reported third party's description or name (rule 5)."""
_PERIOD: Final = re.compile(
    r"\b(?:as an? (?:student|child|kid|teen|teenager|boy|girl|undergrad|undergraduate"
    r"|grad student|graduate student|exchange student|freshman|sophomore|intern)"
    r"|when (?:i|he|she|they|we|you) (?:was|were) (?:a |an |in )?[a-z0-9-]+"
    r"|in (?:college|high school|middle school|elementary school|grad school|university)"
    r"|growing up"
    r"|for (?:a|an|one|two|three|a few|a couple of) (?:summers?|winters?|springs?|falls?"
    r"|autumns?|weeks?|weekends?|months?|semesters?|terms?|days?|nights?))\b",
    re.IGNORECASE,
)
"""A bounded earlier period a member's own time text may name (rule 6): a life stage or
a short span."""
_CONTINUING: Final = frozenset({
    "is", "are", "am", "has", "have", "does", "do", "still", "now", "currently", "since",
    "anymore", "nowadays", "today", "again", "ever",
})
"""Words that keep a statement's fact current or recurring: no bounded episode (rules 6
and 8)."""
_RECURRENCE: Final = re.compile(
    r"\b(?:every|each) (?:other )?(?:day|night|morning|afternoon|evening|week|weekend|month"
    r"|year|summer|winter|spring|fall|autumn|monday|tuesday|wednesday|thursday|friday"
    r"|saturday|sunday)s?\b"
    r"|\bon (?:mondays|tuesdays|wednesdays|thursdays|fridays|saturdays|sundays|weekends"
    r"|weekdays)\b"
    r"|\b(?:daily|nightly|weekly|biweekly|fortnightly|monthly|yearly|annually)\b"
    r"|\b(?:once|twice|thrice|[a-z0-9]+ times) (?:a|per) (?:day|week|month|year)\b",
    re.IGNORECASE,
)
"""A schedule that makes a statement a recurring habit (rule 8)."""
_PAST_SPAN: Final = re.compile(
    r"\blast (?:summer|winter|spring|fall|autumn|week|weekend|month|year|night)\b"
    r"|\b(?:a|an|one|two|three|four|five|six|seven|eight|nine|ten|a few|a couple of"
    r"|several|[0-9]+) (?:days?|weeks?|months?|years?|decades?) ago\b",
    re.IGNORECASE,
)
"""A bounded past span a member's own time text may name besides ``_PERIOD`` (rule 8)."""
_PAST_MARKERS: Final = frozenset({
    "used", "was", "were", "did", "had", "ago", "last", "former", "formerly", "previously",
})
"""Words that place a tombstone's habit in the past rather than the present (rule 8)."""
_REQUESTS: Final = frozenset({
    "asked", "asks", "requested", "requests", "told", "tells", "wants", "wanted", "want",
    "prefers", "preferred", "prefer", "like", "rather",
})
"""Words a request or preference about the assistant opens with (rule 9)."""
_INQUIRIES: Final = frozenset({
    "ask", "asks", "asked", "asking", "talk", "talks", "talked", "talking", "mention",
    "mentions", "mentioned", "mentioning", "discuss", "discusses", "discussed", "discussing",
    "bring", "brings", "brought", "bringing", "raise", "raises", "raised", "raising",
    "comment", "comments", "commented", "commenting", "remind", "reminds", "reminded",
    "reminding", "question", "questions", "questioned", "questioning",
})
"""Verbs of the assistant raising a topic that a request is about (rule 9)."""
_NUMBER_WORDS: Final = frozenset({
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
    "nineteen", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
    "hundred", "thousand", "million", "dozen", "half",
})
"""Number words; a request that states a number may state the forgotten value (rule 9)."""
_REASONS: Final = frozenset({
    "because", "since", "though", "although", "anyway", "actually", "but",
})
"""Words that join a second clause, such as a reason, to a request (rule 9)."""
_CLAUSE_MARKS: Final = re.compile(r"[0-9,;:]")
_COPULAS: Final = frozenset({"is", "are", "am", "was", "were"})
"""Forms of ``be`` that join a tombstone's subject or attribute to its value (rule 9)."""
_PAST_OF: Final = {"was": ("is", "are", "am"), "were": ("is", "are", "am"),
                   "had": ("has", "have")}
"""Past forms of the present auxiliaries a state may be said with (rule 6)."""
_PREPOSITIONS: Final = frozenset({
    "on", "in", "at", "as", "for", "of", "to", "with", "by", "from", "into", "during",
})
"""Words rule 6 leaves out when it compares a period with a tombstone's content, and rule 7
when it compares two statements' topics."""
_UNTOPICAL: Final = frozenset({
    "user", "i", "m", "my", "me", "mine", "we", "our", "us", "you", "your", "re", "ve",
    "ll", "d", "have",
})
"""Words that name no topic in a statement about the user (rule 7): the referent, personal
pronouns and their contractions, and ``have``."""
FORMAL_KINSHIP: Final = {
    "mom": "mother", "mum": "mother", "mommy": "mother", "mummy": "mother",
    "dad": "father", "daddy": "father", "grandma": "grandmother", "granny": "grandmother",
    "grandpa": "grandfather", "granddad": "grandfather", "bro": "brother", "sis": "sister",
}
"""Informal kinship words a restated description may spell formally (``my mom`` as
``the user's mother``); rule 5 and the forget's referent keeper read it."""
_UNREPORTED_CLASSES: Final = frozenset({"credentials"})
"""Sensitivity classes rule 5 never settles: credentials are refused at admission."""
_QUOTATION: Final = re.compile("[\"\u201c\u201d\u00ab\u00bb]")
_LABEL: Final = re.compile(
    r"^\s*(?P<label>[^\W\d_][\w'\u2019-]*(?:[ \t]+[^\W\d_][\w'\u2019-]*){0,2})[ \t]*:\s*")
"""A leading line label of at most three words (``Blair:``, ``Correction:``)."""
_DISCOURSE_LABELS: Final = frozenset({
    "correction", "update", "note", "edit", "fyi", "ps", "btw", "also",
})
"""Labels that name no speaker, so the message's own reporter keeps the sentence."""
_SENTENCE_END: Final = re.compile(r"[.!?\n]")
_UNCONTRACTED: Final = {
    "don": "do", "doesn": "does", "didn": "did", "isn": "is", "aren": "are", "wasn": "was",
    "weren": "were", "hasn": "has", "haven": "have", "hadn": "had", "won": "will",
    "wouldn": "would", "can": "can", "couldn": "could", "shouldn": "should", "mustn": "must",
    "mightn": "might", "needn": "need", "ain": "is",
}
"""The verb a ``n't`` contraction token stands for, so ``isn't`` reads ``is not``."""
_REPORT_DOUBT: Final = frozenset({
    "according", "allegedly", "reportedly", "rumored", "rumoured", "rumors", "rumours",
    "supposed", "presumably", "seemingly", "likely", "unlikely", "surely", "sure", "unsure",
    "uncertain", "doubt", "doubts", "doubted", "hope", "hopes", "hoped", "hoping", "wish",
    "wishes", "wished", "fear", "fears", "feared", "afraid", "worry", "worries", "worried",
    "assume", "assumes", "assumed", "assuming", "imagine", "imagines", "imagined", "pretend",
    "pretends", "pretended", "look", "looks", "looked", "appear", "appears", "appeared",
    "sound", "sounds", "sounded", "kinda", "sorta", "say", "saying", "tell", "telling",
    "claim", "claiming", "hear", "hears", "wondering", "wondered", "or", "either",
})
"""Words besides (2)'s hedges that make a rule-5 sentence relayed, doubted or a choice."""
_PIECE: Final = re.compile(r"\w+|[,;.!?\n]")
"""Words and the punctuation the clause forms of rule 5 read."""
_PUNCTUATION: Final = frozenset({",", ";", ".", "!", "?", "\n"})
_STOPS: Final = frozenset({".", "!", "?", "\n"})
_COORDINATORS: Final = frozenset({"and", "but", ";"})
"""What opens a later clause of the same sentence."""
_SUBORDINATORS: Final = frozenset({
    "if", "unless", "whether", "when", "whenever", "while", "whilst", "once", "until", "till",
    "lest", "that", "which", "who", "whom", "whose", "what", "whatever", "whoever", "how", "why",
    "where", "wherever",
})
"""Words after which a clause may be conditional or embedded, so a later one is not read."""
_SUBJECT_PRONOUNS: Final = {"he": "m", "she": "f"}
"""Pronouns a reported clause may open with; ``they`` is singular or plural, so never."""
_THIRD_PERSON: Final = {
    "he": "m", "him": "m", "his": "m", "himself": "m",
    "she": "f", "her": "f", "hers": "f", "herself": "f",
}
_FEMALE: Final = frozenset({
    "mom", "mum", "mommy", "mummy", "mama", "momma", "mother", "stepmom", "stepmum",
    "stepmother", "grandma", "granny", "grandmother", "gran", "nana", "sister", "sis",
    "stepsister", "aunt", "auntie", "aunty", "daughter", "stepdaughter", "granddaughter",
    "niece", "wife", "girlfriend", "fiancee", "bride", "godmother", "goddaughter", "woman",
    "lady", "girl", "gal", "queen", "princess", "nun", "widow", "actress", "waitress",
    "landlady", "maid", "bridesmaid",
})
_MALE: Final = frozenset({
    "dad", "daddy", "papa", "father", "stepdad", "stepfather", "grandpa", "granddad", "grandad",
    "grandfather", "gramps", "brother", "bro", "stepbrother", "uncle", "son", "stepson",
    "grandson", "nephew", "husband", "boyfriend", "groom", "godfather", "godson", "man", "guy",
    "boy", "gentleman", "dude", "king", "prince", "monk", "widower", "lad", "bloke", "fella",
    "groomsman",
})
_ANYONE: Final = frozenset({
    "friend", "bestie", "bff", "buddy", "pal", "mate", "roommate", "roomie", "flatmate",
    "housemate", "partner", "spouse", "fiance", "cousin", "sibling", "twin", "parent",
    "stepparent", "child", "kid", "baby", "toddler", "infant", "teen", "teenager", "grandchild",
    "grandkid", "godchild", "godparent", "relative", "ex", "crush", "lover", "neighbor",
    "neighbour", "colleague", "coworker", "cofounder", "boss", "manager", "supervisor",
    "employee", "employer", "intern", "assistant", "secretary", "receptionist", "teacher",
    "professor", "prof", "tutor", "student", "classmate", "schoolmate", "teammate", "coach",
    "trainer", "instructor", "mentor", "mentee", "doctor", "doc", "dr", "physician", "surgeon",
    "nurse", "therapist", "psychiatrist", "psychologist", "counselor", "counsellor", "dentist",
    "pharmacist", "vet", "paramedic", "caregiver", "carer", "nanny", "babysitter", "lawyer",
    "attorney", "accountant", "landlord", "tenant", "client", "customer", "patient", "guest",
    "visitor", "stranger", "acquaintance", "pastor", "priest", "rabbi", "imam", "minister",
    "chaplain", "reverend", "bishop", "chef", "cook", "waiter", "server", "bartender",
    "barista", "cashier", "clerk", "driver", "pilot", "officer", "cop", "detective", "soldier",
    "veteran", "firefighter", "plumber", "electrician", "mechanic", "contractor", "builder",
    "gardener", "cleaner", "housekeeper", "hairdresser", "barber", "stylist", "artist",
    "singer", "musician", "writer", "author", "actor", "dancer", "athlete", "player",
    "director", "ceo", "founder", "owner", "president", "mayor", "senator", "judge", "member",
    "volunteer", "leader", "person", "individual", "someone", "somebody", "anyone", "anybody",
    "everyone", "everybody", "nobody", "dog", "doggy", "doggo", "puppy", "pup", "cat", "kitty",
    "kitten", "horse", "pony", "bird", "parrot", "budgie", "rabbit", "bunny", "hamster",
    "gerbil", "pig", "turtle", "tortoise", "fish", "goldfish", "snake", "lizard", "gecko",
    "ferret", "mouse", "rat", "pet", "cow", "goat", "sheep", "lamb", "chicken", "duck",
})
"""People and pets of unknown gender: either pronoun may stand for one."""
_VERB_LIKE: Final = frozenset({
    "cook", "judge", "pet", "coach", "nurse", "mate", "parent", "baby", "server", "date",
})
"""Person nouns often used as verbs: a person only after a determiner or ``my``."""
_GROUPS: Final = frozenset({
    "parents", "folks", "siblings", "brothers", "sisters", "kids", "children", "sons",
    "daughters", "cousins", "friends", "grandparents", "grandkids", "grandchildren", "aunts",
    "uncles", "nieces", "nephews", "neighbors", "neighbours", "roommates", "colleagues",
    "coworkers", "twins", "family", "relatives", "inlaws", "people", "guys", "girls", "boys",
    "men", "women", "ladies", "teammates", "classmates", "students", "teachers", "doctors",
    "nurses", "pets", "dogs", "cats", "team", "staff", "crew", "moms", "dads", "wives",
    "husbands", "bosses", "partners", "guests",
})
_PEOPLE: Final[dict[str, tuple[str, bool]]] = {
    **dict.fromkeys(_ANYONE, ("u", False)), **dict.fromkeys(_GROUPS, ("u", True)),
    **dict.fromkeys(_FEMALE, ("f", False)), **dict.fromkeys(_MALE, ("m", False)),
}
"""A person noun's gender (``f``, ``m`` or unknown ``u``) and whether it is plural."""
_ONE_OF: Final = {
    "mom": "mother", "mum": "mother", "mommy": "mother", "mummy": "mother", "mama": "mother",
    "momma": "mother", "mother": "mother", "dad": "father", "daddy": "father",
    "papa": "father", "father": "father", "wife": "wife", "husband": "husband",
}
"""Relatives a speaker has one of, so two mentions are one person (``my mom``, ``Mom``)."""
_DETERMINERS: Final = frozenset({
    "the", "a", "an", "this", "that", "these", "those", "another", "some", "any", "every",
    "each", "no", "one", "two", "three", "his", "her", "their", "our", "your", "its",
})
_NOT_PERSONS: Final = frozenset({
    "house", "home", "place", "apartment", "flat", "room", "bedroom", "car", "bike", "phone",
    "laptop", "computer", "job", "work", "office", "school", "class", "classes", "birthday",
    "party", "wedding", "trip", "vacation", "holiday", "holidays", "weekend", "day", "week",
    "month", "year", "life", "time", "way", "plan", "plans", "opinion", "mind", "heart", "head",
    "hand", "hands", "health", "stuff", "things", "thing", "money", "bag", "book", "books",
    "garden", "yard", "kitchen", "town", "city", "hometown", "country", "company", "business",
    "project", "homework", "exam", "test", "appointment", "diet", "allergy", "allergies",
    "condition", "diagnosis", "name", "email", "number", "address", "password", "account",
    "food", "meal", "dinner", "lunch", "breakfast", "coffee", "drink", "hobby", "hobbies",
    "career", "dream", "goal", "goals", "visit", "call", "text", "message", "question",
    "problem", "problems", "issue", "help", "advice", "idea", "story", "news", "gift", "present",
    "letter", "photo", "picture", "video", "song", "music", "game", "show", "movie", "recipe",
    "church", "faith", "religion",
})
"""Words after ``my`` that name no person, so ``my car`` is no candidate."""
_WINDOW_BREAKS: Final = frozenset({
    ",", ";", ".", "!", "?", "\n", "s", "and", "or", "but", "nor", "at", "in", "on", "with",
    "to", "from", "for", "of", "about", "by", "into", "over", "under", "after", "before",
    "since", "during", "near", "around", "like", "than", "as", "is", "are", "was", "were",
    "be", "been", "being", "am", "has", "have", "had", "do", "does", "did", "will", "would",
    "can", "could", "should", "may", "might", "must", "shall", "not", "who", "which", "that",
    "whom", "whose", "when", "where", "while", "because", "if", "so", "then", "too", "also",
    "very", "really", "just", "t", "i", "me", "you", "we", "us", "they", "them", "he", "she",
    "him", "her", "his", "hers", "their", "its", "it", "my", "our", "your", "the", "a", "an",
    "this", "these", "those", "yesterday", "today", "tonight", "tomorrow", "now", "again",
    "last", "next", "every",
})
"""Words that end the noun phrase after ``my`` or a determiner."""
_NOT_NAMES: Final = frozenset({
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "january",
    "february", "march", "april", "may", "june", "july", "august", "september", "october",
    "november", "december", "christmas", "easter", "thanksgiving", "halloween", "hanukkah",
    "ramadan", "diwali", "eid", "passover", "catholic", "catholics", "christian", "christians",
    "muslim", "muslims", "islam", "jewish", "jew", "jews", "buddhist", "hindu", "sikh",
    "mormon", "protestant", "orthodox", "baptist", "methodist", "lutheran", "anglican",
    "evangelical", "atheist", "agnostic", "mass", "bible", "god", "jesus", "allah", "american",
    "british", "english", "french", "german", "spanish", "italian", "chinese", "japanese",
    "korean", "indian", "mexican", "canadian", "irish", "russian", "african", "asian",
    "european", "latino", "latina", "hispanic", "arab", "arabic", "there", "here", "then",
    "now", "today", "yesterday", "tomorrow", "tonight", "last", "next", "so", "and", "but",
    "or", "also", "oh", "well", "plus", "yes", "no", "yeah", "ok", "okay", "hi", "hey",
    "hello", "thanks", "thank", "sorry", "please", "just", "actually", "anyway", "btw", "lol",
    "wow", "when", "while", "after", "before", "since", "because", "if", "although", "though",
    "as", "once", "at", "in", "on", "for", "with", "from", "to", "by", "about", "of", "over",
    "during", "what", "who", "why", "how", "where", "which", "maybe", "perhaps", "probably",
    "honestly", "finally", "luckily", "unfortunately", "sadly", "recently", "lately", "still",
    "even", "only", "both", "all", "every", "each", "some", "many", "most", "not", "never",
    "always", "sometimes", "usually", "often", "again", "is", "are", "was", "were", "do",
    "does", "did", "have", "has", "had", "can", "could", "will", "would", "should", "let",
    "speaking", "dear", "great", "good", "nice", "fun", "love", "ps", "fyi", "note", "update",
    "edit", "correction",
})
"""Capitalized words that name no person: dates, faiths, nationalities, sentence openers."""
_LIST_STARTS: Final = frozenset({
    "and", "or", "a", "an", "the", "my", "his", "her", "their", "our", "your",
})
"""Words after an appositive's closing comma that make it an item of a list instead."""
_ATTRIBUTE_NAME: Final = r"[A-Z][\w'\u2019-]*(?: [A-Z][\w'\u2019-]*){0,2}"
"""A bounded capitalized subject, owner or attribute name (rule 10)."""
_ATTRIBUTE_OWNER: Final = (
    rf"(?:[Tt]he user|{_ATTRIBUTE_NAME})['\u2019]s|[Mm]y|[Yy]our|[Hh]is|[Hh]er|[Oo]ur|[Tt]heir"
)
_ATTRIBUTE_DESCRIPTOR: Final = rf"(?:{_ATTRIBUTE_OWNER}) (?P<relation>[a-z]+)"
_IDENTIFIED_SUBJECT: Final = re.compile(
    rf"{_ATTRIBUTE_DESCRIPTOR} (?P<name>{_ATTRIBUTE_NAME}) (?P<predicate>[a-z].*)"
)
_APPOSITIVE_SUBJECT: Final = re.compile(
    rf"(?P<name>{_ATTRIBUTE_NAME}), {_ATTRIBUTE_DESCRIPTOR}, (?P<predicate>[a-z].*)"
)
_ATTRIBUTE_OBJECT: Final = r"(?:[a-z]+|[A-Z][a-z]+)"
_SINGLE_ATTRIBUTE: Final = re.compile(
    rf"(?:lives?|lived|resides?|resided|works?|worked|studies|studied) "
    rf"(?:in|at|near|for) {_ATTRIBUTE_NAME}"
    rf"|(?:is|was) based in {_ATTRIBUTE_NAME}"
    r"|(?:works?|worked) as (?:a|an) [a-z]+"
    r"|(?:is|was) (?:not )?(?:(?:a|an) )?[a-z]+"
    rf"|(?:has|had|owns?|owned|likes?|liked|loves?|loved|enjoys?|enjoyed|prefers?|preferred) "
    rf"(?:not )?(?:(?:a|an|the) )?{_ATTRIBUTE_OBJECT}"
    rf"|(?:is|was) (?:not )?(?:allergic to|sensitive to|interested in) {_ATTRIBUTE_OBJECT}"
    r"|(?:has|had) type [12] diabetes"
)
"""Complete single-attribute predicates whose subject descriptor can add no assertion."""
_ATTRIBUTE_JOINS: Final = frozenset({
    "and", "or", "but", "nor", "who", "whom", "whose", "which", "that", "where", "when",
    "while", "because", "since", "although", "though", "if", "unless", "named", "called",
})
"""Words that keep a subject-descriptor form outside rule 10."""


@dataclass(frozen=True, slots=True)
class StructureQuoteV1:
    """One of the member's own quotes whose message has role ``user``.

    Attributes:
        text (str): The quote's text.
        speaker (str | None): Its message's labelled speaker; None when unlabelled.
        sentence (str | None): The sentence of its message that holds it
            (``quote_sentence``); None when it cannot be told.
        context (str | None): The text of that sentence's line before it
            (``quote_context``); None when the message is only partly shown.
        order (Sigma | None): The message's explicit source chronology; None when
            unavailable, which cannot establish a question/answer or object antecedent.
    """

    text: str
    speaker: str | None
    sentence: str | None = None
    context: str | None = None
    order: Sigma | None = None


@dataclass(frozen=True, slots=True)
class _Referent:
    """A third party a text mentions (rule 5).

    Attributes:
        key (tuple[str, ...]): The same key is the same person.
        gender (str): ``f``, ``m`` or unknown ``u``.
        plural (bool): Whether it is a group.
        said (tuple[str, ...]): Its mention when a statement may restate it (``my`` and a
            description, or a name), casefolded; empty otherwise.
        cased (tuple[str, ...]): That mention as written.
    """

    key: tuple[str, ...]
    gender: str
    plural: bool
    said: tuple[str, ...] = ()
    cased: tuple[str, ...] = ()


type _Runs = list[tuple[list[str], bool]]
"""Predicate token runs a statement may restate, each with whether only exactly."""


@dataclass(frozen=True, slots=True)
class MemberConstraintsV1:
    """The member-level verdict values one member's own form rules out.

    Attributes:
        atomicity (frozenset[str]): Excluded ``atomicity`` values.
        attribution (frozenset[str]): Excluded ``sensitive_attribution`` values.
    """

    atomicity: frozenset[str] = frozenset()
    attribution: frozenset[str] = frozenset()

    def __bool__(self) -> bool:
        """Whether any value is excluded."""
        return bool(self.atomicity or self.attribution)


NO_CONSTRAINTS: Final = MemberConstraintsV1()


def identified_attribute(statement: str) -> bool:
    """Whether one subject descriptor identifies who has a single attribute (rule 10).

    Args:
        statement: The member's complete statement, without using its source verdict.

    Returns:
        True only for a bounded singular relation and name, directly or as a possessive
        appositive, followed by one full single-attribute predicate. Modified
        descriptors, coordination, clauses and ownership with naming are not settled.
    """
    if not _english(statement):
        return False
    text = unicodedata.normalize("NFC", statement).strip().removesuffix(".")
    if _ATTRIBUTE_JOINS.intersection(_tokens(text)):
        return False
    found = _IDENTIFIED_SUBJECT.fullmatch(text) or _APPOSITIVE_SUBJECT.fullmatch(text)
    if found is None:
        return False
    relation = _PEOPLE.get(found["relation"])
    if relation is None or relation[1] or not _attribute_name(found["name"]):
        return False
    predicate = found["predicate"]
    if _SINGLE_ATTRIBUTE.fullmatch(predicate) is None:
        return False
    return all(word.casefold() not in _NOT_NAMES | _NOT_A_NAME
               for word in predicate.split() if word[0].isupper())


def _attribute_name(name: str) -> bool:
    """A rule-10 name contains no capitalized pronoun, date or predicate word."""
    return all(word.casefold() not in _NOT_NAMES | _NOT_A_NAME for word in name.split())


def member_constraints(
    statement: str, *, correction_requested: bool, targets: Sequence[str],
) -> MemberConstraintsV1:
    """Return the member-level verdict values a member's own form rules out (rule 1).

    Args:
        statement (str): The member's statement.
        correction_requested (bool): The member's ``correction_requested``.
        targets (Sequence[str]): The statements of its current targets (memories and
            siblings).

    Returns:
        MemberConstraintsV1: The excluded values; empty when no rule applies.

    Examples:
        ```python
        member_constraints("The user likes birds, not horses.", correction_requested=True,
                           targets=["The user likes horses."]).atomicity
        # frozenset({'compound'})
        ```
    """
    atomicity = frozenset({"compound"}) if correction_requested and any(
        repairs(statement, target) for target in targets
    ) else frozenset[str]()
    return MemberConstraintsV1(atomicity)


_NAMES_REPLACED: Final = re.compile(r",\s+(?:and\s+|but\s+)?not\s+\S", re.IGNORECASE)
"""A statement that names the value it replaces: "<subject> is <new>, not <old>"."""


def explicit_correction(statement: str, corrects_text: str | None) -> bool:
    """Whether a proposal states an explicit correction, not only a ``revises`` hint.

    The extraction writes ``corrects_text`` for a fact that corrects an earlier statement,
    or the statement sets the corrected value against the old one ("Dwyane Wade's sport
    is baseball, not basketball."). Either form names the value it ends: the lean judge
    then updates the memory its ``revises`` names, and the update's ``supersedes`` link
    records it, so a forget of the correction takes the ended value too.

    Args:
        statement (str): The proposal's statement.
        corrects_text (str | None): Its ``corrects_text``.

    Returns:
        bool: True for ``corrects_text`` or a ``, not <old>`` contrast.

    Examples:
        ```python
        explicit_correction("The user lives in Paris, not London.", None)
        # True
        explicit_correction("The user lives in Paris.", None)
        # False
        ```
    """
    return corrects_text is not None or _NAMES_REPLACED.search(statement) is not None


def settle_relation(relation: str, *, denied: bool, naming_head: bool, names: bool) -> str:
    """Return the relation an accepted verdict stores for one target (rules 3-4).

    Args:
        relation (str): The judged relation.
        denied (bool): Whether the member only denies this target (``denies``).
        naming_head (bool): Whether the target is a current naming memory.
        names (bool): Whether the member has a naming annotation.

    Returns:
        str: ``retract`` for a ``correct`` of a denied target, or for an ``update`` of a
            denied naming head by a member without a naming annotation; otherwise the
            judged relation.
    """
    if denied and (relation == "correct" or (
        relation == "update" and naming_head and not names
    )):
        return "retract"
    return relation


def bounded_episode(statement: str, time_text: str | None, tombstone: str, *,
                    tombstone_kind: str, tombstone_timed: bool) -> bool:
    """Whether a member places a tombstone's present state in a bounded earlier period (rule 6).

    Args:
        statement (str): The member's statement.
        time_text (str | None): The member's own time text (``time.text``).
        tombstone (str): The tombstone's statement.
        tombstone_kind (str): The tombstone's kind.
        tombstone_timed (bool): Whether the tombstone's time has a text or a bound.

    Returns:
        bool: True when the tombstone is an untimed ``state``; the time text names a
            bounded earlier period (``_PERIOD``) sharing no content word with the
            tombstone; and the statement, all English, holds every content word of the
            tombstone with its present predicate in the past, no negation on either side
            and no word that keeps it current or recurring.

    Examples:
        ```python
        bounded_episode("The user lived in a treehouse for a month as an intern.",
                        "for a month as an intern", "The user lives in a treehouse.",
                        tombstone_kind="state", tombstone_timed=False)
        # True
        ```
    """
    if not time_text or tombstone_kind != "state" or tombstone_timed or not (
        _english(statement) and _english(time_text) and _english(tombstone)
    ) or _PERIOD.search(time_text) is None:
        return False
    words, said = _tokens(statement), _tokens(tombstone)
    if _CONTINUING & set(words) or _negations(words) or _negations(said):
        return False
    held = _content(said) - _PREPOSITIONS
    period = _content(_tokens(time_text)) - _PREPOSITIONS
    if not held or held & period or not held <= _content(words):
        return False
    present = {_stem(word) for word in said if not word.endswith("ed")}
    return any(
        bool(set(_PAST_OF.get(word, ())) & set(said)) or word == "used"
        or word.endswith("ed") and _stem(word) in present
        for word in words
    )


def settle_tombstone(verdict: str, *, bounded: bool) -> str:
    """Return the tombstone verdict an accepted verdict stores (rules 6, 8, 9).

    Args:
        verdict (str): The judged tombstone verdict.
        bounded (bool): Whether the member is a bounded earlier episode of the tombstone's
            state (``bounded_episode``) or of its recurring habit (``habit_episode``), or
            only asks the assistant about the tombstone's topic (``topic_request``).

    Returns:
        str: ``other`` for a ``restates`` of a bounded earlier episode; otherwise the
            judged value.
    """
    return "other" if bounded and verdict == "restates" else verdict


def habit_episode(statement: str, time_text: str | None, tombstone: str, *,
                  tombstone_time_text: str | None, tombstone_ended: bool) -> bool:
    """Whether a member places a tombstone's recurring habit in a bounded past period (rule 8).

    Args:
        statement (str): The member's statement.
        time_text (str | None): The member's own time text (``time.text``).
        tombstone (str): The tombstone's statement.
        tombstone_time_text (str | None): The tombstone's own time text.
        tombstone_ended (bool): Whether the tombstone's time has an end bound.

    Returns:
        bool: True when all are English; the member's time text names a bounded past
            period (``_PERIOD`` or ``_PAST_SPAN``); the tombstone states a recurring present
            habit (a ``_RECURRENCE`` schedule in its statement or time text, no past marker,
            no end); the member names no schedule, has no negation and no word that keeps
            it current; and it shares a topic word with the habit.

    Examples:
        ```python
        habit_episode("The user's uncle taught the user to play chess when the user "
                      "was eight.", "when I was eight",
                      "The user plays chess every Friday evening.",
                      tombstone_time_text=None, tombstone_ended=False)
        # True
        ```
    """
    if not time_text or tombstone_ended or not (
        _english(statement) and _english(time_text) and _english(tombstone)
    ):
        return False
    if _PERIOD.search(time_text) is None and _PAST_SPAN.search(time_text) is None:
        return False
    schedule = _RECURRENCE.search(tombstone) or (
        _RECURRENCE.search(tombstone_time_text) if tombstone_time_text else None)
    if schedule is None or _RECURRENCE.search(statement) or _RECURRENCE.search(time_text):
        return False
    words, said = _tokens(statement), _tokens(tombstone)
    if (_negations(words) or _negations(said) or _CONTINUING & set(words)
            or _PAST_MARKERS & set(said)):
        return False
    habit = _topic(tombstone) - _content(_tokens(schedule.group(0)))
    return bool(habit & _topic(statement))


def topic_request(statement: str, tombstone: str) -> bool:
    """Whether a member only asks the assistant about a tombstone's topic (rule 9).

    Args:
        statement (str): The member's statement.
        tombstone (str): The tombstone's statement.

    Returns:
        bool: True when both are English; the statement is one clause opening "The user"
            with a request or preference word (``_REQUESTS``) within its next four words,
            then within eight words an inquiry verb (``_INQUIRIES``) whose topic follows
            and holds no ``that``; it has no digit, number word (``_NUMBER_WORDS``),
            comma, semicolon, colon or second-clause word (``_REASONS``); the tombstone
            has a shape ``_asserted`` reads, whose value holds a topic word and has no
            word (``_kin``) in the statement, and every attribute word the statement
            shares lies in that topic.

    Examples:
        ```python
        topic_request("The user wants to stop being asked how much they weigh.",
                      "The user weighs 61 kilograms.")                        # True
        topic_request("The user wants to stop being asked whether they are a pilot.",
                      "The user is a pilot.")                                 # False
        ```
    """
    if not (_english(statement) and _english(tombstone)) or _CLAUSE_MARKS.search(statement):
        return False
    words = _tokens(statement)
    if words[:2] != ["the", "user"] or (_NUMBER_WORDS | _REASONS) & set(words):
        return False
    rest = words[2:]
    request = next((index for index, word in enumerate(rest[:4]) if word in _REQUESTS), None)
    if request is None:
        return False
    after = rest[request + 1:]
    inquiry = next((index for index, word in enumerate(after[:8]) if word in _INQUIRIES),
                   None)
    if inquiry is None:
        return False
    topic = after[inquiry + 1:]
    if not topic or "that" in topic:
        return False
    asserted = _asserted(tombstone)
    if asserted is None:
        return False
    attribute, value = asserted
    said, asked = _topic(statement), _topic_words(topic)
    if not value or any(_kin(word, other) for word in value for other in said):
        return False
    return all(any(_kin(word, other) for other in asked)
               for word in attribute if any(_kin(word, other) for other in said))


def _asserted(tombstone: str) -> tuple[set[str], set[str]] | None:
    """A tombstone's attribute and asserted-value topic stems (rule 9), or None.

    "The user's <attribute> is <value>"; "The user is <value>", whose last word is the
    attribute when a number comes before it ("2.1 meters tall"); "The user <verb>
    <value>", whose verb is the attribute. A tombstone that opens otherwise, or whose
    verb is an auxiliary or a negation, has no shape here.
    """
    words = [FORMAL_KINSHIP.get(word, word) for word in _tokens(tombstone)]
    if words[:2] != ["the", "user"] or len(words) < 4:
        return None
    rest = words[2:]
    if rest[0] == "s":
        copula = next((index for index, word in enumerate(rest) if word in _COPULAS), None)
        if copula is None:
            return None
        attribute, value = rest[1:copula], rest[copula + 1:]
    elif rest[0] in _COPULAS:
        complement = rest[1:]
        measured = not _numeric(complement[-1]) and any(
            _numeric(word) for word in complement[:-1])
        attribute, value = (complement[-1:], complement[:-1]) if measured else ([], complement)
    elif rest[0] in _FUNCTION or _negated(rest, 0) or (
            rest[0] in _CONTRACTED and rest[1] == "t"):
        return None
    else:
        attribute, value = rest[:1], rest[1:]
    return _topic_words(attribute), _topic_words(value)


def _numeric(word: str) -> bool:
    return word.isdigit() or word in _NUMBER_WORDS


def _kin(word: str, other: str) -> bool:
    """Whether two stems name one word: equal, or sharing their first five letters."""
    return word == other or (len(word) >= 5 and len(other) >= 5 and word[:5] == other[:5])


def unrelated(statement: str, sibling: str) -> bool:
    """Whether two statements of one Add share no topic word (rule 7).

    Args:
        statement (str): The member's statement.
        sibling (str): The sibling's statement.

    Returns:
        bool: True when both are English, each has a topic word (a content word other
            than a preposition, the user referent, a personal pronoun or ``have``) and
            they share none.

    Examples:
        ```python
        unrelated("The user has a cousin.", "The user's birthday is June 9.")       # True
        unrelated("The user was born on June 9.", "The user's birthday is June 9.")  # False
        ```
    """
    if not (_english(statement) and _english(sibling)):
        return False
    mine, theirs = _topic(statement), _topic(sibling)
    return bool(mine) and bool(theirs) and not mine & theirs


def settle_sibling(relation: str, *, unrelated: bool) -> str:
    """Return the relation an accepted verdict stores for one sibling (rule 7).

    Args:
        relation (str): The judged relation.
        unrelated (bool): Whether the member and the sibling share no topic word
            (``unrelated``).

    Returns:
        str: ``independent`` for a ``same`` of an unrelated sibling; otherwise the judged
            relation.
    """
    return "independent" if unrelated and relation == "same" else relation


def settle_attribution(attribution: str | None, *, reported: bool) -> str | None:
    """Return the sensitive attribution an accepted verdict stores (rules 2/5).

    Args:
        attribution (str | None): The judged ``sensitive_attribution``.
        reported (bool): Whether the member is an explicit self-report or third-party
            report (``self_report``, ``explicit_report``).

    Returns:
        str | None: ``direct`` for an ``inferred`` explicit report; otherwise the
            judged value.
    """
    return "direct" if reported and attribution == "inferred" else attribution


_SELF: Final = "@self"
_EVENT: Final = "@event"
_SELF_GRAMMAR: Final = {
    "am": "is", "are": "is", "were": "was", "has": "have", "goes": "go",
    "cause": "because", "helps": "help", "folks": "people",
}
_SELF_INTENSIFIERS: Final = frozenset({"so", "really", "very"})
_SELF_ATTRIBUTES: Final = frozenset({
    "health", "diagnosis", "condition", "allergy", "allergies", "faith", "religion",
    "gender", "identity", "orientation", "affiliation",
})
_SELF_FEELINGS: Final = frozenset({
    "happy", "thankful", "glad", "grateful", "sad", "joyful", "anxious", "relieved",
    "welcome", "accepted", "hopeful", "optimistic", "encouraged", "inspired", "upset", "afraid",
})
_SELF_EVALUATIONS: Final = frozenset({
    "powerful", "inspiring", "encouraging", "moving", "meaningful", "helpful",
    "comforting", "uplifting", "overwhelming", "rewarding", "enjoyable", "positive",
})
_ATTENDANCE: Final = re.compile(
    r"^@self (?:went to|attended|visited) (?P<event>.+?)"
    r"(?P<time> yesterday| today| last [a-z]+| on [a-z]+)?$")


def self_report(
    statement: str, *, primary: str, user_quotes: Sequence[StructureQuoteV1],
    user_messages: Sequence[StructureQuoteV1] = (),
) -> bool:
    """Whether a statement restates its speaker's explicit own report (rule 2).

    This reads only fully shown cited sentences and their preceding line context.
    Metadata and literal line labels must agree. The speaker must be the member's
    self or participant primary, named in its statement. Word order and predicate
    roles stay intact: a protected word in an event, story or another person never
    becomes the speaker's identity. Reporting and uncertainty words are retained.

    One explicit attendance event in the speaker's own cited turns can supply its
    definite noun phrase or ``it``. A literal evaluation ``it was moving`` can be
    restated as the speaker finding that event moving; a coordinated event predicate
    keeps its subject and auxiliary. These are grammatical restatements, never a
    sensitive-class override.

    Args:
        statement: The member's statement.
        primary: The binding of its primary anchor.
        user_quotes: Its user-message quotes, with their sentences and line contexts.
        user_messages: Shown user messages, used only to exclude intervening referents.

    Returns:
        True when a readable cited line directly states the speaker's own fact.
    """
    if primary not in ("self", "participant") or not _english(statement):
        return False
    readings: list[tuple[StructureQuoteV1, str | None, str]] = []
    for quote in user_quotes:
        for reporter, text in _self_lines(quote):
            if (primary == "self") != (reporter is None):
                continue
            name = _tokens(reporter) if reporter is not None else ["the", "user"]
            stated = _tokens(statement)
            if not any(stated[start:start + len(name)] == name
                       for start in range(len(stated) - len(name) + 1)):
                continue
            readings.append((quote, reporter, text))
    for quote, reporter, text in readings:
        claimed = _self_words(statement, reporter=reporter, source=False)
        attendance = {
            (event["event"], event["time"]): event
            for _, speaker, line in readings if speaker == reporter
            for part in re.split(r"[.!]", line)
            if (event := _ATTENDANCE.fullmatch(" ".join(_self_words(
                part, reporter=reporter, source=True)).split(" and ")[0])) is not None
        }
        event = next(iter(attendance.values())) if len(attendance) == 1 else None
        if event is not None and (set(event["event"].split()) & _PREPOSITIONS
                                  or _SELF in event["event"].split()):
            event = None
        if event is not None:
            claimed = _event_words(claimed, event, source=False)
        for part in (part.strip() for part in re.split(r"[.!]", text) if part.strip()):
            # Earlier context interprets an event or speaker; it is never a
            # substitute for the sentence holding the member's own quote.
            if quote.sentence is None or part not in quote.sentence:
                continue
            words = _self_words(part, reporter=reporter, source=True)
            if _own_object_action(claimed, words, reporter=reporter, readings=readings,
                                  action=quote, action_part=part, user_messages=user_messages):
                return True
            canonical = _event_words(words, event, source=True) if event else words
            for form in _self_forms(canonical, event is not None):
                if claimed == form or event is not None and (
                    _without_event_location(claimed) == form
                ):
                    return True
                if event is not None and claimed in _perception_forms(
                    form, reporter=reporter, answer=quote, user_quotes=user_quotes,
                ):
                    return True
    return False


def _self_lines(quote: StructureQuoteV1) -> Iterator[tuple[str | None, str]]:
    """Readable cited lines whose metadata and literal speaker label agree."""
    if quote.sentence is None or quote.context is None:
        return
    for line in (quote.context + quote.sentence).splitlines():
        reporter, text = quote.speaker, line.strip()
        label = _LABEL.match(text)
        if label is not None:
            name = label["label"]
            if name.casefold() not in _DISCOURSE_LABELS:
                if not all(word[:1].isupper() for word in name.split()) or (
                    reporter is not None and _tokens(reporter) != _tokens(name)
                ):
                    continue
                reporter = name
            text = text[label.end():]
        if _english(text) and not (WITHHELD_MARKER in text or _QUOTATION.search(text)
                                   or "?" in text):
            yield reporter, text


def _self_words(text: str, *, reporter: str | None, source: bool) -> list[str]:
    """Normalize only person/verb agreement, articles and degree intensifiers."""
    words = _uncontracted(_tokens(text))
    intensifiers = _SELF_INTENSIFIERS if not _negations(words) else frozenset[str]()
    name = _tokens(reporter) if reporter is not None else ["the", "user"]
    out: list[str] = []
    index = 0
    introduced = False
    while index < len(words):
        word = words[index]
        if not source and words[index:index + len(name)] == name:
            out.append(_SELF)
            introduced = True
            index += len(name)
            continue
        if source and word in ("i", "me", "my", "myself"):
            out.append(_SELF)
            if word in ("my", "myself"):
                out.append("s" if word == "my" else "self")
            if word == "i" and words[index + 1:index + 2] in (["m"], ["ve"]):
                out.append("is" if words[index + 1] == "m" else "have")
                index += 1
        elif not source and introduced and word in ("he", "she", "they", "him", "her", "them"):
            out.append(_SELF)
        elif not source and introduced and word in ("himself", "herself", "themselves"):
            out.extend((_SELF, "self"))
        elif word not in ({"the", "a", "an"} | intensifiers):
            out.append(_SELF_GRAMMAR.get(word, word))
        index += 1
    return out


def _event_words(words: list[str], event: re.Match[str], *, source: bool) -> list[str]:
    """Read a definite event noun phrase and the speaker's attendance relative clause."""
    name, time = event["event"].split(), (event["time"] or "").split()
    aliases = [name[index:] for index in range(max(1, len(name) - 1))]
    out = list(words)
    for alias in aliases:
        start = 0
        while start <= len(out) - len(alias):
            if out[start:start + len(alias)] != alias:
                start += 1
                continue
            end = start + len(alias)
            if not source and out[end:end + 2] == [_SELF, "attended"]:
                end += 2
                if time and out[end:end + len(time)] == time:
                    end += len(time)
            out[start:end] = [_EVENT]
            start += 1
    if source:
        out = [_EVENT if word == "it" else word for word in out]
    return out


def _self_forms(words: list[str], event: bool) -> Iterator[list[str]]:
    """Own predicates, event evaluations and predicates coordinated under an auxiliary."""
    own_subject = words[:1] == [_SELF] and words[1:2] != ["s"]
    own_feeling = (words[1:2] in (["feel"], ["felt"]) or words[1:2] in (["is"], ["was"])
                   and bool(_SELF_FEELINGS & set(words[2:3])))
    own_speech = words[1:2] in (["told"], ["said"], ["talked"], ["spoke"]) and any(
        words[index:index + 3] == ["about", _SELF, "s"] for index in range(len(words) - 2))
    own = own_subject and (own_feeling or own_speech or not (
        (_HEDGES | _REPORT_DOUBT) & set(words) and _PEOPLE.keys() & set(words)))
    reported_self = any(
        word == "said" and words[:2] == [_SELF, "s"]
        and 3 <= index <= 5 and bool(set(words[2:index]) & _PEOPLE.keys())
        and not ((_HEDGES | _REPORT_DOUBT | _SUBORDINATORS | {"and", "is", "was", "have"})
                 & set(words[2:index]))
        and (words[index + 1:index + 2] == [_SELF]
             or words[index + 1:index + 3] == ["that", _SELF])
        for index, word in enumerate(words)
    )
    felt_effect = any(words[index:index + 3] == ["made", _SELF, "feel"]
                      for index in range(len(words) - 2))
    given_effect = event and words[:3] == [_EVENT, "have", "given"] and words[3:4] == [_SELF]
    subject_end = next((index for index, word in enumerate(words) if word in (
        "is", "was", "have", "be", "might", "may", "could", "will", "would",
    )), 0)
    subject = words[2:subject_end]
    possessive_attribute = (words[:2] == [_SELF, "s"] and bool(subject)
                            and subject[-1] in _SELF_ATTRIBUTES
                            and "s" not in subject
                            and not set(subject) & _PEOPLE.keys())
    if own or possessive_attribute or reported_self or felt_effect or given_effect:
        yield words
    if not event or (_HEDGES | _REPORT_DOUBT | _SUBORDINATORS) & set(words):
        return
    # The source evaluates its event or content; the restatement retains that object.
    if "was" in words and words[0] != _SELF:
        split = words.index("was")
        if (split and words[split + 1:] and len(words[split + 1:]) == 1
                and words[-1] in _SELF_EVALUATIONS and not set(words[:split]) & _PEOPLE.keys()):
            yield [_SELF, "found", *words[:split], *words[split + 1:]]
            if words[:split] == [_EVENT]:
                yield [*words, "for", _SELF]
    if _negations(words):
        return
    for split, word in enumerate(words):
        if word != "and":
            continue
        left, right = words[:split], words[split + 1:]
        if not right:
            continue
        attended = _ATTENDANCE.fullmatch(" ".join(left)) is not None
        if attended:
            yield left
        if attended and right[0] == _EVENT:
            yield from _self_forms(right, event)
        elif len(left) > 1 and left[0] == _EVENT and left[1] == "have":
            yield left
            yield from _self_forms([_EVENT, "have", *right], event)


def _without_event_location(words: list[str]) -> list[str]:
    """Read an event location only on its content evaluation or the speaker's feeling."""
    if not (words[:2] == [_SELF, "found"] or words[:2] == [_SELF, "was"]
            and _SELF_FEELINGS & set(words[2:])):
        return words
    return [word for index, word in enumerate(words)
            if not (word == "at" and words[index + 1:index + 2] == [_EVENT]
                    or word == _EVENT and index > 0 and words[index - 1] == "at")]


def _perception_forms(
    form: list[str], *, reporter: str | None, answer: StructureQuoteV1,
    user_quotes: Sequence[StructureQuoteV1],
) -> Iterator[list[str]]:
    """An explicitly asked perception frame whose own answer supplies content and evaluation."""
    if (answer.order is None or form[:2] != [_SELF, "found"] or len(form) < 4
            or form[-1] not in _SELF_EVALUATIONS):
        return
    content, evaluation = form[2:-1], form[-1]
    for quote in user_quotes:
        if quote.sentence is None or quote.context is None:
            continue
        if (quote.order is None or quote.order[1] != answer.order[1]
                or quote.order[0] > answer.order[0] or quote.order[2] + 1 != answer.order[2]):
            continue
        question, speaker = quote.sentence.strip(), quote.speaker
        label = _LABEL.match(quote.context or question)
        if label is not None:
            name = label["label"]
            if name.casefold() in _DISCOURSE_LABELS or not all(
                word[:1].isupper() for word in name.split()
            ) or speaker is not None and _tokens(speaker) != _tokens(name):
                continue
            speaker = name
            if not quote.context:
                question = question[label.end():]
        if (speaker == reporter or not question.endswith("?") or _QUOTATION.search(question)
                or WITHHELD_MARKER in question):
            continue
        asked = _tokens(question)
        if len(asked) == 6 and asked[:2] == ["did", "you"] and asked[3] in (
            "any", "some",
        ) and asked[4:] == [evaluation, content[-1]]:
            past = {"hear": "heard", "see": "saw", "read": "read"}.get(asked[2])
            if past is not None:
                yield [_SELF, "found", *content, _SELF, past, "at", _EVENT, evaluation]


def _own_object_action(
    claimed: list[str], said: list[str], *, reporter: str | None,
    readings: Sequence[tuple[StructureQuoteV1, str | None, str]],
    action: StructureQuoteV1, action_part: str, user_messages: Sequence[StructureQuoteV1],
) -> bool:
    """Restate an own action on the one object the speaker explicitly introduced.

    A cited ``here's one of the charities I'm considering`` supplies the noun for
    ``I chose them because they help ...``. The same action and full causal predicate
    must be retained; the source's object pronouns never supply the speaker's trait.
    """
    if (action.order is None or len(said) < 3 or said[0] != _SELF
            or said[2] not in ("it", "them")
            or said[1] in _HEDGES | _REPORT_DOUBT | _SUBORDINATORS
            or claimed[:2] != said[:2]):
        return False
    names: dict[tuple[str, ...], tuple[StructureQuoteV1, str]] = {}
    for quote, speaker, line in readings:
        if (speaker != reporter or quote.sentence is None or quote.order is None
                or quote.order[1] != action.order[1] or quote.order[0] > action.order[0]
                or quote.order[2] > action.order[2]):
            continue
        for part in (part.strip() for part in re.split(r"[.!]", line) if part.strip()):
            if part not in quote.sentence:
                continue
            if quote.order == action.order and (quote.context or "") + quote.sentence == (
                (action.context or "") + (action.sentence or "")
            ) and quote.sentence.find(part) >= quote.sentence.find(action_part):
                continue
            words = _self_words(part, reporter=reporter, source=True)
            if words[:1] == ["and"]:
                words = words[1:]
            if words[:2] not in (["here", "s"], ["here", "is"], ["this", "is"]):
                continue
            words = words[2:]
            if words[:2] == ["one", "of"]:
                words = words[2:]
            end = next((index for index, word in enumerate(words) if word == _SELF), 0)
            name = words[:end]
            if (1 <= len(name) <= 4 and words[end + 1:end + 2] in (["is"], ["was"])
                    and not set(name) & (_PEOPLE.keys() | _HEDGES | _SUBORDINATORS)):
                names[tuple(_singular_noun(word) for word in name)] = (quote, part)
    if len(names) != 1:
        return False
    name = next(iter(names))
    introduction, introduced = names[name]
    if not _object_interval(introduction, introduced, action, action_part, name=name,
                            reporter=reporter, user_messages=user_messages):
        return False
    if tuple(_singular_noun(word) for word in claimed[2:2 + len(name)]) != name:
        return False
    restated = [*claimed[:2], "@object", *claimed[2 + len(name):]]
    restated = ["@object" if word == "it" else word for word in restated]
    stated = ["@object" if word in ("it", "them", "they") else word for word in said]
    return restated == stated or (
        stated[3:5] == ["because", "@object"] and restated[3:4] in (["that"], ["which"])
        and restated == [*stated[:3], restated[3], *stated[5:]])


def _object_interval(
    introduction: StructureQuoteV1, introduced: str, action: StructureQuoteV1, action_part: str,
    *, name: tuple[str, ...], reporter: str | None,
    user_messages: Sequence[StructureQuoteV1],
) -> bool:
    """Require an earlier referent with no competing object in the shown interval."""
    if introduction.order is None or action.order is None or not user_messages or any(
        message.order is None for message in user_messages
    ) or introduction.order[1] != action.order[1]:
        return False
    found: set[Sigma] = set()
    for message in user_messages:
        order = message.order
        if (order is None or order[1] != introduction.order[1]
                or not introduction.order[2] <= order[2] <= action.order[2]):
            continue
        if message.sentence is None or WITHHELD_MARKER in message.text:
            return False
        found.add(order)
        text = message.text
        start = text.find(introduced) if order == introduction.order else 0
        end = text.find(action_part) if order == action.order else len(text)
        if start < 0:
            return False
        if order == introduction.order:
            start += len(introduced)
        if end < start:
            return False
        offset = 0
        for line in text.splitlines(keepends=True):
            line_end = offset + len(line)
            speaker, label = message.speaker, _LABEL.match(line)
            if label is not None and label["label"].casefold() not in _DISCOURSE_LABELS:
                if speaker is not None and _tokens(speaker) != _tokens(label["label"]):
                    return False
                speaker = label["label"]
            # A transcript label governs its own line, not an appended unlabelled
            # attachment annotation. Such text introduces no conversational referent.
            if speaker is not None or reporter is None:
                left = max(start, offset + (0 if label is None else label.end()))
                interval = text[left:min(end, line_end)] if left < end else ""
                if not _object_line(interval, speaker=speaker, reporter=reporter, name=name):
                    return False
            offset = line_end
    shown = {order[2] for order in found}
    ordered = sorted(found, key=lambda order: order[2])
    return ({introduction.order, action.order} <= found
            and len(shown) == action.order[2] - introduction.order[2] + 1
            and all(left[0] <= right[0] for left, right in zip(ordered, ordered[1:], strict=False)))


def _object_line(
    text: str, *, speaker: str | None, reporter: str | None, name: tuple[str, ...],
) -> bool:
    """Whether an attributed intervening line introduces no competing object."""
    if _tokens(speaker or "") == _tokens(reporter or ""):
        for clause in re.split(r"[.!?;]|\bbut\b", text):
            words = _self_words(clause, reporter=reporter, source=True)
            if words[:1] == [_SELF] and not set(words[1:]) <= (
                _SELF_FEELINGS | {"is", "was", "feel", "felt", "feeling", "and"}
            ):
                return False
    pieces, _ = _pieces(text)
    mentions: list[list[str]] = []
    for index, word in enumerate(pieces):
        if word not in {"a", "an", "the", "this", "that", "another", "my", "your"}:
            continue
        noun: list[str] = []
        for following in pieces[index + 1:index + 5]:
            if following in _WINDOW_BREAKS | _HEDGES | _REPORT_DOUBT:
                break
            noun.append(following)
        if noun and noun[-1] not in {"lot", "bit", "amount", "number", "deal", "kind", "sort"}:
            mentions.append(noun)
    return not mentions or _singular_noun(mentions[-1][-1]) == name[-1]


def _singular_noun(word: str) -> str:
    """The regular singular of a noun in an explicit object description."""
    if word.endswith("ies"):
        return word[:-3] + "y"
    return word[:-1] if word.endswith("s") and not word.endswith("ss") else word


def explicit_report(
    statement: str, *, sensitivity_classes: Sequence[str],
    user_quotes: Sequence[StructureQuoteV1],
) -> bool:
    """Whether a member is a speaker's explicit report of a third party's trait (rule 5).

    The statement itself names the person, as the reporter's described relation or by
    name, so the anchor binding the extractor chose does not decide (live, the
    brother was anchored to the user in 1 of 6 runs). A self-report never qualifies:
    its sentence opens with ``I``, which is no description and no name.

    Args:
        statement (str): The member's statement.
        sensitivity_classes (Sequence[str]): Its ``sensitivity_classes``.
        user_quotes (Sequence[StructureQuoteV1]): Its quotes whose message has role
            ``user``, with their speakers, sentences and line contexts.

    Returns:
        bool: True when the member is sensitive (never ``credentials``) and the
            sentence of one of its quotes is restated by ``reports_about``.

    Examples:
        ```python
        explicit_report(
            "The user's brother is allergic to peanuts.", sensitivity_classes=["health"],
            user_quotes=[StructureQuoteV1(
                "My brother is allergic to peanuts.", None,
                "My brother is allergic to peanuts.", "")])
        # True
        explicit_report(
            "The user's mom is a devout Hindu.", sensitivity_classes=["religion"],
            user_quotes=[StructureQuoteV1(
                "She is a devout Hindu", None, " She is a devout Hindu.",
                "I visited my mom.")])
        # True
        ```
    """
    if not sensitivity_classes or _UNREPORTED_CLASSES & set(sensitivity_classes):
        return False
    return any(
        quote.sentence is not None
        and reports_about(statement, quote.sentence, reporter=quote.speaker,
                          context=quote.context)
        for quote in user_quotes
    )


def quote_sentence(message: str, quote: str, *, excerpt: bool) -> str | None:
    """Return the sentence of a message that holds a quote (rule 5).

    Args:
        message (str): The quote's message text as the request shows it.
        quote (str): The quote's text.
        excerpt (bool): Whether the message is only partly shown.

    Returns:
        str | None: The text from the sentence end before the quote to the one after
            it; None when the quote is not in the text, the sentence touches the edge
            of an excerpt or holds withheld text.
    """
    start = message.find(quote) if quote.strip() else -1
    if start < 0:
        return None
    left = max((found.end() for found in _SENTENCE_END.finditer(message, 0, start)),
               default=None)
    right = _SENTENCE_END.search(message, start + len(quote))
    if excerpt and (left is None or right is None):
        return None
    sentence = message[left or 0:len(message) if right is None else right.end()]
    return None if WITHHELD_MARKER in sentence else sentence


def quote_context(message: str, quote: str, *, excerpt: bool) -> str | None:
    """Return the text of a quote's line before its sentence (rule 5).

    A later sentence's pronoun is read against it, and a line label opening it names
    the reporter of every sentence in the line.

    Args:
        message (str): The quote's message text as the request shows it.
        quote (str): The quote's text.
        excerpt (bool): Whether the message is only partly shown.

    Returns:
        str | None: The text from the start of the line holding the quote's sentence
            (``quote_sentence``) to that sentence; empty when the sentence opens its line;
            None when the quote is not in the text or the message is only partly shown.

    Examples:
        ```python
        quote_context("Blair: I visited my mom. She is a devout Hindu.",
                      "She is a devout Hindu", excerpt=False)
        # 'Blair: I visited my mom.'
        ```
    """
    start = message.find(quote) if quote.strip() else -1
    if excerpt or start < 0:
        return None
    left = max((found.end() for found in _SENTENCE_END.finditer(message, 0, start)),
               default=0)
    return message[message.rfind("\n", 0, left) + 1:left]


def denies(statement: str, target: str) -> bool:
    """Whether a statement only denies a target statement's own assertion (rule 3).

    Args:
        statement (str): The member's statement.
        target (str): A target's statement.

    Returns:
        bool: True when both are English, the statement has no contrast tail and one
            negation, the target's main clause has none, and the statement's content
            words (at least two) all occur in that main clause.
    """
    if not (_english(statement) and _english(target)) or _main(statement) != statement.strip():
        return False
    words, claim = _tokens(statement), _tokens(_main(target))
    if _negations(words) != 1 or _negations(claim) != 0:
        return False
    content = _content(words)
    return len(content) >= 2 and content <= _content(claim)


def repairs(statement: str, target: str) -> bool:
    """Whether a statement is the paired repair ``X, not Y`` of a target statement (rule 1).

    Args:
        statement (str): The member's statement.
        target (str): A target's statement.

    Returns:
        bool: True when ``Y`` holds exactly the target's words the core ``X`` replaces,
            the core shares at least two content words with the target and adds no
            more words than it replaces, and neither part is negated or conjoined.
    """
    if not (_english(statement) and _english(target)):
        return False
    match = _CONTRAST.search(statement.strip())
    if match is None:
        return False
    core = statement.strip()[:match.start()]
    words, claim = _tokens(core), _tokens(_main(target))
    if _CONJUNCTIONS & set(words) or ";" in core or _negations(words) or _negations(claim):
        return False
    stated, replaced, held = _content(words), _content(_tokens(match["replaced"])), _content(claim)
    added, removed = stated - held, held - stated
    return (bool(added) and bool(removed) and removed <= replaced <= held
            and len(stated & held) >= 2 and len(added) <= len(removed)
            and not added & _ADDED_PREDICATES)


def one_replacement(statement: str, target: str) -> bool:
    """Whether a statement is one new value in place of the value a target states (rule 11).

    Args:
        statement (str): The member's statement.
        target (str): A current target's statement.

    Returns:
        bool: True when both are English and the statement is ``X instead of Y`` (``rather
            than``, ``, not``, or ``Instead of Y, X``) where:

            - ``Y`` names only what the unnegated target states (each of its topic words is
              one of the target's) and at least one word besides a generic verb
              (``_GENERIC_VERBS``: "instead of playing" names nothing);
            - an opening or closing time phrase, if any, is one of ``_TIME_PHRASES``;
            - ``X`` is the target's subject with one plain predicate: no comma, semicolon,
              negation or joining word (``_REPLACEMENT_JOINS``: a clause, person, cause,
              event or naming fact), no possessive, pronoun or person the target lacks, a
              ``to`` only towards a place, a capitalized word only for a place
              (``near West Lake``), a day or month, or in place of a capitalized ``Y``
              ("Hangzhou, not Shanghai"), and at most ``_REPLACEMENT_WORDS`` topic words the
              target lacks, one at least.

    Examples:
        ```python
        one_replacement("Since the move, the user goes climbing at a gym near West Lake "
                        "instead of playing badminton.",
                        "The user plays badminton every Saturday morning at a club in Xuhui.")
        # True
        one_replacement("After moving to Hangzhou for a job, the user goes climbing instead "
                        "of playing badminton.",
                        "The user plays badminton every Saturday morning at a club in Xuhui.")
        # False
        ```
    """
    if not (_english(statement) and _english(target)):
        return False
    parsed = _replacement(statement)
    if parsed is None:
        return False
    core, replaced = parsed
    words, said, claim = _tokens(core), _tokens(replaced), _tokens(_main(target))
    if not words or not said or _negations(words) or _negations(said) or _negations(claim):
        return False
    held, named = _topic_words(claim), _topic_words(said)
    added = _topic_words(words) - held
    if not (named <= held and _specific(said) and added) or len(added) > _REPLACEMENT_WORDS:
        return False
    return _plain_value(core, words, claim, target=_main(target), replaced=replaced)


def ended_parts(statement: str) -> list[str]:
    """The parts of an English statement that name a value as replaced, ended or denied
    (rule 11; the forget reads them, ``store.forget_admission.names_replaced``).

    Args:
        statement (str): Any statement.

    Returns:
        list[str]: The text after each ``instead of``, ``rather than`` or ``, not`` and an
            opening ``Instead of Y,``, up to the next comma or semicolon; what a move or
            switch left (``moved from X to``, ``switched from X``, ``changed from X``); what an
            ending verb took (``sold X``, ``gave away X``, ``quit X``, ``left X``,
            ``replaced X``); and each clause (between commas, semicolons and sentence ends)
            with an ending word (``no longer``, ``anymore``, ``stopped``, ``formerly``,
            ``used to``). A bare negation ("never", "does not") names no replaced value.
            The markers are English; a value after one may be in any script ("instead of
            打羽毛球").
    """
    text = unicodedata.normalize("NFKC", statement)
    parts = [re.split(r"[,;]", text[found.end():], maxsplit=1)[0]
             for found in _REPLACED_MARKER.finditer(text)]
    leading = _LEADING_CONTRAST.match(text.strip())
    if leading is not None:
        parts.append(leading["replaced"])
    parts += [found["old"] for pattern in (_LEFT_FROM, _ENDED_OBJECT)
              for found in pattern.finditer(text)]
    for clause in re.split(r"[,;]|[.!?](?:\s|$)", text):
        # A bare negation ("has never played badminton") states the opposite and presupposes
        # no old value; only an ending word does.
        if _ENDING_WORDS & set(_tokens(clause)) or "used to" in clause.casefold():
            parts.append(clause)
    return parts


def replaced_ranges(clause: str) -> list[tuple[int, int]]:
    """Where a clause names a value as replaced, ended or left, as ``ended_parts`` reads it:
    character ranges into ``clause`` itself, each without surrounding blanks.

    A part after ``instead of``, ``rather than`` or ``, not`` runs from its marker to the next
    comma or semicolon ("instead of playing badminton"); an opening ``Instead of Y,`` covers
    its marker and Y; a move or switch covers what it left ("from Shanghai" of "moved from
    Shanghai to Hangzhou"); an ending verb covers itself and its object ("sold the blue
    Toyota"); a clause with an ending word is named whole, so it comes back as
    one range. The forget withholds these ranges of a quote naming a forgotten value
    (``store.forget_admission.naming_parts``), not the clause's other words.

    Args:
        clause (str): One clause of a quote, with no clause mark inside.

    Returns:
        list[tuple[int, int]]: The ranges in order, overlapping ones merged; empty when the
            clause names no value so.
    """
    if _ENDING_WORDS & set(_tokens(clause)) or "used to" in clause.casefold():
        found = [(0, len(clause))]
    else:
        found = [(match.start(), match.start() + len(re.split(
            r"[,;]", clause[match.start():], maxsplit=1)[0]))
            for match in _REPLACED_MARKER.finditer(clause)]
        leading = _LEADING_CONTRAST.match(clause)
        if leading is not None:
            found.append((0, leading.end("replaced")))
        found += [match.span("part") for match in _LEFT_FROM.finditer(clause)]
        found += [match.span() for match in _ENDED_OBJECT.finditer(clause)]
    merged: list[tuple[int, int]] = []
    for start, end in sorted(found):
        while start < end and clause[start] in " ,;\t":
            start += 1
        while end > start and clause[end - 1].isspace():
            end -= 1
        if start >= end:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def content_words(text: str) -> set[str]:
    """The topic stems of an English text that say something: ``distinctive_words`` with its
    category words kept ("job" of "quit the user's job"), so a part naming another thing of
    a category the target does not name is read as naming it."""
    return _topic_words([word for word in _tokens(text) if word not in _GENERIC_VERBS
                         and word not in _ENDING_WORDS])


def distinctive_words(text: str) -> set[str]:
    """The topic stems of an English text that name a value of its own: no generic verb
    (``_GENERIC_VERBS``), category or time word (``_CATEGORY_WORDS``: food, car, sport, club,
    morning) or ending word (rule 11)."""
    return _topic_words([word for word in _tokens(text) if word not in _GENERIC_VERBS
                         and word not in _ENDING_WORDS and word not in _CATEGORY_WORDS])


def _replacement(statement: str) -> tuple[str, str] | None:
    """A statement's new value and its one replaced value, without the time phrases of
    ``_TIME_PHRASES`` that open or close it (rule 11); None for any other form."""
    text = unicodedata.normalize("NFKC", statement).strip().removesuffix(".").strip()
    markers = list(_REPLACED_MARKER.finditer(text))
    leading = _LEADING_CONTRAST.match(text)
    if leading is not None and not markers:
        core, replaced = leading["core"], leading["replaced"]
    elif leading is None and len(markers) == 1:
        core, replaced = text[:markers[0].start()], text[markers[0].end():]
        for phrase in _TIME_PHRASES:
            closing = re.search(r"(?:\s*,\s*|\s+)" + phrase + r"$", replaced, re.IGNORECASE)
            if closing is not None:
                replaced = replaced[:closing.start()]
                break
    else:
        return None
    for phrase in _TIME_PHRASES:
        opening = re.match(r"\s*" + phrase + r"(?:\s*,\s*|\s+)", core, re.IGNORECASE)
        if opening is not None:
            core = core[opening.end():]
            break
    if set(",;") & set(core + replaced):
        return None
    return core, replaced


def _specific(words: Sequence[str]) -> set[str]:
    """The topic stems of a value's words besides generic verbs (rule 11)."""
    return _topic_words([word for word in words if word not in _GENERIC_VERBS])


def _plain_value(core: str, words: Sequence[str], claim: Sequence[str], *, target: str,
                 replaced: str) -> bool:
    """Whether a replacement's new value is the target's subject with one plain predicate
    of the replaced value's kind (``one_replacement``)."""
    subject = 2 if words[0] in _DETERMINERS | {"my", "his", "her", "their", "our", "your"} else 1
    if list(words[:subject]) != list(claim[:subject]) or _REPLACEMENT_JOINS & set(words):
        return False
    added = set(words) - set(claim)
    if sum(word in _POSSESSIVES for word in words) > sum(word in _POSSESSIVES for word in claim):
        return False
    if any(word in _THIRD_PARTIES or word in _PEOPLE for word in added):
        return False
    if _STATE_VERBS & set(words) and not _STATE_VERBS & set(claim):
        return False  # a condition in place of an activity: "is pregnant instead of ..."
    if classify([core]):
        return False
    cased = _WORD.findall(unicodedata.normalize("NFKC", core))
    if len(cased) != len(words):
        return False
    for index, word in enumerate(words):
        if word == "to" and not (index + 1 < len(words) and (
                words[index + 1] in _DESTINATIONS or cased[index + 1][:1].isupper())):
            return False
        if not (word in claim or word in _VALUE_WORDS or word in _FUNCTION
                or word in _PREPOSITIONS or word in _UNTOPICAL or word in _CALENDAR
                or word in _NUMBER_WORDS or word.isdigit() or cased[index][:1].isupper()):
            return False  # a word of no kind a replaced value is (``_VALUE_WORDS``)
    return _names_in_place(cased, words, claim, _WORD.findall(
        unicodedata.normalize("NFKC", target)), _WORD.findall(replaced))


def _capital_runs(cased: Sequence[str]) -> list[tuple[int, int]]:
    """The (start, end) runs of capitalized words, the first word and ``I`` left out."""
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, word in enumerate([*cased, ""]):
        capital = index > 0 and word[:1].isupper() and word != "I"
        if capital and start is None:
            start = index
        elif not capital and start is not None:
            runs.append((start, index))
            start = None
    return runs


def _names_in_place(cased: Sequence[str], words: Sequence[str], claim: Sequence[str],
                    target: Sequence[str], replaced: Sequence[str]) -> bool:
    """Whether every capitalized name of a replacement's new value is the target's, a day or
    month, a two-word natural place name ending in ``_NATURE_HEADS`` ("West Lake"), or one
    word in place of one of the replaced value's, after the same word the replaced name
    follows in the target (rule 11): "Hangzhou, not Shanghai" against "The user lives in
    Shanghai.", never "near Lily", "Hangzhou Prison, not Shanghai", "Rehab Park" or "Tom
    Park": a longer name, or one ending in a word a person's or an institution's name may
    end in, names more than a place."""
    slots: list[str] = []
    for start, end in _capital_runs([""] + list(replaced)):
        name = [word.casefold() for word in replaced[start - 1:end - 1]]
        folded = [word.casefold() for word in target]
        for at in range(1, len(folded) - len(name) + 1):
            if folded[at:at + len(name)] == name:
                slots.append(folded[at - 1])
                break
    for start, end in _capital_runs(cased):
        run = words[start:end]
        if all(word in claim for word in run) or (len(run) == 1 and run[0] in _CALENDAR):
            continue
        if run[-1] in _NATURE_HEADS and len(run) == 2:
            continue  # "West Lake"; "Fudan Hospital Tower" names more than a place
        before = words[start - 1]
        if len(run) != 1 or before not in slots:
            return False
        slots.remove(before)
    return True


def reports_about(
    statement: str, sentence: str, *, reporter: str | None, context: str | None = None,
) -> bool:
    """Whether a statement restates a sentence's explicit statement about a third party (rule 5).

    The sentence is read whole, so support quoted in fragments (``My mom`` and ``is a
    devout Hindu``) is read as the sentence that holds them. A leading line label is
    read as written and without it: a discourse label (``Correction:``) keeps the
    reporter; on an unlabelled message, a capitalized name (``Blair:``, a common transcript shape)
    is the reporter, also of the line's later sentences. With the sentence's line context
    (``quote_context``), its later clauses, relative clauses and appositives, and a
    pronoun opening it or a later clause, are read too (``_restates_clause``).

    Args:
        statement (str): The member's statement.
        sentence (str): A sentence of a user message that holds one of its quotes.
        reporter (str | None): That message's labelled speaker; None for the user.
        context (str | None): The text of the sentence's line before it; None when the
            message is only partly shown, which reads the opening subject only.

    Returns:
        bool: True when both are English; the sentence has no hedge, relayed, doubted
            or alternative wording (``_REPORT_DOUBT``), quotation mark or question mark;
            and the statement is a subject phrase followed by a contiguous run of the
            sentence (so a negation inside the predicate must be restated) that directly
            follows the sentence's opening subject phrase (after ``and``, ``but``,
            ``also`` and the like), the two subject phrases corresponding
            (``_corresponds``), or restates one of the forms.
    """
    if not (_english(statement) and _english(sentence)):
        return False
    if _QUOTATION.search(sentence) or sentence.rstrip().endswith("?"):
        return False
    if (_HEDGES | _REPORT_DOUBT) & set(_uncontracted(_tokens(sentence))):
        return False
    line_reporter = _line_reading(sentence, context, reporter)[2] if context else reporter
    readings = [(sentence, line_reporter)]
    label = _LABEL.match(sentence)
    if label is not None:
        rest, name = sentence[label.end():], label["label"]
        if name.casefold() in _DISCOURSE_LABELS:
            readings.append((rest, line_reporter))
        elif line_reporter is None and all(word[:1].isupper() for word in name.split()):
            readings.append((rest, name))
    return (any(_restates(statement, text, speaker) for text, speaker in readings)
            or _restates_clause(statement, sentence, context, reporter))


def _restates(statement: str, sentence: str, reporter: str | None) -> bool:
    """``reports_about`` for one reading of the sentence."""
    words, said = _uncontracted(_tokens(statement)), _uncontracted(_tokens(sentence))
    if _HEDGES & set(said):
        return False
    lead = _lead(said)
    said = said[lead:]
    cased = _WORD.findall(unicodedata.normalize("NFKC", sentence))[lead:]
    possessive = ["the", "user", "s"] if reporter is None else [*_tokens(reporter), "s"]
    for split in range(1, len(words)):
        subject, rest = words[:split], words[split:]
        for start in range(1, min(len(said) - len(rest), _DESCRIPTION_TOKENS + 1) + 1):
            if said[start:start + len(rest)] == rest and _corresponds(
                subject, said[:start], cased[:start], possessive,
            ):
                return True
    return False


def _described(subject: Sequence[str], said: Sequence[str], possessive: Sequence[str]) -> bool:
    """``my`` and a description, restated as the reporter's possessive and its ending
    (an informal kinship word may be restated formally, ``mom`` as ``mother``)."""
    description = [FORMAL_KINSHIP.get(word, word) for word in said[1:]]
    restated = [FORMAL_KINSHIP.get(word, word) for word in subject[len(possessive):]]
    return (list(said[:1]) == ["my"] and 1 <= len(description) <= _DESCRIPTION_TOKENS
            and not _NOT_A_NAME & set(description)
            and list(subject[:len(possessive)]) == list(possessive) and bool(restated)
            and description[len(description) - len(restated):] == restated)


def _named(subject: Sequence[str], said: Sequence[str], cased: Sequence[str]) -> bool:
    """A name of at most three capitalized words, restated as it was said, or as the
    name that ends ``my`` and a description (``Anna`` for ``my friend Anna``)."""
    described = list(said[:1]) == ["my"]
    name, size = list(said[1:] if described else said), len(subject)
    if not 1 <= size <= min(len(name), _DESCRIPTION_TOKENS) or (
        not described and size != len(name)
    ):
        return False
    return (name[len(name) - size:] == list(subject) and not _NOT_A_NAME & set(subject)
            and len(cased) == len(said)
            and all(word[:1].isupper() and word.casefold() == token
                    for word, token in zip(cased[len(said) - size:], subject, strict=True)))


def _corresponds(
    subject: Sequence[str], said: Sequence[str], cased: Sequence[str],
    possessive: Sequence[str],
) -> bool:
    """Whether a statement's subject phrase restates a mention (``_described`` or
    ``_named``); a possessive chain is restated whole, so ``my brother's wife`` is never
    ``the user's wife``."""
    whole = "s" not in said[1:] or len(subject) - len(possessive) == len(said) - 1
    return (whole and _described(subject, said, possessive)) or _named(subject, said, cased)


def _line_reading(
    sentence: str, context: str, reporter: str | None,
) -> tuple[str, str, str | None]:
    """A sentence and its line's earlier text with the line's label read away, and the
    sentence's reporter.

    A discourse label keeps the message's reporter; on an unlabelled message a
    capitalized label is the reporter of every sentence in its line; any other label is
    kept as text.
    """
    label = _LABEL.match(context or sentence)
    if label is None:
        return sentence, context, reporter
    name = label["label"]
    if name.casefold() in _DISCOURSE_LABELS:
        speaker = reporter
    elif reporter is None and all(word[:1].isupper() for word in name.split()):
        speaker = name
    else:
        return sentence, context, reporter
    if context:
        return sentence, context[label.end():], speaker
    return sentence[label.end():], context, speaker


def _restates_clause(
    statement: str, sentence: str, context: str | None, reporter: str | None,
) -> bool:
    """``reports_about``'s third-party forms, read in each sentence of a quote's span.

    Only a fully shown message is read (``context`` is not None). A pronoun is resolved
    against the line's earlier text and its own sentence before it, and not at all when
    withheld text or a quotation mark comes earlier in the line. A sentence ending with a
    question mark is skipped.
    """
    if context is None:
        return False
    sentence, context, reporter = _line_reading(sentence, context, reporter)
    words = _uncontracted(_tokens(statement))
    possessive = ["the", "user", "s"] if reporter is None else [*_tokens(reporter), "s"]
    said, cased = _pieces(sentence)
    earlier_said, earlier_cased = _pieces(context)
    readable = WITHHELD_MARKER not in context and _QUOTATION.search(context) is None
    start = 0
    for end in [*(index for index, piece in enumerate(said) if piece in _STOPS), len(said)]:
        if end > start and (end == len(said) or said[end] != "?"):
            prior = ((earlier_said + said[:start], earlier_cased + cased[:start])
                     if readable else None)
            if any(_restated(words, subject, subject_cased, runs, possessive)
                   for subject, subject_cased, runs
                   in _clauses(said[start:end], cased[start:end], prior)):
                return True
        start = end + 1
    return False


def _clauses(
    said: Sequence[str], cased: Sequence[str],
    prior: tuple[Sequence[str], Sequence[str]] | None,
) -> Iterator[tuple[Sequence[str], Sequence[str], _Runs]]:
    """Yield one sentence's third-party subject mentions, each with the predicate runs after it.

    The mentions are: the antecedent of ``he`` or ``she`` opening the sentence or a later
    clause (``_pronoun_clause``); the opening subject phrase with the predicate
    coordinated with its first one (``_joined``); each subject phrase of a clause after
    ``and``, ``but`` or a semicolon with no subordinating word before it; and each
    description or name (``_mentions``) followed by ``who`` or by an appositive ``, a
    ...`` closed by the sentence end or by a comma that opens no further list item, with
    no subordinating word before it and no ``of`` among the three words before it. When that
    description or name is the opening subject and its ``, who ...`` or appositive closes
    with such a comma, the predicate after the comma is its own too (``My mom, a devout
    Hindu, goes to temple``).
    """
    lead = _lead(said)
    yield from _pronoun_clause(said, cased, lead, prior)
    for end in range(lead + 1, min(lead + _DESCRIPTION_TOKENS + 1, len(said)) + 1):
        if said[end - 1] in _PUNCTUATION:
            break
        if joined := _joined(said, end):
            yield said[lead:end], cased[lead:end], joined
    for index, word in enumerate(said):
        if index < lead or word not in _COORDINATORS or _SUBORDINATORS & set(said[:index]):
            continue
        start = index + 1 + _lead(said[index + 1:])
        yield from _pronoun_clause(said, cased, start, prior)
        for end in range(start + 1, min(start + _DESCRIPTION_TOKENS + 1, len(said)) + 1):
            if said[end - 1] in _PUNCTUATION:
                break
            yield said[start:end], cased[start:end], [*_runs(said, end), *_joined(said, end)]
    for begin, end in _mentions(said, cased):
        if "of" in said[max(0, begin - 3):begin] or _SUBORDINATORS & set(said[:begin]):
            continue
        if end < len(said) and said[end] == "who":
            yield said[begin:end], cased[begin:end], _runs(said, end + 1, contracted=True)
            continue
        if end + 1 >= len(said) or said[end] != "," or said[end + 1] not in ("who", "a", "an"):
            continue
        close = next((index for index in range(end + 2, len(said))
                      if said[index] in _PUNCTUATION), len(said))
        after = said[close + 1] if close + 1 < len(said) else None
        listed = close < len(said) and said[close] == "," and (
            after is None or after in _LIST_STARTS or after in _PUNCTUATION)
        main = ([] if listed or begin != lead or close >= len(said) or said[close] != ","
                else _runs(said, close + 1))
        if said[end + 1] == "who":
            yield (said[begin:end], cased[begin:end],
                   [*_runs(said, end + 2, contracted=True), *main])
        elif not {"and", "or", "but"} & set(said[end + 1:close]) and not listed:
            yield (said[begin:end], cased[begin:end],
                   [(["is", *said[end + 1:close]], True), *main])


def _pronoun_clause(
    said: Sequence[str], cased: Sequence[str], start: int,
    prior: tuple[Sequence[str], Sequence[str]] | None,
) -> Iterator[tuple[Sequence[str], Sequence[str], _Runs]]:
    """Yield the antecedent of ``he`` or ``she`` at ``start`` with the runs after it, when
    the earlier text (``prior`` and the sentence before it) mentions exactly one third
    party it can stand for (``_antecedent``) and a statement may restate that mention."""
    if prior is None or start >= len(said) or said[start] not in _SUBJECT_PRONOUNS:
        return
    referent = _antecedent(_SUBJECT_PRONOUNS[said[start]], _referents(
        [*prior[0], *said[:start]], [*prior[1], *cased[:start]]))
    if referent is not None and referent.said:
        yield (referent.said, referent.cased,
               [*_runs(said, start + 1, contracted=True), *_joined(said, start + 1)])


def _runs(said: Sequence[str], start: int, *, contracted: bool = False) -> _Runs:
    """The run from ``start`` to the sentence end; after a pronoun or ``who`` a contracted
    ``'s`` reads as ``is`` or ``has``."""
    run = list(said[start:])
    if contracted and run[:1] == ["s"]:
        return [(["is", *run[1:]], False), (["has", *run[1:]], False)]
    return [(run, False)] if run else []


def _joined(said: Sequence[str], end: int) -> _Runs:
    """The run after the first ``and`` or ``but`` following a subject that ends at
    ``end``, with no subordinating word or semicolon before it (``My sister is a nurse
    and is gay``)."""
    for index in range(end, len(said)):
        if said[index] in _SUBORDINATORS or said[index] == ";":
            return []
        if said[index] in ("and", "but"):
            return _runs(said, index + 1 + _lead(said[index + 1:]))
    return []


def _mentions(said: Sequence[str], cased: Sequence[str]) -> Iterator[tuple[int, int]]:
    """Yield the spans that may name a third party: ``my`` and one to three words, or one
    to three name words (``_is_name``)."""
    for begin, word in enumerate(said):
        if word == "my":
            for end in range(begin + 2, min(begin + _DESCRIPTION_TOKENS + 1, len(said)) + 1):
                if said[end - 1] in _PUNCTUATION:
                    break
                yield begin, end
            continue
        end = begin
        while end < min(begin + _DESCRIPTION_TOKENS, len(said)) and _is_name(said, cased, end):
            end += 1
            yield begin, end


def _restated(
    words: Sequence[str], said: Sequence[str], cased: Sequence[str], runs: _Runs,
    possessive: Sequence[str],
) -> bool:
    """Whether a statement is a subject phrase restating a mention followed by a prefix of
    one of its runs (all of an exact run)."""
    for split in range(1, len(words)):
        subject, rest = words[:split], list(words[split:])
        if any(run == rest if exact else run[:len(rest)] == rest for run, exact in runs) and (
            _corresponds(subject, said, cased, possessive)
        ):
            return True
    return False


def _referents(said: Sequence[str], cased: Sequence[str]) -> list[_Referent]:
    """Return the third parties a text mentions, in order.

    They are ``my`` and a description (``_mine``), with the noun of a possessive chain
    after it (``my mom's friend`` is two people); a determiner or ``his``/``her`` and a
    person noun (``the nurse``, ``her sister``); a bare person noun (``Mom``, ``kids``); a
    name (``_is_name``); and a third-person pronoun that no earlier mention of its gender
    explains, which stands for someone the text does not name.
    """
    found: list[_Referent] = []
    index = 0
    while index < len(said):
        word = said[index]
        if word == "my":
            index = _possessed(said, cased, index, found)
            continue
        if word in _THIRD_PERSON and not any(
            not seen.plural and seen.gender in (_THIRD_PERSON[word], "u") for seen in found
        ):
            found.append(_Referent(("outside", str(index)), _THIRD_PERSON[word], False))
        if word in _DETERMINERS:
            index = _determined(said, index, found)
            continue
        if word in _PEOPLE and word not in _VERB_LIKE:
            gender, plural = _PEOPLE[word]
            key = ("one", _ONE_OF[word]) if word in _ONE_OF else ("bare", str(index))
            found.append(_Referent(key, gender, plural))
        elif _is_name(said, cased, index):
            end = index + 1
            while end < min(index + _DESCRIPTION_TOKENS, len(said)) and _is_name(
                said, cased, end,
            ):
                end += 1
            found.append(_Referent(("name", *said[index:end]), "u", False,
                                   tuple(said[index:end]), tuple(cased[index:end])))
            index = end
            continue
        index += 1
    return found


def _possessed(
    said: Sequence[str], cased: Sequence[str], index: int, found: list[_Referent],
) -> int:
    """Add ``my`` and its description, then a possessive chain's noun (``my mom's
    friend``), to ``found``; return where the scan resumes."""
    after = _mine(said, cased, index, index + 1, found)
    if after >= len(said) or said[after] != "s":
        return max(after, index + 1)
    return max(_mine(said, cased, index, after + 1, found), after + 1)


def _mine(
    said: Sequence[str], cased: Sequence[str], index: int, start: int,
    found: list[_Referent],
) -> int:
    """Add the person a description from ``start`` names after the ``my`` at ``index``.

    The description runs to its last person noun or name within three words; without
    one it names a person of unknown gender unless a word in it names none (``my car``).
    Returns where the description ends.
    """
    end = _phrase_end(said, start)
    heads = [position for position in range(start, end)
             if said[position] in _PEOPLE or _is_name(said, cased, position)]
    if not heads:
        if end > start and not _NOT_PERSONS & set(said[start:end]):
            found.append(_Referent(("my", str(start)), "u", False,
                                   tuple(said[index:end]), tuple(cased[index:end])))
        return end
    head = heads[-1]
    if _is_name(said, cased, head):
        first = head
        while first > start and _is_name(said, cased, first - 1):
            first -= 1
        gender = next((_PEOPLE[said[position]][0] for position in range(start, first)
                       if said[position] in _PEOPLE), "u")
        key, plural = ("name", *said[first:head + 1]), False
    else:
        gender, plural = _PEOPLE[said[head]]
        key = (("one", _ONE_OF[said[head]]) if head == start == index + 1
               and said[head] in _ONE_OF else ("my", str(head)))
    found.append(_Referent(key, gender, plural, tuple(said[index:head + 1]),
                           tuple(cased[index:head + 1])))
    return head + 1


def _determined(said: Sequence[str], index: int, found: list[_Referent]) -> int:
    """Add the person noun ending a determiner's phrase (``the nurse``) to ``found``;
    return where the scan resumes."""
    end = _phrase_end(said, index + 1)
    heads = [position for position in range(index + 1, end) if said[position] in _PEOPLE]
    if not heads:
        return index + 1
    gender, plural = _PEOPLE[said[heads[-1]]]
    found.append(_Referent(("np", str(heads[-1])), gender, plural))
    return heads[-1] + 1


def _phrase_end(said: Sequence[str], start: int) -> int:
    """Where a noun phrase of at most three words from ``start`` ends."""
    end = start
    while end < min(start + _DESCRIPTION_TOKENS, len(said)) and said[end] not in _WINDOW_BREAKS:
        end += 1
    return end


def _is_name(said: Sequence[str], cased: Sequence[str], index: int) -> bool:
    """Whether a word may be part of a person's name: capitalized and alphabetic, and no
    person noun, date, faith, nationality or common sentence opener, nor a place after
    ``in``."""
    word = said[index]
    return (cased[index][:1].isupper() and cased[index].isalpha() and word not in _PEOPLE
            and word not in _NOT_NAMES and word not in _NOT_A_NAME
            and not (index > 0 and said[index - 1] == "in"))


def _antecedent(gender: str, referents: Sequence[_Referent]) -> _Referent | None:
    """The one mentioned third party ``he`` or ``she`` of this gender stands for.

    Candidates are the singular referents of that gender or of unknown gender; mentions
    with the same key are one candidate. Returns None for none or for several.
    """
    agreeing: dict[tuple[str, ...], _Referent] = {}
    for referent in referents:
        if referent.plural or referent.gender not in (gender, "u"):
            continue
        if referent.key not in agreeing or not agreeing[referent.key].said:
            agreeing[referent.key] = referent
    return next(iter(agreeing.values())) if len(agreeing) == 1 else None


def _pieces(text: str) -> tuple[list[str], list[str]]:
    """A text's words and clause punctuation: casefolded with ``n't`` read as ``not``, and
    as written."""
    cased = _PIECE.findall(unicodedata.normalize("NFKC", text))
    return _uncontracted([piece.casefold() for piece in cased]), cased


def _lead(words: Sequence[str]) -> int:
    """How many opening tokens are ``and``, ``but``, ``also`` and the like."""
    count = 0
    while count < len(words) and words[count] in _REPORT_LEAD:
        count += 1
    return count


def _uncontracted(words: Sequence[str]) -> list[str]:
    """Tokens with each ``n't`` contraction read as its verb and ``not``."""
    out = list(words)
    for index in range(1, len(out)):
        if out[index] == "t" and out[index - 1] in _UNCONTRACTED:
            out[index - 1], out[index] = _UNCONTRACTED[out[index - 1]], "not"
    return out


def _english(text: str) -> bool:
    return _OTHER_SCRIPT.search(text) is None


def _tokens(text: str) -> list[str]:
    return _WORD.findall(unicodedata.normalize("NFKC", text).casefold())


def _main(statement: str) -> str:
    """A statement without its trailing ``, not ...`` contrast."""
    return _CONTRAST.sub("", statement.strip())


def _negated(words: Sequence[str], index: int) -> bool:
    word = words[index]
    return word in _NEGATIONS or word == "t" and index > 0 and words[index - 1] in _CONTRACTED


def _negations(words: Sequence[str]) -> int:
    return sum(_negated(words, index) for index in range(len(words)))


def _content(words: Sequence[str]) -> set[str]:
    """The stems of a token run's content words: no negation, function word or ``n't`` head."""
    found: set[str] = set()
    for index, word in enumerate(words):
        if _negated(words, index) or word in _FUNCTION or (
            word in _CONTRACTED and index + 1 < len(words) and words[index + 1] == "t"
        ):
            continue
        found.add(_stem(word))
    return found


def _topic(statement: str) -> set[str]:
    """The stems of a statement's topic words, an informal kinship word spelled formally."""
    return _topic_words(_tokens(statement))


def _topic_words(words: Sequence[str]) -> set[str]:
    """The topic stems of a token run, read as ``_topic`` reads a statement's."""
    formal = [FORMAL_KINSHIP.get(word, word) for word in words]
    return _content(formal) - _PREPOSITIONS - _UNTOPICAL


def _stem(word: str) -> str:
    if word in _HAVE:
        return _HAVE[word]
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[:-len(suffix)]
    return word
