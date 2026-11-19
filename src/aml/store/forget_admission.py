"""Pure D.3 P1 selection over the selected Item's complete NL revision chain.

This preflight grants no mutation authority. Admission must bind its inputs
under the Writer, check authorization/intent/epochs, and commit atomically.
``plan_stating_spans`` then narrows the cited spans to the stating spans S that
the forget withholds (D.3 P1 #6, UD3 as amended by O11, the bare-name rule, the
referent keeper and the clause keeper).
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Callable, Iterable, Sequence, Set
from dataclasses import dataclass, field, replace

from aml.core.entities import LinkType, MemorySnapshotV2, QuoteSpanV1, SupersedeRelation
from aml.core.errors import AMLError
from aml.core.forget import MAX_FORGET_SOURCES, MAX_FORGET_TOMBSTONES
from aml.core.forget_state import MAX_FORGET_CHAIN
from aml.core.judge_structure import FORMAL_KINSHIP
from aml.core.naming_syntax import bare_naming
from aml.core.privacy import support_words, word_runs
from aml.core.sensitivity_rules import classify
from aml.core.statement import MAX_TOMBSTONE_SOURCES, name_norm, token_contained, token_seq
from aml.store.domain import ItemRowV2, RevisionRowV2


@dataclass(frozen=True)
class CorrectionEdge:
    """A same-scope extant Item link with the source proposal's correction provenance.

    ``correction_requested`` is the link's own flag: on a ``contradicts`` link the
    proposal requested a correction (C.1), on a ``supersedes`` link it stated an
    explicit correction.
    """

    target: str
    kind: LinkType
    relation: SupersedeRelation | None
    correction_requested: bool


def correction_chain(memory_id: str, links: Callable[[str], Iterable[CorrectionEdge]],
                     ) -> tuple[str, ...]:
    """Return every Item the selected Item transitively refuted (D.3, FG-2).

    Args:
        memory_id: The selected Item, excluded from the result even in a cycle.
        links: All revisions' links to extant Items of the same scope. The caller
            proves scope and original proposal provenance before yielding each edge.

    Returns:
        Sorted correction/retraction predecessors, including contradiction targets
        only when their source proposal requested a correction and update targets
        only when it stated an explicit correction. Other updates and ordinary
        disagreements remain outside the retention chain.

    Raises:
        AMLError: The chain exceeds the production admission bound.
    """
    found: set[str] = set()
    pending = [memory_id]
    while pending:
        for edge in links(pending.pop()):
            if not (edge.kind == "contradicts" and edge.correction_requested or
                    edge.kind == "supersedes" and (edge.relation in ("correct", "retract")
                                                   or edge.correction_requested)):
                continue
            if edge.target == memory_id or edge.target in found:
                continue
            if len(found) >= MAX_FORGET_CHAIN:
                raise AMLError("budget_exceeded", "The forget chain exceeds its bound")
            found.add(edge.target)
            pending.append(edge.target)
    return tuple(sorted(found))


@dataclass(frozen=True)
class ForgetVersionSelection:
    """Newest version of one distinct pair, with all of that pair's sources.

    The snapshot carries the original vector; Forget never embeds it again.
    This is private plan input, never a receipt or a read projection.
    """

    revision_id: str
    snapshot: MemorySnapshotV2 = field(repr=False)
    source_evidence_ids: tuple[str, ...]


@dataclass(frozen=True)
class ForgottenCitation:
    """What one non-erased forgotten version cites, for the stating-span plan (O11).

    Attributes:
        entity_id: The version's primary entity.
        names: The ``(entity_id, name_norm)`` pairs its naming annotations give.
        groups: Its support groups, each as the spans it quotes.
        about: Its other anchors (``about_entity_ids``).
        sensitive: Whether this version has any sensitivity class.
        statement: Its statement, read only by the detail-keeper test
            (``detail_names``); private plan input, never rendered.
        vector: Its statement vector and ``space`` its embedding space, read only by
            the clause keeper (``clause_spans``); private plan input.
    """

    entity_id: str
    names: frozenset[tuple[str, str]] = field(repr=False)
    groups: tuple[tuple[QuoteSpanV1, ...], ...]
    about: frozenset[str] = field(default=frozenset[str](), repr=False)
    sensitive: bool = False
    statement: str = field(default="", repr=False)
    vector: tuple[float, ...] = field(default=(), repr=False)
    space: str = field(default="", repr=False)


@dataclass(frozen=True)
class KeeperGroup:
    """One readable support group of a surviving memory that may keep quotes (O11).

    A naming memory's group may keep its quotes on paths (a)-(c); any memory's group
    may keep its referent quote on path (d) (``referent_spans``); a head a forgotten
    correction reworded keeps its own groups on path (f) (``reworded``, OPR-2).

    Attributes:
        names: The ``(entity_id, name_norm)`` pairs the survivor's head gives; empty
            for a memory that names nothing.
        spans: The group's quote spans; each passes P-SPAN at admission.
        bare: The one entity the survivor's head anchors when the head names it
            by type alone ("A person is named Rowan."). Its exact name spans,
            including names within broader quotes, are the only bytes this path keeps;
            otherwise None.
        sensitive: Whether the survivor's head has any sensitivity class; a
            sensitive keeper keeps nothing on any path.
        statement: The survivor head's statement (``detail_names``, ``referent_spans``).
        anchors: The survivor head's anchors, primary first.
        quoted: The text each of ``spans`` quotes, in the same order.
        persons: The named entities whose type is "person" (``detail_names``).
        stated: ``support_words`` of the head's statement and time expression
            (``clause_spans``).
        vector: The head's statement vector and ``space`` its embedding space
            (``clause_spans``).
        reworded: The survivor's head is a rewording by a forgotten correction (C.9.4a):
            the group is its own support, kept for every version that withdraws no
            name (path f, ``plan_stating_spans``).
        speakers: ``support_words`` of the speaker each of ``spans`` is attributed to
            (``Evidence.speaker_entity``), in the same order; empty when unknown. A quote's
            speaker is readable provenance, so it grounds the speaker's name in the
            survivor's statement (``clause_spans``, C26 (a)).
        reference_names: Readable names of the primary entity given by surviving naming
            memories. Only the descriptor-detail fallback reads these; they grant no
            naming annotation, bare-name keeper or identity binding.
        entity_type: The stored broad type of the primary entity, used only to
            reject a referent that restates possession of that same kind.
    """

    names: frozenset[tuple[str, str]] = field(repr=False)
    spans: tuple[QuoteSpanV1, ...]
    bare: str | None = None
    sensitive: bool = False
    statement: str = field(default="", repr=False)
    anchors: tuple[str, ...] = field(default=(), repr=False)
    quoted: tuple[str, ...] = field(default=(), repr=False)
    persons: frozenset[str] = field(default=frozenset[str](), repr=False)
    stated: frozenset[str] = field(default=frozenset[str](), repr=False)
    vector: tuple[float, ...] = field(default=(), repr=False)
    space: str = field(default="", repr=False)
    reworded: bool = False
    speakers: tuple[frozenset[str], ...] = field(default=(), repr=False)
    reference_names: frozenset[tuple[str, str]] = field(default=frozenset(), repr=False)
    entity_type: str = field(default="", repr=False)


@dataclass(frozen=True)
class ForgetAdmissionSelection:
    """Complete distinct tombstone versions and exact cited spans of one Item.

    ``spans`` is every cited span; ``cited`` keeps each non-erased version's own
    groups so that ``plan_stating_spans`` can tell what states the forgotten fact.
    """

    versions: tuple[ForgetVersionSelection, ...] = field(repr=False)
    spans: tuple[QuoteSpanV1, ...]
    cited: tuple[ForgottenCitation, ...] = field(default=(), repr=False)


def plan_forget_admission(
    item: ItemRowV2,
    revisions: Sequence[RevisionRowV2],
) -> ForgetAdmissionSelection:
    """Select D and S before any P1 write, independently of heartbeat traffic.

    Revisions must be the complete chain of exactly the selected Item. All
    lifecycle states are accepted. Erased versions add neither tombstones nor
    spans; an erased head still retains the non-erased versions before it.
    The caller must prove inventory completeness against the durable store.

    Raises:
        AMLError: For an incomplete/misbound chain or D.3's pre-write bounds.
    """
    item = ItemRowV2.model_validate(item.model_dump(mode="python"))
    ordered = sorted(
        (RevisionRowV2.model_validate(row.model_dump(mode="python")) for row in revisions),
        key=lambda row: row.version,
    )
    if not ordered:
        raise AMLError("closure_incomplete", "Forget needs the complete selected revision chain")
    previous: str | None = None
    seen: set[str] = set()
    newest: dict[tuple[str, str], RevisionRowV2] = {}
    sources: dict[tuple[str, str], set[str]] = {}
    spans: dict[tuple[str, int, int], QuoteSpanV1] = {}
    cited: list[ForgottenCitation] = []
    for version, row in enumerate(ordered, 1):
        if row.memory_id != item.object_id:
            raise AMLError("binding_mismatch", "Forget revision belongs to another Item")
        if row.version != version or row.previous_revision_id != previous or row.object_id in seen:
            raise AMLError("closure_incomplete", "Forget revision chain is incomplete or ambiguous")
        seen.add(row.object_id)
        previous = row.object_id
        snapshot = row.snapshot
        if snapshot.erased:
            continue
        cited.append(ForgottenCitation(
            snapshot.entity_id,
            frozenset((entry.entity_id, name_norm(entry.name)) for entry in snapshot.naming),
            tuple(tuple(snapshot.quote_spans[index] for index in group)
                  for group in snapshot.support_groups),
            frozenset(snapshot.about_entity_ids),
            bool(snapshot.sensitivity_classes),
            snapshot.statement,
            tuple(snapshot.embedding),
            snapshot.embedding_space,
        ))
        pair = snapshot.entity_id, snapshot.statement_digest
        newest[pair] = row
        pair_sources = sources.setdefault(pair, set())
        for group in snapshot.support_groups:
            for index in group:
                span = snapshot.quote_spans[index]
                pair_sources.add(span.evidence_id)
                spans[(span.evidence_id, span.byte_start, span.byte_end)] = span
        if len(newest) > MAX_FORGET_TOMBSTONES or len(pair_sources) > MAX_TOMBSTONE_SOURCES:
            raise AMLError("budget_exceeded", "Forget exceeds its tombstone admission bound")
    head = ordered[-1]
    if head.object_id != item.current_revision_id or (
        head.snapshot.entity_id, head.snapshot.statement_digest
    ) != (item.entity_id, item.statement_digest):
        raise AMLError("binding_mismatch", "Forget head differs from the selected Item")
    if len({key[0] for key in spans}) > MAX_FORGET_SOURCES:
        raise AMLError("budget_exceeded", "Forget exceeds its withheld source admission bound")
    return ForgetAdmissionSelection(
        versions=tuple(
            ForgetVersionSelection(newest[pair].object_id, newest[pair].snapshot,
                                   tuple(sorted(sources[pair])))
            for pair in sorted(newest)
        ),
        spans=tuple(spans[key] for key in sorted(spans)),
        cited=tuple(cited),
    )


def plan_earlier_wordings(
    revisions: Sequence[RevisionRowV2], verdicts: Set[str],
) -> tuple[ForgetVersionSelection, ...]:
    """The wordings a surviving Item had before a forgotten correction reworded it (OPR-2).

    The Item's versions older than its newest ``rewording`` whose verdict is one of
    ``verdicts`` (the forgotten correction's) carry the description that correction
    replaced, so they join D like the versions of a forgotten Item and a restatement of
    them is refused (D.4); the Item itself survives and its head keeps its own quotes.

    Args:
        revisions: Every revision of one surviving Item.
        verdicts: The verdicts of the forgotten Items' versions.

    Returns:
        The newest version of each distinct ``(entity_id, statement_digest)`` pair among
        those earlier versions, with all of that pair's sources; empty when no
        rewording carries one of ``verdicts``.

    Raises:
        AMLError: ``budget_exceeded`` past D.3's tombstone admission bounds.
    """
    ordered = sorted(revisions, key=lambda row: row.version)
    cut = max((row.version for row in ordered if row.reason_code == "rewording"
               and row.snapshot.verdict_ref in verdicts), default=None)
    newest: dict[tuple[str, str], RevisionRowV2] = {}
    sources: dict[tuple[str, str], set[str]] = {}
    for row in ordered:
        if cut is None or row.version >= cut or row.snapshot.erased:
            continue
        pair = row.snapshot.entity_id, row.snapshot.statement_digest
        newest[pair] = row
        sources.setdefault(pair, set()).update(row.evidence_ids)
        if len(newest) > MAX_FORGET_TOMBSTONES or len(sources[pair]) > MAX_TOMBSTONE_SOURCES:
            raise AMLError("budget_exceeded", "Forget exceeds its tombstone admission bound")
    return tuple(ForgetVersionSelection(newest[pair].object_id, newest[pair].snapshot,
                                        tuple(sorted(sources[pair]))) for pair in sorted(newest))


def named_by_type(statement: str, given: Set[str], others: Set[tuple[str, ...]]) -> bool:
    """Whether a naming statement names its entity by type alone, as one of ``given``.

    A bare-name keeper's head must (D.3 P1 #6): its type is exactly one allowlisted
    neutral kind, never a modifier, relation, role, occupation, animal or clause.
    It never says "user" nor names
    another anchor of the forgotten versions (a token run of ``others``), so "A person who
    is Avery's spouse is named Rowan." or "A friend of the user is named Rowan." does
    not keep, since it restates "Avery is married to Rowan." or a fact of the user.

    Args:
        statement: The naming head's statement.
        given: The ``name_norm`` of each name the head gives its one anchor.
        others: The ``name_norm`` token runs of "user" and of the other anchors' names.

    Returns:
        True for the bare-name syntax (`bare_naming`: an English neutral kind or an exact
        Chinese classifier kind) with a name of ``given`` and no forbidden kind token run.
    """
    found = bare_naming(statement)
    if found is None or name_norm(found[1]) not in given:
        return False
    kind = token_seq(found[0])
    return not any(
        kind[start:start + len(run)] == list(run)
        for run in others if run for start in range(len(kind) - len(run) + 1)
    )


def bare_name_spans(keeper: KeeperGroup) -> tuple[QuoteSpanV1, ...]:
    """Locate the literal name of a neutral naming head inside its quoted support.

    A broad quote such as "My friend Nora" supports a neutral naming head without
    granting the relationship words readability. Match the complete written name,
    with word boundaries, and convert original character offsets to UTF-8 bytes;
    normalized tokens are never used as offsets into source text.

    Args:
        keeper: A group marked bare only after the neutral-kind and sole-anchor
            checks. The head must give exactly the name its naming syntax states.

    Returns:
        Sorted unique name subspans, or none for an unmatched or non-neutral head.
    """
    found = bare_naming(keeper.statement)
    if (keeper.bare in (None, "", "user") or found is None
            or keeper.anchors != (keeper.bare,)
            or keeper.names != frozenset({(keeper.bare, name_norm(found[1]))})
            or len(keeper.quoted) != len(keeper.spans)):
        return ()
    pattern = re.compile(r"(?<!\w)" + re.escape(found[1]) + r"(?!\w)")
    located: set[QuoteSpanV1] = set()
    for span, text in zip(keeper.spans, keeper.quoted, strict=True):
        if name_norm(text) == name_norm(found[1]):
            located.add(span)
            continue
        for match in pattern.finditer(text):
            if (match.start() and unicodedata.category(text[match.start() - 1]).startswith("M")
                    or match.end() < len(text)
                    and unicodedata.category(text[match.end()]).startswith("M")):
                continue
            start = span.byte_start + len(text[:match.start()].encode("utf-8"))
            located.add(QuoteSpanV1(
                evidence_id=span.evidence_id, byte_start=start,
                byte_end=start + len(match.group().encode("utf-8")),
            ))
    return tuple(sorted(located, key=lambda span: (
        span.evidence_id, span.byte_start, span.byte_end)))


def _stem(token: str) -> str:
    """A crude English stem, so that "bathes", "bathed" and "bathe" compare equal."""
    word = token
    if len(word) >= 5 and word.endswith(("ies", "ied")):
        word = word[:-3] + "y"
    elif len(word) >= 6 and word.endswith("ing"):
        word = word[:-3]
    elif len(word) >= 5 and word.endswith("ed"):
        word = word[:-2]
    elif len(word) >= 4 and word.endswith("s") and not word.endswith("ss"):
        word = word[:-1]
    if len(word) >= 5 and word.endswith("e"):
        word = word[:-1]
    return word


def _stems(words: str) -> frozenset[str]:
    return frozenset(_stem(word) for word in words.split())


_LIGHT = _stems("""
    a an the this that these those some any each every all both either neither such same
    i me my mine myself we us our ours you your yours he him his she her hers they them their
    theirs it its itself who whom whose which what one ones s t re ve ll d m
    is are was were be been being am do does did doing done will would shall should can could
    may might must named called name names nicknamed known goes go by as
    of to in on at for with from into onto about over under after before during until till
    since than then and or but nor so if while because also too just only even still again
    already now currently recently there here not no never longer anymore none
    today yesterday tomorrow tonight morning mornings evening evenings night nights afternoon
    afternoons day days daily week weeks weekly weekend weekends month months monthly year
    years yearly ago last next ever always often sometimes usually soon later early earlier
    monday tuesday wednesday thursday friday saturday sunday january february march april
    june july august september october november december
    new other another second third first older oldest younger youngest elder eldest
    two three four five six seven eight nine ten eleven twelve twenty thirty hundred
""")
"""Words that never carry a fact on their own here: function words, naming verbs, negation,
time and number words, and the set-apart words a naming fact keeps (B.13 #7)."""

_RELATION = _stems("""
    have has had having own owns owned owning owner owners possess possesses possessed
    adopt adopts adopted adopting adoption buy buys bought buying purchase purchased
    rescue rescued acquire acquired receive received gift gifted get gets got gotten getting
    keep keeps kept foster fostered inherit inherited raise raised breed bred
    marry marries married marrying marriage wed weds wedded wedding spouse husband wife
    partner partners engaged engagement divorce divorced dating date dated girlfriend
    boyfriend fiance fiancee sibling siblings sister sisters brother brothers mother father
    mom dad mum parent parents son sons daughter daughters child children kid kids baby
    babies aunt uncle cousin niece nephew grandmother grandfather grandma grandpa grandparent
    grandparents grandchild grandson granddaughter twin relative relatives family friend
    friends roommate roommates neighbor neighbors neighbour neighbours colleague colleagues
    coworker coworkers boss manager employer employee employees mentor student teacher
    classmate teammate landlord tenant client patient doctor therapist dentist work works
    worked working employ employs employed employment hire hired job jobs intern interned
    internship volunteer volunteers volunteered volunteering use uses used using member
    members belong belongs join joins joined attend attends attended study studies studied
    enroll enrolled
""")
"""Possession, acquisition, kinship, partnership, social, employment, use and membership
words: any of them among what a forgotten version adds over a naming memory makes the
naming memory a possible restatement of it, so the detail keeper keeps nothing."""


_QUOTE_LIGHT = _stems("""
    a an the my mine our ours your his her hers their its i we you he she they it me us him
    them s is are was were named called name nicknamed
""")
"""Words a detail keeper's quote may carry beyond its own statement's: determiners, pronouns,
"is" and naming verbs ("Our dog is called Biscuit", "a tortoise named Pip")."""


def _content(text: str) -> frozenset[str]:
    """The stems of ``text``'s tokens (``token_seq``) that may carry a fact."""
    return frozenset(_stem(token) for token in token_seq(text)
                     if not token.isdigit()) - _LIGHT


_POSSESSORS = re.compile(r"(?<![\w'\u2019])(the user|[A-Z][A-Za-z-]*(?: [A-Z][A-Za-z-]*){0,2})"
                         r"['\u2019]s\b", re.IGNORECASE)
"""Every "<possessor>'s" of a statement: "the user's", or a capitalized name's."""


def _owners(keeper: KeeperGroup) -> list[list[str]]:
    """The token runs naming the owner a naming memory describes its entity by: "user"
    when its head anchors the user, and each possessor of "<possessor>'s" in its
    statement ("Avery" in "Avery's hamster is named Mochi.")."""
    owners = [["user"]] if "user" in keeper.anchors else []
    for match in _POSSESSORS.finditer(keeper.statement):
        words = match.group(1)
        if words.casefold() == "the user":
            owners.append(["user"])
        elif words.isascii() and all(word[:1].isupper() for word in words.split()):
            owners.append(token_seq(words))
    return owners


def _owner_takes_part(statement: str, owners: Sequence[Sequence[str]]) -> bool:
    """Whether ``statement`` names one of ``owners`` other than as "<owner>'s"."""
    tokens = token_seq(statement)
    for owner in owners:
        size = len(owner)
        for start in range(len(tokens) - size + 1):
            after = tokens[start + size:start + size + 1]
            if tokens[start:start + size] == list(owner) and after != ["s"]:
                return True
    return False


def detail_names(version: ForgottenCitation, keeper: KeeperGroup) -> set[tuple[str, str]]:
    """The names a detail keeper keeps for ``version`` (D.3 P1 #6 (c), R1).

    A naming memory that also anchors its entity's owner or the user ("The user's
    tortoise is named Pip.") keeps the quote that establishes the entity when the
    forgotten version states a different detail of that entity: "The user bathes the
    user's tortoise Pip every morning." It keeps nothing when the version could be a
    restatement of it: "The user has a tortoise." or "Avery has a hamster named
    Mochi." The test is on words and errs toward withholding:

    * the keeper names exactly one entity e, never the user; e and every anchor of the
      keeper's head are anchors of the version;
    * every word of the keeper's statement other than e's name, function, naming, time,
      number or set-apart words, is a word of the version's statement (it describes e
      exactly as the version does and says nothing the version does not);
    * what the version says beyond the keeper is not empty and has no possession,
      acquisition, kinship, partnership, social, employment, use or membership word;
      those word lists are English, so a word beyond the keeper in another script or
      with a diacritic (ML-1) is one the test cannot judge, and the keeper keeps nothing;
    * when e is a person, the version names the keeper's owner (the user, or the
      possessor of "<owner>'s") only as that possessor: a version in which the owner
      takes part ("The user cohabits with the user's roommate Rowan.", "The user rents
      to the user's tenant Sam.") states a relation between the owner and e, which the
      keeper's description of e ("roommate", "tenant") can restate in other words that
      no closed word list catches (C21 R1), so the keeper keeps nothing. A version about
      e itself ("The user's roommate Rowan works as an engineer.") still keeps;
    * every word of every quote of the keeper is a word of the keeper's own statement,
      a determiner, a pronoun, "is" or a naming verb: the bytes left readable state
      nothing the naming memory does not, so no byte of the forgotten detail, and no
      preposition or verb hinting at it ("at St. Mary's Hospital"), stays readable.

    Args:
        version: One non-erased forgotten version, with its statement.
        keeper: One readable group of a surviving naming memory, with its head's
            statement, anchors and quoted text.

    Returns:
        The keeper's names when it keeps its quotes for ``version``; else empty.
    """
    entities = {entity for entity, _ in keeper.names}
    anchors = {version.entity_id, *version.about}
    if len(entities) != 1 or not version.statement or not keeper.statement:
        return set()
    (entity,) = entities
    if (entity in ("", "user") or entity not in anchors or not keeper.anchors
            or not set(keeper.anchors) <= anchors or len(keeper.quoted) != len(keeper.spans)):
        return set()
    named = frozenset(_stem(token) for _, name in keeper.names for token in name.split())
    said = _content(keeper.statement)
    stated = _content(version.statement)
    if not said - named <= stated:
        return set()
    extra = stated - said
    if not extra or extra & _RELATION or not all(word.isascii() for word in extra):
        return set()
    if entity in keeper.persons and _owner_takes_part(version.statement, _owners(keeper)):
        return set()
    allowed = frozenset(_stem(token) for token in token_seq(keeper.statement)) | _QUOTE_LIGHT
    if any(not frozenset(_stem(token) for token in token_seq(text)) <= allowed
           for text in keeper.quoted):
        return set()
    return set(keeper.names)


_POSSESSOR = re.compile(r"(the user|[A-Z][A-Za-z-]*(?: [A-Z][A-Za-z-]*){0,2})['\u2019]s\s+",
                        re.IGNORECASE)
"""A statement's opening possessor: "The user's", or a capitalized English name of at most
three words ("Avery's"), as a statement restates a speaker's "my"."""

_REFERENT_WORDS = 3
"""Description words a referent keeper's quote may hold after "my" (rule 5's bound)."""

_SUBJECT = re.compile(r"([A-Z][A-Za-z-]*(?: [A-Z][A-Za-z-]*){0,2})(?=,? )")
"""A statement's opening subject: one to three capitalized English words before a space
or an appositive comma ("Avery applied ...", "Blair got ...")."""

_OPENERS = frozenset({
    "yesterday", "today", "tomorrow", "tonight", "last", "next", "every", "each", "once",
    "after", "before", "when", "while", "during", "since", "until", "recently", "currently",
    "previously", "now", "then", "later", "earlier", "sometimes", "often", "always", "never",
    "usually", "both", "all", "most", "many", "several", "no", "none", "not", "one", "two",
    "three", "there", "here", "how", "what", "why", "where", "if", "although", "because",
})
"""Capitalized words that open a sentence without naming its subject."""

_HAVE = frozenset({"has", "have", "had", "own", "owns", "owned"})
"""Verbs by which a statement says its subject possesses something ("Blair has pets.")."""

_GAIN = _HAVE | frozenset({
    "got", "gets", "keeps", "kept", "bought", "buys", "adopted", "adopts", "acquired",
    "received", "found",
})
"""Verbs by which a statement says its subject possesses or gains something."""


def _opening(statement: str) -> tuple[str, list[str]] | None:
    """The possessor and the tokens after it when ``statement`` opens with "<possessor>'s"."""
    match = _POSSESSOR.match(statement)
    if match is None or not match.group(1).isascii() or (
        match.group(1).casefold() != "the user"
        and not all(word[:1].isupper() for word in match.group(1).split())
    ):
        return None
    return match.group(1).casefold(), token_seq(statement[match.end():])


def _formal(words: Sequence[str]) -> list[str]:
    return [FORMAL_KINSHIP.get(word, word) for word in words]


def _describes(rest: Sequence[str], description: Sequence[str]) -> bool:
    """Whether ``rest`` opens with ``description`` as a whole noun phrase: the same words
    (an informal kinship word may be formal on either side), then a word other than "s"."""
    size = len(description)
    return (len(rest) > size and rest[size] != "s"
            and _formal(rest[:size]) == _formal(description))


def _said(words: Sequence[str]) -> frozenset[str]:
    """The content stems of ``words``, each word also in its formal kinship spelling."""
    return frozenset(_stem(form) for word in words if not word.isdigit()
                     for form in (word, FORMAL_KINSHIP.get(word, word))) - _LIGHT


def referent_spans(
    version: ForgottenCitation, keeper: KeeperGroup, forgotten: Set[tuple[str, str]],
) -> tuple[QuoteSpanV1, ...]:
    """The quotes a referent keeper keeps readable for ``version`` (D.3 P1 #6 (d)).

    A forgotten sentence can carry the phrase by which another memory names its own
    subject: "My brother" in "My brother is allergic to peanuts. He loves chess." supports
    both the forgotten "The user's brother is allergic to peanuts." and the surviving
    "The user's brother loves chess." (group "My brother", "He loves chess."). The
    survivor keeps that phrase readable, so it stays readable itself, while every word of
    the forgotten predicate is withheld. A quote counts only when all of these hold, and
    the test errs toward withholding:

    * the keeper's head has no sensitivity class and its primary entity is not the user;
    * the quote is "my" and one to three English description words, none of them "s";
    * the head's statement opens with "<possessor>'s" ("The user's" or a capitalized name)
      and exactly that description as a whole noun phrase (an informal kinship word may be
      formal on either side: "My mom" for "The user's mother loves chess.");
    * the quote has no sensitivity class (``classify``) and contains no name a forgotten
      version gives, so "my diabetic brother" or "my brother Tom" (Tom forgotten) keeps
      nothing when the classifier or the names show it;
    * no content stem of the description is a content stem of what the version states
      about it: its words after the same opening when it opens with one, else all of its
      words. "My brother" keeps for "The user's brother is allergic to peanuts." (the
      predicate is "is allergic to peanuts") and nothing for "The user has a brother.",
      "The user's brother's wife is ill." or "The user's brother is a monk's brother.".

    A possession of the same subject also disqualifies the phrase when its entire
    object matches the primary entity's stored broad type: a pet's "My cat" restates
    "The user has pets.". Negation, quantity, time and additional descriptors are not
    discarded by this kind check. It introduces no taxonomy or identity inference.

    A sensitive forgotten version does not by itself disqualify this path, as for path
    (a): only the survivor's own referring phrase stays readable.

    Args:
        version: One non-erased forgotten version, with its statement.
        keeper: One readable group of a surviving memory, with its head's statement,
            anchors and quoted text.
        forgotten: Every ``(entity_id, name_norm)`` pair a forgotten version gives.

    Returns:
        The keeper's quotes that are its referent for ``version``; empty when none is.
    """
    head = _opening(keeper.statement)
    if (head is None or keeper.sensitive or not version.statement or not keeper.anchors
            or keeper.anchors[0] in ("", "user") or len(keeper.quoted) != len(keeper.spans)):
        return ()
    possessor, rest = head
    subject = _subject(version.statement)
    if subject is not None and possession_entailed(version.statement, keeper.statement):
        tokens = token_seq(version.statement)
        possessed = support_words(" ".join(tokens[len(token_seq(subject)) + 1:]))
        if possessed and possessed == support_words(keeper.entity_type):
            return ()
    stated = _opening(version.statement)
    kept: list[QuoteSpanV1] = []
    for span, text in zip(keeper.spans, keeper.quoted, strict=True):
        words = token_seq(text)
        description = words[1:]
        if (not text.isascii() or words[:1] != ["my"] or "s" in description
                or not 1 <= len(description) <= _REFERENT_WORDS
                or not _describes(rest, description) or classify([text])
                or any(token_contained(name, text) for _, name in forgotten)):
            continue
        if stated is not None and stated[0] == possessor and _describes(stated[1], description):
            predicate = stated[1][len(description):]
        else:
            predicate = token_seq(version.statement)
        if _said(description).isdisjoint(_said(predicate)):
            kept.append(span)
    return tuple(kept)


def _ancestor_naming_words(forgotten: str, survivor: str) -> frozenset[str] | None:
    """The descendant and name alone for an exact ancestor-existence keeper (C26).

    ``The user has a sister.`` states neither the dog nor its name in ``The user's
    sister's dog is named Pip.`` The ancestor phrase can therefore keep its own
    source runs, while the descendant and the name remain forbidden. Match both entire
    canonical statements: ordinary possession/existence forgets, ancestor naming and
    predicates outside the matched nominal phrases receive no exemption. Each
    description has one to three ASCII content words and no negation; the terminal
    name has only capitalized ASCII name words.
    """
    head = re.fullmatch(r"The user has (?:a|an) ([a-z]+(?: [a-z]+){0,2})\.", survivor)
    if (head is None or not all(support_words(word) for word in head[1].split())
            or not {"no", "not", "never"}.isdisjoint(head[1].split())):
        return None
    named = re.fullmatch(
        rf"The user's {re.escape(head[1])}'s ([a-z]+(?: [a-z]+){{0,2}}) "
        r"is named ([A-Z][A-Za-z]*(?: [A-Z][A-Za-z]*)*)\.", forgotten,
    )
    if (named is None or not all(support_words(word) for word in named[1].split())
            or not {"no", "not", "never"}.isdisjoint(named[1].split())):
        return None
    return support_words(f"{named[1]} {named[2]}")


def _lead_name(statement: str) -> str | None:
    """The named subject a statement opens with, casefolded: its possessor ("Blair's pet
    is named Pip.") or its subject ("Avery applied to adoption agencies."); None for
    "The user", a sentence-opening function or time word, and any other opening."""
    opening = _opening(statement)
    if opening is not None:
        return None if opening[0] == "the user" else opening[0]
    match = _SUBJECT.match(statement)
    if (match is None or not match.group(1).isascii() or not support_words(match.group(1))
            or match.group(1).split()[0].casefold() in _OPENERS):
        return None
    return match.group(1).casefold()


def _verb_after(statement: str, lead: str) -> str | None:
    """The token after a statement's lead name ``lead``."""
    tokens, size = token_seq(statement), len(token_seq(lead))
    return tokens[size] if len(tokens) > size else None


def _shared_lead(forgotten: str, survivor: str) -> str | None:
    """The named subject both statements are about (``_lead_name``), unless the survivor
    may restate a forgotten possession (``possession_entailed``)."""
    lead = _lead_name(forgotten)
    if lead is None or lead != _lead_name(survivor) or possession_entailed(forgotten, survivor):
        return None
    return lead


def _subject(statement: str) -> str | None:
    """The subject a statement opens with: "the user" for "The user ..." (not "The
    user's"), else its named lead (``_lead_name``)."""
    if _opening(statement) is None and [
            token.casefold() for token in token_seq(statement)[:2]] == ["the", "user"]:
        return "the user"
    return _lead_name(statement)


def possession_entailed(forgotten: str, survivor: str) -> bool:
    """Whether a survivor may restate a forgotten possession by one of its things (C26E).

    The forgotten statement says its subject has something: a ``_HAVE`` verb after "The
    user" or a named subject ("The user has pets.", "Blair has pets."). The survivor
    states a possession of the same subject: it opens with "<subject>'s" or goes on with a
    ``_GAIN`` verb ("The user got another cat.", "Blair's cat is named Juno."). Its
    thing may be of the forgotten kind, which no word test sees, so such a survivor keeps
    nothing on the clause path (e) or the dependent trim (D.7.3 #4a), whatever its words,
    as a named subject's survivor already did through ``_shared_lead``. The referent
    keeper (d) applies this guard only with an exact match between the possessed
    object and its primary entity's stored broad type.

    Args:
        forgotten: A forgotten version's statement.
        survivor: A surviving memory's statement.

    Returns:
        True when the survivor may restate the forgotten possession.
    """
    subject = _subject(forgotten)
    if subject is None or _opening(forgotten) is not None or (
            _verb_after(forgotten, subject) not in _HAVE):
        return False
    opening = _opening(survivor)
    if opening is not None:
        return opening[0] == subject
    return _subject(survivor) == subject and _verb_after(survivor, subject) in _GAIN


_WORK = frozenset({"work", "works", "worked", "working"})
"""The forms of "work" whose frame says which aspect of a job a statement gives."""


def _work_aspect(statement: str, subject: str) -> str | None:
    """"role" for "<subject> works as ...", "place" for "<subject> works at ..." or "works
    for ...", else None."""
    if _verb_after(statement, subject) not in _WORK:
        return None
    tokens, size = token_seq(statement), len(token_seq(subject))
    frame = tokens[size + 1].casefold() if len(tokens) > size + 1 else None
    return "role" if frame == "as" else "place" if frame in ("at", "for") else None


def predicate_words(forgotten: str, survivor: str) -> frozenset[str]:
    """``support_words`` of what a forgotten statement says about a survivor's subject.

    When both statements open with the same "<possessor>'s <description>" ("The user's
    brother is allergic to peanuts." and "The user's brother loves chess."), the shared
    description is the survivor's own subject, as a referent keeper's "my" phrase is
    (path d): only the words after it, here "allergic" and "peanut", are the forgotten
    predicate. The description is the longest shared run of at most three words before a
    light word, and must describe the forgotten opening as a whole noun phrase. Otherwise
    every word of the forgotten statement counts. An exact ancestor-existence keeper
    also excludes only its own ancestor phrase from an exact descendant naming:
    ``The user has a sister.`` against ``The user's sister's dog is named Pip.``
    leaves ``dog`` and ``Pip`` forbidden (``_ancestor_naming_words``, C26). This
    exemption changes neither ordinary possession nor the ancestor's own naming.

    When both statements are about the same named subject, as possessor or subject
    (``_lead_name``: "Blair's other cat is named Juno." and "Blair's pet is named
    Pip."; "Avery applied to adoption agencies." and "Avery has not adopted a
    child yet."), that name is the survivor's own subject, as "the user" is a light
    word for the user's own facts: it is not a forgotten word, and every other word of
    the forgotten statement still is (C26 (a), (c)). Not when the forgotten statement is
    a possession and the survivor states one of the same subject (``_shared_lead``).

    When the same subject works in both and one statement gives the job's role ("works
    as an engineer") and the other its place ("works at Acme Corp", "works for Acme
    Corp"), the shared "work" is the survivor's own verb (C26 (e)): a role and a place
    entail nothing of each other that their words do not show. Any other frame ("works
    with children", "works in France") keeps "work" forbidden, because a role or a place
    can entail it ("a pediatrician", "the mayor of Paris").

    Args:
        forgotten: A forgotten version's statement.
        survivor: A surviving memory's statement.

    Returns:
        The words a survivor of ``forgotten`` must not show or keep readable.
    """
    ancestor = _ancestor_naming_words(forgotten, survivor)
    if ancestor is not None:
        return ancestor
    lead = _shared_lead(forgotten, survivor)
    words = support_words(forgotten) - (support_words(lead) if lead is not None else frozenset())
    subject = _subject(forgotten)
    if subject is not None and subject == _subject(survivor) and {
            _work_aspect(forgotten, subject), _work_aspect(survivor, subject)} == {"role", "place"}:
        words -= support_words("work")
    head, stated = _opening(survivor), _opening(forgotten)
    if head is None or stated is None or head[0] != stated[0]:
        return words
    description: list[str] = []
    for mine, theirs in zip(head[1], stated[1], strict=False):
        if (mine != theirs or not support_words(mine)
                or len(description) == _REFERENT_WORDS):
            break
        description.append(mine)
    if not description or not _describes(stated[1], description):
        return words
    return support_words(" ".join(stated[1][len(description):]))


def forgotten_keywords(
    statement: str, keywords: Sequence[str], forgotten: Sequence[str],
    quotes: Sequence[str] = (),
) -> tuple[str, ...]:
    """The keywords a version drops for forgotten statements (D.7.3 #4b, KW-1n; C.12 KW-1b).

    A keyword is Search's and nomination's term for the memory, never shown to a reader,
    but a query for the forgotten word would still find the memory through it. A keyword
    is dropped when it carries a word of what a forgotten statement says about the
    version's subject (``predicate_words``) that nothing returned for the version shows:
    neither its statement nor one of its readable ``quotes``. A keyword whose word the
    version's own content already shows reveals nothing new and stays. "Maya" leaves the
    keywords of "The user has a sister." after "A person is named Maya." is forgotten
    (c14b FP03), "birthday" those of the party plan after "The user's birthday is March
    3." is (NM18).

    Args:
        statement: The version's statement.
        keywords: Its keywords, in stored order.
        forgotten: Forgotten statements: the forget's tombstones at P2, the scope's at Add.
        quotes: The text of the version's readable quotes.

    Returns:
        The dropped keywords, in stored order; empty when it keeps them all.
    """
    words = frozenset[str]().union(*(predicate_words(text, statement) for text in forgotten))
    hidden = words - support_words(statement).union(*(support_words(text) for text in quotes))
    return tuple(keyword for keyword in keywords if support_words(keyword) & hidden)


CLAUSE_COSINE = 0.85
"""A survivor whose statement vector is this close to a forgotten version's keeps nothing
on path (e): a restatement in other words, which no word test sees. A 59-text
measurement put every restatement at cosine 0.878 or above; the survivors path (e) keeps
in the saved gates score 0.22-0.48 against their forgotten statements, FPD's chess fact
0.64."""


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    """The cosine of two vectors; 1.0, the refusing value, when one is missing or zero."""
    if not left or len(left) != len(right):
        return 1.0
    norm = math.sqrt(math.fsum(value * value for value in left)
                     * math.fsum(value * value for value in right))
    return 1.0 if norm == 0 else math.fsum(a * b for a, b in zip(left, right, strict=True)) / norm


def _meets(span: QuoteSpanV1, other: QuoteSpanV1) -> bool:
    return (span.evidence_id == other.evidence_id and span.byte_start < other.byte_end
            and other.byte_start < span.byte_end)


def clause_spans(
    version: ForgottenCitation, keeper: KeeperGroup, forgotten: Set[tuple[str, str]],
) -> tuple[QuoteSpanV1, ...]:
    """The parts of a survivor's quotes a clause keeper keeps readable (D.3 P1 #6 (e)).

    One quote can state two facts: "I volunteer at the animal shelter on weekends and I
    run in the mornings" can support both "The user volunteers at the animal shelter on
    weekends." and "The user runs in the mornings." (FP16). Forgetting the first then
    withholds the whole sentence and the second loses its only support. The survivor
    keeps readable the parts of its quotes that carry only its own words, "I run in the
    mornings", and the rest of the forgotten quote is withheld. The test is on words and
    errs toward withholding; a part counts only when all of these hold:

    * the survivor's head has no sensitivity class (the caller also requires that no
      forgotten version has one), and its statement vector, in the version's embedding
      space, is below cosine ``CLAUSE_COSINE`` to the version's: a restatement in other
      words ("never married" for "single") keeps nothing;
    * the quote meets a span the version cites, and is ASCII: a quote in another script
      is never cut;
    * the part is a run of the quote's tokens each of which is a word of the survivor's
      statement or time expression or a light word (``word_runs``): it says nothing the
      survivor does not;
    * no word of the part, and no word of the survivor's statement or time expression,
      is a word of what the version states about the survivor's subject
      (``predicate_words``), so no forgotten word stays readable; a keyword carrying
      one is dropped when the survivor is requoted (D.7.3 #4a, NM18), so none is shown;
    * the part has no sensitivity class (``classify``) and contains no name a forgotten
      version gives;
    * the parts and the survivor's quotes outside the version's spans, with the names of
      their speakers (a quote's readable provenance), hold every word of its statement and
      time expression, so the survivor can be requoted on them (D.7.3 #4a); otherwise it
      keeps nothing.

    A part never carries a word of the forgotten statement. A survivor that restated the
    forgotten fact in other words below the cosine bound would keep its own words, as UD3
    keeps a duplicate whose support lies in readable bytes: it is a separate memory the
    judge did not merge.

    Args:
        version: One non-erased forgotten version, with its groups and statement.
        keeper: One readable group of a surviving memory, with its words.
        forgotten: Every ``(entity_id, name_norm)`` pair a forgotten version gives.

    Returns:
        The kept parts as spans inside the keeper's quotes; empty when it keeps nothing.
    """
    if (keeper.sensitive or not version.statement or not keeper.statement
            or not keeper.stated or len(keeper.quoted) != len(keeper.spans)
            or not keeper.space or keeper.space != version.space
            or _cosine(keeper.vector, version.vector) >= CLAUSE_COSINE):
        return ()
    cited = [span for group in version.groups for span in group]
    said = predicate_words(version.statement, keeper.statement)
    if keeper.stated & said or possession_entailed(version.statement, keeper.statement):
        return ()
    kept: list[QuoteSpanV1] = []
    grounded: set[str] = set()
    speakers = (keeper.speakers if len(keeper.speakers) == len(keeper.spans)
                else (frozenset[str](),) * len(keeper.spans))
    for span, text, speaker in zip(keeper.spans, keeper.quoted, speakers, strict=True):
        if not any(_meets(span, other) for other in cited):
            grounded |= support_words(text) | speaker
            continue
        if not text.isascii():
            return ()
        for start, end in word_runs(text, keeper.stated):
            part = text[start:end]
            if (support_words(part) & said or classify([part])
                    or any(token_contained(name, part) for _, name in forgotten)):
                continue
            kept.append(QuoteSpanV1(evidence_id=span.evidence_id,
                                    byte_start=span.byte_start + start,
                                    byte_end=span.byte_start + end))
            grounded |= support_words(part) | speaker
    if not kept or not keeper.stated <= grounded:
        return ()
    return tuple(kept)


_Range = tuple[int, int]


def _subtract(span: _Range, kept: Sequence[_Range]) -> list[_Range]:
    """The non-empty pieces of ``span`` outside every range of ``kept``."""
    pieces = [span]
    for start, end in kept:
        pieces = [piece for low, high in pieces for piece in (
            ((low, high),) if end <= low or high <= start
            else tuple(part for part in ((low, start), (end, high)) if part[0] < part[1])
        )]
    return pieces


_DESCRIPTOR_STATEMENT = re.compile(
    r"(?P<name>[^,.;:]+?) (?:is|was) (?P<descriptor>[^,;:]+)\.",
)
"""A non-naming head that describes its named primary entity (detail path c)."""


def _reference_detail_names(
    version: ForgottenCitation, keeper: KeeperGroup,
) -> set[tuple[str, str]]:
    """Apply the detail guard to an exact description of a readably named primary.

    After naming/restatement decomposition, ``Pip is the user's sister.`` retains
    ``My sister Pip`` for a different detail just as the former naming descriptor did.
    Its known name is used only by the detail guard, never by the own-name path.
    """
    if keeper.names or not keeper.anchors or keeper.anchors[0] in ("", "user"):
        return set()
    match = _DESCRIPTOR_STATEMENT.fullmatch(keeper.statement)
    if match is None:
        return set()
    reference = (keeper.anchors[0], name_norm(match["name"]))
    if reference not in keeper.reference_names:
        return set()
    return detail_names(version, replace(keeper, names=frozenset({reference})))


def _own_names(version: ForgottenCitation, keeper: KeeperGroup,
               sensitive_forget: bool) -> set[tuple[str, str]]:
    """The names under which ``keeper`` keeps its quotes for ``version`` (D.3 P1 #6).

    A non-sensitive naming memory of the version's primary entity (never the user)
    keeps any quote, even when a forgotten version is sensitive (path a).
    A naming memory of another non-user entity e keeps
    only as a bare-name keeper: its head anchors e alone and names it by type
    alone with an allowlisted neutral kind. Only exact name subspans of the group's
    quotes may survive. This path (b) keeps nothing when any forgotten version
    is sensitive. A sensitive keeper keeps nothing on either path.
    """
    if keeper.sensitive:
        return set()
    if version.entity_id not in ("", "user"):
        own = {pair for pair in keeper.names if pair[0] == version.entity_id}
        if own:
            return own
    if sensitive_forget:
        return set()
    if keeper.bare not in (None, "", "user"):
        return {pair for pair in keeper.names if pair[0] == keeper.bare}
    return detail_names(version, keeper) or _reference_detail_names(version, keeper)


def plan_stating_spans(
    cited: Sequence[ForgottenCitation], keepers: Sequence[KeeperGroup], *,
    withdrawn: Set[tuple[str, str]] = frozenset(),
) -> tuple[QuoteSpanV1, ...]:
    """The stating spans S a forget withholds (D.3 P1 #6; UD3 as amended by O11).

    A forgotten version's quote can carry the phrase that introduces and names
    an entity the version is about (B.13 #5: the establishing phrase of a
    ``new`` referent): "My partner Alex" in "My partner Alex works as a nurse".
    When a surviving naming memory of that entity quotes the phrase as its own
    readable support, the forget withholds the rest and leaves the phrase
    readable for that memory, so the entity keeps its name and the relearn
    guard keeps finding it. Everything else a forgotten version cites is
    withheld, as before:

    * a keeper counts for a version whose primary entity (never the user) it
      names, or, as a bare-name keeper, when its head anchors one non-user entity e
      alone and names it by exactly one allowlisted neutral kind. The bare keeper
      retains only the name itself within its quotes (``bare_name_spans``): "Rowan" in
      "I'm married to Rowan" stays readable for "A person is named Rowan.",
      while a keeper that also anchors or names the user or another anchor
      ("A person who is Avery's spouse is named Rowan.") keeps nothing. A broad
      quote ("married to Rowan") can keep only "Rowan". The bare-name entity need
      not be anchored by the forgotten version: a user-only profession fact
      can quote an employer's name without claiming anything about that entity;
    * a naming memory that also anchors its entity's owner or the user keeps its
      quote as a detail keeper (path c, ``detail_names``) when the version states
      a different detail of an entity it anchors ("The user bathes Pip, the user's
      tortoise, every morning." keeps "a tortoise named Pip" for "The user's
      tortoise is named Pip."), never when the version could restate it ("The
      user has a tortoise."); a non-naming head of the exact form "<known primary
      name> is|was <descriptor>." uses only that same detail guard, with reference
      names given by surviving P-NAME-readable naming memories;
    * any other surviving memory keeps only its referent quote (path d,
      ``referent_spans``): "My brother" in "My brother is allergic to peanuts." stays
      readable for "The user's brother loves chess." (group "My brother", "He loves
      chess."), never a quote that carries a word of the forgotten predicate;
    * a surviving memory that keeps nothing on the paths above keeps, as a clause keeper
      (path e, ``clause_spans``), the runs of its quotes that carry only its own words
      and none of the forgotten predicate: "I run in the mornings" in "I volunteer at
      the animal shelter on weekends and I run in the mornings" for "The user runs in
      the mornings." (FP16);
    * a sensitive keeper keeps nothing on any path; paths (b), (c), (e) and (f) keep nothing
      when any forgotten version has a sensitivity class. Paths (a) and (d) still
      permit a non-sensitive keeper of a sensitive version;
    * never with a name some forgotten version gives that entity (it would
      restate the forgotten naming);
    * a head a forgotten correction reworded (C.9.4a; path f, ``reworded``) keeps its
      own groups for every version that withdraws no name (``withdrawn``: a name a
      forgotten version gives that no surviving memory gives), unless the head or any
      forgotten version is sensitive: its wording states neither value and its quotes
      are its own support, so forgetting the correction leaves it readable (OPR-2),
      while a forgotten name and the correction's other words stay withheld;
    * a group whose every byte the version's keepers quote is withheld whole,
      so every forgotten group keeps a withheld remainder and none stays usable;
    * a keeper any of whose quotes still meets a withheld range is dropped (it
      stays unreadable, D.2), a clause keeper when one of its runs or of its quotes
      outside every cited span does, and the plan is recomputed until none is left.

    Bytes left readable are bytes a surviving, readable naming memory of the
    version's primary entity quotes as its own judged support, a name a
    bare-name keeper quotes, a detail keeper's quote, which carries no word
    of the forgotten detail, a referent keeper's "my" phrase or a clause keeper's
    run, which carry no word of the forgotten predicate, or a reworded head's own
    quotes, which carry no withdrawn name; every other cited byte is withheld, and
    every forgotten group keeps a withheld byte.

    Args:
        cited: Every non-erased version of the forgotten Items, with its groups.
        keepers: The readable groups of the surviving memories whose heads cite an
            Evidence row the forgotten versions cite, and of the heads a forgotten
            correction reworded.
        withdrawn: The ``(entity_id, name_norm)`` pairs the forget withdraws.

    Returns:
        Unique spans in (evidence, start, end) order; a span no keeper overlaps
        is unchanged, an overlapped one is cut into the pieces outside the kept
        quotes, and a fully kept one is absent.
    """
    forgotten_names = {pair for version in cited for pair in version.names}
    sensitive_forget = any(version.sensitive for version in cited)
    cited_spans = [span for version in cited for group in version.groups for span in group]
    live = set(range(len(keepers)))
    while True:
        withheld: dict[str, set[_Range]] = {}
        parts: dict[int, list[QuoteSpanV1]] = {}
        whole_keepers: set[int] = set()
        for version in cited:
            kept: dict[str, list[_Range]] = {}
            for index in sorted(live):
                keeper = keepers[index]
                if keeper.reworded:
                    spans: Sequence[QuoteSpanV1] = keeper.spans if (
                        not keeper.sensitive and not sensitive_forget
                        and version.names.isdisjoint(withdrawn)) else ()
                    if spans:
                        whole_keepers.add(index)
                else:
                    own = _own_names(version, keeper, sensitive_forget)
                    if own and own.isdisjoint(forgotten_names) and keeper.bare is not None:
                        spans = bare_name_spans(keeper)
                        parts.setdefault(index, []).extend(spans)
                    else:
                        spans = (keeper.spans if own and own.isdisjoint(forgotten_names)
                                 else referent_spans(version, keeper, forgotten_names))
                        if spans:
                            whole_keepers.add(index)
                        elif not sensitive_forget:
                            spans = clause_spans(version, keeper, forgotten_names)
                            parts.setdefault(index, []).extend(spans)
                for span in spans:
                    kept.setdefault(span.evidence_id, []).append(
                        (span.byte_start, span.byte_end))
            for group in version.groups:
                whole = all(not _subtract((span.byte_start, span.byte_end),
                                          kept.get(span.evidence_id, ())) for span in group)
                for span in group:
                    withheld.setdefault(span.evidence_id, set()).update(_subtract(
                        (span.byte_start, span.byte_end),
                        () if whole else kept.get(span.evidence_id, ()),
                    ))

        # A clause keeper needs its parts and its quotes outside every cited span; any
        # other keeper needs every quote of its group.
        needed = {index: keepers[index].spans if index in whole_keepers or not parts.get(index)
                  else (*parts[index], *(span for span in keepers[index].spans
                                         if not any(_meets(span, other) for other in cited_spans)))
                  for index in live}
        dropped = {
            index for index in live
            if any(start < span.byte_end and span.byte_start < end
                   for span in needed[index]
                   for start, end in withheld.get(span.evidence_id, ()))
        }
        if not dropped:
            return tuple(
                QuoteSpanV1(evidence_id=evidence_id, byte_start=start, byte_end=end)
                for evidence_id in sorted(withheld)
                for start, end in sorted(withheld[evidence_id])
            )
        live -= dropped
