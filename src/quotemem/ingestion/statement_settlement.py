"""Settlement of an owner's admitted positions before review.

After every unit of the owner is admitted (the barrier), settlement reads
each admitted position in canonical order ``(σ, position_key)`` and settles
what needs no review, using indexed lookups only:

1. a tombstone with the position's resolved primary and statement digest
   rejects it ``denied_by_forget`` and withholds its located spans, unless the owner's
   policy learns again what the user states after a forget (the ``agent``
   mode) and one of the position's support groups is the user's own statement, told after
   that tombstone's forget, whose words restate it (``core/relearn.py``);
2. a ``p_mem`` head with the same digest, primary and about set that already cites
   every located span leaves it unchanged ``restatement_satisfied``, when it
   asks for no revision; otherwise (2b) an active memory with that digest,
   primary and about set whose head fails ``p_ver``, a sibling a forget left
   without a usable group, gains the position's groups as new support
   (``merged_support``, a ``merge``): independent new support makes it readable
   again, and only the new groups render, so nothing withheld does. Its head is
   never shown to a provider; a later position restoring the same memory in the
   owner is that position's duplicate;
3. a position equal to an earlier one (the duplicate criteria across the
   owner's units, or an in-reply merge) is that one's duplicate: its groups
   join the representative's before review when they fit, else it is
   ``support_budget_exceeded`` and mirrors nothing. An in-reply duplicate is
   settled right after the position it merged into, whatever its own σ, and
   joins what that position joined, so a representative is never itself a
   duplicate.

It also raises each position's review concerns. The result is pure, so replay recomputes it
exactly; the settlement transaction persists it.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence, Set
from dataclasses import dataclass, field, replace

from quotemem.core.add_read_view import (
    AddReadView,
    MemoryRecord,
    Sigma,
    TombstoneRecord,
    span_sigma,
)
from quotemem.core.entities import (
    USER,
    EntityChoiceV1,
    ExistingChoiceV1,
    MemoryLifecycle,
    NewEntityChoiceV1,
    PositionKey,
    QuoteSpanV1,
)
from quotemem.core.errors import QuoteMemError
from quotemem.core.extraction_records import DigestSettlementV1
from quotemem.core.forget_match import name_forms, name_request, naming_tie
from quotemem.core.forgotten_names import (
    NAME_LINK,
    NEUTRAL,
    ForgottenName,
    Kind,
    bare,
    cut,
    forgotten_names,
    kind_of,
    names_of,
    occurrences,
    other_kind,
    reword,
    shows,
    tie_of,
    untie,
    without_name,
)
from quotemem.core.identity import canonical_json
from quotemem.core.proposal import Concern
from quotemem.core.recall_agent import sentences
from quotemem.core.relearn import Forgotten, after, group_told, relearned
from quotemem.core.sensitivity import SensitivityClass
from quotemem.core.statement import (
    MAX_MEMORY_SPANS,
    MAX_REFERENT_CANDIDATES,
    MAX_SUPPORT_GROUPS,
    label_speaker_key,
    mint_key,
    name_norm,
    own_token,
    statement_digest,
    token_seq,
)
from quotemem.ingestion.extraction_admission import AdmittedFact, SpanKey, span_key
from quotemem.ingestion.relearn import asserts


@dataclass(frozen=True, slots=True)
class UnitFacts:
    """The admitted positions of one unit of the owner.

    Attributes:
        unit_no (int): The unit.
        facts (tuple[AdmittedFact, ...]): Its admitted positions.
    """

    unit_no: int
    facts: tuple[AdmittedFact, ...]


@dataclass(frozen=True, slots=True)
class Settlement:
    """The settlement of one owner.

    Attributes:
        order (tuple[PositionKey, ...]): Every admitted position in canonical
            order ``(σ, position_key)``.
        records (tuple[DigestSettlementV1, ...]): The positions settlement
            settle, in canonical order.
        withheld (tuple[QuoteSpanV1, ...]): The located spans of every
            rejection, which the settlement transaction withholds.
        groups (dict[PositionKey, tuple[tuple[QuoteSpanV1, ...], ...]]): The
            groups of each representative that duplicates joined: its own, then
            theirs in canonical order.
        sensitivity (dict[PositionKey, tuple[SensitivityClass, ...]]): Each
            such representative's classes, united with its duplicates'.
        concerns (dict[PositionKey, frozenset[Concern]]): The concerns.
        collisions (dict[PositionKey, tuple[str, ...]]): The entities whose live
            name a position's new entity shares (``new_name_matches_existing``).
        unresolved (tuple[PositionKey, ...]): Unsettled positions with an
            unresolved primary: pending ``identity_unresolved`` candidates the
            settlement transaction writes.
        primaries (dict[PositionKey, str | None]): Each position's primary
            resolved at the settlement pin; None while it has no entity row.
        lifecycles (dict[PositionKey, MemoryLifecycle]): The lifecycle of the
            head that holds each restatement, which the settlement
            transaction compares with the head.
        facts (dict[PositionKey, AdmittedFact]): The positions settled without a
            forgotten name (`_without_forgotten_names`), as their proposals state them.
    """

    order: tuple[PositionKey, ...]
    records: tuple[DigestSettlementV1, ...]
    withheld: tuple[QuoteSpanV1, ...]
    groups: dict[PositionKey, tuple[tuple[QuoteSpanV1, ...], ...]]
    sensitivity: dict[PositionKey, tuple[SensitivityClass, ...]]
    concerns: dict[PositionKey, frozenset[Concern]]
    collisions: dict[PositionKey, tuple[str, ...]]
    unresolved: tuple[PositionKey, ...]
    primaries: dict[PositionKey, str | None]
    lifecycles: dict[PositionKey, MemoryLifecycle]
    facts: dict[PositionKey, AdmittedFact] = field(default_factory=dict[PositionKey, AdmittedFact])


@dataclass(frozen=True, slots=True)
class _Position:
    key: PositionKey
    unit_no: int
    fact: AdmittedFact
    sigma: Sigma


def _sigma(fact: AdmittedFact, view: AddReadView) -> Sigma:
    sigmas: list[Sigma] = []
    for span in fact.spans:
        evidence = view.evidence(span.evidence_id)
        if evidence is None:
            raise QuoteMemError("binding_mismatch",
                                "An admitted span cites Evidence outside the scope")
        sigmas.append(span_sigma(evidence, span.byte_start))
    return max(sigmas)


def anchor_entity(
    choice: EntityChoiceV1, unit_no: int, owner_run_id: str, view: AddReadView,
) -> str | None:
    """The entity an anchor resolves to at the pin, or None while it has no row.

    Settlement resolves each position's primary with it (``Settlement.primaries``).
    """
    if choice.kind == "user":
        return USER
    if choice.kind == "existing":
        return choice.entity_id
    if choice.kind == "participant":
        if choice.entity_id is not None:
            return choice.entity_id
        record = view.participant(choice.participant_key)
        return None if record is None else record.entity_id
    if choice.kind == "new":
        record = view.minted(mint_key(owner_run_id, unit_no, choice.local_key))
        return None if record is None else record.entity_id
    return None


def _anchor_identity(choice: EntityChoiceV1, unit_no: int, owner_run_id: str) -> str:
    """An anchor's identity for the duplicate criteria across units."""
    if choice.kind == "new":
        return canonical_json(["new", mint_key(owner_run_id, unit_no, choice.local_key)])
    if choice.kind == "participant":
        return canonical_json(["participant", choice.participant_key])
    if choice.kind == "existing":
        return canonical_json(["entity", choice.entity_id])
    if choice.kind == "user":
        return canonical_json(["entity", USER])
    return canonical_json(["unresolved", unit_no, choice.model_dump(mode="json")])


def _identity(position: _Position, owner_run_id: str) -> str:
    """The duplicate identity: everything but the support."""
    fact, unit_no = position.fact, position.unit_no
    return canonical_json([
        fact.statement_digest, fact.kind,
        _anchor_identity(fact.anchors[0], unit_no, owner_run_id),
        sorted(_anchor_identity(choice, unit_no, owner_run_id) for choice in fact.anchors[1:]),
        sorted((memory.memory_id, memory.revision_id) for memory in fact.revises),
        fact.corrects_text,
    ])


def _keys(spans: Sequence[QuoteSpanV1]) -> frozenset[SpanKey]:
    return frozenset(span_key(span) for span in spans)


def _holds(head: MemoryRecord, fact: AdmittedFact, primary: str, about: set[str]) -> bool:
    """The head states the same fact of the same entities and cites every span."""
    snapshot = head.head.snapshot
    return (
        snapshot.entity_id == primary and set(snapshot.about_entity_ids) == about
        and _keys(fact.spans) <= _keys(snapshot.quote_spans)
    )


def settle(units: Sequence[UnitFacts], view: AddReadView, *, owner_run_id: str,
           relearn: bool = False) -> Settlement:
    """Settle an owner's admitted positions at the settlement pin.

    Args:
        units (Sequence[UnitFacts]): The admitted positions of every unit.
        view (AddReadView): The read view after the barrier. Settlement uses
            its three keyed reads: ``tombstone(entity_id, statement_digest)``,
            ``heads_by_digest(statement_digest)`` and ``entities_by_name(name_norm)``.
            The view owns readability and serves tombstones from the owner's index.
        owner_run_id (str): The Add owner run.
        relearn (bool): The owner's policy learns again what the user states after a
            forget; the same ``agent`` policy settles a position without
            a forgotten name first (`_without_forgotten_names`).

    Returns:
        Settlement: The settled positions, the spans to withhold, the joined
            groups, the concerns and the unresolved primaries.

    Raises:
        QuoteMemError: ``binding_mismatch`` when a span cites Evidence outside the
            scope.
    """
    changed: dict[PositionKey, AdmittedFact] = {}
    refused: dict[PositionKey, str] = {}
    if relearn:
        units, changed, refused = _without_forgotten_names(units, view, owner_run_id)
    positions = sorted((
        _Position((owner_run_id, unit.unit_no, fact.position), unit.unit_no, fact,
                  _sigma(fact, view))
        for unit in units for fact in unit.facts
    ), key=lambda position: (position.sigma, position.key))
    pin = view.pin

    def settlement(key: PositionKey, code: str, **fields: object) -> DigestSettlementV1:
        return DigestSettlementV1.model_validate({
            "position_key": key, "code": code, "memory_id": None, "revision_id": None,
            "tombstone_id": None, "representative": None,
            "read_view_commit_seq": pin.commit_seq, "tombstone_seq": pin.tombstone_seq,
            **fields,
        })

    settled: dict[PositionKey, DigestSettlementV1] = {}
    withheld: dict[SpanKey, QuoteSpanV1] = {}
    by_identity: dict[str, list[_Position]] = {}
    by_key = {position.key: position for position in positions}
    merged: dict[PositionKey, list[_Position]] = {}
    for position in positions:
        if position.fact.duplicate_of is not None:
            merged.setdefault(
                (owner_run_id, position.unit_no, position.fact.duplicate_of), [],
            ).append(position)
    joined: dict[PositionKey, list[tuple[QuoteSpanV1, ...]]] = {}
    classes: dict[PositionKey, set[SensitivityClass]] = {}
    primaries: dict[PositionKey, str | None] = {}
    lifecycles: dict[PositionKey, MemoryLifecycle] = {}
    restoring: dict[str, PositionKey] = {}

    def place(position: _Position, merged_into: _Position | None) -> None:
        fact, key = position.fact, position.key
        primary = anchor_entity(fact.anchors[0], position.unit_no, owner_run_id, view)
        primaries[key] = primary
        about = {
            anchor_entity(choice, position.unit_no, owner_run_id, view)
            for choice in fact.anchors[1:]
        }
        tombstone = None if primary is None else view.tombstone(primary, fact.statement_digest)
        if key in refused:
            # A naming of a name a forget took from its only bearer (`_without_forgotten_names`).
            settled[key] = settlement(key, "denied_by_forget", tombstone_id=refused[key])
            withheld.update((span_key(span), span) for span in fact.spans)
            return
        if tombstone is not None and relearn and relearned(
                (group_told(((view.evidence(span.evidence_id), span.byte_start, span.byte_end)
                             for span in group), asserts) for group in fact.groups),
                Forgotten(tombstone.statement, tombstone.created_at, tombstone.embedding,
                          tombstone.embedding_space), fact.statement):
            tombstone = None
        if tombstone is not None:
            settled[key] = settlement(key, "denied_by_forget", tombstone_id=tombstone.object_id)
            withheld.update((span_key(span), span) for span in fact.spans)
            return
        if primary is not None and None not in about and not fact.revises and (
            fact.corrects_text is None
        ):
            holder = next((
                head for head in view.heads_by_digest(fact.statement_digest)
                if _holds(head, fact, primary, {entity for entity in about if entity})
            ), None)
            if holder is not None:
                settled[key] = settlement(
                    key, "restatement_satisfied", memory_id=holder.memory_id,
                    revision_id=holder.current_revision_id,
                )
                lifecycles[key] = holder.lifecycle_state
                return
            withheld_head = next((
                head for head in view.withheld_heads(fact.statement_digest)
                if head.entity_id == primary
                and set(head.about_entity_ids) == {entity for entity in about if entity}
                # The merged support must fit; a sum bounds the union.
                and head.support_groups + len(fact.groups) <= MAX_SUPPORT_GROUPS
                and head.quote_spans + len(fact.spans) <= MAX_MEMORY_SPANS
            ), None)
            if withheld_head is not None and merged_into is None:
                first = restoring.get(withheld_head.memory_id)
                if first is not None:
                    settled[key] = settlement(
                        key, "duplicate_in_add", representative=first,
                        memory_id=withheld_head.memory_id,
                        revision_id=withheld_head.current_revision_id,
                    )
                    return
                restoring[withheld_head.memory_id] = key
                settled[key] = settlement(
                    key, "merged_support", memory_id=withheld_head.memory_id,
                    revision_id=withheld_head.current_revision_id,
                )
                lifecycles[key] = "active"
                return
        if merged_into is None:
            identity = _identity(position, owner_run_id)
            spans = _keys(fact.spans)
            representative = next((
                earlier for earlier in by_identity.get(identity, ())
                if fact.kind == "state" or spans & _keys(earlier.fact.spans)
            ), None)
            if representative is None:
                by_identity.setdefault(identity, []).append(position)
                return
        else:
            representative = merged_into
            lead = settled.get(representative.key)
            if lead is not None and lead.code == "duplicate_in_add" and (
                lead.representative is not None
            ):
                # The position it merged into joined an earlier one: so does it.
                representative = by_key[lead.representative]
        lead = settled.get(representative.key)
        if lead is not None:
            # A settled representative is never reviewed: its duplicate mirrors it.
            settled[key] = settlement(
                key, "duplicate_in_add", representative=representative.key,
                memory_id=lead.memory_id, revision_id=lead.revision_id,
            )
            return
        groups = joined.get(representative.key) or list(representative.fact.groups)
        present = {frozenset(span_key(span) for span in group) for group in groups}
        added = [
            group for group in fact.groups
            if frozenset(span_key(span) for span in group) not in present
        ]
        total = {span_key(span) for group in (*groups, *added) for span in group}
        if len(groups) + len(added) > MAX_SUPPORT_GROUPS or len(total) > MAX_MEMORY_SPANS:
            settled[key] = settlement(
                key, "support_budget_exceeded", representative=representative.key,
            )
            return
        joined[representative.key] = [*groups, *added]
        classes.setdefault(
            representative.key, set(representative.fact.sensitivity_classes),
        ).update(fact.sensitivity_classes)
        settled[key] = settlement(key, "duplicate_in_add", representative=representative.key)

    for position in positions:
        if position.fact.duplicate_of is None:
            place(position, None)
            for duplicate in merged.get(position.key, ()):
                place(duplicate, position)
    concerns, collisions = _concerns(units, view, owner_run_id)
    return Settlement(
        order=tuple(position.key for position in positions),
        records=tuple(settled[position.key] for position in positions if position.key in settled),
        withheld=tuple(withheld.values()),
        groups={key: tuple(groups) for key, groups in joined.items()},
        sensitivity={key: tuple(sorted(found)) for key, found in classes.items()},
        concerns=concerns,
        collisions=collisions,
        unresolved=tuple(
            position.key for position in positions
            if position.key not in settled and position.fact.anchors[0].kind == "unresolved"
        ),
        primaries=primaries,
        lifecycles=lifecycles,
        facts=changed,
    )


def _without_forgotten_names(
    units: Sequence[UnitFacts], view: AddReadView, owner_run_id: str,
) -> tuple[list[UnitFacts], dict[PositionKey, AdmittedFact], dict[PositionKey, str]]:
    """The positions of an Add after a forget took a name, with that name kept out of what
    they say about the one that bore it (``core/forgotten_names``), those that changed, and
    those settlement refuses with the tombstone that forgot the name.

    Each name is read on its own (`_NamePlan`), position by position. The one that bore it
    may be gone (the forget erased it); its sort ("the user's cat": an animal) still tells a
    later entity apart. An occurrence of the name belongs to the bearer unless the Add gives
    it to someone clearly else (a person declared as the user's coworker beside a cat) in a
    sentence that names nothing of the bearer's sort and that no position about the bearer
    quotes; when in doubt, it is the bearer's. A Han name, which may also be an ordinary word (a
    drink's name, say), is the bearer's only in a sentence naming its sort or quoted about it.

    A naming of the name about the bearer, or quoting the bearer's occurrence, is refused
    here, so every occurrence of the name the Add gives the bearer is withheld before review
    reads the sentence ("The user's pet is named bob." after "forget my cat's name"); a naming of
    someone else keeps its quote. The user's later words that give the name again keep
    those occurrences readable across the Add. Other positions are cut around the remaining
    occurrences they quote, and a statement showing the name says instead the bearer's tie
    ("The user took the user's cat to the vet yesterday."), or "someone" beside another
    bearer or nothing tying it. A position left without a quote, or one whose statement
    cannot drop the name, is refused too: privacy before the fact. A correction is left as
    it is."""
    names = forgotten_names(
        ((record.object_id, record.entity_id, record.statement)
         for record in view.tombstones()),
        lambda name: (entity.entity_id for entity in view.entities_by_name(name_norm(name))))
    positions = [((owner_run_id, unit.unit_no, fact.position), unit.unit_no, fact)
                 for unit in units for fact in unit.facts]
    names = (*names, *_requested_names(positions, view, owner_run_id,
                                       {name.name.casefold() for name in names}))
    if not names:
        return list(units), {}, {}
    tombstones = {record.object_id: record for record in view.tombstones()}
    changed: dict[PositionKey, AdmittedFact] = {}
    refused: dict[PositionKey, str] = {}
    for name in names:
        plan = _NamePlan.of(name, positions, view, owner_run_id)
        # Namings first: one refused withholds the bearer's occurrences before review, where
        # the judge reads each as whom it names (`name_placeholders`), so the other positions
        # may keep the bearer's tie.
        namings = [item for item in positions if name.name.casefold() in names_of(
            item[2].statement, (naming.name for naming in item[2].naming))]
        tombstone = tombstones[name.tombstone_id]
        plan = plan.without_retaught(
            (changed.get(key, fact) for key, _, fact in namings if key not in refused), tombstone)
        shown_as_tie = False
        for key, _unit_no, fact in (*namings, *(item for item in positions
                                                 if item not in namings)):
            if key in refused:
                continue
            current = changed.get(key, fact)
            found, refusing = plan.settle(key, current, tombstone, shown_as_tie=shown_as_tie)
            if found is not current:
                changed[key] = found
            if refusing:
                refused[key] = name.tombstone_id
                shown_as_tie = shown_as_tie or bool(plan.spans)
    settled = [UnitFacts(unit.unit_no, tuple(
        changed.get((owner_run_id, unit.unit_no, fact.position), fact) for fact in unit.facts))
        for unit in units]
    return settled, changed, refused


def name_placeholders(
    view: AddReadView, denied: Iterable[tuple[str, Sequence[QuoteSpanV1], str | None]],
) -> tuple[tuple[QuoteSpanV1, str], ...]:
    """How the review judge sees each forgotten name occurrence an owner's settlement withheld:
    "[the user's cat]" for "bob", the bearer's tie (`tie_of`, in no form or word of the name),
    or "[someone]" without one.

    ``denied`` holds each position settlement refused: its statement, its quote spans
    (what the refusal withheld) and its tombstone. A span is a name occurrence when its text
    is, in one of its `name_forms`, the name of that tombstone (a naming of a non-user entity) or,
    for a request that asked for a name (`_requested_names`), the name the refused naming
    gives. Every other withheld range, of another forget or of a refused restatement, renders
    as the marker. Pure over the view and the refused positions, so a resumed owner rebuilds
    it from its records."""
    tombstones = {record.object_id: record for record in view.tombstones()}
    found: dict[SpanKey, tuple[QuoteSpanV1, str]] = {}
    for statement, spans, tombstone_id in denied:
        record = None if tombstone_id is None else tombstones.get(tombstone_id)
        if record is None:
            continue
        tied = tie_of(record.statement) if record.entity_id != USER else None
        if tied is None and name_request(record.statement):
            tied = tie_of(statement)
        if tied is None:
            continue
        name, tie = tied
        spellings = {form.casefold() for form in name_forms(name)}
        for span in spans:
            part = view.evidence(span.evidence_id)
            if part is None:
                continue
            data = part.text.encode("utf-8")
            text = data[span.byte_start - part.byte_start:span.byte_end - part.byte_start]
            if text.decode("utf-8").strip().casefold() in spellings:
                found[span_key(span)] = (span, f"[{tie}]")
    return tuple(found[key] for key in sorted(found))


def _requested_names(positions: Sequence[tuple[PositionKey, int, AdmittedFact]],
                     view: AddReadView, owner_run_id: str,
                     known: Set[str]) -> list[ForgottenName]:
    """The names an Add gives an existing entity whose name a forget asked for while no
    tombstone keeps the name (a forget that reworded the naming instead of taking it leaves
    only the request's: "my cat's name"): a naming of the Add ("The user's cat is named bob.")
    about an existing entity, by a tie the request asked about (``forget_match.name_request``,
    the name it gives if any), of a name no entity bears now. Its bearer is that entity, its
    tombstone the request's."""
    requests = [(record, request) for record in view.tombstones() if record.entity_id == USER
                for request in name_request(record.statement)]
    found: dict[str, ForgottenName] = {}
    for _key, unit_no, fact in positions if requests else ():
        tied, described = naming_tie(fact.statement), tie_of(fact.statement)
        entity = anchor_entity(fact.anchors[0], unit_no, owner_run_id, view)
        if tied is None or described is None or entity is None or entity == USER or (
                view.entity(entity) is None):
            continue
        name = tied[1]
        tokens = token_seq(name)
        if name.casefold() in known or name.casefold() in found or not tokens or (
                len(tokens) == 1 and own_token(tokens[0])) or view.entities_by_name(
                name_norm(name)):
            continue
        for record, request in requests:
            if request.ties & tied[0] and (request.given is None
                                           or request.given.casefold() == name.casefold()):
                found[name.casefold()] = ForgottenName(entity, name, described[1],
                                                       record.object_id)
                break
    return list(found.values())


Role = str
"""How an anchor stands to a forgotten name's bearer: ``bearer``, ``other`` (clearly someone
else), ``user`` or ``unknown``."""


def _overlaps(span: QuoteSpanV1, other: QuoteSpanV1) -> bool:
    return span.evidence_id == other.evidence_id and (
        span.byte_start < other.byte_end and other.byte_start < span.byte_end)


@dataclass(frozen=True, slots=True)
class _NamePlan:
    """One forgotten name over one Add (`_without_forgotten_names`).

    Attributes:
        name: The name.
        kind: Its bearer's sort.
        view: The settlement view.
        roles: Each position's anchors' roles.
        about: The positions about the bearer: an anchor of it, or a statement naming its
            sort.
        others: Whether a position of the Add anchors someone clearly else.
        held: The bearer's occurrences of the name in the Add's quoted parts, by part
            (character offsets).
        spans: The same occurrences as byte spans, in the order of the offsets in ``held``.
    """

    name: ForgottenName
    kind: Kind
    view: AddReadView
    owner_run_id: str
    roles: dict[PositionKey, tuple[Role, ...]]
    about: frozenset[PositionKey]
    others: bool
    held: dict[str, list[tuple[int, int]]]
    spans: tuple[QuoteSpanV1, ...]
    role: Callable[[EntityChoiceV1, int], Role]

    @staticmethod
    def of(name: ForgottenName, positions: Sequence[tuple[PositionKey, int, AdmittedFact]],
           view: AddReadView, owner_run_id: str) -> _NamePlan:
        bearer = view.entity(name.entity_id)
        kind = kind_of(name.tie, "" if bearer is None else bearer.entity_type)

        def role(choice: EntityChoiceV1, unit_no: int) -> Role:
            resolved = anchor_entity(choice, unit_no, owner_run_id, view)
            if resolved == USER:
                return "user"
            if resolved == name.entity_id:
                return "bearer"
            if isinstance(choice, NewEntityChoiceV1):
                if other_kind(kind, choice.entity_type, choice.descriptor):
                    return "other"
                named = choice.name is not None and bool(occurrences(name.name, choice.name))
                return "bearer" if named or kind.shown(choice.descriptor) else "unknown"
            record = None if resolved is None else view.entity(resolved)
            return "other" if record is not None and other_kind(
                kind, record.entity_type, "") else "unknown"

        roles = {key: tuple(role(choice, unit_no) for choice in fact.anchors)
                 for key, unit_no, fact in positions}
        about = frozenset(key for key, _unit_no, fact in positions
                          if "bearer" in roles[key] or kind.shown(fact.statement))
        other_only = {key for key, _unit_no, _fact in positions if key not in about and any(
            found == "other" for found in roles[key]) and all(
            found in ("other", "user") for found in roles[key])}
        held: dict[str, list[tuple[int, int]]] = {}
        spans: list[QuoteSpanV1] = []
        for evidence_id in sorted({span.evidence_id for _key, _unit_no, fact in positions
                                   for span in fact.spans}):
            part = view.evidence(evidence_id)
            if part is None:
                raise QuoteMemError("binding_mismatch",
                                    "An admitted span cites Evidence outside the scope")
            text = part.text
            found = occurrences(name.name, text)
            for low, high in sentences(text):
                inside = [(start, end) for start, end in found if low <= start < high]
                located = [(start, end, QuoteSpanV1(
                    evidence_id=evidence_id,
                    byte_start=part.byte_start + len(text[:start].encode("utf-8")),
                    byte_end=part.byte_start + len(text[:end].encode("utf-8"))))
                    for start, end in inside]

                def quoted(span: QuoteSpanV1, keys: Set[PositionKey]) -> bool:
                    return any(key in keys and any(_overlaps(span, own) for own in fact.spans)
                               for key, _unit_no, fact in positions)

                if not located:
                    continue
                mine = kind.shown(text[low:high]) or any(
                    quoted(span, about) for _start, _end, span in located)
                if not mine and not name.han:
                    mine = not all(quoted(span, other_only) for _start, _end, span in located)
                if mine:
                    held.setdefault(evidence_id, []).extend(
                        (start, end) for start, end, _span in located)
                    spans.extend(span for _start, _end, span in located)
        return _NamePlan(name, kind, view, owner_run_id, roles, about, any(
            "other" in found for found in roles.values()), held, tuple(spans), role)

    def without_retaught(
        self, namings: Iterable[AdmittedFact], tombstone: TombstoneRecord,
    ) -> _NamePlan:
        """Keep re-taught name occurrences readable for every fact that quotes them."""
        quotes = [span for fact in namings if fact.corrects_text is None
                  and self._relearned(fact, tombstone) for span in fact.spans]
        if not quotes:
            return self
        held: dict[str, list[tuple[int, int]]] = {}
        spans: list[QuoteSpanV1] = []
        offsets = (offset for part in self.held.values() for offset in part)
        for offset, span in zip(offsets, self.spans, strict=True):
            if not any(own.evidence_id == span.evidence_id
                       and own.byte_start <= span.byte_start and span.byte_end <= own.byte_end
                       for own in quotes):
                held.setdefault(span.evidence_id, []).append(offset)
                spans.append(span)
        return replace(self, held=held, spans=tuple(spans))

    def _roles(self, key: PositionKey, fact: AdmittedFact) -> tuple[Role, ...]:
        """The roles of a position's anchors as they are now (an earlier name may have
        changed them)."""
        found = self.roles[key]
        if len(found) == len(fact.anchors):
            return found
        return tuple(self.role(choice, key[1]) for choice in fact.anchors)

    def settle(self, key: PositionKey, fact: AdmittedFact, tombstone: TombstoneRecord, *,
               shown_as_tie: bool = False) -> tuple[AdmittedFact, bool]:
        """One position without the name, and whether settlement refuses it. Its anchors
        declaring the bearer by the name are the bearer (`_rebound`). With ``shown_as_tie``
        a refused naming withholds the bearer's occurrences before review, where the judge
        reads them as the bearer's tie, so a position keeps the tie for the name."""
        found, refusing = self._settle(key, fact, tombstone, shown_as_tie)
        if refusing:
            return found, True
        anchors = self._rebound(key, found)
        return (found if anchors == found.anchors else replace(found, anchors=anchors)), False

    def _rebound(self, key: PositionKey, fact: AdmittedFact) -> tuple[EntityChoiceV1, ...]:
        """A position's anchors with every new entity that declares the bearer by the name
        (in its name or its descriptor, `_NamePlan.of`'s ``bearer`` role) made the bearer
        itself, or, when the forget erased it, kept without the name: no entity is minted
        under a forgotten name or a descriptor showing it."""
        name = self.name
        bearer = self.view.entity(name.entity_id)
        anchors: list[EntityChoiceV1] = []
        for choice, role in zip(fact.anchors, self._roles(key, fact), strict=True):
            if role == "bearer" and isinstance(choice, NewEntityChoiceV1) and (
                    (choice.name is not None and occurrences(name.name, choice.name))
                    or occurrences(name.name, choice.descriptor)):
                if bearer is not None:
                    choice = ExistingChoiceV1(kind="existing", entity_id=name.entity_id,
                                              via="catalog")
                else:
                    named = choice.name is not None and bool(occurrences(name.name, choice.name))
                    choice = choice.model_copy(update={
                        "name": None if named else choice.name,
                        "descriptor": without_name(choice.descriptor, name)})
            anchors.append(choice)
        return tuple(anchors)

    def _settle(self, key: PositionKey, fact: AdmittedFact, tombstone: TombstoneRecord,
                shown_as_tie: bool) -> tuple[AdmittedFact, bool]:
        name = self.name
        if fact.corrects_text is not None:
            return fact, False
        hits = [span for span in self.spans if any(_overlaps(span, own) for own in fact.spans)]
        shown = shows(fact.statement, name)
        if name.name.casefold() in names_of(fact.statement,
                                             (naming.name for naming in fact.naming)):
            if self._relearned(fact, tombstone) or (key not in self.about and not hits):
                return fact, False
            return self._refused(key, fact, hits)
        if not hits and not shown:
            return fact, False
        groups = tuple(group for group in (tuple(dict.fromkeys(
            piece for span in group for piece in self._cut(span))) for group in fact.groups)
            if group)
        if not groups or len({span_key(span) for group in groups for span in group}) > (
                MAX_MEMORY_SPANS):
            return self._refused(key, fact, hits)
        roles = self._roles(key, fact)
        anchors, naming = fact.anchors, fact.naming
        statement = fact.statement
        untied = key in self.about and bool(hits) and not shown_as_tie and not any(
            self.kind.shown(self._text(piece)) for group in groups for piece in group)
        if untied:
            # Only the forgotten name told whom the sentence means ("I took bob to the vet
            # yesterday") and
            # the review reads it as written: it says someone, and no longer anchors the
            # bearer, which nothing the judge reads ties it to.
            kept = [index for index, role in enumerate(roles) if role != "bearer"]
            if roles[0] == "bearer" or any(entry.anchor_index not in kept for entry in naming):
                return self._refused(key, fact, hits)
            reworded = untie(statement, name, self.kind)
            if reworded is None:
                return self._refused(key, fact, hits)
            statement, anchors = reworded, tuple(anchors[index] for index in kept)
            naming = tuple(replace(entry, anchor_index=kept.index(entry.anchor_index))
                           for entry in naming)
        elif shown and (key in self.about or hits or not (name.han or self.others)):
            tie, read_as_tie = NEUTRAL, False
            if "other" not in roles and key in self.about and shown_as_tie:
                # The judge reads the name as the bearer's tie (`name_placeholders`): the
                # statement says that tie too, never a looser one ("the user's pet", "a pet").
                tie, read_as_tie = name.tie, True
            elif "other" not in roles and key in self.about:
                tie = next((described for choice, role in zip(
                    fact.anchors, roles, strict=True) if role == "bearer"
                    and isinstance(choice, NewEntityChoiceV1)
                    and choice.descriptor.casefold().startswith("the user's ")
                    and (described := bare(choice.descriptor, name.name))), name.tie)
            reworded = reword(statement, name, tie, read_as_tie=read_as_tie)
            if reworded is None:
                return self._refused(key, fact, hits)
            statement = reworded
        elif not hits:
            return fact, False
        return replace(
            fact, statement=statement, statement_digest=statement_digest(statement),
            anchors=anchors, naming=naming, groups=groups, keywords=tuple(
                keyword for keyword in fact.keywords if not occurrences(name.name, keyword)
                and not (untied and self.kind.shown(keyword)))), False

    def _text(self, span: QuoteSpanV1) -> str:
        part = self.view.evidence(span.evidence_id)
        if part is None:
            raise QuoteMemError("binding_mismatch",
                                "An admitted span cites Evidence outside the scope")
        data = part.text.encode("utf-8")
        return data[span.byte_start - part.byte_start:span.byte_end - part.byte_start].decode(
            "utf-8")

    def _cut(self, span: QuoteSpanV1) -> list[QuoteSpanV1]:
        part = self.view.evidence(span.evidence_id)
        if part is None:
            raise QuoteMemError("binding_mismatch",
                                "An admitted span cites Evidence outside the scope")
        return cut(span, part.text, part.byte_start, self.held.get(span.evidence_id, []))[0]

    def _relearned(self, fact: AdmittedFact, tombstone: TombstoneRecord) -> bool:
        """The user's own later words give the name again: they restate the
        forgotten naming and give a name (`NAME_LINK`); a sentence merely mentioning the name
        may sit near the naming's vector, and it relearns nothing. After a request that only
        asked for the name (`_requested_names`), the user's dated words giving a name after it
        teach it again."""
        told = [group_told(((self.view.evidence(span.evidence_id), span.byte_start,
                             span.byte_end) for span in group), asserts)
                for group in fact.groups]
        if name_request(tombstone.statement):
            return any(group is not None and after(group, tombstone.created_at)
                       and NAME_LINK.search(group.text) for group in told)
        return relearned(told, Forgotten(
            tombstone.statement, tombstone.created_at, tombstone.embedding,
            tombstone.embedding_space), fact.statement) and any(
            group is not None and NAME_LINK.search(group.text) for group in told)

    def _refused(self, key: PositionKey, fact: AdmittedFact, hits: Sequence[QuoteSpanV1],
                 ) -> tuple[AdmittedFact, bool]:
        """A position settlement refuses, quoting the bearer's occurrences of the name, so its
        refusal withholds them and nothing else: all of the Add's within the span bound, else
        its own; one quoting none keeps its quotes. One whose primary has no entity yet (a
        declaration) is refused about the bearer."""
        group = self.spans if len(self.spans) <= MAX_MEMORY_SPANS else tuple(hits)
        groups = (tuple(group),) if group else fact.groups
        anchors = self._rebound(key, fact)
        if anchor_entity(anchors[0], key[1], self.owner_run_id, self.view) is None:
            anchors = (ExistingChoiceV1(kind="existing", entity_id=self.name.entity_id,
                                        via="catalog"), *anchors[1:])
        return replace(fact, groups=groups, anchors=anchors), True


def _concerns(
    units: Sequence[UnitFacts], view: AddReadView, owner_run_id: str,
) -> tuple[dict[PositionKey, frozenset[Concern]], dict[PositionKey, tuple[str, ...]]]:
    """A new entity's name equals a live name; a named new entity no naming fact names.

    A participant has no entity name, so a new entity's name also matches the
    participant its label or a speaker name of its readable Evidence keys.
    """
    concerns: dict[PositionKey, frozenset[Concern]] = {}
    collisions: dict[PositionKey, tuple[str, ...]] = {}
    for unit in units:
        named = {
            choice.local_key for fact in unit.facts for naming in fact.naming
            if isinstance(choice := fact.anchors[naming.anchor_index], NewEntityChoiceV1)
        }
        for fact in unit.facts:
            key = (owner_run_id, unit.unit_no, fact.position)
            found: set[Concern] = set()
            matches: set[str] = set()
            for choice in fact.anchors:
                if not isinstance(choice, NewEntityChoiceV1) or choice.name is None:
                    continue
                norm = name_norm(choice.name)
                matches.update(entity.entity_id for entity in view.entities_by_name(norm))
                keys = (
                    label_speaker_key(view.scope_id, norm), *view.speakers().get(norm, ()),
                ) if norm else ()
                matches.update(record.entity_id for key in keys
                               if (record := view.participant(key)) is not None)
                if choice.local_key not in named:
                    found.add("new_entity_unnamed")
            if matches:
                found.add("new_name_matches_existing")
                collisions[key] = tuple(sorted(matches))[:MAX_REFERENT_CANDIDATES]
            if found:
                concerns[key] = frozenset(found)
    return concerns, collisions
