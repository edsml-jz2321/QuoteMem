"""Settlement of an owner's admitted positions before review (B.8).

After every unit of the owner is admitted (the C.2 barrier), settlement reads
each admitted position in canonical order ``(σ, position_key)`` and settles
what needs no review, using indexed lookups only:

1. a tombstone with the position's resolved primary and statement digest
   rejects it ``denied_by_forget`` and withholds its located spans;
2. a P-MEM head with the same digest, primary and about set that already cites
   every located span leaves it unchanged ``restatement_satisfied``, when it
   asks for no revision; otherwise (2b) an active Item with that digest,
   primary and about set whose head fails P-VER, a sibling a forget left
   without a usable group (D.2), gains the position's groups as new support
   (``merged_support``, a ``merge``): independent new support makes it readable
   again, and only the new groups render, so nothing withheld does. Its head is
   never shown to a provider; a later position restoring the same Item in the
   owner is that position's duplicate;
3. a position equal to an earlier one (the B.7.3 duplicate criteria across the
   owner's units, or an in-reply merge) is that one's duplicate: its groups
   join the representative's before review when they fit, else it is
   ``support_budget_exceeded`` and mirrors nothing. An in-reply duplicate is
   settled right after the position it merged into, whatever its own σ, and
   joins what that position joined, so a representative is never itself a
   duplicate.

It also raises the concerns of #4. The result is pure, so replay recomputes it
exactly; the settlement transaction (C.12) persists it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from aml.core.add_read_view import AddReadView, MemoryRecord, Sigma, span_sigma
from aml.core.entities import (
    USER,
    EntityChoiceV1,
    MemoryLifecycle,
    NewEntityChoiceV1,
    PositionKey,
    QuoteSpanV1,
)
from aml.core.errors import AMLError
from aml.core.extraction_records import DigestSettlementV1
from aml.core.identity import canonical_json
from aml.core.proposal import Concern
from aml.core.sensitivity import SensitivityClass
from aml.core.statement import (
    MAX_MEMORY_SPANS,
    MAX_REFERENT_CANDIDATES,
    MAX_SUPPORT_GROUPS,
    label_speaker_key,
    mint_key,
    name_norm,
)
from aml.ingestion.extraction_admission import AdmittedFact, SpanKey, span_key


@dataclass(frozen=True, slots=True)
class UnitFacts:
    """The admitted positions of one unit of the owner.

    Attributes:
        unit_no (int): The unit.
        facts (tuple[AdmittedFact, ...]): Its admitted positions (B.7).
    """

    unit_no: int
    facts: tuple[AdmittedFact, ...]


@dataclass(frozen=True, slots=True)
class Settlement:
    """The settlement of one owner (B.8).

    Attributes:
        order (tuple[PositionKey, ...]): Every admitted position in canonical
            order ``(σ, position_key)``.
        records (tuple[DigestSettlementV1, ...]): The positions #1 to #3
            settle, in canonical order.
        withheld (tuple[QuoteSpanV1, ...]): The located spans of every #1
            rejection, which the settlement transaction withholds.
        groups (dict[PositionKey, tuple[tuple[QuoteSpanV1, ...], ...]]): The
            groups of each representative that duplicates joined: its own, then
            theirs in canonical order.
        sensitivity (dict[PositionKey, tuple[SensitivityClass, ...]]): Each
            such representative's classes, united with its duplicates'.
        concerns (dict[PositionKey, frozenset[Concern]]): The #4 concerns.
        collisions (dict[PositionKey, tuple[str, ...]]): The entities whose live
            name a position's new entity shares (``new_name_matches_existing``).
        unresolved (tuple[PositionKey, ...]): Unsettled positions with an
            unresolved primary: pending ``identity_unresolved`` candidates the
            settlement transaction writes (B.7.3).
        primaries (dict[PositionKey, str | None]): Each position's primary
            resolved at the settlement pin; None while it has no entity row.
        lifecycles (dict[PositionKey, MemoryLifecycle]): The lifecycle of the
            head that holds each #2 restatement, which the settlement
            transaction compares with the head.
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
            raise AMLError("binding_mismatch", "An admitted span cites Evidence outside the scope")
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
    """The B.7.3 duplicate identity: everything but the support (B.8 #3)."""
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
    """B.8 #2: the head states the same fact of the same entities and cites every span."""
    snapshot = head.head.snapshot
    return (
        snapshot.entity_id == primary and set(snapshot.about_entity_ids) == about
        and _keys(fact.spans) <= _keys(snapshot.quote_spans)
    )


def settle(units: Sequence[UnitFacts], view: AddReadView, *, owner_run_id: str) -> Settlement:
    """Settle an owner's admitted positions at the settlement pin (B.8).

    Args:
        units (Sequence[UnitFacts]): The admitted positions of every unit.
        view (AddReadView): The read view after the barrier. Settlement uses
            its three keyed reads: ``tombstone(entity_id, statement_digest)``,
            ``heads_by_digest(statement_digest)`` and ``entities_by_name(name_norm)``.
            The view owns readability and serves tombstones from the owner's index.
        owner_run_id (str): The Add owner run.

    Returns:
        Settlement: The settled positions, the spans to withhold, the joined
            groups, the concerns and the unresolved primaries.

    Raises:
        AMLError: ``binding_mismatch`` when a span cites Evidence outside the
            scope.
    """
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
                # The merged support must fit (A0.7); a sum bounds the union.
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
    )


def _concerns(
    units: Sequence[UnitFacts], view: AddReadView, owner_run_id: str,
) -> tuple[dict[PositionKey, frozenset[Concern]], dict[PositionKey, tuple[str, ...]]]:
    """B.8 #4: a new entity's name equals a live name; a named new entity no naming fact names.

    A participant has no entity name (PD:P4), so a new entity's name also matches the
    participant its label or a speaker name of its readable Evidence keys (B.4.2).
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
