"""Plan a round's descriptor rewordings and prepare their vectors for the commit.

After a round is resolved, each component member that writes a requested correction
(``correction_requested``) may reword the heads its correction outdated: those its
write links ``contradicts``, whose description the judge found
the correction makes untrue; for a correction of a name, every other current head
of the renamed entity that still identifies it by the replaced name; and for a correction
of a kind, every other current head that identifies one of the correction's entities by
the replaced kind right before its name ("Avery's hamster Mochi loves carrots."), which
the judge may relate ``independent`` although it states the replaced kind of that entity.
Each keeps its own fact; only the description changes, so the new version is a rewording,
never a correction of the head. A head is reworded when, at the round's pin:

- it is active and readable, and no other member of the component touches it (fence,
  target, end or link), so the component still gives it at most one new head;
- it shares an anchor with the correction (a renamed entity's head anchors that entity,
  and no other of its anchors bears the replaced name);
- the correction's statement names the replaced and the new value in a closed repairing
  form (or the possessive "has a NEW, not a OLD"), or it is a plain kind
  assertion whose subject names the pet ("Avery's pet Mochi is a hamster.") and the
  heads it contradicts name that pet with exactly one other kind word, which is then the
  replaced value (in one recorded case the extraction left out "not a guinea pig"); and no
  quote of the correction lies in a turn wholly in quotation marks: a relayed correction
  reports, it never rewords;
- the head holds the replaced value once, outside every name it gives
  (`core.descriptor_repair`), and its new wording holds neither value: a kind becomes the
  corrected entity's broad type ("The user's pet is named Mochi.", "The user has a pet.").
  A contradicted kind head whose shared anchors give no usable type uses all correction
  anchors only when they give exactly one usable type; without one it is not reworded.
  A name is dropped after the
  descriptor that precedes it ("The user's cat sleeps on the windowsill all day.") or
  becomes "the" and the entity's broad type where a noun phrase can stand. So the head's
  own quotes still support it, and forgetting or erasing the correction leaves it readable;
- neither a tombstone of its primary nor another current head holds the reworded
  statement, so a forgotten statement is never written back and no copy appears.

A naming head can also state a relationship: "Avery's spouse is named Rowan." beside
"Avery is married to Rowan.", both quoted from "I am married to Rowan". A change ends
only the relationship fact: the judge relates the naming head variously (``update``, void;
``contradicts``; ``independent``), and a head kept current with that description restates the
ended fact, a revival (regression cases open-quit-job and
open-married-single-state). So every write that ends a fact, or revises it in place as a
correction or a retraction, also rewords each current naming head that names one of the
fact's entities by the relationship the fact states, whatever the judge related it: the
head has the fact's anchors, each of its quotes overlaps a quote of the fact (one source's
words gave both), the fact gives no name, holds the head's name and identifies nothing by a
possessive description or a name (it states the relationship with that individual:
"Avery is married to Rowan.", or "Rowan is Avery's spouse." of the named entity;
"Avery walks her dog Rex." states the walks and "Avery has a dog." a count, so both
leave "Avery's dog is named Rex." alone), and no member of the round touches the head.
Such a head, and a naming head a requested correction links ``contradicts`` while it ends
a fact of the same anchors, keeps only its name with the entity's declared type ("A person
is named Rowan."), its own quotes and its memory, so the name stays readable and the ended
relationship is stated nowhere.

The reworded statements are embedded through the owner's statement-embedding journal,
so a resumed owner makes no call for a batch it stored and a replay sends the recorded
request. The plan is a pure function of the round's resolution, the members'
content and the pin, so a resumed owner plans the same rewordings. Nothing here writes a
memory: the component commit writes each rewording in the correction's own transaction.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace

from quotemem.core.add_read_view import AddReadView
from quotemem.core.descriptor_repair import (
    DescriptorChangeV1,
    HeadRepairV1,
    descriptor_change,
    generic_descriptor,
    kind_assertion,
    kind_change,
    named_kinds,
    repaired_keywords,
    repaired_statement,
)
from quotemem.core.entities import MemorySnapshotV2, PositionKey, QuoteSpanV1
from quotemem.core.errors import QuoteMemError
from quotemem.core.resolution import ResolutionPlanV1, WriteV1
from quotemem.core.semantic_embedding import StatementEmbeddingV1
from quotemem.core.statement import (
    canonical_statement,
    name_norm,
    statement_digest,
    token_contained,
    token_seq,
)
from quotemem.ingestion.review_round import LogicalReader
from quotemem.ingestion.speaker_labels import scan_labels
from quotemem.ingestion.statement_embedding import (
    BatchStore,
    OwnerEmbedding,
    embed_texts,
    embedding_texts,
)
from quotemem.providers.contracts import EmbeddingProvider
from quotemem.store.component_commit import ComponentMemberV1

_OPENING = {"\"": "\"", "“": "”", "«": "»", "「": "」"}
"""Opening quotation marks and the mark that closes each (a turn wholly in quotes)."""


@dataclass(frozen=True, slots=True)
class PlannedRepairV1:
    """One repair before its vector exists: the head, its planned wording, the correction."""

    position_key: PositionKey
    memory_id: str
    statement_digest: str
    statement: str
    keywords: tuple[str, ...]


def _touched(write: WriteV1) -> set[str]:
    """Every memory a write reads or changes: its fence, target, ends and links."""
    found = set(write.fence)
    if write.target is not None:
        found.add(write.target)
    found.update(ended.memory_id for ended in write.ended)
    found.update(ref.memory_id for ref in write.contradicts if ref.memory_id is not None)
    return found


def _quoted_turn(view: AddReadView, span: QuoteSpanV1) -> bool:
    """Whether a quote lies in a turn wholly in quotation marks, which reports a correction.

    The turn is the quote's line in its Evidence part, after a line-start speaker
    label; it is wholly quoted when its text opens with a quotation mark and ends
    with the mark that closes it. A quote whose part is unknown counts as quoted.
    """
    part = view.evidence(span.evidence_id)
    if part is None:
        return True
    encoded = part.text.encode("utf-8")
    offset = len(encoded[:max(0, span.byte_start - part.byte_start)].decode(
        "utf-8", errors="ignore"))
    text = part.text
    start = text.rfind("\n", 0, offset) + 1
    end = text.find("\n", offset)
    line = text[start:len(text) if end < 0 else end]
    labels = scan_labels(line)
    labelled = bool(labels) and not line[:labels[0].start].strip()
    body = (line[labels[0].end:] if labelled else line).strip()
    closing = _OPENING.get(body[:1])
    return closing is not None and len(body) > 1 and body.rstrip(".!?").endswith(closing)


def _renamed(view: AddReadView, write: WriteV1, change: DescriptorChangeV1) -> str | None:
    """The entity a correction of a name renames: the one a naming head it changes or
    contradicts gives the replaced name as its name."""
    heads = {*(ended.memory_id for ended in write.ended),
             *(ref.memory_id for ref in write.contradicts if ref.memory_id is not None)}
    if write.target is not None:
        heads.add(write.target)
    old = name_norm(change.old)
    found = {annotation.entity_id
             for memory_id in heads if (record := view.memory(memory_id)) is not None
             for annotation in record.head.snapshot.naming
             if annotation.name_kind == "name" and name_norm(annotation.name) == old}
    return found.pop() if len(found) == 1 else None


def _named_elsewhere(view: AddReadView, anchors: Sequence[str], entity: str, old: str) -> bool:
    """Whether an anchor other than the renamed entity bears the replaced name."""
    for anchor in anchors:
        if anchor == entity:
            continue
        record = view.entity(anchor)
        if record is not None and any(name.name_norm == old for name in record.names):
            return True
    return False


def _generic(view: AddReadView, anchors: Iterable[str], change: DescriptorChangeV1) -> str | None:
    """The one broad type the corrected entities among ``anchors`` share as a descriptor."""
    found: set[str] = set()
    for anchor in anchors:
        record = view.entity(anchor)
        value = None if record is None else generic_descriptor(record.entity_type, change)
        if value is not None:
            found.add(value)
    return found.pop() if len(found) == 1 else None


def _kind_from_heads(view: AddReadView, statement: str, own: set[str],
                     contradicted: Iterable[str]) -> DescriptorChangeV1 | None:
    """The kind change of a plain kind assertion, read from the heads it contradicts.

    The pet is an anchor of the correction that its subject names ("Avery's pet
    Mochi"); each contradicted head sharing that anchor gives the kind words it names the
    pet with (`named_kinds`), except the pet's own broad type; the replaced kind is their
    one value besides the new kind (`kind_change`), so heads that disagree, or that name
    the pet with no other kind, give no rewording.
    """
    assertion = kind_assertion(statement)
    if assertion is None:
        return None
    subject = assertion[0]
    kinds: list[str] = []
    for memory_id in sorted(contradicted):
        record = view.memory(memory_id)
        if record is None or record.lifecycle_state != "active":
            continue
        head = record.head.snapshot
        for anchor in sorted(own & set(head.anchors)):
            entity = view.entity(anchor)
            if entity is None:
                continue
            names = [name.name for name in entity.names if token_contained(name.name, subject)]
            kinds += [kind for kind in named_kinds(head.statement, names)
                      if name_norm(kind) != name_norm(entity.entity_type)]
    return kind_change(statement, kinds)


def _identifies(view: AddReadView, statement: str, entity: str, old: str) -> bool:
    """Whether a statement identifies the entity by the replaced kind ("hamster Mochi")."""
    record = view.entity(entity)
    if record is None:
        return False
    names = {name.name for name in record.names}
    return any(re.search(r"(?<![\w-])" + re.escape(old) + r"\s+" + re.escape(name)
                         + r"(?![\w-])", statement, re.IGNORECASE) for name in names)


_NAMED = re.compile(r"^(?P<descriptor>[^,;:]+?) (?:is|was) (?:named|called) (?P<name>[^,.;:]+)\.$")
"""A naming head's statement: ``<descriptor> is named <name>.``"""
_RELATIONAL = re.compile(r"(?:['’]s|\b(?:his|her|their|its|my|your|our))\s", re.IGNORECASE)
"""A descriptor that relates the named entity to another ("Avery's employer")."""
_BARE_KIND = re.compile(r"^[a-z]+(?: [a-z]+){0,2}$")


def _bare_repair(view: AddReadView, memory_id: str) -> tuple[str, str, tuple[str, ...]] | None:
    """The bare wording of a naming head whose relational description a correction refuted.

    "Avery's employer is named Northstar Analytics." beside "Avery no longer works at
    Northstar Analytics.", which ends "Avery works at Northstar Analytics." (a fact of
    the same entities), keeps its own fact, the name, and drops the description the judge
    found the correction makes untrue: "An organization is named Northstar Analytics."
    (the entity's declared type, one to three lowercase words sharing no word with the
    name). Regression case open-quit-job: kept current with the old description, the naming
    head restated the employment the update had ended, a revival.

    Returns:
        tuple[str, str, tuple[str, ...]] | None: The new statement, the head's statement
            digest and its keywords the new statement still holds, or None.
    """
    record = view.memory(memory_id)
    if record is None or record.lifecycle_state != "active":
        return None
    head = record.head.snapshot
    names = [entry for entry in head.naming if entry.name_kind == "name"]
    match = _NAMED.match(head.statement)
    if (head.erased or len(names) != 1 or match is None
            or name_norm(match["name"]) != name_norm(names[0].name)
            or not _RELATIONAL.search(match["descriptor"] + " ")):
        return None
    entity = view.entity(names[0].entity_id)
    kind = "" if entity is None else entity.entity_type.strip()
    if not _BARE_KIND.match(kind) or set(token_seq(kind)) & set(token_seq(names[0].name)):
        return None
    try:
        statement = canonical_statement(
            f"{'An' if kind[0] in 'aeiou' else 'A'} {kind} is named {names[0].name}.")
    except ValueError:
        return None
    digest = statement_digest(statement)
    if statement == head.statement or view.tombstone(head.entity_id, digest) is not None or any(
            other.memory_id != memory_id for other in view.heads_by_digest(digest)):
        return None
    return statement, head.statement_digest, tuple(
        keyword for keyword in head.keywords if token_contained(keyword, statement))


_CHANGE_REASONS = frozenset({"update", "correction", "retraction"})
"""Reasons of a ``revise`` that replaces its target's fact (a correction or a
retraction of the member's own primary revises the head in place)."""
_RESTATING = re.compile(r"^(?P<name>[^,.;:]+?) (?:is|was) (?P<descriptor>[^,;:]+)\.$")
"""A fact that only gives a named entity a description: ``<name> is <description>.``"""
_NAMING_WORDS = re.compile(r"\b(?:named|called)\b", re.IGNORECASE)
"""A fact identifying an entity by a name ("Avery has a dog named Rex.")."""


def _replaced(view: AddReadView, write: WriteV1) -> list[MemorySnapshotV2]:
    """The facts a write ends or replaces, as the pin holds them: the heads it ends and the
    head a correction or retraction revises in place."""
    found = [ended.memory_id for ended in write.ended]
    if write.form == "revise" and write.target is not None and write.reason in _CHANGE_REASONS:
        found.append(write.target)
    return [record.head.snapshot for memory_id in found
            if (record := view.memory(memory_id)) is not None]


def _states_relationship(fact: MemorySnapshotV2, name: str) -> bool:
    """Whether a fact states a relationship with the named entity, by its name: it gives no
    name, holds the name and identifies nothing by a possessive description or a name
    ("Avery is married to Rowan."), or it only gives the named entity its description
    ("Rowan is Avery's spouse."). "Avery walks her dog Rex." states the walks, as
    Avery's ownership only identifies Rex there, and "Avery has a dog." a count that
    may change while Rex stays hers."""
    if (fact.naming or not token_contained(name, fact.statement)
            or _NAMING_WORDS.search(fact.statement)):
        return False
    if not _RELATIONAL.search(fact.statement + " "):
        return True
    match = _RESTATING.match(fact.statement)
    return match is not None and name_norm(match["name"]) == name_norm(name)


def _co_quoted(head: MemorySnapshotV2, fact: MemorySnapshotV2) -> bool:
    """Whether each quote of the head overlaps a quote of the fact: one source's words."""
    return bool(head.quote_spans) and all(
        any(span.evidence_id == other.evidence_id and span.byte_start < other.byte_end
            and other.byte_start < span.byte_end for other in fact.quote_spans)
        for span in head.quote_spans)


def _relationship_heads(view: AddReadView, replaced: Sequence[MemorySnapshotV2],
                        excluded: set[str]) -> list[str]:
    """The current naming heads that name an entity by the relationship a replaced fact
    states: one name, the fact's anchors, quoted from the fact's words."""
    found: list[str] = []
    for record in view.heads():
        head = record.head.snapshot
        names = [entry.name for entry in head.naming if entry.name_kind == "name"]
        if record.memory_id in excluded or len(names) != 1:
            continue
        anchors = frozenset(head.anchors)
        if any(frozenset(fact.anchors) == anchors and _co_quoted(head, fact)
               and _states_relationship(fact, names[0]) for fact in replaced):
            found.append(record.memory_id)
    return sorted(found)


def plan_repairs(
    plan: ResolutionPlanV1, members: Mapping[PositionKey, ComponentMemberV1],
    view: AddReadView,
) -> tuple[PlannedRepairV1, ...]:
    """Plan every descriptor rewording of one round's components, in commit order.

    Args:
        plan (ResolutionPlanV1): The round's resolution.
        members (Mapping[PositionKey, ComponentMemberV1]): Its component members.
        view (AddReadView): The round's pin.

    Returns:
        tuple[PlannedRepairV1, ...]: The rewordings by component, then member in
            canonical order, then head id; each head at most once per component.
    """
    planned: list[PlannedRepairV1] = []
    writes = [members[key].resolution.write for component in plan.components for key in component]
    touched = {memory_id for write in writes if write is not None for memory_id in _touched(write)}
    for component in plan.components:
        touches: dict[str, int] = {}
        written: set[str] = set()
        for key in component:
            write = members[key].resolution.write
            for memory_id in () if write is None else _touched(write):
                touches[memory_id] = touches.get(memory_id, 0) + 1
            if write is not None:
                written.update(ended.memory_id for ended in write.ended)
                if write.target is not None:
                    written.add(write.target)
        taken: set[str] = set()
        for key in component:
            member = members[key]
            write = member.resolution.write
            content = member.content
            if write is None or write.form == "annotate" or content is None:
                continue
            own = {anchor for anchor in member.resolution.anchors if anchor is not None}
            contradicted = {ref.memory_id for ref in write.contradicts
                            if ref.memory_id is not None}
            statement = str(content["statement"])
            if any(_quoted_turn(view, span) for group in write.groups for span in group):
                continue
            change = (descriptor_change(statement) or _kind_from_heads(
                view, statement, own, contradicted)) if member.correction_requested else None
            if change is None:
                # A naming head that names an entity by the relationship a fact this write
                # ends states (quoted from its words), and one a correction contradicts
                # while it ends a fact of the same entities, keep only the name; other
                # members may link a contradicted head, none may write it.
                replaced = _replaced(view, write)
                ended = {frozenset(fact.anchors) for fact in replaced}
                bare_heads = [
                    memory_id for memory_id in sorted(contradicted - written - taken)
                    if member.correction_requested and (record := view.memory(memory_id))
                    is not None and frozenset(record.head.snapshot.anchors) in ended]
                bare_heads += _relationship_heads(
                    view, replaced, touched | taken | {repair.memory_id for repair in planned})
                for memory_id in bare_heads:
                    bare = _bare_repair(view, memory_id)
                    if bare is not None:
                        taken.add(memory_id)
                        planned.append(PlannedRepairV1(
                            position_key=key, memory_id=memory_id, statement_digest=bare[1],
                            statement=bare[0], keywords=bare[2]))
                continue
            candidates = {memory_id: own for memory_id in contradicted
                          if touches.get(memory_id) == 1}
            renamed = _renamed(view, write, change) if change.naming else None
            if renamed is not None:
                old = name_norm(change.old)
                candidates |= {
                    record.memory_id: {renamed} for record in view.heads()
                    if record.memory_id not in candidates
                    and not touches.get(record.memory_id)
                    and renamed in record.head.snapshot.anchors
                    and not _named_elsewhere(view, record.head.snapshot.anchors, renamed, old)
                }
            if not change.naming:
                for record in view.heads():
                    if record.memory_id in candidates or touches.get(record.memory_id):
                        continue
                    shown = [anchor for anchor in own & set(record.head.snapshot.anchors)
                             if _identifies(view, record.head.snapshot.statement, anchor,
                                            change.old)]
                    if len(shown) == 1:
                        candidates[record.memory_id] = {shown[0]}
            for memory_id in sorted(candidates):
                if memory_id in taken:
                    continue
                record = view.memory(memory_id)
                if record is None or record.lifecycle_state != "active":
                    continue
                head = record.head.snapshot
                if head.erased or not candidates[memory_id] & set(head.anchors):
                    continue
                descriptor = _generic(view, candidates[memory_id] & set(head.anchors), change)
                if descriptor is None and not change.naming and memory_id in contradicted:
                    descriptor = _generic(view, own, change)
                if descriptor is None and not change.naming:
                    continue
                statement = repaired_statement(
                    head.statement, change, names=[entry.name for entry in head.naming],
                    descriptor=descriptor)
                if statement is None:
                    continue
                digest = statement_digest(statement)
                if view.tombstone(head.entity_id, digest) is not None or any(
                        other.memory_id != memory_id for other in view.heads_by_digest(digest)):
                    continue
                taken.add(memory_id)
                planned.append(PlannedRepairV1(
                    position_key=key, memory_id=memory_id,
                    statement_digest=head.statement_digest, statement=statement,
                    keywords=repaired_keywords(head.keywords, change, descriptor=descriptor),
                ))
    return tuple(planned)


def prepare_repairs(
    plan: ResolutionPlanV1, members: Mapping[PositionKey, ComponentMemberV1],
    view: AddReadView, *, owner: OwnerEmbedding, provider: EmbeddingProvider,
    store: BatchStore, reader: LogicalReader,
) -> dict[PositionKey, ComponentMemberV1]:
    """Attach each committing correction's planned repairs with their vectors.

    Args:
        plan (ResolutionPlanV1): The round's resolution.
        members (Mapping[PositionKey, ComponentMemberV1]): Its component members.
        view (AddReadView): The round's pin.
        owner (OwnerEmbedding): The owner's embedding identity, ticket and authority.
        provider (EmbeddingProvider): The workspace's single embedder.
        store (BatchStore): The Writer, which keeps the create-once batches.
        reader (LogicalReader): Reads a stored batch by its reference.

    Returns:
        dict[PositionKey, ComponentMemberV1]: Every member; a correction with repairs
            carries them.

    Raises:
        QuoteMemError: ``binding_mismatch`` when a stored batch does not hold the vector
            its entry names; or as `embed_texts`.
    """
    planned = plan_repairs(plan, members, view)
    if not planned:
        return dict(members)
    located = embed_texts(
        embedding_texts([(repair.position_key, repair.statement, None) for repair in planned]),
        owner, view=view, provider=provider, store=store,
    )
    repairs: dict[PositionKey, list[HeadRepairV1]] = {}
    for repair in planned:
        use = located[("statement", repair.statement)]
        batch = reader.read_logical(use.ref, "semantic_embedding")
        if not isinstance(batch, StatementEmbeddingV1) or use.index >= len(batch.entries):
            raise QuoteMemError("binding_mismatch", "A repair names an embedding its batch lacks")
        entry = batch.entries[use.index]
        repairs.setdefault(repair.position_key, []).append(HeadRepairV1(
            memory_id=repair.memory_id, statement_digest=repair.statement_digest,
            statement=repair.statement, keywords=repair.keywords,
            embedding=tuple(entry.vector), embedding_space=batch.embedding_space,
            embedding_input_digest=entry.input_digest,
        ))
    return {key: replace(member, repairs=tuple(repairs.get(key, ())))
            for key, member in members.items()}


__all__ = ["PlannedRepairV1", "plan_repairs", "prepare_repairs"]
