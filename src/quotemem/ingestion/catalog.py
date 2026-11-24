"""The extraction handle catalog: the entities one request may anchor.

The catalog lists ``active`` entities only, deduplicated, in priority order:

1. ``user``, when some rendered source is authenticated-self speech or a
   context memory anchors the user;
2. participants, one per distinct source tuple in first-appearance order;
2b. labelled speakers, one per distinct ``speaker`` label name on sources
   without a tuple, keyed ``aml-label-speaker-v1`` (never ``user``);
3. entities an SDK ``entity_binding`` marks in a rendered source;
4. entities with a readable name token-contained in a rendered source,
   labelled speakers whose label is (a label keys its speaker) and source
   participants with a speaker name of their readable Evidence that is, by the
   number of sources that name them, then by commit order and readable names
   (exact phrase);
5. anchors of the context memories in rank order.

``user`` takes no number; the others are ``E1``..``E64`` in that order, and
any further entity is dropped with ``catalog_truncated`` set. A participant
seen before its first mint is listed by its key alone. Names are the entity's
``p_name`` rows, at most 8 in catalog order; speaker names are the names its
sources were spoken under, or for a participant listed by route 4 the label or
speaker name as the sources write it, and are never entity names.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Container, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

from quotemem.core.add_read_view import AddReadView, EntityNameRecord, EntityRecord
from quotemem.core.entities import USER
from quotemem.core.recall_corpora import EntityRetrieval
from quotemem.core.request_order import entity_order, name_order
from quotemem.core.statement import (
    MAX_CATALOG_ENTITIES,
    MAX_CATALOG_NAMES,
    named_participants,
)
from quotemem.ingestion.sources import RenderedSource

CatalogKind = Literal["user", "participant", "entity"]
CatalogRoute = Literal["self", "participant", "label", "sdk", "name", "memory"]
"""The route that listed an entry; ``self`` and ``memory`` both list ``user``."""


@dataclass(frozen=True, slots=True)
class CatalogEntry:
    """One catalog entry (entry fields).

    Attributes:
        handle (str): ``user`` or ``E1``..``E64``.
        kind (CatalogKind): ``user``, ``participant`` or ``entity``.
        route (CatalogRoute): The first route that listed it.
        entity_id (str | None): Its `EntityRef`; None for a participant before
            its first mint.
        participant_key (str | None): A participant's key, else None.
        entity_type (str): Its stored type; ``person`` for the user and for a
            participant not yet minted.
        names (tuple[EntityNameRecord, ...]): At most 8 readable names, one per
            ``(name_norm, name_kind)``, in catalog order.
        speaker_names (tuple[str, ...]): The distinct names its rendered sources
            were spoken under; for a labelled speaker listed by route 4, its
            label as the sources write it.
        memories (tuple[str, ...]): The M handles of the context memories that
            anchor it.
    """

    handle: str
    kind: CatalogKind
    route: CatalogRoute
    entity_id: str | None
    participant_key: str | None
    entity_type: str
    names: tuple[EntityNameRecord, ...]
    speaker_names: tuple[str, ...]
    memories: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Catalog:
    """The catalog of one request.

    Attributes:
        entries (tuple[CatalogEntry, ...]): ``user`` first when issued, then
            ``E1``..``Ek``.
        truncated (bool): Some entity was dropped by the 64-entry bound.
    """

    entries: tuple[CatalogEntry, ...]
    truncated: bool

    def entry(self, handle: str) -> CatalogEntry | None:
        """Return the entry issued under a handle, or None."""
        return next((entry for entry in self.entries if entry.handle == handle), None)

    def handle_of(self, entity_id: str) -> str | None:
        """Return the handle of an entity, or None when it is not listed."""
        return next(
            (entry.handle for entry in self.entries if entry.entity_id == entity_id), None,
        )

    def handle_of_participant(self, participant_key: str) -> str | None:
        """Return the handle of a participant by its key, or None when it is not listed."""
        return next((
            entry.handle for entry in self.entries if entry.participant_key == participant_key
        ), None)


@dataclass(frozen=True, slots=True)
class SpeakerMentions:
    """The labelled speakers the rendered sources name (route 4).

    Attributes:
        by_source (tuple[frozenset[str], ...]): The participant keys each source
            names, in source order.
        names (Mapping[str, tuple[str, ...]]): Each named key's distinct
            mentions as written, in order of first appearance.
    """

    by_source: tuple[frozenset[str], ...]
    names: Mapping[str, tuple[str, ...]]


def speaker_mentions(
    scope_id: str, sources: Sequence[RenderedSource], keys: Container[str],
    speakers: Mapping[str, Sequence[str]],
) -> SpeakerMentions:
    """Find the participants among ``keys`` that a rendered source names by label or speaker name.

    A labelled speaker is keyed by its label's ``name_norm`` within the scope, so
    a readable stretch that token-contains the label names that speaker, whether
    or not the speaker speaks in the request. A source participant is keyed by
    its tuple, so a stretch names it where it token-contains a speaker name of
    the participant's readable Evidence (``speakers``). Each readable stretch is
    read on its own, as for names: a hidden interval cannot join a label.

    Args:
        scope_id (str): The request's scope.
        sources (Sequence[RenderedSource]): The rendered sources in request order.
        keys (Container[str]): The participant keys of the scope's participants.
        speakers (Mapping[str, Sequence[str]]): The view's `AddReadView.speakers`.

    Returns:
        SpeakerMentions: The keys each source names and every named key's mentions.
    """
    by_source: list[frozenset[str]] = []
    names: dict[str, list[str]] = {}
    for source in sources:
        named: set[str] = set()
        for text in source.readable_texts():
            for key, mention in named_participants(scope_id, text, keys, speakers):
                named.add(key)
                written = names.setdefault(key, [])
                if mention not in written:
                    written.append(mention)
        by_source.append(frozenset(named))
    return SpeakerMentions(tuple(by_source), {key: tuple(value) for key, value in names.items()})


@dataclass(slots=True)
class _Draft:
    route: CatalogRoute
    kind: CatalogKind
    entity_id: str | None
    participant_key: str | None
    entity_type: str
    names: tuple[EntityNameRecord, ...]
    speaker_names: list[str] = field(default_factory=list[str])


def _catalog_names(names: Sequence[EntityNameRecord]) -> tuple[EntityNameRecord, ...]:
    kept: dict[tuple[str, str], EntityNameRecord] = {}
    for row in sorted(names, key=name_order):
        kept.setdefault((row.name_norm, row.name_kind), row)
    return tuple(kept.values())[:MAX_CATALOG_NAMES]


def build_catalog(
    view: AddReadView,
    sources: Sequence[RenderedSource],
    memory_anchors: Sequence[tuple[str, Sequence[str]]] = (),
) -> Catalog:
    """Build the handle catalog of one request.

    Without memory anchors the catalog holds routes 1 to 4, which context
    selection reads for its reserved slots; the request's catalog adds
    route 5 and every entry's memory handles. The routes of 1 to 4 do not read
    memories, so both calls number their entries identically.

    Args:
        view (AddReadView): The owner pin's read view.
        sources (Sequence[RenderedSource]): The rendered sources in request order.
        memory_anchors (Sequence[tuple[str, Sequence[str]]]): The context
            memories in handle order, each with its anchors (primary first).

    Returns:
        Catalog: The entries and whether any entity was dropped.
    """
    user = any(source.attribution.authenticated_self for source in sources)
    drafts: list[_Draft] = []
    by_entity: dict[str, _Draft] = {}
    by_key: dict[str, _Draft] = {}

    def add(draft: _Draft) -> _Draft:
        drafts.append(draft)
        if draft.entity_id is not None:
            by_entity[draft.entity_id] = draft
        if draft.participant_key is not None:
            by_key[draft.participant_key] = draft
        return draft

    def entity(record: EntityRecord, route: CatalogRoute) -> None:
        if record.entity_id not in by_entity:
            add(_Draft(
                route, "participant" if record.participant_key else "entity", record.entity_id,
                record.participant_key, record.entity_type, _catalog_names(record.names),
            ))

    for kind in ("tuple", "label"):
        for source in sources:
            attribution = source.attribution
            key = attribution.participant_key
            if key is None or attribution.participant_kind != kind:
                continue
            draft = by_key.get(key)
            if draft is None:
                record = view.participant(key)
                if record is not None and record.entity_id in by_entity:
                    draft = by_entity[record.entity_id]
                else:
                    draft = add(_Draft(
                        "participant" if kind == "tuple" else "label", "participant",
                        None if record is None else record.entity_id, key,
                        "person" if record is None else record.entity_type,
                        () if record is None else _catalog_names(record.names),
                    ))
            for name in attribution.names:
                if name not in draft.speaker_names:
                    draft.speaker_names.append(name)
    for source in sources:
        for part in source.run.parts:
            for binding in sorted(part.entity_bindings, key=lambda item: item.byte_start):
                if source.rendered_bytes(part.object_id, binding.byte_start, binding.byte_end) and (
                    record := view.entity(binding.entity_id)
                ) is not None:
                    entity(record, "sdk")
    # Query each readable stretch independently: a hidden interval cannot join a name.
    retrieval = EntityRetrieval.of_view(view, include_user=False)
    retrieval_record = {record.entity_id: record for record in view.entities()}
    participants = {
        record.participant_key: entity_id for entity_id, record in retrieval_record.items()
        if record.participant_key is not None
    }
    mentions = speaker_mentions(view.scope_id, sources, participants, view.speakers())
    hits: Counter[str] = Counter()
    for source, named in zip(sources, mentions.by_source, strict=True):
        hits.update({
            entity_id for text in source.readable_texts()
            for entity_id in retrieval.exact(text)
        } | {participants[key] for key in named})
    # The id is the last key: entities equal in hits, commit, names and type are
    # otherwise ordered by the set the hits came from, which follows the hash seed.
    for entity_id in sorted(hits, key=lambda identity: (
        -hits[identity], entity_order(retrieval_record[identity]), identity,
    )):
        if entity_id not in by_entity and (record := view.entity(entity_id)) is not None:
            entity(record, "name")
            key = record.participant_key
            if key is not None and key in mentions.names:
                by_entity[entity_id].speaker_names.extend(mentions.names[key])
    for _, anchors in memory_anchors:
        for anchor in (*anchors[:1], *sorted(
            anchors[1:], key=lambda anchor: entity_order(retrieval_record[anchor])
            if anchor in retrieval_record else (0, (), ""),
        )):
            if anchor == USER:
                user = True
            elif (record := view.entity(anchor)) is not None:
                entity(record, "memory")
    listed = drafts[:MAX_CATALOG_ENTITIES]

    def memories(entity_id: str | None) -> tuple[str, ...]:
        return tuple(
            handle for handle, anchors in memory_anchors
            if entity_id is not None and entity_id in anchors
        )

    entries = [
        CatalogEntry(
            handle=f"E{number}", kind=draft.kind, route=draft.route,
            entity_id=draft.entity_id, participant_key=draft.participant_key,
            entity_type=draft.entity_type, names=draft.names,
            speaker_names=tuple(draft.speaker_names), memories=memories(draft.entity_id),
        )
        for number, draft in enumerate(listed, start=1)
    ]
    if user:
        record = view.entity(USER)
        entries.insert(0, CatalogEntry(
            handle=USER, kind="user",
            route="self" if any(s.attribution.authenticated_self for s in sources) else "memory",
            entity_id=USER, participant_key=None, entity_type="person",
            names=() if record is None else _catalog_names(record.names),
            speaker_names=(), memories=memories(USER),
        ))
    return Catalog(tuple(entries), len(drafts) > len(listed))
