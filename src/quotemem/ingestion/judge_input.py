"""The judge-v8 member input, its digest, and the physical requests.

`build_member_input` renders one nominated member as its logical input:
the ``JudgeMemberV8`` a request lists and every catalog entry it
reaches, keyed by the handles `assign_handles` gives its nomination, so the
member input and its ``NeighborSnapshotV1`` name every object alike.
Message and quote text comes only from ``render_masked``, so withheld bytes
never reach the judge. An entity entry is canonical: its binding
says what the entity is (the user, a participant or a third party, whatever
slot role it has for the member) and its facts are the entity's most recent
readable heads at the pin, so every member of a round shows a shared E handle
alike. `member_input_digest` binds the canonical input under the workspace
key: the member's identity whatever its batch, and with the binding
digest the key of verdict reuse.

A cited message is shown in full. Only a member that would not fit a request
alone that way shows each cited message as the run of whole parts from
its first to its last quoted part, marked ``excerpt``, so a fact quoted from
one part of a 1 MiB message is still reviewed. Neighbours are never cut.

`pack_requests` lays one owner's round of member inputs into physical requests:

- members are taken in canonical order ``(σ, position_key)``, at most
  ``MAX_BATCH_MEMBERS`` (4) a request;
- a member that must choose a referent is reviewed in a request of its own
  (a lesson of the previous implementation): an ``open`` slot with candidates, or a
  ``nominated`` slot with another candidate besides its nominee;
- a member that would give a handle another object, or another entry, than the
  request already holds moves to the next request; with canonical
  entity entries only member-specific entries differ: a head's history beside
  a history correction, a history entry's ``of``, a re-picked declaration's
  mention;
- a request beyond the measure (input 131,072 bytes: the canonical JSON
  of its messages and schema plus 4,096; estimated output 65,536 bytes) is
  bisected in order, and a member that does not fit alone, even with its cited
  messages shown as their quoted runs, is left out for ``review_capacity_exceeded``.

The plan is a pure function of its inputs: no clock, no randomness, and every
order is canonical.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from quotemem.core.add_read_view import (
    AddReadView,
    EndedBy,
    EvidenceRecord,
    MessageKey,
    Sigma,
    VersionRecord,
)
from quotemem.core.entities import USER, PositionKey
from quotemem.core.errors import QuoteMemError
from quotemem.core.extraction_output import render_wire_date
from quotemem.core.identity import canonical_json
from quotemem.core.judge_output import (
    MAX_BATCH_MEMBERS,
    MAX_PACKET_MEMBERS,
    CatalogEntryV8,
    JudgeAnchorV8,
    JudgeBatchItemRequestV8,
    JudgeBatchRequestV8,
    JudgeEntityEntryV8,
    JudgeEntityNameV8,
    JudgeForgottenEntryV8,
    JudgeHistoryEntryV8,
    JudgeMemberV8,
    JudgeMemoryEntryV8,
    JudgeMessageEntryV8,
    JudgeNamingV8,
    JudgeNewEntityEntryV8,
    JudgeQuoteV8,
    JudgeSiblingEntryV8,
    JudgeTimeV8,
    judge_messages,
    judge_request_schema,
    render_said_at,
)
from quotemem.core.readability import Readability
from quotemem.core.request_order import entity_order, name_order
from quotemem.ingestion.nomination import (
    ContentBinder,
    FactTimeV1,
    HandleKey,
    MemberSlotV1,
    NominatedTargetV1,
    NominatedTombstoneV1,
    NominationV1,
    RevisesRouteV1,
    SourceMessageV1,
    assign_handles,
    make_handle_key,
    slot_key,
)
from quotemem.ingestion.sources import Attribution

MEMBER_INPUT_PURPOSE: Final = "judge-v8-member-input"
MAX_REQUEST_INPUT_BYTES: Final = 131_072
MAX_REQUEST_OUTPUT_BYTES: Final = 65_536
BATCHED_REQUEST_INPUT_BYTES: Final = 262_144
BATCHED_REQUEST_OUTPUT_BYTES: Final = 98_304
"""The batched packing's request measure: one Add's members in one request where it fits.

LoCoMo calibration judge replies reached 39,400 completion tokens, reasoning included; about
25k visible tokens (98,304 bytes) leave the rest of the provider's output to reasoning.
"""
REQUEST_MARGIN_BYTES: Final = 4_096
"""Bytes added to a request's measured messages and schema for the transport fields."""

type _Declared = tuple[str | None, str | None, str | None, str | None]
"""A new entity as shown: type, proposed name, descriptor and mention."""


def judge_time(time: FactTimeV1) -> JudgeTimeV8:
    """Render a fact's time as the judge sees it, its dates at their granularity.

    Args:
        time (FactTimeV1): The expression and anchored bounds.

    Returns:
        JudgeTimeV8: The expression text, ``valid_from`` and ``valid_to`` as wire
            dates (null when absent or unanchored), and the granularity.
    """
    granularity = time.granularity
    return JudgeTimeV8(
        text=time.text, valid_from=render_wire_date(time.valid_from, granularity),
        valid_to=render_wire_date(time.valid_to, granularity), granularity=granularity,
    )


def member_input_digest(
    member: JudgeMemberV8, catalog: Mapping[str, CatalogEntryV8], bind: ContentBinder,
) -> str:
    """Return a member's identity: the keyed binding of its canonical input.

    Args:
        member (JudgeMemberV8): The member as a request lists it.
        catalog (Mapping[str, CatalogEntryV8]): Every entry it reaches.
        bind (ContentBinder): The workspace's keyed content binder.

    Returns:
        str: ``bind("judge-v8-member-input", {"catalog": ..., "member": ...})`` over
            the JSON forms; the request's ``member_id``.
    """
    return bind(MEMBER_INPUT_PURPOSE, {
        "catalog": {handle: entry.model_dump(mode="json") for handle, entry in catalog.items()},
        "member": member.model_dump(mode="json"),
    })


def member_output_bytes(member: JudgeMemberV8) -> int:
    """Estimate the bytes of one member's reply.

    Args:
        member (JudgeMemberV8): The member.

    Returns:
        int: ``1,024 + 1,600 + 56·targets + 16·history + 48·tombstones
            + 64·judged slots + 16·groups``.
    """
    judged = sum(anchor.binding in ("nominated", "open") for anchor in member.anchors)
    return (1024 + 1600 + 56 * len(member.targets) + 16 * len(member.history)
            + 48 * len(member.tombstones) + 64 * judged + 16 * len(member.support_groups))


@dataclass(frozen=True, slots=True)
class MemberInputV1:
    """One member's logical input to judge v8.

    Attributes:
        position_key (PositionKey): The member's position.
        order (tuple[Sigma, PositionKey]): Its canonical key.
        member (JudgeMemberV8): The member as a request lists it.
        catalog (Mapping[str, CatalogEntryV8]): Every entry it reaches, by handle.
        identities (Mapping[str, HandleKey]): The object each of its handles names,
            quote handles included.
        digest (str): Its ``member_input_digest``.
    """

    position_key: PositionKey
    order: tuple[Sigma, PositionKey]
    member: JudgeMemberV8
    catalog: Mapping[str, CatalogEntryV8]
    identities: Mapping[str, HandleKey]
    digest: str

    @property
    def chooses_referent(self) -> bool:
        """Whether the judge must choose a referent: it is then reviewed alone.

        True when an ``open`` slot has a candidate, or a ``nominated`` slot has a
        candidate besides its nominee.
        """
        return any(
            anchor.binding == "open" and bool(anchor.candidates)
            or anchor.binding == "nominated" and len(anchor.candidates) > 1
            for anchor in self.member.anchors
        )

    @property
    def output_bytes(self) -> int:
        """`member_output_bytes` of its member."""
        return member_output_bytes(self.member)


type MemberInputObserver = Callable[[MemberInputV1, AddReadView], None]
"""An opted-in observer of each member input a round reviews, with the pin it was built at.

The recorded-replay harness keys a judge member by content identities that only the
pin resolves: a quote's Add and ordinal, an entity handle's participant key. It
reads, never writes, and must not raise; production never reads what it keeps.
"""


def build_member_input(
    nomination: NominationV1, view: AddReadView, *, bind: ContentBinder,
    attributions: Sequence[Attribution],
    placeholders: Mapping[tuple[str, int, int], str] | None = None,
) -> MemberInputV1:
    """Render one nominated member as its judge-v8 input.

    Handles come from `assign_handles`, so the input and the member's
    ``NeighborSnapshotV1`` name every object alike. Catalog entries follow
    A head shows its context history entries, a history entry the head
    of its own memory (``of``); an existing entity its binding by what it is
    (``self``, ``participant`` or ``third_party``; an SDK binding shows on the
    anchor), its names and its ``entity_facts``, the same for every
    member at the pin; a message is rendered masked, a context message only by
    its last ``context_message_chars`` characters.

    A message shows its metadata speaker. A cited message whose
    cited source runs are attributed differently from each other, from that
    metadata speaker, or with a label conflict shows ``speaker`` null
    (unknown), never a guessed name: a labelled speaker is never the user
    (rule 3). Q entries stay whole messages and the frozen prompt is unchanged,
    except for a member whose one-member request would exceed the input
    bound: each cited message it quotes only part of then shows the run of
    whole parts from the first to the last part its quotes lie in, with
    ``excerpt`` set. Its neighbours and context messages are unchanged.

    Args:
        nomination (NominationV1): The member's nomination.
        view (AddReadView): The pin it was nominated at.
        bind (ContentBinder): The workspace's keyed content binder.
        attributions (Sequence[Attribution]): The cited runs' attribution, one
            per member quote in quote order, computed once by the rebuild over
            the frozen label view and shared with ``wake_digest`` (``reentry.cited_attributions``).
        placeholders (Mapping[tuple[str, int, int], str] | None): The forgotten
            name occurrences the owner's settlement withheld, each rendered as whom it
            names ("[the user's cat]") instead of the marker
            (``ExtractionOutcome.placeholders``); none otherwise.

    Returns:
        MemberInputV1: The member, its closed catalog, its handles' objects and
            its digest.

    Raises:
        QuoteMemError: ``binding_mismatch`` when the member is not reviewable (an open
            slot without candidates), a quote is not readable at the pin, an
            entity the input names is not active, or the attributions do not pair
            with the quotes.
    """
    if not nomination.reviewable:
        raise QuoteMemError("binding_mismatch", "An open slot without candidates is not reviewed")
    handles = assign_handles(nomination, bind)
    unattributed = _unattributed(nomination, view, attributions)
    builder = _Builder(nomination, view, handles, placeholders)
    builder.unattributed = unattributed
    built = builder.build(bind)
    runs = _quoted_runs(nomination, view)
    if runs and request_input_bytes(physical_request([built])) > MAX_REQUEST_INPUT_BYTES:
        builder = _Builder(nomination, view, handles, placeholders)
        builder.unattributed, builder.runs = unattributed, runs
        built = builder.build(bind)
    return built


def _message_of(part: EvidenceRecord) -> MessageKey:
    return MessageKey(part.session_id, part.request_id, part.ordinal)


def _quoted_runs(nomination: NominationV1, view: AddReadView) -> dict[MessageKey, tuple[int, int]]:
    """Each cited message's run of whole parts from its first to its last quoted part.

    Only a run that leaves a readable part of its message out is listed: the
    message bytes ``(start, end)`` a member that does not fit alone shows.
    """
    quoted: dict[MessageKey, list[EvidenceRecord]] = {}
    for span in nomination.member.quotes:
        part = view.evidence(span.evidence_id)
        if part is not None:
            quoted.setdefault(_message_of(part), []).append(part)
    runs: dict[MessageKey, tuple[int, int]] = {}
    for source in nomination.sources:
        parts = quoted.get(source.message.key)
        if not source.cited or not parts:
            continue
        start = min(part.byte_start for part in parts)
        end = max(part.byte_end for part in parts)
        if any(part.byte_start < start or part.byte_end > end
               for part in source.message.parts if part.state != "erased"):
            runs[source.message.key] = (start, end)
    return runs


def _unattributed(
    nomination: NominationV1, view: AddReadView, attributions: Sequence[Attribution],
) -> frozenset[MessageKey]:
    """The cited messages whose cited runs disagree in attribution."""
    quotes = nomination.member.quotes
    if len(attributions) != len(quotes):
        raise QuoteMemError("binding_mismatch",
                            "Cited-run attributions do not pair with the quotes")
    shown: dict[MessageKey, str | None] = {}
    for source in nomination.sources:
        if source.cited:
            first = source.message.parts[0]
            shown[_message_of(first)] = first.speaker_entity or None
    found: set[MessageKey] = set()
    for span, attribution in zip(quotes, attributions, strict=True):
        part = view.evidence(span.evidence_id)
        cited = None if part is None else _message_of(part)
        if cited is None or cited not in shown:
            raise QuoteMemError("binding_mismatch",
                                "A member quote lies outside its cited messages")
        if attribution.speaker_conflict or attribution.speaker != shown[cited]:
            found.add(cited)
    return frozenset(found)


class _Builder:
    """One member input while it is rendered."""

    def __init__(self, nomination: NominationV1, view: AddReadView,
                 handles: Mapping[HandleKey, str],
                 placeholders: Mapping[tuple[str, int, int], str] | None = None) -> None:
        self.nomination, self.view, self.handles = nomination, view, handles
        self.placeholders = placeholders or {}
        self.unattributed: frozenset[MessageKey] = frozenset()
        self.runs: Mapping[MessageKey, tuple[int, int]] = {}
        self.read = Readability(view.scope_kind, view.scope_id, view.now, view.evidence)
        self.catalog: dict[str, CatalogEntryV8] = {}
        self.identities: dict[str, HandleKey] = {}
        member = nomination.member
        self.heads = {target.object_id: target for target in nomination.targets
                      if target.kind == "head"}
        slots = [*member.slots, *(slot for target in nomination.targets
                                  if target.sibling is not None for slot in target.sibling.slots)]
        self.participants = {slot.participant_key for slot in slots
                             if slot.entity_id is None and slot.participant_key is not None}
        self.declared: dict[str, _Declared] = {}
        for slot in slots:
            if slot.mint_key is not None:
                self.declared.setdefault(
                    slot.mint_key, (slot.entity_type, slot.name, slot.descriptor, slot.mention),
                )
        for item in nomination.slots:
            for candidate in item.candidates:
                if candidate.declaration is not None:
                    declared = candidate.declaration
                    self.declared.setdefault(declared.mint_key, (
                        declared.entity_type, declared.name, declared.descriptor, None,
                    ))

    # Handles and entries -----------------------------------------------------------------

    def use(self, key: HandleKey) -> str:
        handle = self.handles[key]
        self.identities[handle] = key
        return handle

    def put(self, key: HandleKey, entry: CatalogEntryV8) -> str:
        handle = self.use(key)
        self.catalog.setdefault(handle, entry)
        return handle

    def entity(self, key: HandleKey) -> str:
        """The handle of an entity or a new declaration, with its entry."""
        handle = self.use(key)
        if handle not in self.catalog:
            self.catalog[handle] = self.entity_entry(key)
        return handle

    def entity_entry(self, key: HandleKey) -> CatalogEntryV8:
        kind, object_id, _version = key
        if kind == "new_entity":
            entity_type, name, descriptor, mention = self.declared[object_id]
            if entity_type is None or descriptor is None:
                raise QuoteMemError("binding_mismatch",
                                    "A new entity needs its type and descriptor")
            return JudgeNewEntityEntryV8(type="new_entity", entity_type=entity_type,
                                         proposed_name=name, descriptor=descriptor,
                                         mention=mention)
        if object_id in self.participants:
            return JudgeEntityEntryV8(type="entity", binding="participant",
                                      entity_type="person", names=[], facts=[])
        record = self.view.entity(object_id)
        facts = self.nomination.entity_facts.get(object_id)
        if record is None or facts is None:
            raise QuoteMemError("binding_mismatch",
                                "A member input names an entity that is not active")
        binding = ("self" if object_id == USER else "participant"
                   if record.participant_key is not None else "third_party")
        return JudgeEntityEntryV8(
            type="entity", binding=binding, entity_type=record.entity_type or None,
            names=[JudgeEntityNameV8(name=row.name, kind=row.name_kind)
                   for row in sorted(record.names, key=name_order)],
            facts=[fact.head.snapshot.statement for fact in facts],
        )

    def ordered_anchors(self, anchors: Sequence[str]) -> tuple[str, ...]:
        """Keep the primary first and order stored secondary anchors by entity content."""
        def order(anchor: str) -> tuple[int, tuple[str, ...], str]:
            record = self.view.entity(anchor)
            return (0, (), "") if record is None else entity_order(record)

        return (*anchors[:1], *sorted(anchors[1:], key=order))

    def about(self, keys: Iterable[HandleKey | None]) -> list[str]:
        """The handles of the shown anchors among ``keys``, primary first; others are omitted."""
        found: list[str] = []
        for key in keys:
            if key is not None and key in self.handles:
                handle = self.entity(key)
                if handle not in found:
                    found.append(handle)
        return found

    # Targets, history and tombstones -----------------------------------------------------

    def target(self, target: NominatedTargetV1) -> str:
        if target.sibling is not None:
            sibling = target.sibling
            return self.put(target.handle_key, JudgeSiblingEntryV8(
                type="sibling", status="in_this_add", statement=sibling.statement,
                kind=sibling.kind, time=judge_time(sibling.time),
                said_at=render_said_at(sibling.sigma[0]),
                about=self.about(slot_key(slot) for slot in sibling.slots),
            ))
        version = target.version
        if version is None or (target.kind == "retired") != (target.ended_by is not None):
            raise QuoteMemError("binding_mismatch", "A stored target lost its version or its end")
        if target.ended_by is not None:
            head = self.heads.get(target.object_id)
            return self.earlier(target.handle_key, version, target.ended_by,
                                None if head is None else self.use(head.handle_key))
        snapshot = version.snapshot
        return self.put(target.handle_key, JudgeMemoryEntryV8(
            type="memory", status="current", statement=snapshot.statement, kind=snapshot.kind,
            time=judge_time(FactTimeV1.from_snapshot(snapshot)), said_at=render_said_at(
                version.sigma[0]),
            about=self.about(make_handle_key("entity", anchor)
                             for anchor in self.ordered_anchors(version.anchors)),
            naming=bool(snapshot.naming),
            history=[self.use(item.handle_key) for item in self.nomination.history
                     if item.head_memory_id == target.object_id],
        ))

    def earlier(self, key: HandleKey, version: VersionRecord, ended_by: EndedBy,
                of: str | None) -> str:
        snapshot = version.snapshot
        return self.put(key, JudgeHistoryEntryV8(
            type="history", status="earlier_version", of=of, statement=snapshot.statement,
            kind=snapshot.kind, time=judge_time(FactTimeV1.from_snapshot(snapshot)),
            said_at=render_said_at(version.sigma[0]), ended_by=ended_by,
        ))

    def forgotten(self, item: NominatedTombstoneV1) -> str:
        record = item.record
        return self.put(make_handle_key("forgotten", record.object_id), JudgeForgottenEntryV8(
            type="forgotten", status="forgotten", statement=record.statement, kind=record.kind,
            time=judge_time(FactTimeV1(None, record.valid_from, record.valid_to,
                                       record.time_granularity)),
            about=self.about(make_handle_key("entity", anchor)
                             for anchor in self.ordered_anchors(
                                 (record.entity_id, *record.about_entity_ids))),
        ))

    # Messages and quotes ------------------------------------------------------------------

    def message(self, source: SourceMessageV1) -> str:
        message, limit = source.message, self.nomination.policy.context_message_chars
        run = self.runs.get(message.key) if source.cited else None
        if run is None:
            text = self.read.render_masked(message.parts, placeholders=self.placeholders).text
            excerpt = not source.cited and len(text) > limit
            shown = text[-limit:] if excerpt else text
        else:
            shown = self.read.render_masked(message.parts, start=run[0], end=run[1],
                                            placeholders=self.placeholders).text
            excerpt = True
        part = message.parts[0]
        speaker = None if _message_of(part) in self.unattributed else part.speaker_entity or None
        return self.put(source.handle_key, JudgeMessageEntryV8(
            type="message", role=part.role or "unknown", speaker=speaker,
            said_at=render_said_at(message.sigma[0]), text=shown, excerpt=excerpt,
            order=message.sigma,
        ))

    def quotes(self) -> list[JudgeQuoteV8]:
        cited = {source.message.key: source for source in self.nomination.sources if source.cited}
        quotes: list[JudgeQuoteV8] = []
        for span in self.nomination.member.quotes:
            part = self.view.evidence(span.evidence_id)
            if part is None or not self.read.p_span(span):
                raise QuoteMemError("binding_mismatch", "A member quote is not readable at the pin")
            source = cited[MessageKey(part.session_id, part.request_id, part.ordinal)]
            text = self.read.render_masked(
                source.message.parts, start=span.byte_start, end=span.byte_end,
            ).text
            key = make_handle_key("quote", span.evidence_id, f"{span.byte_start}-{span.byte_end}")
            quotes.append(JudgeQuoteV8(handle=self.use(key), source=self.message(source),
                                       text=text))
        return quotes

    # The member ---------------------------------------------------------------------------

    def anchor(self, slot: MemberSlotV1, candidates: list[str],
               catalog_complete: bool | None) -> JudgeAnchorV8:
        key = slot_key(slot)
        return JudgeAnchorV8(
            slot=slot.slot, role="primary" if slot.slot == 0 else "secondary",
            binding=slot.binding, entity=None if key is None else self.entity(key),
            mention=slot.mention, candidates=candidates, catalog_complete=catalog_complete,
        )

    def revises(self) -> list[str]:
        """The handles of what the ``revises`` references issued: a head or a retired version."""
        handles: list[str] = []
        for route in self.nomination.routes:
            if not isinstance(route, RevisesRouteV1):
                continue
            for reference in route.references:
                memory_id, revision_id = reference.issued_memory_id, reference.issued_revision_id
                if memory_id is None or revision_id is None:
                    continue
                head = self.heads.get(memory_id)
                key = head.handle_key if head is not None and head.revision_id == revision_id \
                    else make_handle_key("history", memory_id, revision_id)
                if (handle := self.use(key)) not in handles:
                    handles.append(handle)
        return handles

    def build(self, bind: ContentBinder) -> MemberInputV1:
        nomination = self.nomination
        member, flags = nomination.member, nomination.flags
        targets = [self.target(target) for target in nomination.targets]
        history = [
            self.earlier(item.handle_key, item.entry.version, item.entry.ended_by,
                         self.use(self.heads[item.head_memory_id].handle_key)
                         if item.entry.version.memory_id == item.head_memory_id else None)
            for item in nomination.history
        ]
        tombstones = [self.forgotten(item) for item in nomination.tombstones]
        anchors = [self.anchor(item.slot, [self.entity(candidate.handle_key)
                                           for candidate in item.candidates],
                               item.catalog_complete) for item in nomination.slots]
        quotes = self.quotes()
        handles = [quote.handle for quote in quotes]
        judged = JudgeMemberV8(
            statement=member.statement, kind=member.kind, time=judge_time(member.time),
            said_at=render_said_at(member.sigma[0]),
            naming=[JudgeNamingV8(slot=item.slot, name=item.name, name_kind=item.name_kind)
                    for item in member.naming],
            anchors=anchors, quotes=quotes,
            support_groups=[[handles[index] for index in group] for group in member.groups],
            context=[self.message(source) for source in nomination.sources if not source.cited],
            sensitivity_classes=list(member.sensitivity_classes),
            correction_requested=flags.correction_requested,
            corrects_text=None if member.cue is None else member.cue.text,
            revises=self.revises(), entity_partition_complete=flags.entity_partition_complete,
            external_only_allowed=flags.external_only_allowed,
            targets=targets, history=history, tombstones=tombstones,
        )
        catalog = dict(sorted(self.catalog.items()))
        digest = member_input_digest(judged, catalog, bind)
        # The one-member request checks that the catalog is closed and typed.
        JudgeBatchRequestV8(catalog=catalog, items=[
            JudgeBatchItemRequestV8(member_id=digest, member=judged),
        ])
        return MemberInputV1(
            position_key=member.position_key, order=(member.sigma, member.position_key),
            member=judged, catalog=catalog, identities=dict(sorted(self.identities.items())),
            digest=digest,
        )


# Physical requests -------------------------------------------------


def physical_request(members: Sequence[MemberInputV1]) -> JudgeBatchRequestV8:
    """Join member inputs into one physical request, each catalog entry once.

    Args:
        members (Sequence[MemberInputV1]): 1 to ``MAX_BATCH_MEMBERS`` members, in request
            order.

    Returns:
        JudgeBatchRequestV8: The shared catalog and one item per member, keyed by
            its ``member_input_digest``.

    Raises:
        QuoteMemError: ``binding_mismatch`` when two members show one handle differently.
    """
    catalog: dict[str, CatalogEntryV8] = {}
    for item in members:
        for handle, entry in item.catalog.items():
            if catalog.setdefault(handle, entry) != entry:
                raise QuoteMemError("binding_mismatch", "Two members show one handle differently")
    return JudgeBatchRequestV8(catalog=dict(sorted(catalog.items())), items=[
        JudgeBatchItemRequestV8(member_id=item.digest, member=item.member) for item in members
    ])


def request_input_bytes(request: JudgeBatchRequestV8) -> int:
    """Measure one physical request's input (``source_verification_batch.py:270-280``).

    Args:
        request (JudgeBatchRequestV8): The request.

    Returns:
        int: UTF-8 bytes of the canonical JSON of its system and user messages and
            its advertised response schema (``judge_request_schema``), plus 4,096.
    """
    system, user = judge_messages(request)
    schema = judge_request_schema(request)
    measured = canonical_json({
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "schema": schema,
    })
    return len(measured.encode("utf-8")) + REQUEST_MARGIN_BYTES


@dataclass(frozen=True, slots=True)
class JudgeRequestPlanV1:
    """One planned physical request: its members in canonical order and its measure.

    Attributes:
        members (tuple[MemberInputV1, ...]): 1 to ``MAX_BATCH_MEMBERS`` member inputs.
        request (JudgeBatchRequestV8): The user message's request.
        input_bytes (int): `request_input_bytes` of it.
        output_bytes (int): Its members' estimated reply bytes.
    """

    members: tuple[MemberInputV1, ...]
    request: JudgeBatchRequestV8
    input_bytes: int
    output_bytes: int


@dataclass(frozen=True, slots=True)
class PackingRuleV1:
    """How a round's members are laid into requests.

    Attributes:
        members (int): At most this many members a request.
        solo_referents (bool): Whether a member that chooses a referent goes alone.
        input_bytes (int | None): A request's input measure bound; None is
            ``MAX_REQUEST_INPUT_BYTES``.
        output_bytes (int | None): A request's estimated reply bound; None is
            ``MAX_REQUEST_OUTPUT_BYTES``.
    """

    members: int
    solo_referents: bool
    input_bytes: int | None = None
    output_bytes: int | None = None

    def limits(self, members: int) -> tuple[int, int]:
        """The input and output bounds of a request of this many members.

        A single member is never held to less than the bounds, so a member that
        fits a request alone is never ``review_capacity_exceeded``.
        """
        given = (MAX_REQUEST_INPUT_BYTES if self.input_bytes is None else self.input_bytes,
                 MAX_REQUEST_OUTPUT_BYTES if self.output_bytes is None else self.output_bytes)
        if members > 1:
            return given
        return max(given[0], MAX_REQUEST_INPUT_BYTES), max(given[1], MAX_REQUEST_OUTPUT_BYTES)


FULL_PACKING: Final = PackingRuleV1(MAX_BATCH_MEMBERS, True)
"""The full mode: four members a request, a member choosing a referent alone."""

BATCHED_PACKING: Final = PackingRuleV1(
    MAX_PACKET_MEMBERS, False, BATCHED_REQUEST_INPUT_BYTES, BATCHED_REQUEST_OUTPUT_BYTES,
)
"""The batched packing (``targeted-v1``'s model stage): one request a round where the bounds
allow, a member choosing a referent included; a request beyond them is bisected."""


@dataclass(frozen=True, slots=True)
class PackingV1:
    """One owner's round laid into physical requests.

    Attributes:
        requests (tuple[JudgeRequestPlanV1, ...]): Requests ordered by their first
            member's canonical key.
        unfit (tuple[MemberInputV1, ...]): Members that do not fit a request even
            alone: ``review_capacity_exceeded``.
    """

    requests: tuple[JudgeRequestPlanV1, ...]
    unfit: tuple[MemberInputV1, ...]


def _joins(group: Sequence[MemberInputV1], item: MemberInputV1, rule: PackingRuleV1) -> bool:
    """Whether a member may join a request: room, a new member id, and no handle that the
    request already gives another object or another entry."""
    if len(group) >= rule.members:
        return False
    for member in group:
        if member.digest == item.digest or any(
            member.identities.get(handle, key) != key for handle, key in item.identities.items()
        ) or any(
            member.catalog.get(handle, entry) != entry for handle, entry in item.catalog.items()
        ):
            return False
    return True


def _place(group: list[MemberInputV1], requests: list[JudgeRequestPlanV1],
           unfit: list[MemberInputV1], rule: PackingRuleV1) -> None:
    """Keep a request within the rule's bounds, bisecting it in order until it fits."""
    request = physical_request(group)
    input_bytes = request_input_bytes(request)
    output_bytes = sum(item.output_bytes for item in group)
    limits = rule.limits(len(group))
    if input_bytes <= limits[0] and output_bytes <= limits[1]:
        requests.append(JudgeRequestPlanV1(tuple(group), request, input_bytes, output_bytes))
    elif len(group) == 1:
        unfit.append(group[0])
    else:
        middle = len(group) // 2
        _place(group[:middle], requests, unfit, rule)
        _place(group[middle:], requests, unfit, rule)


def pack_requests(inputs: Iterable[MemberInputV1], rule: PackingRuleV1 = FULL_PACKING,
                  ) -> PackingV1:
    """Lay one owner's round of member inputs into physical requests.

    Members are taken in canonical order. Under the full rule a member that
    chooses a referent gets a request of its own; every other member joins the
    open request when it has room and no handle clash, else it opens the next
    request. Each request is then bisected in order until it fits the rule's bounds.

    Args:
        inputs (Iterable[MemberInputV1]): The round's member inputs, in any order.
        rule (PackingRuleV1): ``FULL_PACKING`` or the batched packing's ``BATCHED_PACKING``.

    Returns:
        PackingV1: The requests, ordered by their first member, and the members
            that fit no request.

    Raises:
        QuoteMemError: ``binding_mismatch`` when a position appears twice.

    Examples:
        ```python
        inputs = [build_member_input(nomination, view, bind=keyring.digest)
                  for nomination in nominations]
        for planned in pack_requests(inputs).requests:
            system, user = judge_messages(planned.request)
        ```
    """
    ordered = sorted(inputs, key=lambda item: item.order)
    if len({item.position_key for item in ordered}) != len(ordered):
        raise QuoteMemError("binding_mismatch", "A position is packed twice")
    groups: list[list[MemberInputV1]] = []
    current: list[MemberInputV1] | None = None
    for item in ordered:
        if rule.solo_referents and item.chooses_referent:
            groups.append([item])
        elif current is not None and _joins(current, item, rule):
            current.append(item)
        else:
            current = [item]
            groups.append(current)
    groups.sort(key=lambda group: group[0].order)
    requests: list[JudgeRequestPlanV1] = []
    unfit: list[MemberInputV1] = []
    for group in groups:
        _place(group, requests, unfit, rule)
    return PackingV1(tuple(requests), tuple(unfit))
