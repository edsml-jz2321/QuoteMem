"""Rendered extraction sources, their turn labels and speaker attribution (B.4.1, B.3a).

A source is an attributed run of text: a maximal contiguous run of one
message's parts with one role and one attribution (B.4.1 as amended on
2026-10-16). Every part of a message shares its role and speaker metadata, so a
message is cut only where a turn label changes the attribution. An
``ambiguous`` label always opens a source, as it opens a turn whose speaker
nobody may infer. A ``speaker`` label opens one unless it names the speaker of
the source in progress, which it then continues, so a date line followed by a
label of the metadata speaker stays one source. Each source keeps the one label
that governs it, and a source never mixes two speakers.

Speaker attribution follows B.4.1 (PD:P6; labels and metadata must agree):

- the metadata speaker, when no label governs the source or the governing
  label names that speaker;
- else the governing ``speaker`` label's name;
- an ``ambiguous`` label leaves the speaker null and carries no identity;
- a label that disagrees with the metadata speaker leaves the speaker null and
  sets ``speaker_conflict`` (INV O06).

A ``speaker`` label written ``User:`` or ``The user:`` names the authenticated
user rather than a third party, so it neither blocks authenticated-self speech
nor becomes a labelled participant.

Text is rendered only by ``Readability.render_masked`` (D.2): unreadable bytes
become ``[withheld]`` and the segment map relates rendered characters to
message bytes, which quote location (B.11) needs.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from itertools import groupby
from typing import Literal, Protocol

from aml.core.add_read_view import EvidenceRecord, MessageKey
from aml.core.readability import MaskedText, Readability
from aml.core.semantic import TimeGranularity
from aml.core.statement import UNSPACED_SCRIPTS, label_speaker_key, name_norm, participant_key

RENDER_VERSION = "nl-sources-v2"
"""The render version of B.0: how sources are cut, attributed and rendered.

Version 2 cuts a message only where a turn label changes the attribution
(B.4.1 as amended on 2026-10-16).
"""

GENERIC_USER_FORMS = frozenset({"user", "the user"})
"""Label names (``name_norm``) that designate the authenticated user (B.4.2 (1))."""

LabelState = Literal["speaker", "ambiguous"]
Batch = Literal["previous", "tail", "current"]
WireRole = Literal["user", "assistant", "tool", "system", "unknown"]
ParticipantKind = Literal["tuple", "label"]


@dataclass(frozen=True, slots=True)
class TurnLabel:
    """A proven, unquoted line label that opens a turn (B.3a ``SourceLabelView``).

    ``content`` and quoted labels never open a turn and are not turn labels.

    Attributes:
        byte_start (int): First message byte of the label: its envelope header,
            if one opens the line, else its name.
        byte_end (int): Message byte after the label's colon.
        name (str): The label's name as written.
        state (LabelState): ``speaker`` opens that speaker's turn; ``ambiguous``
            blocks every inheritance of speaker identity.
        framed (bool): An envelope header opens the label's line, so the turn
            is a transcript line of the named participant (``core/envelope.py``).
    """

    byte_start: int
    byte_end: int
    name: str
    state: LabelState
    framed: bool = False

    @property
    def generic_user(self) -> bool:
        """Whether this ``speaker`` label designates the authenticated user."""
        return self.state == "speaker" and name_norm(self.name) in GENERIC_USER_FORMS


class LabelView(Protocol):
    """The frozen turn labels of the rendered messages (B.3a, B.4.4 ``labels``)."""

    def turns(self, message: MessageKey) -> Sequence[TurnLabel]:
        """Return a message's turn labels ordered by ``byte_start``.

        Args:
            message (MessageKey): The message.

        Returns:
            Sequence[TurnLabel]: Its turn labels; empty when it has none.
        """
        ...


@dataclass(frozen=True, slots=True)
class MessageLabels:
    """A plain `LabelView` over turn labels keyed by message.

    Attributes:
        labels (Mapping[MessageKey, tuple[TurnLabel, ...]]): Turn labels per
            message; a missing message has none.
    """

    labels: Mapping[MessageKey, tuple[TurnLabel, ...]] = field(
        default_factory=dict[MessageKey, tuple[TurnLabel, ...]],
    )

    def turns(self, message: MessageKey) -> Sequence[TurnLabel]:
        """Return the message's turn labels ordered by ``byte_start``."""
        return tuple(sorted(self.labels.get(message, ()), key=lambda label: label.byte_start))


@dataclass(frozen=True, slots=True)
class SourceRun:
    """One source before rendering: a byte range of one message and its governing label.

    Attributes:
        message (MessageKey): The message.
        parts (tuple[EvidenceRecord, ...]): The parts the range covers, by
            ``part_no``, in any state.
        byte_start (int): First message byte of the source.
        byte_end (int): Message byte after the source.
        label (TurnLabel | None): The turn label that governs the source: the
            one that opened it, or for the first source of a range the latest
            label at or before ``byte_start``; None when no label governs it.
            A later label that continues the source does not replace it.
    """

    message: MessageKey
    parts: tuple[EvidenceRecord, ...]
    byte_start: int
    byte_end: int
    label: TurnLabel | None

    @property
    def head(self) -> EvidenceRecord:
        """The first part that is not erased; it carries the message metadata."""
        return next(part for part in self.parts if part.state != "erased")


def message_key(part: EvidenceRecord) -> MessageKey:
    """Return the message an Evidence part belongs to."""
    return MessageKey(part.session_id, part.request_id, part.ordinal)


def governing_label(turns: Sequence[TurnLabel], position: int) -> TurnLabel | None:
    """Return the latest turn label at or before a message byte, or None.

    Args:
        turns (Sequence[TurnLabel]): A message's turn labels by ``byte_start``.
        position (int): The message byte.

    Returns:
        TurnLabel | None: The label in force there.
    """
    governing: TurnLabel | None = None
    for label in turns:
        if label.byte_start > position:
            break
        governing = label
    return governing


def opens_source(head: EvidenceRecord, current: TurnLabel | None, label: TurnLabel) -> bool:
    """Tell whether a turn label inside a source in progress opens a new source (B.4.1).

    An ``ambiguous`` label always does. A ``speaker`` label does when it changes
    the attribution of the source in progress: its speaker under ``name_norm``,
    the conflict flag, the authenticated-self flag or the kind of participant.

    Args:
        head (EvidenceRecord): A readable part of the message; it carries the
            message's role and speaker metadata.
        current (TurnLabel | None): The label governing the source in progress.
        label (TurnLabel): The label inside it.

    Returns:
        bool: True when the label opens a source, False when it continues one.
    """
    if label.state == "ambiguous":
        return True
    return _identity(_settle(head, current)) != _identity(_settle(head, label))


def source_runs(parts: Sequence[EvidenceRecord], labels: LabelView) -> list[SourceRun]:
    """Cut whole parts into sources: one run per message, cut where a label opens one (B.4.1).

    Args:
        parts (Sequence[EvidenceRecord]): Whole parts in message order; the parts
            of one message are consecutive.
        labels (LabelView): The frozen turn labels.

    Returns:
        list[SourceRun]: The runs in order. A message whose parts are all erased
            has no bytes to render and gives no run.
    """
    runs: list[SourceRun] = []
    for key, group in groupby(parts, key=message_key):
        members = tuple(group)
        placed = [part for part in members if part.state != "erased"]
        if not placed:
            continue
        start = min(part.byte_start for part in placed)
        end = max(part.byte_end for part in placed)
        turns = labels.turns(key)
        opened: list[tuple[int, TurnLabel | None]] = [(start, governing_label(turns, start))]
        for label in turns:
            if start < label.byte_start < end and opens_source(placed[0], opened[-1][1], label):
                opened.append((label.byte_start, label))
        ends = [low for low, _ in opened[1:]] + [end]
        for (low, label), high in zip(opened, ends, strict=True):
            covered = tuple(
                part for part in members
                if part.state == "erased" or part.byte_start < high and low < part.byte_end
            )
            runs.append(SourceRun(key, covered, low, high, label))
    return runs


@dataclass(frozen=True, slots=True)
class Attribution:
    """Who speaks a source (B.4.1, B.4.2, D.9).

    Attributes:
        speaker (str | None): The speaker name shown to the extractor.
        speaker_conflict (bool): A label disagrees with the metadata speaker.
        authenticated_self (bool): The authenticated user speaks the source:
            role ``user``, no participant tuple, no metadata speaker, no label
            other than a generic user label and no conflict (B.4.2 (1)).
        participant_key (str | None): The participant the source speaks as: its
            source tuple, or a labelled speaker keyed ``aml-label-speaker-v1``.
        participant_kind (ParticipantKind | None): ``tuple`` or ``label``.
        names (tuple[str, ...]): Speaker names this source shows for its
            participant (metadata speaker and agreeing label), for
            ``speaker_names`` (B.4.2).
        label (str | None): The governing label's name, else None.
        metadata_speaker (str | None): The metadata speaker, else None.
    """

    speaker: str | None
    speaker_conflict: bool
    authenticated_self: bool
    participant_key: str | None
    participant_kind: ParticipantKind | None
    names: tuple[str, ...]
    label: str | None
    metadata_speaker: str | None


def attribute(run: SourceRun, scope_id: str) -> Attribution:
    """Attribute one source to its speaker (B.4.1, B.4.2, D.9).

    Args:
        run (SourceRun): The source.
        scope_id (str): The scope; labelled speakers are keyed within it.

    Returns:
        Attribution: The speaker, conflict flag, authenticated-self flag and the
            participant the source speaks as.
    """
    first = run.head
    metadata = first.speaker_entity
    participant = first.source_participant
    label = run.label
    settled = _settle(first, label)
    key: str | None = None
    names: tuple[str, ...] = ()
    if settled.kind == "tuple" and participant is not None:
        key = participant_key(
            participant.namespace, participant.conversation_id, participant.participant_id,
        )
        names = tuple(dict.fromkeys(
            name for name in (metadata, label.name if label else None) if name
        ))
    elif settled.kind == "label" and label is not None:
        key = label_speaker_key(scope_id, label.name)
        names = tuple(dict.fromkeys(name for name in (label.name, metadata) if name))
    return Attribution(
        speaker=settled.speaker,
        speaker_conflict=settled.conflict,
        authenticated_self=settled.authenticated,
        participant_key=key,
        participant_kind=settled.kind,
        names=names,
        label=label.name if label is not None else None,
        metadata_speaker=metadata,
    )


@dataclass(frozen=True, slots=True)
class _Settled:
    """Who speaks under one governing label, before keys and names are derived."""

    speaker: str | None
    conflict: bool
    authenticated: bool
    kind: ParticipantKind | None


def _settle(head: EvidenceRecord, label: TurnLabel | None) -> _Settled:
    """Settle the speaker a governing label leaves a source (B.4.1, B.4.2 (1), D.9)."""
    metadata = head.speaker_entity
    conflict = False
    speaker: str | None = metadata
    if label is not None and label.state == "ambiguous":
        speaker = None
    elif label is not None and metadata is not None:
        if name_norm(label.name) != name_norm(metadata):
            speaker, conflict = None, True
    elif label is not None and not label.generic_user:
        speaker = label.name
    blocked = label is not None and (label.state == "ambiguous" or not label.generic_user)
    authenticated = (
        head.role == "user" and head.source_participant is None and metadata is None
        and not conflict and not blocked
    )
    kind: ParticipantKind | None = None
    if head.source_participant is not None:
        if not conflict and (label is None or label.state == "speaker" and metadata is not None):
            kind = "tuple"
    elif label is not None and label.state == "speaker" and not label.generic_user and (
        not conflict
    ):
        kind = "label"
    return _Settled(speaker, conflict, authenticated, kind)


def _identity(settled: _Settled) -> tuple[str | None, bool, bool, ParticipantKind | None]:
    """What two attributions are compared by: the speaker under ``name_norm``, not as written."""
    speaker = None if settled.speaker is None else name_norm(settled.speaker)
    return speaker, settled.conflict, settled.authenticated, settled.kind


@dataclass(frozen=True, slots=True)
class RenderedSource:
    """One source as the extractor sees it, with the program-side facts admission reads.

    Attributes:
        source_id (str): ``S1``..``S600`` in request order.
        batch (Batch): ``previous``, ``tail`` or ``current``; only ``current``
            sources are fresh.
        run (SourceRun): The message range it renders.
        role (WireRole): The message role; a null role is ``unknown``.
        attribution (Attribution): Who speaks it.
        observed_at (int | None): The message's ``source_observed_at``.
        granularity (TimeGranularity): The granularity of ``observed_at``.
        masked (MaskedText): The rendered text and its segment map.
    """

    source_id: str
    batch: Batch
    run: SourceRun
    role: WireRole
    attribution: Attribution
    observed_at: int | None
    granularity: TimeGranularity
    masked: MaskedText

    @property
    def fresh(self) -> bool:
        """Whether the source belongs to the unit itself; only these originate facts."""
        return self.batch == "current"

    @property
    def text(self) -> str:
        """The rendered text, ``[withheld]`` markers included."""
        return self.masked.text

    def readable_texts(self) -> tuple[str, ...]:
        """Return the maximal readable stretches of the text, without any marker.

        Name and time containment read these, so a name never matches across
        withheld text or a marker's own letters.
        """
        return tuple(self.text[start:end] for start, end in self.readable_ranges())

    def readable_ranges(self) -> tuple[tuple[int, int], ...]:
        """Return the rendered character range of each maximal readable stretch."""
        ranges: list[tuple[int, int]] = []
        start = end = -1
        for segment in self.masked.segments:
            if segment.rendered_char_start != end:
                if end > start:
                    ranges.append((start, end))
                start = segment.rendered_char_start
            end = segment.rendered_char_end
        if end > start:
            ranges.append((start, end))
        return tuple(ranges)

    def rendered_bytes(self, evidence_id: str, byte_start: int, byte_end: int) -> str | None:
        """Return the rendered text of a message byte range that is wholly readable.

        Args:
            evidence_id (str): The part the range lies in.
            byte_start (int): First message byte, inclusive.
            byte_end (int): Message byte after the range.

        Returns:
            str | None: The text, or None when some byte of the range is not
                rendered as written (withheld, outside the source, or another
                part).
        """
        for segment in self.masked.segments:
            if segment.evidence_id == evidence_id and (
                segment.msg_byte_start <= byte_start < byte_end <= segment.msg_byte_end
            ):
                data = self.text[segment.rendered_char_start:segment.rendered_char_end].encode()
                offset = segment.msg_byte_start
                return data[byte_start - offset:byte_end - offset].decode()
        return None


def render_source(
    run: SourceRun, *, batch: Batch, source_id: str, read: Readability, scope_id: str,
) -> RenderedSource | None:
    """Render one source through ``render_masked`` (D.2, B.4.1).

    Args:
        run (SourceRun): The source.
        batch (Batch): Its batch.
        source_id (str): Its id.
        read (Readability): The read the context selects under.
        scope_id (str): The scope, for labelled-speaker keys.

    Returns:
        RenderedSource | None: The source, or None when no byte of it is
            readable, since a source made only of markers says nothing.
    """
    masked = read.render_masked(run.parts, start=run.byte_start, end=run.byte_end)
    if not masked.segments:
        return None
    first = run.head
    return RenderedSource(
        source_id=source_id,
        batch=batch,
        run=run,
        role=first.role or "unknown",
        attribution=attribute(run, scope_id),
        observed_at=first.source_observed_at,
        granularity=first.time_granularity,
        masked=masked,
    )


def tail_runs(runs: Sequence[SourceRun], *, max_sources: int, max_words: int) -> list[SourceRun]:
    """Select the same-request tail: the newest runs before a unit (B.3).

    The tail keeps whole runs newest first until ``max_sources`` runs, and cuts
    the oldest kept run at a word start so that at most ``max_words`` words
    remain (``segment.py:621-642``).

    Args:
        runs (Sequence[SourceRun]): The runs of the request's parts before the
            unit, in order.
        max_sources (int): The most tail sources (6).
        max_words (int): The most tail words (600).

    Returns:
        list[SourceRun]: The tail runs in order.
    """
    selected: list[SourceRun] = []
    words = 0
    for run in reversed(runs):
        if len(selected) == max_sources or words == max_words:
            break
        pieces = _pieces(run)
        text = "".join(piece for _, piece in pieces)
        starts = sorted(word_starts(text))
        if len(starts) > max_words - words:
            cut = starts[-(max_words - words)]
            run = replace(run, byte_start=_byte_at(pieces, cut))
            text = text[cut:]
        words += word_count(text)
        selected.append(run)
    return list(reversed(selected))


def _pieces(run: SourceRun) -> list[tuple[int, str]]:
    """Return the raw text of a run's range as ``(message byte, text)`` pieces per part."""
    pieces: list[tuple[int, str]] = []
    for part in run.parts:
        low = max(run.byte_start, part.byte_start)
        high = min(run.byte_end, part.byte_end)
        if part.state != "erased" and low < high:
            data = part.text.encode("utf-8")
            pieces.append((low, data[low - part.byte_start:high - part.byte_start].decode()))
    return pieces


def _byte_at(pieces: Sequence[tuple[int, str]], index: int) -> int:
    """Return the message byte of the character at ``index`` of the joined pieces."""
    for start, text in pieces:
        if index < len(text):
            return start + len(text[:index].encode("utf-8"))
        index -= len(text)
    raise ValueError("Character index is outside the run")


_UNSPACED = re.compile(f"[{UNSPACED_SCRIPTS}]")


def _ideograph(char: str) -> bool:
    point = ord(char)
    return 0x3400 <= point <= 0x9FFF or 0x20000 <= point <= 0x323AF


def word_starts(text: str) -> set[int]:
    """Return the character offsets at which words start (``segment.py:409-422``; ML-1).

    A word is a run of letters and digits other than ideographs and letters of
    an unspaced Southeast Asian script; each ideograph, each such letter and each
    other non-space character is a word of its own. A combining mark (Unicode
    category M) after a letter or digit belongs to it and starts no word, so a
    Hindi word is one word, a Thai cluster one, and no cut at a word start
    separates a mark from its letter; a mark after anything else (an emoji's
    variation selector) is a word of its own, as before.
    """
    starts: set[int] = set()
    in_word = attached = False
    for index, char in enumerate(text):
        if attached and unicodedata.category(char)[0] == "M":
            continue
        unspaced = char.isalnum() and _UNSPACED.match(char) is not None
        ordinary = char.isalnum() and not _ideograph(char) and not unspaced
        if ordinary and not in_word or not ordinary and not char.isspace():
            starts.add(index)
        in_word = ordinary
        attached = ordinary or unspaced
    return starts


def word_count(text: str) -> int:
    """Count words as units and tails bound them (``segment.py:425-427``)."""
    return len(word_starts(text))
