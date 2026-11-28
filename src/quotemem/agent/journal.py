"""The source journal: a release-independent JSON Lines record of an agent store.

An agent store is upgraded across releases by rebuilding it from its own verbatim
evidence, never by migrating internal tables (``quotemem export`` and ``quotemem
rebuild``). The journal is the hand-off between the two: it holds what the user
said, in the order the store received it, and what the user asked to forget, and
nothing derived (no memory, vector, judge record or internal identity).

Format ``quotemem-source-journal``, version 2 (a reader also reads version 1, which
has no ``retired`` ranges). UTF-8, one JSON object per line, keys sorted. Lines in
order:

1. ``header``: ``format``, ``version``, ``exported_at_ms``, ``aml_version``,
   ``store`` (``embedding_model``, ``embedding_dimensions``, ``add_mode`` of the
   exported store; informational, a rebuild may choose others) and ``encryption``
   (``scheme`` ``aes-256-gcm``, the sealing key's ``key_id`` fingerprint and, for
   a passphrase key, its ``kdf``: scrypt salt and cost; ``quotemem.agent.journal_key``).
2. ``add`` records, in the order the store committed them. Each is one Add
   request as stored: ``user_id``, ``session_id``, ``request_id``,
   ``ingested_at_ms`` and ``messages`` (``role``, ``content``, and when present
   ``timestamp`` in Unix milliseconds, ``speaker`` and ``source_participant``).
   Bytes a forget withheld are not in ``content``: each withheld range, and each
   part a privacy erasure emptied, is replaced by the read path's withheld marker
   (``quotemem.core.readability.WITHHELD_MARKER``). ``withheld``
   lists the withheld ranges by coordinates only (``ordinal``, the message's
   position among the stored messages, and ``byte_start``/``byte_end`` in the
   original message's UTF-8 bytes, with the ``reason``), and ``erased_parts``
   counts the erased parts. ``retired`` lists the quoted words of each memory the
   store held ended (superseded, corrected or retracted) whose successor no longer
   exists, because the user asked to forget it or its source was erased: the
   ``message`` (its position in ``messages``), ``byte_start``/``byte_end`` in that
   message's ``content`` as written here, how it ``ended`` (``update``,
   ``correct`` or ``retract``), and ``after``: the ``session_id`` and
   ``request_id`` of the last Add of the same user the store had taken in before
   the memory ended (this Add itself or a later one). The successor's words are
   withheld, so a replay alone would make such a memory current again; a rebuild
   ends the memories that quote these words right after replaying that Add, so the
   later Adds meet it ended as the store did (``quotemem.api.memory_retire``). Words a
   memory the store still holds current also quotes are not named: they cannot
   tell the two apart. Coordinates only: the words are the message's own, already
   in ``content``.
3. ``forget`` records, in the order the store carried them out: ``user_id``,
   ``at_ms``, ``statement_count`` and ``sealed``, the statements of the memories
   one forget operation retired, as a JSON list sealed with AES-256-GCM under the
   header's key (associated data: format, version, key id, user, time and count).
   A rebuild opens them and forgets them again at their place in time among the
   Adds (before the first Add ingested after ``at_ms``), so a restatement the user
   made after a forget meets that forget as it did in the store; each statement,
   whether the replay still finds it or not, is tombstoned over the withheld markers
   (``AgentStore.forget_withheld``). After every Add a safety pass forgets each one
   again over the Evidence of the Adds ingested at or before it alone
   (``AgentStore.forget_earlier``): a current memory the replay drew again from what
   was said before a forget is forgotten again, without a vote when its words or the
   forget shadow match it, else when the judge finds it states the fact. They are the
   only forgotten content
   the journal holds, and never appear in clear; the file is still written with
   mode 0600, since the Adds are the user's own words.
4. ``end``: the record counts and ``sha256``, the SHA-256 of every byte before
   the ``end`` line. A journal without a matching ``end`` is incomplete.

A reader rejects an unknown ``format``, a newer ``version`` and an unknown record
type, so a journal is never half understood.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Final, Literal, cast

from pydantic import ConfigDict, Field, ValidationError

from quotemem.agent.journal_key import JournalEncryptionV1, JournalKey, JournalSealedV1
from quotemem.core.errors import QuoteMemError
from quotemem.core.platform_files import open_private_file
from quotemem.core.readability import WITHHELD_MARKER as WITHHELD_MARKER
from quotemem.store.metadata import StrictModel

JOURNAL_FORMAT: Final = "quotemem-source-journal"
JOURNAL_VERSION: Final = 2
READABLE_VERSIONS: Final = frozenset({1, 2})
"""The versions a reader opens; version 1 has no ``retired`` ranges."""


class _Record(StrictModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class JournalStoreV1(_Record):
    """What the exported store was bound to; a rebuild may bind differently."""

    embedding_model: str | None = None
    embedding_dimensions: Literal[1024, 2048] | None = None
    add_mode: str | None = None


class JournalHeaderV1(_Record):
    """The first line."""

    type: Literal["header"] = "header"
    format: Literal["quotemem-source-journal"] = JOURNAL_FORMAT
    version: Literal[1, 2] = JOURNAL_VERSION
    exported_at_ms: int
    aml_version: str
    store: JournalStoreV1
    encryption: JournalEncryptionV1


class JournalMessageV1(_Record):
    """One stored message, with withheld bytes replaced."""

    role: Literal["user", "assistant", "system"]
    content: Annotated[str, Field(min_length=1)]
    timestamp: int | None = None
    speaker: str | None = None
    source_participant: dict[str, Any] | None = None


class JournalWithheldV1(_Record):
    """One withheld range, by coordinates only."""

    ordinal: Annotated[int, Field(ge=0)]
    byte_start: Annotated[int, Field(ge=0)]
    byte_end: Annotated[int, Field(ge=0)]
    reason: str


class JournalAddRefV1(_Record):
    """One Add of the same user, by its session and request ids."""

    session_id: Annotated[str, Field(min_length=1)]
    request_id: Annotated[str, Field(min_length=1)]


class JournalRetiredV1(_Record):
    """Quoted words of a memory that ended without a surviving successor, and the
    Add of the same user after which it had ended."""

    message: Annotated[int, Field(ge=0)]
    byte_start: Annotated[int, Field(ge=0)]
    byte_end: Annotated[int, Field(ge=1)]
    ended: Literal["update", "correct", "retract"]
    after: JournalAddRefV1


class JournalAddV1(_Record):
    """One Add request as the store holds it."""

    type: Literal["add"] = "add"
    user_id: Annotated[str, Field(min_length=1)]
    session_id: Annotated[str, Field(min_length=1)]
    request_id: Annotated[str, Field(min_length=1)]
    ingested_at_ms: int
    messages: Annotated[list[JournalMessageV1], Field(min_length=1)]
    withheld: list[JournalWithheldV1] = Field(default_factory=list[JournalWithheldV1])
    erased_parts: Annotated[int, Field(ge=0)] = 0
    retired: list[JournalRetiredV1] = Field(default_factory=list[JournalRetiredV1])


class JournalForgetV1(_Record):
    """One forget operation as written: the statements it retired, sealed."""

    type: Literal["forget"] = "forget"
    user_id: Annotated[str, Field(min_length=1)]
    at_ms: int
    statement_count: Annotated[int, Field(ge=1)]
    sealed: JournalSealedV1


@dataclass(frozen=True)
class Forget:
    """One forget operation in clear; it exists only in memory and is sealed on write."""

    user_id: str
    at_ms: int
    statements: tuple[str, ...]


class JournalCountsV1(_Record):
    """Record and content counts of one journal."""

    adds: Annotated[int, Field(ge=0)] = 0
    forgets: Annotated[int, Field(ge=0)] = 0
    messages: Annotated[int, Field(ge=0)] = 0
    content_bytes: Annotated[int, Field(ge=0)] = 0
    withheld_spans: Annotated[int, Field(ge=0)] = 0
    erased_parts: Annotated[int, Field(ge=0)] = 0
    forget_statements: Annotated[int, Field(ge=0)] = 0
    retired_ranges: Annotated[int, Field(ge=0)] = 0


class JournalEndV1(_Record):
    """The last line: counts and the digest of every line before it."""

    type: Literal["end"] = "end"
    counts: JournalCountsV1
    sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


type JournalEntry = JournalAddV1 | JournalForgetV1
"""A record as the file holds it."""
type ReplayEntry = JournalAddV1 | Forget
"""A record with its forgotten statements open."""


def _associated(
    key_id: str, user_id: str, at_ms: int, statement_count: int, version: int = JOURNAL_VERSION,
) -> bytes:
    """What a sealed forget is bound to, besides its key: ``version`` is its journal's."""
    return json.dumps({
        "at_ms": at_ms, "format": JOURNAL_FORMAT, "key_id": key_id,
        "statement_count": statement_count, "user_id": user_id, "version": version,
    }, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def seal_forget(key: JournalKey, forget: Forget) -> JournalForgetV1:
    """``forget`` with its statements sealed under ``key``."""
    if not forget.statements or not all(forget.statements):
        raise QuoteMemError("schema_invalid", "A forget needs at least one nonempty statement")
    plaintext = json.dumps(list(forget.statements), ensure_ascii=False).encode("utf-8")
    return JournalForgetV1(
        user_id=forget.user_id, at_ms=forget.at_ms, statement_count=len(forget.statements),
        sealed=key.seal(plaintext, _associated(
            key.key_id, forget.user_id, forget.at_ms, len(forget.statements))),
    )


def open_forget(
    key: JournalKey, record: JournalForgetV1, version: int = JOURNAL_VERSION,
) -> Forget:
    """The statements of a sealed forget written in a journal of ``version``.

    Raises:
        QuoteMemError: ``parity_mismatch`` when the record fails authentication or its
            statements differ from its count.
    """
    plaintext = key.open(record.sealed, _associated(
        key.key_id, record.user_id, record.at_ms, record.statement_count, version))
    try:
        value: object = json.loads(plaintext)
    except ValueError:
        value = None
    if not isinstance(value, list):
        raise QuoteMemError("parity_mismatch", "A sealed forget does not hold its statements")
    statements = tuple(cast(list[object], value))
    if len(statements) != record.statement_count or not all(
        isinstance(statement, str) and statement for statement in statements
    ):
        raise QuoteMemError("parity_mismatch", "A sealed forget does not hold its statements")
    return Forget(record.user_id, record.at_ms, cast(tuple[str, ...], statements))


def count(entries: Iterable[JournalEntry]) -> JournalCountsV1:
    """The counts of ``entries``."""
    adds = forgets = messages = content = withheld = erased = statements = retired = 0
    for entry in entries:
        if isinstance(entry, JournalAddV1):
            adds += 1
            messages += len(entry.messages)
            content += sum(len(message.content.encode("utf-8")) for message in entry.messages)
            withheld += len(entry.withheld)
            erased += entry.erased_parts
            retired += len(entry.retired)
        else:
            forgets += 1
            statements += entry.statement_count
    return JournalCountsV1(
        adds=adds, forgets=forgets, messages=messages, content_bytes=content,
        withheld_spans=withheld, erased_parts=erased, forget_statements=statements,
        retired_ranges=retired,
    )


def _line(record: _Record) -> bytes:
    return (json.dumps(
        record.model_dump(mode="json", exclude_none=True), ensure_ascii=False, sort_keys=True,
        separators=(",", ":"),
    ) + "\n").encode("utf-8")


def write_journal(
    path: Path, header: JournalHeaderV1, entries: Iterable[ReplayEntry], key: JournalKey,
) -> JournalEndV1:
    """Write a complete journal atomically; never replaces an existing file.

    Every forget is sealed under ``key``, which ``header.encryption`` must name.
    The file is staged beside ``path`` (mode 0600), synced and linked into place.

    Returns:
        The ``end`` record written.

    Raises:
        QuoteMemError: ``operation_conflict`` when ``path`` exists; ``binding_mismatch``
            when the header names another key.
        OSError: When the file cannot be written.
    """
    if header.encryption != key.encryption:
        raise QuoteMemError("binding_mismatch", "The journal header names another key")
    if path.exists() or path.is_symlink():
        raise QuoteMemError("operation_conflict", "The journal file already exists; name a new one")
    staged = path.with_name(f".{path.name}.{os.getpid()}.partial")
    if staged.exists() or staged.is_symlink():
        staged.unlink()
    digest = hashlib.sha256()
    kept: list[JournalEntry] = []
    descriptor = open_private_file(staged, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            for item in (header, *entries):
                record = seal_forget(key, item) if isinstance(item, Forget) else item
                line = _line(record)
                digest.update(line)
                stream.write(line)
                if not isinstance(record, JournalHeaderV1):
                    kept.append(record)
            end = JournalEndV1(counts=count(kept), sha256=digest.hexdigest())
            stream.write(_line(end))
            stream.flush()
            os.fsync(stream.fileno())
        # A hard link fails when ``path`` appeared meanwhile, so nothing is replaced.
        os.link(staged, path)
    finally:
        staged.unlink(missing_ok=True)
    return end


@dataclass(frozen=True)
class Journal:
    """A complete, verified journal."""

    header: JournalHeaderV1
    entries: tuple[JournalEntry, ...]
    end: JournalEndV1

    @property
    def sha256(self) -> str:
        """The digest that identifies this journal's content."""
        return self.end.sha256

    @property
    def sealed(self) -> bool:
        """Whether any record needs the key to be read."""
        return any(isinstance(entry, JournalForgetV1) for entry in self.entries)

    def open(self, key: JournalKey | None) -> tuple[ReplayEntry, ...]:
        """Every record, with the forgotten statements opened under ``key``.

        ``key`` may be None only for a journal without forgets.

        Raises:
            QuoteMemError: ``input_required`` when the journal needs a key and has none;
                ``secret_rejected`` when ``key`` is not the header's;
                ``parity_mismatch`` when a sealed record fails authentication.
        """
        if key is None:
            if self.sealed:
                raise QuoteMemError("input_required", "The journal's forgets need its key")
        elif key.key_id != self.header.encryption.key_id:
            raise QuoteMemError("secret_rejected", "The key is not this journal's key")
        return tuple(
            open_forget(cast(JournalKey, key), entry, self.header.version)
            if isinstance(entry, JournalForgetV1)
            else entry
            for entry in self.entries
        )


def _parse(raw: bytes) -> dict[str, Any]:
    try:
        value: object = json.loads(raw)
    except (ValueError, RecursionError):
        raise QuoteMemError("schema_invalid", "A journal line is not JSON") from None
    if not isinstance(value, dict):
        raise QuoteMemError("schema_invalid", "A journal line is not a JSON object")
    return cast(dict[str, Any], value)


def _lines(path: Path) -> Iterator[bytes]:
    descriptor = open_private_file(path, os.O_RDONLY)
    with os.fdopen(descriptor, "rb") as stream:
        yield from stream


def read_journal(path: Path) -> Journal:
    """Read and verify a journal.

    Raises:
        QuoteMemError: ``schema_invalid`` for an unknown format, a newer version, an
            unknown record, a malformed line or a journal without its ``end``;
            ``parity_mismatch`` when the digest or counts differ.
        OSError: When the file cannot be read or is not private.
    """
    digest = hashlib.sha256()
    header: JournalHeaderV1 | None = None
    entries: list[JournalEntry] = []
    end: JournalEndV1 | None = None
    try:
        for raw in _lines(path):
            if end is not None:
                raise QuoteMemError("schema_invalid", "The journal continues after its end")
            fields = _parse(raw)
            kind = fields.get("type")
            if header is None:
                if kind != "header" or fields.get("format") != JOURNAL_FORMAT:
                    raise QuoteMemError("schema_invalid", "This is not a QuoteMem source journal")
                if fields.get("version") not in READABLE_VERSIONS:
                    raise QuoteMemError(
                        "schema_invalid", "The journal has a version this release cannot read",
                    )
                header = JournalHeaderV1.model_validate_json(raw)
            elif kind == "add":
                entries.append(JournalAddV1.model_validate_json(raw))
            elif kind == "forget":
                entries.append(JournalForgetV1.model_validate_json(raw))
            elif kind == "end":
                end = JournalEndV1.model_validate_json(raw)
                continue
            else:
                raise QuoteMemError("schema_invalid", "The journal has an unknown record type")
            digest.update(raw)
    except ValidationError:
        raise QuoteMemError("schema_invalid", "A journal record is malformed") from None
    if header is None or end is None:
        raise QuoteMemError("schema_invalid", "The journal is incomplete: it has no end record")
    if end.sha256 != digest.hexdigest() or end.counts != count(entries):
        raise QuoteMemError("parity_mismatch", "The journal differs from its end record")
    return Journal(header, tuple(entries), end)
