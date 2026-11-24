"""``quotemem export``: write an agent store's source journal (``quotemem.agent.journal``).

The export reads the store's SQLite file directly, read-only, under the store's
process lock (a running ``quotemem serve`` holds it, so the export refuses). It
reads only a few long-standing columns of six tables, by name, and ignores the
rest, so a later release can still read a store an earlier release wrote, which is
what a rebuild across releases needs:

- Evidence: one row per stored message part, with its user scope, session,
  request, message position, text, withheld ranges and timestamps;
- Control ``scope:<id>`` rows: the user id each scope was created for;
- Tombstones: the statement of every memory a forget retired, with its time;
- Memories, their head revisions and ``supersedes`` links: the quoted ranges of each
  memory that ended without a surviving successor (its successor was forgotten or
  erased), and when it ended, which the journal names as ``retired`` so a rebuild
  ends it again at the same point of the replay. A range a memory still active
  also quotes is left out and counted (``ambiguous_retired``).

The forgotten statements are sealed in the journal (``quotemem.agent.journal_key``):
under ``<data dir>/journal.key``, made by the first export, or under a key derived
from a passphrase. Writing that key file is the export's only write to the data
directory, and it never touches the store.

Nothing is opened through the HeavenBase runtime, no provider is called and no
byte of the store changes: a cleanly closed store is opened ``immutable``; one with
a write-ahead log left by a crash is read through that log without recovering it.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import closing
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Final, Literal, cast
from urllib.request import pathname2url

from quotemem.agent.journal import (
    WITHHELD_MARKER,
    Forget,
    JournalAddRefV1,
    JournalAddV1,
    JournalEndV1,
    JournalHeaderV1,
    JournalMessageV1,
    JournalRetiredV1,
    JournalStoreV1,
    JournalWithheldV1,
    ReplayEntry,
    write_journal,
)
from quotemem.agent.journal_key import JournalKey, data_dir_key
from quotemem.core.errors import QuoteMemError
from quotemem.core.platform_files import OWNER_LOCK_NAME, exclusive_owner, open_private_file

EVIDENCE_TABLE: Final = "agent_memory_evidence"
CONTROL_TABLE: Final = "agent_memory_control"
TOMBSTONE_TABLE: Final = "agent_memory_tombstone"
MEMORY_TABLE: Final = "agent_memory_item"
REVISION_TABLE: Final = "agent_memory_revision"
LINK_TABLE: Final = "agent_memory_link"
"""HeavenBase's physical names of the six entities an export reads."""

_EVIDENCE_COLUMNS: Final = (
    "scope_kind", "scope_id", "session_id", "request_id", "ordinal", "part_no", "kind",
    "state", "text", "byte_start", "source_observed_at", "ingested_at",
    "ingested_commit_seq", "withheld_spans",
)
_EVIDENCE_OPTIONAL: Final = ("role", "speaker_entity", "source_participant")
_TOMBSTONE_COLUMNS: Final = (
    "scope_kind", "scope_id", "statement", "forget_run_id", "tombstone_seq", "created_at",
)
_ENDED: Final[Mapping[str, Literal["update", "correct", "retract"]]] = {
    "superseded": "update", "corrected": "correct", "retracted": "retract",
}
"""An ended memory's lifecycle and the relation that ended it."""
_ROLES: Final[Mapping[str, Literal["user", "assistant", "system"]]] = {
    "user_message": "user", "assistant_message": "assistant", "system_message": "system",
}


def _aml_version() -> str:
    try:
        return version("quotemem")
    except PackageNotFoundError:
        return "unknown"


def _json(value: object) -> object:
    """Decode a stored JSON column, which HeavenBase may encode as a JSON string."""
    for _ in range(3):
        if not isinstance(value, str):
            return value
        try:
            value = json.loads(value)
        except ValueError:
            return value
    return value


def _columns(connection: sqlite3.Connection, table: str) -> set[str]:
    rows = connection.execute(f'PRAGMA table_info("{table}")').fetchall()
    return {cast(str, row[1]) for row in rows}


def _rows(
    connection: sqlite3.Connection, table: str, required: tuple[str, ...],
    optional: tuple[str, ...], order: str,
) -> Iterator[dict[str, Any]]:
    present = _columns(connection, table)
    if not present:
        return
    if missing := [name for name in required if name not in present]:
        raise QuoteMemError(
            "schema_mismatch", f"The store's {table} table lacks {', '.join(missing)}",
        )
    names = [*required, *(name for name in optional if name in present)]
    selected = ", ".join(f'"{name}"' for name in names)
    cursor = connection.execute(f'SELECT {selected} FROM "{table}" ORDER BY {order}')
    for row in cursor:
        yield dict(zip(names, cast(tuple[Any, ...], row), strict=True))


@dataclass
class _Message:
    role: Literal["user", "assistant", "system"]
    timestamp: int | None
    speaker: str | None
    participant: dict[str, Any] | None
    pieces: list[tuple[int, int, str | None, list[tuple[int, int]]]] = field(
        default_factory=lambda: [],
    )
    """(part_no, the part's first message byte, text or None when erased, withheld ranges
    relative to the part)."""


@dataclass
class _Request:
    scope_id: str
    session_id: str
    request_id: str
    ingested_at: int
    ingested_commit_seq: int
    messages: dict[int, _Message] = field(default_factory=lambda: {})
    withheld: list[JournalWithheldV1] = field(default_factory=lambda: [])
    erased: int = 0


def _withheld(raw: object) -> list[dict[str, Any]]:
    value = _json(raw)
    if value is None:
        return []
    if not isinstance(value, list):
        raise QuoteMemError("schema_mismatch", "A stored withheld range list is malformed")
    spans: list[dict[str, Any]] = []
    for item in cast(list[object], value):
        if not isinstance(item, dict):
            raise QuoteMemError("schema_mismatch", "A stored withheld range is malformed")
        span = cast(dict[str, Any], item)
        if not (type(span.get("byte_start")) is int and type(span.get("byte_end")) is int):
            raise QuoteMemError("schema_mismatch", "A stored withheld range is malformed")
        spans.append(span)
    return spans


type _Located = Callable[[int, int], tuple[int, int] | None]
"""Maps an original message byte range to the rendered content's, or None."""


def _render(message: _Message) -> tuple[str, _Located]:
    """The message text with withheld ranges and erased parts replaced by the marker.

    Adjacent markers collapse into one. Also returns where a range of the original
    message's bytes lies in the rendered text: only a range inside one kept run of
    bytes, decoded unchanged, has a place there.
    """
    runs: list[tuple[int, int, str] | None] = []
    """Kept runs as (original start, original end, text), and markers as None."""
    for _, origin, text, ranges in sorted(message.pieces, key=lambda piece: piece[0]):
        if text is None:
            runs.append(None)
            continue
        data = text.encode("utf-8")
        cursor = 0
        for start, end in sorted(ranges):
            start, end = max(start, cursor), min(end, len(data))
            if end <= start:
                continue
            runs.extend(((origin + cursor, origin + start,
                          data[cursor:start].decode("utf-8", errors="replace")), None))
            cursor = end
        runs.append((origin + cursor, origin + len(data),
                     data[cursor:].decode("utf-8", errors="replace")))
    out: list[str] = []
    placed: list[tuple[int, int, int]] = []
    """(original start, original end, rendered start) of each kept run decoded unchanged."""
    size = 0
    marker = False
    for run in runs:
        if run is None:
            if not marker:
                out.append(WITHHELD_MARKER)
                size += len(WITHHELD_MARKER.encode("utf-8"))
            marker = True
            continue
        start, end, text = run
        if not text:
            continue
        encoded = len(text.encode("utf-8"))
        if encoded == end - start:
            placed.append((start, end, size))
        out.append(text)
        size += encoded
        marker = False
    rendered = "".join(out)
    doubled = WITHHELD_MARKER * 2
    if doubled in rendered:
        # The message's own text continues a marker: no range has a reliable place.
        while doubled in rendered:
            rendered = rendered.replace(doubled, WITHHELD_MARKER)
        placed = []

    def located(byte_start: int, byte_end: int) -> tuple[int, int] | None:
        for start, end, at in placed:
            if start <= byte_start and byte_end <= end:
                return at + byte_start - start, at + byte_end - start
        return None

    return rendered, located


@dataclass
class ExportSkips:
    """Rows an export leaves out, by reason; counts only."""

    expired_parts: int = 0
    non_message_parts: int = 0
    unknown_scope_parts: int = 0
    erased_messages: int = 0
    unplaced_retired: int = 0
    """Retired quotes a withheld range or an erased part left without a place."""
    ambiguous_retired: int = 0
    """Retired quotes a memory still active also quotes: left for the replay to judge."""


def _users(connection: sqlite3.Connection) -> dict[str, str]:
    users: dict[str, str] = {}
    present = _columns(connection, CONTROL_TABLE)
    if not {"key", "value"} <= present:
        raise QuoteMemError("schema_mismatch", "The store's control table is unreadable")
    cursor = connection.execute(
        f'SELECT "key", "value" FROM "{CONTROL_TABLE}" WHERE "key" >= ? AND "key" < ?',
        ("scope:", "scope;"),
    )
    for key, value in cast(Iterator[tuple[str, object]], cursor):
        decoded = _json(value)
        if not isinstance(decoded, dict):
            continue
        fields = cast(dict[str, Any], decoded)
        raw = fields.get("raw_external_id")
        if fields.get("scope_kind") == "user" and isinstance(raw, str) and raw:
            users[key.removeprefix("scope:")] = raw
    return users


type _Ended = Literal["update", "correct", "retract"]
type _MessageKey = tuple[str, str, str, int]
"""(scope, session, request, message ordinal)."""


@dataclass(frozen=True)
class _RetiredSpan:
    """A retired memory's quoted range of one message, and when the memory ended."""

    byte_start: int
    byte_end: int
    ended: _Ended
    commit_seq: int
    """The ending commit's sequence, or a later one that touched the memory."""
    retired_at: int


_NO_END: Final = 2 ** 63
"""The ``retired_at`` of a memory that records no end time: no bound."""
type _Retired = Mapping[_MessageKey, list[_RetiredSpan]]


def _retired(
    connection: sqlite3.Connection, users: Mapping[str, str], skips: ExportSkips,
) -> _Retired:
    """The quoted ranges of every memory that ended without a surviving successor.

    A memory ``superseded``, ``corrected`` or ``retracted`` that no ``supersedes``
    link of a remaining, unerased memory names lost its successor to a forget or an
    erasure. Its head's quotes are returned in original message coordinates, with
    the relation its lifecycle names and when it ended. A range that overlaps a
    quote of a memory still active is left out (``skips.ambiguous_retired``): the
    words cannot tell the ended memory from the current one. A store without these
    tables has none.
    """
    tables = {name: _columns(connection, name) for name in (MEMORY_TABLE, REVISION_TABLE,
                                                             LINK_TABLE, EVIDENCE_TABLE)}
    needed = {
        MEMORY_TABLE: {"object_id", "scope_kind", "scope_id", "current_revision_id",
                       "lifecycle_state", "commit_seq", "retired_at"},
        REVISION_TABLE: {"object_id", "memory_id", "snapshot"},
        LINK_TABLE: {"source_revision_id", "link_type", "target_key"},
        EVIDENCE_TABLE: {"object_id", "scope_id", "session_id", "request_id", "ordinal",
                         "state"},
    }
    if any(not tables[name] for name in (MEMORY_TABLE, REVISION_TABLE, LINK_TABLE)):
        return {}
    for name, columns in needed.items():
        if missing := sorted(columns - tables[name]):
            raise QuoteMemError(
                "schema_mismatch", f"The store's {name} table lacks {', '.join(missing)}",
            )
    located: dict[str, tuple[str, str, str, int, str] | None] = {}

    def quotes(scope: str, raw: object) -> Iterator[tuple[_MessageKey, int, int]]:
        """The head's quotes of live Evidence of ``scope``, by message."""
        snapshot = _json(raw)
        spans = cast(dict[str, Any], snapshot).get("quote_spans") if isinstance(
            snapshot, dict) else None
        if not isinstance(spans, list):
            raise QuoteMemError("schema_mismatch", "A stored memory head has no readable quotes")
        for span in cast(list[object], spans):
            fields = cast(dict[str, Any], span) if isinstance(span, dict) else {}
            evidence, start, end = (fields.get("evidence_id"), fields.get("byte_start"),
                                    fields.get("byte_end"))
            if not (isinstance(evidence, str) and type(start) is int and type(end) is int):
                raise QuoteMemError("schema_mismatch", "A stored quote is malformed")
            if evidence not in located:
                row = connection.execute(
                    f'SELECT "scope_id", "session_id", "request_id", "ordinal", "state" '
                    f'FROM "{EVIDENCE_TABLE}" WHERE "object_id" = ?', (evidence,),
                ).fetchone()
                located[evidence] = None if row is None else (
                    str(row[0]), str(row[1]), str(row[2]), int(row[3]), str(row[4]))
            row = located[evidence]
            if row is None or row[0] != scope or row[4] in {"expired", "invalidated",
                                                            "erased"}:
                continue
            yield row[:4], start, end

    current: dict[_MessageKey, list[tuple[int, int]]] = {}
    for scope, raw in cast(Iterator[tuple[str, object]], connection.execute(
        f'SELECT i."scope_id", r."snapshot" FROM "{MEMORY_TABLE}" AS i '
        f'JOIN "{REVISION_TABLE}" AS r ON r."object_id" = i."current_revision_id" '
        f'WHERE i."scope_kind" = ? AND i."lifecycle_state" = ? ORDER BY i."object_id"',
        ("user", "active"),
    )):
        if scope in users:
            for key, start, end in quotes(scope, raw):
                current.setdefault(key, []).append((start, end))
    states = tuple(_ENDED)
    cursor = connection.execute(
        f'SELECT i."scope_id", i."lifecycle_state", i."commit_seq", i."retired_at", '
        f'r."snapshot" FROM "{MEMORY_TABLE}" AS i JOIN "{REVISION_TABLE}" AS r '
        f'ON r."object_id" = i."current_revision_id" '
        f'WHERE i."scope_kind" = ? AND i."lifecycle_state" IN (?, ?, ?) AND NOT EXISTS ('
        f'SELECT 1 FROM "{LINK_TABLE}" AS l JOIN "{REVISION_TABLE}" AS s '
        f'ON s."object_id" = l."source_revision_id" JOIN "{MEMORY_TABLE}" AS o '
        f'ON o."object_id" = s."memory_id" WHERE l."link_type" = ? '
        f'AND l."target_key" = i."object_id" AND o."lifecycle_state" != ?) '
        f'ORDER BY i."object_id"',
        ("user", *states, "supersedes", "erased"),
    )
    found: dict[_MessageKey, list[_RetiredSpan]] = {}
    for scope, state, sequence, retired_at, raw in cast(
            Iterator[tuple[str, str, int, int | None, object]], cursor):
        if scope not in users:
            continue
        for key, start, end in quotes(scope, raw):
            if any(start < other_end and other_start < end
                   for other_start, other_end in current.get(key, ())):
                skips.ambiguous_retired += 1
                continue
            found.setdefault(key, []).append(_RetiredSpan(
                start, end, _ENDED[state], int(sequence),
                _NO_END if retired_at is None else int(retired_at)))
    return found


def _after(entries: list[JournalAddV1], meta: list[tuple[str, int, int]], quoting: int,
           span: _RetiredSpan) -> JournalAddRefV1:
    """The last Add of the quoting Add's user taken in before the memory ended.

    ``meta`` holds each entry's (scope, first commit sequence, ingested time) in
    journal order. An Add ingested by an earlier commit, and no later than the
    memory's end, came before it; the quoting Add itself always did.
    """
    scope = meta[quoting][0]
    chosen = quoting
    for index in range(quoting + 1, len(entries)):
        owner, sequence, ingested = meta[index]
        if owner != scope:
            continue
        if sequence >= span.commit_seq or ingested > span.retired_at:
            break
        chosen = index
    return JournalAddRefV1(session_id=entries[chosen].session_id,
                           request_id=entries[chosen].request_id)


def _requests(
    connection: sqlite3.Connection, users: Mapping[str, str], skips: ExportSkips,
    retired: _Retired,
) -> list[JournalAddV1]:
    requests: dict[tuple[str, str, str], _Request] = {}
    for row in _rows(
        connection, EVIDENCE_TABLE, _EVIDENCE_COLUMNS, _EVIDENCE_OPTIONAL,
        '"ingested_commit_seq", "ingested_at", "scope_id", "session_id", "request_id", '
        '"ordinal", "part_no"',
    ):
        state = row["state"]
        if state in {"expired", "invalidated"}:
            skips.expired_parts += 1
            continue
        role = _ROLES.get(cast(str, row["kind"]))
        if role is None:
            skips.non_message_parts += 1
            continue
        if row["scope_kind"] != "user" or row["scope_id"] not in users:
            skips.unknown_scope_parts += 1
            continue
        key = (cast(str, row["scope_id"]), cast(str, row["session_id"]),
               cast(str, row["request_id"]))
        request = requests.get(key)
        if request is None:
            request = requests[key] = _Request(
                *key, ingested_at=int(row["ingested_at"]),
                ingested_commit_seq=int(row["ingested_commit_seq"]))
        ordinal = int(row["ordinal"])
        message = request.messages.get(ordinal)
        if message is None:
            participant = _json(row.get("source_participant"))
            message = request.messages[ordinal] = _Message(
                role=role,
                timestamp=None if row["source_observed_at"] is None
                else int(row["source_observed_at"]),
                speaker=cast(str | None, row.get("speaker_entity")),
                participant=cast(dict[str, Any], participant)
                if isinstance(participant, dict) else None,
            )
        start = int(row["byte_start"])
        if state == "erased":
            request.erased += 1
            message.pieces.append((int(row["part_no"]), start, None, []))
            continue
        relative: list[tuple[int, int]] = []
        for span in _withheld(row["withheld_spans"]):
            request.withheld.append(JournalWithheldV1(
                ordinal=ordinal, byte_start=span["byte_start"], byte_end=span["byte_end"],
                reason=str(span.get("reason", "forgotten")),
            ))
            relative.append((span["byte_start"] - start, span["byte_end"] - start))
        message.pieces.append((int(row["part_no"]), start, cast(str, row["text"]), relative))
    entries: list[JournalAddV1] = []
    meta: list[tuple[str, int, int]] = []
    pending: list[tuple[int, int, int, int, _RetiredSpan]] = []
    """(entry, message, rendered start, rendered end, span) of each placed retired quote."""
    for request in requests.values():
        messages: list[JournalMessageV1] = []
        for ordinal, message in sorted(request.messages.items()):
            if all(text is None for _, _, text, _ in message.pieces):
                skips.erased_messages += 1
                continue
            content, located = _render(message)
            if not content.strip():
                continue
            for span in retired.get(
                    (request.scope_id, request.session_id, request.request_id, ordinal), ()):
                place = located(span.byte_start, span.byte_end)
                if place is None:
                    skips.unplaced_retired += 1
                    continue
                pending.append((len(entries), len(messages), place[0], place[1], span))
            messages.append(JournalMessageV1(
                role=message.role, content=content, timestamp=message.timestamp,
                speaker=message.speaker if message.role != "system" else None,
                source_participant=message.participant if message.role != "system" else None,
            ))
        if messages:
            entries.append(JournalAddV1(
                user_id=users[request.scope_id], session_id=request.session_id,
                request_id=request.request_id, ingested_at_ms=request.ingested_at,
                messages=messages,
                withheld=sorted(request.withheld, key=lambda span: (
                    span.ordinal, span.byte_start, span.byte_end)),
                erased_parts=request.erased,
            ))
            meta.append((request.scope_id, request.ingested_commit_seq, request.ingested_at))
    ended: dict[int, set[JournalRetiredV1]] = {}
    for index, message_at, byte_start, byte_end, span in pending:
        ended.setdefault(index, set()).add(JournalRetiredV1(
            message=message_at, byte_start=byte_start, byte_end=byte_end, ended=span.ended,
            after=_after(entries, meta, index, span),
        ))
    for index, spans in ended.items():
        entries[index] = entries[index].model_copy(update={"retired": sorted(
            spans, key=lambda span: (span.message, span.byte_start, span.byte_end,
                                     span.ended, span.after.session_id,
                                     span.after.request_id))})
    return entries


def _forgets(
    connection: sqlite3.Connection, users: Mapping[str, str],
) -> list[Forget]:
    operations: dict[tuple[str, str], tuple[int, list[str]]] = {}
    for row in _rows(
        connection, TOMBSTONE_TABLE, _TOMBSTONE_COLUMNS, (),
        '"created_at", "tombstone_seq", "object_id"',
    ):
        if row["scope_kind"] != "user" or row["scope_id"] not in users:
            continue
        key = (cast(str, row["scope_id"]), cast(str, row["forget_run_id"]))
        at, statements = operations.setdefault(key, (int(row["created_at"]), []))
        statement = cast(str, row["statement"])
        if statement and statement not in statements:
            statements.append(statement)
        operations[key] = (min(at, int(row["created_at"])), statements)
    return [
        Forget(users[scope], at, tuple(statements))
        for (scope, _), (at, statements) in sorted(
            operations.items(), key=lambda item: item[1][0],
        )
        if statements
    ]


def _store_binding(data_dir: Path) -> JournalStoreV1:
    """The exported store's binding, read leniently from its own files."""
    def read(path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        descriptor = open_private_file(path, os.O_RDONLY)
        with os.fdopen(descriptor, "rb") as stream:
            try:
                value: object = json.loads(stream.read())
            except ValueError:
                return {}
        return cast(dict[str, Any], value) if isinstance(value, dict) else {}

    agent = read(data_dir / "agent.json")
    providers = read(data_dir / "workspace" / "workspace.json").get("providers")
    stored = cast(dict[str, Any], providers) if isinstance(providers, dict) else {}
    dimensions = agent.get("embedding_dimensions", stored.get("embedding_dimensions"))
    model = agent.get("embedding_model")
    mode = stored.get("add_mode")
    return JournalStoreV1(
        embedding_model=model if isinstance(model, str) else None,
        embedding_dimensions=dimensions if dimensions in (1024, 2048) else None,
        add_mode=mode if isinstance(mode, str) else None,
    )


@dataclass(frozen=True)
class ExportResult:
    """What an export wrote: its end record, what it left out and its key.

    Attributes:
        end: The journal's end record.
        skips: Rows left out, by reason.
        key_id: The sealing key's fingerprint.
        key_file: The data directory key's file; None for a passphrase or given key.
    """

    end: JournalEndV1
    skips: ExportSkips
    key_id: str
    key_file: Path | None


@dataclass(frozen=True)
class StoreContent:
    """A store's journal records, in clear, and the rows left out."""

    store: JournalStoreV1
    entries: list[ReplayEntry]
    skips: ExportSkips


def read_store(data_dir: Path) -> StoreContent:
    """Read an agent store's journal entries without changing the store.

    Args:
        data_dir: The agent data directory (``workspace/`` inside it).

    Raises:
        QuoteMemError: ``reference_missing`` when it holds no store,
            ``workspace_locked`` while a service owns it, ``schema_mismatch`` for
            tables an export cannot read.
    """
    workspace = data_dir.expanduser() / "workspace"
    database = workspace / "memory.db"
    if not database.is_file():
        raise QuoteMemError("reference_missing", "No QuoteMem store is in this data directory")
    with exclusive_owner(workspace):
        if not (workspace / OWNER_LOCK_NAME).is_file():
            raise QuoteMemError("reference_missing", "The store has no owner lock")
        logged = any(
            (workspace / f"memory.db{suffix}").exists() for suffix in ("-wal", "-shm")
        )
        uri = f"file:{pathname2url(str(database))}?" + ("mode=ro" if logged else "immutable=1")
        with closing(sqlite3.connect(uri, uri=True)) as connection:
            connection.execute("PRAGMA query_only = ON")
            skips = ExportSkips()
            users = _users(connection)
            entries: list[ReplayEntry] = [
                *_requests(connection, users, skips, _retired(connection, users, skips)),
                *_forgets(connection, users),
            ]
    return StoreContent(_store_binding(data_dir.expanduser()), entries, skips)


def export_store(
    data_dir: Path, out: Path, *, passphrase: str | None = None, key: JournalKey | None = None,
) -> ExportResult:
    """Export ``data_dir``'s store to a new journal file ``out`` (mode 0600).

    The forgotten statements are sealed under ``key`` when given, else under a key
    derived from ``passphrase`` (a new salt), else under ``<data_dir>/journal.key``,
    which is created once the store has been read.

    Raises:
        QuoteMemError: As ``read_store``; ``operation_conflict`` when ``out`` exists;
            ``schema_invalid`` for a short passphrase or an unreadable key file.
    """
    if out.exists() or out.is_symlink():
        raise QuoteMemError("operation_conflict", "The journal file already exists; name a new one")
    content = read_store(data_dir)
    key_file: Path | None = None
    if key is None:
        if passphrase is not None:
            key = JournalKey.from_passphrase(passphrase)
        else:
            key, key_file = data_dir_key(data_dir.expanduser())
    header = JournalHeaderV1(
        exported_at_ms=time.time_ns() // 1_000_000, aml_version=_aml_version(),
        store=content.store, encryption=key.encryption,
    )
    end = write_journal(out, header, content.entries, key)
    return ExportResult(end, content.skips, key.key_id, key_file)
