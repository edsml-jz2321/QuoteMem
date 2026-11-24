"""Durable dashboard-assistant session logging.

One turn at a time runs per conversation within a dashboard process. Its
provider calls emit write-ahead ``request`` and terminal ``response``/``error``
trace rows. A sealed ``turn`` row binds the question and workspace and is
the authority for replay and transcript repair. Successful retries replay
the stored answer. Retries after a provider failure or interruption can
write fresh call events and replace the provisional turn row on success.

An admission read that would determine whether to call the provider must
succeed before execution. Workspace setup, trace recording and transcript
projection failures otherwise surface through the turn's capture status.
"""

from __future__ import annotations

__all__ = ["AssistSessionLog", "AssistTurnLog"]

import threading
import warnings
import weakref
from time import time
from uuid import uuid4

from heavenbase.extensions.ids import AGENT_LOG_EXTENSION_ID
from heavenbase.llm.trace import LLMTraceEvent, TraceCallback, forward_trace, trace_state
from heavenbase.utils import Any, Callable, Dict, List, Mapping, Optional, Tuple, hash_id

_SESSION_SOURCE = "dashboard-assist"
_SEAL_ENTITY = "agent-log-seal"
_CONFIG_ROOT = "heavenbase.dashboard.assistant"
# Entity Identifier fields accept any non-empty text up to this many chars.
_IDENTIFIER_MAX_CHARS = 63
_WARNED_FAILURES: set = set()
# Registry identifier of the process-memory backend; its aliases come from the registry.
_INMEM_BACKEND = "inmem"
# The packaged in-memory backend's identifier and aliases (backends/inmem/meta.yaml),
# lowercased; consulted only when no module registry can be reached at all.
_INMEM_PACKAGED_NAMES = frozenset({"inmem", "memory", "inmembackend"})
# The Chroma backend, whose client keeps its data in process memory when no path is given.
_CHROMA_BACKEND = "chroma"
_CHROMA_PACKAGED_NAMES = frozenset({"chroma", "chromabackend"})
# The ``running`` holder of a conversation an admission is evicting; no thread has ident 0.
_EVICTION = 0

# Module registries whose packaged modules this module has indexed (or is
# indexing), each with the event set when that one setup finishes; a lookup
# miss sets up a registry at most once per process.
_MODULE_SETUPS: "weakref.WeakKeyDictionary[Any, threading.Event]" = weakref.WeakKeyDictionary()
_MODULE_SETUPS_LOCK = threading.Lock()


class _LogWrites:
    """Coordinate turn admission and sealing for a workspace in one Context tree.

    ``changed`` guards ``active`` (retention pins) and ``running`` (the
    thread, or :data:`_EVICTION`, holding each conversation) and is never
    held across storage I/O. ``retention`` runs one retention pass at a time.
    """

    def __init__(self) -> None:
        self.changed = threading.Condition()
        self.active: Dict[str, int] = {}
        self.running: Dict[str, int] = {}
        self.retention = threading.Lock()


_LOG_WRITES: "weakref.WeakValueDictionary[Tuple[str, str], _LogWrites]" = weakref.WeakValueDictionary()
_LOG_WRITES_LOCK = threading.Lock()


def _retry_event_id(conversation_id: str, turn_id: str, attempt: int, seq: int, kind: str) -> str:
    """Keep a retry's event identity distinct from ordinary turn ids ending in its suffix."""
    return hash_id("assist-retry-event", conversation_id, turn_id, attempt, seq, kind)


def _warn_once(context: str, error: Exception) -> None:
    """Warn once per (context, exception type) that recording degraded."""
    marker = f"{context}:{type(error).__name__}"
    if marker in _WARNED_FAILURES:
        return
    _WARNED_FAILURES.add(marker)
    try:
        warnings.warn(f"assistant session log {context} failed ({type(error).__name__}: {error})", UserWarning, stacklevel=3)
    except Exception:
        pass  # warnings-as-errors must not let a diagnostic fail the turn


def _safe_identifier(value: str, namespace: str) -> str:
    """Return ``value`` when storage-safe, else its deterministic hash.

    The HTTP contract accepts ids up to 128 characters while entity
    ``Identifier`` fields cap at 63, so oversize ids map to a stable hash —
    the same client id always maps to the same stored id, preserving retry
    idempotency.
    """
    return value if len(value) <= _IDENTIFIER_MAX_CHARS else hash_id(namespace, value)


def _session_ref(conversation_id: str) -> str:
    """Return the deterministic id of a conversation's projection session row."""
    return hash_id("assist-log", conversation_id)


def _seal_ref(conversation_id: str) -> str:
    """Return the deterministic id of a conversation's seal counter."""
    return hash_id("assist-log-seal", conversation_id)


def _read_for_admission(read: Callable[[], Any]) -> Any:
    """Retry a failed read once. A storage error never represents an absent row."""
    try:
        return read()
    except Exception:
        try:
            return read()
        except Exception as error:
            raise ValueError("assistant session log could not read the turn or conversation") from error


def _projected_turns(session: Any) -> List[str]:
    """Return the turn ids a projection row records as present in its transcript.

    ``state.turn_ids`` lists every turn the transcript holds, in transcript
    order; ``state.last_turn_id`` is counted too, so a row whose list was lost
    still knows its newest turn.
    """
    state = dict(session.state or {})
    recorded = state.get("turn_ids")
    turns = [str(item) for item in recorded if item] if isinstance(recorded, list) else []
    last = str(state.get("last_turn_id") or "")
    if last and last not in turns:
        turns.append(last)
    return turns


def _usage_only_turns(session: Any) -> List[str]:
    """Return turn ids whose usage was projected without transcript messages."""
    recorded = session.state.get("usage_only_turn_ids")
    return [str(turn) for turn in recorded if turn] if isinstance(recorded, list) else []


def _turn_order(row: Mapping[str, Any]) -> Tuple[bool, int, int, str]:
    """Sort sealed turns by completion order, with a timestamp fallback for older rows."""
    payload = row.get("payload") or {}
    seq = int(row["seq"])
    return (seq > 0, seq or int(payload.get("close_order") or 0), int(payload.get("closed_at") or 0), str(row["object_id"]))


def _is_duplicate_write(error: ValueError) -> bool:
    """Return whether ``error`` is the writer's existing-row refusal.

    ``upsert(overwrite=False)`` raises ``ValueError`` both for duplicates and
    for field validation; only the former is an idempotent replay. Anything
    else is a real recording failure and must mark the turn's capture state.
    """
    return "already exists" in str(error)


def _with_modules(lookup: Callable[[], Any], context: Any = None) -> Any:
    """Run one registry lookup, indexing the packaged modules once if it misses.

    A HOME that was never bootstrapped has no module registrations yet, so
    the in-memory backend's aliases and the database dialects are unknown
    until the packaged module set is indexed, which construction does anyway.
    That indexing takes over a second, so it runs at most once per module
    registry and process: a later miss (a ``file://`` or ``redis://`` URL no
    dialect resolves) only repeats the lookup, and a caller that misses while
    another thread indexes waits for it.

    Args:
        lookup (Callable[[], Any]): Registry lookup to run.
        context (Any): Context whose module registry is set up on a miss.
            Defaults to the process context.

    Returns:
        Any: The lookup's result.
    """
    try:
        return lookup()
    except LookupError:
        from heavenbase.context import context_or_default

        selected = context_or_default(context)
        service = selected.modules()
        with _MODULE_SETUPS_LOCK:
            indexed = _MODULE_SETUPS.get(service)
            first = indexed is None
            if first:
                indexed = _MODULE_SETUPS[service] = threading.Event()
        if first:
            try:
                selected._setup_modules()
            finally:
                indexed.set()
        else:
            indexed.wait()
        return lookup()


def _backend_type_is(kind: Any, identifier: str, packaged_names: frozenset, context: Any = None) -> bool:
    """Return whether a backend type or instance selects one registered backend type.

    A type is normalized as construction normalizes it (stripped and
    lowercased) and resolved through the backend registry, so every
    registered alias counts; a type the registry does not resolve is compared
    with the backend's identifier and registered aliases in any case, or with
    the packaged ones (``packaged_names``) when no module registry can be
    reached.
    """
    from heavenbase import BackendType
    from heavenbase.context import context_or_default

    if isinstance(kind, str):
        kind = kind.strip().lower()
        if not kind:
            return False
    selected = None
    try:
        selected = context_or_default(context)
        return _with_modules(lambda: BackendType.identifier_of(kind, resolver=selected.modules()), selected) == identifier
    except Exception:
        pass
    if not isinstance(kind, str):
        return False
    try:
        aliases = BackendType.aliases(identifier, resolver=selected.modules())  # AttributeError when no context: packaged names
        names = {identifier, *(str(alias).strip().lower() for alias in aliases)}
    except Exception:
        names = set(packaged_names)
    return kind in names


def _in_memory_type(kind: Any, context: Any = None) -> bool:
    """Return whether a backend type or instance selects the in-memory backend (see :func:`_backend_type_is`)."""
    return _backend_type_is(kind, _INMEM_BACKEND, _INMEM_PACKAGED_NAMES, context)


def _chroma_type(kind: Any, context: Any = None) -> bool:
    """Return whether a declared backend type selects the Chroma backend, by any alias in any case."""
    return _backend_type_is(kind, _CHROMA_BACKEND, _CHROMA_PACKAGED_NAMES, context)


def _url_in_memory(value: str) -> bool:
    """Return whether a declared connection URL selects an in-memory store.

    A ``mem://`` or ``memory://`` URL names a memory engine (SurrealDB's
    embedded one, the object-store memory URI LanceDB accepts). A URL whose
    SQLAlchemy dialect is registered is resolved by the database config
    engine construction uses (``sqlite://`` and ``sqlite:///:memory:`` both
    select memory mode), after indexing the packaged dialects once if they
    are not registered yet. Any other URL (the packaged ``file://`` default,
    ``redis://``, ``ws://``) names no SQL database and is left to the
    construction check, without resolving the whole database config on every
    turn.
    """
    from urllib.parse import urlsplit

    if urlsplit(value).scheme.lower() in {"mem", "memory"}:
        return True
    import sqlalchemy

    from heavenbase import DB_ENGINE

    try:
        dialect = sqlalchemy.engine.make_url(value).drivername.partition("+")[0]
        _with_modules(lambda: DB_ENGINE.dialects.get(dialect))  # DB_ENGINE reads the process context
        return DB_ENGINE.resolve({"url": value}).in_memory is True
    except Exception:
        return False


def _declared_in_memory(entry: Mapping[str, Any], context: Any = None) -> Optional[str]:
    """Return why a declared backend entry keeps its data only in process memory, else ``None``.

    Reads the declaration the way construction does: the backend type is
    normalized case-insensitively and resolved through the backend registry,
    and a database is in memory when it says ``in_memory``,
    names ``:memory:``, or is given by a ``url``, ``uri`` or URL-shaped
    ``database`` that selects memory mode. A Chroma backend, declared by any
    alias in any case, without a ``path`` uses Chroma's ephemeral in-memory
    client.
    """
    kind = str(entry.get("type") or "").strip().lower()
    if _in_memory_type(kind, context):
        return "is an in-memory backend"
    if entry.get("in_memory") or str(entry.get("database") or "").strip() == ":memory:":
        return "points at an in-memory database"
    for field in ("url", "uri", "database"):
        value = entry.get(field)
        if isinstance(value, str) and "://" in value and _url_in_memory(value.strip()):
            return f"has a {field} that selects an in-memory store"
    if _chroma_type(kind, context) and entry.get("path") is None:
        return "is a Chroma backend without a path, which uses Chroma's ephemeral in-memory client"
    return None


def _constructed_in_memory(backend: Any, context: Any = None) -> Optional[str]:
    """Return why a constructed backend keeps its data only in process memory, else ``None``.

    The construction-side counterpart of :func:`_declared_in_memory`: it
    catches in-memory stores a declaration reaches only through resolution,
    such as a database preset whose resolved spec is in memory.
    """
    from urllib.parse import urlsplit

    if _in_memory_type(backend, context):
        return "is an in-memory backend"
    if getattr(getattr(getattr(backend, "db", None), "spec", None), "in_memory", False) is True:
        return "resolved to an in-memory database"
    kind = str(getattr(backend, "type", "") or "").lower()
    if kind == "chroma" and getattr(backend, "path", "") is None:
        return "uses Chroma's ephemeral in-memory client"
    for field in ("url", "uri"):
        value = getattr(backend, field, None)
        if isinstance(value, str) and urlsplit(value.strip()).scheme.lower() in {"mem", "memory"}:
            return f"resolved to an in-memory {field}"
    return None


class AssistSessionLog:
    """Owner of the assistant session-log workspace and its write discipline.

    One instance serves one :class:`~heavenbase.api.service.DashboardAPI`.
    The log workspace is opened lazily on the first enabled turn; its
    construction contract (name and backends) comes from
    ``heavenbase.dashboard.assistant.session_log_*`` and is validated by
    :meth:`validate` before anything is written.
    """

    def __init__(self, cm: Any, context: Any = None) -> None:
        """Bind the recorder to one config manager and context.

        Args:
            cm (Any): Config manager resolving the assistant block.
            context (Any): Context owning the log workspace. Defaults to the
                process context.
        """
        self.cm = cm
        self.context = context
        self._ws: Any = None
        self._workspace_lock = threading.RLock()
        self._writes = _LogWrites()
        self._native_groups: Dict[Tuple[str, Tuple[str, ...], Tuple[str, ...]], bool] = {}

    @property
    def _lock(self) -> threading.Condition:
        return self._writes.changed

    @property
    def _active(self) -> Dict[str, int]:
        return self._writes.active

    def _entity(self, name: str) -> Any:
        """Resolve one agent entity class through the workspace's registry.

        The log workspace registers the registry's artifact-compiled classes;
        a directly imported source class is a different type with a different
        structure hash, so resolution must go through the same registry.
        """
        from heavenbase.entity.registry import load_entity_type

        return load_entity_type(name, resolver=self.workspace().module_resolver)

    def _select(self, ws: Any, entity: Any, columns: Tuple[str, ...], equals: Mapping[str, Any], limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Return ``object_id`` plus ``columns`` of the rows whose fields equal ``equals``.

        A filtered ``select`` reads every column of each matched row before
        projecting, request and message payloads included. Where the backend
        groups natively, the rows are read instead as groups of ``object_id``
        and ``columns``, which reads only those columns; ``object_id`` is
        unique, so each group is one row. Other backends, and columns that
        cannot be grouped natively, use the filtered ``select``.
        """
        query = ws.query(entity)
        for name, value in equals.items():
            query = query.where(getattr(entity, name) == value)
        names = ("object_id", *columns)
        key = (entity.identifier, columns, tuple(equals))
        if self._native_groups.get(key) is not False:
            try:
                grouped = query.group_by(*names).aggregate(rows=entity.object_id.count())
                if key not in self._native_groups:
                    steps = grouped.explain().get("steps") or []
                    self._native_groups[key] = any(step.get("node") == "aggregate" and step.get("handler_mode") == "native" for step in steps)
            except ValueError:
                self._native_groups[key] = False  # a column that cannot be grouped, such as a Json payload
            if self._native_groups[key]:
                rows = (grouped if limit is None else grouped.limit(limit)).execute().rows()
                return [{name: row[name] for name in names} for row in rows]
        query = query.select(*names)
        return (query if limit is None else query.limit(limit)).execute().rows()

    def _trace_rows(
        self,
        ws: Any,
        *columns: str,
        conversation_id: str,
        kind: Optional[str] = None,
        turn_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Return ``object_id`` plus ``columns`` of one conversation's matching trace rows (see :meth:`_select`)."""
        equals = {"conversation_id": conversation_id, "kind": kind, "turn_id": turn_id}
        return self._select(ws, self._entity("agent-trace"), columns, {name: value for name, value in equals.items() if value is not None}, limit)

    def _next_seal(self, ws: Any, conversation_id: str) -> int:
        """Advance the durable counter while the caller holds the conversation."""
        seal = ws.entities[_SEAL_ENTITY]
        key = _seal_ref(conversation_id)
        counter = ws.get(key, entity=seal)
        if counter is None:
            # A log written before the counter existed can recover it once
            # from the authoritative summaries.
            turns = self._trace_rows(ws, "seq", conversation_id=conversation_id, kind="turn")
            previous = max((row["seq"] for row in turns), default=0)
        else:
            previous = counter["seq"]
        seq = previous + 1
        # Publish before the summary. An interruption can leave a gap.
        ws.upsert(seal, {"object_id": key, "seq": seq})
        return seq

    # ------------------------------------------------------------- config

    @classmethod
    def enabled(cls, cm: Any) -> bool:
        """Return whether session logging is switched on (strict literal).

        A truthy non-boolean (``"true"``, ``1``) warns once instead of
        silently reading as off — an operator who believes logging is on must
        not discover otherwise from an empty log.
        """
        value = cm.get(f"{_CONFIG_ROOT}.session_log", default=False)
        if value is True:
            return True
        if value not in (False, None):
            _warn_once("config", ValueError(f"session_log must be the literal boolean true, got {value!r}; logging stays OFF"))
        return False

    @classmethod
    def validate(cls, cm: Any, context: Any = None) -> None:
        """Fail loudly on a configuration that cannot keep the log's promise.

        A disabled log validates trivially. An enabled one must declare its
        backends, none of them in memory (preset fallback resolves to
        in-memory stores; :func:`_declared_in_memory` lists the refused
        forms), and must not be combined with ``share_credentials``, which
        would persist raw credentials at rest, unless
        ``session_log_allow_credentials`` is the literal ``true``. Runs at
        server boot and again per turn, so a post-boot config flip stops
        recording instead of persisting secrets. Backend types and database
        URLs resolve through the module registry, whose packaged modules are
        indexed first when a lookup misses (a HOME never bootstrapped).

        Args:
            cm (Any): Config manager resolving the assistant block.
            context (Any): Context whose module registry resolves backend
                types. Defaults to the process context.

        Raises:
            ValueError: If the backend declaration is missing or non-durable,
                or the credential-sharing combination is refused.
        """
        if not cls.enabled(cm):
            return
        share = cm.get(f"{_CONFIG_ROOT}.share_credentials", default=False) is True
        allow = cm.get(f"{_CONFIG_ROOT}.session_log_allow_credentials", default=False) is True
        if share and not allow:
            raise ValueError(
                "session_log with share_credentials would persist raw credentials at rest; "
                "disable one of them, or set heavenbase.dashboard.assistant.session_log_allow_credentials: true to accept that explicitly"
            )
        cls._durable_backends(cm, context)
        cls._max_sessions(cm)

    @classmethod
    def _durable_backends(cls, cm: Any, context: Any = None) -> Dict[str, Any]:
        """Return the declared log backends, refusing in-memory declarations.

        This is a declaration check, not a construction probe; see
        :func:`_declared_in_memory` for the forms it refuses. A declaration
        that reaches an in-memory store only through resolution is refused
        when :meth:`workspace` constructs it. A custom backend that discards
        data anyway is the operator's own declaration.
        """
        key = f"{_CONFIG_ROOT}.session_log_backend"
        backends = cm.get(key, default=None)
        if not hasattr(backends, "items") or not dict(backends):
            raise ValueError(f"{key} must declare the log workspace backends explicitly; preset fallback may resolve to in-memory stores")
        resolved = {str(name): dict(entry) if hasattr(entry, "items") else entry for name, entry in dict(backends).items()}
        for name, entry in resolved.items():
            if not hasattr(entry, "items"):
                raise ValueError(f"{key}.{name} must be a backend mapping")
            reason = _declared_in_memory(entry, context)
            if reason is not None:
                raise ValueError(f"{key}.{name} {reason}; a session log that vanishes with the process is not a log")
        return resolved

    def _workspace_id(self) -> str:
        return str(self.cm.get(f"{_CONFIG_ROOT}.session_log_workspace", default="assistant-log") or "assistant-log")

    @classmethod
    def _max_sessions(cls, cm: Any) -> int:
        value = cm.get(f"{_CONFIG_ROOT}.session_log_max_sessions", default=256)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("heavenbase.dashboard.assistant.session_log_max_sessions must be a non-negative integer")
        return value

    # ---------------------------------------------------------- workspace

    def workspace(self) -> Any:
        """Return the open log workspace, constructing it on first use.

        Returns:
            Any: Workspace with the ``agent-log`` extension (and, through
            it, the ``agent`` extension) enabled.

        Raises:
            ValueError: If validation refuses the configuration, a
                constructed backend keeps its data only in process memory,
                or the workspace name is already registered under a different
                construction contract.
        """
        with self._workspace_lock:
            if self._ws is not None:
                return self._ws
            self.validate(self.cm, self.context)
            from heavenbase.extensions.agent_log.seal import AgentLogSeal
            from heavenbase.workspace import HeavenBase, WorkspaceSpec

            ws_id = self._workspace_id()
            try:
                WorkspaceSpec.load(ws_id, context=self.context)
                registered = True
            except KeyError:
                registered = False
            try:
                ws = HeavenBase(ws_id, context=self.context, backends=self._durable_backends(self.cm, self.context))
            except FileExistsError as error:
                raise ValueError(
                    f"workspace {ws_id!r} is already registered with a different construction contract; "
                    "point session_log_workspace at a fresh name or restore the original session_log_backend"
                ) from error
            for name, backend in ws.backends.items():
                reason = _constructed_in_memory(backend, self.context)
                if reason is None:
                    continue
                # Never leave the refused contract behind: a registration this
                # call created would block the durable declaration the error
                # asks for. Unregistering keeps every backend's data.
                ws.release()
                if not registered:
                    WorkspaceSpec.unregister(ws_id, context=self.context)
                    remedy = "declare a durable backend"
                else:
                    remedy = f"workspace {ws_id!r} was already registered with this contract, so point session_log_workspace at a fresh name or unregister it (hb ws unset {ws_id})"
                raise ValueError(
                    f"{_CONFIG_ROOT}.session_log_backend backend {name!r} {reason}; a session log that vanishes with the process is not a log; {remedy}"
                )
            ws.enable_extension(AGENT_LOG_EXTENSION_ID)  # agent-trace rows; requires agent for the projection
            ws.register(AgentLogSeal, catalog="none")
            with _LOG_WRITES_LOCK:
                self._writes = _LOG_WRITES.setdefault((ws.context.authority_token, ws.id), self._writes)
            self._ws = ws
            return ws

    # -------------------------------------------------------------- turns

    def begin_turn(
        self,
        question: str,
        *,
        conversation_id: Optional[str] = None,
        turn_id: Optional[str] = None,
        ws_id: str = "",
        cancelled: Optional[threading.Event] = None,
    ) -> "Optional[AssistTurnLog]":
        """Open one turn's recorder, or ``None`` when logging is off.

        Serve-time refusals (an unopenable log workspace, a configuration
        turned unsafe after boot) degrade to a dead recorder whose
        ``capture`` names the reason, so the reply still carries a
        ``capture_warning`` instead of silently claiming nothing. The loud
        path for the same errors is :meth:`validate` at server boot.

        Args:
            question (str): The user question opening the turn.
            conversation_id (str | None): Client conversation identity (up to
                the protocol's 128 chars; out-of-shape ids map to a stable
                hash). When omitted, a supplied ``turn_id`` identifies its
                own conversation. With neither id, a new conversation opens
                for each call. Defaults to ``None``.
            turn_id (str | None): Client turn identity; retries that reuse it
                replay completed turns as no-ops and record retried attempts
                under a fresh attempt namespace. Defaults to ``None``.
            ws_id (str): Workspace the turn serves. Defaults to ``""``.
            cancelled (threading.Event | None): Stop waiting and skip admission
                when set. Defaults to ``None``.

        Returns:
            Optional[AssistTurnLog]: Turn recorder (possibly dead), or
            ``None`` when logging is off or admission is cancelled.

        Raises:
            ValueError: If turn identity cannot be read, the recorded question or
                workspace differs, or this thread already runs the conversation.
        """
        if not self.enabled(self.cm):
            return None
        try:
            self.validate(self.cm, self.context)  # per-turn re-check: a post-boot unsafe flip stops recording
            ws = self.workspace()
        except Exception as error:
            _warn_once("workspace", error)
            return AssistTurnLog.dead(self, f"workspace:{type(error).__name__}")
        client_turn = str(turn_id or "")
        conversation = str(conversation_id or "")
        if not conversation:
            conversation = hash_id("assist-conv-for-turn", client_turn) if client_turn else hash_id("assist-conv", uuid4().hex)
        conversation = _safe_identifier(conversation, "assist-conv")
        # A missing id is minted from a random token, never the clock: two
        # calls in one coarse clock tick (about 15.6 ms on Windows before
        # Python 3.13) must still get distinct conversations and turns.
        turn = _safe_identifier(client_turn or hash_id("assist-turn", uuid4().hex), "assist-turn")
        log = AssistTurnLog(self, ws, conversation, turn, ws_id=ws_id, question=question)
        if not self._acquire(conversation, cancelled):
            return None
        admitted = False
        try:
            log._admit()
            admitted = not log._closed
            return log
        finally:
            if not admitted:
                self._release(conversation)

    def _acquire(self, conversation: str, cancelled: Optional[threading.Event] = None) -> bool:
        """Claim the conversation until close. Waiters also pin it against eviction."""
        with self._lock:
            thread = threading.get_ident()
            if self._writes.running.get(conversation) == thread:
                raise ValueError(f"assistant conversation {conversation!r} is already running on this thread")
            self._active[conversation] = self._active.get(conversation, 0) + 1
            acquired = False
            try:
                while cancelled is None or not cancelled.is_set():
                    if conversation not in self._writes.running:
                        self._writes.running[conversation] = thread
                        acquired = True
                        return True
                    self._lock.wait(timeout=None if cancelled is None else 0.05)
                return False
            finally:
                if not acquired:
                    self._unpin(conversation)

    def _unpin(self, conversation: str) -> None:
        """Drop the caller's retention pin under the condition lock."""
        if self._active[conversation] == 1:
            del self._active[conversation]
        else:
            self._active[conversation] -= 1

    def _release(self, conversation: str) -> None:
        """Release the conversation and wake waiting requests."""
        with self._lock:
            del self._writes.running[conversation]
            self._unpin(conversation)
            self._lock.notify_all()

    # -------------------------------------------------------- projection

    def _load_session(self, ws: Any, conversation_id: str) -> Any:
        Session = self._entity("agent-session")
        row = ws.get(_session_ref(conversation_id), entity=Session)
        return Session.from_dict(row, ws=ws) if row is not None else None

    def _open_session(self, ws: Any, conversation_id: str, question: str, *, restore: bool = False) -> Any:
        """Load the session, restoring retained history or admitting a new conversation."""
        session = _read_for_admission(lambda: self._load_session(ws, conversation_id))
        if session is not None:
            return session
        if restore or _read_for_admission(lambda: self._trace_rows(ws, conversation_id=conversation_id, limit=1)):
            return self._rebuild(ws, conversation_id)
        with self._writes.retention:
            session = self._new_session(ws, conversation_id, question, started_at=int(time() * 1000))
            evicted = self._claim_evictions(ws)
            try:
                targets = self._retention_targets(ws, evicted)
                ws.upsert(type(session), session.to_dict())
                if targets:
                    ws.delete_many(targets)
            except BaseException:
                # The failure can arrive after the session row was written.
                # Remove that empty row before another admission counts it.
                try:
                    ws.delete_many([(type(session), session.object_id)])
                except Exception as error:
                    _warn_once("admission cleanup", error)
                raise
            finally:
                for conversation, _ in evicted:
                    self._release(conversation)
        return session

    def _new_session(self, ws: Any, conversation_id: str, question: str, *, started_at: int) -> Any:
        """Construct a projection session without publishing it."""
        Session = self._entity("agent-session")
        session = Session(object_id=_session_ref(conversation_id), title=question.strip()[:80], agent_ref=_SESSION_SOURCE, ws=ws)
        session._set(state={"source": _SESSION_SOURCE, "conversation_id": conversation_id, "started_at": started_at})
        return session

    def rebuild_session(self, conversation_id: str) -> Any:
        """Rebuild one conversation's Session/Message projection from trace rows.

        The trace rows are the authority; the projection is derived state. A
        lost update on the aggregate row, a crash between writes, or a
        skipped append therefore never damages the record — this method
        recomputes the readable transcript from the ``turn`` rows in seal
        order and replaces the projection wholesale. For older turns it
        preserves the order still recorded by the session and uses timestamps
        where that order is missing. Usage is re-aggregated from successful
        turns in that same order, including blank replies that add no
        messages. The merge matches live appends: numbers add up, and a
        number replaces a stored value such as null; any other value keeps
        the first non-empty one.

        Args:
            conversation_id (str): Logged conversation identity (as stored;
                a client id that was hash-normalized rebuilds under its
                stored form).

        Returns:
            Any: The rebuilt ``Session``.

        No trace row and no other conversation is touched: the trace rows
        are only read, and the conversation's own message rows are replaced
        by the rebuilt transcript. A missing projection row is recreated
        from the first retained turn summary's timestamp when available.
        Rebuilding skips retention and may put the log one conversation over
        ``session_log_max_sessions``. The next new admission evicts as many
        oldest idle conversations as needed and available, including rebuilt
        ones.

        Raises:
            ValueError: If the conversation has no trace rows at all.
        """
        ws = self.workspace()
        self._acquire(conversation_id)
        try:
            return self._rebuild(ws, conversation_id)
        finally:
            self._release(conversation_id)

    def _rebuild(self, ws: Any, conversation_id: str) -> Any:
        """Replace one conversation's projection with the transcript its trace rows hold.

        A missing projection row uses the first retained turn summary's
        timestamp when available. Restoring the row skips retention. The
        caller holds the conversation.
        """
        from heavenbase.extensions.agent.session import _merge_usage

        turns = self._trace_rows(ws, "seq", "turn_id", "ws_id", "payload", conversation_id=conversation_id, kind="turn")
        if not turns and not self._trace_rows(ws, conversation_id=conversation_id, limit=1):
            raise ValueError(f"conversation {conversation_id!r} has no trace rows to rebuild from")
        turns.sort(key=_turn_order)
        first_closed = _turn_order(turns[0])[2] if turns else 0
        session = self._load_session(ws, conversation_id)
        if session is not None:
            # For older rows without a seal sequence, preserve the order still
            # recorded by the session. Unprojected rows keep their time-based slots.
            positions = {turn: index for index, turn in enumerate(_projected_turns(session))}
            slots = [index for index, row in enumerate(turns) if not _turn_order(row)[1] and row["turn_id"] in positions]
            ordered = sorted((turns[index] for index in slots), key=lambda row: positions[row["turn_id"]])
            for index, row in zip(slots, ordered):
                turns[index] = row
        transcript: List[Dict[str, Any]] = []
        usage: Dict[str, Any] = {}
        projected: List[str] = []
        usage_only: List[str] = []
        last_ws = ""
        for row in turns:
            payload = row.get("payload") or {}
            if payload.get("error"):
                continue
            turn_id = str(row.get("turn_id") or "")
            turn_usage = payload.get("usage")
            usage = _merge_usage(usage, turn_usage)
            text = str(payload.get("text") or "")
            if not text.strip():
                if turn_usage:
                    usage_only.append(turn_id)
                continue
            transcript.append({"role": "user", "content": str(payload.get("question") or "")})
            transcript.append({"role": "assistant", "content": text})
            projected.append(turn_id)
            last_ws = str(row.get("ws_id") or "")
        if session is None:
            question = transcript[0]["content"] if transcript else ""
            session = self._new_session(ws, conversation_id, question, started_at=first_closed or int(time() * 1000))
        state = {"last_turn_id": projected[-1] if projected else "", "turn_ids": projected, "ws_id": last_ws, "usage_only_turn_ids": usage_only}
        # Keep the previous transcript readable until the replacement session row lands.
        session._set(message_ids=[])
        session.set_messages(transcript, source=_SESSION_SOURCE, usage=usage, state=state, ws=ws)
        self._prune_messages(ws, session)
        return session

    def _message_ids(self, ws: Any, Message: Any, session_id: str) -> set[str]:
        """Look up projection rows by their owning session, including unlisted rows (see :meth:`_select`)."""
        return {row["object_id"] for row in self._select(ws, Message, (), {"session_id": session_id})}

    def _prune_messages(self, ws: Any, session: Any) -> None:
        """Remove unlisted rows after the session has published its message IDs."""
        Message = self._entity("agent-message")
        stale = self._message_ids(ws, Message, session.object_id).difference(session.message_ids)
        if stale:
            ws.delete_many([(Message, message_id) for message_id in sorted(stale)])

    def _restore_turn(self, ws: Any, conversation_id: str, turn_id: str, payload: Mapping[str, Any]) -> None:
        """Rebuild the projection when a replayed turn is missing or differs from its trace.

        A turn marker locates the transcript pair or records a usage-only
        append. A rebuild keeps the conversation's age and reapplies seal
        order to all the turns.
        """
        session = self._load_session(ws, conversation_id)
        text = str(payload.get("text") or "")
        if session is not None and not text.strip():
            if not payload.get("usage") or turn_id in _usage_only_turns(session):
                return
        elif session is not None:
            turns = _projected_turns(session)
            if turn_id in turns and len(session.message_ids) == 2 * len(turns):
                start = 2 * turns.index(turn_id)
                rows = ws.get_many(session.message_ids[start : start + 2], entity=self._entity("agent-message"))
                question = str(payload.get("question") or "")
                expected = ({"role": "user", "content": question}, {"role": "assistant", "content": text})
                if len(rows) == 2 and all(
                    row is not None and row["session_id"] == session.object_id and row["idx"] == index and row["payload"] == message
                    for index, (row, message) in enumerate(zip(rows, expected), start)
                ):
                    self._prune_messages(ws, session)
                    return
        self._rebuild(ws, conversation_id)

    def _claim_evictions(self, ws: Any) -> List[Tuple[str, str]]:
        """Choose and hold the conversations a new admission evicts.

        The oldest conversations (by their projection row's
        ``state.started_at``) are evicted so that the log, with the session
        about to be written, holds at most ``session_log_max_sessions``. The
        projection rows are the conversation index: a row counts when its id
        is the deterministic projection id of the ``state.conversation_id``
        it records, whatever ``state.source`` a later edit left in it. A
        conversation whose projection row is missing is not counted and is
        never evicted, because its trace rows are the authoritative record
        that :meth:`rebuild_session` restores the projection from. Active
        and waiting conversations count and are ineligible. Each chosen
        conversation is held like a running turn until the caller releases
        it, so no turn starts on it while its rows are deleted; the condition
        lock is held only to choose and hold them. The caller holds
        ``retention``.

        Returns:
            List[Tuple[str, str]]: ``(conversation_id, session row id)`` pairs, oldest first.
        """
        cap = self._max_sessions(self.cm)
        if not cap:
            return []
        Session = self._entity("agent-session")
        indexed = []
        for row in ws.query(Session).select("object_id", "state").execute().rows():
            state = row.get("state") or {}
            conversation = str(state.get("conversation_id") or "")
            if conversation and row.get("object_id") == _session_ref(conversation):
                indexed.append((int(state.get("started_at") or 0), conversation, row["object_id"]))
        overflow = len(indexed) + 1 - cap  # the session about to be written counts too
        if overflow <= 0:
            return []
        indexed.sort(key=lambda item: item[0])
        with self._lock:
            evicted = [(conversation, session_id) for _, conversation, session_id in indexed if conversation not in self._active][:overflow]
            for conversation, _ in evicted:
                self._writes.running[conversation] = _EVICTION
                self._active[conversation] = 1
        return evicted

    def _retention_targets(self, ws: Any, evicted: List[Tuple[str, str]]) -> List[tuple]:
        """Plan whole-session eviction before publishing a new session.

        This is the recorder's single sanctioned bulk deletion: each evicted
        conversation (see :meth:`_claim_evictions`) is removed together with
        its message, trace and seal counter rows. Message rows are found by
        session id and trace rows by ``conversation_id``, reading only their
        ids where the backend allows (see :meth:`_select`).
        """
        if not evicted:
            return []
        Session, Message, AgentTrace = self._entity("agent-session"), self._entity("agent-message"), self._entity("agent-trace")
        seal = ws.entities[_SEAL_ENTITY]
        targets: List[tuple] = []
        for conversation, session_id in evicted:
            targets.extend((Message, message_id) for message_id in sorted(self._message_ids(ws, Message, session_id)))
            targets.extend((AgentTrace, trace_row["object_id"]) for trace_row in self._trace_rows(ws, conversation_id=conversation))
            targets.append((seal, _seal_ref(conversation)))
            targets.append((Session, session_id))
        return targets


class AssistTurnLog:
    """Recorder for one assistant turn: trace events plus the projection."""

    def __init__(self, owner: AssistSessionLog, ws: Any, conversation_id: str, turn_id: str, *, ws_id: str, question: str) -> None:
        self._owner = owner
        self._ws = ws
        self.conversation_id = conversation_id
        self.turn_id = turn_id
        self.ws_id = ws_id
        self.question = question
        self._state = trace_state(self._persist_event)
        self._failures: List[str] = []
        self._closed = False
        self._dead_reason = ""
        self._attempt = 1
        self._previous_error = ""
        self._replay_payload: Optional[Dict[str, Any]] = None

    @classmethod
    def dead(cls, owner: AssistSessionLog, reason: str) -> "AssistTurnLog":
        """Return a recorder that records nothing but confesses why.

        Its ``capture`` names the failure, so the reply still carries a
        ``capture_warning`` when the log workspace itself is unavailable.
        """
        log = cls(owner, None, "", "", ws_id="", question="")
        log._dead_reason = reason
        log._failures.append(reason)
        return log

    def llm_trace(self, call: Optional[str] = None) -> TraceCallback:
        """Return the trace callback for one LLM call of this turn.

        Args:
            call (str | None): Override tag for every event of that call
                (e.g. ``"replay_compaction"``); ``None`` keeps the LLM's own
                ``round``/``turn_compaction`` tags. All calls share this
                turn's monotonic sequence.

        Returns:
            TraceCallback: Callback to pass as ``LLM.chat(..., trace=...)``.
        """
        return forward_trace(self._state, call)

    def _admit(self) -> None:
        """Load the authoritative turn before opening an attempt or replaying."""
        trace = self._owner._entity("agent-trace")
        key = trace.event_id(self.conversation_id, self.turn_id, 0, "turn")
        existing = _read_for_admission(lambda: self._ws.get(key, entity=trace))
        if existing is not None:
            payload = self._matching_turn_payload(existing)
            if not payload.get("error"):
                self._replay_payload = payload
                self._closed = True
                self._restore_projection(payload)
                return
            self._previous_error = str(payload["error"])
            self._attempt = int(payload.get("attempt") or 1) + 1

        # A hard stop can leave call events without a summary. Check only the
        # attempt keys for this turn, including the distinct retry id namespace.
        while _read_for_admission(lambda: self._attempt_has_events(trace)):
            self._attempt += 1
        try:
            self._owner._open_session(self._ws, self.conversation_id, self.question, restore=existing is not None)
        except Exception as error:
            _warn_once("session", error)
            self._failures.append(f"session:{type(error).__name__}")

    def _attempt_has_events(self, trace: Any) -> bool:
        """Distinguish an interrupted attempt from a literal suffix-shaped turn id."""
        turn_key = self._event_turn_key()
        rows = self._owner._trace_rows(self._ws, "kind", "seq", conversation_id=self.conversation_id, turn_id=turn_key)
        return any(row["kind"] != "turn" and row["object_id"] == self._call_event_id(trace, row["seq"], row["kind"]) for row in rows)

    def _matching_turn_payload(self, row: Mapping[str, Any]) -> Dict[str, Any]:
        """Read a recorded turn's payload and enforce its retry identity."""
        payload = dict(row.get("payload") or {})
        if str(payload.get("question") or "") != self.question or str(row.get("ws_id") or "") != self.ws_id:
            raise ValueError(f"assistant turn {self.turn_id!r} is already recorded for a different question or workspace")
        return payload

    def replay(self) -> Optional[Dict[str, Any]]:
        """Return the completed turn as an assistant response without provider I/O.

        Args:
            None.

        Returns:
            Dict[str, Any] | None: Dashboard assistant response reconstructed
            from the completed turn row, or ``None`` when this turn still needs
            execution. Trace-only details such as prior reasoning and evidence
            frames are deliberately not projected into the replay response.
        """
        if self._replay_payload is None:
            return None
        payload = self._replay_payload
        reply = {
            "text": str(payload.get("text") or ""),
            "think": "",
            "served_model": str(payload.get("served_model") or ""),
            "frames": [],
            "sql_frames": [],
            "events": [],
            "elapsed": float(payload.get("elapsed") or 0.0),
            "truncated": bool(payload.get("truncated")),
        }
        # The stored capture describes the original recording; this replay's
        # own failure to restore the transcript is added to it.
        stored = str(payload.get("capture") or "complete")
        reasons = set(self._failures)
        if stored != "complete":
            reasons.update(reason for reason in stored.partition(":")[2].split(",") if reason)
        if reasons or stored != "complete":
            reply["capture_warning"] = "incomplete:" + ",".join(sorted(reasons)) if reasons else stored
        return reply

    def _restore_projection(self, payload: Mapping[str, Any]) -> None:
        """Restore this completed turn to the session projection when needed.

        A failure marks this delivery's capture (``projection:<Error>``), so
        the reply carries a ``capture_warning``.
        """
        try:
            self._owner._restore_turn(self._ws, self.conversation_id, self.turn_id, payload)
        except Exception as failure:
            _warn_once("projection", failure)
            self._failures.append(f"projection:{type(failure).__name__}")

    def _event_turn_key(self) -> str:
        """Return the attempt-namespaced turn key for request/terminal rows."""
        return self.turn_id if self._attempt == 1 else f"{self.turn_id}-a{self._attempt}"

    def _call_event_id(self, trace: Any, seq: int, kind: str) -> str:
        if self._attempt == 1:
            return trace.event_id(self.conversation_id, self.turn_id, seq, kind)
        return _retry_event_id(self.conversation_id, self.turn_id, self._attempt, seq, kind)

    def _persist_event(self, event: LLMTraceEvent) -> None:
        """Write one immutable event row; a replayed write is a no-op."""
        if self._dead_reason or self._closed:
            return
        try:
            AgentTrace = self._owner._entity("agent-trace")
            row = AgentTrace.event(
                self.conversation_id, self._event_turn_key(), event.seq, event.kind, call=event.call, ws_id=self.ws_id, payload=event.payload
            )
            row["object_id"] = self._call_event_id(AgentTrace, event.seq, event.kind)
            self._ws.upsert(AgentTrace, row, overwrite=False)
        except ValueError as error:
            if _is_duplicate_write(error):
                return  # the same event coordinates were already recorded (idempotent replay)
            _warn_once("event", error)
            self._failures.append(f"event:{type(error).__name__}")
        except Exception as error:
            _warn_once("event", error)
            self._failures.append(f"event:{type(error).__name__}")

    @property
    def capture(self) -> str:
        """Return ``"complete"`` or ``"incomplete:<reasons>"`` for this turn."""
        return "complete" if not self._failures else "incomplete:" + ",".join(sorted(set(self._failures)))

    def close(
        self,
        *,
        text: str = "",
        served_model: str = "",
        elapsed: float = 0.0,
        truncated: bool = False,
        usage: Optional[Mapping[str, Any]] = None,
        error: str = "",
    ) -> None:
        """Seal the turn: write its ``turn`` row and update the projection.

        Recording exceptions become capture warnings. A successful close supersedes a provisional
        interrupted/errored turn row left by an earlier attempt — the one
        sanctioned overwrite. The projection append is skipped for a turn the
        session already records, so a retry never doubles messages or usage.
        A failure after the turn row is written,
        such as a failed projection append, marks this turn's
        :attr:`capture` (``projection:<Error>`` or ``turn:<Error>``); the
        completed turn row stays, and a retry of the same turn replays it and
        restores the missing projection (see :meth:`AssistSessionLog._restore_turn`).

        Args:
            text (str): Final answer text. Defaults to ``""``.
            served_model (str): Provider-reported model. Defaults to ``""``.
            elapsed (float): Turn latency in seconds. Defaults to ``0.0``.
            truncated (bool): Whether the tool budget cut the answer short.
                Defaults to ``False``.
            usage (Mapping[str, Any] | None): Merged provider usage. Defaults
                to ``None``.
            error (str): Terminal failure description when the turn errored.
                Defaults to ``""``.
        """
        if self._dead_reason:
            return
        with self._owner._lock:
            if self._closed:
                return
            self._closed = True
        try:
            self._close_guarded(text=text, served_model=served_model, elapsed=elapsed, truncated=truncated, usage=usage, error=error)
        except Exception as failure:
            _warn_once("turn", failure)
            self._failures.append(f"turn:{type(failure).__name__}")
        finally:
            self._owner._release(self.conversation_id)

    def _close_guarded(self, *, text: str, served_model: str, elapsed: float, truncated: bool, usage: Optional[Mapping[str, Any]], error: str) -> None:
        """Seal the authoritative turn and append its projection before releasing the conversation."""
        if error and self._previous_error:
            return  # the first failed summary binds the question and workspace
        AgentTrace = self._owner._entity("agent-trace")
        seq = self._owner._next_seal(self._ws, self.conversation_id)
        payload = {
            "question": self.question,
            "text": text,
            "served_model": served_model,
            "elapsed": elapsed,
            "truncated": bool(truncated),
            "events": self._state["seq"],
            "capture": self.capture,
            "attempt": self._attempt,
            "closed_at": int(time() * 1000),
        }
        if usage:
            payload["usage"] = dict(usage)
        if error:
            payload["error"] = error
        if self._previous_error:
            payload["superseded"] = self._previous_error
        row = AgentTrace.event(self.conversation_id, self.turn_id, seq, "turn", ws_id=self.ws_id, payload=payload)
        self._ws.upsert(AgentTrace, row, overwrite=bool(self._previous_error))
        has_answer = bool(text.strip())
        if error or not (has_answer or usage):
            return
        try:
            session = self._owner._open_session(self._ws, self.conversation_id, self.question, restore=True)
            if has_answer:
                projected = _projected_turns(session)
                messages = [{"role": "user", "content": self.question}, {"role": "assistant", "content": text}]
                state = {"last_turn_id": self.turn_id, "ws_id": self.ws_id, "turn_ids": [*projected, self.turn_id]}
            else:
                projected = _usage_only_turns(session)
                messages = []
                state = {"usage_only_turn_ids": [*projected, self.turn_id]}
            if self.turn_id in projected:
                return  # the rebuild already projected this turn
            session.append_messages(messages, source=_SESSION_SOURCE, usage=usage, state=state, ws=self._ws)
        except Exception as failure:
            # The turn row already says "complete": this delivery's capture
            # tells the client when the projection missed the turn.
            _warn_once("projection", failure)
            self._failures.append(f"projection:{type(failure).__name__}")
