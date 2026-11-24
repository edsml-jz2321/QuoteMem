"""``quotemem memories``: list one user's memories, as text, JSON or one HTML snapshot.

By default it asks the running service (``POST /memory/list``), found and
authenticated as the plugins do (`quotemem.agent.client`: ``--url``, else ``QUOTEMEM_URL``,
else ``http://127.0.0.1:8765``; ``QUOTEMEM_TOKEN``, ``QUOTEMEM_TOKEN_FILE`` or
``<data-dir>/token``). When no service on this computer answers there, it reads the
store itself, read-only (`read_offline`): under the store's process lock, so it refuses
while a service or another command holds the store, and through `ExistingStore`, which
opens only a cleanly closed store and never writes. Both give the same list: the
statuses, quotes and order of `quotemem.api.memory_list`, with nothing forgotten (offline,
quotes carry no speaker name, only the role).

Output: one block per memory (status, statement, dates, at most three quotes as
"who, date: text"), then the counts, every control, format and line-separator character
of stored text written as an escape (`printable`); ``--json`` prints the list response;
``--html FILE`` writes one self-contained, read-only HTML page with the same data (no
network, no token, inline CSS, a filter box in inline script over the page itself; light
and dark), mode 0600. ``--limit`` is the most memories listed (default 50; with
``--html`` every one). Every output says where the list came from: the text ends with
one ``source:`` line, and ``--json`` and ``--html``'s report carry ``source``, either
``{"kind": "service", "url": ...}`` or ``{"kind": "store", "data_dir": ...}``.
"""

from __future__ import annotations

import html
import json
import os
import time
import unicodedata
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import ExitStack, contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, cast

import httpx

from quotemem.api.memory_list import (
    MAX_LIST_LIMIT,
    MemoryListRequestV1,
    MemoryListV1,
    StatusCountsV1,
    listing_page,
)
from quotemem.core.errors import QuoteMemError
from quotemem.core.platform_files import open_private_file
from quotemem.core.query import ScopeV1

QUOTES_SHOWN: Final = 3
STATUS_LABELS: Final[Mapping[str, str]] = {
    "current": "current", "past": "past", "possibly_outdated": "possibly outdated",
}


def collect(
    page: Callable[[MemoryListRequestV1], MemoryListV1], *, user_id: str, status: str,
    limit: int | None,
) -> MemoryListV1:
    """Page through one listing and merge it into a single list response.

    Args:
        page: Answers one page request.
        user_id: The user.
        status: The status filter.
        limit: The most memories; None for every one.

    Returns:
        The merged list; ``next_cursor`` is the last page's when ``limit`` cut it short.
    """
    memories: list[Any] = []
    cursor: str | None = None
    counts = StatusCountsV1()
    while True:
        wanted = MAX_LIST_LIMIT if limit is None else min(MAX_LIST_LIMIT, limit - len(memories))
        answered = page(MemoryListRequestV1.model_validate({
            "user_id": user_id, "status": status, "limit": wanted, "cursor": cursor,
        }))
        memories.extend(answered.memories)
        counts, cursor = answered.counts, answered.next_cursor
        if cursor is None or (limit is not None and len(memories) >= limit):
            return MemoryListV1(memories=memories, next_cursor=cursor, counts=counts)


def fetch_online(
    url: str, token: str, *, timeout_seconds: float = 30.0,
    transport: httpx.BaseTransport | None = None,
) -> Callable[[MemoryListRequestV1], MemoryListV1]:
    """A page function over the running service's ``/memory/list``.

    Raises (from the page function):
        httpx.ConnectError: When no service answers at ``url``.
        QuoteMemError: With the service's error code for any answer but 200.
    """
    def page(request: MemoryListRequestV1) -> MemoryListV1:
        with httpx.Client(base_url=url, timeout=timeout_seconds, trust_env=False,
                          transport=transport) as client:
            response = client.post("/memory/list", content=request.model_dump_json(),
                                   headers={"Authorization": f"Bearer {token}",
                                            "Content-Type": "application/json"})
        if response.status_code != 200:
            try:
                code = cast(dict[str, Any], response.json()).get("code")
            except ValueError:
                code = None
            raise QuoteMemError(code if isinstance(code, str) else "service_unhealthy",
                                f"The service answered {response.status_code}")
        return MemoryListV1.model_validate_json(response.content)

    return page


@contextmanager
def read_offline(data_dir: Path) -> Iterator[Callable[[MemoryListRequestV1], MemoryListV1]]:
    """Read a stopped store's listing, read-only, under the store's process lock.

    Yields a page function over one cold read root per user, built from the store's
    committed rows (`ExistingStore.domain_rows`). Nothing is written: no database, log,
    lock content or key. The forget shadow is keyed with the store's own tenant key, so
    exactly what the service would drop is dropped.

    Raises:
        QuoteMemError: ``reference_missing`` without a store; ``workspace_locked`` while a
            service or another command owns it; ``service_unhealthy`` for a store left
            by a crash (start the service once to recover it); ``index_rebuilding``
            while a forget or erasure of the user is unfinished.
    """
    from quotemem.agent.settings import AGENT_JOB_ID
    from quotemem.core.existing_job import open_existing_job
    from quotemem.providers.keyring import LocalKeyring
    from quotemem.service.ownership import OwnershipError, WorkspaceOwner

    workspace = data_dir.expanduser() / "workspace"
    if not (workspace / "memory.db").is_file():
        raise QuoteMemError("reference_missing", "No QuoteMem store is in this data directory; "
                            "pass the service's --url (or set QUOTEMEM_URL), or the store's "
                            "--data-dir (or set QUOTEMEM_DATA_DIR)")
    selected = workspace.resolve(strict=True)
    with ExitStack() as stack:
        try:
            resources = stack.enter_context(open_existing_job(
                selected, AGENT_JOB_ID, owner=WorkspaceOwner(selected, profile="bench")))
        except OwnershipError as error:
            if isinstance(error.__cause__, BlockingIOError):
                raise QuoteMemError("workspace_locked", "Another process owns this store; a "
                                    "service running at another address? Pass its --url or set "
                                    "QUOTEMEM_URL") from None
            raise
        keyring = LocalKeyring.load(workspace / "tenant.key")
        roots: dict[str, Any] = {}

        def page(request: MemoryListRequestV1) -> MemoryListV1:
            return _offline_page(resources.store, keyring.digest, request, roots)

        yield page


def _offline_page(
    store: Any, bind: Callable[[str, object], str], request: MemoryListRequestV1,
    roots: dict[str, Any],
) -> MemoryListV1:
    from quotemem.api.bench_search import agent_context, scope_withdrawn
    from quotemem.core.recall import agent_listing
    from quotemem.core.recall_agent import SHADOW_PURPOSE

    scope = request.scope
    if store.get_control(f"scope:{scope.scope_id}") is None:
        return MemoryListV1(memories=[], next_cursor=None, counts=StatusCountsV1())
    if store.controls(f"erasure:{scope.scope_id}:") or store.controls(
            f"maintenance:{scope.scope_id}:"):
        raise QuoteMemError("index_rebuilding",
                            "A forget or erasure of this user is unfinished; start the service")
    now = time.time_ns() // 1_000_000
    root = roots.get(scope.scope_id)
    if root is None:
        root = roots[scope.scope_id] = offline_root(store, scope)
    listed = agent_listing(
        root, scope_kind=scope.scope_kind, scope_id=scope.scope_id, now=now,
        context=agent_context(store, scope, lambda term: bind(SHADOW_PURPOSE, term), bind),
        withdrawn=scope_withdrawn(store, scope),
    )
    return listing_page(root, listed, request, now=now, speakers=lambda _page: {})


def offline_root(store: Any, scope: ScopeV1) -> Any:
    """A cold read root of one user scope from a stopped store's committed rows.

    It projects the scope's rows as the read model's cold rebuild does
    (``cold_projections``, ``SourceRowV2.from_evidence``) under the store's released
    recall policy and embedding state.
    """
    from quotemem.core.controls import ControlRowV1, parse_control_value
    from quotemem.core.embedding import embedding_state_token
    from quotemem.core.identity import integrity_digest
    from quotemem.core.recall_policy import normalize_recall_policy
    from quotemem.core.recall_snapshot import (
        RecallPolicySelection,
        RecallSnapshot,
        ScopeFence,
        SourceRowV2,
    )
    from quotemem.core.retained_policy import read_released_recall_config
    from quotemem.store.domain import EvidenceRowV1
    from quotemem.store.snapshot_rows import cold_projections

    policy = store.get_control("policy:recall")
    embedding = store.get_control("policy:embedding_space")
    counter = store.get_control("commit_seq")
    if policy is None or embedding is None or counter is None:
        raise QuoteMemError("service_unhealthy", "The store lacks its initialized policy")
    reference = policy["value"]["current_release_ref"]
    normalized = normalize_recall_policy(read_released_recall_config(store, reference))
    selected = scope.scope_id

    def rows(kind: str) -> Iterator[dict[str, Any]]:
        for row in store.domain_rows(kind):
            if kind in {"revision", "link"} or row.get("scope_id") == selected:
                yield row

    projected = cold_projections(rows)
    sources = [SourceRowV2.from_evidence(EvidenceRowV1.model_validate(raw).model_dump(
        mode="json")) for raw in rows("evidence")]
    return RecallSnapshot.rebuild(
        generation=cast(Any, parse_control_value("commit_seq", counter["value"])).seq,
        scopes=[ScopeFence(scope_kind="user", scope_id=selected, scope_seq=0, erase_epoch=0,
                           consent_epoch=0)],
        embedding_space=embedding["value"]["embedding_space"],
        embedding_token=embedding_state_token(ControlRowV1.model_validate(embedding)),
        recall_policy=RecallPolicySelection.from_snapshot(
            reference, normalized.config_digest, normalized, integrity_digest(normalized)),
        sources=sources, memories=projected.memories, entities=projected.entities,
        names=projected.names, held=projected.held,
    )


# Rendering ---------------------------------------------------------------------------

def _date(ms: int | None) -> str:
    if ms is None:
        return "date unknown"
    return datetime.fromtimestamp(ms / 1000, tz=UTC).strftime("%Y-%m-%d")


def _who(quote: Mapping[str, Any]) -> str:
    return cast(str | None, quote.get("speaker")) or cast(str | None, quote.get("role")) \
        or "unknown"


_ESCAPED_CATEGORIES: Final = frozenset({"Cc", "Cf", "Cs", "Co", "Cn", "Zl", "Zp"})
"""Characters a terminal may act on or that break a line: control and format characters
(escape sequences, bidirectional overrides), surrogates, private and unassigned code
points, and line or paragraph separators."""
_NAMED_ESCAPES: Final[Mapping[str, str]] = {"\n": "\\n", "\r": "\\r", "\t": "\\t"}


def printable(text: str) -> str:
    """``text`` with every character of `_ESCAPED_CATEGORIES` written as an escape.

    Stored text is untrusted: a user turn can carry terminal escape sequences (an OSC 52
    clipboard write, hidden text) or newlines that forge listing lines. Spaces of every
    script stay as they are.
    """
    if all(unicodedata.category(character) not in _ESCAPED_CATEGORIES for character in text):
        return text
    return "".join(
        character if unicodedata.category(character) not in _ESCAPED_CATEGORIES
        else _NAMED_ESCAPES.get(character) or (
            f"\\x{ord(character):02x}" if ord(character) < 0x100
            else f"\\u{ord(character):04x}" if ord(character) < 0x10000
            else f"\\U{ord(character):08x}")
        for character in text)


def render_text(document: MemoryListV1) -> str:
    """One block per memory, then the counts; every stored text passes `printable`."""
    blocks: list[str] = []
    for memory in document.memories:
        lines = [f"[{STATUS_LABELS[memory.status]}] {printable(memory.statement)}",
                 f"  remembered {_date(memory.created_at_ms)}, changed "
                 f"{_date(memory.updated_at_ms)}; id {printable(memory.memory_id)}"]
        if memory.superseded_by is not None:
            lines.append(f"  replaced by {printable(memory.superseded_by)}")
        for quote in memory.quotes[:QUOTES_SHOWN]:
            dumped = quote.model_dump()
            lines.append(f"  - {printable(_who(dumped))}, {_date(quote.said_at_ms)}: "
                         f"{printable(quote.text)}")
        blocks.append("\n".join(lines))
    counts = document.counts
    total = counts.current + counts.past + counts.possibly_outdated
    tail = (f"{total} memories: {counts.current} current, {counts.past} past, "
            f"{counts.possibly_outdated} possibly outdated")
    if document.next_cursor is not None:
        tail += f" (showing {len(document.memories)}; raise --limit for more)"
    return "\n\n".join([*blocks, tail]) + "\n"


_STYLE: Final = """
:root{--bg:#fbfaf7;--fg:#1d1c1a;--muted:#6b675f;--card:#ffffff;--line:#e4e0d8;
--current:#1f7a4a;--past:#8a8478;--outdated:#a8620b;color-scheme:light dark}
@media (prefers-color-scheme:dark){:root{--bg:#161513;--fg:#ece9e2;--muted:#a19c91;
--card:#1f1e1b;--line:#34322d;--current:#5cc48d;--past:#8f8a80;--outdated:#e0a24f}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:860px;margin:0 auto;padding:24px 16px 48px}
h1{font-size:1.4rem;margin:0 0 4px}.sub{color:var(--muted);margin:0 0 16px}
.bar{display:flex;flex-wrap:wrap;gap:8px;margin:0 0 16px}
input,select{font:inherit;color:inherit;background:var(--card);border:1px solid var(--line);
border-radius:6px;padding:6px 10px}input{flex:1 1 220px}
article{background:var(--card);border:1px solid var(--line);border-radius:8px;
padding:12px 14px;margin:0 0 10px}article[hidden]{display:none}
.st{font-size:.78rem;font-weight:600;text-transform:uppercase;letter-spacing:.04em}
.current .st{color:var(--current)}.past .st{color:var(--past)}
.possibly_outdated .st{color:var(--outdated)}.past .stmt{color:var(--muted)}
.stmt{margin:2px 0 6px;font-size:1.02rem}.meta{color:var(--muted);font-size:.85rem}
ul{margin:6px 0 0;padding-left:18px}li{margin:2px 0;overflow-wrap:anywhere}
q{quotes:none}.who{color:var(--muted)}
"""

_SCRIPT: Final = """
(function(){var q=document.getElementById('q'),s=document.getElementById('s');
function f(){var t=q.value.toLowerCase(),w=s.value;
document.querySelectorAll('article').forEach(function(a){
a.hidden=!((w==='all'||a.dataset.status===w)&&a.textContent.toLowerCase().indexOf(t)>=0);});}
q.addEventListener('input',f);s.addEventListener('change',f);})();
"""


def render_html(document: MemoryListV1, *, user_id: str, generated_at_ms: int) -> str:
    """One self-contained, read-only HTML page of a listing; every text is escaped."""
    def esc(value: str) -> str:
        return html.escape(value, quote=True)

    cards: list[str] = []
    for memory in document.memories:
        quotes = "".join(
            f'<li><span class="who">{esc(_who(quote.model_dump()))}, '
            f"{esc(_date(quote.said_at_ms))}:</span> <q>{esc(quote.text)}</q></li>"
            for quote in memory.quotes)
        replaced = "" if memory.superseded_by is None else (
            f"; replaced by {esc(memory.superseded_by)}")
        cards.append(
            f'<article class="{esc(memory.status)}" data-status="{esc(memory.status)}">'
            f'<div class="st">{esc(STATUS_LABELS[memory.status])}</div>'
            f'<p class="stmt">{esc(memory.statement)}</p>'
            f'<div class="meta">remembered {esc(_date(memory.created_at_ms))}, changed '
            f"{esc(_date(memory.updated_at_ms))}{replaced}</div>"
            f"{'<ul>' + quotes + '</ul>' if quotes else ''}</article>")
    counts = document.counts
    summary = (f"{counts.current} current, {counts.past} past, "
               f"{counts.possibly_outdated} possibly outdated; snapshot of "
               f"{_date(generated_at_ms)}")
    if document.next_cursor is not None:
        summary += f"; {len(document.memories)} shown"
    return (
        "<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
        "<meta http-equiv=\"Content-Security-Policy\" content=\"default-src 'none'; "
        "style-src 'unsafe-inline'; script-src 'unsafe-inline'\">"
        f"<title>QuoteMem memories</title><style>{_STYLE}</style></head><body><main>"
        f"<h1>Memories of {esc(user_id)}</h1><p class=\"sub\">{esc(summary)}</p>"
        "<div class=\"bar\"><input id=\"q\" type=\"search\" placeholder=\"Filter\" "
        "aria-label=\"Filter memories\"><select id=\"s\" aria-label=\"Status\">"
        "<option value=\"all\">All</option><option value=\"current\">Current</option>"
        "<option value=\"past\">Past</option><option value=\"possibly_outdated\">"
        "Possibly outdated</option></select></div>"
        + "".join(cards)
        + f"</main><script>{_SCRIPT}</script></body></html>\n"
    )


def write_private(path: Path, text: str) -> None:
    """Write ``text`` to ``path``, readable only by this user (a file there is replaced).

    POSIX gives the file mode 0600. Windows has no ``fchmod``: the text goes to a new
    file with this user's private ACL beside ``path``, which then replaces ``path``.
    """
    if os.name == "nt":
        staged = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        descriptor = open_private_file(staged, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                descriptor = -1
                stream.write(text.encode("utf-8"))
            os.replace(staged, path)
        except BaseException:
            if descriptor >= 0:
                os.close(descriptor)
            staged.unlink(missing_ok=True)
            raise
        return
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(
        os, "O_NOFOLLOW", 0), 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            descriptor = -1
            stream.write(text)
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def list_memories(
    data_dir: Path, *, url: str, user_id: str, status: str, limit: int | None,
    environment: Mapping[str, str], transport: httpx.BaseTransport | None = None,
) -> tuple[MemoryListV1, str]:
    """The listing from the running service, else from the stopped store; and its source.

    The service and its token are found as the plugins find them (`quotemem.agent.client`).
    Only a service on this computer falls back to ``data_dir``: the store of a service
    elsewhere is not there. Without a token, the store is read only when nothing answers
    at the URL: a service that answers there owns the list, maybe of another store. A
    service on this computer that serves (``/agent/ready``) and refuses the token while
    no process holds ``data_dir``'s store serves another store, so the stopped store is
    read; while a process holds it, or while whatever answers there does not serve (a
    QuoteMem service still starting checks the token before it opens a store, or another
    program), the refusal stands.

    Returns:
        The merged list and ``service``, ``store`` (nothing answers at the URL) or
        ``store_not_served`` (the service there refused the token and does not hold
        the store).

    Raises:
        QuoteMemError: The service's error, ``service_unhealthy`` when it stops answering
            (a read timeout or a broken connection), ``service_unreachable`` when no
            service on another computer answers, the token's refusals
            (`quotemem.agent.client.service_token`), ``authentication_failed`` when a service
            answers but no token was found; offline, `read_offline`'s refusals.
    """
    from quotemem.agent.client import (
        is_local,
        service_answers,
        service_serving,
        service_token,
        unreachable,
    )

    local = is_local(url)
    try:
        token: str | None = service_token(url, data_dir, environment)
    except QuoteMemError as error:
        # No token yet: a store on this computer that was never served is read directly.
        if (not local or error.code != "authentication_failed"
                or environment.get("QUOTEMEM_TOKEN_FILE", "").strip()):
            raise
        if service_answers(url, transport=transport):
            raise QuoteMemError("authentication_failed", f"A service answers at {url}, but no "
                                "token for it was found: set QUOTEMEM_TOKEN_FILE to a file "
                                "holding its token, or pass its --data-dir (or set "
                                "QUOTEMEM_DATA_DIR)") from None
        token = None
    refused: QuoteMemError | None = None
    if token is not None:
        try:
            return collect(fetch_online(url, token, transport=transport), user_id=user_id,
                           status=status, limit=limit), "service"
        except (httpx.ConnectError, httpx.ConnectTimeout):
            if not local:
                raise unreachable(url) from None
        except httpx.HTTPError:
            # It answered, then stalled or broke off: never fall back to a store it holds.
            raise QuoteMemError("service_unhealthy",
                                "The service did not finish answering the listing") from None
        except QuoteMemError as error:
            if not local or error.code != "authentication_failed":
                raise
            if not service_serving(url, transport=transport):
                raise QuoteMemError("authentication_failed", f"The service at {url} refused the "
                                    "token and is not serving yet, or is not a QuoteMem service; "
                                    "run the command again once it serves") from None
            refused = error
    with ExitStack() as stack:
        try:
            page = stack.enter_context(read_offline(data_dir))
        except QuoteMemError as error:
            # A refusing service that holds this store (or a data directory without
            # one) leaves the service's refusal as the answer.
            if refused is not None and error.code in {"workspace_locked", "reference_missing"}:
                raise refused from None
            raise
        return (collect(page, user_id=user_id, status=status, limit=limit),
                "store" if refused is None else "store_not_served")


def run_memories(
    data_dir: Path, *, url: str, user_id: str, status: str, limit: int | None,
    as_json: bool, html_path: Path | None, environment: Mapping[str, str],
) -> int:
    """``quotemem memories``: print or write one user's listing; 0 on success."""
    if limit is None and html_path is None:
        limit = 50
    if limit is not None and limit < 1:
        raise QuoteMemError("invalid_request", "--limit must be at least 1")
    document, source = list_memories(data_dir, url=url, user_id=user_id, status=status,
                                     limit=limit, environment=environment)
    origin: dict[str, str] = ({"kind": "service", "url": url} if source == "service"
                              else {"kind": "store", "data_dir": str(data_dir.expanduser())})
    why = (f"no service answers at {printable(url)}" if source == "store"
           else f"the service at {printable(url)} does not hold it")
    if html_path is not None:
        write_private(html_path, render_html(document, user_id=user_id,
                                             generated_at_ms=time.time_ns() // 1_000_000))
        print(json.dumps({"status": "written", "html": str(html_path),
                          "memories": len(document.memories), "source": origin},
                         sort_keys=True))
        return 0
    if as_json:
        print(json.dumps({**document.model_dump(mode="json"), "source": origin},
                         ensure_ascii=False, separators=(",", ":")))
    else:
        print(render_text(document), end="")
        print("source: " + (f"the service at {printable(url)}" if source == "service" else
                            f"the stopped store in {printable(origin['data_dir'])} ({why})"))
    return 0
