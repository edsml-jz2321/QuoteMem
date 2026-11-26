"""``quotemem``: serve, check, back up, rebuild and administer an agent store.

``quotemem serve`` runs the service; ``quotemem check`` resolves the same
configuration read-only (it creates, binds and writes nothing), probes both
providers once and prints a JSON report; it exits 1 with ``"ok": false`` on any
failed probe, including a reply from a model other than the configured one
(``model_mismatch``). ``quotemem export`` writes a store's
source journal (a backup and the input of a rebuild) and ``quotemem rebuild``
makes a new store from one, which is how a store moves to a new release
(``quotemem.agent.rebuild``). ``quotemem repair`` and ``quotemem delete-user`` ask the
running service, authenticated with its token, to repair one user's stopped Add owner or
to erase one user's memories; deletion asks for the user id again unless ``--yes`` is
given. ``quotemem memories`` lists one user's memories with their status and quotes, from
the running service or, when none on this computer answers, read-only from the stopped
store (``quotemem.agent.memories``). These three find the service and its token as the plugins
do (``quotemem.agent.client``): ``--url``, else ``QUOTEMEM_URL``, else
``http://127.0.0.1:8765``; ``QUOTEMEM_TOKEN``, then ``QUOTEMEM_TOKEN_FILE``, then, for a
service on this computer, ``<data-dir>/token``. For ``memories`` the user is ``--user``,
else ``QUOTEMEM_USER_ID``, else the OS user name (at most 128 characters), as the plugins
store it, else ``default``. ``--json`` prints the list, ``--html FILE`` writes one
self-contained page. Keys and tokens come only from the environment, the file
``QUOTEMEM_TOKEN_FILE`` names or the data directory, and are never printed.
"""

from __future__ import annotations

import argparse
import getpass
import os
import sys
from collections.abc import Mapping
from dataclasses import asdict, replace
from pathlib import Path
from typing import cast, get_args

import httpx
from pydantic import ValidationError

from quotemem.api.interactive_search import SEARCH_PROFILES
from quotemem.core.errors import QuoteMemError
from quotemem.core.extraction_policy import AddMode, ReviewMode


def add_agent_commands(
    commands: argparse._SubParsersAction[argparse.ArgumentParser],  # pyright: ignore[reportPrivateUsage]
) -> None:
    """Register the ``quotemem`` subcommands."""
    from quotemem.agent.serve import DEFAULT_PORT, DEFAULT_REQUEST_TIMEOUT

    url_help = f"The service (default: $QUOTEMEM_URL, else http://127.0.0.1:{DEFAULT_PORT})"

    operations = commands
    serve = operations.add_parser(
        "serve", help="Serve one agent memory store (provider keys only from the environment)",
    )
    check = operations.add_parser("check", help="Check configuration and probe the providers")
    for command in (serve, check):
        command.add_argument(
            "--data-dir", type=Path, default=None,
            help="The store directory (default: $QUOTEMEM_DATA_DIR or ~/.local/share/quotemem)",
        )
        command.add_argument("--host", default="127.0.0.1")
        command.add_argument("--port", type=int, default=DEFAULT_PORT,
                             help="0 selects a free port, reported on stdout")
        command.add_argument("--allow-remote", action="store_true",
                             help="Permit a non-loopback --host")
        command.add_argument("--user-id", default="default",
                             help="The user of requests that name none")
        command.add_argument("--default-top-k", type=int, default=10)
        command.add_argument("--request-timeout-seconds", type=float,
                             default=DEFAULT_REQUEST_TIMEOUT)
        command.add_argument("--embedding-dimensions", type=int, choices=(1024, 2048),
                             default=None, help="Vector dimension of a new store (1024)")
        command.add_argument("--add-mode", choices=get_args(AddMode), default=None,
                             help="Add mode of a new store (default: agent)")
        command.add_argument(
            "--review", choices=get_args(ReviewMode), default=None,
            help="Review of a new agent store: fast (local rules, no judge model call), "
                 "standard (the targeted judge) or full (the judge model for every new "
                 "memory) (default: $QUOTEMEM_REVIEW, else standard); fixed per store",
        )
    serve.add_argument("--search-profile", choices=SEARCH_PROFILES, default="interactive",
                       help="interactive (default): no model call besides the query "
                            "embedding, compact hits, a deadline; bench: the benchmark Search")
    serve.add_argument("--search-deadline-seconds", type=float, default=2.5,
                       help="The interactive Search deadline (default 2.5)")
    serve.add_argument("--concurrency", choices=("interactive", "bench"), default="interactive",
                       help="interactive (default): Search never waits for an Add")
    export = operations.add_parser(
        "export", help="Write the store's source journal (stop the service first)",
    )
    export.add_argument(
        "--data-dir", type=Path, default=None,
        help="The store directory (default: $QUOTEMEM_DATA_DIR or ~/.local/share/quotemem)",
    )
    export.add_argument("--out", type=Path, required=True,
                        help="A new journal file (JSON Lines, mode 0600); never replaced")
    export.add_argument(
        "--passphrase-env", default=None, metavar="VAR",
        help="Seal forgotten statements with a key derived from the passphrase in this "
             "environment variable (default: the key in <data dir>/journal.key)",
    )
    rebuild = operations.add_parser(
        "rebuild", help="Make a new store from an old store or a journal, with this release",
    )
    rebuild.add_argument("--from", dest="source", type=Path, required=True,
                         help="The old data directory, or a journal from quotemem export")
    rebuild.add_argument("--to", dest="target", type=Path, required=True,
                         help="A new, empty data directory, or this rebuild's unfinished one")
    rebuild.add_argument("--embedding-dimensions", type=int, choices=(1024, 2048),
                         default=None, help="Vector dimension (default: the old store's)")
    rebuild.add_argument("--add-mode", choices=get_args(AddMode), default=None,
                         help="Add mode of the new store (default: agent)")
    rebuild.add_argument("--review", choices=get_args(ReviewMode), default=None,
                         help="Review of the new store: fast, standard or full "
                              "(default: $QUOTEMEM_REVIEW, else standard)")
    rebuild.add_argument("--request-timeout-seconds", type=float,
                         default=DEFAULT_REQUEST_TIMEOUT)
    rebuild.add_argument("--dry-run", action="store_true",
                         help="Export and print the plan and cost estimate, then stop")
    rebuild.add_argument(
        "--key-file", type=Path, default=None,
        help="The key of a journal file: the exporting data directory's journal.key, or "
             "that data directory",
    )
    rebuild.add_argument(
        "--passphrase-env", default=None, metavar="VAR",
        help="The environment variable holding a journal file's passphrase",
    )
    repair = operations.add_parser(
        "repair", help="Inspect, resume or skip one user's stopped or quarantined Add owner; "
                       "a forget that needs repair runs again first",
    )
    delete = operations.add_parser(
        "delete-user", help="Erase one user's memories and sources from the running service",
    )
    for command in (repair, delete):
        command.add_argument("--user-id", required=True)
        command.add_argument(
            "--data-dir", type=Path, default=None,
            help="The service's store directory, for its token file (default: "
                 "$QUOTEMEM_DATA_DIR or ~/.local/share/quotemem)",
        )
        command.add_argument("--url", default=None, help=url_help)
        command.add_argument("--timeout-seconds", type=float, default=900.0)
    memories = operations.add_parser(
        "memories", help="List one user's memories (from the service, else the stopped store)",
    )
    memories.add_argument(
        "--data-dir", type=Path, default=None,
        help="The store directory (default: $QUOTEMEM_DATA_DIR or ~/.local/share/quotemem)",
    )
    memories.add_argument("--url", default=None, help=url_help)
    memories.add_argument(
        "--user", default=None,
        help="The user (default: $QUOTEMEM_USER_ID, else the OS user name, else default)",
    )
    memories.add_argument("--status", default="all",
                          choices=("all", "current", "past", "possibly_outdated"))
    memories.add_argument("--limit", type=int, default=None,
                          help="The most memories (default 50; with --html every one)")
    memories.add_argument("--json", action="store_true", help="Print the list response")
    memories.add_argument("--html", type=Path, default=None, metavar="FILE",
                          help="Write one read-only HTML snapshot (mode 0600)")
    repair.add_argument("--skip", action="store_true",
                        help="Skip the quarantined cycle instead of resuming its owner "
                             "(a forget that needs repair is run again, never skipped)")
    repair.add_argument("--inspect", action="store_true", help="Only report the owner's state")
    delete.add_argument("--yes", action="store_true", help="Do not ask for confirmation")
    for name, command in (("serve", serve), ("check", check), ("export", export),
                          ("rebuild", rebuild), ("repair", repair), ("delete-user", delete),
                          ("memories", memories)):
        command.set_defaults(agent_command=name)


def _options(options: argparse.Namespace) -> object:
    from quotemem.agent.serve import AgentServeOptions
    from quotemem.agent.settings import default_data_dir

    data_dir = cast(Path | None, options.data_dir) or default_data_dir(os.environ)
    resolved = AgentServeOptions(
        data_dir=data_dir.expanduser(), host=cast(str, options.host),
        port=cast(int, options.port), default_user_id=cast(str, options.user_id),
        default_top_k=cast(int, options.default_top_k),
        request_timeout_seconds=cast(float, options.request_timeout_seconds),
        dimensions=options.embedding_dimensions, add_mode=options.add_mode,
        review=options.review, allow_remote=cast(bool, options.allow_remote),
    )
    if options.agent_command != "serve":
        return resolved
    return replace(
        resolved, search_profile=options.search_profile, concurrency=options.concurrency,
        search_deadline_seconds=cast(float, options.search_deadline_seconds),
    )


def check(
    options: object, environment: Mapping[str, str], *,
    transport: httpx.BaseTransport | None = None,
) -> int:
    """Probe both providers once, read-only, and print the report; 0 only when all passed.

    Args:
        options: The resolved ``AgentServeOptions``.
        environment: Provider and token variables.
        transport: A provider transport replacing the network (tests).
    """
    from quotemem.agent.serve import AgentServeOptions, notice
    from quotemem.agent.serve import check as check_store

    assert isinstance(options, AgentServeOptions)
    report = check_store(options, environment, transport=transport)
    notice(report)
    return 0 if report["ok"] else 1


def _operator(options: argparse.Namespace) -> int:
    """``repair`` and ``delete-user``: one authenticated call to the running service."""
    from quotemem.agent.client import service_token, service_url, unreachable
    from quotemem.agent.serve import notice
    from quotemem.agent.settings import default_data_dir

    user_id = cast(str, options.user_id)
    data_dir = (cast(Path | None, options.data_dir) or default_data_dir(os.environ)).expanduser()
    url = service_url(cast(str | None, options.url), os.environ)
    token = service_token(url, data_dir, os.environ)
    if options.agent_command == "delete-user":
        if not options.yes:
            print(f"This erases every memory and source of user {user_id!r}; it cannot be "
                  "undone.\nType the user id to confirm: ", end="", file=sys.stderr, flush=True)
            if sys.stdin.readline().strip() != user_id:
                notice({"status": "cancelled", "code": "confirmation_required"}, error=True)
                return 1
        path, body = "/agent/admin/delete-user", {"user_id": user_id, "confirm": user_id}
    else:
        action = "inspect" if options.inspect else "skip" if options.skip else "resume"
        path, body = "/agent/admin/repair", {"user_id": user_id, "action": action}
    try:
        with httpx.Client(base_url=url, timeout=cast(float, options.timeout_seconds),
                          trust_env=False) as client:
            response = client.post(path, json=body,
                                   headers={"Authorization": f"Bearer {token}"})
    except (httpx.ConnectError, httpx.ConnectTimeout):
        raise unreachable(url) from None
    except httpx.HTTPError:
        raise QuoteMemError("service_unhealthy", "The service did not finish answering") from None
    try:
        document = response.json()
    except ValueError:
        document = {"code": "invalid_response"}
    ok = response.status_code == 200
    notice({"status": "done" if ok else "failed", "http_status": response.status_code,
            **(document if isinstance(document, dict) else {})}, error=not ok)
    return 0 if ok else 1


MAX_USER_ID_CHARS = 128
"""The longest user id the plugins derive from the OS user name."""


def _os_user() -> str | None:
    """The OS user name as the plugins use it (at most ``MAX_USER_ID_CHARS``), else None."""
    try:
        return getpass.getuser()[:MAX_USER_ID_CHARS] or None
    except (OSError, KeyError):
        return None


def memories_user(user: str | None, environment: Mapping[str, str]) -> str:
    """The user ``quotemem memories`` lists: ``--user``, else ``QUOTEMEM_USER_ID``, else the
    OS user name, the ids the plugins store under, else ``default``."""
    return user or environment.get("QUOTEMEM_USER_ID") or _os_user() or "default"


def _memories(options: argparse.Namespace) -> int:
    from quotemem.agent.client import service_url
    from quotemem.agent.memories import run_memories
    from quotemem.agent.settings import default_data_dir

    data_dir = (cast(Path | None, options.data_dir) or default_data_dir(os.environ)).expanduser()
    html = cast(Path | None, options.html)
    return run_memories(
        data_dir, url=service_url(cast(str | None, options.url), os.environ),
        user_id=memories_user(cast(str | None, options.user), os.environ),
        status=cast(str, options.status),
        limit=cast(int | None, options.limit), as_json=cast(bool, options.json),
        html_path=None if html is None else html.expanduser(), environment=os.environ,
    )


def _export(options: argparse.Namespace) -> int:
    from quotemem.agent.export import export_store
    from quotemem.agent.journal_key import passphrase_from
    from quotemem.agent.serve import notice
    from quotemem.agent.settings import default_data_dir

    data_dir = cast(Path | None, options.data_dir) or default_data_dir(os.environ)
    variable = cast(str | None, options.passphrase_env)
    result = export_store(
        data_dir.expanduser(), cast(Path, options.out).expanduser(),
        passphrase=None if variable is None else passphrase_from(os.environ, variable),
    )
    notice({"status": "exported", "out": str(options.out), "sha256": result.end.sha256,
            **result.end.counts.model_dump(), "skipped": asdict(result.skips),
            "key_id": result.key_id,
            "key": f"passphrase in {variable}" if result.key_file is None
            else str(result.key_file),
            "restore_with": f"--passphrase-env {variable}" if result.key_file is None
            else f"--key-file {result.key_file}"})
    return 0


def _rebuild(options: argparse.Namespace) -> int:
    from quotemem.agent.journal_key import KeySource, passphrase_from
    from quotemem.agent.rebuild import RebuildOptions, run_rebuild

    variable = cast(str | None, options.passphrase_env)
    keys = KeySource(
        key_file=cast(Path | None, options.key_file),
        passphrase=None if variable is None else passphrase_from(os.environ, variable),
        passphrase_variable=variable,
    )
    return run_rebuild(RebuildOptions(
        source=cast(Path, options.source), target=cast(Path, options.target),
        dimensions=options.embedding_dimensions, add_mode=options.add_mode,
        review=options.review, dry_run=cast(bool, options.dry_run),
        request_timeout_seconds=cast(float, options.request_timeout_seconds),
    ), os.environ, keys=keys)


def run_agent_command(options: argparse.Namespace) -> int | None:
    """Dispatch ``quotemem``; None for other commands."""
    if getattr(options, "agent_command", None) is None:
        return None
    from quotemem.agent.rebuild import describe_upgrade
    from quotemem.agent.serve import AgentServeOptions, notice, run_agent

    try:
        if options.agent_command == "export":
            return _export(options)
        if options.agent_command == "rebuild":
            return _rebuild(options)
        if options.agent_command in {"repair", "delete-user"}:
            return _operator(options)
        if options.agent_command == "memories":
            return _memories(options)
        resolved = _options(options)
        assert isinstance(resolved, AgentServeOptions)
        if options.agent_command == "check":
            return check(resolved, os.environ)
        return run_agent(resolved, os.environ)
    except (QuoteMemError, ValidationError, OSError, httpx.HTTPError) as error:
        detail: dict[str, object] = {"status": "failed", "code": (
            error.code if isinstance(error, QuoteMemError) else "configuration_failed"
        )}
        if isinstance(error, QuoteMemError):
            # Agent configuration messages name variables and fixes, never values.
            detail["message"] = str(error)
            detail.update(describe_upgrade(error.code) or {})
        notice(detail, error=True)
        return 1

