"""Content-free CLI delivery for manual existing-job status and purge.

A report goes to stdout. ``status`` exits 0, or 1 when its report carries a warning
(overdue traffic or a recorded retention violation). A failure prints only
``{"status": "failed", "code": ...}`` to stderr and exits 1; for an absent workspace
the code is ``workspace_not_found`` when the audit has no record of that job at that
path (check ``--workspace``, ``--job`` and ``--audit-path``), and ``purge_unconfirmed``
when the audit records its completion but no deletion receipt.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from quotemem.core.errors import QuoteMemError
from quotemem.service.job_commands import purge_job, status_job
from quotemem.service.ownership import OwnershipError


def default_audit_path() -> Path:
    """``$XDG_DATA_HOME/quotemem-bench/audit.db``, else under ``~/.local/share``."""
    data_home = os.environ.get("XDG_DATA_HOME", "")
    base = Path(data_home) if os.path.isabs(data_home) else Path("~/.local/share")
    return base / "quotemem-bench" / "audit.db"


def add_job_commands(
    commands: argparse._SubParsersAction[argparse.ArgumentParser],  # pyright: ignore[reportPrivateUsage]
) -> None:
    """Register explicit existing-workspace commands without resolving or creating paths."""
    status = commands.add_parser("status", help="Inspect an existing benchmark job")
    purge = commands.add_parser("purge", help="Delete a stopped job and record its audit receipt")
    for command in (status, purge):
        command.add_argument("--workspace", type=Path, required=True)
        command.add_argument("--job", required=True)
        command.add_argument(
            "--audit-path", type=Path, default=default_audit_path(),
            help="The audit database (default: $XDG_DATA_HOME/quotemem-bench/audit.db)",
        )
    purge.add_argument("--completed-at", required=True,
                       help="Actual completion copied from the platform console, with timezone")


def run_job_command(options: argparse.Namespace) -> int | None:
    """Dispatch before ordinary startup configuration, with no provider credentials."""
    if options.command != "bench" or options.bench_command not in {"status", "purge"}:
        return None
    try:
        if options.bench_command == "status":
            result = status_job(options.workspace, options.job, audit_path=options.audit_path)
            print(json.dumps(result.payload(), sort_keys=True), flush=True)
            return result.exit_code
        receipt = purge_job(options.workspace, options.job, completed_at=options.completed_at,
                            audit_path=options.audit_path)
        print(json.dumps(receipt.model_dump(mode="json"), sort_keys=True), flush=True)
        return 0
    except Exception as error:
        code = error.code if isinstance(error, QuoteMemError) else (
            "workspace_unavailable" if isinstance(error, OwnershipError) else "job_command_failed"
        )
        try:
            print(json.dumps({"status": "failed", "code": code}), file=sys.stderr, flush=True)
        except OSError:
            pass
        return 1
