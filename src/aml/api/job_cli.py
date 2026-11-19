"""Content-free CLI delivery for manual existing-job status and purge."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from aml.core.errors import AMLError
from aml.service.job_commands import purge_job, status_job
from aml.service.ownership import OwnershipError


def add_job_commands(
    commands: argparse._SubParsersAction[argparse.ArgumentParser],  # pyright: ignore[reportPrivateUsage]
) -> None:
    """Register explicit existing-workspace commands without resolving or creating paths."""
    status = commands.add_parser("status", help="Inspect an existing benchmark job")
    bench = commands.add_parser("bench", help="Manual benchmark job operations")
    operations = bench.add_subparsers(dest="bench_command", required=True)
    purge = operations.add_parser("purge", help="Delete a stopped job and record its audit receipt")
    for command in (status, purge):
        command.add_argument("--workspace", type=Path, required=True)
        command.add_argument("--job", required=True)
        command.add_argument("--audit-path", type=Path, default=Path("~/.aml/audit.db"))
    purge.add_argument("--completed-at", required=True,
                       help="Actual completion copied from the platform console, with timezone")


def run_job_command(options: argparse.Namespace) -> int | None:
    """Dispatch before ordinary startup configuration, with no provider credentials."""
    if options.command not in {"status", "bench"}:
        return None
    try:
        if options.command == "status":
            result = status_job(options.workspace, options.job, audit_path=options.audit_path)
            print(json.dumps(result.payload(), sort_keys=True), flush=True)
            return result.exit_code
        receipt = purge_job(options.workspace, options.job, completed_at=options.completed_at,
                            audit_path=options.audit_path)
        print(json.dumps(receipt.model_dump(mode="json"), sort_keys=True), flush=True)
        return 0
    except Exception as error:
        code = error.code if isinstance(error, AMLError) else (
            "workspace_unavailable" if isinstance(error, OwnershipError) else "job_command_failed"
        )
        try:
            print(json.dumps({"status": "failed", "code": code}), file=sys.stderr, flush=True)
        except OSError:
            pass
        return 1
