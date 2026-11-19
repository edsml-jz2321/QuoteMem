"""Explicit HeavenBase composition, isolated storage roots, and privacy gates.

This module owns the explicit-authority boundary described by plan.md §2 and §10.
Upstream imports are lazy; AML additionally rejects any process-default Context.
Every AML consumer imports HeavenBase through this module before using entities.
It contains no application Writer or business workflow.
"""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from types import TracebackType
from typing import Any, Protocol, Self, cast

import heavenbase as hb
from heavenbase._initialization import configure_explicit_context

from aml.core.errors import AMLError
from aml.core.platform_files import OWNER_LOCK_NAME as OWNER_LOCK_NAME
from aml.core.platform_files import (
    ensure_private_directory,
    open_private_file,
    validate_private_path,
)

configure_explicit_context()

import yaml  # noqa: E402
from heavenbase.context import BootstrapSpec, Context  # noqa: E402
from heavenbase.utils.config.manager import ConfigManager  # noqa: E402

PRIVACY_KEYS = (
    "heavenbase.llm.cache.enabled",
    "heavenbase.dashboard.assistant.session_log",
)


class StorageOwner(Protocol):
    """Supply process ownership without coupling storage to a service adapter.

    Entry holds the root and exclusive lock until exit. Database preparation is
    separate so bootstrap can first validate the persisted configuration. The
    runtime borrows this owner; its caller releases it after runtime shutdown.
    """

    @property
    def root(self) -> Path:
        """Return the canonical owned workspace path."""
        ...

    def __enter__(self) -> Self:
        """Acquire process ownership and return this same owner."""
        ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Release ownership after all borrowed runtime resources close."""
        ...

    def probe_storage(self) -> object:
        """Verify the owned filesystem and SQLite capabilities."""
        ...

    def prepare_databases(self) -> None:
        """Prepare and pin both database identities before HeavenBase opens."""
        ...

    def validate(self) -> None:
        """Reject changed ownership, paths, identities, or permissions."""
        ...


def verify_privacy(config: ConfigManager) -> None:
    """Reject any effective privacy flag other than the Boolean false.

    Args:
        config: The owning Context configuration in its effective scope.

    Raises:
        RuntimeError: If either persistent plaintext feature is enabled or
            has a non-Boolean configuration value.
    """
    for key in PRIVACY_KEYS:
        if config.get(key, default=None) is not False:
            raise RuntimeError(f"Unsafe HeavenBase configuration: {key} must be Boolean false")


def force_privacy(config: ConfigManager, workspace_id: str) -> None:
    """Set and read back both actual scopes before any application workspace use."""
    for scope in (config.base_scope, f"{config.base_scope}.{workspace_id}"):
        for key in PRIVACY_KEYS:
            config.set(key, False, scope=scope)
        config.set("heavenbase.extensions.default", [], scope=scope)
        with config.scoped(scope):
            verify_privacy(config)


def _close_runtime_resources(
    close_context: Callable[[], None], keepers: list[sqlite3.Connection],
) -> None:
    def close_keeper(keeper: sqlite3.Connection) -> None:
        keeper.close()
        keepers.remove(keeper)

    with ExitStack() as stack:
        for keeper in reversed(keepers):
            stack.callback(close_keeper, keeper)
        close_context()


@dataclass
class _RuntimeCleanup:
    """Keep one partial or complete Context acquisition until disposal acknowledges."""

    context: Context
    keepers: list[sqlite3.Connection]
    context_closed: bool = False
    terminal_failure: bool = False

    def close(self) -> None:
        def close_context() -> None:
            if self.terminal_failure:
                raise RuntimeError("HeavenBase Context disposal remains incomplete")
            if self.context_closed:
                return
            try:
                self.context.close()
            except BaseException:
                # The pinned Context terminalizes even when disposal raises.
                # Its later no-op close cannot acknowledge those lost resources.
                self.terminal_failure = self.context._closed  # pyright: ignore[reportPrivateUsage]
                raise
            self.context_closed = True

        _close_runtime_resources(close_context, self.keepers)


# HeavenBase runs each read on its own pooled connection (SQLite WAL readers
# never block each other or the sole Writer). The bench pool declares 64
# workers plus reserved and request threads, so the pool must admit that many
# readers at once: the provider default (5 + 10 overflow) would queue them on
# the pool with a 30 s timeout. Overflow connections close when returned; the
# resident set stays at pool_size.
MAIN_POOL: dict[str, object] = {
    "pool_class": "queue", "pool_size": 64, "max_overflow": 64, "pool_timeout": 30,
    # HeavenBase's SQLite default pings every checkout ("SELECT 1"); a local file
    # never drops a connection, and under the normal-load registration the ping
    # was 6.5 percent of the process's running samples (harness X23, 2026-09-12).
    "pool_pre_ping": False,
}


@dataclass
class HBRuntime:
    """Own one Context, its workspace, and non-recycling SQLite keepers.

    Attributes:
        root: Explicit AML-owned directory containing both database files.
        context: Unique machine composition authority for this root.
        workspace: Registered HeavenBase workspace using only memory.db.
        owner: Optional borrowed process owner, held beyond runtime shutdown.
    """

    root: Path
    context: Context
    workspace: hb.HeavenBase
    _keepers: list[sqlite3.Connection] = field(repr=False)
    owner: StorageOwner | None = field(default=None, repr=False)
    _cleanup: _RuntimeCleanup = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._cleanup = _RuntimeCleanup(self.context, self._keepers)

    def close(self) -> None:
        """Attempt Context and every keeper; retain each failed resource for retry."""
        self._cleanup.close()

    def __enter__(self) -> HBRuntime:
        """Return the owned runtime for an explicit resource scope."""
        return self

    def __exit__(self, *_: object) -> None:
        """Release resources when the owning scope exits."""
        self.close()


def open_runtime(
    root: Path, workspace_id: str = "aml", *, owner: StorageOwner | None = None,
    retain_cleanup: Callable[[Callable[[], None]], None] | None = None,
) -> HBRuntime:
    """Open isolated SQLite storage with explicit bootstrap and privacy gates.

    Args:
        root: AML-owned directory; no sibling checkout or personal HB
            configuration is consulted.
        workspace_id: Stable workspace identity below this root.
        owner: Already entered owner with configuration-validated, prepared
            databases. The caller must retain it until this runtime closes.
        retain_cleanup: Optional bootstrap handoff of this same Context's close
            operation, before opening its backends. The caller retains it across
            failed initialization and must not release ownership until it succeeds.

    Returns:
        The runtime whose lifetime the caller must close.

    Raises:
        RuntimeError: If privacy, SQLite mode, or database path checks fail.
        ValueError: If HeavenBase rejects the bootstrap or workspace identity.
    """
    if workspace_id.strip().lower() in {"llm-cache", "assistant-log"}:
        raise ValueError("AML forbids HeavenBase plaintext cache and assistant-log workspaces")
    if owner is None:
        root = root.resolve()
        if os.name == "nt":
            root.parent.mkdir(parents=True, exist_ok=True)
            try:
                ensure_private_directory(root)
            except AMLError as error:
                raise RuntimeError("AML storage root must have private permissions") from error
        else:
            root.mkdir(parents=True, exist_ok=True, mode=0o700)
            if root.stat().st_mode & 0o077:
                raise RuntimeError("AML storage root must have private permissions")
        for filename in ("hb-system.db", "memory.db"):
            path = root / filename
            if os.name == "nt":
                if path.exists():
                    try:
                        validate_private_path(path)
                    except AMLError as error:
                        raise RuntimeError(
                            f"AML database must have private permissions: {filename}"
                        ) from error
                else:
                    descriptor = open_private_file(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                    os.close(descriptor)
            else:
                if path.exists() and path.stat().st_mode & 0o077:
                    raise RuntimeError(f"AML database must have private permissions: {filename}")
                if not path.exists():
                    path.touch(mode=0o600, exist_ok=False)
    else:
        root = root.parent.resolve() / root.name
        if owner.root != root:
            raise ValueError("Storage owner does not match the runtime root")
        owner.validate()
    packaged = resources.files("heavenbase").joinpath("resources/configs/bootstrap.yaml")
    bootstrap = cast(dict[str, Any], yaml.safe_load(packaged.read_text()))
    backend = cast(dict[str, Any], bootstrap["backend"])
    backend["database"] = (root / "hb-system.db").as_uri()
    config = ConfigManager(root=str(root), setup=True)
    # Apply before binding the deferred Registry, so even initial defaults are safe.
    config.default["heavenbase"]["llm"]["cache"]["enabled"] = False
    config.default["heavenbase"]["dashboard"]["assistant"]["session_log"] = False
    config.default["heavenbase"]["extensions"]["default"] = []
    context = Context(
        BootstrapSpec(str(root), workspace_id, backend, bootstrap["registry"]), config=config
    )
    keepers: list[sqlite3.Connection] = []
    cleanup = _RuntimeCleanup(context, keepers)
    try:
        if retain_cleanup is not None:
            retain_cleanup(cleanup.close)
        force_privacy(config, workspace_id)
        workspace: Any = hb.HeavenBase(
            workspace_id,
            context=context,
            backends={"main": {
                "type": "sqlite", "database": (root / "memory.db").as_uri(),
                "pool": dict(MAIN_POOL),
            }},
        )
        with config.scoped(f"{config.base_scope}.{workspace_id}"):
            verify_privacy(config)
        for storage in (context.backend(), workspace.backends.get("main")):
            if storage.db.execute("PRAGMA journal_mode", readonly=True).scalar() != "wal":
                raise RuntimeError("HeavenBase SQLite connection is not in WAL mode")
            if storage.db.execute("PRAGMA synchronous", readonly=True).scalar() != 1:
                raise RuntimeError("HeavenBase SQLite connection is not synchronous NORMAL")
        for filename in ("hb-system.db", "memory.db"):
            path = root / filename
            if not path.is_file():
                raise RuntimeError(f"HeavenBase did not use the explicit database path: {filename}")
            # Startup owns all keeper SQL; the service may hand its serialized
            # final disposal to the already-owned close executor.
            keeper = sqlite3.connect(path, check_same_thread=False)
            keepers.append(keeper)
            mode = keeper.execute("PRAGMA journal_mode").fetchone()[0]
            keeper.execute("PRAGMA synchronous=NORMAL")
            synchronous = keeper.execute("PRAGMA synchronous").fetchone()[0]
            if mode != "wal" or synchronous != 1:
                raise RuntimeError(f"SQLite runtime gate failed for {filename}")
        if owner is not None:
            owner.validate()
        runtime = HBRuntime(root, context, workspace, keepers, owner)
        runtime._cleanup = cleanup  # pyright: ignore[reportPrivateUsage]
        return runtime
    except BaseException:
        cleanup.close()
        raise
