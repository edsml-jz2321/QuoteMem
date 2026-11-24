"""Root `hb` commands."""

from __future__ import annotations

__all__ = ["root_commands"]

from pathlib import Path

from heavenbase.workspace import WorkspaceSpec
from heavenbase.utils.typing import Any

from ..spec import ArgumentSpec, CommandSpec, OptionSpec


def _setup(context: Any, reset: bool = False, force: bool = False) -> None:
    spec = WorkspaceSpec.setup(reset=reset, force=force, context=context.runtime_context())
    context.out.success(f"heavenbase initialized with workspace {spec.id}")


def _dashboard(
    context: Any,
    host: str | None = None,
    port: int | None = None,
    open_window: bool | None = None,
    app_window: bool = False,
    browser_window: bool = False,
    allow_network: bool = False,
    install_app: bool = False,
    yes: bool = False,
) -> None:
    from heavenbase.api import DashboardAPI, resolve_bind
    from heavenbase.dashboard import default_macos_app_path, install_macos_app, serve
    from heavenbase.workspace import WorkspaceSpec

    if app_window and browser_window:
        raise ValueError("--app and --browser are mutually exclusive")
    if install_app:
        target = Path(default_macos_app_path())
        replace = target.exists()
        if replace and not yes and not context.out.confirm(f"{target} already exists. Replace it?", default=False):
            context.out.muted("application installation cancelled")
            return
        installed = install_macos_app(replace=replace)
        context.out.success(f"installed HeavenBase application at {installed}")
        return
    if not WorkspaceSpec.list(context=context.runtime_context()):  # first run: bootstrap so the data browser has a workspace
        spec = WorkspaceSpec.setup(context=context.runtime_context())
        context.out.success(f"first run: registered default workspace {spec.id!r}")
    host, port = resolve_bind(context.cm, host, port)
    # An explicit host choice is also an explicit request to open it. Preserve
    # ``--no-open`` when supplied; otherwise only the automatic mode consults
    # the persisted ``heavenbase.dashboard.open`` preference.
    if open_window is None and (app_window or browser_window):
        open_window = True
    serve(
        host=host,
        port=port,
        open_window=open_window,
        api=DashboardAPI(context=context.runtime_context()),
        host_mode="app" if app_window else "browser" if browser_window else "auto",
        allow_network=allow_network,
        on_started=lambda url: context.out.success(f"dashboard at {url}"),
    )


def _pj(context: Any, parts: list[str]) -> None:
    context.out.echo(context.cm.pj(*(parts or []), abs=True))


def root_commands() -> list[CommandSpec]:
    """Return root command specs.

    Args:
        None.

    Returns:
        list[CommandSpec]: Backend-neutral ``setup``, ``pj``, and ``dashboard``
            command specifications.
    """
    return [
        CommandSpec(
            "setup",
            _setup,
            help="Initialize global HeavenBase config and default workspace.",
            options=[
                OptionSpec.from_flags(
                    "reset",
                    "--reset",
                    "-r",
                    help="DESTRUCTIVE: reset package-owned local state before setup (implies --force).",
                    default=False,
                    is_flag=True,
                    type=bool,
                ),
                OptionSpec.from_flags(
                    "force",
                    "--force",
                    help="Re-index the current packaged module set without resetting local state.",
                    default=False,
                    is_flag=True,
                    type=bool,
                ),
            ],
        ),
        CommandSpec(
            "pj", _pj, help="Resolve a HeavenBase config/resource path.", args=[ArgumentSpec("parts", help="Path parts.", type=str, required=False, nargs=-1)]
        ),
        CommandSpec(
            "dashboard",
            _dashboard,
            help="Serve the local config and data dashboard.",
            options=[
                OptionSpec.from_flags("host", "--host", help="Bind host."),
                OptionSpec.from_flags("port", "--port", "-p", help="Bind port.", type=int),
                OptionSpec.from_flags(
                    "open_window",
                    "--open",
                    "--no-open",
                    help="Open the dashboard host on start (defaults to heavenbase.dashboard.open).",
                    default=None,
                    is_flag=True,
                    type=bool,
                    negatable=True,
                ),
                OptionSpec.from_flags(
                    "app_window", "--app", help="Use a Chromium app window instead of the default system WebView host.", default=False, is_flag=True, type=bool
                ),
                OptionSpec.from_flags("browser_window", "--browser", help="Open the workbench in the system browser.", default=False, is_flag=True, type=bool),
                OptionSpec.from_flags(
                    "install_app",
                    "--install-app",
                    help="Install ~/Applications/HeavenBase.app for Finder and Dock launch.",
                    default=False,
                    is_flag=True,
                    type=bool,
                ),
                OptionSpec.from_flags(
                    "yes",
                    "--yes",
                    "-y",
                    help="Replace an existing HeavenBase.app without confirmation.",
                    default=False,
                    is_flag=True,
                    type=bool,
                ),
                OptionSpec.from_flags(
                    "allow_network",
                    "--allow-network",
                    help="SECURITY: allow a non-loopback bind that exposes the unauthenticated config API.",
                    default=False,
                    is_flag=True,
                    type=bool,
                ),
            ],
        ),
    ]
