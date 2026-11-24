"""Desktop-host selection and dashboard server lifecycle."""

from __future__ import annotations

__all__ = ["HostMode", "open_dashboard", "serve"]

import subprocess
import threading
import time
import webbrowser

from heavenbase.api import DashboardAPI, resolve_bind
from heavenbase.api.http import _run_server
from heavenbase.dashboard.assets import gui_file
from heavenbase.utils import cmd, is_macos
from heavenbase.utils.files import exists_file, exists_path
from heavenbase.utils.typing import Callable, Literal, Optional
from heavenbase.utils.values import raise_mismatch

HostMode = Literal["auto", "webview", "browser", "app"]
HOST_MODES = ("auto", "webview", "browser", "app")
CHROMIUM_APPS = (
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
)


def _gui_file(area: str, *parts: str) -> str:
    """Resolve one GUI asset from a source checkout or installed wheel."""
    return str(gui_file(area, *parts))


def _open_webview(url: str) -> bool:
    """Open the macOS system WebView host when it is available."""
    if not is_macos():
        return False
    from .macos import default_macos_app_path

    app = default_macos_app_path()
    if exists_path(app):
        try:
            return bool(
                cmd(
                    ["/usr/bin/open", "-na", app, "--args", url],
                    include="ok",
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            )
        except OSError:
            pass
    script = _gui_file("host", "macos.js")
    if not exists_file(script):
        return False
    icon = _gui_file("ui", "assets", "heavenbase-logo-concentric-wide-light-128.png")
    try:
        handle = cmd(
            ["/usr/bin/osascript", "-l", "JavaScript", script, url, icon],
            wait=False,
            include="handle",
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError:
        return False
    time.sleep(0.2)
    return handle.poll() is None


def _open_app_window(url: str) -> bool:
    """Open a Chromium `--app` window when one is installed."""
    candidates = [binary for binary in CHROMIUM_APPS if exists_path(binary)]
    candidates.extend(("google-chrome", "chromium", "chromium-browser", "microsoft-edge", "brave-browser"))
    for binary in candidates:
        try:
            cmd([binary, f"--app={url}", "--window-size=1280,800"], wait=False, include="handle")
            return True
        except OSError:
            continue
    return False


def open_dashboard(url: str, mode: HostMode = "auto") -> HostMode:
    """Open the dashboard with the requested desktop host policy.

    `auto` and `webview` prefer the macOS system WebView and fall back to the
    system browser when the host is unavailable. Chromium `--app` remains an
    explicit fallback mode.

    Args:
        url (str): Loopback dashboard URL to open.
        mode (HostMode): Host selection policy.
            Supported values:
            - `auto`: Prefer the system WebView, then use the system browser.
            - `webview`: Request the system WebView with browser fallback.
            - `browser`: Open the system browser directly.
            - `app`: Open Chromium `--app`, with browser fallback.

    Returns:
        HostMode: Host mode that accepted the launch.
    """
    if mode not in HOST_MODES:
        raise_mismatch(HOST_MODES, mode, name="dashboard host mode")
    if mode in {"auto", "webview"} and _open_webview(url):
        return "webview"
    if mode == "app" and _open_app_window(url):
        return "app"
    webbrowser.open(url)
    return "browser"


def serve(
    host: Optional[str] = None,
    port: Optional[int] = None,
    open_window: Optional[bool] = None,
    api: Optional[DashboardAPI] = None,
    host_mode: HostMode = "auto",
    allow_network: bool = False,
    *,
    on_started: Callable[[str], None] | None = None,
) -> None:
    """Launch the dashboard host and run its local HTTP service.

    Args:
        host (str | None): Bind host. Defaults to
            `heavenbase.dashboard.host` (`127.0.0.1`).
        port (int | None): Bind port. Zero requests an OS-assigned port.
            Defaults to `heavenbase.dashboard.port` (`42777`).
        open_window (bool | None): Whether to open a desktop window after the
            server starts. Defaults to `heavenbase.dashboard.open`.
        api (DashboardAPI | None): Application service backing the routes.
            Defaults to a fresh `DashboardAPI()` over the process default
            Context; the boot check, host, port and `open` read that
            service's ConfigManager.
        host_mode (HostMode): Desktop host policy documented by
            `open_dashboard`.
        allow_network (bool): Allow a non-loopback bind that exposes the
            unauthenticated API. Defaults to `False`.
        on_started (Callable[[str], None] | None): Callback receiving the
            dashboard URL once its listener is ready, including when the
            window is disabled. Defaults to `None`.

    Raises:
        ValueError: If ``host_mode`` is unknown, or a non-loopback bind is
            requested without ``allow_network=True``.
        SystemExit: With status 1 if the address cannot be bound (such as a
            busy port), or with status 3 if the server stops before serving
            for another reason, such as a failed application startup.

    Returns:
        None: This function does not return a value.
    """
    if host_mode not in HOST_MODES:
        raise_mismatch(HOST_MODES, host_mode, name="dashboard host mode")
    # The served service's config, not ``CM_HVNB``, decides the boot check,
    # bind and window: an installed default Context may own its ConfigManager.
    service = api if api is not None else DashboardAPI()
    cm = service.cm
    host, port = resolve_bind(cm, host, port)
    should_open = bool(cm.get("heavenbase.dashboard.open", True)) if open_window is None else open_window
    name = host.strip().lower()
    broad = name in {"0.0.0.0", "::", ""}
    open_host = "::1" if name == "::" else "127.0.0.1" if broad else name
    url_host = f"[{open_host}]" if ":" in open_host else open_host

    def after_startup(bound_port: int) -> None:
        url = f"http://{url_host}:{bound_port}"
        if on_started is not None:
            on_started(url)
        if should_open:
            threading.Thread(target=open_dashboard, args=(url, host_mode), daemon=True).start()

    _run_server(host, port, service, allow_network, on_started=after_startup if should_open or on_started is not None else None)
