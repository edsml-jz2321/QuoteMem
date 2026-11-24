"""Install and serve the native macOS HeavenBase dashboard application."""

from __future__ import annotations

__all__ = ["default_macos_app_path", "install_macos_app", "main", "resolve_app_bind"]

import argparse
import json
import os
from pathlib import Path
import plistlib
import shutil
import subprocess
import sys
import tempfile

from heavenbase.api import resolve_bind
from heavenbase.api.http import _run_server
from heavenbase.dashboard.assets import gui_file
from heavenbase.utils import is_macos

_APP_NAME = "HeavenBase.app"
_BUNDLE_ID = "top.ahvn.heavenbase.dashboard"
_HOST_MODULE = "heavenbase.dashboard.macos"


def _gui_file(area: str, *parts: str) -> Path:
    """Resolve one GUI asset from a source checkout or installed wheel."""
    return gui_file(area, *parts)


def default_macos_app_path() -> str:
    """Return the user-local installation path for the dashboard application.

    Returns:
        str: Absolute ``~/Applications/HeavenBase.app`` path.
    """
    return str((Path.home() / "Applications" / _APP_NAME).resolve())


def resolve_app_bind() -> tuple[str, int]:
    """Resolve the loopback-only bind used by a Finder-launched application.

    The application intentionally ignores a configured non-loopback host while
    preserving the configured dashboard port. Network exposure remains an
    explicit CLI-only choice through ``hb dashboard --allow-network``.

    Returns:
        tuple[str, int]: Loopback host and configured dashboard port.
    """
    # Selecting the default Context binds CM_HVNB to the persistent system
    # Registry, so Finder-host writes survive the application exit.
    from heavenbase.context import default_context

    _, port = resolve_bind(default_context().config, host="127.0.0.1")
    return "127.0.0.1", port


def _bundle_plist(python_executable: str) -> dict[str, object]:
    """Return the generated application property list."""
    from heavenbase.version import __version__

    version_parts = __version__.split(".")
    short_version = ".".join(version_parts[:3])
    bundle_version = ".".join(version_parts[-3:])
    return {
        "CFBundleAllowMixedLocalizations": True,
        "CFBundleDevelopmentRegion": "en",
        "CFBundleDisplayName": "HeavenBase",
        "CFBundleExecutable": "applet",
        "CFBundleIconFile": "HeavenBase",
        "CFBundleIconName": "HeavenBase",
        "CFBundleIdentifier": _BUNDLE_ID,
        "CFBundleInfoDictionaryVersion": "6.0",
        "CFBundleName": "HeavenBase",
        "CFBundlePackageType": "APPL",
        "CFBundleSignature": "aplt",
        "CFBundleShortVersionString": short_version,
        "CFBundleSupportedPlatforms": ["MacOSX"],
        "CFBundleVersion": bundle_version,
        "HBPythonExecutable": python_executable,
        "HBPythonModule": _HOST_MODULE,
        "LSApplicationCategoryType": "public.app-category.developer-tools",
        "LSMinimumSystemVersion": "12.0",
        "LSRequiresCarbon": True,
        "NSHighResolutionCapable": True,
        "OSAAppletShowStartupScreen": False,
        "OSAAppletStayOpen": True,
    }


def _run_packager(command: list[str], description: str) -> None:
    """Run one required system application-packaging command."""
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True)
    except OSError as exc:
        raise RuntimeError(f"{description} is unavailable: {exc}") from exc
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip() or f"exit status {result.returncode}"
        raise RuntimeError(f"{description} failed: {detail}")


def _compile_macos_applet(source: Path, destination: Path) -> None:
    """Compile the JavaScript host as a LaunchServices applet."""
    _run_packager(
        ["/usr/bin/osacompile", "-s", "-l", "JavaScript", "-o", str(destination), str(source)],
        "JavaScript application compiler",
    )


def install_macos_app(
    destination: str | os.PathLike[str] | None = None,
    *,
    replace: bool = False,
    python_executable: str | os.PathLike[str] | None = None,
) -> str:
    """Install a Finder-launchable HeavenBase application bundle.

    The generated bundle binds to the Python environment running the installer.
    Re-run the installer after moving or replacing that environment.

    Args:
        destination (str | os.PathLike[str] | None): Exact ``.app`` path.
            Defaults to ``~/Applications/HeavenBase.app``.
        replace (bool): Replace an existing bundle at the exact destination.
        python_executable (str | os.PathLike[str] | None): Python runtime stored
            in the bundle. Defaults to ``sys.executable``.

    Returns:
        str: Absolute installed bundle path.

    Raises:
        RuntimeError: The platform or packaged host assets are unavailable.
        FileExistsError: The destination exists and ``replace`` is false.
        ValueError: The destination is not a ``.app`` path.
    """
    if not is_macos():
        raise RuntimeError("the HeavenBase .app installer is available only on macOS")
    target = Path(destination or default_macos_app_path()).expanduser().resolve()
    if target.suffix.lower() != ".app":
        raise ValueError(f"macOS application destination must end in .app: {target}")
    if target.exists() and not replace:
        raise FileExistsError(f"{target} already exists; pass replace=True to replace it")

    launcher_source = _gui_file("host", "macos_app.js")
    icon_source = _gui_file("host", "HeavenBase.icns")
    missing = [str(path) for path in (launcher_source, icon_source) if not path.is_file()]
    if missing:
        raise RuntimeError(f"macOS application assets are missing: {missing!r}; reinstall HeavenBase")

    # Preserve virtual-environment launcher symlinks. Resolving ``.venv/bin/python``
    # to a uv-managed base interpreter discards the environment that contains
    # HeavenBase and makes Finder launch fail outside the source checkout.
    runtime = Path(python_executable or sys.executable).expanduser().absolute()
    if not runtime.is_file():
        raise RuntimeError(f"Python runtime does not exist: {runtime}")

    target.parent.mkdir(parents=True, exist_ok=True)
    stage_root = Path(tempfile.mkdtemp(prefix=".heavenbase-app-", dir=target.parent))
    staged = stage_root / _APP_NAME
    previous = stage_root / "previous.app"
    try:
        _compile_macos_applet(launcher_source, staged)
        resources = staged / "Contents" / "Resources"
        shutil.copyfile(icon_source, resources / "HeavenBase.icns")
        with (staged / "Contents" / "Info.plist").open("wb") as stream:
            plistlib.dump(_bundle_plist(str(runtime)), stream, sort_keys=True)
        _run_packager(
            ["/usr/bin/codesign", "--force", "--deep", "--sign", "-", str(staged)],
            "application code signing",
        )

        if target.exists():
            os.replace(target, previous)
        try:
            os.replace(staged, target)
        except BaseException:
            if previous.exists() and not target.exists():
                os.replace(previous, target)
            raise
    finally:
        shutil.rmtree(stage_root, ignore_errors=True)
    return str(target)


def _serve(host: str, port: int, ready_file: Path | None = None) -> None:
    """Run the child HTTP service owned by the application process."""
    from heavenbase.workspace import WorkspaceSpec

    if not WorkspaceSpec.list():
        WorkspaceSpec.setup()

    def report_port(bound_port: int) -> None:
        if ready_file is not None:
            with ready_file.open("x", encoding="ascii") as ready:
                ready.write(f"{bound_port}\n")

    _run_server(host, port, None, False, on_started=report_port if ready_file is not None else None)


def main(argv: list[str] | None = None) -> int:
    """Run the private macOS application host protocol.

    Args:
        argv (list[str] | None): Command arguments after the module name.

    Returns:
        int: Process exit status.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--print-bind", action="store_true")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--ready-file", type=Path)
    args = parser.parse_args(argv)
    host, port = resolve_app_bind()
    if args.print_bind:
        print(json.dumps({"host": host, "port": port}, sort_keys=True), flush=True)
        return 0
    if args.serve:
        _serve(args.host or host, port if args.port is None else args.port, args.ready_file)
        return 0
    parser.error("expected --print-bind or --serve")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
