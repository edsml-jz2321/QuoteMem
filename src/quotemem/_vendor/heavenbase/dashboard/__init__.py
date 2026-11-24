"""Thin local launcher for the HeavenBase desktop dashboard."""

from __future__ import annotations

__all__ = ["HostMode", "default_macos_app_path", "install_macos_app", "open_dashboard", "serve"]

from typing import Any

from heavenbase._lazy import load_lazy_export

_LAZY_EXPORTS = {
    "HostMode": (".launch", "HostMode"),
    "default_macos_app_path": (".macos", "default_macos_app_path"),
    "install_macos_app": (".macos", "install_macos_app"),
    "open_dashboard": (".launch", "open_dashboard"),
    "serve": (".launch", "serve"),
}


def __getattr__(name: str) -> Any:
    return load_lazy_export(name, _LAZY_EXPORTS, __name__, globals())
