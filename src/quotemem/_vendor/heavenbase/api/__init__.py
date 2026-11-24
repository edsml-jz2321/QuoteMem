"""Application service and transport boundary for HeavenBase interfaces."""

from __future__ import annotations

__all__ = ["DashboardAPI", "build_app", "openapi_schema", "resolve_bind", "run_server"]

from typing import Any

from heavenbase._lazy import load_lazy_export

_LAZY_EXPORTS = {
    "DashboardAPI": (".service", "DashboardAPI"),
    "build_app": (".http", "build_app"),
    "openapi_schema": (".models", "openapi_schema"),
    "resolve_bind": (".http", "resolve_bind"),
    "run_server": (".http", "run_server"),
}


def __getattr__(name: str) -> Any:
    return load_lazy_export(name, _LAZY_EXPORTS, __name__, globals())
