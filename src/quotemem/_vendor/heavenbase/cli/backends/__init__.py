"""CLI backend compilers."""

from __future__ import annotations

from heavenbase._lazy import _attach_lazy_exports

__all__ = ["build_argparse", "build_click", "build_typer", "run_argparse", "run_click", "run_typer"]

_LAZY_EXPORTS = {
    "build_argparse": (".argparse", "build_argparse"),
    "run_argparse": (".argparse", "run_argparse"),
    "build_click": (".click", "build_click"),
    "run_click": (".click", "run_click"),
    "build_typer": (".typer", "build_typer"),
    "run_typer": (".typer", "run_typer"),
}

__getattr__, __dir__ = _attach_lazy_exports(__name__, globals(), _LAZY_EXPORTS, __all__)
