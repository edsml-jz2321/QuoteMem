"""HeavenBase command-line interface toolkit and app factory."""

from __future__ import annotations

from heavenbase._lazy import _attach_lazy_exports

__all__ = [
    "ArgumentSpec",
    "CLIContext",
    "CLIOutput",
    "CLIRegistry",
    "CommandGroupSpec",
    "CommandSpec",
    "OptionSpec",
    "PackageCLIContext",
    "create_cli",
    "main",
    "raise_cli_error",
]

_LAZY_EXPORTS = {
    "create_cli": (".app", "create_cli"),
    "main": (".app", "main"),
    "CLIContext": (".base", "CLIContext"),
    "PackageCLIContext": (".context", "PackageCLIContext"),
    "raise_cli_error": (".context", "raise_cli_error"),
    "CLIOutput": (".output", "CLIOutput"),
    "CLIRegistry": (".registry", "CLIRegistry"),
    "ArgumentSpec": (".spec", "ArgumentSpec"),
    "CommandGroupSpec": (".spec", "CommandGroupSpec"),
    "CommandSpec": (".spec", "CommandSpec"),
    "OptionSpec": (".spec", "OptionSpec"),
}

__getattr__, __dir__ = _attach_lazy_exports(__name__, globals(), _LAZY_EXPORTS, __all__)
