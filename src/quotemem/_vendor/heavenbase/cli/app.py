"""HeavenBase `hb` CLI app factory."""

from __future__ import annotations

__all__ = ["build_registry", "create_cli", "main"]

import sys
from typing import ClassVar

from heavenbase.context import Context, default_context
from heavenbase.context.authority import omitted_config
from heavenbase.utils import Any, Callable, Dict, Optional, Tuple, dataclass, dget, field
from heavenbase.utils.config import CM_HVNB, ConfigManager
from heavenbase.version import __version__

from .backends import build_argparse, build_click, build_typer, run_argparse, run_click, run_typer
from .base import CLIContext
from .context import PackageCLIContext, raise_cli_error
from .groups import config_group, llm_group, mcp_group, prompt_group, root_commands, ws_group
from .output import CLIOutput
from .registry import CLIRegistry

BackendFactory = Callable[[CLIRegistry, CLIContext], Any]
BackendRunner = Callable[[Any, Optional[list[str]]], Any]
BACKENDS: Dict[str, Tuple[BackendFactory, BackendRunner]] = {
    "typer": (build_typer, run_typer),
    "click": (build_click, run_click),
    "argparse": (build_argparse, run_argparse),
}


def build_registry() -> CLIRegistry:
    """Build the default HeavenBase CLI registry.

    Args:
        None.

    Returns:
        CLIRegistry: Registry populated with root commands and built-in groups.
    """
    registry = CLIRegistry()
    for command in root_commands():
        registry.add_root(command)
    registry.add_group(ws_group())
    registry.add_group(config_group())
    registry.add_group(llm_group())
    registry.add_group(mcp_group())
    registry.add_group(prompt_group())
    return registry


_OMITTED_CONFIG = object()


@dataclass(init=False)
class _DefaultCLIContext(PackageCLIContext):
    """Resolve the process default when commands need its services."""

    # Dataclass copies carry the override without reading the resolved property.
    cm: ClassVar[Any]
    _cm: ConfigManager | None = field(default=None, repr=False, kw_only=True)

    def __init__(
        self,
        package: str = "hb",
        version: str = __version__,
        cm: Any = _OMITTED_CONFIG,
        context: Context | None = None,
        out: CLIOutput | None = None,
        *,
        _cm: ConfigManager | None = None,
    ) -> None:
        super().__init__(package=package, version=version, cm=_cm if cm is _OMITTED_CONFIG else cm, context=context, out=out)

    @property
    def cm(self) -> ConfigManager:
        if self._cm is not None:
            return self._cm
        return omitted_config() if self.context is None else self.context.config

    @cm.setter
    def cm(self, config: ConfigManager | None) -> None:
        self._cm = config

    def runtime_context(self) -> Context:
        if self.context is None and self._cm is None:
            return default_context()
        return super().runtime_context()


def _context(context: Optional[PackageCLIContext] = None) -> PackageCLIContext:
    if context is None:
        return _DefaultCLIContext(out=CLIOutput())
    if context.out is None:
        context.out = CLIOutput()
    return context


def _backend(context: PackageCLIContext, backend: Optional[str] = None) -> str:
    key = "heavenbase.cli.backend"
    selected = backend
    if selected is None:
        cm = context.cm
        package_default = dget(CM_HVNB.default, key) if isinstance(context, _DefaultCLIContext) else None
        selected = cm.get(key, default=dget(getattr(cm, "default", {}), key, default=package_default))
    if not selected:
        raise ValueError("Missing CLI backend config: heavenbase.cli.backend")
    return str(selected).lower()


def create_cli(
    backend: Optional[str] = None,
    context: Optional[PackageCLIContext] = None,
    registry: Optional[CLIRegistry] = None,
) -> Any:
    """Create a CLI app for the selected backend.

    Args:
        backend (str | None): CLI backend name. When omitted, reads
            ``heavenbase.cli.backend`` from the runtime config manager. This can
            select the process default Context and use its saved configuration.
            The default CLI falls back to the packaged backend when the setting
            is absent. Supplying a backend skips the configuration read.
        context (PackageCLIContext | None): Runtime context passed to command
            callbacks. When omitted, callbacks resolve the process default
            Context and its configuration when needed.
        registry (CLIRegistry | None): Optional caller-owned command registry.
            When omitted, builds the built-in ``hb`` registry.

    Returns:
        Any: Backend-specific application object compiled from ``registry``
            (Typer, Click, or argparse depending on ``backend``).

    Raises:
        ValueError: If ``backend`` is unsupported or missing from config.
    """
    ctx = _context(context)
    selected = _backend(ctx, backend)
    selected_registry = build_registry() if registry is None else registry
    if selected not in BACKENDS:
        raise ValueError(f"Unsupported CLI backend {selected!r}; expected one of {sorted(BACKENDS)}")
    return BACKENDS[selected][0](selected_registry, ctx)


def _strip_path_prefix(argv: list[str]) -> list[str]:
    """Remove ``path:`` prefix from args to work around Git Bash bugs
    with tokens starting with ``/``, ``%``, ``&``, etc."""
    prefix = "path:"
    return [a.removeprefix(prefix) if a.startswith(prefix) else a for a in argv]


def main(argv: Optional[list[str]] = None) -> Any:
    """Run the configured CLI entry point.

    Args:
        argv (list[str] | None): Arguments to parse instead of ``sys.argv[1:]``.

    Returns:
        Any: Return value from the selected backend runner, if any.

    Raises:
        SystemExit: Raised for normal CLI exits, including rendered errors.
    """
    argv = _strip_path_prefix(argv if argv is not None else sys.argv[1:])
    ctx = _context()
    try:
        selected = _backend(ctx)
        app = create_cli(selected, context=ctx)
        return BACKENDS[selected][1](app, argv)
    except SystemExit:
        raise
    except KeyboardInterrupt as error:
        raise_cli_error(ctx, error, code=130, msg="interrupted")
    except Exception as error:
        raise_cli_error(ctx, error)


if __name__ == "__main__":
    main()
