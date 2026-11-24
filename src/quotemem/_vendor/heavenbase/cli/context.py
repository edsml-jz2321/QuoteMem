"""CLI runtime context and error handling."""

from __future__ import annotations

__all__ = ["PackageCLIContext", "raise_cli_error"]

from typing import NoReturn

from heavenbase.utils import Any, CM_HVNB, Optional, dataclass
from heavenbase.utils.redact import redact_secrets


def _debug_errors(context: "PackageCLIContext") -> bool:
    cm = getattr(context, "cm", None)
    try:
        return bool(cm.get("heavenbase.cli.debug_errors")) if cm is not None else False
    except Exception:
        return False


def raise_cli_error(context: "PackageCLIContext", error: BaseException, code: int = 1, msg: Optional[str] = None) -> None:
    """Render one user-facing CLI error and exit.

    Args:
        context: Runtime CLI context that provides output and configuration.
        error: Original exception raised by CLI setup or command execution.
        code: Process exit code to raise when rendering succeeds.
        msg: Optional user-facing message override.

    Returns:
        None. This function exits with ``SystemExit`` or re-raises the original error.

    Raises:
        SystemExit: Always raised after rendering the CLI error.
        BaseException: Re-raises ``error`` when debug errors are enabled or no output writer exists.
    """
    if _debug_errors(context):
        raise error
    out = getattr(context, "out", None)
    if out is None:
        raise error
    rendered = redact_secrets(msg or str(error) or error.__class__.__name__)
    out.error(rendered)
    raise SystemExit(code) from error


@dataclass
class PackageCLIContext:
    """Package metadata and shared services passed to CLI command callbacks.

    Args:
        package: CLI executable name to expose in help output.
        version: Package version used by backend-specific version flags.
        cm: Configuration manager used by CLI setup and command callbacks.
        context: Optional HeavenBase machine Context. Workspace commands lazily
            derive one from ``cm`` when omitted.
        out: Output adapter used for user-facing CLI text.
    """

    package: str = "heavenbase"
    version: str = ""
    cm: Any = CM_HVNB
    context: Any = None
    out: Any = None

    def fail(
        self,
        error: BaseException,
        *,
        code: int = 1,
        msg: str | None = None,
    ) -> NoReturn:
        """Render one command failure through the HeavenBase CLI error policy.

        Args:
            error (BaseException): Original command or setup failure.
            code (int): Process exit code to use after rendering.
            msg (str | None): Optional user-facing message override.

        Returns:
            NoReturn: This method always exits or re-raises `error`.
        """
        raise_cli_error(self, error, code=code, msg=msg)

    def runtime_context(self) -> Any:
        """Return the Context that owns HeavenBase workspace operations.

        Args:
            None.

        Returns:
            Any: A memoized ``heavenbase.context.Context`` bound to ``cm``.

        Raises:
            TypeError: If an explicit ``context`` is not a Context.
        """
        from ..context import Context, context_or_default
        from ..context.runtime import _config_context

        if self.context is None:
            self.context = _config_context(self.cm)
        if self.context is None:
            self.context = Context.load(config=self.cm)
        return context_or_default(self.context)
