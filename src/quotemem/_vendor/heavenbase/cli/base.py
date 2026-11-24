"""Parser-neutral CLI context contract."""

from __future__ import annotations

__all__ = ["CLIContext"]

from typing import NoReturn, Protocol, runtime_checkable


@runtime_checkable
class CLIContext(Protocol):
    """Describe the minimum runtime contract consumed by CLI specifications.

    Concrete applications may add configuration, output, service, or lifecycle
    state. Parser compilers and command specifications depend only on the
    executable name, version text, and the context-owned error boundary.
    """

    package: str
    version: str

    def fail(
        self,
        error: BaseException,
        *,
        code: int = 1,
        msg: str | None = None,
    ) -> NoReturn:
        """Render or propagate one command failure and terminate invocation.

        Args:
            error (BaseException): Original command or setup failure.
            code (int): Process exit code to use after rendering.
            msg (str | None): Optional user-facing message override.

        Returns:
            NoReturn: This method always raises.
        """
        ...
