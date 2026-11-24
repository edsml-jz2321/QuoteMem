"""Toolkit error types."""

from __future__ import annotations

__all__ = ["AsyncToolRequired", "ToolkitError", "ToolExportError", "raise_tool_error"]

from typing import NoReturn

from heavenbase.utils import safe_error
from heavenbase.utils.redact import redact_secrets


class ToolkitError(Exception):
    """Base error for Toolkit operations."""


class AsyncToolRequired(ToolkitError, TypeError):
    """Raised when a sync Tool boundary receives an awaitable result."""


class ToolExportError(ToolkitError):
    """Raised when a Tool cannot be exported."""


def raise_tool_error(exc: Exception) -> NoReturn:
    """Raise a redacted runtime error at a public tool boundary.

    Use this helper in external or bundled tool adapters after catching an
    implementation exception. It removes exception chaining and redacts
    credential-like values before the message reaches an MCP client, CLI, log,
    or model.

    Args:
        exc (Exception): Original implementation failure to sanitize.

    Returns:
        NoReturn: This function always raises a sanitized runtime error.

    Raises:
        RuntimeError: Error containing only the redacted public message.
    """
    raise RuntimeError(redact_secrets(safe_error(exc))) from None
