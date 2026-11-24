"""Small CLI output helper used by all backend adapters."""

from __future__ import annotations

__all__ = ["CLIOutput"]

import sys

from heavenbase.utils import color_error, color_green, color_grey, color_warning, dumps_json, dumps_yaml
from heavenbase.utils.config.data import to_mutable_data
from heavenbase.utils.redact import redact_data, redact_secrets
from heavenbase.utils.typing import Any, Iterable, List, Optional


class CLIOutput:
    """Plain output helper with JSON/YAML/table helpers."""

    def confirm(self, prompt: str, *, default: bool = False) -> bool:
        """Ask one interactive yes/no question.

        Args:
            prompt (str): Decision text shown before the yes/no choices.
            default (bool): Value selected by an empty response.

        Returns:
            bool: Whether the user confirmed the operation.
        """
        import rich
        from rich.console import Console
        from rich.prompt import Confirm

        return bool(Confirm.ask(prompt, default=default, console=Console(stderr=True)))

    def echo(self, msg: Any = "", *, err: bool = False, end: str = "\n") -> None:
        """Print a message to stdout or stderr.

        Args:
            msg (Any): Message text to print.
            err (bool): When true, write to stderr instead of stdout.
            end (str): Line terminator appended after ``msg``.

        Returns:
            None: This function does not return a value.
        """
        print(redact_secrets(str(msg)), file=sys.stderr if err else sys.stdout, end=end)

    def success(self, msg: str) -> None:
        """Print a success message.

        Args:
            msg (str): Success text prefixed with ``OK``.

        Returns:
            None: This function does not return a value.
        """
        self.echo(color_green(f"OK {msg}"))

    def warning(self, msg: str) -> None:
        """Print a warning message.

        Args:
            msg (str): Warning text prefixed with ``WARN`` and written to stderr.

        Returns:
            None: This function does not return a value.
        """
        self.echo(color_warning(f"WARN {msg}"), err=True)

    def muted(self, msg: Any = "", *, err: bool = False, end: str = "\n") -> None:
        """Print a low-contrast diagnostic message.

        Args:
            msg (Any): Diagnostic text rendered in grey.
            err (bool): When true, write to stderr instead of stdout.
            end (str): Line terminator appended after ``msg``.

        Returns:
            None: This function does not return a value.
        """
        self.echo(color_grey(msg), err=err, end=end)

    def muted_yaml(self, data: Any, *, err: bool = True) -> None:
        """Print YAML as low-contrast diagnostic output.

        Args:
            data (Any): Structured value serialized with ``dumps_yaml``.
            err (bool): When true, write to stderr instead of stdout.

        Returns:
            None: This function does not return a value.
        """
        self.muted(dumps_yaml(redact_data(to_mutable_data(data))), err=err)

    def error(self, msg: str) -> None:
        """Print an error message.

        Args:
            msg (str): Error text prefixed with ``ERR`` and written to stderr.

        Returns:
            None: This function does not return a value.
        """
        self.echo(color_error(f"ERR {redact_secrets(str(msg))}"), err=True)

    def json(self, data: Any) -> None:
        """Print JSON output.

        Args:
            data (Any): Structured value serialized with ``dumps_json``.

        Returns:
            None: This function does not return a value.
        """
        self.echo(dumps_json(redact_data(to_mutable_data(data))))

    def yaml(self, data: Any) -> None:
        """Print YAML output.

        Args:
            data (Any): Structured value serialized with ``dumps_yaml``.

        Returns:
            None: This function does not return a value.
        """
        self.echo(dumps_yaml(redact_data(to_mutable_data(data))))

    def table(self, columns: List[str], rows: Iterable[Iterable[Any]], title: Optional[str] = None) -> None:
        """Print a compact plain-text table.

        Args:
            columns (List[str]): Column headers rendered on the first row.
            rows (Iterable[Iterable[Any]]): Table body rows aligned to ``columns``.
            title (str | None): Optional heading printed above the table.

        Returns:
            None: This function does not return a value.
        """
        rows_list = []
        for row in rows:
            values = list(row)
            record = redact_data(dict(zip(columns, values)))
            redacted = [record[column] for column in columns[: len(values)]]
            redacted.extend(redact_data(value) for value in values[len(columns) :])
            rows_list.append([str(value) for value in redacted])
        widths = [len(column) for column in columns]
        for row in rows_list:
            widths = [max(width, len(value)) for width, value in zip(widths, row)]
        if title:
            self.echo(title)
        self.echo("  ".join(column.ljust(width) for column, width in zip(columns, widths)))
        self.echo("  ".join("-" * width for width in widths))
        for row in rows_list:
            self.echo("  ".join(value.ljust(width) for value, width in zip(row, widths)))
