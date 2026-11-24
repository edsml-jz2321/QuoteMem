"""Config-routed logging helpers."""

from __future__ import annotations

__all__ = [
    "ColoredFormatter",
    "ErrorRecord",
    "HbLogHandler",
    "SuppressOutput",
    "capture_error",
    "color_black",
    "color_blue",
    "color_cyan",
    "color_debug",
    "color_enabled",
    "color_error",
    "color_green",
    "color_grey",
    "color_info",
    "color_info1",
    "color_info2",
    "color_info3",
    "color_magenta",
    "color_red",
    "color_success",
    "color_warning",
    "color_white",
    "color_yellow",
    "configure_logs",
    "cprint",
    "cstr",
    "error_str",
    "get_logger",
    "no_color",
    "print_debug",
    "print_error",
    "print_exception",
    "print_info",
    "print_success",
    "print_warning",
    "redirect_logs",
    "restore_logs",
    "safe_error",
    "set_log_level",
    "strip_color",
    "suppress_logs",
]

import logging
import os
import re
import sys
import threading
import traceback
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import unquote_to_bytes, urlsplit

from heavenbase._lazy import resolve_late

from .typing import Any, Callable, Dict, List, Optional, Union, dataclass

if TYPE_CHECKING:
    from .config import ConfigManager

SUCCESS_LEVEL = 25
logging.addLevelName(SUCCESS_LEVEL, "SUCCESS")

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_COLORS = {"black": "30", "red": "31", "green": "32", "yellow": "33", "blue": "34", "magenta": "35", "cyan": "36", "white": "37", "grey": "90"}


def color_enabled(console: Optional[bool] = None) -> bool:
    """Return whether terminal output should include ANSI colors.

    Args:
        console (Optional[bool]): Explicit terminal decision. ``None`` checks
            stderr; ``NO_COLOR`` always disables colors.

    Returns:
        bool: Whether ANSI colors should be emitted.
    """
    if os.environ.get("NO_COLOR"):
        return False
    return bool(sys.stderr.isatty() if console is None else console)


def cstr(obj: Any, color: Optional[str] = None, bold: bool = False, console: Optional[bool] = None) -> str:
    """Return text with optional ANSI color for terminal presentation.

    Args:
        obj (Any): Value to convert to text.
        color (Optional[str]): Named color or raw ANSI color code.
        bold (bool): Whether to add bold styling.
        console (Optional[bool]): Explicit terminal decision. ``None`` checks stderr.

    Returns:
        str: Plain or ANSI-colored text.
    """
    text = str(obj)
    if not color or not color_enabled(console):
        return text
    code = _COLORS.get(color, color)
    attrs = ["1"] if bold else []
    return f"\x1b[{';'.join([*attrs, code])}m{text}\x1b[0m"


def strip_color(text: str) -> str:
    """Remove ANSI color escapes from text.

    Args:
        text (str): Text that may contain ANSI color escapes.

    Returns:
        str: Plain text.
    """
    return _ANSI_RE.sub("", text)


def no_color(obj: Any, console: Optional[bool] = None) -> str:
    """Return an object's plain string form.

    Args:
        obj (Any): Value to convert to text.
        console (Optional[bool]): Ignored terminal decision for formatter parity.

    Returns:
        str: Plain text.
    """
    return str(obj)


def _colored(obj: Any, color: str, console: Optional[bool] = None, *, bold: bool = True) -> str:
    return cstr(obj, color, bold=bold, console=console)


def color_black(obj: Any, console: Optional[bool] = None) -> str:
    """Return black terminal text.

    Args:
        obj (Any): Value to format.
        console (Optional[bool]): Explicit terminal decision.

    Returns:
        str: Plain or colored text.
    """
    return _colored(obj, "black", console, bold=False)


def color_red(obj: Any, console: Optional[bool] = None) -> str:
    """Return red terminal text.

    Args:
        obj (Any): Value to format.
        console (Optional[bool]): Explicit terminal decision.

    Returns:
        str: Plain or colored text.
    """
    return _colored(obj, "red", console)


def color_green(obj: Any, console: Optional[bool] = None) -> str:
    """Return green terminal text.

    Args:
        obj (Any): Value to format.
        console (Optional[bool]): Explicit terminal decision.

    Returns:
        str: Plain or colored text.
    """
    return _colored(obj, "green", console)


def color_yellow(obj: Any, console: Optional[bool] = None) -> str:
    """Return yellow terminal text.

    Args:
        obj (Any): Value to format.
        console (Optional[bool]): Explicit terminal decision.

    Returns:
        str: Plain or colored text.
    """
    return _colored(obj, "yellow", console)


def color_blue(obj: Any, console: Optional[bool] = None) -> str:
    """Return blue terminal text.

    Args:
        obj (Any): Value to format.
        console (Optional[bool]): Explicit terminal decision.

    Returns:
        str: Plain or colored text.
    """
    return _colored(obj, "blue", console)


def color_magenta(obj: Any, console: Optional[bool] = None) -> str:
    """Return magenta terminal text.

    Args:
        obj (Any): Value to format.
        console (Optional[bool]): Explicit terminal decision.

    Returns:
        str: Plain or colored text.
    """
    return _colored(obj, "magenta", console)


def color_cyan(obj: Any, console: Optional[bool] = None) -> str:
    """Return cyan terminal text.

    Args:
        obj (Any): Value to format.
        console (Optional[bool]): Explicit terminal decision.

    Returns:
        str: Plain or colored text.
    """
    return _colored(obj, "cyan", console)


def color_white(obj: Any, console: Optional[bool] = None) -> str:
    """Return white terminal text.

    Args:
        obj (Any): Value to format.
        console (Optional[bool]): Explicit terminal decision.

    Returns:
        str: Plain or colored text.
    """
    return _colored(obj, "white", console)


def color_grey(obj: Any, console: Optional[bool] = None) -> str:
    """Return grey terminal text.

    Args:
        obj (Any): Value to format.
        console (Optional[bool]): Explicit terminal decision.

    Returns:
        str: Plain or colored text.
    """
    return _colored(obj, "grey", console)


color_debug = color_grey
color_info = color_blue
color_info1 = color_blue
color_info2 = color_magenta
color_info3 = color_cyan
color_warning = color_yellow
color_error = color_red
color_success = color_green

_COLOR_FUNCTIONS = {
    "blue": color_blue,
    "green": color_green,
    "grey": color_grey,
    "red": color_red,
    "yellow": color_yellow,
}
_LEVEL_COLOR_NAMES = {
    logging.DEBUG: "grey",
    logging.INFO: "blue",
    SUCCESS_LEVEL: "green",
    logging.WARNING: "yellow",
    logging.ERROR: "red",
    logging.CRITICAL: "red",
}


def cprint(obj: Any, color: Optional[str] = None, bold: bool = False, file: Any = None, **kwargs: Any) -> None:
    """Print optional colored text to stderr or a selected stream.

    Args:
        obj (Any): Value to print.
        color (Optional[str]): Named color or raw ANSI color code.
        bold (bool): Whether to add bold styling.
        file (Any): Optional stream target. Defaults to stderr.
        **kwargs (Any): Additional arguments accepted by ``print``.

    Returns:
        None: This function does not return a value.
    """
    target = file or sys.stderr
    console = target.isatty() if hasattr(target, "isatty") else False
    print(cstr(obj, color=color, bold=bold, console=console), file=target, **kwargs)


def _print_level(level: int, *args: Any, **kwargs: Any) -> None:
    cprint(" ".join(str(arg) for arg in args), color=_LEVEL_COLOR_NAMES[level], bold=True, **kwargs)


def print_debug(*args: Any, **kwargs: Any) -> None:
    """Print a logger-consistent debug message.

    Args:
        *args (Any): Values joined with spaces.
        **kwargs (Any): Additional arguments accepted by `cprint`.

    Returns:
        None: This function does not return a value.
    """
    _print_level(logging.DEBUG, *args, **kwargs)


def print_info(*args: Any, **kwargs: Any) -> None:
    """Print a logger-consistent info message.

    Args:
        *args (Any): Values joined with spaces.
        **kwargs (Any): Additional arguments accepted by `cprint`.

    Returns:
        None: This function does not return a value.
    """
    _print_level(logging.INFO, *args, **kwargs)


def print_warning(*args: Any, **kwargs: Any) -> None:
    """Print a logger-consistent warning message.

    Args:
        *args (Any): Values joined with spaces.
        **kwargs (Any): Additional arguments accepted by `cprint`.

    Returns:
        None: This function does not return a value.
    """
    _print_level(logging.WARNING, *args, **kwargs)


def print_error(*args: Any, **kwargs: Any) -> None:
    """Print a logger-consistent error message.

    Args:
        *args (Any): Values joined with spaces.
        **kwargs (Any): Additional arguments accepted by `cprint`.

    Returns:
        None: This function does not return a value.
    """
    _print_level(logging.ERROR, *args, **kwargs)


def print_success(*args: Any, **kwargs: Any) -> None:
    """Print a logger-consistent success message.

    Args:
        *args (Any): Values joined with spaces.
        **kwargs (Any): Additional arguments accepted by `cprint`.

    Returns:
        None: This function does not return a value.
    """
    _print_level(SUCCESS_LEVEL, *args, **kwargs)


_LOGGERS: Dict[str, logging.Logger] = {}
_LOGGER_LOCK = threading.RLock()
_GLOBAL_STREAM: Any = sys.stderr
# Settings that ``HbLogHandler.handle`` read before it took the handler lock.
_HANDLE_SETTINGS = threading.local()
# Global log settings overridden in this process.
_PROCESS_SETTINGS: Dict[str, Any] = {}
_PROCESS_SETTINGS_LOCK = threading.Lock()


@dataclass(frozen=True)
class ErrorRecord:
    """Captured exception with its original object and formatted text."""

    err: BaseException
    type_name: str
    message: str
    text: str

    def to_str(self) -> str:
        """Return the formatted exception text.

        Args:
            None.

        Returns:
            str: Formatted exception text.
        """
        return self.text

    def reraise(self) -> None:
        """Raise the preserved exception object.

        Args:
            None.

        Returns:
            None: This method does not return because it raises the exception.

        Raises:
            BaseException: The original captured exception.
        """
        raise self.err


def capture_error(err: Optional[BaseException], tb: bool = True, chain: bool = True) -> Optional[ErrorRecord]:
    """Capture an exception without losing the original object.

    Args:
        err (Optional[BaseException]): Exception to capture, or ``None``.
        tb (bool): Whether to include traceback frames.
        chain (bool): Whether to include chained exceptions.

    Returns:
        Optional[ErrorRecord]: Captured error, or ``None`` when no error is given.
    """
    if err is None:
        return None
    text = _format_error(err, tb=tb, chain=chain)
    return ErrorRecord(err=err, type_name=type(err).__name__, message=str(err), text=text)


def error_str(err: Optional[object] = None, tb: bool = True, chain: bool = True) -> Optional[str]:
    """Return formatted exception or object text.

    Args:
        err (Optional[object]): Exception, captured error, plain object, or ``None``.
        tb (bool): Whether exception text includes traceback frames.
        chain (bool): Whether exception text includes chained exceptions.

    Returns:
        Optional[str]: Formatted text, or ``None`` when no value is given.
    """
    if err is None:
        return None
    if isinstance(err, ErrorRecord):
        return err.text
    if isinstance(err, BaseException):
        return _format_error(err, tb=tb, chain=chain)
    return str(err).strip()


def print_exception(err: Optional[BaseException] = None, file: Optional[Any] = None, tb: bool = True, chain: bool = True) -> Optional[ErrorRecord]:
    """Print an exception with logger-consistent error presentation.

    Args:
        err (Optional[BaseException]): Exception to print, or ``None``.
        file (Optional[Any]): Optional stream target. Defaults to stderr.
        tb (bool): Whether to include traceback frames.
        chain (bool): Whether to include chained exceptions.

    Returns:
        Optional[ErrorRecord]: Captured error, or ``None`` when no error is given.
    """
    record = capture_error(err, tb=tb, chain=chain)
    if record is not None:
        cprint(record.text, color="red", file=file)
    return record


def _format_error(err: BaseException, tb: bool, chain: bool) -> str:
    if not tb:
        return f"{type(err).__name__}: {err}".strip()
    return "".join(traceback.TracebackException.from_exception(err, capture_locals=False).format(chain=chain)).strip()


class ColoredFormatter(logging.Formatter):
    """Logging formatter that applies terminal colors by log level."""

    colors: Dict[int, Callable[[Any, bool], str]] = {
        logging.NOTSET: no_color,
        **{level: _COLOR_FUNCTIONS[color] for level, color in _LEVEL_COLOR_NAMES.items()},
    }

    def __init__(
        self,
        fmt: Optional[str] = None,
        datefmt: Optional[str] = None,
        style: str = "%",
        *,
        validate: bool = True,
        colors: Optional[Dict[int, Callable[[Any, bool], str]]] = None,
        console: Optional[bool] = None,
    ) -> None:
        """
        Initialize the utility object with the provided runtime settings.

        Use this helper for HeavenBase loggers whose routing can be changed through runtime config.

        Args:
            fmt (Optional[str]): Log or display format string. Defaults to the configured HeavenBase format.
            datefmt (Optional[str]): Optional logging date format. Defaults to ``None``.
            style (str): Logging format style accepted by ``logging.Formatter``. Defaults to ``"%"``.
            validate (bool): Whether ``logging.Formatter`` validates the format string. Defaults to ``True``.
            colors (Optional[Dict[int, Callable[[Any, bool], str]]]): Optional log-level-to-color override mapping. Defaults to ``None``.
            console (Optional[bool]): Whether the destination supports ANSI colors.

        Returns:
            None: This function returns ``None`` after initializing the object.
        """
        super().__init__(fmt, datefmt, style=style, validate=validate)
        self.colors = {**self.colors, **(colors or {})}
        self.console = console

    def format(self, record: logging.LogRecord) -> str:
        """
        Use ``format`` in the HeavenBase utility surface.

        Use this helper for HeavenBase loggers whose routing can be changed through runtime config.

        Args:
            record (logging.LogRecord): Log record being formatted or emitted.

        Returns:
            str: Formatted log message, colored when terminal coloring is enabled.
        """
        message = super().format(record)
        color = getattr(record, "color", self.colors.get(record.levelno, no_color))
        return color(message, console=self.console)


class HbLogHandler(logging.Handler):
    """Handler that routes each record through current HeavenBase log config."""

    terminator = "\n"

    def __init__(
        self,
        stream: Optional[Any] = None,
        fmt: Optional[str] = None,
        datefmt: Optional[str] = None,
        colors: Optional[Dict[int, Callable[[Any, bool], str]]] = None,
    ) -> None:
        """
        Initialize the utility object with the provided runtime settings.

        Use this helper for HeavenBase loggers whose routing can be changed through runtime config.

        Args:
            stream (Optional[Any]): Optional stream-like object used for logging or printing. Defaults to ``None``.
            fmt (Optional[str]): Log or display format string. Defaults to the configured HeavenBase format.
            datefmt (Optional[str]): Optional logging date format. Defaults to ``None``.
            colors (Optional[Dict[int, Callable[[Any, bool], str]]]): Optional log-level-to-color override mapping. Defaults to ``None``.

        Returns:
            None: This function returns ``None`` after initializing the object.
        """
        super().__init__(level=logging.DEBUG)
        self.stream = stream
        self.fmt = fmt
        self.datefmt = datefmt
        self.colors = colors
        self.suppressed = False
        self.level_override: Optional[int] = None
        self.file_target: Optional[str] = None
        self.file_path: Optional[str] = None
        self.file_stream: Optional[Any] = None

    def emit(self, record: logging.LogRecord) -> None:
        """
        Use ``emit`` in the HeavenBase utility surface.

        Use this helper for HeavenBase loggers whose routing can be changed through runtime config.

        Args:
            record (logging.LogRecord): Log record being formatted or emitted.

        Returns:
            None: This function returns ``None`` after completing its side effect.
        """
        settings = getattr(_HANDLE_SETTINGS, "value", None)
        self._emit(record, _log_settings() if settings is None else settings)

    def handle(self, record: logging.LogRecord) -> Union[bool, logging.LogRecord]:
        """
        Filter ``record`` and emit it under this handler's lock.

        The saved log settings are read before the lock is taken. The first
        read can select the default Context, which opens its database and
        creates loggers; :func:`get_logger` takes this lock while it holds the
        logger table lock, so reading under it could deadlock two threads.
        The record then goes to :meth:`emit`, as ``logging.Handler.handle``
        sends it, so a subclass that overrides ``emit`` receives it; the base
        ``emit`` uses the settings read here.

        Args:
            record (logging.LogRecord): Log record being formatted or emitted.

        Returns:
            Union[bool, logging.LogRecord]: The filter result, as ``logging.Handler.handle`` returns it.
        """
        rv = self.filter(record)
        if isinstance(rv, logging.LogRecord):
            record = rv
        if rv:
            previous = getattr(_HANDLE_SETTINGS, "value", None)
            _HANDLE_SETTINGS.value = _log_settings()
            try:
                self.acquire()
                try:
                    self.emit(record)
                finally:
                    self.release()
            finally:
                _HANDLE_SETTINGS.value = previous
        return rv

    def _emit(self, record: logging.LogRecord, settings: Dict[str, Any]) -> None:
        try:
            if self.suppressed or _bool(settings.get("suppress", False)) or settings.get("sink") == "suppress":
                return
            level = self.level_override if self.level_override is not None else _level_value(settings.get("level"), logging.INFO)
            if record.levelno < level:
                return
            target = self.file_target or (settings.get("file") if settings.get("sink") == "file" or settings.get("file") else None)
            if target:
                self._write_file(record, str(target), settings)
            else:
                self._write_stream(record, settings)
        except Exception:
            self.handleError(record)

    def close(self) -> None:
        """Use ``close`` in the HeavenBase utility surface.

        Use this helper for HeavenBase loggers whose routing can be changed through runtime config.

        Args:
            None.

        Returns:
            None: This function returns ``None`` after completing its side effect.
        """
        self._close_file()
        super().close()

    def _write_stream(self, record: logging.LogRecord, settings: Dict[str, Any]) -> None:
        stream = self.stream or _GLOBAL_STREAM
        fmt = self.fmt or settings.get("stream_format") or "%(levelname)s:%(name)s:%(message)s"
        console = stream.isatty() if hasattr(stream, "isatty") else False
        message = ColoredFormatter(fmt, self.datefmt, colors=self.colors, console=console).format(record)
        stream.write(message + self.terminator)
        stream.flush()

    def _write_file(self, record: logging.LogRecord, path: str, settings: Dict[str, Any]) -> None:
        stream = self._file(path)
        fmt = self.fmt or settings.get("format") or "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        message = logging.Formatter(fmt, self.datefmt).format(record)
        stream.write(message + self.terminator)
        stream.flush()

    def _file(self, path: str) -> Any:
        path = _log_path(path)
        if self.file_path != path or self.file_stream is None:
            self._close_file()
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            self.file_stream = open(path, "a", encoding="utf-8")
            self.file_path = path
        return self.file_stream

    def _close_file(self) -> None:
        if self.file_stream is not None:
            self.file_stream.close()
        self.file_stream = None
        self.file_path = None


class SuppressOutput:
    """Suppress stdout/stderr at the file-descriptor level."""

    def __init__(self, suppress_stdout: bool = True, suppress_stderr: bool = True) -> None:
        """
        Initialize the utility object with the provided runtime settings.

        Use this helper for HeavenBase loggers whose routing can be changed through runtime config.

        Args:
            suppress_stdout (bool): Whether stdout should be redirected to ``os.devnull``. Defaults to ``True``.
            suppress_stderr (bool): Whether stderr should be redirected to ``os.devnull``. Defaults to ``True``.

        Returns:
            None: This function returns ``None`` after initializing the object.
        """
        self.suppress_stdout = suppress_stdout
        self.suppress_stderr = suppress_stderr
        self._devnull = None
        self._old_stdout_fd = None
        self._old_stderr_fd = None
        self._suppressing = False

    def __enter__(self) -> "SuppressOutput":
        """
        Enter the context manager and apply its temporary state.

        Use this helper for HeavenBase loggers whose routing can be changed through runtime config.

        Returns:
            'SuppressOutput': The active output-suppression context.
        """
        self._devnull = open(os.devnull, "w")
        if self.suppress_stdout:
            self._old_stdout_fd = os.dup(sys.stdout.fileno())
        if self.suppress_stderr:
            self._old_stderr_fd = os.dup(sys.stderr.fileno())
        self.suppress()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        """
        Exit the context manager and restore the previous state.

        Use this helper for HeavenBase loggers whose routing can be changed through runtime config.

        Args:
            exc_type (Any): Exception type passed by the context-manager protocol, or ``None`` on a clean exit.
            exc_val (Any): Exception instance passed by the context-manager protocol, or ``None`` on a clean exit.
            exc_tb (Any): Traceback passed by the context-manager protocol, or ``None`` on a clean exit.

        Returns:
            bool: ``False`` so context-manager exceptions are never swallowed.
        """
        self.restore()
        for fd in (self._old_stdout_fd, self._old_stderr_fd):
            if fd is not None:
                os.close(fd)
        if self._devnull is not None:
            self._devnull.close()
        return False

    def suppress(self) -> None:
        """Redirect configured streams to devnull.

        Use this helper for HeavenBase loggers whose routing can be changed through runtime config.

        Args:
            None.

        Returns:
            None: This function returns ``None`` after completing its side effect.
        """
        if self._suppressing or self._devnull is None:
            return
        if self.suppress_stdout and self._old_stdout_fd is not None:
            os.dup2(self._devnull.fileno(), sys.stdout.fileno())
        if self.suppress_stderr and self._old_stderr_fd is not None:
            os.dup2(self._devnull.fileno(), sys.stderr.fileno())
        self._suppressing = True

    def restore(self) -> None:
        """Restore configured streams after suppression.

        Use this helper for HeavenBase loggers whose routing can be changed through runtime config.

        Args:
            None.

        Returns:
            None: This function returns ``None`` after completing its side effect.
        """
        if not self._suppressing:
            return
        if self.suppress_stdout and self._old_stdout_fd is not None:
            os.dup2(self._old_stdout_fd, sys.stdout.fileno())
        if self.suppress_stderr and self._old_stderr_fd is not None:
            os.dup2(self._old_stderr_fd, sys.stderr.fileno())
        self._suppressing = False


def get_logger(
    name: str,
    level: Optional[Union[str, int]] = None,
    fmt: Optional[str] = None,
    datefmt: Optional[str] = None,
    style: str = "%",
    *,
    validate: bool = True,
    colors: Optional[Dict[int, Callable[[Any, bool], str]]] = None,
    stream: Optional[Any] = None,
) -> logging.Logger:
    """
    Return a HeavenBase logger whose output is routed through runtime log settings.

    Use this helper for HeavenBase loggers whose routing can be changed through runtime config.

    Args:
        name (str): Name, label, scope, package, or logger identifier used by the helper.
        level (Optional[Union[str, int]]): Logging level name or numeric level. Examples: ``"DEBUG"``, ``"INFO"``, and ``"WARNING"``. Defaults to ``None``.
        fmt (Optional[str]): Log or display format string. Defaults to the configured HeavenBase format.
        datefmt (Optional[str]): Optional logging date format. Defaults to ``None``.
        style (str): Logging format style accepted by ``logging.Formatter``. Defaults to ``"%"``.
        validate (bool): Whether ``logging.Formatter`` validates the format string. Defaults to ``True``.
        colors (Optional[Dict[int, Callable[[Any, bool], str]]]): Optional log-level-to-color override mapping. Defaults to ``None``.
        stream (Optional[Any]): Optional stream-like object used for logging or printing. Defaults to ``None``.

    Returns:
        logging.Logger: Configured ``logging.Logger`` instance.

    Examples:
        >>> from heavenbase.utils.log import get_logger
        >>> logger = get_logger("heavenbase.demo")
        >>> logger.name
        'heavenbase.demo'
    """
    with _LOGGER_LOCK:
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        handler = _handler(logger)
        if handler is None:
            logger.addHandler(HbLogHandler(stream=stream, fmt=fmt, datefmt=datefmt, colors=colors))
        else:
            handler.acquire()
            try:
                handler.stream = stream or handler.stream
                handler.fmt = fmt or handler.fmt
                handler.datefmt = datefmt or handler.datefmt
                handler.colors = colors or handler.colors
            finally:
                handler.release()
        if level is not None:
            _set_logger_level(name, level)
        if not hasattr(logger, "success"):
            logger.success = _success.__get__(logger, logging.Logger)  # type: ignore[attr-defined]
        _LOGGERS[name] = logger
        return logger


def configure_logs(
    sink: Optional[str] = None,
    file: Optional[str] = None,
    level: Optional[Union[str, int]] = None,
    fmt: Optional[str] = None,
    stream_format: Optional[str] = None,
    suppress: Optional[bool] = None,
    *,
    persist: bool = False,
) -> None:
    """
    Update global HeavenBase log routing settings.

    Settings apply to the current process by default. With ``persist=True``,
    save them through the process default's config so processes sharing
    it can apply them after their config cache refreshes. If saving fails,
    apply the settings in this process.

    Args:
        sink (Optional[str]): Global log sink. Allowed values are ``"stream"``, ``"file"``, and ``"suppress"``. Defaults to ``None``.
        file (Optional[str]): Log file path; ``""`` switches back to stream
            output. A relative path, ``~`` and environment variables are
            resolved in the calling process before the path is saved, so every
            process writes the file the caller named. Symbolic links are not
            resolved then; each record follows them. A path containing literal
            variable markers after expansion is saved as a file URI to preserve
            them. A path that starts with
            the ``%`` (HeavenBase home) or ``&`` (package resources) alias is
            saved as written. Defaults to ``None``.
        level (Optional[Union[str, int]]): Logging level name or numeric level. Examples: ``"DEBUG"``, ``"INFO"``, and ``"WARNING"``. Defaults to ``None``.
        fmt (Optional[str]): Log or display format string. Defaults to the configured HeavenBase format.
        stream_format (Optional[str]): Optional format used for stream log output. ``None`` keeps the configured default.
        suppress (Optional[bool]): Optional global suppression flag. ``None`` leaves suppression unchanged.
        persist (bool): Also save the settings through the default Context's config. Defaults to ``False``.

    Returns:
        None: ``None`` after global settings are updated.
    """
    settings: Dict[str, Any] = {}
    if sink is not None:
        if sink not in ("stream", "file", "suppress"):
            raise ValueError("log sink must be 'stream', 'file', or 'suppress'")
        settings["sink"] = sink
    if file is not None:
        file = _saved_log_file(file)
        settings["file"] = file
        settings["sink"] = "file" if file else "stream"
    if level is not None:
        settings["level"] = logging.getLevelName(_level_value(level, logging.INFO))
    if fmt is not None:
        settings["format"] = fmt
    if stream_format is not None:
        settings["stream_format"] = stream_format
    if suppress is not None:
        settings["suppress"] = bool(suppress)
        if suppress:
            settings["sink"] = "suppress"
    _update_log_settings(settings, persist=persist)


def set_log_level(level: Union[str, int], loggers: Optional[List[str]] = None, *, persist: bool = False) -> None:
    """
    Set global or selected logger level.

    Args:
        level (Union[str, int]): Logging level name or numeric level. Examples: ``"DEBUG"``, ``"INFO"``, and ``"WARNING"``.
        loggers (Optional[List[str]]): Optional logger-name list. ``None`` applies the change globally.
        persist (bool): Also save the global level through :func:`configure_logs` when ``loggers`` is omitted. Defaults to ``False``.

    Returns:
        None: ``None`` after the requested log levels are updated.
    """
    if loggers is None:
        configure_logs(level=level, persist=persist)
        return
    for name in loggers:
        _set_logger_level(name, level)


def redirect_logs(
    target: Union[str, Any],
    loggers: Optional[List[str]] = None,
    level: Optional[Union[str, int]] = None,
    fmt: Optional[str] = None,
    *,
    persist: bool = False,
) -> None:
    """
    Redirect global or selected HeavenBase loggers to a path or stream.

    A global path is expanded in the calling process. Each record follows
    any symbolic links in that path. A stream object stays in this process.

    Args:
        target (Union[str, Any]): Path or stream that receives redirected log output.
        loggers (Optional[List[str]]): Optional logger-name list. ``None`` applies the change globally.
        level (Optional[Union[str, int]]): Logging level name or numeric level. Examples: ``"DEBUG"``, ``"INFO"``, and ``"WARNING"``. Defaults to ``None``.
        fmt (Optional[str]): Log or display format string. Defaults to the configured HeavenBase format.
        persist (bool): Also save the global routing through :func:`configure_logs` when ``loggers`` is omitted. Defaults to ``False``.

    Returns:
        None: ``None`` after log routing is changed.
    """
    if hasattr(target, "write"):
        if loggers is None:
            configure_logs(file="", level=level, stream_format=fmt, suppress=False, persist=persist)
        _redirect_stream(target, loggers=loggers)
        if loggers is not None:
            if level is not None:
                set_log_level(level, loggers=loggers)
            if fmt is not None:
                _set_handler_fmt(loggers, fmt)
        return
    if loggers is None:
        configure_logs(file=str(target), level=level, fmt=fmt, persist=persist)
        return
    with _LOGGER_LOCK:
        for name in loggers:
            logger = _LOGGERS.get(name) or get_logger(name)
            handler = _handler(logger)
            if handler is not None:
                handler.acquire()
                try:
                    handler.file_target = str(target)
                    handler.stream = None
                    handler.fmt = fmt or handler.fmt
                finally:
                    handler.release()
    if level is not None:
        set_log_level(level, loggers=loggers)


def suppress_logs(loggers: Optional[List[str]] = None, *, persist: bool = False) -> None:
    """
    Suppress global or selected HeavenBase loggers.

    Args:
        loggers (Optional[List[str]]): Optional logger-name list. ``None`` applies the change globally.
        persist (bool): Also save global suppression through :func:`configure_logs` when ``loggers`` is omitted. Defaults to ``False``.

    Returns:
        None: ``None`` after selected logs are suppressed.
    """
    if loggers is None:
        configure_logs(sink="suppress", suppress=True, persist=persist)
        return
    with _LOGGER_LOCK:
        for name in loggers:
            logger = _LOGGERS.get(name) or get_logger(name)
            handler = _handler(logger)
            if handler is not None:
                handler.acquire()
                try:
                    handler.suppressed = True
                finally:
                    handler.release()


def restore_logs(loggers: Optional[List[str]] = None, *, persist: bool = False) -> None:
    """
    Restore global or selected HeavenBase loggers to stream output.

    Args:
        loggers (Optional[List[str]]): Optional logger-name list. ``None`` applies the change globally.
        persist (bool): Also save the global stream routing through :func:`configure_logs` when ``loggers`` is omitted. Defaults to ``False``.

    Returns:
        None: ``None`` after log routing is restored.
    """
    global _GLOBAL_STREAM
    if loggers is None:
        # Config writes can select the default Context, whose imports create
        # loggers, so they run outside the logger table lock.
        configure_logs(sink="stream", file="", suppress=False, persist=persist)
    with _LOGGER_LOCK:
        if loggers is None:
            _GLOBAL_STREAM = sys.stderr
            targets = list(_LOGGERS.values())
        else:
            targets = [_LOGGERS[name] for name in loggers if name in _LOGGERS]
        for logger in targets:
            handler = _handler(logger)
            if handler is not None:
                handler.acquire()
                try:
                    handler.suppressed = False
                    handler.level_override = None
                    handler.file_target = None
                    handler._close_file()
                finally:
                    handler.release()


def safe_error(exc: BaseException) -> str:
    """
    Return exception text with type information and traceback chains preserved.

    Use this helper for HeavenBase loggers whose routing can be changed through runtime config.

    Args:
        exc (BaseException): Exception instance passed by the context-manager protocol, or ``None`` on a clean exit.

    Returns:
        str: Exception text with the exception type included.
    """
    return error_str(exc, tb=False) or ""


def _success(self: logging.Logger, message: Any, *args: Any, **kwargs: Any) -> None:
    if self.isEnabledFor(SUCCESS_LEVEL):
        self._log(SUCCESS_LEVEL, message, args, **kwargs)


def _handler(logger: logging.Logger) -> Optional[HbLogHandler]:
    for handler in logger.handlers:
        if isinstance(handler, HbLogHandler):
            return handler
    return None


def _redirect_stream(stream: Any, loggers: Optional[List[str]]) -> None:
    global _GLOBAL_STREAM
    if loggers is None:
        with _LOGGER_LOCK:
            _GLOBAL_STREAM = stream
        return
    with _LOGGER_LOCK:
        for name in loggers:
            logger = _LOGGERS.get(name) or get_logger(name)
            handler = _handler(logger)
            if handler is not None:
                handler.acquire()
                try:
                    handler.stream = stream
                    handler.suppressed = False
                finally:
                    handler.release()


def _set_handler_fmt(loggers: List[str], fmt: str) -> None:
    with _LOGGER_LOCK:
        for name in loggers:
            logger = _LOGGERS.get(name)
            handler = _handler(logger) if logger is not None else None
            if handler is not None:
                handler.acquire()
                try:
                    handler.fmt = fmt
                finally:
                    handler.release()


def _set_logger_level(name: str, level: Union[str, int]) -> None:
    with _LOGGER_LOCK:
        logger = _LOGGERS.get(name) or logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        handler = _handler(logger)
        if handler is not None:
            handler.acquire()
            try:
                handler.setLevel(logging.DEBUG)
                handler.level_override = _level_value(level, logging.INFO)
            finally:
                handler.release()


def _log_config() -> ConfigManager:
    """Use the process default's manager when it has been installed."""
    from .config import CM_HVNB

    try:
        default = resolve_late("heavenbase.log.default_context")()
    except KeyError:
        return CM_HVNB
    return CM_HVNB if default is None else default.config


def _log_settings() -> Dict[str, Any]:
    try:
        # A log record never fails over config: an unopenable home or explicit
        # Context mode falls back to the packaged settings, and any other
        # config error to the built-in settings for this record only.
        settings = dict(_log_config()._get_or_packaged("heavenbase.log", {}) or {})
    except Exception:
        settings = {}
    settings.update(_PROCESS_SETTINGS)
    return {
        "level": settings.get("level", "INFO"),
        "sink": settings.get("sink", "stream"),
        "file": settings.get("file"),
        "format": settings.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
        "stream_format": settings.get("stream_format", "%(levelname)s:%(name)s:%(message)s"),
        "suppress": settings.get("suppress", False),
    }


def _update_log_settings(settings: Dict[str, Any], *, persist: bool) -> None:
    """Publish process overrides and clear them for successfully saved keys."""
    global _PROCESS_SETTINGS
    saved = set()
    if persist and settings:
        manager = _log_config()

        for key, value in settings.items():
            try:
                if manager._set_if_openable(f"heavenbase.log.{key}", value) and manager.last_write_persisted:
                    saved.add(key)
            except Exception:
                continue
    with _PROCESS_SETTINGS_LOCK:
        local = dict(_PROCESS_SETTINGS)
        for key, value in settings.items():
            if key in saved:
                local.pop(key, None)
            else:
                local[key] = value
        # Replaced, never mutated, so a log record reads it without the lock.
        _PROCESS_SETTINGS = local


def _saved_log_file(file: Any) -> str:
    """Return a global log file path in the form every process resolves alike.

    The setting is saved to the HeavenBase home and read by other processes,
    each with its own working directory, ``HOME`` and environment. A relative
    path, ``~`` and environment variables are therefore resolved here, in the
    calling process. A path that starts with the ``%`` (HeavenBase home) or
    ``&`` (package resources) alias already names one file for every process
    on the home and is kept as written; ``""`` (stream output) is kept too.

    Symbolic links are not resolved here: each record resolves the saved path
    again, as 0.1.2.2 did, so a path through a link follows the link's current
    target. When the working directory no longer exists, a
    relative path cannot be resolved and is kept as written.
    """
    text = os.fspath(file) if isinstance(file, os.PathLike) else str(file)
    stripped = text.strip()
    if not stripped or stripped.startswith(("%", "&")):
        return text
    expanded = os.path.expandvars(os.path.expanduser(stripped))
    try:
        if os.name == "nt":
            absolute = os.path.abspath(expanded)
        else:
            absolute = expanded if os.path.isabs(expanded) else os.path.join(os.getcwd(), expanded)
    except OSError:
        return text
    if "$" in absolute or (os.name == "nt" and "%" in absolute) or absolute != absolute.rstrip():
        return Path(absolute).as_uri()
    return absolute


def _log_path(path: str) -> str:
    if path.startswith("file://"):
        uri = urlsplit(path)
        literal = os.fsdecode(unquote_to_bytes(uri.path))
        if uri.netloc:
            authority = os.fsdecode(unquote_to_bytes(uri.netloc))
            literal = f"//{authority}{literal}"
        elif os.name == "nt":
            literal = literal.lstrip("/")
        return str(Path(literal).resolve())
    try:
        return _log_config().pj(path, abs=True)
    except Exception:
        return path


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in ("1", "true", "yes", "on") if isinstance(value, str) else bool(value)


def _level_names() -> Dict[str, int]:
    get_mapping = getattr(logging, "getLevelNamesMapping", None)
    if get_mapping is not None:
        return get_mapping()
    return {
        "CRITICAL": logging.CRITICAL,
        "FATAL": logging.FATAL,
        "ERROR": logging.ERROR,
        "WARN": logging.WARNING,
        "WARNING": logging.WARNING,
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG,
        "NOTSET": logging.NOTSET,
        "SUCCESS": SUCCESS_LEVEL,
    }


def _level_value(level: Optional[Union[str, int]], default: int) -> int:
    if level is None:
        return default
    if isinstance(level, str):
        return _level_names().get(level.upper(), default)
    return int(level)
