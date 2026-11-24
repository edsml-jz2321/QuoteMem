"""Command execution helpers with explicit result contracts."""

from __future__ import annotations

__all__ = ["CmdResult", "browse", "clipboard", "cmd", "is_linux", "is_macos", "is_windows"]

import os
import platform
import shutil
import subprocess

from .typing import Any, Dict, List, Optional, Sequence, Union, dataclass


@dataclass(frozen=True)
class CmdResult:
    """Completed command result."""

    args: Union[str, Sequence[str]]
    code: Optional[int]
    out: Optional[str] = None
    err: Optional[str] = None
    handle: Optional[subprocess.Popen] = None

    @property
    def ok(self) -> bool:
        """Return whether the command exited successfully.

        Use this helper when scripts need explicit command results, platform checks, or small desktop integrations.

        Args:
            None.

        Returns:
            bool: The whether the command exited successfully.
        """
        return self.code == 0


def cmd(
    command: Union[str, Sequence[str]],
    wait: bool = True,
    shell: Optional[bool] = None,
    sudo: bool = False,
    check: bool = False,
    include: Optional[Union[str, Sequence[str]]] = None,
    **kwargs: Any,
) -> Any:
    """Run a local command and return an explicit result object or selected fields.

    Use this helper when scripts need explicit command results, platform checks, or small desktop integrations.

    Warnings:
        Passing ``shell=True`` gives the shell control over parsing. Prefer sequence commands for user-provided values.

    Args:
        command (Union[str, Sequence[str]]): Shell command string or argument sequence. Sequence commands avoid a shell unless ``shell=True``.
        wait (bool): Whether to wait for a command to finish. Defaults to ``True``.
        shell (Optional[bool]): Whether to run the command through a shell. ``None`` uses ``True`` for string commands and ``False`` for sequences. Defaults to ``None``.
        sudo (bool): Whether to prefix the command with ``sudo``. Defaults to ``False``.
        check (bool): Whether a non-zero command exit should raise ``CalledProcessError``. Defaults to ``False``.
        include (Optional[Union[str, Sequence[str]]]): Optional result field or fields to return from ``cmd``. Choices include ``"out"``, ``"err"``, ``"ok"``, ``"code"``, ``"handle"``, and ``"result"``. Defaults to ``None``.
        **kwargs (Any): Extra keyword arguments forwarded to the underlying serializer, command, or callable.

    Returns:
        Any: ``CmdResult`` or selected field values, depending on ``include``.

    Examples:
        >>> from heavenbase.utils import cmd
        >>> cmd(["python", "-c", "print('ok')"], include="out")
        'ok'
    """
    keys = [] if include is None else ([include] if isinstance(include, str) else list(include))
    supported = {"handle", "stdout", "stderr", "returncode", "code", "out", "err", "ok", "result"}
    bad = [key for key in keys if key not in supported]
    if bad:
        raise ValueError(f"Unsupported cmd include keys: {bad}. Supported: {sorted(supported)}")

    local_kwargs = dict(kwargs)
    input_value = local_kwargs.pop("input", None)
    if "text" not in local_kwargs and "universal_newlines" not in local_kwargs and "encoding" not in local_kwargs:
        local_kwargs["text"] = True
    if input_value is not None and "stdin" not in local_kwargs:
        local_kwargs["stdin"] = subprocess.PIPE

    args, actual_shell = _cmd_args(command, shell=shell, sudo=sudo)
    want_out = wait and (include is None or any(key in keys for key in ("stdout", "out", "result")))
    want_err = wait and (include is None or any(key in keys for key in ("stderr", "err", "result")))
    if not wait:
        want_out = any(key in keys for key in ("stdout", "out"))
        want_err = any(key in keys for key in ("stderr", "err"))
    if want_out and "stdout" not in local_kwargs:
        local_kwargs["stdout"] = subprocess.PIPE
    if want_err and "stderr" not in local_kwargs:
        local_kwargs["stderr"] = subprocess.PIPE

    proc = subprocess.Popen(args, shell=actual_shell, **local_kwargs)
    if wait:
        out, err = proc.communicate(input=input_value)
        result = CmdResult(command, proc.returncode, _strip(out), _strip(err), proc)
    else:
        result = CmdResult(command, None, proc.stdout if want_out else None, proc.stderr if want_err else None, proc)
    if check and wait and not result.ok:
        raise subprocess.CalledProcessError(result.code, args, output=result.out, stderr=result.err)
    if include is None:
        return result
    values: Dict[str, Any] = {
        "stdout": result.out,
        "stderr": result.err,
        "returncode": result.code,
        "code": result.code,
        "out": result.out,
        "err": result.err,
        "ok": result.ok,
        "handle": result.handle,
        "result": result,
    }
    out = {key: values[key] for key in keys}
    return next(iter(out.values())) if len(out) == 1 else out


def _cmd_args(command: Union[str, Sequence[str]], shell: Optional[bool], sudo: bool) -> tuple[Union[str, List[str]], bool]:
    if isinstance(command, str):
        return (f"sudo {command}" if sudo else command), (True if shell is None else shell)
    args = [str(item) for item in command]
    actual_shell = False if shell is None else shell
    if not sudo:
        return (" ".join(args) if actual_shell else args), actual_shell
    return (f"sudo {' '.join(args)}" if actual_shell else ["sudo", *args]), actual_shell


def _strip(value: Any) -> Any:
    return value.strip() if isinstance(value, str) else value


def is_macos() -> bool:
    """
    Return whether the current platform is macOS.

    Use this helper when scripts need explicit command results, platform checks, or small desktop integrations.
    Args:
        None.


    Returns:
        bool: The whether the current platform is macOS.
    """
    return platform.system() == "Darwin"


def is_windows() -> bool:
    """
    Return whether the current platform is Windows.

    Use this helper when scripts need explicit command results, platform checks, or small desktop integrations.
    Args:
        None.


    Returns:
        bool: The whether the current platform is Windows.
    """
    return platform.system() == "Windows"


def is_linux() -> bool:
    """
    Return whether the current platform is Linux.

    Use this helper when scripts need explicit command results, platform checks, or small desktop integrations.
    Args:
        None.


    Returns:
        bool: The whether the current platform is Linux.
    """
    return platform.system() == "Linux"


def browse(path: str) -> None:
    """
    Open a local path with the operating system default application.

    Use this helper when scripts need explicit command results, platform checks, or small desktop integrations.

    Args:
        path (str): Local path, config path, or dotted dictionary path used by the helper.

    Returns:
        None: ``None`` after the platform launcher is requested.
    """
    if is_windows():
        os.startfile(path)  # type: ignore[attr-defined]
    elif is_macos():
        subprocess.Popen(["open", path])
    elif is_linux():
        subprocess.Popen(["xdg-open", path])
    else:
        subprocess.Popen(["open", path])


def clipboard(text: str) -> bool:
    """
    Copy text to the host clipboard when a supported clipboard command exists.

    Use this helper when scripts need explicit command results, platform checks, or small desktop integrations.

    Args:
        text (str): Text value to format, normalize, copy, or compare.

    Returns:
        bool: ``True`` when text was copied successfully; otherwise ``False``.
    """
    for tool, args in (
        ("pbcopy", ["pbcopy"]),
        ("clip", ["clip"]),
        ("xclip", ["xclip", "-selection", "clipboard"]),
        ("xsel", ["xsel", "--clipboard", "--input"]),
    ):
        if shutil.which(tool):
            return subprocess.run(args, input=text, text=True).returncode == 0
    return False
