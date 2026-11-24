"""In-process Python execution for caller-authored analysis code."""

from __future__ import annotations

__all__ = ["PYTHON_RESULT_CHARS", "run_python_code"]

import ast
import time
import traceback

# Use the installed capture owner even when this runner is loaded as an artifact.
from heavenbase.extensions.python._stdout import _BoundedText, _capture_stdout
from heavenbase.utils import Any, Dict, Optional, coerce_posint
from heavenbase.utils.serialize import validate_json

# Results feed a model's context, so a large frame must not arrive whole. The
# cap is declared in the payload rather than applied quietly, because an
# unmarked cut is what makes a caller report a slice as a total.
PYTHON_RESULT_CHARS = 20000


def run_python_code(code: str, namespace: Dict[str, Any], *, result_chars: int = PYTHON_RESULT_CHARS) -> Dict[str, Any]:
    """Execute one Python program and return its output with its coverage.

    Statements run through :func:`exec`; a trailing expression is evaluated so
    the last line's value is the result, matching the REPL behaviour callers
    expect. Execution is in-process and unrestricted: the program can do
    anything the host process can.

    Each run captures stdout from its executing thread and workers started
    through `threading.Thread.start` or a live `_thread.start_new_thread`
    module lookup. Nested runs get their own capture. Threads outside a run's
    capture write to the host stdout. A run installs a capture view as
    `sys.stdout`. The view is a text stream (an `io.TextIOBase`), as 0.1.2.2's
    run buffer was, so `isinstance`, `with`, iteration and duck typing such as
    pandas' `is_file_like` see a file, and a run's threads see their capture
    through it. On any other thread, a run's thread after its run included, it
    has the public text-stream names that the host's stream has, `write`
    always, and no other public name beyond what a program set on it. There,
    writes and flushes reach the host's stream, and isatty(), writable(),
    tell(), closed, encoding, errors and newlines answer as it does.
    detach(), fileno(), read(), readline(), readlines(), seek() and truncate()
    raise io.UnsupportedOperation, readable() and seekable() are False, and
    `buffer` and `reconfigure()` are not there, as with 0.1.2.2's buffer;
    close() does nothing there, where 0.1.2.2 closed the run's buffer. Leaving
    a `with sys.stdout:` block calls the close a program set on `sys.stdout`,
    if it is still set, or else closes the capture that was entered, as it
    closed 0.1.2.2's buffer, wherever the block is left, and the run's later
    writes are still captured. A direct close() closes the calling thread's
    capture only while an active run uses that `sys.stdout`, so a close that
    one run sets up and another run's thread performs, such as
    `contextlib.closing(sys.stdout)` in a suspended generator, closes nothing
    once every run that used that `sys.stdout` has ended; while a run that
    overlapped the one that set it up still uses it, that run closes its own
    capture by performing it. What a run's program sets on `sys.stdout`, such
    as a patched `write`, ends when the view retires, with that run and any
    run that overlaps it, as it ended with 0.1.2.2's buffer. A run's threads
    write to its capture through any capture view installed no later than the
    run started; other writes through a view go to the stream it wraps.
    Direct redirection therefore follows Python's process-wide stdout
    behavior while runs are active: a redirect entered after a run started
    receives that run's output while it is current, and while `sys.stdout` is
    not a capture view every thread writes to it. A capture view is in use
    from when a run installs it, or finds it in `sys.stdout` at its start,
    until a run ends and no active run is left that started when the view was
    installed or later; if the view is in `sys.stdout` then, the stream it
    wraps is put back. When a run ends and `sys.stdout` is anything but a
    capture view in use, or does not exist, the run puts back what it found,
    as 0.1.2.2's `contextlib.redirect_stdout` did, also while other runs are
    active. So a program that assigns `sys.stdout` itself
    (`sys.stdout = io.StringIO()`) sends its later output to that stream, and
    when its run ends the host, or an outer run on whatever thread, gets back
    the stdout that the run found, as after a program that deletes
    `sys.stdout`. A run that started after the assignment found the assigned
    stream, so while such a run is active its view stays over that stream,
    which comes back when the last such run ends, as it did when such a run
    ended in 0.1.2.2. A run's end undoes in the same way what host code, or
    the program of a run on another thread, assigned during the run, as in
    0.1.2.2. With no other run active, for example, a host `redirect_stdout`
    entered during the run, and whose target is `sys.stdout` when the run
    ends, stops receiving what is written there, and one whose target the run
    found, left during the run, has that target put back.
    A capture view that host code puts back while no run is active, as a
    `contextlib.redirect_stdout` entered during a run and left after it does,
    passes writes from outside runs to the stream it wraps, while that stream
    is alive, until a later run ends with no other run active and puts that
    stream back. A view that is not in use holds its stream weakly, so a
    stream that only `sys.stdout` referred to is released when such a view
    takes its place there, and one that takes no weak references is lost to
    such a view. The view then drops those writes, and instead of the stream
    `sys.__stdout__` comes back, or, where that is None, closed or detached,
    the view stays. A run's end closes its capture to writes: a write from a
    thread the run started that reaches the capture after that goes to the
    stream the view wraps, so every write reaches the run's capture or that
    stream, never neither, and `stdout`, `stdout_truncated` and `stdout_chars`
    describe the same closed capture. Capture never blocks interpreter exit
    while a run still writes on a daemon thread, and it never blocks a child
    forked while a run's threads write. A child forked from a thread outside
    runs writes to and flushes the stream the view wraps; a child forked from
    a thread of a run keeps that thread's routing, so its writes go to its own
    copy of the run's capture, and neither the parent's run nor its stream
    receives them. During a run that started while `sys.stdout` was None,
    threads outside runs see a null text stream: their writes are dropped,
    `isatty()` is False, and closing it does nothing. A thread a run started
    keeps nothing alive once it finishes. A capture view stays alive after it
    leaves `sys.stdout`, so a thread still in `print()` never uses a freed
    one, but once it is no longer in use, it holds the stream it wrapped only
    weakly, so the host's stream is released as soon as the host drops it.
    While runs overlap without a gap, an older view stays in use and keeps its
    stream, where 0.1.2.2 let it go when its own run ended. Copies and deep
    copies of a capture view share the stream it wraps, and hold it weakly if
    it takes weak references. The type of `sys` and of every module is never
    changed.

    A `ResultFrame` result is rendered through the workspace payload builder, so
    a paginated frame still reports `total` and `truncated` instead of letting a
    page read as a whole. Cell values without a JSON representation use text.
    Printed output and rendered results are capped, and a cap that fired is
    stated in the payload.

    Args:
        code (str): Python source to execute.
        namespace (Dict[str, Any]): Globals the program runs against. Mutations
            persist in the mapping the caller supplied.
        result_chars (int): Maximum characters kept for `stdout` and for the
            rendered result. Defaults to :data:`PYTHON_RESULT_CHARS`.

    Returns:
        Dict[str, Any]: ``{"ok", "stdout", "result", "elapsed"}`` on success,
        plus ``"error"``/``"traceback"`` when the program raised, and
        ``"stdout_truncated"``/``"result_truncated"`` when a cap fired.

    Raises:
        SyntaxError: If ``code`` does not parse. Runtime failures are reported
            in the payload rather than raised, so a caller sees the program's
            own output alongside its error.
        AttributeError: If `sys.stdout` does not exist when the run starts,
            as with 0.1.2.2; nothing is changed then.
    """
    result_chars = coerce_posint(result_chars, "run_python result_chars")
    tree = ast.parse(code)
    tail = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
    stdout = _BoundedText(result_chars)
    payload: Dict[str, Any] = {"ok": True}
    started = time.time()
    with _capture_stdout(stdout):
        try:
            if tree.body:
                exec(compile(tree, "<python>", "exec"), namespace, namespace)  # noqa: S102 - caller-authored code is the feature
            value = eval(compile(ast.Expression(tail.value), "<python>", "eval"), namespace, namespace) if tail is not None else None
            _set_bounded(payload, "result", _render_result(value), result_chars)
        except BaseException as error:  # a program's own failure is data for the caller, not an exception here
            payload.update({"ok": False, "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc(limit=8)})
    payload["elapsed"] = round(time.time() - started, 3)
    # The run's end closed the capture, so its text and count are final and read together.
    payload["stdout"], chars = stdout.snapshot()
    if chars > result_chars:
        payload["stdout_truncated"] = True
        payload["stdout_chars"] = chars
    return payload


def _render_result(value: Any) -> Optional[Any]:
    """Return one JSON-ready result, preserving a frame's coverage metadata."""
    if hasattr(value, "rows") and hasattr(value, "col_names"):
        from heavenbase.toolkit.families.workspace.result import frame_dict

        frame = frame_dict(value)
        try:
            validate_json(frame)
        except (TypeError, ValueError, RecursionError):
            rows = [{column: _json_result(cell) for column, cell in row.items()} for row in frame["rows"]]
            frame["rows"] = rows
            frame["columns"] = {column: [row[column] for row in rows] for column in frame["columns"]}
            if "ids" in frame:
                frame["ids"] = [_json_result(cell) for cell in frame["ids"]]
            return _json_result(frame)
        return frame
    return _json_result(value)


def _json_result(value: Any) -> Any:
    try:
        validate_json(value)
        return value
    except (TypeError, ValueError, RecursionError):
        return repr(value)


def _set_bounded(payload: Dict[str, Any], key: str, value: Any, limit: int) -> None:
    """Store one value, declaring the cut when it did not fit."""
    if not isinstance(value, str):
        rendered = value
        text = None if value is None else str(value)
    else:
        rendered = value
        text = value
    if text is not None and len(text) > limit:
        payload[key] = text[:limit]
        payload[f"{key}_truncated"] = True
        payload[f"{key}_chars"] = len(text)
        return
    payload[key] = rendered
