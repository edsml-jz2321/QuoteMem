"""HeavenBase-native parallel runtime.

Sync and async batch execution with bounded concurrency, stable ordering,
completion-order streaming, per-task error capture, graceful cancellation,
and lightweight progress hooks.
"""

from __future__ import annotations

__all__ = [
    "TaskResult",
    "batch",
    "abatch",
    "batch_stream",
    "abatch_stream",
    "pmap",
    "pforeach",
]

import asyncio
import inspect
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from functools import partial
from multiprocessing import cpu_count

from .typing import Any, AsyncGenerator, Callable, Dict, Iterable, Iterator, List, Literal, Mapping, Optional, Tuple, dataclass

ErrorPolicy = Literal["capture", "raise", "skip"]


def _cfg(key: str, default: Any) -> Any:
    from .config import CM_HVNB

    # Worker sizing never fails over config: an unopenable home or explicit
    # Context mode falls back to the packaged value, and any other config
    # error to ``default``, as in 0.1.2.2.
    try:
        return CM_HVNB._get_or_packaged(f"heavenbase.parallel.{key}", default)
    except Exception:
        return default


def _resolve_max_workers(max_workers: Optional[int]) -> int:
    if max_workers is not None:
        return max(1, int(max_workers))
    configured = _cfg("max_workers", None)
    if configured is not None:
        return max(1, int(configured))
    cap = max(1, int(_cfg("fallback_max_workers", 32)))
    extra = int(_cfg("cpu_extra_workers", 4))
    return min(cap, _cpu_count() + extra)


def _cpu_count() -> int:
    try:
        return max(1, int(cpu_count()))
    except NotImplementedError:
        return 1


def _validate_on_error(on_error: str) -> ErrorPolicy:
    if on_error not in ("capture", "raise", "skip"):
        raise ValueError(f"on_error must be 'capture', 'raise', or 'skip', got {on_error!r}")
    return on_error  # type: ignore[return-value]


@dataclass
class TaskResult:
    """Outcome of a single batched task."""

    index: int
    item: Any
    ok: bool
    value: Any = None
    error: Optional[BaseException] = None
    elapsed_ms: Optional[float] = None

    def raise_if_error(self) -> None:
        """Raise the captured error if the task failed.

        Use this helper for bounded local concurrency while keeping result ordering and error policy explicit.

        Args:
            None.

        Returns:
            None: This function returns ``None`` after completing its side effect.
        """
        if not self.ok:
            if self.error is not None:
                raise self.error
            raise RuntimeError(f"Task {self.index} failed without an error object")

    def to_dict(self) -> dict:
        """Return a plain dictionary representation.

        Use this helper for bounded local concurrency while keeping result ordering and error policy explicit.

        Args:
            None.

        Returns:
            dict: The a plain dictionary representation.
        """
        return {
            "index": self.index,
            "item": self.item,
            "ok": self.ok,
            "value": self.value,
            "error": str(self.error) if self.error is not None else None,
            "elapsed_ms": self.elapsed_ms,
        }


def _task_kwargs(index: int, item: Any) -> Dict[str, Any]:
    if not isinstance(item, Mapping):
        raise TypeError(f"parallel item at index {index} must be a mapping of keyword arguments, got {type(item).__name__}")
    return dict(item)


def _run_task(fn: Callable[..., Any], index: int, item: Any) -> TaskResult:
    start = time.perf_counter()
    try:
        value = fn(**_task_kwargs(index, item))
        return TaskResult(index=index, item=item, ok=True, value=value, elapsed_ms=_elapsed_ms(start))
    except Exception as exc:
        return TaskResult(index=index, item=item, ok=False, error=exc, elapsed_ms=_elapsed_ms(start))


def _async_callable(fn: Callable[..., Any]) -> bool:
    return inspect.iscoroutinefunction(fn) or inspect.iscoroutinefunction(getattr(fn, "__call__", None))


async def _arun_task(fn: Callable[..., Any], index: int, item: Any) -> TaskResult:
    start = time.perf_counter()
    try:
        kwargs = _task_kwargs(index, item)
        if _async_callable(fn):
            value = await fn(**kwargs)
        else:
            loop = asyncio.get_running_loop()
            value = await loop.run_in_executor(None, partial(fn, **kwargs))
            if inspect.isawaitable(value):
                value = await value
        return TaskResult(index=index, item=item, ok=True, value=value, elapsed_ms=_elapsed_ms(start))
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        return TaskResult(index=index, item=item, ok=False, error=exc, elapsed_ms=_elapsed_ms(start))


def _elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000


def _indexed_source(items: Iterable[Any], need_total: bool) -> Tuple[Iterator[Tuple[int, Any]], int]:
    if hasattr(items, "__len__"):
        return enumerate(items), len(items)  # type: ignore[arg-type]
    if need_total:
        indexed = list(enumerate(items))
        return iter(indexed), len(indexed)
    return enumerate(items), 0


def _notify(done: int, total: int, hook: Optional[Callable[[int, int], None]]) -> None:
    if hook is not None:
        hook(done, total)


def _keep(result: TaskResult, on_error: ErrorPolicy) -> bool:
    if on_error == "raise" and not result.ok:
        result.raise_if_error()
    return on_error != "skip" or result.ok


def _submit(pool: ThreadPoolExecutor, futures: Dict[Future, int], source: Iterator[Tuple[int, Any]], fn: Callable[..., Any]) -> bool:
    try:
        idx, item = next(source)
    except StopIteration:
        return False
    futures[pool.submit(_run_task, fn, idx, item)] = idx
    return True


def _drain_ordered(buffer: Dict[int, TaskResult], next_idx: int, on_error: ErrorPolicy) -> Tuple[List[TaskResult], int]:
    out = []
    while next_idx in buffer:
        result = buffer.pop(next_idx)
        next_idx += 1
        if _keep(result, on_error):
            out.append(result)
    return out, next_idx


def _sync_results(
    fn: Callable[..., Any],
    items: Iterable[Mapping[str, Any]],
    *,
    max_workers: Optional[int],
    ordered: bool,
    on_error: ErrorPolicy,
    on_progress: Optional[Callable[[int, int], None]],
) -> Iterator[TaskResult]:
    source, total = _indexed_source(items, need_total=on_progress is not None)
    resolved = _resolve_max_workers(max_workers)

    if resolved == 1:
        done = 0
        for idx, item in source:
            result = _run_task(fn, idx, item)
            done += 1
            _notify(done, total, on_progress)
            if _keep(result, on_error):
                yield result
        return

    pool = ThreadPoolExecutor(max_workers=resolved)
    futures: Dict[Future, int] = {}
    buffer: Dict[int, TaskResult] = {}
    next_idx = 0
    done_count = 0
    try:
        for _ in range(resolved):
            if not _submit(pool, futures, source, fn):
                break
        while futures:
            done, _ = wait(set(futures), return_when=FIRST_COMPLETED)
            for future in done:
                futures.pop(future, None)
                result = future.result()
                done_count += 1
                _notify(done_count, total, on_progress)
                _keep(result, on_error)
                _submit(pool, futures, source, fn)
                if ordered:
                    buffer[result.index] = result
                    yielded, next_idx = _drain_ordered(buffer, next_idx, on_error)
                    for item in yielded:
                        yield item
                elif on_error != "skip" or result.ok:
                    yield result
    finally:
        for future in futures:
            future.cancel()
        pool.shutdown(wait=True, cancel_futures=True)


def batch(
    fn: Callable[..., Any],
    items: Iterable[Mapping[str, Any]],
    *,
    max_workers: Optional[int] = None,
    ordered: bool = True,
    on_error: str = "capture",
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> List[TaskResult]:
    """
    Run a callable over items with bounded thread concurrency and return task records.

    Use this helper for bounded local concurrency while keeping result ordering and error policy explicit.
    Each item must be a mapping and is passed as ``fn(**item)``.

    Args:
        fn (Callable[..., Any]): Callable executed once per input item as ``fn(**item)``.
        items (Iterable[Mapping[str, Any]]): Iterable of keyword-argument mappings. Iterators are consumed lazily when progress totals are not requested.
        max_workers (Optional[int]): Maximum concurrent HeavenBase tasks. ``None`` reads ``heavenbase.parallel.max_workers`` and then uses CPU-based defaults. Defaults to ``None``.
        ordered (bool): Whether output order follows input order. Defaults to ``True`` for batch APIs and ``False`` for stream APIs.
        on_error (str): Task error policy. Allowed values are ``"capture"``, ``"raise"``, and ``"skip"``. Defaults to ``'capture'``.
        on_progress (Optional[Callable[[int, int], None]]): Optional callback called as ``on_progress(done, total)`` after each task finishes. Defaults to ``None``.

    Returns:
        List[TaskResult]: List of ``TaskResult`` records, filtered when ``on_error="skip"``.

    Examples:
        >>> from heavenbase.utils.parallel import batch
        >>> [r.value for r in batch(lambda x: x * 2, [{"x": 1}, {"x": 2}], max_workers=1)]
        [2, 4]
    """
    return list(_sync_results(fn, items, max_workers=max_workers, ordered=ordered, on_error=_validate_on_error(on_error), on_progress=on_progress))


def batch_stream(
    fn: Callable[..., Any],
    items: Iterable[Mapping[str, Any]],
    *,
    max_workers: Optional[int] = None,
    ordered: bool = False,
    on_error: str = "capture",
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> Iterator[TaskResult]:
    """Stream task records from bounded thread work as soon as output is available.

    Use this helper for bounded local concurrency while keeping result ordering and error policy explicit.
    Each item must be a mapping and is passed as ``fn(**item)``.

    Args:
        fn (Callable[..., Any]): Callable executed once per input item as ``fn(**item)``.
        items (Iterable[Mapping[str, Any]]): Iterable of keyword-argument mappings. Iterators are consumed lazily when progress totals are not requested.
        max_workers (Optional[int]): Maximum concurrent HeavenBase tasks. ``None`` reads ``heavenbase.parallel.max_workers`` and then uses CPU-based defaults. Defaults to ``None``.
        ordered (bool): Whether output order follows input order. Defaults to ``True`` for batch APIs and ``False`` for stream APIs.
        on_error (str): Task error policy. Allowed values are ``"capture"``, ``"raise"``, and ``"skip"``. Defaults to ``'capture'``.
        on_progress (Optional[Callable[[int, int], None]]): Optional callback called as ``on_progress(done, total)`` after each task finishes. Defaults to ``None``.

    Yields:
        Iterator[TaskResult]: Iterator of ``TaskResult`` records.
    """
    yield from _sync_results(fn, items, max_workers=max_workers, ordered=ordered, on_error=_validate_on_error(on_error), on_progress=on_progress)


def _submit_async(tasks: Dict[asyncio.Task, int], source: Iterator[Tuple[int, Any]], fn: Callable[..., Any]) -> bool:
    try:
        idx, item = next(source)
    except StopIteration:
        return False
    tasks[asyncio.create_task(_arun_task(fn, idx, item))] = idx
    return True


async def _cancel_tasks(tasks: Dict[asyncio.Task, int]) -> None:
    for task in tasks:
        task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


async def _async_results(
    fn: Callable[..., Any],
    items: Iterable[Mapping[str, Any]],
    *,
    max_workers: Optional[int],
    ordered: bool,
    on_error: ErrorPolicy,
    on_progress: Optional[Callable[[int, int], None]],
) -> AsyncGenerator[TaskResult, None]:
    source, total = _indexed_source(items, need_total=on_progress is not None)
    resolved = _resolve_max_workers(max_workers)

    if resolved == 1:
        done = 0
        for idx, item in source:
            result = await _arun_task(fn, idx, item)
            done += 1
            _notify(done, total, on_progress)
            if _keep(result, on_error):
                yield result
        return

    tasks: Dict[asyncio.Task, int] = {}
    buffer: Dict[int, TaskResult] = {}
    next_idx = 0
    done_count = 0
    try:
        for _ in range(resolved):
            if not _submit_async(tasks, source, fn):
                break
        while tasks:
            done, _ = await asyncio.wait(set(tasks), return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                tasks.pop(task, None)
                result = task.result()
                done_count += 1
                _notify(done_count, total, on_progress)
                _keep(result, on_error)
                _submit_async(tasks, source, fn)
                if ordered:
                    buffer[result.index] = result
                    yielded, next_idx = _drain_ordered(buffer, next_idx, on_error)
                    for item in yielded:
                        yield item
                elif on_error != "skip" or result.ok:
                    yield result
    finally:
        await _cancel_tasks(tasks)


async def abatch(
    fn: Callable[..., Any],
    items: Iterable[Mapping[str, Any]],
    *,
    max_workers: Optional[int] = None,
    ordered: bool = True,
    on_error: str = "capture",
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> List[TaskResult]:
    """
    Run async or sync work with bounded async concurrency and return task records.

    Use this helper for bounded local concurrency while keeping result ordering and error policy explicit.
    Each item must be a mapping and is passed as ``fn(**item)``.

    Args:
        fn (Callable[..., Any]): Callable executed once per input item as ``fn(**item)``.
        items (Iterable[Mapping[str, Any]]): Iterable of keyword-argument mappings. Iterators are consumed lazily when progress totals are not requested.
        max_workers (Optional[int]): Maximum concurrent HeavenBase tasks. ``None`` reads ``heavenbase.parallel.max_workers`` and then uses CPU-based defaults. Defaults to ``None``.
        ordered (bool): Whether output order follows input order. Defaults to ``True`` for batch APIs and ``False`` for stream APIs.
        on_error (str): Task error policy. Allowed values are ``"capture"``, ``"raise"``, and ``"skip"``. Defaults to ``'capture'``.
        on_progress (Optional[Callable[[int, int], None]]): Optional callback called as ``on_progress(done, total)`` after each task finishes. Defaults to ``None``.

    Returns:
        List[TaskResult]: List of ``TaskResult`` records, filtered when ``on_error="skip"``.
    """
    return [
        result
        async for result in _async_results(fn, items, max_workers=max_workers, ordered=ordered, on_error=_validate_on_error(on_error), on_progress=on_progress)
    ]


async def abatch_stream(
    fn: Callable[..., Any],
    items: Iterable[Mapping[str, Any]],
    *,
    max_workers: Optional[int] = None,
    ordered: bool = False,
    on_error: str = "capture",
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> AsyncGenerator[TaskResult, None]:
    """Stream task records from bounded async work as soon as output is available.

    Use this helper for bounded local concurrency while keeping result ordering and error policy explicit.
    Each item must be a mapping and is passed as ``fn(**item)``.

    Args:
        fn (Callable[..., Any]): Callable executed once per input item as ``fn(**item)``.
        items (Iterable[Mapping[str, Any]]): Iterable of keyword-argument mappings. Iterators are consumed lazily when progress totals are not requested.
        max_workers (Optional[int]): Maximum concurrent HeavenBase tasks. ``None`` reads ``heavenbase.parallel.max_workers`` and then uses CPU-based defaults. Defaults to ``None``.
        ordered (bool): Whether output order follows input order. Defaults to ``True`` for batch APIs and ``False`` for stream APIs.
        on_error (str): Task error policy. Allowed values are ``"capture"``, ``"raise"``, and ``"skip"``. Defaults to ``'capture'``.
        on_progress (Optional[Callable[[int, int], None]]): Optional callback called as ``on_progress(done, total)`` after each task finishes. Defaults to ``None``.

    Yields:
        AsyncGenerator[TaskResult, None]: Async generator of ``TaskResult`` records.
    """
    async for result in _async_results(fn, items, max_workers=max_workers, ordered=ordered, on_error=_validate_on_error(on_error), on_progress=on_progress):
        yield result


def pmap(
    fn: Callable[..., Any],
    items: Iterable[Mapping[str, Any]],
    max_workers: Optional[int] = None,
) -> List[Any]:
    """
    Map ``fn`` over ``items`` with bounded threads while preserving order.

    Use this helper for bounded local concurrency while keeping result ordering and error policy explicit.

    Args:
        fn (Callable[..., Any]): Callable executed once per input item as ``fn(**item)``.
        items (Iterable[Mapping[str, Any]]): Iterable of keyword-argument mappings. Iterators are consumed lazily when progress totals are not requested.
        max_workers (Optional[int]): Maximum concurrent HeavenBase tasks. ``None`` reads ``heavenbase.parallel.max_workers`` and then uses CPU-based defaults. Defaults to ``None``.

    Returns:
        List[Any]: List of values in input order.

    Examples:
        >>> from heavenbase.utils.parallel import pmap
        >>> pmap(lambda x: x * 2, [{"x": 3}, {"x": 1}], max_workers=1)
        [6, 2]
    """
    return [r.value for r in batch(fn, items, max_workers=max_workers, ordered=True, on_error="raise")]


def pforeach(
    fn: Callable[..., Any],
    items: Iterable[Mapping[str, Any]],
    max_workers: Optional[int] = None,
) -> None:
    """
    Run a side-effect ``fn`` over ``items`` with bounded threads.

    Use this helper for bounded local concurrency while keeping result ordering and error policy explicit.

    Args:
        fn (Callable[..., Any]): Callable executed once per input item as ``fn(**item)``.
        items (Iterable[Mapping[str, Any]]): Iterable of keyword-argument mappings. Iterators are consumed lazily when progress totals are not requested.
        max_workers (Optional[int]): Maximum concurrent HeavenBase tasks. ``None`` reads ``heavenbase.parallel.max_workers`` and then uses CPU-based defaults. Defaults to ``None``.

    Returns:
        None: ``None`` after all side-effect calls complete.
    """
    batch(fn, items, max_workers=max_workers, ordered=False, on_error="raise")
