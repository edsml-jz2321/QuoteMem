"""Neutral aggregation and row-filter transforms for result frames."""

from __future__ import annotations

__all__ = ["aggregate_frame", "filter_frame"]

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from ...utils.serialize import dumps_json
from .result import ResultFrame

_FrameReducer = Callable[[list[Any]], Any]
_FrameMetric = tuple[str, _FrameReducer]


def aggregate_frame(
    frame: ResultFrame,
    *,
    group_by: Sequence[str],
    metrics: Mapping[str, _FrameMetric],
) -> ResultFrame:
    """Fold a complete frame into an identityless aggregate table.

    The caller owns aggregate syntax, reducer selection, option validation, and
    schema binding. This transform owns only deterministic frame grouping and
    reducer invocation. `None` metric values are omitted, input with an identity
    is folded in ascending identity order, and ungrouped empty input still emits
    one metric row.

    Args:
        frame (ResultFrame): Complete materialized rows.
        group_by (Sequence[str]): Ordered result columns used as group keys.
        metrics (Mapping[str, tuple[str, Callable[[list[Any]], Any]]]): Output
            name mapped to a source
            column and a reducer callable over non-null values.

    Returns:
        ResultFrame: Identityless group columns followed by metric columns.
    """
    if frame.has_identity:
        frame = frame.sort(frame.require_identity())
    columns = [*group_by, *metrics]
    if not group_by:
        row = {output: fold([value for value in _column(frame, source) if value is not None]) for output, (source, fold) in metrics.items()}
        return ResultFrame.from_table([row], cols=columns)

    groups: dict[tuple[Any, ...], list[int]] = {}
    group_rows: dict[tuple[Any, ...], tuple[Any, ...]] = {}
    group_columns = {name: _column(frame, name) for name in group_by}
    for index in range(len(frame)):
        raw = tuple(group_columns[name][index] for name in group_by)
        key = tuple(_group_key(value) for value in raw)
        groups.setdefault(key, []).append(index)
        group_rows.setdefault(key, raw)

    metric_columns = {output: _column(frame, source) for output, (source, _) in metrics.items()}
    rows: list[dict[str, Any]] = []
    for key, indices in groups.items():
        row = dict(zip(group_by, group_rows[key]))
        for output, (_, fold) in metrics.items():
            values = [metric_columns[output][index] for index in indices if metric_columns[output][index] is not None]
            row[output] = fold(values)
        rows.append(row)
    return ResultFrame.from_table(rows, cols=columns)


def filter_frame(
    frame: ResultFrame,
    match: Callable[[Mapping[str, Any]], bool],
) -> ResultFrame:
    """Filter a result frame with one caller-prepared row predicate.

    Args:
        frame (ResultFrame): Result table to filter.
        match (Callable[[Mapping[str, Any]], bool]): Predicate over one detached
            result row.

    Returns:
        ResultFrame: Rows accepted by `match`, preserving the input columns.
    """
    return ResultFrame.from_rows(
        [row for row in frame.rows() if match(row)],
        cols=frame.col_names(),
        metadata_columns=frame.metadata_columns,
        identity_column=frame.identity_column,
    )


def _column(frame: ResultFrame, name: str) -> list[Any]:
    """Return one source column, treating an absent optional route as nulls."""
    if name not in frame.col_names():
        return [None] * len(frame)
    return frame.column(name)


def _group_key(value: Any) -> Any:
    """Return a typed deterministic grouping key for one value."""
    if value is None:
        return ("null", None)
    if isinstance(value, bool):
        return ("boolean", value)
    if isinstance(value, (int, float)):
        return ("number", value)
    if isinstance(value, str):
        return ("string", value)
    return ("payload", type(value).__qualname__, dumps_json(value, sort_keys=True, compact=True))
