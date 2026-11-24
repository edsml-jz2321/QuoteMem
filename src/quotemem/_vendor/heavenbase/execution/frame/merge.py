"""Identity-bearing execution-frame merge helpers."""

from __future__ import annotations

__all__ = ["difference", "intersect", "union"]

from heavenbase.utils import Any, Dict, Sequence

from .result import ResultFrame


def intersect(frames: Sequence[ResultFrame]) -> ResultFrame:
    """Keep identities present in every frame and combine their row values.

    The first frame controls row order. Later non-null ordinary values replace
    earlier values, while list-valued metadata is concatenated in frame order.

    Args:
        frames (Sequence[ResultFrame]): Frames to intersect. The first frame
            controls output row order.

    Returns:
        ResultFrame: Rows whose identity value appears in every input frame.

    Raises:
        ValueError: If an input has no identity column or identity column names
            differ between frames.
    """
    if not frames:
        return ResultFrame.empty()
    identity_column = _identity_column(frames)
    ids = set(frames[0].ids)
    for frame in frames[1:]:
        ids &= set(frame.ids)
    metadata_columns = set().union(*(frame.metadata_columns for frame in frames))
    maps = [{row.get(identity_column): row for row in frame.rows()} for frame in frames]
    rows = []
    for object_id in frames[0].ids:
        if object_id not in ids:
            continue
        row: Dict[str, Any] = {}
        for row_map in maps:
            for key, value in row_map[object_id].items():
                row[key] = _merge_value(key, row.get(key), value, metadata_columns) if key in row else value
        rows.append(row)
    return ResultFrame.from_rows(rows, metadata_columns=metadata_columns, identity_column=identity_column)


def union(frames: Sequence[ResultFrame]) -> ResultFrame:
    """Keep identities present in any frame and combine their row values.

    Frames control first-seen row order and value precedence. Later non-null
    ordinary values replace earlier values, while list-valued metadata is
    concatenated in frame order.

    Args:
        frames (Sequence[ResultFrame]): Frames to union in precedence order.

    Returns:
        ResultFrame: Rows from every frame, merged by identity value.

    Raises:
        ValueError: If an input has no identity column or identity column names
            differ between frames.
    """
    identity_column = _identity_column(frames) if frames else "object_id"
    rows: Dict[Any, Dict[str, Any]] = {}
    metadata_columns = set().union(*(frame.metadata_columns for frame in frames))
    for frame in frames:
        for row in frame.rows():
            current = rows.setdefault(row.get(identity_column), {})
            for key, value in row.items():
                current[key] = _merge_value(key, current.get(key), value, metadata_columns) if key in current else value
    return ResultFrame.from_rows(rows.values(), metadata_columns=metadata_columns, identity_column=identity_column)


def difference(left: ResultFrame, right: ResultFrame) -> ResultFrame:
    """Keep left-frame identities absent from the right frame.

    Rows, metadata, and order come from ``left`` without value merging.

    Args:
        left (ResultFrame): Frame that provides candidate rows.
        right (ResultFrame): Frame whose identity values should be removed.

    Returns:
        ResultFrame: Filtered rows from `left`.

    Raises:
        ValueError: If either frame has no identity column or the identity
            column names differ.
    """
    _identity_column((left, right))
    return left.filter_ids(set(left.ids) - set(right.ids))


def _identity_column(frames: Sequence[ResultFrame]) -> str:
    identity = frames[0].require_identity()
    for frame in frames[1:]:
        current = frame.require_identity()
        if current != identity:
            raise ValueError(f"Cannot merge frames with different identity columns: {identity!r} != {current!r}")
    return identity


def _merge_value(key: str, left: Any, right: Any, metadata_columns: set[str]) -> Any:
    if key in metadata_columns and isinstance(left, list) and isinstance(right, list):
        return [*left, *right]
    if right is not None or left is None:
        return right
    return left
