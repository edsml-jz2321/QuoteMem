"""Neutral derived-column transforms for dotted JSON paths."""

from __future__ import annotations

__all__ = ["derive_json_path_columns"]

from collections.abc import Mapping
from typing import Any

from ...types.descriptor import coerce_type
from .result import ResultFrame


def derive_json_path_columns(
    schema: Any,
    frame: ResultFrame,
    paths: Mapping[str, tuple[str, list[str]]],
) -> ResultFrame:
    """Append resolved dotted-path columns to one materialized entity frame.

    Query syntax resolves each dotted name to a base field and path before this
    transform runs. Payloads decode through the base field's logical type; a
    missing or null path yields `None`.

    Args:
        schema (Any): Entity schema that owns each base JSON field.
        frame (ResultFrame): Complete materialized entity rows.
        paths (Mapping[str, tuple[str, list[str]]]): Derived column name mapped
            to its base field and nested key path.

    Returns:
        ResultFrame: Frame with derived columns appended in mapping order, or
        `frame` unchanged when no paths were requested.
    """
    if not paths:
        return frame
    rows = frame.rows()
    for name, (base, path) in paths.items():
        path_value = coerce_type(schema.get(base).dtype).path_value
        for row in rows:
            row[name] = path_value(row.get(base), path)
    cols = frame.col_names()
    cols += [name for name in paths if name not in cols]
    return ResultFrame.from_rows(rows, cols=cols, metadata_columns=frame.metadata_columns, identity_column=frame.identity_column)
