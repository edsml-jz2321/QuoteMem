"""Portable Python scan execution for backends that can enumerate rows."""

from __future__ import annotations

__all__ = ["execute_scan"]

from typing import TYPE_CHECKING
from heapq import nsmallest
from itertools import islice

from heavenbase.utils import Any, Callable, Iterable

from ..execution.contracts import LOGICAL_OPS, QueryFragment
from ..execution.frame.result import ResultFrame

if TYPE_CHECKING:
    from .base import Backend


def _warn_python_fallback(backend: Backend, fragment: QueryFragment) -> None:
    """Warn once when configured and this execution path uses Python scan fallback."""
    enabled = backend.config.get("heavenbase.query.warn_on_python_fallback", default=False)
    if isinstance(enabled, str):
        enabled = enabled.strip().lower() not in ("", "0", "false", "no", "off")
    if not enabled or not fragment.uses_scan():
        return
    import warnings

    warnings.warn(_python_fallback_warning(backend, fragment), UserWarning, stacklevel=4)


def _python_fallback_warning(backend: Backend, fragment: QueryFragment) -> str:
    """Return a user-facing warning for one Python-scan fragment."""
    reason = fragment.fallback_reason()
    detail = f" {reason}" if reason else " Use query.explain() to inspect handler_mode and fallback_reason."
    entity_id = fragment.entity_id()
    subject = f"{entity_id}.{fragment.field}" if entity_id and fragment.field != "*" else fragment.field
    return f"HeavenBase query on backend {backend.name!r} executed {fragment.op!r} for {subject!r} " f"via Python scan fallback.{detail}"


def _row_payload(backend: Backend, fragment: QueryFragment) -> Callable[[dict[str, Any]], Any]:
    payload = fragment.notes.get("row_payload", fragment.payload)
    if not callable(payload):
        raise TypeError(f"{type(backend).__name__} expects a callable row payload, got {type(payload).__name__}")
    return payload


def _metadata_columns(fragment: QueryFragment) -> set[str]:
    columns = {str(column) for column in fragment.notes.get("metadata_columns", ())}
    if fragment.op in ("and", "or"):
        for child in fragment.children():
            columns.update(_metadata_columns(child))
    return columns


def _merge_row_meta(left: dict[str, Any], right: dict[str, Any], metadata_columns: set[str]) -> dict[str, Any]:
    out = dict(left)
    for key, value in right.items():
        if key in metadata_columns and isinstance(out.get(key), list) and isinstance(value, list):
            out[key] = [*out[key], *value]
        elif value is not None or key not in out or out.get(key) is None:
            out[key] = value
    return out


def _row_match(backend: Backend, fragment: QueryFragment, row: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    if fragment.op == "and":
        meta: dict[str, Any] = {}
        metadata_columns = _metadata_columns(fragment)
        for child in fragment.children():
            matched, child_meta = _row_match(backend, child, row)
            if not matched:
                return False, {}
            meta = _merge_row_meta(meta, child_meta, metadata_columns)
        return True, meta
    if fragment.op == "or":
        meta = {}
        matched_any = False
        metadata_columns = _metadata_columns(fragment)
        for child in fragment.children():
            matched, child_meta = _row_match(backend, child, row)
            if matched:
                matched_any = True
                meta = _merge_row_meta(meta, child_meta, metadata_columns)
        return matched_any, meta
    if fragment.op == "not":
        matched, _meta = _row_match(backend, fragment.children()[0], row)
        return not matched, {}
    matched = bool(_row_payload(backend, fragment)(row))
    if not matched:
        return False, {}
    enricher = fragment.notes.get("row_enricher")
    return True, enricher(row) if callable(enricher) else {}


def _top_k(backend: Backend, fragment: QueryFragment) -> int:
    default_top_k = int(backend.config.get("heavenbase.query.near.default_top_k", default=50))
    return fragment.notes.get(
        "top_k",
        fragment.payload.get("top_k", default_top_k) if isinstance(fragment.payload, dict) else default_top_k,
    )


def execute_scan(backend: Backend, fragment: QueryFragment, *, rows: Iterable[dict[str, Any]] | None = None) -> ResultFrame:
    """Execute a compiled fragment by scanning rows with Python predicates.

    This is the portable default behind :meth:`Backend.execute`. Provider
    implementations may replace execution by overriding that method and call
    ``super().execute(fragment)`` when a fragment requires scan fallback.

    Args:
        backend: Backend instance whose ``scan`` method enumerates entity rows.
        fragment: Backend-ready fragment whose payload evaluates scanned rows.
        rows: Optional candidate rows. When omitted, rows are obtained from the
            backend's ``scan`` method.

    Returns:
        ResultFrame: Rows matched by portable Python execution.

    Raises:
        NotImplementedError: If rows are omitted and the backend does not
            implement ``scan``.
        TypeError: If the fragment does not carry a callable row payload.
    """
    _warn_python_fallback(backend, fragment)
    entity_id = fragment.entity_id()
    rows = rows if rows is not None else backend.scan(entity_id) if entity_id else []
    if fragment.op in LOGICAL_OPS:
        out = []
        for row in rows:
            matched, meta = _row_match(backend, fragment, row)
            if matched:
                out.append({**row, **meta})
        return ResultFrame.from_rows(out, metadata_columns=_metadata_columns(fragment))
    payload = _row_payload(backend, fragment)
    if fragment.op == "near":
        filt = fragment.filter_fragment()
        if fragment.notes.get("exact") is True:

            def candidates():
                admitted = (row for row in rows if filt is None or _row_match(backend, filt, row)[0])
                batch_payload = fragment.notes.get("batch_payload")
                batch_size = int(fragment.notes.get("batch_size", backend.config.get("heavenbase.query.near.scan_batch_size", default=256)))
                if batch_size <= 0:
                    raise ValueError("near scan_batch_size must be positive")
                while batch := list(islice(admitted, batch_size)):
                    scores = batch_payload(batch) if callable(batch_payload) else [payload(row) for row in batch]
                    for row, score in zip(batch, scores, strict=True):
                        if score is not None:
                            yield {**row, "score": score}

            selected = nsmallest(_top_k(backend, fragment), candidates(), key=lambda row: (-row["score"], str(row.get("object_id") or "")))
            if (fields := fragment.notes.get("result_fields")) is not None:
                selected = [{name: value for name, value in row.items() if name in fields} for row in selected]
            return ResultFrame.from_rows(selected)
        # A filter evaluated here also fills and declares its result metadata, such as the `match` of
        # `contained_in`, exactly as the filter alone would, whether or not any row matches.
        scored = []
        for row in rows:
            meta: dict[str, Any] = {}
            if filt is not None:
                matched, meta = _row_match(backend, filt, row)
                if not matched:
                    continue
            score = payload(row)
            if score is not None:
                out = {**row, **meta}
                out["score"] = score
                scored.append(out)
        scored.sort(key=lambda row: row.get("score", 0.0), reverse=True)
        return ResultFrame.from_rows(scored[: _top_k(backend, fragment)], metadata_columns=set() if filt is None else _metadata_columns(filt))
    out = []
    for row in rows:
        if payload(row):
            enricher = fragment.notes.get("row_enricher")
            out.append({**row, **(enricher(row) or {})} if callable(enricher) else row)
    return ResultFrame.from_rows(out, metadata_columns=_metadata_columns(fragment))
