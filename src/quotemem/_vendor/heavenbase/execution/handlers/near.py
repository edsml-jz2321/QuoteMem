"""Shared portable vector-near handler compiler."""

from __future__ import annotations

__all__ = ["missing_vector_score", "near_handler"]

from math import isfinite

from heavenbase.types import cosine_score, euclidean_score, inner_product_score
from heavenbase.types.vector import _COSINE_SAFE_NORMS, _scale_cosine_query
from heavenbase.utils import Any, Dict

from ..contracts import HandlerContext, QueryFragment

_SCORERS = {
    "cosine": cosine_score,
    "l2": euclidean_score,
    "euclidean": euclidean_score,
    "ip": inner_product_score,
    "inner": inner_product_score,
    "inner_product": inner_product_score,
    "dot": inner_product_score,
}


def missing_vector_score(metric: str) -> float | None:
    """Return the portable scan score for a row whose vector field is empty.

    Cosine keeps the historical neutral ``0.0``. Euclidean scores are negative
    distances and inner products are unbounded, so a constant would rank an
    empty row above real neighbours; those metrics drop the row instead.

    Args:
        metric (str): Requested near metric, in any case.

    Returns:
        float | None: ``0.0`` for cosine, otherwise ``None`` (the scan excludes the row).
    """
    return 0.0 if str(metric).lower() == "cosine" else None


class _PreparedVectors:
    """Own immutable binary64 coordinates and reusable magnitudes for one generation.

    The coordinates belong to the exact row objects they were derived from. A
    scan may serve rows from another generation (a concurrent reader replaced
    the retained one, or it was evicted), so scoring uses these coordinates only
    for rows that are those same objects.
    """

    def __init__(self, rows: list[dict], field: str) -> None:
        import numpy as np

        present = [row for row in rows if row.get(field) is not None]
        self.rows = present
        self.positions = {row["object_id"]: index for index, row in enumerate(present)}
        self.values = np.asarray([row[field] for row in present], dtype=np.float64) if present else np.empty((0, 0), dtype=np.float64)
        if self.values.ndim != 2 or not np.isfinite(self.values).all():
            raise ValueError("Exact near requires finite, consistently dimensioned vectors")
        with np.errstate(over="ignore", invalid="ignore"):
            self.norms = _row_norms(self.values)
        self.values.flags.writeable = False
        self.norms.flags.writeable = False


def _row_norms(delta: Any) -> Any:
    """Return each row's Euclidean norm in binary64 without losing range or precision.

    The sum of squares is fast and accurate while the norm stays inside roughly
    ``1e-150 .. 1e150``. Outside it the squares overflow or fall below the normal
    range, so only those rows are recomputed from their unit-scaled coordinates.
    A row whose norm exceeds binary64 gets ``inf``. Callers suppress floating-point
    warnings and handle a non-finite norm themselves. Cosine scoring (``_cosine_values``,
    also from the norms ``_PreparedVectors`` keeps) sends such a row to its unit-scaled
    path, so the row is scored by its direction. L2 scoring rejects the batch through
    ``_batch_scores``' final finiteness check.
    """
    import numpy as np

    norms = np.sqrt(np.sum(delta * delta, axis=1))
    rescale = ~(norms < 1e150) | (norms < 1e-150)
    if rescale.any():
        unit, shift = _unit_rows(delta[rescale])
        norms[rescale] = np.ldexp(np.sqrt(np.sum(unit * unit, axis=1)), shift)
    return norms


def _unit_rows(rows: Any) -> tuple[Any, Any]:
    """Scale each row by a power of two so its largest coordinate magnitude is in ``[0.5, 1)``.

    This is the batch form of ``_scale_cosine_query``: the scaling is exact for every
    coordinate that stays a normal number, and a zero row stays zero. Returns the scaled
    rows and each row's exponent, so ``ldexp(value, shift)`` restores a row's magnitude.
    """
    import numpy as np

    _, shift = np.frexp(np.max(np.abs(rows), axis=1, initial=0.0))
    return np.ldexp(rows, -shift[:, None]), shift


def _cosine_values(vectors: Any, stored_norms: Any, q: Any) -> Any:
    """Return the cosine of each row with the scaled query ``q``, by direction only.

    ``q`` comes from ``_scale_cosine_query``. A row whose norm is inside ``_COSINE_SAFE_NORMS``
    keeps every product inside binary64's normal range, so it is scored from its raw
    coordinates. Any other row (including a zero row) is scored from its unit-scaled
    coordinates, as ``cosine_score`` does, so its stored magnitude never enters a product:
    a huge row cannot overflow the dot product or the norm product, a subnormal row does
    not lose its digits, and a finite row whose norm exceeds binary64 still has a direction.
    """
    import numpy as np

    low, high = _COSINE_SAFE_NORMS
    q_norm = np.sqrt(np.sum(q * q))
    norms = stored_norms * q_norm
    values = np.divide(np.sum(vectors * q, axis=1), norms, out=np.zeros(len(vectors)), where=norms != 0)
    outside = ~((stored_norms >= low) & (stored_norms <= high))
    if outside.any():
        unit, _ = _unit_rows(vectors[outside])
        unit_norms = np.sqrt(np.sum(unit * unit, axis=1)) * q_norm
        values[outside] = np.divide(np.sum(unit * q, axis=1), unit_norms, out=np.zeros(len(unit)), where=unit_norms != 0)
    return values


def _prepared_positions(prepared: _PreparedVectors, rows: list[dict], present: list[int]) -> list[int] | None:
    """Map batch rows to prepared coordinates, or return ``None`` unless every row is from that generation."""
    positions = []
    for index in present:
        row = rows[index]
        position = prepared.positions.get(row.get("object_id"))
        if position is None or prepared.rows[position] is not row:
            return None
        positions.append(position)
    return positions


def _batch_scores(
    rows: list[dict], field: str, query: list[float], metric: str, scalar: Any, *, prepared: _PreparedVectors | None = None
) -> list[float | None]:
    """Score a bounded portable batch in binary64, using NumPy when installed."""
    try:
        import numpy as np
    except ImportError:
        return [scalar(row) for row in rows]
    present = [index for index, row in enumerate(rows) if row.get(field) is not None]
    scores: list[float | None] = [None] * len(rows)
    if not present:
        return scores
    positions = _prepared_positions(prepared, rows, present) if prepared is not None else None
    if positions is None:
        prepared = None
    vectors = prepared.values[positions] if prepared is not None else np.asarray([rows[index][field] for index in present], dtype=np.float64)
    q = np.asarray(_scale_cosine_query(query) if metric == "cosine" else query, dtype=np.float64)
    if vectors.ndim != 2 or vectors.shape[1] != len(q):
        raise ValueError("Exact near requires matching vector dimensions")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        if metric == "cosine":
            stored_norms = prepared.norms[positions] if prepared is not None else _row_norms(vectors)
            values = _cosine_values(vectors, stored_norms, q)
        elif metric in ("l2", "euclidean"):
            values = -_row_norms(vectors - q)
        else:
            values = np.sum(vectors * q, axis=1)
    # Overflow or invalid coordinates must not turn into a plausible finite score.
    if not np.isfinite(vectors).all() or not np.isfinite(values).all():
        raise ValueError("Exact near produced a non-finite score")
    for index, value in zip(present, values):
        scores[index] = float(value)
    return scores


def near_handler(field_schema: Any, value: Any, ctx: HandlerContext) -> QueryFragment:
    """Compile a vector-near request to the portable scan fallback.

    Args:
        field_schema (Any): Routed vector field schema.
        value (Any): Query vector validated against the field dtype.
        ctx (HandlerContext): Selected backend and strategy context.

    Returns:
        QueryFragment: Python scan fragment scoring rows with the configured metric.
    """
    query = field_schema.dtype.validate(value)
    metric = str(ctx.metric).lower()
    if metric not in _SCORERS:
        raise ValueError(f"Portable near does not support metric {ctx.metric!r}")
    scorer = _SCORERS[metric]
    if ctx.exact and not all(isfinite(item) for item in query):
        raise ValueError("Exact near requires a finite query vector")
    if ctx.exact and metric == "cosine":
        query = _scale_cosine_query(query)

    def _payload(row: dict) -> float | None:
        vec = row.get(field_schema.name)
        if vec is None:
            return None if ctx.exact else missing_vector_score(metric)
        score = scorer(vec, query)
        if ctx.exact and not isfinite(score):
            raise ValueError(f"Exact near produced a non-finite score for {row.get('object_id')!r}")
        return score

    reason = ctx.fallback_reason("near", type(field_schema.dtype).identifier)
    notes: Dict[str, Any] = {"mode": "py_scan", "top_k": ctx.top_k}
    if ctx.exact:
        notes["exact"] = True
        notes["batch_payload"] = lambda rows: _batch_scores(rows, field_schema.name, query, metric, _payload)
    if reason:
        notes["fallback_reason"] = reason
    return QueryFragment(ctx.backend, "near", field_schema.name, _payload, returns=["score"], notes=notes)
