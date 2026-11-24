"""Vector logical type and scoring helpers."""

from __future__ import annotations

__all__ = [
    "Vector",
    "cosine_score",
    "euclidean_score",
    "inner_product_score",
]

from math import dist, frexp, isfinite, ldexp

from heavenbase.utils import Any, Iterable, Optional

from ..modules import ModuleResolver
from .base import LogicalType


class Vector(LogicalType):
    """Dense numeric vector logical type with an optional dimension.

    Canonical form is a ``list[float]``. Array-like inputs with ``tolist()``
    are accepted. When ``dim`` is set, the list length must match exactly.

    Warnings:
        Non-numeric elements raise ``ValueError`` during float coercion.

    Examples:
        ``hb.Vector().validate([1, 0.5])`` returns ``[1.0, 0.5]``.
        ``hb.Vector(3).validate([1, 2, 3])`` returns ``[1.0, 2.0, 3.0]``.
    """

    identifier = "vector"

    @classmethod
    def from_dict(cls, spec: dict, *, resolver: ModuleResolver | None = None) -> "Vector":
        """Build a Vector logical type from a serialized descriptor.

        Args:
            spec (dict): Serialized descriptor with optional ``dim``,
                ``dimension``, or legacy ``value`` keys.
            resolver (ModuleResolver | None): Context-owned module resolver.
                The default Context is selected when omitted.

        Returns:
            Vector: Configured vector logical type.
        """
        dim = spec.get("dim", spec.get("dimension", spec.get("value")))
        return cls(int(dim) if dim is not None else None)

    def __init__(self, dim: Optional[int] = None):
        """Create a Vector logical type with optional fixed dimension.

        Args:
            dim (int | None): Required list length when set. Must be positive.

        Returns:
            None: This constructor does not return a value.
        """
        if dim is not None:
            dim = int(dim)
            if dim <= 0:
                raise ValueError(f"Vector dimension must be positive, got {dim}")
        self.dim = dim

    def __class_getitem__(cls, dim: int) -> "Vector":
        return cls(dim)

    def validate(self, value: Any) -> list:
        """Validate and coerce a vector to a list of floats.

        Args:
            value: List, tuple, or array-like object with ``tolist()``.

        Returns:
            Normalised ``list[float]`` matching ``dim`` when configured.

        Raises:
            ValueError: If ``value`` is not list-like, contains non-numeric
                elements, or has the wrong length.
        """
        if hasattr(value, "tolist"):
            value = value.tolist()
        if not isinstance(value, (list, tuple)):
            raise ValueError("Vector expects a list or tuple")
        vec = [float(v) for v in value]
        if self.dim is not None and len(vec) != self.dim:
            raise ValueError(f"Vector[{self.dim}] received dimension {len(vec)}")
        return vec

    def to_dict(self) -> dict:
        """Return a JSON-compatible Vector descriptor.

        Args:
            None.

        Returns:
            dict: Descriptor containing ``type`` and optional ``dim``.
        """
        return {"type": type(self).identifier, "dim": self.dim}

    def clone(self) -> "Vector":
        """Return a new Vector with the same dimension.

        Args:
            None.

        Returns:
            Vector: Independent vector logical type with the same dimension.
        """
        return Vector(self.dim)


# Norms in this band keep every square and cross product of two vectors inside binary64's range.
_COSINE_SAFE_NORMS = (2.0**-400, 2.0**400)


def _scale_cosine_query(query: list[float]) -> list[float]:
    """Scale a finite cosine operand by a power of two so its largest coordinate magnitude is in ``[0.5, 1)``.

    Cosine ignores positive scale, and multiplying by a power of two is exact for every coordinate
    that stays a normal binary64 number, so the scores do not change. A finite query of any
    magnitude then squares and sums inside float32's range in a vector store that keeps float32
    coordinates, and inside binary64's range in a scan; unscaled, it could become infinities or
    zeros there. A zero query, or one whose largest magnitude is not finite, is returned unchanged.

    Args:
        query (list[float]): Validated query vector.

    Returns:
        list[float]: A positive power-of-two multiple of the query, or the query itself.
    """
    scale = max(map(abs, query), default=0.0)
    shift = frexp(scale)[1] if scale and isfinite(scale) else 0
    return [ldexp(value, -shift) for value in query] if shift else query


def _cosine_terms(a: list[float], b: list[float]) -> tuple[float, float, float]:
    return sum(x * y for x, y in zip(a, b)), sum(x * x for x in a) ** 0.5, sum(y * y for y in b) ** 0.5


def cosine_score(left: Iterable[float], right: Iterable[float]) -> float:
    """Return a dependency-free cosine score for the in-memory vector handler.

    A finite operand whose squares would leave binary64's range is scored after an exact
    power-of-two scaling, so an extreme magnitude does not turn the score into ``0.0`` or NaN.

    Args:
        left (Iterable[float]): First vector operand.
        right (Iterable[float]): Second vector operand.

    Returns:
        float: Cosine similarity in ``[-1, 1]``, or ``0.0`` when either vector
        has zero magnitude.

    Raises:
        ValueError: If the operands have different lengths.
    """
    a = [float(v) for v in left]
    b = [float(v) for v in right]
    if len(a) != len(b):
        raise ValueError(f"Vector dimension mismatch: {len(a)} != {len(b)}")
    dot, na, nb = _cosine_terms(a, b)
    low, high = _COSINE_SAFE_NORMS
    if not (low <= na <= high and low <= nb <= high):
        dot, na, nb = _cosine_terms(_scale_cosine_query(a), _scale_cosine_query(b))
    return dot / (na * nb) if na and nb else 0.0


def euclidean_score(left: Iterable[float], right: Iterable[float]) -> float:
    """Return negative Euclidean distance for L2 KNN ordering.

    The scan-fallback companion to ``cosine_score`` for an L2-metric vector index. The score
    scale is ordering-only, but the ranking is monotonic in distance, so a Python-scan near
    returns the same top-k order as a native L2 KNN search.

    Args:
        left (Iterable[float]): First vector operand.
        right (Iterable[float]): Second vector operand.

    Returns:
        float: Negative L2 distance; higher values indicate closer vectors.

    Raises:
        ValueError: If the operands have different lengths.
    """
    a = [float(v) for v in left]
    b = [float(v) for v in right]
    if len(a) != len(b):
        raise ValueError(f"Vector dimension mismatch: {len(a)} != {len(b)}")
    return -dist(a, b)


def inner_product_score(left: Iterable[float], right: Iterable[float]) -> float:
    """Return inner product score for IP KNN ordering.

    The scan-fallback companion to ``cosine_score`` for an inner-product vector index. The
    ranking is monotonic in the inner product, so a Python-scan near returns the same top-k
    order as a native IP KNN search.

    Args:
        left (Iterable[float]): First vector operand.
        right (Iterable[float]): Second vector operand.

    Returns:
        float: Dot product; higher values indicate closer vectors under IP KNN.

    Raises:
        ValueError: If the operands have different lengths.
    """
    a = [float(v) for v in left]
    b = [float(v) for v in right]
    if len(a) != len(b):
        raise ValueError(f"Vector dimension mismatch: {len(a)} != {len(b)}")
    return sum(x * y for x, y in zip(a, b))
