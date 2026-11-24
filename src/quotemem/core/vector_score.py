"""Dense vector scoring shared by nomination and Search.

One module owns the two scoring functions so that every dense comparison in
Add and Search uses the same function objects: a float64 unit vector with an
exactly summed norm, and a cosine rounded half-even to 6 decimals.
"""

from __future__ import annotations

import ctypes
import functools
import math
import operator
from array import array
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np

from quotemem.core.errors import QuoteMemError

UnitVector = Sequence[float]
"""A finite float64 embedding scaled to unit L2 norm."""


def unit_vector(vector: Sequence[float]) -> memoryview[float]:
    """Return a vector scaled to unit L2 norm in float64.

    Args:
        vector (Sequence[float]): A finite, non-zero embedding.

    Returns:
        memoryview: Immutable packed float64 values divided by the exactly
            summed norm. No Python float objects are retained per coordinate.

    Raises:
        QuoteMemError: ``binding_mismatch`` when the vector is zero or not finite; an
            active vector always has a direction.
    """
    values = _float_values(vector)
    if values is None:
        norm = math.sqrt(math.fsum(value * value for value in vector))
        if not norm or not math.isfinite(norm):
            raise QuoteMemError("binding_mismatch", "An embedding vector is zero or not finite")
        return memoryview(array("d", (value / norm for value in vector)).tobytes()).cast("d")
    # The same IEEE products, exact sum and divisions as the loop above, element for
    # element: float64 multiplication and division are correctly rounded either way.
    norm = math.sqrt(math.fsum((values * values).tolist()))
    if not norm or not math.isfinite(norm):
        raise QuoteMemError("binding_mismatch", "An embedding vector is zero or not finite")
    return memoryview((values / norm).tobytes()).cast("d")


def _float_values(vector: Sequence[float]) -> UnitArray | None:
    """The vector as a float64 array when every value is a float, else None (the loop path).

    Only Python floats (and packed float64 buffers) take the array path, so ints, bools
    and other number types keep the exact Python arithmetic of the original loop.
    """
    if isinstance(vector, memoryview):
        if vector.format == "d" and vector.ndim == 1:
            return np.frombuffer(vector.tobytes(), dtype=np.float64)
        return None
    if isinstance(vector, np.ndarray):
        return None
    if all(type(value) is float for value in vector):
        return np.fromiter(vector, dtype=np.float64, count=len(vector))
    return None


_FAST_MARGIN = 1e-12
"""Distance from a fast dot product within which the exact sum is certain to lie.

The exact score sums the float64 products with ``math.fsum``. ``math.sumprod`` adds
the exact products in extended precision; the two differ by at most a few units in
the last place of ``sum(abs(p * q))``, which is at most 1 for unit vectors. Even a
plain left-to-right float sum stays within ``n * 2**-53 * sum(abs(p * q))`` of it,
below this margin for every dimension up to ``_FAST_DIMENSIONS``.
"""

_REACH_MARGIN = 1e-6
"""Distance below a threshold from which a fast dot product's rounded cosine may still
reach it (`UnitRows.reaching`): rounding to 6 decimals raises a value by at most 5e-7,
and the exact sum lies within ``_FAST_MARGIN`` of the fast product."""

_FAST_DIMENSIONS = 8192
"""Longest vector whose score may come from the fast dot product."""


def _rounded(value: float) -> float:
    """Clip a dot product to [-1, 1] and round it half-even to 6 decimals, without -0.0."""
    return round(min(1.0, max(-1.0, value)), 6) + 0.0


def cosine(first: UnitVector, second: UnitVector) -> float:
    """Return the cosine of two unit vectors, rounded half-even to 6 decimals.

    The result is always that of the exactly summed float64 products. A fast dot
    product decides it whenever every value within ``_FAST_MARGIN`` of it rounds to
    the same 6 decimals: the rounding is monotone, so the exact sum, which lies in
    that interval, rounds alike. Only a dot product that close to a rounding
    boundary is summed exactly.

    Args:
        first (UnitVector): A unit vector from `unit_vector` or its packed memo.
        second (UnitVector): Another unit vector of the same space.

    Returns:
        float: The exactly summed dot product, clipped to [-1, 1] and rounded
            half-even to 6 decimals; ``-0.0`` becomes ``0.0``.

    Raises:
        QuoteMemError: ``embedding_space_mismatch`` when the dimensions differ.
    """
    if len(first) != len(second):
        raise QuoteMemError("embedding_space_mismatch", "Embedding vectors differ in dimension")
    if len(first) <= _FAST_DIMENSIONS:
        # Iterating packed buffers in sumprod boxes each coordinate again.
        # The contiguous float64 loop reads them directly, with the same
        # conservative rounding interval used by the batched product below.
        fast = (float(np.einsum("i,i->", np.asarray(first, dtype=np.float64),
                               np.asarray(second, dtype=np.float64)))
                if isinstance(first, memoryview) and isinstance(second, memoryview)
                else math.sumprod(first, second))
        if math.isfinite(fast):
            low = _rounded(fast - _FAST_MARGIN)
            if low == _rounded(fast + _FAST_MARGIN):
                return low
    return _rounded(math.fsum(map(operator.mul, first, second)))


UnitArray = np.ndarray[Any, np.dtype[np.float64]]
"""A unit vector's float64 values as one read-only array row (`unit_array`)."""


def unit_array(unit: UnitVector) -> UnitArray:
    """Return a unit vector's values as a read-only float64 array, to stack into `UnitRows`.

    Args:
        unit (UnitVector): A unit vector from `unit_vector` or its packed memo.

    Returns:
        numpy.ndarray: The same float64 values, not writeable.
    """
    packed = (
        isinstance(unit, memoryview) and isinstance(unit.obj, bytes)
        and unit.format == "d" and unit.ndim == 1 and unit.c_contiguous
    )
    array = (np.frombuffer(unit, dtype=np.float64) if packed and isinstance(unit, memoryview)
             else np.array(unit, dtype=np.float64))
    array.flags.writeable = False
    return array


class UnitRows:
    """Unit vectors stacked once so that one query scores against all of them.

    ``cosines(query)`` equals ``[cosine(query, row) for row in rows]`` exactly,
    raising what the first failing ``cosine`` would raise. Every dot product
    comes from one float64 matrix-vector product (single-threaded), which differs from the exact
    sum by at most ``n * 2**-53 * sum(abs(p * q))`` in any summation order
    (pairwise or vectorized included), below ``_FAST_MARGIN`` for every dimension up to
    ``_FAST_DIMENSIONS``; so whenever every value within that margin rounds
    alike the rounded value is the exact one, and only a product that close to
    a 6-decimal boundary is summed exactly, as `cosine` sums it. One Python
    call per row becomes one call per query: a nomination scores each member
    against every head of its user, so this grows with the user's history.
    """

    __slots__ = ("_matrix", "rows")

    def __init__(
        self, rows: Sequence[UnitVector], arrays: Sequence[UnitArray] | None = None,
    ) -> None:
        """Stack the rows, or their already converted ``arrays`` in the same order.

        Args:
            rows (Sequence[UnitVector]): The unit vectors.
            arrays (Sequence[numpy.ndarray] | None): ``unit_array`` of each row,
                when the caller keeps them; None converts the rows here.
        """
        self.rows = tuple(rows)
        if arrays is not None and len(arrays) != len(self.rows):
            raise ValueError("Each row needs its array")
        self._matrix: UnitArray | None = None
        dimensions = {len(row) for row in self.rows}
        if len(dimensions) == 1 and next(iter(dimensions)) <= _FAST_DIMENSIONS:
            self._matrix = (np.array(self.rows, dtype=np.float64) if arrays is None
                            else np.stack(arrays))

    def __len__(self) -> int:
        return len(self.rows)

    def cosines(self, query: UnitVector) -> list[float]:
        """Return the cosine of ``query`` with every row, in row order (`cosine`'s values).

        Args:
            query (UnitVector): A unit vector of the rows' space.

        Returns:
            list[float]: One rounded cosine per row.

        Raises:
            QuoteMemError: ``embedding_space_mismatch`` when a row's dimension differs.
        """
        matrix = self._matrix
        if matrix is None or matrix.shape[1] != len(query):
            return [cosine(query, row) for row in self.rows]
        # einsum's own loop, not BLAS: a BLAS product starts its thread pool,
        # whose spinning threads multiplied a service's CPU time by three.
        return self._finish(np.einsum("ij,j->i", matrix, np.asarray(query, dtype=np.float64)),
                            query)

    def cosines_many(self, queries: Sequence[UnitVector]) -> list[list[float]]:
        """Return ``[self.cosines(query) for query in queries]``, reading the rows once.

        A review round scores every member against every head of the user:
        one matrix-vector product per member streamed the whole head matrix
        from memory once per member, tens of megabytes each for a long user.
        Here one matrix product scores every query. It runs on OpenBLAS with
        that library's thread count set to one for the calling thread only
        (``_blas_threads``), so no BLAS thread pool is started or kept
        spinning, which the per-query products avoid BLAS for (``cosines``);
        without that control it is einsum's own single-threaded loop. Each
        product differs from ``cosines``'s only in summation order, inside the
        margin that both settle against, so every rounded score is the same
        exact value.

        Args:
            queries (Sequence[UnitVector]): Unit vectors of the rows' space.

        Returns:
            list[list[float]]: One ``cosines`` list per query, in query order.

        Raises:
            QuoteMemError: ``embedding_space_mismatch`` when a row's dimension differs.
        """
        if len(queries) < 2 or (products := self._products(queries)) is None:
            return [self.cosines(query) for query in queries]
        return [self._finish(row, query) for row, query in zip(products, queries, strict=True)]

    def reaching(
        self, queries: Sequence[UnitVector], threshold: float,
    ) -> list[dict[int, float]]:
        """Return, for each query, every row whose cosine reaches ``threshold``, by row index.

        Equal to ``{index: score for index, score in enumerate(self.cosines(query))
        if score >= threshold}`` for every query. One matrix product, as in
        `cosines_many`, screens every pair, and only a product within
        ``_REACH_MARGIN`` below ``threshold`` or above is rounded (`cosine`'s check,
        summed exactly near a boundary): the rounded score of a product further below
        is below ``threshold``. A listing of a long user compares every cued text with
        every state memory, and almost none of those pairs reach the threshold.

        Args:
            queries (Sequence[UnitVector]): Unit vectors of the rows' space.
            threshold (float): The least cosine kept.

        Returns:
            list[dict[int, float]]: One map of row index to cosine per query, in query order.

        Raises:
            QuoteMemError: ``embedding_space_mismatch`` when a row's dimension differs.
        """
        products = self._products(queries) if queries else None
        if products is None:
            return [{index: score for index, score in enumerate(self.cosines(query))
                     if score >= threshold} for query in queries]
        reached: list[dict[int, float]] = []
        for row, query in zip(products, queries, strict=True):
            scores: dict[int, float] = {}
            for index in np.flatnonzero(~(row < threshold - _REACH_MARGIN)).tolist():
                score = self._score(row.item(index), query, index)
                if score >= threshold:
                    scores[index] = score
            reached.append(scores)
        return reached

    def _products(self, queries: Sequence[UnitVector]) -> UnitArray | None:
        """Every query's fast dot product with every row, in one product; None when the
        rows are not stacked or a query's dimension differs (score row by row then)."""
        matrix = self._matrix
        if matrix is None or any(matrix.shape[1] != len(query) for query in queries):
            return None
        stacked = np.stack([np.asarray(query, dtype=np.float64) for query in queries])
        threads = _blas_threads()
        if threads is None:
            return np.einsum("kj,ij->ki", stacked, matrix)
        previous = threads(1)
        try:
            return stacked @ matrix.T
        finally:
            threads(previous)

    def _finish(self, products: UnitArray, query: UnitVector) -> list[float]:
        """Round one query's fast products; sum exactly only those near a boundary."""
        scores, unsettled = _settled(products)
        for index in unsettled:
            scores[index] = self._score(products.item(index), query, index)
        return scores

    def _score(self, value: float, query: UnitVector, index: int) -> float:
        """`cosine` of ``query`` and row ``index`` from their fast dot product ``value``."""
        if math.isfinite(value):
            low = _rounded(value - _FAST_MARGIN)
            if low == _rounded(value + _FAST_MARGIN):
                return low
        return _rounded(math.fsum(map(operator.mul, query, self.rows[index])))


@functools.cache
def _blas_threads() -> Callable[[int], int] | None:
    """OpenBLAS's ``openblas_set_num_threads_local`` of numpy's own build, or None.

    It sets the BLAS thread count of the calling thread only and returns the
    previous count. None when numpy's bundled OpenBLAS or the symbol is not
    found; products then avoid BLAS altogether.
    """
    try:
        directory = Path(np.__file__).parent.parent / "numpy.libs"
        found = sorted(directory.glob("libscipy_openblas*.so*"))
        if len(found) != 1:
            return None
        function = ctypes.CDLL(str(found[0])).openblas_set_num_threads_local
    except (OSError, AttributeError):
        return None
    function.argtypes = [ctypes.c_int]
    function.restype = ctypes.c_int
    return cast(Callable[[int], int], function)


_SETTLED_SLACK = 1e-7
"""Distance, in millionths, kept from every rounding boundary by `_settled`.

Scaling a fast dot product's margin interval to millionths and flooring it errs
by a few units of 1e-10 at most, far inside this slack.
"""

_SETTLED_BOUND = 0.999
"""Largest absolute fast dot product that `_settled` rounds itself, clear of the clip at 1."""


def _settled(products: UnitArray) -> tuple[list[float], list[int]]:
    """Round fast dot products whose margin interval holds no 6-decimal boundary, all at once.

    A dot product whose interval ``[p - _FAST_MARGIN, p + _FAST_MARGIN]``, widened
    by ``_SETTLED_SLACK`` millionths, holds no half-millionth rounds to the same
    millionth ``k`` at both ends, which is what `cosine` checks one row at a
    time; ``k / 1e6`` is then the double nearest the decimal ``k`` millionths,
    exactly what ``round(p, 6)`` returns. Every other product (near a boundary,
    near the clip at ±1 or not finite) is left to the caller's per-row check.

    Args:
        products: The fast dot products of one query with every row.

    Returns:
        tuple[list[float], list[int]]: The rounded scores, with 0.0 standing in
            at each unsettled index, and those indices in order.
    """
    low = np.floor((products - _FAST_MARGIN) * 1e6 + (0.5 - _SETTLED_SLACK))
    high = np.floor((products + _FAST_MARGIN) * 1e6 + (0.5 + _SETTLED_SLACK))
    settled = (low == high) & (np.abs(products) < _SETTLED_BOUND)
    scores: list[float] = (np.where(settled, low, 0.0) / 1e6 + 0.0).tolist()
    return scores, np.flatnonzero(~settled).tolist()


def centroid_typicality(vectors: Mapping[str, UnitVector]) -> Mapping[str, float]:
    """Return how typical of its population each unit vector is (hubness).

    A vector's typicality is its `cosine` with the population's centroid (the
    unit-scaled sum of every vector, keys in sorted order), less the mean of
    those cosines, rounded half-even to 6 decimals. Generic texts such as
    greetings, thanks and statements that say little beyond a name lie near the
    centroid and so near every query; Search subtracts typicality from its dense
    cosines so that they stop outranking specific content.

    Args:
        vectors (Mapping[str, UnitVector]): The population's unit vectors by id.

    Returns:
        Mapping[str, float]: Each id's typicality; empty for fewer than two
            vectors, mixed dimensions or a centroid without a direction.
    """
    keys = sorted(vectors)
    if len(keys) < 2 or len({len(vectors[key]) for key in keys}) != 1:
        return {}
    total = np.stack([unit_array(vectors[key]) for key in keys]).sum(axis=0)
    try:
        centroid = unit_vector(total.tolist())
    except QuoteMemError:
        return {}
    cosines = UnitRows([vectors[key] for key in keys]).cosines(centroid)
    mean = math.fsum(cosines) / len(cosines)
    return {key: round(value - mean, 6) + 0.0 for key, value in zip(keys, cosines, strict=True)}
