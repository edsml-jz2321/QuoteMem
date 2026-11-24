"""Stable random generation helpers."""

from __future__ import annotations

__all__ = ["StableRNG", "evolve_seed", "interpret_seed"]

import heapq
import math
import time
from uuid import uuid4

from .config import CM_HVNB
from .hash import md5int, resolve_hash_int
from .typing import Any, Iterable, List, Literal, Optional, Tuple, Union

_DEFAULT_HASH_MODULO = 1061109589
_DEFAULT_HASH_FUNCTION = "crc32"


def _np():
    import numpy as np

    return np


def _reshape(values: List[Any], shape: Tuple[int, ...]) -> Any:
    if len(shape) == 1:
        return values[: shape[0]]
    step = math.prod(shape[1:])
    return [_reshape(values[i * step : (i + 1) * step], shape[1:]) for i in range(shape[0])]


def _shape(n: Optional[Union[int, Tuple[int, ...]]]) -> Tuple[Tuple[int, ...], bool]:
    return ((1,), True) if n is None else (((n,), False) if isinstance(n, int) else (tuple(n), False))


def _shape_out(values: List[Any], n: Optional[Union[int, Tuple[int, ...]]]) -> Any:
    shape, single = _shape(n)
    return values[0] if single else _reshape(values, shape)


def interpret_seed(seed: Optional[int], fallback: Literal["time", "uuid", "none"] = "none") -> Optional[int]:
    """
    Convert an optional seed into the concrete seed used by deterministic helpers.

    Use this helper for deterministic test data, benchmark rows, samples, and synthetic vectors.

    Args:
        seed (Optional[int]): Optional deterministic seed. ``None`` uses the selected fallback behavior.
        fallback (Literal['time', 'uuid', 'none']): Seed fallback policy. Allowed values are ``"time"``, ``"uuid"``, and ``"none"``. Defaults to ``'none'``.

    Returns:
        Optional[int]: Concrete integer seed, or ``None`` when unstable generation is requested.

    Examples:
        >>> from heavenbase.utils.rng import interpret_seed
        >>> interpret_seed(42)
        42
    """
    if seed is not None:
        return int(seed)
    if fallback == "time":
        return int(time.time() * 1_000_000)
    if fallback == "uuid":
        return int(uuid4())
    if fallback == "none":
        return None
    raise ValueError(f"Unknown seed fallback {fallback!r}")


def evolve_seed(seed: Optional[int], *args: Any, **kwargs: Any) -> Optional[int]:
    """
    Derive a deterministic child seed from a base seed and context values.

    Use this helper for deterministic test data, benchmark rows, samples, and synthetic vectors.

    Args:
        seed (Optional[int]): Optional deterministic seed. ``None`` uses the selected fallback behavior.
        *args (Any): Positional values forwarded to the wrapped callable or used as deterministic seed context.
        **kwargs (Any): Extra keyword arguments forwarded to the underlying serializer, command, or callable.

    Returns:
        Optional[int]: Deterministic child seed, or ``None`` when the base seed is ``None``.

    Examples:
        >>> from heavenbase.utils.rng import evolve_seed
        >>> evolve_seed(42, "rows") == evolve_seed(42, "rows")
        True
    """
    return None if seed is None else md5int([args, kwargs], salt=seed)


class StableRNG:
    """A NumPy-backed random generator with deterministic seed evolution."""

    def __init__(
        self,
        seed: Optional[int] = None,
        fallback: Literal["time", "uuid", "none"] = "none",
        backend: str = "philox",
        hash_function: Optional[str] = None,
        hash_modulo: Optional[int] = None,
    ) -> None:
        """
        Initialize the utility object with the provided runtime settings.

        Use this helper for deterministic test data, benchmark rows, samples, and synthetic vectors.

        Omitted hash settings are read from ``CM_HVNB``; when the saved config cannot be opened (an unwritable home,
        an invalid bootstrap file, or explicit Context mode), the packaged settings are used, as in 0.1.2.2. Any other
        config error is raised.

        Args:
            seed (Optional[int]): Optional deterministic seed. ``None`` uses the selected fallback behavior. Defaults to ``None``.
            fallback (Literal['time', 'uuid', 'none']): Seed fallback policy. Allowed values are ``"time"``, ``"uuid"``, and ``"none"``. Defaults to ``'none'``.
            backend (str): NumPy bit-generator backend. Defaults to ``"philox"``.
            hash_function (Optional[str]): Hash method for ``hash_sample`` and ``hash_split``. Defaults to config.
            hash_modulo (Optional[int]): Modulo used by hash sampling weights. Defaults to config.

        Returns:
            None: This function returns ``None`` after initializing the object.
        """
        self._stack = [interpret_seed(seed, fallback=fallback)]
        self._backend = backend
        configured_hash = (
            hash_function if hash_function is not None else CM_HVNB._get_or_packaged("heavenbase.rng.hash_function", default=_DEFAULT_HASH_FUNCTION)
        )
        self._hash_function = str(configured_hash or _DEFAULT_HASH_FUNCTION)
        configured_modulo = hash_modulo if hash_modulo is not None else CM_HVNB._get_or_packaged("heavenbase.rng.hash_modulo", default=_DEFAULT_HASH_MODULO)
        self._hash_modulo = int(_DEFAULT_HASH_MODULO if configured_modulo is None else configured_modulo)
        if self._hash_modulo < 1:
            raise ValueError("StableRNG hash_modulo must be positive")
        resolve_hash_int(self._hash_function)
        self._rng = None
        self._in_context = False
        self._set()

    @property
    def seed(self) -> Optional[int]:
        """Return the current seed.

        Use this helper for deterministic test data, benchmark rows, samples, and synthetic vectors.

        Args:
            None.

        Returns:
            Optional[int]: The the current seed.
        """
        return self._stack[-1]

    @property
    def initial_seed(self) -> Optional[int]:
        """Return the original seed.

        Use this helper for deterministic test data, benchmark rows, samples, and synthetic vectors.

        Args:
            None.

        Returns:
            Optional[int]: The the original seed.
        """
        return self._stack[0]

    @property
    def seeds(self) -> List[Optional[int]]:
        """Return the current seed stack.

        Use this helper for deterministic test data, benchmark rows, samples, and synthetic vectors.

        Args:
            None.

        Returns:
            List[Optional[int]]: The the current seed stack.
        """
        return self._stack.copy()

    @property
    def rng(self) -> Any:
        """Return the active NumPy generator inside a context.

        Use this helper for deterministic test data, benchmark rows, samples, and synthetic vectors.

        Args:
            None.

        Returns:
            Any: The the active NumPy generator inside a context.
        """
        if not self._in_context:
            raise RuntimeError("StableRNG.rng can only be accessed inside a context")
        return self._rng

    @property
    def hash_function(self) -> str:
        """Return the hash function used by order-independent sampling helpers.

        Args:
            None.

        Returns:
            str: Configured hash function name.
        """
        return self._hash_function

    @property
    def hash_modulo(self) -> int:
        """Return the modulo used by order-independent sampling helpers.

        Args:
            None.

        Returns:
            int: Positive hash modulo.
        """
        return self._hash_modulo

    def _set(self) -> None:
        np = _np()
        generators = {"pcg": np.random.PCG64, "pcg64": np.random.PCG64, "philox": np.random.Philox, "mt19937": np.random.MT19937, "sfc64": np.random.SFC64}
        if self._backend not in generators:
            raise ValueError(f"Unknown RNG backend {self._backend!r}")
        bitgen = generators[self._backend]
        self._rng = np.random.Generator(bitgen(self.seed)) if self.seed is not None else np.random.Generator(bitgen())

    def enter(self) -> "StableRNG":
        """Enter deterministic generation context.

        Use this helper for deterministic test data, benchmark rows, samples, and synthetic vectors.

        Args:
            None.

        Returns:
            'StableRNG': The result of ``enter``.
        """
        if not self._in_context:
            self._in_context = True
            self._set()
        return self

    def exit(self) -> None:
        """Exit deterministic generation context.

        Use this helper for deterministic test data, benchmark rows, samples, and synthetic vectors.

        Args:
            None.

        Returns:
            None: This function returns ``None`` after completing its side effect.
        """
        if self._in_context:
            self._set()
            self._in_context = False

    def __enter__(self) -> "StableRNG":
        """
        Enter the context manager and apply its temporary state.

        Use this helper for deterministic test data, benchmark rows, samples, and synthetic vectors.

        Returns:
            'StableRNG': The active deterministic generator context.
        """
        return self.enter()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """
        Exit the context manager and restore the previous state.

        Use this helper for deterministic test data, benchmark rows, samples, and synthetic vectors.

        Args:
            exc_type (Any): Exception type passed by the context-manager protocol, or ``None`` on a clean exit.
            exc_val (Any): Exception instance passed by the context-manager protocol, or ``None`` on a clean exit.
            exc_tb (Any): Traceback passed by the context-manager protocol, or ``None`` on a clean exit.

        Returns:
            None: This function returns ``None`` after completing its side effect.
        """
        self.exit()

    def evolve(self, *args: Any) -> "StableRNG":
        """
        Return a deterministic child stream. Prefer `step` in new code.

        Use this helper for deterministic test data, benchmark rows, samples, and synthetic vectors.

        Args:
            *args (Any): Positional values forwarded to the wrapped callable or used as deterministic seed context.

        Returns:
            'StableRNG': The a deterministic child stream. Prefer `step` in new code.
        """
        return self.step(*args)

    def step(self, *args: Any) -> "StableRNG":
        """
        Return a child generator without mutating this generator.

        Use this helper for deterministic test data, benchmark rows, samples, and synthetic vectors.

        Args:
            *args (Any): Positional values forwarded to the wrapped callable or used as deterministic seed context.

        Returns:
            'StableRNG': The a child generator without mutating this generator.

        Examples:
            >>> from heavenbase.utils.rng import StableRNG
            >>> StableRNG(seed=42).step("rows").rnd_int(0, 10) == StableRNG(seed=42).step("rows").rnd_int(0, 10)
            True
        """
        child = StableRNG(seed=self.initial_seed, backend=self._backend, hash_function=self._hash_function, hash_modulo=self._hash_modulo)
        child._stack = [*self._stack, evolve_seed(self.seed, *args)]
        child._set()
        return child

    def revert(self) -> "StableRNG":
        """Pop the latest child seed if one exists.

        Use this helper for deterministic test data, benchmark rows, samples, and synthetic vectors.

        Args:
            None.

        Returns:
            'StableRNG': The result of ``revert``.
        """
        if len(self._stack) > 1:
            self._stack.pop()
        self._set()
        return self

    def reset(self) -> "StableRNG":
        """Return to the initial seed.

        Use this helper for deterministic test data, benchmark rows, samples, and synthetic vectors.

        Args:
            None.

        Returns:
            'StableRNG': The to the initial seed.
        """
        self._stack = [self.initial_seed]
        self._set()
        return self

    def _auto(self, fn: Any) -> Any:
        was_in_context = self._in_context
        if not was_in_context:
            self.enter()
        try:
            return fn()
        finally:
            if not was_in_context:
                self.exit()

    def rnd(self, n: Optional[Union[int, Tuple[int, ...]]] = None) -> Union[float, List[float], List[List[float]]]:
        """
        Return random floats in [0, 1).

        Use this helper for deterministic test data, benchmark rows, samples, and synthetic vectors.

        Args:
            n (Optional[Union[int, Tuple[int, ...]]]): Optional output count or shape. ``None`` returns one scalar value when supported. Defaults to ``None``.

        Returns:
            Union[float, List[float], List[List[float]]]: The random floats in [0, 1).
        """
        return self._auto(lambda: float(self.rng.random()) if n is None else self.rng.random(size=n).tolist())

    def rnd_int(self, min: int = 0, max: int = 2**31, n: Optional[Union[int, Tuple[int, ...]]] = None) -> Union[int, List[int], List[List[int]]]:
        """
        Return random integers in [min, max).

        Use this helper for deterministic test data, benchmark rows, samples, and synthetic vectors.

        Args:
            min (int): Inclusive lower bound for generated numeric values. Defaults to ``0``.
            max (int): Exclusive upper bound for generated numeric values. Defaults to ``2 ** 31``.
            n (Optional[Union[int, Tuple[int, ...]]]): Optional output count or shape. ``None`` returns one scalar value when supported. Defaults to ``None``.

        Returns:
            Union[int, List[int], List[List[int]]]: The random integers in [min, max).

        Examples:
            >>> from heavenbase.utils.rng import StableRNG
            >>> StableRNG(seed=42).rnd_int(0, 10) == StableRNG(seed=42).rnd_int(0, 10)
            True
        """
        return self._auto(lambda: int(self.rng.integers(min, max)) if n is None else self.rng.integers(min, max, size=n).tolist())

    def rnd_normal(self, mu: float = 0.0, sigma: float = 1.0, n: Optional[Union[int, Tuple[int, ...]]] = None) -> Union[float, List[float], List[List[float]]]:
        """
        Return normally distributed floats.

        Use this helper for deterministic test data, benchmark rows, samples, and synthetic vectors.

        Args:
            mu (float): Mean of the normal distribution. Defaults to ``0.0``.
            sigma (float): Standard deviation of the normal distribution. Defaults to ``1.0``.
            n (Optional[Union[int, Tuple[int, ...]]]): Optional output count or shape. ``None`` returns one scalar value when supported. Defaults to ``None``.

        Returns:
            Union[float, List[float], List[List[float]]]: The normally distributed floats.
        """
        return self._auto(lambda: float(self.rng.normal(loc=mu, scale=sigma)) if n is None else self.rng.normal(loc=mu, scale=sigma, size=n).tolist())

    def rnd_float(self, min: float, max: float, n: Optional[Union[int, Tuple[int, ...]]] = None) -> Union[float, List[float], List[List[float]]]:
        """
        Return uniformly distributed floats in [min, max).

        Use this helper for deterministic test data, benchmark rows, samples, and synthetic vectors.

        Args:
            min (float): Inclusive lower bound for generated numeric values.
            max (float): Exclusive upper bound for generated numeric values.
            n (Optional[Union[int, Tuple[int, ...]]]): Optional output count or shape. ``None`` returns one scalar value when supported. Defaults to ``None``.

        Returns:
            Union[float, List[float], List[List[float]]]: The uniformly distributed floats in [min, max).
        """
        return self._auto(lambda: float(self.rng.uniform(min, max)) if n is None else self.rng.uniform(min, max, size=n).tolist())

    def rnd_str(self, length: int, charset: str = "abcdefghijklmnopqrstuvwxyz0123456789", n: Optional[Union[int, Tuple[int, ...]]] = None) -> Any:
        """
        Return stable random strings from a character set.

        Use this helper for deterministic test data, benchmark rows, samples, and synthetic vectors.

        Args:
            length (int): Required string length or hash width, depending on the helper.
            charset (str): Characters sampled when building random strings. Defaults to lowercase letters and digits.
            n (Optional[Union[int, Tuple[int, ...]]]): Optional output count or shape. ``None`` returns one scalar value when supported. Defaults to ``None``.

        Returns:
            Any: The stable random strings from a character set.
        """

        def run() -> Any:
            shape, _ = _shape(n)
            indices = self.rng.integers(0, len(charset), size=(math.prod(shape), length))
            values = ["".join(charset[int(idx)] for idx in row) for row in indices]
            return _shape_out(values, n)

        return self._auto(run)

    def choice(self, seq: List[Any], k: int = 1, replace: bool = True, p: Optional[List[float]] = None, n: Optional[Union[int, Tuple[int, ...]]] = None) -> Any:
        """
        Return one or more sampled items from a sequence.

        Use this helper for deterministic test data, benchmark rows, samples, and synthetic vectors.

        Args:
            seq (List[Any]): Sequence or iterable being sampled, shuffled, split, or validated.
            k (int): Number of sampled items. Defaults to ``1``.
            replace (bool): Whether sampling is with replacement. Defaults to ``True``.
            p (Optional[List[float]]): Optional sampling probability vector. ``None`` samples uniformly.
            n (Optional[Union[int, Tuple[int, ...]]]): Optional output count or shape. ``None`` returns one scalar value when supported. Defaults to ``None``.

        Returns:
            Any: The one or more sampled items from a sequence.
        """

        def run() -> Any:
            shape = (1,) if n is None else ((n,) if isinstance(n, int) else n)
            out = []
            for _ in range(math.prod(shape)):
                picked = self.rng.choice(len(seq), size=k, replace=replace, p=p)
                values = [seq[int(idx)] for idx in picked]
                out.append(values[0] if k == 1 else values)
            return out[0] if n is None else _reshape(out, shape)

        return self._auto(run)

    def perm(self, dim: int, n: Optional[Union[int, Tuple[int, ...]]] = None) -> Any:
        """
        Return one or more permutations of range(dim).

        Use this helper for deterministic test data, benchmark rows, samples, and synthetic vectors.

        Args:
            dim (int): Vector dimension. Defaults to the helper-specific dimension.
            n (Optional[Union[int, Tuple[int, ...]]]): Optional output count or shape. ``None`` returns one scalar value when supported. Defaults to ``None``.

        Returns:
            Any: The one or more permutations of range(dim).
        """
        return self._auto(
            lambda: _shape_out(self.rng.permuted(_np().broadcast_to(_np().arange(dim), _shape(n)[0] + (dim,)).copy(), axis=-1).reshape((-1, dim)).tolist(), n)
        )

    def shuffled(self, seq: List[Any], n: Optional[Union[int, Tuple[int, ...]]] = None) -> Any:
        """
        Return shuffled copies of a sequence.

        Use this helper for deterministic test data, benchmark rows, samples, and synthetic vectors.

        Args:
            seq (List[Any]): Sequence or iterable being sampled, shuffled, split, or validated.
            n (Optional[Union[int, Tuple[int, ...]]]): Optional output count or shape. ``None`` returns one scalar value when supported. Defaults to ``None``.

        Returns:
            Any: The shuffled copies of a sequence.
        """
        return self._auto(
            lambda: _shape_out(
                [
                    [seq[int(idx)] for idx in perm]
                    for perm in self.rng.permuted(_np().broadcast_to(_np().arange(len(seq)), _shape(n)[0] + (len(seq),)).copy(), axis=-1)
                    .reshape((-1, len(seq)))
                    .tolist()
                ],
                n,
            )
        )

    def hash_split(
        self,
        seq: Iterable[Any],
        r: float = 0.10,
        n: Optional[Union[int, Tuple[int, ...]]] = None,
        hash_function: Optional[str] = None,
    ) -> Any:
        """
        Return stable hash partitions as `(selected, remaining)` pairs.

        Use this helper for deterministic test data, benchmark rows, samples, and synthetic vectors.

        Args:
            seq (Iterable[Any]): Sequence or iterable being sampled, shuffled, split, or validated.
            r (float): Selection ratio used by the hash split. Defaults to ``0.1``.
            n (Optional[Union[int, Tuple[int, ...]]]): Optional output count or shape. ``None`` returns one scalar value when supported. Defaults to ``None``.
            hash_function (Optional[str]): Optional per-call hash function override. Defaults to this generator's hash function.

        Returns:
            Any: The stable hash partitions as `(selected, remaining)` pairs.
        """
        items = list(seq)
        shape, _ = _shape(n)
        hasher = resolve_hash_int(self._hash_function if hash_function is None else hash_function)
        threshold = int(r * self._hash_modulo)
        out = []
        for pos in range(math.prod(shape)):
            selected, remaining = [], []
            for item in items:
                (selected if hasher(item, (self.seed, pos), "||") % self._hash_modulo < threshold else remaining).append(item)
            out.append((selected, remaining))
        return _shape_out(out, n)

    def hash_sample(
        self,
        seq: Iterable[Any],
        k: int = 1,
        n: Optional[Union[int, Tuple[int, ...]]] = None,
        hash_function: Optional[str] = None,
    ) -> Any:
        """
        Return a stable hash-based sample that is independent of input order.

        Use this helper for deterministic test data, benchmark rows, samples, and synthetic vectors.

        Args:
            seq (Iterable[Any]): Sequence or iterable being sampled, shuffled, split, or validated.
            k (int): Number of sampled items. Defaults to ``1``.
            n (Optional[Union[int, Tuple[int, ...]]]): Optional output count or shape. ``None`` returns one scalar value when supported. Defaults to ``None``.
            hash_function (Optional[str]): Optional per-call hash function override. Defaults to this generator's hash function.

        Returns:
            Any: The a stable hash-based sample that is independent of input order.
        """
        items = list(seq)
        shape, _ = _shape(n)
        hasher = resolve_hash_int(self._hash_function if hash_function is None else hash_function)
        out = []
        for pos in range(math.prod(shape)):
            indices = heapq.nsmallest(
                k,
                range(len(items)),
                key=lambda idx: (
                    hasher(items[idx], salt=(self.seed, pos), sep="||") % self._hash_modulo,
                    hasher(items[idx], salt=("tie", self.seed, pos), sep="||"),
                ),
            )
            out.append([items[idx] for idx in indices])
        return _shape_out(out, n)

    def rnd_vec(self, dim: int = 384, n: Optional[Union[int, Tuple[int, ...]]] = None) -> Any:
        """
        Return unit-length normally distributed vector embeddings.

        Use this helper for deterministic test data, benchmark rows, samples, and synthetic vectors.

        Args:
            dim (int): Vector dimension. Defaults to the helper-specific dimension.
            n (Optional[Union[int, Tuple[int, ...]]]): Optional output count or shape. ``None`` returns one scalar value when supported. Defaults to ``None``.

        Returns:
            Any: The unit-length normally distributed vector embeddings.
        """

        def run() -> Any:
            np = _np()
            shape, _ = _shape(n)
            values = self.rng.normal(loc=0.0, scale=1.0, size=shape + (dim,))
            norms = np.linalg.norm(values, axis=-1, keepdims=True)
            return _shape_out((values / norms).reshape((-1, dim)).tolist(), n)

        return self._auto(run)

    def abm(self, base: float, t: float = 0.0, trend: float = 0.0, vol_ratio: float = 0.035, n: Optional[int] = None) -> Any:
        """
        Return Arithmetic Brownian Motion values at time `t`.

        Use this helper for deterministic test data, benchmark rows, samples, and synthetic vectors.

        Args:
            base (float): Starting scalar value for the generated path or process.
            t (float): Time horizon for the generated process. Defaults to ``0.0``.
            trend (float): Linear drift ratio applied over time. Defaults to ``0.0``.
            vol_ratio (float): Volatility ratio applied to random shocks. Defaults to ``0.035``.
            n (Optional[int]): Optional output count or shape. ``None`` returns one scalar value when supported. Defaults to ``None``.

        Returns:
            Any: The Arithmetic Brownian Motion values at time `t`.
        """

        def run() -> Any:
            count = 1 if n is None else n
            zs = self.rng.normal(loc=0.0, scale=1.0, size=count)
            values = base * (1 + trend * t) + vol_ratio * base * zs * math.sqrt(t)
            out = values.tolist()
            return out[0] if n is None else out

        return self._auto(run)

    def abm_path(self, base: float, t: int = 0, trend: float = 0.0, vol_ratio: float = 0.035, n: Optional[Union[int, Tuple[int, ...]]] = None) -> Any:
        """
        Return Arithmetic Brownian Motion paths from step 0 through `t`.

        Use this helper for deterministic test data, benchmark rows, samples, and synthetic vectors.

        Args:
            base (float): Starting scalar value for the generated path or process.
            t (int): Number of discrete path steps. Defaults to ``0``.
            trend (float): Linear drift ratio applied over time. Defaults to ``0.0``.
            vol_ratio (float): Volatility ratio applied to random shocks. Defaults to ``0.035``.
            n (Optional[Union[int, Tuple[int, ...]]]): Optional output count or shape. ``None`` returns one scalar value when supported. Defaults to ``None``.

        Returns:
            Any: The Arithmetic Brownian Motion paths from step 0 through `t`.
        """

        def run() -> Any:
            np = _np()
            shape, _ = _shape(n)
            zs = self.rng.normal(loc=0.0, scale=1.0, size=shape + (t,))
            paths = np.empty(shape + (t + 1,))
            paths[..., 0] = base
            paths[..., 1:] = base + trend * base * np.arange(1, t + 1) + vol_ratio * base * np.cumsum(zs, axis=-1)
            return _shape_out(paths.reshape((-1, t + 1)).tolist(), n)

        return self._auto(run)
