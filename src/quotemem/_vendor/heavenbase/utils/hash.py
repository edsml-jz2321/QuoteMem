"""Deterministic hash helpers."""

from __future__ import annotations

__all__ = [
    "HashIntFn",
    "crc32hash",
    "crc32int",
    "fmt_hash",
    "fmt_short_hash",
    "hash_int",
    "md5hash",
    "md5int",
    "resolve_hash_int",
    "sha256hash",
    "sha256int",
]

from importlib import import_module
from hashlib import md5, sha256
from zlib import crc32

from .serialize import dumps_json
from .typing import Any, Callable, Optional, Union

HashIntFn = Callable[[Any, Any, str], int]


def _serialize(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict) and not obj:
        return "{}"
    if isinstance(obj, list) and not obj:
        return "[]"
    if callable(obj) and hasattr(obj, "__module__") and hasattr(obj, "__qualname__"):
        return f"{obj.__module__}.{obj.__qualname__}"
    try:
        return dumps_json(obj, sort_keys=True, compact=True)
    except Exception:
        return repr(obj)


def _data(obj: Any, salt: Any = None, sep: str = "||") -> str:
    obj_repr = _serialize(obj)
    return f"{obj_repr}" if salt is None else f"{obj_repr}{sep}{_serialize(salt)}"


def md5int(obj: Any, salt: Any = None, sep: str = "||") -> int:
    """
    Return a deterministic MD5 digest as an integer for internal math.

    Use this helper for deterministic local identifiers, stable buckets, and lightweight integrity keys.

    Args:
        obj (Any): Value to serialize, format, print, copy, or pass through the helper.
        salt (Any): Optional salt mixed into the digest before hashing. Defaults to ``None``.
        sep (str): Separator inserted between the serialized value and salt. Defaults to ``"||"``.

    Returns:
        int: Deterministic MD5 digest represented as an integer.
    """
    return int(md5(_data(obj, salt=salt, sep=sep).encode("utf-8")).hexdigest(), 16)


def md5hash(obj: Any, salt: Any = None, sep: str = "||", length: int = 42) -> str:
    """
    Return a deterministic zero-padded decimal MD5 string.

    Use this helper for deterministic local identifiers, stable buckets, and lightweight integrity keys.

    Args:
        obj (Any): Value to serialize, format, print, copy, or pass through the helper.
        salt (Any): Optional salt mixed into the digest before hashing. Defaults to ``None``.
        sep (str): Separator inserted between the serialized value and salt. Defaults to ``"||"``.
        length (int): Minimum number of decimal digits in the returned identifier string. Defaults to ``42``.

    Returns:
        str: Zero-padded decimal MD5 string with the requested length.

    Examples:
        >>> from heavenbase.utils.hash import md5hash
        >>> md5hash({"a": 1}) == md5hash({"a": 1})
        True
    """
    return fmt_hash(md5int(obj, salt=salt, sep=sep), length=length)


def crc32int(obj: Any, salt: Any = None, sep: str = "||") -> int:
    """
    Return a deterministic CRC32 digest as an unsigned integer.

    Args:
        obj (Any): Value to serialize and hash.
        salt (Any): Optional salt mixed into the digest before hashing. Defaults to ``None``.
        sep (str): Separator inserted between the serialized value and salt. Defaults to ``"||"``.

    Returns:
        int: Unsigned CRC32 digest represented as an integer.
    """
    return crc32(_data(obj, salt=salt, sep=sep).encode("utf-8")) & 0xFFFFFFFF


def crc32hash(obj: Any, salt: Any = None, sep: str = "||", length: int = 10) -> str:
    """
    Return a deterministic zero-padded decimal CRC32 string.

    Args:
        obj (Any): Value to serialize and hash.
        salt (Any): Optional salt mixed into the digest before hashing. Defaults to ``None``.
        sep (str): Separator inserted between the serialized value and salt. Defaults to ``"||"``.
        length (int): Minimum number of decimal digits in the returned identifier string. Defaults to ``10``.

    Returns:
        str: Zero-padded decimal CRC32 string with the requested length.
    """
    return fmt_hash(crc32int(obj, salt=salt, sep=sep), length=length)


def sha256hash(obj: Any, salt: Any = None, sep: str = "||") -> str:
    """
    Return a deterministic SHA-256 hex digest.

    Use this helper for deterministic local identifiers, stable buckets, and lightweight integrity keys.

    Args:
        obj (Any): Value to serialize, format, print, copy, or pass through the helper.
        salt (Any): Optional salt mixed into the digest before hashing. Defaults to ``None``.
        sep (str): Separator inserted between the serialized value and salt. Defaults to ``"||"``.

    Returns:
        str: SHA-256 hexadecimal digest string.
    """
    return sha256(_data(obj, salt=salt, sep=sep).encode("utf-8")).hexdigest()


def sha256int(obj: Any, salt: Any = None, sep: str = "||") -> int:
    """
    Return a deterministic SHA-256 digest as an integer.

    Use this helper for deterministic buckets, modulo arithmetic, and stable ordering when MD5's shorter digest is not desired.

    Args:
        obj (Any): Value to serialize and hash.
        salt (Any): Optional salt mixed into the digest before hashing. Defaults to ``None``.
        sep (str): Separator inserted between the serialized value and salt. Defaults to ``"||"``.

    Returns:
        int: Deterministic SHA-256 digest represented as an integer.
    """
    return int(sha256hash(obj, salt=salt, sep=sep), 16)


def _xxhash_int(method: str, obj: Any, salt: Any = None, sep: str = "||") -> int:
    try:
        xxhash = import_module("xxhash")
    except Exception as exc:
        raise ImportError("Install the optional 'xxhash' package to use xxhash64 or xxhash128.") from exc
    data = _data(obj, salt=salt, sep=sep).encode("utf-8")
    candidates = {
        "xxh64": ("xxh64", "xxh3_64"),
        "xxhash64": ("xxh64", "xxh3_64"),
        "xxh364": ("xxh3_64",),
        "xxhash364": ("xxh3_64",),
        "xxh128": ("xxh3_128", "xxh128"),
        "xxhash128": ("xxh3_128", "xxh128"),
        "xxh3128": ("xxh3_128",),
        "xxhash3128": ("xxh3_128",),
    }[method]
    fn = next((getattr(xxhash, name, None) for name in candidates if getattr(xxhash, name, None) is not None), None)
    if fn is None:
        raise AttributeError(f"The installed xxhash package does not provide one of {candidates!r}")
    return int(fn(data).intdigest())


def _resolve_dotted_hash_int(name: str) -> HashIntFn:
    module_name, attr_name = name.rsplit(".", 1)
    fn = getattr(import_module(module_name), attr_name)
    if not callable(fn):
        raise TypeError(f"Hash function {name!r} is not callable")

    def call(obj: Any, salt: Any = None, sep: str = "||") -> int:
        return int(fn(obj, salt=salt, sep=sep))

    return call


def resolve_hash_int(method: str | None = None, *, allow_dotted: bool = True) -> HashIntFn:
    """
    Return a deterministic integer hash function by name.

    Use this helper when caller-selected deterministic hashing is part of runtime policy, such as sampling,
    bucketing, or candidate-index weighting. Built-in names are ``"md5"``, ``"sha256"``, ``"crc32"``,
    ``"xxhash64"``, and ``"xxhash128"``. The xxhash variants are optional and require the external
    ``xxhash`` package. Dotted paths are supported by default for deployment-local callables with the signature
    ``fn(obj, salt=None, sep="||") -> int``.

    Args:
        method (str | None): Hash method name. ``None`` resolves to ``"md5"``.
        allow_dotted (bool): Whether to allow deployment-local dotted callables. Defaults to ``True``.

    Returns:
        HashIntFn: Callable returning a deterministic integer digest.
    """
    name = str(method or "md5").strip()
    key = name.lower().replace("-", "").replace("_", "")
    if key == "md5":
        return md5int
    if key in {"sha256", "sha2"}:
        return sha256int
    if key in {"crc32", "crc"}:
        return crc32int
    if key in {"xxh64", "xxhash64", "xxh364", "xxhash364"}:
        return lambda obj, salt=None, sep="||": _xxhash_int(key, obj, salt=salt, sep=sep)
    if key in {"xxh128", "xxhash128", "xxh3128", "xxhash3128"}:
        return lambda obj, salt=None, sep="||": _xxhash_int(key, obj, salt=salt, sep=sep)
    if allow_dotted and "." in name:
        return _resolve_dotted_hash_int(name)
    expected = "md5, sha256, crc32, xxhash64, xxhash128"
    if allow_dotted:
        expected = f"{expected}, or a dotted callable"
    raise ValueError(f"Unknown hash function {method!r}; expected {expected}")


def hash_int(obj: Any, method: str | None = None, salt: Any = None, sep: str = "||") -> int:
    """
    Return a deterministic integer hash using the selected method.

    Use this helper for configurable stable buckets, sampling, and internal candidate generation. Keep
    persisted identity and integrity hashes pinned to explicit helpers such as ``hash_id`` or ``sha256hash``.

    Args:
        obj (Any): Value to serialize and hash.
        method (str | None): Hash method name. ``None`` resolves to ``"md5"``.
        salt (Any): Optional salt mixed into the digest before hashing. Defaults to ``None``.
        sep (str): Separator inserted between the serialized value and salt. Defaults to ``"||"``.

    Returns:
        int: Deterministic integer digest.
    """
    return resolve_hash_int(method)(obj, salt=salt, sep=sep)


def fmt_hash(value: Optional[Union[str, int]], length: int = 42) -> Optional[str]:
    """
    Format a hash value as a zero-padded decimal identifier string.

    Use this helper for deterministic local identifiers, stable buckets, and lightweight integrity keys.

    Args:
        value (Optional[Union[str, int]]): Value to read, set, validate, hash, or format.
        length (int): Minimum number of decimal digits in the returned identifier string. Defaults to ``42``.

    Returns:
        Optional[str]: Zero-padded string or ``None`` when ``value`` is ``None``.
    """
    if value is None:
        return None
    if length < 1:
        raise ValueError("hash length must be positive")
    return str(value).zfill(length) if isinstance(value, str) else f"{value:0{length}d}"


def fmt_short_hash(value: Optional[Union[str, int]], length: int = 8) -> Optional[str]:
    """
    Format the short tail of an integer or string hash.

    Use this helper for deterministic local identifiers, stable buckets, and lightweight integrity keys.

    Args:
        value (Optional[Union[str, int]]): Value to read, set, validate, hash, or format.
        length (int): Number of trailing hash characters to keep. Defaults to ``8``.

    Returns:
        Optional[str]: Tail hash string or ``None`` when ``value`` is ``None``.
    """
    if value is None:
        return None
    if length < 1:
        raise ValueError("hash length must be positive")
    return str(value)[-length:] if isinstance(value, str) else f"{value % (10**length):0{length}d}"
