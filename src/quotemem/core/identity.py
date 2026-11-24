"""Canonical identities and separate digest roles."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import re
import secrets
import time
from array import array
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from functools import cache
from json.encoder import encode_basestring
from operator import countOf
from typing import Annotated, Any, ClassVar, Final, Self, cast

from pydantic import BaseModel, GetCoreSchemaHandler, StringConstraints
from pydantic_core import CoreSchema, core_schema

from quotemem.core.errors import QuoteMemError

ULID = Annotated[str, StringConstraints(pattern=r"^[0-7][0-9A-HJKMNP-TV-Z]{25}$")]
ControlId = Annotated[str, StringConstraints(pattern=r"^ctl_[A-Za-z0-9_-]{43}$")]
ScopeId = Annotated[str, StringConstraints(pattern=r"^sc_[A-Za-z0-9_-]{40}$")]
PrincipalId = Annotated[str, StringConstraints(pattern=r"^pr_[A-Za-z0-9_-]{40}$")]


class _Digest(str):
    """Validate a string while retaining its semantic digest role at runtime.

    Digests are built and compared millions of times in one large finalize:
    each role compiles its pattern once, a validated field takes its role
    without matching the pattern a second time, and equality with a plain
    string or the same role skips the role check.
    """

    pattern: ClassVar[str] = r"^[0-9a-f]{64}$"
    _fullmatch: ClassVar[Callable[[str], re.Match[str] | None]] = re.compile(pattern).fullmatch

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Compile the role's pattern once, when the role is declared."""
        super().__init_subclass__(**kwargs)
        cls._fullmatch = re.compile(cls.pattern).fullmatch

    def __new__(cls, value: object) -> Self:
        """Reject malformed values and accidental conversion between roles."""
        if isinstance(value, _Digest) and type(value) is not cls:
            raise QuoteMemError("digest_role_mismatch", "Digest roles cannot be interchanged")
        if not isinstance(value, str) or cls._fullmatch(value) is None:
            raise QuoteMemError("schema_invalid", f"Invalid {cls.__name__} encoding")
        return super().__new__(cls, value)

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        """Preserve runtime role checks in strict Python and JSON validation."""
        return core_schema.no_info_before_validator_function(
            cls._validate_role,
            core_schema.no_info_after_validator_function(
                cls._from_matched, core_schema.str_schema(strict=True, pattern=cls.pattern)
            ),
        )

    @classmethod
    def _from_matched(cls, value: str) -> Self:
        """Give the role to a string the schema already matched against ``pattern``.

        ``_validate_role`` has already refused another role's digest.
        """
        return str.__new__(cls, value)

    @classmethod
    def _validate_role(cls, value: object) -> object:
        """Reject a differently typed digest before string normalization."""
        if isinstance(value, _Digest) and type(value) is not cls:
            raise QuoteMemError("digest_role_mismatch", "Digest roles cannot be interchanged")
        return value

    def __eq__(self, other: object) -> bool:
        """Fail on cross-role comparisons instead of equating identical bytes."""
        kind = type(other)
        if kind is not str and kind is not type(self) and isinstance(other, _Digest):
            raise QuoteMemError("digest_role_mismatch", "Cannot compare different digest roles")
        return str.__eq__(self, other)

    def __ne__(self, other: object) -> bool:
        """Apply the same role check to inequality comparisons."""
        return not self == other

    __hash__ = str.__hash__


class ArtifactId(_Digest):
    """Canonical Artifact envelope address, separate from every body digest."""

    pattern = r"^art_[A-Za-z0-9_-]{43}$"


class ContentBindingDigest(_Digest):
    """Tenant-keyed HMAC identity, encoded as unpadded base64url."""

    pattern = r"^[A-Za-z0-9_-]{43}$"


class PublishPlanDigest(_Digest):
    """SHA-256 of the complete strict PublishPlan body."""


class IntegrityDigest(_Digest):
    """SHA-256 for structural integrity or content in the same deletion closure."""


_CANONICAL = json.JSONEncoder(
    sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
)
"""The encoder ``json.dumps`` builds for these options, built once and holding no call state."""


_VECTOR_LENGTH: Final = 64
"""The shortest float array whose encoding a ``vector_encodings`` block keeps."""

_VECTOR_DEPTH: Final = 4
"""How deep in the encoded value a vector is looked for; deeper ones are formatted plainly.

The deepest the store encodes is the read model's root identity: its source changes,
one change, its row and the row's vector.
"""

_VECTORS: ContextVar[dict[bytes, str] | None] = ContextVar("aml_vector_encodings", default=None)


@contextmanager
def vector_encodings() -> Iterator[None]:
    """Encode each embedding vector once inside the block (``canonical_json``).

    One Add canonicalizes its Evidence rows many times: reception validates,
    binds and publishes them, and its owner freezes, scans, sizes and settles
    them. Formatting a vector's floats dominates each of those encodings (about
    0.6 ms per 1,024 values). Inside the block the encoding of a list or tuple of
    at least ``_VECTOR_LENGTH`` exact finite floats is kept under the floats'
    IEEE-754 bytes, which determine it, and reused by every later encoding that
    holds the same vector: each output stays byte for byte what ``_CANONICAL``
    writes. Threads that run in a copy of the block's context share the kept
    encodings; leaving the outermost block drops them.

    Yields:
        None: The block runs with vector encodings kept.
    """
    if _VECTORS.get() is not None:
        yield
        return
    token = _VECTORS.set({})
    try:
        yield
    finally:
        _VECTORS.reset(token)


def canonical_json(value: object) -> str:
    """Encode deterministic UTF-8 JSON without normalizing runtime text bytes.

    Inside ``vector_encodings`` an embedding vector already encoded there is
    reused rather than formatted again; the result is the same.

    Args:
        value: A validated model or JSON-compatible value.

    Returns:
        Sorted, compact JSON with finite numbers only.
    """
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json", exclude_none=False)
    vectors = _VECTORS.get()
    if vectors is not None:
        found: dict[int, bytes] = {}
        if _holds_vector(value, 0, set(), found):
            chunks: list[str] = []
            _write_holding(value, 0, set(), vectors, found, chunks)
            return "".join(chunks)
    return _CANONICAL.encode(value)


_PLAIN_DEPTH: Final = 32
"""How deep ``canonical_equal`` compares structurally before it encodes instead."""


def canonical_equal(left: object, right: object) -> bool:
    """Return ``canonical_json(left) == canonical_json(right)`` without encoding plain values.

    A restored row is checked against its stored physical row this way, and
    encoding both formats every float of the row's embedding twice: about
    3 ms per 2,048-value Evidence row, most of a cold boot's row restoration.
    Values built only of exact ``dict`` (with text keys, ``_text_key``),
    ``list``, ``tuple``, ``str``, ``int``, ``bool``, ``None`` and finite
    ``float`` are compared
    structurally, with the encoder's distinctions: ``1`` and ``1.0`` and
    ``True`` differ, ``-0.0`` and ``0.0`` differ, a list equals a tuple with
    the same members. Anything else, and any difference, is decided by the
    two encodings, so the answer, and every error the encodings raise (a
    non-finite float, a cycle, an unserializable value), is theirs.

    Args:
        left: A validated model or JSON-compatible value.
        right: Another.

    Returns:
        Whether both have the same canonical JSON.

    Raises:
        TypeError: When a value is not JSON-serializable, as ``canonical_json`` raises.
        ValueError: When a value holds a non-finite float or a cycle, likewise.
    """
    if _plain_equal(left, right, 0):
        return True
    return canonical_json(left) == canonical_json(right)


def _plain_equal(left: object, right: object, depth: int) -> bool:
    """Tell whether two plain values certainly encode alike; ``False`` means "encode them".

    ``True`` is returned only when both values are built of the plain types
    ``canonical_equal`` names, every float is finite and their encodings are
    equal, so neither encoding could raise or differ.
    """
    kind, other = type(left), type(right)
    if kind is str:
        return other is str and left == right
    if kind is int:
        return other is int and left == right
    if kind is bool:
        return other is bool and left is right
    if left is None:
        return right is None
    if kind is float:
        return other is float and math.isfinite(cast(float, left)) and (
            array("d", (cast(float, left),)).tobytes()
            == array("d", (cast(float, right),)).tobytes()
        )
    if depth >= _PLAIN_DEPTH:
        return False
    if kind is dict:
        if other is not dict:
            return False
        mapping, against = cast(dict[object, object], left), cast(dict[object, object], right)
        if len(mapping) != len(against) or not all(map(_text_key, map(type, against))):
            return False
        for key, value in mapping.items():
            if not _text_key(type(key)) or key not in against:
                return False
            if not _plain_equal(value, against[key], depth + 1):
                return False
        return True
    if (kind is list or kind is tuple) and (other is list or other is tuple):
        items, against_items = cast(Sequence[object], left), cast(Sequence[object], right)
        if len(items) != len(against_items):
            return False
        if items and countOf(map(type, items), float) == len(items):
            # A vector: equal IEEE-754 bytes of finite floats are equal reprs.
            if countOf(map(type, against_items), float) != len(items):
                return False
            floats = cast(Sequence[float], items)
            if not math.isfinite(sum(floats)):
                return False
            return array("d", floats).tobytes() == array(
                "d", cast(Sequence[float], against_items),
            ).tobytes()
        return all(
            _plain_equal(item, against, depth + 1)
            for item, against in zip(items, against_items, strict=True)
        )
    return False


def _vector_bytes(value: object, found: dict[int, bytes]) -> bytes | None:
    """Return the IEEE-754 bytes of an array of ``_VECTOR_LENGTH`` or more exact finite floats.

    ``_CANONICAL`` writes such an array as its floats' ``repr``s, which these
    bytes determine. Any other array, or one whose sum overflows, has none and is
    formatted plainly. ``found`` keeps the bytes per array for one encoding.
    """
    packed = found.get(id(value))
    if packed is not None:
        return packed
    items = cast(Sequence[object], value)
    if len(items) < _VECTOR_LENGTH or type(items[0]) is not float:
        return None
    floats = cast(Sequence[float], items)
    if countOf(map(type, items), float) != len(items) or not math.isfinite(sum(floats)):
        return None
    packed = found[id(value)] = array("d", floats).tobytes()
    return packed


@cache
def _str_key_type(kind: type[object]) -> bool:
    """Tell whether keys of this type sort and encode as the text they hold.

    SQLAlchemy's ``quoted_name`` column keys do: ``_CANONICAL`` compares keys with
    their ``==`` and ``<`` and writes their text.
    """
    return issubclass(kind, str) and kind.__eq__ is str.__eq__ and kind.__lt__ is str.__lt__


@cache
def _text_key(kind: type[object]) -> bool:
    """Tell whether keys of this type encode, sort, compare and hash as the text they hold.

    Exact ``str`` does, and so do SQLAlchemy's ``quoted_name`` keys of a decoded
    physical row (``_str_key_type``) while they keep ``str``'s hash.
    """
    return kind is str or (_str_key_type(kind) and kind.__hash__ is str.__hash__)


def _holds_vector(value: object, depth: int, path: set[int], found: dict[int, bytes]) -> bool:
    """Tell whether a vector lies in ``value`` through lists, tuples and dicts keyed by text.

    ``path`` holds the containers being encoded around ``value``: a cycle is left
    to ``_CANONICAL``, which refuses it.
    """
    kind = type(value)
    children: Iterable[object]
    if kind is list or kind is tuple:
        if _vector_bytes(value, found) is not None:
            return True
        children = cast(Sequence[object], value)
    elif kind is dict:
        mapping = cast(dict[object, object], value)
        if not all(type(key) is str or _str_key_type(type(key)) for key in mapping):
            return False
        children = mapping.values()
    else:
        return False
    if depth >= _VECTOR_DEPTH or id(value) in path:
        return False
    path.add(id(value))
    try:
        for child in children:
            inner = type(child)
            if (inner is list or inner is tuple or inner is dict) and _holds_vector(
                child, depth + 1, path, found,
            ):
                return True
        return False
    finally:
        path.remove(id(value))


def _write_holding(
    value: object, depth: int, path: set[int], vectors: dict[bytes, str],
    found: dict[int, bytes], chunks: list[str],
) -> None:
    """Write a vector, or a container holding one, as ``_CANONICAL`` writes it.

    A vector's encoding is kept under its bytes. A container is written as the C
    encoder writes it: a dict's keys sorted, then ``,`` and ``:`` separators.
    Like the C encoder, the pieces are joined once, at the end.
    """
    packed = found.get(id(value))
    if packed is not None:
        encoded = vectors.get(packed)
        if encoded is None:
            encoded = vectors[packed] = _CANONICAL.encode(value)
        chunks.append(encoded)
        return
    container = id(value)
    path.add(container)
    try:
        # A container holding a vector is never empty.
        if type(value) is dict:
            mapping = cast(dict[str, object], value)
            separator = "{"
            for key in sorted(mapping):
                chunks.append(separator)
                chunks.append(encode_basestring(key))
                chunks.append(":")
                _write_member(mapping[key], depth + 1, path, vectors, found, chunks)
                separator = ","
            chunks.append("}")
        else:
            separator = "["
            for item in cast(Sequence[object], value):
                chunks.append(separator)
                _write_member(item, depth + 1, path, vectors, found, chunks)
                separator = ","
            chunks.append("]")
    finally:
        path.remove(container)


def _write_member(
    value: object, depth: int, path: set[int], vectors: dict[bytes, str],
    found: dict[int, bytes], chunks: list[str],
) -> None:
    """Write one member of a container that holds a vector, as ``_CANONICAL`` does.

    Members are written in ``_CANONICAL``'s order, so an invalid one raises what
    it raises there.
    """
    kind = type(value)
    if kind is str:
        chunks.append(encode_basestring(cast(str, value)))
    elif kind is int:
        chunks.append(int.__repr__(cast(int, value)))
    elif value is None:
        chunks.append("null")
    elif (kind is list or kind is tuple or kind is dict) and _holds_vector(
        value, depth, path, found,
    ):
        _write_holding(value, depth, path, vectors, found, chunks)
    else:
        chunks.append(_CANONICAL.encode(value))


def integrity_digest(value: object) -> IntegrityDigest:
    """Hash canonical content only where the plan permits unkeyed integrity."""
    return IntegrityDigest(hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest())


def base64url(value: bytes) -> str:
    """Encode bytes without padding, using the URL-safe alphabet."""
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def artifact_id(envelope: object) -> ArtifactId:
    """Address the canonical envelope."""
    return ArtifactId(
        "art_" + base64url(hashlib.sha256(canonical_json(envelope).encode("utf-8")).digest())
    )


def control_id(key: str) -> str:
    """Derive the technical Control primary key from its complete namespace key."""
    return "ctl_" + base64url(hashlib.sha256(key.encode("utf-8")).digest())


def scope_id(scope_kind: str, raw: str) -> str:
    """Minimize an external scope identity using the approved canonical tuple."""
    return (
        "sc_"
        + base64url(hashlib.sha256(canonical_json([scope_kind, raw]).encode("utf-8")).digest())[:40]
    )


def new_ulid() -> str:
    """Create a 128-bit ULID using current milliseconds and cryptographic entropy."""
    value = ((time.time_ns() // 1_000_000) << 80) | secrets.randbits(80)
    alphabet = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
    return "".join(alphabet[(value >> shift) & 31] for shift in range(125, -1, -5))
