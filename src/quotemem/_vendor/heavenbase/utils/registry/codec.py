"""Encode and validate canonical Registry snapshots."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Any, Generic, TypeVar

from .errors import RegistryIntegrityError

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")

_CODEC_VERSION = 1


def _identity(value: Any) -> Any:
    return value


def _require_json_value(value: Any, path: str = "payload") -> None:
    if value is None or isinstance(value, (bool, int, float, str)):
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _require_json_value(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} contains non-string mapping key {key!r}")
            _require_json_value(item, f"{path}.{key}")
        return
    raise TypeError(f"{path} contains non-JSON value {value!r}")


def _default_encode_key(value: Hashable) -> Any:
    if isinstance(value, str):
        return {"type": "str", "value": value}
    if isinstance(value, tuple):
        return {"type": "tuple", "value": [_default_encode_key(item) for item in value]}
    raise TypeError(f"Persistent Registry keys must be strings or recursive tuples, got {value!r}")


def _default_decode_key(value: Any) -> Hashable:
    if not isinstance(value, Mapping):
        raise RegistryIntegrityError(f"Registry key payload must be a mapping, got {value!r}")
    kind = value.get("type")
    data = value.get("value")
    if kind == "str" and isinstance(data, str):
        return data
    if kind == "tuple" and isinstance(data, list):
        return tuple(_default_decode_key(item) for item in data)
    raise RegistryIntegrityError(f"Unknown or invalid Registry key payload {value!r}")


def _pairs(data: Mapping[str, Any], name: str) -> list[list[Any]]:
    values = data.get(name)
    if not isinstance(values, list) or any(not isinstance(item, list) or len(item) != 2 for item in values):
        raise RegistryIntegrityError(f"Registry payload {name!r} must contain pairs")
    return values


def _sequence(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise RegistryIntegrityError(f"Registry payload {name!r} values must be lists")
    return value


def _codec_pair(
    encode: Callable[[Any], Any] | None,
    decode: Callable[[Any], Any] | None,
    *,
    default_encode: Callable[[Any], Any],
    default_decode: Callable[[Any], Any],
    name: str,
) -> tuple[Callable[[Any], Any], Callable[[Any], Any]]:
    if (encode is None) != (decode is None):
        raise TypeError(f"Registry {name} encoder and decoder must be supplied together")
    if encode is None:
        return default_encode, default_decode
    assert decode is not None
    return encode, decode


def _resolve_hash(hash_fn: str | Callable[[bytes], str] | None) -> Callable[[bytes], str] | None:
    if hash_fn is None:
        return None
    if callable(hash_fn):
        return hash_fn
    if hash_fn == "sha256":
        return lambda payload: sha256(payload).hexdigest()
    raise ValueError(f"Unsupported Registry hash function {hash_fn!r}; expected 'sha256', callable, or None")


@dataclass(frozen=True)
class _DecodedState(Generic[K, V]):
    items: dict[K, V]
    aliases: dict[K, K]
    alias_names: dict[K, tuple[K, ...]]
    groups: dict[Hashable, tuple[K, ...]]
    group_names: dict[Hashable, Hashable]


class _RegistryCodec(Generic[K, V]):
    """Encode, decode, and verify canonical Registry snapshot payloads."""

    def __init__(
        self,
        identifier: str,
        *,
        encode_key: Callable[[K], Any] | None,
        decode_key: Callable[[Any], K] | None,
        encode_value: Callable[[V], Any] | None,
        decode_value: Callable[[Any], V] | None,
        hash_fn: str | Callable[[bytes], str] | None,
        max_bytes: int | None,
    ) -> None:
        """Initialize one Registry snapshot codec.

        Args:
            identifier (str): Registry namespace identifier embedded in digests.
            encode_key (Callable[[K], Any] | None): Optional JSON-safe key encoder.
            decode_key (Callable[[Any], K] | None): Optional key decoder.
            encode_value (Callable[[V], Any] | None): Optional JSON-safe value encoder.
            decode_value (Callable[[Any], V] | None): Optional value decoder.
            hash_fn (str | Callable[[bytes], str] | None): Integrity hash selector.
            max_bytes (int | None): Optional encoded snapshot byte limit.

        Returns:
            None: This function does not return a value.
        """
        if max_bytes is not None and (not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes < 1):
            raise ValueError(f"max_bytes must be a positive integer or None, got {max_bytes!r}")
        self.identifier = identifier
        self.encode_key, self.decode_key = _codec_pair(
            encode_key,
            decode_key,
            default_encode=_default_encode_key,
            default_decode=_default_decode_key,
            name="key",
        )
        self.encode_value, self.decode_value = _codec_pair(
            encode_value,
            decode_value,
            default_encode=_identity,
            default_decode=_identity,
            name="value",
        )
        self.hash_fn = _resolve_hash(hash_fn)
        self.max_bytes = max_bytes

    def _encoded_identity(self, value: Hashable) -> Any:
        try:
            return self.encode_key(value)  # type: ignore[arg-type]
        except Exception as error:
            raise TypeError(f"Registry {self.identifier!r} could not encode identity {value!r}: {error}") from error

    def _decoded_identity(self, value: Any) -> K:
        try:
            decoded = self.decode_key(value)
            hash(decoded)
        except Exception as error:
            raise RegistryIntegrityError(f"Registry {self.identifier!r} contains an invalid encoded identity") from error
        return decoded

    def encode(
        self,
        items: Mapping[K, V],
        aliases: Mapping[K, K],
        alias_names: Mapping[K, tuple[K, ...]],
        groups: Mapping[Hashable, tuple[K, ...]],
        group_names: Mapping[Hashable, Hashable],
    ) -> bytes:
        """Encode one complete Registry snapshot to canonical JSON bytes.

        Args:
            items (Mapping[K, V]): Canonical items to persist.
            aliases (Mapping[K, K]): Alias lookup index.
            alias_names (Mapping[K, tuple[K, ...]]): Display aliases by canonical key.
            groups (Mapping[Hashable, tuple[K, ...]]): Group membership index.
            group_names (Mapping[Hashable, Hashable]): Display group names.

        Returns:
            bytes: Canonical UTF-8 JSON snapshot payload.
        """
        payload = {
            "codec": _CODEC_VERSION,
            "items": [[self._encoded_identity(key), self.encode_value(value)] for key, value in items.items()],
            "aliases": [[self._encoded_identity(alias), self._encoded_identity(target)] for alias, target in aliases.items()],
            "alias_names": [[self._encoded_identity(key), [self._encoded_identity(alias) for alias in values]] for key, values in alias_names.items()],
            "groups": [[self._encoded_identity(group), [self._encoded_identity(key) for key in keys]] for group, keys in groups.items()],
            "group_names": [[self._encoded_identity(group), self._encoded_identity(display)] for group, display in group_names.items()],
        }
        try:
            _require_json_value(payload)
            encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
        except Exception as error:
            raise TypeError(f"Registry {self.identifier!r} snapshot is not JSON encodable") from error
        if self.max_bytes is not None and len(encoded) > self.max_bytes:
            raise OverflowError(f"Registry {self.identifier!r} encoded snapshot is {len(encoded)} bytes; capacity is {self.max_bytes}")
        return encoded

    def decode(self, payload: object) -> _DecodedState[K, V]:
        """Decode one persisted Registry snapshot payload.

        Args:
            payload (object): UTF-8 JSON bytes or text snapshot payload.

        Returns:
            _DecodedState[K, V]: Decoded items, aliases, and group indexes.

        Raises:
            RegistryIntegrityError: If the payload is corrupt or inconsistent.
        """
        if isinstance(payload, memoryview):
            payload = payload.tobytes()
        if isinstance(payload, bytes):
            try:
                text = payload.decode("utf-8")
            except UnicodeDecodeError as error:
                raise RegistryIntegrityError(f"Registry {self.identifier!r} payload is not UTF-8") from error
        elif isinstance(payload, str):
            text = payload
        else:
            raise RegistryIntegrityError(f"Registry {self.identifier!r} payload must be bytes")
        if self.max_bytes is not None and len(text.encode("utf-8")) > self.max_bytes:
            raise RegistryIntegrityError(f"Registry {self.identifier!r} persisted payload exceeds max_bytes={self.max_bytes}")
        try:
            data = json.loads(text)
        except Exception as error:
            raise RegistryIntegrityError(f"Registry {self.identifier!r} payload is not valid JSON") from error
        if not isinstance(data, Mapping) or data.get("codec") != _CODEC_VERSION:
            found = data.get("codec") if isinstance(data, Mapping) else None
            raise RegistryIntegrityError(f"Registry {self.identifier!r} uses unsupported codec version {found!r}")
        try:
            item_pairs = _pairs(data, "items")
            alias_pairs = _pairs(data, "aliases")
            alias_name_pairs = _pairs(data, "alias_names")
            group_pairs = _pairs(data, "groups")
            group_name_pairs = _pairs(data, "group_names")
            items = {self._decoded_identity(key): self.decode_value(value) for key, value in item_pairs}
            aliases = {self._decoded_identity(alias): self._decoded_identity(target) for alias, target in alias_pairs}
            alias_names = {
                self._decoded_identity(key): tuple(self._decoded_identity(alias) for alias in _sequence(values, "alias_names"))
                for key, values in alias_name_pairs
            }
            groups = {self._decoded_identity(group): tuple(self._decoded_identity(key) for key in _sequence(values, "groups")) for group, values in group_pairs}
            group_names = {self._decoded_identity(group): self._decoded_identity(display) for group, display in group_name_pairs}
            if any(
                len(values) != len(pairs)
                for values, pairs in (
                    (items, item_pairs),
                    (aliases, alias_pairs),
                    (alias_names, alias_name_pairs),
                    (groups, group_pairs),
                    (group_names, group_name_pairs),
                )
            ):
                raise RegistryIntegrityError(f"Registry {self.identifier!r} payload contains duplicate identities")
        except RegistryIntegrityError:
            raise
        except Exception as error:
            raise RegistryIntegrityError(f"Registry {self.identifier!r} payload has invalid state data") from error
        decoded = _DecodedState(items, aliases, alias_names, groups, group_names)
        self._validate(decoded)
        return decoded

    def _validate(self, state: _DecodedState[K, V]) -> None:
        item_keys = set(state.items)
        if any(alias in item_keys or target not in item_keys for alias, target in state.aliases.items()):
            raise RegistryIntegrityError(f"Registry {self.identifier!r} contains invalid alias targets")
        if set(state.alias_names) != item_keys:
            raise RegistryIntegrityError(f"Registry {self.identifier!r} alias metadata does not match canonical keys")
        display_aliases = [alias for aliases in state.alias_names.values() for alias in aliases]
        if len(display_aliases) != len(set(display_aliases)):
            raise RegistryIntegrityError(f"Registry {self.identifier!r} alias metadata contains duplicate identities")
        if any(len(keys) != len(set(keys)) or any(key not in item_keys for key in keys) for keys in state.groups.values()):
            raise RegistryIntegrityError(f"Registry {self.identifier!r} contains groups with unknown keys")
        if set(state.group_names) != set(state.groups):
            raise RegistryIntegrityError(f"Registry {self.identifier!r} group names do not match group membership")

    def digest(self, revision: int, payload: bytes) -> str:
        """Return the integrity digest for one revision and payload.

        Args:
            revision (int): Persistent revision number.
            payload (bytes): Canonical encoded snapshot bytes.

        Returns:
            str: Hex digest string, or ``""`` when hashing is disabled.
        """
        if self.hash_fn is None:
            return ""
        digest_input = b"\0".join(
            (
                self.identifier.encode("utf-8"),
                str(revision).encode("ascii"),
                str(_CODEC_VERSION).encode("ascii"),
                payload,
            )
        )
        value = self.hash_fn(digest_input)
        if not isinstance(value, str) or not value:
            raise TypeError("Registry hash function must return a non-empty string")
        return value

    def verify(self, revision: int, payload: bytes, digest: object) -> None:
        """Verify one persisted digest against a revision and payload.

        Args:
            revision (int): Persistent revision number.
            payload (bytes): Canonical encoded snapshot bytes.
            digest (object): Stored digest value to compare.

        Returns:
            None: This function does not return a value.

        Raises:
            RegistryIntegrityError: If hashing is enabled and the digest mismatches.
        """
        if self.hash_fn is not None and digest != self.digest(revision, payload):
            raise RegistryIntegrityError(f"Registry {self.identifier!r} digest does not match its payload")
