"""Serialization helpers for local files and deterministic payloads."""

from __future__ import annotations

__all__ = [
    "HbJsonDecoder",
    "HbJsonEncoder",
    "append_jsonl",
    "append_txt",
    "dump_b64",
    "dump_bin",
    "dump_hex",
    "dump_jstr",
    "dump_json",
    "dump_jsonl",
    "dump_pkl",
    "dump_pickle",
    "dump_txt",
    "dump_yaml",
    "dumps_b64",
    "dumps_jstr",
    "dumps_json",
    "dumps_jsonl",
    "dumps_pickle",
    "dumps_yaml",
    "iter_jsonl",
    "iter_txt",
    "load_b64",
    "load_bin",
    "load_hex",
    "load_jstr",
    "load_json",
    "load_jsonl",
    "load_pkl",
    "load_pickle",
    "load_txt",
    "load_yaml",
    "loads_b64",
    "loads_jstr",
    "loads_json",
    "loads_jsonl",
    "loads_pickle",
    "loads_yaml",
    "save_b64",
    "save_bin",
    "save_hex",
    "save_jstr",
    "save_json",
    "save_jsonl",
    "save_pkl",
    "save_pickle",
    "save_txt",
    "save_yaml",
    "validate_json",
]

import base64
import binascii
import dataclasses
import datetime as dt
import json
import pickle
from decimal import Decimal
from pathlib import Path

from .config import CM_HVNB
from .files import exists_file, get_file_dir, pj, touch_dir
from .typing import Any, Dict, Generator, Iterable, Iterator, List, Optional, Union


def _cfg(key: str, default: Any) -> Any:
    # Serialization never fails over config: an unopenable home or explicit
    # Context mode falls back to the packaged value, and any other config
    # error to ``default``, as in 0.1.2.2.
    try:
        return CM_HVNB._get_or_packaged(f"heavenbase.serialize.{key}", default)
    except Exception:
        return default


def _encoding(encoding: Optional[str]) -> str:
    return str(_cfg("encoding", "utf-8") if encoding is None else encoding)


def _indent(indent: Optional[int]) -> Optional[int]:
    configured = _cfg("indent", 4) if indent is None else indent
    return None if configured is None else int(configured)


def _json_ready(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return _json_ready(dataclasses.asdict(obj))
    if isinstance(obj, tuple):
        return {"__obj_type__": "tuple", "__obj_data__": [_json_ready(item) for item in obj]}
    if isinstance(obj, set):
        return {"__obj_type__": "set", "__obj_data__": [_json_ready(item) for item in sorted(obj, key=repr)]}
    if isinstance(obj, dt.datetime):
        return {"__obj_type__": "datetime", "__obj_data__": obj.timestamp()}
    if isinstance(obj, dt.date):
        return {"__obj_type__": "date", "__obj_data__": dt.datetime(obj.year, obj.month, obj.day).timestamp()}
    if isinstance(obj, Decimal):
        return {"__obj_type__": "decimal", "__obj_data__": str(obj)}
    if isinstance(obj, int) and (obj > 1 << 53 or obj < -(1 << 53)):
        return {"__obj_type__": "bigint", "__obj_data__": str(obj)}
    if obj is Ellipsis:
        return {"__obj_type__": "ellipsis"}
    if isinstance(obj, list):
        return [_json_ready(item) for item in obj]
    if isinstance(obj, dict):
        return {str(key): _json_ready(value) for key, value in obj.items()}
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "to_dict") and callable(getattr(obj, "to_dict")):
        return _json_ready(obj.to_dict())
    if hasattr(obj, "tolist") and callable(getattr(obj, "tolist")):
        return _json_ready(obj.tolist())
    if callable(obj) and hasattr(obj, "__module__") and hasattr(obj, "__qualname__"):
        return {"__obj_type__": "callable-ref", "__obj_data__": f"{obj.__module__}.{obj.__qualname__}"}
    return obj


def _json_restore(obj: Any) -> Any:
    if isinstance(obj, list):
        return [_json_restore(item) for item in obj]
    if not isinstance(obj, dict):
        return obj
    obj_type = obj.get("__obj_type__")
    if obj_type == "tuple":
        return tuple(_json_restore(item) for item in obj.get("__obj_data__", []))
    if obj_type == "set":
        return set(_json_restore(item) for item in obj.get("__obj_data__", []))
    if obj_type == "datetime":
        return dt.datetime.fromtimestamp(obj["__obj_data__"])
    if obj_type == "date":
        value = dt.datetime.fromtimestamp(obj["__obj_data__"])
        return dt.date(value.year, value.month, value.day)
    if obj_type == "bigint":
        return int(obj["__obj_data__"])
    if obj_type == "decimal":
        return Decimal(str(obj["__obj_data__"]))
    if obj_type == "ellipsis":
        return ...
    if obj_type == "callable-ref":
        return obj.get("__obj_data__")
    return {key: _json_restore(value) for key, value in obj.items()}


class HbJsonEncoder(json.JSONEncoder):
    """JSON encoder that handles common HeavenBase values.

    ``json.dumps(obj, cls=HbJsonEncoder)`` and ``json.dump(obj, fp,
    cls=HbJsonEncoder)`` write the same tagged text, which
    :class:`HbJsonDecoder` restores.
    """

    def iterencode(self, o: Any, _one_shot: bool = False) -> Iterator[str]:
        """
        Encode ``o`` in chunks after adapting HeavenBase values.

        ``json.dump`` calls this method directly and ``encode`` (used by
        ``json.dumps``) calls it too, so both tag tuples, sets, datetimes,
        dates, Decimals, big integers and ``Ellipsis`` alike.

        Args:
            o (Any): Value to serialize.
            _one_shot (bool): Standard-library flag that permits the C
                encoder's single-pass mode. Defaults to ``False``.

        Returns:
            Iterator[str]: JSON text chunks.
        """
        return super().iterencode(_json_ready(o), _one_shot)


class HbJsonDecoder(json.JSONDecoder):
    """JSON decoder for values emitted by `HbJsonEncoder`."""

    def decode(self, s: str, **kwargs: Any) -> Any:
        """
        Use ``decode`` in the HeavenBase utility surface.

        Use this helper for local JSON, YAML, pickle, text, binary, and JSONL artifacts.

        Args:
            s (str): Serialized string content to parse.
            **kwargs (Any): Extra keyword arguments forwarded to the underlying serializer, command, or callable.

        Returns:
            Any: The result of ``decode``.
        """
        return _json_restore(super().decode(s, **kwargs))


def load_txt(path: str, encoding: Optional[str] = None, strict: bool = False) -> str:
    """
    Load text from a file, returning an empty string when missing unless strict.

    Use this helper for local JSON, YAML, pickle, text, binary, and JSONL artifacts.

    Args:
        path (str): Local path, config path, or dotted dictionary path used by the helper.
        encoding (Optional[str]): Text encoding. ``None`` reads ``heavenbase.serialize.encoding``; the default config is ``"utf-8"``. Defaults to ``None``.
        strict (bool): Whether missing files should raise ``FileNotFoundError``. Defaults to ``False``.

    Returns:
        str: File text, or ``""`` when missing and ``strict=False``.
    """
    value = CM_HVNB.pj(path, abs=True)
    if not exists_file(value):
        if strict:
            raise FileNotFoundError(value)
        return ""
    return Path(value).read_text(encoding=_encoding(encoding), errors="ignore")


def iter_txt(path: str, encoding: Optional[str] = None, strict: bool = False) -> Generator[str, None, None]:
    """Yield file lines without trailing newlines.

    Use this helper for local JSON, YAML, pickle, text, binary, and JSONL artifacts.

    Args:
        path (str): Local path, config path, or dotted dictionary path used by the helper.
        encoding (Optional[str]): Text encoding. ``None`` reads ``heavenbase.serialize.encoding``; the default config is ``"utf-8"``. Defaults to ``None``.
        strict (bool): Whether missing files should raise ``FileNotFoundError``. Defaults to ``False``.

    Yields:
        str: Lines without trailing newlines.
    """
    text = load_txt(path, encoding=encoding, strict=strict)
    for line in text.splitlines():
        yield line


def save_txt(obj: Any, path: str, encoding: Optional[str] = None) -> None:
    """
    Write text plus a trailing newline.

    Use this helper for local JSON, YAML, pickle, text, binary, and JSONL artifacts.

    Args:
        obj (Any): Value to serialize, format, print, copy, or pass through the helper.
        path (str): Local path, config path, or dotted dictionary path used by the helper.
        encoding (Optional[str]): Text encoding. ``None`` reads ``heavenbase.serialize.encoding``; the default config is ``"utf-8"``. Defaults to ``None``.

    Returns:
        None: This function returns ``None`` after completing its side effect.
    """
    touch_dir(get_file_dir(CM_HVNB.pj(path, abs=True)))
    Path(CM_HVNB.pj(path, abs=True)).write_text(f"{obj}\n", encoding=_encoding(encoding), errors="ignore")


dump_txt = save_txt


def append_txt(obj: Any, path: str, encoding: Optional[str] = None) -> None:
    """
    Append text plus a trailing newline.

    Use this helper for local JSON, YAML, pickle, text, binary, and JSONL artifacts.

    Args:
        obj (Any): Value to serialize, format, print, copy, or pass through the helper.
        path (str): Local path, config path, or dotted dictionary path used by the helper.
        encoding (Optional[str]): Text encoding. ``None`` reads ``heavenbase.serialize.encoding``; the default config is ``"utf-8"``. Defaults to ``None``.

    Returns:
        None: This function returns ``None`` after completing its side effect.
    """
    touch_dir(get_file_dir(CM_HVNB.pj(path, abs=True)))
    with open(CM_HVNB.pj(path, abs=True), "a", encoding=_encoding(encoding), errors="ignore") as handle:
        handle.write(f"{obj}\n")


def loads_yaml(s: str, **kwargs: Any) -> Any:
    """
    Load YAML from a string.

    Use this helper for local JSON, YAML, pickle, text, binary, and JSONL artifacts.

    Args:
        s (str): Serialized string content to parse.
        **kwargs (Any): Extra keyword arguments forwarded to the underlying serializer, command, or callable.

    Returns:
        Any: The result of ``loads_yaml``.
    """
    import yaml

    loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
    return yaml.load(s, Loader=loader, **kwargs)


def dumps_yaml(obj: Any, sort_keys: bool = False, indent: Optional[int] = None, allow_unicode: bool = True, **kwargs: Any) -> str:
    """
    Dump YAML to a string.

    Use this helper for local JSON, YAML, pickle, text, binary, and JSONL artifacts.

    Args:
        obj (Any): Value to serialize, format, print, copy, or pass through the helper.
        sort_keys (bool): Whether serialized mapping keys should be sorted. Defaults to ``False``.
        indent (Optional[int]): Indentation width. ``None`` reads ``heavenbase.serialize.indent``; the default config is ``4``. Defaults to ``None``.
        allow_unicode (bool): Whether YAML output may contain Unicode characters. Defaults to ``True``.
        **kwargs (Any): Extra keyword arguments forwarded to the underlying serializer, command, or callable.

    Returns:
        str: The result of ``dumps_yaml``.
    """
    import yaml

    return yaml.safe_dump(obj, sort_keys=sort_keys, indent=_indent(indent), allow_unicode=allow_unicode, **kwargs)


def load_yaml(path: str, encoding: Optional[str] = None, strict: bool = False, **kwargs: Any) -> Any:
    """
    Load a YAML file, returning an empty dict when missing unless strict.

    Use this helper for local JSON, YAML, pickle, text, binary, and JSONL artifacts.

    Args:
        path (str): Local path, config path, or dotted dictionary path used by the helper.
        encoding (Optional[str]): Text encoding. ``None`` reads ``heavenbase.serialize.encoding``; the default config is ``"utf-8"``. Defaults to ``None``.
        strict (bool): Whether missing files should raise ``FileNotFoundError``. Defaults to ``False``.
        **kwargs (Any): Extra keyword arguments forwarded to the underlying serializer, command, or callable.

    Returns:
        Any: YAML value, or ``{}`` when missing and ``strict=False``.
    """
    value = CM_HVNB.pj(path, abs=True)
    if not exists_file(value):
        if strict:
            raise FileNotFoundError(value)
        return {}
    return loads_yaml(Path(value).read_text(encoding=_encoding(encoding), errors="ignore"), **kwargs)


def dump_yaml(obj: Any, path: str, encoding: Optional[str] = None, **kwargs: Any) -> None:
    """
    Write a YAML file.

    Use this helper for local JSON, YAML, pickle, text, binary, and JSONL artifacts.

    Args:
        obj (Any): Value to serialize, format, print, copy, or pass through the helper.
        path (str): Local path, config path, or dotted dictionary path used by the helper.
        encoding (Optional[str]): Text encoding. ``None`` reads ``heavenbase.serialize.encoding``; the default config is ``"utf-8"``. Defaults to ``None``.
        **kwargs (Any): Extra keyword arguments forwarded to the underlying serializer, command, or callable.

    Returns:
        None: This function returns ``None`` after completing its side effect.
    """
    save_txt(dumps_yaml(obj, **kwargs).rstrip("\n"), path, encoding=encoding)


save_yaml = dump_yaml


def load_pkl(path: str, strict: bool = False, **kwargs: Any) -> Any:
    """Load a trusted local pickle file, returning ``None`` for a missing non-strict file.

    Use this helper for local JSON, YAML, pickle, text, binary, and JSONL artifacts.

    Warnings:
        Pickle loading can execute Python object reconstruction. Only load files produced by a trusted environment.

    Args:
        path (str): Local path, config path, or dotted dictionary path used by the helper.
        strict (bool): Whether missing files should raise ``FileNotFoundError``. Defaults to ``False``.
        **kwargs (Any): Extra keyword arguments forwarded to the underlying serializer, command, or callable.

    Returns:
        Any: Pickled object, or ``None`` when missing and ``strict=False``.
    """
    value = CM_HVNB.pj(path, abs=True)
    if not exists_file(value):
        if strict:
            raise FileNotFoundError(value)
        return None
    with open(value, "rb") as handle:
        return pickle.load(handle, **kwargs)


load_pickle = load_pkl


def loads_pickle(data: bytes, **kwargs: Any) -> Any:
    """Load pickle data from bytes.

    Use this helper for local JSON, YAML, pickle, text, binary, and JSONL artifacts.

    Warnings:
        Pickle loading can execute Python object reconstruction. Only load bytes produced by a trusted environment.

    Args:
        data (bytes): Dictionary layer or payload to read, write, merge, or render.
        **kwargs (Any): Extra keyword arguments forwarded to the underlying serializer, command, or callable.

    Returns:
        Any: The result of ``loads_pickle``.
    """
    return pickle.loads(data, **kwargs)


def dumps_pickle(obj: Any, **kwargs: Any) -> bytes:
    """
    Dump pickle data to bytes.

    Use this helper for local JSON, YAML, pickle, text, binary, and JSONL artifacts.

    Args:
        obj (Any): Value to serialize, format, print, copy, or pass through the helper.
        **kwargs (Any): Extra keyword arguments forwarded to the underlying serializer, command, or callable.

    Returns:
        bytes: The result of ``dumps_pickle``.
    """
    return pickle.dumps(obj, **kwargs)


def dump_pkl(obj: Any, path: str, **kwargs: Any) -> None:
    """
    Write a trusted local Python object to a pickle file.

    Use this helper for local JSON, YAML, pickle, text, binary, and JSONL artifacts.

    Args:
        obj (Any): Value to serialize, format, print, copy, or pass through the helper.
        path (str): Local path, config path, or dotted dictionary path used by the helper.
        **kwargs (Any): Extra keyword arguments forwarded to the underlying serializer, command, or callable.

    Returns:
        None: This function returns ``None`` after completing its side effect.
    """
    touch_dir(get_file_dir(CM_HVNB.pj(path, abs=True)))
    with open(CM_HVNB.pj(path, abs=True), "wb") as handle:
        pickle.dump(obj, handle, **kwargs)


save_pkl = dump_pkl
dump_pickle = dump_pkl
save_pickle = dump_pkl


def load_bin(path: str, strict: bool = False) -> bytes:
    """
    Load bytes from a file, returning empty bytes when missing unless strict.

    Use this helper for local JSON, YAML, pickle, text, binary, and JSONL artifacts.

    Args:
        path (str): Local path, config path, or dotted dictionary path used by the helper.
        strict (bool): Whether missing files should raise ``FileNotFoundError``. Defaults to ``False``.

    Returns:
        bytes: File bytes, or ``b""`` when missing and ``strict=False``.
    """
    value = CM_HVNB.pj(path, abs=True)
    if not exists_file(value):
        if strict:
            raise FileNotFoundError(value)
        return b""
    return Path(value).read_bytes()


def dump_bin(obj: bytes, path: str) -> None:
    """
    Write bytes to a file.

    Use this helper for local JSON, YAML, pickle, text, binary, and JSONL artifacts.

    Args:
        obj (bytes): Value to serialize, format, print, copy, or pass through the helper.
        path (str): Local path, config path, or dotted dictionary path used by the helper.

    Returns:
        None: This function returns ``None`` after completing its side effect.
    """
    touch_dir(get_file_dir(CM_HVNB.pj(path, abs=True)))
    Path(CM_HVNB.pj(path, abs=True)).write_bytes(obj)


save_bin = dump_bin


def load_hex(path: str, strict: bool = False, **kwargs: Any) -> str:
    """
    Load a binary file as a hex string.

    Use this helper for local JSON, YAML, pickle, text, binary, and JSONL artifacts.

    Args:
        path (str): Local path, config path, or dotted dictionary path used by the helper.
        strict (bool): Whether missing files should raise ``FileNotFoundError``. Defaults to ``False``.
        **kwargs (Any): Extra keyword arguments forwarded to the underlying serializer, command, or callable.

    Returns:
        str: Hex string for file bytes, or ``""`` for missing non-strict files.
    """
    data = load_bin(path, strict=strict)
    return binascii.hexlify(data, **kwargs).decode("ascii") if data else ""


def dump_hex(obj: Union[str, bytes], path: str, **kwargs: Any) -> None:
    """
    Decode a hex string and write the resulting bytes.

    Use this helper for local JSON, YAML, pickle, text, binary, and JSONL artifacts.

    Args:
        obj (Union[str, bytes]): Value to serialize, format, print, copy, or pass through the helper.
        path (str): Local path, config path, or dotted dictionary path used by the helper.
        **kwargs (Any): Extra keyword arguments forwarded to the underlying serializer, command, or callable.

    Returns:
        None: This function returns ``None`` after completing its side effect.
    """
    text = obj.decode("ascii") if isinstance(obj, bytes) else str(obj)
    dump_bin(binascii.unhexlify(text.encode("ascii"), **kwargs), path)


save_hex = dump_hex


def dumps_b64(data: bytes) -> str:
    """Encode bytes to a standard base64 ASCII string.

    Args:
        data (bytes): Raw bytes to encode.

    Returns:
        str: Standard base64 ASCII string.
    """
    return base64.b64encode(data).decode("ascii")


def loads_b64(text: str, *, validate: bool = True) -> bytes:
    """Decode a standard base64 ASCII string to bytes.

    Args:
        text (str): Base64 ASCII string.
        validate (bool): Whether invalid characters raise. Defaults to ``True``.

    Returns:
        bytes: Decoded byte payload.

    Raises:
        ValueError: If ``text`` is not valid base64.
    """
    try:
        return base64.b64decode(text, validate=validate)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"Invalid base64 payload: {text!r}") from exc


def load_b64(path: str, strict: bool = False) -> str:
    """
    Load a binary file as a Base64 string.

    Use this helper for local JSON, YAML, pickle, text, binary, and JSONL artifacts.

    Args:
        path (str): Local path, config path, or dotted dictionary path used by the helper.
        strict (bool): Whether missing files should raise ``FileNotFoundError``. Defaults to ``False``.

    Returns:
        str: Base64 string for file bytes, or ``""`` for missing non-strict files.
    """
    data = load_bin(path, strict=strict)
    return dumps_b64(data) if data else ""


def dump_b64(obj: Union[str, bytes], path: str) -> None:
    """
    Decode a Base64 string and write the resulting bytes.

    Use this helper for local JSON, YAML, pickle, text, binary, and JSONL artifacts.

    Args:
        obj (Union[str, bytes]): Value to serialize, format, print, copy, or pass through the helper.
        path (str): Local path, config path, or dotted dictionary path used by the helper.

    Returns:
        None: This function returns ``None`` after completing its side effect.
    """
    text = obj.decode("ascii") if isinstance(obj, bytes) else str(obj)
    dump_bin(loads_b64(text, validate=False), path)


save_b64 = dump_b64


def loads_json(s: str, restore: bool = True, **kwargs: Any) -> Any:
    """
    Load JSON from a string.

    Use this helper for local JSON, YAML, pickle, text, binary, and JSONL artifacts.

    Args:
        s (str): Serialized string content to parse.
        restore (bool): Whether to restore HeavenBase tagged values such as
            tuples and large integers. Set to ``False`` at strict JSON boundaries.
            Defaults to ``True``.
        **kwargs (Any): Extra keyword arguments forwarded to the underlying serializer, command, or callable.

    Returns:
        Any: The result of ``loads_json``.
    """
    value = json.loads(s, **kwargs)
    return _json_restore(value) if restore else value


def _validate_json_keys(value: Any) -> None:
    """Reject object keys that the standard encoder would silently stringify."""
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("JSON object keys must be strings")
        for item in value.values():
            _validate_json_keys(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _validate_json_keys(item)


def validate_json(obj: Any) -> None:
    """Validate one value with the strict standard JSON encoder.

    Unlike :func:`dumps_json`, this boundary performs no HeavenBase type
    adaptation. Non-finite floats and values the standard encoder cannot
    represent raise their native ``TypeError`` or ``ValueError``.

    Args:
        obj (Any): Candidate JSON request value.

    Returns:
        None: The function returns after successful validation.
    """
    _validate_json_keys(obj)
    json.dumps(obj, allow_nan=False)


def dumps_json(
    obj: Any,
    sort_keys: bool = False,
    indent: Optional[int] = None,
    ensure_ascii: bool = False,
    compact: bool = False,
    adapt: bool = True,
    **kwargs: Any,
) -> str:
    """
    Dump JSON to a string.

    Use this helper for local JSON, YAML, pickle, text, binary, and JSONL artifacts.

    Args:
        obj (Any): Value to serialize, format, print, copy, or pass through the helper.
        sort_keys (bool): Whether serialized mapping keys should be sorted. Defaults to ``False``.
        indent (Optional[int]): Indentation width. ``None`` reads ``heavenbase.serialize.indent``; the default config is ``4``. Defaults to ``None``.
        ensure_ascii (bool): Whether JSON output escapes non-ASCII characters. Defaults to ``False``.
        compact (bool): Whether to force one-line compact JSON by passing ``indent=None`` to ``json.dumps``. Defaults to ``False``.
        adapt (bool): Whether to encode HeavenBase tagged values such as tuples,
            sets, and large integers. Set to ``False`` at strict JSON
            boundaries. Defaults to ``True``.
        **kwargs (Any): Extra keyword arguments forwarded to the underlying serializer, command, or callable.

    Returns:
        str: The result of ``dumps_json``.
    """
    value = _json_ready(obj) if adapt else obj
    return json.dumps(value, sort_keys=sort_keys, indent=None if compact else _indent(indent), ensure_ascii=ensure_ascii, **kwargs)


def load_json(path: str, encoding: Optional[str] = None, strict: bool = False, **kwargs: Any) -> Any:
    """
    Load JSON from a local file, returning ``{}`` for a missing non-strict file.

    Use this helper for local JSON, YAML, pickle, text, binary, and JSONL artifacts.

    Args:
        path (str): Local path, config path, or dotted dictionary path used by the helper.
        encoding (Optional[str]): Text encoding. ``None`` reads ``heavenbase.serialize.encoding``; the default config is ``"utf-8"``. Defaults to ``None``.
        strict (bool): Whether missing files should raise ``FileNotFoundError``. Defaults to ``False``.
        **kwargs (Any): Extra keyword arguments forwarded to the underlying serializer, command, or callable.

    Returns:
        Any: JSON value, or ``{}`` when missing and ``strict=False``.

    Examples:
        >>> from heavenbase.utils.serialize import load_json
        >>> load_json("missing.json")
        {}
    """
    value = CM_HVNB.pj(path, abs=True)
    if not exists_file(value):
        if strict:
            raise FileNotFoundError(value)
        return {}
    return loads_json(Path(value).read_text(encoding=_encoding(encoding), errors="ignore"), **kwargs)


def dump_json(
    obj: Any,
    path: str,
    sort_keys: bool = False,
    indent: Optional[int] = None,
    encoding: Optional[str] = None,
    ensure_ascii: bool = False,
    compact: bool = False,
    **kwargs: Any,
) -> None:
    """
    Write JSON to a local file with HeavenBase common-value encoding.

    Use this helper for local JSON, YAML, pickle, text, binary, and JSONL artifacts.

    Args:
        obj (Any): Value to serialize, format, print, copy, or pass through the helper.
        path (str): Local path, config path, or dotted dictionary path used by the helper.
        sort_keys (bool): Whether serialized mapping keys should be sorted. Defaults to ``False``.
        indent (Optional[int]): Indentation width. ``None`` reads ``heavenbase.serialize.indent``; the default config is ``4``. Defaults to ``None``.
        encoding (Optional[str]): Text encoding. ``None`` reads ``heavenbase.serialize.encoding``; the default config is ``"utf-8"``. Defaults to ``None``.
        ensure_ascii (bool): Whether JSON output escapes non-ASCII characters. Defaults to ``False``.
        compact (bool): Whether to force one-line compact JSON. Defaults to ``False``.
        **kwargs (Any): Extra keyword arguments forwarded to the underlying serializer, command, or callable.

    Returns:
        None: This function returns ``None`` after completing its side effect.

    Examples:
        >>> from heavenbase.utils.serialize import dump_json, load_json
        >>> dump_json({"items": (1, 2)}, "tmp/payload.json")
        >>> load_json("tmp/payload.json")["items"]
        (1, 2)
    """
    touch_dir(get_file_dir(CM_HVNB.pj(path, abs=True)))
    Path(CM_HVNB.pj(path, abs=True)).write_text(
        dumps_json(obj, sort_keys=sort_keys, indent=indent, ensure_ascii=ensure_ascii, compact=compact, **kwargs),
        encoding=_encoding(encoding),
        errors="ignore",
    )


save_json = dump_json


def dumps_jstr(
    obj: Any,
    sort_keys: bool = False,
    indent: Optional[int] = None,
    ensure_ascii: bool = False,
    compact: bool = True,
    **kwargs: Any,
) -> str:
    """
    Serialize a value to agent-facing text using JSON-or-string rules.

    **jstr** means "JSON when the top-level value is a container, otherwise string".
    Use this at text boundaries such as Tool results, memstate rows, and MCP payloads
    where structured values should stay machine-readable while everything else should
    stay human-readable via ``str(...)``.

    Rules:
        - Top-level ``dict`` or ``list`` values are encoded as JSON through
          :func:`dumps_json`, including nested containers.
        - All other values, including ``int``, ``float``, ``bool``, ``None``,
          tuples, sets, datetimes, and custom objects, use ``str(obj)``.

    Args:
        obj (Any): Value to serialize.
        sort_keys (bool): Whether serialized mapping keys should be sorted. Defaults to ``False``.
        indent (Optional[int]): Indentation width for JSON containers. Defaults to ``None``.
        ensure_ascii (bool): Whether JSON output escapes non-ASCII characters. Defaults to ``False``.
        compact (bool): Whether JSON containers use one-line compact output. Defaults to ``True``.
        **kwargs (Any): Extra keyword arguments forwarded to ``dumps_json`` for container values.

    Returns:
        str: Compact JSON for top-level ``dict``/``list`` values, otherwise ``str(obj)``.

    Examples:
        >>> from heavenbase.utils.serialize import dumps_jstr, loads_jstr
        >>> dumps_jstr(42)
        '42'
        >>> dumps_jstr({"a": 1, "b": [{"c": 2}]})
        '{"a": 1, "b": [{"c": 2}]}'
        >>> dumps_jstr([1, {"nested": {"x": 3}}])
        '[1, {"nested": {"x": 3}}]'
        >>> loads_jstr('{"a": 1}')
        {'a': 1}
    """
    if isinstance(obj, (dict, list)):
        return dumps_json(obj, sort_keys=sort_keys, indent=indent, ensure_ascii=ensure_ascii, compact=compact, **kwargs)
    return str(obj)


def loads_jstr(s: str, **kwargs: Any) -> Any:
    """
    Restore a value written by :func:`dumps_jstr`.

    Text that looks like a JSON object or array is parsed with :func:`loads_json`.
    Invalid JSON in that shape, and all other text, is returned unchanged as a string.

    Args:
        s (str): Serialized jstr text.
        **kwargs (Any): Extra keyword arguments forwarded to ``loads_json``.

    Returns:
        Any: Parsed JSON container or the original text string.
    """
    stripped = s.lstrip()
    if stripped and stripped[0] in "{[":
        try:
            return loads_json(s, **kwargs)
        except json.JSONDecodeError:
            return s
    return s


def load_jstr(path: str, encoding: Optional[str] = None, strict: bool = False, **kwargs: Any) -> Any:
    """
    Load a jstr value from a local file.

    Args:
        path (str): Local path, config path, or dotted dictionary path used by the helper.
        encoding (Optional[str]): Text encoding. ``None`` reads ``heavenbase.serialize.encoding``; the default config is ``"utf-8"``. Defaults to ``None``.
        strict (bool): Whether missing files should raise ``FileNotFoundError``. Defaults to ``False``.
        **kwargs (Any): Extra keyword arguments forwarded to ``loads_jstr``.

    Returns:
        Any: Parsed value, or ``""`` when missing and ``strict=False``.
    """
    value = CM_HVNB.pj(path, abs=True)
    if not exists_file(value):
        if strict:
            raise FileNotFoundError(value)
        return ""
    return loads_jstr(Path(value).read_text(encoding=_encoding(encoding), errors="ignore"), **kwargs)


def dump_jstr(
    obj: Any,
    path: str,
    sort_keys: bool = False,
    indent: Optional[int] = None,
    encoding: Optional[str] = None,
    ensure_ascii: bool = False,
    compact: bool = True,
    **kwargs: Any,
) -> None:
    """
    Write a jstr value to a local file.

    Args:
        obj (Any): Value to serialize.
        path (str): Local path, config path, or dotted dictionary path used by the helper.
        sort_keys (bool): Whether serialized mapping keys should be sorted. Defaults to ``False``.
        indent (Optional[int]): Indentation width for JSON containers. Defaults to ``None``.
        encoding (Optional[str]): Text encoding. ``None`` reads ``heavenbase.serialize.encoding``; the default config is ``"utf-8"``. Defaults to ``None``.
        ensure_ascii (bool): Whether JSON output escapes non-ASCII characters. Defaults to ``False``.
        compact (bool): Whether JSON containers use one-line compact output. Defaults to ``True``.
        **kwargs (Any): Extra keyword arguments forwarded to ``dumps_jstr``.

    Returns:
        None: This function returns ``None`` after completing its side effect.
    """
    touch_dir(get_file_dir(CM_HVNB.pj(path, abs=True)))
    Path(CM_HVNB.pj(path, abs=True)).write_text(
        dumps_jstr(obj, sort_keys=sort_keys, indent=indent, ensure_ascii=ensure_ascii, compact=compact, **kwargs),
        encoding=_encoding(encoding),
        errors="ignore",
    )


save_jstr = dump_jstr


def loads_jsonl(s: str, **kwargs: Any) -> List[Any]:
    """
    Load JSONL from a string.

    Use this helper for local JSON, YAML, pickle, text, binary, and JSONL artifacts.

    Args:
        s (str): Serialized string content to parse.
        **kwargs (Any): Extra keyword arguments forwarded to the underlying serializer, command, or callable.

    Returns:
        List[Any]: The result of ``loads_jsonl``.
    """
    return [loads_json(line, **kwargs) for line in s.splitlines() if line.strip()]


def dumps_jsonl(obj: Iterable[Any], sort_keys: bool = False, ensure_ascii: bool = False, **kwargs: Any) -> str:
    """
    Dump iterable items to JSONL.

    Use this helper for local JSON, YAML, pickle, text, binary, and JSONL artifacts.

    Args:
        obj (Iterable[Any]): Value to serialize, format, print, copy, or pass through the helper.
        sort_keys (bool): Whether serialized mapping keys should be sorted. Defaults to ``False``.
        ensure_ascii (bool): Whether JSON output escapes non-ASCII characters. Defaults to ``False``.
        **kwargs (Any): Extra keyword arguments forwarded to the underlying serializer, command, or callable.

    Returns:
        str: The result of ``dumps_jsonl``.
    """
    return "\n".join(dumps_json(item, sort_keys=sort_keys, compact=True, ensure_ascii=ensure_ascii, **kwargs) for item in obj)


def load_jsonl(path: str, encoding: Optional[str] = None, strict: bool = False, **kwargs: Any) -> List[Any]:
    """
    Load a JSONL file.

    Use this helper for local JSON, YAML, pickle, text, binary, and JSONL artifacts.

    Args:
        path (str): Local path, config path, or dotted dictionary path used by the helper.
        encoding (Optional[str]): Text encoding. ``None`` reads ``heavenbase.serialize.encoding``; the default config is ``"utf-8"``. Defaults to ``None``.
        strict (bool): Whether missing files should raise ``FileNotFoundError``. Defaults to ``False``.
        **kwargs (Any): Extra keyword arguments forwarded to the underlying serializer, command, or callable.

    Returns:
        List[Any]: List of decoded JSONL records.
    """
    return loads_jsonl(load_txt(path, encoding=encoding, strict=strict), **kwargs)


def iter_jsonl(path: str, encoding: Optional[str] = None, strict: bool = False, **kwargs: Any) -> Generator[Any, None, None]:
    """Yield JSON values from a JSONL file.

    Use this helper for local JSON, YAML, pickle, text, binary, and JSONL artifacts.

    Args:
        path (str): Local path, config path, or dotted dictionary path used by the helper.
        encoding (Optional[str]): Text encoding. ``None`` reads ``heavenbase.serialize.encoding``; the default config is ``"utf-8"``. Defaults to ``None``.
        strict (bool): Whether missing files should raise ``FileNotFoundError``. Defaults to ``False``.
        **kwargs (Any): Extra keyword arguments forwarded to the underlying serializer, command, or callable.

    Yields:
        dict[str, Any]: Decoded JSON records.
    """
    for line in iter_txt(path, encoding=encoding, strict=strict):
        if line.strip():
            yield loads_json(line, **kwargs)


def dump_jsonl(obj: Iterable[Any], path: str, encoding: Optional[str] = None, **kwargs: Any) -> None:
    """
    Write JSONL to a file.

    Use this helper for local JSON, YAML, pickle, text, binary, and JSONL artifacts.

    Args:
        obj (Iterable[Any]): Value to serialize, format, print, copy, or pass through the helper.
        path (str): Local path, config path, or dotted dictionary path used by the helper.
        encoding (Optional[str]): Text encoding. ``None`` reads ``heavenbase.serialize.encoding``; the default config is ``"utf-8"``. Defaults to ``None``.
        **kwargs (Any): Extra keyword arguments forwarded to the underlying serializer, command, or callable.

    Returns:
        None: This function returns ``None`` after completing its side effect.
    """
    touch_dir(get_file_dir(CM_HVNB.pj(path, abs=True)))
    Path(CM_HVNB.pj(path, abs=True)).write_text(dumps_jsonl(obj, **kwargs) + "\n", encoding=_encoding(encoding), errors="ignore")


save_jsonl = dump_jsonl


def append_jsonl(obj: Any, path: str, encoding: Optional[str] = None, **kwargs: Any) -> None:
    """
    Append one JSON value to a JSONL file.

    Use this helper for local JSON, YAML, pickle, text, binary, and JSONL artifacts.

    Args:
        obj (Any): Value to serialize, format, print, copy, or pass through the helper.
        path (str): Local path, config path, or dotted dictionary path used by the helper.
        encoding (Optional[str]): Text encoding. ``None`` reads ``heavenbase.serialize.encoding``; the default config is ``"utf-8"``. Defaults to ``None``.
        **kwargs (Any): Extra keyword arguments forwarded to the underlying serializer, command, or callable.

    Returns:
        None: This function returns ``None`` after completing its side effect.

    Examples:
        >>> from heavenbase.utils.serialize import append_jsonl, iter_jsonl
        >>> append_jsonl({"idx": 1}, "tmp/events.jsonl")
        >>> list(iter_jsonl("tmp/events.jsonl"))[-1]
        {'idx': 1}
    """
    append_txt(dumps_json(obj, compact=True, **kwargs), path, encoding=encoding)
