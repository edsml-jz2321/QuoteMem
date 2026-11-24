"""Plain module-record validation and deterministic manifest loading."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import copy
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import re
from typing import Any

from heavenbase.utils.registry import identifier_key, named_key

from .executable import (
    _import_target,
    _path_target,
    _prepare_executable_refs,
)

__all__ = ["InstallReceipt"]

ModuleKey = tuple[str, str]
ModuleGroup = tuple[str, str]

_RECORD_FIELDS = frozenset(("name", "desc", "source", "target", "active", "meta", "dtype", "backend", "type", "strategy", "ws", "field"))
_TEXT_FIELDS = frozenset(("name", "desc", "dtype", "backend", "type", "strategy", "ws", "field"))
_SOURCE_KINDS = frozenset(("python-import", "path", "inline", "capsule"))
_COORDINATE_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*(?:/[a-z0-9][a-z0-9_-]*)+$")
_HEAVENBASE_VERSION_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)\.(\d+)(?:dev(\d+))?$")
_GENERATION_RE = re.compile(r"^[0-9a-f]{32}$")
_FINGERPRINT_RE = re.compile(r"^[0-9a-f]{64}$")
_INSTALLATION_KEY = "installation"


@dataclass(frozen=True, slots=True)
class InstallReceipt:
    """Identify one atomic registration publication.

    Args:
        coordinate: Stable identity of the independently reconciled manifest.
        generation: Unique identity of this successful publication.
        keys: Canonical `(kind, identifier)` keys published by the operation.
        bundles: Bundle identifiers assigned through Registry groups.
        revision: Registry revision visible after publication.
        manifest_fingerprint: Canonical manifest fingerprint.
    """

    coordinate: str
    generation: str
    keys: tuple[ModuleKey, ...]
    bundles: tuple[str, ...]
    revision: int
    manifest_fingerprint: str


@dataclass(frozen=True, slots=True)
class _PreparedItem:
    key: ModuleKey
    record: dict[str, Any]
    aliases: tuple[ModuleKey, ...]
    groups: tuple[ModuleGroup, ...]


@dataclass(frozen=True, slots=True)
class _PreparedInstallation:
    coordinate: str
    version: str
    compatibility: dict[str, Any]
    fingerprint: str
    items: tuple[_PreparedItem, ...]

    @property
    def keys(self) -> tuple[ModuleKey, ...]:
        return tuple(item.key for item in self.items)

    @property
    def bundles(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(value for item in self.items for group, value in item.groups if group == "bundle"))


def _manifest_uses_path_source(value: Any) -> bool:
    """Return whether a manifest contains an author-facing path reference."""
    if isinstance(value, Mapping):
        return value.get("source") == "path" or any(_manifest_uses_path_source(item) for item in value.values())
    if isinstance(value, list):
        return any(_manifest_uses_path_source(item) for item in value)
    return False


def _json_value(value: Any, path: str = "value") -> Any:
    """Return a detached JSON value or raise with its manifest path."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must not contain NaN or infinity")
        return value
    if isinstance(value, list):
        return [_json_value(item, f"{path}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} contains non-string mapping key {key!r}")
            out[key] = _json_value(item, f"{path}.{key}")
        return out
    raise TypeError(f"{path} contains non-JSON value {value!r}")


def _require_json_value(value: Any, path: str = "value") -> None:
    """Validate one JSON value without allocating a detached copy."""
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must not contain NaN or infinity")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _require_json_value(item, f"{path}[{index}]")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} contains non-string mapping key {key!r}")
            _require_json_value(item, f"{path}.{key}")
        return
    raise TypeError(f"{path} contains non-JSON value {value!r}")


def _fingerprint(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    return sha256(payload).hexdigest()


def _text(value: Any, label: str, *, empty: bool = False) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string")
    clean = value.strip()
    if not clean and not empty:
        raise ValueError(f"{label} must be non-empty")
    return clean


def _coordinate(value: Any, label: str = "module coordinate") -> str:
    """Return one normalized slash-separated module coordinate."""
    coordinate = _text(value, label).lower()
    if not _COORDINATE_RE.fullmatch(coordinate):
        raise ValueError(f"{label} must be lowercase slash-separated identifiers, got {value!r}")
    return coordinate


def _heavenbase_version_key(value: Any, label: str) -> tuple[int, ...]:
    """Return a sortable key for one supported HeavenBase version."""
    version = _text(value, label)
    match = _HEAVENBASE_VERSION_RE.fullmatch(version)
    if match is None:
        raise ValueError(f"{label} must be a four-part HeavenBase version, got {value!r}")
    release = tuple(int(part) for part in match.groups()[:4])
    development = match.group(5)
    return (*release, 1, 0) if development is None else (*release, 0, int(development))


def _compatibility(value: Any, label: str = "module compatibility") -> dict[str, Any]:
    """Validate the deliberately narrow module compatibility grammar."""
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    unknown = set(value) - {"heavenbase"}
    if unknown:
        raise ValueError(f"{label} has unsupported fields {sorted(unknown)!r}")
    raw = value.get("heavenbase")
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise TypeError(f"{label}.heavenbase must be a mapping")
    unknown = set(raw) - {"min", "before"}
    if unknown:
        raise ValueError(f"{label}.heavenbase has unsupported fields {sorted(unknown)!r}")
    if not raw:
        raise ValueError(f"{label}.heavenbase must declare min, before, or both")
    normalized: dict[str, str] = {}
    for field in ("min", "before"):
        if field in raw:
            version = _text(raw[field], f"{label}.heavenbase.{field}")
            _heavenbase_version_key(version, f"{label}.heavenbase.{field}")
            normalized[field] = version
    if "min" in normalized and "before" in normalized:
        if _heavenbase_version_key(normalized["min"], f"{label}.heavenbase.min") >= _heavenbase_version_key(normalized["before"], f"{label}.heavenbase.before"):
            raise ValueError(f"{label}.heavenbase.min must be earlier than before")
    return {"heavenbase": normalized}


def _check_compatibility(compatibility: Mapping[str, Any], runtime_version: str, label: str) -> None:
    """Reject a module outside its declared HeavenBase version interval."""
    bounds = compatibility.get("heavenbase")
    if not isinstance(bounds, Mapping):
        return
    current = _heavenbase_version_key(runtime_version, "running HeavenBase version")
    minimum = bounds.get("min")
    before = bounds.get("before")
    if isinstance(minimum, str) and current < _heavenbase_version_key(minimum, f"{label}.min"):
        raise ValueError(f"{label} requires HeavenBase >= {minimum}; running {runtime_version}")
    if isinstance(before, str) and current >= _heavenbase_version_key(before, f"{label}.before"):
        raise ValueError(f"{label} requires HeavenBase < {before}; running {runtime_version}")


def _inline_target(value: Any, meta: Mapping[str, Any], label: str) -> str:
    target = identifier_key(value, f"{label} inline target")
    if target not in meta:
        raise ValueError(f"{label} inline target {target!r} is absent from meta")
    if target in {"schema_version", "restore_layers", "dependencies", _INSTALLATION_KEY}:
        raise ValueError(f"{label} inline target {target!r} is reserved")
    return target


def _layer(
    source: Any,
    target: Any,
    meta: dict[str, Any],
    base: Path | None,
    artifact_digest: str | None,
    label: str,
) -> tuple[str, Any]:
    source_name = _text(source, f"{label}.source")
    if source_name not in _SOURCE_KINDS:
        raise ValueError(f"{label}.source {source_name!r} is unsupported; expected one of {sorted(_SOURCE_KINDS)!r}")
    if source_name == "python-import":
        return source_name, _import_target(target, f"{label}.target")
    if source_name == "path":
        return "artifact", _path_target(target, base, artifact_digest, f"{label}.target")
    if source_name == "capsule":
        return source_name, _text(target, f"{label}.target")
    return source_name, _inline_target(target, meta, label)


def _dependencies(meta: dict[str, Any], label: str) -> None:
    raw = meta.get("dependencies")
    if raw is None:
        return
    if not isinstance(raw, list):
        raise TypeError(f"{label}.meta.dependencies must be a list")
    normalized: list[dict[str, Any]] = []
    for index, dependency in enumerate(raw):
        path = f"{label}.meta.dependencies[{index}]"
        if isinstance(dependency, str):
            kind, separator, identifier = dependency.partition(":")
            if not separator:
                raise ValueError(f"{path} must use 'kind:identifier' syntax")
            normalized.append({"kind": identifier_key(kind, "dependency kind"), "identifier": identifier_key(identifier, "dependency identifier")})
            continue
        if not isinstance(dependency, Mapping):
            raise TypeError(f"{path} must be a string or mapping")
        unknown = set(dependency) - {"kind", "identifier", "optional"}
        if unknown:
            raise ValueError(f"{path} has unsupported fields {sorted(unknown)!r}")
        optional = dependency.get("optional", False)
        if not isinstance(optional, bool):
            raise TypeError(f"{path}.optional must be boolean")
        normalized.append(
            {
                "kind": identifier_key(dependency.get("kind"), "dependency kind"),
                "identifier": identifier_key(dependency.get("identifier"), "dependency identifier"),
                **({"optional": True} if optional else {}),
            }
        )
    meta["dependencies"] = normalized


def _record(raw: Any, base: Path | None, artifact_digest: str | None, label: str) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise TypeError(f"{label} must be a mapping")
    unknown = set(raw) - _RECORD_FIELDS
    if unknown:
        raise ValueError(f"{label} has unsupported fields {sorted(unknown)!r}")
    meta_raw = raw.get("meta")
    if not isinstance(meta_raw, Mapping):
        raise TypeError(f"{label}.meta must be a mapping")
    meta = _prepare_executable_refs(
        _json_value(meta_raw, f"{label}.meta"),
        base=base,
        artifact_digest=artifact_digest,
        label=f"{label}.meta",
    )
    if meta.get("schema_version") != 1 or isinstance(meta.get("schema_version"), bool):
        raise ValueError(f"{label}.meta.schema_version must be 1")
    if _INSTALLATION_KEY in meta:
        raise ValueError(f"{label}.meta.{_INSTALLATION_KEY} is installer-owned")
    if "tags" in meta and not isinstance(meta["tags"], Mapping):
        raise TypeError(f"{label}.meta.tags must be a string-keyed mapping")
    _dependencies(meta, label)
    source, target = _layer(raw.get("source"), raw.get("target"), meta, base, artifact_digest, label)
    restore_layers = meta.get("restore_layers")
    if restore_layers is not None:
        if not isinstance(restore_layers, list):
            raise TypeError(f"{label}.meta.restore_layers must be a list")
        layers: list[dict[str, Any]] = []
        for index, layer in enumerate(restore_layers):
            path = f"{label}.meta.restore_layers[{index}]"
            if not isinstance(layer, Mapping):
                raise TypeError(f"{path} must be a mapping")
            unknown_layer = set(layer) - {"source", "target"}
            if unknown_layer:
                raise ValueError(f"{path} has unsupported fields {sorted(unknown_layer)!r}")
            layer_source, layer_target = _layer(layer.get("source"), layer.get("target"), meta, base, artifact_digest, path)
            layers.append({"source": layer_source, "target": layer_target})
        meta["restore_layers"] = layers
    active = raw.get("active", True)
    if not isinstance(active, bool):
        raise TypeError(f"{label}.active must be boolean")
    record: dict[str, Any] = {
        "name": _text(raw.get("name", ""), f"{label}.name", empty=True),
        "desc": _text(raw.get("desc", ""), f"{label}.desc", empty=True),
        "source": source,
        "target": target,
        "active": active,
        "meta": meta,
    }
    for field in _TEXT_FIELDS - {"name", "desc"}:
        if field in raw:
            record[field] = _text(raw[field], f"{label}.{field}", empty=True)
    return record


def _aliases(kind: str, raw: Any, label: str) -> tuple[ModuleKey, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise TypeError(f"{label}.aliases must be a list")
    values = tuple(dict.fromkeys(named_key(alias, "registration alias") for alias in raw))
    return tuple((kind, alias) for alias in values)


def _groups(raw: Any, bundle: str | None, trusted: Iterable[ModuleGroup], label: str) -> tuple[ModuleGroup, ...]:
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise TypeError(f"{label}.groups must be a mapping of group kinds to lists")
    values: list[ModuleGroup] = []
    for group_kind, members in raw.items():
        kind = identifier_key(group_kind, "registration group kind")
        if not isinstance(members, list):
            raise TypeError(f"{label}.groups.{kind} must be a list")
        values.extend((kind, identifier_key(member, f"{kind} group value")) for member in members)
    if bundle is not None:
        values.append(("bundle", bundle))
    values.extend(trusted)
    return tuple(dict.fromkeys(values))


def _manifest_mapping(source: Mapping[str, Any] | str | Path) -> tuple[dict[str, Any], Path | None]:
    if isinstance(source, Mapping):
        return copy.deepcopy(dict(source)), None
    path = Path(source).expanduser().resolve()
    if path.is_dir():
        path = path / "meta.yaml"
    elif path.name != "meta.yaml":
        raise ValueError(f"Registration descriptor must be named 'meta.yaml', got: {path}")
    if not path.is_file():
        raise ValueError(f"Registration manifest does not exist: {path}")
    import yaml

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise ValueError(f"Invalid registration manifest {path}: {error}") from error
    if not isinstance(raw, Mapping):
        raise TypeError(f"Registration manifest {path} root must be a mapping")
    return copy.deepcopy(dict(raw)), path.parent


def _prepare_manifest(
    source: Mapping[str, Any] | str | Path,
    *,
    artifact_digest: str | None = None,
    expected_coordinate: str | None = None,
    runtime_version: str,
    trusted_groups: Iterable[ModuleGroup] = (),
) -> _PreparedInstallation:
    raw, base = _manifest_mapping(source)
    unknown = set(raw) - {"manifest_version", "coordinate", "version", "compatibility", "bundle", "items"}
    if unknown:
        raise ValueError(f"Registration manifest has unsupported fields {sorted(unknown)!r}")
    if raw.get("manifest_version") != 2 or isinstance(raw.get("manifest_version"), bool):
        raise ValueError("Registration manifest_version must be 2")
    coordinate = _coordinate(raw.get("coordinate"))
    if expected_coordinate is not None and coordinate != _coordinate(expected_coordinate, "expected module coordinate"):
        raise ValueError(f"Registration coordinate {coordinate!r} does not match expected coordinate {expected_coordinate!r}")
    version = _text(raw.get("version"), "module version")
    compatibility = _compatibility(raw.get("compatibility"))
    _check_compatibility(compatibility, runtime_version, f"Module {coordinate!r} compatibility")
    items = raw.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("Registration manifest items must be a non-empty list")
    bundle_raw = raw.get("bundle")
    bundle = identifier_key(bundle_raw, "registration bundle") if bundle_raw is not None else None
    trusted = tuple((identifier_key(group, "trusted group kind"), identifier_key(value, "trusted group value")) for group, value in trusted_groups)
    prepared: list[_PreparedItem] = []
    seen: set[ModuleKey] = set()
    for index, item in enumerate(items):
        label = f"registration item {index}"
        if not isinstance(item, Mapping):
            raise TypeError(f"{label} must be a mapping")
        unknown_item = set(item) - (_RECORD_FIELDS | {"kind", "identifier", "aliases", "groups"})
        if unknown_item:
            raise ValueError(f"{label} has unsupported fields {sorted(unknown_item)!r}")
        kind = identifier_key(item.get("kind"), "registration kind")
        identifier = identifier_key(item.get("identifier"), f"{kind} identifier")
        key = (kind, identifier)
        if key in seen:
            raise ValueError(f"Registration manifest repeats key {key!r}")
        seen.add(key)
        prepared.append(
            _PreparedItem(
                key,
                _record({field: item[field] for field in _RECORD_FIELDS if field in item}, base, artifact_digest, label),
                _aliases(kind, item.get("aliases"), label),
                _groups(item.get("groups"), bundle, trusted, label),
            )
        )
    canonical = {
        "manifest_version": 2,
        "coordinate": coordinate,
        "version": version,
        "compatibility": compatibility,
        "items": [
            {
                "kind": item.key[0],
                "identifier": item.key[1],
                "aliases": [alias for _, alias in item.aliases],
                "groups": [[group, value] for group, value in item.groups],
                **item.record,
            }
            for item in prepared
        ],
    }
    fingerprint = _fingerprint(canonical)
    return _PreparedInstallation(coordinate, version, compatibility, fingerprint, tuple(prepared))


def _record_fingerprint(record: Mapping[str, Any]) -> str:
    _require_json_value(record, "registration record")
    try:
        return _fingerprint(record)
    except TypeError:
        return _fingerprint(_json_value(record, "registration record"))


def _record_installation(record: Mapping[str, Any]) -> Mapping[str, Any] | None:
    meta = record.get("meta")
    installation = meta.get(_INSTALLATION_KEY) if isinstance(meta, Mapping) else None
    if installation is None:
        return None
    if not isinstance(installation, Mapping):
        raise ValueError("Registration record meta.installation must be a mapping")
    if "owner" in installation:
        raise ValueError("Legacy module installation ownership is unsupported; reinstall the manifest as version 2")
    required = {"coordinate", "generation", "version", "compatibility", "manifest_fingerprint", "publisher"}
    if set(installation) != required:
        raise ValueError(f"Registration record meta.installation must contain exactly {sorted(required)!r}")
    coordinate = _coordinate(installation["coordinate"], "registration coordinate")
    if installation["coordinate"] != coordinate:
        raise ValueError(f"Registration coordinate must be canonical, got {installation['coordinate']!r}")
    generation = installation["generation"]
    if not isinstance(generation, str) or _GENERATION_RE.fullmatch(generation) is None:
        raise ValueError("Registration generation must be a lowercase 32-character publication identifier")
    version = _text(installation["version"], "registration version")
    if installation["version"] != version:
        raise ValueError(f"Registration version must be trimmed, got {installation['version']!r}")
    _compatibility(installation["compatibility"], "registration compatibility")
    fingerprint = installation["manifest_fingerprint"]
    if not isinstance(fingerprint, str) or _FINGERPRINT_RE.fullmatch(fingerprint) is None:
        raise ValueError("Registration manifest_fingerprint must be a lowercase SHA-256 digest")
    publisher = installation["publisher"]
    if publisher is not None:
        clean_publisher = _text(publisher, "registration publisher")
        if publisher != clean_publisher:
            raise ValueError(f"Registration publisher must be trimmed, got {publisher!r}")
    return installation


def _record_coordinate(record: Mapping[str, Any]) -> str | None:
    installation = _record_installation(record)
    coordinate = installation.get("coordinate") if installation is not None else None
    return coordinate if isinstance(coordinate, str) and coordinate else None


def _record_generation(record: Mapping[str, Any]) -> str | None:
    installation = _record_installation(record)
    generation = installation.get("generation") if installation is not None else None
    return generation if isinstance(generation, str) and generation else None


def _record_publisher(record: Mapping[str, Any]) -> str | None:
    installation = _record_installation(record)
    publisher = installation.get("publisher") if installation is not None else None
    if publisher is not None and (not isinstance(publisher, str) or not publisher):
        raise ValueError("Registration record publisher must be a non-empty string or null")
    return publisher


def _record_version(record: Mapping[str, Any]) -> str | None:
    installation = _record_installation(record)
    version = installation.get("version") if installation is not None else None
    return version if isinstance(version, str) and version else None


def _record_compatibility(record: Mapping[str, Any]) -> dict[str, Any]:
    installation = _record_installation(record)
    value = installation.get("compatibility") if installation is not None else {}
    return _compatibility(value, "registration compatibility")


def _record_manifest_fingerprint(record: Mapping[str, Any]) -> str | None:
    installation = _record_installation(record)
    fingerprint = installation.get("manifest_fingerprint") if installation is not None else None
    return fingerprint if isinstance(fingerprint, str) and fingerprint else None


def _installed_items(
    installation: _PreparedInstallation,
    *,
    generation: str,
    publisher: str | None,
) -> tuple[_PreparedItem, ...]:
    """Inject one installer-owned identity envelope into prepared records."""
    items: list[_PreparedItem] = []
    envelope = {
        "coordinate": installation.coordinate,
        "generation": generation,
        "version": installation.version,
        "compatibility": copy.deepcopy(installation.compatibility),
        "manifest_fingerprint": installation.fingerprint,
        "publisher": publisher,
    }
    for item in installation.items:
        record = copy.deepcopy(item.record)
        record["meta"][_INSTALLATION_KEY] = copy.deepcopy(envelope)
        items.append(_PreparedItem(item.key, record, item.aliases, item.groups))
    return tuple(items)


def _record_dependencies(record: Mapping[str, Any]) -> tuple[tuple[ModuleKey, bool], ...]:
    meta = record.get("meta")
    raw = meta.get("dependencies", ()) if isinstance(meta, Mapping) else ()
    return tuple((((str(item["kind"]), str(item["identifier"]))), bool(item.get("optional", False))) for item in raw if isinstance(item, Mapping))
