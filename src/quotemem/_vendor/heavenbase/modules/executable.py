"""Versioned executable references shared by module records and consumers."""

from __future__ import annotations

__all__ = [
    "EXECUTABLE_REF_KIND",
    "EXECUTABLE_REF_VERSION",
    "ExecutableRef",
    "capsule_executable_ref",
    "python_executable_ref",
]

from collections.abc import Mapping
import copy
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Literal

from heavenbase.utils.registry import python_target

EXECUTABLE_REF_KIND = "heavenbase.executable"
EXECUTABLE_REF_VERSION = 1

_MODULE_RE = re.compile(r"^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*$")
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string")
    clean = value.strip()
    if not clean:
        raise ValueError(f"{label} must be non-empty")
    return clean


def _import_target(value: Any, label: str) -> str:
    target = _text(value, label)
    module, separator, qualname = target.partition(":")
    if not separator or not _MODULE_RE.fullmatch(module) or not _MODULE_RE.fullmatch(qualname):
        raise ValueError(f"{label} must use 'module:qualified_name' syntax")
    if "<locals>" in qualname or "<lambda>" in qualname:
        raise ValueError(f"{label} must identify an importable top-level object")
    return f"{module}:{qualname}"


def _artifact_target(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    unknown = set(value) - {"artifact", "module", "qualname"}
    if unknown:
        raise ValueError(f"{label} has unsupported fields {sorted(unknown)!r}")
    if set(value) != {"artifact", "module", "qualname"}:
        raise ValueError(f"{label} requires artifact, module, and qualname")
    artifact = _text(value.get("artifact"), f"{label}.artifact")
    if not _DIGEST_RE.fullmatch(artifact):
        raise ValueError(f"{label}.artifact must be a lowercase SHA-256 digest")
    module_raw = value.get("module")
    module = None if module_raw in (None, "") else _text(module_raw, f"{label}.module")
    qualname = _text(value.get("qualname"), f"{label}.qualname")
    if module is not None:
        _import_target(f"{module}:{qualname}", label)
    elif not _MODULE_RE.fullmatch(qualname):
        raise ValueError(f"{label}.qualname must be a dotted top-level attribute path")
    return {"artifact": artifact, "module": module, "qualname": qualname}


def _path_target(
    value: Any,
    base: Path | None,
    artifact_digest: str | None,
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping with module and qualname")
    unknown = set(value) - {"module", "qualname"}
    if unknown:
        raise ValueError(f"{label} has unsupported fields {sorted(unknown)!r}")
    if base is None or artifact_digest is None:
        raise ValueError(f"{label} path sources require installation from one captured module folder")
    module_raw = value.get("module")
    module = None if module_raw in (None, "") else _text(module_raw, f"{label}.module")
    qualname = _text(value.get("qualname"), f"{label}.qualname")
    if module is not None:
        _import_target(f"{module}:{qualname}", label)
    elif not _MODULE_RE.fullmatch(qualname):
        raise ValueError(f"{label}.qualname must be a dotted top-level attribute path")
    return _artifact_target(
        {
            "artifact": artifact_digest,
            "module": module,
            "qualname": qualname,
        },
        label,
    )


@dataclass(frozen=True, slots=True)
class ExecutableRef:
    """Identify one executable restoration source without embedding the object.

    Args:
        source: Canonical runtime source: ``python-import``, ``artifact``, or
            ``capsule``.
        target: Source-specific import target, artifact target, or Capsule id.
        version: Executable-reference schema version.
        kind: Executable-reference kind marker.
    """

    source: Literal["python-import", "artifact", "capsule"]
    target: Any
    version: int = EXECUTABLE_REF_VERSION
    kind: str = EXECUTABLE_REF_KIND

    def __post_init__(self) -> None:
        """Validate and isolate the canonical reference payload."""
        if self.kind != EXECUTABLE_REF_KIND:
            raise ValueError(f"Executable reference kind must be {EXECUTABLE_REF_KIND!r}")
        if self.version != EXECUTABLE_REF_VERSION or isinstance(self.version, bool):
            raise ValueError(f"Executable reference version must be {EXECUTABLE_REF_VERSION}")
        source = str(self.source)
        if source == "python-import":
            target: Any = _import_target(self.target, "Executable reference target")
        elif source == "artifact":
            target = _artifact_target(self.target, "Executable reference target")
        elif source == "capsule":
            target = _text(self.target, "Executable reference Capsule id")
        else:
            raise ValueError(f"Unsupported executable reference source {source!r}")
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "target", copy.deepcopy(target))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExecutableRef":
        """Parse one canonical persisted executable reference.

        Args:
            data (Mapping[str, Any]): Executable-reference envelope with
                ``kind``, ``version``, ``source``, and ``target`` keys.

        Returns:
            ExecutableRef: Validated executable reference value.

        Raises:
            TypeError: If ``data`` is not a mapping.
            ValueError: If required fields are missing or invalid.
        """
        if not isinstance(data, Mapping):
            raise TypeError("ExecutableRef.from_dict expects a mapping")
        unknown = set(data) - {"kind", "version", "source", "target"}
        if unknown:
            raise ValueError(f"Executable reference has unsupported fields {sorted(unknown)!r}")
        if set(data) != {"kind", "version", "source", "target"}:
            raise ValueError("Executable reference requires kind, version, source, and target")
        return cls(
            source=str(data["source"]),
            target=data["target"],
            version=data["version"],
            kind=data["kind"],
        )

    @classmethod
    def python(cls, target: str) -> "ExecutableRef":
        """Build an explicit ambient Python-import reference.

        Args:
            target (str): Import target in ``module:qualified_name`` syntax.

        Returns:
            ExecutableRef: Python-import executable reference.
        """
        return cls("python-import", target)

    @classmethod
    def capsule(cls, capsule_id: str) -> "ExecutableRef":
        """Build a reference to an independently persisted Capsule.

        Args:
            capsule_id (str): Stable Capsule identifier.

        Returns:
            ExecutableRef: Capsule-backed executable reference.
        """
        return cls("capsule", capsule_id)

    def to_dict(self) -> dict[str, Any]:
        """Return an isolated JSON-compatible reference envelope.

        Args:
            None.

        Returns:
            dict[str, Any]: Canonical executable-reference mapping.
        """
        return {
            "kind": self.kind,
            "version": self.version,
            "source": self.source,
            "target": copy.deepcopy(self.target),
        }


def python_executable_ref(value: Any, *, label: str) -> dict[str, Any]:
    """Return a canonical ambient-import reference for an importable object.

    Args:
        value (Any): Callable or importable object to reference.
        label (str): Error-label prefix used when normalization fails.

    Returns:
        dict[str, Any]: Canonical executable-reference mapping.
    """
    return ExecutableRef.python(python_target(value, label=label)).to_dict()


def capsule_executable_ref(capsule_id: str) -> dict[str, Any]:
    """Return a canonical executable reference for one Capsule id.

    Args:
        capsule_id (str): Stable Capsule identifier.

    Returns:
        dict[str, Any]: Canonical executable-reference mapping.
    """
    return ExecutableRef.capsule(capsule_id).to_dict()


def _prepare_executable_ref(
    data: Mapping[str, Any],
    *,
    base: Path | None,
    artifact_digest: str | None,
    label: str,
) -> dict[str, Any]:
    """Normalize one author-facing executable reference for persistence."""
    unknown = set(data) - {"kind", "version", "source", "target"}
    if unknown:
        raise ValueError(f"{label} has unsupported fields {sorted(unknown)!r}")
    if set(data) != {"kind", "version", "source", "target"}:
        raise ValueError(f"{label} requires kind, version, source, and target")
    if data.get("kind") != EXECUTABLE_REF_KIND:
        raise ValueError(f"{label}.kind must be {EXECUTABLE_REF_KIND!r}")
    if data.get("version") != EXECUTABLE_REF_VERSION or isinstance(data.get("version"), bool):
        raise ValueError(f"{label}.version must be {EXECUTABLE_REF_VERSION}")
    source = str(data.get("source") or "")
    if source == "path":
        return ExecutableRef(
            "artifact",
            _path_target(
                data.get("target"),
                base,
                artifact_digest,
                f"{label}.target",
            ),
        ).to_dict()
    if source == "artifact":
        raise ValueError(f"{label}.source 'artifact' is installer-owned; authors must use 'path'")
    return ExecutableRef(
        source,
        data.get("target"),
    ).to_dict()


def _prepare_executable_refs(
    value: Any,
    *,
    base: Path | None,
    artifact_digest: str | None,
    label: str,
) -> Any:
    """Recursively normalize explicitly tagged executable references."""
    if isinstance(value, Mapping):
        if value.get("kind") == EXECUTABLE_REF_KIND:
            return _prepare_executable_ref(
                value,
                base=base,
                artifact_digest=artifact_digest,
                label=label,
            )
        return {
            key: _prepare_executable_refs(
                item,
                base=base,
                artifact_digest=artifact_digest,
                label=f"{label}.{key}",
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _prepare_executable_refs(
                item,
                base=base,
                artifact_digest=artifact_digest,
                label=f"{label}[{index}]",
            )
            for index, item in enumerate(value)
        ]
    return value
