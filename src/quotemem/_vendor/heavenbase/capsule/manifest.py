"""Capsule manifest data model."""

from __future__ import annotations

__all__ = ["CapsuleManifest", "make_capsule_id"]

from time import time

from heavenbase.utils import Any, Dict, List, Mapping, dataclass, field, hash_id
from heavenbase.utils.config import ConfigManager

from ..modules import ModuleResolver


def make_capsule_id(namespace: str, name: str, version: str) -> str:
    """Return the stable default Capsule id for an explicit version.

    Args:
        namespace (str): Capsule namespace. Empty values normalize to
            ``"default"``.
        name (str): Public Capsule name.
        version (str): Explicit Capsule version. Empty values normalize to
            ``"1"``.

    Returns:
        str: Stable content-derived Capsule identifier.
    """
    return hash_id("capsule", namespace or "default", name, version or "1")


@dataclass
class CapsuleManifest:
    """Durable manifest describing one executable Capsule."""

    id: str
    name: str
    namespace: str = "default"
    version: str = "1"
    kind: str = "python-function"
    description: str = ""
    tags: List[str] = field(default_factory=list)
    identity: Dict[str, Any] = field(default_factory=dict)
    signature: Dict[str, Any] = field(default_factory=dict)
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_schema: Dict[str, Any] = field(default_factory=dict)
    restore_layers: List[Dict[str, Any]] = field(default_factory=list)
    dependencies: Dict[str, Any] = field(default_factory=dict)
    capabilities: Dict[str, Any] = field(default_factory=dict)
    runtime: Dict[str, Any] = field(default_factory=dict)
    source: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)
    trust: Dict[str, Any] = field(default_factory=dict)
    fingerprints: Dict[str, Any] = field(default_factory=dict)
    status: str = "active"
    created_at: float = field(default_factory=time)
    updated_at: float = field(default_factory=time)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-safe manifest dictionary.

        Args:
            None.

        Returns:
            Dict[str, Any]: JSON-compatible Capsule manifest payload.
        """
        return {
            "id": self.id,
            "name": self.name,
            "namespace": self.namespace,
            "version": self.version,
            "kind": self.kind,
            "description": self.description,
            "tags": list(self.tags),
            "identity": dict(self.identity),
            "signature": dict(self.signature),
            "input_schema": dict(self.input_schema),
            "output_schema": dict(self.output_schema),
            "restore_layers": [dict(layer) for layer in self.restore_layers],
            "dependencies": dict(self.dependencies),
            "capabilities": dict(self.capabilities),
            "runtime": dict(self.runtime),
            "source": dict(self.source),
            "provenance": dict(self.provenance),
            "trust": dict(self.trust),
            "fingerprints": dict(self.fingerprints),
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CapsuleManifest":
        """Build a CapsuleManifest from a dictionary.

        Args:
            data (Mapping[str, Any]): Manifest mapping. Missing ``id`` values
                are derived from ``namespace``, ``name``, and ``version``.

        Returns:
            CapsuleManifest: Parsed manifest object.
        """
        values = dict(data or {})
        if not values.get("id"):
            values["id"] = make_capsule_id(str(values.get("namespace") or "default"), str(values.get("name") or "capsule"), str(values.get("version") or "1"))
        return cls(**{key: values[key] for key in cls.__dataclass_fields__ if key in values})

    def verify(
        self,
        policy: Any = None,
        *,
        resolver: ModuleResolver | None = None,
        config: ConfigManager | None = None,
    ) -> None:
        """Verify stored source and binary checksums without executing code.

        Args:
            policy (Any): Optional restore policy controlling checksum verification. Defaults to ``None``.
            resolver (ModuleResolver | None): Context-owned module resolver
                used for Capsule layer verification.
            config (ConfigManager | None): Config authority paired with
                `resolver` for trust decisions.

        Returns:
            None: Raises when a stored checksum does not match its payload.
        """
        from .restore import _verify_manifest

        _verify_manifest(self, resolver=resolver, policy=policy, config=config)
