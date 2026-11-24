"""Internal callable serialization strategy for Capsule-aware payloads."""

from __future__ import annotations

__all__ = ["CALLABLE_FORMAT", "CallableSerialization", "callable_ref", "deserialize_callable", "serialize_callable"]

from heavenbase.utils import Any, Callable, Dict, Mapping, dataclass
from heavenbase.utils.config import ConfigManager

from ..modules import ModuleResolver
from .capture import CaptureOptions, capture_function
from .manifest import CapsuleManifest

CALLABLE_FORMAT = "heavenbase-callable"


@dataclass
class CallableSerialization:
    """JSON-safe callable serialization envelope backed by a Capsule manifest."""

    manifest: CapsuleManifest
    format: str = CALLABLE_FORMAT
    version: int = 1

    @classmethod
    def from_func(
        cls,
        func: Callable[..., Any],
        *,
        resolver: ModuleResolver | None = None,
        config: ConfigManager | None = None,
        **options: Any,
    ) -> "CallableSerialization":
        """Capture a callable into a JSON-safe serialization envelope.

        Args:
            func (Callable[..., Any]): Python callable to capture.
            resolver (ModuleResolver | None): Context-owned module resolver.
                The default Context is selected when omitted.
            config (ConfigManager | None): Config authority paired with
                `resolver`. The standalone process config is selected when omitted.
            **options (Any): Capture options forwarded to Capsule capture.

        Returns:
            CallableSerialization: JSON-safe callable serialization envelope.
        """
        return cls(capture_function(func, CaptureOptions(**options), resolver=resolver, config=config))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CallableSerialization":
        """Build a callable serialization envelope from a dictionary.

        Args:
            data (Mapping[str, Any]): Serialization mapping containing a
                ``manifest`` payload or legacy ``__obj_data__`` envelope.

        Returns:
            CallableSerialization: Parsed callable serialization envelope.
        """
        payload = dict(data or {})
        manifest = payload.get("manifest") or payload.get("__obj_data__") or payload
        return cls(
            CapsuleManifest.from_dict(manifest),
            format=str(payload.get("format") or payload.get("__obj_type__") or CALLABLE_FORMAT),
            version=int(payload.get("version") or 1),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-safe callable serialization envelope.

        Args:
            None.

        Returns:
            Dict[str, Any]: Tagged callable serialization payload.
        """
        return {"__obj_type__": CALLABLE_FORMAT, "format": self.format, "version": self.version, "manifest": self.manifest.to_dict()}

    def to_capsule(
        self,
        *,
        resolver: ModuleResolver | None = None,
        config: ConfigManager | None = None,
    ) -> Any:
        """Return a Capsule without executing the callable.

        Args:
            resolver (ModuleResolver | None): Context-owned module resolver
                used for lazy restoration.
            config (ConfigManager | None): Config authority paired with
                `resolver` for lazy restoration.

        Returns:
            Capsule: Lazy Capsule restored from the embedded manifest.
        """
        from .runtime import Capsule

        return Capsule.from_manifest(self.manifest, resolver=resolver, config=config)

    def to_func(
        self,
        *,
        resolver: ModuleResolver | None = None,
        config: ConfigManager | None = None,
    ) -> Callable[..., Any]:
        """Restore and return the callable.

        Args:
            resolver (ModuleResolver | None): Context-owned module resolver
                used for lazy restoration.
            config (ConfigManager | None): Config authority paired with
                `resolver` for lazy restoration.

        Returns:
            Callable[..., Any]: Restored Python callable.
        """
        return self.to_capsule(resolver=resolver, config=config).to_func()


def serialize_callable(
    func: Callable[..., Any],
    *,
    resolver: ModuleResolver | None = None,
    config: ConfigManager | None = None,
    **options: Any,
) -> Dict[str, Any]:
    """Serialize a callable using the Capsule strategy.

    Args:
        func (Callable[..., Any]): Python callable to capture.
        resolver (ModuleResolver | None): Context-owned module resolver.
            The default Context is selected when omitted.
        config (ConfigManager | None): Config authority paired with
            `resolver`. The standalone process config is selected when omitted.
        **options (Any): Capture options forwarded to Capsule capture.

    Returns:
        Dict[str, Any]: JSON-safe callable serialization envelope.
    """
    return CallableSerialization.from_func(func, resolver=resolver, config=config, **options).to_dict()


def callable_ref(func: Callable[..., Any]) -> Dict[str, Any]:
    """Return a lightweight JSON-safe callable reference.

    Args:
        func (Callable[..., Any]): Importable top-level Python callable.

    Returns:
        Dict[str, Any]: Lightweight callable reference envelope.
    """
    return {"__obj_type__": "callable-ref", "__obj_data__": f"{func.__module__}.{func.__qualname__}"}


def deserialize_callable(
    data: Mapping[str, Any],
    *,
    resolver: ModuleResolver | None = None,
    config: ConfigManager | None = None,
) -> Callable[..., Any]:
    """Restore a callable from a Capsule serialization envelope.

    Args:
        data (Mapping[str, Any]): Callable serialization mapping.
        resolver (ModuleResolver | None): Context-owned module resolver
            used for lazy restoration.
        config (ConfigManager | None): Config authority paired with
            `resolver` for lazy restoration.

    Returns:
        Callable[..., Any]: Restored Python callable.
    """
    return CallableSerialization.from_dict(data).to_func(resolver=resolver, config=config)
