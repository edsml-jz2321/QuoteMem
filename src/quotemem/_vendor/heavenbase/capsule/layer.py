"""Capsule layer value contracts."""

from __future__ import annotations

__all__ = ["CapsuleLayer", "CapsuleLayerBuildContext"]

from heavenbase.utils import Any, Callable, Dict, Mapping, Optional, dataclass, field
from heavenbase.utils.config import ConfigManager

from ..modules import ModuleResolver


@dataclass
class CapsuleLayerBuildContext:
    """Carry shared capture state for Capsule layer builders.

    Args:
        fn (Callable[..., Any]): Callable being captured.
        options (Any): Capture options object. Built-in layers receive `CaptureOptions`;
            custom layers may read public fields they define in `layer_options`.
        data (Dict[str, Any]): Shared build data populated by layers. Source-like
            layers may place normalized source metadata under `"source"` and
            warnings under `"warnings"`.
        resolver (ModuleResolver): Module authority selected by the Capsule
            capture boundary.
        config (ConfigManager): Config authority selected by the Capsule capture
            boundary.
    """

    fn: Callable[..., Any]
    options: Any
    resolver: ModuleResolver
    config: ConfigManager
    data: Dict[str, Any] = field(default_factory=dict)

    def layer_options(self, name: str) -> Dict[str, Any]:
        """Return custom options for one layer name.

        Args:
            name (str): Capsule layer name used as the custom options key.

        Returns:
            Dict[str, Any]: Custom options for `name`, or an empty dictionary when
            no layer-specific options are available.
        """
        options = getattr(self.options, "layer_options", {}) or {}
        if not isinstance(options, Mapping):
            return {}
        return dict(options.get(name) or {})


@dataclass(frozen=True)
class CapsuleLayer:
    """Declare one registered Capsule layer implementation.

    Args:
        name (str): Stable layer type stored in manifests.
        builder (Optional[Callable[[CapsuleLayerBuildContext], Mapping[str, Any] | None]]): Capture-time layer
            builder.
        restorer (Optional[Callable[..., Callable[..., Any]]]): Restore-time callable builder.
        verifier (Optional[Callable[..., None]]): Non-executing integrity verifier.
        priority (int): Default restore priority when a layer builder does not set one.
    """

    name: str
    builder: Optional[Callable[[CapsuleLayerBuildContext], Mapping[str, Any] | None]] = None
    restorer: Optional[Callable[..., Callable[..., Any]]] = None
    verifier: Optional[Callable[..., None]] = None
    priority: int = 100
