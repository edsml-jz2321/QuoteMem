"""HeavenBase executable Capsules."""

from __future__ import annotations

from heavenbase._lazy import _attach_lazy_exports

__all__ = [
    "Capsule",
    "CapsuleCapabilityError",
    "CapsuleCaptureError",
    "CapsuleError",
    "CapsuleManifest",
    "CapsuleRegistry",
    "CapsuleLayer",
    "CapsuleLayerBuildContext",
    "CapsuleRestoreError",
    "RestorePolicy",
    "capsule_registry",
    "infer_capabilities",
    "inspect_callable_signature",
    "make_capsule_id",
    "parse_docstring",
    "register_capsule_layer",
    "registered_capsule_layers",
    "reset_capsule_registry",
    "setup_capsule_registry",
    "synthesize_docstring",
]

_LAZY_EXPORTS = {
    "Capsule": (".runtime", "Capsule"),
    "CapsuleCapabilityError": (".errors", "CapsuleCapabilityError"),
    "CapsuleCaptureError": (".errors", "CapsuleCaptureError"),
    "CapsuleError": (".errors", "CapsuleError"),
    "CapsuleManifest": (".manifest", "CapsuleManifest"),
    "CapsuleRegistry": (".registry", "CapsuleRegistry"),
    "CapsuleLayer": (".layer", "CapsuleLayer"),
    "CapsuleLayerBuildContext": (".layer", "CapsuleLayerBuildContext"),
    "CapsuleRestoreError": (".errors", "CapsuleRestoreError"),
    "RestorePolicy": (".restore", "RestorePolicy"),
    "capsule_registry": (".registry", "capsule_registry"),
    "infer_capabilities": (".signature", "infer_capabilities"),
    "inspect_callable_signature": (".signature", "inspect_callable_signature"),
    "make_capsule_id": (".manifest", "make_capsule_id"),
    "parse_docstring": (".docstring", "parse_docstring"),
    "register_capsule_layer": (".layers", "register_capsule_layer"),
    "registered_capsule_layers": (".layers", "registered_capsule_layers"),
    "reset_capsule_registry": (".registry", "reset_capsule_registry"),
    "setup_capsule_registry": (".registry", "setup_capsule_registry"),
    "synthesize_docstring": (".docstring", "synthesize_docstring"),
}

__getattr__, __dir__ = _attach_lazy_exports(__name__, globals(), _LAZY_EXPORTS, __all__)
