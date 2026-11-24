"""Composable Capsule layer publication and execution."""

from __future__ import annotations

__all__ = [
    "CapsuleLayer",
    "CapsuleLayerBuildContext",
    "build_capsule_layers",
    "capture_layer_names",
    "get_capsule_layer",
    "register_capsule_layer",
    "registered_capsule_layers",
    "restore_capsule_layer",
    "verify_capsule_layer",
]

from heavenbase.utils import Any, Callable, Dict, List, Mapping, Optional, raise_mismatch
from heavenbase.utils.config import ConfigManager
from heavenbase.utils.registry import named_key

from ..modules import ModuleResolver, ModuleService, module_resolver, python_executable_ref
from ..modules.base import ResolverCache
from .authority import CapsuleAuthority
from .layer import CapsuleLayer, CapsuleLayerBuildContext

_LAYER_CACHE = ResolverCache[CapsuleLayer]()


def register_capsule_layer(
    name: str,
    *,
    builder: Optional[Callable[[CapsuleLayerBuildContext], Mapping[str, Any] | None]] = None,
    restorer: Optional[Callable[..., Callable[..., Any]]] = None,
    verifier: Optional[Callable[..., None]] = None,
    priority: int = 100,
    replace: bool = False,
    resolver: ModuleResolver | None = None,
) -> None:
    """Publish or extend a programmatically owned Capsule layer record.

    Args:
        name: Stable layer type stored in Capsule manifests.
        builder: Optional capture-time builder.
        restorer: Optional restore-time builder.
        verifier: Optional non-executing integrity verifier.
        priority: Default restore priority.
        replace: Whether supplied callables may replace existing callables.
        resolver: Writable Context-owned module resolver. When omitted, the
            default Context is selected.

    Raises:
        ValueError: If no implementation is supplied, an implementation is
            duplicated, or another installation owns the record.

    Returns:
        None: This function does not return a value.
    """
    clean_name = named_key(name, "Capsule layer name")
    if builder is None and restorer is None and verifier is None:
        raise ValueError("Capsule layer needs a builder, restorer, or verifier")
    chosen = module_resolver(resolver)
    if not isinstance(chosen, ModuleService):
        raise TypeError("Capsule layer registration requires a writable ModuleService")
    service = chosen
    found = service.inspect("capsule_layer", clean_name)
    current = _layer_from_row(found[0], service) if found else None
    if found:
        coordinate = found[0]["record"]["meta"]["installation"]["coordinate"]
        if coordinate != f"programmatic/capsule_layer/{clean_name}":
            raise ValueError(f"Capsule layer {clean_name!r} belongs to {coordinate!r} and cannot be extended")
    if current is not None and not replace:
        duplicates = [
            label
            for label, existing, new in (
                ("builder", current.builder, builder),
                ("restorer", current.restorer, restorer),
                ("verifier", current.verifier, verifier),
            )
            if existing is not None and new is not None and existing is not new
        ]
        if duplicates:
            raise ValueError(f"Capsule layer {clean_name!r} already has {', '.join(duplicates)}")
    merged = CapsuleLayer(
        clean_name,
        builder=builder if builder is not None else (current.builder if current else None),
        restorer=restorer if restorer is not None else (current.restorer if current else None),
        verifier=verifier if verifier is not None else (current.verifier if current else None),
        priority=priority if current is None or priority != 100 else current.priority,
    )
    service.register(
        "capsule_layer",
        clean_name,
        {
            "name": clean_name,
            "desc": f"Capsule {clean_name} capture and restore layer.",
            "source": "inline",
            "target": "definition",
            "active": True,
            "meta": {
                "schema_version": 1,
                "tags": {"capsule": True, "executable": True},
                "definition": _layer_mapping(merged),
            },
        },
    )


def registered_capsule_layers(
    *,
    builders: bool = False,
    restorers: bool = False,
    resolver: ModuleResolver | None = None,
) -> List[str]:
    """Return registered Capsule layer identifiers.

    Args:
        builders: Include only layers with capture builders.
        restorers: Include only layers with restore functions.
        resolver: Context-owned module resolver. When omitted, the default
            Context is selected.

    Returns:
        List[str]: Matching layer names in Registry order.
    """
    chosen = module_resolver(resolver)
    layers = sorted(
        (_layer_from_row(row, chosen) for row in chosen.inspect("capsule_layer")),
        key=lambda layer: (layer.priority, layer.name),
    )
    if builders:
        layers = [layer for layer in layers if layer.builder is not None]
    if restorers:
        layers = [layer for layer in layers if layer.restorer is not None]
    return [layer.name for layer in layers]


def get_capsule_layer(name: str, *, resolver: ModuleResolver | None = None) -> CapsuleLayer:
    """Return one registered Capsule layer.

    Args:
        name: Layer identifier.
        resolver: Context-owned module resolver. When omitted, the default
            Context is selected.

    Returns:
        CapsuleLayer: Materialized layer implementation.

    Raises:
        ValueError: If the layer is unknown.
    """
    chosen = module_resolver(resolver)
    clean_name = str(name or "")
    try:
        rows = chosen.inspect("capsule_layer", clean_name)
    except ValueError:
        rows = ()
    if not rows:
        raise_mismatch(registered_capsule_layers(resolver=chosen), name, name="Capsule layer")
    return _layer_from_row(rows[0], chosen)


def _requested_layers(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


def capture_layer_names(options: Any, *, resolver: ModuleResolver | None = None) -> tuple[str, ...]:
    """Return ordered capture layer names for the supplied options.

    Args:
        options: Capture options with layer selection fields.
        resolver: Context-owned module resolver. When omitted, the default
            Context is selected.

    Returns:
        tuple[str, ...]: Ordered layer identifiers.
    """
    chosen = module_resolver(resolver)
    requested = _requested_layers(getattr(options, "layers", None))
    if requested:
        names = requested
    else:
        names = ("source",)
        if bool(getattr(options, "prefer_import", True)):
            names = (*names, "import")
        if bool(getattr(options, "include_cloudpickle", False)):
            names = (*names, "cloudpickle")
    known = registered_capsule_layers(builders=True, resolver=chosen)
    for name in names:
        if name not in known:
            raise_mismatch(known, name, name="Capsule capture layer")
    return names


def build_capsule_layers(
    fn: Callable[..., Any],
    options: Any,
    *,
    resolver: ModuleResolver | None = None,
    config: ConfigManager | None = None,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Build restore-layer payloads through registered layer builders.

    Args:
        fn: Callable being captured.
        options: Capture options.
        resolver: Context-owned module resolver. When omitted, the default
            Context is selected.
        config: Config authority paired with `resolver`. When omitted, the
            standalone process config is selected.

    Returns:
        tuple[List[Dict[str, Any]], Dict[str, Any]]: Layer payloads and shared
        build metadata.
    """
    authority = CapsuleAuthority.resolve(resolver=resolver, config=config)
    chosen = authority.resolver
    context = CapsuleLayerBuildContext(fn=fn, options=options, resolver=chosen, config=authority.config)
    names = capture_layer_names(options, resolver=chosen)
    context.data["layer_names"] = list(names)
    layers: List[Dict[str, Any]] = []
    for index, name in enumerate(names):
        layer = get_capsule_layer(name, resolver=chosen)
        if layer.builder is None:
            raise ValueError(f"Capsule layer {name!r} cannot build capture payloads")
        payload = layer.builder(context)
        if not payload:
            continue
        data = dict(payload)
        data.setdefault("type", name)
        data.setdefault("priority", layer.priority + index)
        layers.append(data)
    return layers, context.data


def restore_capsule_layer(
    manifest: Any,
    layer: Mapping[str, Any],
    policy: Any,
    *,
    resolver: ModuleResolver | None = None,
) -> Callable[..., Any]:
    """Restore one callable through the layer named by a manifest payload.

    Args:
        manifest (Any): Capsule manifest owning the restore layers.
        layer (Mapping[str, Any]): One restore-layer payload from the manifest.
        policy (Any): Restore policy passed to the layer restorer.
        resolver (ModuleResolver | None): Context-owned module resolver.
            The default Context is selected when omitted.

    Returns:
        Callable[..., Any]: Restored Python callable.

    Raises:
        ValueError: If the named layer cannot restore callables.
    """
    spec = get_capsule_layer(str(layer.get("type") or ""), resolver=resolver)
    if spec.restorer is None:
        raise ValueError(f"Capsule layer {spec.name!r} cannot restore callables")
    return spec.restorer(manifest, dict(layer), policy)


def verify_capsule_layer(
    manifest: Any,
    layer: Mapping[str, Any],
    policy: Any,
    *,
    resolver: ModuleResolver | None = None,
) -> None:
    """Verify one Capsule layer without executing its payload.

    Args:
        manifest (Any): Capsule manifest owning the restore layers.
        layer (Mapping[str, Any]): One restore-layer payload from the manifest.
        policy (Any): Restore policy passed to the layer verifier.
        resolver (ModuleResolver | None): Context-owned module resolver.
            The default Context is selected when omitted.

    Returns:
        None: This function does not return a value.
    """
    chosen = module_resolver(resolver)
    layer_type = str(layer.get("type") or "")
    if not layer_type:
        return
    rows = chosen.inspect("capsule_layer", layer_type)
    if not rows:
        return
    spec = _layer_from_row(rows[0], chosen)
    if spec.verifier is not None:
        spec.verifier(manifest, dict(layer), policy)


def _layer_mapping(layer: CapsuleLayer) -> dict[str, Any]:
    """Return the JSON-safe definition for one Capsule layer."""
    return {
        "builder": python_executable_ref(layer.builder, label=f"Capsule layer {layer.name!r} builder") if layer.builder is not None else None,
        "restorer": python_executable_ref(layer.restorer, label=f"Capsule layer {layer.name!r} restorer") if layer.restorer is not None else None,
        "verifier": python_executable_ref(layer.verifier, label=f"Capsule layer {layer.name!r} verifier") if layer.verifier is not None else None,
        "priority": int(layer.priority),
    }


def _layer_from_row(row: Mapping[str, Any], resolver: ModuleResolver) -> CapsuleLayer:
    """Materialize and cache one Capsule layer inspection row."""

    def build(current_row: Mapping[str, Any]) -> CapsuleLayer:
        identifier = str(current_row["identifier"])
        raw = resolver.resolve("capsule_layer", identifier)
        if not isinstance(raw, Mapping):
            raise TypeError(f"Capsule layer {identifier!r} definition must be a mapping")

        def restored(field: str) -> Any:
            reference = raw.get(field)
            if reference is None:
                return None
            if not isinstance(reference, Mapping):
                raise TypeError(f"Capsule layer {identifier!r} {field} must be an executable reference")
            return resolver.resolve_executable(reference, owner=current_row)

        return CapsuleLayer(
            identifier,
            builder=restored("builder"),
            restorer=restored("restorer"),
            verifier=restored("verifier"),
            priority=int(raw.get("priority", 100)),
        )

    return _LAYER_CACHE.get(resolver, row, build)
