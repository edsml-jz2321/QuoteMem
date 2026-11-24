"""Capsule entity runtime and local callable execution."""

from __future__ import annotations

__all__ = ["Capsule"]

from dataclasses import replace
import inspect
from time import time

from heavenbase.entity import Entity, field
from heavenbase.types import Float, Json, LongText, ShortText
from heavenbase.utils import Any, Callable, Dict, List, Mapping, dumps_json, dumps_yaml, loads_json, loads_yaml, sha256hash
from heavenbase.utils.config import ConfigManager

from ..modules import ModuleResolver
from .authority import CapsuleAuthority
from .capture import CaptureOptions, capture_function
from .docstring import synthesize_docstring
from .manifest import CapsuleManifest
from .restore import RestorePolicy, restore_callable


def _default_namespace(config: ConfigManager) -> str:
    return str(config.get("heavenbase.capsule.namespace", default="default") or "default")


class Capsule(Entity):
    """Executable Capsule with lazy callable restoration.

    A Capsule stores enough source, import, signature, and manifest metadata to restore a Python callable later. It is the
    execution unit behind Prompt and Toolkit persistence. It is also the system entity registered as ``sys-capsule``;
    registry storage uses :meth:`to_row` so the public :meth:`to_dict` manifest contract remains stable.

    Warnings:
        Loading or running a Capsule can execute restored Python code. Only load Capsules from trusted registries or
        manifests that have been reviewed for the current runtime.
    """

    identifier = "sys-capsule"

    name = field(ShortText).desc("Public capsule name")
    namespace = field(ShortText).default("default").desc("Capsule namespace")
    version = field(ShortText).default("1").desc("Explicit user version")
    status = field(ShortText).default("active").desc("Capsule status")
    description = field(LongText).default("").desc("Human-readable description")
    tags = field(Json).default([]).desc("Search tags")
    manifest = field(Json).default({}).desc("Full Capsule manifest")
    capabilities = field(Json).default({}).desc("Export capability matrix")
    fingerprints = field(Json).default({}).desc("Integrity fingerprints")
    last_reason = field(LongText).default("").desc("Last registry write reason")
    created_at = field(Float).default(0.0).desc("Creation timestamp")
    updated_at = field(Float).default(0.0).desc("Last update timestamp")

    def __init__(
        self,
        source: CapsuleManifest | Mapping[str, Any] | Callable[..., Any] | None = None,
        *,
        fn: Callable[..., Any] | None = None,
        register: bool = False,
        registry: Any = None,
        registry_config: Mapping[str, Any] | None = None,
        overwrite: bool = False,
        reason: str = "",
        resolver: ModuleResolver | None = None,
        config: ConfigManager | None = None,
        **options: Any,
    ) -> None:
        """Create a Capsule from a callable, manifest, or decorator options.

        Args:
            source (CapsuleManifest | Mapping[str, Any] | Callable[..., Any] | None): Callable to capture, manifest to
                restore from, or ``None`` when using ``Capsule(...)`` as a decorator.
            fn (Callable[..., Any] | None): Live callable paired with a prebuilt manifest. Defaults to ``None``.
            register (bool): Whether decorator binding should register the Capsule. Defaults to ``False``.
            registry (Any): Optional registry object used when ``register`` is true. Defaults to ``None``.
            registry_config (Mapping[str, Any] | None): Optional registry config used when ``register`` is true.
                Defaults to ``None``.
            overwrite (bool): Whether decorator registration may replace an existing row. Defaults to ``False``.
            reason (str): Registry write reason for decorator registration. Defaults to ``""``.
            resolver (ModuleResolver | None): Context-owned module resolver
                used for Capsule layers. When omitted, the default Context is
                selected.
            config (ConfigManager | None): Config authority paired with
                `resolver`. When omitted, the standalone process config is
                selected.
            **options (Any): Capture options such as ``name``, ``namespace``, ``version``, ``description``, ``tags``,
                ``layers``, ``layer_options``, ``include_cloudpickle``, and ``prefer_import``.

        Returns:
            None: This constructor does not return a value.
        """
        super().__init__()
        self.manifest: CapsuleManifest | None = None
        self._callable: Callable[..., Any] | None = None
        self.warnings: List[Any] = []
        self._authority = CapsuleAuthority.resolve(resolver=resolver, config=config)
        self._resolver = self._authority.resolver
        self._config = self._authority.config
        self._capture_options = self._normalize_capture_options(options)
        self._capsule_register_on_bind = bool(register)
        self._capsule_registry = registry
        self._capsule_registry_config = registry_config
        self._capsule_overwrite = bool(overwrite)
        self._capsule_reason = str(reason or "decorator")
        if isinstance(source, CapsuleManifest):
            self.manifest = source
            self.object_id = source.id
            self._callable = fn
            self._sync_func_attrs(fn)
        elif isinstance(source, Mapping):
            self.manifest = CapsuleManifest.from_dict(source)
            self.object_id = self.manifest.id
            self._callable = fn
            self._sync_func_attrs(fn)
        elif callable(source):
            self._bind_func(source)
        elif source is None:
            if fn is not None:
                self._bind_func(fn)
            else:
                self._sync_func_attrs()
        else:
            raise TypeError("Capsule source must be a callable, manifest mapping, CapsuleManifest, or None")

    @staticmethod
    def _normalize_capture_options(options: Mapping[str, Any]) -> Dict[str, Any]:
        keys = set(CaptureOptions.__dataclass_fields__)
        unknown = sorted(set(options) - keys)
        if unknown:
            raise TypeError(f"Unsupported Capsule capture option(s): {', '.join(unknown)}")
        return dict(options)

    def _display_name(self) -> str:
        if self.manifest is not None:
            return self.manifest.name
        return str(self._capture_options.get("name") or "capsule")

    def _require_manifest(self) -> CapsuleManifest:
        if self.manifest is None:
            raise RuntimeError(f"Capsule {self._display_name()!r} is not bound to a function")
        return self.manifest

    def _sync_func_attrs(self, fn: Callable[..., Any] | None = None) -> None:
        manifest = self.manifest
        name = manifest.name if manifest is not None else str(self._capture_options.get("name") or getattr(fn, "__name__", "capsule"))
        identity = manifest.identity if manifest is not None else {}
        self.__name__ = name
        self.__qualname__ = str(identity.get("qualname") or getattr(fn, "__qualname__", name))
        self.__module__ = str(identity.get("module") or getattr(fn, "__module__", ""))
        self.__doc__ = (manifest.description if manifest is not None else self._capture_options.get("description")) or getattr(fn, "__doc__", None)
        self.__annotations__ = dict(getattr(fn, "__annotations__", {}) or {}) if fn is not None else {}

    def _bind_func(self, fn: Callable[..., Any]) -> "Capsule":
        if not callable(fn):
            raise TypeError("Capsule can only bind callable objects")
        self.manifest = capture_function(fn, CaptureOptions(**self._capture_options), resolver=self._resolver, config=self._config)
        self.object_id = self.manifest.id
        self._callable = fn
        self._sync_func_attrs(fn)
        if self._capsule_register_on_bind:
            self.register(
                registry=self._capsule_registry,
                registry_config=self._capsule_registry_config,
                overwrite=self._capsule_overwrite,
                reason=self._capsule_reason,
            )
        return self

    def _bind_authority(
        self,
        *,
        resolver: ModuleResolver | None = None,
        config: ConfigManager | None = None,
    ) -> None:
        """Replace the bound authority before a workspace-owned restore."""
        self._authority = self._authority.replace(resolver=resolver, config=config)
        self._resolver = self._authority.resolver
        self._config = self._authority.config

    @classmethod
    def from_func(cls, fn: Callable[..., Any], **options: Any) -> "Capsule":
        """Capture a callable into a Capsule.

        Args:
            fn (Callable[..., Any]): Python callable to capture.
            **options (Any): Capture options such as ``name``, ``namespace``, ``version``, ``description``, ``tags``,
                ``layers``, ``layer_options``, ``include_cloudpickle``, and ``prefer_import``.

        Returns:
            Capsule: A lazy executable Capsule containing a durable manifest and the original callable for immediate local use.

        Warnings:
            ``include_cloudpickle`` defaults to ``False``. Enable it only for reviewed trusted-local code that cannot be
            restored from source or import path.

        Examples:
            >>> capsule = Capsule.from_func(add, name="add", namespace="demo", version="1")
            >>> capsule.run(left=2, right=3)
            5
        """
        return cls(fn, **options)

    @classmethod
    def from_code(cls, code: str, fn_name: str | None = None, **options: Any) -> "Capsule":
        """Build a Capsule from a code string containing a function.

        Args:
            code (str): Python source text that defines at least one callable.
            fn_name (str | None): Function name to capture. Defaults to the first public callable in ``code``.
            **options (Any): Capture options forwarded to ``Capsule`` callable capture.

        Returns:
            Capsule: Capsule whose source restore layer uses the provided code string.

        Warnings:
            This method executes ``code`` once during capture. Use it only for trusted local code.

        Examples:
            ```python
            capsule = Capsule.from_code(
                "def add(left, right):\\n    return left + right\\n",
                fn_name="add",
            )
            capsule.run(left=2, right=3)
            ```
        """
        scope: Dict[str, Any] = {}
        exec(compile(code, "<capsule-code>", "exec"), scope, scope)
        selected_name = fn_name or next(name for name, value in scope.items() if callable(value) and not name.startswith("_"))
        capsule = cls(scope[selected_name], name=options.pop("name", selected_name), **options)
        source_hash = sha256hash(code)
        found = False
        for layer in capsule.manifest.restore_layers:
            if layer.get("type") == "source":
                layer["source"] = code
                layer["source_hash"] = source_hash
                layer["qualname"] = selected_name
                layer["filename"] = "<capsule-code>"
                found = True
        if not found:
            capsule.manifest.restore_layers.insert(
                0,
                {
                    "type": "source",
                    "priority": 10,
                    "source": code,
                    "source_hash": source_hash,
                    "qualname": selected_name,
                    "filename": "<capsule-code>",
                    "global_refs": {},
                },
            )
        capsule.manifest.source.update({"ok": True, "source": code, "source_hash": source_hash, "filename": "<capsule-code>", "qualname": selected_name})
        capsule.manifest.fingerprints["source_hash"] = source_hash
        capsule.manifest.fingerprints["restore_hash"] = sha256hash(
            [{key: value for key, value in layer.items() if key != "payload_b64"} for layer in capsule.manifest.restore_layers]
        )
        return capsule

    @classmethod
    def from_manifest(
        cls,
        data: CapsuleManifest | Mapping[str, Any],
        *,
        resolver: ModuleResolver | None = None,
        config: ConfigManager | None = None,
    ) -> "Capsule":
        """Build a Capsule from a manifest object or dictionary.

        Args:
            data (CapsuleManifest | Mapping[str, Any]): Manifest object or JSON-compatible manifest dictionary.
            resolver (ModuleResolver | None): Context-owned module resolver
                used for lazy restoration.
            config (ConfigManager | None): Config authority paired with
                `resolver` for lazy restoration.

        Returns:
            Capsule: Lazy Capsule that restores only when called or converted to a function.
        """
        manifest = data if isinstance(data, CapsuleManifest) else CapsuleManifest.from_dict(data)
        return cls(manifest, resolver=resolver, config=config)

    @classmethod
    def from_row(
        cls,
        row: Mapping[str, Any],
        *,
        resolver: ModuleResolver | None = None,
        config: ConfigManager | None = None,
    ) -> "Capsule":
        """Build a runtime Capsule from a stored ``sys-capsule`` row.

        Args:
            row (Mapping[str, Any]): Persistent Capsule row, usually including
                a ``manifest`` payload.
            resolver (ModuleResolver | None): Context-owned module resolver
                used for lazy restoration.
            config (ConfigManager | None): Config authority paired with
                `resolver` for lazy restoration.

        Returns:
            Capsule: Lazy Capsule restored from the row payload.
        """
        manifest = dict(row.get("manifest") or {})
        if not manifest:
            manifest = {
                "id": row.get("object_id"),
                "name": row.get("name"),
                "namespace": row.get("namespace") or "default",
                "version": str(row.get("version") or "1"),
                "status": row.get("status") or "active",
                "description": row.get("description") or "",
                "tags": list(row.get("tags") or []),
                "capabilities": dict(row.get("capabilities") or {}),
                "fingerprints": dict(row.get("fingerprints") or {}),
                "identity": {},
                "signature": {},
                "input_schema": {},
                "output_schema": {},
                "source": {},
                "restore_layers": [],
                "trust": {},
                "metadata": {},
            }
        return cls.from_manifest(manifest, resolver=resolver, config=config)

    @classmethod
    def from_str(
        cls,
        text: str,
        format: str = "json",
        *,
        resolver: ModuleResolver | None = None,
        config: ConfigManager | None = None,
    ) -> "Capsule":
        """Build a Capsule from a JSON or YAML manifest string.

        Args:
            text (str): Serialized Capsule manifest.
            format (str): ``"json"`` or ``"yaml"``. Defaults to ``"json"``.
            resolver (ModuleResolver | None): Context-owned module resolver
                used for lazy restoration.
            config (ConfigManager | None): Config authority paired with
                `resolver` for lazy restoration.

        Returns:
            Capsule: Lazy Capsule restored from the serialized manifest.
        """
        return cls.from_manifest(loads_yaml(text) if format == "yaml" else loads_json(text), resolver=resolver, config=config)

    @classmethod
    def load(
        cls,
        capsule_id: str | None = None,
        *,
        name: str | None = None,
        namespace: str | None = None,
        version: str | None = None,
        registry: Any = None,
        registry_config: Mapping[str, Any] | None = None,
        resolver: ModuleResolver | None = None,
        config: ConfigManager | None = None,
    ) -> "Capsule":
        """Load a Capsule from the default or provided registry.

        Args:
            capsule_id (str | None): Stable Capsule id. Defaults to ``None``.
            name (str | None): Capsule name when loading by identity. Defaults to ``None``.
            namespace (str | None): Capsule namespace. Defaults to configured Capsule namespace.
            version (str | None): Capsule version. Defaults to latest matching row.
            registry (Any): Optional registry object. Defaults to ``None``.
            registry_config (Mapping[str, Any] | None): Optional registry config. Defaults to ``None``.
            resolver (ModuleResolver | None): Context-owned module resolver
                used for lazy restoration.
            config (ConfigManager | None): Config authority paired with
                `resolver` for registry and trust decisions.

        Returns:
            Capsule: Lazy Capsule object.

        Warnings:
            Loading verifies stored payload checksums once per process/cache key. It does not compare with the current live function source.
        """
        from .registry import capsule_registry

        if registry is None:
            authority = CapsuleAuthority.resolve(resolver=resolver, config=config)
            store = capsule_registry(registry_config, resolver=authority.resolver, config=authority.config)
        else:
            store = registry
            authority = getattr(store, "authority", None) or CapsuleAuthority.resolve(resolver=resolver, config=config)
        return store.load(
            capsule_id,
            name=name,
            namespace=namespace or _default_namespace(authority.config),
            version=version,
        )

    @classmethod
    def list(
        cls,
        *,
        registry: Any = None,
        registry_config: Mapping[str, Any] | None = None,
        namespace: str | None = None,
        status: str | None = "active",
        resolver: ModuleResolver | None = None,
        config: ConfigManager | None = None,
    ) -> List[Dict[str, Any]]:
        """List Capsule registry rows.

        Args:
            registry (Any): Optional registry instance. Defaults to ``None``.
            registry_config (Mapping[str, Any] | None): Optional isolated registry config. Defaults to ``None``.
            namespace (str | None): Optional namespace filter. Defaults to ``None``.
            status (str | None): Optional status filter. Defaults to ``"active"``.
            resolver (ModuleResolver | None): Context-owned module resolver.
            config (ConfigManager | None): Config authority paired with
                `resolver`.

        Returns:
            List[Dict[str, Any]]: Registry rows sorted by the backing workspace order.
        """
        from .registry import capsule_registry

        if registry is not None:
            return registry.list(namespace=namespace, status=status)
        authority = CapsuleAuthority.resolve(resolver=resolver, config=config)
        return capsule_registry(registry_config, resolver=authority.resolver, config=authority.config).list(
            namespace=namespace,
            status=status,
        )

    @classmethod
    def delete(
        cls,
        capsule_id: str,
        *,
        registry: Any = None,
        registry_config: Mapping[str, Any] | None = None,
        tombstone: bool = True,
        resolver: ModuleResolver | None = None,
        config: ConfigManager | None = None,
    ) -> bool:
        """Delete or tombstone a Capsule registry row.

        Args:
            capsule_id (str): Stable Capsule id to remove.
            registry (Any): Optional registry instance. Defaults to ``None``.
            registry_config (Mapping[str, Any] | None): Optional isolated registry config. Defaults to ``None``.
            tombstone (bool): Mark deleted instead of physically deleting. Defaults to ``True``.
            resolver (ModuleResolver | None): Context-owned module resolver.
            config (ConfigManager | None): Config authority paired with
                `resolver`.

        Returns:
            bool: Whether a row was removed or tombstoned.
        """
        from .registry import capsule_registry

        if registry is not None:
            return registry.delete(capsule_id, tombstone=tombstone)
        authority = CapsuleAuthority.resolve(resolver=resolver, config=config)
        return capsule_registry(registry_config, resolver=authority.resolver, config=authority.config).delete(
            capsule_id,
            tombstone=tombstone,
        )

    def register(self, *, registry: Any = None, registry_config: Mapping[str, Any] | None = None, overwrite: bool = False, reason: str = "") -> str:
        """Register this Capsule in the default or provided registry.

        Args:
            registry (Any): Optional registry object. Defaults to ``None``.
            registry_config (Mapping[str, Any] | None): Optional registry config for isolated stores. Defaults to ``None``.
            overwrite (bool): Whether an existing same-id Capsule may be replaced. Defaults to ``False``.
            reason (str): Revision reason stored with the append-only revision row. Defaults to ``""``.

        Returns:
            str: Stable Capsule id.

        Warnings:
            Registry checksums verify loaded records against their stored hashes only. They do not audit whether the code itself is safe.
        """
        from .registry import capsule_registry

        return (
            registry
            or capsule_registry(
                registry_config,
                resolver=self._authority.resolver,
                config=self._authority.config,
            )
        ).register(self, overwrite=overwrite, reason=reason)

    def verify(
        self,
        policy: RestorePolicy | None = None,
        *,
        resolver: ModuleResolver | None = None,
        config: ConfigManager | None = None,
    ) -> None:
        """Verify stored source and binary checksums without executing code.

        Args:
            policy (RestorePolicy | None): Optional restore policy controlling checksum verification. Defaults to ``None``.
            resolver (ModuleResolver | None): Optional resolver override. The
                Capsule-bound resolver is used when omitted.
            config (ConfigManager | None): Optional config override. The
                Capsule-bound config is used when omitted.

        Returns:
            None: Raises when a stored checksum does not match its payload.
        """
        authority = self._authority.replace(resolver=resolver, config=config)
        self._require_manifest().verify(policy=policy, resolver=authority.resolver, config=authority.config)

    def to_dict(self) -> Dict[str, Any]:
        """Return the underlying manifest dictionary.

        Args:
            None.

        Returns:
            Dict[str, Any]: JSON-compatible Capsule manifest payload.
        """
        return self._require_manifest().to_dict()

    def to_row(self, existing: Mapping[str, Any] | None = None, *, reason: str = "") -> Dict[str, Any]:
        """Return the persistent ``sys-capsule`` row for this Capsule.

        Args:
            existing (Mapping[str, Any] | None): Optional prior row whose
                ``created_at`` timestamp should be preserved.
            reason (str): Registry write reason stored on the row. Defaults to
                ``""``.

        Returns:
            Dict[str, Any]: Row payload suitable for ``sys-capsule`` persistence.
        """
        now = time()
        manifest = self._require_manifest()
        return {
            "object_id": manifest.id,
            "name": manifest.name,
            "namespace": manifest.namespace,
            "version": manifest.version,
            "status": manifest.status,
            "description": manifest.description,
            "tags": list(manifest.tags),
            "manifest": manifest.to_dict(),
            "capabilities": dict(manifest.capabilities),
            "fingerprints": dict(manifest.fingerprints),
            "last_reason": reason,
            "created_at": (existing or {}).get("created_at") or now,
            "updated_at": now,
        }

    def to_str(self, format: str = "json") -> str:
        """Return this Capsule as json, yaml, source, or synthesized docstring.

        Args:
            format (str): Output format. Supported values are ``"json"``,
                ``"yaml"``, ``"source"``, and ``"docstring"``. Defaults to
                ``"json"``.

        Returns:
            str: Serialized Capsule representation for the requested format.
        """
        manifest = self._require_manifest()
        if format == "source":
            return str(manifest.source.get("source") or "")
        if format == "docstring":
            return synthesize_docstring(
                manifest.name,
                manifest.description,
                manifest.signature.get("parameters") or [],
                manifest.signature.get("return_annotation") or "",
            )
        if format == "yaml":
            return dumps_yaml(self.to_dict())
        return dumps_json(self.to_dict(), indent=4, sort_keys=True)

    def to_func(
        self,
        policy: RestorePolicy | None = None,
        *,
        layers: List[str] | None = None,
        resolver: ModuleResolver | None = None,
        config: ConfigManager | None = None,
    ) -> Callable[..., Any]:
        """Return the executable callable, restoring it lazily when needed.

        Args:
            policy (RestorePolicy | None): Optional restore policy. Defaults to ``None`` and uses manifest/config trust settings.
            layers (List[str] | None): Optional restore layer order/filter. Defaults to ``None``.
            resolver (ModuleResolver | None): Optional resolver override. The
                resolver bound at construction is used when omitted.
            config (ConfigManager | None): Optional config override. The
                config bound at construction is used when omitted.

        Returns:
            Callable[..., Any]: Restored Python callable.

        Warnings:
            Restoring a Capsule executes Python code. Treat Capsule registries like source repositories.
        """
        manifest = self._require_manifest()
        if self._callable is None:
            if layers is not None:
                policy = replace(policy or RestorePolicy(), layers=tuple(layers))
            authority = self._authority.replace(resolver=resolver, config=config)
            self._callable = restore_callable(
                manifest,
                policy=policy,
                resolver=authority.resolver,
                config=authority.config,
            )
            self._sync_func_attrs(self._callable)
        return self._callable

    @property
    def __signature__(self) -> Any:
        """Return the callable signature for inspect.signature."""
        if self.manifest is None:
            return inspect.Signature([inspect.Parameter("fn", inspect.Parameter.POSITIONAL_OR_KEYWORD)])
        return inspect.signature(self.to_func())

    def __repr__(self) -> str:
        """Return a function-like debug representation."""
        if self.manifest is None:
            namespace = self._capture_options.get("namespace") or _default_namespace(self._config)
            version = self._capture_options.get("version") or "1"
            return f"<Capsule {namespace}/{self._display_name()}:{version} unbound>"
        return f"<Capsule {self.manifest.namespace}/{self.manifest.name}:{self.manifest.version}>"

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Call the Capsule like the wrapped Python function."""
        if self.manifest is None and len(args) == 1 and callable(args[0]) and not kwargs:
            return self._bind_func(args[0])
        return self.run(*args, **kwargs)

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the Capsule locally and return the native Python value.

        Args:
            *args (Any): Positional arguments for the callable.
            **kwargs (Any): Keyword arguments for the callable.

        Returns:
            Any: Callable return value.
        """
        return self.to_func()(*args, **kwargs)

    async def arun(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the Capsule and await awaitable results.

        Args:
            *args (Any): Positional arguments for the callable.
            **kwargs (Any): Keyword arguments for the callable.

        Returns:
            Any: Awaited result for coroutine returns, otherwise the direct result.
        """
        value = self.run(*args, **kwargs)
        if inspect.isawaitable(value):
            return await value
        return value
