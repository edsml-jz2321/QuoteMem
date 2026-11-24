"""Canonical LLM configuration built on the universal config engine."""

from __future__ import annotations

__all__ = ["LLMEngine", "LLMSpec"]

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from heavenbase.context.authority import resolve_runtime_authority
from heavenbase.modules import ModuleResolver, module_resolver
from heavenbase.utils import dmerge, sha256hash
from heavenbase.utils.config import ConfigEngine, ConfigSpec
from heavenbase.utils.config.data import to_mutable_data
from heavenbase.utils.config.engine import redact_paths, redact_url

from .gateway import LLMGateway, _cfg_value, _provider_backend, _truthy

if TYPE_CHECKING:
    from heavenbase.context import Context
    from heavenbase.utils.config import ConfigManager

_ROUTE_KEYS = frozenset({"preset", "endpoint", "gateway", "provider", "model", "args", "default_args", "desc"})
_REASONING_KEYS = ("think", "reasoning_effort", "reasoning_max_tokens", "reasoning_exclude")


def _clean_alias(value: Any) -> str:
    return str(value or "").replace("-", "").replace("_", "").replace(".", "").replace(":", "").replace(" ", "").lower()


@dataclass(frozen=True, eq=False)
class LLMSpec(ConfigSpec):
    """Store canonical semantic LLM config plus one executable route binding."""

    _engine: "LLMEngine | None" = field(default=None, repr=False, compare=False)
    _resolver: ModuleResolver | None = field(default=None, repr=False, compare=False)

    @property
    def engine(self) -> "LLMEngine":
        """Return the configuration authority that created this spec."""
        return self._engine or _default_engine()

    @property
    def resolver(self) -> ModuleResolver:
        """Return the module authority that owns this spec's gateway."""
        return module_resolver(self._resolver)

    @property
    def authority_key(self) -> tuple[str | int, int]:
        """Return resolver identity and publication for runtime cache keys."""
        resolver = self.resolver
        return getattr(resolver, "authority_token", id(resolver)), resolver.catalog.publication

    @property
    def preset(self) -> str | None:
        """Return the selected preset identifier, when enabled."""
        return self.get("preset")

    @property
    def endpoint(self) -> str:
        """Return the canonical request/response grammar identifier."""
        return str(self.get("endpoint") or "")

    @property
    def gateway(self) -> str:
        """Return the canonical transport gateway identifier."""
        return str(self.get("gateway") or "")

    @property
    def provider(self) -> str:
        """Return the canonical model provider identifier."""
        return str(self.get("provider") or "")

    @property
    def model(self) -> str:
        """Return the canonical HeavenBase model identifier."""
        return str(self.get("model") or "")

    @property
    def model_id(self) -> str:
        """Return the selected provider-native model identifier."""
        return str(self.bindings.get("model_id") or "")

    @property
    def args(self) -> Mapping[str, Any]:
        """Return immutable canonical semantic request arguments."""
        return self.get("args", {})

    @property
    def dim(self) -> int | None:
        """Return the configured embedding dimension, when present."""
        value = self.args.get("dim")
        return int(value) if value is not None else None

    def provider_backend(self, gateway: str | None = None) -> str:
        """Return the bound provider backend for this route.

        Args:
            gateway (str | None): Optional route assertion. A different
                gateway requires re-resolving the semantic config.

        Returns:
            str: Provider backend identifier consumed by the gateway.

        Raises:
            ValueError: If ``gateway`` differs from the bound gateway.
        """
        if gateway is not None and gateway != self.gateway:
            raise ValueError("Changing an LLM gateway requires re-resolving the semantic config")
        return str(self.bindings.get("provider_backend") or self.provider)

    def with_args(self, args: Mapping[str, Any]) -> "LLMSpec":
        """Return a copy with semantic or runtime-only arguments overlaid.

        Args:
            args (Mapping[str, Any]): Request arguments merged over canonical
                semantic defaults. Callable values make config Apply uncached.

        Returns:
            LLMSpec: Independent executable specification.
        """
        data = to_mutable_data(self.data)
        patch = to_mutable_data(args)
        if "thinking_budget_tokens" in patch:
            raise ValueError("LLM thinking_budget_tokens was removed; use reasoning_max_tokens")
        data["args"] = dmerge([to_mutable_data(self.args), patch])
        return LLMSpec(
            self.domain,
            data,
            bindings=self.bindings,
            raw=self.raw,
            meta=self.meta,
            _engine=self._engine,
            _resolver=self._resolver,
        )

    def hash_key(self) -> str:
        """Return the complete stable applied LLM identity digest.

        Args:
            None.

        Returns:
            str: SHA-256 digest of canonical and selected runtime values.
        """
        return str(self.engine.apply(self, profile="cache-key"))

    def client_key(self) -> str:
        """Return a stable client-construction identity digest.

        Args:
            None.

        Returns:
            str: SHA-256 digest including route credentials and headers.
        """
        runtime = self.engine.apply(self)
        args = runtime.get("args", {}) or {}
        return sha256hash(
            {
                "endpoint": self.endpoint,
                "gateway": self.gateway,
                "authority": self.authority_key,
                "api_key": runtime.get("api_key"),
                "auth_token": runtime.get("auth_token"),
                "base_url": runtime.get("base_url") or runtime.get("api_base"),
                "headers": runtime.get("headers", {}) or {},
                "timeout": args.get("timeout"),
                "max_retries": args.get("max_retries"),
            }
        )

    def _provider_cfg(self) -> dict[str, Any]:
        return to_mutable_data(self.bindings.get("provider", {}))

    def _gateway_cfg(self, gateway: str | None = None) -> dict[str, Any]:
        if gateway is not None and gateway != self.gateway:
            raise ValueError("Changing an LLM gateway requires re-resolving the semantic config")
        return to_mutable_data(self.bindings.get("gateway", {}))

    def _model_cfg(self) -> dict[str, Any]:
        return to_mutable_data(self.bindings.get("model", {}))


class _DefaultAuthority:
    """Sentinel distinguishing an omitted manager from explicit ``None``."""


_DEFAULT = _DefaultAuthority()


class LLMEngine(ConfigEngine):
    """Resolve LLM presets and semantic options into one bound route.

    Args:
        config (Mapping[str, Any] | None): Optional frozen LLM-root overlay.
        manager (ConfigManager | None): Manager supplying bundled and scoped runtime data.
            Omission selects ``DEFAULT_CONTEXT.config``.
        resolver (ModuleResolver | None): Context-owned gateway catalog.
            Omission selects ``DEFAULT_CONTEXT.modules()``.
        context (Context | None): Context supplying both manager and resolver. It is
            mutually exclusive with explicit ``manager`` or ``resolver``.
    """

    domain = "llm"
    config_path = "heavenbase.llm"
    spec_type = LLMSpec
    binding_contracts = ("llm-route/v1",)

    def __init__(
        self,
        config: Mapping[str, Any] | None = None,
        manager: "ConfigManager | None | _DefaultAuthority" = _DEFAULT,
        *,
        resolver: ModuleResolver | None = None,
        context: "Context | None" = None,
    ) -> None:
        """Initialize this LLM configuration engine.

        Returns:
            None: This function does not return a value.
        """
        if context is not None and manager is not _DEFAULT:
            raise TypeError("LLMEngine context cannot be combined with resolver or manager")
        authority = resolve_runtime_authority(
            context=context,
            resolver=resolver,
            config=None if manager is _DEFAULT else manager,
            label="LLMEngine",
        )
        self.resolver = authority.resolver
        super().__init__(manager=authority.config, root=config)
        self._gateway_publication = -1
        self._gateway_profiles: set[str] = set()
        self._sync_gateways()
        self.register_profile("cache-key", self._apply_cache_key, cache_policy=False)
        self.register_profile("print", self._apply_print)

    def _register_gateway_profile(self, gateway: LLMGateway, *, replace: bool = True) -> None:
        """Publish one resolved gateway's internal Apply profile.

        Args:
            gateway (LLMGateway): Registered gateway descriptor.
            replace (bool): Whether an existing profile may be replaced.

        Returns:
            None: This function updates this engine's Apply registry.
        """

        identifier = gateway.identifier

        def validate(spec: LLMSpec, **kwargs: Any) -> None:
            if spec.gateway != identifier:
                raise ValueError(f"LLM spec is bound to gateway {spec.gateway!r}; re-resolve semantic data to use {identifier!r}")
            current = LLMGateway.load(identifier, resolver=self.resolver)
            expected = f"{spec.endpoint}/{identifier}/{current.compiler}"
            if spec.bindings.get("compiler") != expected:
                raise ValueError(f"LLM gateway {identifier!r} changed; re-resolve the LLM specification")

        def apply(spec: LLMSpec, **kwargs: Any) -> dict[str, Any]:
            current = LLMGateway.load(identifier, resolver=self.resolver)
            runtime = current.compile(spec, gateway=identifier, **kwargs)
            return {
                **runtime,
                "compiler": spec.bindings["compiler"],
                "arg_codec": spec.bindings["arg_codec"],
            }

        self.register_profile(
            gateway.identifier,
            apply,
            validate=validate,
            aliases=gateway.aliases,
            cache_policy=False,
            replace=replace,
        )

    def _sync_gateways(self) -> dict[str, LLMGateway]:
        """Reconcile Apply profiles with one current resolver publication."""
        while True:
            publication = self.resolver.catalog.publication
            gateways = LLMGateway.snapshot(resolver=self.resolver)
            if self.resolver.catalog.publication != publication:
                continue
            if publication == self._gateway_publication:
                return gateways
            active = set(gateways)
            with self._lock:
                if self.resolver.catalog.publication != publication:
                    continue
                for identifier in self._gateway_profiles - active:
                    if identifier in self._profiles:
                        self._profiles.delete(identifier)
                        self._apply_revision += 1
                for gateway in gateways.values():
                    self._register_gateway_profile(gateway, replace=True)
                self._gateway_profiles = active
                self._gateway_publication = publication
                self._resolved_cache.clear()
                self._applied_cache.clear()
                return gateways

    def _normalize_from(self, raw: Mapping[str, Any], source: Any) -> tuple[LLMSpec, Any]:
        """Bind every produced spec to this engine and module authority."""
        spec, context = super()._normalize_from(raw, source)
        return replace(spec, _engine=self, _resolver=self.resolver), context

    def _capture_catalog(self, root: Mapping[str, Any]) -> tuple[Any, Mapping[str, Any]]:
        gateways = self._sync_gateways()
        metadata = {
            name: {
                "aliases": list(gateway.aliases),
                "endpoints": list(gateway.endpoints),
                "compiler": gateway.compiler,
            }
            for name, gateway in gateways.items()
        }
        return (self.resolver.catalog.publication, self._apply_revision), {"gateways": metadata}

    def _normalize(self, data: Mapping[str, Any], *, context: Any) -> Mapping[str, Any]:
        root = context.root
        raw = to_mutable_data(data)
        preset = raw.get("preset")
        preset_cfg = root.get("presets", {}).get(preset, {}) if preset else {}
        provider_hint = self._provider(raw.get("provider"), root, required=False)
        model_token = raw.get("model") or root.get("default_model")
        if not model_token:
            raise ValueError("LLM model is required")
        model = self._model(model_token, provider_hint, root, allow_unknown=self._allows_unknown(provider_hint, root))
        model_cfg = root.get("models", {}).get(model, {})
        provider = self._provider(raw.get("provider") or model_cfg.get("default_provider") or root.get("default_provider"), root)
        provider_cfg = root.get("providers", {}).get(provider, {})
        model = self._model(model_token, provider, root, allow_unknown=_truthy(provider_cfg, "allow_unknown_models"))
        model_cfg = root.get("models", {}).get(model, {})
        provider_record = self._provider_model(model_cfg, provider)
        gateway = self._gateway(raw.get("gateway") or provider_cfg.get("gateway") or root.get("default_gateway"), context)
        gateway_cfg = root.get("gateways", {}).get(gateway, {})
        endpoint = self._endpoint(
            raw.get("endpoint")
            or provider_record.get("endpoint")
            or provider_cfg.get("endpoint")
            or gateway_cfg.get("endpoint")
            or root.get("default_endpoint"),
            root,
        )
        explicit = self._explicit_args(raw, context.explicit_paths)
        args = dmerge(
            [
                _args(root),
                _args(root.get("endpoints", {}).get(endpoint, {})),
                _args(gateway_cfg),
                _args(provider_cfg),
                _args(model_cfg),
                _args(provider_record),
                _args(preset_cfg),
                explicit,
            ]
        )
        if "thinking_budget_tokens" in args:
            raise ValueError("LLM thinking_budget_tokens was removed; use reasoning_max_tokens")
        return {
            "preset": preset,
            "endpoint": endpoint,
            "gateway": gateway,
            "provider": provider,
            "model": model,
            "args": args,
        }

    def _bind(self, data: Mapping[str, Any], *, context: Any) -> Mapping[str, Any]:
        root = context.root
        provider = str(data["provider"])
        gateway = str(data["gateway"])
        endpoint = str(data["endpoint"])
        model = str(data["model"])
        provider_cfg = root.get("providers", {}).get(provider, {})
        gateway_cfg = root.get("gateways", {}).get(gateway, {})
        endpoint_cfg = root.get("endpoints", {}).get(endpoint, {})
        model_cfg = root.get("models", {}).get(model, {})
        gateway_contract = context.catalog["gateways"][gateway]
        provider_record = self._provider_model(model_cfg, provider)
        model_id = provider_record.get("model_id") or provider_record.get("identifier") or model
        # ``context_window`` rides into the binding so optional loop compaction
        # can read the model's declared window from the resolved spec.
        model_binding = {key: to_mutable_data(model_cfg[key]) for key in ("capabilities", "thinking_control", "context_window") if key in model_cfg}
        return {
            "contract": "llm-route/v1",
            "compiler": f"{endpoint}/{gateway}/{gateway_contract['compiler']}",
            "arg_codec": provider_record.get("arg_codec") or gateway_cfg.get("arg_codec") or f"{endpoint}-semantic/v1",
            "model_id": model_id,
            "provider_backend": _provider_backend(provider_cfg, gateway, provider),
            "runtime_args": dmerge([gateway_cfg.get("runtime_args", {}), provider_cfg.get("runtime_args", {}), provider_record.get("runtime_args", {})]),
            "endpoint": to_mutable_data(endpoint_cfg),
            "gateway": to_mutable_data(gateway_cfg),
            "provider": to_mutable_data(provider_cfg),
            "model": model_binding,
            "capabilities": dmerge([model_cfg.get("capabilities", {}), provider_record.get("capabilities", {})]),
        }

    def _validate(self, spec: LLMSpec, *, context: Any) -> None:
        gateway = LLMGateway.load(spec.gateway, resolver=self.resolver)
        if spec.endpoint not in gateway.endpoints:
            raise ValueError(f"LLM endpoint {spec.endpoint!r} is incompatible with gateway {spec.gateway!r}")
        compiler = spec.bindings.get("compiler")
        expected_compiler = f"{spec.endpoint}/{spec.gateway}/{gateway.compiler}"
        if compiler != expected_compiler:
            raise ValueError(f"Unknown LLM compiler contract {compiler!r}")
        provider_cfg = spec._provider_cfg()
        if _truthy(provider_cfg, "require_base_url") and not (_cfg_value(spec.args, "base_url") or _cfg_value(provider_cfg, "base_url")):
            raise ValueError("LLM provider requires base_url; pass base_url=... or configure provider.base_url")
        placeholder = provider_cfg.get("placeholder_model")
        if _truthy(provider_cfg, "require_model_override") and placeholder and spec.model == placeholder and spec.model_id == placeholder:
            raise ValueError("LLM provider requires a concrete model; pass model=... when using this preset")
        if _truthy(provider_cfg, "require_api_key") and not (spec.args.get("api_key") or provider_cfg.get("api_key") or spec._gateway_cfg().get("api_key")):
            raise ValueError(f"LLM route {spec.provider}/{spec.gateway} requires an API key")
        if "think" in spec.args and not isinstance(spec.args["think"], bool):
            raise TypeError("LLM args.think must be bool")
        if "reasoning_max_tokens" in spec.args and (not isinstance(spec.args["reasoning_max_tokens"], int) or spec.args["reasoning_max_tokens"] < 0):
            raise TypeError("LLM args.reasoning_max_tokens must be a non-negative integer")
        capabilities = spec.bindings.get("capabilities", {})
        if spec.args.get("reasoning_effort") is not None and capabilities.get("reasoning_efforts") is False:
            raise ValueError(f"LLM model {spec.model!r} does not support reasoning_effort")

    def _default_profile(self, spec: LLMSpec) -> str:
        return spec.gateway

    def _apply_cache_key(self, spec: LLMSpec) -> str:
        gateway = LLMGateway.load(spec.gateway, resolver=self.resolver)
        expected = f"{spec.endpoint}/{spec.gateway}/{gateway.compiler}"
        if spec.bindings.get("compiler") != expected:
            raise ValueError(f"LLM gateway {spec.gateway!r} changed; re-resolve the LLM specification")
        rows = self.resolver.inspect("llm_gateway", spec.gateway)
        if not rows:
            raise KeyError(f"Unknown LLM gateway {spec.gateway!r}")
        row = rows[0]
        installation = row["record"]["meta"]["installation"]
        runtime = gateway.compile(spec, gateway=spec.gateway)
        return sha256hash(
            {
                "data": spec.data,
                "bindings": spec.bindings,
                "runtime": runtime,
                "gateway_record": {
                    "fingerprint": row["fingerprint"],
                    "coordinate": installation["coordinate"],
                    "generation": installation["generation"],
                },
            }
        )

    def _apply_print(self, spec: LLMSpec) -> dict[str, Any]:
        out = {
            "domain": spec.domain,
            "data": to_mutable_data(spec.data),
            "bindings": to_mutable_data(spec.bindings),
        }
        out = redact_paths(
            out,
            (
                "data.args.api_key",
                "data.args.auth_token",
                "data.args.headers.*",
                "data.args.extra_headers.*",
                "bindings.provider.api_key",
                "bindings.provider.auth_token",
                "bindings.provider.headers.*",
                "bindings.gateway.api_key",
                "bindings.gateway.auth_token",
                "bindings.gateway.headers.*",
            ),
        )
        for owner in (out["data"].get("args", {}), out["bindings"].get("provider", {}), out["bindings"].get("gateway", {})):
            for key in ("base_url", "url"):
                if isinstance(owner.get(key), str):
                    owner[key] = redact_url(
                        owner[key],
                        query_names=("access_token", "api_key", "auth", "key", "password", "token"),
                    )
        return out

    @staticmethod
    def _provider(value: Any, root: Mapping[str, Any], *, required: bool = True) -> str | None:
        if value is None:
            if required:
                raise ValueError("LLM provider is required")
            return None
        providers = root.get("providers", {})
        token = str(value)
        aliases: dict[str, str] = {}
        for canonical, config in providers.items():
            for alias in (canonical, *config.get("aliases", ())):
                clean = _clean_alias(alias)
                found = aliases.get(clean)
                if found is not None and found != canonical:
                    raise ValueError(f"LLM provider alias collision {alias!r}: {found!r} and {canonical!r}")
                aliases[clean] = canonical
        provider = aliases.get(_clean_alias(token))
        if provider is None:
            raise ValueError(f"Unknown LLM provider {token!r}; expected one of: {', '.join(sorted(providers))}")
        return provider

    @staticmethod
    def _allows_unknown(provider: str | None, root: Mapping[str, Any]) -> bool:
        return bool(provider and root.get("providers", {}).get(provider, {}).get("allow_unknown_models"))

    @staticmethod
    def _model(value: Any, provider: str | None, root: Mapping[str, Any], *, allow_unknown: bool) -> str:
        token = str(value)
        models = root.get("models", {})
        aliases: dict[str, str] = {}
        for canonical, config in models.items():
            names = [canonical, *config.get("aliases", ())]
            if provider:
                names.extend(config.get("provider_aliases", {}).get(provider, ()))
            for alias in names:
                clean = _clean_alias(alias)
                found = aliases.get(clean)
                if found is not None and found != canonical:
                    raise ValueError(f"LLM model alias collision {alias!r}: {found!r} and {canonical!r}")
                aliases[clean] = canonical
        for canonical, config in models.items():
            for record in config.get("providers", {}).values():
                identifier = record.get("model_id")
                if identifier:
                    aliases.setdefault(_clean_alias(identifier), canonical)
        model = aliases.get(_clean_alias(token))
        if model is not None:
            return model
        if allow_unknown:
            return token
        raise ValueError(f"Unknown LLM model {token!r}; expected one of: {', '.join(sorted(models))}")

    @staticmethod
    def _provider_model(model_cfg: Mapping[str, Any], provider: str) -> dict[str, Any]:
        providers = model_cfg.get("providers", {})
        if providers:
            if provider not in providers:
                raise ValueError(f"LLM model/provider pair is unsupported: provider={provider!r}")
            return to_mutable_data(providers[provider])
        return {"model_id": None}

    @staticmethod
    def _gateway(value: Any, context: Any) -> str:
        if value is None:
            raise ValueError("LLM gateway is required")
        token = str(value)
        metadata = context.catalog.get("gateways", {})
        aliases = {name: name for name in metadata}
        for name, config in metadata.items():
            for alias in config.get("aliases", ()):
                if alias in aliases and aliases[alias] != name:
                    raise ValueError(f"LLM gateway alias collision {alias!r}")
                aliases[alias] = name
        gateway = aliases.get(token)
        if gateway is None:
            raise ValueError(f"Unknown LLM gateway {token!r}; expected one of: {', '.join(sorted(metadata))}")
        return gateway

    @staticmethod
    def _endpoint(value: Any, root: Mapping[str, Any]) -> str:
        if value is None:
            raise ValueError("LLM endpoint is required")
        endpoints = root.get("endpoints", {})
        token = str(value)
        aliases: dict[str, str] = {}
        for canonical, config in endpoints.items():
            for alias in (canonical, *config.get("aliases", ())):
                if alias in aliases and aliases[alias] != canonical:
                    raise ValueError(f"LLM endpoint alias collision {alias!r}")
                aliases[alias] = canonical
        endpoint = aliases.get(token)
        if endpoint is None:
            raise ValueError(f"Unknown LLM endpoint {token!r}; expected one of: {', '.join(sorted(endpoints))}")
        return endpoint

    @staticmethod
    def _explicit_args(data: Mapping[str, Any], explicit_paths: frozenset[tuple[str, ...]]) -> dict[str, Any]:
        direct = {path[0] for path in explicit_paths if path}
        args = {}
        if "default_args" in direct and isinstance(data.get("default_args"), Mapping):
            args = dmerge([args, to_mutable_data(data["default_args"])])
        if "args" in direct and isinstance(data.get("args"), Mapping):
            args = dmerge([args, to_mutable_data(data["args"])])
        args.update({key: to_mutable_data(value) for key, value in data.items() if key in direct and key not in _ROUTE_KEYS})
        return args


def _args(config: Any) -> dict[str, Any]:
    if not isinstance(config, Mapping):
        return {}
    return dmerge([to_mutable_data(config.get("default_args", {}) or {}), to_mutable_data(config.get("args", {}) or {})])


def _default_engine() -> LLMEngine:
    """Construct an engine from the current default Context lazily."""
    return LLMEngine()


class _DefaultEngineProxy:
    """Compatibility convenience that never becomes a second authority."""

    @staticmethod
    def _for(config: Any = None) -> LLMEngine:
        if isinstance(config, LLMSpec) and config._engine is not None:
            return config._engine
        return _default_engine()

    def resolve(self, config: Any = None) -> LLMSpec:
        return self._for(config).resolve(config)

    def apply(self, config: Any = None, *, profile: str | None = None, **kwargs: Any) -> Any:
        return self._for(config).apply(config, profile=profile, **kwargs)

    def root(self) -> Mapping[str, Any]:
        return _default_engine().root()


LLM_ENGINE = _DefaultEngineProxy()
