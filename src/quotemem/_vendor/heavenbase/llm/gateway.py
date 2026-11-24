"""Context-owned LLM gateway descriptor facade."""

from __future__ import annotations

__all__ = ["LLMGateway"]

from collections.abc import Callable, Iterable, Mapping
from typing import TYPE_CHECKING, Any
import weakref

from heavenbase.modules import ModuleResolver, ModuleService, module_resolver, python_executable_ref
from heavenbase.modules.base import ResolverCache
from heavenbase.utils import dmerge, raise_mismatch
from heavenbase.utils.typing import deepcopy

if TYPE_CHECKING:
    from .config import LLMSpec

_Compiler = Callable[["LLMSpec", str, Mapping[str, Any]], dict[str, Any]]
_GatewayCall = Callable[..., Any]
_CALLABLE_KEYS = ("compile", "available", "chat", "achat", "embed", "aembed", "imagen", "aimagen", "client")
_GATEWAY_CACHE = ResolverCache["LLMGateway"]()


def _cfg_value(config: Mapping[str, Any], key: str, default: Any = None, gateway: str | None = None) -> Any:
    value = config.get(key)
    if isinstance(value, Mapping):
        value = value.get(gateway) if gateway and gateway in value else value.get("default")
    return value if value not in (None, "") else default


def _secret(config: Mapping[str, Any], key: str = "api_key") -> str | None:
    value = config.get(key)
    return str(value) if value else None


def _provider_backend(config: Mapping[str, Any], gateway: str, default: str) -> str:
    backend = config.get("backend", default)
    if isinstance(backend, Mapping):
        return str(backend.get(gateway) or backend.get("default") or default)
    return str(backend or default)


def _truthy(config: Mapping[str, Any], key: str) -> bool:
    return bool(config.get(key))


def _thinking(model: Mapping[str, Any], provider: Mapping[str, Any], gateway_cfg: Mapping[str, Any], gateway: str) -> Any:
    return (
        _cfg_value(model, "thinking_control", gateway=gateway)
        or _cfg_value(provider, "thinking_control", gateway=gateway)
        or _cfg_value(gateway_cfg, "thinking_control", gateway=gateway)
    )


def _prefixed(backend: str, model: str) -> str:
    return model if model.startswith(f"{backend}/") else f"{backend}/{model}"


def _anthropic_url(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    clean = value.rstrip("/")
    return clean[: -len("/v1")] if clean.endswith("/v1") else clean


def _parts(spec: "LLMSpec", gateway: str, override: Mapping[str, Any]) -> dict[str, Any]:
    model_cfg = spec._model_cfg()
    provider_cfg = spec._provider_cfg()
    gateway_cfg = spec._gateway_cfg(gateway)
    args = dmerge([spec.bindings.get("runtime_args", {}), spec.args, dict(override)])
    out = {
        "model_cfg": model_cfg,
        "provider_cfg": provider_cfg,
        "gateway_cfg": gateway_cfg,
        "args": args,
        "arg_api_key": _secret(args),
        "arg_auth_token": _secret(args, "auth_token"),
        "arg_base_url": _cfg_value(args, "base_url"),
        "provider_key": _secret(provider_cfg),
        "provider_token": _secret(provider_cfg, "auth_token"),
        "gateway_key": _secret(gateway_cfg),
        "gateway_token": _secret(gateway_cfg, "auth_token"),
        "backend": _provider_backend(provider_cfg, gateway, spec.provider),
        "headers": deepcopy(gateway_cfg.get("headers", {}) or {}),
        "thinking_control": _thinking(model_cfg, provider_cfg, gateway_cfg, gateway),
    }
    for key in ("api_key", "auth_token", "base_url"):
        args.pop(key, None)
    return out


def _service(resolver: ModuleResolver) -> ModuleService:
    if not isinstance(resolver, ModuleService):
        raise TypeError("LLM gateway registration requires a mutable ModuleService resolver")
    return resolver


def _row(identifier: str, resolver: ModuleResolver) -> Mapping[str, Any]:
    rows = resolver.inspect("llm_gateway", str(identifier).strip())
    if not rows:
        retired = resolver.inspect("llm_gateway", str(identifier).strip(), include_inactive=True)
        if retired:
            raise KeyError(f"LLM gateway {identifier!r} is retired")
        choices = tuple(str(row["identifier"]) for row in resolver.inspect("llm_gateway"))
        return raise_mismatch(choices, identifier, name="LLM gateway")
    return rows[0]


def _restore_callable(
    definition: Mapping[str, Any],
    key: str,
    resolver: ModuleResolver,
    owner: Mapping[str, Any],
    *,
    required: bool = False,
) -> _GatewayCall | None:
    target = definition.get(key)
    if target is None and not required:
        return None
    if not isinstance(target, Mapping):
        raise TypeError(f"LLM gateway definition {key!r} must be an executable reference")
    value = resolver.resolve_executable(target, owner=owner)
    if not callable(value):
        raise TypeError(f"LLM gateway definition {key!r} target must restore a callable")
    return value


def _from_row(row: Mapping[str, Any], resolver: ModuleResolver) -> "LLMGateway":
    def build(current_row: Mapping[str, Any]) -> LLMGateway:
        identifier = str(current_row["identifier"])
        definition = resolver.resolve("llm_gateway", identifier)
        if not isinstance(definition, Mapping):
            raise TypeError(f"LLM gateway {identifier!r} definition must resolve to a mapping")
        endpoints = definition.get("endpoints", ("openai",))
        if not isinstance(endpoints, Iterable) or isinstance(endpoints, (str, bytes)):
            raise TypeError(f"LLM gateway {identifier!r} endpoints must be a sequence")
        return LLMGateway(
            identifier,
            _restore_callable(definition, "compile", resolver, current_row, required=True),
            aliases=tuple(str(alias) for alias in current_row.get("aliases", ())),
            endpoints=tuple(str(endpoint) for endpoint in endpoints),
            compiler=str(definition.get("compiler") or "semantic/v1"),
            available=_restore_callable(definition, "available", resolver, current_row),
            chat=_restore_callable(definition, "chat", resolver, current_row),
            achat=_restore_callable(definition, "achat", resolver, current_row),
            embed=_restore_callable(definition, "embed", resolver, current_row),
            aembed=_restore_callable(definition, "aembed", resolver, current_row),
            imagen=_restore_callable(definition, "imagen", resolver, current_row),
            aimagen=_restore_callable(definition, "aimagen", resolver, current_row),
            client=_restore_callable(definition, "client", resolver, current_row),
            resolver=resolver,
        )

    return _GATEWAY_CACHE.get(resolver, row, build)


class LLMGateway:
    """Describe one compiler and transport set installed as `llm_gateway`.

    Gateway objects are typed views over Context-owned module records. Bundled,
    programmatic, and externally installed gateways all travel through the same
    descriptor loader and resolver-local materialization cache.
    """

    __slots__ = (
        "identifier",
        "fn",
        "aliases",
        "endpoints",
        "compiler",
        "_available",
        "_chat",
        "_achat",
        "_embed",
        "_aembed",
        "_imagen",
        "_aimagen",
        "_client",
        "_resolver_ref",
        "_sealed",
        "__weakref__",
    )

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("LLMGateway descriptors are immutable; publish a replacement through ModuleService")
        object.__setattr__(self, name, value)

    def __init__(
        self,
        identifier: str,
        fn: _Compiler,
        aliases: tuple[str, ...] = (),
        endpoints: tuple[str, ...] = ("openai",),
        compiler: str = "semantic/v1",
        *,
        available: _GatewayCall | None = None,
        chat: _GatewayCall | None = None,
        achat: _GatewayCall | None = None,
        embed: _GatewayCall | None = None,
        aembed: _GatewayCall | None = None,
        imagen: _GatewayCall | None = None,
        aimagen: _GatewayCall | None = None,
        client: _GatewayCall | None = None,
        resolver: ModuleResolver | None = None,
    ) -> None:
        """Create one immutable gateway descriptor view.

        Args:
            identifier (str): Canonical gateway identifier.
            fn (_Compiler): Callable that compiles runtime client arguments.
            aliases (tuple[str, ...]): Optional lookup aliases. Defaults to ``()``.
            endpoints (tuple[str, ...]): Supported endpoint grammars. Defaults to ``("openai",)``.
            compiler (str): Compiler contract version label. Defaults to ``"semantic/v1"``.
            available (_GatewayCall | None): Optional dependency probe callable.
            chat (_GatewayCall | None): Optional sync chat transport callable.
            achat (_GatewayCall | None): Optional async chat transport callable.
            embed (_GatewayCall | None): Optional sync embedding transport callable.
            aembed (_GatewayCall | None): Optional async embedding transport callable.
            imagen (_GatewayCall | None): Optional sync image transport callable.
            aimagen (_GatewayCall | None): Optional async image transport callable.
            client (_GatewayCall | None): Optional SDK client factory callable.
            resolver (ModuleResolver | None): Resolver that materialized this view.

        Returns:
            None: This function does not return a value.
        """
        if not isinstance(identifier, str) or not identifier.strip():
            raise ValueError("LLM gateway identifier is required")
        if not callable(fn):
            raise TypeError(f"LLM gateway {identifier!r} compiler must be callable")
        if not endpoints:
            raise ValueError(f"LLM gateway {identifier!r} must support at least one endpoint")
        calls = {"available": available, "chat": chat, "achat": achat, "embed": embed, "aembed": aembed, "imagen": imagen, "aimagen": aimagen, "client": client}
        invalid = [name for name, value in calls.items() if value is not None and not callable(value)]
        if invalid:
            raise TypeError(f"LLM gateway {identifier!r} transports must be callable: {', '.join(invalid)}")
        self.identifier = identifier.strip()
        self.fn = fn
        self.aliases = tuple(dict.fromkeys(str(alias).strip() for alias in aliases if str(alias).strip()))
        self.endpoints = tuple(dict.fromkeys(str(endpoint).strip() for endpoint in endpoints if str(endpoint).strip()))
        self.compiler = str(compiler)
        self._available = available
        self._chat = chat
        self._achat = achat
        self._embed = embed
        self._aembed = aembed
        self._imagen = imagen
        self._aimagen = aimagen
        self._client = client
        try:
            self._resolver_ref = None if resolver is None else weakref.ref(resolver)
        except TypeError as error:
            raise TypeError("LLMGateway resolvers must support weak references") from error
        self._sealed = True

    @property
    def resolver(self) -> ModuleResolver | None:
        """Return the live resolver that materialized this view, if any."""
        return None if self._resolver_ref is None else self._resolver_ref()

    def register(self, *, resolver: ModuleResolver | None = None, replace: bool = False) -> "LLMGateway":
        """Install this importable descriptor in one Context-owned catalog.

        Args:
            resolver (ModuleResolver | None): Catalog authority. Defaults to this
                view's resolver or the active Context modules service.
            replace (bool): Whether an existing programmatic registration may be
                replaced. Defaults to ``False``.

        Returns:
            LLMGateway: Freshly loaded gateway view after registration.
        """
        selected = module_resolver(resolver if resolver is not None else self.resolver)
        service = _service(selected)
        found = service.inspect("llm_gateway", self.identifier)
        coordinate = f"programmatic/llm_gateway/{self.identifier}"
        if found:
            existing_coordinate = found[0]["record"]["meta"]["installation"]["coordinate"]
            if not replace:
                raise ValueError(f"LLM gateway {self.identifier!r} is already registered")
            if existing_coordinate != coordinate:
                raise ValueError(f"LLM gateway {self.identifier!r} belongs to {existing_coordinate!r} and cannot be replaced programmatically")
        definition: dict[str, Any] = {
            "compiler": self.compiler,
            "endpoints": list(self.endpoints),
            "compile": python_executable_ref(self.fn, label=f"LLM gateway {self.identifier!r} compiler"),
        }
        for key, value in {
            "available": self._available,
            "chat": self._chat,
            "achat": self._achat,
            "embed": self._embed,
            "aembed": self._aembed,
            "imagen": self._imagen,
            "aimagen": self._aimagen,
            "client": self._client,
        }.items():
            if value is not None:
                definition[key] = python_executable_ref(value, label=f"LLM gateway {self.identifier!r} {key}")
        service.register(
            "llm_gateway",
            self.identifier,
            {
                "name": f"{self.identifier} LLM Gateway",
                "desc": (getattr(self.fn, "__doc__", None) or "Registered LLM gateway.").strip().splitlines()[0],
                "source": "inline",
                "target": "definition",
                "active": True,
                "meta": {
                    "schema_version": 1,
                    "tags": {"llm": True},
                    "definition": definition,
                },
            },
            aliases=self.aliases,
            coordinate=coordinate,
        )
        _GATEWAY_CACHE.clear(selected)
        return self.load(self.identifier, resolver=selected)

    @classmethod
    def load(cls, identifier: str, *, resolver: ModuleResolver | None = None) -> "LLMGateway":
        """Load one gateway by canonical identifier or alias.

        Args:
            identifier (str): Gateway identifier or alias.
            resolver (ModuleResolver | None): Catalog authority. Defaults to the
                active Context modules service.

        Returns:
            LLMGateway: Materialized gateway descriptor view.
        """
        selected = module_resolver(resolver)
        return _from_row(_row(identifier, selected), selected)

    @classmethod
    def snapshot(cls, *, resolver: ModuleResolver | None = None) -> dict[str, "LLMGateway"]:
        """Return all active gateways from one coherent Context catalog.

        Args:
            resolver (ModuleResolver | None): Catalog authority. Defaults to the
                active Context modules service.

        Returns:
            dict[str, LLMGateway]: Active gateways keyed by canonical identifier.
        """
        selected = module_resolver(resolver)
        return {str(row["identifier"]): _from_row(row, selected) for row in selected.inspect("llm_gateway")}

    def compile(self, spec: "LLMSpec", *, gateway: str | None = None, **override: Any) -> dict[str, Any]:
        """Compile runtime client arguments for one resolved spec.

        Args:
            spec (LLMSpec): Resolved LLM specification.
            gateway (str | None): Gateway override for compilation. Defaults to
                this descriptor's identifier.
            **override (Any): Per-request runtime argument overrides.

        Returns:
            dict[str, Any]: Gateway-specific client arguments for ``spec``.
        """
        return self.fn(spec, gateway or self.identifier, override)

    def available(self, spec: "LLMSpec") -> bool:
        """Return whether this gateway's transport dependencies are ready.

        Args:
            spec (LLMSpec): Resolved LLM specification to probe.

        Returns:
            bool: ``True`` when no probe is registered or the probe succeeds.
        """
        return True if self._available is None else bool(self._available(spec))

    def chat(self, spec: "LLMSpec", messages: list[dict[str, Any]], *, stream: bool = False, **kwargs: Any) -> Any:
        """Run chat completion through this gateway's registered transport.

        Args:
            spec (LLMSpec): Resolved LLM specification bound to this gateway.
            messages (list[dict[str, Any]]): OpenAI-format chat messages.
            stream (bool): Whether to request a streaming provider response.
                Defaults to ``False``.
            **kwargs (Any): Provider request options forwarded to the transport.

        Returns:
            Any: Provider chat response object or stream iterator.

        Raises:
            NotImplementedError: If this gateway does not register chat transport.
        """
        return self._call("chat", self._chat, spec, messages, stream=stream, **kwargs)

    async def achat(self, spec: "LLMSpec", messages: list[dict[str, Any]], *, stream: bool = False, **kwargs: Any) -> Any:
        """Run async chat completion through this gateway's registered transport.

        Args:
            spec (LLMSpec): Resolved LLM specification bound to this gateway.
            messages (list[dict[str, Any]]): OpenAI-format chat messages.
            stream (bool): Whether to request a streaming provider response.
                Defaults to ``False``.
            **kwargs (Any): Provider request options forwarded to the transport.

        Returns:
            Any: Provider chat response object or async stream iterator.

        Raises:
            NotImplementedError: If this gateway does not register async chat transport.
        """
        return await self._acall("chat", self._achat, spec, messages, stream=stream, **kwargs)

    def embed(self, spec: "LLMSpec", inputs: list[str], **kwargs: Any) -> Any:
        """Run embedding creation through this gateway's registered transport.

        Args:
            spec (LLMSpec): Resolved LLM specification bound to this gateway.
            inputs (list[str]): Embedding input strings.
            **kwargs (Any): Provider request options forwarded to the transport.

        Returns:
            Any: Provider embedding response object.

        Raises:
            NotImplementedError: If this gateway does not register embedding transport.
        """
        return self._call("embeddings", self._embed, spec, inputs, **kwargs)

    async def aembed(self, spec: "LLMSpec", inputs: list[str], **kwargs: Any) -> Any:
        """Run async embedding creation through this gateway's registered transport.

        Args:
            spec (LLMSpec): Resolved LLM specification bound to this gateway.
            inputs (list[str]): Embedding input strings.
            **kwargs (Any): Provider request options forwarded to the transport.

        Returns:
            Any: Provider embedding response object.

        Raises:
            NotImplementedError: If this gateway does not register async embedding transport.
        """
        return await self._acall("embeddings", self._aembed, spec, inputs, **kwargs)

    def imagen(self, spec: "LLMSpec", prompt: str, *, images: Iterable[Any] = (), **kwargs: Any) -> Any:
        """Run image generation through this gateway's registered transport.

        Args:
            spec (LLMSpec): Resolved LLM specification bound to this gateway.
            prompt (str): Text prompt for image generation.
            images (Iterable[Any]): Optional reference image inputs. Defaults to ``()``.
            **kwargs (Any): Provider request options forwarded to the transport.

        Returns:
            Any: Provider image-generation response object.

        Raises:
            NotImplementedError: If this gateway does not register image transport.
        """
        return self._call("image generation", self._imagen, spec, prompt, images=images, **kwargs)

    async def aimagen(self, spec: "LLMSpec", prompt: str, *, images: Iterable[Any] = (), **kwargs: Any) -> Any:
        """Run async image generation through this gateway's registered transport.

        Args:
            spec (LLMSpec): Resolved LLM specification bound to this gateway.
            prompt (str): Text prompt for image generation.
            images (Iterable[Any]): Optional reference image inputs. Defaults to ``()``.
            **kwargs (Any): Provider request options forwarded to the transport.

        Returns:
            Any: Provider image-generation response object.

        Raises:
            NotImplementedError: If this gateway does not register async image transport.
        """
        return await self._acall("image generation", self._aimagen, spec, prompt, images=images, **kwargs)

    def client(self, spec: "LLMSpec", protocol: str, *, asynchronous: bool = False) -> tuple[Any, dict[str, Any]]:
        """Return a provider SDK client for one resolved spec.

        The built-in ``openai``, ``portkey``, ``bifrost`` and ``anthropic``
        gateways cache their SDK clients. A sync client is shared. An async client
        requested inside a running event loop is the client cached for that loop,
        closed when the loop shuts down its async generators (as ``asyncio.run``
        does); requested outside a running loop it is a new client the caller owns
        and closes.

        Args:
            spec (LLMSpec): Resolved LLM specification bound to this gateway.
            protocol (str): Client protocol such as ``openai`` or ``anthropic``.
            asynchronous (bool): Whether to return an async SDK client. Defaults
                to ``False``.

        Returns:
            tuple[Any, dict[str, Any]]: SDK client and compiled runtime arguments.

        Raises:
            ValueError: If this gateway cannot expose the requested protocol.
        """
        if self._client is None:
            raise ValueError(f"Gateway {self.identifier!r} cannot expose a {protocol!r} client")
        return self._client(spec, protocol, asynchronous=asynchronous)

    def _call(self, capability: str, fn: _GatewayCall | None, *args: Any, **kwargs: Any) -> Any:
        if fn is None:
            raise NotImplementedError(f"LLM gateway {self.identifier!r} does not support {capability}")
        return fn(*args, **kwargs)

    async def _acall(self, capability: str, fn: _GatewayCall | None, *args: Any, **kwargs: Any) -> Any:
        if fn is None:
            raise NotImplementedError(f"LLM gateway {self.identifier!r} does not support async {capability}")
        value = fn(*args, **kwargs)
        return await value if hasattr(value, "__await__") else value
