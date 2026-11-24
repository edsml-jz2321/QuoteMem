"""Shared provider-protocol mechanics for LLM gateway module folders."""

from __future__ import annotations

__all__: list[str] = []

import asyncio
from collections import OrderedDict
from collections.abc import AsyncIterator, Mapping
from contextlib import contextmanager, nullcontext
import inspect
import ipaddress
import os
import re
import sys
import threading
import weakref
from functools import partial
from gc import get_referents
from types import CoroutineType, GeneratorType
from urllib.parse import urlsplit

from ..image import LLMImage
from ..config import LLMSpec
from heavenbase.utils import NetworkProxy, dumps_json, loads_json, sha256hash
from heavenbase.utils.typing import Any, Dict, Iterable, List

CLIENT_ARG_KEYS = {
    "batch_max_workers",
    "batch_size",
    "batch_workers",
    "cache",
    "cache_config",
    "cache_policy",
    "dim",
    "embed_max_workers",
    "embedding_batch_size",
    "embedding_max_workers",
    "api_key",
    "auth_token",
    "base_url",
    "enforce_non_stream_structured",
    "http_proxy",
    "https_proxy",
    "max_retries",
    # Loop-level keys consumed in ``_chat_context``, never provider parameters.
    # The gateway runtime re-merges spec args, so without these entries a
    # constructor-channel value reaches ``Completions.create`` and raises on
    # every openai-protocol route.
    "compact_ratio",
    "max_tool_turns",
    "tool_result_capture_chars",
    "max_workers",
    "mock",
    "mock_chat",
    "mock_embed",
    "mock_imagen",
    "mock_pause",
    "mock_seed",
    "mock_text",
    "name",
    "no_proxy",
    "reasoning_exclude",
    "reasoning_max_tokens",
    "repair_tool_calls",
    "stream_chat",
    "structured_stream",
    "think",
    "timeout",
    "tool_call_repair",
}

EMBEDDING_PAYLOAD_KEYS = {
    "dimensions",
    "encoding_format",
    "extra_body",
    "extra_headers",
    "extra_query",
    "input",
    "user",
}

_CLIENT_CACHE_MAX = 64
_ClientKey = tuple[str | int, str]
_OPENAI_CLIENTS: "OrderedDict[_ClientKey, Any]" = OrderedDict()
_ANTHROPIC_CLIENTS: "OrderedDict[_ClientKey, Any]" = OrderedDict()
_LITELLM_MODULE: Any = None
_CLIENT_CACHE_LOCK = threading.RLock()
_CLIENT_BUILDS: Dict[tuple[int, _ClientKey], list[Any]] = {}
_END = object()
_LOOP_ATTRIBUTE = "_heavenbase_llm_clients"
# The openai and anthropic SDKs' default connection limit.
_SDK_MAX_CONNECTIONS = 1000


class _LoopClients:
    """The async SDK clients built inside one event loop.

    An async SDK client binds its connection pool and locks to the loop it first
    runs on, so every loop has its own clients in its own LRU caches. The record
    is stored on the loop and the module keeps only a weak reference to it, so the
    record and its clients live as long as the loop: a loop dropped without being
    closed releases them when the loop object is garbage-collected.

    The loop's shutdown method closes cached clients after it has finalized its
    async generators, including finalizers scheduled through a task factory.
    A client first requested in a shutdown finalizer or on a loop marked as
    closing keeps no idle connection. A loop closed without shutdown
    has its clients dropped when another loop requests its first client or when
    the loop is garbage-collected.
    """

    __slots__ = ("__weakref__", "caches", "closing", "loop", "pid", "_create_task", "_finalizers", "_keeper", "_shutdown_asyncgens", "_shutdown_calls")

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop
        self.pid = os.getpid()
        self.caches: Dict[str, "OrderedDict[_ClientKey, Any]"] = {"openai": OrderedDict(), "anthropic": OrderedDict()}
        self.closing = False
        self._shutdown_calls = 0
        self._finalizers: set[asyncio.Future[Any]] = set()
        self._keeper: Any = None
        create_task = loop.create_task
        previous = getattr(create_task, "__self__", None)
        self._create_task = previous._create_task if isinstance(previous, _LoopClients) else create_task
        shutdown = loop.shutdown_asyncgens
        previous = getattr(shutdown, "__self__", None)
        self._shutdown_asyncgens = previous._shutdown_asyncgens if isinstance(previous, _LoopClients) else shutdown

    def create_task(self, coro: Any, *args: Any, **kwargs: Any) -> Any:
        """Track generator cleanup before a task factory can wrap the coroutine."""
        task = self._create_task(coro, *args, **kwargs)
        if self.pid == os.getpid() and type(coro).__name__ == "async_generator_athrow":
            self._finalizers.add(task)
            task.add_done_callback(self._finalizers.discard)
        return task

    async def shutdown_asyncgens(self) -> None:
        """Drain generator finalizers before closing the clients they can still use."""
        if self.pid != os.getpid() or _running_loop() is not self.loop:
            await self._shutdown_asyncgens()
            return
        current = asyncio.current_task()
        if self._shutdown_calls and current in self._finalizers:
            # An outer shutdown is already waiting for this finalizer.
            return
        self._shutdown_calls += 1
        try:
            await self._shutdown_asyncgens()
            await self._drain_finalizers()
        finally:
            self._shutdown_calls -= 1
        if self._shutdown_calls == 0:
            await self._close_clients()

    async def _drain_finalizers(self) -> None:
        # A collected generator has already left the loop's shutdown set.
        current = asyncio.current_task()
        while pending := [task for task in self._finalizers if task is not current and not task.done()]:
            await asyncio.wait(pending)

    async def _close_clients(self) -> None:
        with _CLIENT_CACHE_LOCK:
            self.closing = True
            clients = self.take_locked()
        for client in clients:
            await _aclose_client(client)

    def start_keeper(self) -> None:
        """Register cleanup with the loop's original shutdown mechanism."""

        async def keeper() -> AsyncIterator[None]:
            try:
                yield
            finally:
                if self.pid == os.getpid() and _running_loop() is self.loop and not self._shutdown_calls:
                    self._shutdown_calls += 1
                    try:
                        # Let shutdown schedule the rest of its finalizers,
                        # including under an eager task factory.
                        await asyncio.sleep(0)
                        await self._drain_finalizers()
                    finally:
                        self._shutdown_calls -= 1
                    if self._shutdown_calls == 0:
                        await self._close_clients()

        # Keep the first-iteration hook for registration. The keeper is held
        # for the loop's lifetime, so GC need not schedule cleanup on it.
        generator = keeper()
        hooks = sys.get_asyncgen_hooks()
        sys.set_asyncgen_hooks(firstiter=hooks.firstiter, finalizer=None)
        try:
            generator.asend(None).send(None)
        except StopIteration:
            pass
        finally:
            sys.set_asyncgen_hooks(firstiter=hooks.firstiter, finalizer=hooks.finalizer)
        self._keeper = generator

    def take_locked(self) -> list[Any]:
        """Remove and return every cached client of this loop."""
        clients = []
        for cache in self.caches.values():
            clients.extend(cache.values())
            cache.clear()
        return clients


_LOOP_RECORDS: "weakref.WeakSet[_LoopClients]" = weakref.WeakSet()
_CLOSING: set[Any] = set()


def _client_key(spec: LLMSpec, digest: str, base_url: Any) -> _ClientKey:
    """Return one Context-resolver-owned SDK client key.

    A client built for a proxy to a loopback ``base_url`` is kept apart from the client that reaches it directly and
    from those built for other proxies: an SDK client keeps the proxy it was built with.
    """
    proxies = _loopback_proxies(spec, base_url)
    return spec.authority_key[0], sha256hash({"client": digest, "proxies": proxies}) if proxies else digest


def _running_loop() -> asyncio.AbstractEventLoop | None:
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return None


def _shutting_down(loop: asyncio.AbstractEventLoop) -> bool:
    """Detect loop shutdown even when a task factory wraps the shutdown coroutine."""
    called = getattr(loop, "_asyncgens_shutdown_called", None)
    if isinstance(called, bool):
        return called
    # uvloop keeps its flag in Cython state. Look for its actual shutdown call
    # among the loop's tasks because aclose() also runs during normal operation.
    shutdown_code = getattr(loop.shutdown_asyncgens, "__code__", None)
    if shutdown_code is None:
        return False
    current = asyncio.current_task(loop)
    pending = [task.get_coro() for task in asyncio.all_tasks(loop) if task is not current]
    while pending:
        coroutine = pending.pop()
        code = getattr(coroutine, "cr_code", None) or getattr(coroutine, "gi_code", None)
        if code is shutdown_code:
            return True
        if isinstance(coroutine, CoroutineType):
            if not coroutine.cr_running:
                pending.append(coroutine.cr_await)
            elif (frame := coroutine.cr_frame) is not None:
                # An eager finalizer can run before its parent's wrapping
                # coroutine has suspended at the await of uvloop shutdown.
                pending.extend(
                    value for value in frame.f_locals.values() if type(value).__name__ == "iterable_coroutine" and getattr(value, "cr_running", False)
                )
        elif isinstance(coroutine, GeneratorType) and not coroutine.gi_running:
            pending.append(coroutine.gi_yieldfrom)
        elif type(coroutine).__name__ == "coroutine_wrapper":
            # The Cython await iterator exposes its coroutine only through GC.
            pending.extend(get_referents(coroutine))
    return False


def _loop_clients() -> _LoopClients | None:
    """Return the client record of the running event loop.

    ``None`` outside a running loop, or for a loop object that cannot hold the record.
    """
    loop = _running_loop()
    if loop is None:
        return None
    owner = getattr(loop, _LOOP_ATTRIBUTE, None)
    if isinstance(owner, _LoopClients) and owner.pid == os.getpid():
        return owner
    owner = _LoopClients(loop)
    # First use during shutdown is too late to wrap the active call. Restrict
    # idle connections in that case or when a loop prohibits wrapping.
    owner.closing = _shutting_down(loop)
    try:
        setattr(loop, _LOOP_ATTRIBUTE, owner)
    except (AttributeError, TypeError):
        return None
    if not owner.closing:
        try:
            setattr(loop, "create_task", owner.create_task)
            setattr(loop, "shutdown_asyncgens", owner.shutdown_asyncgens)
        except (AttributeError, TypeError):
            if getattr(loop, "create_task", None) == owner.create_task:
                setattr(loop, "create_task", owner._create_task)
            owner.closing = True
        else:
            owner.start_keeper()
    with _CLIENT_CACHE_LOCK:
        stale = [item for item in _LOOP_RECORDS if item.loop.is_closed()]
        for item in stale:
            item.take_locked()
            _LOOP_RECORDS.discard(item)
        _LOOP_RECORDS.add(owner)
    return owner


async def _aclose_client(client: Any) -> None:
    close = getattr(client, "close", None) or getattr(client, "aclose", None)
    if not callable(close):
        return
    try:
        result = close()
        if inspect.isawaitable(result):
            await result
    except Exception:
        return


def _close_client(client: Any, loop: asyncio.AbstractEventLoop | None = None) -> None:
    """Best-effort close one SDK client; an async client closes on the loop that owns it.

    A client whose loop is closed is dropped: its connections belong to that loop
    and cannot be closed from another one.
    """
    if loop is not None:
        if loop.is_closed():
            return

        def start() -> None:
            task = loop.create_task(_aclose_client(client))
            _CLOSING.add(task)
            task.add_done_callback(_CLOSING.discard)

        try:
            if _running_loop() is loop:
                start()
            else:
                loop.call_soon_threadsafe(start)
        except RuntimeError:  # the loop closed meanwhile
            return
        return
    close = getattr(client, "close", None) or getattr(client, "aclose", None)
    if not callable(close):
        return
    try:
        result = close()
        if inspect.isawaitable(result):
            # An async client without a live loop: never await it on a foreign loop.
            getattr(result, "close", lambda: None)()
    except Exception:
        return


def _cached_client(cache: Dict[_ClientKey, Any], key: _ClientKey) -> Any:
    """Return and refresh one cached client while holding no caller lock."""
    with _CLIENT_CACHE_LOCK:
        client = cache.get(key)
        if client is not None and isinstance(cache, OrderedDict):
            cache.move_to_end(key)
        return client


def _authority(resolver: Any) -> Any:
    """Return the guard that keeps a resolver authority active, or a no-op context."""
    guard = getattr(resolver, "_authority_guard", None)
    if not callable(guard):
        guard = getattr(getattr(resolver, "catalog", None), "_authority_guard", None)
    return guard() if callable(guard) else nullcontext()


def _publish_client(cache: Dict[_ClientKey, Any], key: _ClientKey, created: Any, *, resolver: Any, loop: Any = None) -> Any:
    """Publish one client while its resolver authority remains active; ``loop`` owns an async cache."""
    evicted = None
    try:
        with _authority(resolver):
            with _CLIENT_CACHE_LOCK:
                client = cache.get(key)
                if client is None:
                    cache[key] = created
                    client = created
                    if isinstance(cache, OrderedDict):
                        cache.move_to_end(key)
                    if len(cache) > _CLIENT_CACHE_MAX:
                        oldest = next(iter(cache))
                        if oldest != key:
                            evicted = cache.pop(oldest)
                elif isinstance(cache, OrderedDict):
                    cache.move_to_end(key)
    except BaseException:
        _close_client(created, loop)
        raise
    if client is not created:
        _close_client(created, loop)
    if evicted is not None:
        _close_client(evicted, loop)
    return client


@contextmanager
def _client_build_lock(cache: Dict[_ClientKey, Any], key: _ClientKey):
    """Hold the one build lock for a cache key; the slot is dropped once unused."""
    slot = (id(cache), key)
    with _CLIENT_CACHE_LOCK:
        entry = _CLIENT_BUILDS.get(slot)
        if entry is None:
            entry = _CLIENT_BUILDS[slot] = [threading.Lock(), 0]
        entry[1] += 1
    try:
        with entry[0]:
            yield
    finally:
        with _CLIENT_CACHE_LOCK:
            entry[1] -= 1
            if entry[1] == 0 and _CLIENT_BUILDS.get(slot) is entry:
                del _CLIENT_BUILDS[slot]


def _shared_client(cache: Dict[_ClientKey, Any], key: _ClientKey, build: Any, *, resolver: Any, loop: Any = None) -> Any:
    """Return the cached client for a key, building it at most once across racing threads."""
    client = _cached_client(cache, key)
    if client is not None:
        return client
    with _client_build_lock(cache, key):
        client = _cached_client(cache, key)
        if client is not None:
            return client
        return _publish_client(cache, key, build(), resolver=resolver, loop=loop)


def _with_http_client(build: Any, http_class: Any, late: bool, mounts: Dict[str, Any] | None = None) -> Any:
    """Build a loop's client on the SDK's default HTTP client class.

    The SDK's own default subclass closes a garbage-collected client on whatever
    loop is running at that moment, which fails for connections of a closed loop.
    A client built after its loop began shutting down keeps no idle connection.
    ``mounts`` (see ``_direct_mounts``) reach a loopback server without a proxy.
    """
    options: Dict[str, Any] = {} if mounts is None else {"mounts": mounts}
    if not late:
        return build(http_client=http_class(**options))
    import httpx

    return build(http_client=http_class(limits=httpx.Limits(max_connections=_SDK_MAX_CONNECTIONS, max_keepalive_connections=0), **options))


def _sdk_http_class(module: str, *, asynchronous: bool = True) -> Any:
    """Return the SDK's public default HTTP client class, or ``None`` when the SDK has none."""
    return getattr(sys.modules.get(module), "DefaultAsyncHttpxClient" if asynchronous else "DefaultHttpxClient", None)


def _is_loopback(host: str) -> bool:
    """Return whether ``host`` names this machine: ``localhost``, a ``*.localhost`` name or a loopback address."""
    name = host.lower().rstrip(".")
    if name == "localhost" or name.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(name).is_loopback
    except ValueError:
        return False


def _loopback_host(base_url: Any) -> str | None:
    """Return the host of a loopback ``base_url``, or ``None`` for any other."""
    if not isinstance(base_url, str):
        return None
    try:
        host = urlsplit(base_url).hostname
    except ValueError:
        return None
    return host if host and _is_loopback(host) else None


def _loopback_proxies(spec: LLMSpec, base_url: Any) -> Dict[str, Any]:
    """Return the proxy settings that reach a loopback ``base_url`` through a proxy, or ``{}`` when it is reached directly or is remote.

    A proxy is the LLM's own ``http_proxy`` or ``https_proxy``, the one passed to this call, or that of a ``NetworkProxy``
    block the caller opened around it. A call enters its options, the LLM's own among them, as a ``NetworkProxy`` block;
    an empty one clears what is named outside it.
    """
    if _loopback_host(base_url) is None:
        return {}
    own = {name: spec.args.get(name) for name in ("http_proxy", "https_proxy", "no_proxy") if spec.args.get(name)}
    entered = NetworkProxy._own_settings()
    named = {"HTTP_PROXY": own.get("http_proxy"), "HTTPS_PROXY": own.get("https_proxy"), **entered}
    return {"own": own, "entered": entered} if named["HTTP_PROXY"] or named["HTTPS_PROXY"] else {}


def _direct_mounts(spec: LLMSpec, base_url: Any) -> Dict[str, Any] | None:
    """Return httpx mounts that reach a loopback ``base_url`` directly, or ``None`` when none are needed.

    The SDK clients take their proxy from the environment and, through ``urllib.request.getproxies()``, from the
    system settings (the Windows registry, macOS System Configuration), but not from the system's bypass list, so
    a system proxy also caught a local model server such as ``http://127.0.0.1:8000/v1``. A loopback ``base_url``
    is therefore reached without a proxy, unless this LLM or this call names one (see ``_loopback_proxies``).
    """
    host = _loopback_host(base_url)
    if host is None or _loopback_proxies(spec, base_url):
        return None
    return {f"all://[{host}]" if ":" in host else f"all://{host}": None}


def _sdk_client(sdk_class: Any, kwargs: Dict[str, Any], http_class: Any, mounts: Dict[str, Any] | None) -> Any:
    """Build an SDK client, on an HTTP client with ``mounts`` when there are any."""
    if mounts is None or http_class is None:
        return sdk_class(**kwargs)
    return sdk_class(**kwargs, http_client=http_class(mounts=mounts))


def _loop_client(kind: str, spec: LLMSpec, digest: str, sdk_class: Any, kwargs: Dict[str, Any], http_class: Any) -> Any:
    """Return the running loop's async SDK client for one key.

    Outside a running loop there is no loop to bind a shared client to, so the
    caller gets a new client it owns.
    """
    owner = _loop_clients()
    base_url = kwargs.get("base_url")
    mounts = _direct_mounts(spec, base_url)
    if owner is None:
        with _authority(spec.resolver):
            return _sdk_client(sdk_class, kwargs, http_class, mounts)
    build = partial(sdk_class, **kwargs)
    if http_class is not None:
        build = partial(_with_http_client, build, http_class, owner.closing, mounts)
    return _shared_client(owner.caches[kind], _client_key(spec, digest, base_url), build, resolver=spec.resolver, loop=owner.loop)


def dispose_llm_clients(authority: Any | None = None) -> None:
    """Close clients owned by one resolver authority, or every client."""
    authority_id = None if authority is None else getattr(authority, "authority_token", id(authority))
    removed = []
    with _CLIENT_CACHE_LOCK:
        caches: list[tuple[Dict[_ClientKey, Any], Any]] = [(_OPENAI_CLIENTS, None), (_ANTHROPIC_CLIENTS, None)]
        caches.extend((cache, owner.loop) for owner in list(_LOOP_RECORDS) for cache in owner.caches.values())
        for cache, loop in caches:
            keys = [key for key in cache if authority_id is None or key[0] == authority_id]
            removed.extend((cache.pop(key), loop) for key in keys)
    for client, loop in removed:
        _close_client(client, loop)


PORTKEY_OPENROUTER_EMBEDDING_ERROR = (
    "Portkey Gateway + OpenRouter embeddings are temporarily unsupported in Portkey Gateway; "
    "see https://github.com/Portkey-AI/gateway/pull/1622. Use gateway='openai' or gateway='litellm' for OpenRouter embeddings for now."
)
BIFROST_IMAGE_ERROR = (
    "Bifrost image generation is temporarily unsupported; see https://github.com/maximhq/bifrost/issues/2367. "
    "Use gateway='openai' or gateway='litellm' for image generation for now."
)


def _reset_client_locks_after_fork() -> None:
    """Discard inherited SDK clients and locks in the child process."""
    global _CLIENT_CACHE_LOCK, _OPENAI_CLIENTS, _ANTHROPIC_CLIENTS, _LOOP_RECORDS, _CLOSING
    _CLIENT_CACHE_LOCK = threading.RLock()
    _CLIENT_BUILDS.clear()
    _OPENAI_CLIENTS = OrderedDict()
    _ANTHROPIC_CLIENTS = OrderedDict()
    _LOOP_RECORDS = weakref.WeakSet()
    _CLOSING = set()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_client_locks_after_fork)


def _as_dict(obj: Any) -> Dict[str, Any]:
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    return dict(obj)


def _openai_client(spec: LLMSpec, mode: str):
    runtime = spec.engine.apply(spec, profile=mode)
    key = _client_key(spec, spec.client_key(), runtime.get("base_url"))
    client = _cached_client(_OPENAI_CLIENTS, key)
    if client is not None:
        return client, runtime
    try:
        from openai import OpenAI
    except Exception as exc:
        raise ImportError("Install the openai package to use HeavenBase LLM OpenAI-compatible gateways.") from exc
    kwargs = {"api_key": runtime.get("api_key")}
    if runtime.get("base_url"):
        kwargs["base_url"] = runtime["base_url"]
    if runtime.get("headers"):
        kwargs["default_headers"] = runtime["headers"]
    if runtime.get("args", {}).get("timeout") is not None:
        kwargs["timeout"] = runtime["args"].get("timeout")
    if runtime.get("args", {}).get("max_retries") is not None:
        kwargs["max_retries"] = runtime["args"].get("max_retries")
    build = partial(_sdk_client, OpenAI, kwargs, _sdk_http_class("openai", asynchronous=False), _direct_mounts(spec, kwargs.get("base_url")))
    client = _shared_client(_OPENAI_CLIENTS, key, build, resolver=spec.resolver)
    return client, runtime


def _openai_async_client(spec: LLMSpec, mode: str):
    runtime = spec.engine.apply(spec, profile=mode)
    digest = spec.client_key()
    owner = _loop_clients()
    if owner is not None:
        client = _cached_client(owner.caches["openai"], _client_key(spec, digest, runtime.get("base_url")))
        if client is not None:
            return client, runtime
    try:
        from openai import AsyncOpenAI
    except Exception as exc:
        raise ImportError("Install the openai package to use HeavenBase LLM async OpenAI-compatible gateways.") from exc
    kwargs = {"api_key": runtime.get("api_key")}
    if runtime.get("base_url"):
        kwargs["base_url"] = runtime["base_url"]
    if runtime.get("headers"):
        kwargs["default_headers"] = runtime["headers"]
    if runtime.get("args", {}).get("timeout") is not None:
        kwargs["timeout"] = runtime["args"].get("timeout")
    if runtime.get("args", {}).get("max_retries") is not None:
        kwargs["max_retries"] = runtime["args"].get("max_retries")
    client = _loop_client("openai", spec, digest, AsyncOpenAI, kwargs, _sdk_http_class("openai"))
    return client, runtime


def _strip_path_suffix(value: Any, suffix: str) -> Any:
    if not isinstance(value, str):
        return value
    clean = value.rstrip("/")
    return clean[: -len(suffix)] if clean.endswith(suffix) else clean


def _bifrost_anthropic_base_url(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    clean = value.rstrip("/")
    if clean.endswith("/anthropic"):
        return clean
    return f"{_strip_path_suffix(clean, '/v1')}/anthropic"


def _anthropic_gateway_runtime(spec: LLMSpec, mode: str) -> Dict[str, Any]:
    runtime = dict(spec.engine.apply(spec, profile=mode))
    runtime["headers"] = dict(runtime.get("headers", {}) or {})
    if mode == "portkey":
        api_key = runtime.get("api_key")
        if (
            api_key
            and api_key != "portkey"
            and "x-portkey-api-key" not in runtime["headers"]
            and runtime["headers"].get("Authorization") != f"Bearer {api_key}"
        ):
            runtime["headers"]["x-portkey-api-key"] = api_key
        runtime["base_url"] = _strip_path_suffix(runtime.get("base_url"), "/v1")
        runtime["api_key"] = "portkey"
    elif mode == "bifrost":
        runtime["base_url"] = _bifrost_anthropic_base_url(runtime.get("base_url"))
    return runtime


def _anthropic_client_key(runtime: Dict[str, Any], authority: Any = None) -> str:
    args = runtime.get("args", {}) or {}
    return sha256hash(
        {
            "gateway": runtime.get("gateway"),
            "endpoint": "anthropic",
            "authority": authority,
            "api_key": runtime.get("api_key"),
            "auth_token": runtime.get("auth_token"),
            "base_url": runtime.get("base_url"),
            "headers": runtime.get("headers", {}) or {},
            "timeout": args.get("timeout"),
            "max_retries": args.get("max_retries"),
        }
    )


def _anthropic_client(spec: LLMSpec, mode: str = "anthropic"):
    runtime = _anthropic_gateway_runtime(spec, mode)
    key = _client_key(spec, _anthropic_client_key(runtime, spec.authority_key), runtime.get("base_url"))
    client = _cached_client(_ANTHROPIC_CLIENTS, key)
    if client is not None:
        return client, runtime
    try:
        from anthropic import Anthropic
    except Exception as exc:
        raise ImportError("Install the anthropic package to use HeavenBase LLM Anthropic gateways.") from exc
    kwargs = _anthropic_client_kwargs(runtime)
    build = partial(_sdk_client, Anthropic, kwargs, _sdk_http_class("anthropic", asynchronous=False), _direct_mounts(spec, kwargs.get("base_url")))
    client = _shared_client(_ANTHROPIC_CLIENTS, key, build, resolver=spec.resolver)
    return client, runtime


def _anthropic_async_client(spec: LLMSpec, mode: str = "anthropic"):
    runtime = _anthropic_gateway_runtime(spec, mode)
    digest = _anthropic_client_key(runtime, spec.authority_key)
    owner = _loop_clients()
    if owner is not None:
        client = _cached_client(owner.caches["anthropic"], _client_key(spec, digest, runtime.get("base_url")))
        if client is not None:
            return client, runtime
    try:
        from anthropic import AsyncAnthropic
    except Exception as exc:
        raise ImportError("Install the anthropic package to use HeavenBase LLM async Anthropic gateways.") from exc
    kwargs = _anthropic_client_kwargs(runtime)
    client = _loop_client("anthropic", spec, digest, AsyncAnthropic, kwargs, _sdk_http_class("anthropic"))
    return client, runtime


def _anthropic_client_kwargs(runtime: Dict[str, Any]) -> Dict[str, Any]:
    args = runtime.get("args", {}) or {}
    kwargs: Dict[str, Any] = {}
    auth_token = runtime.get("auth_token")
    api_key = runtime.get("api_key")
    if auth_token:
        kwargs["auth_token"] = auth_token
    elif api_key and api_key != "anthropic":
        kwargs["api_key"] = api_key
    if runtime.get("base_url"):
        kwargs["base_url"] = runtime["base_url"]
    if runtime.get("headers"):
        kwargs["default_headers"] = runtime["headers"]
    if args.get("timeout") is not None:
        kwargs["timeout"] = args.get("timeout")
    if args.get("max_retries") is not None:
        kwargs["max_retries"] = args.get("max_retries")
    return kwargs


async def _run_sync(fn: Any, *args: Any, **kwargs: Any) -> Any:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, partial(fn, *args, **kwargs))


def _next_item(iterator: Any) -> Any:
    try:
        return next(iterator)
    except StopIteration:
        return _END


async def _async_stream(items: Any):
    iterator = iter(items)
    while True:
        chunk = await _run_sync(_next_item, iterator)
        if chunk is _END:
            return
        yield chunk


def _deep_merge(left: Dict[str, Any], right: Dict[str, Any], path: tuple[str, ...] = ()) -> Dict[str, Any]:
    out = dict(left or {})
    for key, value in (right or {}).items():
        if isinstance(out.get(key), dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value, (*path, key))
        elif key in out and out[key] != value:
            location = ".".join((*path, key))
            raise ValueError(f"LLM semantic argument conflicts with explicit native argument at extra_body.{location}")
        else:
            out[key] = value
    return out


def _set_extra_body(args: Dict[str, Any], data: Dict[str, Any]) -> None:
    args["extra_body"] = _deep_merge(args.get("extra_body", {}) or {}, data)


def _thinking_control(args: Dict[str, Any]) -> Dict[str, Any]:
    control = {
        "think": args.pop("think", None),
        "effort": args.pop("reasoning_effort", None),
        "max_tokens": args.pop("reasoning_max_tokens", None),
        "exclude": args.pop("reasoning_exclude", None),
    }
    if control["think"] is None and any(control[key] is not None for key in ("effort", "max_tokens", "exclude")):
        control["think"] = True
    return control


def _reasoning_body(control: Dict[str, Any]) -> Dict[str, Any]:
    think = control["think"]
    reasoning = {"enabled": bool(think)}
    if think:
        if control["max_tokens"]:
            reasoning["max_tokens"] = control["max_tokens"]
        elif control["effort"]:
            reasoning["effort"] = control["effort"]
        if control["exclude"] is not None:
            reasoning["exclude"] = control["exclude"]
    return {"reasoning": reasoning}


def _thinking_type_body(control: Dict[str, Any]) -> Dict[str, Any]:
    thinking = {"type": "enabled" if control["think"] else "disabled"}
    if control["think"] and control["max_tokens"]:
        thinking["budget_tokens"] = control["max_tokens"]
    elif control["think"] and control["effort"]:
        thinking["effort"] = control["effort"]
    return {"thinking": thinking}


def _native_thinking_body(control: Dict[str, Any], model: str, args: Dict[str, Any]) -> Dict[str, Any]:
    if control["think"] is False:
        if args.get("thinking") not in (None, {"type": "disabled"}):
            raise ValueError("LLM semantic argument conflicts with explicit native argument at thinking")
        generated: Dict[str, Any] = {}
    elif control["max_tokens"] is not None:
        if _anthropic_adaptive_only_model(model):
            raise ValueError(
                f"Anthropic model {model!r} does not support manual thinking budgets; use think=True without reasoning_max_tokens (adaptive thinking)."
            )
        budget = int(control["max_tokens"])
        if budget < 1024:
            raise ValueError("Anthropic manual thinking budgets must be at least 1024 tokens.")
        max_tokens = _output_limit(args)
        if max_tokens is not None and int(max_tokens) <= budget:
            raise ValueError("Anthropic manual thinking budget must be lower than max_tokens (or max_completion_tokens).")
        generated = {"thinking": {"type": "enabled", "budget_tokens": budget, "display": "summarized"}}
    elif _claude_manual_only_model(model):
        # Claude before 4.6 has no adaptive thinking: think=True takes the smallest manual budget.
        max_tokens = _output_limit(args)
        if max_tokens is not None and int(max_tokens) <= _MIN_THINKING_BUDGET:
            raise ValueError(
                f"Claude model {model!r} has no adaptive thinking, so think=True sends a manual budget of {_MIN_THINKING_BUDGET} tokens, "
                f"which needs max_tokens (or max_completion_tokens) above {_MIN_THINKING_BUDGET}; pass a larger one (and reasoning_max_tokens below it for a larger budget)."
            )
        generated = {"thinking": {"type": "enabled", "budget_tokens": _MIN_THINKING_BUDGET, "display": "summarized"}}
    else:
        generated = {"thinking": {"type": "adaptive", "display": "summarized"}}

    for key, value in generated.items():
        if key in args and args[key] != value:
            raise ValueError(f"LLM semantic argument conflicts with explicit native argument at {key}")
    out = {key: value for key, value in generated.items() if key not in args}
    if control["effort"] is not None and _claude_without_effort(model):
        raise ValueError(
            f"Claude model {model!r} does not accept an effort level (reasoning_effort); use think=True or reasoning_max_tokens without reasoning_effort."
        )
    if control["effort"] is not None:
        # The Messages API takes the effort level in output_config; it has no top-level effort field.
        config = dict(args.get("output_config") or {})
        if config.get("effort") not in (None, control["effort"]):
            raise ValueError("LLM semantic argument conflicts with explicit native argument at output_config.effort")
        out["output_config"] = {**config, "effort": control["effort"]}
    return out


_MIN_THINKING_BUDGET = 1024
_ANTHROPIC_DEFAULT_MAX_TOKENS = 1024


def _output_limit(args: Mapping[str, Any]) -> Any:
    """Return the output-token limit a request sets: ``max_tokens``, or else OpenAI's ``max_completion_tokens``."""
    return args.get("max_tokens") if args.get("max_tokens") is not None else args.get("max_completion_tokens")


_CLAUDE_VERSION = re.compile(r"claude-(?:[a-z]+-)?(\d{1,2})(?:[-.](\d{1,2})(?!\d))?")


def _claude_version(model: str) -> tuple[int, int] | None:
    """Return a Claude model id's (major, minor) version, or ``None`` for an id that is not Claude's."""
    match = _CLAUDE_VERSION.search(str(model or "").lower())
    return (int(match.group(1)), int(match.group(2) or 0)) if match else None


def _claude_model(model: str) -> bool:
    return "claude" in str(model or "").lower()


def _claude_manual_only_model(model: str) -> bool:
    """Whether a Claude model predates adaptive thinking (introduced with the 4.6 models)."""
    version = _claude_version(model)
    return version is not None and version < (4, 6)


def _claude_without_effort(model: str) -> bool:
    """Whether a Claude model refuses an effort level: models before 4.6, except Opus 4.5, which introduced it."""
    version = _claude_version(model)
    return version is not None and version < (4, 6) and not (version == (4, 5) and "opus" in str(model).lower())


def _anthropic_adaptive_only_model(model: str) -> bool:
    text = str(model or "").lower()
    return any(
        token in text
        for token in (
            "fable-5",
            "sonnet-5",
            "opus-5",
            "opus-4-7",
            "opus-4.7",
            "opus-4-8",
            "opus-4.8",
        )
    )


def _apply_thinking(runtime: Dict[str, Any], spec: LLMSpec, args: Dict[str, Any]) -> Dict[str, Any]:
    control = _thinking_control(args)
    if control["think"] is None:
        return args
    if runtime.get("thinking_control") == "extra_body_reasoning":
        _set_extra_body(args, _reasoning_body(control))
    elif runtime.get("thinking_control") == "extra_body_thinking_type":
        _set_extra_body(args, _thinking_type_body(control))
    elif runtime.get("thinking_control") == "native_thinking":
        args.update(_native_thinking_body(control, runtime.get("model", spec.model_id), args))
    elif runtime.get("thinking_control") == "extra_body_native_thinking":
        _set_extra_body(args, _extra_body_native_thinking(control, runtime.get("model", spec.model_id), args))
    elif runtime.get("thinking_control") == "anthropic_compat_thinking":
        _set_extra_body(args, _anthropic_compat_thinking_body(control, runtime.get("model", spec.model_id), args))
    elif runtime.get("thinking_control") == "bifrost_claude_reasoning":
        _set_extra_body(args, _bifrost_claude_reasoning_body(control, runtime.get("model", spec.model_id), args))
    elif control["think"] is True:
        raise ValueError(f"LLM route {spec.endpoint}/{spec.gateway} does not support semantic thinking controls")
    return args


def _claude_enabled_budget(control: Dict[str, Any], model: str, args: Dict[str, Any], route: str) -> int | None:
    """Return the manual budget an OpenAI-format Claude route sends for ``control``, or ``None`` with thinking off.

    These routes only carry Claude's ``enabled`` thinking with a budget. The budget follows the
    native rules (``reasoning_max_tokens``, at least 1024 and below ``max_tokens``, refused on the
    adaptive-only models), and ``think=True`` takes the smallest one.
    """
    if control["effort"] is not None:
        raise ValueError(
            f"{route} cannot send Claude's effort level (reasoning_effort). Use gateway='anthropic', "
            "or think=True / reasoning_max_tokens without reasoning_effort."
        )
    if control["think"] is False:
        return None
    thinking = _native_thinking_body(control, model, args).get("thinking") or {}
    budget = int(thinking.get("budget_tokens") or _MIN_THINKING_BUDGET)
    max_tokens = _output_limit(args)
    if max_tokens is not None and int(max_tokens) <= budget:
        raise ValueError(f"{route} sends Claude thinking as a manual budget of {budget} tokens, which needs max_tokens (or max_completion_tokens) above it.")
    return budget


def _extra_body_native_thinking(control: Dict[str, Any], model: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Claude's native ``thinking`` field as an extra body field (Portkey's OpenAI format passes it on to Claude).

    The OpenAI SDK takes no ``thinking`` keyword, and the route has no field for the effort level.
    """
    if control["effort"] is not None:
        raise ValueError(
            "Portkey's OpenAI format cannot send Claude's effort level (reasoning_effort). Use endpoint='anthropic' on "
            "gateway='portkey', or think=True / reasoning_max_tokens without reasoning_effort."
        )
    return _native_thinking_body(control, model, args)


def _anthropic_compat_thinking_body(control: Dict[str, Any], model: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Claude thinking for Anthropic's OpenAI-compatibility endpoint: a top-level ``thinking`` field.

    The endpoint takes ``{"type": "enabled", "budget_tokens": N}`` (N at least 1024 and below
    ``max_tokens``), refuses ``adaptive``, converts ``enabled`` to adaptive thinking on the models
    that refuse ``enabled``, and ignores ``reasoning_effort``.
    """
    budget = _claude_enabled_budget(control, model, args, "Anthropic's OpenAI-compatibility endpoint (gateway='openai')")
    return {} if budget is None else {"thinking": {"type": "enabled", "budget_tokens": budget}}


def _bifrost_claude_reasoning_body(control: Dict[str, Any], model: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Claude thinking for Bifrost's OpenAI format: ``reasoning.max_tokens``, which Bifrost sends as an ``enabled`` budget."""
    if control["think"] is not False and _anthropic_adaptive_only_model(model):
        raise ValueError(
            f"Bifrost's OpenAI format sends Claude thinking as type 'enabled', which {model!r} refuses (adaptive thinking only). "
            "Use endpoint='anthropic' on gateway='bifrost', or gateway='anthropic'."
        )
    budget = _claude_enabled_budget(control, model, args, "Bifrost's OpenAI format")
    return {} if budget is None else {"reasoning": {"max_tokens": budget}}


def _claude_thinking_requested(model: Any, args: Mapping[str, Any]) -> bool:
    """Whether request options turn Claude thinking on (``think``/``reasoning_*``, ``thinking`` or ``extra_body``)."""
    if not _claude_model(model):
        return False
    think = args.get("think")
    if think is True or (think is None and any(args.get(key) is not None for key in ("reasoning_effort", "reasoning_max_tokens", "reasoning_exclude"))):
        return True
    if think is False:
        return False
    extra = args.get("extra_body") if isinstance(args.get("extra_body"), Mapping) else {}
    reasoning = extra.get("reasoning")
    return (
        _anthropic_thinking_on(args.get("thinking"))
        or _anthropic_thinking_on(extra.get("thinking"))
        or (isinstance(reasoning, Mapping) and reasoning.get("enabled") is not False and bool(reasoning))
    )


def _claude_openai_request(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Make an OpenAI-format request for a Claude model with thinking on valid for Claude.

    Claude accepts no ``top_k`` and no ``temperature`` other than its default with thinking on,
    so both are left out, top-level and in ``extra_body``. LiteLLM's native ``thinking`` field has
    no way to carry the effort level, so ``reasoning_effort`` is refused there.
    """
    if not _claude_thinking_requested(payload.get("model"), payload):
        return payload
    if _anthropic_thinking_on(payload.get("thinking")) and (payload.get("output_config") or {}).get("effort") is not None:
        raise ValueError(
            "This OpenAI-format route cannot send Claude's effort level (reasoning_effort): it has no output_config. "
            "Use gateway='anthropic' (or endpoint='anthropic' on Portkey and Bifrost), or think=True / reasoning_max_tokens without reasoning_effort."
        )
    _without_thinking_sampling(payload)
    if isinstance(payload.get("extra_body"), dict):
        payload["extra_body"] = {k: v for k, v in payload["extra_body"].items() if k not in ("temperature", "top_k")}
    return payload


_OPENROUTER_HISTORY_GATEWAYS = {"openai", "portkey", "bifrost"}
_OPENROUTER_REPLAY_GATEWAYS = _OPENROUTER_HISTORY_GATEWAYS | {"litellm"}
_REASONING_CONTENT_BACKENDS = {"dashscope", "deepseek", "hosted_vllm", "minimax", "moonshot", "zai", "zhipu"}
THINK_KEYS = {"reasoning", "reasoning_content", "reasoning_details", "signature", "think", "thinking", "thinking_blocks", "thinking_content"}
THINK_BLOCK_TYPES = {
    "reasoning",
    "reasoning_content",
    "reasoning_delta",
    "reasoning.encrypted",
    "reasoning.summary",
    "reasoning.text",
    "redacted_thinking",
    "signature_delta",
    "thinking",
    "thinking_delta",
}
_REPLAY_FIELDS = THINK_KEYS | {"content_blocks", "encrypted_content"}


def _claude_replay_format(spec: LLMSpec) -> str | None:
    """Return the assistant history extension supported by this Claude Chat Completions route."""
    if spec.endpoint != "openai" or not _claude_model(spec.model_id):
        return None
    backend = spec.provider_backend()
    if backend == "openrouter" and spec.gateway in _OPENROUTER_HISTORY_GATEWAYS:
        return "reasoning_details"
    if backend == "anthropic":
        if spec.gateway == "bifrost":
            return "reasoning_details"
        if spec.gateway == "portkey":
            return "content_blocks"
        if spec.gateway == "litellm":
            return "thinking_blocks"
    return None


def _thinking_history_enabled(args: Mapping[str, Any]) -> bool:
    extra = args.get("extra_body")
    extra = extra if isinstance(extra, Mapping) else {}
    thinking = extra.get("thinking", args.get("thinking"))
    reasoning = extra.get("reasoning")
    return (
        args.get("think") is not False
        and not (isinstance(thinking, Mapping) and thinking.get("type") == "disabled")
        and not (isinstance(reasoning, Mapping) and reasoning.get("enabled") is False)
    )


def _without_thinking_content_blocks(blocks: List[Any]) -> List[Any]:
    return [block for block in blocks if not isinstance(block, dict) or (block.get("type") or block.get("kind")) not in THINK_BLOCK_TYPES]


def _portkey_history_blocks(message: Dict[str, Any], thinking: bool) -> List[Any] | None:
    original = message.get("content_blocks")
    blocks = _replayable_thinking_blocks(message.get("thinking_blocks")) if thinking else []
    if isinstance(original, list):
        if not thinking:
            return _without_thinking_content_blocks(original)
        present = _replayable_thinking_blocks(original)
        missing = [block for block in blocks if block not in present]
        return [*missing, *original]
    if blocks:
        content = message.get("content")
        if isinstance(content, str):
            content = [{"type": "text", "text": content}] if content else []
        return [*blocks, *(content if isinstance(content, list) else [])]
    return None


def _openai_history(messages: Any, spec: LLMSpec | None, thinking: bool) -> Any:
    """Project provider extensions onto the destination's assistant-message schema."""
    if not isinstance(messages, list):
        return messages
    replay_format = _claude_replay_format(spec) if spec is not None else None
    backend = spec.provider_backend() if spec is not None else None
    accepted = set()
    if thinking:
        if backend == "openrouter" and spec is not None and spec.gateway in _OPENROUTER_REPLAY_GATEWAYS:
            accepted.update({"reasoning", "reasoning_content", "reasoning_details"})
        elif backend in _REASONING_CONTENT_BACKENDS:
            accepted.add("reasoning_content")
        if backend == "dashscope":
            accepted.add("encrypted_content")
        if replay_format in {"thinking_blocks", "reasoning_details"}:
            accepted.add(replay_format)
    bifrost_claude = backend == "anthropic" and replay_format == "reasoning_details"
    history = []
    for message in messages:
        if not isinstance(message, dict):
            history.append(message)
            continue
        assistant = message.get("role") == "assistant"
        allowed = accepted if assistant else ()
        wire = {key: value for key, value in message.items() if key not in _REPLAY_FIELDS or key in allowed}
        if assistant:
            if isinstance(wire.get("content"), list):
                wire["content"] = _without_thinking_content_blocks(wire["content"])
            if replay_format == "content_blocks":
                if (blocks := _portkey_history_blocks(message, thinking)) is not None:
                    wire["content_blocks"] = blocks
            if bifrost_claude and isinstance(wire.get("reasoning_details"), list):
                details = [item for detail in wire.pop("reasoning_details") if (item := _bifrost_claude_detail(detail)) is not None]
                if details:
                    wire["reasoning_details"] = details
        history.append(wire)
    return history


def _bifrost_claude_detail(detail: Any) -> Dict[str, Any] | None:
    """Bifrost maps each detail to a native Claude block, which must be signed or redacted."""
    if not isinstance(detail, dict):
        return None
    common = {key: detail[key] for key in ("id", "index", "type") if key in detail}
    if detail.get("type") == "reasoning.text" and detail.get("signature"):
        return {**common, "text": detail.get("text") or "", "signature": detail["signature"]}
    if detail.get("type") == "reasoning.encrypted" and detail.get("data"):
        return {**common, "data": detail["data"]}
    return None


def _openai_payload(runtime: Dict[str, Any], spec: LLMSpec | None = None, **kwargs: Any) -> Dict[str, Any]:
    """Build an OpenAI-compatible request body.

    LiteLLM reads ``thinking_blocks`` directly. Portkey's Anthropic adapter reads native
    blocks from ``content_blocks``. The other routes use their own history fields.
    """
    args = dict(runtime.get("args", {}))
    args.update(kwargs)
    if "messages" in args:
        thinking = _thinking_history_enabled(args)
        args["messages"] = _openai_history(args["messages"], spec, thinking)
        extra = args.get("extra_body")
        if isinstance(extra, dict) and "messages" in extra:
            args["extra_body"] = {**extra, "messages": _openai_history(extra["messages"], spec, thinking)}
    if spec is not None:
        args = _apply_thinking(runtime, spec, args)
    payload = {"model": runtime["model"], **{k: v for k, v in args.items() if v is not None and k not in CLIENT_ARG_KEYS}}
    return _claude_openai_request(payload) if "messages" in payload else payload


def _usage_count(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _anthropic_usage(value: Dict[str, Any]) -> Dict[str, Any]:
    usage = dict(value or {})
    # The SDK reports a count it did not receive as null; derive only from counts that are present.
    if _usage_count(usage.get("input_tokens")):
        usage.setdefault("prompt_tokens", usage["input_tokens"])
    if _usage_count(usage.get("output_tokens")):
        usage.setdefault("completion_tokens", usage["output_tokens"])
    prompt, completion = usage.get("prompt_tokens"), usage.get("completion_tokens")
    if not _usage_count(usage.get("total_tokens")) and _usage_count(prompt) and _usage_count(completion):
        usage["total_tokens"] = prompt + completion
    return usage


def _anthropic_stream_usage(usage: Dict[str, Any], event: Dict[str, Any]) -> Dict[str, Any]:
    """Update the running usage and return the increments reported by this event.

    message_start and every message_delta report counts for the whole message so far, so a later
    count replaces an earlier one instead of adding to it. A null count leaves the known count alone.
    """
    kind = event.get("type")
    if kind == "message_start":
        value = (event.get("message") or {}).get("usage")
    elif kind == "message_delta":
        value = event.get("usage")
    else:
        return {}
    previous = _anthropic_usage(usage)
    for key, item in dict(value or {}).items():
        if item is not None or key not in usage:
            usage[key] = item
    increments = {}
    for key, item in _anthropic_usage(usage).items():
        if _usage_count(item):
            old = previous.get(key)
            if not _usage_count(old) or item != old:
                increments[key] = item - old if _usage_count(old) else item
    return increments


def _anthropic_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(_anthropic_text(item) for item in content)
    if isinstance(content, dict):
        if content.get("type") == "text":
            return str(content.get("text") or "")
        return str(content.get("content") or content.get("text") or "")
    return str(content)


def _anthropic_tool_input(arguments: Any) -> Dict[str, Any]:
    if isinstance(arguments, str):
        return loads_json(arguments) if arguments.strip() else {}
    if arguments is None:
        return {}
    return dict(arguments) if isinstance(arguments, dict) else {"value": arguments}


def _anthropic_image_source(url: str) -> Dict[str, Any]:
    if url.startswith("data:image/") and ";base64," in url:
        header, data = url.split(",", 1)
        media_type = header[len("data:") : header.index(";base64")]
        return {"type": "base64", "media_type": media_type, "data": data}
    return {"type": "url", "url": url}


def _anthropic_content_part(item: Any) -> Dict[str, Any]:
    if isinstance(item, dict):
        kind = item.get("type")
        if kind in {"text", "tool_use", "tool_result", "image"}:
            return dict(item)
        if kind == "image_url" or "image_url" in item:
            image_url = item.get("image_url")
            url = image_url.get("url") if isinstance(image_url, dict) else image_url
            return {"type": "image", "source": _anthropic_image_source(str(url or ""))}
        if "text" in item:
            return {"type": "text", "text": str(item.get("text") or "")}
    return {"type": "text", "text": _anthropic_text(item)}


def _anthropic_content(content: Any) -> Any:
    if isinstance(content, list):
        return [_anthropic_content_part(item) for item in content]
    return content if isinstance(content, str) else _anthropic_text(content)


def _anthropic_tool_result(message: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "type": "tool_result",
        "tool_use_id": message.get("tool_call_id") or message.get("id") or message.get("name") or "",
        "content": _anthropic_text(message.get("content")),
    }


def _anthropic_tool_use(tool_call: Dict[str, Any]) -> Dict[str, Any]:
    function = tool_call.get("function") if isinstance(tool_call.get("function"), dict) else {}
    return {
        "type": "tool_use",
        "id": tool_call.get("id") or function.get("name") or tool_call.get("name") or "",
        "name": function.get("name") or tool_call.get("name") or "",
        "input": _anthropic_tool_input(function.get("arguments", tool_call.get("arguments", {}))),
    }


def _replayable_thinking_block(block: Any) -> Dict[str, Any] | None:
    """Return a Claude thinking block as the Messages API needs it sent back, or ``None``.

    A ``thinking`` block counts only once it carries its signature, and a
    ``redacted_thinking`` block only with its encrypted ``data``. The text,
    signature and data are kept byte for byte: the API rejects a modified block.
    """
    if not isinstance(block, dict):
        return None
    if block.get("type") == "thinking" and isinstance(block.get("signature"), str) and block["signature"]:
        return {"type": "thinking", "thinking": str(block.get("thinking") or ""), "signature": block["signature"]}
    if block.get("type") == "redacted_thinking" and isinstance(block.get("data"), str) and block["data"]:
        return {"type": "redacted_thinking", "data": block["data"]}
    return None


def _replayable_thinking_blocks(blocks: Any) -> List[Dict[str, Any]]:
    """Return the complete thinking blocks of ``blocks`` in order, skipping anything else."""
    if not isinstance(blocks, list):
        return []
    return [item for block in blocks if (item := _replayable_thinking_block(block)) is not None]


def _join_thinking_blocks(fragments: Iterable[Any]) -> List[Dict[str, Any]]:
    """Put streamed thinking blocks back together from their per-delta fragments, in order.

    A stream sends a thinking block as text fragments followed by one fragment that carries its
    signature (the native Anthropic converter and LiteLLM both do), and a redacted block whole.
    A thinking fragment opens a block unless one is open; the signature closes it. Text and
    signature are concatenated as sent, so a complete block comes back byte for byte, and a block
    that never got its signature is left out. A whole block (text and signature in one fragment)
    passes through unchanged. A block whose text is empty but signed, as Claude returns with
    thinking ``display: "omitted"``, is a complete block and is kept.
    """
    joined: List[Dict[str, Any]] = []
    current: Dict[str, Any] | None = None
    for fragment in fragments:
        if not isinstance(fragment, dict):
            continue
        if fragment.get("type") == "redacted_thinking":
            current = None
            joined.append(dict(fragment))
        elif fragment.get("type") == "thinking":
            if current is None:
                current = {"type": "thinking", "thinking": "", "signature": ""}
                joined.append(current)
            current["thinking"] += str(fragment.get("thinking") or "")
            current["signature"] += str(fragment.get("signature") or "")
            if current["signature"]:
                current = None
    return _replayable_thinking_blocks(joined)


def _portkey_thinking_blocks(blocks: Any, stream: bool) -> List[Dict[str, Any]]:
    """Read Portkey's Anthropic content blocks or its stream's type-less deltas."""
    if not stream:
        return _replayable_thinking_blocks(blocks)
    out = []
    for block in blocks or []:
        if not isinstance(block, dict):
            continue
        delta = block.get("delta") or {}
        if "data" in delta:
            out.append({"type": "redacted_thinking", "data": delta["data"]})
        elif "thinking" in delta or "signature" in delta:
            out.append({"type": "thinking", "thinking": delta.get("thinking") or "", "signature": delta.get("signature") or ""})
    return out


def _join_reasoning_details(fragments: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Assemble reasoning detail deltas by block index or ID in first-seen order."""
    joined = []
    by_block = {}
    for fragment in fragments:
        index = fragment.get("index")
        identity = ("index", index) if index is not None else ("id", fragment.get("id"))
        if identity[1] is None or identity not in by_block:
            detail = dict(fragment)
            joined.append(detail)
            if identity[1] is not None:
                by_block[identity] = detail
            continue
        detail = by_block[identity]
        for key, value in fragment.items():
            if key in {"text", "summary", "signature", "data"} and isinstance(value, str):
                detail[key] = (detail.get(key) or "") + value
            elif value is not None:
                detail[key] = value
    return joined


def _anthropic_message(message: Dict[str, Any], thinking: bool = False) -> Dict[str, Any] | None:
    role = message.get("role")
    if role in {"system", "developer"}:
        return None
    if role == "tool":
        return {"role": "user", "content": [_anthropic_tool_result(message)]}
    if role not in {"user", "assistant"}:
        role = "user"
    content = _anthropic_content(message.get("content", ""))
    if role == "assistant" and message.get("tool_calls"):
        blocks = content if isinstance(content, list) else ([{"type": "text", "text": content}] if content else [])
        blocks.extend(_anthropic_tool_use(tool_call) for tool_call in message.get("tool_calls") or [])
        # With thinking on, the API requires an assistant tool_use turn to start
        # with the thinking blocks the model returned in it, unmodified.
        content = [*_replayable_thinking_blocks(message.get("thinking_blocks")), *blocks] if thinking else blocks
    return {"role": role, "content": content}


def _anthropic_messages(messages: List[Dict[str, Any]], thinking: bool = False) -> tuple[List[Dict[str, Any]], str]:
    system = "\n".join(_anthropic_text(message.get("content")) for message in messages if message.get("role") in {"system", "developer"})
    out = [item for message in messages if (item := _anthropic_message(message, thinking)) is not None]
    return out, system


def _anthropic_thinking_on(value: Any) -> bool:
    return isinstance(value, dict) and value.get("type") not in (None, "disabled")


def _merged_output_config(built: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    """Merge an extra_body output_config into the one built from the call, refusing a second effort."""
    if built.get("effort") is not None and extra.get("effort") not in (None, built["effort"]):
        raise ValueError("LLM semantic argument conflicts with explicit native argument at extra_body.output_config.effort")
    return {**built, **extra}


def _without_thinking_sampling(payload: Dict[str, Any]) -> None:
    """Drop the sampling settings Claude refuses while thinking is on (callers check the model is Claude).

    With thinking on, Claude accepts no ``top_k`` and no ``temperature`` other than its default
    of 1, so HeavenBase's default ``temperature`` of 0 would fail every thinking request.
    """
    payload.pop("temperature", None)
    payload.pop("top_k", None)


# Thinking controls that build OpenAI-format fields; a Messages payload uses the native control instead.
_OPENAI_FORMAT_CONTROLS = frozenset({"extra_body_reasoning", "extra_body_native_thinking", "anthropic_compat_thinking", "bifrost_claude_reasoning"})


def _litellm_claude_thinking(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Make a LiteLLM request for Claude carry native thinking, or refuse what LiteLLM cannot forward.

    LiteLLM sends an Anthropic ``thinking`` setting through as given, but drops it for a Claude
    model its bundled model list does not mark as thinking-capable unless the request allows it
    explicitly (``allowed_openai_params``). It drops ``output_config`` too, so an effort level
    cannot reach Claude through it.
    """
    if not (_anthropic_thinking_on(payload.get("thinking")) and _claude_model(payload.get("model"))):
        return payload
    # _openai_payload has already left out the sampling settings and refused an effort level.
    allowed = list(payload.get("allowed_openai_params") or [])
    payload["allowed_openai_params"] = allowed if "thinking" in allowed else [*allowed, "thinking"]
    return payload


def _anthropic_tools(tools: Any) -> List[Dict[str, Any]]:
    out = []
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        if tool.get("name") and tool.get("input_schema"):
            out.append(dict(tool))
            continue
        function = tool.get("function") if isinstance(tool.get("function"), dict) else {}
        name = function.get("name") or tool.get("name")
        if name:
            out.append(
                {
                    "name": name,
                    "description": function.get("description") or tool.get("description") or "",
                    "input_schema": function.get("parameters")
                    or function.get("input_schema")
                    or tool.get("input_schema")
                    or {"type": "object", "properties": {}},
                }
            )
    return out


def _anthropic_tool_choice(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        mapping = {"auto": {"type": "auto"}, "none": {"type": "none"}, "required": {"type": "any"}, "any": {"type": "any"}}
        return mapping.get(value, {"type": "tool", "name": value})
    if isinstance(value, dict):
        kind = value.get("type")
        if kind == "function":
            function = value.get("function") if isinstance(value.get("function"), dict) else {}
            return {"type": "tool", "name": function.get("name") or value.get("name") or ""}
        if kind == "required":
            return {"type": "any"}
    return value


# The request fields the Anthropic SDK's messages.create() takes as keyword arguments.
_ANTHROPIC_CREATE_FIELDS = frozenset(
    "model messages max_tokens stream system metadata stop_sequences temperature top_p top_k thinking tools tool_choice "
    "output_config service_tier container inference_geo cache_control".split()
)
# The request arguments a Messages payload takes from the call's options.
_ANTHROPIC_ARGS = frozenset({"max_tokens", "temperature", "top_p", "metadata", "stop_sequences", "thinking", "output_config"})


def _anthropic_payload(runtime: Dict[str, Any], spec: LLMSpec, messages: List[Dict[str, Any]], stream: bool = False, **kwargs: Any) -> Dict[str, Any]:
    args = dict(runtime.get("args", {}))
    args.update(kwargs)
    extra_body = args.get("extra_body")
    limit = _output_limit(extra_body) if isinstance(extra_body, dict) else None
    limit = _output_limit(args) if limit is None else limit
    if limit is not None:
        args["max_tokens"] = limit
    args.pop("max_completion_tokens", None)
    if isinstance(extra_body, dict):
        args["extra_body"] = {key: value for key, value in extra_body.items() if key not in {"max_tokens", "max_completion_tokens"}}
    if args.get("effort") is not None:
        # An explicit Anthropic effort level belongs in output_config, the only place the API reads it.
        args["output_config"] = {**dict(args.get("output_config") or {}), "effort": args.pop("effort")}
    # extra_body.reasoning is an OpenAI-format field that no Messages API reads (it reached the
    # Anthropic SDK as an unknown argument), so a Messages payload uses the native control instead.
    # Other controls, such as a provider's thinking.type setting, are kept as configured.
    control = "native_thinking" if runtime.get("thinking_control") in _OPENAI_FORMAT_CONTROLS else runtime.get("thinking_control")
    args = _apply_thinking({**runtime, "thinking_control": control}, spec, args)
    tools = _anthropic_tools(args.pop("tools", None))
    tool_choice = _anthropic_tool_choice(args.pop("tool_choice", None))
    extra_body = args.pop("extra_body", {}) or {}
    # extra_body is merged last below, so its thinking setting is the one sent.
    thinking = extra_body["thinking"] if isinstance(extra_body, dict) and extra_body.get("thinking") is not None else args.get("thinking")
    prepared, system = _anthropic_messages(messages, thinking=_anthropic_thinking_on(thinking))
    payload = {
        "model": runtime["model"],
        "messages": prepared,
        "stream": stream,
        **{k: v for k, v in args.items() if v is not None and k in _ANTHROPIC_ARGS},
    }
    if "stop" in args and args["stop"] is not None and "stop_sequences" not in payload:
        payload["stop_sequences"] = args["stop"] if isinstance(args["stop"], list) else [args["stop"]]
    if system:
        payload["system"] = system
    if tools:
        payload["tools"] = tools
    if tool_choice:
        payload["tool_choice"] = tool_choice
    if isinstance(extra_body, dict):
        given = {k: v for k, v in extra_body.items() if v is not None}
        if isinstance(given.get("output_config"), dict) and isinstance(payload.get("output_config"), dict):
            given["output_config"] = _merged_output_config(payload["output_config"], given["output_config"])
        payload.update({k: v for k, v in given.items() if k in _ANTHROPIC_CREATE_FIELDS})
        # Any other field rides the SDK's own extra_body, which adds it to the JSON body: as a keyword
        # argument, messages.create() would refuse it with TypeError.
        if extra := {k: v for k, v in given.items() if k not in _ANTHROPIC_CREATE_FIELDS}:
            payload["extra_body"] = extra
    # Last, so a temperature or top_k given in extra_body is dropped too (top_k reaches a Messages
    # payload only through extra_body).
    if _anthropic_thinking_on(thinking) and _claude_model(runtime["model"]):
        _without_thinking_sampling(payload)
        if thinking.get("type") == "enabled" and thinking.get("budget_tokens") is not None:
            budget = int(thinking["budget_tokens"])
            if "max_tokens" not in payload:
                payload["max_tokens"] = budget + _ANTHROPIC_DEFAULT_MAX_TOKENS
            elif int(payload["max_tokens"]) <= budget:
                raise ValueError("Anthropic manual thinking budget must be lower than max_tokens (or max_completion_tokens).")
    payload.setdefault("max_tokens", _ANTHROPIC_DEFAULT_MAX_TOKENS)
    return payload


def _anthropic_response(raw: Any) -> Dict[str, Any]:
    data = _as_dict(raw)
    text_parts: List[str] = []
    think_parts: List[str] = []
    tool_calls: List[Dict[str, Any]] = []
    for block in data.get("content") or []:
        item = _as_dict(block)
        kind = item.get("type")
        if kind == "text":
            text_parts.append(str(item.get("text") or ""))
        elif kind == "thinking":
            think_parts.append(str(item.get("thinking") or item.get("text") or ""))
        elif kind == "tool_use":
            tool_calls.append(
                {
                    "id": item.get("id") or "",
                    "type": "function",
                    "function": {"name": item.get("name") or "", "arguments": dumps_json(item.get("input") or {}, indent=None)},
                }
            )
    message = {"role": data.get("role") or "assistant", "content": "".join(text_parts)}
    if think_parts:
        message["reasoning_content"] = "".join(think_parts)
    if thinking_blocks := _replayable_thinking_blocks([_as_dict(block) for block in data.get("content") or []]):
        message["thinking_blocks"] = thinking_blocks
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "id": data.get("id"),
        "model": data.get("model"),
        "choices": [{"message": message, "finish_reason": data.get("stop_reason"), "index": 0}],
        "usage": _anthropic_usage(data.get("usage") or {}),
        "raw": data,
    }


def _anthropic_stream_delta(event: Dict[str, Any], blocks: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    kind = event.get("type")
    if kind == "content_block_start":
        index = int(event.get("index") or 0)
        block = _as_dict(event.get("content_block") or {})
        if block.get("type") == "redacted_thinking":
            return {"choices": [{"delta": {"thinking_blocks": [{"type": "redacted_thinking", "data": block.get("data") or ""}]}}]}
        if block.get("type") == "tool_use":
            block["_tool_index"] = sum(1 for item in blocks.values() if item.get("type") == "tool_use")
        blocks[index] = block
        if block.get("type") == "tool_use":
            return {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": block["_tool_index"],
                                    "id": block.get("id") or "",
                                    "type": "function",
                                    "function": {
                                        "name": block.get("name") or "",
                                        "arguments": dumps_json(block.get("input") or {}, indent=None) if block.get("input") else "",
                                    },
                                }
                            ]
                        }
                    }
                ]
            }
    if kind == "content_block_delta":
        index = int(event.get("index") or 0)
        delta = _as_dict(event.get("delta") or {})
        if delta.get("type") == "text_delta":
            return {"choices": [{"delta": {"content": delta.get("text") or ""}}]}
        # Each thinking delta also carries its fragment of the block, as LiteLLM's chunks do, so the
        # gathered message can send the block back (see _join_thinking_blocks).
        if delta.get("type") == "thinking_delta":
            text = delta.get("thinking") or ""
            return {"choices": [{"delta": {"reasoning": text, "thinking_blocks": [{"type": "thinking", "thinking": text, "signature": ""}]}}]}
        if delta.get("type") == "signature_delta":
            # The signature verifies the block when it is sent back; it is not thinking text.
            fragment = {"type": "thinking", "thinking": "", "signature": delta.get("signature") or ""}
            return {"choices": [{"delta": {"thinking_blocks": [fragment]}}]}
        if delta.get("type") == "input_json_delta":
            block = blocks.get(index, {})
            return {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": block.get("_tool_index", index),
                                    "id": block.get("id") or "",
                                    "type": "function",
                                    "function": {"name": block.get("name") or "", "arguments": delta.get("partial_json") or ""},
                                }
                            ]
                        }
                    }
                ]
            }
    return {"choices": [{"delta": {}}]}


def _anthropic_stream_state() -> Dict[str, Any]:
    return {"blocks": {}, "usage": {}, "meta": {}}


def _anthropic_stream_chunk(state: Dict[str, Any], event: Any) -> Dict[str, Any]:
    """Convert one stream event to an OpenAI-style chunk that carries the message id and served model.

    Only message_start names the message and the model that served it. OpenAI chunks carry both on
    every chunk, so these do too, and a reader of any chunk (the last one included) sees the model.
    """
    data = _as_dict(event)
    if data.get("type") == "message_start":
        message = _as_dict(data.get("message") or {})
        state["meta"] = {key: message[key] for key in ("id", "model") if message.get(key)}
    chunk = {**state["meta"], **_anthropic_stream_delta(data, state["blocks"])}
    if usage := _anthropic_stream_usage(state["usage"], data):
        chunk["usage"] = usage
        if data.get("type") == "message_start":
            chunk["type"] = "message_start"
    return chunk


def _anthropic_stream_end(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    # Report non-numeric metadata once, after the provider has finished updating it.
    usage = {key: value for key, value in state["usage"].items() if not _usage_count(value)}
    return [{**state["meta"], "choices": [{"delta": {}}], "usage": usage}] if usage else []


def _anthropic_stream(events: Any):
    state = _anthropic_stream_state()
    for event in events:
        yield _anthropic_stream_chunk(state, event)
    yield from _anthropic_stream_end(state)


async def _anthropic_astream(events: Any):
    state = _anthropic_stream_state()
    source = events if hasattr(events, "__aiter__") else _async_stream(events)
    async for event in source:
        yield _anthropic_stream_chunk(state, event)
    for chunk in _anthropic_stream_end(state):
        yield chunk


def _embedding_payload(runtime: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
    args = dict(runtime.get("args", {}))
    args.update(kwargs)
    return {"model": runtime["model"], **{k: v for k, v in args.items() if v is not None and k in EMBEDDING_PAYLOAD_KEYS}}


def _litellm(spec: LLMSpec):
    global _LITELLM_MODULE
    with _CLIENT_CACHE_LOCK:
        module = _LITELLM_MODULE
    if module is not None:
        return module
    gateway_cfg = spec.engine.root().get("gateways", {}).get("litellm", {})
    for key, value in (gateway_cfg.get("env_defaults", {}) or {}).items():
        os.environ.setdefault(str(key), str(value))
    try:
        import litellm
    except Exception as exc:
        raise ImportError("Install litellm or use gateway='portkey', 'openai', or 'bifrost'.") from exc
    litellm.drop_params = True
    with _CLIENT_CACHE_LOCK:
        if _LITELLM_MODULE is None:
            _LITELLM_MODULE = litellm
        return _LITELLM_MODULE


def _has_mock_request(kwargs: Dict[str, Any]) -> bool:
    return any(key in kwargs for key in ("mock", "mock_chat", "mock_embed", "mock_imagen"))


def _add_litellm_client_args(payload: Dict[str, Any], runtime: Dict[str, Any]) -> Dict[str, Any]:
    args = runtime.get("args", {}) or {}
    if args.get("timeout") is not None:
        payload["timeout"] = args["timeout"]
    if args.get("max_retries") is not None:
        payload["num_retries"] = args["max_retries"]
    return payload


def _openrouter_imagen(spec: LLMSpec, prompt: str, images: Iterable[LLMImage] = (), **kwargs: Any) -> Dict[str, Any]:
    client, runtime = _openai_client(spec, spec.gateway)
    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    content.extend(item.to_dict() for item in images)
    payload = _openai_payload(
        runtime,
        spec=spec,
        messages=[{"role": "user", "content": content}],
        modalities=kwargs.pop("modalities", ["image", "text"]),
        stream=False,
        **kwargs,
    )
    raw = response_dict(client.chat.completions.create(**payload))
    message = (raw.get("choices") or [{}])[0].get("message") or {}
    return {"data": message.get("images") or [], "usage": raw.get("usage") or {}, "raw": raw}


async def _openrouter_aimagen(spec: LLMSpec, prompt: str, images: Iterable[LLMImage] = (), **kwargs: Any) -> Dict[str, Any]:
    client, runtime = _openai_async_client(spec, spec.gateway)
    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    content.extend(item.to_dict() for item in images)
    payload = _openai_payload(
        runtime,
        spec=spec,
        messages=[{"role": "user", "content": content}],
        modalities=kwargs.pop("modalities", ["image", "text"]),
        stream=False,
        **kwargs,
    )
    raw = response_dict(await client.chat.completions.create(**payload))
    message = (raw.get("choices") or [{}])[0].get("message") or {}
    return {"data": message.get("images") or [], "usage": raw.get("usage") or {}, "raw": raw}


def _openrouter_imagen_litellm(spec: LLMSpec, prompt: str, images: Iterable[LLMImage] = (), **kwargs: Any) -> Dict[str, Any]:
    runtime = spec.engine.apply(spec)
    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    content.extend(item.to_dict() for item in images)
    payload = _openai_payload(
        runtime,
        spec=spec,
        messages=[{"role": "user", "content": content}],
        modalities=kwargs.pop("modalities", ["image", "text"]),
        stream=False,
        **kwargs,
    )
    if runtime.get("api_key"):
        payload["api_key"] = runtime["api_key"]
    if runtime.get("api_base"):
        payload["api_base"] = runtime["api_base"]
    _add_litellm_client_args(payload, runtime)
    raw = response_dict(_litellm(spec).completion(**payload))
    message = (raw.get("choices") or [{}])[0].get("message") or {}
    return {"data": message.get("images") or [], "usage": raw.get("usage") or {}, "raw": raw}


async def _openrouter_aimagen_litellm(spec: LLMSpec, prompt: str, images: Iterable[LLMImage] = (), **kwargs: Any) -> Dict[str, Any]:
    runtime = spec.engine.apply(spec)
    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    content.extend(item.to_dict() for item in images)
    payload = _openai_payload(
        runtime,
        spec=spec,
        messages=[{"role": "user", "content": content}],
        modalities=kwargs.pop("modalities", ["image", "text"]),
        stream=False,
        **kwargs,
    )
    if runtime.get("api_key"):
        payload["api_key"] = runtime["api_key"]
    if runtime.get("api_base"):
        payload["api_base"] = runtime["api_base"]
    _add_litellm_client_args(payload, runtime)
    fn = getattr(_litellm(spec), "acompletion", None)
    raw = response_dict(await fn(**payload) if fn else await _run_sync(_litellm(spec).completion, **payload))
    message = (raw.get("choices") or [{}])[0].get("message") or {}
    return {"data": message.get("images") or [], "usage": raw.get("usage") or {}, "raw": raw}


def response_dict(obj: Any) -> Dict[str, Any]:
    """Return provider response as a dictionary."""
    return _as_dict(obj)
