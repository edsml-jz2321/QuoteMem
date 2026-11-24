"""Entity-backed cache service for LLM requests."""

from __future__ import annotations

__all__ = [
    "LLMCache",
    "LLMEmbeddingEntry",
    "LLMImageEntry",
    "LLMTextEntry",
]

from contextvars import ContextVar
import os
import threading
from time import time
from typing import TYPE_CHECKING
import weakref

from .config import LLMSpec
from .image import LLMImage
from heavenbase.entity import Entity, field
from heavenbase.types import Boolean, Float, Integer, Json, LongText, ShortText
from heavenbase.utils import Any, Callable, Dict, Iterable, List, Mapping, Optional, dmerge, dumps_json, hash_id, resolve_interpolations, sha256hash
from heavenbase.utils.log import get_logger

if TYPE_CHECKING:
    from heavenbase.context import Context
    from heavenbase.utils.config import ConfigManager

TEXT = "text"
EMBEDDING = "embedding"
IMAGE = "image"
_logger = get_logger(__name__)
_HIT_WORKSPACE: ContextVar[tuple[LLMCache, Any] | None] = ContextVar("_HIT_WORKSPACE", default=None)

CONTROL_KEYS = {
    "api_key",
    "base_url",
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
    "enforce_non_stream_structured",
    "http_proxy",
    "https_proxy",
    "include",
    "max_retries",
    "mock_pause",
    "mock_seed",
    "max_workers",
    "no_proxy",
    "reduce",
    "stream",
    "stream_chat",
    "structured_stream",
    "timeout",
}

SORTED_LIST_KEYS = {"tools"}


class LLMTextEntry(Entity):
    """Cached normalized chat response."""

    identifier = "llm-text-cache"

    spec_key = field(ShortText).desc("Resolved LLM spec fingerprint")
    request_key = field(ShortText).desc("Semantic request fingerprint")
    messages = field(Json).default([]).desc("Normalized input messages")
    args = field(Json).default({}).desc("Semantic request arguments")
    response = field(Json).default({}).desc("Normalized LLMResponse payload")
    chunks = field(Json).default([]).desc("Normalized stream chunks")
    complete = field(Boolean).default(False).desc("Whether the response completed successfully")
    created_at = field(Float).default(0.0).desc("Creation timestamp")
    accessed_at = field(Float).default(0.0).desc("Last access timestamp")
    n_hits = field(Integer).default(0).desc("Cache hit count")
    expires_at = field(Float).optional().desc("Expiration timestamp")
    policy = field(ShortText).default("deterministic").desc("Write policy")
    size_bytes = field(Integer).default(0).desc("Approximate serialized row size")


class LLMEmbeddingEntry(Entity):
    """Cached single-input embedding response."""

    identifier = "llm-embedding-cache"

    spec_key = field(ShortText).desc("Resolved LLM spec fingerprint")
    input_hash = field(ShortText).desc("Input text fingerprint")
    input_text = field(LongText).default("").desc("Original input text")
    args = field(Json).default({}).desc("Semantic embedding arguments")
    embedding = field(Json).default([]).desc("Embedding vector")
    dim = field(Integer).default(0).desc("Embedding dimension")
    usage = field(Json).default({}).desc("Provider usage payload")
    raw = field(Json).default({}).desc("Provider raw item or response")
    created_at = field(Float).default(0.0).desc("Creation timestamp")
    accessed_at = field(Float).default(0.0).desc("Last access timestamp")
    n_hits = field(Integer).default(0).desc("Cache hit count")
    expires_at = field(Float).optional().desc("Expiration timestamp")
    size_bytes = field(Integer).default(0).desc("Approximate serialized row size")


class LLMImageEntry(Entity):
    """Cached image generation response."""

    identifier = "llm-image-cache"

    spec_key = field(ShortText).desc("Resolved LLM spec fingerprint")
    prompt = field(LongText).default("").desc("Image prompt")
    image_inputs = field(Json).default([]).desc("Reference image digests")
    args = field(Json).default({}).desc("Semantic image arguments")
    images = field(Json).default([]).desc("Normalized generated image payloads")
    usage = field(Json).default({}).desc("Provider usage payload")
    raw = field(Json).default({}).desc("Provider raw response")
    created_at = field(Float).default(0.0).desc("Creation timestamp")
    accessed_at = field(Float).default(0.0).desc("Last access timestamp")
    n_hits = field(Integer).default(0).desc("Cache hit count")
    expires_at = field(Float).optional().desc("Expiration timestamp")
    size_bytes = field(Integer).default(0).desc("Approximate serialized row size")


def _cache_root(manager: "ConfigManager") -> Dict[str, Any]:
    default = ((manager.load_default().get("heavenbase", {}) or {}).get("llm", {}) or {}).get("cache", {}) or {}
    runtime = manager.get("heavenbase.llm.cache", default={}) or {}
    cfg = dmerge([default, runtime])
    if "backend" in runtime:
        cfg["backend"] = runtime["backend"]
    return resolve_interpolations(cfg)


def _storage(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the storage keys of one cache configuration, which name its store."""
    return {key: config[key] for key in ("workspace", "backend", "backends") if key in config}


def _looks_image_part(value: Mapping[str, Any]) -> bool:
    """Return whether a mapping looks like an LLM image content part."""
    kind = value.get("type")
    if kind in {"image_url", "image", "input_image"}:
        return True
    return any(key in value for key in ("image_url", "b64_json")) or ("data" in value and any(key in value for key in ("media_type", "format", "image")))


def _canon_image_part(value: Mapping[str, Any]) -> Dict[str, Any]:
    """Digest one image payload while retaining request-affecting metadata."""
    image = LLMImage.from_any(dict(value))
    out = {"image_digest": _image_digest(image)}
    for name in sorted(value):
        item = value[name]
        if name in {"image", "data", "b64_json", "url"}:
            continue
        if name == "image_url":
            if isinstance(item, Mapping):
                metadata = {str(key): _canon(item[key], key=str(key)) for key in sorted(item) if key != "url"}
                if metadata:
                    out[name] = metadata
            continue
        out[str(name)] = _canon(item, key=str(name))
    return out


def _canon(value: Any, key: Optional[str] = None) -> Any:
    if value is None or isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, str):
        if value.startswith("data:image"):
            return {"data_url_hash": sha256hash(value)}
        return value
    if isinstance(value, LLMImage):
        return _image_digest(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"bytes_hash": sha256hash(bytes(value))}
    if callable(value):
        return {"callable": f"{getattr(value, '__module__', '')}.{getattr(value, '__qualname__', getattr(value, '__name__', 'callable'))}"}
    if hasattr(value, "model_dump"):
        return _canon(value.model_dump(), key=key)
    if hasattr(value, "to_dict"):
        return _canon(value.to_dict(), key=key)
    if hasattr(value, "dict"):
        return _canon(value.dict(), key=key)
    if isinstance(value, Mapping):
        if _looks_image_part(value):
            try:
                return _canon_image_part(value)
            except (OSError, TypeError, ValueError):
                pass
        return {str(name): _canon(value[name], key=str(name)) for name in sorted(value)}
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, bytearray)):
        items = [_canon(item, key=key) for item in value]
        return sorted(items, key=_canon_str) if key in SORTED_LIST_KEYS else items
    return repr(value)


def _canon_str(value: Any) -> str:
    return dumps_json(value, sort_keys=True, compact=True)


def _semantic_args(args: Mapping[str, Any]) -> Dict[str, Any]:
    """Return request arguments that can affect model output."""
    return {key: _canon(value, key=key) for key, value in sorted(dict(args or {}).items()) if key not in CONTROL_KEYS and value is not None}


def _spec_key(spec: LLMSpec, namespace: str) -> str:
    """Return a redacted resolved LLM spec key for one cache namespace."""
    runtime = spec.engine.apply(spec)
    return sha256hash(
        {
            "namespace": namespace,
            "semantic_spec": spec.hash_key(),
            "provider": spec.provider,
            "provider_backend": spec.provider_backend(),
            "model": spec.model,
            "model_id": spec.model_id,
            "gateway": spec.gateway,
            "runtime_model": runtime.get("model"),
            "base_url": runtime.get("base_url") or runtime.get("api_base"),
            "headers_hash": sha256hash(runtime.get("headers", {}) or {}),
            "api_key_hash": sha256hash(runtime.get("api_key")) if runtime.get("api_key") else None,
        }
    )


def _cache_id(namespace: str, spec_hash: str, req_hash: str) -> str:
    return hash_id(namespace, spec_hash, req_hash)


def _image_digest(image: LLMImage) -> Dict[str, Any]:
    """Return a cache-safe reference image digest without fetching remote URLs."""
    if image.data is not None:
        return {"data_hash": sha256hash({"data": image.data, "media_type": image.media_type}), "format": image.format, "media_type": image.media_type}
    return {"url": image.url or "", "format": image.format, "media_type": image.media_type}


class LLMCache:
    """Own semantic request keys and entity-backed LLM cache lifecycle.

    Args:
        config (Mapping[str, Any] | None): Cache configuration merged over
            `heavenbase.llm.cache` defaults.
        config_manager (ConfigManager | None): Configuration authority and owning Context.
            Omission selects the default Context's config.
    """

    def __init__(self, config: Optional[Mapping[str, Any]] = None, *, config_manager: "ConfigManager | None" = None) -> None:
        """Initialize this cache service from configuration.

        Returns:
            None: This function does not return a value.
        """
        if config_manager is None:
            from heavenbase.context import default_context

            config_manager = default_context().config
        cfg = dmerge([_cache_root(config_manager), dict(config or {})])
        if config and "backend" in config:
            cfg["backend"] = config["backend"]
        self.config = cfg
        self._config_manager_ref = weakref.ref(config_manager)
        from heavenbase.context.runtime import _config_context

        context = _config_context(config_manager)
        self._context_ref = None if context is None else weakref.ref(context)
        self._store: Any = None
        # Whether this cache is its authority's cache service, built by `current()`: its storage then belongs to the
        # Context, which keeps one store per storage configuration for every such cache, collected ones included.
        # Any other cache owns its store, which is released with the cache.
        self._service = False
        self._lock = threading.RLock()
        with _LLM_CACHE_LOCK:
            _LLM_CACHE_INSTANCES[id(self)] = self

    @property
    def config_manager(self) -> "ConfigManager":
        """Return the live configuration authority for this cache."""
        manager = self._config_manager_ref()
        if manager is None:
            raise RuntimeError("LLM cache configuration authority is no longer available")
        return manager

    @property
    def context(self) -> "Context | None":
        """Return the live Context that owns this cache, when one exists."""
        return None if self._context_ref is None else self._context_ref()

    @classmethod
    def from_dict(cls, data: Optional[Mapping[str, Any]] = None, *, config_manager: "ConfigManager | None" = None, **values: Any) -> "LLMCache":
        """Build a cache service from configuration values.

        Args:
            data (Mapping[str, Any] | None): Base cache configuration.
            config_manager (ConfigManager | None): Configuration authority.
                Omission selects the default Context's config.
            **values (Any): Configuration values that override `data`.

        Returns:
            LLMCache: Unopened cache service.
        """
        cfg = dict(data or {})
        cfg.update(values)
        return cls(cfg, config_manager=config_manager)

    def to_dict(self) -> Dict[str, Any]:
        """Return this cache service configuration.

        Args:
            None.

        Returns:
            Dict[str, Any]: Independent configuration dictionary.
        """
        return dict(self.config)

    @classmethod
    def current(cls, *, config_manager: "ConfigManager | None" = None) -> "LLMCache":
        """Return the cache service owned by one configuration authority.

        Args:
            config_manager (ConfigManager | None): Configuration authority.
                Omission selects the default Context's config.

        Returns:
            LLMCache: Lazily constructed authority-local cache.
        """
        from heavenbase.context import default_context
        from heavenbase.context.runtime import _config_context, _default_context_instance

        if config_manager is None:
            config_manager = default_context().config
        else:
            # An unowned CM_HVNB selects the lazy default first, so it shares
            # the default cache instead of keying a second one.
            _config_context(config_manager)
        with _LLM_CACHE_LOCK:
            global _DEFAULT_LLM_CACHE

            # One map finds every manager's cache, so a Context installed as the
            # default after its first cached call keeps that cache and its store.
            reference = _LLM_CACHES.get(config_manager)
            found = None if reference is None else reference()
            if found is None:
                found = cls(config_manager=config_manager)
                found._service = True
                _LLM_CACHES[config_manager] = weakref.ref(found)
            default = _default_context_instance()
            if default is not None and config_manager is default.config:
                # The default Context lives for the process; keep its cache alive too.
                _DEFAULT_LLM_CACHE = found
            return found

    @classmethod
    def reset(cls, *, drop: bool = False, config_manager: "ConfigManager | None" = None) -> None:
        """Reset one authority's cache service or every process cache.

        The cache service's storage belongs to its Context, which keeps one
        store per storage configuration for every cache that ``current()``
        returned, including caches whose objects were garbage-collected. A
        reset retires those stores, so the next cached call opens new ones.
        Caches built from a configuration (``LLMCache(...)``, ``from_dict``)
        own their stores and are not reset. A drop removes each store's
        storage on every route it opened, and also the storage that the
        authority's ``heavenbase.llm.cache`` configuration names when no store
        on it is open yet, such as in a new process or a new Context over the
        same root. That configured storage is opened only when it may exist:
        storage whose routes are all local files that do not exist is
        skipped, so a Context that never created it is not initialized for
        the drop, while storage on a server database is opened. The hidden
        stores of any Context on a dropped database reopen on fresh storage
        at their next use. A closed Context owns no storage, so a drop leaves
        its rows in place. A Context that a close or reset has marked counts as
        closed for a drop: configured storage that is not open yet is skipped
        and the reset returns normally, and a drop of open storage raises
        ``RuntimeError``. If that close or reset gives up, the Context is open
        again and still serves the rows.

        Args:
            drop (bool): Whether to drop each selected store's storage first.
            config_manager (ConfigManager | None): Exact configuration
                authority to reset. An unowned ``CM_HVNB`` selects the
                default Context, as ``current()`` does, except in explicit
                Context mode, where it selects none. Omission preserves the
                process-wide maintenance reset. Without ``drop`` it retires
                the stores of the Contexts that hold one and touches no other
                Context. A drop reads the cache configuration of every open
                root Context of the process, the default Context once it
                exists included, whether or not it used the cache. Closed
                Contexts are skipped, and no default Context is built for the
                reset. A Context discarded without ``close()`` stays open
                until it is garbage-collected, so a drop still reaches it;
                close Contexts you no longer use.

        Returns:
            None: This method does not return a value.

        Raises:
            Exception: The failure, unchanged, when ``drop`` is true and one
                configured store of an open selected Context fails to open, or
                one selected store fails to drop. Every other selected store
                is still opened, dropped and retired first.
            RuntimeError: One combined error, chained from the first failure,
                when several stores fail to open or drop.
        """
        from heavenbase import _initialization
        from heavenbase.context.runtime import (
            _config_context,
            _contexts_with_hidden_stores,
            _drop_hidden_stores,
            _local_storage_absent,
            _raise_drop_failures,
            _root_contexts,
        )

        with _LLM_CACHE_LOCK:
            global _DEFAULT_LLM_CACHE

            if config_manager is None:
                services = [cache for reference in _LLM_CACHES.values() if (cache := reference()) is not None]
                _DEFAULT_LLM_CACHE = None
                _LLM_CACHES.clear()
            else:
                reference = _LLM_CACHES.pop(config_manager, None)
                services = [] if reference is None or (found := reference()) is None else [found]
                pinned = _DEFAULT_LLM_CACHE
                if pinned is not None and pinned._config_manager_ref() is config_manager:
                    _DEFAULT_LLM_CACHE = None
            caches = tuple(_LLM_CACHE_INSTANCES.values())
        # Each selected root Context, with the sources of the storage configurations its cache services name:
        # a cache configuration, or a ConfigManager whose `heavenbase.llm.cache` is read only once the Context
        # is known to be open. A drop opens them first, so it also removes storage that no store of this
        # process has opened yet.
        named: dict[int, tuple[Any, list[Any]]] = {}

        def select(context: "Context | None", source: Any) -> None:
            if context is None:
                return
            while context.parent is not None:
                context = context.parent
            sources = named.setdefault(id(context), (context, []))[1]
            if drop and source is not None:
                sources.append(source)

        # The Contexts that hold a cache store, so the reset has something to retire there.
        holding = _contexts_with_hidden_stores(_STORE_KIND)
        holders = {id(context) for context in holding}
        if config_manager is None:
            # A drop also reaches every other open root Context, for the storage its configuration names.
            for context in _root_contexts() if drop else holding:
                select(context, context.config)
        else:
            owner = getattr(config_manager, "_context", None)
            if owner is None and drop and not _initialization.explicit_context_required:
                # An unowned CM_HVNB selects the default Context, as `current()` does; explicit mode has none.
                owner = _config_context(config_manager)
            select(owner, config_manager)
        for service in services:
            select(service.context, service.config)
        # The caches using each store, so each cache is visited once however many Contexts are reset.
        users: dict[int, list[LLMCache]] = {}
        for cache in caches:
            if (used := cache._store) is not None:
                users.setdefault(id(used), []).append(cache)
        # Retiring a store takes the Context lock, which a cached call inside a
        # workspace write already holds while it waits for _LLM_CACHE_LOCK, so
        # that lock is not held here.
        errors: list[Exception] = []
        for context, sources in named.values():
            # The configured storage that may hold rows. A closed Context owns no storage, and its
            # configuration can no longer be read. Local storage that does not exist is not opened, so a
            # Context that never created it is not initialized for the drop.
            storages: list[Dict[str, Any]] = []
            if drop and _is_open(context):
                for source in sources:
                    try:
                        storage = _storage(source if isinstance(source, Mapping) else _cache_root(source))
                    except Exception as exc:
                        errors.append(exc)
                        continue
                    if not _local_storage_absent(storage, context.config):
                        storages.append(storage)
            if not storages and id(context) not in holders:
                continue
            with context._lock:
                if _is_open(context):
                    for storage in storages:
                        try:
                            context._hidden_store(storage, kind=_STORE_KIND, entities=_ENTITIES).workspace
                        except Exception as exc:
                            errors.append(exc)
                stores = context._pop_hidden_stores(_STORE_KIND)
                for store in stores:
                    for cache in users.get(id(store), ()):
                        with cache._lock:
                            if cache._store is store:
                                cache._store = None
                if drop:
                    try:
                        _drop_hidden_stores(stores)
                    except Exception as exc:
                        errors.append(exc)
                else:
                    for store in stores:
                        if (workspace := store._opened()) is not None:
                            _clear_file_pools(workspace)
                        store.close()
        _raise_drop_failures(errors)

    def text_key(self, spec: LLMSpec, messages: List[Dict[str, Any]], args: Mapping[str, Any]) -> Dict[str, Any]:
        """Build a deterministic text request key.

        Args:
            spec (LLMSpec): Resolved LLM provider specification.
            messages (List[Dict[str, Any]]): Normalized chat messages.
            args (Mapping[str, Any]): Request arguments before control-key filtering.

        Returns:
            Dict[str, Any]: Identity and semantic arguments for an `LLMTextEntry` row.
        """
        semantic = _semantic_args(args)
        spec_hash = _spec_key(spec, TEXT)
        req_hash = sha256hash({"messages": _canon(messages), "args": semantic})
        return {"object_id": _cache_id(TEXT, spec_hash, req_hash), "spec_key": spec_hash, "request_key": req_hash, "args": semantic}

    def embedding_key(self, spec: LLMSpec, input_text: str, args: Mapping[str, Any]) -> Dict[str, Any]:
        """Build a deterministic embedding request key.

        Args:
            spec (LLMSpec): Resolved LLM provider specification.
            input_text (str): One embedding input.
            args (Mapping[str, Any]): Request arguments before control-key filtering.

        Returns:
            Dict[str, Any]: Identity and semantic arguments for an `LLMEmbeddingEntry` row.
        """
        semantic = _semantic_args(args)
        spec_hash = _spec_key(spec, EMBEDDING)
        input_hash = sha256hash(input_text)
        req_hash = sha256hash({"input_hash": input_hash, "args": semantic})
        return {"object_id": _cache_id(EMBEDDING, spec_hash, req_hash), "spec_key": spec_hash, "input_hash": input_hash, "args": semantic}

    def image_key(self, spec: LLMSpec, prompt: str, refs: List[LLMImage], args: Mapping[str, Any]) -> Dict[str, Any]:
        """Build a deterministic image request key.

        Args:
            spec (LLMSpec): Resolved LLM provider specification.
            prompt (str): Image generation prompt.
            refs (List[LLMImage]): Normalized reference images.
            args (Mapping[str, Any]): Request arguments before control-key filtering.

        Returns:
            Dict[str, Any]: Identity, reference digests, and semantic arguments for an `LLMImageEntry` row.
        """
        semantic = _semantic_args(args)
        spec_hash = _spec_key(spec, IMAGE)
        image_inputs = [_image_digest(ref) for ref in refs]
        req_hash = sha256hash({"prompt": prompt, "image_inputs": image_inputs, "args": semantic})
        return {
            "object_id": _cache_id(IMAGE, spec_hash, req_hash),
            "spec_key": spec_hash,
            "image_inputs": image_inputs,
            "request_key": req_hash,
            "args": semantic,
        }

    def dump_images(self, images: Any) -> List[Dict[str, Any]]:
        """Serialize generated images for an image cache row.

        Args:
            images (Any): One `LLMImage` or a list of images.

        Returns:
            List[Dict[str, Any]]: JSON-compatible image payloads. Non-image values are omitted.
        """
        items = images if isinstance(images, list) else [images]
        return [
            {"data": image.data, "url": image.url, "media_type": image.media_type, "format": image.format, "revised_prompt": image.revised_prompt}
            for image in items
            if isinstance(image, LLMImage)
        ]

    def load_images(self, payloads: Iterable[Mapping[str, Any]]) -> List[LLMImage]:
        """Hydrate images from an image cache row.

        Args:
            payloads (Iterable[Mapping[str, Any]]): Stored JSON-compatible image payloads.

        Returns:
            List[LLMImage]: Normalized image values in payload order.
        """
        return [
            LLMImage(
                data=item.get("data"),
                url=item.get("url"),
                media_type=item.get("media_type") or "image/png",
                format=item.get("format") or "png",
                revised_prompt=item.get("revised_prompt"),
                _url_timeout=None,
            )._bound(self.config_manager)
            for item in payloads
        ]

    @property
    def workspace(self) -> Any:
        """Lazily created workspace that stores cache entries.

        A drop of a database this workspace uses, from any Context, detaches it,
        and the next access opens a new one on fresh storage.
        """
        with self._lock:
            context = self.context
            if context is None:
                raise RuntimeError(
                    "LLM cache persistence requires a Context-bound ConfigManager; "
                    "construct LLM(context=...), bind the manager to a Context, or disable caching"
                )
            store = self._store
        workspace = None if store is None else store._attached()
        if workspace is not None:
            return workspace
        # Opening the store takes the Context lock, and a Context-lock holder
        # (a write whose hook calls a cached LLM) may reach this property, so
        # the Context lock is taken before this cache's. The store opens
        # outside this cache's lock, which a drop's reset may take meanwhile.
        with context._lock:
            context._need_open()
            with self._lock:
                store = self._store
                if store is None or store.retired:
                    storage = _storage(self.config)
                    if self._service:
                        store = context._hidden_store(storage, kind=_STORE_KIND, entities=_ENTITIES)
                    else:
                        store = context._new_hidden_store(storage, entities=_ENTITIES)
                    self._store = store
        # The store opens outside this hold of the Context lock, so a drop that
        # it waits for does not hold up the Context's other writes.
        try:
            return store.workspace
        except RuntimeError:
            if not store.retired:
                raise
        # A reset retired the store before it opened: choose a store again.
        return self.workspace

    def ns_cfg(self, namespace: str) -> Dict[str, Any]:
        """Return configuration for one cache namespace.

        Args:
            namespace (str): Cache namespace such as `text`, `embedding`, or `image`.

        Returns:
            Dict[str, Any]: Independent namespace configuration dictionary.
        """
        return dict((self.config.get("namespaces") or {}).get(namespace) or {})

    def enabled(self, namespace: str, option: Any = None) -> bool:
        """Return whether one request should use a cache namespace.

        Args:
            namespace (str): Cache namespace.
            option (Any): Request override. `False` disables caching and a truthy value enables it subject to namespace configuration.

        Returns:
            bool: Whether cache reads and eligible writes are enabled.
        """
        if option is False:
            return False
        if not (bool(option) or bool(self.config.get("enabled", False))):
            return False
        return bool(self.ns_cfg(namespace).get("enabled", True))

    def policy(self, namespace: str, args: Mapping[str, Any], policy: Optional[str] = None) -> Optional[str]:
        """Resolve the write policy for one request.

        Args:
            namespace (str): Cache namespace.
            args (Mapping[str, Any]): Semantic provider arguments.
            policy (str | None): Optional request policy override.

        Returns:
            str | None: Effective policy, or `None` when the request must bypass the cache.
        """
        if self._stochastic_without_seed(args):
            return None
        name = str(policy or self.config.get("policy") or "deterministic")
        if name in {"off", "none", "false"}:
            return None
        if name in {"force", "always"}:
            return name
        if namespace == EMBEDDING:
            return name
        return name if self._deterministic(namespace, args) else None

    def _deterministic(self, namespace: str, args: Mapping[str, Any]) -> bool:
        temp = args.get("temperature")
        top_p = args.get("top_p")
        top_k = args.get("top_k")
        seed = args.get("seed")
        if temp not in (None, 0, 0.0) and seed is None:
            return False
        if top_p not in (None, 0, 0.0, 1, 1.0):
            return False
        if top_k not in (None, 0, 1):
            return False
        return namespace != IMAGE or args.get("seed") is not None

    def _stochastic_without_seed(self, args: Mapping[str, Any]) -> bool:
        temp = args.get("temperature")
        return temp not in (None, 0, 0.0) and args.get("seed") is None

    def expires_at(self, namespace: str, now: float) -> Optional[float]:
        """Return the expiration timestamp for a new row.

        Args:
            namespace (str): Cache namespace whose TTL applies.
            now (float): Row creation time as Unix seconds.

        Returns:
            float | None: Expiration time as Unix seconds, or `None` when no TTL applies.
        """
        ttl = self.ns_cfg(namespace).get("ttl_seconds")
        return now + float(ttl) if ttl else None

    def expired(self, row: Mapping[str, Any], now: Optional[float] = None) -> bool:
        """Return whether a cache row has expired.

        Args:
            row (Mapping[str, Any]): Cache row containing an optional `expires_at` value.
            now (float | None): Comparison time as Unix seconds. Defaults to the current time.

        Returns:
            bool: Whether the row's expiration time has passed.
        """
        expires_at = row.get("expires_at")
        return expires_at is not None and float(expires_at) <= (time() if now is None else now)

    def get_text(self, key: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
        """Return a completed text cache row and record the hit.

        Args:
            key (Mapping[str, Any]): Key payload returned by `text_key`.

        Returns:
            Dict[str, Any] | None: Cached row, or `None` when absent, incomplete, or expired.
        """
        workspace = self.workspace
        row = workspace.get(key["object_id"], entity=LLMTextEntry)
        if not row or not row.get("complete") or self.expired(row):
            return None
        self._record_hits(workspace, self.touch, LLMTextEntry, row)
        return row

    def set_text(
        self, key: Mapping[str, Any], messages: List[Dict[str, Any]], response: Mapping[str, Any], chunks: List[Mapping[str, Any]], policy: str
    ) -> None:
        """Write a completed text cache row.

        Image payloads in ``messages`` are digested before persistence so multimodal
        chat requests do not store raw base64/data-URL bytes. Rows that exceed the
        namespace ``max_bytes`` budget are skipped. Persistence failures are logged and
        swallowed so optional cache writes cannot fail an otherwise successful chat.

        Args:
            key (Mapping[str, Any]): Key payload returned by `text_key`.
            messages (List[Dict[str, Any]]): Normalized input messages.
            response (Mapping[str, Any]): Normalized response payload.
            chunks (List[Mapping[str, Any]]): Normalized stream chunks.
            policy (str): Policy that authorized the write.

        Returns:
            None: This method does not return a value.
        """
        now = time()
        row = {
            "object_id": key["object_id"],
            "spec_key": key["spec_key"],
            "request_key": key["request_key"],
            "messages": _canon(messages),
            "args": key["args"],
            "response": dict(response),
            "chunks": list(chunks),
            "complete": True,
            "created_at": now,
            "accessed_at": now,
            "n_hits": 0,
            "expires_at": self.expires_at(TEXT, now),
            "policy": policy,
        }
        if row["expires_at"] is None:
            row.pop("expires_at")
        row["size_bytes"] = self.size(row)
        max_bytes = self.ns_cfg(TEXT).get("max_bytes")
        if max_bytes is not None and int(max_bytes) > 0 and row["size_bytes"] > int(max_bytes):
            _logger.warning(
                "Skipping text cache write for %s: size %s exceeds max_bytes %s",
                key["object_id"],
                row["size_bytes"],
                int(max_bytes),
            )
            return
        try:
            self.workspace.upsert(LLMTextEntry, row)
        except Exception as exc:
            _logger.warning("Text cache persistence failed for %s (%s)", key["object_id"], type(exc).__name__)

    def get_embedding(self, key: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
        """Return a single-input embedding cache row and record the hit.

        Args:
            key (Mapping[str, Any]): Key payload returned by `embedding_key`.

        Returns:
            Dict[str, Any] | None: Cached row, or `None` when absent or expired.
        """
        return self.get_embeddings({"value": key}).get("value")

    def get_embeddings(self, keys: Mapping[str, Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Return embedding cache rows for many inputs and record hits in one batch.

        Args:
            keys (Mapping[str, Mapping[str, Any]]): Input text to embedding-key payload mapping.

        Returns:
            Dict[str, Dict[str, Any]]: Unexpired cached rows keyed by input text.
        """
        items = list(keys.items())
        workspace = self.workspace
        rows = workspace.get_many([key["object_id"] for _, key in items], entity=LLMEmbeddingEntry)
        out = {input_text: row for (input_text, _), row in zip(items, rows) if row and not self.expired(row)}
        if out and self.ns_cfg(EMBEDDING).get("touch_hits"):
            self._record_hits(workspace, self.touch_many, LLMEmbeddingEntry, out.values())
        return out

    def set_embedding(self, key: Mapping[str, Any], input_text: str, embedding: List[Any], dim: int, usage: Mapping[str, Any], raw: Any) -> None:
        """Write a single-input embedding cache row.

        Args:
            key (Mapping[str, Any]): Key payload returned by `embedding_key`.
            input_text (str): Original embedding input.
            embedding (List[Any]): Provider embedding vector.
            dim (int): Vector dimension.
            usage (Mapping[str, Any]): Provider usage payload.
            raw (Any): Provider response payload.

        Returns:
            None: This method does not return a value.
        """
        self.set_embeddings([{"key": key, "input_text": input_text, "embedding": embedding, "dim": dim, "usage": usage, "raw": raw}])

    def set_embeddings(self, items: Iterable[Mapping[str, Any]]) -> None:
        """Write multiple embedding cache rows in one batch.

        Args:
            items (Iterable[Mapping[str, Any]]): Row inputs containing key, text, vector, dimension, usage, and raw response values.

        Returns:
            None: This method does not return a value.
        """
        now = time()
        rows = []
        for item in items:
            key = item["key"]
            row = {
                "object_id": key["object_id"],
                "spec_key": key["spec_key"],
                "input_hash": key["input_hash"],
                "input_text": item["input_text"],
                "args": key["args"],
                "embedding": item["embedding"],
                "dim": item["dim"],
                "usage": dict(item.get("usage") or {}),
                "raw": item.get("raw"),
                "created_at": now,
                "accessed_at": now,
                "n_hits": 0,
                "expires_at": self.expires_at(EMBEDDING, now),
            }
            if row["expires_at"] is None:
                row.pop("expires_at")
            row["size_bytes"] = self.size(row)
            rows.append(row)
        if rows:
            self.workspace.upsert_many(LLMEmbeddingEntry, rows)

    def get_image(self, key: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
        """Return an image cache row and record the hit.

        Args:
            key (Mapping[str, Any]): Key payload returned by `image_key`.

        Returns:
            Dict[str, Any] | None: Cached row, or `None` when absent or expired.
        """
        workspace = self.workspace
        row = workspace.get(key["object_id"], entity=LLMImageEntry)
        if not row or self.expired(row):
            return None
        self._record_hits(workspace, self.touch, LLMImageEntry, row)
        return row

    def set_image(self, key: Mapping[str, Any], prompt: str, images: List[Mapping[str, Any]], usage: Mapping[str, Any], raw: Any) -> None:
        """Write an image generation cache row.

        Args:
            key (Mapping[str, Any]): Key payload returned by `image_key`.
            prompt (str): Image generation prompt.
            images (List[Mapping[str, Any]]): Serialized generated images.
            usage (Mapping[str, Any]): Provider usage payload.
            raw (Any): Provider response payload.

        Returns:
            None: This method does not return a value.
        """
        now = time()
        row = {
            "object_id": key["object_id"],
            "spec_key": key["spec_key"],
            "prompt": prompt,
            "image_inputs": key["image_inputs"],
            "args": key["args"],
            "images": list(images),
            "usage": dict(usage or {}),
            "raw": raw,
            "created_at": now,
            "accessed_at": now,
            "n_hits": 0,
            "expires_at": self.expires_at(IMAGE, now),
        }
        if row["expires_at"] is None:
            row.pop("expires_at")
        row["size_bytes"] = self.size(row)
        self.workspace.upsert(LLMImageEntry, row)

    def touch(self, entity: type, row: Mapping[str, Any]) -> None:
        """Update access metadata for one cache row.

        Args:
            entity (type): Cache entity class that owns the row.
            row (Mapping[str, Any]): Existing cache row.

        Returns:
            None: This method does not return a value.
        """
        self.touch_many(entity, [row])

    def touch_many(self, entity: type, rows: Iterable[Mapping[str, Any]]) -> None:
        """Update access metadata for multiple cache rows.

        Args:
            entity (type): Cache entity class that owns the rows.
            rows (Iterable[Mapping[str, Any]]): Existing cache rows.

        Returns:
            None: This method does not return a value.
        """
        hits = list(rows)
        if not hits:
            return
        origin = _HIT_WORKSPACE.get()
        workspace = origin[1] if origin is not None and origin[0] is self else self.workspace
        # A hit is recorded only on the workspace that read it, while that is
        # still this cache's open workspace; a drop that detaches the workspace
        # waits for an admitted hit before it removes the storage.
        with workspace._lock:
            store = self._store
            if store is None:
                return
            with store._admitted_on(workspace) as admitted:
                if admitted:
                    self._touch_rows(workspace, entity, hits)

    def _touch_rows(self, workspace: Any, entity: type, hits: List[Mapping[str, Any]]) -> None:
        """Record admitted hits on the workspace that read them."""
        from heavenbase.backends.sql.postgres import PostgresBackend
        from heavenbase.backends.sql.sqlite import SQLiteBackend

        now = time()
        if entity in (LLMTextEntry, LLMEmbeddingEntry, LLMImageEntry):
            entity_id = entity.schema().entity_id
            routes = workspace._row_store.routed_backend_names(entity_id)
            if len(routes) == 1:
                backend = workspace.backends.get(routes[0])
                if isinstance(backend, (SQLiteBackend, PostgresBackend)):
                    self._touch_sql_rows(backend, entity_id, hits, now)
                    return

        ids = [hit["object_id"] for hit in hits]
        current = workspace.get_many(ids, entity=entity, columns=["created_at", "n_hits"])
        patches = [
            {"object_id": object_id, "accessed_at": now, "n_hits": int(live.get("n_hits") or 0) + 1}
            for object_id, hit, live in zip(ids, hits, current)
            if live is not None and hit.get("created_at", live.get("created_at")) == live.get("created_at")
        ]
        if patches:
            workspace.set_many(patches, entity=entity)

    @staticmethod
    def _touch_sql_rows(backend: Any, entity_id: str, hits: List[Mapping[str, Any]], now: float) -> None:
        """Update access metadata in place, matching the generation read by the hit."""
        from sqlalchemy import Boolean, bindparam, func, or_, update

        with backend._lock:
            backend._need_db()
            db = backend.db
            backend._pool_slots(db)
            table = backend.tables[entity_id]
            stmt = backend._construct(
                entity_id,
                "llm_cache_touch",
                lambda: update(table)
                .where(
                    table.c.object_id == bindparam("hit_id"),
                    or_(bindparam("any_generation", type_=Boolean()), table.c.created_at == bindparam("hit_created_at")),
                )
                .values(accessed_at=bindparam("hit_time"), n_hits=func.coalesce(table.c.n_hits, 0) + 1),
            )
            params = [
                {"hit_id": hit["object_id"], "hit_created_at": hit.get("created_at"), "any_generation": "created_at" not in hit, "hit_time": now}
                for hit in hits
            ]
            db.orm_execute(stmt, params, readonly=False)

    def _record_hits(self, workspace: Any, touch: Callable[..., None], entity: type, rows: Any) -> None:
        """Keep hit updates on the workspace that read the rows across a reset."""
        token = _HIT_WORKSPACE.set((self, workspace))
        try:
            touch(entity, rows)
        finally:
            _HIT_WORKSPACE.reset(token)

    def size(self, row: Mapping[str, Any]) -> int:
        """Return an approximate serialized row size.

        Args:
            row (Mapping[str, Any]): Cache row to serialize.

        Returns:
            int: UTF-8 byte length of the compact JSON representation.
        """
        return len(dumps_json(row, sort_keys=True, compact=True).encode("utf-8"))


_LLM_CACHES: weakref.WeakKeyDictionary["ConfigManager", weakref.ReferenceType[LLMCache]] = weakref.WeakKeyDictionary()
_LLM_CACHE_LOCK = threading.RLock()
_DEFAULT_LLM_CACHE: LLMCache | None = None
# Every live cache, registered or not: an LLM keeps the cache it was built with
# after a reset unregisters it, and its store reopens on the next cached call.
_LLM_CACHE_INSTANCES: weakref.WeakValueDictionary[int, LLMCache] = weakref.WeakValueDictionary()
# Kind of the Context-held hidden stores of cache services (see `LLMCache.reset`).
_STORE_KIND = "llm-cache"
# Entities registered in every workspace a cache store opens.
_ENTITIES = (LLMTextEntry, LLMEmbeddingEntry, LLMImageEntry)


def _reset_cache_locks_after_fork() -> None:
    """Give the child its own cache locks while preserving workspace references."""
    global _LLM_CACHE_LOCK
    _LLM_CACHE_LOCK = threading.RLock()
    for cache in list(_LLM_CACHE_INSTANCES.values()):
        cache._lock = threading.RLock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_cache_locks_after_fork)


def _clear_file_pools(workspace: Any) -> None:
    """Release idle local-file connections when a reset forgets the workspace."""
    from heavenbase.database.database import Database
    from heavenbase.database.registry import DatabaseEngineRegistry

    for _, backend in workspace.backends.items():
        db = getattr(backend, "db", None)
        if (
            isinstance(db, Database)
            and DatabaseEngineRegistry._local_path(db.spec) is not None
            and DatabaseEngineRegistry.has_engine(db.spec, authority_id=db._authority_id, resource_id=db._resource_id)
        ):
            # Another workspace using the same key can still obtain connections.
            db.engine.dispose()


def _is_open(context: Any) -> bool:
    """Return whether a Context can open storage; a closed Context owns none to drop."""
    try:
        context._need_open()
    except RuntimeError:
        return False
    return True
