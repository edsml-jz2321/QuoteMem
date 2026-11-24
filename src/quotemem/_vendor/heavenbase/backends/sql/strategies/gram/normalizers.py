"""Sparse-GRAM normalizers and normalizer registry."""

from __future__ import annotations

__all__ = [
    "NormalizedText",
    "normalize_many",
    "normalize_text",
    "register_gram_normalizer",
    "resolve_gram_normalizer",
]

import importlib
import threading
import unicodedata
from weakref import WeakKeyDictionary

from heavenbase.utils import Any, Callable, Dict, Iterable, Sequence
from heavenbase.utils.hash import sha256hash
from heavenbase.utils.text_match import NormalizedText

from .....modules import ModuleResolver, module_resolver
from .config import GramConfig
from .registry import register_gram_callable, resolve_gram_callable

Normalizer = Callable[[str, GramConfig], "NormalizedText | str"]

_SPACY_MODELS: Dict[tuple[str, tuple[str, ...]], Any] = {}
_CACHES: WeakKeyDictionary[Any, Dict[tuple[int, str, str, str], "NormalizedText"]] = WeakKeyDictionary()
_SPACY_LOCK = threading.RLock()
_CACHE_LOCK = threading.RLock()


def register_gram_normalizer(
    name: str,
    fn: Normalizer,
    *,
    aliases: Sequence[str] = (),
    replace: bool = False,
    resolver: ModuleResolver | None = None,
) -> Normalizer:
    """Register a named GRAM normalizer.

    Args:
        name: Stable normalizer identifier.
        fn: Callable accepting source text and ``GramConfig``.
        aliases: Optional alternate names that resolve to ``name``.
        replace: Whether to replace an existing normalizer or alias.
        resolver: Context-owned module resolver. Registration requires a
            `ModuleService`.

    Raises:
        ValueError: If the name is already registered.
        TypeError: If ``fn`` is not callable.

    Returns:
        Normalizer: Registered normalizer callable.
    """
    return register_gram_callable(name, fn, kind="gram_normalizer", aliases=aliases, replace=replace, resolver=resolver)


def resolve_gram_normalizer(config: GramConfig, *, resolver: ModuleResolver | None = None) -> Normalizer:
    """Return the configured normalizer callable.

    Args:
        config: Sparse-GRAM runtime config.
        resolver: Context-owned module resolver.

    Returns:
        Normalizer: Callable used to normalize source text.

    Raises:
        ValueError: If the configured normalizer cannot be resolved.
    """
    return resolve_gram_callable("gram_normalizer", config.normalizer, resolver=resolver)


def normalize_text(
    text: Any,
    config: GramConfig | None = None,
    *,
    resolver: ModuleResolver | None = None,
) -> NormalizedText:
    """Normalize one text value with the configured GRAM normalizer.

    Args:
        text: Source value to normalize. ``None`` is treated as an empty string.
        config: Optional sparse-GRAM runtime config. Defaults are loaded when omitted.
        resolver: Context-owned module resolver.

    Returns:
        NormalizedText: Normalized text plus optional origin span metadata.
    """
    selected = module_resolver(resolver)
    return _normalize_text(text, config or GramConfig.load(), selected)


def _normalize_text(text: Any, config: GramConfig, resolver: ModuleResolver) -> NormalizedText:
    """Normalize one text value through a concrete module resolver."""
    source = "" if text is None else str(text)
    key = (resolver.catalog.publication, config.fingerprint(), config.normalizer_lang, sha256hash(source))
    if config.normalize_cache_size > 0:
        with _CACHE_LOCK:
            cached = _CACHES.setdefault(resolver, {}).get(key)
        if cached is not None:
            return cached
    raw = resolve_gram_normalizer(config, resolver=resolver)(source, config)
    normalized = (
        raw
        if isinstance(raw, NormalizedText)
        else NormalizedText(str(raw), origins=None, lang=config.normalizer_lang, normalizer=config.normalizer_id(), precision="normalized_only")
    )
    if not normalized.normalizer:
        normalized = NormalizedText(
            normalized.text, normalized.origins, normalized.lang or config.normalizer_lang, config.normalizer_id(), normalized.precision
        )
    if config.normalize_cache_size > 0:
        with _CACHE_LOCK:
            cache = _CACHES.setdefault(resolver, {})
            if len(cache) >= config.normalize_cache_size:
                cache.pop(next(iter(cache)))
            cache[key] = normalized
    return normalized


def normalize_many(
    texts: Iterable[Any],
    config: GramConfig | None = None,
    *,
    resolver: ModuleResolver | None = None,
) -> list[NormalizedText]:
    """Normalize multiple values, using spaCy batching when that path is active.

    Args:
        texts: Source values to normalize. ``None`` values are treated as empty strings.
        config: Optional sparse-GRAM runtime config. Defaults are loaded when omitted.
        resolver: Context-owned module resolver.

    Returns:
        list[NormalizedText]: Normalized values in input order.
    """
    selected = module_resolver(resolver)
    config = config or GramConfig.load()
    values = ["" if text is None else str(text) for text in texts]
    if config.normalizer in ("auto", "spacy") and _should_use_spacy(values, config):
        spacy = _spacy_normalize_many(values, config)
        if spacy is not None:
            return spacy
    return [_normalize_text(value, config, selected) for value in values]


def _auto_normalize(text: str, config: GramConfig) -> NormalizedText:
    if config.normalizer in ("auto", "spacy") and _should_use_spacy([text], config):
        spacy = _spacy_normalize_many([text], config)
        if spacy is not None:
            return spacy[0]
    return _default_normalize(text, config)


def _default_normalize(text: str, config: GramConfig) -> NormalizedText:
    out = []
    origins: list[tuple[int, int]] = []
    last_space = False
    for pos, char in enumerate(text):
        piece = unicodedata.normalize(config.unicode_normalization, char)
        piece = piece.casefold() if config.casefold else piece
        for norm_char in piece:
            if norm_char.isspace() or unicodedata.category(norm_char).startswith("P"):
                if out and not last_space:
                    out.append(" ")
                    origins.append((pos, pos + 1))
                last_space = True
                continue
            if unicodedata.category(norm_char).startswith("C"):
                continue
            out.append(norm_char)
            origins.append((pos, pos + 1))
            last_space = False
    if out and out[-1] == " ":
        out.pop()
        origins.pop()
    return NormalizedText("".join(out), origins, config.normalizer_lang, config.normalizer_id(), "exact")


def _fast_normalize(text: str, config: GramConfig) -> NormalizedText:
    out: list[str] = []
    last_space = False
    for char in text:
        piece = unicodedata.normalize(config.unicode_normalization, char)
        piece = piece.casefold() if config.casefold else piece
        for norm_char in piece:
            category = unicodedata.category(norm_char)
            if norm_char.isspace() or category.startswith("P"):
                if out and not last_space:
                    out.append(" ")
                last_space = True
                continue
            if category.startswith("C"):
                continue
            out.append(norm_char)
            last_space = False
    if out and out[-1] == " ":
        out.pop()
    return NormalizedText("".join(out), None, config.normalizer_lang, config.normalizer_id(), "normalized_only")


def _char_normalize(text: str, config: GramConfig) -> NormalizedText:
    chars: list[str] = []
    origins: list[tuple[int, int]] = []
    for pos, char in enumerate(text):
        if char.isspace():
            continue
        if chars:
            chars.append(" ")
            origins.append((pos, pos))
        chars.append(char)
        origins.append((pos, pos + 1))
    return NormalizedText("".join(chars), origins, config.normalizer_lang, config.normalizer_id(), "exact")


def _none_normalize(text: str, config: GramConfig) -> NormalizedText:
    return NormalizedText(text, [(pos, pos + 1) for pos, _char in enumerate(text)], config.normalizer_lang, config.normalizer_id(), "exact")


def _contains_cjk(text: str) -> bool:
    return any(
        "\u3400" <= char <= "\u4dbf"
        or "\u4e00" <= char <= "\u9fff"
        or "\uf900" <= char <= "\ufaff"
        or "\u3040" <= char <= "\u30ff"
        or "\uac00" <= char <= "\ud7af"
        for char in text
    )


def _should_use_spacy(texts: list[str], config: GramConfig) -> bool:
    if config.normalizer not in ("auto", "spacy"):
        return False
    if config.normalizer_lang not in ("auto", "en"):
        return config.normalizer == "spacy"
    return bool(texts) and not any(_contains_cjk(text) for text in texts)


def _spacy_model(config: GramConfig) -> Any | None:
    key = (config.spacy_model, config.spacy_disable)
    with _SPACY_LOCK:
        if key in _SPACY_MODELS:
            return _SPACY_MODELS[key]
        try:
            spacy = importlib.import_module("spacy")
            model = spacy.load(config.spacy_model, disable=list(config.spacy_disable))
        except Exception:
            _SPACY_MODELS[key] = None
            return None
        _SPACY_MODELS[key] = model
        return model


def _spacy_normalize_many(texts: list[str], config: GramConfig) -> list[NormalizedText] | None:
    model = _spacy_model(config)
    if model is None:
        return None
    out: list[NormalizedText] = []
    docs = model.pipe(texts, batch_size=config.spacy_batch_size, n_process=config.spacy_n_process)
    for source, doc in zip(texts, docs):
        parts = []
        origins: list[tuple[int, int]] = []
        for token in doc:
            lemma = token.lemma_ if token.lemma_ and token.lemma_ != "-PRON-" else token.text
            norm = _default_normalize(lemma, config).text
            if not norm:
                continue
            if parts:
                parts.append(" ")
                origins.append((max(0, token.idx - 1), token.idx))
            parts.append(norm)
            origins.extend([(token.idx, token.idx + len(token.text)) for _ in norm])
        out.append(NormalizedText("".join(parts), origins, "en", config.normalizer_id(), "token_span"))
    return out
