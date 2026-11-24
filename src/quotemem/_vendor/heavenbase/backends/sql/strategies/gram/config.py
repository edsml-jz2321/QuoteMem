"""Sparse-GRAM strategy adapter configuration."""

from __future__ import annotations

__all__ = [
    "GramConfig",
]

from heavenbase.utils import Any, Dict, dataclass
from heavenbase.utils.config import CM_HVNB
from heavenbase.utils.hash import sha256hash

from .registry import callable_ref


@dataclass(frozen=True)
class GramConfig:
    """Runtime config for sparse-GRAM normalization, hashing, and indexing."""

    normalizer: Any = "default"
    normalizer_lang: str = "auto"
    unicode_normalization: str = "NFC"
    casefold: bool = True
    spacy_model: str = "en_core_web_sm"
    spacy_disable: tuple[str, ...] = ("parser", "ner", "textcat")
    spacy_batch_size: int = 256
    spacy_n_process: int = 1
    normalize_cache_size: int = 100000
    normalize_cache_persist: bool = True
    min_length: int = 2
    gram_unit: str = "char"
    tokenizer: Any = "char"
    tokenizer_args: Dict[str, Any] = None
    gram_width: int = 3
    sparse_policy: str = "cartesian-cover"
    sparse_max_width: int = 0
    sparse_grams_per_keyword: int = 0
    hash_function: str = "crc32"
    hash_bits: int = 32
    hash_salt: str = "heavenbase.gram.v1"
    seed: int = 42
    posting_page_size: int = 10000
    hot_gram_threshold: int = 0

    @classmethod
    def load(cls, *, manager: Any = CM_HVNB, **overrides: Any) -> "GramConfig":
        """Load sparse-GRAM defaults and explicit strategy overrides.

        Args:
            manager: ConfigManager supplying runtime defaults. Pass ``None``
                during bootstrap-safe Backend construction.
            **overrides: Explicit config values from strategy options or tests. ``None``
                values are ignored so callers can layer optional overrides.

        Returns:
            GramConfig: Normalized sparse-GRAM runtime config.

        Raises:
            TypeError: If ``tokenizer_args`` is not a mapping.
        """
        root = {} if manager is None else dict(manager.get("heavenbase.strategy_defaults.sparse_gram", default={}) or {})
        root.update({key: value for key, value in overrides.items() if value is not None})
        spacy_disable = root.get("spacy_disable", cls.spacy_disable)
        if isinstance(spacy_disable, str):
            spacy_disable = tuple(item.strip() for item in spacy_disable.split(",") if item.strip())
        tokenizer_args = root.get("tokenizer_args", cls.tokenizer_args) or {}
        if not isinstance(tokenizer_args, dict):
            raise TypeError("GRAM tokenizer_args must be a mapping")
        return cls(
            normalizer=root.get("normalizer", cls.normalizer),
            normalizer_lang=str(root.get("normalizer_lang", cls.normalizer_lang)),
            unicode_normalization=str(root.get("unicode_normalization", cls.unicode_normalization)),
            casefold=bool(root.get("casefold", cls.casefold)),
            spacy_model=str(root.get("spacy_model", cls.spacy_model)),
            spacy_disable=tuple(spacy_disable),
            spacy_batch_size=max(1, int(root.get("spacy_batch_size", cls.spacy_batch_size))),
            spacy_n_process=max(1, int(root.get("spacy_n_process", cls.spacy_n_process))),
            normalize_cache_size=max(0, int(root.get("normalize_cache_size", cls.normalize_cache_size))),
            normalize_cache_persist=bool(root.get("normalize_cache_persist", cls.normalize_cache_persist)),
            min_length=max(1, int(root.get("min_length", cls.min_length))),
            gram_unit=str(root.get("gram_unit", cls.gram_unit)),
            tokenizer=root.get("tokenizer", cls.tokenizer),
            tokenizer_args=dict(tokenizer_args),
            gram_width=max(2, int(root.get("gram_width", cls.gram_width))),
            sparse_policy=str(root.get("sparse_policy", cls.sparse_policy)),
            sparse_max_width=max(0, int(root.get("sparse_max_width", cls.sparse_max_width))),
            sparse_grams_per_keyword=max(0, int(root.get("sparse_grams_per_keyword", root.get("postings_per_keyword", cls.sparse_grams_per_keyword)))),
            hash_function=str(root.get("hash_function", cls.hash_function)),
            hash_bits=max(1, int(root.get("hash_bits", cls.hash_bits))),
            hash_salt=str(root.get("hash_salt", cls.hash_salt)),
            seed=int(root.get("seed", cls.seed)),
            posting_page_size=max(1, int(root.get("posting_page_size", cls.posting_page_size))),
            hot_gram_threshold=max(0, int(root.get("hot_gram_threshold", cls.hot_gram_threshold))),
        )

    def with_options(self, **overrides: Any) -> "GramConfig":
        """Return this config with explicit strategy overrides applied.

        Args:
            **overrides: Replacement config values. ``None`` values are ignored.

        Returns:
            GramConfig: New config with defaults, current values, and overrides normalized.
        """
        data = {key: getattr(self, key) for key in self.__dataclass_fields__}
        data.update({key: value for key, value in overrides.items() if value is not None})
        return type(self).load(manager=None, **data)

    def normalizer_id(self) -> str:
        """Return the normalizer identity persisted with indexed keywords.

        Args:
            None.

        Returns:
            str: Stable normalizer identity for compatibility checks and match metadata.
        """
        if callable(self.normalizer):
            return f"{callable_ref(self.normalizer)}:{self.normalizer_lang}"
        if self.normalizer == "auto":
            return f"auto:{self.normalizer_lang}:{self.spacy_model}"
        return f"{self.normalizer}:{self.normalizer_lang}"

    def fingerprint(self) -> str:
        """Return an index compatibility fingerprint.

        The config is immutable, so the hash is computed once per instance:
        normalization and indexing ask for it for every keyword.

        Args:
            None.

        Returns:
            str: Stable hash for config fields that affect index semantics.
        """
        cached = self.__dict__.get("_fingerprint")
        if cached is None:
            cached = sha256hash(self.fingerprint_data())
            object.__setattr__(self, "_fingerprint", cached)
        return cached

    def fingerprint_data(self) -> Dict[str, Any]:
        """Return config fields that affect index semantics.

        Args:
            None.

        Returns:
            Dict[str, Any]: JSON-compatible fingerprint input data.
        """
        return {
            "normalizer": self.normalizer_id(),
            "unicode_normalization": self.unicode_normalization,
            "casefold": self.casefold,
            "min_length": self.min_length,
            "gram_unit": self.gram_unit,
            "tokenizer": callable_ref(self.tokenizer) if callable(self.tokenizer) else self.tokenizer,
            "tokenizer_args": self.tokenizer_args or {},
            "gram_width": self.gram_width,
            "sparse_policy": self.sparse_policy,
            "sparse_max_width": self.sparse_max_width,
            "sparse_grams_per_keyword": self.sparse_grams_per_keyword,
            "hash_function": self.hash_function,
            "hash_bits": self.hash_bits,
            "hash_salt": self.hash_salt,
            "seed": self.seed,
        }
