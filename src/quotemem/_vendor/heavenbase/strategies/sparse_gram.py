"""Sparse GRAM keyword index storage strategy."""

from __future__ import annotations

__all__ = [
    "SparseGramIndex",
]

from heavenbase.utils import Any, Callable

from .base import StorageStrategy


class SparseGramIndex(StorageStrategy):
    """Storage strategy value for sparse GRAM reverse-containment indexes."""

    identifier = "sparse-gram"
    option_fields = (
        "normalizer",
        "normalizer_lang",
        "unicode_normalization",
        "casefold",
        "gram_unit",
        "tokenizer",
        "tokenizer_args",
        "gram_width",
        "sparse_policy",
        "sparse_max_width",
        "sparse_grams_per_keyword",
        "hash_function",
        "hash_bits",
        "hash_salt",
        "seed",
        "min_length",
    )

    def __init__(
        self,
        *,
        normalizer: str | Callable[..., Any] | None = None,
        normalizer_lang: str | None = None,
        unicode_normalization: str | None = None,
        casefold: bool | None = None,
        gram_unit: str | None = None,
        tokenizer: str | Callable[..., Any] | None = None,
        tokenizer_args: dict | None = None,
        gram_width: int | None = None,
        sparse_policy: str | None = None,
        sparse_max_width: int | None = None,
        sparse_grams_per_keyword: int | None = None,
        hash_function: str | None = None,
        hash_bits: int | None = None,
        hash_salt: str | None = None,
        seed: int | None = None,
        min_length: int | None = None,
    ) -> None:
        """Create a sparse-GRAM strategy value with strategy-owned options.

        Args:
            normalizer (str | Callable[..., Any] | None): Registered normalizer
                identifier or direct callable.
            normalizer_lang (str | None): Optional language hint for normalizers.
            unicode_normalization (str | None): Unicode normalization form.
            casefold (bool | None): Whether text normalizers should casefold.
            gram_unit (str | None): Gram unit. Supported values are `char` and
                `token`; `None` resolves through sparse-GRAM config.
            tokenizer (str | Callable[..., Any] | None): Registered tokenizer
                identifier or direct callable for token grams.
            tokenizer_args (dict | None): Optional tokenizer-specific arguments.
            gram_width (int | None): Base gram width.
            sparse_policy (str | None): Sparse span policy identifier.
            sparse_max_width (int | None): Optional maximum sparse span width.
            sparse_grams_per_keyword (int | None): Optional cap on stored sparse
                spans per keyword.
            hash_function (str | None): Hash function identifier.
            hash_bits (int | None): Hash bit width.
            hash_salt (str | None): Hash salt.
            seed (int | None): Deterministic sparse/hash seed.
            min_length (int | None): Minimum keyword length.

        Returns:
            None: This initializer stores normalized sparse-GRAM options.
        """
        super().__init__(
            normalizer=normalizer,
            normalizer_lang=normalizer_lang,
            unicode_normalization=unicode_normalization,
            casefold=casefold,
            gram_unit=gram_unit,
            tokenizer=tokenizer,
            tokenizer_args=tokenizer_args,
            gram_width=gram_width,
            sparse_policy=sparse_policy,
            sparse_max_width=sparse_max_width,
            sparse_grams_per_keyword=sparse_grams_per_keyword,
            hash_function=hash_function,
            hash_bits=hash_bits,
            hash_salt=hash_salt,
            seed=seed,
            min_length=min_length,
        )
