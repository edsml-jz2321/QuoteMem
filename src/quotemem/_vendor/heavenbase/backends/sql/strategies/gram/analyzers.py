"""Sparse-GRAM analyzers and tokenizer registry."""

from __future__ import annotations

__all__ = [
    "gram_text",
    "gram_units",
    "register_gram_tokenizer",
    "resolve_gram_tokenizer",
    "unit_count",
]

from heavenbase.utils import Callable, Sequence

from .....modules import ModuleResolver, module_resolver
from .config import GramConfig
from .registry import register_gram_callable, resolve_gram_callable

Tokenizer = Callable[[str, GramConfig], list[str]]

_TOKEN_SEPARATOR = "\x1f"


def register_gram_tokenizer(
    name: str,
    fn: Tokenizer,
    *,
    aliases: Sequence[str] = (),
    replace: bool = False,
    resolver: ModuleResolver | None = None,
) -> Tokenizer:
    """Register a named GRAM tokenizer.

    Args:
        name: Stable tokenizer identifier.
        fn: Callable accepting normalized text and ``GramConfig``.
        aliases: Optional alternate names that resolve to ``name``.
        replace: Whether to replace an existing tokenizer or alias.
        resolver: Context-owned module resolver. Registration requires a
            `ModuleService`.

    Raises:
        ValueError: If the tokenizer name is blank or already registered.
        TypeError: If ``fn`` is not callable.

    Returns:
        Tokenizer: Registered tokenizer callable.
    """
    return register_gram_callable(name, fn, kind="gram_tokenizer", aliases=aliases, replace=replace, resolver=resolver)


def resolve_gram_tokenizer(config: GramConfig, *, resolver: ModuleResolver | None = None) -> Tokenizer:
    """Return the configured tokenizer callable.

    Args:
        config: Sparse-GRAM runtime config.
        resolver: Context-owned module resolver.

    Returns:
        Tokenizer: Callable used to split normalized text into analyzer tokens.

    Raises:
        ValueError: If the configured tokenizer cannot be resolved.
    """
    return resolve_gram_callable("gram_tokenizer", config.tokenizer, resolver=resolver)


def gram_units(text: str, config: GramConfig, *, resolver: ModuleResolver | None = None) -> list[str]:
    """Return analyzer units for a normalized text string.

    Args:
        text: Normalized text to analyze.
        config: Sparse-GRAM runtime config.
        resolver: Context-owned module resolver.

    Returns:
        list[str]: Character or token units used for gram spans.

    Raises:
        ValueError: If ``config.gram_unit`` is unsupported.
    """
    selected = module_resolver(resolver)
    unit = config.gram_unit
    if unit == "char":
        return list(text)
    if unit in ("token", "space"):
        return [item for item in resolve_gram_tokenizer(config, resolver=selected)(text, config) if item]
    raise ValueError(f"Unsupported GRAM unit {unit!r}")


def unit_count(text: str, config: GramConfig, *, resolver: ModuleResolver | None = None) -> int:
    """Return the number of analyzer units in normalized text.

    Args:
        text: Normalized text to measure.
        config: Sparse-GRAM runtime config.
        resolver: Context-owned module resolver.

    Returns:
        int: Analyzer unit count.
    """
    return len(gram_units(text, config, resolver=resolver))


def gram_text(units: list[str], config: GramConfig) -> str:
    """Return the stable hash text for a gram unit span.

    Args:
        units: Analyzer units in one gram span.
        config: Sparse-GRAM runtime config.

    Returns:
        str: Stable text sent to the gram hash function.
    """
    if config.gram_unit == "char":
        return "".join(units)
    separator = str((config.tokenizer_args or {}).get("separator", _TOKEN_SEPARATOR))
    return separator.join(units)


def _char_tokenize(text: str, _config: GramConfig) -> list[str]:
    return [char for char in text if not char.isspace()]


def _space_tokenize(text: str, _config: GramConfig) -> list[str]:
    return [item for item in text.split(" ") if item]
