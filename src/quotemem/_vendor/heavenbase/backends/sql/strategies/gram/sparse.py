"""Sparse gram candidate-generation helpers."""

from __future__ import annotations

__all__ = [
    "GramPart",
    "hash_value",
    "keyword_grams",
    "query_grams",
    "resolve_hash",
    "unit_count",
]

from heavenbase.utils import Any, Callable, dataclass
from heavenbase.utils.hash import resolve_hash_int

from .....modules import ModuleResolver, module_resolver
from .config import GramConfig
from .analyzers import gram_text, gram_units, unit_count

Hasher = Callable[[Any, Any, str], int]


@dataclass(frozen=True)
class GramPart:
    """One indexed gram hash and bit position for a normalized keyword."""

    gram: str
    gram_hash: str
    offset: int
    mask_bit: int


def resolve_hash(config: GramConfig) -> Hasher:
    """Return the configured hash function.

    Args:
        config: Sparse-GRAM runtime config.

    Returns:
        Hasher: Integer hash function used for gram hashes.
    """
    return resolve_hash_int(config.hash_function, allow_dotted=False)


def hash_value(value: str, config: GramConfig) -> int:
    """Return a deterministic gram hash weight after salt and bit truncation.

    Args:
        value: Gram text to hash.
        config: Sparse-GRAM runtime config.

    Returns:
        int: Salted hash truncated to ``config.hash_bits``.
    """
    salt = {"salt": config.hash_salt, "seed": config.seed}
    raw = int(resolve_hash(config)(value, salt=salt, sep="||"))
    return raw & ((1 << config.hash_bits) - 1)


def keyword_grams(
    keyword_norm: str,
    config: GramConfig,
    *,
    resolver: ModuleResolver | None = None,
) -> list[GramPart]:
    """Return required grams for one normalized keyword.

    Args:
        keyword_norm: Normalized keyword text from an indexed row.
        config: Sparse-GRAM runtime config.
        resolver: Context-owned module resolver.

    Returns:
        list[GramPart]: Required gram hashes and mask bits for candidate verification.
    """
    grams = _keyword_grams(keyword_norm, config, module_resolver(resolver))
    return [GramPart(gram, gram_hash, offset, 1 << offset) for offset, (gram, gram_hash) in enumerate(grams)]


def query_grams(
    query_norm: str,
    config: GramConfig,
    *,
    short_widths: set[int] | None = None,
    resolver: ModuleResolver | None = None,
) -> set[str]:
    """Return gram hashes present in one normalized query string.

    Args:
        query_norm: Normalized query text.
        config: Sparse-GRAM runtime config.
        short_widths: Optional short keyword widths observed in the index.
        resolver: Context-owned module resolver.

    Returns:
        set[str]: Candidate gram hashes for lookup.
    """
    selected = module_resolver(resolver)
    return {str(hash_value(gram, config)) for gram in _query_grams(query_norm, config, short_widths=short_widths, resolver=selected)}


def _keyword_grams(text: str, config: GramConfig, resolver: ModuleResolver) -> list[tuple[str, str]]:
    grams = _indexed_spans(gram_units(text, config, resolver=resolver), config)
    parts = [(gram, str(hash_value(gram, config))) for gram in grams]
    if config.sparse_grams_per_keyword and len(parts) > config.sparse_grams_per_keyword:
        parts = sorted(parts, key=lambda item: (-len(item[0]), -int(item[1]), item[0]))[: config.sparse_grams_per_keyword]
    return parts


def _query_grams(text: str, config: GramConfig, *, short_widths: set[int] | None, resolver: ModuleResolver) -> list[str]:
    units = gram_units(text, config, resolver=resolver)
    if not units:
        return []
    out: list[str] = []
    widths = short_widths if short_widths is not None else set(range(1, config.gram_width + 1))
    for width in sorted(width for width in widths if 1 <= width <= config.gram_width):
        if len(units) >= width:
            out.extend(gram_text(units[start : start + width], config) for start in range(0, len(units) - width + 1))
    out.extend(_covering_spans(units, config))
    return list(dict.fromkeys(out))


def _indexed_spans(units: list[str], config: GramConfig) -> list[str]:
    if not units:
        return []
    if len(units) <= config.gram_width:
        return [gram_text(units, config)]
    spans = _all_sparse_spans(units, config)
    if config.sparse_max_width:
        bounded = [span for span in spans if _span_width(span, config) <= config.sparse_max_width]
        spans = bounded or spans
    return spans


def _covering_spans(units: list[str], config: GramConfig) -> list[str]:
    if len(units) <= config.gram_width:
        return []
    hashes = _base_hashes(units, config)
    if len(hashes) <= 1:
        return []
    out: list[str] = []
    stack: list[tuple[int, int]] = [(hashes[0], 0)]
    for index in range(1, len(hashes)):
        current = hashes[index]
        while stack and stack[-1][0] <= current:
            out.append(gram_text(units[stack[-1][1] : index + config.gram_width], config))
            stack.pop()
        if stack:
            out.append(gram_text(units[stack[-1][1] : index + config.gram_width], config))
        stack.append((current, index))
    return list(dict.fromkeys(out))


def _all_sparse_spans(units: list[str], config: GramConfig) -> list[str]:
    if config.sparse_policy not in ("cartesian-cover", "default"):
        raise ValueError(f"Unsupported GRAM sparse policy {config.sparse_policy!r}")
    hashes = _base_hashes(units, config)
    if len(hashes) <= 1:
        return [gram_text(units, config)]
    out: list[str] = []
    global_max = max(hashes)
    running_max = hashes[0]
    start = 0
    for index in range(1, len(hashes)):
        running_max = max(running_max, hashes[index])
        if hashes[index] == global_max or hashes[index] == running_max:
            out.append(gram_text(units[start : index + config.gram_width], config))
            start = index
    stop = len(units)
    running_max = hashes[-1]
    for index in range(len(hashes) - 2, -1, -1):
        running_max = max(running_max, hashes[index])
        if hashes[index] == running_max:
            out.append(gram_text(units[index:stop], config))
            stop = index + config.gram_width
    return list(dict.fromkeys(out))


def _base_hashes(units: list[str], config: GramConfig) -> list[int]:
    return [hash_value(gram_text(units[start : start + config.gram_width], config), config) for start in range(0, len(units) - config.gram_width + 1)]


def _separator(config: GramConfig) -> str:
    return str((config.tokenizer_args or {}).get("separator", "\x1f"))


def _span_width(span: str, config: GramConfig) -> int:
    return len(span) if config.gram_unit == "char" else len(span.split(_separator(config)))
