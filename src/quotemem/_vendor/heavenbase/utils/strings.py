"""String, markdown, and lightweight text-processing helpers."""

from __future__ import annotations

__all__ = [
    "asymmetric_jaccard_score",
    "bullet_block",
    "bullet_dict",
    "bullet_list",
    "code_block",
    "dedent",
    "example_block",
    "fast_prompt_section",
    "file_block",
    "format_value",
    "generate_ngrams",
    "indent",
    "is_delimiter",
    "is_spacy_available",
    "json_block",
    "line_numbered",
    "md_block",
    "md_section",
    "md_symbol",
    "normalize_text",
    "numbered_list",
    "omission_list",
    "shorten",
    "snake",
    "tag_block",
    "truncate",
    "value_repr",
]

import re
import string
import textwrap
from decimal import Decimal

from .typing import Any, Callable, List, Optional, Set, Tuple, Union

_SPACY_NLP = None
_SPACY_AVAILABLE = None
_Messages = List[dict[str, str]]
_SectionInput = Optional[Union[str, List[str], dict[str, Union[str, List[str]]]]]


def truncate(text: object, cutoff: int = -1) -> str:
    """
    Return text shortened with an ellipsis when it exceeds cutoff.

    Use this helper for Markdown snippets, compact display text, and lightweight text normalization.

    Args:
        text (object): Text value to format, normalize, copy, or compare.
        cutoff (int): Maximum string length before truncation. ``-1`` disables truncation. Defaults to ``-1``.

    Returns:
        str: The text shortened with an ellipsis when it exceeds cutoff.
    """
    value = str(text)
    if cutoff < 0 or len(value) <= cutoff:
        return value
    if cutoff <= 3:
        return value[:cutoff]
    return f"{value[: cutoff - 3]}..."


shorten = truncate


def value_repr(value: Any, cutoff: int = -1, round_digits: int = 6) -> str:
    """
    Return a compact repr with float-like values rounded for display.

    Use this helper for Markdown snippets, compact display text, and lightweight text normalization.

    Args:
        value (Any): Value to read, set, validate, hash, or format.
        cutoff (int): Maximum string length before truncation. ``-1`` disables truncation. Defaults to ``-1``.
        round_digits (int): Number of digits used when rounding float-like display values. Defaults to ``6``.

    Returns:
        str: The a compact repr with float-like values rounded for display.
    """
    if isinstance(value, Decimal):
        value = round(float(value), round_digits)
    elif isinstance(value, float):
        value = round(value, round_digits)
    return truncate(repr(value), cutoff=cutoff)


def omission_list(items: List[Any], top: int = -1, bottom: int = 1) -> List[Any]:
    """
    Return a list with the middle omitted when it is too long.

    Use this helper for Markdown snippets, compact display text, and lightweight text normalization.

    Args:
        items (List[Any]): Iterable of input items. Iterators are consumed lazily when progress totals are not requested.
        top (int): Number of leading items to keep before an omission marker. ``-1`` keeps all items.
        bottom (int): Number of trailing items to keep after an omission marker. Defaults to ``1``.

    Returns:
        List[Any]: The a list with the middle omitted when it is too long.
    """
    max_items = -1 if top < 0 else top + bottom
    if max_items < 0 or len(items) <= max_items:
        return items
    n_omit = len(items) - max_items
    return items[:top] + [f"... (omitting {n_omit})"] + (items[-bottom:] if bottom > 0 else [])


def md_symbol(content: object) -> str:
    """
    Return a markdown fence that does not appear in content.

    Use this helper for Markdown snippets, compact display text, and lightweight text normalization.

    Args:
        content (object): Optional file content to write. Defaults to ``None``.

    Returns:
        str: The a markdown fence that does not appear in content.
    """
    symbol, text = "```", str(content)
    while symbol in text:
        symbol += "```"
    return symbol


def line_numbered(content: object, start: int = -1, window: Optional[Tuple[int, int]] = None) -> str:
    """
    Return content with optional line numbers and slicing.

    Use this helper for Markdown snippets, compact display text, and lightweight text normalization.

    Args:
        content (object): Optional file content to write. Defaults to ``None``.
        start (int): Starting line number. ``-1`` disables line numbering when supported. Defaults to ``-1``.
        window (Optional[Tuple[int, int]]): Optional ``(start, end)`` slice applied to rendered lines. Defaults to ``None``.

    Returns:
        str: The content with optional line numbers and slicing.
    """
    lines = str(content).splitlines()
    out = [f"({idx:4d})  {line}" for idx, line in enumerate(lines, start=start)] if start >= 0 else lines
    return "\n".join(out[window[0] : window[1]] if window is not None else out)


def indent(text: object, tab: Union[int, str] = 4, **kwargs: object) -> str:
    """
    Indent every line by spaces or a custom prefix.

    Use this helper for Markdown snippets, compact display text, and lightweight text normalization.

    Args:
        text (object): Text value to format, normalize, copy, or compare.
        tab (Union[int, str]): Indent width or literal prefix used for each line. Defaults to ``4`` spaces.
        **kwargs (object): Extra keyword arguments forwarded to the underlying serializer, command, or callable.

    Returns:
        str: The result of ``indent``.
    """
    return textwrap.indent(str(text), prefix=(" " * tab if isinstance(tab, int) else tab), **kwargs)


def dedent(text: object) -> str:
    """
    Remove shared indentation and surrounding blank space.

    Use this helper for Markdown snippets, compact display text, and lightweight text normalization.

    Args:
        text (object): Text value to format, normalize, copy, or compare.

    Returns:
        str: The result of ``dedent``.
    """
    return textwrap.dedent(str(text)).strip()


def bullet_block(text: object, bullet: Optional[str] = "-") -> str:
    """
    Format a possibly multi-line block as one bullet item.

    Use this helper for Markdown snippets, compact display text, and lightweight text normalization.

    Args:
        text (object): Text value to format, normalize, copy, or compare.
        bullet (Optional[str]): Bullet prefix used for rendered Markdown lists. Defaults vary by helper. Defaults to ``'-'``.

    Returns:
        str: The result of ``bullet_block``.
    """
    if bullet is None:
        return str(text)
    n_indent = len(str(bullet)) + 1
    return indent(str(text), n_indent).replace(" " * n_indent, f"{bullet} ", 1)


def bullet_list(items: List[object], bullet: str = "-") -> str:
    """Format items as a markdown bullet list.

    Use this helper for Markdown snippets, compact display text, and lightweight text normalization.

    Args:
        items (List[object]): Iterable of input items. Iterators are consumed lazily when progress totals are not requested.
        bullet (str): Bullet prefix used for rendered Markdown lists. Defaults vary by helper. Defaults to ``'-'``.

    Returns:
        str: The result of ``bullet_list``.

    Examples:
            >>> from heavenbase.utils.strings import bullet_list
            >>> bullet_list(["alpha", "beta"])
            '- alpha
    - beta'
    """
    return "\n".join(bullet_block(item, bullet=bullet) for item in items)


def numbered_list(items: List[object]) -> str:
    """
    Format items as an aligned numbered list.

    Use this helper for Markdown snippets, compact display text, and lightweight text normalization.

    Args:
        items (List[object]): Iterable of input items. Iterators are consumed lazily when progress totals are not requested.

    Returns:
        str: The result of ``numbered_list``.
    """
    n_digits = len(str(len(items)))
    return "\n".join(bullet_block(item, f"{idx + 1}." + " " * (n_digits - len(str(idx + 1)))) for idx, item in enumerate(items))


def md_block(content: object, lang: str = "", header: str = "") -> str:
    """
    Format content as a fenced markdown block.

    Use this helper for Markdown snippets, compact display text, and lightweight text normalization.

    Args:
        content (object): Optional file content to write. Defaults to ``None``.
        lang (str): Code-block language tag. Defaults to ``""``.
        header (str): Optional heading text rendered before a block. Defaults to ``""``.

    Returns:
        str: The result of ``md_block``.
    """
    text = str(content)
    symbol = md_symbol(text)
    block = f"{symbol}{lang}\n{text}\n{symbol}"
    return f"{header}\n{block}" if header else block


def code_block(content: object, lang: str = "", start: int = -1, window: Optional[Tuple[int, int]] = None, header: str = "") -> str:
    """
    Format code with optional line numbering.

    Use this helper for Markdown snippets, compact display text, and lightweight text normalization.

    Args:
        content (object): Optional file content to write. Defaults to ``None``.
        lang (str): Code-block language tag. Defaults to ``""``.
        start (int): Starting line number. ``-1`` disables line numbering when supported. Defaults to ``-1``.
        window (Optional[Tuple[int, int]]): Optional ``(start, end)`` slice applied to rendered lines. Defaults to ``None``.
        header (str): Optional heading text rendered before a block. Defaults to ``""``.

    Returns:
        str: The result of ``code_block``.
    """
    return md_block(line_numbered(content, start=start, window=window), lang=lang, header=header)


def file_block(file: str, content: object, lang: str = "", start: int = -1, window: Optional[Tuple[int, int]] = None) -> str:
    """
    Format file content with a compact header.

    Use this helper for Markdown snippets, compact display text, and lightweight text normalization.

    Args:
        file (str): File path or stream target, depending on the helper. Defaults to ``None``.
        content (object): Optional file content to write. Defaults to ``None``.
        lang (str): Code-block language tag. Defaults to ``""``.
        start (int): Starting line number. ``-1`` disables line numbering when supported. Defaults to ``-1``.
        window (Optional[Tuple[int, int]]): Optional ``(start, end)`` slice applied to rendered lines. Defaults to ``None``.

    Returns:
        str: The result of ``file_block``.
    """
    text = str(content)
    return code_block(text, lang=lang, start=start, window=window, header=f"File: `{file}` ({len(text.splitlines())} lines)")


def json_block(data: Any, start: int = -1, window: Optional[Tuple[int, int]] = None, header: str = "", **kwargs: Any) -> str:
    """
    Format data as a JSON code block.

    Use this helper for Markdown snippets, compact display text, and lightweight text normalization.

    Args:
        data (Any): Dictionary layer or payload to read, write, merge, or render.
        start (int): Starting line number. ``-1`` disables line numbering when supported. Defaults to ``-1``.
        window (Optional[Tuple[int, int]]): Optional ``(start, end)`` slice applied to rendered lines. Defaults to ``None``.
        header (str): Optional heading text rendered before a block. Defaults to ``""``.
        **kwargs (Any): Extra keyword arguments forwarded to the underlying serializer, command, or callable.

    Returns:
        str: The result of ``json_block``.
    """
    from .serialize import dumps_json

    return code_block(dumps_json(data, **kwargs), lang="json", start=start, window=window, header=header)


def tag_block(tag: str, content: object, inline: bool = False) -> str:
    """
    Wrap content in a simple XML-style tag.

    Use this helper for Markdown snippets, compact display text, and lightweight text normalization.

    Args:
        tag (str): XML-style wrapper tag name.
        content (object): Optional file content to write. Defaults to ``None``.
        inline (bool): Whether to render a tag on one line. Defaults to ``False``.

    Returns:
        str: The result of ``tag_block``.
    """
    return f"<{tag}>{content}</{tag}>" if inline else f"<{tag}>\n{content}\n</{tag}>"


def _check_mode(mode: str, supported: List[str]) -> None:
    if mode not in supported:
        from .values import raise_mismatch

        raise_mismatch(supported, mode, name="mode")


def format_value(
    value: Any, schema: Optional[Union[str, dict]] = None, bullet: Optional[str] = None, key: Optional[str] = None, tag: Optional[str] = None
) -> Optional[str]:
    """
    Format a value through a compact schema-driven Markdown renderer.

    Use this helper for Markdown snippets, compact display text, and lightweight text normalization.

    Args:
        value (Any): Value to read, set, validate, hash, or format.
        schema (Optional[Union[str, dict]]): Optional rendering schema. ``None`` uses the helper default mode.
        bullet (Optional[str]): Bullet prefix used for rendered Markdown lists. Defaults vary by helper. Defaults to ``None``.
        key (Optional[str]): Dictionary key, config path, or display label. Defaults to ``None``.
        tag (Optional[str]): XML-style wrapper tag name. Defaults to ``None``.

    Returns:
        Optional[str]: Rendered text, or ``None`` when schema mode is ``"none"``.
    """
    if schema is None:
        schema = {"mode": "repr", "kwargs": {}}
    if isinstance(schema, str):
        schema = {"mode": schema, "kwargs": {}}
    mode, kwargs = schema.get("mode", "repr"), dict(schema.get("kwargs", {}))
    _check_mode(mode, ["base", "repr", "dump", "code", "file", "json", "nest", "tag", "param", "todo", "none"])
    if mode == "none":
        return None
    if mode == "base":
        out = str(value)
    elif mode == "repr":
        out = repr(value)
    elif mode == "dump":
        from .serialize import dumps_json

        out = dumps_json(value, **({"indent": None} | kwargs))
    elif mode == "code":
        out = code_block(value, **kwargs)
    elif mode == "file":
        out = file_block(key or "value", value, **kwargs)
    elif mode == "json":
        out = json_block(value, **({"indent": 0 if bullet is None else len(str(bullet)) + 1} | kwargs))
    elif mode == "nest":
        if not isinstance(value, dict):
            raise ValueError(f"Value must be a dict when using nest mode. Got {type(value).__name__}.")
        out = bullet_dict(value, bullet=kwargs.get("bullet", bullet or "-"), schema=kwargs.get("schema"))
    elif mode == "tag":
        out = tag_block(
            tag or kwargs.get("tag") or key or "output", format_value(value, schema=kwargs.get("schema", "base")) or "", inline=kwargs.get("inline", True)
        )
    elif mode == "param":
        out = _format_param(value, key=key, required=kwargs.get("required", False))
    else:
        out = "TODO"
    if key is None or mode in ("tag", "param"):
        return out
    return f"{key}:\n{out}" if mode in ("code", "file", "json", "nest") else f"{key}: {out}"


def bullet_dict(data: dict, bullet: str = "-", schema: Optional[Union[str, dict]] = None) -> str:
    """
    Format a dictionary as a markdown bullet list.

    Use this helper for Markdown snippets, compact display text, and lightweight text normalization.

    Args:
        data (dict): Dictionary layer or payload to read, write, merge, or render.
        bullet (str): Bullet prefix used for rendered Markdown lists. Defaults vary by helper. Defaults to ``'-'``.
        schema (Optional[Union[str, dict]]): Optional rendering schema. ``None`` uses the helper default mode.

    Returns:
        str: The result of ``bullet_dict``.
    """
    schema_map = {key: schema for key in data} if isinstance(schema, str) else (schema or {})
    items = [
        item
        for item in (format_value(value, schema=schema_map.get(key, "repr"), bullet=bullet, key=str(key)) for key, value in data.items())
        if item is not None
    ]
    return bullet_list(items, bullet=bullet)


def md_section(
    title: Optional[str] = None, level: int = 1, content: Optional[object] = None, sections: Optional[List[dict]] = None, end: Optional[object] = None
) -> str:
    """Render a Markdown section tree with stable blank-line spacing.

    Use this helper for Markdown snippets, compact display text, and lightweight text normalization.

    Args:
        title (Optional[str]): Optional Markdown section title. Defaults to ``None``.
        level (int): Logging level name or numeric level. Examples: ``"DEBUG"``, ``"INFO"``, and ``"WARNING"``. Defaults to ``1``.
        content (Optional[object]): Optional file content to write. Defaults to ``None``.
        sections (Optional[List[dict]]): Optional child section dictionaries rendered recursively. Defaults to ``None``.
        end (Optional[object]): Optional trailing content appended after rendered child sections. Defaults to ``None``.

    Returns:
        str: Rendered Markdown section string ending with a newline when non-empty.

    Examples:
            >>> from heavenbase.utils.strings import md_section
            >>> md_section(title="Plan", content="Ready")
            '# Plan

    Ready
    '
    """
    parts: List[str] = []
    if title is not None:
        parts.append(f"{'#' * level} {title}")
    if content is not None and str(content).strip():
        parts.append(str(content).strip())
    for section in sections or []:
        rendered = md_section(level=level + 1, **section).strip()
        if rendered:
            parts.append(rendered)
    if end is not None and str(end).strip():
        parts.append(str(end).strip())
    return ("\n\n".join(parts).strip() + "\n") if parts else ""


def example_block(
    inputs: dict,
    output: Any,
    bullet: str = "-",
    inputs_schema: Optional[dict] = None,
    output_schema: Optional[Union[str, dict]] = "tag",
    hints: Optional[Union[str, List[str]]] = None,
    expected: Any = ...,
    notes: Optional[Union[str, List[str]]] = None,
    tag: Optional[str] = None,
    tr: Optional[Callable[[str], str]] = None,
) -> str:
    """
    Render an example with inputs, output, hints, expected value, and notes.

    Use this helper for Markdown snippets, compact display text, and lightweight text normalization.

    Args:
        inputs (dict): Input mapping rendered in the example block.
        output (Any): Expected or actual output rendered in the example block.
        bullet (str): Bullet prefix used for rendered Markdown lists. Defaults vary by helper. Defaults to ``'-'``.
        inputs_schema (Optional[dict]): Optional per-input rendering schema. Defaults to ``None``.
        output_schema (Optional[Union[str, dict]]): Rendering schema used for output and expected values. Defaults to ``"tag"``.
        hints (Optional[Union[str, List[str]]]): Optional hint text or hint list rendered before output. Defaults to ``None``.
        expected (Any): Optional expected value. ``...`` omits the expected section.
        notes (Optional[Union[str, List[str]]]): Optional note text or note list rendered after hints. Defaults to ``None``.
        tag (Optional[str]): XML-style wrapper tag name. Defaults to ``None``.
        tr (Optional[Callable[[str], str]]): Optional text transform callable, such as a translation helper. Defaults to ``None``.

    Returns:
        str: The result of ``example_block``.
    """
    tr = tr or str
    text = f"{tr('Inputs')}:\n{bullet_dict(inputs, bullet=bullet, schema=inputs_schema)}"
    for label, value in (("Hints", hints), ("Notes", notes)):
        items = [value] if isinstance(value, str) else (value or [])
        if items:
            text += f"\n{tr(label)}:\n{bullet_list([tr(item) for item in items], bullet=bullet)}"
    if output is not ...:
        text += f"\n{tr('Output')}:\n{format_value(output, schema=output_schema, tag='output')}"
    if expected is not ...:
        text += f"\n{tr('Expected')}:\n{format_value(expected, schema=output_schema, tag='expected')}"
    return tag_block(tag, text, inline=False) if tag else text


def _prompt_translate(tr: Optional[Callable[[Any], Any]]) -> Callable[[Any], str]:
    """Return a prompt text transform that always produces strings."""
    return (lambda value: str(value)) if tr is None else (lambda value: str(tr(value)))


def _prompt_items(value: Optional[Union[str, List[str]]]) -> List[str]:
    """Normalize one prompt section value into non-empty strings."""
    if value is None:
        return []
    return [str(item) for item in (value if isinstance(value, list) else [value]) if item is not None and str(item).strip()]


def _prompt_section_items(title: str, value: _SectionInput) -> dict[str, List[str]]:
    """Normalize one named or custom-titled prompt section."""
    if isinstance(value, dict):
        return {str(key): _prompt_items(items) for key, items in value.items()}
    return {title: _prompt_items(value)}


def _prompt_entry_dict(entry: Any) -> dict[str, Any]:
    """Normalize one prompt example or instance into a mapping."""
    if hasattr(entry, "to_dict") and callable(entry.to_dict):
        return dict(entry.to_dict())
    return dict(entry) if isinstance(entry, dict) else {"inputs": {"value": entry}}


def _prompt_entry_text(
    entry: Any,
    *,
    idx: int,
    instance: bool,
    inputs_schema: Any,
    output_schema: Any,
    tr: Callable[[Any], str],
) -> str:
    """Render one prompt example or new instance."""
    if hasattr(entry, "to_str") and callable(entry.to_str):
        item = entry.clone(output="TODO") if instance and hasattr(entry, "clone") else entry
        return item.to_str(
            inputs_schema=inputs_schema,
            output_schema="todo" if instance else output_schema,
            tag="new_instance" if instance else f"example_{idx}",
            tr=tr,
        )
    data = _prompt_entry_dict(entry)
    inputs = data.get(
        "inputs",
        data.get(
            "input",
            {key: value for key, value in data.items() if key not in {"output", "expected", "hints", "notes"}},
        ),
    )
    if not isinstance(inputs, dict):
        inputs = {"value": inputs}
    output = "TODO" if instance else data.get("output", data.get("expected", ...))
    return example_block(
        inputs=inputs,
        output=output,
        expected=data.get("expected", ...),
        hints=data.get("hints"),
        notes=data.get("notes"),
        inputs_schema=inputs_schema,
        output_schema="todo" if instance else output_schema,
        tag="new_instance" if instance else f"example_{idx}",
        tr=tr,
    )


def fast_prompt_section(
    system: Optional[str] = None,
    descriptions: _SectionInput = None,
    examples: Optional[List[Any]] = None,
    inputs_schema: Optional[Union[str, dict]] = None,
    output_schema: Optional[Union[str, dict]] = None,
    instructions: _SectionInput = None,
    instance: Optional[Any] = None,
    separate_system: bool = False,
    tr: Optional[Callable[[Any], Any]] = None,
) -> _Messages:
    """Build OpenAI-style messages from common prompt sections.

    This is a plain string-composition helper. It does not construct, persist,
    or otherwise depend on :class:`heavenbase.Prompt`.

    Args:
        system: System or context text. It stays in the user message by
            default.
        descriptions: Task description text, or a mapping from custom section
            titles to text items.
        examples: Example dictionaries or objects with a compatible
            ``to_str(...)`` method.
        inputs_schema: Rendering schema passed to examples and the new
            instance.
        output_schema: Rendering schema passed to examples.
        instructions: Instruction text, or a mapping from custom section
            titles to text items.
        instance: New input instance rendered with a TODO output.
        separate_system: Whether to emit system text as its own message.
        tr: Optional transform applied to system text, titles, and list text.

    Returns:
        OpenAI-style messages containing one user message and, when requested,
        one preceding system message.
    """
    translate = _prompt_translate(tr)
    messages: _Messages = []
    if separate_system and system is not None:
        messages.append({"role": "system", "content": translate(system)})
        system = None

    sections: List[dict[str, str]] = []
    for title, items in _prompt_section_items("Task Descriptions", descriptions).items():
        if items:
            sections.append({"title": translate(title), "content": bullet_list([translate(item) for item in items])})
    if examples:
        schema = output_schema or {"mode": "tag", "kwargs": {"schema": "repr", "inline": True}}
        rendered = [
            _prompt_entry_text(item, idx=idx, instance=False, inputs_schema=inputs_schema, output_schema=schema, tr=translate)
            for idx, item in enumerate(examples, start=1)
        ]
        sections.append({"title": translate("Examples"), "content": "\n\n".join(rendered)})
    for title, items in _prompt_section_items("Instructions", instructions).items():
        if items:
            sections.append({"title": translate(title), "content": bullet_list([translate(item) for item in items])})
    if instance is not None:
        sections.append(
            {
                "title": translate("New Instance"),
                "content": _prompt_entry_text(
                    instance,
                    idx=1,
                    instance=True,
                    inputs_schema=inputs_schema,
                    output_schema=None,
                    tr=translate,
                ),
            }
        )

    messages.append(
        {
            "role": "user",
            "content": md_section(
                content=translate(system) if system is not None else None,
                sections=sections,
            ).strip(),
        }
    )
    return messages


def is_delimiter(char: str) -> bool:
    """
    Return whether a character is a word-boundary delimiter.

    Use this helper for Markdown snippets, compact display text, and lightweight text normalization.

    Args:
        char (str): Single character to classify.

    Returns:
        bool: The whether a character is a word-boundary delimiter.
    """
    return char in string.whitespace or char in string.punctuation


def is_spacy_available() -> bool:
    """
    Return whether spaCy and the small English model are available.

    Use this helper for Markdown snippets, compact display text, and lightweight text normalization.
    Args:
        None.


    Returns:
        bool: The whether spaCy and the small English model are available.
    """
    global _SPACY_AVAILABLE, _SPACY_NLP
    if _SPACY_AVAILABLE is not None:
        return _SPACY_AVAILABLE
    try:
        import spacy

        _SPACY_NLP = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        _SPACY_AVAILABLE = True
    except (ImportError, OSError):
        _SPACY_AVAILABLE = False
    return _SPACY_AVAILABLE


def normalize_text(text: str) -> str:
    """
    Normalize English text through tokenization and lemmatization.

    Use this helper for Markdown snippets, compact display text, and lightweight text normalization.

    Args:
        text (str): Text value to format, normalize, copy, or compare.

    Returns:
        str: The result of ``normalize_text``.
    """
    if _SPACY_NLP is None and not is_spacy_available():
        raise RuntimeError("spacy or en_core_web_sm is unavailable; install it or use a custom normalizer.")
    return " ".join(
        token.lemma_.strip()
        for token in _SPACY_NLP(text.lower().replace("_", "-").replace("-", " "))
        if not token.is_stop and not token.is_punct and token.lemma_.strip()
    )


def generate_ngrams(tokens: list, n: int) -> Set[str]:
    """
    Return all n-grams up to size n.

    Use this helper for Markdown snippets, compact display text, and lightweight text normalization.

    Args:
        tokens (list): Token list used to generate n-grams.
        n (int): Optional output count or shape. ``None`` returns one scalar value when supported.

    Returns:
        Set[str]: The all n-grams up to size n.
    """
    return {" ".join(tokens[idx : idx + size]) for size in range(1, n + 1) for idx in range(len(tokens) - size + 1)}


def asymmetric_jaccard_score(query: str, doc: str, ngram: int = 6) -> float:
    """
    Return containment score of query n-grams in doc n-grams.

    Use this helper for Markdown snippets, compact display text, and lightweight text normalization.

    Args:
        query (str): Query text whose normalized n-grams are compared against the document.
        doc (str): Document text whose normalized n-grams are compared against the query.
        ngram (int): Maximum n-gram size used by the containment score. Defaults to ``6``.

    Returns:
        float: The containment score of query n-grams in doc n-grams.
    """
    query_set = generate_ngrams(normalize_text(query).split(), n=ngram)
    doc_set = generate_ngrams(normalize_text(doc).split(), n=ngram)
    return 1.0 if not query_set else len(query_set.intersection(doc_set)) / len(query_set)


def snake(text: object) -> str:
    """
    Convert common name shapes to snake_case.

    Use this helper for Markdown snippets, compact display text, and lightweight text normalization.

    Args:
        text (object): Text value to format, normalize, copy, or compare.

    Returns:
        str: The result of ``snake``.
    """
    value = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", str(text).replace("-", " "))
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value)
    return re.sub(r"[^0-9a-zA-Z]+", "_", value).strip("_").lower()


def _format_param(value: Any, key: Optional[str], required: bool) -> str:
    p_type = value.get("type", "any") if isinstance(value, dict) else "any"
    item = f"`{key}`: {p_type}{' *(required)*' if required else ''}"
    if isinstance(value, dict):
        if value.get("enum"):
            item += f" ({', '.join(str(item) for item in value['enum'])})"
        if value.get("default") is not None:
            item += f" = {value['default']}"
        if value.get("description"):
            item += f" - {value['description']}"
    return item
