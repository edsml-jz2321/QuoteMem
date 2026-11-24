"""LLM embedding CLI command handler."""

from __future__ import annotations

from heavenbase.utils import Any

from .runtime import _copy_output, _emit_json, _json_text, _llm, _prompt, _verbose

_EMBED_PREVIEW_HEAD = 4
_EMBED_PREVIEW_TAIL = 2
_EMBED_PREVIEW_DIGITS = 6


def _preview_number(value: Any) -> str:
    return f"{value:.{_EMBED_PREVIEW_DIGITS}f}" if type(value) in (float, int) else str(value)


def _preview_embedding(embedding: Any) -> str:
    if not isinstance(embedding, list):
        return str(embedding)
    items = (
        embedding
        if len(embedding) <= _EMBED_PREVIEW_HEAD + _EMBED_PREVIEW_TAIL
        else [*embedding[:_EMBED_PREVIEW_HEAD], "...", *embedding[-_EMBED_PREVIEW_TAIL:]]
    )
    return "[" + ", ".join("..." if item == "..." else _preview_number(item) for item in items) + "]"


def _preview_embeddings(embeddings: Any) -> Any:
    if isinstance(embeddings, list) and embeddings and all(isinstance(item, list) for item in embeddings):
        return [_preview_embedding(item) for item in embeddings]
    return _preview_embedding(embeddings)


def _embed(
    context: Any,
    text: str | None = None,
    input_files: list[str] | None = None,
    preset: str | None = None,
    model: str | None = None,
    provider: str | None = None,
    gateway: str | None = None,
    cache: bool = True,
    as_json: bool = False,
    preview: bool = False,
    copy: bool = False,
    verbose: bool = False,
) -> None:
    llm = _llm(context=context, preset=preset or "embed", model=model, provider=provider, gateway=gateway, cache=cache)
    if verbose:
        _verbose(context, llm, cache=cache)
    result = llm.embed(_prompt(text, input_files), include=["embeddings", "dim", "usage"], reduce=False)
    output_data = dict(result)
    if preview:
        output_data["embeddings"] = _preview_embeddings(result.get("embeddings", []))
    if as_json:
        output = _emit_json(context, output_data)
        if copy:
            _copy_output(context, output)
        return
    output = str(output_data["embeddings"]) if preview else _json_text(output_data.get("embeddings", []), compact=True)
    context.out.echo(output)
    if copy:
        _copy_output(context, output)
