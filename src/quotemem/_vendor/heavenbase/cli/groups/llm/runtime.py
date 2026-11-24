"""Shared LLM CLI runtime helpers."""

from __future__ import annotations

from heavenbase.llm import LLM
from heavenbase.utils import CM_HVNB, Any, clipboard, dumps_json, load_txt

_DEFAULT_MAX_STEPS = 20


def _runtime_context(context: Any) -> Any:
    """Return the machine Context owned by one CLI invocation."""
    runtime = getattr(context, "runtime_context", None)
    if callable(runtime):
        return runtime()
    from heavenbase.context import context_or_default

    return context_or_default(getattr(getattr(context, "cm", CM_HVNB), "_context", None))


def _llm(
    context: Any,
    preset: str | None = None,
    model: str | None = None,
    provider: str | None = None,
    gateway: str | None = None,
    cache: bool = True,
) -> LLM:
    """Construct an LLM from the CLI invocation's exact machine authority."""
    return LLM(context=_runtime_context(context), preset=preset, model=model, provider=provider, gateway=gateway, cache=cache)


def _prompt(prompt: str | None, input_files: list[str] | None = None) -> str:
    parts = [load_txt(path, strict=True).strip() for path in input_files or []]
    if prompt:
        parts.append(prompt)
    return "\n\n".join(part for part in parts if part)


def _verbose(context: Any, llm: LLM, cache: bool = True) -> None:
    context.out.muted_yaml({"llm": llm.spec.engine.apply(llm.spec, profile="print"), "cache": cache}, err=True)


def _json_text(data: Any, compact: bool = False) -> str:
    return dumps_json(data, compact=compact)


def _emit_json(context: Any, data: Any, compact: bool = False) -> str:
    text = _json_text(data, compact=compact)
    context.out.echo(text)
    return text


def _copy_output(context: Any, text: str) -> None:
    if clipboard(text):
        context.out.muted("copied to clipboard", err=True)
    else:
        context.out.warning("clipboard unavailable; output not copied")


def _max_steps(value: int | None = None) -> int:
    return max(1, int(_DEFAULT_MAX_STEPS if value is None else value))


def _max_tool_turns(max_steps: int | None = None) -> int:
    return max(0, _max_steps(max_steps) - 1)


def _active_workspace(context: Any) -> Any:
    from heavenbase.workspace import HeavenBase

    return HeavenBase.load(context=_runtime_context(context))


def _llm_state(llm: Any) -> dict[str, Any]:
    spec = getattr(llm, "spec", None)
    if spec is not None:
        return spec.engine.apply(spec, profile="print")
    return {}
