"""LLM config and catalog CLI command handlers."""

from __future__ import annotations

from heavenbase.llm.config import LLMEngine
from heavenbase.utils import Any
from heavenbase.utils.config.data import to_mutable_data
from heavenbase.utils.redact import redact_data

from .runtime import _llm


def _verbose_cmd(context: Any, preset: str | None = None, model: str | None = None, provider: str | None = None, gateway: str | None = None) -> None:
    spec = _llm(context=context, preset=preset, model=model, provider=provider, gateway=gateway).spec
    context.out.yaml(spec.engine.apply(spec, profile="print"))


def _status(context: Any, preset: str | None = None, model: str | None = None, provider: str | None = None, gateway: str | None = None) -> None:
    spec = _llm(context=context, preset=preset, model=model, provider=provider, gateway=gateway).spec
    context.out.yaml({"preset": spec.preset, "provider": spec.provider, "model": spec.model, "model_id": spec.model_id, "gateway": spec.gateway})


def _catalog_projection(value: Any) -> Any:
    """Return a serializable, secret-redacted copy of one LLM catalog value."""
    return redact_data(to_mutable_data(value))


def _list(context: Any, catalog: str = "presets") -> None:
    root = LLMEngine(context=context.runtime_context()).root()
    catalogs = {"endpoints", "gateways", "models", "presets", "providers"}
    if catalog in {"all", "catalogs"}:
        context.out.yaml({name: sorted((_catalog_projection(root.get(name)) or {}).keys()) for name in sorted(catalogs)})
        return
    if catalog not in catalogs:
        raise ValueError(f"Unsupported LLM catalog {catalog!r}; expected one of {sorted(catalogs)}")
    context.out.yaml(_catalog_projection(root.get(catalog, {})))
