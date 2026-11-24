"""LLM image-generation CLI command handler."""

from __future__ import annotations

from heavenbase.utils import Any

from .runtime import _llm, _verbose


def _imagen(
    context: Any,
    prompt: str,
    output: str | None = None,
    preset: str | None = None,
    model: str | None = None,
    provider: str | None = None,
    gateway: str | None = None,
    cache: bool = True,
    as_json: bool = False,
    verbose: bool = False,
) -> None:
    llm = _llm(context=context, preset=preset or "imagen", model=model, provider=provider, gateway=gateway, cache=cache)
    if verbose:
        _verbose(context, llm, cache=cache)
    result = llm.imagen(prompt, include=["images", "usage"], reduce=False)
    images = result.get("images")
    image_list = images if isinstance(images, list) else [images]
    if output and image_list and image_list[0] is not None:
        image_list[0].save(output)
        context.out.success(f"saved image to {output}")
    elif as_json:
        context.out.json({"images": [image.to_data_url() for image in image_list if image is not None], "usage": result.get("usage")})
    else:
        context.out.yaml({"images": [image.to_dict() for image in image_list if image is not None], "usage": result.get("usage")})
