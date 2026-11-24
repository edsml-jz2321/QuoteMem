"""Docstring parsing and synthesis helpers for Capsules."""

from __future__ import annotations

__all__ = ["parse_docstring", "synthesize_docstring"]

from heavenbase.utils import Any, Dict, List


def parse_docstring(docstring: str | None) -> Dict[str, Any]:
    """Parse a small Google/Numpy-style docstring into summary, params, and returns.

    Args:
        docstring (str | None): Raw docstring text, or `None` for an empty
            result.

    Returns:
        Dict[str, Any]: Mapping with `summary`, `params`, `returns`, and `raw`
            keys.
    """
    text = (docstring or "").strip()
    lines = text.splitlines()
    summary: List[str] = []
    params: Dict[str, str] = {}
    returns = ""
    mode = "summary"
    current = ""
    for raw in lines:
        line = raw.strip()
        key = line.rstrip(":").lower()
        if key in {"args", "arguments", "parameters", "params"}:
            mode = "params"
            current = ""
            continue
        if key in {"return", "returns"}:
            mode = "returns"
            current = ""
            continue
        if mode == "summary":
            if line:
                summary.append(line)
            continue
        if mode == "params" and line:
            if ":" in line:
                field, desc = [item.strip() for item in line.split(":", 1)]
                current = field.split()[0]
                params[current] = desc
            elif current:
                params[current] = f"{params[current]} {line}".strip()
            continue
        if mode == "returns" and line:
            returns = f"{returns} {line}".strip()
    return {"summary": " ".join(summary).strip(), "params": params, "returns": returns, "raw": text}


def synthesize_docstring(name: str, description: str = "", parameters: List[Dict[str, Any]] | None = None, return_annotation: str = "") -> str:
    """Synthesize a compact docstring from Capsule signature metadata.

    Args:
        name (str): Callable name used in the generated summary.
        description (str): Optional leading summary text.
        parameters (List[Dict[str, Any]] | None): Parameter metadata used to
            build an ``Args`` section.
        return_annotation (str): Optional return annotation text for a
            ``Returns`` section.

    Returns:
        str: Compact Google-style docstring text.
    """
    lines = [description or f"Run {name}."]
    params = parameters or []
    if params:
        lines.extend(["", "Args:"])
        for param in params:
            desc = param.get("description") or param.get("annotation") or "value"
            lines.append(f"    {param['name']}: {desc}")
    if return_annotation:
        lines.extend(["", "Returns:", f"    {return_annotation}"])
    return "\n".join(lines)
