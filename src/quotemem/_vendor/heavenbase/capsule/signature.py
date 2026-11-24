"""Callable signature and JSON schema helpers."""

from __future__ import annotations

__all__ = ["CallableSignature", "infer_capabilities", "inspect_callable_signature"]

import inspect
from typing import get_args, get_origin, get_type_hints

from heavenbase.utils import Any, Dict, List, dataclass, field


@dataclass
class CallableSignature:
    """Normalized callable signature metadata."""

    parameters: List[Dict[str, Any]] = field(default_factory=list)
    return_annotation: str = ""
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_schema: Dict[str, Any] = field(default_factory=dict)
    accepts_varargs: bool = False
    accepts_varkw: bool = False
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-safe signature dictionary.

        Args:
            None.

        Returns:
            Dict[str, Any]: Normalized callable signature metadata.
        """
        return {
            "parameters": list(self.parameters),
            "return_annotation": self.return_annotation,
            "accepts_varargs": self.accepts_varargs,
            "accepts_varkw": self.accepts_varkw,
            "warnings": list(self.warnings),
        }


def _annotation_name(annotation: Any) -> str:
    if annotation is inspect._empty:
        return ""
    return getattr(annotation, "__name__", str(annotation).replace("typing.", ""))


def _schema_for(annotation: Any, default: Any = inspect._empty) -> Dict[str, Any]:
    origin = get_origin(annotation)
    args = get_args(annotation)
    if annotation is inspect._empty:
        schema: Dict[str, Any] = {}
    elif annotation is str:
        schema = {"type": "string"}
    elif annotation is int:
        schema = {"type": "integer"}
    elif annotation is float:
        schema = {"type": "number"}
    elif annotation is bool:
        schema = {"type": "boolean"}
    elif annotation in (dict, Dict) or origin is dict:
        schema = {"type": "object"}
    elif annotation in (list, List) or origin is list:
        item_schema = _schema_for(args[0]) if args else {}
        schema = {"type": "array", "items": item_schema}
    elif origin is None and annotation is type(None):
        schema = {"type": "null"}
    elif origin is not None and str(origin).endswith("Union"):
        schemas = [_schema_for(item) for item in args]
        schema = {"anyOf": schemas}
    else:
        schema = {}
    if default is not inspect._empty:
        schema = dict(schema)
        schema["default"] = default
    return schema


def inspect_callable_signature(fn: Any) -> CallableSignature:
    """Inspect a callable and return JSON-schema-adjacent metadata.

    Args:
        fn (Any): Callable object to inspect.

    Returns:
        CallableSignature: Normalized parameter and return metadata.
    """
    signature = inspect.signature(fn)
    try:
        hints = get_type_hints(fn)
    except (NameError, TypeError):
        hints = {}
    if hints:
        parameters = [parameter.replace(annotation=hints.get(name, parameter.annotation)) for name, parameter in signature.parameters.items()]
        signature = signature.replace(
            parameters=parameters,
            return_annotation=hints.get("return", signature.return_annotation),
        )
    properties: Dict[str, Any] = {}
    required: List[str] = []
    parameters: List[Dict[str, Any]] = []
    warnings: List[str] = []
    accepts_varargs = False
    accepts_varkw = False
    for name, parameter in signature.parameters.items():
        if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
            accepts_varargs = True
            warnings.append(f"{name}: varargs are not representable as MCP object properties")
            continue
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            accepts_varkw = True
            warnings.append(f"{name}: kwargs are not representable as MCP object properties")
            continue
        schema = _schema_for(parameter.annotation, default=parameter.default)
        item = {
            "name": name,
            "kind": parameter.kind.name,
            "required": parameter.default is inspect._empty,
            "default": None if parameter.default is inspect._empty else parameter.default,
            "annotation": _annotation_name(parameter.annotation),
            "schema": schema,
        }
        parameters.append(item)
        properties[name] = schema
        if parameter.default is inspect._empty:
            required.append(name)
    input_schema = {"type": "object", "properties": properties, "required": required}
    output_schema = _schema_for(signature.return_annotation)
    return CallableSignature(
        parameters=parameters,
        return_annotation=_annotation_name(signature.return_annotation),
        input_schema=input_schema,
        output_schema=output_schema,
        accepts_varargs=accepts_varargs,
        accepts_varkw=accepts_varkw,
        warnings=warnings,
    )


def infer_capabilities(signature: CallableSignature, restore_layers: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Infer export capabilities for a callable.

    Args:
        signature (CallableSignature): Normalized callable signature metadata.
        restore_layers (List[Dict[str, Any]]): Restore-layer payloads captured
            for the callable.

    Returns:
        Dict[str, Any]: Capability matrix keyed by export target.
    """
    has_callable_layer = bool(restore_layers)
    schema_ok = not signature.accepts_varargs and not signature.accepts_varkw
    reasons = list(signature.warnings)
    return {
        "python_callable": {"ok": has_callable_layer, "reasons": [] if has_callable_layer else ["no restore layer"]},
        "source_transfer": {"ok": any(layer.get("type") == "source" for layer in restore_layers), "reasons": []},
        "json_schema": {"ok": True, "reasons": reasons},
        "mcp_tool": {"ok": schema_ok and has_callable_layer, "reasons": reasons},
        "anthropic_programmatic": {"ok": schema_ok and has_callable_layer, "reasons": reasons},
    }
