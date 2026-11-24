"""Capture Python callables into Capsule manifests."""

from __future__ import annotations

__all__ = ["CaptureOptions", "capture_function"]

import ast
import base64
import inspect
import sys
import textwrap
from time import time

from heavenbase.utils import Any, Callable, Dict, List, dataclass, field, sha256hash
from heavenbase.utils.config import ConfigManager

from ..modules import ModuleResolver
from .authority import CapsuleAuthority
from .docstring import parse_docstring
from .layer import CapsuleLayerBuildContext
from .layers import build_capsule_layers
from .manifest import CapsuleManifest, make_capsule_id
from .signature import infer_capabilities, inspect_callable_signature

_SECRET_NAMES = ("key", "token", "secret", "password", "credential", "authorization", "api_key")


@dataclass
class CaptureOptions:
    """Options for capturing a callable."""

    name: str | None = None
    namespace: str | None = None
    version: str = "1"
    description: str | None = None
    tags: List[str] = field(default_factory=list)
    layers: List[str] | None = None
    layer_options: Dict[str, Any] = field(default_factory=dict)
    include_cloudpickle: bool = False
    prefer_import: bool = True
    allow_binary_payload: bool = False


def _default_namespace(config: ConfigManager) -> str:
    return str(config.get("heavenbase.capsule.namespace", default="default") or "default")


def _verify_source_hash(config: ConfigManager) -> bool:
    return bool(config.get("heavenbase.capsule.registry.trust.verify_source_hash", default=True))


def _imports_from_source(source: str) -> List[Dict[str, Any]]:
    imports: List[Dict[str, Any]] = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return imports
    stdlib = getattr(sys, "stdlib_module_names", set())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                imports.append({"module": alias.name, "kind": "stdlib" if root in stdlib else "package", "required": True})
        elif isinstance(node, ast.ImportFrom) and node.module:
            root = node.module.split(".", 1)[0]
            imports.append({"module": node.module, "kind": "stdlib" if root in stdlib else "package", "required": True})
    return imports


def _builtins() -> set[str]:
    return set(__builtins__) if isinstance(__builtins__, dict) else set(dir(__builtins__))


def _safe_name(name: str) -> bool:
    lowered = name.lower()
    return not any(item in lowered for item in _SECRET_NAMES)


def _json_safe(value: Any) -> bool:
    if value is None or isinstance(value, (str, int, float, bool)):
        return True
    if isinstance(value, (list, tuple)):
        return all(_json_safe(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and _json_safe(item) for key, item in value.items())
    return False


def _bind_target(target: ast.AST, names: set[str]) -> None:
    if isinstance(target, ast.Name):
        names.add(target.id)
    elif isinstance(target, (ast.Tuple, ast.List)):
        for item in target.elts:
            _bind_target(item, names)


class _NameScan(ast.NodeVisitor):
    def __init__(self) -> None:
        self.loads: set[str] = set()
        self.bound: set[str] = set()

    def visit_arg(self, node: ast.arg) -> None:
        self.bound.add(node.arg)

    def visit_Name(self, node: ast.Name) -> None:
        (self.bound if isinstance(node.ctx, ast.Store) else self.loads).add(node.id)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.bound.add(alias.asname or alias.name.split(".", 1)[0])

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            self.bound.add(alias.asname or alias.name)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.bound.add(node.name)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.bound.add(node.name)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.bound.add(node.name)
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.name:
            self.bound.add(node.name)
        self.generic_visit(node)

    def visit_comprehension(self, node: ast.comprehension) -> None:
        _bind_target(node.target, self.bound)
        self.generic_visit(node)


def _global_names(source: str, fn: Callable[..., Any]) -> set[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    scan = _NameScan()
    scan.visit(tree)
    freevars = set(getattr(getattr(fn, "__code__", None), "co_freevars", ()) or ())
    return (scan.loads - scan.bound - _builtins()) | freevars


def _ref_for(name: str, value: Any) -> tuple[Dict[str, Any] | None, Dict[str, Any] | None]:
    if not _safe_name(name):
        return None, {"name": name, "reason": "secret-like-global"}
    if inspect.ismodule(value):
        return {"kind": "module", "module": value.__name__}, None
    if _json_safe(value):
        return {"kind": "value", "value": value}, None
    module = getattr(value, "__module__", "")
    qualname = getattr(value, "__qualname__", "")
    if module and module != "__main__" and qualname and "<locals>" not in qualname:
        return {"kind": "attr", "module": module, "qualname": qualname}, None
    return None, {"name": name, "reason": "unsupported-global", "type": type(value).__name__}


def _closure_items(fn: Callable[..., Any]) -> Dict[str, Any]:
    names = getattr(getattr(fn, "__code__", None), "co_freevars", ()) or ()
    cells = getattr(fn, "__closure__", None) or ()
    out: Dict[str, Any] = {}
    for name, cell in zip(names, cells):
        try:
            out[name] = cell.cell_contents
        except ValueError:
            pass
    return out


def _global_refs(fn: Callable[..., Any], source: str) -> tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    values = dict(getattr(fn, "__globals__", {}) or {}) | _closure_items(fn)
    refs: Dict[str, Dict[str, Any]] = {}
    unresolved: List[Dict[str, Any]] = []
    for name in sorted(_global_names(source, fn)):
        if name not in values:
            continue
        ref, error = _ref_for(name, values[name])
        if ref is not None:
            refs[name] = ref
        elif error is not None:
            unresolved.append(error)
    return refs, unresolved


def _ref_imports(refs: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    imports: List[Dict[str, Any]] = []
    stdlib = getattr(sys, "stdlib_module_names", set())
    for ref in refs.values():
        module = ref.get("module")
        if not module:
            continue
        root = str(module).split(".", 1)[0]
        imports.append({"module": module, "kind": "stdlib" if root in stdlib else "package", "required": True})
    return imports


def _source_layer(fn: Callable[..., Any]) -> tuple[Dict[str, Any] | None, Dict[str, Any]]:
    try:
        source = textwrap.dedent(inspect.getsource(fn))
        tree = ast.parse(source)
        if len(tree.body) == 1 and isinstance(tree.body[0], (ast.FunctionDef, ast.AsyncFunctionDef)) and tree.body[0].decorator_list:
            tree.body[0].decorator_list = []
            source = ast.unparse(tree) + "\n"
        ast.parse(source)
    except Exception as exc:
        return None, {"ok": False, "reason": "source-unavailable", "error": str(exc)}
    filename = inspect.getsourcefile(fn) or inspect.getfile(fn) or ""
    global_refs, unresolved = _global_refs(fn, source)
    data = {
        "ok": True,
        "source": source,
        "source_hash": sha256hash(source),
        "filename": filename,
        "line_start": getattr(getattr(fn, "__code__", None), "co_firstlineno", None),
        "module": getattr(fn, "__module__", ""),
        "qualname": getattr(fn, "__qualname__", getattr(fn, "__name__", "callable")),
        "imports": _imports_from_source(source),
        "global_refs": global_refs,
        "unresolved": unresolved,
    }
    layer = {
        "type": "source",
        "priority": 10,
        "source": source,
        "filename": filename,
        "module": data["module"],
        "qualname": data["qualname"],
        "source_hash": data["source_hash"],
        "global_refs": global_refs,
    }
    return layer, data


def _import_layer(fn: Callable[..., Any]) -> Dict[str, Any] | None:
    module = getattr(fn, "__module__", "")
    qualname = getattr(fn, "__qualname__", getattr(fn, "__name__", ""))
    if not module or module == "__main__" or "<locals>" in qualname:
        return None
    return {"type": "import", "priority": 20, "module": module, "qualname": qualname}


def _cloudpickle_layer(fn: Callable[..., Any]) -> tuple[Dict[str, Any] | None, str | None]:
    try:
        import cloudpickle

        payload = base64.b64encode(cloudpickle.dumps(fn)).decode("ascii")
        return {"type": "cloudpickle", "priority": 90, "payload_b64": payload, "payload_hash": sha256hash(payload), "python": sys.version.split()[0]}, None
    except Exception as exc:
        return None, str(exc)


def _source_layer_builder(ctx: CapsuleLayerBuildContext) -> Dict[str, Any] | None:
    layer, source = _source_layer(ctx.fn)
    ctx.data["source"] = source
    return layer


def _import_layer_builder(ctx: CapsuleLayerBuildContext) -> Dict[str, Any] | None:
    return _import_layer(ctx.fn)


def _cloudpickle_layer_builder(ctx: CapsuleLayerBuildContext) -> Dict[str, Any] | None:
    layer, error = _cloudpickle_layer(ctx.fn)
    if error:
        ctx.data.setdefault("warnings", []).append(error)
    return layer


def capture_function(
    fn: Callable[..., Any],
    options: CaptureOptions | None = None,
    *,
    resolver: ModuleResolver | None = None,
    config: ConfigManager | None = None,
) -> CapsuleManifest:
    """Capture a Python callable into a CapsuleManifest.

    Args:
        fn: Callable to capture.
        options: Optional capture policy and metadata.
        resolver: Context-owned module resolver used for Capsule layers. When
            omitted, the default Context is selected.
        config: Config authority paired with `resolver`. When omitted, the
            standalone process config is selected.

    Returns:
        CapsuleManifest: Durable callable manifest.
    """
    authority = CapsuleAuthority.resolve(resolver=resolver, config=config)
    chosen = authority.resolver
    options = options or CaptureOptions()
    name = options.name or getattr(fn, "__name__", "capsule")
    namespace = options.namespace or _default_namespace(authority.config)
    version = options.version or "1"
    restore_layers, layer_data = build_capsule_layers(fn, options, resolver=chosen, config=authority.config)
    source = dict(layer_data.get("source") or {})
    layer_names = tuple(layer_data.get("layer_names") or ())
    raw_doc = inspect.getdoc(fn) or ""
    doc = parse_docstring(raw_doc)
    signature = inspect_callable_signature(fn)
    for param in signature.parameters:
        if param["name"] in doc["params"]:
            param["description"] = doc["params"][param["name"]]
    capabilities = infer_capabilities(signature, restore_layers)
    now = time()
    dependencies = {
        "python": {"version": sys.version.split()[0]},
        "imports": (source.get("imports", []) if source else []) + _ref_imports(source.get("global_refs", {}) if source else {}),
        "unresolved": source.get("unresolved", []) if source else [],
        "warnings": list(layer_data.get("warnings") or []),
    }
    fingerprints = {
        "source_hash": source.get("source_hash") if source else None,
        "signature_hash": sha256hash(signature.to_dict()),
        "restore_hash": sha256hash([{key: value for key, value in layer.items() if key != "payload_b64"} for layer in restore_layers]),
    }
    return CapsuleManifest(
        id=make_capsule_id(namespace, name, version),
        name=name,
        namespace=namespace,
        version=version,
        description=options.description or doc["summary"] or raw_doc,
        tags=list(options.tags),
        identity={"module": getattr(fn, "__module__", ""), "qualname": getattr(fn, "__qualname__", name), "name": name},
        signature=signature.to_dict(),
        input_schema=signature.input_schema,
        output_schema=signature.output_schema,
        restore_layers=restore_layers,
        dependencies=dependencies,
        capabilities=capabilities,
        source={**source, "doc": doc},
        provenance={"captured_at": now},
        trust={
            "local_only": True,
            "allow_binary_payload": bool(options.allow_binary_payload or options.include_cloudpickle or "cloudpickle" in layer_names),
            "verify_source_hash": _verify_source_hash(authority.config),
        },
        fingerprints=fingerprints,
        created_at=now,
        updated_at=now,
    )
